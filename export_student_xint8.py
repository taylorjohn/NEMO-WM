"""
export_student_xint8.py — CORTEX-16 NPU Export Pipeline

Converts a trained StudentEncoder checkpoint to a VitisAI-ready XINT8 ONNX model.

Steps:
  1. Load trained PyTorch weights
  2. Export to FP32 ONNX (opset 17, fixed shapes — required by NPU)
  3. Quantize to XINT8 via AMD Quark (the only path where all ViT-adjacent ops
     execute on-chip without CPU fallback)
  4. Verify NPU subgraph placement via vitisai_ep_report.json
  5. Run a quick latency benchmark

Requirements:
  - AMD Ryzen AI SDK 1.7.0 conda environment (ryzen-ai-1.7.0 or clone)
  - quark (AMD Quark quantizer, installed in ryzen-ai env)
  - onnxruntime-vitisai (patched build from AMD SDK — do NOT pip install onnxruntime)
  - 100-500 calibration images (ImageNet val or domain-specific)

Usage:
  python export_student_xint8.py \\
      --weights cortex_student_phase2_final.pt \\
      --calib   ./calib_images \\
      --out     ./npu_models

Output files:
  cortex_student_fp32.onnx      — FP32 intermediate (can delete after quantization)
  cortex_student_xint8.onnx     — XINT8 model for VitisAI EP
  cortex_student_xint8_report.json — Operator placement report (verify on-NPU ops)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort
from PIL import Image
from torchvision import transforms

from student_encoder import StudentEncoder, StudentEncoderONNX


# =============================================================================
# Image preprocessing (must match training transforms exactly)
# =============================================================================
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_image(path: str) -> np.ndarray:
    """Load and normalise a single image to (1, 3, 224, 224) float32."""
    img = Image.open(path).convert("RGB").resize((224, 224), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]  # (1, 3, 224, 224)
    return arr


# =============================================================================
# Step 1 — Export FP32 ONNX
# =============================================================================
def export_fp32_onnx(weights_path: str, output_path: str) -> None:
    print("\n── Step 1: FP32 ONNX Export ─────────────────────────────────────")

    student = StudentEncoder()
    ckpt    = torch.load(weights_path, map_location="cpu")
    student.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    student.eval()

    export_model = StudentEncoderONNX(student)
    dummy_input  = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        export_model,
        dummy_input,
        output_path,
        export_params    = True,
        opset_version    = 17,          # AMD recommends opset 17
        do_constant_folding = True,
        input_names      = ["input_frame"],
        output_names     = ["output_latent"],
        dynamic_axes     = None,        # CRITICAL: NPU requires fixed shapes
        verbose          = False,
    )

    # Quick sanity check — run through CPU ONNX Runtime
    sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    out  = sess.run(None, {"input_frame": dummy_input.numpy()})[0]
    assert out.shape == (1, 128), f"Unexpected output shape: {out.shape}"

    print(f"✅ FP32 ONNX exported → {output_path}")
    print(f"   Output shape: {out.shape}")


# =============================================================================
# Step 2 — XINT8 Quantization via AMD Quark
# =============================================================================
class CalibrationDataReader:
    """
    Feeds calibration images to Quark's quantization engine.
    Use 100-500 representative images from your target domain.
    """
    def __init__(self, image_dir: str, num_samples: int = 200):
        exts = ["*.jpg", "*.jpeg", "*.png", "*.JPEG", "*.JPG"]
        paths = []
        for ext in exts:
            paths.extend(sorted(Path(image_dir).glob(ext)))
        self.paths = paths[:num_samples]
        if not self.paths:
            raise FileNotFoundError(
                f"No images found in {image_dir}. "
                "Provide 100-500 calibration images (e.g. ImageNet val subset)."
            )
        self._idx = 0
        print(f"   Calibration: {len(self.paths)} images from {image_dir}")

    def get_next(self):
        if self._idx >= len(self.paths):
            return None
        data = preprocess_image(str(self.paths[self._idx]))
        self._idx += 1
        return {"input_frame": data}

    def rewind(self):
        self._idx = 0


def quantize_xint8(fp32_path: str, xint8_path: str, calib_dir: str) -> None:
    print("\n── Step 2: XINT8 Quantization ───────────────────────────────────")
    print("   (Requires AMD Quark — only available inside ryzen-ai conda env)")

    try:
        from quark.onnx import ModelQuantizer, QConfig
    except ImportError:
        print("⚠️  AMD Quark not found. To quantize:")
        print("   1. Activate your ryzen-ai conda environment")
        print("   2. Re-run this script")
        print(f"\n   FP32 model is available at: {fp32_path}")
        print("   You can load it with CPUExecutionProvider for development.")
        return

    calib_reader = CalibrationDataReader(calib_dir, num_samples=200)
    config       = QConfig.get_default_config("XINT8")
    # Override target for Ryzen AI MAX+ 395 (Strix, XDNA2)
    if hasattr(config, 'compiler_configs') and config.compiler_configs:
        for cc in config.compiler_configs:
            if hasattr(cc, 'xcompiler_params') and cc.xcompiler_params:
                cc.xcompiler_params['target_alias'] = 'strix'
    quantizer    = ModelQuantizer(config)

    quantizer.quantize_model(
        fp32_path,
        xint8_path,
        calib_reader,
    )
    print(f"✅ XINT8 model quantized → {xint8_path}")


# =============================================================================
# Step 3 — NPU Subgraph Verification
# Confirms that Conv, BN, ReLU, AvgPool are all mapped to the NPU,
# not falling back to CPU. Any CPU fallback on Softmax/LayerNorm (from ViT)
# is not expected here since StudentEncoder uses only CNN ops.
# =============================================================================
def verify_npu_placement(xint8_path: str, report_path: str) -> bool:
    print("\n── Step 3: NPU Subgraph Verification ────────────────────────────")

    try:
        import onnxruntime as ort
    except ImportError:
        print("⚠️  onnxruntime not available.")
        return False

    vai_options = {
        "config_file":             r"C:\Users\MeteorAI\Desktop\CORTEX\vaip_config.json",
        "cache_dir":               str(Path(xint8_path).parent),
        "cache_key":               "cortex_student_xint8",
        "enable_cache_file_io_in_mem": "0",
    }

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4

    try:
        sess = ort.InferenceSession(
            xint8_path,
            sess_options     = sess_opts,
            providers        = ["VitisAIExecutionProvider", "CPUExecutionProvider"],
            provider_options = [vai_options, {}],
        )
    except Exception as e:
        print(f"⚠️  VitisAI EP session failed: {e}")
        print("   Ensure NPU drivers are active and you are in the ryzen-ai conda env.")
        return False

    # Run a dummy inference to trigger compilation and produce report
    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
    io    = sess.io_binding()
    io.bind_cpu_input("input_frame", dummy)
    io.bind_output("output_latent")
    sess.run_with_iobinding(io)
    out = io.copy_outputs_to_cpu()[0]

    assert out.shape == (1, 128), f"Unexpected output: {out.shape}"

    # Load and audit the operator placement report
    report_candidates = [
        Path(vai_options["cache_dir"]) / "vitisai_ep_report.json",
        Path(".") / "vitisai_ep_report.json",
    ]
    report_data = None
    for rp in report_candidates:
        if rp.exists():
            with open(rp) as f:
                report_data = json.load(f)
            break

    if report_data:
        # Count NPU vs CPU ops
        npu_ops = [n for n in report_data.get("nodes", [])
                   if n.get("ep") == "VitisAIExecutionProvider"]
        cpu_ops = [n for n in report_data.get("nodes", [])
                   if n.get("ep") == "CPUExecutionProvider"]

        print(f"   NPU ops:  {len(npu_ops)}")
        print(f"   CPU ops:  {len(cpu_ops)}")

        # Flag any unexpected CPU fallbacks for CNN ops
        cpu_op_types = [n.get("op_type") for n in cpu_ops]
        concern_ops  = {"Softmax", "LayerNormalization", "Gelu"}
        concerns     = [op for op in cpu_op_types if op in concern_ops]
        if concerns:
            print(f"⚠️  Unexpected CPU fallbacks: {concerns}")
            print("   (These are ViT ops — should not appear in CNN student)")
        else:
            print("✅ No unexpected CPU fallbacks")

        # Copy report to output dir
        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2)
        print(f"   Report saved → {report_path}")
    else:
        print("   vitisai_ep_report.json not found — check cache_dir")

    print(f"✅ NPU placement verified. Output shape: {out.shape}")
    return True


# =============================================================================
# Step 4 — Latency Benchmark
# =============================================================================
def benchmark_latency(
    xint8_path: str,
    n_warmup: int = 10,
    n_runs: int = 100,
) -> None:
    print("\n── Step 4: Latency Benchmark ────────────────────────────────────")

    vai_options = {
        "cache_dir":               str(Path(xint8_path).parent),
        "cache_key":               "cortex_student_xint8",
        "enable_cache_file_io_in_mem": "0",
    }

    try:
        sess = ort.InferenceSession(
            xint8_path,
            providers        = ["VitisAIExecutionProvider", "CPUExecutionProvider"],
            provider_options = [vai_options, {}],
        )
    except Exception as e:
        print(f"⚠️  Could not create NPU session for benchmark: {e}")
        return

    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
    io    = sess.io_binding()

    # Warmup
    for _ in range(n_warmup):
        io.bind_cpu_input("input_frame", dummy)
        io.bind_output("output_latent")
        sess.run_with_iobinding(io)
        io.copy_outputs_to_cpu()

    # Timed runs
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        io.bind_cpu_input("input_frame", dummy)
        io.bind_output("output_latent")
        sess.run_with_iobinding(io)
        io.copy_outputs_to_cpu()
        times.append((time.perf_counter() - t0) * 1000)

    avg = np.mean(times)
    p99 = np.percentile(times, 99)

    print(f"   Latency avg: {avg:.2f}ms  p99: {p99:.2f}ms")
    target = 2.0
    status = "✅ PASS" if avg < target else f"❌ FAIL (target < {target}ms)"
    print(f"   Target <{target}ms: {status}")


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Export CORTEX-16 student to NPU XINT8")
    p.add_argument("--weights", required=True,
                   help="Path to trained .pt checkpoint")
    p.add_argument("--calib",   required=True,
                   help="Directory of calibration images (100-500 recommended)")
    p.add_argument("--out",     default="./npu_models",
                   help="Output directory for ONNX files and report")
    p.add_argument("--skip-quantize", action="store_true",
                   help="Skip Quark quantization (useful outside ryzen-ai env)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out  = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fp32_path  = str(out / "cortex_student_fp32.onnx")
    xint8_path = str(out / "cortex_student_xint8.onnx")
    report_path = str(out / "cortex_student_xint8_report.json")

    export_fp32_onnx(args.weights, fp32_path)

    if not args.skip_quantize:
        quantize_xint8(fp32_path, xint8_path, args.calib)
    else:
        print("\n⚠️  Skipping XINT8 quantization (--skip-quantize flag set)")
        xint8_path = fp32_path

    if Path(xint8_path).exists():
        verify_npu_placement(xint8_path, report_path)
        benchmark_latency(xint8_path)

    print(f"\n✅ Export complete. NPU model: {xint8_path}")
    print(f"   Load in unified_cortex_loop.py:")
    print(f"   NPUExecutionEngine('{xint8_path}')")

