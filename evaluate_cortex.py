"""
evaluate_cortex.py — CORTEX-16 Evaluation Suite

Ports the validated test protocol from CORTEX-12, adapted for the
StudentEncoder architecture. Run after each training phase to confirm
the model meets the representation quality thresholds established in v12.

Five checks:
  1. Smoke          — forward pass integrity, no NaNs, correct output shape
  2. Stability      — same-concept cosine similarity vs cross-concept separation
  3. Semantic Probe — per-axis geometry (shape > size > depth/velocity separation)
  4. Regression     — compare current checkpoint against previous to detect drift
  5. Latency        — NPU or CPU inference time (asserts < 2ms on NPU)

Thresholds (from CORTEX-12 validated checkpoint at step 5600):
  SAME mean > 0.97   (self-similarity)
  DIFF mean < 0.65   (cross-concept separation — lower = better separated)

Usage:
  python evaluate_cortex.py --weights ./checkpoints/cortex_student_phase1_final.pt
  python evaluate_cortex.py --weights ./checkpoints/cortex_student_phase2_final.pt --baseline ./checkpoints/cortex_student_phase1_final.pt
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw

from student_encoder import StudentEncoder


# =============================================================================
# Synthetic concept generator
# =============================================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

EVAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def make_concept_image(
    shape: str   = "circle",
    color: tuple = (220, 50, 50),
    size: int    = 80,
    position: tuple = (112, 112),
) -> torch.Tensor:
    img  = Image.new("RGB", (224, 224), (128, 128, 128))
    draw = ImageDraw.Draw(img)
    cx, cy = position
    r = size // 2
    if shape == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif shape == "square":
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif shape == "triangle":
        pts = [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)]
        draw.polygon(pts, fill=color)
    return EVAL_TRANSFORM(img)


def concept_batch(n_augments: int = 8, **kwargs) -> torch.Tensor:
    frames = []
    for _ in range(n_augments):
        cx, cy = kwargs.get("position", (112, 112))
        jitter = np.random.randint(-5, 5, 2)
        frames.append(make_concept_image(
            position=(cx + int(jitter[0]), cy + int(jitter[1])),
            **{k: v for k, v in kwargs.items() if k != "position"}
        ))
    return torch.stack(frames)


# =============================================================================
# Test 1 — Smoke
# Student outputs unnormalised vectors — no norm assertion.
# =============================================================================
def test_smoke(model: StudentEncoder) -> bool:
    print("\n── Test 1: Smoke ────────────────────────────────────────────────")
    try:
        dummy = torch.randn(1, 3, 224, 224)
        z, spatial = model(dummy, return_spatial=True)
        assert z.shape       == (1, 128),        f"z shape wrong: {z.shape}"
        assert spatial.shape == (1, 32, 14, 14), f"spatial shape wrong: {spatial.shape}"
        assert not torch.isnan(z).any(),          "NaN in z"
        assert not torch.isnan(spatial).any(),    "NaN in spatial"
        norm = z.norm(dim=-1).item()
        print(f"   z shape:      {tuple(z.shape)}  ✅")
        print(f"   spatial:      {tuple(spatial.shape)}  ✅")
        print(f"   z L2 norm:    {norm:.4f}  (unnormalised — expected)")
        print("✅ Smoke PASSED")
        return True
    except AssertionError as e:
        print(f"❌ Smoke FAILED: {e}")
        return False


# =============================================================================
# Test 2 — Representation Stability
# Normalises vectors before cosine sim — works with unnormalised student output.
# SAME mean > 0.97 | DIFF mean < 0.65
# =============================================================================
def test_stability(
    model: StudentEncoder,
    same_threshold: float = 0.97,
    diff_threshold: float = 0.65,
) -> bool:
    print("\n── Test 2: Representation Stability ─────────────────────────────")

    model.eval()
    concepts = [
        {"shape": "circle",   "color": (220, 50,  50), "size": 80},
        {"shape": "square",   "color": (50,  220, 50), "size": 80},
        {"shape": "triangle", "color": (50,  50, 220), "size": 80},
        {"shape": "circle",   "color": (220, 50,  50), "size": 50},
    ]

    with torch.no_grad():
        embeddings = []
        for c in concepts:
            batch = concept_batch(n_augments=8, **c)
            z     = model(batch)
            z     = F.normalize(z, dim=-1)   # normalise for cosine comparison
            embeddings.append(z)

    same_sims = []
    for z in embeddings:
        for i in range(len(z)):
            for j in range(i + 1, len(z)):
                same_sims.append(F.cosine_similarity(z[i:i+1], z[j:j+1]).item())

    diff_sims = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            for zi in embeddings[i]:
                for zj in embeddings[j]:
                    diff_sims.append(
                        F.cosine_similarity(zi.unsqueeze(0), zj.unsqueeze(0)).item()
                    )

    same_mean = float(np.mean(same_sims))
    same_std  = float(np.std(same_sims))
    diff_mean = float(np.mean(diff_sims))
    diff_std  = float(np.std(diff_sims))

    print(f"   SAME mean = {same_mean:.4f}  std = {same_std:.4f}  (threshold > {same_threshold})")
    print(f"   DIFF mean = {diff_mean:.4f}  std = {diff_std:.4f}  (threshold < {diff_threshold})")

    passed = bool(same_mean >= same_threshold and diff_mean <= diff_threshold)
    print(f"{'✅ Stability PASSED' if passed else '❌ Stability FAILED'}")
    return passed


# =============================================================================
# Test 3 — Semantic Geometry Probe
# =============================================================================
def test_semantic_probe(model: StudentEncoder) -> dict:
    print("\n── Test 3: Semantic Geometry Probe ──────────────────────────────")

    model.eval()
    with torch.no_grad():
        def embed(concept):
            z = model(concept_batch(8, **concept))
            return F.normalize(z, dim=-1).mean(dim=0)

        z_ref        = embed({"shape": "circle", "color": (220, 50, 50),  "size": 80})
        z_same       = embed({"shape": "circle", "color": (220, 50, 50),  "size": 80})
        z_size_diff  = embed({"shape": "circle", "color": (220, 50, 50),  "size": 40})
        z_color_diff = embed({"shape": "circle", "color": (50,  50, 220), "size": 80})
        z_shape_diff = embed({"shape": "square", "color": (220, 50, 50),  "size": 80})

    def sim(a, b):
        an = F.normalize(a.unsqueeze(0), dim=-1)
        bn = F.normalize(b.unsqueeze(0), dim=-1)
        return float(F.cosine_similarity(an, bn).item())

    results = {
        "identity":   sim(z_ref, z_same),
        "size_diff":  sim(z_ref, z_size_diff),
        "color_diff": sim(z_ref, z_color_diff),
        "shape_diff": sim(z_ref, z_shape_diff),
    }

    print(f"   identity  ↔ identity:    {results['identity']:.4f}   (should be ~1.0)")
    print(f"   identity  ↔ size_diff:   {results['size_diff']:.4f}   (should be < identity)")
    print(f"   identity  ↔ color_diff:  {results['color_diff']:.4f}   (should be < size_diff)")
    print(f"   identity  ↔ shape_diff:  {results['shape_diff']:.4f}   (should be lowest)")

    hierarchy_ok = (
        results["identity"]  > results["size_diff"] and
        results["size_diff"] > results["shape_diff"]
    )
    print(f"\n{'✅ Hierarchy PASSED' if hierarchy_ok else '⚠️  Hierarchy unexpected (model may still be training)'}")
    return results


# =============================================================================
# Test 4 — Regression Guard
# =============================================================================
def test_regression(
    current_model: StudentEncoder,
    baseline_path: str,
    drift_threshold: float = 0.02,
) -> bool:
    print("\n── Test 4: Regression Guard ─────────────────────────────────────")

    baseline = StudentEncoder()
    ckpt = torch.load(baseline_path, map_location="cpu")
    baseline.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    baseline.eval()

    concept = {"shape": "circle", "color": (220, 50, 50), "size": 80}

    with torch.no_grad():
        batch  = concept_batch(16, **concept)
        z_curr = F.normalize(current_model(batch), dim=-1)
        z_base = F.normalize(baseline(batch), dim=-1)

    curr_mean = F.normalize(z_curr.mean(dim=0, keepdim=True), dim=-1)
    base_mean = F.normalize(z_base.mean(dim=0, keepdim=True), dim=-1)
    drift = float(1.0 - F.cosine_similarity(curr_mean, base_mean).item())

    print(f"   Embedding drift from baseline: {drift:.4f}  (threshold < {drift_threshold})")
    passed = bool(drift < drift_threshold)
    print(f"{'✅ Regression PASSED' if passed else '⚠️  Drift detected — inspect axis geometry'}")
    return passed


# =============================================================================
# Test 5 — Latency
# =============================================================================
def test_latency(
    onnx_path: str = None,
    pytorch_model: StudentEncoder = None,
    target_ms: float = 2.0,
    n_runs: int = 50,
) -> bool:
    print("\n── Test 5: Latency ──────────────────────────────────────────────")

    dummy_np = np.random.randn(1, 3, 224, 224).astype(np.float32)
    times    = []

    if onnx_path:
        import onnxruntime as ort
        try:
            vai_opts = {"cache_dir": str(Path(onnx_path).parent),
                        "cache_key": "cortex_eval", "log_level": "warning"}
            sess = ort.InferenceSession(
                onnx_path,
                providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
                provider_options=[vai_opts, {}],
            )
            provider_used = "NPU (VitisAI)"
        except Exception:
            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            provider_used = "CPU (fallback)"

        io = sess.io_binding()
        for _ in range(10):
            io.bind_cpu_input("input_frame", dummy_np)
            io.bind_output("output_latent")
            sess.run_with_iobinding(io)
            io.copy_outputs_to_cpu()
        for _ in range(n_runs):
            t0 = time.perf_counter()
            io.bind_cpu_input("input_frame", dummy_np)
            io.bind_output("output_latent")
            sess.run_with_iobinding(io)
            io.copy_outputs_to_cpu()
            times.append((time.perf_counter() - t0) * 1000)

    elif pytorch_model:
        pytorch_model.eval()
        dummy_t = torch.from_numpy(dummy_np)
        provider_used = "PyTorch CPU"
        with torch.no_grad():
            for _ in range(10):
                pytorch_model(dummy_t)
            for _ in range(n_runs):
                t0 = time.perf_counter()
                pytorch_model(dummy_t)
                times.append((time.perf_counter() - t0) * 1000)
    else:
        print("   No model provided for latency test.")
        return False

    avg = float(np.mean(times))
    p99 = float(np.percentile(times, 99))
    print(f"   Provider: {provider_used}")
    print(f"   avg: {avg:.2f}ms  p99: {p99:.2f}ms  (target <{target_ms}ms)")
    passed = bool(avg < target_ms)
    print(f"{'✅ Latency PASSED' if passed else f'⚠️  Latency exceeds target (avg {avg:.2f}ms > {target_ms}ms)'}")
    return passed


# =============================================================================
# JSON serialisation — handles numpy bool_, float32, int64, etc.
# =============================================================================
def make_serialisable(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, dict):
        return {str(kk): make_serialisable(vv) for kk, vv in v.items()}
    if isinstance(v, (list, tuple)):
        return [make_serialisable(x) for x in v]
    if hasattr(v, "item"):      # numpy scalar → python scalar
        return v.item()
    if isinstance(v, (int, float, str)):
        return v
    return str(v)


# =============================================================================
# Full Evaluation Run
# =============================================================================
def run_evaluation(
    weights_path: str,
    baseline_path: str = None,
    onnx_path: str = None,
    output_path: str = "eval_results.json",
) -> dict:
    print("\n" + "="*60)
    print("  CORTEX-16 EVALUATION SUITE")
    print(f"  Weights:  {weights_path}")
    if baseline_path: print(f"  Baseline: {baseline_path}")
    if onnx_path:     print(f"  ONNX:     {onnx_path}")
    print("="*60)

    model = StudentEncoder()
    ckpt  = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    results = {}
    results["smoke"]          = test_smoke(model)
    results["stability"]      = test_stability(model)
    results["semantic_probe"] = test_semantic_probe(model)

    if baseline_path and Path(baseline_path).exists():
        results["regression"] = test_regression(model, baseline_path)
    else:
        print("\n── Test 4: Regression Guard ─────────────────────────────────────")
        print("   Skipped (no baseline provided)")
        results["regression"] = None

    if onnx_path and Path(onnx_path).exists():
        results["latency"] = test_latency(onnx_path=onnx_path)
    else:
        results["latency"] = test_latency(pytorch_model=model, target_ms=50.0)

    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    checks  = ["smoke", "stability", "regression", "latency"]
    skipped = [k for k in checks if results.get(k) is None]
    passed  = [k for k in checks if results.get(k) is True]

    for k in checks:
        v    = results.get(k)
        icon = "✅" if v is True else ("❌" if v is False else "⏭️ ")
        print(f"  {icon}  {k}")

    print(f"\n  {len(passed)}/{len(checks) - len(skipped)} checks passed")

    with open(output_path, "w") as f:
        json.dump(make_serialisable(results), f, indent=2)
    print(f"\n  Results saved → {output_path}")

    return results


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CORTEX-16 Evaluation Suite")
    p.add_argument("--weights",  required=True)
    p.add_argument("--baseline", default=None)
    p.add_argument("--onnx",     default=None)
    p.add_argument("--out",      default="eval_results.json")
    args = p.parse_args()
    run_evaluation(args.weights, args.baseline, args.onnx, args.out)
