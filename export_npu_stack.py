"""
export_npu_stack.py — CORTEX-PE v16.15
NPU Full Stack Export — TransitionPredictor + CardiacStudentEncoder

Exports both remaining models to XINT8 ONNX for AMD Ryzen AI NPU deployment.

Architectures reverse-engineered from checkpoint weight shapes (strict=True verified):

TransitionPredictor:
    Input:  (1, 8)  — [lat, lon, bearing, gps_vel×3, cmd_lin, cmd_ang]
    Net:    Linear(8→64) + LayerNorm(64) + GELU + Linear(64→64) + LayerNorm(64) + GELU + Linear(64→2)
    Output: (1, 2)  — Δ(lat, lon) prediction
    Has LayerNorm → XINT8 required

CardiacStudentEncoder:
    Input:  (1, 1, 2000) — 1-second cardiac audio at 2kHz, mono
    Encoder: Conv1d(1→32,k=10,s=5) + GELU + Conv1d(32→64,k=3,s=2) + GELU +
             Conv1d(64→128,k=3,s=2) + GELU + Conv1d(128→256,k=3,s=2) + GELU +
             AdaptiveAvgPool1d(1)
    Proj:   Linear(256→768) + LayerNorm(768)
    Output: (1, 768) — WavLM-distilled audio embedding
    Has LayerNorm → XINT8 required

Usage:
    python export_npu_stack.py                    # export both
    python export_npu_stack.py --model transition # transition only
    python export_npu_stack.py --model cardiac    # cardiac only
    python export_npu_stack.py --dry-run          # FP32 only, skip quantization
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn


# ── Model definitions (strict=True verified against checkpoint shapes) ─────────

class TransitionPredictor(nn.Module):
    """
    GPS dead-reckoning predictor.
    Input:  (B, 8)  — [lat, lon, bearing, gps_vel×3, cmd_lin, cmd_ang]
    Output: (B, 2)  — Δ(lat, lon)

    Architecture from net.{0..6} weight shapes:
        net.0: Linear(8→64)    + net.1: LayerNorm(64)  → GELU
        net.3: Linear(64→64)   + net.4: LayerNorm(64)  → GELU
        net.6: Linear(64→2)
    """
    def __init__(self, input_dim: int = 8, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CardiacStudentEncoder(nn.Module):
    """
    WavLM-distilled cardiac audio encoder.
    Input:  (B, 1, 2000) — 1-second mono cardiac audio at 2kHz
    Output: (B, 768)     — audio embedding

    Architecture from encoder.{0,2,4,6,8} + proj.{0,1} weight shapes:
        Conv1d(1→32,  k=10, s=5) + GELU
        Conv1d(32→64, k=3,  s=2) + GELU
        Conv1d(64→128,k=3,  s=2) + GELU
        Conv1d(128→256,k=3, s=2) + GELU
        AdaptiveAvgPool1d(1)      → squeeze → (B, 256)
        Linear(256→768) + LayerNorm(768)
    """
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1,   32,  10, stride=5, padding=2), nn.GELU(),
            nn.Conv1d(32,  64,  3,  stride=2, padding=1), nn.GELU(),
            nn.Conv1d(64,  128, 3,  stride=2, padding=1), nn.GELU(),
            nn.Conv1d(128, 256, 3,  stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.encoder(x).squeeze(-1))


# ── Utilities ─────────────────────────────────────────────────────────────────

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def embed_metadata(onnx_path: Path, meta: dict) -> None:
    model = onnx.load(str(onnx_path))
    for k, v in meta.items():
        e = model.metadata_props.add()
        e.key, e.value = k, str(v)
    onnx.save(model, str(onnx_path))


def verify_cosine(pt_model: nn.Module,
                  onnx_path: Path,
                  dummy: torch.Tensor,
                  input_name: str = "input") -> float:
    import onnxruntime as ort
    pt_model.eval()
    with torch.no_grad():
        pt_out = pt_model(dummy).numpy().flatten()
    sess    = ort.InferenceSession(str(onnx_path),
                                   providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {input_name: dummy.numpy()})[0].flatten()
    cos = float(np.dot(pt_out, onnx_out) /
                (np.linalg.norm(pt_out) * np.linalg.norm(onnx_out) + 1e-8))
    return cos


# ── Export pipeline ────────────────────────────────────────────────────────────

def export_model(name: str,
                 model: nn.Module,
                 ckpt_path: Path,
                 state_key: str,
                 dummy_input: torch.Tensor,
                 input_name: str,
                 output_name: str,
                 out_dir: Path,
                 dry_run: bool) -> dict:

    print(f"\n{'─'*60}")
    print(f"  Exporting: {name}")
    print(f"{'─'*60}")

    # ── Load checkpoint ──────────────────────────────────────────────
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = ckpt[state_key]
    missing, unexpected = model.load_state_dict(state, strict=True)
    assert not missing,    f"Missing keys: {missing}"
    assert not unexpected, f"Unexpected keys: {unexpected}"
    model.eval()
    ckpt_hash = sha256(ckpt_path)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params    : {n_params:,}")
    print(f"  SHA256    : {ckpt_hash[:16]}...{ckpt_hash[-8:]}")

    # ── Verify forward pass ───────────────────────────────────────────
    with torch.no_grad():
        out = model(dummy_input)
    print(f"  Output    : {tuple(out.shape)}")

    # ── Export FP32 ONNX ─────────────────────────────────────────────
    fp32_path = out_dir / f"{name}_fp32.onnx"
    torch.onnx.export(
        model, (dummy_input,), str(fp32_path),
        opset_version    = 17,
        do_constant_folding = True,
        input_names      = [input_name],
        output_names     = [output_name],
        dynamic_axes     = None,
    )
    onnx.checker.check_model(str(fp32_path))
    print(f"  FP32 ONNX : {fp32_path.name}  ({fp32_path.stat().st_size/1024:.1f}KB) ✅")

    # ── FP32 cosine verify ────────────────────────────────────────────
    cos_fp32 = verify_cosine(model, fp32_path, dummy_input, input_name)
    print(f"  Cosine FP32 vs PT: {cos_fp32:.6f}  "
          f"{'✅' if cos_fp32 > 0.9999 else '⚠️'}")

    if dry_run:
        print(f"  Dry run — skipping XINT8")
        return {"name": name, "cos_fp32": cos_fp32, "cos_xint8": None}

    # ── XINT8 quantization ────────────────────────────────────────────
    try:
        from quark.onnx import ModelQuantizer, QConfig
    except ImportError:
        print("  ❌ AMD Quark not found — run in ryzen-ai-1.7.0 conda env")
        return {"name": name, "cos_fp32": cos_fp32, "cos_xint8": None}

    xint8_path = out_dir / f"{name}_xint8.onnx"

    class CalibReader:
        def __init__(self, n=200):
            self.data = [
                {input_name: np.random.randn(*dummy_input.shape).astype(np.float32)}
                for _ in range(n)
            ]
            self.idx = 0
        def get_next(self):
            if self.idx >= len(self.data): return None
            r = self.data[self.idx]; self.idx += 1; return r

    print(f"  Quantizing to XINT8 (200 calibration samples)...")
    t0 = time.perf_counter()
    ModelQuantizer(QConfig.get_default_config("XINT8")).quantize_model(
        str(fp32_path), str(xint8_path), CalibReader()
    )
    print(f"  Quantized in {time.perf_counter()-t0:.1f}s  "
          f"({xint8_path.stat().st_size/1024:.1f}KB)")

    # ── XINT8 cosine verify ───────────────────────────────────────────
    cos_xint8 = verify_cosine(model, xint8_path, dummy_input, input_name)
    status = "✅" if cos_xint8 > 0.95 else "⚠️  BELOW THRESHOLD"
    print(f"  Cosine XINT8 vs PT: {cos_xint8:.4f}  {status}")

    # ── Embed metadata ────────────────────────────────────────────────
    meta = {
        "cortex_version":    "v16.15",
        "model_name":        name,
        "checkpoint_sha256": ckpt_hash,
        "checkpoint_path":   str(ckpt_path),
        "export_timestamp":  datetime.datetime.now().isoformat(),
        "quantization":      "XINT8",
        "input_shape":       str(tuple(dummy_input.shape)),
        "output_shape":      str(tuple(out.shape)),
        "n_params":          str(n_params),
        "target_hardware":   "AMD XDNA2 Ryzen AI MAX+ 395",
        "cos_fp32_vs_pt":    str(round(cos_fp32, 6)),
        "cos_xint8_vs_pt":   str(round(cos_xint8, 4)),
    }
    embed_metadata(xint8_path, meta)

    return {"name": name, "cos_fp32": cos_fp32, "cos_xint8": cos_xint8,
            "xint8_path": str(xint8_path), "sha256": ckpt_hash}


# ── Main ───────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    print(f"\n{'='*60}")
    print(f"  CORTEX-PE NPU Full Stack Export — v16.15")
    print(f"{'='*60}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # ── TransitionPredictor ───────────────────────────────────────────
    if args.model in ("transition", "both"):
        ckpt = Path("./checkpoints/recon_transition/transition_best.pt")
        if not ckpt.exists():
            print(f"  ⚠️  {ckpt} not found — skipping TransitionPredictor")
        else:
            # Read hidden dim from checkpoint
            c = torch.load(ckpt, map_location="cpu", weights_only=True)
            hidden = c.get("hidden", 64)
            model  = TransitionPredictor(input_dim=8, hidden=hidden)
            dummy  = torch.zeros(1, 8)
            r = export_model(
                name        = "transition_predictor",
                model       = model,
                ckpt_path   = ckpt,
                state_key   = "model_state_dict",
                dummy_input = dummy,
                input_name  = "state",
                output_name = "delta_pos",
                out_dir     = out_dir,
                dry_run     = args.dry_run,
            )
            results.append(r)

    # ── CardiacStudentEncoder ─────────────────────────────────────────
    if args.model in ("cardiac", "both"):
        ckpt = Path("./checkpoints/cardiac/student_best.pt")
        if not ckpt.exists():
            print(f"  ⚠️  {ckpt} not found — skipping CardiacStudentEncoder")
        else:
            c      = torch.load(ckpt, map_location="cpu", weights_only=True)
            hidden = c.get("hidden_dim", 768)
            model  = CardiacStudentEncoder(hidden_dim=hidden)
            dummy  = torch.zeros(1, 1, 2000)
            r = export_model(
                name        = "cardiac_student",
                model       = model,
                ckpt_path   = ckpt,
                state_key   = "model_state_dict",
                dummy_input = dummy,
                input_name  = "audio",
                output_name = "embedding",
                out_dir     = out_dir,
                dry_run     = args.dry_run,
            )
            results.append(r)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    manifest = {}
    for r in results:
        cos_x = r.get("cos_xint8")
        status = "✅" if cos_x and cos_x > 0.95 else ("⏭️ dry-run" if not cos_x else "⚠️")
        print(f"  {r['name']:30s} FP32={r['cos_fp32']:.6f}  "
              f"XINT8={cos_x:.4f if cos_x is not None else chr(78)+chr(47)+chr(65)}  {status}")
        manifest[r["name"]] = r

    manifest_path = out_dir / "npu_stack_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Manifest: {manifest_path}")
    print(f"\n  Deploy with VitisAI EP:")
    print(f"  providers=['VitisAIExecutionProvider']")
    print(f"  provider_options=[{{'cache_dir':'./npu_cache','target':'X2'}}]")
    print(f"{'='*60}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export TransitionPredictor + CardiacStudentEncoder to NPU"
    )
    parser.add_argument("--model",   default="both",
                        choices=["both", "transition", "cardiac"])
    parser.add_argument("--out",     default="./npu_models")
    parser.add_argument("--dry-run", action="store_true",
                        help="FP32 only, skip XINT8 quantization")
    args = parser.parse_args()
    run(args)
