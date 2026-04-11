"""
test_dualhead_alignment.py — Sprint 6b: Dual-Head CLIP Alignment Test
======================================================================
Tests CLIP semantic alignment using the SemanticHead output rather than
the physical PhysicalHead (proj layer). The backbone is frozen and shared.

Usage:
    python test_dualhead_alignment.py \
        --cwm-ckpt   checkpoints\cwm\cwm_best.pt \
        --dualhead-ckpt checkpoints\dinov2_student\student_dualhead_best.pt \
        --hdf5-dir   recon_data\recon_release \
        --n-files    50 --n-per-file 8 --top-k 30

Author: John Taylor
Date:   2026-04-04
"""

import argparse
import glob
import io
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import clip
from PIL import Image

from train_mvtec import StudentEncoder

DEVICE = torch.device("cpu")

# ── Reproduce SemanticHead architecture ──────────────────────────────────────

class SemanticHead(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class DualHeadEncoder(nn.Module):
    """
    Loads frozen backbone + trained SemanticHead + CLIPBridge.
    Outputs semantic embeddings (not physical particle embeddings).
    """
    def __init__(self, student_ckpt: str, dualhead_ckpt: str):
        super().__init__()
        # Frozen backbone
        enc = StudentEncoder()
        if Path(student_ckpt).exists():
            sd = torch.load(student_ckpt, map_location="cpu", weights_only=False)
            enc.load_state_dict(sd.get("model", sd), strict=False)
        self.features = enc.features
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        for p in self.features.parameters():
            p.requires_grad_(False)

        # Trained SemanticHead + CLIPBridge
        self.semantic_head = SemanticHead()
        self.clip_bridge   = nn.Linear(512, 128, bias=False)

        if Path(dualhead_ckpt).exists():
            ckpt = torch.load(dualhead_ckpt, map_location="cpu", weights_only=False)
            self.semantic_head.load_state_dict(ckpt["semantic_head"])
            cb_sd = ckpt["clip_bridge"]
            cb_sd = {k.replace("proj.", ""): v for k, v in cb_sd.items()}
            self.clip_bridge.load_state_dict(cb_sd)
            print(f"  DualHead loaded: epoch {ckpt.get('epoch', '?')}, "
                  f"L_clip={ckpt.get('loss', '?'):.4f}")
        else:
            print(f"  WARNING: {dualhead_ckpt} not found — using random heads")

        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns 128-D CLIP-aligned semantic embedding."""
        with torch.no_grad():
            feats  = self.features(x)
            pooled = self.pool(feats).flatten(1)  # (B, 256)
        return self.semantic_head(pooled)          # (B, 128)

    def encode_clip_text(self, text_emb: torch.Tensor) -> torch.Tensor:
        """Projects 512-D CLIP text embedding into semantic space."""
        return F.normalize(self.clip_bridge(text_emb), dim=-1)


# ── Data loading ──────────────────────────────────────────────────────────────

def decode_frame(jpeg_bytes, size=224):
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB").resize((size, size))
    return torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0


def clip_normalise(t):
    mean = torch.tensor([0.48145466, 0.4578275,  0.40821073]).view(3,1,1)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3,1,1)
    return (t - mean) / std


def collect_semantic_embeddings(encoder, hdf5_dir, n_files, n_per_file):
    """Collect SemanticHead embeddings + metadata from RECON files."""
    files = sorted(glob.glob(f"{hdf5_dir}/*.hdf5"))[:n_files]
    records = []

    for fi, fpath in enumerate(files):
        if (fi + 1) % 10 == 1:
            print(f"  File {fi+1}/{len(files)}...")
        try:
            with h5py.File(fpath, "r") as hf:
                imgs = hf["images"]["rgb_left"]
                n    = len(imgs)
                idxs = np.linspace(0, n-1, n_per_file, dtype=int)

                lin_vel = hf["commands"]["linear_velocity"][:]
                ang_vel = hf["commands"]["angular_velocity"][:]
                gps     = hf["gps"]["latlong"][:]

                for i in idxs:
                    frame = decode_frame(bytes(imgs[i])).unsqueeze(0)
                    with torch.no_grad():
                        sem = encoder(frame).squeeze(0).numpy()

                    gps_disp = float(np.linalg.norm(gps[min(i+1, n-1)] - gps[i]))
                    records.append({
                        "sem":     sem,
                        "lin_vel": float(lin_vel[i]),
                        "ang_vel": float(ang_vel[i]),
                        "gps":     gps_disp,
                        "file":    fi,
                    })
        except Exception:
            pass

    return records


# ── Text queries ──────────────────────────────────────────────────────────────

QUERIES = [
    ("robot moving fast forward",       "lin_vel",  "high"),
    ("robot driving at high speed",      "lin_vel",  "high"),
    ("robot moving slowly and carefully","lin_vel",  "low"),
    ("robot barely moving",              "lin_vel",  "low"),
    ("robot stopped or stationary",      "lin_vel",  "zero"),
    ("robot turning sharply",            "ang_vel",  "high"),
    ("robot spinning or rotating",       "ang_vel",  "high"),
    ("robot going straight ahead",       "ang_vel",  "zero"),
    ("robot driving in a straight line", "ang_vel",  "zero"),
    ("the quick brown fox",              None,       "null"),
    ("a recipe for chocolate cake",      None,       "null"),
    ("sonnets by shakespeare",           None,       "null"),
]


def run_alignment_test(
    cwm_ckpt: str,
    dualhead_ckpt: str,
    hdf5_dir: str,
    n_files: int = 50,
    n_per_file: int = 8,
    top_k: int = 30,
):
    t0 = time.time()

    print("\nSprint 6b — Dual-Head CLIP Alignment Test")
    print("=" * 60)

    # Load encoder
    enc = DualHeadEncoder(
        student_ckpt  = r"checkpoints\dinov2_student\student_best.pt",
        dualhead_ckpt = dualhead_ckpt,
    ).to(DEVICE)

    # Collect semantic embeddings
    print(f"\nCollecting semantic embeddings from {n_files} files...")
    records = collect_semantic_embeddings(enc, hdf5_dir, n_files, n_per_file)
    print(f"Collected {len(records)} samples")

    sems    = np.stack([r["sem"]    for r in records])  # (N, 128)
    lin_vel = np.array([r["lin_vel"] for r in records])
    ang_vel = np.array([r["ang_vel"] for r in records])

    # Load CLIP
    print("\nLoading CLIP ViT-B/32...")
    clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
    clip_model.eval()

    print("\n" + "=" * 80)
    print(f"  {'Query':<45} {'Signal':<12} {'Top-K':>8} {'Base':>8} {'Ratio':>7}  Result")
    print("=" * 80)

    semantic_results = []
    null_results = []

    for query, signal, category in QUERIES:
        # Encode text via CLIP
        with torch.no_grad():
            tokens  = clip.tokenize([query]).to(DEVICE)
            txt_emb = clip_model.encode_text(tokens).float()
            txt_emb = F.normalize(txt_emb, dim=-1)

        # Project into semantic space via CLIPBridge
        with torch.no_grad():
            query_sem = enc.encode_clip_text(txt_emb).squeeze(0).numpy()  # (128,)

        # Cosine similarity to all semantic embeddings
        sems_n    = sems / (np.linalg.norm(sems, axis=1, keepdims=True) + 1e-8)
        query_n   = query_sem / (np.linalg.norm(query_sem) + 1e-8)
        sims      = sems_n @ query_n                    # (N,)

        # Top-K mean
        topk_idx  = np.argsort(sims)[-top_k:]
        topk_mean = float(sims[topk_idx].mean())
        baseline  = float(sims.mean())
        ratio     = topk_mean / (abs(baseline) + 1e-8) if baseline != 0 else 1.0

        # Signal-specific check
        if signal == "lin_vel":
            topk_signal = float(np.abs(lin_vel[topk_idx]).mean())
            base_signal = float(np.abs(lin_vel).mean())
            sig_label   = "lin_vel"
        elif signal == "ang_vel":
            topk_signal = float(np.abs(ang_vel[topk_idx]).mean())
            base_signal = float(np.abs(ang_vel).mean())
            sig_label   = "ang_vel"
        else:
            topk_signal = 0.0
            base_signal = 0.0
            sig_label   = "—"

        # Determine result
        if category == "null":
            result = "correct null" if ratio < 1.15 else "LEAKING"
            null_results.append((query, ratio, result))
        else:
            if ratio >= 1.5:
                result = "STRONG"
            elif ratio >= 1.2:
                result = "WEAK"
            else:
                result = "not aligned"
            semantic_results.append((query, signal, ratio, result))

        print(f"  {query:<45} {sig_label:<12} {topk_mean:>8.4f} {baseline:>8.4f} "
              f"{ratio:>7.2f}x  {result}")

    print("=" * 80)

    strong = sum(1 for _,_,_,r in semantic_results if r == "STRONG")
    weak   = sum(1 for _,_,_,r in semantic_results if r == "WEAK")
    ok_null= sum(1 for _,_,r in null_results if r == "correct null")

    print(f"\n  Semantic: {len(semantic_results)} total | "
          f"{strong} STRONG | {weak} WEAK | "
          f"{len(semantic_results)-strong-weak} not aligned")
    print(f"  Nulls: {ok_null}/{len(null_results)} correct")

    if strong >= 2:
        verdict = "STRONG ALIGNMENT — language conditioning feasible"
    elif weak >= 2 and ok_null == len(null_results):
        verdict = "WEAK ALIGNMENT — further training needed"
    else:
        verdict = "NO ALIGNMENT"

    print(f"\n  VERDICT: {verdict}")
    print(f"\n  Time: {time.time()-t0:.1f}s")

    return {"strong": strong, "weak": weak, "null_ok": ok_null, "verdict": verdict}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cwm-ckpt",      default=r"checkpoints\cwm\cwm_best.pt")
    parser.add_argument("--dualhead-ckpt", default=r"checkpoints\dinov2_student\student_dualhead_best.pt")
    parser.add_argument("--hdf5-dir",      default="recon_data/recon_release")
    parser.add_argument("--n-files",       type=int, default=50)
    parser.add_argument("--n-per-file",    type=int, default=8)
    parser.add_argument("--top-k",         type=int, default=30)
    args = parser.parse_args()

    run_alignment_test(
        cwm_ckpt      = args.cwm_ckpt,
        dualhead_ckpt = args.dualhead_ckpt,
        hdf5_dir      = args.hdf5_dir,
        n_files       = args.n_files,
        n_per_file    = args.n_per_file,
        top_k         = args.top_k,
    )
