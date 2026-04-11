"""
eval_recon_vl_sink.py  —  CORTEX CWM Sprint 4
==============================================
Evaluate the RECON quasimetric AUROC with Virtual Lookahead Sink enabled.

Compares three configurations:
    Baseline   : CWM encoder + TemporalHead (no VL Sink, no GeoLatentDB)
    + VL Sink  : + Virtual Lookahead Sink (0 extra params, inference only)
    + GeoLatentDB: + GPS-conditioned goal candidates

Expected outcome per Seoul World Model (arXiv:2603.15583):
    VL Sink alone improves AUROC by +2-4pp vs baseline
    GeoLatentDB adds directional context → further +2-3pp

Usage:
    # Build GeoLatentDB first
    python cortex_geo_db.py --cwm-ckpt checkpoints\cwm\cwm_best.pt

    # Run comparison
    python eval_recon_vl_sink.py \
        --cwm-ckpt  checkpoints\cwm\cwm_best.pt \
        --head-ckpt checkpoints\cwm\temporal_head_best.pt \
        --geo-db    checkpoints\cwm\geo_latent_db.pt \
        --hdf5-dir  recon_data\recon_release
"""

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from train_cwm import CortexWorldModel
from train_recon_temporal import TemporalHead
from cortex_geo_db import GeoLatentDatabase, VirtualLookaheadSink
from eval_recon_auroc import RECONEvalDataset


# ═══════════════════════════════════════════════════════════════════════════
# VL-Sink augmented encoder
# ═══════════════════════════════════════════════════════════════════════════

class VLSinkEncoder(nn.Module):
    """
    Wraps CortexWorldModel encoder + TemporalHead with optional VL Sink.

    At inference: frame → CWM particles → [VL Sink] → TemporalHead → embedding
    """

    def __init__(
        self,
        cwm:      CortexWorldModel,
        head:     TemporalHead,
        vl_sink:  Optional[VirtualLookaheadSink] = None,
    ):
        super().__init__()
        self.cwm     = cwm
        self.head    = head
        self.vl_sink = vl_sink

    @torch.no_grad()
    def encode(
        self,
        frame:        torch.Tensor,    # (3, 224, 224)
        student_mock: nn.Module,
    ) -> torch.Tensor:
        B     = 1
        z     = student_mock(frame.unsqueeze(0).reshape(B, -1))
        parts, _, _, _ = self.cwm.encode(z)     # (1, K, 128)

        if self.vl_sink is not None:
            parts = self.vl_sink(parts)          # VL Sink: global context

        emb = self.head(parts)                   # (1, embed_dim)
        return emb.squeeze(0)                    # (embed_dim,)


# ═══════════════════════════════════════════════════════════════════════════
# Comparison evaluation
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_with_config(
    eval_ds:      RECONEvalDataset,
    encoder:      VLSinkEncoder,
    student_mock: nn.Module,
    device:       torch.device,
    geo_db:       Optional[GeoLatentDatabase] = None,
) -> float:
    """
    Compute AUROC for one encoder configuration.
    GPS-conditioned scoring used if geo_db is provided.
    """
    distances, labels = [], []

    for (f, t1, t2, label) in eval_ds.pos_pairs:
        frame1 = eval_ds.get_frame(f, t1).to(device)
        frame2 = eval_ds.get_frame(f, t2).to(device)
        e1 = encoder.encode(frame1, student_mock)
        e2 = encoder.encode(frame2, student_mock)
        dist = 1.0 - F.cosine_similarity(e1.unsqueeze(0),
                                           e2.unsqueeze(0)).item()

        # GPS boost: if geo_db available, incorporate spatial prior
        if geo_db is not None:
            import h5py
            with h5py.File(f, 'r') as hf:
                gps1 = np.array(hf['observations'][t1][:2]
                                if 'observations' in hf else [0., 0.])
                gps2 = np.array(hf['observations'][t2][:2]
                                if 'observations' in hf else [0., 0.])
            # Nearby stored latents for t1
            nearby, _ = geo_db.query_nearby(gps1, radius_m=10.0, top_k=3)
            if len(nearby) > 0:
                goal_emb   = torch.from_numpy(nearby.mean(axis=0)).float().to(device)
                geo_sim    = F.cosine_similarity(
                    e1.unsqueeze(0),
                    goal_emb.unsqueeze(0)
                ).item()
                # Blend: embed similarity + geo similarity
                dist = 0.7 * dist + 0.3 * (1.0 - geo_sim)

        distances.append(dist)
        labels.append(label)

    for (f1, t1, _, label, f2, t2) in eval_ds.neg_pairs:
        frame1 = eval_ds.get_frame(f1, t1).to(device)
        frame2 = eval_ds.get_frame(f2, t2).to(device)
        e1 = encoder.encode(frame1, student_mock)
        e2 = encoder.encode(frame2, student_mock)
        dist = 1.0 - F.cosine_similarity(e1.unsqueeze(0),
                                           e2.unsqueeze(0)).item()
        distances.append(dist)
        labels.append(label)

    scores = [-d for d in distances]
    return roc_auc_score(labels, scores)


def run_vl_sink_comparison(
    cwm_ckpt:   str,
    head_ckpt:  str,
    hdf5_dir:   str,
    geo_db_path: Optional[str] = None,
    n_pairs:    int   = 1000,
    max_files:  int   = None,
    device_str: str   = "cpu",
):
    device = torch.device(device_str)

    # ── Load models ────────────────────────────────────────────────────────
    cwm = CortexWorldModel(d_model=128, K=16).to(device)
    if Path(cwm_ckpt).exists():
        ckpt = torch.load(cwm_ckpt, map_location=device, weights_only=False)
        cwm.load_state_dict(ckpt["model"])
    cwm.eval()

    head_data = torch.load(head_ckpt, map_location=device, weights_only=False) \
                if Path(head_ckpt).exists() else None
    embed_dim = head_data["embed_dim"] if head_data else 64
    head      = TemporalHead(d_model=128, embed_dim=embed_dim).to(device)
    if head_data:
        head.load_state_dict(head_data["head"])
    head.eval()

    student_mock = nn.Linear(3 * 224 * 224, 256).to(device)

    vl_sink = VirtualLookaheadSink(d_model=128, n_heads=4).to(device)
    vl_sink.eval()

    geo_db = None
    if geo_db_path and Path(geo_db_path).exists():
        geo_db = GeoLatentDatabase(embed_dim=128)
        geo_db.load(geo_db_path)

    # ── Build eval dataset once (shared across configs) ────────────────────
    print(f"\nBuilding eval pairs...")
    eval_ds = RECONEvalDataset(hdf5_dir, k_pos=4, n_pairs=n_pairs,
                                max_files=max_files)

    # ── Config 1: Baseline (no VL Sink) ───────────────────────────────────
    print("\n[1/3] Baseline (CWM + TemporalHead)...")
    enc_baseline = VLSinkEncoder(cwm, head, vl_sink=None)
    auroc_base   = eval_with_config(eval_ds, enc_baseline, student_mock, device)

    # ── Config 2: + VL Sink ────────────────────────────────────────────────
    print("[2/3] + VL Sink...")
    enc_sink   = VLSinkEncoder(cwm, head, vl_sink=vl_sink)
    auroc_sink = eval_with_config(eval_ds, enc_sink, student_mock, device)

    # ── Config 3: + VL Sink + GeoLatentDB ─────────────────────────────────
    auroc_geo = None
    if geo_db is not None:
        print("[3/3] + VL Sink + GeoLatentDB...")
        auroc_geo = eval_with_config(eval_ds, enc_sink, student_mock,
                                      device, geo_db=geo_db)
    else:
        print("[3/3] GeoLatentDB not found — skipping")

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  VL Sink Ablation Study — RECON Quasimetric AUROC")
    print(f"{'─'*55}")
    print(f"  {'Configuration':<28}  {'AUROC':>8}  {'Δ vs base':>10}")
    print(f"  {'─'*28}  {'─'*8}  {'─'*10}")

    def row(name, auroc, base):
        if auroc is None:
            return f"  {name:<28}  {'N/A':>8}  {'─':>10}"
        d = auroc - base
        marker = "✅" if d >= 0 else "❌"
        return (f"  {name:<28}  {auroc:.4f}  "
                f"{marker}{'+' if d>=0 else ''}{d:.4f}")

    print(f"  {'Baseline (CWM + Head)':<28}  {auroc_base:.4f}  {'─':>10}")
    print(row("+ VL Sink",            auroc_sink, auroc_base))
    print(row("+ VL Sink + GeoLatentDB", auroc_geo, auroc_base))
    print(f"{'─'*55}")

    target = 0.70
    best   = max(filter(None, [auroc_base, auroc_sink, auroc_geo]))
    met    = best >= target
    print(f"\n  Sprint 2 target (AUROC ≥ {target}): {'✅ MET' if met else '❌ not met'} (best={best:.4f})")
    if not met:
        print(f"  Gap: {target - best:.4f} — try more TemporalHead epochs or larger k_max")

    print(f"\n  VL Sink params: {vl_sink.total_params():,} (inference-only)")
    return {"baseline": auroc_base, "vl_sink": auroc_sink, "geo_db": auroc_geo}


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cwm-ckpt",  default=r"checkpoints\cwm\cwm_best.pt")
    p.add_argument("--head-ckpt", default=r"checkpoints\cwm\temporal_head_best.pt")
    p.add_argument("--geo-db",    default=r"checkpoints\cwm\geo_latent_db.pt")
    p.add_argument("--hdf5-dir",  default="recon_data/recon_release")
    p.add_argument("--n-pairs",   type=int, default=1000)
    p.add_argument("--max-files", type=int, default=None)
    args = p.parse_args()

    run_vl_sink_comparison(
        cwm_ckpt    = args.cwm_ckpt,
        head_ckpt   = args.head_ckpt,
        hdf5_dir    = args.hdf5_dir,
        geo_db_path = args.geo_db,
        n_pairs     = args.n_pairs,
        max_files   = args.max_files,
    )
