"""
eval_rope_temporal.py — CORTEX-PE
A/B Evaluation: RoPETemporalHead vs MLP Baseline on RECON Data
===============================================================

Loads saved checkpoints for both heads, evaluates on held-out
RECON trajectories, and reports:
  - AUROC (close vs far pair discrimination)
  - Separation (far_dist - close_dist)
  - Per-k breakdown: does RoPE improve at short k? long k?
  - Displacement correlation: does quasimetric predict actual metres?

Usage:
  python eval_rope_temporal.py \\
    --data ./recon_data/recon_release \\
    --rope-ckpt ./checkpoints/recon_rope/rope_head_best.pt \\
    --mlp-ckpt  ./checkpoints/recon_contrastive/temporal_head_k7_best.pt
"""

from __future__ import annotations
import argparse, io, json, time
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader

from rope_temporal_head import RoPETemporalHead
from train_rope_temporal import (
    StudentEncoder, MLPTemporalHead, RECONPairDataset,
    evaluate, auroc_np
)


def per_k_auroc(head, pairs: list[dict], device, k_buckets=None):
    """AUROC broken down by temporal offset k."""
    if k_buckets is None:
        k_buckets = {"k1-3": (1,3), "k4-7": (4,7), "k8-15": (8,15)}
    results = {}
    head.eval()

    for name, (k_lo, k_hi) in k_buckets.items():
        sub = [p for p in pairs if k_lo <= p["k"] <= k_hi]
        if not sub:
            continue
        z_t  = torch.from_numpy(np.stack([p["z_t"]  for p in sub])).to(device)
        z_tk = torch.from_numpy(np.stack([p["z_tk"] for p in sub])).to(device)
        labels = np.array([p["eval_label"] for p in sub])

        with torch.no_grad():
            e_t  = head(z_t)
            e_tk = head(z_tk)
            d = 1.0 - F.cosine_similarity(
                F.normalize(e_t, dim=-1), F.normalize(e_tk, dim=-1), dim=-1)

        auc = auroc_np(d.cpu().numpy(), labels)
        results[name] = {"auroc": auc, "n": len(sub)}
    return results


def displacement_correlation(head, pairs: list[dict], device):
    """
    Spearman correlation between quasimetric distance and actual displacement.
    A well-grounded quasimetric should correlate strongly with metres.
    """
    from scipy.stats import spearmanr
    head.eval()

    z_t  = torch.from_numpy(np.stack([p["z_t"]  for p in pairs])).to(device)
    z_tk = torch.from_numpy(np.stack([p["z_tk"] for p in pairs])).to(device)
    disps = np.array([p["disp"] for p in pairs])

    with torch.no_grad():
        e_t  = head(z_t)
        e_tk = head(z_tk)
        d = 1.0 - F.cosine_similarity(
            F.normalize(e_t, dim=-1), F.normalize(e_tk, dim=-1), dim=-1)

    rho, pval = spearmanr(d.cpu().numpy(), disps)
    return float(rho), float(pval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",      required=True)
    ap.add_argument("--rope-ckpt", default=None)
    ap.add_argument("--mlp-ckpt",  default=None)
    ap.add_argument("--encoder-ckpt", default=None)
    ap.add_argument("--max-files", type=int, default=20)
    ap.add_argument("--n-pairs",   type=int, default=2000)
    ap.add_argument("--batch",     type=int, default=128)
    ap.add_argument("--embed-dim", type=int, default=96)
    ap.add_argument("--seed",      type=int, default=99)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Encoder ───────────────────────────────────────────────────────────
    encoder = StudentEncoder()
    if args.encoder_ckpt:
        ckpt  = torch.load(args.encoder_ckpt, map_location="cpu", weights_only=False)
        state = ckpt.get("student", ckpt.get("model", ckpt))
        encoder.load_state_dict(state, strict=False)
    encoder.eval()

    # ── Dataset (held-out files only) ─────────────────────────────────────
    hdf_files = sorted(Path(args.data).glob("jackal_2019-*.hdf5"))
    if args.max_files:
        hdf_files = hdf_files[-args.max_files:]   # take last N (unseen by training)
    print(f"Evaluating on {len(hdf_files)} held-out files")

    print("Building eval dataset...")
    ds = RECONPairDataset(hdf_files, encoder, n_pairs=args.n_pairs,
                          device=device, seed=args.seed)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)
    pairs  = ds._pairs   # for per-k and correlation analysis

    # ── Load heads ────────────────────────────────────────────────────────
    rope = RoPETemporalHead(embed_dim=args.embed_dim).to(device)
    if args.rope_ckpt:
        ckpt = torch.load(args.rope_ckpt, map_location="cpu", weights_only=False)
        rope.load_state_dict(ckpt["head"])
        print(f"RoPE checkpoint: {args.rope_ckpt}  (epoch {ckpt.get('epoch','?')}  AUROC {ckpt.get('auroc',0):.4f})")
    else:
        print("RoPE: random init (no checkpoint)")

    mlp = MLPTemporalHead().to(device)
    if args.mlp_ckpt:
        ckpt = torch.load(args.mlp_ckpt, map_location="cpu", weights_only=False)
        state = ckpt.get("head", ckpt.get("model", ckpt))
        mlp.load_state_dict(state, strict=False)
        print(f"MLP  checkpoint: {args.mlp_ckpt}")
    else:
        print("MLP:  random init (no checkpoint)")

    # ── Full AUROC ────────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"{'Metric':<25} {'MLP':>10} {'RoPE':>10} {'Δ':>8}")
    print(f"{'─'*55}")

    rm = evaluate(mlp,  loader, device)
    rr = evaluate(rope, loader, device)

    for k, label in [("auroc","AUROC"), ("close","Dist close"),
                     ("far","Dist far"), ("sep","Separation")]:
        vm = rm.get(k, 0); vr = rr.get(k, 0)
        print(f"  {label:<23} {vm:>10.4f} {vr:>10.4f} {vr-vm:>+8.4f}")

    # ── Per-k breakdown ───────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("Per-k AUROC breakdown:")
    mlp_k  = per_k_auroc(mlp,  pairs, device)
    rope_k = per_k_auroc(rope, pairs, device)
    for bucket in mlp_k:
        vm = mlp_k[bucket]["auroc"]; vr = rope_k.get(bucket, {}).get("auroc", 0)
        n  = mlp_k[bucket]["n"]
        print(f"  {bucket:<10} n={n:<5}  MLP={vm:.4f}  RoPE={vr:.4f}  Δ={vr-vm:+.4f}")

    # ── Displacement correlation ──────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("Displacement correlation (Spearman ρ, higher = more geometric):")
    try:
        from scipy.stats import spearmanr
        rho_m, pval_m = displacement_correlation(mlp,  pairs, device)
        rho_r, pval_r = displacement_correlation(rope, pairs, device)
        print(f"  MLP   ρ={rho_m:+.4f}  p={pval_m:.2e}")
        print(f"  RoPE  ρ={rho_r:+.4f}  p={pval_r:.2e}  Δρ={rho_r-rho_m:+.4f}")
    except ImportError:
        print("  (scipy not installed — skipping correlation)")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'═'*55}")
    winner = "✅ RoPE wins" if rr["auroc"] > rm["auroc"] else "MLP wins"
    print(f"AUROC:  MLP={rm['auroc']:.4f}  RoPE={rr['auroc']:.4f}  "
          f"Δ={rr['auroc']-rm['auroc']:+.4f}  {winner}")
    print(f"{'═'*55}")


if __name__ == "__main__":
    main()
