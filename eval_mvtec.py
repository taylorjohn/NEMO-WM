"""eval_mvtec.py — CORTEX-PE v16.17
═══════════════════════════════════════════════════════════════════════════════
MVTec AD held-out evaluation.

Runs the full SubspaceAD pipeline per category and reports:
  1. Mean image-level AUROC across all 15 categories
  2. Per-category AUROC breakdown (textures vs objects)
  3. Per-defect-type breakdown (which defect types are detected)
  4. Competitive comparison table vs published baselines
  5. Size / speed efficiency ratio

Usage:
  python eval_mvtec.py --data ./data/mvtec
  python eval_mvtec.py --data ./data/mvtec --encoder-ckpt ./checkpoints/recon_student/student_best.pt
  python eval_mvtec.py --data ./data/mvtec --patch --k 32
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

# Reuse everything from train_mvtec
from train_mvtec import (
    StudentEncoder, SubspaceAD,
    ALL_CATEGORIES, TEXTURES, OBJECTS,
    load_category, encode_images,
    _TRANSFORM_256,
)
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Published baselines for comparison
# ─────────────────────────────────────────────────────────────────────────────

# Mean image-level AUROC from published papers (where available)
# Source: Bergmann et al. 2019/2021, respective papers
BASELINES: dict[str, dict] = {
    "Autoencoder (Bergmann 2019)": {
        "mean": 0.681, "params_m": 0.5, "note": "CNN AE, in-distribution"
    },
    "VAE (Bergmann 2019)": {
        "mean": 0.676, "params_m": 0.5, "note": ""
    },
    "SPADE (Cohen 2020)": {
        "mean": 0.855, "params_m": 25.0, "note": "WideResNet50 features"
    },
    "PaDiM (Defard 2020)": {
        "mean": 0.956, "params_m": 25.0, "note": "WideResNet50 features"
    },
    "PatchCore (Roth 2021)": {
        "mean": 0.992, "params_m": 68.0, "note": "WideResNet50, k-NN memory bank"
    },
    "TinyGLASS (ResNet18 INT8)": {
        "mean": 0.942, "params_m": 11.0, "note": "8MB, 20 FPS, edge-optimised"
    },
    "SimpleNet (Liu 2023)": {
        "mean": 0.980, "params_m": 25.0, "note": "WideResNet50 + discriminator"
    },
}

# Per-category published results from PatchCore (best reference)
PATCHCORE_PER_CAT: dict[str, float] = {
    "carpet": 0.988, "grid": 0.986, "leather": 1.000,
    "tile": 0.991, "wood": 0.990,
    "bottle": 1.000, "cable": 0.999, "capsule": 0.982,
    "hazelnut": 1.000, "metal_nut": 1.000, "pill": 0.969,
    "screw": 0.974, "toothbrush": 1.000, "transistor": 1.000,
    "zipper": 0.998,
}


# ─────────────────────────────────────────────────────────────────────────────
# Per-defect-type evaluation
# ─────────────────────────────────────────────────────────────────────────────

def eval_per_defect(
    cat_root: Path,
    encoder: StudentEncoder,
    device: torch.device,
    k: int = 16,
    patch_mode: bool = False,
) -> dict[str, float]:
    """AUROC per defect type for one category."""
    data = load_category(cat_root)
    train_feats = encode_images(data["train_good"], encoder, device,
                                patch_mode=patch_mode)
    ad = SubspaceAD(k=k)
    ad.fit(train_feats)

    # Score normal test images
    good_feats  = encode_images(data["test_good"], encoder, device,
                                patch_mode=patch_mode)
    good_scores = ad.score(good_feats) if len(good_feats) else np.array([])

    # Per-defect-type AUROC
    test_root = cat_root / "test"
    results = {}
    for sub in sorted(test_root.iterdir()):
        if not sub.is_dir() or sub.name == "good":
            continue
        from train_mvtec import load_images
        defect_paths  = load_images(sub)
        if not defect_paths:
            continue
        defect_feats  = encode_images(defect_paths, encoder, device,
                                      patch_mode=patch_mode)
        defect_scores = ad.score(defect_feats)

        all_scores = np.concatenate([good_scores, defect_scores])
        all_labels = np.array([0] * len(good_scores) + [1] * len(defect_scores))
        if len(np.unique(all_labels)) < 2:
            results[sub.name] = 0.5
        else:
            results[sub.name] = float(roc_auc_score(all_labels, all_scores))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="CORTEX-PE MVTec AD evaluation")
    ap.add_argument("--data",         required=True,       help="MVTec root dir")
    ap.add_argument("--encoder-ckpt", default=None)
    ap.add_argument("--k",            type=int, default=16)
    ap.add_argument("--patch",        action="store_true")
    ap.add_argument("--per-defect",   action="store_true",
                    help="Also report per-defect-type AUROC (slower)")
    ap.add_argument("--categories",   nargs="+", default=None)
    args = ap.parse_args()

    data_root  = Path(args.data)
    categories = args.categories or ALL_CATEGORIES
    mode       = "patch" if args.patch else "global"

    # ── Load encoder ──────────────────────────────────────────────────────────
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = StudentEncoder().to(device)
    if args.encoder_ckpt:
        ckpt  = torch.load(args.encoder_ckpt, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        miss, unexp = encoder.load_state_dict(state, strict=False)
        print(f"Encoder: {args.encoder_ckpt}  "
              f"loaded={len(state)-len(miss)}/{len(state)}")
    else:
        print("Encoder: random weights")
    encoder.eval()

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print(f"MVTec AD Evaluation  —  k={args.k}  mode={mode}")
    print(f"{'═'*65}")
    print(f"\n  {'Category':<15}  {'AUROC':>8}  {'PatchCore':>10}  {'Gap':>7}  {'Status'}")
    print(f"  {'─'*15}  {'─'*8}  {'─'*10}  {'─'*7}  {'─'*10}")

    results = []
    avail   = [c for c in categories if (data_root / c).exists()]
    t0      = time.time()

    for cat in avail:
        cat_root  = data_root / cat
        data      = load_category(cat_root)
        n_train   = len(data["train_good"])
        if n_train == 0:
            print(f"  {cat:<15}  SKIP (no train images)")
            continue

        train_f  = encode_images(data["train_good"], encoder, device,
                                  patch_mode=args.patch)
        test_g   = encode_images(data["test_good"],   encoder, device,
                                  patch_mode=args.patch)
        test_d   = encode_images(data["test_defect"], encoder, device,
                                  patch_mode=args.patch)

        ad = SubspaceAD(k=args.k)
        ad.fit(train_f)

        all_f  = np.concatenate([test_g, test_d])
        labels = np.array([0]*len(test_g) + [1]*len(test_d))
        scores = ad.score(all_f)

        auroc = float(roc_auc_score(labels, scores)) if len(np.unique(labels))>1 else 0.5
        pc    = PATCHCORE_PER_CAT.get(cat, 0.)
        gap   = auroc - pc
        status = "✅" if auroc >= 0.90 else ("🟡" if auroc >= 0.80 else "🔴")

        print(f"  {cat:<15}  {auroc:>8.4f}  {pc:>10.4f}  {gap:>+7.4f}  {status}")
        results.append({"category": cat, "auroc": auroc, "patchcore": pc})

    total_t = time.time() - t0
    aurocs  = [r["auroc"] for r in results]
    mean    = float(np.mean(aurocs)) if aurocs else 0.

    texture_mean = float(np.mean([r["auroc"] for r in results
                                   if r["category"] in TEXTURES])) if results else 0.
    object_mean  = float(np.mean([r["auroc"] for r in results
                                   if r["category"] in OBJECTS]))  if results else 0.

    print(f"\n{'─'*65}")
    print(f"  Mean AUROC     : {mean:.4f}")
    print(f"  Texture mean   : {texture_mean:.4f}")
    print(f"  Object mean    : {object_mean:.4f}")
    print(f"  Best           : {results[np.argmax(aurocs)]['category']}  ({max(aurocs):.4f})")
    print(f"  Worst          : {results[np.argmin(aurocs)]['category']}  ({min(aurocs):.4f})")
    print(f"  vs PatchCore   : {mean - 0.992:+.4f}  (PatchCore mean=0.992)")
    print(f"  vs TinyGLASS   : {mean - 0.942:+.4f}  (TinyGLASS mean=0.942)")
    print(f"  Total time     : {total_t:.1f}s")

    # ── Competitive comparison table ─────────────────────────────────────────
    print(f"\n── Competitive Comparison (image-level AUROC) ─────────────────────")
    print(f"  {'Method':<35}  {'AUROC':>6}  {'Params':>8}  {'Edge?'}")
    print(f"  {'─'*35}  {'─'*6}  {'─'*8}  {'─'*5}")
    for method, info in BASELINES.items():
        edge = "✅" if info["params_m"] < 15 else ""
        print(f"  {method:<35}  {info['mean']:>6.3f}  "
              f"{info['params_m']:>6.1f}M  {edge}")
    print(f"  {'CORTEX-PE StudentEncoder (ours)':<35}  {mean:>6.3f}  "
          f"{encoder.n_params/1e6:>6.3f}M  ✅ NPU")

    # ── Efficiency ratio ──────────────────────────────────────────────────────
    tiny_auroc_per_mb = 0.942 / 8.0
    our_auroc_per_mb  = mean / (encoder.n_params * 4 / 1e6)   # fp32 byte size
    print(f"\n── Efficiency Ratio ────────────────────────────────────────────────")
    print(f"  TinyGLASS   : {tiny_auroc_per_mb:.3f} AUROC/MB")
    print(f"  CORTEX-PE   : {our_auroc_per_mb:.3f} AUROC/MB  "
          f"({'%.1f' % (our_auroc_per_mb/tiny_auroc_per_mb)}× vs TinyGLASS)")

    # ── Per-defect breakdown ──────────────────────────────────────────────────
    if args.per_defect and avail:
        print(f"\n── Per-Defect-Type AUROC ───────────────────────────────────────────")
        for cat in avail[:5]:   # first 5 categories for brevity
            defect_aurocs = eval_per_defect(
                data_root / cat, encoder, device,
                k=args.k, patch_mode=args.patch,
            )
            print(f"\n  {cat}:")
            for defect, auc in sorted(defect_aurocs.items(), key=lambda x: -x[1]):
                bar = "█" * int(auc * 20)
                print(f"    {defect:<20}  {auc:.4f}  {bar}")

    # ── Lineage commit ────────────────────────────────────────────────────────
    try:
        from lineage import Lineage
        lin = Lineage("mvtec")
        lin.commit(
            run_id=f"mvtec_eval_{mode}_k{args.k}",
            metrics={
                "auroc": round(mean, 4),
                "texture_auroc": round(texture_mean, 4),
                "object_auroc":  round(object_mean, 4),
            },
            script=__file__,
            config={"k": args.k, "patch": args.patch,
                    "encoder_ckpt": args.encoder_ckpt},
            notes=f"Held-out eval {mode} k={args.k}",
            parent=lin.best_run_id(),
        )
    except ImportError:
        pass

    print(f"\n{'═'*65}")
    target = 0.88
    status = "✅ above 0.88 target" if mean >= target else f"⚠️  below {target} target"
    print(f"  Final AUROC = {mean:.4f}  {status}")
    print(f"{'═'*65}\n")


if __name__ == "__main__":
    main()
