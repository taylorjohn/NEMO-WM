"""
eval_bearing_subspace.py
=========================
Training-free PCA anomaly detection on CWRU bearing PNG frames.

Uses the frozen StudentEncoder to extract 128-D latent features,
fits PCA on normal frames, then scores all frames by reconstruction error.

Usage:
    python eval_bearing_subspace.py \
        --encoder ./checkpoints/maze_weak_sigreg_straight/cortex_student_phase2_final.pt \
        --data    ./bearing_data/cwru_all \
        --out     ./results/bearing_subspace_auroc.json

Expected: AUROC comparable to PatchCore (0.9929) if encoder geometry is good.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from collections import defaultdict


# ── Transform (matches StudentEncoder training) ──────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def load_encoder(ckpt_path: str, device: str):
    """Load StudentEncoder from checkpoint."""
    from student_encoder import StudentEncoder
    enc = StudentEncoder()
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "student" in state:
        state = state["student"]
    enc.load_state_dict(state, strict=False)
    enc.eval().to(device)
    return enc


@torch.no_grad()
def extract_features(encoder, img_paths, device, batch_size=64):
    """Extract encoder features for a list of image paths."""
    feats = []
    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i:i+batch_size]
        imgs = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(TRANSFORM(img))
            except Exception as e:
                print(f"  Warning: could not load {p}: {e}")
                imgs.append(torch.zeros(3, 224, 224))
        batch = torch.stack(imgs).to(device)
        z = encoder(batch)
        feats.append(z.cpu().numpy())
        if (i // batch_size) % 10 == 0:
            pct = min(100, int(100 * i / len(img_paths)))
            print(f"  Features: {pct}% ({i}/{len(img_paths)})", end="\r")
    print()
    return np.concatenate(feats, axis=0)


def run_subspace_ad(feats_normal, feats_all, n_components=16):
    """
    Fit PCA on normal features, score all by reconstruction error.
    Higher score = more anomalous.
    """
    pca = PCA(n_components=n_components)
    pca.fit(feats_normal)

    # Reconstruct and compute residual error
    z_proj   = pca.transform(feats_all)
    z_recon  = pca.inverse_transform(z_proj)
    errors   = np.linalg.norm(feats_all - z_recon, axis=1)
    return errors, pca


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder",   required=True)
    parser.add_argument("--data",      default="./bearing_data/cwru_all")
    parser.add_argument("--n-components", type=int, default=16,
                        help="PCA components for normal subspace")
    parser.add_argument("--sweep",     action="store_true",
                        help="Sweep n_components 4,8,16,32,64")
    parser.add_argument("--out",       default="./results/bearing_subspace_auroc.json")
    args = parser.parse_args()

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir  = Path(args.data)
    meta_path = data_dir / "metadata.json"

    assert meta_path.exists(), f"metadata.json not found in {data_dir}"

    print(f"Loading metadata from {meta_path}")
    meta = json.load(open(meta_path))
    print(f"  {len(meta)} frames")

    # ── Build frame lists ─────────────────────────────────────────────────
    normal_paths, normal_meta = [], []
    fault_paths,  fault_meta  = [], []

    fault_type_counts = defaultdict(int)
    for entry in meta:
        p = data_dir / entry["frame"]
        if not p.exists():
            continue
        if entry["label"] == 0:
            normal_paths.append(p)
            normal_meta.append(entry)
        else:
            fault_paths.append(p)
            fault_meta.append(entry)
            fault_type_counts[entry["fault_type"]] += 1

    print(f"  Normal frames:  {len(normal_paths)}")
    print(f"  Fault frames:   {len(fault_paths)}")
    print(f"  Fault types:    {dict(fault_type_counts)}")

    # ── Load encoder ──────────────────────────────────────────────────────
    print(f"\nLoading encoder from {args.encoder}")
    encoder = load_encoder(args.encoder, device)
    print(f"  Encoder on {device}")

    # ── Extract features ──────────────────────────────────────────────────
    all_paths = normal_paths + fault_paths
    all_labels = [0] * len(normal_paths) + [1] * len(fault_paths)

    print(f"\nExtracting features ({len(all_paths)} frames)...")
    t0 = time.time()
    feats = extract_features(encoder, all_paths, device)
    print(f"  Done in {time.time()-t0:.1f}s — shape: {feats.shape}")

    feats_normal = feats[:len(normal_paths)]
    feats_all    = feats

    labels = np.array(all_labels)

    # ── Train/test split: use 80% normal for fitting, evaluate on all ─────
    n_train = int(0.8 * len(normal_paths))
    feats_train = feats_normal[:n_train]
    print(f"\nFitting PCA on {n_train} normal frames (80% of normal)")

    # ── Sweep or single run ───────────────────────────────────────────────
    components_to_try = [4, 8, 16, 32, 64] if args.sweep else [args.n_components]

    results = {}
    best_auroc, best_k = 0.0, args.n_components

    for k in components_to_try:
        k = min(k, n_train - 1, feats.shape[1])
        errors, pca = run_subspace_ad(feats_train, feats_all, n_components=k)
        auroc = roc_auc_score(labels, errors)

        var_explained = pca.explained_variance_ratio_.sum()
        print(f"  k={k:3d}  AUROC={auroc:.4f}  var_explained={var_explained:.3f}")

        results[k] = {
            "auroc":           round(float(auroc), 4),
            "n_components":    k,
            "var_explained":   round(float(var_explained), 4),
            "n_train_normal":  n_train,
            "n_test_total":    len(all_paths),
        }

        if auroc > best_auroc:
            best_auroc, best_k = auroc, k

    # ── Per-fault-type breakdown (best k) ─────────────────────────────────
    print(f"\nPer-fault-type breakdown (k={best_k}):")
    errors_best, _ = run_subspace_ad(feats_train, feats_all, n_components=best_k)

    fault_breakdown = {}
    for fault_type in fault_type_counts:
        idxs = [i for i, m in enumerate(fault_meta)
                if m["fault_type"] == fault_type]
        normal_idxs = list(range(len(normal_paths)))

        ft_labels = [0]*len(normal_idxs) + [1]*len(idxs)
        ft_errors = np.concatenate([
            errors_best[normal_idxs],
            errors_best[len(normal_paths) + np.array(idxs)]
        ])

        if len(set(ft_labels)) == 2:
            ft_auroc = roc_auc_score(ft_labels, ft_errors)
            print(f"  {fault_type:20s}: AUROC={ft_auroc:.4f}  n={len(idxs)}")
            fault_breakdown[fault_type] = round(float(ft_auroc), 4)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  SubspaceAD AUROC (best k={best_k}): {best_auroc:.4f}")
    print(f"  PatchCore baseline:                  0.9929")
    print(f"  Delta vs PatchCore:                  {best_auroc - 0.9929:+.4f}")
    print(f"{'='*50}")

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "sweep":           results,
        "best_k":          best_k,
        "best_auroc":      round(best_auroc, 4),
        "patchcore_baseline": 0.9929,
        "delta_patchcore": round(best_auroc - 0.9929, 4),
        "fault_breakdown": fault_breakdown,
        "encoder":         args.encoder,
        "data":            str(data_dir),
    }
    json.dump(output, open(out_path, "w"), indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
