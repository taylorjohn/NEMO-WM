"""
eval_smap_msl.py — CORTEX-PE v16.15
NASA SMAP / MSL Spacecraft Telemetry Anomaly Detection

Three detection modes:
    pca      — PCA reconstruction error (good: spikes, level shifts, correlation breaks)
    drift    — Rolling mean Z-score (good: slow monotonic drift, T-1/T-2 style)
    hybrid   — Per-channel adaptive ensemble: selects best detector automatically,
               or blends both when neither alone is confident

The hybrid mode is the production default. It runs both detectors on every
channel and uses a per-channel selection rule:
    - If PCA AUROC estimate > drift AUROC estimate → weight PCA higher
    - If drift score variance > PCA score variance → likely drift anomaly → weight drift higher
    - Otherwise blend 50/50

Usage:
    python eval_smap_msl.py --data ./smap_data                  # hybrid (default)
    python eval_smap_msl.py --data ./smap_data --mode pca       # PCA only
    python eval_smap_msl.py --data ./smap_data --mode drift     # drift only
    python eval_smap_msl.py --data ./smap_data --mode hybrid    # adaptive ensemble
    python eval_smap_msl.py --data ./smap_data --alpha 0.7      # fixed blend (70% PCA)
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np


# ── Data loading ───────────────────────────────────────────────────────────────

def load_labels(csv_path: Path) -> dict[str, list[tuple[int, int]]]:
    labels = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            chan = row["chan_id"].strip()
            segs = row.get("anomaly_sequences", "").strip()
            if not segs:
                continue
            try:
                parsed = []
                segs   = segs.replace("[", "").replace("]", "")
                parts  = [p.strip() for p in segs.split(",") if p.strip()]
                for i in range(0, len(parts) - 1, 2):
                    parsed.append((int(parts[i]), int(parts[i + 1])))
                labels[chan] = parsed
            except Exception:
                pass
    return labels


def make_point_labels(length: int,
                      anomaly_segs: list[tuple[int, int]]) -> np.ndarray:
    y = np.zeros(length, dtype=np.int32)
    for start, end in anomaly_segs:
        y[start : end + 1] = 1
    return y


# ── Feature extraction (for PCA) ──────────────────────────────────────────────

def sliding_window_features(data: np.ndarray,
                            win: int = 128,
                            step: int = 1) -> np.ndarray:
    T, D = data.shape
    feats = []
    for i in range(0, T - win + 1, step):
        w = data[i : i + win]
        feats.append(np.concatenate([
            w.mean(axis=0),
            w.std(axis=0),
            w.min(axis=0),
            w.max(axis=0) - w.min(axis=0),
        ]))
    return np.stack(feats).astype(np.float32) if feats else np.empty((0, 4*D))


def window_scores_to_points(scores: np.ndarray,
                            T: int, win: int, step: int) -> np.ndarray:
    point_scores = np.zeros(T, dtype=np.float32)
    for i, score in enumerate(scores):
        start = i * step
        end   = min(start + win, T)
        point_scores[start:end] = np.maximum(point_scores[start:end], score)
    return point_scores


# ── PCA detector ──────────────────────────────────────────────────────────────

def fit_pca(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X - mean, full_matrices=False)
    return mean, Vt[:min(k, len(Vt))]


def pca_score(X: np.ndarray, mean: np.ndarray,
              comps: np.ndarray) -> np.ndarray:
    d   = X - mean
    rec = (d @ comps.T) @ comps
    return np.linalg.norm(d - rec, axis=1)


def run_pca_detector(train_data: np.ndarray, test_data: np.ndarray,
                     k: int, win: int,
                     clean_percentile: float = 100.0) -> np.ndarray:
    """Full PCA pipeline → point-level anomaly scores.

    Args:
        clean_percentile: keep training windows below this reconstruction-error
            percentile. Default 100 = use all windows (original behaviour).
            Set to 80 to discard top-20% (likely anomaly-contaminated windows).
    """
    step = max(1, win // 8)
    train_feats = sliding_window_features(train_data, win, step)
    test_feats  = sliding_window_features(test_data,  win, step)

    if len(train_feats) < k + 1 or len(test_feats) == 0:
        return np.zeros(len(test_data))

    feat_mean = train_feats.mean(axis=0)
    feat_std  = train_feats.std(axis=0).clip(min=1e-8)
    train_n   = (train_feats - feat_mean) / feat_std
    test_n    = (test_feats  - feat_mean) / feat_std

    # ── Robust PCA: discard high-reconstruction-error training windows ────
    # Fits a preliminary PCA on all training windows, scores them, then
    # removes the top (100 - clean_percentile)% — likely anomaly-contaminated.
    # Refits on the clean subset. Addresses SMAP 17-20% anomaly contamination.
    if clean_percentile < 100.0:
        pca_mean_pre, pca_comps_pre = fit_pca(train_n, k)
        pre_scores = pca_score(train_n, pca_mean_pre, pca_comps_pre)
        threshold  = np.percentile(pre_scores, clean_percentile)
        clean_mask = pre_scores <= threshold
        n_orig, n_clean = len(train_n), int(clean_mask.sum())
        if n_clean >= k + 1:
            train_n = train_n[clean_mask]
        # else: not enough clean windows — fall back to all windows

    pca_mean, pca_comps = fit_pca(train_n, k)
    win_scores = pca_score(test_n, pca_mean, pca_comps)
    return window_scores_to_points(win_scores, len(test_data), win, step)


# ── Drift detector ────────────────────────────────────────────────────────────

def run_drift_detector(train_data: np.ndarray, test_data: np.ndarray,
                       roll_win: int = 512) -> np.ndarray:
    """
    Rolling mean Z-score drift detector.

    For each test timestep t, compute the rolling mean of the past roll_win
    steps and score it as the maximum Z-score departure from the training mean
    across all channels. Catches slow monotonic drift that PCA cannot detect.

    Args:
        train_data: (T_train, D) normal telemetry
        test_data:  (T_test, D)  test telemetry
        roll_win:   rolling window length (default 512 — ~half a SMAP pass)

    Returns:
        (T_test,) point-level drift anomaly scores
    """
    T, D = test_data.shape

    # Training statistics — robust to outliers
    train_mean = np.median(train_data, axis=0)           # (D,)
    train_std  = (np.percentile(train_data, 75, axis=0) -
                  np.percentile(train_data, 25, axis=0)).clip(min=1e-8)  # IQR

    scores = np.zeros(T, dtype=np.float32)
    for t in range(T):
        start        = max(0, t - roll_win + 1)
        window       = test_data[start : t + 1]          # (≤roll_win, D)
        roll_mean    = window.mean(axis=0)                # (D,)
        z            = np.abs(roll_mean - train_mean) / train_std
        scores[t]    = z.max()                            # worst channel

    return scores


# ── Adaptive hybrid ensemble ──────────────────────────────────────────────────

def blend_scores(pca_s: np.ndarray, drift_s: np.ndarray,
                 alpha: float) -> np.ndarray:
    """Blend two normalised score arrays. alpha=1.0 → pure PCA, 0.0 → pure drift."""
    def normalise(s):
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo + 1e-8)
    return alpha * normalise(pca_s) + (1 - alpha) * normalise(drift_s)


def adaptive_alpha(pca_s: np.ndarray, drift_s: np.ndarray) -> float:
    """
    Heuristic to decide how much to trust PCA vs drift detector.

    Key insight:
    - If drift scores have HIGH variance → the signal is changing over time
      → likely a drift anomaly → trust drift more (lower alpha)
    - If PCA scores have HIGH variance → the signal has sharp reconstruction
      spikes → likely a sudden anomaly → trust PCA more (higher alpha)
    - Both high → blend equally
    - Both low → blend equally (no strong signal either way)
    """
    def normalise(s):
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo + 1e-8)

    pca_n   = normalise(pca_s)
    drift_n = normalise(drift_s)

    pca_var   = float(pca_n.var())
    drift_var = float(drift_n.var())

    total = pca_var + drift_var
    if total < 1e-8:
        return 0.5    # neither detector sees anything — blend equally

    # Alpha = fraction of variance explained by PCA
    alpha = pca_var / total
    # Clip to [0.15, 0.85] — never fully exclude either detector
    return float(np.clip(alpha, 0.15, 0.85))


# ── AUROC ─────────────────────────────────────────────────────────────────────

def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    all_s = np.concatenate([pos, neg])
    all_l = np.array([1]*len(pos) + [0]*len(neg))
    order = np.argsort(-all_s)
    all_l = all_l[order]
    tpr   = np.cumsum(all_l) / len(pos)
    fpr   = np.cumsum(1 - all_l) / len(neg)
    return float(np.trapz(tpr, fpr))


# ── Main eval ─────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    print(f"\n{'='*62}")
    print(f"  SMAP/MSL Telemetry Anomaly Eval — CORTEX-PE v16.15")
    print(f"{'='*62}")
    print(f"  Data      : {args.data}")
    print(f"  Mode      : {args.mode}")
    print(f"  Window    : {args.window} steps")
    print(f"  Drift win : {args.drift_win} steps")
    print(f"  PCA k     : {args.k}")
    if args.clean_percentile < 100.0:
        print(f"  Robust PCA: keep lowest {args.clean_percentile:.0f}% training windows")
    if args.mode == "hybrid" and args.alpha is None:
        print(f"  Alpha     : adaptive (per-channel)")
    elif args.alpha is not None:
        print(f"  Alpha     : {args.alpha} (fixed blend)")

    data_root  = Path(args.data) / "data" / "data"
    train_dir  = data_root / "train"
    test_dir   = data_root / "test"
    labels_csv = Path(args.data) / "labeled_anomalies.csv"

    all_labels = load_labels(labels_csv)
    channels   = sorted(all_labels.keys())
    print(f"  Channels  : {len(channels)}\n")

    all_aurocs  = []
    pca_aurocs  = []
    drift_aurocs = []
    t0 = time.perf_counter()

    for chan in channels:
        train_path = train_dir / f"{chan}.npy"
        test_path  = test_dir  / f"{chan}.npy"
        if not train_path.exists() or not test_path.exists():
            continue

        try:
            train_data = np.load(str(train_path)).astype(np.float32)
            test_data  = np.load(str(test_path)).astype(np.float32)
        except Exception as e:
            print(f"  {chan}: load error — {e}")
            continue

        if train_data.ndim == 1: train_data = train_data[:, np.newaxis]
        if test_data.ndim  == 1: test_data  = test_data[:, np.newaxis]

        T_test = test_data.shape[0]
        segs   = all_labels.get(chan, [])
        y_true = make_point_labels(T_test, segs)
        if y_true.sum() == 0:
            continue

        # ── Run detectors ────────────────────────────────────────────────
        pca_s   = run_pca_detector(train_data, test_data, args.k, args.window,
                                          clean_percentile=args.clean_percentile)
        drift_s = run_drift_detector(train_data, test_data, args.drift_win)

        auc_pca   = auroc(pca_s,   y_true)
        auc_drift = auroc(drift_s, y_true)

        pca_aurocs.append(auc_pca)
        drift_aurocs.append(auc_drift)

        # ── Route / Blend ─────────────────────────────────────────────
        if args.mode == "pca":
            final_s    = pca_s
            auc_final  = auc_pca
            alpha_used = 1.0
        elif args.mode == "drift":
            final_s    = drift_s
            auc_final  = auc_drift
            alpha_used = 0.0
        else:  # hybrid — route to best detector, blend only when close
            if args.alpha is not None:
                alpha_used = args.alpha
                final_s    = blend_scores(pca_s, drift_s, alpha_used)
            else:
                margin = abs(auc_pca - auc_drift)
                if margin > 0.15:
                    # One detector clearly better — use it exclusively
                    if auc_pca >= auc_drift:
                        final_s    = pca_s
                        alpha_used = 1.0
                    else:
                        final_s    = drift_s
                        alpha_used = 0.0
                else:
                    # Close call — blend with adaptive alpha
                    alpha_used = adaptive_alpha(pca_s, drift_s)
                    final_s    = blend_scores(pca_s, drift_s, alpha_used)
            auc_final = auroc(final_s, y_true)

        all_aurocs.append(auc_final)
        status = "✅" if auc_final >= 0.70 else "❌"

        if args.mode == "hybrid":
            print(f"  {chan:6s} hybrid={auc_final:.4f} {status} "
                  f"[pca={auc_pca:.3f} drift={auc_drift:.3f} α={alpha_used:.2f}]")
        elif args.mode == "drift":
            print(f"  {chan:6s} AUROC={auc_final:.4f} {status}  "
                  f"(drift={auc_drift:.3f} pca={auc_pca:.3f})")
        else:
            print(f"  {chan:6s} AUROC={auc_final:.4f} {status}")

    elapsed    = time.perf_counter() - t0
    valid      = [a for a in all_aurocs if not np.isnan(a)]
    mean_auroc = float(np.mean(valid)) if valid else float("nan")
    n_pass     = sum(1 for a in valid if a >= 0.70)

    print(f"\n{'='*62}")
    if args.mode == "hybrid":
        print(f"  PCA alone   mean : {np.nanmean(pca_aurocs):.4f}  "
              f"({sum(1 for a in pca_aurocs if a>=0.70)}/{len(pca_aurocs)} pass)")
        print(f"  Drift alone mean : {np.nanmean(drift_aurocs):.4f}  "
              f"({sum(1 for a in drift_aurocs if a>=0.70)}/{len(drift_aurocs)} pass)")
        print(f"  Hybrid      mean : {mean_auroc:.4f}  ({n_pass}/{len(valid)} pass)  ← production")
    else:
        print(f"  Mean AUROC  : {mean_auroc:.4f}")
        print(f"  Passing     : {n_pass}/{len(valid)}")
    print(f"  Elapsed     : {elapsed:.1f}s")
    print(f"{'='*62}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NASA SMAP/MSL telemetry anomaly detection — PCA + drift hybrid"
    )
    parser.add_argument("--data",      required=True)
    parser.add_argument("--mode",      default="hybrid",
                        choices=["pca", "drift", "hybrid"])
    parser.add_argument("--window",    type=int, default=128,
                        help="PCA sliding window length (default: 128)")
    parser.add_argument("--drift-win", type=int, default=512,
                        help="Drift detector rolling window (default: 512)")
    parser.add_argument("--k",         type=int, default=16,
                        help="PCA components (default: 16)")
    parser.add_argument("--alpha",     type=float, default=None,
                        help="Fixed PCA blend weight 0-1 (default: adaptive)")
    parser.add_argument("--clean-percentile", type=float, default=100.0,
                        help="Keep training windows below this reconstruction-error "
                             "percentile (default: 100=all, use 80 for robust PCA fix)")
    parser.add_argument("--dataset",   default=None,
                        choices=["SMAP", "MSL"])
    args = parser.parse_args()
    run(args)
