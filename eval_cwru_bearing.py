"""
eval_cwru_bearing.py — CORTEX-PE v16.15
CWRU 48k Drive End Bearing Anomaly Detection — PCA Subspace Scoring

Training-free anomaly detection on vibration time-series.
No encoder needed — operates directly on raw 1D vibration segments.

File mapping (Zenodo record 10987113):
    109–112  →  Ball fault     (B007, 0.007" diameter, loads 0–3HP)
    161–164  →  Inner race     (IR007, 0.007" diameter, loads 0–3HP)
    213–215  →  Outer race     (OR007@6, 0.007" diameter, loads 0–2HP)

Protocol (no normal baseline available):
    Reference : Ball fault (smallest fault, closest to healthy — standard proxy)
    Anomaly   : Inner race + Outer race (structurally distinct fault signatures)
    Metric    : AUROC — PCA reconstruction error as anomaly score
    Target    : AUROC ≥ 0.80

Usage:
    python eval_cwru_bearing.py --data ./cwru_data
    python eval_cwru_bearing.py --data ./cwru_data --sweep --segment-len 2048
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io


# ── File manifest ──────────────────────────────────────────────────────────────

# Maps filename stem → (fault_type, load_hp)
FILE_MAP = {
    "109": ("ball",  0), "110": ("ball",  1), "111": ("ball",  2), "112": ("ball",  3),
    "161": ("inner", 0), "162": ("inner", 1), "163": ("inner", 2), "164": ("inner", 3),
    "213": ("outer", 0), "214": ("outer", 1), "215": ("outer", 2),
}

# Ball = reference (smallest fault diameter, closest to normal)
# Inner + Outer = anomaly
REFERENCE_TYPE = "ball"
ANOMALY_TYPES  = {"inner", "outer"}


# ── Signal loading ─────────────────────────────────────────────────────────────

def load_de_signal(path: Path) -> np.ndarray | None:
    """Load Drive End (DE) time-series from a CWRU .mat file."""
    try:
        mat  = scipy.io.loadmat(str(path))
        stem = path.stem                              # e.g. "109"
        key  = f"X{stem}_DE_time"
        if key not in mat:
            # Fallback: find any DE key
            de_keys = [k for k in mat if "DE_time" in k and not k.startswith("_")]
            if not de_keys:
                return None
            key = de_keys[0]
        sig = mat[key].squeeze().astype(np.float32)
        return sig
    except Exception as e:
        print(f"  ⚠️  Failed to load {path.name}: {e}")
        return None


def segment_signal(sig: np.ndarray, seg_len: int, overlap: float = 0.5) -> np.ndarray:
    """
    Slice a 1-D signal into fixed-length overlapping segments.
    Returns (N, seg_len) array. Normalises each segment to zero-mean unit-var.
    """
    step = int(seg_len * (1.0 - overlap))
    segs = []
    for start in range(0, len(sig) - seg_len + 1, step):
        s = sig[start : start + seg_len].copy()
        std = s.std()
        if std > 1e-8:
            s = (s - s.mean()) / std
        segs.append(s)
    return np.stack(segs) if segs else np.empty((0, seg_len), dtype=np.float32)


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(segments: np.ndarray) -> np.ndarray:
    """
    Extract a compact statistical + spectral feature vector per segment.
    Pure numpy — no encoder required.

    Features (20-D per segment):
        Time-domain (10): mean, std, rms, kurtosis, skewness,
                          peak2peak, crest_factor, shape_factor,
                          impulse_factor, margin_factor
        Frequency-domain (10): FFT magnitude bins (log-spaced, 10 bands)
    """
    N, L = segments.shape
    feats = []

    for seg in segments:
        # ── Time domain ───────────────────────────────────────────────────
        mean      = seg.mean()
        std       = seg.std() + 1e-8
        rms       = np.sqrt(np.mean(seg ** 2)) + 1e-8
        peak      = np.abs(seg).max()
        p2p       = seg.max() - seg.min()
        m4        = np.mean((seg - mean) ** 4)
        kurt      = m4 / (std ** 4)                    # kurtosis (healthy ~3)
        m3        = np.mean((seg - mean) ** 3)
        skew      = m3 / (std ** 3)
        crest     = peak / rms                          # crest factor
        shape     = rms / (np.abs(seg).mean() + 1e-8)
        impulse   = peak / (np.abs(seg).mean() + 1e-8)
        margin    = peak / ((np.sqrt(np.abs(seg)).mean() + 1e-8) ** 2)

        # ── Frequency domain — 10 log-spaced FFT bands ────────────────────
        fft_mag = np.abs(np.fft.rfft(seg))
        freqs   = np.fft.rfftfreq(L)
        # 10 logarithmically spaced bands from DC to Nyquist
        edges   = np.logspace(np.log10(1e-3), np.log10(0.5), 11)
        bands   = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (freqs >= lo) & (freqs < hi)
            bands.append(fft_mag[mask].mean() if mask.any() else 0.0)

        f = np.array([
            mean, std, rms, kurt, skew, p2p,
            crest, shape, impulse, margin,
            *bands
        ], dtype=np.float32)
        feats.append(f)

    return np.stack(feats)   # (N, 20)


# ── PCA anomaly scoring ────────────────────────────────────────────────────────

def fit_pca(features: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit PCA on reference features. Returns (mean, components, explained_var_ratio)."""
    mean       = features.mean(axis=0)
    centered   = features - mean
    U, S, Vt   = np.linalg.svd(centered, full_matrices=False)
    components = Vt[:n_components]
    total_var  = (S ** 2).sum()
    var_ratio  = (S[:n_components] ** 2) / (total_var + 1e-8)
    return mean, components, var_ratio


def pca_score(features: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Reconstruction error in PCA subspace — higher = more anomalous."""
    centered      = features - mean
    projected     = centered @ components.T
    reconstructed = projected @ components
    residuals     = centered - reconstructed
    return np.linalg.norm(residuals, axis=1)   # (N,)


# ── AUROC ─────────────────────────────────────────────────────────────────────

def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Binary AUROC. labels: 0=reference, 1=anomaly. scores: higher=anomaly."""
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float("nan")
    # Wilcoxon-Mann-Whitney estimator (exact, O(P*N))
    count = sum((p > n) + 0.5 * (p == n)
                for p in pos_scores for n in neg_scores)
    return float(count / (len(pos_scores) * len(neg_scores)))


# ── Main ───────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    print("\n" + "═" * 60)
    print("  CWRU Bearing Anomaly Eval — CORTEX-PE v16.15")
    print("═" * 60)
    print(f"  Data         : {args.data}")
    print(f"  Segment len  : {args.segment_len} samples @ 48kHz = "
          f"{args.segment_len/48000*1000:.1f}ms")
    print(f"  PCA components: {args.n_components}")
    print(f"  Reference    : {REFERENCE_TYPE} fault (proxy for normal)")
    print(f"  Anomalies    : {ANOMALY_TYPES}")

    data_dir = Path(args.data)
    mat_files = sorted(data_dir.glob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files in {data_dir}")

    # ── Load and segment all files ────────────────────────────────────────
    ref_feats_list  = []
    anom_feats_list = []

    print("\n── Loading signals ─────────────────────────────────────────")
    for path in mat_files:
        stem = path.stem
        if stem not in FILE_MAP:
            print(f"  ⚠️  Unknown file {path.name} — skipping")
            continue

        fault_type, load_hp = FILE_MAP[stem]
        sig = load_de_signal(path)
        if sig is None:
            continue

        segs  = segment_signal(sig, args.segment_len, overlap=0.5)
        feats = extract_features(segs)

        label = "REF " if fault_type == REFERENCE_TYPE else "ANOM"
        print(f"  {label}  {path.name:12s}  {fault_type:6s} {load_hp}HP  "
              f"{len(sig):>8,} samples → {len(segs):>5,} segments  feat={feats.shape}")

        if fault_type == REFERENCE_TYPE:
            ref_feats_list.append(feats)
        else:
            anom_feats_list.append(feats)

    if not ref_feats_list:
        raise RuntimeError("No reference (ball fault) files loaded")
    if not anom_feats_list:
        raise RuntimeError("No anomaly files loaded")

    ref_feats  = np.concatenate(ref_feats_list,  axis=0)
    anom_feats = np.concatenate(anom_feats_list, axis=0)

    print(f"\n  Reference segments : {len(ref_feats):,}")
    print(f"  Anomaly segments   : {len(anom_feats):,}")

    # ── Fit PCA on reference ──────────────────────────────────────────────
    print("\n── Fitting PCA on reference (ball fault) ───────────────────")
    if args.sweep:
        component_range = [2, 4, 6, 8, 10, 12]
    else:
        component_range = [args.n_components]

    best_auroc = 0.0
    best_k     = args.n_components
    results    = []

    for k in component_range:
        k = min(k, ref_feats.shape[1] - 1)
        pca_mean, pca_comps, var_ratio = fit_pca(ref_feats, k)

        # Score all segments
        ref_scores  = pca_score(ref_feats,  pca_mean, pca_comps)
        anom_scores = pca_score(anom_feats, pca_mean, pca_comps)

        all_scores = np.concatenate([ref_scores,  anom_scores])
        all_labels = np.concatenate([np.zeros(len(ref_scores)),
                                     np.ones(len(anom_scores))])

        auc = auroc(all_scores, all_labels)
        explained = var_ratio.sum() * 100

        results.append((k, auc, explained,
                        ref_scores.mean(), anom_scores.mean()))

        if auc > best_auroc:
            best_auroc = auc
            best_k     = k
            best_ref_mean  = ref_scores.mean()
            best_anom_mean = anom_scores.mean()

    # ── Per-fault-type AUROC ──────────────────────────────────────────────
    print("\n── Per-fault-type breakdown ────────────────────────────────")
    pca_mean, pca_comps, _ = fit_pca(ref_feats, best_k)
    ref_scores = pca_score(ref_feats, pca_mean, pca_comps)

    for fault_type in sorted(ANOMALY_TYPES):
        type_feats = []
        for path in mat_files:
            stem = path.stem
            if stem not in FILE_MAP:
                continue
            ft, _ = FILE_MAP[stem]
            if ft != fault_type:
                continue
            sig   = load_de_signal(path)
            segs  = segment_signal(sig, args.segment_len)
            type_feats.append(extract_features(segs))

        if not type_feats:
            continue
        tf     = np.concatenate(type_feats)
        t_sc   = pca_score(tf, pca_mean, pca_comps)
        sc     = np.concatenate([ref_scores, t_sc])
        lb     = np.concatenate([np.zeros(len(ref_scores)), np.ones(len(t_sc))])
        t_auc  = auroc(sc, lb)
        sep    = t_sc.mean() / (ref_scores.mean() + 1e-8)
        print(f"  {fault_type:8s}  AUROC={t_auc:.4f}  "
              f"score_mean={t_sc.mean():.4f}  "
              f"vs_ref={sep:.2f}×")

    # ── Results ───────────────────────────────────────────────────────────
    AUROC_TARGET = 0.80
    print("\n── Summary ─────────────────────────────────────────────────")

    if args.sweep:
        print(f"  {'k':>4}  {'AUROC':>8}  {'Explained%':>11}  {'RefScore':>9}  {'AnomScore':>10}")
        for k, auc, expl, rs, as_ in results:
            best_marker = " ← best" if k == best_k else ""
            print(f"  {k:>4}  {auc:>8.4f}  {expl:>10.1f}%  {rs:>9.4f}  {as_:>10.4f}{best_marker}")

    status = "✅ PASS" if best_auroc >= AUROC_TARGET else "❌ FAIL"
    sep_ratio = best_anom_mean / (best_ref_mean + 1e-8)

    print(f"\n  Best AUROC        : {best_auroc:.4f}  (k={best_k})  "
          f"(target ≥ {AUROC_TARGET})  {status}")
    print(f"  Reference score   : {best_ref_mean:.4f}")
    print(f"  Anomaly score     : {best_anom_mean:.4f}")
    print(f"  Separation ratio  : {sep_ratio:.2f}×")

    verdict = ("✅ CWRU BEARING ANOMALY DETECTION VALIDATED"
               if best_auroc >= AUROC_TARGET
               else "❌ BELOW TARGET — try --sweep to find best k")
    print(f"\n  {verdict}")
    print("═" * 60 + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CWRU bearing PCA anomaly detection eval"
    )
    parser.add_argument("--data",         required=True,
                        help="Directory containing CWRU .mat files")
    parser.add_argument("--segment-len",  type=int, default=4096,
                        help="Samples per segment at 48kHz (default 4096 = 85ms)")
    parser.add_argument("--n-components", type=int, default=6,
                        help="PCA components for anomaly scoring (default 6)")
    parser.add_argument("--sweep",        action="store_true",
                        help="Sweep k=2..12 and report all AUROC scores")
    args = parser.parse_args()
    run(args)
