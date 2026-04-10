"""
eval_mimii_perid.py — CORTEX-PE v16.15
MIMII Industrial Anomaly Detection — Per-ID Protocol with Delta Features

Protocol: For each machine model ID, fit PCA on that ID's own normal sounds
and score anomalies by reconstruction error. This is the correct unsupervised
benchmark — each physical machine gets its own reference distribution.

Features (256-D per clip):
    log-mel mean    (64-D)  — steady-state spectral shape
    log-mel std     (64-D)  — spectral variation
    delta mean      (64-D)  — rate of spectral change (temporal dynamics)
    delta std       (64-D)  — variation in rate of change

Delta features are critical for valve fault detection — valve anomalies are
timing/pressure irregularities that don't change spectral energy but do
change how quickly the spectrum evolves over the valve cycle.

Usage:
    python eval_mimii_perid.py                     # all machines
    python eval_mimii_perid.py --machine valve     # valve only
    python eval_mimii_perid.py --k 12              # more PCA components
    python eval_mimii_perid.py --no-delta          # baseline without delta
"""

from __future__ import annotations
import argparse
import pathlib
import time
import numpy as np
import scipy.io.wavfile as wavfile


# ── Audio loading ──────────────────────────────────────────────────────────────

TARGET_SR = 16000
CLIP_LEN  = 16000   # 1 second — centre of 10-second MIMII clips


def load_wav(path: pathlib.Path) -> np.ndarray | None:
    """Load WAV, take channel 0, resample to 16kHz, extract 1-second centre clip."""
    try:
        sr, sig = wavfile.read(str(path))
        sig = sig.astype(np.float32)
        if sig.ndim > 1:
            sig = sig[:, 0]          # MIMII is 8-channel — take ch0
        if sr != TARGET_SR:
            n   = int(len(sig) * TARGET_SR / sr)
            sig = np.interp(np.linspace(0, len(sig) - 1, n),
                            np.arange(len(sig)), sig).astype(np.float32)
        mid  = len(sig) // 2
        clip = sig[max(0, mid - CLIP_LEN // 2) :
                   max(0, mid - CLIP_LEN // 2) + CLIP_LEN]
        if len(clip) < CLIP_LEN:
            clip = np.pad(clip, (0, CLIP_LEN - len(clip)))
        peak = np.abs(clip).max()
        return clip / peak if peak > 1e-6 else clip
    except Exception:
        return None


# ── Log-mel feature extraction ─────────────────────────────────────────────────

def log_mel_spectrogram(clip: np.ndarray, sr: int = TARGET_SR,
                        n_mels: int = 64, n_fft: int = 1024,
                        hop: int = 256) -> np.ndarray:
    """
    Compute log-mel spectrogram.
    Returns (n_mels, n_frames) array.
    Smaller hop=256 gives more temporal resolution — important for delta features.
    """
    win    = np.hanning(n_fft)
    frames = []
    for i in range(0, len(clip) - n_fft + 1, hop):
        frames.append(np.abs(np.fft.rfft(clip[i : i + n_fft] * win)) ** 2)
    if not frames:
        return np.zeros((n_mels, 10), dtype=np.float32)
    S = np.stack(frames).T   # (n_fft//2+1, n_frames)

    # Mel filterbank
    mel_min = 2595 * np.log10(1 + 20 / 700)
    mel_max = 2595 * np.log10(1 + (sr / 2) / 700)
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts  = 700 * (10 ** (mel_pts / 2595) - 1)
    bins    = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    fb      = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            fb[m, k] = (k - bins[m]) / max(bins[m + 1] - bins[m], 1)
        for k in range(bins[m + 1], bins[m + 2]):
            fb[m, k] = (bins[m + 2] - k) / max(bins[m + 2] - bins[m + 1], 1)

    mel = fb @ S
    return np.log(mel + 1e-6).astype(np.float32)   # (n_mels, n_frames)


def extract_features(clip: np.ndarray, use_delta: bool = True) -> np.ndarray:
    """
    Extract 256-D feature vector from a 1-second clip.

    With delta (use_delta=True, default):
        [log_mel_mean(64) | log_mel_std(64) | delta_mean(64) | delta_std(64)]
        Total: 256-D

    Without delta (baseline, use_delta=False):
        [log_mel_mean(64) | log_mel_std(64)]
        Total: 128-D

    Delta = first-order difference along time axis — captures rate of spectral
    change. Critical for valve faults which manifest as timing irregularities
    rather than steady-state spectral differences.
    """
    log = log_mel_spectrogram(clip)      # (64, n_frames)

    if use_delta:
        # First-order difference along time axis (prepend first frame to preserve shape)
        delta = np.diff(log, axis=1, prepend=log[:, :1])  # (64, n_frames)
        return np.concatenate([
            log.mean(axis=1),    # 64-D steady-state mean
            log.std(axis=1),     # 64-D steady-state variation
            delta.mean(axis=1),  # 64-D temporal change mean
            delta.std(axis=1),   # 64-D temporal change variation
        ]).astype(np.float32)    # 256-D total
    else:
        return np.concatenate([
            log.mean(axis=1),    # 64-D
            log.std(axis=1),     # 64-D
        ]).astype(np.float32)    # 128-D total


# ── PCA anomaly scoring ────────────────────────────────────────────────────────

def fit_pca(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X - mean, full_matrices=False)
    return mean, Vt[:k]


def pca_score(X: np.ndarray, mean: np.ndarray, comps: np.ndarray) -> np.ndarray:
    d   = X - mean
    rec = (d @ comps.T) @ comps
    return np.linalg.norm(d - rec, axis=1)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    count = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return float(count / (len(pos) * len(neg)))


# ── Per-ID eval ────────────────────────────────────────────────────────────────

def eval_id(id_dir: pathlib.Path, k: int, use_delta: bool) -> float:
    norm_paths = list((id_dir / "normal").glob("*.wav"))
    abn_paths  = list((id_dir / "abnormal").glob("*.wav"))

    def feats(paths):
        fs = []
        for p in paths:
            sig = load_wav(p)
            if sig is not None:
                fs.append(extract_features(sig, use_delta=use_delta))
        return np.stack(fs) if fs else None

    nf = feats(norm_paths)
    af = feats(abn_paths)
    if nf is None or af is None:
        return float("nan")

    mean, comps = fit_pca(nf, min(k, nf.shape[1] - 1))
    ns  = pca_score(nf, mean, comps)
    as_ = pca_score(af, mean, comps)

    all_scores = np.concatenate([ns,  as_])
    all_labels = np.array([0] * len(ns) + [1] * len(as_))
    return auroc(all_scores, all_labels)


# ── Main ───────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    data_dir = pathlib.Path(args.data)
    machines = [args.machine] if args.machine else ["fan", "pump", "slider", "valve"]
    use_delta = not args.no_delta
    feat_dim  = 256 if use_delta else 128

    print(f"\n{'='*58}")
    print(f"  MIMII Per-ID Anomaly Eval — CORTEX-PE v16.15")
    print(f"{'='*58}")
    print(f"  Data      : {data_dir}")
    print(f"  Features  : {'256-D log-mel + delta' if use_delta else '128-D log-mel (baseline)'}")
    print(f"  PCA k     : {args.k}")
    print(f"  Protocol  : per-ID (own normal → own abnormal)\n")

    all_aurocs = []
    t0 = time.perf_counter()

    for machine in machines:
        base = data_dir / machine
        if not base.exists():
            print(f"  {machine}: not found at {base}")
            continue

        id_dirs = sorted(base.glob("id_*"))
        if not id_dirs:
            print(f"  {machine}: no id_XX directories found")
            continue

        machine_aurocs = []
        for id_dir in id_dirs:
            auc = eval_id(id_dir, k=args.k, use_delta=use_delta)
            machine_aurocs.append(auc)
            status = "✅" if auc >= 0.70 else "❌"
            print(f"  {machine:8s}  {id_dir.name}  AUROC={auc:.4f}  {status}")

        mean_auc = float(np.nanmean(machine_aurocs))
        all_aurocs.extend(machine_aurocs)
        status = "✅ PASS" if mean_auc >= 0.70 else "❌ FAIL"
        print(f"  {machine:8s}  MEAN ={mean_auc:.4f}  {status}\n")

    overall = float(np.nanmean(all_aurocs))
    elapsed = time.perf_counter() - t0

    print(f"{'='*58}")
    print(f"  Overall mean AUROC : {overall:.4f}")
    print(f"  Elapsed            : {elapsed:.1f}s")
    print(f"  Features           : {'256-D (log-mel + delta)' if use_delta else '128-D (log-mel only)'}")
    print(f"{'='*58}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MIMII per-ID anomaly detection with delta features"
    )
    parser.add_argument("--data",     default="./mimii_data",
                        help="MIMII data directory (default: ./mimii_data)")
    parser.add_argument("--machine",  default=None,
                        choices=["fan", "pump", "slider", "valve"],
                        help="Single machine to eval (default: all)")
    parser.add_argument("--k",        type=int, default=8,
                        help="PCA components (default: 8)")
    parser.add_argument("--no-delta", action="store_true",
                        help="Disable delta features (128-D baseline)")
    args = parser.parse_args()
    run(args)
