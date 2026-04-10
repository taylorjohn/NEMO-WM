"""
eval_mimii.py — CORTEX-PE v16.15
MIMII Industrial Machine Anomaly Detection — WavLM Distillation + PCA Scoring

Two complementary approaches:
    1. Training-free PCA on raw log-mel features (fast, no model needed)
    2. WavLM-student PCA scoring (if cardiac student checkpoint available)

Protocol follows DCASE challenge standard:
    Train : normal/ clips from model id_00
    Test  : normal/ + abnormal/ clips from id_02, id_04, id_06
    Metric: AUROC per model ID, mean AUROC across IDs

Machine types: fan, pump, slider, valve
SNR levels:    -6dB (hard), 0dB (medium), +6dB (easy)

Usage:
    python eval_mimii.py --data ./mimii_data --machine fan
    python eval_mimii.py --data ./mimii_data --machine fan --student checkpoints/cardiac/student_best.pt
    python eval_mimii.py --data ./mimii_data --machine fan --sweep  # all model IDs
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile


# ── Audio loading ──────────────────────────────────────────────────────────────

TARGET_SR  = 16000
CLIP_LEN   = 16000   # 1 second @ 16kHz


def load_wav_mono(path: Path, target_sr: int = TARGET_SR) -> np.ndarray | None:
    """Load WAV as float32 mono, resample if needed."""
    try:
        sr, sig = wavfile.read(str(path))
        sig = sig.astype(np.float32)
        if sig.ndim > 1:
            sig = sig[:, 0]            # take channel 0 (8-channel mic array)
        if sr != target_sr:
            n_out = int(len(sig) * target_sr / sr)
            sig   = np.interp(
                np.linspace(0, len(sig) - 1, n_out),
                np.arange(len(sig)), sig
            ).astype(np.float32)
        peak = np.abs(sig).max()
        if peak > 1e-6:
            sig /= peak
        return sig
    except Exception:
        return None


def segment_signal(sig: np.ndarray, clip_len: int = CLIP_LEN) -> list[np.ndarray]:
    """Non-overlapping 1-second clips."""
    clips = []
    for start in range(0, len(sig) - clip_len + 1, clip_len):
        clips.append(sig[start:start + clip_len].copy())
    return clips


# ── Log-mel feature extraction ────────────────────────────────────────────────

def log_mel_features(clip: np.ndarray, sr: int = TARGET_SR,
                     n_mels: int = 64, n_fft: int = 1024,
                     hop: int = 512) -> np.ndarray:
    """
    Compute log-mel spectrogram and flatten to a 1-D feature vector.
    Pure numpy — no librosa needed.

    Output: (n_mels × n_frames,) flattened log-mel
    """
    # STFT
    frames = []
    win    = np.hanning(n_fft)
    for i in range(0, len(clip) - n_fft + 1, hop):
        frame = clip[i:i + n_fft] * win
        frames.append(np.abs(np.fft.rfft(frame)) ** 2)
    if not frames:
        return np.zeros(n_mels * 10, dtype=np.float32)
    S = np.stack(frames).T   # (n_fft//2+1, n_frames)

    # Mel filterbank
    freqs    = np.linspace(0, sr / 2, n_fft // 2 + 1)
    mel_min  = 2595 * np.log10(1 + 20 / 700)
    mel_max  = 2595 * np.log10(1 + (sr / 2) / 700)
    mel_pts  = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts   = 700 * (10 ** (mel_pts / 2595) - 1)
    bin_pts  = np.floor((n_fft + 1) * hz_pts / sr).astype(int)

    filterbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(n_mels):
        lo, ctr, hi = bin_pts[m], bin_pts[m + 1], bin_pts[m + 2]
        for k in range(lo, ctr):
            filterbank[m, k] = (k - lo) / max(ctr - lo, 1)
        for k in range(ctr, hi):
            filterbank[m, k] = (hi - k) / max(hi - ctr, 1)

    mel_S = filterbank @ S                        # (n_mels, n_frames)
    log_mel = np.log(mel_S + 1e-6)               # log scale

    # Summary statistics per mel band (mean + std) → 128-D
    feat = np.concatenate([log_mel.mean(axis=1),
                           log_mel.std(axis=1)]).astype(np.float32)
    return feat   # (n_mels * 2,) = 128-D


def extract_features_batch(wav_paths: list[Path],
                           verbose: bool = False) -> tuple[np.ndarray, list[str]]:
    """Extract 128-D log-mel features from a list of WAV files."""
    feats, names = [], []
    for i, p in enumerate(wav_paths):
        sig = load_wav_mono(p)
        if sig is None:
            continue
        # Use full signal (MIMII files are 10s), take middle 1s clip for speed
        mid   = len(sig) // 2
        clip  = sig[max(0, mid - CLIP_LEN // 2): max(0, mid - CLIP_LEN // 2) + CLIP_LEN]
        if len(clip) < CLIP_LEN:
            clip = np.pad(clip, (0, CLIP_LEN - len(clip)))
        feats.append(log_mel_features(clip))
        names.append(p.name)
        if verbose and (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(wav_paths)} files processed")
    return np.stack(feats) if feats else np.empty((0, 128)), names


# ── WavLM student features (optional) ─────────────────────────────────────────

def extract_student_features(wav_paths: list[Path],
                             student_path: Path,
                             device: str = "cpu") -> np.ndarray | None:
    """Extract features using the cardiac WavLM student (if available)."""
    try:
        import torch
        import torch.nn as nn

        class CardiacStudentEncoder(nn.Module):
            def __init__(self, hidden_dim: int = 768):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv1d(1, 32,  10, stride=5,  padding=2), nn.GELU(),
                    nn.Conv1d(32, 64,  3, stride=2,  padding=1), nn.GELU(),
                    nn.Conv1d(64, 128, 3, stride=2,  padding=1), nn.GELU(),
                    nn.Conv1d(128, 256, 3, stride=2, padding=1), nn.GELU(),
                    nn.AdaptiveAvgPool1d(1),
                )
                self.proj = nn.Sequential(nn.Linear(256, hidden_dim), nn.LayerNorm(hidden_dim))
            def forward(self, x):
                return self.proj(self.encoder(x).squeeze(-1))

        ckpt    = torch.load(student_path, map_location="cpu", weights_only=False)
        student = CardiacStudentEncoder(hidden_dim=ckpt.get("hidden_dim", 768))
        student.load_state_dict(ckpt["model_state_dict"], strict=True)
        student.eval()

        feats = []
        with torch.no_grad():
            for p in wav_paths:
                sig = load_wav_mono(p)
                if sig is None:
                    feats.append(np.zeros(768, dtype=np.float32))
                    continue
                mid  = len(sig) // 2
                clip = sig[max(0, mid - CLIP_LEN // 2): max(0, mid - CLIP_LEN // 2) + CLIP_LEN]
                if len(clip) < CLIP_LEN:
                    clip = np.pad(clip, (0, CLIP_LEN - len(clip)))
                x    = torch.from_numpy(clip).unsqueeze(0).unsqueeze(0)  # (1,1,16000)
                feats.append(student(x).squeeze(0).numpy())
        return np.stack(feats)
    except Exception as e:
        print(f"  ⚠️  Student features unavailable: {e}")
        return None


# ── PCA anomaly scoring ────────────────────────────────────────────────────────

def fit_pca(X: np.ndarray, k: int):
    mean = X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    return mean, Vt[:k]


def pca_score(X: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    c   = X - mean
    rec = (c @ components.T) @ components
    return np.linalg.norm(c - rec, axis=1)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    count = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return float(count / (len(pos) * len(neg)))


# ── Data loading ───────────────────────────────────────────────────────────────

def find_machine_data(data_dir: Path, machine: str, snr: str) -> dict[str, dict]:
    """
    Find all model ID directories for a given machine+SNR combination.
    Returns {model_id: {"normal": [paths], "abnormal": [paths]}}
    """
    snr_map  = {"min6": "-6_dB", "-6": "-6_dB", "0": "0_dB", "6": "6_dB"}
    snr_str  = snr_map.get(snr, snr)
    base     = data_dir / machine if (data_dir / machine).exists() else data_dir / f"{snr_str}_{machine}"

    if not base.exists():
        # Try alternate structure (extracted with full path)
        candidates = list(data_dir.rglob(f"*{snr_str}*{machine}*"))
        if candidates:
            base = candidates[0]
        else:
            raise FileNotFoundError(
                f"No data found for {snr_str}_{machine} in {data_dir}\n"
                f"Expected: {data_dir}/{snr_str}_{machine}/id_XX/normal/*.wav\n"
                f"Run: python download_mimii.py --machine {machine} --snr {snr}"
            )

    result = {}
    for id_dir in sorted(base.glob("id_*")):
        model_id = id_dir.name
        normal   = sorted((id_dir / "normal").glob("*.wav"))
        abnormal = sorted((id_dir / "abnormal").glob("*.wav"))
        if normal or abnormal:
            result[model_id] = {"normal": normal, "abnormal": abnormal}
    return result


# ── Main eval ──────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    print("\n" + "═" * 60)
    print("  MIMII Industrial Anomaly Eval — CORTEX-PE v16.15")
    print("═" * 60)
    print(f"  Data    : {args.data}")
    print(f"  Machine : {args.machine}")
    print(f"  SNR     : {args.snr}dB")
    print(f"  PCA k   : {args.n_components}")
    if args.student:
        print(f"  Student : {args.student}")

    data_dir = Path(args.data)
    model_data = find_machine_data(data_dir, args.machine, args.snr)

    model_ids = sorted(model_data.keys())
    print(f"\n  Model IDs found: {model_ids}")
    for mid, d in model_data.items():
        print(f"    {mid}: {len(d['normal'])} normal / {len(d['abnormal'])} abnormal")

    if len(model_ids) < 2:
        raise RuntimeError("Need at least 2 model IDs (one for train, rest for test)")

    # DCASE protocol: train on id_00 normal, test on remaining IDs
    train_id   = model_ids[0]
    test_ids   = model_ids[1:]
    print(f"\n  Protocol: train on {train_id}/normal → test on {test_ids}")

    # ── Feature extraction ───────────────────────────────────────────────
    print("\n── Extracting features ─────────────────────────────────────")

    use_student = args.student and Path(args.student).exists()
    feat_label  = "WavLM-student" if use_student else "log-mel (128-D)"
    print(f"  Feature type: {feat_label}")

    def get_feats(paths):
        if use_student:
            f = extract_student_features(paths, Path(args.student))
            if f is not None:
                return f
        f, _ = extract_features_batch(paths, verbose=False)
        return f

    t0 = time.perf_counter()

    train_paths = model_data[train_id]["normal"]
    print(f"  Train ({train_id} normal): {len(train_paths)} files...")
    train_feats = get_feats(train_paths)
    print(f"    → {train_feats.shape}")

    # Fit PCA on training normal sounds
    pca_mean, pca_comps = fit_pca(train_feats, min(args.n_components, train_feats.shape[1] - 1))

    # ── Evaluate per model ID ────────────────────────────────────────────
    print("\n── Per model ID evaluation ─────────────────────────────────")
    aurocs = []

    for test_id in test_ids:
        norm_paths  = model_data[test_id]["normal"]
        anom_paths  = model_data[test_id]["abnormal"]

        norm_feats  = get_feats(norm_paths)
        anom_feats  = get_feats(anom_paths)

        if len(norm_feats) == 0 or len(anom_feats) == 0:
            print(f"  {test_id}: insufficient data — skipping")
            continue

        all_feats  = np.concatenate([norm_feats, anom_feats])
        all_labels = np.array([0] * len(norm_feats) + [1] * len(anom_feats))
        scores     = pca_score(all_feats, pca_mean, pca_comps)
        auc        = auroc(scores, all_labels)
        aurocs.append(auc)

        norm_score = scores[:len(norm_feats)].mean()
        anom_score = scores[len(norm_feats):].mean()
        sep = anom_score / (norm_score + 1e-8)

        print(f"  {test_id}  AUROC={auc:.4f}  "
              f"norm_score={norm_score:.3f}  anom_score={anom_score:.3f}  "
              f"sep={sep:.2f}×")

    elapsed = time.perf_counter() - t0

    # ── Summary ──────────────────────────────────────────────────────────
    AUROC_TARGET = 0.70   # MIMII is harder than CWRU — factory noise
    mean_auroc   = float(np.mean(aurocs)) if aurocs else float("nan")
    auc_pass     = mean_auroc >= AUROC_TARGET

    print("\n── Summary ─────────────────────────────────────────────────")
    print(f"  Machine     : {args.machine}  SNR={args.snr}dB")
    print(f"  Features    : {feat_label}")
    print(f"  Mean AUROC  : {mean_auroc:.4f}  (target ≥ {AUROC_TARGET})  "
          f"{'✅ PASS' if auc_pass else '❌ FAIL'}")
    print(f"  Per-ID      : {[f'{a:.4f}' for a in aurocs]}")
    print(f"  Elapsed     : {elapsed:.1f}s")

    verdict = ("✅ MIMII ANOMALY DETECTION VALIDATED"
               if auc_pass else
               "❌ BELOW TARGET — try --student or different SNR")
    print(f"\n  {verdict}")
    print("═" * 60 + "\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIMII machine anomaly detection eval")
    parser.add_argument("--data",         required=True)
    parser.add_argument("--machine",      default="fan",
                        choices=["fan", "pump", "slider", "valve"])
    parser.add_argument("--snr",          default="0",
                        choices=["min6", "-6", "0", "6"])
    parser.add_argument("--n-components", type=int, default=8)
    parser.add_argument("--student",      default=None,
                        help="Optional: path to cardiac student .pt for richer features")
    args = parser.parse_args()
    run(args)
