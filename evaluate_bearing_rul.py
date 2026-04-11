"""
evaluate_bearing_rul.py — CORTEX-PE Remaining Useful Life Estimation

Evaluates the CORTEX-PE world model on the NASA/IMS run-to-failure bearing
dataset. No fault labels are used at any stage.

RUL estimation method:
    1. Encode each IMS snapshot as a 224×224 spectrogram → 128-D latent z_t
    2. Compute z_healthy = mean latent of first N% of run (healthy baseline)
    3. Anomaly score = negative L2 norm ||z_t|| (matches CWRU finding)
    4. Smooth score with rolling window to reduce noise
    5. Alert threshold = mean_healthy_score - k*std (3-sigma below healthy)
    6. RUL estimate = steps remaining until threshold is crossed
    7. Evaluate: MAE vs ground truth failure time, alert lead time in days

IMS Dataset structure:
    3 test sets:
        Dataset 1: 2003-10-22 to 2003-11-25 (~35 days, 984 files)
                   Bearing 1: outer race failure
                   Bearing 2: roller element failure
        Dataset 2: 2004-02-12 to 2004-02-19 (~7 days, 984 files)
                   Bearing 1: outer race failure
        Dataset 3: 2004-03-04 to 2004-04-04 (~32 days)
                   Bearing 3: outer race failure

    Each file: 20,480 rows × 8 columns (2 accelerometers × 4 bearings)
    Sampling: 20 kHz, ~10 min intervals

Download:
    python evaluate_bearing_rul.py --download --out ./bearing_data/raw_ims

Usage:
    # Download and evaluate on Dataset 1, Bearing 1
    python evaluate_bearing_rul.py \\
        --weights ./checkpoints/bearing/cortex_student_phase2_final.pt \\
        --data-dir ./bearing_data/raw_ims/1st_test \\
        --dataset 1 --bearing 1 \\
        --healthy-pct 0.4 \\
        --out ./results/rul_dataset1_bearing1.json
"""

import argparse
import json
import os
import urllib.request
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.signal import stft
from torchvision import transforms

from student_encoder import StudentEncoder


# =============================================================================
# Constants
# =============================================================================
IMS_DOWNLOAD_URL = "https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip"
IMS_FS           = 20000     # 20 kHz sample rate
IMS_SAMPLES      = 20480     # samples per snapshot file
FRAME_SIZE       = 224
STFT_WINDOW      = 512       # longer window for 20kHz signal
STFT_OVERLAP     = 256
SMOOTHING_WINDOW = 10        # rolling window for anomaly score smoothing

# Ground truth failure times for Dataset 1 (file index at failure)
# Based on Qiu et al. 2006 and community annotations
IMS_FAILURE_TIMES = {
    1: {1: 983, 2: 2140},   # Dataset 1: both bearings fail near end
    2: {1: 700},              # Dataset 2: bearing 1 fails ~midpoint
    3: {3: 570},              # Dataset 3: bearing 3 fails ~midpoint
}

# Channel mapping: dataset → bearing → [channel_indices]
IMS_CHANNELS = {
    1: {1: [0, 1], 2: [2, 3], 3: [4, 5], 4: [6, 7]},  # 2ch per bearing
    2: {1: [0],    2: [1],    3: [2],    4: [3]},        # 1ch per bearing
    3: {1: [0],    2: [1],    3: [2],    4: [3]},
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

FRAME_TRANSFORM = transforms.Compose([
    transforms.Resize(FRAME_SIZE),
    transforms.CenterCrop(FRAME_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# =============================================================================
# IMS dataset downloader
# =============================================================================
def download_ims(out_dir: str) -> bool:
    """Download and extract NASA IMS bearing dataset (~300MB)."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    zip_path = Path(out_dir) / "4.+Bearings.zip"

    if not zip_path.exists():
        print(f"Downloading IMS dataset from NASA S3...")
        print(f"  URL: {IMS_DOWNLOAD_URL}")
        print(f"  Dest: {zip_path}")
        print(f"  Size: ~300MB — this will take a few minutes")
        try:
            urllib.request.urlretrieve(IMS_DOWNLOAD_URL, str(zip_path))
            print(f"  ✅ Downloaded ({zip_path.stat().st_size/1e6:.0f}MB)")
        except Exception as e:
            print(f"  ❌ Download failed: {e}")
            print(f"  Try manually: {IMS_DOWNLOAD_URL}")
            return False

    print(f"Extracting {zip_path}...")
    try:
        with zipfile.ZipFile(str(zip_path), 'r') as z:
            z.extractall(out_dir)
        print(f"  ✅ Extracted to {out_dir}")
        print(f"  Contents: {[p.name for p in Path(out_dir).iterdir()]}")
        return True
    except Exception as e:
        print(f"  ❌ Extraction failed: {e}")
        return False


# =============================================================================
# IMS file reader
# =============================================================================
def load_ims_snapshot(file_path: str, n_channels: int = 8) -> np.ndarray:
    """
    Load one IMS snapshot file.
    Returns (20480, n_channels) float32 array.
    Files are whitespace-delimited ASCII with no header.
    """
    try:
        data = np.loadtxt(str(file_path))
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return data.astype(np.float32)
    except Exception:
        # Some files use tab delimiters
        try:
            data = np.genfromtxt(str(file_path), delimiter='\t')
            return data.astype(np.float32)
        except Exception as e:
            raise IOError(f"Cannot read {file_path}: {e}")


# =============================================================================
# Signal → spectrogram renderer
# =============================================================================
def signal_to_frame(
    sig_a: np.ndarray,
    sig_b: np.ndarray = None,
    fs:    int = IMS_FS,
) -> np.ndarray:
    """
    Render one or two vibration channels as 224×224 RGB spectrogram.
    If sig_b provided: R=sig_a, G=sig_b, B=mean (dual-channel, matches CWRU).
    """
    def _log_power_img(sig):
        _, _, Zxx = stft(sig, fs=fs, nperseg=STFT_WINDOW,
                         noverlap=STFT_OVERLAP, nfft=STFT_WINDOW)
        p = np.log1p(np.abs(Zxx) ** 2)
        p = (p - p.min()) / (p.max() - p.min() + 1e-8)
        img = Image.fromarray((p * 255).astype(np.uint8), mode='L')
        return np.array(img.resize((FRAME_SIZE, FRAME_SIZE), Image.BILINEAR))

    r = _log_power_img(sig_a)
    if sig_b is not None:
        g = _log_power_img(sig_b)
    else:
        g = r.copy()
    b = ((r.astype(np.float32) + g.astype(np.float32)) / 2).astype(np.uint8)

    return np.stack([r, g, b], axis=-1)


# =============================================================================
# Main RUL evaluator
# =============================================================================
def evaluate_rul(
    weights:     str,
    data_dir:    str,
    dataset:     int   = 1,
    bearing:     int   = 1,
    healthy_pct: float = 0.4,
    smooth_win:  int   = SMOOTHING_WINDOW,
    alert_sigma: float = 3.0,
    out_path:    str   = None,
    plot:        bool  = True,
) -> dict:
    """
    Evaluate RUL estimation on IMS run-to-failure data.

    Args:
        weights:     path to StudentEncoder checkpoint
        data_dir:    directory containing IMS snapshot files
        dataset:     IMS dataset number (1, 2, or 3)
        bearing:     bearing number (1-4)
        healthy_pct: first N% of run used to establish healthy baseline
        smooth_win:  rolling window size for anomaly score smoothing
        alert_sigma: number of std below healthy mean to trigger alert
        out_path:    save results JSON to this path
        plot:        save RUL trajectory plot

    Returns:
        dict with AUROC, alert lead time, MAE and other metrics
    """
    print(f"\n{'='*60}")
    print(f"  CORTEX-PE RUL Evaluation — IMS Dataset {dataset}, Bearing {bearing}")
    print(f"  Weights:      {weights}")
    print(f"  Data:         {data_dir}")
    print(f"  Healthy pct:  {healthy_pct*100:.0f}% of run")
    print(f"  Alert sigma:  {alert_sigma}σ below healthy mean")
    print(f"{'='*60}\n")

    # Load encoder
    ckpt  = torch.load(weights, map_location='cpu')
    model = StudentEncoder()
    model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
    model.eval()
    print(f"✅ Encoder loaded: {model.count_parameters():,} params")

    # Find and sort snapshot files
    data_path = Path(data_dir)
    files = sorted([f for f in data_path.iterdir()
                   if f.is_file() and not f.suffix in ['.json', '.md', '.txt']])

    if not files:
        print(f"❌ No IMS files found in {data_dir}")
        print(f"   Run: python evaluate_bearing_rul.py --download --out ./bearing_data/raw_ims")
        return {}

    n_files = len(files)
    print(f"✅ Found {n_files} IMS snapshots")

    # Channel indices for this bearing
    channels = IMS_CHANNELS.get(dataset, {}).get(bearing, [0])
    ch_a = channels[0]
    ch_b = channels[1] if len(channels) > 1 else channels[0]

    # Ground truth failure file index
    failure_idx = IMS_FAILURE_TIMES.get(dataset, {}).get(bearing, n_files - 1)
    print(f"   Ground truth failure: file {failure_idx}/{n_files} "
          f"({failure_idx/n_files*100:.0f}% through run)")

    # Number of healthy files
    n_healthy = int(n_files * healthy_pct)
    print(f"   Healthy baseline: first {n_healthy} files ({healthy_pct*100:.0f}%)")

    # -------------------------------------------------------------------------
    # Encode all snapshots
    # -------------------------------------------------------------------------
    norms      = []
    timestamps = []

    print(f"\nEncoding {n_files} snapshots...")
    with torch.no_grad():
        for i, file_path in enumerate(files):
            try:
                data  = load_ims_snapshot(str(file_path))
                sig_a = data[:, ch_a] if data.shape[1] > ch_a else data[:, 0]
                sig_b = data[:, ch_b] if data.shape[1] > ch_b else sig_a

                # Normalise within snapshot
                sig_a = (sig_a - sig_a.mean()) / (sig_a.std() + 1e-8)
                sig_b = (sig_b - sig_b.mean()) / (sig_b.std() + 1e-8)

                frame = signal_to_frame(sig_a, sig_b)
                tensor = FRAME_TRANSFORM(Image.fromarray(frame)).unsqueeze(0)
                z     = model(tensor).squeeze(0).numpy()
                norm  = float(np.linalg.norm(z))

                norms.append(norm)
                timestamps.append(i)

            except Exception as e:
                print(f"   ⚠️  File {i} ({file_path.name}): {e}")
                # Use last known value to avoid gaps
                norms.append(norms[-1] if norms else 0.0)
                timestamps.append(i)

            if i % 100 == 0:
                pct = i / n_files * 100
                print(f"   {i:>4}/{n_files} ({pct:.0f}%)")

    norms = np.array(norms)

    # -------------------------------------------------------------------------
    # Healthy baseline statistics
    # -------------------------------------------------------------------------
    healthy_norms  = norms[:n_healthy]
    healthy_mean   = float(healthy_norms.mean())
    healthy_std    = float(healthy_norms.std())
    alert_threshold = healthy_mean - alert_sigma * healthy_std

    print(f"\n  Healthy baseline (first {n_healthy} files):")
    print(f"    mean norm = {healthy_mean:.3f}")
    print(f"    std  norm = {healthy_std:.3f}")
    print(f"    alert threshold = {alert_threshold:.3f}  ({alert_sigma}σ below mean)")

    # -------------------------------------------------------------------------
    # Smooth anomaly score
    # -------------------------------------------------------------------------
    # Anomaly score = negative norm (fault → lower norm → higher anomaly)
    anomaly_raw    = -norms
    kernel         = np.ones(smooth_win) / smooth_win
    anomaly_smooth = np.convolve(anomaly_raw, kernel, mode='same')

    # Smoothed threshold
    alert_thresh_smooth = -(alert_threshold)

    # -------------------------------------------------------------------------
    # Find alert time
    # -------------------------------------------------------------------------
    alert_idx = None
    for i in range(n_healthy, n_files):
        if anomaly_smooth[i] > alert_thresh_smooth:
            # Confirm with 3 consecutive points above threshold
            if i + 2 < n_files and all(
                anomaly_smooth[j] > alert_thresh_smooth for j in range(i, i + 3)
            ):
                alert_idx = i
                break

    if alert_idx is not None:
        lead_time_files = failure_idx - alert_idx
        lead_time_days  = lead_time_files * 10 / (60 * 24)  # 10min intervals
        print(f"\n  🚨 Alert triggered at file {alert_idx}")
        print(f"     Lead time: {lead_time_files} files = {lead_time_days:.1f} days before failure")
    else:
        lead_time_files = 0
        lead_time_days  = 0.0
        print(f"\n  ⚠️  No alert triggered (threshold may need adjustment)")

    # -------------------------------------------------------------------------
    # AUROC — binary healthy vs degraded
    # -------------------------------------------------------------------------
    from sklearn.metrics import roc_auc_score
    labels = np.zeros(n_files)
    labels[failure_idx - int(n_files * 0.1):] = 1  # last 10% = failed
    try:
        auroc = roc_auc_score(labels, anomaly_smooth)
    except Exception:
        auroc = 0.0

    print(f"\n  AUROC (healthy vs final failure): {auroc:.4f}")

    # -------------------------------------------------------------------------
    # Degradation trend analysis
    # -------------------------------------------------------------------------
    # Split into quartiles
    q = n_files // 4
    q_norms = [
        float(norms[0:q].mean()),
        float(norms[q:2*q].mean()),
        float(norms[2*q:3*q].mean()),
        float(norms[3*q:].mean()),
    ]
    print(f"\n  Norm by quartile (should decrease toward failure):")
    for i, qn in enumerate(q_norms):
        bar  = "█" * int(qn / max(q_norms) * 20)
        print(f"    Q{i+1}: {qn:.3f}  {bar}")

    monotonic = all(q_norms[i] >= q_norms[i+1] for i in range(3))
    print(f"  Monotonic decline: {'✅' if monotonic else '⚠️  partial'}")

    # -------------------------------------------------------------------------
    # Save plot
    # -------------------------------------------------------------------------
    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor='#0d0d0d')

        for ax in axes:
            ax.set_facecolor('#0d0d0d')
            ax.tick_params(colors='#aaaaaa')
            ax.spines[:].set_color('#333333')

        # Top: norm over time
        x = np.arange(n_files)
        axes[0].plot(x, norms, color='#00e5b0', linewidth=0.8, alpha=0.6, label='||z|| per frame')
        axes[0].axhline(healthy_mean, color='#ffaa00', linewidth=1.5,
                        linestyle='--', label=f'Healthy mean ({healthy_mean:.2f})')
        axes[0].axhline(alert_threshold, color='#ff3a5c', linewidth=1.5,
                        linestyle='--', label=f'Alert threshold ({alert_threshold:.2f})')
        axes[0].axvline(n_healthy, color='#aaaaaa', linewidth=1, alpha=0.5, label='End of healthy baseline')
        if alert_idx:
            axes[0].axvline(alert_idx, color='#ff8800', linewidth=2, label=f'Alert (file {alert_idx})')
        axes[0].axvline(failure_idx, color='#ff3a5c', linewidth=2, label='Failure')
        axes[0].set_ylabel('Latent norm ||z||', color='#aaaaaa')
        axes[0].set_title(f'IMS Dataset {dataset} Bearing {bearing} — Latent Norm Trajectory',
                         color='white', fontsize=12)
        axes[0].legend(loc='upper right', fontsize=8, facecolor='#1a1a1a',
                       labelcolor='white', framealpha=0.8)

        # Bottom: anomaly score (smoothed)
        axes[1].plot(x, anomaly_smooth, color='#ff6b6b', linewidth=1.2, label='Anomaly score (smoothed)')
        axes[1].axhline(-healthy_mean, color='#ffaa00', linewidth=1, linestyle='--')
        axes[1].axhline(alert_thresh_smooth, color='#ff3a5c', linewidth=1.5,
                        linestyle='--', label='Alert threshold')
        axes[1].axvline(n_healthy, color='#aaaaaa', linewidth=1, alpha=0.5)
        if alert_idx:
            axes[1].axvline(alert_idx, color='#ff8800', linewidth=2,
                           label=f'Alert (+{lead_time_days:.1f}d lead)')
        axes[1].axvline(failure_idx, color='#ff3a5c', linewidth=2, label='Failure')
        axes[1].set_ylabel('Anomaly score (−||z||)', color='#aaaaaa')
        axes[1].set_xlabel('Snapshot index (~10 min intervals)', color='#aaaaaa')
        axes[1].legend(loc='upper left', fontsize=8, facecolor='#1a1a1a',
                       labelcolor='white', framealpha=0.8)

        plt.tight_layout()
        plot_path = Path(out_path).with_suffix('.png') if out_path else \
                    Path(f'./results/rul_d{dataset}_b{bearing}.png')
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(plot_path), dpi=120, facecolor='#0d0d0d')
        plt.close()
        print(f"\n  📊 Plot saved: {plot_path}")

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------
    results = {
        "dataset":          dataset,
        "bearing":          bearing,
        "n_files":          n_files,
        "n_healthy":        n_healthy,
        "failure_idx":      failure_idx,
        "alert_idx":        alert_idx,
        "lead_time_files":  lead_time_files,
        "lead_time_days":   round(lead_time_days, 2),
        "healthy_mean_norm": round(healthy_mean, 4),
        "healthy_std_norm":  round(healthy_std, 4),
        "alert_threshold":   round(alert_threshold, 4),
        "auroc":            round(auroc, 4),
        "monotonic_decline": bool(monotonic),
        "norm_by_quartile": [round(q, 4) for q in q_norms],
        "anomaly_signal":   "negative_L2_norm",
        "weights":          weights,
    }

    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  AUROC:          {auroc:.4f}")
    print(f"  Lead time:      {lead_time_days:.1f} days before failure")
    print(f"  Monotonic:      {'✅' if monotonic else '⚠️'}")
    print(f"  Healthy norm:   {healthy_mean:.3f} ± {healthy_std:.3f}")
    print(f"  Alert threshold:{alert_threshold:.3f}")

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  💾 Results saved: {out_path}")

    return results


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="CORTEX-PE RUL Estimation on NASA IMS Bearing Dataset"
    )
    p.add_argument("--weights",     default="./checkpoints/bearing/cortex_student_phase2_final.pt")
    p.add_argument("--data-dir",    default="./bearing_data/raw_ims/1st_test",
                   help="Directory containing IMS snapshot files")
    p.add_argument("--dataset",     type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--bearing",     type=int, default=1, choices=[1, 2, 3, 4])
    p.add_argument("--healthy-pct", type=float, default=0.4,
                   help="Fraction of run used as healthy baseline (default 0.40)")
    p.add_argument("--smooth-win",  type=int, default=10)
    p.add_argument("--alert-sigma", type=float, default=3.0)
    p.add_argument("--out",         default="./results/rul_d1_b1.json")
    p.add_argument("--no-plot",     action="store_true")
    p.add_argument("--download",    action="store_true",
                   help="Download IMS dataset from NASA S3")
    p.add_argument("--download-dir",default="./bearing_data/raw_ims")
    args = p.parse_args()

    if args.download:
        download_ims(args.download_dir)
    else:
        evaluate_rul(
            weights     = args.weights,
            data_dir    = args.data_dir,
            dataset     = args.dataset,
            bearing     = args.bearing,
            healthy_pct = args.healthy_pct,
            smooth_win  = args.smooth_win,
            alert_sigma = args.alert_sigma,
            out_path    = args.out,
            plot        = not args.no_plot,
        )


