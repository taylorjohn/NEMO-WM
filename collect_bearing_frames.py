"""
collect_bearing_frames.py — CORTEX-Perception-Engine Bearing Frame Collector

Converts raw vibration signals from bearing health datasets into 224×224
spectrogram PNG frames compatible with the CORTEX-PE training pipeline.

Supports:
    CWRU  — Case Western Reserve University Bearing Dataset
            Fault types: Normal, Ball, InnerRace, OuterRace
            Fault sizes:  0.007", 0.014", 0.021", 0.028"
            Sample rate:  12kHz (fan end) / 48kHz (drive end)
            Download:     engineering.case.edu/bearingdatacenter

    IMS   — NASA/University of Cincinnati Run-to-Failure Dataset
            4 bearings run continuously to failure (~35 days)
            Used for Remaining Useful Life (RUL) estimation
            Download:     ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository

Signal → Spectrogram pipeline:
    Raw vibration (1-D) →  STFT (window=256, overlap=128)
                        →  log power spectrum
                        →  normalised to [0, 255]
                        →  rendered as 224×224 RGB PNG
                        →  3 channels: R=DE vibration, G=FE vibration, B=mean

Usage:
    # CWRU — healthy baseline frames (Phase 1 semantic data)
    python collect_bearing_frames.py --dataset cwru --split normal --out ./bearing_data/healthy

    # CWRU — all fault types (for evaluation / anomaly detection)
    python collect_bearing_frames.py --dataset cwru --split all --out ./bearing_data/cwru_all

    # CWRU — specific fault type and severity
    python collect_bearing_frames.py --dataset cwru --split inner_race --diameter 0.007 --out ./bearing_data/ir_007

    # IMS — sequential run-to-failure frames (Phase 2 temporal data)
    python collect_bearing_frames.py --dataset ims --bearing 1 --out ./bearing_data/ims_b1

    # Download CWRU automatically
    python collect_bearing_frames.py --dataset cwru --download --out ./bearing_data

Output:
    ./bearing_data/<split>/frame_000001.png    — 224×224 RGB spectrogram
    ./bearing_data/<split>/metadata.json       — labels, fault type, severity, timestamp
"""

import argparse
import json
import os
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.io import loadmat
from scipy.signal import stft


# =============================================================================
# Constants
# =============================================================================
FRAME_SIZE    = 224
STFT_WINDOW   = 256          # STFT window length (samples)
STFT_OVERLAP  = 128          # STFT overlap (samples)
STFT_NFFT     = 256          # FFT points
SEGMENT_LEN   = 4096         # Raw samples per frame (covers ~85ms @ 48kHz)
SEGMENT_HOP   = 2048         # Hop between consecutive frames (50% overlap)

# Colour map — matches CORTEX-PE visual style (dark background, spectral)
CMAP          = "magma"

# CWRU download base URL
CWRU_BASE_URL = "https://engineering.case.edu/sites/default/files/"

# CWRU file manifest — (filename, fault_type, fault_location, diameter_inch, load_hp)
CWRU_MANIFEST = [
    # Normal baseline (48kHz drive end)
    ("97.mat",   "normal",     "none",        0.000, 0),
    ("98.mat",   "normal",     "none",        0.000, 1),
    ("99.mat",   "normal",     "none",        0.000, 2),
    ("100.mat",  "normal",     "none",        0.000, 3),

    # Ball fault — 12kHz drive end
    ("105.mat",  "ball",       "drive_end",   0.007, 0),
    ("106.mat",  "ball",       "drive_end",   0.007, 1),
    ("107.mat",  "ball",       "drive_end",   0.007, 2),
    ("108.mat",  "ball",       "drive_end",   0.007, 3),
    ("169.mat",  "ball",       "drive_end",   0.014, 0),
    ("170.mat",  "ball",       "drive_end",   0.014, 1),
    ("171.mat",  "ball",       "drive_end",   0.014, 2),
    ("172.mat",  "ball",       "drive_end",   0.014, 3),
    ("209.mat",  "ball",       "drive_end",   0.021, 0),
    ("210.mat",  "ball",       "drive_end",   0.021, 1),
    ("211.mat",  "ball",       "drive_end",   0.021, 2),
    ("212.mat",  "ball",       "drive_end",   0.021, 3),

    # Inner race fault — 12kHz drive end
    ("109.mat",  "inner_race", "drive_end",   0.007, 0),
    ("110.mat",  "inner_race", "drive_end",   0.007, 1),
    ("111.mat",  "inner_race", "drive_end",   0.007, 2),
    ("112.mat",  "inner_race", "drive_end",   0.007, 3),
    ("174.mat",  "inner_race", "drive_end",   0.014, 0),
    ("175.mat",  "inner_race", "drive_end",   0.014, 1),
    ("176.mat",  "inner_race", "drive_end",   0.014, 2),
    ("177.mat",  "inner_race", "drive_end",   0.014, 3),
    ("213.mat",  "inner_race", "drive_end",   0.021, 0),
    ("214.mat",  "inner_race", "drive_end",   0.021, 1),
    ("215.mat",  "inner_race", "drive_end",   0.021, 2),
    ("217.mat",  "inner_race", "drive_end",   0.021, 3),

    # Outer race fault — 12kHz drive end (centred)
    ("130.mat",  "outer_race", "drive_end",   0.007, 0),
    ("131.mat",  "outer_race", "drive_end",   0.007, 1),
    ("132.mat",  "outer_race", "drive_end",   0.007, 2),
    ("133.mat",  "outer_race", "drive_end",   0.007, 3),
    ("197.mat",  "outer_race", "drive_end",   0.014, 0),
    ("198.mat",  "outer_race", "drive_end",   0.014, 1),
    ("199.mat",  "outer_race", "drive_end",   0.014, 2),
    ("200.mat",  "outer_race", "drive_end",   0.014, 3),
    ("234.mat",  "outer_race", "drive_end",   0.021, 0),
    ("235.mat",  "outer_race", "drive_end",   0.021, 1),
    ("236.mat",  "outer_race", "drive_end",   0.021, 2),
    ("237.mat",  "outer_race", "drive_end",   0.021, 3),
]


# =============================================================================
# STFT → Spectrogram renderer
# =============================================================================
def signal_to_spectrogram(
    signal: np.ndarray,
    fs: int         = 12000,
    n_window: int   = STFT_WINDOW,
    n_overlap: int  = STFT_OVERLAP,
    n_fft: int      = STFT_NFFT,
    frame_size: int = FRAME_SIZE,
) -> np.ndarray:
    """
    Convert a 1-D vibration signal segment to a 224×224 uint8 RGB spectrogram.

    Uses log-power STFT with magma colormap.
    Frequency axis: 0 to fs/2. Time axis: segment duration.

    Args:
        signal:     (N,) float32 vibration samples
        fs:         sample rate in Hz
        frame_size: output image size (square)

    Returns:
        (224, 224, 3) uint8 RGB array
    """
    # STFT
    freqs, times, Zxx = stft(
        signal,
        fs      = fs,
        nperseg = n_window,
        noverlap= n_overlap,
        nfft    = n_fft,
    )

    # Log power spectrum
    power = np.abs(Zxx) ** 2
    power = np.log1p(power)

    # Normalise to [0, 1]
    p_min, p_max = power.min(), power.max()
    if p_max > p_min:
        power = (power - p_min) / (p_max - p_min)
    else:
        power = np.zeros_like(power)

    # Render via matplotlib
    fig, ax = plt.subplots(
        figsize    = (frame_size / 100, frame_size / 100),
        dpi        = 100,
        facecolor  = "#0d0d0d",
    )
    ax.set_facecolor("#0d0d0d")
    ax.pcolormesh(times, freqs, power, cmap=CMAP, shading="gouraud", vmin=0, vmax=1)
    ax.axis("off")
    fig.tight_layout(pad=0)

    fig.canvas.draw()
    buf   = fig.canvas.buffer_rgba()
    image = np.frombuffer(buf, dtype=np.uint8).reshape(frame_size, frame_size, 4)
    plt.close(fig)

    return image[:, :, :3]   # RGB, drop alpha


def dual_channel_spectrogram(
    de_signal:  np.ndarray,
    fe_signal:  np.ndarray,
    fs:         int = 12000,
    frame_size: int = FRAME_SIZE,
) -> np.ndarray:
    """
    Render drive-end and fan-end vibration as a 3-channel spectrogram:
        R = drive end log power
        G = fan end log power
        B = (DE + FE) / 2 combined

    This directly applies the multi-sensor fusion finding from the literature
    (accuracy improvements of 5.68-10.11% over single-sensor, Springer 2025).
    """
    def _log_power(sig):
        _, _, Zxx = stft(sig, fs=fs, nperseg=STFT_WINDOW,
                         noverlap=STFT_OVERLAP, nfft=STFT_NFFT)
        p = np.log1p(np.abs(Zxx) ** 2)
        p_min, p_max = p.min(), p.max()
        return ((p - p_min) / (p_max - p_min + 1e-8) * 255).astype(np.uint8) \
               if p_max > p_min else np.zeros_like(p, dtype=np.uint8)

    r = _log_power(de_signal)
    g = _log_power(fe_signal)
    b = ((r.astype(np.float32) + g.astype(np.float32)) / 2).astype(np.uint8)

    # Resize each channel to (frame_size, frame_size) and stack
    def _resize(arr):
        img = Image.fromarray(arr, mode='L')
        return np.array(img.resize((frame_size, frame_size), Image.BILINEAR))

    rgb = np.stack([_resize(r), _resize(g), _resize(b)], axis=-1)
    return rgb


# =============================================================================
# CWRU .mat file reader
# =============================================================================
def read_cwru_mat(mat_path: str) -> dict:
    """
    Read a CWRU .mat file and extract DE/FE/BA time series.

    Returns dict with keys:
        de_time:  (N,) float32 drive-end vibration (always present)
        fe_time:  (N,) float32 fan-end vibration   (usually present)
        ba_time:  (N,) float32 base acceleration   (sometimes present)
        rpm:      float  motor speed
    """
    data    = loadmat(mat_path)
    result  = {}

    for key in data:
        if key.startswith('_'):
            continue
        key_lower = key.lower()
        if 'de_time' in key_lower:
            result['de_time'] = data[key].squeeze().astype(np.float32)
        elif 'fe_time' in key_lower:
            result['fe_time'] = data[key].squeeze().astype(np.float32)
        elif 'ba_time' in key_lower:
            result['ba_time'] = data[key].squeeze().astype(np.float32)
        elif 'rpm' in key_lower:
            result['rpm'] = float(data[key].squeeze())

    if 'de_time' not in result:
        # Try to find any time-series key
        for key in data:
            if not key.startswith('_') and hasattr(data[key], 'squeeze'):
                arr = data[key].squeeze()
                if arr.ndim == 1 and len(arr) > 1000:
                    result['de_time'] = arr.astype(np.float32)
                    break

    return result


# =============================================================================
# CWRU frame collection
# =============================================================================
def collect_cwru_frames(
    data_dir:   str,
    out_dir:    str,
    split:      str   = "all",
    diameter:   float = None,
    fs:         int   = 12000,
    max_frames: int   = 2000,
    dual_channel: bool = True,
) -> list:
    """
    Render CWRU .mat files as sequential spectrogram frames.

    Args:
        data_dir:     directory containing CWRU .mat files
        out_dir:      output directory for PNG frames
        split:        "normal" | "ball" | "inner_race" | "outer_race" | "all"
        diameter:     filter by fault diameter (0.007, 0.014, 0.021) or None for all
        fs:           sample rate (12000 or 48000)
        max_frames:   maximum frames per file
        dual_channel: if True, use DE+FE dual-channel RGB spectrogram

    Returns:
        metadata list of dicts
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    metadata = []
    frame_idx = 0

    # Filter manifest
    files = CWRU_MANIFEST
    if split != "all":
        files = [f for f in files if f[1] == split]
    if diameter is not None:
        files = [f for f in files if abs(f[4] - diameter) < 1e-4 or f[4] == 0.0]

    print(f"\n{'='*60}")
    print(f"  CWRU Frame Collection")
    print(f"  Split: {split}  |  Diameter: {diameter or 'all'}")
    print(f"  Files: {len(files)}  |  Max frames: {max_frames}")
    print(f"  Dual channel: {dual_channel}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}\n")

    for filename, fault_type, location, diam, load in files:
        mat_path = Path(data_dir) / filename
        if not mat_path.exists():
            print(f"   ⚠️  {filename} not found — skipping")
            continue

        print(f"   Loading {filename} ({fault_type}, Ø{diam}\", load={load}hp)...")

        try:
            signals = read_cwru_mat(str(mat_path))
        except Exception as e:
            print(f"   ❌ Failed to read {filename}: {e}")
            continue

        de_sig = signals.get('de_time')
        fe_sig = signals.get('fe_time')

        if de_sig is None:
            print(f"   ⚠️  No DE signal in {filename} — skipping")
            continue

        n_samples   = len(de_sig)
        n_frames    = min(max_frames, (n_samples - SEGMENT_LEN) // SEGMENT_HOP)
        file_frames = 0

        for i in range(n_frames):
            start = i * SEGMENT_HOP
            end   = start + SEGMENT_LEN
            de_seg = de_sig[start:end]

            try:
                if dual_channel and fe_sig is not None and len(fe_sig) >= end:
                    fe_seg = fe_sig[start:end]
                    frame  = dual_channel_spectrogram(de_seg, fe_seg, fs=fs)
                else:
                    frame  = signal_to_spectrogram(de_seg, fs=fs)

                frame_name = f"frame_{frame_idx:06d}.png"
                Image.fromarray(frame).save(Path(out_dir) / frame_name)

                metadata.append({
                    "frame":      frame_name,
                    "source":     "cwru",
                    "file":       filename,
                    "fault_type": fault_type,
                    "location":   location,
                    "diameter":   diam,
                    "load_hp":    load,
                    "segment":    i,
                    "start":      int(start),
                    "fs":         fs,
                    "label":      0 if fault_type == "normal" else 1,
                })

                frame_idx  += 1
                file_frames += 1

            except Exception as e:
                print(f"   ⚠️  Frame {i} failed: {e}")
                continue

        print(f"   ✅ {file_frames} frames from {filename}")

    # Save metadata
    meta_path = Path(out_dir) / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    n_healthy = sum(1 for m in metadata if m["label"] == 0)
    n_fault   = sum(1 for m in metadata if m["label"] == 1)
    print(f"\n{'='*60}")
    print(f"  Total frames: {len(metadata)}")
    print(f"  Healthy:      {n_healthy}")
    print(f"  Fault:        {n_fault}")
    print(f"  Saved to:     {out_dir}")
    print(f"{'='*60}\n")

    return metadata


# =============================================================================
# IMS frame collection
# =============================================================================
def collect_ims_frames(
    data_dir:   str,
    out_dir:    str,
    bearing:    int  = 1,
    fs:         int  = 20000,
    max_frames: int  = 5000,
) -> list:
    """
    Render IMS run-to-failure data as ordered sequential frames.

    IMS dataset structure:
        Each file is a ~1 second snapshot (20,480 samples × 8 channels)
        Files are named as timestamps: YYYY.MM.DD.HH.mm.ss
        Channels: [B1Ch1, B1Ch2, B2Ch1, B2Ch2, B3Ch1, B3Ch2, B4Ch1, B4Ch2]
        bearing=1 → channels 0,1  |  bearing=2 → channels 2,3  etc.

    The temporal ordering of files is the key feature — Phase 2 training
    on IMS data teaches the encoder healthy bearing dynamics, enabling
    the predictor to detect deviation from the healthy trajectory.

    Args:
        data_dir:  directory containing IMS snapshot files
        out_dir:   output directory for PNG frames
        bearing:   which bearing to use (1-4)
        fs:        IMS sample rate (20kHz)
        max_frames: maximum total frames
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Find all snapshot files, sorted by timestamp
    all_files = sorted([
        f for f in Path(data_dir).iterdir()
        if f.is_file() and not f.suffix == '.json'
    ])

    if not all_files:
        print(f"❌ No IMS files found in {data_dir}")
        print("   Download from: ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository")
        return []

    ch_a = (bearing - 1) * 2      # primary channel index
    ch_b = ch_a + 1               # secondary channel index
    metadata = []
    frame_idx = 0
    n_files   = min(max_frames, len(all_files))

    print(f"\n{'='*60}")
    print(f"  IMS Frame Collection")
    print(f"  Bearing: {bearing}  |  Channels: {ch_a}, {ch_b}")
    print(f"  Files:   {len(all_files)} → using {n_files}")
    print(f"  Output:  {out_dir}")
    print(f"{'='*60}\n")

    for file_idx, snap_path in enumerate(all_files[:n_files]):
        try:
            # IMS files are plain text with whitespace-separated columns
            data = np.loadtxt(str(snap_path))
            if data.ndim == 1:
                data = data.reshape(-1, 1)
        except Exception:
            try:
                # Some versions are space-delimited with header
                data = np.genfromtxt(str(snap_path), skip_header=0)
            except Exception as e:
                print(f"   ⚠️  Failed to read {snap_path.name}: {e}")
                continue

        if data.shape[1] <= ch_b:
            print(f"   ⚠️  {snap_path.name}: only {data.shape[1]} channels, need {ch_b+1}")
            continue

        sig_a = data[:, ch_a].astype(np.float32)
        sig_b = data[:, ch_b].astype(np.float32)

        # Normalise within file (IMS has varying DC offset)
        sig_a = (sig_a - sig_a.mean()) / (sig_a.std() + 1e-8)
        sig_b = (sig_b - sig_b.mean()) / (sig_b.std() + 1e-8)

        try:
            frame = dual_channel_spectrogram(sig_a, sig_b, fs=fs)
            frame_name = f"frame_{frame_idx:06d}.png"
            Image.fromarray(frame).save(Path(out_dir) / frame_name)

            metadata.append({
                "frame":     frame_name,
                "source":    "ims",
                "file":      snap_path.name,
                "bearing":   bearing,
                "file_idx":  file_idx,
                "n_files":   len(all_files),
                "progress":  file_idx / len(all_files),   # 0.0=start, 1.0=failure
                "label":     0 if file_idx < len(all_files) * 0.6 else 1,
            })

            frame_idx += 1

        except Exception as e:
            print(f"   ⚠️  Frame {file_idx} failed: {e}")
            continue

        if file_idx % 100 == 0:
            pct = file_idx / len(all_files) * 100
            print(f"   {file_idx:>5}/{n_files} ({pct:.0f}%) — {snap_path.name}")

    # Save metadata
    meta_path = Path(out_dir) / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  ✅ {len(metadata)} frames saved to {out_dir}")
    return metadata


# =============================================================================
# CWRU auto-downloader
# =============================================================================
def download_cwru(out_dir: str, split: str = "normal") -> bool:
    """
    Download CWRU .mat files from engineering.case.edu.

    Args:
        out_dir: directory to save .mat files
        split:   "normal" | "ball" | "inner_race" | "outer_race" | "all"
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    files = CWRU_MANIFEST
    if split != "all":
        files = [f for f in files if f[1] == split or f[1] == "normal"]

    print(f"\nDownloading {len(files)} CWRU files to {out_dir}...")
    success = 0

    for filename, fault_type, *_ in files:
        dest = Path(out_dir) / filename
        if dest.exists():
            print(f"   ✅ {filename} already exists")
            success += 1
            continue
        url = CWRU_BASE_URL + filename
        try:
            print(f"   Downloading {filename} from {url}...")
            urllib.request.urlretrieve(url, str(dest))
            print(f"   ✅ {filename}")
            success += 1
        except Exception as e:
            print(f"   ❌ {filename} failed: {e}")

    print(f"\n  Downloaded {success}/{len(files)} files")
    return success > 0


# =============================================================================
# RUL diagnostic — analyse frame metadata for degradation trend
# =============================================================================
def analyse_rul_metadata(metadata_path: str) -> None:
    """
    Print a quick RUL analysis from collected IMS frame metadata.
    Shows the distribution of healthy vs fault frames over time.
    """
    with open(metadata_path) as f:
        meta = json.load(f)

    n          = len(meta)
    n_healthy  = sum(1 for m in meta if m.get("label") == 0)
    n_fault    = sum(1 for m in meta if m.get("label") == 1)
    progress   = [m.get("progress", 0) for m in meta]

    print(f"\n  IMS RUL Analysis")
    print(f"  Total frames: {n}")
    print(f"  Healthy (0-60%): {n_healthy}")
    print(f"  Fault   (60%+):  {n_fault}")
    print(f"  Progress range:  {min(progress):.2f} → {max(progress):.2f}")
    print(f"\n  Training recommendation:")
    print(f"  Phase 2: use frames with progress < 0.5 (healthy dynamics)")
    print(f"  Eval:    use all frames, measure ||z_t - z_healthy|| over time")


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="CORTEX-PE Bearing Spectrogram Frame Collector"
    )
    p.add_argument("--dataset",  choices=["cwru", "ims"], required=True)
    p.add_argument("--out",      required=True,
                   help="Output directory for PNG frames")
    p.add_argument("--data-dir", default="./bearing_data/raw",
                   help="Directory containing raw dataset files")
    p.add_argument("--split",    default="all",
                   choices=["normal", "ball", "inner_race", "outer_race", "all"],
                   help="CWRU: which fault type to collect")
    p.add_argument("--diameter", type=float, default=None,
                   choices=[0.007, 0.014, 0.021, 0.028],
                   help="CWRU: filter by fault diameter (inches)")
    p.add_argument("--bearing",  type=int, default=1, choices=[1, 2, 3, 4],
                   help="IMS: which bearing channel to use")
    p.add_argument("--fs",       type=int, default=12000,
                   help="Sample rate in Hz (CWRU: 12000 or 48000, IMS: 20000)")
    p.add_argument("--max-frames", type=int, default=2000,
                   help="Maximum number of frames to generate per file")
    p.add_argument("--single-channel", action="store_true",
                   help="Use single-channel spectrogram (default: dual DE+FE)")
    p.add_argument("--download", action="store_true",
                   help="CWRU: auto-download .mat files from CWRU website")
    p.add_argument("--analyse",  action="store_true",
                   help="Print RUL analysis from existing metadata.json")
    args = p.parse_args()

    if args.analyse:
        meta_path = Path(args.out) / "metadata.json"
        if meta_path.exists():
            analyse_rul_metadata(str(meta_path))
        else:
            print(f"❌ No metadata.json found at {args.out}")

    elif args.dataset == "cwru":
        if args.download:
            download_cwru(args.data_dir, args.split)

        collect_cwru_frames(
            data_dir     = args.data_dir,
            out_dir      = args.out,
            split        = args.split,
            diameter     = args.diameter,
            fs           = args.fs,
            max_frames   = args.max_frames,
            dual_channel = not args.single_channel,
        )

    elif args.dataset == "ims":
        collect_ims_frames(
            data_dir  = args.data_dir,
            out_dir   = args.out,
            bearing   = args.bearing,
            fs        = args.fs if args.fs != 12000 else 20000,
            max_frames= args.max_frames,
        )
