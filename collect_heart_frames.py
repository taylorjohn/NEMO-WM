"""
collect_heart_frames.py — CORTEX-PE Heart Sound Frame Collector

Converts PhysioNet CirCor DigiScope phonocardiogram (PCG) recordings to
224×224 mel-spectrogram frames compatible with the CORTEX-PE training pipeline.

Dataset: CirCor DigiScope Phonocardiogram Dataset (PhysioNet 2022 Challenge)
    - 5,282 PCG recordings from 1,568 patients (0.1–356 months)
    - 4 auscultation locations per patient: PV, TV, AV, MV
    - Labels: Normal / Murmur Present / Murmur Absent / Unknown
    - Sampling rate: 4,000 Hz (.wav format)
    - Download: https://physionet.org/content/circor-heart-sound/1.0.3/

Signal → Spectrogram pipeline:
    PCG .wav (4kHz) → 2-second windows (50% overlap)
                    → 128-band mel spectrogram
                    → log power
                    → normalised to [0, 255]
                    → 224×224 RGB PNG
                    → R=mel spec, G=harmonic component, B=percussive component

The harmonic/percussive decomposition captures S1/S2 heart sound structure
separately from murmur noise — providing a natural two-channel representation
analogous to the DE/FE dual-channel encoding used for bearings.

Usage:
    # Step 1 — Download CirCor dataset from PhysioNet (~2GB)
    python collect_heart_frames.py --download --out ./heart_data

    # Step 2 — Collect normal frames for Phase 1 training
    python collect_heart_frames.py --split normal --data-dir ./heart_data/training_data --out ./heart_data/normal_frames

    # Step 3 — Collect all frames for evaluation
    python collect_heart_frames.py --split all --data-dir ./heart_data/training_data --out ./heart_data/all_frames

    # Step 4 — Analyse dataset statistics
    python collect_heart_frames.py --analyse --out ./heart_data/all_frames

Output:
    ./heart_data/<split>/frame_000001.png   — 224×224 RGB mel-spectrogram
    ./heart_data/<split>/metadata.json      — patient ID, murmur label, location
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# =============================================================================
# Constants
# =============================================================================
FRAME_SIZE     = 224
PCG_FS         = 4000        # CirCor sampling rate (Hz)
WINDOW_SEC     = 2.0         # seconds per spectrogram window
WINDOW_SAMPLES = int(PCG_FS * WINDOW_SEC)   # 8000 samples
HOP_SEC        = 1.0         # 50% overlap
HOP_SAMPLES    = int(PCG_FS * HOP_SEC)
N_MELS         = 128
N_FFT          = 512
HOP_LENGTH     = 128         # STFT hop for mel spectrogram

# Auscultation locations in CirCor
LOCATIONS = ['PV', 'TV', 'AV', 'MV', 'Phc']

# Murmur label mapping
MURMUR_LABEL = {
    'Present': 1,
    'Absent':  0,
    'Unknown': -1,
}

# Download command (uses wget or curl)
PHYSIONET_BASE = "https://physionet.org/files/circor-heart-sound/1.0.3/"


# =============================================================================
# Downloader
# =============================================================================
def download_circor(out_dir: str) -> bool:
    """
    Download CirCor dataset from PhysioNet using wget.
    Requires no authentication for the training set.
    Total size: ~2GB
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading CirCor DigiScope dataset from PhysioNet...")
    print(f"  URL: {PHYSIONET_BASE}")
    print(f"  Dest: {out_dir}")
    print(f"  Size: ~2GB — this will take several minutes\n")

    # Try wget
    wget_cmd = [
        "wget", "-r", "-N", "-c", "-np",
        "--no-host-directories", "--cut-dirs=3",
        "-P", out_dir,
        PHYSIONET_BASE + "training_data/"
    ]
    try:
        result = subprocess.run(wget_cmd, check=True)
        print(f"\n✅ Download complete: {out_dir}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Try curl
    print("wget not available — trying curl...")
    try:
        csv_url = PHYSIONET_BASE + "training_data.csv"
        csv_dest = Path(out_dir) / "training_data.csv"
        urllib.request.urlretrieve(csv_url, str(csv_dest))
        print(f"  ✅ Downloaded training_data.csv")
        print(f"\n  For full dataset, run manually:")
        print(f"  wget -r -N -c -np {PHYSIONET_BASE}training_data/ -P {out_dir}")
        return True
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print(f"\nManual download instructions:")
        print(f"  1. Visit: https://physionet.org/content/circor-heart-sound/1.0.3/")
        print(f"  2. Click 'Download the ZIP file' or run:")
        print(f"     wget -r -N -c -np {PHYSIONET_BASE} -P {out_dir}")
        return False


# =============================================================================
# Mel spectrogram renderer
# =============================================================================
def pcg_to_mel_frame(
    audio:      np.ndarray,
    fs:         int = PCG_FS,
    n_mels:     int = N_MELS,
    n_fft:      int = N_FFT,
    hop_length: int = HOP_LENGTH,
    frame_size: int = FRAME_SIZE,
) -> np.ndarray:
    """
    Convert a PCG audio segment to a 224×224 RGB mel-spectrogram.

    Three channels:
        R = full mel spectrogram (all frequencies)
        G = harmonic component (S1/S2 tones, periodic)
        B = percussive component (murmur noise, transient)

    The harmonic/percussive split uses median filtering:
        harmonic    = median_filter(S, kernel=(1, L))   — horizontal (time)
        percussive  = median_filter(S, kernel=(L, 1))   — vertical (frequency)

    Args:
        audio:  (N,) float32 PCG samples, normalised to [-1, 1]
        fs:     sample rate in Hz

    Returns:
        (224, 224, 3) uint8 RGB array
    """
    from scipy.signal import stft
    from scipy.ndimage import median_filter

    # STFT → power spectrogram
    freqs, times, Zxx = stft(audio, fs=fs, nperseg=n_fft, noverlap=n_fft - hop_length)
    S = np.abs(Zxx) ** 2   # (freq_bins, time_frames)

    # Mel filterbank
    n_freqs = S.shape[0]
    mel_filters = _mel_filterbank(n_mels, n_freqs, fs, n_fft)
    S_mel = mel_filters @ S  # (n_mels, time_frames)

    # Log power
    S_log = np.log1p(S_mel)

    # Harmonic / percussive decomposition via median filtering
    L = 31  # median filter length
    S_h = median_filter(S_log, size=(1, L))   # horizontal — harmonic
    S_p = median_filter(S_log, size=(L, 1))   # vertical — percussive

    def _to_img(arr):
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        else:
            arr = np.zeros_like(arr)
        return arr  # keep as float [0,1]

    def _colormap(arr, cmap='magma'):
        import matplotlib.cm as cm
        colored = (cm.get_cmap(cmap)(arr)[:, :, :3] * 255).astype(np.uint8)
        img = Image.fromarray(colored)
        return np.array(img.resize((frame_size, frame_size), Image.BILINEAR))

    # Apply magma colormap — gives DINOv2 the color gradient structure it expects
    r = _colormap(_to_img(S_log), 'magma')[:,:,0]
    g = _colormap(_to_img(S_h),   'plasma')[:,:,0]
    b = _colormap(_to_img(S_p),   'inferno')[:,:,0]

    return np.stack([r, g, b], axis=-1)


def _mel_filterbank(n_mels: int, n_freqs: int, fs: int, n_fft: int) -> np.ndarray:
    """Build triangular mel filterbank matrix (n_mels, n_freqs)."""
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)
    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    freq_bins = np.linspace(0, fs / 2, n_freqs)
    mel_min   = hz_to_mel(0)
    mel_max   = hz_to_mel(fs / 2)
    mel_pts   = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts    = mel_to_hz(mel_pts)

    filters = np.zeros((n_mels, n_freqs))
    for m in range(n_mels):
        f_left   = hz_pts[m]
        f_center = hz_pts[m + 1]
        f_right  = hz_pts[m + 2]
        for k, f in enumerate(freq_bins):
            if f_left <= f <= f_center:
                filters[m, k] = (f - f_left) / (f_center - f_left + 1e-8)
            elif f_center < f <= f_right:
                filters[m, k] = (f_right - f) / (f_right - f_center + 1e-8)
    return filters


# =============================================================================
# CirCor dataset reader
# =============================================================================
def load_circor_metadata(data_dir: str) -> list:
    """
    Load patient metadata from training_data.csv.
    Returns list of dicts with patient_id, murmur, outcome, recordings.
    """
    csv_path = Path(data_dir).parent / "training_data.csv"
    if not csv_path.exists():
        # Try in same directory
        csv_path = Path(data_dir) / "training_data.csv"
    if not csv_path.exists():
        print(f"⚠️  training_data.csv not found — scanning directory for .wav files")
        return []

    patients = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            patients.append({
                'patient_id':   row.get('Patient ID', row.get('patient_id', '')),
                'murmur':       row.get('Murmur', 'Unknown'),
                'outcome':      row.get('Outcome', 'Unknown'),
                'age_group':    row.get('Age', ''),
                'sex':          row.get('Sex', ''),
                'locations':    row.get('Recording locations:', '').split('+'),
            })
    print(f"✅ Loaded metadata: {len(patients)} patients")
    return patients


def find_wav_files(data_dir: str, patient_id: str) -> list:
    """Find all .wav files for a patient in the data directory."""
    data_path = Path(data_dir)
    wavs = []
    for loc in LOCATIONS:
        pattern = f"{patient_id}_{loc}*.wav"
        wavs.extend(data_path.glob(pattern))
    if not wavs:
        # Try direct glob
        wavs = list(data_path.glob(f"{patient_id}*.wav"))
    return sorted(wavs)


# =============================================================================
# Main frame collector
# =============================================================================
def collect_heart_frames(
    data_dir:   str,
    out_dir:    str,
    split:      str   = "normal",
    max_frames: int   = 500,
    min_frames: int   = 3,
) -> list:
    """
    Render CirCor PCG recordings as mel-spectrogram frames.

    Args:
        data_dir:   directory containing .wav files and training_data.csv
        out_dir:    output directory for PNG frames
        split:      "normal"  — murmur=Absent patients only (for Phase 1)
                    "murmur"  — murmur=Present patients only (for evaluation)
                    "all"     — all patients
        max_frames: maximum total frames across all patients
        min_frames: minimum frames per patient to include

    Returns:
        list of metadata dicts
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Load metadata
    patients  = load_circor_metadata(data_dir)
    metadata  = []
    frame_idx = 0

    # Filter by split
    if split == "normal":
        target_patients = [p for p in patients if p['murmur'] == 'Absent']
        label = 0
    elif split == "murmur":
        target_patients = [p for p in patients if p['murmur'] == 'Present']
        label = 1
    else:
        target_patients = patients
        label = None

    # If no metadata, scan directory
    if not target_patients:
        print(f"⚠️  No metadata — scanning {data_dir} for .wav files")
        wav_files = sorted(Path(data_dir).glob("*.wav"))
        print(f"   Found {len(wav_files)} .wav files")
        target_patients = [{'patient_id': w.stem, 'murmur': 'Unknown',
                            'outcome': 'Unknown', 'age_group': '', 'sex': ''}
                           for w in wav_files[:200]]

    print(f"\n{'='*60}")
    print(f"  CirCor Heart Sound Frame Collection")
    print(f"  Split:      {split}  ({len(target_patients)} patients)")
    print(f"  Max frames: {max_frames}")
    print(f"  Output:     {out_dir}")
    print(f"{'='*60}\n")

    frames_per_patient = max(1, max_frames // max(len(target_patients), 1))

    for pat in target_patients:
        if frame_idx >= max_frames:
            break

        pid  = pat['patient_id']
        wavs = find_wav_files(data_dir, str(pid))

        if not wavs:
            continue

        pat_frames = 0

        for wav_path in wavs:
            if frame_idx >= max_frames or pat_frames >= frames_per_patient:
                break

            try:
                # Load WAV
                import wave, struct
                with wave.open(str(wav_path), 'rb') as wf:
                    n_channels  = wf.getnchannels()
                    sampwidth   = wf.getsampwidth()
                    fs          = wf.getframerate()
                    n_frames    = wf.getnframes()
                    raw         = wf.readframes(n_frames)

                # Decode to float32
                if sampwidth == 2:
                    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                    audio /= 32768.0
                elif sampwidth == 4:
                    audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
                    audio /= 2147483648.0
                else:
                    audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
                    audio = (audio - 128.0) / 128.0

                # Take first channel if stereo
                if n_channels > 1:
                    audio = audio[::n_channels]

                # Resample to PCG_FS if needed
                if fs != PCG_FS:
                    from scipy.signal import resample
                    n_out = int(len(audio) * PCG_FS / fs)
                    audio = resample(audio, n_out).astype(np.float32)
                    fs    = PCG_FS

                # Normalise
                audio = audio / (np.abs(audio).max() + 1e-8)

                # Extract location from filename
                loc = wav_path.stem.split('_')[1] if '_' in wav_path.stem else 'UNK'

                # Slide window
                n_windows = max(1, (len(audio) - WINDOW_SAMPLES) // HOP_SAMPLES + 1)

                for w in range(n_windows):
                    if frame_idx >= max_frames or pat_frames >= frames_per_patient:
                        break

                    start  = w * HOP_SAMPLES
                    end    = start + WINDOW_SAMPLES
                    if end > len(audio):
                        # Pad last window
                        segment = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
                        segment[:len(audio) - start] = audio[start:]
                    else:
                        segment = audio[start:end]

                    # Render frame
                    frame = pcg_to_mel_frame(segment, fs=fs)

                    # Save
                    frame_name = f"frame_{frame_idx:06d}.png"
                    Image.fromarray(frame).save(Path(out_dir) / frame_name)

                    murmur_val = MURMUR_LABEL.get(pat['murmur'], -1)
                    metadata.append({
                        "frame":      frame_name,
                        "patient_id": str(pid),
                        "murmur":     pat['murmur'],
                        "label":      murmur_val if label is None else label,
                        "outcome":    pat['outcome'],
                        "location":   loc,
                        "window":     w,
                        "age_group":  pat.get('age_group', ''),
                        "sex":        pat.get('sex', ''),
                        "wav_file":   wav_path.name,
                    })

                    frame_idx  += 1
                    pat_frames += 1

            except Exception as e:
                print(f"   ⚠️  {wav_path.name}: {e}")
                continue

        if frame_idx % 100 == 0 and frame_idx > 0:
            n_murmur  = sum(1 for m in metadata if m['label'] == 1)
            n_normal  = sum(1 for m in metadata if m['label'] == 0)
            print(f"   {frame_idx:>5} frames | normal={n_normal} murmur={n_murmur}")

    # Save metadata
    meta_path = Path(out_dir) / "metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    n_normal = sum(1 for m in metadata if m['label'] == 0)
    n_murmur = sum(1 for m in metadata if m['label'] == 1)

    print(f"\n{'='*60}")
    print(f"  Total frames: {len(metadata)}")
    print(f"  Normal:       {n_normal}")
    print(f"  Murmur:       {n_murmur}")
    print(f"  Saved to:     {out_dir}")
    print(f"{'='*60}\n")

    return metadata


# =============================================================================
# Dataset analysis
# =============================================================================
def analyse_dataset(out_dir: str) -> None:
    meta_path = Path(out_dir) / "metadata.json"
    if not meta_path.exists():
        print(f"❌ No metadata.json in {out_dir}")
        return

    meta = json.load(open(meta_path))
    n_normal  = sum(1 for m in meta if m['label'] == 0)
    n_murmur  = sum(1 for m in meta if m['label'] == 1)
    n_unknown = sum(1 for m in meta if m['label'] == -1)
    locations = {}
    for m in meta:
        loc = m.get('location', 'UNK')
        locations[loc] = locations.get(loc, 0) + 1

    print(f"\n  CirCor Dataset Analysis — {out_dir}")
    print(f"  Total frames:  {len(meta)}")
    print(f"  Normal:        {n_normal}  ({n_normal/len(meta)*100:.1f}%)")
    print(f"  Murmur:        {n_murmur}  ({n_murmur/len(meta)*100:.1f}%)")
    print(f"  Unknown:       {n_unknown}")
    print(f"\n  By location:")
    for loc, cnt in sorted(locations.items(), key=lambda x: -x[1]):
        print(f"    {loc:>5}: {cnt}")
    print(f"\n  Training recommendation:")
    print(f"    Phase 1: normal frames only ({n_normal} available)")
    print(f"    Eval:    all frames — AUROC(normal=0, murmur=1)")


# =============================================================================
# AUROC evaluator
# =============================================================================
def evaluate_heart(
    weights:   str,
    data_dir:  str,
    out_path:  str = "./results/heart_auroc.json",
) -> None:
    """
    Quick AUROC evaluation on CirCor frames using norm-based anomaly score.
    Reuses the same pipeline as bearing evaluation.
    """
    import torch
    from torchvision import transforms
    from student_encoder import StudentEncoder
    from sklearn.metrics import roc_auc_score

    T = transforms.Compose([
        transforms.Resize(224), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    ckpt  = torch.load(weights, map_location='cpu')
    model = StudentEncoder()
    model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
    model.eval()

    meta      = json.load(open(Path(data_dir) / "metadata.json"))
    frame_dir = Path(data_dir)
    results   = []

    print(f"Encoding {len(meta)} heart sound frames...")
    with torch.no_grad():
        for i, m in enumerate(meta):
            if m['label'] == -1:
                continue   # skip unknown
            img  = T(Image.open(frame_dir / m['frame'])).unsqueeze(0)
            z    = model(img).squeeze(0).numpy()
            norm = float(np.linalg.norm(z))
            results.append({
                'norm': norm, 'label': m['label'],
                'location': m.get('location', 'UNK'),
                'patient_id': m.get('patient_id', ''),
            })
            if i % 200 == 0:
                print(f"  {i}/{len(meta)}")

    labels = [r['label'] for r in results]
    norms  = [r['norm']  for r in results]

    healthy_mean = np.mean([r['norm'] for r in results if r['label'] == 0])
    murmur_mean  = np.mean([r['norm'] for r in results if r['label'] == 1])

    print(f"\n  Normal norm mean:  {healthy_mean:.3f}")
    print(f"  Murmur norm mean:  {murmur_mean:.3f}")

    # Test both directions — bearing showed fault has LOWER norm
    # For heart: murmur may have HIGHER or LOWER norm depending on encoding
    auroc_pos = roc_auc_score(labels, norms)
    auroc_neg = roc_auc_score(labels, [-n for n in norms])
    auroc = max(auroc_pos, auroc_neg)
    direction = "higher norm = murmur" if auroc_pos > auroc_neg else "lower norm = murmur"

    print(f"  AUROC: {auroc:.4f}  ({direction})")

    # Per location
    print(f"\n  By auscultation location:")
    for loc in ['PV', 'TV', 'AV', 'MV']:
        sub = [r for r in results if r['location'] == loc]
        if len(sub) < 10:
            continue
        l = [r['label'] for r in sub]
        n = [r['norm']  for r in sub]
        if len(set(l)) < 2:
            continue
        a = max(roc_auc_score(l, n), roc_auc_score(l, [-x for x in n]))
        mh = np.mean([r['norm'] for r in sub if r['label'] == 0])
        mm = np.mean([r['norm'] for r in sub if r['label'] == 1])
        print(f"    {loc}: AUROC={a:.4f}  normal={mh:.2f}  murmur={mm:.2f}")

    out = {
        'auroc': round(auroc, 4),
        'direction': direction,
        'normal_norm_mean': round(float(healthy_mean), 4),
        'murmur_norm_mean': round(float(murmur_mean), 4),
        'n_normal': sum(1 for r in results if r['label'] == 0),
        'n_murmur': sum(1 for r in results if r['label'] == 1),
        'weights': weights,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, 'w'), indent=2)
    print(f"\n  💾 Results: {out_path}")


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="CORTEX-PE CirCor Heart Sound Frame Collector")
    p.add_argument("--data-dir",  default="./heart_data/training_data")
    p.add_argument("--out",       default="./heart_data/normal_frames")
    p.add_argument("--split",     default="normal",
                   choices=["normal", "murmur", "all"])
    p.add_argument("--max-frames",type=int, default=2000)
    p.add_argument("--download",  action="store_true",
                   help="Download CirCor from PhysioNet")
    p.add_argument("--download-dir", default="./heart_data")
    p.add_argument("--analyse",   action="store_true")
    p.add_argument("--evaluate",  action="store_true",
                   help="Run AUROC evaluation on collected frames")
    p.add_argument("--weights",   default="./checkpoints/heart/cortex_student_phase2_final.pt")
    args = p.parse_args()

    if args.download:
        download_circor(args.download_dir)
    elif args.analyse:
        analyse_dataset(args.out)
    elif args.evaluate:
        evaluate_heart(args.weights, args.out)
    else:
        collect_heart_frames(
            data_dir  = args.data_dir,
            out_dir   = args.out,
            split     = args.split,
            max_frames= args.max_frames,
        )
