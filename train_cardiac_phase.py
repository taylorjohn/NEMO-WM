"""
train_cardiac_phase.py — CORTEX-PE Self-Supervised Cardiac Phase Encoder

Trains a CNN encoder to predict cardiac phase (S1 / systole / S2 / diastole)
from mel-spectrogram windows. Uses ONLY normal (Murmur Absent) patients.
No external teacher. No fault labels. Uses the free .tsv segmentation
annotations provided with the CirCor DigiScope dataset.

Anomaly detection at inference:
    Normal patient  → encoder predicts cardiac phase confidently → LOW entropy
    Murmur patient  → corrupted phase structure → HIGH prediction entropy
    Anomaly score   = entropy(softmax(phase_logits))

Why this works:
    The cardiac cycle has a highly regular structure: S1 → systole → S2 →
    diastole → repeat. Normal PCG recordings have crisp, identifiable S1/S2
    sounds. Heart murmurs add broadband turbulent noise during systole or
    diastole, blurring the phase boundaries. An encoder trained on normal
    cycles cannot confidently assign a phase to a murmur-corrupted window.

Architecture:
    StudentEncoder (128-D latent)
    → PhaseHead: Linear(128 → 64) → ReLU → Linear(64 → 4)
    → cross-entropy loss on {S1=0, systole=1, S2=2, diastole=3}

Dataset:
    CirCor DigiScope v1.0.3 — normal patients only
    .wav files: 4kHz, 4 auscultation locations (PV, TV, AV, MV)
    .tsv files: S1(1), systole(2), S2(3), diastole(4) per sample

Usage:
    # Train
    python train_cardiac_phase.py \\
        --data-dir ./heart_data/training_data \\
        --csv      ./heart_data/training_data.csv \\
        --out      ./checkpoints/heart_phase \\
        --epochs   30

    # Evaluate AUROC
    python train_cardiac_phase.py \\
        --evaluate \\
        --weights  ./checkpoints/heart_phase/cardiac_phase_final.pt \\
        --data-dir ./heart_data/training_data \\
        --csv      ./heart_data/training_data.csv
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.signal import stft
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from student_encoder import StudentEncoder

# =============================================================================
# Constants
# =============================================================================
PCG_FS         = 4000     # CirCor sample rate
WINDOW_SEC     = 0.5      # seconds per training window
WINDOW_SAMPLES = int(PCG_FS * WINDOW_SEC)   # 2000 samples
HOP_SEC        = 0.25     # 50% overlap → 4 windows/sec
HOP_SAMPLES    = int(PCG_FS * HOP_SEC)
FRAME_SIZE     = 224
N_FFT          = 256
HOP_LENGTH     = 64
N_MELS         = 64

# Phase labels (from .tsv col3)
PHASE_IGNORE   = 0   # unannotated
PHASE_S1       = 1
PHASE_SYSTOLE  = 2
PHASE_S2       = 3
PHASE_DIASTOLE = 4
N_PHASES       = 4   # S1, systole, S2, diastole (ignore unannotated)

PHASE_MAP = {1: 0, 2: 1, 3: 2, 4: 3}  # tsv label → class index
PHASE_NAMES = ['S1', 'Systole', 'S2', 'Diastole']

IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]

FRAME_TRANSFORM = transforms.Compose([
    transforms.Resize(FRAME_SIZE),
    transforms.CenterCrop(FRAME_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# =============================================================================
# Signal → mel spectrogram frame
# =============================================================================
def signal_to_frame(sig: np.ndarray, fs: int = PCG_FS) -> np.ndarray:
    """Convert PCG window to 224×224 RGB mel-spectrogram."""
    from scipy.ndimage import median_filter

    # STFT
    _, _, Zxx = stft(sig, fs=fs, nperseg=N_FFT,
                     noverlap=N_FFT - HOP_LENGTH, nfft=N_FFT)
    S = np.log1p(np.abs(Zxx) ** 2)

    # Harmonic / percussive decomposition
    S_h = median_filter(S, size=(1, 15))
    S_p = median_filter(S, size=(15, 1))

    def _to_img(arr):
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        img = Image.fromarray((arr * 255).astype(np.uint8), mode='L')
        return np.array(img.resize((FRAME_SIZE, FRAME_SIZE), Image.BILINEAR))

    r = _to_img(S)
    g = _to_img(S_h)
    b = _to_img(S_p)
    return np.stack([r, g, b], axis=-1)


# =============================================================================
# Dataset
# =============================================================================
class CardiacPhaseDataset(Dataset):
    """
    Extracts labelled cardiac phase windows from CirCor normal patient recordings.

    Each sample: (mel_frame_tensor, phase_class_int)
    Phase classes: 0=S1, 1=Systole, 2=S2, 3=Diastole
    Unannotated windows (tsv label=0) are skipped.

    Args:
        data_dir:    CirCor training_data/ directory
        csv_path:    training_data.csv path
        split:       'normal'  → Murmur Absent only (training)
                     'all'     → all patients (evaluation)
        max_per_patient: max windows per patient (balance classes)
    """
    def __init__(
        self,
        data_dir:         str,
        csv_path:         str,
        split:            str = 'normal',
        max_per_patient:  int = 50,
    ):
        self.data_dir = Path(data_dir)
        self.tf       = FRAME_TRANSFORM
        self.samples  = []  # (wav_path, tsv_path, start_sample, phase_class, patient_id, murmur)

        # Load patient metadata
        patients = {}
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row.get('Patient ID', row.get('patient_id', '')).strip()
                patients[pid] = {
                    'murmur':  row.get('Murmur', 'Unknown').strip(),
                    'outcome': row.get('Outcome', 'Unknown').strip(),
                }

        if split == 'normal':
            target_pids = {pid for pid, d in patients.items()
                           if d['murmur'] == 'Absent'}
        else:
            target_pids = set(patients.keys())

        print(f"\n  CardiacPhaseDataset ({split}):")
        print(f"    Patients: {len(target_pids)}")

        n_loaded = 0
        n_skipped = 0

        for pid in sorted(target_pids):
            murmur = patients.get(pid, {}).get('murmur', 'Unknown')

            # Find all .wav/.tsv pairs for this patient
            wav_files = sorted(self.data_dir.glob(f"{pid}_*.wav"))

            for wav_path in wav_files:
                tsv_path = wav_path.with_suffix('.tsv')
                if not tsv_path.exists():
                    n_skipped += 1
                    continue

                # Parse TSV annotations
                segments = []
                try:
                    with open(tsv_path) as f:
                        for line in f:
                            parts = line.strip().split('\t')
                            if len(parts) < 3:
                                continue
                            t_start = float(parts[0])
                            t_end   = float(parts[1])
                            label   = int(float(parts[2]))
                            if label == PHASE_IGNORE:
                                continue
                            segments.append((t_start, t_end, label))
                except Exception:
                    n_skipped += 1
                    continue

                if not segments:
                    continue

                # Load audio length only (don't keep in memory)
                try:
                    import wave
                    with wave.open(str(wav_path), 'rb') as wf:
                        n_frames = wf.getnframes()
                        n_ch     = wf.getnchannels()
                    total_samples = n_frames // n_ch
                except Exception:
                    continue

                # Generate windows and assign phase labels
                pat_samples = []
                for t_start, t_end, tsv_label in segments:
                    s_start = int(t_start * PCG_FS)
                    s_end   = int(t_end   * PCG_FS)
                    phase_cls = PHASE_MAP.get(tsv_label, -1)
                    if phase_cls < 0:
                        continue

                    # Slide 0.5s windows through this segment
                    pos = s_start
                    while pos + WINDOW_SAMPLES <= min(s_end, total_samples):
                        pat_samples.append((
                            wav_path, tsv_path, pos,
                            phase_cls, pid, murmur
                        ))
                        pos += HOP_SAMPLES

                # Balance per patient
                if len(pat_samples) > max_per_patient:
                    # Stratified sample across phases
                    by_phase = {i: [] for i in range(N_PHASES)}
                    for s in pat_samples:
                        by_phase[s[3]].append(s)
                    per_phase = max(1, max_per_patient // N_PHASES)
                    pat_samples = []
                    for i in range(N_PHASES):
                        ph_samples = by_phase[i]
                        if ph_samples:
                            idx = np.random.choice(
                                len(ph_samples),
                                min(len(ph_samples), per_phase),
                                replace=False
                            )
                            pat_samples.extend([ph_samples[j] for j in idx])

                self.samples.extend(pat_samples)
                n_loaded += len(pat_samples)

        # Class counts
        class_counts = np.zeros(N_PHASES, dtype=int)
        for s in self.samples:
            class_counts[s[3]] += 1

        print(f"    Windows loaded: {n_loaded}")
        print(f"    Skipped:        {n_skipped}")
        print(f"    Class balance:")
        for i, name in enumerate(PHASE_NAMES):
            print(f"      {name}: {class_counts[i]}")

        if n_loaded == 0:
            raise RuntimeError(
                "No windows loaded. Check --data-dir and --csv paths.\n"
                f"  data_dir: {data_dir}\n"
                f"  csv_path: {csv_path}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_window(
        self, wav_path: Path, start: int
    ) -> np.ndarray:
        """Load a WINDOW_SAMPLES segment from wav file."""
        import wave
        try:
            with wave.open(str(wav_path), 'rb') as wf:
                n_ch  = wf.getnchannels()
                sw    = wf.getsampwidth()
                wf.setpos(start * n_ch)
                raw   = wf.readframes(WINDOW_SAMPLES)

            if sw == 2:
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
                audio = (audio - 128.0) / 128.0

            if n_ch > 1:
                audio = audio[::n_ch]

            # Pad if short
            if len(audio) < WINDOW_SAMPLES:
                pad   = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
                pad[:len(audio)] = audio
                audio = pad

            audio = audio[:WINDOW_SAMPLES]
            audio = audio / (np.abs(audio).max() + 1e-8)
            return audio

        except Exception:
            return np.zeros(WINDOW_SAMPLES, dtype=np.float32)

    def __getitem__(self, idx: int):
        wav_path, _, start, phase_cls, pid, murmur = self.samples[idx]
        audio = self._load_window(wav_path, start)
        frame = signal_to_frame(audio)
        tensor = self.tf(Image.fromarray(frame))
        return tensor, phase_cls, murmur


# =============================================================================
# Phase head
# =============================================================================
class PhaseHead(nn.Module):
    def __init__(self, in_dim: int = 128, n_classes: int = N_PHASES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# =============================================================================
# Training
# =============================================================================
def train(
    data_dir:    str,
    csv_path:    str,
    out_dir:     str,
    epochs:      int   = 30,
    batch_size:  int   = 64,
    lr:          float = 1e-3,
    seed:        int   = 42,
):
    torch.manual_seed(seed)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*62}")
    print(f"  CARDIAC PHASE SELF-SUPERVISED TRAINING")
    print(f"  Data:    {data_dir}")
    print(f"  Epochs:  {epochs}")
    print(f"  Batch:   {batch_size}")
    print(f"{'='*62}")

    dataset = CardiacPhaseDataset(data_dir, csv_path,
                                  split='normal', max_per_patient=40)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         num_workers=0, drop_last=True)

    model     = StudentEncoder()
    head      = PhaseHead(in_dim=128, n_classes=N_PHASES)
    n_params  = sum(p.numel() for p in model.parameters()) + \
                sum(p.numel() for p in head.parameters())
    print(f"\n  Encoder + PhaseHead: {n_params:,} params")

    opt  = torch.optim.Adam(
        list(model.parameters()) + list(head.parameters()), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs * len(loader))
    crit = nn.CrossEntropyLoss()

    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train(); head.train()
        total_loss = 0.0
        correct    = 0
        total      = 0
        t0         = time.time()

        for frames, labels, _ in loader:
            labels = labels.long()
            z      = model(frames)
            logits = head(z)
            loss   = crit(logits, labels)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()), 1.0)
            opt.step()
            sched.step()

            total_loss += float(loss) * len(labels)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += len(labels)

        avg_loss = total_loss / total
        acc      = correct / total * 100
        elapsed  = time.time() - t0

        print(f"Epoch {epoch:>3}/{epochs} | "
              f"loss={avg_loss:.4f}  acc={acc:.1f}%  ({elapsed:.1f}s)")

        if acc > best_acc:
            best_acc = acc
            best_path = Path(out_dir) / "cardiac_phase_best.pt"
            torch.save({
                "encoder":   model.state_dict(),
                "head":      head.state_dict(),
                "epoch":     epoch,
                "acc":       acc,
                "n_phases":  N_PHASES,
            }, str(best_path))

        if epoch % 10 == 0:
            ckpt_path = Path(out_dir) / f"cardiac_phase_ep{epoch:03d}.pt"
            torch.save({
                "encoder": model.state_dict(),
                "head":    head.state_dict(),
                "epoch":   epoch,
            }, str(ckpt_path))
            print(f"  Checkpoint: {ckpt_path}")

    final_path = Path(out_dir) / "cardiac_phase_final.pt"
    torch.save({
        "encoder": model.state_dict(),
        "head":    head.state_dict(),
        "epoch":   epochs,
        "best_acc": best_acc,
    }, str(final_path))
    print(f"\n  Best acc: {best_acc:.1f}%")
    print(f"  Final:    {final_path}")
    return str(final_path)


# =============================================================================
# Evaluation
# =============================================================================
def evaluate(
    weights:  str,
    data_dir: str,
    csv_path: str,
    out_path: str = "./results/heart_phase_auroc.json",
):
    """
    Evaluate murmur detection AUROC using phase prediction entropy.
    Anomaly score = entropy(softmax(phase_logits)) averaged over
    all windows for a recording.
    """
    from sklearn.metrics import roc_auc_score

    ckpt  = torch.load(weights, map_location='cpu')
    model = StudentEncoder()
    head  = PhaseHead(in_dim=128, n_classes=N_PHASES)
    model.load_state_dict(ckpt['encoder'])
    head.load_state_dict(ckpt['head'])
    model.eval(); head.eval()
    print(f"\n  Loaded: {weights}")
    print(f"  Best train acc: {ckpt.get('best_acc', ckpt.get('acc', 'n/a')):.1f}%"
          if 'best_acc' in ckpt or 'acc' in ckpt else "")

    dataset = CardiacPhaseDataset(data_dir, csv_path,
                                  split='all', max_per_patient=20)
    loader  = DataLoader(dataset, batch_size=64, shuffle=False,
                         num_workers=0)

    # Collect per-window entropy and labels
    results = []  # (entropy, murmur_label, patient_id)

    print(f"\n  Evaluating {len(dataset)} windows...")
    with torch.no_grad():
        for frames, phase_labels, murmurs in loader:
            z       = model(frames)
            logits  = head(z)
            probs   = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)

            for i in range(len(frames)):
                murmur_label = 1 if murmurs[i] == 'Present' else \
                               0 if murmurs[i] == 'Absent'  else -1
                results.append({
                    'entropy':      float(entropy[i]),
                    'pred_phase':   int(probs[i].argmax()),
                    'true_phase':   int(phase_labels[i]),
                    'murmur_label': murmur_label,
                    'murmur_str':   murmurs[i],
                })

    # Filter to known labels
    known  = [r for r in results if r['murmur_label'] >= 0]
    labels = [r['murmur_label'] for r in known]
    scores = [r['entropy']      for r in known]

    n_normal = sum(1 for l in labels if l == 0)
    n_murmur = sum(1 for l in labels if l == 1)

    mean_ent_normal = np.mean([r['entropy'] for r in known if r['murmur_label']==0])
    mean_ent_murmur = np.mean([r['entropy'] for r in known if r['murmur_label']==1])

    print(f"\n  Normal entropy mean:  {mean_ent_normal:.4f}")
    print(f"  Murmur entropy mean:  {mean_ent_murmur:.4f}")
    print(f"  Direction: {'murmur=higher entropy ✅' if mean_ent_murmur > mean_ent_normal else 'murmur=lower entropy ⚠️'}")

    try:
        auroc = roc_auc_score(labels, scores)
        print(f"\n  AUROC: {auroc:.4f}")
    except Exception as e:
        print(f"  AUROC error: {e}")
        auroc = 0.0

    # Phase accuracy on normal patients
    normal_results = [r for r in results if r['murmur_label'] == 0]
    if normal_results:
        phase_acc = np.mean([r['pred_phase'] == r['true_phase']
                             for r in normal_results]) * 100
        print(f"  Phase acc (normal): {phase_acc:.1f}%")

    out = {
        'auroc':            round(auroc, 4),
        'normal_entropy':   round(float(mean_ent_normal), 4),
        'murmur_entropy':   round(float(mean_ent_murmur), 4),
        'n_normal_windows': n_normal,
        'n_murmur_windows': n_murmur,
        'method':           'phase_prediction_entropy',
        'weights':          weights,
    }
    Path(out_path).parent.mkdir(exist_ok=True)
    json.dump(out, open(out_path, 'w'), indent=2)
    print(f"\n  Results: {out_path}")
    return out


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="CORTEX-PE Self-Supervised Cardiac Phase Encoder")
    p.add_argument("--data-dir",  default="./heart_data/training_data")
    p.add_argument("--csv",       default="./heart_data/training_data.csv")
    p.add_argument("--out",       default="./checkpoints/heart_phase")
    p.add_argument("--epochs",    type=int,   default=30)
    p.add_argument("--batch",     type=int,   default=64)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--evaluate",  action="store_true")
    p.add_argument("--weights",   default="./checkpoints/heart_phase/cardiac_phase_final.pt")
    p.add_argument("--auroc-out", default="./results/heart_phase_auroc.json")
    args = p.parse_args()

    if args.evaluate:
        evaluate(args.weights, args.data_dir, args.csv, args.auroc_out)
    else:
        train(args.data_dir, args.csv, args.out,
              args.epochs, args.batch, args.lr)
