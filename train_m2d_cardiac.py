"""
train_m2d_cardiac.py
=====================
WavLM-Base teacher distillation for CirCor cardiac anomaly detection.
v2: Raw WAV input to WavLM teacher (fixes mel proxy degradation).

v1 (mel proxy): AUROC 0.6495
v2 (raw WAV):   Expected AUROC +0.03–0.05 improvement

Why M2D:
  - BEATs distillation loss was stuck at 0.0000 (broken pipeline)
  - M2D is specifically designed for general-purpose audio representation
  - Expected: AUROC 0.613 → 0.75-0.78 (+0.10-0.15)
  - M2D outputs 768-D patch embeddings → mean-pool → distill target

Architecture:
  Teacher:  m2d_vit_base-80x608p16x16 (frozen, CPU/GPU)
  Student:  1D-CNN StudentEncoder (32-D backbone, trains on CPU, NPU at inference)
  Loss:     cosine distillation + Weak-SIGReg K=32

Usage:
    # Install M2D teacher (first time only):
    pip install timm

    python train_m2d_cardiac.py \
        --data ./circor/train \
        --steps 6000 \
        --out ./checkpoints/cardiac_m2d \
        --lr 3e-4

    # Then evaluate:
    python train_m2d_cardiac.py --eval \
        --checkpoint ./checkpoints/cardiac_m2d/student_final.pt \
        --data ./circor/test
"""

import argparse
import math
import os
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ── Weak-SIGReg K=32 (same as Phase 1/2) ─────────────────────────────────

def sigreg_loss(z: torch.Tensor, K: int = 32, seed: int = 42) -> torch.Tensor:
    D = z.shape[1]
    torch.manual_seed(seed)
    S = torch.randn(D, K, device=z.device) / (K ** 0.5)
    S = S / S.norm(dim=0, keepdim=True)
    sk = z @ S
    sk_c = sk - sk.mean(0)
    cov = (sk_c.T @ sk_c) / (z.shape[0] - 1)
    return (cov - torch.eye(K, device=z.device)).pow(2).sum() / K


# ── Audio StudentEncoder (1D-CNN for mel-spectrogram input) ───────────────

class AudioStudentEncoder(nn.Module):
    """
    Lightweight 1D-CNN StudentEncoder for cardiac audio.

    Input:  (B, 1, T) raw waveform or (B, F, T) mel-spectrogram
    Output: (B, 32) normalized backbone latent

    Designed to match the 32-D backbone latent of the visual StudentEncoder
    so the same NPU XINT8 export pipeline applies.
    """

    def __init__(self, in_channels: int = 1, latent_dim: int = 32):
        super().__init__()

        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            # Block 2
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            # Block 3
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            # Block 4
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
        )

        # Global average pool → latent projection
        self.projector = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T) — mel or waveform
        returns: (B, latent_dim) L2-normalized
        """
        h = self.encoder(x)            # (B, 256, T')
        h = h.mean(dim=-1)             # (B, 256) global avg pool
        z = self.projector(h)          # (B, 32)
        return F.normalize(z, dim=-1)  # unit sphere


# ── Distillation projector (student → teacher dim) ────────────────────────

class DistillProjector(nn.Module):
    """Projects 32-D student latent to M2D teacher output dimension (768)."""

    def __init__(self, student_dim: int = 32, teacher_dim: int = 768):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(student_dim, 256),
            nn.GELU(),
            nn.Linear(256, teacher_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.proj(z)


# ── M2D teacher wrapper ────────────────────────────────────────────────────

class WavLMTeacher(nn.Module):
    """
    Frozen WavLM-Base teacher (Microsoft, HuggingFace).
    pip install transformers  (already installed in ryzen-ai env)
    Output: mean-pooled hidden states, shape (B, 768).
    """
    def __init__(self, model_id: str = "facebook/w2v-bert-2.0"):
        super().__init__()
        self.model_id = model_id
        self._model = None

    def _load(self, device: torch.device):
        if self._model is not None:
            return
        from transformers import Wav2Vec2BertModel as WavLMModel
        print(f"🔄 Loading WavLM teacher ({self.model_id})...")
        from transformers import AutoFeatureExtractor; self._processor = AutoFeatureExtractor.from_pretrained(self.model_id); self._model = WavLMModel.from_pretrained(self.model_id).to(device)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad = False
        print("✅ WavLM teacher loaded — output dim: 768")

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav: (B, T) raw waveform at 16kHz, float32, normalised to [-1, 1].
        Returns: (B, 768) mean-pooled hidden states.

        WavLM expects raw waveform — not mel spectrogram.
        This is the native input format: no proxy, no flattening.
        """
        self._load(wav.device)
        # Safety: ensure 2D (B, T) and normalised
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        wav = wav / (wav.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8))
        with torch.no_grad():
            input_features = self._processor(wav.cpu().numpy(), sampling_rate=16000, return_tensors="pt", padding=True).input_features.to(wav.device); out = self._model(input_features).last_hidden_state  # (B, T', 768)
            return out.mean(dim=1)                     # (B, 768)


class FallbackAudioTeacher(nn.Module):
    """
    Fallback if M2D is unavailable: uses a simple pretrained audio CNN.
    Quality will be lower than M2D but still better than broken BEATs.
    Uses torchaudio's VGGish if available.
    """

    def __init__(self, output_dim: int = 128):
        super().__init__()
        self.output_dim = output_dim
        self._model = None

    def _load(self, device: torch.device):
        if self._model is not None:
            return
        try:
            import torchaudio
            self._model = torchaudio.pipelines.HUBERT_BASE.get_model().to(device)
            self.output_dim = 768
        except Exception:
            # Minimal fallback: random fixed projection (for smoke testing)
            print("⚠️  No audio teacher found — using random fixed projection (smoke test only)")
            self._model = nn.Linear(80, self.output_dim).to(device)
            with torch.no_grad():
                nn.init.orthogonal_(self._model.weight)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._load(x.device)
        with torch.no_grad():
            if hasattr(self._model, 'extract_features'):
                out, _ = self._model.extract_features(x.squeeze(1))
                return out.mean(dim=1)
            return self._model(x.mean(dim=-1))   # fallback


# ── Dataset ────────────────────────────────────────────────────────────────

class CirCorDataset(torch.utils.data.Dataset):
    """
    CirCor heart sound dataset loader.

    Supports two layouts:

    Layout A — actual CirCor download (what you have):
        heart_data/
            training_data/   *.wav  (named like 12345_AV.wav)
            training_data.csv        (has 'Murmur' column: Present/Absent/Unknown)

    Layout B — pre-split subdirs (alternative):
        data_dir/
            normal/    *.wav
            abnormal/  *.wav

    Pass:
        data_dir  = "./heart_data/training_data"
        csv_path  = "./heart_data/training_data.csv"   (Layout A)
    Or:
        data_dir  = "./heart_data/normal_cls"          (Layout B, train-only)

    Returns (mel_spectrogram, label) where label: 0=normal, 1=abnormal.
    """

    def __init__(
        self,
        data_dir: str,
        csv_path: str = None,
        n_mels: int = 80,
        target_len: int = 608,
        sr: int = 16000,
        split: str = "train",   # "train" = 80%, "test" = 20%
    ):
        self.data_dir = Path(data_dir)
        self.n_mels = n_mels
        self.target_len = target_len
        self.sr = sr
        self.samples = []

        # ── Layout A: CSV-labelled flat directory ──────────────────────
        csv_file = Path(csv_path) if csv_path else self.data_dir.parent / "training_data.csv"
        if csv_file.exists():
            import csv
            # CirCor CSV: Patient ID, Murmur (Present/Absent/Unknown), ...
            with open(csv_file, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            # Find murmur column (case-insensitive)
            murmur_col = next(
                (k for k in rows[0].keys() if "murmur" in k.lower()), None
            )
            id_col = next(
                (k for k in rows[0].keys() if "patient" in k.lower() or k.lower() == "id"), None
            )

            if murmur_col:
                for row in rows:
                    murmur = row[murmur_col].strip().lower()
                    if murmur == "unknown":
                        continue  # skip ambiguous
                    label = 1 if murmur == "present" else 0
                    patient_id = row.get(id_col or list(row.keys())[0], "").strip()

                    # Find all WAVs for this patient (multiple auscultation locations)
                    wavs = sorted(self.data_dir.glob(f"{patient_id}_*.wav"))
                    if not wavs:
                        wavs = sorted(self.data_dir.glob(f"*{patient_id}*.wav"))
                    for wav in wavs:
                        self.samples.append((str(wav), label))

        # ── Layout B: normal/ abnormal/ subdirs ───────────────────────
        if not self.samples:
            for label, subdir in [(0, "normal"), (1, "abnormal")]:
                d = self.data_dir / subdir
                if d.exists():
                    for f in sorted(d.iterdir()):
                        if f.suffix in (".wav", ".npy", ".pt"):
                            self.samples.append((str(f), label))

        # ── Layout C: normal_cls ImageFolder (frames only, train=normal) ─
        if not self.samples:
            # Fall back to all WAVs in the dir, label all as normal (train mode)
            for f in sorted(self.data_dir.glob("*.wav")):
                self.samples.append((str(f), 0))

        if not self.samples:
            raise RuntimeError(
                f"No audio files found in {self.data_dir}\n"
                f"CSV checked: {csv_file}\n"
                f"Expected: training_data/*.wav with training_data.csv"
            )

        # Train/test split
        np.random.seed(42)
        idx = np.random.permutation(len(self.samples))
        n_train = int(len(idx) * 0.8)
        if split == "train":
            self.samples = [self.samples[i] for i in idx[:n_train]]
        else:
            self.samples = [self.samples[i] for i in idx[n_train:]]

        n_normal = sum(1 for _, l in self.samples if l == 0)
        n_abnormal = sum(1 for _, l in self.samples if l == 1)
        print(f"CirCor {split}: {len(self.samples)} files "
              f"({n_normal} normal, {n_abnormal} abnormal)")

    def _load_mel(self, path: str) -> torch.Tensor:
        if path.endswith(".npy"):
            mel = torch.from_numpy(np.load(path)).float()
        elif path.endswith(".pt"):
            mel = torch.load(path, map_location="cpu").float()
        else:
            try:
                from scipy.io import wavfile
                from scipy.signal import resample_poly
                from math import gcd
                sr_file, data = wavfile.read(path)
                if data.dtype == np.int16:
                    data = data.astype(np.float32) / 32768.0
                elif data.dtype == np.int32:
                    data = data.astype(np.float32) / 2147483648.0
                else:
                    data = data.astype(np.float32)
                if data.ndim > 1:
                    data = data.mean(axis=1)
                if sr_file != self.sr:
                    g = gcd(sr_file, self.sr)
                    data = resample_poly(data, self.sr // g, sr_file // g).astype(np.float32)
                data = data / (np.abs(data).max() + 1e-8)
                hop, n_fft = 256, 1024
                frames = []
                window = np.hanning(n_fft).astype(np.float32)
                for i in range(0, len(data) - n_fft, hop):
                    frame = data[i:i + n_fft] * window
                    spectrum = np.abs(np.fft.rfft(frame))[:self.n_mels]
                    frames.append(spectrum)
                if not frames:
                    return torch.zeros(1, self.n_mels, self.target_len)
                spec = np.stack(frames, axis=-1)
                spec = np.log1p(spec)
                mel = torch.from_numpy(spec).float().unsqueeze(0)
            except Exception as e:
                print(f"⚠️  Failed to load {path}: {e}")
                mel = torch.zeros(1, self.n_mels, self.target_len)
        return mel

    def _pad_or_crop(self, mel: torch.Tensor) -> torch.Tensor:
        """Ensure fixed length T=target_len."""
        T = mel.shape[-1]
        if T >= self.target_len:
            start = torch.randint(0, T - self.target_len + 1, (1,)).item()
            return mel[..., start: start + self.target_len]
        repeats = math.ceil(self.target_len / T)
        mel = mel.repeat(1, 1, repeats) if mel.dim() == 3 else mel.repeat(1, repeats)
        return mel[..., : self.target_len]

    def _load_wav(self, path: str) -> torch.Tensor:
        """
        Load raw waveform resampled to 16kHz for WavLM teacher input.
        Returns (T,) float32 tensor normalised to [-1, 1].
        Fixed length: WAV_LEN samples (2s at 16kHz = 32000).
        """
        WAV_LEN = 32000  # 2 seconds at 16kHz
        try:
            from scipy.io import wavfile
            from scipy.signal import resample_poly
            from math import gcd as _gcd
            sr_file, data = wavfile.read(path)
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
            else:
                data = data.astype(np.float32)
            if data.ndim > 1:
                data = data.mean(axis=1)
            if sr_file != self.sr:
                g = _gcd(sr_file, self.sr)
                data = resample_poly(data, self.sr // g, sr_file // g).astype(np.float32)
            # Normalise
            peak = np.abs(data).max()
            if peak > 1e-8:
                data = data / peak
            else:
                data = data + np.random.randn(len(data)).astype(np.float32) * 1e-5
            # Fixed-length crop / pad
            if len(data) >= WAV_LEN:
                start = np.random.randint(0, len(data) - WAV_LEN + 1)
                data = data[start: start + WAV_LEN]
            else:
                repeats = math.ceil(WAV_LEN / len(data))
                data = np.tile(data, repeats)[:WAV_LEN]
            return torch.from_numpy(data)
        except Exception:
            return torch.zeros(WAV_LEN)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        mel = self._load_mel(path)
        mel = self._pad_or_crop(mel)
        wav = self._load_wav(path)          # raw waveform for WavLM teacher
        return mel, wav, label


# ── Training ───────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Models
    student = AudioStudentEncoder(in_channels=80, latent_dim=32).to(device)
    projector = DistillProjector(student_dim=32, teacher_dim=768).to(device)

    # Teacher — try M2D first, fallback gracefully
    # Teacher: WavLM-Base (Microsoft, HuggingFace transformers)
    teacher = WavLMTeacher().to(device)
    teacher_dim = 1024
    print("✅ Using WavLM-Base teacher (microsoft/wavlm-base)")
    projector = DistillProjector(student_dim=32, teacher_dim=teacher_dim).to(device)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Dataset
    dataset = CirCorDataset(args.data, csv_path=getattr(args, "csv", None), split="train")
    if len(dataset) == 0:
        raise RuntimeError(f"No audio files found in {args.data}")

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )

    # Optimizer — student + projector only
    params = list(student.parameters()) + list(projector.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.1
    )

    student.train()
    projector.train()

    step = 0
    best_distill = float("inf")
    log = []

    print(f"\n{'='*60}")
    print(f"  M2D CARDIAC DISTILLATION")
    print(f"  Data: {args.data}")
    print(f"  Steps: {args.steps}")
    print(f"  Loss: L_distill(cosine) + {args.lambda_sigreg}*L_sigreg")
    print(f"{'='*60}\n")

    while step < args.steps:
        for mel, wav, labels in loader:
            if step >= args.steps:
                break

            mel = mel.to(device)   # (B, 80, T) mel for student
            wav = wav.to(device)   # (B, 32000) raw waveform for WavLM teacher

            # Flatten freq into channel if 4D
            if mel.dim() == 4:
                B, C, n_freq, n_time = mel.shape
                mel = mel.view(B, C * n_freq, n_time)

            # ── Teacher forward: raw WAV → WavLM (no grad) ────────────
            with torch.no_grad():
                t_feat = teacher(wav)           # (B, 768) — native wav input
                t_feat = F.normalize(t_feat, dim=-1)

            # ── Student forward: mel → 1D-CNN ─────────────────────────
            z = student(mel)                    # (B, 32) normalized
            s_proj = projector(z)               # (B, 768)
            s_proj = F.normalize(s_proj, dim=-1)

            # Distillation: cosine similarity loss
            l_distill = (1.0 - (s_proj * t_feat).sum(dim=-1)).mean()

            # Weak-SIGReg K=32
            l_sigreg = sigreg_loss(z, K=32)

            loss = l_distill + args.lambda_sigreg * l_sigreg

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            step += 1

            if step % args.log_every == 0:
                entry = {
                    "step": step,
                    "total": loss.item(),
                    "distill": l_distill.item(),
                    "sigreg": l_sigreg.item(),
                }
                log.append(entry)
                print(f"Step {step:5d}/{args.steps} | "
                      f"total={loss.item():.4f}  "
                      f"distill={l_distill.item():.4f}  "
                      f"sigreg={l_sigreg.item():.4f}")

                if l_distill.item() < best_distill:
                    best_distill = l_distill.item()
                    torch.save({
                        "step": step,
                        "student": student.state_dict(),
                        "projector": projector.state_dict(),
                        "distill": best_distill,
                    }, out_dir / "student_best.pt")

            if step % 1000 == 0:
                ckpt_path = out_dir / f"student_step{step:05d}.pt"
                torch.save({
                    "step": step,
                    "student": student.state_dict(),
                    "projector": projector.state_dict(),
                }, ckpt_path)
                print(f"💾 Checkpoint: {ckpt_path}")

    # Final checkpoint
    torch.save({
        "step": step,
        "student": student.state_dict(),
        "projector": projector.state_dict(),
        "distill_final": log[-1]["distill"] if log else None,
    }, out_dir / "student_final.pt")

    # Save log
    with open(out_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n✅ Training complete.")
    print(f"   Best distill:  {best_distill:.4f}")
    print(f"   Final distill: {log[-1]['distill'] if log else 'n/a':.4f}")
    print(f"   Checkpoint:    {out_dir}/student_final.pt")


# ── Evaluation: AUROC on held-out CirCor ─────────────────────────────────

def evaluate(args):
    """
    Compute AUROC using anomaly score = negative cosine similarity
    between student latent and normal-class prototype.
    """
    from sklearn.metrics import roc_auc_score

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load student
    ckpt = torch.load(args.checkpoint, map_location=device)
    student = AudioStudentEncoder(in_channels=80, latent_dim=32).to(device)
    student.load_state_dict(ckpt["student"])
    student.eval()
    print(f"✅ Loaded student from {args.checkpoint} (step {ckpt.get('step', '?')})")

    dataset = CirCorDataset(args.data, csv_path=getattr(args, "csv", None), split="test")
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=2
    )

    # Collect embeddings
    all_z, all_labels = [], []
    with torch.no_grad():
        for mel, wav, labels in loader:   # wav unused in eval — student only
            mel = mel.to(device)
            if mel.dim() == 4:
                B, C, n_freq, n_time = mel.shape
                mel = mel.view(B, C * n_freq, n_time)
            z = student(mel)
            all_z.append(z.cpu())
            all_labels.append(labels)

    all_z = torch.cat(all_z)           # (N, 32)
    all_labels = torch.cat(all_labels) # (N,)

    # Normal prototype = mean of normal embeddings
    normal_mask = (all_labels == 0)
    normal_proto = all_z[normal_mask].mean(0, keepdim=True)  # (1, 32)

    # Anomaly score = distance from normal prototype
    scores = torch.cdist(all_z, normal_proto).squeeze(1).numpy()

    auroc = roc_auc_score(all_labels.numpy(), scores)
    print(f"\n{'='*40}")
    print(f"  CirCor AUROC: {auroc:.4f}")
    print(f"  Baseline (WavLM mel proxy): 0.6495")
    print(f"  Target (WavLM raw WAV):  0.67-0.70")
    print(f"  Delta vs mel proxy: {auroc - 0.6495:+.4f}")
    print(f"{'='*40}")

    return auroc


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="M2D cardiac distillation")
    parser.add_argument("--data", required=True, help="Path to CirCor data dir")
    parser.add_argument("--out", default="./checkpoints/cardiac_m2d")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lambda-sigreg", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval", action="store_true", help="Evaluate mode")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint for eval")
    parser.add_argument("--csv", default=None, help="Path to training_data.csv (CirCor layout)")
    args = parser.parse_args()

    if args.eval:
        if not args.checkpoint:
            parser.error("--checkpoint required for --eval")
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
