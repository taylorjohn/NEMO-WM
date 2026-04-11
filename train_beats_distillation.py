"""
train_beats_distillation.py — CORTEX-PE Heart Sound Encoder Training
                               with BEATs Audio Teacher

Replaces DINOv2 with BEATs (Microsoft, ICML 2023) as the distillation teacher.
BEATs was trained on AudioSet-2M and produces semantically rich 768-D audio
representations from raw 16kHz waveforms.

Key difference from train_distillation.py:
    - Teacher: BEATs (audio-native) instead of DINOv2 (vision)
    - Input to teacher: raw PCG waveform resampled to 16kHz
    - Input to student: 224×224 mel-spectrogram (same as before)
    - Teacher output: mean-pooled patch features (1, 768) per window
    - Projector: backbone_g (32-D) → 768-D to match BEATs output dim

Why this works:
    BEATs was trained on 2M audio clips covering machinery, biological sounds,
    music, speech, and environmental audio. Heart sounds fall within its
    training distribution. The encoder produces meaningful cardiac features
    that correlate with pathology — unlike DINOv2 which has no cardiac priors.

Usage:
    # Phase 1 — semantic grounding on normal PCG recordings
    python train_beats_distillation.py \\
        --data ./heart_data/normal_cls \\
        --wav-dir ./heart_data/training_data \\
        --beats ./BEATs_iter3_plus_AS2M.pt \\
        --steps 6000 --out ./checkpoints/heart_beats \\
        --lambda-sigreg 2.0

    # Evaluate
    python collect_heart_frames.py \\
        --evaluate \\
        --weights ./checkpoints/heart_beats/cortex_student_phase1_final.pt \\
        --out ./heart_data/all_frames
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from student_encoder import StudentEncoder

# =============================================================================
# Constants
# =============================================================================
BEATS_FS        = 16000   # BEATs expects 16kHz
PCG_FS          = 4000    # CirCor native sample rate
WINDOW_SAMPLES  = int(PCG_FS * 2.0)   # 2s windows
HOP_SAMPLES     = int(PCG_FS * 1.0)   # 50% overlap

IMAGENET_MEAN   = [0.485, 0.456, 0.406]
IMAGENET_STD    = [0.229, 0.224, 0.225]

FRAME_TRANSFORM = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# =============================================================================
# SIGReg loss (same as train_distillation.py)
# =============================================================================
def sigreg_loss(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    std  = torch.sqrt(z.var(dim=0) + eps)
    var_loss = torch.mean(F.relu(1 - std))
    z_c      = z - z.mean(dim=0)
    cov      = (z_c.T @ z_c) / (z.shape[0] - 1)
    cov_loss = cov.fill_diagonal_(0).pow(2).sum() / z.shape[1]
    return var_loss + cov_loss


# =============================================================================
# BEATs teacher wrapper
# =============================================================================
class BEATsTeacher(nn.Module):
    """
    Wraps BEATs for use as a distillation teacher.
    Accepts raw 16kHz waveforms, returns mean-pooled 768-D features.
    """
    def __init__(self, checkpoint_path: str):
        super().__init__()
        # Import BEATs from local files
        try:
            from BEATs import BEATs, BEATsConfig
        except ImportError:
            raise ImportError(
                "BEATs.py not found in current directory.\n"
                "Download from: https://github.com/microsoft/unilm/tree/master/beats"
            )
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        cfg        = BEATsConfig(checkpoint['cfg'])
        self.model = BEATs(cfg)
        self.model.load_state_dict(checkpoint['model'])
        self.feat_dim = 768

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wav: (B, T) raw waveform at 16kHz, float32
        Returns:
            (B, 768) mean-pooled audio features
        """
        padding_mask = torch.zeros(wav.shape, dtype=torch.bool,
                                   device=wav.device)
        patches, _   = self.model.extract_features(wav,
                            padding_mask=padding_mask)
        # patches: (B, T_patches, 768) → mean pool → (B, 768)
        return patches.mean(dim=1)


# =============================================================================
# Paired dataset: (mel_frame_path, wav_segment)
# =============================================================================
class HeartPairedDataset(Dataset):
    """
    Returns paired (mel_spectrogram_tensor, raw_waveform_tensor) for each
    training frame. The mel spectrogram is the student input; the raw
    waveform (resampled to 16kHz) is the teacher input.

    Requires:
        - frame_dir: directory with PNG mel-spectrogram frames + metadata.json
        - wav_dir:   directory with original .wav files (CirCor training_data/)
    """
    def __init__(self, frame_dir: str, wav_dir: str, split: str = "normal"):
        self.frame_dir = Path(frame_dir)
        self.wav_dir   = Path(wav_dir)
        self.tf        = FRAME_TRANSFORM

        meta_path = self.frame_dir / "metadata.json"
        if not meta_path.exists():
            # Flat frame directory — no metadata, load all
            self.frames = sorted(self.frame_dir.glob("*.png"))
            self.meta   = [{"frame": f.name, "wav_file": None, "window": 0}
                           for f in self.frames]
        else:
            meta = json.load(open(meta_path))
            if split == "normal":
                meta = [m for m in meta if m.get("label", 0) == 0]
            self.meta   = meta
            self.frames = [self.frame_dir / m["frame"] for m in meta]

        print(f"  HeartPairedDataset: {len(self.frames)} frames ({split})")

    def __len__(self) -> int:
        return len(self.frames)

    def _load_wav_segment(self, wav_name: str, window: int) -> torch.Tensor:
        """Load raw PCG segment, resample to 16kHz, return (T_16k,) tensor."""
        import wave, struct
        wav_path = self.wav_dir / wav_name
        if not wav_path.exists():
            # Fallback: return silence
            return torch.zeros(WINDOW_SAMPLES * (BEATS_FS // PCG_FS))

        try:
            with wave.open(str(wav_path), 'rb') as wf:
                n_ch   = wf.getnchannels()
                sw     = wf.getsampwidth()
                n_fr   = wf.getnframes()
                raw    = wf.readframes(n_fr)

            if sw == 2:
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
                audio = (audio - 128.0) / 128.0
            if n_ch > 1:
                audio = audio[::n_ch]

            # Extract the window corresponding to this frame
            start = window * HOP_SAMPLES
            end   = start + WINDOW_SAMPLES
            if end > len(audio):
                seg = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
                seg[:max(0, len(audio) - start)] = audio[start:min(end, len(audio))]
            else:
                seg = audio[start:end]

            # Resample 4kHz → 16kHz (4× repeat + low-pass via mean)
            seg_16k = np.repeat(seg, BEATS_FS // PCG_FS).astype(np.float32)

            # Normalise
            seg_16k = seg_16k / (np.abs(seg_16k).max() + 1e-8)
            return torch.from_numpy(seg_16k)

        except Exception:
            return torch.zeros(WINDOW_SAMPLES * (BEATS_FS // PCG_FS))

    def __getitem__(self, idx: int):
        m         = self.meta[idx]
        # Student input: mel spectrogram frame
        frame_img = Image.open(self.frames[idx]).convert("RGB")
        frame_t   = self.tf(frame_img)

        # Teacher input: raw waveform
        wav_name  = m.get("wav_file", None)
        window    = m.get("window", 0)
        if wav_name:
            wav_t = self._load_wav_segment(wav_name, window)
        else:
            wav_t = torch.zeros(WINDOW_SAMPLES * (BEATS_FS // PCG_FS))

        return frame_t, wav_t


# =============================================================================
# Training loop
# =============================================================================
def train(
    data_dir:       str,
    wav_dir:        str,
    beats_path:     str,
    out_dir:        str,
    steps:          int   = 6000,
    batch_size:     int   = 16,
    lr:             float = 1e-3,
    lambda_sigreg:  float = 2.0,
    save_every:     int   = 1000,
    seed:           int   = 42,
):
    torch.manual_seed(seed)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*62}")
    print(f"  BEATS DISTILLATION — Heart Sound Encoder")
    print(f"  Data:         {data_dir}")
    print(f"  WAV dir:      {wav_dir}")
    print(f"  BEATs:        {beats_path}")
    print(f"  Steps:        {steps}")
    print(f"  lambda_sig:   {lambda_sigreg}")
    print(f"{'='*62}\n")

    # Load BEATs teacher
    print("Loading BEATs teacher...")
    teacher = BEATsTeacher(beats_path)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"  BEATs loaded — feature dim: {teacher.feat_dim}")

    # Student encoder
    model = StudentEncoder()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Student encoder: {n_params:,} params")

    # Projector: backbone_g (32-D) → 768-D (BEATs feature dim)
    projector = nn.Sequential(
        nn.Linear(32, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Linear(256, teacher.feat_dim),
    )

    opt = torch.optim.Adam(
        list(model.parameters()) + list(projector.parameters()),
        lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    # Dataset — use paired dataset if wav_dir exists, else fallback
    wav_path = Path(wav_dir)
    if wav_path.exists() and (wav_path / "metadata.json").exists() == False:
        # wav_dir is the training_data directory
        meta_path = Path(data_dir) / "metadata.json"
        if meta_path.exists():
            dataset = HeartPairedDataset(data_dir, wav_dir, split="normal")
        else:
            print("  ⚠️  No metadata.json — using mel frames only (no BEATs alignment)")
            dataset = datasets.ImageFolder(data_dir, transform=FRAME_TRANSFORM)
    else:
        try:
            dataset = HeartPairedDataset(data_dir, wav_dir, split="normal")
        except Exception:
            dataset = datasets.ImageFolder(data_dir, transform=FRAME_TRANSFORM)

    loader   = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                          num_workers=0, drop_last=True)
    iterator = iter(loader)

    use_paired = isinstance(dataset, HeartPairedDataset)
    print(f"  Dataset: {len(dataset)} frames  (paired={use_paired})")
    print(f"  Batches per epoch: {len(loader)}\n")

    model.train()
    projector.train()

    for step in range(1, steps + 1):
        t0 = time.time()

        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch    = next(iterator)

        if use_paired:
            frames, wavs = batch
        else:
            frames, _ = batch
            wavs      = None

        # Student forward
        z          = model(frames)
        # backbone_g: first 32 dims (shape axis per ShatteredLatentHead)
        backbone_g = z[:, :32]
        s_proj     = projector(backbone_g)
        s_proj     = F.normalize(s_proj, dim=-1)

        # Teacher forward
        if wavs is not None and use_paired:
            with torch.no_grad():
                t_feat = teacher(wavs)
                t_proj = F.normalize(t_feat, dim=-1)
            distill_loss = 2 - 2 * (s_proj * t_proj).sum(dim=-1).mean()
        else:
            distill_loss = 2 - 2 * (s_proj * t_proj).sum(dim=-1).mean()

        # SIGReg on full latent z
        sig_loss = sigreg_loss(z)

        loss = distill_loss + lambda_sigreg * sig_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(projector.parameters()), 1.0)
        opt.step()
        scheduler.step()

        elapsed = (time.time() - t0) * 1000

        if step % 100 == 0:
            print(f"Step {step:>5}/{steps} | "
                  f"total={float(loss):.4f}  "
                  f"distill={float(distill_loss):.4f}  "
                  f"sigreg={float(sig_loss):.4f}  "
                  f"({elapsed:.1f}ms)")

        if step % save_every == 0:
            ckpt_path = Path(out_dir) / f"cortex_student_beats_step{step:06d}.pt"
            torch.save({
                "step":        step,
                "model":       model.state_dict(),
                "projector":   projector.state_dict(),
                "teacher":     "BEATs_iter3_plus_AS2M",
                "lambda_sig":  lambda_sigreg,
            }, str(ckpt_path))
            print(f"  Checkpoint saved: {ckpt_path}")

    # Save final
    final_path = Path(out_dir) / "cortex_student_phase1_final.pt"
    torch.save({"model": model.state_dict(), "step": steps}, str(final_path))
    print(f"\n  Final checkpoint: {final_path}")
    return str(final_path)


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="CORTEX-PE Heart Sound Encoder — BEATs Distillation")
    p.add_argument("--data",          default="./heart_data/normal_cls",
                   help="ImageFolder or flat dir of normal mel-spec frames")
    p.add_argument("--wav-dir",       default="./heart_data/training_data",
                   help="CirCor training_data directory (raw .wav files)")
    p.add_argument("--beats",         default="./BEATs_iter3_plus_AS2M.pt",
                   help="Path to BEATs pretrained checkpoint")
    p.add_argument("--out",           default="./checkpoints/heart_beats")
    p.add_argument("--steps",         type=int,   default=6000)
    p.add_argument("--batch",         type=int,   default=16)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--lambda-sigreg", type=float, default=2.0)
    p.add_argument("--save-every",    type=int,   default=1000)
    args = p.parse_args()

    train(
        data_dir      = args.data,
        wav_dir       = args.wav_dir,
        beats_path    = args.beats,
        out_dir       = args.out,
        steps         = args.steps,
        batch_size    = args.batch,
        lr            = args.lr,
        lambda_sigreg = args.lambda_sigreg,
        save_every    = args.save_every,
    )
