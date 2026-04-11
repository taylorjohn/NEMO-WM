"""
train_cardiac_distillation.py — CORTEX-PE v16.15
WavLM-Base Teacher → CardiacStudentEncoder Distillation

Trains a lightweight 1D CNN student to match WavLM-Base embeddings
on PhysioNet 2016 heart sound recordings.

Architecture:
    Teacher : WavLM-Base (94M params, frozen, iGPU/CPU)
              Output: 768-D CLS embedding per 1-second window
    Student : CardiacStudentEncoder (~200K params, 1D CNN)
              Input:  (B, 1, 16000) raw waveform @ 16kHz
              Output: (B, 768) — matches teacher dimension

Loss:
    L = cosine_loss(student, teacher) + lambda * anomaly_contrast_loss
    cosine_loss    = 1 - cos_sim(student_emb, teacher_emb)
    anomaly_loss   = max(0, margin - (d_abn - d_norm))  [pushes apart normal/abnormal]

Data:
    PhysioNet 2016 Challenge — training-a through training-f
    Labels: 1=normal, -1=abnormal  (from REFERENCE.csv per subset)
    Resampled to 16kHz, windowed into 1-second clips

Targets:
    val cosine_sim >= 0.80
    val anomaly AUROC >= 0.75
    inference latency <= 5ms

Usage:
    python train_cardiac_distillation.py --data ./cardiac_data
    python train_cardiac_distillation.py --data ./cardiac_data --epochs 20 --max-files 200
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Audio loading — torchaudio preferred, scipy fallback
try:
    import torchaudio
    TORCHAUDIO_OK = True
except ImportError:
    TORCHAUDIO_OK = False
    import scipy.io.wavfile as _wavfile

warnings.filterwarnings("ignore", category=UserWarning)

TARGET_SR   = 16000   # WavLM expects 16kHz
CLIP_LEN    = 16000   # 1 second per clip
TEACHER_DIM = 768     # WavLM-Base hidden size


# ── Student model ──────────────────────────────────────────────────────────────

class CardiacStudentEncoder(nn.Module):
    """
    Lightweight 1D CNN distilled from WavLM-Base.
    Designed for real-time cardiac sound analysis on edge CPU.

    Input : (B, 1, 16000)  raw waveform, 16kHz, 1 second
    Output: (B, 768)       matches WavLM-Base embedding dimension

    Architecture mirrors WavLM's progressive downsampling:
        Conv stride-5 → stride-2 × 3 → AdaptivePool → Linear(256→768)
    """
    def __init__(self, hidden_dim: int = TEACHER_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1,   32,  kernel_size=10, stride=5,  padding=2),  # 3200
            nn.GELU(),
            nn.Conv1d(32,  64,  kernel_size=3,  stride=2,  padding=1),  # 1600
            nn.GELU(),
            nn.Conv1d(64,  128, kernel_size=3,  stride=2,  padding=1),  # 800
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=3,  stride=2,  padding=1),  # 400
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x).squeeze(-1)   # (B, 256)
        return self.proj(h)               # (B, 768)


# ── Audio loading ──────────────────────────────────────────────────────────────

def load_wav(path: Path, target_sr: int = TARGET_SR) -> np.ndarray | None:
    """
    Load a WAV file and resample to target_sr.
    Returns float32 mono array normalised to [-1, 1], or None on failure.

    Always uses scipy — torchaudio struggles with 2kHz int16 PCG recordings.
    """
    try:
        import scipy.io.wavfile as _sf
        sr, sig = _sf.read(str(path))

        # int16 → float32 (scipy returns raw int16 for PCG data)
        sig = sig.astype(np.float32)
        if sig.ndim > 1:
            sig = sig.mean(axis=1)

        # Resample to target_sr if needed (PhysioNet 2016 = 2kHz)
        if sr != target_sr:
            n_out = int(len(sig) * target_sr / sr)
            sig   = np.interp(
                np.linspace(0, len(sig) - 1, n_out),
                np.arange(len(sig)), sig
            ).astype(np.float32)

        # Normalise to [-1, 1]
        peak = np.abs(sig).max()
        if peak > 1e-6:
            sig = sig / peak
        return sig

    except Exception as e:
        return None


def clip_signal(sig: np.ndarray, clip_len: int = CLIP_LEN) -> list[np.ndarray]:
    """Split signal into non-overlapping clips of clip_len samples. Zero-pad last."""
    clips = []
    for start in range(0, len(sig), clip_len):
        clip = sig[start : start + clip_len]
        if len(clip) < clip_len // 2:
            break
        if len(clip) < clip_len:
            clip = np.pad(clip, (0, clip_len - len(clip)))
        clips.append(clip.astype(np.float32))
    return clips


# ── Dataset ────────────────────────────────────────────────────────────────────

def load_reference(data_dir: Path) -> dict[str, int]:
    """
    Load all REFERENCE.csv files from PhysioNet 2016 subsets.
    Returns {recording_name: label} where label is 1=normal, -1=abnormal.
    """
    labels = {}
    for ref_path in sorted(data_dir.rglob("REFERENCE.csv")):
        for line in ref_path.read_text().strip().splitlines():
            parts = line.strip().split(",")
            if len(parts) >= 2:
                name  = parts[0].strip()
                label = int(parts[1].strip())
                labels[name] = label
    return labels


class CardiacDataset(Dataset):
    """
    PhysioNet 2016 heart sound dataset.
    Each item: (waveform_tensor, label, teacher_embed_placeholder)
    Teacher embeddings are computed on-the-fly during training.
    """

    def __init__(
        self,
        data_dir: Path,
        file_list: list[Path],
        labels:    dict[str, int],
        max_files: int | None = None,
        verbose:   bool = True,
    ):
        self.clips  = []   # (float32 array, label_int)
        skipped     = 0

        if max_files:
            file_list = file_list[:max_files]

        for i, wav_path in enumerate(file_list):
            name = wav_path.stem
            if name not in labels:
                skipped += 1
                continue

            label = labels[name]   # 1=normal, -1=abnormal → remap to 0/1
            bin_label = 0 if label == 1 else 1

            sig = load_wav(wav_path)
            if sig is None:
                skipped += 1
                continue

            for clip in clip_signal(sig):
                self.clips.append((clip, bin_label))

            if verbose and (i + 1) % 200 == 0:
                print(f"    {i+1:5d}/{len(file_list)} files  |  {len(self.clips):,} clips")

        if verbose:
            n_norm = sum(1 for _, l in self.clips if l == 0)
            n_abn  = sum(1 for _, l in self.clips if l == 1)
            print(f"    {len(file_list)}/{len(file_list)} files  |  "
                  f"{len(self.clips):,} clips  "
                  f"({n_norm} normal / {n_abn} abnormal)  "
                  f"skipped={skipped}")

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int):
        clip, label = self.clips[idx]
        x = torch.from_numpy(clip).unsqueeze(0)   # (1, 16000)
        return x, label


# ── Loss ───────────────────────────────────────────────────────────────────────

def cosine_distill_loss(student_emb: torch.Tensor,
                        teacher_emb: torch.Tensor) -> torch.Tensor:
    """1 - cosine_similarity, mean over batch."""
    return 1.0 - F.cosine_similarity(student_emb, teacher_emb, dim=-1).mean()


def anomaly_contrast_loss(student_emb: torch.Tensor,
                          labels: torch.Tensor,
                          margin: float = 1.0) -> torch.Tensor:
    """
    Pushes normal and abnormal embeddings apart in student space.
    loss = max(0, margin - mean_dist(abn) + mean_dist(norm))
    where dist = L2 distance from batch centroid.
    """
    centroid  = student_emb.mean(dim=0)
    dists     = torch.norm(student_emb - centroid, dim=1)
    norm_mask = (labels == 0)
    abn_mask  = (labels == 1)
    if norm_mask.sum() == 0 or abn_mask.sum() == 0:
        return torch.tensor(0.0, device=student_emb.device)
    d_norm = dists[norm_mask].mean()
    d_abn  = dists[abn_mask].mean()
    return F.relu(margin - (d_abn - d_norm))


# ── AUROC ──────────────────────────────────────────────────────────────────────

def compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    count = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return float(count / (len(pos) * len(neg)))


# ── Teacher ────────────────────────────────────────────────────────────────────

def load_teacher(device: torch.device):
    """Load WavLM-Base from HuggingFace. Returns model or None if unavailable."""
    try:
        from transformers import WavLMModel
        print("  Loading WavLM-Base teacher (microsoft/wavlm-base)...")
        model = WavLMModel.from_pretrained("microsoft/wavlm-base")
        model.eval().to(device)
        for p in model.parameters():
            p.requires_grad_(False)
        print(f"  ✅ WavLM-Base loaded  [{device}]")
        return model
    except Exception as e:
        print(f"  ⚠️  WavLM-Base unavailable: {e}")
        print("  Falling back to cosine-only distillation with random teacher")
        return None


@torch.no_grad()
def get_teacher_embeddings(teacher, waveforms: torch.Tensor,
                           device: torch.device) -> torch.Tensor:
    """
    Run WavLM-Base on a batch of waveforms.
    Input:  (B, 1, 16000)
    Output: (B, 768) mean-pooled last hidden state
    """
    if teacher is None:
        return torch.randn(waveforms.shape[0], TEACHER_DIM, device=device)
    x   = waveforms.squeeze(1).to(device)   # (B, 16000)
    out = teacher(x)
    # Mean-pool over time frames
    return out.last_hidden_state.mean(dim=1)


# ── Validation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(student, teacher, loader, lambda_contrast, device) -> dict:
    student.eval()
    total_cos  = 0.0
    total_loss = 0.0
    all_scores, all_labels = [], []
    n_batches  = 0

    for waveforms, labels in loader:
        waveforms = waveforms.to(device)
        labels_t  = labels.to(device)

        student_emb = student(waveforms)
        teacher_emb = get_teacher_embeddings(teacher, waveforms, device)

        cos_l   = cosine_distill_loss(student_emb, teacher_emb)
        anom_l  = anomaly_contrast_loss(student_emb, labels_t)
        loss    = cos_l + lambda_contrast * anom_l

        cos_sim = F.cosine_similarity(student_emb, teacher_emb, dim=-1).mean().item()
        total_cos  += cos_sim
        total_loss += loss.item()

        # Anomaly score: distance from training centroid (fitted at eval time)
        scores = torch.norm(student_emb, dim=1).cpu().numpy()
        all_scores.append(scores)
        all_labels.append(labels.numpy())
        n_batches += 1

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    auroc      = compute_auroc(all_scores, all_labels)

    return {
        "loss":     total_loss / max(n_batches, 1),
        "cos_sim":  total_cos  / max(n_batches, 1),
        "auroc":    auroc,
    }


# ── Training ───────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    print("\n" + "═" * 60)
    print("  Cardiac Distillation Training — CORTEX-PE v16.15")
    print("═" * 60)
    print(f"  Data     : {args.data}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Batch    : {args.batch_size}")
    print(f"  LR       : {args.lr}")
    print(f"  Device   : {args.device}")

    device = torch.device(args.device)

    # ── Load labels ───────────────────────────────────────────────────────
    data_dir = Path(args.data)
    labels   = load_reference(data_dir)
    print(f"\n  Labels loaded: {len(labels)} recordings")
    if not labels:
        raise RuntimeError(f"No REFERENCE.csv found in {data_dir} — run download_cardiac.py first")

    n_norm = sum(1 for v in labels.values() if v ==  1)
    n_abn  = sum(1 for v in labels.values() if v == -1)
    print(f"  Normal: {n_norm}  Abnormal: {n_abn}")

    # ── File split ────────────────────────────────────────────────────────
    all_wavs = sorted(data_dir.rglob("*.wav"))
    if not all_wavs:
        raise RuntimeError(f"No .wav files in {data_dir} — run download_cardiac.py first")

    if args.max_files:
        all_wavs = all_wavs[:args.max_files]

    rng        = np.random.default_rng(42)
    idx        = rng.permutation(len(all_wavs))
    split      = max(1, int(len(all_wavs) * 0.9))
    train_wavs = [all_wavs[i] for i in idx[:split]]
    val_wavs   = [all_wavs[i] for i in idx[split:]]

    print(f"\n  Files: {len(all_wavs)} total  ({len(train_wavs)} train / {len(val_wavs)} val)")

    # ── Build datasets ────────────────────────────────────────────────────
    print("\n── Building train dataset ──────────────────────────────────")
    train_ds = CardiacDataset(data_dir, train_wavs, labels, verbose=True)

    print("\n── Building val dataset ────────────────────────────────────")
    val_ds   = CardiacDataset(data_dir, val_wavs, labels, verbose=True)

    if len(train_ds) == 0:
        raise RuntimeError("Train dataset is empty — check labels and WAV files")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)

    # ── Load teacher ──────────────────────────────────────────────────────
    print("\n── Loading teacher ─────────────────────────────────────────")
    teacher_device = torch.device("cpu")   # run teacher on CPU (iGPU via DirectML if available)
    teacher = load_teacher(teacher_device)

    # ── Build student ─────────────────────────────────────────────────────
    student   = CardiacStudentEncoder(hidden_dim=TEACHER_DIM).to(device)
    n_params  = sum(p.numel() for p in student.parameters())
    print(f"\n── CardiacStudentEncoder: {n_params:,} params ───────────────")

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ── Output dir ────────────────────────────────────────────────────────
    out_dir = Path("checkpoints/cardiac")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n── Training: {args.epochs} epochs ──────────")
    print(f"   Train clips : {len(train_ds):,}")
    print(f"   Val clips   : {len(val_ds):,}")
    print(f"   Steps/epoch : {len(train_loader)}")
    print(f"   Target      : cos_sim ≥ 0.80, AUROC ≥ 0.75")

    best_cos   = 0.0
    best_auroc = 0.0
    log        = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        student.train()
        train_loss = 0.0

        for waveforms, labels_batch in train_loader:
            waveforms  = waveforms.to(device)
            labels_t   = labels_batch.to(device)

            # Teacher forward (no grad, CPU)
            with torch.no_grad():
                teacher_emb = get_teacher_embeddings(teacher, waveforms, teacher_device)
                teacher_emb = teacher_emb.to(device)

            student_emb = student(waveforms)

            cos_l  = cosine_distill_loss(student_emb, teacher_emb)
            anom_l = anomaly_contrast_loss(student_emb, labels_t)
            loss   = cos_l + args.lambda_contrast * anom_l

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # Validate
        val_metrics = validate(student, teacher, val_loader, args.lambda_contrast, device)
        elapsed     = time.perf_counter() - t0

        improved_cos   = val_metrics["cos_sim"]  > best_cos
        improved_auroc = val_metrics["auroc"]    > best_auroc
        improved       = improved_cos or improved_auroc

        if improved:
            if improved_cos:   best_cos   = val_metrics["cos_sim"]
            if improved_auroc: best_auroc = val_metrics["auroc"]
            torch.save({
                "model_state_dict": student.state_dict(),
                "epoch":            epoch,
                "val_loss":         val_metrics["loss"],
                "val_cos_sim":      val_metrics["cos_sim"],
                "val_auroc":        val_metrics["auroc"],
                "hidden_dim":       TEACHER_DIM,
            }, out_dir / "student_best.pt")

        tag = "  ✅ New best" if improved else ""
        print(f"  Epoch {epoch:3d}/{args.epochs}  "
              f"train={train_loss:.4f}  "
              f"val_loss={val_metrics['loss']:.4f}  "
              f"cos_sim={val_metrics['cos_sim']:.4f}  "
              f"auroc={val_metrics['auroc']:.4f}  "
              f"{elapsed:.1f}s{tag}")

        log.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})

    # Save final
    torch.save({
        "model_state_dict": student.state_dict(),
        "epoch":            args.epochs,
        "val_loss":         val_metrics["loss"],
        "val_cos_sim":      val_metrics["cos_sim"],
        "val_auroc":        val_metrics["auroc"],
        "hidden_dim":       TEACHER_DIM,
    }, out_dir / "student_final.pt")

    with open(out_dir / "log.json", "w") as f:
        json.dump(log, f, indent=2)

    print("\n" + "═" * 60)
    print(f"  Done.  Best cos_sim={best_cos:.4f}  Best AUROC={best_auroc:.4f}")
    print(f"  Best  : {out_dir / 'student_best.pt'}")
    print(f"  Final : {out_dir / 'student_final.pt'}")

    cos_pass   = best_cos   >= 0.80
    auroc_pass = best_auroc >= 0.75
    print(f"\n  cos_sim  ≥ 0.80 : {'✅ PASS' if cos_pass   else '❌ FAIL'}  ({best_cos:.4f})")
    print(f"  AUROC    ≥ 0.75 : {'✅ PASS' if auroc_pass  else '❌ FAIL'}  ({best_auroc:.4f})")
    print("\n  Evaluate:")
    print(f"  python eval_cardiac_audio.py --student {out_dir / 'student_best.pt'}")
    print("═" * 60 + "\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cardiac audio distillation training")
    parser.add_argument("--data",            required=True,
                        help="cardiac_data directory (from download_cardiac.py)")
    parser.add_argument("--epochs",          type=int,   default=20)
    parser.add_argument("--batch-size",      type=int,   default=32)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--lambda-contrast", type=float, default=0.1,
                        help="Weight on anomaly contrast loss (default 0.1)")
    parser.add_argument("--device",          default="cpu")
    parser.add_argument("--max-files",       type=int,   default=None,
                        help="Cap file count for quick tests")
    args = parser.parse_args()
    train(args)
