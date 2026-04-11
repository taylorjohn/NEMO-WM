"""
train_temporal_contrastive.py — CORTEX-PE v16.12
=================================================
InfoNCE temporal contrastive training on RECON HDF5 trajectory files.

RECON data layout (confirmed from probe)
----------------------------------------
  Each file: ./recon_data/recon_release/jackal_2019-*.hdf5
  T = 70 frames per file at 4Hz

  images/rgb_left   (70,)  |S66958  — JPEG-encoded bytes per frame
  jackal/position   (70, 3)          — x, y, z position (GPS/odometry)
  commands/linear_velocity  (70,)    — forward velocity
  commands/angular_velocity (70,)    — turn rate

Problem being solved
--------------------
StudentEncoder (DINOv2-distilled) is too semantic for 4Hz outdoor frames.
Consecutive frames look nearly identical in latent space. Fix: train a
lightweight TemporalHead on top of the frozen encoder using InfoNCE so that
frames separated by k steps are pushed apart in projection space.

Architecture
------------
  Frozen StudentEncoder (128-D) → TemporalHead (128→64-D) → InfoNCE

Only the TemporalHead trains. Encoder stays frozen.

Success criterion: visual quasimetric AUROC > 0.70 on held-out trajectories.

Usage
-----
  python train_temporal_contrastive.py ^
      --data ./recon_data/recon_release ^
      --encoder ./checkpoints/maze_weak_sigreg_straight/cortex_student_phase2_final.pt ^
      --k 7 --epochs 20 ^
      --out ./checkpoints/recon_contrastive/

  # Probe one file only
  python train_temporal_contrastive.py --data ./recon_data/recon_release --probe-only

  # k-sweep (3 epochs per k, finds best k)
  python train_temporal_contrastive.py ^
      --data ./recon_data/recon_release ^
      --encoder ./checkpoints/maze_weak_sigreg_straight/cortex_student_phase2_final.pt ^
      --k-sweep
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# ══════════════════════════════════════════════════════════════════════════════
# Data utilities
# ══════════════════════════════════════════════════════════════════════════════

def decode_jpeg(raw: bytes) -> np.ndarray:
    """Decode JPEG bytes → (H, W, 3) uint8 numpy array."""
    return np.array(Image.open(io.BytesIO(raw)).convert("RGB"))


def get_trajectory_files(data_dir: str) -> list[str]:
    pattern = str(Path(data_dir) / "*.hdf5")
    files = sorted(glob.glob(pattern))
    if not files:
        pattern2 = str(Path(data_dir) / "*.h5")
        files = sorted(glob.glob(pattern2))
    return files


# Image transform matching StudentEncoder training
TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ══════════════════════════════════════════════════════════════════════════════
# HDF5 probe
# ══════════════════════════════════════════════════════════════════════════════

def probe_one_file(path: str) -> None:
    print(f"\n{'═'*60}")
    print(f"  Probing: {Path(path).name}")
    print(f"{'═'*60}")
    with h5py.File(path, "r") as f:
        def show(name, obj):
            indent = "  " * name.count("/")
            if hasattr(obj, "shape"):
                print(f"  {indent}/{name}  shape={obj.shape}  dtype={obj.dtype}")
            else:
                print(f"  {indent}/{name}/")
        print(f"  Top keys: {list(f.keys())}")
        f.visititems(show)

        # Decode and show first frame
        raw = bytes(f["images/rgb_left"][0])
        img = decode_jpeg(raw)
        print(f"\n  Decoded rgb_left[0]: shape={img.shape}  dtype={img.dtype}")
        print(f"  Position shape:      {f['jackal/position'].shape}")
        print(f"  T = {f['jackal/position'].shape[0]} frames @ 4Hz = "
              f"{f['jackal/position'].shape[0]/4:.1f}s")
    print(f"{'═'*60}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Encoder loading
# ══════════════════════════════════════════════════════════════════════════════

def load_student_encoder(ckpt_path: str, device: str) -> nn.Module:
    from student_encoder import StudentEncoder
    enc = StudentEncoder()
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict):
        for key in ("model_state_dict", "student", "model", "state_dict"):
            if key in state:
                state = state[key]
                break
    enc.load_state_dict(state, strict=False)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    enc.to(device)
    n = sum(p.numel() for p in enc.parameters())
    print(f"  ✅ StudentEncoder frozen — {n:,} params on {device}")
    return enc


# ══════════════════════════════════════════════════════════════════════════════
# Temporal projection head
# ══════════════════════════════════════════════════════════════════════════════

class TemporalHead(nn.Module):
    """
    128-D encoder output → 64-D L2-normalized temporal embedding.
    Only this is trained. ~33K params.
    """
    def __init__(self, in_dim: int = 128, out_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(z), dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
# InfoNCE loss
# ══════════════════════════════════════════════════════════════════════════════

def info_nce_loss(q: torch.Tensor, k: torch.Tensor,
                  temperature: float = 0.07) -> torch.Tensor:
    """
    Symmetric InfoNCE. q, k: (B, D) L2-normalized.
    Diagonal = positive pairs. Off-diagonal = negatives.
    """
    B = q.shape[0]
    logits = torch.mm(q, k.T) / temperature   # (B, B)
    labels = torch.arange(B, device=q.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0


# ══════════════════════════════════════════════════════════════════════════════
# Dataset: pre-encode all frames, build (z_t, z_{t+k}) pairs
# ══════════════════════════════════════════════════════════════════════════════

class RECONTemporalDataset(Dataset):
    """
    Walks all jackal_*.hdf5 files, decodes JPEG frames, encodes with frozen
    StudentEncoder, and builds (z_t, z_{t+k}) pairs for InfoNCE training.

    Pre-encoding at init is much faster than encoding per-batch.
    """

    def __init__(
        self,
        files:      list[str],
        encoder:    nn.Module,
        k:          int   = 7,
        device:     str   = "cpu",
    ):
        self.k = k
        self.pairs: list[tuple[torch.Tensor, torch.Tensor]] = []

        skipped = 0
        encoded = 0

        print(f"\n  Building dataset from {len(files)} files (k={k})...")

        for i, fpath in enumerate(files):
            try:
                with h5py.File(fpath, "r") as f:
                    raw_imgs = f["images/rgb_left"][:]   # (T,) of S bytes
                    T = len(raw_imgs)

                if T < k + 1:
                    skipped += 1
                    continue

                # Decode all frames → tensor (T, 3, 224, 224)
                frames = []
                for raw in raw_imgs:
                    img = decode_jpeg(bytes(raw))
                    frames.append(TRANSFORM(img))
                frame_batch = torch.stack(frames)   # (T, 3, 224, 224)

                # Encode in batches of 32
                with torch.no_grad():
                    chunks = []
                    for start in range(0, T, 32):
                        batch = frame_batch[start:start+32].to(device)
                        chunks.append(encoder(batch).cpu())
                    latents = torch.cat(chunks, dim=0)   # (T, 128)

                # Build (z_t, z_{t+k}) pairs
                for t in range(T - k):
                    self.pairs.append((latents[t], latents[t + k]))

                encoded += 1

            except Exception as e:
                skipped += 1
                if i < 5:
                    print(f"    ⚠️  {Path(fpath).name}: {e}")

            if (i + 1) % 100 == 0 or i == len(files) - 1:
                print(f"    {i+1}/{len(files)} files  |  {len(self.pairs):,} pairs",
                      end="\r")

        print(f"\n  ✅ Dataset: {len(self.pairs):,} pairs from "
              f"{encoded} files  (skipped {skipped})")

        if len(self.pairs) == 0:
            raise RuntimeError(
                "No pairs found. Check that images/rgb_left contains valid JPEG data "
                "and that T >= k+1."
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pairs[idx]


# ══════════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════════

def train(
    data_dir:     str,
    encoder_path: str,
    k:            int   = 7,
    epochs:       int   = 20,
    batch_size:   int   = 256,
    lr:           float = 3e-4,
    temperature:  float = 0.07,
    out_dir:      str   = "./checkpoints/recon_contrastive/",
    max_files:    int | None = None,
    val_frac:     float = 0.1,
    device:       str   = "cpu",
) -> dict:

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── Encoder ─────────────────────────────────────────────────────────────
    print("\n── Loading encoder ─────────────────────────────────────────────")
    encoder = load_student_encoder(encoder_path, device)

    # ── File list ────────────────────────────────────────────────────────────
    all_files = get_trajectory_files(data_dir)
    if max_files:
        all_files = all_files[:max_files]
    n_val   = max(1, int(len(all_files) * val_frac))
    train_f = all_files[:-n_val]
    val_f   = all_files[-n_val:]
    print(f"\n── RECON files: {len(all_files)} total  "
          f"({len(train_f)} train / {len(val_f)} val)")

    # ── Datasets ─────────────────────────────────────────────────────────────
    print("\n── Building train dataset ──────────────────────────────────────")
    train_ds = RECONTemporalDataset(train_f, encoder, k=k, device=device)

    print("\n── Building val dataset ────────────────────────────────────────")
    val_ds   = RECONTemporalDataset(val_f,   encoder, k=k, device=device)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=0, pin_memory=False, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=0, pin_memory=False, drop_last=False)

    # ── Head ─────────────────────────────────────────────────────────────────
    head = TemporalHead(in_dim=128, out_dim=64, hidden_dim=256).to(device)
    n_p  = sum(p.numel() for p in head.parameters())
    print(f"\n── TemporalHead: {n_p:,} params ─────────────────────────────────")

    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * max(1, len(train_dl))
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    log = []
    best_val  = float("inf")
    best_path = out_path / f"temporal_head_k{k}_best.pt"

    print(f"\n── Training: {epochs} epochs, k={k}, τ={temperature} ──────────")
    print(f"   Train pairs: {len(train_ds):,}  |  Val pairs: {len(val_ds):,}")
    print(f"   Batch: {batch_size}  |  Steps/epoch: {len(train_dl)}")
    print(f"   InfoNCE minimum ≈ log({batch_size}) = {math.log(batch_size):.2f}\n")

    for epoch in range(1, epochs + 1):
        # Train
        head.train()
        t0 = time.perf_counter()
        tr_losses = []
        for z_t, z_tk in train_dl:
            z_t  = z_t.to(device)
            z_tk = z_tk.to(device)
            loss = info_nce_loss(head(z_t), head(z_tk), temperature)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            tr_losses.append(loss.item())

        # Val
        head.eval()
        va_losses = []
        with torch.no_grad():
            for z_t, z_tk in val_dl:
                z_t  = z_t.to(device)
                z_tk = z_tk.to(device)
                if z_t.shape[0] > 1:
                    va_losses.append(
                        info_nce_loss(head(z_t), head(z_tk), temperature).item()
                    )

        tr = float(np.mean(tr_losses))
        va = float(np.mean(va_losses)) if va_losses else float("nan")
        elapsed = time.perf_counter() - t0
        log.append({"epoch": epoch, "train": tr, "val": va})

        print(f"  Epoch {epoch:>3}/{epochs}  train={tr:.4f}  val={va:.4f}  {elapsed:.1f}s")

        if va < best_val:
            best_val = va
            torch.save({
                "epoch": epoch, "head": head.state_dict(),
                "val_loss": va, "train_loss": tr,
                "k": k, "temperature": temperature, "encoder": encoder_path,
            }, best_path)
            print(f"    ✅ New best → {best_path.name}")

    final_path = out_path / f"temporal_head_k{k}_final.pt"
    torch.save({
        "epoch": epochs, "head": head.state_dict(),
        "val_loss": va, "train_loss": tr,
        "k": k, "temperature": temperature, "encoder": encoder_path,
    }, final_path)

    log_path = out_path / f"log_k{k}.json"
    json.dump(log, open(log_path, "w"), indent=2)

    print(f"\n{'═'*60}")
    print(f"  Done. Best val loss: {best_val:.4f}")
    print(f"  Best:  {best_path}")
    print(f"  Final: {final_path}")
    print(f"  Log:   {log_path}")
    print(f"{'═'*60}")
    print(f"\n  Evaluate AUROC:")
    print(f"  python eval_recon_quasimetric.py ^")
    print(f"      --head {best_path} ^")
    print(f"      --encoder {encoder_path} ^")
    print(f"      --data {data_dir}")

    return {"best_val_loss": best_val, "best_ckpt": str(best_path),
            "k": k, "n_train": len(train_ds), "n_val": len(val_ds)}


# ══════════════════════════════════════════════════════════════════════════════
# k-sweep
# ══════════════════════════════════════════════════════════════════════════════

def k_sweep(args: argparse.Namespace) -> None:
    results = {}
    for k in [3, 5, 7, 10, 15]:
        print(f"\n{'─'*60}\n  k-sweep: k={k}\n{'─'*60}")
        r = train(
            data_dir=args.data, encoder_path=args.encoder,
            k=k, epochs=3, batch_size=args.batch_size, lr=args.lr,
            temperature=args.temperature, out_dir=args.out,
            max_files=args.max_files, device=args.device,
        )
        results[k] = r["best_val_loss"]

    best_k = min(results, key=results.get)
    print(f"\n{'═'*60}\n  k-sweep summary\n{'═'*60}")
    for k, v in sorted(results.items()):
        mark = " ← best" if k == best_k else ""
        print(f"  k={k:>3}  val={v:.4f}{mark}")
    print(f"\n  Recommended: --k {best_k}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description="RECON temporal contrastive training — CORTEX-PE v16.12"
    )
    p.add_argument("--data",        required=True,
                   help="Directory containing jackal_*.hdf5 files")
    p.add_argument("--encoder",     default=None,
                   help="StudentEncoder checkpoint (required unless --probe-only)")
    p.add_argument("--k",           type=int,   default=7,
                   help="Temporal offset in frames (default 7 ≈ 1.75s @ 4Hz)")
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--batch-size",  type=int,   default=256)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--out",         default="./checkpoints/recon_contrastive/")
    p.add_argument("--max-files",   type=int,   default=None,
                   help="Limit files loaded (debug)")
    p.add_argument("--val-frac",    type=float, default=0.1)
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--probe-only",  action="store_true",
                   help="Probe first file and exit")
    p.add_argument("--k-sweep",     action="store_true")
    args = p.parse_args()

    print(f"\n{'═'*60}")
    print(f"  RECON Temporal Contrastive — CORTEX-PE v16.12")
    print(f"{'═'*60}")

    files = get_trajectory_files(args.data)
    print(f"  Data dir: {args.data}")
    print(f"  Files found: {len(files)}")
    if files:
        print(f"  First: {Path(files[0]).name}")

    if args.probe_only:
        if files:
            probe_one_file(files[0])
        return

    if args.encoder is None:
        p.error("--encoder required unless --probe-only")

    print(f"  Encoder:     {args.encoder}")
    print(f"  k:           {args.k}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Device:      {args.device}")

    if args.k_sweep:
        k_sweep(args)
    else:
        train(
            data_dir=args.data, encoder_path=args.encoder,
            k=args.k, epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, temperature=args.temperature, out_dir=args.out,
            max_files=args.max_files, val_frac=args.val_frac, device=args.device,
        )


if __name__ == "__main__":
    main()
