"""
train_student_clip.py — Sprint 6: CLIP Dual-Distillation
=========================================================
Fine-tunes the StudentEncoder with CLIP as a second teacher alongside
the existing DINOv2 knowledge (preserved via self-distillation).

Dual-distillation objective:
    L_total = (1 - alpha) * L_preserve + alpha * L_clip

    L_preserve = cosine_loss(student(x), frozen_student(x))
        → Prevents catastrophic forgetting of DINOv2 spatial features.
        → Frozen copy of student_best.pt acts as the DINOv2 anchor.

    L_clip = cosine_loss(student(x), clip_image_encoder(x))
        → Pulls student embeddings toward CLIP's semantic space.
        → CLIP ViT-B/32 as fixed teacher (weights frozen).

    alpha = 0.3 (30% CLIP, 70% DINOv2 preservation)

Expected outcome:
    Pre-distillation:  clip_alignment_results.json shows 1.07x ratios (null)
    Post-distillation: clip_particle_alignment_test.py shows 1.3–2.0x ratios

Saves to: checkpoints/dinov2_student/student_clip_best.pt
    (separate from student_best.pt — no collision)

Run:
    python train_student_clip.py --hdf5-dir recon_data/recon_release
    python train_student_clip.py --hdf5-dir recon_data/recon_release --epochs 10 --alpha 0.3

Author: John Taylor — github.com/taylorjohn
Date:   2026-04-03
"""

import argparse
import copy
import glob
import io
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import clip
from PIL import Image

from train_mvtec import StudentEncoder

DEVICE = torch.device("cpu")


# ── Image preprocessing ───────────────────────────────────────────────────────

def decode_frame(jpeg_bytes, size: int = 224) -> torch.Tensor:
    """Decode JPEG bytes to normalised (3, 224, 224) tensor."""
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB").resize((size, size))
    return torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0


def make_clip_preprocess(clip_preprocess, size: int = 224):
    """
    Wrap CLIP's PIL-based preprocess to accept tensors already at `size`.
    CLIP expects 224×224 normalised with its own mean/std.
    """
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    clip_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

    def preprocess(tensor_chw: torch.Tensor) -> torch.Tensor:
        """tensor_chw: (3, H, W) in [0,1] → CLIP-normalised (3, 224, 224)"""
        if tensor_chw.shape[-2:] != (size, size):
            tensor_chw = F.interpolate(
                tensor_chw.unsqueeze(0), size=(size, size),
                mode="bilinear", align_corners=False
            ).squeeze(0)
        return (tensor_chw - clip_mean) / clip_std

    return preprocess


def iter_frames(hdf5_dir: str, max_files: int = None,
                batch_size: int = 32, stride: int = 5):
    """
    Yield batches of (3, 224, 224) tensors from RECON HDF5 files.

    Args:
        hdf5_dir:   directory containing RECON HDF5 files
        max_files:  limit number of files (None = all)
        batch_size: frames per batch
        stride:     sample every Nth frame (reduces correlation)
    """
    files = sorted(glob.glob(f"{hdf5_dir}/*.hdf5"))
    if max_files:
        files = files[:max_files]

    buf = []
    for fpath in files:
        try:
            with h5py.File(fpath, "r") as hf:
                imgs = hf["images"]["rgb_left"]
                for i in range(0, len(imgs), stride):
                    try:
                        frame = decode_frame(bytes(imgs[i]))
                        buf.append(frame)
                        if len(buf) == batch_size:
                            yield torch.stack(buf)
                            buf = []
                    except Exception:
                        pass
        except Exception:
            pass

    if buf:
        yield torch.stack(buf)


def cosine_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Mean cosine embedding loss between two sets of unit vectors.
    Both a and b are (B, D) — normalised internally.
    """
    a_n = F.normalize(a, dim=-1)
    b_n = F.normalize(b, dim=-1)
    return (1 - (a_n * b_n).sum(dim=-1)).mean()


def train(
    hdf5_dir:   str   = "recon_data/recon_release",
    epochs:     int   = 15,
    lr:         float = 5e-5,
    batch_size: int   = 32,
    alpha:      float = 0.3,
    max_files:  int   = None,
    log_every:  int   = 100,
    student_ckpt: str = r"checkpoints\dinov2_student\student_best.pt",
    save_path:  str   = r"checkpoints\dinov2_student\student_clip_best.pt",
):
    """
    Dual-distillation fine-tuning of StudentEncoder with CLIP teacher.

    Args:
        hdf5_dir:     RECON HDF5 directory
        epochs:       training epochs
        lr:           learning rate (lower than original — fine-tuning)
        batch_size:   frames per batch
        alpha:        CLIP loss weight (0.3 = 30% CLIP, 70% DINOv2 preserve)
        max_files:    limit HDF5 files processed per epoch (None = all)
        log_every:    log every N batches
        student_ckpt: existing StudentEncoder checkpoint to fine-tune from
        save_path:    output checkpoint path
    """
    print("\nSprint 6 — CLIP Dual-Distillation")
    print("=" * 50)
    print(f"  Student ckpt : {student_ckpt}")
    print(f"  Save path    : {save_path}")
    print(f"  HDF5 dir     : {hdf5_dir}")
    print(f"  Epochs       : {epochs}")
    print(f"  LR           : {lr}")
    print(f"  Alpha (CLIP) : {alpha}")
    print(f"  Alpha (DINOv2 preserve): {1 - alpha}")
    print("=" * 50)

    # ── Student encoder (trainable) ───────────────────────────────────────────
    student = StudentEncoder().to(DEVICE)
    if Path(student_ckpt).exists():
        ckpt = torch.load(student_ckpt, map_location="cpu", weights_only=False)
        sd   = ckpt.get("model", ckpt.get("state_dict", ckpt))
        student.load_state_dict(sd, strict=False)
        print(f"  StudentEncoder loaded from {student_ckpt}")
    else:
        print(f"  ⚠️  StudentEncoder checkpoint not found — using random weights")
    student.train()

    # ── Frozen anchor (DINOv2 proxy — prevents catastrophic forgetting) ───────
    anchor = copy.deepcopy(student)
    for p in anchor.parameters():
        p.requires_grad_(False)
    anchor.eval()
    print(f"  Anchor (frozen DINOv2 proxy): StudentEncoder frozen copy")

    # ── CLIP teacher (frozen) ─────────────────────────────────────────────────
    print(f"  Loading CLIP ViT-B/32...")
    clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    # CLIP projects to 512-D; student outputs 128-D
    # Bridge: linear projection 512 → 128 (trainable alongside student)
    clip_bridge = nn.Linear(512, 128, bias=False).to(DEVICE)
    nn.init.orthogonal_(clip_bridge.weight)
    print(f"  CLIP bridge: Linear(512 → 128)")

    clip_preprocess = make_clip_preprocess(None)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    params = list(student.parameters()) + list(clip_bridge.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_loss = float("inf")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    for ep in range(epochs):
        ep_losses = []
        ep_l_preserve = []
        ep_l_clip = []
        step = 0
        t_ep = time.time()

        for batch in iter_frames(hdf5_dir, max_files, batch_size):
            B = batch.shape[0]
            batch = batch.to(DEVICE)

            # ── Student forward pass ──────────────────────────────────────────
            student_out = student(batch)       # (B, 128) unit-normalised

            # ── L_preserve: maintain DINOv2 knowledge ────────────────────────
            with torch.no_grad():
                anchor_out = anchor(batch)     # (B, 128) frozen reference
            L_preserve = cosine_loss(student_out, anchor_out.detach())

            # ── L_clip: align with CLIP image features ────────────────────────
            # Preprocess for CLIP (different normalisation)
            clip_batch = torch.stack([
                clip_preprocess(batch[i]) for i in range(B)
            ]).to(DEVICE)

            with torch.no_grad():
                clip_feats = clip_model.encode_image(clip_batch).float()  # (B, 512)
                clip_feats = F.normalize(clip_feats, dim=-1)

            clip_proj = F.normalize(clip_bridge(clip_feats), dim=-1)  # (B, 128)
            L_clip = cosine_loss(student_out, clip_proj.detach())

            # ── Total loss ────────────────────────────────────────────────────
            L_total = (1 - alpha) * L_preserve + alpha * L_clip

            if not torch.isfinite(L_total):
                continue

            opt.zero_grad()
            L_total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            ep_losses.append(L_total.item())
            ep_l_preserve.append(L_preserve.item())
            ep_l_clip.append(L_clip.item())
            step += 1

            if step % log_every == 0:
                print(
                    f"[ep{ep:02d} s{step:05d}] "
                    f"loss={L_total.item():.4f}  "
                    f"L_preserve={L_preserve.item():.4f}  "
                    f"L_clip={L_clip.item():.4f}  "
                    f"lr={opt.param_groups[0]['lr']:.2e}"
                )

        scheduler.step()
        mean_loss     = np.mean(ep_losses)     if ep_losses     else float("inf")
        mean_preserve = np.mean(ep_l_preserve) if ep_l_preserve else 0.0
        mean_clip     = np.mean(ep_l_clip)     if ep_l_clip     else 0.0
        ep_time       = time.time() - t_ep

        print(
            f"Epoch {ep:02d}  loss={mean_loss:.4f}  "
            f"L_preserve={mean_preserve:.4f}  L_clip={mean_clip:.4f}  "
            f"({ep_time:.1f}s)"
        )

        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save({
                "epoch":        ep,
                "loss":         best_loss,
                "alpha":        alpha,
                "model":        student.state_dict(),
                "clip_bridge":  clip_bridge.state_dict(),
            }, save_path)
            print(f"  → saved (loss={best_loss:.4f})")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoint: {save_path}")
    print(
        f"\nNext step: run clip_particle_alignment_test.py "
        f"--cwm-ckpt checkpoints\\cwm\\cwm_best.pt "
        f"--student-ckpt {save_path}"
    )
    return best_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sprint 6: CLIP Dual-Distillation for StudentEncoder"
    )
    parser.add_argument("--hdf5-dir",   default="recon_data/recon_release")
    parser.add_argument("--epochs",     type=int,   default=15)
    parser.add_argument("--lr",         type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--alpha",      type=float, default=0.3,
                        help="CLIP loss weight (default 0.3 = 30%% CLIP)")
    parser.add_argument("--max-files",  type=int,   default=None)
    parser.add_argument("--log-every",  type=int,   default=100)
    parser.add_argument("--student-ckpt",
                        default=r"checkpoints\dinov2_student\student_best.pt")
    parser.add_argument("--save-path",
                        default=r"checkpoints\dinov2_student\student_clip_best.pt")
    args = parser.parse_args()

    train(
        hdf5_dir    = args.hdf5_dir,
        epochs      = args.epochs,
        lr          = args.lr,
        batch_size  = args.batch_size,
        alpha       = args.alpha,
        max_files   = args.max_files,
        log_every   = args.log_every,
        student_ckpt = args.student_ckpt,
        save_path   = args.save_path,
    )
