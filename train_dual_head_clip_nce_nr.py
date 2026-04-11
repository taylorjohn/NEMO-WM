"""
train_dual_head_clip.py — Sprint 6b: Dual-Head CLIP Distillation
=================================================================
Separates physical and semantic encoding into two independent heads
on top of a permanently frozen DINOv2-distilled backbone.

Architecture:
    StudentEncoder backbone (FROZEN — DINOv2 knowledge locked)
         │
         ├── PhysicalHead (existing proj layer, already trained)
         │   → particle space, GPS, contact, neuromodulator
         │   → UNCHANGED — not trained here
         │
         └── SemanticHead (new, 256→256→128 with LayerNorm)
             → CLIP semantic space
             → trained here with L_clip only, no competing objective

    CLIPBridge: Linear(512→128)
             → maps CLIP text/image embeddings into SemanticHead space
             → trained alongside SemanticHead

Why dual-head solves the alpha problem:
    Single encoder: CLIP and DINOv2 fight over the same weights.
    Dual head: backbone frozen, heads independent, no conflict.
    No alpha tuning needed — L_preserve disappears entirely.
    Can use much higher LR (3e-4) — nothing fragile to protect.

At inference:
    Navigation:  backbone → PhysicalHead → particles → MirrorAscent
    Text goal:   CLIP text → CLIPBridge → SemanticHead space → nearest neighbor
    Visual goal: backbone → SemanticHead → CLIP-aligned embedding

Run:
    python train_dual_head_clip.py --hdf5-dir recon_data/recon_release
    python train_dual_head_clip.py --hdf5-dir recon_data/recon_release --epochs 10

Author: John Taylor — github.com/taylorjohn
Date:   2026-04-04
"""

import argparse
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


# ── SemanticHead ──────────────────────────────────────────────────────────────

class SemanticHead(nn.Module):
    """
    Lightweight semantic projection head on top of frozen backbone.
    Learns CLIP-aligned embeddings without touching physical encoding.

    Input:  256-D pooled backbone features (before PhysicalHead proj)
    Output: 128-D unit-normalised semantic embedding
    """
    def __init__(self, in_dim: int = 256, hidden_dim: int = 256,
                 out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class FrozenBackbone(nn.Module):
    """
    StudentEncoder backbone with proj layer removed — outputs pooled features.
    Completely frozen. DINOv2 knowledge is permanently preserved.
    """
    def __init__(self, student_ckpt: str):
        super().__init__()
        enc = StudentEncoder()
        if Path(student_ckpt).exists():
            sd = torch.load(student_ckpt, map_location="cpu", weights_only=False)
            sd = sd.get("model", sd.get("state_dict", sd))
            enc.load_state_dict(sd, strict=False)
        # Only keep the feature extractor — not the proj layer
        self.features = enc.features
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        # Freeze everything
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, 256) pooled feature vector — frozen."""
        with torch.no_grad():
            feats = self.features(x)        # (B, 64, H, W)
            pooled = self.pool(feats)       # (B, 64, 2, 2)
            return pooled.flatten(1)        # (B, 256)


# ── Image preprocessing ───────────────────────────────────────────────────────

def decode_frame(jpeg_bytes: bytes, size: int = 224) -> torch.Tensor:
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB").resize((size, size))
    return torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0


def clip_normalise(tensor_chw: torch.Tensor) -> torch.Tensor:
    """Normalise (3,H,W) tensor in [0,1] to CLIP's mean/std."""
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                        device=tensor_chw.device).view(3, 1, 1)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                        device=tensor_chw.device).view(3, 1, 1)
    return (tensor_chw - mean) / std


def iter_frames(hdf5_dir: str, max_files: int = None,
                batch_size: int = 32, stride: int = 5):
    """Yield (B, 3, 224, 224) batches from RECON HDF5 files."""
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
                        buf.append(decode_frame(bytes(imgs[i])))
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
    return (1 - (F.normalize(a, dim=-1) * F.normalize(b, dim=-1))
            .sum(dim=-1)).mean()


def info_nce_loss(student_emb: torch.Tensor, clip_emb: torch.Tensor,
                  temperature: float = 0.07) -> torch.Tensor:
    """
    InfoNCE contrastive loss. Forces inter-sample discrimination:
    frame i must be closer to CLIP(frame_i) than CLIP(frame_j) for j!=i.
    Prevents centroid collapse that mean cosine loss allows.
    Temperature 0.07 = standard CLIP training value.
    """
    B = student_emb.shape[0]
    s = F.normalize(student_emb, dim=-1)
    c = F.normalize(clip_emb,   dim=-1)
    logits = (s @ c.T) / temperature          # (B, B)
    labels = torch.arange(B, device=logits.device)
    loss = (F.cross_entropy(logits,   labels) +
            F.cross_entropy(logits.T, labels)) / 2
    return loss


# Null text queries that should NOT match navigation frames.
# These define the boundary of the navigation cluster.
NULL_TEXTS = [
    "a recipe for chocolate cake",
    "the quick brown fox jumps over the lazy dog",
    "sonnets by william shakespeare",
    "a quarterly financial earnings report",
    "marine biology ocean research paper",
    "ingredients for beef stew",
    "how to play the piano",
    "a history of ancient rome",
]


def null_repulsion_loss(semantic_emb: torch.Tensor,
                        null_proj: torch.Tensor,
                        margin: float = 0.3) -> torch.Tensor:
    """
    Pushes semantic embeddings away from null text projections.
    For each frame embedding, penalises cosine similarity to null text
    embeddings that exceeds the margin.

    Args:
        semantic_emb: (B, 128) frame semantic embeddings
        null_proj:    (N, 128) null text embeddings via CLIPBridge
        margin:       similarity above this is penalised (default 0.3)
    Returns:
        Scalar repulsion loss >= 0
    """
    s = F.normalize(semantic_emb, dim=-1)    # (B, 128)
    n = F.normalize(null_proj,    dim=-1)    # (N, 128)
    sim = s @ n.T                            # (B, N)
    return F.relu(sim - margin).mean()


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    hdf5_dir:     str   = "recon_data/recon_release",
    epochs:       int   = 10,
    lr:           float = 3e-4,
    batch_size:   int   = 32,
    max_files:    int   = None,
    null_weight:  float = 0.5,
    log_every:    int   = 100,
    student_ckpt: str   = r"checkpoints\dinov2_student\student_best.pt",
    save_path:    str   = r"checkpoints\dinov2_student\student_dualhead_nce_nr_best.pt",
):
    print("\nSprint 6d — Dual-Head InfoNCE + Null Repulsion")
    print("=" * 50)
    print(f"  Backbone     : {student_ckpt} (FROZEN)")
    print(f"  Save path    : {save_path}")
    print(f"  Epochs       : {epochs}")
    print(f"  LR           : {lr}  (higher safe — no backbone to protect)")
    print(f"  Heads        : SemanticHead(256→256→128) + CLIPBridge(512→128)")
    print("=" * 50)

    # ── Frozen backbone ───────────────────────────────────────────────────────
    backbone = FrozenBackbone(student_ckpt).to(DEVICE)
    print(f"  Backbone frozen: DINOv2 knowledge locked permanently")

    # ── Trainable heads ───────────────────────────────────────────────────────
    semantic_head = SemanticHead(in_dim=256, hidden_dim=256, out_dim=128).to(DEVICE)
    clip_bridge   = nn.Linear(512, 128, bias=False).to(DEVICE)
    nn.init.orthogonal_(clip_bridge.weight)
    print(f"  SemanticHead: 256→256→128 (trainable, {sum(p.numel() for p in semantic_head.parameters()):,} params)")
    print(f"  CLIPBridge:   512→128    (trainable, {clip_bridge.weight.numel():,} params)")

    # ── CLIP teacher ──────────────────────────────────────────────────────────
    print(f"  Loading CLIP ViT-B/32...")
    clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)
    print(f"  CLIP ViT-B/32 loaded and frozen")

    # ── Pre-encode null texts (frozen, computed once) ─────────────────────────
    with torch.no_grad():
        null_tokens = clip.tokenize(NULL_TEXTS).to(DEVICE)
        null_text_emb = clip_model.encode_text(null_tokens).float()
        null_text_emb = F.normalize(null_text_emb, dim=-1)  # (N, 512)
    print(f"  Null repulsion: {len(NULL_TEXTS)} null texts pre-encoded")
    print(f"  Null weight: {null_weight}")

    # ── Optimiser — can be aggressive, no backbone risk ───────────────────────
    params = list(semantic_head.parameters()) + list(clip_bridge.parameters())
    print(f"  Trainable params: {sum(p.numel() for p in params):,} (heads only)")

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_loss = float("inf")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    for ep in range(epochs):
        semantic_head.train()
        ep_losses = []
        step = 0
        t_ep = time.time()

        for batch in iter_frames(hdf5_dir, max_files, batch_size):
            B = batch.shape[0]
            batch = batch.to(DEVICE)

            # ── Backbone features (frozen, no grad) ───────────────────────────
            backbone_feats = backbone(batch)          # (B, 256)

            # ── SemanticHead forward ──────────────────────────────────────────
            semantic_emb = semantic_head(backbone_feats)   # (B, 128)

            # ── CLIP image features (frozen) ──────────────────────────────────
            clip_batch = torch.stack([
                clip_normalise(batch[i]) for i in range(B)
            ])
            with torch.no_grad():
                clip_feats = clip_model.encode_image(clip_batch).float()
                clip_feats = F.normalize(clip_feats, dim=-1)  # (B, 512)

            # Bridge CLIP → semantic space
            clip_proj = F.normalize(clip_bridge(clip_feats), dim=-1)  # (B, 128)

            # ── InfoNCE alignment loss ────────────────────────────────────────
            L_clip = info_nce_loss(semantic_emb, clip_proj)

            # ── Null repulsion loss ───────────────────────────────────────────
            null_proj_t = F.normalize(
                clip_bridge(null_text_emb), dim=-1
            )  # (N, 128)
            L_null = null_repulsion_loss(semantic_emb, null_proj_t)

            L_total = L_clip + null_weight * L_null

            if not torch.isfinite(L_total):
                continue

            opt.zero_grad()
            L_total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            ep_losses.append(L_clip.item())
            step += 1

            if step % log_every == 0:
                print(
                    f"[ep{ep:02d} s{step:05d}] "
                    f"L_clip={L_clip.item():.4f}  "
                    f"L_null={L_null.item():.4f}  "
                    f"lr={opt.param_groups[0]['lr']:.2e}"
                )

        scheduler.step()
        mean_loss = np.mean(ep_losses) if ep_losses else float("inf")
        ep_time   = time.time() - t_ep
        print(f"Epoch {ep:02d}  L_clip={mean_loss:.4f}  ({ep_time:.1f}s)")

        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save({
                "epoch":          ep,
                "loss":           best_loss,
                "semantic_head":  semantic_head.state_dict(),
                "clip_bridge":    clip_bridge.state_dict(),
                "backbone_ckpt":  student_ckpt,
                "architecture":   "dual_head",
            }, save_path)
            print(f"  -> saved (L_clip={best_loss:.4f})")

    print(f"\nTraining complete. Best L_clip: {best_loss:.4f}")
    print(f"Checkpoint: {save_path}")
    print(f"\nNote: PhysicalHead (proj layer) is UNCHANGED in student_best.pt")
    print(f"      Physical encoding results remain exactly as confirmed.")
    print(f"\nNext: run clip_particle_alignment_test.py with --dual-head flag")
    return best_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sprint 6b: Dual-Head CLIP Distillation"
    )
    parser.add_argument("--hdf5-dir",    default="recon_data/recon_release")
    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--batch-size",  type=int,   default=32)
    parser.add_argument("--max-files",   type=int,   default=None)
    parser.add_argument("--null-weight", type=float, default=0.5,
                        help="Weight for null repulsion loss")
    parser.add_argument("--log-every",   type=int,   default=100)
    parser.add_argument("--student-ckpt",
                        default=r"checkpoints\dinov2_student\student_best.pt")
    parser.add_argument("--save-path",
                        default=r"checkpoints\dinov2_student\student_dualhead_nce_nr_best.pt")
    args = parser.parse_args()

    train(
        hdf5_dir     = args.hdf5_dir,
        epochs       = args.epochs,
        lr           = args.lr,
        batch_size   = args.batch_size,
        max_files    = args.max_files,
        null_weight  = args.null_weight,
        log_every    = args.log_every,
        student_ckpt = args.student_ckpt,
        save_path    = args.save_path,
    )
