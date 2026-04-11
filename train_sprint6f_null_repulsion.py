"""
train_sprint6f_null_repulsion.py — Sprint 6f: Mined Hard Negatives
==================================================================
Continues from Sprint 6c checkpoint (best L_clip=0.6692, epoch 8)
with null_weight=5.0 and navigation-adjacent hard negatives.

Sprint 6c/6d diagnosis:
  - 9/9 semantic STRONG confirmed (2.2–4.7×)
  - Null raw similarities: 0.166–0.271 (overlapping semantic: 0.244–0.331)
  - L_null peak at null_weight=0.5: 0.0004 — four orders of magnitude below L_clip
  - Root cause: L_null too small to move embeddings

Sprint 6f changes vs 6e:
  1. Mined null texts: automatic hard negative mining (Sprint 9a)
     Outdoor basketball court (0.273), park (0.252), sidewalk (0.247)...
     These overlap with navigation queries in CLIP space.
  2. null_weight: 5.0 maintained
  3. Start from Sprint 6c checkpoint
  4. 3 epochs only

Pass criteria:
  - Semantic: 9/9 STRONG (≥1.5× ratio) — must be preserved
  - Nulls: 3/3 correct (top-k similarity < threshold 0.235)
  - L_clip degradation < 0.05 vs Sprint 6c (alignment must hold)

Run:
    python train_sprint6e_null_repulsion.py \
        --hdf5-dir recon_data\recon_release \
        --resume-ckpt checkpoints\dinov2_student\student_dualhead_nce_best.pt \
        --save-path checkpoints\dinov2_student\student_dualhead_6e_best.pt \
        --epochs 3 --null-weight 5.0

Author: John Taylor — github.com/taylorjohn
Date:   2026-04-05
"""

import argparse
import glob
import io
import random
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    import clip as clip_module
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: clip not installed. Run: pip install git+https://github.com/openai/CLIP.git")

from train_mvtec import StudentEncoder

DEVICE = torch.device("cpu")
IMG_SIZE = 224

# ── Hard negatives: navigation-adjacent but NOT navigation ────────────────────
# These share visual vocabulary (outdoor, movement, vehicles) but are not
# robot navigation commands. Harder than generic nulls (cake, Shakespeare).
NULL_TEXTS = [
    # Mined hard negatives — Sprint 9a (stable across two mining runs)
    # These score 0.241-0.274 vs RECON mean embedding — genuinely confusable
    "an outdoor basketball court",           # 0.273 — highest, consistently #1
    "a park with trees and grass",           # 0.252
    "a sidewalk next to a building",         # 0.247
    "a bus stop on the side of the road",    # 0.247
    "a school gymnasium",                    # 0.241
    "a residential street with houses",      # 0.247
    "a delivery truck making a stop",        # 0.241
    "a satellite in orbit around earth",     # 0.246
    "a tank on a military training ground",  # 0.232
    "a loading dock behind a warehouse",     # 0.227
    # Original easy negatives — keep as anchors
    "a recipe for chocolate cake with frosting",   # 0.175
    "sonnets by william shakespeare",              # ~0.18
]

# ── Semantic queries for live monitoring ─────────────────────────────────────
MONITOR_QUERIES = [
    "robot moving fast forward",
    "robot turning sharply",
    "robot stopped or stationary",
]
MONITOR_NULLS = [
    "a recipe for chocolate cake",
    "a car drives down the highway at high speed",  # hard negative
    "the quick brown fox",
]


# ── Dual-head architecture (same as Sprint 6c/6d) ─────────────────────────────

class SemanticHead(nn.Module):
    def __init__(self, d_in: int = 256, d_out: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.ReLU(),
            nn.Linear(256, d_out),
            nn.LayerNorm(d_out),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class CLIPBridge(nn.Module):
    def __init__(self, d_clip: int = 512, d_out: int = 128):
        super().__init__()
        self.proj = nn.Linear(d_clip, d_out, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), dim=-1)


# ── InfoNCE loss ──────────────────────────────────────────────────────────────

def info_nce(vis_emb: torch.Tensor, txt_emb: torch.Tensor,
             temperature: float = 0.07) -> torch.Tensor:
    """
    Symmetric InfoNCE between visual and text embeddings.
    vis_emb: (B, D), txt_emb: (B, D), both unit-normalised.
    """
    logits = vis_emb @ txt_emb.T / temperature
    labels = torch.arange(len(vis_emb), device=vis_emb.device)
    loss_v = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    return (loss_v + loss_t) / 2


# ── Null repulsion loss ───────────────────────────────────────────────────────

def null_repulsion(vis_emb: torch.Tensor,
                   null_embs: torch.Tensor,
                   margin: float = 0.25) -> torch.Tensor:
    """
    Push visual embeddings away from null text embeddings.
    null_embs: (N_null, D) pre-encoded null text embeddings.

    Uses tighter margin (0.25 vs 0.30 in Sprint 6d) since we have
    harder negatives that sit closer to the navigation cluster.
    Returns mean relu(sim - margin) across all vis × null pairs.
    """
    # vis_emb: (B, D), null_embs: (N, D)
    sims = vis_emb @ null_embs.T            # (B, N)
    return F.relu(sims - margin).mean()


# ── Data loader ───────────────────────────────────────────────────────────────

def load_frame(hf: h5py.File) -> torch.Tensor:
    imgs = hf["images"]["rgb_left"]
    idx = random.randint(0, len(imgs) - 1)
    img = Image.open(io.BytesIO(bytes(imgs[idx]))).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)   # (3, H, W)


def iter_batches(hdf5_dir: str, batch_size: int = 32):
    files = sorted(glob.glob(f"{hdf5_dir}/*.hdf5"))
    random.shuffle(files)
    buf = []
    for path in files:
        try:
            with h5py.File(path, "r") as hf:
                if "images" not in hf or "rgb_left" not in hf["images"]:
                    continue
                for _ in range(4):
                    buf.append(load_frame(hf))
                    if len(buf) == batch_size:
                        yield torch.stack(buf)
                        buf = []
        except Exception:
            pass
    if buf:
        yield torch.stack(buf)


# ── Alignment monitor ─────────────────────────────────────────────────────────

@torch.no_grad()
def monitor_alignment(backbone, semantic_head, clip_bridge,
                      clip_model, hdf5_dir, n_files=10):
    """Quick alignment check — runs in ~10 seconds."""
    files = sorted(glob.glob(f"{hdf5_dir}/*.hdf5"))[:n_files]
    vis_embs = []
    for path in files:
        try:
            with h5py.File(path, "r") as hf:
                if "images" not in hf:
                    continue
                frame = load_frame(hf).unsqueeze(0)
                z = backbone(frame)            # (1, 128)
                z256 = F.pad(z, (0, 128))          # (1, 256) — zero-pad to match 6c
                vis_embs.append(semantic_head(z256).squeeze(0))
        except Exception:
            pass

    if not vis_embs:
        return

    vis_stack = torch.stack(vis_embs)    # (N, 128)
    baseline  = vis_stack.mean(0, keepdim=True)

    print(f"\n  {'Query':<42} {'TopK':>6} {'Base':>6} {'Ratio':>7} {'Result'}")
    print("  " + "─" * 72)

    all_queries = MONITOR_QUERIES + MONITOR_NULLS
    for i, query in enumerate(all_queries):
        tok    = clip_module.tokenize([query]).to(DEVICE)
        txt_e  = clip_model.encode_text(tok).float()
        txt_e  = F.normalize(txt_e, dim=-1)
        sem_e  = F.normalize(clip_bridge(txt_e), dim=-1)

        sims   = (vis_stack @ sem_e.T).squeeze(-1)
        k      = min(30, len(sims))
        topk   = float(sims.topk(k).values.mean())
        base_s = float((baseline @ sem_e.T).squeeze())
        ratio  = topk / (abs(base_s) + 1e-6) if base_s != 0 else float("inf")

        is_null   = i >= len(MONITOR_QUERIES)
        if is_null:
            verdict = "✓ NULL OK" if topk < 0.235 else "LEAKING"
        else:
            verdict = "STRONG" if ratio >= 1.5 else "WEAK"

        tag = "—" if is_null else ""
        print(f"  {query[:40]:<42} {topk:>6.3f} {base_s:>6.3f} {ratio:>6.2f}x  {verdict}")

    print()


# ── Training loop ──────────────────────────────────────────────────────────────

def train(args):
    print(f"\nSprint 6f — Mined Hard Negatives")
    print("=" * 60)
    print(f"  Resume:      {args.resume_ckpt}")
    print(f"  Save:        {args.save_path}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Null weight: {args.null_weight}  (was 0.5 in Sprint 6d)")
    print(f"  Null texts:  {len(NULL_TEXTS)} ({3} generic + {len(NULL_TEXTS)-3} hard negatives)")
    print(f"  Margin:      {args.margin} (was 0.30 in Sprint 6d)")
    print("=" * 60)

    # ── Load models ─────────────────────────────────────────────────────
    backbone = StudentEncoder().to(DEVICE)
    ckpt = torch.load(args.resume_ckpt, map_location="cpu", weights_only=False)

    semantic_head = SemanticHead().to(DEVICE)
    clip_bridge   = CLIPBridge().to(DEVICE)

    # Load head weights from checkpoint
    if "semantic_head" in ckpt:
        missing, unexpected = semantic_head.load_state_dict(ckpt["semantic_head"], strict=False)
        if missing:
            print(f"  SemanticHead: reinitialised layers (size mismatch): {missing}")
        if unexpected:
            print(f"  SemanticHead: unexpected keys: {unexpected}")
        # Remap key: Sprint 6c saved CLIPBridge as bare Linear ("weight")
        # Sprint 6e wraps it as self.proj, so expects "proj.weight"
        cb_sd = ckpt["clip_bridge"]
        if "weight" in cb_sd and "proj.weight" not in cb_sd:
            cb_sd = {"proj." + k: v for k, v in cb_sd.items()}
        clip_bridge.load_state_dict(cb_sd)
        print(f"  Heads loaded from ep{ckpt.get('epoch','?')}, "
              f"L_clip={ckpt.get('loss', '?'):.4f}")
    else:
        print("  Warning: no head weights in checkpoint — starting fresh")

    # Freeze backbone
    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone.eval()

    # Load and freeze CLIP
    if not CLIP_AVAILABLE:
        raise ImportError("Install CLIP: pip install git+https://github.com/openai/CLIP.git")
    clip_model, _ = clip_module.load("ViT-B/32", device=DEVICE)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    # Pre-encode null texts
    with torch.no_grad():
        null_embs = []
        for txt in NULL_TEXTS:
            tok = clip_module.tokenize([txt]).to(DEVICE)
            e   = clip_model.encode_text(tok).float()
            e   = F.normalize(e, dim=-1)
            null_embs.append(clip_bridge(e).squeeze(0))
        null_embs = torch.stack(null_embs)   # (N_null, 128)
    print(f"  Null texts pre-encoded: {len(NULL_TEXTS)}")

    trainable = list(semantic_head.parameters()) + list(clip_bridge.parameters())
    print(f"  Trainable params: {sum(p.numel() for p in trainable):,} (heads only)")

    # ── Optimiser ────────────────────────────────────────────────────────
    # Lower LR than 6c since we're fine-tuning, not training from scratch
    optimizer = AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    n_batches_estimate = 200 * args.hdf5_files_estimate
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * n_batches_estimate,
                                   eta_min=args.lr * 0.01)

    best_lclip = float("inf")
    save_path  = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Monitor before training
    print("\n  Pre-training alignment check:")
    monitor_alignment(backbone, semantic_head, clip_bridge,
                      clip_model, args.hdf5_dir, n_files=8)

    # ── Training ─────────────────────────────────────────────────────────
    for epoch in range(args.epochs):
        semantic_head.train()
        clip_bridge.train()

        epoch_lclip, epoch_lnull, epoch_ltotal = [], [], []
        step = 0

        for batch_imgs in iter_batches(args.hdf5_dir, args.batch_size):
            batch_imgs = batch_imgs.to(DEVICE)
            B = len(batch_imgs)

            with torch.no_grad():
                z_vis = backbone(batch_imgs)      # (B, 128)
            # Pad to 256-D to match Sprint 6c SemanticHead architecture
            z_vis256 = F.pad(z_vis, (0, 128))    # (B, 256)

            # SemanticHead embedding
            vis_emb = semantic_head(z_vis256)    # (B, 128) unit-norm

            # CLIP text embeddings for the batch — use random nav phrases
            nav_phrases = [
                random.choice([
                    "robot moving forward", "robot driving",
                    "robot turning left", "robot turning right",
                    "outdoor robot navigation", "robot moving slowly",
                    "robot at high speed", "robot going straight",
                ]) for _ in range(B)
            ]
            with torch.no_grad():
                tok     = clip_module.tokenize(nav_phrases).to(DEVICE)
                txt_raw = clip_model.encode_text(tok).float()
                txt_raw = F.normalize(txt_raw, dim=-1)
            txt_emb = F.normalize(clip_bridge(txt_raw), dim=-1)

            # Update null_embs with current bridge weights
            with torch.no_grad():
                null_embs_current = []
                for txt in NULL_TEXTS:
                    tok_n = clip_module.tokenize([txt]).to(DEVICE)
                    e_n   = clip_model.encode_text(tok_n).float()
                    e_n   = F.normalize(e_n, dim=-1)
                    null_embs_current.append(clip_bridge(e_n).squeeze(0))
                null_embs_live = torch.stack(null_embs_current).detach()

            L_clip = info_nce(vis_emb, txt_emb)
            L_null = null_repulsion(vis_emb, null_embs_live, margin=args.margin)
            L_total = L_clip + args.null_weight * L_null

            optimizer.zero_grad()
            L_total.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()

            epoch_lclip.append(float(L_clip))
            epoch_lnull.append(float(L_null))
            epoch_ltotal.append(float(L_total))
            step += 1

            if step % args.log_every == 0:
                print(f"[ep{epoch:02d} s{step:05d}] "
                      f"L_clip={float(L_clip):.4f}  "
                      f"L_null={float(L_null):.4f}  "
                      f"lr={scheduler.get_last_lr()[0]:.2e}")

        mean_lclip = float(np.mean(epoch_lclip))
        mean_lnull = float(np.mean(epoch_lnull))
        print(f"Epoch {epoch:02d}  L_clip={mean_lclip:.4f}  "
              f"L_null={mean_lnull:.4f}  "
              f"L_null_max={max(epoch_lnull):.4f}")

        if mean_lclip < best_lclip:
            best_lclip = mean_lclip
            torch.save({
                "epoch":         epoch,
                "loss":          mean_lclip,
                "l_null_mean":   mean_lnull,
                "semantic_head": semantic_head.state_dict(),
                "clip_bridge":   clip_bridge.state_dict(),
            }, save_path)
            print(f"  -> saved (L_clip={mean_lclip:.4f})")

        # Alignment check every epoch
        print(f"\n  Alignment check after epoch {epoch}:")
        monitor_alignment(backbone, semantic_head, clip_bridge,
                          clip_model, args.hdf5_dir, n_files=8)

    print(f"\nTraining complete. Best L_clip: {best_lclip:.4f}")
    print(f"Checkpoint: {save_path}")
    print(f"Pass criteria:")
    print(f"  Semantic: 9/9 STRONG (ratio ≥ 1.5×)")
    print(f"  Nulls: 3/3 correct (top-k < 0.235)")
    print(f"Next: python test_dualhead_alignment.py --dualhead-ckpt {save_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sprint 6f — Mined Hard Negatives")
    parser.add_argument("--hdf5-dir",     required=True)
    parser.add_argument("--resume-ckpt",  default="checkpoints/dinov2_student/student_dualhead_nce_best.pt",
                        help="Sprint 6c checkpoint (best alignment baseline)")
    parser.add_argument("--save-path",    default="checkpoints/dinov2_student/student_dualhead_6f_best.pt")
    parser.add_argument("--epochs",       type=int,   default=3)
    parser.add_argument("--lr",           type=float, default=5e-5,
                        help="Lower than 6c — fine-tuning not training")
    parser.add_argument("--batch-size",   type=int,   default=32)
    parser.add_argument("--null-weight",  type=float, default=5.0)
    parser.add_argument("--margin",       type=float, default=0.25,
                        help="Tighter margin (0.25 vs 0.30) for hard negatives")
    parser.add_argument("--log-every",    type=int,   default=200)
    parser.add_argument("--hdf5-files-estimate", type=int, default=196,
                        help="For scheduler T_max estimation")
    args = parser.parse_args()
    train(args)
