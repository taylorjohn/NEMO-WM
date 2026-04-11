"""
train_recon_temporal.py  —  CORTEX CWM Sprint 2
================================================
Cross-temporal contrastive training for RECON outdoor navigation.

Trains a lightweight TemporalHead on top of the FROZEN CWM encoder to
produce a quasimetric embedding: closer embeddings = reachable in fewer steps.

Architecture:
  CortexWorldModel encoder (FROZEN after Sprint 1)
      ↓  (B, K=16, 128) particles
  TemporalHead (~33K params, trained here)
      ↓  (B, 64) quasimetric embedding
  InfoNCE loss over (z_t, z_{t+k}) pairs

Cross-temporal pairing (from Seoul World Model arXiv:2603.15583):
  Positive pair: (frame_t, frame_{t+k}) from same trajectory, k ∈ [1,8]
  Negative pair: frames from different trajectories
  This forces geometry-stable representations under forward motion.

Success criterion: AUROC > 0.70 on RECON quasimetric eval.

Usage:
    # After Sprint 1 training is complete
    python train_recon_temporal.py \
        --cwm-ckpt checkpoints\cwm\cwm_best.pt \
        --hdf5-dir recon_data\recon_release \
        --epochs 20
"""

import io
import time
import random
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from neuromodulator import NeuromodulatorState
from cwm_neuro_reward import NeuromodulatedCWMLoss
from train_cwm import CortexWorldModel, DOMAIN_IDS, MAX_ACTION_DIM


# ═══════════════════════════════════════════════════════════════════════════
# TemporalHead
# ═══════════════════════════════════════════════════════════════════════════

class TemporalHead(nn.Module):
    """
    Lightweight quasimetric head trained on frozen CWM encoder particles.

    Maps K particle embeddings → 64-D quasimetric embedding.
    Trained with InfoNCE: positive = same trajectory t and t+k,
    negative = different trajectories.

    ~33K params. Frozen encoder — only this head is updated.

    Design notes:
    - Pool particles first (mean) → single 128-D vector
    - Project to 64-D quasimetric space
    - L2-normalise output (cosine similarity = quasimetric distance)
    - Temperature-scaled InfoNCE (τ=0.07 standard)
    """

    def __init__(self, d_model: int = 128, embed_dim: int = 64):
        super().__init__()
        self.pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.proj = nn.Sequential(
            nn.Linear(d_model, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(self, particles: torch.Tensor) -> torch.Tensor:
        """
        particles: (B, K, d_model)
        Returns:   (B, embed_dim) L2-normalised quasimetric embedding
        """
        # Mean pool over K particles
        x = particles.mean(dim=1)      # (B, d_model)
        x = self.pool(x)               # (B, d_model)
        z = self.proj(x)               # (B, embed_dim)
        return F.normalize(z, dim=-1)  # unit sphere

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════════════
# InfoNCE loss
# ═══════════════════════════════════════════════════════════════════════════

def infonce_loss(
    z_t:       torch.Tensor,   # (B, embed_dim)
    z_tk:      torch.Tensor,   # (B, embed_dim)
    temperature: float = 0.07,
) -> Tuple[torch.Tensor, float]:
    """
    Symmetric InfoNCE (NT-Xent) loss.

    For each anchor z_t[i], its positive is z_tk[i] (same trajectory, k steps
    ahead). All other z_tk[j≠i] in the batch are negatives.

    Returns loss and top-1 accuracy (fraction of anchors that rank their
    positive above all negatives — proxy for AUROC before full eval).
    """
    B = z_t.shape[0]

    # Similarity matrix: (B, B) — rows are anchors, cols are keys
    sim = torch.mm(z_t, z_tk.T) / temperature   # (B, B)

    # Labels: diagonal is the positive pair
    labels = torch.arange(B, device=z_t.device)

    # Symmetric: loss in both directions
    loss = (F.cross_entropy(sim, labels) +
            F.cross_entropy(sim.T, labels)) / 2.0

    # Top-1 accuracy
    acc = (sim.argmax(dim=1) == labels).float().mean().item()

    return loss, acc


# ═══════════════════════════════════════════════════════════════════════════
# Cross-temporal RECON Dataset
# ═══════════════════════════════════════════════════════════════════════════

class RECONCrossTemporalDataset(Dataset):
    """
    Returns (frame_t, frame_{t+k}, k, gps_t, gps_{t+k}) tuples.

    Positive pairs come from the same trajectory with temporal gap k ∈ [1, k_max].
    Used to train the TemporalHead with InfoNCE.

    Cross-temporal pairing (Seoul World Model):
    Forces the encoder to produce geometry-stable representations —
    frames within the same trajectory must embed close together
    regardless of short-term appearance changes (lighting, blur, etc.)
    """

    def __init__(
        self,
        hdf5_dir:  str,
        k_max:     int = 8,
        max_files: int = None,
        img_size:  int = 224,
    ):
        self.k_max    = k_max
        self.img_size = img_size

        files = sorted(Path(hdf5_dir).glob("jackal_2019-*.hdf5"))
        if max_files:
            files = files[:max_files]

        # Index: (filepath, t_anchor) — t+k must be valid
        self.samples: List[Tuple[str, int]] = []
        for f in files:
            with h5py.File(f, 'r') as hf:
                n = hf['images']['rgb_left'].shape[0]
                for t in range(n - k_max):
                    self.samples.append((str(f), t))

    def _decode(self, jpeg_bytes: bytes) -> torch.Tensor:
        img = Image.open(io.BytesIO(jpeg_bytes)).convert('RGB')
        img = img.resize((self.img_size, self.img_size))
        t   = torch.from_numpy(np.array(img)).float() / 255.0
        return t.permute(2, 0, 1)   # (3, H, W)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        path, t = self.samples[idx]
        k = random.randint(1, self.k_max)

        with h5py.File(path, 'r') as hf:
            frame_t  = self._decode(bytes(hf['images']['rgb_left'][t]))
            frame_tk = self._decode(bytes(hf['images']['rgb_left'][t]))

            gps_t  = torch.tensor(
                hf['observations'][t][:2]
                if 'observations' in hf else [0., 0.],
                dtype=torch.float32
            )
            gps_tk = torch.tensor(
                hf['observations'][t + k][:2]
                if 'observations' in hf else [0., 0.],
                dtype=torch.float32
            )

        return {
            "frame_t":  frame_t,
            "frame_tk": frame_tk,
            "k":        torch.tensor(k, dtype=torch.float32),
            "gps_t":    gps_t,
            "gps_tk":   gps_tk,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════

def train_temporal(
    cwm_ckpt:   str   = r"checkpoints\cwm\cwm_best.pt",
    hdf5_dir:   str   = "recon_data/recon_release",
    n_epochs:   int   = 20,
    batch_size: int   = 32,
    base_lr:    float = 1e-3,
    k_max:      int   = 8,
    embed_dim:  int   = 64,
    max_files:  int   = None,
    save_dir:   str   = r"checkpoints\cwm",
    log_every:  int   = 50,
    device_str: str   = "cpu",
):
    device = torch.device(device_str)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # ── Load frozen CWM encoder (Sprint 1 output) ─────────────────────────
    cwm = CortexWorldModel(d_model=128, K=16).to(device)
    if Path(cwm_ckpt).exists():
        ckpt = torch.load(cwm_ckpt, map_location=device)
        cwm.load_state_dict(ckpt["model"])
        print(f"Loaded CWM from {cwm_ckpt} (epoch {ckpt['epoch']}, "
              f"loss {ckpt['loss']:.4f})")
    else:
        print(f"WARNING: {cwm_ckpt} not found — using random CWM weights")

    # Freeze encoder — only TemporalHead trains
    for p in cwm.parameters():
        p.requires_grad_(False)
    cwm.eval()

    # ── TemporalHead (the only thing that trains here) ─────────────────────
    head = TemporalHead(d_model=128, embed_dim=embed_dim).to(device)
    print(f"TemporalHead: {head.total_params():,} params")

    optimizer = torch.optim.AdamW(head.parameters(), lr=base_lr,
                                   weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs
    )

    # ── Neuromodulators (for rho temperature guidance) ─────────────────────
    neuro = NeuromodulatorState(session_start=time.time(),
                                 ado_saturate_hours=float(n_epochs))

    # ── Dataset ────────────────────────────────────────────────────────────
    dataset = RECONCrossTemporalDataset(hdf5_dir, k_max=k_max,
                                         max_files=max_files)
    loader  = DataLoader(dataset, batch_size=batch_size,
                          shuffle=True, num_workers=2)
    print(f"Dataset: {len(dataset)} cross-temporal pairs")

    # Mock student encoder (replace with NPU path in production)
    student_mock = nn.Linear(3 * 224 * 224, 256).to(device)

    best_loss = float("inf")
    global_step = 0

    for epoch in range(n_epochs):
        head.train()
        epoch_losses, epoch_accs = [], []

        for batch in loader:
            frame_t  = batch["frame_t"].to(device)
            frame_tk = batch["frame_tk"].to(device)
            B = frame_t.shape[0]

            # ── Encode both frames (frozen CWM) ───────────────────────────
            with torch.no_grad():
                z_t  = student_mock(frame_t.reshape(B, -1))
                z_tk = student_mock(frame_tk.reshape(B, -1))
                particles_t,  _, _, _ = cwm.encode(z_t)
                particles_tk, _, _, _ = cwm.encode(z_tk)

            # ── Quasimetric embeddings (TemporalHead trains) ───────────────
            emb_t  = head(particles_t)    # (B, 64)
            emb_tk = head(particles_tk)   # (B, 64)

            # ── InfoNCE loss ───────────────────────────────────────────────
            loss, acc = infonce_loss(emb_t, emb_tk,
                                      temperature=head.temperature.item())

            # ── Neuromodulator update (rho from Allen, DA from prediction) ─
            signals = neuro.update(
                z_pred           = particles_t.mean(dim=1),
                z_actual         = particles_tk.mean(dim=1),
                rho              = 0.5,
                action_magnitude = 0.0,
            )

            # DA modulation on InfoNCE loss
            # High DA: encoder outputs surprised by t+k → amplify gradient
            da_scale = 0.5 + signals["da_effective"]
            loss = loss * da_scale

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())
            epoch_accs.append(acc)
            global_step += 1

            if global_step % log_every == 0:
                print(
                    f"[ep{epoch:02d} step{global_step:04d}] "
                    f"loss={loss.item():.4f}  "
                    f"top1_acc={acc:.3f}  "
                    f"DA={signals['da']:.3f}  "
                    f"regime={signals['regime']}"
                )

        scheduler.step()
        mean_loss = np.mean(epoch_losses)
        mean_acc  = np.mean(epoch_accs)
        print(f"\nEpoch {epoch:02d}  loss={mean_loss:.4f}  "
              f"top1_acc={mean_acc:.3f}")

        if mean_loss < best_loss:
            best_loss = mean_loss
            path = Path(save_dir) / "temporal_head_best.pt"
            torch.save({
                "epoch":     epoch,
                "loss":      best_loss,
                "acc":       mean_acc,
                "head":      head.state_dict(),
                "embed_dim": embed_dim,
            }, path)
            print(f"  → Saved: {path}")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Next step: python eval_recon_auroc.py "
          f"--cwm-ckpt {cwm_ckpt} "
          f"--head-ckpt {Path(save_dir)/'temporal_head_best.pt'}")
    return head


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cwm-ckpt",   default=r"checkpoints\cwm\cwm_best.pt")
    p.add_argument("--hdf5-dir",   default="recon_data/recon_release")
    p.add_argument("--epochs",     type=int,   default=20)
    p.add_argument("--batch-size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--k-max",      type=int,   default=8)
    p.add_argument("--embed-dim",  type=int,   default=64)
    p.add_argument("--max-files",  type=int,   default=None)
    p.add_argument("--save-dir",   default=r"checkpoints\cwm")
    args = p.parse_args()

    train_temporal(
        cwm_ckpt   = args.cwm_ckpt,
        hdf5_dir   = args.hdf5_dir,
        n_epochs   = args.epochs,
        batch_size = args.batch_size,
        base_lr    = args.lr,
        k_max      = args.k_max,
        embed_dim  = args.embed_dim,
        max_files  = args.max_files,
        save_dir   = args.save_dir,
    )
