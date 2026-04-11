"""
train_cwm_action.py  —  CORTEX CWM Sprint B
============================================
Action-conditioned world model training on RECON.

Sprint B goal: make the CWM predictor genuinely action-conditioned so that
GRASP's regime_gated_plan() can use it for planning.

Architecture delta vs Sprint 3:
  Sprint 3 : (frame_t, frame_t+1) pairs — action is weak conditioning
  Sprint B  : (frame_{t-15..t}, action_t) → predict frame_{t+1} latent
              Action is load-bearing; ablation vs. no-action baseline is the
              pass criterion.

RECON HDF5 structure:
  images/rgb_left         (N, H, W, 3) uint8
  commands/linear_velocity  (N,)        float32
  commands/angular_velocity (N,)        float32
  gps/latlong             (N, 2)        float64

Pass criterion (Sprint B):
  Action-conditioned AUROC on held-out RECON > unconditioned baseline
  Target: +0.02 AUROC lift from action conditioning.

DreamerV3 tricks: symlog on prediction targets, free_bits L>=0.5, AGC λ=0.01.

Usage:
    python train_cwm_action.py \\
        --recon-dir  recon_data/recon_release \\
        --cwm-ckpt   checkpoints/cwm/cwm_multidomain_best.pt \\
        --proprio-ckpt checkpoints/cwm/proprio_6c_best.pt \\
        --epochs 20 \\
        --k-ctx 16
"""

import argparse
import time
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ═══════════════════════════════════════════════════════════════════════════
# DreamerV3 stability utilities
# ═══════════════════════════════════════════════════════════════════════════

def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric log — compresses large values, preserves sign."""
    return torch.sign(x) * torch.log1p(x.abs())


def free_bits_loss(loss: torch.Tensor, free_bits: float = 0.5) -> torch.Tensor:
    """Clamp loss from below — prevents posterior collapse."""
    return torch.clamp(loss, min=free_bits)


def agc_clip(parameters, clip_factor: float = 0.01, eps: float = 1e-3):
    """Adaptive Gradient Clipping (Brock et al. 2021)."""
    for p in parameters:
        if p.grad is None:
            continue
        p_norm  = p.detach().norm(2)
        g_norm  = p.grad.detach().norm(2)
        max_g   = clip_factor * p_norm.clamp(min=eps)
        if g_norm > max_g:
            p.grad.mul_(max_g / g_norm)


# ═══════════════════════════════════════════════════════════════════════════
# Action encoder
# ═══════════════════════════════════════════════════════════════════════════

class ActionEncoder(nn.Module):
    """
    Maps (linear_vel, angular_vel) → d_model embedding.

    Two-layer MLP with LayerNorm output — compatible with particle space.
    Unimix (1%) prevents degenerate action representations early in training.
    """

    def __init__(self, action_dim: int = 2, d_model: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.GELU(),
            nn.Linear(64, d_model),
            nn.LayerNorm(d_model),
        )
        self.d_model    = d_model
        self.action_dim = action_dim

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        """
        action: (B, action_dim) — normalised to [-1, 1] before calling.
        Returns: (B, d_model)
        """
        # Unimix 1%: blend with uniform to avoid saturated embeddings
        action = 0.99 * action + 0.01 * torch.zeros_like(action)
        return self.net(action)


# ═══════════════════════════════════════════════════════════════════════════
# Action-conditioned predictor head
# ═══════════════════════════════════════════════════════════════════════════

class ActionConditionedPredictor(nn.Module):
    """
    Injects action embedding into particle states before cwm.predict().

    Use inject(particles, action_emb) to get modified particles,
    then pass them to cwm.predict() normally — all cwm.predict() kwargs
    (context_h, domain_id, regime, ach) flow through unchanged.

    Gate starts at zero → Sprint 3 checkpoint behaviour preserved at ep0.
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(d_model))

    def inject(
        self,
        particles:  torch.Tensor,   # (B, K, d_model)
        action_emb: torch.Tensor,   # (B, d_model)
    ) -> torch.Tensor:
        """Returns action-conditioned particles (B, K, d_model)."""
        g = torch.sigmoid(self.gate)        # (d_model,)
        a = action_emb.unsqueeze(1)         # (B, 1, d_model)
        return particles + g * a


# ═══════════════════════════════════════════════════════════════════════════
# RECON action-conditioned dataset
# ═══════════════════════════════════════════════════════════════════════════

class RECONActionDataset(Dataset):
    """
    Returns k consecutive frames + the action at frame k as a sequence.

    Each sample:
        frames  : (k, 3, H, W)   float32 [0, 1]
        action  : (2,)            float32 [linear_vel, angular_vel] normalised
        frame_t1: (3, H, W)       float32 — target frame (t+1)
        gps     : (2,)            float64

    Normalisation:
        linear_vel  ∈ [-max_lin, max_lin]  → [-1, 1]
        angular_vel ∈ [-max_ang, max_ang]  → [-1, 1]
    """

    def __init__(
        self,
        hdf5_paths: list,
        k: int = 16,
        img_size: int = 64,
        max_lin: float = 2.0,
        max_ang: float = 3.14,
        max_files: Optional[int] = None,
    ):
        self.k       = k
        self.img_size = img_size
        self.max_lin  = max_lin
        self.max_ang  = max_ang

        if max_files is not None:
            hdf5_paths = hdf5_paths[:max_files]

        self.samples = []   # list of (path, start_idx)
        n_skipped = 0
        for p in hdf5_paths:
            try:
                with h5py.File(p, "r") as f:
                    if "images/rgb_left" not in f:
                        n_skipped += 1
                        continue
                    n = f["images/rgb_left"].shape[0]
                    if n < k + 1:
                        n_skipped += 1
                        continue
                    for i in range(n - k):
                        self.samples.append((str(p), i))
            except Exception as e:
                n_skipped += 1
                print(f"  WARNING: skipping {p}: {e}")

        if n_skipped:
            print(f"  ({n_skipped} files skipped — too short or missing key)")
        print(f"RECONActionDataset: {len(self.samples):,} samples "
              f"from {len(hdf5_paths)} files (k={k})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, start = self.samples[idx]
        k = self.k

        with h5py.File(path, "r") as f:
            # Context frames: t-k+1 … t
            frames_raw = f["images/rgb_left"][start : start + k]       # (k,H,W,3)
            # Target frame: t+1
            frame_t1   = f["images/rgb_left"][start + k]               # (H,W,3)
            # Action at frame t (command that transitions t → t+1)
            lin_vel    = float(f["commands/linear_velocity"][start + k - 1])
            ang_vel    = float(f["commands/angular_velocity"][start + k - 1])
            # GPS at t+1
            gps        = f["gps/latlong"][start + k].astype(np.float32)

        # Resize + normalise frames
        frames = self._proc_frames(frames_raw)        # (k, 3, H, W)
        tgt    = self._proc_frame(frame_t1)            # (3, H, W)

        # Normalise action to [-1, 1]
        action = np.array([
            np.clip(lin_vel / self.max_lin, -1.0, 1.0),
            np.clip(ang_vel / self.max_ang, -1.0, 1.0),
        ], dtype=np.float32)

        return {
            "frames":   torch.from_numpy(frames),
            "action":   torch.from_numpy(action),
            "frame_t1": torch.from_numpy(tgt),
            "gps":      torch.from_numpy(gps),
        }

    # ── helpers ──────────────────────────────────────────────────────────────

    def _proc_frame(self, img: np.ndarray) -> np.ndarray:
        """
        Accepts any of:
            (H, W, 3)  uint8  — RECON standard
            (3, H, W)  uint8  — CHW-first variant
            (H, W)     uint8  — grayscale (replicated to 3 ch)
        Returns (3, H', W') float32 [0, 1].
        """
        s = self.img_size
        img = np.asarray(img)

        # Flatten object arrays (rare HDF5 encoding)
        if img.dtype == object:
            img = np.array(img.tolist(), dtype=np.uint8)

        # Ensure at least 2-D
        if img.ndim < 2:
            return np.zeros((3, s, s), dtype=np.float32)

        # CHW-first → HWC
        if img.ndim == 3 and img.shape[0] == 3 and img.shape[1] != 3:
            img = img.transpose(1, 2, 0)          # (H, W, 3)

        # Grayscale → RGB
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)

        # Ensure 3-channel (drop alpha if RGBA)
        if img.shape[-1] == 4:
            img = img[..., :3]
        elif img.shape[-1] != 3:
            # Unknown channel count — zero frame
            return np.zeros((3, s, s), dtype=np.float32)

        h, w = img.shape[:2]
        if h == 0 or w == 0:
            return np.zeros((3, s, s), dtype=np.float32)

        yi = np.linspace(0, h - 1, s).astype(int)
        xi = np.linspace(0, w - 1, s).astype(int)
        img = img[np.ix_(yi, xi)]                          # (s, s, 3)
        return (img.astype(np.float32) / 255.0).transpose(2, 0, 1)  # (3,s,s)

    def _proc_frames(self, imgs: np.ndarray) -> np.ndarray:
        """(k, ...) → (k, 3, H', W') float32."""
        return np.stack([self._proc_frame(imgs[i]) for i in range(len(imgs))])


def build_recon_loader(
    recon_dir:  str,
    k:          int  = 16,
    batch_size: int  = 8,
    max_files:  Optional[int] = None,
    shuffle:    bool = True,
) -> DataLoader:
    paths = sorted(Path(recon_dir).glob("**/*.hdf5"))
    if not paths:
        raise FileNotFoundError(f"No HDF5 files in {recon_dir}")
    ds = RECONActionDataset(paths, k=k, max_files=max_files)
    return DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = 2,
        pin_memory  = True,
        drop_last   = True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Temporal frame encoder (k frames → single particle set)
# ═══════════════════════════════════════════════════════════════════════════

class TemporalFrameEncoder(nn.Module):
    """
    Encodes k consecutive frames into a single particle representation.

    Mirrors the k_ctx=16 proprioceptive encoder from Sprint 6c/9, but
    operates on visual frames rather than proprio signals.

    Architecture:
        Per-frame StudentEncoder → temporal mean pool over k → linear head
    """

    def __init__(self, student_enc: nn.Module, d_model: int = 128, k: int = 16):
        super().__init__()
        self.enc    = student_enc   # frozen
        self.k      = k
        # Temporal attention pool: learns which frames matter most
        self.t_attn = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),       # (B, k, 1) → softmax over k
        )
        self.proj   = nn.Linear(d_model, d_model)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames: (B, k, 3, H, W)
        Returns: (B, d_model) — temporally pooled frame embedding
        """
        B, k = frames.shape[:2]
        # Encode each frame independently
        flat   = frames.view(B * k, *frames.shape[2:])          # (B*k, 3, H, W)
        z_flat = self.enc(flat)                                  # (B*k, d_model)
        z      = z_flat.view(B, k, -1)                          # (B, k, d_model)

        # Temporal attention pool
        w = torch.softmax(self.t_attn(z), dim=1)                # (B, k, 1)
        z_pool = (w * z).sum(dim=1)                             # (B, d_model)
        return self.proj(z_pool)


# ═══════════════════════════════════════════════════════════════════════════
# Sprint B training loop
# ═══════════════════════════════════════════════════════════════════════════

def train_sprint_b(
    recon_dir:       str,
    cwm_ckpt:        str,
    proprio_ckpt:    Optional[str] = None,
    n_epochs:        int   = 20,
    batch_size:      int   = 8,
    base_lr:         float = 5e-5,
    k_ctx:           int   = 16,
    save_dir:        str   = r"checkpoints/cwm",
    log_every:       int   = 50,
    max_files:       Optional[int] = None,
    device_str:      str   = "cpu",
):
    """
    Sprint B: action-conditioned CWM training.

    Training objective:
        Given k context frames + action, predict the next frame's latent.
        L = free_bits(MSE(symlog(z_pred), symlog(z_target)), 0.5)

    Ablation baseline (for AUROC comparison):
        Same architecture, action zeroed out → unconditioned baseline.
    """
    device = torch.device(device_str)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # ── Load base CWM ────────────────────────────────────────────────────────
    from train_cwm import CortexWorldModel
    MAX_ACTION_DIM = 9   # MoEJEPAPredictor.action_proj weight shape
    cwm = CortexWorldModel(d_model=128, K=16).to(device)
    if Path(cwm_ckpt).exists():
        ckpt = torch.load(cwm_ckpt, map_location=device, weights_only=False)
        cwm.load_state_dict(ckpt["model"])
        print(f"Loaded CWM: {cwm_ckpt}  (loss={ckpt.get('loss', '?')})")
    else:
        print(f"WARNING: {cwm_ckpt} not found — training from scratch")

    # ── Load frozen StudentEncoder ───────────────────────────────────────────
    from train_mvtec import StudentEncoder
    student_enc = StudentEncoder().to(device)
    _enc_path = Path("checkpoints/dinov2_student/student_best.pt")
    if _enc_path.exists():
        _sd = torch.load(_enc_path, map_location="cpu", weights_only=False)
        _sd = _sd.get("model", _sd.get("state_dict", _sd))
        student_enc.load_state_dict(_sd, strict=False)
        print(f"StudentEncoder loaded: {_enc_path}")
    student_enc.eval()
    for p in student_enc.parameters():
        p.requires_grad_(False)

    # ── Build Sprint B modules ───────────────────────────────────────────────
    d_model     = 128
    action_enc  = ActionEncoder(action_dim=2, d_model=d_model).to(device)
    temp_enc    = TemporalFrameEncoder(student_enc, d_model=d_model, k=k_ctx).to(device)
    ac_pred     = ActionConditionedPredictor(d_model=d_model).to(device)

    # Load proprio_ckpt temporal weights if available (warm-start temp_enc)
    if proprio_ckpt and Path(proprio_ckpt).exists():
        try:
            _pc = torch.load(proprio_ckpt, map_location="cpu", weights_only=False)
            _sd = _pc.get("model", _pc)
            temp_enc.load_state_dict(_sd, strict=False)
            print(f"Temporal encoder warm-started from: {proprio_ckpt}")
        except Exception as e:
            print(f"WARNING: could not load proprio_ckpt: {e}")

    # ── Data loader ───────────────────────────────────────────────────────────
    loader = build_recon_loader(
        recon_dir, k=k_ctx, batch_size=batch_size, max_files=max_files
    )

    # ── Trainable parameters ──────────────────────────────────────────────────
    # Freeze: cwm.encoder_moe, cwm.particle_enc, student_enc
    # Train:  action_enc, temp_enc (t_attn + proj only), ac_pred.gate + base predictor
    trainable = (
        list(action_enc.parameters())
        + list(temp_enc.t_attn.parameters())
        + list(temp_enc.proj.parameters())
        + [ac_pred.gate]
        + list(cwm.predictor.parameters())
        + list(cwm.thick_gru.parameters())
    )
    n_params = sum(p.numel() for p in trainable)
    print(f"Trainable params: {n_params:,}")

    optimizer = torch.optim.AdamW(trainable, lr=base_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr      = base_lr * 5,
        total_steps = n_epochs * len(loader),
        pct_start   = 0.1,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_loss   = float("inf")
    global_step = 0
    gate_log    = []    # track gate magnitude to confirm action is being used

    for epoch in range(n_epochs):
        cwm.predictor.train()
        cwm.thick_gru.train()
        action_enc.train()
        temp_enc.t_attn.train()
        temp_enc.proj.train()
        cwm.encoder_moe.eval()
        cwm.particle_enc.eval()

        epoch_losses         = []
        epoch_losses_no_ac   = []   # ablation losses (action zeroed)

        for batch in loader:
            frames   = batch["frames"].to(device)     # (B, k, 3, H, W)
            action   = batch["action"].to(device)     # (B, 2)
            frame_t1 = batch["frame_t1"].to(device)   # (B, 3, H, W)

            B = frames.shape[0]

            # ── Encode context (with gradient through temp_enc projection) ──
            with torch.no_grad():
                z_t1_raw = student_enc(frame_t1)               # (B, d_model)
            z_ctx   = temp_enc(frames)                         # (B, d_model)
            z_ctx_p = z_ctx.unsqueeze(1).expand(-1, cwm.K, -1).contiguous()

            # ── Encode target (stop-gradient) ─────────────────────────────
            with torch.no_grad():
                particles_t1, _, _, _ = cwm.encode(z_t1_raw)
                z_target = particles_t1.detach()               # (B, K, d_model)

            # ── Action embedding ──────────────────────────────────────────
            action_emb = action_enc(action)                # (B, d_model)

            # ── Action-conditioned prediction ─────────────────────────────
            context_h   = cwm.thick_gru.init_context(B, device)
            particles_t = z_ctx_p.clone()

            particles_ac = ac_pred.inject(particles_t, action_emb)
            action_9     = F.pad(action, (0, MAX_ACTION_DIM - action.shape[-1]))
            z_pred_dict  = cwm.predict(
                particles  = particles_ac,
                action     = action_9,
                context_h  = context_h,
                positions  = torch.zeros(B, cwm.K, 2, device=device),
                domain_id  = torch.zeros(B, dtype=torch.long, device=device),
                regime     = "EXPLOIT",
                ach        = 0.5,
            )
            z_pred = z_pred_dict["z_pred"]                 # (B, K, d_model)

            # ── Loss: symlog MSE (free_bits applies to KL, not MSE) ──────
            L_pred = F.mse_loss(symlog(z_pred), symlog(z_target.detach()))

            # ── Ablation baseline (no action, same graph) ─────────────────
            with torch.no_grad():
                action_zero    = torch.zeros_like(action)
                action_emb_0   = action_enc(action_zero)
                action_zero_9   = F.pad(action_zero, (0, MAX_ACTION_DIM - action_zero.shape[-1]))
                particles_no_ac = ac_pred.inject(particles_t, action_emb_0)
                z_pred_no_ac    = cwm.predict(
                    particles  = particles_no_ac,
                    action     = action_zero_9,
                    context_h  = cwm.thick_gru.init_context(B, device),
                    positions  = torch.zeros(B, cwm.K, 2, device=device),
                    domain_id  = torch.zeros(B, dtype=torch.long, device=device),
                    regime     = "EXPLOIT",
                    ach        = 0.5,
                )["z_pred"]
                L_no_ac = F.mse_loss(
                    symlog(z_pred_no_ac), symlog(z_target)
                ).item()
            epoch_losses_no_ac.append(L_no_ac)

            # ── MoE router anti-collapse ───────────────────────────────────
            try:
                layer0 = cwm.predictor.layers[0]
                logits = layer0.moe_ffn.router(particles_t.reshape(-1, d_model))
                probs  = F.softmax(logits, dim=-1)
                f_i    = probs.mean(0)
                L_lb   = 0.01 * logits.shape[-1] * (f_i * probs.mean(0)).sum()
                L_z    = 0.0002 * (torch.logsumexp(logits, dim=-1) ** 2).mean()
                L_router = L_lb + L_z
            except Exception:
                L_router = torch.tensor(0.0, device=device)

            total_loss = L_pred + L_router

            if not torch.isfinite(total_loss):
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            total_loss.backward()
            agc_clip(trainable)                            # DreamerV3 AGC
            optimizer.step()
            scheduler.step()

            loss_val = total_loss.item()
            epoch_losses.append(loss_val)
            global_step += 1

            if global_step % log_every == 0:
                gate_val = torch.sigmoid(ac_pred.gate).mean().item()
                gate_log.append(gate_val)
                lr_now   = optimizer.param_groups[0]["lr"]
                ac_lift  = L_no_ac - loss_val          # positive = action helps
                print(
                    f"[ep{epoch:02d} s{global_step:05d}] "
                    f"L={loss_val:.4f}  "
                    f"L_no_ac={L_no_ac:.4f}  "
                    f"ac_lift={ac_lift:+.4f}  "
                    f"gate={gate_val:.3f}  "
                    f"lr={lr_now:.2e}"
                )

        # ── Epoch summary ──────────────────────────────────────────────────
        mean_L     = np.mean(epoch_losses)     if epoch_losses     else 0
        mean_no_ac = np.mean(epoch_losses_no_ac) if epoch_losses_no_ac else 0
        ac_lift    = mean_no_ac - mean_L
        gate_mean  = np.mean(gate_log[-100:]) if gate_log else 0
        print(
            f"\nEpoch {epoch:02d}  "
            f"loss={mean_L:.4f}  "
            f"baseline={mean_no_ac:.4f}  "
            f"ac_lift={ac_lift:+.4f}  "
            f"gate={gate_mean:.3f}"
        )

        # ── Pass criterion check ───────────────────────────────────────────
        # ac_lift should be positive and growing — action is helping
        if epoch >= 3 and ac_lift < 0:
            print("  ⚠️  Action conditioning not helping yet — gate may need warmup")
        if epoch >= 5 and ac_lift > 0.005:
            print("  ✓  Action conditioning lifting prediction quality")

        # ── Save best ─────────────────────────────────────────────────────
        if mean_L < best_loss:
            best_loss = mean_L
            save_path = Path(save_dir) / "cwm_action_best.pt"
            torch.save({
                "epoch":      epoch,
                "loss":       best_loss,
                "ac_lift":    ac_lift,
                "model":      cwm.state_dict(),
                "action_enc": action_enc.state_dict(),
                "temp_enc":   temp_enc.state_dict(),
                "ac_pred_gate": ac_pred.gate.data,
                "k_ctx":      k_ctx,
            }, save_path)
            print(f"  → Saved: {save_path}")

    print(f"\nSprint B complete. Best loss: {best_loss:.4f}")
    print(f"Final gate magnitude: {torch.sigmoid(ac_pred.gate).mean().item():.3f}")
    print("Next: run eval_goal_reaching.py --cwm-ckpt cwm_action_best.pt "
          "to confirm GRASP SR improvement over no-action baseline.")
    return cwm, action_enc


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--recon-dir",      required=True)
    p.add_argument("--cwm-ckpt",       default=r"checkpoints/cwm/cwm_multidomain_best.pt")
    p.add_argument("--proprio-ckpt",   default=r"checkpoints/cwm/proprio_6c_best.pt")
    p.add_argument("--epochs",         type=int,   default=20)
    p.add_argument("--batch-size",     type=int,   default=8)
    p.add_argument("--lr",             type=float, default=5e-5)
    p.add_argument("--k-ctx",          type=int,   default=16)
    p.add_argument("--save-dir",       default=r"checkpoints/cwm")
    p.add_argument("--log-every",      type=int,   default=50)
    p.add_argument("--max-files",      type=int,   default=None)
    p.add_argument("--device",         default="cpu")
    args = p.parse_args()

    train_sprint_b(
        recon_dir    = args.recon_dir,
        cwm_ckpt     = args.cwm_ckpt,
        proprio_ckpt = args.proprio_ckpt,
        n_epochs     = args.epochs,
        batch_size   = args.batch_size,
        base_lr      = args.lr,
        k_ctx        = args.k_ctx,
        save_dir     = args.save_dir,
        log_every    = args.log_every,
        max_files    = args.max_files,
        device_str   = args.device,
    )
