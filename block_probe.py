"""
block_probe.py  —  NeMo-WM Block Position Probe
================================================
Lightweight linear probe: z (128-dim latent) → block_pos (2-dim).

Closes the hot/cold feedback loop in the CEM planner:
  Before: score = -||z_pred - z_goal||      (latent distance, not task-aligned)
  After:  score = DA(block_pred, goal)       (obs-space warmth, task-aligned)

Training:
  - Freezes the world model encoder
  - Trains a 2-layer MLP on (z, block_pos) pairs from the existing dataset
  - Takes ~2 min on CPU, saves alongside the world model checkpoint

Planning integration:
  - ProbeScorer replaces raw latent distance in MPCPlanner
  - Each candidate rollout is scored by accumulated GoalDA warmth
    computed from decoded block positions
  - The child that gets "hot/cold" feedback after each imagined step

Usage:
    # Train probe on existing checkpoint + synthetic data
    python block_probe.py \
        --ckpt  checkpoints/action_wm/action_wm_pusht_full_best.pt \
        --epochs 20 \
        --save  checkpoints/action_wm/block_probe_best.pt

    # Eval with probe scorer
    python eval_pusht_sr.py \
        --ckpt  checkpoints/action_wm/action_wm_pusht_full_best.pt \
        --probe checkpoints/action_wm/block_probe_best.pt \
        --n-episodes 50 --no-video --synthetic
"""

import argparse
import math
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ═══════════════════════════════════════════════════════════════════════════
# Block position probe
# ═══════════════════════════════════════════════════════════════════════════

class BlockProbe(nn.Module):
    """
    Two-layer MLP: latent z (128-dim) → block position (2-dim).

    Trained frozen on top of the world model encoder.
    Predicts normalised block (x, y) ∈ [0, 1]^2.

    Design choice: sigmoid output ensures predictions stay in [0,1].
    L1 loss is more robust than MSE to occasional outliers from
    the synthetic trajectory distribution.
    """

    def __init__(self, d_model: int = 128, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 2),
            nn.Sigmoid(),    # → [0, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, d_model) → block_pos (B, 2)"""
        return self.net(z)

    def predict_np(self, z: torch.Tensor) -> np.ndarray:
        """Convenience: tensor → numpy."""
        with torch.no_grad():
            return self.forward(z).cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════
# Probe scorer — replaces latent distance in CEM planner
# ═══════════════════════════════════════════════════════════════════════════

class ProbeScorer:
    """
    Scores CEM rollouts using GoalDA warmth over decoded block positions.

    Instead of:
        score = -||z_final - z_goal||

    Does:
        for each step in rollout:
            block_pred = probe(z_step)
            da += GoalDA.shaping(block_pred, goal)
        score = total_da

    This mirrors how a blindfolded child accumulates hot/cold feedback
    along an imagined trajectory before choosing which action to take.

    The probe replaces the abstract "distance in latent space" with
    a concrete, task-aligned signal: is the block getting closer to goal?
    """

    def __init__(
        self,
        probe:      BlockProbe,
        goal:       np.ndarray,    # (2,) normalised goal position
        device:     torch.device,
        approach_scale: float = 4.0,
        retreat_scale:  float = 3.0,
        goal_thr:   float = 0.10,
    ):
        self.probe  = probe
        self.goal   = goal
        self.device = device
        self.approach_scale = approach_scale
        self.retreat_scale  = retreat_scale
        self.goal_thr = goal_thr

    @torch.no_grad()
    def score_rollout(
        self,
        z_preds:   List[torch.Tensor],  # list of (K, d_model) tensors
        prev_dist: Optional[np.ndarray] = None,  # (K,) starting dist
    ) -> torch.Tensor:
        """
        Score K candidate rollouts by accumulated DA warmth.
        Returns: (K,) scores — higher = warmer = better.
        """
        K = z_preds[0].shape[0]
        scores = torch.zeros(K, device=self.device)

        # Initial block position (from first pred or passed in)
        if prev_dist is None:
            block0 = self.probe(z_preds[0])   # (K, 2)
            prev_dist_t = torch.norm(
                block0 - torch.tensor(self.goal, device=self.device), dim=-1
            )  # (K,)
        else:
            prev_dist_t = torch.tensor(prev_dist, device=self.device)

        for z_step in z_preds:
            block_pred = self.probe(z_step)   # (K, 2)
            goal_t     = torch.tensor(self.goal, device=self.device)
            dist       = torch.norm(block_pred - goal_t, dim=-1)  # (K,)

            # DA shaping: positive when getting closer
            delta = prev_dist_t - dist   # (K,) positive = approaching

            # Approach reward
            approach_mask = delta > 0.003
            scores += approach_mask.float() * delta * self.approach_scale

            # Retreat penalty
            retreat_mask = delta < -0.003
            scores += retreat_mask.float() * delta * self.retreat_scale  # negative

            # Goal bonus
            near_goal = dist < self.goal_thr
            scores += near_goal.float() * 2.0

            prev_dist_t = dist

        return scores   # (K,) — higher is better

    @torch.no_grad()
    def score_final(self, z_final: torch.Tensor) -> torch.Tensor:
        """
        Fast single-step scoring: just final block position distance.
        Used as fallback or for simple horizon=1 planning.
        Returns: (K,) scores.
        """
        block_pred = self.probe(z_final)   # (K, 2)
        goal_t     = torch.tensor(self.goal, device=self.device)
        dist       = torch.norm(block_pred - goal_t, dim=-1)  # (K,)
        return -dist   # higher = closer = better


# ═══════════════════════════════════════════════════════════════════════════
# Probe training
# ═══════════════════════════════════════════════════════════════════════════

def train_probe(
    ckpt_path:  str,
    save_path:  str,
    n_episodes: int = 500,
    n_epochs:   int = 20,
    batch_size: int = 256,
    lr:         float = 1e-3,
    device_str: str = "cpu",
):
    """
    Train block position probe on synthetic PushT data.
    Encoder is frozen — only the probe trains.
    """
    from train_action_wm import StateEncoder, PushTStateDataset

    device = torch.device(device_str)

    # ── Load frozen encoder ───────────────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    obs_dim = ckpt.get("obs_dim", 5)
    D       = 128

    encoder = StateEncoder(obs_dim, D).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    print(f"Encoder loaded (frozen): ep={ckpt.get('epoch','?')} "
          f"ac_lift={ckpt.get('diagnostics',{}).get('ac_lift',0):+.4f}")

    # ── Build dataset: (z, block_pos) pairs ──────────────────────────────
    print("Building probe dataset...")
    ds = PushTStateDataset(k_steps=1, n_episodes=n_episodes)
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)

    z_list, block_list = [], []
    with torch.no_grad():
        for batch in loader:
            obs = batch["obs"][:, 0].to(device)   # (B, obs_dim) — t=0 obs
            z   = encoder(obs)                     # (B, D)
            block_pos = obs[:, 2:4]                # (B, 2) — block (x, y) ∈ [0,1]
            z_list.append(z.cpu())
            block_list.append(block_pos.cpu())

    z_all     = torch.cat(z_list,     dim=0)   # (N, D)
    block_all = torch.cat(block_list, dim=0)   # (N, 2)

    print(f"  Probe dataset: {len(z_all):,} samples")
    print(f"  Block pos range: [{block_all.min():.3f}, {block_all.max():.3f}]")

    probe_ds  = TensorDataset(z_all, block_all)
    probe_loader = DataLoader(probe_ds, batch_size=batch_size,
                               shuffle=True, num_workers=0)

    # ── Train probe ───────────────────────────────────────────────────────
    probe = BlockProbe(D).to(device)
    opt   = torch.optim.AdamW(probe.parameters(), lr=lr)

    best_loss = float("inf")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(n_epochs):
        probe.train()
        losses = []
        for z_batch, block_batch in probe_loader:
            z_batch     = z_batch.to(device)
            block_batch = block_batch.to(device)
            pred  = probe(z_batch)
            loss  = F.smooth_l1_loss(pred, block_batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        mean_L = np.mean(losses)

        # Eval: MAE in normalised coords
        probe.eval()
        with torch.no_grad():
            pred_all = probe(z_all.to(device)).cpu()
            mae      = (pred_all - block_all).abs().mean().item()

        print(f"  Epoch {epoch:02d}  loss={mean_L:.4f}  MAE={mae:.4f}"
              f"  ({mae*100:.1f}% of arena)")

        if mean_L < best_loss:
            best_loss = mean_L
            torch.save({
                "epoch":    epoch,
                "loss":     best_loss,
                "mae":      mae,
                "d_model":  D,
                "probe":    probe.state_dict(),
                "ckpt_src": ckpt_path,
            }, save_path)
            print(f"  → Saved: {save_path}")

    print(f"\nProbe training complete. Best MAE: {mae:.4f} "
          f"({mae * 512:.1f} pixels in gym space)")
    return probe


# ═══════════════════════════════════════════════════════════════════════════
# Load probe
# ═══════════════════════════════════════════════════════════════════════════

def load_probe(
    probe_path: str,
    device:     torch.device,
) -> BlockProbe:
    ckpt  = torch.load(probe_path, map_location=device, weights_only=False)
    probe = BlockProbe(ckpt.get("d_model", 128)).to(device)
    probe.load_state_dict(ckpt["probe"])
    probe.eval()
    print(f"Block probe loaded: ep={ckpt.get('epoch','?')} "
          f"MAE={ckpt.get('mae',0):.4f} "
          f"({ckpt.get('mae',0)*100:.1f}% of arena)")
    return probe


# ═══════════════════════════════════════════════════════════════════════════
# Patched MPCPlanner that uses ProbeScorer
# ═══════════════════════════════════════════════════════════════════════════

class ProbeMPCPlanner:
    """
    CEM planner that scores rollouts using ProbeScorer (GoalDA warmth)
    instead of latent distance to z_goal.

    Drop-in replacement for MPCPlanner in eval_pusht_sr.py.
    Add --probe flag to eval script to activate.
    """

    def __init__(
        self,
        encoder:     nn.Module,
        transition:  nn.Module,
        action_dim:  int,
        device:      torch.device,
        probe:       BlockProbe,
        goal:        np.ndarray,          # (2,) normalised
        n_candidates: int = 512,
        horizon:     int = 8,
        n_elite:     int = 64,
        n_iters:     int = 3,
        action_low:  float = 0.0,
        action_high: float = 1.0,
    ):
        self.enc      = encoder
        self.trans    = transition
        self.act_dim  = action_dim
        self.device   = device
        self.K        = n_candidates
        self.H        = horizon
        self.n_elite  = n_elite
        self.n_iters  = n_iters
        self.a_low    = action_low
        self.a_high   = action_high
        self.scorer   = ProbeScorer(probe, goal, device)

    @torch.no_grad()
    def plan(
        self,
        obs:      np.ndarray,
        goal_obs: np.ndarray = None,    # ignored — uses probe scorer
    ) -> np.ndarray:
        """obs: (5,) normalised. Returns action (2,) in [0,1]."""
        # Detect and normalise if gym pixel space
        if obs[:4].max() > 1.5:
            obs = obs.copy(); obs[:4] /= 512.0

        z_cur = self.enc(
            torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        )  # (1, D)

        # Get starting block dist for DA shaping baseline
        with torch.no_grad():
            block_cur = self.scorer.probe(z_cur).cpu().numpy()[0]  # (2,)
            prev_dist = np.linalg.norm(block_cur - self.scorer.goal, keepdims=True)
            prev_dist = np.broadcast_to(prev_dist, (self.K,))

        # CEM
        mu  = torch.full((self.H, self.act_dim), 0.5, device=self.device)
        std = torch.full((self.H, self.act_dim), 0.3, device=self.device)

        for _ in range(self.n_iters):
            actions = torch.clamp(
                mu + std * torch.randn(self.K, self.H, self.act_dim, device=self.device),
                self.a_low, self.a_high
            )   # (K, H, act_dim)

            z_batch = z_cur.expand(self.K, -1)
            z_preds = []
            z = z_batch
            for h in range(self.H):
                z = self.trans(z, actions[:, h])
                z_preds.append(z)

            # Score by accumulated GoalDA warmth (not latent distance)
            scores = self.scorer.score_rollout(z_preds, prev_dist)

            elite_idx = scores.topk(self.n_elite).indices
            elite_ac  = actions[elite_idx]
            mu  = elite_ac.mean(0)
            std = elite_ac.std(0).clamp(min=0.05)

        best_action = actions[scores.argmax(), 0].cpu().numpy()
        return best_action


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",   required=True,
                   help="Path to action_wm_pusht_full_best.pt")
    p.add_argument("--save",   default="checkpoints/action_wm/block_probe_best.pt")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--n-episodes", type=int, default=500)
    p.add_argument("--device", default="cpu")
    p.add_argument("--eval",   action="store_true",
                   help="Quick eval: predict block pos for 5 random obs")
    args = p.parse_args()

    if args.eval and Path(args.save).exists():
        # Quick sanity check
        device = torch.device(args.device)
        probe  = load_probe(args.save, device)
        from train_action_wm import StateEncoder
        ckpt   = torch.load(args.ckpt, map_location=device, weights_only=False)
        enc    = StateEncoder(5, 128).to(device)
        enc.load_state_dict(ckpt["encoder"])
        enc.eval()

        print("\nProbe sanity check (5 synthetic obs):")
        rng = np.random.RandomState(0)
        for i in range(5):
            obs = rng.uniform(0, 1, 5).astype(np.float32)
            obs[4] = obs[4]  # angle norm
            z = enc(torch.from_numpy(obs).unsqueeze(0).to(device))
            pred = probe.predict_np(z)[0]
            true = obs[2:4]
            err  = np.linalg.norm(pred - true)
            print(f"  obs_block={true.round(3)}  pred={pred.round(3)}  "
                  f"err={err:.3f}  ({err*100:.1f}%)")
    else:
        train_probe(
            ckpt_path  = args.ckpt,
            save_path  = args.save,
            n_episodes = args.n_episodes,
            n_epochs   = args.epochs,
            device_str = args.device,
        )
