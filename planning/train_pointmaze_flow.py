"""
train_pointmaze_flow.py — PointMaze Flow Matching Policy
=========================================================
Port of NeMoFlowPolicyV2 (PushT, 100% SR) to PointMaze-UMaze.

Changes from PushT:
  - obs_dim: 5 → 4 (x, y, vx, vy — no block/angle)
  - action_dim: 3 → 2 (force_x, force_y — no dangle)
  - goal_dim: 3 → 2 (goal_x, goal_y — no target angle)
  - state_dim: 3 → 4 (full obs IS the state — no probe needed)
  - No block probe — agent IS the entity being controlled

Architecture:
  Conditioning: z(128) + state(4) + DA(1) + goal(2) + t(1) + x_t(H*2)
  MLP: cond_dim → 256 → 256 → 256 → 256 → H*2
  Same LayerNorm + GELU stack as PushT

Training:
  1. Generate scripted demos via pointmaze_flow_demos.py (or load Minari)
  2. Encode obs through StateEncoder → z (128-D)
  3. Flow matching: x_0 = clean actions, x_1 = noise, v = x_1 - x_0
  4. Train policy to predict v given (z, state, DA, goal, t, x_t)

Eval:
  Sample action chunks via ODE, execute first action, re-plan

Usage:
    python train_pointmaze_flow.py --n-demos 2000 --epochs 50
    python train_pointmaze_flow.py --eval --n-episodes 100
    python train_pointmaze_flow.py --minari --epochs 50

Author: John Taylor
Sprint: PointMaze flow policy port (Paper 2 blocker)
"""

import argparse
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pointmaze_flow_demos import (
    generate_pointmaze_demos, load_minari_demos,
    step_physics, normalize_pos, normalize_vel, UMAZE_OPEN, cell_to_pos,
)


# ──────────────────────────────────────────────────────────────────────────────
# StateEncoder for PointMaze (reuse from train_action_wm if available)
# ──────────────────────────────────────────────────────────────────────────────

class PointMazeEncoder(nn.Module):
    """
    Simple encoder: obs(4) → z(128).
    Same architecture as PushT StateEncoder but obs_dim=4.
    """
    def __init__(self, obs_dim=4, d_latent=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, d_latent),
            nn.LayerNorm(d_latent),
        )

    def forward(self, obs):
        return self.net(obs)


# ──────────────────────────────────────────────────────────────────────────────
# Flow Policy for PointMaze
# ──────────────────────────────────────────────────────────────────────────────

class PointMazeFlowPolicy(nn.Module):
    """
    Flow matching policy for PointMaze-UMaze.

    Conditioning:
        z       (128) — encoder latent
        state     (4) — [x, y, vx, vy] normalised
        DA        (1) — goal proximity signal
        goal      (2) — [goal_x, goal_y] normalised
        t         (1) — flow time [0, 1]
        x_t     (H*2) — noisy action chunk

    Output: v_theta(x_t, t, cond) → (B, H*2) velocity field
    """
    ACTION_DIM = 2

    def __init__(self, H=8, d_z=128, d_hidden=256, n_layers=4):
        super().__init__()
        self.H = H
        self.d_z = d_z
        self.action_dim = self.ACTION_DIM

        # z(128) + state(4) + DA(1) + goal(2) + t(1) + x_t(H*2)
        cond_dim = d_z + 4 + 1 + 2 + 1 + H * self.ACTION_DIM

        layers = []
        in_dim = cond_dim
        for _ in range(n_layers):
            layers += [
                nn.Linear(in_dim, d_hidden),
                nn.LayerNorm(d_hidden),
                nn.GELU(),
            ]
            in_dim = d_hidden
        layers.append(nn.Linear(d_hidden, H * self.ACTION_DIM))
        self.net = nn.Sequential(*layers)

    def forward(self, z, state, da, goal, t, x_t):
        """Predict velocity field v_θ(x_t, t, cond) → (B, H*2)."""
        cond = torch.cat([z, state, da, goal, t, x_t], dim=-1)
        return self.net(cond)

    def sample(self, z, state, da, goal, n_steps=10, H=None):
        """Sample action chunk via Euler ODE integration."""
        H = H or self.H
        B = z.shape[0]
        device = z.device

        # DA-conditioned noise temperature
        temp = 1.0 - da.mean().item()
        x_t = torch.randn(B, H * self.ACTION_DIM, device=device) * (0.5 + 0.5 * temp)

        dt = 1.0 / n_steps
        for i in range(n_steps):
            t_val = torch.full((B, 1), i * dt, device=device)
            with torch.no_grad():
                v = self.forward(z, state, da, goal, t_val, x_t)
            x_t = x_t + v * dt

        return x_t.view(B, H, self.ACTION_DIM).clamp(-1, 1)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class PointMazeFlowDataset(torch.utils.data.Dataset):
    """
    (obs, action_chunk, goal) tuples for flow matching.
    obs:     (4,) — x, y, vx, vy normalised
    actions: (H, 2) — force_x, force_y
    goal:    (2,) — goal_x, goal_y normalised
    """
    def __init__(self, demos, H=8):
        self.H = H
        self.items = []
        for demo in demos:
            obs = demo['obs']       # (T, 4)
            actions = demo['actions']  # (T, 2)
            goal = demo['goal']     # (2,)
            T = min(len(obs), len(actions))
            for t in range(T - H):
                self.items.append({
                    'obs': obs[t].astype(np.float32),
                    'actions': actions[t:t+H].astype(np.float32),
                    'goal': goal.astype(np.float32),
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return (
            torch.from_numpy(item['obs']),
            torch.from_numpy(item['actions']),
            torch.from_numpy(item['goal']),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  PointMaze Flow Policy Training")
    print("=" * 60)

    # Generate or load demos
    if args.minari:
        print("  Loading Minari demos...")
        demos = load_minari_demos(n_demos=args.n_demos, H=args.H)
    else:
        print(f"  Generating {args.n_demos} scripted demos...")
        demos = generate_pointmaze_demos(
            n_demos=args.n_demos, H=args.H, seed=args.seed)

    sr = sum(d['success'] for d in demos) / len(demos)
    print(f"  Demo SR: {sr:.1%}")

    # Dataset
    ds = PointMazeFlowDataset(demos, H=args.H)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(f"  Dataset: {len(ds)} samples, batch={args.batch_size}")

    # Models
    encoder = PointMazeEncoder(obs_dim=4, d_latent=128).to(device)
    policy = PointMazeFlowPolicy(H=args.H, d_z=128).to(device)

    print(f"  Encoder: {sum(p.numel() for p in encoder.parameters()):,} params")
    print(f"  Policy: {policy.count_params():,} params")

    # Joint training (encoder + policy)
    params = list(encoder.parameters()) + list(policy.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * len(loader))

    best_loss = float('inf')
    t0_total = time.time()

    for epoch in range(args.epochs):
        epoch_losses = []
        encoder.train()
        policy.train()

        for obs_batch, action_batch, goal_batch in loader:
            obs_batch = obs_batch.to(device)        # (B, 4)
            action_batch = action_batch.to(device)  # (B, H, 2)
            goal_batch = goal_batch.to(device)      # (B, 2)
            B = obs_batch.shape[0]

            # Encode
            z = encoder(obs_batch)  # (B, 128)

            # State = obs itself (no probe needed for PointMaze)
            state = obs_batch  # (B, 4)

            # DA = goal proximity
            pos = obs_batch[:, :2]  # normalised position
            goal_dist = (pos - goal_batch).norm(dim=-1, keepdim=True)
            da = (1 - goal_dist.clamp(0, 2) / 2)  # (B, 1) in [0, 1]

            # Flow matching
            x0 = action_batch.reshape(B, args.H * 2)  # clean
            x1 = torch.randn_like(x0)                  # noise
            t_val = torch.rand(B, 1, device=device)
            x_t = (1 - t_val) * x0 + t_val * x1
            v_gt = x0 - x1

            v_pred = policy(z, state, da, goal_batch, t_val, x_t)
            loss = F.mse_loss(v_pred, v_gt)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            epoch_losses.append(loss.item())

        mean_loss = np.mean(epoch_losses)
        lr_now = sched.get_last_lr()[0]

        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save({
                'encoder': encoder.state_dict(),
                'policy': policy.state_dict(),
                'epoch': epoch,
                'loss': best_loss,
                'H': args.H,
                'obs_dim': 4,
                'action_dim': 2,
                'goal_dim': 2,
            }, str(out_dir / 'pointmaze_flow_best.pt'))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = time.time() - t0_total
            print(f"  ep{epoch+1:3d}  loss={mean_loss:.4f}  "
                  f"best={best_loss:.4f}  lr={lr_now:.2e}  "
                  f"[{elapsed:.0f}s]")

    print(f"\n  Training complete. Best loss: {best_loss:.4f}")
    print(f"  Saved: {out_dir / 'pointmaze_flow_best.pt'}")
    return encoder, policy


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(args):
    device = torch.device(args.device)
    ckpt_path = Path(args.out_dir) / 'pointmaze_flow_best.pt'

    print("=" * 60)
    print("  PointMaze Flow Policy Evaluation")
    print("=" * 60)

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    H = ckpt.get('H', 8)

    encoder = PointMazeEncoder(obs_dim=4, d_latent=128).to(device)
    encoder.load_state_dict(ckpt['encoder'])
    encoder.eval()

    policy = PointMazeFlowPolicy(H=H, d_z=128).to(device)
    policy.load_state_dict(ckpt['policy'])
    policy.eval()

    print(f"  Loaded: {ckpt_path}")
    print(f"  Epoch {ckpt['epoch']}, loss={ckpt['loss']:.4f}, H={H}")

    rng = np.random.RandomState(args.seed)
    successes = 0
    total_steps = 0
    results = []

    for ep in range(args.n_episodes):
        # Random start and goal
        cells = rng.choice(len(UMAZE_OPEN), 2, replace=False)
        start_pos = cell_to_pos(*UMAZE_OPEN[cells[0]]) + rng.uniform(-0.3, 0.3, 2)
        goal_pos = cell_to_pos(*UMAZE_OPEN[cells[1]]) + rng.uniform(-0.3, 0.3, 2)
        start_pos = np.clip(start_pos, [0.2, 0.2], [2.8, 4.8])
        goal_pos = np.clip(goal_pos, [0.2, 0.2], [2.8, 4.8])

        pos = start_pos.copy()
        vel = np.zeros(2, dtype=np.float32)
        goal_norm = normalize_pos(goal_pos)

        success = False
        for step in range(300):
            obs = np.concatenate([normalize_pos(pos), normalize_vel(vel)])
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            goal_t = torch.from_numpy(goal_norm).float().unsqueeze(0).to(device)

            with torch.no_grad():
                z = encoder(obs_t)
                state = obs_t
                goal_dist = (obs_t[:, :2] - goal_t).norm(dim=-1, keepdim=True)
                da = (1 - goal_dist.clamp(0, 2) / 2)
                actions = policy.sample(z, state, da, goal_t,
                                         n_steps=args.n_ode_steps)

            force = actions[0, 0].cpu().numpy()  # first action of chunk
            pos, vel = step_physics(pos, vel, force)

            if np.linalg.norm(pos - goal_pos) < 0.15:
                success = True
                successes += 1
                total_steps += step + 1
                break
        else:
            total_steps += 300

        dist = np.linalg.norm(pos - goal_pos)
        tag = "ok" if success else f"d={dist:.2f}"
        results.append((success, step + 1 if success else 300, dist))

        if (ep + 1) % 10 == 0:
            sr_so_far = successes / (ep + 1)
            print(f"  ep{ep+1:3d}  SR={sr_so_far:.1%}  "
                  f"({successes}/{ep+1})")

    sr = successes / args.n_episodes
    avg_steps = total_steps / args.n_episodes
    avg_dist = np.mean([r[2] for r in results])
    succ_steps = np.mean([r[1] for r in results if r[0]]) if successes else 0

    print(f"\n  Results (n={args.n_episodes}):")
    print(f"    SR: {sr:.1%} ({successes}/{args.n_episodes})")
    print(f"    Avg steps (all): {avg_steps:.0f}")
    print(f"    Avg steps (success): {succ_steps:.0f}")
    print(f"    Avg final dist: {avg_dist:.3f}")
    print("=" * 60)

    return sr


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="PointMaze Flow Policy")
    ap.add_argument("--eval", action="store_true", help="Eval only")
    ap.add_argument("--minari", action="store_true",
                    help="Use Minari data instead of scripted demos")
    ap.add_argument("--n-demos", type=int, default=2000)
    ap.add_argument("--n-episodes", type=int, default=100,
                    help="Episodes for eval")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--H", type=int, default=8,
                    help="Action chunk length")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-ode-steps", type=int, default=10,
                    help="ODE steps for flow sampling at eval")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default="checkpoints/pointmaze_flow")
    args = ap.parse_args()

    if args.eval:
        evaluate(args)
    else:
        train(args)
        print("\nRunning eval on trained policy...")
        evaluate(args)


if __name__ == "__main__":
    main()
