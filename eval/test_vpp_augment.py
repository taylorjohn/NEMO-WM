"""
test_vpp_augment.py — VPP-Inspired Belief Augmentation Test
==============================================================
Compares two vision encoders on PushT:
  A) Standard CNN: 96x96x6 (2-frame RGB) → 128D latent
  B) Augmented CNN: 96x96x8 (2-frame RGB + 2ch belief map) → 128D latent

The belief map is a spatial attention channel generated from the
agent's previous action and belief state — "where I expect the
T-block to be." This is VPP's core idea applied to world models:
sparse hints (belief) amplify dense signals (pixels).

Usage:
    python test_vpp_augment.py --epochs 200 --eval
"""

import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Encoders
# ──────────────────────────────────────────────────────────────────────────────

class StandardEncoder(nn.Module):
    """Standard 2-frame CNN encoder (current method)."""
    def __init__(self, d_latent=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, d_latent),
        )

    def forward(self, x):
        return self.net(x)


class BeliefAugmentedEncoder(nn.Module):
    """
    VPP-inspired encoder: projects belief state as spatial attention
    maps onto the image before encoding.

    Two extra channels:
      ch7: action direction map — Gaussian centered where the agent
           expects the effect of its last action
      ch8: belief confidence map — spatial map from episodic memory
           indicating where similar states were seen before

    This is the VPP insight: sparse hints amplify dense signals.
    """
    def __init__(self, d_latent=128, d_belief=4):
        super().__init__()
        # Belief → 2 spatial attention maps
        self.belief_to_maps = nn.Sequential(
            nn.Linear(d_belief, 64),
            nn.GELU(),
            nn.Linear(64, 2 * 12 * 12),  # 2 channels at 12x12, upsampled to 96x96
        )

        # CNN takes 8 channels: 6 (2-frame RGB) + 2 (belief maps)
        self.net = nn.Sequential(
            nn.Conv2d(8, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, d_latent),
        )

    def forward(self, x_rgb, belief):
        """
        x_rgb: (B, 6, 96, 96) — 2-frame stacked RGB
        belief: (B, 4) — [agent_x, agent_y, last_action_x, last_action_y]
        """
        B = x_rgb.shape[0]

        # Generate spatial belief maps
        maps = self.belief_to_maps(belief)  # (B, 2*12*12)
        maps = maps.view(B, 2, 12, 12)
        maps = F.interpolate(maps, size=(96, 96), mode='bilinear',
                              align_corners=False)
        maps = torch.sigmoid(maps)  # normalize to [0, 1]

        # Concatenate: 6ch RGB + 2ch belief maps = 8ch
        x = torch.cat([x_rgb, maps], dim=1)
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# Flow Policy (shared between both)
# ──────────────────────────────────────────────────────────────────────────────

class FlowPolicy(nn.Module):
    """Conditional flow matching policy."""
    def __init__(self, d_z=128, d_action=2, H=8):
        super().__init__()
        self.H = H
        self.net = nn.Sequential(
            nn.Linear(d_z + d_action + 1, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, d_action),
        )

    def forward(self, z, x_t, t):
        inp = torch.cat([z, x_t, t.unsqueeze(-1)], dim=-1)
        return self.net(inp)

    def sample(self, z, n_steps=10):
        B = z.shape[0]
        x = torch.randn(B, 2)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B,), i * dt)
            v = self.forward(z, x, t)
            x = x + v * dt
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

def load_pusht_data():
    """Load PushT human demo data."""
    import imageio
    import pyarrow.parquet as pq

    video_path = "pusht_data/lerobot_pusht/videos/observation.image_episode_000000.mp4"
    parquet_path = "pusht_data/lerobot_pusht/data/chunk-000/episode_000000.parquet"

    # Try loading all episodes
    import glob
    parquet_files = sorted(glob.glob(
        "pusht_data/lerobot_pusht/data/chunk-*/*.parquet"))

    if not parquet_files:
        print("  No parquet files found, generating synthetic data")
        return generate_synthetic_data()

    video_files = sorted(glob.glob(
        "pusht_data/lerobot_pusht/videos/observation.image_episode_*.mp4"))

    print(f"  Found {len(parquet_files)} parquet files, {len(video_files)} videos")

    all_frames = []
    all_actions = []
    all_states = []

    max_episodes = 50  # limit for speed
    for vi, vpath in enumerate(video_files[:max_episodes]):
        try:
            reader = imageio.get_reader(vpath)
            frames = []
            for frame in reader:
                img = np.array(frame)
                if img.shape[0] != 96 or img.shape[1] != 96:
                    from PIL import Image
                    img = np.array(Image.fromarray(img).resize((96, 96)))
                frames.append(img)
            reader.close()
            all_frames.extend(frames)
        except Exception:
            continue

    for ppath in parquet_files[:max_episodes]:
        try:
            df = pq.read_table(ppath).to_pandas()
            if 'action' in df.columns:
                actions = np.stack(df['action'].values)
            elif 'action.x' in df.columns:
                actions = np.column_stack([
                    df['action.x'].values, df['action.y'].values])
            else:
                # Find action columns
                act_cols = [c for c in df.columns if 'action' in c.lower()]
                if act_cols:
                    actions = df[act_cols].values
                else:
                    actions = np.zeros((len(df), 2))
            all_actions.extend(actions)

            # Extract state if available
            state_cols = [c for c in df.columns if 'state' in c.lower()]
            if state_cols:
                states = df[state_cols[:4]].values
                all_states.extend(states)
        except Exception:
            continue

    n = min(len(all_frames), len(all_actions))
    if n < 100:
        print(f"  Only {n} samples, generating synthetic")
        return generate_synthetic_data()

    frames = np.array(all_frames[:n])
    actions = np.array(all_actions[:n], dtype=np.float32)

    if all_states:
        states = np.array(all_states[:n], dtype=np.float32)
    else:
        # Generate dummy state: [agent_x, agent_y, action_x, action_y]
        states = np.zeros((n, 4), dtype=np.float32)
        states[:, 2:] = actions[:, :2]  # last action as state

    print(f"  Loaded {n} samples from human demos")
    return frames, actions, states


def generate_synthetic_data(n=5000):
    """Generate synthetic PushT-like data."""
    rng = np.random.RandomState(42)

    frames = rng.randint(0, 255, (n, 96, 96, 3), dtype=np.uint8)
    actions = rng.randn(n, 2).astype(np.float32) * 0.3
    states = np.zeros((n, 4), dtype=np.float32)

    # Agent position walks randomly
    pos = np.array([0.5, 0.5])
    for i in range(n):
        pos = np.clip(pos + actions[i] * 0.1, 0, 1)
        states[i, :2] = pos
        states[i, 2:] = actions[i]

        # Draw agent position on frame as a dim circle
        cx, cy = int(pos[0] * 95), int(pos[1] * 95)
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                px, py = cx + dx, cy + dy
                if 0 <= px < 96 and 0 <= py < 96 and dx*dx + dy*dy <= 9:
                    frames[i, py, px] = [200, 100, 100]

    print(f"  Generated {n} synthetic samples")
    return frames, actions, states


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_model(encoder, policy, frames, actions, states,
                is_augmented, epochs=200, lr=3e-4):
    """Train one encoder+policy pair."""
    # Build 2-frame dataset
    n = len(frames) - 1
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(policy.parameters()),
        lr=lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    encoder.train()
    policy.train()

    best_loss = float('inf')
    losses = []

    for epoch in range(1, epochs + 1):
        # Random batch
        idx = np.random.choice(n, size=min(256, n), replace=False)

        # Current + previous frame
        curr = torch.from_numpy(
            frames[idx + 1].astype(np.float32) / 255.0
        ).permute(0, 3, 1, 2)
        prev = torch.from_numpy(
            frames[idx].astype(np.float32) / 255.0
        ).permute(0, 3, 1, 2)
        x_rgb = torch.cat([curr, prev], dim=1)  # (B, 6, 96, 96)

        action_gt = torch.from_numpy(actions[idx + 1])
        state = torch.from_numpy(states[idx + 1])

        # Encode
        if is_augmented:
            belief = state[:, :4]  # [agent_x, agent_y, action_x, action_y]
            z = encoder(x_rgb, belief)
        else:
            z = encoder(x_rgb)

        # Flow matching loss
        t = torch.rand(len(idx))
        noise = torch.randn_like(action_gt)
        x_t = t.unsqueeze(-1) * action_gt + (1 - t.unsqueeze(-1)) * noise
        v_pred = policy(z, x_t, t)
        v_target = action_gt - noise
        loss = F.mse_loss(v_pred, v_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        if loss.item() < best_loss:
            best_loss = loss.item()

        if epoch % 50 == 0 or epoch == 1:
            print(f"    ep {epoch:4d}/{epochs}  loss={loss.item():.4f}  "
                  f"best={best_loss:.4f}  lr={scheduler.get_last_lr()[0]:.6f}")

    return best_loss, losses


def eval_model(encoder, policy, is_augmented, n_episodes=30):
    """Evaluate on gym-pusht."""
    try:
        import gymnasium
        import gym_pusht
    except ImportError:
        print("    gym-pusht not available, skipping eval")
        return None

    env = gymnasium.make('gym_pusht/PushT-v0',
                          obs_type='pixels', render_mode='rgb_array')

    encoder.eval()
    policy.eval()

    coverages = []
    prev_action = np.zeros(2, dtype=np.float32)

    for ep in range(n_episodes):
        obs, info = env.reset(seed=42 + ep)
        prev_obs = obs.copy()
        best_cov = info.get('coverage', 0)
        agent_pos = np.array([0.5, 0.5], dtype=np.float32)

        for step in range(300):
            curr = obs.astype(np.float32) / 255.0
            prev = prev_obs.astype(np.float32) / 255.0
            stacked = np.concatenate([curr, prev], axis=2)
            x_rgb = torch.from_numpy(
                np.transpose(stacked, (2, 0, 1))
            ).unsqueeze(0)

            with torch.no_grad():
                if is_augmented:
                    belief = torch.tensor(
                        [[agent_pos[0], agent_pos[1],
                          prev_action[0], prev_action[1]]],
                        dtype=torch.float32)
                    z = encoder(x_rgb, belief)
                else:
                    z = encoder(x_rgb)
                action = policy.sample(z, n_steps=10)

            action_np = action[0].numpy() * 512.0
            action_np = np.clip(action_np, 0, 512).astype(np.float32)

            prev_action = action[0].numpy()
            agent_pos = np.clip(agent_pos + prev_action * 0.1, 0, 1)

            prev_obs = obs.copy()
            obs, reward, term, trunc, info = env.step(action_np)
            cov = info.get('coverage', 0)
            best_cov = max(best_cov, cov)
            if cov >= 0.95 or term or trunc:
                break

        coverages.append(best_cov)

    env.close()
    return coverages


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--n-eval", type=int, default=30)
    args = ap.parse_args()

    print("=" * 65)
    print("  VPP-Inspired Belief Augmentation Test")
    print("  Standard CNN vs Belief-Augmented CNN on PushT")
    print("=" * 65)

    # Load data
    print("\n-- Loading data --")
    frames, actions, states = load_pusht_data()
    print(f"  Frames: {frames.shape}")
    print(f"  Actions: {actions.shape}")
    print(f"  States: {states.shape}")

    # ── Model A: Standard (current method) ──
    print("\n-- Model A: Standard CNN (6ch, no belief) --")
    enc_a = StandardEncoder(d_latent=128)
    pol_a = FlowPolicy(d_z=128)
    params_a = sum(p.numel() for p in enc_a.parameters()) + \
               sum(p.numel() for p in pol_a.parameters())
    print(f"  Params: {params_a:,}")

    best_a, losses_a = train_model(
        enc_a, pol_a, frames, actions, states,
        is_augmented=False, epochs=args.epochs)

    # ── Model B: Belief-Augmented (VPP-inspired) ──
    print("\n-- Model B: Belief-Augmented CNN (8ch, VPP) --")
    enc_b = BeliefAugmentedEncoder(d_latent=128, d_belief=4)
    pol_b = FlowPolicy(d_z=128)
    params_b = sum(p.numel() for p in enc_b.parameters()) + \
               sum(p.numel() for p in pol_b.parameters())
    print(f"  Params: {params_b:,}")

    best_b, losses_b = train_model(
        enc_b, pol_b, frames, actions, states,
        is_augmented=True, epochs=args.epochs)

    # ── Comparison ──
    print(f"\n{'='*65}")
    print(f"  Training Results")
    print(f"{'='*65}")
    print(f"  {'Model':<30} {'Params':>10} {'Best Loss':>10}")
    print(f"  {'-'*52}")
    print(f"  {'A: Standard CNN (6ch)':<30} {params_a:>10,} {best_a:>10.4f}")
    print(f"  {'B: Belief-Augmented (8ch)':<30} {params_b:>10,} {best_b:>10.4f}")

    improvement = (best_a - best_b) / best_a * 100
    winner = "B (Augmented)" if best_b < best_a else "A (Standard)"
    print(f"\n  Loss improvement: {improvement:+.1f}%")
    print(f"  Winner: {winner}")
    print(f"  Extra params for augmentation: {params_b - params_a:,}")

    # ── Eval ──
    if args.eval:
        print(f"\n-- Evaluating on gym-pusht ({args.n_eval} episodes) --")

        print(f"\n  Model A: Standard CNN")
        cov_a = eval_model(enc_a, pol_a, is_augmented=False,
                            n_episodes=args.n_eval)
        if cov_a:
            print(f"    Max coverage: {max(cov_a):.3f}")
            print(f"    Avg coverage: {np.mean(cov_a):.3f}")

        print(f"\n  Model B: Belief-Augmented")
        cov_b = eval_model(enc_b, pol_b, is_augmented=True,
                            n_episodes=args.n_eval)
        if cov_b:
            print(f"    Max coverage: {max(cov_b):.3f}")
            print(f"    Avg coverage: {np.mean(cov_b):.3f}")

        if cov_a and cov_b:
            print(f"\n  {'Model':<30} {'Max Cov':>8} {'Avg Cov':>8}")
            print(f"  {'-'*48}")
            print(f"  {'A: Standard':<30} {max(cov_a):>8.3f} {np.mean(cov_a):>8.3f}")
            print(f"  {'B: Augmented (VPP)':<30} {max(cov_b):>8.3f} {np.mean(cov_b):>8.3f}")

    print(f"\n{'='*65}")


if __name__ == "__main__":
    main()
