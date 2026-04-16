"""
train_pusht_from_demos.py — Learn from human demonstrations
=============================================================
Extracts state/action pairs from the LeRobot PushT dataset,
converts to our format, and retrains the NeMo flow policy
on real human demonstrations instead of scripted ones.

The dataset has:
  - observation.state: [agent_x, agent_y] in pixel coords (0-512)
  - action: [target_x, target_y] in pixel coords (0-512)
  - 206 episodes, 25,650 total frames
  - Videos of each episode

Usage:
    python train_pusht_from_demos.py
    python train_pusht_from_demos.py --epochs 100 --eval
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

import pandas as pd
import gymnasium
import gym_pusht


# ──────────────────────────────────────────────────────────────────────────────
# 1. Extract demonstrations
# ──────────────────────────────────────────────────────────────────────────────

def load_demos(data_dir: str = "data/lerobot_pusht"):
    """Load and parse LeRobot PushT demonstrations."""
    parquet_path = Path(data_dir) / "data" / "chunk-000" / "file-000.parquet"
    df = pd.read_parquet(parquet_path)

    print(f"  Loaded {len(df)} frames from {len(df['episode_index'].unique())} episodes")

    episodes = []
    for ep_idx in df['episode_index'].unique():
        ep = df[df['episode_index'] == ep_idx]

        states_raw = np.array(ep['observation.state'].tolist(), dtype=np.float32)
        actions = np.array(ep['action'].tolist(), dtype=np.float32)
        rewards = np.array(ep['next.reward'].tolist(), dtype=np.float32)
        success = ep['next.success'].iloc[-1]

        episodes.append({
            # Augment with block state from env if available
            'states': states_raw,      # (T, 2) agent position [0, 512]
            'actions': actions,    # (T, 2) target position [0, 512]
            'rewards': rewards,    # (T,) coverage reward
            'success': success,
            'length': len(ep),
            'max_reward': rewards.max(),
        })

    # Stats
    n_success = sum(1 for e in episodes if e['success'])
    lengths = [e['length'] for e in episodes]
    max_rewards = [e['max_reward'] for e in episodes]

    print(f"  Successful episodes: {n_success}/{len(episodes)}")
    print(f"  Episode lengths: {np.mean(lengths):.0f} avg, "
          f"{np.min(lengths)}-{np.max(lengths)} range")
    print(f"  Max coverage: {np.mean(max_rewards):.3f} avg, "
          f"{np.max(max_rewards):.3f} best")
    print(f"  State range: [{states_raw.min():.0f}, {states_raw.max():.0f}]")
    print(f"  Action range: [{actions.min():.0f}, {actions.max():.0f}]")

    return episodes


# ──────────────────────────────────────────────────────────────────────────────
# 2. Build training dataset
# ──────────────────────────────────────────────────────────────────────────────

class PushTDemoDataset:
    """
    Converts human demos into (obs, action_chunk) pairs for flow training.

    obs = [agent_x, agent_y] normalised to [0, 1]
    action_chunk = [a1, a2, ..., aH] normalised to [0, 1], each (2,)
    """

    def __init__(self, episodes, H=8, min_reward=0.0):
        self.H = H
        self.obs = []
        self.actions = []
        self.rewards = []

        for ep in episodes:
            states = ep['states'] / 512.0  # normalise to [0, 1]
            acts = ep['actions'] / 512.0
            rews = ep['rewards']

            T = len(states)
            for t in range(T - H):
                if rews[t] < min_reward:
                    continue
                self.obs.append(states[t])
                self.actions.append(acts[t:t+H])
                self.rewards.append(rews[t])

        self.obs = np.array(self.obs, dtype=np.float32)
        self.actions = np.array(self.actions, dtype=np.float32)
        self.rewards = np.array(self.rewards, dtype=np.float32)

        print(f"  Dataset: {len(self.obs)} samples, H={H}")
        print(f"  Obs shape: {self.obs.shape}")
        print(f"  Action chunk shape: {self.actions.shape}")

    def sample(self, batch_size):
        idx = np.random.randint(0, len(self.obs), batch_size)
        return (
            torch.from_numpy(self.obs[idx]),
            torch.from_numpy(self.actions[idx]),
            torch.from_numpy(self.rewards[idx]),
        )


# ──────────────────────────────────────────────────────────────────────────────
# 3. Flow policy (same architecture as PointMaze)
# ──────────────────────────────────────────────────────────────────────────────

class PushTEncoder(nn.Module):
    def __init__(self, obs_dim=2, d_latent=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, d_latent), nn.LayerNorm(d_latent),
        )
    def forward(self, obs):
        return self.net(obs)


class PushTFlowPolicy(nn.Module):
    """Flow matching policy conditioned on agent state."""

    def __init__(self, H=8, d_z=128, d_obs=2, d_action=2):
        super().__init__()
        self.H = H
        self.d_action = d_action
        # Input: z(128) + obs(2) + t(1) + x_t(H*2) = 128 + 2 + 1 + 16 = 147
        d_in = d_z + d_obs + 1 + H * d_action
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, H * d_action),
        )

    def forward(self, z, obs, t, x_t):
        inp = torch.cat([z, obs, t, x_t], dim=-1)
        return self.net(inp)

    @torch.no_grad()
    def sample(self, z, obs, n_steps=10):
        B = z.shape[0]
        x_t = torch.randn(B, self.H * self.d_action, device=z.device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t_val = torch.full((B, 1), i * dt, device=z.device)
            v = self.forward(z, obs, t_val, x_t)
            x_t = x_t + v * dt
        return x_t.view(B, self.H, self.d_action).clamp(0, 1)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Training
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    print("=" * 65)
    print("  Train NeMo Flow Policy on Human PushT Demonstrations")
    print("=" * 65)

    # Load demos
    print("\n── Loading demonstrations ──")
    episodes = load_demos(args.data_dir)

    # Build dataset
    print("\n── Building dataset ──")
    dataset = PushTDemoDataset(episodes, H=args.H)

    # Models
    encoder = PushTEncoder(obs_dim=2, d_latent=128)
    policy = PushTFlowPolicy(H=args.H, d_z=128, d_obs=2, d_action=2)

    params = list(encoder.parameters()) + list(policy.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    total_params = sum(p.numel() for p in params)
    print(f"\n  Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"  Policy params: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"  Total: {total_params:,}")

    # Training loop
    print(f"\n── Training ({args.epochs} epochs) ──")
    best_loss = float('inf')
    t0 = time.time()

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        n_batches = max(1, len(dataset.obs) // args.batch_size)

        for batch in range(n_batches):
            obs, action_chunks, rewards = dataset.sample(args.batch_size)

            # Encode
            z = encoder(obs)

            # Flow matching loss
            B = obs.shape[0]
            t = torch.rand(B, 1)
            noise = torch.randn_like(action_chunks.view(B, -1))
            x_0 = action_chunks.view(B, -1)  # target (real actions)
            x_t = (1 - t) * noise + t * x_0   # interpolate

            # Velocity field: v = x_0 - noise (flow from noise to target)
            v_gt = x_0 - noise

            # Predict
            v_pred = policy(z, obs, t, x_t)
            loss = F.mse_loss(v_pred, v_gt)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / n_batches

        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(f"  ep {epoch+1:3d}/{args.epochs}  "
                  f"loss={avg_loss:.4f}  ({elapsed:.1f}s)")

        if avg_loss < best_loss:
            best_loss = avg_loss

    elapsed = time.time() - t0
    print(f"\n  Training complete: {elapsed:.1f}s, best loss={best_loss:.4f}")

    # Save
    ckpt_dir = Path("checkpoints/pusht_human_flow")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "pusht_human_flow_best.pt"

    torch.save({
        'encoder': encoder.state_dict(),
        'policy': policy.state_dict(),
        'H': args.H,
        'loss': best_loss,
        'epochs': args.epochs,
        'n_demos': len(episodes),
        'n_samples': len(dataset.obs),
    }, ckpt_path)
    print(f"  Saved: {ckpt_path}")

    # ── Evaluate in real sim ──
    if args.eval:
        print(f"\n── Evaluating in gym-pusht ──")
        encoder.eval()
        policy.eval()

        env = gymnasium.make('gym_pusht/PushT-v0', obs_type='state')
        n_eval = 50
        successes = 0
        coverages = []

        for ep in range(n_eval):
            obs, info = env.reset(seed=42 + ep)
            coverage = info.get('coverage', 0)

            for step in range(300):
                # obs is (5,): agent_x, agent_y, block_x, block_y, block_angle
                # Our policy only uses agent position
                agent_pos = obs[:2].astype(np.float32) / 512.0
                obs_t = torch.from_numpy(agent_pos).unsqueeze(0)

                with torch.no_grad():
                    z = encoder(obs_t)
                    actions = policy.sample(z, obs_t, n_steps=10)

                # First action in chunk, scale to [0, 512]
                action = actions[0, 0].numpy() * 512.0
                action = np.clip(action, 0, 512).astype(np.float32)

                obs, reward, term, trunc, info = env.step(action)
                coverage = info.get('coverage', 0)

                if coverage >= 0.95:
                    successes += 1
                    break

            coverages.append(coverage)

            if (ep + 1) % 10 == 0:
                sr = successes / (ep + 1)
                avg_cov = np.mean(coverages)
                print(f"    ep {ep+1}/{n_eval}: SR={sr:.1%}, "
                      f"avg_cov={avg_cov:.3f}")

        env.close()

        sr = successes / n_eval
        avg_cov = np.mean(coverages)
        max_cov = np.max(coverages)

        print(f"\n  Results (n={n_eval}):")
        print(f"    SR (>= 0.95): {sr:.1%} ({successes}/{n_eval})")
        print(f"    Avg coverage: {avg_cov:.3f}")
        print(f"    Max coverage: {max_cov:.3f}")

        # Coverage histogram
        bins = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.01]
        print(f"    Coverage distribution:")
        for i in range(len(bins) - 1):
            count = sum(1 for c in coverages if bins[i] <= c < bins[i+1])
            bar = "#" * count
            print(f"      [{bins[i]:.2f}, {bins[i+1]:.2f}): {count:3d} {bar}")

    print(f"\n{'='*65}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/lerobot_pusht")
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
