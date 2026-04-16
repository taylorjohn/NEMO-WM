"""
test_vpp_real.py — VPP Augment on Real Human Demo Data
========================================================
Uses the 25,650 real human demonstrations from lerobot/pusht.
Generates 96x96 frames from state positions, tests Standard
vs Belief-Augmented CNN with real action distributions.

This is the definitive VPP test — real data, not synthetic random.
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

DATA_DIR = Path("data/lerobot_pusht")


class StandardEncoder(nn.Module):
    """Standard 2-frame CNN encoder."""
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
    """VPP-inspired: 2-frame RGB + 2ch belief maps."""
    def __init__(self, d_latent=128, d_belief=4):
        super().__init__()
        self.belief_to_maps = nn.Sequential(
            nn.Linear(d_belief, 64),
            nn.GELU(),
            nn.Linear(64, 2 * 12 * 12),
        )
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
        B = x_rgb.shape[0]
        maps = self.belief_to_maps(belief).view(B, 2, 12, 12)
        maps = F.interpolate(maps, size=(96, 96), mode='bilinear',
                              align_corners=False)
        maps = torch.sigmoid(maps)
        x = torch.cat([x_rgb, maps], dim=1)
        return self.net(x)


class FlowPolicy(nn.Module):
    def __init__(self, d_z=128, d_action=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_z + d_action + 1, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, d_action),
        )

    def forward(self, z, x_t, t):
        return self.net(torch.cat([z, x_t, t.unsqueeze(-1)], dim=-1))

    def sample(self, z, n_steps=10):
        B = z.shape[0]
        x = torch.randn(B, 2)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B,), i * dt)
            x = x + self.forward(z, x, t) * dt
        return x


def load_real_data():
    """Load real PushT data from parquet + generate frames."""
    import pyarrow.parquet as pq

    parquet_path = DATA_DIR / "data" / "chunk-000" / "file-000.parquet"
    if not parquet_path.exists():
        print("ERROR: No parquet file found at", parquet_path)
        return None, None, None

    df = pq.read_table(str(parquet_path)).to_pandas()
    states = np.stack(df['observation.state'].values).astype(np.float32)
    actions = np.stack(df['action'].values).astype(np.float32)
    episodes = df['episode_index'].values

    n = len(states)
    print(f"  Loaded {n} real human demo samples")
    print(f"  Episodes: {len(np.unique(episodes))}")
    print(f"  State range: [{states.min():.0f}, {states.max():.0f}]")
    print(f"  Action range: [{actions.min():.0f}, {actions.max():.0f}]")

    # Generate 96x96 frames from state positions
    print(f"  Generating {n} frames from state positions...")
    frames = np.zeros((n, 96, 96, 3), dtype=np.uint8)

    pos_min = states.min(axis=0)
    pos_max = states.max(axis=0)
    pos_range = pos_max - pos_min + 1e-8
    norm_pos = ((states - pos_min) / pos_range * 88 + 4).astype(int)
    norm_pos = np.clip(norm_pos, 4, 91)

    for i in range(n):
        frames[i, :, :, :] = 40
        # T-shape target
        frames[i, 30:35, 35:60, 1] = 120
        frames[i, 35:55, 43:52, 1] = 120
        # Agent
        cx, cy = norm_pos[i, 0], norm_pos[i, 1]
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                px, py = cx + dx, cy + dy
                if 0 <= px < 96 and 0 <= py < 96 and dx*dx + dy*dy <= 9:
                    frames[i, py, px, 0] = 200
                    frames[i, py, px, 2] = 100

    # Belief states: [agent_x, agent_y, prev_action_x, prev_action_y]
    beliefs = np.zeros((n, 4), dtype=np.float32)
    beliefs[:, :2] = states / 512.0
    beliefs[1:, 2:] = actions[:-1] / 512.0

    print(f"  Frames: {frames.shape}")
    print(f"  Beliefs: {beliefs.shape}")
    return frames, actions / 512.0, beliefs


def train_model(encoder, policy, frames, actions, beliefs,
                is_augmented, epochs=200, lr=3e-4, seed=42):
    """Train one encoder+policy pair."""
    torch.manual_seed(seed)
    np.random.seed(seed)

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
        idx = np.random.choice(n, size=min(256, n), replace=False)

        curr = torch.from_numpy(
            frames[idx + 1].astype(np.float32) / 255.0
        ).permute(0, 3, 1, 2)
        prev = torch.from_numpy(
            frames[idx].astype(np.float32) / 255.0
        ).permute(0, 3, 1, 2)
        x_rgb = torch.cat([curr, prev], dim=1)

        action_gt = torch.from_numpy(actions[idx + 1])
        belief = torch.from_numpy(beliefs[idx + 1])

        if is_augmented:
            z = encoder(x_rgb, belief)
        else:
            z = encoder(x_rgb)

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
        best_loss = min(best_loss, loss.item())

        if epoch % 50 == 0 or epoch == 1:
            print(f"    ep {epoch:4d}/{epochs}  loss={loss.item():.4f}  "
                  f"best={best_loss:.4f}  lr={scheduler.get_last_lr()[0]:.6f}")

    return best_loss, losses


def eval_gym(encoder, policy, is_augmented, n_episodes=20):
    """Evaluate on gym-pusht."""
    try:
        import gymnasium
        import gym_pusht
    except ImportError:
        print("    gym-pusht not available")
        return None

    env = gymnasium.make('gym_pusht/PushT-v0',
                          obs_type='pixels', render_mode='rgb_array')
    encoder.eval()
    policy.eval()
    coverages = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=42 + ep)
        prev_obs = obs.copy()
        best_cov = 0
        agent_pos = np.array([0.5, 0.5], dtype=np.float32)
        prev_action = np.zeros(2, dtype=np.float32)

        for step in range(300):
            curr = obs.astype(np.float32) / 255.0
            prev = prev_obs.astype(np.float32) / 255.0
            stacked = np.concatenate([curr, prev], axis=2)
            x_rgb = torch.from_numpy(
                np.transpose(stacked, (2, 0, 1))).unsqueeze(0)

            with torch.no_grad():
                if is_augmented:
                    b = torch.tensor(
                        [[agent_pos[0], agent_pos[1],
                          prev_action[0], prev_action[1]]],
                        dtype=torch.float32)
                    z = encoder(x_rgb, b)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--n-eval", type=int, default=20)
    ap.add_argument("--runs", type=int, default=3,
                    help="Number of runs for variance")
    args = ap.parse_args()

    print("=" * 65)
    print("  VPP Augmentation on REAL Human Demo Data")
    print("  25,650 real PushT demonstrations")
    print("=" * 65)

    print("\n-- Loading real data --")
    frames, actions, beliefs = load_real_data()
    if frames is None:
        return

    all_results = []

    for run in range(1, args.runs + 1):
        seed = 42 + run * 100
        print(f"\n{'='*65}")
        print(f"  Run {run}/{args.runs} (seed={seed})")
        print(f"{'='*65}")

        # Model A: Standard
        print(f"\n-- Model A: Standard CNN (6ch) --")
        enc_a = StandardEncoder(d_latent=128)
        pol_a = FlowPolicy(d_z=128)
        params_a = sum(p.numel() for p in enc_a.parameters()) + \
                   sum(p.numel() for p in pol_a.parameters())
        if run == 1:
            print(f"  Params: {params_a:,}")
        best_a, _ = train_model(enc_a, pol_a, frames, actions, beliefs,
                                 is_augmented=False, epochs=args.epochs,
                                 seed=seed)

        # Model B: Belief-Augmented
        print(f"\n-- Model B: Belief-Augmented CNN (8ch, VPP) --")
        enc_b = BeliefAugmentedEncoder(d_latent=128, d_belief=4)
        pol_b = FlowPolicy(d_z=128)
        params_b = sum(p.numel() for p in enc_b.parameters()) + \
                   sum(p.numel() for p in pol_b.parameters())
        if run == 1:
            print(f"  Params: {params_b:,}")
        best_b, _ = train_model(enc_b, pol_b, frames, actions, beliefs,
                                 is_augmented=True, epochs=args.epochs,
                                 seed=seed)

        improvement = (best_a - best_b) / best_a * 100
        all_results.append({
            'run': run, 'loss_a': best_a, 'loss_b': best_b,
            'improvement': improvement
        })

        print(f"\n  Run {run}: A={best_a:.4f}, B={best_b:.4f}, "
              f"VPP={improvement:+.1f}%")

    # Summary
    print(f"\n{'='*65}")
    print(f"  Summary across {args.runs} runs")
    print(f"{'='*65}")
    print(f"  {'Run':<6} {'Standard':>10} {'VPP':>10} {'Improvement':>12}")
    print(f"  {'-'*40}")
    for r in all_results:
        print(f"  {r['run']:<6} {r['loss_a']:>10.4f} {r['loss_b']:>10.4f} "
              f"{r['improvement']:>+11.1f}%")

    avg_a = np.mean([r['loss_a'] for r in all_results])
    avg_b = np.mean([r['loss_b'] for r in all_results])
    avg_imp = np.mean([r['improvement'] for r in all_results])
    std_imp = np.std([r['improvement'] for r in all_results])

    print(f"  {'Mean':<6} {avg_a:>10.4f} {avg_b:>10.4f} {avg_imp:>+11.1f}%")
    print(f"  Std improvement: {std_imp:.1f}%")
    print(f"  Params overhead: {params_b - params_a:,} ({(params_b-params_a)/params_a*100:.1f}%)")

    # Eval
    if args.eval:
        print(f"\n-- Gym eval ({args.n_eval} episodes) --")
        # Use last run's models
        cov_a = eval_gym(enc_a, pol_a, False, args.n_eval)
        cov_b = eval_gym(enc_b, pol_b, True, args.n_eval)
        if cov_a and cov_b:
            print(f"\n  {'Model':<25} {'Max':>8} {'Avg':>8}")
            print(f"  {'-'*43}")
            print(f"  {'A: Standard':<25} {max(cov_a):>8.3f} {np.mean(cov_a):>8.3f}")
            print(f"  {'B: VPP Augmented':<25} {max(cov_b):>8.3f} {np.mean(cov_b):>8.3f}")

    print(f"\n{'='*65}")
    consistent = all(r['improvement'] > 0 for r in all_results)
    print(f"  VPP consistently better: {consistent}")
    print(f"  Average improvement: {avg_imp:+.1f}% +/- {std_imp:.1f}%")
    print(f"  Data source: REAL human demos (25,650 samples)")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
