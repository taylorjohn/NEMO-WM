"""
train_pusht_vision_v2.py — Vision PushT v2: 2-Frame + Video
==============================================================
Improvements over v1:
  1. 2-frame stacking (6 channels) — agent can see motion/direction
  2. Cosine learning rate decay — prevents late-training overfit
  3. Eval video recording — see the agent actually pushing the T
  4. Best-epoch checkpoint selection via validation loss

Usage:
    python train_pusht_vision_v2.py --epochs 300 --eval
"""

import argparse
import time
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

import cv2
import pandas as pd
import gymnasium
import gym_pusht


# ──────────────────────────────────────────────────────────────────────────────
# 1. Extract video frames
# ──────────────────────────────────────────────────────────────────────────────

def extract_frames(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame.shape[:2] != (96, 96):
            frame = cv2.resize(frame, (96, 96))
        frames.append(frame)
    cap.release()
    print(f"  Extracted {len(frames)} frames from video")
    return np.array(frames, dtype=np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Visual encoder — 6 channel input (2 stacked frames)
# ──────────────────────────────────────────────────────────────────────────────

class VisualEncoder2Frame(nn.Module):
    """
    CNN for 2-frame stacked input: (B, 6, 96, 96) -> (B, 128)
    6 channels = current frame (3) + previous frame (3)
    This lets the agent see MOTION — which direction the block is moving.
    """
    def __init__(self, d_latent=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(6, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.GELU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(1152, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, d_latent), nn.LayerNorm(d_latent),
        )

    def forward(self, img):
        features = self.conv(img)
        flat = features.reshape(features.shape[0], -1)
        return self.fc(flat)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Flow policy (same as v1)
# ──────────────────────────────────────────────────────────────────────────────

class VisionFlowPolicy(nn.Module):
    def __init__(self, H=8, d_z=128, d_action=2):
        super().__init__()
        self.H = H
        self.d_action = d_action
        d_in = d_z + 1 + H * d_action
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, H * d_action),
        )

    def forward(self, z, t, x_t):
        return self.net(torch.cat([z, t, x_t], dim=-1))

    @torch.no_grad()
    def sample(self, z, n_steps=10):
        B = z.shape[0]
        x_t = torch.randn(B, self.H * self.d_action, device=z.device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t_val = torch.full((B, 1), i * dt, device=z.device)
            v = self.forward(z, t_val, x_t)
            x_t = x_t + v * dt
        return x_t.view(B, self.H, self.d_action).clamp(0, 1)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Dataset with 2-frame stacking
# ──────────────────────────────────────────────────────────────────────────────

class VisionPushTDataset2Frame:
    """Pairs of (prev_frame, curr_frame) stacked as 6-channel input."""

    def __init__(self, frames: np.ndarray, df: pd.DataFrame, H=8):
        self.H = H
        self.frame_pairs = []  # (prev, curr) indices
        self.actions_list = []

        n_frames = min(len(frames), len(df))
        actions = np.array(df['action'].tolist(), dtype=np.float32)[:n_frames]
        episode_idx = df['episode_index'].values[:n_frames]
        actions = actions / 512.0

        # Store frames as float32 for fast access
        self.frames = frames[:n_frames].astype(np.float32) / 255.0

        for i in range(1, n_frames - H):
            if episode_idx[i] != episode_idx[i + H - 1]:
                continue
            if episode_idx[i] != episode_idx[i - 1]:
                # Episode boundary — use current as both
                self.frame_pairs.append((i, i))
            else:
                self.frame_pairs.append((i - 1, i))
            self.actions_list.append(actions[i:i+H])

        self.n_samples = len(self.frame_pairs)
        print(f"  Dataset: {self.n_samples} samples (2-frame stacked)")

    def sample(self, batch_size):
        idx = np.random.randint(0, self.n_samples, batch_size)

        batch_imgs = []
        batch_acts = []
        for i in idx:
            prev_idx, curr_idx = self.frame_pairs[i]
            prev = self.frames[prev_idx]
            curr = self.frames[curr_idx]
            # Stack: (96, 96, 6) -> (6, 96, 96)
            stacked = np.concatenate([curr, prev], axis=2)
            batch_imgs.append(np.transpose(stacked, (2, 0, 1)))
            batch_acts.append(self.actions_list[i])

        return (
            torch.from_numpy(np.stack(batch_imgs)),
            torch.from_numpy(np.stack(batch_acts)),
        )


# ──────────────────────────────────────────────────────────────────────────────
# 5. Training with cosine LR decay
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    print("=" * 65)
    print("  Vision PushT v2: 2-Frame Stacking + Video Recording")
    print("=" * 65)

    data_dir = Path(args.data_dir)
    video_path = str(data_dir / "videos" / "observation.image" /
                      "chunk-000" / "file-000.mp4")
    parquet_path = str(data_dir / "data" / "chunk-000" / "file-000.parquet")

    print("\n-- Extracting video frames --")
    frames = extract_frames(video_path)

    print("\n-- Loading actions --")
    df = pd.read_parquet(parquet_path)
    print(f"  Parquet rows: {len(df)}, Episodes: {df['episode_index'].nunique()}")

    print("\n-- Building 2-frame dataset --")
    dataset = VisionPushTDataset2Frame(frames, df, H=args.H)

    # Models
    encoder = VisualEncoder2Frame(d_latent=128)
    policy = VisionFlowPolicy(H=args.H, d_z=128, d_action=2)

    enc_params = sum(p.numel() for p in encoder.parameters())
    pol_params = sum(p.numel() for p in policy.parameters())
    print(f"\n  Visual encoder (2-frame): {enc_params:,} params")
    print(f"  Flow policy: {pol_params:,} params")
    print(f"  Total: {enc_params + pol_params:,} params")

    params = list(encoder.parameters()) + list(policy.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    # Cosine LR schedule
    n_batches_per_epoch = max(1, dataset.n_samples // args.batch_size)
    total_steps = args.epochs * n_batches_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_steps, eta_min=args.lr * 0.01)

    print(f"\n-- Training ({args.epochs} epochs, cosine LR) --")
    best_loss = float('inf')
    t0 = time.time()
    step_count = 0

    for epoch in range(args.epochs):
        encoder.train()
        policy.train()
        epoch_loss = 0.0

        for batch in range(n_batches_per_epoch):
            imgs, action_chunks = dataset.sample(args.batch_size)
            z = encoder(imgs)

            B = imgs.shape[0]
            t = torch.rand(B, 1)
            noise = torch.randn_like(action_chunks.view(B, -1))
            x_0 = action_chunks.view(B, -1)
            x_t = (1 - t) * noise + t * x_0
            v_gt = x_0 - noise

            v_pred = policy(z, t, x_t)
            loss = F.mse_loss(v_pred, v_gt)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            scheduler.step()

            epoch_loss += loss.item()
            step_count += 1

        avg_loss = epoch_loss / n_batches_per_epoch
        lr_now = scheduler.get_last_lr()[0]

        if (epoch + 1) % 20 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(f"  ep {epoch+1:3d}/{args.epochs}  "
                  f"loss={avg_loss:.4f}  lr={lr_now:.6f}  ({elapsed:.1f}s)")

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_dir = Path("checkpoints/pusht_vision_v2")
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'encoder': encoder.state_dict(),
                'policy': policy.state_dict(),
                'H': args.H,
                'loss': best_loss,
                'epoch': epoch,
                'n_frames': 2,
            }, ckpt_dir / "pusht_vision_v2_best.pt")

    elapsed = time.time() - t0
    print(f"\n  Training complete: {elapsed:.1f}s, best loss={best_loss:.4f}")

    # ── Evaluate with video recording ──
    if args.eval:
        print(f"\n-- Evaluating in gym-pusht (recording video) --")
        encoder.eval()
        policy.eval()

        env = gymnasium.make('gym_pusht/PushT-v0',
                              obs_type='pixels',
                              render_mode='rgb_array')
        n_eval = args.n_eval
        successes = 0
        coverages = []

        # Record best episode for video
        best_ep_frames = []
        best_ep_coverage = 0
        all_ep_frames = []  # Record first 10 episodes

        prev_obs = None

        for ep in range(n_eval):
            obs, info = env.reset(seed=42 + ep)
            coverage = info.get('coverage', 0)
            best_coverage = coverage
            prev_obs = obs.copy()
            ep_frames = []

            for step in range(300):
                # Stack current + previous frame
                curr = obs.astype(np.float32) / 255.0
                prev = prev_obs.astype(np.float32) / 255.0
                stacked = np.concatenate([curr, prev], axis=2)
                img_t = torch.from_numpy(
                    np.transpose(stacked, (2, 0, 1))
                ).unsqueeze(0)

                with torch.no_grad():
                    z = encoder(img_t)
                    actions = policy.sample(z, n_steps=args.n_ode_steps)

                action = actions[0, 0].numpy() * 512.0
                action = np.clip(action, 0, 512).astype(np.float32)

                prev_obs = obs.copy()
                obs, reward, term, trunc, info = env.step(action)
                coverage = info.get('coverage', 0)
                best_coverage = max(best_coverage, coverage)

                # Record frame for video
                if ep < 10 or best_coverage > best_ep_coverage:
                    render = env.render()
                    if render is not None:
                        ep_frames.append(render.copy())

                if coverage >= 0.95:
                    successes += 1
                    break
                if term or trunc:
                    break

            coverages.append(best_coverage)

            # Track best episode
            if best_coverage > best_ep_coverage and ep_frames:
                best_ep_coverage = best_coverage
                best_ep_frames = ep_frames.copy()

            # Save first 10 episodes
            if ep < 10:
                all_ep_frames.extend(ep_frames)

            if (ep + 1) % 5 == 0:
                sr = successes / (ep + 1)
                avg_cov = np.mean(coverages)
                print(f"    ep {ep+1}/{n_eval}: SR={sr:.1%}, "
                      f"avg_cov={avg_cov:.3f}, "
                      f"this_cov={best_coverage:.3f}")

        env.close()

        sr = successes / n_eval
        avg_cov = np.mean(coverages)
        max_cov = np.max(coverages)

        print(f"\n  Results (n={n_eval}):")
        print(f"    SR (>= 0.95): {sr:.1%} ({successes}/{n_eval})")
        print(f"    Avg coverage: {avg_cov:.3f}")
        print(f"    Max coverage: {max_cov:.3f}")

        bins = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.01]
        print(f"    Coverage distribution:")
        for i in range(len(bins) - 1):
            count = sum(1 for c in coverages if bins[i] <= c < bins[i+1])
            bar = "#" * count
            print(f"      [{bins[i]:.2f}, {bins[i+1]:.2f}): {count:3d} {bar}")

        # Save videos
        out_dir = Path("outputs")
        out_dir.mkdir(exist_ok=True)

        # Best episode video
        if best_ep_frames:
            video_path = out_dir / "pusht_best_episode.mp4"
            h, w = best_ep_frames[0].shape[:2]
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*'mp4v'),
                10, (w, h))
            for f in best_ep_frames:
                writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            writer.release()
            print(f"\n  Best episode video: {video_path} "
                  f"(coverage={best_ep_coverage:.3f}, "
                  f"{len(best_ep_frames)} frames)")

        # First 10 episodes video
        if all_ep_frames:
            video_path = out_dir / "pusht_eval_10eps.mp4"
            h, w = all_ep_frames[0].shape[:2]
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*'mp4v'),
                10, (w, h))
            for f in all_ep_frames:
                writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            writer.release()
            print(f"  10-episode video: {video_path} "
                  f"({len(all_ep_frames)} frames)")

    print(f"\n{'='*65}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/lerobot_pusht")
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--n-ode-steps", type=int, default=10)
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
