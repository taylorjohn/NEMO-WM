"""
train_phase3_value.py — CORTEX-PE Phase 3A: Value-Guided Latent Geometry

Fine-tunes the encoder head so that Euclidean distance between embedded
states approximates the negative goal-conditioned value function for a
reaching cost. This gives the MeZO planner a geometrically better cost
landscape — distances encode reachability, not just visual similarity.

Current state:
    ‖z_t − z_goal‖ ≈ visual similarity (how similar do two frames look?)

After Phase 3A:
    ‖z_t − z_goal‖ ≈ expected steps to reach goal (how far away is it?)

Based on:
    "Value-guided action planning with JEPA world models"
    arXiv:2601.00844, World Modeling Workshop ICLR 2026
    (IQL loss for JEPA representation shaping)

    "Optimal Goal-Reaching RL via Quasimetric Learning"
    Wang, Torralba, Isola, Zhang — ICML 2023

Loss:
    L_IQL = E[ℓ_τ(V(z_t) − (c(s,g) + γ * V(z_next)))]
    c(s,g) = 1  (binary reaching cost — 1 step per non-goal state)
    τ = 0.9     (optimistic value estimate)
    γ = 0.99    (discount factor)

Approach:
    "Sep" (from paper): train value head alone first, then jointly fine-tune
    encoder + value head. Separating phases prevents gradient interference.

Usage:
    python train_phase3_value.py \
        --encoder ./checkpoints/maze/cortex_student_phase2_final.pt \
        --trajectories ./benchmark_data/umaze/trajectories.npy \
        --env umaze \
        --out ./checkpoints/phase3_maze_value
    
    # Then benchmark the Phase 3 encoder:
    python calibrate_threshold.py \
        --encoder ./checkpoints/phase3_maze_value/encoder_phase3.pt \
        --option 1 --pred-dir ./predictors --pct 50 \
        --out ./benchmark_thresholds_phase3.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from student_encoder import StudentEncoder

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--encoder',      default='./checkpoints/maze/cortex_student_phase2_final.pt')
parser.add_argument('--trajectories', default='./benchmark_data/umaze/trajectories.npy')
parser.add_argument('--env',          default='umaze')
parser.add_argument('--out',          default='./checkpoints/phase3_maze_value')
parser.add_argument('--epochs-head',  type=int,   default=10, help='Value head pre-training epochs')
parser.add_argument('--epochs-joint', type=int,   default=5,  help='Joint fine-tuning epochs')
parser.add_argument('--batch',        type=int,   default=256)
parser.add_argument('--lr-head',      type=float, default=1e-3)
parser.add_argument('--lr-joint',     type=float, default=1e-4, help='Lower LR for encoder fine-tune')
parser.add_argument('--gamma',        type=float, default=0.99)
parser.add_argument('--tau',          type=float, default=0.9, help='IQL expectile')
parser.add_argument('--n-traj',       type=int,   default=500, help='Trajectories to use')
args = parser.parse_args()

Path(args.out).mkdir(parents=True, exist_ok=True)

TRANSFORM = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Dataset: transition pairs from trajectories ───────────────────────────────
class TransitionDataset(Dataset):
    """
    Samples (s_t, s_t+1, goal) triples from maze trajectories.
    Labels: d(s_t, goal) = number of steps to reach goal in this trajectory.
    
    This is the "ground truth" for value function supervision.
    """
    def __init__(self, traj_path, n_traj=500, transform=None):
        print(f"Loading trajectories: {traj_path}")
        trajs = np.load(traj_path, allow_pickle=True)
        if n_traj < len(trajs):
            idx   = np.random.choice(len(trajs), n_traj, replace=False)
            trajs = [trajs[i] for i in idx]

        self.transform = transform or TRANSFORM
        self.samples   = []  # (obs_t, obs_next, obs_goal, steps_to_goal)

        for traj in trajs:
            if isinstance(traj, dict):
                obs  = traj.get('observations', traj.get('obs', []))
                goal = traj.get('goal', obs[-1] if len(obs) > 0 else None)
            else:
                obs  = traj
                goal = obs[-1]

            if len(obs) < 2:
                continue

            T = len(obs)
            for t in range(T - 1):
                steps_to_goal = T - 1 - t  # steps remaining
                self.samples.append((obs[t], obs[t + 1], goal, steps_to_goal))

        print(f"  {len(self.samples)} transition samples from {len(trajs)} trajectories")

    def __len__(self):
        return len(self.samples)

    def _to_tensor(self, obs):
        """Convert observation (numpy array or path) to tensor."""
        if isinstance(obs, np.ndarray):
            if obs.dtype == np.uint8:
                img = Image.fromarray(obs)
            else:
                img = Image.fromarray((obs * 255).astype(np.uint8))
        elif isinstance(obs, str):
            img = Image.open(obs).convert('RGB')
        else:
            img = obs
        return self.transform(img.convert('RGB') if hasattr(img, 'convert') else img)

    def __getitem__(self, idx):
        obs_t, obs_next, goal, steps = self.samples[idx]
        return (
            self._to_tensor(obs_t),
            self._to_tensor(obs_next),
            self._to_tensor(goal),
            torch.tensor([float(steps)], dtype=torch.float32),
        )


# ── Value head ────────────────────────────────────────────────────────────────
class ValueHead(nn.Module):
    """
    Goal-conditioned value function: V(z_t, z_goal) → scalar
    Predicts negative expected steps to reach goal.
    Input: concat(z_t, z_goal) = 256-D
    ~16K params
    """
    def __init__(self, z_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim * 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, z_t, z_goal):
        return self.net(torch.cat([z_t, z_goal], dim=-1))


# ── IQL expectile loss ────────────────────────────────────────────────────────
def iql_value_loss(V_t, V_next, steps_to_goal, gamma=0.99, tau=0.9):
    """
    Bellman target: y = c(s,g) + γ * V(s_next, g)
    c(s,g) = 1 if s ≠ goal (1 step cost), approximated by steps_to_goal > 0
    
    Expectile regression: ℓ_τ(V_t - y)
    τ=0.9: optimistic value estimate (upper quantile)
    """
    # Binary reaching cost: 1 step everywhere except at goal
    cost    = (steps_to_goal > 0).float()
    # Bellman target
    target  = cost + gamma * V_next.detach()
    # Expectile loss
    diff    = V_t - target
    weight  = torch.where(diff >= 0,
                          torch.full_like(diff, tau),
                          torch.full_like(diff, 1.0 - tau))
    return (weight * diff.pow(2)).mean()


# ── Load encoder ──────────────────────────────────────────────────────────────
def load_encoder(path, requires_grad=False):
    ckpt    = torch.load(path, map_location='cpu')
    encoder = StudentEncoder()
    state   = ckpt.get('model', ckpt)
    new_state = {}
    for k, v in state.items():
        k2 = k.replace('backbone.stem.0', 'backbone.block1.0') \
              .replace('backbone.stem.1', 'backbone.block1.1') \
              .replace('backbone.stem.2', 'backbone.block1.2')
        new_state[k2] = v
    encoder.load_state_dict(new_state, strict=False)
    if not requires_grad:
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad_(False)
    return encoder


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}")
    print(f"  PHASE 3A — VALUE-GUIDED LATENT GEOMETRY")
    print(f"  Environment: {args.env}")
    print(f"  IQL τ={args.tau}  γ={args.gamma}")
    print(f"  Stage 1 (head only):   {args.epochs_head} epochs")
    print(f"  Stage 2 (joint):       {args.epochs_joint} epochs")
    print(f"{'='*60}\n")

    # ── Stage 1: Train value head with frozen encoder ────────────────────────
    print("Stage 1: Pre-training value head (encoder frozen)...")
    encoder = load_encoder(args.encoder, requires_grad=False)
    value   = ValueHead(z_dim=128)
    print(f"  Encoder:    {sum(p.numel() for p in encoder.parameters()):,} params (frozen)")
    print(f"  ValueHead:  {sum(p.numel() for p in value.parameters()):,} params")

    try:
        dataset = TransitionDataset(args.trajectories, n_traj=args.n_traj)
    except Exception as e:
        print(f"  Could not load trajectories: {e}")
        print(f"  Ensure trajectories.npy exists at: {args.trajectories}")
        return

    n_val   = max(1, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)

    opt1 = torch.optim.AdamW(value.parameters(), lr=args.lr_head)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=args.epochs_head)

    results = {'stage1': [], 'stage2': []}

    for epoch in range(1, args.epochs_head + 1):
        value.train()
        train_loss = 0.0
        t0 = time.time()
        for obs_t, obs_next, goal, steps in train_loader:
            with torch.no_grad():
                z_t    = encoder(obs_t)
                z_next = encoder(obs_next)
                z_goal = encoder(goal)
            V_t    = value(z_t,    z_goal)
            V_next = value(z_next, z_goal)
            loss   = iql_value_loss(V_t, V_next, steps, args.gamma, args.tau)
            opt1.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(value.parameters(), 1.0)
            opt1.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        value.eval()
        val_loss = 0.0
        with torch.no_grad():
            for obs_t, obs_next, goal, steps in val_loader:
                z_t    = encoder(obs_t)
                z_next = encoder(obs_next)
                z_goal = encoder(goal)
                V_t    = value(z_t,    z_goal)
                V_next = value(z_next, z_goal)
                val_loss += iql_value_loss(V_t, V_next, steps, args.gamma, args.tau).item()
        val_loss /= len(val_loader)
        sch1.step()

        print(f"  [Stage1] Epoch {epoch:>2}/{args.epochs_head} | "
              f"train={train_loss:.4f}  val={val_loss:.4f}  ({time.time()-t0:.1f}s)")
        results['stage1'].append({'epoch': epoch, 'train': round(train_loss, 6),
                                  'val': round(val_loss, 6)})

    torch.save(value.state_dict(), f'{args.out}/value_head_stage1.pt')

    # ── Stage 2: Joint fine-tuning with lower LR ─────────────────────────────
    print(f"\nStage 2: Joint fine-tuning (encoder + value head, lr={args.lr_joint})...")
    encoder_ft = load_encoder(args.encoder, requires_grad=True)
    # Only fine-tune the head layers, not the backbone
    for name, p in encoder_ft.named_parameters():
        if 'head' in name or 'pool' in name:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

    trainable = sum(p.numel() for p in encoder_ft.parameters() if p.requires_grad)
    print(f"  Encoder trainable params: {trainable:,} (head only)")

    opt2_params = list(filter(lambda p: p.requires_grad, encoder_ft.parameters())) + \
                  list(value.parameters())
    opt2 = torch.optim.AdamW(opt2_params, lr=args.lr_joint, weight_decay=1e-4)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=args.epochs_joint)

    best_val = float('inf')
    for epoch in range(1, args.epochs_joint + 1):
        encoder_ft.train()
        value.train()
        train_loss = 0.0
        t0 = time.time()
        for obs_t, obs_next, goal, steps in train_loader:
            z_t    = encoder_ft(obs_t)
            z_next = encoder_ft(obs_next)
            z_goal = encoder_ft(goal)
            V_t    = value(z_t,    z_goal)
            V_next = value(z_next, z_goal)
            loss   = iql_value_loss(V_t, V_next, steps, args.gamma, args.tau)
            opt2.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(opt2_params, 1.0)
            opt2.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        encoder_ft.eval()
        value.eval()
        val_loss = 0.0
        with torch.no_grad():
            for obs_t, obs_next, goal, steps in val_loader:
                z_t    = encoder_ft(obs_t)
                z_next = encoder_ft(obs_next)
                z_goal = encoder_ft(goal)
                V_t    = value(z_t,    z_goal)
                V_next = value(z_next, z_goal)
                val_loss += iql_value_loss(V_t, V_next, steps, args.gamma, args.tau).item()
        val_loss /= len(val_loader)
        sch2.step()

        print(f"  [Stage2] Epoch {epoch:>2}/{args.epochs_joint} | "
              f"train={train_loss:.4f}  val={val_loss:.4f}  ({time.time()-t0:.1f}s)")
        results['stage2'].append({'epoch': epoch, 'train': round(train_loss, 6),
                                  'val': round(val_loss, 6)})

        if val_loss < best_val:
            best_val = val_loss
            torch.save({'model': encoder_ft.state_dict()},
                       f'{args.out}/encoder_phase3.pt')
            torch.save(value.state_dict(), f'{args.out}/value_head_phase3.pt')

    json.dump(results, open(f'{args.out}/training_results.json', 'w'), indent=2)

    print(f"\n{'='*60}")
    print(f"  PHASE 3A COMPLETE — {args.env}")
    print(f"  Encoder saved → {args.out}/encoder_phase3.pt")
    print(f"  Value head  → {args.out}/value_head_phase3.pt")
    print(f"\n  Next steps:")
    print(f"  python calibrate_threshold.py \\")
    print(f"      --encoder {args.out}/encoder_phase3.pt \\")
    print(f"      --option 1 --pred-dir ./predictors --pct 50 \\")
    print(f"      --out ./benchmark_thresholds_phase3.json")
    print(f"  python run_benchmark.py \\")
    print(f"      --encoder {args.out}/encoder_phase3.pt \\")
    print(f"      --env {args.env} --planner mezo \\")
    print(f"      --threshold-file ./benchmark_thresholds_phase3.json")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
