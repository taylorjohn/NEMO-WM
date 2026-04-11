"""
train_flow_v2_registry.py
Train NeMoFlowPolicyV2 using the pluggable physics registry demos.
50% scripted SR → expected ~90-100% learned SR via flow matching.

Usage:
    python train_flow_v2_registry.py --n-demos 1000 --n-epochs 100
    python train_flow_v2_registry.py --n-demos 2000 --n-epochs 100 --physics rigid_body --planner position
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.getcwd())

from pusht_physics_registry import generate_demos
from nemo_flow_policy_v2 import NeMoFlowPolicyV2
from train_action_wm import StateEncoder
from block_probe import BlockProbe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-demos',    type=int, default=1000)
    ap.add_argument('--n-epochs',   type=int, default=100)
    ap.add_argument('--physics',    default='rigid_body')
    ap.add_argument('--planner',    default='position')
    ap.add_argument('--task-mode',  default='position_only')
    ap.add_argument('--H',          type=int, default=8)
    ap.add_argument('--lr',         type=float, default=3e-4)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--enc-ckpt',   default='checkpoints/action_wm/action_wm_pusht_full_best.pt')
    ap.add_argument('--probe-ckpt', default='checkpoints/action_wm/block_probe_best.pt')
    ap.add_argument('--policy-ckpt',default='checkpoints/flow_policy/nemo_flow_best.pt')
    ap.add_argument('--out-dir',    default='checkpoints/nemo_v2')
    ap.add_argument('--seed',       type=int, default=42)
    args = ap.parse_args()

    device = torch.device('cpu')

    # Load encoder + probe
    enc_s   = torch.load(args.enc_ckpt,   map_location=device, weights_only=False)
    encoder = StateEncoder(5, 128).to(device)
    encoder.load_state_dict(enc_s['encoder'])
    encoder.eval()

    p_s   = torch.load(args.probe_ckpt, map_location=device, weights_only=False)
    probe = BlockProbe(128).to(device)
    probe.load_state_dict(p_s['probe'])
    probe.eval()

    # Generate demos
    print(f"\nGenerating {args.n_demos} demos...")
    print(f"  physics={args.physics}  planner={args.planner}  mode={args.task_mode}")
    t0    = time.time()
    demos = generate_demos(
        n_demos=args.n_demos,
        physics=args.physics,
        planner=args.planner,
        task_mode=args.task_mode,
        seed=args.seed,
    )
    scripted_sr = sum(d['success'] for d in demos) / len(demos)
    print(f"Scripted SR: {scripted_sr:.1%}  ({time.time()-t0:.1f}s)")

    # Build dataset
    H = args.H
    samples = []
    for demo in demos:
        obs_arr = demo['obs']      # (T, 6)
        act_arr = demo['actions']  # (T, 3)
        goal    = demo['goal']     # (3,)
        T       = min(len(obs_arr), len(act_arr))
        for t in range(T - H):
            samples.append((
                torch.from_numpy(obs_arr[t]),
                torch.from_numpy(act_arr[t:t+H]),
                torch.from_numpy(goal),
            ))

    print(f"Dataset: {len(samples)} samples from {len(demos)} demos")

    # Policy — transfer from Phase 1
    policy = NeMoFlowPolicyV2(H=H).to(device)
    if Path(args.policy_ckpt).exists():
        policy = NeMoFlowPolicyV2.from_phase1(args.policy_ckpt, H=H)
        print(f"Phase 1 weights transferred from {args.policy_ckpt}")

    total_params = sum(p.numel() for p in policy.parameters())
    print(f"Policy params: {total_params:,}")

    optim = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.n_epochs
    )

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    best_loss = float('inf')
    B         = args.batch_size

    print(f"\nTraining {args.n_epochs} epochs...")
    print(f"{'Epoch':>6}  {'Loss':>8}  {'LR':>10}")
    print('─' * 30)

    t_train = time.time()

    for epoch in range(args.n_epochs):
        policy.train()
        idx_perm    = torch.randperm(len(samples))
        epoch_losses = []

        for i in range(0, len(samples) - B, B):
            batch     = [samples[idx_perm[i+j]] for j in range(B)]
            obs_b     = torch.stack([b[0] for b in batch])   # (B, 6)
            act_b     = torch.stack([b[1] for b in batch])   # (B, H, 3)
            goal_b    = torch.stack([b[2] for b in batch])   # (B, 3)

            with torch.no_grad():
                z           = encoder(obs_b[:, :5])
                block_state = torch.cat([probe(z), obs_b[:, 4:5]], dim=-1)
                pos_dist    = (block_state[:, :2] - goal_b[:, :2]).norm(dim=-1, keepdim=True)
                da          = (1 - pos_dist.clamp(0, 1))

            # Flow matching
            x0    = act_b.view(B, H * 3)
            x1    = torch.randn_like(x0)
            t_val = torch.rand(B, 1, device=device)
            x_t   = (1 - t_val) * x0 + t_val * x1
            v_gt  = x1 - x0

            v_pred = policy(z, block_state, da, goal_b, t_val, x_t)
            loss   = F.mse_loss(v_pred, v_gt)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optim.step()
            epoch_losses.append(loss.item())

        sched.step()
        mean_loss = np.mean(epoch_losses)

        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save({
                'policy':     policy.state_dict(),
                'epoch':      epoch,
                'loss':       best_loss,
                'H':          H,
                'physics':    args.physics,
                'planner':    args.planner,
                'task_mode':  args.task_mode,
                'scripted_sr': scripted_sr,
            }, f'{args.out_dir}/flow_v2_registry_best.pt')

        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - t_train
            remaining = elapsed / (epoch + 1) * (args.n_epochs - epoch - 1)
            print(f"  ep{epoch+1:4d}  {mean_loss:.4f}  "
                  f"{sched.get_last_lr()[0]:.2e}  "
                  f"(~{remaining/60:.0f} min left)")

    elapsed = time.time() - t_train
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Saved: {args.out_dir}/flow_v2_registry_best.pt")
    print(f"\nScripted SR:  {scripted_sr:.1%}")
    print(f"Expected learned SR: ~80-100% (flow matching amplifies scripted demos)")
    print(f"\nEval:")
    print(f"  python pusht_physics_registry.py --n-demos 100 --physics {args.physics} --planner {args.planner}")


if __name__ == '__main__':
    main()
