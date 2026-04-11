"""
eval_imagination_rollout.py — NeMo-WM Sprint B Validation
===========================================================
Validates the ProprioDecoder imagination quality quantitatively.

Tests:
  1. Single-step reconstruction MAE (static)
  2. H-step imagination rollout error growth
  3. Cortisol signal vs actual drift correlation
  4. Imagination vs real trajectory comparison

Usage:
    python eval_imagination_rollout.py \
        --enc-ckpt  checkpoints/action_wm/action_wm_pusht_full_best.pt \
        --dec-ckpt  checkpoints/nemo_v2/decoder_best.pt \
        --trans-ckpt checkpoints/action_wm/action_wm_pusht_full_best.pt \
        --n-rollouts 100 --H 16
"""

import argparse
import sys
import os
import time
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
from proprio_decoder import ProprioDecoder, ImaginationRollout
from train_action_wm import StateEncoder, ActionConditionedTransition


def load_models(enc_ckpt: str, dec_ckpt: str, trans_ckpt: str, device: torch.device):
    """Load encoder, decoder, and transition model."""

    # Encoder
    enc_state = torch.load(enc_ckpt, map_location=device, weights_only=False)
    encoder   = StateEncoder(5, 128).to(device)
    encoder.load_state_dict(enc_state['encoder'])
    encoder.eval()
    print(f"  Encoder:    ep={enc_state.get('epoch','?')}")

    # Decoder
    dec_state = torch.load(dec_ckpt, map_location=device, weights_only=False)
    decoder   = ProprioDecoder(128, 5, d_hidden=128, output_act='sigmoid').to(device)
    decoder.load_state_dict(dec_state['decoder'])
    decoder.eval()
    print(f"  Decoder:    MAE={dec_state.get('mae', [0])[0]:.4f} (static)")

    # Transition
    trans_state = torch.load(trans_ckpt, map_location=device, weights_only=False)
    transition  = ActionConditionedTransition(128, 2).to(device)
    transition.load_state_dict(trans_state['transition'])
    transition.eval()
    print(f"  Transition: loaded")

    return encoder, decoder, transition


def generate_pusht_rollout(T: int = 32, seed: int = 0) -> tuple:
    """
    Generate a synthetic PushT rollout for testing.
    Returns (obs_sequence, action_sequence) as numpy arrays.
    obs: (T, 5) — [agent_x, agent_y, block_x, block_y, block_angle]
    actions: (T-1, 2) — [target_x, target_y]
    """
    rng = np.random.RandomState(seed)

    agent_pos   = rng.uniform(0.1, 0.4, 2)
    block_pos   = rng.uniform(0.3, 0.7, 2)
    block_angle = rng.uniform(0, 1)
    goal        = rng.uniform(0.4, 0.7, 2)

    obs_list    = []
    action_list = []

    for t in range(T):
        obs = np.array([
            agent_pos[0], agent_pos[1],
            block_pos[0], block_pos[1],
            block_angle,
        ], dtype=np.float32)
        obs_list.append(obs)

        if t < T - 1:
            # Move toward block then toward goal
            if np.linalg.norm(agent_pos - block_pos) > 0.1:
                target = block_pos
            else:
                target = goal

            action = np.clip(target + rng.normal(0, 0.01, 2), 0, 1).astype(np.float32)
            action_list.append(action)

            # Physics
            agent_pos += (target - agent_pos) * 0.3 + rng.normal(0, 0.008, 2)
            agent_pos  = np.clip(agent_pos, 0, 1)

            if np.linalg.norm(agent_pos - block_pos) < 0.08:
                push       = (agent_pos - block_pos) * 0.15
                block_pos  = np.clip(block_pos - push, 0, 1)
                block_angle = (block_angle + rng.normal(0, 0.02)) % 1.0

    return np.array(obs_list), np.array(action_list)


def eval_static_reconstruction(
    encoder: nn.Module,
    decoder: nn.Module,
    n_samples: int,
    device: torch.device,
) -> dict:
    """Test single-step z → obs reconstruction."""
    obs_gt = torch.rand(n_samples, 5, device=device)

    with torch.no_grad():
        z        = encoder(obs_gt)
        obs_pred = decoder(z)

    mae_per_dim = (obs_pred - obs_gt).abs().mean(dim=0).cpu().numpy()
    mae_mean    = mae_per_dim.mean()

    return {
        'mae_mean':    mae_mean,
        'mae_per_dim': mae_per_dim,
        'pass':        mae_mean < 0.05,
    }


def eval_imagination_rollout(
    encoder:    nn.Module,
    decoder:    nn.Module,
    transition: nn.Module,
    n_rollouts: int,
    H:          int,
    device:     torch.device,
) -> dict:
    """
    Test H-step imagination quality.

    For each rollout:
      1. Generate real trajectory (obs_0..obs_H, actions_0..actions_{H-1})
      2. Encode obs_0 → z_0
      3. Imagine H steps: z_t+1 = transition(z_t, a_t), obs_t+1 = decoder(z_t+1)
      4. Compare imagined obs vs real obs at each step
      5. Track drift and cortisol signal

    Returns per-step MAE, drift, and cortisol correlation.
    """
    rollout_module = ImaginationRollout(encoder, transition, decoder, gamma=0.95)

    step_errors  = [[] for _ in range(H + 1)]
    drift_scores = [[] for _ in range(H)]
    cortisol_sig = []

    for r in range(n_rollouts):
        obs_np, act_np = generate_pusht_rollout(T=H + 1, seed=r)

        obs_gt  = torch.from_numpy(obs_np).to(device)    # (H+1, 5)
        actions = torch.from_numpy(act_np).to(device)    # (H, 2)

        # Encode initial state
        with torch.no_grad():
            z0 = encoder(obs_gt[0:1])   # (1, 128)

        # H-step imagination with grad for drift
        actions_batch = actions.unsqueeze(0)   # (1, H, 2)
        obs_gt_batch  = obs_gt.unsqueeze(0)    # (1, H+1, 5)

        _, obs_traj, drift = rollout_module.imagine_with_grad(z0, actions_batch)

        # Per-step error
        for t in range(H + 1):
            err = (obs_traj[0, t] - obs_gt_batch[0, t]).abs().mean().item()
            step_errors[t].append(err)

        # Drift per step
        for t in range(H):
            drift_scores[t].append(drift[0, t].item())

        # Cortisol = mean drift across rollout
        cortisol_sig.append(drift[0].mean().item())

    # Aggregate
    mean_step_errors = [np.mean(step_errors[t]) for t in range(H + 1)]
    mean_drift       = [np.mean(drift_scores[t]) for t in range(H)]

    # Error growth rate (how fast imagination degrades)
    if len(mean_step_errors) > 1 and mean_step_errors[0] > 0:
        error_growth = mean_step_errors[-1] / (mean_step_errors[0] + 1e-8)
    else:
        error_growth = 0.0

    return {
        'mean_step_errors': mean_step_errors,
        'mean_drift':       mean_drift,
        'cortisol_mean':    np.mean(cortisol_sig),
        'cortisol_std':     np.std(cortisol_sig),
        'error_at_H1':      mean_step_errors[1] if len(mean_step_errors) > 1 else 0,
        'error_at_H':       mean_step_errors[-1],
        'error_growth':     error_growth,
        'viable_horizon':   next(
            (t for t, e in enumerate(mean_step_errors) if e > 0.10),
            H
        ),
    }


def print_results(static: dict, rollout: dict, H: int):
    """Print formatted results."""

    print(f"\n{'='*60}")
    print(f"  Imagination Rollout Eval — NeMo-WM Sprint B")
    print(f"{'='*60}")

    print(f"\n── Static Reconstruction ──────────────────────────────")
    print(f"  Mean MAE:    {static['mae_mean']:.4f}  "
          f"({'✅ PASS' if static['pass'] else '❌ FAIL'})")
    dims = ['agent_x', 'agent_y', 'block_x', 'block_y', 'angle']
    for i, (d, m) in enumerate(zip(dims, static['mae_per_dim'])):
        bar = '█' * int(m * 200)
        print(f"  {d:12s}: {m:.4f}  {bar}")

    print(f"\n── {H}-Step Imagination Rollout ────────────────────────")
    print(f"  {'Step':6s}  {'MAE':8s}  {'Drift':8s}  {'Status'}")
    print(f"  {'─'*45}")

    for t in range(min(H + 1, len(rollout['mean_step_errors']))):
        mae    = rollout['mean_step_errors'][t]
        drift  = rollout['mean_drift'][t] if t < len(rollout['mean_drift']) else 0
        status = '✅' if mae < 0.05 else ('⚠️ ' if mae < 0.10 else '❌')
        bar    = '█' * min(int(mae * 100), 30)
        print(f"  t={t:<5d}  {mae:.4f}    {drift:.4f}    {status} {bar}")

    print(f"\n── Summary ─────────────────────────────────────────────")
    print(f"  Error at t=1:      {rollout['error_at_H1']:.4f}")
    print(f"  Error at t={H}:     {rollout['error_at_H']:.4f}")
    print(f"  Error growth:      {rollout['error_growth']:.2f}×")
    print(f"  Viable horizon:    t={rollout['viable_horizon']} steps "
          f"(MAE < 0.10)")
    print(f"  Cortisol signal:   {rollout['cortisol_mean']:.4f} "
          f"± {rollout['cortisol_std']:.4f}")

    print(f"\n── Interpretation ──────────────────────────────────────")
    vh = rollout['viable_horizon']
    eg = rollout['error_growth']

    if eg < 2.0 and vh >= H:
        print(f"  ✅ EXCELLENT: imagination stable for full {H} steps")
        print(f"     Error growth {eg:.1f}× — suitable for multi-step planning")
    elif eg < 5.0 and vh >= H // 2:
        print(f"  ✅ GOOD: imagination viable for {vh} steps")
        print(f"     Error growth {eg:.1f}× — suitable for short-horizon planning")
    elif vh >= 2:
        print(f"  ⚠️  PARTIAL: imagination degrades after {vh} steps")
        print(f"     Error growth {eg:.1f}× — use short rollouts only")
    else:
        print(f"  ❌ POOR: imagination breaks immediately")
        print(f"     Transition model may not preserve obs information")

    print(f"\n  Cortisol ↔ drift correlation:")
    print(f"    High cortisol = fast-moving transitions = more drift")
    print(f"    This signal should gate when to re-observe vs imagine")
    print(f"{'='*60}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--enc-ckpt',   default='checkpoints/action_wm/action_wm_pusht_full_best.pt')
    ap.add_argument('--dec-ckpt',   default='checkpoints/nemo_v2/decoder_best.pt')
    ap.add_argument('--trans-ckpt', default='checkpoints/action_wm/action_wm_pusht_full_best.pt')
    ap.add_argument('--n-rollouts', type=int, default=100)
    ap.add_argument('--H',          type=int, default=16)
    ap.add_argument('--device',     default='cpu')
    args = ap.parse_args()

    device = torch.device(args.device)

    print(f"Loading models...")
    encoder, decoder, transition = load_models(
        args.enc_ckpt, args.dec_ckpt, args.trans_ckpt, device
    )

    print(f"\nRunning static reconstruction ({2000} samples)...")
    static = eval_static_reconstruction(encoder, decoder, 2000, device)

    print(f"Running {args.H}-step imagination ({args.n_rollouts} rollouts)...")
    t0      = time.time()
    rollout = eval_imagination_rollout(
        encoder, decoder, transition,
        args.n_rollouts, args.H, device
    )
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")

    print_results(static, rollout, args.H)


if __name__ == '__main__':
    main()
