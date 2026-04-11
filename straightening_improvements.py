"""
straightening_improvements.py
==============================
Two improvements from Wang et al. 2026 (arXiv:2603.12231):

  1. lambda_curv = 0.1  (was 1.0 — 10× reduction)
     Paper Fig 11: 0.1 is optimal for [agg] pooling head across all envs.
     Over-straightening (1.0) degrades multi-step rollout quality.

  2. Linear-ramp intermediate cost weighting in the MPC loop.
     w_t = (t+1)/H  →  earlier steps contribute less, terminal most.
     Provides denser planning gradient vs uniform or terminal-only.

Integration:
    from straightening_improvements import (
        weighted_rollout_cost,
        measure_curvature,
        LAMBDA_CURV_OPTIMAL,
    )
"""

import torch
import torch.nn.functional as F
from typing import Literal

# ── 1. Optimal lambda_curv ─────────────────────────────────────────────────

LAMBDA_CURV_OPTIMAL: float = 0.1
"""
Wang et al. Fig 11: λ=0.1 is best for [agg] pooling head.
λ=1.0 over-straightens → encoder loses dynamic information.
λ=0.01 is too weak for reliable corner navigation.

Phase 2 training command:
    python train_distillation.py --phase 2 \\
        --resume ./checkpoints/maze_weak_sigreg/cortex_student_phase1_final.pt \\
        --data ./phase2_frames \\
        --lambda-curv 0.1 \\
        --lambda-sigreg 5.0 \\
        --steps 6000 --gc none \\
        --out ./checkpoints/maze_weak_sigreg_straight
"""


# ── 2. Weighted intermediate rollout cost ──────────────────────────────────

WeightMode = Literal['linear', 'exponential', 'uniform', 'terminal']

def weighted_rollout_cost(
    z_rollout: torch.Tensor,
    z_goal: torch.Tensor,
    mode: WeightMode = 'linear',
    terminal_weight: float = 1.0,
) -> torch.Tensor:
    """
    Weighted sum of step-wise L2 distances along a predicted latent rollout.

    Wang et al. 2026 §B.1: weighted intermediate loss gives more direct
    paths, especially around corners. Linear ramp is paper default.

    Args:
        z_rollout:       (H, D) — predicted latents across planning horizon.
        z_goal:          (D,)   — target latent.
        mode:            Weighting scheme:
                           'linear'      w_t = (t+1)/H          [paper default]
                           'exponential' w_t = exp((t+1)/H) / Z
                           'uniform'     w_t = 1.0              [current CORTEX-PE]
                           'terminal'    only final step
        terminal_weight: Extra multiplier on the last step. 1.0 = no change.
    Returns:
        Scalar cost tensor (differentiable).
    """
    H, D = z_rollout.shape

    if mode == 'terminal':
        return torch.norm(z_rollout[-1] - z_goal)

    # Per-step L2 distances: (H,)
    dists = torch.norm(z_rollout - z_goal.unsqueeze(0), dim=-1)

    t = torch.arange(1, H + 1, dtype=torch.float32, device=z_rollout.device)

    if mode == 'linear':
        weights = t / H

    elif mode == 'exponential':
        weights = torch.exp(t / H)
        weights = weights / weights.sum() * H   # rescale: sum = H

    elif mode == 'uniform':
        weights = torch.ones(H, device=z_rollout.device)

    else:
        raise ValueError(
            f"mode must be 'linear', 'exponential', 'uniform' or 'terminal', got {mode!r}"
        )

    weights = weights.clone()
    weights[-1] = weights[-1] * terminal_weight

    # Mean (not sum) so cost is horizon-length independent
    return (weights * dists).sum() / H


# ── 3. Curvature diagnostic ────────────────────────────────────────────────

def measure_curvature(latents: torch.Tensor) -> float:
    """
    Mean cosine similarity between consecutive latent velocity vectors.
    Wang et al. Eq. (4):  C = mean( cosine_sim(v_t, v_{t+1}) )

    Interpretation:
        ~1.0  → perfectly straight trajectory (ideal)
        ~0.0  → random-walk (untrained / DINOv2 baseline)
        > 0.5 → well-straightened encoder, sufficient for MPC

    Args:
        latents: (T, D) — sequence of latent states along a trajectory.
    Returns:
        Scalar float. Returns nan if T < 3.
    """
    if latents.shape[0] < 3:
        return float('nan')

    v = latents[1:] - latents[:-1]                          # (T-1, D)
    cos = F.cosine_similarity(v[:-1], v[1:], dim=-1)        # (T-2,)
    return cos.mean().item()
