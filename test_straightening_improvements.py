"""
test_straightening_improvements.py
====================================
pytest suite for straightening_improvements.py
Run: pytest test_straightening_improvements.py -v
"""

import math
import pytest
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, '/home/claude')
from straightening_improvements import (
    LAMBDA_CURV_OPTIMAL,
    weighted_rollout_cost,
    measure_curvature,
)


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def rollout():
    torch.manual_seed(0)
    return torch.randn(25, 128)   # H=25, D=128

@pytest.fixture
def goal():
    torch.manual_seed(1)
    return torch.randn(128)


# ── lambda_curv ────────────────────────────────────────────────────────────

def test_lambda_curv_value():
    assert LAMBDA_CURV_OPTIMAL == 0.1, "Paper optimal is 0.1, not 1.0"

def test_lambda_curv_is_float():
    assert isinstance(LAMBDA_CURV_OPTIMAL, float)


# ── weighted_rollout_cost: output shape & type ─────────────────────────────

@pytest.mark.parametrize("mode", ['linear', 'exponential', 'uniform', 'terminal'])
def test_output_is_scalar(rollout, goal, mode):
    cost = weighted_rollout_cost(rollout, goal, mode=mode)
    assert cost.shape == torch.Size([]), f"Expected scalar, got {cost.shape}"

@pytest.mark.parametrize("mode", ['linear', 'exponential', 'uniform', 'terminal'])
def test_output_is_positive(rollout, goal, mode):
    cost = weighted_rollout_cost(rollout, goal, mode=mode)
    assert cost.item() > 0

def test_invalid_mode_raises(rollout, goal):
    with pytest.raises(ValueError, match="mode must be"):
        weighted_rollout_cost(rollout, goal, mode='quadratic')


# ── weighted_rollout_cost: weighting correctness ───────────────────────────

def test_linear_weights_favour_later_steps():
    """Linear ramp should give same cost as manually computed weighted sum."""
    torch.manual_seed(42)
    H, D = 10, 8
    z_rollout = torch.randn(H, D)
    z_goal = torch.zeros(D)

    dists = torch.norm(z_rollout, dim=-1)
    t = torch.arange(1, H + 1, dtype=torch.float32)
    weights = t / H
    expected = (weights * dists).sum() / H

    actual = weighted_rollout_cost(z_rollout, z_goal, mode='linear')
    assert torch.isclose(actual, expected, atol=1e-5), (
        f"Linear cost mismatch: {actual:.6f} vs {expected:.6f}"
    )

def test_uniform_equals_mean_distance():
    """Uniform mode should equal mean of per-step L2 distances."""
    torch.manual_seed(7)
    H, D = 8, 16
    z_rollout = torch.randn(H, D)
    z_goal = torch.zeros(D)

    dists = torch.norm(z_rollout, dim=-1)
    expected = dists.mean()
    actual = weighted_rollout_cost(z_rollout, z_goal, mode='uniform')
    assert torch.isclose(actual, expected, atol=1e-5)

def test_terminal_equals_final_step_distance():
    """Terminal mode should equal ||z[-1] - z_goal||."""
    torch.manual_seed(3)
    H, D = 12, 32
    z_rollout = torch.randn(H, D)
    z_goal = torch.randn(D)

    expected = torch.norm(z_rollout[-1] - z_goal)
    actual = weighted_rollout_cost(z_rollout, z_goal, mode='terminal')
    assert torch.isclose(actual, expected, atol=1e-5)

def test_terminal_weight_scales_last_step():
    """terminal_weight=2.0 should give a different (larger) cost than 1.0."""
    torch.manual_seed(5)
    H, D = 10, 16
    z_rollout = torch.randn(H, D)
    z_goal = torch.randn(D)

    cost_1 = weighted_rollout_cost(z_rollout, z_goal, mode='linear', terminal_weight=1.0)
    cost_2 = weighted_rollout_cost(z_rollout, z_goal, mode='linear', terminal_weight=2.0)
    assert cost_2 > cost_1, "terminal_weight=2.0 should increase cost"

def test_linear_strictly_greater_than_terminal(rollout, goal):
    """
    Linear cost uses all H steps; since distances to goal are >0,
    it will generally be != terminal-only cost.
    (Can't guarantee > since weighting normalises by H, but must differ.)
    """
    cost_linear   = weighted_rollout_cost(rollout, goal, mode='linear')
    cost_terminal = weighted_rollout_cost(rollout, goal, mode='terminal')
    assert not torch.isclose(cost_linear, cost_terminal, atol=1e-4), (
        "Linear and terminal costs should differ for random inputs"
    )

def test_cost_differentiable():
    """Cost must be differentiable w.r.t. z_rollout for gradient-based planning."""
    torch.manual_seed(9)
    z_rollout = torch.randn(10, 32, requires_grad=True)
    z_goal    = torch.randn(32)
    cost = weighted_rollout_cost(z_rollout, z_goal, mode='linear')
    cost.backward()
    assert z_rollout.grad is not None
    assert not torch.isnan(z_rollout.grad).any()

def test_horizon_independence():
    """Mean normalisation means cost should be in similar range for H=5 vs H=25."""
    torch.manual_seed(11)
    goal = torch.zeros(32)
    # Both rollouts drawn from same distribution
    short = torch.ones(5,  32)
    long_  = torch.ones(25, 32)
    c_short = weighted_rollout_cost(short, goal, mode='linear').item()
    c_long  = weighted_rollout_cost(long_,  goal, mode='linear').item()
    # Both should be ~sqrt(32) ≈ 5.66 — within 20%
    assert abs(c_short - c_long) / c_long < 0.20, (
        f"Horizon sensitivity too high: short={c_short:.3f}, long={c_long:.3f}"
    )


# ── measure_curvature ──────────────────────────────────────────────────────

def test_curvature_straight_trajectory():
    """Straight-line trajectory should give curvature ≈ 1.0."""
    direction = F.normalize(torch.randn(64), dim=0)
    latents = torch.stack([direction * t for t in range(20)])
    c = measure_curvature(latents)
    assert abs(c - 1.0) < 1e-4, f"Expected ~1.0, got {c:.4f}"

def test_curvature_random_walk_near_zero():
    """Random walk should give curvature close to 0."""
    torch.manual_seed(42)
    latents = torch.cumsum(torch.randn(200, 64), dim=0)
    c = measure_curvature(latents)
    assert abs(c) < 0.2, f"Expected near 0, got {c:.4f}"

def test_curvature_returns_nan_for_short_seq():
    latents = torch.randn(2, 32)   # need at least 3
    assert math.isnan(measure_curvature(latents))

def test_curvature_returns_float():
    latents = torch.randn(10, 32)
    assert isinstance(measure_curvature(latents), float)

def test_curvature_range():
    """Curvature should be in [-1, 1] for any input (it's cosine similarity)."""
    torch.manual_seed(0)
    for _ in range(20):
        latents = torch.randn(15, 64)
        c = measure_curvature(latents)
        assert -1.0 <= c <= 1.0, f"Out of range: {c}"
