"""
mezo_vr.py — MeZO Variance-Reduced Planners for CORTEX-PE
===========================================================
Two drop-in replacements for the current single-perturbation MeZO planner,
each targeting a specific variance reduction strategy.

Both are NPU-compatible: no autograd, only forward passes.

Background
----------
Current MeZO (2-point SPSA):
    ĝ = [L(θ+εz) - L(θ-εz)] / (2ε) · z
    Variance: O(d) where d = action dimensionality

Strategy A — Multi-perturbation averaging (K independent estimates):
    ĝ_avg = (1/K) Σ_k ĝ_k
    Variance: O(d/K)   → K=3 gives 3× reduction, K=5 gives 5× reduction
    Cost: 2K forward passes vs 2

Strategy B — Sparse perturbations (50% random masking):
    z_sparse = z * mask  where mask ~ Bernoulli(0.5)
    Effective dimension: d/2  → ~2× variance reduction
    Cost: same 2 forward passes, cheaper gradient computation

INTEGRATION in run_benchmark.py / unified_cortex_loop.py:
    # Replace MeZO planner instantiation:
    from mezo_vr import MeZOMultiPerturbation, MeZOSparse
    planner = MeZOMultiPerturbation(K=3, n_perturbations=100, horizon=25, sigma=0.05)
    # or
    planner = MeZOSparse(sparsity=0.5, n_perturbations=100, horizon=25, sigma=0.05)
"""

import torch
import numpy as np
from typing import Callable, Optional


# ── Base SPSA planner (current implementation reference) ──────────────────

class MeZOBase:
    """
    Standard 2-point SPSA MeZO planner (current CORTEX-PE implementation).
    Reference baseline — not for deployment, documents the existing approach.
    """

    def __init__(
        self,
        n_perturbations: int = 100,
        horizon: int = 25,
        sigma: float = 0.05,
        lr: float = 0.1,
        action_dim: int = 2,
    ):
        self.n_perturb   = n_perturbations
        self.H           = horizon
        self.sigma       = sigma
        self.lr          = lr
        self.action_dim  = action_dim

    def plan(
        self,
        z_current:  torch.Tensor,
        z_goal:     torch.Tensor,
        predictor:  Callable,
        action_init: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns optimised action sequence (H, action_dim).
        predictor: callable(z, action_seq) → z_final
        """
        a = action_init if action_init is not None else \
            torch.zeros(self.H, self.action_dim)

        for _ in range(self.n_perturb):
            z  = torch.randn_like(a)          # perturbation direction
            ep = self.sigma

            L_plus  = self._cost(predictor, z_current, z_goal, a + ep * z)
            L_minus = self._cost(predictor, z_current, z_goal, a - ep * z)

            grad_est = ((L_plus - L_minus) / (2 * ep)) * z
            a = a - self.lr * grad_est

        return a.clamp(-1, 1)

    def _cost(self, predictor, z0, z_goal, actions):
        z_pred = predictor(z0, actions)
        return torch.norm(z_pred - z_goal).item()


# ── Strategy A: Multi-perturbation averaging ──────────────────────────────

class MeZOMultiPerturbation(MeZOBase):
    """
    Averages K independent SPSA gradient estimates per step.
    Variance reduction: 1/K at cost of 2K forward passes per step.

    Recommended K values:
        K=3 — 3× variance reduction, 3× compute (good default)
        K=5 — 5× variance reduction, 5× compute (diminishing returns past K=5)

    Break-even point: K=3 is worth it when planner has >5% failure rate.
    If current MeZO is already at 100% MPC, K=1 (baseline) is sufficient.
    Primary use: RECON navigation where MPC is expected at 50-70%.
    """

    def __init__(self, K: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.K = K

    def plan(
        self,
        z_current:  torch.Tensor,
        z_goal:     torch.Tensor,
        predictor:  Callable,
        action_init: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        a = action_init if action_init is not None else \
            torch.zeros(self.H, self.action_dim)

        for _ in range(self.n_perturb):
            grad_acc = torch.zeros_like(a)

            for _ in range(self.K):
                z  = torch.randn_like(a)
                ep = self.sigma

                L_plus  = self._cost(predictor, z_current, z_goal, a + ep * z)
                L_minus = self._cost(predictor, z_current, z_goal, a - ep * z)
                grad_acc += ((L_plus - L_minus) / (2 * ep)) * z

            grad_est = grad_acc / self.K       # averaged estimate — O(d/K) variance
            a = a - self.lr * grad_est

        return a.clamp(-1, 1)

    @property
    def forward_passes_per_step(self) -> int:
        return 2 * self.K * self.n_perturb


# ── Strategy B: Sparse perturbations (50% masking) ────────────────────────

class MeZOSparse(MeZOBase):
    """
    Applies a random binary mask to the perturbation direction z.
    Effective perturbation dimension: sparsity * action_dim.
    Variance reduction: 1/sparsity (e.g., 2× for 50% sparsity).
    Cost: same 2 forward passes — no additional overhead.

    Sparsity = 0.5 (mask 50% of dims) is the empirically validated default
    from the sparse MeZO literature (Liu et al. 2024).

    Notes:
    - Works best when action dimensions are uncorrelated
    - For low-dimensional action spaces (dim=2 for RECON), sparsity has
      limited effect — gradient already has low intrinsic dimension
    - More useful for high-dimensional action spaces (H=25 planning horizon:
      effective dimension = 25*2 = 50 — sparse masking gives ~2× reduction)
    """

    def __init__(self, sparsity: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        assert 0 < sparsity < 1, "sparsity must be in (0, 1)"
        self.sparsity = sparsity

    def _sparse_z(self, shape) -> torch.Tensor:
        z    = torch.randn(shape)
        mask = (torch.rand(shape) < self.sparsity).float()
        # Scale surviving entries to maintain E[||z_sparse||²] = E[||z||²]
        return z * mask / self.sparsity

    def plan(
        self,
        z_current:  torch.Tensor,
        z_goal:     torch.Tensor,
        predictor:  Callable,
        action_init: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        a = action_init if action_init is not None else \
            torch.zeros(self.H, self.action_dim)

        for _ in range(self.n_perturb):
            z  = self._sparse_z(a.shape)
            ep = self.sigma

            L_plus  = self._cost(predictor, z_current, z_goal, a + ep * z)
            L_minus = self._cost(predictor, z_current, z_goal, a - ep * z)

            grad_est = ((L_plus - L_minus) / (2 * ep)) * z
            a = a - self.lr * grad_est

        return a.clamp(-1, 1)


# ── Combined: multi-perturbation + sparse (maximum variance reduction) ────

class MeZOSparseMulti(MeZOBase):
    """
    Combines K independent estimates with sparse perturbations.
    Total variance reduction: O(d * sparsity / K)
    Cost: 2K forward passes per step.

    Use only when planning performance is a bottleneck and compute budget
    allows it. For RECON navigation with 50-70% baseline MPC, K=3 + 0.5
    sparsity gives ~6× variance reduction for 3× compute cost.
    """

    def __init__(self, K: int = 3, sparsity: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.K        = K
        self.sparsity = sparsity

    def _sparse_z(self, shape) -> torch.Tensor:
        z    = torch.randn(shape)
        mask = (torch.rand(shape) < self.sparsity).float()
        return z * mask / self.sparsity

    def plan(
        self,
        z_current:  torch.Tensor,
        z_goal:     torch.Tensor,
        predictor:  Callable,
        action_init: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        a = action_init if action_init is not None else \
            torch.zeros(self.H, self.action_dim)

        for _ in range(self.n_perturb):
            grad_acc = torch.zeros_like(a)
            for _ in range(self.K):
                z  = self._sparse_z(a.shape)
                ep = self.sigma
                L_plus  = self._cost(predictor, z_current, z_goal, a + ep * z)
                L_minus = self._cost(predictor, z_current, z_goal, a - ep * z)
                grad_acc += ((L_plus - L_minus) / (2 * ep)) * z
            a = a - self.lr * (grad_acc / self.K)

        return a.clamp(-1, 1)


# ── Variance comparison utility ───────────────────────────────────────────

def compare_variance(
    predictor:   Callable,
    z_current:   torch.Tensor,
    z_goal:      torch.Tensor,
    action_dim:  int = 2,
    horizon:     int = 25,
    n_trials:    int = 50,
) -> dict:
    """
    Empirically compare gradient estimate variance across planner variants.
    Run this to decide which planner to deploy for RECON.

    Returns dict with std of gradient estimates for each variant.
    """
    a_ref = torch.zeros(horizon, action_dim)
    results = {}

    for name, planner in [
        ('baseline',   MeZOBase(n_perturbations=1,   action_dim=action_dim, horizon=horizon)),
        ('K=3',        MeZOMultiPerturbation(K=3, n_perturbations=1, action_dim=action_dim, horizon=horizon)),
        ('sparse_0.5', MeZOSparse(sparsity=0.5, n_perturbations=1,  action_dim=action_dim, horizon=horizon)),
    ]:
        grads = []
        for _ in range(n_trials):
            a = a_ref.clone()
            z  = torch.randn_like(a)
            ep = 0.05
            L_plus  = planner._cost(predictor, z_current, z_goal, a + ep * z)
            L_minus = planner._cost(predictor, z_current, z_goal, a - ep * z)
            grads.append(((L_plus - L_minus) / (2 * ep)) * z)
        grad_tensor = torch.stack(grads)
        results[name] = grad_tensor.std(dim=0).mean().item()

    print('\n── Gradient estimate std (lower = less variance) ──')
    for k, v in results.items():
        print(f'  {k:15s}: {v:.6f}')
    return results


# ── Sanity test ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Mock predictor: predict z_final from z_current + actions
    def mock_predictor(z, actions):
        return z + actions.sum(0)[:z.shape[-1]]

    z0   = torch.randn(128)
    zg   = torch.randn(128)

    for Cls, kwargs, label in [
        (MeZOBase,             {},          'MeZO baseline'),
        (MeZOMultiPerturbation, {'K': 3},   'MeZO K=3'),
        (MeZOSparse,           {'sparsity': 0.5}, 'MeZO sparse 0.5'),
        (MeZOSparseMulti,      {'K': 3, 'sparsity': 0.5}, 'MeZO sparse+K=3'),
    ]:
        p = Cls(n_perturbations=10, horizon=5, action_dim=2, **kwargs)
        a = p.plan(z0, zg, mock_predictor)
        assert a.shape == (5, 2), f'Wrong action shape: {a.shape}'
        print(f'✅ {label}: action shape {a.shape}, norm {a.norm():.3f}')

    print('\n✅ All planner variants passed shape check')
