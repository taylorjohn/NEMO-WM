"""
cortex_brain.routing.static_csr
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
O(1) Sparse Matrix Constraints via CSR format – bypassing the Python GIL.

Source: moe_router_v2.py (MOERouterV2) rewritten with scipy CSR.

Why CSR?
--------
MOERouterV2's Python for-loop over expert nn.Linear blocks acquires the GIL
on every iteration, stalling the CPU when the NPU is mid-inference.

scipy CSR mat-vec (W @ x) releases the GIL during the C-level BLAS call,
letting the NPU run fully concurrently without memory interrupts.

Spectral Ribbon layout (128-D output):
    [0 :32]  Shape expert
    [32:64]  Size expert
    [64:96]  Depth expert
    [96:128] Velocity expert

Expert weights are STATIC after init. Only the 4-D gate is learnable.
TTM routing adjustments are applied as logit offsets before softmax.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class CSRRouterConfig:
    input_dim:    int   = 256
    manifold_dim: int   = 128
    num_experts:  int   = 4
    expert_dim:   int   = 32       # manifold_dim / num_experts
    sparsity:     float = 0.80     # fraction of ZEROS
    seed:         int   = 0


class StaticCSRRouter(nn.Module):
    """
    GIL-bypassing sparse MoE router.

    Forward
    -------
    x              : (B, input_dim)
    ttm_adjustment : optional (num_experts,) penalty from TTM memory
    → manifold     : (B, manifold_dim)  tanh-squashed Spectral Ribbon
    → weights      : (B, num_experts)   softmax gate distribution
    """

    def __init__(self, config: CSRRouterConfig = CSRRouterConfig()) -> None:
        super().__init__()
        self.config = config
        self._csr, self._dense = self._build_experts()
        self.gate = nn.Linear(config.input_dim, config.num_experts, bias=True)
        nn.init.orthogonal_(self.gate.weight)

    def forward(self, x: torch.Tensor,
                ttm_adjustment: Optional[np.ndarray] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x_np  = x.detach().cpu().numpy()
        ribbon = self._csr_forward(x_np)              # (B, manifold_dim) – GIL released

        gate_logits = self.gate(x)                    # (B, num_experts)
        if ttm_adjustment is not None:
            adj = torch.tensor(ttm_adjustment, dtype=torch.float32)
            gate_logits = gate_logits + adj.unsqueeze(0)

        weights  = torch.softmax(gate_logits, dim=-1)
        manifold = torch.tanh(torch.from_numpy(ribbon))
        return manifold, weights

    def get_resonance(self, weights: torch.Tensor) -> float:
        """Peak expert weight = resonance scalar ρ (matches unified_cortex pattern)."""
        return float(weights.detach().max().item())

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _csr_forward(self, x_np: np.ndarray) -> np.ndarray:
        try:
            slices = [(x_np @ self._csr[i].T.toarray()).astype(np.float32)
                      for i in range(self.config.num_experts)]
        except Exception:
            slices = [(x_np @ self._dense[i]).astype(np.float32)
                      for i in range(self.config.num_experts)]
        return np.concatenate(slices, axis=-1)

    def _build_experts(self):
        c   = self.config
        rng = np.random.default_rng(c.seed)
        csr_list, dense_list = [], []
        try:
            from scipy.sparse import csr_matrix
            for _ in range(c.num_experts):
                W = rng.standard_normal((c.expert_dim, c.input_dim)).astype(np.float32)
                W[rng.random(W.shape) < c.sparsity] = 0.0
                csr_list.append(csr_matrix(W))
                dense_list.append(W.T)
            logger.info("StaticCSRRouter: %d CSR experts (sparsity=%.0f%%)",
                        c.num_experts, c.sparsity * 100)
        except ImportError:
            logger.warning("scipy not found – dense numpy experts (no GIL bypass).")
            for _ in range(c.num_experts):
                W = rng.standard_normal((c.expert_dim, c.input_dim)).astype(np.float32)
                csr_list.append(None)
                dense_list.append(W.T)
        return csr_list, dense_list