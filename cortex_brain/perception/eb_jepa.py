"""
cortex_brain.perception.eb_jepa
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Energy-Based JEPA (EB-JEPA) Latent Predictor + Proprioception.

Sandwich Norm (INT8 / NPU-ready)
---------------------------------
Standard predictor block:
    h → Linear → LayerNorm → Linear → output

Sandwich Norm predictor block:
    h → LayerNorm → Linear → RMSNorm → Linear → RMSNorm → DynamicTanh

The outer RMSNorm bounds activation outlier spikes that cause INT8
quantisation loss on AMD Ryzen NPU. DynamicTanh constrains output to a
learnable [-1, 1] range, enabling lossless INT8 mapping.

Enable with: EBJEPAConfig(sandwich_norm=True)  [default: True]
"""
from __future__ import annotations
import logging, time
from dataclasses import dataclass, field
from typing import Tuple
import numpy as np
import psutil
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Primitives: RMSNorm + DynamicTanh
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalisation (Zhang & Sennrich 2019).

    No mean subtraction, no bias — simpler than LayerNorm.
    Bounds activation magnitude for INT8 quantisation without the
    centering bias that causes attention sink drift.

        y = x / RMS(x) * gamma       RMS(x) = sqrt(mean(x^2) + eps)
    """
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps   = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.scale


class DynamicTanh(nn.Module):
    """
    Learnable-range tanh gate for output bounding (Chen et al. 2025).

        y = tanh(alpha * x)     alpha in R+, initialised to 1.0

    alpha is learned per-channel, adapting compression range per feature.
    Combined with RMSNorm gives guaranteed bounded output dynamic range —
    prerequisite for lossless INT8 NPU mapping.
    """
    def __init__(self, dim: int, alpha_init: float = 1.0) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.full((dim,), alpha_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.alpha * x)


# ---------------------------------------------------------------------------
# ProprioceptionPulse
# ---------------------------------------------------------------------------

class ProprioceptionPulse:
    """
    Phase 18-Aux: Hardware & Kinetic Telemetry Injection.
    Produces (1, 3) tensor: [cpu_temp_normalised, fovea_vel_x, fovea_vel_y].
    """
    def __init__(self) -> None:
        self._last_pos  = np.zeros(2, dtype=np.float32)
        self._last_time = time.time()

    def get_hardware_thermal(self) -> float:
        try:
            temps = psutil.sensors_temperatures()
            for key in ("k10temp", "coretemp", "cpu_thermal"):
                if key in temps and temps[key]:
                    return float(temps[key][0].current)
        except Exception:
            pass
        return 45.0

    def fovea_velocity(self, fovea_xy: Tuple[float, float]) -> np.ndarray:
        now = time.time()
        dt  = now - self._last_time + 1e-6
        pos = np.array(fovea_xy, dtype=np.float32)
        vel = (pos - self._last_pos) / dt
        self._last_pos  = pos
        self._last_time = now
        return vel

    def get_aux_tensor(self, fovea_xy: Tuple[float, float] = (0.5, 0.5)) -> torch.Tensor:
        temp = self.get_hardware_thermal()
        vel  = self.fovea_velocity(fovea_xy)
        tn   = float(np.clip((temp - 30.0) / 60.0, 0.0, 1.0))
        return torch.tensor([[tn, vel[0], vel[1]]], dtype=torch.float32)


# ---------------------------------------------------------------------------
# CJEPAPredictor
# ---------------------------------------------------------------------------

@dataclass
class EBJEPAConfig:
    latent_dim:       int   = 128
    compressed_dim:   int   = 16
    aux_dim:          int   = 3
    hidden_dim:       int   = 64
    planning_horizon: int   = 5
    num_candidates:   int   = 64
    gamma_horizon:    float = 2.0
    sandwich_norm:    bool  = True   # RMSNorm + DynamicTanh for INT8 NPU


class CJEPAPredictor(nn.Module):
    """
    Causal-JEPA with optional Sandwich Norm for INT8/NPU deployment.

    sandwich_norm=True  (default):
        LayerNorm(h) -> Linear -> RMSNorm -> Linear -> RMSNorm -> DynamicTanh
        Output is guaranteed in (-1, 1) — lossless INT8 quantisation.

    sandwich_norm=False (ablation):
        Linear -> LayerNorm -> Linear
        Original architecture, no output bounds.
    """
    def __init__(self, config: EBJEPAConfig = EBJEPAConfig()) -> None:
        super().__init__()
        c = config
        self.config = config

        self.compressor = nn.Sequential(
            nn.Linear(c.latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, c.compressed_dim),
        )
        self.causal_bridge = nn.Linear(c.compressed_dim + c.aux_dim, c.hidden_dim)

        if config.sandwich_norm:
            self.predictor = nn.Sequential(
                nn.LayerNorm(c.hidden_dim),          # pre-norm
                nn.Linear(c.hidden_dim, c.latent_dim),
                RMSNorm(c.latent_dim),               # inner post-norm
                nn.Linear(c.latent_dim, c.compressed_dim),
                RMSNorm(c.compressed_dim),           # outer post-norm: bounds spikes
                DynamicTanh(c.compressed_dim),       # learnable range gate -> (-1,1)
            )
            logger.info("CJEPAPredictor: Sandwich Norm enabled (INT8/NPU-ready)")
        else:
            self.predictor = nn.Sequential(
                nn.Linear(c.hidden_dim, c.latent_dim),
                nn.LayerNorm(c.latent_dim),
                nn.Linear(c.latent_dim, c.compressed_dim),
            )
            logger.info("CJEPAPredictor: Standard norm (sandwich_norm=False)")

    def forward(self, z_entities: torch.Tensor,
                u_action: torch.Tensor,
                proprioception: torch.Tensor) -> torch.Tensor:
        z_c  = self.compressor(z_entities)
        prop = proprioception.unsqueeze(1).expand(-1, z_c.size(1), -1)
        h    = torch.relu(self.causal_bridge(torch.cat([z_c, prop], dim=-1)))
        return self.predictor(h)


# ---------------------------------------------------------------------------
# EBJEPAPlanner
# ---------------------------------------------------------------------------

class EBJEPAPlanner:
    """Best-of-K mirror ascent planner using cumulative EB-JEPA energy."""

    def __init__(self, predictor: CJEPAPredictor, config: EBJEPAConfig,
                 action_dim: int = 2, action_scale: float = 0.2,
                 eta: float = 0.05) -> None:
        self.predictor    = predictor
        self.config       = config
        self.action_dim   = action_dim
        self.action_scale = action_scale
        self.eta          = eta
        self._last_action = np.zeros(action_dim, dtype=np.float32)
        self.predictor.eval()

    def plan(self, z_current: np.ndarray, z_goal: np.ndarray,
             proprioception: torch.Tensor, rho: float = 0.0) -> np.ndarray:
        K   = self.config.num_candidates
        H   = self.config.planning_horizon
        rng = np.random.default_rng()
        candidates = rng.normal(0, self.action_scale, (K, H, self.action_dim)).astype(np.float32)

        costs = np.array([
            self._traj_cost(z_current, candidates[k], z_goal, proprioception)
            for k in range(K)
        ], dtype=np.float32)

        gradient = -costs / (1.0 + rho)
        weights  = np.exp(gradient / (self.eta + 1e-8))
        weights /= weights.sum() + 1e-12

        optimal = (weights[:, None] * candidates[:, 0, :]).sum(axis=0)
        self._last_action = optimal
        return optimal

    def _traj_cost(self, z0: np.ndarray, actions: np.ndarray,
                   z_goal: np.ndarray, prop: torch.Tensor) -> float:
        z   = torch.tensor(z0,     dtype=torch.float32).view(1, 1, -1)
        z_g = torch.tensor(z_goal, dtype=torch.float32).view(1, 1, -1)
        total = 0.0
        with torch.no_grad():
            z_g_c = self.predictor.compressor(z_g)
            for h in range(len(actions)):
                z_next = self.predictor(z, torch.zeros(1, self.action_dim), prop)
                total += float(torch.norm(z_g_c - z_next))
                z = torch.nn.functional.pad(z_next, (0, z.size(-1) - z_next.size(-1)))
        return total
