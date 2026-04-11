"""
cortex_brain.perception.lsm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Liquid State Machine (LSM) – reservoir computing for temporal perception.

Theory
------
A fixed randomly-connected reservoir whose high-dimensional transient dynamics
are read out by a thin trainable layer.  Reservoir weights are NEVER updated,
making it GIL-friendly and NPU-safe.

    r(t) = (1-α)·r(t-1) + α·tanh(W_res·r(t-1) + W_in·x(t))
    z(t) = W_out · r(t)

Spectral radius ρ(W_res) < 1 guarantees the echo-state property (fading memory).
Only W_out is trained, via a single ridge regression pass.

Domain inputs
-------------
Trading  : [price, spread, volume, change, volatility] per symbol (→ 256-D after projection)
Robotics : camera patch features + joint encoders + IMU readings
Vision   : CNN patch tokens from MarketFeatureEncoder / NPU encoder
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LSMConfig:
    input_dim:       int   = 256
    reservoir_dim:   int   = 512
    output_dim:      int   = 128
    spectral_radius: float = 0.9    # must be < 1.0
    input_scaling:   float = 0.5
    sparsity:        float = 0.10   # fraction of NON-zero weights
    leak_rate:       float = 0.3    # leaky-integrator α
    seed:            int   = 42


class LiquidStateMachine:
    """
    Fixed reservoir + trainable linear readout.

    Example
    -------
    >>> lsm = LiquidStateMachine(LSMConfig(input_dim=5, reservoir_dim=128, output_dim=64))
    >>> z = lsm.step(np.random.randn(5).astype(np.float32))   # (64,)
    >>> lsm.reset_state()
    """

    def __init__(self, config: LSMConfig = LSMConfig()) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        self._build_reservoir(rng)
        self._build_readout(rng)
        self._state = np.zeros(config.reservoir_dim, dtype=np.float32)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def step(self, x: np.ndarray) -> np.ndarray:
        """One time-step. x: (input_dim,) → z: (output_dim,)."""
        x = self._pad(x.astype(np.float32))
        pre = self._W_res @ self._state + self._W_in @ x
        new = np.tanh(pre)
        a = self.config.leak_rate
        self._state = (1.0 - a) * self._state + a * new
        return (self._W_out @ self._state).astype(np.float32)

    def step_sequence(self, xs: np.ndarray) -> np.ndarray:
        """xs: (T, input_dim) → zs: (T, output_dim)."""
        return np.stack([self.step(x) for x in xs])

    def reset_state(self) -> None:
        self._state = np.zeros(self.config.reservoir_dim, dtype=np.float32)

    def train_readout(self, X: np.ndarray, Y: np.ndarray, ridge: float = 1e-4) -> float:
        """Fit W_out via ridge regression on raw reservoir states. Only training step in an LSM."""
        self.reset_state()
        states = np.stack([self._raw_step(x.astype(np.float32)) for x in X])  # (T, reservoir_dim)
        reg = ridge * np.eye(self.config.reservoir_dim)
        self._W_out = np.linalg.solve(states.T @ states + reg, states.T @ Y).T
        mse = float(np.mean((states @ self._W_out.T - Y) ** 2))
        logger.info("LSM readout trained. MSE=%.6f", mse)
        return mse

    def _raw_step(self, x: np.ndarray) -> np.ndarray:
        """Step reservoir, return raw state (reservoir_dim,) without readout projection."""
        x = self._pad(x)
        pre = self._W_res @ self._state + self._W_in @ x
        new_s = np.tanh(pre)
        a = self.config.leak_rate
        self._state = (1.0 - a) * self._state + a * new_s
        return self._state.copy()

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build_reservoir(self, rng: np.random.Generator) -> None:
        c = self.config
        W = rng.standard_normal((c.reservoir_dim, c.reservoir_dim)).astype(np.float32)
        mask = rng.random((c.reservoir_dim, c.reservoir_dim)) > c.sparsity
        W[mask] = 0.0
        sr = float(np.max(np.abs(np.linalg.eigvals(W))))
        if sr > 1e-8:
            W *= c.spectral_radius / sr
        self._W_res = W
        self._W_in = (rng.standard_normal((c.reservoir_dim, c.input_dim)) * c.input_scaling).astype(np.float32)

    def _build_readout(self, rng: np.random.Generator) -> None:
        c = self.config
        W = rng.standard_normal((c.output_dim, c.reservoir_dim)).astype(np.float32)
        norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-8
        self._W_out = W / norms

    def _pad(self, x: np.ndarray) -> np.ndarray:
        n = self.config.input_dim
        x = x.flatten()[:n]
        if len(x) < n:
            x = np.pad(x, (0, n - len(x)))
        return x