"""
cortex_brain.perception.gaze_controller
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Begin-of-Glance (BOG) Token + Conditional Attention Gate.

Problem: Saccade Hallucination
--------------------------------
When the fovea makes a rapid saccade (eye movement to a new location),
the predictor receives a frame where the visual content has completely
changed.  Without a reset signal, the causal bridge carries stale hidden
state from the previous fixation, producing "hallucinated" predictions
that blend the old and new scenes.

BOG Token Solution
------------------
A learnable <BOG> token is injected as entity-0 at the moment of saccade
detection.  This serves as a deliberate attention sink:

    - The causal bridge's attention mechanism routes its "reset" energy
      to entity-0 (the BOG token) rather than contaminating real entities.
    - The BOG token absorbs the mathematical noise of the scene transition.
    - After one tick, the BOG token is retired and real prediction resumes
      with a clean prior.

This mirrors the role of the [CLS] token in BERT and the [BOS] token in
GPT — a position that absorbs broad contextual information so other
positions can focus on specific features.

Conditional Attention Gate
--------------------------
If the foveal crop is low-arousal (staring at a blank surface, uniform
texture, or static background), all attention computation is wasted.
The ConditionalAttentionGate checks the resonance signal ρ against a
threshold.  Below threshold, it returns a zero gate (g=0), shutting down
attention heads entirely and saving NPU cycles.

Biological analogy: microsaccade suppression.  When the visual cortex
detects that a fixation target is uninteresting, thalamic gating reduces
cortical gain before attention can be allocated.

Usage
-----
    gc = GazeController(GazeConfig())

    # Each tick, pass fovea position and arousal:
    bog_entities, saccade_flag = gc.step(
        z_entities  = z_compressed,   # (B, N, D)
        fovea_xy    = (0.3, 0.7),
        resonance   = rho,
    )
    # Use bog_entities instead of z_compressed in the predictor call
"""
from __future__ import annotations
import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class GazeConfig:
    latent_dim:             int   = 16     # compressed entity dim
    num_entities:           int   = 8      # N in (B, N, D)

    # Saccade detection
    saccade_velocity_thresh: float = 2.0  # normalised fovea units/sec
    velocity_window:         int   = 3    # ticks for velocity smoothing

    # BOG token behaviour
    bog_decay_ticks:         int   = 2    # how many ticks BOG token persists
    bog_position:            int   = 0    # which entity slot to inject BOG into

    # Conditional gate
    gate_rho_threshold:      float = 0.05  # below this ρ → shut down attention
    gate_blend_ticks:        int   = 3     # smooth gate transitions (avoid clicks)


class SaccadeDetector:
    """
    Detects rapid foveal shifts (saccades) from the fovea_xy trajectory.

    Uses a sliding window velocity estimate:
        v(t) = ||pos(t) - pos(t-W)|| / W

    Parameters
    ----------
    config : GazeConfig
    """

    def __init__(self, config: GazeConfig) -> None:
        self.config   = config
        self._history: deque = deque(maxlen=config.velocity_window + 1)
        self._last_velocity: float = 0.0

    def step(self, fovea_xy: Tuple[float, float]) -> Tuple[bool, float]:
        """
        Parameters
        ----------
        fovea_xy : (x, y) normalised to [0, 1]

        Returns
        -------
        is_saccade : bool   — True if velocity exceeded threshold
        velocity   : float  — smoothed fovea speed (units/tick)
        """
        pos = np.array(fovea_xy, dtype=np.float32)
        self._history.append(pos)

        if len(self._history) < 2:
            return False, 0.0

        # Velocity over the available window
        oldest = self._history[0]
        newest = self._history[-1]
        W      = len(self._history) - 1
        vel    = float(np.linalg.norm(newest - oldest)) / max(W, 1)

        # Exponential smoothing
        alpha = 0.4
        self._last_velocity = alpha * vel + (1.0 - alpha) * self._last_velocity

        is_saccade = self._last_velocity > self.config.saccade_velocity_thresh
        return is_saccade, self._last_velocity

    def reset(self) -> None:
        self._history.clear()
        self._last_velocity = 0.0


class ConditionalAttentionGate(nn.Module):
    """
    Learnable gate that shuts down attention when resonance ρ is very low.

    Gate value g ∈ [0, 1]:
        g = 1  →  full attention (normal operation)
        g = 0  →  attention zeroed (blank-surface suppression)

    The gate is soft — it fades smoothly via an EMA rather than hard-
    switching, preventing audio-like clicks in downstream activations.

    Parameters
    ----------
    config   : GazeConfig
    """

    def __init__(self, config: GazeConfig) -> None:
        super().__init__()
        self.config    = config
        self._gate_ema = 1.0   # starts fully open

    def forward(
        self,
        z_entities: torch.Tensor,   # (B, N, D)
        resonance:  float,
    ) -> Tuple[torch.Tensor, float]:
        """
        Returns
        -------
        z_gated  : (B, N, D)  — z_entities * gate (zero when low-arousal)
        gate_val : float      — current gate value [0, 1]
        """
        target_gate = 0.0 if resonance < self.config.gate_rho_threshold else 1.0
        alpha       = 1.0 / max(self.config.gate_blend_ticks, 1)
        self._gate_ema = (1.0 - alpha) * self._gate_ema + alpha * target_gate

        if self._gate_ema < 0.01:
            logger.debug("ConditionalGate: CLOSED (ρ=%.4f < %.4f)",
                         resonance, self.config.gate_rho_threshold)
            return torch.zeros_like(z_entities), 0.0

        return z_entities * self._gate_ema, self._gate_ema


class BOGController(nn.Module):
    """
    Injects a learnable Begin-of-Glance token on saccade events.

    The BOG token occupies entity slot `config.bog_position` (default: 0)
    for `config.bog_decay_ticks` ticks after a saccade is detected.
    During that window, real entities are shifted right by one position.

    After bog_decay_ticks, the token is retired and the original entity-0
    slot is restored.

    Parameters
    ----------
    config : GazeConfig
    """

    def __init__(self, config: GazeConfig) -> None:
        super().__init__()
        self.config = config
        # Learnable BOG embedding — initialised near zero so it's a
        # gentle perturbation rather than a hard reset
        self.bog_token = nn.Parameter(
            torch.zeros(config.latent_dim))
        self._active_ticks: int = 0

    def inject(
        self,
        z_entities:  torch.Tensor,   # (B, N, D)
        is_saccade:  bool,
    ) -> Tuple[torch.Tensor, bool]:
        """
        Parameters
        ----------
        z_entities  : current compressed entity latents  (B, N, D)
        is_saccade  : saccade detector output

        Returns
        -------
        z_out    : (B, N, D)  — modified entity latents
        bog_active : bool     — True while BOG token is injected
        """
        B, N, D = z_entities.shape

        if is_saccade:
            self._active_ticks = self.config.bog_decay_ticks
            logger.debug("BOGController: saccade detected → injecting BOG for %d ticks",
                         self._active_ticks)

        if self._active_ticks > 0:
            self._active_ticks -= 1
            z_out = z_entities.clone()
            pos   = self.config.bog_position
            # Replace entity at `pos` with the BOG token
            z_out[:, pos, :] = self.bog_token.unsqueeze(0).expand(B, -1)
            return z_out, True

        return z_entities, False

    def reset(self) -> None:
        """Call between episodes or after hard resets."""
        self._active_ticks = 0


class GazeController(nn.Module):
    """
    Unified gaze management: SaccadeDetector + BOGController + ConditionalGate.

    Call `step()` once per engine tick with the current fovea position
    and resonance.  Returns modified entity latents ready for the predictor.

    Usage
    -----
        gc = GazeController(GazeConfig(latent_dim=16, num_entities=8))

        z_out, meta = gc.step(z_entities, fovea_xy=(0.3, 0.7), resonance=0.42)
        # z_out: (B, N, D) — use this in predictor, not z_entities
        # meta: dict with saccade, bog_active, gate_val, velocity
    """

    def __init__(self, config: GazeConfig = GazeConfig()) -> None:
        super().__init__()
        self.config    = config
        self.saccade   = SaccadeDetector(config)
        self.bog       = BOGController(config)
        self.gate      = ConditionalAttentionGate(config)
        self._tick     = 0

    def step(
        self,
        z_entities:  torch.Tensor,            # (B, N, D)
        fovea_xy:    Tuple[float, float],
        resonance:   float,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Full gaze processing pipeline per tick.

        Pipeline
        --------
        1. SaccadeDetector  — is this a saccade tick?
        2. BOGController    — inject BOG token if saccade or BOG still active
        3. ConditionalGate  — zero out if low-arousal blank surface

        Returns
        -------
        z_out : (B, N, D)
        meta  : {saccade, bog_active, gate_val, velocity, tick}
        """
        self._tick += 1

        # Step 1: Saccade detection
        is_saccade, velocity = self.saccade.step(fovea_xy)

        # Step 2: BOG injection
        z_bog, bog_active = self.bog.inject(z_entities, is_saccade)

        # Step 3: Conditional gate
        z_out, gate_val = self.gate(z_bog, resonance)

        meta = dict(
            saccade    = is_saccade,
            bog_active = bog_active,
            gate_val   = round(gate_val, 4),
            velocity   = round(velocity, 4),
            tick       = self._tick,
        )

        if is_saccade:
            logger.info("GazeController tick=%d: SACCADE vel=%.3f gate=%.3f",
                        self._tick, velocity, gate_val)

        return z_out, meta

    def reset(self) -> None:
        """Hard reset — use between episodes."""
        self.saccade.reset()
        self.bog.reset()
        self.gate._gate_ema = 1.0
        self._tick = 0
