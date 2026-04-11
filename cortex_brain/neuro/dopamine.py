"""
cortex_brain.neuro.dopamine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Biologically-inspired neuromodulation for CORTEX-16.

Three analog signals modulate the engine each tick:

  Dopamine (DA)  — reward prediction error → sharpens exploitation
  Cortisol (CRT) — sustained stress        → degrades performance if unmanaged
  Norepinephrine (NE) — arousal gain       → SNR enhancement of LSM inputs

Theory
------
Biological dopamine signals *prediction error* (δ = actual − predicted reward),
not reward itself.  Positive δ reinforces the current manifold; negative δ
broadens exploration.  Cortisol accumulates under thermal stress and chronic
negative δ; its tonic elevation flattens the DA signal (stress → anhedonia).

Mappings to CORTEX-16
---------------------
  DA  → modulate_rho(ρ)          lower ρ on positive RPE → exploitation
  NE  → modulate_input_scaling() higher scaling under high DA → SNR boost
  CRT → stress penalty applied to DA, preventing manifold lock-in

Usage
-----
    dm = DopamineSystem()
    dm.update(pnl=+1.2, resonance=0.41, temp_norm=0.35)
    effective_rho    = dm.modulate_rho(resonance)
    effective_scale  = dm.modulate_input_scaling(base_scale=0.5)
    status           = dm.status()
"""
from __future__ import annotations
import logging
from collections import deque
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DopamineConfig:
    # Tonic baselines
    da_tonic:    float = 0.5    # resting dopamine [0, 1]
    crt_tonic:   float = 0.2    # resting cortisol [0, 1]

    # Learning rates
    da_lr:       float = 0.15   # how fast DA updates toward RPE
    crt_lr:      float = 0.08   # how fast cortisol rises under stress
    crt_decay:   float = 0.01   # how fast cortisol recovers per tick

    # RPE window for baseline expected reward
    rpe_window:  int   = 20     # ticks used to estimate expected reward

    # Modulation strengths
    rho_da_gain:    float = 0.4  # max ρ reduction from DA (0 = no effect, 1 = full)
    snr_da_gain:    float = 0.6  # max input_scaling boost from DA
    crt_da_penalty: float = 0.5  # cortisol dampening of DA signal

    # Thermal stress threshold (normalised temp above which cortisol rises)
    thermal_stress_threshold: float = 0.65  # ~69°C (30 + 0.65*60)


class DopamineSystem:
    """
    Tracks DA, CRT, and NE across ticks and exposes three modulation functions.

    Attributes (read-only)
    ----------------------
    da        : float  current dopamine level     [0, 1]
    cortisol  : float  current cortisol level     [0, 1]
    rpe       : float  last reward prediction error
    rpe_mean  : float  rolling mean expected reward
    """

    def __init__(self, config: DopamineConfig = DopamineConfig()) -> None:
        self.config   = config
        self.da       = config.da_tonic
        self.cortisol = config.crt_tonic
        self.rpe      = 0.0
        self._reward_history: deque = deque(maxlen=config.rpe_window)
        self._tick = 0

    # ------------------------------------------------------------------
    # Core update  (call once per engine tick, AFTER actuator.act())
    # ------------------------------------------------------------------

    def update(self, pnl: float, resonance: float, temp_norm: float) -> None:
        """
        Update all neuromodulators given the current tick's outcome.

        Parameters
        ----------
        pnl        : reward signal (trading PnL, robot task score, etc.)
        resonance  : reservoir resonance ρ from StaticCSRRouter
        temp_norm  : normalised CPU temperature from ProprioceptionPulse [0, 1]
        """
        self._tick += 1

        # ── Reward Prediction Error (RPE) ──────────────────────────────
        expected = float(np.mean(self._reward_history)) if self._reward_history else 0.0
        self._reward_history.append(pnl)
        self.rpe = pnl - expected                        # δ: +ve = better than expected

        # ── Dopamine update ────────────────────────────────────────────
        # Positive RPE → DA spike; negative RPE → DA dip
        # Cortisol dampens the signal (chronic stress → blunted response)
        crt_dampen = 1.0 - self.cortisol * self.config.crt_da_penalty
        da_target  = self._sigmoid(self.rpe) * crt_dampen
        self.da    = self._lerp(self.da, da_target, self.config.da_lr)
        self.da    = float(np.clip(self.da, 0.0, 1.0))

        # ── Cortisol update ────────────────────────────────────────────
        # Rises under thermal stress OR sustained negative RPE
        thermal_stress = max(0.0, temp_norm - self.config.thermal_stress_threshold)
        rpe_stress     = max(0.0, -self.rpe)             # chronic negative RPE
        stress_input   = thermal_stress + 0.5 * rpe_stress

        self.cortisol += self.config.crt_lr * stress_input
        self.cortisol -= self.config.crt_decay            # natural recovery each tick
        self.cortisol  = float(np.clip(self.cortisol, 0.0, 1.0))

        logger.debug("DA=%.3f CRT=%.3f RPE=%.4f ρ=%.3f temp=%.2f",
                     self.da, self.cortisol, self.rpe, resonance, temp_norm)

    # ------------------------------------------------------------------
    # Modulation functions  (call during planning, before LSM step)
    # ------------------------------------------------------------------

    def modulate_rho(self, rho: float) -> float:
        """
        Apply dopamine-based exploitation pressure to reservoir resonance.

        High DA (positive RPE) → lower effective ρ → sharper exploitation.
        High cortisol partially counteracts the reduction (stress → instability).

        Parameters
        ----------
        rho : raw resonance from StaticCSRRouter  [0, 1]

        Returns
        -------
        effective_rho : float  modulated resonance  [0, 1]
        """
        da_above_tonic = max(0.0, self.da - self.config.da_tonic)  # only spike, not dip
        reduction      = self.config.rho_da_gain * da_above_tonic
        crt_restore    = self.cortisol * 0.3 * reduction            # stress partially undoes it
        effective_rho  = rho - reduction + crt_restore
        return float(np.clip(effective_rho, 0.0, rho))              # never increase rho

    def modulate_input_scaling(self, base_scale: float = 0.5) -> float:
        """
        SNR enhancement: boost LSM input driving strength when DA is elevated.

        High DA → input_scaling increases → reservoir more responsive to signal.
        High cortisol caps the boost (thermal throttle limits gain).

        Parameters
        ----------
        base_scale : nominal LSM input_scaling from LSMConfig

        Returns
        -------
        effective_scale : float  modulated input_scaling
        """
        max_boost     = self.config.snr_da_gain * base_scale
        crt_ceiling   = 1.0 - self.cortisol * 0.5                  # cortisol caps gain
        boost         = max_boost * self.da * crt_ceiling
        return float(np.clip(base_scale + boost, base_scale, base_scale * (1 + self.config.snr_da_gain)))

    @property
    def rpe_mean(self) -> float:
        """Rolling mean of reward used as the prediction baseline."""
        return float(np.mean(self._reward_history)) if self._reward_history else 0.0

    def status(self) -> dict:
        """Snapshot of current neuromodulator state for logging/telemetry."""
        return dict(
            da       = round(self.da, 4),
            cortisol = round(self.cortisol, 4),
            rpe      = round(self.rpe, 4),
            rpe_mean = round(self.rpe_mean, 4),
            tick     = self._tick,
        )

    def reset(self) -> None:
        """Return to tonic baseline (use between episodes)."""
        self.da       = self.config.da_tonic
        self.cortisol = self.config.crt_tonic
        self.rpe      = 0.0
        self._reward_history.clear()
        self._tick    = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid(x: float, gain: float = 3.0) -> float:
        """Soft sigmoid mapping RPE → [0, 1].  gain controls sharpness."""
        return float(1.0 / (1.0 + np.exp(-gain * x)))

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        return a + t * (b - a)
