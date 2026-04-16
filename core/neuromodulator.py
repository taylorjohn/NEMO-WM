"""
neuromodulator.py  —  CORTEX-PE v16.11
========================================
Eight-signal biologically-grounded neuromodulator system.

ORIGINAL (v16.10)          EXTENDED (v16.11)
DA   dopamine              ACh  acetylcholine
5HT  serotonin             E/I  excitation/inhibition ratio
NE   norepinephrine (rho)  Ado  adenosine
                           eCB  endocannabinoid

All signals in [0,1] except E/I which is in [0.5, 2.0].

Quick start:
    import time
    from neuromodulator import NeuromodulatorState, ModulatedPlanner, neuro_to_packet

    neuro = NeuromodulatorState(session_start=time.time())
    signals = neuro.update(z_pred, z_actual, rho=rho,
                           action_magnitude=np.linalg.norm(last_action))
    packet.update(neuro_to_packet(signals))
"""

import math
import time
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F



# ═══════════════════════════════════════════════════════════════════════════
# CortisolSignal — eighth neuromodulatory signal (validated r=0.768 lag-1)
# ═══════════════════════════════════════════════════════════════════════════

class CortisolSignal:
    """
    Slow-timescale stress signal tracking sustained loss elevation above baseline.

    Biologically: cortisol rises during sustained stress and drives
    behavioural adaptation — increased vigilance, suppression of habitual
    responses, re-anchoring to ground truth.

    In NeMo-WM: detected the Sprint 3 distribution shift (epoch 19→20 loss
    rise from 0.5674→0.5696) one epoch ahead (r=0.768 lag-1, p<0.0001).

    Effect:
        High cortisol → amplify NE (spatial re-anchoring via GPS loss)
                      → suppress eCB (break habitual suppression)
        Low cortisol  → no modulation

    Usage:
        cortisol = CortisolSignal(baseline=0.567)
        cort = cortisol.update(current_loss)
        signals = cortisol.modulate(signals, cort)
    """

    def __init__(self, window: int = 100, baseline: float = None,
                 decay: float = 0.99, scale: float = 10.0):
        """
        Args:
            window:   rolling window size for sustained stress detection
            baseline: expected loss at convergence (None = use EMA)
            decay:    EMA decay for loss baseline tracking
            scale:    amplification factor for cortisol signal
        """
        self.window   = window
        self.baseline = baseline
        self.decay    = decay
        self.scale    = scale
        self._loss_ema: Optional[float] = None
        self._history: deque = deque(maxlen=window)

    def update(self, loss: float) -> float:
        """
        Update cortisol from current loss value.

        Returns cortisol in [0, 1]. High when loss has been
        consistently elevated above baseline for many steps.
        """
        self._history.append(loss)

        # Initialise EMA baseline on first call
        if self._loss_ema is None:
            self._loss_ema = loss

        # Update EMA baseline
        self._loss_ema = self.decay * self._loss_ema + (1 - self.decay) * loss

        # Reference: explicit baseline if given, else EMA
        ref = self.baseline if self.baseline is not None else self._loss_ema

        # Cortisol = sustained excess above reference
        if len(self._history) >= self.window // 2:
            recent_mean = float(np.mean(list(self._history)))
        else:
            recent_mean = loss

        cortisol = max(0.0, (recent_mean - ref) / (abs(ref) + 1e-8))
        return min(1.0, cortisol * self.scale)

    def modulate(self, signals: dict, cortisol: float,
                 ne_boost: float = 0.3, ecb_suppress: float = 0.3) -> dict:
        """
        Apply cortisol modulation to neuromodulator signals.

        High cortisol:
            NE  += ne_boost * cortisol   (spatial re-anchoring)
            eCB -= ecb_suppress * cortisol (break habit suppression)

        Args:
            signals:      dict from NeuromodulatorState.update()
            cortisol:     cortisol value in [0, 1]
            ne_boost:     NE amplification per unit cortisol
            ecb_suppress: eCB suppression per unit cortisol

        Returns modified signals dict.
        """
        if cortisol < 0.01:
            return signals  # no modulation below threshold

        signals = dict(signals)  # don't mutate original
        signals['rho'] = min(1.0, signals.get('rho', 0.0) + ne_boost * cortisol)
        signals['ecb'] = max(0.0, signals.get('ecb', 0.0) - ecb_suppress * cortisol)
        signals['cortisol'] = round(cortisol, 4)
        return signals

    def reset(self):
        """Reset cortisol state (call on session reset)."""
        self._loss_ema = None
        self._history.clear()


# ═══════════════════════════════════════════════════════════════════════════
# Regime
# ═══════════════════════════════════════════════════════════════════════════

class Regime:
    EXPLOIT   = "EXPLOIT"
    EXPLORE   = "EXPLORE"
    WAIT      = "WAIT"
    REOBSERVE = "REOBSERVE"


def classify_regime(da: float, sht: float,
                    da_thresh: float = 0.5,
                    sht_thresh: float = 0.5) -> str:
    if da >= da_thresh and sht >= sht_thresh:  return Regime.EXPLORE
    if da >= da_thresh and sht <  sht_thresh:  return Regime.WAIT
    if da <  da_thresh and sht <  sht_thresh:  return Regime.REOBSERVE
    return Regime.EXPLOIT


# ═══════════════════════════════════════════════════════════════════════════
# NeuromodulatorState
# ═══════════════════════════════════════════════════════════════════════════

class NeuromodulatorState:
    """
    Eight-signal neuromodulator for CORTEX-PE (v16.12 adds Cortisol).

    Signal summary
    --------------
    DA  (dopamine, [0,1])
        Prediction error: (1 - cos_sim(z_pred, z_actual)) / 2.
        High when world deviates from model. Drives exploration and
        MeZO perturbation width.

    5HT (serotonin, [0,1])
        Latent stability: exp(-10 * std(z_history)).
        High when encoder output is consistent. Gates action scale.

    NE  (norepinephrine, [0,1])
        Global arousal from Allen Neuropixels spike rate.
        Passed in externally as `rho`.

    ACh (acetylcholine, [0,1])
        Attention gate. Rises when input is surprising AND unstable.
        Dynamically adjusts EMA decay rates: high ACh -> fast adaptation.
        Also scales online learning rate for LoRA adapters.

    E/I (excit/inhib ratio, [0.5, 2.0])
        DA / (1 - 5HT + eps). Free: derived from existing signals.
        Multiplies MeZO action std: high E/I -> broad exploration.

    Ado (adenosine, [0,1])
        Monotone fatigue from session elapsed time.
        Saturates at ado_saturate_hours. Reduces candidates and action
        scale late in session. Cleared by reset(full=True).

    eCB (endocannabinoid, [0,1])
        Retrograde dampening. Produced when large action taken in
        high-DA state. Suppresses effective DA by up to 40%, preventing
        EXPLORE oscillation on repeated novel inputs.

    Parameters
    ----------
    da_decay, sht_decay : EMA decay constants (higher = slower).
    history_len : rolling window for 5HT and ACh.
    da_thresh, sht_thresh : regime classification thresholds.
    session_start : unix timestamp of session start.
    ado_saturate_hours : hours until Ado = 1.0.
    """

    def __init__(
        self,
        da_decay:           float = 0.95,
        sht_decay:          float = 0.90,
        history_len:        int   = 8,
        da_thresh:          float = 0.5,
        sht_thresh:         float = 0.5,
        session_start:      Optional[float] = None,
        ado_saturate_hours: float = 4.0,
    ):
        self.da_decay           = da_decay
        self.sht_decay          = sht_decay
        self.da_thresh          = da_thresh
        self.sht_thresh         = sht_thresh
        self._session_start     = session_start or time.time()
        self._ado_sat           = ado_saturate_hours * 3600.0

        # ── Signal state ───────────────────────────────────────────────────
        self.da  = 0.5   # dopamine EMA
        self.sht = 0.5   # serotonin EMA
        self.rho = 0.0   # norepinephrine (external)
        self.ach = 0.5   # acetylcholine EMA
        self.ei  = 1.0   # E/I ratio (derived)
        self.ado = 0.0   # adenosine (monotone)
        self.ecb = 0.0   # endocannabinoid EMA

        # ── Buffers ────────────────────────────────────────────────────────
        self._z_history:  deque = deque(maxlen=history_len)
        self._log:        deque = deque(maxlen=1000)

        # ── Cortisol (eighth signal) ───────────────────────────────────────
        self.cortisol_signal = CortisolSignal()
        self.cortisol = 0.0

    # ──────────────────────────────────────────────────────────────────────
    # update
    # ──────────────────────────────────────────────────────────────────────

    def update(
        self,
        z_pred:           torch.Tensor,
        z_actual:         torch.Tensor,
        rho:              float = 0.0,
        action_magnitude: float = 0.0,
    ) -> dict:
        """
        Update all six signals from one latent transition.

        Parameters
        ----------
        z_pred : (D,) or (1,D) tensor — predictor's anticipated next latent.
        z_actual : (D,) or (1,D) tensor — encoder's observed next latent.
        rho : norepinephrine scalar from Allen Neuropixels.
        action_magnitude : L2 norm of last executed action (for eCB).

        Returns dict with all signals and derived planner parameters.
        """
        z_pred   = z_pred.detach().flatten()
        z_actual = z_actual.detach().flatten()

        # 1. Dopamine ─────────────────────────────────────────────────────
        cs        = float(F.cosine_similarity(z_pred.unsqueeze(0),
                                              z_actual.unsqueeze(0)))
        cs        = max(-1.0, min(1.0, cs))
        da_phasic = (1.0 - cs) / 2.0
        self.da   = self.da_decay * self.da + (1.0 - self.da_decay) * da_phasic

        # 2. Serotonin ────────────────────────────────────────────────────
        self._z_history.append(z_actual.cpu())
        if len(self._z_history) >= 4:
            _hist = list(self._z_history)
            _s    = _hist[-1].shape
            _hist = [z for z in _hist if z.shape == _s]
            zs    = torch.stack(_hist)
            stability = math.exp(-10.0 * zs.std(dim=0).mean().item())
            stability = max(0.0, min(1.0, stability))
        else:
            stability = 0.5
        self.sht = self.sht_decay * self.sht + (1.0 - self.sht_decay) * stability

        # 3. Norepinephrine (external) ─────────────────────────────────────
        self.rho = float(rho)

        # 4. Acetylcholine ────────────────────────────────────────────────
        # Rises when input is both surprising (high DA) and unstable (low 5HT).
        # Signals "attend to new input over stored model".
        # Modulates EMA decay rates and online learning gate.
        ach_raw  = (da_phasic + (1.0 - stability)) / 2.0
        self.ach = 0.85 * self.ach + 0.15 * ach_raw

        # 5. E/I ratio (derived — free) ────────────────────────────────────
        # E/I = DA / (1 - 5HT + eps). Scales action gaussian width.
        # High E/I -> broad exploration; Low E/I -> focused exploitation.
        self.ei = max(0.5, min(2.0, self.da / (1.0 - self.sht + 0.1)))

        # 6. Adenosine ─────────────────────────────────────────────────────
        # Monotone session fatigue. Saturates at ado_saturate_hours.
        # Reduces candidates and action scale late in long sessions.
        self.ado = min(1.0, (time.time() - self._session_start) / self._ado_sat)

        # 7. Endocannabinoid ───────────────────────────────────────────────
        # Retrograde: produced when large action taken in high-DA state.
        # Suppresses effective DA, breaking EXPLORE oscillation loops.
        ecb_raw  = da_phasic * min(1.0, float(action_magnitude))
        self.ecb = 0.85 * self.ecb + 0.15 * ecb_raw

        # 8. Cortisol (eighth signal — slow timescale stress) ─────────────
        # Updated from loss proxy: cos_distance serves as per-step loss signal
        loss_proxy = da_phasic  # prediction error as loss proxy
        self.cortisol = self.cortisol_signal.update(loss_proxy)

        # Effective DA: eCB retrograde suppression (up to -40%)
        da_eff = self.da * (1.0 - self.ecb * 0.4)

        # Derive outputs ───────────────────────────────────────────────────
        regime  = classify_regime(da_eff, self.sht,
                                  self.da_thresh, self.sht_thresh)
        entry = {
            "da":           round(self.da,  4),
            "sht":          round(self.sht, 4),
            "rho":          round(self.rho, 4),
            "ach":          round(self.ach, 4),
            "ei":           round(self.ei,  4),
            "ado":          round(self.ado, 4),
            "ecb":          round(self.ecb, 4),
            "da_effective": round(da_eff,   4),
            "regime":       regime,
            "confidence":   self._confidence(da_eff),
            "action_scale": round(self._action_scale(regime), 3),
            "eps_scale":    round(self._eps_scale(da_eff),    3),
            "action_std":   round(self._action_std(),         4),
            "n_candidates": self._n_candidates(da_eff),
            "lr_scale":     round(self._lr_scale(),           3),
            "da_phasic":    round(da_phasic,     4),
            "cos_sim":      round(cs,            4),
            "cortisol":     round(self.cortisol, 4),
        }
        self._log.append(entry)
        return entry

    # ── Derived planner parameters ─────────────────────────────────────────

    def _action_scale(self, regime: str) -> float:
        base = {Regime.WAIT: 0.0, Regime.REOBSERVE: 0.4,
                Regime.EXPLORE: 1.0, Regime.EXPLOIT: 1.0}.get(regime, 1.0)
        return base * (1.0 - self.ado * 0.15)   # adenosine conservatism

    def _eps_scale(self, da_eff: float) -> float:
        return 1.0 + da_eff * 0.8 + self.rho * 0.3

    def _action_std(self) -> float:
        """MeZO gaussian std, E/I modulated. Range [0.05, 0.20]."""
        return max(0.05, min(0.20, 0.1 * self.ei))

    def _n_candidates(self, da_eff: float) -> int:
        """Adaptive candidates [16, 96]. Adenosine reduces late-session."""
        base = int(64 * (0.5 + da_eff))
        return max(16, min(96, int(base * (1.0 - self.ado * 0.5))))

    def _lr_scale(self) -> float:
        """Online LR scale for LoRA adapters. ACh-gated. Range [0.2, 2.0]."""
        return max(0.2, min(2.0, 0.5 + self.ach * 1.5))

    def _confidence(self, da_eff: float) -> str:
        score = self.sht * (1.0 - da_eff * 0.5)
        if score > 0.65: return "HIGH"
        if score > 0.35: return "MEDIUM"
        return "LOW"

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def should_act(self) -> bool:
        da_eff = self.da * (1.0 - self.ecb * 0.4)
        return classify_regime(da_eff, self.sht,
                               self.da_thresh, self.sht_thresh) != Regime.WAIT

    @property
    def is_novel(self) -> bool:
        return self.da >= self.da_thresh

    @property
    def is_stable(self) -> bool:
        return self.sht >= self.sht_thresh

    @property
    def is_fatigued(self) -> bool:
        return self.ado > 0.7

    @property
    def is_oscillating(self) -> bool:
        return self.ecb > 0.5

    def recent_log(self, n: int = 10) -> list:
        return list(self._log)[-n:]

    def reset(self, full: bool = True):
        """
        Reset signals. full=True clears adenosine (sleep); full=False preserves it.
        """
        self.da = self.sht = 0.5
        self.rho = self.ecb = 0.0
        self.ach = 0.5
        self.ei  = 1.0
        self._z_history.clear()
        self._log.clear()
        if full:
            self.ado = 0.0
            self._session_start = time.time()
            self.cortisol_signal.reset()
            self.cortisol = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# ModulatedPlanner
# ═══════════════════════════════════════════════════════════════════════════

class ModulatedPlanner:
    """
    Drop-in wrapper for BoKMirrorAscentPlanner with six-signal neuromodulation.

    Usage:
        planner = ModulatedPlanner(base_planner, neuro)
        action  = planner.optimize_saccade(
            z, z_pred=z_pred, z_actual=z_actual, rho=rho)
        packet.update(planner.last_neuro_signals)
    """

    def __init__(self, base_planner, neuro: NeuromodulatorState):
        self.planner             = base_planner
        self.neuro               = neuro
        self.last_neuro_signals: dict  = {}
        self._last_action_mag:   float = 0.0

    def optimize_saccade(
        self,
        initial_latent,
        z_pred:         Optional[torch.Tensor] = None,
        z_actual:       Optional[torch.Tensor] = None,
        rho:            float = 0.0,
        k_steps:        int   = 5,
        num_candidates: int   = 64,
        gamma_horizon:  float = 2.0,
    ) -> np.ndarray:
        if z_pred is not None and z_actual is not None:
            sig = self.neuro.update(z_pred, z_actual, rho=rho,
                                    action_magnitude=self._last_action_mag)
            self.last_neuro_signals = neuro_to_packet(sig)
        else:
            sig = {"regime": Regime.EXPLOIT, "action_scale": 1.0,
                   "n_candidates": num_candidates, "action_std": 0.1}

        if not self.neuro.should_act:
            return np.zeros(2)

        n_cands    = sig.get("n_candidates", num_candidates)
        action_std = sig.get("action_std", 0.1)

        il = torch.tensor(initial_latent, device=self.planner.device,
                          dtype=torch.float32).unsqueeze(0)
        cands = torch.randn(n_cands, k_steps, 2,
                            device=self.planner.device) * action_std

        best_cost, best_seq = float("inf"), None
        with torch.no_grad():
            for i in range(n_cands):
                cost = self.planner.predict_trajectory_cost(il, cands[i], gamma_horizon)
                if cost < best_cost:
                    best_cost, best_seq = cost, cands[i]

        action = best_seq[0].cpu().numpy() * sig.get("action_scale", 1.0)
        self._last_action_mag = float(np.linalg.norm(action))
        return action


# ═══════════════════════════════════════════════════════════════════════════
# Cockpit telemetry
# ═══════════════════════════════════════════════════════════════════════════

def neuro_to_packet(s: dict) -> dict:
    """Format all signals for UDP broadcast to Mac Glass Cockpit."""
    return {
        "DA":        s.get("da",           0.0),
        "5HT":       s.get("sht",          0.0),
        "NE":        s.get("rho",          0.0),
        "ACH":       s.get("ach",          0.0),
        "EI":        s.get("ei",           1.0),
        "ADO":       s.get("ado",          0.0),
        "ECB":       s.get("ecb",          0.0),
        "DA_EFF":    s.get("da_effective", s.get("da", 0.0)),
        "REGIME":    s.get("regime",       "EXPLOIT"),
        "CONF":      s.get("confidence",   "MEDIUM"),
        "ACT_SCALE": s.get("action_scale", 1.0),
        "EPS_SCALE": s.get("eps_scale",    1.0),
        "ACT_STD":   s.get("action_std",   0.1),
        "N_CAND":    s.get("n_candidates", 64),
        "LR_SCALE":  s.get("lr_scale",     1.0),
        "CORTISOL":  s.get("cortisol",    0.0),
    }
