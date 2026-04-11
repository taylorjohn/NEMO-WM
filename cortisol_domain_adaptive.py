"""
cortisol_domain_adaptive.py
NeMo-WM Sprint 9 — Cortisol Baseline Reset Protocol

Implements three changes over the v16.12 cortisol signal:
  1. Sensitivity increase: 0.05 → 0.10 (faster response to distribution shifts)
  2. Baseline reset on domain-entry detection (GPS bbox or visual cluster shift)
  3. NE/eCB amplification tuning: NE 1.14→1.22, eCB 0.89→0.82

Drop-in replacement for the CortisolSignal class in train_cwm_multidomain.py.
Compatible with existing checkpoint format — no retraining required.

Usage:
    from cortisol_domain_adaptive import CortisolSignalAdaptive
    cort = CortisolSignalAdaptive(sensitivity=0.10)

    # In training loop:
    cort_val = cort.step(loss=loss.item(), gps=gps_batch, z_enc=z_enc)
    ne_scale  = cort.ne_scale()     # NE amplification
    ecb_scale = cort.ecb_scale()    # eCB suppression

    # On domain boundary (optional, detected automatically):
    cort.reset_baseline(reason="tworoom_entry")
"""

import math
import time
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger("cortisol")


# ── Configuration ───────────────────────────────────────────────────────────

@dataclass
class CortisolConfig:
    # Core signal
    sensitivity:     float = 0.10      # Sprint 9: was 0.05 in v16.12
    window:          int   = 20        # baseline rolling window (batches)
    high_threshold:  float = 0.25      # CORT > this → high-cortisol regime
    decay:           float = 0.92      # per-step exponential decay

    # NE/eCB scaling (Sprint 9 tuning)
    ne_scale_high:   float = 1.22      # was 1.14 in v16.12
    ne_scale_low:    float = 1.00
    ecb_scale_high:  float = 0.82      # was 0.89 in v16.12
    ecb_scale_low:   float = 1.00

    # Domain detection
    gps_bbox_thresh: float = 50.0      # metres — bbox expansion > this → new domain
    visual_cos_thresh: float = 0.65    # cosine sim drop below this → visual domain shift
    visual_window:   int   = 10        # frames to compute running visual mean

    # Logging
    log_transitions: bool  = True
    log_dir:         str   = "outputs/cortisol_logs"


# ── Domain Detector ─────────────────────────────────────────────────────────

class DomainDetector:
    """
    Detects domain boundaries from two independent signals:
      (A) GPS bounding box expansion — robot enters physically new territory
      (B) Visual embedding cosine drift — visual distribution shift

    Either signal alone triggers a domain-entry event.
    """

    def __init__(self, cfg: CortisolConfig):
        self.cfg = cfg
        # GPS bbox state
        self._gps_min = np.array([+1e9, +1e9])
        self._gps_max = np.array([-1e9, -1e9])
        self._gps_bbox_area = 0.0

        # Visual running mean (unit-normalised)
        self._z_mean: Optional[torch.Tensor] = None
        self._z_buf:  deque = deque(maxlen=cfg.visual_window)

        # State
        self._step         = 0
        self._last_event   = -9999
        self._cooldown     = 50    # batches between detections

    def update(
        self,
        gps: Optional[torch.Tensor] = None,   # (B, 2) north/east metres
        z_enc: Optional[torch.Tensor] = None,  # (B, D) unit-normalised embeddings
    ) -> tuple[bool, str]:
        """
        Returns (domain_changed, reason_string).
        """
        self._step += 1
        if self._step - self._last_event < self._cooldown:
            return False, ""

        changed, reason = False, ""

        # (A) GPS bbox
        if gps is not None:
            pts = gps.detach().cpu().numpy().reshape(-1, 2)
            batch_min = pts.min(axis=0)
            batch_max = pts.max(axis=0)
            new_min = np.minimum(self._gps_min, batch_min)
            new_max = np.maximum(self._gps_max, batch_max)
            new_area = float(np.prod(new_max - new_min + 1e-6))
            expansion = new_area - self._gps_bbox_area
            if self._gps_bbox_area > 0 and expansion > self.cfg.gps_bbox_thresh:
                changed, reason = True, f"gps_bbox_expansion={expansion:.1f}m"
            self._gps_min, self._gps_max = new_min, new_max
            self._gps_bbox_area = new_area

        # (B) Visual cosine drift
        if z_enc is not None and not changed:
            z_mean_batch = F.normalize(
                z_enc.detach().float().mean(0, keepdim=True), dim=-1
            )
            self._z_buf.append(z_mean_batch)
            if len(self._z_buf) == self._z_buf.maxlen:
                running_mean = F.normalize(
                    torch.cat(list(self._z_buf)).mean(0, keepdim=True), dim=-1
                )
                if self._z_mean is not None:
                    sim = F.cosine_similarity(running_mean, self._z_mean).item()
                    if sim < self.cfg.visual_cos_thresh:
                        changed = True
                        reason = f"visual_cos_drop={sim:.3f}"
                self._z_mean = running_mean

        if changed:
            self._last_event = self._step

        return changed, reason


# ── Cortisol Signal ──────────────────────────────────────────────────────────

class CortisolSignalAdaptive:
    """
    Adaptive cortisol signal with domain-aware baseline reset.

    CORT_t = clip(sensitivity * (loss_t - baseline_t), 0, 1)

    baseline_t = rolling mean of recent losses (window=20 batches).
    On domain-entry detection, baseline_t is reset to current loss,
    preventing stale training-domain history from suppressing the cortisol
    response in new environments.

    Sprint 9 changes vs v16.12:
      - sensitivity: 0.05 → 0.10
      - NE scale high: 1.14 → 1.22
      - eCB scale high: 0.89 → 0.82
      - baseline_t reset on domain detection
    """

    def __init__(self, cfg: Optional[CortisolConfig] = None, log_path: Optional[str] = None):
        self.cfg = cfg or CortisolConfig()
        self._loss_buf: deque = deque(maxlen=self.cfg.window)
        self._cort: float = 0.0
        self._step: int = 0
        self._resets: list = []
        self._history: list = []     # (step, loss, baseline, cort)
        self.detector = DomainDetector(self.cfg)

        # Logging
        if self.cfg.log_transitions:
            Path(self.cfg.log_dir).mkdir(parents=True, exist_ok=True)
        self._log_path = log_path or f"{self.cfg.log_dir}/cort_{int(time.time())}.jsonl"

    # ── Core step ────────────────────────────────────────────────────────────

    def step(
        self,
        loss: float,
        gps: Optional[torch.Tensor] = None,
        z_enc: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Call once per training batch.

        Args:
            loss:  scalar loss value for this batch
            gps:   (B, 2) GPS north/east metres — triggers bbox detection
            z_enc: (B, D) visual embeddings — triggers visual domain detection

        Returns:
            cortisol value in [0, 1]
        """
        self._step += 1

        # Domain detection → baseline reset
        domain_changed, reason = self.detector.update(gps=gps, z_enc=z_enc)
        if domain_changed:
            self.reset_baseline(reason=reason)

        # Baseline from rolling window
        self._loss_buf.append(loss)
        baseline = float(np.mean(self._loss_buf)) if len(self._loss_buf) > 0 else loss

        # CORT_t = sensitivity × (loss_t − baseline_t), clipped to [0, 1]
        raw = self.cfg.sensitivity * (loss - baseline)
        self._cort = float(np.clip(raw, 0.0, 1.0))

        # Exponential decay toward zero in stable periods
        # (only applied after baseline window is full)
        if len(self._loss_buf) >= self.cfg.window:
            self._cort *= self.cfg.decay

        # Log
        record = {
            "step": self._step, "loss": round(loss, 5),
            "baseline": round(baseline, 5), "cort": round(self._cort, 5),
            "high": self._cort > self.cfg.high_threshold,
        }
        self._history.append(record)
        if self.cfg.log_transitions and self._step % 100 == 0:
            self._flush_log()

        return self._cort

    # ── Baseline reset ────────────────────────────────────────────────────────

    def reset_baseline(self, reason: str = "manual") -> None:
        """
        Reset rolling loss buffer to current value.
        Use on domain-entry to prevent stale training history from
        suppressing the cortisol response in new environments.
        """
        current_mean = (
            float(np.mean(self._loss_buf)) if self._loss_buf else 0.0
        )
        self._loss_buf.clear()
        # Seed with current mean so baseline doesn't cold-start at zero
        for _ in range(min(3, self.cfg.window)):
            self._loss_buf.append(current_mean)

        event = {
            "step": self._step, "reason": reason,
            "baseline_before": current_mean, "ts": time.time()
        }
        self._resets.append(event)

        if self.cfg.log_transitions:
            with open(self._log_path, "a") as f:
                f.write(json.dumps({"type": "reset", **event}) + "\n")
        logger.info(f"[Cortisol] Baseline reset at step {self._step}: {reason}")

    # ── Neuromodulator scaling ────────────────────────────────────────────────

    def ne_scale(self) -> float:
        """Norepinephrine amplification — ramps with cortisol level."""
        t = min(self._cort / self.cfg.high_threshold, 1.0)
        return self.cfg.ne_scale_low + t * (self.cfg.ne_scale_high - self.cfg.ne_scale_low)

    def ecb_scale(self) -> float:
        """Endocannabinoid suppression — ramps with cortisol level."""
        t = min(self._cort / self.cfg.high_threshold, 1.0)
        return self.cfg.ecb_scale_low + t * (self.cfg.ecb_scale_high - self.cfg.ecb_scale_low)

    def is_high(self) -> bool:
        return self._cort > self.cfg.high_threshold

    @property
    def value(self) -> float:
        return self._cort

    # ── Regime suggestion ─────────────────────────────────────────────────────

    def suggested_regime(self) -> str:
        """
        Regime suggestion based on cortisol level.
        Replaces fixed cosine-schedule regime assignment during Sprint 9.
        """
        if self._cort > self.cfg.high_threshold:
            return "REOBSERVE"
        elif self._cort > self.cfg.high_threshold * 0.5:
            return "EXPLORE"
        else:
            return "EXPLOIT"

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Call at end of epoch for logging."""
        if not self._history:
            return {}
        corts = [h["cort"] for h in self._history]
        highs = [h for h in self._history if h["high"]]
        return {
            "cort_mean":     round(float(np.mean(corts)), 4),
            "cort_max":      round(float(np.max(corts)), 4),
            "cort_high_pct": round(100 * len(highs) / len(self._history), 1),
            "ne_scale_mean": round(self.ne_scale(), 4),
            "ecb_scale_mean":round(self.ecb_scale(), 4),
            "domain_resets": len(self._resets),
            "reset_reasons": [r["reason"] for r in self._resets],
            "regime":        self.suggested_regime(),
        }

    def reset_epoch(self):
        """Call at start of each epoch to clear per-epoch history."""
        self._history.clear()

    def _flush_log(self):
        with open(self._log_path, "a") as f:
            for r in self._history[-100:]:
                f.write(json.dumps({"type": "step", **r}) + "\n")


# ── Training loop integration ────────────────────────────────────────────────

class NeuromodulatorState:
    """
    Convenience wrapper: holds all eight signal values for one batch.
    Pass into loss weighting functions.
    """
    __slots__ = ("da", "sht", "ne", "ach", "ecb", "ado", "ei", "cort")

    def __init__(self,
                 da=0.0, sht=0.0, ne=1.0, ach=1.0,
                 ecb=1.0, ado=1.0, ei=0.5, cort=0.0):
        self.da   = da    # dopamine — prediction surprise
        self.sht  = sht   # serotonin — diversity
        self.ne   = ne    # norepinephrine — uncertainty scale
        self.ach  = ach   # acetylcholine — contact weight
        self.ecb  = ecb   # endocannabinoid — context novelty
        self.ado  = ado   # adenosine — fatigue
        self.ei   = ei    # E/I balance — arousal
        self.cort = cort  # cortisol — stress / domain shift

    def apply_cortisol(self, cort_signal: CortisolSignalAdaptive) -> None:
        """Update NE and eCB in-place from cortisol signal."""
        self.ne   *= cort_signal.ne_scale()
        self.ecb  *= cort_signal.ecb_scale()
        self.cort  = cort_signal.value


# ── Example integration snippet ───────────────────────────────────────────────
#
# In train_cwm_multidomain.py, replace:
#
#   cort_signal = CortisolSignal(sensitivity=0.05)
#
# with:
#
#   from cortisol_domain_adaptive import CortisolSignalAdaptive, CortisolConfig
#   cort_cfg = CortisolConfig(sensitivity=0.10, ne_scale_high=1.22, ecb_scale_high=0.82)
#   cort_signal = CortisolSignalAdaptive(cfg=cort_cfg)
#
# Then in the batch loop:
#
#   # Extract GPS from HDF5 batch (north_m, east_m)
#   gps_batch = batch.get("gps_ne")           # (B, 2) or None
#   cort_val  = cort_signal.step(
#       loss=total_loss.item(),
#       gps=gps_batch,
#       z_enc=z_enc,                           # StudentEncoder output
#   )
#   neuromod.apply_cortisol(cort_signal)
#   regime    = cort_signal.suggested_regime() # replaces fixed schedule
#
# At end of epoch:
#   print(cort_signal.summary())
#   cort_signal.reset_epoch()
#
# ─────────────────────────────────────────────────────────────────────────────


# ── Standalone test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Cortisol Domain-Adaptive Signal — Self Test")
    print("=" * 60)

    cfg = CortisolConfig(sensitivity=0.10, log_transitions=False)
    cort = CortisolSignalAdaptive(cfg=cfg)

    # Phase 1: stable training (RECON-like, loss declining)
    print("\nPhase 1: stable RECON training (50 steps)")
    for i in range(50):
        loss = 0.55 - i * 0.003 + random.gauss(0, 0.005)
        c = cort.step(loss)
    s = cort.summary()
    print(f"  cort_mean={s['cort_mean']}, cort_max={s['cort_max']}, regime={s['regime']}")
    assert s["cort_max"] < 0.15, "Stable phase should have low cortisol"
    cort.reset_epoch()

    # Phase 2: domain shift (TwoRoom entry — loss spikes)
    print("\nPhase 2: TwoRoom domain shift (20 steps, loss spike)")
    for i in range(20):
        loss = 0.40 + i * 0.025 + random.gauss(0, 0.008)
        gps_shift = torch.tensor([[i * 5.0, i * 3.0]])   # rapid bbox expansion
        c = cort.step(loss, gps=gps_shift)
    s = cort.summary()
    print(f"  cort_mean={s['cort_mean']}, cort_max={s['cort_max']}, regime={s['regime']}")
    print(f"  domain_resets={s['domain_resets']}, reasons={s['reset_reasons']}")
    print(f"  ne_scale={s['ne_scale_mean']}, ecb_scale={s['ecb_scale_mean']}")
    cort.reset_epoch()

    # Phase 3: recovery after domain reset
    print("\nPhase 3: TwoRoom learning (loss recovering)")
    for i in range(30):
        loss = 0.80 - i * 0.015 + random.gauss(0, 0.005)
        c = cort.step(loss)
    s = cort.summary()
    print(f"  cort_mean={s['cort_mean']}, cort_max={s['cort_max']}, regime={s['regime']}")
    cort.reset_epoch()

    # Check NE/eCB scaling
    cort._cort = 0.30   # force high cortisol
    print(f"\nHigh cortisol (0.30):")
    print(f"  ne_scale  = {cort.ne_scale():.3f}  (target ~1.22)")
    print(f"  ecb_scale = {cort.ecb_scale():.3f}  (target ~0.82)")
    print(f"  regime    = {cort.suggested_regime()}")
    assert abs(cort.ne_scale() - 1.22) < 0.01
    assert abs(cort.ecb_scale() - 0.82) < 0.01

    print("\n✅ All assertions passed")
    print("=" * 60)
    print("Drop cortisol_domain_adaptive.py into CORTEX/ and follow")
    print("the integration snippet at the bottom of this file.")
