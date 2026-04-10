"""
neuromodulator_base.py — Shared Neuromodulator Core
====================================================
Convergence point for CORTEX-16 (trading) and NeMo-WM (navigation).

Both systems implement the same 8-signal biological architecture:
  DA   — Dopamine:        prediction error / surprise
  5HT  — Serotonin:       stability / patience
  NE   — Norepinephrine:  global arousal (Allen Neuropixels / rho)
  ACh  — Acetylcholine:   attention gate / temporal precision
  eCB  — Endocannabinoid: retrograde suppression of DA after action
  CORT — Cortisol:        sustained stress / domain shift
  E/I  — E/I ratio:       excitation/inhibition balance (derived)
  Ado  — Adenosine:       fatigue / compute budget (optional)

Subclass for each domain:
  TradingNeuromodulator  — CORTEX-16 (z-score input, RTT-based 5HT)
  NavigationNeuromodulator — NeMo-WM (loss-based DA, GPS-based domain)

Usage:
  from neuromodulator_base import TradingNeuromodulator, NavigationNeuromodulator

  # CORTEX-16
  neuro = TradingNeuromodulator()
  ns = neuro.tick(z=2.1, rho=3.0, is_toxic=False, rtt_ms=45.0, qty=0.0)
  print(ns.regime, ns.z_entry, ns.pos_scale)

  # NeMo-WM
  neuro = NavigationNeuromodulator()
  ns = neuro.tick(loss=0.42, gps=gps_tensor, z_enc=enc_tensor, rho=2.0)
  print(ns.cortisol, ns.ne_scale, ns.ecb_scale)

Tests:
  python neuromodulator_base.py
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ══════════════════════════════════════════════════════════════════════════════
# Shared state dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class NeuroState:
    """
    Full 8-signal neuromodulator state.
    Shared between trading and navigation domains.
    """
    # Primary signals [0, 1]
    da:       float = 0.500   # Dopamine — prediction error
    sht:      float = 0.800   # Serotonin — stability
    ne:       float = 0.200   # Norepinephrine — arousal
    ach:      float = 0.500   # Acetylcholine — attention
    ecb:      float = 0.000   # Endocannabinoid — retrograde
    cortisol: float = 0.000   # Cortisol — sustained stress

    # Derived signals
    ei:       float = 1.000   # E/I ratio [0.5, 2.0]
    ado:      float = 0.000   # Adenosine — fatigue [0, 1]

    # Regime classification
    regime:   str   = "EXPLOIT"   # EXPLOIT | EXPLORE | WAIT | REOBSERVE

    # Domain-specific derived params (set by subclass)
    # Trading
    z_entry:    float = 2.920
    pos_scale:  float = 1.000
    # Navigation
    ne_scale:   float = 1.000
    ecb_scale:  float = 1.000

    def __repr__(self):
        return (f"NeuroState(da={self.da:.3f} sht={self.sht:.3f} "
                f"ne={self.ne:.3f} ach={self.ach:.3f} ecb={self.ecb:.3f} "
                f"cort={self.cortisol:.3f} ei={self.ei:.3f} "
                f"regime={self.regime})")


# ══════════════════════════════════════════════════════════════════════════════
# Core neuromodulator base
# ══════════════════════════════════════════════════════════════════════════════

class NeuromodulatorBase:
    """
    Shared biological neuromodulator core.

    Implements the 8-signal update equations used by both
    CORTEX-16 (trading) and NeMo-WM (navigation).

    Subclasses override:
      _compute_da_raw(...)  → float in [0,1]   (domain-specific surprise)
      _compute_stab(...)    → float in [0,1]   (domain-specific stability)
      _compute_adverse(...) → float in {0,1}   (domain-specific stress)
      _derive_params(ns)    → None              (set domain-specific params)
      tick(...)             → NeuroState        (domain-specific signature)
    """

    # EMA decay constants (biological timescales)
    DA_DECAY:   float = 0.95
    SHT_DECAY:  float = 0.90
    ECB_DECAY:  float = 0.85
    CORT_DECAY: float = 0.97
    CORT_SENS:  float = 0.10
    ADO_RATE:   float = 0.0     # Adenosine accumulation per tick (override if used)

    # Regime thresholds
    WAIT_SHT:      float = 0.25   # 5HT below → WAIT
    EXPLORE_DA:    float = 0.65   # DA_eff above → EXPLORE

    def __init__(self, session_start: Optional[float] = None):
        self.state = NeuroState()
        self._adverse_acc = 0.0
        self._n_ticks     = 0
        self._session_start = session_start or time.time()

    # ── Core EMA update (same for all domains) ────────────────────────────────

    def _update_core(
        self,
        da_raw:   float,
        stab:     float,
        rho:      float,
        adverse:  float,
        action_magnitude: float = 0.0,
    ) -> NeuroState:
        """
        Apply one tick of the 8-signal update.
        Called by subclass tick() after computing domain-specific inputs.
        """
        s = self.state
        self._n_ticks += 1

        # ── DA: prediction error ──────────────────────────────────────────────
        s.da = self.DA_DECAY * s.da + (1 - self.DA_DECAY) * min(1.0, da_raw)

        # ── 5HT: stability ────────────────────────────────────────────────────
        s.sht = self.SHT_DECAY * s.sht + (1 - self.SHT_DECAY) * max(0.0, min(1.0, stab))

        # ── NE: global arousal from rho ───────────────────────────────────────
        s.ne = min(1.0, max(0.0, rho / 10.0))

        # ── ACh: attention gate (DA + instability) ────────────────────────────
        ach_raw = (da_raw + (1.0 - stab)) / 2.0
        s.ach = self.ECB_DECAY * s.ach + (1 - self.ECB_DECAY) * ach_raw

        # ── eCB: retrograde after action in high-DA state ─────────────────────
        ecb_raw = s.da * min(1.0, action_magnitude)
        s.ecb   = self.ECB_DECAY * s.ecb + (1 - self.ECB_DECAY) * ecb_raw
        da_eff  = s.da * (1.0 - s.ecb * 0.4)

        # ── Cortisol: sustained adverse ───────────────────────────────────────
        self._adverse_acc = self._adverse_acc * 0.9 + adverse
        cort_raw = max(0.0, self._adverse_acc / 10.0 - 0.2)
        s.cortisol = min(2.0,
            self.CORT_DECAY * s.cortisol + self.CORT_SENS * cort_raw)

        # ── Adenosine: fatigue (optional, most subclasses leave at 0) ─────────
        if self.ADO_RATE > 0:
            elapsed = time.time() - self._session_start
            s.ado = min(1.0, elapsed * self.ADO_RATE)

        # ── E/I: derived ──────────────────────────────────────────────────────
        s.ei = float(np.clip(da_eff / (1.0 - s.sht + 0.1), 0.5, 2.0))

        # ── Regime ────────────────────────────────────────────────────────────
        if s.sht < self.WAIT_SHT:
            s.regime = "WAIT"
        elif da_eff > self.EXPLORE_DA:
            s.regime = "EXPLORE"
        else:
            s.regime = "EXPLOIT"

        # ── Domain-specific derived params ────────────────────────────────────
        self._derive_params(s, da_eff)

        return s

    # ── Override hooks ────────────────────────────────────────────────────────

    def _derive_params(self, ns: NeuroState, da_eff: float) -> None:
        """Override to set domain-specific output params on ns."""
        pass

    def reset(self, full: bool = True):
        """Reset state. full=True clears adenosine (sleep analogy)."""
        self.state = NeuroState()
        self._adverse_acc = 0.0
        if full:
            self._session_start = time.time()

    @property
    def signals(self) -> NeuroState:
        return self.state


# ══════════════════════════════════════════════════════════════════════════════
# CORTEX-16: Trading neuromodulator
# ══════════════════════════════════════════════════════════════════════════════

class TradingNeuromodulator(NeuromodulatorBase):
    """
    CORTEX-16 trading neuromodulator.

    DA source:   z-score prediction error (rho-based)
    5HT source:  RTT stability (market connection quality)
    Adverse:     is_toxic OR rtt_ms > RTT_ADVERSE_THRESH
    Derived:     z_entry (cortisol-elevated), pos_scale (cortisol-reduced)

    Replaces DopamineSystem in cortex_brain_inject.py.
    """

    BASE_Z_ENTRY:      float = 2.92
    RTT_ADVERSE_THRESH: float = 120.0  # ms — calibrated to GMKtec↔Alpaca RTT
    WARMUP_TICKS:      int   = 200     # suppress cortisol during warmup

    def __init__(self):
        super().__init__()
        self._z_pred = 0.0

    def tick(
        self,
        z:         float,
        rho:       float,
        is_toxic:  bool,
        rtt_ms:    float,
        qty:       float = 0.0,
    ) -> NeuroState:
        """
        One trading tick.

        Args:
            z:        rho z-score (from cortex_math.calculate_z_score)
            rho:      Allen Neuropixels spike count
            is_toxic: True if order book imbalance above threshold
            rtt_ms:   Alpaca API round-trip time in milliseconds
            qty:      current position size (for eCB)

        Returns:
            NeuroState with z_entry and pos_scale set
        """
        # DA: z-score prediction error
        da_raw = min(1.0, abs(z - self._z_pred) / (abs(self._z_pred) + 0.1))
        self._z_pred = z

        # 5HT: RTT stability (low RTT = high stability)
        stab = 0.8  # fixed floor — RTT tracked via cortisol instead

        # Adverse: only real market stress, not cold-start noise
        if self._n_ticks > self.WARMUP_TICKS:
            adverse = float(is_toxic or rtt_ms > self.RTT_ADVERSE_THRESH)
        else:
            adverse = 0.0

        # eCB magnitude: position size as action proxy
        action_mag = min(1.0, qty / 10.0)

        return self._update_core(da_raw, stab, rho, adverse, action_mag)

    def _derive_params(self, ns: NeuroState, da_eff: float):
        """Set trading-specific params."""
        ns.z_entry   = self.BASE_Z_ENTRY * (1.0 + ns.cortisol * 0.5)
        ns.pos_scale = max(0.05, 1.0 - ns.cortisol * 0.7)


# ══════════════════════════════════════════════════════════════════════════════
# NeMo-WM: Navigation neuromodulator
# ══════════════════════════════════════════════════════════════════════════════

class NavigationNeuromodulator(NeuromodulatorBase):
    """
    NeMo-WM navigation neuromodulator.

    DA source:   training loss prediction error (world model surprise)
    5HT source:  loss stability (rolling variance)
    Adverse:     loss spike OR visual domain shift OR GPS bbox expansion
    Derived:     ne_scale, ecb_scale (for NeMo-WM train loop)

    Replaces CortisolSignalAdaptive in train_cwm_multidomain.py.
    Now includes full 8-signal dynamics, not just cortisol.
    """

    SENSITIVITY:    float = 0.10   # Sprint 9 value
    NE_SCALE_HIGH:  float = 1.22
    NE_SCALE_LOW:   float = 1.00
    ECB_SCALE_HIGH: float = 0.82
    ECB_SCALE_LOW:  float = 1.00
    HIGH_CORT:      float = 0.25   # cortisol above → high-cortisol regime

    def __init__(self, sensitivity: float = 0.10):
        super().__init__()
        self.SENSITIVITY = sensitivity
        self._loss_hist  = deque(maxlen=20)
        self._loss_pred  = 0.5     # initial loss prediction

        # GPS domain detection
        self._gps_min = np.array([+1e9, +1e9])
        self._gps_max = np.array([-1e9, -1e9])

        # Visual domain detection
        self._z_mean = None

    def tick(
        self,
        loss:      float,
        rho:       float       = 0.0,
        gps:       Optional[object] = None,    # torch.Tensor (B, 2) or None
        z_enc:     Optional[object] = None,    # torch.Tensor (B, D) or None
        action_magnitude: float = 0.0,
    ) -> NeuroState:
        """
        One navigation training tick.

        Args:
            loss:    current batch loss value
            rho:     Allen Neuropixels arousal (optional, 0 if unavailable)
            gps:     GPS coordinates for domain detection (optional)
            z_enc:   encoder embeddings for visual domain detection (optional)
            action_magnitude: magnitude of last action for eCB

        Returns:
            NeuroState with ne_scale and ecb_scale set
        """
        self._loss_hist.append(loss)

        # DA: loss prediction error
        da_raw = min(1.0, abs(loss - self._loss_pred) / (self._loss_pred + 0.01))
        self._loss_pred = (0.95 * self._loss_pred + 0.05 * loss)

        # 5HT: loss stability (low loss variance = stable training)
        if len(self._loss_hist) > 5:
            variance = float(np.var(list(self._loss_hist)))
            stab = math.exp(-10.0 * variance)
        else:
            stab = 0.8

        # Adverse: domain shift detection
        adverse = self._detect_domain_shift(loss, gps, z_enc)

        return self._update_core(da_raw, stab, rho, adverse, action_magnitude)

    def _detect_domain_shift(self, loss, gps, z_enc) -> float:
        """Returns 1.0 if domain shift detected, 0.0 otherwise."""
        # Loss spike: loss > 2× running mean
        if len(self._loss_hist) > 5:
            mean_loss = float(np.mean(list(self._loss_hist)[:-1]))
            if loss > mean_loss * 2.0:
                return 1.0

        # GPS bbox expansion
        if gps is not None and HAS_TORCH:
            import torch
            if isinstance(gps, torch.Tensor):
                g = gps.detach().cpu().numpy()[:, :2]
                old_area = float(np.prod(np.maximum(0, self._gps_max - self._gps_min)))
                self._gps_min = np.minimum(self._gps_min, g.min(0))
                self._gps_max = np.maximum(self._gps_max, g.max(0))
                new_area = float(np.prod(np.maximum(0, self._gps_max - self._gps_min)))
                if new_area > old_area * 1.5 and old_area > 0:
                    return 1.0

        # Visual cosine drift
        if z_enc is not None and HAS_TORCH:
            import torch
            z = F.normalize(z_enc.detach().float(), dim=-1).mean(0)
            if self._z_mean is None:
                self._z_mean = z.clone()
            else:
                cos_sim = float((self._z_mean * z).sum())
                self._z_mean = 0.95 * self._z_mean + 0.05 * z
                if cos_sim < 0.65:
                    return 1.0

        return 0.0

    def _derive_params(self, ns: NeuroState, da_eff: float):
        """Set navigation-specific params."""
        high = ns.cortisol > self.HIGH_CORT
        ns.ne_scale  = self.NE_SCALE_HIGH  if high else self.NE_SCALE_LOW
        ns.ecb_scale = self.ECB_SCALE_HIGH if high else self.ECB_SCALE_LOW

    def reset_baseline(self, reason: str = ""):
        """Reset cortisol baseline on domain entry (Sprint 9 protocol)."""
        self._adverse_acc = 0.0
        self._loss_hist.clear()
        self._loss_pred = self.state.cortisol * 0.3  # partial memory
        if reason:
            print(f"[NavigationNeuromodulator] baseline reset: {reason}")


# ══════════════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════════════

def _test():
    PASS = "PASS"
    FAIL = "FAIL"
    results = []

    def check(name, cond, detail=""):
        ok = bool(cond)
        results.append((name, ok))
        print(f"  {'OK' if ok else 'XX'}  {name}" + (f" -- {detail}" if detail else ""))
        return ok

    print("\n" + "="*58)
    print("  NeuromodulatorBase test suite")
    print("="*58)

    # ── T1: NeuroState ────────────────────────────────────────────────────────
    print("\n[T1] NeuroState")
    ns = NeuroState()
    check("Initial DA=0.5", abs(ns.da - 0.5) < 0.01)
    check("Initial cortisol=0.0", ns.cortisol == 0.0)
    check("Repr works", "NeuroState" in repr(ns))

    # ── T2: TradingNeuromodulator ─────────────────────────────────────────────
    print("\n[T2] TradingNeuromodulator")
    tm = TradingNeuromodulator()

    # Warmup
    for _ in range(250):
        ns = tm.tick(z=0.5, rho=2.0, is_toxic=False, rtt_ms=45.0)
    check("Warmup: DA in [0,1]", 0 <= ns.da <= 1, f"da={ns.da:.4f}")
    check("Warmup: z_entry near base", 2.9 <= ns.z_entry <= 3.1,
          f"z_entry={ns.z_entry:.3f}")
    check("Warmup: pos_scale near 1.0", ns.pos_scale > 0.9,
          f"scale={ns.pos_scale:.3f}")
    check("Warmup: cortisol low", ns.cortisol < 0.1,
          f"cortisol={ns.cortisol:.4f}")

    # Stress scenario
    for _ in range(50):
        ns = tm.tick(z=5.0, rho=8.0, is_toxic=True, rtt_ms=200.0)
    check("Stress: cortisol rises", ns.cortisol > 0.1,
          f"cortisol={ns.cortisol:.4f}")
    check("Stress: z_entry elevated", ns.z_entry > 2.92,
          f"z_entry={ns.z_entry:.3f}")
    check("Stress: pos_scale reduced", ns.pos_scale < 1.0,
          f"pos_scale={ns.pos_scale:.3f}")

    # Recovery
    for _ in range(100):
        ns = tm.tick(z=0.3, rho=1.0, is_toxic=False, rtt_ms=40.0)
    check("Recovery: cortisol decays", ns.cortisol < 0.5,
          f"cortisol={ns.cortisol:.4f}")

    # WAIT regime
    tm2 = TradingNeuromodulator()
    tm2.state.sht = 0.1
    ns2 = tm2.tick(z=1.0, rho=1.0, is_toxic=False, rtt_ms=40.0)
    check("WAIT when 5HT low", ns2.regime == "WAIT",
          f"regime={ns2.regime}")

    # ── T3: NavigationNeuromodulator ─────────────────────────────────────────
    print("\n[T3] NavigationNeuromodulator")
    nm = NavigationNeuromodulator(sensitivity=0.10)

    # Normal training
    for i in range(30):
        ns = nm.tick(loss=0.5 - i*0.01, rho=2.0)
    check("Nav warmup: signals in [0,1]", 0 <= ns.da <= 1,
          f"da={ns.da:.4f}")
    check("Nav warmup: cortisol low", ns.cortisol < 0.3,
          f"cortisol={ns.cortisol:.4f}")

    # Loss spike → domain shift
    ns_spike = nm.tick(loss=5.0, rho=2.0)  # 10× normal
    check("Loss spike raises adverse", nm._adverse_acc > 0,
          f"adverse={nm._adverse_acc:.3f}")

    # ne_scale / ecb_scale set
    check("ne_scale set", ns.ne_scale in [nm.NE_SCALE_HIGH, nm.NE_SCALE_LOW],
          f"ne_scale={ns.ne_scale}")
    check("ecb_scale set", ns.ecb_scale in [nm.ECB_SCALE_HIGH, nm.ECB_SCALE_LOW],
          f"ecb_scale={ns.ecb_scale}")

    # reset_baseline
    nm.reset_baseline("test_domain_entry")
    check("Reset: adverse cleared", nm._adverse_acc == 0.0)

    # ── T4: Shared base invariants ────────────────────────────────────────────
    print("\n[T4] Shared base invariants")
    for Cls, name in [(TradingNeuromodulator, "Trading"),
                      (NavigationNeuromodulator, "Navigation")]:
        m = Cls()
        if name == "Trading":
            for _ in range(100): m.tick(z=1.0, rho=3.0, is_toxic=False, rtt_ms=45.0)
        else:
            for _ in range(100): m.tick(loss=0.4, rho=2.0)
        ns = m.state
        check(f"{name}: DA in [0,1]",       0 <= ns.da       <= 1, f"{ns.da:.4f}")
        check(f"{name}: 5HT in [0,1]",      0 <= ns.sht      <= 1, f"{ns.sht:.4f}")
        check(f"{name}: NE in [0,1]",       0 <= ns.ne       <= 1, f"{ns.ne:.4f}")
        check(f"{name}: eCB in [0,1]",      0 <= ns.ecb      <= 1, f"{ns.ecb:.4f}")
        check(f"{name}: cortisol in [0,2]", 0 <= ns.cortisol <= 2, f"{ns.cortisol:.4f}")
        check(f"{name}: E/I in [0.5,2]",    0.5 <= ns.ei     <= 2, f"{ns.ei:.4f}")
        check(f"{name}: regime valid",
              ns.regime in ["EXPLOIT","EXPLORE","WAIT","REOBSERVE"])

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*58}")
    passed = sum(1 for _, ok in results if ok)
    total  = len(results)
    print(f"  {passed}/{total} tests passed")
    if passed < total:
        print("  Failed:")
        for name, ok in results:
            if not ok: print(f"    XX {name}")
    print(f"{'='*58}\n")
    return passed == total


if __name__ == "__main__":
    import sys
    sys.exit(0 if _test() else 1)
