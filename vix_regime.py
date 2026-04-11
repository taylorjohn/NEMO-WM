"""
vix_regime.py -- VIX-driven market regime classifier for CORTEX-16
===================================================================
Queries VIX at session open and returns an immutable RegimeConfig
that gates the actuator's Z threshold, spread max, and position size
for the entire session.

Regime table (VIX-based):
    LOW     VIX < 18   -> Z >= 2.0, spread <= $0.25, alloc 1.5%  (calm, more aggressive)
    NORMAL  VIX 18-25  -> Z >= 2.5, spread <= $0.15, alloc 1.0%  (standard)
    HIGH    VIX 25-40  -> Z >= 3.0, spread <= $0.10, alloc 0.5%  (volatile, tighten)
    EXTREME VIX >= 40  -> SUSPEND -- no entries, vault protection only
    UNKNOWN (fallback) -> NORMAL thresholds, flagged in log

Empirical calibration (2026-03-31 / 2026-04-01):
    2026-03-31: VIX=30.61 at close -> HIGH regime
    2026-04-01: VIX ~24.5 intraday -> NORMAL regime
    SPY spreads $0.33-$0.57 on 2026-04-01 open -> HIGH or EXTREME conditions
    HIGH regime spread threshold ($0.10) correctly blocks these entries

Wire-in points for run_trading.py:
    1. Import at top:
         from vix_regime import get_session_regime, RegimeConfig, RegimeLabel
    2. After market open confirmed, call once:
         regime = get_session_regime(api)
         actuator.set_regime(regime)
    3. In InstitutionalActuator.set_regime():
         self._regime = regime
         self._rtt_max = regime.rtt_max
         # Phase 9 spread threshold is passed through run_pre_execution_audit
    4. In InstitutionalActuator.act() before run_pre_execution_audit():
         if self._regime.suspend:
             log.warning("REGIME SUSPEND: %s -- no entries", self._regime.label)
             return 0.0
    5. Pass regime thresholds to run_pre_execution_audit():
         results = run_pre_execution_audit(
             ...
             z_score=z_score,
             max_spread=self._regime.max_spread,  # replaces hardcoded $0.15
             ...
         )

Usage:
    python vix_regime.py --test          # self-test without Alpaca
    python vix_regime.py --vix 30.5      # manual override, print regime
"""

from __future__ import annotations

import logging
import argparse
from dataclasses import dataclass
from enum import Enum
from typing import Optional

log = logging.getLogger("cortex.regime")


# ===========================================================================
# Regime labels
# ===========================================================================

class RegimeLabel(str, Enum):
    LOW     = "LOW"      # VIX < 18  -- calm, more aggressive
    NORMAL  = "NORMAL"   # VIX 18-25 -- standard session
    HIGH    = "HIGH"     # VIX 25-40 -- volatile, tighten everything
    EXTREME = "EXTREME"  # VIX >= 40 -- suspend entries, vault protection only
    UNKNOWN = "UNKNOWN"  # VIX fetch failed -- fall back to NORMAL


# ===========================================================================
# Regime config
# ===========================================================================

@dataclass(frozen=True)
class RegimeConfig:
    """
    Immutable per-session trading parameters derived from VIX.

    Fields:
        label:      RegimeLabel
        vix:        raw VIX at session open (0.0 if unknown)
        z_alpha:    minimum Z-score to pass Phase 5 / Phase 14
        max_spread: maximum bid-ask spread in dollars (Phase 9)
        alloc_pct:  target allocation as fraction of active pool
        rtt_max:    RTT threshold in ms (Phase 2) -- tighter in HIGH
        suspend:    if True, no entries at all (EXTREME only)
        note:       optional warning message
    """
    label:      RegimeLabel
    vix:        float
    z_alpha:    float
    max_spread: float
    alloc_pct:  float
    rtt_max:    float = 500.0
    suspend:    bool  = False
    note:       str   = ""

    def __str__(self) -> str:
        parts = [
            f"[REGIME] {self.label.value}",
            f"VIX={self.vix:.1f}",
            f"Z>={self.z_alpha}",
            f"spread<=${self.max_spread:.2f}",
            f"alloc={self.alloc_pct*100:.1f}%",
            f"RTT<{self.rtt_max:.0f}ms",
        ]
        if self.suspend:
            parts.append("*** ENTRIES SUSPENDED ***")
        if self.note:
            parts.append(f"note={self.note}")
        return " | ".join(parts)


# ===========================================================================
# Regime boundaries and parameters
# ===========================================================================

_LOW_VIX_CEILING     = 18.0
_HIGH_VIX_FLOOR      = 25.0
_EXTREME_VIX_FLOOR   = 40.0

# Full parameter table per regime
_REGIME_PARAMS = {
    RegimeLabel.LOW: dict(
        z_alpha    = 2.0,
        max_spread = 0.25,
        alloc_pct  = 0.015,
        rtt_max    = 500.0,
        suspend    = False,
    ),
    RegimeLabel.NORMAL: dict(
        z_alpha    = 2.5,
        max_spread = 0.15,
        alloc_pct  = 0.010,
        rtt_max    = 500.0,
        suspend    = False,
    ),
    RegimeLabel.HIGH: dict(
        z_alpha    = 3.0,
        max_spread = 0.10,
        alloc_pct  = 0.005,
        rtt_max    = 300.0,   # tighter RTT in high volatility
        suspend    = False,
    ),
    RegimeLabel.EXTREME: dict(
        z_alpha    = 9.99,    # effectively infinite -- nothing passes
        max_spread = 0.00,
        alloc_pct  = 0.000,
        rtt_max    = 100.0,
        suspend    = True,    # hard suspend flag
    ),
}

# UNKNOWN uses NORMAL thresholds with a warning note
_REGIME_PARAMS[RegimeLabel.UNKNOWN] = _REGIME_PARAMS[RegimeLabel.NORMAL].copy()


# ===========================================================================
# Classification and construction
# ===========================================================================

def classify_vix(vix: float) -> RegimeLabel:
    """Pure function -- classify a VIX level into a regime label."""
    if vix >= _EXTREME_VIX_FLOOR:
        return RegimeLabel.EXTREME
    if vix >= _HIGH_VIX_FLOOR:
        return RegimeLabel.HIGH
    if vix < _LOW_VIX_CEILING:
        return RegimeLabel.LOW
    return RegimeLabel.NORMAL


def make_regime(
    label: RegimeLabel,
    vix:   float = 0.0,
    note:  str   = "",
) -> RegimeConfig:
    """Construct a RegimeConfig from a label."""
    return RegimeConfig(
        label=label,
        vix=vix,
        note=note,
        **_REGIME_PARAMS[label],
    )


# ===========================================================================
# VIX fetch
# ===========================================================================

def _fetch_vix(api) -> Optional[float]:
    """
    Fetch the latest VIX level from Alpaca market data.
    Returns None on failure so caller can fall back gracefully.
    """
    # Primary: Alpaca bars for VIX index
    for symbol in ("VIX", "^VIX", "VIXY"):
        try:
            bar = api.get_latest_bar(symbol)
            if bar and hasattr(bar, "c"):
                vix = float(bar.c)
                if 5.0 <= vix <= 150.0:   # sanity bounds
                    log.debug("VIX fetched via %s: %.2f", symbol, vix)
                    return vix
        except Exception:
            continue

    # Fallback: SPY intraday range as rough vol proxy
    # 1% daily range ~ VIX 16, 2% ~ VIX 32 (empirical approximation)
    try:
        snap = api.get_snapshot("SPY")
        bar  = snap.daily_bar
        if bar and bar.l > 0:
            range_pct   = (bar.h - bar.l) / bar.l * 100.0
            implied_vix = range_pct * 16.0
            log.warning(
                "VIX feed unavailable -- SPY range proxy: range=%.2f%% "
                "implied_vix=%.1f",
                range_pct, implied_vix,
            )
            return implied_vix
    except Exception:
        pass

    return None


# ===========================================================================
# Public API
# ===========================================================================

def get_session_regime(
    api,
    *,
    force_vix: Optional[float] = None,
) -> RegimeConfig:
    """
    Query VIX and return the immutable session RegimeConfig.

    Call ONCE after market open is confirmed (9:30 ET).
    Result governs all entries for the entire session.

    Args:
        api:        Alpaca REST client (already authenticated)
        force_vix:  Manual VIX override for testing or emergency

    Returns:
        RegimeConfig with session-appropriate thresholds

    Example log output:
        [REGIME] HIGH | VIX=30.6 | Z>=3.0 | spread<=$0.10 | alloc=0.5% | RTT<300ms
    """
    if force_vix is not None:
        label  = classify_vix(force_vix)
        regime = make_regime(label, vix=force_vix, note="manual override")
        log.info("%s", regime)
        return regime

    vix = _fetch_vix(api)

    if vix is None:
        regime = make_regime(
            RegimeLabel.UNKNOWN,
            vix=0.0,
            note="VIX unavailable -- NORMAL fallback",
        )
        log.warning("%s", regime)
        return regime

    label  = classify_vix(vix)
    regime = make_regime(label, vix=vix)
    log.info("%s", regime)

    if regime.suspend:
        log.critical(
            "EXTREME REGIME (VIX=%.1f >= %.0f) -- "
            "ALL ENTRIES SUSPENDED. Vault protection only.",
            vix, _EXTREME_VIX_FLOOR,
        )

    return regime


# Default regime used before market open / in tests
DEFAULT_REGIME = make_regime(RegimeLabel.NORMAL, vix=20.0, note="pre-open default")


# ===========================================================================
# Wire-in patch for run_trading.py
# ===========================================================================

WIRE_IN_PATCH = """
# ── vix_regime wire-in for run_trading.py ────────────────────────────────────
# Add to imports:
from vix_regime import get_session_regime, DEFAULT_REGIME

# Add to InstitutionalActuator.__init__ (after self._heartbeat line):
self._regime = DEFAULT_REGIME

# Add to InstitutionalActuator as new method:
def set_regime(self, regime):
    self._regime = regime
    self._rtt_max = regime.rtt_max
    log.info("Regime set: %s", regime)

# Add to InstitutionalActuator.act() BEFORE the circuit breaker check:
if self._regime.suspend:
    log.warning("REGIME=%s SUSPEND -- no entries", self._regime.label.value)
    return 0.0

# In run_pre_execution_audit kwargs, add:
max_spread = self._regime.max_spread,   # was hardcoded 0.15

# Add to main() after market open confirmed (around line 416):
if not args.sim:
    regime = get_session_regime(api)
    actuator.set_regime(regime)
# ─────────────────────────────────────────────────────────────────────────────
"""


# ===========================================================================
# Self-test
# ===========================================================================

def self_test():
    """Verify all regimes and thresholds without Alpaca connection."""
    print("vix_regime self-test...")

    cases = [
        (10.0,  RegimeLabel.LOW,     2.0,  0.25, False),
        (17.9,  RegimeLabel.LOW,     2.0,  0.25, False),
        (18.0,  RegimeLabel.NORMAL,  2.5,  0.15, False),
        (24.9,  RegimeLabel.NORMAL,  2.5,  0.15, False),
        (25.0,  RegimeLabel.HIGH,    3.0,  0.10, False),
        (30.61, RegimeLabel.HIGH,    3.0,  0.10, False),  # 2026-03-31 close
        (39.9,  RegimeLabel.HIGH,    3.0,  0.10, False),
        (40.0,  RegimeLabel.EXTREME, 9.99, 0.00, True),   # suspend
        (85.0,  RegimeLabel.EXTREME, 9.99, 0.00, True),   # COVID-level
    ]

    for vix, expected_label, expected_z, expected_spread, expected_suspend in cases:
        label  = classify_vix(vix)
        regime = make_regime(label, vix=vix)
        assert label == expected_label, \
            f"VIX={vix}: expected {expected_label}, got {label}"
        assert regime.z_alpha == expected_z, \
            f"VIX={vix}: expected z_alpha={expected_z}, got {regime.z_alpha}"
        assert regime.max_spread == expected_spread, \
            f"VIX={vix}: expected spread={expected_spread}, got {regime.max_spread}"
        assert regime.suspend == expected_suspend, \
            f"VIX={vix}: expected suspend={expected_suspend}"
        print(f"  VIX={vix:>5.1f} -> {regime}")

    # Test force_vix path
    class MockAPI:
        pass
    regime = get_session_regime(MockAPI(), force_vix=30.61)
    assert regime.label == RegimeLabel.HIGH
    assert regime.suspend is False
    print(f"\n  force_vix=30.61: {regime}")

    regime = get_session_regime(MockAPI(), force_vix=45.0)
    assert regime.suspend is True
    print(f"  force_vix=45.0:  {regime}")

    print("\n  All assertions passed.\n")

    # Today's session diagnosis
    print("  2026-04-01 session diagnosis:")
    print("  VIX at March 31 close: 30.61")
    r = make_regime(classify_vix(30.61), vix=30.61)
    print(f"  -> {r}")
    print(f"  -> SPY spreads $0.33-$0.57 vs max_spread=${r.max_spread:.2f}")
    print(f"  -> Phase 9 would still abort (spread > max even in HIGH regime)")
    print(f"  -> Correct behavior: system refusing to trade into toxic spreads")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    p = argparse.ArgumentParser(description="vix_regime -- CORTEX-16 regime classifier")
    p.add_argument("--test",   action="store_true", help="Run self-test without Alpaca")
    p.add_argument("--vix",    type=float,          help="Classify a specific VIX level")
    p.add_argument("--patch",  action="store_true", help="Print wire-in patch for run_trading.py")
    args = p.parse_args()

    if args.test:
        self_test()

    if args.vix is not None:
        label  = classify_vix(args.vix)
        regime = make_regime(label, vix=args.vix)
        print(regime)

    if args.patch:
        print(WIRE_IN_PATCH)

    if not any([args.test, args.vix, args.patch]):
        self_test()
