# CORTEX-16 — Path to Profit
> 2026-04-01 · GMKtec EVO-X2 · Alpaca paper API · Vault $101,133.47

---

## Current Reality (honest assessment)

CORTEX-16 is a **vault preservation system that occasionally trades**.
It is not yet a money-making system.

**What it does well:**
- Preserves capital under adverse conditions (March 31: 21,822 aborts, vault intact)
- Correctly identifies untradeable conditions and does nothing
- All 50 safety tests pass every session
- Circuit breaker, RTT guard, spread filter all working

**What it does not do yet:**
- Generate consistent entries
- Produce measurable returns on paper
- Validate that the Z signal actually makes money at scale

Zero entries across multiple sessions is not a failure of execution — it is
the correct behaviour of a system optimised for preservation over returns.
But preservation without returns is not a business. The system needs to trade
to prove the signal works.

---

## Why It's Not Trading — Root Causes

### 1. Z threshold too conservative (primary blocker)
Current: **Z ≥ 2.18**
This threshold requires resonance in the top ~3% of signal distribution.
On a normal day this fires 2-3 times in the 9:30–10:15 window.
On quarter-end / adverse days it fires zero times.
Result: most sessions produce 0 entries.

### 2. Alpaca snapshot endpoint timeouts (infrastructure)
The data feed times out every ~30 ticks (3-second retry visible in March 31 logs).
When the snapshot times out, RTT spikes above 150ms and kills the RTT gate.
A qualifying Z signal at tick 102,895 could be blocked by a snapshot timeout
at tick 102,894. This is data feed reliability, not signal quality.
Result: some entries that should happen are silently blocked.

### 3. Trading window too narrow (opportunity)
Current: **9:30–10:15 ET** (45 minutes)
The Z signal is strongest in the first 30 minutes but evidence suggests it
can persist to 10:30 on high-rho days. 45 minutes is conservative.
Result: valid signals after 10:15 are ignored.

### 4. Entry cooldown interactions (minor)
60-second cooldown after each entry (or abort) means rapid signal sequences
produce only 1 entry per minute maximum. On high-volatility days this is
fine. On quiet days the single qualifying signal appears and the cooldown
prevents follow-up.

---

## Three Paths to Profit (prioritised)

### Path 1 — Fix the data feed (highest ROI, lowest risk)
**Action:** Resolve Alpaca snapshot endpoint timeout issue.
**What to do:**
- Switch from snapshot endpoint to streaming quotes for SPY
- Or implement a local price cache that falls back to last-known-good
  rather than triggering the 3-second retry that spikes RTT
- Target: RTT < 50ms on 95% of ticks (currently ~30% of ticks spike)

**Why first:** This is infrastructure, not signal tuning. It doesn't change
the risk profile. It just removes an artificial blocker that's killing entries
the system should be taking. Free alpha.

**Expected impact:** 2-4x more qualifying entries on normal days without
touching the Z threshold or risk parameters.

---

### Path 2 — Lower Z threshold to 2.00 (medium effort, medium risk)
**Current:** Z ≥ 2.18
**Proposed:** Z ≥ 2.00

**What changes:** Entries fire on the top ~5% of signal distribution
instead of top ~3%. Approximately doubles the number of qualifying signals.

**Risk:** More entries on weaker signals. The signal at Z=2.05 is less
reliable than Z=2.18. Some of these entries will be losers.

**Validation approach:** Run 30 days of paper trading at Z=2.00.
Need minimum 20 entries to have statistical power.
If win rate > 55% and average win > average loss, signal is confirmed.
If win rate < 50%, revert to 2.18 or investigate signal quality.

**Do not lower below 2.00** without the data feed fix in place — combining
a lower threshold with unreliable RTT data creates false confidence.

**Expected impact:** 3-8 entries per week on normal market days.

---

### Path 3 — Extend trading window to 10:30 ET (low effort, low risk)
**Current:** 9:30–10:15 ET
**Proposed:** 9:30–10:30 ET

**What changes:** 15 additional minutes of opportunity.
The rho signal from Allen Neuropixels has shown sustained elevation
to 10:30+ on high-activity sessions (rho=0.7806 at close on March 31).

**Risk:** Z collapses faster on low-rho days. Extending the window on
quiet days captures noise not signal. Consider making the extension
conditional: only extend to 10:30 if rho > 0.60 at 10:15.

**Implementation:**
```python
# In MarketHoursGuard:
# Current: close_et = time(16, 0) - entry_cutoff = time(10, 15)
# Change: entry_cutoff = time(10, 15) if rho < 0.60 else time(10, 30)
```

**Expected impact:** 1-2 additional entries per week, higher quality
on high-rho days.

---

## Implementation Order

| Priority | Action | Risk | Effort | Expected entries/week |
|----------|--------|------|--------|-----------------------|
| 1 | Fix Alpaca snapshot timeout | None | Medium | 2-4x current |
| 2 | Lower Z to 2.00 | Low-medium | Low | 3-8 |
| 3 | Extend window to 10:30 (rho-gated) | Low | Low | +1-2 |

**Do all three in order.** Do not skip to Path 2 before fixing the data feed.
Changing the signal threshold while the infrastructure is broken means you
cannot interpret the results — you don't know if a losing entry was a bad
signal or a bad data point.

---

## Validation Criteria (know when it's working)

The system is making money when:

1. **Minimum 20 entries** across 30 trading days (current pace: ~0/day)
2. **Win rate > 55%** on completed round-trips
3. **Average win ≥ 1.5× average loss** (asymmetric payoff)
4. **Maximum drawdown < 1%** of vault per session (current CB at 2%)
5. **Vault growth > 0** over 30-day period

Until criterion 1 is met, you cannot evaluate criteria 2-5.
The most important metric right now is **entries per week**, not P&L.

---

## What the Signal Actually Is

The Z score is derived from the MOERouterV2 spectral ribbon (128-D latent
from Allen Brain Observatory Neuropixels mouse V1 recordings). High Z means
the neural population rate (rho) has deviated significantly from its rolling
baseline — a measure of population-level arousal that correlates with
market microstructure changes in SPY.

The theory: when mouse V1 population activity spikes above baseline, it
reflects something about the shared biological substrate that also drives
human market-making behaviour. This is speculative but testable.

The Z signal has never been validated at scale with real entries.
The 13-phase safeguard system exists precisely because the signal is
unvalidated — it limits downside while the signal accumulates enough
trades to be evaluated.

**The signal is either:**
a) A genuine edge that survives transaction costs and slippage
b) A sophisticated noise generator with excellent risk management

30 days of entries at Z=2.00 will tell you which one.

---

## Morning Startup Checklist (unchanged)

```powershell
# Terminal 1 — NUC
pytest test_run_trading.py          # must be 50/50

# Terminal 2 — NUC (run at 9:30 ET)
python run_trading.py --hud-ip 127.0.0.1

# Mac — Glass Cockpit
python mac_glass_cockpit.py
```

Locked parameters (do not change without validation):
- Z threshold: 2.18 (lower to 2.00 only after data feed fix)
- RTT threshold: 150ms
- Entry cooldown: 60s
- Circuit breaker: -2% vault
- Window: 9:30–10:15 ET (extend to 10:30 after 2 weeks clean)

---

## Session Log

| Date | Entries | Aborts | Vault | Notes |
|------|---------|--------|-------|-------|
| 2026-03-31 | 0 | 21,822 | $101,133.47 | Quarter-end, spreads toxic |
| 2026-04-01 | TBD | — | $101,133.47 | First clean day of Q2 |

---

*CORTEX-16 Path to Profit · 2026-04-01 · GMKtec EVO-X2 NUC*
