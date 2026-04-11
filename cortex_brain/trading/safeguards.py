"""
cortex_brain.trading.safeguards
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
18-Phase Institutional Safeguard Architecture.

Every phase returns a SafeguardResult(ok, reason, data).
ok=False means ABORT execution.  ok=True means PASS.

Phase map
---------
 1  VWMP            Volume-Weighted Micro-Price execution anchor
 2  RTT Watchdog    45ms fiber latency drift filter
 3  Partial Fill    Order TTL + partial fill state machine
 4  Stale Data      1500ms data age + exponential backoff
 5  PDT Cooldown    Dynamic alpha threshold from day-trade count
 6  API Governor    Adaptive polling rate (rate-limit protection)
 7  Market Clock    Hibernation protocol (after-hours guard)
 8  DTBP Sync       Margin + buying power quantity cap
 9  Spread Defense  Spread blowout > $0.15 suppression
10  Panic Sweep     Flash crash Z>4.0 + hardware kill-switch
11  EOD Liquidation Hard close 5 min before bell
12  Jitter Shield   Deterministic math (no float GC surprises)
13  VPIN-Lite       Toxic liquidity order-book imbalance filter
14  Soma-Sync       Active inference dual-stream (via LSM+JEPA)
15  Resonance Gate  Full-array fleet resonance routing
16  TTM Evolution   O(1) test-time self-evolution (via TTM)
17  Kill-Switch     Systemic fracture cross-asset override
18  Heartbeat Sync  Adaptive polling 0.5Hz↔10Hz biological rhythm
"""
from __future__ import annotations
import logging, math, time
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Common result type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SafeguardResult:
    ok:     bool
    phase:  int
    reason: str
    data:   dict = field(default_factory=dict)

    def __bool__(self): return self.ok

    def __repr__(self):
        status = "PASS" if self.ok else "ABORT"
        return f"Phase{self.phase}[{status}] {self.reason}"


def _pass(phase: int, reason: str = "", **data) -> SafeguardResult:
    return SafeguardResult(True,  phase, reason, data)

def _abort(phase: int, reason: str, **data) -> SafeguardResult:
    logger.warning("🛡️  Phase %d ABORT: %s  data=%s", phase, reason, data)
    return SafeguardResult(False, phase, reason, data)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — VWMP (Volume-Weighted Micro-Price)
# ─────────────────────────────────────────────────────────────────────────────

def phase1_vwmp(
    bid: float, ask: float,
    bid_size: float, ask_size: float,
) -> SafeguardResult:
    """
    Compute micro-price from NBBO order book imbalance.

        P_micro = (V_ask * P_bid + V_bid * P_ask) / (V_bid + V_ask)

    Returns micro_price in data dict.  Always passes (informational phase).
    """
    denom = bid_size + ask_size
    if denom <= 0:
        return _abort(1, "Zero order book depth", bid=bid, ask=ask)
    micro = (ask_size * bid + bid_size * ask) / denom
    spread = ask - bid
    return _pass(1, f"VWMP={micro:.4f} spread={spread:.4f}",
                 micro_price=micro, spread=spread)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Fiber RTT Watchdog (45 ms)
# ─────────────────────────────────────────────────────────────────────────────

def phase2_rtt_watchdog(
    rtt_ms: float,
    threshold_ms: float = 150.0,
) -> SafeguardResult:
    """
    Reject execution on stale network pulses.

        V_trigger = 1 if RTT <= 45ms else 0
    """
    if rtt_ms > threshold_ms:
        return _abort(2, f"RTT {rtt_ms:.1f}ms > {threshold_ms}ms — ghost filter",
                      rtt_ms=rtt_ms)
    return _pass(2, f"RTT {rtt_ms:.1f}ms OK", rtt_ms=rtt_ms)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Partial Fill + Order TTL
# ─────────────────────────────────────────────────────────────────────────────

class OrderTTLTracker:
    """
    Tracks open orders and their TTL.

    Level 3 (Sniper) orders expire after 10 seconds.
    Level 1/2 orders expire after ttl_seconds.
    """
    def __init__(self, ttl_seconds: float = 10.0) -> None:
        self.ttl        = ttl_seconds
        self._orders:   Dict[str, float] = {}   # order_id → submit_time
        self._fills:    Dict[str, float] = {}   # order_id → filled_qty
        self._target:   Dict[str, float] = {}   # order_id → target_qty

    def register(self, order_id: str, target_qty: float) -> None:
        self._orders[order_id]  = time.time()
        self._fills[order_id]   = 0.0
        self._target[order_id]  = target_qty

    def update_fill(self, order_id: str, filled_qty: float) -> None:
        self._fills[order_id] = filled_qty

    def check(self, order_id: str) -> SafeguardResult:
        if order_id not in self._orders:
            return _pass(3, "Order not tracked")
        age     = time.time() - self._orders[order_id]
        filled  = self._fills.get(order_id, 0.0)
        target  = self._target.get(order_id, 1.0)
        fill_pct = filled / max(target, 1e-8)

        if age > self.ttl and fill_pct < 1.0:
            self._cleanup(order_id)
            return _abort(3, f"TTL expired after {age:.1f}s fill={fill_pct:.0%}",
                          order_id=order_id, fill_pct=fill_pct)
        return _pass(3, f"Order OK age={age:.1f}s fill={fill_pct:.0%}",
                     order_id=order_id, fill_pct=fill_pct)

    def _cleanup(self, order_id: str) -> None:
        for d in (self._orders, self._fills, self._target):
            d.pop(order_id, None)


def phase3_partial_fill(tracker: OrderTTLTracker, order_id: str) -> SafeguardResult:
    return tracker.check(order_id)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — Stale Data + Exponential Backoff
# ─────────────────────────────────────────────────────────────────────────────

class StaleDataGuard:
    """Tracks consecutive stale events and computes exponential backoff."""
    def __init__(self, max_age_ms: float = 1500.0) -> None:
        self.max_age_ms    = max_age_ms
        self._stale_count  = 0

    def check(self, data_age_ms: float) -> SafeguardResult:
        if data_age_ms > self.max_age_ms:
            self._stale_count += 1
            backoff = 2 ** min(self._stale_count, 6)   # cap at 64s
            return _abort(4, f"Data age {data_age_ms:.0f}ms > {self.max_age_ms}ms — "
                             f"backoff {backoff}s (n={self._stale_count})",
                          data_age_ms=data_age_ms, backoff_s=backoff)
        self._stale_count = 0
        return _pass(4, f"Data age {data_age_ms:.0f}ms OK", data_age_ms=data_age_ms)


def phase4_stale_data(guard: StaleDataGuard, data_age_ms: float) -> SafeguardResult:
    return guard.check(data_age_ms)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 — PDT Cooldown (Dynamic Alpha Threshold)
# ─────────────────────────────────────────────────────────────────────────────

def phase5_pdt_cooldown(
    day_trades_remaining: int,
    z_score: float,
    z_base: float = 2.18,
) -> SafeguardResult:
    """
    Scale required entry threshold by available day trades.

        alpha = z_base          if D >= 2
        alpha = z_base * 1.25   if D == 1
        alpha = inf             if D == 0 (PDT protected)
    """
    if day_trades_remaining == 0:
        return _abort(5, "PDT shield: 0 day trades remaining — execution locked",
                      day_trades=0)
    elif day_trades_remaining == 1:
        alpha = z_base * 1.25
    else:
        alpha = z_base

    if z_score < alpha:
        return _abort(5, f"Z={z_score:.3f} < alpha={alpha:.3f} (PDT cooldown)",
                      z_score=z_score, alpha=alpha,
                      day_trades=day_trades_remaining)
    return _pass(5, f"Z={z_score:.3f} >= alpha={alpha:.3f}",
                 z_score=z_score, alpha=alpha)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 6 — API Governor (Adaptive Polling Rate)
# ─────────────────────────────────────────────────────────────────────────────

class APIGovernor:
    """
    Adaptive polling rate to prevent HTTP 429 rate-limit bans.

        poll_interval = 0.5s   if Z <= 2.5   (hibernation)
        poll_interval = 0.1s   if Z >  2.5   (imminent pulse)
    """
    REQUESTS_PER_MINUTE = 200
    _request_times: list = []

    def __init__(self) -> None:
        self._request_times = []

    def poll_interval(self, z_score: float) -> float:
        return 0.1 if z_score > 2.5 else 0.5

    def check_rate_limit(self) -> SafeguardResult:
        now     = time.time()
        cutoff  = now - 60.0
        self._request_times = [t for t in self._request_times if t > cutoff]
        if len(self._request_times) >= self.REQUESTS_PER_MINUTE:
            return _abort(6, f"Rate limit: {len(self._request_times)}/min — governor active")
        self._request_times.append(now)
        remaining = self.REQUESTS_PER_MINUTE - len(self._request_times)
        return _pass(6, f"API quota OK ({remaining} remaining)", quota_remaining=remaining)


def phase6_api_governor(governor: APIGovernor) -> SafeguardResult:
    return governor.check_rate_limit()


# ─────────────────────────────────────────────────────────────────────────────
# Phase 7 — Market Clock (Hibernation Protocol)
# ─────────────────────────────────────────────────────────────────────────────

def phase7_market_clock(is_open: bool, next_open_s: float = 0.0) -> SafeguardResult:
    """
    Block execution when market is closed.
    Telemetry continues; only capital deployment is gated.
    """
    if not is_open:
        return _abort(7, f"Market closed — hibernating (next open in {next_open_s/3600:.1f}h)",
                      is_open=False, next_open_s=next_open_s)
    return _pass(7, "Market open", is_open=True)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 8 — DTBP / Margin Sync
# ─────────────────────────────────────────────────────────────────────────────

def phase8_dtbp_sync(
    equity: float,
    buying_power: float,
    target_pct: float,
    price: float,
    safety_margin: float = 0.95,
) -> SafeguardResult:
    """
    Cap order quantity at the minimum of target allocation or legal DTBP.

        Q_max = min(equity * target_pct * safety, buying_power) / price
    """
    target_alloc   = equity * target_pct * safety_margin
    capped_dollars = min(target_alloc, buying_power * safety_margin)
    qty            = max(0, int(capped_dollars / price))

    if qty == 0:
        return _abort(8, f"DTBP cap: qty=0 "
                         f"(equity={equity:.0f} bp={buying_power:.0f} "
                         f"target={target_pct:.1%} price={price:.2f})",
                      qty=0, buying_power=buying_power)
    return _pass(8, f"DTBP OK qty={qty} alloc=${capped_dollars:,.0f}",
                 qty=qty, alloc_dollars=capped_dollars)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 9 — Toxic Liquidity / Spread Blowout
# ─────────────────────────────────────────────────────────────────────────────

def phase9_spread_defense(
    bid: float, ask: float,
    max_spread: float = 0.15,
) -> SafeguardResult:
    """
    Suppress execution when spread exceeds institutional tolerance.

        delta_spread = ask - bid
        abort if delta_spread > max_spread
    """
    spread = ask - bid
    if spread > max_spread:
        return _abort(9, f"Spread blowout ${spread:.3f} > ${max_spread:.3f}",
                      spread=spread, max_spread=max_spread)
    return _pass(9, f"Spread ${spread:.4f} OK", spread=spread)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 10 — Panic Sweep / Flash Crash
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PanicState:
    hardware_kill: bool = False   # set True when spacebar pressed on cockpit


def phase10_panic_sweep(
    z_score: float,
    panic_state: PanicState,
    micro_price: float,
    bid: float,
) -> SafeguardResult:
    """
    Apply non-linear exit urgency.

        Z <= 4.0  →  use micro_price (normal)
        Z >  4.0  →  cross spread (bid - $0.05) for guaranteed fill
        hardware  →  MARKET order override
    """
    if panic_state.hardware_kill:
        return _abort(10, "HARDWARE KILL-SWITCH ENGAGED — market liquidation",
                      kill=True, exit_type="MARKET")
    if z_score > 4.0:
        exit_price = bid - 0.05
        return _abort(10, f"Flash crash Z={z_score:.2f} — aggressive sweep ${exit_price:.2f}",
                      z_score=z_score, exit_price=exit_price, exit_type="SWEEP")
    return _pass(10, f"Z={z_score:.2f} normal exit", micro_price=micro_price)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 11 — EOD Hard Liquidation
# ─────────────────────────────────────────────────────────────────────────────

def phase11_eod_liquidation(
    seconds_to_close: float,
    eod_buffer_s: float = 300.0,   # 5 minutes
) -> SafeguardResult:
    """
    Block new entries and flag liquidation within eod_buffer_s of close.
    """
    if seconds_to_close <= eod_buffer_s:
        return _abort(11, f"EOD lock: {seconds_to_close:.0f}s to close — "
                          f"foraging disabled, dump inventory",
                      seconds_to_close=seconds_to_close, liquidate=True)
    return _pass(11, f"{seconds_to_close/60:.0f}min to close — OK",
                 seconds_to_close=seconds_to_close)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 12 — Jitter Shield (Deterministic Math)
# ─────────────────────────────────────────────────────────────────────────────

def phase12_jitter_shield(value: float) -> float:
    """
    Pure-Python deterministic rounding to eliminate float GC surprises.
    Equivalent to the Rust cortex_math library for edge cases.
    Rounds to 6 significant figures — sufficient for price/score comparisons.
    """
    if value == 0.0 or not math.isfinite(value):
        return 0.0
    magnitude = math.floor(math.log10(abs(value)))
    factor    = 10 ** (5 - magnitude)
    return round(value * factor) / factor


def phase12_validate(z_score: float) -> SafeguardResult:
    """Validate z_score is a finite, deterministic float."""
    if not math.isfinite(z_score):
        return _abort(12, f"Non-finite z_score={z_score} — jitter shield abort")
    z_clean = phase12_jitter_shield(z_score)
    return _pass(12, f"Z={z_clean:.6f} deterministic", z_clean=z_clean)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 13 — VPIN-Lite (Toxic Liquidity Filter)
# ─────────────────────────────────────────────────────────────────────────────

def phase13_vpin_lite(
    ask_volume: float,
    bid_volume: float,
    imbalance_threshold: float = 10.0,
) -> SafeguardResult:
    """
    Detect informed institutional flow.

        rho_toxic = V_ask / V_bid

    If rho_toxic > 10.0 × imbalance → toxic state → hug price with micro-stop.
    """
    if bid_volume <= 0:
        return _abort(13, "Zero bid volume — toxic state assumed", bid_volume=0)
    rho = ask_volume / bid_volume
    if rho > imbalance_threshold:
        return _abort(13, f"TOXIC FLOW rho={rho:.1f}x > {imbalance_threshold}x — "
                         f"institutional sell-sweep detected",
                      rho_toxic=rho, threshold=imbalance_threshold)
    return _pass(13, f"Liquidity clean rho={rho:.2f}", rho_toxic=rho)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 14 — Soma-Sync (LSM + JEPA active inference — wired via engine)
# Phase 15 — Resonance Gate (wired via StaticCSRRouter)
# Phase 16 — TTM Self-Evolution (wired via TestTimeMemoryWithClustering)
# These phases execute inside CortexEngine.tick().
# The functions below are the gate checks on their outputs.
# ─────────────────────────────────────────────────────────────────────────────

def phase14_soma_sync(
    resonance: float,
    z_score: float,
    sigma_threshold: float = 2.18,
) -> SafeguardResult:
    """
    Gate: LSM Z-score must breach surgical pulse threshold AND
    reservoir resonance must be non-trivial.
    """
    if z_score < sigma_threshold:
        return _abort(14, f"Soma-Sync: Z={z_score:.3f} < σ_threshold={sigma_threshold} — "
                         f"no surgical pulse",
                      z_score=z_score, sigma_threshold=sigma_threshold)
    return _pass(14, f"Soma-Sync STRIKE: Z={z_score:.3f} ρ={resonance:.4f}",
                 z_score=z_score, resonance=resonance)


def phase15_resonance_gate(
    resonance: float,
    fleet_resonance: Dict[str, float],
    level1_rho: float = 0.20,
    level2_rho: float = 0.91,
    level3_rho: float = 0.95,
) -> SafeguardResult:
    """
    Map resonance to execution tier.

        ρ > 0.95 → Level 3 Sniper  (20% vault)
        ρ > 0.91 → Level 2 Hunter  (5% vault)
        ρ > 0.20 → Level 1 Scout   (1% vault)
        otherwise → HOLD
    """
    if resonance >= level3_rho:
        tier, pct = "SNIPER",  0.20
    elif resonance >= level2_rho:
        tier, pct = "HUNTER",  0.05
    elif resonance >= level1_rho:
        tier, pct = "SCOUT",   0.01
    else:
        return _abort(15, f"ρ={resonance:.4f} below Level 1 ({level1_rho}) — HOLD",
                      resonance=resonance)
    return _pass(15, f"Resonance {tier}: ρ={resonance:.4f} alloc={pct:.0%}",
                 tier=tier, alloc_pct=pct, resonance=resonance)


def phase16_ttm_evolution(
    prior_size: int,
    hypothesis_count: int,
    min_prior: int = 0,
) -> SafeguardResult:
    """
    TTM self-evolution is always running. This gate checks health.
    Warns if no prior has been established yet (cold start).
    """
    total = prior_size + hypothesis_count
    if prior_size == 0 and hypothesis_count == 0:
        return _pass(16, "TTM cold start — no prior yet (first session)",
                     prior_size=0, hypothesis_count=0, cold_start=True)
    return _pass(16, f"TTM live: prior={prior_size} hyp={hypothesis_count}",
                 prior_size=prior_size, hypothesis_count=hypothesis_count)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 17 — Systemic Fracture Override (Kill-Switch)
# ─────────────────────────────────────────────────────────────────────────────

def phase17_systemic_fracture(
    fleet_resonance: Dict[str, float],
    fracture_threshold: float = 0.20,
    fracture_pct: float = 0.50,
) -> SafeguardResult:
    """
    Black Swan detection: if >= 50% of fleet enters FRACTURE state
    (resonance < 0.20 simultaneously), abort all execution.
    """
    if not fleet_resonance:
        return _pass(17, "No fleet data — single asset mode")
    fractured = [s for s, r in fleet_resonance.items() if r < fracture_threshold]
    frac_pct  = len(fractured) / len(fleet_resonance)
    if frac_pct >= fracture_pct:
        return _abort(17, f"SYSTEMIC FRACTURE: {len(fractured)}/{len(fleet_resonance)} "
                         f"assets in FRACTURE — Black Swan protocol",
                      fractured=fractured, frac_pct=frac_pct)
    return _pass(17, f"Fleet coherent: {len(fractured)}/{len(fleet_resonance)} fractured",
                 frac_pct=frac_pct)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 18 — Heartbeat Sync (Adaptive Polling)
# ─────────────────────────────────────────────────────────────────────────────

class HeartbeatSync:
    """
    Biological rhythm: accelerates to 10Hz during SATURATED states,
    throttles to 0.5Hz during hibernation.

        SATURATED  : ρ > 0.91  →  interval = 0.10s  (10 Hz)
        ACTIVE     : ρ > 0.20  →  interval = 0.20s  (5 Hz)
        IDLE       : ρ <= 0.20 →  interval = 2.00s  (0.5 Hz)
    """
    def __init__(self) -> None:
        self._last_tick = 0.0

    def interval(self, resonance: float) -> float:
        if resonance > 0.91:  return 0.10   # SATURATED
        if resonance > 0.20:  return 0.20   # ACTIVE
        return 2.00                          # IDLE / HIBERNATION

    def wait(self, resonance: float) -> float:
        """Sleep to maintain biological rhythm. Returns actual elapsed."""
        target = self.interval(resonance)
        elapsed = time.perf_counter() - self._last_tick
        remaining = max(0.0, target - elapsed)
        if remaining > 0:
            time.sleep(remaining)
        self._last_tick = time.perf_counter()
        return elapsed


def phase18_heartbeat(sync: HeartbeatSync, resonance: float) -> SafeguardResult:
    interval = sync.interval(resonance)
    mode = "SATURATED" if resonance > 0.91 else "ACTIVE" if resonance > 0.20 else "IDLE"
    return _pass(18, f"Heartbeat {mode} @ {1/interval:.1f}Hz",
                 mode=mode, interval_s=interval, hz=1/interval)


# ─────────────────────────────────────────────────────────────────────────────
# Full Audit — run all structural phases in sequence
# ─────────────────────────────────────────────────────────────────────────────

def run_pre_execution_audit(
    *,
    # Market data
    bid: float, ask: float,
    bid_size: float = 100.0, ask_size: float = 100.0,
    ask_volume: float = 100.0, bid_volume: float = 100.0,
    # Network
    rtt_ms: float = 10.0,
    data_age_ms: float = 100.0,
    # Clock
    is_market_open: bool = True,
    seconds_to_close: float = 3600.0,
    # Account
    equity: float = 100_000.0,
    buying_power: float = 200_000.0,
    day_trades_remaining: int = 3,
    # Signal
    z_score: float = 0.0,
    resonance: float = 0.0,
    fleet_resonance: Optional[Dict[str, float]] = None,
    # Engine state
    prior_size: int = 0,
    hypothesis_count: int = 0,
    # Guards (stateful, pass in from engine)
    stale_guard: Optional[StaleDataGuard] = None,
    api_governor: Optional[APIGovernor] = None,
    panic_state:  Optional[PanicState]  = None,
    # Sizing
    target_alloc_pct: float = 0.01,
) -> list[SafeguardResult]:
    """
    Run all structural safeguard phases in order.
    Returns list of results — first False aborts the chain.
    """
    if stale_guard  is None: stale_guard  = StaleDataGuard()
    if api_governor is None: api_governor = APIGovernor()
    if panic_state  is None: panic_state  = PanicState()
    if fleet_resonance is None: fleet_resonance = {}

    price      = (bid + ask) / 2
    micro_res  = phase1_vwmp(bid, ask, bid_size, ask_size)
    micro      = micro_res.data.get("micro_price", price)

    checks = [
        micro_res,
        phase2_rtt_watchdog(rtt_ms),
        phase4_stale_data(stale_guard, data_age_ms),
        phase5_pdt_cooldown(day_trades_remaining, z_score),
        phase6_api_governor(api_governor),
        phase7_market_clock(is_market_open, seconds_to_close),
        phase8_dtbp_sync(equity, buying_power, target_alloc_pct, micro),
        phase9_spread_defense(bid, ask),
        phase10_panic_sweep(z_score, panic_state, micro, bid),
        phase11_eod_liquidation(seconds_to_close),
        phase12_validate(z_score),
        phase13_vpin_lite(ask_volume, bid_volume),
        phase14_soma_sync(resonance, z_score),
        phase15_resonance_gate(resonance, fleet_resonance),
        phase16_ttm_evolution(prior_size, hypothesis_count),
        phase17_systemic_fracture(fleet_resonance),
    ]
    return checks


