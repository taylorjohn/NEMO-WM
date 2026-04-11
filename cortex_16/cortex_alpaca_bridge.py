"""
cortex_alpaca_bridge.py — CORTEX-16 Action → Paper Trading Bridge

Signal architecture (Option C — momentum-gated):
    Direction:  5-bar price momentum from live Alpaca bars
                buy  when momentum > +MOM_THRESH (+0.05%)
                sell when momentum < -MOM_THRESH (-0.05%)
                flat when |momentum| < MOM_THRESH → no trade

    Confidence: encoder action magnitude ||action|| from UDP
                acts as a filter — only trade when the encoder
                confirms directional conviction
                σ=0: ||action|| < 0.05 → skip (low conviction)
                σ=1: 0.05–0.09 → $100
                σ=2: 0.09–0.12 → $400
                σ=3: > 0.12    → $1000

    Exit:       momentum reversal OR strong negative action[0]
                Close position when 3-bar momentum turns negative
                (after being in a long) or action[0] < -EXIT_THRESH

This separates concerns cleanly:
    - Price momentum provides the directional signal (real edge)
    - Encoder magnitude provides the conviction filter (avoids low-confidence trades)
    - Neuropixels rho modulates in the planner (biological arousal)

13-Phase safeguards run as before on every entry attempt.

Usage:
    python cortex_alpaca_bridge.py --paper
"""

import argparse
import os
import socket
import struct
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np
from dotenv import load_dotenv

# cortex_brain neuromodulation (additive)
try:
    from cortex_brain.neuro.dopamine import DopamineSystem, DopamineConfig
    from cortex_brain.memory.ttm_clustering import TestTimeMemoryWithClustering, TTMConfig
    _CORTEX_BRAIN = True
except ImportError:
    _CORTEX_BRAIN = False
    print("cortex_brain not available -- DA disabled")


load_dotenv()
_TIMEOUT_PATCHED = True

# Alpaca timeout config
ALPACA_TIMEOUT_S    = 3.0    # per-request timeout
ALPACA_MAX_RETRIES  = 3      # exponential backoff retries
ALPACA_BACKOFF_BASE = 1.5    # backoff multiplier



# =============================================================================
# Configuration
# =============================================================================
VAULT_BASELINE   = 500.0
LOOP_BUDGET_MS   = 2000.0
MAX_RTT_MS       = 45.0
MAX_DATA_AGE_S   = 5.0
MAX_SPREAD       = 0.15
OBI_TOXIC_RATIO  = 10.0
FLASH_CRASH_PCT  = 0.02
EOD_HOUR         = 15
EOD_MINUTE       = 45
MARKET_OPEN_H    = 9
MARKET_OPEN_M    = 35
COOLDOWN_TICKS   = 5

# Encoder conviction thresholds (unchanged)
SIGMA_L1_THRESH  = 0.05
SIGMA_L2_THRESH  = 0.09
SIGMA_L3_THRESH  = 0.12

# Momentum thresholds — price signal for direction
MOM_THRESH       = 0.0005   # 0.05% 5-bar momentum required to enter
MOM_BARS         = 5        # bars to look back for momentum
MOM_EXIT_BARS    = 3        # bars for exit momentum check

# Exit thresholds
EXIT_MOM_THRESH  = -0.0002  # 3-bar momentum reversal exit
EXIT_ACT_THRESH  = 0.08     # action[0] < -0.08 exit

PDT_ALPHA = {0: float("inf"), 1: 1.50, 2: 1.25, 3: 1.00}


# =============================================================================
# Alpaca API
# =============================================================================
def get_api(paper: bool = True):
    from alpaca_trade_api.rest import REST
    if paper:
        return REST(os.getenv("ALPACA_PAPER_KEY"),
                    os.getenv("ALPACA_PAPER_SECRET"),
                    "https://paper-api.alpaca.markets")
    return REST(os.getenv("ALPACA_LIVE_KEY"),
                os.getenv("ALPACA_LIVE_SECRET"),
                "https://api.alpaca.markets")


# =============================================================================
# Live momentum from Alpaca
# =============================================================================
class MomentumSignal:
    """
    Fetches recent 1-min bars from Alpaca and computes momentum.
    Caches bars to avoid excess API calls — refreshes every tick.

    Returns:
        momentum: float  (5-bar return)
        side:     str    "buy" | "sell" | "flat"
        bars:     DataFrame for P10 flash crash check
    """
    def __init__(self, api, symbol: str, lookback: int = 20):
        self.api      = api
        self.symbol   = symbol
        self.lookback = lookback
        self.bars     = None

    def update(self):
        """Fetch latest bars with retry + timeout."""
        from alpaca_trade_api.rest import TimeFrame
        import socket
        delay = 0.5
        for attempt in range(ALPACA_MAX_RETRIES):
            try:
                # requests timeout via socket default
                old_timeout = socket.getdefaulttimeout()
                socket.setdefaulttimeout(ALPACA_TIMEOUT_S)
                raw = self.api.get_bars(
                    self.symbol, TimeFrame.Minute,
                    limit=self.lookback, feed="iex",
                ).df
                socket.setdefaulttimeout(old_timeout)
                raw.columns = [c.lower() for c in raw.columns]
                self.bars   = raw.reset_index(drop=True)
                self._consecutive_failures = 0
                return  # success
            except Exception as e:
                socket.setdefaulttimeout(None)
                self._consecutive_failures = getattr(
                    self, "_consecutive_failures", 0) + 1
                if attempt < ALPACA_MAX_RETRIES - 1:
                    time.sleep(delay)
                    delay *= ALPACA_BACKOFF_BASE
                else:
                    # All retries exhausted -- keep stale bars
                    if self._consecutive_failures % 10 == 1:
                        print(f"   [WARN] Alpaca bars timeout "
                              f"(attempt {self._consecutive_failures}): {e}")

    def get_momentum(self, n_bars: int = MOM_BARS) -> float:
        """Returns n-bar price momentum. 0.0 if bars unavailable."""
        if self.bars is None or len(self.bars) < n_bars:
            return 0.0
        closes = self.bars["close"].values
        return float((closes[-1] - closes[-n_bars]) / closes[-n_bars])

    def get_signal(self) -> tuple:
        """
        Returns (momentum, side, bars_df).
        side: "buy" | "sell" | "flat"
        """
        mom = self.get_momentum(MOM_BARS)
        if mom > MOM_THRESH:
            side = "buy"
        elif mom < -MOM_THRESH:
            side = "sell"
        else:
            side = "flat"
        return mom, side, self.bars

    def should_exit_long(self, action: np.ndarray) -> tuple:
        """
        Returns (should_exit, reason).
        Exit long when momentum reverses OR action is strongly negative.
        """
        # Action-based exit
        if float(action[0]) < -EXIT_ACT_THRESH:
            return True, f"action[0]={action[0]:.3f} < -{EXIT_ACT_THRESH}"

        # Momentum reversal exit
        mom_exit = self.get_momentum(MOM_EXIT_BARS)
        if mom_exit < EXIT_MOM_THRESH:
            return True, f"momentum reversal {mom_exit*100:+.3f}%"

        return False, ""


# =============================================================================
# UDP Action Receiver
# =============================================================================
class ActionReceiver:
    STRUCT = struct.Struct("<d 2f 128f")

    def __init__(self, port: int = 5005):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", port))
        self.sock.settimeout(2.0)
        self.last_timestamp = 0.0
        self.last_action    = np.zeros(2, dtype=np.float32)
        self.last_latent    = np.zeros(128, dtype=np.float32)
        self._lock          = threading.Lock()
        threading.Thread(target=self._listen, daemon=True).start()
        print(f"✅ ActionReceiver listening on UDP port {port}")

    def _listen(self):
        while True:
            try:
                data, _ = self.sock.recvfrom(528)
                if len(data) == self.STRUCT.size:
                    u = self.STRUCT.unpack(data)
                    with self._lock:
                        self.last_timestamp = u[0]
                        self.last_action    = np.array(u[1:3], dtype=np.float32)
                        self.last_latent    = np.array(u[3:],  dtype=np.float32)
            except socket.timeout:
                pass
            except Exception:
                pass

    def get(self):
        with self._lock:
            return (self.last_timestamp,
                    self.last_action.copy(),
                    self.last_latent.copy())


# =============================================================================
# Signal decoder — magnitude only (direction comes from momentum)
# =============================================================================
def decode_magnitude(action: np.ndarray) -> dict:
    """
    Decodes action magnitude to sigma level and notional.
    Direction is ignored here — momentum provides direction.
    """
    magnitude = float(np.linalg.norm(action))

    if magnitude >= SIGMA_L3_THRESH:
        sigma_level, notional = 3, 1000.0
    elif magnitude >= SIGMA_L2_THRESH:
        sigma_level, notional = 2, 400.0
    elif magnitude >= SIGMA_L1_THRESH:
        sigma_level, notional = 1, 100.0
    else:
        sigma_level, notional = 0, 0.0

    return {
        "magnitude":   magnitude,
        "sigma_level": sigma_level,
        "notional":    notional,
    }


# =============================================================================
# VWMP
# =============================================================================
def get_vwmp(api, symbol: str) -> tuple:
    try:
        snap  = api.get_snapshot(symbol)
        bid_p = snap.latest_quote.bp
        ask_p = snap.latest_quote.ap
        bid_s = snap.latest_quote.bs or 1
        ask_s = snap.latest_quote.as_ or 1
        micro = (ask_s * bid_p + bid_s * ask_p) / (bid_s + ask_s)
        return float(micro), float(bid_p), float(ask_p)
    except Exception:
        try:
            p = float(api.get_latest_trade(symbol).price)
            return p, p, p
        except Exception:
            return 0.0, 0.0, 0.0


def get_position_qty(api, symbol: str) -> float:
    try:
        for p in api.list_positions():
            if p.symbol == symbol:
                return float(p.qty)
    except Exception:
        pass
    return 0.0


# =============================================================================
# 13-Phase Safeguard Stack
# =============================================================================
class SafeguardStack:
    def __init__(self, api, symbol: str):
        self.api          = api
        self.symbol       = symbol
        self.price_buffer = deque(maxlen=60)
        self.last_api_t   = 0.0

    def p7_market_open(self) -> bool:
        now = datetime.now()
        h, m = now.hour, now.minute
        if h < MARKET_OPEN_H or (h == MARKET_OPEN_H and m < MARKET_OPEN_M):
            return False
        if h > EOD_HOUR or (h == EOD_HOUR and m >= EOD_MINUTE):
            return False
        return True

    def p4_data_fresh(self, last_ts: float) -> bool:
        return (time.time() - last_ts) < MAX_DATA_AGE_S

    def p2_rtt_ok(self) -> bool:
        try:
            t0 = time.perf_counter()
            self.api.get_clock()
            return (time.perf_counter() - t0) * 1000 < MAX_RTT_MS
        except Exception:
            return False

    def p6_rate_limit(self) -> bool:
        if time.time() - self.last_api_t < 1.0:
            return False
        self.last_api_t = time.time()
        return True

    def p5_dynamic_alpha(self, sigma_level: int) -> tuple:
        try:
            account    = self.api.get_account()
            equity     = float(account.equity)
            if equity >= 25000 and not account.pattern_day_trader:
                return True, "1.00"
            day_trades  = int(account.daytrade_count)
            trades_left = max(0, 3 - day_trades)
            multiplier  = PDT_ALPHA.get(min(trades_left, 3), 1.0)
            if multiplier == float("inf"):
                return False, "inf"
            return sigma_level * multiplier >= 1.0, f"{multiplier:.2f}"
        except Exception:
            return True, "1.00"

    def p8_buying_power(self, notional: float) -> bool:
        try:
            account   = self.api.get_account()
            dtbp      = float(account.daytrading_buying_power)
            bp        = float(account.buying_power)
            effective = dtbp if dtbp > 0 else bp
            return effective >= notional * 1.1
        except Exception:
            return False

    def p9_spread_ok(self, bid: float, ask: float) -> bool:
        return bid > 0 and ask > 0 and (ask - bid) <= MAX_SPREAD

    def p13_obi_ok(self) -> bool:
        try:
            snap  = self.api.get_snapshot(self.symbol)
            bid_s = snap.latest_quote.bs or 1
            ask_s = snap.latest_quote.as_ or 1
            return (ask_s / bid_s) < OBI_TOXIC_RATIO
        except Exception:
            return True

    def p10_flash_crash(self, price: float) -> bool:
        self.price_buffer.append(price)
        if len(self.price_buffer) < 2:
            return True
        ref  = self.price_buffer[-min(60, len(self.price_buffer))]
        return abs(price - ref) / ref < FLASH_CRASH_PCT

    def p11_eod_check(self) -> bool:
        now = datetime.now()
        if now.hour == EOD_HOUR and now.minute >= EOD_MINUTE:
            try:
                for pos in self.api.list_positions():
                    self.api.submit_order(
                        symbol=pos.symbol, qty=abs(int(float(pos.qty))),
                        side="sell", type="market", time_in_force="ioc"
                    )
                    print(f"⚡ P11 EOD: Liquidated {pos.symbol}")
            except Exception as e:
                print(f"⚠️  P11 EOD error: {e}")
            return False
        return True

    def run_all(self, signal: dict, last_ts: float,
                mom_side: str = "flat") -> tuple:
        if not self.p7_market_open():
            return False, "P7: market closed"
        if not self.p11_eod_check():
            return False, "P11: EOD"
        if not self.p4_data_fresh(last_ts):
            return False, "P4: stale encoder data"
        if not self.p6_rate_limit():
            return False, "P6: rate limit"
        if signal["sigma_level"] == 0:
            return False, "σ=0: low conviction"
        if mom_side == "flat":
            return False, "momentum flat: no direction"
        if mom_side == "sell":
            return False, "momentum bearish: no long entry"
        if not self.p2_rtt_ok():
            return False, "P2: RTT > 45ms"

        ok, mult = self.p5_dynamic_alpha(signal["sigma_level"])
        if not ok:
            label = "PDT limit" if mult == "inf" else f"PDT alpha ×{mult}"
            return False, f"P5: {label}"
        if not self.p8_buying_power(signal["notional"]):
            return False, f"P8: insufficient buying power"

        micro_p, bid, ask = get_vwmp(self.api, self.symbol)
        if micro_p <= 0:
            return False, "P1: VWMP unavailable"
        if not self.p9_spread_ok(bid, ask):
            return False, f"P9: spread ${ask-bid:.3f}"
        if not self.p13_obi_ok():
            return False, "P13: toxic OBI"
        if not self.p10_flash_crash(micro_p):
            return False, "P10: flash crash"

        signal["micro_price"] = micro_p
        signal["side"]        = "buy"   # only buy on confirmed momentum
        return True, "all phases passed"


# =============================================================================
# Order execution
# =============================================================================
def execute_order(api, symbol: str, signal: dict) -> tuple:
    side     = signal["side"]
    notional = signal["notional"]

    pos_qty = get_position_qty(api, symbol)
    if pos_qty > 0.001 and side == "buy":
        print(f"   ⏭️  Already holding {symbol} — skip buy")
        return False, 0.0
    if pos_qty <= 0.001 and side == "sell":
        print(f"   ⏭️  No position to sell")
        return False, 0.0

    try:
        order = api.submit_order(
            symbol=symbol, notional=notional,
            side=side, type="market", time_in_force="day",
        )
        print(f"   📋 Order: {side.upper()} ${notional:.0f} {symbol} (market)")
        time.sleep(1.5)
        status = api.get_order(order.id)
        if status.status in ("filled", "partially_filled"):
            qty   = float(status.filled_qty or 0)
            price = float(status.filled_avg_price or 0)
            val   = qty * price
            print(f"   ✅ Filled: {qty:.4f} shares @ ${price:.2f} (${val:.2f})")
            return True, val
        else:
            print(f"   ❌ Status: {status.status}")
            return False, 0.0
    except Exception as e:
        print(f"   🚨 Order error: {e}")
        return False, 0.0


def execute_exit(api, symbol: str) -> bool:
    """Market sell entire position using close_position to avoid qty rounding errors."""
    pos_qty = get_position_qty(api, symbol)
    if pos_qty <= 0.001:
        return False
    try:
        # Use close_position to avoid fractional qty rounding errors
        api.close_position(symbol)
        print(f"   🔴 EXIT: Closed {pos_qty:.6f} shares {symbol}")
        return True
    except Exception as e:
        print(f"   🚨 Exit error: {e}")
        return False


def vault_sweep(vault: float, realised_pnl: float) -> float:
    if realised_pnl > 0:
        sweep  = realised_pnl * 0.50
        vault += sweep
        print(f"   💰 Vault sweep: +${sweep:.2f} (vault=${vault:.2f})")
    return vault


# =============================================================================
# Main bridge loop
# =============================================================================
def run_bridge(symbol: str = "SPY", paper: bool = True):
    print("\n" + "="*60)
    print("  CORTEX-16 ALPACA BRIDGE  (momentum-gated)")
    print(f"  Symbol:     {symbol}")
    print(f"  Mode:       {'PAPER' if paper else '⚡ LIVE'}")
    print(f"  Direction:  5-bar price momentum (thresh={MOM_THRESH*100:.2f}%)")
    print(f"  Filter:     encoder ||action|| conviction")
    print(f"  Exit:       {MOM_EXIT_BARS}-bar reversal OR action[0]<-{EXIT_ACT_THRESH}")
    print(f"  Cooldown:   {COOLDOWN_TICKS} ticks")
    print("="*60 + "\n")

    api       = get_api(paper)
    receiver  = ActionReceiver(port=5005)
    guards    = SafeguardStack(api, symbol)
    momentum  = MomentumSignal(api, symbol)

    vault           = VAULT_BASELINE
    session_pnl     = 0.0

    # cortex_brain: DA + TTM session memory
    da_system = DopamineSystem() if _CORTEX_BRAIN else None
    ttm       = TestTimeMemoryWithClustering() if _CORTEX_BRAIN else None
    _last_pnl = 0.0

    entry_price     = 0.0
    last_entry_tick = -COOLDOWN_TICKS
    in_position     = False

    try:
        clock = api.get_clock()
        print(f"✅ Alpaca connected ({'paper' if paper else 'LIVE'}) | "
              f"Market {'open' if clock.is_open else 'closed'}")
        pos_qty     = get_position_qty(api, symbol)
        in_position = pos_qty > 0.001
        if in_position:
            print(f"   📌 Existing position: {pos_qty:.4f} shares {symbol}")
    except Exception as e:
        print(f"❌ Alpaca connection failed: {e}")
        return

    print("\n🟢 Bridge active. Waiting for actions...\n")
    tick = 0

    while True:
        t0 = time.perf_counter()

        # Fetch latest bars and compute momentum
        momentum.update()
        mom_val, mom_side, bars_df = momentum.get_signal()

        # Receive encoder action
        last_ts, action, latent = receiver.get()

        # cortex_brain: update DA from last tick PnL
        if da_system is not None:
            da_system.update(pnl=_last_pnl, resonance=0.5, temp_norm=0.3)
            da_status = da_system.status()
            # DA modulates conviction threshold:
            # high DA (positive RPE) -> sharper exploitation
            _da_boost = da_status["da"] - 0.5  # above tonic
            _sigma_thresh_adj = -_da_boost * 0.01  # lower threshold when DA high
        else:
            da_status = {}
            _sigma_thresh_adj = 0.0

        if ttm is not None and latent is not None:
            _moe_w = np.ones(4, dtype=np.float32) / 4
            ttm.observe(latent, _moe_w, resonance=0.5, pnl=_last_pnl)
            _ttm_adj = ttm.get_routing_adjustment(latent)
        else:
            _ttm_adj = np.zeros(4, dtype=np.float32)

        signal = decode_magnitude(action)
        mag    = signal["magnitude"]
        lvl    = signal["sigma_level"]

        # --- Exit check ---
        if in_position:
            should_exit, exit_reason = momentum.should_exit_long(action)
            if should_exit:
                exited = execute_exit(api, symbol)
                if exited:
                    in_position     = False
                    last_entry_tick = tick
                    print(f"   Reason: {exit_reason}")

        # --- Entry check ---
        proceed, reason = guards.run_all(signal, last_ts, mom_side)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        if elapsed_ms > LOOP_BUDGET_MS:
            tick += 1
            continue

        _stale = getattr(momentum, "_consecutive_failures", 0)
        if tick % 10 == 0 or proceed:
            _stale_str = f" [STALE x{_stale}]" if _stale > 0 else ""
            mom_str = f"mom={mom_val*100:+.3f}%({mom_side}){_stale_str}"
            pos_str = f"pos={'Y' if in_position else 'N'}"
            status  = "🚀 ENTRY" if proceed else f"⏸  {reason}"
            da_str = f"DA={da_status.get('da',0):.3f} CRT={da_status.get('cortisol',0):.3f}" if da_status else ""
            print(f"Tick {tick:>5} | mag={mag:.4f} | "
                  f"{mom_str:<22} | {pos_str} | {da_str} | {status}")

        if proceed and not in_position:
            if tick - last_entry_tick < COOLDOWN_TICKS:
                remaining = COOLDOWN_TICKS - (tick - last_entry_tick)
                print(f"   ⏸  Cooldown ({remaining} ticks remaining)")
            else:
                filled, fill_notional = execute_order(api, symbol, signal)
                if filled:
                    in_position     = True
                    last_entry_tick = tick
                    _last_pnl       = fill_notional * 0.001  # proxy RPE
                    entry_price     = fill_notional / signal["notional"] * \
                                      (signal["notional"] / fill_notional) \
                                      if fill_notional > 0 else 0
                    session_pnl    += fill_notional

        tick += 1
        time.sleep(max(0.0, 1.0 - elapsed_ms / 1000.0))


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CORTEX-16 Alpaca Bridge")
    p.add_argument("--symbol", default="SPY")
    p.add_argument("--paper",  action="store_true", default=True)
    p.add_argument("--live",   action="store_true")
    args = p.parse_args()
    if args.live:
        if input("⚡ LIVE MODE — type CONFIRM: ") != "CONFIRM":
            print("Aborted."); exit()
    run_bridge(symbol=args.symbol, paper=not args.live)
