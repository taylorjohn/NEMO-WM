"""
test_run_trading.py
===================
Unit + integration tests for run_trading.py components.

Tests are self-contained — no Alpaca API keys, no network, no file I/O required.
All external dependencies (Alpaca, HUD socket, CortexEngine) are mocked.

Run:
    python -m pytest test_run_trading.py -v
    python -m pytest test_run_trading.py -v -k "vault"
    python -m pytest test_run_trading.py -v --tb=short
"""
import csv
import json
import math
import socket
import tempfile
import threading
import time
import types
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

# ── Import the module under test ──────────────────────────────────────────────
# run_trading.py uses argparse at module level so we patch sys.argv first.
import sys
sys.argv = ["run_trading.py", "--sim", "--ticks", "5"]

import run_trading as rt


# ═════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def vault():
    return rt.VaultState(seed=100_000.0)


@pytest.fixture
def tmp_csv(tmp_path):
    return str(tmp_path / "test_combat.csv")


@pytest.fixture
def logger(tmp_csv):
    return rt.CombatLogger(tmp_csv)


def _make_actuator(vault=None, logger=None, tmp_path=None, **kwargs):
    """Build a fully mocked InstitutionalActuator for testing."""
    if vault is None:
        vault = rt.VaultState(100_000.0)
    if logger is None:
        p = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        logger = rt.CombatLogger(p.name)
    return rt.InstitutionalActuator(
        symbol="SPY",
        vault=vault,
        logger=logger,
        api=None,
        sim=True,
        fleet=["SPY", "QQQ"],
        hud_sock=None,
        hud_addr=None,
        panic=rt.PanicState(),
        **kwargs,
    )


@pytest.fixture
def actuator(vault, logger):
    return _make_actuator(vault=vault, logger=logger)


# ═════════════════════════════════════════════════════════════════════════════
# VAULT STATE
# ═════════════════════════════════════════════════════════════════════════════

class TestVaultState:
    def test_initial_state(self, vault):
        assert vault.total_equity == 100_000.0
        assert vault.active_pool  == 100_000.0
        assert vault.banked       == 0.0
        assert vault.seed         == 100_000.0

    def test_profit_splits_50_50(self, vault):
        vault.record_pnl(200.0)
        assert vault.banked       == pytest.approx(100.0)
        assert vault.active_pool  == pytest.approx(100_100.0)
        assert vault.total_equity == pytest.approx(100_200.0)

    def test_loss_comes_from_active_pool_only(self, vault):
        vault.record_pnl(200.0)   # bank $100
        vault.record_pnl(-50.0)
        assert vault.banked       == pytest.approx(100.0)   # banked untouched
        assert vault.active_pool  == pytest.approx(100_050.0)

    def test_session_pnl_pct_positive(self, vault):
        vault.record_pnl(1_000.0)
        pct = vault.session_pnl_pct()
        assert pct == pytest.approx(1.0)

    def test_session_pnl_pct_negative(self, vault):
        vault.record_pnl(-2_000.0)
        pct = vault.session_pnl_pct()
        assert pct == pytest.approx(-2.0)

    def test_alloc_dollars_respects_active_pool(self, vault):
        vault.record_pnl(-90_000.0)   # drain active pool
        alloc = vault.alloc_dollars(0.20)
        # active_pool = 10_000, 20% = 2_000
        assert alloc == pytest.approx(2_000.0)

    def test_alloc_dollars_never_negative(self, vault):
        vault.record_pnl(-200_000.0)  # overdraw
        assert vault.alloc_dollars(0.20) == 0.0

    def test_status_dict_keys(self, vault):
        s = vault.status()
        assert set(s.keys()) >= {"total_equity","active_pool","banked",
                                  "gain_pct","session_pnl_pct"}

    def test_thread_safety(self, vault):
        """Concurrent record_pnl calls must not corrupt state."""
        errors = []
        def worker():
            try:
                for _ in range(50):
                    vault.record_pnl(1.0)
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert errors == []
        assert vault.total_equity == pytest.approx(100_000.0 + 400.0)


# ═════════════════════════════════════════════════════════════════════════════
# COMBAT LOGGER
# ═════════════════════════════════════════════════════════════════════════════

class TestCombatLogger:
    def test_creates_csv_with_header(self, tmp_csv):
        rt.CombatLogger(tmp_csv)
        with open(tmp_csv) as f:
            header = f.readline().strip().split(",")
        assert "timestamp" in header
        assert "symbol"    in header
        assert "action"    in header

    def test_log_writes_row(self, logger, tmp_csv):
        logger.log(symbol="SPY", action="BUY", qty=10, price=500.0,
                   resonance=0.85, z_score=3.0, tier="HUNTER",
                   alloc_pct=0.05, realised_pnl=0.0, total_equity=100_000.0)
        rows = list(csv.DictReader(open(tmp_csv)))
        assert len(rows) == 1
        assert rows[0]["symbol"] == "SPY"
        assert rows[0]["action"] == "BUY"

    def test_log_fills_timestamp(self, logger, tmp_csv):
        logger.log(symbol="SPY", action="ABORT")
        rows = list(csv.DictReader(open(tmp_csv)))
        assert rows[0]["timestamp"] != ""

    def test_log_thread_safe(self, logger, tmp_csv):
        def worker():
            for _ in range(20):
                logger.log(symbol="SPY", action="HOLD")
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()
        rows = list(csv.DictReader(open(tmp_csv)))
        assert len(rows) == 100


# ═════════════════════════════════════════════════════════════════════════════
# SNIPER WINDOW
# ═════════════════════════════════════════════════════════════════════════════

class TestSniperWindow:
    def test_requires_full_streak(self):
        sw = rt.SniperWindow(threshold=0.95, window=3)
        assert sw.update(0.96) == False
        assert sw.update(0.97) == False
        assert sw.update(0.98) == True    # third consecutive

    def test_resets_on_dip(self):
        sw = rt.SniperWindow(threshold=0.95, window=3)
        sw.update(0.96); sw.update(0.97)  # streak=2
        sw.update(0.80)                    # dip resets
        assert sw._streak == 0
        assert sw.update(0.96) == False   # back to 1

    def test_stays_confirmed_above_threshold(self):
        sw = rt.SniperWindow(threshold=0.95, window=2)
        sw.update(0.96); sw.update(0.97)  # confirmed
        assert sw.update(0.99) == True    # stays confirmed

    def test_window_1_fires_immediately(self):
        sw = rt.SniperWindow(threshold=0.95, window=1)
        assert sw.update(0.96) == True

    def test_reset_method(self):
        sw = rt.SniperWindow(threshold=0.95, window=3)
        sw.update(0.96); sw.update(0.97); sw.update(0.98)
        sw.reset()
        assert sw._streak == 0
        assert sw.update(0.99) == False


# ═════════════════════════════════════════════════════════════════════════════
# DAILY CIRCUIT BREAKER
# ═════════════════════════════════════════════════════════════════════════════

class TestDailyCircuitBreaker:
    def test_does_not_trip_within_limit(self):
        cb = rt.DailyCircuitBreaker(max_loss_pct=2.0)
        assert cb.check(-1.99) == False
        assert cb.tripped       == False

    def test_trips_at_limit(self):
        cb = rt.DailyCircuitBreaker(max_loss_pct=2.0)
        assert cb.check(-2.0) == True
        assert cb.tripped      == True

    def test_trips_below_limit(self):
        cb = rt.DailyCircuitBreaker(max_loss_pct=2.0)
        cb.check(-5.0)
        assert cb.tripped == True

    def test_stays_tripped_after_recovery(self):
        cb = rt.DailyCircuitBreaker(max_loss_pct=2.0)
        cb.check(-3.0)       # trip
        assert cb.check(+10.0) == True   # still tripped even if pnl recovers

    def test_positive_pnl_never_trips(self):
        cb = rt.DailyCircuitBreaker(max_loss_pct=2.0)
        assert cb.check(+5.0) == False

    def test_zero_pnl_does_not_trip(self):
        cb = rt.DailyCircuitBreaker(max_loss_pct=2.0)
        assert cb.check(0.0) == False


# ═════════════════════════════════════════════════════════════════════════════
# MARKET HOURS GUARD
# ═════════════════════════════════════════════════════════════════════════════

class TestMarketHoursGuard:
    def test_open_with_time_remaining(self):
        g = rt.MarketHoursGuard()
        ok, reason = g.is_safe(is_open=True, secs_to_close=3600.0)
        assert ok     == True
        assert reason == "OK"

    def test_closed_market(self):
        g = rt.MarketHoursGuard()
        ok, reason = g.is_safe(is_open=False, secs_to_close=3600.0)
        assert ok == False
        assert "closed" in reason.lower()

    def test_eod_buffer_blocks_entry(self):
        g = rt.MarketHoursGuard()
        ok, reason = g.is_safe(is_open=True, secs_to_close=299.0)
        assert ok == False
        assert "EOD" in reason or "close" in reason.lower()

    def test_exactly_at_buffer_is_safe(self):
        g = rt.MarketHoursGuard()
        ok, _ = g.is_safe(is_open=True, secs_to_close=300.0)
        assert ok == True

    def test_one_second_before_buffer_blocked(self):
        g = rt.MarketHoursGuard()
        ok, _ = g.is_safe(is_open=True, secs_to_close=299.9)
        assert ok == False


# ═════════════════════════════════════════════════════════════════════════════
# MARKET ENCODER
# ═════════════════════════════════════════════════════════════════════════════

class TestMarketEncoder:
    def test_sim_returns_256d_float32(self):
        enc = rt.MarketEncoder(sim=True)
        feat = enc.encode(None)
        assert feat.shape  == (256,)
        assert feat.dtype  == np.float32

    def test_none_snap_returns_256d(self):
        enc = rt.MarketEncoder(sim=False)
        feat = enc.encode(None)
        assert feat.shape == (256,)

    def test_real_snap_returns_256d(self):
        enc = rt.MarketEncoder(sim=False)
        snap = MagicMock()
        snap.latest_trade.price = 500.0
        snap.latest_quote.bp    = 499.95
        snap.latest_quote.ap    = 500.05
        snap.daily_bar.v        = 1_000_000
        snap.daily_bar.o        = 498.0
        snap.daily_bar.h        = 502.0
        snap.daily_bar.l        = 497.0
        feat = enc.encode(snap)
        assert feat.shape == (256,)
        assert feat.dtype == np.float32

    def test_bad_snap_falls_back_gracefully(self):
        enc = rt.MarketEncoder(sim=False)
        bad_snap = MagicMock()
        bad_snap.latest_trade.price = "not_a_float"
        feat = enc.encode(bad_snap)
        assert feat.shape == (256,)

    def test_output_bounded(self):
        enc = rt.MarketEncoder(sim=False)
        snap = MagicMock()
        snap.latest_trade.price = 500.0
        snap.latest_quote.bp = 499.95; snap.latest_quote.ap = 500.05
        snap.daily_bar.v = 1_000_000; snap.daily_bar.o = 498.0
        snap.daily_bar.h = 502.0; snap.daily_bar.l = 497.0
        feat = enc.encode(snap)
        # tanh output bounded in (-1, 1)
        assert feat.max() <= 1.0
        assert feat.min() >= -1.0


# ═════════════════════════════════════════════════════════════════════════════
# INSTITUTIONAL ACTUATOR — GATE LOGIC
# ═════════════════════════════════════════════════════════════════════════════

class TestActuatorGates:
    """Test that each safety gate returns 0.0 and does NOT trade."""

    def _act(self, actuator, resonance=0.85, signal=0.1,
              is_open=True, secs=3600.0, session_pnl_pct=0.0):
        action = np.array([signal, 0.0], dtype=np.float32)
        actuator.update_market_context(
            snap=None, rtt_ms=5.0, data_age_ms=50.0,
            is_open=is_open, secs=secs,
            equity=100_000.0, bp=200_000.0, dt=3,
        )
        # Adjust vault to produce desired session_pnl_pct
        if session_pnl_pct != 0.0:
            actuator.vault.record_pnl(
                actuator.vault.seed * session_pnl_pct / 100.0
            )
        return actuator.act(action, resonance, {})

    def test_gate_a_circuit_breaker_halts(self, actuator):
        """Circuit breaker trips at -2% and returns 0 for all subsequent ticks."""
        pnl = self._act(actuator, resonance=0.85, session_pnl_pct=-3.0)
        assert actuator._breaker.tripped == True
        # Subsequent ticks also return 0
        assert self._act(actuator, resonance=0.99) == 0.0

    def test_gate_b_market_closed_blocks(self):
        a = _make_actuator()
        pnl = _make_actuator()
        act = _make_actuator()
        action = np.array([0.5, 0.0], dtype=np.float32)
        act.update_market_context(None, 5.0, 50.0, False, 3600.0,
                                  100_000.0, 200_000.0, 3)
        result = act.act(action, 0.99, {})
        assert result == 0.0

    def test_gate_b_eod_buffer_blocks(self):
        act = _make_actuator()
        action = np.array([0.5, 0.0], dtype=np.float32)
        act.update_market_context(None, 5.0, 50.0, True, 200.0,
                                  100_000.0, 200_000.0, 3)
        result = act.act(action, 0.99, {})
        assert result == 0.0

    def test_gate_c_sniper_demoted_without_streak(self):
        """rho=0.98 normally = SNIPER but without streak → Hunter tier used."""
        act = _make_actuator(sniper_window=3)
        action = np.array([0.5, 0.0], dtype=np.float32)
        act.update_market_context(None, 5.0, 50.0, True, 3600.0,
                                  100_000.0, 200_000.0, 3)
        # Single tick at high rho — streak=1, not confirmed
        act._fleet_rho = {}   # no fleet → no fracture
        # Just verify sniper streak is not confirmed after one tick
        act.act(action, 0.98, {})
        assert act._sniper._streak == 1
        assert act._sniper.update(0.98) == False   # still only 2 after this

    def test_gate_c_sniper_fires_after_full_streak(self):
        """After window=2 ticks above threshold, SNIPER should be confirmed."""
        sw = rt.SniperWindow(threshold=0.95, window=2)
        sw.update(0.96)
        confirmed = sw.update(0.97)
        assert confirmed == True

    def test_no_trade_on_hold_signal(self):
        act = _make_actuator()
        action = np.array([0.0, 0.0], dtype=np.float32)   # signal=0 → HOLD
        act.update_market_context(None, 5.0, 50.0, True, 3600.0,
                                  100_000.0, 200_000.0, 3)
        result = act.act(action, 0.50, {})
        assert result == 0.0

    def test_ticks_counter_increments(self, actuator):
        action = np.array([0.0, 0.0], dtype=np.float32)
        actuator.update_market_context(None, 5.0, 50.0, True, 3600.0,
                                       100_000.0, 200_000.0, 3)
        for i in range(5):
            actuator.act(action, 0.5, {})
        assert actuator._ticks == 5


# ═════════════════════════════════════════════════════════════════════════════
# BROADCAST / HUD
# ═════════════════════════════════════════════════════════════════════════════

class TestBroadcast:
    def test_broadcast_sends_valid_json(self):
        received = []
        server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server.bind(("127.0.0.1", 0))
        port = server.getsockname()[1]
        server.settimeout(1.0)

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        act = _make_actuator()
        act._hud_sock = sock
        act._hud_addr = ("127.0.0.1", port)

        act._broadcast(500.0, 0.85, 2.975)

        try:
            data, _ = server.recvfrom(65536)
            received.append(json.loads(data.decode()))
        except socket.timeout:
            pass
        server.close(); sock.close()

        assert len(received) == 1
        pkt = received[0]
        assert pkt["SYM"]  == "SPY"
        assert pkt["P"]    == pytest.approx(500.0, abs=0.01)
        assert pkt["R"]    == pytest.approx(0.85,  abs=0.001)
        assert pkt["Z"]    == pytest.approx(2.975, abs=0.001)
        assert "VAULT"     in pkt
        assert "TS"        in pkt

    def test_broadcast_silently_skips_without_socket(self):
        act = _make_actuator()
        # _hud_sock is None — must not raise
        act._broadcast(500.0, 0.85, 2.975)

    def test_broadcast_includes_circuit_breaker_state(self):
        received = []
        server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server.bind(("127.0.0.1", 0))
        port = server.getsockname()[1]
        server.settimeout(1.0)

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        act = _make_actuator()
        act._hud_sock = sock
        act._hud_addr = ("127.0.0.1", port)
        act._breaker.tripped = True

        act._broadcast(500.0, 0.5, 1.75)

        try:
            data, _ = server.recvfrom(65536)
            received.append(json.loads(data.decode()))
        except socket.timeout:
            pass
        server.close(); sock.close()

        assert received[0].get("CB") == True

    def test_broadcast_fires_on_every_act_call(self):
        """Even when all phases abort, broadcast must still fire."""
        packets = []
        server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server.bind(("127.0.0.1", 0))
        port = server.getsockname()[1]
        server.settimeout(0.2)

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        act = _make_actuator()
        act._hud_sock = sock
        act._hud_addr = ("127.0.0.1", port)

        action = np.array([0.0, 0.0], dtype=np.float32)
        act.update_market_context(None, 5.0, 50.0,
                                  False,   # market closed → Gate B
                                  3600.0, 100_000.0, 200_000.0, 3)
        act.act(action, 0.3, {})

        try:
            data, _ = server.recvfrom(65536)
            packets.append(json.loads(data.decode()))
        except socket.timeout:
            pass
        server.close(); sock.close()

        # Heartbeat broadcast fires before Gate B check
        assert len(packets) == 1


# ═════════════════════════════════════════════════════════════════════════════
# VAULT ↔ ACTUATOR INTEGRATION
# ═════════════════════════════════════════════════════════════════════════════

class TestVaultActuatorIntegration:
    def test_circuit_breaker_uses_live_vault_pnl(self):
        """Circuit breaker reads session_pnl_pct from the real vault."""
        vault = rt.VaultState(100_000.0)
        act   = _make_actuator(vault=vault, max_loss_pct=2.0)

        # Drain vault by 2.1% to trigger breaker
        vault.record_pnl(-2_100.0)

        action = np.array([0.5, 0.0], dtype=np.float32)
        act.update_market_context(None, 5.0, 50.0, True, 3600.0,
                                  100_000.0, 200_000.0, 3)
        act.act(action, 0.99, {})

        assert act._breaker.tripped == True

    def test_profit_banked_after_execute(self):
        vault = rt.VaultState(100_000.0)
        act   = _make_actuator(vault=vault)
        # Simulate a win by directly calling record_pnl (execute returns 0 in sim)
        vault.record_pnl(500.0)
        assert vault.banked      == pytest.approx(250.0)
        assert vault.active_pool == pytest.approx(100_250.0)


# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

class TestActuatorSummary:
    def test_summary_contains_key_fields(self, actuator):
        s = actuator.summary()
        assert "ticks"   in s
        assert "entries" in s
        assert "equity"  in s
        assert "CB"      in s

    def test_summary_shows_cb_ok_initially(self, actuator):
        assert "OK" in actuator.summary()

    def test_summary_shows_cb_tripped(self, actuator):
        actuator._breaker.tripped = True
        assert "TRIPPED" in actuator.summary()
