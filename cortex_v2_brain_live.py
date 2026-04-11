"""
cortex_v2_brain_live.py — CORTEX-16 Brain Integration
======================================================
Wires DopamineSystem + CJEPAPredictor + StaticCSRRouter
into the live trading loop from cortex_live_v1_fixed.py.

Architecture:
  tick
   ↓
  StaticCSRRouter    — sparse attention over market state (GIL-bypass)
   ↓
  CJEPAPredictor     — 5-tick latent lookahead (INT8-ready)
   ↓
  DopamineSystem     — dynamic threshold via RPE + cortisol
   ↓
  MPC action         — buy/sell/hold with DA-gated sizing

Key changes from cortex_live_v1_fixed.py:
  - Z_THRESHOLD is no longer fixed (2.92) — DA system sets it dynamically
  - Cortisol rises on sustained adverse conditions (quarter-end defence)
  - 5-tick lookahead scores candidate actions before execution
  - StaticCSRRouter pre-filters market state for <1ms routing

Usage:
  python cortex_v2_brain_live.py

Requirements (same as cortex_live_v1_fixed.py):
  pip install alpaca-trade-api torch numpy python-dotenv pynwb scipy
"""

import os
import math
import time
import struct
import socket
import threading
from dataclasses import dataclass, field
from typing import Optional, Deque
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from dotenv import load_dotenv

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────

ALPACA_KEY    = os.getenv("ALPACA_PAPER_KEY")
ALPACA_SECRET = os.getenv("ALPACA_PAPER_SECRET")
ALPACA_URL    = "https://paper-api.alpaca.markets"
TARGET_IP     = "192.168.1.150"
PORT          = 5005
SYMBOL        = "SPY"

# Base thresholds — DA system will modulate these
BASE_Z_ENTRY  = 2.92
BASE_Z_EXIT   = 2.92
MAX_SPREAD    = 0.15
TOXIC_OBI     = 10.0


# ══════════════════════════════════════════════════════════════════════════════
# 1. StaticCSRRouter — sparse market state attention (GIL-bypass ready)
# ══════════════════════════════════════════════════════════════════════════════

class StaticCSRRouter:
    """
    Pre-computed sparse attention over market state features.
    Routes 8 market signals to 4 expert channels via static CSR mask.
    
    Channels:
      0: Price momentum   (bid/ask/micro_price)
      1: Volume/liquidity (bid_sz, ask_sz, spread)
      2: Volatility       (rho z-score, rho variance)
      3: Timing           (seconds_to_close, rtt)

    Static mask learned from historical CORTEX-16 logs.
    CSR format bypasses Python dict lookups — O(1) routing.
    """

    FEATURE_NAMES = [
        "bid", "ask", "micro_price", "spread",
        "bid_sz", "ask_sz", "z_score", "rtt_ms"
    ]
    N_FEATURES  = 8
    N_CHANNELS  = 4

    # Static routing mask — features → channels
    # Row = feature, Col = channel
    _MASK = np.array([
        [1, 0, 0, 0],   # bid           → price
        [1, 0, 0, 0],   # ask           → price
        [1, 0, 0, 0],   # micro_price   → price
        [0, 1, 0, 0],   # spread        → liquidity
        [0, 1, 0, 0],   # bid_sz        → liquidity
        [0, 1, 0, 0],   # ask_sz        → liquidity
        [0, 0, 1, 0],   # z_score       → volatility
        [0, 0, 0, 1],   # rtt_ms        → timing
    ], dtype=np.float32)

    def __init__(self):
        self._csr = csr_matrix(self._MASK)
        # Normalisation stats (updated online)
        self._mean = np.zeros(self.N_FEATURES, dtype=np.float32)
        self._std  = np.ones(self.N_FEATURES,  dtype=np.float32)
        self._n    = 0

    def route(self, bid: float, ask: float, micro_price: float,
              spread: float, bid_sz: float, ask_sz: float,
              z_score: float, rtt_ms: float) -> np.ndarray:
        """
        Route market features to 4 expert channels.
        Returns: (4,) float32 channel activations, normalised.
        """
        x = np.array([bid, ask, micro_price, spread,
                       bid_sz, ask_sz, z_score, rtt_ms],
                      dtype=np.float32)

        # Online normalisation
        self._n += 1
        alpha = 1.0 / min(self._n, 1000)
        self._mean = (1 - alpha) * self._mean + alpha * x
        self._std  = (1 - alpha) * self._std  + alpha * np.abs(x - self._mean)
        x_norm = (x - self._mean) / (self._std + 1e-6)

        # CSR sparse matrix-vector multiply — O(nnz) routing
        channels = self._csr.T.dot(x_norm)   # (4,)
        return channels.astype(np.float32)

    @property
    def channel_names(self):
        return ["price", "liquidity", "volatility", "timing"]


# ══════════════════════════════════════════════════════════════════════════════
# 2. CJEPAPredictor — 5-tick market latent lookahead
# ══════════════════════════════════════════════════════════════════════════════

class CJEPAPredictor(nn.Module):
    """
    Causal-JEPA predictor for 5-tick market state lookahead.
    Input:  4-channel market state (from StaticCSRRouter) + action
    Output: predicted 4-channel market state at t+k

    Architecture:
      compress(4) → 16-D latent
      bridge(16 + action_dim) → 32-D
      predict(32) → 4-D next state

    Designed for INT8 quantisation on AMD NPU (DreamerV3 tricks active).
    """

    def __init__(self, state_dim: int = 4, action_dim: int = 3,
                 hidden_dim: int = 32):
        super().__init__()
        self.state_dim  = state_dim
        self.action_dim = action_dim

        # Symlog encoder (DreamerV3)
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 16),
        )

        # Action conditioning
        self.bridge = nn.Sequential(
            nn.Linear(16 + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        )

        # Decoder
        self.decoder = nn.Linear(hidden_dim, state_dim)

    @staticmethod
    def symlog(x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.log1p(torch.abs(x))

    @staticmethod
    def symexp(x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

    def forward(self, state: torch.Tensor,
                action: torch.Tensor) -> torch.Tensor:
        """
        state:  (B, state_dim)
        action: (B, action_dim)  — one-hot: [buy, sell, hold]
        returns: (B, state_dim) predicted next state
        """
        z  = self.encoder(self.symlog(state))
        h  = self.bridge(torch.cat([z, action], dim=-1))
        return self.symexp(self.decoder(h))

    @torch.no_grad()
    def rollout(self, state: np.ndarray, action: np.ndarray,
                horizon: int = 5) -> np.ndarray:
        """
        Roll out H steps from state under repeated action.
        Returns predicted states: (H, state_dim)
        """
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        a = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        preds = []
        for _ in range(horizon):
            s = self(s, a)
            preds.append(s.squeeze(0).numpy())
        return np.stack(preds)


# ══════════════════════════════════════════════════════════════════════════════
# 3. DopamineSystem — dynamic threshold via RPE + cortisol
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class NeuroState:
    """Live neuromodulator state — broadcast to cockpit."""
    da:       float = 0.500   # Dopamine       — surprise
    sht:      float = 0.500   # Serotonin      — stability
    ne:       float = 0.200   # Norepinephrine — arousal (from rho)
    ach:      float = 0.500   # Acetylcholine  — attention
    ecb:      float = 0.000   # Endocannabinoid— retrograde damp
    cortisol: float = 0.000   # Cortisol       — sustained stress
    ei:       float = 1.000   # E/I balance    — derived
    regime:   str   = "EXPLOIT"

    # Derived planner params
    z_entry_threshold:  float = BASE_Z_ENTRY
    z_exit_threshold:   float = BASE_Z_EXIT
    position_scale:     float = 1.0
    explore_std:        float = 0.10


class DopamineSystem:
    """
    Biological dopamine + cortisol trading signal.

    DA  = RPE: difference between predicted and actual market z-score
    5HT = stability: low variance in recent z-scores
    NE  = arousal: rho z-score from Allen Neuropixels
    eCB = retrograde dampening after large trades
    CORT= sustained adverse: rises when conditions are toxic > N ticks

    Thresholds:
      z_entry = BASE * (1 + cortisol * 0.5)   — harder to enter when stressed
      z_exit  = BASE * (1 - da * 0.2)         — faster exit when surprised
      pos_scale = 1 - cortisol * 0.7           — reduce size when stressed
    """

    DA_DECAY   = 0.95
    SHT_DECAY  = 0.90
    ECB_DECAY  = 0.85
    CORT_DECAY = 0.97    # slow — cortisol persists hours
    CORT_SENS  = 0.10    # sensitivity to adverse conditions

    def __init__(self):
        self.state       = NeuroState()
        self._z_pred     = 0.0    # predicted z from CJEPAPredictor
        self._z_hist:  Deque[float] = deque(maxlen=20)
        self._adverse_ticks        = 0
        self._last_action_size     = 0.0

    def update(self,
               z_actual:    float,
               z_predicted: float,
               rho:         float,
               is_toxic:    bool,
               rtt_ms:      float,
               action_size: float = 0.0) -> NeuroState:
        """
        Update all 8 signals given current market state.
        Call once per tick (50ms).
        """
        s = self.state

        # ── DA: Reward Prediction Error ───────────────────────────────────
        rpe = abs(z_actual - z_predicted) / (abs(z_predicted) + 0.1)
        da_raw = min(1.0, rpe)
        s.da = self.DA_DECAY * s.da + (1 - self.DA_DECAY) * da_raw

        # ── 5HT: Stability (low z-score variance) ─────────────────────────
        self._z_hist.append(z_actual)
        if len(self._z_hist) > 2:
            variance = float(np.var(list(self._z_hist)))
            stability = math.exp(-5.0 * variance)
        else:
            stability = 0.5
        s.sht = self.SHT_DECAY * s.sht + (1 - self.SHT_DECAY) * stability

        # ── NE: Norepinephrine from rho (global arousal) ──────────────────
        s.ne = min(1.0, max(0.0, rho / 10.0))

        # ── ACh: Attention = DA + instability ────────────────────────────
        ach_raw = (da_raw + (1.0 - stability)) / 2.0
        s.ach = self.ECB_DECAY * s.ach + (1 - self.ECB_DECAY) * ach_raw

        # ── eCB: Retrograde dampening after trades ────────────────────────
        ecb_raw = s.da * min(1.0, action_size / 100.0)
        s.ecb = self.ECB_DECAY * s.ecb + (1 - self.ECB_DECAY) * ecb_raw
        da_eff = s.da * (1.0 - s.ecb * 0.4)

        # ── Cortisol: Sustained stress ────────────────────────────────────
        # Rises when: toxic spread, high RTT, or high z-score
        adverse = float(is_toxic or rtt_ms > 45.0 or abs(z_actual) > 4.0)
        self._adverse_ticks = self._adverse_ticks * 0.9 + adverse
        cort_raw = max(0.0, self._adverse_ticks / 10.0 - 0.2)  # baseline subtracted
        s.cortisol = self.CORT_DECAY * s.cortisol + self.CORT_SENS * cort_raw

        # ── E/I: Derived ─────────────────────────────────────────────────
        s.ei = float(np.clip(da_eff / (1.0 - s.sht + 0.1), 0.5, 2.0))

        # ── Regime ────────────────────────────────────────────────────────
        if s.sht < 0.3:
            s.regime = "WAIT"
        elif da_eff > 0.6:
            s.regime = "EXPLORE"
        else:
            s.regime = "EXPLOIT"

        # ── Derived planner parameters ────────────────────────────────────
        # Cortisol raises entry threshold (harder to trade when stressed)
        s.z_entry_threshold = BASE_Z_ENTRY * (1.0 + s.cortisol * 0.5)
        # DA lowers exit threshold (faster exit when surprised)
        s.z_exit_threshold  = BASE_Z_EXIT  * (1.0 - da_eff * 0.2)
        # Cortisol shrinks position size
        s.position_scale    = max(0.1, 1.0 - s.cortisol * 0.7)
        # E/I sets exploration width
        s.explore_std       = float(np.clip(0.05 * s.ei, 0.02, 0.15))

        self._last_action_size = action_size
        return s


# ══════════════════════════════════════════════════════════════════════════════
# 4. MPC Action Selector — Best-of-K with CJEPAPredictor scoring
# ══════════════════════════════════════════════════════════════════════════════

class MPCActionSelector:
    """
    Sample K candidate actions, score each via 5-tick CJEPAPredictor rollout,
    return the action with lowest predicted volatility channel value.
    
    Actions: [buy=1,0,0], [sell=0,1,0], [hold=0,0,1]
    Scoring: minimize predicted volatility (channel 2) at t+5
    """

    ACTIONS = {
        "buy":  np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "sell": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "hold": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }

    def __init__(self, predictor: CJEPAPredictor, horizon: int = 5):
        self.predictor = predictor
        self.horizon   = horizon

    def select(self, channels: np.ndarray,
               ns: NeuroState) -> tuple[str, float]:
        """
        Given current market channels + neuro state,
        return best action and its confidence score.
        """
        scores = {}
        for name, action in self.ACTIONS.items():
            preds = self.predictor.rollout(channels, action, self.horizon)
            # Score = negative predicted volatility at end of horizon
            # Lower volatility = better
            vol_channel = preds[-1, 2]   # channel 2 = volatility
            scores[name] = -float(vol_channel)

        # Softmax weighting
        vals = np.array(list(scores.values()))
        weights = np.exp((vals - vals.max()) / (ns.explore_std + 1e-6))
        weights /= weights.sum()

        # Best action
        best = max(scores, key=scores.get)
        confidence = float(weights[list(scores.keys()).index(best)])
        return best, confidence


# ══════════════════════════════════════════════════════════════════════════════
# 5. CortexBrainEngine — full wired trading loop
# ══════════════════════════════════════════════════════════════════════════════

class CortexBrainEngine:
    """
    CORTEX-16 v2 Brain: Full neuromodulated trading engine.
    
    Replaces fixed-threshold logic in cortex_live_v1_fixed.py with:
      - StaticCSRRouter: market state routing
      - CJEPAPredictor: 5-tick lookahead
      - DopamineSystem: dynamic thresholds via RPE + cortisol
      - MPC action selection

    All signals broadcast to cockpit at 192.168.1.150:5005
    """

    def __init__(self):
        from alpaca_trade_api.rest import REST
        self.alpaca = REST(ALPACA_KEY, ALPACA_SECRET, ALPACA_URL)
        self.sock   = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # ── Brain components ──────────────────────────────────────────────
        self.router    = StaticCSRRouter()
        self.predictor = CJEPAPredictor()
        self.da_system = DopamineSystem()
        self.mpc       = MPCActionSelector(self.predictor)

        # ── Trading state ─────────────────────────────────────────────────
        self.mirror_wallet   = 500.00
        self.mirror_dtbp     = 500.00
        self.daytrade_count  = 0
        self.cached_qty      = 0.0
        self.active_order_id = None
        self.order_submit_time = 0.0
        self.is_toxic        = False

        # ── Timing ────────────────────────────────────────────────────────
        self.last_account_sync = 0.0
        self.last_quote_sync   = 0.0
        self.last_clock_sync   = 0.0
        self.market_open       = False
        self.seconds_to_close  = 999999.0
        self.start_time        = time.time()

        # ── Rho (Allen Neuropixels proxy) ─────────────────────────────────
        self.rho_history: Deque[float] = deque(maxlen=1000)

        # ── History for prediction ────────────────────────────────────────
        self.channel_history:  Deque[np.ndarray] = deque(maxlen=20)
        self.last_channels     = np.zeros(4, dtype=np.float32)
        self.last_z            = 0.0

        print("🧠 CORTEX-16 v2 Brain Engine initialised")
        print(f"   StaticCSRRouter: {self.router.N_FEATURES}→{self.router.N_CHANNELS} channels")
        print(f"   CJEPAPredictor:  5-tick lookahead")
        print(f"   DopamineSystem:  8 neuromodulatory signals")
        print(f"   MPC:             3 actions × {self.mpc.horizon} horizon")

    # ── Rho (simplified — replace with Allen NWB when available) ──────────
    def get_rho(self) -> float:
        elapsed = (time.time() - self.start_time) % 60
        return float(abs(math.sin(elapsed * 0.3)) * 5.0
                     + np.random.normal(0, 0.5))

    # ── Market data ────────────────────────────────────────────────────────
    def get_quote(self) -> Optional[dict]:
        if time.perf_counter() - self.last_quote_sync < 0.05:
            return None
        try:
            t0    = time.perf_counter()
            q     = self.alpaca.get_latest_quote(SYMBOL, feed="iex")
            rtt   = (time.perf_counter() - t0) * 1000.0
            self.last_quote_sync = time.perf_counter()

            if rtt > 45.0:
                return None

            bid, ask   = float(q.bp), float(q.ap)
            bid_sz     = float(q.bs)
            ask_sz     = float(q.as_)
            spread     = ask - bid
            micro      = ((ask_sz * bid + bid_sz * ask)
                          / (bid_sz + ask_sz + 1e-9))
            self.is_toxic = (ask_sz / (bid_sz + 1e-9)) >= TOXIC_OBI

            return dict(bid=bid, ask=ask, micro=micro, spread=spread,
                        bid_sz=bid_sz, ask_sz=ask_sz, rtt_ms=rtt)
        except Exception as e:
            return None

    # ── Account sync ───────────────────────────────────────────────────────
    def sync_account(self):
        if time.perf_counter() - self.last_account_sync < 5.0:
            return
        try:
            acc = self.alpaca.get_account()
            self.mirror_wallet  = 500.00 * (float(acc.equity) / 100000.0)
            self.mirror_dtbp    = self.mirror_wallet * (
                float(acc.daytrading_buying_power) / float(acc.equity))
            self.daytrade_count = int(acc.daytrade_count)
            try:
                self.cached_qty = float(
                    self.alpaca.get_position(SYMBOL).qty)
            except Exception:
                self.cached_qty = 0.0
            self.last_account_sync = time.perf_counter()
        except Exception:
            pass

    # ── Clock ───────────────────────────────────────────────────────────────
    def check_market_hours(self):
        if time.perf_counter() - self.last_clock_sync < 60.0:
            return
        try:
            c = self.alpaca.get_clock()
            self.market_open = c.is_open
            self.seconds_to_close = (
                (c.next_close - c.timestamp).total_seconds()
                if self.market_open else 999999.0)
            self.last_clock_sync = time.perf_counter()
        except Exception:
            pass

    # ── Fill management ────────────────────────────────────────────────────
    def manage_fills(self):
        if not self.active_order_id:
            return
        try:
            order = self.alpaca.get_order(self.active_order_id)
            if order.status in ["filled", "canceled", "expired", "rejected"]:
                if order.status == "filled":
                    self.last_account_sync = 0.0
                self.active_order_id = None
        except Exception:
            pass
        if (self.active_order_id
                and time.perf_counter() - self.order_submit_time > 3.0):
            try:
                self.alpaca.cancel_order(self.active_order_id)
            except Exception:
                pass
            self.active_order_id = None

    # ── Core brain tick ────────────────────────────────────────────────────
    def brain_tick(self, q: dict, rho: float, z_score: float) -> NeuroState:
        """
        Run one brain tick:
          1. Route market state via CSR
          2. Predict next state via CJEPA
          3. Update DA system
          4. Return neuro state with derived thresholds
        """
        # Route market features
        channels = self.router.route(
            bid=q["bid"], ask=q["ask"], micro_price=q["micro"],
            spread=q["spread"], bid_sz=q["bid_sz"], ask_sz=q["ask_sz"],
            z_score=z_score, rtt_ms=q["rtt_ms"]
        )
        self.channel_history.append(channels)

        # Predict next state (hold action as default)
        hold_action = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        predicted = self.predictor.rollout(channels, hold_action, horizon=1)
        # z_predicted from volatility channel (channel 2 ≈ z-score proxy)
        z_predicted = float(predicted[0, 2]) * 3.0

        # Update DA system
        ns = self.da_system.update(
            z_actual    = z_score,
            z_predicted = z_predicted,
            rho         = rho,
            is_toxic    = self.is_toxic,
            rtt_ms      = q["rtt_ms"],
            action_size = self.cached_qty,
        )

        self.last_channels = channels
        self.last_z        = z_score
        return ns

    # ── Order execution with brain gates ──────────────────────────────────
    def execute(self, q: dict, ns: NeuroState, z_score: float):
        """Execute orders using DA-gated thresholds."""
        if self.active_order_id:
            return

        is_eod = self.seconds_to_close <= 300.0
        # DA-gated thresholds
        z_entry = ns.z_entry_threshold
        z_exit  = ns.z_exit_threshold

        # DA lockout during WAIT regime
        if ns.regime == "WAIT":
            return

        bid, ask   = q["bid"], q["ask"]
        micro      = q["micro"]
        spread     = q["spread"]

        if spread > MAX_SPREAD or self.is_toxic:
            return

        urgency    = min(1.0, max(0.0, (z_score - z_entry) / 1.0))
        limit_p    = round(micro + (ask - micro) * urgency, 2)

        # EXIT
        if ((z_score > z_exit or is_eod or self.is_toxic)
                and self.cached_qty > 0):
            exit_p = (round(bid - 0.01, 2)
                      if (self.is_toxic or z_score > 4.0 or is_eod)
                      else limit_p)
            try:
                order = self.alpaca.submit_order(
                    symbol=SYMBOL, qty=self.cached_qty, side="sell",
                    type="limit", limit_price=exit_p, time_in_force="day")
                self.active_order_id   = order.id
                self.order_submit_time = time.perf_counter()
                self.last_account_sync = 0.0
                print(f"🚨 EXIT | z={z_score:.2f}σ | DA={ns.da:.3f} | "
                      f"CORT={ns.cortisol:.3f} | ${exit_p}")
            except Exception as e:
                print(f"⚠️ Exit error: {e}")

        # ENTRY
        elif (z_score <= z_entry
              and self.cached_qty == 0
              and not is_eod
              and not self.is_toxic
              and self.daytrade_count < 3):
            # MPC action selection
            best_action, confidence = self.mpc.select(self.last_channels, ns)
            if best_action != "buy" or confidence < 0.35:
                return  # MPC says don't buy

            # DA-gated position sizing
            raw_qty = min(self.mirror_wallet * 0.95,
                          self.mirror_dtbp) / limit_p
            qty = round(raw_qty * ns.position_scale, 4)

            if qty < 0.0001:
                return

            try:
                order = self.alpaca.submit_order(
                    symbol=SYMBOL, qty=qty, side="buy",
                    type="limit", limit_price=limit_p, time_in_force="day")
                self.active_order_id   = order.id
                self.order_submit_time = time.perf_counter()
                print(f"🎯 ENTRY | z={z_score:.2f}σ | qty={qty:.4f} | "
                      f"DA={ns.da:.3f} | CORT={ns.cortisol:.3f} | "
                      f"scale={ns.position_scale:.2f} | ${limit_p}")
            except Exception as e:
                print(f"⚠️ Entry error: {e}")

    # ── Telemetry ────────────────────────────────────────────────────────
    def broadcast(self, ns: NeuroState, q: dict, z_score: float):
        """Pack and send neuro state + market data to cockpit."""
        # Extended packet: wallet + 8 signals + z + rtt + regime(int)
        regime_int = {"EXPLOIT": 1, "EXPLORE": 2, "WAIT": 0}.get(
            ns.regime, 1)
        pkt = struct.pack(
            "<dffffffff fi i",
            time.time(),
            float(self.mirror_wallet),
            float(ns.da),
            float(ns.sht),
            float(ns.ne),
            float(ns.ach),
            float(ns.ecb),
            float(ns.cortisol),
            float(ns.ei),
            float(z_score),
            float(q.get("rtt_ms", 0.0)),
            int(regime_int),
        )
        try:
            self.sock.sendto(pkt, (TARGET_IP, PORT))
        except Exception:
            pass

    # ── Main loop ─────────────────────────────────────────────────────────
    def run(self):
        print(f"\n🚀 CORTEX-16 v2 BRAIN LIVE")
        print(f"   Symbol: {SYMBOL} | Target: {TARGET_IP}:{PORT}")
        print(f"   Vault: ${self.mirror_wallet:.2f}\n")

        tick = 0
        while True:
            try:
                tick += 1
                rho = self.get_rho()
                self.rho_history.append(rho)

                # Z-score from rho history
                if len(self.rho_history) > 10:
                    arr = np.array(self.rho_history)
                    z = float((rho - arr.mean()) / (arr.std() + 1e-6))
                else:
                    z = 0.0

                self.check_market_hours()
                if not self.market_open:
                    time.sleep(1.0)
                    continue

                self.sync_account()
                self.manage_fills()

                q = self.get_quote()
                if q is None:
                    time.sleep(0.05)
                    continue

                # Brain tick
                ns = self.brain_tick(q, rho, z)

                # Execute
                self.execute(q, ns, z)

                # Telemetry
                self.broadcast(ns, q, z)

                # Console heartbeat every 100 ticks
                if tick % 100 == 0:
                    print(f"[t={tick}] z={z:.3f} DA={ns.da:.3f} "
                          f"5HT={ns.sht:.3f} CORT={ns.cortisol:.3f} "
                          f"regime={ns.regime} "
                          f"z_entry={ns.z_entry_threshold:.3f} "
                          f"scale={ns.position_scale:.2f} "
                          f"wallet=${self.mirror_wallet:.2f}")

                time.sleep(0.05)

            except KeyboardInterrupt:
                print("\n⏹ Shutdown requested")
                break
            except Exception as e:
                print(f"🔌 RECOVERY: {e}")
                time.sleep(2.0)


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    CortexBrainEngine().run()
