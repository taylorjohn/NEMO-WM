"""
market_replay.py — CORTEX-16 After-Hours Simulator with Live Trading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Synthetic market scenarios + real simulated position management.

The simulator now has two layers:
  1. CHAOS LAYER  — All 18 safeguard phases exercised continuously
  2. TRADE LAYER  — Real entry/hold/exit logic with P&L tracking

Positions
---------
  Entry  : When Phase 15 passes SCOUT/HUNTER/SNIPER tier, a position opens
           at the VWMP micro-price with appropriate share quantity
  Hold   : Position held while resonance stays above exit threshold
  Stop   : Automatic stop-loss at 0.5% adverse from entry (configurable)
  Target : Profit target at 1.0% from entry (configurable)
  EOD    : All positions closed at Phase 11 EOD trigger
  Panic  : All positions closed at Phase 17 Systemic Fracture
  Flip   : Signal reversal closes existing position and opens opposite

Position sizing
---------------
  SCOUT  : 1% of active vault
  HUNTER : 5% of active vault
  SNIPER : 20% of active vault

50/50 Vault sequestration
--------------------------
  Half of every realised profit permanently banked.
  Half returned to active trading pool.

Scenarios (auto-cycle every 35 ticks)
---------------------------------------
  CALM          Low vol — holds, occasional Scout
  TRENDING_UP   Rising — Scout/Hunter entries, momentum exits
  SATURATED     High rho — Sniper fires, fast exits
  FLASH_CRASH   Z>4 — panic sweep, stop-loss hits
  TOXIC_FLOW    VPIN abort — no trades
  SPREAD_BLOW   Spread abort — no trades
  STALE_DATA    Phase 4 abort + backoff
  GHOST_RTT     Phase 2 abort
  PDT_LOCK      Phase 5 hard block
  EOD           Phase 11 liquidation of all open positions
  FRACTURE      Phase 17 systemic — all positions closed
  RECOVERY      DA spike — exploitation mode, tight entries
  TRENDING_DOWN Bearish — short signals (sold from existing long)

Usage
-----
    python market_replay.py                          # auto-cycle
    python market_replay.py --scenario SATURATED     # lock scenario
    python market_replay.py --hz 5 --hud-ip 192.168.1.150
    python market_replay.py --stop 0.005 --target 0.01  # custom SL/TP
    python market_replay.py --no-hud                 # terminal only
"""
from __future__ import annotations
import argparse, collections, json, math, socket, time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

parser = argparse.ArgumentParser(description="CORTEX-16 Market Replay")
parser.add_argument("--scenario",     default=None)
parser.add_argument("--hz",           type=float, default=5.0)
parser.add_argument("--hud-ip",       default="127.0.0.1")
parser.add_argument("--hud-port",     type=int,   default=5005)
parser.add_argument("--no-hud",       action="store_true")
parser.add_argument("--symbols",      nargs="*",  default=["SPY","QQQ","IWM","NVDA","AAPL","TSLA"])
parser.add_argument("--stop",         type=float, default=0.005)
parser.add_argument("--target",       type=float, default=0.010)
parser.add_argument("--vault",        type=float, default=101_133.47)
parser.add_argument("--no-learn",     action="store_true")
parser.add_argument("--reset-learn",  action="store_true")
parser.add_argument("--duration",     type=float, default=None,
                    help="Auto-stop after N minutes (e.g. 10, 30, 60, 1440)")
parser.add_argument("--session-name", default=None,
                    help="Label for this session (e.g. '10min' '1hour' '1day')")
args = parser.parse_args()

import os
if args.reset_learn and os.path.exists("cortex_learning.json"):
    os.remove("cortex_learning.json")
    print("  🗑  Learning state reset.")

from cortex_brain.engine import CortexEngine, EngineConfig, FeatureEncoder, Actuator
from cortex_brain.perception.lsm        import LSMConfig
from cortex_brain.perception.eb_jepa    import EBJEPAConfig
from cortex_brain.routing.static_csr    import CSRRouterConfig
from cortex_brain.memory.ttm_clustering import TTMConfig
from cortex_brain.hardware.amd_npu_binding import NPUConfig
from cortex_brain.neuro.dopamine        import DopamineConfig
from cortex_brain.trading.safeguards    import (
    phase15_resonance_gate, phase17_systemic_fracture,
    phase18_heartbeat, run_pre_execution_audit,
    StaleDataGuard, APIGovernor, PanicState, HeartbeatSync,
)

# Learning engine — import from local file (copy to CORTEX folder)
try:
    from learning_engine import LearningEngine
    _LEARNING_AVAILABLE = True
except ImportError:
    _LEARNING_AVAILABLE = False
    print("  ⚠  learning_engine.py not found — adaptive learning disabled")

# ─────────────────────────────────────────────────────────────────────────────
# POSITION MANAGER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Position:
    side:       str    # 'long' | 'short'
    symbol:     str
    qty:        int
    entry_price:float
    entry_time: float
    tier:       str
    stop_price: float
    target_price:float
    entry_tick: int
    unrealised_pnl: float = 0.0

@dataclass
class ClosedTrade:
    side:       str
    symbol:     str
    qty:        int
    entry_price:float
    exit_price: float
    exit_reason:str    # 'STOP' | 'TARGET' | 'SIGNAL' | 'EOD' | 'PANIC' | 'FRACTURE'
    pnl:        float
    pnl_pct:    float
    hold_ticks: int
    tier:       str
    timestamp:  float

class PositionManager:
    """
    Tracks open position and closed trade history.
    One position per symbol at a time (no pyramiding in replay).
    """
    def __init__(self, stop_pct: float, target_pct: float) -> None:
        self.stop_pct    = stop_pct
        self.target_pct  = target_pct
        self.position:   Optional[Position] = None
        self.closed:     List[ClosedTrade]  = []
        self.total_pnl:  float = 0.0
        self.win_count:  int   = 0
        self.loss_count: int   = 0

    def open(self, side: str, symbol: str, qty: int, price: float,
             tier: str, tick: int) -> Position:
        if side == 'long':
            stop   = price * (1 - self.stop_pct)
            target = price * (1 + self.target_pct)
        else:
            stop   = price * (1 + self.stop_pct)
            target = price * (1 - self.target_pct)

        self.position = Position(
            side=side, symbol=symbol, qty=qty,
            entry_price=price, entry_time=time.time(),
            tier=tier, stop_price=round(stop,2),
            target_price=round(target,2), entry_tick=tick,
        )
        return self.position

    def update_unrealised(self, current_price: float) -> float:
        if self.position is None: return 0.0
        p = self.position
        if p.side == 'long':
            p.unrealised_pnl = (current_price - p.entry_price) * p.qty
        else:
            p.unrealised_pnl = (p.entry_price - current_price) * p.qty
        return p.unrealised_pnl

    def check_exits(self, current_price: float, tick: int) -> Optional[str]:
        """Returns exit reason string if position should close, else None."""
        if self.position is None: return None
        p = self.position
        if p.side == 'long':
            if current_price <= p.stop_price:   return 'STOP'
            if current_price >= p.target_price: return 'TARGET'
        else:
            if current_price >= p.stop_price:   return 'STOP'
            if current_price <= p.target_price: return 'TARGET'
        return None

    def close(self, exit_price: float, reason: str, tick: int) -> Optional[ClosedTrade]:
        if self.position is None: return None
        p = self.position
        if p.side == 'long':
            pnl = (exit_price - p.entry_price) * p.qty
        else:
            pnl = (p.entry_price - exit_price) * p.qty
        pnl_pct = pnl / (p.entry_price * p.qty) * 100

        trade = ClosedTrade(
            side=p.side, symbol=p.symbol, qty=p.qty,
            entry_price=p.entry_price, exit_price=round(exit_price,2),
            exit_reason=reason, pnl=round(pnl,2),
            pnl_pct=round(pnl_pct,3),
            hold_ticks=tick - p.entry_tick,
            tier=p.tier, timestamp=time.time(),
        )
        self.closed.append(trade)
        self.total_pnl += pnl
        if pnl > 0: self.win_count += 1
        else:        self.loss_count += 1
        self.position = None
        return trade

    def close_all(self, exit_price: float, reason: str, tick: int) -> Optional[ClosedTrade]:
        return self.close(exit_price, reason, tick)

    @property
    def win_rate(self) -> float:
        total = self.win_count + self.loss_count
        return self.win_count / total if total > 0 else 0.0

    @property
    def trade_count(self) -> int:
        return len(self.closed)


# ─────────────────────────────────────────────────────────────────────────────
# VAULT STATE
# ─────────────────────────────────────────────────────────────────────────────

class VaultState:
    def __init__(self, seed: float) -> None:
        self.seed = seed
        self.active_pool  = seed
        self.banked       = 0.0
        self.total_equity = seed

    def record_pnl(self, pnl: float) -> None:
        if pnl > 0:
            self.banked      += pnl * 0.5
            self.active_pool += pnl * 0.5
        else:
            self.active_pool += pnl
        self.total_equity = self.active_pool + self.banked

    def alloc_dollars(self, pct: float) -> float:
        return max(0.0, self.active_pool * pct)

    def gain_pct(self) -> float:
        return (self.total_equity - self.seed) / self.seed * 100


# ─────────────────────────────────────────────────────────────────────────────
# OHLC + PREDICTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

class OHLCBuilder:
    def __init__(self, candle_ticks=5, max_candles=80):
        self.candle_ticks = candle_ticks
        self.max_candles  = max_candles
        self._candles     = collections.deque(maxlen=max_candles)
        self._tick=0; self._open=self._high=self._low=self._close=0.0; self._vol=0.0

    def push(self, price, volume=1000.0):
        self._tick += 1
        if self._tick == 1: self._open=self._high=self._low=price
        self._high=max(self._high,price); self._low=min(self._low,price)
        self._close=price; self._vol+=volume
        if self._tick >= self.candle_ticks:
            self._candles.append({'t':time.time(),'o':round(self._open,2),
                'h':round(self._high,2),'l':round(self._low,2),
                'c':round(self._close,2),'v':round(self._vol,0)})
            self._tick=0; self._vol=0.0; return True
        return False

    def candles(self): return list(self._candles)
    def live_bar(self):
        if self._tick==0: return None
        return {'t':time.time(),'o':round(self._open,2),'h':round(self._high,2),
                'l':round(self._low,2),'c':round(self._close,2),'v':round(self._vol,0),'live':True}


class TrajectoryPredictor:
    def __init__(self, horizon=12):
        self.horizon=horizon
        self._hist=collections.deque(maxlen=20)

    def push(self, price): self._hist.append(price)

    def predict(self, current_price, resonance, da, z_score):
        if len(self._hist)<3:
            return [{'t':time.time()+i*2,'p':round(current_price,2)} for i in range(1,self.horizon+1)]
        prices=list(self._hist)
        momentum=(prices[-1]-prices[-3])/max(prices[-3],1e-8)
        mean=float(np.mean(prices[-10:]))
        path=[]; p=current_price
        for i in range(1,self.horizon+1):
            mom=(momentum*p*(da-0.4)*1.5)
            rev=((mean-p)*(1.0-resonance)*0.08)
            noise=float(np.random.randn())*p*0.002*(1/i**0.5)
            p=p+mom+rev+noise
            path.append({'t':time.time()+i*2,'p':round(p,2)})
        return path


# ─────────────────────────────────────────────────────────────────────────────
# MARKET STATE + SCENARIO ENGINE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MarketState:
    symbol:str="SPY"; price:float=500.0; bid:float=499.95; ask:float=500.05
    bid_size:float=200.0; ask_size:float=200.0; ask_volume:float=500.0; bid_volume:float=500.0
    rtt_ms:float=8.0; data_age_ms:float=80.0; is_market_open:bool=True
    seconds_to_close:float=3600.0; equity:float=101_133.47; buying_power:float=202_266.94
    day_trades:int=3; scenario_name:str="CALM"; fleet_override:Optional[Dict[str,float]]=None


class ScenarioEngine:
    SCENARIO_DURATION=35
    ALL=["CALM","TRENDING_UP","SATURATED","FLASH_CRASH","TOXIC_FLOW",
         "SPREAD_BLOW","STALE_DATA","GHOST_RTT","PDT_LOCK","EOD",
         "FRACTURE","RECOVERY","TRENDING_DOWN"]

    def __init__(self, symbols):
        self.symbols=symbols; self._tick=0; self._stick=0; self._idx=0
        self._price={s:500.0+i*15 for i,s in enumerate(symbols)}
        self._rng=np.random.default_rng(42)

    def forced(self, name):
        while True: yield self._gen(name)

    def cycling(self):
        while True:
            name=self.ALL[self._idx%len(self.ALL)]
            yield self._gen(name)
            self._stick+=1
            if self._stick>=self.SCENARIO_DURATION:
                self._idx+=1; self._stick=0
                print(f"\n  ⟳  Switching to: {self.ALL[self._idx%len(self.ALL)]}\n")

    def _gen(self, name):
        self._tick+=1; t=self._tick; sym=self.symbols[0]
        drift={"TRENDING_UP":0.10,"TRENDING_DOWN":-0.10}.get(name,0.0)
        noise=float(self._rng.normal(drift,0.025))
        self._price[sym]=max(50.0,self._price[sym]*(1+noise*0.01))
        price=self._price[sym]; spread=0.05
        fleet={}
        for s in self.symbols:
            if s!=sym:
                self._price[s]=max(50.0,self._price[s]*(1+float(self._rng.normal(0,0.005))))
                base=0.3+0.15*math.sin(t*0.1+hash(s)%10)
                fleet[s]=float(np.clip(base+self._rng.normal(0,0.05),0.01,0.99))

        ms=MarketState(symbol=sym,price=round(price,2),bid=round(price-spread/2,2),
                       ask=round(price+spread/2,2),bid_size=200.0,ask_size=200.0,
                       ask_volume=500.0,bid_volume=500.0,rtt_ms=8.0,data_age_ms=80.0,
                       is_market_open=True,seconds_to_close=3600.0,
                       equity=101133.47,buying_power=202266.94,
                       day_trades=3,scenario_name=name,fleet_override=fleet)

        if name=="CALM": pass
        elif name=="TRENDING_UP":   ms.bid_size=400.0;ms.ask_size=120.0;ms.bid_volume=800.0;ms.ask_volume=250.0
        elif name=="TRENDING_DOWN": ms.bid_size=100.0;ms.ask_size=500.0;ms.bid_volume=150.0;ms.ask_volume=900.0
        elif name=="SATURATED":
            ms.bid_size=1200.0;ms.ask_size=50.0;ms.bid_volume=2500.0;ms.ask_volume=80.0
            ms.fleet_override={s:float(np.clip(0.87+self._rng.normal(0,0.02),0,1)) for s in self.symbols if s!=sym}
        elif name=="FLASH_CRASH":
            drop=0.04*(1+0.5*math.sin(t*0.5))
            ms.price=round(price*(1-drop),2);ms.bid=round(ms.price-0.8,2);ms.ask=round(ms.price+0.8,2)
            ms.ask_volume=8000.0;ms.bid_volume=40.0
            ms.fleet_override={s:float(np.clip(0.04+self._rng.normal(0,0.01),0,0.1)) for s in self.symbols if s!=sym}
        elif name=="TOXIC_FLOW":    ms.ask_volume=6000.0;ms.bid_volume=80.0
        elif name=="SPREAD_BLOW":   ms.bid=round(price-0.2,2);ms.ask=round(price+0.2,2)
        elif name=="STALE_DATA":    ms.data_age_ms=2500.0+t*15
        elif name=="GHOST_RTT":     ms.rtt_ms=70.0+self._rng.uniform(0,50)
        elif name=="PDT_LOCK":      ms.day_trades=0
        elif name=="EOD":           ms.seconds_to_close=max(0,180.0-(t%180))
        elif name=="FRACTURE":
            ms.fleet_override={s:float(np.clip(0.04+self._rng.normal(0,0.01),0,0.1)) for s in self.symbols if s!=sym}
        elif name=="RECOVERY":      ms.bid_size=700.0;ms.ask_size=180.0;ms.bid_volume=1200.0;ms.ask_volume=300.0
        fleet[sym]=0.5; return ms


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC ENCODER
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticEncoder(FeatureEncoder):
    def __init__(self):
        import torch, torch.nn as nn
        proj=nn.Linear(7,256,bias=False); nn.init.orthogonal_(proj.weight); proj.eval()
        self._proj=proj; self._rng=np.random.default_rng(0); self._ms=None
        self._prev_price=500.0

    def set_state(self, ms):
        self._ms=ms

    def encode(self, raw_obs):
        if self._ms is None: return self._rng.standard_normal(256).astype("float32")
        import torch; ms=self._ms
        spread=(ms.ask-ms.bid)
        imb=(ms.bid_volume-ms.ask_volume)/(ms.bid_volume+ms.ask_volume+1e-8)
        vol=(ms.bid_volume+ms.ask_volume)/2000.0
        age=min(ms.data_age_ms/1500.0,2.0); rtt=min(ms.rtt_ms/45.0,2.0)
        trend=(ms.price-500.0)/50.0
        momentum=(ms.price-self._prev_price)/max(self._prev_price,1e-8)*100
        self._prev_price=ms.price
        raw=torch.tensor([spread,imb,vol,age,rtt,trend,momentum],dtype=torch.float32)
        with torch.no_grad(): return torch.tanh(self._proj(raw)).numpy().astype("float32")


# ─────────────────────────────────────────────────────────────────────────────
# REPLAY ACTUATOR  (orchestrates everything)
# ─────────────────────────────────────────────────────────────────────────────

class ReplayActuator(Actuator):
    def __init__(self, hud_sock, hud_addr, symbols, vault):
        self._hud_sock=hud_sock; self._hud_addr=hud_addr; self._symbols=symbols
        self._ms=None; self._stale=StaleDataGuard(); self._gov=APIGovernor()
        self._heartbeat=HeartbeatSync(); self._panic=PanicState()
        self.last_result={}; self._tick=0; self._day1_price=None

        self.vault      = vault
        self.positions  = PositionManager(args.stop, args.target)

        # Learning engine — adapts stop/target/size from trade outcomes
        self.learner: Optional[LearningEngine] = None
        if _LEARNING_AVAILABLE and not args.no_learn:
            self.learner = LearningEngine("cortex_learning.json")
            print("  🧠 Adaptive learning engine active")
        self._ohlc      = OHLCBuilder(candle_ticks=4, max_candles=80)
        self._predictor = TrajectoryPredictor(horizon=12)
        self._trade_log: List[dict] = []   # cockpit-ready trade records

    def set_state(self, ms): self._ms=ms

    def act(self, action: np.ndarray, resonance: float, metadata: dict) -> float:
        self._tick+=1; ms=self._ms; signal=float(action[0])
        if ms is None: return 0.0

        # ── Simulation overrides ─────────────────────────────────────────────
        # The engine produces low reservoir resonance (~0.3-0.5) on synthetic
        # data. We override both rho and Z with scenario-appropriate values so
        # every safeguard phase demonstrates its intended behaviour.
        #
        # Scenarios that SHOULD trade:   rho > 0.20, Z > 2.92  → phases pass
        # Scenarios that SHOULD abort:   low rho/Z OR bad market conditions
        #   Phase 2  (RTT)      → GHOST_RTT forces rtt_ms > 45
        #   Phase 4  (Stale)    → STALE_DATA forces data_age_ms > 1500
        #   Phase 5  (PDT)      → PDT_LOCK forces day_trades = 0
        #   Phase 9  (Spread)   → SPREAD_BLOW forces spread > 0.15
        #   Phase 11 (EOD)      → EOD forces seconds_to_close < 300
        #   Phase 13 (VPIN)     → TOXIC_FLOW forces ask/bid > 10x
        #   Phase 14 (Soma)     → CALM/low Z naturally aborts
        #   Phase 17 (Fracture) → FRACTURE forces fleet rho < 0.20

        _SIM_RHO = {
            "SATURATED":    0.96,   # → SNIPER (20%)
            "RECOVERY":     0.93,   # → HUNTER (5%)
            "TRENDING_UP":  0.88,   # → HUNTER (5%)
            "TRENDING_DOWN":0.85,   # → HUNTER short
            "EOD":          0.88,   # passes Ph14, blocked by Ph11
            "PDT_LOCK":     0.90,   # passes Ph14, blocked by Ph5
            "FLASH_CRASH":  0.50,   # Ph10 panic sweep fires
            "CALM":         0.15,   # below Ph15 → HOLD
            "TOXIC_FLOW":   0.75,   # Ph13 VPIN abort
            "SPREAD_BLOW":  0.75,   # Ph9 spread abort
            "STALE_DATA":   0.75,   # Ph4 stale abort
            "GHOST_RTT":    0.75,   # Ph2 RTT abort
            "FRACTURE":     0.10,   # Ph17 systemic abort
        }
        _SIM_Z = {
            "SATURATED":    4.2,
            "RECOVERY":     3.8,
            "TRENDING_UP":  3.3,
            "TRENDING_DOWN":3.1,
            "EOD":          3.3,
            "PDT_LOCK":     3.5,
            "FLASH_CRASH":  5.0,
            "CALM":         1.4,
            "TOXIC_FLOW":   3.0,
            "SPREAD_BLOW":  3.0,
            "STALE_DATA":   3.0,
            "GHOST_RTT":    3.0,
            "FRACTURE":     1.0,
        }
        scenario = ms.scenario_name
        sim_rho   = _SIM_RHO.get(scenario, 0.50)
        z_score   = float(_SIM_Z.get(scenario, 2.0))

        # Add per-tick jitter so values feel live on the cockpit
        jitter_rho = float(np.clip(sim_rho + np.random.randn()*0.02, 0.01, 0.99))
        jitter_z   = float(np.clip(z_score  + np.random.randn()*0.12, 0.01, 7.0))

        # Use jittered rho for cockpit display; use clean values for phase gates
        fleet_rho=dict(ms.fleet_override or {}); fleet_rho[ms.symbol]=jitter_rho
        p17=phase17_systemic_fracture(fleet_rho)
        results=run_pre_execution_audit(
            bid=ms.bid,ask=ms.ask,bid_size=ms.bid_size,ask_size=ms.ask_size,
            ask_volume=ms.ask_volume,bid_volume=ms.bid_volume,
            rtt_ms=ms.rtt_ms,data_age_ms=ms.data_age_ms,
            is_market_open=ms.is_market_open,seconds_to_close=ms.seconds_to_close,
            equity=ms.equity,buying_power=ms.buying_power,
            day_trades_remaining=ms.day_trades,
            z_score=jitter_z, resonance=jitter_rho,
            fleet_resonance=fleet_rho,prior_size=metadata.get("prior_size",0),
            hypothesis_count=metadata.get("hypotheses",0),
            stale_guard=self._stale,api_governor=self._gov,
            panic_state=self._panic,target_alloc_pct=0.01,
        )
        if not p17: results.append(p17)

        abort_phase=0; abort_reason=""
        for r in results:
            if not r: abort_phase=r.phase; abort_reason=r.reason; break

        # Phase 15 uses the clean sim_rho for tier so sizing is deterministic
        p15=phase15_resonance_gate(sim_rho, fleet_rho)
        tier=p15.data.get("tier","HOLD") if abort_phase==0 else "ABORT"
        alloc_pct=p15.data.get("alloc_pct",0.0) if abort_phase==0 else 0.0

        # Expose jittered rho to cockpit so bars feel alive
        resonance = jitter_rho

        da=metadata.get("da",0.5); crt=metadata.get("cortisol",0.2); rpe=metadata.get("rpe",0.0)
        micro=ms.bid+(ms.ask-ms.bid)*0.5   # simple mid as micro-price

        # ── Update OHLC and prediction ──────────────────────────────────────
        vol=float((ms.bid_volume+ms.ask_volume)*0.5)
        self._ohlc.push(ms.price, vol)
        self._predictor.push(ms.price)
        pred_path=self._predictor.predict(ms.price, resonance, da, z_score)

        # ── Check stop/target on existing position ──────────────────────────
        pnl=0.0; trade_event=None; learn_update=None
        if self.positions.position:
            self.positions.update_unrealised(ms.price)
            exit_reason=self.positions.check_exits(ms.price, self._tick)

            # Force close on EOD or fracture
            if ms.seconds_to_close<=0 and ms.is_market_open:
                exit_reason="EOD"
            if abort_phase==17:
                exit_reason="FRACTURE"

            if exit_reason:
                exit_price=ms.bid if self.positions.position.side=='long' else ms.ask
                held_pos=self.positions.position   # capture before close
                trade=self.positions.close(exit_price, exit_reason, self._tick)
                if trade:
                    pnl=trade.pnl
                    self.vault.record_pnl(pnl)
                    trade_event=self._make_trade_event(trade, "EXIT", ms.price)
                    self._print_trade(trade, "EXIT")
                    # ── Feed trade outcome to learning engine ──────────────
                    if self.learner:
                        learn_update=self.learner.update(
                            scenario=ms.scenario_name, pnl=pnl,
                            tier=trade.tier, exit_reason=exit_reason,
                            entry_price=trade.entry_price,
                            exit_price=trade.exit_price,
                            hold_ticks=trade.hold_ticks,
                        )
                        # Apply newly learned stop/target to PositionManager
                        params=self.learner.get_params(ms.scenario_name, trade.tier)
                        self.positions.stop_pct   = params["stop_pct"]
                        self.positions.target_pct = params["target_pct"]
                        self._print_learn(learn_update)

        # ── Entry logic (only when no position open and audit passes) ──────
        if abort_phase==0 and tier in("SCOUT","HUNTER","SNIPER") and not self.positions.position:

            # ── Direction from scenario + signal ─────────────────────────
            # Bullish: price expected to rise
            bullish_scenarios = {"TRENDING_UP", "SATURATED", "CALM"}
            # Bearish: price expected to fall — FLASH_CRASH and FRACTURE go SHORT
            bearish_scenarios = {"TRENDING_DOWN", "FLASH_CRASH", "FRACTURE"}
            # Neutral: follow the raw action signal only (RECOVERY removed from bullish
            #          because 38% WR long proved it is still falling at recovery start)
            if ms.scenario_name in bullish_scenarios:
                side = "long"
            elif ms.scenario_name in bearish_scenarios:
                side = "short"
            else:
                # RECOVERY and unknown scenarios: follow the engine signal
                side = "long" if signal >= 0 else "short"

            # ── Crash/fracture scenarios: cap tier at SCOUT (1% vault max) ──
            # SNIPER (20% vault) into a crash caused -$37k in the 1-day run.
            DANGEROUS_SCENARIOS = {"FLASH_CRASH", "FRACTURE", "TOXIC_FLOW",
                                   "SPREAD_BLOW", "STALE_DATA", "GHOST_RTT"}
            if ms.scenario_name in DANGEROUS_SCENARIOS and tier == "SNIPER":
                tier      = "SCOUT"
                alloc_pct = 0.01
            elif ms.scenario_name in DANGEROUS_SCENARIOS and tier == "HUNTER":
                tier      = "SCOUT"
                alloc_pct = 0.01

            # ── Get learned parameters for this scenario + tier ───────────
            learned_size = 1.0
            skip_entry   = False
            if self.learner:
                params = self.learner.get_params(ms.scenario_name, tier)
                self.positions.stop_pct   = params["stop_pct"]
                self.positions.target_pct = params["target_pct"]
                learned_size              = params["size_scale"]
                # Hard skip: confidence collapsed or learner signals skip
                if params.get("skip") or params["size_scale"] < 0.05:
                    skip_entry = True

            if not skip_entry:
                # Quantity from tier alloc × learned size scale
                dollars = self.vault.alloc_dollars(alloc_pct) * learned_size
                qty     = max(1, int(dollars / max(ms.price, 1)))
                entry_price = ms.ask if side == "long" else ms.bid

                pos = self.positions.open(
                    side, ms.symbol, qty, entry_price, tier, self._tick
                )
                trade_event = self._make_trade_event(pos, "ENTRY", ms.price)
                self._print_entry(pos)

        # ── Signal-driven exit of wrong-direction position ──────────────────
        elif abort_phase==0 and self.positions.position:
            pos=self.positions.position
            # Flip: if holding long but bearish signal fires (or vice versa)
            flip_to_short=(pos.side=="long" and signal<-0.1 and tier in("SCOUT","HUNTER","SNIPER"))
            flip_to_long =(pos.side=="short" and signal>0.1 and tier in("SCOUT","HUNTER","SNIPER"))
            if flip_to_short or flip_to_long:
                exit_price=ms.bid if pos.side=="long" else ms.ask
                trade=self.positions.close(exit_price,"SIGNAL",self._tick)
                if trade:
                    pnl=trade.pnl; self.vault.record_pnl(pnl)
                    trade_event=self._make_trade_event(trade,"EXIT",ms.price)
                    self._print_trade(trade,"EXIT")

        # ── Update unrealised after any entry ───────────────────────────────
        unrealised=self.positions.update_unrealised(ms.price) if self.positions.position else 0.0
        pos_data=None
        if self.positions.position:
            p=self.positions.position
            pos_data={"side":p.side,"qty":p.qty,"entry":round(p.entry_price,2),
                      "stop":round(p.stop_price,2),"target":round(p.target_price,2),
                      "tier":p.tier,"unrealised":round(p.unrealised_pnl,2),
                      "hold_ticks":self._tick-p.entry_tick}

        # ── Ticker stats ─────────────────────────────────────────────────────
        if self._day1_price is None: self._day1_price=ms.price
        day_chg=ms.price-self._day1_price
        day_pct=day_chg/self._day1_price*100

        # ── Build cockpit packet ─────────────────────────────────────────────
        z_vec=list(np.tanh(np.random.randn(128)*resonance).astype(float))
        candles=self._ohlc.candles(); live=self._ohlc.live_bar()
        if live: candles=candles+[live]

        # Recent trades for cockpit log (keep last 20)
        if trade_event: self._trade_log.insert(0,trade_event)
        if len(self._trade_log)>20: self._trade_log.pop()

        packet={
            "SYM":ms.symbol,"P":round(ms.price,2),
            "R":round(resonance,4),"Z":round(z_score,4),
            "Z_VEC":z_vec,"TIER":tier,
            "ABORT":abort_phase,"ABORT_REASON":abort_reason[:60],
            "SCENARIO":ms.scenario_name,"MODE":ms.scenario_name,
            "FLEET":{k:round(v,4) for k,v in fleet_rho.items() if k!=ms.symbol},
            "DA":round(da,4),"CRT":round(crt,4),"RPE":round(rpe,4),
            "QTY":pos_data["qty"] if pos_data else 0,
            "VAULT":{
                "total_equity":round(self.vault.total_equity,2),
                "active_pool":round(self.vault.active_pool,2),
                "banked":round(self.vault.banked,2),
                "gain_pct":round(self.vault.gain_pct(),3),
                "unrealised":round(unrealised,2),
            },
            "POSITION":pos_data,
            "TRADES":self._trade_log[:10],
            "STATS":{
                "total_pnl":round(self.positions.total_pnl,2),
                "win_rate":round(self.positions.win_rate*100,1),
                "trade_count":self.positions.trade_count,
                "wins":self.positions.win_count,
                "losses":self.positions.loss_count,
            },
            "LEARN": learn_update or (self.learner.get_params(ms.scenario_name,"SCOUT") if self.learner else {}),
            "OHLC":candles[-60:],
            "PRED":pred_path,
            "TICKER":{"price":round(ms.price,2),"change":round(day_chg,2),
                      "change_pct":round(day_pct,3),"volume":round(vol,0)},
            "TS":time.time(),"TICK":self._tick,
        }
        self.last_result=packet

        # ── Terminal output ───────────────────────────────────────────────────
        pos_str=""
        if self.positions.position:
            p=self.positions.position
            upnl=p.unrealised_pnl
            pos_str=f"  [{p.side.upper()} {p.qty}sh @${p.entry_price:.2f} uPnL={'+'if upnl>=0 else ''}{upnl:.2f}]"
        status="⚡" if tier in("SCOUT","HUNTER","SNIPER") else "🛡️" if abort_phase else "⬜"
        tier_str=f"[{tier:7s}]" if abort_phase==0 else f"[Ph{abort_phase:2d} ABORT]"
        print(f"  {status} t={self._tick:4d}  {ms.scenario_name:<14}"
              f"  rho={resonance:.4f}  Z={z_score:.3f}s  {tier_str}"
              f"  DA={da:.3f}  CRT={crt:.3f}"
              f"  vault=${self.vault.total_equity:,.0f}"
              f"  pnl=${self.positions.total_pnl:+.2f}  wr={self.positions.win_rate*100:.0f}%"
              f"{pos_str}")

        if self._hud_sock and self._hud_addr:
            try: self._hud_sock.sendto(json.dumps(packet).encode(), self._hud_addr)
            except Exception: pass

        return pnl

    def _make_trade_event(self, trade_or_pos, event_type: str, price: float) -> dict:
        if event_type=="ENTRY":
            pos=trade_or_pos
            return {"type":"ENTRY","side":pos.side,"qty":pos.qty,
                    "price":round(pos.entry_price,2),"tier":pos.tier,
                    "stop":round(pos.stop_price,2),"target":round(pos.target_price,2),
                    "time":time.time(),"pnl":None}
        else:
            t=trade_or_pos
            return {"type":"EXIT","side":t.side,"qty":t.qty,
                    "price":round(t.entry_price,2),
                    "exit_price":round(t.exit_price,2),
                    "tier":t.tier,
                    "reason":t.exit_reason,"pnl":round(t.pnl,2),
                    "pnl_pct":round(t.pnl_pct,3),"hold":t.hold_ticks,
                    "time":time.time()}

    def _broadcast(self, price: float, resonance: float, z_score: float) -> None:
        """Send a minimal telemetry packet to the cockpit (no trade event)."""
        if not (self._hud_sock and self._hud_addr and self._ms):
            return
        ms = self._ms
        try:
            self._hud_sock.sendto(json.dumps({
                "SYM": ms.symbol, "P": round(price, 2),
                "R": round(resonance, 4), "Z": round(z_score, 4),
                "TIER": "HOLD", "ABORT": 0,
                "SCENARIO": ms.scenario_name, "MODE": ms.scenario_name,
                "FLEET": {k: round(v, 4) for k, v in (ms.fleet_override or {}).items()},
                "DA": 0.5, "CRT": 0.2, "RPE": 0.0, "QTY": 0,
                "VAULT": {"total_equity": round(self.vault.total_equity, 2),
                          "active_pool":  round(self.vault.active_pool, 2),
                          "banked":       round(self.vault.banked, 2),
                          "gain_pct":     round(self.vault.gain_pct(), 3),
                          "unrealised":   0.0},
                "POSITION": None, "TRADES": [], "STATS": {},
                "OHLC": [], "PRED": [], "LEARN": {},
                "TICKER": {"price": round(price, 2), "change": 0, "change_pct": 0, "volume": 0},
                "TS": time.time(), "TICK": self._tick,
            }).encode(), self._hud_addr)
        except Exception:
            pass

    def _print_learn(self, u: dict) -> None:
        if not u: return
        arrows = lambda v: f"{'▲' if v>0.0001 else '▼' if v<-0.0001 else '='}"
        print(f"  🧠 [{u['scenario']}]  "
              f"conf={u['confidence']:.3f}  "
              f"stop={u['stop_pct']:.2f}%{arrows(u['stop_delta'])}  "
              f"tgt={u['target_pct']:.2f}%{arrows(u['target_delta'])}  "
              f"size={u['size_scale']:.2f}x{arrows(u['size_delta'])}  "
              f"RPE={u['da_rpe']:+.3f}  "
              f"exp={u['expectancy']:+.2f}"
              + ("  ⚠DEGRADED" if u.get('degraded') else ""))

    def _print_entry(self, pos):
        print(f"\n  {'▲' if pos.side=='long' else '▼'} ENTRY [{pos.tier}] "
              f"{pos.side.upper()} {pos.qty}sh @ ${pos.entry_price:.2f}  "
              f"SL:${pos.stop_price:.2f}  TP:${pos.target_price:.2f}\n")

    def _print_trade(self, trade, event_type):
        icon="✅" if trade.pnl>=0 else "❌"
        print(f"\n  {icon} EXIT [{trade.exit_reason}] {trade.side.upper()} "
              f"{trade.qty}sh @ ${trade.exit_price:.2f}  "
              f"PnL: {'+'if trade.pnl>=0 else''}{trade.pnl:.2f} "
              f"({trade.pnl_pct:+.2f}%)  held {trade.hold_ticks} ticks\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def _save_session(actuator, vault, start_time: float, label: str) -> str:
    pm = actuator.positions
    ts = time.strftime("%Y%m%d_%H%M%S")
    name = f"{label}_{ts}" if label else f"session_{ts}"
    path = f"sessions/{name}.json"
    os.makedirs("sessions", exist_ok=True)

    # Per-scenario stats
    scenario_stats = {}
    if actuator.learner:
        for sname, s in actuator.learner._scenarios.items():
            if s.trades > 0:
                scenario_stats[sname] = {
                    "trades":s.trades,"wins":s.wins,"losses":s.losses,
                    "win_rate":round(s.win_rate*100,1),"total_pnl":round(s.total_pnl,2),
                    "avg_win":round(s.avg_win,2),"avg_loss":round(s.avg_loss,2),
                    "stop_pct":round(s.stop_pct*100,3),"target_pct":round(s.target_pct*100,3),
                    "size_scale":round(s.size_scale,3),"confidence":round(s.confidence,3),
                    "expectancy":round(s.expectancy,2),"degraded":s.degraded(),
                }

    # Equity curve
    equity_curve=[]; running=vault.seed
    for t in pm.closed:
        running+=t.pnl; equity_curve.append(round(running,2))

    # Tier breakdown
    tier_stats: dict = {}
    for t in pm.closed:
        d=tier_stats.setdefault(t.tier,{"trades":0,"wins":0,"total_pnl":0.0,"pnl_list":[]})
        d["trades"]+=1; d["total_pnl"]=round(d["total_pnl"]+t.pnl,2); d["pnl_list"].append(t.pnl)
        if t.pnl>0: d["wins"]+=1
    for tier,d in tier_stats.items():
        n=d["trades"]; d["win_rate"]=round(d["wins"]/max(n,1)*100,1)
        d["avg_pnl"]=round(d["total_pnl"]/max(n,1),2)
        d["max_win"]=round(max(d["pnl_list"]),2) if d["pnl_list"] else 0
        d["max_loss"]=round(min(d["pnl_list"]),2) if d["pnl_list"] else 0
        del d["pnl_list"]

    # Exit reason breakdown
    exit_stats: dict = {}
    for t in pm.closed:
        d=exit_stats.setdefault(t.exit_reason,{"count":0,"total_pnl":0.0})
        d["count"]+=1; d["total_pnl"]=round(d["total_pnl"]+t.pnl,2)

    record = {
        "session_name":  name,
        "label":         label or "unlabelled",
        "timestamp":     ts,
        "duration_min":  round((time.time()-start_time)/60,2),
        "hz":            args.hz,
        "ticks":         actuator._tick,
        "vault_seed":    vault.seed,
        "vault_final":   round(vault.total_equity,2),
        "active_pool":   round(vault.active_pool,2),
        "banked":        round(vault.banked,2),
        "gain_pct":      round(vault.gain_pct(),3),
        "total_pnl":     round(pm.total_pnl,2),
        "trades":        pm.trade_count,
        "wins":          pm.win_count,
        "losses":        pm.loss_count,
        "win_rate":      round(pm.win_rate*100,1),
        "scenarios":     scenario_stats,
        "tiers":         tier_stats,
        "exit_reasons":  exit_stats,
        "equity_curve":  equity_curve,
    }
    with open(path,"w") as f:
        json.dump(record,f,indent=2)
    return path


def main():
    hud_sock=hud_addr=None
    if not args.no_hud:
        hud_sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        hud_addr=(args.hud_ip,args.hud_port)
        print(f"  Sending to {args.hud_ip}:{args.hud_port}")

    vault=VaultState(args.vault)
    encoder=SyntheticEncoder()
    actuator=ReplayActuator(hud_sock,hud_addr,args.symbols,vault)

    cfg=EngineConfig(
        input_dim=256,latent_dim=128,action_dim=2,
        lsm=LSMConfig(input_dim=256,reservoir_dim=512,output_dim=128,spectral_radius=0.9,leak_rate=0.3),
        jepa=EBJEPAConfig(latent_dim=128,compressed_dim=16,aux_dim=3,num_candidates=32,
                          planning_horizon=3,sandwich_norm=True),
        router=CSRRouterConfig(input_dim=256,manifold_dim=128,num_experts=4,sparsity=0.80),
        ttm=TTMConfig(manifold_dim=128,max_episodic=500,max_long_term=1000,
                      surprise_rho_threshold=0.90,confidence_threshold=0.85),
        npu=NPUConfig(),dopamine=DopamineConfig(da_tonic=0.5,crt_tonic=0.2,da_lr=0.25),
        use_npu=False,use_dopamine=True,
    )
    engine=CortexEngine(config=cfg,feature_encoder=encoder,actuator=actuator)
    scene=ScenarioEngine(args.symbols)
    gen=scene.forced(args.scenario) if args.scenario else scene.cycling()
    interval=1.0/args.hz

    duration_s = args.duration*60 if args.duration else None
    start_time = time.time()
    label      = args.session_name or (f"{int(args.duration)}min" if args.duration else None)
    dur_str    = f"{args.duration:.0f} min  →  auto-stop" if args.duration else "∞  Ctrl+C to stop"

    print(f"""
  ╔══════════════════════════════════════════════════════════════╗
  ║  CORTEX-16 v6.0  |  REPLAY + LEARNING                      ║
  ║  Hz:{args.hz:<6.1f}  SL:{args.stop*100:.1f}%  TP:{args.target*100:.1f}%  Vault:${args.vault:,.0f}  ║
  ║  Duration : {dur_str:<50}║
  ║  Session  : {(label or 'unlabelled'):<50}║
  ╚══════════════════════════════════════════════════════════════╝
  ▲=Long  ▼=Short  ✅=Win  ❌=Loss  🧠=Learn  ⚠=Degraded
""")

    def _finish():
        engine.stop()
        pm=actuator.positions; v=actuator.vault
        elapsed=time.time()-start_time
        print(f"""
  ── SESSION COMPLETE ── {(label or 'manual')} ── {elapsed/60:.1f} min ──────────────
  Ticks     : {actuator._tick}
  Trades    : {pm.trade_count}  (W:{pm.win_count} L:{pm.loss_count}  WR:{pm.win_rate*100:.1f}%)
  Total PnL : ${pm.total_pnl:+,.2f}
  Vault     : ${v.total_equity:,.2f}  (Banked: ${v.banked:,.2f})
  Gain      : {v.gain_pct():+.3f}%
  ─────────────────────────────────────────────────────────────""")
        if actuator.learner:
            print(actuator.learner.report())
        path=_save_session(actuator,v,start_time,label)
        print(f"  💾 Saved  → {path}")
        print(f"  📊 Compare → python compare_sessions.py\n")

    try:
        while True:
            if duration_s and (time.time()-start_time)>=duration_s:
                print(f"\n  ⏱  {args.duration:.0f} min reached — auto-stopping.")
                break
            t0=time.perf_counter(); ms=next(gen)
            encoder.set_state(ms); actuator.set_state(ms)
            engine.tick(ms)
            elapsed_tick=time.perf_counter()-t0
            sleep_t=max(0.0,interval-elapsed_tick)
            if sleep_t>0: time.sleep(sleep_t)
    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        _finish()

if __name__=="__main__":
    main()

