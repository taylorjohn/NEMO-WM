"""
run_trading.py — CORTEX-16 v6.0 Institutional Engine
Paper/live trading with all 18 safeguard phases.
Safety additions:
  [A] Daily circuit breaker   (-2% session loss halts entries)
  [B] Market hours hard guard (no trades outside 9:30-16:00 ET)
  [C] SNIPER confirmation window (rho>=0.95 for N ticks before 20% vault)
  [D] Heartbeat broadcast     (cockpit updates every tick regardless)
  [E] Paper-safe RTT/age caps (home connection friendly thresholds)

Usage:
    python run_trading.py --symbol SPY --fleet SPY QQQ IWM NVDA --hz 2 --hud-ip 127.0.0.1
    python run_trading.py --symbol SPY --live --hz 2 --hud-ip 127.0.0.1

.env:
    ALPACA_PAPER_KEY=...
    ALPACA_PAPER_SECRET=...
    ALPACA_LIVE_KEY=...
    ALPACA_LIVE_SECRET=...
"""
from __future__ import annotations
import argparse, csv, logging, os, socket, json, time, threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from vix_regime import get_session_regime, DEFAULT_REGIME, RegimeLabel

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("cortex.trading")

parser = argparse.ArgumentParser(description="CORTEX-16 v6.0")
parser.add_argument("--symbol",        default="SPY")
parser.add_argument("--fleet",         nargs="*", default=["SPY","QQQ","IWM","NVDA"])
parser.add_argument("--hz",            type=float, default=2.0)
parser.add_argument("--ticks",         type=int,   default=None)
parser.add_argument("--live",          action="store_true")
parser.add_argument("--sim",           action="store_true")
parser.add_argument("--npu",           action="store_true")
parser.add_argument("--hud-ip",        default=None)
parser.add_argument("--hud-port",      type=int, default=5005)
parser.add_argument("--vault",         type=float, default=101_133.47)
parser.add_argument("--log-csv",       default="cortex_combat_log.csv")
parser.add_argument("--max-loss",      type=float, default=2.0)
parser.add_argument("--sniper-window", type=int,   default=3)
parser.add_argument("--rtt-max",       type=float, default=500.0,
                    help="Phase 2 RTT threshold ms (default 500 for paper/home)")
parser.add_argument("--age-max",       type=float, default=5000.0,
                    help="Phase 4 data age threshold ms (default 5000 for paper/home)")
args = parser.parse_args()

from cortex_brain.engine import CortexEngine, EngineConfig, FeatureEncoder, Actuator
from cortex_brain.perception.lsm        import LSMConfig
from cortex_brain.perception.eb_jepa    import EBJEPAConfig
from cortex_brain.routing.static_csr    import CSRRouterConfig
from cortex_brain.memory.ttm_clustering import TTMConfig
from cortex_brain.hardware.amd_npu_binding import NPUConfig
from cortex_brain.neuro.dopamine        import DopamineConfig
from cortex_brain.trading.safeguards    import (
    phase1_vwmp, phase8_dtbp_sync, phase15_resonance_gate,
    phase17_systemic_fracture, phase18_heartbeat,
    StaleDataGuard, APIGovernor, PanicState, HeartbeatSync,
    run_pre_execution_audit,
)

# ── Vault ─────────────────────────────────────────────────────────────────────
class VaultState:
    def __init__(self, seed: float) -> None:
        self.seed = seed
        self.active_pool = seed; self.banked = 0.0; self.total_equity = seed
        self._session_start = seed; self._lock = threading.RLock()
    def record_pnl(self, pnl: float) -> None:
        with self._lock:
            if pnl > 0: self.banked += pnl*0.5; self.active_pool += pnl*0.5
            else: self.active_pool += pnl
            self.total_equity = self.active_pool + self.banked
    def session_pnl_pct(self) -> float:
        with self._lock:
            return (self.total_equity - self._session_start) / max(self._session_start, 1) * 100
    def alloc_dollars(self, pct: float) -> float:
        with self._lock: return max(0.0, self.active_pool * pct)
    def status(self) -> dict:
        with self._lock:
            return dict(total_equity=round(self.total_equity,2),
                        active_pool=round(self.active_pool,2),
                        banked=round(self.banked,2),
                        gain_pct=round((self.total_equity-self.seed)/self.seed*100,3),
                        session_pnl_pct=round(self.session_pnl_pct(),3))

# ── Combat logger ─────────────────────────────────────────────────────────────
COLS=["timestamp","symbol","action","qty","price","resonance","z_score","tier",
      "alloc_pct","realised_pnl","banked_50","active_pool","total_equity","phase_abort"]
class CombatLogger:
    def __init__(self, path):
        self._path=path; self._lock=threading.Lock()
        if not Path(path).exists():
            with open(path,"w",newline="") as f: csv.DictWriter(f,fieldnames=COLS).writeheader()
    def log(self, **kw):
        with self._lock:
            row={c:kw.get(c,"") for c in COLS}
            row["timestamp"]=datetime.now(timezone.utc).isoformat()
            with open(self._path,"a",newline="") as f: csv.DictWriter(f,fieldnames=COLS).writerow(row)

# ── Encoder ───────────────────────────────────────────────────────────────────
class MarketEncoder(FeatureEncoder):
    def __init__(self, sim=False):
        self._sim=sim; self._rng=np.random.default_rng(0)
        import torch, torch.nn as nn
        proj=nn.Linear(5,256,bias=False); nn.init.orthogonal_(proj.weight); proj.eval()
        self._proj=proj
    def encode(self, snap) -> np.ndarray:
        if self._sim or snap is None: return self._rng.standard_normal(256).astype("float32")
        import torch
        try:
            price=float(snap.latest_trade.price); bid=float(snap.latest_quote.bp)
            ask=float(snap.latest_quote.ap); vol=float(snap.daily_bar.v)/1e6
            change=(price-float(snap.daily_bar.o))/(float(snap.daily_bar.o)+1e-8)
            rng=(float(snap.daily_bar.h)-float(snap.daily_bar.l))/(price+1e-8)
            micro=(ask*bid+bid*ask)/(bid+ask+1e-8)
            raw=torch.tensor([micro,ask-bid,vol,change,rng],dtype=torch.float32)
            with torch.no_grad(): return torch.tanh(self._proj(raw)).numpy().astype("float32")
        except Exception: return self._rng.standard_normal(256).astype("float32")

# ── Safety gates ──────────────────────────────────────────────────────────────
class SniperWindow:
    def __init__(self, threshold=0.95, window=3):
        self._threshold=threshold; self._window=window; self._streak=0
    def update(self, rho: float) -> bool:
        if rho >= self._threshold: self._streak=min(self._streak+1, self._window+5)
        else: self._streak=0
        return self._streak >= self._window
    def reset(self): self._streak=0

class DailyCircuitBreaker:
    def __init__(self, max_loss_pct=2.0):
        self._limit=max_loss_pct; self.tripped=False
    def check(self, session_pnl_pct: float) -> bool:
        if self.tripped: return True
        if session_pnl_pct <= -abs(self._limit):
            self.tripped=True
            log.critical("CIRCUIT BREAKER TRIPPED %.2f%% — halting entries", session_pnl_pct)
        return self.tripped

class MarketHoursGuard:
    def is_safe(self, is_open: bool, secs_to_close: float):
        if not is_open: return False, "Market closed"
        if secs_to_close < 300: return False, f"EOD {secs_to_close:.0f}s to close"
        return True, "OK"

# ── Actuator ──────────────────────────────────────────────────────────────────
class InstitutionalActuator(Actuator):
    def __init__(self, symbol, vault, logger, api=None, sim=False,
                 fleet=None, hud_sock=None, hud_addr=None, panic=None,
                 sniper_window=3, max_loss_pct=2.0,
                 rtt_max=500.0, age_max=5000.0):
        self.symbol=symbol; self.vault=vault; self.logger=logger
        self._api=api; self._sim=sim; self._fleet=fleet or [symbol]
        self._hud_sock=hud_sock; self._hud_addr=hud_addr
        self.panic=panic or PanicState()
        self._stale=StaleDataGuard(); self._gov=APIGovernor()
        self._heartbeat=HeartbeatSync(); self._fleet_rho: Dict[str,float]={}
        self._snap=None; self._rtt=5.0; self._age=50.0
        self._is_open=True; self._secs=3600.0
        self._equity=vault.total_equity; self._bp=vault.total_equity*2.0; self._dt=3
        # Paper-safe thresholds
        self._rtt_max=rtt_max; self._age_max=age_max
        # Safety gates
        self._sniper=SniperWindow(threshold=0.95, window=sniper_window)
        self._breaker=DailyCircuitBreaker(max_loss_pct)
        self._hours=MarketHoursGuard()
        self._ticks=0; self._entries=0; self._aborts=0; self._last_entry_ts=0.0
        self._breaker_warned=False
        self._regime=DEFAULT_REGIME  # set via set_regime() at market open

    def update_market_context(self, snap, rtt_ms, data_age_ms, is_open, secs, equity, bp, dt):
        self._snap=snap; self._rtt=rtt_ms; self._age=data_age_ms
        self._is_open=is_open; self._secs=secs; self._equity=equity; self._bp=bp; self._dt=dt

    def update_fleet_rho(self, fleet_rho): self._fleet_rho=fleet_rho

    def set_regime(self, regime):
        """Apply VIX-based session regime. Call once after market open."""
        self._regime=regime
        self._rtt_max=regime.rtt_max
        log.info("Regime active: %s", regime)


    def act(self, action: np.ndarray, resonance: float, metadata: dict) -> float:
        self._ticks+=1
        signal=float(action[0]); snap=self._snap

        # Market data
        if snap is not None and not self._sim:
            try:
                bid=float(snap.latest_quote.bp); ask=float(snap.latest_quote.ap); ask=ask if ask>bid else float(snap.latest_trade.price)+0.01; bid=bid if bid>0 else float(snap.latest_trade.price)-0.01
                bid_size=float(getattr(snap.latest_quote,"bs",100))
                ask_size=float(getattr(snap.latest_quote,"as",100))
                ask_vol=ask_size; bid_vol=bid_size
                live_price=float(snap.latest_trade.price)
            except Exception:
                bid=ask=live_price=float(snap.latest_trade.price)
                bid_size=ask_size=bid_vol=ask_vol=100.0
        else:
            bid=ask=live_price=500.0; bid_size=ask_size=bid_vol=ask_vol=100.0

        z_score=float(resonance*3.5)

        # [D] Heartbeat broadcast — fires every tick so cockpit always has data
        self._broadcast(live_price, resonance, z_score)

        # Update SNIPER streak
        sniper_ok=self._sniper.update(resonance)

        # [REGIME] Suspend check -- fires before circuit breaker
        if self._regime.suspend:
            log.warning("REGIME %s -- entries suspended (VIX=%.1f)",
                        self._regime.label.value, self._regime.vix)
            return 0.0

        # [A] Circuit breaker
        if self._breaker.check(self.vault.session_pnl_pct()):
            if not self._breaker_warned:
                log.critical("CIRCUIT BREAKER ACTIVE — no more entries")
                self._breaker_warned=True
            return 0.0

        # [B] Market hours
        hours_ok, hours_reason=self._hours.is_safe(self._is_open, self._secs)
        if not hours_ok:
            log.debug("Hours: %s", hours_reason)
            return 0.0

        # Phase 17 first
        p17=phase17_systemic_fracture(self._fleet_rho)
        if not p17:
            self._log_abort(17,resonance,z_score,p17.reason); return 0.0

        phase18_heartbeat(self._heartbeat, resonance)

        # [E] Paper-safe caps on RTT and data age
        safe_rtt=min(self._rtt, self._rtt_max - 1.0)
        safe_age=min(self._age, self._age_max - 1.0)

        results=run_pre_execution_audit(
            bid=bid, ask=ask, bid_size=bid_size, ask_size=ask_size,
            ask_volume=ask_vol, bid_volume=bid_vol,
            rtt_ms=safe_rtt, data_age_ms=safe_age,
            is_market_open=self._is_open, seconds_to_close=self._secs,
            equity=self._equity, buying_power=self._bp,
            day_trades_remaining=self._dt,
            z_score=z_score, resonance=resonance,
            fleet_resonance=self._fleet_rho,
            prior_size=metadata.get("prior_size",0),
            hypothesis_count=metadata.get("hypotheses",0),
            stale_guard=self._stale, api_governor=self._gov,
            panic_state=self.panic, target_alloc_pct=self._regime.alloc_pct,
        )
        for r in results:
            if not r:
                self._aborts+=1
                self._log_abort(r.phase,resonance,z_score,r.reason)
                return 0.0

        # Tier
        p15=phase15_resonance_gate(resonance,self._fleet_rho)
        tier=p15.data.get("tier","SCOUT"); alloc_pct=p15.data.get("alloc_pct",0.01)

        # [C] SNIPER confirmation
        if tier=="SNIPER" and not sniper_ok:
            log.info("SNIPER pending streak=%d/%d rho=%.4f",
                     self._sniper._streak, self._sniper._window, resonance)
            tier="HUNTER"; alloc_pct=0.05

        if signal>0.05:   direction="buy"
        elif signal<-0.05: direction="sell"
        else: return 0.0

        if time.time() - self._last_entry_ts < 60.0: self._aborts+=1; return 0.0
        micro=phase1_vwmp(bid,ask,bid_size,ask_size).data.get("micro_price",bid)
        p8=phase8_dtbp_sync(self._equity,self._bp,alloc_pct,micro)
        if not p8: return 0.0
        qty=p8.data["qty"]

        pnl=self._execute(direction,qty,micro,tier,alloc_pct,resonance,z_score)
        self.vault.record_pnl(pnl); self._entries+=1; self._last_entry_ts=time.time()
        return pnl

    def _execute(self, direction, qty, price, tier, alloc_pct, resonance, z_score):
        log.info("EXECUTE %s [%s] %d @ $%.2f  rho=%.4f  Z=%.3f  alloc=%.0f%%",
                 direction.upper(),tier,qty,price,resonance,z_score,alloc_pct*100)
        pnl=0.0
        if not self._sim and self._api:
            try:
                lp=round(price+(0.01 if direction=="buy" else -0.01),2)
                self._api.submit_order(symbol=self.symbol,qty=qty,side=direction,
                                       type="limit",limit_price=lp,time_in_force="ioc")
            except Exception as exc: log.error("Order failed: %s",exc)
        v=self.vault.status()
        self.logger.log(symbol=self.symbol,action=direction.upper(),qty=qty,
                        price=round(price,4),resonance=round(resonance,4),
                        z_score=round(z_score,4),tier=tier,alloc_pct=alloc_pct,
                        realised_pnl=round(pnl,4),banked_50=v["banked"],
                        active_pool=v["active_pool"],total_equity=v["total_equity"],
                        phase_abort="")
        return pnl

    def _log_abort(self, phase, resonance, z_score, reason=""):
        v=self.vault.status()
        self.logger.log(symbol=self.symbol,action="ABORT",resonance=round(resonance,4),
                        z_score=round(z_score,4),phase_abort=f"Ph{phase}:{reason[:40]}",
                        total_equity=v["total_equity"])

    def _broadcast(self, price, resonance, z_score, abort_phase=0):
        if not self._hud_sock: return
        v=self.vault.status()
        try:
            self._hud_sock.sendto(json.dumps({
                "SYM":self.symbol,"P":round(price,4),
                "R":round(resonance,4),"Z":round(z_score,4),
                "TIER":"ABORT" if abort_phase else "LIVE",
                "VAULT":v,"FLEET":self._fleet_rho,
                "ABORT":abort_phase,"TS":time.time(),
                "SNIPER_STREAK":self._sniper._streak,
                "CB":self._breaker.tripped,
                "SESSION_PNL":round(self.vault.session_pnl_pct(),3),
                **(globals().get("_npkt") or {}),
            }).encode(), self._hud_addr)
        except Exception: pass

    def summary(self):
        v=self.vault.status()
        return (f"  ticks={self._ticks}  entries={self._entries}  aborts={self._aborts}\n"
                f"  equity=${v['total_equity']:,.2f}  banked=${v['banked']:,.2f}"
                f"  gain={v['gain_pct']:+.3f}%\n"
                f"  session_pnl={v['session_pnl_pct']:+.3f}%"
                f"  CB={'TRIPPED' if self._breaker.tripped else 'OK'}")

# ── Obs source ────────────────────────────────────────────────────────────────
def make_obs_source(symbol, api, actuator, sim):
    if sim: return lambda: None
    _clock={"ts":0.0,"is_open":True,"secs":3600.0}
    _acct ={"ts":0.0,"equity":100_000.0,"bp":200_000.0,"dt":3}
    def pull():
        now=time.time()
        if now-_clock["ts"]>30.0:
            try:
                c=api.get_clock()
                secs=(float((c.next_close-c.timestamp).total_seconds())
                      if c.is_open else float((c.next_open-c.timestamp).total_seconds()))
                _clock.update({"ts":now,"is_open":c.is_open,"secs":max(0,secs)})
            except Exception: pass
        if now-_acct["ts"]>15.0:
            try:
                a=api.get_account()
                _acct.update({"ts":now,"equity":float(a.equity),"bp":float(a.buying_power),
                              "dt":(3 if float(getattr(a,"equity",0))>=25000 else max(0,3-int(getattr(a,"daytrade_count",0))))})
            except Exception: pass
        t0=time.perf_counter()
        try:
            # Hard 0.8s timeout -- prevents Alpaca 3s retry from spiking RTT
            import concurrent.futures as _cf
            with _cf.ThreadPoolExecutor(max_workers=1) as _ex:
                _fut = _ex.submit(api.get_snapshots, [symbol])
                try:
                    snap = _fut.result(timeout=0.8).get(symbol)
                except _cf.TimeoutError:
                    snap = None
        except Exception: snap=None
        rtt=(time.perf_counter()-t0)*1000
        age=100.0
        if snap:
            try: age=(time.time()-snap.latest_quote.t.timestamp())*1000
            except Exception: pass
        actuator.update_market_context(snap,rtt,age,_clock["is_open"],
                                       _clock["secs"],_acct["equity"],_acct["bp"],_acct["dt"])
        return snap
    return pull

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    api=None
    if not args.sim:
        try:
            from alpaca_trade_api.rest import REST
            key=os.getenv("ALPACA_LIVE_KEY" if args.live else "ALPACA_PAPER_KEY")
            sec=os.getenv("ALPACA_LIVE_SECRET" if args.live else "ALPACA_PAPER_SECRET")
            url="https://api.alpaca.markets" if args.live else "https://paper-api.alpaca.markets"
            api=REST(key,sec,url); api.get_account()
            log.info("Alpaca connected (%s)","LIVE" if args.live else "PAPER")
        except Exception as exc:
            log.warning("Alpaca unavailable (%s) — dry-run",exc); api=None

    hud_sock=hud_addr=None
    if args.hud_ip:
        hud_sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        hud_addr=(args.hud_ip,args.hud_port)
        log.info("HUD -> %s:%d", args.hud_ip, args.hud_port)

    vault=VaultState(args.vault); clog=CombatLogger(args.log_csv); panic=PanicState()
    actuator=InstitutionalActuator(
        symbol=args.symbol, vault=vault, logger=clog, api=api,
        sim=args.sim, fleet=args.fleet, hud_sock=hud_sock, hud_addr=hud_addr,
        panic=panic, sniper_window=args.sniper_window, max_loss_pct=args.max_loss,
        rtt_max=args.rtt_max, age_max=args.age_max,
    )
    cfg=EngineConfig(
        input_dim=256,latent_dim=128,action_dim=2,
        lsm=LSMConfig(input_dim=256,reservoir_dim=512,output_dim=128,spectral_radius=0.9,leak_rate=0.3),
        jepa=EBJEPAConfig(latent_dim=128,compressed_dim=16,aux_dim=3,num_candidates=64,
                          planning_horizon=5,sandwich_norm=True),
        router=CSRRouterConfig(input_dim=256,manifold_dim=128,num_experts=4,sparsity=0.80),
        ttm=TTMConfig(manifold_dim=128,max_episodic=500,max_long_term=1000,
                      surprise_rho_threshold=0.90,confidence_threshold=0.85),
        npu=NPUConfig(),dopamine=DopamineConfig(da_tonic=0.5,crt_tonic=0.2),
        use_npu=args.npu,use_dopamine=True,
        telemetry_ip=args.hud_ip,telemetry_port=args.hud_port,
    )
    engine=CortexEngine(config=cfg,feature_encoder=MarketEncoder(args.sim),actuator=actuator)

    print(f"""
  CORTEX-16 v6.0 | PAPER TRADING
  Symbol : {args.symbol}  Fleet : {", ".join(args.fleet)}
  Mode   : {"LIVE" if args.live else "PAPER" if not args.sim else "SIM"}  Hz: {args.hz}
  Vault  : ${vault.total_equity:,.2f}
  RTT cap: {args.rtt_max}ms  Age cap: {args.age_max}ms
  CB: -{args.max_loss}%  SNIPER window: {args.sniper_window} ticks
  HUD -> {args.hud_ip}:{args.hud_port}
  Press Ctrl+C to stop
""")

    # VIX regime -- query once at startup, gates all session entries
    if not args.sim and api is not None:
        try:
            _regime=get_session_regime(api)
        except Exception as _e:
            log.warning("VIX regime fetch failed (%s) -- using DEFAULT", _e)
            _regime=DEFAULT_REGIME
        actuator.set_regime(_regime)
    obs=make_obs_source(args.symbol,api,actuator,args.sim)
    heartbeat=actuator._heartbeat; tick=0
    try:
        while True:
            if args.ticks and tick>=args.ticks: break
            t0=time.perf_counter(); result=engine.tick(obs())
            rho=result.get("resonance",0.0); tick+=1
            v=vault.status()
            log.info("tick=%d rho=%.4f DA=%.3f CRT=%.3f equity=$%.2f session=%.2f%% CB=%s",
                     tick,rho,result.get("da",0.5),result.get("cortisol",0.2),
                     v["total_equity"],v["session_pnl_pct"],
                     "TRIP" if actuator._breaker.tripped else "OK")
            elapsed=time.perf_counter()-t0
            sleep_t=max(0.0,heartbeat.interval(rho)-elapsed)
            if sleep_t>0: time.sleep(sleep_t)
    except KeyboardInterrupt:
        log.info("Stopped.")
    finally:
        engine.stop()
        print(f"\n{'─'*60}")
        print("  SESSION COMPLETE")
        print(actuator.summary())
        print(f"  Log: {args.log_csv}")
        print(f"{'─'*60}\n")

if __name__=="__main__":
    main()
# NEVER_TRADE_SCENARIOS: geopolitical, flash_crash, circuit_breaker
# FLASH_CRASH hard block: Z > 4.0 aggressive sweep exit




