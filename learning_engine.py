"""
cortex_brain/trading/learning_engine.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Online learning layer for the CORTEX-16 trading engine.

Learns from every closed trade to adapt:
  1. Per-scenario stop/target widths  (gradient descent on Sharpe)
  2. Per-scenario position size scale (TTM confidence gating)
  3. Dopamine RPE feedback            (modulates next entry aggression)
  4. Tier confidence thresholds       (raise bar after consecutive losses)

All state persists to JSON so learning carries across sessions.
"""
from __future__ import annotations
import json, math, time, os
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# PER-SCENARIO MEMORY
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScenarioStats:
    name:         str
    trades:       int   = 0
    wins:         int   = 0
    losses:       int   = 0
    total_pnl:    float = 0.0
    avg_win:      float = 0.0
    avg_loss:     float = 0.0
    stop_pct:     float = 0.005   # learned stop width
    target_pct:   float = 0.010   # learned target width
    size_scale:   float = 1.0     # 0.0-1.0 position size multiplier
    confidence:   float = 0.5     # TTM-style confidence 0-1
    last_updated: float = 0.0
    recent_pnl:   List[float] = field(default_factory=list)  # last 20

    @property
    def win_rate(self) -> float:
        return self.wins / max(self.trades, 1)

    @property
    def expectancy(self) -> float:
        """Expected PnL per trade = WR*avg_win - (1-WR)*avg_loss"""
        return self.win_rate * self.avg_win - (1-self.win_rate) * abs(self.avg_loss)

    @property
    def profit_factor(self) -> float:
        gross_win  = self.avg_win  * max(self.wins, 0)
        gross_loss = abs(self.avg_loss) * max(self.losses, 1)
        return gross_win / max(gross_loss, 1e-8)

    def degraded(self) -> bool:
        """True if recent performance is bad enough to reduce size."""
        if len(self.recent_pnl) < 5:
            return False
        recent_wr = sum(1 for p in self.recent_pnl[-10:] if p > 0) / min(len(self.recent_pnl), 10)
        # Stricter than before: flag degraded at 40% recent WR or conf < 0.35
        return recent_wr < 0.40 or self.confidence < 0.35

    def to_dict(self) -> dict:
        d = asdict(self)
        d['recent_pnl'] = list(self.recent_pnl)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'ScenarioStats':
        s = cls(name=d['name'])
        for k, v in d.items():
            if k == 'recent_pnl':
                s.recent_pnl = list(v)
            elif hasattr(s, k):
                setattr(s, k, v)
        return s


# ─────────────────────────────────────────────────────────────────────────────
# LEARNING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class LearningEngine:
    """
    Online learning layer. Call update() after every trade close.
    Call get_params() before every entry to get adapted stop/target/size.
    """

    # Default stop/target bounds (learned values stay within these)
    STOP_MIN,   STOP_MAX   = 0.002, 0.025   # widened max — crashes need room
    TARGET_MIN, TARGET_MAX = 0.004, 0.030
    SIZE_MIN,   SIZE_MAX   = 0.00,  1.50    # allow zero so skip fires cleanly

    # Learning rates
    LR_STOP    = 0.15   # how fast stop adapts
    LR_TARGET  = 0.10
    LR_CONF    = 0.20   # how fast confidence updates
    LR_SIZE    = 0.08

    # Hard floor — below this confidence, skip entry entirely
    TRADE_FLOOR = 0.20

    # Scenarios that learn catastrophically fast (large losses per trade)
    # Use 3× the normal learning rate for confidence and size
    FAST_LEARN_SCENARIOS = {
        "FLASH_CRASH", "FRACTURE", "TOXIC_FLOW", "SPREAD_BLOW"
    }

    def __init__(self, save_path: str = "cortex_learning.json") -> None:
        self._save_path = save_path
        self._scenarios: Dict[str, ScenarioStats] = {}
        self._global_pnl: deque = deque(maxlen=100)
        self._consecutive_losses: int = 0
        self._session_trades: int = 0
        self._da_rpe: float = 0.0       # dopamine RPE from last trade
        self._da_fatigue: float = 0.0   # cortisol proxy
        self._load()

    def _get_or_create(self, scenario: str) -> ScenarioStats:
        if scenario not in self._scenarios:
            self._scenarios[scenario] = ScenarioStats(name=scenario)
        return self._scenarios[scenario]

    def update(
        self,
        scenario:   str,
        pnl:        float,
        tier:       str,
        exit_reason:str,
        entry_price:float,
        exit_price: float,
        hold_ticks: int,
    ) -> dict:
        """
        Called after every trade close. Updates all learned parameters.
        Returns a dict of what changed (for logging/cockpit).
        """
        s = self._get_or_create(scenario)
        self._session_trades += 1
        self._global_pnl.append(pnl)

        # ── Update basic stats ──────────────────────────────────────────────
        s.trades += 1
        s.total_pnl += pnl
        s.recent_pnl = (s.recent_pnl + [pnl])[-20:]
        s.last_updated = time.time()

        if pnl > 0:
            s.wins += 1
            # EMA of avg_win
            s.avg_win = s.avg_win * 0.85 + abs(pnl) * 0.15
            self._consecutive_losses = 0
        else:
            s.losses += 1
            s.avg_loss = s.avg_loss * 0.85 + abs(pnl) * 0.15
            self._consecutive_losses += 1

        # ── Dopamine RPE ──────────────────────────────────────────────────
        expected = s.expectancy
        self._da_rpe = float(np.tanh((pnl - expected) / max(abs(expected), 1.0)))
        # Cortisol rises on consecutive losses
        if pnl < 0:
            self._da_fatigue = min(1.0, self._da_fatigue + 0.15)
        else:
            self._da_fatigue = max(0.0, self._da_fatigue - 0.08)

        # ── Confidence update (TTM-style) ────────────────────────────────
        # Use 3× learning rate for scenarios known to cause large losses
        lr_conf = self.LR_CONF * 3.0 if scenario in self.FAST_LEARN_SCENARIOS else self.LR_CONF
        signal = 1.0 if pnl > 0 else -1.0
        s.confidence = float(np.clip(
            s.confidence + lr_conf * signal * (1 - s.confidence if pnl > 0 else s.confidence),
            0.05, 0.95
        ))

        # ── Adapt stop width ──────────────────────────────────────────────
        # If stop was hit → stop was too tight, widen it
        # If target was hit → stop was fine, keep or tighten slightly
        # If signal exit → neutral
        old_stop = s.stop_pct
        if exit_reason == "STOP":
            # Widen stop — we got stopped out, maybe noise
            s.stop_pct = float(np.clip(
                s.stop_pct * (1 + self.LR_STOP),
                self.STOP_MIN, self.STOP_MAX
            ))
        elif exit_reason == "TARGET":
            # Target hit cleanly — we can tighten stop to protect more
            s.stop_pct = float(np.clip(
                s.stop_pct * (1 - self.LR_STOP * 0.3),
                self.STOP_MIN, self.STOP_MAX
            ))

        # ── Adapt target width ────────────────────────────────────────────
        old_target = s.target_pct
        r_multiple = abs(pnl) / max(abs(entry_price * s.stop_pct), 1e-8)
        if exit_reason == "TARGET":
            # Reached target — was it too conservative? Expand if >5 win streak
            recent_wins = sum(1 for p in s.recent_pnl[-5:] if p > 0)
            if recent_wins >= 4:
                s.target_pct = float(np.clip(
                    s.target_pct * (1 + self.LR_TARGET * 0.5),
                    self.TARGET_MIN, self.TARGET_MAX
                ))
        elif exit_reason == "STOP" and r_multiple < 0.5:
            # Stopped out with large loss — price moved fast, widen target too
            s.target_pct = float(np.clip(
                s.target_pct * (1 + self.LR_TARGET * 0.2),
                self.TARGET_MIN, self.TARGET_MAX
            ))

        # ── Adapt size scale ──────────────────────────────────────────────
        old_size = s.size_scale
        # Gradient: increase size when expectancy positive + confidence high
        # Decrease when expectancy negative or confidence low
        if s.trades >= 5:
            target_size = 1.0
            if s.expectancy > 0 and s.confidence > 0.60:
                target_size = min(self.SIZE_MAX, 1.0 + (s.confidence - 0.6) * 2.0)
            elif s.expectancy < 0 or s.confidence < 0.40:
                target_size = max(self.SIZE_MIN, s.confidence * 0.8)
            # Also penalise consecutive losses
            if self._consecutive_losses >= 3:
                target_size *= max(0.3, 1.0 - self._consecutive_losses * 0.15)
            s.size_scale = float(np.clip(
                s.size_scale + self.LR_SIZE * (target_size - s.size_scale),
                self.SIZE_MIN, self.SIZE_MAX
            ))

        # ── Maintain stop/target ratio (target >= 1.5× stop always) ──────
        if s.target_pct < s.stop_pct * 1.5:
            s.target_pct = float(np.clip(s.stop_pct * 1.8, self.TARGET_MIN, self.TARGET_MAX))

        self._save()

        return {
            "scenario":     scenario,
            "pnl":          round(pnl, 2),
            "win_rate":     round(s.win_rate * 100, 1),
            "confidence":   round(s.confidence, 3),
            "da_rpe":       round(self._da_rpe, 3),
            "da_fatigue":   round(self._da_fatigue, 3),
            "stop_pct":     round(s.stop_pct * 100, 3),
            "target_pct":   round(s.target_pct * 100, 3),
            "size_scale":   round(s.size_scale, 3),
            "stop_delta":   round((s.stop_pct - old_stop) * 100, 4),
            "target_delta": round((s.target_pct - old_target) * 100, 4),
            "size_delta":   round(s.size_scale - old_size, 4),
            "degraded":     s.degraded(),
            "expectancy":   round(s.expectancy, 2),
        }

    def get_params(self, scenario: str, tier: str) -> dict:
        """
        Returns adapted trading parameters for an upcoming entry.
        Call this before opening a position.
        If 'skip' is True in the returned dict, do not open the position.
        """
        s = self._get_or_create(scenario)

        # DA fatigue reduces aggression globally
        fatigue_scale = max(0.3, 1.0 - self._da_fatigue * 0.5)

        # Consecutive loss drawdown guard
        loss_scale = max(0.2, 1.0 - self._consecutive_losses * 0.12)

        size = s.size_scale * fatigue_scale * loss_scale

        # Tier-specific adjustments
        tier_boost = {"SNIPER": 1.0, "HUNTER": 0.85, "SCOUT": 0.65}.get(tier, 0.5)
        size *= tier_boost

        # Hard floor: if confidence has collapsed, skip the trade entirely
        skip = (s.confidence < self.TRADE_FLOOR) and (s.trades >= 5)
        if skip:
            size = 0.0

        return {
            "stop_pct":   float(np.clip(s.stop_pct,   self.STOP_MIN,   self.STOP_MAX)),
            "target_pct": float(np.clip(s.target_pct, self.TARGET_MIN, self.TARGET_MAX)),
            "size_scale": float(np.clip(size,          self.SIZE_MIN,   self.SIZE_MAX)),
            "confidence": s.confidence,
            "degraded":   s.degraded(),
            "skip":       skip,
            "da_rpe":     self._da_rpe,
            "da_fatigue": self._da_fatigue,
            "trades":     s.trades,
            "win_rate":   s.win_rate,
            "expectancy": s.expectancy,
        }

    def report(self) -> str:
        """Human-readable report of all learned scenario parameters."""
        lines = [
            "",
            "  ╔══════════════════════════════════════════════════════════════╗",
            "  ║  CORTEX-16 LEARNING ENGINE — SESSION REPORT                ║",
            "  ╚══════════════════════════════════════════════════════════════╝",
            "",
            f"  {'SCENARIO':<16} {'TRADES':>6} {'WR':>6} {'EXP':>8} {'STOP':>6} {'TGT':>6} {'SIZE':>6} {'CONF':>6} {'PnL':>10}",
            "  " + "─"*72,
        ]
        for name, s in sorted(self._scenarios.items(), key=lambda x: -x[1].total_pnl):
            if s.trades == 0:
                continue
            deg = "⚠" if s.degraded() else " "
            lines.append(
                f"  {deg}{name:<15} {s.trades:>6} {s.win_rate*100:>5.1f}%"
                f" {s.expectancy:>+8.2f} {s.stop_pct*100:>5.2f}% {s.target_pct*100:>5.2f}%"
                f" {s.size_scale:>5.2f}x {s.confidence:>5.3f} {s.total_pnl:>+10.2f}"
            )
        lines += [
            "  " + "─"*72,
            f"  Consecutive losses: {self._consecutive_losses}",
            f"  DA RPE: {self._da_rpe:+.3f}   Fatigue: {self._da_fatigue:.3f}",
            f"  Global trades: {self._session_trades}",
            "",
        ]
        return "\n".join(lines)

    def _save(self) -> None:
        try:
            data = {
                "scenarios":    {k: v.to_dict() for k, v in self._scenarios.items()},
                "da_rpe":       self._da_rpe,
                "da_fatigue":   self._da_fatigue,
                "consecutive_losses": self._consecutive_losses,
                "saved_at":     time.time(),
            }
            with open(self._save_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _load(self) -> None:
        if not os.path.exists(self._save_path):
            return
        try:
            with open(self._save_path) as f:
                data = json.load(f)
            for k, v in data.get("scenarios", {}).items():
                self._scenarios[k] = ScenarioStats.from_dict(v)
            self._da_rpe     = data.get("da_rpe", 0.0)
            self._da_fatigue = data.get("da_fatigue", 0.0)
            self._consecutive_losses = data.get("consecutive_losses", 0)
            age = time.time() - data.get("saved_at", 0)
            hrs = age / 3600
            if self._scenarios:
                total = sum(s.trades for s in self._scenarios.values())
                print(f"  📚 Loaded learning state: {total} prior trades, {hrs:.1f}h ago")
        except Exception as e:
            print(f"  ⚠ Could not load learning state: {e}")
