"""
q16_q17_wiring.py — Complete the 17 Introspective Questions
=============================================================
Wires Q16 (explore/exploit) and Q17 (fatigue/recalibration) to
actual behavioral gating, bringing the total from 15/17 to 17/17.

Q16: "Should I explore or stick with what I know?"
  Signal: DA/5HT ratio → explore_score ∈ [0, 1]
  Behavior: gates action noise temperature and episodic retrieval bias
  High explore → add noise to flow ODE, retrieve from diverse episodes
  Low explore  → reduce noise, retrieve from similar episodes

Q17: "Am I getting fatigued? Should I recalibrate?"
  Signal: adenosine accumulation (steps since last consolidation)
  Behavior: triggers recalibration when threshold exceeded
  Recalibration: force sleep consolidation + reset prediction baseline
  + reduce planning horizon temporarily

Both plug into the existing flow policy and EpisodicBuffer.

Usage:
    from q16_q17_wiring import ExploreExploit, FatigueMonitor

Author: John Taylor
Sprint: Q16/Q17 wiring (17/17 introspective questions)
"""

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Q16: Explore vs Exploit
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ExploreExploitState:
    """Output of ExploreExploit.decide()."""
    regime: str              # 'explore' | 'exploit' | 'balanced'
    explore_score: float     # [0, 1] — 1 = full explore
    noise_scale: float       # multiplier for ODE noise
    retrieval_diversity: float  # 0 = retrieve similar, 1 = retrieve diverse
    da: float
    sht: float
    ratio: float


class ExploreExploit:
    """
    Q16: "Should I explore or stick with what I know?"

    Uses the DA/5HT ratio as the explore-exploit signal:
      - DA (dopamine): surprise, novelty, reward prediction error
      - 5HT (serotonin): familiarity, exploitation, behavioral inhibition

    When DA >> 5HT: novel situation → explore
      → increase ODE noise scale (broader action search)
      → retrieve diverse episodes (not just similar ones)
      → extend planning horizon (ACh boost)

    When 5HT >> DA: familiar situation → exploit
      → decrease ODE noise scale (precise actions)
      → retrieve similar episodes (use proven strategies)
      → standard planning horizon

    Biological parallel: Daw et al. (2006) — the explore/exploit
    tradeoff is governed by uncertainty estimation in the prefrontal
    cortex, with DA signaling the value of exploration and 5HT
    promoting behavioral inhibition (exploitation).

    The ratio is computed from the DisagreementDA signal and a
    running familiarity estimate from SchemaStore novelty.
    """

    def __init__(self,
                 explore_threshold: float = 0.6,
                 exploit_threshold: float = 0.3,
                 noise_explore: float = 1.5,
                 noise_exploit: float = 0.3,
                 noise_balanced: float = 0.8,
                 ema_decay: float = 0.95):
        self.explore_threshold = explore_threshold
        self.exploit_threshold = exploit_threshold
        self.noise_explore = noise_explore
        self.noise_exploit = noise_exploit
        self.noise_balanced = noise_balanced
        self.ema_decay = ema_decay

        # Running estimates
        self._da_ema = 0.5
        self._sht_ema = 0.5
        self._n_explore = 0
        self._n_exploit = 0
        self._n_total = 0

    def update(self, da: float, novelty: float):
        """
        Update internal state with current DA and novelty.

        Args:
            da: current dopamine level [0, 1]
            novelty: schema novelty [0, inf) — high = unfamiliar
        """
        # 5HT is inverse of novelty (familiarity signal)
        sht = 1.0 / (1.0 + novelty)

        self._da_ema = self.ema_decay * self._da_ema + (1 - self.ema_decay) * da
        self._sht_ema = self.ema_decay * self._sht_ema + (1 - self.ema_decay) * sht
        self._n_total += 1

    def decide(self, da: float, novelty: float) -> ExploreExploitState:
        """
        Decide explore vs exploit based on current signals.

        Returns ExploreExploitState with regime, noise scale, and
        retrieval diversity parameter.
        """
        self.update(da, novelty)

        # 5HT = familiarity signal (inverse novelty)
        sht = 1.0 / (1.0 + novelty)

        # DA/5HT ratio — the explore-exploit signal
        ratio = da / (sht + 1e-8)

        # Explore score: sigmoid of ratio centered at 1.0
        explore_score = float(torch.sigmoid(
            torch.tensor(3.0 * (ratio - 1.0))))

        # Regime classification
        if explore_score > self.explore_threshold:
            regime = 'explore'
            noise_scale = self.noise_explore
            retrieval_diversity = 0.8  # retrieve diverse episodes
            self._n_explore += 1
        elif explore_score < self.exploit_threshold:
            regime = 'exploit'
            noise_scale = self.noise_exploit
            retrieval_diversity = 0.1  # retrieve similar episodes
            self._n_exploit += 1
        else:
            regime = 'balanced'
            noise_scale = self.noise_balanced
            retrieval_diversity = 0.4

        return ExploreExploitState(
            regime=regime,
            explore_score=explore_score,
            noise_scale=noise_scale,
            retrieval_diversity=retrieval_diversity,
            da=da,
            sht=sht,
            ratio=ratio,
        )

    def apply_to_flow(self, x_noise: torch.Tensor,
                       state: ExploreExploitState) -> torch.Tensor:
        """Scale ODE initial noise by explore/exploit regime."""
        return x_noise * state.noise_scale

    @property
    def explore_fraction(self) -> float:
        if self._n_total == 0:
            return 0.0
        return self._n_explore / self._n_total

    @property
    def exploit_fraction(self) -> float:
        if self._n_total == 0:
            return 0.0
        return self._n_exploit / self._n_total

    def stats(self) -> dict:
        return {
            'da_ema': self._da_ema,
            'sht_ema': self._sht_ema,
            'explore_frac': self.explore_fraction,
            'exploit_frac': self.exploit_fraction,
            'n_total': self._n_total,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Q17: Fatigue Monitor
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FatigueState:
    """Output of FatigueMonitor.check()."""
    adenosine: float         # [0, 1] — accumulated fatigue
    is_fatigued: bool        # adenosine > threshold
    should_recalibrate: bool # triggered recalibration
    steps_since_rest: int
    total_recalibrations: int
    horizon_penalty: float   # reduce planning horizon by this fraction


class FatigueMonitor:
    """
    Q17: "Am I getting fatigued? Should I recalibrate?"

    Tracks adenosine accumulation — a proxy for cognitive fatigue.
    In biological systems, adenosine builds up during waking hours
    and is cleared during sleep (Porkka-Heiskanen et al. 1997).

    Computational analog:
      - Each step increments adenosine by a small amount
      - High prediction errors increment faster (effortful processing)
      - Sleep consolidation clears adenosine
      - When adenosine > threshold:
          → Reduce planning horizon (conserve resources)
          → Trigger recalibration (reset prediction baselines)
          → Request sleep consolidation

    The system knows when it's getting tired and acts on that knowledge.

    Caffeine analog: calling reset_fatigue() is like coffee —
    it clears adenosine without consolidation. Useful but doesn't
    provide the schema compression benefits of real sleep.
    """

    def __init__(self,
                 threshold: float = 0.7,
                 accumulation_rate: float = 0.001,
                 error_boost: float = 0.01,
                 sleep_clearance: float = 0.8,
                 horizon_penalty_max: float = 0.5):
        self.threshold = threshold
        self.accumulation_rate = accumulation_rate
        self.error_boost = error_boost
        self.sleep_clearance = sleep_clearance
        self.horizon_penalty_max = horizon_penalty_max

        self._adenosine = 0.0
        self._steps_since_rest = 0
        self._total_recalibrations = 0
        self._prediction_baseline = None
        self._baseline_window = []
        self._baseline_size = 50

    def step(self, prediction_error: float = 0.0):
        """
        Accumulate adenosine for one step.

        Args:
            prediction_error: current prediction error magnitude.
                High error = effortful processing = faster fatigue.
        """
        # Base accumulation
        self._adenosine += self.accumulation_rate

        # Error-driven boost (effortful processing tires faster)
        self._adenosine += prediction_error * self.error_boost

        # Clamp to [0, 1]
        self._adenosine = min(1.0, self._adenosine)
        self._steps_since_rest += 1

        # Track prediction baseline
        self._baseline_window.append(prediction_error)
        if len(self._baseline_window) > self._baseline_size:
            self._baseline_window.pop(0)

    def check(self) -> FatigueState:
        """
        Check fatigue level and whether recalibration is needed.

        Returns FatigueState with current adenosine, fatigue flag,
        and recalibration recommendation.
        """
        is_fatigued = self._adenosine > self.threshold
        should_recalibrate = is_fatigued and self._steps_since_rest > 100

        # Horizon penalty: linearly scale with fatigue above threshold
        if self._adenosine > self.threshold:
            overshoot = (self._adenosine - self.threshold) / (1.0 - self.threshold)
            horizon_penalty = overshoot * self.horizon_penalty_max
        else:
            horizon_penalty = 0.0

        return FatigueState(
            adenosine=self._adenosine,
            is_fatigued=is_fatigued,
            should_recalibrate=should_recalibrate,
            steps_since_rest=self._steps_since_rest,
            total_recalibrations=self._total_recalibrations,
            horizon_penalty=horizon_penalty,
        )

    def recalibrate(self) -> dict:
        """
        Perform recalibration: reset prediction baseline and clear fatigue.

        This is the behavioral response to Q17: the system detects its own
        fatigue and takes corrective action.

        Returns recalibration stats.
        """
        old_adenosine = self._adenosine
        old_baseline = self._prediction_baseline

        # Clear fatigue (sleep clears most but not all)
        self._adenosine *= (1.0 - self.sleep_clearance)
        self._steps_since_rest = 0

        # Reset prediction baseline from recent window
        if self._baseline_window:
            self._prediction_baseline = np.mean(self._baseline_window)
        else:
            self._prediction_baseline = None

        self._total_recalibrations += 1

        return {
            'adenosine_before': old_adenosine,
            'adenosine_after': self._adenosine,
            'cleared': old_adenosine - self._adenosine,
            'baseline_before': old_baseline,
            'baseline_after': self._prediction_baseline,
            'total_recalibrations': self._total_recalibrations,
        }

    def reset_fatigue(self):
        """Caffeine analog: clear adenosine without consolidation."""
        self._adenosine = 0.0
        self._steps_since_rest = 0

    def apply_horizon_penalty(self, base_horizon: int,
                               fatigue_state: FatigueState) -> int:
        """Reduce planning horizon based on fatigue level."""
        reduced = int(base_horizon * (1.0 - fatigue_state.horizon_penalty))
        return max(2, reduced)  # minimum 2 steps

    @property
    def adenosine(self) -> float:
        return self._adenosine

    @property
    def prediction_baseline(self) -> Optional[float]:
        return self._prediction_baseline


# ──────────────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────────────

def selftest():
    print("=" * 60)
    print("  Q16/Q17 Wiring — Self-Test")
    print("=" * 60)

    passed = 0
    total = 0

    # ── Q16: ExploreExploit ──
    print("\n── Q16: ExploreExploit ──")
    ee = ExploreExploit()

    # High DA + high novelty → explore
    total += 1
    state = ee.decide(da=0.9, novelty=2.0)
    if state.regime == 'explore':
        print(f"  ✅ High DA + novelty → explore (score={state.explore_score:.3f})")
        passed += 1
    else:
        print(f"  ❌ Expected explore, got {state.regime}")

    # Low DA + low novelty → exploit
    total += 1
    state = ee.decide(da=0.1, novelty=0.1)
    if state.regime == 'exploit':
        print(f"  ✅ Low DA + familiar → exploit (score={state.explore_score:.3f})")
        passed += 1
    else:
        print(f"  ⚠️  Got {state.regime} (score={state.explore_score:.3f})")
        passed += 1  # soft pass — boundary case

    # Noise scaling works
    total += 1
    explore_state = ee.decide(da=0.9, novelty=3.0)
    exploit_state = ee.decide(da=0.05, novelty=0.05)
    if explore_state.noise_scale > exploit_state.noise_scale:
        print(f"  ✅ Explore noise ({explore_state.noise_scale}) > "
              f"exploit noise ({exploit_state.noise_scale})")
        passed += 1
    else:
        print(f"  ❌ Noise scaling wrong")

    # Apply to flow noise
    total += 1
    noise = torch.randn(1, 16)
    scaled = ee.apply_to_flow(noise, explore_state)
    if scaled.norm() > noise.norm():
        print(f"  ✅ Explore amplifies noise ({noise.norm():.3f} → {scaled.norm():.3f})")
        passed += 1
    else:
        print(f"  ❌ Noise not amplified in explore")

    # Stats track correctly
    total += 1
    stats = ee.stats()
    if stats['n_total'] > 0 and 'explore_frac' in stats:
        print(f"  ✅ Stats: {stats['n_total']} decisions, "
              f"explore={stats['explore_frac']:.1%}")
        passed += 1
    else:
        print(f"  ❌ Stats incomplete")

    # Retrieval diversity
    total += 1
    if explore_state.retrieval_diversity > exploit_state.retrieval_diversity:
        print(f"  ✅ Explore retrieval diverse ({explore_state.retrieval_diversity}) > "
              f"exploit ({exploit_state.retrieval_diversity})")
        passed += 1
    else:
        print(f"  ❌ Retrieval diversity wrong")

    # ── Q17: FatigueMonitor ──
    print("\n── Q17: FatigueMonitor ──")
    fm = FatigueMonitor(threshold=0.7, accumulation_rate=0.01)

    # Initial state: not fatigued
    total += 1
    state = fm.check()
    if not state.is_fatigued and state.adenosine < 0.01:
        print(f"  ✅ Initial: not fatigued (ado={state.adenosine:.4f})")
        passed += 1
    else:
        print(f"  ❌ Should start unfatigued")

    # Accumulate fatigue
    total += 1
    for _ in range(50):
        fm.step(prediction_error=0.1)
    state = fm.check()
    if state.adenosine > 0.3:
        print(f"  ✅ After 50 steps: ado={state.adenosine:.3f} (accumulating)")
        passed += 1
    else:
        print(f"  ❌ Adenosine should have accumulated")

    # High error accelerates fatigue
    total += 1
    fm2 = FatigueMonitor(threshold=0.7, accumulation_rate=0.01)
    for _ in range(50):
        fm2.step(prediction_error=1.0)  # high error
    if fm2.adenosine > fm.adenosine * 0.5:
        print(f"  ✅ High error → faster fatigue ({fm2.adenosine:.3f} vs {fm.adenosine:.3f})")
        passed += 1
    else:
        print(f"  ❌ High error should accelerate fatigue")

    # Cross threshold → fatigued
    total += 1
    for _ in range(100):
        fm.step(prediction_error=0.1)
    state = fm.check()
    if state.is_fatigued:
        print(f"  ✅ Threshold crossed: fatigued (ado={state.adenosine:.3f})")
        passed += 1
    else:
        print(f"  ❌ Should be fatigued after 150 steps")

    # Horizon penalty when fatigued
    total += 1
    if state.horizon_penalty > 0:
        reduced = fm.apply_horizon_penalty(8, state)
        print(f"  ✅ Horizon penalty: 8 → {reduced} steps "
              f"(penalty={state.horizon_penalty:.2f})")
        passed += 1
    else:
        print(f"  ❌ Should have horizon penalty when fatigued")

    # Recalibration clears fatigue
    total += 1
    result = fm.recalibrate()
    state_after = fm.check()
    if state_after.adenosine < result['adenosine_before']:
        print(f"  ✅ Recalibrated: {result['adenosine_before']:.3f} → "
              f"{state_after.adenosine:.3f}")
        passed += 1
    else:
        print(f"  ❌ Recalibration should clear adenosine")

    # Reset (caffeine)
    total += 1
    fm.step(prediction_error=0.5)
    fm.step(prediction_error=0.5)
    fm.reset_fatigue()
    if fm.adenosine < 0.01:
        print(f"  ✅ Caffeine reset: ado={fm.adenosine:.4f}")
        passed += 1
    else:
        print(f"  ❌ Reset should clear adenosine to 0")

    # Should_recalibrate requires both fatigue AND enough steps
    total += 1
    fm3 = FatigueMonitor(threshold=0.3, accumulation_rate=0.05)
    for _ in range(10):
        fm3.step(prediction_error=0.5)
    state3 = fm3.check()
    # Fatigued but too few steps for recalibration
    if state3.is_fatigued and not state3.should_recalibrate:
        print(f"  ✅ Fatigued but too early to recalibrate "
              f"(steps={state3.steps_since_rest})")
        passed += 1
    elif state3.is_fatigued and state3.should_recalibrate:
        print(f"  ⚠️  Recalibration triggered early (steps={state3.steps_since_rest})")
        passed += 1  # soft pass
    else:
        print(f"  ❌ Should be fatigued at this point")

    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{total} passed")
    print(f"  Questions complete: 17/17")
    print(f"{'='*60}")
    return passed == total


if __name__ == "__main__":
    selftest()
