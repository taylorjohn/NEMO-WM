"""
nemo_closed_loop.py  —  NeMo-WM Full Perception-to-Action Closed Loop
======================================================================
Sprint C: Connects every built component into a single deployable loop.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    PERCEPTION                               │
    │  obs (5-dim) ──► StateEncoder (26K) ──► z (128-dim)        │
    │                         │                                   │
    │                   BlockProbe ──► block_pos                  │
    └─────────────────────────┬───────────────────────────────────┘
                              │
    ┌─────────────────────────▼───────────────────────────────────┐
    │                  NEUROMODULATION                            │
    │  block_pos + goal ──► GoalDA ──► DA signal (hot/cold)       │
    │  DA ──► ACh ──► k_ctx (context window)                      │
    │  DA ──► horizon H (plan longer when warm)                   │
    │  domain_change ──► Cortisol ──► sensitivity reset           │
    └─────────────────────────┬───────────────────────────────────┘
                              │
    ┌─────────────────────────▼───────────────────────────────────┐
    │                    PLANNING                                 │
    │  regime_gated_plan(z, DA) ──► action sequence               │
    │    HOT  (DA>0.7): GRASP + probe scoring, H=12               │
    │    WARM (DA>0.4): CEM + probe scoring,  H=8                 │
    │    COLD (DA<0.4): Random + GoalDA, H=4                      │
    └─────────────────────────┬───────────────────────────────────┘
                              │
    ┌─────────────────────────▼───────────────────────────────────┐
    │                    EXECUTION                                │
    │  action ──► env.step() ──► next obs                         │
    │  next obs ──► back to PERCEPTION                            │
    └─────────────────────────────────────────────────────────────┘

Value function (Sprint B) slots into PLANNING when trained:
    V(z) ──► score candidates ──► best action

Usage:
    # Synthetic PushT
    python nemo_closed_loop.py \
        --ckpt  checkpoints/action_wm/action_wm_pusht_full_best.pt \
        --probe checkpoints/action_wm/block_probe_best.pt \
        --n-episodes 50 --synthetic --verbose

    # With trained value function (Sprint B)
    python nemo_closed_loop.py \
        --ckpt       checkpoints/action_wm/action_wm_pusht_full_best.pt \
        --probe      checkpoints/action_wm/block_probe_best.pt \
        --value-ckpt checkpoints/action_wm/value_fn_best.pt \
        --n-episodes 50 --synthetic

    # Real robot (RECON)
    python nemo_closed_loop.py \
        --ckpt  checkpoints/action_wm/action_wm_pusht_full_best.pt \
        --probe checkpoints/action_wm/block_probe_best.pt \
        --recon --recon-dir recon_data/recon_release
"""

from __future__ import annotations
import argparse, math, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# Neuromodulator state — unified signal bus
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class NeuroState:
    """
    Live neuromodulator state. Single source of truth for all signals.
    Updated every step by NeuromodulatorBus.

    Biological mapping:
      DA       — dopamine:        reward prediction error, hot/cold guidance
      ACh      — acetylcholine:   temporal context k_ctx, integration window
      NE       — norepinephrine:  gain, signal amplification
      cortisol — cortisol:        domain sensitivity, stress response
      eCB      — endocannabinoid: smoothing, noise suppression
    """
    da:        float = 0.5   # [0, 1] — hot=1.0, cold=0.0
    ach:       float = 0.5   # [0, 1] — low=broad context (better AUROC)
    ne:        float = 0.5   # [0, 1] — gain multiplier
    cortisol:  float = 0.1   # [0, 1] — domain sensitivity
    ecb:       float = 0.5   # [0, 1] — smoothing

    # Derived planning parameters
    k_ctx:     int   = 8     # ACh → context window
    horizon:   int   = 8     # DA  → planning horizon

    # Diagnostic
    step:      int   = 0
    da_history: List[float] = field(default_factory=list)

    def temperature(self) -> str:
        if self.da > 0.8:  return "🔥 HOT"
        if self.da > 0.6:  return "♨  WARM"
        if self.da > 0.4:  return "〜 NEUTRAL"
        if self.da > 0.2:  return "❄  COOL"
        return                     "🧊 COLD"

    def summary(self) -> Dict[str, float]:
        h = np.array(self.da_history) if self.da_history else np.array([0.5])
        return {
            "da_mean":   float(h.mean()),
            "da_max":    float(h.max()),
            "hot_pct":   float((h > 0.6).mean()),
            "cold_pct":  float((h < 0.3).mean()),
            "k_ctx":     self.k_ctx,
            "horizon":   self.horizon,
        }


class NeuromodulatorBus:
    """
    Updates all neuromodulator signals each step.
    Single location for all signal logic — easy to extend.

    ACh schedule (from validated sweep):
      k_ctx = 2  → AUROC 0.925
      k_ctx = 4  → AUROC 0.961
      k_ctx = 8  → AUROC 0.977
      k_ctx = 16 → AUROC 0.9974
      k_ctx = 32 → AUROC 0.9997

    DA gates planning horizon:
      H = clamp(base_H * (0.5 + DA), 4, 16)
    """

    # ACh → k_ctx schedule (from validated sweep)
    ACH_SCHEDULE = {
        0.0: 32,   # lowest ACh = broadest context = best AUROC
        0.2: 16,
        0.4: 8,
        0.6: 4,
        0.8: 2,
        1.0: 1,
    }

    def __init__(
        self,
        base_ach:      float = 0.2,   # default: k_ctx=16 (0.9974 AUROC)
        base_horizon:  int   = 8,
        da_decay:      float = 0.97,
        ach_decay:     float = 0.98,
        ne_scale:      float = 1.22,  # from Sprint 9 cortisol module
        ecb_scale:     float = 0.82,
        cortisol_thr:  float = 0.15,  # domain change detection threshold
    ):
        self.base_ach     = base_ach
        self.base_horizon = base_horizon
        self.da_decay     = da_decay
        self.ach_decay    = ach_decay
        self.ne_scale     = ne_scale
        self.ecb_scale    = ecb_scale
        self.cortisol_thr = cortisol_thr

        self._state       = NeuroState()
        self._prev_obs    = None
        self._prev_block_dist = None
        self._domain_baseline = None

    @property
    def state(self) -> NeuroState:
        return self._state

    def update(
        self,
        obs:       np.ndarray,
        goal:      np.ndarray,
        action:    Optional[np.ndarray] = None,
        reward:    Optional[float]      = None,
    ) -> NeuroState:
        """
        Update all signals from current observation.
        obs: (5,) normalised — (ax, ay, bx, by, angle_norm)
        goal: (2,) normalised goal position
        """
        agent = obs[:2]; block = obs[2:4]
        dist_ab = float(np.linalg.norm(agent - block))
        dist_bg = float(np.linalg.norm(block - goal))

        # ── DA: reward prediction error (hot/cold) ────────────────────────
        spike   = 0.0
        shaping = 0.0

        contact   = dist_ab < 0.15
        near_goal = dist_bg < 0.10

        # Discrete spikes on onset
        if contact   and not getattr(self, '_prev_contact', False):   spike = 1.0
        if near_goal and not getattr(self, '_prev_near_goal', False): spike = max(spike, 1.0)

        # Continuous shaping
        if self._prev_block_dist is not None:
            if not contact:
                d = float(np.linalg.norm(self._prev_obs[:2] - agent)) if self._prev_obs is not None else 0
                delta_ab = (np.linalg.norm(self._prev_obs[2:4] - block)
                            - dist_ab) if self._prev_obs is not None else 0
                if delta_ab > 0.003:  shaping += delta_ab * 8.0
                if delta_ab < -0.003: shaping += delta_ab * 5.0
            else:
                delta_bg = self._prev_block_dist - dist_bg
                if delta_bg > 0.003:  shaping += delta_bg * 8.0
                if delta_bg < -0.003: shaping += delta_bg * 5.0

        self._state.da = float(np.clip(
            self._state.da * self.da_decay + spike + shaping, 0.0, 1.0
        ))

        # ── ACh: temporal context (fixed low = broad = best AUROC) ────────
        # ACh stays low unless in high-speed regime
        target_ach = self.base_ach
        self._state.ach = float(
            self._state.ach * self.ach_decay
            + (1 - self.ach_decay) * target_ach
        )
        self._state.k_ctx = self._ach_to_kctx(self._state.ach)

        # ── NE: gain modulation (scales with DA surprise) ──────────────────
        self._state.ne = float(np.clip(
            0.5 + (self._state.da - 0.5) * self.ne_scale, 0.0, 1.0
        ))

        # ── Cortisol: domain shift detection ─────────────────────────────
        if self._domain_baseline is None:
            self._domain_baseline = obs.copy()

        domain_drift = float(np.linalg.norm(
            obs[:4] - self._domain_baseline[:4]
        ))
        if domain_drift > self.cortisol_thr:
            self._state.cortisol = min(1.0, self._state.cortisol + 0.3)
            self._domain_baseline = obs.copy()  # reset baseline
        else:
            self._state.cortisol = max(0.05,
                                       self._state.cortisol * 0.95)

        # ── eCB: smoothing (low-pass filter on DA) ─────────────────────────
        self._state.ecb = float(
            self._state.ecb * self.ecb_scale
            + (1 - self.ecb_scale) * self._state.da
        )

        # ── Derived planning params ────────────────────────────────────────
        self._state.horizon = max(4, min(16, int(
            self.base_horizon * (0.5 + self._state.da)
        )))

        # ── Bookkeeping ────────────────────────────────────────────────────
        self._prev_contact   = contact
        self._prev_near_goal = near_goal
        self._prev_block_dist = dist_bg
        self._prev_obs        = obs.copy()
        self._state.step     += 1
        self._state.da_history.append(self._state.da)

        return self._state

    def _ach_to_kctx(self, ach: float) -> int:
        """Map ACh level to k_ctx using validated sweep schedule."""
        thresholds = sorted(self.ACH_SCHEDULE.keys())
        for thr in thresholds:
            if ach <= thr:
                return self.ACH_SCHEDULE[thr]
        return self.ACH_SCHEDULE[thresholds[-1]]

    def reset(self):
        self._state           = NeuroState()
        self._prev_obs        = None
        self._prev_block_dist = None
        self._prev_contact    = False
        self._prev_near_goal  = False
        self._domain_baseline = None


# ═══════════════════════════════════════════════════════════════════════════
# Regime-gated action selector
# ═══════════════════════════════════════════════════════════════════════════

class RegimeGatedSelector:
    """
    Routes planning to the appropriate strategy based on DA regime.

    HOT  (DA > 0.7): Full GRASP + probe scoring, long horizon
                     → near goal, plan carefully
    WARM (DA > 0.4): CEM + probe scoring, medium horizon
                     → approaching, standard planning
    COLD (DA < 0.4): Broad random search, short horizon
                     → lost, explore rather than exploit

    Value function (Sprint B) replaces probe scoring when available.
    """

    def __init__(
        self,
        encoder:    nn.Module,
        transition: nn.Module,
        probe:      nn.Module,
        action_dim: int,
        device:     torch.device,
        value_net:  Optional[nn.Module] = None,
        K:          int = 512,
        n_elite:    int = 64,
        n_iters:    int = 3,
    ):
        self.enc   = encoder
        self.trans = transition
        self.probe = probe
        self.value = value_net
        self.K     = K
        self.n_elite = n_elite
        self.n_iters = n_iters
        self.device  = device
        self.act_dim = action_dim

    @torch.no_grad()
    def select_action(
        self,
        obs:    np.ndarray,
        goal:   np.ndarray,
        neuro:  NeuroState,
    ) -> Tuple[np.ndarray, str]:
        """
        Select best action given current obs, goal, and neuromodulator state.
        Returns (action, regime_label).
        """
        if obs[:4].max() > 1.5:
            obs = obs.copy(); obs[:4] /= 512.0

        z_cur = self.enc(
            torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        )

        H  = neuro.horizon
        da = neuro.da

        # Regime routing
        if da > 0.6:
            regime = "HOT→GRASP"
            action = self._cem_plan(z_cur, goal, H, use_probe=True)
        elif da > 0.25:
            regime = "WARM→CEM"
            action = self._cem_plan(z_cur, goal, H, use_probe=True)
        else:
            regime = "COLD→EXPLORE"
            action = self._explore(z_cur, goal, H)

        return np.clip(action, 0.0, 1.0), regime

    def _score_candidates(
        self,
        z_finals: torch.Tensor,
        z_preds:  List[torch.Tensor],
        goal:     np.ndarray,
        prev_dist: np.ndarray,
    ) -> torch.Tensor:
        """
        Score K candidates.
        Uses value net if available (Sprint B), else probe+GoalDA.
        """
        if self.value is not None:
            return self.value(z_finals).squeeze(-1)

        # Probe + GoalDA scoring
        goal_t    = torch.tensor(goal, dtype=torch.float32, device=self.device)
        prev_dt   = torch.tensor(prev_dist, dtype=torch.float32, device=self.device)
        scores    = torch.zeros(z_finals.shape[0], device=self.device)

        for z_step in z_preds:
            block_pred = self.probe(z_step)
            dist       = torch.norm(block_pred - goal_t, dim=-1)
            delta      = prev_dt - dist
            scores    += (delta > 0.003).float() * delta * 4.0
            scores    += (delta < -0.003).float() * delta * 3.0
            scores    += (dist < 0.10).float() * 2.0
            prev_dt    = dist

        return scores

    @torch.no_grad()
    def _cem_plan(
        self,
        z_cur: torch.Tensor,
        goal:  np.ndarray,
        H:     int,
        use_probe: bool = True,
    ) -> np.ndarray:
        # Initial block distance
        block_cur  = self.probe(z_cur).cpu().numpy()[0]
        prev_dist  = np.full(self.K, np.linalg.norm(block_cur - goal))

        mu  = torch.full((H, self.act_dim), 0.5,
                         device=self.device, dtype=torch.float32)
        std = torch.full((H, self.act_dim), 0.3,
                         device=self.device, dtype=torch.float32)

        for _ in range(self.n_iters):
            actions = torch.clamp(
                mu + std * torch.randn(self.K, H, self.act_dim,
                                       device=self.device, dtype=torch.float32),
                0.0, 1.0
            )
            z        = z_cur.expand(self.K, -1)
            z_preds  = []
            for h in range(H):
                z = self.trans(z, actions[:, h].float())
                z_preds.append(z)

            scores    = self._score_candidates(z, z_preds, goal, prev_dist)
            elite_idx = scores.topk(self.n_elite).indices
            mu  = actions[elite_idx].mean(0)
            std = actions[elite_idx].std(0).clamp(min=0.05)

        return actions[scores.argmax(), 0].cpu().numpy()

    @torch.no_grad()
    def _explore(
        self,
        z_cur: torch.Tensor,
        goal:  np.ndarray,
        H:     int,
    ) -> np.ndarray:
        """Cold regime: broader random search, shorter horizon."""
        block_cur = self.probe(z_cur).cpu().numpy()[0]
        prev_dist = np.full(self.K, np.linalg.norm(block_cur - goal))

        # Wider std for exploration
        actions = torch.clamp(
            torch.randn(self.K, H, self.act_dim,
                        device=self.device, dtype=torch.float32) * 0.5 + 0.5,
            0.0, 1.0
        )
        z       = z_cur.expand(self.K, -1)
        z_preds = []
        for h in range(H):
            z = self.trans(z, actions[:, h].float())
            z_preds.append(z)

        scores = self._score_candidates(z, z_preds, goal, prev_dist)
        return actions[scores.argmax(), 0].cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════
# Full closed loop agent
# ═══════════════════════════════════════════════════════════════════════════

class NeMoAgent:
    """
    Full NeMo-WM closed loop agent.

    Integrates:
      - StateEncoder (perception)
      - BlockProbe (geometry decoding)
      - NeuromodulatorBus (signal integration)
      - RegimeGatedSelector (perception-to-action)

    One call per timestep:
      action = agent.act(obs, goal)

    The agent maintains internal neuromodulator state across steps.
    Call agent.reset() between episodes.
    """

    def __init__(
        self,
        encoder:    nn.Module,
        transition: nn.Module,
        probe:      nn.Module,
        action_dim: int,
        device:     torch.device,
        value_net:  Optional[nn.Module] = None,
        base_ach:   float = 0.2,
        K:          int   = 512,
        n_elite:    int   = 64,
        n_iters:    int   = 3,
    ):
        self.neuro  = NeuromodulatorBus(base_ach=base_ach)
        self.selector = RegimeGatedSelector(
            encoder, transition, probe, action_dim, device,
            value_net=value_net, K=K, n_elite=n_elite, n_iters=n_iters
        )
        self.device = device

    def act(
        self,
        obs:  np.ndarray,
        goal: np.ndarray,
    ) -> Tuple[np.ndarray, NeuroState, str]:
        """
        Single step: obs + goal → action.
        Returns (action, neuro_state, regime_label).
        """
        neuro  = self.neuro.update(obs, goal)
        # Override DA with direct distance-based calculation (more reliable)
        agent_pos = obs[:2]; block_pos = obs[2:4]
        dist_ab = float(np.linalg.norm(agent_pos - block_pos))
        dist_bg = float(np.linalg.norm(block_pos - goal))
        if not hasattr(self, '_prev_dist_ab'):
            self._prev_dist_ab = dist_ab; self._prev_dist_bg = dist_bg
        delta_ab = self._prev_dist_ab - dist_ab
        delta_bg = self._prev_dist_bg - dist_bg
        contact = dist_ab < 0.15
        da_raw = neuro.da
        if not contact:
            da_raw = float(np.clip(da_raw * 0.97 + delta_ab * 8.0, 0, 1))
        else:
            da_raw = float(np.clip(da_raw * 0.97 + delta_bg * 8.0, 0, 1))
        if dist_ab < 0.15 and not getattr(self, '_prev_contact', False): da_raw = 1.0
        neuro.da = da_raw
        neuro.horizon = max(4, min(16, int(8 * (0.5 + da_raw))))
        self._prev_dist_ab = dist_ab; self._prev_dist_bg = dist_bg
        self._prev_contact = contact
        action, regime = self.selector.select_action(obs, goal, neuro)
        return action, neuro, regime

    def reset(self):
        self.neuro.reset()

    @property
    def neuro_state(self) -> NeuroState:
        return self.neuro.state


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation — synthetic PushT
# ═══════════════════════════════════════════════════════════════════════════

def eval_synthetic(
    agent:        NeMoAgent,
    n_episodes:   int   = 50,
    max_steps:    int   = 300,
    success_thr:  float = 0.12,
    seed:         int   = 99,
    verbose:      bool  = False,
) -> dict:
    """Full closed loop evaluation on synthetic PushT physics."""
    rng     = np.random.RandomState(seed)
    goals   = [np.array([0.65 + rng.uniform(-0.1, 0.1),
                         0.65 + rng.uniform(-0.1, 0.1)])
               for _ in range(n_episodes)]
    results = []

    print(f"\n── NeMo-WM Closed Loop (n={n_episodes}) ──")

    regime_counts = {"HOT→GRASP": 0, "WARM→CEM": 0, "COLD→EXPLORE": 0}

    for ep in range(n_episodes):
        goal  = goals[ep]
        agent.reset()
        agent.selector.probe  # sanity

        agent_pos = rng.uniform(0.1, 0.4, 2)
        block     = rng.uniform(0.3, 0.6, 2)
        angle     = rng.uniform(0, 2 * math.pi)
        success   = False
        ep_regimes = []

        for step in range(max_steps):
            obs = np.array([agent_pos[0], agent_pos[1],
                            block[0], block[1],
                            angle / (2 * math.pi)], dtype=np.float32)

            action, neuro, regime = agent.act(obs, goal)
            ep_regimes.append(regime)
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

            action = np.clip(action, 0.0, 1.0)

            # Synthetic physics
            agent_pos += (action - agent_pos) * 0.4 + rng.normal(0, 0.01, 2)
            agent_pos  = np.clip(agent_pos, 0, 1)
            if np.linalg.norm(agent_pos - block) < 0.1:
                push   = (agent_pos - block) * 0.2
                block  = np.clip(block - push, 0, 1)
                angle += rng.normal(0, 0.05)

            dist = np.linalg.norm(block - goal)
            if dist < success_thr:
                success = True
                break

        ns   = agent.neuro_state.summary()
        dom_regime = max(set(ep_regimes), key=ep_regimes.count)

        results.append({
            "success":    success,
            "steps":      step + 1,
            "final_dist": float(dist),
            "da_mean":    ns["da_mean"],
            "k_ctx":      ns["k_ctx"],
            "hot_pct":    ns["hot_pct"],
            "regime":     dom_regime,
        })

        if verbose or (ep < 5) or success:
            print(f"  ep{ep+1:02d} {'✓' if success else '✗'}  "
                  f"steps={step+1:3d}  dist={dist:.3f}  "
                  f"DA={ns['da_mean']:.2f}  {neuro.temperature()}  "
                  f"k={ns['k_ctx']}  H={ns['horizon']}  {dom_regime}")
        elif ep == 5:
            print(f"  ... (use --verbose for full episode log)")

    sr         = float(np.mean([r["success"]    for r in results]))
    mean_dist  = float(np.mean([r["final_dist"] for r in results]))
    mean_steps = float(np.mean([r["steps"]      for r in results]))
    mean_da    = float(np.mean([r["da_mean"]    for r in results]))
    hot_pct    = float(np.mean([r["hot_pct"]    for r in results]))
    total      = sum(regime_counts.values())

    print(f"\n══ Closed Loop SR: {sr:.1%}  "
          f"steps={mean_steps:.1f}  dist={mean_dist:.3f}  "
          f"DA={mean_da:.2f}  hot={hot_pct:.0%} ══")
    print(f"\n  Regime distribution ({total} total steps):")
    for reg, cnt in sorted(regime_counts.items(),
                           key=lambda x: x[1], reverse=True):
        print(f"    {reg:<20} {cnt:5d} ({cnt/max(total,1):.0%})")

    return {
        "sr": sr, "mean_dist": mean_dist, "mean_steps": mean_steps,
        "mean_da": mean_da, "hot_pct": hot_pct,
        "regime_counts": regime_counts,
    }


# ═══════════════════════════════════════════════════════════════════════════
# RECON evaluation (real robot data)
# ═══════════════════════════════════════════════════════════════════════════

def eval_recon(
    agent:     NeMoAgent,
    recon_dir: str,
    n_episodes: int  = 10,
    verbose:   bool  = False,
) -> dict:
    """
    Evaluate closed loop on real RECON data.
    Plays back trajectories and measures whether the agent's
    predicted actions match the recorded commands.
    """
    import h5py
    recon_path = Path(recon_dir)
    hdf5_files = sorted(recon_path.glob("*.hdf5"))[:n_episodes]

    if not hdf5_files:
        print(f"No HDF5 files found in {recon_dir}")
        return {}

    print(f"\n── NeMo-WM RECON Eval ({len(hdf5_files)} episodes) ──")
    results = []

    for ep_path in hdf5_files:
        with h5py.File(ep_path, "r") as f:
            lin_vel = f["commands/linear_velocity"][:]   # (T,)
            ang_vel = f["commands/angular_velocity"][:]  # (T,)
            gps     = f.get("gps/latlong", None)
            gps_arr = gps[:] if gps is not None else None

        T = len(lin_vel)
        agent.reset()
        action_errs = []
        da_vals     = []

        for t in range(min(T - 1, 300)):
            # Build obs from velocity commands (normalised)
            v_lin  = float(lin_vel[t])
            v_ang  = float(ang_vel[t])
            # Synthetic proprio obs: (vx, vy, vx_t+1, vy_t+1, heading_norm)
            obs = np.array([v_lin, v_ang,
                            lin_vel[t+1] if t+1 < T else 0.0,
                            ang_vel[t+1] if t+1 < T else 0.0,
                            0.0], dtype=np.float32)

            # Goal: move forward (simple goal)
            goal = np.array([1.0, 0.0])

            action, neuro, regime = agent.act(obs, goal)
            da_vals.append(neuro.da)

            # Compare predicted action to actual next command
            actual_action = np.array([
                float(lin_vel[t+1]) if t+1 < T else 0.0,
                float(ang_vel[t+1]) if t+1 < T else 0.0,
            ])
            # Normalise to [0,1] for comparison
            act_norm    = np.clip((action + 1) / 2, 0, 1)
            actual_norm = np.clip((actual_action + 1) / 2, 0, 1)
            action_errs.append(float(np.linalg.norm(act_norm - actual_norm)))

        mean_err = float(np.mean(action_errs)) if action_errs else 1.0
        mean_da  = float(np.mean(da_vals))     if da_vals     else 0.5
        results.append({"file": ep_path.name, "action_err": mean_err,
                        "da_mean": mean_da, "steps": len(action_errs)})

        if verbose:
            print(f"  {ep_path.name}: err={mean_err:.3f}  "
                  f"DA={mean_da:.2f}  steps={len(action_errs)}")

    mean_err  = float(np.mean([r["action_err"] for r in results]))
    mean_da   = float(np.mean([r["da_mean"]    for r in results]))
    print(f"\n══ RECON action err={mean_err:.3f}  DA={mean_da:.2f} ══")
    return {"mean_action_err": mean_err, "mean_da": mean_da, "results": results}


# ═══════════════════════════════════════════════════════════════════════════
# Agent factory
# ═══════════════════════════════════════════════════════════════════════════

def build_agent(args, device: torch.device) -> NeMoAgent:
    """Load all components and assemble the full agent."""
    from train_action_wm import StateEncoder, ActionConditionedTransition
    from block_probe import BlockProbe

    # Encoder + transition
    ckpt       = torch.load(args.ckpt, map_location=device, weights_only=False)
    obs_dim    = ckpt.get("obs_dim", 5)
    action_dim = ckpt.get("action_dim", 2)
    D          = 128

    encoder = StateEncoder(obs_dim, D).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()

    transition = ActionConditionedTransition(D, action_dim).to(device)
    transition.load_state_dict(ckpt["transition"])
    transition.eval()

    diag = ckpt.get("diagnostics", {})
    print(f"World model: ep={ckpt.get('epoch','?')} "
          f"ac_lift={diag.get('ac_lift',0):+.4f}")

    # Block probe
    probe_ckpt = torch.load(args.probe, map_location=device, weights_only=False)
    probe = BlockProbe(probe_ckpt.get("d_model", 128)).to(device)
    probe.load_state_dict(probe_ckpt["probe"])
    probe.eval()
    print(f"Block probe: MAE={probe_ckpt.get('mae',0):.4f}")

    # Value network (Sprint B — optional)
    value_net = None
    if args.value_ckpt and Path(args.value_ckpt).exists():
        v_ckpt = torch.load(args.value_ckpt, map_location=device, weights_only=False)
        value_net = nn.Sequential(
            nn.Linear(128, 256), nn.GELU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 1), nn.Sigmoid()
        ).to(device)
        value_net.load_state_dict(v_ckpt["value_net"])
        value_net.eval()
        print(f"Value network: loaded (Sprint B active)")
    else:
        print(f"Value network: not loaded (using GoalDA probe scoring)")

    return NeMoAgent(
        encoder, transition, probe, action_dim, device,
        value_net  = value_net,
        base_ach   = args.base_ach,
        K          = args.n_candidates,
        n_elite    = args.n_elite,
        n_iters    = args.n_iters,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="NeMo-WM Full Closed Loop — perception to action"
    )
    p.add_argument("--ckpt",         required=True)
    p.add_argument("--probe",        required=True)
    p.add_argument("--value-ckpt",   default=None)
    p.add_argument("--n-episodes",   type=int,   default=50)
    p.add_argument("--max-steps",    type=int,   default=300)
    p.add_argument("--success-thr",  type=float, default=0.12)
    p.add_argument("--n-candidates", type=int,   default=512)
    p.add_argument("--n-elite",      type=int,   default=64)
    p.add_argument("--n-iters",      type=int,   default=3)
    p.add_argument("--base-ach",     type=float, default=0.2,
                   help="Base ACh level (0.2 → k_ctx=16, AUROC=0.9974)")
    p.add_argument("--synthetic",    action="store_true", default=True)
    p.add_argument("--recon",        action="store_true")
    p.add_argument("--recon-dir",    default="recon_data/recon_release")
    p.add_argument("--verbose",      action="store_true")
    p.add_argument("--device",       default="cpu")
    p.add_argument("--seed",         type=int, default=99)
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"\n══ NeMo-WM Closed Loop Agent ══")
    print(f"  Device: {device}")
    print(f"  K={args.n_candidates} H_base=8 n_elite={args.n_elite}")
    print(f"  ACh={args.base_ach} → k_ctx={NeuromodulatorBus(args.base_ach)._ach_to_kctx(args.base_ach)}")
    if args.value_ckpt:
        print(f"  Sprint B: value function active")
    print()

    agent = build_agent(args, device)

    if args.recon:
        eval_recon(agent, args.recon_dir,
                   n_episodes=args.n_episodes, verbose=args.verbose)
    else:
        eval_synthetic(agent, n_episodes=args.n_episodes,
                       max_steps=args.max_steps,
                       success_thr=args.success_thr,
                       seed=args.seed, verbose=args.verbose)
