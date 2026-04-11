"""
eval_goal_reaching.py  —  NeMo-WM Goal Reaching Evaluation
===========================================================
Sprint A: GRASP planner wired in, replacing mirror_ascent_step().

Pipeline:
  obs → encoder → z → block_probe → block_pos
  block_pos + goal → GoalDA (hot/cold) → DA signal
  DA → regime_gated_plan() → action
  action → env step → next obs

Three planners compared:
  1. mirror_ascent  — BoK + KL-anchored mirror ascent (baseline)
  2. cem            — Cross-entropy method with latent distance scoring
  3. grasp          — GRASP + GoalDA probe scoring (Sprint A)

Sprint B value function slot is reserved (--planner value).

Usage:
    python eval_goal_reaching.py \
        --ckpt  checkpoints/action_wm/action_wm_pusht_full_best.pt \
        --probe checkpoints/action_wm/block_probe_best.pt \
        --planner grasp \
        --n-episodes 50 --synthetic

    # Compare all three
    python eval_goal_reaching.py \
        --ckpt  checkpoints/action_wm/action_wm_pusht_full_best.pt \
        --probe checkpoints/action_wm/block_probe_best.pt \
        --compare --n-episodes 50 --synthetic
"""

from __future__ import annotations
import argparse, math, time
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# Checkpoint loading
# ═══════════════════════════════════════════════════════════════════════════

def load_checkpoint(ckpt_path: str, device: torch.device):
    """Load encoder + transition from action_wm checkpoint."""
    from train_action_wm import StateEncoder, ActionConditionedTransition
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    obs_dim    = ckpt.get("obs_dim", 5)
    action_dim = ckpt.get("action_dim", 2)
    D          = 128
    encoder    = StateEncoder(obs_dim, D).to(device)
    transition = ActionConditionedTransition(D, action_dim).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    transition.load_state_dict(ckpt["transition"])
    encoder.eval(); transition.eval()
    diag = ckpt.get("diagnostics", {})
    print(f"Loaded: {ckpt_path}")
    print(f"  obs_dim={obs_dim} action_dim={action_dim}")
    print(f"  epoch={ckpt.get('epoch','?')} loss={ckpt.get('loss',0):.4f}")
    print(f"  ac_lift={diag.get('ac_lift',0):+.4f}")
    return encoder, transition, obs_dim, action_dim


def load_probe(probe_path: str, device: torch.device):
    """Load block position probe."""
    from block_probe import BlockProbe
    ckpt  = torch.load(probe_path, map_location=device, weights_only=False)
    probe = BlockProbe(ckpt.get("d_model", 128)).to(device)
    probe.load_state_dict(ckpt["probe"])
    probe.eval()
    print(f"Block probe loaded: MAE={ckpt.get('mae',0):.4f} "
          f"({ckpt.get('mae',0)*100:.1f}% arena)")
    return probe


# ═══════════════════════════════════════════════════════════════════════════
# GoalDA — hot/cold reward shaping
# ═══════════════════════════════════════════════════════════════════════════

class GoalDA:
    """
    Continuous hot/cold DA signal for goal-directed planning.
    Phase 1: agent→block approach shaping (pre-contact)
    Phase 2: block→goal approach shaping (post-contact)
    Discrete spikes on contact onset and goal proximity.
    """

    def __init__(
        self,
        decay:          float = 0.92,
        approach_scale: float = 4.0,
        retreat_scale:  float = 3.0,
        contact_thr:    float = 0.15,
        goal_thr:       float = 0.10,
    ):
        self.decay          = decay
        self.approach_scale = approach_scale
        self.retreat_scale  = retreat_scale
        self.contact_thr    = contact_thr
        self.goal_thr       = goal_thr
        self._da            = 0.5
        self._prev_ab       = None
        self._prev_bg       = None
        self._prev_contact  = False
        self._prev_goal     = False
        self._history: List[float] = []

    def update(self, obs: np.ndarray, goal: np.ndarray) -> float:
        agent = obs[:2]; block = obs[2:4]
        dist_ab = float(np.linalg.norm(agent - block))
        dist_bg = float(np.linalg.norm(block - goal))
        contact   = dist_ab < self.contact_thr
        near_goal = dist_bg < self.goal_thr

        spike = 0.0
        if contact   and not self._prev_contact: spike = 1.0
        if near_goal and not self._prev_goal:    spike = max(spike, 1.0)

        shaping = 0.0
        if not contact and self._prev_ab is not None:
            d = self._prev_ab - dist_ab
            shaping += d * (self.approach_scale if d > 0.005 else
                            self.retreat_scale)  # retreat_scale used as penalty when d<0
        if contact and self._prev_bg is not None:
            d = self._prev_bg - dist_bg
            shaping += d * (self.approach_scale if d > 0.003 else
                            self.retreat_scale)

        self._da = float(np.clip(
            self._da * self.decay + spike + shaping, 0.0, 1.0
        ))
        self._prev_ab      = dist_ab
        self._prev_bg      = dist_bg
        self._prev_contact = contact
        self._prev_goal    = near_goal
        self._history.append(self._da)
        return self._da

    def reset(self):
        self._da = 0.5; self._prev_ab = None; self._prev_bg = None
        self._prev_contact = False; self._prev_goal = False
        self._history.clear()

    @property
    def da(self) -> float: return self._da

    def temperature(self) -> str:
        if self._da > 0.8:  return "HOT"
        if self._da > 0.6:  return "WARM"
        if self._da > 0.4:  return "NEUTRAL"
        if self._da > 0.2:  return "COOL"
        return                      "COLD"

    def summary(self) -> dict:
        h = np.array(self._history) if self._history else np.array([0.5])
        return {"da_mean": float(h.mean()), "da_max": float(h.max()),
                "hot_pct": float((h > 0.6).mean())}


# ═══════════════════════════════════════════════════════════════════════════
# Planner 1 — Mirror Ascent (baseline)
# ═══════════════════════════════════════════════════════════════════════════

class MirrorAscentPlanner:
    """
    Best-of-K + KL-anchored mirror ascent.
    Baseline planner — scores by latent distance, no probe.
    """

    def __init__(
        self,
        encoder:    nn.Module,
        transition: nn.Module,
        action_dim: int,
        device:     torch.device,
        K:          int   = 512,
        horizon:    int   = 8,
        lr:         float = 0.05,
        kl_lambda:  float = 0.1,
    ):
        self.enc   = encoder
        self.trans = transition
        self.K     = K
        self.H     = horizon
        self.lr    = lr
        self.kl    = kl_lambda
        self.device = device
        self.act_dim = action_dim
        self._last_action = np.zeros(action_dim)

    @torch.no_grad()
    def plan(self, obs: np.ndarray, goal_obs_or_da=None) -> np.ndarray:
        if obs[:4].max() > 1.5:
            obs = obs.copy(); obs[:4] /= 512.0
        z_cur  = self.enc(torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device))
        # Use real goal if passed as array, else default to center
        if hasattr(goal_obs_or_da, "__len__") and len(goal_obs_or_da) == 2:
            g = goal_obs_or_da
        else:
            g = np.array([0.65, 0.65])
        goal_obs = np.array([obs[0], obs[1], g[0], g[1], 0.0], dtype=np.float32)
        z_goal = self.enc(torch.from_numpy(goal_obs).unsqueeze(0).to(self.device))

        actions = torch.clamp(
            torch.randn(self.K, self.H, self.act_dim, device=self.device, dtype=torch.float32) * 0.3
            + torch.tensor(self._last_action, dtype=torch.float32, device=self.device),
            -1.0, 1.0
        )
        z = z_cur.expand(self.K, -1)
        for h in range(self.H):
            z = self.trans(z, actions[:, h])

        scores = -torch.norm(z - z_goal, dim=-1)  # latent distance
        # KL anchor toward last action
        kl_pen = self.kl * torch.norm(
            actions[:, 0] - torch.tensor(self._last_action, device=self.device), dim=-1
        )
        scores = scores - kl_pen

        best = actions[scores.argmax(), 0].cpu().numpy()
        self._last_action = best
        return np.clip(best, -1.0, 1.0)

    def reset(self): self._last_action = np.zeros(self.act_dim)


# ═══════════════════════════════════════════════════════════════════════════
# Planner 2 — CEM with latent distance
# ═══════════════════════════════════════════════════════════════════════════

class CEMPlanner:
    """Cross-entropy method with latent distance scoring."""

    def __init__(self, encoder, transition, action_dim, device,
                 K=512, horizon=8, n_elite=64, n_iters=3):
        self.enc   = encoder; self.trans = transition
        self.K     = K; self.H = horizon
        self.n_elite = n_elite; self.n_iters = n_iters
        self.device = device; self.act_dim = action_dim

    @torch.no_grad()
    def plan(self, obs: np.ndarray, goal_obs_or_da=None) -> np.ndarray:
        if obs[:4].max() > 1.5:
            obs = obs.copy(); obs[:4] /= 512.0
        z_cur  = self.enc(torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device))
        # Use real goal if passed as array, else default to center
        if hasattr(goal_obs_or_da, "__len__") and len(goal_obs_or_da) == 2:
            g = goal_obs_or_da
        else:
            g = np.array([0.65, 0.65])
        goal_obs = np.array([obs[0], obs[1], g[0], g[1], 0.0], dtype=np.float32)
        z_goal = self.enc(torch.from_numpy(goal_obs).unsqueeze(0).to(self.device))

        mu  = torch.zeros(self.H, self.act_dim, device=self.device, dtype=torch.float32)
        std = torch.ones(self.H, self.act_dim, device=self.device, dtype=torch.float32) * 0.5

        for _ in range(self.n_iters):
            actions = torch.clamp(
                mu + std * torch.randn(self.K, self.H, self.act_dim, device=self.device, dtype=torch.float32),
                -1.0, 1.0
            )
            z = z_cur.expand(self.K, -1)
            for h in range(self.H):
                z = self.trans(z, actions[:, h].float())
            scores    = -torch.norm(z - z_goal, dim=-1)
            elite_idx = scores.topk(self.n_elite).indices
            mu  = actions[elite_idx].mean(0)
            std = actions[elite_idx].std(0).clamp(min=0.05)

        return actions[scores.argmax(), 0].cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════
# Planner 3 — GRASP + GoalDA (Sprint A)
# ═══════════════════════════════════════════════════════════════════════════

class GRASPPlanner:
    """
    GRASP-style planner with GoalDA probe scoring.

    Replaces latent distance with GoalDA warmth accumulated
    along imagined trajectories — decoded through block probe.

    The planner scores K rollouts by:
      sum_t [ DA_shaping(probe(z_t), goal) ]
    instead of:
      -||z_final - z_goal||

    This closes the discriminative→generative gap:
    the probe decodes task-relevant geometry (block_pos)
    from the discriminative latent, and GoalDA provides
    the hot/cold gradient that latent distance cannot.

    regime_gated_plan() applies DA to gate planning horizon:
      hot (DA > 0.7) → extend H → plan further ahead
      cold (DA < 0.3) → shrink H → don't waste compute
    """

    def __init__(
        self,
        encoder:    nn.Module,
        transition: nn.Module,
        probe:      nn.Module,
        action_dim: int,
        device:     torch.device,
        goal:       np.ndarray,
        K:          int   = 512,
        horizon:    int   = 8,
        n_elite:    int   = 64,
        n_iters:    int   = 3,
        approach_scale: float = 4.0,
        retreat_scale:  float = 3.0,
        goal_thr:   float = 0.10,
    ):
        self.enc   = encoder
        self.trans = transition
        self.probe = probe
        self.goal  = goal
        self.K     = K
        self.H     = horizon
        self.n_elite    = n_elite
        self.n_iters    = n_iters
        self.approach_scale = approach_scale
        self.retreat_scale  = retreat_scale
        self.goal_thr   = goal_thr
        self.device     = device
        self.act_dim    = action_dim
        self._goal_da   = GoalDA(approach_scale=approach_scale,
                                 retreat_scale=retreat_scale,
                                 goal_thr=goal_thr)

    def regime_gated_plan(
        self,
        obs:    np.ndarray,
        da:     float,
    ) -> np.ndarray:
        """
        DA-gated planning horizon.
        Hot  → extend H (plan further ahead when close to goal)
        Cold → shrink H (don't waste compute when lost)
        """
        H = max(4, min(16, int(self.H * (0.5 + da))))
        return self._plan_with_horizon(obs, H)

    @torch.no_grad()
    def _score_rollout(
        self,
        z_preds: List[torch.Tensor],
        prev_dist: np.ndarray,
    ) -> torch.Tensor:
        """Score K rollouts by accumulated GoalDA warmth."""
        K       = z_preds[0].shape[0]
        scores  = torch.zeros(K, device=self.device)
        goal_t  = torch.tensor(self.goal, dtype=torch.float32, device=self.device)
        prev_dt = torch.tensor(prev_dist, dtype=torch.float32, device=self.device)

        for z_step in z_preds:
            block_pred = self.probe(z_step)                          # (K, 2)
            dist       = torch.norm(block_pred - goal_t, dim=-1)    # (K,)
            delta      = prev_dt - dist                              # + = closer

            approach_mask = delta >  0.003
            retreat_mask  = delta < -0.003
            near_goal     = dist  < self.goal_thr

            scores += approach_mask.float() * delta * self.approach_scale
            scores += retreat_mask.float()  * delta * self.retreat_scale
            scores += near_goal.float()     * 2.0
            prev_dt = dist

        return scores

    @torch.no_grad()
    def _plan_with_horizon(self, obs: np.ndarray, H: int) -> np.ndarray:
        if obs[:4].max() > 1.5:
            obs = obs.copy(); obs[:4] /= 512.0

        z_cur = self.enc(
            torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        )

        # Initial block distance for DA shaping baseline
        block_cur  = self.probe(z_cur).cpu().numpy()[0]
        prev_dist  = np.full(self.K, np.linalg.norm(block_cur - self.goal))

        # CEM loop
        mu  = torch.full((H, self.act_dim), 0.5, device=self.device, dtype=torch.float32)
        std = torch.full((H, self.act_dim), 0.3, device=self.device, dtype=torch.float32)

        for _ in range(self.n_iters):
            actions = torch.clamp(
                mu + std * torch.randn(self.K, H, self.act_dim, device=self.device, dtype=torch.float32),
                0.0, 1.0
            )
            z      = z_cur.expand(self.K, -1)
            z_preds = []
            for h in range(H):
                z = self.trans(z, actions[:, h].float())
                z_preds.append(z)

            scores    = self._score_rollout(z_preds, prev_dist)
            elite_idx = scores.topk(self.n_elite).indices
            mu  = actions[elite_idx].mean(0)
            std = actions[elite_idx].std(0).clamp(min=0.05)

        return actions[scores.argmax(), 0].cpu().numpy()

    def plan(self, obs: np.ndarray, da: float = 0.5) -> np.ndarray:
        """Main entry point — DA-gated horizon selection."""
        return self.regime_gated_plan(obs, da)

    def reset(self):
        self._goal_da.reset()


# ═══════════════════════════════════════════════════════════════════════════
# Sprint B placeholder — Value function planner
# ═══════════════════════════════════════════════════════════════════════════

class ValueFunctionPlanner:
    """
    Sprint B: Learned value function V(z) → scalar reward.
    Replaces GoalDA heuristic with a trained value head.

    Architecture:
      V: z (128) → [256, 128] → 1  (trained on GoalDA-labelled rollouts)

    Training signal:
      V(z) = GoalDA(probe(z), goal) accumulated over episode
      Loss = MSE(V(z_t), sum_{t'>=t} gamma^{t'-t} * da_t')

    Once trained, the planner scores:
      score = V(z_final) instead of GoalDA warmth sum

    This gives a smooth gradient over the full latent space,
    not just near-contact/near-goal regions.

    Status: NOT YET TRAINED — placeholder for Sprint B.
    """

    def __init__(self, encoder, transition, probe, action_dim, device, goal,
                 value_ckpt: Optional[str] = None, K=512, horizon=8,
                 n_elite=64, n_iters=3):
        self.enc   = encoder; self.trans = transition; self.probe = probe
        self.goal  = goal; self.device = device; self.act_dim = action_dim
        self.K     = K; self.H = horizon
        self.n_elite = n_elite; self.n_iters = n_iters

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(128, 256), nn.GELU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 1), nn.Sigmoid()
        ).to(device)

        if value_ckpt and Path(value_ckpt).exists():
            ckpt = torch.load(value_ckpt, map_location=device, weights_only=False)
            self.value_net.load_state_dict(ckpt["value_net"])
            self.value_net.eval()
            print(f"Value network loaded from {value_ckpt}")
        else:
            print("Value network: NOT TRAINED (Sprint B pending)")
            print("  Falling back to GoalDA probe scoring")
            self._fallback = GRASPPlanner(
                encoder, transition, probe, action_dim, device, goal,
                K=K, horizon=horizon, n_elite=n_elite, n_iters=n_iters
            )

    @torch.no_grad()
    def plan(self, obs: np.ndarray, da: float = 0.5) -> np.ndarray:
        if hasattr(self, '_fallback'):
            return self._fallback.plan(obs, da)

        if obs[:4].max() > 1.5:
            obs = obs.copy(); obs[:4] /= 512.0
        z_cur = self.enc(
            torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        )
        H   = max(4, min(16, int(self.H * (0.5 + da))))
        mu  = torch.full((H, self.act_dim), 0.5, device=self.device, dtype=torch.float32)
        std = torch.full((H, self.act_dim), 0.3, device=self.device, dtype=torch.float32)

        for _ in range(self.n_iters):
            actions = torch.clamp(
                mu + std * torch.randn(self.K, H, self.act_dim, device=self.device, dtype=torch.float32),
                0.0, 1.0
            )
            z = z_cur.expand(self.K, -1)
            for h in range(H):
                z = self.trans(z, actions[:, h].float())
            scores    = self.value_net(z).squeeze(-1)   # (K,)
            elite_idx = scores.topk(self.n_elite).indices
            mu  = actions[elite_idx].mean(0)
            std = actions[elite_idx].std(0).clamp(min=0.05)

        return actions[scores.argmax(), 0].cpu().numpy()

    def reset(self):
        if hasattr(self, '_fallback'): self._fallback.reset()


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation loop — synthetic PushT physics
# ═══════════════════════════════════════════════════════════════════════════

def eval_synthetic(
    planner_name: str,
    planner,
    n_episodes:   int   = 50,
    max_steps:    int   = 300,
    success_thr:  float = 0.12,
    seed:         int   = 99,
) -> dict:
    """
    Evaluate a planner on synthetic PushT physics.
    Returns summary dict with SR, mean_dist, mean_steps, mean_da.
    """
    rng     = np.random.RandomState(seed)
    goals   = [np.array([0.65 + rng.uniform(-0.1, 0.1),
                         0.65 + rng.uniform(-0.1, 0.1)])
               for _ in range(n_episodes)]
    results = []

    print(f"\n── {planner_name} Planner (n={n_episodes}) ──")

    for ep in range(n_episodes):
        goal  = goals[ep]
        agent = rng.uniform(0.1, 0.4, 2)
        block = rng.uniform(0.3, 0.6, 2)
        angle = rng.uniform(0, 2 * math.pi)

        # Per-planner goal update
        if hasattr(planner, 'goal'):
            planner.goal = goal
        if hasattr(planner, '_fallback') and hasattr(planner._fallback, 'goal'):
            planner._fallback.goal = goal

        goal_da = GoalDA()
        success = False
        da_mean = 0.5

        for step in range(max_steps):
            obs = np.array([agent[0], agent[1], block[0], block[1],
                            angle / (2 * math.pi)], dtype=np.float32)

            # GoalDA update
            da = goal_da.update(obs, goal)

            # Planner step
            # Pass real goal to all planners
            goal_obs = np.array([obs[0], obs[1], goal[0], goal[1], 0.0], dtype=np.float32)
            if planner_name == "mirror_ascent":
                action = planner.plan(obs, goal_obs)
            elif planner_name == "cem":
                action = planner.plan(obs, goal)  # pass real goal array
            else:
                action = planner.plan(obs, da)

            action = np.clip(action, 0.0, 1.0)

            # Synthetic physics
            agent += (action - agent) * 0.4 + rng.normal(0, 0.01, 2)
            agent  = np.clip(agent, 0, 1)
            if np.linalg.norm(agent - block) < 0.1:
                push   = (agent - block) * 0.2
                block  = np.clip(block - push, 0, 1)
                angle += rng.normal(0, 0.05)

            dist = np.linalg.norm(block - goal)
            if dist < success_thr:
                success = True
                break

        da_info  = goal_da.summary()
        da_mean  = da_info.get("da_mean", 0.5)
        results.append({
            "success":    success,
            "steps":      step + 1,
            "final_dist": float(dist),
            "da_mean":    da_mean,
        })
        temp = goal_da.temperature()
        print(f"  ep{ep+1:02d} {'✓' if success else '✗'}  "
              f"steps={step+1:3d}  dist={dist:.3f}  "
              f"DA={da_mean:.2f}  {temp}")

    sr        = float(np.mean([r["success"]    for r in results]))
    mean_dist = float(np.mean([r["final_dist"] for r in results]))
    mean_steps= float(np.mean([r["steps"]      for r in results]))
    mean_da   = float(np.mean([r["da_mean"]    for r in results]))

    print(f"\n══ {planner_name} SR: {sr:.1%}  "
          f"steps={mean_steps:.1f}  dist={mean_dist:.3f}  "
          f"DA={mean_da:.2f} ══")

    return {"planner": planner_name, "sr": sr, "mean_dist": mean_dist,
            "mean_steps": mean_steps, "mean_da": mean_da}


# ═══════════════════════════════════════════════════════════════════════════
# Value function training (Sprint B)
# ═══════════════════════════════════════════════════════════════════════════

def train_value_function(
    encoder:    nn.Module,
    transition: nn.Module,
    probe:      nn.Module,
    device:     torch.device,
    save_path:  str,
    n_episodes: int   = 2000,
    gamma:      float = 0.95,
    epochs:     int   = 20,
    batch_size: int   = 512,
    lr:         float = 1e-3,
    seed:       int   = 42,
) -> nn.Module:
    """
    Sprint B: Train value function V(z) → GoalDA return.

    Collects rollouts using GRASP planner (GoalDA guidance),
    computes discounted GoalDA returns as targets,
    trains a 3-layer MLP to predict them from latent z.

    This upgrades the planner from heuristic (GoalDA sum)
    to learned (V(z) approximation) — smoother gradient,
    better SR expected.
    """
    print(f"\n══ Sprint B: Training Value Function ══")
    print(f"  Collecting {n_episodes} rollouts for training data...")

    rng = np.random.RandomState(seed)
    value_net = nn.Sequential(
        nn.Linear(128, 256), nn.GELU(), nn.LayerNorm(256),
        nn.Linear(256, 128), nn.GELU(),
        nn.Linear(128, 1), nn.Sigmoid()
    ).to(device)
    opt = torch.optim.AdamW(value_net.parameters(), lr=lr)

    # ── Collect rollout data ─────────────────────────────────────────────
    all_z, all_returns = [], []

    goals_train = [np.array([0.65 + rng.uniform(-0.15, 0.15),
                              0.65 + rng.uniform(-0.15, 0.15)])
                   for _ in range(n_episodes)]

    grasp = GRASPPlanner(
        encoder, transition, probe, 2, device,
        goal=goals_train[0], K=256, horizon=6, n_elite=32, n_iters=2
    )

    for ep in range(n_episodes):
        goal  = goals_train[ep]
        grasp.goal = goal
        agent = rng.uniform(0.1, 0.4, 2)
        block = rng.uniform(0.3, 0.6, 2)
        angle = rng.uniform(0, 2 * math.pi)

        goal_da  = GoalDA()
        ep_z:    List[torch.Tensor] = []
        ep_da:   List[float]        = []

        for step in range(150):
            obs = np.array([agent[0], agent[1], block[0], block[1],
                            angle / (2 * math.pi)], dtype=np.float32)

            with torch.no_grad():
                z = encoder(torch.from_numpy(obs).unsqueeze(0).to(device))
            ep_z.append(z.squeeze(0).cpu())

            da     = goal_da.update(obs, goal)
            ep_da.append(da)
            action = np.clip(grasp.plan(obs, da), 0.0, 1.0)

            agent += (action - agent) * 0.4 + rng.normal(0, 0.01, 2)
            agent  = np.clip(agent, 0, 1)
            if np.linalg.norm(agent - block) < 0.1:
                push   = (agent - block) * 0.2
                block  = np.clip(block - push, 0, 1)
                angle += rng.normal(0, 0.05)

        # Compute discounted returns
        T = len(ep_da)
        returns = np.zeros(T)
        G = 0.0
        for t in range(T - 1, -1, -1):
            G = ep_da[t] + gamma * G
            returns[t] = G
        # Normalise returns to [0, 1]
        returns = (returns - returns.min()) / (returns.max() - returns.min() + 1e-8)

        all_z.extend(ep_z)
        all_returns.extend(returns.tolist())

        if (ep + 1) % 200 == 0:
            print(f"  Collected {ep+1}/{n_episodes} episodes "
                  f"(mean_da={np.mean(ep_da):.3f})")

    z_tensor = torch.stack(all_z)
    r_tensor  = torch.tensor(all_returns, dtype=torch.float32).unsqueeze(1)
    dataset   = torch.utils.data.TensorDataset(z_tensor, r_tensor)
    loader    = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    # ── Train value network ──────────────────────────────────────────────
    print(f"\n  Training value network on {len(all_z):,} samples...")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(epochs):
        value_net.train()
        losses = []
        for z_b, r_b in loader:
            z_b = z_b.to(device); r_b = r_b.to(device)
            pred = value_net(z_b)
            loss = F.mse_loss(pred, r_b)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())

        mean_L = float(np.mean(losses))
        print(f"  Epoch {epoch:02d}  loss={mean_L:.4f}")

        if mean_L < best_loss:
            best_loss = mean_L
            torch.save({
                "epoch":      epoch,
                "loss":       best_loss,
                "value_net":  value_net.state_dict(),
                "gamma":      gamma,
            }, save_path)
            print(f"  → Saved: {save_path}")

    print(f"\nValue function training complete. Best loss: {best_loss:.4f}")
    return value_net


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def build_planner(name: str, encoder, transition, probe, action_dim,
                  device, goal, args):
    """Factory for planners by name."""
    if name == "mirror_ascent":
        return MirrorAscentPlanner(encoder, transition, action_dim, device,
                                   K=args.n_candidates, horizon=args.horizon)
    if name == "cem":
        return CEMPlanner(encoder, transition, action_dim, device,
                          K=args.n_candidates, horizon=args.horizon,
                          n_elite=args.n_elite, n_iters=args.n_iters)
    if name == "grasp":
        return GRASPPlanner(encoder, transition, probe, action_dim, device,
                            goal=goal, K=args.n_candidates, horizon=args.horizon,
                            n_elite=args.n_elite, n_iters=args.n_iters)
    if name == "value":
        return ValueFunctionPlanner(
            encoder, transition, probe, action_dim, device, goal,
            value_ckpt=args.value_ckpt,
            K=args.n_candidates, horizon=args.horizon,
            n_elite=args.n_elite, n_iters=args.n_iters
        )
    raise ValueError(f"Unknown planner: {name}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="NeMo-WM Goal Reaching Evaluation")
    p.add_argument("--ckpt",         required=True, help="action_wm checkpoint")
    p.add_argument("--probe",        default=None,  help="block_probe checkpoint")
    p.add_argument("--value-ckpt",   default=None,  help="value function checkpoint (Sprint B)")
    p.add_argument("--planner",      default="grasp",
                   choices=["mirror_ascent","cem","grasp","value"],
                   help="Planner to use")
    p.add_argument("--compare",      action="store_true",
                   help="Compare all planners side by side")
    p.add_argument("--train-value",  action="store_true",
                   help="Train Sprint B value function")
    p.add_argument("--n-episodes",   type=int,   default=50)
    p.add_argument("--max-steps",    type=int,   default=300)
    p.add_argument("--success-thr",  type=float, default=0.12)
    p.add_argument("--n-candidates", type=int,   default=512)
    p.add_argument("--horizon",      type=int,   default=8)
    p.add_argument("--n-elite",      type=int,   default=64)
    p.add_argument("--n-iters",      type=int,   default=3)
    p.add_argument("--synthetic",    action="store_true", default=True)
    p.add_argument("--device",       default="cpu")
    p.add_argument("--seed",         type=int, default=99)
    args = p.parse_args()

    device = torch.device(args.device)
    encoder, transition, obs_dim, action_dim = load_checkpoint(args.ckpt, device)

    probe = None
    if args.probe and Path(args.probe).exists():
        probe = load_probe(args.probe, device)

    default_goal = np.array([0.65, 0.65])

    # ── Train value function (Sprint B) ──────────────────────────────────
    if args.train_value:
        if probe is None:
            print("ERROR: --probe required for --train-value")
            exit(1)
        train_value_function(
            encoder, transition, probe, device,
            save_path="checkpoints/action_wm/value_fn_best.pt",
            n_episodes=2000, epochs=20,
        )
        exit(0)

    # ── Single planner eval ───────────────────────────────────────────────
    if not args.compare:
        planner = build_planner(
            args.planner, encoder, transition, probe,
            action_dim, device, default_goal, args
        )
        eval_synthetic(args.planner, planner,
                       n_episodes=args.n_episodes,
                       max_steps=args.max_steps,
                       success_thr=args.success_thr,
                       seed=args.seed)
        exit(0)

    # ── Comparison: all planners ──────────────────────────────────────────
    planners_to_compare = ["mirror_ascent", "cem", "grasp"]
    if args.value_ckpt: planners_to_compare.append("value")

    if probe is None and "grasp" in planners_to_compare:
        print("WARNING: no --probe provided, skipping grasp+value planners")
        planners_to_compare = ["mirror_ascent", "cem"]

    all_results = []
    for name in planners_to_compare:
        planner = build_planner(
            name, encoder, transition, probe,
            action_dim, device, default_goal, args
        )
        r = eval_synthetic(name, planner,
                           n_episodes=args.n_episodes,
                           max_steps=args.max_steps,
                           success_thr=args.success_thr,
                           seed=args.seed)
        all_results.append(r)

    print("\n\n══════════════════════════════════════════════════")
    print("  COMPARISON SUMMARY")
    print("══════════════════════════════════════════════════")
    print(f"  {'Planner':<16} {'SR':>6} {'Dist':>7} {'Steps':>7} {'DA':>6}")
    print("  " + "─"*46)
    for r in sorted(all_results, key=lambda x: x["sr"], reverse=True):
        marker = " ◆" if r["sr"] == max(x["sr"] for x in all_results) else ""
        print(f"  {r['planner']:<16} {r['sr']:>5.1%} "
              f"{r['mean_dist']:>7.3f} {r['mean_steps']:>7.1f} "
              f"{r['mean_da']:>6.2f}{marker}")
    print("══════════════════════════════════════════════════")
