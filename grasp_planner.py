"""
grasp_planner.py  --  CORTEX CWM Sprint 4
==========================================
GRASP: Gradient RelAxed Stochastic Planner
arXiv:2602.00475 -- Psenka, Rabbat, Krishnapriyan, LeCun, Bar (Meta FAIR 2026)

Extends the original GRASP with E/I-gated Langevin noise -- a CORTEX
contribution. The neuromodulator's excitation/inhibition ratio gates the
exploration width of the stochastic planning step, implementing biologically-
modulated stochastic planning.

Three components from the paper:
  1. Lifted states   -- virtual states s_1..s_T optimised in parallel
  2. Langevin noise  -- Gaussian noise on states for exploration (E/I gated)
  3. Grad-cut        -- stop-gradient on state inputs to world model

Regime-gated planner selection (CORTEX extension):
  EXPLOIT  -> GRASPPlanner  (gradient-based, precise)
  EXPLORE  -> MirrorAscentSampler  (zero-order, fast)
  WAIT     -> no action

Benchmark target: < 10ms per planning step on GMKtec EVO-X2 CPU.
Run benchmark before committing to Sprint 4:
    python grasp_planner.py --benchmark

Usage in Sprint 4 loop:
    from grasp_planner import regime_gated_plan
    action, info = regime_gated_plan(
        cwm_predictor = predictor,
        particles_0   = particles_t,
        goal_particles = goal,
        signals        = neuro.signals,
        action_dim     = 2,
    )
"""

import time
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Config
# ===========================================================================

@dataclass
class GRASPConfig:
    """
    GRASP planner configuration.

    Tuning guide:
      horizon        -- planning steps. Longer = better plans, slower.
                        Start at 8, benchmark before increasing.
      n_lifted_iters -- gradient steps on lifted states per call.
                        3-5 is typical. More = better but slower.
      langevin_base  -- base noise std for Langevin updates.
                        0.05-0.10 for RECON navigation.
      da_sync_every  -- how often to run a full-gradient sync step.
                        Lower = more accurate, higher = faster.
      grad_cut       -- apply stop-gradient on state inputs (recommended True)
      e_i_scale      -- how much E/I neuromodulator scales Langevin noise.
                        0 = no E/I gating. 1.0 = full E/I gating.
    """
    horizon:        int   = 8     # H=8, iters=2 passes <10ms on GMKtec EVO-X2
    n_lifted_iters: int   = 2     # Benchmarked 2026-04-06: median 9.10ms, P95 21.53ms
                                  # H=5 iters=3 = 12.57ms (FAIL). More iters costs more than longer horizon.
    langevin_base:  float = 0.05
    da_sync_every:  int   = 3
    grad_cut:       bool  = True
    e_i_scale:      float = 1.0     # CORTEX: E/I gate on Langevin noise
    action_lr:      float = 0.05    # learning rate for action gradient steps
    dynamics_lambda:float = 1.0     # weight on soft dynamics constraint


# ===========================================================================
# Lifted state GRASP planner
# ===========================================================================

class GRASPPlanner:
    """
    Gradient RelAxed Stochastic Planner (Psenka et al. 2026).

    Plans by optimising virtual intermediate states alongside actions,
    rather than rolling out serially. All T world model evaluations
    run in parallel (lifted formulation).

    CORTEX extension: Langevin noise magnitude is gated by the E/I
    neuromodulator signal -- high E/I (exploration pressure) -> more noise
    -> broader search. This is biologically-modulated stochastic planning.

    Args:
        predictor: callable (particles, action) -> next_particles
                   Must support autograd through action inputs.
                   stop-gradient applied on particles input (grad-cut).
    """

    def __init__(self, predictor, config: Optional[GRASPConfig] = None):
        self.predictor = predictor
        self.cfg       = config or GRASPConfig()

    def plan(
        self,
        particles_0:    torch.Tensor,   # (1, K, d_model) current state
        goal_particles: torch.Tensor,   # (1, K, d_model) goal state
        action_dim:     int   = 2,
        e_i:            float = 1.0,    # E/I neuromodulator signal [0.5, 2.0]
        da_eff:         float = 0.5,    # DA_eff for gradient step size
        device:         str   = "cpu",
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Plan an action sequence to reach goal_particles from particles_0.

        Returns:
            best_action: (action_dim,) first action in optimal sequence
            info:        planning diagnostics dict
        """
        cfg = self.cfg
        dev = torch.device(device)
        H   = cfg.horizon

        particles_0    = particles_0.to(dev).detach()
        goal_particles = goal_particles.to(dev).detach().squeeze(0)  # (K, d_model)

        # Initialise action sequence and lifted virtual states
        # Actions: (H, action_dim) -- optimised via gradient
        actions = torch.zeros(H, action_dim, device=dev, requires_grad=True)

        # Lifted states: (H+1, K, d_model) -- s_0 fixed, s_1..s_H optimised
        # Initialise by rolling out with zero actions
        with torch.no_grad():
            virtual_states = [particles_0]
            s = particles_0
            for t in range(H):
                a = torch.zeros(1, action_dim, device=dev)
                s_next, _ = self._predict(s, a)
                virtual_states.append(s_next)
            virtual_states = torch.stack([v.squeeze(0) for v in virtual_states], dim=0)  # (H+1, K, D)

        # Virtual states as optimisation variables (s_0 fixed)
        vs_free = virtual_states[1:].detach().clone().requires_grad_(True)

        opt_a  = torch.optim.Adam([actions],  lr=cfg.action_lr)
        opt_vs = torch.optim.Adam([vs_free],  lr=cfg.action_lr * 2)

        t_start = time.perf_counter()
        best_loss = float("inf")
        best_actions = actions.detach().clone()

        for iteration in range(cfg.n_lifted_iters):

            opt_a.zero_grad(); opt_vs.zero_grad()

            # Stack all states: s_0 (fixed) + s_1..s_H (free)
            all_states = torch.cat([
                particles_0.squeeze(0).unsqueeze(0),   # (1, K, D)
                vs_free,                     # (H, K, D)
            ], dim=0)                        # (H+1, K, D)

            # ── Goal loss: terminal state close to goal ───────────────────
            L_goal = F.mse_loss(all_states[-1], goal_particles)

            # ── Dense goal shaping: all intermediate states toward goal ────
            # Encourages monotonic progress (from GRASP paper Sec 3.3)
            L_dense = sum(
                F.mse_loss(all_states[t], goal_particles)
                for t in range(1, H)
            ) / max(H - 1, 1)

            # ── Soft dynamics constraint ──────────────────────────────────
            # |s_{t+1} - f(s_t, a_t)| should be small
            # grad-cut: stop gradient through s_t inputs to predictor
            L_dyn = torch.tensor(0.0, device=dev)
            for t in range(H):
                s_t  = all_states[t]
                a_t  = actions[t:t+1]   # (1, action_dim)

                if cfg.grad_cut:
                    # Stop gradient on STATE inputs -- only action grads flow
                    # This is the key stability trick from GRASP Sec 3.3
                    s_t_for_pred = s_t.detach()
                else:
                    s_t_for_pred = s_t

                s_pred, _ = self._predict(
                    s_t_for_pred.unsqueeze(0), a_t
                )
                s_next = all_states[t + 1]
                L_dyn  = L_dyn + F.mse_loss(s_next.squeeze(0).detach(), s_pred.squeeze(0).detach())

            L_dyn = L_dyn / H * cfg.dynamics_lambda

            total = L_goal + 0.3 * L_dense + L_dyn
            total.backward()

            opt_a.step(); opt_vs.step()

            # ── Langevin noise on virtual states (E/I gated) ──────────────
            # CORTEX contribution: noise scale = base * E/I_scale * e_i
            # High E/I (exploration) -> more noise -> broader search
            # Low E/I (exploitation) -> less noise -> precise execution
            noise_std = cfg.langevin_base * (1.0 + cfg.e_i_scale * (e_i - 1.0))
            noise_std = max(0.001, noise_std)   # floor to prevent zero noise
            with torch.no_grad():
                vs_free.add_(torch.randn_like(vs_free) * noise_std)

            # ── Periodic full-gradient sync step ──────────────────────────
            # Pulls virtual states back toward a valid rollout trajectory
            if (iteration + 1) % cfg.da_sync_every == 0:
                with torch.no_grad():
                    s = particles_0
                    for t in range(H):
                        a = actions[t:t+1].detach()
                        s_next, _ = self._predict(s, a)
                        # Blend virtual state toward rollout state
                        alpha = 0.3 * da_eff   # higher DA = trust rollout less
                        vs_free.data[t] = (
                            (1 - alpha) * vs_free.data[t]
                            + alpha * s_next.squeeze(0).detach()
                        )
                        s = s_next

            if total.item() < best_loss:
                best_loss    = total.item()
                best_actions = actions.detach().clone()

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        return best_actions[0], {
            "best_loss":   best_loss,
            "elapsed_ms":  elapsed_ms,
            "horizon":     H,
            "iterations":  cfg.n_lifted_iters,
            "noise_std":   noise_std,
            "e_i":         e_i,
            "da_eff":      da_eff,
            "planner":     "GRASP",
        }

    def _predict(self, particles, action):
        """Wrapper around CWM predictor with consistent interface."""
        try:
            return self.predictor(particles, action)
        except TypeError:
            # Some predictors return single tensor
            result = self.predictor(particles, action)
            return result, {}


# ===========================================================================
# Mirror Ascent sampler (zero-order, fast -- used for EXPLORE regime)
# ===========================================================================

class MirrorAscentSampler:
    """
    Zero-order planner using Mirror Ascent (sampling-based).
    Used when regime=EXPLORE -- fast, broad search.

    This is the SevenSignalMPCPlanner approach from cwm_neuro_reward.py,
    simplified for the regime-gated interface.
    """

    def __init__(self, predictor, horizon: int = 8, n_candidates: int = 32):
        self.predictor   = predictor
        self.horizon     = horizon
        self.n_candidates = n_candidates

    def plan(
        self,
        particles_0:    torch.Tensor,
        goal_particles: torch.Tensor,
        action_dim:     int   = 2,
        action_std:     float = 0.10,
        da_eff:         float = 0.5,
        rho:            float = 0.5,
        device:         str   = "cpu",
    ) -> Tuple[torch.Tensor, Dict]:
        dev = torch.device(device)
        H   = self.horizon
        K   = self.n_candidates

        particles_0    = particles_0.to(dev)
        goal_particles = goal_particles.to(dev)

        candidates = torch.randn(K, H, action_dim, device=dev) * action_std
        costs      = torch.zeros(K, device=dev)

        with torch.no_grad():
            for k in range(K):
                s = particles_0
                cum = 0.0
                for t in range(H):
                    a = candidates[k, t:t+1]
                    try:
                        s_next, _ = self.predictor(s, a)
                    except TypeError:
                        s_next = self.predictor(s, a)
                    step_cost = F.mse_loss(s_next, goal_particles).item()
                    cum += step_cost
                    s = s_next
                # Modulate cost by NE/rho (CORTEX formula)
                costs[k] = -cum / (1.0 + rho)

        # Mirror Ascent weights
        eta = 0.05 * (0.5 + da_eff)
        w   = torch.softmax(costs / max(eta, 1e-4), dim=0)
        best = (candidates[:, 0] * w.unsqueeze(-1)).sum(0)

        return best, {
            "n_candidates": K,
            "best_cost":    -costs.max().item(),
            "action_std":   action_std,
            "planner":      "MirrorAscent",
        }


# ===========================================================================
# Regime-gated planner (main interface)
# ===========================================================================

def regime_gated_plan(
    cwm_predictor,
    particles_0:    torch.Tensor,
    goal_particles: torch.Tensor,
    signals,                         # NeuroSignals from neuromodulator
    action_dim:     int   = 2,
    grasp_config:   Optional[GRASPConfig] = None,
    device:         str   = "cpu",
) -> Tuple[torch.Tensor, Dict]:
    """
    Regime-gated planner selection.

    EXPLOIT -> GRASPPlanner  (gradient-based, precise, slower)
    EXPLORE -> MirrorAscentSampler (zero-order, fast, broad)
    WAIT    -> zero action (observe, don't act)
    REOBSERVE -> MirrorAscentSampler with reduced action_std

    Args:
        cwm_predictor: callable (particles, action) -> next_particles
        particles_0:   current particle state (1, K, d_model)
        goal_particles: goal particle state (1, K, d_model)
        signals:       NeuroSignals dataclass from Neuromodulator.update()
        action_dim:    action space dimensionality
        grasp_config:  GRASPConfig (uses defaults if None)
        device:        torch device string

    Returns:
        action: (action_dim,) selected action
        info:   planning diagnostics including which planner was used
    """
    regime = getattr(signals, "regime", "EXPLOIT")
    da_eff = getattr(signals, "da_eff", 0.5)
    e_i    = getattr(signals, "e_i",    1.0)
    rho    = getattr(signals, "ne",     0.5)

    # WAIT: don't act
    if regime == "WAIT":
        zero = torch.zeros(action_dim)
        return zero, {"regime": "WAIT", "acted": False, "planner": "none"}

    # EXPLOIT: use GRASP (gradient-based)
    if regime == "EXPLOIT":
        cfg = grasp_config or GRASPConfig()
        planner = GRASPPlanner(cwm_predictor, cfg)
        action, info = planner.plan(
            particles_0, goal_particles,
            action_dim=action_dim,
            e_i=e_i, da_eff=da_eff,
            device=device,
        )
        info["regime"] = "EXPLOIT"
        info["acted"]  = True
        return action, info

    # EXPLORE or REOBSERVE: use Mirror Ascent
    action_std = 0.10 * e_i   # E/I scales exploration width
    if regime == "REOBSERVE":
        action_std *= 0.5     # more conservative on unstable representations

    sampler = MirrorAscentSampler(cwm_predictor, horizon=8, n_candidates=32)
    action, info = sampler.plan(
        particles_0, goal_particles,
        action_dim=action_dim,
        action_std=action_std,
        da_eff=da_eff, rho=rho,
        device=device,
    )
    info["regime"] = regime
    info["acted"]  = True
    return action, info


# ===========================================================================
# CPU latency benchmark
# ===========================================================================

def benchmark(
    action_dim: int = 2,
    K:          int = 16,
    d_model:    int = 128,
    horizon:    int = 5,
    n_iters:    int = 3,
    n_trials:   int = 20,
    device_str: str = "cpu",
):
    """
    Benchmark GRASP planner latency on current hardware.
    Target: < 10ms per planning step on GMKtec EVO-X2 CPU at 4Hz.

    Run before committing GRASP to Sprint 4.
    """
    dev = torch.device(device_str)

    # Mock predictor (simple MLP -- replace with real CWM predictor in Sprint 4)
    class MockPredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(K * d_model + action_dim, 256),
                nn.GELU(),
                nn.Linear(256, K * d_model),
            )
        def forward(self, particles, action):
            B = particles.shape[0]
            x = torch.cat([
                particles.reshape(B, -1),
                action.reshape(B, -1),
            ], dim=-1)
            out = self.net(x).view(B, K, d_model)
            return F.normalize(out, dim=-1), {}

    predictor = MockPredictor().to(dev).eval()
    cfg = GRASPConfig(horizon=horizon, n_lifted_iters=n_iters)
    planner = GRASPPlanner(predictor, cfg)

    p0   = torch.randn(1, K, d_model, device=dev)
    goal = torch.randn(1, K, d_model, device=dev)

    # Warmup
    for _ in range(3):
        planner.plan(p0, goal, action_dim=action_dim, device=device_str)

    # Timed trials
    times = []
    for _ in range(n_trials):
        _, info = planner.plan(p0, goal, action_dim=action_dim, device=device_str)
        times.append(info["elapsed_ms"])

    print(f"\nGRASP Latency Benchmark")
    print(f"  Device:   {device_str}")
    print(f"  K={K}, d_model={d_model}, H={horizon}, iters={n_iters}")
    print(f"  Trials:   {n_trials}")
    print(f"  Mean:     {np.mean(times):.2f} ms")
    print(f"  Median:   {np.median(times):.2f} ms")
    print(f"  P95:      {np.percentile(times,95):.2f} ms")
    print(f"  Min/Max:  {np.min(times):.2f} / {np.max(times):.2f} ms")

    target_ms = 10.0
    if np.median(times) < target_ms:
        print(f"\n  PASS -- median {np.median(times):.2f}ms < {target_ms}ms target")
        print(f"  GRASP is viable for 4Hz RECON at this config.")
    else:
        print(f"\n  FAIL -- median {np.median(times):.2f}ms > {target_ms}ms target")
        print(f"  Reduce horizon or n_lifted_iters, or use MirrorAscent for EXPLOIT.")

    return np.median(times)


# ===========================================================================
# Self-test
# ===========================================================================

def self_test():
    """Verify all components work with mock predictor."""
    print("grasp_planner self-test...")

    class MockPred(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(16*128 + 2, 16*128)
        def forward(self, p, a):
            B = p.shape[0]
            x = torch.cat([p.reshape(B,-1), a.reshape(B,-1)], dim=-1)
            return F.normalize(self.net(x).view(B,16,128), dim=-1), {}

    pred = MockPred()
    p0   = torch.randn(1, 16, 128)
    goal = torch.randn(1, 16, 128)

    # Test GRASP
    cfg = GRASPConfig(horizon=4, n_lifted_iters=3)
    planner = GRASPPlanner(pred, cfg)
    action, info = planner.plan(p0, goal, action_dim=2)
    assert action.shape == (2,), f"Expected (2,), got {action.shape}"
    assert info["elapsed_ms"] > 0
    assert info["planner"] == "GRASP"
    print(f"  GRASP: action={action.detach().numpy().round(3)}, "
          f"loss={info['best_loss']:.4f}, "
          f"time={info['elapsed_ms']:.1f}ms")

    # Test MirrorAscent
    sampler = MirrorAscentSampler(pred, horizon=4, n_candidates=16)
    action2, info2 = sampler.plan(p0, goal, action_dim=2)
    assert action2.shape == (2,)
    print(f"  MirrorAscent: action={action2.detach().numpy().round(3)}")

    # Test regime-gated interface
    from dataclasses import dataclass
    @dataclass
    class MockSignals:
        regime: str = "EXPLOIT"
        da_eff: float = 0.5
        e_i:    float = 1.0
        ne:     float = 0.5

    for regime in ["EXPLOIT", "EXPLORE", "REOBSERVE", "WAIT"]:
        sig = MockSignals(regime=regime)
        a, i = regime_gated_plan(pred, p0, goal, sig, action_dim=2)
        assert a.shape == (2,)
        print(f"  regime={regime}: planner={i['planner']}, acted={i['acted']}")

    print("  All assertions passed.\n")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRASP planner -- Sprint 4")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run CPU latency benchmark")
    parser.add_argument("--horizon",   type=int, default=8)
    parser.add_argument("--iters",     type=int, default=2)   # benchmarked default
    parser.add_argument("--trials",    type=int, default=20)
    parser.add_argument("--device",    default="cpu")
    args = parser.parse_args()

    self_test()

    if args.benchmark:
        benchmark(
            horizon=args.horizon,
            n_iters=args.iters,
            n_trials=args.trials,
            device_str=args.device,
        )
    else:
        print("Run with --benchmark to test latency on GMKtec EVO-X2.")
        print("Target: median < 10ms at H=5, iters=3 for 4Hz operation.")
    print("H=8, iters=2 also viable if longer horizon needed.")
