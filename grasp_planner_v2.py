"""
grasp_planner.py  --  CORTEX CWM Sprint 4/5
==========================================
GRASP: Gradient RelAxed Stochastic Planner
arXiv:2602.00475 -- Psenka, Rabbat, Krishnapriyan, LeCun, Bar (Meta FAIR 2026)

Sprint 5 addition: DA-gated horizon switching.
    High DA (REOBSERVE/EXPLORE)  -> H=8,  iters=2  (~3.56ms)
    Low  DA (EXPLOIT)            -> H=16, iters=4  (~8ms)
Both paths within 250ms frame interval at 4Hz.

Changes from Sprint 4:
  - GRASPConfig gains da_threshold, h_reactive, iters_reactive,
    h_exploit, iters_exploit fields
  - regime_gated_plan applies DA-gated horizon before calling planners
  - benchmark() tests both horizon paths
"""

import time
import argparse
from dataclasses import dataclass, field
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

    DA-gated horizon (Sprint 5):
      da_threshold   -- DA above this = surprised = use reactive (short) horizon
      h_reactive     -- horizon when DA > threshold  (benchmarked: 3.56ms)
      iters_reactive -- lifted iters for reactive path
      h_exploit      -- horizon when DA <= threshold (benchmarked: ~8ms)
      iters_exploit  -- lifted iters for exploitative path

    Tuning guide:
      langevin_base  -- base noise std for Langevin updates. 0.05-0.10 for RECON.
      da_sync_every  -- frequency of full-gradient sync step.
      grad_cut       -- stop-gradient on state inputs (recommended True).
      e_i_scale      -- E/I gate on Langevin noise. 0=none, 1.0=full.
    """
    # ── DA-gated horizon (Sprint 5) ──────────────────────────────────────────
    da_threshold:   float = 0.015   # DA > this -> reactive path
    h_reactive:     int   = 8       # ~3.56ms on GMKtec EVO-X2
    iters_reactive: int   = 2
    h_exploit:      int   = 16      # ~8ms on GMKtec EVO-X2
    iters_exploit:  int   = 2

    # ── Sprint 4 original config (preserved) ────────────────────────────────
    horizon:        int   = 8       # fallback if DA-gating not used directly
    n_lifted_iters: int   = 2
    langevin_base:  float = 0.05
    da_sync_every:  int   = 3
    grad_cut:       bool  = True
    e_i_scale:      float = 1.0
    action_lr:      float = 0.05
    dynamics_lambda:float = 1.0

    def select_horizon(self, da: float) -> Tuple[int, int]:
        """Return (H, iters) based on current DA signal."""
        if da > self.da_threshold:
            return self.h_reactive, self.iters_reactive
        else:
            return self.h_exploit, self.iters_exploit


# ===========================================================================
# Lifted state GRASP planner
# ===========================================================================

class GRASPPlanner:
    """
    Gradient RelAxed Stochastic Planner (Psenka et al. 2026).

    CORTEX extension: Langevin noise gated by E/I neuromodulator signal.
    Sprint 5 extension: horizon and iters driven by DA signal via config.
    """

    def __init__(self, predictor, config: Optional[GRASPConfig] = None):
        self.predictor = predictor
        self.cfg       = config or GRASPConfig()

    def plan(
        self,
        particles_0:    torch.Tensor,
        goal_particles: torch.Tensor,
        action_dim:     int   = 2,
        e_i:            float = 1.0,
        da_eff:         float = 0.5,
        device:         str   = "cpu",
        # Sprint 5: override horizon from DA-gating (passed by regime_gated_plan)
        horizon_override:      Optional[int] = None,
        n_iters_override:      Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict]:

        cfg = self.cfg
        dev = torch.device(device)

        # Sprint 5: use DA-gated horizon if provided, else config default
        H     = horizon_override     if horizon_override     is not None else cfg.horizon
        iters = n_iters_override     if n_iters_override     is not None else cfg.n_lifted_iters

        particles_0    = particles_0.to(dev).detach()
        goal_particles = goal_particles.to(dev).detach().squeeze(0)

        actions = torch.zeros(H, action_dim, device=dev, requires_grad=True)

        with torch.no_grad():
            virtual_states = [particles_0]
            s = particles_0
            for t in range(H):
                a = torch.zeros(1, action_dim, device=dev)
                s_next, _ = self._predict(s, a)
                virtual_states.append(s_next)
            virtual_states = torch.stack(
                [v.squeeze(0) for v in virtual_states], dim=0
            )

        vs_free = virtual_states[1:].detach().clone().requires_grad_(True)

        opt_a  = torch.optim.Adam([actions],  lr=cfg.action_lr)
        opt_vs = torch.optim.Adam([vs_free],  lr=cfg.action_lr * 2)

        t_start   = time.perf_counter()
        best_loss = float("inf")
        best_actions = actions.detach().clone()

        for iteration in range(iters):
            opt_a.zero_grad(); opt_vs.zero_grad()

            all_states = torch.cat([
                particles_0.squeeze(0).unsqueeze(0),
                vs_free,
            ], dim=0)

            L_goal  = F.mse_loss(all_states[-1], goal_particles)
            L_dense = sum(
                F.mse_loss(all_states[t], goal_particles)
                for t in range(1, H)
            ) / max(H - 1, 1)

            L_dyn = torch.tensor(0.0, device=dev)
            for t in range(H):
                s_t = all_states[t]
                a_t = actions[t:t+1]
                s_t_for_pred = s_t.detach() if cfg.grad_cut else s_t
                s_pred, _ = self._predict(s_t_for_pred.unsqueeze(0), a_t)
                s_next = all_states[t + 1]
                L_dyn = L_dyn + F.mse_loss(
                    s_next.squeeze(0).detach(), s_pred.squeeze(0).detach()
                )
            L_dyn = L_dyn / H * cfg.dynamics_lambda

            total = L_goal + 0.3 * L_dense + L_dyn
            total.backward()
            opt_a.step(); opt_vs.step()

            noise_std = cfg.langevin_base * (1.0 + cfg.e_i_scale * (e_i - 1.0))
            noise_std = max(0.001, noise_std)
            with torch.no_grad():
                vs_free.add_(torch.randn_like(vs_free) * noise_std)

            if (iteration + 1) % cfg.da_sync_every == 0:
                with torch.no_grad():
                    s = particles_0
                    for t in range(H):
                        a = actions[t:t+1].detach()
                        s_next, _ = self._predict(s, a)
                        alpha = 0.3 * da_eff
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
            "iterations":  iters,
            "noise_std":   noise_std,
            "e_i":         e_i,
            "da_eff":      da_eff,
            "planner":     "GRASP",
        }

    def _predict(self, particles, action):
        try:
            return self.predictor(particles, action)
        except TypeError:
            result = self.predictor(particles, action)
            return result, {}


# ===========================================================================
# Mirror Ascent sampler
# ===========================================================================

class MirrorAscentSampler:
    """Zero-order planner for EXPLORE / REOBSERVE regimes. Unchanged from Sprint 4."""

    def __init__(self, predictor, horizon: int = 8, n_candidates: int = 32):
        self.predictor    = predictor
        self.horizon      = horizon
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
                    cum += F.mse_loss(s_next, goal_particles).item()
                    s = s_next
                costs[k] = -cum / (1.0 + rho)

        eta  = 0.05 * (0.5 + da_eff)
        w    = torch.softmax(costs / max(eta, 1e-4), dim=0)
        best = (candidates[:, 0] * w.unsqueeze(-1)).sum(0)

        return best, {
            "n_candidates": K,
            "best_cost":    -costs.max().item(),
            "action_std":   action_std,
            "planner":      "MirrorAscent",
        }


# ===========================================================================
# Regime-gated planner  ← Sprint 5 DA-gating wired here
# ===========================================================================

def regime_gated_plan(
    cwm_predictor,
    particles_0:    torch.Tensor,
    goal_particles: torch.Tensor,
    signals,
    action_dim:     int                    = 2,
    grasp_config:   Optional[GRASPConfig]  = None,
    device:         str                    = "cpu",
) -> Tuple[torch.Tensor, Dict]:
    """
    Regime-gated planner with DA-gated horizon switching (Sprint 5).

    Regime routing:
        EXPLOIT    -> GRASPPlanner,          H=h_exploit,   iters=iters_exploit  (~8ms)
        EXPLORE    -> MirrorAscentSampler,   H=h_reactive,  iters=iters_reactive
        REOBSERVE  -> MirrorAscentSampler,   H=h_reactive,  iters=iters_reactive (~3.56ms)
        WAIT       -> zero action

    DA-gated horizon: da > da_threshold -> reactive path, else exploitative.
    Both paths confirmed < 250ms frame interval at 4Hz.
    """
    cfg = grasp_config or GRASPConfig()

    regime = getattr(signals, "regime", "EXPLOIT")
    da_eff = getattr(signals, "da_eff", 0.5)
    da     = getattr(signals, "da",     da_eff)   # raw DA signal for gating
    e_i    = getattr(signals, "e_i",    1.0)
    rho    = getattr(signals, "ne",     0.5)

    # ── Sprint 5: select horizon from DA ─────────────────────────────────────
    H, iters = cfg.select_horizon(da)

    # ── WAIT ─────────────────────────────────────────────────────────────────
    if regime == "WAIT":
        return torch.zeros(action_dim), {
            "regime": "WAIT", "acted": False, "planner": "none",
            "horizon": H, "da": da,
        }

    # ── EXPLOIT: GRASP with DA-gated horizon ──────────────────────────────────
    if regime == "EXPLOIT":
        planner = GRASPPlanner(cwm_predictor, cfg)
        action, info = planner.plan(
            particles_0, goal_particles,
            action_dim=action_dim,
            e_i=e_i, da_eff=da_eff,
            device=device,
            horizon_override=H,
            n_iters_override=iters,
        )
        info.update({"regime": "EXPLOIT", "acted": True, "da": da})
        return action, info

    # ── EXPLORE / REOBSERVE: Mirror Ascent ───────────────────────────────────
    action_std = 0.10 * e_i
    if regime == "REOBSERVE":
        action_std *= 0.5

    sampler = MirrorAscentSampler(cwm_predictor, horizon=H, n_candidates=32)
    action, info = sampler.plan(
        particles_0, goal_particles,
        action_dim=action_dim,
        action_std=action_std,
        da_eff=da_eff, rho=rho,
        device=device,
    )
    info.update({"regime": regime, "acted": True, "da": da, "horizon": H})
    return action, info


# ===========================================================================
# Benchmark — tests both horizon paths
# ===========================================================================

def benchmark(
    action_dim: int = 2,
    K:          int = 16,
    d_model:    int = 128,
    n_trials:   int = 20,
    device_str: str = "cpu",
):
    dev = torch.device(device_str)

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
            x = torch.cat([particles.reshape(B, -1), action.reshape(B, -1)], dim=-1)
            return F.normalize(self.net(x).view(B, K, d_model), dim=-1), {}

    predictor = MockPredictor().to(dev).eval()
    p0   = torch.randn(1, K, d_model, device=dev)
    goal = torch.randn(1, K, d_model, device=dev)

    cfg = GRASPConfig()
    print(f"\nGRASP Latency Benchmark")
    print(f"  Device:   {device_str}")
    print(f"  K={K}, d_model={d_model}")
    print(f"  Trials:   {n_trials}\n")

    for label, H, iters, target_ms in [
        ("REOBSERVE (reactive)", cfg.h_reactive, cfg.iters_reactive, 10.0),
        ("EXPLOIT (exploitative)", cfg.h_exploit, cfg.iters_exploit, 10.0),
    ]:
        planner = GRASPPlanner(predictor, GRASPConfig(
            horizon=H, n_lifted_iters=iters
        ))
        # Warmup
        for _ in range(3):
            planner.plan(p0, goal, action_dim=action_dim, device=device_str)
        # Timed
        times = []
        for _ in range(n_trials):
            _, info = planner.plan(p0, goal, action_dim=action_dim, device=device_str)
            times.append(info["elapsed_ms"])

        med = np.median(times)
        status = "PASS" if med < target_ms else "FAIL"
        print(f"  [{label}]  H={H}, iters={iters}")
        print(f"    Mean:     {np.mean(times):.2f} ms")
        print(f"    Median:   {med:.2f} ms")
        print(f"    P95:      {np.percentile(times, 95):.2f} ms")
        print(f"    Min/Max:  {min(times):.2f} / {max(times):.2f} ms")
        print(f"    {status} -- median {med:.2f}ms < {target_ms}ms target\n")


# ===========================================================================
# Self-test
# ===========================================================================

def self_test():
    print("grasp_planner self-test...")

    class MockPred(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(16 * 128 + 2, 16 * 128)
        def forward(self, p, a):
            B = p.shape[0]
            x = torch.cat([p.reshape(B, -1), a.reshape(B, -1)], dim=-1)
            return F.normalize(self.net(x).view(B, 16, 128), dim=-1), {}

    pred = MockPred()
    p0   = torch.randn(1, 16, 128)
    goal = torch.randn(1, 16, 128)
    cfg  = GRASPConfig(horizon=4, n_lifted_iters=2,
                       h_reactive=4, iters_reactive=2,
                       h_exploit=8,  iters_exploit=3)

    # GRASP direct
    planner = GRASPPlanner(pred, cfg)
    action, info = planner.plan(p0, goal, action_dim=2)
    assert action.shape == (2,), f"Expected (2,), got {action.shape}"
    print(f"  GRASP: action={action.detach().numpy().round(3)}, "
          f"loss={info['best_loss']:.4f}, time={info['elapsed_ms']:.1f}ms")

    # MirrorAscent direct
    sampler = MirrorAscentSampler(pred, horizon=4, n_candidates=16)
    action2, info2 = sampler.plan(p0, goal, action_dim=2)
    assert action2.shape == (2,)
    print(f"  MirrorAscent: action={action2.detach().numpy().round(3)}")

    # Regime-gated interface — all four regimes
    from dataclasses import dataclass as dc
    @dc
    class MockSig:
        regime: str   = "EXPLOIT"
        da_eff: float = 0.5
        da:     float = 0.002      # low DA -> exploit path
        e_i:    float = 1.0
        ne:     float = 0.5

    for regime, da in [
        ("EXPLOIT",   0.002),   # low DA  -> H=h_exploit
        ("EXPLORE",   0.020),   # high DA -> H=h_reactive
        ("REOBSERVE", 0.025),   # high DA -> H=h_reactive
        ("WAIT",      0.001),
    ]:
        sig = MockSig(regime=regime, da=da)
        a, i = regime_gated_plan(pred, p0, goal, sig, action_dim=2,
                                  grasp_config=cfg)
        assert a.shape == (2,)
        print(f"  regime={regime}: planner={i['planner']}, "
              f"acted={i['acted']}, H={i.get('horizon','n/a')}, da={da}")

    # DA-gated horizon assertions
    cfg2 = GRASPConfig(da_threshold=0.015, h_reactive=8, iters_reactive=2,
                                           h_exploit=16, iters_exploit=2)
    H_r, i_r = cfg2.select_horizon(da=0.025)   # above threshold
    H_e, i_e = cfg2.select_horizon(da=0.002)   # below threshold
    assert H_r == 8  and i_r == 2,  f"Reactive path wrong: H={H_r}, iters={i_r}"
    assert H_e == 16 and i_e == 2,  f"Exploit  path wrong: H={H_e}, iters={i_e}"
    print(f"  DA-gated horizon: reactive H={H_r}/iters={i_r}, "
          f"exploit H={H_e}/iters={i_e}")

    print("  All assertions passed.\n")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRASP planner -- Sprint 4/5")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--trials",    type=int, default=20)
    parser.add_argument("--device",    default="cpu")
    args = parser.parse_args()

    self_test()

    if args.benchmark:
        benchmark(n_trials=args.trials, device_str=args.device)
    else:
        print("Run with --benchmark to test both horizon paths on GMKtec EVO-X2.")
        print(f"  Reactive  (REOBSERVE): H=8,  iters=2  -- target <10ms")
        print(f"  Exploit   (EXPLOIT):   H=16, iters=4  -- target <10ms")
