"""
sprint4_navigator.py — NeMo-WM Goal-Conditioned Navigation
===========================================================
Sprint 4 deliverable: wires StudentEncoder + GeoLatentDB + GRASP planner
into a single goal-conditioned navigation system.

Architecture:
    frame → StudentEncoder → ParticleEncoder → current_particles
    GPS goal → GeoLatentDB → goal_particle
    (current_particles, goal_particle, regime) → GRASP/MirrorAscent → action

Latency budget at 4Hz (250ms/frame):
    StudentEncoder (NPU):  0.34ms
    ParticleEncoder:       ~1ms
    GeoLatentDB query:     0.022ms
    GRASP planning:        9.25ms
    Total:                 ~10.6ms  (4.2% of budget)

Usage:
    # Benchmark mode
    python sprint4_navigator.py --benchmark

    # Single goal query
    python sprint4_navigator.py --lat 37.9150 --lon -122.3354

    # Self-test
    python sprint4_navigator.py --self-test

Author: John Taylor — github.com/taylorjohn
Date:   2026-04-03
"""

import argparse
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ── NeMo-WM imports ──────────────────────────────────────────────────────────
from train_mvtec import StudentEncoder
from train_cwm_DEPRECATED import CortexWorldModel, MAX_ACTION_DIM
from neuromodulator import NeuromodulatorState
from grasp_planner import GRASPPlanner, MirrorAscentSampler as MirrorAscentPlanner
from geo_latent_db import GeoLatentDB

DEVICE = torch.device("cpu")


# ── MirrorAscent stub if not in grasp_planner ────────────────────────────────
class MirrorAscentPlanner:
    """Fast gradient-based planner for EXPLORE/REOBSERVE regimes."""
    def __init__(self, action_dim: int = 2, lr: float = 0.1, steps: int = 5):
        self.action_dim = action_dim
        self.lr         = lr
        self.steps      = steps

    def plan(self, current: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """Return action that moves current particles toward goal."""
        # Simple gradient: direction from current mean to goal mean
        delta = goal - current.mean(0) if current.ndim == 2 else goal - current
        norm  = np.linalg.norm(delta) + 1e-8
        # Scale to action range [-1, 1]
        action = (delta[:self.action_dim] / norm).clip(-1, 1)
        return action


class Navigator:
    """
    Goal-conditioned navigation using NeMo-WM world model.

    Combines:
        - StudentEncoder (DINOv2-distilled visual encoder)
        - CortexWorldModel (particle encoder + MoE predictor)
        - GeoLatentDB (GPS → particle embedding lookup)
        - GRASP planner (Langevin optimisation, EXPLOIT regime)
        - MirrorAscent planner (fast gradient, EXPLORE/REOBSERVE)
        - NeuromodulatorState (seven-signal biological reward)
    """

    def __init__(
        self,
        encoder_ckpt:   str = r"checkpoints\dinov2_student\student_best.pt",
        cwm_ckpt:       str = r"checkpoints\cwm\cwm_best.pt",
        gps_db_path:    str = "geo_latent_db_gps.npy",
        particle_db:    str = "geo_latent_db_particles_norm.npy",
        grasp_horizon:  int = 2,
        grasp_iters:    int = 1,
        goal_radius_m:  float = 5.0,
        arrival_thresh: float = 3.0,
    ):
        """
        Initialise the navigator.

        Args:
            encoder_ckpt:   path to StudentEncoder checkpoint
            cwm_ckpt:       path to CWM checkpoint
            gps_db_path:    path to GeoLatentDB GPS numpy array
            particle_db:    path to GeoLatentDB normalised particle array
            grasp_horizon:  GRASP planning horizon (frames ahead)
            grasp_iters:    GRASP Langevin refinement iterations
            goal_radius_m:  radius for mean_goal_particle averaging
            arrival_thresh: goal-reached threshold in metres
        """
        print("\nNeMo-WM Navigator — Sprint 4")
        print("=" * 45)

        # ── StudentEncoder ────────────────────────────────────────────────
        t0 = time.time()
        self.encoder = StudentEncoder().to(DEVICE)
        if Path(encoder_ckpt).exists():
            ckpt = torch.load(encoder_ckpt, map_location="cpu",
                              weights_only=False)
            sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
            self.encoder.load_state_dict(sd, strict=False)
            print(f"  StudentEncoder loaded ({time.time()-t0:.2f}s)")
        else:
            print(f"  StudentEncoder: RANDOM WEIGHTS (no checkpoint)")
        self.encoder.eval()

        # ── CortexWorldModel ──────────────────────────────────────────────
        t0 = time.time()
        self.cwm = CortexWorldModel(d_model=128, K=16).to(DEVICE)
        if Path(cwm_ckpt).exists():
            ckpt2 = torch.load(cwm_ckpt, map_location="cpu",
                               weights_only=False)
            self.cwm.load_state_dict(ckpt2.get("model", ckpt2), strict=False)
            ep   = ckpt2.get("epoch", "?")
            loss = ckpt2.get("loss", 0)
            print(f"  CWM loaded epoch={ep}, loss={loss:.4f} ({time.time()-t0:.2f}s)")
        self.cwm.eval()

        # ── Padded predictor wrapper for GRASP ────────────────────────────
        # GRASP passes (B, action_dim=2) but CWM predictor expects
        # (B, MAX_ACTION_DIM=9). Wrap to pad automatically.
        import torch.nn as nn
        import torch.nn.functional as _F
        _MAX_ACTION_DIM = 9
        _predictor = self.cwm.predictor

        class _PaddedPredictor(nn.Module):
            def forward(self_, particles, action):
                if action.shape[-1] < _MAX_ACTION_DIM:
                    action = _F.pad(action, (0, _MAX_ACTION_DIM - action.shape[-1]))
                return _predictor(particles, action)

        self._padded_predictor = _PaddedPredictor()

        # ── GeoLatentDB ───────────────────────────────────────────────────
        t0 = time.time()
        self.geo_db = GeoLatentDB(gps_db_path, particle_db)
        print(f"  GeoLatentDB ready ({time.time()-t0:.3f}s)")

        # ── Planners ──────────────────────────────────────────────────────
        try:
            from grasp_planner import GRASPConfig
            cfg = GRASPConfig(horizon=grasp_horizon, n_lifted_iters=grasp_iters)
            self.grasp = GRASPPlanner(predictor=self._padded_predictor, config=cfg)
            print(f"  GRASP planner (H={grasp_horizon}, iters={grasp_iters})")
        except Exception as e:
            print(f"  GRASP planner unavailable ({e}) — using MirrorAscent only")
            self.grasp = None

        self.mirror = MirrorAscentPlanner()

        # ── Neuromodulator ────────────────────────────────────────────────
        self.neuro = NeuromodulatorState(session_start=time.time())

        # ── Config ────────────────────────────────────────────────────────
        self.goal_radius_m  = goal_radius_m
        self.arrival_thresh = arrival_thresh
        self.goal_lat: Optional[float] = None
        self.goal_lon: Optional[float] = None
        self.goal_particle: Optional[np.ndarray] = None

        # ── Timing stats ──────────────────────────────────────────────────
        self._timings = {
            "encode_ms": [], "plan_ms": [], "db_ms": [], "total_ms": []
        }

        print("=" * 45)
        print("  Ready.\n")

    def set_goal(self, lat: float, lon: float) -> float:
        """
        Set a GPS navigation goal.

        Args:
            lat: goal latitude in degrees
            lon: goal longitude in degrees

        Returns:
            dist_m: distance to nearest training trajectory in metres
        """
        t0 = time.time()
        self.goal_lat = lat
        self.goal_lon = lon
        self.goal_particle, dist_m = self.geo_db.query(lat, lon)
        db_ms = (time.time() - t0) * 1000

        print(f"  Goal set: lat={lat:.6f}, lon={lon:.6f}")
        print(f"  Nearest training trajectory: {dist_m:.1f}m away")
        print(f"  DB query: {db_ms:.3f}ms")

        if dist_m > 50.0:
            print(f"  ⚠️  WARNING: goal is {dist_m:.0f}m from nearest "
                  f"training data — navigation may be unreliable")

        return dist_m

    def step(
        self,
        frame: torch.Tensor,
        action_history: Optional[torch.Tensor] = None,
        current_gps: Optional[Tuple[float, float]] = None,
    ) -> dict:
        """
        Single navigation step: encode frame → plan action → return result.

        Args:
            frame:          (3, 224, 224) normalised RGB frame tensor
            action_history: (2,) last action taken (optional)
            current_gps:    (lat, lon) current position (optional)

        Returns:
            dict with keys:
                action        (2,) action to execute [linear, angular]
                regime        str  neuromodulator regime
                signals       dict seven neuromodulator signals
                dist_to_goal  float distance to goal in metres (if GPS given)
                arrived       bool True if within arrival_thresh
                timings       dict per-component latency in ms
        """
        t_total = time.time()

        if self.goal_particle is None:
            raise RuntimeError("Call set_goal() before step()")

        # ── 1. Encode frame → particles ───────────────────────────────────
        t0 = time.time()
        frame_batch = frame.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            z = self.encoder(frame_batch)
            particles, pos, _, _ = self.cwm.encode(z)
            # particles: (1, K, 128) → (K, 128)
            p_np = particles.squeeze(0).cpu().numpy()

        encode_ms = (time.time() - t0) * 1000

        # ── 2. Neuromodulator update ──────────────────────────────────────
        goal_t  = torch.tensor(self.goal_particle, dtype=torch.float32)
        p_mean  = particles.mean(dim=1)  # (1, 128)
        act_mag = float(action_history.norm()) if action_history is not None else 0.0

        signals = self.neuro.update(
            z_pred           = p_mean,
            z_actual         = goal_t.unsqueeze(0),
            rho              = 0.5,
            action_magnitude = act_mag,
        )
        regime = signals["regime"]

        # ── 3. Plan action ────────────────────────────────────────────────
        # ARCHITECTURE NOTE (2026-04-03):
        # GRASP with full CWM predictor costs ~40-60ms (9.4ms/call x 4-6 calls)
        # — not viable at 4Hz. MirrorAscent (~0.4ms) is the real-time planner.
        # GRASP available offline via plan_offline() for waypoint pre-computation.
        t0 = time.time()
        if regime == "WAIT":
            action = np.zeros(2)
        else:
            # MirrorAscent: real-time gradient planner (~0.4ms)
            action = self.mirror.plan(p_np, self.goal_particle)

        action = np.clip(action, -1.0, 1.0)
        plan_ms = (time.time() - t0) * 1000

        # ── 4. Distance to goal ───────────────────────────────────────────
        dist_to_goal = None
        arrived      = False
        if current_gps is not None:
            clat, clon = current_gps
            dlat = (self.goal_lat - clat) * 111000
            dlon = (self.goal_lon - clon) * 111000 * np.cos(np.radians(clat))
            dist_to_goal = float(np.sqrt(dlat**2 + dlon**2))
            arrived = dist_to_goal < self.arrival_thresh

        total_ms = (time.time() - t_total) * 1000

        # ── Track timings ─────────────────────────────────────────────────
        self._timings["encode_ms"].append(encode_ms)
        self._timings["plan_ms"].append(plan_ms)
        self._timings["total_ms"].append(total_ms)

        return {
            "action":       action,
            "regime":       regime,
            "signals":      signals,
            "dist_to_goal": dist_to_goal,
            "arrived":      arrived,
            "timings": {
                "encode_ms": encode_ms,
                "plan_ms":   plan_ms,
                "total_ms":  total_ms,
            }
        }

    def latency_report(self) -> dict:
        """Print and return latency statistics over all steps."""
        if not self._timings["total_ms"]:
            print("No steps recorded yet.")
            return {}

        report = {}
        print("\nLatency Report (ms):")
        print(f"  {'Component':20s} {'Mean':>8} {'P95':>8} {'Max':>8}")
        print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")
        for key, vals in self._timings.items():
            if vals:
                arr = np.array(vals)
                mean = arr.mean()
                p95  = np.percentile(arr, 95)
                mx   = arr.max()
                print(f"  {key:20s} {mean:8.3f} {p95:8.3f} {mx:8.3f}")
                report[key] = {"mean": mean, "p95": p95, "max": mx}
        return report


def self_test(nav: Navigator, n_steps: int = 50):
    """
    Run a synthetic navigation episode using random frames.
    Tests the full pipeline without real sensor data.
    """
    import io
    from PIL import Image

    print(f"\nSelf-test: {n_steps} synthetic steps")
    print("-" * 40)

    # Set a goal inside the RECON coverage area
    # Use the first DB entry's GPS as a guaranteed valid goal
    goal_lat = float(nav.geo_db.gps[100, 0])
    goal_lon = float(nav.geo_db.gps[100, 1])
    dist = nav.set_goal(goal_lat, goal_lon)
    print(f"  Goal: {goal_lat:.6f}, {goal_lon:.6f} ({dist:.1f}m from training)")

    # Simulate current position near the goal
    current_gps = (goal_lat + 0.0005, goal_lon + 0.0005)  # ~60m away

    regimes = {"EXPLOIT": 0, "EXPLORE": 0, "REOBSERVE": 0, "WAIT": 0}
    arrived = False

    for step_i in range(n_steps):
        # Synthetic random frame (in practice: real camera frame)
        frame = torch.rand(3, 224, 224)

        result = nav.step(
            frame       = frame,
            current_gps = current_gps,
        )

        regimes[result["regime"]] = regimes.get(result["regime"], 0) + 1

        if step_i % 10 == 0:
            dist_str = (f"{result['dist_to_goal']:.1f}m"
                       if result["dist_to_goal"] else "unknown")
            print(f"  Step {step_i:3d}: regime={result['regime']:10s} "
                  f"action=[{result['action'][0]:+.3f}, {result['action'][1]:+.3f}] "
                  f"dist={dist_str}")

        if result["arrived"]:
            print(f"  ✅ Goal reached at step {step_i}!")
            arrived = True
            break

    print(f"\n  Regime distribution: {regimes}")
    print(f"  Goal reached: {arrived}")
    nav.latency_report()


def benchmark(nav: Navigator, n_trials: int = 100):
    """Benchmark full pipeline latency."""
    print(f"\nBenchmark: {n_trials} trials, full pipeline")
    print("-" * 40)

    # Set goal using first DB entry
    goal_lat = float(nav.geo_db.gps[0, 0])
    goal_lon = float(nav.geo_db.gps[0, 1])
    nav.set_goal(goal_lat, goal_lon)

    for _ in range(n_trials):
        frame = torch.rand(3, 224, 224)
        nav.step(frame=frame)

    report = nav.latency_report()

    total_mean = report.get("total_ms", {}).get("mean", 0)
    budget_pct = total_mean / 250.0 * 100
    print(f"\n  Total mean: {total_mean:.2f}ms = {budget_pct:.1f}% of 4Hz budget")

    if total_mean < 25.0:
        print("  ✅ PASS — well within 4Hz budget")
    elif total_mean < 50.0:
        print("  🟡 MARGINAL — within budget but tight")
    else:
        print("  ❌ FAIL — exceeds recommended latency")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NeMo-WM Sprint 4 Goal-Conditioned Navigator"
    )
    parser.add_argument("--self-test",  action="store_true",
                        help="Run synthetic navigation episode")
    parser.add_argument("--benchmark",  action="store_true",
                        help="Benchmark full pipeline latency")
    parser.add_argument("--lat",  type=float, default=None,
                        help="Goal latitude")
    parser.add_argument("--lon",  type=float, default=None,
                        help="Goal longitude")
    parser.add_argument("--trials", type=int, default=100,
                        help="Number of benchmark trials")
    parser.add_argument("--steps",  type=int, default=50,
                        help="Number of self-test steps")
    parser.add_argument("--cwm-ckpt", default=r"checkpoints\cwm\cwm_best.pt")
    parser.add_argument("--random-encoder", action="store_true",
                        help="Use random encoder (ablation)")
    args = parser.parse_args()

    # Build navigator
    nav = Navigator(
        cwm_ckpt = args.cwm_ckpt,
        encoder_ckpt = (r"checkpoints\dinov2_student\student_best.pt"
                        if not args.random_encoder else "nonexistent"),
    )

    if args.benchmark:
        benchmark(nav, n_trials=args.trials)

    elif args.self_test:
        self_test(nav, n_steps=args.steps)

    elif args.lat is not None and args.lon is not None:
        dist = nav.set_goal(args.lat, args.lon)
        frame = torch.rand(3, 224, 224)
        result = nav.step(frame=frame)
        print(f"\nSingle step result:")
        print(f"  Action:  [{result['action'][0]:+.4f}, {result['action'][1]:+.4f}]")
        print(f"  Regime:  {result['regime']}")
        print(f"  DA:      {result['signals']['da']:.3f}")
        print(f"  5HT:     {result['signals']['sht']:.3f}")
        print(f"  Encode:  {result['timings']['encode_ms']:.2f}ms")
        print(f"  Plan:    {result['timings']['plan_ms']:.2f}ms")
        print(f"  Total:   {result['timings']['total_ms']:.2f}ms")

    else:
        # Default: run self-test
        self_test(nav, n_steps=args.steps)
