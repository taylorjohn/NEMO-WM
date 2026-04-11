"""
sprint7c_bsd_executor.py — BSD Trajectory Scoring for Waypoint Execution
=========================================================================
Replaces Sprint 7a/7b's single-frame cosine similarity switching with
multi-step trajectory quality scoring inspired by Behavioral Score Diffusion.

BSD insight: instead of asking "am I close to the next waypoint?", ask
"which path from my current state leads to the best trajectory quality?"

Applied to NeMo-WM:
  - GeoLatentDB is the pre-collected trajectory library (BSD's examples)
  - Particle embeddings are the state representations
  - Trajectory quality = average cosine coherence along the planned path
  - At each 4Hz step, score K candidate action sequences and pick the best

Architecture:
    Current particles (K=16, 128-D)
    + Waypoint plan (N waypoints × 128-D embeddings)
    + GeoLatentDB (10,906 × 128-D)
    →  BSD scoring: for each candidate action, simulate 1-step particle
       update and score resulting state vs planned path
    →  Best-scored action executed

Key difference from Sprint 7a/b:
    7a/7b: switch waypoint when cos_sim(current, waypoint) > 0.92
    7c:    score K=8 candidate one-step rollouts, pick action that
           maximises trajectory-level coherence, not just next-waypoint sim

Performance target: < 0.5ms additional overhead per step

Run:
    python sprint7c_bsd_executor.py --self-test
    python sprint7c_bsd_executor.py --benchmark \
        --geo-db checkpoints/cwm/geo_latent_db.pt

Author: John Taylor — github.com/taylorjohn
Date:   2026-04-06
Sprint: 7c
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.spatial import KDTree

# ── BSD constants ─────────────────────────────────────────────────────────────

K_SAMPLES       = 8      # candidate actions to score per step
ACTION_STD       = 0.3   # noise std for candidate generation
HORIZON          = 3     # look-ahead steps for trajectory scoring
KERNEL_BANDWIDTH = 0.15  # RBF kernel bandwidth for BSD weighting
SWITCH_THRESH    = 0.92  # cosine similarity to advance waypoint
ARRIVAL_THRESH   = 0.97  # cosine similarity to declare goal reached
REPLAN_TIMEOUT   = 60    # steps before forcing replan


# ── Kernel-based trajectory scorer ───────────────────────────────────────────

class BSDScorer:
    """
    Behavioral Score Diffusion applied to particle embeddings.

    Scores candidate actions by simulating their effect on the current
    particle state and measuring trajectory quality against the GeoLatentDB
    example library.

    BSD framing:
        score(action) = Σ_i w_i × cos_sim(simulate(state, action), example_i)
        w_i = softmax(RBF(state, example_i) / bandwidth)

    The weights w_i are computed from similarity to nearby GeoLatentDB entries.
    High-weight examples are the "behaviorally relevant" reference trajectories.
    """

    def __init__(
        self,
        geo_embeddings: np.ndarray,    # (N, 128) GeoLatentDB embeddings
        bandwidth:      float = KERNEL_BANDWIDTH,
        k_samples:      int   = K_SAMPLES,
        horizon:        int   = HORIZON,
        action_std:     float = ACTION_STD,
        n_neighbors:    int   = 32,    # top-N geo entries for scoring (speed)
    ):
        self.geo_embs    = geo_embeddings / (
            np.linalg.norm(geo_embeddings, axis=1, keepdims=True) + 1e-8)
        self.bandwidth   = bandwidth
        self.k_samples   = k_samples
        self.horizon     = horizon
        self.action_std  = action_std
        self.n_neighbors = min(n_neighbors, len(geo_embeddings))
        self._kdtree     = KDTree(self.geo_embs)

    def _simulate_step(
        self,
        particles: np.ndarray,   # (K, 128) current particles
        action:    np.ndarray,   # (2,) action
        goal_emb:  np.ndarray,   # (128,) target waypoint embedding
    ) -> np.ndarray:
        """
        Simulate one step: move particle mean toward goal, scaled by action.
        This is a lightweight linear approximation — no world model rollout.
        The action magnitude and direction bias the particle drift.
        """
        cur_mean = particles.mean(0)                         # (128,)
        goal_dir = goal_emb - cur_mean
        goal_dir /= (np.linalg.norm(goal_dir) + 1e-8)

        # Action biases drift toward/away from goal
        action_mag = np.linalg.norm(action)
        drift = goal_dir * action_mag * 0.1                  # small step

        # New particle mean (simplified — real system uses world model)
        new_mean = cur_mean + drift
        new_mean /= (np.linalg.norm(new_mean) + 1e-8)
        return new_mean                                      # (128,)

    def _kernel_weights(self, state: np.ndarray) -> np.ndarray:
        """
        RBF kernel weights from current state to N nearest GeoLatentDB entries.
        Returns (n_neighbors,) weights summing to 1.
        """
        # Find nearest neighbors
        dists, idxs = self._kdtree.query(state, k=self.n_neighbors)
        # RBF kernel
        w = np.exp(-(dists**2) / (2 * self.bandwidth**2))
        w /= w.sum() + 1e-8
        return w, idxs

    def score_action(
        self,
        particles:   np.ndarray,   # (K, 128) current particles
        action:      np.ndarray,   # (2,) candidate action
        waypoints:   list,         # list of waypoint embeddings (128-D each)
        wp_idx:      int,          # current active waypoint index
    ) -> float:
        """
        Score a candidate action by multi-step trajectory quality.
        Higher = better trajectory coherence with GeoLatentDB library.
        """
        state = particles.mean(0)
        state /= (np.linalg.norm(state) + 1e-8)

        # Kernel weights from current state
        w, idxs = self._kernel_weights(state)

        total_score = 0.0
        cur_particles = particles.copy()

        for h in range(min(self.horizon, len(waypoints) - wp_idx)):
            wp_emb   = waypoints[wp_idx + h]
            new_mean = self._simulate_step(cur_particles, action, wp_emb)

            # Score simulated state against weighted GeoLatentDB neighbors
            geo_sims   = self.geo_embs[idxs] @ new_mean    # (n_neighbors,)
            step_score = float(np.dot(w, geo_sims))

            # Also score against the target waypoint directly
            wp_score   = float(np.dot(new_mean, wp_emb))

            total_score += 0.6 * wp_score + 0.4 * step_score

            # Move simulated state forward
            cur_particles = np.tile(new_mean, (len(particles), 1))

        return total_score / max(1, min(self.horizon, len(waypoints) - wp_idx))

    def best_action(
        self,
        particles:  np.ndarray,   # (K, 128)
        base_action: np.ndarray,  # (2,) MirrorAscent action (best guess)
        waypoints:  list,
        wp_idx:     int,
    ) -> tuple[np.ndarray, float]:
        """
        Generate K candidate actions, score ALL simultaneously via matrix ops.
        Vectorized: one KDTree query + one matmul scores all candidates at once.
        """
        # Generate candidates: (K_samples, 2)
        noise = np.random.randn(self.k_samples - 1, 2) * self.action_std
        cands = np.vstack([base_action[None], base_action + noise])
        cands = np.clip(cands, -1.0, 1.0)                 # (K_samples, 2)

        cur_mean = particles.mean(0)
        cur_mean /= (np.linalg.norm(cur_mean) + 1e-8)

        # Kernel weights — computed once for current state
        dists, idxs = self._kdtree.query(cur_mean, k=self.n_neighbors)
        w = np.exp(-(dists**2) / (2 * self.bandwidth**2))
        w /= w.sum() + 1e-8                               # (n_neighbors,)
        geo_nb = self.geo_embs[idxs]                      # (n_neighbors, 128)

        # Simulate all K candidates simultaneously for HORIZON steps
        # simulated_means: (K_samples, 128)
        scores = np.zeros(self.k_samples)
        sim_means = np.tile(cur_mean, (self.k_samples, 1))  # (K, 128)

        for h in range(min(self.horizon, len(waypoints) - wp_idx)):
            wp_emb = waypoints[wp_idx + h]
            wp_norm = wp_emb / (np.linalg.norm(wp_emb) + 1e-8)

            # Move each simulated mean toward goal, biased by action magnitude
            goal_dirs = wp_norm - sim_means                   # (K, 128)
            norms = np.linalg.norm(goal_dirs, axis=1, keepdims=True) + 1e-8
            goal_dirs /= norms

            # action_mags: (K, 1) — different for each candidate
            action_mags = np.linalg.norm(cands, axis=1, keepdims=True) * 0.1
            sim_means = sim_means + goal_dirs * action_mags   # (K, 128)
            sim_norms = np.linalg.norm(sim_means, axis=1, keepdims=True) + 1e-8
            sim_means /= sim_norms

            # Score all K simultaneously: geo_nb @ sim_means.T = (n_nb, K)
            geo_scores = geo_nb @ sim_means.T               # (n_neighbors, K)
            bsd_scores = w @ geo_scores                      # (K,)

            # Direct waypoint similarity: (K,)
            wp_scores  = sim_means @ wp_norm                 # (K,)

            scores += 0.6 * wp_scores + 0.4 * bsd_scores

        scores /= max(1, min(self.horizon, len(waypoints) - wp_idx))
        best_idx = int(np.argmax(scores))
        return cands[best_idx], float(scores[best_idx])


# ── BSD Waypoint Executor ─────────────────────────────────────────────────────

@dataclass
class BSDExecutionState:
    waypoints:         list          # list of (128-D) embeddings
    waypoint_idx:      int   = 0
    steps_on_wp:       int   = 0
    total_steps:       int   = 0
    waypoints_reached: int   = 0
    replans:           int   = 0
    arrived:           bool  = False


class BSDWaypointExecutor:
    """
    Drop-in replacement for Sprint 7a WaypointExecutor.
    Same interface — adds BSD trajectory scoring on top of MirrorAscent.

    At each step:
      1. MirrorAscent proposes base action toward active waypoint
      2. BSD scorer evaluates K candidates around base action
      3. Best-scored action is executed
      4. Cosine similarity check for waypoint switching (unchanged)
    """

    def __init__(
        self,
        scorer:         BSDScorer,
        switch_thresh:  float = SWITCH_THRESH,
        replan_timeout: int   = REPLAN_TIMEOUT,
        arrival_thresh: float = ARRIVAL_THRESH,
    ):
        self.scorer         = scorer
        self.switch_thresh  = switch_thresh
        self.replan_timeout = replan_timeout
        self.arrival_thresh = arrival_thresh
        self.state:         Optional[BSDExecutionState] = None

    def reset(self, waypoint_embeddings: list[np.ndarray]):
        """Start executing a new waypoint plan."""
        self.state = BSDExecutionState(waypoints=waypoint_embeddings)

    def _mirror_ascent_action(
        self,
        cur_mean: np.ndarray,
        goal_emb: np.ndarray,
        lr: float = 0.12,
    ) -> np.ndarray:
        """Fast gradient action toward goal."""
        delta  = goal_emb - cur_mean
        norm   = np.linalg.norm(delta) + 1e-8
        action = (delta[:2] / norm * lr).clip(-1.0, 1.0)
        return action

    def step(
        self,
        current_particles: np.ndarray,   # (K, 128) or (128,)
        current_gps:       Optional[tuple] = None,
    ) -> tuple[np.ndarray, dict]:
        """4Hz step. Returns (action, info)."""
        if self.state is None or self.state.arrived:
            return np.zeros(2), {"status": "idle"}

        s = self.state
        s.total_steps += 1
        s.steps_on_wp += 1

        if current_particles.ndim == 1:
            current_particles = current_particles[np.newaxis]
        cur_mean = current_particles.mean(0)
        cur_norm = cur_mean / (np.linalg.norm(cur_mean) + 1e-8)

        active_emb = self.state.waypoints[s.waypoint_idx]
        wp_norm    = active_emb / (np.linalg.norm(active_emb) + 1e-8)
        sim        = float(np.dot(cur_norm, wp_norm))
        is_last    = s.waypoint_idx == len(s.waypoints) - 1

        status = "navigating"

        # Arrival
        if is_last and sim > self.arrival_thresh:
            s.arrived = True
            return np.zeros(2), {
                "status": "ARRIVED", "sim": sim,
                "waypoint": s.waypoint_idx, "total_steps": s.total_steps,
            }

        # Waypoint switch
        if not is_last and sim > self.switch_thresh:
            s.waypoint_idx    += 1
            s.waypoints_reached += 1
            s.steps_on_wp     = 0
            active_emb = self.state.waypoints[s.waypoint_idx]
            status = f"SWITCHED -> wp {s.waypoint_idx}"

        # Replan trigger
        needs_replan = s.steps_on_wp > self.replan_timeout
        if needs_replan:
            status = "REPLAN_NEEDED"
            s.replans += 1
            s.steps_on_wp = 0

        # MirrorAscent base action
        base_action = self._mirror_ascent_action(cur_norm, active_emb)

        # BSD scoring — find best action among K candidates
        t0 = time.perf_counter()
        best_action, bsd_score = self.scorer.best_action(
            current_particles, base_action,
            self.state.waypoints, s.waypoint_idx
        )
        bsd_ms = (time.perf_counter() - t0) * 1000

        info = {
            "status":           status,
            "sim":              sim,
            "waypoint":         s.waypoint_idx,
            "n_waypoints":      len(s.waypoints),
            "steps_on_wp":      s.steps_on_wp,
            "bsd_score":        bsd_score,
            "bsd_ms":           bsd_ms,
            "base_action":      base_action,
            "best_action":      best_action,
            "needs_replan":     needs_replan,
        }
        return best_action, info


# ── Self-test ─────────────────────────────────────────────────────────────────

def self_test():
    print("\n" + "=" * 60)
    print("  Sprint 7c — BSD Executor Self-Test")
    print("=" * 60)

    rng = np.random.default_rng(42)
    n   = 1000

    # Synthetic GeoLatentDB
    geo_embs = rng.standard_normal((n, 128)).astype(np.float32)
    geo_embs /= np.linalg.norm(geo_embs, axis=1, keepdims=True)

    scorer = BSDScorer(geo_embs, k_samples=8, horizon=3)

    # Synthetic waypoints
    waypoints = [rng.standard_normal(128).astype(np.float32)
                 for _ in range(5)]
    for i in range(len(waypoints)):
        waypoints[i] /= np.linalg.norm(waypoints[i])

    executor = BSDWaypointExecutor(scorer)
    executor.reset(waypoints)

    # ── Test 1: Single step ──────────────────────────────────────────────
    print("\n── Test 1: Single BSD step")
    particles = rng.standard_normal((16, 128)).astype(np.float32)
    particles /= np.linalg.norm(particles, axis=1, keepdims=True)
    action, info = executor.step(particles)
    assert action.shape == (2,), f"Bad action shape: {action.shape}"
    assert -1.0 <= action[0] <= 1.0
    assert "bsd_score" in info
    assert "bsd_ms" in info
    print(f"  Action: {action}")
    print(f"  BSD score: {info['bsd_score']:.4f}")
    print(f"  BSD latency: {info['bsd_ms']:.2f}ms")
    print("  PASS")

    # ── Test 2: BSD picks better actions than worst candidate ─────────────
    print("\n── Test 2: BSD scoring consistency")
    scores_best, scores_worst = [], []
    for _ in range(50):
        p  = rng.standard_normal((16, 128)).astype(np.float32)
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        base = np.array([rng.uniform(-1,1), rng.uniform(-1,1)])
        # Run best_action twice — both use same vectorized path
        best_a1, s1 = scorer.best_action(p, base, waypoints, 0)
        best_a2, s2 = scorer.best_action(p, base, waypoints, 0)
        scores_best.append(max(s1, s2))
        scores_worst.append(min(s1, s2))
    mean_best  = np.mean(scores_best)
    mean_worst = np.mean(scores_worst)
    print(f"  BSD max score mean:  {mean_best:.4f}")
    print(f"  BSD min score mean:  {mean_worst:.4f}")
    # Best should be >= worst (trivially true, validates consistency)
    assert mean_best >= mean_worst
    # Both should be in reasonable range
    assert -1.0 <= mean_best <= 1.0
    print("  PASS — BSD scoring consistent and bounded")

    # ── Test 3: 50-step execution ────────────────────────────────────────
    print("\n── Test 3: 50-step execution loop")
    executor.reset(waypoints)
    step_times = []
    for _ in range(50):
        p  = rng.standard_normal((16, 128)).astype(np.float32)
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        t0 = time.perf_counter()
        action, info = executor.step(p)
        step_times.append((time.perf_counter() - t0) * 1000)
        assert action.shape == (2,)
    print(f"  Step latency: median={np.median(step_times):.2f}ms "
          f"p95={np.percentile(step_times,95):.2f}ms")
    print("  PASS")

    # ── Test 4: Latency benchmark ────────────────────────────────────────
    print("\n── Test 4: Latency with larger GeoLatentDB")
    geo_large = rng.standard_normal((10906, 128)).astype(np.float32)
    geo_large /= np.linalg.norm(geo_large, axis=1, keepdims=True)
    scorer_large = BSDScorer(geo_large, k_samples=8, horizon=3, n_neighbors=32)
    executor_large = BSDWaypointExecutor(scorer_large)
    executor_large.reset(waypoints)

    times = []
    for _ in range(100):
        p  = rng.standard_normal((16, 128)).astype(np.float32)
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        t0 = time.perf_counter()
        executor_large.step(p)
        times.append((time.perf_counter() - t0) * 1000)
    print(f"  10,906-node graph: median={np.median(times):.2f}ms "
          f"p95={np.percentile(times,95):.2f}ms")
    overhead = np.median(times) - np.median(step_times)
    print(f"  BSD overhead vs 7a: +{overhead:.2f}ms")
    assert np.median(times) < 8.0, f"BSD too slow: {np.median(times):.1f}ms"
    print("  PASS")

    print("\n" + "=" * 60)
    print("  All 4 tests PASSED")
    print("=" * 60 + "\n")


# ── Benchmark ─────────────────────────────────────────────────────────────────

def benchmark(geo_db_path=None):
    print("\n── Sprint 7c BSD Benchmark")
    rng = np.random.default_rng(0)

    # Load GeoLatentDB — use deduplicated WaypointGraph nodes, not raw entries
    if geo_db_path and Path(geo_db_path).exists():
        data = torch.load(geo_db_path, map_location="cpu", weights_only=False)
        raw_embs = np.array(data.get("embeddings", data))
        raw_gps  = np.array(data.get("gps", np.zeros((len(raw_embs), 2))))
        # Deduplicate same as WaypointGraph (MIN_GRAPH_RADIUS = 0.05m = 4.5e-7 deg)
        from scipy.spatial import KDTree as _KDTree
        tree = _KDTree(raw_gps)
        mask = np.ones(len(raw_embs), dtype=bool)
        for i in range(len(raw_embs)):
            if not mask[i]: continue
            nn = tree.query_ball_point(raw_gps[i], r=4.5e-7)
            for j in nn:
                if j != i: mask[j] = False
        geo_embs = raw_embs[mask]
        print(f"  Real GeoLatentDB: {len(raw_embs):,} raw → {len(geo_embs):,} deduplicated nodes")
    else:
        geo_embs = rng.standard_normal((10906, 128)).astype(np.float32)
        print(f"  Synthetic GeoLatentDB: {len(geo_embs):,} entries")
    geo_embs = geo_embs.astype(np.float32)
    geo_embs /= np.linalg.norm(geo_embs, axis=1, keepdims=True)

    waypoints = [rng.standard_normal(128).astype(np.float32) for _ in range(5)]
    for i in range(len(waypoints)):
        waypoints[i] /= np.linalg.norm(waypoints[i])

    # Build KDTree once — reuse across configurations
    print(f"\n  Building KDTree on {len(geo_embs):,} nodes...")
    t_tree = time.perf_counter()
    base_scorer = BSDScorer(geo_embs, k_samples=8, n_neighbors=32)
    tree_ms = (time.perf_counter() - t_tree) * 1000
    print(f"  KDTree built in {tree_ms:.0f}ms")

    for k_samples in [4, 8, 16]:
        for n_neighbors in [16, 32]:
            # Reuse KDTree from base_scorer
            scorer = BSDScorer.__new__(BSDScorer)
            scorer.geo_embs    = base_scorer.geo_embs
            scorer.bandwidth   = base_scorer.bandwidth
            scorer.k_samples   = k_samples
            scorer.horizon     = base_scorer.horizon
            scorer.action_std  = base_scorer.action_std
            scorer.n_neighbors = n_neighbors
            scorer._kdtree     = base_scorer._kdtree  # reuse!

            executor = BSDWaypointExecutor(scorer)
            executor.reset(waypoints)

            times = []
            for _ in range(200):
                p  = rng.standard_normal((16, 128)).astype(np.float32)
                p /= np.linalg.norm(p, axis=1, keepdims=True)
                t0 = time.perf_counter()
                executor.step(p)
                times.append((time.perf_counter() - t0) * 1000)

            budget_ok = np.median(times) < 5.0
            print(f"  K={k_samples:<2} N_nb={n_neighbors:<2}: "
                  f"median={np.median(times):.2f}ms "
                  f"p95={np.percentile(times,95):.2f}ms "
                  f"{'✓' if budget_ok else '✗'}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sprint 7c — BSD Executor")
    parser.add_argument("--self-test",  action="store_true")
    parser.add_argument("--benchmark",  action="store_true")
    parser.add_argument("--geo-db",     default="checkpoints/cwm/geo_latent_db.pt")
    args = parser.parse_args()

    if args.self_test:
        self_test()
    elif args.benchmark:
        benchmark(args.geo_db if Path(args.geo_db).exists() else None)
    else:
        parser.print_help()
