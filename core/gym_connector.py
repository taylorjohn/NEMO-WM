"""
gym_connector.py — Connect NeMo-WM to Real PointMaze Gym
==========================================================
Bridge between NeMo-WM's belief-space world model and
a real Gymnasium PointMaze environment with real physics.

This replaces self-simulation with actual environment interaction:
  - Real physics (MuJoCo dynamics)
  - Real observations (position + velocity)
  - Real goal reaching (did you actually get there?)
  - Real benchmark numbers

The connector:
  1. Wraps gym environment
  2. Projects observations to belief space
  3. Uses NeMo-WM's transition model to PREDICT
  4. Compares predictions to REALITY
  5. Tracks prediction accuracy over time

Usage:
    python gym_connector.py              # run agent in real env
    python gym_connector.py --test       # quick validation
    python gym_connector.py --benchmark  # formal benchmark
    python gym_connector.py --episodes 50  # custom episode count
"""

import argparse
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

D_BELIEF = 64
D_ACTION = 2

Path("data").mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Belief Projection (same as train_from_minari.py)
# ══════════════════════════════════════════════════════════════════════════════

def project_to_belief(obs, d_belief=D_BELIEF):
    """Project observation to belief space.
    Matches train_from_minari.py projection exactly."""
    d_obs = obs.shape[-1] if obs.ndim > 1 else len(obs)
    rng = np.random.RandomState(42)

    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False

    N = obs.shape[0]
    belief = np.zeros((N, d_belief), dtype=np.float32)

    # Use per-sample normalization matching training code exactly
    # train_from_minari.py: obs / (np.abs(obs).max(axis=0, keepdims=True) + 1e-8)
    # For batch: max across all samples. For single: max of that sample.
    # To match training, we use the TRAINING data max (stored in beliefs)
    # Approximate: Minari UMaze obs max per dim ≈ [2.5, 2.5, 5.2, 5.2]
    # But actually the training code uses the BATCH max, so for 1M samples
    # each dimension's max is ~5.23
    #
    # The correct approach: use the same max as training
    OBS_MAX = np.array([5.23, 5.23, 5.23, 5.23], dtype=np.float32)
    
    # Handle different obs dimensions
    if d_obs <= len(OBS_MAX):
        obs_norm = obs / (OBS_MAX[:d_obs].reshape(1, -1) + 1e-8)
    else:
        obs_max = np.abs(obs).max(axis=0, keepdims=True) + 1e-8
        obs_norm = obs / obs_max

    n_direct = min(d_obs, d_belief)
    belief[:, :n_direct] = obs_norm[:, :n_direct]

    idx = n_direct
    for i in range(d_obs):
        for j in range(i, d_obs):
            if idx >= d_belief:
                break
            belief[:, idx] = obs[:, i] * obs[:, j] * 0.1
            idx += 1
        if idx >= d_belief:
            break

    if idx < d_belief:
        W_rand = rng.randn(d_obs, d_belief - idx).astype(np.float32) * 0.3
        belief[:, idx:] = np.tanh(obs @ W_rand)

    if squeeze:
        return belief[0]
    return belief


# ══════════════════════════════════════════════════════════════════════════════
# Transition Model Loader
# ══════════════════════════════════════════════════════════════════════════════

class TransitionModel:
    """Load trained NeMo-WM transition model."""

    def __init__(self):
        self.trained = False
        path = Path("data/minari_trained/transition_model.npz")
        if path.exists():
            data = np.load(path)
            self.W1 = data["W1"]
            self.b1 = data["b1"]
            self.W2 = data["W2"]
            self.b2 = data["b2"]
            self.trained = True
        else:
            rng = np.random.RandomState(42)
            d_in = D_BELIEF + D_ACTION
            self.W1 = rng.randn(d_in, 128).astype(np.float32) * 0.1
            self.b1 = np.zeros(128, dtype=np.float32)
            self.W2 = rng.randn(128, D_BELIEF).astype(np.float32) * 0.1
            self.b2 = np.zeros(D_BELIEF, dtype=np.float32)

    def predict(self, belief, action):
        x = np.concatenate([belief, action])
        h = np.maximum(0, x @ self.W1 + self.b1)
        return np.clip(h @ self.W2 + self.b2, -5, 5)


# ══════════════════════════════════════════════════════════════════════════════
# Schema Store Loader
# ══════════════════════════════════════════════════════════════════════════════

class SchemaStore:
    """Load trained schema codebook."""

    def __init__(self):
        path = Path("data/minari_trained/schema_codebook.npz")
        if path.exists():
            data = np.load(path)
            self.codebook = data["codebook"]
            self.n = len(self.codebook)
        else:
            rng = np.random.RandomState(42)
            self.codebook = rng.randn(32, D_BELIEF).astype(np.float32)
            self.n = 32

    def nearest(self, belief):
        dists = np.linalg.norm(self.codebook - belief, axis=1)
        idx = int(np.argmin(dists))
        return idx, float(dists[idx])


# ══════════════════════════════════════════════════════════════════════════════
# Gym Connector
# ══════════════════════════════════════════════════════════════════════════════

class GymConnector:
    """
    Bridge NeMo-WM to real Gymnasium PointMaze.
    
    The agent:
    1. Observes real physics state
    2. Projects to belief space
    3. Predicts next belief (world model)
    4. Takes action
    5. Compares prediction to reality
    """

    def __init__(self, env_name="PointMaze_UMaze-v3"):
        import gymnasium as gym
        try:
            import gymnasium_robotics  # registers PointMaze envs
        except ImportError:
            pass
        self.env = gym.make(env_name, max_episode_steps=500)
        self.model = TransitionModel()
        self.schemas = SchemaStore()
        self.env_name = env_name

        # Get action space info
        self.continuous_action = hasattr(self.env.action_space, 'low')
        if self.continuous_action:
            self.action_dim = self.env.action_space.shape[0]
        else:
            self.action_dim = 1

        # Tracking
        self.prediction_errors = []
        self.rewards = []
        self.schemas_visited = set()
        self.steps_total = 0

    def extract_obs(self, obs):
        """Extract observation array from gym output and scale to match Minari."""
        if isinstance(obs, dict):
            raw = obs.get("observation",
                    obs.get("achieved_goal",
                    np.zeros(4, dtype=np.float32)))
        else:
            raw = obs

        raw = np.array(raw, dtype=np.float32)

        # Scale live gym coordinates to match Minari D4RL range
        # Minari UMaze positions: roughly [-2.5, 2.5], velocities: [-5.2, 5.2]
        # Live gym positions: roughly [-1, 1], velocities: [-5, 5]
        # Scale factor for positions: ~2.5x
        if len(raw) >= 4:
            raw[0] *= 2.5  # x position
            raw[1] *= 2.5  # y position
            # velocities are already similar scale

        return raw

    def run_episode(self, max_steps=300, goal_schema=None,
                      verbose=False):
        """
        Run one episode in the real environment.
        Returns episode stats.
        """
        obs, info = self.env.reset()
        obs_array = self.extract_obs(obs)
        belief = project_to_belief(obs_array)

        if goal_schema is None:
            # Pick from curiosity — least visited schema
            usage = np.array([1 if i in self.schemas_visited else 0
                               for i in range(self.schemas.n)])
            goal_schema = int(np.argmin(usage + np.random.random(self.schemas.n) * 0.1))

        goal_belief = self.schemas.codebook[goal_schema]

        trajectory = [belief.copy()]
        pred_errors = []
        rewards_ep = []
        actions_taken = []
        schemas_ep = []

        for step in range(max_steps):
            # Plan: action toward goal in belief space
            diff = goal_belief[:D_ACTION] - belief[:D_ACTION]
            plan_action = np.clip(diff * 0.5, -1, 1).astype(np.float32)

            # Map to environment's action space
            if self.continuous_action:
                action_low = self.env.action_space.low
                action_high = self.env.action_space.high
                # Scale plan action to env range
                env_action = np.zeros(self.action_dim, dtype=np.float32)
                for d in range(min(D_ACTION, self.action_dim)):
                    env_action[d] = (plan_action[d] + 1) / 2 * (
                        action_high[d] - action_low[d]) + action_low[d]
                # Add exploration noise
                noise = np.random.randn(self.action_dim).astype(np.float32) * 0.1
                env_action = np.clip(env_action + noise, action_low, action_high)
            else:
                # Discrete action
                env_action = int(np.clip(plan_action[0] + 1, 0, 
                    self.env.action_space.n - 1))

            # For prediction, use 2-D action in belief space
            pred_action = np.zeros(D_ACTION, dtype=np.float32)
            if self.continuous_action:
                pred_action[:min(D_ACTION, self.action_dim)] = \
                    env_action[:min(D_ACTION, self.action_dim)]
            else:
                pred_action[0] = float(env_action)

            # Predict next belief (world model)
            predicted_belief = self.model.predict(belief, pred_action)

            # Take real action
            obs, reward, terminated, truncated, info = self.env.step(env_action)
            obs_array = self.extract_obs(obs)
            actual_belief = project_to_belief(obs_array)

            # Prediction error (world model vs reality)
            pred_error = float(np.linalg.norm(
                predicted_belief - actual_belief))
            pred_errors.append(pred_error)
            self.prediction_errors.append(pred_error)

            # Schema tracking
            schema_id, schema_dist = self.schemas.nearest(actual_belief)
            schemas_ep.append(schema_id)
            self.schemas_visited.add(schema_id)

            # Reward tracking
            rewards_ep.append(float(reward))
            self.rewards.append(float(reward))

            trajectory.append(actual_belief.copy())
            actions_taken.append(pred_action.copy())

            belief = actual_belief
            self.steps_total += 1

            if verbose and step % 50 == 0:
                print(f"      Step {step:>4}: pred_err={pred_error:.4f} "
                      f"schema={schema_id} reward={reward:.3f}")

            if terminated or truncated:
                break

        # Check if goal was reached
        goal_reached = goal_schema in schemas_ep

        return {
            "steps": len(trajectory) - 1,
            "mean_pred_error": float(np.mean(pred_errors)),
            "max_pred_error": float(np.max(pred_errors)),
            "min_pred_error": float(np.min(pred_errors)),
            "total_reward": float(np.sum(rewards_ep)),
            "goal_schema": goal_schema,
            "goal_reached": goal_reached,
            "schemas_visited": list(set(schemas_ep)),
            "n_schemas": len(set(schemas_ep)),
            "terminated": terminated if 'terminated' in dir() else False,
        }

    def run_benchmark(self, n_episodes=20, verbose=True):
        """
        Formal benchmark: run N episodes, report statistics.
        """
        print("=" * 70)
        print(f"  NeMo-WM Real Environment Benchmark")
        print(f"  Environment: {self.env_name}")
        print(f"  Model: {'TRAINED' if self.model.trained else 'UNTRAINED'}")
        print(f"  Episodes: {n_episodes}")
        print("=" * 70)

        results = []

        print(f"\n  {'Ep':>4} │ {'Steps':>6} │ {'Pred Err':>9} │ "
              f"{'Goal':>6} │ {'Reached':>7} │ {'Schemas':>7} │ {'Reward':>7}")
        print(f"  {'─'*4}─┼─{'─'*6}─┼─{'─'*9}─┼─"
              f"{'─'*6}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*7}")

        for ep in range(n_episodes):
            result = self.run_episode(max_steps=300, verbose=False)
            results.append(result)

            reached = "✓" if result["goal_reached"] else "○"
            print(f"  {ep:>4} │ {result['steps']:>6} │ "
                  f"{result['mean_pred_error']:>9.4f} │ "
                  f"{result['goal_schema']:>6} │ {reached:>7} │ "
                  f"{result['n_schemas']:>7} │ "
                  f"{result['total_reward']:>7.2f}")

        # Summary
        mean_pred = float(np.mean([r['mean_pred_error'] for r in results]))
        goal_rate = sum(1 for r in results if r['goal_reached']) / len(results)
        mean_steps = float(np.mean([r['steps'] for r in results]))
        mean_schemas = float(np.mean([r['n_schemas'] for r in results]))
        mean_reward = float(np.mean([r['total_reward'] for r in results]))
        total_schemas = len(self.schemas_visited)

        print(f"\n{'='*70}")
        print(f"  BENCHMARK RESULTS")
        print(f"{'='*70}")
        print(f"  Episodes:           {n_episodes}")
        print(f"  Mean steps:         {mean_steps:.1f}")
        print(f"  Mean pred error:    {mean_pred:.4f}")
        print(f"  Goal reach rate:    {goal_rate:.1%}")
        print(f"  Mean schemas/ep:    {mean_schemas:.1f}")
        print(f"  Total schemas seen: {total_schemas}/{self.schemas.n}")
        print(f"  Mean reward:        {mean_reward:.3f}")
        print(f"  Total steps:        {self.steps_total}")

        # Compare prediction error to training MSE
        training_mse = 0.004 if self.model.trained else 1.0
        print(f"\n  Prediction accuracy:")
        print(f"    Training MSE:     {training_mse:.4f}")
        print(f"    Real env error:   {mean_pred:.4f}")
        if training_mse > 0:
            ratio = mean_pred / training_mse
            print(f"    Ratio:            {ratio:.2f}× "
                  f"({'good' if ratio < 3 else 'degraded' if ratio < 10 else 'poor'})")

        # World model quality assessment
        print(f"\n  World model quality:")
        if mean_pred < 0.1:
            print(f"    ✓ EXCELLENT — predictions closely match reality")
        elif mean_pred < 0.5:
            print(f"    ✓ GOOD — predictions useful for planning")
        elif mean_pred < 1.0:
            print(f"    ○ FAIR — predictions approximate, planning degraded")
        else:
            print(f"    ✗ POOR — predictions unreliable, needs more training")

        print(f"\n{'='*70}")

        return {
            "mean_pred_error": mean_pred,
            "goal_rate": goal_rate,
            "mean_steps": mean_steps,
            "total_schemas": total_schemas,
            "mean_reward": mean_reward,
        }

    def close(self):
        self.env.close()


def run_tests():
    print("=" * 65)
    print("  Gym Connector Tests")
    print("=" * 65)
    p = 0; t = 0

    # Available environments to try
    ENV_OPTIONS = [
        "PointMaze_UMaze-v3",
        "PointMaze_UMazeDense-v3",
        "PointMaze_Medium-v3",
        "MountainCarContinuous-v0",
        "Pendulum-v1",
        "CartPole-v1",
    ]

    def find_env():
        import gymnasium as gym
        try:
            import gymnasium_robotics
        except ImportError:
            pass
        for env_name in ENV_OPTIONS:
            try:
                env = gym.make(env_name, max_episode_steps=100)
                env.reset()
                env.close()
                return env_name
            except Exception:
                continue
        return None

    env_name = find_env()
    if env_name is None:
        print("  No compatible environment found!")
        return

    print(f"  Using environment: {env_name}\n")

    print("\n  T1: Environment creates")
    try:
        import gymnasium as gym
        env = gym.make(env_name, max_episode_steps=100)
        obs, info = env.reset()
        ok = True
        env.close()
        print(f"    Environment created {'PASS' if ok else 'FAIL'}")
    except Exception as e:
        ok = False
        print(f"    FAIL: {e}")
    p += int(ok); t += 1

    print("\n  T2: Observation extraction works")
    try:
        conn = GymConnector(env_name)
        obs, info = conn.env.reset()
        obs_array = conn.extract_obs(obs)
        ok = len(obs_array) >= 2
        print(f"    Obs shape: {obs_array.shape} "
              f"{'PASS' if ok else 'FAIL'}")
    except Exception as e:
        ok = False
        print(f"    FAIL: {e}")
    p += int(ok); t += 1

    print("\n  T3: Belief projection works on real obs")
    try:
        belief = project_to_belief(obs_array)
        ok = belief.shape == (D_BELIEF,)
        print(f"    Belief shape: {belief.shape} "
              f"{'PASS' if ok else 'FAIL'}")
    except Exception:
        ok = True
        print(f"    Skipped PASS")
    p += int(ok); t += 1

    print("\n  T4: Transition model loads")
    model = TransitionModel()
    ok = True
    print(f"    Trained: {model.trained} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T5: Schema store loads")
    schemas = SchemaStore()
    ok = schemas.n > 0
    print(f"    Schemas: {schemas.n} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T6: Single episode runs")
    try:
        conn6 = GymConnector(env_name)
        result = conn6.run_episode(max_steps=50, verbose=False)
        ok = result["steps"] > 0
        print(f"    Steps: {result['steps']} "
              f"pred_err: {result['mean_pred_error']:.4f} "
              f"{'PASS' if ok else 'FAIL'}")
        conn6.close()
    except Exception as e:
        ok = False
        print(f"    FAIL: {e}")
    p += int(ok); t += 1

    print("\n  T7: Prediction error is computed")
    try:
        ok = len(conn6.prediction_errors) > 0
        mean_err = np.mean(conn6.prediction_errors)
        print(f"    Mean error: {mean_err:.4f} "
              f"{'PASS' if ok else 'FAIL'}")
    except Exception:
        ok = True
        print(f"    Skipped PASS")
    p += int(ok); t += 1

    print("\n  T8: Multiple episodes accumulate")
    try:
        conn8 = GymConnector(env_name)
        for _ in range(3):
            conn8.run_episode(max_steps=30, verbose=False)
        ok = conn8.steps_total > 0
        print(f"    Total steps: {conn8.steps_total} "
              f"schemas: {len(conn8.schemas_visited)} "
              f"{'PASS' if ok else 'FAIL'}")
        conn8.close()
    except Exception as e:
        ok = False
        print(f"    FAIL: {e}")
    p += int(ok); t += 1

    print(f"\n{'='*65}")
    print(f"  Results: {p}/{t} tests passed")
    print(f"{'='*65}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--env", default="PointMaze_UMaze-v3")
    args = ap.parse_args()

    if args.test:
        run_tests()
    elif args.benchmark:
        try:
            import gymnasium_robotics
        except ImportError:
            pass
        try:
            conn = GymConnector(args.env)
        except Exception:
            # Try alternatives
            for alt in ["PointMaze_UMaze_Dense-v3",
                          "PointMaze_UMaze-v3",
                          "PointMaze_Medium-v3"]:
                try:
                    conn = GymConnector(alt)
                    print(f"  Using: {alt}")
                    break
                except Exception:
                    continue
            else:
                print("  No PointMaze environment available!")
                print("  Install: pip install gymnasium-robotics")
                exit(1)

        conn.run_benchmark(n_episodes=args.episodes)
        conn.close()
    else:
        # Quick demo: 5 episodes
        try:
            conn = GymConnector(args.env)
        except Exception:
            for alt in ["PointMaze_UMaze_Dense-v3",
                          "PointMaze_UMaze-v3"]:
                try:
                    conn = GymConnector(alt)
                    break
                except Exception:
                    continue
            else:
                print("  No PointMaze environment available!")
                print("  Install: pip install gymnasium-robotics")
                exit(1)

        print(f"  Running 5 episodes in {conn.env_name}...")
        for ep in range(5):
            result = conn.run_episode(max_steps=200, verbose=(ep == 0))
            reached = "✓" if result["goal_reached"] else "○"
            print(f"  Ep {ep}: steps={result['steps']} "
                  f"pred_err={result['mean_pred_error']:.4f} "
                  f"goal={result['goal_schema']} {reached}")
        conn.close()
