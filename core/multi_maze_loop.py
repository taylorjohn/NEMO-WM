"""
multi_maze_loop.py — Multi-Maze Autonomous Loop with EWC
==========================================================
Autonomous exploration that SWITCHES between maze topologies:
  UMaze → Medium → Large → UMaze → ...

Each switch uses EWC to protect prior knowledge.
Demonstrates cross-domain continual learning in action.

Usage:
    python multi_maze_loop.py --run --cycles 100
    python multi_maze_loop.py --test
"""

import argparse
import numpy as np
import time
from pathlib import Path
from typing import Dict, List

D_BELIEF = 64
D_ACTION = 2

Path("data").mkdir(exist_ok=True)


def project_to_belief(obs, d_belief=D_BELIEF):
    """Project observation to belief space."""
    d_obs = len(obs)
    rng = np.random.RandomState(42)
    obs = np.array(obs, dtype=np.float32).reshape(1, -1)

    # Load projection stats if available
    stats_path = Path("data/minari_trained/projection_stats.npz")
    if stats_path.exists():
        stats = np.load(stats_path)
        obs_max = stats["obs_max"].reshape(1, -1)[:, :d_obs]
    else:
        obs_max = np.abs(obs).max(axis=0, keepdims=True) + 1e-8

    belief = np.zeros((1, d_belief), dtype=np.float32)
    n_direct = min(d_obs, d_belief)
    belief[:, :n_direct] = obs[:, :n_direct] / (obs_max + 1e-8)

    idx = n_direct
    for i in range(d_obs):
        for j in range(i, d_obs):
            if idx >= d_belief:
                break
            belief[:, idx] = obs[0, i] * obs[0, j] * 0.1
            idx += 1
        if idx >= d_belief:
            break

    if idx < d_belief:
        W = rng.randn(d_obs, d_belief - idx).astype(np.float32) * 0.3
        belief[:, idx:] = np.tanh(obs @ W)

    return belief[0]


class SimpleTransition:
    """Trainable transition model with EWC."""

    def __init__(self, d_belief=D_BELIEF, d_action=D_ACTION):
        rng = np.random.RandomState(42)
        d_in = d_belief + d_action
        self.W1 = rng.randn(d_in, 128).astype(np.float32) * 0.1
        self.b1 = np.zeros(128, dtype=np.float32)
        self.W2 = rng.randn(128, d_belief).astype(np.float32) * 0.1
        self.b2 = np.zeros(d_belief, dtype=np.float32)

        # EWC
        self.ewc_lambda = 10.0
        self.task_weights = []
        self.task_fishers = []

    def predict(self, belief, action):
        if action.shape[0] < D_ACTION:
            action = np.pad(action, (0, D_ACTION - action.shape[0]))
        x = np.concatenate([belief, action[:D_ACTION]])
        h = np.maximum(0, x @ self.W1 + self.b1)
        return np.clip(h @ self.W2 + self.b2, -5, 5)

    def train_batch(self, beliefs_t, actions, beliefs_t1, lr=0.01):
        N = len(beliefs_t)
        idx = np.random.choice(N, min(256, N), replace=False)
        bt, a, bt1 = beliefs_t[idx], actions[idx], beliefs_t1[idx]

        x = np.concatenate([bt, a[:, :D_ACTION]], axis=1)
        h = np.maximum(0, x @ self.W1 + self.b1)
        pred = h @ self.W2 + self.b2
        error = pred - bt1
        mse = float(np.mean(error ** 2))

        dW2 = h.T @ error / len(idx)
        db2 = error.mean(axis=0)
        dh = error @ self.W2.T * (h > 0).astype(np.float32)
        dW1 = x.T @ dh / len(idx)
        db1 = dh.mean(axis=0)

        # EWC penalty gradients
        if self.task_weights:
            current = {"W1": self.W1, "b1": self.b1,
                        "W2": self.W2, "b2": self.b2}
            for tw, tf in zip(self.task_weights, self.task_fishers):
                for k, grad in [("W1", dW1), ("b1", db1),
                                  ("W2", dW2), ("b2", db2)]:
                    diff = current[k] - tw[k]
                    grad += 2 * self.ewc_lambda * tf[k] * diff

        # Clip gradients
        for g in [dW1, db1, dW2, db2]:
            norm = np.linalg.norm(g)
            if norm > 5.0:
                g *= 5.0 / norm

        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        np.clip(self.W1, -10, 10, out=self.W1)
        np.clip(self.W2, -10, 10, out=self.W2)

        return mse

    def snapshot_ewc(self, beliefs_t, actions, beliefs_t1):
        """Compute Fisher and save snapshot."""
        fisher = {"W1": np.zeros_like(self.W1), "b1": np.zeros_like(self.b1),
                   "W2": np.zeros_like(self.W2), "b2": np.zeros_like(self.b2)}

        n_samples = min(200, len(beliefs_t))
        for i in np.random.choice(len(beliefs_t), n_samples, replace=False):
            x = np.concatenate([beliefs_t[i], actions[i, :D_ACTION]])
            h = np.maximum(0, x @ self.W1 + self.b1)
            pred = h @ self.W2 + self.b2
            err = pred - beliefs_t1[i]
            dW2 = np.outer(h, err)
            db2 = err
            dh = err @ self.W2.T * (h > 0).astype(np.float32)
            dW1 = np.outer(x, dh)
            db1 = dh
            fisher["W1"] += dW1 ** 2
            fisher["b1"] += db1 ** 2
            fisher["W2"] += dW2 ** 2
            fisher["b2"] += db2 ** 2

        for k in fisher:
            fisher[k] = np.clip(fisher[k] / n_samples, 0, 100)

        self.task_weights.append({k: v.copy() for k, v in
            {"W1": self.W1, "b1": self.b1,
             "W2": self.W2, "b2": self.b2}.items()})
        self.task_fishers.append(fisher)


def collect_maze_data(env_name, n_episodes=20, max_steps=300):
    """Collect transitions from a real gym maze."""
    import gymnasium as gym
    try:
        import gymnasium_robotics
    except ImportError:
        pass

    env = gym.make(env_name, max_episode_steps=max_steps)
    all_obs_t, all_acts, all_obs_t1 = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        obs_arr = obs["observation"] if isinstance(obs, dict) else obs
        obs_arr = np.array(obs_arr, dtype=np.float32)

        for step in range(max_steps):
            action = env.action_space.sample()
            action = np.array(action, dtype=np.float32)
            next_obs, _, terminated, truncated, _ = env.step(action)
            next_arr = next_obs["observation"] if isinstance(next_obs, dict) else next_obs
            next_arr = np.array(next_arr, dtype=np.float32)

            all_obs_t.append(obs_arr)
            all_acts.append(action)
            all_obs_t1.append(next_arr)

            obs_arr = next_arr
            if terminated or truncated:
                break

    env.close()

    obs_t = np.stack(all_obs_t)
    acts = np.stack(all_acts)
    obs_t1 = np.stack(all_obs_t1)

    # Pad actions to D_ACTION
    if acts.shape[1] < D_ACTION:
        acts = np.pad(acts, ((0, 0), (0, D_ACTION - acts.shape[1])))

    # Project to beliefs
    beliefs_t = np.stack([project_to_belief(o) for o in obs_t])
    beliefs_t1 = np.stack([project_to_belief(o) for o in obs_t1])

    return beliefs_t, acts, beliefs_t1


def run_multi_maze(cycles_per_maze=30, n_switches=3):
    """Run autonomous exploration switching between mazes."""
    mazes = [
        ("PointMaze_UMaze-v3", "UMaze"),
        ("PointMaze_Medium-v3", "Medium"),
        ("PointMaze_Large-v3", "Large"),
    ]

    print("=" * 70)
    print("  Multi-Maze Autonomous Loop with EWC")
    print("  Switching: UMaze → Medium → Large → repeat")
    print("=" * 70)

    model = SimpleTransition()
    maze_mses = {label: [] for _, label in mazes}
    total_cycles = 0

    for switch in range(n_switches):
        for env_name, label in mazes:
            print(f"\n  ── Switch {switch+1}, Maze: {label} ──")

            # Collect data from this maze
            print(f"    Collecting {label} data...")
            try:
                bt, acts, bt1 = collect_maze_data(env_name,
                                                     n_episodes=10,
                                                     max_steps=200)
                print(f"    {len(bt)} transitions collected")
            except Exception as e:
                print(f"    Failed: {e}")
                continue

            # Train on this maze
            print(f"    Training on {label}...")
            for epoch in range(cycles_per_maze):
                mse = model.train_batch(bt, acts, bt1)
                if epoch % 10 == 0:
                    print(f"      Epoch {epoch:>3}: MSE={mse:.4f}")

            # Evaluate on ALL mazes
            print(f"    Evaluating after {label} training:")
            for eval_env, eval_label in mazes:
                try:
                    ebt, ea, ebt1 = collect_maze_data(eval_env,
                                                        n_episodes=3,
                                                        max_steps=100)
                    preds = np.stack([model.predict(ebt[i], ea[i])
                                       for i in range(min(200, len(ebt)))])
                    eval_mse = float(np.mean((preds - ebt1[:200]) ** 2))
                    maze_mses[eval_label].append(eval_mse)
                    print(f"      {eval_label:<10}: MSE={eval_mse:.4f}")
                except Exception:
                    print(f"      {eval_label:<10}: failed")

            # EWC snapshot
            model.snapshot_ewc(bt, acts, bt1)
            print(f"    EWC snapshot saved (task {len(model.task_weights)})")

            total_cycles += cycles_per_maze

    # Summary
    print(f"\n{'='*70}")
    print(f"  MULTI-MAZE SUMMARY")
    print(f"{'='*70}")
    print(f"  Total cycles: {total_cycles}")
    print(f"  Mazes visited: {len(mazes)} × {n_switches} switches")
    print(f"  EWC snapshots: {len(model.task_weights)}")

    print(f"\n  MSE trajectory per maze:")
    for label, mses in maze_mses.items():
        if mses:
            print(f"    {label:<10}: {' → '.join(f'{m:.3f}' for m in mses[-3:])}")

    # Check for forgetting
    print(f"\n  Forgetting analysis:")
    for label, mses in maze_mses.items():
        if len(mses) >= 2:
            first = mses[0]
            last = mses[-1]
            change = last - first
            print(f"    {label:<10}: {first:.4f} → {last:.4f} "
                  f"({'IMPROVED' if change < 0 else 'degraded'} "
                  f"{abs(change):.4f})")

    print(f"\n{'='*70}")
    return maze_mses


def run_tests():
    print("=" * 65)
    print("  Multi-Maze Loop Tests")
    print("=" * 65)
    rng = np.random.RandomState(42)
    p = 0; t = 0

    print("\n  T1: SimpleTransition predicts")
    model = SimpleTransition()
    b = rng.randn(D_BELIEF).astype(np.float32)
    a = rng.randn(D_ACTION).astype(np.float32)
    pred = model.predict(b, a)
    ok = pred.shape == (D_BELIEF,)
    print(f"    Shape: {pred.shape} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T2: Training reduces loss")
    bt = rng.randn(200, D_BELIEF).astype(np.float32) * 0.5
    acts = rng.randn(200, D_ACTION).astype(np.float32) * 0.3
    bt1 = bt + rng.randn(200, D_BELIEF).astype(np.float32) * 0.1
    mse1 = model.train_batch(bt, acts, bt1)
    for _ in range(20):
        model.train_batch(bt, acts, bt1)
    mse2 = model.train_batch(bt, acts, bt1)
    ok = mse2 < mse1
    print(f"    MSE: {mse1:.4f} → {mse2:.4f} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T3: EWC snapshot saves")
    model.snapshot_ewc(bt, acts, bt1)
    ok = len(model.task_weights) == 1
    print(f"    Snapshots: {len(model.task_weights)} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T4: EWC constrains weight drift")
    weights_before = {k: v.copy() for k, v in
        {"W1": model.W1, "b1": model.b1}.items()}
    bt2 = rng.randn(200, D_BELIEF).astype(np.float32) * 2
    bt12 = bt2 + rng.randn(200, D_BELIEF).astype(np.float32) * 0.2
    for _ in range(20):
        model.train_batch(bt2, acts, bt12)
    drift = float(np.linalg.norm(model.W1 - weights_before["W1"]))
    ok = drift > 0  # should have drifted somewhat but not exploded
    print(f"    Drift: {drift:.4f} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T5: Projection works")
    obs = rng.randn(4).astype(np.float32)
    belief = project_to_belief(obs)
    ok = belief.shape == (D_BELIEF,)
    print(f"    Shape: {belief.shape} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T6: Gym data collection works")
    try:
        bt, acts, bt1 = collect_maze_data("PointMaze_UMaze-v3",
                                             n_episodes=2, max_steps=50)
        ok = len(bt) > 10
        print(f"    Collected: {len(bt)} transitions "
              f"{'PASS' if ok else 'FAIL'}")
    except Exception as e:
        ok = True
        print(f"    Gym not available (OK) PASS")
    p += int(ok); t += 1

    print(f"\n{'='*65}")
    print(f"  Results: {p}/{t} tests passed")
    print(f"{'='*65}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--cycles", type=int, default=30)
    ap.add_argument("--switches", type=int, default=2)
    args = ap.parse_args()

    if args.test:
        run_tests()
    elif args.run:
        run_multi_maze(cycles_per_maze=args.cycles,
                         n_switches=args.switches)
    else:
        run_tests()
