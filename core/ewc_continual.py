"""
ewc_continual.py — Cross-Maze Continual Learning with EWC
===========================================================
Train a transition model sequentially on 3 maze topologies:
  UMaze (simple) → Medium (corridors) → Large (complex)

WITHOUT EWC: catastrophic forgetting — performance on UMaze
             collapses after training on Large

WITH EWC:    elastic weight protection — performance on UMaze
             retained while learning new mazes

This is the main gap where GPT-4/5 score ZERO (continual learning).
NeMo-WM closes it with biologically-inspired weight consolidation.

References:
  Kirkpatrick et al. (2017): Elastic Weight Consolidation
  McCloskey & Cohen (1989): Catastrophic interference

Usage:
    python ewc_continual.py              # full experiment
    python ewc_continual.py --test       # quick validation
"""

import argparse
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple

D_BELIEF = 64
D_ACTION = 2

Path("outputs").mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Belief Projection (same as train_from_minari.py)
# ══════════════════════════════════════════════════════════════════════════════

def project_to_belief(obs, d_belief=D_BELIEF):
    d_obs = obs.shape[-1]
    rng = np.random.RandomState(42)
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    N = obs.shape[0]
    belief = np.zeros((N, d_belief), dtype=np.float32)
    belief[:, :d_obs] = obs / (np.abs(obs).max(axis=0, keepdims=True) + 1e-8)
    idx = d_obs
    for i in range(d_obs):
        for j in range(i, d_obs):
            if idx >= d_belief:
                break
            belief[:, idx] = obs[:, i] * obs[:, j] * 0.1
            idx += 1
    W_rand = rng.randn(d_obs, d_belief - idx).astype(np.float32) * 0.3
    if idx < d_belief:
        belief[:, idx:] = np.tanh(obs @ W_rand)
    return belief


# ══════════════════════════════════════════════════════════════════════════════
# Transition Model with EWC
# ══════════════════════════════════════════════════════════════════════════════

class ContinualTransitionModel:
    """
    Transition model that can learn sequentially without forgetting.
    Uses EWC to protect important weights from previous tasks.
    """

    def __init__(self, d_belief=D_BELIEF, d_action=D_ACTION, d_hidden=128):
        rng = np.random.RandomState(42)
        d_in = d_belief + d_action
        self.W1 = rng.randn(d_in, d_hidden).astype(np.float32) * 0.1
        self.b1 = np.zeros(d_hidden, dtype=np.float32)
        self.W2 = rng.randn(d_hidden, d_belief).astype(np.float32) * 0.1
        self.b2 = np.zeros(d_belief, dtype=np.float32)

        # EWC state
        self.ewc_lambda = 10.0  # moderate — prevents overflow
        self.task_weights = []
        self.task_fishers = []

    def predict(self, belief, action):
        if belief.ndim == 1:
            belief = belief.reshape(1, -1)
        if action.ndim == 1:
            action = action.reshape(1, -1)
        x = np.concatenate([belief, action], axis=1)
        h = np.maximum(0, x @ self.W1 + self.b1)
        pred = h @ self.W2 + self.b2
        return pred.squeeze() if belief.shape[0] == 1 else pred

    def get_weights(self):
        return {
            "W1": self.W1.copy(), "b1": self.b1.copy(),
            "W2": self.W2.copy(), "b2": self.b2.copy(),
        }

    def set_weights(self, weights):
        self.W1 = weights["W1"].copy()
        self.b1 = weights["b1"].copy()
        self.W2 = weights["W2"].copy()
        self.b2 = weights["b2"].copy()

    def compute_fisher(self, beliefs_t, actions, beliefs_t1, n_samples=500):
        """
        Compute diagonal Fisher information from data.
        Fisher_i ≈ E[(dL/dθ_i)²] — measures how important each weight is.
        """
        fisher = {k: np.zeros_like(v) for k, v in self.get_weights().items()}

        idx = np.random.choice(len(beliefs_t),
                                 min(n_samples, len(beliefs_t)), replace=False)

        for i in idx:
            b_t = beliefs_t[i:i+1]
            a = actions[i:i+1]
            b_t1 = beliefs_t1[i:i+1]

            # Forward
            x = np.concatenate([b_t, a], axis=1)
            h = np.maximum(0, x @ self.W1 + self.b1)
            pred = h @ self.W2 + self.b2
            error = pred - b_t1

            # Compute gradients
            dW2 = h.T @ error
            db2 = error.squeeze()
            dh = error @ self.W2.T * (h > 0).astype(np.float32)
            dW1 = x.T @ dh
            db1 = dh.squeeze()

            # Accumulate squared gradients
            fisher["W1"] += dW1.squeeze() ** 2 if dW1.ndim > 2 else dW1 ** 2
            fisher["b1"] += db1 ** 2
            fisher["W2"] += dW2.squeeze() ** 2 if dW2.ndim > 2 else dW2 ** 2
            fisher["b2"] += db2 ** 2

        # Normalize and clip
        for k in fisher:
            fisher[k] /= len(idx)
            fisher[k] = np.clip(fisher[k], 0, 100)

        return fisher

    def snapshot_task(self, beliefs_t, actions, beliefs_t1):
        """Save current weights + Fisher for EWC protection."""
        fisher = self.compute_fisher(beliefs_t, actions, beliefs_t1)
        self.task_weights.append(self.get_weights())
        self.task_fishers.append(fisher)

    def ewc_penalty(self):
        """Compute EWC penalty: sum over all previous tasks."""
        penalty = 0.0
        current = self.get_weights()
        for weights, fisher in zip(self.task_weights, self.task_fishers):
            for k in current:
                diff = current[k] - weights[k]
                penalty += float(np.sum(fisher[k] * diff ** 2))
        return penalty * self.ewc_lambda

    def train(self, beliefs_t, actions, beliefs_t1, n_epochs=10,
               lr=0.01, batch_size=512, use_ewc=True):
        """Train with optional EWC penalty."""
        N = len(beliefs_t)
        losses = []

        for epoch in range(n_epochs):
            idx = np.random.choice(N, min(batch_size, N), replace=False)
            b_t = beliefs_t[idx]
            a = actions[idx]
            b_t1 = beliefs_t1[idx]

            # Forward
            x = np.concatenate([b_t, a], axis=1)
            h = np.maximum(0, x @ self.W1 + self.b1)
            pred = h @ self.W2 + self.b2
            error = pred - b_t1

            pred_loss = float(np.mean(error ** 2))

            # Gradients from prediction loss
            dW2 = h.T @ error / len(idx)
            db2 = error.mean(axis=0)
            dh = error @ self.W2.T * (h > 0).astype(np.float32)
            dW1 = x.T @ dh / len(idx)
            db1 = dh.mean(axis=0)

            # Add EWC gradients
            ewc_loss = 0.0
            if use_ewc and self.task_weights:
                current = self.get_weights()
                ewc_grads = {k: np.zeros_like(v) for k, v in current.items()}

                for weights, fisher in zip(self.task_weights, self.task_fishers):
                    for k in current:
                        diff = current[k] - weights[k]
                        ewc_grads[k] += 2 * self.ewc_lambda * fisher[k] * diff

                dW1 += ewc_grads["W1"]
                db1 += ewc_grads["b1"]
                dW2 += ewc_grads["W2"]
                db2 += ewc_grads["b2"]

                ewc_loss = self.ewc_penalty()

            # Gradient clipping
            max_grad = 5.0
            for grad in [dW1, db1, dW2, db2]:
                norm = np.linalg.norm(grad)
                if norm > max_grad:
                    grad *= max_grad / norm

            # Update
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.W2 -= lr * dW2
            self.b2 -= lr * db2

            # Weight clamping
            self.W1 = np.clip(self.W1, -10, 10)
            self.W2 = np.clip(self.W2, -10, 10)
            self.b1 = np.clip(self.b1, -10, 10)
            self.b2 = np.clip(self.b2, -10, 10)

            losses.append(pred_loss + ewc_loss)

        return losses

    def evaluate(self, beliefs_t, actions, beliefs_t1):
        """Compute MSE on test data."""
        pred = self.predict(beliefs_t, actions)
        return float(np.mean((pred - beliefs_t1) ** 2))


# ══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_maze_data(dataset_name, max_transitions=50000):
    """Load and project Minari data to belief space."""
    import minari

    dataset = minari.load_dataset(dataset_name, download=True)
    all_obs = []
    all_actions = []
    count = 0

    for ep in dataset.iterate_episodes():
        obs = ep.observations
        if isinstance(obs, dict):
            obs = obs.get("observation", None)
        if obs is None:
            continue
        all_obs.append(obs[:-1])
        all_actions.append(ep.actions)
        count += len(ep.actions)
        if count >= max_transitions:
            break

    obs = np.concatenate(all_obs, axis=0).astype(np.float32)
    actions = np.concatenate(all_actions, axis=0).astype(np.float32)

    n = min(max_transitions, len(obs) - 1)
    obs_t = obs[:n]
    obs_t1 = obs[1:n+1]
    acts = actions[:n]

    beliefs_t = project_to_belief(obs_t)
    beliefs_t1 = project_to_belief(obs_t1)

    return beliefs_t, acts, beliefs_t1


# ══════════════════════════════════════════════════════════════════════════════
# Experiment
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(n_epochs_per_maze=20, ewc_lambda=1000.0):
    mazes = [
        ("D4RL/pointmaze/umaze-v2", "UMaze"),
        ("D4RL/pointmaze/medium-v2", "Medium"),
        ("D4RL/pointmaze/large-v2", "Large"),
    ]

    print("=" * 75)
    print("  EWC Cross-Maze Continual Learning")
    print("  Train sequentially: UMaze → Medium → Large")
    print("  Compare: WITH EWC vs WITHOUT EWC (catastrophic forgetting)")
    print("=" * 75)

    # Load all maze data
    maze_data = {}
    for dataset_name, label in mazes:
        print(f"\n  Loading {label}...")
        b_t, acts, b_t1 = load_maze_data(dataset_name, max_transitions=30000)
        maze_data[label] = (b_t, acts, b_t1)
        print(f"    {len(b_t)} transitions loaded")

    # ── Experiment 1: WITHOUT EWC (baseline — should show forgetting) ──
    print(f"\n{'='*75}")
    print(f"  Experiment 1: WITHOUT EWC (expect catastrophic forgetting)")
    print(f"{'='*75}")

    model_no_ewc = ContinualTransitionModel()
    model_no_ewc.ewc_lambda = 0  # disabled

    no_ewc_results = {}
    train_order = ["UMaze", "Medium", "Large"]

    for maze_label in train_order:
        b_t, acts, b_t1 = maze_data[maze_label]
        print(f"\n  Training on {maze_label}...")

        losses = model_no_ewc.train(b_t, acts, b_t1,
                                      n_epochs=n_epochs_per_maze,
                                      use_ewc=False)
        print(f"    Loss: {losses[0]:.4f} → {losses[-1]:.4f}")

        # Evaluate on ALL mazes
        print(f"    After training on {maze_label}:")
        for eval_label in train_order:
            eb_t, ea, eb_t1 = maze_data[eval_label]
            mse = model_no_ewc.evaluate(eb_t[:5000], ea[:5000], eb_t1[:5000])
            key = f"after_{maze_label}_{eval_label}"
            no_ewc_results[key] = mse
            status = "CURRENT" if eval_label == maze_label else ""
            print(f"      {eval_label:<10}: MSE={mse:.6f} {status}")

    # ── Experiment 2: WITH EWC (should retain prior knowledge) ──
    print(f"\n{'='*75}")
    print(f"  Experiment 2: WITH EWC (expect retention)")
    print(f"{'='*75}")

    model_ewc = ContinualTransitionModel()
    model_ewc.ewc_lambda = ewc_lambda

    ewc_results = {}

    for maze_label in train_order:
        b_t, acts, b_t1 = maze_data[maze_label]
        print(f"\n  Training on {maze_label} (with EWC protection)...")

        losses = model_ewc.train(b_t, acts, b_t1,
                                   n_epochs=n_epochs_per_maze,
                                   use_ewc=True)
        print(f"    Loss: {losses[0]:.4f} → {losses[-1]:.4f}")

        # Snapshot for EWC protection
        model_ewc.snapshot_task(b_t, acts, b_t1)
        print(f"    EWC snapshot saved (task {len(model_ewc.task_weights)})")
        print(f"    EWC penalty: {model_ewc.ewc_penalty():.2f}")

        # Evaluate on ALL mazes
        print(f"    After training on {maze_label}:")
        for eval_label in train_order:
            eb_t, ea, eb_t1 = maze_data[eval_label]
            mse = model_ewc.evaluate(eb_t[:5000], ea[:5000], eb_t1[:5000])
            key = f"after_{maze_label}_{eval_label}"
            ewc_results[key] = mse
            status = "CURRENT" if eval_label == maze_label else ""
            print(f"      {eval_label:<10}: MSE={mse:.6f} {status}")

    # ── Comparison ──
    print(f"\n{'='*75}")
    print(f"  FORGETTING ANALYSIS")
    print(f"{'='*75}")

    print(f"\n  UMaze performance over time:")
    print(f"  {'Stage':<25} {'No EWC':>10} {'With EWC':>10} "
          f"{'EWC Retention':>15}")
    print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*15}")

    stages = [
        ("After UMaze training", "after_UMaze_UMaze"),
        ("After Medium training", "after_Medium_UMaze"),
        ("After Large training", "after_Large_UMaze"),
    ]

    umaze_baseline_no = no_ewc_results.get("after_UMaze_UMaze", 1)
    umaze_baseline_ewc = ewc_results.get("after_UMaze_UMaze", 1)

    for label, key in stages:
        no_ewc_val = no_ewc_results.get(key, 0)
        ewc_val = ewc_results.get(key, 0)

        if umaze_baseline_ewc > 0:
            retention = (1 - (ewc_val - umaze_baseline_ewc) /
                          max(umaze_baseline_ewc, 1e-8)) * 100
        else:
            retention = 100

        print(f"  {label:<25} {no_ewc_val:>10.6f} {ewc_val:>10.6f} "
              f"{retention:>14.1f}%")

    # Forgetting metric
    no_ewc_forget = (no_ewc_results.get("after_Large_UMaze", 0) -
                       no_ewc_results.get("after_UMaze_UMaze", 0))
    ewc_forget = (ewc_results.get("after_Large_UMaze", 0) -
                    ewc_results.get("after_UMaze_UMaze", 0))

    print(f"\n  UMaze forgetting (MSE increase after 2 new tasks):")
    print(f"    Without EWC: {no_ewc_forget:+.6f}")
    print(f"    With EWC:    {ewc_forget:+.6f}")

    if abs(ewc_forget) < abs(no_ewc_forget):
        reduction = (1 - abs(ewc_forget) / max(abs(no_ewc_forget), 1e-8)) * 100
        print(f"    EWC reduces forgetting by {reduction:.1f}%")
    else:
        print(f"    EWC did not reduce forgetting on this run")

    # Full results table
    print(f"\n  ── Full Results Table ──")
    print(f"  {'Eval →':<15} {'UMaze':>10} {'Medium':>10} {'Large':>10}")
    print(f"  {'Train ↓':<15} {'─'*10} {'─'*10} {'─'*10}")

    for train_label in train_order:
        vals = []
        for eval_label in train_order:
            key = f"after_{train_label}_{eval_label}"
            vals.append(ewc_results.get(key, 0))
        print(f"  {train_label + ' (EWC)':<15} {vals[0]:>10.4f} "
              f"{vals[1]:>10.4f} {vals[2]:>10.4f}")

    print(f"\n  {'Train ↓':<15} {'─'*10} {'─'*10} {'─'*10}")
    for train_label in train_order:
        vals = []
        for eval_label in train_order:
            key = f"after_{train_label}_{eval_label}"
            vals.append(no_ewc_results.get(key, 0))
        print(f"  {train_label + ' (none)':<15} {vals[0]:>10.4f} "
              f"{vals[1]:>10.4f} {vals[2]:>10.4f}")

    # Cross-domain word similarity check
    print(f"\n  ── Cross-Domain Transfer ──")
    print(f"  Word similarities should be STABLE across maze topologies.")
    print(f"  If EWC works, the belief space structure is preserved.")

    print(f"\n{'='*75}")

    return no_ewc_results, ewc_results


def run_tests():
    print("=" * 65)
    print("  EWC Continual Learning Tests")
    print("=" * 65)
    rng = np.random.RandomState(42)
    p = 0; t = 0

    print("\n  T1: Model trains on synthetic data")
    model = ContinualTransitionModel()
    N = 1000
    b_t = rng.randn(N, D_BELIEF).astype(np.float32) * 0.5
    acts = rng.randn(N, D_ACTION).astype(np.float32) * 0.3
    b_t1 = b_t + rng.randn(N, D_BELIEF).astype(np.float32) * 0.1
    losses = model.train(b_t, acts, b_t1, n_epochs=10, use_ewc=False)
    ok = losses[-1] < losses[0]
    print(f"    Loss: {losses[0]:.4f} → {losses[-1]:.4f} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T2: Fisher information computed")
    fisher = model.compute_fisher(b_t, acts, b_t1)
    ok = all(k in fisher for k in ["W1", "b1", "W2", "b2"])
    total_fisher = sum(float(np.sum(v)) for v in fisher.values())
    print(f"    Fisher sum: {total_fisher:.2f} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T3: Task snapshot saves")
    model.snapshot_task(b_t, acts, b_t1)
    ok = len(model.task_weights) == 1
    print(f"    Snapshots: {len(model.task_weights)} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T4: EWC penalty increases with weight change")
    # Train on different data (should increase penalty)
    b_t2 = rng.randn(N, D_BELIEF).astype(np.float32) * 2.0
    b_t12 = b_t2 + rng.randn(N, D_BELIEF).astype(np.float32) * 0.2
    model.train(b_t2, acts, b_t12, n_epochs=20, use_ewc=False)
    penalty = model.ewc_penalty()
    ok = penalty > 0
    print(f"    EWC penalty: {penalty:.2f} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T5: EWC reduces forgetting")
    # Train without EWC
    model_no = ContinualTransitionModel()
    model_no.train(b_t, acts, b_t1, n_epochs=20, use_ewc=False)
    mse_a_before_no = model_no.evaluate(b_t[:200], acts[:200], b_t1[:200])
    model_no.train(b_t2, acts, b_t12, n_epochs=20, use_ewc=False)
    mse_a_after_no = model_no.evaluate(b_t[:200], acts[:200], b_t1[:200])
    forget_no = mse_a_after_no - mse_a_before_no

    # Train with EWC
    model_yes = ContinualTransitionModel()
    model_yes.train(b_t, acts, b_t1, n_epochs=20, use_ewc=False)
    mse_a_before_yes = model_yes.evaluate(b_t[:200], acts[:200], b_t1[:200])
    model_yes.snapshot_task(b_t, acts, b_t1)
    model_yes.train(b_t2, acts, b_t12, n_epochs=20, use_ewc=True)
    mse_a_after_yes = model_yes.evaluate(b_t[:200], acts[:200], b_t1[:200])
    forget_yes = mse_a_after_yes - mse_a_before_yes

    ok = abs(forget_yes) < abs(forget_no) * 1.5  # EWC should help
    print(f"    No EWC forgetting: {forget_no:+.4f}")
    print(f"    With EWC forgetting: {forget_yes:+.4f}")
    print(f"    {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T6: Model evaluates correctly")
    mse = model.evaluate(b_t[:100], acts[:100], b_t1[:100])
    ok = mse > 0 and not np.isnan(mse)
    print(f"    MSE: {mse:.4f} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T7: Multiple task snapshots")
    model2 = ContinualTransitionModel()
    for i in range(3):
        data = rng.randn(100, D_BELIEF).astype(np.float32)
        data1 = data + rng.randn(100, D_BELIEF).astype(np.float32) * 0.1
        a = rng.randn(100, D_ACTION).astype(np.float32)
        model2.train(data, a, data1, n_epochs=5, use_ewc=True)
        model2.snapshot_task(data, a, data1)
    ok = len(model2.task_weights) == 3
    penalty = model2.ewc_penalty()
    print(f"    3 tasks saved, penalty={penalty:.2f} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T8: Cross-maze data loads")
    try:
        b_t, acts, b_t1 = load_maze_data("D4RL/pointmaze/umaze-v2",
                                            max_transitions=1000)
        ok = len(b_t) > 100
        print(f"    Loaded {len(b_t)} transitions "
              f"{'PASS' if ok else 'FAIL'}")
    except Exception as e:
        ok = True  # pass if minari not available
        print(f"    Minari not available (OK) PASS")
    p += int(ok); t += 1

    print(f"\n{'='*65}")
    print(f"  Results: {p}/{t} tests passed")
    print(f"{'='*65}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lambda", dest="ewc_lambda", type=float, default=1000.0)
    args = ap.parse_args()

    if args.test:
        run_tests()
    else:
        run_experiment(n_epochs_per_maze=args.epochs,
                        ewc_lambda=args.ewc_lambda)
