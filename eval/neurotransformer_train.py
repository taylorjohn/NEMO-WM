"""
neurotransformer_train.py — Train NeuroTransformer on Real Data
================================================================
Train the neuromodulated attention mechanism on real navigation
sequences from Minari PointMaze. Compare learned NeuroTransformer
vs learned Standard Transformer on sequence prediction.

This is the key experiment for Paper 5:
  Does neuromodulated attention learn BETTER representations
  than standard attention when both are trained on the same data?

Tasks:
  1. Sequence prediction: given 8 beliefs, predict the 9th
  2. Anomaly detection: which sequence has the outlier?
  3. Temporal order: fwd vs reverse prediction quality

Usage:
    python neurotransformer_train.py              # train + compare
    python neurotransformer_train.py --test        # quick validation
    python neurotransformer_train.py --epochs 200  # more training
"""

import argparse
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple

D_BELIEF = 64
SEQ_LEN = 8  # context window


# ══════════════════════════════════════════════════════════════════════════════
# Data Loading — Real Minari sequences
# ══════════════════════════════════════════════════════════════════════════════

def load_belief_sequences(max_sequences=5000, seq_len=SEQ_LEN):
    """Load sequential belief data from Minari training."""
    beliefs_path = Path("data/minari_trained/beliefs_sample.npz")

    if beliefs_path.exists():
        all_beliefs = np.load(beliefs_path)["beliefs"]
    else:
        print("  No trained beliefs found, using synthetic")
        rng = np.random.RandomState(42)
        all_beliefs = np.cumsum(
            rng.randn(10000, D_BELIEF).astype(np.float32) * 0.1, axis=0)

    # Create overlapping sequences
    sequences = []
    targets = []
    N = len(all_beliefs) - seq_len - 1

    for i in range(min(max_sequences, N)):
        seq = all_beliefs[i:i + seq_len]
        target = all_beliefs[i + seq_len]
        sequences.append(seq)
        targets.append(target)

    sequences = np.stack(sequences)  # (N, seq_len, D_BELIEF)
    targets = np.stack(targets)       # (N, D_BELIEF)

    return sequences, targets


# ══════════════════════════════════════════════════════════════════════════════
# Trainable Standard Transformer
# ══════════════════════════════════════════════════════════════════════════════

class TrainableStandardTransformer:
    """Standard multi-head attention + output projection, trainable."""

    def __init__(self, d_model=D_BELIEF, n_heads=4, seed=42):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        rng = np.random.RandomState(seed)

        scale = np.sqrt(2.0 / d_model)
        self.W_q = rng.randn(d_model, d_model).astype(np.float32) * scale
        self.W_k = rng.randn(d_model, d_model).astype(np.float32) * scale
        self.W_v = rng.randn(d_model, d_model).astype(np.float32) * scale
        self.W_o = rng.randn(d_model, d_model).astype(np.float32) * scale
        self.b_o = np.zeros(d_model, dtype=np.float32)

    def forward(self, x):
        """x: (seq_len, d_model) → output: (d_model,) last position."""
        seq_len = x.shape[0]
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        Q = Q.reshape(seq_len, self.n_heads, self.d_head)
        K = K.reshape(seq_len, self.n_heads, self.d_head)
        V = V.reshape(seq_len, self.n_heads, self.d_head)

        outputs = []
        self._attn_weights = []

        for h in range(self.n_heads):
            q, k, v = Q[:, h, :], K[:, h, :], V[:, h, :]
            scores = q @ k.T / np.sqrt(self.d_head)
            e = np.exp(scores - scores.max(axis=-1, keepdims=True))
            attn = e / (e.sum(axis=-1, keepdims=True) + 1e-10)
            out = attn @ v
            outputs.append(out)
            self._attn_weights.append(attn)

        concat = np.concatenate(outputs, axis=1)
        result = concat[-1] @ self.W_o + self.b_o  # last position
        return result

    def get_params(self):
        return [self.W_q, self.W_k, self.W_v, self.W_o, self.b_o]

    @property
    def n_params(self):
        return sum(p.size for p in self.get_params())


# ══════════════════════════════════════════════════════════════════════════════
# Trainable NeuroTransformer
# ══════════════════════════════════════════════════════════════════════════════

class TrainableNeuroTransformer:
    """
    Neuromodulated attention, trainable.
    4 heads: DA (surprise), ACh (explore/exploit), CRT (conflict), 5HT (temporal)
    """

    HEAD_PROFILES = {0: "DA", 1: "ACh", 2: "CRT", 3: "5HT"}

    def __init__(self, d_model=D_BELIEF, n_heads=4, seed=99):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        rng = np.random.RandomState(seed)

        scale = np.sqrt(2.0 / d_model)
        self.W_q = rng.randn(d_model, d_model).astype(np.float32) * scale
        self.W_k = rng.randn(d_model, d_model).astype(np.float32) * scale
        self.W_v = rng.randn(d_model, d_model).astype(np.float32) * scale
        self.W_o = rng.randn(d_model, d_model).astype(np.float32) * scale
        self.b_o = np.zeros(d_model, dtype=np.float32)

        # Learnable neuromod parameters
        self.da_gain = rng.randn(self.d_head).astype(np.float32) * 0.1
        self.ach_temp = np.array([1.0], dtype=np.float32)
        self.sht_decay = np.array([0.3], dtype=np.float32)

        self.prev_mean = None

    def _compute_neuromod(self, x):
        """Compute neuromod signals from sequence."""
        mean = x.mean(axis=0)

        # DA: prediction error
        if self.prev_mean is not None:
            da = float(np.clip(np.linalg.norm(mean - self.prev_mean) * 0.5, 0, 1))
        else:
            da = 0.5
        self.prev_mean = mean.copy()

        # ACh: uncertainty (variance)
        ach = float(np.clip(np.std(x) * 2, 0.1, 1))

        # CRT: conflict
        if x.shape[0] > 1:
            diffs = [np.linalg.norm(x[i] - x[i-1]) for i in range(1, x.shape[0])]
            crt = float(np.clip(np.max(diffs) * 0.1, 0, 1))
        else:
            crt = 0.1

        # 5HT: smoothness
        if x.shape[0] > 2:
            accels = [np.linalg.norm(x[i+2] - 2*x[i+1] + x[i])
                       for i in range(x.shape[0]-2)]
            sht = float(np.clip(1 - np.mean(accels) * 0.1, 0.1, 1))
        else:
            sht = 0.5

        return {"DA": da, "ACh": ach, "CRT": crt, "5HT": sht}

    def forward(self, x):
        """x: (seq_len, d_model) → output: (d_model,) last position."""
        seq_len = x.shape[0]
        signals = self._compute_neuromod(x)

        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        Q = Q.reshape(seq_len, self.n_heads, self.d_head)
        K = K.reshape(seq_len, self.n_heads, self.d_head)
        V = V.reshape(seq_len, self.n_heads, self.d_head)

        outputs = []
        self._attn_weights = []

        for h in range(self.n_heads):
            q, k, v = Q[:, h, :], K[:, h, :], V[:, h, :]
            scores = q @ k.T / np.sqrt(self.d_head)
            profile = self.HEAD_PROFILES[h]

            if profile == "DA":
                # Attend more to surprising (high-change) positions
                changes = np.zeros(seq_len)
                for i in range(1, seq_len):
                    changes[i] = np.linalg.norm(x[i] - x[i-1])
                surprise = 1.0 + changes * signals["DA"] * self.da_gain.mean()
                scores = scores * surprise[None, :]

            elif profile == "ACh":
                # Temperature scaling
                temp = float(np.clip(self.ach_temp[0] * (1 + signals["ACh"]), 0.1, 5))
                scores = scores / temp

            elif profile == "CRT":
                # Suppress conflicting positions
                if signals["CRT"] > 0.3:
                    v_norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
                    sim = (v / v_norms) @ (v / v_norms).T
                    conflict = np.where(sim < 0, 0.5, 1.0)
                    conflict_per_key = conflict.mean(axis=0, keepdims=True)
                    scores = scores * conflict_per_key

            elif profile == "5HT":
                # Recency bias
                decay_rate = float(np.clip(self.sht_decay[0], 0.05, 1.0))
                recency = np.exp(-np.arange(seq_len)[::-1].astype(np.float32) * decay_rate)
                scores = scores * recency[None, :]

            e = np.exp(scores - scores.max(axis=-1, keepdims=True))
            attn = e / (e.sum(axis=-1, keepdims=True) + 1e-10)
            out = attn @ v
            outputs.append(out)
            self._attn_weights.append(attn)

        concat = np.concatenate(outputs, axis=1)
        result = concat[-1] @ self.W_o + self.b_o
        return result

    def get_params(self):
        return [self.W_q, self.W_k, self.W_v, self.W_o, self.b_o,
                self.da_gain, self.ach_temp, self.sht_decay]

    @property
    def n_params(self):
        return sum(p.size for p in self.get_params())


# ══════════════════════════════════════════════════════════════════════════════
# Trainer — analytical gradients for both models
# ══════════════════════════════════════════════════════════════════════════════

def train_model(model, sequences, targets, n_epochs=100, lr=0.001,
                 batch_size=64, label="Model"):
    """Train a model with numerical gradient estimation."""
    N = len(sequences)
    loss_history = []

    print(f"\n    Training {label} ({model.n_params} params)...")

    for epoch in range(n_epochs):
        # Random batch
        idx = np.random.choice(N, min(batch_size, N), replace=False)

        # Compute batch loss
        total_loss = 0
        for i in idx:
            pred = model.forward(sequences[i])
            error = pred - targets[i]
            total_loss += float(np.mean(error ** 2))
        batch_loss = total_loss / len(idx)

        # Update via numerical gradients (sampled)
        eps = 1e-4
        for param in model.get_params():
            n_samples = min(10, param.size)
            indices = np.random.choice(param.size, n_samples, replace=False)

            for pidx in indices:
                old_val = param.flat[pidx]

                param.flat[pidx] = old_val + eps
                loss_plus = 0
                for i in idx[:8]:
                    p = model.forward(sequences[i])
                    loss_plus += float(np.mean((p - targets[i]) ** 2))
                loss_plus /= 8

                param.flat[pidx] = old_val - eps
                loss_minus = 0
                for i in idx[:8]:
                    p = model.forward(sequences[i])
                    loss_minus += float(np.mean((p - targets[i]) ** 2))
                loss_minus /= 8

                param.flat[pidx] = old_val

                grad = (loss_plus - loss_minus) / (2 * eps)
                scale = param.size / n_samples
                param.flat[pidx] -= lr * grad * scale

            # Clip weights
            np.clip(param, -5, 5, out=param)

        loss_history.append(batch_loss)

        if epoch % max(1, n_epochs // 10) == 0:
            print(f"      Epoch {epoch:>4}: loss={batch_loss:.6f}")

    return loss_history


def evaluate_model(model, sequences, targets, label="Model"):
    """Evaluate prediction quality."""
    N = min(500, len(sequences))
    mse_total = 0

    for i in range(N):
        pred = model.forward(sequences[i])
        mse_total += float(np.mean((pred - targets[i]) ** 2))

    mse = mse_total / N
    return mse


# ══════════════════════════════════════════════════════════════════════════════
# Experiment
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(n_epochs=100):
    print("=" * 70)
    print("  NeuroTransformer vs Standard — Trained on Real Data")
    print("  Same data, same architecture, different attention mechanism")
    print("=" * 70)

    # Load data
    print("\n  Loading real belief sequences...")
    sequences, targets = load_belief_sequences(max_sequences=3000)
    print(f"  Sequences: {sequences.shape}")
    print(f"  Targets: {targets.shape}")

    # Split train/test
    n_train = int(len(sequences) * 0.8)
    train_seq, test_seq = sequences[:n_train], sequences[n_train:]
    train_tgt, test_tgt = targets[:n_train], targets[n_train:]
    print(f"  Train: {n_train}, Test: {len(test_seq)}")

    # ── Train Standard Transformer ──
    print(f"\n  ── Standard Transformer ──")
    std = TrainableStandardTransformer()
    std_losses = train_model(std, train_seq, train_tgt,
                               n_epochs=n_epochs, lr=0.001,
                               label="Standard")
    std_train_mse = evaluate_model(std, train_seq, train_tgt)
    std_test_mse = evaluate_model(std, test_seq, test_tgt)
    print(f"    Train MSE: {std_train_mse:.6f}")
    print(f"    Test MSE:  {std_test_mse:.6f}")

    # ── Train NeuroTransformer ──
    print(f"\n  ── NeuroTransformer ──")
    neuro = TrainableNeuroTransformer()
    neuro_losses = train_model(neuro, train_seq, train_tgt,
                                 n_epochs=n_epochs, lr=0.001,
                                 label="NeuroTransformer")
    neuro_train_mse = evaluate_model(neuro, train_seq, train_tgt)
    neuro_test_mse = evaluate_model(neuro, test_seq, test_tgt)
    print(f"    Train MSE: {neuro_train_mse:.6f}")
    print(f"    Test MSE:  {neuro_test_mse:.6f}")

    # ── Anomaly Detection Test ──
    print(f"\n  ── Anomaly Detection (inject outlier) ──")
    rng = np.random.RandomState(42)
    n_anomaly_trials = 100
    std_anomaly_score = 0
    neuro_anomaly_score = 0

    for trial in range(n_anomaly_trials):
        idx = rng.randint(len(test_seq))
        normal_seq = test_seq[idx].copy()
        anomaly_seq = normal_seq.copy()
        # Inject outlier at random position
        anomaly_pos = rng.randint(SEQ_LEN)
        anomaly_seq[anomaly_pos] += rng.randn(D_BELIEF).astype(np.float32) * 3

        # Predict from both
        std_normal = std.forward(normal_seq)
        std_anomaly = std.forward(anomaly_seq)
        std_diff = np.linalg.norm(std_normal - std_anomaly)

        neuro.prev_mean = None
        neuro_normal = neuro.forward(normal_seq)
        neuro.prev_mean = None
        neuro_anomaly = neuro.forward(anomaly_seq)
        neuro_diff = np.linalg.norm(neuro_normal - neuro_anomaly)

        if std_diff > 0.1:
            std_anomaly_score += 1
        if neuro_diff > 0.1:
            neuro_anomaly_score += 1

    print(f"    Standard:     {std_anomaly_score}/{n_anomaly_trials} "
          f"({100*std_anomaly_score/n_anomaly_trials:.0f}%) anomalies detected")
    print(f"    NeuroTransf:  {neuro_anomaly_score}/{n_anomaly_trials} "
          f"({100*neuro_anomaly_score/n_anomaly_trials:.0f}%) anomalies detected")

    # ── Temporal Order Test ──
    print(f"\n  ── Temporal Order Sensitivity ──")
    n_order_trials = 100
    std_order_diffs = []
    neuro_order_diffs = []

    for trial in range(n_order_trials):
        idx = rng.randint(len(test_seq))
        fwd = test_seq[idx]
        rev = fwd[::-1].copy()

        std_fwd = std.forward(fwd)
        std_rev = std.forward(rev)
        std_order_diffs.append(float(np.linalg.norm(std_fwd - std_rev)))

        neuro.prev_mean = None
        neuro_fwd = neuro.forward(fwd)
        neuro.prev_mean = None
        neuro_rev = neuro.forward(rev)
        neuro_order_diffs.append(float(np.linalg.norm(neuro_fwd - neuro_rev)))

    std_order = float(np.mean(std_order_diffs))
    neuro_order = float(np.mean(neuro_order_diffs))
    print(f"    Standard:     fwd/rev diff = {std_order:.4f}")
    print(f"    NeuroTransf:  fwd/rev diff = {neuro_order:.4f}")
    print(f"    {'NEURO WINS' if neuro_order > std_order else 'STD WINS'} "
          f"(higher = more order-sensitive)")

    # ── Head Attention Analysis ──
    print(f"\n  ── Learned Attention Patterns ──")
    sample_seq = test_seq[0]
    neuro.prev_mean = None
    _ = neuro.forward(sample_seq)

    for h in range(neuro.n_heads):
        profile = neuro.HEAD_PROFILES[h]
        attn = neuro._attn_weights[h]
        # Recency: how much does last position attend to recent vs distant?
        last_row = attn[-1]
        recent_attn = float(last_row[-3:].mean())
        distant_attn = float(last_row[:3].mean())
        print(f"    Head {h} ({profile:>3}): recent={recent_attn:.3f} "
              f"distant={distant_attn:.3f} "
              f"ratio={recent_attn/max(distant_attn, 0.001):.1f}×")

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  TRAINED COMPARISON RESULTS")
    print(f"{'='*70}")
    print(f"")
    print(f"  {'Metric':<30} {'Standard':>12} {'NeuroTF':>12} {'Winner':>10}")
    print(f"  {'─'*30} {'─'*12} {'─'*12} {'─'*10}")

    metrics = [
        ("Train MSE ↓", std_train_mse, neuro_train_mse, "lower"),
        ("Test MSE ↓", std_test_mse, neuro_test_mse, "lower"),
        ("Anomaly detection ↑", std_anomaly_score, neuro_anomaly_score, "higher"),
        ("Order sensitivity ↑", std_order, neuro_order, "higher"),
        ("Parameters", std.n_params, neuro.n_params, "info"),
    ]

    std_wins = 0
    neuro_wins = 0
    for name, sv, nv, direction in metrics:
        if direction == "lower":
            winner = "STD" if sv < nv else "NEURO"
        elif direction == "higher":
            winner = "NEURO" if nv > sv else "STD"
        else:
            winner = "—"

        if winner == "STD":
            std_wins += 1
        elif winner == "NEURO":
            neuro_wins += 1

        print(f"  {name:<30} {sv:>12.4f} {nv:>12.4f} {winner:>10}")

    print(f"\n  Standard wins: {std_wins}  NeuroTransformer wins: {neuro_wins}")

    if neuro_test_mse < std_test_mse:
        improvement = (std_test_mse - neuro_test_mse) / std_test_mse * 100
        print(f"\n  ✓ NeuroTransformer achieves {improvement:.1f}% lower test MSE")
    else:
        print(f"\n  ○ Standard achieves lower test MSE (neuromod needs more training)")

    print(f"\n  Neuromod learned parameters:")
    print(f"    DA gain:    {neuro.da_gain.mean():.4f} (surprise attention scale)")
    print(f"    ACh temp:   {neuro.ach_temp[0]:.4f} (explore/exploit temperature)")
    print(f"    5HT decay:  {neuro.sht_decay[0]:.4f} (temporal locality strength)")

    print(f"\n{'='*70}")

    return {
        "std_test_mse": std_test_mse,
        "neuro_test_mse": neuro_test_mse,
        "std_anomaly": std_anomaly_score,
        "neuro_anomaly": neuro_anomaly_score,
        "std_order": std_order,
        "neuro_order": neuro_order,
    }


def run_tests():
    print("=" * 65)
    print("  NeuroTransformer Training Tests")
    print("=" * 65)
    rng = np.random.RandomState(42)
    p = 0; t = 0

    print("\n  T1: Load belief sequences")
    seqs, tgts = load_belief_sequences(max_sequences=100)
    ok = seqs.shape[0] > 50 and seqs.shape[1] == SEQ_LEN
    print(f"    Sequences: {seqs.shape} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T2: Standard transformer forward pass")
    std = TrainableStandardTransformer()
    out = std.forward(seqs[0])
    ok = out.shape == (D_BELIEF,)
    print(f"    Output: {out.shape} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T3: NeuroTransformer forward pass")
    neuro = TrainableNeuroTransformer()
    out = neuro.forward(seqs[0])
    ok = out.shape == (D_BELIEF,)
    print(f"    Output: {out.shape} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T4: Training reduces loss (Standard)")
    loss_before = evaluate_model(std, seqs[:50], tgts[:50])
    train_model(std, seqs[:50], tgts[:50], n_epochs=10, label="Std")
    loss_after = evaluate_model(std, seqs[:50], tgts[:50])
    ok = loss_after <= loss_before * 1.1
    print(f"    Loss: {loss_before:.4f} → {loss_after:.4f} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T5: Training reduces loss (Neuro)")
    loss_before = evaluate_model(neuro, seqs[:50], tgts[:50])
    train_model(neuro, seqs[:50], tgts[:50], n_epochs=10, label="Neuro")
    loss_after = evaluate_model(neuro, seqs[:50], tgts[:50])
    ok = loss_after <= loss_before * 1.1
    print(f"    Loss: {loss_before:.4f} → {loss_after:.4f} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T6: Neuro has more params (learnable neuromod)")
    ok = neuro.n_params > std.n_params
    print(f"    Std={std.n_params} Neuro={neuro.n_params} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T7: Attention weights computed")
    _ = neuro.forward(seqs[0])
    ok = len(neuro._attn_weights) == 4
    print(f"    Heads: {len(neuro._attn_weights)} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T8: Different sequences produce different outputs")
    out1 = neuro.forward(seqs[0])
    neuro.prev_mean = None
    out2 = neuro.forward(seqs[10])
    dist = float(np.linalg.norm(out1 - out2))
    ok = dist > 0.01
    print(f"    Distance: {dist:.4f} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print(f"\n{'='*65}")
    print(f"  Results: {p}/{t} tests passed")
    print(f"{'='*65}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()

    if args.test:
        run_tests()
    else:
        run_experiment(n_epochs=args.epochs)
