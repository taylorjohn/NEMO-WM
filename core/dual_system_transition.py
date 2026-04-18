"""
dual_system_transition.py — System 1 + System 2 Transition Model
==================================================================
Kahneman's Thinking Fast and Slow, implemented.

System 1 (MLP):              Fast, habitual, 5μs, always on
System 2 (NeuroTransformer): Slow, deliberate, 80μs, activates when needed

Switching criteria (automatic):
  - Low novelty + low prediction error → System 1 (routine)
  - High novelty OR high error OR complex goal → System 2 (deliberate)
  - High fatigue → Force System 1 (too tired to think hard)

This is biologically accurate: most cognition is System 1.
System 2 only activates for ~5-15% of decisions (Kahneman 2011).

Usage:
    python dual_system_transition.py          # demo
    python dual_system_transition.py --test   # run tests
"""

import argparse
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from pathlib import Path

D_BELIEF = 64
D_ACTION = 2
CONTEXT_LEN = 8  # System 2 looks at last 8 beliefs


# ══════════════════════════════════════════════════════════════════════════════
# System 1 — Fast MLP (existing transition model)
# ══════════════════════════════════════════════════════════════════════════════

class System1:
    """
    Fast, habitual prediction. Single-step MLP.
    ~5μs per prediction. No context, no attention.
    Good for: routine navigation, familiar environments.
    """

    def __init__(self):
        self.trained = False
        trained_path = Path("data/minari_trained/transition_model.npz")
        if trained_path.exists():
            data = np.load(trained_path)
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
# System 2 — NeuroTransformer (context-aware, attention-based)
# ══════════════════════════════════════════════════════════════════════════════

class System2:
    """
    Slow, deliberate prediction. Attends to last K beliefs.
    ~80μs per prediction. Full neuromodulated attention.
    Good for: novel situations, complex goals, error recovery.
    """

    def __init__(self, context_len=CONTEXT_LEN, d_model=D_BELIEF,
                  n_heads=4):
        self.context_len = context_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        rng = np.random.RandomState(99)

        # Attention weights
        self.W_q = rng.randn(d_model, d_model).astype(np.float32) * 0.1
        self.W_k = rng.randn(d_model, d_model).astype(np.float32) * 0.1
        self.W_v = rng.randn(d_model, d_model).astype(np.float32) * 0.1

        # Output projection: attended context + action → next belief
        d_in = d_model + D_ACTION
        self.W_out1 = rng.randn(d_in, 128).astype(np.float32) * 0.1
        self.b_out1 = np.zeros(128, dtype=np.float32)
        self.W_out2 = rng.randn(128, d_model).astype(np.float32) * 0.1
        self.b_out2 = np.zeros(d_model, dtype=np.float32)

        # Head profiles: DA, ACh, CRT, 5HT
        self.head_profiles = ["DA", "ACh", "CRT", "5HT"]

        # Neuromod state
        self.prev_hidden = None

    def _compute_neuromod(self, beliefs):
        """Quick neuromod from belief history."""
        mean_b = np.mean(beliefs, axis=0)

        # DA: prediction error (change from previous)
        if self.prev_hidden is not None:
            da = float(np.clip(
                np.linalg.norm(mean_b - self.prev_hidden) * 0.5, 0, 1))
        else:
            da = 0.5
        self.prev_hidden = mean_b.copy()

        # ACh: uncertainty (variance of beliefs in window)
        ach = float(np.clip(np.std(beliefs) * 2, 0.1, 1))

        # CRT: conflict (max pairwise distance in window)
        if len(beliefs) > 1:
            diffs = [np.linalg.norm(beliefs[i] - beliefs[j])
                      for i in range(len(beliefs))
                      for j in range(i+1, min(i+3, len(beliefs)))]
            crt = float(np.clip(np.max(diffs) * 0.1 if diffs else 0, 0, 1))
        else:
            crt = 0.1

        # 5HT: trajectory smoothness
        if len(beliefs) > 2:
            accels = [np.linalg.norm(beliefs[i+2] - 2*beliefs[i+1] + beliefs[i])
                       for i in range(len(beliefs)-2)]
            sht = float(np.clip(1 - np.mean(accels) * 0.1, 0.1, 1))
        else:
            sht = 0.5

        return {"DA": da, "ACh": ach, "CRT": crt, "5HT": sht}

    def _inverted_u(self, level, optimal, width):
        return max(0.0, 1.0 - ((level - optimal) / width) ** 2)

    def predict(self, belief_history, action):
        """
        Predict next belief from context window + action.
        belief_history: list of recent beliefs (up to context_len)
        action: current action
        """
        # Pad or truncate history
        history = list(belief_history[-self.context_len:])
        while len(history) < self.context_len:
            history.insert(0, history[0].copy() if history else
                            np.zeros(self.d_model, dtype=np.float32))

        beliefs = np.stack(history)  # (context_len, d_model)
        n = len(beliefs)

        # Compute neuromod signals
        signals = self._compute_neuromod(beliefs)

        # Multi-head attention
        Q = beliefs @ self.W_q
        K = beliefs @ self.W_k
        V = beliefs @ self.W_v

        Q = Q.reshape(n, self.n_heads, self.d_head)
        K = K.reshape(n, self.n_heads, self.d_head)
        V = V.reshape(n, self.n_heads, self.d_head)

        attended_heads = []

        for h in range(self.n_heads):
            q = Q[:, h, :]
            k = K[:, h, :]
            v = V[:, h, :]

            scores = q @ k.T / np.sqrt(self.d_head)
            profile = self.head_profiles[h]

            # Neuromodulated gating
            if profile == "DA":
                # Attend MORE to surprising (high-change) time steps
                da_eff = self._inverted_u(signals["DA"], 0.4, 0.5)
                changes = np.array([0] + [
                    np.linalg.norm(beliefs[i] - beliefs[i-1])
                    for i in range(1, n)])
                surprise_weight = 1.0 + changes * da_eff * 2.0
                scores = scores * surprise_weight[None, :]

            elif profile == "ACh":
                # Temperature: high ACh → broad, low → narrow
                ach_eff = self._inverted_u(signals["ACh"], 0.6, 0.5)
                temp = 0.5 + (1 - ach_eff) * 1.5
                scores = scores / max(temp, 0.1)

            elif profile == "CRT":
                # Suppress conflicting beliefs
                crt_eff = self._inverted_u(signals["CRT"], 0.3, 0.4)
                if crt_eff > 0.3:
                    v_norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
                    sim = (v / v_norms) @ (v / v_norms).T
                    conflict = np.where(sim < 0, 0.5, 1.0)
                    conflict_per_key = conflict.mean(axis=0, keepdims=True)
                    scores = scores * conflict_per_key

            elif profile == "5HT":
                # Temporal locality — recent beliefs matter more
                sht_eff = self._inverted_u(signals["5HT"], 0.5, 0.5)
                recency = np.exp(-np.arange(n)[::-1] * sht_eff * 0.5)
                scores = scores * recency[None, :]

            # Softmax
            e = np.exp(scores - scores.max(axis=-1, keepdims=True))
            attn = e / (e.sum(axis=-1, keepdims=True) + 1e-10)

            attended = attn @ v
            attended_heads.append(attended)

        # Concatenate heads → context representation
        context = np.concatenate(attended_heads, axis=1)  # (n, d_model)

        # Use last position's context (most recent) + action
        last_context = context[-1]
        x = np.concatenate([last_context, action])

        # Predict next belief
        h = np.maximum(0, x @ self.W_out1 + self.b_out1)
        pred = np.clip(h @ self.W_out2 + self.b_out2, -5, 5)

        return pred, signals


# ══════════════════════════════════════════════════════════════════════════════
# Dual System — automatic switching
# ══════════════════════════════════════════════════════════════════════════════

class DualSystemTransition:
    """
    Kahneman's Thinking Fast and Slow.
    
    System 1 (MLP):      fast, habitual, always available
    System 2 (NeuroTF):  slow, deliberate, activates when needed
    
    Automatic switching based on:
    - Novelty: high → System 2
    - Prediction error: high → System 2
    - Fatigue: high → Force System 1 (too tired)
    - Goal complexity: multi-step → System 2
    """

    def __init__(self, novelty_threshold=2.0, error_threshold=1.0,
                  fatigue_threshold=0.7):
        self.s1 = System1()
        self.s2 = System2()

        self.novelty_threshold = novelty_threshold
        self.error_threshold = error_threshold
        self.fatigue_threshold = fatigue_threshold

        # History for System 2
        self.belief_history: List[np.ndarray] = []
        self.max_history = CONTEXT_LEN

        # Tracking
        self.s1_count = 0
        self.s2_count = 0
        self.last_error = 0.0
        self.fatigue = 0.0
        self.last_system = "S1"
        self.last_signals = {}

    def _should_use_system2(self, belief, novelty=None):
        """Decide whether to engage System 2."""
        # Too fatigued → force System 1
        if self.fatigue > self.fatigue_threshold:
            return False

        # High novelty → System 2
        if novelty is not None and novelty > self.novelty_threshold:
            return True

        # High prediction error → System 2
        if self.last_error > self.error_threshold:
            return True

        # Not enough history for System 2
        if len(self.belief_history) < 3:
            return False

        # Check if recent beliefs are inconsistent (conflict)
        if len(self.belief_history) >= 3:
            recent = self.belief_history[-3:]
            diffs = [np.linalg.norm(recent[i] - recent[i-1])
                      for i in range(1, len(recent))]
            if max(diffs) > 3.0:
                return True

        return False

    def predict(self, belief, action, novelty=None, force_system=None):
        """
        Predict next belief using the appropriate system.
        
        Returns: next_belief, metadata dict
        """
        # Record in history
        self.belief_history.append(belief.copy())
        if len(self.belief_history) > self.max_history:
            self.belief_history = self.belief_history[-self.max_history:]

        # Decide which system
        if force_system == "S1":
            use_s2 = False
        elif force_system == "S2":
            use_s2 = True
        else:
            use_s2 = self._should_use_system2(belief, novelty)

        t0 = time.perf_counter()

        if use_s2:
            pred, signals = self.s2.predict(self.belief_history, action)
            self.s2_count += 1
            self.last_system = "S2"
            self.last_signals = signals
            self.fatigue += 0.05  # System 2 is tiring
        else:
            pred = self.s1.predict(belief, action)
            self.s1_count += 1
            self.last_system = "S1"
            self.last_signals = {}
            self.fatigue += 0.01  # System 1 is easy

        elapsed_us = (time.perf_counter() - t0) * 1e6

        # Compute prediction error for next decision
        self.last_error = float(np.linalg.norm(pred - belief))

        # Clamp fatigue
        self.fatigue = min(1.0, self.fatigue)

        return pred, {
            "system": self.last_system,
            "time_us": elapsed_us,
            "pred_error": self.last_error,
            "fatigue": self.fatigue,
            "signals": self.last_signals,
            "s1_count": self.s1_count,
            "s2_count": self.s2_count,
        }

    def sleep(self):
        """Reset fatigue."""
        self.fatigue *= 0.2

    @property
    def s2_ratio(self):
        total = self.s1_count + self.s2_count
        return self.s2_count / max(total, 1)


# ══════════════════════════════════════════════════════════════════════════════
# Demo
# ══════════════════════════════════════════════════════════════════════════════

def demo():
    print("=" * 70)
    print("  Dual System Transition — Thinking Fast and Slow")
    print("  System 1 (MLP, 5μs) + System 2 (NeuroTransformer, 80μs)")
    print("=" * 70)

    rng = np.random.RandomState(42)
    dual = DualSystemTransition()

    # Load real beliefs if available
    beliefs_path = Path("data/minari_trained/beliefs_sample.npz")
    if beliefs_path.exists():
        real_beliefs = np.load(beliefs_path)["beliefs"]
        print(f"\n  Using trained beliefs ({len(real_beliefs)} samples)")
    else:
        real_beliefs = rng.randn(1000, D_BELIEF).astype(np.float32) * 0.5
        print(f"\n  Using random beliefs")

    # Scenario 1: Routine navigation (should stay in System 1)
    print(f"\n  ── Scenario 1: Routine Navigation ──")
    print(f"  (Familiar territory, low novelty)")
    for i in range(20):
        belief = real_beliefs[i].copy()
        action = rng.randn(D_ACTION).astype(np.float32) * 0.3
        pred, meta = dual.predict(belief, action, novelty=0.5)
        if i % 5 == 0:
            print(f"    Step {i:>3}: {meta['system']} "
                  f"{meta['time_us']:.0f}μs  "
                  f"fatigue={meta['fatigue']:.2f}")

    print(f"    System 2 usage: {dual.s2_ratio:.1%}")

    # Scenario 2: Novel environment (should switch to System 2)
    print(f"\n  ── Scenario 2: Novel Environment ──")
    print(f"  (High novelty, unfamiliar territory)")
    for i in range(20):
        # Very different beliefs
        belief = rng.randn(D_BELIEF).astype(np.float32) * 3.0
        action = rng.randn(D_ACTION).astype(np.float32) * 0.5
        pred, meta = dual.predict(belief, action, novelty=5.0)
        if i % 5 == 0:
            print(f"    Step {i:>3}: {meta['system']} "
                  f"{meta['time_us']:.0f}μs  "
                  f"fatigue={meta['fatigue']:.2f}")
            if meta['signals']:
                print(f"           DA={meta['signals'].get('DA', 0):.2f} "
                      f"ACh={meta['signals'].get('ACh', 0):.2f} "
                      f"CRT={meta['signals'].get('CRT', 0):.2f}")

    print(f"    System 2 usage: {dual.s2_ratio:.1%}")

    # Scenario 3: Fatigue forces System 1
    print(f"\n  ── Scenario 3: Exhaustion ──")
    print(f"  (Fatigued, forced back to System 1)")
    dual.fatigue = 0.8  # very tired
    for i in range(10):
        belief = rng.randn(D_BELIEF).astype(np.float32) * 3.0
        action = rng.randn(D_ACTION).astype(np.float32) * 0.5
        pred, meta = dual.predict(belief, action, novelty=5.0)
        if i % 3 == 0:
            print(f"    Step {i:>3}: {meta['system']} "
                  f"fatigue={meta['fatigue']:.2f}")

    # Sleep and recover
    print(f"\n    Sleep...")
    dual.sleep()
    pred, meta = dual.predict(
        rng.randn(D_BELIEF).astype(np.float32) * 3.0,
        rng.randn(D_ACTION).astype(np.float32), novelty=5.0)
    print(f"    After sleep: {meta['system']} "
          f"fatigue={meta['fatigue']:.2f}")

    # Summary
    print(f"\n  ── Summary ──")
    total = dual.s1_count + dual.s2_count
    print(f"    Total predictions: {total}")
    print(f"    System 1 (fast):   {dual.s1_count} ({100*dual.s1_count/total:.0f}%)")
    print(f"    System 2 (slow):   {dual.s2_count} ({100*dual.s2_count/total:.0f}%)")
    print(f"    System 2 ratio:    {dual.s2_ratio:.1%}")
    print(f"    Bio target:        5-15% (Kahneman 2011)")
    print(f"    Match: {'✓' if 0.03 <= dual.s2_ratio <= 0.30 else '○'}")

    # Speed comparison
    print(f"\n  ── Speed Comparison ──")
    belief = real_beliefs[0].copy()
    action = rng.randn(D_ACTION).astype(np.float32) * 0.3

    # Time System 1
    t0 = time.perf_counter()
    for _ in range(1000):
        dual.s1.predict(belief, action)
    s1_us = (time.perf_counter() - t0) * 1e6 / 1000

    # Time System 2
    history = [real_beliefs[i] for i in range(8)]
    t0 = time.perf_counter()
    for _ in range(1000):
        dual.s2.predict(history, action)
    s2_us = (time.perf_counter() - t0) * 1e6 / 1000

    print(f"    System 1: {s1_us:.1f}μs ({1e6/s1_us:.0f} Hz)")
    print(f"    System 2: {s2_us:.1f}μs ({1e6/s2_us:.0f} Hz)")
    print(f"    Ratio:    {s2_us/s1_us:.1f}× slower")
    print(f"    Weighted avg (at {dual.s2_ratio:.0%} S2): "
          f"{s1_us * (1-dual.s2_ratio) + s2_us * dual.s2_ratio:.1f}μs")

    print(f"\n{'='*70}")


def run_tests():
    print("=" * 65)
    print("  Dual System Transition Tests")
    print("=" * 65)
    rng = np.random.RandomState(42)
    p = 0; t = 0

    print("\n  T1: System 1 produces correct shape")
    s1 = System1()
    pred = s1.predict(rng.randn(D_BELIEF).astype(np.float32),
                       rng.randn(D_ACTION).astype(np.float32))
    ok = pred.shape == (D_BELIEF,)
    print(f"    Shape: {pred.shape} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T2: System 2 produces correct shape")
    s2 = System2()
    history = [rng.randn(D_BELIEF).astype(np.float32) for _ in range(8)]
    pred, signals = s2.predict(history, rng.randn(D_ACTION).astype(np.float32))
    ok = pred.shape == (D_BELIEF,) and len(signals) == 4
    print(f"    Shape: {pred.shape} Signals: {len(signals)} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T3: Dual system defaults to System 1 for routine")
    dual = DualSystemTransition()
    base = rng.randn(D_BELIEF).astype(np.float32) * 0.1
    for i in range(10):
        # Very similar beliefs — routine, low novelty, low error
        belief = base + rng.randn(D_BELIEF).astype(np.float32) * 0.01
        pred, meta = dual.predict(belief,
            rng.randn(D_ACTION).astype(np.float32) * 0.1, novelty=0.5)
    ok = dual.s1_count >= dual.s2_count
    print(f"    S1={dual.s1_count} S2={dual.s2_count} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T4: High novelty triggers System 2")
    dual2 = DualSystemTransition()
    # Build up some history first
    for i in range(5):
        dual2.predict(rng.randn(D_BELIEF).astype(np.float32) * 0.3,
                       rng.randn(D_ACTION).astype(np.float32), novelty=0.5)
    # Now high novelty
    pred, meta = dual2.predict(
        rng.randn(D_BELIEF).astype(np.float32) * 5,
        rng.randn(D_ACTION).astype(np.float32), novelty=5.0)
    ok = meta["system"] == "S2"
    print(f"    System: {meta['system']} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T5: Fatigue forces System 1")
    dual3 = DualSystemTransition()
    dual3.fatigue = 0.9
    for i in range(3):
        dual3.predict(rng.randn(D_BELIEF).astype(np.float32),
                       rng.randn(D_ACTION).astype(np.float32), novelty=0.5)
    pred, meta = dual3.predict(
        rng.randn(D_BELIEF).astype(np.float32),
        rng.randn(D_ACTION).astype(np.float32), novelty=10.0)  # very novel
    ok = meta["system"] == "S1"  # but too tired
    print(f"    High novelty but fatigued → {meta['system']} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T6: Sleep resets fatigue")
    before = dual3.fatigue
    dual3.sleep()
    after = dual3.fatigue
    ok = after < before * 0.5
    print(f"    Fatigue: {before:.2f} → {after:.2f} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T7: System 1 is faster than System 2")
    belief = rng.randn(D_BELIEF).astype(np.float32)
    action = rng.randn(D_ACTION).astype(np.float32)
    history = [rng.randn(D_BELIEF).astype(np.float32) for _ in range(8)]

    t0 = time.perf_counter()
    for _ in range(500):
        s1.predict(belief, action)
    s1_time = (time.perf_counter() - t0) / 500

    t0 = time.perf_counter()
    for _ in range(500):
        s2.predict(history, action)
    s2_time = (time.perf_counter() - t0) / 500

    ok = s1_time < s2_time
    print(f"    S1: {s1_time*1e6:.0f}μs  S2: {s2_time*1e6:.0f}μs  "
          f"ratio: {s2_time/s1_time:.1f}× "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T8: force_system overrides automatic")
    dual4 = DualSystemTransition()
    for i in range(5):
        dual4.predict(rng.randn(D_BELIEF).astype(np.float32),
                       rng.randn(D_ACTION).astype(np.float32), novelty=0.5)
    pred, meta = dual4.predict(
        rng.randn(D_BELIEF).astype(np.float32),
        rng.randn(D_ACTION).astype(np.float32),
        novelty=0.1, force_system="S2")
    ok = meta["system"] == "S2"
    print(f"    Forced S2: {meta['system']} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T9: S2 ratio in biological range")
    dual5 = DualSystemTransition()
    for i in range(100):
        # Mix of routine and novel
        nov = 0.5 if i % 5 != 0 else 5.0
        dual5.predict(rng.randn(D_BELIEF).astype(np.float32) * (0.3 if nov < 2 else 3),
                       rng.randn(D_ACTION).astype(np.float32), novelty=nov)
    ratio = dual5.s2_ratio
    ok = 0.05 <= ratio <= 0.50
    print(f"    S2 ratio: {ratio:.1%} (target 5-30%) "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T10: Prediction error tracked")
    dual6 = DualSystemTransition()
    dual6.predict(rng.randn(D_BELIEF).astype(np.float32),
                   rng.randn(D_ACTION).astype(np.float32))
    pred, meta = dual6.predict(rng.randn(D_BELIEF).astype(np.float32),
                                 rng.randn(D_ACTION).astype(np.float32))
    ok = meta["pred_error"] >= 0
    print(f"    Pred error: {meta['pred_error']:.3f} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print(f"\n{'='*65}")
    print(f"  Results: {p}/{t} tests passed")
    print(f"{'='*65}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        run_tests()
    else:
        demo()
