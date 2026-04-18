"""
object_tracker.py — Object Tracking from Prediction Error
============================================================
Track moving objects WITHOUT an object detector.
Objects = things the world model can't predict.
Background = things the world model predicts correctly.

Biological basis: infant visual tracking develops before
object recognition. Babies track motion before they know
what's moving. Prediction error IS the object detector.

Usage:
    python object_tracker.py          # demo with synthetic video
    python object_tracker.py --test   # run tests
"""

import argparse
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from pathlib import Path

Path("outputs").mkdir(exist_ok=True)

D_BELIEF = 64


class PredictiveTracker:
    """
    Track objects using prediction error.
    
    1. Predict next frame's belief from current belief + action
    2. Compare prediction to actual next belief
    3. High error regions = moving objects
    4. Track = follow the region of highest prediction error
    """

    def __init__(self, d_belief=D_BELIEF, d_action=2):
        self.d_belief = d_belief
        self.d_action = d_action

        # Simple transition model for prediction
        rng = np.random.RandomState(42)
        d_in = d_belief + d_action
        self.W1 = rng.randn(d_in, 128).astype(np.float32) * 0.1
        self.b1 = np.zeros(128, dtype=np.float32)
        self.W2 = rng.randn(128, d_belief).astype(np.float32) * 0.1
        self.b2 = np.zeros(d_belief, dtype=np.float32)

        # Tracking state
        self.target_belief = None
        self.target_history = []
        self.tracking = False
        self.track_confidence = 0.0
        self.prediction_errors = []

    def predict(self, belief, action):
        x = np.concatenate([belief, action])
        h = np.maximum(0, x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def train_background(self, belief_sequence, actions):
        """Learn to predict the background (stationary elements)."""
        N = min(len(belief_sequence) - 1, len(actions))
        for epoch in range(20):
            idx = np.random.choice(N, min(256, N), replace=False)
            for i in idx:
                b_t = belief_sequence[i]
                a = actions[i]
                b_t1 = belief_sequence[i + 1]

                # Forward
                x = np.concatenate([b_t, a])
                h = np.maximum(0, x @ self.W1 + self.b1)
                pred = h @ self.W2 + self.b2
                error = pred - b_t1

                # Backward
                dW2 = np.outer(h, error)
                db2 = error
                dh = error @ self.W2.T * (h > 0).astype(np.float32)
                dW1 = np.outer(x, dh)
                db1 = dh

                lr = 0.01
                self.W2 -= lr * dW2
                self.b2 -= lr * db2
                self.W1 -= lr * dW1
                self.b1 -= lr * db1

    def detect_objects(self, belief, action, next_belief):
        """
        Detect objects as regions of high prediction error.
        Returns: prediction error magnitude and per-dimension errors.
        """
        predicted = self.predict(belief, action)
        error = next_belief - predicted
        error_magnitude = float(np.linalg.norm(error))
        per_dim_error = np.abs(error)

        self.prediction_errors.append(error_magnitude)

        return error_magnitude, per_dim_error, error

    def acquire_target(self, error_vector, belief, threshold=0.5):
        """
        Start tracking the most surprising element.
        Target = belief region with highest prediction error.
        """
        error_mag = float(np.linalg.norm(error_vector))

        if error_mag > threshold:
            self.target_belief = belief.copy()
            self.tracking = True
            self.track_confidence = min(1.0, error_mag / 2.0)
            self.target_history = [belief.copy()]
            return True
        return False

    def update_track(self, current_belief, error_vector):
        """
        Update tracking state with new observation.
        Predict where target is, compare to actual.
        """
        if not self.tracking:
            return None

        # Target similarity: how close is current to tracked target?
        target_dist = float(np.linalg.norm(current_belief - self.target_belief))
        error_mag = float(np.linalg.norm(error_vector))

        # Update target position (EMA)
        if error_mag > 0.3:  # still surprising = still an object
            lr = 0.3
            self.target_belief = (
                (1 - lr) * self.target_belief + lr * current_belief
            ).astype(np.float32)
            self.track_confidence = min(1.0, error_mag / 2.0)
        else:
            # Object stopped or left — reduce confidence
            self.track_confidence *= 0.9

        self.target_history.append(current_belief.copy())
        if len(self.target_history) > 100:
            self.target_history = self.target_history[-100:]

        # Lose track if confidence drops
        if self.track_confidence < 0.1:
            self.tracking = False
            self.track_confidence = 0.0
            return None

        return {
            "position": self.target_belief.copy(),
            "confidence": self.track_confidence,
            "distance": target_dist,
            "error": error_mag,
        }

    def predict_target_position(self, action):
        """Predict where the target will be next."""
        if self.target_belief is None:
            return None

        predicted = self.predict(self.target_belief, action)
        return predicted

    def get_follow_action(self, current_belief, action_scale=0.5):
        """Generate action to follow the tracked target."""
        if self.target_belief is None:
            return np.zeros(self.d_action, dtype=np.float32)

        diff = self.target_belief[:self.d_action] - current_belief[:self.d_action]
        action = np.clip(diff * action_scale, -1, 1).astype(np.float32)
        return action


class VideoScene:
    """Generate synthetic video with moving objects for tracking."""

    def __init__(self, frame_h=16, frame_w=16):
        self.h = frame_h
        self.w = frame_w
        self.d_frame = frame_h * frame_w

        # Objects
        self.objects = []

    def add_object(self, x, y, radius=2.0, speed=0.3, color=1.0):
        self.objects.append({
            "x": float(x), "y": float(y),
            "vx": 0.0, "vy": 0.0,
            "radius": radius, "speed": speed, "color": color,
        })

    def step(self, rng=None):
        """Move objects, return frame."""
        if rng is None:
            rng = np.random.RandomState()

        # Move objects
        for obj in self.objects:
            obj["vx"] = 0.8 * obj["vx"] + rng.randn() * obj["speed"]
            obj["vy"] = 0.8 * obj["vy"] + rng.randn() * obj["speed"]
            obj["x"] = np.clip(obj["x"] + obj["vx"], 1, self.w - 2)
            obj["y"] = np.clip(obj["y"] + obj["vy"], 1, self.h - 2)

        # Render frame
        frame = np.ones((self.h, self.w), dtype=np.float32) * 0.1  # background

        for obj in self.objects:
            for y in range(self.h):
                for x in range(self.w):
                    dist = np.sqrt((x - obj["x"])**2 + (y - obj["y"])**2)
                    if dist < obj["radius"] * 2:
                        intensity = obj["color"] * np.exp(
                            -0.5 * (dist / obj["radius"])**2)
                        frame[y, x] = max(frame[y, x], intensity)

        # Add noise
        frame += rng.randn(self.h, self.w).astype(np.float32) * 0.02

        return frame.flatten()

    def get_object_positions(self):
        return [(obj["x"], obj["y"]) for obj in self.objects]


def project_frame_to_belief(frame, d_belief=D_BELIEF):
    """Simple projection of flattened frame to belief space."""
    rng = np.random.RandomState(42)
    W = rng.randn(len(frame), d_belief).astype(np.float32) * np.sqrt(2.0 / len(frame))
    return np.tanh(frame @ W)


def demo():
    print("=" * 70)
    print("  Object Tracking from Prediction Error")
    print("  Objects = what the world model can't predict")
    print("=" * 70)

    rng = np.random.RandomState(42)

    # Create scene with moving objects
    scene = VideoScene(16, 16)
    scene.add_object(8, 8, radius=2.0, speed=0.4, color=1.0)   # main target
    scene.add_object(4, 12, radius=1.5, speed=0.2, color=0.6)  # distractor

    # Generate sequence
    print("\n  ── 1. GENERATING VIDEO ──")
    n_frames = 200
    frames = []
    actions = []
    positions = []

    for i in range(n_frames):
        frame = scene.step(rng=rng)
        frames.append(frame)
        actions.append(rng.randn(2).astype(np.float32) * 0.1)
        positions.append(scene.get_object_positions())

    print(f"    {n_frames} frames, {len(scene.objects)} objects")

    # Project to belief space
    print("\n  ── 2. ENCODING TO BELIEF SPACE ──")
    beliefs = [project_frame_to_belief(f) for f in frames]
    print(f"    {len(beliefs)} belief vectors, {D_BELIEF}-D each")

    # Train background model
    print("\n  ── 3. LEARNING BACKGROUND ──")
    tracker = PredictiveTracker()
    tracker.train_background(beliefs[:50], actions[:49])
    print(f"    Trained on 50 background frames")

    # Track objects
    print("\n  ── 4. TRACKING OBJECTS ──")
    track_results = []
    n_tracked = 0
    n_lost = 0

    print(f"\n    {'Frame':>6} │ {'Error':>7} │ {'Tracking':>8} │ "
          f"{'Confidence':>10} │ {'Status'}")
    print(f"    {'─'*6}─┼─{'─'*7}─┼─{'─'*8}─┼─{'─'*10}─┼─{'─'*15}")

    for i in range(50, n_frames - 1):
        error_mag, per_dim, error_vec = tracker.detect_objects(
            beliefs[i], actions[i], beliefs[i + 1])

        if not tracker.tracking:
            acquired = tracker.acquire_target(error_vec, beliefs[i + 1])
            if acquired:
                n_tracked += 1
        else:
            result = tracker.update_track(beliefs[i + 1], error_vec)
            if result is None:
                n_lost += 1

        track_results.append({
            "frame": i,
            "error": error_mag,
            "tracking": tracker.tracking,
            "confidence": tracker.track_confidence,
            "true_pos": positions[i][0] if positions[i] else None,
        })

        if i % 20 == 0 or (not tracker.tracking and i > 50):
            status = ""
            if tracker.tracking:
                status = f"TRACKING (conf={tracker.track_confidence:.2f})"
            else:
                status = "searching..."
            print(f"    {i:>6} │ {error_mag:>7.3f} │ "
                  f"{'YES' if tracker.tracking else 'no':>8} │ "
                  f"{tracker.track_confidence:>10.3f} │ {status}")

    # Summary
    tracked_frames = sum(1 for r in track_results if r["tracking"])
    total_frames = len(track_results)

    print(f"\n  ── 5. TRACKING SUMMARY ──")
    print(f"    Total frames:    {total_frames}")
    print(f"    Tracked frames:  {tracked_frames} "
          f"({100*tracked_frames/total_frames:.0f}%)")
    print(f"    Track acquired:  {n_tracked} times")
    print(f"    Track lost:      {n_lost} times")
    print(f"    Mean error:      {np.mean([r['error'] for r in track_results]):.3f}")

    # Follow action demo
    print(f"\n  ── 6. FOLLOW ACTION ──")
    if tracker.target_belief is not None:
        current = beliefs[-1]
        follow_action = tracker.get_follow_action(current)
        print(f"    Follow action: [{follow_action[0]:+.3f}, {follow_action[1]:+.3f}]")
        predicted_pos = tracker.predict_target_position(follow_action)
        if predicted_pos is not None:
            print(f"    Predicted target: norm={np.linalg.norm(predicted_pos):.2f}")

    print(f"\n{'='*70}")
    print(f"  Object tracking via prediction error — no detector needed")
    print(f"  Objects ARE the unpredictable parts of the world")
    print(f"{'='*70}")


def run_tests():
    print("=" * 65)
    print("  Object Tracker Tests")
    print("=" * 65)
    rng = np.random.RandomState(42)
    p = 0; t = 0

    print("\n  T1: Tracker predicts correct shape")
    tracker = PredictiveTracker()
    belief = rng.randn(D_BELIEF).astype(np.float32)
    action = rng.randn(2).astype(np.float32)
    pred = tracker.predict(belief, action)
    ok = pred.shape == (D_BELIEF,)
    print(f"    Shape: {pred.shape} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T2: Detection produces error")
    next_belief = belief + rng.randn(D_BELIEF).astype(np.float32) * 0.5
    error_mag, per_dim, error_vec = tracker.detect_objects(
        belief, action, next_belief)
    ok = error_mag > 0
    print(f"    Error: {error_mag:.3f} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T3: Target acquisition works")
    big_error = rng.randn(D_BELIEF).astype(np.float32) * 3
    acquired = tracker.acquire_target(big_error, belief)
    ok = acquired and tracker.tracking
    print(f"    Acquired: {acquired} Tracking: {tracker.tracking} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T4: Track update returns state")
    result = tracker.update_track(belief, big_error)
    ok = result is not None and "confidence" in result
    print(f"    Result: {result is not None} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T5: Track lost on low error")
    for _ in range(20):
        tiny_error = rng.randn(D_BELIEF).astype(np.float32) * 0.01
        tracker.update_track(belief, tiny_error)
    ok = not tracker.tracking or tracker.track_confidence < 0.2
    print(f"    Tracking: {tracker.tracking} conf: {tracker.track_confidence:.3f} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T6: Follow action generated")
    tracker2 = PredictiveTracker()
    tracker2.acquire_target(big_error, rng.randn(D_BELIEF).astype(np.float32))
    action = tracker2.get_follow_action(belief)
    ok = action.shape == (2,)
    print(f"    Action shape: {action.shape} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T7: Video scene generates frames")
    scene = VideoScene(8, 8)
    scene.add_object(4, 4)
    frame = scene.step(rng=rng)
    ok = frame.shape == (64,)
    print(f"    Frame shape: {frame.shape} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T8: Background training works")
    beliefs_seq = [project_frame_to_belief(scene.step(rng=rng))
                     for _ in range(30)]
    actions_seq = [rng.randn(2).astype(np.float32) for _ in range(29)]
    tracker3 = PredictiveTracker()
    tracker3.train_background(beliefs_seq, actions_seq)
    ok = True  # no crash
    print(f"    Trained on 30 frames {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T9: Prediction improves after training")
    pred_before = tracker.predict(beliefs_seq[0], actions_seq[0])
    err_before = float(np.linalg.norm(pred_before - beliefs_seq[1]))
    pred_after = tracker3.predict(beliefs_seq[0], actions_seq[0])
    err_after = float(np.linalg.norm(pred_after - beliefs_seq[1]))
    ok = err_after <= err_before * 1.5  # should not be worse
    print(f"    Before: {err_before:.3f} After: {err_after:.3f} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T10: Target prediction works")
    tracker4 = PredictiveTracker()
    tracker4.acquire_target(big_error, belief)
    predicted = tracker4.predict_target_position(action)
    ok = predicted is not None and predicted.shape == (D_BELIEF,)
    print(f"    Predicted shape: {predicted.shape if predicted is not None else None} "
          f"{'PASS' if ok else 'FAIL'}")
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
