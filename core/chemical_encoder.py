"""
chemical_encoder.py — Olfactory/Chemical Sensing for NeMo-WM
===============================================================
Projects chemical sensor data into 64-D belief space.
Completes 5/5 senses: vision + proprio + audio + tactile + CHEMICAL.

Chemical modalities:
  - Concentration gradients (follow scent trails)
  - Chemical identity (distinguish substances)  
  - Intensity (how strong is the signal)
  - Temporal change (getting stronger = approaching source)

Grounded words: sweet, bitter, acrid, faint, strong, smoke,
                fresh, stale, chemical, gradient, source, near

Usage:
    python chemical_encoder.py          # demo
    python chemical_encoder.py --test   # run tests
"""

import argparse
import numpy as np
from typing import Dict, List, Optional

D_BELIEF = 64
N_CHEM_DIMS = 16  # 16 chemical receptor types


class ChemicalSensor:
    """Simulate chemical/olfactory sensor array."""

    def __init__(self, n_receptors=N_CHEM_DIMS):
        self.n_receptors = n_receptors

    def generate(self, substance="neutral", concentration=0.5,
                   distance=1.0, rng=None):
        """Generate synthetic chemical reading."""
        if rng is None:
            rng = np.random.RandomState()

        data = np.zeros(self.n_receptors, dtype=np.float32)

        # Each substance activates different receptor patterns
        profiles = {
            "sweet":    [0.9, 0.7, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0,
                          0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.3, 0.0],
            "bitter":   [0.1, 0.0, 0.8, 0.9, 0.3, 0.0, 0.0, 0.0,
                          0.0, 0.1, 0.0, 0.0, 0.4, 0.0, 0.0, 0.1],
            "acrid":    [0.0, 0.0, 0.2, 0.3, 0.9, 0.8, 0.5, 0.0,
                          0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.3],
            "smoke":    [0.0, 0.0, 0.1, 0.4, 0.6, 0.9, 0.7, 0.5,
                          0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
            "fresh":    [0.3, 0.5, 0.0, 0.0, 0.0, 0.0, 0.1, 0.3,
                          0.6, 0.8, 0.4, 0.2, 0.0, 0.0, 0.2, 0.0],
            "chemical": [0.0, 0.0, 0.0, 0.1, 0.3, 0.5, 0.2, 0.0,
                          0.0, 0.0, 0.7, 0.9, 0.8, 0.4, 0.0, 0.1],
            "neutral":  [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                          0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        }

        profile = np.array(profiles.get(substance, profiles["neutral"]),
                             dtype=np.float32)

        # Scale by concentration and distance (inverse square)
        intensity = concentration / (distance ** 2 + 0.1)
        data = profile * intensity

        # Add sensor noise
        data += rng.randn(self.n_receptors).astype(np.float32) * 0.03
        return np.clip(data, 0, 1)


class ChemicalBeliefEncoder:
    """Project chemical sensor data into 64-D belief space."""

    def __init__(self, n_chem=N_CHEM_DIMS, d_belief=D_BELIEF):
        rng = np.random.RandomState(seed=789)
        self.W_proj = rng.randn(n_chem, d_belief).astype(np.float32) * 0.3
        self.b_proj = rng.randn(d_belief).astype(np.float32) * 0.1

    def encode(self, chem_data):
        if chem_data.ndim == 1:
            chem_data = chem_data.reshape(1, -1)
        beliefs = np.tanh(chem_data @ self.W_proj + self.b_proj)
        return beliefs.squeeze() if chem_data.shape[0] == 1 else beliefs


class ChemicalGradientTracker:
    """Track concentration gradients to find source."""

    def __init__(self):
        self.history = []
        self.gradient_direction = None

    def update(self, belief, concentration):
        self.history.append((belief.copy(), concentration))
        if len(self.history) >= 2:
            prev_c = self.history[-2][1]
            curr_c = self.history[-1][1]
            prev_b = self.history[-2][0]
            curr_b = self.history[-1][0]
            if curr_c > prev_c:
                # Getting stronger — direction is good
                self.gradient_direction = curr_b - prev_b
            else:
                # Getting weaker — reverse
                self.gradient_direction = prev_b - curr_b

        if len(self.history) > 50:
            self.history = self.history[-50:]

    def get_follow_direction(self):
        if self.gradient_direction is not None:
            norm = np.linalg.norm(self.gradient_direction)
            if norm > 1e-6:
                return self.gradient_direction / norm
        return None

    def is_approaching(self):
        if len(self.history) >= 3:
            recent = [c for _, c in self.history[-3:]]
            return recent[-1] > recent[0]
        return False


class ChemicalWordGrounder:
    """Ground smell/chemical words to belief space."""

    def __init__(self):
        self.words = {}

    def ground_from_examples(self, word, beliefs):
        if beliefs:
            self.words[word] = np.mean(beliefs, axis=0).astype(np.float32)

    def ground_vocabulary(self, substance_beliefs):
        for substance, beliefs in substance_beliefs.items():
            if beliefs:
                self.ground_from_examples(substance, beliefs)

        # Derived words
        if "sweet" in self.words and "bitter" in self.words:
            self.words["sour"] = (self.words["sweet"] * 0.3 +
                                    self.words["bitter"] * 0.7).astype(np.float32)

    def similarity(self, w1, w2):
        b1, b2 = self.words.get(w1), self.words.get(w2)
        if b1 is None or b2 is None:
            return 0.0
        n1, n2 = np.linalg.norm(b1), np.linalg.norm(b2)
        if n1 < 1e-8 or n2 < 1e-8:
            return 0.0
        return float(np.dot(b1, b2) / (n1 * n2))

    @property
    def size(self):
        return len(self.words)


class OmnimodalFusion5:
    """Fuse all 5 senses into unified belief."""

    def __init__(self):
        self.weights = {
            "vision": 0.25, "proprio": 0.25,
            "audio": 0.15, "tactile": 0.20, "chemical": 0.15
        }

    def fuse(self, beliefs):
        result = np.zeros(D_BELIEF, dtype=np.float32)
        total = 0
        for mod, belief in beliefs.items():
            w = self.weights.get(mod, 0.1)
            result += w * belief
            total += w
        if total > 0:
            result /= total
        return result

    def fuse_with_attention(self, beliefs, novelty_scores):
        result = np.zeros(D_BELIEF, dtype=np.float32)
        total = 0
        for mod, belief in beliefs.items():
            base_w = self.weights.get(mod, 0.1)
            novelty = novelty_scores.get(mod, 0)
            w = min(base_w + novelty * 0.1, 0.8)
            result += w * belief
            total += w
        if total > 0:
            result /= total
        return result


def run_tests():
    print("=" * 65)
    print("  Chemical/Olfactory Encoder Tests")
    print("=" * 65)
    rng = np.random.RandomState(42)
    p = 0; t = 0

    sensor = ChemicalSensor()
    encoder = ChemicalBeliefEncoder()

    print("\n  T1: Sensor generates correct shape")
    data = sensor.generate("sweet", rng=rng)
    ok = data.shape == (N_CHEM_DIMS,)
    print(f"    Shape: {data.shape} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T2: Belief encoding produces 64-D")
    belief = encoder.encode(data)
    ok = belief.shape == (D_BELIEF,)
    print(f"    Shape: {belief.shape} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T3: Different substances produce different beliefs")
    sweet = encoder.encode(sensor.generate("sweet", rng=rng))
    smoke = encoder.encode(sensor.generate("smoke", rng=rng))
    dist = np.linalg.norm(sweet - smoke)
    ok = dist > 0.1
    print(f"    Distance: {dist:.3f} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T4: Concentration affects belief magnitude")
    faint = encoder.encode(sensor.generate("sweet", concentration=0.1, rng=rng))
    strong = encoder.encode(sensor.generate("sweet", concentration=0.9, rng=rng))
    ok = np.linalg.norm(strong) > np.linalg.norm(faint)
    print(f"    Faint={np.linalg.norm(faint):.2f} "
          f"Strong={np.linalg.norm(strong):.2f} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T5: Distance reduces signal")
    near = sensor.generate("smoke", concentration=0.8, distance=0.5, rng=rng)
    far = sensor.generate("smoke", concentration=0.8, distance=5.0, rng=rng)
    ok = near.mean() > far.mean()
    print(f"    Near={near.mean():.3f} Far={far.mean():.3f} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T6: Gradient tracker detects approach")
    tracker = ChemicalGradientTracker()
    for d in [5.0, 4.0, 3.0, 2.0, 1.0]:
        data = sensor.generate("sweet", concentration=0.5, distance=d, rng=rng)
        belief = encoder.encode(data)
        tracker.update(belief, float(data.mean()))
    ok = tracker.is_approaching()
    print(f"    Approaching: {tracker.is_approaching()} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T7: Word grounding works")
    grounder = ChemicalWordGrounder()
    substance_beliefs = {}
    for sub in ["sweet", "bitter", "smoke", "fresh", "chemical"]:
        beliefs = [encoder.encode(sensor.generate(sub, rng=rng))
                     for _ in range(10)]
        substance_beliefs[sub] = beliefs
    grounder.ground_vocabulary(substance_beliefs)
    ok = grounder.size >= 5
    print(f"    Words: {grounder.size} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T8: sweet/bitter are different")
    sim = grounder.similarity("sweet", "bitter")
    ok = sim < 0.99
    print(f"    sim(sweet, bitter) = {sim:+.3f} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T9: 5-sense fusion works")
    fusion = OmnimodalFusion5()
    beliefs = {
        "vision": rng.randn(D_BELIEF).astype(np.float32),
        "proprio": rng.randn(D_BELIEF).astype(np.float32),
        "audio": rng.randn(D_BELIEF).astype(np.float32),
        "tactile": rng.randn(D_BELIEF).astype(np.float32),
        "chemical": encoder.encode(data),
    }
    fused = fusion.fuse(beliefs)
    ok = fused.shape == (D_BELIEF,)
    print(f"    Fused shape: {fused.shape} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T10: Batch encoding works")
    batch = np.stack([sensor.generate("neutral", rng=rng) for _ in range(10)])
    beliefs_batch = encoder.encode(batch)
    ok = beliefs_batch.shape == (10, D_BELIEF)
    print(f"    Batch shape: {beliefs_batch.shape} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print(f"\n{'='*65}")
    print(f"  Results: {p}/{t} tests passed")
    print(f"  5/5 biological senses complete!")
    print(f"  Vision + Proprioception + Audio + Tactile + Chemical")
    print(f"{'='*65}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        run_tests()
    else:
        run_tests()
