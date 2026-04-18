"""
tactile_encoder.py — Touch Perception for NeMo-WM
=====================================================
Projects tactile sensor data into 64-D belief space.
Completes 4/5 senses: vision + proprio + audio + TOUCH.

Tactile modalities encoded:
  - Pressure (force magnitude)
  - Texture (surface roughness via vibration frequency)
  - Temperature (thermal sensing)
  - Slip (object sliding detection)
  - Contact (binary touch / no-touch)

Grounded words: hard, soft, rough, smooth, hot, cold,
                contact, slip, grip, press, release, edge, flat

Usage:
    python tactile_encoder.py          # demo
    python tactile_encoder.py --test   # run tests
"""

import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional

D_BELIEF = 64
N_TACTILE_DIMS = 32  # 32 sensor elements (like a pressure pad array)


class TactileSensor:
    """
    Simulate a tactile sensor array.
    32 elements: 16 pressure + 4 temperature + 4 vibration + 4 slip + 4 contact
    """

    def __init__(self, n_elements=N_TACTILE_DIMS):
        self.n_elements = n_elements
        self.n_pressure = 16
        self.n_temp = 4
        self.n_vibration = 4
        self.n_slip = 4
        self.n_contact = 4

    def generate(self, surface="neutral", force=0.5, temp=0.5,
                   moving=False, rng=None):
        """Generate synthetic tactile reading."""
        if rng is None:
            rng = np.random.RandomState()

        data = np.zeros(self.n_elements, dtype=np.float32)

        # Pressure (0-15): spatial force distribution
        if surface == "hard":
            data[:self.n_pressure] = force * (0.8 + rng.randn(self.n_pressure).astype(np.float32) * 0.05)
        elif surface == "soft":
            data[:self.n_pressure] = force * (0.3 + rng.randn(self.n_pressure).astype(np.float32) * 0.15)
        elif surface == "edge":
            data[:self.n_pressure//2] = force * 0.9
            data[self.n_pressure//2:self.n_pressure] = force * 0.1
        else:
            data[:self.n_pressure] = force * (0.5 + rng.randn(self.n_pressure).astype(np.float32) * 0.1)

        # Temperature (16-19)
        idx = self.n_pressure
        if surface == "hot":
            data[idx:idx+self.n_temp] = 0.8 + rng.randn(self.n_temp).astype(np.float32) * 0.05
        elif surface == "cold":
            data[idx:idx+self.n_temp] = 0.2 + rng.randn(self.n_temp).astype(np.float32) * 0.05
        else:
            data[idx:idx+self.n_temp] = temp + rng.randn(self.n_temp).astype(np.float32) * 0.05

        # Vibration / texture (20-23)
        idx += self.n_temp
        if surface == "rough":
            data[idx:idx+self.n_vibration] = 0.7 + rng.randn(self.n_vibration).astype(np.float32) * 0.15
        elif surface == "smooth":
            data[idx:idx+self.n_vibration] = 0.1 + rng.randn(self.n_vibration).astype(np.float32) * 0.03
        else:
            data[idx:idx+self.n_vibration] = 0.3 + rng.randn(self.n_vibration).astype(np.float32) * 0.1

        # Slip (24-27)
        idx += self.n_vibration
        if moving:
            data[idx:idx+self.n_slip] = 0.6 + rng.randn(self.n_slip).astype(np.float32) * 0.1
        else:
            data[idx:idx+self.n_slip] = 0.05 + rng.randn(self.n_slip).astype(np.float32) * 0.02

        # Contact (28-31)
        idx += self.n_slip
        if force > 0.1:
            data[idx:idx+self.n_contact] = 1.0
        else:
            data[idx:idx+self.n_contact] = 0.0

        return np.clip(data, 0, 1)


class TactileBeliefEncoder:
    """Project tactile sensor data into 64-D belief space."""

    def __init__(self, n_tactile=N_TACTILE_DIMS, d_belief=D_BELIEF):
        rng = np.random.RandomState(seed=456)
        self.W_proj = rng.randn(n_tactile, d_belief).astype(np.float32) * 0.3
        self.b_proj = rng.randn(d_belief).astype(np.float32) * 0.1

    def encode(self, tactile_data):
        """Encode tactile reading to belief space."""
        if tactile_data.ndim == 1:
            tactile_data = tactile_data.reshape(1, -1)
        beliefs = np.tanh(tactile_data @ self.W_proj + self.b_proj)
        return beliefs.squeeze() if tactile_data.shape[0] == 1 else beliefs


class TactileNoveltyDetector:
    """Detect unusual touch sensations."""

    def __init__(self, threshold=2.0):
        self.ref_beliefs = []
        self.ref_mean = None
        self.ref_std = None
        self.threshold = threshold

    def fit_reference(self, beliefs):
        self.ref_beliefs = beliefs
        stacked = np.stack(beliefs)
        self.ref_mean = stacked.mean(axis=0)
        self.ref_std = stacked.std(axis=0) + 1e-8

    def score(self, belief):
        if self.ref_mean is None:
            return 0.0
        z = (belief - self.ref_mean) / self.ref_std
        return float(np.sqrt(np.mean(z ** 2)))

    def is_anomaly(self, belief):
        s = self.score(belief)
        return s > self.threshold, s


class TactileWordGrounder:
    """Ground touch words to belief space."""

    def __init__(self):
        self.words = {}

    def ground_from_examples(self, word, beliefs):
        if beliefs:
            self.words[word] = np.mean(beliefs, axis=0).astype(np.float32)

    def ground_tactile_vocabulary(self, surface_beliefs):
        """
        Auto-ground tactile words from categorized examples.
        surface_beliefs: dict of surface_type → list of beliefs
        """
        for surface, beliefs in surface_beliefs.items():
            if beliefs:
                self.ground_from_examples(surface, beliefs)

        # Derive compound words
        if "hard" in self.words and "soft" in self.words:
            self.words["firm"] = (self.words["hard"] * 0.7 +
                                    self.words["soft"] * 0.3).astype(np.float32)
        if "rough" in self.words and "smooth" in self.words:
            self.words["textured"] = (self.words["rough"] * 0.6 +
                                        self.words["smooth"] * 0.4).astype(np.float32)

    def lookup(self, word):
        return self.words.get(word)

    def similarity(self, w1, w2):
        b1, b2 = self.lookup(w1), self.lookup(w2)
        if b1 is None or b2 is None:
            return 0.0
        n1, n2 = np.linalg.norm(b1), np.linalg.norm(b2)
        if n1 < 1e-8 or n2 < 1e-8:
            return 0.0
        return float(np.dot(b1, b2) / (n1 * n2))

    @property
    def size(self):
        return len(self.words)


class OmnimodalFusion:
    """
    Fuse all 4 senses into unified belief.
    Vision + Proprio + Audio + Tactile → single 64-D state.
    """

    def __init__(self, weights=None):
        self.weights = weights or {
            "vision": 0.3, "proprio": 0.3,
            "audio": 0.15, "tactile": 0.25
        }

    def fuse(self, beliefs):
        """
        beliefs: dict of modality → 64-D belief
        Returns: weighted 64-D belief
        """
        result = np.zeros(D_BELIEF, dtype=np.float32)
        total_weight = 0

        for modality, belief in beliefs.items():
            w = self.weights.get(modality, 0.1)
            result += w * belief
            total_weight += w

        if total_weight > 0:
            result /= total_weight

        return result

    def fuse_with_attention(self, beliefs, novelty_scores):
        """
        Attention-gated: anomalous modalities get more weight.
        Like the audio encoder's attention gating but across ALL senses.
        """
        result = np.zeros(D_BELIEF, dtype=np.float32)
        total_weight = 0

        for modality, belief in beliefs.items():
            base_w = self.weights.get(modality, 0.1)
            novelty = novelty_scores.get(modality, 0)
            # Anomalous modalities get boosted attention
            attention_w = base_w + novelty * 0.1
            attention_w = min(attention_w, 0.8)
            result += attention_w * belief
            total_weight += attention_w

        if total_weight > 0:
            result /= total_weight

        return result


def demo():
    print("=" * 70)
    print("  Tactile Perception for NeMo-WM")
    print("  Touch sensing → 64-D belief space → grounded words")
    print("=" * 70)

    rng = np.random.RandomState(42)
    sensor = TactileSensor()
    encoder = TactileBeliefEncoder()

    # Generate touch data for different surfaces
    print("\n  ── 1. TACTILE SENSING ──")
    surfaces = {
        "hard": [sensor.generate("hard", force=0.8, rng=rng) for _ in range(20)],
        "soft": [sensor.generate("soft", force=0.3, rng=rng) for _ in range(20)],
        "rough": [sensor.generate("rough", force=0.5, rng=rng) for _ in range(20)],
        "smooth": [sensor.generate("smooth", force=0.5, rng=rng) for _ in range(20)],
        "hot": [sensor.generate("hot", force=0.5, temp=0.8, rng=rng) for _ in range(20)],
        "cold": [sensor.generate("cold", force=0.5, temp=0.2, rng=rng) for _ in range(20)],
        "edge": [sensor.generate("edge", force=0.7, rng=rng) for _ in range(20)],
        "contact": [sensor.generate("neutral", force=0.6, rng=rng) for _ in range(20)],
        "slip": [sensor.generate("neutral", force=0.4, moving=True, rng=rng) for _ in range(20)],
        "grip": [sensor.generate("neutral", force=0.9, moving=False, rng=rng) for _ in range(20)],
        "release": [sensor.generate("neutral", force=0.05, rng=rng) for _ in range(20)],
    }

    for name, readings in surfaces.items():
        mean_pressure = np.mean([r[:16].mean() for r in readings])
        print(f"    {name:<10}: {len(readings)} samples, "
              f"mean pressure={mean_pressure:.3f}")

    # Encode to belief space
    print("\n  ── 2. BELIEF ENCODING ──")
    surface_beliefs = {}
    for name, readings in surfaces.items():
        beliefs = [encoder.encode(r) for r in readings]
        surface_beliefs[name] = beliefs
        print(f"    {name:<10}: belief norm={np.linalg.norm(beliefs[0]):.2f}")

    # Novelty detection
    print("\n  ── 3. TACTILE NOVELTY ──")
    detector = TactileNoveltyDetector(threshold=2.0)
    detector.fit_reference(surface_beliefs["contact"])

    for name in ["contact", "hard", "slip", "hot"]:
        score = np.mean([detector.score(b) for b in surface_beliefs[name]])
        is_anom = score > 2.0
        print(f"    {name:<10}: score={score:.2f} "
              f"({'ANOMALY' if is_anom else 'normal'})")

    # Word grounding
    print("\n  ── 4. TACTILE WORD GROUNDING ──")
    grounder = TactileWordGrounder()
    grounder.ground_tactile_vocabulary(surface_beliefs)
    print(f"    Grounded {grounder.size} tactile words:")
    for word in sorted(grounder.words.keys()):
        print(f"      {word}")

    print(f"\n    Word similarities:")
    pairs = [
        ("hard", "soft"), ("rough", "smooth"), ("hot", "cold"),
        ("contact", "release"), ("slip", "grip"), ("hard", "edge"),
    ]
    for w1, w2 in pairs:
        sim = grounder.similarity(w1, w2)
        if sim != 0:
            print(f"      sim({w1:>8}, {w2:<8}) = {sim:+.3f}")

    # Omnimodal fusion
    print("\n  ── 5. OMNIMODAL FUSION (4 senses) ──")
    fusion = OmnimodalFusion()

    # Simulate all 4 modalities
    proprio_belief = rng.randn(D_BELIEF).astype(np.float32) * 0.5
    vision_belief = rng.randn(D_BELIEF).astype(np.float32) * 0.5
    audio_belief = rng.randn(D_BELIEF).astype(np.float32) * 0.5
    tactile_belief = encoder.encode(surfaces["hard"][0])

    fused = fusion.fuse({
        "vision": vision_belief,
        "proprio": proprio_belief,
        "audio": audio_belief,
        "tactile": tactile_belief,
    })

    print(f"    Vision norm:   {np.linalg.norm(vision_belief):.2f}")
    print(f"    Proprio norm:  {np.linalg.norm(proprio_belief):.2f}")
    print(f"    Audio norm:    {np.linalg.norm(audio_belief):.2f}")
    print(f"    Tactile norm:  {np.linalg.norm(tactile_belief):.2f}")
    print(f"    Fused norm:    {np.linalg.norm(fused):.2f}")

    # Attention-gated
    fused_calm = fusion.fuse_with_attention(
        {"vision": vision_belief, "proprio": proprio_belief,
         "audio": audio_belief, "tactile": tactile_belief},
        {"vision": 0.1, "proprio": 0.1, "audio": 0.1, "tactile": 0.1})
    fused_touch_alert = fusion.fuse_with_attention(
        {"vision": vision_belief, "proprio": proprio_belief,
         "audio": audio_belief, "tactile": tactile_belief},
        {"vision": 0.1, "proprio": 0.1, "audio": 0.1, "tactile": 5.0})

    touch_influence_calm = np.linalg.norm(fused_calm - proprio_belief)
    touch_influence_alert = np.linalg.norm(fused_touch_alert - proprio_belief)
    print(f"\n    Calm: tactile influence = {touch_influence_calm:.2f}")
    print(f"    Touch alert: influence = {touch_influence_alert:.2f}")
    print(f"    → Alert pays {touch_influence_alert/max(touch_influence_calm,0.01):.1f}× "
          f"more attention to touch")

    print(f"\n{'='*70}")
    print(f"  Omnimodal: NeMo-WM can now SEE, MOVE, HEAR, and TOUCH")
    print(f"  4/5 biological senses encoded in shared 64-D belief space")
    print(f"{'='*70}")


def run_tests():
    print("=" * 65)
    print("  Tactile Encoder Tests")
    print("=" * 65)
    rng = np.random.RandomState(42)
    p = 0; t = 0

    sensor = TactileSensor()
    encoder = TactileBeliefEncoder()

    print("\n  T1: Sensor generates correct shape")
    data = sensor.generate("hard", rng=rng)
    ok = data.shape == (N_TACTILE_DIMS,)
    print(f"    Shape: {data.shape} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T2: Belief encoding produces 64-D")
    belief = encoder.encode(data)
    ok = belief.shape == (D_BELIEF,)
    print(f"    Shape: {belief.shape} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T3: Different surfaces produce different beliefs")
    hard = encoder.encode(sensor.generate("hard", force=0.8, rng=rng))
    soft = encoder.encode(sensor.generate("soft", force=0.3, rng=rng))
    dist = np.linalg.norm(hard - soft)
    ok = dist > 0.1
    print(f"    Distance: {dist:.3f} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T4: Novelty detector works")
    det = TactileNoveltyDetector(threshold=1.5)
    normal_beliefs = [encoder.encode(sensor.generate("neutral", rng=rng))
                       for _ in range(20)]
    det.fit_reference(normal_beliefs)
    hot_belief = encoder.encode(sensor.generate("hot", temp=0.9, rng=rng))
    is_anom, score = det.is_anomaly(hot_belief)
    ok = is_anom
    print(f"    Hot anomaly={is_anom} score={score:.2f} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T5: Normal touch not flagged")
    normal = encoder.encode(sensor.generate("neutral", rng=rng))
    is_anom, score = det.is_anomaly(normal)
    ok = not is_anom
    print(f"    Normal anomaly={is_anom} score={score:.2f} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T6: Word grounding works")
    grounder = TactileWordGrounder()
    surface_beliefs = {
        "hard": [encoder.encode(sensor.generate("hard", rng=rng))
                  for _ in range(10)],
        "soft": [encoder.encode(sensor.generate("soft", rng=rng))
                  for _ in range(10)],
        "rough": [encoder.encode(sensor.generate("rough", rng=rng))
                   for _ in range(10)],
        "smooth": [encoder.encode(sensor.generate("smooth", rng=rng))
                    for _ in range(10)],
    }
    grounder.ground_tactile_vocabulary(surface_beliefs)
    ok = grounder.size >= 4
    print(f"    Words: {grounder.size} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T7: hard/soft are different")
    sim = grounder.similarity("hard", "soft")
    ok = sim < 0.95
    print(f"    sim(hard, soft) = {sim:+.3f} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T8: Omnimodal fusion works")
    fusion = OmnimodalFusion()
    beliefs = {
        "vision": rng.randn(D_BELIEF).astype(np.float32),
        "proprio": rng.randn(D_BELIEF).astype(np.float32),
        "audio": rng.randn(D_BELIEF).astype(np.float32),
        "tactile": encoder.encode(data),
    }
    fused = fusion.fuse(beliefs)
    ok = fused.shape == (D_BELIEF,)
    print(f"    Fused shape: {fused.shape} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T9: Attention gating increases for anomalous touch")
    f_calm = fusion.fuse_with_attention(beliefs,
        {"vision": 0.1, "proprio": 0.1, "audio": 0.1, "tactile": 0.1})
    f_alert = fusion.fuse_with_attention(beliefs,
        {"vision": 0.1, "proprio": 0.1, "audio": 0.1, "tactile": 5.0})
    calm_shift = np.linalg.norm(f_calm - beliefs["proprio"])
    alert_shift = np.linalg.norm(f_alert - beliefs["proprio"])
    ok = alert_shift > calm_shift
    print(f"    Calm={calm_shift:.2f} Alert={alert_shift:.2f} "
          f"{'PASS' if ok else 'FAIL'}")
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
    print(f"{'='*65}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        run_tests()
    else:
        demo()
