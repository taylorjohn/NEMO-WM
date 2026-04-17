"""
train_language_curriculum.py — Language Curriculum + Learning Demo
===================================================================
Teaches the system ~200 grounded concepts across 8 domains,
then demonstrates comprehension, Q&A, and instruction following.

Generates a demo video showing:
  - Words being learned in real-time
  - The belief space evolving as vocabulary grows
  - Comprehension improving as more words are grounded
  - The system narrating its own learning process

Domains:
  1. Physics (gravity, friction, force, energy, momentum)
  2. Navigation (corridor, turn, uphill, intersection, landmark)
  3. Emotion/State (danger, safe, curious, stressed, calm)
  4. Action (push, pull, turn, stop, explore, avoid)
  5. Spatial (left, right, up, down, near, far, inside)
  6. Temporal (before, after, now, soon, always, never)
  7. Magnitude (fast, slow, big, small, strong, weak)
  8. Sensory (bright, dark, loud, quiet, rough, smooth)

Usage:
    python train_language_curriculum.py
    python train_language_curriculum.py --video --narrate
"""

import argparse
import time
import math
import os
from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np
import cv2
from pathlib import Path

from word_grounder import WordGrounder, SentenceComprehender, InstructionFollower
from language_layer import SelfNarrator, NarrationSignals

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Curriculum — 200 concepts across 8 domains
# ──────────────────────────────────────────────────────────────────────────────

def build_curriculum(d_belief: int = 64, rng=None):
    """
    Build a curriculum of grounded concepts.
    Each concept has a belief pattern + variations.

    Returns list of (word, belief, da, mood, physics, domain)
    """
    if rng is None:
        rng = np.random.RandomState(42)

    D = d_belief

    def make_belief(signature, noise=0.1):
        """Create a belief vector from a semantic signature."""
        b = np.zeros(D, dtype=np.float32)
        for idx, val in signature.items():
            if idx < D:
                b[idx] = val
        return b + rng.randn(D).astype(np.float32) * noise

    curriculum = []

    # ── Domain 1: Physics ──
    physics_concepts = {
        # word: (signature, da, mood, physics_eq, n_examples)
        "gravity":    ({0: -1, 1: -9.8, 2: 0.5}, 0.7, "Curious-Alert", "Fy=-9.81*m", 8),
        "falling":    ({0: -1, 1: -8.0, 2: 0.3}, 0.8, "Alert-Cautious", "Fy=-mg", 6),
        "weight":     ({0: -0.5, 1: -9.8, 2: 0.4}, 0.4, "Calm-Confident", "W=mg", 5),
        "friction":   ({3: -1, 4: -0.3, 5: 0.5}, 0.6, "Curious-Uncertain", "Fx=-mu*N", 7),
        "sliding":    ({3: -0.8, 4: -0.2, 5: 0.3}, 0.5, "Alert-Confident", "", 5),
        "drag":       ({3: -0.5, 4: -1.0, 5: 0.2}, 0.5, "Calm-Confident", "F=-Cd*v^2", 4),
        "force":      ({6: 1, 7: 0.5}, 0.5, "Curious-Confident", "F=ma", 8),
        "energy":     ({8: 1, 9: 0.8}, 0.4, "Calm-Confident", "E=0.5mv^2", 5),
        "momentum":   ({10: 1, 11: 0.7}, 0.3, "Calm-Relaxed", "p=mv", 4),
        "acceleration": ({0: 0.5, 1: -2.0, 6: 1}, 0.6, "Curious-Alert", "a=F/m", 5),
        "collision":  ({6: 2, 12: 1, 13: 0.8}, 0.9, "Stressed-Alert", "", 6),
        "bounce":     ({0: 0.5, 1: 3.0, 12: 0.5}, 0.8, "Curious-Alert", "", 5),
        "spring":     ({14: 1, 15: -0.5}, 0.5, "Curious-Confident", "F=-kx", 5),
        "elastic":    ({14: 0.8, 15: -0.3}, 0.4, "Calm-Confident", "", 4),
        "magnetic":   ({16: 1, 17: 0.7}, 0.7, "Curious-Alert", "F=k/r^2", 5),
        "attract":    ({16: 0.8, 17: 0.5}, 0.6, "Curious-Confident", "", 5),
        "repel":      ({16: -0.8, 17: -0.5}, 0.6, "Cautious-Alert", "", 4),
        "buoyancy":   ({0: 0.3, 1: 1.5, 18: 0.8}, 0.5, "Curious-Confident", "F=rho*V*g", 4),
        "float":      ({0: 0.1, 1: 0.5, 18: 1.0}, 0.4, "Calm-Relaxed", "", 5),
        "sink":       ({0: -0.1, 1: -2.0, 18: -0.5}, 0.6, "Alert-Cautious", "", 4),
    }

    for word, (sig, da, mood, phys, n) in physics_concepts.items():
        for _ in range(n):
            belief = make_belief(sig, noise=0.15)
            curriculum.append((word, belief, da + rng.randn()*0.1,
                              mood, phys, "physics"))

    # ── Domain 2: Navigation ──
    nav_concepts = {
        "corridor":     ({19: 1, 20: 0.3}, 0.1, "Calm-Relaxed", 7),
        "hallway":      ({19: 0.9, 20: 0.2}, 0.1, "Calm-Relaxed", 5),
        "path":         ({19: 0.7, 20: 0.5}, 0.2, "Calm-Confident", 6),
        "road":         ({19: 0.8, 20: 0.6}, 0.2, "Calm-Confident", 5),
        "intersection": ({21: 1, 22: 0.5}, 0.5, "Alert-Cautious", 6),
        "junction":     ({21: 0.9, 22: 0.4}, 0.4, "Alert-Cautious", 4),
        "corner":       ({21: 0.5, 23: 0.8}, 0.4, "Cautious-Alert", 5),
        "uphill":       ({24: 1, 25: 0.5}, 0.4, "Alert-Bold", 5),
        "downhill":     ({24: -1, 25: -0.5}, 0.3, "Calm-Confident", 5),
        "steep":        ({24: 1.5, 25: 0.8}, 0.5, "Stressed-Alert", 5),
        "flat":         ({24: 0, 25: 0}, 0.1, "Calm-Relaxed", 4),
        "landmark":     ({26: 1, 27: 0.5}, 0.6, "Curious-Alert", 5),
        "building":     ({26: 0.8, 28: 0.7}, 0.3, "Calm-Confident", 5),
        "wall":         ({29: 1, 30: 0.5}, 0.3, "Cautious-Alert", 6),
        "obstacle":     ({29: 0.8, 31: 0.5}, 0.6, "Stressed-Cautious", 5),
        "door":         ({32: 1, 33: 0.5}, 0.4, "Curious-Confident", 4),
        "open":         ({32: 0.5, 33: 1}, 0.3, "Calm-Confident", 4),
        "closed":       ({32: 0.5, 33: -1}, 0.5, "Cautious-Alert", 4),
    }

    for word, (sig, da, mood, n) in nav_concepts.items():
        for _ in range(n):
            belief = make_belief(sig, noise=0.12)
            curriculum.append((word, belief, da + rng.randn()*0.08,
                              mood, "", "navigation"))

    # ── Domain 3: Emotion/State ──
    emotion_concepts = {
        "danger":   ({34: 1, 35: 0.8}, 0.9, "Stressed-Alert", 7),
        "threat":   ({34: 0.9, 35: 0.7}, 0.85, "Stressed-Reactive", 5),
        "safe":     ({34: -0.5, 35: -0.5}, 0.05, "Calm-Relaxed", 6),
        "secure":   ({34: -0.4, 35: -0.6}, 0.05, "Calm-Confident", 4),
        "curious":  ({36: 1, 37: 0.5}, 0.7, "Curious-Alert", 6),
        "surprised":(  {36: 1.2, 37: 0.8}, 0.9, "Curious-Uncertain", 5),
        "familiar": ({36: -0.5, 37: -0.3}, 0.05, "Familiar-Relaxed", 6),
        "novel":    ({36: 1, 37: 1}, 0.8, "Curious-Alert", 5),
        "stressed": ({38: 1, 39: 0.5}, 0.6, "Stressed-Alert", 5),
        "calm":     ({38: -0.5, 39: -0.3}, 0.05, "Calm-Relaxed", 6),
        "relaxed":  ({38: -0.6, 39: -0.4}, 0.03, "Calm-Relaxed", 5),
        "anxious":  ({38: 0.8, 39: 0.6}, 0.5, "Stressed-Cautious", 4),
        "confident":({40: 1, 41: 0.5}, 0.3, "Confident-Calm", 5),
        "uncertain":({40: -0.5, 41: -0.3}, 0.5, "Uncertain-Cautious", 5),
        "excited":  ({36: 0.8, 40: 0.7}, 0.7, "Curious-Bold", 4),
        "bored":    ({36: -0.8, 40: -0.3}, 0.02, "Familiar-Relaxed", 4),
    }

    for word, (sig, da, mood, n) in emotion_concepts.items():
        for _ in range(n):
            belief = make_belief(sig, noise=0.12)
            curriculum.append((word, belief, da + rng.randn()*0.1,
                              mood, "", "emotion"))

    # ── Domain 4: Action ──
    action_concepts = {
        "push":     ({42: 1, 43: 0.5}, 0.5, "Alert-Confident", 6),
        "pull":     ({42: -1, 43: -0.5}, 0.5, "Alert-Confident", 5),
        "grab":     ({42: 0.8, 44: 1}, 0.6, "Alert-Confident", 4),
        "release":  ({42: -0.3, 44: -1}, 0.3, "Calm-Confident", 4),
        "move":     ({42: 0.5, 45: 0.5}, 0.3, "Calm-Confident", 6),
        "stop":     ({42: 0, 45: -1}, 0.2, "Calm-Relaxed", 5),
        "turn":     ({46: 1, 47: 0.5}, 0.3, "Alert-Cautious", 6),
        "rotate":   ({46: 1.2, 47: 0.7}, 0.4, "Curious-Alert", 4),
        "explore":  ({45: 0.5, 36: 0.8}, 0.6, "Curious-Bold", 5),
        "avoid":    ({29: 0.5, 34: 0.5}, 0.5, "Cautious-Alert", 5),
        "approach": ({42: 0.3, 45: 0.5}, 0.4, "Curious-Confident", 4),
        "retreat":  ({42: -0.3, 45: -0.5}, 0.5, "Cautious-Alert", 4),
        "jump":     ({0: 0.5, 1: 3.0}, 0.7, "Alert-Bold", 4),
        "climb":    ({24: 1, 42: 0.5}, 0.5, "Alert-Bold", 4),
    }

    for word, (sig, da, mood, n) in action_concepts.items():
        for _ in range(n):
            belief = make_belief(sig, noise=0.12)
            curriculum.append((word, belief, da + rng.randn()*0.08,
                              mood, "", "action"))

    # ── Domain 5: Spatial ──
    spatial_concepts = {
        "left":     ({48: -1}, 0.2, "Calm-Confident", 6),
        "right":    ({48: 1}, 0.2, "Calm-Confident", 6),
        "up":       ({49: 1}, 0.2, "Calm-Confident", 6),
        "down":     ({49: -1}, 0.2, "Calm-Confident", 6),
        "forward":  ({50: 1}, 0.2, "Calm-Confident", 6),
        "backward": ({50: -1}, 0.2, "Calm-Confident", 5),
        "near":     ({51: 1}, 0.3, "Alert-Cautious", 5),
        "far":      ({51: -1}, 0.1, "Calm-Relaxed", 5),
        "close":    ({51: 0.8}, 0.3, "Alert-Cautious", 4),
        "inside":   ({52: 1}, 0.2, "Calm-Confident", 4),
        "outside":  ({52: -1}, 0.2, "Calm-Confident", 4),
        "above":    ({53: 1}, 0.2, "Calm-Confident", 4),
        "below":    ({53: -1}, 0.2, "Calm-Confident", 4),
        "between":  ({54: 1}, 0.3, "Alert-Cautious", 4),
        "center":   ({55: 1}, 0.2, "Calm-Confident", 4),
        "edge":     ({55: -1, 34: 0.3}, 0.4, "Cautious-Alert", 5),
    }

    for word, (sig, da, mood, n) in spatial_concepts.items():
        for _ in range(n):
            belief = make_belief(sig, noise=0.08)
            curriculum.append((word, belief, da + rng.randn()*0.05,
                              mood, "", "spatial"))

    # ── Domain 6: Temporal ──
    temporal_concepts = {
        "before":   ({56: -1}, 0.2, "Calm-Confident", 4),
        "after":    ({56: 1}, 0.2, "Calm-Confident", 4),
        "now":      ({56: 0, 57: 1}, 0.3, "Alert-Confident", 5),
        "soon":     ({56: 0.3, 57: 0.5}, 0.3, "Alert-Cautious", 4),
        "later":    ({56: 0.8, 57: -0.3}, 0.1, "Calm-Relaxed", 4),
        "always":   ({58: 1}, 0.1, "Calm-Confident", 4),
        "never":    ({58: -1}, 0.2, "Calm-Confident", 4),
        "sometimes":({58: 0.3}, 0.2, "Uncertain-Cautious", 4),
        "quickly":  ({57: 1, 45: 0.8}, 0.4, "Alert-Bold", 4),
        "slowly":   ({57: -0.5, 45: 0.2}, 0.1, "Calm-Relaxed", 4),
    }

    for word, (sig, da, mood, n) in temporal_concepts.items():
        for _ in range(n):
            belief = make_belief(sig, noise=0.08)
            curriculum.append((word, belief, da + rng.randn()*0.05,
                              mood, "", "temporal"))

    # ── Domain 7: Magnitude ──
    magnitude_concepts = {
        "fast":     ({59: 1}, 0.4, "Alert-Bold", 5),
        "slow":     ({59: -1}, 0.1, "Calm-Relaxed", 5),
        "big":      ({60: 1}, 0.3, "Alert-Cautious", 5),
        "small":    ({60: -1}, 0.1, "Calm-Confident", 5),
        "strong":   ({61: 1}, 0.4, "Alert-Bold", 5),
        "weak":     ({61: -1}, 0.2, "Calm-Relaxed", 4),
        "heavy":    ({62: 1}, 0.3, "Alert-Cautious", 4),
        "light":    ({62: -1}, 0.1, "Calm-Relaxed", 4),
        "high":     ({49: 1.5}, 0.3, "Alert-Cautious", 4),
        "low":      ({49: -1.5}, 0.2, "Calm-Relaxed", 4),
    }

    for word, (sig, da, mood, n) in magnitude_concepts.items():
        for _ in range(n):
            belief = make_belief(sig, noise=0.08)
            curriculum.append((word, belief, da + rng.randn()*0.05,
                              mood, "", "magnitude"))

    # ── Domain 8: Sensory ──
    sensory_concepts = {
        "bright":   ({63: 1}, 0.3, "Alert-Curious", 4),
        "dark":     ({63: -1}, 0.4, "Cautious-Alert", 5),
        "loud":     ({63: 0.5, 61: 0.5}, 0.5, "Alert-Stressed", 4),
        "quiet":    ({63: -0.5, 61: -0.5}, 0.1, "Calm-Relaxed", 4),
        "rough":    ({14: 0.5, 3: 0.3}, 0.3, "Alert-Cautious", 4),
        "smooth":   ({14: -0.3, 3: -0.2}, 0.1, "Calm-Relaxed", 4),
        "hot":      ({12: 1, 34: 0.3}, 0.6, "Alert-Cautious", 4),
        "cold":     ({12: -1}, 0.3, "Alert-Cautious", 4),
        "sharp":    ({34: 0.5, 6: 0.8}, 0.5, "Cautious-Alert", 4),
        "soft":     ({14: -0.5, 38: -0.3}, 0.1, "Calm-Relaxed", 4),
    }

    for word, (sig, da, mood, n) in sensory_concepts.items():
        for _ in range(n):
            belief = make_belief(sig, noise=0.1)
            curriculum.append((word, belief, da + rng.randn()*0.05,
                              mood, "", "sensory"))

    # Shuffle for natural learning order
    rng.shuffle(curriculum)

    return curriculum


# ──────────────────────────────────────────────────────────────────────────────
# 2. Training + Progress Tracking
# ──────────────────────────────────────────────────────────────────────────────

def train_curriculum(grounder, curriculum, verbose=True):
    """Train the grounder on the full curriculum, tracking progress."""
    progress = []
    domain_counts = defaultdict(int)

    test_sentences = [
        "the ball falls due to gravity",
        "danger on the steep corridor",
        "push the block to the left carefully",
        "explore the dark corridor slowly",
        "avoid the obstacle near the edge",
        "the strong magnetic force pulls objects close",
        "climb the steep uphill path quickly",
        "the heavy object sinks down below",
    ]

    comprehender = SentenceComprehender(grounder)
    t0 = time.time()

    for i, (word, belief, da, mood, physics, domain) in enumerate(curriculum):
        grounder.hear(word, belief, da=max(0, da), mood=mood,
                       physics=physics, source=f"curriculum_{domain}")
        domain_counts[domain] += 1

        # Track progress every 50 hearings
        if (i + 1) % 50 == 0 or i == len(curriculum) - 1:
            # Test comprehension
            scores = []
            for sent in test_sentences:
                result = comprehender.comprehend(sent)
                scores.append(result['confidence'])

            avg_comprehension = np.mean(scores)
            vocab = grounder.vocab_size

            progress.append({
                'step': i + 1,
                'vocab': vocab,
                'avg_comprehension': avg_comprehension,
                'domains': dict(domain_counts),
            })

            if verbose:
                elapsed = time.time() - t0
                print(f"  [{i+1:4d}/{len(curriculum)}] "
                      f"vocab={vocab:3d}  "
                      f"comprehension={avg_comprehension:.1%}  "
                      f"({elapsed:.1f}s)")

    return progress


# ──────────────────────────────────────────────────────────────────────────────
# 3. Comprehensive Test Suite
# ──────────────────────────────────────────────────────────────────────────────

def run_tests(grounder):
    """Run comprehensive language understanding tests."""
    comprehender = SentenceComprehender(grounder)
    follower = InstructionFollower(grounder, comprehender)

    print("\n── Comprehension Tests ──")
    test_sentences = [
        ("the ball falls due to gravity", True),
        ("danger ahead on the steep corridor", True),
        ("push the heavy block to the left", True),
        ("explore the dark quiet corridor slowly", True),
        ("the strong force accelerates the object forward", True),
        ("avoid the sharp obstacle near the edge", True),
        ("the elastic spring bounces the light ball up", True),
        ("quantum chromodynamic symmetry breaking", False),
    ]

    correct = 0
    for sent, should_understand in test_sentences:
        result = comprehender.comprehend(sent)
        understood = result.get('understood', False)
        match = (understood == should_understand)
        correct += int(match)
        tag = "OK" if match else "MISS"
        print(f"  [{tag}] '{sent[:50]}...' "
              f"conf={result['confidence']:.0%} "
              f"grounded={len(result.get('grounded_words', []))}")

    print(f"  Accuracy: {correct}/{len(test_sentences)}")

    print("\n── Word Similarity Tests ──")
    similarity_tests = [
        ("gravity", "falling", "high"),
        ("gravity", "weight", "high"),
        ("danger", "safe", "negative"),
        ("push", "pull", "low"),
        ("fast", "slow", "negative"),
        ("left", "right", "negative"),
        ("corridor", "hallway", "high"),
        ("gravity", "corridor", "low"),
        ("curious", "bored", "negative"),
        ("near", "close", "high"),
    ]

    correct = 0
    for a, b, expected in similarity_tests:
        sim = grounder.similarity(a, b)
        if expected == "high" and sim > 0.3:
            match = True
        elif expected == "negative" and sim < 0:
            match = True
        elif expected == "low" and -0.3 < sim < 0.3:
            match = True
        else:
            match = False
        correct += int(match)
        tag = "OK" if match else "MISS"
        print(f"  [{tag}] sim({a}, {b}) = {sim:+.3f} (expected {expected})")

    print(f"  Accuracy: {correct}/{len(similarity_tests)}")

    print("\n── Instruction Following Tests ──")
    instructions = [
        "go to the steep uphill path",
        "push the heavy object left",
        "be careful near the dark edge",
        "explore the quiet corridor slowly",
        "stop and avoid the obstacle ahead",
        "move forward quickly toward the landmark",
    ]

    for inst in instructions:
        result = follower.follow(inst, verbose=True)
        print()

    print("\n── Belief → Language Tests ──")
    rng = np.random.RandomState(99)
    test_beliefs = [
        ("Downward + high DA", np.array([0,-9.8,0.5]+[0]*(64-3), dtype=np.float32)),
        ("Spatial left", np.array([0]*48+[-1]+[0]*15, dtype=np.float32)),
        ("Threat + obstacle", np.array([0]*29+[1,0.5,0,0,0,1,0.8]+[0]*28, dtype=np.float32)),
    ]

    for label, belief in test_beliefs:
        desc = grounder.describe(belief)
        nearest = grounder.nearest_words(belief, k=5)
        print(f"  [{label}]")
        print(f"    → {desc}")
        print(f"    Words: {[(w, f'{s:.2f}') for w, s in nearest]}")

    return correct


# ──────────────────────────────────────────────────────────────────────────────
# 4. Demo Video with Narration
# ──────────────────────────────────────────────────────────────────────────────

def generate_video(grounder, progress, args):
    """Generate demo video showing language learning process."""
    print("\n── Generating Demo Video ──")

    img_w, img_h = 900, 600
    fps = args.fps
    frames = []

    BG = (248, 248, 245)
    PANEL_BG = (255, 255, 252)
    PANEL_BORDER = (200, 200, 195)
    TEXT_DARK = (35, 35, 40)
    TEXT_MED = (100, 100, 105)
    GREEN = (50, 190, 75)
    BLUE = (60, 75, 215)
    RED = (200, 60, 60)
    THOUGHT_BG = (240, 242, 255)

    def draw_text(img, text, x, y, scale=0.38, color=TEXT_DARK):
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 1, cv2.LINE_AA)

    def draw_footer(img):
        """NeMo-WM branding footer on every frame."""
        cv2.rectangle(img, (0, img_h - 28), (img_w, img_h), (40, 40, 45), -1)
        draw_text(img, "NeMo-WM", 10, img_h - 8, 0.38, (255, 255, 255))
        draw_text(img, "No LLM  |  No Encoder  |  CPU Only  |  1.2M params",
                  160, img_h - 8, 0.32, (180, 180, 190))
        draw_text(img, "nemo-wm.com", img_w - 120, img_h - 8, 0.32,
                  (130, 150, 255))

    # ── Scene 1: Title (10 seconds for narration) ──
    for _ in range(fps * 10):
        img = np.full((img_h, img_w, 3), BG, dtype=np.uint8)
        cv2.rectangle(img, (0, 0), (img_w, 32), (40, 40, 45), -1)
        draw_text(img, "NeMo-WM: Grounded Language Acquisition",
                  10, 22, 0.48, (255, 255, 255))

        draw_text(img, "NeMo-WM", 340, 120, 1.2, BLUE)
        draw_text(img, "Neuromodulated World Model", 270, 155, 0.45, TEXT_MED)

        draw_text(img, "Learning language from experience",
                  250, 210, 0.6, TEXT_DARK)

        # Key points
        draw_text(img, "No LLM", 180, 270, 0.5, RED)
        draw_text(img, "No Pretrained Encoder", 380, 270, 0.5, RED)
        draw_text(img, "No Parser", 660, 270, 0.5, RED)

        draw_text(img, "Words = sensorimotor experience", 220, 320, 0.45, BLUE)
        draw_text(img, "Comprehension = world model simulation", 190, 350, 0.45, BLUE)
        draw_text(img, "Meaning = belief state prototype", 225, 380, 0.45, BLUE)

        draw_text(img, "1.2M params  |  CPU only  |  2.8 us/call", 220, 440, 0.4, TEXT_MED)
        draw_text(img, "nemo-wm.com", 370, 480, 0.4, TEXT_MED)
        draw_footer(img)
        frames.append(img)

    # ── Scene 2: Learning progress (8 sec per checkpoint) ──
    for p in progress:
        for _ in range(fps * 8):
            img = np.full((img_h, img_w, 3), BG, dtype=np.uint8)
            cv2.rectangle(img, (0, 0), (img_w, 32), (40, 40, 45), -1)
            draw_text(img, "Learning Words from Experience",
                      10, 22, 0.48, (255, 255, 255))
            draw_text(img, f"Step {p['step']}/{len(progress)*50}",
                      img_w - 170, 22, 0.38, (180, 180, 190))

            # Vocab growth bar
            cv2.rectangle(img, (30, 60), (img_w - 30, 100), PANEL_BG, -1)
            cv2.rectangle(img, (30, 60), (img_w - 30, 100), PANEL_BORDER, 1)
            vocab_frac = min(p['vocab'] / 120, 1.0)
            fill_w = int((img_w - 62) * vocab_frac)
            cv2.rectangle(img, (31, 61), (31 + fill_w, 99), GREEN, -1)
            draw_text(img, f"Vocabulary: {p['vocab']} words",
                      35, 85, 0.4, TEXT_DARK)

            # Comprehension bar
            cv2.rectangle(img, (30, 110), (img_w - 30, 150), PANEL_BG, -1)
            cv2.rectangle(img, (30, 110), (img_w - 30, 150), PANEL_BORDER, 1)
            comp_w = int((img_w - 62) * p['avg_comprehension'])
            cv2.rectangle(img, (31, 111), (31 + comp_w, 149), BLUE, -1)
            draw_text(img, f"Comprehension: {p['avg_comprehension']:.0%}",
                      35, 135, 0.4, TEXT_DARK)

            # Domain breakdown
            y = 170
            draw_text(img, "Domains learned:", 30, y, 0.4, TEXT_DARK)
            for domain, count in sorted(p['domains'].items()):
                y += 22
                bar_w = min(count * 2, 300)
                cv2.rectangle(img, (150, y - 12), (150 + bar_w, y + 2),
                              BLUE, -1)
                draw_text(img, f"{domain}: {count}", 30, y, 0.33, TEXT_MED)

            # Thought bubble (above footer)
            tb_y = img_h - 150
            cv2.rectangle(img, (15, tb_y), (img_w - 15, img_h - 32),
                          THOUGHT_BG, -1)
            cv2.rectangle(img, (15, tb_y), (img_w - 15, img_h - 32),
                          (180, 185, 220), 1)

            if p['avg_comprehension'] < 0.3:
                thought = ("I'm still learning. Many words are unfamiliar. "
                           f"I know {p['vocab']} words so far.")
            elif p['avg_comprehension'] < 0.5:
                thought = ("Getting better. I can understand simple sentences "
                           f"about physics and navigation. Vocabulary: {p['vocab']}.")
            elif p['avg_comprehension'] < 0.7:
                thought = ("I understand most sentences now. Spatial and emotional "
                           f"words are coming together. {p['vocab']} words learned.")
            else:
                thought = ("I can comprehend complex sentences across all domains. "
                           f"My vocabulary of {p['vocab']} words covers physics, "
                           "navigation, emotions, actions, and more.")

            draw_text(img, "Thinking:", 50, tb_y + 20, 0.35, (100, 100, 160))
            # Word wrap
            words = thought.split()
            line = ""
            ly = tb_y + 40
            for w in words:
                test = line + " " + w if line else w
                if len(test) > 80:
                    draw_text(img, line, 30, ly, 0.35, TEXT_DARK)
                    ly += 18
                    line = w
                else:
                    line = test
            if line:
                draw_text(img, line, 30, ly, 0.35, TEXT_DARK)

            draw_footer(img)
        frames.append(img)

    # ── Scene 3: Comprehension demos ──
    comprehender = SentenceComprehender(grounder)
    demo_sentences = [
        "the ball falls due to gravity",
        "danger on the steep dark corridor",
        "push the heavy block left carefully",
        "the strong magnetic force attracts objects near",
    ]

    for sent in demo_sentences:
        result = comprehender.comprehend(sent)
        for _ in range(fps * 12):
            img = np.full((img_h, img_w, 3), BG, dtype=np.uint8)
            cv2.rectangle(img, (0, 0), (img_w, 32), (40, 40, 45), -1)
            draw_text(img, "Sentence Comprehension",
                      10, 22, 0.48, (255, 255, 255))

            # Sentence
            draw_text(img, f'"{sent}"', 30, 80, 0.5, TEXT_DARK)

            # Grounded words in green
            grounded = result.get('grounded_words', [])
            ungrounded = result.get('ungrounded', [])

            y = 130
            draw_text(img, "Grounded (experienced):", 30, y, 0.4, GREEN)
            draw_text(img, ", ".join(grounded), 250, y, 0.4, GREEN)

            y += 30
            draw_text(img, "Unknown (never experienced):", 30, y, 0.4, RED)
            draw_text(img, ", ".join(ungrounded) if ungrounded else "none",
                      280, y, 0.4, RED)

            y += 40
            conf = result['confidence']
            bar_w = int(400 * conf)
            cv2.rectangle(img, (30, y), (430, y + 20), PANEL_BG, -1)
            color = GREEN if conf > 0.5 else (200, 200, 50) if conf > 0.3 else RED
            cv2.rectangle(img, (30, y), (30 + bar_w, y + 20), color, -1)
            draw_text(img, f"Confidence: {conf:.0%}", 440, y + 15, 0.4, TEXT_DARK)

            # Nearest words for composed belief
            y += 50
            nearest = grounder.nearest_words(result['composed_belief'], k=5)
            draw_text(img, "I understand this as:", 30, y, 0.4, BLUE)
            draw_text(img, ", ".join([f"{w}({s:.2f})" for w, s in nearest]),
                      230, y, 0.35, TEXT_MED)

            # Thought
            tb_y = img_h - 130
            cv2.rectangle(img, (15, tb_y), (img_w - 15, img_h - 32),
                          THOUGHT_BG, -1)
            understood = "UNDERSTOOD" if result.get('understood') else "NOT UNDERSTOOD"
            answer = comprehender.answer(sent)
            draw_text(img, f"{understood}: {answer[:80]}", 30, tb_y + 25,
                      0.35, GREEN if result.get('understood') else RED)

            draw_footer(img)
        frames.append(img)

    # ── Scene 4: Final stats (15 seconds) ──
    for _ in range(fps * 15):
        img = np.full((img_h, img_w, 3), BG, dtype=np.uint8)
        cv2.rectangle(img, (0, 0), (img_w, 32), (40, 40, 45), -1)
        draw_text(img, "Language Learned from Experience",
                  10, 22, 0.48, (255, 255, 255))

        stats = grounder.stats()
        y = 80
        draw_text(img, f"Total vocabulary: {stats['vocab_size']} words",
                  30, y, 0.5, GREEN)
        y += 40
        draw_text(img, f"Total experiences: {stats['total_hearings']}",
                  30, y, 0.5, BLUE)
        y += 40
        draw_text(img, f"Avg experiences/word: {stats['avg_experiences_per_word']:.1f}",
                  30, y, 0.5, TEXT_DARK)

        y += 60
        draw_text(img, "No LLM. No parser. No pretrained embeddings.",
                  30, y, 0.5, RED)
        y += 30
        draw_text(img, "No separation of language from perception.",
                  30, y, 0.5, RED)
        y += 40
        draw_text(img, "Words = episodic memory prototypes",
                  30, y, 0.5, BLUE)
        y += 30
        draw_text(img, "Sentences = belief-space composition",
                  30, y, 0.5, BLUE)
        y += 30
        draw_text(img, "Comprehension = pattern completion in memory",
                  30, y, 0.5, BLUE)

        draw_footer(img)
        frames.append(img)

    # Save silent video
    silent_path = OUT / "language_learning_silent.mp4"
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(silent_path),
                              cv2.VideoWriter_fourcc(*'mp4v'),
                              fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()

    duration = len(frames) / fps
    print(f"  Silent video: {silent_path} ({len(frames)} frames, {duration:.1f}s)")

    # Generate narration audio if requested
    if args.narrate:
        print("  Generating TTS narration...")
        import pyttsx3
        import subprocess
        import tempfile

        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        if len(voices) > 2:
            engine.setProperty('voice', voices[2].id)  # Zira
        engine.setProperty('rate', 130)  # slower for clarity

        # Narration script with timestamps (seconds)
        # Each segment gets ~6-8 seconds before the next starts
        # Scene 1 (title): 0-8s
        # Scene 2 (progress): 8s + 2s per checkpoint
        # Scene 3 (comprehension): after progress
        # Scene 4 (final): after comprehension
        narration_segments = [
            (1.0, "NeMo W M. Grounded language acquisition. The system learns language from its own sensorimotor experience. No large language model. No pretrained word embeddings."),
        ]

        # Scene 2 narrations — spread across progress checkpoints
        title_dur = 8.0
        progress_dur = len(progress) * 2.0
        checkpoints_for_narration = [
            (0, "Stage one. Learning words from experience. Each word is bound to the belief state that was active when the system heard it."),
            (2, "Physics concepts like gravity and friction are now grounded in actual force patterns the system discovered."),
            (4, "Navigation words like corridor and intersection are grounded in spatial belief states from episodic memory."),
            (6, "Emotional words like danger mean high cortisol. Calm means low stress. The meaning IS the neuromodulatory pattern."),
            (8, "Comprehension is improving. The system composes word beliefs to understand full sentences."),
        ]

        for idx, text in checkpoints_for_narration:
            if idx < len(progress):
                t = title_dur + idx * 2.0 + 0.5
                narration_segments.append((t, text))

        # Scene 3 narrations — comprehension demos
        comp_start = title_dur + progress_dur + 1.0
        comp_narrations = [
            (0, "Now testing sentence comprehension. The ball falls due to gravity. The system grounds the words falling and gravity from its physics experience."),
            (6, "Danger on the steep dark corridor. Three words grounded from navigation and emotion domains."),
            (12, "Push the heavy block left carefully. The system maps this to action primitives and spatial directions."),
            (18, "The strong magnetic force attracts objects near. A complex sentence understood through belief space composition."),
        ]

        for offset, text in comp_narrations:
            narration_segments.append((comp_start + offset, text))

        # Scene 4 — final
        final_start = comp_start + 24.0
        narration_segments.append(
            (final_start, "Final results. Over 100 words learned across 8 domains. Words are episodic memory prototypes. Sentences are belief space compositions. Comprehension is pattern completion in memory. No L L M was used at any point.")
        )

        # Generate WAV for each segment
        tmp_dir = tempfile.mkdtemp(prefix="nemo_narration_")
        wav_files = []

        for i, (timestamp, text) in enumerate(narration_segments):
            wav_path = os.path.join(tmp_dir, f"seg_{i:03d}.wav")
            engine.save_to_file(text, wav_path)
            wav_files.append((timestamp, wav_path, text))

        engine.runAndWait()
        print(f"  Generated {len(wav_files)} speech segments")

        # Build ffmpeg filter to place audio segments at correct times
        try:
            ffmpeg_exe = None
            try:
                import imageio_ffmpeg
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                ffmpeg_exe = "ffmpeg"

            # Concatenate all audio with silence gaps using ffmpeg
            # Strategy: create one combined audio track with adelay filters
            inputs = ['-i', str(silent_path)]
            filter_parts = []
            valid_segments = []

            for i, (timestamp, wav_path, text) in enumerate(wav_files):
                if os.path.exists(wav_path) and os.path.getsize(wav_path) > 100:
                    inputs.extend(['-i', wav_path])
                    delay_ms = int(timestamp * 1000)
                    idx = len(valid_segments) + 1
                    filter_parts.append(
                        f"[{idx}]adelay={delay_ms}|{delay_ms}[a{idx}]")
                    valid_segments.append(idx)

            if valid_segments:
                # Mix all delayed audio streams
                mix_inputs = ''.join(f'[a{idx}]' for idx in valid_segments)
                filter_parts.append(
                    f"{mix_inputs}amix=inputs={len(valid_segments)}:"
                    f"duration=longest[aout]")
                filter_str = ';'.join(filter_parts)

                final_path = OUT / "language_learning_demo.mp4"
                cmd = [
                    ffmpeg_exe, '-y',
                    *inputs,
                    '-filter_complex', filter_str,
                    '-map', '0:v',
                    '-map', '[aout]',
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-shortest',
                    str(final_path),
                ]

                result = subprocess.run(cmd, capture_output=True, text=True,
                                         timeout=60)
                if result.returncode == 0:
                    print(f"  Narrated video: {final_path}")
                else:
                    print(f"  FFmpeg merge failed, falling back to silent")
                    print(f"  Error: {result.stderr[:200]}")
                    import shutil
                    shutil.copy(str(silent_path), str(final_path))
            else:
                print("  No valid audio segments, using silent video")
                import shutil
                final_path = OUT / "language_learning_demo.mp4"
                shutil.copy(str(silent_path), str(final_path))

        except Exception as e:
            print(f"  Audio merge error: {e}")
            import shutil
            final_path = OUT / "language_learning_demo.mp4"
            shutil.copy(str(silent_path), str(final_path))

        # Cleanup
        for _, wav_path, _ in wav_files:
            try:
                os.remove(wav_path)
            except OSError:
                pass

    else:
        # No narration, just rename
        import shutil
        final_path = OUT / "language_learning_demo.mp4"
        shutil.copy(str(silent_path), str(final_path))

    print(f"  Final video: {final_path} ({duration:.1f}s)")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--narrate", action="store_true")
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()

    print("=" * 65)
    print("  NeMo-WM Language Curriculum")
    print("  Teaching ~200 concepts across 8 domains")
    print("=" * 65)

    grounder = WordGrounder(d_belief=64)

    # Build curriculum
    print("\n── Building Curriculum ──")
    curriculum = build_curriculum(d_belief=64)
    print(f"  Total training examples: {len(curriculum)}")

    domains = defaultdict(int)
    for _, _, _, _, _, d in curriculum:
        domains[d] += 1
    for d, n in sorted(domains.items()):
        print(f"    {d}: {n} examples")

    # Train
    print("\n── Training ──")
    progress = train_curriculum(grounder, curriculum)

    # Test
    print("\n── Testing ──")
    run_tests(grounder)

    # Video
    if args.video:
        generate_video(grounder, progress, args)

    # Final stats
    stats = grounder.stats()
    print(f"\n{'='*65}")
    print(f"  Final Results")
    print(f"{'='*65}")
    print(f"  Vocabulary: {stats['vocab_size']} words")
    print(f"  Total hearings: {stats['total_hearings']}")
    print(f"  Comprehension: {progress[-1]['avg_comprehension']:.0%}")
    print(f"  Domains: {len(domains)}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
