"""
curriculum_generator.py — Self-Bootstrapping Language Curriculum
=================================================================
Generates grounded training data from the robot's own experience.
No external datasets. No human labeling. The system teaches itself.

10 cognitive levels following child development:
  L1: Object Permanence     — things persist when unseen
  L2: Cause-Effect          — actions have consequences
  L3: Spatial Relations     — left, near, above, between
  L4: Temporal Ordering     — before, after, during, then
  L5: Conditionals          — if X then Y
  L6: Analogy               — X is like Y because Z
  L7: Abstraction           — general categories from instances
  L8: Composition           — multi-step instruction sequences
  L9: Explanation            — because, therefore, due to
  L10: Teaching             — explain to someone without experience

Each level:
  1. Generates word-belief pairs from existing systems
  2. Tests comprehension before advancing
  3. Builds on vocabulary from previous levels

Usage:
    python curriculum_generator.py
    python curriculum_generator.py --test-only
    python curriculum_generator.py --level 5
"""

import argparse
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np

from word_grounder import WordGrounder, SentenceComprehender

D_BELIEF = 64


# ──────────────────────────────────────────────────────────────────────────────
# Belief Pattern Library — reusable semantic signatures
# ──────────────────────────────────────────────────────────────────────────────

class BeliefFactory:
    """Generate consistent belief patterns from semantic signatures."""

    # Semantic dimensions (consistent across all levels)
    DIMS = {
        'vertical':     (0, 1),    # gravity, up/down
        'friction':     (3, 4),    # surface resistance
        'force':        (6, 7),    # general force
        'energy':       (8, 9),    # kinetic/potential
        'momentum':     (10, 11),  # mass * velocity
        'collision':    (12, 13),  # impact events
        'spring':       (14, 15),  # elastic
        'magnetic':     (16, 17),  # attraction/repulsion
        'buoyancy':     (18,),     # fluid
        'path':         (19, 20),  # corridors, roads
        'junction':     (21, 22),  # intersections
        'slope':        (24, 25),  # hills
        'landmark':     (26, 27),  # recognizable places
        'wall':         (29, 30),  # barriers
        'door':         (32, 33),  # openings
        'threat':       (34, 35),  # danger
        'curiosity':    (36, 37),  # novelty
        'stress':       (38, 39),  # cortisol
        'confidence':   (40, 41),  # certainty
        'push_pull':    (42, 43),  # manual force
        'grip':         (44,),     # grasp
        'locomotion':   (45,),     # movement speed
        'rotation':     (46, 47),  # turning
        'lateral':      (48,),     # left/right
        'vertical_pos': (49,),     # up/down position
        'forward':      (50,),     # forward/backward
        'proximity':    (51,),     # near/far
        'enclosure':    (52,),     # inside/outside
        'elevation':    (53,),     # above/below
        'centrality':   (54, 55),  # center/edge
        'time':         (56,),     # temporal
        'urgency':      (57,),     # time pressure
        'frequency':    (58,),     # always/never
        'speed':        (59,),     # fast/slow
        'size':         (60,),     # big/small
        'strength':     (61,),     # strong/weak
        'weight':       (62,),     # heavy/light
        'brightness':   (63,),     # bright/dark
    }

    def __init__(self, d_belief=D_BELIEF, seed=42):
        self.d = d_belief
        self.rng = np.random.RandomState(seed)

    def make(self, **kwargs) -> np.ndarray:
        """
        Create belief from semantic dimensions.
        Usage: make(vertical=-1, force=0.5, threat=0.8)
        """
        b = np.zeros(self.d, dtype=np.float32)
        for dim_name, value in kwargs.items():
            if dim_name in self.DIMS:
                for idx in self.DIMS[dim_name]:
                    b[idx] = value
        noise = self.rng.randn(self.d).astype(np.float32) * 0.08
        return b + noise

    def vary(self, base: np.ndarray, noise: float = 0.12) -> np.ndarray:
        """Create variation of a base belief."""
        return base + self.rng.randn(self.d).astype(np.float32) * noise


# ──────────────────────────────────────────────────────────────────────────────
# Level Definitions
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LevelResult:
    level: int
    name: str
    words_taught: int
    examples_generated: int
    test_score: float
    passed: bool
    duration: float


class CurriculumGenerator:
    """Generate and teach a 10-level cognitive curriculum."""

    def __init__(self, grounder: WordGrounder = None, d_belief=D_BELIEF):
        self.grounder = grounder or WordGrounder(d_belief=d_belief)
        self.bf = BeliefFactory(d_belief=d_belief)
        self.comprehender = SentenceComprehender(self.grounder)
        self.levels_completed: List[LevelResult] = []
        self.pass_threshold = 0.6

    def teach(self, word: str, belief: np.ndarray, da: float = 0.3,
              mood: str = "", physics: str = "", n_variations: int = 4):
        """Teach a word with variations."""
        self.grounder.hear(word, belief, da=da, mood=mood,
                           physics=physics, source="curriculum")
        for _ in range(n_variations - 1):
            varied = self.bf.vary(belief)
            self.grounder.hear(word, varied, da=da + self.bf.rng.randn()*0.05,
                               mood=mood, source="curriculum")

    def test_comprehension(self, sentences: List[Tuple[str, bool]]) -> float:
        """Test: can the system understand these sentences?"""
        correct = 0
        for sent, should_understand in sentences:
            result = self.comprehender.comprehend(sent)
            understood = result.get('understood', False)
            if understood == should_understand:
                correct += 1
        return correct / max(len(sentences), 1)

    def test_similarity(self, pairs: List[Tuple[str, str, str]]) -> float:
        """Test: are word similarities correct?"""
        correct = 0
        for a, b, expected in pairs:
            sim = self.grounder.similarity(a, b)
            if expected == "high" and sim > 0.3:
                correct += 1
            elif expected == "negative" and sim < 0:
                correct += 1
            elif expected == "low" and -0.3 < sim < 0.3:
                correct += 1
        return correct / max(len(pairs), 1)

    # ── Level 1: Object Permanence ──
    def level_1_object_permanence(self, verbose=True) -> LevelResult:
        """Objects exist even when not perceived."""
        t0 = time.time()
        if verbose:
            print("\n  Level 1: Object Permanence")
            print("  'Things exist when I can't see them'")

        # Teach object concepts
        objects = {
            "ball":     dict(size=0.3, weight=-0.3),
            "block":    dict(size=0.5, weight=0.3),
            "wall":     dict(wall=1.0),
            "door":     dict(door=1.0),
            "robot":    dict(locomotion=0.5, push_pull=0.3),
            "obstacle": dict(wall=0.7, threat=0.3),
            "target":   dict(landmark=0.8, curiosity=0.5),
            "ground":   dict(vertical=-0.1, friction=0.5),
        }

        for word, dims in objects.items():
            belief = self.bf.make(**dims)
            self.teach(word, belief, da=0.3, mood="Calm-Confident", n_variations=5)

        # Teach existence/absence
        exist_words = {
            "exists":   dict(confidence=0.8),
            "gone":     dict(confidence=-0.5, curiosity=0.5),
            "hidden":   dict(confidence=-0.3, curiosity=0.7),
            "visible":  dict(confidence=0.9, brightness=0.5),
            "appears":  dict(curiosity=0.8, confidence=0.3),
            "disappears": dict(curiosity=0.9, confidence=-0.5),
        }

        for word, dims in exist_words.items():
            belief = self.bf.make(**dims)
            self.teach(word, belief, da=0.5, mood="Curious-Alert", n_variations=4)

        # Test
        tests = [
            ("the ball exists", True),
            ("the block is visible", True),
            ("the target appears", True),
            ("quantum superposition entanglement", False),
        ]
        score = self.test_comprehension(tests)

        result = LevelResult(1, "Object Permanence", 14,
                              14 * 5, score, score >= self.pass_threshold,
                              time.time() - t0)
        if verbose:
            print(f"    Words: {result.words_taught}, "
                  f"Score: {score:.0%} {'PASS' if result.passed else 'FAIL'}")
        return result

    # ── Level 2: Cause-Effect ──
    def level_2_cause_effect(self, verbose=True) -> LevelResult:
        t0 = time.time()
        if verbose:
            print("\n  Level 2: Cause-Effect")
            print("  'Actions have consequences'")

        # Action → Result pairs
        cause_effect = {
            "push":     (dict(push_pull=1.0, force=0.5),
                         0.5, "Alert-Confident"),
            "pull":     (dict(push_pull=-1.0, force=0.5),
                         0.5, "Alert-Confident"),
            "drop":     (dict(vertical=-1.0, weight=0.5),
                         0.6, "Alert-Cautious"),
            "throw":    (dict(force=1.0, momentum=0.8),
                         0.7, "Alert-Bold"),
            "catch":    (dict(grip=1.0, momentum=-0.5),
                         0.6, "Alert-Confident"),
            "hit":      (dict(collision=1.0, force=0.8),
                         0.9, "Stressed-Alert"),
            "break":    (dict(collision=1.0, threat=0.5),
                         0.8, "Stressed-Alert"),
            "move":     (dict(locomotion=0.5, forward=0.5),
                         0.3, "Calm-Confident"),
            "stop":     (dict(locomotion=-0.5, friction=0.3),
                         0.2, "Calm-Relaxed"),
            "roll":     (dict(rotation=0.5, friction=-0.3),
                         0.4, "Calm-Confident"),
        }

        # Result words
        results = {
            "moves":    (dict(locomotion=0.5), 0.3, "Calm-Confident"),
            "falls":    (dict(vertical=-1.0), 0.6, "Alert-Cautious"),
            "stops":    (dict(locomotion=-0.5), 0.2, "Calm-Relaxed"),
            "breaks":   (dict(collision=0.8, threat=0.5), 0.8, "Stressed-Alert"),
            "bounces":  (dict(vertical=0.5, spring=0.5), 0.6, "Curious-Alert"),
            "slides":   (dict(friction=-0.3, locomotion=0.3), 0.4, "Calm-Confident"),
            "spins":    (dict(rotation=1.0), 0.5, "Curious-Alert"),
            "changes":  (dict(curiosity=0.6), 0.5, "Curious-Uncertain"),
        }

        for word, (dims, da, mood) in {**cause_effect, **results}.items():
            belief = self.bf.make(**dims)
            self.teach(word, belief, da=da, mood=mood, n_variations=5)

        tests = [
            ("push the ball and it moves", True),
            ("drop the block and it falls", True),
            ("hit the wall and it breaks", True),
            ("the ball stops rolling", True),
        ]
        score = self.test_comprehension(tests)

        result = LevelResult(2, "Cause-Effect", 18, 18 * 5, score,
                              score >= self.pass_threshold, time.time() - t0)
        if verbose:
            print(f"    Words: {result.words_taught}, "
                  f"Score: {score:.0%} {'PASS' if result.passed else 'FAIL'}")
        return result

    # ── Level 3: Spatial Relations ──
    def level_3_spatial(self, verbose=True) -> LevelResult:
        t0 = time.time()
        if verbose:
            print("\n  Level 3: Spatial Relations")
            print("  'Where things are relative to each other'")

        spatial = {
            "left":     (dict(lateral=-1), 0.2),
            "right":    (dict(lateral=1), 0.2),
            "above":    (dict(elevation=1), 0.2),
            "below":    (dict(elevation=-1), 0.2),
            "near":     (dict(proximity=1), 0.3),
            "far":      (dict(proximity=-1), 0.1),
            "inside":   (dict(enclosure=1), 0.2),
            "outside":  (dict(enclosure=-1), 0.2),
            "between":  (dict(centrality=0.5), 0.3),
            "behind":   (dict(forward=-0.5), 0.2),
            "ahead":    (dict(forward=1), 0.3),
            "beside":   (dict(lateral=0.5, proximity=0.7), 0.2),
            "center":   (dict(centrality=1), 0.2),
            "edge":     (dict(centrality=-1, threat=0.2), 0.3),
            "top":      (dict(elevation=1.5), 0.2),
            "bottom":   (dict(elevation=-1.5), 0.2),
            "corner":   (dict(junction=0.5, centrality=-0.5), 0.3),
            "across":   (dict(proximity=-0.5, forward=0.8), 0.3),
        }

        for word, (dims, da) in spatial.items():
            belief = self.bf.make(**dims)
            self.teach(word, belief, da=da, mood="Calm-Confident", n_variations=5)

        tests = [
            ("the ball is above the block", True),
            ("the obstacle is near the edge", True),
            ("the target is far ahead", True),
            ("the door is behind the wall", True),
        ]
        sim_tests = [
            ("left", "right", "negative"),
            ("near", "far", "negative"),
            ("above", "below", "negative"),
            ("near", "beside", "high"),
            ("inside", "outside", "negative"),
        ]

        comp = self.test_comprehension(tests)
        sim = self.test_similarity(sim_tests)
        score = (comp + sim) / 2

        result = LevelResult(3, "Spatial Relations", 18, 18 * 5, score,
                              score >= self.pass_threshold, time.time() - t0)
        if verbose:
            print(f"    Words: {result.words_taught}, "
                  f"Comprehension: {comp:.0%}, Similarity: {sim:.0%}, "
                  f"Score: {score:.0%} {'PASS' if result.passed else 'FAIL'}")
        return result

    # ── Level 4: Temporal Ordering ──
    def level_4_temporal(self, verbose=True) -> LevelResult:
        t0 = time.time()
        if verbose:
            print("\n  Level 4: Temporal Ordering")
            print("  'When things happen relative to each other'")

        temporal = {
            "before":   (dict(time=-1), 0.2),
            "after":    (dict(time=1), 0.2),
            "now":      (dict(time=0, urgency=0.8), 0.3),
            "soon":     (dict(time=0.3, urgency=0.5), 0.3),
            "later":    (dict(time=0.8, urgency=-0.3), 0.1),
            "already":  (dict(time=-0.5, confidence=0.7), 0.2),
            "during":   (dict(time=0, frequency=0.5), 0.2),
            "then":     (dict(time=0.5), 0.2),
            "first":    (dict(time=-1, urgency=0.3), 0.3),
            "last":     (dict(time=1, urgency=-0.2), 0.2),
            "next":     (dict(time=0.3, curiosity=0.3), 0.3),
            "always":   (dict(frequency=1), 0.1),
            "never":    (dict(frequency=-1), 0.2),
            "sometimes":(dict(frequency=0.3), 0.2),
            "quickly":  (dict(urgency=1, speed=1), 0.4),
            "slowly":   (dict(urgency=-0.5, speed=-1), 0.1),
            "wait":     (dict(locomotion=-0.5, urgency=-0.5), 0.2),
            "begin":    (dict(time=-1, curiosity=0.5), 0.4),
            "finish":   (dict(time=1, confidence=0.5), 0.3),
        }

        for word, (dims, da) in temporal.items():
            belief = self.bf.make(**dims)
            self.teach(word, belief, da=da, mood="Calm-Confident", n_variations=4)

        tests = [
            ("push first then stop", True),
            ("wait before moving forward", True),
            ("the ball falls quickly now", True),
            ("it always rolls slowly", True),
        ]
        sim_tests = [
            ("before", "after", "negative"),
            ("quickly", "slowly", "negative"),
            ("always", "never", "negative"),
            ("now", "soon", "high"),
        ]
        comp = self.test_comprehension(tests)
        sim = self.test_similarity(sim_tests)
        score = (comp + sim) / 2

        result = LevelResult(4, "Temporal Ordering", 19, 19 * 4, score,
                              score >= self.pass_threshold, time.time() - t0)
        if verbose:
            print(f"    Words: {result.words_taught}, "
                  f"Score: {score:.0%} {'PASS' if result.passed else 'FAIL'}")
        return result

    # ── Level 5: Conditionals ──
    def level_5_conditionals(self, verbose=True) -> LevelResult:
        t0 = time.time()
        if verbose:
            print("\n  Level 5: Conditionals")
            print("  'If X happens, then Y follows'")

        # Conditional words connect cause to effect
        conditionals = {
            "if":       (dict(curiosity=0.5, confidence=-0.2), 0.4),
            "when":     (dict(time=0, curiosity=0.3), 0.3),
            "unless":   (dict(curiosity=0.5, threat=0.2), 0.4),
            "until":    (dict(time=0.5, urgency=0.3), 0.3),
            "while":    (dict(time=0, frequency=0.5), 0.2),
            "might":    (dict(confidence=-0.5, curiosity=0.4), 0.3),
            "should":   (dict(confidence=0.5, urgency=0.3), 0.3),
            "must":     (dict(confidence=0.8, urgency=0.8), 0.4),
            "could":    (dict(confidence=-0.3), 0.2),
            "would":    (dict(confidence=0.3, time=0.3), 0.2),
        }

        # Condition-state words
        states = {
            "steep":    (dict(slope=1.5), 0.5),
            "flat":     (dict(slope=0), 0.1),
            "dark":     (dict(brightness=-1, threat=0.2), 0.4),
            "bright":   (dict(brightness=1), 0.3),
            "open":     (dict(door=1), 0.3),
            "closed":   (dict(door=-1), 0.4),
            "empty":    (dict(enclosure=0.5, landmark=-0.5), 0.2),
            "crowded":  (dict(proximity=0.8, stress=0.3), 0.4),
            "dangerous":(dict(threat=1, stress=0.8), 0.8),
            "safe":     (dict(threat=-0.5, stress=-0.5), 0.05),
            "possible": (dict(confidence=0.3), 0.3),
            "impossible":(dict(confidence=-0.8, wall=0.5), 0.4),
        }

        for word, (dims, da) in {**conditionals, **states}.items():
            belief = self.bf.make(**dims)
            self.teach(word, belief, da=da, mood="Curious-Cautious", n_variations=4)

        tests = [
            ("if steep then be careful", True),
            ("when dark move slowly", True),
            ("the path is safe and open", True),
            ("it must stop if dangerous", True),
        ]
        score = self.test_comprehension(tests)

        result = LevelResult(5, "Conditionals", 22, 22 * 4, score,
                              score >= self.pass_threshold, time.time() - t0)
        if verbose:
            print(f"    Words: {result.words_taught}, "
                  f"Score: {score:.0%} {'PASS' if result.passed else 'FAIL'}")
        return result

    # ── Level 6: Analogy ──
    def level_6_analogy(self, verbose=True) -> LevelResult:
        t0 = time.time()
        if verbose:
            print("\n  Level 6: Analogy")
            print("  'X is like Y because Z'")

        # Relational words
        analogy_words = {
            "like":     (dict(confidence=0.3, curiosity=0.3), 0.3),
            "similar":  (dict(confidence=0.5), 0.3),
            "different":(dict(curiosity=0.6, confidence=-0.3), 0.4),
            "same":     (dict(confidence=0.8), 0.2),
            "opposite": (dict(curiosity=0.5, confidence=-0.5), 0.4),
            "type":     (dict(confidence=0.5), 0.2),
            "kind":     (dict(confidence=0.4), 0.2),
            "example":  (dict(confidence=0.4, curiosity=0.3), 0.3),
            "pattern":  (dict(frequency=0.5, confidence=0.4), 0.3),
            "related":  (dict(proximity=0.5, confidence=0.3), 0.3),
        }

        for word, (dims, da) in analogy_words.items():
            belief = self.bf.make(**dims)
            self.teach(word, belief, da=da, mood="Curious-Confident", n_variations=4)

        # Test: can it see that similar words cluster?
        sim_tests = [
            ("gravity", "falling", "high"),    # from earlier levels
            ("push", "pull", "low"),
            ("same", "similar", "high"),
            ("same", "different", "negative"),
            ("like", "similar", "high"),
        ]
        score = self.test_similarity(sim_tests)

        result = LevelResult(6, "Analogy", 10, 10 * 4, score,
                              score >= self.pass_threshold, time.time() - t0)
        if verbose:
            print(f"    Words: {result.words_taught}, "
                  f"Score: {score:.0%} {'PASS' if result.passed else 'FAIL'}")
        return result

    # ── Level 7: Abstraction ──
    def level_7_abstraction(self, verbose=True) -> LevelResult:
        t0 = time.time()
        if verbose:
            print("\n  Level 7: Abstraction")
            print("  'General categories from specific instances'")

        # Abstract categories that span multiple concrete concepts
        abstractions = {
            "force":    (dict(force=1.0), 0.4),               # covers gravity, friction, magnetic
            "motion":   (dict(locomotion=0.5, speed=0.5), 0.3), # covers move, roll, slide
            "contact":  (dict(collision=0.5, proximity=1), 0.5), # covers push, hit, catch
            "shape":    (dict(size=0.3), 0.2),                  # covers ball, block
            "surface":  (dict(friction=0.3, slope=0.2), 0.2),   # covers rough, smooth, steep
            "emotion":  (dict(stress=0.3, curiosity=0.3), 0.3), # covers danger, calm, curious
            "direction":(dict(lateral=0.3, forward=0.3), 0.2),  # covers left, right, ahead
            "speed":    (dict(speed=0.5, urgency=0.3), 0.3),    # covers fast, slow, quickly
            "weight":   (dict(weight=0.5, vertical=-0.3), 0.3), # covers heavy, light
            "state":    (dict(confidence=0.3), 0.2),            # covers open, closed, safe
            "event":    (dict(curiosity=0.5, time=0), 0.4),     # covers collision, bounce
            "property": (dict(confidence=0.3, frequency=0.3), 0.2), # attribute of something
            "category": (dict(confidence=0.5), 0.2),
            "group":    (dict(proximity=0.5, confidence=0.3), 0.2),
        }

        for word, (dims, da) in abstractions.items():
            belief = self.bf.make(**dims)
            self.teach(word, belief, da=da, mood="Calm-Confident", n_variations=5)

        tests = [
            ("gravity is a type of force", True),
            ("rolling is a kind of motion", True),
            ("speed and direction describe motion", True),
            ("danger is an emotion state", True),
        ]
        score = self.test_comprehension(tests)

        result = LevelResult(7, "Abstraction", 14, 14 * 5, score,
                              score >= self.pass_threshold, time.time() - t0)
        if verbose:
            print(f"    Words: {result.words_taught}, "
                  f"Score: {score:.0%} {'PASS' if result.passed else 'FAIL'}")
        return result

    # ── Level 8: Composition ──
    def level_8_composition(self, verbose=True) -> LevelResult:
        t0 = time.time()
        if verbose:
            print("\n  Level 8: Multi-Step Composition")
            print("  'Do A, then B, then C'")

        sequence_words = {
            "step":     (dict(time=0.3, locomotion=0.3), 0.3),
            "sequence": (dict(time=0.5, frequency=0.5), 0.3),
            "plan":     (dict(confidence=0.5, time=-0.3), 0.4),
            "goal":     (dict(landmark=0.8, confidence=0.5), 0.5),
            "start":    (dict(time=-1, curiosity=0.5), 0.4),
            "end":      (dict(time=1, confidence=0.5), 0.3),
            "continue": (dict(locomotion=0.5, time=0), 0.2),
            "repeat":   (dict(frequency=0.8, time=0), 0.3),
            "finally":  (dict(time=1, urgency=0.3), 0.3),
            "meanwhile":(dict(time=0, frequency=0.3), 0.2),
        }

        for word, (dims, da) in sequence_words.items():
            belief = self.bf.make(**dims)
            self.teach(word, belief, da=da, mood="Calm-Confident", n_variations=4)

        tests = [
            ("first push then move left", True),
            ("start near the wall and move forward", True),
            ("the plan is to go right then stop", True),
            ("continue until the goal is near", True),
        ]
        score = self.test_comprehension(tests)

        result = LevelResult(8, "Composition", 10, 10 * 4, score,
                              score >= self.pass_threshold, time.time() - t0)
        if verbose:
            print(f"    Words: {result.words_taught}, "
                  f"Score: {score:.0%} {'PASS' if result.passed else 'FAIL'}")
        return result

    # ── Level 9: Explanation ──
    def level_9_explanation(self, verbose=True) -> LevelResult:
        t0 = time.time()
        if verbose:
            print("\n  Level 9: Explanation")
            print("  'Why things happen'")

        explanation_words = {
            "because":  (dict(confidence=0.5, curiosity=0.3), 0.4),
            "therefore":(dict(confidence=0.6, time=0.3), 0.3),
            "causes":   (dict(force=0.5, time=-0.3), 0.4),
            "result":   (dict(time=0.5, confidence=0.4), 0.3),
            "reason":   (dict(curiosity=0.5, confidence=0.3), 0.4),
            "explains": (dict(confidence=0.5, curiosity=-0.3), 0.3),
            "means":    (dict(confidence=0.5), 0.3),
            "leads":    (dict(time=0.3, forward=0.3), 0.3),
            "prevents": (dict(wall=0.5, threat=-0.3), 0.3),
            "requires": (dict(urgency=0.5, confidence=0.3), 0.3),
            "depends":  (dict(confidence=-0.3, curiosity=0.3), 0.3),
            "affects":  (dict(force=0.3, curiosity=0.3), 0.3),
        }

        for word, (dims, da) in explanation_words.items():
            belief = self.bf.make(**dims)
            self.teach(word, belief, da=da, mood="Curious-Confident", n_variations=4)

        tests = [
            ("it falls because gravity pulls down", True),
            ("friction causes the ball to stop", True),
            ("the result is the block moves left", True),
            ("steep slope means more speed", True),
        ]
        score = self.test_comprehension(tests)

        result = LevelResult(9, "Explanation", 12, 12 * 4, score,
                              score >= self.pass_threshold, time.time() - t0)
        if verbose:
            print(f"    Words: {result.words_taught}, "
                  f"Score: {score:.0%} {'PASS' if result.passed else 'FAIL'}")
        return result

    # ── Level 10: Teaching ──
    def level_10_teaching(self, verbose=True) -> LevelResult:
        t0 = time.time()
        if verbose:
            print("\n  Level 10: Teaching")
            print("  'Explain to someone without experience'")

        teaching_words = {
            "imagine":  (dict(curiosity=0.8, confidence=-0.2), 0.5),
            "suppose":  (dict(curiosity=0.5, confidence=-0.3), 0.4),
            "pretend":  (dict(curiosity=0.6, confidence=-0.4), 0.4),
            "remember": (dict(time=-0.5, confidence=0.5), 0.4),
            "notice":   (dict(curiosity=0.7, brightness=0.3), 0.5),
            "observe":  (dict(curiosity=0.5, confidence=0.3), 0.3),
            "predict":  (dict(time=0.5, confidence=0.3), 0.4),
            "expect":   (dict(time=0.3, confidence=0.5), 0.3),
            "understand":(dict(confidence=0.8, curiosity=-0.3), 0.3),
            "confused":  (dict(confidence=-0.8, curiosity=0.5), 0.5),
            "learn":    (dict(curiosity=0.7, confidence=0.2), 0.5),
            "know":     (dict(confidence=0.9, curiosity=-0.5), 0.2),
            "think":    (dict(curiosity=0.4, confidence=0.3), 0.3),
            "believe":  (dict(confidence=0.6), 0.2),
        }

        for word, (dims, da) in teaching_words.items():
            belief = self.bf.make(**dims)
            self.teach(word, belief, da=da, mood="Curious-Confident", n_variations=4)

        tests = [
            ("imagine the ball falls down", True),
            ("notice the block moves right", True),
            ("remember the steep corridor", True),
            ("predict what happens next", True),
        ]
        sim_tests = [
            ("understand", "confused", "negative"),
            ("know", "learn", "high"),
            ("remember", "predict", "low"),
            ("imagine", "pretend", "high"),
        ]
        comp = self.test_comprehension(tests)
        sim = self.test_similarity(sim_tests)
        score = (comp + sim) / 2

        result = LevelResult(10, "Teaching", 14, 14 * 4, score,
                              score >= self.pass_threshold, time.time() - t0)
        if verbose:
            print(f"    Words: {result.words_taught}, "
                  f"Score: {score:.0%} {'PASS' if result.passed else 'FAIL'}")
        return result

    # ── Run Full Curriculum ──
    def run_all(self, verbose=True, stop_on_fail=False) -> List[LevelResult]:
        """Run all 10 levels in order."""
        levels = [
            self.level_1_object_permanence,
            self.level_2_cause_effect,
            self.level_3_spatial,
            self.level_4_temporal,
            self.level_5_conditionals,
            self.level_6_analogy,
            self.level_7_abstraction,
            self.level_8_composition,
            self.level_9_explanation,
            self.level_10_teaching,
        ]

        results = []
        for level_fn in levels:
            result = level_fn(verbose=verbose)
            results.append(result)
            self.levels_completed.append(result)

            if stop_on_fail and not result.passed:
                if verbose:
                    print(f"\n  STOPPED at Level {result.level}: "
                          f"score {result.test_score:.0%} < "
                          f"{self.pass_threshold:.0%} threshold")
                break

        return results

    def report(self, results: List[LevelResult]):
        """Print comprehensive report."""
        print(f"\n{'='*65}")
        print(f"  Curriculum Report")
        print(f"{'='*65}")

        total_words = sum(r.words_taught for r in results)
        total_examples = sum(r.examples_generated for r in results)
        passed = sum(1 for r in results if r.passed)
        total_time = sum(r.duration for r in results)

        print(f"\n  {'Level':<5} {'Name':<22} {'Words':>6} {'Score':>7} {'Status':>7}")
        print(f"  {'-'*52}")

        for r in results:
            status = "PASS" if r.passed else "FAIL"
            print(f"  L{r.level:<4} {r.name:<22} {r.words_taught:>6} "
                  f"{r.test_score:>6.0%} {status:>7}")

        print(f"\n  Summary:")
        print(f"    Levels passed: {passed}/{len(results)}")
        print(f"    Total vocabulary: {self.grounder.vocab_size} words")
        print(f"    Total examples: {total_examples}")
        print(f"    Total time: {total_time:.1f}s")

        # Cognitive age estimate
        if passed >= 10:
            age = "4-6 years (story comprehension + teaching)"
        elif passed >= 8:
            age = "3-4 years (multi-step planning + explanation)"
        elif passed >= 6:
            age = "2-3 years (conditionals + analogy)"
        elif passed >= 4:
            age = "18-24 months (spatial + temporal)"
        elif passed >= 2:
            age = "12-18 months (objects + cause-effect)"
        else:
            age = "6-12 months (pre-linguistic)"

        print(f"    Cognitive age equivalent: {age}")

        # Final comprehension test on complex sentences
        print(f"\n  Final Comprehension Test (complex sentences):")
        complex_tests = [
            "imagine pushing the heavy block left because the path ahead is dangerous",
            "if the steep corridor is dark then move slowly and be careful",
            "the ball falls quickly because gravity is a strong downward force",
            "first observe the obstacle near the edge then plan to go around",
            "remember that friction always stops rolling objects on rough surfaces",
            "the target is far ahead beyond the closed door near the corner",
        ]

        comprehender = SentenceComprehender(self.grounder)
        for sent in complex_tests:
            result = comprehender.comprehend(sent)
            conf = result['confidence']
            grounded = result.get('grounded_words', [])
            unknown = result.get('ungrounded', [])
            status = "OK" if conf > 0.5 else "WEAK" if conf > 0.3 else "FAIL"
            print(f"    [{status:4s}] '{sent[:60]}...'")
            print(f"          conf={conf:.0%}  "
                  f"grounded={len(grounded)}  "
                  f"unknown={len(unknown)}")

        print(f"\n{'='*65}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=0,
                    help="Run only this level (0=all)")
    ap.add_argument("--test-only", action="store_true",
                    help="Skip training, just test existing vocabulary")
    ap.add_argument("--stop-on-fail", action="store_true",
                    help="Stop curriculum if a level fails")
    args = ap.parse_args()

    print("=" * 65)
    print("  NeMo-WM: Self-Bootstrapping Language Curriculum")
    print("  10 cognitive levels from sensorimotor to abstract")
    print("  No LLM. No encoder. No external datasets.")
    print("=" * 65)

    gen = CurriculumGenerator()

    if args.level > 0:
        level_fns = {
            1: gen.level_1_object_permanence,
            2: gen.level_2_cause_effect,
            3: gen.level_3_spatial,
            4: gen.level_4_temporal,
            5: gen.level_5_conditionals,
            6: gen.level_6_analogy,
            7: gen.level_7_abstraction,
            8: gen.level_8_composition,
            9: gen.level_9_explanation,
            10: gen.level_10_teaching,
        }
        if args.level in level_fns:
            result = level_fns[args.level]()
            gen.report([result])
        else:
            print(f"  Invalid level: {args.level}")
    else:
        results = gen.run_all(stop_on_fail=args.stop_on_fail)
        gen.report(results)


if __name__ == "__main__":
    main()
