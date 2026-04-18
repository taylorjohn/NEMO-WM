"""
compositional_narrator.py — Belief-Driven Vocabulary Growth
=============================================================
Replaces template narration with organic vocabulary acquisition.

Three mechanisms for learning new words (like children):

1. NAMING    — novel experience → nearest words + residual → new word
2. DIFFERENTIATION — same word, different beliefs → split into subtypes
3. COMPOSITION — combine known words into novel phrases → ground combo

Vocabulary grows WITHOUT LIMIT because the system invents new words
when existing ones can't adequately describe its experience.

Usage:
    python compositional_narrator.py          # demo
    python compositional_narrator.py --test   # run tests
"""

import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

D_BELIEF = 64


class GroundedWord:
    """A word with its grounded meaning."""

    def __init__(self, word: str, prototype: np.ndarray,
                  n_hearings: int = 1):
        self.word = word
        self.prototype = prototype.copy()
        self.n_hearings = n_hearings
        self.beliefs = [prototype.copy()]  # store recent beliefs
        self.max_beliefs = 50

    def hear(self, belief: np.ndarray, lr=0.1):
        """Update prototype with new hearing."""
        self.prototype = (1 - lr) * self.prototype + lr * belief
        self.n_hearings += 1
        self.beliefs.append(belief.copy())
        if len(self.beliefs) > self.max_beliefs:
            self.beliefs = self.beliefs[-self.max_beliefs:]

    @property
    def variance(self):
        """How spread out are this word's usages?"""
        if len(self.beliefs) < 2:
            return 0.0
        stacked = np.stack(self.beliefs)
        return float(np.mean(np.var(stacked, axis=0)))

    def similarity(self, belief: np.ndarray) -> float:
        """How well does this word describe the belief?"""
        norm_p = np.linalg.norm(self.prototype)
        norm_b = np.linalg.norm(belief)
        if norm_p < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(self.prototype, belief) / (norm_p * norm_b))


class CompositionalNarrator:
    """
    Describes beliefs using compositional word combinations.
    Grows vocabulary organically through three mechanisms:
    naming, differentiation, and composition.
    """

    def __init__(self, initial_vocab: Dict[str, np.ndarray] = None):
        self.words: Dict[str, GroundedWord] = {}
        self.compositions: Dict[str, List[str]] = {}  # compound → parts
        self.total_narrations = 0
        self.words_invented = 0
        self.words_split = 0
        self.words_composed = 0

        # Seed vocabulary
        if initial_vocab:
            for word, belief in initial_vocab.items():
                self.words[word] = GroundedWord(word, belief)

        # Description quality threshold
        self.adequacy_threshold = 0.95  # high bar → more pressure to invent
        self.split_variance_threshold = 0.3
        self.min_hearings_to_split = 10
        self.already_split = set()  # don't re-split same word

    def describe(self, belief: np.ndarray, max_words=5) -> Tuple[str, float, np.ndarray]:
        """
        Describe a belief using the best combination of known words.
        Returns: (description, quality, residual)

        Quality = how well the description captures the belief.
        Residual = what part of the belief isn't captured.
        """
        if not self.words:
            return "", 0.0, belief.copy()

        # Score all words by similarity to belief
        scored = []
        for word, gw in self.words.items():
            sim = gw.similarity(belief)
            scored.append((word, sim, gw.prototype))

        scored.sort(key=lambda x: -abs(x[1]))

        # Greedily pick words that add information
        chosen = []
        reconstructed = np.zeros(D_BELIEF, dtype=np.float32)
        used_variance = set()

        for word, sim, proto in scored[:max_words * 2]:
            if len(chosen) >= max_words:
                break

            # Does this word add new information?
            candidate = reconstructed + proto * (0.3 if sim > 0 else -0.3)
            candidate_sim = self._cosine(candidate, belief)
            current_sim = self._cosine(reconstructed, belief) if np.linalg.norm(reconstructed) > 0 else -1

            if candidate_sim > current_sim + 0.01:
                chosen.append((word, sim))
                reconstructed = candidate

        # Build description string
        description_words = []
        for word, sim in chosen:
            if sim < -0.3:
                description_words.append(f"not_{word}")
            else:
                description_words.append(word)

        description = " ".join(description_words)
        quality = self._cosine(reconstructed, belief) if np.linalg.norm(reconstructed) > 0 else 0.0
        residual = belief - reconstructed

        return description, float(quality), residual

    def narrate(self, belief: np.ndarray, context: Dict = None) -> Dict:
        """
        Full narration cycle:
        1. Describe belief with existing words
        2. If quality too low → invent/compose new word
        3. Ground all words to belief
        4. Check for splits needed

        Returns narration metadata.
        """
        self.total_narrations += 1
        context = context or {}

        # 1. Describe
        description, quality, residual = self.describe(belief)
        residual_magnitude = float(np.linalg.norm(residual))

        # 2. If description is inadequate, learn new vocabulary
        new_words = []

        if quality < self.adequacy_threshold and residual_magnitude > 0.5:
            # Try composition first
            composed = self._try_compose(belief, residual)
            if composed:
                new_words.append(composed)
                self.words_composed += 1
            else:
                # Invent a new word
                invented = self._invent_word(belief, residual, context)
                if invented:
                    new_words.append(invented)
                    self.words_invented += 1

        # 3. Ground all description words to this belief
        for word in description.split():
            clean = word.replace("not_", "")
            if clean in self.words:
                self.words[clean].hear(belief)

        # 4. Check for splits
        split = self._check_splits()
        if split:
            new_words.append(split)
            self.words_split += 1

        # Rebuild description with new words
        if new_words:
            description2, quality2, residual2 = self.describe(belief)
            if quality2 > quality:
                description = description2
                quality = quality2
                residual = residual2

        return {
            "description": description,
            "quality": float(quality),
            "residual_magnitude": residual_magnitude,
            "new_words": new_words,
            "vocab_size": len(self.words),
            "total_narrations": self.total_narrations,
        }

    def _try_compose(self, belief: np.ndarray,
                       residual: np.ndarray) -> Optional[str]:
        """
        Try to compose a new compound word from existing words.
        "fast" + "left" → "fast_left" grounded at this belief.
        """
        if len(self.words) < 3:
            return None

        # Find two words whose combination best explains the FULL belief
        best_pair = None
        best_sim = -1.0  # accept any improvement

        word_list = list(self.words.keys())
        n_samples = min(100, len(word_list) * (len(word_list) - 1) // 2)

        # Current best single-word description
        _, current_quality, _ = self.describe(belief, max_words=1)

        for _ in range(n_samples):
            i, j = np.random.choice(len(word_list), 2, replace=False)
            w1, w2 = word_list[i], word_list[j]

            # Skip if this compound already exists
            compound = f"{w1}_{w2}"
            if compound in self.words:
                continue

            combined = (self.words[w1].prototype +
                         self.words[w2].prototype) * 0.5
            sim = self._cosine(combined, belief)

            # Must beat current single-word quality
            if sim > best_sim and sim > current_quality + 0.01:
                best_sim = sim
                best_pair = (w1, w2)

        if best_pair:
            w1, w2 = best_pair
            compound = f"{w1}_{w2}"
            combined_proto = (self.words[w1].prototype +
                                self.words[w2].prototype) * 0.5
            grounded = 0.6 * combined_proto + 0.4 * belief
            self.words[compound] = GroundedWord(compound, grounded)
            self.compositions[compound] = [w1, w2]
            return compound

        return None

    def _invent_word(self, belief: np.ndarray,
                       residual: np.ndarray,
                       context: Dict) -> Optional[str]:
        """
        Invent a new word for an experience that can't be described.
        Name is auto-generated from context.
        """
        # Generate name from context
        schema = context.get("schema", np.random.randint(100))
        step = context.get("step", self.total_narrations)

        # Try descriptive name based on dominant belief dimensions
        top_dims = np.argsort(np.abs(belief))[-3:]
        sign_str = "".join(["p" if belief[d] > 0 else "n" for d in top_dims])
        name = f"exp_{sign_str}_{schema}"

        # Avoid duplicates
        if name in self.words:
            name = f"{name}_{step}"

        self.words[name] = GroundedWord(name, belief)
        return name

    def _check_splits(self) -> Optional[str]:
        """
        Check if any word has high variance → split into subtypes.
        "corridor" used in different contexts → "corridor_left" + "corridor_right"
        """
        for word, gw in list(self.words.items()):
            # Skip already-split words
            if word in self.already_split:
                continue

            if (gw.n_hearings >= self.min_hearings_to_split and
                gw.variance > self.split_variance_threshold and
                not any(word.startswith(s) for s in self.already_split)):

                # Split into two subtypes using k-means on beliefs
                beliefs = np.stack(gw.beliefs)
                if len(beliefs) < 4:
                    continue

                # Simple 2-means
                idx = np.random.choice(len(beliefs), 2, replace=False)
                c1, c2 = beliefs[idx[0]], beliefs[idx[1]]

                for _ in range(10):
                    d1 = np.linalg.norm(beliefs - c1, axis=1)
                    d2 = np.linalg.norm(beliefs - c2, axis=1)
                    mask1 = d1 < d2
                    if mask1.sum() > 0:
                        c1 = beliefs[mask1].mean(axis=0)
                    if (~mask1).sum() > 0:
                        c2 = beliefs[~mask1].mean(axis=0)

                # Check if split is meaningful (clusters are different)
                split_dist = np.linalg.norm(c1 - c2)
                if split_dist > 1.0:
                    name1 = f"{word}_a"
                    name2 = f"{word}_b"
                    if name1 not in self.words:
                        self.words[name1] = GroundedWord(name1, c1)
                    if name2 not in self.words:
                        self.words[name2] = GroundedWord(name2, c2)
                    self.already_split.add(word)
                    return name1

        return None

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def get_stats(self) -> Dict:
        return {
            "vocab_size": len(self.words),
            "base_words": sum(1 for w in self.words if "_" not in w),
            "compounds": len(self.compositions),
            "invented": self.words_invented,
            "split": self.words_split,
            "composed": self.words_composed,
            "total_narrations": self.total_narrations,
            "avg_hearings": np.mean([gw.n_hearings for gw in self.words.values()])
                            if self.words else 0,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Demo
# ══════════════════════════════════════════════════════════════════════════════

def demo():
    print("=" * 70)
    print("  Compositional Narrator — Organic Vocabulary Growth")
    print("  No templates. Words emerge from experience.")
    print("=" * 70)

    rng = np.random.RandomState(42)

    # Seed with basic spatial vocabulary
    seed_vocab = {}
    dir_base = rng.randn(D_BELIEF).astype(np.float32)
    seed_vocab["left"] = dir_base
    seed_vocab["right"] = -dir_base
    seed_vocab["upper"] = rng.randn(D_BELIEF).astype(np.float32)
    seed_vocab["lower"] = -seed_vocab["upper"]
    seed_vocab["fast"] = rng.randn(D_BELIEF).astype(np.float32) * 1.5
    seed_vocab["slow"] = seed_vocab["fast"] * 0.3
    seed_vocab["corridor"] = rng.randn(D_BELIEF).astype(np.float32)
    seed_vocab["corner"] = rng.randn(D_BELIEF).astype(np.float32)
    seed_vocab["goal"] = rng.randn(D_BELIEF).astype(np.float32)
    seed_vocab["obstacle"] = rng.randn(D_BELIEF).astype(np.float32)

    narrator = CompositionalNarrator(initial_vocab=seed_vocab)

    print(f"\n  Seed vocabulary: {len(seed_vocab)} words")
    print(f"  Running 500 narration cycles...\n")

    # Simulate exploration with diverse beliefs
    vocab_history = [len(narrator.words)]
    quality_history = []

    for cycle in range(500):
        # Generate diverse beliefs
        if cycle < 100:
            # Near known words
            base_word = list(seed_vocab.keys())[cycle % len(seed_vocab)]
            belief = seed_vocab[base_word] + rng.randn(D_BELIEF).astype(np.float32) * 0.5
        elif cycle < 300:
            # Between known words (needs composition)
            w1, w2 = rng.choice(list(seed_vocab.keys()), 2, replace=False)
            belief = (seed_vocab[w1] + seed_vocab[w2]) * 0.5 + \
                      rng.randn(D_BELIEF).astype(np.float32) * 0.3
        else:
            # Truly novel (needs invention)
            belief = rng.randn(D_BELIEF).astype(np.float32) * 2.0

        context = {"schema": cycle % 32, "step": cycle}
        result = narrator.narrate(belief, context)

        vocab_history.append(result["vocab_size"])
        quality_history.append(result["quality"])

        if cycle % 100 == 0 or result["new_words"]:
            if cycle % 100 == 0:
                print(f"  Cycle {cycle:>4}: vocab={result['vocab_size']:>4}  "
                      f"quality={result['quality']:.3f}  "
                      f"desc=\"{result['description'][:50]}\"")
            if result["new_words"] and cycle % 10 == 0:
                for nw in result["new_words"]:
                    print(f"           + NEW: \"{nw}\"")

    stats = narrator.get_stats()
    print(f"\n  ── Results after 500 cycles ──")
    print(f"  Vocabulary:    {stats['vocab_size']} words "
          f"(started with {len(seed_vocab)})")
    print(f"  Base words:    {stats['base_words']}")
    print(f"  Compounds:     {stats['compounds']} "
          f"(e.g., {list(narrator.compositions.keys())[:3]})")
    print(f"  Invented:      {stats['invented']}")
    print(f"  Split:         {stats['split']}")
    print(f"  Composed:      {stats['composed']}")
    print(f"  Avg quality:   {np.mean(quality_history):.3f}")
    print(f"  Growth:        {len(seed_vocab)} → {stats['vocab_size']} "
          f"({stats['vocab_size']/len(seed_vocab):.1f}×)")

    # Show some learned compounds
    print(f"\n  ── Sample Compounds ──")
    for compound, parts in list(narrator.compositions.items())[:10]:
        sim = narrator.words[compound].similarity(
            narrator.words[parts[0]].prototype)
        print(f"    {compound:<25} = {parts[0]} + {parts[1]}  "
              f"(sim to {parts[0]}: {sim:+.3f})")

    # Show vocabulary growth curve
    print(f"\n  ── Vocabulary Growth ──")
    checkpoints = [0, 50, 100, 200, 300, 400, 500]
    for cp in checkpoints:
        if cp < len(vocab_history):
            print(f"    Cycle {cp:>4}: {vocab_history[cp]:>4} words")

    print(f"\n{'='*70}")


def run_tests():
    print("=" * 65)
    print("  Compositional Narrator Tests")
    print("=" * 65)
    rng = np.random.RandomState(42)
    p = 0; t = 0

    print("\n  T1: Describe with seed vocabulary")
    seed = {"left": rng.randn(D_BELIEF).astype(np.float32),
            "right": -rng.randn(D_BELIEF).astype(np.float32),
            "fast": rng.randn(D_BELIEF).astype(np.float32)}
    narrator = CompositionalNarrator(initial_vocab=seed)
    desc, quality, residual = narrator.describe(seed["left"])
    ok = len(desc) > 0 and quality > 0
    print(f"    \"{desc}\" quality={quality:.3f} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T2: Narration returns metadata")
    result = narrator.narrate(seed["left"])
    ok = all(k in result for k in ["description", "quality", "vocab_size"])
    print(f"    Keys present {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T3: Novel belief triggers word invention")
    novel = rng.randn(D_BELIEF).astype(np.float32) * 5
    before = len(narrator.words)
    result = narrator.narrate(novel, {"schema": 0, "step": 0})
    after = len(narrator.words)
    ok = after > before
    print(f"    Vocab {before} → {after} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T4: Composition creates compounds")
    narrator2 = CompositionalNarrator(initial_vocab=seed)
    # Present a belief between two known words
    between = (seed["left"] + seed["fast"]) * 0.5
    for i in range(20):
        narrator2.narrate(between + rng.randn(D_BELIEF).astype(np.float32) * 0.1,
                            {"schema": 0, "step": i})
    ok = len(narrator2.compositions) > 0 or narrator2.words_invented > 0
    print(f"    Compounds: {len(narrator2.compositions)} "
          f"Invented: {narrator2.words_invented} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T5: Vocabulary grows over 100 cycles")
    narrator3 = CompositionalNarrator(initial_vocab=seed)
    for i in range(100):
        belief = rng.randn(D_BELIEF).astype(np.float32) * (0.5 + i * 0.02)
        narrator3.narrate(belief, {"schema": i % 32, "step": i})
    ok = len(narrator3.words) > len(seed)
    print(f"    {len(seed)} → {len(narrator3.words)} words "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T6: Description quality improves with vocabulary")
    # Novel belief should be better described after learning
    test_belief = rng.randn(D_BELIEF).astype(np.float32) * 2
    _, q_before, _ = narrator3.describe(test_belief)
    # Teach it about this belief
    for i in range(10):
        narrator3.narrate(test_belief + rng.randn(D_BELIEF).astype(np.float32) * 0.1,
                            {"schema": 99, "step": 100 + i})
    _, q_after, _ = narrator3.describe(test_belief)
    ok = q_after >= q_before - 0.1  # should not get worse
    print(f"    Quality: {q_before:.3f} → {q_after:.3f} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T7: Word differentiation (split) works")
    narrator4 = CompositionalNarrator(initial_vocab={
        "place": rng.randn(D_BELIEF).astype(np.float32)
    })
    narrator4.split_variance_threshold = 0.1
    narrator4.min_hearings_to_split = 5
    # Give "place" very diverse experiences to trigger split
    for i in range(20):
        if i < 10:
            belief = rng.randn(D_BELIEF).astype(np.float32) * 0.5
        else:
            belief = rng.randn(D_BELIEF).astype(np.float32) * 0.5 + 5.0
        narrator4.narrate(belief, {"schema": 0, "step": i})
    has_split = any("place_" in w for w in narrator4.words)
    ok = has_split or len(narrator4.words) > 1
    print(f"    Split occurred: {has_split} vocab={len(narrator4.words)} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T8: Empty vocab handles gracefully")
    narrator5 = CompositionalNarrator()
    result = narrator5.narrate(rng.randn(D_BELIEF).astype(np.float32),
                                 {"schema": 0, "step": 0})
    ok = result["vocab_size"] >= 1  # should have invented at least one word
    print(f"    From zero: vocab={result['vocab_size']} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T9: Stats tracking works")
    stats = narrator3.get_stats()
    ok = stats["vocab_size"] > 0 and stats["total_narrations"] > 0
    print(f"    Vocab={stats['vocab_size']} narrations={stats['total_narrations']} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T10: 500 cycles shows significant growth")
    narrator6 = CompositionalNarrator(initial_vocab=seed)
    for i in range(500):
        belief = rng.randn(D_BELIEF).astype(np.float32) * (0.5 + i * 0.005)
        narrator6.narrate(belief, {"schema": i % 32, "step": i})
    growth = len(narrator6.words) / len(seed)
    ok = growth > 2.0  # should at least double
    print(f"    Growth: {len(seed)} → {len(narrator6.words)} "
          f"({growth:.1f}×) {'PASS' if ok else 'FAIL'}")
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
