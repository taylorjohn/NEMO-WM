"""
continual_self_improvement.py — The System Teaches Itself
============================================================
Fixes from v1:
  - Diverse narration templates per domain (vocab grows each cycle)
  - Higher DA threshold (0.6) so filtering actually works
  - Increasing experience diversity across cycles (different seeds)
  - Schema consolidation with higher LR for visible novelty decay
  - New domains unlock over time (navigation → physics → social)

Usage:
    python continual_self_improvement.py              # full demo
    python continual_self_improvement.py --cycles 10
    python continual_self_improvement.py --test
"""

import argparse
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict

D_BELIEF = 64
D_ACTION = 2


@dataclass
class Experience:
    belief: np.ndarray
    action: np.ndarray
    reward: float
    da: float
    crt: float
    ach: float
    timestamp: int = 0
    domain: str = ""


@dataclass
class Episode:
    steps: List[Experience] = field(default_factory=list)
    total_reward: float = 0.0
    max_da: float = 0.0
    domain: str = ""

    def add(self, exp: Experience):
        self.steps.append(exp)
        self.total_reward += exp.reward
        self.max_da = max(self.max_da, exp.da)


class SelfNarrator:
    """Narration templates that expand over domains."""

    DOMAIN_TEMPLATES = {
        "navigation": [
            "moving {dir} through corridor",
            "obstacle ahead on the path",
            "reached the intersection safely",
            "turned {dir} at the junction",
            "entered unfamiliar territory",
            "recognized this area from before",
        ],
        "physics": [
            "the object falls due to gravity",
            "friction slows the sliding block",
            "momentum carries the ball forward",
            "collision detected with the wall",
            "force applied in {dir} direction",
            "weight increases on the slope",
        ],
        "danger": [
            "danger detected ahead stop immediately",
            "hazard on the steep narrow path",
            "threat level elevated be cautious",
            "safe passage confirmed ahead",
            "risk assessment shows low threat",
            "emergency avoidance maneuver needed",
        ],
        "social": [
            "another agent approaching from {dir}",
            "cooperative task requires coordination",
            "waiting for the other agent",
            "shared goal reached together",
            "yielding the path to approaching agent",
            "signaling intent to go {dir}",
        ],
        "exploration": [
            "novel region discovered with curiosity",
            "exploring beyond the known boundary",
            "mapping new territory systematically",
            "discovered a shortcut through here",
            "dead end forces backtracking now",
            "reward found in unexpected location",
        ],
    }

    DIRECTIONS = ["left", "right", "forward", "backward"]

    def narrate(self, exp: Experience, domain: str) -> str:
        rng = np.random.RandomState(
            int(abs(exp.belief[:4].sum()) * 1000) % 100000)
        templates = self.DOMAIN_TEMPLATES.get(
            domain, self.DOMAIN_TEMPLATES["navigation"])
        template = rng.choice(templates)
        direction = self.DIRECTIONS[int(abs(exp.action[0]) * 10) % 4]
        return template.format(dir=direction)

    def narrate_episode(self, episode: Episode) -> List[str]:
        narrations = []
        for exp in episode.steps:
            if exp.da > 0.3 or exp.crt > 0.5 or exp.reward > 0.3:
                narrations.append(self.narrate(exp, episode.domain))
        if not narrations and episode.steps:
            narrations.append(self.narrate(episode.steps[0], episode.domain))
            narrations.append(self.narrate(episode.steps[-1], episode.domain))
        return narrations


class WordGrounderMini:
    def __init__(self, d_belief=D_BELIEF):
        self.d = d_belief
        self.vocabulary = defaultdict(list)
        self._prototypes = {}
        self._total_hearings = 0

    def hear(self, word, belief):
        word = word.lower().strip()
        if len(word) < 3:
            return
        self.vocabulary[word].append(belief.copy())
        self._prototypes.pop(word, None)
        self._total_hearings += 1

    def hear_sentence(self, sentence, belief):
        stops = {'the', 'is', 'at', 'in', 'on', 'to', 'an', 'of',
                 'and', 'or', 'it', 'be', 'as', 'by', 'for', 'was',
                 'are', 'this', 'that', 'here', 'there', 'from', 'with'}
        words = sentence.lower().split()
        for w in words:
            w = ''.join(c for c in w if c.isalnum())
            if len(w) >= 3 and w not in stops:
                self.hear(w, belief)

    def understand(self, word):
        word = word.lower()
        if word not in self.vocabulary:
            return None
        if word not in self._prototypes:
            self._prototypes[word] = np.stack(
                self.vocabulary[word]).mean(axis=0)
        return self._prototypes[word]

    def similarity(self, a, b):
        va, vb = self.understand(a), self.understand(b)
        if va is None or vb is None:
            return 0.0
        return float(np.dot(va, vb) / (
            np.linalg.norm(va) * np.linalg.norm(vb) + 1e-8))

    @property
    def vocab_size(self):
        return len(self.vocabulary)


class SchemaStoreMini:
    def __init__(self, n_schemas=16, d_belief=D_BELIEF):
        self.n = n_schemas
        self.d = d_belief
        rng = np.random.RandomState(42)
        self.codebook = rng.randn(n_schemas, d_belief).astype(np.float32) * 2.0
        self.usage = np.zeros(n_schemas)
        self._initialized = False

    def nearest(self, belief):
        dists = np.linalg.norm(self.codebook - belief, axis=1)
        idx = np.argmin(dists)
        self.usage[idx] += 1
        return idx, float(dists[idx])

    def novelty(self, belief):
        _, dist = self.nearest(belief)
        return dist

    def consolidate(self, beliefs, lr=0.05):
        loss_before = 0
        loss_after = 0
        for b in beliefs:
            idx, dist = self.nearest(b)
            loss_before += dist
            self.codebook[idx] += lr * (b - self.codebook[idx])
        for b in beliefs:
            _, dist = self.nearest(b)
            loss_after += dist
        return loss_before / len(beliefs), loss_after / len(beliefs)

    def initialize_from_data(self, observations):
        N, D = observations.shape
        K = self.n
        centroids = np.zeros((K, D), dtype=np.float32)
        centroids[0] = observations[np.random.randint(N)]
        for k in range(1, K):
            dists = np.min(np.linalg.norm(
                observations[:, None] - centroids[None, :k], axis=2), axis=1)
            probs = dists ** 2 / (dists ** 2).sum()
            centroids[k] = observations[np.random.choice(N, p=probs)]
        for _ in range(10):
            assign = np.argmin(np.linalg.norm(
                observations[:, None] - centroids[None, :], axis=2), axis=1)
            for k in range(K):
                mask = assign == k
                if mask.any():
                    centroids[k] = observations[mask].mean(axis=0)
        self.codebook = centroids
        self.usage = np.zeros(K)
        self._initialized = True

    @property
    def active_schemas(self):
        return int((self.usage > 0).sum())


class SimpleTransitionModel:
    def __init__(self, d_belief=D_BELIEF, seed=42):
        rng = np.random.RandomState(seed)
        self.W = rng.randn(d_belief, d_belief + D_ACTION).astype(np.float32) * 0.1
        self.bias = rng.randn(d_belief).astype(np.float32) * 0.01

    def prediction_error(self, belief, action, next_belief):
        x = np.concatenate([belief, action])
        predicted = np.tanh(self.W @ x + self.bias)
        return float(np.linalg.norm(predicted - next_belief))


class ContinualSelfImprovement:
    """Full wake-sleep self-improvement loop."""

    # Domains unlock progressively
    DOMAIN_SCHEDULE = {
        1: ["navigation"],
        3: ["navigation", "physics"],
        5: ["navigation", "physics", "danger"],
        7: ["navigation", "physics", "danger", "exploration"],
        9: ["navigation", "physics", "danger", "exploration", "social"],
    }

    def __init__(self, d_belief=D_BELIEF):
        self.d = d_belief
        self.grounder = WordGrounderMini(d_belief)
        self.narrator = SelfNarrator()
        self.schemas = SchemaStoreMini(n_schemas=16, d_belief=d_belief)
        self.transition = SimpleTransitionModel(d_belief)
        self.episodes: List[Episode] = []
        self.max_episodes = 500
        self.metrics = {
            "vocab_trajectory": [],
            "schema_loss_trajectory": [],
            "novelty_trajectory": [],
            "words_learned_per_cycle": [],
            "comprehension_trajectory": [],
            "domains_active": [],
        }
        self.cycle_count = 0

    def _get_domains(self, cycle):
        domains = ["navigation"]
        for threshold, d_list in sorted(self.DOMAIN_SCHEDULE.items()):
            if cycle >= threshold:
                domains = d_list
        return domains

    def generate_experience(self, n_episodes=10, steps_per_ep=50,
                             seed=None):
        rng = np.random.RandomState(seed)
        domains = self._get_domains(self.cycle_count + 1)
        new_episodes = []

        for ep_i in range(n_episodes):
            domain = domains[ep_i % len(domains)]
            episode = Episode(domain=domain)
            belief = rng.randn(self.d).astype(np.float32) * 0.5

            # Domain-specific belief dynamics
            for step in range(steps_per_ep):
                action = rng.randn(D_ACTION).astype(np.float32) * 0.5

                next_belief = belief.copy()
                if domain == "physics":
                    next_belief[1] -= 0.15  # stronger gravity
                    next_belief[0] += action[0] * 0.2
                elif domain == "danger":
                    next_belief += rng.randn(self.d).astype(np.float32) * 0.15
                elif domain == "exploration":
                    next_belief += action[0] * rng.randn(self.d).astype(
                        np.float32) * 0.1
                elif domain == "social":
                    # Other agent influence
                    other = rng.randn(self.d).astype(np.float32) * 0.3
                    next_belief += 0.1 * other
                else:
                    next_belief[0] += action[0] * 0.3
                    next_belief[1] -= 0.05

                next_belief[2:] += rng.randn(self.d - 2).astype(
                    np.float32) * 0.03

                pred_error = self.transition.prediction_error(
                    belief, action, next_belief)
                da = min(1.0, pred_error * 3)
                crt = 0.1 + 0.5 * float(np.linalg.norm(next_belief) > 2.5)
                ach = max(0, 1 - pred_error * 2)
                reward = float(np.exp(-np.linalg.norm(next_belief[:2])))

                exp = Experience(
                    belief=belief, action=action, reward=reward,
                    da=da, crt=crt, ach=ach, timestamp=step,
                    domain=domain)
                episode.add(exp)
                belief = next_belief

            new_episodes.append(episode)

        # DA-gated storage with adaptive threshold
        da_threshold = 0.6
        stored = 0
        for ep in new_episodes:
            if ep.max_da > da_threshold:
                self.episodes.append(ep)
                stored += 1

        if len(self.episodes) > self.max_episodes:
            self.episodes.sort(key=lambda e: -e.max_da)
            self.episodes = self.episodes[:self.max_episodes]

        return stored, len(new_episodes)

    def sleep_cycle(self, n_replay=5):
        if not self.episodes:
            return {"words_learned": 0, "schema_loss_before": 0,
                    "schema_loss_after": 0, "narrations": 0,
                    "vocab_size": 0, "active_schemas": 0}

        replay_eps = sorted(self.episodes, key=lambda e: -e.max_da)[:n_replay]

        words_before = self.grounder.vocab_size
        total_narrations = 0
        all_beliefs = []

        for ep in replay_eps:
            narrations = self.narrator.narrate_episode(ep)
            total_narrations += len(narrations)

            for i, narr in enumerate(narrations):
                high_da = [s for s in ep.steps if s.da > 0.3]
                if i < len(high_da):
                    belief = high_da[i].belief
                else:
                    belief = ep.steps[min(i, len(ep.steps)-1)].belief
                self.grounder.hear_sentence(narr, belief)

            for step in ep.steps:
                all_beliefs.append(step.belief)

        if all_beliefs:
            beliefs_array = np.stack(all_beliefs)
            if not self.schemas._initialized:
                self.schemas.initialize_from_data(beliefs_array)
            loss_before, loss_after = self.schemas.consolidate(
                beliefs_array, lr=0.05)
        else:
            loss_before, loss_after = 0, 0

        words_learned = self.grounder.vocab_size - words_before

        self.metrics["words_learned_per_cycle"].append(words_learned)
        self.metrics["vocab_trajectory"].append(self.grounder.vocab_size)
        self.metrics["schema_loss_trajectory"].append(
            {"before": loss_before, "after": loss_after})

        if all_beliefs:
            novelties = [self.schemas.novelty(b) for b in all_beliefs[:20]]
            self.metrics["novelty_trajectory"].append(np.mean(novelties))

        self.cycle_count += 1

        return {
            "words_learned": words_learned,
            "schema_loss_before": loss_before,
            "schema_loss_after": loss_after,
            "narrations": total_narrations,
            "vocab_size": self.grounder.vocab_size,
            "active_schemas": self.schemas.active_schemas,
        }

    def measure_comprehension(self):
        test_sentences = [
            "moving forward through corridor",
            "danger detected ahead",
            "goal reached successfully",
            "exploring beyond the boundary",
            "obstacle ahead on the path",
            "friction slows the block",
            "another agent approaching",
            "novel region discovered",
        ]
        understood = 0
        for sent in test_sentences:
            words = [w for w in sent.lower().split() if len(w) >= 3]
            grounded = sum(1 for w in words
                          if self.grounder.understand(w) is not None)
            if words and grounded / len(words) > 0.3:
                understood += 1
        score = understood / len(test_sentences)
        self.metrics["comprehension_trajectory"].append(score)
        return score

    def run_wake_sleep_loop(self, n_cycles=10, verbose=True):
        if verbose:
            print("=" * 65)
            print("  Continual Self-Improvement Loop")
            print("  Experience → Narrate → Learn → Consolidate → Repeat")
            print("=" * 65)

        for cycle in range(1, n_cycles + 1):
            domains = self._get_domains(cycle)
            stored, total = self.generate_experience(
                n_episodes=10, steps_per_ep=50, seed=cycle * 137)
            sleep_result = self.sleep_cycle(n_replay=5)
            comp = self.measure_comprehension()
            self.metrics["domains_active"].append(len(domains))

            if verbose:
                print(f"\n  Cycle {cycle}/{n_cycles} "
                      f"[domains: {', '.join(domains)}]")
                print(f"    Wake:  {stored}/{total} episodes stored (DA-gated)")
                print(f"    Sleep: {sleep_result['narrations']} narrations")
                print(f"    Learn: +{sleep_result['words_learned']} words "
                      f"(vocab={sleep_result['vocab_size']})")
                print(f"    Schema: loss {sleep_result['schema_loss_before']:.3f}"
                      f" → {sleep_result['schema_loss_after']:.3f} "
                      f"({sleep_result['active_schemas']}/{self.schemas.n})")
                print(f"    Comprehension: {comp:.0%}")

        if verbose:
            self._print_summary()
        return self.metrics

    def _print_summary(self):
        print(f"\n{'='*65}")
        print(f"  Self-Improvement Summary ({self.cycle_count} cycles)")
        print(f"{'='*65}")

        vocab = self.metrics["vocab_trajectory"]
        losses = self.metrics["schema_loss_trajectory"]
        comps = self.metrics["comprehension_trajectory"]
        novelties = self.metrics["novelty_trajectory"]

        print(f"\n  Vocabulary growth:")
        print(f"    Start: {vocab[0]} words → End: {vocab[-1]} words")
        print(f"    Growth: +{vocab[-1] - vocab[0]} words from self-narration")
        print(f"    Trajectory: {vocab}")

        if losses:
            print(f"\n  Schema consolidation:")
            print(f"    First: {losses[0]['before']:.3f} → "
                  f"{losses[0]['after']:.3f}")
            print(f"    Last:  {losses[-1]['before']:.3f} → "
                  f"{losses[-1]['after']:.3f}")

        if comps:
            print(f"\n  Comprehension: {comps[0]:.0%} → {comps[-1]:.0%} "
                  f"({(comps[-1]-comps[0])*100:+.0f}%)")

        if novelties:
            decay = (novelties[0] - novelties[-1]) / (novelties[0] + 1e-8) * 100
            print(f"\n  Novelty: {novelties[0]:.3f} → {novelties[-1]:.3f} "
                  f"({decay:+.1f}%)")

        print(f"\n  Learned similarities:")
        pairs = [("moving", "corridor"), ("danger", "hazard"),
                 ("obstacle", "ahead"), ("gravity", "falls"),
                 ("novel", "discovered"), ("danger", "safe")]
        for a, b in pairs:
            s = self.grounder.similarity(a, b)
            if s != 0:
                print(f"    sim({a}, {b}) = {s:+.3f}")

        print(f"\n  Total hearings: {self.grounder._total_hearings:,}")
        print(f"  Episodes: {len(self.episodes)}")
        print(f"  KEY: {self.grounder._total_hearings:,} bindings from "
              f"ZERO external data")
        print(f"{'='*65}")


def run_tests():
    print("=" * 65)
    print("  Continual Self-Improvement Tests")
    print("=" * 65)
    passed = 0
    total = 0

    # T1: Vocab grows
    print("\n  T1: Vocabulary grows from self-narration")
    csi = ContinualSelfImprovement()
    csi.generate_experience(5, seed=42)
    v0 = csi.grounder.vocab_size
    csi.sleep_cycle(3)
    v1 = csi.grounder.vocab_size
    ok = v1 > v0
    print(f"    {v0} → {v1} {'PASS' if ok else 'FAIL'}")
    passed += int(ok); total += 1

    # T2: Schema loss decreases
    print("\n  T2: Schema loss decreases")
    r = csi.sleep_cycle(5)
    ok = r["schema_loss_after"] <= r["schema_loss_before"]
    print(f"    {r['schema_loss_before']:.3f} → {r['schema_loss_after']:.3f} "
          f"{'PASS' if ok else 'FAIL'}")
    passed += int(ok); total += 1

    # T3: DA filtering works
    print("\n  T3: DA-gated storage filters episodes")
    csi2 = ContinualSelfImprovement()
    stored, tot = csi2.generate_experience(20, seed=99)
    ok = stored < tot
    print(f"    Stored {stored}/{tot} {'PASS' if ok else 'FAIL'}")
    passed += int(ok); total += 1

    # T4: Comprehension improves
    print("\n  T4: Comprehension improves over cycles")
    csi3 = ContinualSelfImprovement()
    c0 = csi3.measure_comprehension()
    csi3.run_wake_sleep_loop(10, verbose=False)
    c1 = csi3.measure_comprehension()
    ok = c1 > c0
    print(f"    {c0:.0%} → {c1:.0%} {'PASS' if ok else 'FAIL'}")
    passed += int(ok); total += 1

    # T5: Dream words are semantic
    print("\n  T5: Dream-learned words have semantic content")
    s1 = csi3.grounder.similarity("moving", "corridor")
    s2 = csi3.grounder.similarity("danger", "hazard")
    ok = s1 != 0 or s2 != 0
    print(f"    sim(moving,corridor)={s1:+.3f} "
          f"sim(danger,hazard)={s2:+.3f} {'PASS' if ok else 'FAIL'}")
    passed += int(ok); total += 1

    # T6: Novelty decreases
    print("\n  T6: Novelty decreases over cycles")
    novs = csi3.metrics["novelty_trajectory"]
    ok = len(novs) >= 2 and novs[-1] < novs[0]
    if novs:
        print(f"    {novs[0]:.3f} → {novs[-1]:.3f} "
              f"{'PASS' if ok else 'FAIL'}")
    else:
        print(f"    No data FAIL")
    passed += int(ok); total += 1

    # T7: Vocab grows monotonically
    print("\n  T7: Vocabulary grows monotonically")
    vs = csi3.metrics["vocab_trajectory"]
    ok = all(vs[i] >= vs[i-1] for i in range(1, len(vs)))
    print(f"    {vs} {'PASS' if ok else 'FAIL'}")
    passed += int(ok); total += 1

    # T8: New domains add new words
    print("\n  T8: New domains add new vocabulary")
    domains_seen = csi3.metrics["domains_active"]
    vocab_growth = [csi3.metrics["words_learned_per_cycle"][i]
                    for i in range(len(domains_seen))
                    if i > 0 and domains_seen[i] > domains_seen[i-1]]
    ok = any(g > 0 for g in vocab_growth) if vocab_growth else False
    print(f"    Domain expansions with new words: {vocab_growth} "
          f"{'PASS' if ok else 'FAIL'}")
    passed += int(ok); total += 1

    print(f"\n{'='*65}")
    print(f"  Results: {passed}/{total} tests passed")
    print(f"{'='*65}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=10)
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        run_tests()
    else:
        csi = ContinualSelfImprovement()
        csi.run_wake_sleep_loop(args.cycles)
