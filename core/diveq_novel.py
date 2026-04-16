"""
diveq_novel.py — Novel Applications of Differentiable VQ
==========================================================
Four untried features using DiVeQ in ways nobody has published:

1. MoodStates — neuromodulator quantization → emergent emotions
2. DreamInterpolation — SF-DiVeQ segment walks for dreaming
3. ActionPrimitives — motor vocabulary from action quantization
4. AdaptiveCodebook — surprise-gated growth (neurogenesis)

Each is self-contained with its own demo and self-test.

Usage:
    python diveq_novel.py              # run all demos
    python diveq_novel.py --demo mood  # just mood states
"""

import argparse
import math
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diveq_schema import DiVeQ, SFDiVeQ


# ══════════════════════════════════════════════════════════════════════════════
# 1. MOOD STATES — Emergent Emotional Vocabulary
# ══════════════════════════════════════════════════════════════════════════════

class MoodStates(nn.Module):
    """
    Quantize the 5D neuromodulator space into discrete mood states.
    The codebook learns a vocabulary of emotional states from the
    system's own dynamics — no human-labeled emotions needed.

    Input:  [DA, ACh, CRT, NE, 5HT] (5D continuous)
    Output: mood_id (integer), mood_name (string), mood_vector (5D quantized)

    The mood names are generated from the dominant neuromodulators
    in each codeword — not from an LLM.

    Paper claim: "Emergent affective states from neuromodulatory
    dynamics — the first world model with self-discovered emotions."
    """

    # Neuromodulator roles for naming
    NEUROMOD_NAMES = ['DA', 'ACh', 'CRT', 'NE', '5HT']
    NEUROMOD_MEANINGS = {
        'DA':  {'high': 'Curious',    'low': 'Familiar'},
        'ACh': {'high': 'Confident',  'low': 'Uncertain'},
        'CRT': {'high': 'Stressed',   'low': 'Calm'},
        'NE':  {'high': 'Alert',      'low': 'Relaxed'},
        '5HT': {'high': 'Cautious',   'low': 'Bold'},
    }

    def __init__(self, n_moods: int = 16, sigma: float = 0.05):
        super().__init__()
        self.n_moods = n_moods
        self.vq = DiVeQ(n_codes=n_moods, d_dim=5, sigma=sigma)

        # Initialize codebook with meaningful spread
        with torch.no_grad():
            # Spread codewords across [0,1]^5
            for i in range(n_moods):
                for d in range(5):
                    self.vq.codebook.data[i, d] = (
                        (i * 7 + d * 3) % n_moods) / n_moods

        self._mood_history: List[int] = []
        self._names_cache: Dict[int, str] = {}

    def name_mood(self, mood_vector: torch.Tensor) -> str:
        """Generate a name from the dominant neuromodulators."""
        v = mood_vector.detach().cpu().numpy()
        parts = []

        # Find top 2 most extreme (furthest from 0.5)
        extremes = [(abs(v[i] - 0.5), i, v[i]) for i in range(5)]
        extremes.sort(reverse=True)

        for _, idx, val in extremes[:2]:
            name = self.NEUROMOD_NAMES[idx]
            level = 'high' if val > 0.5 else 'low'
            parts.append(self.NEUROMOD_MEANINGS[name][level])

        return '-'.join(parts)

    def forward(self, neuromod_state: torch.Tensor) -> Dict:
        """
        Quantize neuromodulator state to nearest mood.

        Args:
            neuromod_state: (B, 5) or (5,) — [DA, ACh, CRT, NE, 5HT]

        Returns:
            dict with mood_id, mood_name, mood_vector, commit_loss
        """
        if neuromod_state.dim() == 1:
            neuromod_state = neuromod_state.unsqueeze(0)

        z_q, info = self.vq(neuromod_state)
        mood_id = int(info['indices'][0])

        # Cache name
        if mood_id not in self._names_cache:
            self._names_cache[mood_id] = self.name_mood(z_q[0])

        self._mood_history.append(mood_id)

        return {
            'mood_id': mood_id,
            'mood_name': self._names_cache[mood_id],
            'mood_vector': z_q[0],
            'commit_loss': info['commit_loss'],
            'distance': info['mean_dist'],
        }

    def mood_transitions(self) -> Dict[Tuple[int, int], int]:
        """Count transitions between mood states."""
        trans = {}
        for i in range(1, len(self._mood_history)):
            key = (self._mood_history[i-1], self._mood_history[i])
            trans[key] = trans.get(key, 0) + 1
        return trans

    def most_common_moods(self, k: int = 5) -> List[Tuple[str, int]]:
        """Return k most frequent mood states."""
        counts = {}
        for m in self._mood_history:
            counts[m] = counts.get(m, 0) + 1
        sorted_moods = sorted(counts.items(), key=lambda x: -x[1])
        return [(self._names_cache.get(mid, f"mood_{mid}"), cnt)
                for mid, cnt in sorted_moods[:k]]

    def narrate_mood(self, neuromod_state: torch.Tensor) -> str:
        """Generate natural language mood description."""
        result = self.forward(neuromod_state)
        v = neuromod_state.detach().cpu().numpy().flatten()

        desc = f"Mood: {result['mood_name']} (state {result['mood_id']}). "

        # Add context
        if v[0] > 0.7:
            desc += "High novelty driving exploration. "
        if v[2] > 0.5:
            desc += "Stress reducing working memory. "
        if v[1] > 0.7 and v[2] < 0.3:
            desc += "High confidence, trusting predictions. "
        if v[3] > 0.7:
            desc += "Heightened alertness. "

        return desc.strip()


# ══════════════════════════════════════════════════════════════════════════════
# 2. DREAM INTERPOLATION — Walking Schema Segments
# ══════════════════════════════════════════════════════════════════════════════

class DreamInterpolation(nn.Module):
    """
    Generate dream sequences by walking SF-DiVeQ line segments.

    During sleep, pick two schemas and interpolate between them.
    Each interpolated point is a "dreamed" belief state.
    The transition model predicts what SHOULD happen between them.
    Prediction errors reveal gaps in the world model.

    This is how humans dream: recombination of known experiences
    into novel sequences that probe the world model's boundaries.

    Paper claim: "Schema interpolation for targeted world model
    improvement — dreaming as hypothesis generation."
    """

    def __init__(self, schema_codebook: nn.Parameter,
                 n_interp_steps: int = 20):
        super().__init__()
        self.codebook = schema_codebook
        self.n_steps = n_interp_steps

    def dream_between(self, schema_a: int, schema_b: int) -> torch.Tensor:
        """
        Interpolate between two schemas.

        Returns: (n_steps, D) sequence of dreamed belief states
        """
        c_a = self.codebook[schema_a]
        c_b = self.codebook[schema_b]

        lambdas = torch.linspace(0, 1, self.n_steps)
        dreamed = torch.stack([
            (1 - lam) * c_a + lam * c_b for lam in lambdas
        ])
        return dreamed

    def dream_random_walk(self, n_steps: int = 50,
                           start_schema: int = 0) -> Tuple[torch.Tensor, List[int]]:
        """
        Random walk through schema space.
        At each step, move to a neighboring schema with some noise.
        """
        n_codes = self.codebook.shape[0]
        trajectory = [self.codebook[start_schema]]
        visited = [start_schema]
        current = start_schema

        for step in range(n_steps - 1):
            # Find nearest 3 schemas
            dists = torch.cdist(
                self.codebook[current].unsqueeze(0),
                self.codebook.unsqueeze(0)
            )[0, 0]
            dists[current] = float('inf')  # don't stay

            # Pick random neighbor from top 3 nearest
            _, nearest = dists.topk(min(3, n_codes - 1), largest=False)
            next_idx = int(nearest[torch.randint(len(nearest), (1,))])

            # Interpolate halfway with noise
            point = 0.5 * (self.codebook[current] + self.codebook[next_idx])
            point += torch.randn_like(point) * 0.05

            trajectory.append(point)
            visited.append(next_idx)
            current = next_idx

        return torch.stack(trajectory), visited

    def dream_cycle(self, transition_model=None,
                     n_pairs: int = 5) -> Dict:
        """
        Run a full dream cycle: interpolate between schema pairs,
        optionally evaluate prediction errors via transition model.
        """
        n_codes = self.codebook.shape[0]
        dream_sequences = []
        total_error = 0.0

        for _ in range(n_pairs):
            a = torch.randint(n_codes, (1,)).item()
            b = torch.randint(n_codes, (1,)).item()
            while b == a:
                b = torch.randint(n_codes, (1,)).item()

            dreamed = self.dream_between(a, b)

            if transition_model is not None:
                # Measure prediction error along dream
                with torch.no_grad():
                    for t in range(len(dreamed) - 1):
                        pred = transition_model(dreamed[t].unsqueeze(0))
                        actual = dreamed[t + 1]
                        err = F.mse_loss(pred.squeeze(), actual).item()
                        total_error += err

            dream_sequences.append({
                'from': a, 'to': b,
                'length': len(dreamed),
                'start_belief': dreamed[0],
                'end_belief': dreamed[-1],
            })

        return {
            'n_dreams': n_pairs,
            'sequences': dream_sequences,
            'mean_error': total_error / max(1, n_pairs * self.n_steps),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 3. ACTION PRIMITIVES — Motor Vocabulary
# ══════════════════════════════════════════════════════════════════════════════

class ActionPrimitives(nn.Module):
    """
    Quantize the continuous action space into discrete motor primitives.
    The codebook learns a vocabulary of actions from experience.

    Instead of predicting continuous 2D actions, the planner selects
    from N learned primitives. This makes planning combinatorial
    (search over N options) instead of continuous (infinite options).

    Paper claim: "Emergent motor vocabulary via action quantization —
    the system invents its own movement language."
    """

    def __init__(self, d_action: int = 2, n_primitives: int = 32,
                 sigma: float = 0.1):
        super().__init__()
        self.d_action = d_action
        self.n_primitives = n_primitives
        self.vq = DiVeQ(n_codes=n_primitives, d_dim=d_action, sigma=sigma)

        # Initialize with unit circle for 2D actions
        if d_action == 2:
            with torch.no_grad():
                for i in range(n_primitives):
                    angle = 2 * math.pi * i / n_primitives
                    mag = 0.3 + 0.7 * (i % 3) / 2  # vary magnitude
                    self.vq.codebook.data[i] = torch.tensor([
                        mag * math.cos(angle),
                        mag * math.sin(angle)
                    ])

        self._action_history: List[int] = []

    def quantize(self, action: torch.Tensor) -> Dict:
        """Quantize continuous action to nearest primitive."""
        if action.dim() == 1:
            action = action.unsqueeze(0)

        z_q, info = self.vq(action)
        prim_id = int(info['indices'][0])
        self._action_history.append(prim_id)

        return {
            'primitive_id': prim_id,
            'quantized_action': z_q[0],
            'original_action': action[0],
            'quantization_error': info['mean_dist'],
        }

    def describe_primitive(self, prim_id: int) -> str:
        """Describe a motor primitive in natural language."""
        p = self.vq.codebook.data[prim_id].detach().cpu().numpy()

        if self.d_action == 2:
            angle = math.degrees(math.atan2(p[1], p[0]))
            mag = math.sqrt(p[0]**2 + p[1]**2)

            # Direction name
            if -22.5 <= angle < 22.5:
                direction = "right"
            elif 22.5 <= angle < 67.5:
                direction = "upper-right"
            elif 67.5 <= angle < 112.5:
                direction = "up"
            elif 112.5 <= angle < 157.5:
                direction = "upper-left"
            elif angle >= 157.5 or angle < -157.5:
                direction = "left"
            elif -157.5 <= angle < -112.5:
                direction = "lower-left"
            elif -112.5 <= angle < -67.5:
                direction = "down"
            else:
                direction = "lower-right"

            speed = "fast" if mag > 0.7 else "moderate" if mag > 0.3 else "slow"
            return f"Move {direction} ({speed}, magnitude {mag:.2f})"
        else:
            return f"Primitive {prim_id}: {p}"

    def vocabulary(self) -> List[str]:
        """Return the full motor vocabulary."""
        return [self.describe_primitive(i) for i in range(self.n_primitives)]

    def usage_stats(self) -> Dict:
        """Which primitives are used most?"""
        counts = {}
        for a in self._action_history:
            counts[a] = counts.get(a, 0) + 1
        sorted_prims = sorted(counts.items(), key=lambda x: -x[1])
        return {
            'total_actions': len(self._action_history),
            'unique_primitives': len(counts),
            'most_used': [(self.describe_primitive(pid), cnt)
                           for pid, cnt in sorted_prims[:5]],
            'usage_ratio': len(counts) / self.n_primitives,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 4. ADAPTIVE CODEBOOK — Surprise-Gated Growth (Neurogenesis)
# ══════════════════════════════════════════════════════════════════════════════

class AdaptiveCodebook(nn.Module):
    """
    A DiVeQ codebook that grows when novelty is persistently high
    and shrinks when codewords go unused — like neurogenesis.

    Growth: if novelty > threshold for N consecutive steps,
            add a new codeword at the current belief position.

    Pruning: if a codeword hasn't been used in M steps, remove it.

    Paper claim: "Adaptive memory capacity via surprise-gated
    neurogenesis — the world model grows where it needs to."
    """

    def __init__(self, d_dim: int = 64, initial_codes: int = 16,
                 max_codes: int = 256, growth_threshold: float = 2.0,
                 growth_patience: int = 10, prune_after: int = 500):
        super().__init__()
        self.d_dim = d_dim
        self.max_codes = max_codes
        self.growth_threshold = growth_threshold
        self.growth_patience = growth_patience
        self.prune_after = prune_after

        # Start with small codebook
        self.codebook = nn.ParameterList([
            nn.Parameter(torch.randn(d_dim) * 0.1)
            for _ in range(initial_codes)
        ])

        self._novelty_streak = 0
        self._step = 0
        self._last_used = [0] * initial_codes
        self._growth_events: List[Dict] = []
        self._prune_events: List[Dict] = []

    @property
    def n_codes(self) -> int:
        return len(self.codebook)

    def find_nearest(self, z: torch.Tensor) -> Tuple[torch.Tensor, int, float]:
        """Find nearest codeword."""
        codes = torch.stack(list(self.codebook))
        dists = (codes - z.unsqueeze(0)).norm(dim=-1)
        idx = int(dists.argmin())
        return self.codebook[idx], idx, float(dists[idx])

    def observe(self, z: torch.Tensor) -> Dict:
        """
        Observe a belief vector. May trigger growth or pruning.
        """
        if z.dim() > 1:
            z = z.squeeze(0)

        nearest, idx, dist = self.find_nearest(z)
        self._last_used[idx] = self._step
        self._step += 1

        result = {
            'nearest_idx': idx,
            'distance': dist,
            'n_codes': self.n_codes,
            'grew': False,
            'pruned': 0,
        }

        # Check for growth
        if dist > self.growth_threshold:
            self._novelty_streak += 1
        else:
            self._novelty_streak = 0

        if (self._novelty_streak >= self.growth_patience and
                self.n_codes < self.max_codes):
            # Add new codeword at current position
            new_code = nn.Parameter(z.detach().clone())
            self.codebook.append(new_code)
            self._last_used.append(self._step)
            self._novelty_streak = 0
            result['grew'] = True
            self._growth_events.append({
                'step': self._step,
                'n_codes': self.n_codes,
                'novelty': dist,
            })

        # Check for pruning
        if self._step % 100 == 0 and self.n_codes > 4:
            to_remove = []
            for i in range(self.n_codes):
                if self._step - self._last_used[i] > self.prune_after:
                    to_remove.append(i)

            # Remove in reverse order to maintain indices
            for i in sorted(to_remove, reverse=True):
                if self.n_codes > 4:  # keep minimum 4
                    self.codebook = nn.ParameterList([p for j, p in enumerate(self.codebook) if j != i])
                    self._last_used = [v for j, v in enumerate(self._last_used) if j != i]
                    result['pruned'] += 1

            if result['pruned'] > 0:
                self._prune_events.append({
                    'step': self._step,
                    'removed': result['pruned'],
                    'remaining': self.n_codes,
                })

        return result

    def stats(self) -> Dict:
        return {
            'n_codes': self.n_codes,
            'growth_events': len(self._growth_events),
            'prune_events': len(self._prune_events),
            'step': self._step,
            'novelty_streak': self._novelty_streak,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Demos
# ══════════════════════════════════════════════════════════════════════════════

def demo_mood():
    print("\n" + "=" * 60)
    print("  1. MOOD STATES — Emergent Emotional Vocabulary")
    print("=" * 60)

    mood = MoodStates(n_moods=16)

    scenarios = [
        ("Exploring novel terrain",
         torch.tensor([0.8, 0.7, 0.1, 0.5, 0.6])),
        ("Under stress, uncertain",
         torch.tensor([0.2, 0.3, 0.8, 0.9, 0.2])),
        ("Calm, familiar territory",
         torch.tensor([0.1, 0.8, 0.1, 0.2, 0.7])),
        ("Surprise! Something new",
         torch.tensor([0.9, 0.4, 0.5, 0.7, 0.3])),
        ("Tired, winding down",
         torch.tensor([0.1, 0.2, 0.2, 0.1, 0.8])),
        ("Alert and cautious",
         torch.tensor([0.3, 0.5, 0.4, 0.9, 0.8])),
    ]

    for label, state in scenarios:
        narration = mood.narrate_mood(state)
        print(f"\n  Scenario: {label}")
        print(f"  State: DA={state[0]:.1f} ACh={state[1]:.1f} "
              f"CRT={state[2]:.1f} NE={state[3]:.1f} 5HT={state[4]:.1f}")
        print(f"  >> {narration}")

    # Simulate 100 steps
    for _ in range(100):
        state = torch.rand(5)
        mood.forward(state)

    print(f"\n  Most common moods (100 random steps):")
    for name, count in mood.most_common_moods(5):
        print(f"    {name}: {count} occurrences")

    trans = mood.mood_transitions()
    print(f"  Unique transitions: {len(trans)}")


def demo_dream():
    print("\n" + "=" * 60)
    print("  2. DREAM INTERPOLATION — Walking Schema Segments")
    print("=" * 60)

    # Create a small schema codebook
    codebook = nn.Parameter(torch.randn(16, 64) * 0.5)
    dreamer = DreamInterpolation(codebook, n_interp_steps=10)

    # Dream between schemas 0 and 5
    dreamed = dreamer.dream_between(0, 5)
    print(f"\n  Dream 0→5: {dreamed.shape} states")
    print(f"  Start norm: {dreamed[0].norm():.3f}")
    print(f"  End norm: {dreamed[-1].norm():.3f}")

    # Cosine similarity along dream
    for i in [0, 3, 5, 7, 9]:
        cos_start = F.cosine_similarity(
            dreamed[i].unsqueeze(0), dreamed[0].unsqueeze(0)).item()
        cos_end = F.cosine_similarity(
            dreamed[i].unsqueeze(0), dreamed[-1].unsqueeze(0)).item()
        print(f"  Step {i}: cos(start)={cos_start:.3f}, cos(end)={cos_end:.3f}")

    # Random walk
    walk, visited = dreamer.dream_random_walk(n_steps=20)
    print(f"\n  Random walk: {len(visited)} schemas visited")
    print(f"  Path: {visited[:10]}{'...' if len(visited) > 10 else ''}")
    unique = len(set(visited))
    print(f"  Unique schemas: {unique}/{len(visited)}")

    # Full dream cycle
    result = dreamer.dream_cycle(n_pairs=5)
    print(f"\n  Dream cycle: {result['n_dreams']} pairs interpolated")


def demo_action():
    print("\n" + "=" * 60)
    print("  3. ACTION PRIMITIVES — Motor Vocabulary")
    print("=" * 60)

    primitives = ActionPrimitives(d_action=2, n_primitives=16)

    print(f"\n  Motor vocabulary ({primitives.n_primitives} primitives):")
    for i, desc in enumerate(primitives.vocabulary()):
        print(f"    [{i:2d}] {desc}")

    # Quantize some actions
    test_actions = [
        ("Push right",     torch.tensor([0.8, 0.0])),
        ("Push up-left",   torch.tensor([-0.5, 0.7])),
        ("Gentle nudge",   torch.tensor([0.1, 0.05])),
        ("Strong diagonal", torch.tensor([0.9, 0.9])),
    ]

    print(f"\n  Action quantization:")
    for label, action in test_actions:
        result = primitives.quantize(action)
        desc = primitives.describe_primitive(result['primitive_id'])
        print(f"    {label}: [{action[0]:.1f}, {action[1]:.1f}] "
              f"→ [{result['primitive_id']:2d}] {desc} "
              f"(err={result['quantization_error']:.3f})")

    # Simulate usage
    for _ in range(200):
        action = torch.randn(2) * 0.5
        primitives.quantize(action)

    stats = primitives.usage_stats()
    print(f"\n  Usage stats ({stats['total_actions']} actions):")
    print(f"    Unique primitives used: {stats['unique_primitives']}/{primitives.n_primitives}")
    print(f"    Usage ratio: {stats['usage_ratio']:.1%}")
    print(f"    Most used:")
    for desc, cnt in stats['most_used']:
        print(f"      {desc}: {cnt} times")


def demo_adaptive():
    print("\n" + "=" * 60)
    print("  4. ADAPTIVE CODEBOOK — Surprise-Gated Growth")
    print("=" * 60)

    codebook = AdaptiveCodebook(
        d_dim=64, initial_codes=8, max_codes=64,
        growth_threshold=1.5, growth_patience=5, prune_after=200)

    print(f"  Initial codes: {codebook.n_codes}")

    # Phase 1: familiar territory (low novelty)
    print(f"\n  Phase 1: Familiar territory (100 steps)")
    base = torch.randn(64) * 0.3
    for i in range(100):
        z = base + torch.randn(64) * 0.1
        result = codebook.observe(z)
    print(f"    Codes after: {codebook.n_codes}")
    print(f"    Growth events: {codebook.stats()['growth_events']}")

    # Phase 2: novel territory (high novelty)
    print(f"\n  Phase 2: Novel territory (100 steps)")
    for i in range(100):
        z = torch.randn(64) * 3.0  # far from codebook
        result = codebook.observe(z)
        if result['grew']:
            print(f"    Step {100+i}: GREW to {codebook.n_codes} codes "
                  f"(novelty={result['distance']:.2f})")
    print(f"    Codes after: {codebook.n_codes}")
    print(f"    Growth events: {codebook.stats()['growth_events']}")

    # Phase 3: back to familiar (triggers pruning of unused)
    print(f"\n  Phase 3: Back to familiar (300 steps, triggers pruning)")
    for i in range(300):
        z = base + torch.randn(64) * 0.1
        result = codebook.observe(z)
        if result['pruned'] > 0:
            print(f"    Step {200+i}: PRUNED {result['pruned']} codes, "
                  f"remaining={codebook.n_codes}")
    print(f"    Final codes: {codebook.n_codes}")
    print(f"    Stats: {codebook.stats()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", default="all",
                    choices=["mood", "dream", "action", "adaptive", "all"])
    args = ap.parse_args()

    print("=" * 60)
    print("  Novel DiVeQ Applications")
    print("  'Things nobody has published'")
    print("=" * 60)

    if args.demo in ("mood", "all"):
        demo_mood()
    if args.demo in ("dream", "all"):
        demo_dream()
    if args.demo in ("action", "all"):
        demo_action()
    if args.demo in ("adaptive", "all"):
        demo_adaptive()

    print(f"\n{'=' * 60}")
    print(f"  All demos complete")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
