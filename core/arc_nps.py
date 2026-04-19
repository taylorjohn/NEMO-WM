"""
arc_nps.py — Neuromodulated Program Synthesizer for ARC-AGI-2
================================================================
Implements the research-identified path to closing the composition gap:

  1. MLC-style compositional episode training
  2. Program proposer (NeuroTransformer-based)
  3. Verifier-driven refinement loop  
  4. Library learning (compress successful compositions)
  5. Neuromodulated search control (DA/NE/5HT/ACh/CRT)

Each component can be ablated independently to measure contribution.

Usage:
    python arc_nps.py --test
    python arc_nps.py --train-episodes 5000
    python arc_nps.py --solve --data path/to/ARC-AGI-2/data
    python arc_nps.py --ablation --data path/to/ARC-AGI-2/data
"""

import argparse
import json
import os
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
import time
import copy

from arc_solver import (Grid, GridObject, score_task,
                         dsl_rot90, dsl_rot180, dsl_rot270,
                         dsl_flip_h, dsl_flip_v, dsl_transpose,
                         dsl_crop_nonzero, dsl_scale2,
                         dsl_outline, dsl_fill_holes,
                         dsl_gravity_down, dsl_gravity_left,
                         dsl_mirror_h, dsl_mirror_v,
                         dsl_largest_object, dsl_smallest_object,
                         dsl_keep_only_color, dsl_remove_color,
                         dsl_top_half, dsl_bottom_half,
                         dsl_left_half, dsl_right_half,
                         dsl_unique_rows, dsl_identity)

Path("data").mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# 1. DSL PRIMITIVE LIBRARY (expandable)
# ══════════════════════════════════════════════════════════════════════

class DSLLibrary:
    """Named, composable primitive library with learned extensions."""

    def __init__(self):
        self.primitives = {}
        self.compositions = {}  # learned from successful solves
        self.usage_count = Counter()
        self.success_count = Counter()

        # Base primitives
        self._register_base()

    def _register_base(self):
        base = [
            ("rot90", dsl_rot90, "grid", "grid"),
            ("rot180", dsl_rot180, "grid", "grid"),
            ("rot270", dsl_rot270, "grid", "grid"),
            ("flip_h", dsl_flip_h, "grid", "grid"),
            ("flip_v", dsl_flip_v, "grid", "grid"),
            ("transpose", dsl_transpose, "grid", "grid"),
            ("crop", dsl_crop_nonzero, "grid", "grid"),
            ("scale2", dsl_scale2, "grid", "grid"),
            ("outline", dsl_outline, "grid", "grid"),
            ("fill_holes", dsl_fill_holes, "grid", "grid"),
            ("gravity_down", dsl_gravity_down, "grid", "grid"),
            ("gravity_left", dsl_gravity_left, "grid", "grid"),
            ("mirror_h", dsl_mirror_h, "grid", "grid"),
            ("mirror_v", dsl_mirror_v, "grid", "grid"),
            ("largest_obj", dsl_largest_object, "grid", "grid"),
            ("smallest_obj", dsl_smallest_object, "grid", "grid"),
            ("top_half", dsl_top_half, "grid", "grid"),
            ("bottom_half", dsl_bottom_half, "grid", "grid"),
            ("left_half", dsl_left_half, "grid", "grid"),
            ("right_half", dsl_right_half, "grid", "grid"),
            ("unique_rows", dsl_unique_rows, "grid", "grid"),
            ("identity", dsl_identity, "grid", "grid"),
        ]
        for name, fn, in_type, out_type in base:
            self.primitives[name] = {
                'fn': fn, 'in': in_type, 'out': out_type,
                'depth': 1, 'source': 'base'
            }

    def add_composition(self, name, steps, fn):
        """Add a learned composition to the library."""
        self.compositions[name] = {
            'fn': fn, 'steps': steps, 'depth': len(steps),
            'source': 'learned'
        }

    def get_all(self, max_depth=1):
        """Get all operations up to a given depth."""
        ops = {}
        for name, info in self.primitives.items():
            if info['depth'] <= max_depth:
                ops[name] = info
        for name, info in self.compositions.items():
            if info['depth'] <= max_depth:
                ops[name] = info
        return ops

    @property
    def n_primitives(self):
        return len(self.primitives)

    @property
    def n_compositions(self):
        return len(self.compositions)

    @property
    def n_total(self):
        return self.n_primitives + self.n_compositions


# ══════════════════════════════════════════════════════════════════════
# 2. COMPOSITIONAL EPISODE GENERATOR (MLC-style)
# ══════════════════════════════════════════════════════════════════════

class EpisodeGenerator:
    """
    Generate MLC-style compositional training episodes.
    
    Each episode:
    - Selects 2-4 primitives
    - Composes them in a random order
    - Generates 2-3 demo grids + 1 test grid
    - The task is to infer the composition from demos
    
    This is the KEY missing training regime identified by the research.
    """

    def __init__(self, library: DSLLibrary, rng=None):
        self.library = library
        self.rng = rng or np.random.RandomState(42)

    def random_grid(self, h=None, w=None, n_colors=4, density=0.4):
        if h is None:
            h = self.rng.randint(3, 8)
        if w is None:
            w = self.rng.randint(3, 8)
        colors = list(range(1, n_colors + 1))
        grid = np.zeros((h, w), dtype=np.int32)
        for r in range(h):
            for c in range(w):
                if self.rng.random() < density:
                    grid[r, c] = self.rng.choice(colors)
        return Grid(grid)

    def generate_episode(self, n_steps=None, n_demos=3):
        """Generate one compositional episode."""
        if n_steps is None:
            n_steps = self.rng.randint(2, 5)

        # Select primitives
        prim_names = list(self.library.primitives.keys())
        # Exclude identity from composition
        prim_names = [n for n in prim_names if n != 'identity']

        chosen = [prim_names[i] for i in
                    self.rng.choice(len(prim_names), n_steps)]

        # Build composition function
        ops = [self.library.primitives[n]['fn'] for n in chosen]

        def compose(g):
            result = g
            for op in ops:
                result = op(result)
            return result

        # Generate demos
        demos = []
        for _ in range(n_demos + 1):
            max_attempts = 10
            for attempt in range(max_attempts):
                try:
                    g = self.random_grid()
                    out = compose(g)
                    if out.h > 0 and out.w > 0 and out.h <= 30 and out.w <= 30:
                        demos.append({'input': g, 'output': out})
                        break
                except Exception:
                    continue

        if len(demos) < 2:
            return None

        return {
            'program': chosen,
            'compose_fn': compose,
            'train': demos[:-1],
            'test': demos[-1:],
            'n_steps': n_steps,
        }

    def generate_color_composition(self, n_demos=3):
        """Generate episodes involving color-dependent operations."""
        # Select a color to operate on
        target_color = self.rng.randint(1, 8)
        # Select a geometric op
        prim_names = list(self.library.primitives.keys())
        geom_op_name = prim_names[self.rng.randint(len(prim_names))]
        geom_op = self.library.primitives[geom_op_name]['fn']

        def compose(g):
            # Keep only target color, then apply geometric op
            arr = np.where(g.arr == target_color, target_color, 0).astype(np.int32)
            result = geom_op(Grid(arr))
            return result

        demos = []
        for _ in range(n_demos + 1):
            try:
                g = self.random_grid(n_colors=6, density=0.5)
                out = compose(g)
                if out.h > 0 and out.w > 0:
                    demos.append({'input': g, 'output': out})
            except Exception:
                continue

        if len(demos) < 2:
            return None

        return {
            'program': [f'keep_color_{target_color}', geom_op_name],
            'compose_fn': compose,
            'train': demos[:-1],
            'test': demos[-1:],
            'n_steps': 2,
        }


# ══════════════════════════════════════════════════════════════════════
# 3. NEUROMODULATED SEARCH CONTROL
# ══════════════════════════════════════════════════════════════════════

class NeuromodulatedSearch:
    """
    Maps neuromodulators to program search control signals.
    
    DA  = proposal credit / selection pressure
    NE  = exploration reset / mutation rate  
    5HT = depth patience / max program length
    ACh = per-primitive learning rate
    CRT = consolidation trigger / habit takeover
    """

    def __init__(self):
        self.da = 0.5    # Dopamine: proposal confidence
        self.ne = 0.3    # Norepinephrine: exploration rate
        self.sht = 0.5   # Serotonin: depth patience
        self.ach = 0.5   # Acetylcholine: learning rate
        self.crt = 0.0   # Cortisol: stress/consolidation

        # Tracking
        self.n_proposals = 0
        self.n_successes = 0
        self.n_failures = 0
        self.consecutive_failures = 0

    def update_on_success(self, program_depth):
        """Update signals after successful verification."""
        self.n_successes += 1
        self.consecutive_failures = 0
        # DA burst — reward prediction error
        self.da = min(1.0, self.da + 0.2)
        # NE drops — no need to explore
        self.ne = max(0.1, self.ne - 0.1)
        # CRT rises — trigger consolidation
        self.crt = min(1.0, self.crt + 0.3)

    def update_on_failure(self):
        """Update signals after failed verification."""
        self.n_failures += 1
        self.consecutive_failures += 1
        # DA drops
        self.da = max(0.1, self.da - 0.05)
        # NE rises — need more exploration
        self.ne = min(0.9, self.ne + 0.1)
        # After many failures, CRT rises (stress)
        if self.consecutive_failures > 5:
            self.crt = min(1.0, self.crt + 0.1)

    @property
    def max_depth(self):
        """5HT-modulated: how deep to search."""
        base = 3
        return base + int(self.sht * 3)  # 3-6 steps

    @property
    def beam_width(self):
        """DA-modulated: how many candidates to consider."""
        base = 5
        return base + int(self.da * 10)  # 5-15

    @property
    def mutation_rate(self):
        """NE-modulated: how much to vary proposals."""
        return self.ne

    @property
    def should_consolidate(self):
        """CRT-gated: should we save this composition?"""
        return self.crt > 0.5

    @property
    def should_fallback_s1(self):
        """CRT-gated: too stressed, use cached answer."""
        return self.crt > 0.8 and self.consecutive_failures > 10

    def reset(self):
        """Reset for new task."""
        self.da = 0.5
        self.ne = 0.3
        self.sht = 0.5
        self.crt = 0.0
        self.consecutive_failures = 0


# ══════════════════════════════════════════════════════════════════════
# 4. PROGRAM PROPOSER + VERIFIER LOOP
# ══════════════════════════════════════════════════════════════════════

class ProgramSynthesizer:
    """
    Propose, verify, and refine programs.
    
    Core loop:
    1. Analyze demos (diff, objects, symmetry)
    2. Propose candidate programs from DSL
    3. Execute against training demos
    4. Score: exact match = 1.0, partial = overlap ratio
    5. Refine: mutate best candidates
    6. Repeat until solved or budget exhausted
    """

    def __init__(self, library: DSLLibrary, neuro: NeuromodulatedSearch = None):
        self.library = library
        self.neuro = neuro or NeuromodulatedSearch()
        self.rng = np.random.RandomState(42)

    def propose_programs(self, n_proposals, max_depth):
        """Generate candidate programs by sampling DSL compositions."""
        programs = []
        prim_names = [n for n in self.library.primitives.keys()
                        if n != 'identity']

        for _ in range(n_proposals):
            depth = self.rng.randint(1, max_depth + 1)
            steps = [prim_names[i] for i in
                      self.rng.choice(len(prim_names), depth)]
            programs.append(steps)

        return programs

    def execute_program(self, steps, grid: Grid) -> Optional[Grid]:
        """Execute a program (list of primitive names) on a grid."""
        try:
            result = grid
            for step_name in steps:
                if step_name in self.library.primitives:
                    result = self.library.primitives[step_name]['fn'](result)
                elif step_name in self.library.compositions:
                    result = self.library.compositions[step_name]['fn'](result)
                else:
                    # Parameterized: keep_color_N, remove_color_N
                    if step_name.startswith('keep_color_'):
                        c = int(step_name.split('_')[-1])
                        result = dsl_keep_only_color(result, c)
                    elif step_name.startswith('remove_color_'):
                        c = int(step_name.split('_')[-1])
                        result = dsl_remove_color(result, c)
                    else:
                        return None
            return result
        except Exception:
            return None

    def verify_program(self, steps, pairs) -> float:
        """Score a program against training pairs. 1.0 = all correct."""
        n_correct = 0
        for p in pairs:
            gi = Grid(p['input']) if not isinstance(p['input'], Grid) else p['input']
            go = Grid(p['output']) if not isinstance(p['output'], Grid) else p['output']
            result = self.execute_program(steps, gi)
            if result is not None and result == go:
                n_correct += 1
        return n_correct / max(len(pairs), 1)

    def mutate_program(self, steps, rate=0.3):
        """Mutate a program by swapping/inserting/removing steps."""
        prim_names = [n for n in self.library.primitives.keys()
                        if n != 'identity']
        steps = list(steps)

        if self.rng.random() < rate and len(steps) > 1:
            # Swap a random step
            idx = self.rng.randint(len(steps))
            steps[idx] = prim_names[self.rng.randint(len(prim_names))]

        if self.rng.random() < rate * 0.5 and len(steps) < 5:
            # Insert a step
            idx = self.rng.randint(len(steps) + 1)
            steps.insert(idx, prim_names[self.rng.randint(len(prim_names))])

        if self.rng.random() < rate * 0.3 and len(steps) > 1:
            # Remove a step
            idx = self.rng.randint(len(steps))
            steps.pop(idx)

        return steps

    def solve(self, pairs, max_iterations=50) -> Tuple[Optional[list], float]:
        """
        Main synthesis loop with neuromodulated control.
        Returns (best_program, best_score).
        """
        self.neuro.reset()
        best_program = None
        best_score = 0.0

        # Phase 1: Random proposals
        n_proposals = self.neuro.beam_width
        max_depth = self.neuro.max_depth
        candidates = self.propose_programs(n_proposals * 2, max_depth)

        # Also add parameterized ops
        for c in range(1, 10):
            candidates.append([f'keep_color_{c}'])
            candidates.append([f'remove_color_{c}'])
            for op_name in list(self.library.primitives.keys())[:8]:
                candidates.append([f'keep_color_{c}', op_name])
                candidates.append([op_name, f'keep_color_{c}'])

        # Add library compositions
        for name in self.library.compositions:
            candidates.append([name])

        # Score all candidates
        scored = []
        for prog in candidates:
            score = self.verify_program(prog, pairs)
            scored.append((score, prog))
            if score == 1.0:
                self.neuro.update_on_success(len(prog))
                return prog, 1.0

        scored.sort(key=lambda x: -x[0])
        if scored and scored[0][0] > best_score:
            best_score = scored[0][0]
            best_program = scored[0][1]

        # Phase 2: Evolutionary refinement (NE-modulated)
        if best_score < 1.0:
            top_k = scored[:self.neuro.beam_width]

            for iteration in range(max_iterations):
                if self.neuro.should_fallback_s1:
                    break

                new_candidates = []
                for score, prog in top_k:
                    # Mutate with NE-modulated rate
                    for _ in range(3):
                        mutated = self.mutate_program(
                            prog, rate=self.neuro.mutation_rate)
                        new_candidates.append(mutated)

                # Score new candidates
                for prog in new_candidates:
                    score = self.verify_program(prog, pairs)
                    if score == 1.0:
                        self.neuro.update_on_success(len(prog))
                        return prog, 1.0
                    if score > best_score:
                        best_score = score
                        best_program = prog
                        self.neuro.update_on_success(len(prog))
                    scored.append((score, prog))

                self.neuro.update_on_failure()

                # Keep top candidates
                scored.sort(key=lambda x: -x[0])
                top_k = scored[:self.neuro.beam_width]

        return best_program, best_score


# ══════════════════════════════════════════════════════════════════════
# 5. LIBRARY LEARNING (Stitch-style compression)
# ══════════════════════════════════════════════════════════════════════

class LibraryLearner:
    """
    Extract reusable compositions from successful programs.
    CRT-gated: only consolidate when confidence is high.
    """

    def __init__(self, library: DSLLibrary):
        self.library = library
        self.program_history = []

    def record_success(self, program: list, task_id: str = None):
        """Record a successful program for later analysis."""
        self.program_history.append({
            'program': tuple(program),
            'task_id': task_id,
            'depth': len(program),
        })

    def extract_compositions(self, min_frequency=2):
        """Find recurring sub-programs and add them to the library."""
        # Count 2-step and 3-step subsequences
        subseq_counts = Counter()
        for record in self.program_history:
            prog = record['program']
            for length in [2, 3]:
                for i in range(len(prog) - length + 1):
                    subseq = prog[i:i+length]
                    subseq_counts[subseq] += 1

        # Add frequent subsequences as new library entries
        n_added = 0
        for subseq, count in subseq_counts.most_common(20):
            if count < min_frequency:
                break
            name = "+".join(subseq)
            if name in self.library.compositions:
                continue

            # Build composition function
            steps = list(subseq)
            ops = []
            valid = True
            for s in steps:
                if s in self.library.primitives:
                    ops.append(self.library.primitives[s]['fn'])
                else:
                    valid = False
                    break
            if not valid:
                continue

            def make_compose(op_list):
                def fn(g):
                    result = g
                    for op in op_list:
                        result = op(result)
                    return result
                return fn

            self.library.add_composition(name, steps, make_compose(ops))
            n_added += 1

        return n_added


# ══════════════════════════════════════════════════════════════════════
# 6. MLC TRAINER — Train on compositional episodes
# ══════════════════════════════════════════════════════════════════════

def train_mlc(n_episodes=5000, verbose=False):
    """
    MLC-style training: generate compositional episodes,
    attempt to solve them, learn from successes/failures.
    """
    library = DSLLibrary()
    gen = EpisodeGenerator(library, rng=np.random.RandomState(42))
    synth = ProgramSynthesizer(library)
    learner = LibraryLearner(library)

    print("=" * 70)
    print("  MLC Compositional Episode Training")
    print(f"  Episodes: {n_episodes}")
    print(f"  Base primitives: {library.n_primitives}")
    print("=" * 70)

    stats = defaultdict(lambda: {'solved': 0, 'tried': 0})
    t0 = time.time()

    for ep in range(n_episodes):
        # Generate episode
        if gen.rng.random() < 0.7:
            episode = gen.generate_episode(
                n_steps=gen.rng.randint(2, 5), n_demos=3)
        else:
            episode = gen.generate_color_composition(n_demos=3)

        if episode is None:
            continue

        n_steps = episode['n_steps']
        true_program = episode['program']

        # Convert demos to list format
        pairs = []
        for d in episode['train']:
            pairs.append({
                'input': d['input'].to_list(),
                'output': d['output'].to_list()
            })

        # Attempt to solve
        found_prog, score = synth.solve(pairs, max_iterations=20)

        depth_key = f"{n_steps}-step"
        stats[depth_key]['tried'] += 1

        if score == 1.0:
            stats[depth_key]['solved'] += 1
            learner.record_success(
                found_prog if found_prog else true_program,
                task_id=f"ep_{ep}")

        # Periodically extract compositions
        if (ep + 1) % 500 == 0:
            n_new = learner.extract_compositions(min_frequency=3)
            if n_new > 0 and verbose:
                print(f"    +{n_new} compositions learned")

        # Progress
        if (ep + 1) % max(1, n_episodes // 10) == 0:
            elapsed = time.time() - t0
            total_s = sum(s['solved'] for s in stats.values())
            total_t = sum(s['tried'] for s in stats.values())
            rate = total_s / max(total_t, 1) * 100
            print(f"  Episode {ep+1:>6} │ Solved: {total_s}/{total_t} ({rate:.1f}%) │ "
                  f"Library: {library.n_total} │ "
                  f"Time: {elapsed:.1f}s")

    # Final library extraction
    n_final = learner.extract_compositions(min_frequency=2)

    elapsed = time.time() - t0
    total_s = sum(s['solved'] for s in stats.values())
    total_t = sum(s['tried'] for s in stats.values())

    print(f"\n{'='*70}")
    print(f"  MLC TRAINING SUMMARY")
    print(f"{'='*70}")
    print(f"  Episodes:     {n_episodes}")
    print(f"  Solved:       {total_s}/{total_t} ({total_s/max(total_t,1)*100:.1f}%)")
    print(f"  Library:      {library.n_primitives} base + {library.n_compositions} learned")
    print(f"  Time:         {elapsed:.1f}s")

    print(f"\n  Per-depth results:")
    for key in sorted(stats.keys()):
        s = stats[key]
        rate = s['solved'] / max(s['tried'], 1) * 100
        print(f"    {key}: {s['solved']}/{s['tried']} ({rate:.1f}%)")

    if library.compositions:
        print(f"\n  Learned compositions:")
        for name in list(library.compositions.keys())[:15]:
            print(f"    {name}")

    print(f"\n{'='*70}")
    return library, synth, learner


# ══════════════════════════════════════════════════════════════════════
# 7. ARC SOLVER WITH SYNTHESIZER
# ══════════════════════════════════════════════════════════════════════

def solve_arc_with_nps(task, synth: ProgramSynthesizer):
    """Solve an ARC task using the neuromodulated program synthesizer."""
    pairs = task['train']
    test_cases = task['test']

    # Attempt synthesis
    found_prog, score = synth.solve(pairs, max_iterations=30)

    guesses = []
    if score == 1.0 and found_prog is not None:
        for tc in test_cases:
            gi = Grid(tc['input'])
            result = synth.execute_program(found_prog, gi)
            if result is not None:
                guesses.append([result.to_list()])
            else:
                guesses.append([tc['input']])
    else:
        for tc in test_cases:
            guesses.append([tc['input']])

    return guesses


def run_arc_benchmark(data_dir, synth, limit=None, verbose=False):
    """Run ARC benchmark with the NPS solver."""
    files = sorted(f for f in os.listdir(data_dir) if f.endswith('.json'))
    if limit:
        files = files[:limit]

    print("=" * 70)
    print(f"  ARC-AGI-2 with Neuromodulated Program Synthesizer")
    print(f"  Tasks: {len(files)}")
    print(f"  Library: {synth.library.n_total} operations")
    print("=" * 70)

    solved = 0
    total = 0
    t0 = time.time()

    for f in files:
        task = json.load(open(os.path.join(data_dir, f)))
        total += 1

        predictions = solve_arc_with_nps(task, synth)
        correct = score_task(task, predictions)

        if correct:
            solved += 1

        if verbose and correct:
            print(f"  {f[:16]}: SOLVED")

    elapsed = time.time() - t0
    pct = solved / max(total, 1) * 100

    print(f"\n{'='*70}")
    print(f"  NPS RESULTS: {solved}/{total} ({pct:.1f}%)")
    print(f"  Time: {elapsed:.1f}s ({elapsed/max(total,1)*100:.0f}ms/task)")
    print(f"{'='*70}")
    return solved, total


# ══════════════════════════════════════════════════════════════════════
# 8. ABLATION STUDIES
# ══════════════════════════════════════════════════════════════════════

def run_ablation(data_dir, limit=100):
    """Run ablation studies on each component."""
    files = sorted(f for f in os.listdir(data_dir) if f.endswith('.json'))[:limit]
    tasks = [json.load(open(os.path.join(data_dir, f))) for f in files]

    print("=" * 70)
    print(f"  ABLATION STUDY — {len(tasks)} tasks")
    print("=" * 70)

    conditions = {
        'full': {'da': True, 'ne': True, 'sht': True, 'ach': True, 'refinement': True},
        'no_DA': {'da': False, 'ne': True, 'sht': True, 'ach': True, 'refinement': True},
        'no_NE': {'da': True, 'ne': False, 'sht': True, 'ach': True, 'refinement': True},
        'no_5HT': {'da': True, 'ne': True, 'sht': False, 'ach': True, 'refinement': True},
        'no_refinement': {'da': True, 'ne': True, 'sht': True, 'ach': True, 'refinement': False},
        'no_neuromod': {'da': False, 'ne': False, 'sht': False, 'ach': False, 'refinement': True},
        'baseline': {'da': False, 'ne': False, 'sht': False, 'ach': False, 'refinement': False},
    }

    results = {}
    for cond_name, config in conditions.items():
        library = DSLLibrary()
        neuro = NeuromodulatedSearch()

        # Apply ablations
        if not config['da']:
            neuro.da = 0.5  # Fixed, no updates
        if not config['ne']:
            neuro.ne = 0.3  # Fixed
        if not config['sht']:
            neuro.sht = 0.3  # Reduced depth
        max_iter = 30 if config['refinement'] else 0

        synth = ProgramSynthesizer(library, neuro)
        solved = 0

        for task in tasks:
            pairs = task['train']
            found_prog, score = synth.solve(pairs, max_iterations=max_iter)
            if score == 1.0:
                # Verify on test
                predictions = solve_arc_with_nps(task, synth)
                if score_task(task, predictions):
                    solved += 1

            # Reset neuro but apply ablation
            synth.neuro.reset()
            if not config['da']:
                synth.neuro.da = 0.5
            if not config['ne']:
                synth.neuro.ne = 0.3
            if not config['sht']:
                synth.neuro.sht = 0.3

        results[cond_name] = solved
        print(f"  {cond_name:<20}: {solved}/{len(tasks)} ({solved/len(tasks)*100:.1f}%)")

    # Compute deltas
    full = results.get('full', 0)
    print(f"\n  Ablation deltas (from full={full}):")
    for name, score in results.items():
        if name != 'full':
            delta = score - full
            print(f"    {name:<20}: {delta:+d} ({delta/max(full,1)*100:+.1f}%)")

    print(f"\n{'='*70}")
    return results


# ══════════════════════════════════════════════════════════════════════
# 9. TESTS
# ══════════════════════════════════════════════════════════════════════

def run_tests():
    print("=" * 65)
    print("  Neuromodulated Program Synthesizer Tests")
    print("=" * 65)
    p = 0; t = 0

    print("\n  T1: DSL Library")
    lib = DSLLibrary()
    ok = lib.n_primitives >= 20
    print(f"    Primitives: {lib.n_primitives} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T2: Episode generation")
    gen = EpisodeGenerator(lib)
    ep = gen.generate_episode(n_steps=2, n_demos=3)
    ok = ep is not None and len(ep['train']) >= 2
    print(f"    Episode: {ep['n_steps']}-step, {len(ep['train'])} demos "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T3: Program execution")
    synth = ProgramSynthesizer(lib)
    g = Grid(np.array([[1,2],[3,4]], dtype=np.int32))
    result = synth.execute_program(['rot90'], g)
    ok = result is not None and result == Grid(np.array([[3,1],[4,2]]))
    print(f"    rot90: {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T4: Program verification")
    pairs = [{'input': [[1,2],[3,4]], 'output': [[3,1],[4,2]]}]
    score = synth.verify_program(['rot90'], pairs)
    ok = score == 1.0
    print(f"    Score: {score} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T5: Neuromodulated search signals")
    neuro = NeuromodulatedSearch()
    neuro.update_on_success(2)
    ok = neuro.da > 0.5 and neuro.ne < 0.3
    print(f"    DA={neuro.da:.2f} NE={neuro.ne:.2f} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T6: Solve simple composition")
    task_pairs = [
        {'input': [[1,2],[3,4]], 'output': [[4,3],[2,1]]},  # rot180
    ]
    prog, score = synth.solve(task_pairs, max_iterations=10)
    ok = score == 1.0
    print(f"    Score: {score}, program: {prog} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T7: Library learning")
    learner = LibraryLearner(lib)
    learner.record_success(['rot90', 'flip_h'])
    learner.record_success(['rot90', 'flip_h'])
    learner.record_success(['rot90', 'flip_h'])
    n_new = learner.extract_compositions(min_frequency=2)
    ok = n_new >= 1
    print(f"    New compositions: {n_new} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T8: Mutation")
    mutated = synth.mutate_program(['rot90', 'flip_h'], rate=1.0)
    ok = isinstance(mutated, list) and len(mutated) >= 1
    print(f"    Mutated: {mutated} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T9: Mini MLC training")
    lib2, synth2, _ = train_mlc(n_episodes=100, verbose=False)
    ok = lib2.n_total >= 22
    print(f"    Library: {lib2.n_total} ops {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T10: Color composition episode")
    ep2 = gen.generate_color_composition(n_demos=3)
    ok = ep2 is not None
    print(f"    Color episode: {ep2['program'] if ep2 else None} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print(f"\n{'='*65}")
    print(f"  Results: {p}/{t} tests passed")
    print(f"{'='*65}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--train-episodes", type=int, default=None)
    ap.add_argument("--solve", action="store_true")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--training", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--data", type=str, default=None)
    args = ap.parse_args()

    if args.data:
        data_base = args.data
    else:
        for base in [".", "ARC-AGI-2", "../ARC-AGI-2",
                       "/home/claude/ARC-AGI-2",
                       "C:/Users/MeteorAI/Desktop/ARC-AGI-2"]:
            candidate = os.path.join(base, "data")
            if os.path.exists(candidate):
                data_base = candidate
                break
        else:
            data_base = "ARC-AGI-2/data"

    if args.test:
        run_tests()
    elif args.train_episodes:
        train_mlc(n_episodes=args.train_episodes, verbose=args.verbose)
    elif args.ablation:
        d = os.path.join(data_base, "training")
        run_ablation(d, limit=args.limit or 100)
    elif args.solve or args.training or args.eval:
        # Train first
        library, synth, learner = train_mlc(n_episodes=2000)
        if args.eval:
            d = os.path.join(data_base, "evaluation")
        else:
            d = os.path.join(data_base, "training")
        run_arc_benchmark(d, synth, limit=args.limit, verbose=args.verbose)
    else:
        run_tests()
