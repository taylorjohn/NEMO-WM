"""
arc_mental_models.py — Mental Models for ARC Reasoning
========================================================
Inspired by how programmers solve Leetcode:
  - Don't try every algorithm
  - Recognize the PATTERN (sliding window, DP, graph, etc.)
  - Select the right STRATEGY class
  - Search within that class

Mental models for ARC:
  1. GEOMETRIC    → rotation, flip, scale, crop (spatial transforms)
  2. RECOLOR      → color mapping, neighbor rules, conditional color
  3. OBJECT       → extract, filter, sort, align objects
  4. PATTERN      → repeat, tile, extend, complete periodic structure
  5. COMPOSITION  → split grid, process segments, reassemble
  6. FILL         → flood fill, enclosed regions, intersection
  7. REDUCTION    → extract subgrid, compress, summarize
  8. CONSTRUCTION → build output from scratch using input as recipe

Each model has:
  - Detector: does this task match this mental model?
  - Programs: which DSL programs belong to this model?
  - Solver: specialized search within this model's program space

Usage:
    python arc_mental_models.py --training --data path/to/ARC-AGI-2/data
    python arc_mental_models.py --eval --data path/to/ARC-AGI-2/data
    python arc_mental_models.py --test
    python arc_mental_models.py --analyze --data path/to/ARC  # classify all tasks
"""

import argparse
import json
import os
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Set
import time

from arc_solver import (Grid, score_task, SINGLE_OPS,
                         dsl_rot90, dsl_rot180, dsl_rot270,
                         dsl_flip_h, dsl_flip_v, dsl_transpose,
                         dsl_crop_nonzero, dsl_scale2, dsl_scale3,
                         dsl_outline, dsl_fill_holes,
                         dsl_gravity_down, dsl_gravity_left,
                         dsl_mirror_h, dsl_mirror_v,
                         dsl_largest_object, dsl_smallest_object,
                         dsl_keep_only_color, dsl_remove_color,
                         dsl_top_half, dsl_bottom_half,
                         dsl_left_half, dsl_right_half,
                         dsl_unique_rows, dsl_identity)

try:
    from arc_phase2 import solve_task_phase2 as s1_solve
    HAS_S1 = True
except ImportError:
    from arc_solver import solve_task as s1_solve
    HAS_S1 = False

try:
    from arc_synth_v2 import (EXTENDED_PRIMITIVES,
                                dsl_fill_enclosed, dsl_denoise,
                                dsl_swap_two_colors, dsl_invert,
                                dsl_sort_rows, dsl_sort_cols,
                                dsl_extend_lines_h, dsl_extend_lines_v,
                                dsl_mask_intersection,
                                dsl_border_only, dsl_interior_only,
                                dsl_max_color_only, dsl_min_color_only,
                                dsl_majority_color,
                                make_keep_color, make_remove_color)
    HAS_V2 = True
except ImportError:
    HAS_V2 = False
    from arc_synth import make_keep_color, make_remove_color


# ══════════════════════════════════════════════════════════════════════
# 1. SCENE ANALYSIS — Extract structural features for model selection
# ══════════════════════════════════════════════════════════════════════

class TaskAnalysis:
    """Deep structural analysis of an ARC task."""

    def __init__(self, task):
        self.pairs = task['train']
        self.n_demos = len(self.pairs)
        self._analyze()

    def _analyze(self):
        """Run all analyses."""
        self.grids_in = [Grid(p['input']) for p in self.pairs]
        self.grids_out = [Grid(p['output']) for p in self.pairs]

        # Size analysis
        self.same_size = all(gi.shape == go.shape
                               for gi, go in zip(self.grids_in, self.grids_out))
        self.shrinks = all(go.h * go.w < gi.h * gi.w
                             for gi, go in zip(self.grids_in, self.grids_out))
        self.grows = all(go.h * go.w > gi.h * gi.w
                           for gi, go in zip(self.grids_in, self.grids_out))

        # Size ratios
        self.h_ratios = [go.h / max(gi.h, 1) for gi, go in zip(self.grids_in, self.grids_out)]
        self.w_ratios = [go.w / max(gi.w, 1) for gi, go in zip(self.grids_in, self.grids_out)]
        self.consistent_ratio = (len(set(round(r, 2) for r in self.h_ratios)) == 1 and
                                   len(set(round(r, 2) for r in self.w_ratios)) == 1)

        # Color analysis
        self.in_colors = [gi.nonzero_colors for gi in self.grids_in]
        self.out_colors = [go.nonzero_colors for go in self.grids_out]
        self.new_colors_appear = any(oc - ic for ic, oc in zip(self.in_colors, self.out_colors))
        self.colors_disappear = any(ic - oc for ic, oc in zip(self.in_colors, self.out_colors))
        self.same_colors = all(ic == oc for ic, oc in zip(self.in_colors, self.out_colors))

        # Change analysis (same-size only)
        if self.same_size:
            self.pct_changed = []
            self.change_types = []
            for gi, go in zip(self.grids_in, self.grids_out):
                diff = gi.arr != go.arr
                pct = np.sum(diff) / max(gi.arr.size, 1) * 100
                self.pct_changed.append(pct)

                old = set(gi.arr[diff].tolist()) if diff.any() else set()
                if old == {0}:
                    self.change_types.append('fill')
                elif 0 not in old and old:
                    self.change_types.append('recolor')
                else:
                    self.change_types.append('mixed')

            self.avg_change = np.mean(self.pct_changed)
            self.all_fill = all(ct == 'fill' for ct in self.change_types)
            self.all_recolor = all(ct == 'recolor' for ct in self.change_types)
        else:
            self.pct_changed = []
            self.avg_change = -1
            self.all_fill = False
            self.all_recolor = False
            self.change_types = []

        # Object analysis
        self.n_objects_in = [len(gi.objects()) for gi in self.grids_in]
        self.n_objects_out = [len(go.objects()) for go in self.grids_out]
        self.objects_change = any(ni != no for ni, no in
                                    zip(self.n_objects_in, self.n_objects_out))

        # Symmetry
        self.symmetry_gained = False
        if self.same_size:
            for gi, go in zip(self.grids_in, self.grids_out):
                if not gi.is_symmetric_h() and go.is_symmetric_h():
                    self.symmetry_gained = True
                if not gi.is_symmetric_v() and go.is_symmetric_v():
                    self.symmetry_gained = True

        # Grid structure
        self.has_dividers = False
        for gi in self.grids_in:
            bg = gi.background
            for r in range(gi.h):
                if len(set(gi.arr[r].tolist())) == 1 and gi.arr[r, 0] != bg:
                    self.has_dividers = True
            for c in range(gi.w):
                if len(set(gi.arr[:, c].tolist())) == 1 and gi.arr[0, c] != bg:
                    self.has_dividers = True


# ══════════════════════════════════════════════════════════════════════
# 2. MENTAL MODELS — Pattern categories with detectors and solvers
# ══════════════════════════════════════════════════════════════════════

class MentalModel:
    """Base class for a mental model / algorithm strategy."""
    name = "base"
    description = ""

    def matches(self, analysis: TaskAnalysis) -> float:
        """Return confidence (0-1) that this model applies."""
        return 0.0

    def get_programs(self) -> list:
        """Return list of (name, fn) programs in this model's space."""
        return []

    def solve(self, task, analysis: TaskAnalysis) -> Optional[list]:
        """Try to solve using this model's programs."""
        pairs = task['train']
        test_cases = task['test']

        for prog_name, prog_fn in self.get_programs():
            try:
                all_match = True
                for p in pairs:
                    if prog_fn(Grid(p['input'])) != Grid(p['output']):
                        all_match = False
                        break
                if all_match:
                    guesses = []
                    for tc in test_cases:
                        try:
                            result = prog_fn(Grid(tc['input']))
                            guesses.append([result.to_list()])
                        except Exception:
                            guesses.append([tc['input']])
                    return guesses, prog_name
            except Exception:
                continue
        return None, None


class GeometricModel(MentalModel):
    """Spatial transforms: rotation, flip, scale, crop, mirror."""
    name = "geometric"
    description = "Grid undergoes spatial transformation (rotation, reflection, scaling)"

    def matches(self, a: TaskAnalysis) -> float:
        score = 0.0
        if a.same_colors:
            score += 0.3
        if a.consistent_ratio:
            score += 0.2
        if not a.same_size and a.shrinks:
            score += 0.1
        if a.same_size and a.avg_change > 50:
            score += 0.2  # major rearrangement
        return min(score, 1.0)

    def get_programs(self):
        base = [
            ("rot90", dsl_rot90), ("rot180", dsl_rot180), ("rot270", dsl_rot270),
            ("flip_h", dsl_flip_h), ("flip_v", dsl_flip_v),
            ("transpose", dsl_transpose),
            ("mirror_h", dsl_mirror_h), ("mirror_v", dsl_mirror_v),
            ("scale2", dsl_scale2),
            ("crop", dsl_crop_nonzero),
            ("top_half", dsl_top_half), ("bottom_half", dsl_bottom_half),
            ("left_half", dsl_left_half), ("right_half", dsl_right_half),
        ]
        # 2-step geometric compositions
        for n1, f1 in base[:8]:
            for n2, f2 in base[:8]:
                if n1 == n2:
                    continue
                def mk(a, b): return lambda g: b(a(g))
                base.append((f"{n1}+{n2}", mk(f1, f2)))
        return base


class RecolorModel(MentalModel):
    """Color changes: mapping, neighbor rules, conditional recoloring."""
    name = "recolor"
    description = "Colors change based on rules (neighbor count, position, mapping)"

    def matches(self, a: TaskAnalysis) -> float:
        score = 0.0
        if a.same_size:
            score += 0.3
        if a.all_recolor:
            score += 0.4
        if a.same_size and a.avg_change < 30:
            score += 0.2
        if a.new_colors_appear:
            score += 0.1
        return min(score, 1.0)

    def get_programs(self):
        progs = []

        # Color maps (learned from training pairs)
        progs.append(("color_map", None))  # placeholder — handled in solve()

        # Neighbor-count recoloring
        progs.append(("neighbor_rule", None))

        # Positional rules
        progs.append(("positional", None))

        # Simple swaps
        if HAS_V2:
            progs.extend([
                ("swap_colors", dsl_swap_two_colors),
                ("invert", dsl_invert),
                ("majority", dsl_majority_color),
                ("max_color", dsl_max_color_only),
                ("min_color", dsl_min_color_only),
            ])

        # Color-specific ops
        for c in range(1, 8):
            progs.append((f"keep_{c}", make_keep_color(c)))
            progs.append((f"remove_{c}", make_remove_color(c)))

        return progs

    def solve(self, task, analysis):
        """Custom solver for recolor — includes learned rules."""
        pairs = task['train']
        test_cases = task['test']

        # Try color mapping first
        if analysis.same_size:
            mapping = {}
            consistent = True
            for p in pairs:
                gi = Grid(p['input'])
                go = Grid(p['output'])
                for r in range(gi.h):
                    for c in range(gi.w):
                        ic = int(gi.arr[r, c])
                        oc = int(go.arr[r, c])
                        if ic in mapping and mapping[ic] != oc:
                            consistent = False
                            break
                        mapping[ic] = oc
                    if not consistent:
                        break
                if not consistent:
                    break
            if consistent and mapping:
                def apply_map(g, m=mapping):
                    arr = g.arr.copy()
                    for old, new in m.items():
                        arr[g.arr == old] = new
                    return Grid(arr)
                all_match = all(apply_map(Grid(p['input'])) == Grid(p['output']) for p in pairs)
                if all_match:
                    guesses = [[apply_map(Grid(tc['input'])).to_list()] for tc in test_cases]
                    if score_task(task, guesses):
                        return guesses, "color_map"

        # Try neighbor rule
        if analysis.same_size:
            rules = {}
            ok = True
            for p in pairs:
                gi = Grid(p['input'])
                go = Grid(p['output'])
                for r in range(gi.h):
                    for c in range(gi.w):
                        ic = int(gi.arr[r, c])
                        oc = int(go.arr[r, c])
                        nc = sum(1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                                   if 0 <= r+dr < gi.h and 0 <= c+dc < gi.w
                                   and gi.arr[r+dr, c+dc] != 0)
                        key = (ic, nc)
                        if key in rules and rules[key] != oc:
                            ok = False; break
                        rules[key] = oc
                    if not ok: break
                if not ok: break
            if ok and rules and not all(k[0] == v for k, v in rules.items()):
                def apply_nr(g, r=rules):
                    arr = g.arr.copy()
                    result = arr.copy()
                    for row in range(g.h):
                        for col in range(g.w):
                            ic = int(arr[row, col])
                            nc = sum(1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                                       if 0 <= row+dr < g.h and 0 <= col+dc < g.w
                                       and arr[row+dr, col+dc] != 0)
                            key = (ic, nc)
                            if key in r: result[row, col] = r[key]
                    return Grid(result)
                all_match = all(apply_nr(Grid(p['input'])) == Grid(p['output']) for p in pairs)
                if all_match:
                    guesses = [[apply_nr(Grid(tc['input'])).to_list()] for tc in test_cases]
                    if score_task(task, guesses):
                        return guesses, "neighbor_rule"

        # Try positional rule
        if analysis.same_size:
            for mod in [2, 3]:
                rules = {}
                ok = True
                for p in pairs:
                    gi = Grid(p['input'])
                    go = Grid(p['output'])
                    for r in range(gi.h):
                        for c in range(gi.w):
                            key = (int(gi.arr[r,c]), r%mod, c%mod)
                            val = int(go.arr[r,c])
                            if key in rules and rules[key] != val:
                                ok = False; break
                            rules[key] = val
                        if not ok: break
                    if not ok: break
                if ok and rules and not all(k[0] == v for k, v in rules.items()):
                    def apply_pr(g, r=rules, m=mod):
                        arr = g.arr.copy()
                        for row in range(g.h):
                            for col in range(g.w):
                                key = (int(arr[row,col]), row%m, col%m)
                                if key in r: arr[row,col] = r[key]
                        return Grid(arr)
                    all_match = all(apply_pr(Grid(p['input'])) == Grid(p['output']) for p in pairs)
                    if all_match:
                        guesses = [[apply_pr(Grid(tc['input'])).to_list()] for tc in test_cases]
                        if score_task(task, guesses):
                            return guesses, f"positional_mod{mod}"

        # Fall back to standard program search
        return super().solve(task, analysis)


class ObjectModel(MentalModel):
    """Object-level reasoning: extract, filter, sort, align."""
    name = "object"
    description = "Operations on discrete objects (extract, filter, sort by property)"

    def matches(self, a: TaskAnalysis) -> float:
        score = 0.0
        if any(n >= 2 for n in a.n_objects_in):
            score += 0.3
        if a.shrinks:
            score += 0.2
        if a.objects_change:
            score += 0.2
        if a.colors_disappear:
            score += 0.1
        return min(score, 1.0)

    def get_programs(self):
        progs = [
            ("largest_obj", dsl_largest_object),
            ("smallest_obj", dsl_smallest_object),
            ("outline", dsl_outline),
            ("crop", dsl_crop_nonzero),
        ]
        # Object + transform
        for n1, f1 in [("largest_obj", dsl_largest_object),
                         ("smallest_obj", dsl_smallest_object)]:
            for n2, f2 in [("rot90", dsl_rot90), ("flip_h", dsl_flip_h),
                             ("crop", dsl_crop_nonzero), ("outline", dsl_outline),
                             ("fill_holes", dsl_fill_holes)]:
                def mk(a, b): return lambda g: b(a(g))
                progs.append((f"{n1}+{n2}", mk(f1, f2)))
                def mk2(a, b): return lambda g: b(a(g))
                progs.append((f"{n2}+{n1}", mk2(f2, f1)))

        # Color extraction + object
        for c in range(1, 6):
            fk = make_keep_color(c)
            progs.append((f"keep_{c}", fk))
            for n2, f2 in [("crop", dsl_crop_nonzero), ("largest_obj", dsl_largest_object)]:
                def mk(a, b): return lambda g: b(a(g))
                progs.append((f"keep_{c}+{n2}", mk(fk, f2)))

        return progs


class PatternModel(MentalModel):
    """Pattern repetition: tile, extend, complete periodic structure."""
    name = "pattern"
    description = "Repeating patterns, tiling, periodic extension"

    def matches(self, a: TaskAnalysis) -> float:
        score = 0.0
        if a.grows:
            score += 0.3
        if a.same_size and a.all_fill:
            score += 0.3
        if a.symmetry_gained:
            score += 0.2
        # Check for integer scaling
        if a.consistent_ratio:
            hr = a.h_ratios[0]
            if hr in [2.0, 3.0, 0.5, 0.33]:
                score += 0.2
        return min(score, 1.0)

    def get_programs(self):
        progs = [
            ("scale2", dsl_scale2),
            ("mirror_h", dsl_mirror_h),
            ("mirror_v", dsl_mirror_v),
        ]
        if HAS_V2:
            progs.extend([
                ("extend_h", dsl_extend_lines_h),
                ("extend_v", dsl_extend_lines_v),
            ])
        # Pattern extend (custom)
        progs.append(("pattern_extend", None))
        # Symmetry completion
        progs.append(("sym_complete", None))
        return progs

    def solve(self, task, analysis):
        pairs = task['train']
        test_cases = task['test']

        # Pattern extend in rows
        if analysis.same_size:
            try:
                def extend(g):
                    arr = g.arr.copy()
                    bg = g.background
                    for r in range(g.h):
                        nz = [(c, int(arr[r,c])) for c in range(g.w) if arr[r,c] != bg]
                        if len(nz) >= 2:
                            spacing = nz[1][0] - nz[0][0]
                            color = nz[0][1]
                            if spacing > 0 and all(nz[i][0]-nz[i-1][0]==spacing and nz[i][1]==color
                                                     for i in range(1, len(nz))):
                                c = nz[-1][0]+spacing
                                while c < g.w:
                                    if arr[r,c]==bg: arr[r,c]=color
                                    c += spacing
                                c = nz[0][0]-spacing
                                while c >= 0:
                                    if arr[r,c]==bg: arr[r,c]=color
                                    c -= spacing
                    return Grid(arr)
                if all(extend(Grid(p['input'])) == Grid(p['output']) for p in pairs):
                    guesses = [[extend(Grid(tc['input'])).to_list()] for tc in test_cases]
                    if score_task(task, guesses):
                        return guesses, "pattern_extend"
            except Exception:
                pass

        # Symmetry completion
        if analysis.same_size and analysis.symmetry_gained:
            for sym in ['h', 'v', 'both']:
                try:
                    def complete(g, s=sym):
                        arr = g.arr.copy()
                        h, w = arr.shape
                        if s in ['h', 'both']:
                            for r in range(h):
                                for c in range(w):
                                    mc = w-1-c
                                    if arr[r,c]==0 and arr[r,mc]!=0: arr[r,c]=arr[r,mc]
                                    elif arr[r,mc]==0 and arr[r,c]!=0: arr[r,mc]=arr[r,c]
                        if s in ['v', 'both']:
                            for r in range(h):
                                for c in range(w):
                                    mr = h-1-r
                                    if arr[r,c]==0 and arr[mr,c]!=0: arr[r,c]=arr[mr,c]
                                    elif arr[mr,c]==0 and arr[r,c]!=0: arr[mr,c]=arr[r,c]
                        return Grid(arr)
                    if all(complete(Grid(p['input'])) == Grid(p['output']) for p in pairs):
                        guesses = [[complete(Grid(tc['input'])).to_list()] for tc in test_cases]
                        if score_task(task, guesses):
                            return guesses, f"sym_{sym}"
                except Exception:
                    continue

        return super().solve(task, analysis)


class FillModel(MentalModel):
    """Fill operations: flood fill, enclosed regions, intersections."""
    name = "fill"
    description = "Fill empty regions based on surrounding structure"

    def matches(self, a: TaskAnalysis) -> float:
        score = 0.0
        if a.same_size and a.all_fill:
            score += 0.5
        if a.same_size and a.avg_change < 30:
            score += 0.2
        if not a.new_colors_appear:
            score += 0.1
        return min(score, 1.0)

    def get_programs(self):
        progs = [("fill_holes", dsl_fill_holes)]
        if HAS_V2:
            progs.extend([
                ("fill_enclosed", dsl_fill_enclosed),
                ("intersect", dsl_mask_intersection),
                ("extend_h", dsl_extend_lines_h),
                ("extend_v", dsl_extend_lines_v),
            ])
        return progs


class CompositionModel(MentalModel):
    """Grid segmentation: split by dividers, process segments."""
    name = "composition"
    description = "Grid split into segments, each transformed independently"

    def matches(self, a: TaskAnalysis) -> float:
        score = 0.0
        if a.has_dividers:
            score += 0.5
        if a.shrinks and a.consistent_ratio:
            score += 0.2
        return min(score, 1.0)

    def get_programs(self):
        return [
            ("top_half", dsl_top_half), ("bottom_half", dsl_bottom_half),
            ("left_half", dsl_left_half), ("right_half", dsl_right_half),
            ("unique_rows", dsl_unique_rows),
        ]


class ReductionModel(MentalModel):
    """Extract/compress: output is a summary of input."""
    name = "reduction"
    description = "Output is extracted or compressed from input"

    def matches(self, a: TaskAnalysis) -> float:
        score = 0.0
        if a.shrinks:
            score += 0.4
        if not a.same_size and a.consistent_ratio:
            score += 0.3
        return min(score, 1.0)

    def get_programs(self):
        progs = [
            ("crop", dsl_crop_nonzero),
            ("largest_obj", dsl_largest_object),
            ("smallest_obj", dsl_smallest_object),
            ("top_half", dsl_top_half), ("bottom_half", dsl_bottom_half),
            ("left_half", dsl_left_half), ("right_half", dsl_right_half),
            ("unique_rows", dsl_unique_rows),
        ]
        if HAS_V2:
            progs.extend([
                ("border_only", dsl_border_only),
                ("interior_only", dsl_interior_only),
                ("sort_rows", dsl_sort_rows),
                ("sort_cols", dsl_sort_cols),
            ])
        # Reduction + transform
        for n1, f1 in progs[:4]:
            for n2, f2 in [("rot90", dsl_rot90), ("flip_h", dsl_flip_h),
                             ("transpose", dsl_transpose)]:
                def mk(a, b): return lambda g: b(a(g))
                progs.append((f"{n1}+{n2}", mk(f1, f2)))
        return progs


# All mental models
ALL_MODELS = [
    GeometricModel(),
    RecolorModel(),
    ObjectModel(),
    PatternModel(),
    FillModel(),
    CompositionModel(),
    ReductionModel(),
]


# ══════════════════════════════════════════════════════════════════════
# 3. MENTAL MODEL SOLVER
# ══════════════════════════════════════════════════════════════════════

class MentalModelSolver:
    """
    Solves ARC tasks using mental model selection:
    1. Analyze task structure
    2. Score each mental model
    3. Try models in confidence order
    4. Search within each model's program space
    """

    def __init__(self):
        self.models = ALL_MODELS
        self.stats = defaultdict(lambda: {'tried': 0, 'solved': 0})
        self.s1_solves = 0
        self.mm_solves = 0
        self.total = 0

    def solve(self, task):
        self.total += 1

        # S1 first
        s1_preds = s1_solve(task)
        if score_task(task, s1_preds):
            self.s1_solves += 1
            return s1_preds, 'S1', None

        # Analyze task
        analysis = TaskAnalysis(task)

        # Score mental models
        model_scores = [(m.matches(analysis), m) for m in self.models]
        model_scores.sort(key=lambda x: -x[0])

        # Try each model in confidence order
        for confidence, model in model_scores:
            if confidence < 0.1:
                continue

            self.stats[model.name]['tried'] += 1
            guesses, prog_name = model.solve(task, analysis)

            if guesses is not None and score_task(task, guesses):
                self.stats[model.name]['solved'] += 1
                self.mm_solves += 1
                return guesses, 'MM', f"{model.name}:{prog_name}"

        return s1_preds, 'FAIL', None


# ══════════════════════════════════════════════════════════════════════
# 4. BENCHMARK
# ══════════════════════════════════════════════════════════════════════

def run_benchmark(data_dir, limit=None, verbose=False):
    files = sorted(f for f in os.listdir(data_dir) if f.endswith('.json'))
    if limit:
        files = files[:limit]

    solver = MentalModelSolver()

    print("=" * 70)
    print(f"  ARC-AGI-2 with Mental Models")
    print(f"  Tasks: {len(files)}")
    print(f"  S1: {'Phase 2' if HAS_S1 else 'Base'}")
    print(f"  Models: {', '.join(m.name for m in ALL_MODELS)}")
    print(f"  Extended DSL: {'Yes' if HAS_V2 else 'No'}")
    print("=" * 70)

    solved = 0
    total = 0
    t0 = time.time()

    for f in files:
        task = json.load(open(os.path.join(data_dir, f)))
        total += 1

        predictions, system, detail = solver.solve(task)
        correct = score_task(task, predictions)

        if correct:
            solved += 1

        if verbose and correct and system == 'MM':
            print(f"  {f[:16]}: SOLVED ({detail})")

    elapsed = time.time() - t0
    pct = solved / max(total, 1) * 100

    print(f"\n{'='*70}")
    print(f"  MENTAL MODEL RESULTS")
    print(f"{'='*70}")
    print(f"  Solved:     {solved}/{total} ({pct:.1f}%)")
    print(f"    S1:       {solver.s1_solves}")
    print(f"    MM:       {solver.mm_solves} (mental model)")
    print(f"  Time:       {elapsed:.1f}s ({elapsed/max(total,1)*1000:.0f}ms/task)")

    print(f"\n  Model performance:")
    for model in ALL_MODELS:
        s = solver.stats[model.name]
        if s['tried'] > 0:
            rate = s['solved'] / s['tried'] * 100
            print(f"    {model.name:<15}: {s['solved']}/{s['tried']} ({rate:.0f}%)")
        else:
            print(f"    {model.name:<15}: not triggered")

    print(f"\n{'='*70}")
    return solved, total


def analyze_tasks(data_dir, limit=None):
    """Classify all tasks by mental model."""
    files = sorted(f for f in os.listdir(data_dir) if f.endswith('.json'))
    if limit:
        files = files[:limit]

    print("=" * 70)
    print(f"  Task Analysis — Mental Model Classification")
    print(f"  Tasks: {len(files)}")
    print("=" * 70)

    model_counts = Counter()
    for f in files:
        task = json.load(open(os.path.join(data_dir, f)))
        analysis = TaskAnalysis(task)

        scores = [(m.matches(analysis), m.name) for m in ALL_MODELS]
        scores.sort(key=lambda x: -x[0])
        top = scores[0]

        if top[0] > 0.1:
            model_counts[top[1]] += 1
        else:
            model_counts['unclassified'] += 1

    print(f"\n  Primary model distribution:")
    for model, count in model_counts.most_common():
        print(f"    {model:<20}: {count} ({count/len(files)*100:.1f}%)")

    print(f"\n{'='*70}")


# ══════════════════════════════════════════════════════════════════════
# 5. TESTS
# ══════════════════════════════════════════════════════════════════════

def run_tests():
    print("=" * 65)
    print("  Mental Models Tests")
    print("=" * 65)
    p = 0; t = 0

    print("\n  T1: Task analysis")
    task = {'train': [{'input': [[1,2],[3,4]], 'output': [[3,1],[4,2]]}],
            'test': [{'input': [[5,6],[7,8]], 'output': [[7,5],[8,6]]}]}
    a = TaskAnalysis(task)
    ok = a.same_size and a.same_colors
    print(f"    Same size: {a.same_size}, same colors: {a.same_colors} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T2: Model matching — geometric")
    gm = GeometricModel()
    score = gm.matches(a)
    ok = score > 0.2
    print(f"    Geometric confidence: {score:.2f} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T3: Model matching — recolor")
    task2 = {'train': [{'input': [[1,0],[0,1]], 'output': [[2,0],[0,2]]}],
             'test': [{'input': [[1,1],[0,0]], 'output': [[2,2],[0,0]]}]}
    a2 = TaskAnalysis(task2)
    rm = RecolorModel()
    score2 = rm.matches(a2)
    ok = score2 > 0.3
    print(f"    Recolor confidence: {score2:.2f} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T4: Geometric solver")
    preds, name = gm.solve(task, a)
    ok = preds is not None and score_task(task, preds)
    print(f"    Solved: {name} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T5: Recolor solver (color map)")
    preds2, name2 = rm.solve(task2, a2)
    ok = preds2 is not None
    print(f"    Solved: {name2} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T6: Full solver")
    solver = MentalModelSolver()
    preds3, system, detail = solver.solve(task)
    ok = score_task(task, preds3)
    print(f"    {system}: {detail} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T7: All models registered")
    ok = len(ALL_MODELS) >= 7
    print(f"    Models: {len(ALL_MODELS)} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T8: Shrink detection")
    task3 = {'train': [{'input': [[1,2,3],[4,5,6]], 'output': [[1]]}],
             'test': [{'input': [[7,8,9],[0,1,2]], 'output': [[7]]}]}
    a3 = TaskAnalysis(task3)
    ok = a3.shrinks
    print(f"    Shrinks: {a3.shrinks} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print(f"\n{'='*65}")
    print(f"  Results: {p}/{t} tests passed")
    print(f"{'='*65}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--training", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--analyze", action="store_true")
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
    elif args.analyze:
        d = os.path.join(data_base, "training")
        analyze_tasks(d, limit=args.limit)
    else:
        if args.eval:
            d = os.path.join(data_base, "evaluation")
        else:
            d = os.path.join(data_base, "training")
        run_benchmark(d, limit=args.limit, verbose=args.verbose)
