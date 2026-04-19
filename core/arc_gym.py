"""
arc_gym.py — ARC Training Gym for NeMo-WM
=============================================
Procedurally generates grid puzzles of increasing difficulty.
NeMo-WM explores, attempts solutions, learns from errors,
and builds a library of program templates.

Curriculum:
  Level 1: Single-op transforms (rotation, flip, color swap)
  Level 2: Two-step compositions (crop + rotate, recolor + flip)
  Level 3: Conditional rules (neighbor count, position, containment)
  Level 4: Object-level reasoning (extract, sort, align)
  Level 5: Abstract patterns (symmetry completion, pattern extension)
  Level 6: Multi-step synthesis (3-5 chained operations)

The gym tracks what the agent has learned and generates new
puzzles that target its weaknesses.

Usage:
    python arc_gym.py --train --cycles 1000
    python arc_gym.py --eval --data path/to/ARC-AGI-2/data
    python arc_gym.py --test
    python arc_gym.py --status
"""

import argparse
import json
import os
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
import time

from arc_solver import (Grid, GridObject, score_task,
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

Path("data").mkdir(exist_ok=True)
SAVE_PATH = Path("data/arc_gym_state.npz")


# ══════════════════════════════════════════════════════════════════════
# 1. PUZZLE GENERATORS — One per concept
# ══════════════════════════════════════════════════════════════════════

class PuzzleGenerator:
    """Generates training puzzles for a specific concept."""

    def __init__(self, rng=None):
        self.rng = rng or np.random.RandomState(42)

    def random_grid(self, h=None, w=None, n_colors=3, density=0.4):
        """Generate a random grid."""
        if h is None:
            h = self.rng.randint(3, 10)
        if w is None:
            w = self.rng.randint(3, 10)
        colors = list(range(1, n_colors + 1))
        grid = np.zeros((h, w), dtype=np.int32)
        for r in range(h):
            for c in range(w):
                if self.rng.random() < density:
                    grid[r, c] = self.rng.choice(colors)
        return Grid(grid)

    def random_objects_grid(self, h=8, w=8, n_objects=3, max_obj_size=3):
        """Generate grid with distinct objects."""
        grid = np.zeros((h, w), dtype=np.int32)
        for _ in range(n_objects):
            color = self.rng.randint(1, 8)
            oh = self.rng.randint(1, max_obj_size + 1)
            ow = self.rng.randint(1, max_obj_size + 1)
            r = self.rng.randint(0, h - oh + 1)
            c = self.rng.randint(0, w - ow + 1)
            grid[r:r+oh, c:c+ow] = color
        return Grid(grid)


# ── Level 1: Single transforms ──

def gen_rotation(pg: PuzzleGenerator):
    """Generate rotation puzzle."""
    g = pg.random_grid(h=pg.rng.randint(3, 7), w=pg.rng.randint(3, 7))
    op = pg.rng.choice([dsl_rot90, dsl_rot180, dsl_rot270])
    return g, op(g), "rotation"

def gen_flip(pg: PuzzleGenerator):
    g = pg.random_grid()
    op = pg.rng.choice([dsl_flip_h, dsl_flip_v])
    return g, op(g), "flip"

def gen_transpose(pg: PuzzleGenerator):
    g = pg.random_grid(h=pg.rng.randint(3, 6), w=pg.rng.randint(3, 6))
    return g, dsl_transpose(g), "transpose"

def gen_color_swap(pg: PuzzleGenerator):
    """Swap two colors."""
    g = pg.random_grid(n_colors=3, density=0.5)
    c1, c2 = pg.rng.choice(range(1, 4), 2, replace=False)
    arr = g.arr.copy()
    mask1 = arr == c1
    mask2 = arr == c2
    arr[mask1] = c2
    arr[mask2] = c1
    return g, Grid(arr), "color_swap"

def gen_crop(pg: PuzzleGenerator):
    g = pg.random_grid(h=pg.rng.randint(5, 10), w=pg.rng.randint(5, 10), density=0.2)
    return g, dsl_crop_nonzero(g), "crop"

def gen_scale(pg: PuzzleGenerator):
    g = pg.random_grid(h=pg.rng.randint(2, 5), w=pg.rng.randint(2, 5))
    return g, dsl_scale2(g), "scale"


# ── Level 2: Two-step compositions ──

def gen_crop_rotate(pg: PuzzleGenerator):
    g = pg.random_grid(h=pg.rng.randint(5, 10), w=pg.rng.randint(5, 10), density=0.2)
    cropped = dsl_crop_nonzero(g)
    return g, dsl_rot90(cropped), "crop+rot90"

def gen_flip_crop(pg: PuzzleGenerator):
    g = pg.random_grid(h=pg.rng.randint(5, 9), w=pg.rng.randint(5, 9), density=0.25)
    return g, dsl_crop_nonzero(dsl_flip_h(g)), "flip+crop"

def gen_mirror_crop(pg: PuzzleGenerator):
    g = pg.random_grid(h=pg.rng.randint(3, 6), w=pg.rng.randint(3, 6), density=0.4)
    return g, dsl_left_half(dsl_mirror_h(g)), "mirror+half"


# ── Level 3: Conditional rules ──

def gen_neighbor_recolor(pg: PuzzleGenerator):
    """Recolor cells based on neighbor count."""
    g = pg.random_grid(h=pg.rng.randint(4, 8), w=pg.rng.randint(4, 8),
                         n_colors=2, density=0.4)
    target_color = pg.rng.randint(3, 8)
    arr = g.arr.copy()
    result = arr.copy()
    for r in range(g.h):
        for c in range(g.w):
            if arr[r, c] != 0:
                n = sum(1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                        if 0 <= r+dr < g.h and 0 <= c+dc < g.w
                        and arr[r+dr, c+dc] != 0)
                if n >= 3:
                    result[r, c] = target_color
    return g, Grid(result), "neighbor_recolor"

def gen_positional_color(pg: PuzzleGenerator):
    """Color depends on position modulo."""
    h, w = pg.rng.randint(4, 8), pg.rng.randint(4, 8)
    c1 = pg.rng.randint(1, 5)
    c2 = pg.rng.randint(5, 9)
    arr = np.zeros((h, w), dtype=np.int32)
    for r in range(h):
        for c in range(w):
            arr[r, c] = c1 if (r + c) % 2 == 0 else c2
    # Input has some cells zeroed
    inp = arr.copy()
    mask = pg.rng.random((h, w)) > 0.6
    inp[mask] = 0
    return Grid(inp), Grid(arr), "positional_color"

def gen_fill_enclosed(pg: PuzzleGenerator):
    """Fill enclosed regions."""
    h, w = pg.rng.randint(6, 10), pg.rng.randint(6, 10)
    arr = np.zeros((h, w), dtype=np.int32)
    border_color = pg.rng.randint(1, 5)
    # Draw rectangle border
    r0, c0 = pg.rng.randint(0, h//3), pg.rng.randint(0, w//3)
    r1, c1 = pg.rng.randint(h*2//3, h), pg.rng.randint(w*2//3, w)
    arr[r0, c0:c1] = border_color
    arr[r1-1, c0:c1] = border_color
    arr[r0:r1, c0] = border_color
    arr[r0:r1, c1-1] = border_color
    result = arr.copy()
    fill_color = pg.rng.randint(1, 9)
    result[r0+1:r1-1, c0+1:c1-1] = fill_color
    return Grid(arr), Grid(result), "fill_enclosed"


# ── Level 4: Object-level reasoning ──

def gen_extract_largest(pg: PuzzleGenerator):
    g = pg.random_objects_grid(n_objects=pg.rng.randint(2, 5))
    return g, dsl_largest_object(g), "extract_largest"

def gen_extract_smallest(pg: PuzzleGenerator):
    g = pg.random_objects_grid(n_objects=pg.rng.randint(2, 5))
    return g, dsl_smallest_object(g), "extract_smallest"

def gen_sort_by_size(pg: PuzzleGenerator):
    """Recolor objects: smallest=1, medium=2, largest=3."""
    g = pg.random_objects_grid(h=10, w=10, n_objects=3, max_obj_size=4)
    objs = g.objects()
    if len(objs) < 2:
        return gen_extract_largest(pg)
    objs.sort(key=lambda o: o.size)
    result = np.zeros_like(g.arr)
    for i, obj in enumerate(objs):
        for r, c in obj.cells:
            result[r, c] = i + 1
    return g, Grid(result), "sort_by_size"

def gen_extract_by_color(pg: PuzzleGenerator):
    """Extract all objects of a specific color."""
    g = pg.random_objects_grid(n_objects=4)
    objs = g.objects()
    if not objs:
        return gen_extract_largest(pg)
    target = objs[0].color
    result = np.zeros_like(g.arr)
    for obj in objs:
        if obj.color == target:
            for r, c in obj.cells:
                result[r, c] = obj.color
    return g, Grid(result), "extract_by_color"


# ── Level 5: Abstract patterns ──

def gen_symmetry_complete(pg: PuzzleGenerator):
    """Create a partially symmetric grid, output completes it."""
    n = pg.rng.randint(4, 8)
    arr = np.zeros((n, n), dtype=np.int32)
    # Fill one quadrant
    for r in range(n // 2):
        for c in range(n // 2):
            if pg.rng.random() < 0.4:
                arr[r, c] = pg.rng.randint(1, 5)
    # Complete symmetry
    result = arr.copy()
    for r in range(n):
        for c in range(n):
            if result[r, c] == 0 and result[r, n-1-c] != 0:
                result[r, c] = result[r, n-1-c]
            if result[r, c] == 0 and result[n-1-r, c] != 0:
                result[r, c] = result[n-1-r, c]
    return Grid(arr), Grid(result), "symmetry_complete"

def gen_pattern_extend(pg: PuzzleGenerator):
    """Extend a repeating row pattern."""
    w = pg.rng.randint(8, 15)
    h = pg.rng.randint(3, 6)
    arr = np.zeros((h, w), dtype=np.int32)
    result = np.zeros((h, w), dtype=np.int32)
    for r in range(h):
        color = pg.rng.randint(1, 6)
        spacing = pg.rng.randint(2, 4)
        # Input has partial pattern
        n_shown = pg.rng.randint(2, 4)
        for i in range(n_shown):
            c = i * spacing
            if c < w:
                arr[r, c] = color
        # Output has full pattern
        c = 0
        while c < w:
            result[r, c] = color
            c += spacing
    return Grid(arr), Grid(result), "pattern_extend"

def gen_tile_self(pg: PuzzleGenerator):
    """Tile grid where nonzero cells are."""
    g = pg.random_grid(h=pg.rng.randint(3, 5), w=pg.rng.randint(3, 5),
                         density=0.4)
    h, w = g.shape
    result = np.zeros((h*h, w*w), dtype=np.int32)
    for r in range(h):
        for c in range(w):
            if g.arr[r, c] != 0:
                result[r*h:(r+1)*h, c*w:(c+1)*w] = g.arr
    return g, Grid(result), "tile_self"


# ── Level 6: Multi-step synthesis ──

def gen_3step(pg: PuzzleGenerator):
    """Chain 3 random operations."""
    g = pg.random_grid(h=pg.rng.randint(4, 7), w=pg.rng.randint(4, 7), density=0.4)
    ops = [
        ("rot90", dsl_rot90), ("flip_h", dsl_flip_h),
        ("flip_v", dsl_flip_v), ("transpose", dsl_transpose),
    ]
    chosen = [ops[i] for i in pg.rng.choice(len(ops), 3)]
    result = g
    for _, op in chosen:
        result = op(result)
    name = "+".join(n for n, _ in chosen)
    return g, result, f"3step:{name}"


# ══════════════════════════════════════════════════════════════════════
# 2. CURRICULUM — Ordered puzzle generators by difficulty
# ══════════════════════════════════════════════════════════════════════

CURRICULUM = {
    1: [("rotation", gen_rotation), ("flip", gen_flip),
        ("transpose", gen_transpose), ("color_swap", gen_color_swap),
        ("crop", gen_crop), ("scale", gen_scale)],
    2: [("crop+rot", gen_crop_rotate), ("flip+crop", gen_flip_crop),
        ("mirror+half", gen_mirror_crop)],
    3: [("neighbor_recolor", gen_neighbor_recolor),
        ("positional_color", gen_positional_color),
        ("fill_enclosed", gen_fill_enclosed)],
    4: [("extract_largest", gen_extract_largest),
        ("extract_smallest", gen_extract_smallest),
        ("sort_by_size", gen_sort_by_size),
        ("extract_by_color", gen_extract_by_color)],
    5: [("symmetry", gen_symmetry_complete),
        ("pattern_extend", gen_pattern_extend),
        ("tile_self", gen_tile_self)],
    6: [("3step", gen_3step)],
}

ALL_GENERATORS = []
for level, gens in CURRICULUM.items():
    for name, fn in gens:
        ALL_GENERATORS.append((level, name, fn))


# ══════════════════════════════════════════════════════════════════════
# 3. PROGRAM TEMPLATE LIBRARY — Learned from experience
# ══════════════════════════════════════════════════════════════════════

class TemplateLibrary:
    """Library of learned program templates."""

    def __init__(self):
        self.templates = {}  # name → {op, n_solved, n_tried, success_rate}
        self.concept_scores = defaultdict(lambda: {'solved': 0, 'tried': 0})

    def record(self, concept: str, solved: bool, program_name: str = None):
        """Record a solve attempt."""
        self.concept_scores[concept]['tried'] += 1
        if solved:
            self.concept_scores[concept]['solved'] += 1
        if program_name and solved:
            if program_name not in self.templates:
                self.templates[program_name] = {'n_solved': 0, 'n_tried': 0}
            self.templates[program_name]['n_solved'] += 1
            self.templates[program_name]['n_tried'] += 1

    def mastery(self, concept: str) -> float:
        """Return mastery level for a concept (0-1)."""
        s = self.concept_scores[concept]
        if s['tried'] == 0:
            return 0.0
        return s['solved'] / s['tried']

    def weakest_concepts(self, n=5) -> list:
        """Return concepts with lowest mastery."""
        all_concepts = list(self.concept_scores.keys())
        return sorted(all_concepts, key=lambda c: self.mastery(c))[:n]

    def summary(self) -> dict:
        """Return summary of learning progress."""
        return {
            'n_templates': len(self.templates),
            'n_concepts': len(self.concept_scores),
            'concept_mastery': {c: f"{self.mastery(c):.0%}"
                                  for c in sorted(self.concept_scores.keys())},
            'total_solved': sum(s['solved'] for s in self.concept_scores.values()),
            'total_tried': sum(s['tried'] for s in self.concept_scores.values()),
        }


# ══════════════════════════════════════════════════════════════════════
# 4. SOLVER — Attempts to solve gym puzzles
# ══════════════════════════════════════════════════════════════════════

from arc_solver import SINGLE_OPS

def attempt_solve(input_grid: Grid, output_grid: Grid):
    """Try to solve a puzzle. Returns (solved, program_name)."""

    # Try single ops
    for name, op in SINGLE_OPS:
        try:
            if op(input_grid) == output_grid:
                return True, name
        except Exception:
            continue

    # Try color mapping
    from arc_solver import infer_color_map, apply_color_map
    mapping = {}
    if input_grid.shape == output_grid.shape:
        consistent = True
        for r in range(input_grid.h):
            for c in range(input_grid.w):
                ic = int(input_grid.arr[r, c])
                oc = int(output_grid.arr[r, c])
                if ic in mapping and mapping[ic] != oc:
                    consistent = False
                    break
                mapping[ic] = oc
            if not consistent:
                break
        if consistent and mapping:
            arr = input_grid.arr.copy()
            for old, new in mapping.items():
                arr[input_grid.arr == old] = new
            if Grid(arr) == output_grid:
                return True, "color_map"

    # Try two-step compositions
    fast_ops = [
        ("rot90", dsl_rot90), ("rot180", dsl_rot180),
        ("flip_h", dsl_flip_h), ("flip_v", dsl_flip_v),
        ("transpose", dsl_transpose), ("crop", dsl_crop_nonzero),
        ("outline", dsl_outline), ("fill", dsl_fill_holes),
        ("gravity", dsl_gravity_down), ("largest", dsl_largest_object),
        ("half_t", dsl_top_half), ("half_b", dsl_bottom_half),
        ("half_l", dsl_left_half), ("half_r", dsl_right_half),
    ]
    for n1, op1 in fast_ops:
        for n2, op2 in fast_ops:
            try:
                if op2(op1(input_grid)) == output_grid:
                    return True, f"{n1}+{n2}"
            except Exception:
                continue

    # ── Dedicated solvers for failing concepts ──

    # Pattern extend: detect spacing in rows, extend
    if input_grid.shape == output_grid.shape:
        try:
            arr = input_grid.arr.copy()
            bg = int(np.argmax(np.bincount(arr.flatten())))
            result = arr.copy()
            for r in range(input_grid.h):
                nz = [(c, int(arr[r, c])) for c in range(input_grid.w) if arr[r, c] != bg]
                if len(nz) >= 2:
                    spacing = nz[1][0] - nz[0][0]
                    color = nz[0][1]
                    if spacing > 0 and all(nz[i][0] - nz[i-1][0] == spacing and nz[i][1] == color for i in range(1, len(nz))):
                        c = nz[-1][0] + spacing
                        while c < input_grid.w:
                            result[r, c] = color
                            c += spacing
                        c = nz[0][0] - spacing
                        while c >= 0:
                            result[r, c] = color
                            c -= spacing
            if Grid(result) == output_grid:
                return True, "pattern_extend"
        except Exception:
            pass

    # Positional color: output depends on (r%mod, c%mod)
    if input_grid.shape == output_grid.shape:
        for mod in [2, 3, 4]:
            try:
                rules = {}
                consistent = True
                for r in range(output_grid.h):
                    for c in range(output_grid.w):
                        key = (r % mod, c % mod)
                        val = int(output_grid.arr[r, c])
                        if key in rules and rules[key] != val:
                            consistent = False
                            break
                        rules[key] = val
                    if not consistent:
                        break
                if consistent and rules:
                    result = np.zeros_like(input_grid.arr)
                    for r in range(input_grid.h):
                        for c in range(input_grid.w):
                            result[r, c] = rules.get((r % mod, c % mod), 0)
                    if Grid(result) == output_grid:
                        return True, f"positional_mod{mod}"
            except Exception:
                continue

        # Also try: output color depends on (input_color, r%mod, c%mod)
        for mod in [2, 3]:
            try:
                rules = {}
                consistent = True
                for r in range(input_grid.h):
                    for c in range(input_grid.w):
                        key = (int(input_grid.arr[r, c]), r % mod, c % mod)
                        val = int(output_grid.arr[r, c])
                        if key in rules and rules[key] != val:
                            consistent = False
                            break
                        rules[key] = val
                    if not consistent:
                        break
                if consistent and rules:
                    all_id = all(k[0] == v for k, v in rules.items())
                    if not all_id:
                        result = np.zeros_like(input_grid.arr)
                        for r in range(input_grid.h):
                            for c in range(input_grid.w):
                                key = (int(input_grid.arr[r, c]), r % mod, c % mod)
                                result[r, c] = rules.get(key, input_grid.arr[r, c])
                        if Grid(result) == output_grid:
                            return True, f"input_positional_mod{mod}"
            except Exception:
                continue

    # Tile self: nonzero cells of input get replaced by full input
    try:
        ih, iw = input_grid.shape
        oh, ow = output_grid.shape
        if oh == ih * ih and ow == iw * iw:
            result = np.zeros((oh, ow), dtype=np.int32)
            for r in range(ih):
                for c in range(iw):
                    if input_grid.arr[r, c] != 0:
                        result[r*ih:(r+1)*ih, c*iw:(c+1)*iw] = input_grid.arr
            if Grid(result) == output_grid:
                return True, "tile_self"
    except Exception:
        pass

    # Symmetry complete: fill zeros to make grid symmetric
    if input_grid.shape == output_grid.shape:
        for sym in ['h', 'v', 'both', 'diag']:
            try:
                arr = input_grid.arr.copy()
                h, w = arr.shape
                if sym == 'h':
                    for r in range(h):
                        for c in range(w):
                            mc = w - 1 - c
                            if arr[r, c] == 0 and arr[r, mc] != 0:
                                arr[r, c] = arr[r, mc]
                            elif arr[r, mc] == 0 and arr[r, c] != 0:
                                arr[r, mc] = arr[r, c]
                elif sym == 'v':
                    for r in range(h):
                        for c in range(w):
                            mr = h - 1 - r
                            if arr[r, c] == 0 and arr[mr, c] != 0:
                                arr[r, c] = arr[mr, c]
                            elif arr[mr, c] == 0 and arr[r, c] != 0:
                                arr[mr, c] = arr[r, c]
                elif sym == 'both':
                    for r in range(h):
                        for c in range(w):
                            mr, mc = h-1-r, w-1-c
                            vals = [arr[r,c], arr[r,mc], arr[mr,c], arr[mr,mc]]
                            nz = [v for v in vals if v != 0]
                            if nz:
                                fill = Counter(nz).most_common(1)[0][0]
                                if arr[r,c] == 0: arr[r,c] = fill
                                if arr[r,mc] == 0: arr[r,mc] = fill
                                if arr[mr,c] == 0: arr[mr,c] = fill
                                if arr[mr,mc] == 0: arr[mr,mc] = fill
                elif sym == 'diag' and h == w:
                    for r in range(h):
                        for c in range(w):
                            if arr[r, c] == 0 and arr[c, r] != 0:
                                arr[r, c] = arr[c, r]
                            elif arr[c, r] == 0 and arr[r, c] != 0:
                                arr[c, r] = arr[r, c]
                if Grid(arr) == output_grid:
                    return True, f"symmetry_{sym}"
            except Exception:
                continue

    # Fill enclosed: flood fill interior bg regions with enclosing color
    if input_grid.shape == output_grid.shape:
        try:
            arr = input_grid.arr.copy()
            h, w = arr.shape
            bg = int(np.argmax(np.bincount(arr.flatten())))
            exterior = np.zeros((h, w), dtype=bool)
            stack = []
            for r in range(h):
                for c in [0, w-1]:
                    if arr[r, c] == bg:
                        stack.append((r, c))
            for c in range(w):
                for r in [0, h-1]:
                    if arr[r, c] == bg:
                        stack.append((r, c))
            while stack:
                r, c = stack.pop()
                if r < 0 or r >= h or c < 0 or c >= w:
                    continue
                if exterior[r, c] or arr[r, c] != bg:
                    continue
                exterior[r, c] = True
                stack.extend([(r-1,c),(r+1,c),(r,c-1),(r,c+1)])
            # Fill interior with enclosing color
            visited = np.zeros((h, w), dtype=bool)
            for r in range(h):
                for c in range(w):
                    if arr[r, c] == bg and not exterior[r, c] and not visited[r, c]:
                        region = []
                        border_colors = Counter()
                        rs = [(r, c)]
                        while rs:
                            cr, cc = rs.pop()
                            if cr < 0 or cr >= h or cc < 0 or cc >= w:
                                continue
                            if visited[cr, cc]:
                                continue
                            if arr[cr, cc] != bg:
                                border_colors[int(arr[cr, cc])] += 1
                                continue
                            if exterior[cr, cc]:
                                continue
                            visited[cr, cc] = True
                            region.append((cr, cc))
                            rs.extend([(cr-1,cc),(cr+1,cc),(cr,cc-1),(cr,cc+1)])
                        if region and border_colors:
                            fc = border_colors.most_common(1)[0][0]
                            for cr, cc in region:
                                arr[cr, cc] = fc
            if Grid(arr) == output_grid:
                return True, "fill_enclosed"
        except Exception:
            pass

    # Neighbor recolor: color depends on (input_color, n_nonzero_neighbors)
    if input_grid.shape == output_grid.shape:
        try:
            rules = {}
            consistent = True
            h, w = input_grid.h, input_grid.w
            for r in range(h):
                for c in range(w):
                    ic = int(input_grid.arr[r, c])
                    oc = int(output_grid.arr[r, c])
                    n_count = sum(1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                                    if 0 <= r+dr < h and 0 <= c+dc < w
                                    and input_grid.arr[r+dr, c+dc] != 0)
                    key = (ic, n_count)
                    if key in rules and rules[key] != oc:
                        consistent = False
                        break
                    rules[key] = oc
                if not consistent:
                    break
            if consistent and rules and not all(k[0] == v for k, v in rules.items()):
                result = input_grid.arr.copy()
                for r in range(h):
                    for c in range(w):
                        ic = int(input_grid.arr[r, c])
                        n_count = sum(1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                                        if 0 <= r+dr < h and 0 <= c+dc < w
                                        and input_grid.arr[r+dr, c+dc] != 0)
                        key = (ic, n_count)
                        if key in rules:
                            result[r, c] = rules[key]
                if Grid(result) == output_grid:
                    return True, "neighbor_rule"
        except Exception:
            pass

    return False, None


# ══════════════════════════════════════════════════════════════════════
# 5. TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def train(n_cycles=1000, verbose=False):
    """Run the training gym."""
    pg = PuzzleGenerator(rng=np.random.RandomState(int(time.time()) % 2**31))
    library = TemplateLibrary()

    print("=" * 70)
    print("  ARC Training Gym — NeMo-WM Grid Reasoning")
    print(f"  Cycles: {n_cycles}")
    print(f"  Levels: {len(CURRICULUM)} ({sum(len(v) for v in CURRICULUM.values())} concepts)")
    print("=" * 70)

    t0 = time.time()
    level_stats = defaultdict(lambda: {'solved': 0, 'tried': 0})

    for cycle in range(n_cycles):
        # Adaptive curriculum: focus on weakest concepts
        if cycle < n_cycles * 0.3:
            # Early: uniform across levels 1-3
            level = pg.rng.randint(1, 4)
        elif cycle < n_cycles * 0.6:
            # Mid: levels 1-5
            level = pg.rng.randint(1, 6)
        else:
            # Late: all levels, biased toward weakness
            weak = library.weakest_concepts(3)
            if weak and pg.rng.random() < 0.5:
                # Find which level the weak concept belongs to
                for lv, gens in CURRICULUM.items():
                    for gname, _ in gens:
                        if gname in weak[0]:
                            level = lv
                            break
                    else:
                        continue
                    break
                else:
                    level = pg.rng.randint(1, 7)
            else:
                level = pg.rng.randint(1, 7)

        # Pick a generator from this level
        generators = CURRICULUM.get(level, CURRICULUM[1])
        gen_name, gen_fn = generators[pg.rng.randint(len(generators))]

        # Generate puzzle
        try:
            input_grid, output_grid, concept = gen_fn(pg)
        except Exception:
            continue

        # Attempt solve
        solved, program = attempt_solve(input_grid, output_grid)

        # Record
        library.record(concept, solved, program)
        level_stats[level]['tried'] += 1
        if solved:
            level_stats[level]['solved'] += 1

        # Progress
        if (cycle + 1) % max(1, n_cycles // 10) == 0 or verbose:
            elapsed = time.time() - t0
            total_s = sum(s['solved'] for s in level_stats.values())
            total_t = sum(s['tried'] for s in level_stats.values())
            rate = total_s / max(total_t, 1) * 100
            if not verbose:
                print(f"  Cycle {cycle+1:>6} │ Solved: {total_s}/{total_t} ({rate:.1f}%) │ "
                      f"Templates: {len(library.templates)} │ "
                      f"Time: {elapsed:.1f}s")

    # Summary
    elapsed = time.time() - t0
    total_s = sum(s['solved'] for s in level_stats.values())
    total_t = sum(s['tried'] for s in level_stats.values())

    print(f"\n{'='*70}")
    print(f"  TRAINING SUMMARY")
    print(f"{'='*70}")
    print(f"  Cycles:     {n_cycles}")
    print(f"  Solved:     {total_s}/{total_t} ({total_s/max(total_t,1)*100:.1f}%)")
    print(f"  Templates:  {len(library.templates)}")
    print(f"  Time:       {elapsed:.1f}s")

    print(f"\n  Per-level results:")
    for lv in sorted(level_stats.keys()):
        s = level_stats[lv]
        rate = s['solved'] / max(s['tried'], 1) * 100
        print(f"    Level {lv}: {s['solved']}/{s['tried']} ({rate:.1f}%)")

    print(f"\n  Concept mastery:")
    for concept in sorted(library.concept_scores.keys()):
        m = library.mastery(concept)
        bar = "█" * int(m * 20) + "░" * (20 - int(m * 20))
        cs = library.concept_scores[concept]
        print(f"    {concept:<25} {bar} {m:.0%} ({cs['solved']}/{cs['tried']})")

    weak = library.weakest_concepts(5)
    if weak:
        print(f"\n  Weakest concepts: {', '.join(weak)}")

    print(f"\n  Top templates:")
    top = sorted(library.templates.items(), key=lambda x: -x[1]['n_solved'])[:10]
    for name, info in top:
        print(f"    {name:<30}: {info['n_solved']} solves")

    print(f"\n{'='*70}")

    # Save state
    np.savez(SAVE_PATH,
             templates=json.dumps(library.templates),
             concepts=json.dumps(dict(library.concept_scores)),
             total_solved=total_s, total_tried=total_t)
    print(f"  State saved to {SAVE_PATH}")

    return library


def show_status():
    """Show current gym training status."""
    if not SAVE_PATH.exists():
        print("  No training data found. Run: python arc_gym.py --train --cycles 1000")
        return

    data = np.load(SAVE_PATH, allow_pickle=True)
    templates = json.loads(str(data['templates']))
    concepts = json.loads(str(data['concepts']))

    print("=" * 70)
    print("  ARC Training Gym Status")
    print("=" * 70)
    print(f"  Total solved: {int(data['total_solved'])}/{int(data['total_tried'])}")
    print(f"  Templates learned: {len(templates)}")
    print(f"  Concepts practiced: {len(concepts)}")

    print(f"\n  Concept mastery:")
    for concept, scores in sorted(concepts.items()):
        m = scores['solved'] / max(scores['tried'], 1)
        bar = "█" * int(m * 20) + "░" * (20 - int(m * 20))
        print(f"    {concept:<25} {bar} {m:.0%}")

    print(f"\n{'='*70}")


# ══════════════════════════════════════════════════════════════════════
# 6. TESTS
# ══════════════════════════════════════════════════════════════════════

def run_tests():
    print("=" * 65)
    print("  ARC Training Gym Tests")
    print("=" * 65)
    p = 0; t = 0
    pg = PuzzleGenerator(rng=np.random.RandomState(42))

    print("\n  T1: Generate rotation puzzle")
    inp, out, concept = gen_rotation(pg)
    ok = inp.shape != (0, 0) and out.shape != (0, 0)
    print(f"    {inp.shape} -> {out.shape} ({concept}) {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T2: Solve rotation puzzle")
    solved, prog = attempt_solve(inp, out)
    ok = solved
    print(f"    Solved: {solved}, program: {prog} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T3: Generate neighbor recolor")
    inp, out, concept = gen_neighbor_recolor(pg)
    ok = concept == "neighbor_recolor"
    print(f"    {inp.shape} -> {out.shape} ({concept}) {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T4: Generate symmetry puzzle")
    inp, out, concept = gen_symmetry_complete(pg)
    ok = concept == "symmetry_complete"
    print(f"    {inp.shape} -> {out.shape} ({concept}) {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T5: Template library")
    lib = TemplateLibrary()
    lib.record("rotation", True, "rot90")
    lib.record("rotation", True, "rot90")
    lib.record("rotation", False, None)
    ok = lib.mastery("rotation") == 2/3
    print(f"    Mastery: {lib.mastery('rotation'):.1%} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T6: All generators produce valid puzzles")
    valid = 0
    for level, name, gen_fn in ALL_GENERATORS:
        try:
            inp, out, concept = gen_fn(pg)
            if inp.shape[0] > 0 and out.shape[0] > 0:
                valid += 1
        except Exception:
            pass
    total_gens = len(ALL_GENERATORS)
    ok = valid >= total_gens - 1
    print(f"    Valid: {valid}/{total_gens} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T7: Mini training run")
    lib = train(n_cycles=50, verbose=False)
    ok = lib.summary()['total_tried'] >= 40
    print(f"    Tried: {lib.summary()['total_tried']} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T8: Curriculum covers all levels")
    levels = set(CURRICULUM.keys())
    ok = levels == {1, 2, 3, 4, 5, 6}
    print(f"    Levels: {sorted(levels)} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print(f"\n{'='*65}")
    print(f"  Results: {p}/{t} tests passed")
    print(f"{'='*65}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--cycles", type=int, default=1000)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if args.test:
        run_tests()
    elif args.status:
        show_status()
    elif args.train:
        train(n_cycles=args.cycles, verbose=args.verbose)
    else:
        run_tests()
