"""
arc_solver.py — ARC-AGI-2 Solver with Grid Parser + DSL + Program Search
==========================================================================
Builds on arc_agi2_harness.py with deeper capabilities:

  1. Grid Parser: extract objects, borders, backgrounds, symmetry
  2. DSL: composable grid primitives (40+ operations)
  3. Program Search: find operation chains that map input → output
  4. Abstract Pattern Matching: size/count/color relationships

Usage:
    python arc_solver.py --test
    python arc_solver.py --training --limit 100 -v
    python arc_solver.py --eval -v
    python arc_solver.py --task 007bbfb7
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set, Callable
from collections import Counter, defaultdict
from itertools import product
import time
import copy

# ══════════════════════════════════════════════════════════════════════
# 1. GRID PARSER — Extract structure from 2D grids
# ══════════════════════════════════════════════════════════════════════

class Grid:
    """Parsed representation of a 2D color grid."""

    def __init__(self, data):
        if isinstance(data, np.ndarray):
            self.arr = data.astype(np.int32)
        else:
            self.arr = np.array(data, dtype=np.int32)
        self.h, self.w = self.arr.shape

    @property
    def shape(self):
        return (self.h, self.w)

    def copy(self):
        return Grid(self.arr.copy())

    def to_list(self):
        return self.arr.tolist()

    def __eq__(self, other):
        if not isinstance(other, Grid):
            return False
        return self.shape == other.shape and np.array_equal(self.arr, other.arr)

    def __hash__(self):
        return hash(self.arr.tobytes())

    @property
    def colors(self):
        return set(np.unique(self.arr))

    @property
    def nonzero_colors(self):
        return self.colors - {0}

    @property
    def background(self):
        """Most common color (usually 0)."""
        vals, counts = np.unique(self.arr, return_counts=True)
        return int(vals[np.argmax(counts)])

    @property
    def color_counts(self):
        vals, counts = np.unique(self.arr, return_counts=True)
        return dict(zip(vals.tolist(), counts.tolist()))

    def objects(self, bg=None):
        """Extract connected components (4-connected)."""
        if bg is None:
            bg = self.background
        visited = np.zeros_like(self.arr, dtype=bool)
        objs = []

        for r in range(self.h):
            for c in range(self.w):
                if visited[r, c] or self.arr[r, c] == bg:
                    continue
                # BFS
                color = self.arr[r, c]
                cells = []
                stack = [(r, c)]
                while stack:
                    cr, cc = stack.pop()
                    if cr < 0 or cr >= self.h or cc < 0 or cc >= self.w:
                        continue
                    if visited[cr, cc] or self.arr[cr, cc] != color:
                        continue
                    visited[cr, cc] = True
                    cells.append((cr, cc))
                    stack.extend([(cr-1,cc),(cr+1,cc),(cr,cc-1),(cr,cc+1)])
                if cells:
                    objs.append(GridObject(cells, color, self))
        return objs

    def subgrid(self, r0, c0, r1, c1):
        """Extract rectangular subgrid [r0:r1, c0:c1]."""
        return Grid(self.arr[r0:r1, c0:c1].copy())

    def is_symmetric_h(self):
        return np.array_equal(self.arr, self.arr[:, ::-1])

    def is_symmetric_v(self):
        return np.array_equal(self.arr, self.arr[::-1, :])

    def is_symmetric_diag(self):
        if self.h != self.w:
            return False
        return np.array_equal(self.arr, self.arr.T)

    def unique_rows(self):
        return len(set(tuple(row) for row in self.arr.tolist()))

    def unique_cols(self):
        return len(set(tuple(col) for col in self.arr.T.tolist()))


class GridObject:
    """A connected component extracted from a grid."""

    def __init__(self, cells, color, parent_grid):
        self.cells = cells
        self.color = color
        rs = [r for r, c in cells]
        cs = [c for r, c in cells]
        self.r0, self.r1 = min(rs), max(rs) + 1
        self.c0, self.c1 = min(cs), max(cs) + 1
        self.bbox_h = self.r1 - self.r0
        self.bbox_w = self.c1 - self.c0
        self.size = len(cells)
        self.parent = parent_grid

    @property
    def bbox(self):
        return (self.r0, self.c0, self.r1, self.c1)

    @property
    def center(self):
        return ((self.r0 + self.r1) / 2, (self.c0 + self.c1) / 2)

    def mask(self):
        """Binary mask of object within its bounding box."""
        m = np.zeros((self.bbox_h, self.bbox_w), dtype=np.int32)
        for r, c in self.cells:
            m[r - self.r0, c - self.c0] = 1
        return m

    def as_grid(self, bg=0):
        """Extract object as a small grid."""
        g = np.full((self.bbox_h, self.bbox_w), bg, dtype=np.int32)
        for r, c in self.cells:
            g[r - self.r0, c - self.c0] = self.color
        return Grid(g)

    def is_rectangle(self):
        return self.size == self.bbox_h * self.bbox_w

    def is_line(self):
        return self.bbox_h == 1 or self.bbox_w == 1


# ══════════════════════════════════════════════════════════════════════
# 2. DSL — Composable Grid Operations
# ══════════════════════════════════════════════════════════════════════

def dsl_identity(g: Grid) -> Grid:
    return g.copy()

def dsl_rot90(g: Grid) -> Grid:
    return Grid(np.rot90(g.arr, k=-1))

def dsl_rot180(g: Grid) -> Grid:
    return Grid(np.rot90(g.arr, k=2))

def dsl_rot270(g: Grid) -> Grid:
    return Grid(np.rot90(g.arr, k=1))

def dsl_flip_h(g: Grid) -> Grid:
    return Grid(g.arr[:, ::-1])

def dsl_flip_v(g: Grid) -> Grid:
    return Grid(g.arr[::-1, :])

def dsl_transpose(g: Grid) -> Grid:
    return Grid(g.arr.T)

def dsl_crop_nonzero(g: Grid) -> Grid:
    nz = np.argwhere(g.arr != 0)
    if len(nz) == 0:
        return Grid(np.array([[0]]))
    r0, c0 = nz.min(axis=0)
    r1, c1 = nz.max(axis=0)
    return Grid(g.arr[r0:r1+1, c0:c1+1].copy())

def dsl_crop_to_bg(g: Grid) -> Grid:
    """Crop to bounding box of non-background cells."""
    bg = g.background
    nz = np.argwhere(g.arr != bg)
    if len(nz) == 0:
        return Grid(np.array([[bg]]))
    r0, c0 = nz.min(axis=0)
    r1, c1 = nz.max(axis=0)
    return Grid(g.arr[r0:r1+1, c0:c1+1].copy())

def dsl_scale2(g: Grid) -> Grid:
    return Grid(np.repeat(np.repeat(g.arr, 2, axis=0), 2, axis=1))

def dsl_scale3(g: Grid) -> Grid:
    return Grid(np.repeat(np.repeat(g.arr, 3, axis=0), 3, axis=1))

def dsl_tile2x2(g: Grid) -> Grid:
    return Grid(np.tile(g.arr, (2, 2)))

def dsl_tile3x3(g: Grid) -> Grid:
    return Grid(np.tile(g.arr, (3, 3)))

def dsl_tile2x1(g: Grid) -> Grid:
    return Grid(np.tile(g.arr, (2, 1)))

def dsl_tile1x2(g: Grid) -> Grid:
    return Grid(np.tile(g.arr, (1, 2)))

def dsl_gravity_down(g: Grid) -> Grid:
    """Drop nonzero cells to bottom."""
    arr = g.arr.copy()
    for c in range(arr.shape[1]):
        col = arr[:, c]
        nonzero = col[col != 0]
        arr[:, c] = 0
        arr[arr.shape[0]-len(nonzero):, c] = nonzero
    return Grid(arr)

def dsl_gravity_left(g: Grid) -> Grid:
    arr = g.arr.copy()
    for r in range(arr.shape[0]):
        row = arr[r, :]
        nonzero = row[row != 0]
        arr[r, :] = 0
        arr[r, :len(nonzero)] = nonzero
    return Grid(arr)

def dsl_sort_rows(g: Grid) -> Grid:
    return Grid(np.sort(g.arr, axis=1))

def dsl_sort_cols(g: Grid) -> Grid:
    return Grid(np.sort(g.arr, axis=0))

def dsl_replace_bg_with(g: Grid, color: int) -> Grid:
    arr = g.arr.copy()
    bg = g.background
    arr[arr == bg] = color
    return Grid(arr)

def dsl_keep_only_color(g: Grid, color: int) -> Grid:
    arr = np.where(g.arr == color, color, 0).astype(np.int32)
    return Grid(arr)

def dsl_remove_color(g: Grid, color: int) -> Grid:
    arr = g.arr.copy()
    arr[arr == color] = 0
    return Grid(arr)

def dsl_recolor(g: Grid, old: int, new: int) -> Grid:
    arr = g.arr.copy()
    arr[arr == old] = new
    return Grid(arr)

def dsl_outline(g: Grid) -> Grid:
    """Keep only border cells of non-zero regions."""
    arr = g.arr.copy()
    result = np.zeros_like(arr)
    h, w = arr.shape
    for r in range(h):
        for c in range(w):
            if arr[r, c] != 0:
                is_border = False
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if nr < 0 or nr >= h or nc < 0 or nc >= w or arr[nr, nc] == 0:
                        is_border = True
                        break
                if is_border:
                    result[r, c] = arr[r, c]
    return Grid(result)

def dsl_fill_holes(g: Grid) -> Grid:
    """Fill interior 0s surrounded by non-zero cells."""
    arr = g.arr.copy()
    h, w = arr.shape
    # Flood fill from edges to find exterior 0s
    exterior = np.zeros((h, w), dtype=bool)
    stack = []
    for r in range(h):
        for c in [0, w-1]:
            if arr[r, c] == 0:
                stack.append((r, c))
    for c in range(w):
        for r in [0, h-1]:
            if arr[r, c] == 0:
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        if r < 0 or r >= h or c < 0 or c >= w:
            continue
        if exterior[r, c] or arr[r, c] != 0:
            continue
        exterior[r, c] = True
        stack.extend([(r-1,c),(r+1,c),(r,c-1),(r,c+1)])
    # Fill interior 0s with most common neighbor color
    for r in range(h):
        for c in range(w):
            if arr[r, c] == 0 and not exterior[r, c]:
                neighbors = []
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < h and 0 <= nc < w and arr[nr, nc] != 0:
                        neighbors.append(arr[nr, nc])
                if neighbors:
                    arr[r, c] = Counter(neighbors).most_common(1)[0][0]
    return Grid(arr)

def dsl_mirror_h(g: Grid) -> Grid:
    """Concatenate grid with its horizontal mirror."""
    return Grid(np.concatenate([g.arr, g.arr[:, ::-1]], axis=1))

def dsl_mirror_v(g: Grid) -> Grid:
    return Grid(np.concatenate([g.arr, g.arr[::-1, :]], axis=0))

def dsl_largest_object(g: Grid) -> Grid:
    """Extract the largest connected component."""
    objs = g.objects()
    if not objs:
        return g.copy()
    largest = max(objs, key=lambda o: o.size)
    return largest.as_grid(bg=0)

def dsl_smallest_object(g: Grid) -> Grid:
    objs = g.objects()
    if not objs:
        return g.copy()
    smallest = min(objs, key=lambda o: o.size)
    return smallest.as_grid(bg=0)

def dsl_count_colors_grid(g: Grid) -> Grid:
    """Output is 1x1 grid with count of unique non-zero colors."""
    return Grid(np.array([[len(g.nonzero_colors)]]))

def dsl_count_objects_grid(g: Grid) -> Grid:
    return Grid(np.array([[len(g.objects())]]))

def dsl_top_half(g: Grid) -> Grid:
    return Grid(g.arr[:g.h//2, :].copy())

def dsl_bottom_half(g: Grid) -> Grid:
    return Grid(g.arr[g.h//2:, :].copy())

def dsl_left_half(g: Grid) -> Grid:
    return Grid(g.arr[:, :g.w//2].copy())

def dsl_right_half(g: Grid) -> Grid:
    return Grid(g.arr[:, g.w//2:].copy())

def dsl_unique_rows(g: Grid) -> Grid:
    """Keep only unique rows."""
    seen = set()
    rows = []
    for row in g.arr:
        key = tuple(row)
        if key not in seen:
            seen.add(key)
            rows.append(row)
    if not rows:
        return g.copy()
    return Grid(np.array(rows))

def dsl_unique_cols(g: Grid) -> Grid:
    t = dsl_transpose(g)
    u = dsl_unique_rows(t)
    return dsl_transpose(u)


# ══════════════════════════════════════════════════════════════════════
# 3. COLOR MAPPING INFERENCE
# ══════════════════════════════════════════════════════════════════════

def infer_color_map(pairs):
    """Infer per-cell color mapping from examples."""
    if not all(Grid(p['input']).shape == Grid(p['output']).shape for p in pairs):
        return None

    mapping = {}
    for p in pairs:
        gi = Grid(p['input'])
        go = Grid(p['output'])
        for r in range(gi.h):
            for c in range(gi.w):
                ic = int(gi.arr[r, c])
                oc = int(go.arr[r, c])
                if ic in mapping and mapping[ic] != oc:
                    return None
                mapping[ic] = oc
    return mapping

def apply_color_map(g: Grid, mapping: dict) -> Grid:
    arr = g.arr.copy()
    for old, new in mapping.items():
        arr[g.arr == old] = new
    return Grid(arr)


# ══════════════════════════════════════════════════════════════════════
# 4. SIZE/SHAPE RELATIONSHIP INFERENCE
# ══════════════════════════════════════════════════════════════════════

def infer_size_relationship(pairs):
    """Infer how output size relates to input size or content."""
    relationships = []
    for p in pairs:
        gi = Grid(p['input'])
        go = Grid(p['output'])
        n_obj = len(gi.objects())
        n_colors = len(gi.nonzero_colors)

        relationships.append({
            'in_h': gi.h, 'in_w': gi.w,
            'out_h': go.h, 'out_w': go.w,
            'h_ratio': go.h / gi.h if gi.h else 0,
            'w_ratio': go.w / gi.w if gi.w else 0,
            'n_obj': n_obj,
            'n_colors': n_colors,
        })
    return relationships


# ══════════════════════════════════════════════════════════════════════
# 5. PROGRAM SEARCH — Try single and paired operations
# ══════════════════════════════════════════════════════════════════════

# Single operations (no parameters)
SINGLE_OPS = [
    ("identity", dsl_identity),
    ("rot90", dsl_rot90),
    ("rot180", dsl_rot180),
    ("rot270", dsl_rot270),
    ("flip_h", dsl_flip_h),
    ("flip_v", dsl_flip_v),
    ("transpose", dsl_transpose),
    ("crop_nz", dsl_crop_nonzero),
    ("crop_bg", dsl_crop_to_bg),
    ("scale2", dsl_scale2),
    ("scale3", dsl_scale3),
    ("tile2x2", dsl_tile2x2),
    ("tile3x3", dsl_tile3x3),
    ("tile2x1", dsl_tile2x1),
    ("tile1x2", dsl_tile1x2),
    ("grav_down", dsl_gravity_down),
    ("grav_left", dsl_gravity_left),
    ("sort_rows", dsl_sort_rows),
    ("sort_cols", dsl_sort_cols),
    ("outline", dsl_outline),
    ("fill_holes", dsl_fill_holes),
    ("mirror_h", dsl_mirror_h),
    ("mirror_v", dsl_mirror_v),
    ("largest_obj", dsl_largest_object),
    ("smallest_obj", dsl_smallest_object),
    ("top_half", dsl_top_half),
    ("bottom_half", dsl_bottom_half),
    ("left_half", dsl_left_half),
    ("right_half", dsl_right_half),
    ("unique_rows", dsl_unique_rows),
    ("unique_cols", dsl_unique_cols),
    ("count_colors", dsl_count_colors_grid),
    ("count_objects", dsl_count_objects_grid),
]


def search_single(pairs):
    """Try every single DSL operation."""
    for name, op in SINGLE_OPS:
        try:
            matches = True
            for p in pairs:
                gi = Grid(p['input'])
                go = Grid(p['output'])
                result = op(gi)
                if result != go:
                    matches = False
                    break
            if matches:
                return name, op
        except Exception:
            continue
    return None, None


def search_color_map(pairs):
    """Try color mapping."""
    mapping = infer_color_map(pairs)
    if mapping is None:
        return None, None
    # Verify
    for p in pairs:
        gi = Grid(p['input'])
        go = Grid(p['output'])
        result = apply_color_map(gi, mapping)
        if result != go:
            return None, None
    return "color_map", lambda g: apply_color_map(g, mapping)


def search_parameterized(pairs):
    """Try parameterized operations."""
    # Try recolor for each color pair
    all_in_colors = set()
    all_out_colors = set()
    for p in pairs:
        gi = Grid(p['input'])
        go = Grid(p['output'])
        all_in_colors |= gi.colors
        all_out_colors |= go.colors

    # Try keep_only_color
    for color in range(10):
        try:
            matches = True
            for p in pairs:
                gi = Grid(p['input'])
                go = Grid(p['output'])
                if dsl_keep_only_color(gi, color) != go:
                    matches = False
                    break
            if matches:
                return f"keep_color_{color}", lambda g, c=color: dsl_keep_only_color(g, c)
        except Exception:
            continue

    # Try remove_color
    for color in range(10):
        try:
            matches = True
            for p in pairs:
                gi = Grid(p['input'])
                go = Grid(p['output'])
                if dsl_remove_color(gi, color) != go:
                    matches = False
                    break
            if matches:
                return f"remove_color_{color}", lambda g, c=color: dsl_remove_color(g, c)
        except Exception:
            continue

    return None, None


def search_two_step(pairs):
    """Try compositions of two operations."""
    # Only try promising pairs to keep search fast
    fast_ops = [
        ("rot90", dsl_rot90), ("rot180", dsl_rot180), ("rot270", dsl_rot270),
        ("flip_h", dsl_flip_h), ("flip_v", dsl_flip_v), ("transpose", dsl_transpose),
        ("crop_nz", dsl_crop_nonzero), ("crop_bg", dsl_crop_to_bg),
        ("top_half", dsl_top_half), ("bottom_half", dsl_bottom_half),
        ("left_half", dsl_left_half), ("right_half", dsl_right_half),
        ("outline", dsl_outline), ("fill_holes", dsl_fill_holes),
        ("grav_down", dsl_gravity_down),
        ("largest_obj", dsl_largest_object),
        ("unique_rows", dsl_unique_rows),
    ]

    for name1, op1 in fast_ops:
        for name2, op2 in fast_ops:
            if name1 == name2 == "identity":
                continue
            try:
                matches = True
                for p in pairs:
                    gi = Grid(p['input'])
                    go = Grid(p['output'])
                    result = op2(op1(gi))
                    if result != go:
                        matches = False
                        break
                if matches:
                    return f"{name1}+{name2}", lambda g, a=op1, b=op2: b(a(g))
            except Exception:
                continue

    return None, None


def search_object_extraction(pairs):
    """Try extracting specific objects by index, size, or color."""
    # Check if output is one of the input's objects
    for p in pairs:
        gi = Grid(p['input'])
        go = Grid(p['output'])
        objs = gi.objects()
        if not objs:
            return None, None

    # Try: output = object at specific index (sorted by size descending)
    for sort_key in ['size_desc', 'size_asc', 'top', 'left']:
        try:
            for idx in range(5):
                matches = True
                for p in pairs:
                    gi = Grid(p['input'])
                    go = Grid(p['output'])
                    objs = gi.objects()
                    if sort_key == 'size_desc':
                        objs.sort(key=lambda o: -o.size)
                    elif sort_key == 'size_asc':
                        objs.sort(key=lambda o: o.size)
                    elif sort_key == 'top':
                        objs.sort(key=lambda o: o.r0)
                    elif sort_key == 'left':
                        objs.sort(key=lambda o: o.c0)
                    if idx >= len(objs):
                        matches = False
                        break
                    obj_grid = objs[idx].as_grid(bg=0)
                    if obj_grid != go:
                        matches = False
                        break
                if matches:
                    def make_extractor(sk, ix):
                        def extract(g):
                            objs = g.objects()
                            if sk == 'size_desc':
                                objs.sort(key=lambda o: -o.size)
                            elif sk == 'size_asc':
                                objs.sort(key=lambda o: o.size)
                            elif sk == 'top':
                                objs.sort(key=lambda o: o.r0)
                            elif sk == 'left':
                                objs.sort(key=lambda o: o.c0)
                            if ix < len(objs):
                                return objs[ix].as_grid(bg=0)
                            return g.copy()
                        return extract
                    return f"obj_{sort_key}[{idx}]", make_extractor(sort_key, idx)
        except Exception:
            continue

    # Try: output = object of a specific color
    for color in range(1, 10):
        try:
            matches = True
            for p in pairs:
                gi = Grid(p['input'])
                go = Grid(p['output'])
                objs = [o for o in gi.objects() if o.color == color]
                if len(objs) != 1:
                    matches = False
                    break
                if objs[0].as_grid(bg=0) != go:
                    matches = False
                    break
            if matches:
                def make_color_extract(c):
                    def extract(g):
                        objs = [o for o in g.objects() if o.color == c]
                        if objs:
                            return objs[0].as_grid(bg=0)
                        return g.copy()
                    return extract
                return f"obj_color_{color}", make_color_extract(color)
        except Exception:
            continue

    return None, None


def search_majority_color_fill(pairs):
    """Replace minority colors with majority, or fill bg with dominant."""
    # Try: output replaces all non-bg colors with the most common non-bg color
    try:
        matches = True
        for p in pairs:
            gi = Grid(p['input'])
            go = Grid(p['output'])
            if gi.shape != go.shape:
                matches = False
                break
            bg = gi.background
            nz_counts = {c: n for c, n in gi.color_counts.items() if c != bg}
            if not nz_counts:
                matches = False
                break
            majority = max(nz_counts, key=nz_counts.get)
            arr = gi.arr.copy()
            for c in nz_counts:
                if c != majority:
                    arr[arr == c] = majority
            if not np.array_equal(arr, go.arr):
                matches = False
                break
        if matches:
            def majority_fill(g):
                bg = g.background
                nz = {c: n for c, n in g.color_counts.items() if c != bg}
                if not nz:
                    return g.copy()
                maj = max(nz, key=nz.get)
                arr = g.arr.copy()
                for c in nz:
                    if c != maj:
                        arr[arr == c] = maj
                return Grid(arr)
            return "majority_fill", majority_fill
    except Exception:
        pass
    return None, None


def search_neighbor_count_rule(pairs):
    """Conditional: cell color depends on count of nonzero neighbors."""
    if not all(Grid(p['input']).shape == Grid(p['output']).shape for p in pairs):
        return None, None

    # Learn: for each (input_color, neighbor_count) -> output_color
    rules = {}
    consistent = True
    for p in pairs:
        gi = Grid(p['input'])
        go = Grid(p['output'])
        h, w = gi.h, gi.w
        for r in range(h):
            for c in range(w):
                ic = int(gi.arr[r, c])
                oc = int(go.arr[r, c])
                n_count = 0
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < h and 0 <= nc < w and gi.arr[nr, nc] != 0:
                        n_count += 1
                key = (ic, n_count)
                if key in rules and rules[key] != oc:
                    consistent = False
                    break
                rules[key] = oc
            if not consistent:
                break
        if not consistent:
            break

    if consistent and rules:
        # Verify it actually changes something
        all_identity = all(k[0] == v for k, v in rules.items())
        if all_identity:
            return None, None

        def apply_neighbor_rule(g, r=rules):
            arr = g.arr.copy()
            h, w = g.h, g.w
            result = arr.copy()
            for row in range(h):
                for col in range(w):
                    ic = int(arr[row, col])
                    n_count = 0
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = row+dr, col+dc
                        if 0 <= nr < h and 0 <= nc < w and arr[nr, nc] != 0:
                            n_count += 1
                    key = (ic, n_count)
                    if key in r:
                        result[row, col] = r[key]
                    else:
                        result[row, col] = ic
            return Grid(result)

        # Final verify
        for p in pairs:
            if apply_neighbor_rule(Grid(p['input'])) != Grid(p['output']):
                return None, None
        return "neighbor_rule", apply_neighbor_rule

    return None, None


def search_symmetry_completion(pairs):
    """Complete a grid to be symmetric (h, v, or diagonal)."""
    if not all(Grid(p['input']).shape == Grid(p['output']).shape for p in pairs):
        return None, None

    for sym_type in ['h', 'v', 'both', 'diag']:
        try:
            matches = True
            for p in pairs:
                gi = Grid(p['input'])
                go = Grid(p['output'])
                arr = gi.arr.copy()
                h, w = arr.shape

                if sym_type == 'h':
                    for r in range(h):
                        for c in range(w):
                            mc = w - 1 - c
                            if arr[r, c] == 0 and arr[r, mc] != 0:
                                arr[r, c] = arr[r, mc]
                            elif arr[r, mc] == 0 and arr[r, c] != 0:
                                arr[r, mc] = arr[r, c]
                elif sym_type == 'v':
                    for r in range(h):
                        for c in range(w):
                            mr = h - 1 - r
                            if arr[r, c] == 0 and arr[mr, c] != 0:
                                arr[r, c] = arr[mr, c]
                            elif arr[mr, c] == 0 and arr[r, c] != 0:
                                arr[mr, c] = arr[r, c]
                elif sym_type == 'both':
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
                elif sym_type == 'diag':
                    if h != w:
                        matches = False
                        break
                    for r in range(h):
                        for c in range(w):
                            if arr[r, c] == 0 and arr[c, r] != 0:
                                arr[r, c] = arr[c, r]
                            elif arr[c, r] == 0 and arr[r, c] != 0:
                                arr[c, r] = arr[r, c]

                if not np.array_equal(arr, go.arr):
                    matches = False
                    break

            if matches:
                def make_sym_complete(st):
                    def complete(g):
                        arr = g.arr.copy()
                        h, w = arr.shape
                        if st == 'h':
                            for r in range(h):
                                for c in range(w):
                                    mc = w-1-c
                                    if arr[r,c]==0 and arr[r,mc]!=0: arr[r,c]=arr[r,mc]
                                    elif arr[r,mc]==0 and arr[r,c]!=0: arr[r,mc]=arr[r,c]
                        elif st == 'v':
                            for r in range(h):
                                for c in range(w):
                                    mr = h-1-r
                                    if arr[r,c]==0 and arr[mr,c]!=0: arr[r,c]=arr[mr,c]
                                    elif arr[mr,c]==0 and arr[r,c]!=0: arr[mr,c]=arr[r,c]
                        elif st == 'both':
                            for r in range(h):
                                for c in range(w):
                                    mr,mc = h-1-r, w-1-c
                                    vals = [arr[r,c],arr[r,mc],arr[mr,c],arr[mr,mc]]
                                    nz = [v for v in vals if v!=0]
                                    if nz:
                                        fill = Counter(nz).most_common(1)[0][0]
                                        if arr[r,c]==0: arr[r,c]=fill
                                        if arr[r,mc]==0: arr[r,mc]=fill
                                        if arr[mr,c]==0: arr[mr,c]=fill
                                        if arr[mr,mc]==0: arr[mr,mc]=fill
                        elif st == 'diag' and h == w:
                            for r in range(h):
                                for c in range(w):
                                    if arr[r,c]==0 and arr[c,r]!=0: arr[r,c]=arr[c,r]
                                    elif arr[c,r]==0 and arr[r,c]!=0: arr[c,r]=arr[r,c]
                        return Grid(arr)
                    return complete
                return f"sym_{sym_type}", make_sym_complete(sym_type)
        except Exception:
            continue

    return None, None


def search_grid_diff_apply(pairs):
    """Learn diff between input/output on training, apply to test."""
    if not all(Grid(p['input']).shape == Grid(p['output']).shape for p in pairs):
        return None, None

    # Check if the SAME set of cells changes in every example
    # and the change is always input_color -> fixed_output_color
    ref_positions = None
    position_rule = {}  # (r, c) -> output_color

    for p in pairs:
        gi = Grid(p['input'])
        go = Grid(p['output'])
        changed = set()
        for r in range(gi.h):
            for c in range(gi.w):
                if gi.arr[r, c] != go.arr[r, c]:
                    changed.add((r, c))
                    position_rule[(r, c)] = int(go.arr[r, c])

        if ref_positions is None:
            ref_positions = changed
        elif changed != ref_positions:
            return None, None

    if ref_positions and len(ref_positions) < gi.h * gi.w // 2:
        def apply_diff(g, pr=position_rule):
            arr = g.arr.copy()
            for (r, c), color in pr.items():
                if r < arr.shape[0] and c < arr.shape[1]:
                    arr[r, c] = color
            return Grid(arr)

        # Verify
        for p in pairs:
            if apply_diff(Grid(p['input'])) != Grid(p['output']):
                return None, None
        return "grid_diff", apply_diff

    return None, None


def search_per_object_transform(pairs):
    """Apply same transformation to each object independently."""
    if not all(Grid(p['input']).shape == Grid(p['output']).shape for p in pairs):
        return None, None

    # Check: each object in output is a transformed version of corresponding input object
    # Try: recolor each object based on some property (size, position)
    for p in pairs:
        gi = Grid(p['input'])
        go = Grid(p['output'])
        in_objs = gi.objects()
        out_objs = go.objects()
        if len(in_objs) != len(out_objs) or len(in_objs) == 0:
            return None, None

    # Try: each object's color changes based on its size
    size_to_color = {}
    consistent = True
    for p in pairs:
        gi = Grid(p['input'])
        go = Grid(p['output'])
        in_objs = sorted(gi.objects(), key=lambda o: (o.r0, o.c0))
        out_objs = sorted(go.objects(), key=lambda o: (o.r0, o.c0))
        if len(in_objs) != len(out_objs):
            consistent = False
            break
        for io, oo in zip(in_objs, out_objs):
            if io.cells != oo.cells:
                # Objects moved or reshaped — too complex
                consistent = False
                break
            s = io.size
            if s in size_to_color and size_to_color[s] != oo.color:
                consistent = False
                break
            size_to_color[s] = oo.color
        if not consistent:
            break

    if consistent and size_to_color:
        def recolor_by_size(g, stc=size_to_color):
            arr = g.arr.copy()
            for obj in g.objects():
                if obj.size in stc:
                    for r, c in obj.cells:
                        arr[r, c] = stc[obj.size]
            return Grid(arr)

        for p in pairs:
            if recolor_by_size(Grid(p['input'])) != Grid(p['output']):
                return None, None
        return "recolor_by_size", recolor_by_size

    return None, None


def search_tile_with_pattern(pairs):
    """Output is input tiled where nonzero cells of input determine tile placement."""
    # Classic ARC pattern: input is a small grid, each nonzero cell gets replaced
    # by the input itself, zero cells become zero blocks
    for p in pairs:
        gi = Grid(p['input'])
        go = Grid(p['output'])
        ih, iw = gi.shape
        oh, ow = go.shape
        if oh % ih != 0 or ow % iw != 0:
            return None, None
        if oh // ih != ih or ow // iw != iw:
            return None, None

    try:
        matches = True
        for p in pairs:
            gi = Grid(p['input'])
            go = Grid(p['output'])
            ih, iw = gi.shape
            result = np.zeros((ih * ih, iw * iw), dtype=np.int32)
            for r in range(ih):
                for c in range(iw):
                    if gi.arr[r, c] != 0:
                        result[r*ih:(r+1)*ih, c*iw:(c+1)*iw] = gi.arr
            if not np.array_equal(result, go.arr):
                matches = False
                break
        if matches:
            def tile_self(g):
                h, w = g.shape
                result = np.zeros((h*h, w*w), dtype=np.int32)
                for r in range(h):
                    for c in range(w):
                        if g.arr[r, c] != 0:
                            result[r*h:(r+1)*h, c*w:(c+1)*w] = g.arr
                return Grid(result)
            return "tile_self", tile_self
    except Exception:
        pass

    return None, None


def search_flood_fill_enclosed(pairs):
    """Fill enclosed regions with a specific color."""
    if not all(Grid(p['input']).shape == Grid(p['output']).shape for p in pairs):
        return None, None

    try:
        matches = True
        fill_color = None
        for p in pairs:
            gi = Grid(p['input'])
            go = Grid(p['output'])
            diff = go.arr - gi.arr
            changed = np.argwhere(diff != 0)
            if len(changed) == 0:
                matches = False
                break
            new_colors = set(int(go.arr[r, c]) for r, c in changed)
            if len(new_colors) != 1:
                matches = False
                break
            fc = new_colors.pop()
            if fill_color is None:
                fill_color = fc
            elif fill_color != fc:
                matches = False
                break

        if matches and fill_color is not None:
            result = dsl_fill_holes(Grid(pairs[0]['input']))
            # Check if fill_holes with recolor matches
            def fill_enclosed(g, fc=fill_color):
                arr = g.arr.copy()
                h, w = arr.shape
                exterior = np.zeros((h, w), dtype=bool)
                stack = []
                for r in range(h):
                    for c in [0, w-1]:
                        if arr[r, c] == 0:
                            stack.append((r, c))
                for c in range(w):
                    for r in [0, h-1]:
                        if arr[r, c] == 0:
                            stack.append((r, c))
                while stack:
                    r, c = stack.pop()
                    if r < 0 or r >= h or c < 0 or c >= w:
                        continue
                    if exterior[r, c] or arr[r, c] != 0:
                        continue
                    exterior[r, c] = True
                    stack.extend([(r-1,c),(r+1,c),(r,c-1),(r,c+1)])
                for r in range(h):
                    for c in range(w):
                        if arr[r, c] == 0 and not exterior[r, c]:
                            arr[r, c] = fc
                return Grid(arr)

            for p in pairs:
                if fill_enclosed(Grid(p['input'])) != Grid(p['output']):
                    return None, None
            return f"fill_enclosed_{fill_color}", fill_enclosed
    except Exception:
        pass

    return None, None


def search_8neighbor_rule(pairs):
    """Like neighbor_rule but with 8-connectivity (includes diagonals)."""
    if not all(Grid(p['input']).shape == Grid(p['output']).shape for p in pairs):
        return None, None

    rules = {}
    consistent = True
    for p in pairs:
        gi = Grid(p['input'])
        go = Grid(p['output'])
        h, w = gi.h, gi.w
        for r in range(h):
            for c in range(w):
                ic = int(gi.arr[r, c])
                oc = int(go.arr[r, c])
                n_count = 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < h and 0 <= nc < w and gi.arr[nr, nc] != 0:
                            n_count += 1
                key = (ic, n_count)
                if key in rules and rules[key] != oc:
                    consistent = False
                    break
                rules[key] = oc
            if not consistent:
                break
        if not consistent:
            break

    if consistent and rules:
        all_identity = all(k[0] == v for k, v in rules.items())
        if all_identity:
            return None, None

        def apply_8n_rule(g, r=rules):
            arr = g.arr.copy()
            h, w = g.h, g.w
            result = arr.copy()
            for row in range(h):
                for col in range(w):
                    ic = int(arr[row, col])
                    n_count = 0
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = row+dr, col+dc
                            if 0 <= nr < h and 0 <= nc < w and arr[nr, nc] != 0:
                                n_count += 1
                    key = (ic, n_count)
                    if key in r:
                        result[row, col] = r[key]
                return Grid(result)

        for p in pairs:
            if apply_8n_rule(Grid(p['input'])) != Grid(p['output']):
                return None, None
        return "8neighbor_rule", apply_8n_rule
    return None, None


def search_color_neighbor_rule(pairs):
    """Cell output depends on its color AND the set of neighbor colors."""
    if not all(Grid(p['input']).shape == Grid(p['output']).shape for p in pairs):
        return None, None

    rules = {}
    consistent = True
    for p in pairs:
        gi = Grid(p['input'])
        go = Grid(p['output'])
        h, w = gi.h, gi.w
        for r in range(h):
            for c in range(w):
                ic = int(gi.arr[r, c])
                oc = int(go.arr[r, c])
                neighbors = set()
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < h and 0 <= nc < w:
                        neighbors.add(int(gi.arr[nr, nc]))
                key = (ic, frozenset(neighbors))
                if key in rules and rules[key] != oc:
                    consistent = False
                    break
                rules[key] = oc
            if not consistent:
                break
        if not consistent:
            break

    if consistent and rules:
        all_identity = all(k[0] == v for k, v in rules.items())
        if all_identity:
            return None, None

        def apply_cn_rule(g, r=rules):
            arr = g.arr.copy()
            h, w = g.h, g.w
            result = arr.copy()
            for row in range(h):
                for col in range(w):
                    ic = int(arr[row, col])
                    neighbors = set()
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = row+dr, col+dc
                        if 0 <= nr < h and 0 <= nc < w:
                            neighbors.add(int(arr[nr, nc]))
                    key = (ic, frozenset(neighbors))
                    if key in r:
                        result[row, col] = r[key]
            return Grid(result)

        for p in pairs:
            if apply_cn_rule(Grid(p['input'])) != Grid(p['output']):
                return None, None
        return "color_neighbor_rule", apply_cn_rule
    return None, None


def search_subgrid_repeat(pairs):
    """Output is made of repeating subgrids extracted from input."""
    for p in pairs:
        gi = Grid(p['input'])
        go = Grid(p['output'])
        # Try: output is tiled version of a subgrid of input
        for sh in range(1, min(gi.h+1, 8)):
            for sw in range(1, min(gi.w+1, 8)):
                if go.h % sh != 0 or go.w % sw != 0:
                    continue
                sub = gi.arr[:sh, :sw]
                tiled = np.tile(sub, (go.h // sh, go.w // sw))
                if tiled.shape == go.arr.shape and np.array_equal(tiled, go.arr):
                    # Found! Check if consistent across examples
                    all_match = True
                    for p2 in pairs:
                        gi2 = Grid(p2['input'])
                        go2 = Grid(p2['output'])
                        sub2 = gi2.arr[:sh, :sw]
                        if go2.h % sh != 0 or go2.w % sw != 0:
                            all_match = False
                            break
                        tiled2 = np.tile(sub2, (go2.h // sh, go2.w // sw))
                        if not np.array_equal(tiled2, go2.arr):
                            all_match = False
                            break
                    if all_match:
                        def make_subgrid_tile(height, width):
                            def tile(g):
                                sub = g.arr[:height, :width]
                                # Guess output size from training
                                return Grid(np.tile(sub, (g.h, g.w)))
                            return tile
                        # Need to figure out output size...
                        rh = go.h // sh
                        rw = go.w // sw
                        def make_fixed_tile(height, width, rh_, rw_):
                            def tile(g):
                                sub = g.arr[:height, :width]
                                return Grid(np.tile(sub, (rh_, rw_)))
                            return tile
                        return f"subgrid_{sh}x{sw}_tile", make_fixed_tile(sh, sw, rh, rw)
        break  # Only check first pair for subgrid detection
    return None, None


def search_color_frequency_map(pairs):
    """Map colors based on their frequency rank in the input."""
    if not all(Grid(p['input']).shape == Grid(p['output']).shape for p in pairs):
        return None, None

    # For each example: rank colors by frequency, map rank -> output color
    rank_maps = []
    consistent = True
    for p in pairs:
        gi = Grid(p['input'])
        go = Grid(p['output'])
        in_counts = gi.color_counts
        # Sort by count descending
        ranked = sorted(in_counts.keys(), key=lambda c: -in_counts[c])

        # Build rank -> output color mapping
        rank_map = {}
        for r in range(gi.h):
            for c in range(gi.w):
                ic = int(gi.arr[r, c])
                oc = int(go.arr[r, c])
                rank = ranked.index(ic) if ic in ranked else -1
                if rank in rank_map and rank_map[rank] != oc:
                    consistent = False
                    break
                rank_map[rank] = oc
            if not consistent:
                break
        if not consistent:
            break
        rank_maps.append(rank_map)

    if consistent and rank_maps:
        # Check if all examples use the same rank mapping
        ref_map = rank_maps[0]
        if all(rm == ref_map for rm in rank_maps):
            def apply_freq_map(g, rm=ref_map):
                in_counts = g.color_counts
                ranked = sorted(in_counts.keys(), key=lambda c: -in_counts[c])
                arr = g.arr.copy()
                for r in range(g.h):
                    for c in range(g.w):
                        ic = int(arr[r, c])
                        rank = ranked.index(ic) if ic in ranked else -1
                        if rank in rm:
                            arr[r, c] = rm[rank]
                return Grid(arr)

            for p in pairs:
                if apply_freq_map(Grid(p['input'])) != Grid(p['output']):
                    return None, None
            return "color_freq_map", apply_freq_map
    return None, None


def search_row_col_pattern(pairs):
    """Each row or column follows a pattern learned from training."""
    if not all(Grid(p['input']).shape == Grid(p['output']).shape for p in pairs):
        return None, None

    # Try: output row i = some function of input row i
    # Check if there's a per-row transformation (e.g., sort, reverse, shift)
    for transform_name, transform_fn in [
        ("row_reverse", lambda row: row[::-1].tolist()),
        ("row_sort", lambda row: sorted(row.tolist())),
        ("row_sort_rev", lambda row: sorted(row.tolist(), reverse=True)),
        ("row_shift1", lambda row: np.roll(row, 1).tolist()),
        ("row_shift_1", lambda row: np.roll(row, -1).tolist()),
    ]:
        matches = True
        for p in pairs:
            gi = Grid(p['input'])
            go = Grid(p['output'])
            for r in range(gi.h):
                if transform_fn(gi.arr[r]) != go.arr[r].tolist():
                    matches = False
                    break
            if not matches:
                break
        if matches:
            def apply_row_transform(g, fn=transform_fn):
                arr = g.arr.copy()
                for r in range(g.h):
                    arr[r] = fn(arr[r])
                return Grid(arr)
            return f"per_{transform_name}", apply_row_transform

    # Try per-column transforms
    for transform_name, transform_fn in [
        ("col_reverse", lambda col: col[::-1].tolist()),
        ("col_sort", lambda col: sorted(col.tolist())),
        ("col_sort_rev", lambda col: sorted(col.tolist(), reverse=True)),
    ]:
        matches = True
        for p in pairs:
            gi = Grid(p['input'])
            go = Grid(p['output'])
            for c in range(gi.w):
                if transform_fn(gi.arr[:, c]) != go.arr[:, c].tolist():
                    matches = False
                    break
            if not matches:
                break
        if matches:
            def apply_col_transform(g, fn=transform_fn):
                arr = g.arr.copy()
                for c in range(g.w):
                    arr[:, c] = fn(arr[:, c])
                return Grid(arr)
            return f"per_{transform_name}", apply_col_transform

    return None, None


def search_horizontal_concat(pairs):
    """Output = left half OP right half, or similar split-and-combine."""
    for split in ['h_halves', 'v_halves']:
        for op_name, op_fn in [
            ('and', lambda a, b: np.where((a != 0) & (b != 0), a, 0)),
            ('xor', lambda a, b: np.where((a != 0) ^ (b != 0), np.where(a != 0, a, b), 0)),
            ('or', lambda a, b: np.where(a != 0, a, b)),
            ('diff', lambda a, b: np.where(a != b, a, 0)),
            ('max', lambda a, b: np.maximum(a, b)),
            ('min_nz', lambda a, b: np.where((a != 0) & (b != 0), np.minimum(a, b), 0)),
        ]:
            try:
                matches = True
                for p in pairs:
                    gi = Grid(p['input'])
                    go = Grid(p['output'])
                    if split == 'h_halves':
                        if gi.w % 2 != 0:
                            matches = False
                            break
                        left = gi.arr[:, :gi.w//2]
                        right = gi.arr[:, gi.w//2:]
                    else:
                        if gi.h % 2 != 0:
                            matches = False
                            break
                        left = gi.arr[:gi.h//2, :]
                        right = gi.arr[gi.h//2:, :]
                    if left.shape != right.shape or left.shape != (go.h, go.w):
                        matches = False
                        break
                    result = op_fn(left, right)
                    if not np.array_equal(result, go.arr):
                        matches = False
                        break
                if matches:
                    def make_combine(sp, fn):
                        def combine(g):
                            if sp == 'h_halves':
                                l = g.arr[:, :g.w//2]
                                r = g.arr[:, g.w//2:]
                            else:
                                l = g.arr[:g.h//2, :]
                                r = g.arr[g.h//2:, :]
                            return Grid(fn(l, r))
                        return combine
                    return f"{split}_{op_name}", make_combine(split, op_fn)
            except Exception:
                continue
    return None, None


def search_mask_from_color(pairs):
    """One color acts as a mask/template, applied to transform another region."""
    if not all(Grid(p['input']).shape == Grid(p['output']).shape for p in pairs):
        return None, None

    # Try: cells of color X become color Y in output, rest unchanged
    for mask_color in range(1, 10):
        for target_color in range(0, 10):
            if mask_color == target_color:
                continue
            try:
                matches = True
                for p in pairs:
                    gi = Grid(p['input'])
                    go = Grid(p['output'])
                    arr = gi.arr.copy()
                    arr[arr == mask_color] = target_color
                    if not np.array_equal(arr, go.arr):
                        matches = False
                        break
                if matches:
                    def make_mask_replace(mc, tc):
                        def replace(g):
                            arr = g.arr.copy()
                            arr[arr == mc] = tc
                            return Grid(arr)
                        return replace
                    return f"mask_{mask_color}_to_{target_color}", make_mask_replace(mask_color, target_color)
            except Exception:
                continue
    return None, None


def search_halves_combine(pairs):
    """Try combining halves of the grid (XOR, AND, OR)."""
    for p in pairs:
        gi = Grid(p['input'])
        go = Grid(p['output'])
        # Check if input can be split into two halves
        if gi.h % 2 == 0:
            top = gi.arr[:gi.h//2, :]
            bot = gi.arr[gi.h//2:, :]
            if top.shape == bot.shape and top.shape == (go.h, go.w):
                # Try XOR-like: nonzero from either
                xor = np.where(top != 0, top, bot)
                if np.array_equal(xor, go.arr):
                    def combine_v_or(g):
                        t = g.arr[:g.h//2, :]
                        b = g.arr[g.h//2:, :]
                        return Grid(np.where(t != 0, t, b))
                    # Verify all
                    if all(combine_v_or(Grid(pp['input'])) == Grid(pp['output'])
                           for pp in pairs):
                        return "halves_v_or", combine_v_or

        if gi.w % 2 == 0:
            left = gi.arr[:, :gi.w//2]
            right = gi.arr[:, gi.w//2:]
            if left.shape == right.shape and left.shape == (go.h, go.w):
                xor = np.where(left != 0, left, right)
                if np.array_equal(xor, go.arr):
                    def combine_h_or(g):
                        l = g.arr[:, :g.w//2]
                        r = g.arr[:, g.w//2:]
                        return Grid(np.where(l != 0, l, r))
                    if all(combine_h_or(Grid(pp['input'])) == Grid(pp['output'])
                           for pp in pairs):
                        return "halves_h_or", combine_h_or

    return None, None


def search_border_extract(pairs):
    """Try extracting border or interior."""
    for mode in ['border', 'interior']:
        try:
            matches = True
            for p in pairs:
                gi = Grid(p['input'])
                go = Grid(p['output'])
                if gi.shape != go.shape:
                    matches = False
                    break
                if mode == 'border':
                    result = Grid(np.zeros_like(gi.arr))
                    h, w = gi.h, gi.w
                    result.arr[0, :] = gi.arr[0, :]
                    result.arr[-1, :] = gi.arr[-1, :]
                    result.arr[:, 0] = gi.arr[:, 0]
                    result.arr[:, -1] = gi.arr[:, -1]
                else:
                    result = gi.copy()
                    result.arr[0, :] = 0
                    result.arr[-1, :] = 0
                    result.arr[:, 0] = 0
                    result.arr[:, -1] = 0
                if result != go:
                    matches = False
                    break
            if matches:
                if mode == 'border':
                    def extract_border(g):
                        r = Grid(np.zeros_like(g.arr))
                        r.arr[0, :] = g.arr[0, :]
                        r.arr[-1, :] = g.arr[-1, :]
                        r.arr[:, 0] = g.arr[:, 0]
                        r.arr[:, -1] = g.arr[:, -1]
                        return r
                    return "border", extract_border
                else:
                    def extract_interior(g):
                        r = g.copy()
                        r.arr[0, :] = 0; r.arr[-1, :] = 0
                        r.arr[:, 0] = 0; r.arr[:, -1] = 0
                        return r
                    return "interior", extract_interior
        except Exception:
            continue
    return None, None


# ══════════════════════════════════════════════════════════════════════
# 6. MASTER SOLVER
# ══════════════════════════════════════════════════════════════════════

def solve_task(task):
    """Try all search strategies in order of speed."""
    pairs = task['train']
    test_cases = task['test']

    searches = [
        ("single", search_single),
        ("color_map", search_color_map),
        ("parameterized", search_parameterized),
        ("object_extract", search_object_extraction),
        ("halves_combine", search_halves_combine),
        ("h_concat", search_horizontal_concat),
        ("border_extract", search_border_extract),
        ("majority_fill", search_majority_color_fill),
        ("neighbor_rule", search_neighbor_count_rule),
        ("8neighbor", search_8neighbor_rule),
        ("color_neighbor", search_color_neighbor_rule),
        ("symmetry", search_symmetry_completion),
        ("grid_diff", search_grid_diff_apply),
        ("per_object", search_per_object_transform),
        ("tile_pattern", search_tile_with_pattern),
        ("fill_enclosed", search_flood_fill_enclosed),
        ("subgrid_repeat", search_subgrid_repeat),
        ("color_freq", search_color_frequency_map),
        ("row_col", search_row_col_pattern),
        ("mask_color", search_mask_from_color),
        ("two_step", search_two_step),
    ]

    guesses_per_test = [[] for _ in test_cases]

    for search_name, search_fn in searches:
        try:
            name, op = search_fn(pairs)
            if op is not None:
                for i, tc in enumerate(test_cases):
                    if len(guesses_per_test[i]) < 2:
                        try:
                            result = op(Grid(tc['input']))
                            result_list = result.to_list()
                            # Avoid duplicate guesses
                            is_dup = any(result_list == g for g in guesses_per_test[i])
                            if not is_dup:
                                guesses_per_test[i].append(result_list)
                        except Exception:
                            pass
        except Exception:
            continue

    # Fallback: submit input as-is
    for i, tc in enumerate(test_cases):
        if not guesses_per_test[i]:
            guesses_per_test[i].append(tc['input'])

    return guesses_per_test


def score_task(task, predictions):
    """Score: correct if ANY attempt matches for ALL test cases."""
    all_correct = True
    for i, tc in enumerate(task['test']):
        if 'output' not in tc:
            continue
        expected = tc['output']
        preds = predictions[i] if i < len(predictions) else []
        correct = any(p == expected for p in preds)
        if not correct:
            all_correct = False
    return all_correct


# ══════════════════════════════════════════════════════════════════════
# 7. BENCHMARK RUNNER
# ══════════════════════════════════════════════════════════════════════

def run_benchmark(data_dir, limit=None, verbose=False):
    files = sorted(f for f in os.listdir(data_dir) if f.endswith('.json'))
    if limit:
        files = files[:limit]

    print("=" * 70)
    print(f"  ARC-AGI-2 Solver — Grid Parser + DSL + Program Search")
    print(f"  Tasks: {len(files)}")
    print(f"  Single ops: {len(SINGLE_OPS)} | Search strategies: 21")
    print("=" * 70)

    solved = 0
    total = 0
    strategy_wins = Counter()
    t0 = time.time()

    for f in files:
        task = json.load(open(os.path.join(data_dir, f)))
        total += 1
        predictions = solve_task(task)
        correct = score_task(task, predictions)

        if correct:
            solved += 1
            # Identify winning strategy
            for sname, sfn in [("single", search_single), ("color_map", search_color_map),
                                  ("parameterized", search_parameterized),
                                  ("object_extract", search_object_extraction),
                                  ("halves_combine", search_halves_combine),
                                  ("h_concat", search_horizontal_concat),
                                  ("border_extract", search_border_extract),
                                  ("majority_fill", search_majority_color_fill),
                                  ("neighbor_rule", search_neighbor_count_rule),
                                  ("8neighbor", search_8neighbor_rule),
                                  ("color_neighbor", search_color_neighbor_rule),
                                  ("symmetry", search_symmetry_completion),
                                  ("grid_diff", search_grid_diff_apply),
                                  ("per_object", search_per_object_transform),
                                  ("tile_pattern", search_tile_with_pattern),
                                  ("fill_enclosed", search_flood_fill_enclosed),
                                  ("subgrid_repeat", search_subgrid_repeat),
                                  ("color_freq", search_color_frequency_map),
                                  ("row_col", search_row_col_pattern),
                                  ("mask_color", search_mask_from_color),
                                  ("two_step", search_two_step)]:
                try:
                    name, op = sfn(task['train'])
                    if op:
                        strategy_wins[f"{sname}:{name}"] += 1
                        break
                except Exception:
                    continue

        if verbose:
            status = "SOLVED" if correct else "failed"
            print(f"  {f[:16]}: {status}")

    elapsed = time.time() - t0
    pct = solved / max(total, 1) * 100

    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    print(f"  Solved:     {solved}/{total} ({pct:.1f}%)")
    print(f"  Time:       {elapsed:.1f}s ({elapsed/max(total,1)*100:.0f}ms/task)")

    if strategy_wins:
        print(f"\n  Winning strategies:")
        for name, count in strategy_wins.most_common(15):
            print(f"    {name:<35}: {count}")

    print(f"\n{'='*70}")
    return solved, total


def run_tests():
    print("=" * 65)
    print("  ARC Solver Tests")
    print("=" * 65)
    p = 0; t = 0

    print("\n  T1: Grid parser")
    g = Grid([[1,0,0],[0,2,2],[0,2,2]])
    ok = g.background == 0 and len(g.objects()) == 2
    print(f"    Objects: {len(g.objects())}, bg={g.background} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T2: Object extraction")
    objs = g.objects()
    sizes = sorted([o.size for o in objs])
    ok = sizes == [1, 4]
    print(f"    Sizes: {sizes} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T3: DSL operations")
    g2 = Grid([[1,2],[3,4]])
    ok = dsl_rot90(g2) == Grid([[3,1],[4,2]])
    print(f"    Rot90: {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T4: Crop nonzero")
    g3 = Grid([[0,0,0],[0,5,6],[0,7,0]])
    cropped = dsl_crop_nonzero(g3)
    ok = cropped == Grid([[5,6],[7,0]])
    print(f"    Crop: {cropped.shape} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T5: Color map inference")
    pairs = [{'input': [[0,1],[1,0]], 'output': [[0,2],[2,0]]}]
    m = infer_color_map(pairs)
    ok = m == {0: 0, 1: 2}
    print(f"    Map: {m} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T6: Search single finds rotation")
    task = {'train': [{'input': [[1,2],[3,4]], 'output': [[3,1],[4,2]]}],
            'test': [{'input': [[5,6],[7,8]], 'output': [[7,5],[8,6]]}]}
    name, op = search_single(task['train'])
    ok = name == "rot90"
    print(f"    Found: {name} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T7: Two-step search")
    task2 = {'train': [{'input': [[1,2,3],[4,5,6]], 'output': [[4,1],[5,2],[6,3]]}],
             'test': [{'input': [[7,8],[9,0]], 'output': [[9,7],[0,8]]}]}
    name, op = search_two_step(task2['train'])
    ok = name is not None
    print(f"    Found: {name} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T8: Gravity down")
    g4 = Grid([[1,0,2],[0,3,0],[0,0,4]])
    gd = dsl_gravity_down(g4)
    ok = gd.arr[-1, 0] == 1 and gd.arr[-1, 1] == 3
    print(f"    Bottom row: {gd.arr[-1].tolist()} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T9: Solve task end-to-end")
    preds = solve_task(task)
    ok = score_task(task, preds)
    print(f"    Rot90 task solved: {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T10: Run mini benchmark")
    arc_dir = None
    for base in ["ARC-AGI-2", "/home/claude/ARC-AGI-2"]:
        d = os.path.join(base, "data", "training")
        if os.path.exists(d):
            arc_dir = d
            break
    if arc_dir:
        solved, total = run_benchmark(arc_dir, limit=50, verbose=False)
        ok = total == 50
        print(f"    {solved}/{total} solved {'PASS' if ok else 'FAIL'}")
    else:
        ok = True
        print(f"    No ARC data (skip) PASS")
    p += int(ok); t += 1

    print(f"\n{'='*65}")
    print(f"  Results: {p}/{t} tests passed")
    print(f"{'='*65}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--training", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--task", type=str, default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--data", type=str, default=None,
                     help="Path to ARC-AGI-2/data directory")
    args = ap.parse_args()

    # Find data directory
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
    elif args.task:
        for sub in ["training", "evaluation"]:
            d = os.path.join(data_base, sub)
            path = os.path.join(d, f"{args.task}.json")
            if os.path.exists(path):
                task = json.load(open(path))
                preds = solve_task(task)
                correct = score_task(task, preds)
                print(f"Task {args.task}: {'SOLVED' if correct else 'FAILED'}")
                break
    else:
        if args.eval:
            d = os.path.join(data_base, "evaluation")
        else:
            d = os.path.join(data_base, "training")
        run_benchmark(d, limit=args.limit, verbose=args.verbose)
