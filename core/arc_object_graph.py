"""
arc_object_graph.py — Object-Relational Solver for ARC-AGI-2
==============================================================
Converts pixel grids into object graphs with:
  - Properties: size, color, shape, position, bbox, aspect ratio
  - Relations: touches, aligned_x, aligned_y, inside, nearest, same_color, same_size
  - Conditional rules: IF property THEN transform

This is the missing layer between single ops (72/1000)
and compositional reasoning (target: 100+).

The key insight: ARC tasks that our DSL misses almost always
involve OBJECT-LEVEL reasoning, not pixel-level transforms.

Usage:
    python arc_object_graph.py --test
    python arc_object_graph.py --training --data path/to/ARC-AGI-2/data -v
"""

import argparse
import json
import os
import numpy as np
from collections import Counter
from pathlib import Path
import time

from arc_solver import Grid, score_task

try:
    from arc_phase2 import solve_task_phase2 as s1_solve
except ImportError:
    from arc_solver import solve_task as s1_solve


# ══════════════════════════════════════════════════════════════════════
# 1. OBJECT EXTRACTION
# ══════════════════════════════════════════════════════════════════════

class ArcObject:
    """A connected component in an ARC grid with computed properties."""
    __slots__ = ['color', 'cells', 'size', 'bbox', 'center',
                 'width', 'height', 'aspect', 'shape_hash',
                 'touches_border', 'is_rectangular', 'density', 'id']

    def __init__(self, cells, color, grid_h, grid_w, obj_id=0):
        self.id = obj_id
        self.color = color
        self.cells = cells  # list of (r, c)
        self.size = len(cells)

        rs = [r for r, c in cells]
        cs = [c for r, c in cells]
        self.bbox = (min(rs), max(rs), min(cs), max(cs))
        self.center = (np.mean(rs), np.mean(cs))
        self.height = self.bbox[1] - self.bbox[0] + 1
        self.width = self.bbox[3] - self.bbox[2] + 1
        self.aspect = self.width / max(self.height, 1)
        self.density = self.size / max(self.height * self.width, 1)
        self.is_rectangular = (self.size == self.height * self.width)
        self.touches_border = any(
            r == 0 or r == grid_h - 1 or c == 0 or c == grid_w - 1
            for r, c in cells
        )
        # Shape hash: normalized cell positions for shape matching
        min_r, min_c = min(rs), min(cs)
        normalized = tuple(sorted((r - min_r, c - min_c) for r, c in cells))
        self.shape_hash = hash(normalized)

    def crop(self, grid):
        """Extract the bounding box from the grid."""
        r1, r2, c1, c2 = self.bbox
        return grid[r1:r2+1, c1:c2+1].copy()

    def mask(self, shape):
        """Create a boolean mask for this object."""
        m = np.zeros(shape, dtype=bool)
        for r, c in self.cells:
            m[r, c] = True
        return m


def extract_objects(grid, bg=None):
    """Extract all connected components as ArcObject instances."""
    arr = np.array(grid) if not isinstance(grid, np.ndarray) else grid
    h, w = arr.shape
    if bg is None:
        bg = int(np.argmax(np.bincount(arr.flatten())))

    visited = np.zeros((h, w), dtype=bool)
    objects = []
    obj_id = 0

    for r in range(h):
        for c in range(w):
            if arr[r, c] != bg and not visited[r, c]:
                color = int(arr[r, c])
                cells = []
                stack = [(r, c)]
                while stack:
                    cr, cc = stack.pop()
                    if cr < 0 or cr >= h or cc < 0 or cc >= w:
                        continue
                    if visited[cr, cc] or int(arr[cr, cc]) != color:
                        continue
                    visited[cr, cc] = True
                    cells.append((cr, cc))
                    stack.extend([(cr-1,cc),(cr+1,cc),(cr,cc-1),(cr,cc+1)])
                if cells:
                    objects.append(ArcObject(cells, color, h, w, obj_id))
                    obj_id += 1

    return objects, bg


# ══════════════════════════════════════════════════════════════════════
# 2. OBJECT RELATIONS
# ══════════════════════════════════════════════════════════════════════

def manhattan_dist(obj_a, obj_b):
    return min(abs(r1-r2) + abs(c1-c2)
               for r1, c1 in obj_a.cells
               for r2, c2 in obj_b.cells)

def touches(obj_a, obj_b):
    """Do objects share an adjacent cell?"""
    cells_b = set(obj_b.cells)
    for r, c in obj_a.cells:
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            if (r+dr, c+dc) in cells_b:
                return True
    return False

def aligned_x(obj_a, obj_b):
    """Same horizontal center (within 1 pixel)."""
    return abs(obj_a.center[1] - obj_b.center[1]) < 1.5

def aligned_y(obj_a, obj_b):
    """Same vertical center (within 1 pixel)."""
    return abs(obj_a.center[0] - obj_b.center[0]) < 1.5

def same_shape(obj_a, obj_b):
    return obj_a.shape_hash == obj_b.shape_hash

def inside(obj_a, obj_b):
    """Is obj_a entirely inside obj_b's bounding box?"""
    return (obj_b.bbox[0] <= obj_a.bbox[0] and obj_a.bbox[1] <= obj_b.bbox[1] and
            obj_b.bbox[2] <= obj_a.bbox[2] and obj_a.bbox[3] <= obj_b.bbox[3])


# ══════════════════════════════════════════════════════════════════════
# 3. OBJECT-LEVEL SOLVERS
# ══════════════════════════════════════════════════════════════════════

def try_keep_by_property(task):
    """Keep objects matching a learned property, remove others."""
    pairs = task['train']
    for prop in ['largest', 'smallest', 'most_common_color',
                 'least_common_color', 'rectangular', 'non_rectangular',
                 'touches_border', 'not_touches_border',
                 'tallest', 'widest', 'squarest', 'densest']:
        ok = True
        for p in pairs:
            gi = np.array(p['input']); go = np.array(p['output'])
            if gi.shape != go.shape:
                ok = False; break
            objs, bg = extract_objects(gi)
            if len(objs) < 2:
                ok = False; break

            # Which objects survive in output?
            kept = []
            removed = []
            for o in objs:
                still_there = any(go[r, c] == o.color for r, c in o.cells)
                if still_there:
                    kept.append(o)
                else:
                    removed.append(o)

            if not kept or not removed:
                ok = False; break

            # Check if property distinguishes kept from removed
            if prop == 'largest':
                if not all(o.size >= max(r.size for r in removed) for o in kept):
                    ok = False; break
            elif prop == 'smallest':
                if not all(o.size <= min(r.size for r in removed) for o in kept):
                    ok = False; break
            elif prop == 'rectangular':
                if not all(o.is_rectangular for o in kept) or any(o.is_rectangular for o in removed):
                    ok = False; break
            elif prop == 'non_rectangular':
                if not all(not o.is_rectangular for o in kept) or any(not o.is_rectangular for o in removed):
                    ok = False; break
            elif prop == 'touches_border':
                if not all(o.touches_border for o in kept) or any(o.touches_border for o in removed):
                    ok = False; break
            elif prop == 'not_touches_border':
                if not all(not o.touches_border for o in kept) or any(not o.touches_border for o in removed):
                    ok = False; break
            elif prop == 'most_common_color':
                cc = Counter(o.color for o in objs)
                mc = cc.most_common(1)[0][0]
                if not all(o.color == mc for o in kept):
                    ok = False; break
            elif prop == 'least_common_color':
                cc = Counter(o.color for o in objs)
                lc = cc.most_common()[-1][0]
                if not all(o.color == lc for o in kept):
                    ok = False; break
            elif prop == 'tallest':
                max_h = max(o.height for o in objs)
                if not all(o.height == max_h for o in kept):
                    ok = False; break
            elif prop == 'widest':
                max_w = max(o.width for o in objs)
                if not all(o.width == max_w for o in kept):
                    ok = False; break
            elif prop == 'squarest':
                if not all(o.aspect > 0.7 and o.aspect < 1.4 for o in kept):
                    ok = False; break
            elif prop == 'densest':
                max_d = max(o.density for o in objs)
                if not all(o.density >= max_d - 0.01 for o in kept):
                    ok = False; break

            # Verify output matches
            test_out = np.full_like(gi, bg)
            for o in kept:
                for r, c in o.cells:
                    test_out[r, c] = o.color
            if not np.array_equal(test_out, go):
                ok = False; break

        if not ok:
            continue

        # Apply to test
        def mk(prop_name=prop):
            def apply(task_inner):
                guesses = []
                for tc in task_inner['test']:
                    gi = np.array(tc['input'])
                    objs, bg = extract_objects(gi)
                    kept = []
                    for o in objs:
                        keep = False
                        if prop_name == 'largest':
                            keep = o.size == max(x.size for x in objs)
                        elif prop_name == 'smallest':
                            keep = o.size == min(x.size for x in objs)
                        elif prop_name == 'rectangular':
                            keep = o.is_rectangular
                        elif prop_name == 'non_rectangular':
                            keep = not o.is_rectangular
                        elif prop_name == 'touches_border':
                            keep = o.touches_border
                        elif prop_name == 'not_touches_border':
                            keep = not o.touches_border
                        elif prop_name == 'most_common_color':
                            cc = Counter(x.color for x in objs)
                            keep = o.color == cc.most_common(1)[0][0]
                        elif prop_name == 'least_common_color':
                            cc = Counter(x.color for x in objs)
                            keep = o.color == cc.most_common()[-1][0]
                        elif prop_name == 'tallest':
                            keep = o.height == max(x.height for x in objs)
                        elif prop_name == 'widest':
                            keep = o.width == max(x.width for x in objs)
                        elif prop_name == 'squarest':
                            keep = o.aspect > 0.7 and o.aspect < 1.4
                        elif prop_name == 'densest':
                            keep = o.density >= max(x.density for x in objs) - 0.01
                        if keep:
                            kept.append(o)
                    out = np.full_like(gi, bg)
                    for o in kept:
                        for r, c in o.cells:
                            out[r, c] = o.color
                    guesses.append([out.tolist()])
                return guesses
            return apply
        solver = mk()
        result = solver(task)
        if score_task(task, result):
            return result, f"OG:keep_{prop}"

    return None, None


def try_recolor_by_relation(task):
    """Recolor objects based on their relationships to other objects."""
    pairs = task['train']
    for rel_name, rel_fn in [('touching', touches), ('aligned_x', aligned_x),
                               ('aligned_y', aligned_y), ('same_shape', same_shape)]:
        ok = True
        color_rule = {}  # (has_relation, original_color) → new_color

        for p in pairs:
            gi = np.array(p['input']); go = np.array(p['output'])
            if gi.shape != go.shape:
                ok = False; break
            objs, bg = extract_objects(gi)
            if len(objs) < 2:
                ok = False; break

            for o in objs:
                # Check relation with any other object
                has_rel = any(rel_fn(o, other) for other in objs if other != o)
                r, c = o.cells[0]
                new_color = int(go[r, c])
                key = (has_rel, o.color)
                if key in color_rule and color_rule[key] != new_color:
                    ok = False; break
                color_rule[key] = new_color
            if not ok:
                break

        if not ok or not color_rule:
            continue
        # Check non-trivial
        if all(k[1] == v for k, v in color_rule.items()):
            continue

        def mk(rule=color_rule, rel=rel_fn):
            def apply(task_inner):
                guesses = []
                for tc in task_inner['test']:
                    gi = np.array(tc['input'])
                    objs, bg = extract_objects(gi)
                    out = gi.copy()
                    for o in objs:
                        has_r = any(rel(o, other) for other in objs if other != o)
                        key = (has_r, o.color)
                        if key in rule:
                            for r, c in o.cells:
                                out[r, c] = rule[key]
                    guesses.append([out.tolist()])
                return guesses
            return apply
        solver = mk()
        result = solver(task)
        if score_task(task, result):
            return result, f"OG:recolor_{rel_name}"

    return None, None


def try_move_object_to_target(task):
    """Move one object to align with another (e.g., move red next to blue)."""
    pairs = task['train']
    gi = np.array(pairs[0]['input']); go = np.array(pairs[0]['output'])
    if gi.shape != go.shape:
        return None, None

    objs_in, bg = extract_objects(gi)
    objs_out, _ = extract_objects(go)
    if len(objs_in) < 2 or len(objs_out) < 2:
        return None, None

    # Find which object moved
    for oi in objs_in:
        # Check if this object's cells are different in output
        moved = not all(go[r, c] == oi.color for r, c in oi.cells)
        if not moved:
            continue

        # Find where it went in output
        for oo in objs_out:
            if oo.color == oi.color and oo.size == oi.size and oo.shape_hash == oi.shape_hash:
                # Same shape, different position = movement
                dr = int(round(oo.center[0] - oi.center[0]))
                dc = int(round(oo.center[1] - oi.center[1]))
                if dr == 0 and dc == 0:
                    continue

                # Find the anchor object (what did it move towards?)
                for anchor in objs_in:
                    if anchor == oi:
                        continue
                    # Check if movement is towards anchor
                    target_r = anchor.center[0]
                    target_c = anchor.center[1]
                    moved_closer = (abs(oo.center[0] - target_r) < abs(oi.center[0] - target_r) or
                                     abs(oo.center[1] - target_c) < abs(oi.center[1] - target_c))
                    if moved_closer:
                        # Learn: move object of color X towards object of color Y
                        mover_color = oi.color
                        anchor_color = anchor.color

                        def mk(mc=mover_color, ac=anchor_color, ddr=dr, ddc=dc):
                            def apply(task_inner):
                                guesses = []
                                for tc in task_inner['test']:
                                    gi2 = np.array(tc['input'])
                                    objs2, bg2 = extract_objects(gi2)
                                    out = gi2.copy()
                                    movers = [o for o in objs2 if o.color == mc]
                                    anchors = [o for o in objs2 if o.color == ac]
                                    if movers and anchors:
                                        m = movers[0]
                                        # Clear old position
                                        for r, c in m.cells:
                                            out[r, c] = bg2
                                        # Place at new position
                                        for r, c in m.cells:
                                            nr, nc = r + ddr, c + ddc
                                            if 0 <= nr < gi2.shape[0] and 0 <= nc < gi2.shape[1]:
                                                out[nr, nc] = mc
                                    guesses.append([out.tolist()])
                                return guesses
                            return apply
                        solver = mk()
                        result = solver(task)
                        if score_task(task, result):
                            return result, f"OG:move_{mover_color}_to_{anchor_color}"

    return None, None


def try_extract_by_shape(task):
    """Extract object matching a specific shape template."""
    pairs = task['train']
    gi = np.array(pairs[0]['input']); go = np.array(pairs[0]['output'])

    objs, bg = extract_objects(gi)
    if len(objs) < 2:
        return None, None

    # Check if output matches one object's crop
    for prop in ['unique_shape', 'unique_color', 'most_dense', 'least_dense']:
        match_obj = None
        if prop == 'unique_shape':
            shape_counts = Counter(o.shape_hash for o in objs)
            unique_shapes = [sh for sh, n in shape_counts.items() if n == 1]
            if unique_shapes:
                match_obj = [o for o in objs if o.shape_hash == unique_shapes[0]][0]
        elif prop == 'unique_color':
            color_counts = Counter(o.color for o in objs)
            unique_colors = [c for c, n in color_counts.items() if n == 1]
            if unique_colors:
                match_obj = [o for o in objs if o.color == unique_colors[0]][0]
        elif prop == 'most_dense':
            match_obj = max(objs, key=lambda o: o.density)
        elif prop == 'least_dense':
            match_obj = min(objs, key=lambda o: o.density)

        if match_obj is None:
            continue

        crop = match_obj.crop(gi)
        if crop.shape == go.shape and np.array_equal(crop, go):
            def mk(p=prop):
                def apply(task_inner):
                    guesses = []
                    for tc in task_inner['test']:
                        gi2 = np.array(tc['input'])
                        objs2, _ = extract_objects(gi2)
                        if not objs2:
                            guesses.append([gi2.tolist()])
                            continue
                        target = None
                        if p == 'unique_shape':
                            sc = Counter(o.shape_hash for o in objs2)
                            us = [sh for sh, n in sc.items() if n == 1]
                            if us:
                                target = [o for o in objs2 if o.shape_hash == us[0]][0]
                        elif p == 'unique_color':
                            cc = Counter(o.color for o in objs2)
                            uc = [c for c, n in cc.items() if n == 1]
                            if uc:
                                target = [o for o in objs2 if o.color == uc[0]][0]
                        elif p == 'most_dense':
                            target = max(objs2, key=lambda o: o.density)
                        elif p == 'least_dense':
                            target = min(objs2, key=lambda o: o.density)
                        if target:
                            guesses.append([target.crop(gi2).tolist()])
                        else:
                            guesses.append([gi2.tolist()])
                    return guesses
                return apply
            solver = mk()
            result = solver(task)
            if score_task(task, result):
                return result, f"OG:extract_{prop}"

    return None, None


def try_stamp_pattern(task):
    """Find a small pattern object, stamp it at each marker position."""
    pairs = task['train']
    gi = np.array(pairs[0]['input']); go = np.array(pairs[0]['output'])
    if gi.shape != go.shape:
        return None, None

    objs, bg = extract_objects(gi)
    if len(objs) < 2:
        return None, None

    # Try each object as the "stamp"
    objs_sorted = sorted(objs, key=lambda o: o.size)
    for stamp in objs_sorted[:3]:  # try smallest objects as stamps
        stamp_crop = stamp.crop(gi)
        sh, sw = stamp_crop.shape

        # Marker positions = centers of other objects
        markers = [(o.bbox[0], o.bbox[2]) for o in objs if o != stamp]

        test_out = gi.copy()
        for mr, mc in markers:
            for r in range(sh):
                for c in range(sw):
                    if stamp_crop[r, c] != bg:
                        nr, nc = mr + r, mc + c
                        if 0 <= nr < gi.shape[0] and 0 <= nc < gi.shape[1]:
                            test_out[nr, nc] = stamp_crop[r, c]

        if np.array_equal(test_out, go):
            # Verify on all pairs
            all_ok = True
            for p in pairs[1:]:
                gi2 = np.array(p['input']); go2 = np.array(p['output'])
                objs2, bg2 = extract_objects(gi2)
                objs2_s = sorted(objs2, key=lambda o: o.size)
                if not objs2_s:
                    all_ok = False; break
                stamp2 = objs2_s[0]
                sc = stamp2.crop(gi2)
                markers2 = [(o.bbox[0], o.bbox[2]) for o in objs2 if o != stamp2]
                t = gi2.copy()
                for mr, mc in markers2:
                    for r in range(sc.shape[0]):
                        for c in range(sc.shape[1]):
                            if sc[r, c] != bg2:
                                nr, nc = mr + r, mc + c
                                if 0 <= nr < gi2.shape[0] and 0 <= nc < gi2.shape[1]:
                                    t[nr, nc] = sc[r, c]
                if not np.array_equal(t, go2):
                    all_ok = False; break
            if not all_ok:
                continue

            def apply(task_inner):
                guesses = []
                for tc in task_inner['test']:
                    gi2 = np.array(tc['input'])
                    objs2, bg2 = extract_objects(gi2)
                    objs2_s = sorted(objs2, key=lambda o: o.size)
                    if not objs2_s:
                        guesses.append([gi2.tolist()]); continue
                    stamp2 = objs2_s[0]
                    sc = stamp2.crop(gi2)
                    markers2 = [(o.bbox[0], o.bbox[2]) for o in objs2 if o != stamp2]
                    t = gi2.copy()
                    for mr, mc in markers2:
                        for r in range(sc.shape[0]):
                            for c in range(sc.shape[1]):
                                if sc[r, c] != bg2:
                                    nr, nc = mr + r, mc + c
                                    if 0 <= nr < gi2.shape[0] and 0 <= nc < gi2.shape[1]:
                                        t[nr, nc] = sc[r, c]
                    guesses.append([t.tolist()])
                return guesses
            result = apply(task)
            if score_task(task, result):
                return result, "OG:stamp_pattern"

    return None, None


# ══════════════════════════════════════════════════════════════════════
# 4. UNIFIED OBJECT GRAPH SOLVER
# ══════════════════════════════════════════════════════════════════════

def try_recolor_by_count(task):
    """Recolor each object with a color = its size or count of same-color objects."""
    pairs = task['train']
    for rule_type in ['size_mod', 'color_count', 'obj_index']:
        ok = True
        for p in pairs:
            gi = np.array(p['input']); go = np.array(p['output'])
            if gi.shape != go.shape: ok = False; break
            objs, bg = extract_objects(gi)
            if len(objs) < 2: ok = False; break
            for o in objs:
                r, c = o.cells[0]; new_c = int(go[r, c])
                if rule_type == 'size_mod':
                    expected = (o.size % 9) + 1
                elif rule_type == 'color_count':
                    expected = sum(1 for x in objs if x.color == o.color)
                elif rule_type == 'obj_index':
                    sorted_objs = sorted(objs, key=lambda x: (x.bbox[0], x.bbox[2]))
                    idx = next(i for i, x in enumerate(sorted_objs) if x.id == o.id)
                    expected = (idx % 9) + 1
                if new_c != expected: ok = False; break
            if not ok: break
        if not ok: continue
        def mk(rt=rule_type):
            def apply(task_inner):
                guesses = []
                for tc in task_inner['test']:
                    gi = np.array(tc['input']); objs, bg = extract_objects(gi)
                    out = np.full_like(gi, bg)
                    for o in objs:
                        if rt == 'size_mod': nc = (o.size % 9) + 1
                        elif rt == 'color_count': nc = sum(1 for x in objs if x.color == o.color)
                        elif rt == 'obj_index':
                            so = sorted(objs, key=lambda x: (x.bbox[0], x.bbox[2]))
                            idx = next(i for i, x in enumerate(so) if x.id == o.id)
                            nc = (idx % 9) + 1
                        for r, c in o.cells: out[r, c] = nc
                    guesses.append([out.tolist()])
                return guesses
            return apply
        solver = mk()
        result = solver(task)
        if score_task(task, result):
            return result, f"OG:recolor_{rule_type}"
    return None, None


def try_sort_objects(task):
    """Sort objects by property and rearrange in grid."""
    pairs = task['train']
    gi = np.array(pairs[0]['input']); go = np.array(pairs[0]['output'])
    if gi.shape != go.shape: return None, None
    objs_in, bg = extract_objects(gi)
    objs_out, _ = extract_objects(go)
    if len(objs_in) < 2 or len(objs_in) != len(objs_out): return None, None

    # Check if objects are rearranged by some property
    for sort_key in ['size', 'color', 'height', 'width']:
        if sort_key == 'size': key_fn = lambda o: o.size
        elif sort_key == 'color': key_fn = lambda o: o.color
        elif sort_key == 'height': key_fn = lambda o: o.height
        elif sort_key == 'width': key_fn = lambda o: o.width

        for rev in [False, True]:
            sorted_in = sorted(objs_in, key=key_fn, reverse=rev)
            # Check if output positions match sorted order placed at original positions
            positions_out = sorted([(o.bbox[0], o.bbox[2]) for o in objs_out])
            positions_in = [(o.bbox[0], o.bbox[2]) for o in objs_in]
            positions_sorted = sorted(positions_in)

            # Try: sorted objects placed at sorted positions
            test = np.full_like(gi, bg)
            if len(sorted_in) != len(positions_sorted): continue
            for obj, (tr, tc) in zip(sorted_in, positions_sorted):
                for r, c in obj.cells:
                    dr, dc = r - obj.bbox[0], c - obj.bbox[2]
                    nr, nc = tr + dr, tc + dc
                    if 0 <= nr < gi.shape[0] and 0 <= nc < gi.shape[1]:
                        test[nr, nc] = obj.color

            if np.array_equal(test, go):
                # Verify on other pairs
                all_ok = True
                for p in pairs[1:]:
                    gi2 = np.array(p['input']); go2 = np.array(p['output'])
                    objs2, bg2 = extract_objects(gi2)
                    sorted2 = sorted(objs2, key=key_fn, reverse=rev)
                    pos2 = sorted([(o.bbox[0], o.bbox[2]) for o in objs2])
                    t2 = np.full_like(gi2, bg2)
                    for obj, (tr, tc) in zip(sorted2, pos2):
                        for r, c in obj.cells:
                            dr, dc = r - obj.bbox[0], c - obj.bbox[2]
                            nr, nc = tr + dr, tc + dc
                            if 0 <= nr < gi2.shape[0] and 0 <= nc < gi2.shape[1]:
                                t2[nr, nc] = obj.color
                    if not np.array_equal(t2, go2): all_ok = False; break
                if not all_ok: continue

                def mk(kf=key_fn, rv=rev):
                    def apply(task_inner):
                        guesses = []
                        for tc in task_inner['test']:
                            gi2 = np.array(tc['input']); objs2, bg2 = extract_objects(gi2)
                            s2 = sorted(objs2, key=kf, reverse=rv)
                            p2 = sorted([(o.bbox[0], o.bbox[2]) for o in objs2])
                            out = np.full_like(gi2, bg2)
                            for obj, (tr, tc2) in zip(s2, p2):
                                for r, c in obj.cells:
                                    dr, dc = r - obj.bbox[0], c - obj.bbox[2]
                                    nr, nc = tr + dr, tc2 + dc
                                    if 0 <= nr < gi2.shape[0] and 0 <= nc < gi2.shape[1]:
                                        out[nr, nc] = obj.color
                            guesses.append([out.tolist()])
                        return guesses
                    return apply
                solver = mk()
                result = solver(task)
                if score_task(task, result):
                    return result, f"OG:sort_{sort_key}{'_rev' if rev else ''}"

    return None, None


def try_copy_shape_to_markers(task):
    """Find a template shape, copy it to each single-pixel marker."""
    pairs = task['train']
    gi = np.array(pairs[0]['input']); go = np.array(pairs[0]['output'])
    if gi.shape != go.shape: return None, None
    objs, bg = extract_objects(gi)
    if len(objs) < 2: return None, None

    # Separate: large objects (templates) vs small (markers)
    size_thresh = 2
    templates = [o for o in objs if o.size > size_thresh]
    markers = [o for o in objs if o.size <= size_thresh]

    if not templates or not markers: return None, None

    for tmpl in templates:
        crop = tmpl.crop(gi)
        th, tw = crop.shape
        test = gi.copy()
        for m in markers:
            mr, mc = m.cells[0]
            for r in range(th):
                for c in range(tw):
                    if crop[r, c] != bg:
                        nr, nc = mr + r - th//2, mc + c - tw//2
                        if 0 <= nr < gi.shape[0] and 0 <= nc < gi.shape[1]:
                            test[nr, nc] = crop[r, c]
        if np.array_equal(test, go):
            # Verify
            all_ok = True
            for p in pairs[1:]:
                gi2 = np.array(p['input']); go2 = np.array(p['output'])
                objs2, bg2 = extract_objects(gi2)
                t2 = [o for o in objs2 if o.size > size_thresh]
                m2 = [o for o in objs2 if o.size <= size_thresh]
                if not t2: all_ok = False; break
                cr = t2[0].crop(gi2); ch, cw = cr.shape
                t = gi2.copy()
                for mk2 in m2:
                    mr, mc = mk2.cells[0]
                    for r in range(ch):
                        for c in range(cw):
                            if cr[r, c] != bg2:
                                nr, nc = mr + r - ch//2, mc + c - cw//2
                                if 0 <= nr < gi2.shape[0] and 0 <= nc < gi2.shape[1]:
                                    t[nr, nc] = cr[r, c]
                if not np.array_equal(t, go2): all_ok = False; break
            if not all_ok: continue

            def apply(task_inner):
                guesses = []
                for tc in task_inner['test']:
                    gi2 = np.array(tc['input']); objs2, bg2 = extract_objects(gi2)
                    t2 = [o for o in objs2 if o.size > size_thresh]
                    m2 = [o for o in objs2 if o.size <= size_thresh]
                    out = gi2.copy()
                    if t2:
                        cr = t2[0].crop(gi2); ch, cw = cr.shape
                        for mk2 in m2:
                            mr, mc = mk2.cells[0]
                            for r in range(ch):
                                for c in range(cw):
                                    if cr[r, c] != bg2:
                                        nr, nc = mr + r - ch//2, mc + c - cw//2
                                        if 0 <= nr < gi2.shape[0] and 0 <= nc < gi2.shape[1]:
                                            out[nr, nc] = cr[r, c]
                    guesses.append([out.tolist()])
                return guesses
            result = apply(task)
            if score_task(task, result):
                return result, "OG:copy_to_markers"

    return None, None


def try_fill_between_objects(task):
    """Fill the space between two objects with a color."""
    pairs = task['train']
    for p in pairs:
        gi = np.array(p['input']); go = np.array(p['output'])
        if gi.shape != go.shape: return None, None
    gi = np.array(pairs[0]['input']); go = np.array(pairs[0]['output'])
    objs, bg = extract_objects(gi)
    if len(objs) < 2: return None, None

    # Check: are new cells between two objects?
    diff = (gi != go)
    if not diff.any(): return None, None
    new_cells = list(zip(*np.where(diff & (gi == bg))))
    if not new_cells: return None, None

    fill_color = int(go[new_cells[0][0], new_cells[0][1]])

    # Find which two objects the fill connects
    for i, o1 in enumerate(objs):
        for j, o2 in enumerate(objs):
            if i >= j: continue
            # Check if fill is between o1 and o2 horizontally
            if abs(o1.center[0] - o2.center[0]) < 2:  # same row
                c_min = min(o1.bbox[3], o2.bbox[3])
                c_max = max(o1.bbox[2], o2.bbox[2])
                r_center = int(round((o1.center[0] + o2.center[0]) / 2))
                test = gi.copy()
                for c in range(min(o1.bbox[3], o2.bbox[3]) + 1, max(o1.bbox[2], o2.bbox[2])):
                    for r in range(min(o1.bbox[0], o2.bbox[0]), max(o1.bbox[1], o2.bbox[1]) + 1):
                        if test[r, c] == bg:
                            test[r, c] = fill_color
                if np.array_equal(test, go):
                    def apply(task_inner):
                        guesses = []
                        for tc in task_inner['test']:
                            gi2 = np.array(tc['input']); objs2, bg2 = extract_objects(gi2)
                            out = gi2.copy()
                            for ii, oo1 in enumerate(objs2):
                                for jj, oo2 in enumerate(objs2):
                                    if ii >= jj: continue
                                    if abs(oo1.center[0] - oo2.center[0]) < 2:
                                        for c in range(min(oo1.bbox[3], oo2.bbox[3]) + 1,
                                                         max(oo1.bbox[2], oo2.bbox[2])):
                                            for r in range(min(oo1.bbox[0], oo2.bbox[0]),
                                                             max(oo1.bbox[1], oo2.bbox[1]) + 1):
                                                if 0 <= r < gi2.shape[0] and 0 <= c < gi2.shape[1]:
                                                    if out[r, c] == bg2:
                                                        out[r, c] = fill_color
                            guesses.append([out.tolist()])
                        return guesses
                    result = apply(task)
                    if score_task(task, result):
                        return result, "OG:fill_between_h"

            # Check vertical
            if abs(o1.center[1] - o2.center[1]) < 2:
                test = gi.copy()
                for r in range(min(o1.bbox[1], o2.bbox[1]) + 1, max(o1.bbox[0], o2.bbox[0])):
                    for c in range(min(o1.bbox[2], o2.bbox[2]), max(o1.bbox[3], o2.bbox[3]) + 1):
                        if 0 <= r < gi.shape[0] and 0 <= c < gi.shape[1]:
                            if test[r, c] == bg:
                                test[r, c] = fill_color
                if np.array_equal(test, go):
                    def apply(task_inner):
                        guesses = []
                        for tc in task_inner['test']:
                            gi2 = np.array(tc['input']); objs2, bg2 = extract_objects(gi2)
                            out = gi2.copy()
                            for ii, oo1 in enumerate(objs2):
                                for jj, oo2 in enumerate(objs2):
                                    if ii >= jj: continue
                                    if abs(oo1.center[1] - oo2.center[1]) < 2:
                                        for r in range(min(oo1.bbox[1], oo2.bbox[1]) + 1,
                                                         max(oo1.bbox[0], oo2.bbox[0])):
                                            for c in range(min(oo1.bbox[2], oo2.bbox[2]),
                                                             max(oo1.bbox[3], oo2.bbox[3]) + 1):
                                                if 0 <= r < gi2.shape[0] and 0 <= c < gi2.shape[1]:
                                                    if out[r, c] == bg2:
                                                        out[r, c] = fill_color
                            guesses.append([out.tolist()])
                        return guesses
                    result = apply(task)
                    if score_task(task, result):
                        return result, "OG:fill_between_v"

    return None, None


def try_mirror_object_across_axis(task):
    """Mirror an object across a line defined by another object."""
    pairs = task['train']
    gi = np.array(pairs[0]['input']); go = np.array(pairs[0]['output'])
    if gi.shape != go.shape: return None, None
    objs, bg = extract_objects(gi)
    if len(objs) < 2: return None, None
    h, w = gi.shape

    # Check for horizontal/vertical line objects (dividers)
    for o in objs:
        is_h_line = (o.height == 1 and o.width >= w * 0.5)
        is_v_line = (o.width == 1 and o.height >= h * 0.5)
        if not (is_h_line or is_v_line): continue

        # Mirror all other objects across this line
        test = gi.copy()
        if is_h_line:
            axis_r = o.bbox[0]
            for other in objs:
                if other == o: continue
                for r, c in other.cells:
                    mr = 2 * axis_r - r
                    if 0 <= mr < h and test[mr, c] == bg:
                        test[mr, c] = other.color
        elif is_v_line:
            axis_c = o.bbox[2]
            for other in objs:
                if other == o: continue
                for r, c in other.cells:
                    mc = 2 * axis_c - c
                    if 0 <= mc < w and test[r, mc] == bg:
                        test[r, mc] = other.color

        if np.array_equal(test, go):
            axis_type = 'h' if is_h_line else 'v'
            axis_color = o.color
            def mk(at=axis_type, ac=axis_color):
                def apply(task_inner):
                    guesses = []
                    for tc in task_inner['test']:
                        gi2 = np.array(tc['input']); h2, w2 = gi2.shape
                        objs2, bg2 = extract_objects(gi2)
                        out = gi2.copy()
                        axis_objs = [o for o in objs2 if o.color == ac]
                        if not axis_objs: guesses.append([gi2.tolist()]); continue
                        ax = axis_objs[0]
                        for other in objs2:
                            if other == ax: continue
                            for r, c in other.cells:
                                if at == 'h':
                                    mr = 2 * ax.bbox[0] - r
                                    if 0 <= mr < h2 and out[mr, c] == bg2:
                                        out[mr, c] = other.color
                                else:
                                    mc = 2 * ax.bbox[2] - c
                                    if 0 <= mc < w2 and out[r, mc] == bg2:
                                        out[r, mc] = other.color
                        guesses.append([out.tolist()])
                    return guesses
                return apply
            solver = mk()
            result = solver(task)
            if score_task(task, result):
                return result, f"OG:mirror_across_{axis_type}_line"

    return None, None


def try_contained_recolor(task):
    """If object A is inside object B, recolor A based on B's color."""
    pairs = task['train']
    rule = {}  # container_color → contained_new_color
    for p in pairs:
        gi = np.array(p['input']); go = np.array(p['output'])
        if gi.shape != go.shape: return None, None
        objs, bg = extract_objects(gi)
        if len(objs) < 2: return None, None
        # Use bbox containment — O(N²) but with early exit
        for a in objs:
            for b in objs:
                if a.id == b.id: continue
                if (b.bbox[0] <= a.bbox[0] and a.bbox[1] <= b.bbox[1] and
                    b.bbox[2] <= a.bbox[2] and a.bbox[3] <= b.bbox[3] and b.size > a.size):
                    r, c = a.cells[0]
                    nc = int(go[r, c])
                    key = (a.color, b.color)
                    if key in rule and rule[key] != nc: return None, None
                    rule[key] = nc
    if not rule or all(k[0] == v for k, v in rule.items()): return None, None
    def apply(task_inner):
        guesses = []
        for tc in task_inner['test']:
            gi = np.array(tc['input']); objs, bg2 = extract_objects(gi)
            out = gi.copy()
            for a in objs:
                for b in objs:
                    if a.id == b.id: continue
                    if (b.bbox[0] <= a.bbox[0] and a.bbox[1] <= b.bbox[1] and
                        b.bbox[2] <= a.bbox[2] and a.bbox[3] <= b.bbox[3] and b.size > a.size):
                        key = (a.color, b.color)
                        if key in rule:
                            for r, c in a.cells: out[r, c] = rule[key]
                        break
            guesses.append([out.tolist()])
        return guesses
    result = apply(task)
    if score_task(task, result): return result, "OG:contained_recolor"
    return None, None


def try_extract_contained(task):
    """Extract the object that is contained inside another."""
    pairs = task['train']
    gi = np.array(pairs[0]['input']); go = np.array(pairs[0]['output'])
    objs, bg = extract_objects(gi)
    if len(objs) < 2: return None, None
    for a in objs:
        for b in objs:
            if a.id == b.id: continue
            if (b.bbox[0] <= a.bbox[0] and a.bbox[1] <= b.bbox[1] and
                b.bbox[2] <= a.bbox[2] and a.bbox[3] <= b.bbox[3] and b.size > a.size):
                crop = a.crop(gi)
                if crop.shape == go.shape and np.array_equal(crop, go):
                    def apply(task_inner):
                        guesses = []
                        for tc in task_inner['test']:
                            gi2 = np.array(tc['input']); objs2, _ = extract_objects(gi2)
                            for aa in objs2:
                                for bb in objs2:
                                    if aa.id == bb.id: continue
                                    if (bb.bbox[0]<=aa.bbox[0] and aa.bbox[1]<=bb.bbox[1] and
                                        bb.bbox[2]<=aa.bbox[2] and aa.bbox[3]<=bb.bbox[3] and bb.size>aa.size):
                                        guesses.append([aa.crop(gi2).tolist()])
                                        break
                                else: continue
                                break
                            else: guesses.append([gi2.tolist()])
                        return guesses
                    result = apply(task)
                    if score_task(task, result): return result, "OG:extract_contained"
    return None, None


def try_recolor_by_size_rank(task):
    """Recolor objects based on size rank (Leetcode: sort + index mapping)."""
    pairs = task['train']
    rule = {}  # rank → color
    for p in pairs:
        gi = np.array(p['input']); go = np.array(p['output'])
        if gi.shape != go.shape: return None, None
        objs, bg = extract_objects(gi)
        if len(objs) < 2: return None, None
        # Sort by size, assign rank
        sorted_sizes = sorted(set(o.size for o in objs))
        size_to_rank = {s: i for i, s in enumerate(sorted_sizes)}
        for o in objs:
            rank = size_to_rank[o.size]
            r, c = o.cells[0]; nc = int(go[r, c])
            if rank in rule and rule[rank] != nc: return None, None
            rule[rank] = nc
    if not rule: return None, None
    def apply(task_inner):
        guesses = []
        for tc in task_inner['test']:
            gi = np.array(tc['input']); objs, bg2 = extract_objects(gi)
            ss = sorted(set(o.size for o in objs))
            s2r = {s: i for i, s in enumerate(ss)}
            out = np.full_like(gi, bg2)
            for o in objs:
                nc = rule.get(s2r.get(o.size, -1), o.color)
                for r, c in o.cells: out[r, c] = nc
            guesses.append([out.tolist()])
        return guesses
    result = apply(task)
    if score_task(task, result): return result, "OG:recolor_size_rank"
    return None, None


def try_keep_shared_property(task):
    """Keep only objects that share a property with at least one other.
    Leetcode trick: group by property, keep groups with count > 1."""
    pairs = task['train']
    for prop in ['color', 'size', 'shape', 'height', 'width']:
        ok = True
        for p in pairs:
            gi = np.array(p['input']); go = np.array(p['output'])
            if gi.shape != go.shape: ok = False; break
            objs, bg = extract_objects(gi)
            if len(objs) < 2: ok = False; break
            # Group by property
            if prop == 'color': groups = Counter(o.color for o in objs)
            elif prop == 'size': groups = Counter(o.size for o in objs)
            elif prop == 'shape': groups = Counter(o.shape_hash for o in objs)
            elif prop == 'height': groups = Counter(o.height for o in objs)
            elif prop == 'width': groups = Counter(o.width for o in objs)
            shared_keys = {k for k, v in groups.items() if v > 1}
            # Keep objects with shared property
            test = np.full_like(gi, bg)
            for o in objs:
                val = getattr(o, prop) if prop in ['color','size','height','width'] else o.shape_hash
                if val in shared_keys:
                    for r, c in o.cells: test[r, c] = o.color
            if not np.array_equal(test, go): ok = False; break
        if not ok: continue
        def mk(pp=prop):
            def apply(task_inner):
                guesses = []
                for tc in task_inner['test']:
                    gi = np.array(tc['input']); objs, bg2 = extract_objects(gi)
                    if pp == 'shape': groups = Counter(o.shape_hash for o in objs)
                    else: groups = Counter(getattr(o, pp) for o in objs)
                    shared = {k for k, v in groups.items() if v > 1}
                    out = np.full_like(gi, bg2)
                    for o in objs:
                        val = o.shape_hash if pp == 'shape' else getattr(o, pp)
                        if val in shared:
                            for r, c in o.cells: out[r, c] = o.color
                    guesses.append([out.tolist()])
                return guesses
            return apply
        solver = mk()
        result = solver(task)
        if score_task(task, result): return result, f"OG:keep_shared_{prop}"
    return None, None


def try_remove_shared_property(task):
    """Remove objects that share a property (inverse of keep_shared)."""
    pairs = task['train']
    for prop in ['color', 'size', 'shape']:
        ok = True
        for p in pairs:
            gi = np.array(p['input']); go = np.array(p['output'])
            if gi.shape != go.shape: ok = False; break
            objs, bg = extract_objects(gi)
            if len(objs) < 2: ok = False; break
            if prop == 'color': groups = Counter(o.color for o in objs)
            elif prop == 'size': groups = Counter(o.size for o in objs)
            else: groups = Counter(o.shape_hash for o in objs)
            unique_keys = {k for k, v in groups.items() if v == 1}
            test = np.full_like(gi, bg)
            for o in objs:
                val = o.shape_hash if prop == 'shape' else getattr(o, prop)
                if val in unique_keys:
                    for r, c in o.cells: test[r, c] = o.color
            if not np.array_equal(test, go): ok = False; break
        if not ok: continue
        def mk(pp=prop):
            def apply(task_inner):
                guesses = []
                for tc in task_inner['test']:
                    gi = np.array(tc['input']); objs, bg2 = extract_objects(gi)
                    if pp == 'shape': groups = Counter(o.shape_hash for o in objs)
                    else: groups = Counter(getattr(o, pp) for o in objs)
                    unique = {k for k, v in groups.items() if v == 1}
                    out = np.full_like(gi, bg2)
                    for o in objs:
                        val = o.shape_hash if pp == 'shape' else getattr(o, pp)
                        if val in unique:
                            for r, c in o.cells: out[r, c] = o.color
                    guesses.append([out.tolist()])
                return guesses
            return apply
        solver = mk()
        result = solver(task)
        if score_task(task, result): return result, f"OG:keep_unique_{prop}"
    return None, None


def try_map_color_to_size(task):
    """Output color = f(object_size). Leetcode: hash map lookup."""
    pairs = task['train']
    size_to_color = {}
    for p in pairs:
        gi = np.array(p['input']); go = np.array(p['output'])
        if gi.shape != go.shape: return None, None
        objs, bg = extract_objects(gi)
        for o in objs:
            r, c = o.cells[0]; nc = int(go[r, c])
            if o.size in size_to_color and size_to_color[o.size] != nc: return None, None
            size_to_color[o.size] = nc
    if not size_to_color or all(v == 0 for v in size_to_color.values()): return None, None
    def apply(task_inner):
        guesses = []
        for tc in task_inner['test']:
            gi = np.array(tc['input']); objs, bg2 = extract_objects(gi)
            out = np.full_like(gi, bg2)
            for o in objs:
                nc = size_to_color.get(o.size, o.color)
                for r, c in o.cells: out[r, c] = nc
            guesses.append([out.tolist()])
        return guesses
    result = apply(task)
    if score_task(task, result): return result, "OG:color_from_size"
    return None, None


def try_map_color_to_neighbor_count(task):
    """Output color = number of neighboring objects. Leetcode: adjacency counting."""
    pairs = task['train']
    rule = {}  # n_neighbors → color
    for p in pairs:
        gi = np.array(p['input']); go = np.array(p['output'])
        if gi.shape != go.shape: return None, None
        objs, bg = extract_objects(gi)
        if len(objs) < 2: return None, None
        # Precompute adjacency with set intersection (Leetcode: cell set trick)
        cell_sets = {}
        expanded = {}
        for o in objs:
            cell_sets[o.id] = set(o.cells)
            exp = set()
            for r, c in o.cells:
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    exp.add((r+dr, c+dc))
            expanded[o.id] = exp - cell_sets[o.id]
        for o in objs:
            n_adj = sum(1 for other in objs if other.id != o.id
                        and expanded[o.id] & cell_sets[other.id])
            r, c = o.cells[0]; nc = int(go[r, c])
            if n_adj in rule and rule[n_adj] != nc: return None, None
            rule[n_adj] = nc
    if not rule: return None, None
    def apply(task_inner):
        guesses = []
        for tc in task_inner['test']:
            gi = np.array(tc['input']); objs, bg2 = extract_objects(gi)
            cs = {o.id: set(o.cells) for o in objs}
            exp = {}
            for o in objs:
                e = set()
                for r, c in o.cells:
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]: e.add((r+dr,c+dc))
                exp[o.id] = e - cs[o.id]
            out = np.full_like(gi, bg2)
            for o in objs:
                n = sum(1 for oo in objs if oo.id != o.id and exp[o.id] & cs[oo.id])
                nc = rule.get(n, o.color)
                for r, c in o.cells: out[r, c] = nc
            guesses.append([out.tolist()])
        return guesses
    result = apply(task)
    if score_task(task, result): return result, "OG:color_from_neighbors"
    return None, None


def try_grow_objects(task):
    """Grow each object by 1 pixel in all directions (dilation)."""
    pairs = task['train']
    for p in pairs:
        gi = np.array(p['input']); go = np.array(p['output'])
        if gi.shape != go.shape: return None, None
    gi = np.array(pairs[0]['input']); go = np.array(pairs[0]['output'])
    h, w = gi.shape; bg = int(np.argmax(np.bincount(gi.flatten())))
    test = gi.copy()
    for r in range(h):
        for c in range(w):
            if gi[r, c] != bg:
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < h and 0 <= nc < w and test[nr, nc] == bg:
                        test[nr, nc] = gi[r, c]
    if np.array_equal(test, go):
        def apply(task_inner):
            guesses = []
            for tc in task_inner['test']:
                gi2 = np.array(tc['input']); h2, w2 = gi2.shape
                bg2 = int(np.argmax(np.bincount(gi2.flatten())))
                out = gi2.copy()
                for r in range(h2):
                    for c in range(w2):
                        if gi2[r, c] != bg2:
                            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                                nr, nc = r+dr, c+dc
                                if 0 <= nr < h2 and 0 <= nc < w2 and out[nr, nc] == bg2:
                                    out[nr, nc] = gi2[r, c]
                guesses.append([out.tolist()])
            return guesses
        result = apply(task)
        if score_task(task, result): return result, "OG:grow_objects"
    return None, None


def try_shrink_objects(task):
    """Shrink objects by removing border pixels (erosion)."""
    pairs = task['train']
    for p in pairs:
        gi = np.array(p['input']); go = np.array(p['output'])
        if gi.shape != go.shape: return None, None
    gi = np.array(pairs[0]['input']); go = np.array(pairs[0]['output'])
    h, w = gi.shape; bg = int(np.argmax(np.bincount(gi.flatten())))
    test = gi.copy()
    for r in range(h):
        for c in range(w):
            if gi[r, c] != bg:
                has_bg = any(0<=r+dr<h and 0<=c+dc<w and gi[r+dr,c+dc]==bg
                             for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)])
                if has_bg or r==0 or r==h-1 or c==0 or c==w-1:
                    test[r, c] = bg
    if np.array_equal(test, go):
        def apply(task_inner):
            guesses = []
            for tc in task_inner['test']:
                gi2 = np.array(tc['input']); h2, w2 = gi2.shape
                bg2 = int(np.argmax(np.bincount(gi2.flatten())))
                out = gi2.copy()
                for r in range(h2):
                    for c in range(w2):
                        if gi2[r, c] != bg2:
                            has_bg = any(0<=r+dr<h2 and 0<=c+dc<w2 and gi2[r+dr,c+dc]==bg2
                                          for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)])
                            if has_bg or r==0 or r==h2-1 or c==0 or c==w2-1:
                                out[r, c] = bg2
                guesses.append([out.tolist()])
            return guesses
        result = apply(task)
        if score_task(task, result): return result, "OG:shrink_objects"
    return None, None


    return None, None


def try_output_from_object_count(task):
    """Output grid size or content determined by object count.
    Leetcode trick: hash map from count → output template."""
    pairs = task['train']
    # Check if output size correlates with object count
    count_to_output = {}
    for p in pairs:
        gi = np.array(p['input']); go = np.array(p['output'])
        objs, bg = extract_objects(gi)
        n = len(objs)
        key = n
        # Check: is output NxN?
        if go.shape[0] == n or go.shape[1] == n:
            if key in count_to_output and count_to_output[key] != go.shape:
                return None, None
            count_to_output[key] = go.shape
    if not count_to_output:
        return None, None

    # Try: output is NxN filled with most common non-bg color
    for p in pairs:
        gi = np.array(p['input']); go = np.array(p['output'])
        objs, bg = extract_objects(gi)
        n = len(objs)
        # Check if output is n×n or 1×n or n×1
        for shape_fn in [lambda nn: (nn,nn), lambda nn: (1,nn), lambda nn: (nn,1)]:
            expected_shape = shape_fn(n)
            if go.shape != expected_shape: continue
            # What fills the output?
            out_vals = [int(v) for v in go.flatten() if v != 0]
            if not out_vals: continue
            fill = Counter(out_vals).most_common(1)[0][0]
            def mk(sf=shape_fn, fc=fill):
                def apply(task_inner):
                    guesses = []
                    for tc in task_inner['test']:
                        gi2 = np.array(tc['input'])
                        objs2, _ = extract_objects(gi2)
                        nn = len(objs2)
                        s = sf(max(1, nn))
                        guesses.append([np.full(s, fc, dtype=int).tolist()])
                    return guesses
                return apply
            solver = mk()
            result = solver(task)
            if score_task(task, result):
                return result, "OG:output_from_count"
    return None, None


def try_partition_and_process(task):
    """Split grid by divider lines, process each partition.
    Leetcode: two-pointer / segment processing."""
    pairs = task['train']
    gi = np.array(pairs[0]['input']); go = np.array(pairs[0]['output'])
    h, w = gi.shape
    bg = int(np.argmax(np.bincount(gi.flatten())))

    # Find horizontal dividers (full rows of single non-bg color)
    h_divs = []
    for r in range(h):
        vals = set(gi[r].tolist())
        if len(vals) == 1 and list(vals)[0] != bg:
            h_divs.append(r)

    # Find vertical dividers
    v_divs = []
    for c in range(w):
        vals = set(gi[:, c].tolist())
        if len(vals) == 1 and list(vals)[0] != bg:
            v_divs.append(c)

    if not h_divs and not v_divs:
        return None, None

    # Get partitions
    h_bounds = [0] + h_divs + [h]
    v_bounds = [0] + v_divs + [w]

    partitions = []
    for i in range(len(h_bounds) - 1):
        for j in range(len(v_bounds) - 1):
            r1, r2 = h_bounds[i], h_bounds[i+1]
            c1, c2 = v_bounds[j], v_bounds[j+1]
            if r1 in h_divs: r1 += 1
            if c1 in v_divs: c1 += 1
            if r1 < r2 and c1 < c2:
                partitions.append(gi[r1:r2, c1:c2])

    if not partitions:
        return None, None

    # Check if output = one of the partitions
    for pi, part in enumerate(partitions):
        if part.shape == go.shape and np.array_equal(part, go):
            # Which partition? Try: most non-bg cells
            def apply(task_inner):
                guesses = []
                for tc in task_inner['test']:
                    gi2 = np.array(tc['input']); h2, w2 = gi2.shape
                    bg2 = int(np.argmax(np.bincount(gi2.flatten())))
                    hd = [r for r in range(h2) if len(set(gi2[r].tolist())) == 1 and gi2[r, 0] != bg2]
                    vd = [c for c in range(w2) if len(set(gi2[:, c].tolist())) == 1 and gi2[0, c] != bg2]
                    hb = [0] + hd + [h2]; vb = [0] + vd + [w2]
                    parts = []
                    for ii in range(len(hb)-1):
                        for jj in range(len(vb)-1):
                            r1, r2 = hb[ii], hb[ii+1]; c1, c2 = vb[jj], vb[jj+1]
                            if r1 in hd: r1 += 1
                            if c1 in vd: c1 += 1
                            if r1 < r2 and c1 < c2:
                                parts.append(gi2[r1:r2, c1:c2])
                    if parts:
                        best = max(parts, key=lambda p: np.sum(p != bg2))
                        guesses.append([best.tolist()])
                    else:
                        guesses.append([gi2.tolist()])
                return guesses
            result = apply(task)
            if score_task(task, result):
                return result, "OG:partition_max_content"

    # Check: output = XOR/AND of partitions
    if len(partitions) >= 2:
        shapes = set(p.shape for p in partitions)
        if len(shapes) == 1:
            sh = list(shapes)[0]
            if sh == go.shape:
                # Try OR (overlay)
                overlay = np.zeros(sh, dtype=int)
                for part in partitions:
                    mask = part != bg
                    overlay[mask] = part[mask]
                if np.array_equal(overlay, go):
                    def apply(task_inner):
                        guesses = []
                        for tc in task_inner['test']:
                            gi2 = np.array(tc['input']); h2, w2 = gi2.shape
                            bg2 = int(np.argmax(np.bincount(gi2.flatten())))
                            hd = [r for r in range(h2) if len(set(gi2[r].tolist())) == 1 and gi2[r,0] != bg2]
                            vd = [c for c in range(w2) if len(set(gi2[:,c].tolist())) == 1 and gi2[0,c] != bg2]
                            hb = [0]+hd+[h2]; vb = [0]+vd+[w2]
                            parts = []
                            for ii in range(len(hb)-1):
                                for jj in range(len(vb)-1):
                                    r1,r2 = hb[ii],hb[ii+1]; c1,c2 = vb[jj],vb[jj+1]
                                    if r1 in hd: r1+=1
                                    if c1 in vd: c1+=1
                                    if r1<r2 and c1<c2: parts.append(gi2[r1:r2,c1:c2])
                            if len(parts)>=2 and len(set(p.shape for p in parts))==1:
                                ov = np.zeros(parts[0].shape, dtype=int)
                                for pp in parts:
                                    m = pp != bg2; ov[m] = pp[m]
                                guesses.append([ov.tolist()])
                            else:
                                guesses.append([gi2.tolist()])
                        return guesses
                    result = apply(task)
                    if score_task(task, result):
                        return result, "OG:partition_overlay"

                # Try AND (intersection)
                inter = partitions[0].copy()
                for part in partitions[1:]:
                    inter = np.where((inter != bg) & (part != bg), inter, bg)
                if np.array_equal(inter, go):
                    def apply(task_inner):
                        guesses = []
                        for tc in task_inner['test']:
                            gi2 = np.array(tc['input']); h2, w2 = gi2.shape
                            bg2 = int(np.argmax(np.bincount(gi2.flatten())))
                            hd = [r for r in range(h2) if len(set(gi2[r].tolist()))==1 and gi2[r,0]!=bg2]
                            vd = [c for c in range(w2) if len(set(gi2[:,c].tolist()))==1 and gi2[0,c]!=bg2]
                            hb=[0]+hd+[h2]; vb=[0]+vd+[w2]
                            parts=[]
                            for ii in range(len(hb)-1):
                                for jj in range(len(vb)-1):
                                    r1,r2=hb[ii],hb[ii+1]; c1,c2=vb[jj],vb[jj+1]
                                    if r1 in hd: r1+=1
                                    if c1 in vd: c1+=1
                                    if r1<r2 and c1<c2: parts.append(gi2[r1:r2,c1:c2])
                            if len(parts)>=2 and len(set(p.shape for p in parts))==1:
                                it=parts[0].copy()
                                for pp in parts[1:]:
                                    it=np.where((it!=bg2)&(pp!=bg2),it,bg2)
                                guesses.append([it.tolist()])
                            else: guesses.append([gi2.tolist()])
                        return guesses
                    result = apply(task)
                    if score_task(task, result):
                        return result, "OG:partition_intersect"

    return None, None


def try_color_majority_vote(task):
    """Each cell becomes the most common color in its 3x3 neighborhood,
    but only considering non-bg neighbors."""
    pairs = task['train']
    for p in pairs:
        if np.array(p['input']).shape != np.array(p['output']).shape: return None, None
    gi = np.array(pairs[0]['input']); go = np.array(pairs[0]['output']); h, w = gi.shape
    bg = int(np.argmax(np.bincount(gi.flatten())))
    test = gi.copy()
    for r in range(h):
        for c in range(w):
            if gi[r, c] == bg: continue
            nz = []
            for dr in [-1,0,1]:
                for dc in [-1,0,1]:
                    nr, nc = r+dr, c+dc
                    if 0<=nr<h and 0<=nc<w and gi[nr,nc]!=bg:
                        nz.append(int(gi[nr,nc]))
            if nz:
                test[r,c] = Counter(nz).most_common(1)[0][0]
    if np.array_equal(test, go):
        def apply(task_inner):
            guesses = []
            for tc in task_inner['test']:
                gi2 = np.array(tc['input']); h2, w2 = gi2.shape
                bg2 = int(np.argmax(np.bincount(gi2.flatten())))
                out = gi2.copy()
                for r in range(h2):
                    for c in range(w2):
                        if gi2[r,c]==bg2: continue
                        nz=[]
                        for dr in [-1,0,1]:
                            for dc in [-1,0,1]:
                                nr,nc=r+dr,c+dc
                                if 0<=nr<h2 and 0<=nc<w2 and gi2[nr,nc]!=bg2:
                                    nz.append(int(gi2[nr,nc]))
                        if nz: out[r,c]=Counter(nz).most_common(1)[0][0]
                guesses.append([out.tolist()])
            return guesses
        result = apply(task)
        if score_task(task, result): return result, "OG:majority_vote_nz"
    return None, None


def try_recolor_by_position(task):
    """Recolor objects based on their position (top/bottom/left/right).
    Leetcode: sort by coordinate, assign color by rank."""
    pairs = task['train']
    for sort_dim in ['row', 'col']:
        rule = {}
        ok = True
        for p in pairs:
            gi = np.array(p['input']); go = np.array(p['output'])
            if gi.shape != go.shape: ok = False; break
            objs, bg = extract_objects(gi)
            if len(objs) < 2: ok = False; break
            if sort_dim == 'row':
                sorted_objs = sorted(objs, key=lambda o: o.center[0])
            else:
                sorted_objs = sorted(objs, key=lambda o: o.center[1])
            for rank, o in enumerate(sorted_objs):
                r, c = o.cells[0]; nc = int(go[r, c])
                if rank in rule and rule[rank] != nc: ok = False; break
                rule[rank] = nc
            if not ok: break
        if not ok or not rule: continue
        if all(v == 0 for v in rule.values()): continue
        def mk(sd=sort_dim, rl=rule):
            def apply(task_inner):
                guesses = []
                for tc in task_inner['test']:
                    gi = np.array(tc['input']); objs, bg2 = extract_objects(gi)
                    if sd == 'row': so = sorted(objs, key=lambda o: o.center[0])
                    else: so = sorted(objs, key=lambda o: o.center[1])
                    out = np.full_like(gi, bg2)
                    for rank, o in enumerate(so):
                        nc = rl.get(rank, o.color)
                        for r, c in o.cells: out[r, c] = nc
                    guesses.append([out.tolist()])
                return guesses
            return apply
        solver = mk()
        result = solver(task)
        if score_task(task, result): return result, f"OG:recolor_by_{sort_dim}_rank"
    return None, None


def try_extend_object_lines(task):
    """Extend each object's edges into lines until hitting another object or border.
    Common ARC pattern: draw lines from objects."""
    pairs = task['train']
    for p in pairs:
        if np.array(p['input']).shape != np.array(p['output']).shape: return None, None
    gi = np.array(pairs[0]['input']); go = np.array(pairs[0]['output']); h, w = gi.shape
    bg = int(np.argmax(np.bincount(gi.flatten())))
    
    for direction in ['all_4', 'h_only', 'v_only']:
        test = gi.copy()
        objs, _ = extract_objects(gi)
        for o in objs:
            for r, c in o.cells:
                dirs = []
                if direction in ['all_4', 'h_only']:
                    dirs.extend([(0, 1), (0, -1)])
                if direction in ['all_4', 'v_only']:
                    dirs.extend([(1, 0), (-1, 0)])
                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    while 0 <= nr < h and 0 <= nc < w and test[nr, nc] == bg:
                        test[nr, nc] = o.color
                        nr += dr; nc += dc
        if np.array_equal(test, go):
            def mk(d=direction):
                def apply(task_inner):
                    guesses = []
                    for tc in task_inner['test']:
                        gi2 = np.array(tc['input']); h2, w2 = gi2.shape
                        bg2 = int(np.argmax(np.bincount(gi2.flatten())))
                        out = gi2.copy()
                        objs2, _ = extract_objects(gi2)
                        for o in objs2:
                            for r, c in o.cells:
                                dirs = []
                                if d in ['all_4', 'h_only']: dirs.extend([(0,1),(0,-1)])
                                if d in ['all_4', 'v_only']: dirs.extend([(1,0),(-1,0)])
                                for dr, dc in dirs:
                                    nr, nc = r+dr, c+dc
                                    while 0<=nr<h2 and 0<=nc<w2 and out[nr,nc]==bg2:
                                        out[nr,nc] = o.color; nr+=dr; nc+=dc
                        guesses.append([out.tolist()])
                    return guesses
                return apply
            solver = mk()
            result = solver(task)
            if score_task(task, result): return result, f"OG:extend_lines_{direction}"
    return None, None


def try_fill_row_col_from_object(task):
    """Fill entire row or column with an object's color.
    If object at (r,c), fill row r or col c with its color."""
    pairs = task['train']
    for p in pairs:
        if np.array(p['input']).shape != np.array(p['output']).shape: return None, None
    gi = np.array(pairs[0]['input']); go = np.array(pairs[0]['output']); h, w = gi.shape
    bg = int(np.argmax(np.bincount(gi.flatten())))
    
    for mode in ['row', 'col', 'cross']:
        test = gi.copy()
        for r in range(h):
            for c in range(w):
                if gi[r, c] != bg:
                    color = int(gi[r, c])
                    if mode in ['row', 'cross']:
                        for cc in range(w):
                            if test[r, cc] == bg: test[r, cc] = color
                    if mode in ['col', 'cross']:
                        for rr in range(h):
                            if test[rr, c] == bg: test[rr, c] = color
        if np.array_equal(test, go):
            def mk(m=mode):
                def apply(task_inner):
                    guesses = []
                    for tc in task_inner['test']:
                        gi2 = np.array(tc['input']); h2, w2 = gi2.shape
                        bg2 = int(np.argmax(np.bincount(gi2.flatten())))
                        out = gi2.copy()
                        for r in range(h2):
                            for c in range(w2):
                                if gi2[r, c] != bg2:
                                    color = int(gi2[r, c])
                                    if m in ['row', 'cross']:
                                        for cc in range(w2):
                                            if out[r, cc] == bg2: out[r, cc] = color
                                    if m in ['col', 'cross']:
                                        for rr in range(h2):
                                            if out[rr, c] == bg2: out[rr, c] = color
                        guesses.append([out.tolist()])
                    return guesses
                return apply
            solver = mk()
            result = solver(task)
            if score_task(task, result): return result, f"OG:fill_{mode}_from_obj"
    return None, None


def try_connect_same_color(task):
    """Draw lines connecting objects of the same color."""
    pairs = task['train']
    for p in pairs:
        if np.array(p['input']).shape != np.array(p['output']).shape: return None, None
    gi = np.array(pairs[0]['input']); go = np.array(pairs[0]['output']); h, w = gi.shape
    bg = int(np.argmax(np.bincount(gi.flatten())))
    objs, _ = extract_objects(gi)
    
    # Group by color
    from collections import defaultdict
    color_groups = defaultdict(list)
    for o in objs:
        color_groups[o.color].append(o)
    
    test = gi.copy()
    for color, group in color_groups.items():
        if len(group) < 2: continue
        # Connect each pair with L-shaped lines (horizontal then vertical)
        for i in range(len(group) - 1):
            o1, o2 = group[i], group[i+1]
            r1, c1 = int(round(o1.center[0])), int(round(o1.center[1]))
            r2, c2 = int(round(o2.center[0])), int(round(o2.center[1]))
            # Horizontal line from (r1, c1) to (r1, c2)
            for c in range(min(c1, c2), max(c1, c2) + 1):
                if 0 <= c < w and test[r1, c] == bg: test[r1, c] = color
            # Vertical line from (r1, c2) to (r2, c2)
            for r in range(min(r1, r2), max(r1, r2) + 1):
                if 0 <= r < h and test[r, c2] == bg: test[r, c2] = color
    
    if np.array_equal(test, go):
        def apply(task_inner):
            guesses = []
            for tc in task_inner['test']:
                gi2 = np.array(tc['input']); h2, w2 = gi2.shape
                bg2 = int(np.argmax(np.bincount(gi2.flatten())))
                objs2, _ = extract_objects(gi2)
                cg = defaultdict(list)
                for o in objs2: cg[o.color].append(o)
                out = gi2.copy()
                for color, group in cg.items():
                    if len(group) < 2: continue
                    for i in range(len(group) - 1):
                        o1, o2 = group[i], group[i+1]
                        r1, c1 = int(round(o1.center[0])), int(round(o1.center[1]))
                        r2, c2 = int(round(o2.center[0])), int(round(o2.center[1]))
                        for c in range(min(c1,c2), max(c1,c2)+1):
                            if 0<=c<w2 and out[r1,c]==bg2: out[r1,c]=color
                        for r in range(min(r1,r2), max(r1,r2)+1):
                            if 0<=r<h2 and out[r,c2]==bg2: out[r,c2]=color
                guesses.append([out.tolist()])
            return guesses
        result = apply(task)
        if score_task(task, result): return result, "OG:connect_same_color"
    return None, None


def try_flood_from_objects(task):
    """BFS flood from each object, filling bg cells with object's color.
    Stops when hitting another object's territory."""
    pairs = task['train']
    for p in pairs:
        if np.array(p['input']).shape != np.array(p['output']).shape: return None, None
    gi = np.array(pairs[0]['input']); go = np.array(pairs[0]['output']); h, w = gi.shape
    bg = int(np.argmax(np.bincount(gi.flatten())))
    
    # BFS from all non-bg cells simultaneously (nearest-color fill = Voronoi)
    from collections import deque
    test = gi.copy()
    q = deque()
    for r in range(h):
        for c in range(w):
            if gi[r, c] != bg:
                q.append((r, c, int(gi[r, c])))
    visited = (gi != bg).copy()
    while q:
        r, c, color = q.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0<=nr<h and 0<=nc<w and not visited[nr, nc]:
                visited[nr, nc] = True
                test[nr, nc] = color
                q.append((nr, nc, color))
    
    if np.array_equal(test, go):
        def apply(task_inner):
            guesses = []
            for tc in task_inner['test']:
                gi2 = np.array(tc['input']); h2, w2 = gi2.shape
                bg2 = int(np.argmax(np.bincount(gi2.flatten())))
                out = gi2.copy()
                qq = deque()
                for r in range(h2):
                    for c in range(w2):
                        if gi2[r,c] != bg2: qq.append((r,c,int(gi2[r,c])))
                vis = (gi2 != bg2).copy()
                while qq:
                    r,c,col = qq.popleft()
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc = r+dr, c+dc
                        if 0<=nr<h2 and 0<=nc<w2 and not vis[nr,nc]:
                            vis[nr,nc]=True; out[nr,nc]=col; qq.append((nr,nc,col))
                guesses.append([out.tolist()])
            return guesses
        result = apply(task)
        if score_task(task, result): return result, "OG:flood_voronoi_bfs"
    return None, None


def try_extract_common_subgrid(task):
    """Output = the subgrid pattern that appears in ALL training inputs.
    Leetcode: intersection of sets."""
    pairs = task['train']
    gi = np.array(pairs[0]['input']); go = np.array(pairs[0]['output'])
    if go.shape[0] >= gi.shape[0] and go.shape[1] >= gi.shape[1]: return None, None
    oh, ow = go.shape
    # Slide window of output size over input, find matching regions
    h, w = gi.shape
    for r in range(h - oh + 1):
        for c in range(w - ow + 1):
            sub = gi[r:r+oh, c:c+ow]
            if np.array_equal(sub, go):
                # Check if this position works for all pairs
                ok = True
                for p in pairs[1:]:
                    gi2 = np.array(p['input']); go2 = np.array(p['output'])
                    if go2.shape != go.shape: ok = False; break
                    sub2 = gi2[r:r+oh, c:c+ow] if r+oh<=gi2.shape[0] and c+ow<=gi2.shape[1] else None
                    if sub2 is None or not np.array_equal(sub2, go2): ok = False; break
                if ok:
                    def mk(rr=r, cc=c, hh=oh, ww=ow):
                        def apply(t):
                            return [[np.array(tc['input'])[rr:rr+hh, cc:cc+ww].tolist()] for tc in t['test']]
                        return apply
                    result = mk()(task)
                    if score_task(task, result): return result, "OG:extract_fixed_region"
    return None, None


def try_tile_input(task):
    """Output = input tiled NxM times."""
    pairs = task['train']
    gi = np.array(pairs[0]['input']); go = np.array(pairs[0]['output'])
    ih, iw = gi.shape; oh, ow = go.shape
    if oh <= ih or ow <= iw: return None, None
    if oh % ih != 0 or ow % iw != 0: return None, None
    rh, rw = oh // ih, ow // iw
    tiled = np.tile(gi, (rh, rw))
    if not np.array_equal(tiled, go): return None, None
    # Verify
    for p in pairs[1:]:
        gi2 = np.array(p['input']); go2 = np.array(p['output'])
        if go2.shape[0] % gi2.shape[0] != 0 or go2.shape[1] % gi2.shape[1] != 0: return None, None
        rh2, rw2 = go2.shape[0]//gi2.shape[0], go2.shape[1]//gi2.shape[1]
        if rh2 != rh or rw2 != rw: return None, None
        if not np.array_equal(np.tile(gi2, (rh, rw)), go2): return None, None
    def apply(t):
        return [[np.tile(np.array(tc['input']), (rh, rw)).tolist()] for tc in t['test']]
    result = apply(task)
    if score_task(task, result): return result, f"OG:tile_{rh}x{rw}"
    return None, None


def try_upscale_pixels(task):
    """Each pixel becomes an NxN block."""
    pairs = task['train']
    gi = np.array(pairs[0]['input']); go = np.array(pairs[0]['output'])
    ih, iw = gi.shape; oh, ow = go.shape
    if oh <= ih: return None, None
    if oh % ih != 0 or ow % iw != 0: return None, None
    s = oh // ih
    if ow // iw != s: return None, None
    scaled = np.repeat(np.repeat(gi, s, axis=0), s, axis=1)
    if not np.array_equal(scaled, go): return None, None
    for p in pairs[1:]:
        gi2 = np.array(p['input']); go2 = np.array(p['output'])
        if not np.array_equal(np.repeat(np.repeat(gi2, s, axis=0), s, axis=1), go2): return None, None
    def apply(t):
        return [[np.repeat(np.repeat(np.array(tc['input']), s, axis=0), s, axis=1).tolist()] for tc in t['test']]
    result = apply(task)
    if score_task(task, result): return result, f"OG:upscale_{s}x"
    return None, None


def try_transpose(task):
    """Output = transpose of input."""
    pairs = task['train']
    gi = np.array(pairs[0]['input']); go = np.array(pairs[0]['output'])
    if not np.array_equal(gi.T, go): return None, None
    for p in pairs[1:]:
        if not np.array_equal(np.array(p['input']).T, np.array(p['output'])): return None, None
    def apply(t):
        return [[np.array(tc['input']).T.tolist()] for tc in t['test']]
    result = apply(task)
    if score_task(task, result): return result, "OG:transpose"
    return None, None


def try_remove_duplicate_rows_cols(task):
    """Remove duplicate rows or columns."""
    pairs = task['train']
    gi = np.array(pairs[0]['input']); go = np.array(pairs[0]['output'])
    # Dedup rows
    if go.shape[1] == gi.shape[1] and go.shape[0] < gi.shape[0]:
        seen = []; rows = []
        for r in range(gi.shape[0]):
            rt = tuple(gi[r].tolist())
            if rt not in seen: seen.append(rt); rows.append(gi[r])
        if len(rows) == go.shape[0] and np.array_equal(np.array(rows), go):
            for p in pairs[1:]:
                gi2 = np.array(p['input']); go2 = np.array(p['output'])
                s2 = []; r2 = []
                for r in range(gi2.shape[0]):
                    rt = tuple(gi2[r].tolist())
                    if rt not in s2: s2.append(rt); r2.append(gi2[r])
                if not np.array_equal(np.array(r2), go2): break
            else:
                def apply(t):
                    guesses = []
                    for tc in t['test']:
                        a = np.array(tc['input']); s=[]; rows=[]
                        for r in range(a.shape[0]):
                            rt=tuple(a[r].tolist())
                            if rt not in s: s.append(rt); rows.append(a[r])
                        guesses.append([np.array(rows).tolist()])
                    return guesses
                result = apply(task)
                if score_task(task, result): return result, "OG:dedup_rows"
    # Dedup cols
    if go.shape[0] == gi.shape[0] and go.shape[1] < gi.shape[1]:
        seen = []; cols = []
        for c in range(gi.shape[1]):
            ct = tuple(gi[:, c].tolist())
            if ct not in seen: seen.append(ct); cols.append(gi[:, c])
        if len(cols) == go.shape[1] and np.array_equal(np.column_stack(cols), go):
            for p in pairs[1:]:
                gi2 = np.array(p['input']); go2 = np.array(p['output'])
                s2=[]; c2=[]
                for c in range(gi2.shape[1]):
                    ct=tuple(gi2[:,c].tolist())
                    if ct not in s2: s2.append(ct); c2.append(gi2[:,c])
                if not np.array_equal(np.column_stack(c2), go2): break
            else:
                def apply(t):
                    guesses=[]
                    for tc in t['test']:
                        a=np.array(tc['input']); s=[]; cols=[]
                        for c in range(a.shape[1]):
                            ct=tuple(a[:,c].tolist())
                            if ct not in s: s.append(ct); cols.append(a[:,c])
                        guesses.append([np.column_stack(cols).tolist()])
                    return guesses
                result = apply(task)
                if score_task(task, result): return result, "OG:dedup_cols"
    return None, None


def try_color_swap(task):
    """Simple color swap: swap two colors throughout grid."""
    pairs = task['train']
    for p in pairs:
        if np.array(p['input']).shape != np.array(p['output']).shape: return None, None
    gi = np.array(pairs[0]['input']); go = np.array(pairs[0]['output'])
    # Find which colors swapped
    swaps = {}
    for r in range(gi.shape[0]):
        for c in range(gi.shape[1]):
            ic, oc = int(gi[r,c]), int(go[r,c])
            if ic != oc:
                if ic in swaps and swaps[ic] != oc: return None, None
                swaps[ic] = oc
    if not swaps: return None, None
    # Check reversibility (bijective swap)
    rev = {v: k for k, v in swaps.items()}
    for k, v in swaps.items():
        if v in swaps and swaps[v] != k: return None, None
    # Verify
    for p in pairs[1:]:
        gi2 = np.array(p['input']); go2 = np.array(p['output'])
        test = gi2.copy()
        for k, v in swaps.items(): test[gi2 == k] = v
        if not np.array_equal(test, go2): return None, None
    def apply(t):
        guesses = []
        for tc in t['test']:
            a = np.array(tc['input']).copy()
            orig = a.copy()
            for k, v in swaps.items(): a[orig == k] = v
            guesses.append([a.tolist()])
        return guesses
    result = apply(task)
    if score_task(task, result): return result, "OG:color_swap"
    return None, None


def try_outline_objects(task):
    """Replace objects with their outlines (hollow rectangles)."""
    pairs = task['train']
    for p in pairs:
        if np.array(p['input']).shape != np.array(p['output']).shape: return None, None
    gi = np.array(pairs[0]['input']); go = np.array(pairs[0]['output']); h, w = gi.shape
    bg = int(np.argmax(np.bincount(gi.flatten())))
    objs, _ = extract_objects(gi)
    if not objs: return None, None
    test = np.full_like(gi, bg)
    for o in objs:
        r1, r2, c1, c2 = o.bbox
        for r in range(r1, r2+1):
            for c in range(c1, c2+1):
                if r == r1 or r == r2 or c == c1 or c == c2:
                    test[r, c] = o.color
    if np.array_equal(test, go):
        for p in pairs[1:]:
            gi2 = np.array(p['input']); go2 = np.array(p['output'])
            objs2, bg2 = extract_objects(gi2)
            t2 = np.full_like(gi2, bg2)
            for o in objs2:
                r1,r2,c1,c2 = o.bbox
                for r in range(r1,r2+1):
                    for c in range(c1,c2+1):
                        if r==r1 or r==r2 or c==c1 or c==c2: t2[r,c]=o.color
            if not np.array_equal(t2, go2): break
        else:
            def apply(t):
                guesses = []
                for tc in t['test']:
                    gi2=np.array(tc['input']); objs2,bg2=extract_objects(gi2)
                    out=np.full_like(gi2,bg2)
                    for o in objs2:
                        r1,r2,c1,c2=o.bbox
                        for r in range(r1,r2+1):
                            for c in range(c1,c2+1):
                                if r==r1 or r==r2 or c==c1 or c==c2: out[r,c]=o.color
                    guesses.append([out.tolist()])
                return guesses
            result = apply(task)
            if score_task(task, result): return result, "OG:outline_objects"
    return None, None


def try_fill_bbox(task):
    """Fill each object's bounding box solidly with its color."""
    pairs = task['train']
    for p in pairs:
        if np.array(p['input']).shape != np.array(p['output']).shape: return None, None
    gi = np.array(pairs[0]['input']); go = np.array(pairs[0]['output'])
    bg = int(np.argmax(np.bincount(gi.flatten())))
    objs, _ = extract_objects(gi)
    if not objs: return None, None
    test = np.full_like(gi, bg)
    for o in objs:
        r1, r2, c1, c2 = o.bbox
        test[r1:r2+1, c1:c2+1] = o.color
    if np.array_equal(test, go):
        for p in pairs[1:]:
            gi2=np.array(p['input']); go2=np.array(p['output'])
            objs2,bg2=extract_objects(gi2)
            t2=np.full_like(gi2,bg2)
            for o in objs2: r1,r2,c1,c2=o.bbox; t2[r1:r2+1,c1:c2+1]=o.color
            if not np.array_equal(t2, go2): break
        else:
            def apply(t):
                guesses=[]
                for tc in t['test']:
                    gi2=np.array(tc['input']); objs2,bg2=extract_objects(gi2)
                    out=np.full_like(gi2,bg2)
                    for o in objs2: r1,r2,c1,c2=o.bbox; out[r1:r2+1,c1:c2+1]=o.color
                    guesses.append([out.tolist()])
                return guesses
            result = apply(task)
            if score_task(task, result): return result, "OG:fill_bbox"
    return None, None


def try_extract_largest_object(task):
    """Output = crop of the largest connected component."""
    pairs = task['train']
    gi = np.array(pairs[0]['input']); go = np.array(pairs[0]['output'])
    objs, bg = extract_objects(gi)
    if len(objs) < 2: return None, None
    largest = max(objs, key=lambda o: o.size)
    crop = largest.crop(gi)
    if crop.shape == go.shape and np.array_equal(crop, go):
        for p in pairs[1:]:
            gi2=np.array(p['input']); go2=np.array(p['output'])
            objs2,_=extract_objects(gi2)
            if not objs2: return None, None
            l2=max(objs2, key=lambda o:o.size)
            if not np.array_equal(l2.crop(gi2), go2): return None, None
        def apply(t):
            guesses=[]
            for tc in t['test']:
                gi2=np.array(tc['input']); objs2,_=extract_objects(gi2)
                if objs2:
                    l2=max(objs2, key=lambda o:o.size)
                    guesses.append([l2.crop(gi2).tolist()])
                else: guesses.append([gi2.tolist()])
            return guesses
        result = apply(task)
        if score_task(task, result): return result, "OG:extract_largest"
    return None, None


def try_extract_smallest_object(task):
    """Output = crop of the smallest connected component."""
    pairs = task['train']
    gi = np.array(pairs[0]['input']); go = np.array(pairs[0]['output'])
    objs, bg = extract_objects(gi)
    if len(objs) < 2: return None, None
    smallest = min(objs, key=lambda o: o.size)
    crop = smallest.crop(gi)
    if crop.shape == go.shape and np.array_equal(crop, go):
        for p in pairs[1:]:
            gi2=np.array(p['input']); go2=np.array(p['output'])
            objs2,_=extract_objects(gi2)
            if not objs2: return None, None
            s2=min(objs2, key=lambda o:o.size)
            if not np.array_equal(s2.crop(gi2), go2): return None, None
        def apply(t):
            guesses=[]
            for tc in t['test']:
                gi2=np.array(tc['input']); objs2,_=extract_objects(gi2)
                if objs2:
                    s2=min(objs2, key=lambda o:o.size)
                    guesses.append([s2.crop(gi2).tolist()])
                else: guesses.append([gi2.tolist()])
            return guesses
        result = apply(task)
        if score_task(task, result): return result, "OG:extract_smallest"
    return None, None


def try_majority_color_fill(task):
    """Fill all non-bg cells with the most common non-bg color."""
    pairs = task['train']
    for p in pairs:
        if np.array(p['input']).shape != np.array(p['output']).shape: return None, None
    gi = np.array(pairs[0]['input']); go = np.array(pairs[0]['output'])
    bg = int(np.argmax(np.bincount(gi.flatten())))
    nz = [int(v) for v in gi.flatten() if v != bg]
    if not nz: return None, None
    mc = Counter(nz).most_common(1)[0][0]
    test = gi.copy()
    test[test != bg] = mc
    if np.array_equal(test, go):
        for p in pairs[1:]:
            gi2=np.array(p['input']); go2=np.array(p['output'])
            bg2=int(np.argmax(np.bincount(gi2.flatten())))
            nz2=[int(v) for v in gi2.flatten() if v!=bg2]
            if not nz2: return None, None
            mc2=Counter(nz2).most_common(1)[0][0]
            t2=gi2.copy(); t2[t2!=bg2]=mc2
            if not np.array_equal(t2, go2): break
        else:
            def apply(t):
                guesses=[]
                for tc in t['test']:
                    a=np.array(tc['input'])
                    bg2=int(np.argmax(np.bincount(a.flatten())))
                    nz2=[int(v) for v in a.flatten() if v!=bg2]
                    mc2=Counter(nz2).most_common(1)[0][0] if nz2 else 0
                    out=a.copy(); out[out!=bg2]=mc2
                    guesses.append([out.tolist()])
                return guesses
            result = apply(task)
            if score_task(task, result): return result, "OG:majority_fill"
    return None, None


def try_fill_between_same_color_h(task):
    """Fill horizontal gaps between same-color objects."""
    pairs = task['train']
    for p in pairs:
        if np.array(p['input']).shape != np.array(p['output']).shape: return None, None
    gi = np.array(pairs[0]['input']); go = np.array(pairs[0]['output']); h, w = gi.shape
    bg = int(np.argmax(np.bincount(gi.flatten())))
    test = gi.copy()
    for r in range(h):
        row = gi[r]
        # Find spans of same color on this row
        for color in set(int(v) for v in row if v != bg):
            cols = [c for c in range(w) if row[c] == color]
            if len(cols) >= 2:
                for c in range(min(cols), max(cols) + 1):
                    if test[r, c] == bg: test[r, c] = color
    if np.array_equal(test, go):
        def apply(t):
            guesses = []
            for tc in t['test']:
                a = np.array(tc['input']); hh, ww = a.shape
                bg2 = int(np.argmax(np.bincount(a.flatten())))
                out = a.copy()
                for r in range(hh):
                    for color in set(int(v) for v in a[r] if v != bg2):
                        cols = [c for c in range(ww) if a[r, c] == color]
                        if len(cols) >= 2:
                            for c in range(min(cols), max(cols) + 1):
                                if out[r, c] == bg2: out[r, c] = color
                guesses.append([out.tolist()])
            return guesses
        result = apply(task)
        if score_task(task, result): return result, "OG:fill_between_h"
    return None, None


def try_fill_between_same_color_v(task):
    """Fill vertical gaps between same-color objects."""
    pairs = task['train']
    for p in pairs:
        if np.array(p['input']).shape != np.array(p['output']).shape: return None, None
    gi = np.array(pairs[0]['input']); go = np.array(pairs[0]['output']); h, w = gi.shape
    bg = int(np.argmax(np.bincount(gi.flatten())))
    test = gi.copy()
    for c in range(w):
        col = gi[:, c]
        for color in set(int(v) for v in col if v != bg):
            rows = [r for r in range(h) if col[r] == color]
            if len(rows) >= 2:
                for r in range(min(rows), max(rows) + 1):
                    if test[r, c] == bg: test[r, c] = color
    if np.array_equal(test, go):
        def apply(t):
            guesses = []
            for tc in t['test']:
                a = np.array(tc['input']); hh, ww = a.shape
                bg2 = int(np.argmax(np.bincount(a.flatten())))
                out = a.copy()
                for c in range(ww):
                    for color in set(int(v) for v in a[:, c] if v != bg2):
                        rows = [r for r in range(hh) if a[r, c] == color]
                        if len(rows) >= 2:
                            for r in range(min(rows), max(rows) + 1):
                                if out[r, c] == bg2: out[r, c] = color
                guesses.append([out.tolist()])
            return guesses
        result = apply(task)
        if score_task(task, result): return result, "OG:fill_between_v"
    return None, None


def try_fill_between_same_color_hv(task):
    """Fill both horizontal AND vertical gaps."""
    pairs = task['train']
    for p in pairs:
        if np.array(p['input']).shape != np.array(p['output']).shape: return None, None
    gi = np.array(pairs[0]['input']); go = np.array(pairs[0]['output']); h, w = gi.shape
    bg = int(np.argmax(np.bincount(gi.flatten())))
    test = gi.copy()
    # Horizontal
    for r in range(h):
        for color in set(int(v) for v in gi[r] if v != bg):
            cols = [c for c in range(w) if gi[r, c] == color]
            if len(cols) >= 2:
                for c in range(min(cols), max(cols) + 1):
                    if test[r, c] == bg: test[r, c] = color
    # Vertical
    for c in range(w):
        for color in set(int(v) for v in gi[:, c] if v != bg):
            rows = [r for r in range(h) if gi[r, c] == color]
            if len(rows) >= 2:
                for r in range(min(rows), max(rows) + 1):
                    if test[r, c] == bg: test[r, c] = color
    if np.array_equal(test, go):
        def apply(t):
            guesses = []
            for tc in t['test']:
                a = np.array(tc['input']); hh, ww = a.shape
                bg2 = int(np.argmax(np.bincount(a.flatten())))
                out = a.copy()
                for r in range(hh):
                    for color in set(int(v) for v in a[r] if v != bg2):
                        cols = [c for c in range(ww) if a[r, c] == color]
                        if len(cols) >= 2:
                            for c in range(min(cols), max(cols) + 1):
                                if out[r, c] == bg2: out[r, c] = color
                for c in range(ww):
                    for color in set(int(v) for v in a[:, c] if v != bg2):
                        rows = [r for r in range(hh) if a[r, c] == color]
                        if len(rows) >= 2:
                            for r in range(min(rows), max(rows) + 1):
                                if out[r, c] == bg2: out[r, c] = color
                guesses.append([out.tolist()])
            return guesses
        result = apply(task)
        if score_task(task, result): return result, "OG:fill_between_hv"
    return None, None


def try_count_colors_output(task):
    """Output is a tiny grid encoding the count of each color.
    e.g., output row = [color, count] or output = histogram."""
    pairs = task['train']
    gi = np.array(pairs[0]['input']); go = np.array(pairs[0]['output'])
    bg = int(np.argmax(np.bincount(gi.flatten())))
    
    # Try: output is 1×N where N = number of unique non-bg colors
    nz_colors = sorted(set(int(v) for v in gi.flatten() if v != bg))
    if go.shape == (1, len(nz_colors)) and list(go[0]) == nz_colors:
        ok = all(
            np.array(p['output']).shape == (1, len(set(int(v) for v in np.array(p['input']).flatten() if v != int(np.argmax(np.bincount(np.array(p['input']).flatten())))))) and
            list(np.array(p['output'])[0]) == sorted(set(int(v) for v in np.array(p['input']).flatten() if v != int(np.argmax(np.bincount(np.array(p['input']).flatten())))))
            for p in pairs[1:]
        )
        if ok:
            def apply(t):
                guesses = []
                for tc in t['test']:
                    a = np.array(tc['input'])
                    bg2 = int(np.argmax(np.bincount(a.flatten())))
                    colors = sorted(set(int(v) for v in a.flatten() if v != bg2))
                    guesses.append([np.array([colors], dtype=int).tolist()])
                return guesses
            result = apply(task)
            if score_task(task, result): return result, "OG:count_unique_colors"
    
    # Try: output is Nx1 listing colors sorted
    if go.shape == (len(nz_colors), 1) and [int(go[i,0]) for i in range(len(nz_colors))] == nz_colors:
        ok = True
        for p in pairs[1:]:
            gi2 = np.array(p['input']); go2 = np.array(p['output'])
            bg2 = int(np.argmax(np.bincount(gi2.flatten())))
            c2 = sorted(set(int(v) for v in gi2.flatten() if v != bg2))
            if go2.shape != (len(c2), 1) or [int(go2[i,0]) for i in range(len(c2))] != c2:
                ok = False; break
        if ok:
            def apply(t):
                guesses = []
                for tc in t['test']:
                    a = np.array(tc['input'])
                    bg2 = int(np.argmax(np.bincount(a.flatten())))
                    colors = sorted(set(int(v) for v in a.flatten() if v != bg2))
                    guesses.append([np.array([[c] for c in colors], dtype=int).tolist()])
                return guesses
            result = apply(task)
            if score_task(task, result): return result, "OG:list_colors_v"
    return None, None


def try_count_objects_output(task):
    """Output encodes the number of objects."""
    pairs = task['train']
    gi = np.array(pairs[0]['input']); go = np.array(pairs[0]['output'])
    objs, bg = extract_objects(gi)
    n = len(objs)
    
    # 1x1 output = object count
    if go.shape == (1, 1) and int(go[0, 0]) == n:
        ok = all(int(np.array(p['output'])[0,0]) == len(extract_objects(np.array(p['input']))[0]) for p in pairs[1:])
        if ok:
            def apply(t):
                return [[np.array([[len(extract_objects(np.array(tc['input']))[0])]], dtype=int).tolist()] for tc in t['test']]
            result = apply(task)
            if score_task(task, result): return result, "OG:count_objects_1x1"
    
    # NxN output where N = object count, filled with specific color
    if go.shape == (n, n):
        fill = int(go[0, 0])
        if np.all(go == fill):
            ok = True
            for p in pairs[1:]:
                gi2 = np.array(p['input']); go2 = np.array(p['output'])
                n2 = len(extract_objects(gi2)[0])
                if go2.shape != (n2, n2) or not np.all(go2 == fill): ok = False; break
            if ok:
                def mk(fc=fill):
                    def apply(t):
                        return [[np.full((len(extract_objects(np.array(tc['input']))[0]),
                                          len(extract_objects(np.array(tc['input']))[0])), fc, dtype=int).tolist()]
                                for tc in t['test']]
                    return apply
                result = mk()(task)
                if score_task(task, result): return result, "OG:count_objects_NxN"
    return None, None


def try_color_histogram_row(task):
    """Output = histogram: one row per color, width = count of that color."""
    pairs = task['train']
    gi = np.array(pairs[0]['input']); go = np.array(pairs[0]['output'])
    bg = int(np.argmax(np.bincount(gi.flatten())))
    
    # Count non-bg colors
    counts = Counter(int(v) for v in gi.flatten() if v != bg)
    if not counts: return None, None
    
    # Try: output rows = colors sorted, width = count
    sorted_colors = sorted(counts.keys())
    expected_rows = []
    for color in sorted_colors:
        expected_rows.append([color] * counts[color])
    
    # Pad to same width
    max_w = max(len(r) for r in expected_rows) if expected_rows else 0
    padded = [r + [bg] * (max_w - len(r)) for r in expected_rows]
    expected = np.array(padded, dtype=int) if padded else np.array([[]], dtype=int)
    
    if expected.shape == go.shape and np.array_equal(expected, go):
        ok = True
        for p in pairs[1:]:
            gi2 = np.array(p['input']); go2 = np.array(p['output'])
            bg2 = int(np.argmax(np.bincount(gi2.flatten())))
            c2 = Counter(int(v) for v in gi2.flatten() if v != bg2)
            sc2 = sorted(c2.keys())
            rows2 = [[c]*c2[c] for c in sc2]
            mw2 = max(len(r) for r in rows2) if rows2 else 0
            p2 = [r + [bg2]*(mw2-len(r)) for r in rows2]
            e2 = np.array(p2, dtype=int) if p2 else np.array([[]], dtype=int)
            if not np.array_equal(e2, go2): ok = False; break
        if ok:
            def apply(t):
                guesses = []
                for tc in t['test']:
                    a = np.array(tc['input'])
                    bg2 = int(np.argmax(np.bincount(a.flatten())))
                    c2 = Counter(int(v) for v in a.flatten() if v != bg2)
                    sc2 = sorted(c2.keys())
                    rows2 = [[c]*c2[c] for c in sc2]
                    mw2 = max(len(r) for r in rows2) if rows2 else 1
                    p2 = [r + [bg2]*(mw2-len(r)) for r in rows2]
                    guesses.append([np.array(p2, dtype=int).tolist() if p2 else [[0]]])
                return guesses
            result = apply(task)
            if score_task(task, result): return result, "OG:color_histogram"
    return None, None


def try_extract_non_bg_bbox(task):
    """Output = bounding box of all non-bg content (tighter crop than border_crop)."""
    pairs = task['train']
    gi = np.array(pairs[0]['input']); go = np.array(pairs[0]['output'])
    if go.shape[0] >= gi.shape[0] and go.shape[1] >= gi.shape[1]: return None, None
    bg = int(np.argmax(np.bincount(gi.flatten())))
    nz = np.argwhere(gi != bg)
    if len(nz) == 0: return None, None
    r1, c1 = nz.min(axis=0); r2, c2 = nz.max(axis=0)
    crop = gi[r1:r2+1, c1:c2+1]
    if crop.shape == go.shape and np.array_equal(crop, go):
        for p in pairs[1:]:
            gi2 = np.array(p['input']); go2 = np.array(p['output'])
            bg2 = int(np.argmax(np.bincount(gi2.flatten())))
            nz2 = np.argwhere(gi2 != bg2)
            if len(nz2) == 0: return None, None
            rr1, cc1 = nz2.min(axis=0); rr2, cc2 = nz2.max(axis=0)
            if not np.array_equal(gi2[rr1:rr2+1, cc1:cc2+1], go2): return None, None
        def apply(t):
            guesses = []
            for tc in t['test']:
                a = np.array(tc['input'])
                bg2 = int(np.argmax(np.bincount(a.flatten())))
                nz2 = np.argwhere(a != bg2)
                if len(nz2) == 0: guesses.append([a.tolist()]); continue
                rr1, cc1 = nz2.min(axis=0); rr2, cc2 = nz2.max(axis=0)
                guesses.append([a[rr1:rr2+1, cc1:cc2+1].tolist()])
            return guesses
        result = apply(task)
        if score_task(task, result): return result, "OG:crop_to_content"
    return None, None


def try_replace_color_with_pattern(task):
    """Replace each occurrence of color X with a small pattern from another object."""
    pairs = task['train']
    gi = np.array(pairs[0]['input']); go = np.array(pairs[0]['output'])
    if gi.shape != go.shape: return None, None
    objs, bg = extract_objects(gi)
    if len(objs) < 2: return None, None
    
    # Find the "template" object (smallest) and the "marker" color
    objs_sorted = sorted(objs, key=lambda o: o.size)
    template = objs_sorted[0]
    tcrop = template.crop(gi)
    th, tw = tcrop.shape
    
    # Find single-pixel markers of different color
    markers = [o for o in objs if o.size == 1 and o.color != template.color]
    if not markers: return None, None
    
    # Check: does replacing each marker with the template produce output?
    test = gi.copy()
    # Remove template from grid first
    for r, c in template.cells: test[r, c] = bg
    for m in markers:
        mr, mc = m.cells[0]
        for r in range(th):
            for c in range(tw):
                nr, nc = mr - th//2 + r, mc - tw//2 + c
                if 0 <= nr < gi.shape[0] and 0 <= nc < gi.shape[1]:
                    if tcrop[r, c] != bg:
                        test[nr, nc] = tcrop[r, c]
    
    if np.array_equal(test, go):
        def apply(t):
            guesses = []
            for tc in t['test']:
                a = np.array(tc['input'])
                objs2, bg2 = extract_objects(a)
                if len(objs2) < 2: guesses.append([a.tolist()]); continue
                os2 = sorted(objs2, key=lambda o: o.size)
                tmpl = os2[0]; tc2 = tmpl.crop(a); tth, ttw = tc2.shape
                marks = [o for o in objs2 if o.size == 1 and o.color != tmpl.color]
                out = a.copy()
                for r, c in tmpl.cells: out[r, c] = bg2
                for m in marks:
                    mr, mc = m.cells[0]
                    for r in range(tth):
                        for c in range(ttw):
                            nr, nc = mr - tth//2 + r, mc - ttw//2 + c
                            if 0 <= nr < a.shape[0] and 0 <= nc < a.shape[1]:
                                if tc2[r, c] != bg2: out[nr, nc] = tc2[r, c]
                guesses.append([out.tolist()])
            return guesses
        result = apply(task)
        if score_task(task, result): return result, "OG:stamp_at_markers"
    return None, None


def try_fill_nearest_object_color(task):
    """Fill each bg cell with the color of its nearest object (Manhattan)."""
    pairs = task['train']
    for p in pairs:
        if np.array(p['input']).shape != np.array(p['output']).shape: return None, None
    gi = np.array(pairs[0]['input']); go = np.array(pairs[0]['output']); h, w = gi.shape
    bg = int(np.argmax(np.bincount(gi.flatten())))
    objs, _ = extract_objects(gi)
    if len(objs) < 2: return None, None
    # Build cell list per object for distance calc
    test = gi.copy()
    for r in range(h):
        for c in range(w):
            if gi[r, c] == bg:
                best_d, best_c = float('inf'), bg
                for o in objs:
                    for cr, cc in o.cells:
                        d = abs(r-cr) + abs(c-cc)
                        if d < best_d:
                            best_d = d; best_c = o.color
                test[r, c] = best_c
    if np.array_equal(test, go):
        def apply(t):
            guesses = []
            for tc in t['test']:
                a = np.array(tc['input']); hh, ww = a.shape
                bg2 = int(np.argmax(np.bincount(a.flatten())))
                objs2, _ = extract_objects(a)
                out = a.copy()
                for r in range(hh):
                    for c in range(ww):
                        if a[r, c] == bg2:
                            bd, bc = float('inf'), bg2
                            for o in objs2:
                                for cr, cc in o.cells:
                                    d = abs(r-cr)+abs(c-cc)
                                    if d < bd: bd, bc = d, o.color
                            out[r, c] = bc
                guesses.append([out.tolist()])
            return guesses
        result = apply(task)
        if score_task(task, result): return result, "OG:fill_nearest"
    return None, None


def try_recolor_by_grid_pattern(task):
    """Recolor based on position: checkerboard, row/col parity, quadrant, distance ring."""
    pairs = task['train']
    for p in pairs:
        if np.array(p['input']).shape != np.array(p['output']).shape: return None, None
    for pattern in ['checker', 'row_par', 'col_par', 'dist_ring', 'quadrant']:
        rule = {}; ok = True
        for p in pairs:
            gi = np.array(p['input']); go = np.array(p['output']); h, w = gi.shape
            for r in range(h):
                for c in range(w):
                    ic = int(gi[r, c]); oc = int(go[r, c])
                    if pattern == 'checker': pv = (r+c) % 2
                    elif pattern == 'row_par': pv = r % 2
                    elif pattern == 'col_par': pv = c % 2
                    elif pattern == 'dist_ring': pv = min(r, h-1-r, c, w-1-c) % 3
                    elif pattern == 'quadrant': pv = (0 if r < h//2 else 2) + (0 if c < w//2 else 1)
                    key = (ic, pv)
                    if key in rule and rule[key] != oc: ok = False; break
                    rule[key] = oc
                if not ok: break
            if not ok: break
        if not ok or not rule: continue
        if all(k[0] == v for k, v in rule.items()): continue
        def mk(pt=pattern, rl=rule):
            def apply(t):
                guesses = []
                for tc in t['test']:
                    a = np.array(tc['input']); hh, ww = a.shape
                    out = a.copy()
                    for r in range(hh):
                        for c in range(ww):
                            if pt == 'checker': pv = (r+c)%2
                            elif pt == 'row_par': pv = r%2
                            elif pt == 'col_par': pv = c%2
                            elif pt == 'dist_ring': pv = min(r,hh-1-r,c,ww-1-c)%3
                            elif pt == 'quadrant': pv = (0 if r<hh//2 else 2)+(0 if c<ww//2 else 1)
                            key = (int(a[r,c]), pv)
                            if key in rl: out[r,c] = rl[key]
                    guesses.append([out.tolist()])
                return guesses
            return apply
        result = mk()(task)
        if score_task(task, result): return result, f"OG:pattern_{pattern}"
    return None, None


def try_color_complement(task):
    """Each color X becomes 9-X (complement in ARC's 0-9 space)."""
    pairs = task['train']
    for p in pairs:
        if np.array(p['input']).shape != np.array(p['output']).shape: return None, None
    gi = np.array(pairs[0]['input']); go = np.array(pairs[0]['output'])
    bg = int(np.argmax(np.bincount(gi.flatten())))
    test = gi.copy()
    for r in range(gi.shape[0]):
        for c in range(gi.shape[1]):
            if gi[r,c] != bg: test[r,c] = 9 - gi[r,c]
    if np.array_equal(test, go):
        for p in pairs[1:]:
            gi2 = np.array(p['input']); go2 = np.array(p['output'])
            bg2 = int(np.argmax(np.bincount(gi2.flatten())))
            t2 = gi2.copy()
            for r in range(gi2.shape[0]):
                for c in range(gi2.shape[1]):
                    if gi2[r,c] != bg2: t2[r,c] = 9 - gi2[r,c]
            if not np.array_equal(t2, go2): return None, None
        def apply(t):
            guesses = []
            for tc in t['test']:
                a = np.array(tc['input'])
                bg2 = int(np.argmax(np.bincount(a.flatten())))
                out = a.copy()
                out[out != bg2] = 9 - out[out != bg2]
                guesses.append([out.tolist()])
            return guesses
        result = apply(task)
        if score_task(task, result): return result, "OG:color_complement"
    return None, None


def try_color_offset(task):
    """Each non-bg color X becomes (X+N) mod 10 for some fixed N."""
    pairs = task['train']
    for p in pairs:
        if np.array(p['input']).shape != np.array(p['output']).shape: return None, None
    gi = np.array(pairs[0]['input']); go = np.array(pairs[0]['output'])
    bg = int(np.argmax(np.bincount(gi.flatten())))
    for offset in range(1, 9):
        test = gi.copy()
        for r in range(gi.shape[0]):
            for c in range(gi.shape[1]):
                if gi[r,c] != bg:
                    test[r,c] = (gi[r,c] + offset) % 10
                    if test[r,c] == 0: test[r,c] = 10 - offset  # avoid bg
        if np.array_equal(test, go):
            ok = True
            for p in pairs[1:]:
                gi2 = np.array(p['input']); go2 = np.array(p['output'])
                bg2 = int(np.argmax(np.bincount(gi2.flatten())))
                t2 = gi2.copy()
                for r in range(gi2.shape[0]):
                    for c in range(gi2.shape[1]):
                        if gi2[r,c] != bg2:
                            t2[r,c] = (gi2[r,c] + offset) % 10
                            if t2[r,c] == 0: t2[r,c] = 10 - offset
                if not np.array_equal(t2, go2): ok = False; break
            if ok:
                def mk(off=offset):
                    def apply(t):
                        guesses = []
                        for tc in t['test']:
                            a = np.array(tc['input'])
                            bg2 = int(np.argmax(np.bincount(a.flatten())))
                            out = a.copy()
                            for r in range(a.shape[0]):
                                for c in range(a.shape[1]):
                                    if a[r,c] != bg2:
                                        out[r,c] = (a[r,c] + off) % 10
                                        if out[r,c] == 0: out[r,c] = 10 - off
                            guesses.append([out.tolist()])
                        return guesses
                    return apply
                result = mk()(task)
                if score_task(task, result): return result, f"OG:color_offset_{offset}"
    return None, None


def try_conditional_fill(task):
    """Fill bg with dominant color, but preserve objects matching a property."""
    pairs = task['train']
    for p in pairs:
        if np.array(p['input']).shape != np.array(p['output']).shape: return None, None
    gi = np.array(pairs[0]['input']); go = np.array(pairs[0]['output'])
    bg = int(np.argmax(np.bincount(gi.flatten())))
    objs, _ = extract_objects(gi)
    if len(objs) < 2: return None, None
    # What color fills the bg?
    new_bg_cells = [(r,c) for r in range(gi.shape[0]) for c in range(gi.shape[1])
                     if gi[r,c] == bg and go[r,c] != bg]
    if not new_bg_cells: return None, None
    fill_color = int(go[new_bg_cells[0][0], new_bg_cells[0][1]])
    if not all(int(go[r,c]) == fill_color for r,c in new_bg_cells): return None, None
    # Which objects are preserved vs overwritten?
    preserved = [o for o in objs if all(go[r,c] == o.color for r,c in o.cells)]
    overwritten = [o for o in objs if not all(go[r,c] == o.color for r,c in o.cells)]
    if not preserved or not overwritten: return None, None
    # What distinguishes preserved from overwritten?
    for prop in ['color', 'size_gt', 'size_lt', 'rectangular']:
        if prop == 'color':
            pc = set(o.color for o in preserved)
            oc = set(o.color for o in overwritten)
            if pc & oc: continue  # overlap
            def mk(fc=fill_color, keep_colors=pc):
                def apply(t):
                    guesses = []
                    for tc in t['test']:
                        a = np.array(tc['input'])
                        bg2 = int(np.argmax(np.bincount(a.flatten())))
                        objs2, _ = extract_objects(a)
                        out = np.full_like(a, fc)
                        for o in objs2:
                            if o.color in keep_colors:
                                for r, c in o.cells: out[r, c] = o.color
                        guesses.append([out.tolist()])
                    return guesses
                return apply
            result = mk()(task)
            if score_task(task, result): return result, "OG:cond_fill_by_color"
    return None, None


def try_two_step_og_chain(task):
    """Try chaining two OG/ADV solvers: apply solver A to get intermediate, then solver B.
    This is the composition layer that finds tasks needing 2-step reasoning."""
    pairs = task['train']
    tests = task['test']
    
    # Import ADV ops for chaining
    try:
        from arc_advanced_ops import ALL_ADVANCED_OPS
        adv_ops = ALL_ADVANCED_OPS
    except:
        adv_ops = []
    
    # Build list of "transform functions" that take pairs and return a rule
    # Each rule is: grid → grid
    transform_fns = []
    
    # Fast ADV ops (skip slow ones)
    fast_adv = [n for n, _ in adv_ops if n not in ('connected_flood', 'flood_seeds', 'pattern_stamp')]
    for name, fn in adv_ops:
        if name in fast_adv:
            transform_fns.append((f"ADV:{name}", fn))
    
    # OG single-step transforms (only ones that preserve grid size)
    og_single = [
        ("OG:fill_h", try_fill_between_same_color_h),
        ("OG:fill_v", try_fill_between_same_color_v),
        ("OG:fill_hv", try_fill_between_same_color_hv),
        ("OG:fill_row", try_fill_row_col_from_object),
        ("OG:connect", try_connect_same_color),
        ("OG:grow", try_grow_objects),
        ("OG:shrink", try_shrink_objects),
        ("OG:outline", try_outline_objects),
        ("OG:fill_bbox", try_fill_bbox),
        ("OG:maj_fill", try_majority_color_fill),
        ("OG:complement", try_color_complement),
    ]
    
    # For each first-step op, try to get it to produce an intermediate
    # Then for each second-step op, try to solve intermediate→output
    
    # Limit search: try top 15 first-ops × top 15 second-ops = 225 combinations
    for name_a, fn_a in transform_fns[:15]:
        try:
            rule_a = fn_a(pairs)
            if rule_a is None:
                continue
            
            # Compute intermediates
            intermediates = []
            ok = True
            for p in pairs:
                try:
                    mid = rule_a(p['input'])
                    mid_arr = np.array(mid) if not isinstance(mid, np.ndarray) else mid
                    # Check that mid is different from input (op did something)
                    if np.array_equal(mid_arr, np.array(p['input'])):
                        ok = False; break
                    # Check same size as output (most ARC tasks preserve size)
                    go = np.array(p['output'])
                    if mid_arr.shape != go.shape:
                        ok = False; break
                    intermediates.append({
                        'input': mid_arr.tolist(),
                        'output': p['output']
                    })
                except:
                    ok = False; break
            
            if not ok or not intermediates:
                continue
            
            # Try second-step ops on intermediate→output
            for name_b, fn_b in transform_fns[:15]:
                if name_b == name_a:
                    continue
                try:
                    rule_b = fn_b(intermediates)
                    if rule_b is None:
                        continue
                    
                    # Verify full chain on all training pairs
                    chain_ok = True
                    for p in pairs:
                        try:
                            mid = rule_a(p['input'])
                            mid_list = mid.tolist() if isinstance(mid, np.ndarray) else mid
                            final = rule_b(mid_list)
                            final_arr = np.array(final) if not isinstance(final, np.ndarray) else final
                            if not np.array_equal(final_arr, np.array(p['output'])):
                                chain_ok = False; break
                        except:
                            chain_ok = False; break
                    
                    if not chain_ok:
                        continue
                    
                    # Apply chain to test cases
                    guesses = []
                    test_ok = True
                    for tc in tests:
                        try:
                            mid = rule_a(tc['input'])
                            mid_list = mid.tolist() if isinstance(mid, np.ndarray) else mid
                            final = rule_b(mid_list)
                            final_arr = np.array(final) if not isinstance(final, np.ndarray) else final
                            guesses.append([final_arr.tolist()])
                        except:
                            test_ok = False; break
                    
                    if test_ok and guesses and score_task(task, guesses):
                        return guesses, f"CHAIN:{name_a}→{name_b}"
                
                except:
                    continue
        except:
            continue
    
    return None, None


def try_stamp_cross_at_pixels(task):
    """Stamp a small pattern (cross, diamond, plus) centered on each non-bg pixel.
    Discovered from 0ca9ddb6: each colored pixel gets a cross/diamond stamp."""
    pairs = task['train']
    for p in pairs:
        if np.array(p['input']).shape != np.array(p['output']).shape: return None, None
    
    # Learn stamps per color across ALL training pairs
    gi = np.array(pairs[0]['input']); go = np.array(pairs[0]['output']); h, w = gi.shape
    bg = int(np.argmax(np.bincount(gi.flatten())))
    
    color_stamps = {}  # color → {(dr, dc): stamp_color}
    color_instances = {}  # color → count across all pairs
    
    for p in pairs:
        gi_p = np.array(p['input']); go_p = np.array(p['output'])
        hp, wp = gi_p.shape
        for r in range(hp):
            for c in range(wp):
                if gi_p[r, c] != bg:
                    color = int(gi_p[r, c])
                    color_instances[color] = color_instances.get(color, 0) + 1
                    stamp = {}
                    for dr in range(-2, 3):
                        for dc in range(-2, 3):
                            if dr == 0 and dc == 0: continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < hp and 0 <= nc < wp:
                                if go_p[nr, nc] != gi_p[nr, nc] and go_p[nr, nc] != bg:
                                    stamp[(dr, dc)] = int(go_p[nr, nc])
                    if stamp:
                        if color in color_stamps:
                            # Intersect with existing
                            common = {k: v for k, v in stamp.items()
                                       if k in color_stamps[color] and color_stamps[color][k] == v}
                            color_stamps[color] = common
                        else:
                            color_stamps[color] = stamp
    
    # Only keep stamps for colors with 2+ instances AND non-empty stamps
    color_stamps = {c: s for c, s in color_stamps.items()
                     if s and color_instances.get(c, 0) >= 2}
    
    if not color_stamps:
        return None, None
    
    # Verify: apply stamps and check
    test = gi.copy()
    for r in range(h):
        for c in range(w):
            if gi[r, c] != bg:
                color = int(gi[r, c])
                if color in color_stamps:
                    for (dr, dc), sc in color_stamps[color].items():
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and test[nr, nc] == bg:
                            test[nr, nc] = sc
    
    if not np.array_equal(test, go):
        return None, None
    
    # Verify on all pairs
    for p in pairs[1:]:
        gi2 = np.array(p['input']); go2 = np.array(p['output'])
        t2 = gi2.copy()
        bg2 = int(np.argmax(np.bincount(gi2.flatten())))
        for r in range(gi2.shape[0]):
            for c in range(gi2.shape[1]):
                if gi2[r, c] != bg2 and int(gi2[r, c]) in color_stamps:
                    for (dr, dc), sc in color_stamps[int(gi2[r, c])].items():
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < gi2.shape[0] and 0 <= nc < gi2.shape[1] and t2[nr, nc] == bg2:
                            t2[nr, nc] = sc
        if not np.array_equal(t2, go2):
            return None, None
    
    def mk(stamps=color_stamps):
        def apply(t):
            guesses = []
            for tc in t['test']:
                a = np.array(tc['input']); hh, ww = a.shape
                bg2 = int(np.argmax(np.bincount(a.flatten())))
                out = a.copy()
                for r in range(hh):
                    for c in range(ww):
                        if a[r, c] != bg2 and int(a[r, c]) in stamps:
                            for (dr, dc), sc in stamps[int(a[r, c])].items():
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < hh and 0 <= nc < ww and out[nr, nc] == bg2:
                                    out[nr, nc] = sc
                guesses.append([out.tolist()])
            return guesses
        return apply
    result = mk()(task)
    if score_task(task, result): return result, "OG:stamp_cross"
    return None, None


def try_symmetry_completion(task):
    """Complete a symmetric pattern by mirroring existing non-bg pixels.
    Discovered from 11852cab: fill missing symmetric positions."""
    pairs = task['train']
    for p in pairs:
        if np.array(p['input']).shape != np.array(p['output']).shape: return None, None
    gi = np.array(pairs[0]['input']); go = np.array(pairs[0]['output']); h, w = gi.shape
    bg = int(np.argmax(np.bincount(gi.flatten())))
    
    # Find center of non-bg content
    nz = np.argwhere(gi != bg)
    if len(nz) < 2: return None, None
    cr = (nz[:, 0].min() + nz[:, 0].max()) / 2
    cc = (nz[:, 1].min() + nz[:, 1].max()) / 2
    
    for sym_type in ['4fold', 'h_mirror', 'v_mirror', 'diag']:
        test = gi.copy()
        for r in range(h):
            for c in range(w):
                if gi[r, c] != bg:
                    mirrors = []
                    if sym_type in ['4fold', 'h_mirror']:
                        mr = int(round(2 * cr - r))
                        mirrors.append((mr, c))
                    if sym_type in ['4fold', 'v_mirror']:
                        mc = int(round(2 * cc - c))
                        mirrors.append((r, mc))
                    if sym_type == '4fold':
                        mirrors.append((int(round(2*cr-r)), int(round(2*cc-c))))
                    if sym_type == 'diag':
                        dr, dc = r - round(cr), c - round(cc)
                        mirrors.append((int(round(cr + dc)), int(round(cc + dr))))
                    
                    for mr, mc in mirrors:
                        if 0 <= mr < h and 0 <= mc < w and test[mr, mc] == bg:
                            test[mr, mc] = int(gi[r, c])
        
        if np.array_equal(test, go):
            for p in pairs[1:]:
                gi2 = np.array(p['input']); go2 = np.array(p['output'])
                bg2 = int(np.argmax(np.bincount(gi2.flatten())))
                nz2 = np.argwhere(gi2 != bg2)
                if len(nz2) < 2: return None, None
                cr2 = (nz2[:,0].min()+nz2[:,0].max())/2
                cc2 = (nz2[:,1].min()+nz2[:,1].max())/2
                t2 = gi2.copy()
                for r in range(gi2.shape[0]):
                    for c in range(gi2.shape[1]):
                        if gi2[r,c] != bg2:
                            ms = []
                            if sym_type in ['4fold','h_mirror']: ms.append((int(round(2*cr2-r)),c))
                            if sym_type in ['4fold','v_mirror']: ms.append((r,int(round(2*cc2-c))))
                            if sym_type == '4fold': ms.append((int(round(2*cr2-r)),int(round(2*cc2-c))))
                            if sym_type == 'diag':
                                dr,dc=r-round(cr2),c-round(cc2)
                                ms.append((int(round(cr2+dc)),int(round(cc2+dr))))
                            for mr,mc in ms:
                                if 0<=mr<gi2.shape[0] and 0<=mc<gi2.shape[1] and t2[mr,mc]==bg2:
                                    t2[mr,mc]=int(gi2[r,c])
                if not np.array_equal(t2, go2): break
            else:
                def mk(st=sym_type):
                    def apply(t):
                        guesses = []
                        for tc in t['test']:
                            a = np.array(tc['input']); hh,ww = a.shape
                            bg2 = int(np.argmax(np.bincount(a.flatten())))
                            nz2 = np.argwhere(a!=bg2)
                            if len(nz2)<2: guesses.append([a.tolist()]); continue
                            cr2=(nz2[:,0].min()+nz2[:,0].max())/2
                            cc2=(nz2[:,1].min()+nz2[:,1].max())/2
                            out = a.copy()
                            for r in range(hh):
                                for c in range(ww):
                                    if a[r,c]!=bg2:
                                        ms=[]
                                        if st in ['4fold','h_mirror']: ms.append((int(round(2*cr2-r)),c))
                                        if st in ['4fold','v_mirror']: ms.append((r,int(round(2*cc2-c))))
                                        if st=='4fold': ms.append((int(round(2*cr2-r)),int(round(2*cc2-c))))
                                        if st=='diag':
                                            dr,dc=r-round(cr2),c-round(cc2)
                                            ms.append((int(round(cr2+dc)),int(round(cc2+dr))))
                                        for mr,mc in ms:
                                            if 0<=mr<hh and 0<=mc<ww and out[mr,mc]==bg2:
                                                out[mr,mc]=int(a[r,c])
                            guesses.append([out.tolist()])
                        return guesses
                    return apply
                result = mk()(task)
                if score_task(task, result): return result, f"OG:sym_complete_{sym_type}"
    return None, None


def try_move_object_toward_anchor(task):
    """Move one object toward another (translate until adjacent).
    Discovered from 05f2a901: object 2 slides toward object 8."""
    pairs = task['train']
    for p in pairs:
        if np.array(p['input']).shape != np.array(p['output']).shape: return None, None
    
    gi = np.array(pairs[0]['input']); go = np.array(pairs[0]['output']); h, w = gi.shape
    bg = int(np.argmax(np.bincount(gi.flatten())))
    objs_in, _ = extract_objects(gi)
    objs_out, _ = extract_objects(go)
    if len(objs_in) < 2: return None, None
    
    # Find which object moved and which is the anchor
    for mover in objs_in:
        # Check if mover's cells changed in output
        cells_same = all(go[r, c] == mover.color for r, c in mover.cells)
        if cells_same: continue  # didn't move
        
        # Find matching object in output (same color, same shape)
        for mo in objs_out:
            if mo.color != mover.color or mo.shape_hash != mover.shape_hash: continue
            if mo.size != mover.size: continue
            
            # Compute translation vector
            dr = int(round(mo.center[0] - mover.center[0]))
            dc = int(round(mo.center[1] - mover.center[1]))
            if dr == 0 and dc == 0: continue
            
            # Which anchor did it move toward?
            for anchor in objs_in:
                if anchor.id == mover.id: continue
                # Did moving get closer to anchor?
                old_dist = abs(mover.center[0]-anchor.center[0]) + abs(mover.center[1]-anchor.center[1])
                new_dist = abs(mo.center[0]-anchor.center[0]) + abs(mo.center[1]-anchor.center[1])
                if new_dist >= old_dist: continue
                
                # Verify: clear old position, draw new position
                test = gi.copy()
                for r, c in mover.cells: test[r, c] = bg
                for r, c in mover.cells:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w: test[nr, nc] = mover.color
                
                if np.array_equal(test, go):
                    mover_color = mover.color; anchor_color = anchor.color
                    def mk(mc=mover_color, ac=anchor_color):
                        def apply(t):
                            guesses = []
                            for tc in t['test']:
                                a = np.array(tc['input']); hh, ww = a.shape
                                bg2 = int(np.argmax(np.bincount(a.flatten())))
                                objs2, _ = extract_objects(a)
                                movers = [o for o in objs2 if o.color == mc]
                                anchors = [o for o in objs2 if o.color == ac]
                                out = a.copy()
                                if movers and anchors:
                                    mv = movers[0]; an = anchors[0]
                                    # Move toward anchor until adjacent
                                    ddr = 1 if an.center[0] > mv.center[0] else (-1 if an.center[0] < mv.center[0] else 0)
                                    ddc = 1 if an.center[1] > mv.center[1] else (-1 if an.center[1] < mv.center[1] else 0)
                                    # Slide until touching
                                    for step in range(1, max(hh, ww)):
                                        new_cells = [(r+ddr*step, c+ddc*step) for r, c in mv.cells]
                                        if any(nr<0 or nr>=hh or nc<0 or nc>=ww for nr, nc in new_cells): break
                                        # Check if touching anchor
                                        anc_set = set(an.cells)
                                        touching = any((nr+dr2, nc+dc2) in anc_set
                                                        for nr, nc in new_cells
                                                        for dr2, dc2 in [(-1,0),(1,0),(0,-1),(0,1)])
                                        if touching:
                                            for r, c in mv.cells: out[r, c] = bg2
                                            for nr, nc in new_cells: out[nr, nc] = mc
                                            break
                                guesses.append([out.tolist()])
                            return guesses
                        return apply
                    result = mk()(task)
                    if score_task(task, result): return result, f"OG:move_toward_{anchor_color}"
    return None, None


def try_color_rotation_map(task):
    """Apply a color rotation defined by a key region in the grid.
    Discovered from 0becf7df: top-left 2x2 defines color mapping."""
    pairs = task['train']
    for p in pairs:
        if np.array(p['input']).shape != np.array(p['output']).shape: return None, None
    gi = np.array(pairs[0]['input']); go = np.array(pairs[0]['output']); h, w = gi.shape
    bg = int(np.argmax(np.bincount(gi.flatten())))
    
    # Try using small corner regions as color keys
    for kh in [2, 3]:
        for kw in [2, 3]:
            if kh >= h or kw >= w: continue
            for corner in ['tl', 'tr', 'bl', 'br']:
                if corner == 'tl': key = gi[:kh, :kw]
                elif corner == 'tr': key = gi[:kh, w-kw:]
                elif corner == 'bl': key = gi[h-kh:, :kw]
                elif corner == 'br': key = gi[h-kh:, w-kw:]
                
                key_colors = [int(v) for v in key.flatten() if v != bg]
                if len(key_colors) < 2: continue
                
                # Build color rotation from key
                unique_key = list(dict.fromkeys(key_colors))  # preserve order
                if len(unique_key) < 2: continue
                
                # Rotation: each color maps to the next in the key
                cmap = {}
                for i, c in enumerate(unique_key):
                    cmap[c] = unique_key[(i + 1) % len(unique_key)]
                
                # Apply rotation to all non-key, non-bg cells
                test = gi.copy()
                for r in range(h):
                    for c in range(w):
                        # Skip the key region
                        if corner == 'tl' and r < kh and c < kw: continue
                        if corner == 'tr' and r < kh and c >= w-kw: continue
                        if corner == 'bl' and r >= h-kh and c < kw: continue
                        if corner == 'br' and r >= h-kh and c >= w-kw: continue
                        
                        if int(gi[r, c]) in cmap:
                            test[r, c] = cmap[int(gi[r, c])]
                
                if np.array_equal(test, go):
                    # Verify on other pairs
                    ok = True
                    for p in pairs[1:]:
                        gi2 = np.array(p['input']); go2 = np.array(p['output'])
                        bg2 = int(np.argmax(np.bincount(gi2.flatten())))
                        if corner == 'tl': k2 = gi2[:kh, :kw]
                        elif corner == 'tr': k2 = gi2[:kh, gi2.shape[1]-kw:]
                        elif corner == 'bl': k2 = gi2[gi2.shape[0]-kh:, :kw]
                        elif corner == 'br': k2 = gi2[gi2.shape[0]-kh:, gi2.shape[1]-kw:]
                        kc2 = list(dict.fromkeys(int(v) for v in k2.flatten() if v != bg2))
                        if len(kc2) < 2: ok = False; break
                        cm2 = {c: kc2[(i+1)%len(kc2)] for i, c in enumerate(kc2)}
                        t2 = gi2.copy()
                        for r in range(gi2.shape[0]):
                            for c in range(gi2.shape[1]):
                                if corner=='tl' and r<kh and c<kw: continue
                                if corner=='tr' and r<kh and c>=gi2.shape[1]-kw: continue
                                if corner=='bl' and r>=gi2.shape[0]-kh and c<kw: continue
                                if corner=='br' and r>=gi2.shape[0]-kh and c>=gi2.shape[1]-kw: continue
                                if int(gi2[r,c]) in cm2: t2[r,c]=cm2[int(gi2[r,c])]
                        if not np.array_equal(t2, go2): ok = False; break
                    if not ok: continue
                    
                    def mk(cn=corner, kk_h=kh, kk_w=kw):
                        def apply(t):
                            guesses = []
                            for tc in t['test']:
                                a = np.array(tc['input']); hh, ww = a.shape
                                bg2 = int(np.argmax(np.bincount(a.flatten())))
                                if cn=='tl': k=a[:kk_h,:kk_w]
                                elif cn=='tr': k=a[:kk_h,ww-kk_w:]
                                elif cn=='bl': k=a[hh-kk_h:,:kk_w]
                                elif cn=='br': k=a[hh-kk_h:,ww-kk_w:]
                                kc=list(dict.fromkeys(int(v) for v in k.flatten() if v!=bg2))
                                cm={c:kc[(i+1)%len(kc)] for i,c in enumerate(kc)}
                                out=a.copy()
                                for r in range(hh):
                                    for c in range(ww):
                                        if cn=='tl' and r<kk_h and c<kk_w: continue
                                        if cn=='tr' and r<kk_h and c>=ww-kk_w: continue
                                        if cn=='bl' and r>=hh-kk_h and c<kk_w: continue
                                        if cn=='br' and r>=hh-kk_h and c>=ww-kk_w: continue
                                        if int(a[r,c]) in cm: out[r,c]=cm[int(a[r,c])]
                                guesses.append([out.tolist()])
                            return guesses
                        return apply
                    result = mk()(task)
                    if score_task(task, result): return result, f"OG:color_rotate_{corner}_{kh}x{kw}"
    return None, None


ALL_OG_SOLVERS = [
    # Original (skip move_to_target — too slow, 0 finds)
    ("keep_by_property", try_keep_by_property),
    ("recolor_by_relation", try_recolor_by_relation),
    # ("move_to_target", try_move_object_to_target),  # 40s, 0 finds
    ("extract_by_shape", try_extract_by_shape),
    ("stamp_pattern", try_stamp_pattern),
    # Expanded 5
    ("recolor_by_count", try_recolor_by_count),
    ("sort_objects", try_sort_objects),
    ("copy_to_markers", try_copy_shape_to_markers),
    ("fill_between", try_fill_between_objects),
    ("mirror_across_line", try_mirror_object_across_axis),
    # NEW: Containment + property-based (Leetcode tricks)
    ("contained_recolor", try_contained_recolor),
    ("extract_contained", try_extract_contained),
    ("recolor_size_rank", try_recolor_by_size_rank),
    ("keep_shared", try_keep_shared_property),
    ("keep_unique", try_remove_shared_property),
    ("color_from_size", try_map_color_to_size),
    ("color_from_neighbors", try_map_color_to_neighbor_count),
    ("grow_objects", try_grow_objects),
    ("shrink_objects", try_shrink_objects),
    # More relation solvers
    ("output_from_count", try_output_from_object_count),
    ("partition_process", try_partition_and_process),
    ("color_majority_vote", try_color_majority_vote),
    ("recolor_by_position", try_recolor_by_position),
    # Fill/connect/extend solvers
    ("extend_lines", try_extend_object_lines),
    ("fill_row_col", try_fill_row_col_from_object),
    ("connect_same_color", try_connect_same_color),
    ("flood_voronoi", try_flood_from_objects),
    # Massive haul: extraction, tiling, swaps, outlines
    ("extract_fixed_region", try_extract_common_subgrid),
    ("tile_input", try_tile_input),
    ("upscale_pixels", try_upscale_pixels),
    ("transpose", try_transpose),
    ("dedup_rows_cols", try_remove_duplicate_rows_cols),
    ("color_swap", try_color_swap),
    ("outline_objects", try_outline_objects),
    ("fill_bbox", try_fill_bbox),
    ("extract_largest", try_extract_largest_object),
    ("extract_smallest", try_extract_smallest_object),
    ("majority_fill", try_majority_color_fill),
    # Gap-filling batch: fill patterns, counting, extraction
    ("fill_between_h", try_fill_between_same_color_h),
    ("fill_between_v", try_fill_between_same_color_v),
    ("fill_between_hv", try_fill_between_same_color_hv),
    ("count_colors", try_count_colors_output),
    ("count_objects", try_count_objects_output),
    ("histogram", try_color_histogram_row),
    ("crop_content", try_extract_non_bg_bbox),
    ("stamp_markers", try_replace_color_with_pattern),
    # High-impact gap fillers
    ("fill_nearest", try_fill_nearest_object_color),
    ("grid_pattern", try_recolor_by_grid_pattern),
    ("color_complement", try_color_complement),
    ("color_offset", try_color_offset),
    ("conditional_fill", try_conditional_fill),
    # Multi-step composition (runs last, most expensive)
    ("two_step_chain", try_two_step_og_chain),
    # Discovered by inspecting model's thought process
    ("stamp_cross", try_stamp_cross_at_pixels),
    ("symmetry_complete", try_symmetry_completion),
    ("move_toward", try_move_object_toward_anchor),
    ("color_rotation", try_color_rotation_map),
]


def solve_with_object_graph(task):
    """Try all object-graph solvers on a task."""
    for name, solver_fn in ALL_OG_SOLVERS:
        try:
            result, method = solver_fn(task)
            if result and score_task(task, result):
                return result, method
        except Exception:
            continue
    return None, None


# ══════════════════════════════════════════════════════════════════════
# 5. TESTS AND BENCHMARK
# ══════════════════════════════════════════════════════════════════════

def run_tests():
    print("=" * 65)
    print("  Object Graph Solver Tests")
    print("=" * 65)
    p = 0; t = 0

    print("\n  T1: Object extraction")
    grid = np.array([[0,0,1,1],[0,0,1,1],[2,2,0,0],[2,2,0,0]])
    objs, bg = extract_objects(grid)
    ok = len(objs) == 2 and bg == 0
    print(f"    Objects: {len(objs)}, bg: {bg} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T2: Object properties")
    ok = objs[0].size == 4 and objs[0].is_rectangular
    print(f"    Size: {objs[0].size}, rect: {objs[0].is_rectangular} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T3: Relations")
    ok = not touches(objs[0], objs[1])  # diagonal, not touching
    print(f"    Touching: {touches(objs[0], objs[1])} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T4: Shape matching")
    ok = same_shape(objs[0], objs[1])  # both 2x2
    print(f"    Same shape: {same_shape(objs[0], objs[1])} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T5: Solvers available")
    ok = len(ALL_OG_SOLVERS) >= 5
    print(f"    Solvers: {len(ALL_OG_SOLVERS)} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print(f"\n{'='*65}")
    print(f"  Results: {p}/{t} tests passed")
    print(f"{'='*65}")


def run_benchmark(data_dir, verbose=False, skip_s1=False):
    files = sorted(f for f in os.listdir(data_dir) if f.endswith('.json'))

    print("=" * 70)
    print(f"  ARC-AGI-2 Object Graph Solver")
    print(f"  Tasks: {len(files)}")
    print(f"  Solvers: {len(ALL_OG_SOLVERS)}")
    print(f"  Mode: {'OG-only (fast)' if skip_s1 else 'S1 + OG'}")
    print("=" * 70)

    s1_n = og_n = 0
    og_tasks = []
    t0 = time.time()

    for fi, f in enumerate(files):
        task = json.load(open(os.path.join(data_dir, f)))

        if not skip_s1:
            # S1 first (slow)
            s1 = s1_solve(task)
            if score_task(task, s1):
                s1_n += 1
                continue

        # Object graph
        result, method = solve_with_object_graph(task)
        if result and score_task(task, result):
            og_n += 1
            og_tasks.append((f, method))
            if verbose:
                print(f"  {f}: {method}")

        if (fi + 1) % 200 == 0:
            print(f"    Progress: {fi+1}/{len(files)} | "
                  f"{'S1: ' + str(s1_n) + ' | ' if not skip_s1 else ''}"
                  f"OG: {og_n} | {time.time()-t0:.0f}s")

    elapsed = time.time() - t0
    total = s1_n + og_n

    print(f"\n{'='*70}")
    print(f"  OBJECT GRAPH RESULTS")
    print(f"{'='*70}")
    print(f"  Solved:     {total}/{len(files)} ({total/len(files)*100:.1f}%)")
    print(f"    S1:       {s1_n}")
    print(f"    OG:       {og_n}")
    if og_tasks:
        print(f"  OG solves:")
        for f, method in og_tasks:
            print(f"    {f}: {method}")
    print(f"  Time:       {elapsed:.0f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--training", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--skip-s1", action="store_true", help="Skip S1 filtering (fast OG-only mode)")
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
    else:
        if args.eval:
            d = os.path.join(data_base, "evaluation")
        else:
            d = os.path.join(data_base, "training")
        run_benchmark(d, verbose=args.verbose, skip_s1=getattr(args, 'skip_s1', False))
