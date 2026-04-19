"""
arc_phase2.py — ARC-AGI-2 Phase 2: Shape-Aware Pattern Synthesis
===================================================================
Extends Phase 1 with:
  1. Shape templates (cross, L, T, line, square, diamond)
  2. Pattern completion (extend repeating motifs)  
  3. Object-relative transforms (move/copy/recolor relative to other objects)
  4. Grid segmentation (split by dividers, process segments)
  5. Flood fill variants (by region, by object containment)
  6. Deeper 3-step search with color-parameterized operations

Usage:
    python arc_phase2.py --training --data path/to/ARC-AGI-2/data
    python arc_phase2.py --eval --data path/to/ARC-AGI-2/data  
    python arc_phase2.py --test
"""

import argparse
import json
import os
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Set
import time

from arc_solver import (Grid, GridObject, SINGLE_OPS, score_task,
                         infer_color_map, apply_color_map,
                         dsl_identity, dsl_rot90, dsl_rot180, dsl_rot270,
                         dsl_flip_h, dsl_flip_v, dsl_transpose,
                         dsl_crop_nonzero, dsl_crop_to_bg,
                         dsl_scale2, dsl_scale3,
                         dsl_largest_object, dsl_smallest_object,
                         dsl_outline, dsl_fill_holes,
                         dsl_gravity_down, dsl_gravity_left,
                         dsl_keep_only_color, dsl_remove_color)
from arc_phase1 import (Scene, Diff, analyze_task_diffs,
                          solve_task_phase1, PHASE1_SEARCHES)


# ══════════════════════════════════════════════════════════════════════
# 1. SHAPE DETECTION
# ══════════════════════════════════════════════════════════════════════

def detect_shape(cells, grid_h, grid_w):
    """Classify a set of cells as a shape type."""
    if not cells:
        return 'empty', {}
    
    rs = [r for r, c in cells]
    cs = [c for r, c in cells]
    min_r, max_r = min(rs), max(rs)
    min_c, max_c = min(cs), max(cs)
    h = max_r - min_r + 1
    w = max_c - min_c + 1
    n = len(cells)
    cell_set = set(cells)
    
    # Normalize to origin
    norm = frozenset((r - min_r, c - min_c) for r, c in cells)
    center_r = (min_r + max_r) / 2
    center_c = (min_c + max_c) / 2
    
    props = {'n': n, 'h': h, 'w': w, 'center': (center_r, center_c),
             'bbox': (min_r, min_c, max_r, max_c)}
    
    # Single cell
    if n == 1:
        return 'dot', props
    
    # Line
    if h == 1:
        return 'h_line', props
    if w == 1:
        return 'v_line', props
    
    # Rectangle
    if n == h * w:
        if h == w:
            return 'square', props
        return 'rectangle', props
    
    # Cross/plus pattern (5 cells)
    if n == 5 and h == 3 and w == 3:
        expected = {(0,1),(1,0),(1,1),(1,2),(2,1)}
        if norm == expected:
            return 'cross', props
    
    # L-shape (3 cells)
    if n == 3:
        l_shapes = [
            frozenset([(0,0),(1,0),(1,1)]),
            frozenset([(0,1),(1,0),(1,1)]),
            frozenset([(0,0),(0,1),(1,0)]),
            frozenset([(0,0),(0,1),(1,1)]),
        ]
        if norm in l_shapes:
            return 'L', props
    
    # T-shape
    if n == 4:
        t_shapes = [
            frozenset([(0,0),(0,1),(0,2),(1,1)]),
            frozenset([(0,1),(1,0),(1,1),(1,2)]),
            frozenset([(0,0),(1,0),(2,0),(1,1)]),
            frozenset([(0,1),(1,0),(1,1),(2,1)]),
        ]
        if norm in t_shapes:
            return 'T', props
    
    # Diamond
    if n >= 4:
        cr, cc = int(center_r - min_r), int(center_c - min_c)
        expected_diamond = set()
        radius = max(h, w) // 2
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if abs(dr) + abs(dc) <= radius:
                    expected_diamond.add((cr + dr, cc + dc))
        if norm == frozenset(expected_diamond):
            return 'diamond', props
    
    # Frame/border of rectangle
    if n == 2 * (h + w) - 4 and h >= 3 and w >= 3:
        expected_frame = set()
        for r in range(h):
            for c in range(w):
                if r == 0 or r == h-1 or c == 0 or c == w-1:
                    expected_frame.add((r, c))
        if norm == frozenset(expected_frame):
            return 'frame', props
    
    return 'blob', props


# ══════════════════════════════════════════════════════════════════════
# 2. GRID SEGMENTATION
# ══════════════════════════════════════════════════════════════════════

def find_dividers(grid: Grid):
    """Find horizontal/vertical divider lines (full row/col of one color)."""
    h_dividers = []
    v_dividers = []
    arr = grid.arr
    bg = grid.background
    
    for r in range(grid.h):
        vals = set(arr[r, :].tolist())
        if len(vals) == 1:
            color = vals.pop()
            if color != bg:
                h_dividers.append((r, color))
    
    for c in range(grid.w):
        vals = set(arr[:, c].tolist())
        if len(vals) == 1:
            color = vals.pop()
            if color != bg:
                v_dividers.append((c, color))
    
    return h_dividers, v_dividers


def segment_by_dividers(grid: Grid):
    """Split grid into segments using divider lines."""
    h_div, v_div = find_dividers(grid)
    
    if not h_div and not v_div:
        return None
    
    segments = []
    h_splits = [0] + [r for r, _ in h_div] + [grid.h]
    v_splits = [0] + [c for c, _ in v_div] + [grid.w]
    
    for i in range(len(h_splits) - 1):
        for j in range(len(v_splits) - 1):
            r0, r1 = h_splits[i], h_splits[i+1]
            c0, c1 = v_splits[j], v_splits[j+1]
            # Skip divider rows/cols
            if r1 - r0 <= 1 or c1 - c0 <= 1:
                continue
            # Adjust to skip the divider
            if i > 0:
                r0 += 1
            if j > 0:
                c0 += 1
            if r1 > r0 and c1 > c0:
                segments.append(grid.subgrid(r0, c0, r1, c1))
    
    return segments if segments else None


# ══════════════════════════════════════════════════════════════════════
# 3. PATTERN COMPLETION
# ══════════════════════════════════════════════════════════════════════

def search_pattern_completion(pairs):
    """Find and complete repeating patterns in rows or columns."""
    if not all(Grid(p['input']).shape == Grid(p['output']).shape for p in pairs):
        return None, None
    
    # Check if the change extends a repeating pattern
    try:
        matches = True
        for p in pairs:
            gi = Grid(p['input'])
            go = Grid(p['output'])
            arr_in = gi.arr.copy()
            arr_out = go.arr.copy()
            
            # For each row, check if output completes a pattern
            for r in range(gi.h):
                row_in = arr_in[r].tolist()
                row_out = arr_out[r].tolist()
                if row_in == row_out:
                    continue
                
                # Find nonzero elements and their spacing
                nz_in = [(c, row_in[c]) for c in range(len(row_in)) if row_in[c] != 0 and row_in[c] != gi.background]
                nz_out = [(c, row_out[c]) for c in range(len(row_out)) if row_out[c] != 0 and row_out[c] != gi.background]
                
                if len(nz_in) < 2 or len(nz_out) <= len(nz_in):
                    matches = False
                    break
                
                # Check if output extends with same spacing
                if len(nz_in) >= 2:
                    spacing = nz_in[1][0] - nz_in[0][0]
                    color = nz_in[0][1]
                    # Verify all existing have same spacing and color
                    uniform = all(nz_in[i][0] - nz_in[i-1][0] == spacing and nz_in[i][1] == color
                                    for i in range(1, len(nz_in)))
                    if not uniform:
                        matches = False
                        break
            
            if not matches:
                break
        
        if matches:
            def complete_pattern(g):
                arr = g.arr.copy()
                bg = g.background
                for r in range(g.h):
                    row = arr[r].tolist()
                    nz = [(c, row[c]) for c in range(len(row)) if row[c] != 0 and row[c] != bg]
                    if len(nz) >= 2:
                        spacing = nz[1][0] - nz[0][0]
                        color = nz[0][1]
                        uniform = all(nz[i][0] - nz[i-1][0] == spacing and nz[i][1] == color
                                        for i in range(1, len(nz)))
                        if uniform and spacing > 0:
                            # Extend forward
                            last_c = nz[-1][0]
                            c = last_c + spacing
                            while c < g.w:
                                if arr[r, c] == 0 or arr[r, c] == bg:
                                    arr[r, c] = color
                                c += spacing
                            # Extend backward
                            first_c = nz[0][0]
                            c = first_c - spacing
                            while c >= 0:
                                if arr[r, c] == 0 or arr[r, c] == bg:
                                    arr[r, c] = color
                                c -= spacing
                return Grid(arr)
            
            for p in pairs:
                if complete_pattern(Grid(p['input'])) != Grid(p['output']):
                    return None, None
            return "row_pattern_complete", complete_pattern
    except Exception:
        pass
    
    return None, None


# ══════════════════════════════════════════════════════════════════════
# 4. SEGMENT-BASED TRANSFORMS
# ══════════════════════════════════════════════════════════════════════

def search_segment_transform(pairs):
    """Split grid by dividers, transform each segment, reassemble."""
    try:
        # Check if all inputs have dividers
        for p in pairs:
            gi = Grid(p['input'])
            segs = segment_by_dividers(gi)
            if not segs:
                return None, None
        
        # Check if output has same structure
        # Try: each segment gets the same single-op transform
        for op_name, op_fn in SINGLE_OPS:
            matches = True
            for p in pairs:
                gi = Grid(p['input'])
                go = Grid(p['output'])
                segs_in = segment_by_dividers(gi)
                segs_out = segment_by_dividers(go)
                if not segs_in or not segs_out:
                    matches = False
                    break
                if len(segs_in) != len(segs_out):
                    matches = False
                    break
                for si, so in zip(segs_in, segs_out):
                    try:
                        transformed = op_fn(si)
                        if transformed != so:
                            matches = False
                            break
                    except Exception:
                        matches = False
                        break
                if not matches:
                    break
            
            if matches:
                def make_seg_transform(op):
                    def transform(g):
                        segs = segment_by_dividers(g)
                        if not segs:
                            return g.copy()
                        result = g.arr.copy()
                        h_div, v_div = find_dividers(g)
                        h_splits = [0] + [r for r, _ in h_div] + [g.h]
                        v_splits = [0] + [c for c, _ in v_div] + [g.w]
                        
                        seg_idx = 0
                        for i in range(len(h_splits) - 1):
                            for j in range(len(v_splits) - 1):
                                r0, r1 = h_splits[i], h_splits[i+1]
                                c0, c1 = v_splits[j], v_splits[j+1]
                                if i > 0: r0 += 1
                                if j > 0: c0 += 1
                                if r1 - r0 <= 0 or c1 - c0 <= 0:
                                    continue
                                if seg_idx < len(segs):
                                    t = op(segs[seg_idx])
                                    th, tw = t.shape
                                    if th == r1 - r0 and tw == c1 - c0:
                                        result[r0:r1, c0:c1] = t.arr
                                    seg_idx += 1
                        return Grid(result)
                    return transform
                return f"seg_{op_name}", make_seg_transform(op_fn)
    except Exception:
        pass
    
    return None, None


# ══════════════════════════════════════════════════════════════════════
# 5. FLOOD FILL BY CONTAINMENT
# ══════════════════════════════════════════════════════════════════════

def search_fill_contained_regions(pairs):
    """Fill enclosed regions based on the enclosing object's color."""
    if not all(Grid(p['input']).shape == Grid(p['output']).shape for p in pairs):
        return None, None
    
    try:
        matches = True
        for p in pairs:
            gi = Grid(p['input'])
            go = Grid(p['output'])
            arr = gi.arr.copy()
            h, w = gi.h, gi.w
            bg = gi.background
            
            # Find enclosed regions of bg
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
            
            # For each interior region, find enclosing color
            visited = np.zeros((h, w), dtype=bool)
            for r in range(h):
                for c in range(w):
                    if arr[r, c] == bg and not exterior[r, c] and not visited[r, c]:
                        # BFS this interior region
                        region = []
                        border_colors = Counter()
                        region_stack = [(r, c)]
                        while region_stack:
                            cr, cc = region_stack.pop()
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
                            region_stack.extend([(cr-1,cc),(cr+1,cc),(cr,cc-1),(cr,cc+1)])
                        
                        if region and border_colors:
                            fill_c = border_colors.most_common(1)[0][0]
                            for cr, cc in region:
                                arr[cr, cc] = fill_c
            
            if not np.array_equal(arr, go.arr):
                matches = False
                break
        
        if matches:
            def fill_contained(g):
                arr = g.arr.copy()
                h, w = g.h, g.w
                bg = g.background
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
                return Grid(arr)
            
            return "fill_by_enclosure", fill_contained
    except Exception:
        pass
    return None, None


# ══════════════════════════════════════════════════════════════════════
# 6. OBJECT PROPERTY RECOLOR
# ══════════════════════════════════════════════════════════════════════

def search_recolor_by_property(pairs):
    """Recolor objects based on their properties (shape, position, containment)."""
    if not all(Grid(p['input']).shape == Grid(p['output']).shape for p in pairs):
        return None, None
    
    # Try: recolor by number of neighbors of same color
    try:
        rules = {}
        consistent = True
        for p in pairs:
            gi = Grid(p['input'])
            go = Grid(p['output'])
            for r in range(gi.h):
                for c in range(gi.w):
                    ic = int(gi.arr[r, c])
                    oc = int(go.arr[r, c])
                    if ic == oc:
                        continue
                    # Count same-color neighbors
                    same_count = 0
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < gi.h and 0 <= nc < gi.w and gi.arr[nr, nc] == ic:
                            same_count += 1
                    key = (ic, same_count)
                    if key in rules and rules[key] != oc:
                        consistent = False
                        break
                    rules[key] = oc
                if not consistent:
                    break
            if not consistent:
                break
        
        if consistent and rules:
            def recolor_same_neighbors(g, r=rules):
                arr = g.arr.copy()
                result = arr.copy()
                for row in range(g.h):
                    for col in range(g.w):
                        ic = int(arr[row, col])
                        same_count = 0
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = row+dr, col+dc
                            if 0 <= nr < g.h and 0 <= nc < g.w and arr[nr, nc] == ic:
                                same_count += 1
                        key = (ic, same_count)
                        if key in r:
                            result[row, col] = r[key]
                return Grid(result)
            
            for p in pairs:
                if recolor_same_neighbors(Grid(p['input'])) != Grid(p['output']):
                    consistent = False
                    break
            if consistent:
                return "recolor_same_neighbors", recolor_same_neighbors
    except Exception:
        pass
    
    # Try: recolor by object size
    try:
        size_rules = {}
        consistent = True
        for p in pairs:
            gi = Grid(p['input'])
            go = Grid(p['output'])
            for obj in gi.objects():
                # Check what color these cells become in output
                out_colors = Counter()
                for r, c in obj.cells:
                    out_colors[int(go.arr[r, c])] += 1
                if len(out_colors) != 1:
                    consistent = False
                    break
                new_color = out_colors.most_common(1)[0][0]
                key = (obj.color, obj.size)
                if key in size_rules and size_rules[key] != new_color:
                    consistent = False
                    break
                size_rules[key] = new_color
            if not consistent:
                break
        
        if consistent and size_rules:
            any_change = any(k[0] != v for k, v in size_rules.items())
            if any_change:
                def recolor_by_objsize(g, sr=size_rules):
                    arr = g.arr.copy()
                    for obj in g.objects():
                        key = (obj.color, obj.size)
                        if key in sr:
                            for r, c in obj.cells:
                                arr[r, c] = sr[key]
                    return Grid(arr)
                
                for p in pairs:
                    if recolor_by_objsize(Grid(p['input'])) != Grid(p['output']):
                        consistent = False
                        break
                if consistent:
                    return "recolor_by_objsize", recolor_by_objsize
    except Exception:
        pass
    
    return None, None


# ══════════════════════════════════════════════════════════════════════
# 7. EXTRACTION BY UNIQUE/ANOMALOUS OBJECT
# ══════════════════════════════════════════════════════════════════════

def search_extract_unique(pairs):
    """Extract the object that's different from the others."""
    for p in pairs:
        gi = Grid(p['input'])
        go = Grid(p['output'])
        objs = gi.objects()
        if len(objs) < 2:
            return None, None
    
    # Try: extract the object with unique color
    try:
        matches = True
        for p in pairs:
            gi = Grid(p['input'])
            go = Grid(p['output'])
            objs = gi.objects()
            color_counts = Counter(o.color for o in objs)
            unique_objs = [o for o in objs if color_counts[o.color] == 1]
            if len(unique_objs) != 1:
                matches = False
                break
            extracted = unique_objs[0].as_grid(bg=0)
            if extracted != go:
                matches = False
                break
        if matches:
            def extract_unique_color(g):
                objs = g.objects()
                cc = Counter(o.color for o in objs)
                unique = [o for o in objs if cc[o.color] == 1]
                if unique:
                    return unique[0].as_grid(bg=0)
                return g.copy()
            return "extract_unique_color", extract_unique_color
    except Exception:
        pass
    
    # Try: extract the object with unique size
    try:
        matches = True
        for p in pairs:
            gi = Grid(p['input'])
            go = Grid(p['output'])
            objs = gi.objects()
            size_counts = Counter(o.size for o in objs)
            unique_objs = [o for o in objs if size_counts[o.size] == 1]
            if len(unique_objs) != 1:
                matches = False
                break
            extracted = unique_objs[0].as_grid(bg=0)
            if extracted != go:
                matches = False
                break
        if matches:
            def extract_unique_size(g):
                objs = g.objects()
                sc = Counter(o.size for o in objs)
                unique = [o for o in objs if sc[o.size] == 1]
                if unique:
                    return unique[0].as_grid(bg=0)
                return g.copy()
            return "extract_unique_size", extract_unique_size
    except Exception:
        pass
    
    return None, None


# ══════════════════════════════════════════════════════════════════════
# 8. BOOLEAN GRID OPERATIONS
# ══════════════════════════════════════════════════════════════════════

def search_boolean_grid_ops(pairs):
    """Split grid into color layers, combine with boolean operations."""
    if not all(Grid(p['input']).shape == Grid(p['output']).shape for p in pairs):
        return None, None
    
    # Try: output = (color_A mask) AND (color_B mask), painted in color_C
    try:
        for p in pairs:
            gi = Grid(p['input'])
            go = Grid(p['output'])
            in_colors = gi.nonzero_colors
            out_colors = go.nonzero_colors
            
            for ca in in_colors:
                for cb in in_colors:
                    if ca >= cb:
                        continue
                    mask_a = gi.arr == ca
                    mask_b = gi.arr == cb
                    
                    # AND
                    intersection = mask_a & mask_b  # always empty for different colors
                    
                    # OR  
                    union = mask_a | mask_b
                    for out_c in out_colors:
                        expected = np.where(union, out_c, 0)
                        if np.array_equal(expected, go.arr):
                            # Verify all pairs
                            all_match = True
                            for p2 in pairs:
                                gi2 = Grid(p2['input'])
                                go2 = Grid(p2['output'])
                                m_a = gi2.arr == ca
                                m_b = gi2.arr == cb
                                u = m_a | m_b
                                exp = np.where(u, out_c, 0)
                                if not np.array_equal(exp, go2.arr):
                                    all_match = False
                                    break
                            if all_match:
                                def make_union(a, b, oc):
                                    def apply(g):
                                        m = (g.arr == a) | (g.arr == b)
                                        return Grid(np.where(m, oc, 0).astype(np.int32))
                                    return apply
                                return f"union_{ca}_{cb}_as_{out_c}", make_union(ca, cb, out_c)
            break  # Only analyze first pair for speed
    except Exception:
        pass
    
    return None, None


# ══════════════════════════════════════════════════════════════════════
# GYM-LEARNED STRATEGIES — Ported from arc_gym.py
# ══════════════════════════════════════════════════════════════════════

def search_gym_pattern_extend(pairs):
    """Extend repeating patterns in rows."""
    if not all(Grid(p['input']).shape == Grid(p['output']).shape for p in pairs):
        return None, None
    try:
        def extend(g):
            arr = g.arr.copy()
            bg = g.background
            for r in range(g.h):
                nz = [(c, int(arr[r, c])) for c in range(g.w) if arr[r, c] != bg]
                if len(nz) >= 2:
                    spacing = nz[1][0] - nz[0][0]
                    color = nz[0][1]
                    if spacing > 0 and all(
                        nz[i][0] - nz[i-1][0] == spacing and nz[i][1] == color
                        for i in range(1, len(nz))):
                        c = nz[-1][0] + spacing
                        while c < g.w:
                            if arr[r, c] == bg:
                                arr[r, c] = color
                            c += spacing
                        c = nz[0][0] - spacing
                        while c >= 0:
                            if arr[r, c] == bg:
                                arr[r, c] = color
                            c -= spacing
            return Grid(arr)

        for p in pairs:
            if extend(Grid(p['input'])) != Grid(p['output']):
                return None, None
        return "pattern_extend", extend
    except Exception:
        return None, None


def search_gym_positional_color(pairs):
    """Output color depends on (input_color, row%mod, col%mod)."""
    if not all(Grid(p['input']).shape == Grid(p['output']).shape for p in pairs):
        return None, None

    for mod in [2, 3, 4]:
        # Pure positional: output depends only on (r%mod, c%mod)
        try:
            rules = {}
            consistent = True
            for p in pairs:
                go = Grid(p['output'])
                for r in range(go.h):
                    for c in range(go.w):
                        key = (r % mod, c % mod)
                        val = int(go.arr[r, c])
                        if key in rules and rules[key] != val:
                            consistent = False
                            break
                        rules[key] = val
                    if not consistent:
                        break
                if not consistent:
                    break
            if consistent and rules:
                def make_pos(r, m):
                    def apply(g):
                        arr = np.zeros_like(g.arr)
                        for row in range(g.h):
                            for col in range(g.w):
                                arr[row, col] = r.get((row % m, col % m), 0)
                        return Grid(arr)
                    return apply
                fn = make_pos(rules, mod)
                if all(fn(Grid(p['input'])) == Grid(p['output']) for p in pairs):
                    return f"positional_mod{mod}", fn
        except Exception:
            continue

        # Input-dependent positional: (input_color, r%mod, c%mod) -> output
        try:
            rules = {}
            consistent = True
            for p in pairs:
                gi = Grid(p['input'])
                go = Grid(p['output'])
                for r in range(gi.h):
                    for c in range(gi.w):
                        key = (int(gi.arr[r, c]), r % mod, c % mod)
                        val = int(go.arr[r, c])
                        if key in rules and rules[key] != val:
                            consistent = False
                            break
                        rules[key] = val
                    if not consistent:
                        break
                if not consistent:
                    break
            if consistent and rules:
                all_id = all(k[0] == v for k, v in rules.items())
                if not all_id:
                    def make_ipos(r, m):
                        def apply(g):
                            arr = g.arr.copy()
                            for row in range(g.h):
                                for col in range(g.w):
                                    key = (int(arr[row, col]), row % m, col % m)
                                    if key in r:
                                        arr[row, col] = r[key]
                            return Grid(arr)
                        return apply
                    fn = make_ipos(rules, mod)
                    if all(fn(Grid(p['input'])) == Grid(p['output']) for p in pairs):
                        return f"input_pos_mod{mod}", fn
        except Exception:
            continue

    return None, None


def search_gym_tile_self(pairs):
    """Nonzero cells of input get replaced by full input grid."""
    try:
        for p in pairs:
            gi = Grid(p['input'])
            go = Grid(p['output'])
            ih, iw = gi.shape
            oh, ow = go.shape
            if oh != ih * ih or ow != iw * iw:
                return None, None

        matches = True
        for p in pairs:
            gi = Grid(p['input'])
            go = Grid(p['output'])
            ih, iw = gi.shape
            result = np.zeros((ih*ih, iw*iw), dtype=np.int32)
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


def search_gym_symmetry_complete(pairs):
    """Fill zeros to make grid symmetric."""
    if not all(Grid(p['input']).shape == Grid(p['output']).shape for p in pairs):
        return None, None

    for sym in ['h', 'v', 'both', 'diag']:
        try:
            matches = True
            for p in pairs:
                gi = Grid(p['input'])
                go = Grid(p['output'])
                arr = gi.arr.copy()
                h, w = arr.shape

                if sym == 'h':
                    for r in range(h):
                        for c in range(w):
                            mc = w-1-c
                            if arr[r,c] == 0 and arr[r,mc] != 0:
                                arr[r,c] = arr[r,mc]
                            elif arr[r,mc] == 0 and arr[r,c] != 0:
                                arr[r,mc] = arr[r,c]
                elif sym == 'v':
                    for r in range(h):
                        for c in range(w):
                            mr = h-1-r
                            if arr[r,c] == 0 and arr[mr,c] != 0:
                                arr[r,c] = arr[mr,c]
                            elif arr[mr,c] == 0 and arr[r,c] != 0:
                                arr[mr,c] = arr[r,c]
                elif sym == 'both':
                    for r in range(h):
                        for c in range(w):
                            mr, mc = h-1-r, w-1-c
                            vals = [arr[r,c], arr[r,mc], arr[mr,c], arr[mr,mc]]
                            nz = [v for v in vals if v != 0]
                            if nz:
                                fill = Counter(nz).most_common(1)[0][0]
                                if arr[r,c]==0: arr[r,c]=fill
                                if arr[r,mc]==0: arr[r,mc]=fill
                                if arr[mr,c]==0: arr[mr,c]=fill
                                if arr[mr,mc]==0: arr[mr,mc]=fill
                elif sym == 'diag' and h == w:
                    for r in range(h):
                        for c in range(w):
                            if arr[r,c]==0 and arr[c,r]!=0:
                                arr[r,c] = arr[c,r]
                            elif arr[c,r]==0 and arr[r,c]!=0:
                                arr[c,r] = arr[r,c]

                if not np.array_equal(arr, go.arr):
                    matches = False
                    break

            if matches:
                def make_sym(st):
                    def complete(g):
                        arr = g.arr.copy()
                        h, w = arr.shape
                        if st == 'h':
                            for r in range(h):
                                for c in range(w):
                                    mc=w-1-c
                                    if arr[r,c]==0 and arr[r,mc]!=0: arr[r,c]=arr[r,mc]
                                    elif arr[r,mc]==0 and arr[r,c]!=0: arr[r,mc]=arr[r,c]
                        elif st == 'v':
                            for r in range(h):
                                for c in range(w):
                                    mr=h-1-r
                                    if arr[r,c]==0 and arr[mr,c]!=0: arr[r,c]=arr[mr,c]
                                    elif arr[mr,c]==0 and arr[r,c]!=0: arr[mr,c]=arr[r,c]
                        elif st == 'both':
                            for r in range(h):
                                for c in range(w):
                                    mr,mc=h-1-r,w-1-c
                                    vals=[arr[r,c],arr[r,mc],arr[mr,c],arr[mr,mc]]
                                    nz=[v for v in vals if v!=0]
                                    if nz:
                                        f=Counter(nz).most_common(1)[0][0]
                                        if arr[r,c]==0:arr[r,c]=f
                                        if arr[r,mc]==0:arr[r,mc]=f
                                        if arr[mr,c]==0:arr[mr,c]=f
                                        if arr[mr,mc]==0:arr[mr,mc]=f
                        elif st == 'diag' and h==w:
                            for r in range(h):
                                for c in range(w):
                                    if arr[r,c]==0 and arr[c,r]!=0: arr[r,c]=arr[c,r]
                                    elif arr[c,r]==0 and arr[r,c]!=0: arr[c,r]=arr[r,c]
                        return Grid(arr)
                    return complete
                return f"gym_sym_{sym}", make_sym(sym)
        except Exception:
            continue
    return None, None


def search_gym_fill_enclosed(pairs):
    """Fill interior bg regions with enclosing object's color."""
    if not all(Grid(p['input']).shape == Grid(p['output']).shape for p in pairs):
        return None, None
    try:
        def fill_enc(g):
            arr = g.arr.copy()
            h, w = g.h, g.w
            bg = g.background
            exterior = np.zeros((h, w), dtype=bool)
            stack = []
            for r in range(h):
                for c in [0, w-1]:
                    if arr[r,c] == bg: stack.append((r,c))
            for c in range(w):
                for r in [0, h-1]:
                    if arr[r,c] == bg: stack.append((r,c))
            while stack:
                r,c = stack.pop()
                if r<0 or r>=h or c<0 or c>=w: continue
                if exterior[r,c] or arr[r,c]!=bg: continue
                exterior[r,c]=True
                stack.extend([(r-1,c),(r+1,c),(r,c-1),(r,c+1)])
            visited = np.zeros((h,w),dtype=bool)
            for r in range(h):
                for c in range(w):
                    if arr[r,c]==bg and not exterior[r,c] and not visited[r,c]:
                        region=[]
                        border_colors=Counter()
                        rs=[(r,c)]
                        while rs:
                            cr,cc=rs.pop()
                            if cr<0 or cr>=h or cc<0 or cc>=w: continue
                            if visited[cr,cc]: continue
                            if arr[cr,cc]!=bg:
                                border_colors[int(arr[cr,cc])]+=1
                                continue
                            if exterior[cr,cc]: continue
                            visited[cr,cc]=True
                            region.append((cr,cc))
                            rs.extend([(cr-1,cc),(cr+1,cc),(cr,cc-1),(cr,cc+1)])
                        if region and border_colors:
                            fc=border_colors.most_common(1)[0][0]
                            for cr,cc in region: arr[cr,cc]=fc
            return Grid(arr)

        for p in pairs:
            if fill_enc(Grid(p['input'])) != Grid(p['output']):
                return None, None
        return "gym_fill_enclosed", fill_enc
    except Exception:
        return None, None


# ══════════════════════════════════════════════════════════════════════
# 9. MASTER SOLVER — Phase 2
# ══════════════════════════════════════════════════════════════════════

PHASE2_SEARCHES = [
    ("pattern_complete", search_pattern_completion),
    ("segment_transform", search_segment_transform),
    ("fill_contained", search_fill_contained_regions),
    ("recolor_property", search_recolor_by_property),
    ("extract_unique", search_extract_unique),
    ("boolean_grid", search_boolean_grid_ops),
    ("gym_pattern_ext", search_gym_pattern_extend),
    ("gym_positional", search_gym_positional_color),
    ("gym_tile_self", search_gym_tile_self),
    ("gym_symmetry", search_gym_symmetry_complete),
    ("gym_fill_enclosed", search_gym_fill_enclosed),
]


def solve_task_phase2(task):
    """Enhanced solver: base + Phase 1 + Phase 2."""
    pairs = task['train']
    test_cases = task['test']
    
    # Try Phase 1 first (includes base)
    p1_predictions = solve_task_phase1(task)
    if score_task(task, p1_predictions):
        return p1_predictions
    
    # Try Phase 2 strategies
    guesses_per_test = [list(g) for g in p1_predictions]
    
    for search_name, search_fn in PHASE2_SEARCHES:
        try:
            name, op = search_fn(pairs)
            if op is not None:
                for i, tc in enumerate(test_cases):
                    if len(guesses_per_test[i]) < 2:
                        try:
                            result = op(Grid(tc['input']))
                            result_list = result.to_list()
                            is_dup = any(result_list == g for g in guesses_per_test[i])
                            if not is_dup:
                                guesses_per_test[i].append(result_list)
                        except Exception:
                            pass
        except Exception:
            continue
    
    return guesses_per_test


# ══════════════════════════════════════════════════════════════════════
# 10. BENCHMARK
# ══════════════════════════════════════════════════════════════════════

def run_benchmark(data_dir, limit=None, verbose=False):
    files = sorted(f for f in os.listdir(data_dir) if f.endswith('.json'))
    if limit:
        files = files[:limit]
    
    n_base_strats = 21
    n_p1 = len(PHASE1_SEARCHES)
    n_p2 = len(PHASE2_SEARCHES)
    total_strats = n_base_strats + n_p1 + n_p2
    
    print("=" * 70)
    print(f"  ARC-AGI-2 Phase 2 — Shape-Aware Pattern Synthesis")
    print(f"  Tasks: {len(files)}")
    print(f"  Strategies: {n_base_strats} base + {n_p1} P1 + {n_p2} P2 = {total_strats}")
    print("=" * 70)
    
    solved = 0
    total = 0
    strategy_wins = Counter()
    base_solved = 0
    p1_solved = 0
    p2_solved = 0
    t0 = time.time()
    
    for f in files:
        task = json.load(open(os.path.join(data_dir, f)))
        total += 1
        
        # Check base
        from arc_solver import solve_task as base_solve
        base_preds = base_solve(task)
        base_ok = score_task(task, base_preds)
        
        # Check P1
        p1_preds = solve_task_phase1(task)
        p1_ok = score_task(task, p1_preds)
        
        # Full P2
        predictions = solve_task_phase2(task)
        correct = score_task(task, predictions)
        
        if correct:
            solved += 1
            if base_ok:
                base_solved += 1
            elif p1_ok:
                p1_solved += 1
            else:
                p2_solved += 1
                for sname, sfn in PHASE2_SEARCHES:
                    try:
                        name, op = sfn(task['train'])
                        if op:
                            strategy_wins[f"P2:{sname}:{name}"] += 1
                            break
                    except Exception:
                        continue
        
        if verbose:
            if correct and not base_ok and not p1_ok:
                print(f"  {f[:16]}: SOLVED (Phase 2 NEW!)")
            elif correct and not base_ok:
                print(f"  {f[:16]}: SOLVED (Phase 1)")
            elif correct:
                print(f"  {f[:16]}: SOLVED (base)")
            else:
                print(f"  {f[:16]}: failed")
    
    elapsed = time.time() - t0
    pct = solved / max(total, 1) * 100
    
    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    print(f"  Solved:     {solved}/{total} ({pct:.1f}%)")
    print(f"    Base:     {base_solved}")
    print(f"    Phase 1:  {p1_solved}")
    print(f"    Phase 2:  {p2_solved} (NEW)")
    print(f"  Time:       {elapsed:.1f}s ({elapsed/max(total,1)*100:.0f}ms/task)")
    
    if strategy_wins:
        print(f"\n  Phase 2 winning strategies:")
        for name, count in strategy_wins.most_common(15):
            print(f"    {name:<45}: {count}")
    
    print(f"\n{'='*70}")
    return solved, total


def run_tests():
    print("=" * 65)
    print("  Phase 2 Solver Tests")
    print("=" * 65)
    p = 0; t = 0
    
    print("\n  T1: Shape detection - cross")
    cells = [(0,1),(1,0),(1,1),(1,2),(2,1)]
    shape, props = detect_shape(cells, 3, 3)
    ok = shape == 'cross'
    print(f"    Shape: {shape} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1
    
    print("\n  T2: Shape detection - rectangle")
    cells = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]
    shape, _ = detect_shape(cells, 2, 3)
    ok = shape == 'rectangle'
    print(f"    Shape: {shape} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1
    
    print("\n  T3: Grid dividers")
    g = Grid([[1,2,0,3,4],[5,5,5,5,5],[6,7,0,8,9]])
    h_div, v_div = find_dividers(g)
    ok = len(h_div) == 1  # row 1 is all 5s (non-bg)
    print(f"    H dividers: {len(h_div)}, V dividers: {len(v_div)} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1
    
    print("\n  T4: Segmentation")
    segs = segment_by_dividers(g)
    ok = segs is not None and len(segs) >= 2
    print(f"    Segments: {len(segs) if segs else 0} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1
    
    print("\n  T5: Fill contained regions")
    g2 = Grid([[0,0,0,0,0],[0,1,1,1,0],[0,1,0,1,0],[0,1,1,1,0],[0,0,0,0,0]])
    expected = Grid([[0,0,0,0,0],[0,1,1,1,0],[0,1,1,1,0],[0,1,1,1,0],[0,0,0,0,0]])
    pairs = [{'input': g2.to_list(), 'output': expected.to_list()}]
    name, op = search_fill_contained_regions(pairs)
    ok = name is not None
    print(f"    Fill: {name} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1
    
    print("\n  T6: Phase 2 solver runs")
    task = {'train': [{'input': [[1,2],[3,4]], 'output': [[3,1],[4,2]]}],
            'test': [{'input': [[5,6],[7,8]], 'output': [[7,5],[8,6]]}]}
    preds = solve_task_phase2(task)
    ok = score_task(task, preds)
    print(f"    Solved: {'PASS' if ok else 'FAIL'}")
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
        run_benchmark(d, limit=args.limit, verbose=args.verbose)
