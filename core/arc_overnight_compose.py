"""
NeMo-WM Overnight Compositional Search
=======================================
Chains 2-3 atomic transform primitives together and brute-force tests
every combination against unsolved ARC-AGI-2 tasks.

This actually runs 1-2 hours and can discover multi-step solvers
that single-step solvers miss.

Usage:
    python arc_overnight_compose.py --data /path/to/ARC-AGI-2/data

Output:
    overnight_compose_results.json — all discovered chains
    overnight_new_solvers.py       — auto-generated solver code
"""
import json
import os
import sys
import time
import numpy as np
import argparse
from collections import Counter, defaultdict
from itertools import product as iterproduct

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arc_object_graph import extract_objects


# ═══════════════════════════════════════════════════════════
# ATOMIC PRIMITIVES — small composable transforms
# ═══════════════════════════════════════════════════════════

def prim_identity(grid):
    return grid.copy()

def prim_fliph(grid):
    return np.fliplr(grid)

def prim_flipv(grid):
    return np.flipud(grid)

def prim_rot90(grid):
    return np.rot90(grid, 1)

def prim_rot180(grid):
    return np.rot90(grid, 2)

def prim_rot270(grid):
    return np.rot90(grid, 3)

def prim_transpose(grid):
    return grid.T

def prim_invert_colors(grid):
    bg = int(np.argmax(np.bincount(grid.flatten())))
    out = grid.copy()
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] == bg:
                # Find most common non-bg
                nz = [int(v) for v in grid.flatten() if v != bg]
                if nz:
                    out[r, c] = Counter(nz).most_common(1)[0][0]
            else:
                out[r, c] = bg
    return out

def prim_remove_bg_rows(grid):
    bg = int(np.argmax(np.bincount(grid.flatten())))
    keep = [r for r in range(grid.shape[0]) if any(grid[r, c] != bg for c in range(grid.shape[1]))]
    return grid[keep, :] if keep else grid

def prim_remove_bg_cols(grid):
    bg = int(np.argmax(np.bincount(grid.flatten())))
    keep = [c for c in range(grid.shape[1]) if any(grid[r, c] != bg for r in range(grid.shape[0]))]
    return grid[:, keep] if keep else grid

def prim_crop_content(grid):
    bg = int(np.argmax(np.bincount(grid.flatten())))
    rows = [r for r in range(grid.shape[0]) if any(grid[r, c] != bg for c in range(grid.shape[1]))]
    cols = [c for c in range(grid.shape[1]) if any(grid[r, c] != bg for r in range(grid.shape[0]))]
    if rows and cols:
        return grid[min(rows):max(rows)+1, min(cols):max(cols)+1]
    return grid

def prim_sort_rows(grid):
    return np.array(sorted(grid.tolist()))

def prim_sort_cols(grid):
    return np.array(sorted(grid.T.tolist())).T

def prim_reverse_rows(grid):
    return grid[::-1]

def prim_reverse_cols(grid):
    return grid[:, ::-1]

def prim_unique_rows(grid):
    seen = []
    result = []
    for r in range(grid.shape[0]):
        row = grid[r].tolist()
        if row not in seen:
            seen.append(row)
            result.append(row)
    return np.array(result) if result else grid

def prim_unique_cols(grid):
    return prim_unique_rows(grid.T).T

def prim_tile_2x2(grid):
    return np.tile(grid, (2, 2))

def prim_tile_2x1(grid):
    return np.tile(grid, (2, 1))

def prim_tile_1x2(grid):
    return np.tile(grid, (1, 2))

def prim_mirror_h(grid):
    return np.hstack([grid, np.fliplr(grid)])

def prim_mirror_v(grid):
    return np.vstack([grid, np.flipud(grid)])

def prim_mirror_hv(grid):
    top = np.hstack([grid, np.fliplr(grid)])
    return np.vstack([top, np.flipud(top)])

def prim_upscale_2x(grid):
    return np.repeat(np.repeat(grid, 2, axis=0), 2, axis=1)

def prim_upscale_3x(grid):
    return np.repeat(np.repeat(grid, 3, axis=0), 3, axis=1)

def prim_downscale_2x(grid):
    h, w = grid.shape
    if h % 2 != 0 or w % 2 != 0:
        return grid
    return grid[::2, ::2]

def prim_downscale_3x(grid):
    h, w = grid.shape
    if h % 3 != 0 or w % 3 != 0:
        return grid
    return grid[::3, ::3]

def prim_top_half(grid):
    return grid[:grid.shape[0]//2, :]

def prim_bottom_half(grid):
    return grid[grid.shape[0]//2:, :]

def prim_left_half(grid):
    return grid[:, :grid.shape[1]//2]

def prim_right_half(grid):
    return grid[:, grid.shape[1]//2:]

def prim_extract_largest_obj(grid):
    objs, _ = extract_objects(grid)
    if not objs:
        return grid
    largest = max(objs, key=lambda o: o.size)
    return largest.crop(grid)

def prim_extract_smallest_obj(grid):
    objs, _ = extract_objects(grid)
    if not objs:
        return grid
    smallest = min(objs, key=lambda o: o.size)
    return smallest.crop(grid)

def prim_fill_enclosed(grid):
    bg = int(np.argmax(np.bincount(grid.flatten())))
    h, w = grid.shape
    out = grid.copy()
    visited = np.zeros((h, w), dtype=bool)
    # BFS from edges to find reachable bg
    from collections import deque
    q = deque()
    for r in range(h):
        for c in [0, w-1]:
            if grid[r, c] == bg:
                q.append((r, c))
                visited[r, c] = True
    for c in range(w):
        for r in [0, h-1]:
            if grid[r, c] == bg:
                q.append((r, c))
                visited[r, c] = True
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and grid[nr, nc] == bg:
                visited[nr, nc] = True
                q.append((nr, nc))
    # Fill unreachable bg
    fill_color = None
    for r in range(h):
        for c in range(w):
            if grid[r, c] != bg and fill_color is None:
                fill_color = int(grid[r, c])
    if fill_color is not None:
        for r in range(h):
            for c in range(w):
                if grid[r, c] == bg and not visited[r, c]:
                    out[r, c] = fill_color
    return out

def prim_sym_complete_h(grid):
    bg = int(np.argmax(np.bincount(grid.flatten())))
    out = grid.copy()
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            mc = w - 1 - c
            if out[r, c] == bg and out[r, mc] != bg:
                out[r, c] = out[r, mc]
            elif out[r, mc] == bg and out[r, c] != bg:
                out[r, mc] = out[r, c]
    return out

def prim_sym_complete_v(grid):
    bg = int(np.argmax(np.bincount(grid.flatten())))
    out = grid.copy()
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            mr = h - 1 - r
            if out[r, c] == bg and out[mr, c] != bg:
                out[r, c] = out[mr, c]
            elif out[mr, c] == bg and out[r, c] != bg:
                out[mr, c] = out[r, c]
    return out

def prim_gravity_down(grid):
    bg = int(np.argmax(np.bincount(grid.flatten())))
    h, w = grid.shape
    out = np.full_like(grid, bg)
    for c in range(w):
        vals = [int(grid[r, c]) for r in range(h) if grid[r, c] != bg]
        for i, v in enumerate(vals):
            out[h - len(vals) + i, c] = v
    return out

def prim_gravity_up(grid):
    bg = int(np.argmax(np.bincount(grid.flatten())))
    h, w = grid.shape
    out = np.full_like(grid, bg)
    for c in range(w):
        vals = [int(grid[r, c]) for r in range(h) if grid[r, c] != bg]
        for i, v in enumerate(vals):
            out[i, c] = v
    return out

def prim_gravity_left(grid):
    return prim_gravity_down(grid.T).T

def prim_gravity_right(grid):
    return prim_gravity_up(grid.T).T

def prim_border_1px(grid):
    bg = int(np.argmax(np.bincount(grid.flatten())))
    nz = [int(v) for v in grid.flatten() if v != bg]
    bc = Counter(nz).most_common(1)[0][0] if nz else 1
    h, w = grid.shape
    out = np.full((h+2, w+2), bc, dtype=int)
    out[1:-1, 1:-1] = grid
    return out

def prim_remove_border(grid):
    if grid.shape[0] < 3 or grid.shape[1] < 3:
        return grid
    return grid[1:-1, 1:-1]


# ═══════════════════════════════════════════════════════════
# PRIMITIVE REGISTRY
# ═══════════════════════════════════════════════════════════

PRIMITIVES = {
    # Geometric
    'fliph': prim_fliph,
    'flipv': prim_flipv,
    'rot90': prim_rot90,
    'rot180': prim_rot180,
    'rot270': prim_rot270,
    'transpose': prim_transpose,
    # Crop/Extract
    'crop': prim_crop_content,
    'rm_bg_rows': prim_remove_bg_rows,
    'rm_bg_cols': prim_remove_bg_cols,
    'top_half': prim_top_half,
    'bot_half': prim_bottom_half,
    'left_half': prim_left_half,
    'right_half': prim_right_half,
    'extract_lg': prim_extract_largest_obj,
    'extract_sm': prim_extract_smallest_obj,
    'rm_border': prim_remove_border,
    # Tiling
    'tile_2x2': prim_tile_2x2,
    'tile_2x1': prim_tile_2x1,
    'tile_1x2': prim_tile_1x2,
    'mirror_h': prim_mirror_h,
    'mirror_v': prim_mirror_v,
    'mirror_hv': prim_mirror_hv,
    # Scale
    'up_2x': prim_upscale_2x,
    'up_3x': prim_upscale_3x,
    'down_2x': prim_downscale_2x,
    'down_3x': prim_downscale_3x,
    # Sorting/Gravity
    'sort_rows': prim_sort_rows,
    'sort_cols': prim_sort_cols,
    'rev_rows': prim_reverse_rows,
    'rev_cols': prim_reverse_cols,
    'grav_down': prim_gravity_down,
    'grav_up': prim_gravity_up,
    'grav_left': prim_gravity_left,
    'grav_right': prim_gravity_right,
    'uniq_rows': prim_unique_rows,
    'uniq_cols': prim_unique_cols,
    # Fill/Symmetry
    'fill_enc': prim_fill_enclosed,
    'sym_h': prim_sym_complete_h,
    'sym_v': prim_sym_complete_v,
    # Border
    'add_border': prim_border_1px,
}

# ═══════════════════════════════════════════════════════════
# NUMERICAL PRIMITIVES — adaptive transforms that COUNT first
# These compute N from the grid, then use N to transform.
# ═══════════════════════════════════════════════════════════

def _bg(grid):
    return int(np.argmax(np.bincount(grid.flatten())))

def _objs(grid):
    try:
        objs, _ = extract_objects(grid)
        return objs
    except:
        return []

def prim_scale_by_n_objects(grid):
    """Scale grid by N where N = number of objects."""
    bg = _bg(grid); objs = _objs(grid); n = len(objs)
    if n < 2 or n > 8: return None
    return np.repeat(np.repeat(grid, n, axis=0), n, axis=1)

def prim_scale_by_n_colors(grid):
    """Scale grid by N where N = number of non-bg colors."""
    bg = _bg(grid); n = len(set(int(v) for v in grid.flatten()) - {bg})
    if n < 2 or n > 8: return None
    return np.repeat(np.repeat(grid, n, axis=0), n, axis=1)

def prim_tile_by_n_objects(grid):
    """Tile grid N times horizontally where N = n_objects."""
    n = len(_objs(grid))
    if n < 2 or n > 8: return None
    return np.tile(grid, (1, n))

def prim_tile_v_by_n_objects(grid):
    """Tile grid N times vertically where N = n_objects."""
    n = len(_objs(grid))
    if n < 2 or n > 8: return None
    return np.tile(grid, (n, 1))

def prim_extract_nth_largest(grid):
    """Extract the Nth largest object where N = n_colors."""
    bg = _bg(grid); objs = _objs(grid)
    n = len(set(int(v) for v in grid.flatten()) - {bg})
    if n < 1 or n > len(objs): return None
    s = sorted(objs, key=lambda o: o.size, reverse=True)
    return s[n-1].crop(grid)

def prim_extract_by_n_objects(grid):
    """Extract the Nth largest object where N = n_objects."""
    objs = _objs(grid); n = len(objs)
    if n < 1: return None
    s = sorted(objs, key=lambda o: o.size, reverse=True)
    idx = min(n-1, len(s)-1)
    return s[idx].crop(grid)

def prim_recolor_by_obj_size(grid):
    """Recolor each object to color = its pixel count (clamped 1-9)."""
    bg = _bg(grid); objs = _objs(grid)
    if len(objs) < 2: return None
    out = grid.copy()
    for o in objs:
        c = min(o.size, 9)
        for r, cc in o.cells:
            out[r, cc] = c
    return out

def prim_recolor_by_size_rank(grid):
    """Recolor objects: smallest→1, next→2, etc."""
    bg = _bg(grid); objs = _objs(grid)
    if len(objs) < 2: return None
    s = sorted(objs, key=lambda o: o.size)
    out = grid.copy()
    for i, o in enumerate(s):
        c = min(i + 1, 9)
        for r, cc in o.cells:
            out[r, cc] = c
    return out

def prim_recolor_by_neighbor_count(grid):
    """Recolor each non-bg pixel by count of same-color 4-neighbors."""
    bg = _bg(grid); h, w = grid.shape
    out = grid.copy()
    for r in range(h):
        for c in range(w):
            if grid[r, c] == bg: continue
            ic = int(grid[r, c])
            nn = sum(1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                     if 0 <= r+dr < h and 0 <= c+dc < w and grid[r+dr, c+dc] == ic)
            out[r, c] = min(nn + 1, 9)  # +1 so isolated pixels → 1 not 0
    return out

def prim_output_1x1_n_objects(grid):
    """Output = 1x1 grid with value = n_objects."""
    n = len(_objs(grid))
    if n < 1 or n > 9: return None
    return np.array([[n]], dtype=int)

def prim_output_1x1_n_colors(grid):
    """Output = 1x1 grid with value = n_colors."""
    bg = _bg(grid)
    n = len(set(int(v) for v in grid.flatten()) - {bg})
    if n < 1 or n > 9: return None
    return np.array([[n]], dtype=int)

def prim_output_NxN_majority(grid):
    """Output = NxN solid grid, N = n_objects, color = majority."""
    bg = _bg(grid); objs = _objs(grid); n = len(objs)
    if n < 1 or n > 15: return None
    nz = [int(v) for v in grid.flatten() if v != bg]
    if not nz: return None
    c = Counter(nz).most_common(1)[0][0]
    return np.full((n, n), c, dtype=int)

def prim_keep_majority_color_objs(grid):
    """Keep only objects whose color is the most common object color."""
    bg = _bg(grid); objs = _objs(grid)
    if len(objs) < 2: return None
    cc = Counter(o.color for o in objs)
    maj = cc.most_common(1)[0][0]
    out = np.full_like(grid, bg)
    for o in objs:
        if o.color == maj:
            for r, c in o.cells:
                out[r, c] = o.color
    return out

def prim_keep_minority_color_objs(grid):
    """Keep only objects whose color is the least common object color."""
    bg = _bg(grid); objs = _objs(grid)
    if len(objs) < 2: return None
    cc = Counter(o.color for o in objs)
    minor = cc.most_common()[-1][0]
    out = np.full_like(grid, bg)
    for o in objs:
        if o.color == minor:
            for r, c in o.cells:
                out[r, c] = o.color
    return out

def prim_downsample_by_n_colors(grid):
    """Downsample by N where N = n_colors."""
    bg = _bg(grid)
    n = len(set(int(v) for v in grid.flatten()) - {bg})
    h, w = grid.shape
    if n < 2 or h % n != 0 or w % n != 0: return None
    return grid[::n, ::n]

def prim_crop_to_NxN_by_objects(grid):
    """Crop to NxN where N = n_objects."""
    n = len(_objs(grid))
    if n < 1 or n > grid.shape[0] or n > grid.shape[1]: return None
    return grid[:n, :n]

def prim_repeat_rows_by_n_colors(grid):
    """Repeat each row N times where N = n_colors."""
    bg = _bg(grid)
    n = len(set(int(v) for v in grid.flatten()) - {bg})
    if n < 2 or n > 8: return None
    return np.repeat(grid, n, axis=0)

def prim_repeat_cols_by_n_colors(grid):
    """Repeat each col N times where N = n_colors."""
    bg = _bg(grid)
    n = len(set(int(v) for v in grid.flatten()) - {bg})
    if n < 2 or n > 8: return None
    return np.repeat(grid, n, axis=1)

# Add numerical primitives to registry
NUMERICAL_PRIMITIVES = {
    'num_scale_objs': prim_scale_by_n_objects,
    'num_scale_colors': prim_scale_by_n_colors,
    'num_tile_h_objs': prim_tile_by_n_objects,
    'num_tile_v_objs': prim_tile_v_by_n_objects,
    'num_ext_nth_lg': prim_extract_nth_largest,
    'num_ext_by_nobj': prim_extract_by_n_objects,
    'num_rcol_size': prim_recolor_by_obj_size,
    'num_rcol_rank': prim_recolor_by_size_rank,
    'num_rcol_nn': prim_recolor_by_neighbor_count,
    'num_1x1_objs': prim_output_1x1_n_objects,
    'num_1x1_colors': prim_output_1x1_n_colors,
    'num_NxN_maj': prim_output_NxN_majority,
    'num_keep_maj': prim_keep_majority_color_objs,
    'num_keep_min': prim_keep_minority_color_objs,
    'num_down_colors': prim_downsample_by_n_colors,
    'num_crop_NxN': prim_crop_to_NxN_by_objects,
    'num_rep_rows_c': prim_repeat_rows_by_n_colors,
    'num_rep_cols_c': prim_repeat_cols_by_n_colors,
}

PRIMITIVES.update(NUMERICAL_PRIMITIVES)

PRIM_NAMES = list(PRIMITIVES.keys())
N_PRIMS = len(PRIM_NAMES)


# ═══════════════════════════════════════════════════════════
# CHAIN TESTER
# ═══════════════════════════════════════════════════════════

def apply_chain(grid, chain):
    """Apply a chain of primitives to a grid."""
    result = grid.copy()
    for pname in chain:
        try:
            fn = PRIMITIVES[pname]
            result = fn(result)
            if result is None or not isinstance(result, np.ndarray) or result.size == 0 or result.size > 10000:
                return None
        except:
            return None
    return result


def test_chain_on_task(chain, task):
    """Test if a chain solves all training pairs + test."""
    pairs = task['train']
    for p in pairs:
        gi = np.array(p['input'])
        go = np.array(p['output'])
        result = apply_chain(gi, chain)
        if result is None or result.shape != go.shape or not np.array_equal(result, go):
            return False
    return True


def score_chain_on_task(chain, task):
    """Test chain including test output."""
    if not test_chain_on_task(chain, task):
        return False
    for tc in task['test']:
        gi = np.array(tc['input'])
        go = np.array(tc['output'])
        result = apply_chain(gi, chain)
        if result is None or result.shape != go.shape or not np.array_equal(result, go):
            return False
    return True


# ═══════════════════════════════════════════════════════════
# SEARCH ENGINE
# ═══════════════════════════════════════════════════════════

def search_depth_1(tasks, known_solved):
    """Try all single primitives."""
    found = {}
    for pname in PRIM_NAMES:
        chain = [pname]
        for tid, task in tasks.items():
            if tid in known_solved or tid in found:
                continue
            try:
                if score_chain_on_task(chain, task):
                    found[tid] = chain[:]
            except:
                pass
    return found


def search_depth_2(tasks, known_solved, progress_every=1000):
    """Try all pairs of primitives."""
    found = {}
    unsolved = {tid: task for tid, task in tasks.items() if tid not in known_solved}
    total_combos = N_PRIMS * N_PRIMS
    tested = 0
    
    for p1 in PRIM_NAMES:
        for p2 in PRIM_NAMES:
            chain = [p1, p2]
            tested += 1
            
            if tested % progress_every == 0:
                print(f"    Depth 2: {tested}/{total_combos} combos | Found: {len(found)}")
            
            for tid, task in list(unsolved.items()):
                if tid in found:
                    continue
                try:
                    if score_chain_on_task(chain, task):
                        found[tid] = chain[:]
                        print(f"  ★ {tid}: {' → '.join(chain)}")
                except:
                    pass
    
    return found


def search_depth_3(tasks, known_solved, progress_every=5000):
    """Try all triples of primitives (this is the long one)."""
    found = {}
    unsolved = {tid: task for tid, task in tasks.items() if tid not in known_solved}
    
    # Pre-filter: only try chains where first step produces valid output on first pair
    total_combos = N_PRIMS * N_PRIMS * N_PRIMS
    tested = 0
    
    # Smart pruning: for each task, cache which single-step outputs are valid
    print(f"    Depth 3: {total_combos} total combos, {len(unsolved)} unsolved tasks")
    print(f"    Estimated time: {total_combos * len(unsolved) * 0.00001:.0f}s")
    
    for p1 in PRIM_NAMES:
        for p2 in PRIM_NAMES:
            # Quick check: does p1→p2 produce valid grids on ANY task?
            any_valid = False
            for tid, task in list(unsolved.items())[:10]:
                try:
                    gi = np.array(task['train'][0]['input'])
                    r1 = PRIMITIVES[p1](gi)
                    if r1 is not None and r1.size > 0 and r1.size < 10000:
                        r2 = PRIMITIVES[p2](r1)
                        if r2 is not None and r2.size > 0 and r2.size < 10000:
                            any_valid = True
                            break
                except:
                    pass
            
            if not any_valid:
                tested += N_PRIMS
                continue
            
            for p3 in PRIM_NAMES:
                chain = [p1, p2, p3]
                tested += 1
                
                if tested % progress_every == 0:
                    print(f"    Depth 3: {tested}/{total_combos} ({100*tested/total_combos:.1f}%) | Found: {len(found)}")
                
                for tid, task in list(unsolved.items()):
                    if tid in found:
                        continue
                    try:
                        if score_chain_on_task(chain, task):
                            found[tid] = chain[:]
                            print(f"  ★★ {tid}: {' → '.join(chain)}")
                    except:
                        pass
    
    return found


# ═══════════════════════════════════════════════════════════
# CODE GENERATOR — turn discovered chains into solver code
# ═══════════════════════════════════════════════════════════

def generate_solver_code(discoveries, output_path):
    """Auto-generate Python solver code for discovered chains."""
    lines = [
        '"""',
        'Auto-generated compositional solvers from overnight search.',
        f'Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}',
        f'Discovered: {len(discoveries)} new chains',
        '"""',
        'import numpy as np',
        'from arc_overnight_compose import PRIMITIVES, apply_chain, score_chain_on_task',
        '',
        'DISCOVERED_CHAINS = {'
    ]
    
    for tid, chain in sorted(discoveries.items()):
        lines.append(f'    "{tid}": {chain},')
    
    lines.append('}')
    lines.append('')
    lines.append('')
    lines.append('def try_discovered_chains(task):')
    lines.append('    """Try all auto-discovered compositional chains."""')
    lines.append('    for tid, chain in DISCOVERED_CHAINS.items():')
    lines.append('        try:')
    lines.append('            if score_chain_on_task(chain, task):')
    lines.append('                def mk(ch=chain):')
    lines.append('                    def apply(t):')
    lines.append('                        gs = []')
    lines.append('                        for tc in t["test"]:')
    lines.append('                            result = apply_chain(np.array(tc["input"]), ch)')
    lines.append('                            gs.append([result.tolist()])')
    lines.append('                        return gs')
    lines.append('                    return apply')
    lines.append('                result = mk()(task)')
    lines.append('                return result, f"COMPOSE:{\"→\".join(chain)}"')
    lines.append('        except:')
    lines.append('            pass')
    lines.append('    return None, None')
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"  Generated solver code: {output_path}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='NeMo-WM Overnight Compositional Search')
    parser.add_argument('--data', required=True, help='Path to ARC-AGI-2 data directory')
    parser.add_argument('--max-depth', type=int, default=3, help='Max chain depth (2=fast, 3=thorough)')
    parser.add_argument('--output', default='overnight_compose_results.json', help='Output JSON')
    args = parser.parse_args()
    
    print("="*60)
    print("  NeMo-WM Overnight Compositional Search")
    print(f"  Primitives: {N_PRIMS}")
    print(f"  Depth 1: {N_PRIMS} chains")
    print(f"  Depth 2: {N_PRIMS**2} chains")
    print(f"  Depth 3: {N_PRIMS**3} chains")
    print(f"  Max depth: {args.max_depth}")
    print("="*60)
    
    # Load tasks
    training_dir = os.path.join(args.data, 'training')
    tasks = {}
    for f in sorted(os.listdir(training_dir)):
        if f.endswith('.json'):
            tasks[f.replace('.json', '')] = json.load(open(os.path.join(training_dir, f)))
    print(f"\nLoaded {len(tasks)} tasks")
    
    # Load known solved (from our existing system)
    known_solved = set()
    # Run existing solvers first to establish baseline
    print(f"\nRunning existing OG solvers to find baseline...")
    from arc_object_graph import ALL_OG_SOLVERS
    t0 = time.time()
    for tid, task in tasks.items():
        for sname, sfn in ALL_OG_SOLVERS:
            try:
                result, method = sfn(task)
                if result:
                    # Quick check
                    ok = True
                    for i, tc in enumerate(task['test']):
                        if i >= len(result):
                            ok = False; break
                        pred = result[i]
                        if isinstance(pred, list) and pred and isinstance(pred[0], list) and isinstance(pred[0][0], list):
                            pred = pred[0]
                        if pred != tc['output']:
                            ok = False; break
                    if ok:
                        known_solved.add(tid)
                        break
            except:
                pass
    print(f"  Baseline: {len(known_solved)} solved in {time.time()-t0:.1f}s")
    
    all_found = {}
    t_total = time.time()
    
    # Depth 1
    print(f"\n{'─'*50}")
    print(f"DEPTH 1: Testing {N_PRIMS} single primitives")
    print(f"{'─'*50}")
    t0 = time.time()
    d1 = search_depth_1(tasks, known_solved | set(all_found.keys()))
    all_found.update(d1)
    print(f"  Depth 1: {len(d1)} new ({time.time()-t0:.1f}s)")
    
    # Depth 2
    print(f"\n{'─'*50}")
    print(f"DEPTH 2: Testing {N_PRIMS**2} pairs")
    print(f"{'─'*50}")
    t0 = time.time()
    d2 = search_depth_2(tasks, known_solved | set(all_found.keys()))
    all_found.update(d2)
    print(f"  Depth 2: {len(d2)} new ({time.time()-t0:.1f}s)")
    
    # Depth 3 (the long one)
    if args.max_depth >= 3:
        print(f"\n{'─'*50}")
        print(f"DEPTH 3: Testing up to {N_PRIMS**3} triples (with pruning)")
        print(f"{'─'*50}")
        t0 = time.time()
        d3 = search_depth_3(tasks, known_solved | set(all_found.keys()))
        all_found.update(d3)
        print(f"  Depth 3: {len(d3)} new ({time.time()-t0:.1f}s)")
    
    total_time = time.time() - t_total
    
    # Results
    print(f"\n{'='*60}")
    print(f"  COMPOSITIONAL SEARCH COMPLETE")
    print(f"  Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Baseline (existing solvers): {len(known_solved)}")
    print(f"  New from composition: {len(all_found)}")
    print(f"  New total potential: {len(known_solved) + len(all_found)}")
    print(f"{'='*60}")
    
    for tid, chain in sorted(all_found.items()):
        print(f"  {tid}: {' → '.join(chain)}")
    
    # Save results
    results = {
        'meta': {
            'n_primitives': N_PRIMS,
            'max_depth': args.max_depth,
            'total_time_s': round(total_time, 1),
            'baseline_solved': len(known_solved),
            'new_found': len(all_found),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        },
        'baseline_solved': sorted(known_solved),
        'discovered': {tid: chain for tid, chain in sorted(all_found.items())},
    }
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {args.output}")
    
    # Generate solver code
    if all_found:
        code_path = args.output.replace('.json', '_solvers.py')
        generate_solver_code(all_found, code_path)
    
    print(f"\n  Done! Share {args.output} next session for integration.")


if __name__ == '__main__':
    main()
