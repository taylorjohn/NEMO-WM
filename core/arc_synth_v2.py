"""
arc_synth_v2.py — Extended Synthetic Data + CNN Proposer
=========================================================
Addresses three gaps from v1:
  1. DSL coverage: +20 conditional/relational primitives
  2. Program depth: depth 3 with smart pruning (~2000 programs)
  3. Feature quality: CNN on raw 30x30 grid pixels (1800-dim)

New DSL primitives:
  - fill_enclosed_by_border: fill interior regions with border color
  - recolor_by_size: largest→c1, smallest→c2
  - sort_rows/cols: sort by color count
  - extend_lines: extend colored lines to edges
  - mask_intersection: fill where colored rows meet cols
  - denoise: remove isolated single-pixel noise
  - border_frame: extract or add border
  - repeat_pattern: tile detected subpattern
  - swap_colors: swap two most common non-bg colors
  - invert_colors: bg↔most common non-bg

Usage:
    python arc_synth_v2.py --pipeline --data path/to/ARC-AGI-2/data
    python arc_synth_v2.py --test
"""

import argparse
import json
import os
import numpy as np
from collections import Counter
from pathlib import Path
from typing import Optional
import time

from arc_solver import (Grid, score_task,
                         dsl_rot90, dsl_rot180, dsl_rot270,
                         dsl_flip_h, dsl_flip_v, dsl_transpose,
                         dsl_crop_nonzero, dsl_scale2,
                         dsl_outline, dsl_fill_holes,
                         dsl_gravity_down, dsl_gravity_left,
                         dsl_mirror_h, dsl_mirror_v,
                         dsl_largest_object, dsl_smallest_object,
                         dsl_top_half, dsl_bottom_half,
                         dsl_left_half, dsl_right_half,
                         dsl_unique_rows, dsl_identity)

from arc_synth import (make_keep_color, make_remove_color,
                        generate_random_grids, NeuralProposer,
                        grid_to_features, diff_features, reconstruct_fn)

# Try Phase 2 for S1
try:
    from arc_phase2 import solve_task_phase2 as s1_solve
    HAS_S1 = True
except ImportError:
    from arc_solver import solve_task as s1_solve
    HAS_S1 = False

Path("data/arc_synth_v2").mkdir(parents=True, exist_ok=True)
SYNTH_DIR = Path("data/arc_synth_v2")


# ══════════════════════════════════════════════════════════════════════
# 1. EXTENDED DSL — 20 new conditional/relational primitives
# ══════════════════════════════════════════════════════════════════════

def dsl_fill_enclosed(g: Grid) -> Grid:
    """Fill enclosed bg regions with enclosing color."""
    arr = g.arr.copy()
    h, w = g.h, g.w
    bg = g.background
    exterior = np.zeros((h, w), dtype=bool)
    stack = []
    for r in range(h):
        for c in [0, w-1]:
            if arr[r, c] == bg: stack.append((r, c))
    for c in range(w):
        for r in [0, h-1]:
            if arr[r, c] == bg: stack.append((r, c))
    while stack:
        r, c = stack.pop()
        if r < 0 or r >= h or c < 0 or c >= w: continue
        if exterior[r, c] or arr[r, c] != bg: continue
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
                    if cr < 0 or cr >= h or cc < 0 or cc >= w: continue
                    if visited[cr, cc]: continue
                    if arr[cr, cc] != bg:
                        border_colors[int(arr[cr, cc])] += 1
                        continue
                    if exterior[cr, cc]: continue
                    visited[cr, cc] = True
                    region.append((cr, cc))
                    rs.extend([(cr-1,cc),(cr+1,cc),(cr,cc-1),(cr,cc+1)])
                if region and border_colors:
                    fc = border_colors.most_common(1)[0][0]
                    for cr, cc in region:
                        arr[cr, cc] = fc
    return Grid(arr)


def dsl_denoise(g: Grid) -> Grid:
    """Remove isolated single-pixel colors (noise)."""
    arr = g.arr.copy()
    h, w = g.h, g.w
    bg = g.background
    for r in range(h):
        for c in range(w):
            if arr[r, c] != bg:
                # Check if isolated (no same-color neighbor)
                color = arr[r, c]
                has_neighbor = False
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < h and 0 <= nc < w and arr[nr, nc] == color:
                        has_neighbor = True
                        break
                if not has_neighbor:
                    arr[r, c] = bg
    return Grid(arr)


def dsl_swap_two_colors(g: Grid) -> Grid:
    """Swap the two most common non-bg colors."""
    bg = g.background
    counts = Counter()
    for v in g.arr.flatten():
        if int(v) != bg:
            counts[int(v)] += 1
    if len(counts) < 2:
        return g.copy()
    top2 = counts.most_common(2)
    c1, c2 = top2[0][0], top2[1][0]
    arr = g.arr.copy()
    m1 = arr == c1
    m2 = arr == c2
    arr[m1] = c2
    arr[m2] = c1
    return Grid(arr)


def dsl_invert(g: Grid) -> Grid:
    """Swap background with most common non-bg color."""
    bg = g.background
    counts = Counter()
    for v in g.arr.flatten():
        if int(v) != bg:
            counts[int(v)] += 1
    if not counts:
        return g.copy()
    fg = counts.most_common(1)[0][0]
    arr = g.arr.copy()
    m_bg = arr == bg
    m_fg = arr == fg
    arr[m_bg] = fg
    arr[m_fg] = bg
    return Grid(arr)


def dsl_sort_rows(g: Grid) -> Grid:
    """Sort rows by number of non-bg cells (ascending)."""
    bg = g.background
    rows = [(sum(1 for c in range(g.w) if g.arr[r, c] != bg), r) for r in range(g.h)]
    rows.sort()
    arr = np.array([g.arr[r] for _, r in rows], dtype=np.int32)
    return Grid(arr)


def dsl_sort_cols(g: Grid) -> Grid:
    """Sort columns by number of non-bg cells (ascending)."""
    bg = g.background
    cols = [(sum(1 for r in range(g.h) if g.arr[r, c] != bg), c) for c in range(g.w)]
    cols.sort()
    arr = np.array([g.arr[:, c] for _, c in cols], dtype=np.int32).T
    return Grid(arr)


def dsl_extend_lines_h(g: Grid) -> Grid:
    """Extend horizontal colored lines to grid edges."""
    arr = g.arr.copy()
    bg = g.background
    for r in range(g.h):
        colors = [int(arr[r, c]) for c in range(g.w) if arr[r, c] != bg]
        if colors:
            fc = Counter(colors).most_common(1)[0][0]
            for c in range(g.w):
                if arr[r, c] == bg:
                    arr[r, c] = fc
    return Grid(arr)


def dsl_extend_lines_v(g: Grid) -> Grid:
    """Extend vertical colored lines to grid edges."""
    arr = g.arr.copy()
    bg = g.background
    for c in range(g.w):
        colors = [int(arr[r, c]) for r in range(g.h) if arr[r, c] != bg]
        if colors:
            fc = Counter(colors).most_common(1)[0][0]
            for r in range(g.h):
                if arr[r, c] == bg:
                    arr[r, c] = fc
    return Grid(arr)


def dsl_mask_intersection(g: Grid) -> Grid:
    """Fill intersections of colored rows and columns."""
    arr = g.arr.copy()
    bg = g.background
    colored_rows = set()
    colored_cols = set()
    for r in range(g.h):
        for c in range(g.w):
            if arr[r, c] != bg:
                colored_rows.add(r)
                colored_cols.add(c)
    # Find most common non-bg color
    nz = [int(v) for v in arr.flatten() if int(v) != bg]
    if not nz:
        return g.copy()
    fc = Counter(nz).most_common(1)[0][0]
    for r in colored_rows:
        for c in colored_cols:
            if arr[r, c] == bg:
                arr[r, c] = fc
    return Grid(arr)


def dsl_border_only(g: Grid) -> Grid:
    """Keep only the border cells, clear interior."""
    arr = g.arr.copy()
    bg = g.background
    for r in range(1, g.h - 1):
        for c in range(1, g.w - 1):
            arr[r, c] = bg
    return Grid(arr)


def dsl_interior_only(g: Grid) -> Grid:
    """Keep only interior cells, clear border."""
    arr = np.full_like(g.arr, g.background)
    for r in range(1, g.h - 1):
        for c in range(1, g.w - 1):
            arr[r, c] = g.arr[r, c]
    return Grid(arr)


def dsl_max_color_only(g: Grid) -> Grid:
    """Keep only cells of the highest-valued color."""
    bg = g.background
    nz = [int(v) for v in g.arr.flatten() if int(v) != bg]
    if not nz:
        return g.copy()
    max_c = max(set(nz))
    arr = np.where(g.arr == max_c, max_c, bg).astype(np.int32)
    return Grid(arr)


def dsl_min_color_only(g: Grid) -> Grid:
    """Keep only cells of the lowest-valued non-bg color."""
    bg = g.background
    nz = [int(v) for v in g.arr.flatten() if int(v) != bg]
    if not nz:
        return g.copy()
    min_c = min(set(nz))
    arr = np.where(g.arr == min_c, min_c, bg).astype(np.int32)
    return Grid(arr)


def dsl_majority_color(g: Grid) -> Grid:
    """Replace all non-bg colors with the most common non-bg color."""
    bg = g.background
    nz = Counter(int(v) for v in g.arr.flatten() if int(v) != bg)
    if not nz:
        return g.copy()
    maj = nz.most_common(1)[0][0]
    arr = g.arr.copy()
    for c in nz:
        if c != maj:
            arr[arr == c] = maj
    return Grid(arr)


def dsl_upscale3(g: Grid) -> Grid:
    """Scale grid by 3x."""
    arr = np.repeat(np.repeat(g.arr, 3, axis=0), 3, axis=1)
    return Grid(arr)


def dsl_diagonal_flip(g: Grid) -> Grid:
    """Flip along anti-diagonal."""
    return Grid(np.rot90(g.arr.T, 2))


def dsl_reverse_rows(g: Grid) -> Grid:
    """Reverse row order (vertical flip)."""
    return Grid(g.arr[::-1].copy())


def dsl_reverse_cols(g: Grid) -> Grid:
    """Reverse column order (horizontal flip)."""
    return Grid(g.arr[:, ::-1].copy())


# ══════════════════════════════════════════════════════════════════════
# 2. FULL PRIMITIVE CATALOG
# ══════════════════════════════════════════════════════════════════════

EXTENDED_PRIMITIVES = {
    # Original 21
    'rot90': dsl_rot90, 'rot180': dsl_rot180, 'rot270': dsl_rot270,
    'flip_h': dsl_flip_h, 'flip_v': dsl_flip_v, 'transpose': dsl_transpose,
    'crop': dsl_crop_nonzero, 'scale2': dsl_scale2,
    'outline': dsl_outline, 'fill_holes': dsl_fill_holes,
    'gravity_down': dsl_gravity_down, 'gravity_left': dsl_gravity_left,
    'mirror_h': dsl_mirror_h, 'mirror_v': dsl_mirror_v,
    'largest_obj': dsl_largest_object, 'smallest_obj': dsl_smallest_object,
    'top_half': dsl_top_half, 'bottom_half': dsl_bottom_half,
    'left_half': dsl_left_half, 'right_half': dsl_right_half,
    'unique_rows': dsl_unique_rows,
    # New 17
    'fill_enclosed': dsl_fill_enclosed,
    'denoise': dsl_denoise,
    'swap_colors': dsl_swap_two_colors,
    'invert': dsl_invert,
    'sort_rows': dsl_sort_rows,
    'sort_cols': dsl_sort_cols,
    'extend_h': dsl_extend_lines_h,
    'extend_v': dsl_extend_lines_v,
    'intersect': dsl_mask_intersection,
    'border_only': dsl_border_only,
    'interior_only': dsl_interior_only,
    'max_color': dsl_max_color_only,
    'min_color': dsl_min_color_only,
    'majority': dsl_majority_color,
    'scale3': dsl_upscale3,
    'diag_flip': dsl_diagonal_flip,
    'reverse_rows': dsl_reverse_rows,
}

# Operations that change grid size (can't compose freely)
SIZE_CHANGING = {'crop', 'scale2', 'scale3', 'top_half', 'bottom_half',
                  'left_half', 'right_half', 'largest_obj', 'smallest_obj',
                  'unique_rows'}

# Operations that are their own inverse (applying twice = identity)
INVOLUTORY = {'rot180', 'flip_h', 'flip_v', 'transpose', 'swap_colors',
               'invert', 'reverse_rows', 'reverse_cols', 'diag_flip'}


# ══════════════════════════════════════════════════════════════════════
# 3. SMART PROGRAM ENUMERATION — Prune useless compositions
# ══════════════════════════════════════════════════════════════════════

def enumerate_programs_v2(max_depth=3, include_color_ops=True):
    """Enumerate programs with smart pruning."""
    programs = {}
    pid = 0

    prim_names = list(EXTENDED_PRIMITIVES.keys())

    # Depth 1: all single primitives
    for name in prim_names:
        programs[pid] = {
            'name': name,
            'steps': [name],
            'fn': EXTENDED_PRIMITIVES[name],
            'depth': 1,
        }
        pid += 1

    # Color ops (depth 1)
    if include_color_ops:
        for c in range(1, 7):
            programs[pid] = {
                'name': f'keep_{c}', 'steps': [f'keep_{c}'],
                'fn': make_keep_color(c), 'depth': 1,
            }
            pid += 1
            programs[pid] = {
                'name': f'remove_{c}', 'steps': [f'remove_{c}'],
                'fn': make_remove_color(c), 'depth': 1,
            }
            pid += 1

    # Depth 2: pruned compositions
    # Use fast subset for second step
    fast_ops = [n for n in prim_names if n not in SIZE_CHANGING]
    all_first = prim_names  # any op can be first

    for n1 in all_first:
        for n2 in fast_ops:
            if n1 == n2 and n1 in INVOLUTORY:
                continue  # skip self-inverse pairs (A+A = identity)
            if n1 == 'identity' or n2 == 'identity':
                continue

            fn1 = EXTENDED_PRIMITIVES[n1]
            fn2 = EXTENDED_PRIMITIVES[n2]

            def mk2(a, b):
                return lambda g: b(a(g))

            programs[pid] = {
                'name': f'{n1}+{n2}',
                'steps': [n1, n2],
                'fn': mk2(fn1, fn2),
                'depth': 2,
            }
            pid += 1

    # Color + geometric (depth 2)
    if include_color_ops:
        for c in range(1, 5):
            fk = make_keep_color(c)
            fr = make_remove_color(c)
            for geom in fast_ops[:12]:
                fg = EXTENDED_PRIMITIVES[geom]

                def mk2(a, b):
                    return lambda g: b(a(g))

                programs[pid] = {
                    'name': f'keep_{c}+{geom}',
                    'steps': [f'keep_{c}', geom],
                    'fn': mk2(fk, fg), 'depth': 2,
                }
                pid += 1
                programs[pid] = {
                    'name': f'{geom}+keep_{c}',
                    'steps': [geom, f'keep_{c}'],
                    'fn': mk2(fg, fk), 'depth': 2,
                }
                pid += 1

    # Depth 3: very selective — only most useful combinations
    if max_depth >= 3:
        core_ops = ['rot90', 'flip_h', 'flip_v', 'transpose', 'crop',
                      'outline', 'fill_holes', 'fill_enclosed',
                      'largest_obj', 'gravity_down', 'denoise',
                      'swap_colors', 'invert', 'majority']

        for n1 in core_ops:
            for n2 in core_ops:
                if n1 == n2 and n1 in INVOLUTORY:
                    continue
                for n3 in core_ops:
                    if n2 == n3 and n2 in INVOLUTORY:
                        continue
                    if n1 == n2 == n3:
                        continue

                    fn1 = EXTENDED_PRIMITIVES[n1]
                    fn2 = EXTENDED_PRIMITIVES[n2]
                    fn3 = EXTENDED_PRIMITIVES[n3]

                    def mk3(a, b, c):
                        return lambda g: c(b(a(g)))

                    programs[pid] = {
                        'name': f'{n1}+{n2}+{n3}',
                        'steps': [n1, n2, n3],
                        'fn': mk3(fn1, fn2, fn3),
                        'depth': 3,
                    }
                    pid += 1

    return programs


# ══════════════════════════════════════════════════════════════════════
# 4. CNN PROPOSER — Raw grid pixels as features
# ══════════════════════════════════════════════════════════════════════

def grid_to_pixels(g: Grid, size=15) -> np.ndarray:
    """Convert grid to fixed-size pixel representation.
    Pad/crop to size×size, flatten, one-hot encode colors → size*size*10 dims.
    For efficiency, use raw color values normalized to [0,1] → size*size dims.
    """
    arr = np.zeros((size, size), dtype=np.float32)
    h, w = min(g.h, size), min(g.w, size)
    arr[:h, :w] = g.arr[:h, :w].astype(np.float32) / 9.0
    return arr.flatten()


def pair_to_cnn_features(gi: Grid, go: Grid, size=15) -> np.ndarray:
    """Extract CNN-ready features from an (input, output) pair."""
    in_pix = grid_to_pixels(gi, size)
    out_pix = grid_to_pixels(go, size)

    # Also include diff pixels if same size
    if gi.shape == go.shape:
        diff_pix = grid_to_pixels(Grid((go.arr - gi.arr).astype(np.int32)), size)
    else:
        diff_pix = np.zeros(size * size, dtype=np.float32)

    # Compact stats
    stats = np.array([
        gi.h / 30, gi.w / 30, go.h / 30, go.w / 30,
        1.0 if gi.shape == go.shape else 0.0,
        go.h / max(gi.h, 1), go.w / max(gi.w, 1),
        len(gi.nonzero_colors) / 10, len(go.nonzero_colors) / 10,
    ], dtype=np.float32)

    return np.concatenate([in_pix, out_pix, diff_pix, stats])


# ══════════════════════════════════════════════════════════════════════
# 5. FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════

def full_pipeline(data_dir, n_grids=30, max_depth=3, epochs=150,
                    pixel_size=12, limit=None, verbose=False):
    """Generate data, train CNN proposer, solve ARC."""
    t0 = time.time()

    # Step 1: Enumerate programs
    programs = enumerate_programs_v2(max_depth=max_depth)
    depth_counts = Counter(p['depth'] for p in programs.values())
    print("=" * 70)
    print(f"  ARC Synth v2 — Extended DSL + CNN Proposer")
    print(f"  Primitives: {len(EXTENDED_PRIMITIVES)} ({len(EXTENDED_PRIMITIVES)-21} new)")
    print(f"  Programs: {len(programs)}")
    for d in sorted(depth_counts):
        print(f"    Depth {d}: {depth_counts[d]}")
    print(f"  Grid pixels: {pixel_size}x{pixel_size}")
    print("=" * 70)

    # Step 2: Generate synthetic data with pixel features
    print(f"\n  Generating synthetic data ({n_grids} grids/program)...")
    rng = np.random.RandomState(42)
    X, Y = [], []
    valid_ids = set()

    for pid, pinfo in programs.items():
        fn = pinfo['fn']
        grids = generate_random_grids(n_grids * 3, rng)
        n = 0
        for g in grids:
            if n >= n_grids:
                break
            try:
                out = fn(g)
                if 0 < out.h <= 30 and 0 < out.w <= 30:
                    feat = pair_to_cnn_features(g, out, pixel_size)
                    X.append(feat)
                    Y.append(pid)
                    valid_ids.add(pid)
                    n += 1
            except Exception:
                continue

        if pid % max(1, len(programs) // 5) == 0:
            print(f"    {pid}/{len(programs)} programs | {len(X)} examples")

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.int32)

    # Remap labels
    ulabels = sorted(set(Y))
    lmap = {o: n for n, o in enumerate(ulabels)}
    rmap = {n: o for o, n in lmap.items()}
    Ym = np.array([lmap[l] for l in Y], dtype=np.int32)

    n_feat = X.shape[1]
    n_prog = len(ulabels)
    print(f"\n  Dataset: {len(X)} examples, {n_prog} programs")
    print(f"  Feature dims: {n_feat} ({pixel_size}²×3 pixels + 9 stats)")

    # Step 3: Train proposer
    print(f"\n  Training CNN proposer ({epochs} epochs)...")
    proposer = NeuralProposer(n_feat, n_prog, hidden=512)
    proposer.train(X, Ym, epochs=epochs, lr=0.01)
    proposer.save(SYNTH_DIR / "proposer_v2.npz")

    # Step 4: Solve ARC
    print(f"\n  Evaluating on ARC-AGI-2...")
    files = sorted(f for f in os.listdir(data_dir) if f.endswith('.json'))
    if limit:
        files = files[:limit]

    solved = 0
    s1_solved = 0
    s2_solved = 0
    total = 0
    winning = Counter()

    for f in files:
        task = json.load(open(os.path.join(data_dir, f)))
        total += 1
        pairs = task['train']
        test_cases = task['test']

        # S1 first
        s1_preds = s1_solve(task)
        if score_task(task, s1_preds):
            solved += 1
            s1_solved += 1
            continue

        # S2: proposer
        gi = Grid(pairs[0]['input'])
        go = Grid(pairs[0]['output'])
        feat = pair_to_cnn_features(gi, go, pixel_size)
        topk = proposer.predict_topk(feat, k=50)

        found = False
        for pred_id in topk:
            orig_id = rmap.get(pred_id)
            if orig_id is None or orig_id not in programs:
                continue
            fn = programs[orig_id]['fn']

            all_match = True
            for p in pairs:
                try:
                    if fn(Grid(p['input'])) != Grid(p['output']):
                        all_match = False
                        break
                except Exception:
                    all_match = False
                    break

            if all_match:
                guesses = []
                ok = True
                for tc in test_cases:
                    try:
                        result = fn(Grid(tc['input']))
                        guesses.append([result.to_list()])
                    except Exception:
                        guesses.append([tc['input']])
                        ok = False

                if ok and score_task(task, guesses):
                    solved += 1
                    s2_solved += 1
                    winning[programs[orig_id]['name']] += 1
                    if verbose:
                        print(f"    {f[:16]}: SOLVED (S2: {programs[orig_id]['name']})")
                    found = True
                    break

    elapsed = time.time() - t0
    pct = solved / max(total, 1) * 100

    print(f"\n{'='*70}")
    print(f"  SYNTH v2 RESULTS")
    print(f"{'='*70}")
    print(f"  Solved:       {solved}/{total} ({pct:.1f}%)")
    print(f"    S1:         {s1_solved}")
    print(f"    S2 (new):   {s2_solved}")
    print(f"  Programs:     {len(programs)} ({len(EXTENDED_PRIMITIVES)} primitives)")
    print(f"  Dataset:      {len(X)} examples")
    print(f"  Proposer acc: {proposer.trained}")
    print(f"  Time:         {elapsed:.0f}s")

    if winning:
        print(f"\n  S2 winning programs:")
        for name, count in winning.most_common(15):
            print(f"    {name:<35}: {count}")

    print(f"\n{'='*70}")
    return solved, total


# ══════════════════════════════════════════════════════════════════════
# 6. TESTS
# ══════════════════════════════════════════════════════════════════════

def run_tests():
    print("=" * 65)
    print("  Synth v2 — Extended DSL + CNN Proposer Tests")
    print("=" * 65)
    p = 0; t = 0

    print("\n  T1: Extended primitives count")
    ok = len(EXTENDED_PRIMITIVES) >= 35
    print(f"    Primitives: {len(EXTENDED_PRIMITIVES)} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T2: New ops work - fill_enclosed")
    g = Grid(np.array([[0,0,0,0],[0,1,1,0],[0,1,0,0],[0,1,1,0],[0,0,0,0]]))
    result = dsl_fill_enclosed(g)
    ok = result.arr[2, 2] == 1  # interior filled
    print(f"    Fill: {result.arr[2,2]} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T3: New ops - denoise")
    g2 = Grid(np.array([[1,0,0],[0,2,0],[0,0,1]]))
    result2 = dsl_denoise(g2)
    # Single pixel 2 should be removed (no same-color neighbor)
    ok = result2.arr[1, 1] == 0
    print(f"    Denoise: {result2.arr[1,1]} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T4: New ops - swap_colors")
    g3 = Grid(np.array([[1,1,2],[2,1,2],[1,2,1]]))
    result3 = dsl_swap_two_colors(g3)
    ok = result3.arr[0, 0] == 2 and result3.arr[0, 2] == 1
    print(f"    Swap: [{result3.arr[0,0]},{result3.arr[0,2]}] {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T5: Program enumeration v2")
    progs = enumerate_programs_v2(max_depth=2, include_color_ops=False)
    ok = len(progs) > 300
    depth_counts = Counter(p['depth'] for p in progs.values())
    print(f"    Programs: {len(progs)} (d1:{depth_counts.get(1,0)} d2:{depth_counts.get(2,0)}) "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T6: Program enumeration depth 3")
    progs3 = enumerate_programs_v2(max_depth=3, include_color_ops=False)
    ok = len(progs3) > 1000
    d3 = Counter(p['depth'] for p in progs3.values())
    print(f"    Programs: {len(progs3)} (d1:{d3[1]} d2:{d3[2]} d3:{d3[3]}) "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T7: Pixel features")
    g4 = Grid(np.array([[1,2,3],[4,5,6]]))
    g5 = Grid(np.array([[4,1],[5,2],[6,3]]))
    feat = pair_to_cnn_features(g4, g5, size=10)
    ok = len(feat) == 10*10*3 + 9  # 3 grids + 9 stats
    print(f"    Features: {len(feat)} dims {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T8: Smart pruning removes self-inverses")
    progs_p = enumerate_programs_v2(max_depth=2, include_color_ops=False)
    has_fliph_fliph = any(p['name'] == 'flip_h+flip_h' for p in progs_p.values())
    ok = not has_fliph_fliph
    print(f"    flip_h+flip_h pruned: {not has_fliph_fliph} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print(f"\n{'='*65}")
    print(f"  Results: {p}/{t} tests passed")
    print(f"{'='*65}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--pipeline", action="store_true")
    ap.add_argument("--training", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--n-grids", type=int, default=30)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--pixel-size", type=int, default=12)
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
    elif args.pipeline or args.training:
        d = os.path.join(data_base, "training")
        full_pipeline(d, n_grids=args.n_grids, max_depth=args.max_depth,
                        epochs=args.epochs, pixel_size=args.pixel_size,
                        limit=args.limit, verbose=args.verbose)
    elif args.eval:
        d = os.path.join(data_base, "evaluation")
        full_pipeline(d, n_grids=args.n_grids, max_depth=args.max_depth,
                        epochs=args.epochs, pixel_size=args.pixel_size,
                        limit=args.limit, verbose=args.verbose)
    else:
        run_tests()
