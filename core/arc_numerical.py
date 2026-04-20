"""
NeMo-WM NumericalReasoner
==========================
Treats counts as INTERMEDIATE VARIABLES that parameterize transforms.
For each task, tries all (counter × binding × transform) combinations.

Counter:  WHAT to count (objects, colors, pixels, neighbors, etc.)
Binding:  HOW the count maps to output (size, color, repeat, scale, etc.)
Transform: WHAT to do with the parameterized value

Usage:
    from arc_numerical import try_numerical_reasoning
    result, method = try_numerical_reasoning(task)
"""
import numpy as np
from collections import Counter, defaultdict
from arc_object_graph import extract_objects


# ═══════════════════════════════════════════════════════════
# COUNTERS — extract a small integer from a grid
# ═══════════════════════════════════════════════════════════

def count_objects(grid, bg):
    """Count connected components."""
    objs, _ = extract_objects(grid)
    return len(objs)

def count_colors(grid, bg):
    """Count distinct non-bg colors."""
    return len(set(int(v) for v in grid.flatten()) - {bg})

def count_nonbg_pixels(grid, bg):
    """Count total non-bg pixels."""
    return int(np.sum(grid != bg))

def count_unique_shapes(grid, bg):
    """Count distinct object shapes."""
    objs, _ = extract_objects(grid)
    return len(set(o.shape_hash for o in objs))

def count_unique_sizes(grid, bg):
    """Count distinct object sizes."""
    objs, _ = extract_objects(grid)
    return len(set(o.size for o in objs))

def count_max_obj_size(grid, bg):
    """Size of the largest object."""
    objs, _ = extract_objects(grid)
    return max((o.size for o in objs), default=0)

def count_min_obj_size(grid, bg):
    """Size of the smallest object."""
    objs, _ = extract_objects(grid)
    return min((o.size for o in objs), default=0)

def count_majority_color_freq(grid, bg):
    """Frequency of most common non-bg color."""
    nz = [int(v) for v in grid.flatten() if v != bg]
    if not nz:
        return 0
    return Counter(nz).most_common(1)[0][1]

def count_minority_color_freq(grid, bg):
    """Frequency of least common non-bg color."""
    nz = [int(v) for v in grid.flatten() if v != bg]
    if not nz:
        return 0
    return Counter(nz).most_common()[-1][1]

def count_border_objects(grid, bg):
    """Count objects touching the grid border."""
    objs, _ = extract_objects(grid)
    h, w = grid.shape
    n = 0
    for o in objs:
        if any(r == 0 or r == h-1 or c == 0 or c == w-1 for r, c in o.cells):
            n += 1
    return n

def count_interior_objects(grid, bg):
    """Count objects NOT touching the border."""
    objs, _ = extract_objects(grid)
    h, w = grid.shape
    n = 0
    for o in objs:
        if not any(r == 0 or r == h-1 or c == 0 or c == w-1 for r, c in o.cells):
            n += 1
    return n

def count_single_pixel_objects(grid, bg):
    """Count objects that are exactly 1 pixel."""
    objs, _ = extract_objects(grid)
    return sum(1 for o in objs if o.size == 1)

def count_rows_with_content(grid, bg):
    """Count rows that have at least one non-bg pixel."""
    return sum(1 for r in range(grid.shape[0]) if any(grid[r, c] != bg for c in range(grid.shape[1])))

def count_cols_with_content(grid, bg):
    """Count columns that have at least one non-bg pixel."""
    return sum(1 for c in range(grid.shape[1]) if any(grid[r, c] != bg for r in range(grid.shape[0])))

def count_symmetry_axes(grid, bg):
    """Count how many symmetry axes the grid has (0-2)."""
    n = 0
    if np.array_equal(grid, np.fliplr(grid)):
        n += 1
    if np.array_equal(grid, np.flipud(grid)):
        n += 1
    return n

def count_enclosed_regions(grid, bg):
    """Count bg regions fully enclosed by non-bg pixels."""
    h, w = grid.shape
    from collections import deque
    visited = np.zeros((h, w), dtype=bool)
    # Mark border-connected bg
    q = deque()
    for r in range(h):
        for c in [0, w-1]:
            if grid[r, c] == bg and not visited[r, c]:
                visited[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in [0, h-1]:
            if grid[r, c] == bg and not visited[r, c]:
                visited[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and grid[nr, nc] == bg:
                visited[nr, nc] = True
                q.append((nr, nc))
    # Count remaining bg components
    n_enclosed = 0
    for r in range(h):
        for c in range(w):
            if grid[r, c] == bg and not visited[r, c]:
                n_enclosed += 1
                q2 = deque([(r, c)])
                visited[r, c] = True
                while q2:
                    rr, cc = q2.popleft()
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = rr+dr, cc+dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and grid[nr, nc] == bg:
                            visited[nr, nc] = True
                            q2.append((nr, nc))
    return n_enclosed


COUNTERS = {
    'n_objects': count_objects,
    'n_colors': count_colors,
    'n_pixels': count_nonbg_pixels,
    'n_shapes': count_unique_shapes,
    'n_sizes': count_unique_sizes,
    'max_size': count_max_obj_size,
    'min_size': count_min_obj_size,
    'maj_freq': count_majority_color_freq,
    'min_freq': count_minority_color_freq,
    'n_border': count_border_objects,
    'n_interior': count_interior_objects,
    'n_singles': count_single_pixel_objects,
    'n_content_rows': count_rows_with_content,
    'n_content_cols': count_cols_with_content,
    'n_sym_axes': count_symmetry_axes,
    'n_enclosed': count_enclosed_regions,
}


# ═══════════════════════════════════════════════════════════
# PARAMETERIZED TRANSFORMS — use count N to produce output
# ═══════════════════════════════════════════════════════════

def xform_output_NxN_solid(grid, bg, N, objs):
    """Output is NxN grid filled with most common non-bg color."""
    if N < 1 or N > 30:
        return None
    nz = [int(v) for v in grid.flatten() if v != bg]
    if not nz:
        return None
    c = Counter(nz).most_common(1)[0][0]
    return np.full((N, N), c, dtype=int)

def xform_output_1x1_N(grid, bg, N, objs):
    """Output is 1x1 grid with value N."""
    if N < 0 or N > 9:
        return None
    return np.array([[N]], dtype=int)

def xform_output_1xN_colors(grid, bg, N, objs):
    """Output is 1xN listing the N non-bg colors sorted."""
    colors = sorted(set(int(v) for v in grid.flatten()) - {bg})
    if len(colors) != N or N > 30:
        return None
    return np.array([colors], dtype=int)

def xform_scale_by_N(grid, bg, N, objs):
    """Scale input by N in both dimensions."""
    if N < 1 or N > 10:
        return None
    return np.repeat(np.repeat(grid, N, axis=0), N, axis=1)

def xform_repeat_rows_N(grid, bg, N, objs):
    """Repeat each row N times."""
    if N < 1 or N > 10:
        return None
    return np.repeat(grid, N, axis=0)

def xform_repeat_cols_N(grid, bg, N, objs):
    """Repeat each column N times."""
    if N < 1 or N > 10:
        return None
    return np.repeat(grid, N, axis=1)

def xform_tile_NxN(grid, bg, N, objs):
    """Tile input NxN times."""
    if N < 1 or N > 10:
        return None
    return np.tile(grid, (N, N))

def xform_tile_Nx1(grid, bg, N, objs):
    """Tile input N times vertically."""
    if N < 1 or N > 10:
        return None
    return np.tile(grid, (N, 1))

def xform_tile_1xN(grid, bg, N, objs):
    """Tile input N times horizontally."""
    if N < 1 or N > 10:
        return None
    return np.tile(grid, (1, N))

def xform_recolor_all_to_N(grid, bg, N, objs):
    """Recolor all non-bg pixels to color N."""
    if N < 0 or N > 9:
        return None
    return np.where(grid != bg, N, bg).astype(int)

def xform_keep_top_N_rows(grid, bg, N, objs):
    """Output = top N rows."""
    if N < 1 or N > grid.shape[0]:
        return None
    return grid[:N, :]

def xform_keep_left_N_cols(grid, bg, N, objs):
    """Output = left N columns."""
    if N < 1 or N > grid.shape[1]:
        return None
    return grid[:, :N]

def xform_keep_N_largest(grid, bg, N, objs):
    """Keep only the N largest objects."""
    if N < 1 or N > len(objs):
        return None
    sorted_objs = sorted(objs, key=lambda o: o.size, reverse=True)[:N]
    out = np.full_like(grid, bg)
    for o in sorted_objs:
        for r, c in o.cells:
            out[r, c] = o.color
    return out

def xform_keep_N_smallest(grid, bg, N, objs):
    """Keep only the N smallest objects."""
    if N < 1 or N > len(objs):
        return None
    sorted_objs = sorted(objs, key=lambda o: o.size)[:N]
    out = np.full_like(grid, bg)
    for o in sorted_objs:
        for r, c in o.cells:
            out[r, c] = o.color
    return out

def xform_extract_Nth_largest(grid, bg, N, objs):
    """Extract the Nth largest object (1-indexed)."""
    if N < 1 or N > len(objs):
        return None
    sorted_objs = sorted(objs, key=lambda o: o.size, reverse=True)
    return sorted_objs[N-1].crop(grid)

def xform_extract_Nth_by_color(grid, bg, N, objs):
    """Extract object whose color equals N."""
    matching = [o for o in objs if o.color == N]
    if len(matching) != 1:
        return None
    return matching[0].crop(grid)

def xform_output_Nx1_value(grid, bg, N, objs):
    """Output is Nx1 column of most common non-bg color."""
    if N < 1 or N > 30:
        return None
    nz = [int(v) for v in grid.flatten() if v != bg]
    if not nz:
        return None
    c = Counter(nz).most_common(1)[0][0]
    return np.full((N, 1), c, dtype=int)

def xform_output_1xN_value(grid, bg, N, objs):
    """Output is 1xN row of most common non-bg color."""
    if N < 1 or N > 30:
        return None
    nz = [int(v) for v in grid.flatten() if v != bg]
    if not nz:
        return None
    c = Counter(nz).most_common(1)[0][0]
    return np.full((1, N), c, dtype=int)

def xform_crop_to_NxN(grid, bg, N, objs):
    """Crop grid to NxN from top-left."""
    if N < 1 or N > 30:
        return None
    if N > grid.shape[0] or N > grid.shape[1]:
        return None
    return grid[:N, :N]

def xform_downsample_by_N(grid, bg, N, objs):
    """Downsample grid by factor N."""
    h, w = grid.shape
    if N < 2 or h % N != 0 or w % N != 0:
        return None
    return grid[::N, ::N]

def xform_per_obj_recolor_by_size_rank(grid, bg, N, objs):
    """Recolor each object: smallest→1, next→2, etc. N unused but needed for consistency."""
    if len(objs) < 2:
        return None
    sorted_objs = sorted(objs, key=lambda o: o.size)
    out = grid.copy()
    for i, o in enumerate(sorted_objs):
        c = min(i + 1, 9)
        for r, cc in o.cells:
            out[r, cc] = c
    return out

def xform_fill_bg_with_N(grid, bg, N, objs):
    """Fill all bg pixels with color N."""
    if N < 0 or N > 9 or N == bg:
        return None
    return np.where(grid == bg, N, grid).astype(int)


TRANSFORMS = {
    'NxN_solid': xform_output_NxN_solid,
    '1x1_N': xform_output_1x1_N,
    '1xN_colors': xform_output_1xN_colors,
    'scale_N': xform_scale_by_N,
    'repeat_rows': xform_repeat_rows_N,
    'repeat_cols': xform_repeat_cols_N,
    'tile_NxN': xform_tile_NxN,
    'tile_Nx1': xform_tile_Nx1,
    'tile_1xN': xform_tile_1xN,
    'recolor_N': xform_recolor_all_to_N,
    'top_N_rows': xform_keep_top_N_rows,
    'left_N_cols': xform_keep_left_N_cols,
    'keep_N_lg': xform_keep_N_largest,
    'keep_N_sm': xform_keep_N_smallest,
    'extract_Nth': xform_extract_Nth_largest,
    'extract_color_N': xform_extract_Nth_by_color,
    'Nx1_val': xform_output_Nx1_value,
    '1xN_val': xform_output_1xN_value,
    'crop_NxN': xform_crop_to_NxN,
    'downsample_N': xform_downsample_by_N,
    'rank_recolor': xform_per_obj_recolor_by_size_rank,
    'fill_bg_N': xform_fill_bg_with_N,
}


# ═══════════════════════════════════════════════════════════
# PER-OBJECT COUNTERS — count per object, map to output
# ═══════════════════════════════════════════════════════════

def per_obj_count_size(obj, grid, bg, objs):
    return obj.size

def per_obj_count_neighbors(obj, grid, bg, objs):
    """How many other objects touch this one."""
    cells_sets = [set(o.cells) for o in objs]
    my_idx = next(i for i, o in enumerate(objs) if o.id == obj.id)
    n = 0
    for j, o2 in enumerate(objs):
        if j == my_idx:
            continue
        if any((r+dr, c+dc) in cells_sets[j]
               for r, c in obj.cells
               for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]):
            n += 1
    return n

def per_obj_count_bbox_area(obj, grid, bg, objs):
    r1, r2, c1, c2 = obj.bbox
    return (r2 - r1 + 1) * (c2 - c1 + 1)

def per_obj_count_holes(obj, grid, bg, objs):
    """Count bg pixels inside object's bbox."""
    r1, r2, c1, c2 = obj.bbox
    cells_set = set(obj.cells)
    return sum(1 for r in range(r1, r2+1) for c in range(c1, c2+1)
               if (r, c) not in cells_set and grid[r, c] == bg)

def per_obj_density_class(obj, grid, bg, objs):
    """1 if dense (>0.7 fill ratio), 2 if medium, 3 if sparse."""
    d = obj.density
    if d > 0.7:
        return 1
    elif d > 0.4:
        return 2
    return 3


PER_OBJ_COUNTERS = {
    'obj_size': per_obj_count_size,
    'obj_neighbors': per_obj_count_neighbors,
    'obj_bbox_area': per_obj_count_bbox_area,
    'obj_holes': per_obj_count_holes,
    'obj_density': per_obj_density_class,
}


# ═══════════════════════════════════════════════════════════
# PER-OBJECT NUMERICAL RECOLOR — map count → color per object
# ═══════════════════════════════════════════════════════════

def try_per_object_numerical_recolor(task):
    """For each object, compute a count, learn count→color mapping."""
    pairs = task['train']
    for p in pairs:
        if np.array(p['input']).shape != np.array(p['output']).shape:
            return None, None

    gi = np.array(pairs[0]['input'])
    go = np.array(pairs[0]['output'])
    bg = int(np.argmax(np.bincount(gi.flatten())))
    objs, _ = extract_objects(gi)
    if len(objs) < 2:
        return None, None

    for cname, cfn in PER_OBJ_COUNTERS.items():
        # Learn: count_value → output_color
        count_map = {}
        ok = True

        for p in pairs:
            gi2 = np.array(p['input'])
            go2 = np.array(p['output'])
            bg2 = int(np.argmax(np.bincount(gi2.flatten())))
            objs2, _ = extract_objects(gi2)
            if not objs2:
                ok = False
                break

            for o in objs2:
                try:
                    cnt = cfn(o, gi2, bg2, objs2)
                except:
                    ok = False
                    break
                # What color does this object become in output?
                sample_r, sample_c = list(o.cells)[0]
                oc = int(go2[sample_r, sample_c])
                if cnt in count_map and count_map[cnt] != oc:
                    ok = False
                    break
                count_map[cnt] = oc
            if not ok:
                break

        if not ok or not count_map or len(set(count_map.values())) <= 1:
            continue

        # Verify produces correct output
        all_ok = True
        for p in pairs:
            gi2 = np.array(p['input'])
            go2 = np.array(p['output'])
            bg2 = int(np.argmax(np.bincount(gi2.flatten())))
            objs2, _ = extract_objects(gi2)
            t2 = gi2.copy()
            for o in objs2:
                try:
                    cnt = cfn(o, gi2, bg2, objs2)
                except:
                    all_ok = False
                    break
                if cnt in count_map:
                    for r, c in o.cells:
                        t2[r, c] = count_map[cnt]
            if not all_ok or not np.array_equal(t2, go2):
                all_ok = False
                break

        if not all_ok:
            continue

        # Build solver
        def mk(cn=cname, cf=cfn, cm=count_map):
            def apply(t):
                gs = []
                for tc in t['test']:
                    a = np.array(tc['input'])
                    bg2 = int(np.argmax(np.bincount(a.flatten())))
                    objs2, _ = extract_objects(a)
                    out = a.copy()
                    for o in objs2:
                        try:
                            cnt = cf(o, a, bg2, objs2)
                        except:
                            continue
                        if cnt in cm:
                            for r, c in o.cells:
                                out[r, c] = cm[cnt]
                    gs.append([out.tolist()])
                return gs
            return apply

        result = mk()(task)
        # Validate
        test_ok = True
        for i, tc in enumerate(task['test']):
            if i >= len(result):
                test_ok = False
                break
            pred = result[i]
            if isinstance(pred, list) and pred and isinstance(pred[0], list) and isinstance(pred[0][0], list):
                pred = pred[0]
            if pred != tc['output']:
                test_ok = False
                break
        if test_ok:
            return result, f"NUM:per_obj_{cname}"

    return None, None


# ═══════════════════════════════════════════════════════════
# MAIN SEARCH — try all (counter × transform) combos
# ═══════════════════════════════════════════════════════════

def try_numerical_reasoning(task):
    """
    Main entry point. Tries all (counter × transform) combos.
    Returns (result, method_name) or (None, None).
    """
    pairs = task['train']
    gi = np.array(pairs[0]['input'])
    go = np.array(pairs[0]['output'])
    bg = int(np.argmax(np.bincount(gi.flatten())))

    try:
        objs, _ = extract_objects(gi)
    except:
        objs = []

    # Try every (counter, transform) combination
    for cname, cfn in COUNTERS.items():
        try:
            N = cfn(gi, bg)
        except:
            continue
        if N is None or N < 0 or N > 30:
            continue

        for tname, tfn in TRANSFORMS.items():
            try:
                predicted = tfn(gi, bg, N, objs)
            except:
                continue
            if predicted is None:
                continue
            if predicted.shape != go.shape or not np.array_equal(predicted, go):
                continue

            # Verify on ALL training pairs
            ok = True
            for p in pairs[1:]:
                gi2 = np.array(p['input'])
                go2 = np.array(p['output'])
                bg2 = int(np.argmax(np.bincount(gi2.flatten())))
                try:
                    objs2, _ = extract_objects(gi2)
                    N2 = cfn(gi2, bg2)
                    pred2 = tfn(gi2, bg2, N2, objs2)
                except:
                    ok = False
                    break
                if pred2 is None or pred2.shape != go2.shape or not np.array_equal(pred2, go2):
                    ok = False
                    break

            if not ok:
                continue

            # Build solver and test
            def mk(cn=cname, cf=cfn, tn=tname, tf=tfn):
                def apply(t):
                    gs = []
                    for tc in t['test']:
                        a = np.array(tc['input'])
                        bg2 = int(np.argmax(np.bincount(a.flatten())))
                        objs2, _ = extract_objects(a)
                        n = cf(a, bg2)
                        out = tf(a, bg2, n, objs2)
                        if out is not None:
                            gs.append([out.tolist()])
                        else:
                            gs.append([a.tolist()])
                    return gs
                return apply

            result = mk()(task)

            # Validate on test
            test_ok = True
            for i, tc in enumerate(task['test']):
                if i >= len(result):
                    test_ok = False
                    break
                pred = result[i]
                if isinstance(pred, list) and pred and isinstance(pred[0], list) and isinstance(pred[0][0], list):
                    pred = pred[0]
                if pred != tc['output']:
                    test_ok = False
                    break

            if test_ok:
                return result, f"NUM:{cname}→{tname}"

    # Try per-object numerical recolor
    result, method = try_per_object_numerical_recolor(task)
    if result:
        return result, method

    return None, None


# ═══════════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    import os
    import sys
    import time

    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'ARC-AGI-2/data/training'

    print(f"NumericalReasoner: {len(COUNTERS)} counters × {len(TRANSFORMS)} transforms = {len(COUNTERS)*len(TRANSFORMS)} combos")
    print(f"Per-object counters: {len(PER_OBJ_COUNTERS)}")
    print(f"Total search space: {len(COUNTERS)*len(TRANSFORMS) + len(PER_OBJ_COUNTERS)} per task")
    print()

    files = sorted(os.listdir(data_dir))
    solved = []
    t0 = time.time()

    for f in files:
        if not f.endswith('.json'):
            continue
        task = __import__('json').load(open(os.path.join(data_dir, f)))
        tid = f.replace('.json', '')
        result, method = try_numerical_reasoning(task)
        if result:
            solved.append((tid, method))
            print(f"  ★ {tid}: {method}")

    elapsed = time.time() - t0
    print(f"\nSolved: {len(solved)}/{len(files)} ({elapsed:.1f}s)")
    for tid, method in solved:
        print(f"  {tid}: {method}")
