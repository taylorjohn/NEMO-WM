"""
NeMo-WM Iterative Simulation Solver
======================================
Apply a local rule repeatedly until the grid converges.
Many ARC tasks are "run this automaton until stable."

Patterns:
  - Flood fill until no more changes
  - Gravity settling (drop objects until they land)  
  - Cellular automaton (neighbor-based rules)
  - Pattern propagation (spread color along connected regions)
  - Erosion/dilation (shrink or grow objects iteratively)

Usage:
    from arc_iterative import try_iterative_sim
    result, method = try_iterative_sim(task)
"""
import numpy as np
from collections import Counter
from arc_solver import score_task


def _bg(g):
    return int(np.argmax(np.bincount(g.flatten())))


# ═══════════════════════════════════════════════════════════
# SIMULATION RULES
# ═══════════════════════════════════════════════════════════

def sim_gravity_settle(gi, direction='down', max_steps=30):
    """Drop all non-bg pixels in a direction until they land on something."""
    bg = _bg(gi)
    h, w = gi.shape
    grid = gi.copy()
    
    for _ in range(max_steps):
        moved = False
        if direction == 'down':
            for r in range(h - 2, -1, -1):
                for c in range(w):
                    if grid[r, c] != bg and r + 1 < h and grid[r + 1, c] == bg:
                        grid[r + 1, c] = grid[r, c]
                        grid[r, c] = bg
                        moved = True
        elif direction == 'up':
            for r in range(1, h):
                for c in range(w):
                    if grid[r, c] != bg and grid[r - 1, c] == bg:
                        grid[r - 1, c] = grid[r, c]
                        grid[r, c] = bg
                        moved = True
        elif direction == 'left':
            for r in range(h):
                for c in range(1, w):
                    if grid[r, c] != bg and grid[r, c - 1] == bg:
                        grid[r, c - 1] = grid[r, c]
                        grid[r, c] = bg
                        moved = True
        elif direction == 'right':
            for r in range(h):
                for c in range(w - 2, -1, -1):
                    if grid[r, c] != bg and c + 1 < w and grid[r, c + 1] == bg:
                        grid[r, c + 1] = grid[r, c]
                        grid[r, c] = bg
                        moved = True
        if not moved:
            break
    return grid


def sim_flood_spread(gi, max_steps=30):
    """Spread each non-bg color to adjacent bg cells, iteratively."""
    bg = _bg(gi)
    h, w = gi.shape
    grid = gi.copy()
    
    for _ in range(max_steps):
        new_grid = grid.copy()
        changed = False
        for r in range(h):
            for c in range(w):
                if grid[r, c] == bg:
                    neighbors = []
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != bg:
                            neighbors.append(int(grid[nr, nc]))
                    if len(neighbors) == 1:
                        new_grid[r, c] = neighbors[0]
                        changed = True
        grid = new_grid
        if not changed:
            break
    return grid


def sim_majority_spread(gi, max_steps=20):
    """Each bg pixel becomes the majority color of its 4-neighbors (if unanimous)."""
    bg = _bg(gi)
    h, w = gi.shape
    grid = gi.copy()
    
    for _ in range(max_steps):
        new_grid = grid.copy()
        changed = False
        for r in range(h):
            for c in range(w):
                if grid[r, c] == bg:
                    neighbors = []
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != bg:
                            neighbors.append(int(grid[nr, nc]))
                    if len(neighbors) >= 2:
                        counts = Counter(neighbors)
                        top = counts.most_common(1)[0]
                        if top[1] >= 2:
                            new_grid[r, c] = top[0]
                            changed = True
        grid = new_grid
        if not changed:
            break
    return grid


def sim_erosion(gi, steps=1):
    """Remove non-bg pixels that have any bg neighbor (erode edges)."""
    bg = _bg(gi)
    h, w = gi.shape
    grid = gi.copy()
    
    for _ in range(steps):
        new_grid = grid.copy()
        for r in range(h):
            for c in range(w):
                if grid[r, c] != bg:
                    has_bg_neighbor = False
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if nr < 0 or nr >= h or nc < 0 or nc >= w or grid[nr, nc] == bg:
                            has_bg_neighbor = True
                            break
                    if has_bg_neighbor:
                        new_grid[r, c] = bg
        grid = new_grid
    return grid


def sim_dilation(gi, steps=1):
    """Grow non-bg pixels to adjacent bg cells."""
    bg = _bg(gi)
    h, w = gi.shape
    grid = gi.copy()
    
    for _ in range(steps):
        new_grid = grid.copy()
        for r in range(h):
            for c in range(w):
                if grid[r, c] == bg:
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != bg:
                            new_grid[r, c] = int(grid[nr, nc])
                            break
        grid = new_grid
    return grid


def sim_line_extend_h(gi):
    """Extend horizontal lines of non-bg pixels to fill their row."""
    bg = _bg(gi)
    h, w = gi.shape
    grid = gi.copy()
    for r in range(h):
        colors = [int(gi[r, c]) for c in range(w) if gi[r, c] != bg]
        if colors and len(set(colors)) == 1:
            for c in range(w):
                grid[r, c] = colors[0]
    return grid


def sim_line_extend_v(gi):
    """Extend vertical lines of non-bg pixels to fill their column."""
    bg = _bg(gi)
    h, w = gi.shape
    grid = gi.copy()
    for c in range(w):
        colors = [int(gi[r, c]) for r in range(h) if gi[r, c] != bg]
        if colors and len(set(colors)) == 1:
            for r in range(h):
                grid[r, c] = colors[0]
    return grid


def sim_line_extend_between(gi):
    """For each row, fill bg pixels between two same-colored pixels."""
    bg = _bg(gi)
    h, w = gi.shape
    grid = gi.copy()
    # Horizontal
    for r in range(h):
        for c1 in range(w):
            if gi[r, c1] != bg:
                color = int(gi[r, c1])
                for c2 in range(c1 + 2, w):
                    if gi[r, c2] == color:
                        for c in range(c1 + 1, c2):
                            if grid[r, c] == bg:
                                grid[r, c] = color
                        break
                    elif gi[r, c2] != bg:
                        break
    # Vertical
    for c in range(w):
        for r1 in range(h):
            if gi[r1, c] != bg:
                color = int(gi[r1, c])
                for r2 in range(r1 + 2, h):
                    if gi[r2, c] == color:
                        for r in range(r1 + 1, r2):
                            if grid[r, c] == bg:
                                grid[r, c] = color
                        break
                    elif gi[r2, c] != bg:
                        break
    return grid


# ═══════════════════════════════════════════════════════════
# SOLVER — try all simulation rules
# ═══════════════════════════════════════════════════════════

def try_iterative_sim(task):
    """Try all iterative simulation rules."""
    pairs = task['train']
    
    sims = [
        ('grav_settle_d', lambda g: sim_gravity_settle(g, 'down')),
        ('grav_settle_u', lambda g: sim_gravity_settle(g, 'up')),
        ('grav_settle_l', lambda g: sim_gravity_settle(g, 'left')),
        ('grav_settle_r', lambda g: sim_gravity_settle(g, 'right')),
        ('flood_spread', sim_flood_spread),
        ('majority_spread', sim_majority_spread),
        ('erode_1', lambda g: sim_erosion(g, 1)),
        ('erode_2', lambda g: sim_erosion(g, 2)),
        ('dilate_1', lambda g: sim_dilation(g, 1)),
        ('dilate_2', lambda g: sim_dilation(g, 2)),
        ('extend_h', sim_line_extend_h),
        ('extend_v', sim_line_extend_v),
        ('extend_between', sim_line_extend_between),
    ]
    
    for name, sim_fn in sims:
        all_match = True
        for pair in pairs:
            gi = np.array(pair['input'])
            go = np.array(pair['output'])
            try:
                pred = sim_fn(gi)
                if pred.shape != go.shape or not np.array_equal(pred, go):
                    all_match = False
                    break
            except:
                all_match = False
                break
        
        if not all_match:
            continue
        
        # Apply to test
        results = []
        for tc in task['test']:
            gi = np.array(tc['input'])
            try:
                pred = sim_fn(gi)
                results.append([pred.tolist()])
            except:
                return None, None
        
        if score_task(task, results):
            return results, f'SIM:{name}'
    
    return None, None


# ═══════════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    import json, os, sys, time
    
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  NeMo-WM Iterative Simulation Solver                        ║")
    print("║  Gravity, flood fill, automata, erosion/dilation             ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    
    data_dir = sys.argv[1] if len(sys.argv) > 1 else None
    
    if data_dir and os.path.exists(data_dir):
        files = sorted(os.listdir(data_dir))
        solved = []
        t0 = time.time()
        
        for f in files:
            if not f.endswith('.json'): continue
            task = json.load(open(os.path.join(data_dir, f)))
            result, method = try_iterative_sim(task)
            if result:
                solved.append((f, method))
                print(f"    ★ {f}: {method}")
        
        elapsed = time.time() - t0
        print(f"\n  Iterative sim: {len(solved)}/{len(files)} in {elapsed:.1f}s")
