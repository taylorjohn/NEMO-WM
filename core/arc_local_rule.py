"""
NeMo-WM Local Neighborhood Rule Solver
=========================================
Completely different paradigm from object-level reasoning.
Treats each pixel independently — its output color is a function
of its 3x3 local neighborhood in the input.

This is essentially learning a cellular automaton rule from examples.

Found: a699fb00 (54 patterns), dc1df850 (63 patterns)
"""
import numpy as np
from arc_solver import score_task


def neighborhood_key(gi, r, c, radius=1):
    """Get the local neighborhood pattern as a hashable key."""
    h, w = gi.shape
    key = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                key.append(int(gi[nr, nc]))
            else:
                key.append(-1)  # border sentinel
    return tuple(key)


def try_local_rule(task, radius=1):
    """Learn: output[r,c] = f(neighborhood(input, r, c))."""
    pairs = task['train']
    gi0, go0 = np.array(pairs[0]['input']), np.array(pairs[0]['output'])
    if gi0.shape != go0.shape:
        return None, None

    # Build lookup table from ALL training pairs
    lookup = {}
    for pair in pairs:
        gi, go = np.array(pair['input']), np.array(pair['output'])
        if gi.shape != go.shape:
            return None, None
        for r in range(gi.shape[0]):
            for c in range(gi.shape[1]):
                key = neighborhood_key(gi, r, c, radius)
                val = int(go[r, c])
                if key in lookup and lookup[key] != val:
                    return None, None  # Inconsistent — not a local rule
                lookup[key] = val

    if not lookup:
        return None, None

    # Verify: all training neighborhoods must be in lookup
    for pair in pairs:
        gi, go = np.array(pair['input']), np.array(pair['output'])
        for r in range(gi.shape[0]):
            for c in range(gi.shape[1]):
                key = neighborhood_key(gi, r, c, radius)
                if key not in lookup:
                    return None, None

    # Apply to test — only if ALL test neighborhoods are known
    results = []
    for tc in task['test']:
        gi = np.array(tc['input'])
        pred = gi.copy()
        for r in range(gi.shape[0]):
            for c in range(gi.shape[1]):
                key = neighborhood_key(gi, r, c, radius)
                if key in lookup:
                    pred[r, c] = lookup[key]
                else:
                    return None, None  # Unseen neighborhood
        results.append([pred.tolist()])

    if score_task(task, results):
        return results, f'LOCAL:r{radius}_{len(lookup)}pat'
    return None, None


if __name__ == '__main__':
    import json, os, sys, time

    print("NeMo-WM Local Neighborhood Rule Solver")
    data_dir = sys.argv[1] if len(sys.argv) > 1 else None
    if data_dir and os.path.exists(data_dir):
        files = sorted(os.listdir(data_dir))
        solved = []
        t0 = time.time()
        for f in files:
            if not f.endswith('.json'): continue
            task = json.load(open(os.path.join(data_dir, f)))
            r, m = try_local_rule(task)
            if r:
                solved.append((f, m))
                print(f"  ★ {f}: {m}")
        print(f"\n  Local rule: {len(solved)}/{len(files)} in {time.time()-t0:.1f}s")
