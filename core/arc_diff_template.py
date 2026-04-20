"""
NeMo-WM Diff Template Solver
===============================
Learns pixel offset patterns from training pairs.
For each non-bg pixel, what offsets get new pixels drawn?

Example: d364b489 — each colored pixel gets 4 new pixels
at specific learned offsets.
"""
import numpy as np
from collections import Counter
from arc_solver import score_task


def bg(g):
    return int(np.argmax(np.bincount(g.flatten())))


def try_diff_template(task):
    """Learn per-pixel offset template from training, apply to test."""
    pairs = task['train']
    gi0, go0 = np.array(pairs[0]['input']), np.array(pairs[0]['output'])
    if gi0.shape != go0.shape:
        return None, None

    b = bg(gi0)
    h, w = gi0.shape

    # For each pixel that was CREATED (bg→non-bg), find nearest source pixel
    offsets = {}
    for r in range(h):
        for c in range(w):
            if gi0[r, c] == b and go0[r, c] != b:
                best_dist = 999
                best_src = None
                for sr in range(h):
                    for sc in range(w):
                        if gi0[sr, sc] != b:
                            d = abs(r - sr) + abs(c - sc)
                            if d < best_dist:
                                best_dist = d
                                best_src = (sr, sc)
                if best_src and best_dist <= 5:
                    dr, dc = r - best_src[0], c - best_src[1]
                    new_c = int(go0[r, c])
                    key = (dr, dc)
                    if key not in offsets:
                        offsets[key] = new_c
                    elif offsets[key] != new_c:
                        return None, None  # Inconsistent

    if not offsets:
        return None, None

    # Verify on all training pairs
    for pair in pairs:
        gi, go = np.array(pair['input']), np.array(pair['output'])
        if gi.shape != go.shape:
            return None, None
        h2, w2 = gi.shape
        pred = gi.copy()
        for r in range(h2):
            for c in range(w2):
                if gi[r, c] != b:
                    for (dr, dc), nc in offsets.items():
                        nr, ncol = r + dr, c + dc
                        if 0 <= nr < h2 and 0 <= ncol < w2:
                            if pred[nr, ncol] == b:
                                pred[nr, ncol] = nc
        if not np.array_equal(pred, go):
            return None, None

    # Apply to test
    results = []
    for tc in task['test']:
        gi = np.array(tc['input'])
        h2, w2 = gi.shape
        pred = gi.copy()
        for r in range(h2):
            for c in range(w2):
                if gi[r, c] != b:
                    for (dr, dc), nc in offsets.items():
                        nr, ncol = r + dr, c + dc
                        if 0 <= nr < h2 and 0 <= ncol < w2:
                            if pred[nr, ncol] == b:
                                pred[nr, ncol] = nc
        results.append([pred.tolist()])

    if score_task(task, results):
        return results, f'DIFF_TMPL:{len(offsets)}_offsets'
    return None, None


if __name__ == '__main__':
    import json, os, sys, time

    print("NeMo-WM Diff Template Solver")
    data_dir = sys.argv[1] if len(sys.argv) > 1 else None
    if data_dir and os.path.exists(data_dir):
        files = sorted(os.listdir(data_dir))
        solved = []
        t0 = time.time()
        for f in files:
            if not f.endswith('.json'): continue
            task = json.load(open(os.path.join(data_dir, f)))
            r, m = try_diff_template(task)
            if r:
                solved.append((f, m))
                print(f"  ★ {f}: {m}")
        print(f"\n  Diff template: {len(solved)}/{len(files)} in {time.time()-t0:.1f}s")
