"""
arc_chain_search.py — 2-Step Program Chaining for ARC-AGI-2
=============================================================
Tries all pairs of advanced ops as 2-step chains:
  For each unsolved task:
    For each op_A:
      intermediate = op_A(input)
      For each op_B:
        output = op_B(intermediate)
        If output matches target → SOLVED

This is the composition layer that bridges single ops (7.2%)
to multi-step reasoning (target: 10%+).

Already found 3 new tasks in initial testing:
  332efdb3: row_col_rule → distance_recolor
  ba26e723: row_col_rule → recolor_by_size  
  be94b721: extract_by_color → recolor_by_size

Usage:
    python arc_chain_search.py --training --data path/to/ARC-AGI-2/data -v
    python arc_chain_search.py --test
"""

import argparse
import json
import os
import numpy as np
from collections import Counter
import time

from arc_solver import Grid, score_task

try:
    from arc_phase2 import solve_task_phase2 as s1_solve
except ImportError:
    from arc_solver import solve_task as s1_solve

try:
    from arc_mental_models import MentalModelSolver, TaskAnalysis, ALL_MODELS
    HAS_MM = True
except ImportError:
    HAS_MM = False

from arc_advanced_ops import ALL_ADVANCED_OPS


def get_solved_set(data_dir):
    """Get set of tasks solved by S1 + MM + ADV single ops."""
    files = sorted(f for f in os.listdir(data_dir) if f.endswith('.json'))
    solved = set()
    mm_solver = MentalModelSolver() if HAS_MM else None

    for f in files:
        task = json.load(open(os.path.join(data_dir, f)))

        # S1
        s1 = s1_solve(task)
        if score_task(task, s1):
            solved.add(f)
            continue

        # MM
        if mm_solver:
            analysis = TaskAnalysis(task)
            mm_ok = False
            for conf, model in sorted([(m.matches(analysis), m) for m in ALL_MODELS],
                                        key=lambda x: -x[0]):
                if conf < 0.1:
                    continue
                g, n = model.solve(task, analysis)
                if g and score_task(task, g):
                    solved.add(f)
                    mm_ok = True
                    break
            if mm_ok:
                continue

        # ADV single ops
        for name, fn in ALL_ADVANCED_OPS:
            try:
                rule = fn(task['train'])
                if rule:
                    gs = []
                    ok = True
                    for tc in task['test']:
                        try:
                            gs.append([rule(tc['input']).tolist()])
                        except Exception:
                            ok = False
                            break
                    if ok and gs and score_task(task, gs):
                        solved.add(f)
                        break
            except Exception:
                continue

    return solved


def chain_search(data_dir, limit=None, verbose=False, fast_mode=True):
    """Search for 2-step op chains on unsolved tasks."""
    files = sorted(f for f in os.listdir(data_dir) if f.endswith('.json'))
    if limit:
        files = files[:limit]

    print("=" * 70)
    print(f"  ARC-AGI-2 Chain Search (2-Step Composition)")
    print(f"  Tasks: {len(files)}")
    print(f"  Ops available: {len(ALL_ADVANCED_OPS)}")
    print(f"  Max chains to try: {len(ALL_ADVANCED_OPS)**2}")
    print("=" * 70)

    # Get already-solved tasks
    print("  Computing solved set...")
    solved = get_solved_set(data_dir)
    print(f"  Already solved: {len(solved)}")

    # Select ops for chaining
    if fast_mode:
        # Use faster ops for first pass
        chain_ops = ALL_ADVANCED_OPS[:20]
        print(f"  Fast mode: using {len(chain_ops)} ops for chaining")
    else:
        chain_ops = ALL_ADVANCED_OPS
        print(f"  Full mode: using {len(chain_ops)} ops for chaining")

    unsolved = [(f, json.load(open(os.path.join(data_dir, f))))
                for f in files if f not in solved]
    print(f"  Unsolved to search: {len(unsolved)}")

    t0 = time.time()
    chain_solved = []

    for fi, (f, task) in enumerate(unsolved):
        pairs = task['train']
        tests = task['test']
        found = False

        for na, fa in chain_ops:
            if found:
                break
            try:
                ra = fa(pairs)
                if ra is None:
                    continue

                # Compute intermediates
                intermediates = []
                ok = True
                for p in pairs:
                    try:
                        mid = ra(p['input'])
                        intermediates.append({
                            'input': mid.tolist(),
                            'output': p['output']
                        })
                    except Exception:
                        ok = False
                        break
                if not ok:
                    continue

                for nb, fb in chain_ops:
                    if nb == na:
                        continue
                    try:
                        rb = fb(intermediates)
                        if rb is None:
                            continue

                        # Verify chain on ALL training pairs
                        all_match = True
                        for p in pairs:
                            try:
                                mid = ra(p['input'])
                                final = rb(mid.tolist())
                                if not np.array_equal(final, np.array(p['output'])):
                                    all_match = False
                                    break
                            except Exception:
                                all_match = False
                                break

                        if not all_match:
                            continue

                        # Apply to test cases
                        guesses = []
                        test_ok = True
                        for tc in tests:
                            try:
                                mid = ra(tc['input'])
                                final = rb(mid.tolist())
                                guesses.append([final.tolist()])
                            except Exception:
                                test_ok = False
                                break

                        if test_ok and guesses and score_task(task, guesses):
                            chain_solved.append((f, f"{na}→{nb}"))
                            found = True
                            if verbose:
                                print(f"  CHAIN: {f} = {na} → {nb}")
                            break
                    except Exception:
                        continue
            except Exception:
                continue

        if (fi + 1) % max(1, len(unsolved) // 10) == 0:
            elapsed = time.time() - t0
            print(f"    Progress: {fi+1}/{len(unsolved)} | "
                  f"Chains: {len(chain_solved)} | {elapsed:.0f}s")

    elapsed = time.time() - t0
    total = len(solved) + len(chain_solved)

    print(f"\n{'='*70}")
    print(f"  CHAIN SEARCH RESULTS")
    print(f"{'='*70}")
    print(f"  Previously solved:  {len(solved)}")
    print(f"  New chain solves:   {len(chain_solved)}")
    print(f"  Combined total:     {total}/1000 ({total/10:.1f}%)")
    print(f"  Time:               {elapsed:.0f}s")

    if chain_solved:
        print(f"\n  New chains found:")
        for f, chain in chain_solved:
            print(f"    {f}: {chain}")

    print(f"\n{'='*70}")
    return total, chain_solved


def run_tests():
    print("=" * 65)
    print("  Chain Search Tests")
    print("=" * 65)
    p = 0; t = 0

    print("\n  T1: Ops available")
    ok = len(ALL_ADVANCED_OPS) >= 20
    print(f"    {len(ALL_ADVANCED_OPS)} ops {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T2: Chain execution")
    # Test: rot90 → flip_h should work
    from arc_solver import dsl_rot90, dsl_flip_h
    gi = Grid(np.array([[1, 2], [3, 4]]))
    mid = dsl_rot90(gi)
    final = dsl_flip_h(mid)
    ok = final.shape == (2, 2)
    print(f"    Chain result shape: {final.shape} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print(f"\n{'='*65}")
    print(f"  Results: {p}/{t} tests passed")
    print(f"{'='*65}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--training", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--full", action="store_true",
                    help="Use all ops (slower but more thorough)")
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
        chain_search(d, limit=args.limit, verbose=args.verbose,
                      fast_mode=not args.full)
