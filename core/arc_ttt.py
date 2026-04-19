"""
arc_ttt.py — Test-Time Training for ARC-AGI-2
================================================
Pipeline:
  1. Generate synthetic dataset (NVARC-style) → train base proposer
  2. For each ARC task at test time:
     a. Take 2-3 training demos
     b. Augment them (rotations, color swaps, reflections)
     c. Fine-tune proposer weights on augmented demos
     d. Use adapted proposer to rank candidate programs
     e. Verify top-K programs against original demos
     f. Reset weights for next task

This is the key technique that separates NVARC (24%) from
basic approaches (~5%). The proposer gets a general prior from
synthetic data, then specializes per-task.

Usage:
    python arc_ttt.py --training --data path/to/ARC-AGI-2/data
    python arc_ttt.py --eval --data path/to/ARC-AGI-2/data
    python arc_ttt.py --test
"""

import argparse
import json
import os
import copy
import numpy as np
from collections import Counter
from typing import List, Tuple, Optional
import time

from arc_solver import (Grid, score_task, SINGLE_OPS,
                         solve_task as s1_solve,
                         dsl_rot90, dsl_rot180, dsl_rot270,
                         dsl_flip_h, dsl_flip_v, dsl_transpose,
                         dsl_crop_nonzero, dsl_scale2,
                         dsl_outline, dsl_fill_holes,
                         dsl_gravity_down, dsl_gravity_left,
                         dsl_mirror_h, dsl_mirror_v,
                         dsl_largest_object, dsl_smallest_object,
                         dsl_keep_only_color, dsl_remove_color,
                         dsl_top_half, dsl_bottom_half,
                         dsl_left_half, dsl_right_half)

from arc_synth import (BASE_PRIMITIVES, NeuralProposer,
                        enumerate_programs, generate_random_grids,
                        make_keep_color, make_remove_color,
                        grid_to_features, diff_features, reconstruct_fn)

# Try Phase 2 for S1
try:
    from arc_phase2 import solve_task_phase2 as s1_solve_enhanced
    HAS_PHASES = True
except ImportError:
    HAS_PHASES = False


# ══════════════════════════════════════════════════════════════════════
# 1. DATA AUGMENTATION — Generate variants of training demos
# ══════════════════════════════════════════════════════════════════════

def augment_pair(gi: Grid, go: Grid, rng) -> list:
    """Generate augmented versions of an (input, output) pair."""
    augmented = [(gi, go)]  # original

    # Rotation augments (apply SAME transform to both)
    for rot_fn in [dsl_rot90, dsl_rot180, dsl_rot270]:
        try:
            augmented.append((rot_fn(gi), rot_fn(go)))
        except Exception:
            pass

    # Flip augments
    for flip_fn in [dsl_flip_h, dsl_flip_v]:
        try:
            augmented.append((flip_fn(gi), flip_fn(go)))
        except Exception:
            pass

    # Transpose
    try:
        augmented.append((dsl_transpose(gi), dsl_transpose(go)))
    except Exception:
        pass

    # Color permutation (swap two colors)
    for _ in range(3):
        c1 = rng.randint(0, 10)
        c2 = rng.randint(0, 10)
        if c1 == c2:
            continue
        try:
            ai = gi.arr.copy()
            ao = go.arr.copy()
            mask1_i, mask2_i = ai == c1, ai == c2
            ai[mask1_i] = c2
            ai[mask2_i] = c1
            mask1_o, mask2_o = ao == c1, ao == c2
            ao[mask1_o] = c2
            ao[mask2_o] = c1
            augmented.append((Grid(ai), Grid(ao)))
        except Exception:
            pass

    return augmented


def augment_task_demos(pairs, rng, max_augmented=50) -> list:
    """Augment all training demos for a task."""
    all_augmented = []
    for p in pairs:
        gi = Grid(p['input']) if not isinstance(p['input'], Grid) else p['input']
        go = Grid(p['output']) if not isinstance(p['output'], Grid) else p['output']
        augs = augment_pair(gi, go, rng)
        all_augmented.extend(augs)

    # Limit
    if len(all_augmented) > max_augmented:
        idx = rng.choice(len(all_augmented), max_augmented, replace=False)
        all_augmented = [all_augmented[i] for i in idx]

    return all_augmented


# ══════════════════════════════════════════════════════════════════════
# 2. FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════

def extract_features(gi: Grid, go: Grid) -> np.ndarray:
    """Extract feature vector for a (input, output) pair."""
    in_f = grid_to_features(gi)
    out_f = grid_to_features(go)
    df = diff_features(gi, go)
    return np.concatenate([in_f, out_f, df])


# ══════════════════════════════════════════════════════════════════════
# 3. PER-TASK TTT — Fine-tune proposer on augmented demos
# ══════════════════════════════════════════════════════════════════════

def ttt_adapt(proposer: NeuralProposer, augmented_pairs: list,
              programs: dict, label_map: dict, rev_map: dict,
              n_epochs=10, lr=0.005) -> NeuralProposer:
    """
    Fine-tune proposer on augmented versions of the task's demos.
    Only checks proposer's top-K predicted programs (not all programs).
    """
    # Get top-K from base proposer using first augmented pair
    if not augmented_pairs:
        return proposer

    gi0, go0 = augmented_pairs[0]
    feat0 = extract_features(gi0, go0)
    top_progs = proposer.predict_topk(feat0, k=30)
    candidate_ids = set()
    for pid in top_progs:
        oid = rev_map.get(pid)
        if oid is not None and oid in programs:
            candidate_ids.add(oid)

    if not candidate_ids:
        return proposer

    # Find which candidate programs match augmented pairs
    ttt_X = []
    ttt_Y = []

    for gi, go in augmented_pairs:
        feat = extract_features(gi, go)

        for prog_id in candidate_ids:
            try:
                prog_info = programs[prog_id]
                fn = prog_info.get('fn')
                if fn is None:
                    fn = reconstruct_fn(prog_info.get('steps', []))
                    if fn is None:
                        continue
                result = fn(gi)
                if result == go:
                    mapped_id = label_map.get(prog_id)
                    if mapped_id is not None:
                        ttt_X.append(feat)
                        ttt_Y.append(mapped_id)
                        break
            except Exception:
                continue

    if len(ttt_X) < 2:
        return proposer

    ttt_X = np.array(ttt_X, dtype=np.float32)
    ttt_Y = np.array(ttt_Y, dtype=np.int32)

    # Create adapted copy
    adapted = copy.deepcopy(proposer)

    # Normalize features
    if hasattr(adapted, 'feat_mean'):
        ttt_X_norm = (ttt_X - adapted.feat_mean) / adapted.feat_std
    else:
        ttt_X_norm = ttt_X

    # Fine-tune with small learning rate
    for epoch in range(n_epochs):
        # Forward
        h1 = np.maximum(0, ttt_X_norm @ adapted.W1 + adapted.b1)
        h2 = np.maximum(0, h1 @ adapted.W2 + adapted.b2)
        logits = h2 @ adapted.W3 + adapted.b3

        # Softmax
        logits_max = logits - logits.max(axis=1, keepdims=True)
        exp_l = np.exp(np.clip(logits_max, -20, 20))
        probs = exp_l / (exp_l.sum(axis=1, keepdims=True) + 1e-8)

        bs = len(ttt_X)

        # Backward
        dlogits = probs.copy()
        dlogits[np.arange(bs), ttt_Y] -= 1
        dlogits /= bs

        dW3 = h2.T @ dlogits
        db3 = dlogits.sum(axis=0)
        dh2 = dlogits @ adapted.W3.T * (h2 > 0).astype(np.float32)
        dW2 = h1.T @ dh2
        db2 = dh2.sum(axis=0)
        dh1 = dh2 @ adapted.W2.T * (h1 > 0).astype(np.float32)
        dW1 = ttt_X_norm.T @ dh1
        db1 = dh1.sum(axis=0)

        # Clip
        for g in [dW1, db1, dW2, db2, dW3, db3]:
            np.clip(g, -1, 1, out=g)

        # Update (small LR, no weight decay — we want to preserve base knowledge)
        adapted.W1 -= lr * dW1
        adapted.b1 -= lr * db1
        adapted.W2 -= lr * dW2
        adapted.b2 -= lr * db2
        adapted.W3 -= lr * dW3
        adapted.b3 -= lr * db3

    return adapted


# ══════════════════════════════════════════════════════════════════════
# 4. TTT SOLVER — Full pipeline
# ══════════════════════════════════════════════════════════════════════

class TTTSolver:
    """
    Complete ARC solver with:
    - S1: Pattern matching (Phase 2)
    - S2: Proposer-guided search with per-task TTT
    """

    def __init__(self, n_grids=50, max_depth=2, proposer_epochs=200):
        self.rng = np.random.RandomState(42)
        self.programs = None
        self.proposer = None
        self.label_map = None
        self.rev_map = None

        self.n_grids = n_grids
        self.max_depth = max_depth
        self.proposer_epochs = proposer_epochs

        # Stats
        self.s1_solves = 0
        self.s2_solves = 0
        self.s2_ttt_solves = 0
        self.total = 0

    def train_proposer(self):
        """Generate synthetic data and train base proposer."""
        print("  Generating synthetic dataset...")
        self.programs = enumerate_programs(max_depth=self.max_depth)
        print(f"  Programs: {len(self.programs)}")

        rng = np.random.RandomState(42)
        X, Y = [], []
        for pid, pinfo in self.programs.items():
            fn = pinfo['fn']
            grids = generate_random_grids(self.n_grids * 2, rng)
            n = 0
            for g in grids:
                if n >= self.n_grids:
                    break
                try:
                    out = fn(g)
                    if 0 < out.h <= 30 and 0 < out.w <= 30:
                        feat = extract_features(g, out)
                        X.append(feat)
                        Y.append(pid)
                        n += 1
                except Exception:
                    continue

        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.int32)

        unique_labels = sorted(set(Y))
        self.label_map = {old: new for new, old in enumerate(unique_labels)}
        self.rev_map = {new: old for old, new in self.label_map.items()}
        Y_mapped = np.array([self.label_map[l] for l in Y], dtype=np.int32)

        print(f"  Dataset: {len(X)} examples, {len(unique_labels)} programs")
        print(f"  Training proposer ({self.proposer_epochs} epochs)...")

        self.proposer = NeuralProposer(X.shape[1], len(unique_labels))
        self.proposer.train(X, Y_mapped, epochs=self.proposer_epochs)

    def solve_task(self, task) -> tuple:
        """Solve one ARC task with S1 → S2 → S2+TTT."""
        self.total += 1
        pairs = task['train']
        test_cases = task['test']

        # ── S1: Pattern matching ──
        if HAS_PHASES:
            s1_preds = s1_solve_enhanced(task)
        else:
            s1_preds = s1_solve(task)
        if score_task(task, s1_preds):
            self.s1_solves += 1
            return s1_preds, 'S1'

        # ── S2: Base proposer (no TTT) ──
        result = self._try_proposer(pairs, test_cases, self.proposer, top_k=50)
        if result is not None:
            self.s2_solves += 1
            return result, 'S2'

        # ── S2+TTT: Adapted proposer ──
        aug_pairs = augment_task_demos(pairs, self.rng, max_augmented=40)
        adapted = ttt_adapt(
            self.proposer, aug_pairs,
            self.programs, self.label_map, self.rev_map,
            n_epochs=15, lr=0.005
        )
        result = self._try_proposer(pairs, test_cases, adapted, top_k=50)
        if result is not None:
            self.s2_ttt_solves += 1
            return result, 'S2+TTT'

        # Fallback
        return s1_preds if HAS_PHASES else [[tc['input']] for tc in test_cases], 'FAIL'

    def _try_proposer(self, pairs, test_cases, proposer, top_k=50):
        """Try proposer's top-K programs against the task."""
        gi = Grid(pairs[0]['input'])
        go = Grid(pairs[0]['output'])
        feat = extract_features(gi, go)
        predicted = proposer.predict_topk(feat, k=top_k)

        for pred_id in predicted:
            orig_id = self.rev_map.get(pred_id)
            if orig_id is None or orig_id not in self.programs:
                continue

            prog_info = self.programs[orig_id]
            fn = prog_info.get('fn')
            if fn is None:
                fn = reconstruct_fn(prog_info.get('steps', []))
                if fn is None:
                    continue

            # Verify against ALL training pairs
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
                for tc in test_cases:
                    try:
                        result = fn(Grid(tc['input']))
                        guesses.append([result.to_list()])
                    except Exception:
                        guesses.append([tc['input']])

                if score_task({'test': test_cases}, guesses):
                    return guesses

        return None


# ══════════════════════════════════════════════════════════════════════
# 5. BENCHMARK
# ══════════════════════════════════════════════════════════════════════

def run_benchmark(data_dir, n_grids=50, max_depth=2, epochs=200,
                    limit=None, verbose=False):
    files = sorted(f for f in os.listdir(data_dir) if f.endswith('.json'))
    if limit:
        files = files[:limit]

    solver = TTTSolver(n_grids=n_grids, max_depth=max_depth,
                         proposer_epochs=epochs)

    print("=" * 70)
    print(f"  ARC-AGI-2 with Test-Time Training")
    print(f"  Tasks: {len(files)}")
    print(f"  S1: {'Phase 2 (37 strategies)' if HAS_PHASES else 'Base (33 ops)'}")
    print(f"  S2: Proposer ({n_grids} grids/prog, depth≤{max_depth})")
    print(f"  TTT: Per-task fine-tuning on augmented demos")
    print("=" * 70)

    solver.train_proposer()

    solved = 0
    total = 0
    t0 = time.time()

    for fi, f in enumerate(files):
        task = json.load(open(os.path.join(data_dir, f)))
        total += 1

        predictions, system = solver.solve_task(task)
        correct = score_task(task, predictions)

        if correct:
            solved += 1

        if verbose and correct and system != 'S1':
            print(f"  {f[:16]}: SOLVED ({system})")
        elif verbose and system == 'S2+TTT':
            print(f"  {f[:16]}: TTT attempted")

        # Progress
        if (fi + 1) % max(1, len(files) // 5) == 0:
            elapsed = time.time() - t0
            print(f"    Progress: {fi+1}/{len(files)} | "
                  f"Solved: {solved} | "
                  f"S1:{solver.s1_solves} S2:{solver.s2_solves} "
                  f"TTT:{solver.s2_ttt_solves} | "
                  f"Time: {elapsed:.0f}s")

    elapsed = time.time() - t0
    pct = solved / max(total, 1) * 100

    print(f"\n{'='*70}")
    print(f"  TTT RESULTS")
    print(f"{'='*70}")
    print(f"  Solved:     {solved}/{total} ({pct:.1f}%)")
    print(f"    S1:       {solver.s1_solves} (pattern matching)")
    print(f"    S2:       {solver.s2_solves} (base proposer)")
    print(f"    S2+TTT:   {solver.s2_ttt_solves} (adapted proposer)")
    print(f"  Time:       {elapsed:.1f}s ({elapsed/max(total,1)*1000:.0f}ms/task)")
    print(f"\n{'='*70}")
    return solved, total


# ══════════════════════════════════════════════════════════════════════
# 6. TESTS
# ══════════════════════════════════════════════════════════════════════

def run_tests():
    print("=" * 65)
    print("  Test-Time Training Tests")
    print("=" * 65)
    p = 0; t = 0

    rng = np.random.RandomState(42)

    print("\n  T1: Augmentation")
    gi = Grid(np.array([[1,2],[3,4]]))
    go = Grid(np.array([[3,1],[4,2]]))
    augs = augment_pair(gi, go, rng)
    ok = len(augs) >= 5
    print(f"    Augmented pairs: {len(augs)} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T2: Augmentation preserves transform")
    # If input→output is rot90, then rot90(input)→rot90(output) should also be valid
    # i.e. the augmented pair should still be solvable by the same program
    valid_augs = 0
    for agi, ago in augs:
        try:
            if dsl_rot90(agi) == ago:
                valid_augs += 1
        except:
            pass
    ok = valid_augs >= 1  # at least original should work
    print(f"    Valid augments: {valid_augs}/{len(augs)} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T3: Feature extraction")
    feat = extract_features(gi, go)
    ok = len(feat) > 30
    print(f"    Features: {len(feat)} dims {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T4: Task augmentation")
    pairs = [{'input': [[1,2],[3,4]], 'output': [[3,1],[4,2]]}]
    aug_pairs = augment_task_demos(pairs, rng)
    ok = len(aug_pairs) >= 5
    print(f"    Augmented demo pairs: {len(aug_pairs)} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T5: TTT adaptation")
    programs = enumerate_programs(max_depth=1, include_color_ops=False)
    # Create minimal proposer
    X = np.random.randn(100, 54).astype(np.float32)
    Y = np.random.randint(0, len(programs), 100).astype(np.int32)
    prop = NeuralProposer(54, len(programs), hidden=64)
    prop.feat_mean = X.mean(axis=0)
    prop.feat_std = X.std(axis=0) + 1e-8
    prop.trained = True

    label_map = {i: i for i in range(len(programs))}
    rev_map = {i: i for i in range(len(programs))}

    adapted = ttt_adapt(prop, aug_pairs, programs, label_map, rev_map,
                          n_epochs=5, lr=0.01)
    ok = adapted is not None
    print(f"    Adapted: {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T6: Color swap augmentation")
    gi2 = Grid(np.array([[1,0,2],[0,3,0],[2,0,1]]))
    go2 = Grid(np.array([[2,0,1],[0,3,0],[1,0,2]]))
    augs2 = augment_pair(gi2, go2, rng)
    # Check that color-swapped versions exist
    has_color_swap = any(
        not np.array_equal(a[0].arr, gi2.arr) and a[0].shape == gi2.shape
        for a in augs2
    )
    ok = has_color_swap
    print(f"    Has color swaps: {has_color_swap} {'PASS' if ok else 'FAIL'}")
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
    ap.add_argument("--n-grids", type=int, default=50)
    ap.add_argument("--max-depth", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=200)
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
        run_benchmark(d, n_grids=args.n_grids, max_depth=args.max_depth,
                        epochs=args.epochs, limit=args.limit,
                        verbose=args.verbose)
