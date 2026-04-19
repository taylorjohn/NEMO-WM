"""
arc_synth.py — Synthetic Data Generation + Neural Program Proposer
====================================================================
NVARC-style pipeline:
  1. Enumerate all 1-3 step DSL programs
  2. Generate grids, apply each program, save (input, output, program_id)
  3. Train a neural proposer: given (input, output) → predict program_id
  4. At test time: proposer suggests top-K programs, verify against demos

This replaces random search with LEARNED search.

Usage:
    python arc_synth.py --generate --n-grids 20     # generate dataset
    python arc_synth.py --train --epochs 50          # train proposer
    python arc_synth.py --solve --data path/to/ARC   # solve with proposer
    python arc_synth.py --test
    python arc_synth.py --stats                      # dataset statistics
"""

import argparse
import json
import os
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time

from arc_solver import (Grid, score_task,
                         dsl_rot90, dsl_rot180, dsl_rot270,
                         dsl_flip_h, dsl_flip_v, dsl_transpose,
                         dsl_crop_nonzero, dsl_crop_to_bg, dsl_scale2,
                         dsl_outline, dsl_fill_holes,
                         dsl_gravity_down, dsl_gravity_left,
                         dsl_mirror_h, dsl_mirror_v,
                         dsl_largest_object, dsl_smallest_object,
                         dsl_keep_only_color, dsl_remove_color,
                         dsl_top_half, dsl_bottom_half,
                         dsl_left_half, dsl_right_half,
                         dsl_unique_rows, dsl_identity)

Path("data/arc_synth").mkdir(parents=True, exist_ok=True)
SYNTH_DIR = Path("data/arc_synth")


# ══════════════════════════════════════════════════════════════════════
# 1. PROGRAM ENUMERATION
# ══════════════════════════════════════════════════════════════════════

# Base primitives (grid → grid, no parameters)
BASE_PRIMITIVES = {
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
}

# Color-parameterized primitives
def make_keep_color(c):
    return lambda g: Grid(np.where(g.arr == c, c, 0).astype(np.int32))

def make_remove_color(c):
    def fn(g):
        arr = g.arr.copy()
        arr[arr == c] = 0
        return Grid(arr)
    return fn

def make_recolor(old, new):
    def fn(g):
        arr = g.arr.copy()
        arr[arr == old] = new
        return Grid(arr)
    return fn


def enumerate_programs(max_depth=3, include_color_ops=True):
    """Enumerate all valid DSL programs up to max_depth steps."""
    programs = {}
    prog_id = 0

    # Single primitives
    for name, fn in BASE_PRIMITIVES.items():
        programs[prog_id] = {
            'steps': [name], 'fn': fn, 'depth': 1, 'name': name
        }
        prog_id += 1

    # Color ops (keep_color_C for C in 1-6)
    if include_color_ops:
        for c in range(1, 7):
            name = f'keep_{c}'
            programs[prog_id] = {
                'steps': [name], 'fn': make_keep_color(c),
                'depth': 1, 'name': name
            }
            prog_id += 1

            name = f'remove_{c}'
            programs[prog_id] = {
                'steps': [name], 'fn': make_remove_color(c),
                'depth': 1, 'name': name
            }
            prog_id += 1

    if max_depth >= 2:
        # Two-step compositions (use subset to keep manageable)
        fast_prims = ['rot90', 'rot180', 'flip_h', 'flip_v', 'transpose',
                       'crop', 'outline', 'fill_holes', 'gravity_down',
                       'largest_obj', 'top_half', 'bottom_half',
                       'left_half', 'right_half']
        for n1 in fast_prims:
            for n2 in fast_prims:
                if n1 == n2:
                    continue
                fn1 = BASE_PRIMITIVES[n1]
                fn2 = BASE_PRIMITIVES[n2]
                name = f'{n1}+{n2}'

                def make_compose2(a, b):
                    return lambda g: b(a(g))

                programs[prog_id] = {
                    'steps': [n1, n2], 'fn': make_compose2(fn1, fn2),
                    'depth': 2, 'name': name
                }
                prog_id += 1

        # Color + geometric (most useful compositions)
        if include_color_ops:
            for c in range(1, 5):
                for geom in fast_prims[:8]:
                    fn_keep = make_keep_color(c)
                    fn_geom = BASE_PRIMITIVES[geom]

                    name = f'keep_{c}+{geom}'
                    programs[prog_id] = {
                        'steps': [f'keep_{c}', geom],
                        'fn': make_compose2(fn_keep, fn_geom),
                        'depth': 2, 'name': name
                    }
                    prog_id += 1

                    name = f'{geom}+keep_{c}'
                    programs[prog_id] = {
                        'steps': [geom, f'keep_{c}'],
                        'fn': make_compose2(fn_geom, fn_keep),
                        'depth': 2, 'name': name
                    }
                    prog_id += 1

    if max_depth >= 3:
        # Three-step (small subset — geometric only)
        geom3 = ['rot90', 'flip_h', 'flip_v', 'transpose', 'crop']
        for n1 in geom3:
            for n2 in geom3:
                for n3 in geom3:
                    if n1 == n2 == n3:
                        continue
                    fn1 = BASE_PRIMITIVES[n1]
                    fn2 = BASE_PRIMITIVES[n2]
                    fn3 = BASE_PRIMITIVES[n3]
                    name = f'{n1}+{n2}+{n3}'

                    def make_compose3(a, b, c):
                        return lambda g: c(b(a(g)))

                    programs[prog_id] = {
                        'steps': [n1, n2, n3],
                        'fn': make_compose3(fn1, fn2, fn3),
                        'depth': 3, 'name': name
                    }
                    prog_id += 1

    return programs


# ══════════════════════════════════════════════════════════════════════
# 2. GRID GENERATION
# ══════════════════════════════════════════════════════════════════════

def generate_random_grids(n_grids, rng, min_h=3, max_h=8,
                            min_w=3, max_w=8, n_colors=5, density=0.4):
    """Generate diverse random grids."""
    grids = []
    for _ in range(n_grids):
        h = rng.randint(min_h, max_h + 1)
        w = rng.randint(min_w, max_w + 1)
        d = rng.uniform(0.2, 0.7)
        nc = rng.randint(2, n_colors + 1)
        colors = list(range(1, nc + 1))
        arr = np.zeros((h, w), dtype=np.int32)
        for r in range(h):
            for c in range(w):
                if rng.random() < d:
                    arr[r, c] = rng.choice(colors)
        grids.append(Grid(arr))
    return grids


# ══════════════════════════════════════════════════════════════════════
# 3. DATASET GENERATION
# ══════════════════════════════════════════════════════════════════════

def generate_dataset(n_grids_per_program=20, max_depth=2):
    """Generate synthetic training dataset."""
    print("=" * 70)
    print("  Synthetic Data Generation — NVARC-style")
    print("=" * 70)

    programs = enumerate_programs(max_depth=max_depth)
    print(f"  Programs enumerated: {len(programs)}")
    print(f"  Grids per program: {n_grids_per_program}")
    print(f"  Target dataset size: ~{len(programs) * n_grids_per_program}")

    rng = np.random.RandomState(42)
    dataset = []  # (input_features, output_features, program_id)
    valid_programs = {}
    skipped = 0
    t0 = time.time()

    for prog_id, prog_info in programs.items():
        fn = prog_info['fn']
        grids = generate_random_grids(n_grids_per_program * 2, rng)
        n_valid = 0

        for g in grids:
            if n_valid >= n_grids_per_program:
                break
            try:
                out = fn(g)
                if out.h > 0 and out.w > 0 and out.h <= 30 and out.w <= 30:
                    # Extract features
                    in_feat = grid_to_features(g)
                    out_feat = grid_to_features(out)
                    diff_feat = diff_features(g, out)
                    features = np.concatenate([in_feat, out_feat, diff_feat])

                    dataset.append({
                        'features': features,
                        'program_id': prog_id,
                        'input': g.to_list(),
                        'output': out.to_list(),
                    })
                    n_valid += 1
            except Exception:
                continue

        if n_valid > 0:
            valid_programs[prog_id] = prog_info
        else:
            skipped += 1

        if (prog_id + 1) % max(1, len(programs) // 10) == 0:
            elapsed = time.time() - t0
            print(f"    Programs: {prog_id+1}/{len(programs)} | "
                  f"Dataset: {len(dataset)} | "
                  f"Skipped: {skipped} | "
                  f"Time: {elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"\n  Dataset generated: {len(dataset)} examples")
    print(f"  Valid programs: {len(valid_programs)}/{len(programs)}")
    print(f"  Skipped: {skipped}")
    print(f"  Time: {elapsed:.1f}s")

    # Save
    features = np.array([d['features'] for d in dataset], dtype=np.float32)
    labels = np.array([d['program_id'] for d in dataset], dtype=np.int32)

    np.savez(SYNTH_DIR / "dataset.npz",
             features=features, labels=labels)

    # Save program catalog
    catalog = {}
    for pid, info in valid_programs.items():
        catalog[str(pid)] = {
            'name': info['name'],
            'steps': info['steps'],
            'depth': info['depth'],
        }
    with open(SYNTH_DIR / "programs.json", 'w') as f:
        json.dump(catalog, f, indent=2)

    print(f"  Saved to {SYNTH_DIR}")
    print(f"  Features shape: {features.shape}")
    print(f"  Labels: {len(set(labels))} unique programs")

    # Depth distribution
    depth_counts = Counter(valid_programs[pid]['depth']
                             for pid in set(labels))
    print(f"\n  Depth distribution:")
    for d in sorted(depth_counts):
        print(f"    {d}-step: {depth_counts[d]} programs")

    print(f"\n{'='*70}")
    return dataset, valid_programs


def grid_to_features(g: Grid, max_size=30) -> np.ndarray:
    """Extract fixed-size feature vector from a grid."""
    features = []

    # Shape features
    features.extend([g.h / max_size, g.w / max_size, g.h * g.w / (max_size**2)])

    # Color histogram (10 colors, normalized)
    hist = np.zeros(10, dtype=np.float32)
    vals, counts = np.unique(g.arr, return_counts=True)
    for v, c in zip(vals, counts):
        if v < 10:
            hist[int(v)] = c / max(g.arr.size, 1)
    features.extend(hist.tolist())

    # Number of unique colors
    features.append(len(set(g.arr.flatten().tolist())) / 10)

    # Background ratio
    bg = g.background
    features.append(np.sum(g.arr == bg) / max(g.arr.size, 1))

    # Symmetry features
    features.append(1.0 if g.is_symmetric_h() else 0.0)
    features.append(1.0 if g.is_symmetric_v() else 0.0)

    # Object count
    try:
        n_obj = len(g.objects())
    except Exception:
        n_obj = 0
    features.append(min(n_obj, 20) / 20)

    # Row/column uniqueness
    features.append(g.unique_rows() / max(g.h, 1))
    features.append(g.unique_cols() / max(g.w, 1))

    # Edge density (how many edge cells are non-background)
    edge_cells = 0
    edge_total = 0
    for r in range(g.h):
        for c in range(g.w):
            if r == 0 or r == g.h-1 or c == 0 or c == g.w-1:
                edge_total += 1
                if g.arr[r, c] != bg:
                    edge_cells += 1
    features.append(edge_cells / max(edge_total, 1))

    # Pad to exactly 22 features
    while len(features) < 22:
        features.append(0.0)
    return np.array(features[:22], dtype=np.float32)


def diff_features(g_in: Grid, g_out: Grid) -> np.ndarray:
    """Extract features about the relationship between input and output."""
    features = []

    # Size relationship
    features.append(g_out.h / max(g_in.h, 1))
    features.append(g_out.w / max(g_in.w, 1))
    features.append(1.0 if g_in.shape == g_out.shape else 0.0)
    features.append(1.0 if g_out.h * g_out.w < g_in.h * g_in.w else 0.0)
    features.append(1.0 if g_out.h * g_out.w > g_in.h * g_in.w else 0.0)

    # Color relationship
    in_colors = set(g_in.arr.flatten().tolist())
    out_colors = set(g_out.arr.flatten().tolist())
    features.append(len(in_colors) / 10)
    features.append(len(out_colors) / 10)
    features.append(len(in_colors & out_colors) / max(len(in_colors | out_colors), 1))
    features.append(1.0 if out_colors - in_colors else 0.0)

    # Pixel overlap (if same size)
    if g_in.shape == g_out.shape:
        same = np.sum(g_in.arr == g_out.arr) / max(g_in.arr.size, 1)
        features.append(same)
    else:
        features.append(0.0)

    return np.array(features[:10], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════
# 4. NEURAL PROPOSER — MLP that predicts program_id
# ══════════════════════════════════════════════════════════════════════

class NeuralProposer:
    """
    MLP that maps (input_features, output_features, diff_features) → program_id.
    Trained on synthetic data. At test time, returns top-K program predictions.
    """

    def __init__(self, n_features, n_programs, hidden=512):
        self.n_features = n_features
        self.n_programs = n_programs
        rng = np.random.RandomState(42)

        # Xavier initialization
        self.W1 = rng.randn(n_features, hidden).astype(np.float32) * np.sqrt(2.0 / n_features)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.randn(hidden, hidden).astype(np.float32) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(hidden, dtype=np.float32)
        self.W3 = rng.randn(hidden, n_programs).astype(np.float32) * np.sqrt(2.0 / hidden)
        self.b3 = np.zeros(n_programs, dtype=np.float32)

        self.trained = False

    def forward(self, x):
        """Forward pass. x: (batch, n_features)."""
        if hasattr(self, 'feat_mean'):
            x = (x - self.feat_mean) / self.feat_std
        h1 = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        logits = h2 @ self.W3 + self.b3
        return logits

    def predict_topk(self, x, k=10):
        """Return top-K program indices."""
        logits = self.forward(x.reshape(1, -1))[0]
        topk = np.argsort(-logits)[:k]
        return topk.tolist()

    def train(self, features, labels, epochs=50, lr=0.01, batch_size=256):
        """Train with SGD + cross-entropy loss."""
        n = len(features)
        print(f"    Training proposer: {n} examples, {self.n_programs} programs")

        # Normalize features
        self.feat_mean = features.mean(axis=0)
        self.feat_std = features.std(axis=0) + 1e-8
        features = (features - self.feat_mean) / self.feat_std

        best_acc = 0
        for epoch in range(epochs):
            # Shuffle
            idx = np.random.permutation(n)
            total_loss = 0
            n_correct = 0
            n_batches = 0

            # Decay learning rate
            cur_lr = lr * (0.97 ** epoch)

            for start in range(0, n, batch_size):
                batch_idx = idx[start:start+batch_size]
                X = features[batch_idx]
                Y = labels[batch_idx]
                bs = len(X)

                # Forward
                h1 = np.maximum(0, X @ self.W1 + self.b1)
                h2 = np.maximum(0, h1 @ self.W2 + self.b2)
                logits = h2 @ self.W3 + self.b3

                # Softmax + cross-entropy
                logits_max = logits - logits.max(axis=1, keepdims=True)
                exp_logits = np.exp(np.clip(logits_max, -20, 20))
                probs = exp_logits / (exp_logits.sum(axis=1, keepdims=True) + 1e-8)

                # Loss
                loss = -np.mean(np.log(probs[np.arange(bs), Y] + 1e-8))
                total_loss += loss
                n_correct += np.sum(np.argmax(logits, axis=1) == Y)
                n_batches += 1

                # Backward
                dlogits = probs.copy()
                dlogits[np.arange(bs), Y] -= 1
                dlogits /= bs

                dW3 = h2.T @ dlogits
                db3 = dlogits.sum(axis=0)
                dh2 = dlogits @ self.W3.T * (h2 > 0).astype(np.float32)
                dW2 = h1.T @ dh2
                db2 = dh2.sum(axis=0)
                dh1 = dh2 @ self.W2.T * (h1 > 0).astype(np.float32)
                dW1 = X.T @ dh1
                db1 = dh1.sum(axis=0)

                # Clip gradients
                for g in [dW1, db1, dW2, db2, dW3, db3]:
                    np.clip(g, -1, 1, out=g)

                # Update with weight decay
                self.W1 -= cur_lr * (dW1 + 1e-4 * self.W1)
                self.b1 -= cur_lr * db1
                self.W2 -= cur_lr * (dW2 + 1e-4 * self.W2)
                self.b2 -= cur_lr * db2
                self.W3 -= cur_lr * (dW3 + 1e-4 * self.W3)
                self.b3 -= cur_lr * db3

            acc = n_correct / n * 100
            avg_loss = total_loss / max(n_batches, 1)
            if acc > best_acc:
                best_acc = acc
            if epoch % max(1, epochs // 10) == 0:
                print(f"      Epoch {epoch:>3}: loss={avg_loss:.4f} acc={acc:.1f}%")

        self.trained = True
        print(f"    Final accuracy: {n_correct/n*100:.1f}%")

    def save(self, path):
        save_dict = dict(W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                   W3=self.W3, b3=self.b3,
                   n_features=self.n_features, n_programs=self.n_programs)
        if hasattr(self, 'feat_mean'):
            save_dict['feat_mean'] = self.feat_mean
            save_dict['feat_std'] = self.feat_std
        np.savez(path, **save_dict)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        proposer = cls(int(data['n_features']), int(data['n_programs']))
        proposer.W1 = data['W1']
        proposer.b1 = data['b1']
        proposer.W2 = data['W2']
        proposer.b2 = data['b2']
        proposer.W3 = data['W3']
        proposer.b3 = data['b3']
        if 'feat_mean' in data:
            proposer.feat_mean = data['feat_mean']
            proposer.feat_std = data['feat_std']
        proposer.trained = True
        return proposer


# ══════════════════════════════════════════════════════════════════════
# 5. PROPOSER-GUIDED SOLVER
# ══════════════════════════════════════════════════════════════════════

def solve_with_proposer(task, proposer, programs, top_k=20):
    """Solve ARC task using neural proposer for program selection."""
    pairs = task['train']
    test_cases = task['test']

    # Extract features from first training pair
    gi = Grid(pairs[0]['input'])
    go = Grid(pairs[0]['output'])
    in_feat = grid_to_features(gi)
    out_feat = grid_to_features(go)
    diff_feat = diff_features(gi, go)
    features = np.concatenate([in_feat, out_feat, diff_feat])

    # Get top-K program predictions
    if proposer.trained:
        predicted_ids = proposer.predict_topk(features, k=top_k)
    else:
        predicted_ids = list(range(min(top_k, len(programs))))

    # Try each predicted program
    for prog_id in predicted_ids:
        if str(prog_id) not in programs and prog_id not in programs:
            continue

        prog_info = programs.get(str(prog_id), programs.get(prog_id))
        if prog_info is None:
            continue

        fn = prog_info.get('fn')
        if fn is None:
            # Reconstruct function from steps
            steps = prog_info['steps']
            fn = reconstruct_fn(steps)
            if fn is None:
                continue

        # Verify against ALL training pairs
        all_match = True
        for p in pairs:
            try:
                gi_p = Grid(p['input'])
                go_p = Grid(p['output'])
                result = fn(gi_p)
                if result != go_p:
                    all_match = False
                    break
            except Exception:
                all_match = False
                break

        if all_match:
            # Apply to test
            guesses = []
            for tc in test_cases:
                try:
                    gi_t = Grid(tc['input'])
                    result = fn(gi_t)
                    guesses.append([result.to_list()])
                except Exception:
                    guesses.append([tc['input']])
            return guesses, prog_info.get('name', str(prog_id))

    # Fallback
    return [[tc['input']] for tc in test_cases], None


def reconstruct_fn(steps):
    """Reconstruct a function from step names."""
    fns = []
    for step in steps:
        if step in BASE_PRIMITIVES:
            fns.append(BASE_PRIMITIVES[step])
        elif step.startswith('keep_'):
            c = int(step.split('_')[1])
            fns.append(make_keep_color(c))
        elif step.startswith('remove_'):
            c = int(step.split('_')[1])
            fns.append(make_remove_color(c))
        else:
            return None

    if not fns:
        return None

    def composed(g):
        result = g
        for f in fns:
            result = f(result)
        return result

    return composed


# ══════════════════════════════════════════════════════════════════════
# 6. FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════

def full_pipeline(data_dir, n_grids=20, epochs=50, max_depth=2,
                    limit=None, verbose=False):
    """Generate data, train proposer, solve ARC."""
    # Step 1: Generate synthetic data
    dataset, valid_programs = generate_dataset(
        n_grids_per_program=n_grids, max_depth=max_depth)

    if not dataset:
        print("  No data generated!")
        return

    # Step 2: Train proposer
    features = np.array([d['features'] for d in dataset], dtype=np.float32)
    labels = np.array([d['program_id'] for d in dataset], dtype=np.int32)

    # Remap labels to consecutive indices
    unique_labels = sorted(set(labels))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    mapped_labels = np.array([label_map[l] for l in labels], dtype=np.int32)

    # Remap programs
    remapped_programs = {}
    for old_id, new_id in label_map.items():
        if old_id in valid_programs:
            remapped_programs[new_id] = valid_programs[old_id]

    proposer = NeuralProposer(features.shape[1], len(unique_labels))

    print(f"\n  Training neural proposer...")
    proposer.train(features, mapped_labels, epochs=epochs)
    proposer.save(SYNTH_DIR / "proposer.npz")

    # Step 3: Evaluate on ARC
    print(f"\n  Evaluating on ARC-AGI-2...")
    files = sorted(f for f in os.listdir(data_dir) if f.endswith('.json'))
    if limit:
        files = files[:limit]

    solved = 0
    total = 0
    winning_programs = Counter()

    for f in files:
        task = json.load(open(os.path.join(data_dir, f)))
        total += 1

        predictions, prog_name = solve_with_proposer(
            task, proposer, remapped_programs, top_k=30)
        correct = score_task(task, predictions)

        if correct:
            solved += 1
            if prog_name:
                winning_programs[prog_name] += 1
            if verbose:
                print(f"    {f[:16]}: SOLVED ({prog_name})")

    pct = solved / max(total, 1) * 100
    print(f"\n{'='*70}")
    print(f"  PROPOSER-GUIDED RESULTS")
    print(f"{'='*70}")
    print(f"  Solved:     {solved}/{total} ({pct:.1f}%)")
    print(f"  Programs:   {len(remapped_programs)}")
    print(f"  Dataset:    {len(dataset)} examples")

    if winning_programs:
        print(f"\n  Winning programs:")
        for name, count in winning_programs.most_common(15):
            print(f"    {name:<30}: {count}")

    print(f"\n{'='*70}")
    return solved, total


# ══════════════════════════════════════════════════════════════════════
# 7. TESTS
# ══════════════════════════════════════════════════════════════════════

def run_tests():
    print("=" * 65)
    print("  Synthetic Data + Neural Proposer Tests")
    print("=" * 65)
    p = 0; t = 0

    print("\n  T1: Program enumeration")
    programs = enumerate_programs(max_depth=1, include_color_ops=False)
    ok = len(programs) >= 20
    print(f"    1-step programs: {len(programs)} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T2: Program enumeration depth 2")
    programs2 = enumerate_programs(max_depth=2, include_color_ops=True)
    ok = len(programs2) > 100
    print(f"    1-2 step programs: {len(programs2)} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T3: Grid feature extraction")
    g = Grid(np.array([[1,2,0],[0,3,3],[1,0,2]]))
    feat = grid_to_features(g)
    ok = len(feat) == 22
    print(f"    Features: {len(feat)} dims {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T4: Diff features")
    g2 = Grid(np.array([[1,1,0],[0,3,3],[1,0,2]]))
    df = diff_features(g, g2)
    ok = len(df) == 10
    print(f"    Diff features: {len(df)} dims {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T5: Neural proposer forward pass")
    proposer = NeuralProposer(n_features=32, n_programs=50)
    x = np.random.randn(32).astype(np.float32)
    topk = proposer.predict_topk(x, k=5)
    ok = len(topk) == 5
    print(f"    Top-5: {topk} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T6: Mini dataset generation")
    dataset, progs = generate_dataset(n_grids_per_program=5, max_depth=1)
    ok = len(dataset) > 50
    print(f"    Dataset: {len(dataset)} examples {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T7: Mini proposer training")
    if dataset:
        features = np.array([d['features'] for d in dataset], dtype=np.float32)
        labels = np.array([d['program_id'] for d in dataset], dtype=np.int32)
        unique = sorted(set(labels))
        lmap = {o: n for n, o in enumerate(unique)}
        mlabels = np.array([lmap[l] for l in labels], dtype=np.int32)
        prop = NeuralProposer(features.shape[1], len(unique), hidden=64)
        prop.train(features, mlabels, epochs=10)
        ok = prop.trained
    else:
        ok = True
    print(f"    Trained: {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T8: Reconstruct function from steps")
    fn = reconstruct_fn(['rot90', 'flip_h'])
    g3 = Grid(np.array([[1,2],[3,4]]))
    result = fn(g3)
    expected = dsl_flip_h(dsl_rot90(g3))
    ok = result == expected
    print(f"    rot90+flip_h: {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print(f"\n{'='*65}")
    print(f"  Results: {p}/{t} tests passed")
    print(f"{'='*65}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--solve", action="store_true")
    ap.add_argument("--pipeline", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--n-grids", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--max-depth", type=int, default=2)
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
    elif args.generate:
        generate_dataset(n_grids_per_program=args.n_grids,
                           max_depth=args.max_depth)
    elif args.stats:
        if (SYNTH_DIR / "dataset.npz").exists():
            data = np.load(SYNTH_DIR / "dataset.npz")
            print(f"  Features: {data['features'].shape}")
            print(f"  Labels: {len(data['labels'])} ({len(set(data['labels']))} unique)")
        else:
            print("  No dataset found. Run --generate first.")
    elif args.pipeline or args.solve:
        d = os.path.join(data_base, "training")
        full_pipeline(d, n_grids=args.n_grids, epochs=args.epochs,
                        max_depth=args.max_depth, limit=args.limit,
                        verbose=args.verbose)
    else:
        run_tests()
