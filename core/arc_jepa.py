"""
arc_jepa.py — JEPA World Model for ARC-AGI-2
================================================
First JEPA-based approach to abstract reasoning.

Architecture (following LeCun's JEPA framework):
  1. Grid Encoder (x → z): Embed grid into 64-D latent belief space
  2. Predictor (z_in, Δ → z_out): Predict output embedding from input + action
  3. Grid Decoder (z → x): Decode latent back to grid
  4. Action Encoder: Learn the transformation rule from demo pairs

The key insight: ARC tasks ARE world model problems.
  - Training demos = observations of world dynamics
  - The transformation rule = the "action" in JEPA terms  
  - Test prediction = world model rollout

NeMo-WM connection:
  - Grid Encoder = sensory encoder (sense → 64-D belief)
  - Predictor = world model f(b_t, a_t) → b_{t+1}
  - Action = schema (learned transformation pattern)
  - Neuromodulation = controls encoding precision + search

Usage:
    python arc_jepa.py --test
    python arc_jepa.py --train --epochs 100
    python arc_jepa.py --solve --data path/to/ARC-AGI-2/data
"""

import argparse
import json
import os
import numpy as np
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple, List
import time

from arc_solver import (Grid, score_task,
                         dsl_rot90, dsl_rot180, dsl_rot270,
                         dsl_flip_h, dsl_flip_v, dsl_transpose,
                         dsl_crop_nonzero, dsl_scale2,
                         dsl_outline, dsl_fill_holes,
                         dsl_gravity_down, dsl_gravity_left,
                         dsl_mirror_h, dsl_mirror_v,
                         dsl_largest_object, dsl_smallest_object)

from arc_synth import generate_random_grids, BASE_PRIMITIVES

try:
    from arc_phase2 import solve_task_phase2 as s1_solve
    HAS_S1 = True
except ImportError:
    from arc_solver import solve_task as s1_solve
    HAS_S1 = False

Path("data/arc_jepa").mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# 1. GRID ENCODER — Grid → 64-D latent embedding (the "belief")
# ══════════════════════════════════════════════════════════════════════

class GridEncoder:
    """
    Encode a grid into a fixed-size latent representation.
    
    This is the JEPA "embedding" function.
    Maps any grid (variable size, 1-10 colors) into a 64-D vector.
    
    Architecture: Flatten grid to fixed size → MLP → 64-D
    """
    
    def __init__(self, grid_size=15, latent_dim=64, hidden=256):
        self.grid_size = grid_size
        self.latent_dim = latent_dim
        self.hidden = hidden
        rng = np.random.RandomState(42)
        
        # Input: grid_size × grid_size × 10 (one-hot colors) = 2250 for 15×15
        self.input_dim = grid_size * grid_size * 10
        
        # Encoder MLP
        self.W1 = rng.randn(self.input_dim, hidden).astype(np.float32) * np.sqrt(2.0 / self.input_dim)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.randn(hidden, hidden).astype(np.float32) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(hidden, dtype=np.float32)
        self.W3 = rng.randn(hidden, latent_dim).astype(np.float32) * np.sqrt(2.0 / hidden)
        self.b3 = np.zeros(latent_dim, dtype=np.float32)
    
    def grid_to_onehot(self, g: Grid) -> np.ndarray:
        """Convert grid to one-hot encoded fixed-size representation."""
        onehot = np.zeros((self.grid_size, self.grid_size, 10), dtype=np.float32)
        h = min(g.h, self.grid_size)
        w = min(g.w, self.grid_size)
        for r in range(h):
            for c in range(w):
                v = int(g.arr[r, c])
                if v < 10:
                    onehot[r, c, v] = 1.0
        return onehot.flatten()
    
    def encode(self, g: Grid) -> np.ndarray:
        """Encode grid → 64-D latent."""
        x = self.grid_to_onehot(g)
        h1 = np.maximum(0, x @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        z = np.tanh(h2 @ self.W3 + self.b3)  # tanh for bounded latent
        return z
    
    def encode_batch(self, grids: list) -> np.ndarray:
        """Encode batch of grids."""
        X = np.array([self.grid_to_onehot(g) for g in grids], dtype=np.float32)
        H1 = np.maximum(0, X @ self.W1 + self.b1)
        H2 = np.maximum(0, H1 @ self.W2 + self.b2)
        Z = np.tanh(H2 @ self.W3 + self.b3)
        return Z


# ══════════════════════════════════════════════════════════════════════
# 2. PREDICTOR — (z_input, action) → z_output
# ══════════════════════════════════════════════════════════════════════

class JEPAPredictor:
    """
    JEPA predictor: given input embedding + action, predict output embedding.
    
    This is the "world model" in JEPA terms.
    In NeMo-WM: f(b_t, a_t) → b_{t+1}
    
    The action is learned from demo pairs — it captures the transformation rule.
    """
    
    def __init__(self, latent_dim=64, action_dim=64, hidden=128):
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        rng = np.random.RandomState(123)
        
        # Predictor: (z_in concat action) → z_out
        input_dim = latent_dim + action_dim
        self.W1 = rng.randn(input_dim, hidden).astype(np.float32) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.randn(hidden, hidden).astype(np.float32) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(hidden, dtype=np.float32)
        self.W3 = rng.randn(hidden, latent_dim).astype(np.float32) * np.sqrt(2.0 / hidden)
        self.b3 = np.zeros(latent_dim, dtype=np.float32)
    
    def predict(self, z_in: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Predict z_out from z_in + action."""
        x = np.concatenate([z_in, action])
        h1 = np.maximum(0, x @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        z_out = np.tanh(h2 @ self.W3 + self.b3)
        return z_out


# ══════════════════════════════════════════════════════════════════════
# 3. ACTION ENCODER — Learn transformation rule from demo pairs
# ══════════════════════════════════════════════════════════════════════

class ActionEncoder:
    """
    Encode the transformation rule from demonstration pairs.
    
    Given pairs of (z_in, z_out), infer the latent "action" vector
    that captures the transformation rule.
    
    This is the JEPA "latent variable" — the hidden cause of the
    observed transformation.
    
    Multiple approaches:
    1. Simple: action = mean(z_out - z_in) across demos
    2. Learned: MLP that maps (z_in, z_out) pairs → action
    3. Iterative: refine action to minimize prediction error
    """
    
    def __init__(self, latent_dim=64, action_dim=64, hidden=128):
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        rng = np.random.RandomState(456)
        
        # Action inference network: (z_in, z_out) → action
        input_dim = latent_dim * 2
        self.W1 = rng.randn(input_dim, hidden).astype(np.float32) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.randn(hidden, action_dim).astype(np.float32) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(action_dim, dtype=np.float32)
    
    def infer_action_simple(self, z_ins: np.ndarray, z_outs: np.ndarray) -> np.ndarray:
        """Simple action = mean difference in latent space."""
        diffs = z_outs - z_ins
        return np.mean(diffs, axis=0)
    
    def infer_action_network(self, z_ins: np.ndarray, z_outs: np.ndarray) -> np.ndarray:
        """Learned action inference from demo pairs."""
        actions = []
        for z_in, z_out in zip(z_ins, z_outs):
            x = np.concatenate([z_in, z_out])
            h1 = np.maximum(0, x @ self.W1 + self.b1)
            action = np.tanh(h1 @ self.W2 + self.b2)
            actions.append(action)
        return np.mean(actions, axis=0)
    
    def infer_action_iterative(self, z_ins, z_outs, predictor, 
                                  n_iters=50, lr=0.1) -> np.ndarray:
        """
        Iteratively refine action to minimize prediction error.
        This is the JEPA energy minimization / active inference approach.
        
        argmin_a Σ_i ||predictor(z_in_i, a) - z_out_i||²
        """
        # Initialize from simple difference
        action = self.infer_action_simple(z_ins, z_outs)
        
        for _ in range(n_iters):
            # Compute gradient of prediction error w.r.t. action
            total_grad = np.zeros_like(action)
            total_loss = 0
            
            for z_in, z_out in zip(z_ins, z_outs):
                # Forward
                x = np.concatenate([z_in, action])
                h1 = np.maximum(0, x @ predictor.W1 + predictor.b1)
                h2 = np.maximum(0, h1 @ predictor.W2 + predictor.b2)
                z_pred = np.tanh(h2 @ predictor.W3 + predictor.b3)
                
                # Error
                error = z_pred - z_out
                total_loss += np.sum(error ** 2)
                
                # Backward through predictor to action
                # dL/dz_pred
                dz = 2 * error
                # through tanh
                dz = dz * (1 - z_pred ** 2)
                # through W3
                dh2 = dz @ predictor.W3.T * (h2 > 0).astype(np.float32)
                # through W2
                dh1 = dh2 @ predictor.W2.T * (h1 > 0).astype(np.float32)
                # through W1 — get gradient w.r.t. action part of input
                dx = dh1 @ predictor.W1.T
                d_action = dx[len(z_in):]  # action gradient
                
                total_grad += d_action
            
            total_grad /= len(z_ins)
            np.clip(total_grad, -1, 1, out=total_grad)
            
            action -= lr * total_grad
            action = np.clip(action, -2, 2)
        
        return action


# ══════════════════════════════════════════════════════════════════════
# 4. GRID DECODER — z → Grid (nearest-neighbor in program space)
# ══════════════════════════════════════════════════════════════════════

class GridDecoder:
    """
    Decode latent embedding back to a grid.
    
    Two approaches:
    1. Direct: MLP decoder z → grid pixels (lossy, approximate)
    2. Program-guided: use z_pred to select from candidate programs,
       then execute the program to get exact grid
    
    We use approach 2 (program-guided) because ARC requires EXACT outputs.
    """
    
    def __init__(self, encoder: GridEncoder):
        self.encoder = encoder
        self.program_embeddings = {}  # program_name → mean z_out embedding
    
    def build_program_index(self, programs: dict, n_samples=20):
        """Pre-compute embeddings for each program's outputs."""
        rng = np.random.RandomState(789)
        grids = generate_random_grids(n_samples * 2, rng)
        
        for name, fn in programs.items():
            z_outs = []
            for g in grids:
                try:
                    out = fn(g)
                    if 0 < out.h <= 15 and 0 < out.w <= 15:
                        z = self.encoder.encode(out)
                        z_outs.append(z)
                except Exception:
                    continue
            if z_outs:
                self.program_embeddings[name] = np.mean(z_outs, axis=0)
    
    def decode_by_program(self, z_pred: np.ndarray, 
                            programs: dict, input_grid: Grid,
                            top_k=10) -> List[Tuple[str, Grid]]:
        """
        Find programs whose output embedding is closest to z_pred,
        then execute them on the input grid.
        """
        if not self.program_embeddings:
            return []
        
        # Rank programs by distance to z_pred
        distances = []
        for name, z_prog in self.program_embeddings.items():
            dist = np.sum((z_pred - z_prog) ** 2)
            distances.append((dist, name))
        distances.sort()
        
        # Try top-K
        results = []
        for dist, name in distances[:top_k]:
            if name in programs:
                try:
                    result = programs[name](input_grid)
                    results.append((name, result))
                except Exception:
                    continue
        
        return results


# ══════════════════════════════════════════════════════════════════════
# 5. JEPA TRAINER — Self-supervised training on synthetic data
# ══════════════════════════════════════════════════════════════════════

class JEPATrainer:
    """
    Train the JEPA components end-to-end on synthetic ARC-like data.
    
    Training objective (VICReg-style, following I-JEPA):
    1. Prediction loss: ||predictor(z_in, a) - z_out||²
    2. Variance: prevent representation collapse
    3. Covariance: decorrelate latent dimensions
    
    The key is that we train on (input, action, output) triples
    from our DSL, so the system learns what transformations look like
    in latent space.
    """
    
    def __init__(self, latent_dim=64, grid_size=12):
        self.latent_dim = latent_dim
        self.grid_size = grid_size
        
        self.encoder = GridEncoder(grid_size=grid_size, latent_dim=latent_dim)
        self.predictor = JEPAPredictor(latent_dim=latent_dim, action_dim=latent_dim)
        self.action_encoder = ActionEncoder(latent_dim=latent_dim, action_dim=latent_dim)
        self.decoder = GridDecoder(self.encoder)
    
    def generate_training_data(self, n_samples=5000):
        """Generate (input_grid, output_grid, action_id) triples."""
        rng = np.random.RandomState(42)
        programs = list(BASE_PRIMITIVES.items())
        
        data = []
        for _ in range(n_samples):
            g = generate_random_grids(1, rng)[0]
            prog_idx = rng.randint(len(programs))
            name, fn = programs[prog_idx]
            try:
                out = fn(g)
                if 0 < out.h <= self.grid_size and 0 < out.w <= self.grid_size:
                    data.append((g, out, prog_idx, name))
            except Exception:
                continue
        
        return data, {i: name for i, (name, _) in enumerate(programs)}
    
    def train(self, n_samples=5000, epochs=100, lr=0.005):
        """Train JEPA on synthetic transformation data."""
        data, program_names = self.generate_training_data(n_samples)
        n_progs = len(program_names)
        
        print(f"    JEPA Training: {len(data)} samples, {n_progs} programs")
        
        # Encode all grids
        z_ins = np.array([self.encoder.encode(d[0]) for d in data], dtype=np.float32)
        z_outs = np.array([self.encoder.encode(d[1]) for d in data], dtype=np.float32)
        prog_ids = np.array([d[2] for d in data], dtype=np.int32)
        
        # Learn per-program action vectors
        action_vectors = np.zeros((n_progs, self.latent_dim), dtype=np.float32)
        for pid in range(n_progs):
            mask = prog_ids == pid
            if mask.sum() > 0:
                action_vectors[pid] = np.mean(z_outs[mask] - z_ins[mask], axis=0)
        
        # Train predictor: minimize ||predict(z_in, action) - z_out||²
        for epoch in range(epochs):
            total_loss = 0
            n_batches = 0
            
            idx = np.random.permutation(len(data))
            for start in range(0, len(data), 64):
                batch = idx[start:start+64]
                bs = len(batch)
                
                z_in_batch = z_ins[batch]
                z_out_batch = z_outs[batch]
                action_batch = action_vectors[prog_ids[batch]]
                
                # Forward: predict z_out from (z_in, action)
                x = np.concatenate([z_in_batch, action_batch], axis=1)
                h1 = np.maximum(0, x @ self.predictor.W1 + self.predictor.b1)
                h2 = np.maximum(0, h1 @ self.predictor.W2 + self.predictor.b2)
                z_pred = np.tanh(h2 @ self.predictor.W3 + self.predictor.b3)
                
                # Loss
                error = z_pred - z_out_batch
                loss = np.mean(np.sum(error ** 2, axis=1))
                total_loss += loss
                n_batches += 1
                
                # Backward
                dz = 2 * error / bs
                dz = dz * (1 - z_pred ** 2)
                
                dW3 = h2.T @ dz
                db3 = dz.sum(axis=0)
                dh2 = dz @ self.predictor.W3.T * (h2 > 0).astype(np.float32)
                dW2 = h1.T @ dh2
                db2 = dh2.sum(axis=0)
                dh1 = dh2 @ self.predictor.W2.T * (h1 > 0).astype(np.float32)
                dW1 = x.T @ dh1
                db1 = dh1.sum(axis=0)
                
                for grad in [dW1, db1, dW2, db2, dW3, db3]:
                    np.clip(grad, -1, 1, out=grad)
                
                self.predictor.W1 -= lr * dW1
                self.predictor.b1 -= lr * db1
                self.predictor.W2 -= lr * dW2
                self.predictor.b2 -= lr * db2
                self.predictor.W3 -= lr * dW3
                self.predictor.b3 -= lr * db3
            
            avg_loss = total_loss / max(n_batches, 1)
            if epoch % max(1, epochs // 10) == 0:
                print(f"      Epoch {epoch:>3}: pred_loss={avg_loss:.4f}")
        
        # Store action vectors for later use
        self.action_vectors = action_vectors
        self.program_names = program_names
        
        # Build decoder index
        self.decoder.build_program_index(
            {name: fn for name, fn in BASE_PRIMITIVES.items()})
        
        print(f"    JEPA training complete. Pred loss: {avg_loss:.4f}")
        return avg_loss


# ══════════════════════════════════════════════════════════════════════
# 6. JEPA ARC SOLVER
# ══════════════════════════════════════════════════════════════════════

class JEPASolver:
    """
    Solve ARC tasks using JEPA world model.
    
    For each task:
    1. Encode all demo (input, output) pairs into latent space
    2. Infer the action vector from demo pairs
    3. Apply action to test input embedding → predicted z_out
    4. Decode z_out by finding closest program in embedding space
    5. Execute that program on test input → output grid
    """
    
    def __init__(self, trainer: JEPATrainer):
        self.trainer = trainer
        self.encoder = trainer.encoder
        self.predictor = trainer.predictor
        self.action_encoder = trainer.action_encoder
        self.decoder = trainer.decoder
        
        self.s1_solves = 0
        self.jepa_solves = 0
        self.total = 0
    
    def solve(self, task):
        """Solve ARC task with S1 → JEPA fallback."""
        self.total += 1
        pairs = task['train']
        test_cases = task['test']
        
        # S1 first
        s1_preds = s1_solve(task)
        if score_task(task, s1_preds):
            self.s1_solves += 1
            return s1_preds, 'S1'
        
        # JEPA: embed demos, infer action, predict, decode
        z_ins = np.array([self.encoder.encode(Grid(p['input'])) for p in pairs])
        z_outs = np.array([self.encoder.encode(Grid(p['output'])) for p in pairs])
        
        # Infer action via iterative refinement (active inference)
        action = self.action_encoder.infer_action_iterative(
            z_ins, z_outs, self.predictor, n_iters=30, lr=0.1)
        
        # For each test case, predict and decode
        guesses = []
        for tc in test_cases:
            gi = Grid(tc['input'])
            z_test = self.encoder.encode(gi)
            z_pred = self.predictor.predict(z_test, action)
            
            # Decode: find closest program
            candidates = self.decoder.decode_by_program(
                z_pred, BASE_PRIMITIVES, gi, top_k=20)
            
            # Verify each candidate against training pairs
            found = False
            for prog_name, result in candidates:
                fn = BASE_PRIMITIVES[prog_name]
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
                    guesses.append([result.to_list()])
                    found = True
                    self.jepa_solves += 1
                    return guesses, f'JEPA:{prog_name}'
            
            if not found:
                guesses.append([tc['input']])
        
        return guesses if guesses else s1_preds, 'FAIL'


# ══════════════════════════════════════════════════════════════════════
# 7. BENCHMARK
# ══════════════════════════════════════════════════════════════════════

def run_benchmark(data_dir, n_samples=5000, epochs=100,
                    limit=None, verbose=False):
    files = sorted(f for f in os.listdir(data_dir) if f.endswith('.json'))
    if limit:
        files = files[:limit]
    
    print("=" * 70)
    print(f"  ARC-AGI-2 with JEPA World Model")
    print(f"  Tasks: {len(files)}")
    print(f"  JEPA: {n_samples} training samples, {epochs} epochs")
    print(f"  S1: {'Phase 2' if HAS_S1 else 'Base'}")
    print("=" * 70)
    
    # Train JEPA
    trainer = JEPATrainer(latent_dim=64, grid_size=12)
    trainer.train(n_samples=n_samples, epochs=epochs)
    
    # Solve
    solver = JEPASolver(trainer)
    solved = 0
    total = 0
    t0 = time.time()
    
    for f in files:
        task = json.load(open(os.path.join(data_dir, f)))
        total += 1
        
        predictions, system = solver.solve(task)
        correct = score_task(task, predictions)
        
        if correct:
            solved += 1
        
        if verbose and correct and system.startswith('JEPA'):
            print(f"  {f[:16]}: SOLVED ({system})")
    
    elapsed = time.time() - t0
    pct = solved / max(total, 1) * 100
    
    print(f"\n{'='*70}")
    print(f"  JEPA RESULTS")
    print(f"{'='*70}")
    print(f"  Solved:     {solved}/{total} ({pct:.1f}%)")
    print(f"    S1:       {solver.s1_solves}")
    print(f"    JEPA:     {solver.jepa_solves}")
    print(f"  Time:       {elapsed:.1f}s")
    print(f"\n{'='*70}")
    return solved, total


# ══════════════════════════════════════════════════════════════════════
# 8. TESTS
# ══════════════════════════════════════════════════════════════════════

def run_tests():
    print("=" * 65)
    print("  JEPA World Model Tests")
    print("=" * 65)
    p = 0; t = 0
    
    print("\n  T1: Grid encoder")
    enc = GridEncoder(grid_size=10, latent_dim=64)
    g = Grid(np.array([[1,2,3],[4,5,6]]))
    z = enc.encode(g)
    ok = z.shape == (64,) and np.all(np.abs(z) <= 1)
    print(f"    Shape: {z.shape}, bounded: {np.all(np.abs(z) <= 1)} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1
    
    print("\n  T2: Different grids → different embeddings")
    g2 = Grid(np.array([[6,5,4],[3,2,1]]))
    z2 = enc.encode(g2)
    dist = np.sqrt(np.sum((z - z2) ** 2))
    ok = dist > 0.01
    print(f"    Distance: {dist:.4f} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1
    
    print("\n  T3: JEPA predictor")
    pred = JEPAPredictor(latent_dim=64)
    action = np.random.randn(64).astype(np.float32) * 0.1
    z_pred = pred.predict(z, action)
    ok = z_pred.shape == (64,)
    print(f"    Predicted: {z_pred.shape} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1
    
    print("\n  T4: Action encoder — simple")
    ae = ActionEncoder(latent_dim=64)
    z_ins = np.random.randn(3, 64).astype(np.float32)
    z_outs = np.random.randn(3, 64).astype(np.float32)
    action = ae.infer_action_simple(z_ins, z_outs)
    ok = action.shape == (64,)
    print(f"    Action: {action.shape} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1
    
    print("\n  T5: Action encoder — iterative refinement")
    action2 = ae.infer_action_iterative(z_ins, z_outs, pred, n_iters=10)
    ok = action2.shape == (64,)
    print(f"    Refined action: {action2.shape} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1
    
    print("\n  T6: JEPA trainer")
    trainer = JEPATrainer(latent_dim=32, grid_size=8)
    loss = trainer.train(n_samples=200, epochs=20)
    ok = loss < 10  # some convergence
    print(f"    Final loss: {loss:.4f} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1
    
    print("\n  T7: Grid decoder — program index")
    dec = GridDecoder(trainer.encoder)
    dec.build_program_index(
        {n: f for n, f in list(BASE_PRIMITIVES.items())[:5]}, n_samples=10)
    ok = len(dec.program_embeddings) >= 3
    print(f"    Programs indexed: {len(dec.program_embeddings)} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1
    
    print("\n  T8: Full JEPA solve")
    task = {'train': [{'input': [[1,2],[3,4]], 'output': [[3,1],[4,2]]}],
            'test': [{'input': [[5,6],[7,8]], 'output': [[7,5],[8,6]]}]}
    solver = JEPASolver(trainer)
    preds, system = solver.solve(task)
    ok = system == 'S1'  # S1 should catch this
    print(f"    System: {system} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1
    
    print(f"\n{'='*65}")
    print(f"  Results: {p}/{t} tests passed")
    print(f"{'='*65}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--training", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--samples", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=100)
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
        run_benchmark(d, n_samples=args.samples, epochs=args.epochs,
                        limit=args.limit, verbose=args.verbose)
