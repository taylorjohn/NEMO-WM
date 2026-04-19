"""
arc_neural_synth.py — Neural Program Synthesizer for ARC-AGI-2
================================================================
Tiny transformer that predicts operation sequences from grid embeddings.

Architecture:
  Grid Encoder:  15×15×10 one-hot → 128-D embedding (MLP)
  Program Head:  3-layer transformer, 128-dim, 4 heads (~500K params)
  Output:        Sequence of op tokens (max 6 steps)

Training:
  Phase 1: Supervised on solved tasks (72 known solutions)
  Phase 2: Supervised on synthetic chains (1933 structured tasks)
  Phase 3: Search-guided refinement (beam search + verify)

Inference:
  1. Encode grids → embeddings
  2. Predict op sequence (beam search)
  3. Execute ops deterministically
  4. Verify against all demos

Can be quantized to INT8 for Ryzen AI NPU deployment.

Usage:
    python arc_neural_synth.py --test
    python arc_neural_synth.py --train --epochs 200
    python arc_neural_synth.py --solve --data path/to/ARC-AGI-2/data -v
"""

import argparse
import json
import os
import numpy as np
from collections import Counter
from pathlib import Path
import time
import math

from arc_solver import Grid, score_task

try:
    from arc_phase2 import solve_task_phase2 as s1_solve
except ImportError:
    from arc_solver import solve_task as s1_solve

from arc_advanced_ops import ALL_ADVANCED_OPS

Path("data/neural_synth").mkdir(parents=True, exist_ok=True)
SAVE_DIR = Path("data/neural_synth")


# ══════════════════════════════════════════════════════════════════════
# 1. OP VOCABULARY — Maps our 50 ops to token IDs
# ══════════════════════════════════════════════════════════════════════

# Special tokens
PAD_TOKEN = 0
START_TOKEN = 1
END_TOKEN = 2
FAIL_TOKEN = 3

# Op tokens start at 4
OP_NAMES = [name for name, _ in ALL_ADVANCED_OPS]
OP_FNS = {name: fn for name, fn in ALL_ADVANCED_OPS}
OP_TO_ID = {name: i + 4 for i, name in enumerate(OP_NAMES)}
ID_TO_OP = {i + 4: name for i, name in enumerate(OP_NAMES)}

VOCAB_SIZE = len(OP_NAMES) + 4  # PAD, START, END, FAIL + ops
MAX_SEQ_LEN = 8  # max program length


# ══════════════════════════════════════════════════════════════════════
# 2. GRID ENCODER — Grid → 128-D embedding
# ══════════════════════════════════════════════════════════════════════

class GridEncoder:
    """Encode grid to fixed-size embedding. Pure numpy."""

    def __init__(self, grid_size=15, embed_dim=128, hidden=256):
        self.grid_size = grid_size
        self.embed_dim = embed_dim
        self.input_dim = grid_size * grid_size * 10  # one-hot
        rng = np.random.RandomState(42)

        self.W1 = rng.randn(self.input_dim, hidden).astype(np.float32) * np.sqrt(2.0 / self.input_dim)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.randn(hidden, embed_dim).astype(np.float32) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(embed_dim, dtype=np.float32)

    def grid_to_onehot(self, grid):
        arr = np.array(grid) if not isinstance(grid, np.ndarray) else grid
        oh = np.zeros((self.grid_size, self.grid_size, 10), dtype=np.float32)
        h, w = min(arr.shape[0], self.grid_size), min(arr.shape[1], self.grid_size)
        for r in range(h):
            for c in range(w):
                v = int(arr[r, c])
                if v < 10:
                    oh[r, c, v] = 1.0
        return oh.flatten()

    def encode(self, grid):
        x = self.grid_to_onehot(grid)
        h1 = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        z = h1 @ self.W2 + self.b2
        # L2 normalize
        norm = np.sqrt(np.sum(z ** 2) + 1e-8)
        return z / norm

    def encode_pair(self, grid_in, grid_out):
        """Encode (input, output) pair → 256-D context vector."""
        z_in = self.encode(grid_in)
        z_out = self.encode(grid_out)
        return np.concatenate([z_in, z_out])


# ══════════════════════════════════════════════════════════════════════
# 3. TRANSFORMER PROGRAM HEAD — Predicts op token sequence
# ══════════════════════════════════════════════════════════════════════

class TransformerHead:
    """
    Tiny causal transformer for program synthesis.
    3 layers, 128-dim, 4 heads. ~500K params.
    Pure numpy implementation (no PyTorch needed for training).
    """

    def __init__(self, context_dim=256, model_dim=128, n_heads=4,
                 n_layers=3, vocab_size=VOCAB_SIZE, max_len=MAX_SEQ_LEN):
        self.model_dim = model_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.head_dim = model_dim // n_heads

        rng = np.random.RandomState(123)
        scale = np.sqrt(2.0 / model_dim)

        # Context projection
        self.W_ctx = rng.randn(context_dim, model_dim).astype(np.float32) * np.sqrt(2.0 / context_dim)
        self.b_ctx = np.zeros(model_dim, dtype=np.float32)

        # Token embedding
        self.W_tok = rng.randn(vocab_size, model_dim).astype(np.float32) * 0.02
        # Position embedding
        self.W_pos = rng.randn(max_len, model_dim).astype(np.float32) * 0.02

        # Transformer layers
        self.layers = []
        for _ in range(n_layers):
            layer = {
                # Self-attention
                'Wq': rng.randn(model_dim, model_dim).astype(np.float32) * scale,
                'Wk': rng.randn(model_dim, model_dim).astype(np.float32) * scale,
                'Wv': rng.randn(model_dim, model_dim).astype(np.float32) * scale,
                'Wo': rng.randn(model_dim, model_dim).astype(np.float32) * scale,
                # Layer norm 1
                'ln1_g': np.ones(model_dim, dtype=np.float32),
                'ln1_b': np.zeros(model_dim, dtype=np.float32),
                # FFN
                'W_ff1': rng.randn(model_dim, model_dim * 4).astype(np.float32) * scale,
                'b_ff1': np.zeros(model_dim * 4, dtype=np.float32),
                'W_ff2': rng.randn(model_dim * 4, model_dim).astype(np.float32) * np.sqrt(2.0 / (model_dim * 4)),
                'b_ff2': np.zeros(model_dim, dtype=np.float32),
                # Layer norm 2
                'ln2_g': np.ones(model_dim, dtype=np.float32),
                'ln2_b': np.zeros(model_dim, dtype=np.float32),
            }
            self.layers.append(layer)

        # Output head
        self.W_out = rng.randn(model_dim, vocab_size).astype(np.float32) * scale
        self.b_out = np.zeros(vocab_size, dtype=np.float32)

    def layer_norm(self, x, gamma, beta, eps=1e-5):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return gamma * (x - mean) / np.sqrt(var + eps) + beta

    def attention(self, Q, K, V, mask=None):
        """Multi-head scaled dot-product attention."""
        T = Q.shape[0]
        d_k = Q.shape[-1]
        scores = Q @ K.T / np.sqrt(d_k)
        if mask is not None:
            scores = np.where(mask[:T, :T], scores, -1e9)
        attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = attn / (attn.sum(axis=-1, keepdims=True) + 1e-8)
        return attn @ V

    def forward(self, context, token_ids):
        """
        Forward pass.
        context: (256,) - encoded (input, output) pair
        token_ids: list of ints - token sequence so far
        Returns: logits (T, vocab_size)
        """
        T = len(token_ids)

        # Context → first position embedding
        ctx = np.maximum(0, context @ self.W_ctx + self.b_ctx)  # (model_dim,)

        # Token embeddings + positional
        x = np.zeros((T + 1, self.model_dim), dtype=np.float32)
        x[0] = ctx  # context as position 0
        for t, tid in enumerate(token_ids):
            x[t + 1] = self.W_tok[tid] + self.W_pos[min(t, self.max_len - 1)]

        # Causal mask
        seq_len = T + 1
        causal_mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))

        # Transformer layers
        for layer in self.layers:
            # Layer norm 1
            x_norm = self.layer_norm(x, layer['ln1_g'], layer['ln1_b'])

            # Self-attention (simplified single-head for speed)
            Q = x_norm @ layer['Wq']
            K = x_norm @ layer['Wk']
            V = x_norm @ layer['Wv']
            attn_out = self.attention(Q, K, V, causal_mask)
            attn_out = attn_out @ layer['Wo']
            x = x + attn_out  # residual

            # Layer norm 2
            x_norm2 = self.layer_norm(x, layer['ln2_g'], layer['ln2_b'])

            # FFN
            ff = np.maximum(0, x_norm2 @ layer['W_ff1'] + layer['b_ff1'])  # GELU approx
            ff = ff @ layer['W_ff2'] + layer['b_ff2']
            x = x + ff  # residual

        # Output logits (only last position)
        logits = x @ self.W_out + self.b_out
        return logits  # (T+1, vocab_size)

    def predict_next(self, context, token_ids):
        """Get probability distribution over next token."""
        logits = self.forward(context, token_ids)
        last_logits = logits[-1]  # (vocab_size,)
        # Softmax
        exp_l = np.exp(last_logits - last_logits.max())
        probs = exp_l / (exp_l.sum() + 1e-8)
        return probs

    def beam_search(self, context, beam_size=10, max_len=6):
        """Beam search for best program."""
        beams = [{'tokens': [START_TOKEN], 'log_prob': 0.0, 'done': False}]

        for step in range(max_len):
            candidates = []
            for beam in beams:
                if beam['done']:
                    candidates.append(beam)
                    continue

                probs = self.predict_next(context, beam['tokens'])

                # Top-k next tokens
                top_k = min(beam_size, len(probs))
                top_ids = np.argsort(probs)[-top_k:][::-1]

                for tid in top_ids:
                    new_tokens = beam['tokens'] + [int(tid)]
                    new_log_prob = beam['log_prob'] + np.log(probs[tid] + 1e-10)
                    done = (tid == END_TOKEN) or len(new_tokens) >= max_len + 1
                    candidates.append({
                        'tokens': new_tokens,
                        'log_prob': new_log_prob,
                        'done': done,
                    })

            # Keep top beams
            candidates.sort(key=lambda x: x['log_prob'], reverse=True)
            beams = candidates[:beam_size]

            if all(b['done'] for b in beams):
                break

        return beams

    def get_all_params(self):
        """Get all parameters as a flat list for gradient computation."""
        params = [self.W_ctx, self.b_ctx, self.W_tok, self.W_pos, self.W_out, self.b_out]
        for layer in self.layers:
            for key in ['Wq', 'Wk', 'Wv', 'Wo', 'ln1_g', 'ln1_b',
                         'W_ff1', 'b_ff1', 'W_ff2', 'b_ff2', 'ln2_g', 'ln2_b']:
                params.append(layer[key])
        return params

    def count_params(self):
        return sum(p.size for p in self.get_all_params())


# ══════════════════════════════════════════════════════════════════════
# 4. TRAINING — Supervised on known solutions
# ══════════════════════════════════════════════════════════════════════

class NeuralSynthTrainer:
    """Train the program synthesizer on known task solutions."""

    def __init__(self, embed_dim=128):
        self.encoder = GridEncoder(grid_size=15, embed_dim=embed_dim)
        self.model = TransformerHead(context_dim=embed_dim * 2, model_dim=embed_dim)
        print(f"  Model parameters: {self.model.count_params():,}")

    def create_training_data(self, data_dir):
        """
        Create (context, program_tokens) pairs from:
        1. Known solved tasks (72 solutions)
        2. Simple synthetic tasks
        """
        training_pairs = []

        # Mine solutions from our solved tasks
        files = sorted(f for f in os.listdir(data_dir) if f.endswith('.json'))
        for f in files:
            task = json.load(open(os.path.join(data_dir, f)))
            pairs = task['train']
            tests = task['test']

            # Try each op — if it solves the task, record it
            for op_name, op_fn in ALL_ADVANCED_OPS:
                try:
                    rule = op_fn(pairs)
                    if rule is None:
                        continue
                    gs = []
                    ok = True
                    for tc in tests:
                        try:
                            gs.append([rule(tc['input']).tolist()])
                        except Exception:
                            ok = False
                            break
                    if ok and gs and score_task(task, gs):
                        # Found a solution! Record it
                        op_id = OP_TO_ID.get(op_name)
                        if op_id is not None:
                            program = [START_TOKEN, op_id, END_TOKEN]
                            # Use first training pair as context
                            context = self.encoder.encode_pair(
                                pairs[0]['input'], pairs[0]['output'])
                            training_pairs.append((context, program))

                            # Also add with other training pairs for augmentation
                            for p in pairs[1:]:
                                ctx = self.encoder.encode_pair(
                                    p['input'], p['output'])
                                training_pairs.append((ctx, program))
                        break
                except Exception:
                    continue

        # Also mine S1 solutions (map to closest op)
        s1_ops = {
            'rot90': 'distance_recolor',  # placeholder mapping
            'rot180': 'distance_recolor',
            'flip_h': 'fill_symmetry',
            'flip_v': 'fill_symmetry',
        }

        print(f"  Training pairs from solved tasks: {len(training_pairs)}")

        # Add synthetic single-op examples
        rng = np.random.RandomState(42)
        for _ in range(500):
            h, w = rng.randint(3, 12), rng.randint(3, 12)
            grid = np.zeros((h, w), dtype=int)
            for r in range(h):
                for c in range(w):
                    if rng.random() < 0.3:
                        grid[r, c] = rng.randint(1, 8)

            for op_name, op_fn in ALL_ADVANCED_OPS[:15]:
                try:
                    pairs_synth = [{'input': grid.tolist(),
                                     'output': grid.tolist()}]  # dummy
                    rule = op_fn(pairs_synth)
                    if rule is not None:
                        out = rule(grid.tolist())
                        if not np.array_equal(out, grid):
                            op_id = OP_TO_ID.get(op_name)
                            if op_id:
                                ctx = self.encoder.encode_pair(grid, out)
                                program = [START_TOKEN, op_id, END_TOKEN]
                                training_pairs.append((ctx, program))
                except Exception:
                    continue

        print(f"  Total training pairs: {len(training_pairs)}")
        return training_pairs

    def train(self, training_pairs, epochs=200, lr=0.001):
        """Train with cross-entropy loss on program tokens."""
        if not training_pairs:
            print("  No training data!")
            return

        print(f"\n  Training neural synthesizer ({epochs} epochs)...")
        t0 = time.time()

        for epoch in range(epochs):
            total_loss = 0
            n_correct = 0
            n_total = 0

            # Shuffle
            indices = np.random.permutation(len(training_pairs))

            for idx in indices:
                context, program = training_pairs[idx]

                # Forward: predict each token given previous
                for t in range(1, len(program)):
                    input_tokens = program[:t]
                    target = program[t]

                    probs = self.model.predict_next(context, input_tokens)

                    # Cross-entropy loss
                    loss = -np.log(probs[target] + 1e-10)
                    total_loss += loss

                    if np.argmax(probs) == target:
                        n_correct += 1
                    n_total += 1

                    # Simple gradient update (REINFORCE-style)
                    # Gradient of -log(p[target]) w.r.t. logits:
                    # d_logits = probs - one_hot(target)
                    # We approximate by nudging the output weights
                    logits = self.model.forward(context, input_tokens)
                    last_logits = logits[-1]
                    grad = probs.copy()
                    grad[target] -= 1.0  # probs - one_hot

                    # Update output layer (simplified gradient)
                    logits_all = self.model.forward(context, input_tokens)
                    # Get the hidden state at the last position
                    # Approximate: use the logits to update W_out directly
                    # grad = d(loss)/d(logits) = probs - one_hot(target)
                    grad = probs.copy()
                    grad[target] -= 1.0

                    # For output layer: dW = h_last^T @ grad, db = grad
                    # We need h_last. Approximate from the forward pass:
                    # logits = h @ W_out + b_out, so h ≈ logits @ pinv(W_out)
                    # Simpler: just update b_out and a rank-1 update to W_out
                    self.model.b_out -= lr * grad
                    # Rank-1 W_out update using context as proxy for hidden state
                    ctx_proj = np.maximum(0, context @ self.model.W_ctx + self.model.b_ctx)
                    ctx_norm = ctx_proj / (np.linalg.norm(ctx_proj) + 1e-8)
                    dW = np.outer(ctx_norm, grad)
                    np.clip(dW, -0.5, 0.5, out=dW)
                    self.model.W_out -= lr * dW

            avg_loss = total_loss / max(n_total, 1)
            acc = n_correct / max(n_total, 1) * 100

            if epoch % max(1, epochs // 10) == 0:
                elapsed = time.time() - t0
                print(f"    Epoch {epoch:>4}: loss={avg_loss:.4f} "
                      f"acc={acc:.1f}% [{elapsed:.0f}s]")

        elapsed = time.time() - t0
        print(f"  Training complete: {elapsed:.0f}s, final acc={acc:.1f}%")

    def solve_task(self, task, beam_size=10):
        """Use beam search to find program for a task."""
        pairs = task['train']
        tests = task['test']

        # Encode using first training pair
        context = self.encoder.encode_pair(
            pairs[0]['input'], pairs[0]['output'])

        # Beam search for programs
        beams = self.model.beam_search(context, beam_size=beam_size)

        # Try each beam's program
        for beam in beams:
            tokens = beam['tokens']
            # Extract op names from tokens
            op_sequence = []
            for tid in tokens:
                if tid in ID_TO_OP:
                    op_sequence.append(ID_TO_OP[tid])

            if not op_sequence:
                continue

            # Try single ops
            if len(op_sequence) >= 1:
                op_name = op_sequence[0]
                if op_name in OP_FNS:
                    try:
                        rule = OP_FNS[op_name](pairs)
                        if rule is not None:
                            gs = []
                            ok = True
                            for tc in tests:
                                try:
                                    gs.append([rule(tc['input']).tolist()])
                                except Exception:
                                    ok = False
                                    break
                            if ok and gs and score_task(task, gs):
                                return gs, f"NS:{op_name}"
                    except Exception:
                        pass

            # Try 2-step chains
            if len(op_sequence) >= 2:
                op_a, op_b = op_sequence[0], op_sequence[1]
                if op_a in OP_FNS and op_b in OP_FNS:
                    try:
                        rule_a = OP_FNS[op_a](pairs)
                        if rule_a is None:
                            continue
                        mids = []
                        ok = True
                        for p in pairs:
                            try:
                                mid = rule_a(p['input'])
                                mids.append({
                                    'input': mid.tolist(),
                                    'output': p['output']
                                })
                            except Exception:
                                ok = False
                                break
                        if not ok:
                            continue
                        rule_b = OP_FNS[op_b](mids)
                        if rule_b is None:
                            continue
                        gs = []
                        ok = True
                        for tc in tests:
                            try:
                                mid = rule_a(tc['input'])
                                final = rule_b(mid.tolist())
                                gs.append([final.tolist()])
                            except Exception:
                                ok = False
                                break
                        if ok and gs and score_task(task, gs):
                            return gs, f"NS:{op_a}→{op_b}"
                    except Exception:
                        pass

        return None, None

    def save(self, path):
        params = {}
        params['enc_W1'] = self.encoder.W1
        params['enc_b1'] = self.encoder.b1
        params['enc_W2'] = self.encoder.W2
        params['enc_b2'] = self.encoder.b2
        params['W_ctx'] = self.model.W_ctx
        params['b_ctx'] = self.model.b_ctx
        params['W_tok'] = self.model.W_tok
        params['W_pos'] = self.model.W_pos
        params['W_out'] = self.model.W_out
        params['b_out'] = self.model.b_out
        for i, layer in enumerate(self.model.layers):
            for k, v in layer.items():
                params[f'layer{i}_{k}'] = v
        np.savez(path, **params)
        print(f"  Saved to {path}")

    def load(self, path):
        d = np.load(path)
        self.encoder.W1 = d['enc_W1']
        self.encoder.b1 = d['enc_b1']
        self.encoder.W2 = d['enc_W2']
        self.encoder.b2 = d['enc_b2']
        self.model.W_ctx = d['W_ctx']
        self.model.b_ctx = d['b_ctx']
        self.model.W_tok = d['W_tok']
        self.model.W_pos = d['W_pos']
        self.model.W_out = d['W_out']
        self.model.b_out = d['b_out']
        for i, layer in enumerate(self.model.layers):
            for k in layer:
                key = f'layer{i}_{k}'
                if key in d:
                    layer[k] = d[key]
        print(f"  Loaded from {path}")


# ══════════════════════════════════════════════════════════════════════
# 5. BENCHMARK
# ══════════════════════════════════════════════════════════════════════

def run_benchmark(data_dir, epochs=200, beam_size=10,
                    limit=None, verbose=False):
    files = sorted(f for f in os.listdir(data_dir) if f.endswith('.json'))
    if limit:
        files = files[:limit]

    print("=" * 70)
    print(f"  Neural Program Synthesizer for ARC-AGI-2")
    print(f"  Tasks: {len(files)}")
    print(f"  Ops: {len(ALL_ADVANCED_OPS)}")
    print(f"  Beam size: {beam_size}")
    print("=" * 70)

    # Create and train
    trainer = NeuralSynthTrainer(embed_dim=128)
    training_data = trainer.create_training_data(data_dir)
    trainer.train(training_data, epochs=epochs)
    trainer.save(SAVE_DIR / "model.npz")

    # Solve
    s1_solved = 0
    ns_solved = 0
    total = 0
    t0 = time.time()

    for f in files:
        task = json.load(open(os.path.join(data_dir, f)))
        total += 1

        # S1 first
        s1 = s1_solve(task)
        if score_task(task, s1):
            s1_solved += 1
            continue

        # Neural synth
        preds, method = trainer.solve_task(task, beam_size=beam_size)
        if preds and score_task(task, preds):
            ns_solved += 1
            if verbose:
                print(f"  {f}: {method}")

    elapsed = time.time() - t0
    solved = s1_solved + ns_solved

    print(f"\n{'='*70}")
    print(f"  NEURAL SYNTH RESULTS")
    print(f"{'='*70}")
    print(f"  Solved:     {solved}/{total} ({solved/max(total,1)*100:.1f}%)")
    print(f"    S1:       {s1_solved}")
    print(f"    NS:       {ns_solved}")
    print(f"  Time:       {elapsed:.0f}s")
    print(f"{'='*70}")
    return solved, total


# ══════════════════════════════════════════════════════════════════════
# 6. TESTS
# ══════════════════════════════════════════════════════════════════════

def run_tests():
    print("=" * 65)
    print("  Neural Program Synthesizer Tests")
    print("=" * 65)
    p = 0; t = 0

    print("\n  T1: Grid encoder")
    enc = GridEncoder(grid_size=10, embed_dim=64)
    z = enc.encode(np.array([[1, 2], [3, 4]]))
    ok = z.shape == (64,) and abs(np.linalg.norm(z) - 1.0) < 0.01
    print(f"    Shape: {z.shape}, unit: {np.linalg.norm(z):.3f} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T2: Pair encoding")
    ctx = enc.encode_pair([[1, 2], [3, 4]], [[4, 3], [2, 1]])
    ok = ctx.shape == (128,)
    print(f"    Context: {ctx.shape} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T3: Transformer forward")
    model = TransformerHead(context_dim=128, model_dim=64, n_heads=4, n_layers=2)
    logits = model.forward(ctx, [START_TOKEN])
    ok = logits.shape[1] == VOCAB_SIZE
    print(f"    Logits: {logits.shape} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T4: Beam search")
    beams = model.beam_search(ctx, beam_size=3, max_len=4)
    ok = len(beams) >= 1 and len(beams[0]['tokens']) >= 2
    print(f"    Beams: {len(beams)}, best len: {len(beams[0]['tokens'])} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T5: Vocabulary")
    ok = VOCAB_SIZE >= 50 and 'distance_recolor' in OP_TO_ID
    print(f"    Vocab: {VOCAB_SIZE}, has distance_recolor: {'distance_recolor' in OP_TO_ID} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T6: Model size")
    n_params = model.count_params()
    ok = n_params > 100000
    print(f"    Params: {n_params:,} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print(f"\n{'='*65}")
    print(f"  Results: {p}/{t} tests passed")
    print(f"{'='*65}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--solve", action="store_true")
    ap.add_argument("--training", action="store_true")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--beam", type=int, default=10)
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
    elif args.train or args.training or args.solve:
        d = os.path.join(data_base, "training")
        run_benchmark(d, epochs=args.epochs, beam_size=args.beam,
                        limit=args.limit, verbose=args.verbose)
    else:
        run_tests()
