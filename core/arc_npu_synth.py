"""
arc_npu_synth.py — PyTorch Program Synthesizer for Ryzen AI NPU
================================================================
~2M param transformer that predicts ARC operation sequences.
Trains on CPU/DirectML, deploys on NPU via ONNX INT8.

Pipeline:
  1. Train: CPU/iGPU (PyTorch) on solved tasks + synthetic data
  2. Export: ONNX with static shapes
  3. Quantize: INT8 for NPU
  4. Infer: ONNX Runtime + Vitis AI EP (2-4ms per task)
  5. Execute: CPU runs predicted ops, verifies against demos

Architecture:
  Grid Encoder: 15x15x10 → 128-D (MLP, shared with JEPA)
  Program Head: 3-layer transformer, 256-dim, 4 heads (~2M params)
  Output: Sequence of op tokens (max 6 steps)

Usage:
    python arc_npu_synth.py --test
    python arc_npu_synth.py --train --data path/to/ARC-AGI-2/data --epochs 50
    python arc_npu_synth.py --export
    python arc_npu_synth.py --solve --data path/to/ARC-AGI-2/data -v
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path
from collections import Counter
import time

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not found. Install: pip install torch")

# Check for ONNX Runtime
try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

from arc_solver import Grid, score_task

try:
    from arc_phase2 import solve_task_phase2 as s1_solve
except ImportError:
    from arc_solver import solve_task as s1_solve

from arc_advanced_ops import ALL_ADVANCED_OPS

Path("data/npu_synth").mkdir(parents=True, exist_ok=True)
SAVE_DIR = Path("data/npu_synth")

# ══════════════════════════════════════════════════════════════════════
# 1. OP VOCABULARY
# ══════════════════════════════════════════════════════════════════════

PAD, START, END = 0, 1, 2
OP_NAMES = [name for name, _ in ALL_ADVANCED_OPS]
OP_FNS = {name: fn for name, fn in ALL_ADVANCED_OPS}
OP_TO_ID = {name: i + 3 for i, name in enumerate(OP_NAMES)}
ID_TO_OP = {i + 3: name for i, name in enumerate(OP_NAMES)}
VOCAB_SIZE = len(OP_NAMES) + 3
MAX_SEQ = 8
GRID_SIZE = 15
EMBED_DIM = 128
MODEL_DIM = 256


# ══════════════════════════════════════════════════════════════════════
# 2. GRID ENCODER (numpy, shared with JEPA)
# ══════════════════════════════════════════════════════════════════════

def grid_to_tensor(grid, size=GRID_SIZE):
    """Grid → one-hot tensor [size*size*10]."""
    arr = np.array(grid) if not isinstance(grid, np.ndarray) else grid
    oh = np.zeros((size, size, 10), dtype=np.float32)
    h, w = min(arr.shape[0], size), min(arr.shape[1], size)
    for r in range(h):
        for c in range(w):
            v = int(arr[r, c])
            if v < 10:
                oh[r, c, v] = 1.0
    return oh.flatten()


def encode_pair_numpy(grid_in, grid_out, size=GRID_SIZE):
    """Encode (input, output) → 2*size*size*10 flat vector."""
    return np.concatenate([grid_to_tensor(grid_in, size),
                            grid_to_tensor(grid_out, size)])


# ══════════════════════════════════════════════════════════════════════
# 3. PYTORCH MODEL (~2M params)
# ══════════════════════════════════════════════════════════════════════

if HAS_TORCH:
    class ArcSynthesizer(nn.Module):
        """Tiny causal transformer for ARC program synthesis."""

        def __init__(self, input_dim=GRID_SIZE*GRID_SIZE*10*2,
                     d_model=MODEL_DIM, nhead=4, nlayers=3,
                     vocab_size=VOCAB_SIZE, max_len=MAX_SEQ):
            super().__init__()
            self.d_model = d_model
            self.max_len = max_len
            self.vocab_size = vocab_size

            # Grid pair → context embedding
            self.ctx_encoder = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Linear(512, d_model),
                nn.LayerNorm(d_model),
            )

            # Token + position embeddings
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(max_len + 1, d_model)

            # Transformer
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=d_model * 2,
                batch_first=True, dropout=0.1,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=nlayers)
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size)

        def forward(self, context, tokens):
            """
            context: [B, input_dim] grid pair features
            tokens: [B, T] program tokens (teacher-forced)
            Returns: [B, T, vocab_size] logits
            """
            B, T = tokens.shape
            ctx = self.ctx_encoder(context).unsqueeze(1)  # [B, 1, D]
            tok = self.tok_emb(tokens) + self.pos_emb(torch.arange(T, device=tokens.device))
            x = torch.cat([ctx, tok], dim=1)  # [B, T+1, D]

            # Static causal mask (fixed size, NPU-safe)
            seq_len = x.size(1)
            mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
            out = self.transformer(x, mask=mask, is_causal=True)
            logits = self.head(self.norm(out[:, 1:, :]))  # skip context token
            return logits

        @torch.no_grad()
        def generate(self, context, max_len=6, temperature=1.0):
            """Autoregressive generation."""
            self.eval()
            B = context.size(0)
            tokens = torch.full((B, 1), START, dtype=torch.long, device=context.device)

            for _ in range(max_len):
                logits = self.forward(context, tokens)
                next_logits = logits[:, -1, :] / temperature
                probs = F.softmax(next_logits, dim=-1)
                next_tok = torch.multinomial(probs, 1)
                tokens = torch.cat([tokens, next_tok], dim=1)
                if (next_tok == END).all():
                    break

            return tokens[:, 1:]  # drop START

        @torch.no_grad()
        def beam_search(self, context, beam_size=10, max_len=6):
            """Beam search for best program."""
            self.eval()
            B = context.size(0)
            assert B == 1, "Beam search supports batch_size=1"

            beams = [{'tokens': torch.tensor([[START]], device=context.device),
                       'score': 0.0, 'done': False}]

            for step in range(max_len):
                candidates = []
                for beam in beams:
                    if beam['done']:
                        candidates.append(beam)
                        continue

                    logits = self.forward(context, beam['tokens'])
                    probs = F.log_softmax(logits[:, -1, :], dim=-1)
                    topk = torch.topk(probs, beam_size, dim=-1)

                    for i in range(beam_size):
                        tok_id = topk.indices[0, i].item()
                        score = beam['score'] + topk.values[0, i].item()
                        new_tokens = torch.cat([beam['tokens'],
                                                 torch.tensor([[tok_id]], device=context.device)], dim=1)
                        done = (tok_id == END) or new_tokens.size(1) >= max_len + 1
                        candidates.append({'tokens': new_tokens, 'score': score, 'done': done})

                candidates.sort(key=lambda x: x['score'], reverse=True)
                beams = candidates[:beam_size]
                if all(b['done'] for b in beams):
                    break

            return [b['tokens'][0, 1:].tolist() for b in beams]  # drop START


# ══════════════════════════════════════════════════════════════════════
# 4. TRAINING DATA GENERATOR
# ══════════════════════════════════════════════════════════════════════

def create_training_data(data_dir):
    """Create (context_vector, program_tokens) pairs with augmentation."""
    pairs = []
    files = sorted(f for f in os.listdir(data_dir) if f.endswith('.json'))

    # Try loading cached training data first
    cache_path = Path("data/npu_synth/training_data.json")
    if cache_path.exists():
        cached = json.load(open(cache_path))
        for t in cached:
            op_id = OP_TO_ID.get(t['op'])
            if op_id is None:
                continue
            program = [START, op_id, END]
            ctx = encode_pair_numpy(t['input'], t['output'])
            pairs.append((ctx, program))
        print(f"  Loaded {len(pairs)} cached training pairs ({len(set(t['op'] for t in cached))} ops)")
        return pairs

    # Mine from ARC data with augmentation
    for f in files:
        task = json.load(open(os.path.join(data_dir, f)))
        train_pairs = task['train']
        tests = task['test']

        for op_name, op_fn in ALL_ADVANCED_OPS:
            try:
                rule = op_fn(train_pairs)
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
                    op_id = OP_TO_ID.get(op_name)
                    if op_id is None:
                        continue
                    program = [START, op_id, END]
                    for p in train_pairs:
                        ctx = encode_pair_numpy(p['input'], p['output'])
                        pairs.append((ctx, program))
                        # Augmentation: flip vertically
                        gi = np.array(p['input'])
                        go = np.array(p['output'])
                        if gi.shape == go.shape:
                            ctx_fv = encode_pair_numpy(gi[::-1].tolist(), go[::-1].tolist())
                            pairs.append((ctx_fv, program))
                            ctx_fh = encode_pair_numpy(gi[:,::-1].tolist(), go[:,::-1].tolist())
                            pairs.append((ctx_fh, program))
                    break
            except Exception:
                continue

    print(f"  Training pairs: {len(pairs)}")
    return pairs


def generate_synthetic_pairs(n_samples=5000):
    """Generate synthetic (context, program) pairs.
    Directly apply grid transforms (no rule learning needed)."""
    rng = np.random.RandomState(42)
    pairs = []
    
    for _ in range(n_samples * 3):
        if len(pairs) >= n_samples:
            break
            
        h, w = rng.randint(4, 12), rng.randint(4, 12)
        grid = np.zeros((h, w), dtype=int)
        n_obj = rng.randint(2, 5)
        for _ in range(n_obj):
            oh, ow = rng.randint(1, max(2, h//3)), rng.randint(1, max(2, w//3))
            r, c = rng.randint(0, h-oh), rng.randint(0, w-ow)
            grid[r:r+oh, c:c+ow] = rng.randint(1, 9)
        if grid.sum() == 0:
            continue

        # Pick a transform and apply it directly
        transform_id = rng.randint(0, 12)
        try:
            if transform_id == 0:
                # Gravity down
                out = np.zeros_like(grid)
                bg = int(np.argmax(np.bincount(grid.flatten())))
                for c in range(w):
                    vals = [int(grid[r, c]) for r in range(h) if grid[r, c] != bg]
                    for i, v in enumerate(reversed(vals)):
                        out[h-1-i, c] = v
                op_name = 'gravity_variants'
            elif transform_id == 1:
                # Fill symmetry (horizontal)
                out = grid.copy()
                for r in range(h):
                    for c in range(w):
                        mc = w - 1 - c
                        if out[r, c] == 0 and out[r, mc] != 0:
                            out[r, c] = out[r, mc]
                op_name = 'fill_symmetry'
            elif transform_id == 2:
                # Distance recolor
                bg = int(np.argmax(np.bincount(grid.flatten())))
                nz = [(r, c) for r in range(h) for c in range(w) if grid[r, c] != bg]
                if not nz: continue
                out = grid.copy()
                colors = rng.choice(range(1, 9), 3, replace=False)
                for r in range(h):
                    for c in range(w):
                        if grid[r, c] == bg:
                            d = min(max(abs(r-nr), abs(c-nc)) for nr, nc in nz)
                            out[r, c] = int(colors[min(d, 2)])
                op_name = 'distance_recolor'
            elif transform_id == 3:
                # Recolor by size (rank objects)
                bg = int(np.argmax(np.bincount(grid.flatten())))
                visited = np.zeros((h, w), dtype=bool)
                objs = []
                for r in range(h):
                    for c in range(w):
                        if grid[r, c] != bg and not visited[r, c]:
                            cells = []
                            stack = [(r, c)]
                            color = int(grid[r, c])
                            while stack:
                                cr, cc = stack.pop()
                                if cr < 0 or cr >= h or cc < 0 or cc >= w: continue
                                if visited[cr, cc] or int(grid[cr, cc]) != color: continue
                                visited[cr, cc] = True
                                cells.append((cr, cc))
                                stack.extend([(cr-1,cc),(cr+1,cc),(cr,cc-1),(cr,cc+1)])
                            objs.append((cells, color))
                if len(objs) < 2: continue
                objs.sort(key=lambda o: len(o[0]), reverse=True)
                rank_colors = rng.choice(range(1, 9), len(objs), replace=False)
                out = np.full_like(grid, bg)
                for i, (cells, _) in enumerate(objs):
                    for r, c in cells:
                        out[r, c] = int(rank_colors[i])
                op_name = 'recolor_by_size'
            elif transform_id == 4:
                # Fill enclosed regions
                bg = int(np.argmax(np.bincount(grid.flatten())))
                ext = np.zeros((h, w), dtype=bool)
                stk = []
                for r in range(h):
                    for cc in [0, w-1]:
                        if grid[r, cc] == bg: stk.append((r, cc))
                for cc in range(w):
                    for r in [0, h-1]:
                        if grid[r, cc] == bg: stk.append((r, cc))
                while stk:
                    r, c = stk.pop()
                    if r < 0 or r >= h or c < 0 or c >= w or ext[r, c] or grid[r, c] != bg: continue
                    ext[r, c] = True
                    stk.extend([(r-1,c),(r+1,c),(r,c-1),(r,c+1)])
                out = grid.copy()
                fc = rng.randint(1, 9)
                filled = False
                for r in range(h):
                    for c in range(w):
                        if grid[r, c] == bg and not ext[r, c]:
                            out[r, c] = fc
                            filled = True
                if not filled: continue
                op_name = 'fill_enclosed'
            elif transform_id == 5:
                # Row/col fill
                bg = int(np.argmax(np.bincount(grid.flatten())))
                out = grid.copy()
                for r in range(h):
                    nz = [int(grid[r, c]) for c in range(w) if grid[r, c] != bg]
                    if len(nz) == 1:
                        for c in range(w): out[r, c] = nz[0]
                op_name = 'fill_by_projection'
            elif transform_id == 6:
                # Extract unique color
                bg = int(np.argmax(np.bincount(grid.flatten())))
                colors = [int(v) for v in grid.flatten() if v != bg]
                if not colors: continue
                cc = Counter(colors)
                unique = [c for c, n in cc.items() if n == min(cc.values())]
                if not unique: continue
                out = np.full_like(grid, bg)
                out[grid == unique[0]] = unique[0]
                if np.array_equal(out, grid): continue
                op_name = 'extract_unique'
            elif transform_id == 7:
                # Boolean AND of two color masks
                bg = int(np.argmax(np.bincount(grid.flatten())))
                colors = sorted(set(int(v) for v in grid.flatten()) - {bg})
                if len(colors) < 2: continue
                m1 = (grid == colors[0])
                m2 = (grid == colors[1])
                out = np.full_like(grid, bg)
                out[m1 & m2] = colors[0]
                if out.sum() == 0: continue
                op_name = 'boolean_ops'
            elif transform_id == 8:
                # Scale by 2
                out = np.repeat(np.repeat(grid, 2, axis=0), 2, axis=1)
                op_name = 'scale_by_count'
            elif transform_id == 9:
                # Boundary trace (inner)
                bg = int(np.argmax(np.bincount(grid.flatten())))
                out = np.full_like(grid, bg)
                for r in range(h):
                    for c in range(w):
                        if grid[r, c] != bg:
                            has_bg = any(0<=r+dr<h and 0<=c+dc<w and grid[r+dr, c+dc]==bg
                                          for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)])
                            if has_bg: out[r, c] = grid[r, c]
                if np.array_equal(out, grid): continue
                op_name = 'boundary_trace'
            elif transform_id == 10:
                # Majority neighbor
                out = np.zeros_like(grid)
                for r in range(h):
                    for c in range(w):
                        vals = []
                        for dr in [-1,0,1]:
                            for dc in [-1,0,1]:
                                nr, nc = r+dr, c+dc
                                if 0<=nr<h and 0<=nc<w: vals.append(int(grid[nr, nc]))
                        out[r, c] = Counter(vals).most_common(1)[0][0]
                op_name = 'majority_neighbor'
            elif transform_id == 11:
                # Voronoi fill
                bg = int(np.argmax(np.bincount(grid.flatten())))
                seeds = [(r, c, int(grid[r, c])) for r in range(h) for c in range(w) if grid[r, c] != bg]
                if not seeds: continue
                out = np.zeros_like(grid)
                for r in range(h):
                    for c in range(w):
                        bd, bc = float('inf'), 0
                        for sr, sc, scolor in seeds:
                            d = abs(r-sr) + abs(c-sc)
                            if d < bd: bd, bc = d, scolor
                        out[r, c] = bc
                op_name = 'fill_voronoi'
            else:
                continue
            
            if op_name not in OP_TO_ID:
                continue
            if np.array_equal(out, grid):
                continue
                
            ctx = encode_pair_numpy(grid, out)
            program = [START, OP_TO_ID[op_name], END]
            pairs.append((ctx, program))
            
        except Exception:
            continue
    
    print(f"  Synthetic pairs generated: {len(pairs)}")
    # Show distribution
    op_dist = Counter()
    for _, prog in pairs:
        if len(prog) >= 2:
            op_id = prog[1]
            if op_id in ID_TO_OP:
                op_dist[ID_TO_OP[op_id]] += 1
    for op, n in op_dist.most_common():
        print(f"    {op}: {n}")
    return pairs


# ══════════════════════════════════════════════════════════════════════
# 5. TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def train_model(data_dir, epochs=50, lr=3e-4, batch_size=32):
    """Train the synthesizer."""
    if not HAS_TORCH:
        print("PyTorch required for training!")
        return None

    # Detect device
    device = "cpu"
    try:
        import torch_directml
        device = torch_directml.device()
        print(f"  Using DirectML (iGPU)")
    except ImportError:
        print(f"  Using CPU")

    # Create data
    print("  Creating training data...")
    data = create_training_data(data_dir)
    
    # Add synthetic data
    print("  Generating synthetic pretraining data...")
    synthetic = generate_synthetic_pairs(n_samples=5000)
    data = data + synthetic
    np.random.shuffle(data)
    print(f"  Total training data: {len(data)} (real + synthetic)")
    
    if not data:
        print("  No training data found!")
        return None

    # Prepare tensors
    max_prog_len = max(len(p) for _, p in data)
    X = np.array([d[0] for d in data], dtype=np.float32)
    Y = np.zeros((len(data), max_prog_len), dtype=np.int64)
    for i, (_, prog) in enumerate(data):
        Y[i, :len(prog)] = prog

    X_t = torch.tensor(X, device=device)
    Y_t = torch.tensor(Y, device=device)

    # Model
    input_dim = X.shape[1]
    model = ArcSynthesizer(input_dim=input_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n  Training ({epochs} epochs, batch={batch_size})...")
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_t))
        total_loss = 0
        n_correct = 0
        n_total = 0

        for start in range(0, len(X_t), batch_size):
            batch = perm[start:start + batch_size]
            ctx = X_t[batch]
            tokens = Y_t[batch]

            # Teacher-forced: input = tokens[:, :-1], target = tokens[:, 1:]
            inp_tokens = tokens[:, :-1]
            tgt_tokens = tokens[:, 1:]

            logits = model(ctx, inp_tokens)
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                tgt_tokens.reshape(-1),
                ignore_index=PAD,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            mask = tgt_tokens != PAD
            n_correct += (preds == tgt_tokens)[mask].sum().item()
            n_total += mask.sum().item()

        scheduler.step()
        avg_loss = total_loss / max(1, len(X_t) // batch_size)
        acc = n_correct / max(1, n_total) * 100

        if epoch % max(1, epochs // 10) == 0:
            elapsed = time.time() - t0
            print(f"    Epoch {epoch:>4}: loss={avg_loss:.4f} acc={acc:.1f}% "
                  f"lr={scheduler.get_last_lr()[0]:.6f} [{elapsed:.0f}s]")

    elapsed = time.time() - t0
    print(f"  Training complete: {elapsed:.0f}s, final acc={acc:.1f}%")

    # Save
    torch.save(model.state_dict(), SAVE_DIR / "model.pt")
    print(f"  Saved to {SAVE_DIR / 'model.pt'}")

    return model


# ══════════════════════════════════════════════════════════════════════
# 6. ONNX EXPORT + INT8 QUANTIZATION
# ══════════════════════════════════════════════════════════════════════

def export_onnx(model=None, path=None):
    """Export to ONNX for NPU deployment."""
    if not HAS_TORCH:
        print("PyTorch required!")
        return

    if model is None:
        model = ArcSynthesizer()
        model.load_state_dict(torch.load(SAVE_DIR / "model.pt", map_location="cpu"))

    model.eval()
    path = path or str(SAVE_DIR / "model.onnx")

    # Static shapes for NPU
    dummy_ctx = torch.randn(1, GRID_SIZE * GRID_SIZE * 10 * 2)
    dummy_tokens = torch.randint(0, VOCAB_SIZE, (1, MAX_SEQ - 1))

    torch.onnx.export(
        model, (dummy_ctx, dummy_tokens), path,
        input_names=["context", "tokens"],
        output_names=["logits"],
        opset_version=14,
        do_constant_folding=True,
    )
    print(f"  Exported ONNX: {path}")

    # INT8 quantization
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        int8_path = path.replace(".onnx", "_int8.onnx")
        quantize_dynamic(path, int8_path, weight_type=QuantType.QInt8)
        print(f"  Quantized INT8: {int8_path}")
    except Exception as e:
        print(f"  Quantization failed: {e}")


# ══════════════════════════════════════════════════════════════════════
# 7. NPU/ONNX INFERENCE
# ══════════════════════════════════════════════════════════════════════

def solve_with_npu(task, model=None, onnx_session=None, beam_size=10):
    """Solve ARC task using trained model with multi-pair context."""
    pairs = task['train']
    tests = task['test']

    # Try encoding from EACH training pair (not just first)
    all_ops_to_try = set()
    
    if model is not None and HAS_TORCH:
        for p in pairs[:3]:  # use up to 3 pairs for diversity
            ctx = encode_pair_numpy(p['input'], p['output'])
            ctx_t = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)
            beams = model.beam_search(ctx_t, beam_size=beam_size)
            for beam_tokens in beams:
                ops = [ID_TO_OP[t] for t in beam_tokens if t in ID_TO_OP]
                if ops:
                    all_ops_to_try.add(tuple(ops))
    else:
        return None, None

    # Execute each unique program
    for ops in all_ops_to_try:
        ops = list(ops)
        
        # Single op
        if len(ops) >= 1:
            op_name = ops[0]
            if op_name in OP_FNS:
                try:
                    rule = OP_FNS[op_name](pairs)
                    if rule:
                        gs = []
                        ok = True
                        for tc in tests:
                            try:
                                gs.append([rule(tc['input']).tolist()])
                            except Exception:
                                ok = False
                                break
                        if ok and gs and score_task(task, gs):
                            return gs, f"NPU:{op_name}"
                except Exception:
                    pass

        # 2-step chain
        if len(ops) >= 2:
            try:
                rule_a = OP_FNS[ops[0]](pairs)
                if rule_a:
                    mids = [{'input': rule_a(p['input']).tolist(),
                              'output': p['output']} for p in pairs]
                    rule_b = OP_FNS[ops[1]](mids)
                    if rule_b:
                        gs = []
                        ok = True
                        for tc in tests:
                            try:
                                mid = rule_a(tc['input'])
                                gs.append([rule_b(mid.tolist()).tolist()])
                            except Exception:
                                ok = False
                                break
                        if ok and gs and score_task(task, gs):
                            return gs, f"NPU:{ops[0]}→{ops[1]}"
            except Exception:
                pass

    return None, None


# ══════════════════════════════════════════════════════════════════════
# 8. TESTS
# ══════════════════════════════════════════════════════════════════════

def run_tests():
    print("=" * 65)
    print("  NPU Program Synthesizer Tests")
    print("=" * 65)
    p = 0; t = 0

    print(f"\n  T1: PyTorch available")
    ok = HAS_TORCH
    print(f"    PyTorch: {HAS_TORCH} {'PASS' if ok else 'SKIP'}")
    p += int(ok); t += 1

    print(f"\n  T2: Grid encoding")
    ctx = encode_pair_numpy([[1, 2], [3, 4]], [[4, 3], [2, 1]])
    ok = ctx.shape == (GRID_SIZE * GRID_SIZE * 10 * 2,)
    print(f"    Shape: {ctx.shape} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print(f"\n  T3: Vocabulary")
    ok = VOCAB_SIZE >= 50
    print(f"    Vocab: {VOCAB_SIZE} ops: {len(OP_NAMES)} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    if HAS_TORCH:
        print(f"\n  T4: Model forward")
        model = ArcSynthesizer(input_dim=ctx.shape[0])
        ctx_t = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)
        tok_t = torch.tensor([[START, 5]], dtype=torch.long)
        logits = model(ctx_t, tok_t)
        ok = logits.shape == (1, 2, VOCAB_SIZE)
        n_params = sum(pp.numel() for pp in model.parameters())
        print(f"    Logits: {logits.shape}, params: {n_params:,} {'PASS' if ok else 'FAIL'}")
        p += int(ok); t += 1

        print(f"\n  T5: Beam search")
        beams = model.beam_search(ctx_t, beam_size=3, max_len=4)
        ok = len(beams) >= 1
        print(f"    Beams: {len(beams)}, best: {beams[0] if beams else 'none'} {'PASS' if ok else 'FAIL'}")
        p += int(ok); t += 1

        print(f"\n  T6: Generation")
        tokens = model.generate(ctx_t, max_len=4)
        ok = tokens.shape[1] >= 1
        print(f"    Generated: {tokens.shape} {'PASS' if ok else 'FAIL'}")
        p += int(ok); t += 1
    else:
        print(f"\n  T4-6: Skipped (no PyTorch)")
        t += 3

    print(f"\n{'='*65}")
    print(f"  Results: {p}/{t} tests passed")
    print(f"{'='*65}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--solve", action="store_true")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--beam", type=int, default=10)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--data", type=str, default=None)
    args = ap.parse_args()

    if args.data:
        data_base = args.data
    else:
        for base in [".", "ARC-AGI-2", "../ARC-AGI-2",
                       "C:/Users/MeteorAI/Desktop/ARC-AGI-2"]:
            candidate = os.path.join(base, "data")
            if os.path.exists(candidate):
                data_base = candidate
                break
        else:
            data_base = "ARC-AGI-2/data"

    if args.test:
        run_tests()
    elif args.train:
        d = os.path.join(data_base, "training")
        model = train_model(d, epochs=args.epochs)
        if model and args.export:
            export_onnx(model)
    elif args.export:
        export_onnx()
    elif args.solve:
        if not HAS_TORCH:
            print("PyTorch required!")
        else:
            d = os.path.join(data_base, "training")
            model = ArcSynthesizer(input_dim=GRID_SIZE * GRID_SIZE * 10 * 2)
            pt_path = SAVE_DIR / "model.pt"
            if pt_path.exists():
                model.load_state_dict(torch.load(pt_path, map_location="cpu"))
            else:
                print("  No trained model. Training first...")
                model = train_model(d, epochs=args.epochs)

            model.eval()
            files = sorted(f for f in os.listdir(d) if f.endswith('.json'))
            s1_n = ns_n = 0
            for f in files:
                task = json.load(open(os.path.join(d, f)))
                s1 = s1_solve(task)
                if score_task(task, s1):
                    s1_n += 1
                    continue
                preds, method = solve_with_npu(task, model=model, beam_size=args.beam)
                if preds and score_task(task, preds):
                    ns_n += 1
                    if args.verbose:
                        print(f"  {f}: {method}")
            print(f"\n  Results: {s1_n + ns_n}/{len(files)} (S1:{s1_n}, NPU:{ns_n})")
    else:
        run_tests()
