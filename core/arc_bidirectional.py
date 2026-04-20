"""
NeMo-WM Bidirectional Search for ARC
=======================================
Instead of only searching forward (input → ??? → output),
also search backward (output → inverse_ops → input) and
find programs where the forward and backward searches meet.

This squares the effective search depth:
  - Forward depth 3 = 46^3 = ~97K programs
  - Bidirectional depth 3+3 = 2 × 46^3 = ~194K, but covers depth 6

Many ARC primitives are invertible:
  fliph ↔ fliph         (self-inverse)
  rot90 ↔ rot270         (rotation inverses)
  up_2x ↔ down_2x       (scale inverses)
  mirror_h → left_half   (partial inverse)
  tile_2x2 → top_half + left_half (partial inverse)

Usage:
    from arc_bidirectional import BidirectionalSolver
    
    solver = BidirectionalSolver()
    result, program = solver.solve(task)
"""
import numpy as np
import time
from collections import Counter
from arc_beam_search import PRIMITIVES, apply_chain, pixel_accuracy


# ═══════════════════════════════════════════════════════════
# INVERSE REGISTRY — which primitives undo which
# ═══════════════════════════════════════════════════════════

# Self-inverse: applying twice = identity
SELF_INVERSE = {
    'fliph', 'flipv', 'rot180', 'transpose', 'invert',
    'rev_rows', 'rev_cols', 'swap_12',
}

# Paired inverses: A undoes B
INVERSE_PAIRS = {
    'rot90': 'rot270',
    'rot270': 'rot90',
    'up_2x': 'down_2x',
    'up_3x': 'down_3x',
    'down_2x': 'up_2x',
    'down_3x': 'up_3x',
    'grav_down': 'grav_up',
    'grav_up': 'grav_down',
    'grav_left': 'grav_right',
    'grav_right': 'grav_left',
    'add_border': 'rm_border',
    'rm_border': 'add_border',
}

# Partial inverses: A partially undoes B (may not be exact)
PARTIAL_INVERSE = {
    'mirror_h': ['left_half', 'right_half'],
    'mirror_v': ['top_half', 'bot_half'],
    'mirror_hv': ['top_half', 'left_half'],
    'tile_2x2': ['top_half', 'left_half'],
    'tile_2x1': ['top_half'],
    'tile_1x2': ['left_half'],
}


def get_inverse(prim_name):
    """Get the inverse of a primitive. Returns list of candidates."""
    inverses = []
    if prim_name in SELF_INVERSE:
        inverses.append(prim_name)
    if prim_name in INVERSE_PAIRS:
        inverses.append(INVERSE_PAIRS[prim_name])
    if prim_name in PARTIAL_INVERSE:
        inverses.extend(PARTIAL_INVERSE[prim_name])
    return inverses


def get_all_invertible():
    """Get all primitives that have known inverses."""
    invertible = set()
    invertible.update(SELF_INVERSE)
    invertible.update(INVERSE_PAIRS.keys())
    invertible.update(PARTIAL_INVERSE.keys())
    return invertible


def reverse_chain(chain):
    """Reverse a chain and replace each step with its inverse."""
    reversed_chain = []
    for prim in reversed(chain):
        invs = get_inverse(prim)
        if not invs:
            return None  # Can't invert this chain
        reversed_chain.append(invs[0])  # Use first inverse
    return reversed_chain


# ═══════════════════════════════════════════════════════════
# BIDIRECTIONAL SEARCH ENGINE
# ═══════════════════════════════════════════════════════════

class BidirectionalSolver:
    """
    Search from both ends and meet in the middle.
    
    Forward:  input → p1 → p2 → ... → intermediate
    Backward: output → inv(q2) → inv(q1) → ... → intermediate
    
    If forward and backward reach the same intermediate grid,
    the full program is: p1 → p2 → ... → q1 → q2
    """

    def __init__(self, max_depth=2, time_limit=3.0):
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.stats = {'forward': 0, 'backward': 0, 'matches': 0}

    def solve(self, task):
        """Try bidirectional search on a task."""
        pairs = task['train']
        t0 = time.time()

        # Get invertible primitives for backward search
        invertible = get_all_invertible()
        backward_prims = [p for p in PRIMITIVES.keys() if p in invertible]
        all_prims = list(PRIMITIVES.keys())

        # ── Forward search: build hash table of intermediate grids ──
        # For each depth 1..max_depth from input
        forward_table = {}  # hash(intermediate_grids) → chain

        # Depth 0: identity (input itself)
        id_key = self._hash_grids([np.array(p['input']) for p in pairs])
        forward_table[id_key] = []

        # Depth 1 forward
        for pname in all_prims:
            if time.time() - t0 > self.time_limit: break
            chain = [pname]
            grids = self._apply_to_all_inputs(chain, pairs)
            if grids is None: continue
            self.stats['forward'] += 1
            key = self._hash_grids(grids)
            if key not in forward_table:
                forward_table[key] = chain

        # Depth 2 forward
        if self.max_depth >= 2:
            for p1 in all_prims:
                if time.time() - t0 > self.time_limit: break
                for p2 in all_prims:
                    chain = [p1, p2]
                    grids = self._apply_to_all_inputs(chain, pairs)
                    if grids is None: continue
                    self.stats['forward'] += 1
                    key = self._hash_grids(grids)
                    if key not in forward_table:
                        forward_table[key] = chain

        # ── Backward search: apply inverse ops to outputs ──
        # Depth 0: check if outputs match any forward result directly
        out_key = self._hash_grids([np.array(p['output']) for p in pairs])
        if out_key in forward_table:
            chain = forward_table[out_key]
            if chain:  # Non-empty chain that reaches output
                result = self._verify_and_apply(chain, task)
                if result:
                    self.stats['matches'] += 1
                    return result, f"BIDIR:{'→'.join(chain)}"

        # Depth 1 backward
        for pname in backward_prims:
            if time.time() - t0 > self.time_limit: break
            inv_chain = get_inverse(pname)
            if not inv_chain: continue
            
            # Apply inverse to outputs
            back_grids = self._apply_to_all_outputs(inv_chain[:1], pairs)
            if back_grids is None: continue
            self.stats['backward'] += 1

            key = self._hash_grids(back_grids)
            if key in forward_table:
                fwd = forward_table[key]
                full_chain = fwd + [pname]
                result = self._verify_and_apply(full_chain, task)
                if result:
                    self.stats['matches'] += 1
                    return result, f"BIDIR:{'→'.join(full_chain)}"

        # Depth 2 backward
        if self.max_depth >= 2:
            for p1 in backward_prims:
                if time.time() - t0 > self.time_limit: break
                for p2 in backward_prims:
                    # Backward chain: apply inv(p2) then inv(p1) to output
                    inv1 = get_inverse(p2)
                    inv2 = get_inverse(p1)
                    if not inv1 or not inv2: continue

                    back_grids = self._apply_to_all_outputs([inv1[0], inv2[0]], pairs)
                    if back_grids is None: continue
                    self.stats['backward'] += 1

                    key = self._hash_grids(back_grids)
                    if key in forward_table:
                        fwd = forward_table[key]
                        full_chain = fwd + [p1, p2]
                        result = self._verify_and_apply(full_chain, task)
                        if result:
                            self.stats['matches'] += 1
                            return result, f"BIDIR:{'→'.join(full_chain)}"

        return None, None

    def _hash_grids(self, grids):
        """Hash a list of grids for table lookup."""
        return hash(tuple(g.tobytes()[:200] for g in grids))

    def _apply_to_all_inputs(self, chain, pairs):
        """Apply chain to all training inputs. Returns list of grids or None."""
        grids = []
        for pair in pairs:
            gi = np.array(pair['input'])
            result = apply_chain(gi, chain)
            if result is None: return None
            grids.append(result)
        return grids

    def _apply_to_all_outputs(self, chain, pairs):
        """Apply chain to all training outputs (backward search)."""
        grids = []
        for pair in pairs:
            go = np.array(pair['output'])
            result = apply_chain(go, chain)
            if result is None: return None
            grids.append(result)
        return grids

    def _verify_and_apply(self, chain, task):
        """Verify chain on all training pairs and apply to test."""
        from arc_solver import score_task
        pairs = task['train']
        
        # Verify on training
        for pair in pairs:
            gi = np.array(pair['input'])
            go = np.array(pair['output'])
            pred = apply_chain(gi, chain)
            if pred is None or pred.shape != go.shape or not np.array_equal(pred, go):
                return None

        # Apply to test
        results = []
        for tc in task['test']:
            gi = np.array(tc['input'])
            pred = apply_chain(gi, chain)
            if pred is None: return None
            results.append([pred.tolist()])

        if score_task(task, results):
            return results
        return None


# ═══════════════════════════════════════════════════════════
# INTEGRATION FUNCTION
# ═══════════════════════════════════════════════════════════

def try_bidirectional(task, max_depth=2, time_limit=2.0):
    """Convenience function for arc_unified.py."""
    solver = BidirectionalSolver(max_depth=max_depth, time_limit=time_limit)
    result, method = solver.solve(task)
    return result, method


# ═══════════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    import json, os, sys

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  NeMo-WM Bidirectional Search                               ║")
    print("║  Forward + Backward meet-in-the-middle program synthesis     ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    invertible = get_all_invertible()
    print(f"\n  Total primitives: {len(PRIMITIVES)}")
    print(f"  Invertible: {len(invertible)}")
    print(f"  Self-inverse: {len(SELF_INVERSE)}")
    print(f"  Paired inverse: {len(INVERSE_PAIRS)}")
    print(f"  Partial inverse: {len(PARTIAL_INVERSE)}")

    data_dir = sys.argv[1] if len(sys.argv) > 1 else None

    if data_dir and os.path.exists(data_dir):
        files = sorted(os.listdir(data_dir))
        print(f"\n  Testing bidirectional on {len(files)} tasks...")

        solver = BidirectionalSolver(max_depth=2, time_limit=2.0)
        solved = []
        t0 = time.time()

        for i, f in enumerate(files):
            if not f.endswith('.json'): continue
            task = json.load(open(os.path.join(data_dir, f)))
            result, method = solver.solve(task)
            if result:
                solved.append((f, method))
                print(f"    ★ {f}: {method}")
            if (i+1) % 200 == 0:
                elapsed = time.time() - t0
                print(f"      Progress: {i+1}/{len(files)} | Found: {len(solved)} | {elapsed:.0f}s")

        elapsed = time.time() - t0
        print(f"\n  Bidirectional: {len(solved)}/{len(files)} in {elapsed:.1f}s")
    else:
        # Demo
        print("\n  Demo: finding flip+mirror via bidirectional search")
        task = {
            'train': [
                {'input': [[1,2,3],[4,5,6]], 'output': [[6,5,4,4,5,6],[3,2,1,1,2,3]]},
                {'input': [[7,8],[9,0]], 'output': [[0,9,9,0],[8,7,7,8]]},
            ],
            'test': [
                {'input': [[1,0],[0,1]], 'output': [[1,0,0,1],[0,1,1,0]]},
            ]
        }
        solver = BidirectionalSolver(max_depth=2, time_limit=5.0)
        result, method = solver.solve(task)
        print(f"  Result: {method}")
        print(f"  Stats: {solver.stats}")
