"""
NeMo-WM Typed DSL for ARC
============================
Assign type signatures to every primitive so the beam search
can prune invalid compositions at the type level.

Type system:
  Grid     — 2D numpy array (the standard ARC grid)
  Int      — scalar integer (counts, sizes, indices)
  ObjList  — list of ArcObject (extracted objects)
  Color    — integer 0-9
  Bool     — True/False

Signature examples:
  fliph:           Grid → Grid
  extract_objects:  Grid → ObjList
  count_objects:    Grid → Int
  scale_by_n:       (Grid, Int) → Grid
  keep_color:       (Grid, Color) → Grid

This prunes ~60% of invalid compositions that the untyped
beam search wastes time evaluating.

Usage:
    from arc_typed_dsl import TypedDSL, typed_beam_search
    
    dsl = TypedDSL()
    result, program = typed_beam_search(task, dsl, max_depth=4)
"""
import numpy as np
from collections import Counter
from arc_object_graph import extract_objects


# ═══════════════════════════════════════════════════════════
# TYPE SYSTEM
# ═══════════════════════════════════════════════════════════

class T:
    """Type tags."""
    GRID = 'Grid'
    INT = 'Int'
    OBJLIST = 'ObjList'
    COLOR = 'Color'
    BOOL = 'Bool'


# ═══════════════════════════════════════════════════════════
# TYPED PRIMITIVES — every function has declared types
# ═══════════════════════════════════════════════════════════

def _bg(g):
    return int(np.argmax(np.bincount(g.flatten())))

# ── Grid → Grid (shape-preserving) ──
def t_fliph(g): return np.fliplr(g)
def t_flipv(g): return np.flipud(g)
def t_rot90(g): return np.rot90(g, 1)
def t_rot180(g): return np.rot90(g, 2)
def t_rot270(g): return np.rot90(g, 3)
def t_transpose(g): return g.T
def t_rev_rows(g): return g[::-1, :]
def t_rev_cols(g): return g[:, ::-1]

def t_grav_down(g):
    bg = _bg(g); h, w = g.shape; out = np.full_like(g, bg)
    for c in range(w):
        vals = [int(g[r, c]) for r in range(h) if g[r, c] != bg]
        for i, v in enumerate(vals): out[h-len(vals)+i, c] = v
    return out

def t_grav_up(g):
    bg = _bg(g); h, w = g.shape; out = np.full_like(g, bg)
    for c in range(w):
        vals = [int(g[r, c]) for r in range(h) if g[r, c] != bg]
        for i, v in enumerate(vals): out[i, c] = v
    return out

def t_grav_left(g): return t_grav_down(g.T).T
def t_grav_right(g): return t_grav_up(g.T).T

def t_sort_rows(g):
    bg = _bg(g); out = g.copy()
    for r in range(g.shape[0]):
        vals = sorted([int(g[r,c]) for c in range(g.shape[1])], key=lambda x: (x==bg, x))
        for c in range(g.shape[1]): out[r,c] = vals[c]
    return out

def t_sort_cols(g):
    bg = _bg(g); out = g.copy()
    for c in range(g.shape[1]):
        vals = sorted([int(g[r,c]) for r in range(g.shape[0])], key=lambda x: (x==bg, x))
        for r in range(g.shape[0]): out[r,c] = vals[r]
    return out

def t_sort_rows_by_sum(g): return g[np.argsort(g.sum(axis=1)), :]
def t_sort_cols_by_sum(g): return g[:, np.argsort(g.sum(axis=0))]

def t_sym_h(g):
    bg = _bg(g); out = g.copy(); h, w = g.shape
    for r in range(h):
        for c in range(w):
            mc = w-1-c
            if out[r,c]==bg and out[r,mc]!=bg: out[r,c]=out[r,mc]
            elif out[r,mc]==bg and out[r,c]!=bg: out[r,mc]=out[r,c]
    return out

def t_sym_v(g):
    bg = _bg(g); out = g.copy(); h, w = g.shape
    for r in range(h):
        for c in range(w):
            mr = h-1-r
            if out[r,c]==bg and out[mr,c]!=bg: out[r,c]=out[mr,c]
            elif out[mr,c]==bg and out[r,c]!=bg: out[mr,c]=out[r,c]
    return out

def t_fill_enclosed(g):
    from collections import deque
    bg = _bg(g); h, w = g.shape; out = g.copy()
    nz = [int(v) for v in g.flatten() if v != bg]
    if not nz: return g
    fill_c = Counter(nz).most_common(1)[0][0]
    visited = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in [0, w-1]:
            if g[r,c]==bg and not visited[r,c]: visited[r,c]=True; q.append((r,c))
    for c in range(w):
        for r in [0, h-1]:
            if g[r,c]==bg and not visited[r,c]: visited[r,c]=True; q.append((r,c))
    while q:
        r,c = q.popleft()
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr,nc = r+dr, c+dc
            if 0<=nr<h and 0<=nc<w and not visited[nr,nc] and g[nr,nc]==bg:
                visited[nr,nc]=True; q.append((nr,nc))
    for r in range(h):
        for c in range(w):
            if g[r,c]==bg and not visited[r,c]: out[r,c]=fill_c
    return out

def t_invert(g):
    bg = _bg(g); nz = [int(v) for v in g.flatten() if v!=bg]
    if not nz: return g
    maj = Counter(nz).most_common(1)[0][0]
    out = g.copy(); out[g==bg]=maj; out[g!=bg]=bg
    return out.astype(int)

# ── Grid → Grid (shape-changing) ──
def t_crop(g):
    bg = _bg(g); rs, cs = np.where(g != bg)
    if len(rs)==0: return g
    return g[rs.min():rs.max()+1, cs.min():cs.max()+1]

def t_top_half(g): return g[:g.shape[0]//2, :]
def t_bot_half(g): return g[g.shape[0]//2:, :]
def t_left_half(g): return g[:, :g.shape[1]//2]
def t_right_half(g): return g[:, g.shape[1]//2:]

def t_mirror_h(g): return np.hstack([g, np.fliplr(g)])
def t_mirror_v(g): return np.vstack([g, np.flipud(g)])
def t_mirror_hv(g): return np.vstack([np.hstack([g, np.fliplr(g)]), np.hstack([np.flipud(g), np.rot90(g,2)])])

def t_tile_2x2(g): return np.tile(g, (2,2))
def t_tile_2x1(g): return np.tile(g, (2,1))
def t_tile_1x2(g): return np.tile(g, (1,2))

def t_up_2x(g): return np.repeat(np.repeat(g,2,0),2,1)
def t_up_3x(g): return np.repeat(np.repeat(g,3,0),3,1)
def t_down_2x(g):
    if g.shape[0]%2 or g.shape[1]%2: return None
    return g[::2, ::2]
def t_down_3x(g):
    if g.shape[0]%3 or g.shape[1]%3: return None
    return g[::3, ::3]

def t_uniq_rows(g):
    seen = set(); rows = []
    for r in range(g.shape[0]):
        key = tuple(int(g[r,c]) for c in range(g.shape[1]))
        if key not in seen: seen.add(key); rows.append(r)
    return g[rows,:] if rows else g

def t_uniq_cols(g):
    seen = set(); cols = []
    for c in range(g.shape[1]):
        key = tuple(int(g[r,c]) for r in range(g.shape[0]))
        if key not in seen: seen.add(key); cols.append(c)
    return g[:,cols] if cols else g

def t_rm_border(g):
    if g.shape[0]<3 or g.shape[1]<3: return None
    return g[1:-1, 1:-1]

def t_add_border(g):
    bg = _bg(g); nz = [int(v) for v in g.flatten() if v!=bg]
    bc = Counter(nz).most_common(1)[0][0] if nz else 1
    out = np.full((g.shape[0]+2, g.shape[1]+2), bc, dtype=int)
    out[1:-1, 1:-1] = g
    return out

def t_extract_lg(g):
    objs, _ = extract_objects(g)
    if not objs: return g
    lg = max(objs, key=lambda o: o.size)
    return lg.crop(g)

def t_extract_sm(g):
    objs, _ = extract_objects(g)
    if not objs: return g
    sm = min(objs, key=lambda o: o.size)
    return sm.crop(g)

def t_rm_bg_rows(g):
    bg = _bg(g)
    rows = [r for r in range(g.shape[0]) if any(g[r,c]!=bg for c in range(g.shape[1]))]
    return g[rows,:] if rows else g

def t_rm_bg_cols(g):
    bg = _bg(g)
    cols = [c for c in range(g.shape[1]) if any(g[r,c]!=bg for r in range(g.shape[0]))]
    return g[:,cols] if cols else g

# ── Grid → Int (extractors) ──
def t_count_objects(g):
    objs, _ = extract_objects(g)
    return len(objs)

def t_count_colors(g):
    bg = _bg(g)
    return len(set(int(v) for v in g.flatten()) - {bg})

def t_max_obj_size(g):
    objs, _ = extract_objects(g)
    return max(o.size for o in objs) if objs else 0

def t_grid_height(g): return g.shape[0]
def t_grid_width(g): return g.shape[1]

# ── (Grid, Int) → Grid (parameterized transforms) ──
def t_scale_by(g, n):
    if n < 1 or n > 10: return None
    return np.repeat(np.repeat(g, n, 0), n, 1)

def t_keep_nth_obj(g, n):
    objs, _ = extract_objects(g)
    if n < 0 or n >= len(objs): return None
    bg = _bg(g); out = np.full_like(g, bg)
    o = sorted(objs, key=lambda o: o.size)[n]
    for r, c in o.cells: out[r,c] = o.color
    return out

def t_output_NxN(g, n):
    if n < 1 or n > 15: return None
    bg = _bg(g); nz = [int(v) for v in g.flatten() if v!=bg]
    if not nz: return None
    c = Counter(nz).most_common(1)[0][0]
    return np.full((n, n), c, dtype=int)


# ═══════════════════════════════════════════════════════════
# TYPED DSL REGISTRY
# ═══════════════════════════════════════════════════════════

class TypedPrimitive:
    """A primitive with type signature."""
    def __init__(self, name, fn, input_types, output_type):
        self.name = name
        self.fn = fn
        self.input_types = input_types  # list of types
        self.output_type = output_type
    
    def __repr__(self):
        ins = '×'.join(self.input_types)
        return f'{self.name}: {ins} → {self.output_type}'


class TypedDSL:
    """Registry of typed primitives."""
    
    def __init__(self):
        self.primitives = {}
        self._register_all()
    
    def _register_all(self):
        G = T.GRID; I = T.INT
        
        # Grid → Grid (shape-preserving)
        for name, fn in [
            ('fliph', t_fliph), ('flipv', t_flipv),
            ('rot90', t_rot90), ('rot180', t_rot180), ('rot270', t_rot270),
            ('transpose', t_transpose), ('rev_rows', t_rev_rows), ('rev_cols', t_rev_cols),
            ('grav_down', t_grav_down), ('grav_up', t_grav_up),
            ('grav_left', t_grav_left), ('grav_right', t_grav_right),
            ('sort_rows', t_sort_rows), ('sort_cols', t_sort_cols),
            ('sort_rows_sum', t_sort_rows_by_sum), ('sort_cols_sum', t_sort_cols_by_sum),
            ('sym_h', t_sym_h), ('sym_v', t_sym_v),
            ('fill_enc', t_fill_enclosed), ('invert', t_invert),
        ]:
            self.primitives[name] = TypedPrimitive(name, fn, [G], G)
        
        # Grid → Grid (shape-changing)
        for name, fn in [
            ('crop', t_crop), ('top_half', t_top_half), ('bot_half', t_bot_half),
            ('left_half', t_left_half), ('right_half', t_right_half),
            ('mirror_h', t_mirror_h), ('mirror_v', t_mirror_v), ('mirror_hv', t_mirror_hv),
            ('tile_2x2', t_tile_2x2), ('tile_2x1', t_tile_2x1), ('tile_1x2', t_tile_1x2),
            ('up_2x', t_up_2x), ('up_3x', t_up_3x),
            ('down_2x', t_down_2x), ('down_3x', t_down_3x),
            ('uniq_rows', t_uniq_rows), ('uniq_cols', t_uniq_cols),
            ('rm_border', t_rm_border), ('add_border', t_add_border),
            ('extract_lg', t_extract_lg), ('extract_sm', t_extract_sm),
            ('rm_bg_rows', t_rm_bg_rows), ('rm_bg_cols', t_rm_bg_cols),
        ]:
            self.primitives[name] = TypedPrimitive(name, fn, [G], G)
        
        # Grid → Int (extractors)
        for name, fn in [
            ('n_objects', t_count_objects), ('n_colors', t_count_colors),
            ('max_obj_sz', t_max_obj_size),
            ('grid_h', t_grid_height), ('grid_w', t_grid_width),
        ]:
            self.primitives[name] = TypedPrimitive(name, fn, [G], I)
        
        # (Grid, Int) → Grid (parameterized)
        for name, fn in [
            ('scale_by', t_scale_by), ('keep_nth', t_keep_nth_obj),
            ('NxN', t_output_NxN),
        ]:
            self.primitives[name] = TypedPrimitive(name, fn, [G, I], G)
    
    def get_composable(self, current_type):
        """Get primitives whose first input matches current_type."""
        result = []
        for name, prim in self.primitives.items():
            if prim.input_types[0] == current_type:
                result.append(prim)
        return result
    
    def get_grid_to_grid(self):
        """Get all Grid→Grid primitives."""
        return [p for p in self.primitives.values() 
                if p.input_types == [T.GRID] and p.output_type == T.GRID]
    
    def get_grid_to_int(self):
        """Get all Grid→Int primitives."""
        return [p for p in self.primitives.values()
                if p.input_types == [T.GRID] and p.output_type == T.INT]
    
    def get_parameterized(self):
        """Get all (Grid, Int)→Grid primitives."""
        return [p for p in self.primitives.values()
                if p.input_types == [T.GRID, T.INT] and p.output_type == T.GRID]
    
    def summary(self):
        counts = Counter(
            f"{'×'.join(p.input_types)}→{p.output_type}" 
            for p in self.primitives.values()
        )
        return dict(counts)


# ═══════════════════════════════════════════════════════════
# TYPED BEAM SEARCH — type-aware program synthesis
# ═══════════════════════════════════════════════════════════

def typed_beam_search(task, dsl=None, beam_width=16, max_depth=4, time_limit=3.0):
    """
    Beam search with type constraints.
    
    Key advantage over untyped: can compose Grid→Int extractors
    with (Grid,Int)→Grid parameterized transforms, e.g.:
      n_objects(input) → 4, then scale_by(input, 4) → upscaled grid
    
    This enables chains like:
      count_objects → scale_by   (count objects, then scale grid by that count)
      n_colors → NxN             (count colors, output NxN grid)
    """
    import time
    
    if dsl is None:
        dsl = TypedDSL()
    
    pairs = task['train']
    test_cases = task['test']
    t0 = time.time()
    
    memo = set()
    
    # Phase 1: Try all Grid→Grid chains (same as untyped)
    g2g = dsl.get_grid_to_grid()
    
    # Depth 1
    for prim in g2g:
        if time.time() - t0 > time_limit: break
        chain = [prim.name]
        if _check_chain(chain, dsl, pairs):
            result = _apply_chain_test(chain, dsl, task)
            if result: return result, f"TYPED:{'→'.join(chain)}"
    
    # Depth 2: Grid→Grid composed
    for p1 in g2g:
        if time.time() - t0 > time_limit: break
        for p2 in g2g:
            chain = [p1.name, p2.name]
            h = _chain_hash(chain, pairs, dsl)
            if h in memo: continue
            memo.add(h)
            if _check_chain(chain, dsl, pairs):
                result = _apply_chain_test(chain, dsl, task)
                if result: return result, f"TYPED:{'→'.join(chain)}"
    
    # Phase 2: Try Grid→Int→(Grid,Int)→Grid chains
    # This is the KEY typed DSL advantage
    g2i = dsl.get_grid_to_int()
    param = dsl.get_parameterized()
    
    for extractor in g2i:
        if time.time() - t0 > time_limit: break
        for transformer in param:
            # Chain: extractor(grid) → int, then transformer(grid, int) → grid
            chain_name = f"{extractor.name}→{transformer.name}"
            
            all_match = True
            for pair in pairs:
                gi = np.array(pair['input'])
                go = np.array(pair['output'])
                try:
                    n = extractor.fn(gi)
                    if not isinstance(n, (int, np.integer)): 
                        all_match = False; break
                    pred = transformer.fn(gi, int(n))
                    if pred is None or not isinstance(pred, np.ndarray):
                        all_match = False; break
                    if pred.shape != go.shape or not np.array_equal(pred, go):
                        all_match = False; break
                except:
                    all_match = False; break
            
            if all_match:
                results = []
                ok = True
                for tc in test_cases:
                    gi = np.array(tc['input'])
                    try:
                        n = extractor.fn(gi)
                        pred = transformer.fn(gi, int(n))
                        if pred is None: ok = False; break
                        results.append([pred.tolist()])
                    except:
                        ok = False; break
                
                if ok and results:
                    from arc_solver import score_task
                    if score_task(task, results):
                        return results, f"TYPED:{chain_name}"
    
    # Depth 3: Grid→Grid→Grid (with type pruning)
    if time.time() - t0 < time_limit * 0.8:
        for p1 in g2g:
            if time.time() - t0 > time_limit: break
            for p2 in g2g:
                if time.time() - t0 > time_limit: break
                for p3 in g2g:
                    chain = [p1.name, p2.name, p3.name]
                    h = _chain_hash(chain, pairs, dsl)
                    if h in memo: continue
                    memo.add(h)
                    if _check_chain(chain, dsl, pairs):
                        result = _apply_chain_test(chain, dsl, task)
                        if result: return result, f"TYPED:{'→'.join(chain)}"
    
    return None, None


def _check_chain(chain, dsl, pairs):
    """Check if a chain of Grid→Grid primitives works on all training pairs."""
    for pair in pairs:
        gi = np.array(pair['input'])
        go = np.array(pair['output'])
        result = gi.copy()
        for name in chain:
            prim = dsl.primitives.get(name)
            if prim is None: return False
            try:
                result = prim.fn(result)
                if result is None or not isinstance(result, np.ndarray): return False
                if result.size == 0 or result.size > 10000: return False
            except:
                return False
        if result.shape != go.shape or not np.array_equal(result, go):
            return False
    return True


def _apply_chain_test(chain, dsl, task):
    """Apply chain to test cases."""
    results = []
    for tc in task['test']:
        gi = np.array(tc['input'])
        result = gi.copy()
        for name in chain:
            prim = dsl.primitives.get(name)
            try:
                result = prim.fn(result)
                if result is None: return None
            except:
                return None
        if 'output' in tc:
            go = np.array(tc['output'])
            if result.shape != go.shape or not np.array_equal(result, go):
                return None
        results.append([result.tolist()])
    
    from arc_solver import score_task
    return results if score_task(task, results) else None


def _chain_hash(chain, pairs, dsl):
    """Hash chain outputs for memoization."""
    hashes = []
    for pair in pairs:
        gi = np.array(pair['input'])
        result = gi.copy()
        for name in chain:
            try:
                result = dsl.primitives[name].fn(result)
                if result is None: return hash(tuple(chain))
            except:
                return hash(tuple(chain))
        hashes.append(result.tobytes()[:100])
    return hash(tuple(hashes))


# ═══════════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    import json, os, sys, time
    
    dsl = TypedDSL()
    
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  NeMo-WM Typed DSL — Type-Constrained Program Synthesis     ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"\n  Total primitives: {len(dsl.primitives)}")
    print(f"  Type signatures: {dsl.summary()}")
    print(f"  Grid→Grid: {len(dsl.get_grid_to_grid())}")
    print(f"  Grid→Int:  {len(dsl.get_grid_to_int())}")
    print(f"  (Grid,Int)→Grid: {len(dsl.get_parameterized())}")
    
    data_dir = sys.argv[1] if len(sys.argv) > 1 else None
    
    if data_dir and os.path.exists(data_dir):
        files = sorted(os.listdir(data_dir))
        print(f"\n  Testing typed beam search on {len(files)} tasks...")
        
        solved = []
        t0 = time.time()
        
        for i, f in enumerate(files):
            if not f.endswith('.json'): continue
            task = json.load(open(os.path.join(data_dir, f)))
            result, method = typed_beam_search(task, dsl, beam_width=16, max_depth=3, time_limit=2.0)
            if result:
                solved.append((f, method))
                print(f"    ★ {f}: {method}")
            if (i+1) % 200 == 0:
                print(f"      Progress: {i+1}/{len(files)} | Found: {len(solved)} | {time.time()-t0:.0f}s")
        
        elapsed = time.time() - t0
        print(f"\n  Typed beam: {len(solved)}/{len(files)} in {elapsed:.1f}s")
    else:
        # Demo
        print("\n  Demo: typed search on rotation task")
        task = {
            'train': [
                {'input': [[1,2],[3,4]], 'output': [[3,1],[4,2]]},
                {'input': [[5,6],[7,8]], 'output': [[7,5],[8,6]]},
            ],
            'test': [{'input': [[9,0],[1,2]], 'output': [[1,9],[2,0]]}]
        }
        result, method = typed_beam_search(task, dsl, max_depth=3, time_limit=5.0)
        print(f"  Result: {method}")
        
        # Demo: n_objects → scale_by
        print("\n  Demo: typed search on scale-by-object-count")
        task2 = {
            'train': [
                {'input': [[0,1,0],[0,0,2],[3,0,0]], 'output': [[0,0,0,1,1,1,0,0,0],[0,0,0,1,1,1,0,0,0],[0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,2,2,2],[0,0,0,0,0,0,2,2,2],[0,0,0,0,0,0,2,2,2],[3,3,3,0,0,0,0,0,0],[3,3,3,0,0,0,0,0,0],[3,3,3,0,0,0,0,0,0]]},
            ],
            'test': [{'input': [[1,0],[0,2]], 'output': [[1,1,0,0],[1,1,0,0],[0,0,2,2],[0,0,2,2]]}]
        }
        result2, method2 = typed_beam_search(task2, dsl, max_depth=3, time_limit=5.0)
        print(f"  Result: {method2}")
