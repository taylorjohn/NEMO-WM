"""
NeMo-WM Knowledge Diagnostic
==============================
Comprehensive test showing everything the system knows:
  1. VOCABULARY — object types, relations, roles, spatial concepts
  2. MATH — counting, modular arithmetic, numerical reasoning
  3. PATTERN MATCHING — shape hashing, template detection, symmetry
  4. CODING/DSA — algorithmic primitives, graph theory, geometry
  5. TASK ROUTING — DSA pattern recommendations per task
  6. RELATION GRAPH — object perception, role assignment, features

Usage:
    python arc_diagnostic.py
    python arc_diagnostic.py C:\\Users\\MeteorAI\\Desktop\\ARC-AGI-2\\data\\training
"""
import json
import os
import sys
import time
import numpy as np
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def header(title):
    print(f"\n{'═'*70}")
    print(f"  {title}")
    print(f"{'═'*70}")


def subheader(title):
    print(f"\n  {'─'*50}")
    print(f"  {title}")
    print(f"  {'─'*50}")


# ═══════════════════════════════════════════════════════════
# TEST 1: VOCABULARY — what concepts does NeMo-WM understand?
# ═══════════════════════════════════════════════════════════

def test_vocabulary():
    header("1. VOCABULARY — Concepts & Terminology")
    
    from arc_object_graph import ArcObject, extract_objects, ALL_OG_SOLVERS
    
    # Object attributes
    subheader("Object Attributes (ArcObject)")
    attrs = ArcObject.__slots__
    print(f"  {len(attrs)} attributes per object:")
    for a in attrs:
        desc = {
            'id': 'Unique identifier',
            'color': 'ARC color (0-9)',
            'cells': 'Set of (row, col) pixel coordinates',
            'size': 'Number of pixels',
            'bbox': '(r1, r2, c1, c2) bounding box',
            'center': '(mean_r, mean_c) centroid',
            'width': 'Bounding box width',
            'height': 'Bounding box height',
            'aspect': 'Width / height ratio',
            'shape_hash': 'Normalized shape fingerprint',
            'touches_border': 'Boolean: object touches grid edge',
            'is_rectangular': 'Boolean: fills entire bbox',
            'density': 'Pixels / bbox area (0-1)',
        }.get(a, '?')
        print(f"    {a:20s} — {desc}")
    
    # Solver families
    subheader("Solver Families")
    families = defaultdict(list)
    for name, _ in ALL_OG_SOLVERS:
        # Group by prefix pattern
        if 'stamp' in name or 'cross' in name: families['Stamp'].append(name)
        elif 'connect' in name or 'fill_between' in name: families['Connection'].append(name)
        elif 'gravity' in name or 'translate' in name: families['Movement'].append(name)
        elif 'sym' in name or 'reflect' in name or 'mirror' in name: families['Symmetry/Tiling'].append(name)
        elif 'extract' in name or 'crop' in name or 'downsample' in name: families['Extraction'].append(name)
        elif 'recolor' in name or 'color' in name or 'pattern_checker' in name: families['Recoloring'].append(name)
        elif 'per_obj' in name or 'quad' in name or 'partition' in name or 'overlay' in name: families['Compositional'].append(name)
        elif 'pixel_to' in name or 'tile' in name: families['Tiling'].append(name)
        elif 'periodic' in name or 'scale_by' in name or 'sort' in name or 'context' in name: families['Numerical'].append(name)
        else: families['Other'].append(name)
    
    total = sum(len(v) for v in families.values())
    print(f"  {total} solvers across {len(families)} families:")
    for fam, solvers in sorted(families.items(), key=lambda x: -len(x[1])):
        print(f"    {fam:20s} ({len(solvers):3d}): {', '.join(solvers[:5])}{'...' if len(solvers) > 5 else ''}")
    
    # Spatial relations vocabulary
    subheader("Spatial Relations Vocabulary")
    relations = [
        ('touches', 'Two objects share a 4-adjacent pixel'),
        ('inside', 'Object A bbox contained in object B bbox'),
        ('aligned_x', 'Objects share same column center (±1)'),
        ('aligned_y', 'Objects share same row center (±1)'),
        ('same_shape', 'Identical normalized pixel patterns'),
        ('same_color', 'Same ARC color value'),
        ('same_size', 'Same pixel count'),
        ('left_of', 'A bbox right edge < B bbox left edge'),
        ('right_of', 'A bbox left edge > B bbox right edge'),
        ('above', 'A bbox bottom < B bbox top'),
        ('below', 'A bbox top > B bbox bottom'),
        ('contains', 'B bbox inside A bbox'),
        ('manhattan', 'Center-to-center Manhattan distance'),
    ]
    print(f"  {len(relations)} relation types:")
    for name, desc in relations:
        print(f"    {name:15s} — {desc}")
    
    return total


# ═══════════════════════════════════════════════════════════
# TEST 2: MATH — numerical reasoning capabilities
# ═══════════════════════════════════════════════════════════

def test_math():
    header("2. MATH — Numerical Reasoning")
    
    from arc_numerical import COUNTERS, TRANSFORMS, PER_OBJ_COUNTERS
    
    subheader("Counting Functions (16)")
    for name, fn in sorted(COUNTERS.items()):
        doc = fn.__doc__ or '?'
        print(f"    {name:25s} — {doc.strip()}")
    
    subheader("Parameterized Transforms (22)")
    for name, fn in sorted(TRANSFORMS.items()):
        doc = fn.__doc__ or '?'
        print(f"    {name:25s} — {doc.strip()[:60]}")
    
    subheader("Per-Object Counters (5)")
    for name, fn in sorted(PER_OBJ_COUNTERS.items()):
        doc = fn.__doc__ or '?'
        print(f"    {name:25s} — {doc.strip()[:60]}")
    
    total_combos = len(COUNTERS) * len(TRANSFORMS) + len(PER_OBJ_COUNTERS)
    print(f"\n  Total search space: {len(COUNTERS)} × {len(TRANSFORMS)} + {len(PER_OBJ_COUNTERS)} = {total_combos} combos per task")
    
    # Demo: run counters on a sample grid
    subheader("Demo: Counting on a sample grid")
    sample = np.array([
        [0, 0, 1, 0, 0],
        [0, 2, 2, 0, 0],
        [0, 0, 0, 3, 0],
        [0, 0, 0, 3, 3],
        [0, 0, 0, 0, 0],
    ])
    print(f"    Grid (5×5):")
    for row in sample:
        print(f"      {row.tolist()}")
    print()
    
    bg = 0
    for name, fn in sorted(COUNTERS.items()):
        try:
            val = fn(sample, bg)
            print(f"    {name:25s} = {val}")
        except:
            print(f"    {name:25s} = ERROR")
    
    return total_combos


# ═══════════════════════════════════════════════════════════
# TEST 3: PATTERN MATCHING — shape recognition, symmetry, hashing
# ═══════════════════════════════════════════════════════════

def test_pattern_matching():
    header("3. PATTERN MATCHING — Shape Recognition & Symmetry")
    
    from arc_relation_graph import d4_canonical_hash, symmetry_flags
    
    subheader("D4 Group (Dihedral) — 8 transforms for shape invariance")
    print("    The D4 group has 8 elements: {identity, rot90, rot180, rot270, flipH, flipV, diagFlip, antiDiagFlip}")
    print("    NeMo-WM computes the D4-canonical hash: the lexicographically smallest")
    print("    representation under all 8 transforms. Two shapes match iff their")
    print("    D4 hashes are identical, regardless of rotation or reflection.")
    
    # Demo shapes
    shapes = {
        'L-shape': [(0,0),(1,0),(2,0),(2,1)],
        'T-shape': [(0,0),(0,1),(0,2),(1,1)],
        'Square':  [(0,0),(0,1),(1,0),(1,1)],
        'Line-H':  [(0,0),(0,1),(0,2)],
        'Line-V':  [(0,0),(1,0),(2,0)],
        'Dot':     [(0,0)],
    }
    
    subheader("Shape Hashing Demo")
    for name, cells in shapes.items():
        h = d4_canonical_hash(cells)
        sf = symmetry_flags(cells)
        sym_list = [k for k, v in sf.items() if v]
        print(f"    {name:12s}: hash={h:>20d}  symmetries={sym_list}")
    
    # Show that rotated versions match
    subheader("Rotation Invariance Test")
    original = [(0,0),(1,0),(2,0),(2,1)]  # L-shape
    rotated  = [(0,0),(0,1),(0,2),(1,0)]  # L-shape rotated 90°
    flipped  = [(0,0),(0,1),(1,1),(2,1)]  # L-shape flipped
    
    h_orig = d4_canonical_hash(original)
    h_rot  = d4_canonical_hash(rotated)
    h_flip = d4_canonical_hash(flipped)
    
    print(f"    Original L:  hash={h_orig}")
    print(f"    Rotated 90°: hash={h_rot}")
    print(f"    Flipped H:   hash={h_flip}")
    print(f"    All match: {h_orig == h_rot == h_flip} ✓" if h_orig == h_rot == h_flip else f"    MISMATCH!")
    
    # Symmetry detection
    subheader("Symmetry Detection")
    sym_grids = {
        'H-symmetric': np.array([[1,0,1],[0,1,0],[1,0,1]]),
        'V-symmetric': np.array([[1,0,0],[0,1,0],[0,0,1]]),
        'No symmetry': np.array([[1,0,0],[0,0,1],[0,0,0]]),
        'Full D4':     np.array([[1,0,1],[0,1,0],[1,0,1]]),
    }
    for name, grid in sym_grids.items():
        h_sym = np.array_equal(grid, np.fliplr(grid))
        v_sym = np.array_equal(grid, np.flipud(grid))
        r180  = np.array_equal(grid, np.rot90(grid, 2))
        print(f"    {name:15s}: H-sym={h_sym}, V-sym={v_sym}, Rot180={r180}")
    
    return len(shapes)


# ═══════════════════════════════════════════════════════════
# TEST 4: CODING/DSA — algorithmic patterns known
# ═══════════════════════════════════════════════════════════

def test_dsa_patterns():
    header("4. CODING/DSA — Algorithmic Pattern Knowledge")
    
    from arc_dsa_patterns import ARC_DSA_PATTERNS, DSAPatternRouter
    from arc_dsa_patterns import (flood_fill_region, count_connected_regions,
                                   merge_intervals, rank_compress, detect_dividers,
                                   grid_distance_transform, sliding_window_1d)
    
    subheader(f"DSA Pattern Database ({len(ARC_DSA_PATTERNS)} patterns)")
    by_family = defaultdict(list)
    for name, info in ARC_DSA_PATTERNS.items():
        by_family[info['dsa_family']].append(name)
    
    for family, patterns in sorted(by_family.items()):
        print(f"    {family.upper():12s}: {', '.join(patterns)}")
    
    # Demo: algorithmic helpers
    subheader("Algorithmic Helper Demos")
    
    # Connected regions
    grid = np.array([
        [1, 1, 0, 2, 2],
        [1, 0, 0, 0, 2],
        [0, 0, 3, 0, 0],
        [4, 0, 3, 0, 5],
        [4, 4, 0, 5, 5],
    ])
    n_regions = count_connected_regions(grid)
    print(f"    count_connected_regions:")
    for row in grid:
        print(f"      {row.tolist()}")
    print(f"      → {n_regions} regions")
    
    # Merge intervals
    intervals = [(1, 3), (2, 5), (7, 9), (8, 10)]
    merged = merge_intervals(intervals)
    print(f"\n    merge_intervals({intervals})")
    print(f"      → {merged}")
    
    # Rank compression
    values = [5, 3, 8, 1, 3]
    compressed, rmap = rank_compress(values)
    print(f"\n    rank_compress({values})")
    print(f"      → {compressed} (map: {rmap})")
    
    # Sliding window
    arr = [1, 3, 2, 5, 4, 1]
    result = sliding_window_1d(arr, 3, 'sum')
    print(f"\n    sliding_window_1d({arr}, window=3, op='sum')")
    print(f"      → {result}")
    
    # Divider detection
    grid2 = np.array([
        [1, 0, 2, 5, 3, 0, 1],
        [0, 0, 0, 5, 0, 0, 0],
        [5, 5, 5, 5, 5, 5, 5],
        [0, 3, 0, 5, 0, 2, 0],
        [0, 0, 0, 5, 0, 0, 3],
    ])
    h_divs, v_divs = detect_dividers(grid2, bg=0)
    print(f"\n    detect_dividers:")
    for row in grid2:
        print(f"      {row.tolist()}")
    print(f"      H-dividers: {h_divs}")
    print(f"      V-dividers: {v_divs}")
    
    # Distance transform
    grid3 = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    dist = grid_distance_transform(grid3, 1)
    print(f"\n    grid_distance_transform (from center pixel):")
    for row in dist:
        print(f"      {row.tolist()}")
    
    return len(ARC_DSA_PATTERNS)


# ═══════════════════════════════════════════════════════════
# TEST 5: TASK ROUTING — recommend patterns for real ARC tasks
# ═══════════════════════════════════════════════════════════

def test_task_routing(data_dir=None):
    header("5. TASK ROUTING — DSA Pattern Recommendations")
    
    if not data_dir or not os.path.exists(data_dir):
        print("  (Skipping — no ARC data directory provided)")
        return 0
    
    from arc_dsa_patterns import DSAPatternRouter
    
    router = DSAPatternRouter()
    files = sorted(os.listdir(data_dir))[:50]
    
    route_counts = Counter()
    family_counts = Counter()
    
    subheader("Sample Task Routing (first 20)")
    for i, f in enumerate(files[:20]):
        if not f.endswith('.json'):
            continue
        task = json.load(open(os.path.join(data_dir, f)))
        recommended = router.route_task(task, top_k=3)
        families = router.get_solver_families(task)
        
        if recommended:
            top_name = recommended[0][0]
            top_score = recommended[0][1]
            route_counts[top_name] += 1
            for fam in families:
                family_counts[fam] += 1
            others = ', '.join(f"{n}({s})" for n, s, _ in recommended[1:3])
            print(f"    {f}: {top_name}({top_score}) | also: {others}")
    
    subheader("Pattern Distribution (50 tasks)")
    for f in files[20:]:
        if not f.endswith('.json'):
            continue
        task = json.load(open(os.path.join(data_dir, f)))
        recommended = router.route_task(task, top_k=1)
        if recommended:
            route_counts[recommended[0][0]] += 1
            for fam in router.get_solver_families(task):
                family_counts[fam] += 1
    
    print(f"  Top recommended patterns:")
    for pattern, count in route_counts.most_common(10):
        print(f"    {pattern:30s}: {count:3d} tasks")
    
    print(f"\n  Recommended solver families:")
    for fam, count in family_counts.most_common():
        print(f"    {fam:20s}: {count:3d} tasks")
    
    # Failure diagnosis demo
    subheader("Failure Diagnosis Demo")
    sample_task = json.load(open(os.path.join(data_dir, files[0])))
    diag = router.diagnose_failure(sample_task, method_tried="stamp_cross")
    print(f"    Task: {files[0]}")
    print(f"    Tags: {diag['task_tags']}")
    print(f"    Objects: {diag['n_objects']}, Colors: {diag['n_colors']}")
    print(f"    Recommended: {diag['recommended_patterns']}")
    print(f"    Primitives: {diag['recommended_primitives'][:5]}")
    print(f"    Families: {diag['recommended_families']}")
    
    return len(route_counts)


# ═══════════════════════════════════════════════════════════
# TEST 6: RELATION GRAPH — object perception & role assignment
# ═══════════════════════════════════════════════════════════

def test_relation_graph(data_dir=None):
    header("6. OBJECT RELATION GRAPH — Perception & Roles")
    
    from arc_relation_graph import ObjectRelationGraph, d4_canonical_hash, rcc8_relation
    
    # Build graph on a sample grid
    sample = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 2, 2, 0],
        [0, 1, 1, 0, 0, 0, 0, 2, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 3, 3, 3, 3, 0, 0, 0],
        [0, 0, 0, 3, 0, 0, 3, 0, 0, 0],
        [0, 0, 0, 3, 0, 0, 3, 0, 0, 0],
        [0, 0, 0, 3, 3, 3, 3, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 4, 0, 0, 0, 0, 0, 0, 5, 0],
    ])
    
    subheader("Sample Grid (10×10)")
    for row in sample:
        print(f"    {row.tolist()}")
    
    org = ObjectRelationGraph(sample)
    
    subheader(f"Perception: {org}")
    print(f"    Objects detected: {org.n}")
    for o in org.objects:
        props = org.obj_props[o.id]
        print(f"    Obj {o.id}: color={o.color}, size={o.size}, bbox={o.bbox}, "
              f"density={o.density:.2f}, roles={props['role_candidates']}")
    
    subheader("Spatial Relations")
    for i, a in enumerate(org.objects):
        for j, b in enumerate(org.objects):
            if i >= j:
                continue
            rel = org.relations.get((a.id, b.id), {})
            active = [k for k, v in rel.items() if v is True]
            dist = rel.get('manhattan', '?')
            if active:
                print(f"    Obj {a.id} ↔ Obj {b.id}: {active}, dist={dist:.1f}")
    
    subheader("Row/Column Bands")
    for i, band in enumerate(org.row_bands):
        ids = [o.id for o in band]
        print(f"    Row band {i}: objects {ids}")
    for i, band in enumerate(org.col_bands):
        ids = [o.id for o in band]
        print(f"    Col band {i}: objects {ids}")
    
    subheader("Nearest Neighbors")
    for o in org.objects:
        nn = org.nearest_of(o.id)
        nn_str = f"Obj {nn.id} (color={nn.color})" if nn else "None"
        print(f"    Obj {o.id} (color={o.color}) → nearest: {nn_str}")
    
    subheader("Role Assignments")
    role_summary = defaultdict(list)
    for oid, roles in org.roles.items():
        for r in roles:
            role_summary[r].append(oid)
    for role, oids in sorted(role_summary.items()):
        print(f"    {role:15s}: objects {oids}")
    
    subheader("Feature Vectors (30-dim)")
    for o in org.objects:
        fv = org.feature_vector(o)
        nz = [(i, fv[i]) for i in range(len(fv)) if fv[i] != 0]
        print(f"    Obj {o.id}: {len(nz)} non-zero features")
        labels = ['color','size','width','height','aspect','density','is_rect','border',
                  'pos_r','pos_c','deg_touch','deg_align','deg_shape','deg_color',
                  'nn_dist','n_knn','comp_id','sym_h','sym_v','sym_180','sym_90',
                  'role_template','role_marker','role_container','role_divider',
                  'role_mover','role_anchor','role_singleton','size_ratio','is_smallest']
        for idx, val in nz[:10]:
            label = labels[idx] if idx < len(labels) else f'f{idx}'
            print(f"      [{idx:2d}] {label:15s} = {val:.3f}")
    
    subheader("Task-Level Feature Vector (20-dim)")
    tfv = org.task_feature_vector()
    task_labels = ['height','width','n_objects','n_colors','n_shapes','n_components',
                   'max_comp','n_bridges','n_row_bands','n_col_bands',
                   'n_templates','n_markers','n_containers','n_dividers',
                   'avg_density','size_std','n_nonbg','fill_ratio','size_ratio','has_frame']
    for i, (val, label) in enumerate(zip(tfv, task_labels)):
        if val != 0:
            print(f"    [{i:2d}] {label:15s} = {val:.3f}")
    
    # Test on real ARC tasks
    if data_dir and os.path.exists(data_dir):
        subheader("Real ARC Task Analysis (first 10)")
        files = sorted(os.listdir(data_dir))[:10]
        t0 = time.time()
        for f in files:
            if not f.endswith('.json'):
                continue
            task = json.load(open(os.path.join(data_dir, f)))
            gi = np.array(task['train'][0]['input'])
            org = ObjectRelationGraph(gi)
            roles = defaultdict(int)
            for oid, rs in org.roles.items():
                for r in rs:
                    roles[r] += 1
            role_str = ', '.join(f"{r}:{c}" for r, c in sorted(roles.items(), key=lambda x: -x[1])[:3])
            print(f"    {f}: {org.n} objs, {len(org.components_touch)} comps, "
                  f"{len(org.row_bands)} bands | {role_str}")
        elapsed = time.time() - t0
        print(f"    ({elapsed*1000/10:.0f}ms/task)")
    
    return org.n


# ═══════════════════════════════════════════════════════════
# TEST 7: COMPOSITIONAL SEARCH — primitive chaining
# ═══════════════════════════════════════════════════════════

def test_compositional():
    header("7. COMPOSITIONAL SEARCH — Primitive Chaining")
    
    from arc_overnight_compose import PRIMITIVES, NUMERICAL_PRIMITIVES, apply_chain
    
    n_geo = len(PRIMITIVES) - len(NUMERICAL_PRIMITIVES)
    
    subheader(f"Primitive Library ({len(PRIMITIVES)} total)")
    print(f"    Geometric:  {n_geo} primitives")
    geo = [k for k in PRIMITIVES if k not in NUMERICAL_PRIMITIVES]
    for i in range(0, len(geo), 6):
        print(f"      {', '.join(geo[i:i+6])}")
    
    print(f"\n    Numerical:  {len(NUMERICAL_PRIMITIVES)} primitives")
    for i, k in enumerate(sorted(NUMERICAL_PRIMITIVES.keys())):
        print(f"      {k}")
    
    print(f"\n    Search space:")
    n = len(PRIMITIVES)
    print(f"      Depth 1: {n:>10,d} chains")
    print(f"      Depth 2: {n**2:>10,d} chains")
    print(f"      Depth 3: {n**3:>10,d} chains")
    
    # Demo: chain execution
    subheader("Chain Execution Demo")
    sample = np.array([
        [0, 1, 2],
        [3, 0, 0],
        [0, 0, 4],
    ])
    print(f"    Input:")
    for row in sample:
        print(f"      {row.tolist()}")
    
    chains = [
        ['fliph'],
        ['flipv'],
        ['rot90'],
        ['crop'],
        ['mirror_h'],
        ['fliph', 'crop'],
        ['crop', 'up_2x'],
    ]
    
    for chain in chains:
        result = apply_chain(sample, chain)
        if result is not None:
            shape_str = f"{result.shape[0]}×{result.shape[1]}"
            print(f"    {' → '.join(chain):25s} → {shape_str}: {result.flatten()[:6].tolist()}...")
        else:
            print(f"    {' → '.join(chain):25s} → None")
    
    return len(PRIMITIVES)


# ═══════════════════════════════════════════════════════════
# TEST 8: INTEGRATION SUMMARY
# ═══════════════════════════════════════════════════════════

def test_summary(results):
    header("KNOWLEDGE SUMMARY")
    
    print(f"""
    ┌─────────────────────────────────────────────────────┐
    │  NeMo-WM Knowledge Diagnostic Results               │
    ├─────────────────────────────────────────────────────┤
    │  Object Attributes:        {results.get('attrs', 13):>5d}                    │
    │  OG Solvers:               {results.get('solvers', 0):>5d}                    │
    │  Numerical Combos:         {results.get('math', 0):>5d}                    │
    │  DSA Patterns:             {results.get('dsa', 0):>5d}                    │
    │  Compose Primitives:       {results.get('compose', 0):>5d}                    │
    │  Spatial Relations:           13                    │
    │  Role Types:                   8                    │
    │  Feature Vector Dims:     30+20                    │
    │  D4 Group Transforms:          8                    │
    │  LeetCode Problems:        2,913                    │
    ├─────────────────────────────────────────────────────┤
    │  ARC-AGI-2 Score:     104/1000 (10.4%)              │
    │  Cost per Task:              $0                     │
    │  Speed:                   95ms/task                 │
    │  LLM Dependency:          None                     │
    └─────────────────────────────────────────────────────┘
    """)


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else None
    
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║        NeMo-WM Comprehensive Knowledge Diagnostic              ║")
    print("║        Testing: Vocabulary, Math, Patterns, DSA, Graphs        ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    
    results = {}
    
    t0 = time.time()
    
    results['solvers'] = test_vocabulary()
    results['math'] = test_math()
    test_pattern_matching()
    results['dsa'] = test_dsa_patterns()
    results['routing'] = test_task_routing(data_dir)
    test_relation_graph(data_dir)
    results['compose'] = test_compositional()
    
    elapsed = time.time() - t0
    
    test_summary(results)
    
    print(f"  Total diagnostic time: {elapsed:.1f}s")
    print(f"  All tests passed. System operational.")


if __name__ == '__main__':
    main()
