"""
NeMo-WM DSA Pattern Integration
=================================
Maps LeetCode/DSA algorithmic patterns to ARC solver families.
Provides:
  1. Pattern-based task routing (match task fingerprint → solver family)
  2. Algorithmic primitives enriching the ObjectRelationGraph
  3. Reusable helpers extracted from DSA patterns
  4. Diagnostic triggers for failure analysis

Usage:
    from arc_dsa_patterns import DSAPatternRouter, get_arc_relevant_patterns
    
    router = DSAPatternRouter()
    recommended = router.route_task(task)
    # → ['connected_components', 'flood_fill', 'interval_overlap']
"""
import json
import os
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path


# ═══════════════════════════════════════════════════════════
# ARC-RELEVANT DSA PATTERNS — mapped from LeetCode to grid reasoning
# ═══════════════════════════════════════════════════════════

ARC_DSA_PATTERNS = {
    # ─── GRAPH PATTERNS ──────────────────────────────────
    'connected_components': {
        'dsa_family': 'graph',
        'arc_use': 'Object extraction, touch groups, region labeling',
        'algorithm': 'BFS/DFS flood fill on 4/8-connected pixels',
        'solver_families': ['stamp', 'relational', 'movement'],
        'primitives': ['extract_objects', 'flood_fill', 'label_regions'],
        'triggers': ['multiple objects', 'separate regions', 'groups', 'clusters'],
        'complexity': 'O(H*W)',
    },
    'shortest_path': {
        'dsa_family': 'graph',
        'arc_use': 'Bridge detection, connecting objects, path drawing',
        'algorithm': 'BFS on adjacency graph between objects',
        'solver_families': ['relational', 'movement'],
        'primitives': ['shortest_path', 'connect_objects', 'draw_line'],
        'triggers': ['connect', 'bridge', 'path between', 'link'],
        'complexity': 'O(V+E)',
    },
    'bridges_articulation': {
        'dsa_family': 'graph',
        'arc_use': 'Finding critical objects that connect groups',
        'algorithm': 'Tarjan bridge detection on touch graph',
        'solver_families': ['relational'],
        'primitives': ['find_bridges', 'is_articulation_point'],
        'triggers': ['critical', 'bridge', 'connecting'],
        'complexity': 'O(V+E)',
    },
    'bipartite_matching': {
        'dsa_family': 'graph',
        'arc_use': 'Object correspondence between input and output',
        'algorithm': 'Hungarian algorithm on object similarity matrix',
        'solver_families': ['compositional', 'relational'],
        'primitives': ['match_objects', 'hungarian_matching'],
        'triggers': ['pairs', 'matching', 'correspondence'],
        'complexity': 'O(N^3)',
    },
    'topological_sort': {
        'dsa_family': 'graph',
        'arc_use': 'Ordering objects by containment/dependency',
        'algorithm': 'DFS-based topological sort on containment graph',
        'solver_families': ['compositional'],
        'primitives': ['containment_order', 'dependency_sort'],
        'triggers': ['ordering', 'layers', 'nesting'],
        'complexity': 'O(V+E)',
    },
    
    # ─── GRID / GEOMETRY PATTERNS ─────────────────────────
    'flood_fill': {
        'dsa_family': 'grid',
        'arc_use': 'Region filling, enclosure detection, object extraction',
        'algorithm': 'BFS/DFS from seed pixel, fill with color',
        'solver_families': ['stamp', 'symmetry', 'relational'],
        'primitives': ['fill_enclosed', 'fill_region', 'mark_reachable'],
        'triggers': ['fill', 'enclosed', 'inside', 'region'],
        'complexity': 'O(H*W)',
    },
    'island_counting': {
        'dsa_family': 'grid',
        'arc_use': 'Counting objects, connected components on grid',
        'algorithm': 'BFS/DFS counting distinct connected regions',
        'solver_families': ['numerical', 'relational'],
        'primitives': ['count_objects', 'count_regions', 'count_colors'],
        'triggers': ['count', 'how many', 'number of'],
        'complexity': 'O(H*W)',
    },
    'perimeter_detection': {
        'dsa_family': 'grid',
        'arc_use': 'Edge detection, border drawing, outline extraction',
        'algorithm': 'For each cell, count non-same neighbors',
        'solver_families': ['stamp', 'relational'],
        'primitives': ['object_outline', 'border_pixels', 'edge_detect'],
        'triggers': ['border', 'outline', 'edge', 'perimeter'],
        'complexity': 'O(H*W)',
    },
    'convex_hull': {
        'dsa_family': 'geometry',
        'arc_use': 'Bounding shape, enclosure, spatial extent',
        'algorithm': 'Andrew monotone chain on integer coords',
        'solver_families': ['relational', 'symmetry'],
        'primitives': ['convex_hull', 'is_convex', 'fill_convex'],
        'triggers': ['boundary', 'enclosing', 'convex'],
        'complexity': 'O(N log N)',
    },
    'interval_overlap': {
        'dsa_family': 'geometry',
        'arc_use': 'Row/column band detection, alignment, gap analysis',
        'algorithm': 'Allen interval algebra on bbox projections',
        'solver_families': ['compositional', 'relational', 'stamp'],
        'primitives': ['row_bands', 'col_bands', 'overlap_1d', 'gap_features'],
        'triggers': ['aligned', 'same row', 'same column', 'gap', 'between'],
        'complexity': 'O(N^2)',
    },
    'line_sweep': {
        'dsa_family': 'geometry',
        'arc_use': 'Scanning rows/cols for patterns, divider detection',
        'algorithm': 'Process events sorted by x or y coordinate',
        'solver_families': ['compositional', 'stamp'],
        'primitives': ['scan_rows', 'scan_cols', 'find_dividers'],
        'triggers': ['divider', 'partition', 'split', 'line'],
        'complexity': 'O(N log N)',
    },
    
    # ─── ORDERING / SORTING PATTERNS ──────────────────────
    'sort_by_key': {
        'dsa_family': 'ordering',
        'arc_use': 'Object ranking, spatial ordering, size-based recolor',
        'algorithm': 'Sort objects by (size, color, position, etc.)',
        'solver_families': ['numerical', 'relational'],
        'primitives': ['sort_objects', 'rank_by_size', 'rank_by_position'],
        'triggers': ['order', 'sort', 'rank', 'largest', 'smallest'],
        'complexity': 'O(N log N)',
    },
    'interval_merge': {
        'dsa_family': 'ordering',
        'arc_use': 'Row/col band merging, gap collapse, region unification',
        'algorithm': 'Sort intervals by start, merge overlapping',
        'solver_families': ['compositional', 'relational'],
        'primitives': ['merge_row_bands', 'merge_regions', 'collapse_gaps'],
        'triggers': ['merge', 'combine', 'unify', 'connect'],
        'complexity': 'O(N log N)',
    },
    'rank_compression': {
        'dsa_family': 'ordering',
        'arc_use': 'Color mapping by rank, canonical encoding',
        'algorithm': 'Map values to their sorted rank (0-indexed)',
        'solver_families': ['numerical'],
        'primitives': ['color_rank', 'normalize_colors', 'canonical_order'],
        'triggers': ['rank', 'compress', 'normalize', 'canonical'],
        'complexity': 'O(N log N)',
    },
    
    # ─── PATTERN MATCHING / HASHING ───────────────────────
    'shape_hashing': {
        'dsa_family': 'hashing',
        'arc_use': 'Same-shape detection, template matching, D4 invariance',
        'algorithm': 'Normalize under D4 group, hash canonical form',
        'solver_families': ['stamp', 'relational', 'compositional'],
        'primitives': ['d4_hash', 'shape_match', 'find_template'],
        'triggers': ['same shape', 'repeated', 'pattern', 'template'],
        'complexity': 'O(N)',
    },
    'frequency_map': {
        'dsa_family': 'hashing',
        'arc_use': 'Color counting, majority/minority detection, histogram',
        'algorithm': 'Counter/histogram over colors or object properties',
        'solver_families': ['numerical', 'relational'],
        'primitives': ['color_histogram', 'majority_color', 'frequency_rank'],
        'triggers': ['most common', 'frequent', 'count', 'majority'],
        'complexity': 'O(N)',
    },
    'deduplication': {
        'dsa_family': 'hashing',
        'arc_use': 'Remove duplicate rows/cols, unique objects, canonical forms',
        'algorithm': 'Hash-based dedup with canonical representation',
        'solver_families': ['compositional'],
        'primitives': ['unique_rows', 'unique_cols', 'unique_objects'],
        'triggers': ['unique', 'distinct', 'remove duplicates', 'dedup'],
        'complexity': 'O(N)',
    },
    
    # ─── DYNAMIC PROGRAMMING / SEQUENCES ──────────────────
    'subsequence': {
        'dsa_family': 'dp',
        'arc_use': 'Pattern continuation, sequence completion',
        'algorithm': 'LIS/LCS-style DP on color or position sequences',
        'solver_families': ['numerical', 'compositional'],
        'primitives': ['find_sequence', 'continue_pattern', 'complete_series'],
        'triggers': ['sequence', 'continue', 'next', 'pattern'],
        'complexity': 'O(N^2)',
    },
    'grid_dp': {
        'dsa_family': 'dp',
        'arc_use': 'Minimum path, maximal rectangle, optimal subgrid',
        'algorithm': 'DP on grid cells with directional transitions',
        'solver_families': ['movement', 'compositional'],
        'primitives': ['min_path_sum', 'max_rectangle', 'optimal_region'],
        'triggers': ['path', 'optimal', 'minimum', 'maximum'],
        'complexity': 'O(H*W)',
    },
    
    # ─── SIMULATION / TRANSFORM PATTERNS ──────────────────
    'gravity_simulation': {
        'dsa_family': 'simulation',
        'arc_use': 'Object falling, sliding, stacking with collision',
        'algorithm': 'Move objects step-by-step until blocked',
        'solver_families': ['movement'],
        'primitives': ['gravity_down', 'gravity_right', 'slide_until_blocked'],
        'triggers': ['fall', 'drop', 'slide', 'gravity', 'push'],
        'complexity': 'O(N*H)',
    },
    'cellular_automaton': {
        'dsa_family': 'simulation',
        'arc_use': 'Context-dependent recoloring, neighbor-based rules',
        'algorithm': 'Apply local rule based on neighbor configuration',
        'solver_families': ['relational', 'stamp'],
        'primitives': ['neighbor_count', 'context_recolor', 'local_rule'],
        'triggers': ['neighbors', 'surrounding', 'adjacent', 'touching'],
        'complexity': 'O(H*W)',
    },
    'symmetry_group': {
        'dsa_family': 'math',
        'arc_use': 'Symmetry completion, reflection, rotation detection',
        'algorithm': 'D4 group operations, wallpaper group detection',
        'solver_families': ['symmetry', 'tiling'],
        'primitives': ['d4_symmetry', 'complete_symmetry', 'detect_symmetry'],
        'triggers': ['symmetric', 'mirror', 'reflect', 'rotate', 'complete'],
        'complexity': 'O(H*W)',
    },
    'modular_arithmetic': {
        'dsa_family': 'math',
        'arc_use': 'Periodic patterns, checkerboard, stripe generation',
        'algorithm': 'Color = f(r%N, c%M) for periodic patterns',
        'solver_families': ['numerical'],
        'primitives': ['periodic_color', 'checkerboard', 'stripe_pattern'],
        'triggers': ['periodic', 'repeating', 'alternating', 'checkerboard'],
        'complexity': 'O(H*W)',
    },
}


# ═══════════════════════════════════════════════════════════
# TASK FINGERPRINTING — extract features that match DSA triggers
# ═══════════════════════════════════════════════════════════

def fingerprint_task(task):
    """
    Extract a structural fingerprint from an ARC task.
    Returns a dict of boolean/numeric features that can match DSA triggers.
    """
    from arc_object_graph import extract_objects
    
    pairs = task['train']
    gi = np.array(pairs[0]['input'])
    go = np.array(pairs[0]['output'])
    ih, iw = gi.shape
    oh, ow = go.shape
    bg = int(np.argmax(np.bincount(gi.flatten())))
    
    try:
        objs, _ = extract_objects(gi)
    except:
        objs = []
    
    fp = {
        # Size change
        'same_size': gi.shape == go.shape,
        'grows': oh > ih or ow > iw,
        'shrinks': oh < ih or ow < iw,
        'scale_ratio': (oh / max(ih, 1), ow / max(iw, 1)),
        
        # Object properties
        'n_objects': len(objs),
        'n_colors': len(set(int(v) for v in gi.flatten()) - {bg}),
        'has_single_pixel_objects': any(o.size == 1 for o in objs),
        'all_same_shape': len(set(o.shape_hash for o in objs)) == 1 if objs else False,
        'all_same_color': len(set(o.color for o in objs)) == 1 if objs else False,
        'has_large_object': any(o.size > ih * iw * 0.3 for o in objs),
        'has_frame': any(o.density < 0.5 and o.size > 8 for o in objs),
        
        # Spatial properties
        'has_divider_h': any(
            all(gi[r, c] == gi[r, 0] and gi[r, 0] != bg for c in range(iw))
            for r in range(ih)
        ),
        'has_divider_v': any(
            all(gi[r, c] == gi[0, c] and gi[0, c] != bg for r in range(ih))
            for c in range(iw)
        ),
        
        # Diff properties
        'n_changed_pixels': int(np.sum(gi != go)) if gi.shape == go.shape else -1,
        'pct_changed': float(np.sum(gi != go) / gi.size * 100) if gi.shape == go.shape else -1,
        
        # Symmetry
        'output_h_sym': bool(np.array_equal(go, np.fliplr(go))),
        'output_v_sym': bool(np.array_equal(go, np.flipud(go))),
        
        # Structural tags (for trigger matching)
        'tags': set(),
    }
    
    # Add semantic tags
    if fp['n_objects'] >= 2:
        fp['tags'].add('multiple objects')
    if fp['n_objects'] >= 5:
        fp['tags'].add('many objects')
    if fp['has_single_pixel_objects']:
        fp['tags'].add('markers')
    if fp['has_frame']:
        fp['tags'].add('frame')
        fp['tags'].add('enclosed')
    if fp['has_divider_h'] or fp['has_divider_v']:
        fp['tags'].add('divider')
        fp['tags'].add('partition')
    if fp['all_same_shape'] and fp['n_objects'] >= 3:
        fp['tags'].add('repeated')
        fp['tags'].add('template')
    if fp['output_h_sym'] or fp['output_v_sym']:
        fp['tags'].add('symmetric')
    if fp['grows']:
        fp['tags'].add('grows')
        sr = fp['scale_ratio']
        if sr[0] == sr[1] and sr[0] == int(sr[0]):
            fp['tags'].add(f'scale_{int(sr[0])}x')
    if fp['shrinks']:
        fp['tags'].add('shrinks')
        fp['tags'].add('extract')
    if fp['same_size'] and 0 < fp['pct_changed'] < 30:
        fp['tags'].add('small diff')
        fp['tags'].add('recolor')
    if fp['n_colors'] >= 4:
        fp['tags'].add('many colors')
    
    return fp


# ═══════════════════════════════════════════════════════════
# DSA PATTERN ROUTER — recommend patterns for a task
# ═══════════════════════════════════════════════════════════

class DSAPatternRouter:
    """
    Routes ARC tasks to relevant DSA patterns based on task fingerprint.
    """
    
    def __init__(self, pattern_cards_path=None):
        self.patterns = ARC_DSA_PATTERNS
        self.leetcode_cards = []
        
        # Load LeetCode pattern cards if available
        if pattern_cards_path and os.path.exists(pattern_cards_path):
            with open(pattern_cards_path) as f:
                self.leetcode_cards = json.load(f)
    
    def route_task(self, task, top_k=5):
        """
        Given an ARC task, recommend the most relevant DSA patterns.
        Returns list of (pattern_name, score, pattern_info).
        """
        fp = fingerprint_task(task)
        scores = {}
        
        for pname, pinfo in self.patterns.items():
            score = 0
            
            # Match triggers against task tags
            for trigger in pinfo.get('triggers', []):
                trigger_lower = trigger.lower()
                for tag in fp['tags']:
                    if trigger_lower in tag.lower() or tag.lower() in trigger_lower:
                        score += 2
            
            # Match based on structural features
            if 'connected_components' in pname and fp['n_objects'] >= 2:
                score += 1
            if 'flood_fill' in pname and fp.get('has_frame'):
                score += 3
            if 'interval_overlap' in pname and (fp.get('has_divider_h') or fp.get('has_divider_v')):
                score += 3
            if 'gravity' in pname and fp['same_size'] and fp['n_objects'] >= 2:
                score += 1
            if 'symmetry' in pname and (fp['output_h_sym'] or fp['output_v_sym']):
                score += 3
            if 'shape_hashing' in pname and fp.get('all_same_shape'):
                score += 2
            if 'frequency' in pname and fp['n_colors'] >= 3:
                score += 1
            if 'sort_by_key' in pname and fp['n_objects'] >= 3:
                score += 1
            if 'island_counting' in pname and fp['n_objects'] >= 2:
                score += 1
            if 'modular' in pname and fp['same_size']:
                score += 1
            if 'cellular' in pname and fp['same_size'] and 0 < fp.get('pct_changed', 100) < 50:
                score += 2
            
            if score > 0:
                scores[pname] = (score, pinfo)
        
        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: -x[1][0])
        return [(name, score, info) for name, (score, info) in ranked[:top_k]]
    
    def get_solver_families(self, task):
        """Get recommended solver families for a task."""
        recommended = self.route_task(task)
        families = set()
        for _, _, info in recommended:
            families.update(info.get('solver_families', []))
        return sorted(families)
    
    def get_primitives(self, task):
        """Get recommended algorithmic primitives for a task."""
        recommended = self.route_task(task)
        primitives = []
        for _, _, info in recommended:
            primitives.extend(info.get('primitives', []))
        return list(dict.fromkeys(primitives))  # deduplicate preserving order
    
    def diagnose_failure(self, task, method_tried=None):
        """
        After a solver fails, suggest what DSA pattern might help.
        Returns diagnostic recommendations.
        """
        fp = fingerprint_task(task)
        recommended = self.route_task(task, top_k=10)
        
        diagnosis = {
            'task_tags': sorted(fp['tags']),
            'n_objects': fp['n_objects'],
            'n_colors': fp['n_colors'],
            'size_change': 'same' if fp['same_size'] else ('grow' if fp['grows'] else 'shrink'),
            'recommended_patterns': [(name, score) for name, score, _ in recommended[:5]],
            'recommended_primitives': self.get_primitives(task)[:10],
            'recommended_families': self.get_solver_families(task),
        }
        
        if method_tried:
            diagnosis['tried'] = method_tried
            diagnosis['suggestion'] = f"Try patterns: {', '.join(name for name, _, _ in recommended[:3])}"
        
        return diagnosis


# ═══════════════════════════════════════════════════════════
# ALGORITHMIC HELPERS — reusable DSA implementations for ARC
# ═══════════════════════════════════════════════════════════

def flood_fill_region(grid, start_r, start_c, target_color, fill_color):
    """BFS flood fill from a starting cell."""
    from collections import deque
    h, w = grid.shape
    if grid[start_r, start_c] != target_color:
        return grid
    out = grid.copy()
    q = deque([(start_r, start_c)])
    out[start_r, start_c] = fill_color
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w and out[nr, nc] == target_color:
                out[nr, nc] = fill_color
                q.append((nr, nc))
    return out


def count_connected_regions(grid, target_color=None):
    """Count connected regions (islands) in a grid."""
    from collections import deque
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if visited[r, c]:
                continue
            if target_color is not None and grid[r, c] != target_color:
                continue
            if grid[r, c] == 0:  # skip bg
                continue
            # BFS
            count += 1
            q = deque([(r, c)])
            visited[r, c] = True
            while q:
                cr, cc = q.popleft()
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = cr+dr, cc+dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                        if target_color is None and grid[nr, nc] == grid[cr, cc]:
                            visited[nr, nc] = True
                            q.append((nr, nc))
                        elif target_color is not None and grid[nr, nc] == target_color:
                            visited[nr, nc] = True
                            q.append((nr, nc))
    return count


def object_perimeter(cells, grid_shape):
    """Compute perimeter pixels of an object (cells touching non-object)."""
    cells_set = set(cells)
    h, w = grid_shape
    perimeter = []
    for r, c in cells:
        is_border = any(
            (r+dr, c+dc) not in cells_set
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
        )
        if is_border:
            perimeter.append((r, c))
    return perimeter


def merge_intervals(intervals):
    """Merge overlapping intervals. Input: list of (start, end)."""
    if not intervals:
        return []
    sorted_ints = sorted(intervals)
    merged = [sorted_ints[0]]
    for start, end in sorted_ints[1:]:
        if start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def rank_compress(values):
    """Map values to their sorted rank (0-indexed)."""
    sorted_unique = sorted(set(values))
    rank_map = {v: i for i, v in enumerate(sorted_unique)}
    return [rank_map[v] for v in values], rank_map


def sliding_window_1d(arr, window_size, op='sum'):
    """Apply sliding window operation on 1D array."""
    results = []
    for i in range(len(arr) - window_size + 1):
        window = arr[i:i + window_size]
        if op == 'sum':
            results.append(sum(window))
        elif op == 'max':
            results.append(max(window))
        elif op == 'min':
            results.append(min(window))
        elif op == 'majority':
            results.append(Counter(window).most_common(1)[0][0])
    return results


def grid_distance_transform(grid, target_color):
    """Compute Manhattan distance from each cell to nearest target_color cell."""
    from collections import deque
    h, w = grid.shape
    dist = np.full((h, w), h + w, dtype=int)
    q = deque()
    for r in range(h):
        for c in range(w):
            if grid[r, c] == target_color:
                dist[r, c] = 0
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w and dist[nr, nc] > dist[r, c] + 1:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def detect_dividers(grid, bg=0):
    """Find horizontal and vertical divider lines in a grid."""
    h, w = grid.shape
    h_dividers = []
    v_dividers = []
    
    for r in range(h):
        vals = set(int(grid[r, c]) for c in range(w))
        if len(vals) == 1 and int(grid[r, 0]) != bg:
            h_dividers.append((r, int(grid[r, 0])))
    
    for c in range(w):
        vals = set(int(grid[r, c]) for r in range(h))
        if len(vals) == 1 and int(grid[0, c]) != bg:
            v_dividers.append((c, int(grid[0, c])))
    
    return h_dividers, v_dividers


def voronoi_regions(grid, seed_cells, bg=0):
    """Compute discrete Voronoi regions from seed cells using BFS."""
    from collections import deque
    h, w = grid.shape
    labels = np.full((h, w), -1, dtype=int)
    q = deque()
    
    for label, (r, c) in enumerate(seed_cells):
        labels[r, c] = label
        q.append((r, c, label))
    
    while q:
        r, c, label = q.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w and labels[nr, nc] == -1:
                labels[nr, nc] = label
                q.append((nr, nc, label))
    
    return labels


# ═══════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════

def get_arc_relevant_patterns():
    """Return all ARC-relevant DSA patterns."""
    return ARC_DSA_PATTERNS


def pattern_summary():
    """Print a summary of all patterns with their ARC relevance."""
    by_family = defaultdict(list)
    for name, info in ARC_DSA_PATTERNS.items():
        by_family[info['dsa_family']].append((name, info))
    
    print(f"ARC-DSA Pattern Database: {len(ARC_DSA_PATTERNS)} patterns")
    print(f"{'='*60}")
    for family, patterns in sorted(by_family.items()):
        print(f"\n{family.upper()} ({len(patterns)} patterns)")
        for name, info in patterns:
            solvers = ', '.join(info['solver_families'])
            print(f"  {name:30s} → [{solvers}]")
            print(f"    {info['arc_use']}")


# ═══════════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    
    pattern_summary()
    
    # Test routing on ARC tasks if data available
    data_dir = sys.argv[1] if len(sys.argv) > 1 else None
    if data_dir and os.path.exists(data_dir):
        print(f"\n{'='*60}")
        print(f"Testing router on ARC tasks from {data_dir}")
        print(f"{'='*60}")
        
        router = DSAPatternRouter()
        files = sorted(os.listdir(data_dir))[:20]
        
        for f in files:
            if not f.endswith('.json'):
                continue
            task = json.load(open(os.path.join(data_dir, f)))
            recommended = router.route_task(task)
            if recommended:
                top = recommended[0]
                print(f"  {f}: {top[0]} (score={top[1]}) → {top[2]['solver_families']}")
