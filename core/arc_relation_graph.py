"""
NeMo-WM ObjectRelationGraph v2
================================
Persistent relation structure computed ONCE per grid, reusable across all solver families.

Computes:
  - Spatial relations: left/right/above/below, row/col bands, interval overlap
  - Proximity: nearest neighbor, kNN, gap features, distance matrix
  - Topology: touches, containment, enclosure, frame detection
  - Graph features: degree, connected components, bridges, paths
  - Roles: template, marker, container, divider, mover, anchor candidates
  - Feature vectors: 30-dim per-object numeric vector for learning/clustering

Usage:
    from arc_relation_graph import ObjectRelationGraph
    
    org = ObjectRelationGraph(grid)
    
    # Query relations
    templates = org.objects_with_role('template')
    markers = org.objects_with_role('marker')
    nearest = org.nearest_of(obj.id)
    
    # Get feature vector for learning
    fv = org.feature_vector(obj)
    
    # Check relations
    if org.related(a.id, b.id, 'touches'):
        ...
"""
import numpy as np
from collections import defaultdict, Counter, deque
from arc_object_graph import ArcObject, extract_objects


# ═══════════════════════════════════════════════════════════
# D4 CANONICAL SHAPE HASH — invariant under dihedral group
# ═══════════════════════════════════════════════════════════

def d4_canonical_hash(cells):
    """
    Compute shape hash invariant under the dihedral group D4 
    (4 rotations × 2 reflections = 8 transforms).
    Returns the lexicographically smallest normalized cell set.
    """
    rs = [r for r, c in cells]
    cs = [c for r, c in cells]
    min_r, min_c = min(rs), min(cs)
    normalized = [(r - min_r, c - min_c) for r, c in cells]
    
    transforms = []
    for rotation in range(4):
        # Rotate 90° rotation times
        rotated = normalized
        for _ in range(rotation):
            rotated = [(c, -r) for r, c in rotated]
        # Normalize to (0,0) origin
        rr = [r for r, c in rotated]
        rc = [c for r, c in rotated]
        mr, mc = min(rr), min(rc)
        norm = tuple(sorted((r - mr, c - mc) for r, c in rotated))
        transforms.append(norm)
        
        # Also add horizontal flip
        flipped = [(-r, c) for r, c in rotated]
        rr = [r for r, c in flipped]
        rc = [c for r, c in flipped]
        mr, mc = min(rr), min(rc)
        norm = tuple(sorted((r - mr, c - mc) for r, c in flipped))
        transforms.append(norm)
    
    return hash(min(transforms))


def symmetry_flags(cells):
    """Compute which D4 symmetries the shape possesses."""
    rs = [r for r, c in cells]
    cs = [c for r, c in cells]
    min_r, min_c = min(rs), min(cs)
    max_r, max_c = max(rs), max(cs)
    norm = set((r - min_r, c - min_c) for r, c in cells)
    h, w = max_r - min_r, max_c - min_c
    
    flags = {}
    # Horizontal flip
    flags['hflip'] = norm == set((r, w - c) for r, c in norm)
    # Vertical flip
    flags['vflip'] = norm == set((h - r, c) for r, c in norm)
    # 180° rotation
    flags['rot180'] = norm == set((h - r, w - c) for r, c in norm)
    # 90° rotation (only if square-ish)
    if h == w:
        flags['rot90'] = norm == set((c, h - r) for r, c in norm)
    else:
        flags['rot90'] = False
    
    return flags


# ═══════════════════════════════════════════════════════════
# RCC-8 SPATIAL RELATIONS (adapted for discrete grids)
# ═══════════════════════════════════════════════════════════

def rcc8_relation(a_cells, b_cells, grid_shape):
    """
    Compute RCC-8 relation between two cell sets on a discrete grid.
    Returns one of: DC, EC, PO, EQ, TPP, NTPP, TPPi, NTPPi
    """
    a_set = set(a_cells)
    b_set = set(b_cells)
    
    if a_set == b_set:
        return 'EQ'
    
    intersection = a_set & b_set
    
    if not intersection:
        # Check if externally connected (adjacent but non-overlapping)
        for r, c in a_set:
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                if (r+dr, c+dc) in b_set:
                    return 'EC'
        return 'DC'  # Disconnected
    
    # They overlap
    a_only = a_set - b_set
    b_only = b_set - a_set
    
    if not a_only and not b_only:
        return 'EQ'
    elif not a_only:
        # a ⊂ b — check tangential vs non-tangential
        h, w = grid_shape
        a_boundary = set()
        for r, c in a_set:
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if (nr, nc) not in a_set:
                    a_boundary.add((r, c))
        b_boundary = set()
        for r, c in b_set:
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if (nr, nc) not in b_set:
                    b_boundary.add((r, c))
        if a_boundary & b_boundary:
            return 'TPP'  # Tangential proper part
        return 'NTPP'  # Non-tangential proper part
    elif not b_only:
        # b ⊂ a
        b_boundary = set()
        for r, c in b_set:
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                if (r+dr, c+dc) not in b_set:
                    b_boundary.add((r, c))
        a_boundary = set()
        for r, c in a_set:
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                if (r+dr, c+dc) not in a_set:
                    a_boundary.add((r, c))
        if a_boundary & b_boundary:
            return 'TPPi'
        return 'NTPPi'
    else:
        return 'PO'  # Partial overlap


# ═══════════════════════════════════════════════════════════
# GAP FEATURES between two objects
# ═══════════════════════════════════════════════════════════

def compute_gap_features(a, b):
    """Compute spatial gap between two objects."""
    ar1, ar2, ac1, ac2 = a.bbox
    br1, br2, bc1, bc2 = b.bbox
    
    # Horizontal gap (positive = separated, negative = overlapping)
    h_gap = max(bc1 - ac2 - 1, ac1 - bc2 - 1, 0)
    # Vertical gap
    v_gap = max(br1 - ar2 - 1, ar1 - br2 - 1, 0)
    
    # Row overlap
    row_overlap = max(0, min(ar2, br2) - max(ar1, br1) + 1)
    # Col overlap
    col_overlap = max(0, min(ac2, bc2) - max(ac1, bc1) + 1)
    
    # Between box (the rectangular region between them)
    between_r1 = min(ar2 + 1, br2 + 1)
    between_r2 = max(ar1 - 1, br1 - 1)
    between_c1 = min(ac2 + 1, bc2 + 1)
    between_c2 = max(ac1 - 1, bc1 - 1)
    
    # Center-to-center distance
    dr = abs(a.center[0] - b.center[0])
    dc = abs(a.center[1] - b.center[1])
    
    return {
        'h_gap': h_gap,
        'v_gap': v_gap,
        'row_overlap': row_overlap,
        'col_overlap': col_overlap,
        'center_dr': dr,
        'center_dc': dc,
        'manhattan': dr + dc,
        'between_box': (between_r1, between_r2, between_c1, between_c2),
    }


# ═══════════════════════════════════════════════════════════
# GRAPH UTILITIES
# ═══════════════════════════════════════════════════════════

def connected_components(adj):
    """Find connected components in adjacency dict."""
    visited = set()
    components = []
    for node in adj:
        if node in visited:
            continue
        component = []
        stack = [node]
        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n)
            component.append(n)
            stack.extend(adj[n] - visited)
        components.append(sorted(component))
    # Add isolated nodes
    all_nodes = set(adj.keys())
    for node in all_nodes:
        if node not in visited:
            components.append([node])
    return components


def find_bridges(adj):
    """Find bridge edges (removing them disconnects the graph)."""
    bridges = set()
    visited = set()
    disc = {}
    low = {}
    parent = {}
    timer = [0]
    
    def dfs(u):
        visited.add(u)
        disc[u] = low[u] = timer[0]
        timer[0] += 1
        for v in adj[u]:
            if v not in visited:
                parent[v] = u
                dfs(v)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    bridges.add((min(u, v), max(u, v)))
            elif v != parent.get(u):
                low[u] = min(low[u], disc[v])
    
    for node in adj:
        if node not in visited:
            parent[node] = -1
            dfs(node)
    
    return bridges


def shortest_path(adj, src, dst):
    """BFS shortest path between two nodes."""
    if src == dst:
        return [src]
    visited = {src}
    queue = deque([(src, [src])])
    while queue:
        node, path = queue.popleft()
        for neighbor in adj[node]:
            if neighbor == dst:
                return path + [dst]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None  # No path


# ═══════════════════════════════════════════════════════════
# ROW/COLUMN BAND CLUSTERING
# ═══════════════════════════════════════════════════════════

def cluster_into_bands(objects, axis='row', tol=1.5):
    """
    Cluster objects into row or column bands.
    Objects whose center positions are within `tol` of each other are grouped.
    """
    if not objects:
        return []
    
    if axis == 'row':
        keyfn = lambda o: o.center[0]
    else:
        keyfn = lambda o: o.center[1]
    
    sorted_objs = sorted(objects, key=keyfn)
    bands = [[sorted_objs[0]]]
    
    for obj in sorted_objs[1:]:
        if abs(keyfn(obj) - keyfn(bands[-1][-1])) <= tol:
            bands[-1].append(obj)
        else:
            bands.append([obj])
    
    # Sort within each band by the other axis
    for band in bands:
        if axis == 'row':
            band.sort(key=lambda o: o.center[1])
        else:
            band.sort(key=lambda o: o.center[0])
    
    return bands


# ═══════════════════════════════════════════════════════════
# FRAME / ENCLOSURE DETECTION
# ═══════════════════════════════════════════════════════════

def is_frame_like(obj):
    """Check if an object looks like a frame/border (hollow rectangle)."""
    if obj.size < 8:
        return False
    r1, r2, c1, c2 = obj.bbox
    h, w = r2 - r1 + 1, c2 - c1 + 1
    if h < 3 or w < 3:
        return False
    
    cells_set = set(obj.cells)
    # Count cells on bbox perimeter vs interior
    perimeter_cells = 0
    interior_cells = 0
    for r in range(r1, r2 + 1):
        for c in range(c1, c2 + 1):
            if (r, c) in cells_set:
                if r == r1 or r == r2 or c == c1 or c == c2:
                    perimeter_cells += 1
                else:
                    interior_cells += 1
    
    # Frame-like if most cells are on perimeter
    if perimeter_cells == 0:
        return False
    return interior_cells / (perimeter_cells + interior_cells) < 0.3


def frame_contains(frame_obj, inner_obj):
    """Check if inner_obj is enclosed by frame_obj (beyond simple bbox)."""
    if not is_frame_like(frame_obj):
        return False
    fr1, fr2, fc1, fc2 = frame_obj.bbox
    ir1, ir2, ic1, ic2 = inner_obj.bbox
    return ir1 > fr1 and ir2 < fr2 and ic1 > fc1 and ic2 < fc2


# ═══════════════════════════════════════════════════════════
# RECTANGLE ALGEBRA (Allen's intervals in 2D)
# ═══════════════════════════════════════════════════════════

def allen_1d(a1, a2, b1, b2):
    """
    Allen's interval relation between [a1,a2] and [b1,b2].
    Returns one of: before, meets, overlaps, starts, during, finishes, 
                     equal, finishes_i, during_i, starts_i, overlaps_i, meets_i, after
    """
    if a2 < b1:
        return 'before'
    elif a2 == b1:
        return 'meets'
    elif a1 < b1 and a2 > b1 and a2 < b2:
        return 'overlaps'
    elif a1 == b1 and a2 < b2:
        return 'starts'
    elif a1 > b1 and a2 < b2:
        return 'during'
    elif a1 > b1 and a2 == b2:
        return 'finishes'
    elif a1 == b1 and a2 == b2:
        return 'equal'
    elif a1 > b1 and a1 < b2 and a2 > b2:
        return 'overlaps_i'
    elif a1 == b1 and a2 > b2:
        return 'starts_i'
    elif a1 < b1 and a2 > b2:
        return 'during_i'
    elif a1 < b1 and a2 == b2:
        return 'finishes_i'
    elif a1 == b2:
        return 'meets_i'
    else:
        return 'after'


def rectangle_algebra(a, b):
    """2D Rectangle Algebra — Allen's intervals on both axes."""
    ar1, ar2, ac1, ac2 = a.bbox
    br1, br2, bc1, bc2 = b.bbox
    return {
        'row': allen_1d(ar1, ar2, br1, br2),
        'col': allen_1d(ac1, ac2, bc1, bc2),
    }


# ═══════════════════════════════════════════════════════════
# MAIN CLASS: ObjectRelationGraph
# ═══════════════════════════════════════════════════════════

class ObjectRelationGraph:
    """
    Persistent relation structure computed once per grid.
    Reusable across all solver families.
    """
    
    def __init__(self, grid, bg=None):
        self.grid = np.array(grid)
        self.h, self.w = self.grid.shape
        
        if bg is None:
            bg = int(np.argmax(np.bincount(self.grid.flatten())))
        self.bg = bg
        
        self.objects, _ = extract_objects(self.grid, bg=bg)
        self.n = len(self.objects)
        self.by_id = {o.id: o for o in self.objects}
        
        # Relation stores
        self.relations = {}      # (a_id, b_id) → dict of relation booleans/values
        self.gaps = {}           # (a_id, b_id) → gap features dict
        self.rect_algebra = {}   # (a_id, b_id) → {row: Allen, col: Allen}
        
        # Adjacency graphs (by relation type)
        self.adj_touch = defaultdict(set)
        self.adj_align = defaultdict(set)
        self.adj_same_shape = defaultdict(set)
        self.adj_same_color = defaultdict(set)
        
        # Proximity
        self.nearest = {}        # obj_id → nearest_obj_id
        self.knn = {}            # obj_id → [id1, id2, ...]
        self.dist_matrix = {}    # (a_id, b_id) → float distance
        
        # Groups
        self.row_bands = []      # [[obj, ...], ...]
        self.col_bands = []
        
        # Components
        self.components_touch = []
        self.components_shape = []
        
        # Bridges
        self.bridge_edges = set()
        
        # Roles
        self.roles = defaultdict(set)
        
        # D4 hashes
        self.d4_hashes = {}
        self.sym_flags = {}
        
        # Euler numbers (components - holes)
        self.euler_numbers = {}
        
        # Build everything
        if self.n >= 1:
            self._compute_d4_hashes()
            self._compute_pairwise()
            self._compute_neighborhoods()
            self._compute_groups()
            self._compute_components()
            self._compute_bridges()
            self._assign_roles()
            self._decorate_objects()
    
    # ─── COMPUTATION PHASES ───────────────────────────────
    
    def _compute_d4_hashes(self):
        """Compute D4-invariant shape hashes and symmetry flags."""
        for o in self.objects:
            self.d4_hashes[o.id] = d4_canonical_hash(o.cells)
            self.sym_flags[o.id] = symmetry_flags(o.cells)
    
    def _compute_pairwise(self):
        """Compute all pairwise relations between objects."""
        for i, a in enumerate(self.objects):
            for j, b in enumerate(self.objects):
                if i >= j:
                    continue
                
                pair = (a.id, b.id)
                pair_r = (b.id, a.id)
                
                # Cell sets for relation computation
                a_cells = set(a.cells)
                b_cells = set(b.cells)
                
                # Touching (4-adjacent)
                t = any((r+dr, c+dc) in b_cells
                        for r, c in a_cells
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)])
                
                # Bbox containment
                ar1, ar2, ac1, ac2 = a.bbox
                br1, br2, bc1, bc2 = b.bbox
                a_in_b = ar1 >= br1 and ar2 <= br2 and ac1 >= bc1 and ac2 <= bc2
                b_in_a = br1 >= ar1 and br2 <= ar2 and bc1 >= ac1 and bc2 <= ac2
                
                # Alignment
                ax = abs(a.center[1] - b.center[1]) < 1.0
                ay = abs(a.center[0] - b.center[0]) < 1.0
                
                # Same properties
                ss = self.d4_hashes[a.id] == self.d4_hashes[b.id]
                sc = a.color == b.color
                sz = a.size == b.size
                
                # Directional
                left = ac2 < bc1
                right = ac1 > bc2
                above = ar2 < br1
                below = ar1 > br2
                
                # Distance
                d = abs(a.center[0] - b.center[0]) + abs(a.center[1] - b.center[1])
                
                # Store forward and reverse
                rel = {
                    'touches': t, 'inside': a_in_b, 'contains': b_in_a,
                    'aligned_x': ax, 'aligned_y': ay,
                    'same_shape': ss, 'same_color': sc, 'same_size': sz,
                    'left_of': left, 'right_of': right,
                    'above': above, 'below': below,
                    'manhattan': d,
                }
                self.relations[pair] = rel
                self.relations[pair_r] = {
                    'touches': t, 'inside': b_in_a, 'contains': a_in_b,
                    'aligned_x': ax, 'aligned_y': ay,
                    'same_shape': ss, 'same_color': sc, 'same_size': sz,
                    'left_of': right, 'right_of': left,
                    'above': below, 'below': above,
                    'manhattan': d,
                }
                
                # Gap features
                gap = compute_gap_features(a, b)
                self.gaps[pair] = gap
                self.gaps[pair_r] = gap
                
                # Rectangle algebra
                ra = rectangle_algebra(a, b)
                self.rect_algebra[pair] = ra
                
                # Distance matrix
                self.dist_matrix[pair] = d
                self.dist_matrix[pair_r] = d
                
                # Adjacency graphs
                if t:
                    self.adj_touch[a.id].add(b.id)
                    self.adj_touch[b.id].add(a.id)
                if ax or ay:
                    self.adj_align[a.id].add(b.id)
                    self.adj_align[b.id].add(a.id)
                if ss:
                    self.adj_same_shape[a.id].add(b.id)
                    self.adj_same_shape[b.id].add(a.id)
                if sc:
                    self.adj_same_color[a.id].add(b.id)
                    self.adj_same_color[b.id].add(a.id)
    
    def _compute_neighborhoods(self, k=3):
        """Compute nearest neighbor and kNN for each object."""
        for a in self.objects:
            others = []
            for b in self.objects:
                if b.id == a.id:
                    continue
                d = self.dist_matrix.get((a.id, b.id), 999)
                others.append((b.id, d))
            others.sort(key=lambda x: x[1])
            
            self.knn[a.id] = [bid for bid, _ in others[:k]]
            if others:
                self.nearest[a.id] = others[0][0]
    
    def _compute_groups(self):
        """Cluster objects into row and column bands."""
        self.row_bands = cluster_into_bands(self.objects, axis='row')
        self.col_bands = cluster_into_bands(self.objects, axis='col')
    
    def _compute_components(self):
        """Find connected components in touch and shape graphs."""
        if self.adj_touch:
            self.components_touch = connected_components(self.adj_touch)
        if self.adj_same_shape:
            self.components_shape = connected_components(self.adj_same_shape)
    
    def _compute_bridges(self):
        """Find bridge edges in the touch graph."""
        if len(self.adj_touch) >= 2:
            try:
                self.bridge_edges = find_bridges(self.adj_touch)
            except RecursionError:
                self.bridge_edges = set()
    
    def _assign_roles(self):
        """Assign candidate roles to each object based on properties."""
        if not self.objects:
            return
        
        sizes = [o.size for o in self.objects]
        max_size = max(sizes)
        min_size = min(sizes)
        
        shape_counts = Counter(self.d4_hashes[o.id] for o in self.objects)
        color_counts = Counter(o.color for o in self.objects)
        
        for o in self.objects:
            roles = set()
            
            # Template candidate: smallest unique-shape object, or dense small object
            if o.size == min_size and shape_counts[self.d4_hashes[o.id]] == 1:
                roles.add('template')
            if o.size <= 4 and o.density > 0.8:
                roles.add('template')
            
            # Marker candidate: size 1-2, or many copies of same shape
            if o.size <= 2:
                roles.add('marker')
            if shape_counts[self.d4_hashes[o.id]] >= 3:
                roles.add('marker')
            
            # Container candidate: largest object, or encloses others
            if o.size == max_size and max_size > min_size * 3:
                roles.add('container')
            if is_frame_like(o):
                roles.add('container')
                roles.add('frame')
            
            # Divider candidate: long thin line spanning most of grid
            if (o.width >= self.w * 0.8 and o.height <= 2) or \
               (o.height >= self.h * 0.8 and o.width <= 2):
                roles.add('divider')
            
            # Anchor candidate: touches border and is unique
            if o.touches_border and shape_counts[self.d4_hashes[o.id]] == 1:
                roles.add('anchor')
            
            # Mover candidate: small, not touching border, not a divider
            if not o.touches_border and o.size < max_size * 0.5 and 'divider' not in roles:
                roles.add('mover')
            
            # Singleton: unique color
            if color_counts[o.color] == 1:
                roles.add('singleton')
            
            self.roles[o.id] = roles
    
    def _decorate_objects(self):
        """Store computed properties in the graph (ArcObject uses __slots__)."""
        self.obj_props = {}
        for o in self.objects:
            self.obj_props[o.id] = {
                'neighbor_ids': sorted(self.adj_touch.get(o.id, set())),
                'nearest_id': self.nearest.get(o.id),
                'degree': len(self.adj_touch.get(o.id, set())),
                'role_candidates': sorted(self.roles.get(o.id, set())),
                'd4_hash': self.d4_hashes.get(o.id),
                'component_id': None,
            }
            for ci, comp in enumerate(self.components_touch):
                if o.id in comp:
                    self.obj_props[o.id]['component_id'] = ci
                    break
    
    # ─── QUERY INTERFACE ──────────────────────────────────
    
    def related(self, a_id, b_id, rel):
        """Check if relation holds between two objects."""
        pair = (a_id, b_id)
        if pair in self.relations:
            return self.relations[pair].get(rel, False)
        return False
    
    def nearest_of(self, obj_id):
        """Get the nearest object to the given object."""
        nid = self.nearest.get(obj_id)
        return self.by_id.get(nid) if nid is not None else None
    
    def objects_with_role(self, role):
        """Get all objects that have a given role candidate."""
        return [self.by_id[oid] for oid, roles in self.roles.items() if role in roles]
    
    def objects_by_color(self, color):
        """Get all objects of a given color."""
        return [o for o in self.objects if o.color == color]
    
    def objects_in_row_band(self, obj):
        """Get all objects in the same row band as the given object."""
        for band in self.row_bands:
            if any(o.id == obj.id for o in band):
                return band
        return [obj]
    
    def objects_in_col_band(self, obj):
        """Get all objects in the same column band."""
        for band in self.col_bands:
            if any(o.id == obj.id for o in band):
                return band
        return [obj]
    
    def gap_between(self, a_id, b_id):
        """Get gap features between two objects."""
        return self.gaps.get((a_id, b_id), {})
    
    def path_between(self, a_id, b_id, graph='touch'):
        """Find shortest path between two objects in the given graph."""
        adj = {
            'touch': self.adj_touch,
            'align': self.adj_align,
            'shape': self.adj_same_shape,
            'color': self.adj_same_color,
        }.get(graph, self.adj_touch)
        return shortest_path(adj, a_id, b_id)
    
    def color_groups(self):
        """Group objects by color."""
        groups = defaultdict(list)
        for o in self.objects:
            groups[o.color].append(o)
        return dict(groups)
    
    def shape_groups(self):
        """Group objects by D4-canonical shape."""
        groups = defaultdict(list)
        for o in self.objects:
            groups[self.d4_hashes[o.id]].append(o)
        return dict(groups)
    
    def size_rank(self):
        """Return objects sorted by size (smallest first) with rank."""
        sorted_objs = sorted(self.objects, key=lambda o: o.size)
        return [(i, o) for i, o in enumerate(sorted_objs)]
    
    # ─── FEATURE VECTOR ───────────────────────────────────
    
    def feature_vector(self, obj):
        """
        30-dim numeric feature vector for an object.
        Suitable for XGBoost, kNN retrieval, or GNN input.
        """
        fv = np.zeros(30, dtype=np.float32)
        
        # Basic properties (0-9)
        fv[0] = obj.color
        fv[1] = obj.size
        fv[2] = obj.width
        fv[3] = obj.height
        fv[4] = obj.aspect
        fv[5] = obj.density
        fv[6] = float(obj.is_rectangular)
        fv[7] = float(obj.touches_border)
        fv[8] = obj.center[0] / max(self.h, 1)  # Normalized position
        fv[9] = obj.center[1] / max(self.w, 1)
        
        # Graph properties (10-16)
        fv[10] = len(self.adj_touch.get(obj.id, set()))     # Touch degree
        fv[11] = len(self.adj_align.get(obj.id, set()))      # Align degree
        fv[12] = len(self.adj_same_shape.get(obj.id, set())) # Same-shape degree
        fv[13] = len(self.adj_same_color.get(obj.id, set())) # Same-color degree
        nn = self.nearest.get(obj.id)
        fv[14] = self.dist_matrix.get((obj.id, nn), 99) if nn else 99
        fv[15] = len(self.knn.get(obj.id, []))
        fv[16] = self.obj_props.get(obj.id, {}).get('component_id', -1)
        
        # Symmetry flags (17-20)
        sf = self.sym_flags.get(obj.id, {})
        fv[17] = float(sf.get('hflip', False))
        fv[18] = float(sf.get('vflip', False))
        fv[19] = float(sf.get('rot180', False))
        fv[20] = float(sf.get('rot90', False))
        
        # Role flags (21-27)
        roles = self.roles.get(obj.id, set())
        fv[21] = float('template' in roles)
        fv[22] = float('marker' in roles)
        fv[23] = float('container' in roles)
        fv[24] = float('divider' in roles)
        fv[25] = float('mover' in roles)
        fv[26] = float('anchor' in roles)
        fv[27] = float('singleton' in roles)
        
        # Relative properties (28-29)
        if self.objects:
            fv[28] = obj.size / max(o.size for o in self.objects)  # Size ratio
            fv[29] = float(obj.size == min(o.size for o in self.objects))  # Is smallest
        
        return fv
    
    def task_feature_vector(self):
        """
        Global task-level feature vector (for task routing/retrieval).
        """
        fv = np.zeros(20, dtype=np.float32)
        fv[0] = self.h
        fv[1] = self.w
        fv[2] = self.n  # Number of objects
        fv[3] = len(set(o.color for o in self.objects))  # Unique colors
        fv[4] = len(set(self.d4_hashes[o.id] for o in self.objects))  # Unique shapes
        fv[5] = len(self.components_touch)  # Touch components
        fv[6] = max((len(c) for c in self.components_touch), default=0)  # Largest component
        fv[7] = len(self.bridge_edges)  # Bridge count
        fv[8] = len(self.row_bands)  # Row bands
        fv[9] = len(self.col_bands)  # Col bands
        fv[10] = sum(1 for o in self.objects if 'template' in self.roles[o.id])
        fv[11] = sum(1 for o in self.objects if 'marker' in self.roles[o.id])
        fv[12] = sum(1 for o in self.objects if 'container' in self.roles[o.id])
        fv[13] = sum(1 for o in self.objects if 'divider' in self.roles[o.id])
        fv[14] = np.mean([o.density for o in self.objects]) if self.objects else 0
        fv[15] = np.std([o.size for o in self.objects]) if len(self.objects) > 1 else 0
        fv[16] = int(np.sum(self.grid != self.bg))  # Non-bg pixel count
        fv[17] = fv[16] / max(self.h * self.w, 1)  # Fill ratio
        fv[18] = max((o.size for o in self.objects), default=0) / max(min((o.size for o in self.objects), default=1), 1)  # Max/min size ratio
        fv[19] = float(any('frame' in self.roles[o.id] for o in self.objects))  # Has frame
        return fv
    
    # ─── STRING REPRESENTATION ────────────────────────────
    
    def __repr__(self):
        return (f"ObjectRelationGraph(grid={self.h}×{self.w}, "
                f"objects={self.n}, bg={self.bg}, "
                f"touch_edges={sum(len(v) for v in self.adj_touch.values())//2}, "
                f"components={len(self.components_touch)}, "
                f"row_bands={len(self.row_bands)}, "
                f"col_bands={len(self.col_bands)})")


# ═══════════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    import json, os, sys, time
    
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'ARC-AGI-2/data/training'
    
    files = sorted(os.listdir(data_dir))[:20]  # Test on first 20
    
    print(f"Testing ObjectRelationGraph on {len(files)} tasks...")
    t0 = time.time()
    
    for f in files:
        if not f.endswith('.json'):
            continue
        task = json.load(open(os.path.join(data_dir, f)))
        gi = np.array(task['train'][0]['input'])
        
        org = ObjectRelationGraph(gi)
        
        roles_summary = defaultdict(int)
        for oid, roles in org.roles.items():
            for r in roles:
                roles_summary[r] += 1
        
        print(f"  {f}: {org}")
        if roles_summary:
            print(f"    Roles: {dict(roles_summary)}")
    
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.2f}s ({elapsed/len(files)*1000:.0f}ms/task)")
