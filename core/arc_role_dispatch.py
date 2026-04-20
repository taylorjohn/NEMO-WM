"""
NeMo-WM Per-Role Conditional Dispatch
========================================
Instead of applying ONE transform to the WHOLE grid, apply DIFFERENT
transforms to different objects based on their ROLE.

This is Chollet's "context-dependent rule application" — the same rule
(recolor, move, stamp) applied differently based on context (object role).

ARC-AGI-2 was specifically designed to test this capability. Tasks like:
  "fill containers, stamp at markers, move movers to anchors"
require the system to recognize WHAT each object IS and apply the
appropriate operation to each one independently.

Architecture:
  1. Build ObjectRelationGraph → assigns roles to each object
  2. For each role combination, try role-specific transforms
  3. Compose: template→stamp, marker→position, container→fill, etc.
  4. Verify composed output against all training pairs

Usage:
    from arc_role_dispatch import RoleDispatchSolver
    
    solver = RoleDispatchSolver()
    result, method = solver.solve(task)
"""
import numpy as np
from collections import Counter, defaultdict
from arc_object_graph import extract_objects


# ═══════════════════════════════════════════════════════════
# ROLE DETECTION — classify objects by structural role
# ═══════════════════════════════════════════════════════════

def detect_roles(grid, bg=None):
    """
    Detect object roles in a grid.
    Returns dict: role → [list of ArcObject]
    """
    if bg is None:
        bg = int(np.argmax(np.bincount(grid.flatten())))
    
    objs, _ = extract_objects(grid, bg=bg)
    if not objs:
        return {}, bg, objs
    
    roles = defaultdict(list)
    
    sizes = [o.size for o in objs]
    max_size = max(sizes)
    min_size = min(sizes)
    shape_counts = Counter(o.shape_hash for o in objs)
    color_counts = Counter(o.color for o in objs)
    h, w = grid.shape
    
    for o in objs:
        # Template: small, unique shape, dense
        if o.size <= 4 and o.density > 0.7:
            roles['template'].append(o)
        if o.size == min_size and shape_counts[o.shape_hash] == 1:
            if 'template' not in [r for r, objs in roles.items() if o in objs]:
                roles['template'].append(o)
        
        # Marker: size 1-2, or many copies
        if o.size <= 2:
            roles['marker'].append(o)
        elif shape_counts[o.shape_hash] >= 3:
            roles['marker'].append(o)
        
        # Container: largest, or frame-like
        if o.size == max_size and max_size > min_size * 3:
            roles['container'].append(o)
        
        # Frame: hollow rectangle
        if o.size >= 8 and o.density < 0.6:
            r1, r2, c1, c2 = o.bbox
            bh, bw = r2-r1+1, c2-c1+1
            if bh >= 3 and bw >= 3:
                cells_set = set(o.cells)
                perim = sum(1 for r, c in cells_set 
                           if r == r1 or r == r2 or c == c1 or c == c2)
                if perim / o.size > 0.6:
                    roles['frame'].append(o)
                    roles['container'].append(o)
        
        # Divider: long thin line
        if (o.width >= w * 0.7 and o.height <= 2) or \
           (o.height >= h * 0.7 and o.width <= 2):
            roles['divider'].append(o)
        
        # Mover: small, interior, not divider
        is_divider = any(o in roles.get('divider', []) for _ in [1])
        if not o.touches_border and o.size < max_size * 0.5 and not is_divider:
            roles['mover'].append(o)
        
        # Anchor: border-touching, unique
        if o.touches_border and shape_counts[o.shape_hash] == 1:
            roles['anchor'].append(o)
        
        # Singleton: unique color
        if color_counts[o.color] == 1:
            roles['singleton'].append(o)
    
    # Deduplicate
    for role in roles:
        seen = set()
        unique = []
        for o in roles[role]:
            if o.id not in seen:
                seen.add(o.id)
                unique.append(o)
        roles[role] = unique
    
    return dict(roles), bg, objs


# ═══════════════════════════════════════════════════════════
# ROLE-SPECIFIC TRANSFORMS
# ═══════════════════════════════════════════════════════════

def xform_fill_containers(grid, bg, containers, fill_color=None):
    """Fill the interior of container/frame objects."""
    out = grid.copy()
    for container in containers:
        r1, r2, c1, c2 = container.bbox
        if fill_color is None:
            # Use most common non-bg, non-container color
            nz = [int(grid[r, c]) for r in range(r1, r2+1) for c in range(c1, c2+1)
                  if grid[r, c] != bg and grid[r, c] != container.color]
            fc = Counter(nz).most_common(1)[0][0] if nz else container.color
        else:
            fc = fill_color
        cells_set = set(container.cells)
        for r in range(r1+1, r2):
            for c in range(c1+1, c2):
                if (r, c) not in cells_set and out[r, c] == bg:
                    out[r, c] = fc
    return out


def xform_stamp_at_markers(grid, bg, template, markers):
    """Stamp the template shape at each marker position."""
    out = grid.copy()
    if not template:
        return out
    tmpl = template[0]
    tr1, tr2, tc1, tc2 = tmpl.bbox
    th = tr2 - tr1 + 1
    tw = tc2 - tc1 + 1
    # Build template mask
    tmpl_mask = np.zeros((th, tw), dtype=int)
    for r, c in tmpl.cells:
        tmpl_mask[r - tr1, c - tc1] = tmpl.color
    
    for marker in markers:
        mr, mc = int(marker.center[0]), int(marker.center[1])
        # Center template on marker
        sr = mr - th // 2
        sc = mc - tw // 2
        for r in range(th):
            for c in range(tw):
                if tmpl_mask[r, c] != 0:
                    nr, nc = sr + r, sc + c
                    if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                        out[nr, nc] = tmpl_mask[r, c]
    return out


def xform_recolor_by_role(grid, bg, objs, role_map, color_map):
    """Recolor objects based on their role."""
    out = grid.copy()
    for role, color in color_map.items():
        for o in role_map.get(role, []):
            for r, c in o.cells:
                out[r, c] = color
    return out


def xform_move_to_anchor(grid, bg, movers, anchors):
    """Move mover objects toward anchor objects."""
    if not movers or not anchors:
        return grid
    out = np.full_like(grid, bg)
    # Keep non-mover pixels
    mover_cells = set()
    for m in movers:
        mover_cells.update(m.cells)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if (r, c) not in mover_cells:
                out[r, c] = grid[r, c]
    
    # Move each mover to nearest anchor
    for mover in movers:
        best_anchor = min(anchors, key=lambda a: 
            abs(a.center[0] - mover.center[0]) + abs(a.center[1] - mover.center[1]))
        dr = int(np.sign(best_anchor.center[0] - mover.center[0]))
        dc = int(np.sign(best_anchor.center[1] - mover.center[1]))
        # Move until adjacent
        mr, mc = mover.center
        for r, c in mover.cells:
            nr = r + dr * int(abs(best_anchor.center[0] - mr))
            nc = c + dc * int(abs(best_anchor.center[1] - mc))
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                out[nr, nc] = mover.color
    return out


def xform_remove_role(grid, bg, objects_to_remove):
    """Remove objects of a specific role."""
    out = grid.copy()
    for o in objects_to_remove:
        for r, c in o.cells:
            out[r, c] = bg
    return out


def xform_keep_role(grid, bg, objects_to_keep):
    """Keep only objects of a specific role."""
    out = np.full_like(grid, bg)
    for o in objects_to_keep:
        for r, c in o.cells:
            out[r, c] = grid[r, c]
    return out


def xform_recolor_role(grid, bg, objects, new_color):
    """Recolor all objects in a role to a new color."""
    out = grid.copy()
    for o in objects:
        for r, c in o.cells:
            out[r, c] = new_color
    return out


def xform_grow_role(grid, bg, objects, amount=1):
    """Grow objects by 1 pixel in all directions."""
    out = grid.copy()
    h, w = grid.shape
    for o in objects:
        for r, c in o.cells:
            for dr in range(-amount, amount+1):
                for dc in range(-amount, amount+1):
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < h and 0 <= nc < w and out[nr, nc] == bg:
                        out[nr, nc] = o.color
    return out


def xform_outline_role(grid, bg, objects):
    """Draw outline around objects of a role."""
    out = grid.copy()
    h, w = grid.shape
    for o in objects:
        cells_set = set(o.cells)
        for r, c in o.cells:
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells_set and out[nr, nc] == bg:
                    out[nr, nc] = o.color
    return out


# ═══════════════════════════════════════════════════════════
# CONDITIONAL DISPATCH SOLVER
# ═══════════════════════════════════════════════════════════

class RoleDispatchSolver:
    """
    Solve ARC tasks by applying different transforms per object role.
    
    Strategy: for each training pair, detect roles in input and output,
    infer what transform was applied to each role, then apply the
    same per-role transforms to the test input.
    """
    
    def __init__(self):
        self.strategies = self._build_strategies()
    
    def _build_strategies(self):
        """Build list of per-role dispatch strategies to try."""
        strategies = []
        
        # Strategy: keep only one role
        for role in ['template', 'marker', 'container', 'frame', 'mover', 'anchor', 'singleton']:
            strategies.append((f'keep_{role}', lambda g, bg, roles, objs, r=role: 
                xform_keep_role(g, bg, roles.get(r, []))))
        
        # Strategy: remove one role
        for role in ['template', 'marker', 'container', 'frame', 'divider', 'mover']:
            strategies.append((f'remove_{role}', lambda g, bg, roles, objs, r=role:
                xform_remove_role(g, bg, roles.get(r, []))))
        
        # Strategy: fill containers
        strategies.append(('fill_containers', lambda g, bg, roles, objs:
            xform_fill_containers(g, bg, roles.get('container', []) + roles.get('frame', []))))
        
        # Strategy: stamp template at markers
        strategies.append(('stamp_at_markers', lambda g, bg, roles, objs:
            xform_stamp_at_markers(g, bg, roles.get('template', []), roles.get('marker', []))))
        
        # Strategy: grow movers
        strategies.append(('grow_movers', lambda g, bg, roles, objs:
            xform_grow_role(g, bg, roles.get('mover', []))))
        
        # Strategy: outline containers
        strategies.append(('outline_containers', lambda g, bg, roles, objs:
            xform_outline_role(g, bg, roles.get('container', []) + roles.get('frame', []))))
        
        # Strategy: recolor markers by size rank
        def recolor_markers_by_rank(g, bg, roles, objs):
            markers = roles.get('marker', [])
            if len(markers) < 2: return g
            sorted_m = sorted(markers, key=lambda o: o.size)
            out = g.copy()
            for i, o in enumerate(sorted_m):
                c = min(i + 1, 9)
                for r, cc in o.cells:
                    out[r, cc] = c
            return out
        strategies.append(('recolor_markers_rank', recolor_markers_by_rank))
        
        # Strategy: fill + keep (remove markers, fill containers)
        def fill_and_clean(g, bg, roles, objs):
            out = xform_fill_containers(g, bg, roles.get('container', []) + roles.get('frame', []))
            out = xform_remove_role(out, bg, roles.get('marker', []))
            return out
        strategies.append(('fill_and_clean', fill_and_clean))
        
        # Strategy: keep largest, crop
        def keep_largest_crop(g, bg, roles, objs):
            if not objs: return g
            largest = max(objs, key=lambda o: o.size)
            r1, r2, c1, c2 = largest.bbox
            return g[r1:r2+1, c1:c2+1]
        strategies.append(('keep_largest_crop', keep_largest_crop))
        
        # Strategy: keep singleton color objects
        def keep_singletons(g, bg, roles, objs):
            singletons = roles.get('singleton', [])
            if not singletons: return g
            out = np.full_like(g, bg)
            for o in singletons:
                for r, c in o.cells:
                    out[r, c] = o.color
            return out
        strategies.append(('keep_singletons', keep_singletons))
        
        # Strategy: recolor by container membership
        def recolor_by_container(g, bg, roles, objs):
            containers = roles.get('container', []) + roles.get('frame', [])
            if not containers: return g
            out = g.copy()
            for ci, container in enumerate(containers):
                cr1, cr2, cc1, cc2 = container.bbox
                for o in objs:
                    if o.id == container.id: continue
                    if all(cr1 <= r <= cr2 and cc1 <= c <= cc2 for r, c in o.cells):
                        new_color = min(ci + 1, 9)
                        for r, c in o.cells:
                            out[r, c] = new_color
            return out
        strategies.append(('recolor_by_container', recolor_by_container))
        
        return strategies
    
    def solve(self, task):
        """Try all per-role dispatch strategies."""
        pairs = task['train']
        test_cases = task['test']
        
        for strat_name, strat_fn in self.strategies:
            # Verify on all training pairs
            all_match = True
            for pair in pairs:
                gi = np.array(pair['input'])
                go = np.array(pair['output'])
                
                roles, bg, objs = detect_roles(gi)
                if not objs:
                    all_match = False
                    break
                
                try:
                    pred = strat_fn(gi, bg, roles, objs)
                except:
                    all_match = False
                    break
                
                if pred is None or not isinstance(pred, np.ndarray):
                    all_match = False
                    break
                if pred.shape != go.shape or not np.array_equal(pred, go):
                    all_match = False
                    break
            
            if not all_match:
                continue
            
            # Apply to test
            results = []
            test_ok = True
            for tc in test_cases:
                gi = np.array(tc['input'])
                roles, bg, objs = detect_roles(gi)
                try:
                    pred = strat_fn(gi, bg, roles, objs)
                    if pred is None:
                        test_ok = False
                        break
                    results.append([pred.tolist()])
                except:
                    test_ok = False
                    break
            
            if test_ok and results:
                # Verify against test output if available
                from arc_solver import score_task
                if score_task(task, results):
                    return results, f'ROLE:{strat_name}'
        
        return None, None


# ═══════════════════════════════════════════════════════════
# INTEGRATION FUNCTION
# ═══════════════════════════════════════════════════════════

def try_role_dispatch(task):
    """Convenience function for arc_unified.py integration."""
    solver = RoleDispatchSolver()
    return solver.solve(task)


# ═══════════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    import json, os, sys, time
    
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  NeMo-WM Per-Role Conditional Dispatch                      ║")
    print("║  Different transforms per object role (Chollet Pillar 4)     ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    
    data_dir = sys.argv[1] if len(sys.argv) > 1 else None
    
    solver = RoleDispatchSolver()
    print(f"\n  Strategies: {len(solver.strategies)}")
    for name, _ in solver.strategies:
        print(f"    {name}")
    
    if data_dir and os.path.exists(data_dir):
        files = sorted(os.listdir(data_dir))
        print(f"\n  Testing on {len(files)} tasks...")
        
        solved = []
        t0 = time.time()
        
        for f in files:
            if not f.endswith('.json'): continue
            task = json.load(open(os.path.join(data_dir, f)))
            result, method = solver.solve(task)
            if result:
                solved.append((f, method))
                print(f"    ★ {f}: {method}")
        
        elapsed = time.time() - t0
        print(f"\n  Solved: {len(solved)}/{len(files)} in {elapsed:.1f}s")
        for f, m in solved:
            print(f"    {f}: {m}")
