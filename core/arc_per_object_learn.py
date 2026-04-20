"""
NeMo-WM Per-Object Learning Solver
=====================================
The biggest gap in our 119/1000: tasks where different objects
get different transforms based on learned rules.

Architecture:
  1. Extract objects from input AND output of each training pair
  2. Match input objects → output objects (by position, shape, size)
  3. Infer what changed for each matched pair (recolor, move, delete, grow, etc.)
  4. Generalize the rule: WHY did each object change? (by size, color, position, shape)
  5. Apply the generalized rule to test input objects

This is fundamentally different from beam search (which tries whole-grid transforms).
Here we learn CONDITIONAL, PER-OBJECT rules from examples.

Example tasks this catches:
  "Recolor the smallest object red, keep others unchanged"
  "Delete all objects touching the border"
  "Move each object to the nearest corner"
  "Fill objects that are hollow, leave solid ones alone"
"""
import numpy as np
from collections import Counter, defaultdict
from arc_solver import score_task
from arc_object_graph import extract_objects


# ═══════════════════════════════════════════════════════════
# OBJECT MATCHING — align input objects to output objects
# ═══════════════════════════════════════════════════════════

def match_objects(in_objs, out_objs, gi, go):
    """
    Match input objects to output objects.
    Returns list of (in_obj, out_obj_or_None, change_type).
    """
    bg_i = int(np.argmax(np.bincount(gi.flatten())))
    bg_o = int(np.argmax(np.bincount(go.flatten())))
    
    matches = []
    used_out = set()
    
    for in_obj in in_objs:
        best_match = None
        best_score = -1
        
        for j, out_obj in enumerate(out_objs):
            if j in used_out:
                continue
            score = _match_score(in_obj, out_obj, gi.shape)
            if score > best_score:
                best_score = score
                best_match = (j, out_obj)
        
        if best_match and best_score > 0.3:
            j, out_obj = best_match
            used_out.add(j)
            change = _detect_change(in_obj, out_obj)
            matches.append((in_obj, out_obj, change))
        else:
            matches.append((in_obj, None, 'deleted'))
    
    # Unmatched output objects (new objects)
    for j, out_obj in enumerate(out_objs):
        if j not in used_out:
            matches.append((None, out_obj, 'created'))
    
    return matches


def _match_score(in_obj, out_obj, grid_shape):
    """Score how well two objects match (0-1)."""
    # Position overlap
    in_cells = set(in_obj.cells)
    out_cells = set(out_obj.cells)
    
    if in_cells and out_cells:
        overlap = len(in_cells & out_cells)
        union = len(in_cells | out_cells)
        iou = overlap / max(union, 1)
    else:
        iou = 0.0
    
    # Centroid distance
    dist = abs(in_obj.center[0] - out_obj.center[0]) + abs(in_obj.center[1] - out_obj.center[1])
    max_dist = grid_shape[0] + grid_shape[1]
    pos_score = 1.0 - dist / max(max_dist, 1)
    
    # Size similarity
    size_score = 1.0 - abs(in_obj.size - out_obj.size) / max(in_obj.size, out_obj.size, 1)
    
    # Shape similarity (bbox aspect ratio)
    in_h = in_obj.bbox[1] - in_obj.bbox[0] + 1
    in_w = in_obj.bbox[3] - in_obj.bbox[2] + 1
    out_h = out_obj.bbox[1] - out_obj.bbox[0] + 1
    out_w = out_obj.bbox[3] - out_obj.bbox[2] + 1
    in_aspect = in_h / max(in_w, 1)
    out_aspect = out_h / max(out_w, 1)
    shape_score = 1.0 - abs(in_aspect - out_aspect) / max(in_aspect, out_aspect, 1)
    
    return 0.4 * iou + 0.25 * pos_score + 0.2 * size_score + 0.15 * shape_score


def _detect_change(in_obj, out_obj):
    """Detect what changed between matched objects."""
    if in_obj.color != out_obj.color and set(in_obj.cells) == set(out_obj.cells):
        return 'recolored'
    if in_obj.color == out_obj.color and set(in_obj.cells) != set(out_obj.cells):
        if in_obj.size < out_obj.size:
            return 'grown'
        elif in_obj.size > out_obj.size:
            return 'shrunk'
        else:
            return 'moved'
    if in_obj.color != out_obj.color and set(in_obj.cells) != set(out_obj.cells):
        return 'recolored+moved'
    return 'unchanged'


# ═══════════════════════════════════════════════════════════
# FEATURE EXTRACTION — describe objects for rule learning
# ═══════════════════════════════════════════════════════════

def object_features(obj, all_objs, grid):
    """Extract features that might determine the rule."""
    bg = int(np.argmax(np.bincount(grid.flatten())))
    h, w = grid.shape
    
    sizes = [o.size for o in all_objs]
    avg_size = np.mean(sizes) if sizes else 1
    
    feats = {
        'size': obj.size,
        'color': obj.color,
        'center_r': obj.center[0],
        'center_c': obj.center[1],
        'bbox_h': obj.bbox[1] - obj.bbox[0] + 1,
        'bbox_w': obj.bbox[3] - obj.bbox[2] + 1,
        'density': obj.density,
        'touches_border': obj.touches_border,
        'is_largest': obj.size == max(sizes),
        'is_smallest': obj.size == min(sizes),
        'size_rank': sorted(sizes).index(obj.size),
        'n_same_color': sum(1 for o in all_objs if o.color == obj.color),
        'is_unique_color': sum(1 for o in all_objs if o.color == obj.color) == 1,
        'is_square': (obj.bbox[1]-obj.bbox[0]) == (obj.bbox[3]-obj.bbox[2]),
        'is_tall': (obj.bbox[1]-obj.bbox[0]+1) > (obj.bbox[3]-obj.bbox[2]+1),
        'is_wide': (obj.bbox[3]-obj.bbox[2]+1) > (obj.bbox[1]-obj.bbox[0]+1),
        'above_center': obj.center[0] < h / 2,
        'left_of_center': obj.center[1] < w / 2,
        'is_hollow': obj.density < 0.7 and obj.size > 4,
        'larger_than_avg': obj.size > avg_size,
    }
    return feats


# ═══════════════════════════════════════════════════════════
# RULE LEARNING — infer conditional transform rules
# ═══════════════════════════════════════════════════════════

def learn_recolor_rule(training_data):
    """
    Learn: which feature determines the new color?
    training_data: list of (features_dict, old_color, new_color)
    """
    if not training_data:
        return None
    
    # Check if all recolored objects share a distinguishing feature
    recolored = [(f, oc, nc) for f, oc, nc in training_data if oc != nc]
    unchanged = [(f, oc, nc) for f, oc, nc in training_data if oc == nc]
    
    if not recolored:
        return None
    
    # Try each boolean feature as the discriminator
    bool_features = ['is_largest', 'is_smallest', 'is_unique_color', 'touches_border',
                     'is_square', 'is_tall', 'is_wide', 'above_center', 'left_of_center',
                     'is_hollow', 'larger_than_avg']
    
    for feat_name in bool_features:
        # Check: do all recolored objects have feat=True and all unchanged have feat=False?
        recolored_vals = [f[feat_name] for f, _, _ in recolored]
        unchanged_vals = [f[feat_name] for f, _, _ in unchanged]
        
        if recolored_vals and all(recolored_vals) and unchanged_vals and not any(unchanged_vals):
            new_color = recolored[0][2]
            if all(nc == new_color for _, _, nc in recolored):
                return {'type': 'recolor_if', 'feature': feat_name, 'value': True, 'new_color': new_color}
        
        if recolored_vals and not any(recolored_vals) and unchanged_vals and all(unchanged_vals):
            new_color = recolored[0][2]
            if all(nc == new_color for _, _, nc in recolored):
                return {'type': 'recolor_if', 'feature': feat_name, 'value': False, 'new_color': new_color}
    
    # Try: recolor by size rank
    size_color_map = {}
    for f, oc, nc in training_data:
        rank = f['size_rank']
        if rank in size_color_map and size_color_map[rank] != nc:
            break
        size_color_map[rank] = nc
    else:
        if len(set(size_color_map.values())) > 1:
            return {'type': 'recolor_by_rank', 'map': size_color_map}
    
    # Try: all objects get same new color
    new_colors = [nc for _, _, nc in recolored]
    if len(set(new_colors)) == 1 and len(recolored) == len(training_data):
        return {'type': 'recolor_all', 'new_color': new_colors[0]}
    
    return None


def learn_delete_rule(training_data):
    """
    Learn: which feature determines deletion?
    training_data: list of (features_dict, was_deleted)
    """
    deleted = [(f, d) for f, d in training_data if d]
    kept = [(f, d) for f, d in training_data if not d]
    
    if not deleted or not kept:
        return None
    
    bool_features = ['is_largest', 'is_smallest', 'is_unique_color', 'touches_border',
                     'is_square', 'is_tall', 'is_wide', 'is_hollow', 'larger_than_avg']
    
    for feat_name in bool_features:
        del_vals = [f[feat_name] for f, _ in deleted]
        kept_vals = [f[feat_name] for f, _ in kept]
        
        if all(del_vals) and not any(kept_vals):
            return {'type': 'delete_if', 'feature': feat_name, 'value': True}
        if not any(del_vals) and all(kept_vals):
            return {'type': 'delete_if', 'feature': feat_name, 'value': False}
    
    # Delete by color
    del_colors = set(f['color'] for f, _ in deleted)
    kept_colors = set(f['color'] for f, _ in kept)
    if not del_colors & kept_colors and len(del_colors) <= 2:
        return {'type': 'delete_by_color', 'colors': del_colors}
    
    # Delete by size (smaller than threshold)
    del_sizes = [f['size'] for f, _ in deleted]
    kept_sizes = [f['size'] for f, _ in kept]
    if max(del_sizes) < min(kept_sizes):
        return {'type': 'delete_smaller_than', 'threshold': min(kept_sizes)}
    
    return None


# ═══════════════════════════════════════════════════════════
# SOLVER — combines matching, features, and rule learning
# ═══════════════════════════════════════════════════════════

class PerObjectLearner:
    """
    Learn per-object transformation rules from training pairs.
    """
    
    def solve(self, task):
        pairs = task['train']
        test_cases = task['test']
        
        # Strategy 1: Learn recolor rules
        result = self._try_recolor_rules(task)
        if result and result[0]: return result
        
        # Strategy 2: Recolor by nearest singleton
        result = self._try_recolor_nearest(task)
        if result and result[0]: return result
        
        # Strategy 3: Learn delete rules  
        result = self._try_delete_rules(task)
        if result and result[0]: return result
        
        # Strategy 4: Learn keep-one rules
        result = self._try_keep_rules(task)
        if result and result[0]: return result
        
        return None, None
    
    def _try_recolor_nearest(self, task):
        """Recolor each majority-color object to its nearest singleton's color."""
        pairs = task['train']
        for pair in pairs:
            gi, go = np.array(pair['input']), np.array(pair['output'])
            if gi.shape != go.shape: return None, None
        
        bg = int(np.argmax(np.bincount(np.array(pairs[0]['input']).flatten())))
        
        all_ok = True
        for pair in pairs:
            gi, go = np.array(pair['input']), np.array(pair['output'])
            try:
                in_objs, _ = extract_objects(gi, bg=bg)
            except: return None, None
            if not in_objs: return None, None
            
            color_counts = Counter(o.color for o in in_objs)
            if len(color_counts) < 2: return None, None
            singletons = [o for o in in_objs if color_counts[o.color] == 1 and o.size <= 3]
            majority_color = max(color_counts, key=color_counts.get)
            targets = [o for o in in_objs if o.color == majority_color]
            
            if not singletons or not targets:
                all_ok = False; break
            
            pred = gi.copy()
            for obj in targets:
                dists = [(abs(obj.center[0]-s.center[0]) + abs(obj.center[1]-s.center[1]), s)
                         for s in singletons]
                nearest = min(dists, key=lambda x: x[0])[1]
                for r, c in obj.cells:
                    pred[r, c] = nearest.color
            
            if not np.array_equal(pred, go):
                all_ok = False; break
        
        if not all_ok:
            return None, None
        
        results = []
        for tc in task['test']:
            gi = np.array(tc['input'])
            try:
                in_objs, _ = extract_objects(gi, bg=bg)
            except: return None, None
            color_counts = Counter(o.color for o in in_objs)
            singletons = [o for o in in_objs if color_counts[o.color] == 1 and o.size <= 3]
            majority_color = max(color_counts, key=color_counts.get) if color_counts else None
            if not majority_color or not singletons: return None, None
            targets = [o for o in in_objs if o.color == majority_color]
            pred = gi.copy()
            for obj in targets:
                dists = [(abs(obj.center[0]-s.center[0]) + abs(obj.center[1]-s.center[1]), s)
                         for s in singletons]
                nearest = min(dists, key=lambda x: x[0])[1]
                for r, c in obj.cells:
                    pred[r, c] = nearest.color
            results.append([pred.tolist()])
        
        if results and score_task(task, results):
            return results, 'LEARN:recolor_nearest'
        return None, None
    
    def _try_recolor_rules(self, task):
        """Try to learn a recolor rule from training pairs."""
        pairs = task['train']
        
        # Collect training data across all pairs
        all_training = []
        for pair in pairs:
            gi = np.array(pair['input'])
            go = np.array(pair['output'])
            if gi.shape != go.shape:
                return None, None
            
            bg = int(np.argmax(np.bincount(gi.flatten())))
            in_objs, _ = extract_objects(gi, bg=bg)
            out_objs, _ = extract_objects(go, bg=bg)
            
            if not in_objs:
                return None, None
            
            matches = match_objects(in_objs, out_objs, gi, go)
            
            for in_obj, out_obj, change in matches:
                if in_obj is None or out_obj is None:
                    continue
                feats = object_features(in_obj, in_objs, gi)
                all_training.append((feats, in_obj.color, out_obj.color))
        
        if not all_training:
            return None, None
        
        rule = learn_recolor_rule(all_training)
        if rule is None:
            return None, None
        
        # Apply rule to test
        return self._apply_recolor_rule(task, rule)
    
    def _apply_recolor_rule(self, task, rule):
        """Apply a learned recolor rule to test cases."""
        results = []
        for tc in task['test']:
            gi = np.array(tc['input'])
            bg = int(np.argmax(np.bincount(gi.flatten())))
            objs, _ = extract_objects(gi, bg=bg)
            pred = gi.copy()
            
            for obj in objs:
                feats = object_features(obj, objs, gi)
                
                if rule['type'] == 'recolor_if':
                    if feats[rule['feature']] == rule['value']:
                        for r, c in obj.cells:
                            pred[r, c] = rule['new_color']
                elif rule['type'] == 'recolor_by_rank':
                    rank = feats['size_rank']
                    if rank in rule['map']:
                        for r, c in obj.cells:
                            pred[r, c] = rule['map'][rank]
                elif rule['type'] == 'recolor_all':
                    for r, c in obj.cells:
                        pred[r, c] = rule['new_color']
            
            results.append([pred.tolist()])
        
        if score_task(task, results):
            rule_desc = rule.get('feature', rule['type'])
            return results, f'LEARN:recolor_{rule_desc}'
        return None, None
    
    def _try_delete_rules(self, task):
        """Try to learn a delete rule from training pairs."""
        pairs = task['train']
        
        all_training = []
        for pair in pairs:
            gi = np.array(pair['input'])
            go = np.array(pair['output'])
            if gi.shape != go.shape:
                return None, None
            
            bg = int(np.argmax(np.bincount(gi.flatten())))
            in_objs, _ = extract_objects(gi, bg=bg)
            out_objs, _ = extract_objects(go, bg=bg)
            
            if not in_objs:
                return None, None
            
            matches = match_objects(in_objs, out_objs, gi, go)
            
            for in_obj, out_obj, change in matches:
                if in_obj is None:
                    continue
                feats = object_features(in_obj, in_objs, gi)
                was_deleted = (out_obj is None or change == 'deleted')
                all_training.append((feats, was_deleted))
        
        if not all_training:
            return None, None
        
        rule = learn_delete_rule(all_training)
        if rule is None:
            return None, None
        
        # Apply rule
        results = []
        for tc in task['test']:
            gi = np.array(tc['input'])
            bg = int(np.argmax(np.bincount(gi.flatten())))
            objs, _ = extract_objects(gi, bg=bg)
            pred = gi.copy()
            
            for obj in objs:
                feats = object_features(obj, objs, gi)
                should_delete = False
                
                if rule['type'] == 'delete_if':
                    should_delete = (feats[rule['feature']] == rule['value'])
                elif rule['type'] == 'delete_by_color':
                    should_delete = (feats['color'] in rule['colors'])
                elif rule['type'] == 'delete_smaller_than':
                    should_delete = (feats['size'] < rule['threshold'])
                
                if should_delete:
                    for r, c in obj.cells:
                        pred[r, c] = bg
            
            results.append([pred.tolist()])
        
        if score_task(task, results):
            rule_desc = rule.get('feature', rule['type'])
            return results, f'LEARN:delete_{rule_desc}'
        return None, None
    
    def _try_keep_rules(self, task):
        """Try: keep only objects matching a condition, delete rest."""
        pairs = task['train']
        
        for pair in pairs:
            gi = np.array(pair['input'])
            go = np.array(pair['output'])
            if gi.shape != go.shape:
                return None, None
        
        bg = int(np.argmax(np.bincount(np.array(pairs[0]['input']).flatten())))
        
        # For each boolean feature, try "keep only objects with feature=True"
        bool_features = ['is_largest', 'is_smallest', 'is_unique_color', 'touches_border',
                         'is_square', 'is_tall', 'is_wide', 'is_hollow', 'larger_than_avg']
        
        for feat in bool_features:
            all_ok = True
            for pair in pairs:
                gi = np.array(pair['input'])
                go = np.array(pair['output'])
                objs, _ = extract_objects(gi, bg=bg)
                pred = np.full_like(gi, bg)
                
                for obj in objs:
                    feats = object_features(obj, objs, gi)
                    if feats[feat]:
                        for r, c in obj.cells:
                            pred[r, c] = gi[r, c]
                
                if not np.array_equal(pred, go):
                    all_ok = False
                    break
            
            if all_ok:
                results = []
                for tc in task['test']:
                    gi = np.array(tc['input'])
                    objs, _ = extract_objects(gi, bg=bg)
                    pred = np.full_like(gi, bg)
                    for obj in objs:
                        feats = object_features(obj, objs, gi)
                        if feats[feat]:
                            for r, c in obj.cells:
                                pred[r, c] = gi[r, c]
                    results.append([pred.tolist()])
                
                if score_task(task, results):
                    return results, f'LEARN:keep_{feat}'
        
        return None, None


# ═══════════════════════════════════════════════════════════
# INTEGRATION FUNCTION
# ═══════════════════════════════════════════════════════════

def try_per_object_learning(task):
    """Convenience function for arc_unified.py."""
    solver = PerObjectLearner()
    return solver.solve(task)


# ═══════════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    import json, os, sys, time
    
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  NeMo-WM Per-Object Learning Solver                         ║")
    print("║  Learn conditional per-object rules from training pairs      ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    
    data_dir = sys.argv[1] if len(sys.argv) > 1 else None
    
    if data_dir and os.path.exists(data_dir):
        files = sorted(os.listdir(data_dir))
        solver = PerObjectLearner()
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
        print(f"\n  Per-object learning: {len(solved)}/{len(files)} in {elapsed:.1f}s")
    else:
        print("  Usage: python arc_per_object_learn.py <data_dir>")
