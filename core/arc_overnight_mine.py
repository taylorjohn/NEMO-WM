"""
NeMo-WM Overnight Solver Mining Pipeline
=========================================
Runs all 112 OG solvers against all available datasets to:
1. Find tasks our solvers ALREADY solve (coverage analysis)
2. Discover patterns in UNSOLVED tasks (gap analysis)
3. Auto-generate new solver candidates from common patterns
4. Build a solve-rate heatmap per solver family

Usage:
    python arc_overnight_mine.py --data C:\Users\MeteorAI\Desktop\ARC-AGI-2\data --all-datasets
    
    # Quick test (ARC-AGI-2 only):
    python arc_overnight_mine.py --data C:\Users\MeteorAI\Desktop\ARC-AGI-2\data
    
    # Full overnight run (all datasets):
    python arc_overnight_mine.py --data C:\Users\MeteorAI\Desktop\ARC-AGI-2\data --all-datasets --mine-patterns
"""
import json
import os
import sys
import time
import numpy as np
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import our solvers
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arc_object_graph import ALL_OG_SOLVERS, extract_objects


# ═══════════════════════════════════════════════════════════
# PHASE 1: Run all solvers against all tasks
# ═══════════════════════════════════════════════════════════

def score_task(task, result):
    """Check if result matches test outputs."""
    if not result:
        return False
    for i, test_pair in enumerate(task['test']):
        expected = test_pair['output']
        if i >= len(result):
            return False
        predicted = result[i]
        if isinstance(predicted, list) and len(predicted) > 0:
            if isinstance(predicted[0], list) and isinstance(predicted[0][0], list):
                predicted = predicted[0]  # unwrap extra nesting
        if predicted != expected:
            return False
    return True


def run_solver_on_task(solver_name, solver_fn, task):
    """Run a single solver on a single task. Returns (solver_name, result, method) or None."""
    try:
        result, method = solver_fn(task)
        if result and score_task(task, result):
            return (solver_name, method)
    except Exception:
        pass
    return None


def run_all_solvers_on_task(task):
    """Run all OG solvers on a task. Returns list of (solver_name, method) that solved it."""
    solutions = []
    for name, fn in ALL_OG_SOLVERS:
        try:
            result, method = fn(task)
            if result and score_task(task, result):
                solutions.append((name, method))
        except Exception:
            pass
    return solutions


def load_tasks_from_dir(data_dir):
    """Load all JSON tasks from a directory."""
    tasks = {}
    data_path = Path(data_dir)
    if not data_path.exists():
        return tasks
    for f in sorted(data_path.glob('*.json')):
        try:
            task = json.loads(f.read_text())
            if 'train' in task and 'test' in task:
                tasks[f.stem] = task
        except:
            pass
    return tasks


# ═══════════════════════════════════════════════════════════
# PHASE 2: Pattern Analysis on Unsolved Tasks
# ═══════════════════════════════════════════════════════════

def analyze_task_properties(task):
    """Extract properties of a task for pattern mining."""
    props = {}
    train = task['train']
    gi = np.array(train[0]['input'])
    go = np.array(train[0]['output'])
    
    props['same_size'] = gi.shape == go.shape
    props['input_h'], props['input_w'] = gi.shape
    props['output_h'], props['output_w'] = go.shape
    
    if props['same_size']:
        props['size_change'] = 'same'
    elif go.shape[0] > gi.shape[0] or go.shape[1] > gi.shape[1]:
        props['size_change'] = 'grow'
    else:
        props['size_change'] = 'shrink'
    
    # Scale ratios
    if gi.shape[0] > 0 and gi.shape[1] > 0:
        props['h_ratio'] = go.shape[0] / gi.shape[0]
        props['w_ratio'] = go.shape[1] / gi.shape[1]
    
    bg = int(np.argmax(np.bincount(gi.flatten())))
    props['bg_color'] = bg
    props['n_input_colors'] = len(set(int(v) for v in gi.flatten()) - {bg})
    props['n_output_colors'] = len(set(int(v) for v in go.flatten()) - {bg})
    
    # Object properties
    try:
        objs, _ = extract_objects(gi)
        props['n_objects'] = len(objs)
        if objs:
            props['obj_sizes'] = sorted([o.size for o in objs])
            props['obj_colors'] = sorted(set(o.color for o in objs))
            props['all_same_shape'] = len(set(o.shape_hash for o in objs)) == 1
            props['all_same_color'] = len(set(o.color for o in objs)) == 1
            props['all_same_size'] = len(set(o.size for o in objs)) == 1
    except:
        props['n_objects'] = -1
    
    # Diff analysis
    if props['same_size']:
        diff = gi != go
        props['n_changed_pixels'] = int(np.sum(diff))
        props['pct_changed'] = round(props['n_changed_pixels'] / gi.size * 100, 1)
    
    # Symmetry check
    if props['same_size']:
        props['output_h_sym'] = np.array_equal(go, np.fliplr(go))
        props['output_v_sym'] = np.array_equal(go, np.flipud(go))
    
    # Consistent across pairs?
    props['consistent_sizes'] = all(
        np.array(p['input']).shape == gi.shape and np.array(p['output']).shape == go.shape
        for p in train
    )
    
    return props


def mine_unsolved_patterns(unsolved_tasks):
    """Analyze unsolved tasks to find common patterns."""
    print(f"\n{'='*60}")
    print(f"PATTERN MINING: Analyzing {len(unsolved_tasks)} unsolved tasks")
    print(f"{'='*60}")
    
    all_props = {}
    for tid, task in unsolved_tasks.items():
        try:
            all_props[tid] = analyze_task_properties(task)
        except:
            pass
    
    # Group by key properties
    by_size_change = defaultdict(list)
    by_n_objects = defaultdict(list)
    by_ratio = defaultdict(list)
    
    for tid, props in all_props.items():
        by_size_change[props.get('size_change', '?')].append(tid)
        n_obj = props.get('n_objects', -1)
        if n_obj >= 0:
            by_n_objects[min(n_obj, 10)].append(tid)
        hr = props.get('h_ratio', 0)
        wr = props.get('w_ratio', 0)
        if hr == wr and hr == int(hr) and hr > 0:
            by_ratio[int(hr)].append(tid)
    
    print(f"\nBy size change:")
    for k, v in sorted(by_size_change.items(), key=lambda x: -len(x[1])):
        print(f"  {k}: {len(v)} tasks")
    
    print(f"\nBy object count:")
    for k, v in sorted(by_n_objects.items()):
        label = f"{k}+" if k == 10 else str(k)
        print(f"  {label} objects: {len(v)} tasks")
    
    print(f"\nBy scale ratio:")
    for k, v in sorted(by_ratio.items()):
        if len(v) >= 3:
            print(f"  {k}x scale: {len(v)} tasks")
    
    # Find tasks with specific properties we haven't covered
    sym_tasks = [tid for tid, p in all_props.items() 
                 if p.get('output_h_sym') or p.get('output_v_sym')]
    same_shape_tasks = [tid for tid, p in all_props.items()
                        if p.get('all_same_shape') and p.get('n_objects', 0) >= 3]
    few_diff_tasks = [tid for tid, p in all_props.items()
                      if p.get('pct_changed', 100) < 20 and p.get('pct_changed', 0) > 0]
    
    print(f"\nHigh-potential unsolved clusters:")
    print(f"  Symmetric outputs: {len(sym_tasks)} tasks")
    print(f"  Same-shape objects: {len(same_shape_tasks)} tasks")
    print(f"  Small diff (<20% changed): {len(few_diff_tasks)} tasks")
    
    return all_props


# ═══════════════════════════════════════════════════════════
# PHASE 3: Cross-Dataset Solver Coverage
# ═══════════════════════════════════════════════════════════

def run_coverage_analysis(datasets, max_per_dataset=500):
    """Run all solvers against multiple datasets."""
    results = {}
    
    for ds_name, tasks in datasets.items():
        print(f"\n{'─'*50}")
        print(f"Dataset: {ds_name} ({len(tasks)} tasks)")
        print(f"{'─'*50}")
        
        solved = {}
        t0 = time.time()
        task_items = list(tasks.items())[:max_per_dataset]
        
        for i, (tid, task) in enumerate(task_items):
            solutions = run_all_solvers_on_task(task)
            if solutions:
                solved[tid] = solutions
            
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                print(f"  Progress: {i+1}/{len(task_items)} | Solved: {len(solved)} | {elapsed:.1f}s")
        
        elapsed = time.time() - t0
        
        # Solver family stats
        family_counts = Counter()
        for tid, sols in solved.items():
            for sname, method in sols:
                family_counts[sname] += 1
        
        print(f"\n  Results: {len(solved)}/{len(task_items)} solved ({100*len(solved)/max(len(task_items),1):.1f}%)")
        print(f"  Time: {elapsed:.1f}s ({1000*elapsed/max(len(task_items),1):.0f}ms/task)")
        
        if family_counts:
            print(f"  Top solvers:")
            for name, count in family_counts.most_common(10):
                print(f"    {name}: {count}")
        
        results[ds_name] = {
            'total': len(task_items),
            'solved': len(solved),
            'details': solved,
            'families': dict(family_counts),
            'time': elapsed
        }
    
    return results


# ═══════════════════════════════════════════════════════════
# PHASE 4: Auto-Generate Solver Candidates
# ═══════════════════════════════════════════════════════════

def auto_generate_candidates(unsolved_tasks, n_candidates=20):
    """Attempt to auto-discover new solver patterns from unsolved tasks."""
    print(f"\n{'='*60}")
    print(f"AUTO-DISCOVERY: Trying pattern templates on {len(unsolved_tasks)} tasks")
    print(f"{'='*60}")
    
    candidates = []
    
    # Template 1: Fixed color map (with specific colors)
    for tid, task in list(unsolved_tasks.items())[:200]:
        try:
            pairs = task['train']
            gi = np.array(pairs[0]['input'])
            go = np.array(pairs[0]['output'])
            if gi.shape != go.shape:
                continue
            
            # Try every possible (input_color, neighbor_count_8) → output_color mapping
            bg = int(np.argmax(np.bincount(gi.flatten())))
            rule = {}
            ok = True
            for p in pairs:
                gi2 = np.array(p['input'])
                go2 = np.array(p['output'])
                h, w = gi2.shape
                for r in range(h):
                    for c in range(w):
                        ic = int(gi2[r, c])
                        if ic == bg:
                            continue
                        # Count 8-connected same-color neighbors
                        nn8 = sum(1 for dr in [-1,0,1] for dc in [-1,0,1]
                                  if (dr != 0 or dc != 0)
                                  and 0 <= r+dr < h and 0 <= c+dc < w
                                  and gi2[r+dr, c+dc] == ic)
                        oc = int(go2[r, c])
                        key = (ic, nn8)
                        if key in rule and rule[key] != oc:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            
            if ok and rule and not all(k[0] == v for k, v in rule.items()):
                if len(set(rule.values())) > 1:
                    candidates.append({
                        'task_id': tid,
                        'type': 'context_recolor_8conn',
                        'rule': {str(k): v for k, v in rule.items()},
                    })
        except:
            pass
    
    # Template 2: Per-row or per-column transform
    for tid, task in list(unsolved_tasks.items())[:200]:
        try:
            pairs = task['train']
            gi = np.array(pairs[0]['input'])
            go = np.array(pairs[0]['output'])
            if gi.shape != go.shape:
                continue
            h, w = gi.shape
            
            # Check if each row is independently transformed
            row_maps = []
            for r in range(h):
                rmap = {}
                for c in range(w):
                    ic, oc = int(gi[r, c]), int(go[r, c])
                    if ic in rmap and rmap[ic] != oc:
                        rmap = None
                        break
                    rmap[ic] = oc
                row_maps.append(rmap)
            
            if all(rm is not None for rm in row_maps):
                # Check if row map depends on row index
                unique_maps = set(tuple(sorted(rm.items())) for rm in row_maps if rm)
                if len(unique_maps) > 1 and len(unique_maps) <= h:
                    candidates.append({
                        'task_id': tid,
                        'type': 'per_row_color_map',
                        'n_unique_maps': len(unique_maps),
                    })
        except:
            pass
    
    print(f"\n  Found {len(candidates)} potential solver candidates:")
    type_counts = Counter(c['type'] for c in candidates)
    for t, n in type_counts.most_common():
        print(f"    {t}: {n}")
    
    return candidates


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='NeMo-WM Overnight Solver Mining')
    parser.add_argument('--data', required=True, help='Path to ARC-AGI-2 data directory')
    parser.add_argument('--all-datasets', action='store_true', help='Run against all available datasets')
    parser.add_argument('--mine-patterns', action='store_true', help='Run pattern mining on unsolved tasks')
    parser.add_argument('--max-per-dataset', type=int, default=500, help='Max tasks per dataset')
    parser.add_argument('--output', default='mining_results.json', help='Output file for results')
    args = parser.parse_args()
    
    print("="*60)
    print("  NeMo-WM Overnight Solver Mining Pipeline")
    print(f"  Solvers: {len(ALL_OG_SOLVERS)}")
    print("="*60)
    
    t_start = time.time()
    
    # Load primary dataset
    datasets = {}
    training_dir = os.path.join(args.data, 'training')
    if os.path.exists(training_dir):
        datasets['arc_agi2_train'] = load_tasks_from_dir(training_dir)
    
    eval_dir = os.path.join(args.data, 'evaluation')  
    if os.path.exists(eval_dir):
        datasets['arc_agi2_eval'] = load_tasks_from_dir(eval_dir)
    
    # Load additional datasets
    if args.all_datasets:
        base = os.path.dirname(args.data)  # Parent of ARC-AGI-2/data
        base = os.path.dirname(base)       # Desktop level
        
        # RE-ARC
        rearc_dir = os.path.join(base, 're-arc', 're_arc_data', 're_arc', 'tasks')
        if os.path.exists(rearc_dir):
            rearc_tasks = {}
            for f in sorted(os.listdir(rearc_dir))[:100]:  # Sample 100
                if not f.endswith('.json'):
                    continue
                data = json.loads(open(os.path.join(rearc_dir, f)).read())
                if len(data) >= 4:
                    rearc_tasks[f'rearc_{f.replace(".json","")}'] = {
                        'train': [{'input': ex['input'], 'output': ex['output']} for ex in data[:3]],
                        'test': [{'input': data[3]['input'], 'output': data[3]['output']}]
                    }
            if rearc_tasks:
                datasets['re_arc'] = rearc_tasks
        
        # ConceptARC
        concept_dir = os.path.join(base, 'ConceptARC', 'corpus')
        if os.path.exists(concept_dir):
            concept_tasks = {}
            for cdir in sorted(os.listdir(concept_dir)):
                cpath = os.path.join(concept_dir, cdir)
                if not os.path.isdir(cpath):
                    continue
                for f in sorted(os.listdir(cpath)):
                    if f.endswith('.json'):
                        try:
                            task = json.loads(open(os.path.join(cpath, f)).read())
                            if 'train' in task:
                                concept_tasks[f'concept_{cdir}_{f.replace(".json","")}'] = task
                        except:
                            pass
            if concept_tasks:
                datasets['concept_arc'] = concept_tasks
        
        # Grid-Math
        gm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'grid_math_dataset.json')
        if not os.path.exists(gm_path):
            gm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'grid_math_train.json')
        if os.path.exists(gm_path):
            gm_data = json.loads(open(gm_path).read())
            gm_tasks = {item['id']: item['task'] for item in gm_data}
            datasets['grid_math'] = gm_tasks
    
    print(f"\nDatasets loaded:")
    for name, tasks in datasets.items():
        print(f"  {name}: {len(tasks)} tasks")
    
    # PHASE 1: Coverage analysis
    results = run_coverage_analysis(datasets, max_per_dataset=args.max_per_dataset)
    
    # PHASE 2: Pattern mining on unsolved
    if args.mine_patterns and 'arc_agi2_train' in datasets:
        solved_ids = set(results.get('arc_agi2_train', {}).get('details', {}).keys())
        unsolved = {tid: task for tid, task in datasets['arc_agi2_train'].items() if tid not in solved_ids}
        
        props = mine_unsolved_patterns(unsolved)
        candidates = auto_generate_candidates(unsolved)
        
        results['pattern_analysis'] = {
            'n_unsolved': len(unsolved),
            'n_candidates': len(candidates),
            'candidates': candidates[:50],  # Save top 50
        }
    
    # Save results
    total_time = time.time() - t_start
    results['meta'] = {
        'n_solvers': len(ALL_OG_SOLVERS),
        'total_time_s': round(total_time, 1),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    with open(args.output, 'w') as f:
        # Convert any non-serializable items
        def clean(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json.dump(results, f, indent=2, default=clean)
    
    print(f"\n{'='*60}")
    print(f"  MINING COMPLETE")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  Results saved to: {args.output}")
    print(f"{'='*60}")
    
    # Summary
    print(f"\n  SUMMARY:")
    for ds_name, res in results.items():
        if ds_name == 'meta' or ds_name == 'pattern_analysis':
            continue
        print(f"    {ds_name}: {res['solved']}/{res['total']} ({100*res['solved']/max(res['total'],1):.1f}%)")


if __name__ == '__main__':
    main()
