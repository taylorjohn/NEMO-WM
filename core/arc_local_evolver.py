"""
LOCAL Darwinian Evolver for ARC-AGI-2.
Instead of LLM-powered mutations, uses our solver library + combinatorial search.
No API costs. Runs on CPU.

Key idea: Instead of asking an LLM to mutate Python code,
we search over COMPOSITIONS of our existing 77 atomic operations.

Flow:
1. For each unsolved task, try all single solvers (already done)
2. Try all 2-step chains: apply solver A, then solver B
3. Try "partial match" solvers that get CLOSE but not exact
4. Combine partial matches to build complete solutions
"""
import json, os, numpy as np, time, sys
from collections import Counter, defaultdict
from arc_solver import score_task
from arc_object_graph import extract_objects, ALL_OG_SOLVERS

data_dir = '/home/claude/ARC-AGI-2/data/training'

def pixel_accuracy(predicted, target):
    """How many pixels match between predicted and target?"""
    pred = np.array(predicted)
    tgt = np.array(target)
    if pred.shape != tgt.shape:
        return 0.0
    return np.mean(pred == tgt)

def find_partial_matches(task, solvers, threshold=0.5):
    """Find solvers that get PARTIALLY close to the answer."""
    partials = []
    for name, fn in solvers:
        try:
            result, method = fn(task)
            if result:
                # Check accuracy on train outputs
                accuracies = []
                for i, p in enumerate(task['train']):
                    go = p['output']
                    if i < len(result) and result[i]:
                        pred = result[i][0] if isinstance(result[i][0], list) else result[i]
                        acc = pixel_accuracy(pred, go)
                        accuracies.append(acc)
                if accuracies:
                    avg_acc = np.mean(accuracies)
                    if avg_acc >= threshold:
                        partials.append((name, method, avg_acc, result))
        except:
            pass
    partials.sort(key=lambda x: x[2], reverse=True)
    return partials

def try_two_step_chain(task, solver_a_fn, solver_b_fn):
    """Apply solver A to get intermediate, then solver B to get final."""
    try:
        # Apply A to input
        result_a, _ = solver_a_fn(task)
        if not result_a:
            return None
        
        # Build intermediate task: input = A's output, output = original output
        inter_task = {
            'train': [],
            'test': task['test']
        }
        for i, p in enumerate(task['train']):
            if i < len(result_a) and result_a[i]:
                inter_input = result_a[i][0] if isinstance(result_a[i][0], list) else result_a[i]
                inter_task['train'].append({
                    'input': inter_input,
                    'output': p['output']
                })
        
        if len(inter_task['train']) != len(task['train']):
            return None
        
        # Apply B to intermediate
        result_b, method_b = solver_b_fn(inter_task)
        if result_b and score_task(task, result_b):
            return result_b
    except:
        pass
    return None

def evolve_task(task_id, task, max_chains=50):
    """Try to solve a task using compositional search."""
    
    # Step 1: Direct solve (already done in unified, but let's confirm)
    for name, fn in ALL_OG_SOLVERS:
        try:
            result, method = fn(task)
            if result and score_task(task, result):
                return result, f"DIRECT:{method}"
        except:
            pass
    
    # Step 2: Find partial matches
    partials = find_partial_matches(task, ALL_OG_SOLVERS, threshold=0.3)
    if partials:
        best_name, best_method, best_acc, _ = partials[0]
        # print(f"  Best partial: {best_name} ({best_acc:.1%})")
    
    # Step 3: Two-step chains (try top partial + all solvers)
    chains_tried = 0
    for name_a, fn_a in ALL_OG_SOLVERS[:20]:  # Limit for speed
        for name_b, fn_b in ALL_OG_SOLVERS[:20]:
            if chains_tried >= max_chains:
                break
            chains_tried += 1
            result = try_two_step_chain(task, fn_a, fn_b)
            if result:
                return result, f"CHAIN:{name_a}+{name_b}"
        if chains_tried >= max_chains:
            break
    
    return None, None

def run_evolver(num_tasks=None, verbose=True):
    """Run the local evolver on all unsolved tasks."""
    files = sorted(os.listdir(data_dir))
    
    # Known solved tasks
    known = {'070dd51e','08ed6ac7','0ca9ddb6','1f876c06','22168020','22eb0ac0',
        '2dc579da','3194b014','32597951','3906de3d','3af2c5a8','4258a5f9','4c4377d9',
        '5582e5ca','5bd6f4ac','62c24649','67e8384a','68b67ca3','913fb3ed','9565186b',
        '9ddd00f0','ac0a08a4','b230c067','b91ae062','bf699163','c909285e','c9f8e694',
        'cd3c21df','ce22a75a','ce602527','d9fac9be','ded97339','e57337a4','e729b7be',
        'e98196ab','ea32f347'}
    
    t0 = time.time()
    new_solves = []
    tasks_tried = 0
    
    for f in files:
        if not f.endswith('.json'): continue
        fn = f.replace('.json', '')
        if fn in known: continue
        if num_tasks and tasks_tried >= num_tasks: break
        
        task = json.load(open(os.path.join(data_dir, f)))
        tasks_tried += 1
        
        result, method = evolve_task(fn, task, max_chains=50)
        if result and score_task(task, result):
            new_solves.append((fn, method))
            if verbose:
                print(f"  ★ {fn}: {method}")
    
    elapsed = time.time() - t0
    print(f"\nLocal Evolver Results:")
    print(f"  Tasks tried: {tasks_tried}")
    print(f"  New solves:  {len(new_solves)}")
    print(f"  Time:        {elapsed:.0f}s")
    
    for fn, method in new_solves:
        print(f"  {fn}: {method}")
    
    return new_solves

if __name__ == '__main__':
    num = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    run_evolver(num_tasks=num)

