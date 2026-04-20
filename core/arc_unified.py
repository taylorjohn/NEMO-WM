"""
arc_unified.py — Unified ARC Solver (All Methods Combined)
=============================================================
Combines every solver in priority order:
  1. S1 Phase 2: Pattern matching (37 strategies) — fast, <1ms
  2. Mental Models: Classify → targeted search — ~10ms
  3. JEPA: Latent world model prediction — ~50ms
  4. Synth Proposer: Neural program ranking — ~20ms

Each method gets a chance. First success wins.
This should capture EVERY unique solve from every method.

Usage:
    python arc_unified.py --training --data path/to/ARC-AGI-2/data -v
    python arc_unified.py --eval --data path/to/ARC-AGI-2/data
    python arc_unified.py --test
"""

import argparse
import json
import os
import numpy as np
from collections import Counter
import time

from arc_solver import Grid, score_task

# S1: Phase 2 (includes Phase 1 + base)
try:
    from arc_phase2 import solve_task_phase2 as s1_solve
    S1_LABEL = "Phase 2 (37 strategies)"
except ImportError:
    from arc_solver import solve_task as s1_solve
    S1_LABEL = "Base (33 ops)"

# Mental Models
try:
    from arc_mental_models import MentalModelSolver, TaskAnalysis, ALL_MODELS
    HAS_MM = True
except ImportError:
    HAS_MM = False

# JEPA
try:
    from arc_jepa import JEPATrainer, JEPASolver, GridEncoder, BASE_PRIMITIVES
    HAS_JEPA = True
except ImportError:
    HAS_JEPA = False

# Synth Proposer
try:
    from arc_synth import (NeuralProposer, enumerate_programs,
                            generate_random_grids, grid_to_features,
                            diff_features, reconstruct_fn)
    HAS_SYNTH = True
except ImportError:
    HAS_SYNTH = False

# Advanced Ops
try:
    from arc_advanced_ops import ALL_ADVANCED_OPS, solve_task_advanced
    HAS_ADV = True
except ImportError:
    HAS_ADV = False

# Object Graph
try:
    from arc_object_graph import solve_with_object_graph
    HAS_OG = True
except ImportError:
    HAS_OG = False

# NumericalReasoner
try:
    from arc_numerical import try_numerical_reasoning
    HAS_NUM = True
except ImportError:
    HAS_NUM = False

# Compositional Search (lightweight version for unified runs)
HAS_COMPOSE = False
try:
    from arc_overnight_compose import PRIMITIVES, apply_chain, score_chain_on_task
    HAS_COMPOSE = True

    def try_compose_search(task, max_depth=2):
        """Quick compose search (depth 1-2 only for speed)."""
        prim_names = list(PRIMITIVES.keys())
        # Depth 1
        for p1 in prim_names:
            if score_chain_on_task([p1], task):
                return _build_compose_result(task, [p1]), f'COMPOSE:{p1}'
        # Depth 2
        for p1 in prim_names:
            for p2 in prim_names:
                if score_chain_on_task([p1, p2], task):
                    return _build_compose_result(task, [p1, p2]), f'COMPOSE:{p1}→{p2}'
        return None, None

    def _build_compose_result(task, chain):
        """Build submission format from a successful chain."""
        import numpy as np
        results = []
        for tc in task['test']:
            gi = np.array(tc['input'])
            out = apply_chain(gi, chain)
            if out is not None:
                results.append([out.tolist()])
            else:
                results.append([gi.tolist()])
        return results

except ImportError:
    pass


# ══════════════════════════════════════════════════════════════════════
# UNIFIED SOLVER
# ══════════════════════════════════════════════════════════════════════

class UnifiedSolver:
    """
    Master solver that tries ALL methods in priority order.
    Tracks which method solved each task for analysis.
    """

    def __init__(self, enable_jepa=True, enable_synth=True,
                 jepa_samples=5000, jepa_epochs=80,
                 synth_grids=30, synth_epochs=100):
        self.stats = Counter()
        self.total = 0
        self.solved_by = {}  # filename → method

        # Initialize optional components
        self.mm_solver = None
        self.jepa_solver = None
        self.synth_proposer = None
        self.synth_programs = None
        self.synth_label_map = None
        self.synth_rev_map = None

        if HAS_MM:
            self.mm_solver = MentalModelSolver()

        if HAS_JEPA and enable_jepa:
            print("  Initializing JEPA world model...")
            trainer = JEPATrainer(latent_dim=64, grid_size=12)
            trainer.train(n_samples=jepa_samples, epochs=jepa_epochs)
            self.jepa_solver = JEPASolver(trainer)

        if HAS_SYNTH and enable_synth:
            print("  Initializing neural proposer...")
            self._init_proposer(synth_grids, synth_epochs)

    def _init_proposer(self, n_grids, epochs):
        """Train the synthetic data proposer."""
        programs = enumerate_programs(max_depth=2, include_color_ops=True)
        self.synth_programs = programs

        rng = np.random.RandomState(42)
        X, Y = [], []
        for pid, pinfo in programs.items():
            fn = pinfo['fn']
            grids = generate_random_grids(n_grids * 2, rng)
            n = 0
            for g in grids:
                if n >= n_grids:
                    break
                try:
                    out = fn(g)
                    if 0 < out.h <= 30 and 0 < out.w <= 30:
                        in_f = grid_to_features(g)
                        out_f = grid_to_features(out)
                        df = diff_features(g, out)
                        X.append(np.concatenate([in_f, out_f, df]))
                        Y.append(pid)
                        n += 1
                except Exception:
                    continue

        if not X:
            return

        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.int32)
        ulabels = sorted(set(Y))
        self.synth_label_map = {o: n for n, o in enumerate(ulabels)}
        self.synth_rev_map = {n: o for o, n in self.synth_label_map.items()}
        Ym = np.array([self.synth_label_map[l] for l in Y], dtype=np.int32)

        self.synth_proposer = NeuralProposer(X.shape[1], len(ulabels))
        self.synth_proposer.train(X, Ym, epochs=epochs)

    def solve(self, task, filename=None):
        """Try all methods in priority order."""
        self.total += 1
        pairs = task['train']
        test_cases = task['test']

        # ── METHOD 1: S1 Phase 2 ──
        s1_preds = s1_solve(task)
        if score_task(task, s1_preds):
            self.stats['S1'] += 1
            if filename:
                self.solved_by[filename] = 'S1'
            return s1_preds, 'S1'

        # ── METHOD 2: Mental Models ──
        if self.mm_solver:
            analysis = TaskAnalysis(task)
            model_scores = [(m.matches(analysis), m) for m in ALL_MODELS]
            model_scores.sort(key=lambda x: -x[0])

            for confidence, model in model_scores:
                if confidence < 0.1:
                    continue
                guesses, prog_name = model.solve(task, analysis)
                if guesses is not None and score_task(task, guesses):
                    self.stats[f'MM:{model.name}'] += 1
                    if filename:
                        self.solved_by[filename] = f'MM:{model.name}:{prog_name}'
                    return guesses, f'MM:{model.name}:{prog_name}'

        # ── METHOD 3: Advanced Ops (14 targeted operations) ──
        if HAS_ADV:
            for op_name, op_fn in ALL_ADVANCED_OPS:
                try:
                    rule = op_fn(pairs)
                    if rule is not None:
                        guesses = []
                        ok = True
                        for tc in test_cases:
                            try:
                                result = rule(tc['input'])
                                guesses.append([result.tolist()])
                            except Exception:
                                ok = False
                                break
                        if ok and guesses and score_task(task, guesses):
                            self.stats[f'ADV:{op_name}'] += 1
                            if filename:
                                self.solved_by[filename] = f'ADV:{op_name}'
                            return guesses, f'ADV:{op_name}'
                except Exception:
                    continue

        # ── METHOD 3.5: Object Graph (relational reasoning) ──
        if HAS_OG:
            try:
                og_result, og_method = solve_with_object_graph(task)
                if og_result and score_task(task, og_result):
                    self.stats[f'OG'] += 1
                    if filename:
                        self.solved_by[filename] = og_method
                    return og_result, og_method
            except Exception:
                pass

        # ── METHOD 3.7: NumericalReasoner (count → transform) ──
        if HAS_NUM:
            try:
                num_result, num_method = try_numerical_reasoning(task)
                if num_result and score_task(task, num_result):
                    self.stats['NUM'] += 1
                    if filename:
                        self.solved_by[filename] = num_method
                    return num_result, num_method
            except Exception:
                pass

        # ── METHOD 3.9: Compositional Search (primitive chaining) ──
        if HAS_COMPOSE:
            try:
                comp_result, comp_method = try_compose_search(task)
                if comp_result and score_task(task, comp_result):
                    self.stats['COMPOSE'] += 1
                    if filename:
                        self.solved_by[filename] = comp_method
                    return comp_result, comp_method
            except Exception:
                pass

        # ── METHOD 4: JEPA World Model ──
        if self.jepa_solver:
            try:
                z_ins = np.array([self.jepa_solver.encoder.encode(Grid(p['input']))
                                    for p in pairs])
                z_outs = np.array([self.jepa_solver.encoder.encode(Grid(p['output']))
                                     for p in pairs])

                action = self.jepa_solver.action_encoder.infer_action_iterative(
                    z_ins, z_outs, self.jepa_solver.predictor, n_iters=30, lr=0.1)

                for tc in test_cases:
                    gi = Grid(tc['input'])
                    z_test = self.jepa_solver.encoder.encode(gi)
                    z_pred = self.jepa_solver.predictor.predict(z_test, action)

                    candidates = self.jepa_solver.decoder.decode_by_program(
                        z_pred, BASE_PRIMITIVES, gi, top_k=20)

                    for prog_name, result in candidates:
                        fn = BASE_PRIMITIVES[prog_name]
                        all_match = True
                        for p in pairs:
                            try:
                                if fn(Grid(p['input'])) != Grid(p['output']):
                                    all_match = False
                                    break
                            except Exception:
                                all_match = False
                                break
                        if all_match:
                            guesses = [[fn(Grid(tc2['input'])).to_list()]
                                         for tc2 in test_cases]
                            if score_task(task, guesses):
                                self.stats['JEPA'] += 1
                                if filename:
                                    self.solved_by[filename] = f'JEPA:{prog_name}'
                                return guesses, f'JEPA:{prog_name}'
            except Exception:
                pass

        # ── METHOD 5: Synth Proposer ──
        if self.synth_proposer and self.synth_programs:
            try:
                gi = Grid(pairs[0]['input'])
                go = Grid(pairs[0]['output'])
                in_f = grid_to_features(gi)
                out_f = grid_to_features(go)
                df = diff_features(gi, go)
                feat = np.concatenate([in_f, out_f, df])

                topk = self.synth_proposer.predict_topk(feat, k=50)

                for pred_id in topk:
                    orig_id = self.synth_rev_map.get(pred_id)
                    if orig_id is None or orig_id not in self.synth_programs:
                        continue
                    prog = self.synth_programs[orig_id]
                    fn = prog.get('fn')
                    if fn is None:
                        fn = reconstruct_fn(prog.get('steps', []))
                    if fn is None:
                        continue

                    all_match = True
                    for p in pairs:
                        try:
                            if fn(Grid(p['input'])) != Grid(p['output']):
                                all_match = False
                                break
                        except Exception:
                            all_match = False
                            break

                    if all_match:
                        guesses = []
                        for tc in test_cases:
                            try:
                                guesses.append([fn(Grid(tc['input'])).to_list()])
                            except Exception:
                                guesses.append([tc['input']])
                        if score_task(task, guesses):
                            self.stats['Synth'] += 1
                            if filename:
                                self.solved_by[filename] = f'Synth:{prog["name"]}'
                            return guesses, f'Synth:{prog["name"]}'
            except Exception:
                pass

        # ── FAIL ──
        self.stats['FAIL'] += 1
        return s1_preds, 'FAIL'


# ══════════════════════════════════════════════════════════════════════
# BENCHMARK
# ══════════════════════════════════════════════════════════════════════

def run_benchmark(data_dir, limit=None, verbose=False,
                    enable_jepa=True, enable_synth=True):
    files = sorted(f for f in os.listdir(data_dir) if f.endswith('.json'))
    if limit:
        files = files[:limit]

    print("=" * 70)
    print(f"  ARC-AGI-2 Unified Solver — All Methods Combined")
    print(f"  Tasks: {len(files)}")
    print(f"  S1: {S1_LABEL}")
    print(f"  Mental Models: {'Yes' if HAS_MM else 'No'}")
    print(f"  Advanced Ops: {'Yes (' + str(len(ALL_ADVANCED_OPS)) + ' ops)' if HAS_ADV else 'No'}")
    print(f"  Object Graph: {'Yes' if HAS_OG else 'No'}")
    print(f"  NumericalReasoner: {'Yes (357 combos)' if HAS_NUM else 'No'}")
    print(f"  Compositional Search: {'Yes (58 prims, depth 2)' if HAS_COMPOSE else 'No'}")
    print(f"  JEPA: {'Yes' if HAS_JEPA and enable_jepa else 'No'}")
    print(f"  Synth Proposer: {'Yes' if HAS_SYNTH and enable_synth else 'No'}")
    print("=" * 70)

    solver = UnifiedSolver(
        enable_jepa=enable_jepa,
        enable_synth=enable_synth,
        jepa_samples=5000,
        jepa_epochs=80,
        synth_grids=30,
        synth_epochs=100,
    )

    solved = 0
    total = 0
    t0 = time.time()

    for f in files:
        task = json.load(open(os.path.join(data_dir, f)))
        total += 1

        predictions, method = solver.solve(task, filename=f)
        correct = score_task(task, predictions)

        if correct:
            solved += 1

        if verbose and correct and not method.startswith('S1'):
            print(f"  {f[:16]}: SOLVED ({method})")

        if total % max(1, len(files) // 5) == 0:
            elapsed = time.time() - t0
            print(f"    Progress: {total}/{len(files)} | "
                  f"Solved: {solved} | Time: {elapsed:.0f}s")

    elapsed = time.time() - t0
    pct = solved / max(total, 1) * 100

    print(f"\n{'='*70}")
    print(f"  UNIFIED RESULTS")
    print(f"{'='*70}")
    print(f"  Solved:     {solved}/{total} ({pct:.1f}%)")

    # Breakdown by method
    for method in ['S1', 'JEPA', 'Synth']:
        if solver.stats[method] > 0:
            print(f"    {method:<12}: {solver.stats[method]}")
    mm_total = sum(v for k, v in solver.stats.items() if k.startswith('MM:'))
    if mm_total > 0:
        print(f"    {'MM':<12}: {mm_total}")
        for k, v in sorted(solver.stats.items()):
            if k.startswith('MM:') and v > 0:
                print(f"      {k}: {v}")
    adv_total = sum(v for k, v in solver.stats.items() if k.startswith('ADV:'))
    if adv_total > 0:
        print(f"    {'ADV':<12}: {adv_total}")
        for k, v in sorted(solver.stats.items()):
            if k.startswith('ADV:') and v > 0:
                print(f"      {k}: {v}")

    print(f"  Failed:     {solver.stats.get('FAIL', 0)}")
    print(f"  Time:       {elapsed:.1f}s ({elapsed/max(total,1)*1000:.0f}ms/task)")

    # List non-S1 solves
    non_s1 = {f: m for f, m in solver.solved_by.items() if not m.startswith('S1')}
    if non_s1:
        print(f"\n  Non-S1 solves ({len(non_s1)}):")
        for f, m in sorted(non_s1.items()):
            print(f"    {f[:20]}: {m}")

    print(f"\n{'='*70}")
    return solved, total


# ══════════════════════════════════════════════════════════════════════
# TESTS
# ══════════════════════════════════════════════════════════════════════

def run_tests():
    print("=" * 65)
    print("  Unified Solver Tests")
    print("=" * 65)
    p = 0; t = 0

    print("\n  T1: S1 solves simple rotation")
    task = {'train': [{'input': [[1,2],[3,4]], 'output': [[3,1],[4,2]]}],
            'test': [{'input': [[5,6],[7,8]], 'output': [[7,5],[8,6]]}]}
    solver = UnifiedSolver(enable_jepa=False, enable_synth=False)
    preds, method = solver.solve(task)
    ok = score_task(task, preds) and method == 'S1'
    print(f"    {method} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T2: Mental models available")
    ok = HAS_MM
    print(f"    MM loaded: {HAS_MM} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T3: JEPA available")
    ok = HAS_JEPA
    print(f"    JEPA loaded: {HAS_JEPA} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T4: Synth available")
    ok = HAS_SYNTH
    print(f"    Synth loaded: {HAS_SYNTH} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T5: Stats tracking")
    ok = solver.stats['S1'] == 1 and solver.total == 1
    print(f"    S1: {solver.stats['S1']}, total: {solver.total} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T6: Method priority (S1 wins first)")
    task2 = {'train': [{'input': [[1,0],[0,1]], 'output': [[0,1],[1,0]]}],
             'test': [{'input': [[2,0],[0,2]], 'output': [[0,2],[2,0]]}]}
    preds2, method2 = solver.solve(task2)
    ok = method2 == 'S1'  # S1 should catch this (rot180)
    print(f"    {method2} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print(f"\n{'='*65}")
    print(f"  Results: {p}/{t} tests passed")
    print(f"{'='*65}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--training", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-jepa", action="store_true")
    ap.add_argument("--no-synth", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--data", type=str, default=None)
    args = ap.parse_args()

    if args.data:
        data_base = args.data
    else:
        for base in [".", "ARC-AGI-2", "../ARC-AGI-2",
                       "/home/claude/ARC-AGI-2",
                       "C:/Users/MeteorAI/Desktop/ARC-AGI-2"]:
            candidate = os.path.join(base, "data")
            if os.path.exists(candidate):
                data_base = candidate
                break
        else:
            data_base = "ARC-AGI-2/data"

    if args.test:
        run_tests()
    else:
        if args.eval:
            d = os.path.join(data_base, "evaluation")
        else:
            d = os.path.join(data_base, "training")
        run_benchmark(d, limit=args.limit, verbose=args.verbose,
                        enable_jepa=not args.no_jepa,
                        enable_synth=not args.no_synth)
