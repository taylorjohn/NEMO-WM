"""
run_all_tests.py — Full NeMo-WM Regression Test Suite
=======================================================
Runs ALL test files in one command, collects results,
saves a report with pass/fail counts and timing.

Usage:
    python run_all_tests.py            # run all tests
    python run_all_tests.py --quick    # skip slow tests
    python run_all_tests.py --save     # save report to file
"""

import argparse
import subprocess
import sys
import time
import re
from pathlib import Path
from datetime import datetime

# All test files and their test commands
TEST_SUITE = [
    ("Autonomous Loop",          "autonomous_loop.py --test"),
    ("Benchmark 20Q",            "benchmark_20q.py"),
    ("AGI Capability",           "agi_test.py --test"),
    ("AGI Gap Closure",          "agi_gap_closure.py --test"),
    ("Construction Grammar",     "construction_grammar.py --test"),
    ("Semantic Dementia",        "semantic_dementia.py --test"),
    ("Instruction Following",    "instruction_follower.py --test"),
    ("Cognitive Quick Wins",     "cognitive_quickwins.py --test"),
    ("Cognitive Extensions",     "cognitive_extensions.py --test"),
    ("FAISS Schema",             "faiss_schema.py --test"),
    ("Audio Encoder",            "audio_encoder.py --test"),
    ("Tactile Encoder",          "tactile_encoder.py --test"),
    ("Chemical Encoder",         "chemical_encoder.py --test"),
    ("Object Tracker",           "object_tracker.py --test"),
    ("Physics Gym Discovery",    "physics_gym_discovery.py --test"),
    ("NeuroTransformer",         "neurotransformer.py --test"),
    ("NeuroTransformer Benchmark", "neurotransformer_benchmark.py --test"),
    ("NeuroTransformer Trained", "neurotransformer_train.py --test"),
    ("Dual System Transition",   "dual_system_transition.py --test"),
    ("Compositional Narrator",   "compositional_narrator.py --test"),
    ("Vocab Trainer",            "vocab_trainer.py --test"),
    ("NPA Encoder",              "npa_encoder.py --test"),
    ("NPA Crossover",            "npa_crossover_demo.py --test"),
    ("EWC Continual",            "ewc_continual.py --test"),
    ("Gym Connector",            "gym_connector.py --test"),
]

# Slow tests (skip with --quick)
SLOW_TESTS = {
    "NPA Crossover",
    "NeuroTransformer Benchmark",
    "NeuroTransformer Trained",
    "Physics Gym Discovery",
}


def parse_results(output):
    """Extract pass/total from test output."""
    # Look for "Results: X/Y tests passed" or "X/Y"
    patterns = [
        r'Results?:\s*(\d+)/(\d+)',
        r'(\d+)/(\d+)\s*tests?\s*passed',
        r'TOTAL[:\s]+(\d+)/(\d+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, output)
        if match:
            return int(match.group(1)), int(match.group(2))

    # Count PASS/FAIL lines
    passes = len(re.findall(r'\bPASS\b', output))
    fails = len(re.findall(r'\bFAIL\b', output))
    if passes + fails > 0:
        return passes, passes + fails

    return 0, 0


def run_test(name, command, verbose=False):
    """Run a single test file and capture results."""
    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable] + command.split(),
            capture_output=True, text=True, timeout=120,
            cwd=str(Path.cwd()),
            env={**__import__('os').environ, 'PYTHONIOENCODING': 'utf-8'})

        output = result.stdout + result.stderr
        elapsed = time.time() - t0
        passed, total = parse_results(output)

        # Check for crashes
        if result.returncode != 0 and total == 0:
            return {
                "name": name,
                "passed": 0, "total": 1,
                "time": elapsed,
                "status": "CRASH",
                "error": result.stderr[-200:] if result.stderr else "unknown",
            }

        return {
            "name": name,
            "passed": passed, "total": total,
            "time": elapsed,
            "status": "OK" if passed == total else "PARTIAL",
            "error": None,
        }
    except subprocess.TimeoutExpired:
        return {
            "name": name,
            "passed": 0, "total": 1,
            "time": 120,
            "status": "TIMEOUT",
            "error": "exceeded 120s",
        }
    except Exception as e:
        return {
            "name": name,
            "passed": 0, "total": 1,
            "time": time.time() - t0,
            "status": "ERROR",
            "error": str(e)[:100],
        }


def run_suite(quick=False, verbose=False, save=False):
    print("=" * 70)
    print("  NeMo-WM Full Regression Test Suite")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Mode: {'QUICK' if quick else 'FULL'}")
    print("=" * 70)

    suite = TEST_SUITE
    if quick:
        suite = [(n, c) for n, c in suite if n not in SLOW_TESTS]
        print(f"  Skipping {len(SLOW_TESTS)} slow tests\n")

    results = []
    total_passed = 0
    total_tests = 0
    total_time = 0

    print(f"\n  {'#':>3} │ {'Test':<30} │ {'Result':>8} │ "
          f"{'Time':>6} │ {'Status':>8}")
    print(f"  {'─'*3}─┼─{'─'*30}─┼─{'─'*8}─┼─{'─'*6}─┼─{'─'*8}")

    for i, (name, command) in enumerate(suite):
        result = run_test(name, command, verbose)
        results.append(result)

        total_passed += result["passed"]
        total_tests += result["total"]
        total_time += result["time"]

        ratio = f"{result['passed']}/{result['total']}"

        if result["status"] == "OK":
            status_str = "✓ PASS"
        elif result["status"] == "PARTIAL":
            status_str = "○ PART"
        elif result["status"] == "CRASH":
            status_str = "✗ CRASH"
        elif result["status"] == "TIMEOUT":
            status_str = "⏱ TIMEOUT"
        else:
            status_str = "✗ ERROR"

        print(f"  {i+1:>3} │ {name:<30} │ {ratio:>8} │ "
              f"{result['time']:>5.1f}s │ {status_str:>8}")

    # Summary
    pass_rate = total_passed / max(total_tests, 1) * 100

    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Tests run:    {len(results)}")
    print(f"  Total passed: {total_passed}/{total_tests} ({pass_rate:.1f}%)")
    print(f"  Total time:   {total_time:.1f}s")

    # Failures
    failures = [r for r in results if r["status"] != "OK"]
    if failures:
        print(f"\n  Failures ({len(failures)}):")
        for f in failures:
            print(f"    {f['name']}: {f['passed']}/{f['total']} "
                  f"({f['status']})")
            if f["error"]:
                print(f"      → {f['error'][:80]}")
    else:
        print(f"\n  ✓ ALL TESTS PASSING!")

    print(f"\n{'='*70}")

    # Save report
    if save:
        report_path = f"outputs/test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, 'w') as f:
            f.write(f"NeMo-WM Test Report\n")
            f.write(f"Date: {datetime.now().isoformat()}\n")
            f.write(f"Total: {total_passed}/{total_tests} ({pass_rate:.1f}%)\n")
            f.write(f"Time: {total_time:.1f}s\n\n")
            for r in results:
                f.write(f"{r['name']}: {r['passed']}/{r['total']} "
                        f"({r['status']}) [{r['time']:.1f}s]\n")
                if r["error"]:
                    f.write(f"  ERROR: {r['error']}\n")
        print(f"  Report saved: {report_path}")

    return total_passed, total_tests


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                     help="Skip slow tests")
    ap.add_argument("--save", action="store_true",
                     help="Save report to file")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    run_suite(quick=args.quick, verbose=args.verbose, save=args.save)
