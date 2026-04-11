"""
run_gc_ablation.py — Controlled GC Ablation for CORTEX-PE Phase 1 Distillation
================================================================================
Tests whether Gradient Centralization variants improve DINOv2 cosine distillation.

Design:
  - Domain:   maze frames (Phase 1, DINOv2 teacher, cosine similarity loss)
  - Steps:    3000 per run (30 min each)
  - Baseline: AdamW (current)
  - Variants: GCAdamW, GCC2AdamW, MCAdamW
  - Metric:   distill loss at steps 500/1000/2000/3000 + final AUROC on CWRU

Usage (run from CORTEX root):
    python run_gc_ablation.py

This script launches 4 sequential training runs and logs results to
./results/gc_ablation_YYYYMMDD.json for easy comparison.
"""

import subprocess
import json
import time
from datetime import datetime
from pathlib import Path

RESULTS_FILE = Path("./results/gc_ablation_{}.json".format(
    datetime.now().strftime("%Y%m%d_%H%M")))
RESULTS_FILE.parent.mkdir(exist_ok=True)

# ── Experiment matrix ──────────────────────────────────────────────────────
RUNS = [
    {
        "name":    "baseline_adamw",
        "gc":      "none",
        "label":   "AdamW (baseline)",
        "cmd_extra": []
    },
    {
        "name":    "gc_standard",
        "gc":      "standard",
        "label":   "GCAdamW (raw gradient)",
        "cmd_extra": ["--gc", "standard"]
    },
    {
        "name":    "gc_gcc2",
        "gc":      "gcc2",
        "label":   "GCC2AdamW (final update)",
        "cmd_extra": ["--gc", "gcc2"]
    },
    {
        "name":    "gc_moment",
        "gc":      "moment",
        "label":   "MCAdamW (moment centralization)",
        "cmd_extra": ["--gc", "moment"]
    },
]

BASE_CMD = [
    "python", "train_distillation.py",
    "--phase", "1",
    "--data", "./tiny-imagenet-200/train",          # maze frames — fast, well understood
    "--steps", "3000",
    "--batch", "32",
    "--lr", "1e-3",
    "--lambda-sigreg", "5.0",
    "--lambda-deep", "1.0",

]

CHECKPOINT_STEPS = [500, 1000, 2000, 3000]

def parse_log_line(line: str) -> dict | None:
    """Parse a training log line into a dict of metrics."""
    # Format: Step  500/3000 | total=X  distill=X  sigreg=X  deep=X  (Xms)
    if "Step" not in line or "distill=" not in line:
        return None
    try:
        parts = line.strip().split("|")
        step_str = parts[0].strip().split()[1].split("/")[0]
        metrics = {}
        for tok in parts[1].split():
            if "=" in tok:
                k, v = tok.split("=")
                try:
                    metrics[k.strip()] = float(v)
                except ValueError:
                    pass
        metrics["step"] = int(step_str)
        return metrics
    except Exception:
        return None


def run_experiment(run: dict) -> dict:
    """Run one training experiment and return parsed metrics."""
    out_dir = f"./checkpoints/gc_ablation/{run['name']}"
    cmd = BASE_CMD + run["cmd_extra"] + ["--out", out_dir]

    print(f"\n{'='*60}")
    print(f"  {run['label']}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    t0 = time.time()
    checkpoints = {}
    log_lines = []

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace"
        )
        for raw in proc.stdout:
            line = raw.rstrip()
            print(line)  # live output
            log_lines.append(line)
            m = parse_log_line(line)
            if m and m.get("step") in CHECKPOINT_STEPS:
                checkpoints[m["step"]] = m
        proc.wait()
    except Exception as e:
        print(f"ERROR: {e}")
        return {"name": run["name"], "error": str(e)}

    elapsed = time.time() - t0
    return {
        "name":       run["name"],
        "label":      run["label"],
        "gc":         run["gc"],
        "elapsed_s":  round(elapsed, 1),
        "checkpoints": checkpoints,
        "final_distill": checkpoints.get(3000, {}).get("distill"),
        "final_total":   checkpoints.get(3000, {}).get("total"),
    }


def summarise(results: list[dict]) -> None:
    """Print a clean comparison table."""
    print("\n" + "="*70)
    print("  GC ABLATION RESULTS — Phase 1 Distillation (3000 steps, maze)")
    print("="*70)
    print(f"{'Variant':<30} {'distill@500':>12} {'distill@1000':>12} "
          f"{'distill@3000':>12} {'time(s)':>8}")
    print("-"*70)
    baseline_d3k = None
    for r in results:
        if "error" in r:
            print(f"{r['name']:<30} ERROR: {r['error']}")
            continue
        cp = r.get("checkpoints", {})
        d500  = cp.get(500,  {}).get("distill", float("nan"))
        d1000 = cp.get(1000, {}).get("distill", float("nan"))
        d3000 = cp.get(3000, {}).get("distill", float("nan"))
        t     = r.get("elapsed_s", 0)
        if r["gc"] == "none":
            baseline_d3k = d3000
        delta = ""
        if baseline_d3k and r["gc"] != "none":
            pct = (d3000 - baseline_d3k) / baseline_d3k * 100
            delta = f"  ({pct:+.1f}%)"
        print(f"{r['label']:<30} {d500:>12.4f} {d1000:>12.4f} "
              f"{d3000:>12.4f}{delta}  {t:>7.0f}s")
    print("="*70)
    print("\nInterpretation:")
    print("  Negative delta = lower distill loss = GC helps")
    print("  Positive delta = GC hurts convergence")
    print("  < 0.5% difference = neutral (within noise)")


def main():
    print("CORTEX-PE Gradient Centralization Ablation")
    print(f"Results → {RESULTS_FILE}\n")

    all_results = []
    for run in RUNS:
        result = run_experiment(run)
        all_results.append(result)
        # Save incrementally
        with open(RESULTS_FILE, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✅ Saved intermediate results → {RESULTS_FILE}")

    summarise(all_results)
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Final results saved → {RESULTS_FILE}")


if __name__ == "__main__":
    main()
