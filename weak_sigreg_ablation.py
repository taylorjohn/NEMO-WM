"""
weak_sigreg_ablation.py — SIGReg Variant Ablation for CORTEX-PE
================================================================
Tests whether Weak-SIGReg (Akbar, arXiv:2603.05924, ICLR 2026 Workshop GRaM)
improves on CORTEX-PE's current VICReg-style sigreg_loss.

Three variants tested:
  1. VICReg-style (current CORTEX-PE) — variance threshold + off-diagonal cov penalty
  2. Weak-SIGReg  — sketched covariance toward identity (Frobenius norm)
  3. Strong-SIGReg — full Epps-Pulley characteristic function matching

Background
----------
CORTEX-PE currently uses a VICReg-inspired formulation:
    L_sig = relu(1 - std(z)) + ||off_diag(cov(z))||_F^2

Weak-SIGReg (Akbar 2026) is simpler:
    S = random sketch matrix (128 x K, K << 128)
    L_weak = ||cov(z @ S) - I||_F^2

LeWM warning: SIGReg may overconstrain low-intrinsic-dimensionality tasks
(e.g. 2D maze navigation). Weak-SIGReg's softer constraint may be better.

Key question: does relaxing from full-covariance to sketched-covariance
constraint improve distillation quality + downstream MPC?

Usage:
    python weak_sigreg_ablation.py
    # runs 3 variants x 3000 steps each, ~90 min total
"""

import subprocess, json, time, torch, torch.nn.functional as F
from datetime import datetime
from pathlib import Path

# ── Loss implementations ──────────────────────────────────────────────────

def sigreg_vicreg(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Current CORTEX-PE implementation — VICReg-style.
    Variance threshold (soft hinge) + off-diagonal covariance penalty.
    O(batch × D²) where D=128.
    """
    std = torch.sqrt(z.var(dim=0) + eps)
    var_loss = torch.mean(F.relu(1 - std))

    z_c = z - z.mean(dim=0)
    cov = (z_c.T @ z_c) / (z.shape[0] - 1)
    cov_loss = cov.fill_diagonal_(0).pow(2).sum() / z.shape[1]
    return var_loss + cov_loss


def sigreg_weak(z: torch.Tensor, K: int = 32, seed: int = 42) -> torch.Tensor:
    """
    Weak-SIGReg (Akbar, arXiv:2603.05924).
    Projects z via fixed random sketch S (D×K), then penalises
    ||cov(z@S) - I||_F^2.

    K=32 means we constrain a 25% random subspace of the 128-D latent.
    O(batch × K²) — 16× cheaper than full covariance at D=128, K=32.

    The sketch S is fixed per training run (same seed) and registered as
    a buffer — never updated by the optimiser.
    """
    D = z.shape[1]
    # Fixed random sketch — same across all calls within a run
    torch.manual_seed(seed)
    S = torch.randn(D, K, device=z.device) / (K ** 0.5)  # (D, K)
    S = S / S.norm(dim=0, keepdim=True)  # normalise columns

    z_sketch = z @ S                                  # (N, K)
    z_c = z_sketch - z_sketch.mean(dim=0)             # centre
    cov = (z_c.T @ z_c) / (z.shape[0] - 1)           # (K, K)
    # ||cov - I||_F^2
    loss = (cov - torch.eye(K, device=z.device)).pow(2).sum()
    return loss / K  # normalise to be comparable across K values


def sigreg_epps_pulley(z: torch.Tensor, M: int = 16, T: int = 17) -> torch.Tensor:
    """
    Strong SIGReg — Epps-Pulley characteristic function matching.
    This is the original LeJEPA/LeWM formulation.
    Projects z to M random directions, tests each 1D projection against N(0,1)
    using the Epps-Pulley statistic.

    M=16, T=17 quadrature points (LeWM defaults).
    O(N × M × T) — more expensive but theoretically optimal.
    """
    N, D = z.shape
    # Normalise to zero mean, unit variance per dimension first
    z_norm = (z - z.mean(0)) / (z.std(0) + 1e-6)

    torch.manual_seed(0)
    # Random projection directions
    directions = F.normalize(torch.randn(M, D, device=z.device), dim=1)  # (M, D)
    projections = z_norm @ directions.T  # (N, M)

    # Epps-Pulley statistic: compare ECF to Gaussian CF at T quadrature points
    t_vals = torch.linspace(-4, 4, T, device=z.device)  # (T,)

    total_loss = torch.tensor(0.0, device=z.device)
    for m_idx in range(M):
        x = projections[:, m_idx]  # (N,)
        # ECF: E[exp(i*t*x)] ≈ (1/N) Σ exp(i*t*x_n)
        # Gaussian CF: exp(-t^2/2)
        # Use real part: E[cos(t*x)] vs exp(-t^2/2)
        ecf_real = (torch.cos(t_vals.unsqueeze(0) * x.unsqueeze(1))).mean(0)  # (T,)
        gcf = torch.exp(-0.5 * t_vals ** 2)  # (T,) Gaussian CF
        total_loss = total_loss + (ecf_real - gcf).pow(2).mean()

    return total_loss / M


# ── Verification ──────────────────────────────────────────────────────────

def verify_losses():
    """Confirm all three losses are non-trivially different on random input."""
    print("=== Loss Function Verification ===")
    z = torch.randn(32, 128)
    l1 = sigreg_vicreg(z).item()
    l2 = sigreg_weak(z, K=32).item()
    l3 = sigreg_epps_pulley(z, M=16, T=17).item()
    print(f"VICReg-style:   {l1:.4f}")
    print(f"Weak-SIGReg:    {l2:.4f}")
    print(f"Epps-Pulley:    {l3:.4f}")
    print()
    print("Collapse test (constant z → should penalise heavily):")
    z_collapsed = torch.zeros(32, 128)
    print(f"  VICReg-style: {sigreg_vicreg(z_collapsed):.4f}")
    print(f"  Weak-SIGReg:  {sigreg_weak(z_collapsed, K=32):.4f}")
    print(f"  Epps-Pulley:  {sigreg_epps_pulley(z_collapsed, M=16):.4f}")
    print()
    print("Isotropic Gaussian z → should be near-zero:")
    z_ideal = torch.randn(256, 128)
    print(f"  VICReg-style: {sigreg_vicreg(z_ideal):.4f}")
    print(f"  Weak-SIGReg:  {sigreg_weak(z_ideal, K=32):.4f}")
    print(f"  Epps-Pulley:  {sigreg_epps_pulley(z_ideal, M=16):.4f}")


# ── Ablation runner ───────────────────────────────────────────────────────

RESULTS_FILE = Path(f"./results/sigreg_ablation_{datetime.now():%Y%m%d_%H%M}.json")
RESULTS_FILE.parent.mkdir(exist_ok=True)

RUNS = [
    {
        "name": "vicreg_style",
        "label": "VICReg-style (current CORTEX-PE)",
        "sigreg": "vicreg",
        "cmd_extra": []          # no flag = default (current behaviour)
    },
    {
        "name": "weak_sigreg",
        "label": "Weak-SIGReg K=32 (Akbar 2026)",
        "sigreg": "weak",
        "cmd_extra": ["--sigreg", "weak"]
    },
    {
        "name": "epps_pulley",
        "label": "Strong SIGReg Epps-Pulley (LeWM)",
        "sigreg": "epps_pulley",
        "cmd_extra": ["--sigreg", "strong"]
    },
]

BASE_CMD = [
    "python", "train_distillation.py",
    "--phase", "1",
    "--data", "./tiny-imagenet-200/train",
    "--steps", "3000",
    "--batch", "32",
    "--lr", "1e-3",
    "--lambda-sigreg", "5.0",
    "--lambda-deep", "1.0",
    "--gc", "none",   # keep GC out of this ablation
]

CHECKPOINT_STEPS = [500, 1000, 2000, 3000]


def parse_log_line(line: str) -> dict | None:
    if "Step" not in line or "distill=" not in line:
        return None
    try:
        parts = line.strip().split("|")
        step = int(parts[0].strip().split()[1].split("/")[0])
        metrics = {"step": step}
        for tok in parts[1].split():
            if "=" in tok:
                k, v = tok.split("=")
                try:
                    metrics[k.strip()] = float(v)
                except ValueError:
                    pass
        return metrics
    except Exception:
        return None


def run_experiment(run: dict) -> dict:
    out_dir = f"./checkpoints/sigreg_ablation/{run['name']}"
    cmd = BASE_CMD + run["cmd_extra"] + ["--out", out_dir]
    print(f"\n{'='*60}\n  {run['label']}\n  {' '.join(cmd)}\n{'='*60}")
    t0 = time.time()
    checkpoints = {}
    try:
        import subprocess, os
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", env=env
        )
        for raw in proc.stdout:
            line = raw.rstrip()
            print(line)
            m = parse_log_line(line)
            if m and m.get("step") in CHECKPOINT_STEPS:
                checkpoints[m["step"]] = m
        proc.wait()
    except Exception as e:
        print(f"ERROR: {e}")
        return {"name": run["name"], "error": str(e)}
    return {
        "name": run["name"], "label": run["label"], "sigreg": run["sigreg"],
        "elapsed_s": round(time.time() - t0, 1),
        "checkpoints": checkpoints,
        "final_distill": checkpoints.get(3000, {}).get("distill"),
    }


def summarise(results):
    print("\n" + "="*72)
    print("  SIGReg VARIANT ABLATION — Phase 1 Distillation (3000 steps)")
    print("="*72)
    print(f"{'Variant':<35} {'d@500':>8} {'d@1000':>8} {'d@3000':>8} {'delta':>8}")
    print("-"*72)
    baseline = None
    for r in results:
        if "error" in r:
            print(f"{r['name']:<35} ERROR")
            continue
        cp = r.get("checkpoints", {})
        d = {s: cp.get(s, {}).get("distill", float("nan")) for s in [500,1000,3000]}
        if r["sigreg"] == "vicreg":
            baseline = d[3000]
        delta = ""
        if baseline and r["sigreg"] != "vicreg":
            pct = (d[3000] - baseline) / baseline * 100
            delta = f"  {pct:+.1f}%"
        print(f"{r['label']:<35} {d[500]:>8.4f} {d[1000]:>8.4f} {d[3000]:>8.4f}{delta}")
    print("="*72)
    print("\nLeWM warning: SIGReg may overconstrain low-intrinsic-dim tasks (maze).")
    print("Negative delta = lower distill loss = variant helps.")
    print("Decision threshold: |delta| < 0.5% = noise.")


def main():
    print("CORTEX-PE SIGReg Variant Ablation (arXiv:2603.05924)")
    verify_losses()
    all_results = []
    for run in RUNS:
        r = run_experiment(run)
        all_results.append(r)
        RESULTS_FILE.write_text(json.dumps(all_results, indent=2))
        print(f"\n✅ Saved → {RESULTS_FILE}")
    summarise(all_results)
    RESULTS_FILE.write_text(json.dumps(all_results, indent=2))
    print(f"\n✅ Final results → {RESULTS_FILE}")


if __name__ == "__main__":
    main()
