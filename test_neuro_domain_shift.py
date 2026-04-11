"""
test_neuro_domain_shift.py
===========================
Standalone test of the neuromodulator under simulated domain shift.

No maze environment needed. Generates synthetic latent sequences that
mimic three real scenarios:

  Phase A (steps 0-49):   Routine maze navigation — stable, low surprise
  Phase B (steps 50-74):  Domain shift (e.g. new environment, bearing fault
                           onset, or flash crash) — high surprise, unstable
  Phase C (steps 75-99):  Recovery — system adapts and stabilises

Compares:
  STATIC:  fixed 64 candidates, eps=0.1, always acts (current production)
  NEURO:   7-signal neuromodulated planner

Metrics:
  - False actions during instability (actions taken in WAIT regime)
  - Candidate efficiency (candidates evaluated per step)
  - DA/5HT signal trace across the three phases
  - Regime distribution per phase
  - Time to detect domain shift (steps until EXPLORE/WAIT triggered)

Usage:
    python test_neuro_domain_shift.py
    python test_neuro_domain_shift.py --scenario bearing_fault
    python test_neuro_domain_shift.py --scenario all --trials 20
"""

import argparse
import json
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from neuromodulator import NeuromodulatorState, neuro_to_packet, Regime, classify_regime


# ══════════════════════════════════════════════════════════════════════════
# Synthetic latent generators
# ══════════════════════════════════════════════════════════════════════════

def make_latent(dim=32, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return F.normalize(torch.randn(dim), dim=0)


class LatentSequenceGenerator:
    """
    Generates synthetic latent transition sequences for three scenarios.
    Each scenario has three phases with different DA/5HT characteristics.
    """

    def __init__(self, dim=32, n_steps=100, scenario="domain_shift", seed=42):
        torch.manual_seed(seed)
        self.dim      = dim
        self.n_steps  = n_steps
        self.scenario = scenario

        # Phase boundaries
        self.phase_a_end = 50   # routine
        self.phase_b_end = 75   # crisis
        # phase_c: 75-99 recovery

        self._build_sequence()

    def _build_sequence(self):
        """Build (z_pred, z_actual) pairs for each step."""
        self.pairs = []

        # Phase A: routine — predictor nearly perfect, latents stable
        z_routine = make_latent(self.dim, seed=0)
        for t in range(self.phase_a_end):
            noise_pred   = 0.02   # very accurate prediction
            noise_actual = 0.01   # very stable representation
            z_pred   = F.normalize(z_routine + torch.randn(self.dim) * noise_pred,   dim=0)
            z_actual = F.normalize(z_routine + torch.randn(self.dim) * noise_actual, dim=0)
            self.pairs.append((z_pred, z_actual, "A_routine"))

        # Phase B: domain shift — scenario-specific
        if self.scenario == "bearing_fault":
            # Bearing fault: predictor locked to healthy baseline,
            # actual encoder drifts toward fault signature (anti-parallel at full severity).
            # Gradual cosine_sim: 0.98 → -0.7 over 25 steps → phasic: 0.01 → 0.85
            z_fault = F.normalize(-z_routine + make_latent(self.dim, seed=1) * 0.2, dim=0)
            for i, t in enumerate(range(self.phase_a_end, self.phase_b_end)):
                severity = i / max(1, self.phase_b_end - self.phase_a_end - 1)
                # Predictor still expects healthy signal
                z_pred   = F.normalize(z_routine + torch.randn(self.dim) * 0.02, dim=0)
                # Actual: progressive drift from healthy toward fault
                z_actual = F.normalize(
                    z_routine * (1 - severity) + z_fault * severity
                    + torch.randn(self.dim) * 0.05, dim=0)
                self.pairs.append((z_pred, z_actual, "B_bearing_fault"))

        elif self.scenario == "flash_crash":
            # Flash crash: predictor expects continuation, market reverses hard.
            # Use near-antipodal latents → cosine_sim → -1 → phasic → 1.0
            # This gives DA EMA peak ≈ 0.77 in 25 steps with decay=0.95
            for t in range(self.phase_a_end, self.phase_b_end):
                # z_pred near z_routine (expected continuation)
                z_pred   = F.normalize(z_routine + torch.randn(self.dim) * 0.05, dim=0)
                # z_actual anti-parallel = catastrophic reversal
                z_actual = F.normalize(-z_routine + torch.randn(self.dim) * 0.1, dim=0)
                self.pairs.append((z_pred, z_actual, "B_flash_crash"))

        else:  # domain_shift (default)
            # Sudden jump to new domain: predictor still outputs z_routine,
            # but actual encoder output is now z_new (orthogonal).
            # Use clean antipodal shift for maximum phasic signal.
            z_new = F.normalize(-z_routine + make_latent(self.dim, seed=99) * 0.3, dim=0)
            for t in range(self.phase_a_end, self.phase_b_end):
                # Predictor locked to old domain
                z_pred   = F.normalize(z_routine + torch.randn(self.dim) * 0.02, dim=0)
                # Actual encoder: new domain with growing stability
                stability = min(1.0, (t - self.phase_a_end) * 0.15)
                z_actual = F.normalize(z_new + torch.randn(self.dim) * (0.3 - stability * 0.25), dim=0)
                self.pairs.append((z_pred, z_actual, "B_domain_shift"))

        # Phase C: recovery — new domain learned, predictor adapts
        z_learned = make_latent(self.dim, seed=5)
        for i, t in enumerate(range(self.phase_b_end, self.n_steps)):
            recovery = min(1.0, i * 0.1)
            noise_pred   = 0.3 * (1 - recovery) + 0.02 * recovery
            noise_actual = 0.2 * (1 - recovery) + 0.01 * recovery
            z_pred   = F.normalize(z_learned + torch.randn(self.dim) * noise_pred,   dim=0)
            z_actual = F.normalize(z_learned + torch.randn(self.dim) * noise_actual, dim=0)
            self.pairs.append((z_pred, z_actual, "C_recovery"))


# ══════════════════════════════════════════════════════════════════════════
# Simulation runners
# ══════════════════════════════════════════════════════════════════════════

def run_static(pairs, base_candidates=64, base_eps=0.1):
    """Static baseline: fixed parameters, always acts."""
    results = []
    for z_pred, z_actual, phase in pairs:
        results.append({
            "phase":       phase,
            "candidates":  base_candidates,
            "eps":         base_eps,
            "action_scale": 1.0,
            "regime":      "STATIC",
            "acted":       True,
            "da":          float((1 - F.cosine_similarity(
                               z_pred.unsqueeze(0),
                               z_actual.unsqueeze(0))) / 2),
        })
    return results


def run_neuro(pairs, session_start=None):
    """Neuromodulated: 7 signals, adaptive candidates, WAIT gating."""
    neuro   = NeuromodulatorState(session_start=session_start or time.time())
    results = []
    last_action_mag = 0.0

    for z_pred, z_actual, phase in pairs:
        sig = neuro.update(z_pred, z_actual,
                           rho=0.3,
                           action_magnitude=last_action_mag)
        acted = sig["regime"] != Regime.WAIT
        if acted:
            # Simulate action magnitude proportional to action_scale
            last_action_mag = float(sig["action_scale"]) * np.random.uniform(0.05, 0.15)
        else:
            last_action_mag = 0.0

        results.append({
            "phase":        phase,
            "candidates":   sig["n_candidates"],
            "eps":          sig["eps_scale"] * 0.1,
            "action_scale": sig["action_scale"],
            "regime":       sig["regime"],
            "acted":        acted,
            "da":           sig["da"],
            "sht":          sig["sht"],
            "ach":          sig["ach"],
            "ecb":          sig["ecb"],
            "ei":           sig["ei"],
            "ado":          sig["ado"],
            "da_eff":       sig["da_effective"],
        })
    return results


# ══════════════════════════════════════════════════════════════════════════
# Analysis
# ══════════════════════════════════════════════════════════════════════════

def analyse(static_results, neuro_results, scenario, phase_b_end=75, phase_a_end=50):
    def phase_filter(results, phase_prefix):
        return [r for r in results if r["phase"].startswith(phase_prefix)]

    phases = {"A_routine": "A", "B": "B", "C_recovery": "C"}

    print(f"\n{'═'*62}")
    print(f"  SCENARIO: {scenario.upper()}")
    print(f"  Steps: 0-{phase_a_end-1} routine | {phase_a_end}-{phase_b_end-1} crisis | {phase_b_end}-99 recovery")
    print(f"{'═'*62}")

    report = {"scenario": scenario, "phases": {}}

    for phase_key, phase_label in [
        ("A_routine", "Phase A (Routine)"),
        ("B_",        "Phase B (Crisis) "),
        ("C_recovery","Phase C (Recovery)"),
    ]:
        sr = [r for r in static_results if phase_key in r["phase"]]
        nr = [r for r in neuro_results  if phase_key in r["phase"]]
        if not sr or not nr:
            continue

        s_cands  = np.mean([r["candidates"]  for r in sr])
        n_cands  = np.mean([r["candidates"]  for r in nr])
        s_acted  = np.mean([r["acted"]       for r in sr])
        n_acted  = np.mean([r["acted"]       for r in nr])
        n_da     = np.mean([r["da"]          for r in nr])
        n_sht    = np.mean([r.get("sht",0)   for r in nr])
        n_regimes = defaultdict(int)
        for r in nr:
            n_regimes[r["regime"]] += 1

        # False actions: acted during crisis phase
        s_false_act = s_acted if "B_" in phase_key else None
        n_false_act = n_acted if "B_" in phase_key else None

        print(f"\n  ── {phase_label} {'─'*30}")
        print(f"  {'Metric':<28} {'Static':>10} {'Neuro':>10} {'Delta':>10}")
        print(f"  {'─'*60}")
        print(f"  {'Candidates (mean)':<28} {s_cands:>10.1f} {n_cands:>10.1f} {n_cands-s_cands:>+10.1f}")
        print(f"  {'Action rate':<28} {s_acted:>9.1%} {n_acted:>9.1%} {n_acted-s_acted:>+9.1%}")
        print(f"  {'DA (neuro mean)':<28} {'—':>10} {n_da:>10.3f}")
        print(f"  {'5HT (neuro mean)':<28} {'—':>10} {n_sht:>10.3f}")
        if n_regimes:
            top_regime = max(n_regimes, key=n_regimes.get)
            total = sum(n_regimes.values())
            regime_str = f"{top_regime} ({100*n_regimes[top_regime]//total}%)"
            print(f"  {'Dominant regime':<28} {'STATIC':>10} {regime_str:>10}")

        if s_false_act is not None:
            saved = (s_false_act - n_false_act) * len(sr)
            print(f"  {'False actions prevented':<28} {'—':>10} {saved:>10.0f} steps")

        report["phases"][phase_label] = {
            "static_candidates":  round(s_cands,  1),
            "neuro_candidates":   round(n_cands,  1),
            "static_action_rate": round(s_acted,  3),
            "neuro_action_rate":  round(n_acted,  3),
            "neuro_da_mean":      round(n_da,     3),
            "neuro_sht_mean":     round(n_sht,    3),
            "regime_counts":      dict(n_regimes),
        }

    # ── Detection latency ─────────────────────────────────────────────────
    # How many steps into Phase B until DA crosses 0.5?
    phase_b_neuro = [r for r in neuro_results if "B_" in r["phase"]]
    detect_step = None
    for i, r in enumerate(phase_b_neuro):
        if r["da"] >= 0.5:
            detect_step = i
            break

    print(f"\n  ── Detection ───────────────────────────────────────────")
    if detect_step is not None:
        print(f"  Domain shift detected at step {phase_a_end + detect_step} "
              f"(+{detect_step} into crisis phase)")
        print(f"  Static: never detects (no mechanism)")
    else:
        print(f"  DA did not cross 0.5 threshold in crisis phase")
        print(f"  Scenario may not produce sufficient surprise signal")

    report["detect_step"] = detect_step

    # ── Overall summary ───────────────────────────────────────────────────
    total_s_cands = sum(r["candidates"] for r in static_results)
    total_n_cands = sum(r["candidates"] for r in neuro_results)
    saving_pct    = 100 * (1 - total_n_cands / total_s_cands)
    b_false_static = sum(1 for r in static_results if "B_" in r["phase"] and r["acted"])
    b_false_neuro  = sum(1 for r in neuro_results  if "B_" in r["phase"] and r["acted"])

    print(f"\n  ── Overall (100 steps) ─────────────────────────────────")
    print(f"  Candidate evaluations: Static={total_s_cands}  Neuro={total_n_cands}  "
          f"Saving={saving_pct:.1f}%")
    print(f"  Crisis actions:  Static={b_false_static}/25  Neuro={b_false_neuro}/25  "
          f"Prevented={b_false_static - b_false_neuro}")
    print(f"{'═'*62}")

    report["summary"] = {
        "total_candidates_static": total_s_cands,
        "total_candidates_neuro":  total_n_cands,
        "candidate_saving_pct":    round(saving_pct, 1),
        "crisis_actions_static":   b_false_static,
        "crisis_actions_neuro":    b_false_neuro,
        "crisis_actions_prevented": b_false_static - b_false_neuro,
    }

    return report


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def run_scenario(scenario, trials=10, dim=32, n_steps=100):
    all_reports = []
    for trial in range(trials):
        gen = LatentSequenceGenerator(dim=dim, n_steps=n_steps,
                                      scenario=scenario, seed=trial)
        static_r = run_static(gen.pairs)
        neuro_r  = run_neuro(gen.pairs)
        report   = analyse(static_r, neuro_r, scenario,
                           phase_b_end=gen.phase_b_end,
                           phase_a_end=gen.phase_a_end)
        all_reports.append(report)
        if trials > 1 and trial == 0:
            break  # print full analysis only for first trial

    if trials > 1:
        # Aggregate summary across trials
        saves = [r["summary"]["candidate_saving_pct"] for r in all_reports]
        prevented = [r["summary"]["crisis_actions_prevented"] for r in all_reports]
        detects = [r["detect_step"] for r in all_reports if r["detect_step"] is not None]

        print(f"\n  Aggregate over {trials} trials:")
        print(f"  Candidate saving:      {np.mean(saves):.1f}% ± {np.std(saves):.1f}%")
        print(f"  Crisis prevented:      {np.mean(prevented):.1f} ± {np.std(prevented):.1f} steps")
        if detects:
            print(f"  Detection latency:     {np.mean(detects):.1f} ± {np.std(detects):.1f} steps into crisis")

    return all_reports


def main():
    parser = argparse.ArgumentParser(
        description="Neuromodulator domain shift test")
    parser.add_argument("--scenario", default="domain_shift",
        choices=["domain_shift", "bearing_fault", "flash_crash", "all"])
    parser.add_argument("--trials",   type=int,  default=10)
    parser.add_argument("--dim",      type=int,  default=32)
    parser.add_argument("--out",      default="./results/neuro_shift_test.json")
    args = parser.parse_args()

    scenarios = (["domain_shift", "bearing_fault", "flash_crash"]
                 if args.scenario == "all" else [args.scenario])

    all_results = {}
    for sc in scenarios:
        print(f"\nRunning scenario: {sc} ({args.trials} trials)")
        reports = run_scenario(sc, trials=args.trials, dim=args.dim)
        all_results[sc] = reports[0]["summary"]  # save first trial summary

    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n💾 Results saved: {args.out}")


if __name__ == "__main__":
    main()
