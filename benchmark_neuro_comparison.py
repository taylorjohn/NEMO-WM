"""
benchmark_neuro_comparison.py
===============================
Head-to-head benchmark: static planner vs neuromodulated planner.

Runs the same maze environments (Wall / UMaze / Medium / PushT) with
identical seeds, comparing MPC success rate and reporting neuromodulator
signal statistics for the neuro run.

Usage:
    python benchmark_neuro_comparison.py \
        --encoder ./checkpoints/maze_weak_sigreg_straight/cortex_student_phase2_final.pt \
        --envs wall umaze \
        --seeds 1 2 3 \
        --trials 50 \
        --out ./results/neuro_comparison.json

Requires: run_benchmark environment (dmc_recon or equivalent maze env wrappers).
This script patches the planner at import time so it works with whatever
maze environment run_benchmark.py uses.
"""

import argparse
import json
import time
import importlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ── Import neuromodulator ─────────────────────────────────────────────────
try:
    from neuromodulator import NeuromodulatorState, neuro_to_packet, Regime
    print("✅ neuromodulator.py loaded")
except ImportError:
    print("❌ neuromodulator.py not found — copy it to CORTEX root first")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════
# Planner wrappers
# ══════════════════════════════════════════════════════════════════════════

class StaticPlannerWrapper:
    """
    Baseline: identical to current production planner.
    Fixed 64 candidates, fixed eps=0.1, no action gating.
    """
    def __init__(self, planner):
        self.planner = planner
        self.call_count = 0

    def optimize_saccade(self, latent, k_steps=5,
                         num_candidates=64, gamma_horizon=2.0, **kwargs):
        self.call_count += 1
        return self.planner.optimize_saccade(
            latent, k_steps=k_steps,
            num_candidates=num_candidates,
            gamma_horizon=gamma_horizon,
        )

    def reset(self):
        self.call_count = 0


class NeuromodulatedPlannerWrapper:
    """
    Neuromodulated planner. Wraps the same base planner with 7-signal
    neuromodulation: adaptive candidates, E/I action_std, eCB dampening,
    WAIT gating, ACh lr_scale reporting.
    """

    def __init__(self, planner):
        self.planner     = planner
        self.neuro       = NeuromodulatorState(session_start=time.time())
        self.call_count  = 0
        self._last_z:    torch.Tensor = None
        self._last_action_mag: float  = 0.0

        # Per-episode stats
        self.regime_counts: dict = defaultdict(int)
        self.da_history:    list = []
        self.sht_history:   list = []
        self.ach_history:   list = []
        self.ecb_history:   list = []
        self.wait_steps:    int  = 0
        self.total_steps:   int  = 0

    def optimize_saccade(self, latent, k_steps=5,
                         num_candidates=64, gamma_horizon=2.0,
                         z_actual=None, rho=0.0, **kwargs):
        """
        Drop-in replacement for planner.optimize_saccade.
        Pass z_actual each tick to get full neuro benefit; works without it too.
        """
        self.call_count  += 1
        self.total_steps += 1

        # Build z tensors for neuro update
        if isinstance(latent, np.ndarray):
            z_t = torch.from_numpy(latent.flatten()).float()
        else:
            z_t = latent.detach().flatten().float()

        if z_actual is not None:
            if isinstance(z_actual, np.ndarray):
                z_next = torch.from_numpy(z_actual.flatten()).float()
            else:
                z_next = z_actual.detach().flatten().float()
        elif self._last_z is not None:
            # Approximate: use current z as z_actual for previous prediction
            z_next = z_t
        else:
            z_next = z_t

        # Update neuromodulator
        sig = self.neuro.update(z_t, z_next, rho=rho,
                                action_magnitude=self._last_action_mag)
        self._last_z = z_t.clone()

        # Track stats
        self.regime_counts[sig["regime"]] += 1
        self.da_history.append(sig["da"])
        self.sht_history.append(sig["sht"])
        self.ach_history.append(sig["ach"])
        self.ecb_history.append(sig["ecb"])

        # WAIT regime: return zero action
        if not self.neuro.should_act:
            self.wait_steps += 1
            return np.zeros(2, dtype=np.float32)

        # Adaptive candidates and action_std from neuro
        n_cands    = sig["n_candidates"]
        action_std = sig["action_std"]

        # Run planner with adaptive parameters
        device  = getattr(self.planner, "device", "cpu")
        il      = torch.tensor(latent, device=device, dtype=torch.float32)
        if il.dim() == 1:
            il = il.unsqueeze(0)

        cands   = torch.randn(n_cands, k_steps, 2, device=device) * action_std
        best_cost, best_seq = float("inf"), None

        with torch.no_grad():
            for i in range(n_cands):
                cost = self.planner.predict_trajectory_cost(il, cands[i], gamma_horizon)
                if float(cost) < best_cost:
                    best_cost = float(cost)
                    best_seq  = cands[i]

        if best_seq is None:
            return np.zeros(2, dtype=np.float32)

        action = best_seq[0].cpu().numpy() * sig["action_scale"]
        self._last_action_mag = float(np.linalg.norm(action))
        return action

    def episode_stats(self) -> dict:
        """Return per-episode neuromodulator statistics."""
        total = max(1, self.total_steps)
        return {
            "regime_pct":    {k: round(100*v/total,1)
                              for k, v in self.regime_counts.items()},
            "wait_pct":      round(100 * self.wait_steps / total, 1),
            "da_mean":       round(float(np.mean(self.da_history)),  3) if self.da_history  else 0,
            "sht_mean":      round(float(np.mean(self.sht_history)), 3) if self.sht_history else 0,
            "ach_mean":      round(float(np.mean(self.ach_history)), 3) if self.ach_history else 0,
            "ecb_mean":      round(float(np.mean(self.ecb_history)), 3) if self.ecb_history else 0,
            "total_steps":   self.total_steps,
        }

    def reset_episode(self):
        """Call between episodes to clear per-episode counters."""
        self.regime_counts.clear()
        self.da_history.clear()
        self.sht_history.clear()
        self.ach_history.clear()
        self.ecb_history.clear()
        self.wait_steps  = 0
        self.total_steps = 0
        self._last_z     = None
        self._last_action_mag = 0.0
        self.neuro.reset(full=False)   # preserve Ado across episodes

    def reset(self):
        self.reset_episode()
        self.call_count = 0


# ══════════════════════════════════════════════════════════════════════════
# Benchmark runner
# ══════════════════════════════════════════════════════════════════════════

def run_single_trial(env, planner_wrapper, horizon=25) -> tuple[bool, int]:
    """
    Run one trial to max horizon. Returns (success, steps_taken).
    Assumes env follows run_benchmark.py interface:
        obs = env.reset()
        obs, reward, done, info = env.step(action)
        info["mpc_success"] or info.get("success", False)
    """
    obs = env.reset()
    latent = _encode(obs, planner_wrapper)

    for step in range(horizon):
        action  = planner_wrapper.optimize_saccade(latent, k_steps=5,
                                                    num_candidates=64,
                                                    gamma_horizon=2.0)
        obs, reward, done, info = env.step(action)
        prev_latent = latent
        latent      = _encode(obs, planner_wrapper)

        # Provide z_actual for neuromodulated wrapper
        if hasattr(planner_wrapper, "neuro") and hasattr(planner_wrapper, "_last_z"):
            # retroactively pass actual z for last step's update
            # (neuro was updated with predicted z, now correct with actual)
            pass  # approximation already handled in optimize_saccade

        success = info.get("mpc_success", False) or info.get("success", False)
        if done or success:
            return success, step + 1

    return False, horizon


def _encode(obs, planner_wrapper) -> np.ndarray:
    """Extract latent from observation. Expects encoder on planner or in scope."""
    if hasattr(planner_wrapper, "_encoder"):
        with torch.no_grad():
            t = torch.from_numpy(obs).float().unsqueeze(0)
            return planner_wrapper._encoder(t).squeeze(0).numpy()
    # Fallback: obs is already a latent (some envs return latent directly)
    return np.array(obs, dtype=np.float32).flatten()


# ══════════════════════════════════════════════════════════════════════════
# Main comparison
# ══════════════════════════════════════════════════════════════════════════

def compare(args):
    """
    Run head-to-head comparison using run_benchmark.py's environment factory.
    Falls back to the subprocess approach if environment cannot be imported.
    """

    # ── Try importing benchmark environment ───────────────────────────────
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from run_benchmark import (
            make_env,
            load_encoder,
            make_predictor,
            make_planner,
        )
        _use_subprocess = False
    except ImportError:
        _use_subprocess = True

    if _use_subprocess:
        _compare_via_subprocess(args)
        return

    # ── Direct comparison ─────────────────────────────────────────────────
    device  = "cpu"
    encoder = load_encoder(args.encoder, device)
    encoder.eval()

    results = {"args": vars(args), "envs": {}}

    for env_name in args.envs:
        print(f"\n{'─'*56}")
        print(f"  ENV: {env_name.upper()}")
        print(f"{'─'*56}")

        baseline_mpcs = []
        neuro_mpcs    = []
        neuro_stats   = []

        for seed in args.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Build base planner — shared predictor, independent wrappers
            predictor   = make_predictor(device)
            base_plan   = make_planner(predictor, device)
            static_wrap = StaticPlannerWrapper(base_plan)
            static_wrap._encoder = encoder

            predictor2  = make_predictor(device)
            base_plan2  = make_planner(predictor2, device)
            neuro_wrap  = NeuromodulatedPlannerWrapper(base_plan2)
            neuro_wrap._encoder = encoder

            # ── Baseline seed ─────────────────────────────────────────────
            env_b = make_env(env_name, seed=seed)
            b_succ = 0
            for _ in range(args.trials):
                ok, _ = run_single_trial(env_b, static_wrap, args.horizon)
                if ok: b_succ += 1
            env_b.close()
            b_mpc = round(100 * b_succ / args.trials, 1)
            baseline_mpcs.append(b_mpc)
            print(f"  Seed {seed} | Baseline:     {b_mpc:5.1f}% MPC")

            # ── Neuro seed ────────────────────────────────────────────────
            env_n = make_env(env_name, seed=seed)
            n_succ = 0
            ep_stats_list = []
            for _ in range(args.trials):
                neuro_wrap.reset_episode()
                ok, _ = run_single_trial(env_n, neuro_wrap, args.horizon)
                if ok: n_succ += 1
                ep_stats_list.append(neuro_wrap.episode_stats())
            env_n.close()
            n_mpc = round(100 * n_succ / args.trials, 1)
            neuro_mpcs.append(n_mpc)

            # Aggregate episode stats
            agg = {
                "regime_pct": defaultdict(list),
                "wait_pct":   [],
                "da_mean":    [],
                "sht_mean":   [],
            }
            for ep in ep_stats_list:
                agg["wait_pct"].append(ep["wait_pct"])
                agg["da_mean"].append(ep["da_mean"])
                agg["sht_mean"].append(ep["sht_mean"])
                for reg, pct in ep["regime_pct"].items():
                    agg["regime_pct"][reg].append(pct)
            agg_out = {
                "wait_pct_mean": round(float(np.mean(agg["wait_pct"])), 1),
                "da_mean":       round(float(np.mean(agg["da_mean"])),  3),
                "sht_mean":      round(float(np.mean(agg["sht_mean"])), 3),
                "regime_pct":    {k: round(float(np.mean(v)), 1)
                                  for k, v in agg["regime_pct"].items()},
            }
            neuro_stats.append(agg_out)
            print(f"  Seed {seed} | Neuromodulated:{n_mpc:5.1f}% MPC  "
                  f"(WAIT={agg_out['wait_pct_mean']}% "
                  f"DA={agg_out['da_mean']:.2f} 5HT={agg_out['sht_mean']:.2f})")

        # ── Summary ───────────────────────────────────────────────────────
        b_mean = float(np.mean(baseline_mpcs))
        n_mean = float(np.mean(neuro_mpcs))
        delta  = n_mean - b_mean

        print(f"\n  {'─'*40}")
        print(f"  {env_name.upper():12s}  Baseline: {b_mean:.1f}%   "
              f"Neuro: {n_mean:.1f}%   Δ: {delta:+.1f}pp")
        print(f"  {'─'*40}")

        results["envs"][env_name] = {
            "baseline_mpc_per_seed": baseline_mpcs,
            "neuro_mpc_per_seed":    neuro_mpcs,
            "baseline_mean":         round(b_mean, 1),
            "neuro_mean":            round(n_mean, 1),
            "delta_pp":              round(delta, 1),
            "neuro_signal_stats":    neuro_stats,
        }

    # ── Final table ───────────────────────────────────────────────────────
    print(f"\n{'═'*56}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'═'*56}")
    print(f"  {'Env':<12} {'Baseline':>10} {'Neuro':>10} {'Δ':>8}")
    print(f"  {'─'*44}")
    for env_name, r in results["envs"].items():
        delta_str = f"{r['delta_pp']:+.1f}pp"
        marker = "✅" if r["delta_pp"] >= 0 else "⚠️ "
        print(f"  {marker} {env_name:<10} {r['baseline_mean']:>9.1f}%"
              f" {r['neuro_mean']:>9.1f}% {delta_str:>8}")
    print(f"{'═'*56}")

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved: {out_path}")


def _compare_via_subprocess(args):
    """
    Fallback: run_benchmark.py can't be imported directly.
    Instructs user to run both variants via CLI and prints comparison template.
    """
    print("\n⚠️  Could not import run_benchmark environment directly.")
    print("Run these two commands then compare the JSON outputs:\n")

    enc = args.encoder
    envs = " ".join(args.envs)
    seeds = " ".join(str(s) for s in args.seeds)

    print("# 1. Baseline (current production)")
    print(f"python run_benchmark.py \\")
    print(f"  --encoder {enc} \\")
    print(f"  --option 1 --env {envs} --seeds {seeds} \\")
    print(f"  --out ./results/baseline_mpc.json\n")

    print("# 2. Neuromodulated (add --neuro flag after patch)")
    print(f"python run_benchmark.py \\")
    print(f"  --encoder {enc} \\")
    print(f"  --option 1 --env {envs} --seeds {seeds} \\")
    print(f"  --neuro \\")
    print(f"  --out ./results/neuro_mpc.json\n")

    print("# 3. Compare")
    print("python benchmark_neuro_comparison.py --compare \\")
    print("  --baseline ./results/baseline_mpc.json \\")
    print("  --neuro    ./results/neuro_mpc.json")


def compare_json(args):
    """Compare two existing JSON result files from run_benchmark.py."""
    b = json.load(open(args.baseline))
    n = json.load(open(args.neuro_file))

    print(f"\n{'═'*56}")
    print(f"  JSON COMPARISON: Baseline vs Neuromodulated")
    print(f"{'═'*56}")

    # Extract MPC values — handle run_benchmark.py JSON format
    def extract_mpc(data: dict) -> dict:
        """Extract env→mpc_mean from run_benchmark.py output format."""
        out = {}
        for key, val in data.items():
            if isinstance(val, dict) and "mpc" in val:
                out[key] = val["mpc"]
            elif isinstance(val, (int, float)):
                out[key] = float(val)
        return out

    b_mpc = extract_mpc(b)
    n_mpc = extract_mpc(n)

    all_envs = sorted(set(list(b_mpc.keys()) + list(n_mpc.keys())))
    for env in all_envs:
        b_val = b_mpc.get(env, float("nan"))
        n_val = n_mpc.get(env, float("nan"))
        delta = n_val - b_val
        marker = "✅" if delta >= 0 else "⚠️ "
        print(f"  {marker} {env:<12} Baseline: {b_val:.1f}%   "
              f"Neuro: {n_val:.1f}%   Δ: {delta:+.1f}pp")

    print(f"{'═'*56}")


# ── patch_benchmark helper ─────────────────────────────────────────────────

PATCH_SNIPPET = '''
# ── Neuromodulator patch (added by benchmark_neuro_comparison.py) ─────────
import argparse as _ap
_neuro_flag = "--neuro" in __import__("sys").argv
if _neuro_flag:
    from neuromodulator import NeuromodulatorState as _NS
    _neuro_state = _NS(session_start=__import__("time").time())
    print("🧠 Neuromodulator ENABLED")
else:
    _neuro_state = None
    print("📊 Neuromodulator DISABLED (baseline)")

def _wrap_optimize_saccade(planner_instance):
    """Monkey-patch planner.optimize_saccade with neuro version."""
    if not _neuro_flag or _neuro_state is None:
        return
    import numpy as np
    _orig = planner_instance.optimize_saccade
    _last_mag = [0.0]
    def _neuro_saccade(initial_latent, k_steps=5, num_candidates=64,
                       gamma_horizon=2.0, **kw):
        import torch
        z = torch.from_numpy(
            initial_latent.flatten() if hasattr(initial_latent,"flatten")
            else initial_latent).float()
        sig = _neuro_state.update(z, z, action_magnitude=_last_mag[0])
        if sig["regime"] == "WAIT":
            return np.zeros(2, dtype=np.float32)
        import torch as _torch
        dev = getattr(planner_instance, "device", "cpu")
        il  = _torch.tensor(initial_latent, device=dev, dtype=_torch.float32)
        if il.dim()==1: il=il.unsqueeze(0)
        cands = _torch.randn(sig["n_candidates"], k_steps, 2, device=dev) * sig["action_std"]
        best_cost, best_seq = float("inf"), None
        with _torch.no_grad():
            for i in range(len(cands)):
                cost = planner_instance.predict_trajectory_cost(il, cands[i], gamma_horizon)
                if float(cost) < best_cost:
                    best_cost, best_seq = float(cost), cands[i]
        if best_seq is None: return np.zeros(2, dtype=np.float32)
        action = best_seq[0].cpu().numpy() * sig["action_scale"]
        _last_mag[0] = float(np.linalg.norm(action))
        return action
    planner_instance.optimize_saccade = _neuro_saccade
# ─────────────────────────────────────────────────────────────────────────
'''


def patch_run_benchmark(args):
    """Inject --neuro flag support into run_benchmark.py."""
    target = Path("run_benchmark.py")
    if not target.exists():
        print("❌ run_benchmark.py not found — run from CORTEX root")
        return

    src = target.read_text(encoding="utf-8")
    if "Neuromodulator patch" in src:
        print("✅ run_benchmark.py already patched")
        return

    import shutil
    shutil.copy(target, target.with_suffix(".py.bak_neuro"))
    print(f"📦 Backup: {target.with_suffix('.py.bak_neuro')}")

    # Inject after imports, before main logic
    inject_after = "import argparse"
    if inject_after in src:
        src = src.replace(inject_after, inject_after + "\n" + PATCH_SNIPPET, 1)
    else:
        src = PATCH_SNIPPET + "\n" + src

    # Inject _wrap call after planner is created
    planner_create = "planner = "
    if planner_create in src:
        idx = src.index(planner_create)
        end = src.index("\n", idx)
        src = src[:end+1] + "_wrap_optimize_saccade(planner)\n" + src[end+1:]

    target.write_text(src, encoding="utf-8")
    print("✅ run_benchmark.py patched with --neuro flag support")
    print("\nNow run:")
    print("  python run_benchmark.py --encoder ... --option 1 --env wall --seeds 1 2 3 --out baseline.json")
    print("  python run_benchmark.py --encoder ... --option 1 --env wall --seeds 1 2 3 --neuro --out neuro.json")


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Neuromodulator vs baseline benchmark comparison"
    )
    sub = parser.add_subparsers(dest="cmd")

    # run subcommand
    run_p = sub.add_parser("run", help="Direct head-to-head comparison")
    run_p.add_argument("--encoder", required=True)
    run_p.add_argument("--envs",    nargs="+", default=["wall", "umaze"])
    run_p.add_argument("--seeds",   nargs="+", type=int, default=[1, 2, 3])
    run_p.add_argument("--trials",  type=int,  default=50)
    run_p.add_argument("--horizon", type=int,  default=25)
    run_p.add_argument("--out",     default="./results/neuro_comparison.json")

    # compare subcommand
    cmp_p = sub.add_parser("compare", help="Compare two existing JSON result files")
    cmp_p.add_argument("--baseline",   required=True)
    cmp_p.add_argument("--neuro-file", required=True, dest="neuro_file")

    # patch subcommand
    ptc_p = sub.add_parser("patch", help="Patch run_benchmark.py with --neuro flag")

    args = parser.parse_args()

    if args.cmd == "run":
        compare(args)
    elif args.cmd == "compare":
        compare_json(args)
    elif args.cmd == "patch":
        patch_run_benchmark(args)
    else:
        # Default: if called without subcommand but with --encoder, run comparison
        parser2 = argparse.ArgumentParser()
        parser2.add_argument("--encoder",  required=True)
        parser2.add_argument("--envs",     nargs="+", default=["wall", "umaze"])
        parser2.add_argument("--seeds",    nargs="+", type=int, default=[1, 2, 3])
        parser2.add_argument("--trials",   type=int,  default=50)
        parser2.add_argument("--horizon",  type=int,  default=25)
        parser2.add_argument("--out",      default="./results/neuro_comparison.json")
        parser2.add_argument("--compare",  action="store_true")
        parser2.add_argument("--baseline", default=None)
        parser2.add_argument("--neuro-file", default=None, dest="neuro_file")
        parser2.add_argument("--patch",    action="store_true")
        args2 = parser2.parse_args()

        if args2.patch:
            patch_run_benchmark(args2)
        elif args2.compare:
            compare_json(args2)
        else:
            compare(args2)


if __name__ == "__main__":
    main()
