"""lineage.py — CORTEX-PE v16.17
═══════════════════════════════════════════════════════════════════════════════
Lineage logging utility. Records every training run as a (solution, score) pair
in a domain-specific JSONL file, building the 𝒫_t lineage structure from AVO
(arXiv:2603.24517).

Each line in the JSONL is one committed run:
  {
    "ts":        "2026-03-28T23:14:00Z",      # UTC timestamp
    "domain":    "recon",                       # domain key
    "run_id":    "recon_student_v1",            # human-readable ID
    "script":    "train_student_temporal.py",   # script that produced it
    "checkpoint": "checkpoints/recon_student/student_best.pt",
    "metrics":   {"auroc": 0.9499, "triplet": 0.9420},
    "config":    {"epochs": 30, "n_pairs": 8000, ...},
    "notes":     "Phase 2 RoPE head, frozen encoder",
    "parent":    "recon_student_v1"             # which run this improved on
  }

Usage — from a training script:
  from lineage import Lineage
  lin = Lineage(domain="recon")
  lin.commit(
      run_id="recon_student_v2",
      script=__file__,
      checkpoint="checkpoints/recon_student/student_best.pt",
      metrics={"auroc": 0.9499, "triplet": 0.9420},
      config=vars(args),
      notes="Phase 2 RoPE head on frozen encoder",
      parent=lin.best_run_id(),
  )

Usage — query:
  lin = Lineage(domain="recon")
  print(lin.best())           # best run by primary metric
  print(lin.history())        # all runs, chronological
  lin.summary()               # print scorecard to stdout
  lin.plateau(metric="auroc", window=3, threshold=0.005)  # stagnation check

Lineage files:
  lineage/<domain>.jsonl       — append-only, one JSON object per line
  lineage/all_domains.jsonl    — cross-domain summary (written on each commit)
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LINEAGE_DIR = Path("lineage")
ALL_DOMAINS_FILE = LINEAGE_DIR / "all_domains.jsonl"

# Primary metric per domain — used for best() and plateau() comparisons
DOMAIN_PRIMARY_METRIC: dict[str, str] = {
    "recon":   "auroc",
    "bearing": "auroc",
    "mimii":   "auroc",
    "cardiac": "auroc",
    "smap":    "auroc",
    "trading": "win_rate",
    "npu":     "cos_sim",
    "perception": "test_pass_rate",
}


class Lineage:
    """Per-domain lineage logger and query interface.

    Parameters
    ----------
    domain : str
        Domain key, e.g. "recon", "bearing", "cardiac".
    lineage_dir : str or Path, optional
        Directory for JSONL files. Defaults to ./lineage/.
    primary_metric : str, optional
        Override the default primary metric for this domain.
    """

    def __init__(
        self,
        domain: str,
        lineage_dir: str | Path = LINEAGE_DIR,
        primary_metric: str | None = None,
    ) -> None:
        self.domain = domain
        self.lineage_dir = Path(lineage_dir)
        self.lineage_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.lineage_dir / f"{domain}.jsonl"
        self._primary = primary_metric or DOMAIN_PRIMARY_METRIC.get(domain, "auroc")
        self._runs: list[dict] = self._load()

    # ── I/O ──────────────────────────────────────────────────────────────────

    def _load(self) -> list[dict]:
        if not self._path.exists():
            return []
        runs = []
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        runs.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return runs

    def _append(self, record: dict) -> None:
        with open(self._path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
        # Also append to cross-domain file
        ALL_DOMAINS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(ALL_DOMAINS_FILE, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    # ── Commit ────────────────────────────────────────────────────────────────

    def commit(
        self,
        run_id: str,
        metrics: dict[str, float],
        script: str = "",
        checkpoint: str = "",
        config: dict | None = None,
        notes: str = "",
        parent: str | None = None,
        tags: list[str] | None = None,
    ) -> dict:
        """Record a completed training run.

        Parameters
        ----------
        run_id : str
            Unique human-readable identifier for this run.
        metrics : dict
            Evaluation metrics, e.g. {"auroc": 0.9499, "triplet": 0.9420}.
        script : str
            Script that produced this run (use __file__).
        checkpoint : str
            Path to the best checkpoint saved by this run.
        config : dict
            Training hyperparameters (e.g. vars(args)).
        notes : str
            Free-text description of what was tried and why.
        parent : str
            run_id of the run this improved upon (use self.best_run_id()).
        tags : list[str]
            Optional tags for filtering, e.g. ["phase1", "triplet_infonce"].

        Returns
        -------
        dict — the committed record.
        """
        record = {
            "ts":         datetime.now(timezone.utc).isoformat(),
            "domain":     self.domain,
            "run_id":     run_id,
            "script":     os.path.basename(str(script)),
            "checkpoint": str(checkpoint),
            "metrics":    metrics,
            "config":     config or {},
            "notes":      notes,
            "parent":     parent,
            "tags":       tags or [],
        }
        self._runs.append(record)
        self._append(record)
        primary = metrics.get(self._primary, 0.)
        prev_best = self._primary_score(self.best()) if self._runs[:-1] else 0.
        delta = primary - prev_best
        sign = "★ NEW BEST" if delta > 0 else ("=" if delta == 0 else f"Δ{delta:+.4f}")
        print(f"[lineage] {self.domain}/{run_id}  "
              f"{self._primary}={primary:.4f}  {sign}")
        return record

    # ── Query ─────────────────────────────────────────────────────────────────

    def history(self) -> list[dict]:
        """All runs in chronological order."""
        return list(self._runs)

    def best(self) -> dict | None:
        """Run with the highest primary metric score."""
        if not self._runs:
            return None
        return max(self._runs, key=lambda r: self._primary_score(r))

    def best_run_id(self) -> str | None:
        """run_id of the best run, or None if no runs yet."""
        b = self.best()
        return b["run_id"] if b else None

    def best_score(self) -> float:
        """Primary metric score of the best run, or 0.0 if no runs."""
        b = self.best()
        return self._primary_score(b) if b else 0.0

    def latest(self) -> dict | None:
        """Most recently committed run."""
        return self._runs[-1] if self._runs else None

    def since_best(self) -> int:
        """Number of runs committed since the last improvement."""
        if not self._runs:
            return 0
        best_score = self.best_score()
        for i in range(len(self._runs) - 1, -1, -1):
            if self._primary_score(self._runs[i]) >= best_score:
                return len(self._runs) - 1 - i
        return len(self._runs)

    def _primary_score(self, run: dict | None) -> float:
        if run is None:
            return 0.
        return float(run.get("metrics", {}).get(self._primary, 0.))

    # ── Stagnation detection ──────────────────────────────────────────────────

    def plateau(self, metric: str | None = None, window: int = 3,
                threshold: float = 0.005) -> bool:
        """Return True if the last `window` runs improved by less than `threshold`.

        Use this to trigger architecture changes or new interventions.

        Parameters
        ----------
        metric : str, optional
            Metric to check. Defaults to the domain's primary metric.
        window : int
            Number of recent runs to consider.
        threshold : float
            Minimum improvement across the window to NOT be considered stalled.

        Example
        -------
        if lin.plateau(window=3, threshold=0.005):
            print("Stagnated — try next intervention from the table")
        """
        m = metric or self._primary
        if len(self._runs) < window:
            return False
        recent = [float(r.get("metrics", {}).get(m, 0.)) for r in self._runs[-window:]]
        return (max(recent) - min(recent)) < threshold

    def next_intervention(self, domain_interventions: dict[str, list[str]] | None = None) -> str | None:
        """Suggest the next intervention when plateau is detected.

        Tries each intervention in order. Returns the first one not yet tagged
        in the lineage. Returns None if all interventions have been tried.

        Parameters
        ----------
        domain_interventions : dict, optional
            Override the default intervention table for this domain.
        """
        table = domain_interventions or INTERVENTION_TABLE
        interventions = table.get(self.domain, [])
        tried_tags = set()
        for run in self._runs:
            tried_tags.update(run.get("tags", []))
        for intervention in interventions:
            if intervention not in tried_tags:
                return intervention
        return None

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self, n: int = 10) -> None:
        """Print a scorecard of recent runs."""
        runs = self._runs[-n:]
        best = self.best()
        print(f"\n{'─'*60}")
        print(f"Lineage: {self.domain}  ({len(self._runs)} total runs)")
        print(f"{'─'*60}")
        print(f"  {'run_id':<35}  {self._primary:>8}  notes")
        print(f"  {'─'*35}  {'─'*8}  {'─'*15}")
        for r in runs:
            score = self._primary_score(r)
            marker = " ★" if r is best else ""
            notes  = r.get("notes", "")[:40]
            print(f"  {r['run_id']:<35}  {score:>8.4f}  {notes}{marker}")
        if best:
            print(f"\n  Best: {best['run_id']}  {self._primary}={self._primary_score(best):.4f}")
        plateau_flag = "⚠️  PLATEAU" if self.plateau() else "✅ progressing"
        print(f"  Status: {plateau_flag}  ({self.since_best()} runs since best)")
        nxt = self.next_intervention()
        if nxt and self.plateau():
            print(f"  Suggested next: {nxt}")
        print(f"{'─'*60}\n")


# ── Intervention table ────────────────────────────────────────────────────────
# Ordered list of interventions per domain.
# plateau() + next_intervention() use this to suggest the next step.
# Tag each training run with the intervention it implements so the system
# knows what has been tried.

INTERVENTION_TABLE: dict[str, list[str]] = {
    "recon": [
        "triplet_infonce",           # same-trajectory hard negatives (done ✅)
        "rope_head",                 # RoPETemporalHead on frozen encoder (done ✅)
        "increase_k_far",            # push k_far_max from 40 → 70
        "colour_jitter_strong",      # stronger augmentation (brightness 0.4, contrast 0.4)
        "dinov2_distillation",       # distil DINOv2-small features into StudentEncoder
        "coordinate_encoder",        # feed (x,y) directly into encoder as extra channels
        "larger_capacity",           # increase encoder to 128→256→128 (double params)
    ],
    "bearing": [
        "subspace_ad_k16",           # SubspaceAD k=16 (done ✅)
        "subspace_ad_k32",           # try k=32 for richer subspace
        "patchcore",                 # PatchCore baseline comparison
        "sigreg_distillation",       # SIGReg distillation from DINOv2
    ],
    "cardiac": [
        "beats_distillation",        # BEATs teacher distillation (done ✅)
        "wav2vec2_distillation",     # try wav2vec2-base as teacher
        "m2d_distillation",          # M2D teacher (audio masked autoencoder)
        "longer_training",           # extend from 6 → 20 epochs
        "stronger_augmentation",     # pitch shift + time warp
    ],
    "smap": [
        "subspace_ad_k16_w128",
        "hybrid_pca_drift_k16",
        "robust_pca_cleaning",
        "fft_features",
        "subspace_ad_k32",
        "per_channel_normalisation",
        "longer_windows",
    ],
    "npu": [
        "xint8_export",              # XINT8 via AMD Quark (done ✅, cos_sim 0.9997)
        "tile_size_search",          # agentic search over ONNX tile sizes
        "operator_fusion",           # fuse conv+bn+relu in Vitis AI EP
        "bf16_fallback",             # test BF16 on non-LayerNorm ops
    ],
    "trading": [
        "scenario_classification",   # FLASH_CRASH fix (done ✅)
        "vault_rlock",               # RLock deadlock fix (done ✅)
        "neuromodulator_last_z",     # DA=0.500 fix (Monday pre-flight)
        "sniperwindow_tuning",       # tune resonance threshold
        "live_session_baseline",     # first live session post-fixes
    ],
    "mvtec": [
        "global_subspace_ad",        # ✅ 0.685 k=16 global random
        "patch_subspace_ad",         # ✅ 0.745 k=16 patch
        "subspace_k32",              # ✅ 0.790 ensemble k=32 (best random)
        "pretrained_encoder",        # ✅ 0.752 RECON k=32
        "subspace_k64",              # ✅ 0.776 degraded
        "dinov2_distillation",       # ✅ 0.815 DINOv2 ep30 (BEST)
        "murf_distillation",         # ✅ 0.796 MuRF [0.5,1.0,1.5]
        "murf_recon_init",           # ✅ 0.781 MuRF+RECON init
        "murf_scales_075",           # 🔄 running [0.75,1.0,1.5]
        "larger_capacity",           # NEXT — double CNN channels if needed
        "lpwm_distillation",         # future — particle-level teacher
    ],
    "perception": [
        "mockbackend_tests",         # 47/47 passing (done ✅)
        "ollama_install",            # install ollama + qwen3:0.6b (done ✅)
        "real_llm_responses",        # switch from MockBackend (done ✅)
        "recon_domain_test",         # NEXT — test domain=recon with real latents
    ],
}


# ── Cross-domain summary ──────────────────────────────────────────────────────

def all_domains_summary(lineage_dir: str | Path = LINEAGE_DIR) -> None:
    """Print best result per domain across all lineage files."""
    lineage_dir = Path(lineage_dir)
    if not lineage_dir.exists():
        print("No lineage directory found.")
        return

    print(f"\n{'═'*65}")
    print(f"CORTEX-PE Lineage Summary  —  all domains")
    print(f"{'═'*65}")
    print(f"  {'Domain':<12}  {'Best run':<32}  {'Metric':>10}  {'Runs':>5}")
    print(f"  {'─'*12}  {'─'*32}  {'─'*10}  {'─'*5}")

    for jl in sorted(lineage_dir.glob("*.jsonl")):
        if jl.name == "all_domains.jsonl":
            continue
        domain = jl.stem
        lin = Lineage(domain, lineage_dir=lineage_dir)
        best = lin.best()
        if best:
            score  = lin._primary_score(best)
            metric = lin._primary
            n      = len(lin._runs)
            plateau_flag = " ⚠️" if lin.plateau() else ""
            print(f"  {domain:<12}  {best['run_id']:<32}  "
                  f"{metric}={score:.4f}  {n:>5}{plateau_flag}")
        else:
            print(f"  {domain:<12}  (no runs yet)")

    print(f"{'═'*65}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="CORTEX-PE lineage query CLI")
    ap.add_argument("command", choices=["summary", "best", "history", "plateau"],
                    help="Command to run")
    ap.add_argument("--domain", default=None,
                    help="Domain to query (omit for all-domain summary)")
    ap.add_argument("--lineage-dir", default=str(LINEAGE_DIR))
    ap.add_argument("--window", type=int, default=3,
                    help="Window size for plateau detection")
    ap.add_argument("--threshold", type=float, default=0.005,
                    help="Plateau threshold")
    args = ap.parse_args()

    if args.command == "summary" and args.domain is None:
        all_domains_summary(args.lineage_dir)
    elif args.domain:
        lin = Lineage(args.domain, lineage_dir=args.lineage_dir)
        if args.command == "summary":
            lin.summary()
        elif args.command == "best":
            b = lin.best()
            print(json.dumps(b, indent=2, default=str) if b else "No runs yet.")
        elif args.command == "history":
            for r in lin.history():
                print(json.dumps(r, default=str))
        elif args.command == "plateau":
            p = lin.plateau(window=args.window, threshold=args.threshold)
            print(f"Plateau: {p}")
            if p:
                nxt = lin.next_intervention()
                print(f"Next intervention: {nxt}")
    else:
        ap.error("Specify --domain or use 'summary' without --domain")
