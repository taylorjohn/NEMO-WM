"""
watchdog.py — NeMo-WM Training Watchdog
========================================
Orchestrates log tailing, status updates, checkpoint guarding,
and contradiction detection in parallel threads.

Usage:
    # Watch all configured tabs overnight:
    python watchdog.py --config watchdog_config.json

    # Quick mode — just watch one log:
    python watchdog.py --log logs/sprint6d.log --tab sprint6d --format dual_head_nce

    # Contradiction check only:
    python watchdog.py --check-only

Author: John Taylor — github.com/taylorjohn
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path

from log_parser     import LogFormat, tail_file
from status_updater import StatusUpdater
from checkpoint_guard import CheckpointGuard, CheckpointMismatch
from contradiction  import run_check


# ── Default config ───────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "tabs": {
        "tab1_ablation": {
            "log":        "logs/tab1_random_encoder.log",
            "checkpoint": "checkpoints/cwm/cwm_random_encoder_best.pt",
            "format":     "world_model",
            "alerts": {
                "da_zero_streak_threshold": 150000,
                "da_spike_threshold":       0.002,
                "loss_spike_pct":           15,
            }
        },
        "sprint6d": {
            "log":        "logs/sprint6d.log",
            "checkpoint": "checkpoints/dinov2_student/student_dualhead_nce_nr_best.pt",
            "format":     "dual_head_nce",
            "alerts": {
                "lclip_stall_epochs":              3,
                "lnull_expected_active_by_epoch":  2,
                "loss_spike_pct":                  25,
            }
        },
    },
    "status_doc":          "CWM_STATUS.md",
    "contradictions_doc":  "CONTRADICTIONS.md",
    "registry_path":       "watchdog_registry.json",
    "check_interval_s":    30,
    "contradiction_check_every_n_epochs": 5,
}


# ── Tab watcher thread ───────────────────────────────────────────────────────

class TabWatcher:
    def __init__(self, tab: str, cfg: dict, updater: StatusUpdater, guard: CheckpointGuard):
        self.tab     = tab
        self.cfg     = cfg
        self.updater = updater
        self.guard   = guard
        self.fmt     = _parse_fmt(cfg.get("format", "world_model"))
        self.log     = Path(cfg["log"])
        self.epoch_count = 0
        self._thread: threading.Thread = None

    def start(self):
        self._thread = threading.Thread(
            target=self._run, name=f"tab-{self.tab}", daemon=True
        )
        self._thread.start()

    def _run(self):
        # Wait for log file to appear
        while not self.log.exists():
            print(f"[watchdog] Waiting for log: {self.log}")
            time.sleep(10)

        print(f"[watchdog] Tailing: {self.log} (tab={self.tab}, fmt={self.fmt.value})")

        def on_step(ev):
            self.updater.on_step(ev)

        def on_epoch(ev):
            self.updater.on_epoch(ev)
            self.epoch_count += 1
            # Auto-register checkpoint if it appears in the epoch event
            if ev.checkpoint_path:
                ckpt = Path(ev.checkpoint_path)
                if ckpt.exists():
                    try:
                        self.guard.register(str(ckpt))
                    except Exception as e:
                        print(f"[watchdog] Could not register ckpt: {e}")

        def on_alert(ev):
            self.updater.on_alert(ev)

        tail_file(
            path           = self.log,
            tab            = self.tab,
            cfg            = self.cfg,
            on_step        = on_step,
            on_epoch       = on_epoch,
            on_alert       = on_alert,
            poll_interval  = 2.0,
            from_start     = False,   # tail from current end
        )


# ── Contradiction check thread ───────────────────────────────────────────────

class ContradictionChecker:
    def __init__(self, cfg: dict):
        self.cfg   = cfg
        self.every = cfg.get("contradiction_check_every_n_epochs", 5)
        self._last = 0
        self._thread: threading.Thread = None

    def start(self, watchers: list[TabWatcher]):
        self._watchers = watchers
        self._thread = threading.Thread(
            target=self._run, name="contradiction-checker", daemon=True
        )
        self._thread.start()

    def _run(self):
        while True:
            total_epochs = sum(w.epoch_count for w in self._watchers)
            if total_epochs > 0 and total_epochs % self.every == 0 and total_epochs != self._last:
                self._last = total_epochs
                print(f"[watchdog] Running contradiction check (total epochs completed: {total_epochs})")
                try:
                    run_check(
                        status_doc  = self.cfg.get("status_doc",         "CWM_STATUS.md"),
                        ckpt_path   = "checkpoints/cwm/cwm_best.pt",
                        log_path    = "logs/training.log",
                        probe_json  = "PROBE_RESULTS_EP12_N1752.json",
                        output      = self.cfg.get("contradictions_doc", "CONTRADICTIONS.md"),
                    )
                except Exception as e:
                    print(f"[watchdog] Contradiction check error: {e}")
            time.sleep(60)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_fmt(s: str) -> LogFormat:
    return {
        "world_model":   LogFormat.WORLD_MODEL,
        "multidomain":   LogFormat.MULTIDOMAIN,
        "dual_head_nce": LogFormat.DUAL_HEAD_NCE,
    }.get(s, LogFormat.WORLD_MODEL)


def _print_banner(cfg: dict):
    print("\n" + "=" * 60)
    print("  NeMo-WM Cortex Watchdog")
    print("=" * 60)
    for tab, tc in cfg["tabs"].items():
        log_exists = "✓" if Path(tc["log"]).exists() else "⏳"
        print(f"  {log_exists} {tab:<25} {tc['format']:<16} {tc['log']}")
    print(f"\n  Status doc:   {cfg['status_doc']}")
    print(f"  Registry:     {cfg['registry_path']}")
    print(f"  Check every:  {cfg['check_interval_s']}s")
    print("=" * 60 + "\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NeMo-WM Training Watchdog")
    parser.add_argument("--config",     default="watchdog_config.json",
                        help="Config file (created if not found)")
    parser.add_argument("--log",        default=None,
                        help="Single log file to watch (quick mode)")
    parser.add_argument("--tab",        default="tab",
                        help="Tab name for quick mode")
    parser.add_argument("--format",     default="world_model",
                        choices=["world_model", "multidomain", "dual_head_nce"])
    parser.add_argument("--check-only", action="store_true",
                        help="Run contradiction check then exit")
    parser.add_argument("--init-guard", action="store_true",
                        help="Register all production checkpoints and exit")
    parser.add_argument("--from-start", action="store_true",
                        help="Tail logs from beginning (for testing)")
    args = parser.parse_args()

    # ── Load or create config ─────────────────────────────────────────────
    cfg_path = Path(args.config)
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        print(f"[watchdog] Config loaded: {cfg_path}")
    else:
        cfg = DEFAULT_CONFIG
        cfg_path.write_text(json.dumps(cfg, indent=2))
        print(f"[watchdog] Config created: {cfg_path}")

    # ── Init registry ─────────────────────────────────────────────────────
    guard = CheckpointGuard(cfg.get("registry_path", "watchdog_registry.json"))

    if args.init_guard:
        from checkpoint_guard import init_registry
        init_registry(cfg.get("registry_path", "watchdog_registry.json"))
        return

    # ── Check-only mode ───────────────────────────────────────────────────
    if args.check_only:
        issues = run_check(
            status_doc  = cfg.get("status_doc",         "CWM_STATUS.md"),
            ckpt_path   = "checkpoints/cwm/cwm_best.pt",
            log_path    = "logs/training.log",
            probe_json  = "PROBE_RESULTS_EP12_N1752.json",
            output      = cfg.get("contradictions_doc", "CONTRADICTIONS.md"),
        )
        sys.exit(0 if not issues else 1)

    # ── Quick single-log mode ─────────────────────────────────────────────
    if args.log:
        cfg["tabs"] = {
            args.tab: {
                "log":     args.log,
                "format":  args.format,
                "alerts":  {},
            }
        }

    # ── Status updater ────────────────────────────────────────────────────
    updater = StatusUpdater(
        status_path          = cfg.get("status_doc", "CWM_STATUS.md"),
        flush_every_n_epochs = 1,
    )

    # ── Start tab watchers ────────────────────────────────────────────────
    _print_banner(cfg)
    watchers = []
    for tab, tab_cfg in cfg["tabs"].items():
        w = TabWatcher(tab, tab_cfg, updater, guard)
        w.start()
        watchers.append(w)

    # ── Start contradiction checker ───────────────────────────────────────
    checker = ContradictionChecker(cfg)
    checker.start(watchers)

    # ── Keep main thread alive ────────────────────────────────────────────
    print("[watchdog] Running. Press Ctrl+C to stop.\n")
    try:
        while True:
            time.sleep(cfg.get("check_interval_s", 30))
            alive = [w.tab for w in watchers if w._thread and w._thread.is_alive()]
            if not alive:
                print("[watchdog] All tab threads stopped.")
                break
    except KeyboardInterrupt:
        print("\n[watchdog] Stopped by user.")


if __name__ == "__main__":
    main()
