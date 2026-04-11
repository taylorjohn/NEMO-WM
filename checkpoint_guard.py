"""
checkpoint_guard.py — NeMo-WM checkpoint registry and pre-eval guard
=====================================================================
Verifies checkpoints match expected metadata before evaluation scripts run.
Prevents the "ran alignment test on wrong encoder" failure mode.

Usage:
    # Register a checkpoint after training:
    guard = CheckpointGuard("watchdog_registry.json")
    guard.register("checkpoints/cwm/cwm_best.pt", epoch=29, loss=0.5673)

    # Verify before running eval:
    guard.verify("checkpoints/cwm/cwm_best.pt")  # raises if mismatch

    # CLI — wrap any eval command:
    python checkpoint_guard.py verify checkpoints/cwm/cwm_best.pt
    python checkpoint_guard.py register checkpoints/cwm/cwm_best.pt --epoch 29 --loss 0.5673

Author: John Taylor — github.com/taylorjohn
"""

import hashlib
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


class CheckpointMismatch(Exception):
    pass


@dataclass
class CheckpointRecord:
    path:           str
    sha256:         str              # first 16 hex chars — unique enough, fast
    size_bytes:     int
    epoch:          Optional[int]    = None
    loss:           Optional[float]  = None
    registered_at:  Optional[str]    = None
    note:           Optional[str]    = None
    protected:      bool             = False  # if True, write-protect warning


def _sha256_prefix(path: Path, prefix_bytes: int = 65536) -> str:
    """Hash first 64KB — fast, unique enough for checkpoint verification."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read(prefix_bytes))
    return h.hexdigest()[:16]


def _load_checkpoint_meta(path: Path) -> tuple[Optional[int], Optional[float]]:
    """Try to load epoch/loss from checkpoint without importing torch."""
    try:
        import torch
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict):
            epoch = ckpt.get('epoch')
            loss  = ckpt.get('loss')
            return epoch, loss
    except Exception:
        pass
    return None, None


class CheckpointGuard:
    """
    Registry-based checkpoint verifier.
    Reads/writes a JSON registry file.
    """

    def __init__(self, registry_path: str = "watchdog_registry.json"):
        self.registry_path = Path(registry_path)
        self._records: dict[str, CheckpointRecord] = {}
        self._load()

    def _load(self):
        if self.registry_path.exists():
            data = json.loads(self.registry_path.read_text())
            for k, v in data.items():
                self._records[k] = CheckpointRecord(**v)

    def _save(self):
        data = {k: asdict(v) for k, v in self._records.items()}
        self.registry_path.write_text(json.dumps(data, indent=2))

    def register(
        self,
        path: str,
        epoch: Optional[int]   = None,
        loss:  Optional[float] = None,
        note:  Optional[str]   = None,
        protected: bool        = False,
    ) -> CheckpointRecord:
        """Register a checkpoint. Call after training saves a new checkpoint."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        # Try to load metadata from checkpoint itself
        ckpt_epoch, ckpt_loss = _load_checkpoint_meta(p)
        epoch = epoch if epoch is not None else ckpt_epoch
        loss  = loss  if loss  is not None else ckpt_loss

        rec = CheckpointRecord(
            path          = str(p),
            sha256        = _sha256_prefix(p),
            size_bytes    = p.stat().st_size,
            epoch         = epoch,
            loss          = loss,
            registered_at = time.strftime("%Y-%m-%d %H:%M:%S"),
            note          = note,
            protected     = protected,
        )
        self._records[str(p)] = rec
        self._save()
        print(f"[guard] Registered: {p.name} | epoch={epoch} | loss={loss} | sha256={rec.sha256}")
        return rec

    def verify(self, path: str, strict: bool = True) -> CheckpointRecord:
        """
        Verify a checkpoint matches its registry entry.
        Raises CheckpointMismatch if:
          - Not in registry (if strict=True)
          - SHA256 prefix doesn't match
          - Size changed
          - Epoch/loss mismatch (if both recorded)
        Returns the registry record on success.
        """
        p = Path(path)
        key = str(p)

        if key not in self._records:
            if strict:
                raise CheckpointMismatch(
                    f"[guard] UNREGISTERED checkpoint: {path}\n"
                    f"  Run: python checkpoint_guard.py register {path}\n"
                    f"  Or add --no-guard flag to skip verification."
                )
            else:
                print(f"[guard] WARNING: unregistered checkpoint: {path}")
                return None

        if not p.exists():
            raise CheckpointMismatch(f"[guard] File not found: {path}")

        rec = self._records[key]
        current_sha = _sha256_prefix(p)
        current_size = p.stat().st_size

        errors = []

        if current_sha != rec.sha256:
            errors.append(
                f"  SHA256 mismatch: expected {rec.sha256}, got {current_sha}\n"
                f"  File has been modified since registration."
            )

        if current_size != rec.size_bytes:
            errors.append(
                f"  Size mismatch: expected {rec.size_bytes:,}B, got {current_size:,}B"
            )

        # Try to verify epoch/loss from checkpoint metadata
        ckpt_epoch, ckpt_loss = _load_checkpoint_meta(p)
        if rec.epoch is not None and ckpt_epoch is not None:
            if rec.epoch != ckpt_epoch:
                errors.append(
                    f"  Epoch mismatch: registered ep{rec.epoch}, checkpoint says ep{ckpt_epoch}"
                )
        if rec.loss is not None and ckpt_loss is not None:
            if abs(rec.loss - ckpt_loss) > 0.001:
                errors.append(
                    f"  Loss mismatch: registered {rec.loss:.4f}, checkpoint says {ckpt_loss:.4f}"
                )

        if errors:
            raise CheckpointMismatch(
                f"[guard] CHECKPOINT MISMATCH: {path}\n" + "\n".join(errors)
            )

        if rec.protected:
            print(f"[guard] ⚠  PROTECTED checkpoint verified OK: {p.name}")
            print(f"           Note: {rec.note or 'Do not overwrite.'}")
        else:
            print(f"[guard] ✓ Checkpoint verified: {p.name} (ep{rec.epoch}, loss={rec.loss})")

        return rec

    def auto_register_if_saved(self, line: str) -> Optional[CheckpointRecord]:
        """
        Call with each training log line.
        Auto-registers when a checkpoint save line is detected.
        Returns the record if registered, None otherwise.
        """
        import re
        if not ('saved' in line.lower() or '→' in line):
            return None

        ckpt_m = re.search(r'checkpoints[\\/][\w\\/.:-]+\.pt', line)
        if not ckpt_m:
            return None

        path = Path(ckpt_m[0].replace('\\', '/'))
        if not path.exists():
            return None

        if str(path) in self._records:
            # Already registered — check if it changed (new save)
            current_sha = _sha256_prefix(path)
            if current_sha == self._records[str(path)].sha256:
                return self._records[str(path)]

        return self.register(str(path))

    def list_all(self) -> list[CheckpointRecord]:
        return list(self._records.values())

    def print_status(self):
        print(f"\n[guard] Checkpoint registry ({len(self._records)} entries)")
        print(f"{'Path':<55} {'Epoch':>6} {'Loss':>8} {'SHA256':>16} {'Protected'}")
        print("-" * 100)
        for rec in self._records.values():
            p = Path(rec.path).name
            ep   = str(rec.epoch) if rec.epoch is not None else "—"
            loss = f"{rec.loss:.4f}" if rec.loss is not None else "—"
            prot = "🔒 YES" if rec.protected else ""
            print(f"  {p:<53} {ep:>6} {loss:>8} {rec.sha256:>16}  {prot}")
        print()


# ── Pre-built registry for current NeMo-WM production checkpoints ────────────

PRODUCTION_CHECKPOINTS = {
    "checkpoints/dinov2_student/student_best.pt": {
        "note": "Original DINOv2 distilled encoder. NEVER overwrite with Sprint 6 distillation.",
        "protected": True,
    },
    "checkpoints/cwm/cwm_best.pt": {
        "epoch": 29,
        "loss": 0.5673,
        "note": "Tab 2 production checkpoint. ep29, loss 0.5673.",
        "protected": False,
    },
    "checkpoints/cwm/temporal_head_best.pt": {
        "note": "Quasimetric AUROC temporal head. Required for k-sweep eval.",
        "protected": False,
    },
}

def init_registry(registry_path: str = "watchdog_registry.json", verbose: bool = True):
    """
    Register all production checkpoints that exist on disk.
    Safe to run multiple times — only registers if not already present.
    """
    guard = CheckpointGuard(registry_path)
    for path, meta in PRODUCTION_CHECKPOINTS.items():
        p = Path(path)
        if not p.exists():
            if verbose:
                print(f"[guard] Skip (not found): {path}")
            continue
        key = str(p)
        if key in guard._records:
            if verbose:
                print(f"[guard] Already registered: {p.name}")
            continue
        guard.register(
            path      = path,
            epoch     = meta.get("epoch"),
            loss      = meta.get("loss"),
            note      = meta.get("note"),
            protected = meta.get("protected", False),
        )
    return guard


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NeMo-WM Checkpoint Guard")
    sub = parser.add_subparsers(dest="cmd")

    reg = sub.add_parser("register", help="Register a checkpoint")
    reg.add_argument("path")
    reg.add_argument("--epoch", type=int, default=None)
    reg.add_argument("--loss",  type=float, default=None)
    reg.add_argument("--note",  default=None)
    reg.add_argument("--protected", action="store_true")
    reg.add_argument("--registry", default="watchdog_registry.json")

    ver = sub.add_parser("verify", help="Verify a checkpoint")
    ver.add_argument("path")
    ver.add_argument("--registry", default="watchdog_registry.json")
    ver.add_argument("--no-strict", action="store_true")

    lst = sub.add_parser("list", help="List all registered checkpoints")
    lst.add_argument("--registry", default="watchdog_registry.json")

    ini = sub.add_parser("init", help="Register all production checkpoints")
    ini.add_argument("--registry", default="watchdog_registry.json")

    args = parser.parse_args()

    if args.cmd == "register":
        g = CheckpointGuard(args.registry)
        g.register(args.path, args.epoch, args.loss, args.note, args.protected)

    elif args.cmd == "verify":
        g = CheckpointGuard(args.registry)
        try:
            g.verify(args.path, strict=not args.no_strict)
        except CheckpointMismatch as e:
            print(str(e))
            sys.exit(1)

    elif args.cmd == "list":
        g = CheckpointGuard(args.registry)
        g.print_status()

    elif args.cmd == "init":
        init_registry(args.registry)

    else:
        parser.print_help()
