"""
log_parser.py — NeMo-WM training log parser
============================================
Parses all three training log formats into structured events.

Formats handled:
  World model:   [ep07 s287500] loss=0.7882 L_jepa=0.5000 regime=REOBSERVE DA=0.001
  Multidomain:   [ep21 s716000] domain=recon loss=0.0744 DA=0.001 5HT=0.108 ACh=0.446
  Dual-head NCE: [ep00 s00600] L_clip=1.4819 L_null=0.0004 lr=3.00e-04

Author: John Taylor — github.com/taylorjohn
"""

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Callable


class LogFormat(Enum):
    WORLD_MODEL   = "world_model"    # Tab 1/2 — L_jepa, DA, regime
    MULTIDOMAIN   = "multidomain"    # Sprint 3 — domain=, 5HT, ACh
    DUAL_HEAD_NCE = "dual_head_nce"  # Sprint 6d — L_clip, L_null


@dataclass
class StepEvent:
    """Emitted on every parsed log line."""
    tab:    str
    epoch:  int
    step:   int
    fmt:    LogFormat
    loss:   Optional[float] = None
    da:     Optional[float] = None
    sht:    Optional[float] = None
    ach:    Optional[float] = None
    l_jepa: Optional[float] = None
    l_clip: Optional[float] = None
    l_null: Optional[float] = None
    lr:     Optional[float] = None
    regime: Optional[str]   = None
    domain: Optional[str]   = None


@dataclass
class EpochEvent:
    """Emitted when an epoch completes."""
    tab:        str
    epoch:      int
    fmt:        LogFormat
    mean_loss:  Optional[float] = None
    mean_lclip: Optional[float] = None
    checkpoint_saved: bool = False
    checkpoint_path:  Optional[str] = None
    elapsed_s:  Optional[float] = None


@dataclass
class AlertEvent:
    """Emitted when a threshold is crossed."""
    tab:    str
    kind:   str           # 'da_spike', 'da_zero_streak', 'lclip_stall', etc.
    detail: str
    step:   int
    epoch:  int
    value:  Optional[float] = None


# ── Regex patterns ───────────────────────────────────────────────────────────

_STEP_WM = re.compile(
    r'\[ep(\d+)\s+s(\d+)\]\s+loss=([\d.]+)\s+L_jepa=([\d.]+)\s+'
    r'regime=(\w+)\s+DA=([\d.]+)'
)
_STEP_MD = re.compile(
    r'\[ep(\d+)\s+s(\d+)\]\s+domain=(\w+)\s+loss=([\d.]+)\s+'
    r'regime=(\w+)\s+DA=([\d.]+)\s+5HT=([\d.]+)\s+ACh=([\d.]+)'
)
_STEP_DH = re.compile(
    r'\[ep(\d+)\s+s(\d+)\]\s+L_clip=([\d.]+)\s+L_null=([\d.]+)'
)
_STEP_DH_NOLNULL = re.compile(
    r'\[ep(\d+)\s+s(\d+)\]\s+L_clip=([\d.]+)\s+lr='
)
_EPOCH_WM = re.compile(
    r'Epoch\s+(\d+)\s+(?:mean=|mean_loss=)([\d.]+)'
)
_EPOCH_DH = re.compile(
    r'Epoch\s+(\d+)\s+L_clip=([\d.]+)\s+\(([\d.]+)s\)'
)
_EPOCH_MD = re.compile(
    r'Epoch\s+(\d+)\s+mean_loss=([\d.]+)'
)
_SAVED_GENERIC = re.compile(
    r'(?:→\s*saved|Saved:|-> saved).*?(?:checkpoints[\\/].+?\.pt)?'
)
_CKPT_PATH = re.compile(r'checkpoints[\\/\w.-]+\.pt')


def detect_format(line: str) -> Optional[LogFormat]:
    if _STEP_MD.search(line):
        return LogFormat.MULTIDOMAIN
    if _STEP_DH.search(line) or _STEP_DH_NOLNULL.search(line):
        return LogFormat.DUAL_HEAD_NCE
    if _STEP_WM.search(line):
        return LogFormat.WORLD_MODEL
    return None


def parse_step(line: str, tab: str, fmt: Optional[LogFormat] = None) -> Optional[StepEvent]:
    if fmt is None:
        fmt = detect_format(line)
    if fmt is None:
        return None

    if fmt == LogFormat.WORLD_MODEL:
        m = _STEP_WM.search(line)
        if not m:
            return None
        return StepEvent(
            tab=tab, epoch=int(m[1]), step=int(m[2]),
            fmt=fmt, loss=float(m[3]), l_jepa=float(m[4]),
            regime=m[5], da=float(m[6])
        )

    elif fmt == LogFormat.MULTIDOMAIN:
        m = _STEP_MD.search(line)
        if not m:
            return None
        return StepEvent(
            tab=tab, epoch=int(m[1]), step=int(m[2]),
            fmt=fmt, domain=m[3], loss=float(m[4]),
            regime=m[5], da=float(m[6]),
            sht=float(m[7]), ach=float(m[8])
        )

    elif fmt == LogFormat.DUAL_HEAD_NCE:
        m = _STEP_DH.search(line)
        if m:
            lr_m = re.search(r'lr=([\d.e+-]+)', line)
            return StepEvent(
                tab=tab, epoch=int(m[1]), step=int(m[2]),
                fmt=fmt, l_clip=float(m[3]), l_null=float(m[4]),
                lr=float(lr_m[1]) if lr_m else None
            )
        m = _STEP_DH_NOLNULL.search(line)
        if m:
            lr_m = re.search(r'lr=([\d.e+-]+)', line)
            return StepEvent(
                tab=tab, epoch=int(m[1]), step=int(m[2]),
                fmt=fmt, l_clip=float(m[3]),
                lr=float(lr_m[1]) if lr_m else None
            )
    return None


def parse_epoch(line: str, tab: str) -> Optional[EpochEvent]:
    m = _EPOCH_DH.search(line)
    if m:
        return EpochEvent(
            tab=tab, epoch=int(m[1]), fmt=LogFormat.DUAL_HEAD_NCE,
            mean_lclip=float(m[2]), elapsed_s=float(m[3])
        )
    m = _EPOCH_MD.search(line)
    if m:
        return EpochEvent(
            tab=tab, epoch=int(m[1]), fmt=LogFormat.MULTIDOMAIN,
            mean_loss=float(m[2])
        )
    m = _EPOCH_WM.search(line)
    if m:
        return EpochEvent(
            tab=tab, epoch=int(m[1]), fmt=LogFormat.WORLD_MODEL,
            mean_loss=float(m[2])
        )
    return None


# ── Alert state machine ──────────────────────────────────────────────────────

class AlertState:
    """Tracks running state needed to fire threshold alerts."""
    def __init__(self, tab: str, cfg: dict):
        self.tab = tab
        self.cfg = cfg
        self.da_zero_streak  = 0
        self.last_epoch_loss: Optional[float] = None
        self.lclip_history:  list[float] = []
        self.lnull_activated = False
        self.last_epoch_lclip: Optional[float] = None
        self.lclip_stall_count = 0

    def process_step(self, ev: StepEvent) -> list[AlertEvent]:
        alerts = []
        thresh = self.cfg.get("alerts", {})

        # DA zero streak (world model only)
        if ev.da is not None:
            if ev.da == 0.0:
                self.da_zero_streak += 1
                limit = thresh.get("da_zero_streak_threshold", 100_000)
                if self.da_zero_streak == limit:
                    alerts.append(AlertEvent(
                        tab=self.tab, kind="da_zero_streak",
                        detail=f"DA=0.000 for {limit:,} consecutive steps",
                        step=ev.step, epoch=ev.epoch
                    ))
            else:
                self.da_zero_streak = 0
                if ev.da >= thresh.get("da_spike_threshold", 0.003):
                    alerts.append(AlertEvent(
                        tab=self.tab, kind="da_spike",
                        detail=f"DA={ev.da:.3f} at step {ev.step:,}",
                        step=ev.step, epoch=ev.epoch, value=ev.da
                    ))

        # L_null first activation (dual head)
        if ev.l_null is not None and ev.l_null > 0.0 and not self.lnull_activated:
            self.lnull_activated = True
            alerts.append(AlertEvent(
                tab=self.tab, kind="lnull_active",
                detail=f"L_null first activated: {ev.l_null:.4f} at step {ev.step:,}",
                step=ev.step, epoch=ev.epoch, value=ev.l_null
            ))

        return alerts

    def process_epoch(self, ev: EpochEvent) -> list[AlertEvent]:
        alerts = []
        thresh = self.cfg.get("alerts", {})

        # Loss spike (world model / multidomain)
        if ev.mean_loss is not None and self.last_epoch_loss is not None:
            pct = (ev.mean_loss - self.last_epoch_loss) / self.last_epoch_loss * 100
            spike_limit = thresh.get("loss_spike_pct", 15)
            if pct > spike_limit:
                alerts.append(AlertEvent(
                    tab=self.tab, kind="loss_spike",
                    detail=f"Loss spike +{pct:.1f}%: {self.last_epoch_loss:.4f} -> {ev.mean_loss:.4f}",
                    step=0, epoch=ev.epoch, value=pct
                ))
        if ev.mean_loss is not None:
            self.last_epoch_loss = ev.mean_loss

        # L_clip stall (dual head)
        if ev.mean_lclip is not None:
            stall_epochs = thresh.get("lclip_stall_epochs", 3)
            if self.last_epoch_lclip is not None:
                delta = abs(ev.mean_lclip - self.last_epoch_lclip)
                if delta < 0.005:
                    self.lclip_stall_count += 1
                    if self.lclip_stall_count >= stall_epochs:
                        alerts.append(AlertEvent(
                            tab=self.tab, kind="lclip_stall",
                            detail=f"L_clip stalled at ~{ev.mean_lclip:.4f} for {self.lclip_stall_count} epochs",
                            step=0, epoch=ev.epoch, value=ev.mean_lclip
                        ))
                else:
                    self.lclip_stall_count = 0
            self.last_epoch_lclip = ev.mean_lclip

        # L_null expected by epoch N but not yet activated
        expected_by = thresh.get("lnull_expected_active_by_epoch", 2)
        if ev.epoch >= expected_by and not self.lnull_activated:
            alerts.append(AlertEvent(
                tab=self.tab, kind="lnull_not_active",
                detail=f"L_null not activated by epoch {expected_by} — increase null_weight",
                step=0, epoch=ev.epoch
            ))

        return alerts


# ── File tailer ──────────────────────────────────────────────────────────────

def tail_file(
    path: Path,
    tab: str,
    cfg: dict,
    on_step:  Callable[[StepEvent], None],
    on_epoch: Callable[[EpochEvent], None],
    on_alert: Callable[[AlertEvent], None],
    poll_interval: float = 2.0,
    from_start: bool = False,
):
    """
    Tail a training log file, emitting structured events.
    Runs indefinitely — call from a thread.
    """
    state = AlertState(tab, cfg)
    current_fmt: Optional[LogFormat] = None
    pending_epoch: Optional[EpochEvent] = None
    checkpoint_next = False

    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        if not from_start:
            f.seek(0, 2)          # seek to end

        while True:
            line = f.readline()

            if not line:
                time.sleep(poll_interval)
                continue

            line = line.rstrip()

            # Auto-detect format from first step line seen
            if current_fmt is None:
                current_fmt = detect_format(line)

            # Step event
            ev = parse_step(line, tab, current_fmt)
            if ev:
                on_step(ev)
                for a in state.process_step(ev):
                    on_alert(a)
                continue

            # Epoch event
            ep = parse_epoch(line, tab)
            if ep:
                pending_epoch = ep
                checkpoint_next = True
                for a in state.process_epoch(ep):
                    on_alert(a)
                continue

            # Checkpoint save line
            if checkpoint_next and _SAVED_GENERIC.search(line):
                ckpt_m = _CKPT_PATH.search(line)
                if pending_epoch:
                    pending_epoch.checkpoint_saved = True
                    pending_epoch.checkpoint_path = ckpt_m[0] if ckpt_m else None
                    on_epoch(pending_epoch)
                    pending_epoch = None
                    checkpoint_next = False
                continue

            # Flush pending epoch if next step line appears
            if pending_epoch and '[ep' in line:
                on_epoch(pending_epoch)
                pending_epoch = None
                checkpoint_next = False


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python log_parser.py <logfile> [tab_name]")
        sys.exit(1)
    log = Path(sys.argv[1])
    tab = sys.argv[2] if len(sys.argv) > 2 else log.stem

    def show_step(ev):
        if ev.da and ev.da > 0:
            print(f"  STEP  ep{ev.epoch} s{ev.step:,} | loss={ev.loss} DA={ev.da}")

    def show_epoch(ev):
        print(f"  EPOCH ep{ev.epoch} | loss={ev.mean_loss or ev.mean_lclip:.4f} | ckpt={ev.checkpoint_saved}")

    def show_alert(ev):
        print(f"  *** ALERT [{ev.kind}] ep{ev.epoch}: {ev.detail}")

    tail_file(log, tab, {}, show_step, show_epoch, show_alert, from_start=True)
