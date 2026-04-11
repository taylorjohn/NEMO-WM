"""
status_updater.py — Auto-updates CWM_STATUS.md with training events
=====================================================================
Appends structured epoch summaries and alerts without human intervention.
Replaces the manual paste-and-document loop.

Author: John Taylor — github.com/taylorjohn
"""

import time
from pathlib import Path
from typing import Optional
from log_parser import StepEvent, EpochEvent, AlertEvent, LogFormat


class StatusUpdater:
    """
    Maintains a running summary of all training activity and
    auto-appends to CWM_STATUS.md.
    """

    def __init__(
        self,
        status_path: str = "CWM_STATUS.md",
        flush_every_n_epochs: int = 1,
    ):
        self.status_path = Path(status_path)
        self.flush_every  = flush_every_n_epochs

        # Per-tab running state
        self._tabs: dict[str, "_TabState"] = {}
        # Pending lines to flush
        self._pending: list[str] = []

    def _tab(self, tab: str) -> "_TabState":
        if tab not in self._tabs:
            self._tabs[tab] = _TabState(tab)
        return self._tabs[tab]

    # ── Public callbacks (wire to log_parser.tail_file) ──────────────────────

    def on_step(self, ev: StepEvent):
        t = self._tab(ev.tab)
        t.process_step(ev)

    def on_epoch(self, ev: EpochEvent):
        t = self._tab(ev.tab)
        t.process_epoch(ev)
        lines = t.format_epoch_block(ev)
        self._append_to_status(lines)

    def on_alert(self, ev: AlertEvent):
        lines = [
            f"\n### ⚠ ALERT [{ev.kind.upper()}] — {ev.tab} ep{ev.epoch} — {_ts()}",
            f"{ev.detail}",
            "",
        ]
        self._append_to_status(lines)
        # Also print to console immediately
        print(f"\n*** WATCHDOG ALERT [{ev.kind}] {ev.tab} ep{ev.epoch}: {ev.detail}")

    # ── Internal ─────────────────────────────────────────────────────────────

    def _append_to_status(self, lines: list[str]):
        block = "\n".join(lines) + "\n"
        with open(self.status_path, 'a', encoding='utf-8') as f:
            f.write(block)


class _TabState:
    """Running state for one training tab."""

    def __init__(self, tab: str):
        self.tab = tab
        self.da_readings:    list[float] = []
        self.da_nonzero:     list[tuple[int, float]] = []  # (step, value)
        self.loss_readings:  list[float] = []
        self.lclip_readings: list[float] = []
        self.lnull_readings: list[float] = []
        self.max_da:         float = 0.0
        self.max_da_step:    int   = 0
        self.lnull_activated_step: Optional[int] = None
        self.last_epoch:     int   = -1
        self.fmt:            Optional[LogFormat] = None

    def process_step(self, ev: StepEvent):
        self.fmt = ev.fmt
        if ev.loss is not None:
            self.loss_readings.append(ev.loss)
        if ev.da is not None:
            self.da_readings.append(ev.da)
            if ev.da > 0:
                self.da_nonzero.append((ev.step, ev.da))
                if ev.da > self.max_da:
                    self.max_da = ev.da
                    self.max_da_step = ev.step
        if ev.l_clip is not None:
            self.lclip_readings.append(ev.l_clip)
        if ev.l_null is not None:
            self.lnull_readings.append(ev.l_null)
            if ev.l_null > 0 and self.lnull_activated_step is None:
                self.lnull_activated_step = ev.step

    def process_epoch(self, ev: EpochEvent):
        self.last_epoch = ev.epoch

    def format_epoch_block(self, ev: EpochEvent) -> list[str]:
        ts = _ts()
        lines = [f"\n### Auto [{self.tab}] Epoch {ev.epoch:02d} — {ts}"]

        # Loss summary
        if ev.mean_loss is not None:
            prev = self.loss_readings[-100] if len(self.loss_readings) > 100 else None
            delta = ""
            if prev is not None:
                d = ev.mean_loss - prev
                delta = f" (delta {d:+.4f})"
            lines.append(f"- Loss: **{ev.mean_loss:.4f}**{delta}")

        # L_clip summary (dual head)
        if ev.mean_lclip is not None:
            lines.append(f"- L_clip: **{ev.mean_lclip:.4f}**")
            if self.lclip_readings:
                mn = min(self.lclip_readings[-1000:])
                mx = max(self.lclip_readings[-1000:])
                lines.append(f"- L_clip range this epoch: {mn:.4f} – {mx:.4f}")

        # L_null status
        if self.fmt == LogFormat.DUAL_HEAD_NCE:
            if self.lnull_activated_step:
                lnull_mean = sum(x for x in self.lnull_readings[-500:] if x > 0)
                count = sum(1 for x in self.lnull_readings[-500:] if x > 0)
                mean_str = f"{lnull_mean/count:.4f}" if count else "0.0000"
                lines.append(f"- L_null: **ACTIVE** (first at step {self.lnull_activated_step:,}, mean non-zero: {mean_str})")
            else:
                lines.append(f"- L_null: 0.0000 (pre-activation — expected until alignment builds)")

        # DA summary (world model / multidomain)
        if self.da_readings:
            nonzero_this_epoch = [(s, d) for s, d in self.da_nonzero
                                  if s > self.max_da_step - 50000]
            zero_pct = 100 * (1 - len([d for d in self.da_readings[-1000:] if d > 0]) /
                              max(1, len(self.da_readings[-1000:])))
            lines.append(f"- DA: peak={self.max_da:.3f} at step {self.max_da_step:,} | zero {zero_pct:.1f}% this epoch")
            if self.da_nonzero:
                recent = self.da_nonzero[-3:]
                lines.append(f"- DA non-zero events (recent): {[(s, d) for s, d in recent]}")
            else:
                lines.append("- DA: **0.000 throughout** — ablation property holding")

        # Checkpoint
        if ev.checkpoint_saved:
            lines.append(f"- Checkpoint: **saved** {'→ ' + ev.checkpoint_path if ev.checkpoint_path else ''}")

        # Elapsed
        if ev.elapsed_s is not None:
            lines.append(f"- Epoch time: {ev.elapsed_s:.0f}s ({ev.elapsed_s/3600:.1f}h)")

        # Clear per-epoch accumulators
        self.loss_readings  = []
        self.da_readings    = []
        self.lclip_readings = []
        self.lnull_readings = []

        lines.append("")
        return lines


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M")


if __name__ == "__main__":
    # Quick smoke test with synthetic events
    import tempfile, os
    from log_parser import LogFormat

    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write("# Test Status\n\n")
        tmp = f.name

    up = StatusUpdater(status_path=tmp)

    # Simulate world model events
    ev_step = StepEvent(tab="tab1", epoch=7, step=280000,
                        fmt=LogFormat.WORLD_MODEL, loss=0.789, da=0.001)
    up.on_step(ev_step)

    ev_epoch = EpochEvent(tab="tab1", epoch=7, fmt=LogFormat.WORLD_MODEL,
                          mean_loss=0.797, checkpoint_saved=True,
                          checkpoint_path="checkpoints/cwm/cwm_random_encoder_best.pt")
    up.on_epoch(ev_epoch)

    alert = AlertEvent(tab="tab1", kind="da_zero_streak",
                       detail="DA=0.000 for 100,000 steps", step=280000, epoch=7)
    up.on_alert(alert)

    with open(tmp) as f:
        print(f.read())
    os.unlink(tmp)
    print("Smoke test passed.")
