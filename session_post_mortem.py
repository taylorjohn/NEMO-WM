"""
session_post_mortem.py — CORTEX-16 Brain Session Analyser
==========================================================
Parses terminal output piped to a log file, or reads from
the live Alpaca account, and prints a clean session summary.

Usage (pipe terminal output first):
  python cortex_live_v1_fixed.py 2>&1 | Tee-Object session.log

Then run:
  python session_post_mortem.py
  python session_post_mortem.py --log session.log
  python session_post_mortem.py --live   # Alpaca account only
"""

import os
import re
import sys
import argparse
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# ── Parse heartbeat log ───────────────────────────────────────────────────────

def parse_log(path: str) -> list[dict]:
    """Parse [t=N] heartbeat lines from terminal log."""
    pattern = re.compile(
        r"\[t=(\d+)\] z=([-\d.]+) DA=([\d.]+) 5HT=([\d.]+)"
        r".*?CORT=([\d.]+) regime=(\w+) z_entry=([\d.]+)"
        r" scale=([\d.]+) wallet=\$([\d.]+)"
    )
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                rows.append({
                    "tick":     int(m.group(1)),
                    "z":        float(m.group(2)),
                    "da":       float(m.group(3)),
                    "sht":      float(m.group(4)),
                    "cortisol": float(m.group(5)),
                    "regime":   m.group(6),
                    "z_entry":  float(m.group(7)),
                    "scale":    float(m.group(8)),
                    "wallet":   float(m.group(9)),
                })
    return rows


def analyse_rows(rows: list[dict]) -> dict:
    if not rows:
        return {}

    ticks      = [r["tick"]     for r in rows]
    z_scores   = [r["z"]        for r in rows]
    da_vals    = [r["da"]       for r in rows]
    sht_vals   = [r["sht"]      for r in rows]
    cort_vals  = [r["cortisol"] for r in rows]
    wallets    = [r["wallet"]   for r in rows]
    regimes    = [r["regime"]   for r in rows]
    scales     = [r["scale"]    for r in rows]

    regime_counts = {}
    for r in regimes:
        regime_counts[r] = regime_counts.get(r, 0) + 1
    total = len(regimes)

    return {
        "ticks_logged":    len(rows),
        "tick_range":      f"{ticks[0]}–{ticks[-1]}",
        "wall_start":      wallets[0],
        "wall_end":        wallets[-1],
        "wallet_delta":    wallets[-1] - wallets[0],
        "z_min":           min(z_scores),
        "z_max":           max(z_scores),
        "z_mean":          sum(z_scores) / len(z_scores),
        "da_mean":         sum(da_vals)   / len(da_vals),
        "da_max":          max(da_vals),
        "sht_mean":        sum(sht_vals)  / len(sht_vals),
        "sht_min":         min(sht_vals),
        "cort_mean":       sum(cort_vals) / len(cort_vals),
        "cort_max":        max(cort_vals),
        "scale_mean":      sum(scales)    / len(scales),
        "regime_exploit":  f"{regime_counts.get('EXPLOIT',0)/total*100:.1f}%",
        "regime_explore":  f"{regime_counts.get('EXPLORE',0)/total*100:.1f}%",
        "regime_wait":     f"{regime_counts.get('WAIT',0)/total*100:.1f}%",
    }


# ── Alpaca live account ───────────────────────────────────────────────────────

def alpaca_summary() -> dict:
    try:
        from alpaca_trade_api.rest import REST
        api = REST(
            os.getenv("ALPACA_PAPER_KEY"),
            os.getenv("ALPACA_PAPER_SECRET"),
            "https://paper-api.alpaca.markets"
        )
        acc    = api.get_account()
        orders = api.list_orders(status="all", limit=50, after=datetime.now().strftime("%Y-%m-%d"))

        filled  = [o for o in orders if o.status == "filled"]
        entries = [o for o in filled if o.side == "buy"]
        exits   = [o for o in filled if o.side == "sell"]
        aborts  = [o for o in orders if o.status in ["canceled", "expired"]]

        return {
            "equity":          float(acc.equity),
            "cash":            float(acc.cash),
            "daytrade_count":  int(acc.daytrade_count),
            "orders_today":    len(orders),
            "filled_entries":  len(entries),
            "filled_exits":    len(exits),
            "aborts":          len(aborts),
        }
    except Exception as e:
        return {"alpaca_error": str(e)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log",  default="session.log", help="Path to terminal log file")
    p.add_argument("--live", action="store_true",   help="Alpaca account only, no log file")
    args = p.parse_args()

    print(f"\n{'='*60}")
    print(f"  CORTEX-16 Brain Session Post-Mortem")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S EST')}")
    print(f"{'='*60}")

    # ── Alpaca account ────────────────────────────────────────────────────────
    print("\n[Alpaca Account]")
    alp = alpaca_summary()
    if "alpaca_error" in alp:
        print(f"  Error: {alp['alpaca_error']}")
    else:
        print(f"  Equity:          ${alp['equity']:,.2f}")
        print(f"  Cash:            ${alp['cash']:,.2f}")
        print(f"  Day trade count: {alp['daytrade_count']}/3")
        print(f"  Orders today:    {alp['orders_today']}")
        print(f"  Filled entries:  {alp['filled_entries']}")
        print(f"  Filled exits:    {alp['filled_exits']}")
        print(f"  Aborts:          {alp['aborts']}")

    if args.live:
        print(f"{'='*60}\n")
        return

    # ── Log file ──────────────────────────────────────────────────────────────
    log_path = Path(args.log)
    if not log_path.exists():
        print(f"\n[Log] No log file at {log_path}")
        print("  Tip: pipe terminal output with:")
        print("  python cortex_live_v1_fixed.py 2>&1 | Tee-Object session.log")
        print(f"{'='*60}\n")
        return

    rows = parse_log(str(log_path))
    if not rows:
        print(f"\n[Log] No heartbeat lines found in {log_path}")
        print(f"{'='*60}\n")
        return

    s = analyse_rows(rows)

    print(f"\n[Brain Session — {log_path}]")
    print(f"  Heartbeats logged: {s['ticks_logged']} × 100 ticks")
    print(f"  Tick range:        {s['tick_range']}")
    print(f"  Wallet start:      ${s['wall_start']:.2f}")
    print(f"  Wallet end:        ${s['wall_end']:.2f}")
    delta = s['wallet_delta']
    sign  = "+" if delta >= 0 else ""
    print(f"  Wallet delta:      {sign}${delta:.2f}")

    print(f"\n[Signal Summary]")
    print(f"  z-score range:     {s['z_min']:.3f} → {s['z_max']:.3f}  (mean {s['z_mean']:.3f})")
    print(f"  DA mean/max:       {s['da_mean']:.3f} / {s['da_max']:.3f}")
    print(f"  5HT mean/min:      {s['sht_mean']:.3f} / {s['sht_min']:.3f}")
    print(f"  Cortisol mean/max: {s['cort_mean']:.3f} / {s['cort_max']:.3f}")
    print(f"  Position scale:    {s['scale_mean']:.3f} mean")

    print(f"\n[Regime Distribution]")
    print(f"  EXPLOIT:  {s['regime_exploit']}")
    print(f"  EXPLORE:  {s['regime_explore']}")
    print(f"  WAIT:     {s['regime_wait']}")

    print(f"\n[Diagnosis]")
    if alp.get("filled_entries", 0) == 0:
        print("  ⚠️  0 entries — check entry gate wiring (dynamic_z_threshold)")
    if float(s["regime_wait"].rstrip("%")) > 50:
        print("  ⚠️  >50% WAIT — 5HT or cortisol too sensitive")
    if s["cort_max"] >= 1.9:
        print("  ⚠️  Cortisol hit ceiling — cold-start or RTT spike")
    if s["da_mean"] > 0.5:
        print("  ✅ DA active — z-score variance present, signal working")
    if alp.get("daytrade_count", 0) < 3:
        print(f"  ✅ {3 - alp.get('daytrade_count',0)} day trades remaining tomorrow")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
