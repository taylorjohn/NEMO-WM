"""
compare_sessions.py — CORTEX-16 Session Comparison Tool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Reads all JSON files from the sessions/ folder and prints
a side-by-side comparison table, then a per-scenario breakdown.

Usage:
    python compare_sessions.py                  # compare all sessions
    python compare_sessions.py --last 4         # last 4 sessions only
    python compare_sessions.py --label 10min    # sessions matching label
    python compare_sessions.py --sort gain      # sort by gain%
"""
import argparse, json, os, glob
from typing import List, Dict

parser = argparse.ArgumentParser()
parser.add_argument("--last",  type=int,   default=None, help="Show last N sessions")
parser.add_argument("--label", type=str,   default=None, help="Filter by label substring")
parser.add_argument("--sort",  type=str,   default="timestamp",
                    choices=["timestamp","gain","pnl","trades","winrate","duration"])
args = parser.parse_args()

# ── Load sessions ─────────────────────────────────────────────────────────────
files = sorted(glob.glob("sessions/*.json"))
if not files:
    print("\n  No sessions found in sessions/ folder.")
    print("  Run: python market_replay.py --duration 10 --session-name 10min\n")
    exit(0)

sessions: List[Dict] = []
for f in files:
    try:
        with open(f) as fp:
            s = json.load(fp)
            s["_file"] = f
            sessions.append(s)
    except Exception as e:
        print(f"  Skipping {f}: {e}")

# Filter
if args.label:
    sessions = [s for s in sessions if args.label.lower() in s.get("label","").lower()]

# Sort
sort_key = {
    "timestamp": lambda s: s.get("timestamp",""),
    "gain":      lambda s: s.get("gain_pct", 0),
    "pnl":       lambda s: s.get("total_pnl", 0),
    "trades":    lambda s: s.get("trades", 0),
    "winrate":   lambda s: s.get("win_rate", 0),
    "duration":  lambda s: s.get("duration_min", 0),
}.get(args.sort, lambda s: s.get("timestamp",""))

sessions.sort(key=sort_key)
if args.last:
    sessions = sessions[-args.last:]

if not sessions:
    print(f"\n  No sessions match filter '{args.label}'\n")
    exit(0)

# ── Summary table ─────────────────────────────────────────────────────────────
W = 14
print(f"""
  ╔══════════════════════════════════════════════════════════════════════════════════════╗
  ║  CORTEX-16  SESSION COMPARISON  ({len(sessions)} sessions)
  ╚══════════════════════════════════════════════════════════════════════════════════════╝
""")

col_w = max(12, max(len(s.get("label","?")[:12]) for s in sessions) + 2)
header = (f"  {'METRIC':<22}" +
          "".join(f"  {s.get('label','?')[:col_w]:<{col_w}}" for s in sessions))
print(header)
print("  " + "─" * (22 + (col_w + 2) * len(sessions)))

def row(label, getter, fmt=lambda x: str(x), color_fn=None):
    vals = [getter(s) for s in sessions]
    line = f"  {label:<22}"
    for v in vals:
        cell = fmt(v)[:col_w]
        line += f"  {cell:<{col_w}}"
    print(line)

def pnl_fmt(v):
    return f"+${v:,.2f}" if v >= 0 else f"-${abs(v):,.2f}"

def pct_fmt(v):
    return f"{v:+.3f}%"

row("Duration (min)",  lambda s: s.get("duration_min",0),  lambda v: f"{v:.1f} min")
row("Ticks",           lambda s: s.get("ticks",0),          lambda v: f"{v:,}")
row("Hz",              lambda s: s.get("hz",0),             lambda v: f"{v:.1f}")
print()
row("Total Trades",    lambda s: s.get("trades",0),         lambda v: f"{v:,}")
row("Wins",            lambda s: s.get("wins",0),           lambda v: f"{v:,}")
row("Losses",          lambda s: s.get("losses",0),         lambda v: f"{v:,}")
row("Win Rate",        lambda s: s.get("win_rate",0),       lambda v: f"{v:.1f}%")
print()
row("Total PnL",       lambda s: s.get("total_pnl",0),      pnl_fmt)
row("Vault Final",     lambda s: s.get("vault_final",0),    lambda v: f"${v:,.2f}")
row("Banked",          lambda s: s.get("banked",0),         lambda v: f"${v:,.2f}")
row("Gain %",          lambda s: s.get("gain_pct",0),       pct_fmt)
print()
row("Trades/min",      lambda s: s.get("trades",0)/max(s.get("duration_min",1),0.01),
                                                             lambda v: f"{v:.1f}")
row("PnL/trade",       lambda s: s.get("total_pnl",0)/max(s.get("trades",1),1),
                                                             pnl_fmt)

# ── Per-scenario breakdown across sessions ────────────────────────────────────
all_scenarios = sorted(set(
    sname for s in sessions for sname in s.get("scenarios",{}).keys()
))
if all_scenarios:
    print(f"""
  ── SCENARIO BREAKDOWN ───────────────────────────────────────────────────────
  {'SCENARIO':<16}  {'SESSION':<14}  {'TRADES':>6}  {'WR':>6}  {'PnL':>10}  {'STOP':>5}  {'SIZE':>5}  {'CONF':>5}
  {"─"*75}""")
    for sname in all_scenarios:
        first = True
        for s in sessions:
            sc = s.get("scenarios",{}).get(sname)
            if not sc: continue
            label = s.get("label","?")[:14] if first else ""
            deg = "⚠" if sc.get("degraded") else " "
            print(f"  {deg}{sname:<15}  {label:<14}  {sc['trades']:>6}  "
                  f"{sc['win_rate']:>5.1f}%  {sc['total_pnl']:>+10.2f}  "
                  f"{sc['stop_pct']:>4.2f}%  {sc['size_scale']:>4.2f}x  "
                  f"{sc['confidence']:>5.3f}")
            first = False
        if len(sessions) > 1:
            print()

# ── Per-tier breakdown ────────────────────────────────────────────────────────
all_tiers = sorted(set(
    tier for s in sessions for tier in s.get("tiers",{}).keys()
))
if all_tiers:
    print(f"""  ── TIER BREAKDOWN ───────────────────────────────────────────────────────────
  {'TIER':<10}  {'SESSION':<14}  {'TRADES':>6}  {'WR':>6}  {'AVG PnL':>9}  {'MAX WIN':>9}  {'MAX LOSS':>10}
  {"─"*72}""")
    for tier in all_tiers:
        first = True
        for s in sessions:
            t = s.get("tiers",{}).get(tier)
            if not t: continue
            label = s.get("label","?")[:14] if first else ""
            print(f"  {tier:<10}  {label:<14}  {t['trades']:>6}  "
                  f"{t['win_rate']:>5.1f}%  {t['avg_pnl']:>+9.2f}  "
                  f"{t['max_win']:>+9.2f}  {t['max_loss']:>+10.2f}")
            first = False
        if len(sessions) > 1:
            print()

# ── Exit reason summary ───────────────────────────────────────────────────────
print(f"""  ── EXIT REASONS ─────────────────────────────────────────────────────────────
  {'REASON':<10}  {'SESSION':<14}  {'COUNT':>6}  {'TOTAL PnL':>12}
  {"─"*48}""")
all_reasons = sorted(set(
    r for s in sessions for r in s.get("exit_reasons",{}).keys()
))
for reason in all_reasons:
    first = True
    for s in sessions:
        er = s.get("exit_reasons",{}).get(reason)
        if not er: continue
        label = s.get("label","?")[:14] if first else ""
        print(f"  {reason:<10}  {label:<14}  {er['count']:>6}  {er['total_pnl']:>+12.2f}")
        first = False
    if len(sessions) > 1:
        print()

# ── Learning progress (stop/target evolution) ─────────────────────────────────
print(f"""  ── LEARNING EVOLUTION (how stop/target adapted over sessions) ───────────────
  Scenario SATURATED:""")
for s in sessions:
    sc = s.get("scenarios",{}).get("SATURATED")
    if sc:
        print(f"    [{s.get('label','?'):<12}]  "
              f"stop={sc['stop_pct']:.2f}%  "
              f"target={sc['target_pct']:.2f}%  "
              f"size={sc['size_scale']:.2f}x  "
              f"conf={sc['confidence']:.3f}  "
              f"pnl={sc['total_pnl']:+.2f}")
print()
print(f"  ── {len(sessions)} sessions loaded from sessions/")
print(f"  Add sessions with: python market_replay.py --duration 60 --session-name 1hour\n")
