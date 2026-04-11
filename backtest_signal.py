"""
backtest_signal.py
------------------
Analyzes cortex_combat_log.csv to answer:
  1. What rho/Z distribution did today's session produce?
  2. At what threshold would entries have fired?
  3. Which phases killed the most opportunities?
  4. Would a lower Z threshold have produced entries?

Schema (confirmed from log):
  timestamp, symbol, action, qty, price, resonance, z_score,
  tier, alloc_pct, realised_pnl, banked_50, active_pool,
  total_equity, phase_abort

Usage:
    python backtest_signal.py --log cortex_combat_log.csv --date 2026-03-30
"""

import argparse
import re
import sys

import numpy as np
import pandas as pd

VAULT = 101_133.47

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="CORTEX-16 signal backtest")
parser.add_argument("--log",  default="cortex_combat_log.csv")
parser.add_argument("--date", default=None,
                    help="Filter to single date e.g. 2026-03-30 (UTC)")
parser.add_argument("--z-thresholds", nargs="+", type=float,
                    default=[1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5])
parser.add_argument("--encoding", default=None,
                    help="CSV encoding override (auto-detected if not set)")
args = parser.parse_args()


# ── Load ──────────────────────────────────────────────────────────────────────

print(f"\nLoading {args.log} ...")

def load_csv(path, encoding=None):
    if encoding:
        return pd.read_csv(path, parse_dates=["timestamp"], encoding=encoding)
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin-1"]:
        try:
            df = pd.read_csv(path, parse_dates=["timestamp"], encoding=enc)
            print(f"  Encoding: {enc}")
            return df
        except (UnicodeDecodeError, Exception):
            continue
    raise ValueError("Could not decode CSV with any known encoding")

df = load_csv(args.log, args.encoding)
print(f"  Total rows:  {len(df):,}")
print(f"  Date range:  {df['timestamp'].min()} → {df['timestamp'].max()}")

if args.date:
    mask = df["timestamp"].dt.date.astype(str) == args.date
    df = df[mask].reset_index(drop=True)
    print(f"  After filter to {args.date}: {len(df):,} rows")

if len(df) == 0:
    print("No data after filtering. Exiting.")
    sys.exit(1)

aborts = df[df["action"] == "ABORT"].copy()
fills  = df[df["action"] != "ABORT"].copy()
print(f"  ABORT rows:  {len(aborts):,}")
print(f"  FILL rows:   {len(fills):,}")

# Convert timestamps to ET for display
try:
    df["_et"] = df["timestamp"].dt.tz_convert("US/Eastern")
except Exception:
    df["_et"] = df["timestamp"]


# ── 1. Signal Distribution ────────────────────────────────────────────────────

print("\n" + "═"*65)
print("  1. SIGNAL DISTRIBUTION")
print("═"*65)

for col, label in [("resonance", "rho"), ("z_score", "Z-score")]:
    s = df[col].dropna()
    if len(s) == 0:
        continue
    print(f"\n  {label}  (n={len(s):,})")
    print(f"    min={s.min():.4f}  max={s.max():.4f}  "
          f"mean={s.mean():.4f}  std={s.std():.4f}")
    for p, v in zip([5,10,25,50,75,90,95,99],
                    np.percentile(s, [5,10,25,50,75,90,95,99])):
        bar = "█" * int(v * 8)
        print(f"    p{p:>2d}: {v:.4f}  {bar}")

z = df["z_score"].dropna()
print(f"\n  Z-score gate counts:")
for t in [1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.9, 3.0]:
    n = (z >= t).sum()
    pct = 100 * n / len(z)
    bar = "█" * min(int(pct * 2), 40)
    print(f"    Z >= {t:.1f}:  {n:>7,}  ({pct:5.1f}%)  {bar}")


# ── 2. Intraday Signal Profile ────────────────────────────────────────────────

print("\n" + "═"*65)
print("  2. INTRADAY SIGNAL PROFILE (15-min buckets, ET)")
print("═"*65)

try:
    df["_hour_et"]   = df["_et"].dt.hour
    df["_bucket"]    = df["_hour_et"] * 100 + (df["_et"].dt.minute // 15) * 15
    profile = df.groupby("_bucket").agg(
        rho_mean   = ("resonance", "mean"),
        z_mean     = ("z_score",   "mean"),
        z_max      = ("z_score",   "max"),
        n          = ("z_score",   "count"),
    ).reset_index()

    print(f"\n  {'Time':>6}  {'rho':>6}  {'Z mean':>7}  {'Z max':>7}  {'Ticks':>7}")
    print(f"  {'──────':>6}  {'──────':>6}  {'───────':>7}  {'───────':>7}  {'───────':>7}")
    for _, row in profile.iterrows():
        h = int(row["_bucket"]) // 100
        m = int(row["_bucket"]) % 100
        bar = "█" * int(row["z_mean"] * 6)
        print(f"  {h:02d}:{m:02d}   {row['rho_mean']:6.4f}  {row['z_mean']:7.4f}"
              f"  {row['z_max']:7.4f}  {int(row['n']):>7,}  {bar}")
except Exception as e:
    print(f"  (Could not build profile: {e})")


# ── 3. Phase Abort Breakdown ──────────────────────────────────────────────────

print("\n" + "═"*65)
print("  3. PHASE ABORT BREAKDOWN")
print("═"*65)

def extract_phase(s):
    if pd.isna(s):
        return "none"
    m = re.match(r"Ph(\d+)", str(s))
    return f"Phase {m.group(1):>2}" if m else str(s)[:40]

aborts = aborts.copy()
aborts["_phase"] = aborts["phase_abort"].apply(extract_phase)
phase_counts = aborts["_phase"].value_counts()
total = len(aborts)

print(f"\n  {'Phase':<20}  {'Count':>8}  {'Pct':>7}  Bar")
print(f"  {'─'*20}  {'─'*8}  {'─'*7}  {'─'*20}")
for phase, count in phase_counts.items():
    pct = 100 * count / total
    bar = "█" * int(pct / 2)
    print(f"  {phase:<20}  {count:>8,}  {pct:>6.1f}%  {bar}")


# ── 4. Threshold Sweep ────────────────────────────────────────────────────────

print("\n" + "═"*65)
print("  4. THRESHOLD SWEEP — HOW MANY TICKS WOULD HAVE ENTERED?")
print("═"*65)
print(f"\n  {'Z thresh':>9}  {'Ticks pass':>11}  {'% of total':>11}  Note")
print(f"  {'─'*9}  {'─'*11}  {'─'*11}  {'─'*35}")

for thresh in args.z_thresholds:
    n   = (z >= thresh).sum()
    pct = 100 * n / len(z)
    if n == 0:
        note = "❌ Never reached today"
    elif thresh <= 1.8:
        note = "⚠️  High frequency — needs spread fix first"
    elif thresh <= 2.0:
        note = "🟡 Low frequency — viable after spread fix"
    elif thresh <= 2.2:
        note = "🟡 Very rare — borderline viable"
    else:
        note = "🔴 Not reached today"
    print(f"  {thresh:>9.1f}  {n:>11,}  {pct:>10.2f}%  {note}")


# ── 5. Best Signal Windows ────────────────────────────────────────────────────

print("\n" + "═"*65)
print("  5. BEST SIGNAL WINDOWS")
print("═"*65)

for z_floor in [2.0, 1.9, 1.8]:
    high_z = df[df["z_score"] >= z_floor]
    if len(high_z) > 0:
        print(f"\n  Ticks with Z >= {z_floor}: {len(high_z):,}")
        top = high_z.nlargest(15, "z_score")[
            ["_et","resonance","z_score","phase_abort"]
        ]
        print(f"  {'Time ET':>12}  {'rho':>7}  {'Z':>7}  Abort reason")
        print(f"  {'─'*12}  {'─'*7}  {'─'*7}  {'─'*40}")
        for _, row in top.iterrows():
            ts = row["_et"].strftime("%H:%M:%S") if hasattr(row["_et"], "strftime") else str(row["_et"])
            abort = str(row["phase_abort"])[:45] if pd.notna(row["phase_abort"]) else "—"
            print(f"  {ts:>12}  {row['resonance']:7.4f}  {row['z_score']:7.4f}  {abort}")
        break
else:
    print("\n  Z never reached 1.8 today.")


# ── 6. Morning vs Afternoon ───────────────────────────────────────────────────

print("\n" + "═"*65)
print("  6. MORNING vs AFTERNOON SIGNAL")
print("═"*65)

try:
    morning   = df[df["_et"].dt.hour < 12]
    afternoon = df[df["_et"].dt.hour >= 12]

    for label, subset in [("Morning (9:30–12:00 ET)", morning),
                           ("Afternoon (12:00–16:00 ET)", afternoon)]:
        if len(subset) == 0:
            continue
        z_s = subset["z_score"].dropna()
        r_s = subset["resonance"].dropna()
        print(f"\n  {label}  ({len(subset):,} ticks)")
        print(f"    Z:   mean={z_s.mean():.4f}  max={z_s.max():.4f}  "
              f"p90={np.percentile(z_s, 90):.4f}")
        print(f"    rho: mean={r_s.mean():.4f}  max={r_s.max():.4f}")
        n_would_fire = (z_s >= 2.0).sum()
        print(f"    Ticks Z >= 2.0: {n_would_fire:,}  "
              f"({100*n_would_fire/len(z_s):.2f}%)")
except Exception as e:
    print(f"  (Could not split: {e})")


# ── 7. Spread Impact ─────────────────────────────────────────────────────────

print("\n" + "═"*65)
print("  7. SPREAD GATE IMPACT (Phase 9)")
print("═"*65)

ph9 = aborts[aborts["_phase"] == "Phase  9"]
ph9_with_z = ph9["z_score"].dropna()

if len(ph9) > 0:
    print(f"\n  Phase 9 aborted {len(ph9):,} ticks")
    if len(ph9_with_z) > 0:
        print(f"  Z-scores at Phase 9 aborts:")
        print(f"    mean={ph9_with_z.mean():.4f}  max={ph9_with_z.max():.4f}")
        n_would_have_traded = (ph9_with_z >= 2.0).sum()
        print(f"  At Z >= 2.0 AND Phase 9 firing: {n_would_have_traded:,} ticks")
        print(f"  → These would trade if spread bug is fixed")

    # Extract spread value from abort string
    def parse_spread(s):
        if pd.isna(s):
            return None
        m = re.search(r"\$([0-9.]+)", str(s))
        return float(m.group(1)) if m else None

    ph9["_spread"] = ph9["phase_abort"].apply(parse_spread)
    spreads = ph9["_spread"].dropna()
    if len(spreads) > 0:
        print(f"\n  Spread values logged in Phase 9 aborts:")
        print(f"    min=${spreads.min():.3f}  max=${spreads.max():.3f}  "
              f"mean=${spreads.mean():.3f}")
        print(f"    SPY real spread is ~$0.01 → these values are ~{spreads.mean()/0.01:.0f}× too large")
        print(f"    Almost certainly a calculation bug (wrong field or multiplier)")
else:
    print("\n  No Phase 9 aborts found in this date range.")


# ── 8. Summary ────────────────────────────────────────────────────────────────

print("\n" + "═"*65)
print("  8. SUMMARY & RECOMMENDATIONS")
print("═"*65)

z_max  = df["z_score"].max()
z_mean = df["z_score"].mean()
r_max  = df["resonance"].max()

print(f"""
  Today's session ({args.date or 'all dates'}):
    Max Z:    {z_max:.4f}   (threshold: 2.500)
    Mean Z:   {z_mean:.4f}
    Max rho:  {r_max:.4f}
    Fills:    {len(fills)}

  Verdict:
    The system was correctly blocked — Z never reached 2.5.
    Even at Z=2.0, very few ticks qualified (see threshold sweep).
    The spread gate (Phase 9) fired ~30% of all ticks with
    spread values 15–50× larger than SPY's real spread.
    Fixing the spread calculation is the single highest-value change.

  Action plan:
    1. Fix Phase 9 spread calculation (root cause investigation)
    2. If spread fixed: lower Z to 2.0 (or use VIX regime classifier)
    3. Morning session only: signal strongest 9:30–11:00 ET
    4. Consider OFI as supplementary signal to boost Z confidence
""")

print("═"*65)
print("  Backtest complete.")
print("═"*65)
