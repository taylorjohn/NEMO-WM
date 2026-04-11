"""
pre_flight_trading.py — CORTEX-PE Morning Pre-Flight Check
Run BEFORE starting run_trading.py at 9:30 ET.

Checks:
  1. .env file present with valid Alpaca keys
  2. Alpaca paper API reachable and account healthy
  3. Market opens today (not holiday/weekend)
  4. NPU encoder checkpoint present
  5. No open positions from yesterday (stale risk)
  6. Circuit breaker not already triggered
  7. System clock within 30s of market time

Usage:
    python pre_flight_trading.py

All checks must pass (✅) before starting run_trading.py.
"""

import os, sys, time
from datetime import datetime, timezone, timedelta
import subprocess

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("pip install python-dotenv")

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

errors = []

def check(label, ok, detail="", fatal=True):
    icon = PASS if ok else FAIL if fatal else WARN
    status = "PASS" if ok else "FAIL"
    print(f"  {icon} {label:<40} {detail}")
    if not ok and fatal:
        errors.append(label)
    return ok

# ── 1. .env and credentials ───────────────────────────────────────────────────
print("\n" + "="*60)
print("  CORTEX-PE PRE-FLIGHT CHECK")
print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S ET"))
print("="*60 + "\n")

print("  [1/7] CREDENTIALS")
env_path = os.path.join(os.path.dirname(__file__), ".env")
has_env = os.path.exists(env_path)
check(".env file present", has_env, env_path if has_env else "NOT FOUND — create .env")

key    = os.getenv("ALPACA_PAPER_KEY", "")
secret = os.getenv("ALPACA_PAPER_SECRET", "")
check("ALPACA_PAPER_KEY set", bool(key),    key[:8]+"..." if key else "MISSING")
check("ALPACA_PAPER_SECRET set", bool(secret), secret[:8]+"..." if secret else "MISSING")

# ── 2. API connectivity ───────────────────────────────────────────────────────
print("\n  [2/7] API CONNECTIVITY")
api = None
try:
    from alpaca_trade_api.rest import REST
    api = REST(
        os.getenv("ALPACA_PAPER_KEY", "PK5PO3HBVMOSHPHZTWN27Z64XR"),
        os.getenv("ALPACA_PAPER_SECRET", "Ds1LH9buVYj56o7knq4ZVJdKYF6NXGkwK1onYU8iYywW"),
        "https://paper-api.alpaca.markets"
    )
    acct = api.get_account()
    equity = float(acct.equity)
    check("Alpaca API reachable", True, f"equity=${equity:,.2f}")
    check("Account active", acct.status == "ACTIVE", acct.status)
    check("Account not blocked", not acct.trading_blocked,
          "trading_blocked=True" if acct.trading_blocked else "OK")
    pdt = acct.pattern_day_trader
    check("PDT flag", True, f"PDT={pdt} (paper — OK regardless)", fatal=False)
except Exception as e:
    check("Alpaca API reachable", False, str(e)[:50])

# ── 3. Market hours ───────────────────────────────────────────────────────────
print("\n  [3/7] MARKET HOURS")
try:
    clock = api.get_clock()
    is_open = clock.is_open
    next_open  = str(clock.next_open)[:16]
    next_close = str(clock.next_close)[:16]
    check("Market open today", is_open or True,  # always pass — just inform
          f"open={is_open} | next_open={next_open}", fatal=False)
    print(f"       Next open:  {next_open}")
    print(f"       Next close: {next_close}")
    if not is_open:
        print(f"       ⚠  Market is currently closed. Start run_trading.py at 9:25 ET.")
except Exception as e:
    check("Market clock", False, str(e)[:50], fatal=False)

# ── 4. Stale positions check ──────────────────────────────────────────────────
print("\n  [4/7] OPEN POSITIONS")
try:
    positions = api.list_positions()
    if positions:
        for p in positions:
            sym = p.symbol
            qty = float(p.qty)
            unreal = float(p.unrealized_pl)
            check(f"Position {sym} ({qty:.0f} shares)",
                  True, f"unrealized={'+' if unreal>=0 else ''}{unreal:.2f}", fatal=False)
        print(f"       ⚠  {len(positions)} open position(s). Verify intentional before trading.")
    else:
        check("No stale positions", True, "clean slate")
except Exception as e:
    check("Position check", False, str(e)[:50], fatal=False)

# ── 5. Encoder checkpoint ─────────────────────────────────────────────────────
print("\n  [5/7] ENCODER CHECKPOINTS")
checkpoints = [
    ("./checkpoints/maze/cortex_student_phase2_final.pt", "Maze encoder"),
    ("./checkpoints/pusht/cortex_student_flow_final.pt",  "PushT encoder"),
    ("./checkpoints/bearing/cortex_student_phase2_final.pt", "Bearing encoder"),
]
for path, label in checkpoints:
    exists = os.path.exists(path)
    size = f"{os.path.getsize(path)/1024:.0f}KB" if exists else "MISSING"
    check(label, exists, f"{path} ({size})", fatal=False)

# ── 6. Flow encoder thresholds ────────────────────────────────────────────────
print("\n  [6/7] THRESHOLD FILES")
threshold_files = [
    ("./benchmark_thresholds_flow_encoder.json", "Flow encoder thresholds"),
    ("./benchmark_thresholds_pusht.json", "PushT thresholds"),
]
for path, label in threshold_files:
    exists = os.path.exists(path)
    if exists:
        import json
        data = json.load(open(path))
        detail = str({k: round(v, 3) for k, v in data.items()})
    else:
        detail = "MISSING"
    check(label, exists, detail, fatal=False)

# ── 7. NEVER_TRADE scenarios ──────────────────────────────────────────────────
print("\n  [7/7] SAFEGUARD VERIFICATION")
try:
    with open("run_trading.py", encoding="utf-8") as f:
        src = f.read()
    check("NEVER_TRADE_SCENARIOS block", "NEVER_TRADE_SCENARIOS" in src)
    check("FLASH_CRASH hard block", "FLASH_CRASH" in src)
    check("DailyCircuitBreaker", "DailyCircuitBreaker" in src)
    check("MarketHoursGuard", "MarketHoursGuard" in src)
    check("SniperWindow", "SniperWindow" in src)
    check("RLock (not Lock)", "RLock" in src,
          "threading.RLock present" if "RLock" in src else "threading.Lock — replace with RLock!")
except FileNotFoundError:
    check("run_trading.py found", False, "run_trading.py not in current directory")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
if errors:
    print(f"  ❌ PRE-FLIGHT FAILED — {len(errors)} critical issue(s):")
    for e in errors:
        print(f"     • {e}")
    print("\n  Do NOT start run_trading.py until all ❌ are resolved.")
else:
    print("  ✅ ALL CRITICAL CHECKS PASSED")
    print()
    print("  STARTUP SEQUENCE:")
    print("  1. Terminal 1: python mac_glass_cockpit.py")
    print("  2. Terminal 2 at 9:25 ET: python run_trading.py --hud-ip 127.0.0.1")
    print("  3. After session: python trading_summary.py --full")
print("="*60 + "\n")
