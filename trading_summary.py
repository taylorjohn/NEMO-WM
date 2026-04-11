"""
trading_summary.py — CORTEX-PE Paper Trading PnL Review
Queries Alpaca paper API and prints full session summary.

Usage:
    python trading_summary.py              # Current session summary
    python trading_summary.py --full       # All closed orders ever
    python trading_summary.py --reset      # Show account reset baseline

Requires: .env with ALPACA_PAPER_KEY and ALPACA_PAPER_SECRET
"""

import os, sys, json
from datetime import datetime, timezone
from collections import defaultdict

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from alpaca_trade_api.rest import REST
except ImportError:
    print("pip install alpaca-trade-api python-dotenv")
    sys.exit(1)

# ── Credentials ───────────────────────────────────────────────────────────────
KEY    = os.getenv("ALPACA_PAPER_KEY",    "PK5PO3HBVMOSHPHZTWN27Z64XR")
SECRET = os.getenv("ALPACA_PAPER_SECRET", "Ds1LH9buVYj56o7knq4ZVJdKYF6NXGkwK1onYU8iYywW")
URL    = "https://paper-api.alpaca.markets"

api = REST(KEY, SECRET, URL)

def fmt_dollar(v):
    sign = "+" if v >= 0 else ""
    return f"{sign}${v:,.2f}"

def fmt_pct(v):
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.2f}%"

# ── 1. Account snapshot ────────────────────────────────────────────────────────
def print_account():
    acct = api.get_account()
    equity       = float(acct.equity)
    cash         = float(acct.cash)
    buying_power = float(acct.buying_power)
    last_equity  = float(acct.last_equity)
    day_pnl      = equity - last_equity
    day_pnl_pct  = (day_pnl / last_equity * 100) if last_equity else 0

    print("\n" + "═"*56)
    print("  CORTEX-PE PAPER TRADING — ACCOUNT SNAPSHOT")
    print("═"*56)
    print(f"  Equity:        ${equity:>12,.2f}")
    print(f"  Cash:          ${cash:>12,.2f}")
    print(f"  Buying Power:  ${buying_power:>12,.2f}")
    print(f"  Day P&L:       {fmt_dollar(day_pnl):>13}  ({fmt_pct(day_pnl_pct)})")
    print(f"  Status:        {acct.status}")
    print(f"  PDT flag:      {acct.pattern_day_trader}")
    print()
    return equity, cash

# ── 2. Open positions ──────────────────────────────────────────────────────────
def print_positions():
    try:
        positions = api.list_positions()
    except Exception as e:
        print(f"  ⚠ Could not fetch positions: {e}")
        return

    if not positions:
        print("  No open positions.")
        return

    print("  OPEN POSITIONS")
    print(f"  {'Symbol':<8} {'Qty':>6} {'Avg Entry':>12} {'Current':>10} {'Unrealized':>12} {'%':>7}")
    print("  " + "-"*58)
    total_unrealized = 0.0
    for p in positions:
        sym    = p.symbol
        qty    = float(p.qty)
        avg    = float(p.avg_entry_price)
        curr   = float(p.current_price)
        unreal = float(p.unrealized_pl)
        pct    = float(p.unrealized_plpc) * 100
        total_unrealized += unreal
        print(f"  {sym:<8} {qty:>6.0f} ${avg:>11,.2f} ${curr:>9,.2f} {fmt_dollar(unreal):>12} {fmt_pct(pct):>7}")
    print(f"  {'TOTAL':<8} {'':>6} {'':>12} {'':>10} {fmt_dollar(total_unrealized):>12}")
    print()

# ── 3. Closed orders analysis ──────────────────────────────────────────────────
def print_orders(full=False):
    limit = 500 if full else 100
    try:
        orders = api.list_orders(status='closed', limit=limit, direction='desc')
    except Exception as e:
        print(f"  ⚠ Could not fetch orders: {e}")
        return

    filled = [o for o in orders if o.status == 'filled' and o.filled_avg_price]
    if not filled:
        print("  No filled orders found.")
        return

    # Pair buys and sells per symbol
    by_symbol = defaultdict(list)
    for o in filled:
        by_symbol[o.symbol].append(o)

    total_realized = 0.0
    trade_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0.0, 'trades': 0})

    # Simple FIFO matching
    all_trades = []
    for sym, sym_orders in by_symbol.items():
        buys  = [(float(o.filled_avg_price), float(o.filled_qty), o.submitted_at)
                 for o in sym_orders if o.side == 'buy']
        sells = [(float(o.filled_avg_price), float(o.filled_qty), o.submitted_at)
                 for o in sym_orders if o.side == 'sell']

        for sell_px, sell_qty, sell_time in sells:
            remaining = sell_qty
            for i, (buy_px, buy_qty, buy_time) in enumerate(buys):
                if remaining <= 0:
                    break
                matched = min(remaining, buy_qty)
                pnl = (sell_px - buy_px) * matched
                total_realized += pnl
                trade_stats[sym]['pnl'] += pnl
                trade_stats[sym]['trades'] += 1
                if pnl >= 0:
                    trade_stats[sym]['wins'] += 1
                else:
                    trade_stats[sym]['losses'] += 1
                all_trades.append((sym, buy_px, sell_px, matched, pnl, sell_time))
                buys[i] = (buy_px, buy_qty - matched, buy_time)
                remaining -= matched

    # Print by symbol
    print(f"  TRADE SUMMARY ({'all time' if full else 'last 100 orders'})")
    print(f"  {'Symbol':<8} {'Trades':>7} {'Wins':>5} {'Losses':>7} {'Win%':>6} {'P&L':>12}")
    print("  " + "-"*50)
    for sym, stats in sorted(trade_stats.items()):
        t = stats['trades']
        w = stats['wins']
        l = stats['losses']
        pnl = stats['pnl']
        win_pct = (w / t * 100) if t else 0
        print(f"  {sym:<8} {t:>7} {w:>5} {l:>7} {win_pct:>5.0f}% {fmt_dollar(pnl):>12}")
    print("  " + "-"*50)
    print(f"  {'TOTAL':<8} {sum(s['trades'] for s in trade_stats.values()):>7} "
          f"{'':>5} {'':>7} {'':>6} {fmt_dollar(total_realized):>12}")

    # Recent trades
    print()
    print(f"  RECENT FILLS (last 10)")
    print(f"  {'Symbol':<8} {'Buy':>10} {'Sell':>10} {'Qty':>6} {'P&L':>10}  Time")
    print("  " + "-"*62)
    for sym, buy_px, sell_px, qty, pnl, ts in sorted(all_trades, key=lambda x: x[5], reverse=True)[:10]:
        ts_str = str(ts)[:16] if ts else "—"
        marker = "✓" if pnl >= 0 else "✗"
        print(f"  {sym:<8} ${buy_px:>9,.2f} ${sell_px:>9,.2f} {qty:>6.0f} {fmt_dollar(pnl):>10}  {ts_str} {marker}")
    print()

# ── 4. Safeguard audit ─────────────────────────────────────────────────────────
def print_safeguard_audit():
    """Check log file for safeguard trigger counts."""
    log_path = "cortex_trading.log"
    if not os.path.exists(log_path):
        print("  ⚠ No trading log found (cortex_trading.log). Run trading first.")
        return

    with open(log_path, encoding='utf-8', errors='ignore') as f:
        content = f.read()

    triggers = {
        "CIRCUIT_BREAKER": content.count("CIRCUIT_BREAKER"),
        "NEVER_TRADE":     content.count("NEVER_TRADE"),
        "FLASH_CRASH":     content.count("FLASH_CRASH"),
        "MARKET_CLOSED":   content.count("MARKET_CLOSED"),
        "SNIPER_FIRE":     content.count("SNIPER"),
        "FORCE_CLOSE":     content.count("FORCE_CLOSE"),
    }

    print("  SAFEGUARD AUDIT (from cortex_trading.log)")
    for k, v in triggers.items():
        status = "✓" if (k == "FLASH_CRASH" and v == 0) else ("⚠" if v > 0 else "—")
        print(f"  {status} {k:<22} {v:>4} triggers")
    print()

# ── 5. Phase 3 directional signal gap ─────────────────────────────────────────
def print_gap_warning():
    print("  ⚠ KNOWN GAP: No directional signal")
    print("    Current encoder produces anomaly scores only (−‖z‖).")
    print("    Phase 3 return supervision not yet implemented.")
    print("    Action: add L_return = MSE(Linear(z_t), next_bar_return)")
    print("    File:   collect_phase3_frames.py → retrain SPY encoder")
    print()

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    full_mode = "--full" in sys.argv

    try:
        equity, cash = print_account()
        print_positions()
        print_orders(full=full_mode)
        print_safeguard_audit()
        print_gap_warning()
        print("═"*56)
        print("  Run with --full for complete order history")
        print("═"*56 + "\n")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("   Check .env file has ALPACA_PAPER_KEY and ALPACA_PAPER_SECRET")
