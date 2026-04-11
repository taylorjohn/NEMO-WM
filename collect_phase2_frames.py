"""
collect_phase2_frames.py — CORTEX-16 Phase 2 Training Data Collector

Pulls historical 1-minute OHLCV bars from Alpaca and renders each as a
224×224 candlestick chart window. Saves sequentially numbered PNGs for
use as Phase 2 training data in train_distillation.py.

Each saved frame represents a 60-candle rolling window of price action.
Consecutive frames overlap by 59 candles (one new candle per step), so
the triplet loader in SequentialFrameDataset picks up genuine temporal
dynamics — not random jumps.

Why this domain:
  The student encoder runs on chart frames at inference time. Training on
  rendered chart frames means Phase 2 teaches the encoder to produce
  straight latent trajectories in the exact visual domain it will operate in.

Output:
  ./phase2_frames/frame_000001.png
  ./phase2_frames/frame_000002.png
  ... (one frame per 1-minute candle after the initial 60-candle warmup)

Usage:
  # Collect 2 years of SPY 1-minute bars (~196k frames, takes ~10 min)
  python collect_phase2_frames.py --symbol SPY --days 504 --out ./phase2_frames

  # Quick test run (5 trading days, ~1950 frames, fast)
  python collect_phase2_frames.py --symbol SPY --days 5 --out ./phase2_frames

  # Multiple symbols for richer domain coverage
  python collect_phase2_frames.py --symbol QQQ --days 60 --out ./phase2_frames --append

Requirements:
  pip install alpaca-trade-api matplotlib mplfinance python-dotenv
"""

import argparse
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless — no display needed on NUC
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dotenv import load_dotenv


# =============================================================================
# Alpaca Data Fetcher
# Uses paper API keys (read-only market data — same as live keys for historical)
# =============================================================================
def get_alpaca_client():
    load_dotenv()
    key    = os.getenv("ALPACA_PAPER_KEY")    or os.getenv("ALPACA_LIVE_KEY")
    secret = os.getenv("ALPACA_PAPER_SECRET") or os.getenv("ALPACA_LIVE_SECRET")

    if not key or not secret:
        raise EnvironmentError(
            "Alpaca credentials not found. Set ALPACA_PAPER_KEY and "
            "ALPACA_PAPER_SECRET in your .env file."
        )

    from alpaca_trade_api.rest import REST, TimeFrame
    api = REST(key, secret, "https://paper-api.alpaca.markets")
    return api, TimeFrame


def fetch_bars_alpaca(symbol, days_back, api, TimeFrame):
    """Fetch from Alpaca using IEX feed (free tier)."""
    end   = datetime.now()
    start = end - timedelta(days=days_back + 5)
    bars = api.get_bars(
        symbol, TimeFrame.Minute,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        limit=10000, adjustment="raw", feed="iex",
    ).df
    bars.index = pd.to_datetime(bars.index, utc=True).tz_convert("America/New_York")
    bars = bars.between_time("09:30", "16:00")
    return bars[["open", "high", "low", "close", "volume"]].dropna()


def fetch_bars_yfinance(symbol, days_back):
    """Fetch from yfinance — no API key required."""
    import yfinance as yf
    end   = datetime.now()
    start = end - timedelta(days=days_back + 5)
    ticker = yf.Ticker(symbol)
    bars = ticker.history(
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1m",
        auto_adjust=True,
    )
    if bars.empty:
        raise ValueError(f"yfinance returned no data for {symbol}. Try fewer days (yfinance limits 1m data to 30 days).")
    bars.index = bars.index.tz_convert("America/New_York")
    bars = bars.between_time("09:30", "16:00")
    bars.columns = [c.lower() for c in bars.columns]
    return bars[["open", "high", "low", "close", "volume"]].dropna()


def fetch_bars(
    symbol: str,
    days_back: int,
    api=None,
    TimeFrame=None,
) -> pd.DataFrame:
    """
    Fetch 1-minute OHLCV bars. Tries Alpaca IEX first, falls back to yfinance.
    yfinance is free and requires no API key but limits 1m data to last 30 days.
    """
    end   = datetime.now()
    start = end - timedelta(days=days_back + 5)
    print(f"📡 Fetching {symbol} 1-min bars: "
          f"{start.date()} → {end.date()} (~{days_back} trading days)...")

    # Try Alpaca first
    if api is not None:
        try:
            bars = fetch_bars_alpaca(symbol, days_back, api, TimeFrame)
            print(f"   ✅ {len(bars):,} bars via Alpaca IEX ({len(bars)/390:.1f} days)")
            return bars
        except Exception as e:
            print(f"   ⚠️  Alpaca failed: {e}")
            print(f"   Falling back to yfinance...")

    # yfinance fallback
    bars = fetch_bars_yfinance(symbol, min(days_back, 29))
    print(f"   ✅ {len(bars):,} bars via yfinance ({len(bars)/390:.1f} days)")
    return bars


# =============================================================================
# Chart Renderer
# Renders a 60-candle window as a 224×224 PNG.
# Dark background matches the live HUD display style from CORTEX-16.
# Volume bars rendered at 20% height to preserve price structure visibility.
# =============================================================================
CANDLE_WINDOW = 60   # Number of candles visible per frame
FRAME_SIZE    = 224  # Pixels (must match StudentEncoder input)

# Colour scheme matching CORTEX-16 HUD (dark background, cyan/red candles)
BG_COLOR    = "#0d0d0d"
UP_COLOR    = "#00e5b0"   # Teal/cyan — bullish
DOWN_COLOR  = "#ff3a5c"   # Red — bearish
WICK_COLOR  = "#555555"
VOL_COLOR   = "#1a3a4a"


def render_candlestick(window: pd.DataFrame) -> np.ndarray:
    """
    Renders a 60-candle OHLCV window to a (224, 224, 3) uint8 array.

    Args:
        window: DataFrame with columns [open, high, low, close, volume]
                Length must be exactly CANDLE_WINDOW (60)
    Returns:
        (224, 224, 3) numpy array — RGB, uint8
    """
    assert len(window) == CANDLE_WINDOW, f"Expected {CANDLE_WINDOW} rows, got {len(window)}"

    fig, (ax_price, ax_vol) = plt.subplots(
        2, 1,
        figsize       = (FRAME_SIZE / 100, FRAME_SIZE / 100),
        dpi           = 100,
        gridspec_kw   = {"height_ratios": [4, 1], "hspace": 0},
        facecolor     = BG_COLOR,
    )
    ax_price.set_facecolor(BG_COLOR)
    ax_vol.set_facecolor(BG_COLOR)

    opens  = window["open"].values
    highs  = window["high"].values
    lows   = window["low"].values
    closes = window["close"].values
    vols   = window["volume"].values
    n      = len(window)
    xs     = np.arange(n)

    # Candle bodies
    for i in xs:
        color  = UP_COLOR if closes[i] >= opens[i] else DOWN_COLOR
        bottom = min(opens[i], closes[i])
        height = abs(closes[i] - opens[i]) or (highs[i] - lows[i]) * 0.01
        ax_price.bar(i, height, bottom=bottom, color=color,
                     width=0.8, linewidth=0)
        # Wicks
        ax_price.plot([i, i], [lows[i], highs[i]],
                      color=WICK_COLOR, linewidth=0.6, zorder=0)

    # Price axis styling
    price_range = highs.max() - lows.min()
    pad         = price_range * 0.05
    ax_price.set_ylim(lows.min() - pad, highs.max() + pad)
    ax_price.set_xlim(-0.5, n - 0.5)
    ax_price.axis("off")

    # Volume bars
    ax_vol.bar(xs, vols, color=VOL_COLOR, width=0.8, linewidth=0)
    ax_vol.set_xlim(-0.5, n - 0.5)
    ax_vol.axis("off")

    # Render to numpy array
    fig.canvas.draw()
    buf   = fig.canvas.buffer_rgba()
    image = np.frombuffer(buf, dtype=np.uint8).reshape(FRAME_SIZE, FRAME_SIZE, 4)
    plt.close(fig)

    return image[:, :, :3]   # Drop alpha channel → (224, 224, 3) RGB


# =============================================================================
# Frame Collection Pipeline
# =============================================================================
def collect_frames(
    bars:        pd.DataFrame,
    output_dir:  str,
    start_index: int = 1,
    log_every:   int = 500,
) -> int:
    """
    Renders all valid 60-candle rolling windows from bars DataFrame.
    Saves each as a zero-padded PNG for sequential loading.

    Args:
        bars:        Full OHLCV DataFrame
        output_dir:  Where to save frames
        start_index: Starting frame number (for appending to existing dataset)
        log_every:   Print progress every N frames

    Returns:
        Number of frames saved
    """
    out    = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_bars = len(bars)

    if n_bars < CANDLE_WINDOW:
        raise ValueError(
            f"Need at least {CANDLE_WINDOW} bars, got {n_bars}. "
            f"Try --days 2 or more."
        )

    total_frames = n_bars - CANDLE_WINDOW + 1
    print(f"\n🎬 Rendering {total_frames:,} frames → {output_dir}")
    print(f"   Each frame: {CANDLE_WINDOW}-candle window, {FRAME_SIZE}×{FRAME_SIZE}px")
    print(f"   Estimated size: ~{total_frames * 15 // 1024} MB\n")

    saved  = 0
    t0     = time.perf_counter()
    errors = 0

    for i in range(total_frames):
        window = bars.iloc[i : i + CANDLE_WINDOW]
        frame_num = start_index + i
        out_path  = out / f"frame_{frame_num:07d}.png"

        try:
            rgb = render_candlestick(window)
            # Save as PNG via PIL (avoids extra matplotlib overhead)
            from PIL import Image
            Image.fromarray(rgb).save(out_path, optimize=False)
            saved += 1
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"   ⚠️  Frame {frame_num} failed: {e}")
            continue

        if saved % log_every == 0:
            elapsed    = time.perf_counter() - t0
            rate       = saved / elapsed
            remaining  = (total_frames - saved) / max(rate, 1e-6)
            print(
                f"   Frame {saved:>7,}/{total_frames:,}  "
                f"({rate:.0f} fps)  "
                f"ETA: {remaining/60:.1f} min"
            )

    elapsed = time.perf_counter() - t0
    print(f"\n✅ Saved {saved:,} frames in {elapsed/60:.1f} min "
          f"({saved/elapsed:.0f} fps)")
    if errors:
        print(f"   ⚠️  {errors} frames skipped due to render errors")
    return saved


# =============================================================================
# Dataset Health Check
# Verifies the output directory is ready for SequentialFrameDataset
# =============================================================================
def verify_dataset(output_dir: str) -> None:
    frames = sorted(Path(output_dir).glob("frame_*.png"))
    if not frames:
        print(f"❌ No frames found in {output_dir}")
        return

    print(f"\n── Dataset Verification ─────────────────────────────────────────")
    print(f"   Directory:    {output_dir}")
    print(f"   Total frames: {len(frames):,}")
    print(f"   Valid triplets (Phase 2): {max(0, len(frames) - 2):,}")

    # Check a few files are valid images
    from PIL import Image
    sample_errors = 0
    for f in frames[:5]:
        try:
            img = Image.open(f)
            assert img.size == (FRAME_SIZE, FRAME_SIZE), f"Wrong size: {img.size}"
        except Exception as e:
            print(f"   ⚠️  {f.name}: {e}")
            sample_errors += 1

    if sample_errors == 0:
        print(f"   Image size:   {FRAME_SIZE}×{FRAME_SIZE}px  ✅")

    # Minimum required for Phase 2 training
    min_triplets = 500
    valid_triplets = len(frames) - 2
    if valid_triplets >= min_triplets:
        print(f"   Phase 2 ready: ✅ ({valid_triplets:,} triplets ≥ {min_triplets} minimum)")
    else:
        print(f"   Phase 2 ready: ❌ ({valid_triplets} triplets < {min_triplets} minimum)")
        print(f"   Add more data: python collect_phase2_frames.py --days 3 --append")

    print(f"\n   To train Phase 2:")
    print(f"   python train_distillation.py --phase 2 \\")
    print(f"       --data {output_dir} \\")
    print(f"       --resume cortex_student_phase1_final.pt")


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Collect Phase 2 training frames from Alpaca historical bars"
    )
    p.add_argument("--symbol",  default="SPY",
                   help="Ticker symbol (default: SPY)")
    p.add_argument("--days",    type=int, default=60,
                   help="Calendar days of history to fetch (default: 60 ≈ 3mo)")
    p.add_argument("--out",     default="./phase2_frames",
                   help="Output directory for frames")
    p.add_argument("--append",  action="store_true",
                   help="Append to existing frames instead of starting from 1")
    p.add_argument("--verify-only", action="store_true",
                   help="Only run dataset verification, no collection")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.verify_only:
        verify_dataset(args.out)
    else:
        # Determine start index for appending
        start_index = 1
        if args.append:
            existing = sorted(Path(args.out).glob("frame_*.png"))
            if existing:
                last_num    = int(existing[-1].stem.split("_")[1])
                start_index = last_num + 1
                print(f"📂 Appending from frame {start_index:,} "
                      f"({len(existing):,} existing frames)")

        # Fetch bars
        api, TimeFrame = get_alpaca_client()
        bars = fetch_bars(args.symbol, args.days, api, TimeFrame)

        # Render and save
        collect_frames(bars, args.out, start_index=start_index)

        # Verify output
        verify_dataset(args.out)
