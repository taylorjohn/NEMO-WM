"""
compute_goal_states.py — CORTEX-16 Goal State Computation

Computes two goal latent vectors from historical SPY chart frames:
    z_goal_bull — centroid of latents preceding positive next-bar returns
    z_goal_bear — centroid of latents preceding negative next-bar returns

These replace the fixed z_goal = zeros(128) in unified_cortex_loop.py,
giving the mirror ascent planner a market-grounded objective.

How it works:
    1. Fetch historical 1-min SPY bars from Alpaca
    2. For each bar t, render the 60-candle window ending at t as a 224×224 frame
    3. Encode the frame to z_t using the Phase 2 ONNX encoder
    4. Label z_t as bullish if close[t+1] > close[t], bearish otherwise
    5. z_goal_bull = mean(z_t | bullish) — weighted by return magnitude
    6. z_goal_bear = mean(z_t | bearish) — weighted by return magnitude

The planner then navigates toward z_goal_bull when momentum is positive
(bullish regime) and z_goal_bear when momentum is negative (bearish regime).

At inference time, unified_cortex_loop.py:
    - Computes 5-bar momentum from recent Alpaca bars
    - Sets z_goal = z_goal_bull if momentum > 0, else z_goal_bear
    - Mirror ascent plans toward that goal

Output:
    ./goals/z_goal_bull.npy   — (128,) float32 bullish goal latent
    ./goals/z_goal_bear.npy   — (128,) float32 bearish goal latent
    ./goals/goal_stats.json   — diagnostics (n_bull, n_bear, separation)

Usage:
    python compute_goal_states.py --days 25 --symbol SPY
    python compute_goal_states.py --days 60 --symbol SPY --top-pct 30
"""

import argparse
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
from dotenv import load_dotenv
from torchvision import transforms

load_dotenv()


# =============================================================================
# Config
# =============================================================================
CANDLE_WINDOW = 60
FRAME_SIZE    = 224
BG_COLOR      = "#0d0d0d"
UP_COLOR      = "#00e5b0"
DOWN_COLOR    = "#ff3a5c"
WICK_COLOR    = "#555555"
VOL_COLOR     = "#1a3a4a"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

FRAME_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# =============================================================================
# Alpaca bar fetcher
# =============================================================================
def fetch_bars(symbol: str, days: int) -> pd.DataFrame:
    try:
        key    = os.getenv("ALPACA_PAPER_KEY")
        secret = os.getenv("ALPACA_PAPER_SECRET")
        from alpaca_trade_api.rest import REST, TimeFrame
        api   = REST(key, secret, "https://paper-api.alpaca.markets")
        end   = datetime.now()
        start = end - timedelta(days=days + 5)
        bars  = api.get_bars(
            symbol, TimeFrame.Minute,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            limit=50000, adjustment="raw", feed="iex",
        ).df
        bars.columns = [c.lower() for c in bars.columns]
        bars         = bars[["open", "high", "low", "close", "volume"]].copy()
        bars         = bars[bars["volume"] > 0].reset_index(drop=True)
        print(f"✅ Alpaca: {len(bars):,} bars ({symbol}, {days} days)")
        return bars
    except Exception as e:
        print(f"⚠️  Alpaca failed: {e} — trying yfinance")
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        bars   = ticker.history(period=f"{min(days, 29)}d", interval="1m")
        bars.columns = [c.lower() for c in bars.columns]
        bars   = bars[["open", "high", "low", "close", "volume"]].reset_index(drop=True)
        print(f"✅ yfinance: {len(bars):,} bars ({symbol})")
        return bars


# =============================================================================
# Chart renderer — matches Phase 2 training renderer exactly
# =============================================================================
def render_window(window: pd.DataFrame) -> np.ndarray:
    fig, (ax_p, ax_v) = plt.subplots(
        2, 1, figsize=(2.24, 2.24), dpi=100,
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0},
        facecolor=BG_COLOR,
    )
    ax_p.set_facecolor(BG_COLOR)
    ax_v.set_facecolor(BG_COLOR)

    opens  = window["open"].values
    highs  = window["high"].values
    lows   = window["low"].values
    closes = window["close"].values
    vols   = window["volume"].values
    xs     = np.arange(len(window))

    for i in xs:
        color  = UP_COLOR if closes[i] >= opens[i] else DOWN_COLOR
        bottom = min(opens[i], closes[i])
        height = abs(closes[i] - opens[i]) or (highs[i] - lows[i]) * 0.01
        ax_p.bar(i, height, bottom=bottom, color=color, width=0.8, linewidth=0)
        ax_p.plot([i, i], [lows[i], highs[i]], color=WICK_COLOR, linewidth=0.6, zorder=0)

    pad = (highs.max() - lows.min()) * 0.05
    ax_p.set_ylim(lows.min() - pad, highs.max() + pad)
    ax_p.set_xlim(-0.5, len(window) - 0.5)
    ax_p.axis("off")
    ax_v.bar(xs, vols, color=VOL_COLOR, width=0.8, linewidth=0)
    ax_v.set_xlim(-0.5, len(window) - 0.5)
    ax_v.axis("off")

    fig.canvas.draw()
    buf   = fig.canvas.buffer_rgba()
    image = np.frombuffer(buf, dtype=np.uint8).reshape(224, 224, 4)
    plt.close(fig)
    return image[:, :, :3]


# =============================================================================
# ONNX encoder
# =============================================================================
def build_encoder(onnx_path: str) -> ort.InferenceSession:
    """Load encoder — tries VitisAI NPU first, falls back to DirectML, then CPU."""
    # VitisAI NPU
    try:
        opts = {"cache_dir": str(Path(onnx_path).parent),
                "cache_key": "cortex_goals", "log_level": "warning"}
        sess = ort.InferenceSession(
            onnx_path,
            providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
            provider_options=[opts, {}],
        )
        if "VitisAI" in sess.get_providers()[0]:
            print(f"⚡ Encoder: VitisAI NPU")
            return sess
    except Exception:
        pass

    # DirectML
    try:
        sess = ort.InferenceSession(
            onnx_path,
            providers=["DmlExecutionProvider", "CPUExecutionProvider"],
        )
        if "Dml" in sess.get_providers()[0]:
            print(f"⚡ Encoder: DirectML (AMD Radeon iGPU)")
            return sess
    except Exception:
        pass

    # CPU fallback
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    print(f"⚡ Encoder: CPU fallback")
    return sess


def encode_frame(sess: ort.InferenceSession, frame_np: np.ndarray) -> np.ndarray:
    """Encode (H, W, 3) uint8 RGB frame → (128,) float32 latent."""
    import torch
    t   = FRAME_TRANSFORM(frame_np).unsqueeze(0).numpy()
    out = sess.run(None, {"input_frame": t})
    return out[0].squeeze(0)


# =============================================================================
# Main computation
# =============================================================================
def compute_goal_states(
    onnx_path:  str,
    symbol:     str   = "SPY",
    days:       int   = 25,
    top_pct:    float = 100.0,
    out_dir:    str   = "./goals",
    min_bars:   int   = 200,
) -> dict:
    print("\n" + "="*60)
    print("  CORTEX-16 GOAL STATE COMPUTATION")
    print(f"  Encoder:  {onnx_path}")
    print(f"  Symbol:   {symbol}")
    print(f"  Days:     {days}")
    print(f"  Top pct:  {top_pct}%  (use strongest returns only)")
    print("="*60 + "\n")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Load encoder
    sess = build_encoder(onnx_path)

    # Fetch bars
    bars = fetch_bars(symbol, days)
    if len(bars) < CANDLE_WINDOW + min_bars:
        print(f"❌ Not enough bars: {len(bars)} < {CANDLE_WINDOW + min_bars}")
        return {}

    # Compute next-bar returns
    bars["next_return"] = bars["close"].shift(-1) / bars["close"] - 1.0
    bars = bars.dropna(subset=["next_return"]).reset_index(drop=True)

    print(f"Processing {len(bars) - CANDLE_WINDOW:,} windows...")

    bull_latents  = []
    bear_latents  = []
    bull_weights  = []
    bear_weights  = []
    n_processed   = 0
    t0            = time.perf_counter()

    for i in range(CANDLE_WINDOW, len(bars)):
        window     = bars.iloc[i - CANDLE_WINDOW:i]
        ret        = float(bars["next_return"].iloc[i - 1])
        abs_ret    = abs(ret)

        if abs_ret < 1e-6:   # skip flat bars
            continue

        # Render and encode
        try:
            frame = render_window(window)
            z     = encode_frame(sess, frame)
        except Exception:
            continue

        if ret > 0:
            bull_latents.append(z)
            bull_weights.append(abs_ret)
        else:
            bear_latents.append(z)
            bear_weights.append(abs_ret)

        n_processed += 1
        if n_processed % 100 == 0:
            elapsed = time.perf_counter() - t0
            rate    = n_processed / elapsed
            remain  = (len(bars) - CANDLE_WINDOW - n_processed) / max(rate, 1)
            print(f"   {n_processed:>5} frames | "
                  f"bull={len(bull_latents)} bear={len(bear_latents)} | "
                  f"ETA {remain/60:.1f}min")

    if not bull_latents or not bear_latents:
        print("❌ Not enough labelled frames")
        return {}

    # Convert to arrays
    bull_latents = np.stack(bull_latents)   # (N_bull, 128)
    bear_latents = np.stack(bear_latents)   # (N_bear, 128)
    bull_weights = np.array(bull_weights)
    bear_weights = np.array(bear_weights)

    # Apply top-pct filter — use only the strongest returns
    if top_pct < 100.0:
        bull_thresh = np.percentile(bull_weights, 100 - top_pct)
        bear_thresh = np.percentile(bear_weights, 100 - top_pct)
        bull_mask   = bull_weights >= bull_thresh
        bear_mask   = bear_weights >= bear_thresh
        bull_latents = bull_latents[bull_mask]
        bear_latents = bear_latents[bear_mask]
        bull_weights = bull_weights[bull_mask]
        bear_weights = bear_weights[bear_mask]
        print(f"\n   Top {top_pct}% filter: "
              f"bull {len(bull_latents)} frames, bear {len(bear_latents)} frames")

    # Weighted centroid
    bull_w      = bull_weights / bull_weights.sum()
    bear_w      = bear_weights / bear_weights.sum()
    z_goal_bull = (bull_latents * bull_w[:, None]).sum(axis=0).astype(np.float32)
    z_goal_bear = (bear_latents * bear_w[:, None]).sum(axis=0).astype(np.float32)

    # Diagnostics
    separation  = float(np.linalg.norm(z_goal_bull - z_goal_bear))
    bull_norm   = float(np.linalg.norm(z_goal_bull))
    bear_norm   = float(np.linalg.norm(z_goal_bear))
    cosine_sim  = float(
        np.dot(z_goal_bull, z_goal_bear) / (bull_norm * bear_norm + 1e-8)
    )

    print(f"\n{'='*60}")
    print(f"  GOAL STATE DIAGNOSTICS")
    print(f"{'='*60}")
    print(f"  Bull frames:  {len(bull_latents):,}")
    print(f"  Bear frames:  {len(bear_latents):,}")
    print(f"  Bull ratio:   {len(bull_latents)/(len(bull_latents)+len(bear_latents))*100:.1f}%")
    print(f"  ||z_bull||:   {bull_norm:.4f}")
    print(f"  ||z_bear||:   {bear_norm:.4f}")
    print(f"  Separation:   {separation:.4f}  (higher = more distinct goals)")
    print(f"  Cosine sim:   {cosine_sim:.4f}  (lower = more distinct, target < 0.95)")

    if cosine_sim > 0.99:
        print(f"  ⚠️  Goals nearly identical (cosine={cosine_sim:.4f})")
        print(f"     The encoder may not distinguish bull/bear chart patterns.")
        print(f"     Consider Phase 3 training with return supervision.")
    elif cosine_sim < 0.90:
        print(f"  ✅ Goals well separated — strong bull/bear distinction")
    else:
        print(f"  ✅ Goals moderately separated — usable signal")

    # Save
    bull_path = Path(out_dir) / "z_goal_bull.npy"
    bear_path = Path(out_dir) / "z_goal_bear.npy"
    np.save(bull_path, z_goal_bull)
    np.save(bear_path, z_goal_bear)

    stats = {
        "symbol":       symbol,
        "days":         days,
        "top_pct":      top_pct,
        "n_bull":       int(len(bull_latents)),
        "n_bear":       int(len(bear_latents)),
        "bull_ratio":   float(len(bull_latents) / (len(bull_latents) + len(bear_latents))),
        "bull_norm":    bull_norm,
        "bear_norm":    bear_norm,
        "separation":   separation,
        "cosine_sim":   cosine_sim,
        "z_goal_bull":  bull_path.as_posix(),
        "z_goal_bear":  bear_path.as_posix(),
        "encoder":      onnx_path,
        "timestamp":    datetime.now().isoformat(),
    }
    with open(Path(out_dir) / "goal_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n  💾 Saved:")
    print(f"     {bull_path}")
    print(f"     {bear_path}")
    print(f"     {Path(out_dir) / 'goal_stats.json'}")
    print(f"\n  Next: restart unified_cortex_loop.py")
    print(f"  It will auto-load goal states from {out_dir}/")

    return stats


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CORTEX-16 Goal State Computation")
    p.add_argument("--onnx",    default="./npu_models/cortex_student_xint8.onnx",
                   help="ONNX encoder path")
    p.add_argument("--symbol",  default="SPY")
    p.add_argument("--days",    type=int, default=25,
                   help="Days of historical data to use")
    p.add_argument("--top-pct", type=float, default=100.0,
                   help="Use only top N%% strongest returns (e.g. 30 = top 30%%)")
    p.add_argument("--out",     default="./goals",
                   help="Output directory for goal state files")
    args = p.parse_args()

    compute_goal_states(
        onnx_path = args.onnx,
        symbol    = args.symbol,
        days      = args.days,
        top_pct   = args.top_pct,
        out_dir   = args.out,
    )
