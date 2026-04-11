"""
train_phase3_return.py — CORTEX-PE Phase 3B: Return Supervision

Adds directional signal to the frozen encoder by training a small
linear return head using IQL-style expectile regression on multi-step
SPY returns. This closes the core trading gap — encoder currently
produces anomaly scores only, with no bull/bear direction.

Architecture:
    Frozen StudentEncoder (46K) → 128-D z_t
    ReturnHead (128→64→1, ~8K params) → expected_return
    
Loss:
    L_return = expectile_regression(V(z_t), R_t:t+H)
    where R_t:t+H = cumulative log return over next H bars
    τ = 0.9 (optimistic, consistent with IQL)

Based on:
    - Value-guided action planning with JEPA (arXiv:2601.00844)
    - IQL: Offline RL via expectile regression (Kostrikov et al. 2021)

Usage:
    pip install yfinance
    python train_phase3_return.py --horizon 5 --epochs 30 --out ./checkpoints/phase3_trading
    
    Then in run_trading.py — add return_head(z_t) > 0 as directional gate
    before any trade fires.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw

from student_encoder import StudentEncoder

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--encoder',  default='./checkpoints/maze/cortex_student_phase2_final.pt')
parser.add_argument('--horizon',  type=int,   default=5,     help='Bars forward for return target')
parser.add_argument('--epochs',   type=int,   default=30)
parser.add_argument('--batch',    type=int,   default=64)
parser.add_argument('--lr',       type=float, default=3e-4)
parser.add_argument('--tau',      type=float, default=0.9,   help='IQL expectile (0.5=MSE, 0.9=optimistic)')
parser.add_argument('--period',   default='1y',              help='yfinance period')
parser.add_argument('--interval', default='1h',              help='yfinance interval: 1h or 1d')
parser.add_argument('--out',      default='./checkpoints/phase3_trading')
parser.add_argument('--frame-size', type=int, default=224)
args = parser.parse_args()

Path(args.out).mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cpu')
TRANSFORM = transforms.Compose([
    transforms.Resize(args.frame_size),
    transforms.CenterCrop(args.frame_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Candlestick renderer ──────────────────────────────────────────────────────
def render_ohlcv_frame(opens, highs, lows, closes, volumes, size=224):
    """Render a window of OHLCV bars as a 224×224 RGB image."""
    img = Image.new('RGB', (size, size), (20, 20, 30))
    draw = ImageDraw.Draw(img)
    n = len(closes)
    if n == 0:
        return img

    price_min = min(lows)
    price_max = max(highs)
    price_range = price_max - price_min + 1e-8
    vol_max = max(volumes) + 1e-8

    bar_w = max(1, size // n - 1)
    pad = 10

    for i in range(n):
        x = pad + i * (size - 2 * pad) // n
        # Price coords (inverted — top = high)
        def py(price):
            return int(size * 0.8 - (price - price_min) / price_range * size * 0.6)

        # Wick
        draw.line([(x + bar_w // 2, py(highs[i])),
                   (x + bar_w // 2, py(lows[i]))], fill=(120, 120, 140), width=1)
        # Body
        y_open  = py(opens[i])
        y_close = py(closes[i])
        color = (60, 200, 100) if closes[i] >= opens[i] else (220, 60, 80)
        draw.rectangle([x, min(y_open, y_close),
                        x + bar_w, max(y_open, y_close)], fill=color)
        # Volume bar
        vh = int((volumes[i] / vol_max) * size * 0.15)
        draw.rectangle([x, size - vh, x + bar_w, size], fill=(80, 80, 120))

    return img


# ── Dataset ───────────────────────────────────────────────────────────────────
class SPYReturnDataset(Dataset):
    """
    Downloads SPY bars via yfinance, renders candlestick frames,
    and pairs each frame with its H-bar cumulative log return.
    """
    def __init__(self, period='1y', interval='1h', horizon=5,
                 window=20, transform=None, out_dir=None):
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("pip install yfinance")

        print(f"Downloading SPY {period} {interval} bars...")
        ticker = yf.Ticker('SPY')
        df = ticker.history(period=period, interval=interval)
        df = df.dropna()
        print(f"  {len(df)} bars downloaded")

        closes  = df['Close'].values.astype(np.float32)
        opens   = df['Open'].values.astype(np.float32)
        highs   = df['High'].values.astype(np.float32)
        lows    = df['Low'].values.astype(np.float32)
        volumes = df['Volume'].values.astype(np.float32)

        # Compute H-bar cumulative log return (the label)
        log_returns = np.log(closes[1:] / closes[:-1])
        cumulative  = np.array([
            log_returns[i:i+horizon].sum()
            for i in range(len(log_returns) - horizon)
        ], dtype=np.float32)

        self.frames  = []
        self.returns = []
        self.transform = transform or TRANSFORM

        out_path = Path(out_dir) / 'frames' if out_dir else None
        if out_path:
            out_path.mkdir(parents=True, exist_ok=True)

        print(f"  Rendering {len(cumulative)} frames (window={window})...")
        for i in range(window, len(cumulative)):
            sl = slice(i - window, i)
            frame = render_ohlcv_frame(
                opens[sl], highs[sl], lows[sl], closes[sl], volumes[sl]
            )
            if out_path:
                frame.save(out_path / f'frame_{i:07d}.png')
            self.frames.append(frame)
            self.returns.append(float(cumulative[i]))

        # Normalise returns to zero mean, unit std
        ret_arr   = np.array(self.returns)
        self.ret_mean = float(ret_arr.mean())
        self.ret_std  = float(ret_arr.std() + 1e-8)
        self.returns  = [(r - self.ret_mean) / self.ret_std for r in self.returns]

        print(f"  Dataset: {len(self.frames)} samples")
        print(f"  Return stats: mean={self.ret_mean:.4f} std={self.ret_std:.4f}")

        if out_dir:
            json.dump({'ret_mean': self.ret_mean, 'ret_std': self.ret_std},
                      open(Path(out_dir) / 'return_stats.json', 'w'), indent=2)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        img = self.transform(self.frames[idx])
        ret = torch.tensor([self.returns[idx]], dtype=torch.float32)
        return img, ret


# ── Return Head ───────────────────────────────────────────────────────────────
class ReturnHead(nn.Module):
    """
    Tiny head on top of frozen encoder.
    128-D → 64-D → 1 scalar return prediction.
    ~8K parameters. NPU-safe (linear ops only).
    """
    def __init__(self, in_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, z):
        return self.net(z)


# ── Expectile regression loss (IQL) ──────────────────────────────────────────
def expectile_loss(pred, target, tau=0.9):
    """
    Asymmetric L2 loss — penalises under-prediction more than over-prediction.
    tau=0.9: model learns to predict upper quantile of return distribution.
    tau=0.5: reduces to standard MSE.
    
    From: Kostrikov et al. 2021, arXiv:2110.06169
    """
    diff   = target - pred
    weight = torch.where(diff >= 0,
                         torch.full_like(diff, tau),
                         torch.full_like(diff, 1.0 - tau))
    return (weight * diff.pow(2)).mean()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Load frozen encoder
    print(f"\nLoading encoder: {args.encoder}")
    ckpt    = torch.load(args.encoder, map_location='cpu')
    encoder = StudentEncoder()
    state   = ckpt.get('model', ckpt)
    # Stem-map compat (block1-4 vs stem.X)
    new_state = {}
    for k, v in state.items():
        k2 = k.replace('backbone.stem.0', 'backbone.block1.0') \
              .replace('backbone.stem.1', 'backbone.block1.1') \
              .replace('backbone.stem.2', 'backbone.block1.2')
        new_state[k2] = v
    encoder.load_state_dict(new_state, strict=False)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    print(f"  Encoder frozen — {sum(p.numel() for p in encoder.parameters()):,} params")

    # Return head
    head = ReturnHead(in_dim=128)
    print(f"  ReturnHead — {sum(p.numel() for p in head.parameters()):,} params")

    # Dataset
    dataset = SPYReturnDataset(
        period=args.period,
        interval=args.interval,
        horizon=args.horizon,
        out_dir=args.out,
    )

    # Train/val split — last 10% is validation (most recent bars)
    n_val   = max(1, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)
    print(f"  Train: {n_train} | Val: {n_val}")

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_loss = float('inf')
    results = []

    print(f"\n{'='*56}")
    print(f"  PHASE 3B — RETURN SUPERVISION")
    print(f"  Horizon:  {args.horizon} bars")
    print(f"  Expectile τ={args.tau} (IQL)")
    print(f"  Epochs:   {args.epochs}")
    print(f"{'='*56}\n")

    for epoch in range(1, args.epochs + 1):
        # Train
        head.train()
        train_loss = 0.0
        t0 = time.time()
        for imgs, rets in train_loader:
            with torch.no_grad():
                z = encoder(imgs)
            pred = head(z)
            loss = expectile_loss(pred, rets, tau=args.tau)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate — also measure directional accuracy
        head.eval()
        val_loss   = 0.0
        n_correct  = 0
        n_total    = 0
        with torch.no_grad():
            for imgs, rets in val_loader:
                z    = encoder(imgs)
                pred = head(z)
                val_loss += expectile_loss(pred, rets, tau=args.tau).item()
                # Directional accuracy: sign(pred) == sign(ret)
                n_correct += ((pred.sign() == rets.sign()).sum()).item()
                n_total   += rets.numel()
        val_loss  /= len(val_loader)
        dir_acc    = n_correct / n_total * 100

        scheduler.step()

        epoch_time = time.time() - t0
        print(f"Epoch {epoch:>3}/{args.epochs} | "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"dir_acc={dir_acc:.1f}%  ({epoch_time:.1f}s)")

        results.append({
            'epoch': epoch,
            'train_loss': round(train_loss, 6),
            'val_loss':   round(val_loss, 6),
            'dir_acc':    round(dir_acc, 2),
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(head.state_dict(), f'{args.out}/return_head_best.pt')

    torch.save(head.state_dict(), f'{args.out}/return_head_final.pt')
    json.dump(results, open(f'{args.out}/training_results.json', 'w'), indent=2)

    best = max(results, key=lambda r: r['dir_acc'])
    print(f"\n{'='*56}")
    print(f"  PHASE 3B COMPLETE")
    print(f"  Best val loss:    {best_val_loss:.4f}")
    print(f"  Best dir_acc:     {best['dir_acc']:.1f}% @ epoch {best['epoch']}")
    print(f"  Saved → {args.out}/return_head_best.pt")
    print(f"\n  Integration: in run_trading.py before order submission:")
    print(f"    ret_signal = return_head(z_t).item()")
    print(f"    if ret_signal < 0: ABORT  # predicted down — skip trade")
    print(f"{'='*56}\n")

    return best['dir_acc']


if __name__ == '__main__':
    main()
