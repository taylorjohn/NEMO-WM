"""
eval_proprio_robustness.py — NeMo-WM Robustness Ablations
===========================================================
Tests proprioceptive encoder robustness to sensor failures and noise.

Ablations:
  baseline      : normal (no corruption)
  drop_heading  : zero sin_h, cos_h, delta_h (compass fails)
  drop_velocity : zero vel, contact (wheel encoder fails)
  drop_angular  : zero ang, delta_h (gyro fails)
  speed_2x      : multiply vel by 2.0 (faster robot)
  speed_0.5x    : multiply vel by 0.5 (slower robot)
  vel_noise     : add gaussian noise to vel, ang
  corrupt_10pct : randomly zero 10% of frames
  corrupt_30pct : randomly zero 30% of frames

These test whether the encoder truly generalises or memorises
RECON-specific signal statistics.

Usage:
    python eval_proprio_robustness.py \
        --proprio-ckpt checkpoints/cwm/proprio_kctx16_recon_ft.pt \
        --hdf5-dir recon_data/recon_release \
        --n-pairs 500

Output: AUROC per ablation — shows which sensors are critical.
"""

import argparse
import math
import random
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


# ── Load proprio encoder ──────────────────────────────────────────────────────

def load_proprio_encoder(ckpt_path: str, device: torch.device) -> nn.Module:
    """Load ProprioEncoderTemporal from checkpoint."""
    import sys, os
    sys.path.insert(0, os.getcwd())

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Detect architecture
    k_ctx = ckpt.get('k_ctx', 16)
    from train_proprio_6c import ProprioEncoderTemporal
    model = ProprioEncoderTemporal(k_ctx=k_ctx).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    print(f"  Loaded: {ckpt_path}")
    print(f"  epoch={ckpt.get('epoch','?')}  "
          f"top1_acc={ckpt.get('top1_acc',0):.4f}  "
          f"k_ctx={k_ctx}")
    return model, k_ctx


# ── Feature extraction (mirrors eval_recon_auroc.py) ─────────────────────────

def extract_frame(
    hf:          h5py.File,
    t:           int,
    corruption:  Callable,
) -> np.ndarray:
    """
    Extract 8-dim proprio frame at timestep t with optional corruption.

    Signals: [vel, ang, sin_h, cos_h, contact, d_lat, d_lon, delta_h]
    """
    vel_val = float(hf["commands"]["linear_velocity"][t])
    ang_val = float(hf["commands"]["angular_velocity"][t])

    # Heading from integrated angular velocity
    ang_all       = hf["commands"]["angular_velocity"][:t+1]
    heading_angle = float(sum(ang_all)) / 4.0
    sin_h = math.sin(heading_angle)
    cos_h = math.cos(heading_angle)

    contact = 1.0 if abs(vel_val) > 0.3 else 0.0

    # Heading delta
    if t > 0:
        prev_ang  = hf["commands"]["angular_velocity"][:t]
        prev_head = float(sum(prev_ang)) / 4.0
        delta_h   = heading_angle - prev_head
    else:
        delta_h = 0.0

    # No GPS (proprio-only)
    frame = np.array([
        vel_val, ang_val, sin_h, cos_h, contact, 0.0, 0.0, delta_h
    ], dtype=np.float32)

    return corruption(frame, t)


def extract_window(
    hf:         h5py.File,
    anchor_t:   int,
    k_ctx:      int,
    corruption: Callable,
) -> np.ndarray:
    """Extract k_ctx-frame window ending at anchor_t."""
    T      = len(hf["commands"]["linear_velocity"])
    frames = []
    for i in range(k_ctx):
        t = max(0, anchor_t - k_ctx + 1 + i)
        t = min(t, T - 1)
        frames.append(extract_frame(hf, t, corruption))
    return np.stack(frames, axis=0)  # (k_ctx, 8)


# ── Corruption functions ──────────────────────────────────────────────────────

def no_corruption(frame: np.ndarray, t: int) -> np.ndarray:
    return frame.copy()


def drop_heading(frame: np.ndarray, t: int) -> np.ndarray:
    """Zero sin_h, cos_h, delta_h — compass failure."""
    f = frame.copy()
    f[2] = 0.0   # sin_h
    f[3] = 0.0   # cos_h
    f[7] = 0.0   # delta_h
    return f


def drop_velocity(frame: np.ndarray, t: int) -> np.ndarray:
    """Zero vel, contact — wheel encoder failure."""
    f = frame.copy()
    f[0] = 0.0   # vel
    f[4] = 0.0   # contact
    return f


def drop_angular(frame: np.ndarray, t: int) -> np.ndarray:
    """Zero ang, delta_h — gyro failure."""
    f = frame.copy()
    f[1] = 0.0   # ang
    f[7] = 0.0   # delta_h
    return f


def speed_2x(frame: np.ndarray, t: int) -> np.ndarray:
    """Double velocity — faster robot."""
    f = frame.copy()
    f[0] *= 2.0
    return f


def speed_half(frame: np.ndarray, t: int) -> np.ndarray:
    """Halve velocity — slower robot."""
    f = frame.copy()
    f[0] *= 0.5
    return f


def vel_noise(frame: np.ndarray, t: int) -> np.ndarray:
    """Add gaussian noise to vel and ang."""
    f = frame.copy()
    f[0] += np.random.normal(0, 0.05)
    f[1] += np.random.normal(0, 0.02)
    return f


def corrupt_10pct(frame: np.ndarray, t: int) -> np.ndarray:
    """Zero entire frame 10% of the time."""
    if np.random.random() < 0.10:
        return np.zeros_like(frame)
    return frame.copy()


def corrupt_30pct(frame: np.ndarray, t: int) -> np.ndarray:
    """Zero entire frame 30% of the time."""
    if np.random.random() < 0.30:
        return np.zeros_like(frame)
    return frame.copy()


CORRUPTIONS = {
    'baseline':     (no_corruption,  'No corruption'),
    'drop_heading': (drop_heading,   'Heading zeroed (sin_h, cos_h, delta_h)'),
    'drop_velocity':(drop_velocity,  'Velocity zeroed (vel, contact)'),
    'drop_angular': (drop_angular,   'Angular zeroed (ang, delta_h)'),
    'speed_2x':     (speed_2x,       'Velocity ×2.0 (faster robot)'),
    'speed_0.5x':   (speed_half,     'Velocity ×0.5 (slower robot)'),
    'vel_noise':    (vel_noise,       'Gaussian noise on vel, ang'),
    'corrupt_10pct':(corrupt_10pct,  'Random frame dropout 10%'),
    'corrupt_30pct':(corrupt_30pct,  'Random frame dropout 30%'),
}


# ── Evaluation ────────────────────────────────────────────────────────────────

def get_hdf5_files(hdf5_dir: str, max_files: int) -> List[Path]:
    files = sorted(Path(hdf5_dir).glob('*.hdf5'))[:max_files]
    if not files:
        files = sorted(Path(hdf5_dir).glob('*.h5'))[:max_files]
    return files


def encode_window(
    model:      nn.Module,
    window:     np.ndarray,
    device:     torch.device,
) -> np.ndarray:
    """Encode a (k_ctx, 8) window to a unit-normalised embedding."""
    x = torch.from_numpy(window).unsqueeze(0).to(device)   # (1, k_ctx, 8)
    with torch.no_grad():
        z = model(x)
        z = torch.nn.functional.normalize(z, dim=-1)
    return z.cpu().numpy()[0]


def run_ablation(
    model:      nn.Module,
    k_ctx:      int,
    files:      List[Path],
    corruption: Callable,
    n_pairs:    int,
    k_pos_max:  int,
    k_neg_min:  int,
    device:     torch.device,
    seed:       int = 42,
) -> float:
    """
    Run dissociation eval with a given corruption function.
    Returns AUROC.
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    scores = []
    labels = []

    attempts = 0
    while len(scores) < n_pairs * 2 and attempts < n_pairs * 10:
        attempts += 1
        f = rng.choice(files)

        try:
            with h5py.File(str(f), 'r') as hf:
                T = len(hf["commands"]["linear_velocity"])
                if T < k_ctx + k_neg_min + 5:
                    continue

                anchor_t = rng.randint(k_ctx, T - k_neg_min - 1)

                # Positive: nearby frame
                k_pos   = rng.randint(1, min(k_pos_max, T - anchor_t - 1))
                pos_t   = anchor_t + k_pos

                # Negative: far frame (same file)
                neg_t   = rng.randint(
                    min(anchor_t + k_neg_min, T - 1),
                    T - 1
                )

                if pos_t >= T or neg_t >= T:
                    continue

                z_a   = encode_window(model, extract_window(hf, anchor_t, k_ctx, corruption), device)
                z_pos = encode_window(model, extract_window(hf, pos_t,    k_ctx, corruption), device)
                z_neg = encode_window(model, extract_window(hf, neg_t,    k_ctx, corruption), device)

                dist_pos = float(1 - np.dot(z_a, z_pos))
                dist_neg = float(1 - np.dot(z_a, z_neg))

                scores.extend([dist_pos, dist_neg])
                labels.extend([0, 1])

        except Exception:
            continue

    if len(scores) < 10:
        return 0.5

    try:
        auroc = roc_auc_score(labels, scores)
        return max(auroc, 1 - auroc)
    except Exception:
        return 0.5


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Proprio robustness ablations')
    ap.add_argument('--proprio-ckpt', required=True)
    ap.add_argument('--hdf5-dir',     default='recon_data/recon_release')
    ap.add_argument('--n-pairs',      type=int, default=500)
    ap.add_argument('--max-files',    type=int, default=500)
    ap.add_argument('--k-pos-max',    type=int, default=4)
    ap.add_argument('--k-neg-min',    type=int, default=32)
    ap.add_argument('--ablations',    nargs='+',
                    default=list(CORRUPTIONS.keys()))
    ap.add_argument('--device',       default='cpu')
    ap.add_argument('--seed',         type=int, default=42)
    args = ap.parse_args()

    device = torch.device(args.device)

    print(f"\nProprioceptive Robustness Ablations")
    print(f"{'='*60}")
    model, k_ctx = load_proprio_encoder(args.proprio_ckpt, device)

    files = get_hdf5_files(args.hdf5_dir, args.max_files)
    print(f"  Files: {len(files)}  n_pairs: {args.n_pairs}")
    print(f"  k_ctx: {k_ctx}  k_pos_max: {args.k_pos_max}  "
          f"k_neg_min: {args.k_neg_min}\n")

    results = {}
    baseline_auroc = None

    for ablation_name in args.ablations:
        if ablation_name not in CORRUPTIONS:
            print(f"  Unknown ablation: {ablation_name}, skipping")
            continue

        corruption_fn, description = CORRUPTIONS[ablation_name]

        t0    = time.time()
        auroc = run_ablation(
            model, k_ctx, files, corruption_fn,
            args.n_pairs, args.k_pos_max, args.k_neg_min,
            device, args.seed,
        )
        elapsed = time.time() - t0

        if ablation_name == 'baseline':
            baseline_auroc = auroc

        delta = f'{auroc - baseline_auroc:+.4f}' if baseline_auroc else '—'
        status = '✅' if auroc >= 0.90 else ('⚠️ ' if auroc >= 0.70 else '❌')

        results[ablation_name] = auroc
        print(f"  {ablation_name:15s}  AUROC={auroc:.4f}  {delta}  "
              f"{status}  ({elapsed:.0f}s)")

    # Summary table
    print(f"\n{'='*60}")
    print(f"  Robustness Summary")
    print(f"{'='*60}")
    print(f"  {'Ablation':20s}  {'AUROC':8s}  {'vs Baseline':12s}  "
          f"{'Description'}")
    print(f"  {'─'*75}")

    baseline = results.get('baseline', 0)
    for name, auroc in results.items():
        delta   = auroc - baseline
        _, desc = CORRUPTIONS[name]
        status  = '✅' if auroc >= 0.90 else ('⚠️ ' if auroc >= 0.70 else '❌')
        print(f"  {name:20s}  {auroc:.4f}    {delta:+.4f}       {status}  {desc}")

    # Key findings
    print(f"\n  Key Findings:")
    if 'drop_heading' in results and 'baseline' in results:
        drop = results['baseline'] - results['drop_heading']
        print(f"  Heading importance:  -{drop:.4f} AUROC when zeroed")
    if 'drop_velocity' in results and 'baseline' in results:
        drop = results['baseline'] - results['drop_velocity']
        print(f"  Velocity importance: -{drop:.4f} AUROC when zeroed")
    if 'speed_2x' in results and 'speed_0.5x' in results:
        diff = abs(results['speed_2x'] - results['speed_0.5x'])
        print(f"  Speed invariance:    {diff:.4f} AUROC gap (2× vs 0.5×)")
    if 'corrupt_30pct' in results and 'baseline' in results:
        drop = results['baseline'] - results['corrupt_30pct']
        print(f"  Frame dropout 30%:   -{drop:.4f} AUROC degradation")

    print(f"\n  Biological interpretation:")
    if 'drop_heading' in results:
        h_auroc = results['drop_heading']
        if h_auroc < 0.80:
            print(f"  Heading is CRITICAL (AUROC={h_auroc:.4f}) — "
                  f"mirrors Taube 1998 HD cell lesion effects")
        else:
            print(f"  Heading loss tolerated (AUROC={h_auroc:.4f}) — "
                  f"velocity compensates at short k_ctx")

    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
