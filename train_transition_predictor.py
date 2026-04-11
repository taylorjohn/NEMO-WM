"""
train_transition_predictor.py — CORTEX-PE v16.14
GPS + Ego-Motion → Delta Position Transition Predictor

Learns to predict (Δlat, Δlon) from current navigation state,
enabling dead-reckoning that complements the visual quasimetric.

Input  (8-D):
    lat, lon                    — current GPS position
    compass_bearing             — heading (degrees)
    gps_vx, gps_vy, gps_vz     — GPS velocity
    cmd_linear_vel              — commanded linear velocity
    cmd_angular_vel             — commanded angular velocity

Output (2-D):
    Δlat, Δlon                  — position delta to next timestep

Architecture:
    MLP: Linear(8→64) → LayerNorm → GELU → Linear(64→64) → LayerNorm → GELU → Linear(64→2)
    ~8.5K params — deliberately tiny, this is a kinematics approximator not a world model

Training:
    Loss : MSE on normalized deltas
    Data : consecutive (t, t+1) pairs from RECON HDF5 files
    Split: 90/10 train/val by file

Usage:
    python train_transition_predictor.py --data ./recon_data/recon_release
    python train_transition_predictor.py --data ./recon_data/recon_release --epochs 20 --max-files 100
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ── Model ──────────────────────────────────────────────────────────────────

class TransitionPredictor(nn.Module):
    """
    Predicts normalized (Δlat, Δlon) from 8-D navigation state.

    Deliberately small: kinematics is low-complexity, overfitting
    a large model to GPS noise is counterproductive.
    """
    INPUT_DIM  = 8
    OUTPUT_DIM = 2

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.INPUT_DIM, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, self.OUTPUT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Dataset ────────────────────────────────────────────────────────────────

def _extract_pairs(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Extract consecutive (state_t, delta_t) pairs from one HDF5 file.

    State  : [lat, lon, bearing, gps_vx, gps_vy, gps_vz, cmd_lin, cmd_ang]
    Delta  : [Δlat, Δlon]  (position at t+1 minus position at t)

    Returns None if file lacks required keys or has < 2 valid GPS frames.
    """
    required = [
        "gps/latlong", "gps/velocity", "gps/is_fixed",
        "imu/compass_bearing",
        "commands/linear_velocity", "commands/angular_velocity",
    ]
    try:
        with h5py.File(path, "r") as f:
            for key in required:
                if key not in f:
                    return None

            is_fixed  = f["gps/is_fixed"][:]               # (T,) bool
            latlong   = f["gps/latlong"][:]                 # (T, 2)
            gps_vel   = f["gps/velocity"][:]                # (T, 3)
            bearing   = f["imu/compass_bearing"][:]         # (T,)
            cmd_lin   = f["commands/linear_velocity"][:]    # (T,)
            cmd_ang   = f["commands/angular_velocity"][:]   # (T,)

        T = latlong.shape[0]
        if T < 2:
            return None

        # Only use frames where GPS is locked
        valid = is_fixed.astype(bool)

        states, deltas = [], []
        for t in range(T - 1):
            if not (valid[t] and valid[t + 1]):
                continue

            state = np.array([
                latlong[t, 0],   # lat
                latlong[t, 1],   # lon
                bearing[t],      # compass heading (degrees)
                gps_vel[t, 0],   # vx
                gps_vel[t, 1],   # vy
                gps_vel[t, 2],   # vz
                cmd_lin[t],      # commanded linear velocity
                cmd_ang[t],      # commanded angular velocity
            ], dtype=np.float32)

            delta = np.array([
                latlong[t + 1, 0] - latlong[t, 0],  # Δlat
                latlong[t + 1, 1] - latlong[t, 1],  # Δlon
            ], dtype=np.float32)

            # Hard threshold: at 4Hz max ~10 m/s → ~0.000022° per step
            # Anything > 0.001° (~111m) is a GPS glitch — discard
            if np.any(np.abs(delta) > 0.001):
                continue

            # Discard any pair with NaN/Inf in state or delta
            if not (np.isfinite(state).all() and np.isfinite(delta).all()):
                continue

            states.append(state)
            deltas.append(delta)

        if len(states) < 1:
            return None

        return np.stack(states), np.stack(deltas)

    except Exception:
        return None


class TransitionDataset(Dataset):
    """
    Aggregated (state, delta) pairs from multiple RECON HDF5 files.
    Normalizes inputs and outputs using per-channel statistics.
    """

    def __init__(
        self,
        files: list[Path],
        state_mean: np.ndarray | None = None,
        state_std:  np.ndarray | None = None,
        delta_mean: np.ndarray | None = None,
        delta_std:  np.ndarray | None = None,
        verbose: bool = True,
    ):
        all_states, all_deltas = [], []
        skipped = 0

        for i, path in enumerate(files):
            result = _extract_pairs(path)
            if result is None:
                skipped += 1
                continue
            states, deltas = result
            all_states.append(states)
            all_deltas.append(deltas)

            if verbose and (i + 1) % 500 == 0:
                pairs_so_far = sum(len(s) for s in all_states)
                print(f"    {i+1:5d}/{len(files)} files  |  {pairs_so_far:,} pairs")

        if not all_states:
            raise RuntimeError("No valid pairs extracted — check GPS fix coverage")

        self.states = np.concatenate(all_states, axis=0)  # (N, 8)
        self.deltas = np.concatenate(all_deltas, axis=0)  # (N, 2)

        if verbose:
            print(f"    {len(files)}/{len(files)} files  |  {len(self.states):,} pairs  "
                  f"(skipped {skipped})")

        # ── Normalization stats (fit on train, reuse on val) ──────────────
        # Use median + IQR-based scale for robustness against GPS outliers
        if state_mean is None:
            self.state_mean = np.nanmedian(self.states, axis=0)
            q75s = np.nanpercentile(self.states, 75, axis=0)
            q25s = np.nanpercentile(self.states, 25, axis=0)
            self.state_std  = (q75s - q25s).clip(min=1e-8)
            self.delta_mean = np.nanmedian(self.deltas, axis=0)
            q75d = np.nanpercentile(self.deltas, 75, axis=0)
            q25d = np.nanpercentile(self.deltas, 25, axis=0)
            self.delta_std  = (q75d - q25d).clip(min=1e-8)
        else:
            self.state_mean = state_mean
            self.state_std  = state_std
            self.delta_mean = delta_mean
            self.delta_std  = delta_std

        # Drop any rows that still contain NaN/Inf after stacking
        valid_rows = (np.isfinite(self.states).all(axis=1) &
                      np.isfinite(self.deltas).all(axis=1))
        n_before = len(self.states)
        self.states = self.states[valid_rows]
        self.deltas = self.deltas[valid_rows]
        n_dropped = n_before - len(self.states)
        if n_dropped > 0:
            print(f"    ⚠️  Dropped {n_dropped} rows with NaN/Inf inputs")

        # Normalize then clip to [-10, 10] to prevent outlier explosion
        self._states_n = np.clip(
            (self.states - self.state_mean) / self.state_std, -10, 10
        ).astype(np.float32)
        self._deltas_n = np.clip(
            (self.deltas - self.delta_mean) / self.delta_std, -10, 10
        ).astype(np.float32)

        # Sanity check
        assert not np.isnan(self._states_n).any(), "NaN in normalized states after cleaning"
        assert not np.isnan(self._deltas_n).any(), "NaN in normalized deltas after cleaning"

    def __len__(self) -> int:
        return len(self._states_n)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self._states_n[idx]),
            torch.from_numpy(self._deltas_n[idx]),
        )

    @property
    def norm_stats(self) -> dict[str, np.ndarray]:
        return {
            "state_mean": self.state_mean,
            "state_std":  self.state_std,
            "delta_mean": self.delta_mean,
            "delta_std":  self.delta_std,
        }


# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluate_mae_meters(
    model: TransitionPredictor,
    loader: DataLoader,
    delta_mean: np.ndarray,
    delta_std:  np.ndarray,
    device: torch.device,
) -> tuple[float, float]:
    """
    Compute mean absolute error in meters (not normalized space).

    Lat/lon → meters conversion:
        1 degree lat ≈ 111,139 m
        1 degree lon ≈ 111,139 × cos(lat) m  (use mean lat for simplicity)
    """
    model.eval()
    all_pred, all_true = [], []

    with torch.no_grad():
        for states, deltas in loader:
            states = states.to(device)
            pred_n = model(states).cpu().numpy()
            # Denormalize
            pred = pred_n * delta_std + delta_mean
            true = deltas.numpy() * delta_std + delta_mean
            all_pred.append(pred)
            all_true.append(true)

    pred = np.concatenate(all_pred)  # (N, 2)
    true = np.concatenate(all_true)

    # Δlat → meters, Δlon → meters (approximate, good enough for eval)
    lat_m = 111_139.0
    lon_m = 111_139.0 * np.cos(np.radians(37.0))  # RECON ~California lat

    err_lat_m = np.abs(pred[:, 0] - true[:, 0]) * lat_m
    err_lon_m = np.abs(pred[:, 1] - true[:, 1]) * lon_m
    err_dist_m = np.sqrt((pred[:, 0] - true[:, 0])**2 * lat_m**2
                       + (pred[:, 1] - true[:, 1])**2 * lon_m**2)

    return float(err_dist_m.mean()), float(err_dist_m.std())


# ── Training loop ──────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    print("\n" + "═" * 60)
    print("  RECON Transition Predictor — CORTEX-PE v16.14")
    print("═" * 60)
    print(f"  Data dir : {args.data}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Batch    : {args.batch_size}")
    print(f"  Hidden   : {args.hidden}")
    print(f"  Device   : {args.device}")

    device = torch.device(args.device)

    # ── File split ────────────────────────────────────────────────────────
    data_dir = Path(args.data)
    files = sorted(data_dir.rglob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No .hdf5 files found in {data_dir}")

    if args.max_files:
        files = files[: args.max_files]

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(files))
    split = max(1, int(len(files) * 0.9))
    train_files = [files[i] for i in idx[:split]]
    val_files   = [files[i] for i in idx[split:]]

    print(f"\n  Files: {len(files)} total  ({len(train_files)} train / {len(val_files)} val)")

    # ── Datasets ──────────────────────────────────────────────────────────
    print("\n── Building train dataset " + "─" * 33)
    train_ds = TransitionDataset(train_files, verbose=True)
    stats = train_ds.norm_stats

    print("\n── Building val dataset " + "─" * 35)
    val_ds = TransitionDataset(val_files, verbose=True, **stats)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = TransitionPredictor(hidden=args.hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n── TransitionPredictor: {n_params:,} params " + "─" * 20)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    criterion = nn.MSELoss()

    # ── Output dir ────────────────────────────────────────────────────────
    out_dir = Path("checkpoints/recon_transition")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n── Training: {args.epochs} epochs ──────────")
    print(f"   Train pairs: {len(train_ds):,}  |  Val pairs: {len(val_ds):,}")
    print(f"   Steps/epoch: {len(train_loader)}")

    best_val  = float("inf")
    best_mae  = float("inf")
    log       = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()

        # Train
        model.train()
        train_loss = 0.0
        for states, deltas in train_loader:
            states, deltas = states.to(device), deltas.to(device)
            optimizer.zero_grad()
            pred = model(states)
            loss = criterion(pred, deltas)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        scheduler.step()

        # Val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for states, deltas in val_loader:
                states, deltas = states.to(device), deltas.to(device)
                val_loss += criterion(model(states), deltas).item()
        val_loss /= len(val_loader)

        # MAE in meters
        mae_m, mae_std = evaluate_mae_meters(
            model, val_loader,
            stats["delta_mean"], stats["delta_std"], device,
        )

        elapsed = time.perf_counter() - t0
        improved = ""
        if val_loss < best_val:
            best_val = val_loss
            best_mae = mae_m
            improved = "  ✅ New best"
            # Save best checkpoint
            torch.save({
                "model_state_dict": model.state_dict(),
                "norm_stats":       {k: v.tolist() for k, v in stats.items()},
                "hidden":           args.hidden,
                "epoch":            epoch,
                "val_loss":         val_loss,
                "val_mae_m":        mae_m,
            }, out_dir / "transition_best.pt")

        print(f"  Epoch {epoch:3d}/{args.epochs}  "
              f"train={train_loss:.6f}  val={val_loss:.6f}  "
              f"MAE={mae_m:.3f}±{mae_std:.3f}m  "
              f"{elapsed:.1f}s{improved}")

        log.append({"epoch": epoch, "train": train_loss,
                    "val": val_loss, "mae_m": mae_m})

    # Save final
    torch.save({
        "model_state_dict": model.state_dict(),
        "norm_stats":       {k: v.tolist() for k, v in stats.items()},
        "hidden":           args.hidden,
        "epoch":            args.epochs,
        "val_loss":         val_loss,
        "val_mae_m":        mae_m,
    }, out_dir / "transition_final.pt")

    with open(out_dir / "log.json", "w") as f:
        json.dump(log, f, indent=2)

    print("\n" + "═" * 60)
    print(f"  Done.  Best val MAE: {best_mae:.3f} m")
    print(f"  Best:  {out_dir / 'transition_best.pt'}")
    print(f"  Final: {out_dir / 'transition_final.pt'}")
    print("═" * 60 + "\n")

    # ── Quick inference demo ───────────────────────────────────────────────
    print("  Inference demo (10 val samples):")
    model.eval()
    sample_states = torch.from_numpy(val_ds._states_n[:10]).to(device)
    sample_deltas_true = val_ds._deltas_n[:10] * stats["delta_std"] + stats["delta_mean"]
    with torch.no_grad():
        sample_pred_n = model(sample_states).cpu().numpy()
    sample_pred = sample_pred_n * stats["delta_std"] + stats["delta_mean"]

    lat_m, lon_m = 111_139.0, 111_139.0 * np.cos(np.radians(37.0))
    print(f"  {'Sample':>6}  {'True Δlat(mm)':>14}  {'Pred Δlat(mm)':>14}  {'Err(m)':>8}")
    for i in range(10):
        err = np.sqrt(
            ((sample_pred[i, 0] - sample_deltas_true[i, 0]) * lat_m) ** 2 +
            ((sample_pred[i, 1] - sample_deltas_true[i, 1]) * lon_m) ** 2
        )
        print(f"  {i:6d}  "
              f"{sample_deltas_true[i,0]*1e6:14.3f}  "
              f"{sample_pred[i,0]*1e6:14.3f}  "
              f"{err:8.3f}m")


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RECON transition predictor")
    parser.add_argument("--data",       required=True,       help="RECON data directory")
    parser.add_argument("--epochs",     type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden",     type=int, default=64)
    parser.add_argument("--lr",         type=float, default=3e-3)
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--max-files",  type=int, default=None,
                        help="Cap file count for quick tests")
    args = parser.parse_args()
    train(args)
