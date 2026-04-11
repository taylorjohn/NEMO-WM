"""
train_recon_position_planner.py
================================
Position-only quasimetric planner for RECON outdoor navigation.

Bypasses the visual encoder entirely. Uses jackal/position (T,3) GPS/odometry
coordinates as the state representation directly.

Why this works better:
  - Visual encoder has no outdoor priors → consecutive frames map to near-identical latents
  - GPS position is ground truth navigation state — no representation learning needed
  - Quasimetric structure is natural in position space: d(A→B) reflects actual
    traversal cost, not symmetric Euclidean distance (obstacles make it asymmetric)

State representation: (x, y) from jackal/position[:, :2]
Action representation: (linear_vel, angular_vel) from commands/

At inference time: encode current GPS + goal GPS → plan actions via MeZO.
Integrates with the production loop via a position encoder stub that wraps
GPS readings into the same interface as the visual StudentEncoder.

Usage:
    python train_recon_position_planner.py --train \
        --data ./recon_data/recon_release \
        --steps 5000 --max-files 2000 \
        --out ./checkpoints/recon_pos_planner

    python train_recon_position_planner.py --eval \
        --checkpoint ./checkpoints/recon_pos_planner/planner_final.pt \
        --data ./recon_data/recon_release
"""

import argparse
import json
import random
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Position normalisation ─────────────────────────────────────────────────
# RECON jackal positions are in metres relative to start.
# Normalise to [-1, 1] over a ~100m arena.
POS_SCALE = 50.0   # metres — clip and normalise


def normalise_pos(pos: np.ndarray) -> np.ndarray:
    """(T, 2) metres → (T, 2) in [-1, 1]"""
    return np.clip(pos / POS_SCALE, -1.0, 1.0).astype(np.float32)


def normalise_action(lv: np.ndarray, av: np.ndarray) -> np.ndarray:
    """Combine and normalise (linear_vel, angular_vel) to [-1, 1]"""
    lv = np.clip(lv, -1.5, 1.5) / 1.5
    av = np.clip(av, -2.0, 2.0) / 2.0
    return np.stack([lv, av], axis=-1).astype(np.float32)


# ── Dataset ────────────────────────────────────────────────────────────────

class RECONPositionDataset(torch.utils.data.Dataset):
    """
    Loads (pos_t, action_t, pos_t1) triplets from RECON HDF5 files.
    No image loading — position only.

    Keys used:
        jackal/position  (T, 3) float64  → take [:, :2] for x, y
        commands/linear_velocity  (T,)
        commands/angular_velocity (T,)
    """

    def __init__(
        self,
        data_dir: str,
        max_files: int = 2000,
        triplets_per_file: int = 16,
        horizon: int = 1,          # steps between t and t+horizon
        seed: int = 42,
    ):
        import h5py
        self.h5py = h5py
        self.data_dir = Path(data_dir)
        self.triplets = []   # (pos_t, action_t, pos_t1)
        self.goals    = []   # (pos_t, pos_goal) for quasimetric training

        all_files = sorted(self.data_dir.glob("*.hdf5"))
        if not all_files:
            all_files = sorted(self.data_dir.glob("**/*.hdf5"))
        if not all_files:
            raise RuntimeError(f"No .hdf5 files in {data_dir}")

        random.seed(seed)
        files = random.sample(all_files, min(max_files, len(all_files)))
        print(f"Loading RECON positions: {len(files)}/{len(all_files)} files "
              f"× {triplets_per_file} triplets...")

        n_loaded = 0
        for fpath in files:
            try:
                t, g = self._load_file(str(fpath), triplets_per_file, horizon)
                self.triplets.extend(t)
                self.goals.extend(g)
                n_loaded += 1
            except Exception:
                continue

        print(f"  Loaded {n_loaded} files → "
              f"{len(self.triplets)} triplets, {len(self.goals)} goal pairs")

    def _load_file(self, fpath: str, n_triplets: int, horizon: int):
        import h5py
        triplets = []
        goals = []

        with h5py.File(fpath, "r") as f:
            # Position
            if "jackal" not in f or "position" not in f["jackal"]:
                return [], []
            pos_raw = f["jackal"]["position"][:, :2].astype(np.float64)  # (T, 2)

            # Actions
            if "commands" not in f:
                return [], []
            lv = f["commands"]["linear_velocity"][:].astype(np.float64)
            av = f["commands"]["angular_velocity"][:].astype(np.float64)

            T = min(len(pos_raw), len(lv), len(av))
            if T < horizon + 2:
                return [], []

            # Make positions relative to trajectory start
            pos_raw = pos_raw[:T] - pos_raw[0]   # origin at start
            pos = normalise_pos(pos_raw)
            actions = normalise_action(lv[:T], av[:T])

            # Sample consecutive triplets (t, t+horizon)
            indices = random.sample(range(T - horizon), min(n_triplets, T - horizon))
            for t in indices:
                p_t   = torch.from_numpy(pos[t].copy())
                p_t1  = torch.from_numpy(pos[t + horizon].copy())
                act_t = torch.from_numpy(actions[t].copy())
                triplets.append((p_t, act_t, p_t1))

            # Sample goal pairs: (current, distant goal) — for quasimetric
            # Use pairs with varying temporal distance
            for _ in range(n_triplets):
                t_s = random.randint(0, T - 2)
                t_g = random.randint(t_s + 1, min(t_s + 30, T - 1))
                goals.append((
                    torch.from_numpy(pos[t_s].copy()),
                    torch.from_numpy(pos[t_g].copy()),
                    t_g - t_s,   # temporal distance (steps)
                ))

        return triplets, goals

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int):
        return self.triplets[idx]


# ── Quasimetric planner ────────────────────────────────────────────────────

class PositionQuasimetricPlanner(nn.Module):
    """
    Learns three things:
      1. Transition predictor: pos_t + action → pos_t1
      2. Quasimetric distance: d(pos_s → pos_g) — asymmetric navigation cost
      3. Implicit policy: which action minimises d(pred_pos → goal)

    State dim: 2 (normalised x, y)
    Action dim: 2 (normalised linear_vel, angular_vel)
    """

    def __init__(self, state_dim: int = 2, action_dim: int = 2, hidden: int = 128):
        super().__init__()

        # Transition predictor: pos_t + action → delta_pos
        self.predictor = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, state_dim),
            nn.Tanh(),   # output in [-1, 1] matching normalised positions
        )

        # Quasimetric distance head: d(s → g) ≥ 0, asymmetric
        # Input: concat(pos_s, pos_g) → scalar distance
        self.dist_head = nn.Sequential(
            nn.Linear(state_dim * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Softplus(),   # ensures d ≥ 0
        )

        # Value function: V(pos_s, pos_g) ≈ -d(s→g) for planning
        self.value_head = nn.Sequential(
            nn.Linear(state_dim * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def predict_next(self, pos: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """pos: (B, 2), action: (B, 2) → pos_next: (B, 2)"""
        x = torch.cat([pos, action], dim=-1)
        delta = self.predictor(x)
        return torch.clamp(pos + delta * 0.1, -1.0, 1.0)  # small residual steps

    def quasimetric(self, pos_s: torch.Tensor, pos_g: torch.Tensor) -> torch.Tensor:
        """Asymmetric navigation cost d(s → g): (B, 2), (B, 2) → (B, 1)"""
        return self.dist_head(torch.cat([pos_s, pos_g], dim=-1))

    def value(self, pos_s: torch.Tensor, pos_g: torch.Tensor) -> torch.Tensor:
        return self.value_head(torch.cat([pos_s, pos_g], dim=-1))

    def rollout(self, pos0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Roll out H steps from pos0.
        pos0:    (2,) initial position
        actions: (H, 2) action sequence
        Returns: (H+1, 2) trajectory
        """
        traj = [pos0.unsqueeze(0)]
        p = pos0.unsqueeze(0)
        for t in range(len(actions)):
            p = self.predict_next(p, actions[t].unsqueeze(0))
            traj.append(p)
        return torch.cat(traj, dim=0)


# ── Training ───────────────────────────────────────────────────────────────

def compute_losses(planner, pos_t, action, pos_t1):
    """
    Loss with contrastive quasimetric on varied-horizon goal pairs.

    The key fix: quasimetric is trained contrastively —
      - nearby pairs (small Euclidean dist) should have small d
      - distant pairs should have large d
      - asymmetry emerges naturally from ordering constraints
    """
    # 1. Prediction loss (MSE on next position)
    pos_pred = planner.predict_next(pos_t, action)
    l_pred = F.mse_loss(pos_pred, pos_t1)

    # 2. Contrastive quasimetric:
    #    d(pos_t, pos_t1) should be proportional to Euclidean distance
    #    d(pos_t1, pos_t) should be >= d(pos_t, pos_t1) (directional cost)
    eucl = torch.norm(pos_t1 - pos_t, dim=-1, keepdim=True).detach()
    d_fwd = planner.quasimetric(pos_t, pos_t1)
    d_bwd = planner.quasimetric(pos_t1, pos_t)

    # Calibrate forward distance to Euclidean (MSE)
    l_calib = F.mse_loss(d_fwd, eucl)

    # Asymmetry: d_bwd >= d_fwd (navigation is not reversible with same cost)
    l_asym = F.relu(d_fwd - d_bwd + 0.01).mean()

    # 3. Contrastive: shuffled pairs should have larger distance than aligned
    idx = torch.randperm(len(pos_t), device=pos_t.device)
    pos_neg = pos_t1[idx]                           # shuffled targets
    d_neg = planner.quasimetric(pos_t, pos_neg)     # random pairs
    eucl_neg = torch.norm(pos_neg - pos_t, dim=-1, keepdim=True).detach()
    l_contrast = F.mse_loss(d_neg, eucl_neg)        # also calibrate negatives

    # 4. Triangle inequality
    pos_m = pos_t[idx]
    d_ac = planner.quasimetric(pos_t, pos_t1)
    d_am = planner.quasimetric(pos_t, pos_m)
    d_mc = planner.quasimetric(pos_m, pos_t1)
    l_tri = F.relu(d_ac - d_am - d_mc + 0.001).mean()

    loss = l_pred + l_calib + 0.5 * l_asym + 0.2 * l_contrast + 0.1 * l_tri

    return {
        "total": loss,
        "pred": l_pred,
        "calib": l_calib,
        "asym": l_asym,
        "tri": l_tri,
    }


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = RECONPositionDataset(
        args.data,
        max_files=args.max_files,
        triplets_per_file=args.triplets_per_file,
    )

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    planner = PositionQuasimetricPlanner(state_dim=2, action_dim=2).to(device)
    n_params = sum(p.numel() for p in planner.parameters())
    print(f"\n{'='*60}")
    print(f"  RECON POSITION QUASIMETRIC PLANNER")
    print(f"  State: (x, y) normalised GPS — no visual encoder")
    print(f"  Triplets: {len(dataset)}  |  Params: {n_params:,}")
    print(f"  Steps: {args.steps}")
    print(f"  Loss: L_pred(MSE) + 0.5*L_qm + 0.1*L_tri + 0.1*L_calib")
    print(f"{'='*60}\n")

    optimizer = torch.optim.AdamW(planner.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.05
    )

    step = 0
    log = []

    while step < args.steps:
        for pos_t, action, pos_t1 in loader:
            if step >= args.steps:
                break

            pos_t  = pos_t.to(device)
            action = action.to(device)
            pos_t1 = pos_t1.to(device)

            losses = compute_losses(planner, pos_t, action, pos_t1)

            optimizer.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(planner.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % 100 == 0:
                entry = {k: v.item() for k, v in losses.items()}
                entry["step"] = step
                log.append(entry)
                print(f"Step {step:4d}/{args.steps} | "
                      f"total={losses['total'].item():.5f}  "
                      f"pred={losses['pred'].item():.5f}  "
                      f"calib={losses['calib'].item():.5f}  "
                      f"asym={losses['asym'].item():.5f}")

            if step % 1000 == 0:
                ckpt = out_dir / f"planner_step{step:05d}.pt"
                torch.save({
                    "step": step,
                    "planner": planner.state_dict(),
                    "state_dim": 2,
                    "action_dim": 2,
                    "pos_scale": POS_SCALE,
                }, ckpt)
                print(f"💾 Checkpoint: {ckpt}")

    # Final save
    final = out_dir / "planner_final.pt"
    torch.save({
        "step": step,
        "planner": planner.state_dict(),
        "state_dim": 2,
        "action_dim": 2,
        "pos_scale": POS_SCALE,
        "final_losses": log[-1] if log else {},
    }, final)

    with open(out_dir / "train_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n✅ Saved: {final}")
    print(f"   Final pred loss: {log[-1]['pred']:.5f}  (target < 0.005)")
    print(f"   Final qm loss:   {log[-1]['qm']:.5f}   (target < 0.02)")


# ── Evaluation: predict trajectories and measure positional error ──────────

def evaluate(args):
    device = torch.device("cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)

    planner = PositionQuasimetricPlanner(state_dim=2, action_dim=2)
    planner.load_state_dict(ckpt["planner"])
    planner.eval()
    print(f"✅ Loaded planner from {args.checkpoint}")

    import h5py
    all_files = sorted(Path(args.data).glob("*.hdf5"))[:50]  # eval on 50 files
    errors = []

    with torch.no_grad():
        for fpath in all_files:
            try:
                with h5py.File(str(fpath), "r") as f:
                    pos_raw = f["jackal"]["position"][:, :2].astype(np.float64)
                    lv = f["commands"]["linear_velocity"][:].astype(np.float64)
                    av = f["commands"]["angular_velocity"][:].astype(np.float64)

                T = min(len(pos_raw), len(lv), len(av), 50)
                pos_raw = (pos_raw[:T] - pos_raw[0])
                pos = normalise_pos(pos_raw)
                actions = normalise_action(lv[:T], av[:T])

                # Roll out predictor from t=0
                p = torch.from_numpy(pos[0]).unsqueeze(0)
                preds = [p.squeeze(0).numpy()]
                for t in range(T - 1):
                    act = torch.from_numpy(actions[t]).unsqueeze(0)
                    p = planner.predict_next(p, act)
                    preds.append(p.squeeze(0).numpy())

                preds = np.array(preds) * POS_SCALE   # back to metres
                truth = pos_raw[:T]
                err = np.mean(np.linalg.norm(preds - truth, axis=-1))
                errors.append(err)
            except Exception:
                continue

    if not errors:
        print("No files evaluated")
        return

    print(f"\n{'='*45}")
    print(f"  RECON Position Planner Evaluation")
    print(f"  Files: {len(errors)}")
    print(f"  Mean positional error: {np.mean(errors):.3f}m")
    print(f"  Median:                {np.median(errors):.3f}m")
    print(f"  90th percentile:       {np.percentile(errors, 90):.3f}m")
    print(f"  Target: < 2.0m mean error")
    print(f"{'='*45}")


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RECON position-only quasimetric planner")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--data", default="./recon_data/recon_release")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-files", type=int, default=2000)
    parser.add_argument("--triplets-per-file", type=int, default=16)
    parser.add_argument("--out", default="./checkpoints/recon_pos_planner")
    args = parser.parse_args()

    if args.train:
        train(args)
    elif args.eval:
        if not args.checkpoint:
            parser.error("--checkpoint required for --eval")
        evaluate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
