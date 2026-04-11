"""
train_rope_temporal.py — CORTEX-PE
RoPE TemporalHead Training on Real RECON Data
==============================================

Trains the RoPETemporalHead on RECON HDF5 trajectories using:
  - jackal/position  → (t, x, y) coordinates for RoPE
  - images/rgb_left  → JPEG frames → StudentEncoder → 32-D latents
  - x-prediction loss with EMA target (collapse-free from UniFluids §3.4)

Replaces train_temporal_contrastive.py for the RoPE experiment.
Saves checkpoint compatible with export_temporal_head.py for NPU deployment.

Probe-first protocol:
  1. python train_rope_temporal.py --probe-only         (check data loading)
  2. python train_rope_temporal.py --max-files 5 --epochs 2   (smoke test)
  3. python train_rope_temporal.py --max-files 50 --epochs 20 (full run)

Compare against MLP baseline:
  python train_rope_temporal.py --baseline-compare --max-files 50 --epochs 20

Usage:
  python train_rope_temporal.py --data ./recon_data/recon_release
"""

from __future__ import annotations
import argparse, io, json, time
from pathlib import Path
import numpy as np

# Numpy version-safe trapezoid (trapz in 1.x, trapezoid in 2.x)
import numpy as _np
_np_trapz = getattr(_np, 'trapezoid', None) or getattr(_np, 'trapz', None)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# ── Local modules ─────────────────────────────────────────────────────────
from rope_temporal_head import (
    RoPETemporalHead, infonce_loss, xpred_loss, vicreg_loss
)

# StudentEncoder (copied from recon_navigator.py — no import dependency)
class StudentEncoder(nn.Module):
    """32-D CNN backbone matching CORTEX checkpoint format.

    Architecture matches saved checkpoints:
        backbone.stem — 4 × (Conv2d + BN + ReLU) with strides 2,2,2,1
                        channels: 3→16→32→64→32
        AdaptiveAvgPool2d(1,1) → 32-D latent

    Key layout in .pt files:
        backbone.stem.{0,3,6,9}.weight   — Conv2d weights
        backbone.stem.{1,4,7,10}.*       — BatchNorm params
        head.heads.*                     — task heads (ignored here)
    """
    LATENT_DIM = 32

    def __init__(self):
        super().__init__()
        self.backbone = _StudentBackbone()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.backbone(x)).flatten(1)   # [B, 32]

    def load_cortex_ckpt(self, path: str) -> None:
        """Load from CORTEX .pt checkpoint, ignoring head.* keys."""
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        state = ckpt.get("model", ckpt.get("student", ckpt))
        backbone_state = {
            k[len("backbone."):]: v
            for k, v in state.items()
            if k.startswith("backbone.")
        }
        missing, unexpected = self.backbone.load_state_dict(
            backbone_state, strict=False
        )
        loaded = len(backbone_state) - len(missing)
        print(f"  Loaded {loaded}/{len(backbone_state)} backbone tensors "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")


class _StudentBackbone(nn.Module):
    """Matches backbone.stem.* key layout in saved checkpoints."""
    def __init__(self):
        super().__init__()
        # stem indices: 0=Conv, 1=BN, 2=ReLU, 3=Conv, 4=BN, 5=ReLU, ...
        self.stem = nn.Sequential(
            nn.Conv2d(3,  16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),   # 0,1,2
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),   # 3,4,5
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),   # 6,7,8
            nn.Conv2d(64, 32, 3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),   # 9,10,11
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class MLPTemporalHead(nn.Module):
    """Baseline — current production head."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(StudentEncoder.LATENT_DIM, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, 64),
        )
    def forward(self, z, *a, **k): return self.net(z)
    @property
    def n_params(self): return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

IMG_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _load_trajectory(hdf_path: Path) -> dict | None:
    """Load one RECON HDF5 file → dict of latents + positions."""
    try:
        import h5py
        from PIL import Image
        with h5py.File(hdf_path, "r") as f:
            imgs_raw = f["images/rgb_left"][:]    # JPEG bytes, shape [T]
            pos      = f["jackal/position"][:]    # [T, 3]
        return {"imgs": imgs_raw, "pos": pos[:, :2]}  # keep x,y only
    except Exception as e:
        return None


class RECONPairDataset(Dataset):
    """
    Pre-encodes all frames with StudentEncoder, then builds (t, t+k) pairs.

    Coordinates:
      coord_t  = [t/T_max,  x_t/XY_SCALE,  y_t/XY_SCALE]
      coord_tk = [tk/T_max, x_tk/XY_SCALE, y_tk/XY_SCALE]

    where (x,y) are metres relative to trajectory origin.

    Labels for AUROC evaluation:
      close = 1  if displacement < CLOSE_THRESH metres
      far   = 0  if displacement > FAR_THRESH metres
    """

    T_MAX      = 70       # RECON trajectory length
    XY_SCALE   = 10.0     # metres → normalised
    CLOSE_THRESH = 1.0    # metres — "close" pair
    FAR_THRESH   = 3.0    # metres — "far" pair

    def __init__(
        self,
        hdf_files: list[Path],
        encoder: nn.Module,
        k_min: int   = 1,
        k_max: int   = 15,
        n_pairs: int = 8000,
        device: torch.device = torch.device("cpu"),
        seed: int = 42,
    ):
        rng  = np.random.default_rng(seed)
        enc  = encoder.to(device).eval()

        self._trajs: list[dict] = []

        from PIL import Image
        print(f"  Encoding {len(hdf_files)} trajectories...")
        t0 = time.time()

        for path in hdf_files:
            raw = _load_trajectory(path)
            if raw is None:
                continue

            imgs_raw = raw["imgs"]
            pos_raw  = raw["pos"]
            T        = len(imgs_raw)
            origin   = pos_raw[0].copy()
            pos_xy   = (pos_raw - origin) / self.XY_SCALE   # normalised metres

            # Encode all frames
            latents = np.zeros((T, StudentEncoder.LATENT_DIM), dtype=np.float32)
            for t in range(T):
                try:
                    img = Image.open(io.BytesIO(bytes(imgs_raw[t]))).convert("RGB")
                    inp = IMG_TRANSFORM(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        z = enc(inp).squeeze(0).cpu().numpy()
                    latents[t] = z
                except Exception:
                    latents[t] = latents[t-1] if t > 0 else 0.

            self._trajs.append({"latents": latents, "pos_xy": pos_xy, "T": T})

        print(f"  Encoded {len(self._trajs)} trajectories in {time.time()-t0:.1f}s")

        # Build pairs
        self._pairs: list[dict] = []
        for _ in range(n_pairs):
            if not self._trajs:
                break
            tr  = self._trajs[rng.integers(len(self._trajs))]
            T   = tr["T"]
            t   = int(rng.integers(0, max(1, T - k_max)))
            k   = int(rng.integers(k_min, min(k_max, T - t - 1) + 1))
            tk  = t + k

            pos_t  = tr["pos_xy"][t]
            pos_tk = tr["pos_xy"][tk]
            disp   = float(np.linalg.norm(pos_t - pos_tk) * self.XY_SCALE)

            # Coord vectors (already normalised by XY_SCALE during encoding)
            c_t  = np.array([t  / self.T_MAX, pos_t[0],  pos_t[1]],  dtype=np.float32)
            c_tk = np.array([tk / self.T_MAX, pos_tk[0], pos_tk[1]], dtype=np.float32)

            label = 1 if disp < self.CLOSE_THRESH else (
                    0 if disp > self.FAR_THRESH  else -1)   # -1 = ambiguous, skip eval

            self._pairs.append({
                "z_t":  tr["latents"][t].copy(),
                "z_tk": tr["latents"][tk].copy(),
                "c_t":  c_t, "c_tk": c_tk,
                "label": float(max(label, 0)),              # -1→0 for training
                "eval_label": label,
                "disp": disp, "k": k,
            })

        labels = np.array([p["eval_label"] for p in self._pairs])
        cd = np.mean([p["disp"] for p in self._pairs if p["eval_label"]==1]) if (labels==1).any() else 0
        fd = np.mean([p["disp"] for p in self._pairs if p["eval_label"]==0]) if (labels==0).any() else 0
        print(f"  {len(self._pairs)} pairs | "
              f"close={( labels==1).sum()} far={(labels==0).sum()} ambig={(labels==-1).sum()} | "
              f"close_disp={cd:.2f}m far_disp={fd:.2f}m")

    def __len__(self): return len(self._pairs)

    def __getitem__(self, i):
        p = self._pairs[i]
        return (torch.from_numpy(p["z_t"]),  torch.from_numpy(p["z_tk"]),
                torch.from_numpy(p["c_t"]),  torch.from_numpy(p["c_tk"]),
                torch.tensor(p["label"],      dtype=torch.float32))


# ─────────────────────────────────────────────────────────────────────────────
# AUROC
# ─────────────────────────────────────────────────────────────────────────────

def auroc_np(scores, labels):
    labels = np.asarray(labels, dtype=float)
    # Filter out ambiguous labels (-1)
    mask   = labels >= 0
    scores = np.asarray(scores)[mask]
    labels = labels[mask]
    if len(labels) == 0 or labels.sum() == 0 or (1-labels).sum() == 0:
        return 0.5   # degenerate — not enough class diversity
    order  = np.argsort(scores)          # ascending distance = more likely close
    tpr=[0.]; fpr=[0.]; tp=fp=0
    n_pos=labels.sum(); n_neg=len(labels)-n_pos
    for l in labels[order]:
        if l==1: tp+=1
        else:    fp+=1
        tpr.append(tp/n_pos); fpr.append(fp/n_neg)
    return float(_np_trapz(tpr, fpr))


# ─────────────────────────────────────────────────────────────────────────────
# Eval  (single-frame cosine distance for both heads — fair comparison)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(head, loader, device):
    head.eval()
    scores=[]; labels=[]; close_d=[]; far_d=[]
    with torch.no_grad():
        for z_t, z_tk, *_, lbl in loader:
            z_t=z_t.to(device); z_tk=z_tk.to(device)
            e_t  = head(z_t)
            e_tk = head(z_tk)
            d = 1.0 - F.cosine_similarity(
                F.normalize(e_t,dim=-1), F.normalize(e_tk,dim=-1), dim=-1)
            d_np=d.cpu().numpy(); l_np=lbl.numpy()
            scores.extend(d_np.tolist()); labels.extend(l_np.tolist())
            close_d.extend(d_np[l_np==1].tolist())
            far_d.extend(d_np[l_np==0].tolist())
    return {
        "auroc": auroc_np(scores, labels),
        "close": np.mean(close_d) if close_d else 0.,
        "far":   np.mean(far_d)   if far_d   else 0.,
        "sep":   np.mean(far_d)-np.mean(close_d) if close_d and far_d else 0.,
        "n_eval": int(np.array(labels).clip(0,1).sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_rope_epoch(head, loader, opt, device, T=0.07, ema_decay=0.996):
    head.train(); losses=[]; mses=[]; contrasts=[]
    for z_t, z_tk, c_t, c_tk, _ in loader:
        z_t=z_t.to(device); z_tk=z_tk.to(device)
        c_t=c_t.to(device); c_tk=c_tk.to(device)
        pred   = head(z_t, z_tk, c_t, c_tk)
        target = head.target_embed(z_tk)
        loss, bd = xpred_loss(pred, target, T)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step(); head.update_ema(ema_decay)
        losses.append(loss.item())
        mses.append(bd["mse"]); contrasts.append(bd["contrast"])
    return {"loss": np.mean(losses), "mse": np.mean(mses), "contrast": np.mean(contrasts)}


def train_mlp_epoch(head, loader, opt, device, T=0.07):
    head.train(); losses=[]
    for z_t, z_tk, *_ in loader:
        z_t=z_t.to(device); z_tk=z_tk.to(device)
        loss = infonce_loss(head(z_t), head(z_tk), T) + 0.1*vicreg_loss(head(z_t))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step(); losses.append(loss.item())
    return {"loss": np.mean(losses)}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="RoPE TemporalHead — RECON training")
    ap.add_argument("--data",      default="./recon_data/recon_release",
                    help="Directory of jackal_2019-*.hdf5 files")
    ap.add_argument("--encoder-ckpt", default=None,
                    help="Path to StudentEncoder checkpoint (.pt)")
    ap.add_argument("--max-files", type=int, default=None,
                    help="Limit number of HDF5 files (None = all)")
    ap.add_argument("--epochs",    type=int,   default=20)
    ap.add_argument("--batch",     type=int,   default=128)
    ap.add_argument("--lr",        type=float, default=1e-3)
    ap.add_argument("--k-min",     type=int,   default=1)
    ap.add_argument("--k-max",     type=int,   default=15)
    ap.add_argument("--n-pairs",   type=int,   default=8000)
    ap.add_argument("--T",         type=float, default=0.07, dest="temperature")
    ap.add_argument("--embed-dim", type=int,   default=96)
    ap.add_argument("--ema-decay", type=float, default=0.996)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--seed",      type=int,   default=42)
    ap.add_argument("--out-dir",   default="./checkpoints/recon_rope",
                    help="Checkpoint output directory")
    ap.add_argument("--probe-only", action="store_true",
                    help="Data probe only — no training")
    ap.add_argument("--baseline-compare", action="store_true",
                    help="Train MLP baseline in parallel for comparison")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load StudentEncoder ───────────────────────────────────────────────
    encoder = StudentEncoder()
    if args.encoder_ckpt:
        print(f"Loading StudentEncoder from {args.encoder_ckpt}")
        encoder.load_cortex_ckpt(args.encoder_ckpt)
        print(f"StudentEncoder loaded from {args.encoder_ckpt}")
    else:
        print("StudentEncoder: random weights (no checkpoint provided)")
    encoder.eval()

    # ── Discover HDF5 files ───────────────────────────────────────────────
    data_dir = Path(args.data)
    hdf_files = sorted(data_dir.glob("jackal_2019-*.hdf5"))
    if not hdf_files:
        raise FileNotFoundError(f"No jackal_2019-*.hdf5 files in {data_dir}")

    if args.max_files:
        hdf_files = hdf_files[:args.max_files]

    print(f"\nFound {len(hdf_files)} HDF5 files")

    # ── Probe ─────────────────────────────────────────────────────────────
    if args.probe_only:
        print("\n── Probe ──")
        for path in hdf_files[:3]:
            raw = _load_trajectory(path)
            if raw:
                T = len(raw["imgs"])
                print(f"  {path.name}: T={T}  pos_range_x={raw['pos'][:,0].ptp():.2f}m")
            else:
                print(f"  {path.name}: FAILED")
        return

    # Train / val split by file — minimum 2 val files for AUROC diversity
    n_val       = max(2, int(len(hdf_files) * args.val_split))
    val_files   = hdf_files[:n_val]
    train_files = hdf_files[n_val:]
    print(f"Train: {len(train_files)} files | Val: {len(val_files)} files")

    # ── Build datasets (encoding happens here) ────────────────────────────
    print("\nBuilding train dataset:")
    train_ds = RECONPairDataset(train_files, encoder,
                                k_min=args.k_min, k_max=args.k_max,
                                n_pairs=args.n_pairs, device=device, seed=args.seed)
    print("Building val dataset:")
    val_ds   = RECONPairDataset(val_files, encoder,
                                k_min=args.k_min, k_max=args.k_max,
                                n_pairs=max(500, args.n_pairs // 8),
                                device=device, seed=args.seed + 1)

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)

    # ── Models ────────────────────────────────────────────────────────────
    rope = RoPETemporalHead(latent_dim=StudentEncoder.LATENT_DIM, embed_dim=args.embed_dim).to(device)
    rope_opt = torch.optim.AdamW(rope.parameters(), lr=args.lr, weight_decay=1e-4)
    rope_sch = torch.optim.lr_scheduler.CosineAnnealingLR(rope_opt, args.epochs)

    mlp = mlp_opt = mlp_sch = None
    if args.baseline_compare:
        mlp     = MLPTemporalHead().to(device)
        mlp_opt = torch.optim.AdamW(mlp.parameters(), lr=args.lr, weight_decay=1e-4)
        mlp_sch = torch.optim.lr_scheduler.CosineAnnealingLR(mlp_opt, args.epochs)

    print(f"\nRoPE params: {rope.n_params:,}")
    if mlp: print(f"MLP  params: {mlp.n_params:,}")

    # ── Baseline eval ─────────────────────────────────────────────────────
    r0 = evaluate(rope, val_dl, device)
    print(f"\nInit RoPE AUROC={r0['auroc']:.4f}  "
          f"close={r0['close']:.4f}  far={r0['far']:.4f}")
    if mlp:
        m0 = evaluate(mlp, val_dl, device)
        print(f"Init MLP  AUROC={m0['auroc']:.4f}  "
              f"close={m0['close']:.4f}  far={m0['far']:.4f}")

    header = (f"\n{'Ep':>3}  {'RLoss':>7} {'RopeAUC':>8} {'Sep':>6}")
    if mlp: header += f"  {'MLoss':>7} {'MlpAUC':>7} {'Sep':>6}  {'ΔAUC':>6}"
    print(header)
    print("─" * (len(header) - 1))

    best_rope = {"auroc": 0., "epoch": 0, "sep": 0.}
    best_mlp  = {"auroc": 0., "epoch": 0}
    history   = []

    # ── Training loop ─────────────────────────────────────────────────────
    for ep in range(1, args.epochs + 1):
        t0 = time.time()

        rs = train_rope_epoch(rope, train_dl, rope_opt, device,
                              args.temperature, args.ema_decay)
        rope_sch.step()
        rv = evaluate(rope, val_dl, device)

        if rv["auroc"] > best_rope["auroc"]:
            best_rope = {"auroc": rv["auroc"], "epoch": ep, "sep": rv["sep"]}
            torch.save({
                "head":  rope.state_dict(),
                "epoch": ep,
                "auroc": rv["auroc"],
                "config": {"embed_dim": args.embed_dim, "k_min": args.k_min,
                           "k_max": args.k_max},
            }, out_dir / "rope_head_best.pt")

        row = (f"{ep:3d}  {rs['loss']:7.4f} {rv['auroc']:8.4f} {rv['sep']:+6.4f}")

        if mlp and mlp_opt and mlp_sch:
            ms = train_mlp_epoch(mlp, train_dl, mlp_opt, device, args.temperature)
            mlp_sch.step()
            mv = evaluate(mlp, val_dl, device)
            if mv["auroc"] > best_mlp["auroc"]:
                best_mlp = {"auroc": mv["auroc"], "epoch": ep}
            row += (f"  {ms['loss']:7.4f} {mv['auroc']:7.4f} {mv['sep']:+6.4f}  "
                    f"{rv['auroc']-mv['auroc']:+6.4f}")
        else:
            mv = None

        print(row + f"  ({time.time()-t0:.1f}s)")
        history.append({"ep": ep, "rope": rv, "mlp": mv})

    # Save final checkpoint
    torch.save({
        "head": rope.state_dict(), "epoch": args.epochs,
        "auroc": evaluate(rope, val_dl, device)["auroc"],
        "config": {"embed_dim": args.embed_dim},
    }, out_dir / "rope_head_final.pt")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'═'*50}")
    print(f"RESULTS — {args.epochs} epochs")
    print(f"{'═'*50}")
    fv = evaluate(rope, val_dl, device)
    print(f"RoPE  AUROC={fv['auroc']:.4f}  "
          f"close={fv['close']:.4f}  far={fv['far']:.4f}  sep={fv['sep']:+.4f}")
    print(f"Best  AUROC={best_rope['auroc']:.4f}  (epoch {best_rope['epoch']})")
    if mlp:
        mv_final = evaluate(mlp, val_dl, device)
        print(f"MLP   AUROC={mv_final['auroc']:.4f}  "
              f"close={mv_final['close']:.4f}  far={mv_final['far']:.4f}")
        print(f"Δ AUROC = {fv['auroc']-mv_final['auroc']:+.4f}")

    target = 0.70
    status = "✅ TARGET MET" if best_rope["auroc"] >= target else \
             f"⚠️  below {target} — try more epochs or real encoder checkpoint"
    print(f"\n{status}")
    print(f"\nCheckpoints: {out_dir}/rope_head_best.pt")

    # Save history
    hist_path = out_dir / "training_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2, default=str)


if __name__ == "__main__":
    main()
