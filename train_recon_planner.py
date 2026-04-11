"""
train_recon_planner.py
=======================
RECON outdoor navigation planner training from raw HDF5 trajectory files.

Data format: RECON dataset (Shah et al. 2021)
    recon_data/recon_release/jackal_DATE_SEQ_rROLL.hdf5
    Each file: one trajectory with keys:
        observations/image   (T, H, W, 3) uint8 RGB
        observations/position (T, 2)  x,y in metres
        actions              (T, 2)  (linear_vel, angular_vel)

Phase 3C result: ego-motion dynamics provide real learning signal.
This trains the quasimetric predictor on the actual HDF5 files.

Usage:
    # Inspect one HDF5 file to confirm keys:
    python train_recon_planner.py --inspect `
        --data ./recon_data/recon_release

    # Print ENV_CONFIG patch for train_predictor.py:
    python train_recon_planner.py --patch-env-config

    # Train:
    python train_recon_planner.py --train `
        --encoder ./checkpoints/maze_weak_sigreg_straight/cortex_student_phase2_final.pt `
        --data ./recon_data/recon_release `
        --steps 3000 `
        --out ./checkpoints/recon_planner
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


# ── ENV_CONFIG patch ───────────────────────────────────────────────────────

RECON_ENV_CONFIG = {
    "recon": {
        "n_train_files": 11836,
        "steps": 3000,
        "epochs": 3,
        "action_dim": 2,
        "latent_dim": 32,
        "planner": "mezo",
        "horizon": 8,
        "n_candidates": 64,
        "quasimetric": True,
        "data_format": "hdf5",
        "notes": "Jackal outdoor navigation. HDF5 files. Phase 3C confirmed learning signal."
    }
}

ENV_CONFIG_PATCH = '''
# Add to ENV_CONFIG in train_predictor.py:
"recon": {
    "n_train_files": 11836,        # jackal_*.hdf5 files
    "steps": 3000,
    "epochs": 3,
    "action_dim": 2,               # (linear_vel, angular_vel)
    "latent_dim": 32,
    "planner": "mezo",
    "horizon": 8,                  # 2s at 4Hz
    "n_candidates": 64,
    "quasimetric": True,
    "data_format": "hdf5",
},
'''


# ── Image transform ────────────────────────────────────────────────────────

RECON_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── HDF5 RECON Dataset ─────────────────────────────────────────────────────

class RECONHdf5Dataset(torch.utils.data.Dataset):
    """
    RECON dataset from raw HDF5 files.

    Each HDF5 file is one trajectory. We sample (frame_t, action_t, frame_t1)
    triplets for predictor training.

    Keys read:
        observations/image    (T, H, W, 3) uint8
        actions               (T, 2) float32
    """

    def __init__(
        self,
        data_dir: str,
        encoder: nn.Module,
        device: torch.device,
        max_files: int = 500,       # cap for fast experimentation
        triplets_per_file: int = 8,  # random triplets sampled per trajectory
        transform=None,
    ):
        import h5py
        self.h5py = h5py
        self.data_dir = Path(data_dir)
        self.encoder = encoder
        self.device = device
        self.transform = transform or RECON_TRANSFORM
        self.triplets = []

        # Find all HDF5 files
        all_files = sorted(self.data_dir.glob("*.hdf5"))
        if not all_files:
            all_files = sorted(self.data_dir.glob("**/*.hdf5"))

        if not all_files:
            raise RuntimeError(f"No .hdf5 files found in {data_dir}")

        # Subsample for speed
        random.seed(42)
        files = random.sample(all_files, min(max_files, len(all_files)))
        print(f"Loading RECON: {len(files)}/{len(all_files)} HDF5 files "
              f"× {triplets_per_file} triplets...")

        self.encoder.eval()
        n_loaded = 0
        with torch.no_grad():
            for fpath in files:
                try:
                    triplets = self._load_file(str(fpath), triplets_per_file)
                    self.triplets.extend(triplets)
                    n_loaded += 1
                except Exception as e:
                    continue  # skip malformed files silently

        print(f"  Loaded {n_loaded} files → {len(self.triplets)} triplets")

    def _decode_jpeg(self, raw_bytes) -> np.ndarray:
        """Decode JPEG bytes (stored as HDF5 variable-length string) to RGB array."""
        from io import BytesIO
        from PIL import Image as PILImage
        # HDF5 stores as bytes or numpy bytes — handle both
        if isinstance(raw_bytes, (bytes, np.bytes_)):
            data = bytes(raw_bytes)
        else:
            data = bytes(raw_bytes.tobytes())
        img = PILImage.open(BytesIO(data)).convert("RGB")
        return np.array(img)

    def _decode_jpeg(self, raw_bytes) -> "np.ndarray":
        from io import BytesIO
        from PIL import Image as PILImage
        data = bytes(raw_bytes) if isinstance(raw_bytes, (bytes, __import__("numpy").bytes_)) else bytes(raw_bytes.tobytes())
        return __import__("numpy").array(PILImage.open(BytesIO(data)).convert("RGB"))

    def _load_file(self, fpath: str, n_triplets: int) -> list:
        """
        Load triplets from RECON HDF5 file.
        Confirmed key layout:
            images/rgb_left              (T,) JPEG bytes
            commands/linear_velocity     (T,) float64
            commands/angular_velocity    (T,) float64
        """
        import h5py, random as _random
        triplets = []
        try:
            with h5py.File(fpath, "r") as f:
                # Images
                if "images" in f and "rgb_left" in f["images"]:
                    raw_imgs = f["images"]["rgb_left"][:]
                    use_jpeg = True
                elif "observations" in f and "image" in f["observations"]:
                    raw_imgs = f["observations"]["image"][:]
                    use_jpeg = False
                else:
                    return []

                # Actions
                if "commands" in f and "linear_velocity" in f["commands"]:
                    lv = f["commands"]["linear_velocity"][:].astype("float32")
                    av = f["commands"]["angular_velocity"][:].astype("float32")
                    actions = __import__("numpy").stack([lv, av], axis=-1)
                elif "actions" in f:
                    actions = f["actions"][:].astype("float32")
                elif "jackal" in f and "position" in f["jackal"]:
                    pos = f["jackal"]["position"][:, :2].astype("float32")
                    actions = __import__("numpy").diff(pos, axis=0)
                    raw_imgs = raw_imgs[:-1]
                else:
                    return []

                T = min(len(raw_imgs), len(actions))
                if T < 2:
                    return []

                raw_imgs = raw_imgs[:T]
                actions  = (__import__("numpy").clip(actions[:T], -2.0, 2.0) / 2.0).astype("float32")

                indices = _random.sample(range(T - 1), min(n_triplets, T - 1))
                import torch
                for t in indices:
                    try:
                        arr_t  = self._decode_jpeg(raw_imgs[t])   if use_jpeg else raw_imgs[t]
                        arr_t1 = self._decode_jpeg(raw_imgs[t+1]) if use_jpeg else raw_imgs[t+1]
                        img_t  = self.transform(arr_t).unsqueeze(0).to(self.device)
                        img_t1 = self.transform(arr_t1).unsqueeze(0).to(self.device)
                        z_t  = self.encoder(img_t).squeeze(0).cpu()
                        z_t1 = self.encoder(img_t1).squeeze(0).cpu()
                        act  = torch.from_numpy(actions[t].copy())
                        triplets.append((z_t, act, z_t1))
                    except Exception:
                        continue
        except Exception:
            return []
        return triplets

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int):
        return self.triplets[idx]


# ── HDF5 inspection ────────────────────────────────────────────────────────

def inspect_hdf5(data_dir: str, n_files: int = 3):
    """Print the key structure of the first N HDF5 files."""
    try:
        import h5py
    except ImportError:
        print("pip install h5py")
        return

    files = sorted(Path(data_dir).glob("*.hdf5"))[:n_files]
    if not files:
        files = sorted(Path(data_dir).glob("**/*.hdf5"))[:n_files]

    for fpath in files:
        print(f"\n── {fpath.name} ──")
        with h5py.File(str(fpath), 'r') as f:
            def _print_keys(obj, prefix=""):
                for key in obj.keys():
                    item = obj[key]
                    if hasattr(item, 'shape'):
                        print(f"  {prefix}{key}: {item.shape} {item.dtype}")
                    else:
                        print(f"  {prefix}{key}/")
                        _print_keys(item, prefix + "  ")
            _print_keys(f)


# ── Quasimetric predictor ──────────────────────────────────────────────────

class QuasimetricPredictor(nn.Module):
    def __init__(self, latent_dim: int = 32, action_dim: int = 2):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 128),
            nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, latent_dim),
        )
        self.dist_head = nn.Sequential(
            nn.Linear(latent_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, action], dim=-1)
        return F.normalize(z + self.predictor(x), dim=-1)

    def quasimetric(self, z_s: torch.Tensor, z_g: torch.Tensor) -> torch.Tensor:
        return self.dist_head(torch.cat([z_s, z_g], dim=-1))

    def rollout(self, z0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        zs = [z0.unsqueeze(0)]
        z = z0.unsqueeze(0)
        for t in range(len(actions)):
            z = self(z, actions[t].unsqueeze(0))
            zs.append(z)
        return torch.cat(zs, dim=0)


# ── Training ───────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Encoder stub (loads actual weights from checkpoint)
    class EncoderStub(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.GELU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),
                nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.GELU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(256, 32),
            )
        def forward(self, x):
            return F.normalize(self.net(x), dim=-1)

    encoder = EncoderStub().to(device)
    ckpt = torch.load(args.encoder, map_location=device)
    state = ckpt.get("student") or ckpt
    try:
        encoder.load_state_dict(state, strict=False)
        print(f"✅ Encoder loaded from {args.encoder}")
    except Exception as e:
        print(f"⚠️  Partial encoder load ({e})")
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Dataset
    dataset = RECONHdf5Dataset(
        args.data, encoder, device,
        max_files=args.max_files,
        triplets_per_file=args.triplets_per_file,
    )
    if len(dataset) == 0:
        raise RuntimeError("No triplets loaded — check HDF5 keys with --inspect")

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    # Predictor
    predictor = QuasimetricPredictor(latent_dim=32, action_dim=2).to(device)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.1
    )

    print(f"\n{'='*60}")
    print(f"  RECON QUASIMETRIC PLANNER")
    print(f"  Triplets: {len(dataset)}  |  Steps: {args.steps}")
    print(f"  Loss: L_pred(cosine) + 0.5*L_qm + 0.1*L_triangle")
    print(f"{'='*60}\n")

    step = 0
    log = []

    while step < args.steps:
        for z_t, action, z_t1 in loader:
            if step >= args.steps:
                break

            z_t    = z_t.to(device)
            action = action.to(device)
            z_t1   = z_t1.to(device)

            # Prediction loss
            z_pred = predictor(z_t, action)
            l_pred = (1.0 - F.cosine_similarity(z_pred, z_t1, dim=-1)).mean()

            # Quasimetric: consecutive should be close, reverse should be larger
            d_fwd = predictor.quasimetric(z_t, z_t1)
            d_bwd = predictor.quasimetric(z_t1, z_t)
            l_qm = F.relu(d_fwd - d_bwd + 0.1).mean()

            # Triangle inequality
            idx = torch.randperm(len(z_t), device=device)
            z_m = z_t[idx]
            l_tri = F.relu(
                predictor.quasimetric(z_t, z_t1) -
                predictor.quasimetric(z_t, z_m) -
                predictor.quasimetric(z_m, z_t1)
            ).mean()

            loss = l_pred + 0.5 * l_qm + 0.1 * l_tri

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % 100 == 0:
                entry = {
                    "step": step,
                    "total": loss.item(),
                    "pred": l_pred.item(),
                    "qm": l_qm.item(),
                }
                log.append(entry)
                print(f"Step {step:4d}/{args.steps} | "
                      f"total={loss.item():.4f}  "
                      f"pred={l_pred.item():.4f}  "
                      f"qm={l_qm.item():.4f}")

            if step % 1000 == 0:
                torch.save({
                    "step": step,
                    "predictor": predictor.state_dict(),
                }, out_dir / f"predictor_step{step:05d}.pt")
                print(f"💾 Checkpoint saved (step {step})")

    torch.save({
        "step": step,
        "predictor": predictor.state_dict(),
    }, out_dir / "predictor_final.pt")

    with open(out_dir / "train_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n✅ RECON planner trained: {out_dir}/predictor_final.pt")


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RECON HDF5 quasimetric planner")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--inspect", action="store_true",
                        help="Print HDF5 key structure of first 3 files")
    parser.add_argument("--patch-env-config", action="store_true")
    parser.add_argument("--encoder", default=None)
    parser.add_argument("--data", default="./recon_data/recon_release")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-files", type=int, default=500,
                        help="HDF5 files to load (500 = ~4000 triplets, fast)")
    parser.add_argument("--triplets-per-file", type=int, default=8)
    parser.add_argument("--out", default="./checkpoints/recon_planner")
    args = parser.parse_args()

    if args.patch_env_config:
        print(ENV_CONFIG_PATCH)
        print(json.dumps(RECON_ENV_CONFIG, indent=2))
        return

    if args.inspect:
        inspect_hdf5(args.data)
        return

    if args.train:
        if not args.encoder:
            parser.error("--encoder required for --train")
        train(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
