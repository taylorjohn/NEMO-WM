"""
train_phase3_anomaly.py — CORTEX-PE Phase 3E: Prediction-Error Anomaly Signal

Replaces the static −‖z‖ anomaly score with a richer prediction-error signal:
    a(x_t) = ‖z_predicted − z_observed‖

Healthy signals have small prediction error (normal dynamics are predictable).
Fault conditions produce large prediction error (dynamics become chaotic).

For bearing fault detection, this is superior to norm-based anomaly because:
  - Norm collapse: all latents converge toward similar magnitudes over time
  - Prediction error: exploits the predictor's learned dynamics model
  - Expected AUROC improvement: 0.9929 → potentially 0.99+

Architecture:
    Train a simple 1-step dynamics predictor on normal-only frames:
        z_predictor: (z_t, delta_t) → z_t+1_hat
    At inference, anomaly score = ‖z_t+1_hat − z_t+1_actual‖

Based on:
    "World Models for Anomaly Detection during Model-Based RL Inference"
    arXiv:2503.02552

Usage:
    # Bearing (CWRU)
    python train_phase3_anomaly.py \
        --encoder ./checkpoints/bearing/cortex_student_phase2_final.pt \
        --data ./phase2_bearing_frames \
        --domain bearing \
        --out ./checkpoints/phase3_bearing

    # MIMII fan
    python train_phase3_anomaly.py \
        --encoder ./checkpoints/mimii_fan/cortex_student_phase1_final.pt \
        --data ./mimii_frames/fan/id_00/normal \
        --domain mimii_fan \
        --out ./checkpoints/phase3_mimii_fan

    # Evaluate
    python train_phase3_anomaly.py --eval-only \
        --encoder ./checkpoints/bearing/... \
        --pred-ckpt ./checkpoints/phase3_bearing/dynamics_predictor.pt \
        --normal-dir ./phase2_bearing_frames \
        --fault-dir ./bearing_data/cwru_fault_frames
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
from PIL import Image

from student_encoder import StudentEncoder

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--encoder',   default='./checkpoints/bearing/cortex_student_phase2_final.pt')
parser.add_argument('--data',      default='./phase2_bearing_frames', help='Normal frames directory')
parser.add_argument('--domain',    default='bearing')
parser.add_argument('--out',       default='./checkpoints/phase3_bearing')
parser.add_argument('--epochs',    type=int,   default=20)
parser.add_argument('--batch',     type=int,   default=128)
parser.add_argument('--lr',        type=float, default=3e-4)
parser.add_argument('--window',    type=int,   default=2, help='Steps ahead to predict')
# Evaluation mode
parser.add_argument('--eval-only',   action='store_true')
parser.add_argument('--pred-ckpt',   default=None)
parser.add_argument('--normal-dir',  default=None)
parser.add_argument('--fault-dir',   default=None)
args = parser.parse_args()

Path(args.out).mkdir(parents=True, exist_ok=True)

TRANSFORM = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Sequential frame dataset ──────────────────────────────────────────────────
class SequentialFrameDataset(Dataset):
    """Load consecutive frame pairs (z_t, z_t+window) from sorted PNGs."""
    def __init__(self, frame_dir, window=2, transform=None):
        paths = sorted(
            list(Path(frame_dir).glob('*.png')) +
            list(Path(frame_dir).glob('*.jpg')),
            key=lambda p: p.name
        )
        self.pairs     = [(paths[i], paths[i + window])
                          for i in range(len(paths) - window)]
        self.transform = transform or TRANSFORM
        if len(self.pairs) == 0:
            raise ValueError(f"No frame pairs found in {frame_dir}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p0, p1     = self.pairs[idx]
        img0       = Image.open(p0).convert('RGB')
        img1       = Image.open(p1).convert('RGB')
        return self.transform(img0), self.transform(img1)


class SingleFrameDataset(Dataset):
    """Load individual frames for anomaly scoring."""
    def __init__(self, frame_dir, transform=None):
        self.paths     = sorted(
            list(Path(frame_dir).glob('*.png')) +
            list(Path(frame_dir).glob('*.jpg')),
            key=lambda p: p.name
        )
        self.transform = transform or TRANSFORM

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.paths[idx]).convert('RGB'))


# ── Dynamics predictor ────────────────────────────────────────────────────────
class DynamicsPredictor(nn.Module):
    """
    Predicts z_t+window from z_t.
    Trained on normal-only frames — learns healthy dynamics.
    Large prediction error at inference = anomaly.
    ~32K parameters, NPU-safe.
    """
    def __init__(self, z_dim=128, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, z_dim),
        )

    def forward(self, z):
        return self.net(z)


# ── Load encoder ──────────────────────────────────────────────────────────────
def load_encoder(path):
    ckpt    = torch.load(path, map_location='cpu')
    encoder = StudentEncoder()
    state   = ckpt.get('model', ckpt)
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
    return encoder


# ── Training ──────────────────────────────────────────────────────────────────
def train():
    print(f"\n{'='*56}")
    print(f"  PHASE 3E — PREDICTION-ERROR ANOMALY TRAINING")
    print(f"  Domain:   {args.domain}")
    print(f"  Data:     {args.data}")
    print(f"  Window:   {args.window} frames")
    print(f"  Epochs:   {args.epochs}")
    print(f"{'='*56}\n")

    encoder   = load_encoder(args.encoder)
    predictor = DynamicsPredictor(z_dim=128, hidden=256)
    print(f"  Encoder:   {sum(p.numel() for p in encoder.parameters()):,} params (frozen)")
    print(f"  Predictor: {sum(p.numel() for p in predictor.parameters()):,} params")

    dataset = SequentialFrameDataset(args.data, window=args.window)
    n_val   = max(1, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)
    print(f"  Pairs — train: {n_train} | val: {n_val}")

    optimizer = torch.optim.AdamW(predictor.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val = float('inf')
    results  = []

    for epoch in range(1, args.epochs + 1):
        predictor.train()
        train_loss = 0.0
        t0 = time.time()
        for img0, img1 in train_loader:
            with torch.no_grad():
                z0 = encoder(img0)
                z1 = encoder(img1)
            z1_hat  = predictor(z0)
            loss    = F.mse_loss(z1_hat, z1.detach())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        predictor.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img0, img1 in val_loader:
                z0     = encoder(img0)
                z1     = encoder(img1)
                z1_hat = predictor(z0)
                val_loss += F.mse_loss(z1_hat, z1).item()
        val_loss /= len(val_loader)
        scheduler.step()

        print(f"Epoch {epoch:>3}/{args.epochs} | "
              f"train={train_loss:.5f}  val={val_loss:.5f}  ({time.time()-t0:.1f}s)")

        results.append({'epoch': epoch, 'train': round(train_loss, 6), 'val': round(val_loss, 6)})

        if val_loss < best_val:
            best_val = val_loss
            torch.save(predictor.state_dict(), f'{args.out}/dynamics_predictor_best.pt')

    torch.save(predictor.state_dict(), f'{args.out}/dynamics_predictor_final.pt')
    json.dump(results, open(f'{args.out}/training_results.json', 'w'), indent=2)
    print(f"\nSaved → {args.out}/dynamics_predictor_best.pt")
    print(f"Run evaluation with --eval-only to compute AUROC improvement.")


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate():
    """
    Compare two anomaly signals:
      Signal A (baseline): −‖z‖  (Phase 1/2 norm anomaly)
      Signal B (Phase 3E): ‖z_t+1_hat − z_t+1‖  (prediction error)
    """
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        raise ImportError("pip install scikit-learn")

    encoder   = load_encoder(args.encoder)
    predictor = DynamicsPredictor(z_dim=128, hidden=256)
    predictor.load_state_dict(torch.load(args.pred_ckpt, map_location='cpu'))
    predictor.eval()

    def score_directory(frame_dir, label):
        ds     = SequentialFrameDataset(frame_dir, window=args.window)
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
        norm_scores = []
        pred_scores = []
        with torch.no_grad():
            for img0, img1 in loader:
                z0     = encoder(img0)
                z1     = encoder(img1)
                z1_hat = predictor(z0)
                # Signal A: negative norm of current frame
                norm_scores.extend((-torch.norm(z0, dim=-1)).tolist())
                # Signal B: prediction error (higher = more anomalous)
                pred_scores.extend(torch.norm(z1_hat - z1, dim=-1).tolist())
        n = len(norm_scores)
        labels_arr = [label] * n
        return norm_scores, pred_scores, labels_arr

    print(f"\nScoring normal frames: {args.normal_dir}")
    n_norm, p_norm, l_norm = score_directory(args.normal_dir, 0)
    print(f"  {len(n_norm)} frames")

    print(f"Scoring fault frames:  {args.fault_dir}")
    n_fault, p_fault, l_fault = score_directory(args.fault_dir, 1)
    print(f"  {len(n_fault)} frames")

    all_norm  = n_norm  + n_fault
    all_pred  = p_norm  + p_fault
    all_labels = l_norm + l_fault

    # For norm: higher norm = more anomalous (already negative, flip back)
    auroc_norm = roc_auc_score(all_labels, [-x for x in all_norm])
    auroc_pred = roc_auc_score(all_labels, all_pred)

    improvement = auroc_pred - auroc_norm

    print(f"\n{'='*56}")
    print(f"  PHASE 3E EVALUATION — {args.domain}")
    print(f"  Signal A (−‖z‖ baseline):   AUROC = {auroc_norm:.4f}")
    print(f"  Signal B (prediction error): AUROC = {auroc_pred:.4f}")
    print(f"  Improvement: {improvement:+.4f}")
    print(f"{'='*56}\n")

    json.dump({
        'domain': args.domain,
        'auroc_norm_baseline': round(auroc_norm, 4),
        'auroc_prediction_error': round(auroc_pred, 4),
        'improvement': round(improvement, 4),
        'n_normal': len(n_norm),
        'n_fault': len(n_fault),
    }, open(f'{args.out}/eval_results.json', 'w'), indent=2)

    return auroc_pred


# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if args.eval_only:
        evaluate()
    else:
        train()
