"""
train_phase3_quasimetric.py — CORTEX-PE Phase 3C: Quasimetric Latent Distance

Trains an asymmetric distance head on top of the frozen RECON encoder.
Standard Euclidean distance is symmetric: d(A→B) = d(B→A).
For outdoor robot navigation, A→B and B→A may have very different costs
due to terrain, obstacles, or one-way paths.

A quasimetric encodes directionality: d(A→B) ≠ d(B→A)
This gives the MeZO planner a more accurate reachability estimate.

Architecture:
    Frozen RECON encoder → z_t, z_goal  (128-D each)
    QuasimetricHead(z_t, z_goal) → d(t → goal)  (scalar, asymmetric)
    
    Asymmetry achieved by: head(z_t, z_goal) ≠ head(z_goal, z_t)
    i.e., the head is NOT a symmetric function of its inputs.

Loss:
    Contrastive quasimetric loss (Wang et al. ICML 2023):
    For each trajectory (s0, s1, ..., sT):
        Positive pairs: (s_t, s_t+k) for k > 0 (s_t+k is reachable from s_t)
        d(s_t, s_t+k) should be small (~ k steps)
        
    Triangle inequality regularisation:
        d(s_t, s_goal) ≤ d(s_t, s_mid) + d(s_mid, s_goal)

Based on:
    "Optimal Goal-Reaching RL via Quasimetric Learning"
    Wang, Torralba, Isola, Zhang — ICML 2023

Usage:
    python train_phase3_quasimetric.py \
        --encoder ./checkpoints/recon/cortex_student_flow_final.pt \
        --frames ./phase2_recon_frames \
        --out ./checkpoints/phase3_recon_quasi
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
parser.add_argument('--encoder', default='./checkpoints/recon/cortex_student_flow_final.pt')
parser.add_argument('--frames',  default='./phase2_recon_frames')
parser.add_argument('--out',     default='./checkpoints/phase3_recon_quasi')
parser.add_argument('--epochs',  type=int,   default=15)
parser.add_argument('--batch',   type=int,   default=128)
parser.add_argument('--lr',      type=float, default=3e-4)
parser.add_argument('--k-max',   type=int,   default=20, help='Max steps for positive pairs')
parser.add_argument('--lambda-tri', type=float, default=0.1, help='Triangle inequality weight')
args = parser.parse_args()

Path(args.out).mkdir(parents=True, exist_ok=True)

TRANSFORM = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Dataset: trajectory triples for quasimetric training ─────────────────────
class QuasimetricDataset(Dataset):
    """
    Samples (anchor, positive, negative) triples from sequential frames.
    
    anchor   = frame at time t
    positive = frame at time t+k (k ~ U[1, k_max]) — reachable, forward in time
    negative = frame at time t-k (backwards — hard negative, unreachable via forward path)
    
    For true quasimetric: d(anchor→positive) << d(anchor→negative)
    And: d(anchor→positive) ≈ k / k_max  (normalised step count)
    """
    def __init__(self, frame_dir, k_max=20, transform=None):
        paths = sorted(
            list(Path(frame_dir).glob('*.png')) +
            list(Path(frame_dir).glob('*.jpg')),
            key=lambda p: p.name
        )
        self.paths     = paths
        self.k_max     = k_max
        self.transform = transform or TRANSFORM
        # Only use frames that have both a forward and backward neighbour
        self.valid_idx = list(range(k_max, len(paths) - k_max))
        print(f"  QuasimetricDataset: {len(paths)} frames, {len(self.valid_idx)} valid anchors")

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, idx):
        t      = self.valid_idx[idx]
        k      = np.random.randint(1, self.k_max + 1)
        t_pos  = t + k      # Forward — reachable
        t_neg  = t - k      # Backward — not reachable via forward motion

        anchor  = self.transform(Image.open(self.paths[t]).convert('RGB'))
        pos     = self.transform(Image.open(self.paths[t_pos]).convert('RGB'))
        neg     = self.transform(Image.open(self.paths[t_neg]).convert('RGB'))
        # Normalised step distance as regression target
        dist_target = torch.tensor([k / self.k_max], dtype=torch.float32)

        return anchor, pos, neg, dist_target


# ── Quasimetric head ──────────────────────────────────────────────────────────
class QuasimetricHead(nn.Module):
    """
    Asymmetric distance function: d(z_from, z_to) → scalar ≥ 0
    
    NOT symmetric: d(z_a, z_b) ≠ d(z_b, z_a) by construction.
    Uses separate pathways for "from" and "to" representations before combining.
    ~20K params.
    """
    def __init__(self, z_dim=128, hidden=128):
        super().__init__()
        # Separate encoding of source and target directions
        self.from_enc = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU()
        )
        self.to_enc = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU()
        )
        # Asymmetric combination
        self.combine = nn.Sequential(
            nn.Linear(hidden * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus(),  # Ensures non-negative distance
        )

    def forward(self, z_from, z_to):
        h_from = self.from_enc(z_from)
        h_to   = self.to_enc(z_to)
        return self.combine(torch.cat([h_from, h_to], dim=-1))


# ── Quasimetric loss ──────────────────────────────────────────────────────────
def quasimetric_loss(head, z_anchor, z_pos, z_neg, dist_target, lambda_tri=0.1):
    """
    Three-part quasimetric loss:
    
    1. Regression: d(anchor→pos) ≈ dist_target  (Huber loss)
    2. Contrastive: d(anchor→pos) < d(anchor→neg)  (margin hinge)
    3. Triangle inequality: d(anchor→pos) ≤ d(anchor→mid) + d(mid→pos)
       Enforced via random interpolation midpoints in latent space.
    """
    d_pos = head(z_anchor, z_pos)   # Should be small (reachable, forward)
    d_neg = head(z_anchor, z_neg)   # Should be large (backward, unreachable)

    # 1. Regression loss on positive distance
    l_reg = F.huber_loss(d_pos, dist_target)

    # 2. Contrastive: margin loss pushes d_pos << d_neg
    margin = 0.3
    l_con  = F.relu(d_pos - d_neg + margin).mean()

    # 3. Triangle inequality regularisation
    # Midpoint between anchor and pos in latent space
    alpha  = torch.rand(z_anchor.shape[0], 1)
    z_mid  = alpha * z_anchor + (1 - alpha) * z_pos
    d_am   = head(z_anchor, z_mid)
    d_mp   = head(z_mid,    z_pos)
    # Penalise violation: d(a,p) > d(a,m) + d(m,p)
    l_tri  = F.relu(d_pos - d_am - d_mp).mean()

    total = l_reg + l_con + lambda_tri * l_tri
    return total, l_reg.item(), l_con.item(), l_tri.item()


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


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}")
    print(f"  PHASE 3C — QUASIMETRIC LATENT DISTANCE")
    print(f"  Encoder: {args.encoder}")
    print(f"  Frames:  {args.frames}")
    print(f"  k_max:   {args.k_max} steps")
    print(f"  λ_tri:   {args.lambda_tri}")
    print(f"  Epochs:  {args.epochs}")
    print(f"{'='*60}\n")

    encoder  = load_encoder(args.encoder)
    head     = QuasimetricHead(z_dim=128, hidden=128)
    print(f"  Encoder: {sum(p.numel() for p in encoder.parameters()):,} params (frozen)")
    print(f"  QHead:   {sum(p.numel() for p in head.parameters()):,} params")

    dataset = QuasimetricDataset(args.frames, k_max=args.k_max)
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float('inf')
    results  = []

    for epoch in range(1, args.epochs + 1):
        head.train()
        tl = tr = tc = tt = 0.0
        t0 = time.time()
        for anchor, pos, neg, dist_target in train_loader:
            with torch.no_grad():
                z_a = encoder(anchor)
                z_p = encoder(pos)
                z_n = encoder(neg)
            loss, lr, lc, lt = quasimetric_loss(
                head, z_a, z_p, z_n, dist_target, args.lambda_tri
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            tl += loss.item(); tr += lr; tc += lc; tt += lt
        n = len(train_loader)
        tl /= n; tr /= n; tc /= n; tt /= n

        head.eval()
        val_loss = 0.0
        with torch.no_grad():
            for anchor, pos, neg, dist_target in val_loader:
                z_a = encoder(anchor)
                z_p = encoder(pos)
                z_n = encoder(neg)
                l, _, _, _ = quasimetric_loss(
                    head, z_a, z_p, z_n, dist_target, args.lambda_tri
                )
                val_loss += l.item()
        val_loss /= len(val_loader)
        scheduler.step()

        print(f"Epoch {epoch:>3}/{args.epochs} | "
              f"total={tl:.4f}  reg={tr:.4f}  con={tc:.4f}  tri={tt:.4f}  "
              f"val={val_loss:.4f}  ({time.time()-t0:.1f}s)")

        results.append({'epoch': epoch, 'total': round(tl, 5), 'val': round(val_loss, 5)})

        if val_loss < best_val:
            best_val = val_loss
            torch.save(head.state_dict(), f'{args.out}/quasimetric_head_best.pt')

    torch.save(head.state_dict(), f'{args.out}/quasimetric_head_final.pt')
    json.dump(results, open(f'{args.out}/training_results.json', 'w'), indent=2)

    print(f"\n{'='*60}")
    print(f"  PHASE 3C COMPLETE")
    print(f"  QHead saved → {args.out}/quasimetric_head_best.pt")
    print(f"\n  Integration: replace ‖z_t − z_goal‖ in MeZO planner with:")
    print(f"    d = quasi_head(z_t, z_goal).item()")
    print(f"  This gives asymmetric reachability distances for RECON planning.")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
