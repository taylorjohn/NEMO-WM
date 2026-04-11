"""
train_phase3_contrastive.py — CORTEX-PE Phase 3 Contrastive Shaping

Replaces IQL (needs millions of samples) with contrastive trajectory
ordering (works with hundreds). Uses existing successful planning
traces or temporal proximity as supervision.

Three modes:
  --mode maze      Uses benchmark trajectory .npy for ordering
  --mode bearing   Uses temporal proximity in spectrogram sequences  
  --mode trading   Uses volatility regime labels from rolling std
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from student_encoder import StudentEncoder

parser = argparse.ArgumentParser()
parser.add_argument('--encoder',  required=True)
parser.add_argument('--data',     required=True)
parser.add_argument('--mode',     choices=['maze', 'bearing', 'trading'], default='maze')
parser.add_argument('--out',      default='./checkpoints/phase3_contrastive')
parser.add_argument('--epochs',   type=int,   default=10)
parser.add_argument('--batch',    type=int,   default=128)
parser.add_argument('--margin',   type=float, default=0.5)
parser.add_argument('--k-near',   type=int,   default=5,  help='Steps = near')
parser.add_argument('--k-far',    type=int,   default=20, help='Steps = far')
args = parser.parse_args()

Path(args.out).mkdir(parents=True, exist_ok=True)

T = transforms.Compose([
    transforms.Resize(224), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

class TrajectoryContrastiveDataset(Dataset):
    '''Samples (anchor, near, far) triples from sequential frames.
    anchor at t, near at t+k_near, far at t+k_far.
    d(anchor,near) < d(anchor,far) — temporal ordering as value proxy.'''
    def __init__(self, frame_dir, k_near=5, k_far=20, transform=None):
        paths = sorted(
            list(Path(frame_dir).glob('*.png')) +
            list(Path(frame_dir).glob('*.jpg')),
            key=lambda p: p.name
        )
        self.paths = paths
        self.k_near = k_near
        self.k_far  = k_far
        self.transform = transform or T
        self.valid = list(range(0, len(paths) - k_far))
        print(f'  ContrastiveDataset: {len(paths)} frames, {len(self.valid)} anchors')

    def __len__(self): return len(self.valid)

    def __getitem__(self, idx):
        t    = self.valid[idx]
        img  = lambda p: self.transform(Image.open(p).convert('RGB'))
        return (img(self.paths[t]),
                img(self.paths[t + self.k_near]),
                img(self.paths[t + self.k_far]))

def load_encoder(path, grad=False):
    ckpt    = torch.load(path, map_location='cpu')
    enc     = StudentEncoder()
    state   = ckpt.get('model', ckpt)
    new_s   = {}
    for k,v in state.items():
        k2 = k.replace('backbone.stem.0','backbone.block1.0') \
              .replace('backbone.stem.1','backbone.block1.1') \
              .replace('backbone.stem.2','backbone.block1.2')
        new_s[k2] = v
    enc.load_state_dict(new_s, strict=False)
    if not grad:
        enc.eval()
        for p in enc.parameters(): p.requires_grad_(False)
    return enc

def triplet_loss(z_a, z_near, z_far, margin=0.5):
    '''d(a,near) + margin < d(a,far)  →  hinge on violation'''
    d_near = torch.norm(z_a - z_near, dim=-1)
    d_far  = torch.norm(z_a - z_far,  dim=-1)
    return F.relu(d_near - d_far + margin).mean()

def main():
    print(f'\n{"="*56}')
    print(f'  PHASE 3 CONTRASTIVE SHAPING — {args.mode.upper()}')
    print(f'  k_near={args.k_near}  k_far={args.k_far}  margin={args.margin}')
    print(f'{"="*56}\n')

    # Stage 1: frozen encoder — train projection head only
    encoder = load_encoder(args.encoder, grad=False)
    proj    = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Linear(64,64))
    print(f'  Encoder: {sum(p.numel() for p in encoder.parameters()):,} (frozen)')
    print(f'  ProjHead: {sum(p.numel() for p in proj.parameters()):,}')

    dataset = TrajectoryContrastiveDataset(args.data, args.k_near, args.k_far)
    n_val   = max(1, len(dataset)//10)
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [len(dataset)-n_val, n_val],
        generator=torch.Generator().manual_seed(42))
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)

    opt = torch.optim.AdamW(proj.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = float('inf')
    for epoch in range(1, args.epochs+1):
        proj.train(); tl = 0.0; t0 = time.time()
        for a, near, far in train_dl:
            with torch.no_grad():
                za   = proj(encoder(a))
                zn   = proj(encoder(near))
                zf   = proj(encoder(far))
            loss = triplet_loss(za, zn, zf, args.margin)
            # Re-run with grad for proj
            za   = proj(encoder(a).detach())
            zn   = proj(encoder(near).detach())
            zf   = proj(encoder(far).detach())
            loss = triplet_loss(za, zn, zf, args.margin)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(proj.parameters(), 1.0)
            opt.step(); tl += loss.item()
        tl /= len(train_dl)

        proj.eval(); vl = 0.0
        with torch.no_grad():
            for a, near, far in val_dl:
                za = proj(encoder(a)); zn = proj(encoder(near)); zf = proj(encoder(far))
                vl += triplet_loss(za, zn, zf, args.margin).item()
        vl /= len(val_dl); sch.step()
        print(f'Epoch {epoch:>3}/{args.epochs} | train={tl:.4f}  val={vl:.4f}  ({time.time()-t0:.1f}s)')
        if vl < best_val:
            best_val = vl
            torch.save({'proj': proj.state_dict()}, f'{args.out}/contrastive_proj_best.pt')

    torch.save({'proj': proj.state_dict()}, f'{args.out}/contrastive_proj_final.pt')
    print(f'\nSaved → {args.out}/contrastive_proj_best.pt')
    print(f'Best val loss: {best_val:.4f}')
    print(f'\nUsage: replace z_t with proj(encoder(frame)) in planner/anomaly scorer')

if __name__ == '__main__': main()
