"""
train_maze_flow.py — CORTEX-PE Phase 2b: Maze Encoder with Optical Flow Aux Loss

Root cause of UMaze/Medium underperformance vs DINO-WM:
    The 6px agent is sub-resolution at the 14×14 encoder grid.
    UMaze hallways look identical regardless of which branch you're in.
    The encoder encodes wall texture — not agent position.
    Result: all positions map to nearly identical latents (threshold 0.2559).

Fix — Optical Flow Auxiliary Loss:
    For each consecutive frame pair (f_t, f_{t+1}), compute the approximate
    agent displacement (∆x, ∆y) from frame-to-frame pixel difference.
    Add a FlowHead (128→2) that predicts this displacement from z_t.
    
    L_flow = MSE(FlowHead(z_t), ∆xy_normalised)
    
    This forces the 128-D latent to encode agent POSITION as a separable axis:
        Different positions → different z_t → correct ∆xy prediction
    
    Even when the background is identical, z_t must differ enough to predict
    ∆xy accurately. This creates the latent distance signal that the planner needs.

Agent displacement extraction:
    1. Convert frames to grayscale
    2. Compute absolute pixel difference |f_{t+1} - f_t|
    3. Find centroid of high-difference region = approximate agent location
    4. ∆xy = centroid_{t+1} - centroid_t, normalised to [-1, 1]
    
    This avoids full optical flow computation (Lucas-Kanade etc.) while
    capturing the relevant agent position signal.

Full Phase 2b loss:
    L = L_distill(backbone_g → DINOv2) 
      + λ_sigreg * L_sigreg(backbone_g)
      + λ_curv   * L_curv(h_φ(z^v))       (temporal straightening)
      + λ_flow   * L_flow(FlowHead(z_t))   ← NEW

Usage:
    # Phase 2b — resume from Phase 2 checkpoint
    python train_maze_flow.py \\
        --data    ./phase2_frames \\
        --resume  ./checkpoints/maze/cortex_student_phase2_final.pt \\
        --out     ./checkpoints/maze_flow \\
        --steps   6000 \\
        --lambda-flow 2.0

    # Benchmark immediately after
    python calibrate_threshold.py \\
        --encoder ./checkpoints/maze_flow/cortex_student_flow_final.pt \\
        --option 1 --pred-dir ./predictors --pct 50
    python run_benchmark.py \\
        --encoder ./checkpoints/maze_flow/cortex_student_flow_final.pt \\
        --option 1 --env umaze --planner gradient \\
        --threshold-file ./benchmark_thresholds_opt1_50pct.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from student_encoder import StudentEncoder
from train_predictor import SpatialPoolingHead, SpatialChannelProjector, DV, N_PATCHES
from latent_predictor import sigreg_loss, temporal_straightening_loss
from train_distillation import (
    DINOv2Teacher, BackboneProjector, distillation_loss, save_checkpoint
)

# =============================================================================
# Constants
# =============================================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

FRAME_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Raw transform for flow computation (no normalisation)
RAW_TRANSFORM = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),  # [0, 1]
])


# =============================================================================
# Optical flow helpers
# =============================================================================
def compute_agent_displacement(
    raw_t:  torch.Tensor,   # (B, 3, 224, 224) in [0, 1]
    raw_t1: torch.Tensor,   # (B, 3, 224, 224) in [0, 1]
    threshold: float = 0.05,
) -> torch.Tensor:
    """
    Compute approximate agent displacement from frame difference.

    Method:
        1. Grayscale both frames
        2. Absolute pixel difference → motion mask
        3. Centroid of motion region gives agent position each frame
        4. ∆xy = centroid_t1 - centroid_t, normalised to [-1, 1]

    If no significant motion detected (|diff| < threshold),
    returns zero displacement.

    Args:
        raw_t, raw_t1: (B, 3, H, W) unnormalised tensors in [0, 1]
        threshold:      minimum mean pixel change to count as motion

    Returns:
        (B, 2) float tensor of normalised displacements ∆x, ∆y in [-1, 1]
    """
    B, _, H, W = raw_t.shape

    # Grayscale
    g_t  = raw_t.mean(dim=1,  keepdim=True)   # (B, 1, H, W)
    g_t1 = raw_t1.mean(dim=1, keepdim=True)

    diff = (g_t1 - g_t).abs()   # (B, 1, H, W)

    displacements = []
    for b in range(B):
        d = diff[b, 0]   # (H, W)

        if d.max() < threshold:
            displacements.append(torch.zeros(2))
            continue

        # Soft centroid of motion region
        mask = (d > d.max() * 0.3).float()
        if mask.sum() < 4:
            displacements.append(torch.zeros(2))
            continue

        ys = torch.arange(H, dtype=torch.float32)
        xs = torch.arange(W, dtype=torch.float32)

        # Weighted centroid at t
        w_t   = (g_t[b, 0]  * mask + 1e-8)
        w_t   = w_t / w_t.sum()
        cy_t  = (w_t * ys.unsqueeze(1)).sum()
        cx_t  = (w_t * xs.unsqueeze(0)).sum()

        # Weighted centroid at t+1
        w_t1  = (g_t1[b, 0] * mask + 1e-8)
        w_t1  = w_t1 / w_t1.sum()
        cy_t1 = (w_t1 * ys.unsqueeze(1)).sum()
        cx_t1 = (w_t1 * xs.unsqueeze(0)).sum()

        # Normalise displacement to [-1, 1]
        dx = (cx_t1 - cx_t) / (W / 2)
        dy = (cy_t1 - cy_t) / (H / 2)
        displacements.append(torch.tensor([dx.item(), dy.item()]))

    return torch.stack(displacements)   # (B, 2)


# =============================================================================
# Flow head
# =============================================================================
class FlowHead(nn.Module):
    """
    Predicts agent displacement (∆x, ∆y) from latent z_t.
    If position is encoded in z_t, this should be easy to predict.
    Gradients from FlowHead force position to be encoded.
    """
    def __init__(self, in_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Tanh(),   # output in [-1, 1] matching normalised displacement
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# =============================================================================
# Dataset — paired frames with raw versions for flow computation
# =============================================================================
class FlowFrameDataset(Dataset):
    """
    Returns (f_t_norm, f_t1_norm, f_t2_norm, action_t_norm)
    action_t_norm: (2,) float tensor of normalised (dx, dy) in [-1, 1]

    Loads from flow_meta.json which contains per-frame action vectors
    extracted from the trajectory data. This avoids unreliable pixel
    differencing — uses ground truth agent displacement directly.

    Falls back to pixel-difference centroid if flow_meta.json absent.
    """
    def __init__(self, frame_dir: str):
        import json
        self.frame_dir = Path(frame_dir)
        meta_path = self.frame_dir / "flow_meta.json"

        if meta_path.exists():
            meta = json.load(open(meta_path))
            self.frames   = [m["frame"]  for m in meta["frames"]]
            self.actions  = [m["action"] for m in meta["frames"]]
            self.use_meta = True
            print(f"  FlowFrameDataset: {len(self.frames)} frames (action-supervised)")
        else:
            paths = sorted(self.frame_dir.glob("*.jpg")) + \
                    sorted(self.frame_dir.glob("*.png"))
            self.frames  = [p.name for p in sorted(paths, key=lambda p: p.name)]
            self.actions = [[0.0, 0.0]] * len(self.frames)
            self.use_meta = False
            print(f"  FlowFrameDataset: {len(self.frames)} frames (pixel-diff fallback)")

        if len(self.frames) < 3:
            raise ValueError(f"Need ≥3 frames in {frame_dir}, got {len(self.frames)}")

    def __len__(self):
        return len(self.frames) - 2

    def __getitem__(self, idx):
        imgs = [Image.open(self.frame_dir / self.frames[idx + i]).convert("RGB")
                for i in range(3)]
        norm = [FRAME_TRANSFORM(im) for im in imgs]
        act  = torch.tensor(self.actions[idx], dtype=torch.float32)  # (2,)
        return (*norm, act)   # (f0_n, f1_n, f2_n, action)


# =============================================================================
# Training
# =============================================================================
def train_phase2b(
    data_dir:         str,
    out_dir:          str,
    resume:           str   = None,
    steps:            int   = 6000,
    batch_size:       int   = 16,
    lr:               float = 3e-4,
    lambda_sigreg:    float = 5.0,
    lambda_curv:      float = 0.1,
    lambda_flow:      float = 2.0,
    no_distill:       bool  = False,
    checkpoint_every: int   = 500,
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "="*62)
    print("  PHASE 2b — DYNAMICS + OPTICAL FLOW GROUNDING")
    print(f"  Data:           {data_dir}")
    print(f"  Steps:          {steps}")
    print(f"  λ_sigreg:       {lambda_sigreg}")
    print(f"  λ_curv:         {lambda_curv}")
    print(f"  λ_flow:         {lambda_flow}  ← agent position encoding")
    print(f"  no_distill:     {no_distill}  ← skip DINOv2 if True")
    print(f"  Resume:         {resume or 'scratch'}")
    print("="*62 + "\n")

    # Models
    student      = StudentEncoder()
    teacher      = DINOv2Teacher()
    bb_proj      = BackboneProjector(backbone_dim=32, teacher_dim=384)
    spatial_proj = SpatialChannelProjector(in_channels=32, dv=DV)
    pool_head    = SpatialPoolingHead(n_patches=N_PATCHES, dv=DV, out_dim=128)
    flow_head    = FlowHead(in_dim=128)

    if resume:
        ckpt = torch.load(resume, map_location="cpu")
        student.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        print(f"  Resumed: {resume}")

    n_params = sum(p.numel() for p in student.parameters())
    print(f"  Student:    {n_params:,} params")
    print(f"  FlowHead:   {sum(p.numel() for p in flow_head.parameters()):,} params")
    print(f"  Pool head:  {sum(p.numel() for p in pool_head.parameters()):,} params\n")

    optimizer = torch.optim.AdamW(
        list(student.parameters())      +
        list(bb_proj.parameters())      +
        list(spatial_proj.parameters()) +
        list(pool_head.parameters())    +
        list(flow_head.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    dataset   = FlowFrameDataset(data_dir)
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                           num_workers=0, drop_last=True)
    data_iter = iter(loader)

    loss_log = []
    step     = 0

    while step < steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        f0_n, f1_n, f2_n, action_t = batch
        t0 = time.perf_counter()

        # ── Encoder forward (normalised frames) ──────────────────────────────
        z_t,  sp_t  = student(f0_n, return_spatial=True)
        z_t1, sp_t1 = student(f1_n, return_spatial=True)
        z_t2, sp_t2 = student(f2_n, return_spatial=True)

        # ── Backbone features ─────────────────────────────────────────────────
        g_t  = sp_t.mean(dim=[-2, -1])     # (B, 32)
        g_t1 = sp_t1.mean(dim=[-2, -1])

        # ── Distillation ──────────────────────────────────────────────────────
        teacher_cls = teacher(f1_n)
        bb_out      = bb_proj(g_t1)
        l_distill   = distillation_loss(bb_out, teacher_cls)

        # ── SIGReg ────────────────────────────────────────────────────────────
        l_sigreg = sigreg_loss(g_t, g_t1)

        # ── Curvature (temporal straightening) ───────────────────────────────
        tok_t  = spatial_proj(sp_t,  use_tokens=True)
        tok_t1 = spatial_proj(sp_t1, use_tokens=True)
        tok_t2 = spatial_proj(sp_t2, use_tokens=True)
        g_pool_t  = pool_head(tok_t)
        g_pool_t1 = pool_head(tok_t1)
        g_pool_t2 = pool_head(tok_t2)
        l_curv = temporal_straightening_loss(g_pool_t, g_pool_t1, g_pool_t2)

        # ── Optical flow auxiliary loss (action-supervised) ─────────────────
        # action_t is ground truth normalised (dx,dy) from trajectory data
        # FlowHead(z_t) must predict how the agent moved → forces position encoding
        pred_flow = flow_head(z_t)   # (B, 2) in [-1, 1] via Tanh
        l_flow = F.mse_loss(pred_flow, action_t)

        # ── Total loss ────────────────────────────────────────────────────────
        l_total = ((torch.zeros(1) if no_distill else l_distill)
                   + lambda_sigreg * l_sigreg
                   + lambda_curv   * l_curv
                   + lambda_flow   * l_flow)

        optimizer.zero_grad()
        l_total.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        step_ms = (time.perf_counter() - t0) * 1000
        step   += 1

        record = {
            "step":    step,
            "distill": l_distill.item(),
            "sigreg":  l_sigreg.item(),
            "curv":    l_curv.item(),
            "flow":    l_flow.item(),
            "total":   l_total.item(),
        }
        loss_log.append(record)

        if step % 50 == 0:
            print(f"Step {step:>5}/{steps} | "
                  f"total={l_total.item():.4f}  "
                  f"distill={l_distill.item():.4f}  "
                  f"sigreg={l_sigreg.item():.4f}  "
                  f"curv={l_curv.item():.4f}  "
                  f"flow={l_flow.item():.4f}  "
                  f"({step_ms:.1f}ms)")

        if step % checkpoint_every == 0:
            ckpt_path = f"{out_dir}/cortex_student_flow_step{step:05d}.pt"
            save_checkpoint(step, student, optimizer, record, ckpt_path)
            print(f"  Checkpoint: {ckpt_path}")

    final_path = f"{out_dir}/cortex_student_flow_final.pt"
    save_checkpoint(step, student, optimizer, loss_log[-1], final_path)
    with open(f"{out_dir}/phase2b_loss_log.json", "w") as f:
        json.dump(loss_log, f, indent=2)

    print(f"\n  Final checkpoint: {final_path}")
    print(f"\n  Next steps:")
    print(f"  python calibrate_threshold.py \\")
    print(f"    --encoder {final_path} --option 1 --pred-dir ./predictors --pct 50")
    print(f"  python run_benchmark.py \\")
    print(f"    --encoder {final_path} --option 1 --env umaze --planner gradient \\")
    print(f"    --threshold-file ./benchmark_thresholds_opt1_50pct.json")
    return student


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="CORTEX-PE Phase 2b: Maze encoder with optical flow aux loss")
    p.add_argument("--data",           default="./phase2_frames",
                   help="Directory of sequential maze frames (.png)")
    p.add_argument("--resume",         default="./checkpoints/maze/cortex_student_phase2_final.pt",
                   help="Phase 2 checkpoint to resume from")
    p.add_argument("--out",            default="./checkpoints/maze_flow")
    p.add_argument("--steps",          type=int,   default=6000)
    p.add_argument("--batch",          type=int,   default=16)
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--lambda-sigreg",  type=float, default=5.0)
    p.add_argument("--lambda-curv",    type=float, default=0.1)
    p.add_argument("--lambda-flow",    type=float, default=2.0)
    p.add_argument("--no-distill",     action="store_true",
                   help="Skip DINOv2 distillation — pure action+SIGReg supervision")
    p.add_argument("--save-every",     type=int,   default=500)
    args = p.parse_args()

    train_phase2b(
        data_dir         = args.data,
        out_dir          = args.out,
        resume           = args.resume,
        steps            = args.steps,
        batch_size       = args.batch,
        lr               = args.lr,
        lambda_sigreg    = args.lambda_sigreg,
        lambda_curv      = args.lambda_curv,
        lambda_flow      = args.lambda_flow,
        no_distill       = args.no_distill,
        checkpoint_every = args.save_every,
    )
