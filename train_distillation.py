"""
train_distillation.py — CORTEX-16 Knowledge Distillation Training Loop

Two-phase training:

  Phase 1 — Semantic Grounding
    Loss: L_distill(backbone_g) + λ_sigreg * L_sigreg(backbone_g)

    KEY ARCHITECTURAL FIX (v3):
    Distillation is applied directly on backbone_g (32-D) through a small
    32→384 linear projector — NOT on the final 128-D head output.

    Why this works:
    - backbone_g must produce DIVERSE 32-D vectors to predict diverse 384-D
      DINOv2 targets through a linear layer (a constant backbone_g can only
      map to one direction)
    - SIGReg simultaneously enforces Gaussian spread on backbone_g
    - Both losses now pull in the same direction: diverse backbone features
    - The head learns to decode diverse backbone features into useful 128-D space

    Previous failures:
    - v1: distill on student_z (128-D, normalised) → SIGReg couldn't move it
    - v2: distill on student_z (128-D, unnormalised) → head became constant
    - v3 (this): distill on backbone_g (32-D) → backbone forced to be diverse

  Phase 2 — Dynamics Grounding
    Adds L_curv on sequential domain frames.

Usage:
  python train_distillation.py --phase 1 --data ./tiny-imagenet-200/train --steps 12000 --out ./checkpoints --lambda-sigreg 5.0
  python train_distillation.py --phase 2 --data ./phase2_frames --steps 6000 --resume ./checkpoints/cortex_student_phase1_final.pt --out ./checkpoints
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np

from student_encoder import StudentEncoder
from train_predictor import SpatialPoolingHead, SpatialChannelProjector, DV, N_PATCHES
from latent_predictor import sigreg_loss, temporal_straightening_loss


# =============================================================================
# Transforms
# =============================================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# =============================================================================
# Sequential Frame Dataset (Phase 2)
# =============================================================================
class SequentialFrameDataset(Dataset):
    def __init__(self, frame_dir: str, transform=None):
        paths = sorted(Path(frame_dir).glob("*.jpg")) + \
                sorted(Path(frame_dir).glob("*.png"))
        self.paths     = sorted(paths, key=lambda p: p.name)
        self.transform = transform or TRAIN_TRANSFORM
        if len(self.paths) < 3:
            raise ValueError(f"Need ≥3 frames in {frame_dir}, found {len(self.paths)}")

    def __len__(self):
        return len(self.paths) - 2

    def __getitem__(self, idx):
        t0 = Image.open(self.paths[idx]).convert("RGB")
        t1 = Image.open(self.paths[idx + 1]).convert("RGB")
        t2 = Image.open(self.paths[idx + 2]).convert("RGB")
        return self.transform(t0), self.transform(t1), self.transform(t2)


# =============================================================================
# DINOv2 Teacher — frozen, outputs normalised CLS features (B, 384)
# =============================================================================
class DINOv2Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        print("🔄 Loading DINOv2 teacher (facebook/dinov2-small)...")
        try:
            from transformers import AutoModel
            self.model    = AutoModel.from_pretrained("facebook/dinov2-small")
            self._use_hf  = True
            print("✅ Loaded via HuggingFace transformers")
        except Exception:
            self.model    = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
            self._use_hf  = False
            print("✅ Loaded via torch.hub")
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_hf:
            out = self.model(pixel_values=x)
            cls = out.last_hidden_state[:, 0, :].float()
        else:
            cls = self.model(x)
        return F.normalize(cls, dim=-1)   # (B, 384)


# =============================================================================
# BackboneProjector — 32-D backbone → 384-D teacher space
#
# This is the critical architectural change vs previous versions.
# By projecting backbone_g (32-D) to teacher space (384-D), distillation
# gradient flows directly into the backbone, forcing diversity.
# A constant backbone vector cannot fit diverse 384-D DINOv2 targets.
# =============================================================================
class BackboneProjector(nn.Module):
    """
    Projects 32-D backbone features to 384-D DINOv2 teacher space.
    Small MLP with one hidden layer for non-linear capacity.
    NOT exported to NPU — training only.
    """
    def __init__(self, backbone_dim: int = 32, teacher_dim: int = 384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(backbone_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, teacher_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)   # (B, 384) normalised


class DeepSupervisionProjectors(nn.Module):
    """
    V-JEPA 2.1 Deep Self-Supervision: four projectors, one per CNN block.

    Maps each intermediate block descriptor to teacher space (384-D DINOv2).
    Block dims match CortexCNNBackbone:
        Block 1: 16-D  (after stride-2 from input)
        Block 2: 32-D
        Block 3: 64-D  (widest — most discriminative features)
        Block 4: 32-D  (same as existing BackboneProjector)

    Training only — never exported to NPU.
    """
    def __init__(self, teacher_dim: int = 384):
        super().__init__()
        block_dims = [16, 32, 64, 32]
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, max(64, d * 2)),
                nn.ReLU(inplace=True),
                nn.Linear(max(64, d * 2), teacher_dim),
            )
            for d in block_dims
        ])
        # Deeper layers are weighted more — linear ramp per V-JEPA 2.1
        self.level_weights = [0.25, 0.5, 0.75, 1.0]

    def forward_all(
        self,
        intermediates: list,
        teacher_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute weighted deep supervision loss across all 4 levels.

        Args:
            intermediates:  list of 4 tensors [(B,16),(B,32),(B,64),(B,32)]
            teacher_target: (B, 384) stop-gradient DINOv2 CLS features

        Returns:
            Scalar deep supervision loss (normalised to match single-level scale)
        """
        from latent_predictor import deep_supervision_loss
        return deep_supervision_loss(
            intermediates  = intermediates,
            projectors     = [p for p in self.projectors],
            teacher_target = teacher_target,
            level_weights  = self.level_weights,
        )

    def param_list(self):
        return list(self.parameters())


# =============================================================================
# Losses
# =============================================================================
def distillation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cosine similarity loss between backbone projection and teacher CLS."""
    return (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()


# =============================================================================
# Checkpoint utilities
# =============================================================================
def save_checkpoint(step, student, optimizer, losses, path):
    torch.save({
        "step":      step,
        "model":     student.state_dict(),
        "optimizer": optimizer.state_dict(),
        "losses":    losses,
    }, path)


def load_checkpoint(path, student, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    student.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    print(f"✅ Resumed from {path} (step {ckpt['step']})")
    return ckpt["step"]


# =============================================================================
# Phase 1 — Semantic Grounding
#
# Loss architecture (v3):
#   backbone_g → BackboneProjector → (B, 384)  ← distillation vs DINOv2 CLS
#   backbone_g                               ← SIGReg spread enforcement
#
# Both losses act on backbone_g → backbone must produce diverse features.
# The head (backbone_g → z 128-D) benefits from diverse input automatically.
# =============================================================================
def train_phase1(
    data_dir,
    steps=12000,
    batch_size=32,
    lr=1e-3,
    lambda_sigreg=5.0,
    lambda_deep=1.0,
    resume=None,
    checkpoint_every=1000,
    output_dir=".",
    gc="none",
):
    print("\n" + "="*60)
    print("  PHASE 1 — SEMANTIC GROUNDING (v4 — V-JEPA 2.1 Deep Supervision)")
    print(f"  Dataset:  {data_dir}")
    print(f"  Steps:    {steps}")
    print(f"  Loss:     L_distill(backbone_g->384) + {lambda_sigreg}*L_sigreg")
    print(f"            + {lambda_deep}*L_deep (all 4 CNN blocks supervised)")
    print("="*60 + "\n")

    student    = StudentEncoder()
    teacher    = DINOv2Teacher()
    bb_proj    = BackboneProjector(backbone_dim=32, teacher_dim=384)
    deep_proj  = DeepSupervisionProjectors(teacher_dim=384)

    n_deep = sum(p.numel() for p in deep_proj.parameters())
    print(f"  DeepSupervisionProjectors: {n_deep:,} params (training only)\n")

    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(bb_proj.parameters()) +
        deep_proj.param_list(),
        lr=lr, weight_decay=1e-4,
    )

    if gc != 'none':
        try:
            from gc_optimizer import GCAdamW as _GC
            optimizer = _GC(optimizer.param_groups, lr=optimizer.defaults.get('lr',1e-3), weight_decay=optimizer.defaults.get('weight_decay',0), use_gc=(gc in ('standard','gcc2')), use_mc=(gc=='moment'))
            print(f'  Optimizer: GCAdamW gc={gc}')
        except Exception as e:
            print(f'  GC failed {e}, using AdamW')
    else:
        print('  Optimizer: AdamW baseline')
    # -- SIGReg variant selection ---------------------------------------
    _sig = getattr(args, "sigreg", "vicreg")
    if _sig == "weak":
        import torch as _t
        def sigreg_loss(z, *a, K=32, **kw):
            D=z.shape[1]; _t.manual_seed(42)
            S=_t.randn(D,K,device=z.device)/(K**0.5); S=S/S.norm(dim=0,keepdim=True)
            sk=z@S; sk_c=sk-sk.mean(0); cov=(sk_c.T@sk_c)/(z.shape[0]-1)
            return (cov-_t.eye(K,device=z.device)).pow(2).sum()/K
        print("  SIGReg: Weak-SIGReg K=32")
    elif _sig == "strong":
        import torch as _t, torch.nn.functional as _F
        def sigreg_loss(z, *a, M=16, T=17, **kw):
            z_n=(z-z.mean(0))/(z.std(0)+1e-6); d=_F.normalize(_t.randn(M,z.shape[1],device=z.device),dim=1)
            p2=z_n@d.T; t=_t.linspace(-4,4,T,device=z.device); loss=_t.tensor(0.,device=z.device)
            for m in range(M): loss=loss+(_t.cos(t.unsqueeze(0)*p2[:,m].unsqueeze(1)).mean(0)-_t.exp(-0.5*t**2)).pow(2).mean()
            return loss/M
        print("  SIGReg: Strong Epps-Pulley (LeWM)")
    else:
        from latent_predictor import sigreg_loss
        print("  SIGReg: VICReg-style (current)")
    # -------------------------------------------------------------------
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    start_step = 0
    if resume:
        start_step = load_checkpoint(resume, student, optimizer)

    dataset = ImageFolder(data_dir, transform=TRAIN_TRANSFORM)
    loader  = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=False, drop_last=True,
    )
    data_iter = iter(loader)

    print(f"Dataset: {len(dataset)} images | {len(dataset)//batch_size} batches/epoch\n")

    loss_log = []
    step     = start_step

    while step < steps:
        try:
            frames, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            frames, _ = next(data_iter)

        t0 = time.perf_counter()

        # --- Forward with deep supervision intermediates ---
        _, spatial, intermediates = student(frames, return_intermediates=True)
        backbone_g   = spatial.mean(dim=[-2, -1])              # (B, 32)

        teacher_cls  = teacher(frames)                         # (B, 384) normalised
        bb_proj_out  = bb_proj(backbone_g)                     # (B, 384) normalised

        # Distillation: backbone_g → 384-D must match DINOv2 teacher
        l_distill = distillation_loss(bb_proj_out, teacher_cls)

        # SIGReg: enforce Gaussian spread on backbone_g
        l_sigreg  = sigreg_loss(backbone_g, backbone_g.roll(1, dims=0))

        # V-JEPA 2.1 Deep Self-Supervision: all 4 CNN blocks supervised
        # Forces intermediate representations to stay semantically grounded
        # Deeper blocks weighted more (0.25, 0.5, 0.75, 1.0)
        l_deep = deep_proj.forward_all(intermediates, teacher_cls)

        l_total = l_distill + lambda_sigreg * l_sigreg + lambda_deep * l_deep

        # --- Backward ---
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
            "deep":    l_deep.item(),
            "total":   l_total.item(),
        }
        loss_log.append(record)

        if step % 100 == 0:
            print(
                f"Step {step:>6}/{steps} | "
                f"total={l_total.item():.4f}  "
                f"distill={l_distill.item():.4f}  "
                f"sigreg={l_sigreg.item():.4f}  "
                f"deep={l_deep.item():.4f}  "
                f"({step_ms:.1f}ms)"
            )

        if step % checkpoint_every == 0:
            ckpt_path = f"{output_dir}/cortex_student_phase1_step{step:06d}.pt"
            save_checkpoint(step, student, optimizer, record, ckpt_path)
            print(f"💾 Checkpoint saved: {ckpt_path}")

    final_path = f"{output_dir}/cortex_student_phase1_final.pt"
    save_checkpoint(step, student, optimizer, loss_log[-1], final_path)
    with open(f"{output_dir}/phase1_loss_log.json", "w") as f:
        json.dump(loss_log, f, indent=2)

    print(f"\n✅ Phase 1 complete. Final checkpoint: {final_path}")
    return student


# =============================================================================
# Phase 2 — Dynamics Grounding
# Same backbone distillation fix applied here.
# =============================================================================
def train_phase2(
    data_dir,
    steps=6000,
    batch_size=16,
    lr=3e-4,
    lambda_sigreg=5.0,
    lambda_curv=1.0,
    resume=None,
    checkpoint_every=500,
    output_dir=".",
    gc="none",
):
    print("\n" + "="*60)
    print("  PHASE 2 — DYNAMICS GROUNDING")
    print(f"  Dataset:  {data_dir}")
    print(f"  Steps:    {steps}")
    print(f"  Loss:     L_distill(backbone) + {lambda_sigreg}*L_sigreg + {lambda_curv}*L_curv")
    print("="*60 + "\n")

    student     = StudentEncoder()
    teacher     = DINOv2Teacher()
    bb_proj     = BackboneProjector(backbone_dim=32, teacher_dim=384)
    spatial_proj = SpatialChannelProjector(in_channels=32, dv=DV)
    pool_head   = SpatialPoolingHead(n_patches=N_PATCHES, dv=DV, out_dim=128)
    print(f"✅ SpatialPoolingHead added — curvature loss via [agg] pooling (Wang et al. B.5)")
    print(f"   lambda_curv={lambda_curv} (use 0.1 for agg per Wang et al.)")

    if resume:
        load_checkpoint(resume, student)
        print(f"📦 Phase 2 initialised from Phase 1 weights: {resume}")

    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(bb_proj.parameters()) +
        list(spatial_proj.parameters()) + list(pool_head.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    dataset   = SequentialFrameDataset(data_dir, transform=TRAIN_TRANSFORM)
    loader    = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=False, drop_last=True,
    )
    data_iter = iter(loader)
    print(f"Dataset: {len(dataset)} triplets\n")

    loss_log = []
    step     = 0

    while step < steps:
        try:
            f_t, f_t1, f_t2 = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            f_t, f_t1, f_t2 = next(data_iter)

        t0 = time.perf_counter()

        # Forward over triplet
        z_t,  sp_t  = student(f_t,  return_spatial=True)
        z_t1, sp_t1 = student(f_t1, return_spatial=True)
        z_t2, sp_t2 = student(f_t2, return_spatial=True)

        g_t  = sp_t.mean(dim=[-2, -1])     # (B, 32) backbone features
        g_t1 = sp_t1.mean(dim=[-2, -1])

        teacher_cls = teacher(f_t1)         # (B, 384)
        bb_proj_out = bb_proj(g_t1)         # (B, 384)

        l_distill = distillation_loss(bb_proj_out, teacher_cls)
        l_sigreg  = sigreg_loss(g_t, g_t1)

        # Curvature loss via learnable pooling head [agg] — Wang et al. B.5
        # Project 32-ch spatial maps to patch tokens, then pool to 128-D
        B = sp_t.shape[0]
        def to_tokens(sp):
            return sp.permute(0, 2, 3, 1).reshape(B, N_PATCHES, 32)
        tok_t  = spatial_proj(sp_t,  use_tokens=True)   # (B, 196, 8)
        tok_t1 = spatial_proj(sp_t1, use_tokens=True)
        tok_t2 = spatial_proj(sp_t2, use_tokens=True)
        g_pool_t  = pool_head(tok_t)    # (B, 128)
        g_pool_t1 = pool_head(tok_t1)
        g_pool_t2 = pool_head(tok_t2)
        l_curv = temporal_straightening_loss(g_pool_t, g_pool_t1, g_pool_t2)
        l_total   = l_distill + lambda_sigreg * l_sigreg + lambda_curv * l_curv

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
            "total":   l_total.item(),
        }
        loss_log.append(record)

        if step % 50 == 0:
            print(
                f"Step {step:>5}/{steps} | "
                f"total={l_total.item():.4f}  "
                f"distill={l_distill.item():.4f}  "
                f"sigreg={l_sigreg.item():.4f}  "
                f"curv={l_curv.item():.4f}  "
                f"({step_ms:.1f}ms)"
            )

        if step % checkpoint_every == 0:
            ckpt_path = f"{output_dir}/cortex_student_phase2_step{step:05d}.pt"
            save_checkpoint(step, student, optimizer, record, ckpt_path)
            print(f"💾 Checkpoint saved: {ckpt_path}")

    final_path = f"{output_dir}/cortex_student_phase2_final.pt"
    save_checkpoint(step, student, optimizer, loss_log[-1], final_path)
    with open(f"{output_dir}/phase2_loss_log.json", "w") as f:
        json.dump(loss_log, f, indent=2)

    print(f"\n✅ Phase 2 complete. Final checkpoint: {final_path}")
    return student


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="CORTEX-16 Distillation Trainer v4 (V-JEPA 2.1)")
    p.add_argument("--phase",   type=int, required=True, choices=[1, 2])
    p.add_argument("--data",    type=str, required=True)
    p.add_argument("--steps",   type=int, default=None)
    p.add_argument("--batch",   type=int, default=None)
    p.add_argument("--lr",      type=float, default=None)
    p.add_argument("--resume",  type=str, default=None)
    p.add_argument("--out",     type=str, default=".")
    p.add_argument("--lambda-sigreg", type=float, default=5.0)
    p.add_argument("--lambda-curv",   type=float, default=1.0)
    p.add_argument("--lambda-deep",   type=float, default=1.0,
                   help="V-JEPA 2.1 deep self-supervision weight (phase 1 only)")
    p.add_argument("--gc", default="none", choices=["none","standard","gcc2","moment"])
    p.add_argument("--sigreg", default="vicreg", choices=["vicreg","weak","strong"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)

    if args.phase == 1:
        train_phase1(
            data_dir      = args.data,
            steps         = args.steps or 12000,
            batch_size    = args.batch or 32,
            lr            = args.lr or 1e-3,
            lambda_sigreg = args.lambda_sigreg,
            lambda_deep   = args.lambda_deep,
            resume        = args.resume,
            output_dir    = args.out,
            gc            = args.gc,
        )
    else:
        train_phase2(
            data_dir      = args.data,
            steps         = args.steps or 6000,
            batch_size    = args.batch or 16,
            lr            = args.lr or 3e-4,
            lambda_sigreg = args.lambda_sigreg,
            lambda_curv   = args.lambda_curv,
            resume        = args.resume,
            output_dir    = args.out,
            gc            = args.gc,
        )
