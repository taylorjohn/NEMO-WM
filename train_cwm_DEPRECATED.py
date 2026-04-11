import sys; sys.path.insert(0, 'cortex_wm')
"""
train_cwm.py  —  CORTEX World Model  Sprint 1
==============================================
Full training loop: MoE-JEPA predictor + contact head + THICK context GRU
+ 7-signal neuromodulator reward shaping.

Architecture stack (bottom to top):
  StudentEncoder (FROZEN, XINT8 NPU)
      ↓ 128-D latent
  MOERouterV2 (encoder-side: dense soft-gate, 4 spectral experts)
      ↓ 128-D spectral ribbon (Shape/Size/Depth/Velocity)
  ParticleEncoder (SpatialSoftmax → K=16 particles)
      ↓ (B, K, 128) particle set
  MoEJEPAPredictor (dynamics: sparse top-2 of 4, neuro-temperature)
      ↓ (B, K, 128) predicted next particles
  ContactSignedDistanceHead (ACh-gated contact attention, +8K params)
      ↓ (B, K, K) pairwise signed distances
  THICKContextGRU (slow-timescale skill dynamics, +25K params)
      ↓ (B, context_dim) skill context
  NeuromodulatedCWMLoss (7-signal, DA×5HT×NE×ACh×E/I×Ado×eCB)
      ↓ scalar loss
  RegimeGatedTrainer (lr, clip, domain_diversity, n_candidates)

JEPA stop-gradient:
  Context path (no stop-grad):  particles_t → predictor → z_pred
  Target path  (stop-gradient): frames_t1 → encoder → z_target.detach()
  Loss: MSE(z_pred, sg(z_target)) — no pixel reconstruction ever

Two-level MoE:
  Encoder MoE (MOERouterV2):  dense soft-gate, 4 spectral features
  Predictor MoE (SparseMoEFFN): sparse top-2, 4 physics experts
"""

import io
import math
import time
from pathlib import Path
from typing import Optional, Dict, Any

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# ── CORTEX imports ────────────────────────────────────────────────────────
from neuromodulator import NeuromodulatorState, Regime
from cwm_neuro_reward import (
    NeuromodulatedCWMLoss,
    RegimeGatedTrainer,
    _sigreg_loss,
    _contact_loss,
    _skill_transition_loss,
    _straightening_loss,
)
from cwm_moe_jepa import (
    MoEJEPAPredictor,
    jepa_moe_loss,
    log_expert_utilisation,
)
from moe_router_v2 import MOERouterV2


# ═══════════════════════════════════════════════════════════════════════════
# Domain registry
# ═══════════════════════════════════════════════════════════════════════════

DOMAIN_IDS = {
    "recon":       0,
    "ogbench":     1,
    "tworoom":     2,
    "pusht":       3,
    "smap":        4,
    "cardiac":     5,
    "hexapod":     6,
    "quadruped":   7,
}

DOMAIN_ACTION_DIM = {
    "recon":     2,
    "ogbench":   9,
    "tworoom":   2,
    "pusht":     2,
    "smap":      0,   # observation-only → zeros
    "cardiac":   0,
    "hexapod":   4,   # CPG-encoded
    "quadruped": 12,
}

MAX_ACTION_DIM = 9   # OGBench-Cube sets the ceiling


# ═══════════════════════════════════════════════════════════════════════════
# Contact Signed Distance Head  (+8K params)
# ═══════════════════════════════════════════════════════════════════════════

class ContactSignedDistanceHead(nn.Module):
    """
    Predicts pairwise signed distances between particles.
    Supervised free from SpatialSoftmax positions.
    ACh-gated: high ACh (surprising + unstable) → sharpen contact attention.

    Architecture:
      particle_i, particle_j → MLP → scalar signed distance
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        # Pairwise MLP: concat two particle embeddings → scalar distance
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        # ACh-gated attention scale (set each forward from neuro signals)
        self._ach_scale: float = 1.0

    def set_ach(self, ach: float):
        # [0.5, 2.0] — matches lr_scale formula from neuromodulator.py
        self._ach_scale = 0.5 + ach * 1.5

    def forward(
        self,
        particles: torch.Tensor,      # (B, K, d_model)
        positions: torch.Tensor,      # (B, K, 2)  SpatialSoftmax XY
    ) -> torch.Tensor:
        B, K, D = particles.shape

        # Pairwise particle combinations
        pi = particles.unsqueeze(2).expand(B, K, K, D)  # (B,K,K,D)
        pj = particles.unsqueeze(1).expand(B, K, K, D)
        pair = torch.cat([pi, pj], dim=-1)               # (B,K,K,2D)

        signed_dist = self.mlp(pair).squeeze(-1)         # (B,K,K)

        # Ground truth: euclidean distances between SpatialSoftmax positions
        pos_i = positions.unsqueeze(2).expand(B, K, K, 2)
        pos_j = positions.unsqueeze(1).expand(B, K, K, 2)
        true_dist = (pos_i - pos_j).norm(dim=-1)         # (B,K,K)

        # Contact supervision loss (MSE between predicted and true distances)
        contact_loss = F.mse_loss(signed_dist, true_dist) * self._ach_scale

        return signed_dist, contact_loss


# ═══════════════════════════════════════════════════════════════════════════
# THICK Context GRU  (+25K params)
# ═══════════════════════════════════════════════════════════════════════════

class THICKContextGRU(nn.Module):
    """
    Temporal Hierarchies from Invariant Context Kernels (THICK, ICLR 2024).

    Operates at the slow timescale — updates only when a skill boundary
    is detected (particle configuration changes discontinuously).

    For OGBench-Cube: reduces effective planning horizon from 50 → ~6 steps
    by representing skill-level context (approach / grasp / lift / place).

    Parameters
    ----------
    d_model     : particle embedding dim (128)
    context_dim : slow context dimension (32) → +25K params
    K           : number of particles (16)
    """

    def __init__(
        self,
        d_model:     int = 128,
        context_dim: int = 32,
        K:           int = 16,
    ):
        super().__init__()
        self.context_dim = context_dim

        # Fast → slow: compress K particles into context update
        self.particle_pool = nn.Linear(d_model * K, context_dim * 4)

        # Sparsity gate: is this timestep a skill boundary?
        # Near 1.0 → context change (new skill starting)
        # Near 0.0 → within-skill (context stays constant)
        self.change_gate = nn.Sequential(
            nn.Linear(d_model * K, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Slow GRU: updates context only at skill boundaries
        self.gru = nn.GRUCell(context_dim * 4, context_dim)

        # High-level predictor: context × action → next context
        # Used for abstract skill-level MPC (6-step horizon)
        self.skill_pred = nn.GRUCell(context_dim + MAX_ACTION_DIM, context_dim)

    def forward(
        self,
        particles:   torch.Tensor,   # (B, K, d_model) current particles
        context_h:   torch.Tensor,   # (B, context_dim) slow context state
        action:      torch.Tensor,   # (B, MAX_ACTION_DIM) padded action
    ) -> tuple:
        B, K, D = particles.shape
        flat = particles.reshape(B, -1)  # (B, K*D)

        # Detect skill boundary
        change_prob = self.change_gate(flat)  # (B, 1)

        # Candidate context update
        pool    = F.gelu(self.particle_pool(flat))          # (B, context_dim*4)
        ctx_cand = self.gru(pool, context_h)                 # (B, context_dim)

        # Sparse update: only update at skill boundaries
        # During training: straight-through for gradient flow
        context_h_new = (change_prob * ctx_cand +
                          (1.0 - change_prob) * context_h)

        # Abstract skill prediction (for long-horizon planning)
        skill_inp  = torch.cat([context_h_new, action], dim=-1)
        ctx_future = self.skill_pred(skill_inp, context_h_new)

        return context_h_new, ctx_future, change_prob.squeeze(-1)

    def init_context(self, B: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(B, self.context_dim, device=device)


# ═══════════════════════════════════════════════════════════════════════════
# SpatialSoftmax Particle Encoder
# ═══════════════════════════════════════════════════════════════════════════

class ParticleEncoder(nn.Module):
    """
    Converts a 128-D latent (from StudentEncoder) into K spatial particles
    via SpatialSoftmax keypoint detection.

    Each particle carries a position (x,y) and a d_model-dim feature vector.
    """

    def __init__(self, d_model: int = 128, K: int = 16, img_size: int = 7):
        super().__init__()
        self.K        = K
        self.d_model  = d_model
        self.img_size = img_size

        # Project latent into K spatial feature maps
        self.spatial_proj = nn.Sequential(
            nn.Linear(d_model, img_size * img_size * K),
            nn.Unflatten(-1, (K, img_size, img_size)),
        )

        # Per-particle feature projection
        self.particle_proj = nn.Linear(d_model + 2, d_model)

        # Temperature for SpatialSoftmax
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, z: torch.Tensor) -> tuple:
        """
        z : (B, d_model)

        Returns:
          particles  : (B, K, d_model)
          positions  : (B, K, 2)  normalised XY in [-1, 1]
        """
        B, D = z.shape

        # Project to K spatial feature maps
        feat_maps = self.spatial_proj(z)  # (B, K, H, W)
        B, K, H, W = feat_maps.shape

        # SpatialSoftmax: compute expected XY position per keypoint
        flat  = feat_maps.reshape(B, K, -1)
        attn  = F.softmax(flat / self.temperature, dim=-1)
        attn  = attn.reshape(B, K, H, W)

        # Expected position grid
        xs = torch.linspace(-1, 1, W, device=z.device)
        ys = torch.linspace(-1, 1, H, device=z.device)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')
        grid_x = grid_x.expand(B, K, H, W)
        grid_y = grid_y.expand(B, K, H, W)

        pos_x = (attn * grid_x).sum(dim=(-2, -1))   # (B, K)
        pos_y = (attn * grid_y).sum(dim=(-2, -1))   # (B, K)
        positions = torch.stack([pos_x, pos_y], dim=-1)  # (B, K, 2)

        # Particle feature: latent + position
        z_expanded = z.unsqueeze(1).expand(B, K, D)
        particles  = self.particle_proj(
            torch.cat([z_expanded, positions], dim=-1)
        )   # (B, K, d_model)

        return particles, positions


# ═══════════════════════════════════════════════════════════════════════════
# RECON Dataset
# ═══════════════════════════════════════════════════════════════════════════

class RECONDataset(Dataset):
    """
    RECON outdoor navigation dataset.
    Loads (frame_t, frame_t1, action, gps) tuples from HDF5 files.

    Each HDF5 file: one Jackal robot trajectory.
    Keys expected: 'images' (JPEG bytes), 'actions', 'observations' (GPS).
    """

    def __init__(
        self,
        hdf5_dir:  str,
        domain:    str = "recon",
        max_files: int = None,
        img_size:  int = 224,
    ):
        self.domain    = domain
        self.domain_id = DOMAIN_IDS[domain]
        self.img_size  = img_size

        files = sorted(Path(hdf5_dir).glob("jackal_2019-*.hdf5"))
        if max_files:
            files = files[:max_files]

        self.samples = []
        n_skip = 0
        for f in files:
            try:
                with h5py.File(f, 'r') as hf:
                    n = len(hf['images']['rgb_left']) - 1
                    for i in range(n):
                        self.samples.append((str(f), i))
            except Exception:
                n_skip += 1
        if n_skip:
            print(f'  Skipped {n_skip} corrupted HDF5 files')

    def _decode_frame(self, jpeg_bytes: bytes) -> torch.Tensor:
        img = Image.open(io.BytesIO(jpeg_bytes)).convert('RGB')
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img = img.crop((0, 0, self.img_size, self.img_size))
        t   = torch.from_numpy(np.array(img)).float() / 255.0
        return t.permute(2, 0, 1)  # (3, H, W)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        path, i = self.samples[idx]
        with h5py.File(path, 'r') as hf:
            imgs     = hf['images']['rgb_left']
            frame_t  = self._decode_frame(bytes(imgs[i]))
            frame_t1 = self._decode_frame(bytes(imgs[i + 1]))

            # Action: [linear_velocity, angular_velocity]
            lin = float(hf['commands']['linear_velocity'][i])
            ang = float(hf['commands']['angular_velocity'][i])
            action = torch.tensor([lin, ang], dtype=torch.float32)
            action_padded = F.pad(action, (0, MAX_ACTION_DIM - 2))

            gps_raw = list(hf['gps']['latlong'][i]) if 'gps' in hf else [0., 0.]
            gps     = torch.tensor(gps_raw, dtype=torch.float32)

        return {
            "frame_t":    frame_t,
            "frame_t1":   frame_t1,
            "action":     action_padded,
            "gps":        gps,
            "domain_id":  torch.tensor(self.domain_id, dtype=torch.long),
            "domain":     self.domain,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Full CWM Model (assembled)
# ═══════════════════════════════════════════════════════════════════════════

class CortexWorldModel(nn.Module):
    """
    Full CORTEX World Model:
      Encoder MoE (spectral ribbon) → Particle encoder
      → MoE-JEPA predictor → Contact head → THICK GRU

    Two-level MoE:
      Level 1 (encoder): MOERouterV2 — dense soft-gate, 4 spectral experts
        Shape/Size/Depth/Velocity — runs on the raw latent
      Level 2 (predictor): SparseMoEFFN — sparse top-2 of 4 physics experts
        Contact/Navigation/Drift/Locomotion — runs on particles
    """

    def __init__(
        self,
        d_model:     int = 128,
        K:           int = 16,
        n_layers:    int = 2,
        n_heads:     int = 4,
        n_experts:   int = 4,
        k_active:    int = 2,
        context_dim: int = 32,
        student_ckpt: Optional[str] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.K       = K

        # ── Encoder pipeline ───────────────────────────────────────────────
        # StudentEncoder loaded separately (FROZEN, XINT8 NPU path)
        # MOERouterV2: encoder-side dense soft-gate (spectral ribbon)
        self.encoder_moe   = MOERouterV2(input_dim=d_model, manifold_dim=d_model)
        # Note: StudentEncoder produces 256-D CLS token → MOERouterV2 → 128-D

        self.particle_enc  = ParticleEncoder(d_model=d_model, K=K)

        # ── Predictor (MoE-JEPA dynamics) ─────────────────────────────────
        self.predictor = MoEJEPAPredictor(
            d_model=d_model, K=K, n_layers=n_layers, n_heads=n_heads,
            n_experts=n_experts, k_active=k_active,
            max_action_dim=MAX_ACTION_DIM,
        )

        # ── Contact head ───────────────────────────────────────────────────
        self.contact_head = ContactSignedDistanceHead(d_model=d_model)

        # ── Slow context (THICK GRU) ───────────────────────────────────────
        self.thick_gru = THICKContextGRU(
            d_model=d_model, context_dim=context_dim, K=K
        )

        # ── GPS grounding head ─────────────────────────────────────────────
        # Projects mean particle position → 2-D GPS coordinates
        self.gps_head = nn.Linear(2, 2)   # identity init, learns offset/scale

    def encode(self, z_student: torch.Tensor) -> tuple:
        """
        z_student : (B, 256) — raw StudentEncoder CLS token

        Returns particles (B, K, d_model) and positions (B, K, 2).
        """
        # Level-1 MoE: spectral ribbon (dense soft-gate)
        spectral, moe_weights = self.encoder_moe(z_student)  # (B, 128), (B, 4)
        # Particles from spectral latent
        particles, positions  = self.particle_enc(spectral)
        return particles, positions, spectral, moe_weights

    def predict(
        self,
        particles:  torch.Tensor,       # (B, K, d_model)
        action:     torch.Tensor,       # (B, MAX_ACTION_DIM)
        context_h:  torch.Tensor,       # (B, context_dim) THICK state
        positions:  torch.Tensor,       # (B, K, 2)
        domain_id:  torch.Tensor,       # (B,) int
        regime:     str = "EXPLOIT",
        ach:        float = 0.5,
    ) -> dict:
        """Full predictor forward: MoE-JEPA → contact head → THICK GRU."""

        # Level-2 MoE: sparse physics predictor
        z_pred, pred_aux = self.predictor(
            particles=particles,
            action=action,
            domain_id=domain_id,
            regime=regime,
        )

        # Contact head (ACh-gated)
        self.contact_head.set_ach(ach)
        signed_dist, contact_loss = self.contact_head(z_pred, positions)

        # THICK context GRU (skill-level slow dynamics)
        context_h_new, ctx_future, change_prob = self.thick_gru(
            particles=z_pred,
            context_h=context_h,
            action=action,
        )

        # GPS from mean particle position
        mean_pos = positions.mean(dim=1)   # (B, 2)
        gps_pred = self.gps_head(mean_pos)

        return {
            "z_pred":        z_pred,
            "signed_dist":   signed_dist,
            "contact_loss":  contact_loss,
            "context_h":     context_h_new,
            "ctx_future":    ctx_future,
            "change_prob":   change_prob,
            "gps_pred":      gps_pred,
            "moe_aux_loss":  pred_aux["moe_aux_loss"],
            "attn_weights":  pred_aux["attn_weights"],
        }

    def total_params(self) -> dict:
        def cnt(m): return sum(p.numel() for p in m.parameters())
        return {
            "encoder_moe":   cnt(self.encoder_moe),
            "particle_enc":  cnt(self.particle_enc),
            "predictor":     cnt(self.predictor),
            "contact_head":  cnt(self.contact_head),
            "thick_gru":     cnt(self.thick_gru),
            "gps_head":      cnt(self.gps_head),
            "total":         cnt(self),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════════════

def train_cwm(
    hdf5_dir:     str   = "recon_data/recon_release",
    student_ckpt: str   = r"checkpoints\dinov2_student\student_best.pt",
    n_epochs:     int   = 30,
    batch_size:   int   = 16,
    base_lr:      float = 3e-4,
    max_files:    int   = None,
    save_dir:     str   = r"checkpoints\cwm",
    log_every:    int   = 50,
    device_str:   str   = "cpu",
):
    device = torch.device(device_str)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # ── Neuromodulator system ──────────────────────────────────────────────
    neuro   = NeuromodulatorState(session_start=time.time(),
                                   ado_saturate_hours=float(n_epochs))
    loss_fn = NeuromodulatedCWMLoss(
        lambda_gaussian=0.10,
        lambda_gps=0.05,
        lambda_contact=0.01,
        lambda_skill=0.05,
        lambda_curv=0.02,
    )

    # ── Dataset & loader ──────────────────────────────────────────────────
    dataset = RECONDataset(hdf5_dir, domain="recon", max_files=max_files)
    loader  = DataLoader(dataset, batch_size=batch_size,
                          shuffle=True, num_workers=2, pin_memory=False)
    print(f"Dataset: {len(dataset)} samples from {hdf5_dir}")

    # ── Model ─────────────────────────────────────────────────────────────
    # StudentEncoder lives on NPU — we receive its 256-D output as input
    # For training on CPU, mock with a linear projection
    from train_mvtec import StudentEncoder as _SE
    student_mock = _SE().to(device)
    student_mock.eval()
    print("NOTE: Using StudentEncoder (random weights) for smoke test.")

    model = CortexWorldModel(
        d_model=128, K=16, n_layers=2, n_heads=4,
        n_experts=4, k_active=2, context_dim=32,
    ).to(device)

    p_counts = model.total_params()
    print("\nParameter budget:")
    for k, v in p_counts.items():
        print(f"  {k:20s}: {v:>8,}")
    print()

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(student_mock.parameters()),
        lr=base_lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs * len(loader)
    )

    # ── Training ───────────────────────────────────────────────────────────
    global_step = 0
    best_loss   = float("inf")

    for epoch in range(n_epochs):
        model.train()
        epoch_losses = []

        for batch in loader:
            frame_t  = batch["frame_t"].to(device)   # (B, 3, 224, 224)
            frame_t1 = batch["frame_t1"].to(device)
            action   = batch["action"].to(device)     # (B, 9) padded
            gps      = batch["gps"].to(device)        # (B, 2)
            domain_id = batch["domain_id"].to(device) # (B,)

            B = frame_t.shape[0]

            # ── Context path (grad flows) ──────────────────────────────────
            z_t  = student_mock(frame_t) if hasattr(student_mock, "features") else F.normalize(student_mock(frame_t.reshape(B,-1).narrow(1,0,min(150528,frame_t.reshape(B,-1).shape[1]))), dim=-1)   # (B, 256)
            z_t1_ctx = student_mock(frame_t1) if hasattr(student_mock, "features") else F.normalize(student_mock(frame_t1.reshape(B,-1).narrow(1,0,min(150528,frame_t1.reshape(B,-1).shape[1]))), dim=-1)

            particles_t,  pos_t,  spectral_t,  _ = model.encode(z_t)
            particles_t1, pos_t1, spectral_t1, _ = model.encode(z_t1_ctx)

            # ── Target path (stop-gradient — JEPA) ────────────────────────
            with torch.no_grad():
                z_t1_tgt       = student_mock(frame_t1) if hasattr(student_mock, "features") else F.normalize(student_mock(frame_t1.reshape(B,-1).narrow(1,0,min(150528,frame_t1.reshape(B,-1).shape[1]))), dim=-1)
                p_t1_target, _, _, _ = model.encode(z_t1_tgt)
                z_target           = p_t1_target.detach()   # ← stop-gradient

            # ── Neuromodulators ────────────────────────────────────────────
            signals = neuro.update(
                z_pred           = particles_t.mean(dim=1),
                z_actual         = z_target.mean(dim=1),
                rho              = 0.5,   # replace w/ live Allen spike rate
                action_magnitude = action.norm(dim=-1).mean().item(),
            )
            loss_fn.update_from_neuro(signals)
            config = RegimeGatedTrainer.get_training_config(signals)

            # ── THICK context init ─────────────────────────────────────────
            context_h = model.thick_gru.init_context(B, device)

            # ── Predictor forward ──────────────────────────────────────────
            out = model.predict(
                particles=particles_t,
                action=action,
                context_h=context_h,
                positions=pos_t,
                domain_id=domain_id,
                regime=signals["regime"],
                ach=signals["ach"],
            )

            z_pred      = out["z_pred"]        # (B, K, 128)
            signed_dist = out["signed_dist"]   # (B, K, K)
            gps_pred    = out["gps_pred"]      # (B, 2)
            moe_aux     = out["moe_aux_loss"]

            # ── JEPA prediction loss (DA-modulated, MoE auxiliary) ─────────
            L_jepa, jepa_stats = jepa_moe_loss(
                z_pred       = z_pred,
                z_target     = z_target,
                moe_aux_loss = moe_aux,
                neuro_da_eff = signals["da_effective"],
                lambda_aux   = 0.01,
            )

            # ── Full neuromodulated loss ───────────────────────────────────
            L_total, neuro_stats = loss_fn(
                z_pred             = z_pred,
                z_target           = z_target,
                signed_dist        = signed_dist,
                particle_positions = pos_t,
                gps_pred           = gps_pred,
                gps_target         = gps,
            )

            # Contact head supervised loss (ACh-gated, already computed)
            L_contact_sup = out["contact_loss"]

            L_jepa      = torch.nan_to_num(L_jepa,      0.0)
            L_jepa        = torch.nan_to_num(L_jepa,        0.0)
            L_total       = torch.nan_to_num(L_total,       0.0)
            L_contact_sup = torch.nan_to_num(L_contact_sup, 0.0)
            total_loss    = L_jepa + L_total + L_contact_sup
            # ── Optimise ───────────────────────────────────────────────────
            optimizer.zero_grad()
            total_loss.backward()

            # Regime-gated gradient clip
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["gradient_clip"]
            )

            # Regime-gated learning rate
            for pg in optimizer.param_groups:
                pg["lr"] = base_lr * config["lr_multiplier"]

            optimizer.step()
            scheduler.step()

            epoch_losses.append(total_loss.item())
            global_step += 1

            # ── Logging ────────────────────────────────────────────────────
            if global_step % log_every == 0:
                usage = log_expert_utilisation(
                    model.predictor, particles_t, action
                )
                print(
                    f"[ep{epoch:02d} step{global_step:05d}] "
                    f"loss={total_loss.item():.4f} "
                    f"L_jepa={L_jepa.item():.4f} "
                    f"L_contact={L_contact_sup.item():.4f} "
                    f"regime={signals['regime']:10s} "
                    f"DA={signals['da']:.3f} "
                    f"5HT={signals['sht']:.3f} "
                    f"Ado={signals['ado']:.3f} "
                    f"lr_x={config['lr_multiplier']:.2f} "
                    f"experts={usage}"
                )

        # ── Epoch checkpoint ────────────────────────────────────────────────
        mean_loss = np.mean(epoch_losses)
        print(f"\nEpoch {epoch:02d} mean_loss={mean_loss:.4f}")

        if mean_loss < best_loss:
            best_loss = mean_loss
            ckpt_path = Path(save_dir) / "cwm_best.pt"
            torch.save({
                "epoch":     epoch,
                "loss":      best_loss,
                "model":     model.state_dict(),
                "signals":   signals,
            }, ckpt_path)
            print(f"  → Saved best checkpoint: {ckpt_path}")

        # Expert specialisation check (every 5 epochs)
        if epoch % 5 == 0 and epoch > 0:
            _check_expert_specialisation(model, loader, device, epoch)

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    return model


# ═══════════════════════════════════════════════════════════════════════════
# Expert Specialisation Diagnostic
# ═══════════════════════════════════════════════════════════════════════════

def _check_expert_specialisation(
    model:   CortexWorldModel,
    loader:  DataLoader,
    device:  torch.device,
    epoch:   int,
    n_batches: int = 10,
):
    """
    Verify domain → expert routing is emerging.

    After ~10 epochs: expect RECON to consistently route to Expert 1 (nav).
    After ~30 epochs: clear specialisation across all 4 domains.
    """
    model.eval()
    domain_expert_counts = {d: torch.zeros(4) for d in DOMAIN_IDS}

    from train_mvtec import StudentEncoder as _SE2
    student_mock = _SE2().to(device)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            frame_t  = batch["frame_t"].to(device)
            action   = batch["action"].to(device)
            domain   = batch["domain"][0]

            B = frame_t.shape[0]
            z = student_mock(frame_t.reshape(B, -1))
            particles, pos, _, _ = model.encode(z)

            # Get router probabilities from first MoE layer
            layer0 = model.predictor.layers[0]
            x_flat = particles.reshape(-1, model.d_model)
            logits = layer0.moe_ffn.router(x_flat)
            probs  = F.softmax(logits, dim=-1)
            top1   = probs.argmax(dim=-1)  # (B*K,)

            for e in range(4):
                domain_expert_counts[domain][e] += (top1 == e).float().sum()

    print(f"\n[Epoch {epoch}] Expert specialisation:")
    print(f"  {'Domain':12s}  Expert0  Expert1  Expert2  Expert3")
    for domain, counts in domain_expert_counts.items():
        total = counts.sum().item()
        if total == 0:
            continue
        pcts = [f"{100*c/total:6.1f}%" for c in counts]
        print(f"  {domain:12s}  {'  '.join(pcts)}")

    model.train()


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════

def smoke_test():
    """Verify all components connect without shape errors."""
    print("Running smoke test...")
    device = torch.device("cpu")
    B, K, D = 4, 16, 128

    model = CortexWorldModel(d_model=D, K=K).to(device)

    # Fake inputs
    z_student   = torch.randn(B, 256)        # StudentEncoder output
    action      = torch.randn(B, MAX_ACTION_DIM)
    gps_target  = torch.randn(B, 2)
    domain_id   = torch.zeros(B, dtype=torch.long)  # RECON

    # Encode
    particles, positions, spectral, moe_w = model.encode(z_student)
    assert particles.shape  == (B, K, D),  f"particles: {particles.shape}"
    assert positions.shape  == (B, K, 2),  f"positions: {positions.shape}"
    assert moe_w.shape      == (B, 4),     f"moe_weights: {moe_w.shape}"

    # Target (stop-grad)
    z_t1 = torch.randn(B, 256)
    p_t1, _, _, _ = model.encode(z_t1)
    z_target = p_t1.detach()

    # Predict
    context_h = model.thick_gru.init_context(B, device)
    out = model.predict(particles, action, context_h, positions, domain_id)

    assert out["z_pred"].shape      == (B, K, D),    f"z_pred: {out['z_pred'].shape}"
    assert out["signed_dist"].shape == (B, K, K),    f"signed_dist: {out['signed_dist'].shape}"
    assert out["gps_pred"].shape    == (B, 2),       f"gps_pred: {out['gps_pred'].shape}"
    assert out["context_h"].shape   == (B, 32),      f"context_h: {out['context_h'].shape}"

    # Loss
    neuro   = NeuromodulatorState(session_start=time.time())
    loss_fn = NeuromodulatedCWMLoss()
    signals = neuro.update(
        particles.mean(1), z_target.mean(1), rho=0.5, action_magnitude=0.1
    )
    loss_fn.update_from_neuro(signals)

    L_jepa, _ = jepa_moe_loss(
        out["z_pred"], z_target, out["moe_aux_loss"],
        neuro_da_eff=signals["da_effective"]
    )
    L_neuro, _ = loss_fn(out["z_pred"], z_target,
                          signed_dist=out["signed_dist"],
                          particle_positions=positions,
                          gps_pred=out["gps_pred"], gps_target=gps_target)

    total = L_jepa + L_neuro + out["contact_loss"]
    total.backward()

    p = model.total_params()
    print(f"\n  Shapes:  OK")
    print(f"  Loss:    {total.item():.4f}")
    print(f"  Params:  {p['total']:,} total")
    print(f"  Regime:  {signals['regime']}")
    print(f"\nSmoke test passed.")


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train CORTEX World Model")
    parser.add_argument("--smoke",      action="store_true",
                        help="Run smoke test only (no data needed)")
    parser.add_argument("--hdf5-dir",   default="recon_data/recon_release")
    parser.add_argument("--student",    default=r"checkpoints\dinov2_student\student_best.pt")
    parser.add_argument("--epochs",     type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--max-files",  type=int, default=None)
    parser.add_argument("--save-dir",   default=r"checkpoints\cwm")
    parser.add_argument("--log-every",  type=int, default=50)
    args = parser.parse_args()

    if args.smoke:
        smoke_test()
    else:
        train_cwm(
            hdf5_dir    = args.hdf5_dir,
            student_ckpt= args.student,
            n_epochs    = args.epochs,
            batch_size  = args.batch_size,
            base_lr     = args.lr,
            max_files   = args.max_files,
            save_dir    = args.save_dir,
            log_every   = args.log_every,
        )

