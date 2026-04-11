"""
train_predictor.py — CORTEX-16 Predictor Training

Four predictor architectures:

  Option 1 — GeometricHorizonPredictor
    Global 128-D MLP, γ-horizon conditioned. Fast baseline.
    Collapses on MuJoCo (frames too similar at frameskip=1).

  Option 2 — CausalTransformerPredictor
    Global 128-D, K=3 history, causal Transformer.
    Matches Wang et al. 2026 protocol exactly.

  Option 3 — SpatialMLPPredictor  (Path A)
    14×14×8 = 1568-D spatial features, residual MLP.
    Preserves spatial structure. No temporal history.
    Collapses on MuJoCo despite spatial preservation.

  Option 4 — SpatialTransformerPredictor  (Path A+B — recommended)
    14×14×8 spatial features treated as 196 patch tokens.
    K=3 frame history. Causal cross-frame attention.
    Action encoded per-patch via broadcast.
    Matches Wang et al. ViT predictor architecture applied to
    CORTEX-16 spatial features instead of DINOv2 patch features.
    ~2.1M params. The publishable comparison predictor.

Training protocol (Wang et al. 2026 Table 4):
  Predictor lr:  5e-4  |  Batch: 32  |  Encoder: frozen

Frameskip:
  frameskip=1  consecutive frames (may collapse on subtle-motion envs)
  frameskip=5  Wang et al. standard — larger displacement, harder prediction

Usage:
  # Option 4 — spatial Transformer (recommended for benchmark)
  python train_predictor.py --encoder ./checkpoints/cortex_student_phase2_final.pt --env wall --option 4
  python train_predictor.py --encoder ./checkpoints/cortex_student_phase2_final.pt --env umaze --option 4 --frameskip 5
  python train_predictor.py --encoder ./checkpoints/cortex_student_phase2_final.pt --env medium --option 4 --frameskip 5

  # Option 3 — spatial MLP baseline
  python train_predictor.py --encoder ./checkpoints/cortex_student_phase2_final.pt --env wall --option 3
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from student_encoder import StudentEncoder
from latent_predictor import GeometricHorizonPredictor


# =============================================================================
# Environment configuration
# =============================================================================
ENV_CONFIG = {
    "wall":   {"n_train": 1920,  "steps": 50,  "epochs": 3,  "action_dim": 2},
    "umaze":  {"n_train": 2000,  "steps": 100, "epochs": 3,  "action_dim": 2},
    "medium": {"n_train": 4000,  "steps": 100, "epochs": 20, "action_dim": 2},
    "pusht":  {"n_train": 18500, "steps": 200, "epochs": 2,  "action_dim": 2},
    "recon":  {"n_train": 184,   "steps": 50,  "epochs": 3,  "action_dim": 2},
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

FRAME_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Spatial configuration
DV          = 8                      # channels per patch (Wang et al. optimal)
N_PATCHES   = 14 * 14                # 196 patches from 14×14 backbone grid
SPATIAL_H   = 14
SPATIAL_W   = 14
SPATIAL_DIM = N_PATCHES * DV        # 1568


# =============================================================================
# Shared: Spatial Channel Projector (options 3 and 4)
# 1×1 conv: 32 backbone channels → dv=8 planning channels
# Preserves 14×14 spatial grid. Trained with predictor, not exported to NPU.
# =============================================================================
class SpatialChannelProjector(nn.Module):
    """
    32→8 channel projection via 1×1 conv, preserving 14×14 spatial structure.

    Input:  (B, 32, 14, 14)
    Output: (B, 196, 8)  as patch token sequence  (use_tokens=True)
            (B, 1568)    flattened                 (use_tokens=False, default)
    """
    def __init__(self, in_channels: int = 32, dv: int = DV):
        super().__init__()
        self.dv   = dv
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, dv, kernel_size=1, bias=False),
            nn.BatchNorm2d(dv),
            nn.ReLU(inplace=True),
        )

    def forward(self, spatial: torch.Tensor,
                use_tokens: bool = False) -> torch.Tensor:
        """
        Args:
            spatial:    (B, 32, 14, 14) backbone spatial map
            use_tokens: if True return (B, 196, 8) patch tokens
                        if False return (B, 1568) flattened (default)
        """
        out = self.proj(spatial)               # (B, 8, 14, 14)
        if use_tokens:
            B, C, H, W = out.shape
            return out.permute(0, 2, 3, 1).reshape(B, H * W, C)   # (B, 196, 8)
        return out.flatten(1)                  # (B, 1568)


# =============================================================================
# Spatial Pooling Head — for curvature loss on spatial features
# Wang et al. 2026 B.5: learnable [agg] pooling outperforms patch/mean/flatten
# h_phi: R^{196×8} -> R^{128}  — used in train_distillation.py Phase 2
# =============================================================================
class SpatialPoolingHead(nn.Module):
    """
    Learnable pooling head that aggregates 196 spatial patch tokens to a
    single 128-D global vector for curvature loss computation.

    Wang et al. ablation (Figure 11): [agg] variant consistently outperforms
    patch-level, mean-pooled, and flattened variants across all environments.
    The intuition: straightening should act on global trajectory representations,
    not local patch-level variations.

    Used in train_distillation.py Phase 2:
        L_curv = 1 - cos(h_phi(v_t), h_phi(v_{t+1}))
        where v_t = z_t+1 - z_t are latent velocity vectors

    Parameters: ~50K  (negligible)
    Wang et al. setting: lambda=0.1 for agg (vs 0.01 for others)
    """
    def __init__(self, n_patches: int = N_PATCHES, dv: int = DV, out_dim: int = 128):
        super().__init__()
        self.n_patches = n_patches
        self.dv        = dv
        # Per-patch attention weight
        self.attn = nn.Sequential(
            nn.Linear(dv, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )
        # Channel projection to output dim
        self.proj = nn.Linear(dv, out_dim)

    def forward(self, z_spatial: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_spatial: (B, N_patches, dv)  or  (B, 1568) flattened
        Returns:
            pooled: (B, out_dim)
        """
        if z_spatial.dim() == 2:
            # Unflatten if needed
            z_spatial = z_spatial.reshape(-1, self.n_patches, self.dv)
        weights = self.attn(z_spatial).softmax(dim=1)   # (B, N_patches, 1)
        pooled  = (weights * z_spatial).sum(dim=1)       # (B, dv)
        return self.proj(pooled)                         # (B, out_dim)


# =============================================================================
# Option 3 — Spatial MLP Predictor (Path A)
# =============================================================================
class SpatialMLPPredictor(nn.Module):
    """
    Predicts next spatial state from current spatial state + action.
    Residual MLP: z_next = z + MLP([z, action])

    Parameters: ~1.87M
    """
    def __init__(self, spatial_dim: int = SPATIAL_DIM, action_dim: int = 2):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.net = nn.Sequential(
            nn.Linear(spatial_dim + action_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, spatial_dim),
        )

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return z + self.net(torch.cat([z, action], dim=-1))


# =============================================================================
# Option 4 — Spatial Transformer Predictor with Block AttnRes
#
# Architecture:
#   Treats spatial features as 196 patch tokens × 8 channels.
#   K=3 frame history → 588-token sequence.
#   Action broadcast per frame across all 196 patches.
#   Frame-level causal masking.
#   Residual output: predicts delta from last observed frame.
#
# Block Attention Residuals (Block AttnRes):
#   Replaces fixed additive residual h_l = h_{l-1} + f(h_{l-1}) with
#   learned softmax attention over past block summaries:
#
#     h = Σ_n softmax(w_l · RMSNorm(V_n)) * V_n
#
#   where V = [block_0, block_1, ..., partial_block_current]
#
#   For n_layers=4 we use Full AttnRes (N=4 blocks, one per layer),
#   which gives maximum compositional reasoning depth.
#   Pseudo-query vectors w_l initialised to zero → uniform attention at init.
#
# Parameters: ~215K (similar to standard version, +~2K for AttnRes projections)
# =============================================================================

class BlockAttnRes(nn.Module):
    """
    Inter-block attention residual connection.

    Attends over completed block summaries plus the current partial block.
    RMSNorm prevents magnitude-dominant blocks from overwhelming softmax.
    Pseudo-query w_l is a learned 1D projection (init=0 → uniform at start).

    Args:
        d_model: feature dimension
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.proj = nn.Linear(d_model, 1, bias=False)
        nn.init.zeros_(self.proj.weight)   # Critical: zero-init → uniform attention

    def forward(self, blocks: list, partial: torch.Tensor) -> torch.Tensor:
        """
        Args:
            blocks:  list of N tensors (B, T, D) — completed block summaries
            partial: (B, T, D) — current intra-block running sum
        Returns:
            (B, T, D) — weighted combination of all blocks
        """
        sources = blocks + [partial]
        V       = torch.stack(sources, dim=0)        # (N+1, B, T, D)
        K       = self.norm(V)                        # normalise
        logits  = torch.einsum(
            'd, n b t d -> n b t',
            self.proj.weight.squeeze(0), K
        )                                             # (N+1, B, T)
        weights = logits.softmax(dim=0)               # (N+1, B, T)
        h       = torch.einsum(
            'n b t, n b t d -> b t d',
            weights, V
        )                                             # (B, T, D)
        return h


class SpatialTransformerLayer(nn.Module):
    """
    Single transformer layer with Block AttnRes on both Attn and MLP sub-layers.
    Maintains (blocks, partial_block) state across layers.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn      = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                               batch_first=True)
        self.ff1        = nn.Linear(d_model, d_model * 4)
        self.ff2        = nn.Linear(d_model * 4, d_model)
        self.attn_norm  = nn.RMSNorm(d_model)
        self.mlp_norm   = nn.RMSNorm(d_model)
        self.attn_res   = BlockAttnRes(d_model)
        self.mlp_res    = BlockAttnRes(d_model)
        self.drop       = nn.Dropout(dropout)

    def forward(
        self,
        partial: torch.Tensor,
        blocks:  list,
        attn_mask: torch.Tensor = None,
    ):
        """
        Args:
            partial:   (B, T, D) intra-block running sum
            blocks:    list of completed block tensors (B, T, D)
            attn_mask: (T, T) causal mask
        Returns:
            (partial, blocks) updated
        """
        # --- Self-Attention sub-layer with Block AttnRes ---
        h        = self.attn_res(blocks, partial)
        x_norm   = self.attn_norm(h)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask,
                                need_weights=False)
        partial  = partial + self.drop(attn_out)

        # Completed one sub-layer → save as new block for next layer
        blocks   = blocks + [partial]

        # --- MLP sub-layer with Block AttnRes ---
        h        = self.mlp_res(blocks, partial)
        x_norm   = self.mlp_norm(h)
        mlp_out  = self.ff2(self.drop(F.gelu(self.ff1(x_norm))))
        partial  = partial + self.drop(mlp_out)

        # Completed second sub-layer → save as new block
        blocks   = blocks + [partial]

        return partial, blocks


class SpatialTransformerPredictor(nn.Module):
    """
    Causal Transformer predictor on 196 spatial patch tokens with K=3 history
    and Block Attention Residuals (Full AttnRes, N=n_layers blocks).

    Input:
        z_history:  (B, K, N_patches, dv)  = (B, 3, 196, 8)
        a_history:  (B, K, action_dim)     = (B, 3, 2)
    Output:
        z_next:     (B, 1568)   flattened predicted next frame patches
    """
    def __init__(
        self,
        n_patches:   int = N_PATCHES,
        dv:          int = DV,
        action_dim:  int = 2,
        history_len: int = 3,
        d_model:     int = 64,
        n_heads:     int = 4,
        n_layers:    int = 4,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.n_patches   = n_patches
        self.dv          = dv
        self.history_len = history_len
        self.d_model     = d_model
        self.seq_len     = history_len * n_patches   # 588

        self.patch_proj   = nn.Linear(dv, d_model)
        self.action_proj  = nn.Linear(action_dim, d_model)
        self.spatial_pos  = nn.Parameter(torch.zeros(1, n_patches, d_model))
        self.temporal_pos = nn.Parameter(torch.zeros(1, history_len, 1, d_model))
        nn.init.trunc_normal_(self.spatial_pos,  std=0.02)
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)

        self.layers = nn.ModuleList([
            SpatialTransformerLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.register_buffer(
            "causal_mask",
            self._build_frame_causal_mask(history_len, n_patches)
        )

        self.output_proj = nn.Sequential(
            nn.RMSNorm(d_model),
            nn.Linear(d_model, dv),
        )

    @staticmethod
    def _build_frame_causal_mask(history_len: int, n_patches: int) -> torch.Tensor:
        seq_len = history_len * n_patches
        mask    = torch.zeros(seq_len, seq_len, dtype=torch.bool)
        for f in range(history_len):
            for f2 in range(f + 1, history_len):
                rows = slice(f2 * n_patches, (f2 + 1) * n_patches)
                cols = slice(f  * n_patches, (f  + 1) * n_patches)
                mask[rows, cols] = True
        return mask

    def forward(
        self,
        z_history: torch.Tensor,
        a_history: torch.Tensor,
    ) -> torch.Tensor:
        B, K, P, C = z_history.shape

        # Project patches and add positional embeddings
        x  = self.patch_proj(z_history)                         # (B, K, P, d_model)
        x  = x + self.spatial_pos.unsqueeze(1)
        x  = x + self.temporal_pos
        a  = self.action_proj(a_history).unsqueeze(2)           # (B, K, 1, d_model)
        x  = x + a

        # Flatten to sequence: (B, K*P, d_model)
        x  = x.reshape(B, K * P, self.d_model)

        # Block AttnRes forward pass
        # blocks list grows as each sub-layer completes
        # Initialise blocks with the token embeddings (b_0 = h_1 per paper)
        blocks  = [x]
        partial = x

        for layer in self.layers:
            partial, blocks = layer(partial, blocks, self.causal_mask)

        # Extract last frame patches and project back to dv
        x_last = partial[:, (K-1)*P : K*P, :]                  # (B, 196, d_model)
        delta  = self.output_proj(x_last)                       # (B, 196, dv)
        z_next = z_history[:, -1, :, :] + delta                 # residual
        return z_next.reshape(B, -1)                            # (B, 1568)

# =============================================================================
# Option 2 — Causal Transformer Predictor (global 128-D, K=3)
# =============================================================================
class CausalTransformerPredictor(nn.Module):
    """
    Causal Transformer on global 128-D latents with K=3 frame history.
    Matches Wang et al. 2026 protocol exactly.
    """
    def __init__(self, latent_dim=128, action_dim=2, d_model=256,
                 n_heads=4, n_layers=4, history_len=3, dropout=0.1):
        super().__init__()
        self.latent_dim  = latent_dim
        self.history_len = history_len
        self.input_proj  = nn.Linear(latent_dim + action_dim, d_model)
        self.pos_embed   = nn.Parameter(torch.zeros(1, history_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(history_len, history_len), diagonal=1).bool()
        )
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, latent_dim),
        )

    def forward(self, z_history: torch.Tensor, a_history: torch.Tensor) -> torch.Tensor:
        B, K, _ = z_history.shape
        x = torch.cat([z_history, a_history], dim=-1)
        x = self.input_proj(x) + self.pos_embed[:, :K, :]
        x = self.transformer(x, mask=self.causal_mask[:K, :K])
        return z_history[:, -1, :] + self.output_proj(x[:, -1, :])


# =============================================================================
# Trajectory Dataset
# Option 4: stores (B, K, 196, 8) patch token history instead of (B, K, 1568)
# =============================================================================
class TrajectoryDataset(Dataset):
    def __init__(self, traj_path, encoder, n_train, history_len=1,
                 frameskip=1, option=1, spatial_proj=None,
                 min_dist=0.0):
        """
        Args:
            min_dist: skip pairs where ||z_t - z_next|| < min_dist
                      (goal relabelling filter — prevents trivial pairs)
        """
        print(f"   Loading trajectories from {traj_path}...")
        raw = np.load(traj_path, allow_pickle=True)[:n_train]
        self.pairs   = []
        self.option  = option
        use_spatial  = option in (3, 4) and spatial_proj is not None
        use_tokens   = option == 4   # return (196, 8) instead of (1568,)

        print(f"   Pre-encoding {len(raw)} trajectories "
              f"({'spatial tokens 196×8' if use_tokens else 'spatial 1568-D' if use_spatial else 'global 128-D'})...")
        encoder.eval()
        if use_spatial:
            spatial_proj.eval()

        skipped_trivial = 0

        with torch.no_grad():
            for traj_idx, traj in enumerate(raw):
                obs     = traj["observations"]
                actions = traj["actions"]
                T       = len(obs)

                frames = torch.stack([FRAME_TRANSFORM(obs[t]) for t in range(T)])

                latents = []
                for i in range(0, T, 32):
                    batch = frames[i:i+32]
                    if use_spatial:
                        _, spatial = encoder(batch, return_spatial=True)
                        if use_tokens:
                            # (batch, 196, 8) patch tokens
                            z = spatial_proj(spatial, use_tokens=True)
                        else:
                            # (batch, 1568) flattened
                            z = spatial_proj(spatial, use_tokens=False)
                    else:
                        z = encoder(batch)   # (batch, 128)
                    latents.append(z)
                latents = torch.cat(latents, dim=0)   # (T, D) or (T, 196, 8)

                for t in range(history_len - 1, T - frameskip):
                    t_next = t + frameskip
                    if t_next >= T:
                        break

                    # Goal relabelling filter — skip trivially close pairs
                    if min_dist > 0:
                        z_curr = latents[t].flatten()
                        z_nxt  = latents[t_next].flatten()
                        dist   = torch.norm(z_curr - z_nxt, p=2).item()
                        if dist < min_dist:
                            skipped_trivial += 1
                            continue

                    if history_len == 1:
                        z_hist = latents[t].unsqueeze(0)
                        a_hist = torch.tensor(
                            actions[t], dtype=torch.float32
                        ).unsqueeze(0)
                    else:
                        hist_i = [max(0, t - (history_len-1-k))
                                  for k in range(history_len)]
                        z_hist = torch.stack([latents[i] for i in hist_i])
                        a_hist = torch.stack([
                            torch.tensor(actions[i], dtype=torch.float32)
                            for i in hist_i
                        ])

                    self.pairs.append((z_hist, a_hist, latents[t_next]))

                if (traj_idx + 1) % 500 == 0:
                    print(f"   Encoded {traj_idx+1}/{len(raw)} "
                          f"({len(self.pairs):,} pairs)")

        if skipped_trivial > 0:
            print(f"   ⚡ Goal filter: skipped {skipped_trivial:,} trivial pairs "
                  f"(||z_t - z_next|| < {min_dist})")

        shape_str = str(tuple(latents[0].shape))
        print(f"   ✅ Dataset: {len(self.pairs):,} pairs | token shape: {shape_str}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


# =============================================================================
# Training
# =============================================================================
def train_predictor(
    encoder_path,
    env_name,
    option,
    data_root    = "./benchmark_data",
    out_dir      = "./predictors",
    batch_size   = 32,
    lr           = 5e-4,
    frameskip    = 1,
    min_dist     = 0.0,
    d_model      = 64,
    n_layers     = 4,
):
    config   = ENV_CONFIG[env_name]
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    option_names = {
        1: "GeometricHorizon MLP (global 128-D)",
        2: "CausalTransformer (global 128-D, K=3)",
        3: f"SpatialMLP (14×14×{DV}={SPATIAL_DIM}-D)",
        4: f"SpatialTransformer (196px×{DV}ch, K=3, d_model={d_model}, n_layers={n_layers})",
    }

    print("\n" + "="*60)
    print(f"  PREDICTOR TRAINING")
    print(f"  Environment: {env_name}")
    print(f"  Option:      {option} — {option_names[option]}")
    print(f"  Frameskip:   {frameskip}")
    print(f"  Min dist:    {min_dist}")
    print(f"  Epochs:      {config['epochs']}")
    print(f"  Encoder:     frozen ({encoder_path})")
    print("="*60 + "\n")

    # Load frozen encoder
    encoder = StudentEncoder()
    ckpt    = torch.load(encoder_path, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    stem_map = {"backbone.stem.0":"backbone.block1.0","backbone.stem.1":"backbone.block1.1",
                "backbone.stem.3":"backbone.block2.0","backbone.stem.4":"backbone.block2.1",
                "backbone.stem.6":"backbone.block3.0","backbone.stem.7":"backbone.block3.1",
                "backbone.stem.9":"backbone.block4.0","backbone.stem.10":"backbone.block4.1"}
    state = {next((k.replace(op,np,1) for op,np in stem_map.items() if k.startswith(op+".")),k):v
             for k,v in state.items()}
    encoder.load_state_dict(state, strict=False)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Build predictor + spatial projector
    spatial_proj = None
    history_K    = 1

    if option == 1:
        predictor = GeometricHorizonPredictor(128, config["action_dim"])

    elif option == 2:
        predictor = CausalTransformerPredictor(128, config["action_dim"], history_len=3)
        history_K = 3

    elif option == 3:
        spatial_proj = SpatialChannelProjector(in_channels=32, dv=DV)
        predictor    = SpatialMLPPredictor(SPATIAL_DIM, config["action_dim"])

    elif option == 4:
        spatial_proj = SpatialChannelProjector(in_channels=32, dv=DV)
        # n_heads must divide d_model evenly
        n_heads_4 = min(4, d_model // 8) or 1
        while d_model % n_heads_4 != 0:
            n_heads_4 -= 1
        predictor    = SpatialTransformerPredictor(
            n_patches   = N_PATCHES,
            dv          = DV,
            action_dim  = config["action_dim"],
            history_len = 3,
            d_model     = d_model,
            n_heads     = n_heads_4,
            n_layers    = n_layers,
        )
        history_K = 3

    n_pred = sum(p.numel() for p in predictor.parameters())
    n_proj = sum(p.numel() for p in spatial_proj.parameters()) if spatial_proj else 0
    print(f"✅ Predictor:  {n_pred:,} params")
    if n_proj:
        print(f"   Projector:  {n_proj:,} params")
    print(f"   Total:      {n_pred + n_proj:,} params\n")

    # Build dataset
    traj_path = Path(data_root) / env_name / "trajectories.npy"
    dataset   = TrajectoryDataset(
        traj_path    = str(traj_path),
        encoder      = encoder,
        n_train      = config["n_train"],
        history_len  = history_K,
        frameskip    = frameskip,
        option       = option,
        spatial_proj = spatial_proj,
        min_dist     = min_dist,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=True,
    )

    # Optimiser
    params = list(predictor.parameters())
    if spatial_proj:
        params += list(spatial_proj.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"] * len(loader)
    )

    loss_log = []
    print(f"Training: {config['epochs']} epochs × {len(loader)} batches\n")

    for epoch in range(config["epochs"]):
        epoch_losses = []
        pred_losses  = []
        cont_losses  = []
        t0 = time.perf_counter()

        for z_hist, a_hist, z_next in loader:
            if option == 1:
                z_pred = predictor(z_hist[:, 0, :], a_hist[:, 0, :], gamma=1.0)

            elif option == 2:
                z_pred = predictor(z_hist, a_hist)

            elif option == 3:
                z_pred = predictor(z_hist[:, 0, :], a_hist[:, 0, :])

            elif option == 4:
                # z_hist: (B, K, 196, 8)  a_hist: (B, K, 2)
                z_pred = predictor(z_hist, a_hist)   # (B, 1568)
                # z_next is (B, 196, 8) — flatten for loss
                z_next = z_next.reshape(z_next.shape[0], -1)

            # MSE prediction loss
            pred_loss = F.mse_loss(z_pred, z_next.detach())

            # V-JEPA 2.1 Dense Context Loss
            # Penalises encoder for losing spatial precision in visible frames.
            # z_curr should reconstruct itself under the predictor — forces
            # the encoder to maintain explicit position/state information.
            # λ_ctx=0.1 (conservative default — tune per domain)
            if option in (1, 2, 3):
                z_curr = z_hist[:, 0, :]   # (B, D) current visible state
                ctx_loss = F.smooth_l1_loss(z_curr, z_curr.detach().roll(1, dims=0))
                # Self-consistency: z_curr must be distinguishable from its
                # batch-shifted version — prevents all states mapping identically
                ctx_loss = F.relu(0.05 - (z_curr - z_curr.roll(1, dims=0)).norm(dim=-1)).mean()
            else:
                ctx_loss = torch.tensor(0.0)

            # Contrastive anti-collapse loss
            # Penalises z_pred ≈ z_current when action magnitude is large.
            # Directly prevents the predictor learning the identity mapping.
            # Only applies to options 1/2/3 (global or flat spatial features).
            # For option 4 z_hist has shape (B, K, P, dv) — use last frame.
            if option in (1, 2, 3):
                z_curr_flat = z_hist[:, 0, :].flatten(1)
                z_pred_flat = z_pred.flatten(1)
                action_mag  = a_hist[:, 0, :].norm(dim=-1)            # (B,)
                z_disp      = (z_pred_flat - z_curr_flat).norm(dim=-1) # (B,)
                # Fire when: action is large but displacement is small
                contrastive = F.relu(0.5 * action_mag - z_disp).mean()
            elif option == 4:
                z_curr_flat = z_hist[:, -1, :, :].flatten(1)
                z_pred_flat = z_pred.flatten(1)
                action_mag  = a_hist[:, -1, :].norm(dim=-1)
                z_disp      = (z_pred_flat - z_curr_flat).norm(dim=-1)
                contrastive = F.relu(0.5 * action_mag - z_disp).mean()
            else:
                contrastive = torch.tensor(0.0)

            loss = pred_loss + 0.1 * contrastive + 0.1 * ctx_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            epoch_losses.append(loss.item())
            pred_losses.append(pred_loss.item())
            cont_losses.append(contrastive.item() if hasattr(contrastive, 'item') else 0.0)

        epoch_avg  = float(np.mean(epoch_losses))
        pred_avg   = float(np.mean(pred_losses))
        cont_avg   = float(np.mean(cont_losses))
        loss_log.append({"epoch": epoch+1, "loss": epoch_avg,
                         "pred": pred_avg, "contrastive": cont_avg})
        print(f"Epoch {epoch+1:>3}/{config['epochs']} | "
              f"loss={epoch_avg:.6f}  pred={pred_avg:.6f}  "
              f"contrast={cont_avg:.6f}  ({time.perf_counter()-t0:.1f}s)")

    # Save
    save_path = out_path / f"predictor_{env_name}_opt{option}.pt"
    save_dict = {
        "option":      option,
        "env":         env_name,
        "frameskip":   frameskip,
        "state_dict":  predictor.state_dict(),
        "config":      config,
        "loss_log":    loss_log,
        "spatial_dim": SPATIAL_DIM,
        "n_patches":   N_PATCHES,
        "dv":          DV,
    }
    if spatial_proj:
        save_dict["spatial_proj_state_dict"] = spatial_proj.state_dict()

    torch.save(save_dict, save_path)
    print(f"\n💾 Predictor saved → {save_path}")

    with open(out_path / f"loss_{env_name}_opt{option}.json", "w") as f:
        json.dump(loss_log, f, indent=2)

    return predictor, spatial_proj


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CORTEX-16 Predictor Training")
    p.add_argument("--encoder",    required=True)
    p.add_argument("--env",        required=True,
                   choices=["wall", "umaze", "medium", "pusht"])
    p.add_argument("--option",     type=int, required=True, choices=[1, 2, 3, 4])
    p.add_argument("--data",       default="./benchmark_data")
    p.add_argument("--out",        default="./predictors")
    p.add_argument("--batch",      type=int, default=32)
    p.add_argument("--lr",         type=float, default=5e-4)
    p.add_argument("--frameskip",  type=int, default=1,
                   help="Steps between pairs. 1=consecutive, 5=Wang et al.")
    p.add_argument("--min-dist",   type=float, default=0.0,
                   help="Skip pairs with ||z_t - z_next|| < min_dist (goal filter)")
    p.add_argument("--d-model",    type=int, default=64,
                   help="Transformer d_model for option 4. Use 32 for fast training.")
    p.add_argument("--n-layers",   type=int, default=4,
                   help="Transformer n_layers for option 4. Use 2 for fast training.")
    args = p.parse_args()

    train_predictor(
        encoder_path = args.encoder,
        env_name     = args.env,
        option       = args.option,
        data_root    = args.data,
        out_dir      = args.out,
        batch_size   = args.batch,
        lr           = args.lr,
        frameskip    = args.frameskip,
        min_dist     = args.min_dist,
        d_model      = args.d_model,
        n_layers     = args.n_layers,
    )
