import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# [SIGReg] Representation Stabilization Loss
# Epps-Pulley Gaussianity-inspired. Maximizes variance across dimensions and
# penalizes off-diagonal covariance to prevent dimensional collapse during
# continuous online learning on noisy Neuropixels data.
# =============================================================================
def sigreg_loss(z1, *args, K: int = 32, seed: int = 42, eps: float = 1e-4, **kwargs):
    """Weak-SIGReg K=32  (Akbar, arXiv:2603.05924, ICLR 2026 Workshop GRaM).

    Replaces VICReg-style sigreg_loss after controlled ablation:
        VICReg-style  distill@3000 = 0.7240  (baseline)
        Weak-SIGReg   distill@3000 = 0.7055  (-2.6% improvement)
        Strong SIGReg distill@3000 = 0.7097  (-2.0% improvement)

    Why Weak beats Strong for maze tasks:
        LeWM (arXiv:2603.19312) warns that full Epps-Pulley SIGReg
        overconstrained low-intrinsic-dimensionality environments.
        Weak-SIGReg's covariance-only constraint on K=32 random projections
        matches the effective intrinsic dimensionality of the 32-D backbone,
        freeing gradient budget for distillation.

    API: backward-compatible with original (z1, z2) signature.
        z2 (and any extra args) are accepted but ignored.
        The sketch matrix S is fixed per seed for deterministic training.

    Args:
        z1: Latent embeddings, shape (batch, dim). Primary input.
        *args: Extra positional args ignored (z2 compat).
        K: Number of random sketch directions (default 32).
        seed: Random seed for sketch matrix (default 42, fixed per run).
        eps: Unused, kept for API compat.
    Returns:
        Scalar covariance isotropy loss.
    """
    z = z1
    D = z.shape[1]
    torch.manual_seed(seed)
    S = torch.randn(D, K, device=z.device) / (K ** 0.5)   # (D, K)
    S = S / S.norm(dim=0, keepdim=True)                    # unit columns
    sk = z @ S                                              # (N, K) sketch
    sk_c = sk - sk.mean(dim=0)                             # centre
    cov = (sk_c.T @ sk_c) / (z.shape[0] - 1)              # (K, K)
    # Frobenius norm: ||cov - I||_F^2 / K
    return (cov - torch.eye(K, device=z.device)).pow(2).sum() / K


def temporal_straightening_loss(
    z_t: torch.Tensor,
    z_t1: torch.Tensor,
    z_t2: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Penalizes curvature along a 3-step latent trajectory triplet.

    Args:
        z_t:  Latent at step t,   shape (batch, dim)
        z_t1: Latent at step t+1, shape (batch, dim)
        z_t2: Latent at step t+2, shape (batch, dim)
        eps:  Numerical stability floor for norm division.

    Returns:
        Scalar curvature loss. Zero = perfectly straight. One = 90-degree turn.
    """
    v_t  = z_t1 - z_t    # velocity vector t → t+1
    v_t1 = z_t2 - z_t1   # velocity vector t+1 → t+2

    # Cosine similarity between consecutive velocity vectors
    dot   = (v_t * v_t1).sum(dim=-1)
    norm  = v_t.norm(dim=-1) * v_t1.norm(dim=-1) + eps
    cos_C = dot / norm

    # L_curv = 1 - C  (per Eq. 6 in Wang et al. 2026)
    return (1.0 - cos_C).mean()


# =============================================================================
# [EB-JEPA] Bottleneck Projector
# Required for SIGReg to function: the bottleneck (128→512→128) forces the
# model to learn a maximally informative compressed representation before
# regularization is applied.
# =============================================================================
class ProjectorBottleneck(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_dim: int = 512, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# [GHM] Geometric Horizon Predictor (Jumpy World Model)
# Conditions predictions on a temporal horizon scalar (gamma), allowing the
# BoK planner to evaluate macro-saccades at arbitrary future distances
# without computing each intermediate step.
# =============================================================================
class GeometricHorizonPredictor(nn.Module):
    def __init__(self, latent_dim: int = 128, action_dim: int = 2):
        super().__init__()
        # +1 for the temporal horizon scalar gamma
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + action_dim + 1, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim),
        )

    def forward(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        gamma: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """
        Args:
            z:      Current latent state (batch, 128)
            action: Proposed saccade vector (batch, 2)
            gamma:  Temporal horizon scalar — 1=local, 5=jumpy macro-saccade
        Returns:
            Predicted next latent state (batch, 128)
        """
        if isinstance(gamma, (int, float)):
            gamma = torch.full((z.shape[0], 1), float(gamma), device=z.device)
        x = torch.cat([z, action, gamma], dim=-1)
        # Residual connection preserves identity when prediction delta is small
        return z + self.predictor(x)


# =============================================================================
# [C-JEPA] Causal-JEPA Predictor (Compact Inference Path)
# Object-centric compression for fast BoK sampling on the Ryzen CPU.
# 128-D latents → 16-D object centroids → 1.02% feature footprint.
# Allows num_candidates to scale from 5 → 100 within the same 33ms window.
# =============================================================================
class CJEPAPredictor(nn.Module):
    def __init__(self, latent_dim: int = 128, aux_dim: int = 2):
        super().__init__()
        self.compressor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 16),
        )
        self.causal_bridge = nn.Linear(16 + aux_dim, 64)
        self.predictor = nn.Sequential(
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.Linear(128, 16),
        )

    def forward(
        self,
        z_entities: torch.Tensor,
        u_action: torch.Tensor,
        proprioception: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_entities:    (num_samples, num_objects, 128)
            u_action:      action tensor (unused directly; routing via proprioception)
            proprioception: (num_samples, aux_dim) — camera velocity + arousal scalar
        Returns:
            Predicted compressed future states (num_samples, num_objects, 16)
        """
        z_c = self.compressor(z_entities)
        combined = torch.cat(
            [z_c, proprioception.unsqueeze(1).expand(-1, z_c.size(1), -1)],
            dim=-1,
        )
        h = self.causal_bridge(combined)
        return self.predictor(h)


# =============================================================================
# CORTEX-16 Unified Training Module
# Combines SIGReg + Temporal Straightening into a single training objective:
#
#   L_total = L_pred + λ_sigreg * L_sigreg + λ_curv * L_curv
#
# λ_curv = 1.0 is the default from Wang et al. 2026 ablations (Table 3).
# λ_sigreg = 1.0 matches standard VICReg coefficient practice.
# =============================================================================
class CORTEX16LatentSystem(nn.Module):
    def __init__(self, latent_dim: int = 128, action_dim: int = 2):
        super().__init__()
        self.predictor  = GeometricHorizonPredictor(latent_dim, action_dim)
        self.projector  = ProjectorBottleneck(latent_dim, hidden_dim=512, output_dim=latent_dim)

    def forward(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        gamma: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        return self.predictor(z, action, gamma)

    def compute_loss(
        self,
        z_t:  torch.Tensor,
        z_t1: torch.Tensor,
        z_t2: torch.Tensor,
        action: torch.Tensor,
        gamma: float = 1.0,
        lambda_sigreg: float = 1.0,
        lambda_curv: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """
        Full CORTEX-16 training loss for one trajectory triplet (t, t+1, t+2).

        L_total = L_pred + λ_sigreg * L_sigreg + λ_curv * L_curv

        Args:
            z_t, z_t1, z_t2: Consecutive encoded latents from NPU encoder
            action:           Action taken at step t
            gamma:            Temporal horizon scalar
            lambda_sigreg:    Weight for SIGReg collapse prevention
            lambda_curv:      Weight for trajectory straightening (Wang et al. 2026)

        Returns:
            Dict with individual loss components and 'total' for optimizer.step().
        """
        # 1. Prediction loss — stop-gradient on target prevents collapse
        z_pred = self.predictor(z_t, action, gamma)
        l_pred = F.mse_loss(z_pred, z_t1.detach())

        # 2. SIGReg — enforce Gaussianity across the 128-D semantic axes
        z_proj_t  = self.projector(z_t)
        z_proj_t1 = self.projector(z_t1)
        l_sigreg  = sigreg_loss(z_proj_t, z_proj_t1)

        # 3. Temporal Straightening — enforce linear velocity in latent space
        #    Uses raw (unprojected) latents so the encoder geometry is directly shaped.
        #    Ref: Wang et al. 2026, Eq. 6-7 (arXiv:2603.12231)
        l_curv = temporal_straightening_loss(z_t, z_t1, z_t2)

        l_total = l_pred + lambda_sigreg * l_sigreg + lambda_curv * l_curv

        return {
            "pred":    l_pred,
            "sigreg":  l_sigreg,
            "curv":    l_curv,
            "total":   l_total,
        }


# =============================================================================
# [V-JEPA 2.1] Dense Prediction Loss
# Mur-Labadia et al. 2026, "V-JEPA 2.1: Unlocking Dense Features in Video
# Self-Supervised Learning" (arXiv:2603.14482).
#
# Standard JEPA predictors only apply loss to masked/predicted tokens.
# V-JEPA 2.1 shows this causes visible (context) tokens to discard local
# spatial structure and collapse into global aggregators — exactly what
# produces the CORTEX-PE "position not encoded" failure on UMaze.
#
# Fix: add a weighted context loss on the visible (current) tokens so
# the encoder is explicitly penalised for losing spatial structure.
#
# For CORTEX-PE with global 128-D latents (Option 1):
#   - "masked" token  = z_next prediction target
#   - "context" token = z_curr (the current visible state)
#   L_dense = L_pred(z_pred, sg(z_next)) + λ_ctx * L_ctx(z_curr, sg(z_tgt))
#
# For Option 4 spatial tokens (196 patches × 8ch):
#   Both predicted and context patch tokens are penalised.
#   This directly enforces spatial grounding at the patch level.
#
# λ_ctx default=0.1 (conservative — 0.5 from V-JEPA 2.1 is not confirmed
# in public sources; tune empirically per domain).
# =============================================================================
def context_loss(
    z_curr: torch.Tensor,
    z_curr_target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    V-JEPA 2.1 context loss on visible (current) tokens.
    Forces encoder to maintain precise spatial structure in observed frames.

    Uses smooth L1 (Huber) loss — more robust than MSE for spatial features
    with occasional large errors from fast-moving objects.

    Args:
        z_curr:        (B, D) or (B, N, D) current state encoding
        z_curr_target: (B, D) or (B, N, D) stop-gradient target (EMA or sg())
        reduction:     'mean' | 'sum'

    Returns:
        Scalar context loss.
    """
    return F.smooth_l1_loss(z_curr, z_curr_target.detach(), reduction=reduction)


def dense_prediction_loss(
    z_pred:        torch.Tensor,
    z_next_target: torch.Tensor,
    z_curr:        torch.Tensor,
    z_curr_target: torch.Tensor,
    lambda_ctx:    float = 0.1,
) -> dict:
    """
    V-JEPA 2.1 Dense Predictive Loss combining prediction and context losses.

    L_dense = L_pred(z_pred, sg(z_next)) + λ_ctx * L_ctx(z_curr, sg(z_curr_tgt))

    Args:
        z_pred:        (B, D) predicted next state from predictor
        z_next_target: (B, D) ground truth next state (stop-gradient)
        z_curr:        (B, D) current state encoding from encoder
        z_curr_target: (B, D) current state target for context supervision
                       (can be same as z_curr for self-supervision, or EMA encoder)
        lambda_ctx:    weight for context loss (default 0.1, tune per domain)

    Returns:
        dict with keys: pred, ctx, total
    """
    l_pred = F.mse_loss(z_pred, z_next_target.detach())
    l_ctx  = context_loss(z_curr, z_curr_target)
    l_total = l_pred + lambda_ctx * l_ctx
    return {"pred": l_pred, "ctx": l_ctx, "total": l_total}


# =============================================================================
# [V-JEPA 2.1] Deep Self-Supervision Loss
# Hierarchical distillation applied at all 4 CNN block outputs.
# Forces each intermediate representation to remain semantically grounded.
#
# For CORTEX-PE's 4-block CNN backbone:
#   Block 1: (B, 16) → project to teacher_dim
#   Block 2: (B, 32) → project to teacher_dim
#   Block 3: (B, 64) → project to teacher_dim
#   Block 4: (B, 32) → project to teacher_dim  (existing backbone_g)
#
# Loss: weighted sum of cosine distillation at each level.
# Deeper layers weighted more heavily (closer to final representation).
# Default weights: [0.25, 0.5, 0.75, 1.0] — linear ramp toward output.
# =============================================================================
def deep_supervision_loss(
    intermediates:   list,
    projectors:      list,
    teacher_target:  torch.Tensor,
    level_weights:   list = None,
) -> torch.Tensor:
    """
    V-JEPA 2.1 Deep Self-Supervision across all CNN block outputs.

    Args:
        intermediates:  list of 4 tensors [(B,16),(B,32),(B,64),(B,32)]
                        from CortexCNNBackbone(return_intermediates=True)
        projectors:     list of 4 nn.Linear/MLP projectors mapping each
                        intermediate to teacher_dim (384 for DINOv2-small)
        teacher_target: (B, teacher_dim) stop-gradient teacher features
        level_weights:  per-level loss weights (deeper = higher weight)
                        default: [0.25, 0.5, 0.75, 1.0]

    Returns:
        Scalar weighted deep supervision loss.
    """
    if level_weights is None:
        level_weights = [0.25, 0.5, 0.75, 1.0]

    assert len(intermediates) == len(projectors) == len(level_weights), \
        "intermediates, projectors, and level_weights must have same length"

    tgt = teacher_target.detach()
    tgt_norm = F.normalize(tgt, dim=-1)

    total = torch.tensor(0.0, requires_grad=True)
    for feat, proj, w in zip(intermediates, projectors, level_weights):
        pred_norm = F.normalize(proj(feat), dim=-1)
        cos_loss  = (1.0 - F.cosine_similarity(pred_norm, tgt_norm, dim=-1)).mean()
        total = total + w * cos_loss

    return total / sum(level_weights)  # normalise so total ≈ single distill loss
