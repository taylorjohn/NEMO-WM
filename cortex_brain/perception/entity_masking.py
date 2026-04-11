"""
cortex_brain.perception.entity_masking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
C-JEPA Entity Trajectory Masking for Object Permanence.

Problem with standard JEPA masking
------------------------------------
Random patch masking trains the predictor to texture-match — it learns
"what colour pixels appear near other colour pixels."  It does NOT learn
physics or object permanence.

C-JEPA solution
---------------
Mask 100% of a specific entity's latent across T consecutive frames.
The predictor must now answer: "where does entity-3 go after it
disappears behind entity-1?"  This forces it to internalise:
  - Object permanence  (hidden ≠ gone)
  - Trajectory continuity  (momentum, collision physics)
  - Entity-entity causality  (push → move)

Architecture
------------
EntityMasker selects the highest-salience entity each window using a
composite score:

    salience(i) = alpha * velocity(i) + beta * surprise(i)

where velocity is estimated from the delta between consecutive latents
and surprise is cosine distance from the TTM prior.

During TRAINING:  selected entity's rows are zeroed → predictor must
                  reconstruct them.
During INFERENCE: pass-through — no masking.

MaskedJEPALoss computes prediction error ONLY on the masked entity
rows, preventing the model from optimising trivially-visible entities.

Usage
-----
    masker  = EntityMasker(EntityMaskConfig(latent_dim=16))
    z_t     = torch.randn(B, N, 16)   # current compressed latents
    z_tm1   = torch.randn(B, N, 16)   # previous compressed latents

    z_masked, mask_info = masker(z_t, z_tm1)
    z_pred              = predictor(z_masked, u, prop)
    loss                = MaskedJEPALoss()(z_pred, z_t, mask_info)
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class EntityMaskConfig:
    latent_dim:        int   = 16      # compressed entity dim (post-compressor)
    num_entities:      int   = 8       # N in (B, N, latent_dim)
    velocity_weight:   float = 0.6     # alpha: how much motion drives salience
    surprise_weight:   float = 0.4     # beta:  how much novelty drives salience
    mask_value:        float = 0.0     # value to fill masked positions
    min_salience:      float = 0.01    # below this, mask randomly instead
    trajectory_frames: int   = 1       # how many frames to mask (1 = current frame)


class EntityMasker(nn.Module):
    """
    Selects the highest-salience entity and masks its latent vector.

    Operates on compressed latents (post CJEPAPredictor.compressor),
    so it sees 16-D vectors rather than 128-D.  This keeps masking
    decisions fast and NPU-compatible.

    Parameters
    ----------
    config       : EntityMaskConfig
    training     : bool  — if False (eval/inference) masking is skipped
    """

    def __init__(self, config: EntityMaskConfig = EntityMaskConfig()) -> None:
        super().__init__()
        self.config = config
        # Learnable mask token — replaces the masked entity's latent
        # (better than zeros: model learns "this is masked" signal)
        self.mask_token = nn.Parameter(
            torch.zeros(config.latent_dim))

    def forward(
        self,
        z_t:   torch.Tensor,                    # (B, N, latent_dim) current
        z_tm1: Optional[torch.Tensor] = None,   # (B, N, latent_dim) previous
        prior: Optional[torch.Tensor] = None,   # (N, latent_dim)    TTM prior
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Parameters
        ----------
        z_t   : current frame entity latents  (B, N, D)
        z_tm1 : previous frame latents        (B, N, D) or None
        prior : long-term memory prior        (N, D)    or None

        Returns
        -------
        z_masked  : (B, N, D)  latents with selected entity replaced by mask_token
        mask_info : dict with keys:
            entity_idx   : int   — which entity was masked
            salience     : float — salience score of masked entity
            was_masked   : bool  — False in eval mode (pass-through)
        """
        B, N, D = z_t.shape

        if not self.training:
            return z_t, {"entity_idx": -1, "salience": 0.0, "was_masked": False}

        salience = self._compute_salience(z_t, z_tm1, prior)  # (B, N)
        # Pick the entity with max salience (averaged over batch)
        mean_sal  = salience.mean(dim=0)                       # (N,)
        entity_idx = int(mean_sal.argmax().item())

        # Replace selected entity rows with the learnable mask token
        z_masked = z_t.clone()
        z_masked[:, entity_idx, :] = self.mask_token.unsqueeze(0).expand(B, -1)

        logger.debug("EntityMasker: masked entity %d  salience=%.4f",
                     entity_idx, float(mean_sal[entity_idx]))

        return z_masked, {
            "entity_idx": entity_idx,
            "salience":   float(mean_sal[entity_idx]),
            "was_masked": True,
        }

    def _compute_salience(
        self,
        z_t:   torch.Tensor,
        z_tm1: Optional[torch.Tensor],
        prior: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Returns (B, N) salience scores."""
        B, N, D = z_t.shape
        cfg = self.config

        # ── Velocity salience: L2 distance between consecutive latents ──
        if z_tm1 is not None:
            vel = (z_t - z_tm1).norm(dim=-1)                   # (B, N)
        else:
            vel = torch.zeros(B, N, device=z_t.device)

        # ── Surprise salience: cosine distance from TTM prior ────────────
        if prior is not None:
            # prior: (N, D) → expand to (B, N, D)
            p = prior.unsqueeze(0).expand(B, -1, -1)
            cos_sim = F.cosine_similarity(z_t, p, dim=-1)       # (B, N)
            surprise = (1.0 - cos_sim).clamp(0.0, 2.0) / 2.0   # [0,1]
        else:
            surprise = torch.rand(B, N, device=z_t.device) * 0.1

        salience = cfg.velocity_weight * vel + cfg.surprise_weight * surprise

        # Normalise to [0, 1] per batch item
        sal_min = salience.min(dim=-1, keepdim=True).values
        sal_max = salience.max(dim=-1, keepdim=True).values
        salience = (salience - sal_min) / (sal_max - sal_min + 1e-8)

        return salience                                          # (B, N)


class MaskedJEPALoss(nn.Module):
    """
    Prediction loss restricted to the masked entity.

    Computing loss over ALL entities would let the model trivially
    minimise by fitting the visible ones.  This loss forces the model
    to earn its gradient only from the hard, masked positions.

        L = (1/B) * sum_b ||z_pred[b, idx, :] - z_target[b, idx, :]||_2

    Parameters
    ----------
    reduction : 'mean' | 'sum'
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        z_pred:    torch.Tensor,   # (B, N, D)  predictor output
        z_target:  torch.Tensor,   # (B, N, D)  ground-truth latents
        mask_info: Dict,
    ) -> torch.Tensor:
        """
        Returns scalar loss.  If was_masked=False (eval/no-op), returns 0.
        """
        if not mask_info.get("was_masked", False):
            return z_pred.new_tensor(0.0)

        idx = mask_info["entity_idx"]
        pred   = z_pred[:, idx, :]    # (B, D)
        target = z_target[:, idx, :]  # (B, D)

        loss = (pred - target).norm(dim=-1)  # (B,)
        return loss.mean() if self.reduction == "mean" else loss.sum()


# ---------------------------------------------------------------------------
# Convenience: build EntityMasker from EBJEPAConfig
# ---------------------------------------------------------------------------

def make_entity_masker(
    num_entities:    int = 8,
    compressed_dim:  int = 16,
    velocity_weight: float = 0.6,
    surprise_weight: float = 0.4,
) -> EntityMasker:
    cfg = EntityMaskConfig(
        latent_dim      = compressed_dim,
        num_entities    = num_entities,
        velocity_weight = velocity_weight,
        surprise_weight = surprise_weight,
    )
    return EntityMasker(cfg)
