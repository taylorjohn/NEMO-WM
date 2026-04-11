"""
rope_temporal_head.py — RoPE-Enhanced TemporalHead for RECON (v2)
==================================================================

v2 fixes the collapse problem from v1:
  - Training: x-prediction with EMA target (no symmetric collapse)
  - Eval:     single-frame embeddings (same as MLP — fair comparison)
  - VICReg:   variance regularisation prevents dead dimensions

Architecture:
    Shared encoder: Linear(128→embed_dim) + LayerNorm  [trains via both paths]
    Pair path:      2-token RoPE self-attention + FFN   [training only]
    Single path:    encoder output only                 [train + eval]

Training (x-pred from UniFluids §3.4):
    pred  = pair_head(z_t, z_tk, c_t, c_tk)   ← learns from geometric context
    target = encoder_ema(z_tk).detach()         ← EMA of shared encoder, no grad
    loss  = cosine_distance(pred, target)       ← x-prediction, not velocity

Eval (same as MLP baseline — fair A/B test):
    emb_t  = encoder(z_t)    ← single frame
    emb_tk = encoder(z_tk)   ← single frame
    dist   = cosine_distance(emb_t, emb_tk)

The RoPE geometry learned during training is baked into the shared encoder
weights — close scenes become more similar in the shared embedding space
because the pair path was trained to predict them from a geometric context.
"""

import math, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# 3D RoPE  (t, x, y)
# ─────────────────────────────────────────────────────────────────────────────

def build_rope_freqs(head_dim: int, n_axes: int = 3) -> torch.Tensor:
    assert head_dim % (2 * n_axes) == 0
    dpa = head_dim // (2 * n_axes)       # dims per axis
    return 1.0 / (10000 ** (torch.arange(0, dpa, dtype=torch.float32) / dpa))


def apply_rope_3d(x: torch.Tensor, coords: torch.Tensor,
                  freqs: torch.Tensor) -> torch.Tensor:
    """x:[...,S,D]  coords:[...,S,3]  freqs:[D/6]"""
    *B, S, D = x.shape
    dpa = freqs.shape[0]
    freqs = freqs.to(x.device)

    theta = torch.cat([
        coords[..., i:i+1] * freqs.unsqueeze(-2)   # [...,S,dpa]
        for i in range(3)
    ], dim=-1)                                        # [...,S,D/2]

    xp = x.reshape(*B, S, D // 2, 2)
    x1, x2 = xp[..., 0], xp[..., 1]
    out = torch.stack([x1*theta.cos() - x2*theta.sin(),
                       x1*theta.sin() + x2*theta.cos()], dim=-1)
    return out.reshape(*B, S, D)


# ─────────────────────────────────────────────────────────────────────────────
# RoPE Attention
# ─────────────────────────────────────────────────────────────────────────────

class RoPEAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.H, self.Hd = num_heads, embed_dim // num_heads
        assert self.Hd % 6 == 0
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out  = nn.Linear(embed_dim, embed_dim, bias=False)
        self.register_buffer("rope_freqs", build_rope_freqs(self.Hd))

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        H, Hd = self.H, self.Hd
        q, k, v = self.qkv(x).reshape(B,S,3,H,Hd).permute(2,0,3,1,4)
        ch = coords.unsqueeze(1).expand(-1,H,-1,-1)
        q  = apply_rope_3d(q, ch, self.rope_freqs)
        k  = apply_rope_3d(k, ch, self.rope_freqs)
        attn = F.softmax((q @ k.transpose(-2,-1)) / math.sqrt(Hd), dim=-1)
        return self.out((attn @ v).transpose(1,2).reshape(B,S,D))


# ─────────────────────────────────────────────────────────────────────────────
# RoPE TemporalHead  (v2 — collapse-free)
# ─────────────────────────────────────────────────────────────────────────────

class RoPETemporalHead(nn.Module):
    """
    Collapse-free RoPE TemporalHead.

    Two modes:
      Pair mode  (training):  head(z_t, z_tk, c_t, c_tk) → prediction of z_tk embedding
      Single mode (eval):     head(z_t)                   → frame embedding

    The shared encoder is trained via both paths. Geometric structure learned
    in pair mode transfers to single-mode embeddings through shared weights.
    """

    def __init__(self, latent_dim=128, embed_dim=96, out_dim=64,
                 num_heads=4, dropout=0.1):
        super().__init__()
        assert embed_dim % (num_heads * 6) == 0

        # Shared encoder (both modes)
        self.encoder = nn.Sequential(
            nn.Linear(latent_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Pair-mode: attention + FFN (training only)
        self.attn    = RoPEAttention(embed_dim, num_heads)
        self.attn_ln = nn.LayerNorm(embed_dim)
        self.ffn     = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(embed_dim * 2, embed_dim),
        )
        self.ffn_ln = nn.LayerNorm(embed_dim)

        # Output projection (both modes)
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

        # EMA target encoder (no gradient — updated externally)
        self.target_encoder = copy.deepcopy(nn.Sequential(
            self.encoder, self.out_proj
        ))
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        self.embed_dim  = embed_dim
        self.out_dim    = out_dim
        self.latent_dim = latent_dim

    @torch.no_grad()
    def update_ema(self, decay: float = 0.996):
        """Call after each optimizer step."""
        src_params  = list(self.encoder.parameters()) + \
                      list(self.out_proj.parameters())
        tgt_params  = list(self.target_encoder.parameters())
        for s, t in zip(src_params, tgt_params):
            t.data.mul_(decay).add_(s.data, alpha=1.0 - decay)

    def _zero_coords(self, z):
        return torch.zeros(z.shape[0], 3, device=z.device, dtype=z.dtype)

    def forward(self, z_t: torch.Tensor,
                z_tk: Optional[torch.Tensor] = None,
                c_t:  Optional[torch.Tensor] = None,
                c_tk: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Single mode (z_tk=None):  returns encoder(z_t) → out_proj  [B, out_dim]
        Pair mode   (z_tk given): returns pair prediction             [B, out_dim]
        """
        if z_tk is None:
            return self.out_proj(self.encoder(z_t))

        # Pair mode
        tok_t  = self.encoder(z_t)
        tok_tk = self.encoder(z_tk)
        tokens = torch.stack([tok_t, tok_tk], dim=1)   # [B, 2, embed_dim]

        c_t  = c_t  if c_t  is not None else self._zero_coords(z_t)
        c_tk = c_tk if c_tk is not None else self._zero_coords(z_tk)
        coords = torch.stack([c_t, c_tk], dim=1)       # [B, 2, 3]

        attn_out = self.attn(tokens, coords)
        tokens   = self.attn_ln(tokens + attn_out)
        tokens   = self.ffn_ln(tokens + self.ffn(tokens))

        # Use token 0 (z_t attended to context) to predict target z_tk
        # Asymmetry: we predict where we're going (z_tk) from where we are (z_t)
        return self.out_proj(tokens[:, 0, :])

    def target_embed(self, z: torch.Tensor) -> torch.Tensor:
        """EMA embedding of target frame. No gradient. Used in x-pred loss."""
        with torch.no_grad():
            return self.target_encoder(z)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Losses
# ─────────────────────────────────────────────────────────────────────────────

def infonce_loss(z_a, z_p, T=0.07):
    a = F.normalize(z_a, dim=-1); p = F.normalize(z_p, dim=-1)
    logits = (a @ p.T) / T
    labels = torch.arange(len(a), device=a.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def vicreg_loss(z, sim_coeff=25., var_coeff=25., cov_coeff=1.):
    """Collapse prevention. Always apply alongside contrastive loss."""
    std  = z.std(dim=0)
    var  = F.relu(1.0 - std).mean()
    zc   = z - z.mean(dim=0)
    cov  = (zc.T @ zc) / (z.shape[0] - 1)
    cov_loss = (cov ** 2).fill_diagonal_(0.).sum() / z.shape[1]
    return var_coeff * var + cov_coeff * cov_loss


def xpred_loss(pred, target, T=0.07, λ_vicreg=0.1):
    """
    x-prediction loss (UniFluids §3.4):
      - MSE between prediction and EMA target
      - InfoNCE contrastive term
      - VICReg variance regularisation on predictions
    """
    mse      = F.mse_loss(pred, target.detach())
    contrast = infonce_loss(pred, target.detach(), T)
    vic      = vicreg_loss(pred)
    total    = 0.5 * mse + 0.5 * contrast + λ_vicreg * vic
    return total, {"mse": mse.item(), "contrast": contrast.item(), "vicreg": vic.item()}


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np
    print("=== RoPETemporalHead v2 Smoke Test ===\n")
    torch.manual_seed(42)

    head = RoPETemporalHead()
    print(f"Trainable params: {head.n_params:,}")

    B = 8
    z_t  = torch.randn(B, 128)
    z_tk = torch.randn(B, 128)
    c_t  = torch.rand(B, 3)
    c_tk = torch.rand(B, 3)

    # Single mode
    emb = head(z_t)
    assert emb.shape == (B, 64)
    print(f"✅ Single mode:   {emb.shape}")

    # Pair mode
    pred = head(z_t, z_tk, c_t, c_tk)
    assert pred.shape == (B, 64)
    print(f"✅ Pair mode:     {pred.shape}")

    # Coords change output
    pred0 = head(z_t, z_tk)
    assert not torch.allclose(pred, pred0, atol=1e-4)
    print(f"✅ Coords matter: delta={( pred - pred0).norm():.4f}")

    # EMA target
    target = head.target_embed(z_tk)
    assert target.shape == (B, 64)
    assert target.requires_grad == False
    print(f"✅ EMA target:    {target.shape}  requires_grad={target.requires_grad}")

    # x-pred loss
    loss, breakdown = xpred_loss(pred, target)
    print(f"✅ x-pred loss:   {loss.item():.4f}  {breakdown}")

    # VICReg
    vic = vicreg_loss(pred)
    print(f"✅ VICReg:        {vic.item():.4f}")

    # Gradient
    loss.backward()
    gnorms = [p.grad.norm().item() for p in head.parameters()
              if p.grad is not None and p.requires_grad]
    print(f"✅ Gradients:     {len(gnorms)} tensors  mean={np.mean(gnorms):.6f}")

    # EMA update
    head.update_ema(decay=0.996)
    print(f"✅ EMA update:    OK")

    print(f"\n✅ All tests passed")
