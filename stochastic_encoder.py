"""
stochastic_encoder.py — NeMo-WM Sprint A
==========================================
Drop-in VAE-style replacement for StateEncoder.

Key design decisions:
  - Same forward() interface: obs → z  (downstream unchanged)
  - Adds mu, log_var as attributes for KL computation
  - Free bits clamping (DreamerV3) prevents posterior collapse
  - Symlog on KL for training stability
  - DA signal = KL divergence (biological: surprise = novelty)
  - Can be used in deterministic mode (eval) for reproducibility

Usage:
    from stochastic_encoder import StochasticEncoder, kl_loss

    enc = StochasticEncoder(obs_dim=5, d_model=128, d_latent=64)

    # Training
    z, mu, log_var = enc(obs, deterministic=False)
    kl = kl_loss(mu, log_var, free_bits=0.5)
    loss = recon_loss + beta * kl

    # Inference (deterministic — use mean)
    z = enc(obs)   # deterministic=True by default in eval mode

    # DA signal from KL
    da = kl.detach().clamp(0, 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ── Symlog / symexp (DreamerV3) ───────────────────────────────────────────────

def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric log — compresses large values, linear near zero."""
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog."""
    return torch.sign(x) * (x.abs().exp() - 1)


# ── KL loss ───────────────────────────────────────────────────────────────────

def kl_loss(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    free_bits: float = 0.5,
    use_symlog: bool = True,
) -> torch.Tensor:
    """
    KL divergence: q(z|x) || p(z) where p(z) = N(0, I).

    Args:
        mu:        (B, D) posterior mean
        log_var:   (B, D) posterior log-variance
        free_bits: minimum KL per dimension (DreamerV3 trick)
                   prevents trivial solution where posterior = prior
        use_symlog: apply symlog before clamping for stability

    Returns:
        scalar KL loss (mean over batch and dimensions)

    Biological interpretation:
        KL = surprise = how much the posterior differs from prior
        High KL → DA spike (REOBSERVE)
        Low KL  → prior matches observation (EXPLOIT)
    """
    # Per-dimension KL: -0.5 * (1 + log_var - mu^2 - var)
    kl_per_dim = -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp())  # (B, D)

    if use_symlog:
        kl_per_dim = symlog(kl_per_dim)

    # Free bits: clamp minimum KL per dimension
    # Prevents posterior collapse (model ignoring latent)
    kl_per_dim = kl_per_dim.clamp(min=free_bits)

    return kl_per_dim.mean()


# ── Prior network p(z | z_{t-1}, a) ──────────────────────────────────────────

class TransitionPrior(nn.Module):
    """
    Learned prior: p(z_t | z_{t-1}, a_{t-1}).

    Replaces fixed N(0, I) prior with a dynamics-aware prior.
    KL between posterior q(z_t | obs_t) and this prior is the
    world model's prediction error — biologically = DA signal.

    Without this: KL = deviation from standard normal
    With this:    KL = deviation from what the WM predicted
    The second is much more informative.
    """
    def __init__(self, d_latent: int, action_dim: int, d_hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_latent + action_dim, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_latent * 2),  # → (mu_prior, log_var_prior)
        )
        self.d_latent = d_latent

    def forward(
        self, z_prev: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_prev:  (B, d_latent) previous latent
            action:  (B, action_dim) action taken

        Returns:
            mu_prior, log_var_prior — each (B, d_latent)
        """
        h = self.net(torch.cat([z_prev, action], dim=-1))
        mu_prior, log_var_prior = h.chunk(2, dim=-1)
        log_var_prior = log_var_prior.clamp(-4, 4)  # numerical stability
        return mu_prior, log_var_prior


def kl_with_prior(
    mu_post: torch.Tensor,
    log_var_post: torch.Tensor,
    mu_prior: torch.Tensor,
    log_var_prior: torch.Tensor,
    free_bits: float = 0.5,
) -> torch.Tensor:
    """
    KL between posterior q(z|obs) and learned prior p(z|z_prev, a).
    This is the RSSM-style world model KL (DreamerV3).

    KL(q||p) = 0.5 * [log(σ_p/σ_q) + (σ_q + (μ_q-μ_p)^2)/σ_p - 1]
    """
    var_post  = log_var_post.exp()
    var_prior = log_var_prior.exp()

    kl = 0.5 * (
        log_var_prior - log_var_post
        + (var_post + (mu_post - mu_prior).pow(2)) / var_prior
        - 1.0
    )
    kl = symlog(kl)
    kl = kl.clamp(min=free_bits)
    return kl.mean()


# ── Stochastic Encoder ────────────────────────────────────────────────────────

class StochasticEncoder(nn.Module):
    """
    VAE-style encoder: obs → (z, mu, log_var).

    Architecture:
        obs → backbone → [mu_head | log_var_head] → sample z

    Drop-in replacement for StateEncoder:
        - Same forward() call: z = enc(obs)
        - Adds .mu and .log_var attributes after each forward pass
        - deterministic=True uses mean (no noise) — default in eval mode

    Args:
        obs_dim:       input observation dimension
        d_model:       hidden dimension
        d_latent:      latent dimension (can differ from d_model)
        log_var_min:   clamp log-variance minimum (-4 → std ≥ 0.14)
        log_var_max:   clamp log-variance maximum (+4 → std ≤ 7.4)
    """

    def __init__(
        self,
        obs_dim: int,
        d_model: int,
        d_latent: Optional[int] = None,
        log_var_min: float = -4.0,
        log_var_max: float = 4.0,
    ):
        super().__init__()
        d_latent = d_latent or d_model

        # Shared backbone (same as original StateEncoder)
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Separate heads for mean and log-variance
        self.mu_head      = nn.Linear(d_model, d_latent)
        self.log_var_head = nn.Linear(d_model, d_latent)

        self.log_var_min = log_var_min
        self.log_var_max = log_var_max
        self.d_latent    = d_latent

        # Stored after each forward pass — used by training loop
        self.mu:      Optional[torch.Tensor] = None
        self.log_var: Optional[torch.Tensor] = None

    def encode(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode obs → (mu, log_var). Does not sample."""
        h       = self.backbone(obs)
        mu      = self.mu_head(h)
        log_var = self.log_var_head(h).clamp(self.log_var_min, self.log_var_max)
        return mu, log_var

    def reparameterise(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        """Sample z ~ N(mu, exp(log_var)) via reparameterisation trick."""
        std = (log_var * 0.5).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Args:
            obs:           (B, obs_dim) observations
            deterministic: if True, return mu (no noise)
                           if None, use self.training (False=stochastic)

        Returns:
            z: (B, d_latent) latent sample or mean

        Side effects:
            self.mu, self.log_var updated for KL computation
        """
        mu, log_var = self.encode(obs)
        self.mu      = mu
        self.log_var = log_var

        use_det = deterministic if deterministic is not None else (not self.training)
        if use_det:
            return mu
        return self.reparameterise(mu, log_var)

    @property
    def da_signal(self) -> Optional[torch.Tensor]:
        """
        Dopamine proxy: per-sample KL from standard normal.
        High DA = encoder surprised by observation.
        Returns scalar per batch element — shape (B,).
        """
        if self.mu is None:
            return None
        kl = -0.5 * (1 + self.log_var - self.mu.pow(2) - self.log_var.exp())
        return symlog(kl).clamp(min=0).mean(dim=-1)  # (B,)


# ── Training helper ───────────────────────────────────────────────────────────

class StochasticWMLoss(nn.Module):
    """
    Combined loss for stochastic world model training.

    L_total = L_recon + beta * L_kl + L_predict

    Biological mapping:
        L_recon  → serotonin (representation diversity / accuracy)
        L_kl     → dopamine (surprise / novelty)
        L_predict → norepinephrine (spatial grounding)
        beta     → acetylcholine (temporal precision gate)
    """

    def __init__(
        self,
        beta: float = 1.0,
        free_bits: float = 0.5,
        use_prior: bool = False,
    ):
        super().__init__()
        self.beta      = beta
        self.free_bits = free_bits
        self.use_prior = use_prior

    def forward(
        self,
        recon_loss: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        predict_loss: Optional[torch.Tensor] = None,
        mu_prior: Optional[torch.Tensor] = None,
        log_var_prior: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Returns:
            total_loss: scalar
            metrics:    dict with individual loss components
        """
        # KL loss
        if self.use_prior and mu_prior is not None:
            kl = kl_with_prior(mu, log_var, mu_prior, log_var_prior,
                               self.free_bits)
        else:
            kl = kl_loss(mu, log_var, self.free_bits)

        total = recon_loss + self.beta * kl
        if predict_loss is not None:
            total = total + predict_loss

        metrics = {
            'L_recon':   recon_loss.item(),
            'L_kl':      kl.item(),
            'L_predict': predict_loss.item() if predict_loss is not None else 0.0,
            'L_total':   total.item(),
            'DA':        kl.item(),   # KL = dopamine signal
        }
        return total, metrics


# ── Quick test ────────────────────────────────────────────────────────────────

def _test():
    print("StochasticEncoder self-test")
    B, obs_dim, d_model, d_latent = 8, 5, 128, 64

    # Encoder
    enc = StochasticEncoder(obs_dim, d_model, d_latent)
    print(f"  Params: {sum(p.numel() for p in enc.parameters()):,}")

    obs = torch.randn(B, obs_dim)

    # Training mode — stochastic
    enc.train()
    z = enc(obs)
    assert z.shape == (B, d_latent), f"Bad shape: {z.shape}"
    assert enc.mu is not None
    kl = kl_loss(enc.mu, enc.log_var)
    da = enc.da_signal
    print(f"  Train: z={z.shape}  KL={kl.item():.4f}  DA={da.mean().item():.4f}")

    # Eval mode — deterministic
    enc.eval()
    with torch.no_grad():
        z_det = enc(obs)
    assert z_det.shape == (B, d_latent)
    print(f"  Eval:  z={z_det.shape}  (deterministic=True)")

    # Prior network
    prior = TransitionPrior(d_latent, action_dim=2)
    z_prev  = torch.randn(B, d_latent)
    action  = torch.randn(B, 2)
    mu_p, lv_p = prior(z_prev, action)
    kl_p = kl_with_prior(enc.mu, enc.log_var, mu_p, lv_p)
    print(f"  Prior KL: {kl_p.item():.4f}")

    # Loss module
    loss_fn = StochasticWMLoss(beta=1.0, free_bits=0.5)
    recon = F.mse_loss(z, obs[:, :d_latent] if d_latent <= obs_dim else
                       torch.zeros(B, d_latent))
    total, metrics = loss_fn(recon, enc.mu, enc.log_var)
    print(f"  Loss: {metrics}")

    # Backward
    total.backward()
    print("  Backward: OK")

    print("  All assertions passed. ✅")


if __name__ == '__main__':
    _test()
