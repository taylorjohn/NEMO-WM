"""
proprio_decoder.py — NeMo-WM Sprint B
=======================================
Decoder: z → obs reconstruction.

Closes the "no imagination" gap — the model can now:
  1. Encode:  obs → z  (encoder)
  2. Predict: z_t + a → z_{t+1}  (transition)
  3. Decode:  z_{t+1} → obs_{t+1}  (this file)
  4. Imagine: chain steps 2+3 for H steps without real observations

Components:
  ProprioDecoder       — z → obs (MSE reconstruction)
  TransitionDecoder    — (z_t, a_t) → obs_{t+1} (one-step imagination)
  ImaginationRollout   — chain H steps for multi-step imagination
  DecoderLoss          — combined recon + imagination loss

Biological mapping:
  Reconstruction loss → serotonin (representation accuracy / diversity)
  Imagination error   → norepinephrine (spatial grounding failure)
  Multi-step drift    → cortisol (compounding error = domain shift signal)

Usage:
    from proprio_decoder import ProprioDecoder, DecoderLoss, ImaginationRollout

    decoder  = ProprioDecoder(d_latent=128, obs_dim=5)
    loss_fn  = DecoderLoss(recon_weight=1.0, imagine_weight=0.5)
    rollout  = ImaginationRollout(encoder, transition, decoder)

    # Training
    z        = encoder(obs)
    obs_recon = decoder(z)
    L_recon   = loss_fn.recon(obs_recon, obs)

    # Imagination (H steps)
    obs_traj  = rollout.imagine(z0, actions)   # (B, H, obs_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


# ── Symlog for stable reconstruction ─────────────────────────────────────────

def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def symlog_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE in symlog space — stable for large observation values."""
    return F.mse_loss(symlog(pred), symlog(target))


# ── Proprio Decoder ───────────────────────────────────────────────────────────

class ProprioDecoder(nn.Module):
    """
    Decodes latent z → raw observation.

    Architecture mirrors encoder (inverted):
        z (d_latent) → hidden → hidden → obs (obs_dim)

    For PushT obs = [agent_x, agent_y, block_x, block_y, block_angle_norm]
    All in [0, 1] → sigmoid output.

    For RECON obs = [vel, ang, heading, sin_h, cos_h, d_heading, contact, delta_h]
    Mixed range → tanh + scale output.

    Args:
        d_latent:   latent dimension (matches encoder output)
        obs_dim:    observation dimension to reconstruct
        d_hidden:   hidden layer width
        output_act: 'sigmoid' (bounded [0,1]), 'tanh', or 'linear'
    """

    def __init__(
        self,
        d_latent: int,
        obs_dim: int,
        d_hidden: int = 128,
        output_act: str = 'sigmoid',
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_latent, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, obs_dim),
        )

        if output_act == 'sigmoid':
            self.out_act = nn.Sigmoid()
        elif output_act == 'tanh':
            self.out_act = nn.Tanh()
        else:
            self.out_act = nn.Identity()

        self.obs_dim  = obs_dim
        self.d_latent = d_latent

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, d_latent) latent vector

        Returns:
            obs_recon: (B, obs_dim) reconstructed observation
        """
        return self.out_act(self.net(z))


# ── Block Position Decoder (task-specific) ────────────────────────────────────

class BlockPositionDecoder(nn.Module):
    """
    Specialised decoder for PushT block position prediction.

    Instead of reconstructing full obs, predicts:
        block_x, block_y  in [0, 1]
        block_angle_norm  in [0, 1]

    This is the probe equivalent but as a generative decoder.
    Useful for: checking if z contains enough info to imagine block position.

    Compare with BlockProbe (discriminative) — both test same information
    but decoder is generative and enables imagination.
    """

    def __init__(self, d_latent: int, d_hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_latent, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 3),  # [block_x, block_y, block_angle]
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Returns (B, 3): [block_x, block_y, block_angle_norm]"""
        return self.net(z)


# ── Transition Decoder — one-step imagination ─────────────────────────────────

class TransitionDecoder(nn.Module):
    """
    One-step imagination: (z_t, a_t) → obs_{t+1}

    Chains:
        z_{t+1} = transition(z_t, a_t)
        obs_{t+1} = decoder(z_{t+1})

    This is the minimum viable imagination unit.
    Chain H times for multi-step imagination.
    """

    def __init__(self, transition: nn.Module, decoder: nn.Module):
        super().__init__()
        self.transition = transition
        self.decoder    = decoder

    def forward(
        self,
        z_t: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_t:    (B, d_latent) current latent
            action: (B, action_dim) action

        Returns:
            z_next:    (B, d_latent) predicted next latent
            obs_next:  (B, obs_dim) imagined next observation
        """
        z_next   = self.transition(z_t, action)
        obs_next = self.decoder(z_next)
        return z_next, obs_next


# ── Imagination Rollout — H-step ─────────────────────────────────────────────

class ImaginationRollout(nn.Module):
    """
    H-step imagination rollout entirely in latent space.

    Given:
        z_0      — initial latent (from encoder)
        actions  — action sequence (B, H, action_dim)

    Produces:
        z_traj   — latent trajectory (B, H+1, d_latent)
        obs_traj — imagined observation trajectory (B, H+1, obs_dim)

    Biological analogy:
        This is hippocampal replay — the model "plays forward"
        what would happen if it took action sequence A from state z_0.
        Cortisol monitors drift between imagined and real observations.

    DreamerV3 tricks applied:
        - symlog on observations before MSE (stable large values)
        - geometric discounting on future steps (near > far)
        - stops gradient through z_traj (straight-through)
    """

    def __init__(
        self,
        encoder:    nn.Module,
        transition: nn.Module,
        decoder:    nn.Module,
        gamma:      float = 0.95,
    ):
        super().__init__()
        self.encoder    = encoder
        self.transition = transition
        self.decoder    = decoder
        self.gamma      = gamma

    @torch.no_grad()
    def imagine(
        self,
        z0:      torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pure inference rollout (no gradients).

        Args:
            z0:      (B, d_latent) initial latent
            actions: (B, H, action_dim) action sequence

        Returns:
            z_traj:   (B, H+1, d_latent)
            obs_traj: (B, H+1, obs_dim)
        """
        B, H, _ = actions.shape
        z_traj   = [z0]
        obs_traj = [self.decoder(z0)]

        z = z0
        for t in range(H):
            z   = self.transition(z, actions[:, t])
            obs = self.decoder(z)
            z_traj.append(z)
            obs_traj.append(obs)

        return (
            torch.stack(z_traj,   dim=1),   # (B, H+1, d_latent)
            torch.stack(obs_traj, dim=1),   # (B, H+1, obs_dim)
        )

    def imagine_with_grad(
        self,
        z0:      torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training rollout (with gradients for actor-critic).

        Returns:
            z_traj:       (B, H+1, d_latent)
            obs_traj:     (B, H+1, obs_dim)
            drift_scores: (B, H) cortisol proxy — error growth per step
        """
        B, H, _ = actions.shape
        z_traj   = [z0]
        obs_traj = [self.decoder(z0)]
        drift    = []

        z = z0
        for t in range(H):
            z_prev = z
            z      = self.transition(z, actions[:, t])
            obs    = self.decoder(z)

            # Cortisol proxy: prediction error per step
            # High drift = compounding error = cortisol spike
            step_drift = (z - z_prev).norm(dim=-1)  # (B,)
            drift.append(step_drift)

            z_traj.append(z)
            obs_traj.append(obs)

        return (
            torch.stack(z_traj,   dim=1),   # (B, H+1, d_latent)
            torch.stack(obs_traj, dim=1),   # (B, H+1, obs_dim)
            torch.stack(drift,    dim=1),   # (B, H) cortisol signal
        )

    def imagination_loss(
        self,
        z0:      torch.Tensor,
        actions: torch.Tensor,
        obs_gt:  torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compare imagined trajectory against ground truth observations.

        Args:
            z0:      (B, d_latent) initial latent from real obs_0
            actions: (B, H, action_dim) real actions taken
            obs_gt:  (B, H+1, obs_dim) real observations (ground truth)

        Returns:
            loss:    scalar imagination loss
            metrics: dict with per-step errors and cortisol signal
        """
        z_traj, obs_traj, drift = self.imagine_with_grad(z0, actions)

        H = actions.shape[1]

        # Geometric discounting — near steps weighted more
        weights = torch.tensor(
            [self.gamma ** t for t in range(H + 1)],
            device=z0.device, dtype=z0.dtype
        )

        # Per-step reconstruction error in symlog space
        step_errors = []
        for t in range(H + 1):
            err = symlog_mse(obs_traj[:, t], obs_gt[:, t])
            step_errors.append(err * weights[t])

        loss = torch.stack(step_errors).mean()

        metrics = {
            'L_imagine':     loss.item(),
            'step_errors':   [e.item() for e in step_errors],
            'drift_mean':    drift.mean().item(),
            'drift_max':     drift.max().item(),
            'cortisol_signal': drift.mean().item(),
        }
        return loss, metrics


# ── Decoder Loss ──────────────────────────────────────────────────────────────

class DecoderLoss(nn.Module):
    """
    Combined reconstruction + imagination loss.

    L_total = w_recon * L_recon + w_imagine * L_imagine

    Args:
        recon_weight:   weight on single-step reconstruction
        imagine_weight: weight on multi-step imagination
        use_symlog:     apply symlog before MSE (DreamerV3)
    """

    def __init__(
        self,
        recon_weight:   float = 1.0,
        imagine_weight: float = 0.5,
        use_symlog:     bool = True,
    ):
        super().__init__()
        self.recon_weight   = recon_weight
        self.imagine_weight = imagine_weight
        self.use_symlog     = use_symlog

    def recon(
        self,
        obs_pred: torch.Tensor,
        obs_gt:   torch.Tensor,
    ) -> torch.Tensor:
        """Single-step reconstruction loss."""
        if self.use_symlog:
            return symlog_mse(obs_pred, obs_gt)
        return F.mse_loss(obs_pred, obs_gt)

    def forward(
        self,
        obs_pred:    torch.Tensor,
        obs_gt:      torch.Tensor,
        imagine_loss: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        L_recon = self.recon(obs_pred, obs_gt)
        total   = self.recon_weight * L_recon

        metrics = {'L_recon': L_recon.item()}

        if imagine_loss is not None:
            total = total + self.imagine_weight * imagine_loss
            metrics['L_imagine'] = imagine_loss.item()

        metrics['L_total'] = total.item()
        return total, metrics


# ── Quick test ────────────────────────────────────────────────────────────────

def _test():
    print("ProprioDecoder self-test")

    import sys, os
    sys.path.insert(0, os.getcwd())

    B, obs_dim, d_latent = 8, 5, 128

    # Decoder
    dec = ProprioDecoder(d_latent, obs_dim, d_hidden=128, output_act='sigmoid')
    print(f"  Params: {sum(p.numel() for p in dec.parameters()):,}")

    z        = torch.randn(B, d_latent)
    obs_recon = dec(z)
    assert obs_recon.shape == (B, obs_dim)
    assert obs_recon.min() >= 0 and obs_recon.max() <= 1, "sigmoid out of range"
    print(f"  Decode: z={z.shape} → obs={obs_recon.shape}  range=[{obs_recon.min():.2f},{obs_recon.max():.2f}]")

    # Block position decoder
    block_dec = BlockPositionDecoder(d_latent)
    block_pos = block_dec(z)
    assert block_pos.shape == (B, 3)
    print(f"  Block decoder: z={z.shape} → block={block_pos.shape}")

    # Loss
    obs_gt  = torch.rand(B, obs_dim)
    loss_fn = DecoderLoss(recon_weight=1.0, imagine_weight=0.5)
    L, metrics = loss_fn(obs_recon, obs_gt)
    L.backward()
    print(f"  Loss: {metrics}")
    print(f"  Backward: OK")

    # Imagination rollout (mock transition)
    class MockTransition(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.fc = nn.Linear(d + 2, d)
        def forward(self, z, a):
            return torch.tanh(self.fc(torch.cat([z, a], dim=-1)))

    class MockEncoder(nn.Module):
        def __init__(self, obs_dim, d):
            super().__init__()
            self.fc = nn.Linear(obs_dim, d)
        def forward(self, x):
            return self.fc(x)

    enc   = MockEncoder(obs_dim, d_latent)
    trans = MockTransition(d_latent)
    H     = 8

    rollout = ImaginationRollout(enc, trans, dec, gamma=0.95)

    # Inference rollout
    z0      = torch.randn(B, d_latent)
    actions = torch.rand(B, H, 2)
    with torch.no_grad():
        z_traj, obs_traj = rollout.imagine(z0, actions)
    assert z_traj.shape   == (B, H + 1, d_latent)
    assert obs_traj.shape == (B, H + 1, obs_dim)
    print(f"  Imagine: z_traj={z_traj.shape}  obs_traj={obs_traj.shape}")

    # Training rollout with loss
    obs_gt_traj = torch.rand(B, H + 1, obs_dim)
    L_im, im_metrics = rollout.imagination_loss(z0, actions, obs_gt_traj)
    L_im.backward()
    print(f"  Imagination loss: {im_metrics['L_imagine']:.4f}  "
          f"drift={im_metrics['drift_mean']:.4f}  "
          f"cortisol={im_metrics['cortisol_signal']:.4f}")
    print(f"  Backward: OK")

    print("  All assertions passed. ✅")


if __name__ == '__main__':
    _test()
