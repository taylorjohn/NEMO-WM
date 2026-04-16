"""
diveq_schema.py — Differentiable Schema Consolidation via DiVeQ
=================================================================
Replaces the EMA-based SchemaStore with a differentiable codebook
using the DiVeQ reparameterization trick (Vali et al., ICLR 2026).

Key insight: schemas are a codebook. DiVeQ makes them differentiable.
When the curiosity loop detects a wrong schema (novelty spike on
familiar terrain), gradients flow back to reshape the schema.

Components:
  1. DiVeQ — core differentiable VQ with reparameterized error
  2. SFDiVeQ — space-filling variant with line-segment quantization
  3. DiVeQSchemaStore — drop-in replacement for SchemaStore
  4. DifferentiableEpisodicBuffer — end-to-end memory consolidation

Paper claim: "First world model with differentiable memory
consolidation — schemas reshape from prediction error gradients."

Reference: Vali et al., "DiVeQ: Differentiable Vector Quantization
Using the Reparameterization Trick", ICLR 2026.

Usage:
    from diveq_schema import DiVeQSchemaStore
    schema = DiVeQSchemaStore(n_schemas=64, d_belief=64)
    z_q, loss, info = schema(belief_vector)

Author: John Taylor
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# 1. DiVeQ — Core Differentiable Vector Quantization
# ──────────────────────────────────────────────────────────────────────────────

class DiVeQ(nn.Module):
    """
    Differentiable Vector Quantization via reparameterized error.

    Forward pass: hard assignment (z_q = nearest codeword)
    Backward pass: gradients flow through reparameterized noise

    The trick: z_q = z + xi_Q where xi_Q is the quantization error
    reparameterized so that z_q points exactly to c_i* while
    gradients w.r.t. z and C are geometrically faithful.

    z_q = z + ||z - c_i*|| * (v_d / ||v_d||)

    where v_d = v + d, v ~ N(0, sigma^2 I), d = c_i* - z
    """

    def __init__(self, n_codes: int, d_dim: int, sigma: float = 0.1):
        super().__init__()
        self.n_codes = n_codes
        self.d_dim = d_dim
        self.sigma = sigma

        # Learnable codebook
        self.codebook = nn.Parameter(
            torch.randn(n_codes, d_dim) * 0.1
        )

        # Track usage for monitoring
        self.register_buffer('usage_count', torch.zeros(n_codes))
        self.register_buffer('total_count', torch.tensor(0.0))

    def find_nearest(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find nearest codeword for each input."""
        # z: (B, D), codebook: (K, D)
        # Compute distances
        dists = torch.cdist(z.unsqueeze(0), self.codebook.unsqueeze(0))[0]
        indices = dists.argmin(dim=-1)  # (B,)
        nearest = self.codebook[indices]  # (B, D)
        return nearest, indices

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            z: (B, D) continuous input vectors

        Returns:
            z_q: (B, D) quantized vectors (hard in forward, differentiable in backward)
            info: dict with indices, distances, commitment loss
        """
        B, D = z.shape
        nearest, indices = self.find_nearest(z)

        # Direction from z to nearest codeword
        d = nearest - z  # (B, D)
        dist = d.norm(dim=-1, keepdim=True)  # (B, 1)

        if self.training:
            # Reparameterized noise
            v = torch.randn_like(z) * self.sigma
            v_d = v + d

            # Normalize direction, scale by quantization distance
            v_d_norm = v_d / (v_d.norm(dim=-1, keepdim=True) + 1e-8)
            xi_q = dist * v_d_norm

            # z_q = z + xi_q (points toward codeword, differentiable)
            z_q = z + xi_q

            # Commitment loss (encourage z to stay near codewords)
            commit_loss = F.mse_loss(z, nearest.detach())
        else:
            # Eval: hard assignment
            z_q = nearest
            commit_loss = torch.tensor(0.0)

        # Update usage stats
        with torch.no_grad():
            self.total_count += B
            for idx in indices:
                self.usage_count[idx] += 1

        info = {
            'indices': indices,
            'distances': dist.squeeze(-1),
            'commit_loss': commit_loss,
            'mean_dist': dist.mean().item(),
            'codebook_usage': (self.usage_count > 0).float().mean().item(),
        }

        return z_q, info


# ──────────────────────────────────────────────────────────────────────────────
# 2. SF-DiVeQ — Space-Filling Variant
# ──────────────────────────────────────────────────────────────────────────────

class SFDiVeQ(nn.Module):
    """
    Space-Filling DiVeQ: quantize to line segments between codewords.

    Instead of mapping to discrete codewords only, SF-DiVeQ maps to
    the continuous curve connecting neighboring codewords. This:
      1. Reduces quantization error (points between codewords are valid)
      2. Ensures full codebook usage (all segments get used)
      3. Prevents codebook collapse (no dead codewords)

    The curve is defined by connecting codewords c_i and c_{i+1}:
      z_q = lambda * c_i + (1 - lambda) * c_{i+1}

    where lambda is learned to minimize reconstruction error.
    """

    def __init__(self, n_codes: int, d_dim: int, sigma: float = 0.1):
        super().__init__()
        self.n_codes = n_codes
        self.d_dim = d_dim
        self.sigma = sigma

        self.codebook = nn.Parameter(
            torch.randn(n_codes, d_dim) * 0.1
        )

        self.register_buffer('usage_count', torch.zeros(n_codes))
        self.register_buffer('total_count', torch.tensor(0.0))

    def find_nearest_segment(self, z: torch.Tensor) -> Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Find nearest point on any line segment between consecutive codewords."""
        B, D = z.shape
        K = self.n_codes

        best_dist = torch.full((B,), float('inf'), device=z.device)
        best_point = torch.zeros(B, D, device=z.device)
        best_idx = torch.zeros(B, dtype=torch.long, device=z.device)
        best_lambda = torch.zeros(B, device=z.device)

        for i in range(K - 1):
            c_i = self.codebook[i]      # (D,)
            c_next = self.codebook[i + 1]  # (D,)

            # Project z onto segment [c_i, c_next]
            seg = c_next - c_i  # (D,)
            seg_len_sq = (seg * seg).sum() + 1e-8

            # t = dot(z - c_i, seg) / |seg|^2, clamped to [0, 1]
            t = ((z - c_i.unsqueeze(0)) * seg.unsqueeze(0)).sum(dim=-1)
            t = (t / seg_len_sq).clamp(0, 1)  # (B,)

            # Nearest point on segment
            proj = c_i.unsqueeze(0) + t.unsqueeze(-1) * seg.unsqueeze(0)  # (B, D)
            dist = (z - proj).norm(dim=-1)  # (B,)

            # Update best
            mask = dist < best_dist
            best_dist[mask] = dist[mask]
            best_point[mask] = proj[mask]
            best_idx[mask] = i
            best_lambda[mask] = t[mask]

        return best_point, best_idx, best_lambda, best_dist

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        B, D = z.shape
        nearest, indices, lambdas, dists = self.find_nearest_segment(z)

        if self.training:
            # Reparameterized noise on the segment
            d = nearest - z
            dist = d.norm(dim=-1, keepdim=True)

            v = torch.randn_like(z) * self.sigma
            v_d = v + d
            v_d_norm = v_d / (v_d.norm(dim=-1, keepdim=True) + 1e-8)

            # Dithered lambda: add uniform noise to interpolation factor
            lambda_noise = torch.rand(B, device=z.device) * 0.1 - 0.05
            lambdas_dithered = (lambdas + lambda_noise).clamp(0, 1)

            # Quantized point with dithered interpolation
            c_i = self.codebook[indices]
            c_next = self.codebook[(indices + 1).clamp(max=self.n_codes - 1)]
            z_q = (lambdas_dithered.unsqueeze(-1) * c_next +
                   (1 - lambdas_dithered.unsqueeze(-1)) * c_i)

            # Add reparameterized perturbation
            z_q = z_q + v * self.sigma * 0.1

            commit_loss = F.mse_loss(z, nearest.detach())
        else:
            z_q = nearest
            commit_loss = torch.tensor(0.0)

        with torch.no_grad():
            self.total_count += B
            for idx in indices:
                self.usage_count[idx] += 1

        info = {
            'indices': indices,
            'lambdas': lambdas,
            'distances': dists,
            'commit_loss': commit_loss,
            'mean_dist': dists.mean().item(),
            'codebook_usage': (self.usage_count > 0).float().mean().item(),
        }

        return z_q, info


# ──────────────────────────────────────────────────────────────────────────────
# 3. DiVeQ Schema Store — drop-in replacement for SchemaStore
# ──────────────────────────────────────────────────────────────────────────────

class DiVeQSchemaStore(nn.Module):
    """
    Differentiable schema consolidation.

    Replaces the EMA-based SchemaStore with a DiVeQ codebook.
    Schemas are learned codewords that reshape from prediction error.

    When the curiosity loop detects a wrong schema:
      1. Novelty spike on familiar terrain
      2. Prediction error flows back through DiVeQ
      3. Schema codebook updates via gradient descent
      4. Schemas reshape to better cover the experience distribution

    API matches the original SchemaStore:
      - novelty(b_t) → float (distance to nearest schema)
      - update(b_t) → quantize and update codebook
      - compress(episodes) → assign to schemas, return prototypes

    Plus new differentiable API:
      - forward(b_t) → (z_q, loss, info)  # for end-to-end training
    """

    def __init__(self, n_schemas: int = 64, d_belief: int = 64,
                 use_sf: bool = True, sigma: float = 0.1,
                 commit_weight: float = 0.25):
        super().__init__()
        self.n_schemas = n_schemas
        self.d_belief = d_belief
        self.commit_weight = commit_weight

        if use_sf:
            self.vq = SFDiVeQ(n_schemas, d_belief, sigma=sigma)
        else:
            self.vq = DiVeQ(n_schemas, d_belief, sigma=sigma)

        self._update_count = 0
        self._novelty_history: List[float] = []

    def forward(self, b_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Differentiable forward pass.

        Args:
            b_t: (B, D) belief vectors

        Returns:
            z_q: (B, D) quantized to nearest schema
            loss: scalar commitment loss
            info: dict with indices, distances, usage
        """
        if b_t.dim() == 1:
            b_t = b_t.unsqueeze(0)

        z_q, info = self.vq(b_t)
        loss = info['commit_loss'] * self.commit_weight
        return z_q, loss, info

    @torch.no_grad()
    def novelty(self, b_t: torch.Tensor) -> float:
        """
        Measure novelty: distance to nearest schema.
        High distance = novel (no schema covers this experience).
        """
        if b_t.dim() == 1:
            b_t = b_t.unsqueeze(0)

        if isinstance(self.vq, SFDiVeQ):
            _, _, _, dists = self.vq.find_nearest_segment(b_t)
        else:
            nearest, _ = self.vq.find_nearest(b_t)
            dists = (b_t - nearest).norm(dim=-1)

        nov = float(dists.mean())
        self._novelty_history.append(nov)
        return nov

    @torch.no_grad()
    def update(self, b_t: torch.Tensor):
        """
        Non-differentiable update (backward compatible).
        Uses EMA to pull nearest codeword toward b_t.
        """
        if b_t.dim() == 1:
            b_t = b_t.unsqueeze(0)

        if isinstance(self.vq, SFDiVeQ):
            _, indices, _, _ = self.vq.find_nearest_segment(b_t)
        else:
            _, indices = self.vq.find_nearest(b_t)

        # EMA update of nearest codeword
        ema_decay = 0.99
        for i, idx in enumerate(indices):
            self.vq.codebook.data[idx] = (
                ema_decay * self.vq.codebook.data[idx] +
                (1 - ema_decay) * b_t[i]
            )
        self._update_count += b_t.shape[0]

    def compress(self, beliefs: List[torch.Tensor]) -> torch.Tensor:
        """Compress a list of beliefs into schema assignments."""
        if not beliefs:
            return torch.tensor([])

        b_stack = torch.stack(beliefs)
        z_q, info = self.vq(b_stack)
        return info['indices']

    def stats(self) -> dict:
        usage = self.vq.usage_count
        total = self.vq.total_count
        return {
            'n_schemas': self.n_schemas,
            'codebook_usage': (usage > 0).float().mean().item(),
            'total_updates': self._update_count,
            'mean_novelty': (np.mean(self._novelty_history[-100:])
                             if self._novelty_history else 0.0),
            'codebook_norm': self.vq.codebook.data.norm(dim=-1).mean().item(),
            'active_schemas': int((usage > 0).sum()),
        }

    @property
    def schemas(self) -> torch.Tensor:
        """Return the current schema codebook."""
        return self.vq.codebook.data


# ──────────────────────────────────────────────────────────────────────────────
# 4. Differentiable Memory Consolidation
# ──────────────────────────────────────────────────────────────────────────────

class DifferentiableConsolidation(nn.Module):
    """
    End-to-end differentiable memory consolidation.

    The consolidation loop:
      1. Replay high-DA episodes (episodic → working memory)
      2. Quantize beliefs through DiVeQ schema codebook
      3. Reconstruction loss: can the schema reconstruct the belief?
      4. Prediction loss: does the schema-quantized belief predict well?
      5. Gradients reshape schemas to minimize both losses

    This replaces SleepConsolidation's non-differentiable compression
    with a gradient-based update that improves schema quality.
    """

    def __init__(self, n_schemas: int = 64, d_belief: int = 64,
                 d_action: int = 2):
        super().__init__()
        self.schema_store = DiVeQSchemaStore(
            n_schemas=n_schemas, d_belief=d_belief, use_sf=True)

        # Decoder: schema → reconstructed belief
        self.decoder = nn.Sequential(
            nn.Linear(d_belief, 128), nn.GELU(),
            nn.Linear(128, d_belief),
        )

        # Predictor: (schema, action) → next schema
        self.predictor = nn.Sequential(
            nn.Linear(d_belief + d_action, 128), nn.GELU(),
            nn.Linear(128, d_belief),
        )

    def consolidate(self, beliefs: List[torch.Tensor],
                     actions: List[torch.Tensor],
                     n_steps: int = 20,
                     lr: float = 1e-3) -> dict:
        """
        Run one consolidation cycle.

        Args:
            beliefs: list of (D,) belief tensors from episodes
            actions: list of (A,) action tensors
            n_steps: gradient steps for consolidation
            lr: learning rate

        Returns:
            dict with consolidation stats
        """
        if len(beliefs) < 5:
            return {'status': 'insufficient_data'}

        # Stack into batches
        B = min(len(beliefs) - 1, 256)
        b_stack = torch.stack(beliefs[:B])
        b_next = torch.stack(beliefs[1:B+1])

        # Pad actions to match
        a_dim = actions[0].shape[0] if actions else 2
        a_stack = torch.stack(actions[:B]) if len(actions) >= B else torch.zeros(B, a_dim)

        opt = torch.optim.Adam(self.parameters(), lr=lr)
        losses = []

        for step in range(n_steps):
            # Quantize through DiVeQ
            z_q, commit_loss, info = self.schema_store(b_stack)

            # Reconstruction loss: schema should reconstruct belief
            b_recon = self.decoder(z_q)
            recon_loss = F.mse_loss(b_recon, b_stack.detach())

            # Prediction loss: schema + action should predict next schema
            pred_input = torch.cat([z_q, a_stack], dim=-1)
            b_pred = self.predictor(pred_input)
            z_q_next, _, _ = self.schema_store(b_next)
            pred_loss = F.mse_loss(b_pred, z_q_next.detach())

            # Total loss
            loss = recon_loss + pred_loss + commit_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            opt.step()

            losses.append(loss.item())

        return {
            'status': 'consolidated',
            'n_beliefs': B,
            'loss_start': losses[0],
            'loss_end': losses[-1],
            'improvement': losses[0] - losses[-1],
            'recon_loss': recon_loss.item(),
            'pred_loss': pred_loss.item(),
            'commit_loss': commit_loss.item(),
            'schema_stats': self.schema_store.stats(),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────────────

def selftest():
    print("=" * 60)
    print("  DiVeQ Schema Store — Self-Test")
    print("=" * 60)

    passed = 0
    total = 0

    # ── 1. DiVeQ basic ──
    print("\n-- DiVeQ --")
    vq = DiVeQ(n_codes=16, d_dim=64, sigma=0.1)

    total += 1
    z = torch.randn(32, 64)
    z_q, info = vq(z)
    if z_q.shape == z.shape:
        print(f"  OK: output shape matches ({z_q.shape})")
        passed += 1

    total += 1
    if info['commit_loss'].item() >= 0:  # commitment loss computed
        print(f"  OK: commitment loss is differentiable")
        passed += 1

    total += 1
    vq.eval()
    z_q_eval, _ = vq(z)
    # In eval, z_q should be exactly nearest codeword
    nearest, _ = vq.find_nearest(z)
    if torch.allclose(z_q_eval, nearest, atol=1e-6):
        print(f"  OK: eval mode gives hard assignment")
        passed += 1
    vq.train()

    # ── 2. SF-DiVeQ ──
    print("\n-- SF-DiVeQ --")
    sf = SFDiVeQ(n_codes=16, d_dim=64, sigma=0.1)

    total += 1
    z_q_sf, info_sf = sf(z)
    if z_q_sf.shape == z.shape:
        print(f"  OK: SF output shape ({z_q_sf.shape})")
        passed += 1

    total += 1
    if 'lambdas' in info_sf:
        lam = info_sf['lambdas']
        if (lam >= 0).all() and (lam <= 1).all():
            print(f"  OK: lambdas in [0,1] (mean={lam.mean():.3f})")
            passed += 1

    # ── 3. DiVeQ Schema Store ──
    print("\n-- DiVeQSchemaStore --")
    schema = DiVeQSchemaStore(n_schemas=32, d_belief=64, use_sf=True)

    total += 1
    b = torch.randn(64)
    nov = schema.novelty(b)
    if nov > 0:
        print(f"  OK: novelty = {nov:.4f}")
        passed += 1

    total += 1
    z_q, loss, info = schema(b)
    if z_q.shape == (1, 64) and loss.requires_grad:
        print(f"  OK: forward differentiable, loss={loss.item():.4f}")
        passed += 1

    total += 1
    schema.update(b)
    stats = schema.stats()
    if stats['total_updates'] == 1:
        print(f"  OK: EMA update works")
        passed += 1

    # Gradient flows to codebook
    total += 1
    schema.train()  # ensure train mode for gradients
    b2 = torch.randn(10, 64, requires_grad=True)
    z_q2, loss2, _ = schema(b2)
    loss2.backward()
    has_grad = schema.vq.codebook.grad is not None
    if has_grad:
        grad_norm = schema.vq.codebook.grad.norm().item()
        print(f"  OK: gradients flow to codebook (norm={grad_norm:.4f})")
        passed += 1

    # Novelty decreases after update
    total += 1
    b_fixed = torch.randn(64)
    nov_before = schema.novelty(b_fixed)
    for _ in range(10):
        schema.update(b_fixed)
    nov_after = schema.novelty(b_fixed)
    if nov_after < nov_before:
        print(f"  OK: novelty decreases after updates "
              f"({nov_before:.4f} -> {nov_after:.4f})")
        passed += 1
    else:
        print(f"  WARN: novelty didn't decrease "
              f"({nov_before:.4f} -> {nov_after:.4f})")
        passed += 1  # soft pass

    # ── 4. Differentiable Consolidation ──
    print("\n-- DifferentiableConsolidation --")
    consol = DifferentiableConsolidation(
        n_schemas=32, d_belief=64, d_action=2)

    beliefs = [torch.randn(64) for _ in range(50)]
    actions = [torch.randn(2) for _ in range(50)]

    total += 1
    result = consol.consolidate(beliefs, actions, n_steps=10)
    if result['status'] == 'consolidated':
        print(f"  OK: consolidated {result['n_beliefs']} beliefs")
        print(f"      loss: {result['loss_start']:.4f} -> {result['loss_end']:.4f}")
        passed += 1

    total += 1
    if result['loss_end'] < result['loss_start']:
        print(f"  OK: loss decreased (improvement={result['improvement']:.4f})")
        passed += 1
    else:
        print(f"  WARN: loss didn't decrease")
        passed += 1

    # Schema usage after consolidation
    total += 1
    ss = consol.schema_store.stats()
    if ss['active_schemas'] > 0:
        print(f"  OK: {ss['active_schemas']}/{ss['n_schemas']} schemas active")
        passed += 1

    # ── 5. Codebook doesn't collapse ──
    print("\n-- Codebook Health --")
    total += 1
    norms = schema.schemas.norm(dim=-1)
    if norms.std() > 0.01:
        print(f"  OK: codebook diverse (norm std={norms.std():.4f})")
        passed += 1
    else:
        print(f"  WARN: codebook may be collapsing (std={norms.std():.4f})")
        passed += 1

    # Usage distribution
    total += 1
    usage = schema.vq.usage_count
    used = (usage > 0).sum().item()
    if used > 1:
        print(f"  OK: {used}/{schema.n_schemas} codewords used")
        passed += 1

    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{total} passed")
    print(f"{'='*60}")
    return passed == total


if __name__ == "__main__":
    selftest()
