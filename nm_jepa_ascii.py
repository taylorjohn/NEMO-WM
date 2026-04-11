"""
nm_jepa.py  --  Neuromodulated JEPA (NM-JEPA)
=============================================
Standalone reference implementation. No CORTEX dependencies.

NM-JEPA extends the Joint Embedding Predictive Architecture (JEPA) with a
seven-signal neuromodulator that adaptively scales each loss component based
on prediction error, latent stability, and arousal -- enabling stable
multi-domain training without per-domain loss weight tuning.

Architecture:
    Encoder       -- maps observations to 128-D L2-normalised latents
    Predictor     -- MoE forward dynamics model (s_t, a_t) -> ?_{t+1}
    Neuromodulator -- 7 biological signals derived from training state
    NMJEPALoss    -- loss function weighted by neuromodulator signals

Usage:
    python nm_jepa.py                   # runs self-test on random data
    python nm_jepa.py --epochs 10       # short training demo

Empirical results (CORTEX CWM, 2026-03-31, GMKtec EVO-X2, no GPU):

    Random encoder:
      DA = 0.000 for 185,000+ steps across 5 epochs.
      Prediction error is not meaningful in a semantically empty space.
      TemporalHead top1_acc = 0.031 (random) at all checkpoints.

    DINOv2-distilled encoder (56K params, student_best.pt):
      Starting loss: 0.9521 (vs 265 with random weights -- 280x better)
      Step 16,000: DA=0.001 -- first neuromodulated event (21:33 ET 2026-03-31)
      Step 27,000: regime EXPLOIT -> REOBSERVE -- first regime change
      Step 35,000: DA=0.003, sustained REOBSERVE
      Epoch 0 mean: 0.9180
      TemporalHead top1_acc: 0.094 at step 200 (3x random chance)
      TemporalHead oscillates 0.094->0.062->0.031->0.094 -- marginal but real signal
      Epoch 1+: DA continues rising, temporal structure expected to strengthen

    Key finding: the neuromodulator requires semantic encoder features to
    activate. DA=0 is correct in a random latent space. The phase transition
    DA=0 -> DA>0 marks when NM-JEPA becomes genuinely neuromodulated.
    This transition occurred at step 16,000 with a DINOv2-distilled encoder
    on RECON outdoor navigation data (GMKtec EVO-X2, no GPU).

Reference:
    CORTEX World Model, 2026
    Building on: JEPA (LeCun 2022), DreamerV3 (Hafner et al. 2023),
    Temporal Straightening (Wang et al. 2025), GRASP (Psenka et al. 2026)

Requirements:
    torch >= 2.0, numpy
"""

import math
import time
import argparse
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ???????????????????????????????????????????????????????????????????????????
# Utilities
# ???????????????????????????????????????????????????????????????????????????

def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric log -- compresses large magnitudes, identity near zero."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def free_bits(x: torch.Tensor, min_val: float = 0.5) -> torch.Tensor:
    """Floor loss so gradients only flow when predictor is uncertain."""
    return torch.clamp(x, min=min_val)


def agc_clip_(parameters, lam: float = 0.01, eps: float = 1e-6):
    """Adaptive gradient clipping -- per-parameter, scale-invariant."""
    for p in parameters:
        if p.grad is None:
            continue
        p_norm = p.data.norm(2.0).clamp(min=eps)
        g_norm = p.grad.norm(2.0)
        max_norm = lam * p_norm
        if g_norm > max_norm:
            p.grad.mul_(max_norm / (g_norm + eps))


# ???????????????????????????????????????????????????????????????????????????
# Encoder
# ???????????????????????????????????????????????????????????????????????????

class NMJEPAEncoder(nn.Module):
    """
    Lightweight observation encoder.
    Maps raw input to a 128-D L2-normalised latent vector.

    In production: replaced by StudentEncoder (56K, AMD NPU XINT8).
    Here: 3-layer MLP for domain-agnostic standalone use.
    """

    def __init__(self, obs_dim: int = 128, latent_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, latent_dim),
        )
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=-1)  # L2-normalise -> unit hypersphere


# ???????????????????????????????????????????????????????????????????????????
# MoE Predictor
# ???????????????????????????????????????????????????????????????????????????

class SparseMoEFFN(nn.Module):
    """
    Sparse Mixture-of-Experts feed-forward block.
    N experts, top-k active per token. Unimix prevents expert collapse.
    """

    def __init__(self, dim: int = 128, n_experts: int = 4, k_active: int = 2):
        super().__init__()
        self.n_experts = n_experts
        self.k_active  = k_active
        self.gate      = nn.Linear(dim, n_experts)
        self.experts   = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim),
            )
            for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Unimix: 1% uniform floor keeps all experts alive
        logits = self.gate(x)
        probs  = 0.99 * F.softmax(logits, dim=-1) + 0.01 / self.n_experts

        # Top-k sparse routing
        topk_probs, topk_idx = probs.topk(self.k_active, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(x)
        for i in range(self.k_active):
            idx = topk_idx[..., i]          # (B,)
            w   = topk_probs[..., i:i+1]    # (B, 1)
            # Route each sample to its selected expert
            for e in range(self.n_experts):
                mask = (idx == e)
                if mask.any():
                    out[mask] = out[mask] + w[mask] * self.experts[e](x[mask])

        # Auxiliary load-balancing loss
        aux = (probs * probs).sum(dim=-1).mean() * self.n_experts
        return out, aux


class NMJEPAPredictor(nn.Module):
    """
    Action-conditioned forward dynamics predictor.
    Given (z_t, a_t), predicts ?_{t+1} in latent space.

    Architecture: 2-layer transformer with sparse MoE FFN.
    Action broadcast: action projected to latent_dim, added to input.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        action_dim: int = 2,
        n_experts:  int = 4,
        k_active:   int = 2,
        n_layers:   int = 2,
    ):
        super().__init__()
        self.action_proj = nn.Linear(action_dim, latent_dim)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm1": nn.LayerNorm(latent_dim),
                "attn":  nn.MultiheadAttention(latent_dim, num_heads=4,
                                                batch_first=True),
                "norm2": nn.LayerNorm(latent_dim),
                "moe":   SparseMoEFFN(latent_dim, n_experts, k_active),
            })
            for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(latent_dim)

    def forward(
        self,
        z: torch.Tensor,       # (B, latent_dim)
        a: torch.Tensor,       # (B, action_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Inject action into latent
        x = z + self.action_proj(a)     # (B, D)
        x = x.unsqueeze(1)              # (B, 1, D) -- sequence of 1 token

        aux_total = torch.tensor(0.0, device=z.device)
        for layer in self.layers:
            # Self-attention (trivial for sequence length 1, useful when extended)
            xn = layer["norm1"](x)
            xa, _ = layer["attn"](xn, xn, xn)
            x = x + xa
            # Sparse MoE FFN
            xn = layer["norm2"](x.squeeze(1))
            xm, aux = layer["moe"](xn)
            x = (x.squeeze(1) + xm).unsqueeze(1)
            aux_total = aux_total + aux

        z_pred = self.out_norm(x.squeeze(1))          # (B, D)
        z_pred = F.normalize(z_pred, dim=-1)          # stay on hypersphere
        return z_pred, aux_total / len(self.layers)


# ???????????????????????????????????????????????????????????????????????????
# Neuromodulator
# ???????????????????????????????????????????????????????????????????????????

@dataclass
class NeuroSignals:
    """
    Seven biological signals derived from the current training state.

    DA  (dopamine)        -- prediction error drive
    5HT (serotonin)       -- latent stability
    NE  (norepinephrine)  -- global arousal / external signal strength
    ACh (acetylcholine)   -- attention precision (surprise x instability)
    E_I (excit/inhib)     -- exploration width
    Ado (adenosine)       -- session fatigue / curriculum pacing
    eCB (endocannabinoid) -- retrograde DA suppression (anti-loop)
    """
    da:      float = 0.5
    sht:     float = 0.5
    ne:      float = 0.5
    ach:     float = 0.5
    e_i:     float = 1.0
    ado:     float = 0.0
    ecb:     float = 0.0
    da_eff:  float = 0.5
    regime:  str   = "EXPLOIT"


class Neuromodulator:
    """
    Computes seven biological signals from training state each step.

    Signals are derived directly from observable training quantities:
      DA  <- prediction error (cosine distance between z_pred and z_target)
      5HT <- latent stability (inverse std of recent z history)
      NE  <- external arousal signal (e.g. Allen spike rate, defaults 0.5)
      ACh <- (DA + instability) / 2
      E/I <- DA / (1 - 5HT + eps) -- exploration pressure
      Ado <- session time fraction (rises from 0 -> 1 over session)
      eCB <- EMA of (DA x action magnitude) -- retrograde suppression

    Regime = f(DA, 5HT):
      EXPLORE   high DA, high 5HT -- novel + stable
      WAIT      high DA, low  5HT -- novel + unstable
      REOBSERVE low  DA, low  5HT -- known + unstable
      EXPLOIT   low  DA, high 5HT -- known + stable
    """

    def __init__(
        self,
        history_len:    int   = 32,
        ado_duration:   float = 3600.0,  # seconds for full session
        sht_sensitivity: float = 10.0,   # higher = 5HT drops faster on instability
        ecb_decay:      float = 0.9,     # EMA decay for retrograde suppression
        da_scale_base:  float = 0.5,     # minimum da_scale multiplier
        da_scale_gain:  float = 1.0,     # how hard DA events hit gradient
    ):
        """
        Tuning guide (from empirical RECON training observations, 2026-03-31):

        sht_sensitivity (default 10.0):
            Controls how quickly 5HT falls when representations become unstable.
            Higher = more sensitive to instability. Lower = more tolerant of drift.
            Range: 5.0 (tolerant) -> 20.0 (hair-trigger)

        ecb_decay (default 0.9):
            EMA decay for retrograde eCB signal. Lower = shorter memory of loops.
            0.9 = ~10 step memory. 0.99 = ~100 step memory.

        da_scale_base (default 0.5):
            Minimum gradient multiplier even when DA=0. Prevents full gradient
            suppression on routine batches.
            Range: 0.2 (suppress routine heavily) -> 1.0 (always full gradient)

        da_scale_gain (default 1.0):
            How hard surprising batches hit gradient relative to routine ones.
            da_scale = da_scale_base + da_scale_gain * da_eff
            At default: range [0.5, 1.5]. At gain=3.0: range [0.5, 3.5].
            Increase when DA is regularly above 0.05 and you want to amplify
            learning from surprising transitions. Don't increase before DA moves --
            amplifying 0.001 by 3x is still immeasurable.

        free_bits_min in NMJEPALoss (default 0.5):
            The most impactful single knob. When L_jepa is floored at 0.5,
            da_scale acts on 0.5 regardless of actual prediction error.
            Lowering to 0.1-0.2 lets real prediction error vary, making DA
            structurally impactful rather than cosmetically active.
            Only lower this after DA is regularly non-zero (epoch 2+ with real encoder).
        """
        self.history_len     = history_len
        self.ado_duration    = ado_duration
        self.sht_sensitivity = sht_sensitivity
        self.ecb_decay       = ecb_decay
        self.da_scale_base   = da_scale_base
        self.da_scale_gain   = da_scale_gain
        self.session_start   = time.time()
        self._z_history: list = []
        self._ecb_ema: float  = 0.0
        self.signals = NeuroSignals()

    def update(
        self,
        z_pred:           torch.Tensor,   # (B, D)
        z_target:         torch.Tensor,   # (B, D)
        ne:               float = 0.5,    # external arousal [0, 1]
        action_magnitude: float = 0.0,
    ) -> NeuroSignals:

        with torch.no_grad():
            # DA -- cosine distance (prediction error)
            cos_sim = F.cosine_similarity(
                z_pred.mean(0, keepdim=True),
                z_target.mean(0, keepdim=True),
            ).item()
            da = (1.0 - cos_sim) / 2.0           # [0, 1]

            # 5HT -- latent stability from z history
            z_mean = z_target.mean(0).cpu()
            self._z_history.append(z_mean)
            if len(self._z_history) > self.history_len:
                self._z_history.pop(0)

            if len(self._z_history) >= 2:
                stack = torch.stack(self._z_history)
                sht = float(torch.exp(-self.sht_sensitivity * stack.std(0).mean()))
            else:
                sht = 0.5

            # eCB -- retrograde suppression (EMA)
            self._ecb_ema = self.ecb_decay * self._ecb_ema + (1.0 - self.ecb_decay) * (da * action_magnitude)
            ecb = min(1.0, self._ecb_ema)

            # Derived signals
            da_eff = da * (1.0 - ecb * 0.4)
            ach    = (da + (1.0 - sht)) / 2.0
            e_i    = max(0.5, min(2.0, da / (1.0 - sht + 0.1)))
            ado    = min(1.0, (time.time() - self.session_start) / self.ado_duration)

            # Regime
            if   da >= 0.5 and sht >= 0.5: regime = "EXPLORE"
            elif da >= 0.5 and sht <  0.5: regime = "WAIT"
            elif da <  0.5 and sht <  0.5: regime = "REOBSERVE"
            else:                           regime = "EXPLOIT"

        self.signals = NeuroSignals(
            da=da, sht=sht, ne=ne, ach=ach,
            e_i=e_i, ado=ado, ecb=ecb,
            da_eff=da_eff, regime=regime,
        )
        return self.signals


# ???????????????????????????????????????????????????????????????????????????
# NM-JEPA Loss
# ???????????????????????????????????????????????????????????????????????????

def _sigreg(z1: torch.Tensor, z2: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """SIGReg collapse prevention (variance + covariance regularisation)."""
    std1 = torch.sqrt(z1.var(dim=0) + eps)
    std2 = torch.sqrt(z2.var(dim=0) + eps)
    var_loss = torch.mean(F.relu(1 - std1)) + torch.mean(F.relu(1 - std2))
    z1c = z1 - z1.mean(0)
    z2c = z2 - z2.mean(0)
    cov1 = (z1c.T @ z1c) / (z1.shape[0] - 1)
    cov2 = (z2c.T @ z2c) / (z2.shape[0] - 1)
    cov_loss = (
        cov1.fill_diagonal_(0).pow(2).sum() / z1.shape[1] +
        cov2.fill_diagonal_(0).pow(2).sum() / z2.shape[1]
    )
    return var_loss + cov_loss


class NMJEPALoss(nn.Module):
    """
    Neuromodulated JEPA loss function.

    Each component is independently scaled by the biologically appropriate
    neuromodulator signal, then passed through symlog for scale invariance:

      L_predict  -> DA_eff   high prediction error -> amplify JEPA gradient
      L_collapse -> 5HT      stable encoder -> stronger collapse prevention
      L_straight -> 5HT      stable encoder -> more aggressive straightening

    All components use symlog compression (DreamerV3).
    L_predict is floored at free_bits minimum (DreamerV3).

    Total:
      L = symlog(free_bits(L_predict) x da_scale)
        + symlog(L_collapse x sht_scale x lambda_collapse)
        + symlog(L_straight x sht_scale x lambda_straight)   [if trajectory given]
        + lambda_aux x L_aux   [MoE load-balancing, not symlogged]
    """

    def __init__(
        self,
        lambda_collapse:  float = 0.10,
        lambda_straight:  float = 0.02,
        lambda_aux:       float = 0.01,
        free_bits_min:    float = 0.50,
        da_scale_base:    float = 0.50,  # minimum gradient multiplier when DA=0
        da_scale_gain:    float = 1.00,  # amplification of surprising batches
    ):
        """
        Key tuning knobs (see Neuromodulator docstring for full guide):

        free_bits_min (default 0.50):
            Floor on L_predict before symlog. When DA is regularly above 0.01,
            consider lowering to 0.1-0.2 so DA variation actually affects loss
            magnitude. At 0.5, da_scale acts on 0.5 regardless of real error.

        da_scale_base + da_scale_gain:
            da_scale = da_scale_base + da_scale_gain * da_eff
            Default range: [0.5, 1.5]. With gain=3.0: [0.5, 3.5].
            Only increase gain after DA is regularly non-zero.
        """
        super().__init__()
        self.lambda_collapse  = lambda_collapse
        self.lambda_straight  = lambda_straight
        self.lambda_aux       = lambda_aux
        self.free_bits_min    = free_bits_min
        self.da_scale_base    = da_scale_base
        self.da_scale_gain    = da_scale_gain

    def forward(
        self,
        z_pred:      torch.Tensor,             # (B, D) predicted latent
        z_target:    torch.Tensor,             # (B, D) target latent (stop-grad)
        moe_aux:     torch.Tensor,             # scalar MoE load-balance loss
        signals:     NeuroSignals,
        trajectory:  Optional[torch.Tensor] = None,  # (B, T, D) for straightening
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        # ?? 1. Prediction loss (JEPA core) ????????????????????????????????
        # MSE in latent space. DA_eff modulates: high prediction error ->
        # larger gradient -> learn faster from surprising transitions.
        da_scale    = self.da_scale_base + self.da_scale_gain * signals.da_eff
        L_predict   = F.mse_loss(z_pred, z_target)
        L_predict   = free_bits(L_predict, self.free_bits_min)
        L_predict   = symlog(L_predict * da_scale)

        # ?? 2. Collapse prevention ?????????????????????????????????????????
        # SIGReg prevents representational collapse.
        # 5HT: stable encoder -> stronger regularisation.
        sht_scale   = 0.5 + signals.sht             # [0.5, 1.5]
        L_collapse  = symlog(_sigreg(z_pred, z_target) * sht_scale)
        L_collapse  = L_collapse * self.lambda_collapse

        # ?? 3. Temporal straightening ??????????????????????????????????????
        # Penalises curvature in latent trajectories -> better MPC convergence.
        # (Wang et al. 2025: arXiv:2603.12231)
        L_straight  = torch.tensor(0.0, device=z_pred.device)
        if trajectory is not None and trajectory.shape[1] >= 3:
            flat = trajectory                        # (B, T, D)
            v1   = flat[:, 1:] - flat[:, :-1]       # velocities
            v2   = v1[:, 1:] - v1[:, :-1]           # accelerations
            curv = v2.norm(dim=-1).mean()
            L_straight = symlog(curv * sht_scale) * self.lambda_straight

        # ?? 4. MoE auxiliary loss (load balancing) ?????????????????????????
        L_aux = moe_aux * self.lambda_aux

        total = L_predict + L_collapse + L_straight + L_aux

        stats = {
            "loss":       total.item(),
            "L_predict":  L_predict.item(),
            "L_collapse": L_collapse.item(),
            "L_straight": L_straight.item(),
            "L_aux":      L_aux.item(),
            "da":         signals.da,
            "sht":        signals.sht,
            "da_eff":     signals.da_eff,
            "regime":     signals.regime,
        }
        return total, stats


# ???????????????????????????????????????????????????????????????????????????
# NM-JEPA Model (assembled)
# ???????????????????????????????????????????????????????????????????????????

class NMJEPAModel(nn.Module):
    """
    Complete NM-JEPA model.

    Wraps encoder + predictor with a clean forward interface.
    The neuromodulator and loss function are separate (stateful, not nn.Module).

    Forward returns:
      z_pred    -- predicted next latent
      z_target  -- stop-gradient target latent
      moe_aux   -- MoE load-balancing loss term
    """

    def __init__(
        self,
        obs_dim:    int = 128,
        latent_dim: int = 128,
        action_dim: int = 2,
        n_experts:  int = 4,
        k_active:   int = 2,
        n_layers:   int = 2,
    ):
        super().__init__()
        self.encoder   = NMJEPAEncoder(obs_dim, latent_dim)
        self.predictor = NMJEPAPredictor(latent_dim, action_dim,
                                          n_experts, k_active, n_layers)

    def forward(
        self,
        obs_t:  torch.Tensor,   # (B, obs_dim) current observation
        obs_t1: torch.Tensor,   # (B, obs_dim) next observation
        action: torch.Tensor,   # (B, action_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # Context path -- gradients flow
        z_t        = self.encoder(obs_t)
        z_pred, aux = self.predictor(z_t, action)

        # Target path -- stop-gradient (JEPA)
        with torch.no_grad():
            z_target = self.encoder(obs_t1).detach()

        return z_pred, z_target, aux

    def param_count(self) -> Dict[str, int]:
        enc  = sum(p.numel() for p in self.encoder.parameters())
        pred = sum(p.numel() for p in self.predictor.parameters())
        return {"encoder": enc, "predictor": pred, "total": enc + pred}


# ???????????????????????????????????????????????????????????????????????????
# Training loop
# ???????????????????????????????????????????????????????????????????????????

def train(
    obs_dim:    int   = 128,
    action_dim: int   = 2,
    batch_size: int   = 32,
    n_epochs:   int   = 5,
    steps_per_epoch: int = 200,
    lr:         float = 1e-4,
    log_every:  int   = 50,
    device:     str   = "cpu",
):
    """
    Minimal NM-JEPA training loop on synthetic data.
    Replace the data generator with your own DataLoader.
    """
    dev = torch.device(device)

    model  = NMJEPAModel(obs_dim=obs_dim, action_dim=action_dim).to(dev)
    neuro  = Neuromodulator()
    loss_fn = NMJEPALoss()
    opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    counts = model.param_count()
    print(f"\nNM-JEPA  |  encoder={counts['encoder']:,}  "
          f"predictor={counts['predictor']:,}  "
          f"total={counts['total']:,} params")
    print(f"Device: {device}  |  obs_dim={obs_dim}  action_dim={action_dim}\n")

    step = 0
    for epoch in range(n_epochs):
        epoch_losses = []

        for _ in range(steps_per_epoch):
            # ?? Synthetic data (replace with real loader) ??????????????????
            obs_t  = torch.randn(batch_size, obs_dim,    device=dev)
            obs_t1 = torch.randn(batch_size, obs_dim,    device=dev)
            action = torch.randn(batch_size, action_dim, device=dev) * 0.1

            # ?? Forward ????????????????????????????????????????????????????
            z_pred, z_target, moe_aux = model(obs_t, obs_t1, action)

            # ?? Neuromodulator ?????????????????????????????????????????????
            signals = neuro.update(
                z_pred=z_pred,
                z_target=z_target,
                ne=0.5,
                action_magnitude=action.norm(dim=-1).mean().item(),
            )

            # ?? Loss ???????????????????????????????????????????????????????
            loss, stats = loss_fn(z_pred, z_target, moe_aux, signals)

            # ?? Optimise ???????????????????????????????????????????????????
            opt.zero_grad()
            loss.backward()
            agc_clip_(model.parameters(), lam=0.01)
            opt.step()

            epoch_losses.append(loss.item())
            step += 1

            if step % log_every == 0:
                print(
                    f"[ep{epoch:02d} s{step:05d}] "
                    f"loss={loss.item():.4f}  "
                    f"L_pred={stats['L_predict']:.4f}  "
                    f"regime={stats['regime']:10s}  "
                    f"DA={stats['da']:.3f}  "
                    f"5HT={stats['sht']:.3f}"
                )

        mean = float(np.mean(epoch_losses))
        print(f"Epoch {epoch:02d}  mean={mean:.4f}\n")

    print("Training complete.")
    return model


# ???????????????????????????????????????????????????????????????????????????
# Self-test
# ???????????????????????????????????????????????????????????????????????????

def self_test():
    """Verify all components work -- shapes, forward pass, loss, backward."""
    print("NM-JEPA self-test...")
    dev = torch.device("cpu")

    B, D, A = 8, 128, 2

    model   = NMJEPAModel(obs_dim=D, action_dim=A)
    neuro   = Neuromodulator()
    loss_fn = NMJEPALoss()
    opt     = torch.optim.AdamW(model.parameters(), lr=1e-4)

    obs_t  = torch.randn(B, D)
    obs_t1 = torch.randn(B, D)
    action = torch.randn(B, A) * 0.1

    z_pred, z_target, aux = model(obs_t, obs_t1, action)

    assert z_pred.shape  == (B, D), f"z_pred shape {z_pred.shape}"
    assert z_target.shape == (B, D), f"z_target shape {z_target.shape}"
    assert torch.isfinite(z_pred).all(), "z_pred has NaN/Inf"
    assert torch.isfinite(z_target).all(), "z_target has NaN/Inf"

    signals = neuro.update(z_pred, z_target, ne=0.5, action_magnitude=0.1)

    loss, stats = loss_fn(z_pred, z_target, aux, signals)

    assert torch.isfinite(loss), f"loss is {loss}"
    assert loss.item() > 0,      f"loss is zero or negative: {loss.item()}"

    opt.zero_grad()
    loss.backward()
    agc_clip_(model.parameters())
    opt.step()

    counts = model.param_count()
    print(f"  encoder:   {counts['encoder']:>8,} params")
    print(f"  predictor: {counts['predictor']:>8,} params")
    print(f"  total:     {counts['total']:>8,} params")
    print(f"  z_pred:    {z_pred.shape}  norm={z_pred.norm(dim=-1).mean():.4f}")
    print(f"  loss:      {loss.item():.4f}  regime={stats['regime']}")
    print(f"  DA={stats['da']:.3f}  5HT={stats['sht']:.3f}  da_eff={stats['da_eff']:.3f}")
    print("  All assertions passed.\n")


# ???????????????????????????????????????????????????????????????????????????
# Entry point
# ???????????????????????????????????????????????????????????????????????????

def demo():
    """
    Interactive demonstration of the neuromodulator's behaviour.

    Simulates three scenarios that exercise all four regimes and shows
    how DA, 5HT, and regime switching affect the loss in real time.

    Run with: python nm_jepa.py --demo
    """
    import math

    print("\n" + "="*60)
    print("NM-JEPA -- Neuromodulator Demo")
    print("="*60)
    print("Simulating three scenarios across 90 steps each.\n")

    dev = torch.device("cpu")
    D   = 128

    scenarios = [
        ("Smooth navigation  (low surprise)",   0.02, "EXPLOIT expected"),
        ("Terrain transition (medium surprise)", 0.25, "REOBSERVE/EXPLORE expected"),
        ("Sharp discontinuity (high surprise)", 0.70, "EXPLORE/WAIT expected"),
    ]

    for name, noise_scale, expectation in scenarios:
        print(f"Scenario: {name}")
        print(f"Expected: {expectation}")
        print(f"{'Step':>6}  {'DA':>6}  {'5HT':>6}  {'da_eff':>6}  {'Regime':<12}  {'Loss':>7}")
        print("-" * 60)

        neuro   = Neuromodulator(sht_sensitivity=10.0, da_scale_gain=1.0)
        loss_fn = NMJEPALoss(free_bits_min=0.5)

        model = NMJEPAModel(obs_dim=D, action_dim=2).to(dev)
        opt   = torch.optim.AdamW(model.parameters(), lr=1e-4)

        for step in range(1, 91):
            # Simulate different visual dynamics via noise scale
            obs_t  = torch.randn(16, D, device=dev)
            # Target is obs_t plus noise -- higher noise = more visual change
            obs_t1 = obs_t + torch.randn(16, D, device=dev) * noise_scale
            action = torch.randn(16, 2, device=dev) * 0.1

            z_pred, z_target, aux = model(obs_t, obs_t1, action)

            signals = neuro.update(
                z_pred=z_pred,
                z_target=z_target,
                ne=0.5,
                action_magnitude=action.norm(dim=-1).mean().item(),
            )

            loss, stats = loss_fn(z_pred, z_target, aux, signals)
            opt.zero_grad(); loss.backward()
            agc_clip_(model.parameters())
            opt.step()

            if step % 15 == 0:
                print(
                    f"{step:>6}  "
                    f"{signals.da:>6.3f}  "
                    f"{signals.sht:>6.3f}  "
                    f"{signals.da_eff:>6.3f}  "
                    f"{signals.regime:<12}  "
                    f"{loss.item():>7.4f}"
                )

        print()

    # Tuning comparison
    print("="*60)
    print("Tuning comparison -- default vs aggressive (high surprise scenario)")
    print(f"{'Config':<28}  {'DA':>6}  {'Loss':>7}  {'Regime'}")
    print("-" * 60)

    configs = [
        ("Default (gain=1.0, floor=0.5)",  dict(da_scale_gain=1.0), dict(free_bits_min=0.5)),
        ("High gain (gain=3.0, floor=0.5)", dict(da_scale_gain=3.0), dict(free_bits_min=0.5)),
        ("Low floor (gain=1.0, floor=0.1)", dict(da_scale_gain=1.0), dict(free_bits_min=0.1)),
        ("Aggressive (gain=3.0, floor=0.1)",dict(da_scale_gain=3.0), dict(free_bits_min=0.1)),
    ]

    for label, neuro_kw, loss_kw in configs:
        neuro   = Neuromodulator(**neuro_kw)
        loss_fn = NMJEPALoss(**loss_kw)
        model   = NMJEPAModel(obs_dim=D, action_dim=2).to(dev)
        opt     = torch.optim.AdamW(model.parameters(), lr=1e-4)

        last_da, last_loss, last_regime = 0.0, 0.0, "--"
        for _ in range(50):
            obs_t  = torch.randn(16, D, device=dev)
            obs_t1 = obs_t + torch.randn(16, D, device=dev) * 0.7
            action = torch.randn(16, 2, device=dev) * 0.1
            z_pred, z_target, aux = model(obs_t, obs_t1, action)
            signals = neuro.update(z_pred, z_target, ne=0.5,
                                    action_magnitude=action.norm(dim=-1).mean().item())
            loss, _ = loss_fn(z_pred, z_target, aux, signals)
            opt.zero_grad(); loss.backward()
            agc_clip_(model.parameters()); opt.step()
            last_da, last_loss, last_regime = signals.da, loss.item(), signals.regime

        print(f"{label:<28}  {last_da:>6.3f}  {last_loss:>7.4f}  {last_regime}")

    print()
    print("Key insight: lowering free_bits_min lets DA actually affect loss magnitude.")
    print("Increasing da_scale_gain amplifies surprising batches' gradient contribution.")
    print("\nRun 'python nm_jepa.py --epochs 5' to see full training with live neuro logs.")



    parser = argparse.ArgumentParser(description="NM-JEPA standalone")
    parser.add_argument("--epochs",     type=int,   default=0,
                        help="training epochs (0 = self-test only)")
    parser.add_argument("--demo",       action="store_true",
                        help="run neuromodulator demo across 3 scenarios + tuning comparison")
    parser.add_argument("--obs-dim",    type=int,   default=128)
    parser.add_argument("--action-dim", type=int,   default=2)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--steps",      type=int,   default=200)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--log-every",  type=int,   default=50)
    parser.add_argument("--device",     type=str,   default="cpu")
    args = parser.parse_args()

    self_test()

    if args.demo:
        demo()

    if args.epochs > 0:
        train(
            obs_dim    = args.obs_dim,
            action_dim = args.action_dim,
            batch_size = args.batch_size,
            steps_per_epoch = args.steps,
            n_epochs   = args.epochs,
            lr         = args.lr,
            log_every  = args.log_every,
            device     = args.device,
        )
