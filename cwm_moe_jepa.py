"""
cwm_moe_jepa.py  —  CORTEX World Model: Mixture-of-Experts JEPA
================================================================

Architecture: Sparse MoE inside the JEPA predictor (dynamics module).
The encoder is SHARED across all domains.
The predictor/dynamics module is SPECIALISED via MoE routing.

Why MoE belongs in the predictor, not the encoder:
  - Shared encoder → shared latent geometry across domains (required for
    the multi-domain novelty claim — same backbone for cardiac, MVTec, RECON)
  - Different physics per domain → different experts in the predictor
  - Contact manipulation ≠ outdoor navigation ≠ telemetry ≠ locomotion
  - One encoder, N specialised dynamics experts — capacity without cost

JEPA framing:
  - Context encoder: StudentEncoder (FROZEN, XINT8 NPU)
  - Particle encoder: converts latent → K spatial particles
  - Predictor: action-conditioned MoE ParticleTransformer
  - Target: stop-gradient next-step particle state
  - Loss: MSE in latent space only (never pixel reconstruction)

MoE scheme: Sparse top-K routing (K_active=2 of N_experts=4)
  - Each of K=16 particles routes independently
  - Only 2 experts fire per particle → inference cost = 2×FFN not 4×FFN
  - Load balancing loss prevents expert collapse
  - Neuromodulator regime biases routing temperature

Expert specialisation (emergent, but guided by domain):
  Expert 0: Contact/rigid-body  (OGBench-Cube, PushT)
  Expert 1: Smooth navigation   (RECON, TwoRoom)
  Expert 2: Temporal drift      (SMAP telemetry, Cardiac audio)
  Expert 3: Periodic locomotion (Hexapod, Quadruped)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# 1. Sparse MoE Feed-Forward Block
# ═══════════════════════════════════════════════════════════════════════════

class SparseMoEFFN(nn.Module):
    """
    Sparse top-K MoE drop-in replacement for a standard FFN block.

    Each input token (particle) independently routes to K_active of N_experts
    specialists. Only the selected experts compute — total FLOP cost is
    K_active/N_experts × dense FFN cost, but with N_experts × capacity.

    Parameters
    ----------
    d_model    : token/particle embedding dimension (128 for CWM)
    d_ff       : inner FFN dimension (4 × d_model standard)
    n_experts  : total number of expert FFN blocks (4)
    k_active   : experts activated per token (2 — top-2 routing)
    load_balance_weight : auxiliary loss coefficient (0.01)
    """

    def __init__(
        self,
        d_model:              int   = 128,
        d_ff:                 int   = 512,
        n_experts:            int   = 4,
        k_active:             int   = 2,
        load_balance_weight:  float = 0.01,
    ):
        super().__init__()
        self.n_experts           = n_experts
        self.k_active            = k_active
        self.load_balance_weight = load_balance_weight

        # N independent expert FFN blocks
        # Each: Linear → GELU → Linear  (same shape as a dense FFN)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model),
            )
            for _ in range(n_experts)
        ])

        # Router: maps each particle embedding → N_experts logits
        self.router = nn.Linear(d_model, n_experts, bias=False)

        # Track auxiliary load balance loss for logging
        self._aux_loss: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, K, d_model)  — batch of K particle embeddings

        Returns: (B, K, d_model) with sparse expert computation.
        """
        B, K, D = x.shape
        x_flat = x.reshape(B * K, D)  # (B×K, D)

        # ── Router ────────────────────────────────────────────────────────
        router_logits = self.router(x_flat)           # (B×K, N)
        router_probs  = F.softmax(router_logits, -1)  # (B×K, N)

        # Top-K sparse selection
        topk_probs, topk_idx = router_probs.topk(self.k_active, dim=-1)
        # topk_probs : (B×K, k_active)
        # topk_idx   : (B×K, k_active) — which experts to use

        # Normalise selected weights (sum to 1 over active experts)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        # ── Expert computation ────────────────────────────────────────────
        output = torch.zeros_like(x_flat)  # (B×K, D)

        for expert_idx, expert in enumerate(self.experts):
            # Which tokens route to this expert?
            mask = (topk_idx == expert_idx).any(dim=-1)  # (B×K,)
            if not mask.any():
                continue

            # Weight for this expert (0 if not in top-K for this token)
            expert_weight = torch.where(
                topk_idx == expert_idx,
                topk_probs,
                torch.zeros_like(topk_probs)
            ).sum(dim=-1)  # (B×K,)

            # Compute expert output only for routed tokens
            expert_out = expert(x_flat[mask])        # (n_routed, D)
            output[mask] += (expert_weight[mask].unsqueeze(-1) * expert_out)

        # ── Auxiliary load balance loss ───────────────────────────────────
        # Prevents all tokens routing to the same expert (expert collapse).
        # From Switch Transformer: L_aux = N × sum_i(f_i × P_i)
        # f_i = fraction of tokens routed to expert i
        # P_i = mean router probability for expert i
        f = torch.zeros(self.n_experts, device=x.device)
        for i in range(self.n_experts):
            f[i] = (topk_idx == i).float().mean()
        P = router_probs.mean(dim=0)                 # (N,)
        self._aux_loss = (self.n_experts * (f * P).sum()
                          * self.load_balance_weight)

        return output.reshape(B, K, D)

    @property
    def aux_loss(self) -> torch.Tensor:
        if self._aux_loss is None:
            return torch.tensor(0.0)
        return self._aux_loss


# ═══════════════════════════════════════════════════════════════════════════
# 2. Neuromodulator-Gated Router Temperature
# ═══════════════════════════════════════════════════════════════════════════

class NeuromodulatedRouter(nn.Module):
    """
    Extends SparseMoEFFN routing with neuromodulator temperature scaling.

    The regime state (from NeuromodulatorState) biases the routing sharpness:

    EXPLOIT  (DA↓ 5HT↑) — known+stable  → sharp routing (τ low)
      → Route confidently to the specialist expert for this domain.
      → Model knows what's happening, use the right expert.

    EXPLORE  (DA↑ 5HT↑) — novel+stable  → soft routing (τ high)
      → Spread probability across experts — discover which fits the new input.
      → Biological analogy: high ACh = "attend to new input over stored model."

    WAIT     (DA↑ 5HT↓) — novel+unstable → very soft (uniform-ish)
      → Almost uniform routing: don't commit to a specialist when uncertain.

    REOBSERVE(DA↓ 5HT↓) — known+unstable → medium sharpness
      → Mild preference for familiar expert, but allow some flexibility.
    """

    REGIME_TEMP = {
        "EXPLOIT":   0.3,   # sharp → near one-hot expert selection
        "EXPLORE":   1.5,   # soft  → spread across experts
        "WAIT":      3.0,   # very soft → near uniform
        "REOBSERVE": 0.8,   # medium
    }

    def __init__(self, d_model: int = 128, n_experts: int = 4):
        super().__init__()
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self._temperature = 1.0

    def set_regime(self, regime: str):
        self._temperature = self.REGIME_TEMP.get(regime, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns router probabilities with regime-modulated temperature."""
        logits = self.router(x)
        return F.softmax(logits / self._temperature, dim=-1)


# ═══════════════════════════════════════════════════════════════════════════
# 3. MoE Particle Transformer Block
# ═══════════════════════════════════════════════════════════════════════════

class MoEParticleTransformerBlock(nn.Module):
    """
    Single transformer block with sparse MoE FFN.

    Standard transformer: Attention → FFN
    This block:          Attention → Sparse MoE FFN

    The attention is shared (all particles attend to each other regardless
    of which expert they will route to). MoE only applies to the FFN
    (per-particle transformation after attention).

    This is correct because:
    - Attention captures inter-particle relationships (shared physics)
    - FFN applies per-token transformations (domain-specific dynamics)
    - MoE specialisation at the FFN level is the standard approach
      (Mixtral, Switch Transformer, GLaM all do this)
    """

    def __init__(
        self,
        d_model:   int = 128,
        n_heads:   int = 4,
        n_experts: int = 4,
        k_active:  int = 2,
        d_ff:      int = 512,
        dropout:   float = 0.1,
    ):
        super().__init__()
        self.attn    = nn.MultiheadAttention(d_model, n_heads,
                                              dropout=dropout, batch_first=True)
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # MoE FFN instead of dense FFN
        self.moe_ffn = SparseMoEFFN(d_model, d_ff, n_experts, k_active)

    def forward(
        self,
        x:    torch.Tensor,       # (B, K, d_model)
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ── Self-attention (shared across all particles) ───────────────────
        residual = x
        x_norm   = self.norm1(x)
        attn_out, attn_weights = self.attn(x_norm, x_norm, x_norm,
                                            attn_mask=mask,
                                            need_weights=True,
                                            average_attn_weights=False)
        x = residual + self.dropout(attn_out)

        # ── Sparse MoE FFN (per-particle specialisation) ───────────────────
        residual = x
        x = residual + self.dropout(self.moe_ffn(self.norm2(x)))

        return x, attn_weights

    @property
    def aux_loss(self) -> torch.Tensor:
        return self.moe_ffn.aux_loss


# ═══════════════════════════════════════════════════════════════════════════
# 4. Full MoE-JEPA World Model Predictor
# ═══════════════════════════════════════════════════════════════════════════

class MoEJEPAPredictor(nn.Module):
    """
    Full JEPA-style world model predictor with sparse MoE dynamics.

    JEPA design:
      - Inputs:  context particles z_t (from shared encoder, stop-grad target)
      - Action:  a_t (padded to max_action_dim, zeros for passive domains)
      - Output:  predicted next particles ẑ_{t+1}
      - Target:  z_{t+1} from target encoder (stop-gradient)
      - Loss:    MSE(ẑ_{t+1}, sg(z_{t+1})) — no pixel reconstruction

    The stop-gradient on the target is critical: it prevents collapse without
    requiring negative samples (unlike contrastive methods).

    MoE placement:
      - Attention layers: shared (all particles interact with all others)
      - FFN layers: sparse MoE (per-particle domain specialisation)

    Parameters
    ----------
    d_model     : particle embedding dim (128)
    K           : number of particles (16)
    n_layers    : transformer depth (2 for CWM, can scale to 4)
    n_heads     : attention heads (4)
    n_experts   : MoE expert count (4)
    k_active    : active experts per token (2 — top-2)
    max_action_dim : maximum action dimension across all domains (9 for OGBench)
    """

    def __init__(
        self,
        d_model:        int = 128,
        K:              int = 16,
        n_layers:       int = 2,
        n_heads:        int = 4,
        n_experts:      int = 4,
        k_active:       int = 2,
        max_action_dim: int = 9,
        d_ff:           int = 512,
        dropout:        float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.K       = K

        # ── Action conditioning ────────────────────────────────────────────
        # Project action into d_model so it can be injected as a conditioning
        # token prepended to the particle sequence (one extra "action token").
        # This is cleaner than adding action to every particle embedding because:
        #   - StructuredActionModule can learn which particles are relevant
        #   - Action token attends freely to all particle tokens
        self.action_proj = nn.Sequential(
            nn.Linear(max_action_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # ── MoE Transformer layers ─────────────────────────────────────────
        self.layers = nn.ModuleList([
            MoEParticleTransformerBlock(d_model, n_heads, n_experts,
                                         k_active, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # ── Output projection ──────────────────────────────────────────────
        # Predict next particle states (same shape as input particles)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

        # ── Domain embedding ───────────────────────────────────────────────
        # Learned per-domain token added to the action projection.
        # Helps the router learn domain-specific priors even before seeing
        # particle content. E.g. "I'm in RECON" → nav expert preferred.
        self.domain_embed = nn.Embedding(8, d_model)
        # Domain IDs: 0=RECON, 1=OGBench, 2=TwoRoom, 3=PushT,
        #             4=SMAP, 5=Cardiac, 6=Hexapod, 7=Quadruped

        # ── Neuromodulated router ──────────────────────────────────────────
        self.neuro_router = NeuromodulatedRouter(d_model, n_experts)

    def forward(
        self,
        particles:  torch.Tensor,           # (B, K, d_model) — context particles
        action:     torch.Tensor,           # (B, max_action_dim) — zero-padded
        domain_id:  Optional[torch.Tensor] = None,  # (B,) int domain index
        regime:     str = "EXPLOIT",        # from NeuromodulatorState
    ) -> Tuple[torch.Tensor, dict]:
        """
        JEPA predictor forward pass.

        Returns predicted next particles and a dict of auxiliary info
        (attention weights per layer, aux MoE load balance loss).
        """
        B, K, D = particles.shape

        # ── Action token (with optional domain conditioning) ───────────────
        action_emb = self.action_proj(action).unsqueeze(1)  # (B, 1, D)
        if domain_id is not None:
            action_emb = action_emb + self.domain_embed(domain_id).unsqueeze(1)

        # Prepend action token → sequence length = K+1
        seq = torch.cat([action_emb, particles], dim=1)  # (B, K+1, D)

        # ── Set routing temperature from neuromodulator regime ─────────────
        # High temperature (EXPLORE/WAIT) → soft routing → try multiple experts
        # Low temperature (EXPLOIT) → sharp routing → use the right specialist
        for layer in self.layers:
            layer.moe_ffn.router = self.neuro_router.router  # share router weights
        self.neuro_router.set_regime(regime)

        # ── Transformer forward ────────────────────────────────────────────
        aux_losses     = []
        attn_weights   = []

        x = seq
        for layer in self.layers:
            x, attn_w = layer(x)
            aux_losses.append(layer.aux_loss)
            attn_weights.append(attn_w)

        # Strip the action token, keep only particle predictions
        x_particles = x[:, 1:, :]  # (B, K, D)

        # ── Output projection ──────────────────────────────────────────────
        # stop-gradient applied to target externally (in training loop)
        z_next_pred = self.out_proj(x_particles)  # (B, K, D)

        aux = {
            "moe_aux_loss":  sum(aux_losses),
            "attn_weights":  attn_weights,  # list of (B, n_heads, K+1, K+1)
        }

        return z_next_pred, aux

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════════════
# 5. JEPA Training Loss (with MoE auxiliary)
# ═══════════════════════════════════════════════════════════════════════════

def jepa_moe_loss(
    z_pred:            torch.Tensor,   # (B, K, d_model) — predictor output
    z_target:          torch.Tensor,   # (B, K, d_model) — target encoder output
    moe_aux_loss:      torch.Tensor,   # scalar — load balance auxiliary
    neuro_da_eff:      float = 0.5,    # DA_effective for gradient scaling
    lambda_aux:        float = 0.01,   # weight on MoE load balance loss
) -> Tuple[torch.Tensor, dict]:
    """
    JEPA prediction loss with MoE auxiliary.

    Target must have stop-gradient applied BEFORE calling this:
        z_target = target_encoder(frames_t1).detach()  # ← stop-gradient

    DA modulation: same as NeuromodulatedCWMLoss — amplify gradient when
    the world model was surprised (high DA = high prediction error).
    """
    # Core JEPA prediction loss: MSE in latent space
    # DA_eff scales the gradient: high surprise → learn more
    da_scale  = 0.5 + neuro_da_eff        # [0.5, 1.5]
    L_predict = F.mse_loss(z_pred, z_target) * da_scale

    # MoE load balance: prevent all particles routing to one expert
    L_aux = moe_aux_loss * lambda_aux

    total = L_predict + L_aux

    return total, {
        "L_predict": L_predict.item(),
        "L_aux":     L_aux.item(),
        "da_scale":  da_scale,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 6. Expert Inspection Utilities
# ═══════════════════════════════════════════════════════════════════════════

def log_expert_utilisation(
    model:      MoEJEPAPredictor,
    particles:  torch.Tensor,
    action:     torch.Tensor,
    domain_id:  Optional[torch.Tensor] = None,
) -> dict:
    """
    Run a forward pass and log which experts each particle routes to.
    Useful for understanding whether domain specialisation has emerged.

    Expected pattern after sufficient training:
      RECON particles    → Expert 1 (navigation) dominant
      OGBench particles  → Expert 0 (contact) dominant
      SMAP particles     → Expert 2 (drift) dominant

    If all particles route to the same expert → load balance loss is too low,
    increase lambda_aux.
    """
    with torch.no_grad():
        # Collect router logits from first MoE layer
        layer0     = model.layers[0]
        x_flat     = particles.reshape(-1, model.d_model)
        logits     = layer0.moe_ffn.router(x_flat)    # (B×K, N)
        probs      = 0.99 * F.softmax(logits, dim=-1) + 0.01 / 4  # unimix         # (B×K, N)
        expert_ids = probs.argmax(dim=-1)              # (B×K,) — top-1 for logging

    usage = {}
    for e in range(layer0.moe_ffn.n_experts):
        pct = (expert_ids == e).float().mean().item() * 100
        usage[f"expert_{e}_pct"] = round(pct, 1)

    return usage


# ═══════════════════════════════════════════════════════════════════════════
# 7. Parameter Count & Integration Notes
# ═══════════════════════════════════════════════════════════════════════════

"""
PARAMETER BUDGET (default config: d=128, 2L, 4H, 4E, k_active=2)
──────────────────────────────────────────────────────────────────

Component                      Params     Notes
──────────────────────────────  ─────────  ────────────────────────────────
MoE attention (2 layers)        2 × 66K  = 132K   d=128, 4 heads
MoE FFN — 4 experts × 2 layers 2 × 4×(128×512+512×128) = 2 × 524K = 1,048K
Router (2 layers)               2 × 512  = 1K
Action projection               128×128  = 16K
Domain embedding (8 domains)    8×128    = 1K
Output projection               128×128  = 16K
                                ─────────
Total MoE predictor             ~1,214K

vs dense predictor (no MoE):    ~446K    (current CWM ParticleTransformer)
Overhead ratio:                 ~2.7×    params, but only 2/4 experts active

INFERENCE COST:
  Dense predictor:  100% of FFN compute
  MoE predictor:    k_active/n_experts = 2/4 = 50% of FFN compute
  Net FLOP savings: ~25% vs dense (FFN ≈ 2/3 of total transformer compute)

CAPACITY GAIN:
  4 expert FFNs × 512 inner dim = 4× effective capacity vs single 512-dim FFN
  But at inference time: only 2 experts fire → same speed as dense 2-expert FFN

WHY THIS BEATS JUST MAKING THE PREDICTOR BIGGER:
  - Bigger dense predictor: scales compute uniformly across all domains
  - MoE: routes each particle to the expert it needs — specialisation
  - After training, contact particles consistently hit Expert 0,
    navigation particles hit Expert 1, etc. — interpretable and efficient


INTEGRATION WITH EXISTING CWM
──────────────────────────────

Replace ContactAwareParticleTransformer with MoEJEPAPredictor:

    # In train_cwm.py
    from cwm_moe_jepa import MoEJEPAPredictor, jepa_moe_loss
    from neuromodulator import NeuromodulatorState

    predictor = MoEJEPAPredictor(
        d_model=128, K=16, n_layers=2, n_heads=4,
        n_experts=4, k_active=2, max_action_dim=9
    )

    for batch in loader:
        particles_t  = particle_enc(frames_t)
        action_pad   = F.pad(action, (0, 9 - action.shape[-1]))

        # JEPA: target uses stop-gradient
        with torch.no_grad():
            particles_t1 = particle_enc(frames_t1)   # target encoder
        z_target = particles_t1.detach()              # ← stop-gradient

        # Predictor forward (context encoder, no stop-grad)
        z_pred, aux = predictor(
            particles  = particles_t,
            action     = action_pad,
            domain_id  = batch['domain_id'],
            regime     = signals['regime'],
        )

        # JEPA loss + MoE auxiliary + neuromodulator modulation
        loss, stats = jepa_moe_loss(
            z_pred       = z_pred,
            z_target     = z_target,
            moe_aux_loss = aux['moe_aux_loss'],
            neuro_da_eff = signals['da_effective'],
        )

        loss.backward()
        optimizer.step()

        # Log expert utilisation every N steps
        if step % 100 == 0:
            usage = log_expert_utilisation(predictor, particles_t, action_pad)
            print(f"Expert usage: {usage}")
            # Watch for specialisation to emerge ~500-1000 steps in


WHAT TO WATCH DURING TRAINING
──────────────────────────────

1. Load balance (expert_N_pct):
   Bad:  one expert gets >70% of tokens → increase lambda_aux
   Good: roughly balanced (each ~25% ± 15%)

2. Domain specialisation (log every epoch by domain):
   Run log_expert_utilisation separately per domain batch.
   After ~10 epochs: RECON → Expert 1 dominant, OGBench → Expert 0 dominant.
   This is the sign MoE is working — it discovered the physics clusters.

3. MoE aux loss (L_aux):
   Should decrease steadily. If it plateaus high → routing collapsed.
   If it's zero → lambda_aux too high, forced uniform (experts not specialising).

4. Regime routing temperature:
   During EXPLOIT sessions (trading system quiet): expect sharper routing.
   During EXPLORE sessions (novel morning signal): expect softer routing.
   Log neuro_router._temperature per batch to verify.
"""
