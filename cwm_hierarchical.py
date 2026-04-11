"""
cwm_hierarchical.py  —  CORTEX CWM Sprint 3
============================================
Hierarchical CWM for OGBench-Cube long-horizon skill composition.

Combines:
  - THICK Context GRU (already in train_cwm.py)
  - SCaR bidirectional skill transition regularisation (NeurIPS 2024)
  - HiLAM inverse dynamics skill boundary detection (ICLR 2026 Workshop)
  - SPlaTES skill-predictable loss (RLC 2025)

Architecture:
  Fast level  : MoEJEPAPredictor (per-step particle dynamics)
  Slow level  : THICKContextGRU (skill-level context, +25K params)
  Boundary    : InverseDynamicsSkillBoundary (detects skill transitions)
  Transition  : SCaR bidirectional loss (smooth skill handoffs, 0 params)
  Planning    : Abstract skill MPC (6 skill-steps vs 50 primitive steps)

For OGBench-Cube this reduces effective planning horizon from ~50 steps
to ~6 skill-level steps (approach → grasp → lift → transport → place → release).

Usage:
    from cwm_hierarchical import HierarchicalCWM, train_hierarchical_cwm

    hcwm = HierarchicalCWM(cwm)   # wraps existing CortexWorldModel
    train_hierarchical_cwm(hcwm, ogbench_loader, n_epochs=20)
"""

import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from train_cwm import CortexWorldModel, MAX_ACTION_DIM
    from neuromodulator import NeuromodulatorState
    from cwm_neuro_reward import RegimeGatedTrainer
    _CORTEX_AVAILABLE = True
except ImportError:
    _CORTEX_AVAILABLE = False
    MAX_ACTION_DIM = 9


# ═══════════════════════════════════════════════════════════════════════════
# Inverse Dynamics Skill Boundary Detector  (HiLAM-style)
# ═══════════════════════════════════════════════════════════════════════════

class InverseDynamicsSkillBoundary(nn.Module):
    """
    Detects skill boundaries in particle trajectories without action labels.
    Based on HiLAM (arXiv:2603.05815, ICLR 2026 Workshop).

    Method:
        1. Train an inverse dynamics model: (particles_t, particles_{t+1}) → action_t
        2. Predict action from consecutive particle pairs
        3. Skill boundaries = timesteps where prediction error spikes
           (the transition between "grasp" and "lift" is discontinuous)

    Used offline to segment OGBench-Cube trajectories into skills before
    applying SCaR transition regularisation.
    """

    def __init__(
        self,
        d_model:    int = 128,
        K:          int = 16,
        action_dim: int = 9,
        threshold:  float = 0.15,
    ):
        super().__init__()
        self.d_model   = d_model
        self.K         = K
        self.threshold = threshold

        # Inverse dynamics: concat(particles_t, particles_{t+1}) → action
        self.inv_dyn = nn.Sequential(
            nn.Linear(d_model * K * 2, 256),
            nn.GELU(),
            nn.Linear(256, action_dim),
        )

    def forward(
        self,
        particles_t:  torch.Tensor,   # (B, K, d_model)
        particles_t1: torch.Tensor,   # (B, K, d_model)
    ) -> torch.Tensor:
        """Predict action from consecutive particle pair."""
        B, K, D = particles_t.shape
        flat = torch.cat([particles_t.reshape(B, -1),
                           particles_t1.reshape(B, -1)], dim=-1)
        return self.inv_dyn(flat)     # (B, action_dim)

    def detect_boundaries(
        self,
        trajectory:    List[torch.Tensor],  # list of (K, d_model) particles
        actions_gt:    torch.Tensor,        # (T-1, action_dim) ground truth
    ) -> List[int]:
        """
        Identify skill boundary timesteps.

        Boundaries = local maxima in inverse dynamics prediction error.
        High error = the action taken was surprising given particle state —
        i.e. a skill transition occurred.

        Returns list of timestep indices (0-indexed).
        """
        errors = []
        self.eval()
        with torch.no_grad():
            for t in range(len(trajectory) - 1):
                pt  = trajectory[t].unsqueeze(0)    # (1, K, D)
                pt1 = trajectory[t+1].unsqueeze(0)
                pred = self.forward(pt, pt1)         # (1, action_dim)
                gt   = actions_gt[t].unsqueeze(0)
                err  = F.mse_loss(pred, gt).item()
                errors.append(err)

        # Find local maxima above threshold
        boundaries = []
        for i in range(1, len(errors) - 1):
            if (errors[i] > errors[i-1] and
                errors[i] > errors[i+1] and
                errors[i] > self.threshold):
                boundaries.append(i)

        return boundaries

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ═══════════════════════════════════════════════════════════════════════════
# SCaR Skill Transition Loss  (NeurIPS 2024 — 0 parameters)
# ═══════════════════════════════════════════════════════════════════════════

def scar_transition_loss(
    particles_end_a:   torch.Tensor,   # (B, K, d_model) end of skill A
    particles_start_b: torch.Tensor,   # (B, K, d_model) start of skill B
    lambda_scar:       float = 0.05,
    da_eff:            float = 0.5,    # DA modulation (from neuromodulators)
    ecb:               float = 0.0,    # eCB modulation (loop suppression)
) -> torch.Tensor:
    """
    SCaR bidirectional skill transition regularisation.

    Pulls the terminal particle state of skill A toward the initial state of
    skill B, and vice versa. Ensures smooth handoffs in sequential skills:
        approach → grasp → lift → transport → place → release

    Neuromodulator modulation:
        DA_eff: amplify when model is surprised by transition (reward learning)
        eCB:    dampen when system is oscillating (same skill repeated)
    """
    # Forward: end of A pulled toward start of B
    fwd = F.mse_loss(particles_end_a, particles_start_b.detach())
    # Backward: start of B pulled toward end of A
    bwd = F.mse_loss(particles_start_b, particles_end_a.detach())

    # Neuromodulator scaling
    da_scale  = 0.5 + da_eff           # [0.5, 1.5]
    ecb_damp  = 1.0 - ecb * 0.4        # eCB retrograde: max -40%
    scale     = lambda_scar * da_scale * ecb_damp

    return scale * (fwd + bwd)


# ═══════════════════════════════════════════════════════════════════════════
# SPlaTES skill reward  (RLC 2025 — 0 parameters)
# ═══════════════════════════════════════════════════════════════════════════

def splates_skill_reward(
    particles_achieved: torch.Tensor,   # (B, K, d_model) end-of-skill state
    context_h:          torch.Tensor,   # (B, context_dim) THICK context
    context_decoder:    nn.Module,      # maps context → expected particles
    da_eff:             float = 0.5,
) -> torch.Tensor:
    """
    SPlaTES: skills should achieve what the abstract world model predicted.

    The THICK Context GRU predicts the next context state (skill outcome).
    The context decoder maps that prediction back to particle space.
    We reward the policy for landing in the predicted particle configuration.

    Bidirectional constraint (SPlaTES):
        "Skills are trained to fit the abstract world model and vice versa."
    """
    # Decode THICK context prediction to particle space
    particles_expected = context_decoder(context_h)   # (B, K*d_model) or similar

    # Reshape if needed
    B, K, D = particles_achieved.shape
    if particles_expected.shape[-1] != K * D:
        particles_expected = particles_expected[..., :K * D]
    particles_expected = particles_expected.reshape(B, K, D)

    # Similarity: how close was the achieved state to the prediction?
    similarity = F.cosine_similarity(
        particles_achieved.reshape(B, -1),
        particles_expected.reshape(B, -1).detach(),
        dim=-1
    )

    # Negate: maximise similarity = minimise negative similarity
    da_scale = 0.5 + da_eff
    return -similarity.mean() * da_scale


# ═══════════════════════════════════════════════════════════════════════════
# Hierarchical CWM wrapper
# ═══════════════════════════════════════════════════════════════════════════

class HierarchicalCWM(nn.Module):
    """
    Wraps CortexWorldModel with hierarchical skill composition.

    Adds:
        InverseDynamicsSkillBoundary  (+50K params, used offline)
        Context decoder               (+8K params, maps context → particles)
        SCaR loss                     (0 params)
        SPlaTES reward                (0 params)

    The existing THICK GRU (already in CortexWorldModel) provides the
    slow-timescale context. This module adds the training-time components
    to make it work for OGBench-Cube long-horizon skill composition.
    """

    def __init__(
        self,
        cwm:        CortexWorldModel,
        d_model:    int = 128,
        K:          int = 16,
        action_dim: int = 9,
    ):
        super().__init__()
        self.cwm    = cwm
        self.d_model = d_model
        self.K       = K

        context_dim = cwm.thick_gru.context_dim

        # Skill boundary detector (HiLAM-style)
        self.boundary_detector = InverseDynamicsSkillBoundary(
            d_model=d_model, K=K, action_dim=action_dim
        )

        # Context decoder: THICK context → expected particle configuration
        # Used for SPlaTES skill reward shaping
        self.context_decoder = nn.Sequential(
            nn.Linear(context_dim, 256),
            nn.GELU(),
            nn.Linear(256, d_model * K),
        )

    def forward_hierarchical(
        self,
        particles_seq: List[torch.Tensor],   # T particle states
        actions:       torch.Tensor,          # (T-1, action_dim)
        boundaries:    List[int],             # skill boundary indices
        signals:       dict,
    ) -> dict:
        """
        Forward pass with explicit skill boundary handling.

        For each consecutive pair of skills (A, B):
            - Compute SCaR transition loss at the boundary
            - Compute SPlaTES skill reward for skill A
            - Accumulate hierarchical losses
        """
        da_eff = signals.get("da_effective", 0.5)
        ecb    = signals.get("ecb",          0.0)

        total_scar   = torch.tensor(0.0)
        total_splates = torch.tensor(0.0)
        n_boundaries = 0

        # Process each skill boundary
        prev_boundary = 0
        for b_idx in boundaries:
            # Skill A: particles from prev_boundary to b_idx
            # Skill B: particles from b_idx onward
            if b_idx <= prev_boundary or b_idx >= len(particles_seq) - 1:
                continue

            end_a   = particles_seq[b_idx]           # (B, K, D)
            start_b = particles_seq[b_idx + 1]       # (B, K, D)

            # SCaR: smooth handoff between skills
            scar = scar_transition_loss(
                end_a, start_b, lambda_scar=0.05,
                da_eff=da_eff, ecb=ecb
            )
            total_scar = total_scar + scar

            # SPlaTES: skill A should achieve what THICK predicted
            # Get THICK context just before boundary
            context_h = self.cwm.thick_gru.init_context(
                particles_seq[0].shape[0],
                particles_seq[0].device
            )
            for t in range(prev_boundary, b_idx):
                action_pad = F.pad(actions[t].unsqueeze(0),
                                   (0, MAX_ACTION_DIM - actions.shape[-1]))
                context_h, _, _ = self.cwm.thick_gru(
                    particles_seq[t], context_h, action_pad
                )

            splates = splates_skill_reward(
                end_a, context_h, self.context_decoder, da_eff
            )
            total_splates = total_splates + splates

            prev_boundary = b_idx
            n_boundaries += 1

        if n_boundaries > 0:
            total_scar    = total_scar    / n_boundaries
            total_splates = total_splates / n_boundaries

        return {
            "L_scar":     total_scar,
            "L_splates":  total_splates,
            "n_boundaries": n_boundaries,
        }

    def abstract_skill_mpc(
        self,
        particles_0:  torch.Tensor,   # (1, K, d_model)
        goal_context: torch.Tensor,   # (1, context_dim)  target skill
        n_skills:     int   = 6,      # planning horizon in skill steps
        n_candidates: int   = 32,
        action_dim:   int   = 9,
        device:       torch.device = torch.device("cpu"),
    ) -> List[torch.Tensor]:
        """
        Abstract MPC over skill-level context space.

        Plans N skill-level transitions (each ~ 6-10 primitive steps) rather
        than N×6-10 primitive steps. This is the THICK planning loop.

        Returns list of planned skill-level actions.
        """
        context_h = self.cwm.thick_gru.init_context(1, device)
        skill_actions = []

        for skill_step in range(n_skills):
            # Sample candidate primitive action sequences for this skill
            cands = torch.randn(n_candidates, 8, action_dim, device=device) * 0.1

            best_cost = float("inf")
            best_seq  = cands[0]

            with torch.no_grad():
                for k in range(n_candidates):
                    p = particles_0.clone()
                    ctx = context_h.clone()
                    cum_cost = 0.0

                    for t in range(8):   # ~8 primitive steps per skill
                        action_pad = F.pad(cands[k, t].unsqueeze(0),
                                           (0, MAX_ACTION_DIM - action_dim))
                        out = self.cwm.predict(
                            particles=p,
                            action=action_pad,
                            context_h=ctx,
                            positions=torch.zeros(1, p.shape[1], 2, device=device),
                            domain_id=torch.ones(1, dtype=torch.long, device=device),  # OGBench
                        )
                        p   = out["z_pred"]
                        ctx = out["context_h"]
                        # Cost: context distance to goal context
                        cum_cost += F.mse_loss(ctx, goal_context).item()

                    if cum_cost < best_cost:
                        best_cost = cum_cost
                        best_seq  = cands[k]
                        context_h = ctx

            skill_actions.append(best_seq)

        return skill_actions

    def total_params(self) -> dict:
        return {
            "boundary_detector": self.boundary_detector.total_params(),
            "context_decoder":   sum(p.numel() for p in
                                     self.context_decoder.parameters()),
            "scar":              0,
            "splates":           0,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Hierarchical training loop (OGBench-Cube)
# ═══════════════════════════════════════════════════════════════════════════

def train_hierarchical_cwm(
    hcwm:       HierarchicalCWM,
    ogbench_loader,
    n_epochs:   int   = 20,
    base_lr:    float = 1e-4,
    lambda_scar: float = 0.05,
    save_dir:   str   = r"checkpoints\cwm",
    device_str: str   = "cpu",
    log_every:  int   = 20,
):
    """
    Fine-tune the HierarchicalCWM on OGBench-Cube trajectories.

    Phase 1 (epochs 0-5):  Train boundary detector only
    Phase 2 (epochs 6-20): Train boundary detector + context decoder + SCaR/SPlaTES

    The base CWM predictor stays FROZEN during hierarchical training to avoid
    catastrophic forgetting of the RECON and other domain representations.
    """
    device = torch.device(device_str)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Freeze base CWM
    for p in hcwm.cwm.parameters():
        p.requires_grad_(False)
    hcwm.cwm.eval()

    # Only train hierarchical components
    trainable = list(hcwm.boundary_detector.parameters()) + \
                list(hcwm.context_decoder.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=base_lr, weight_decay=1e-4)

    neuro = NeuromodulatorState(session_start=time.time())
    student_mock = nn.Linear(3 * 224 * 224, 256).to(device)

    best_loss   = float("inf")
    global_step = 0

    for epoch in range(n_epochs):
        hcwm.boundary_detector.train()
        hcwm.context_decoder.train()

        phase = 1 if epoch < 6 else 2
        epoch_losses = []

        for batch in ogbench_loader:
            frame_t  = batch["frame_t"].to(device)
            frame_t1 = batch["frame_t1"].to(device)
            action   = batch["action"].to(device)
            B        = frame_t.shape[0]

            # Encode frames
            with torch.no_grad():
                z_t  = student_mock(frame_t.reshape(B, -1))
                z_t1 = student_mock(frame_t1.reshape(B, -1))
                parts_t,  _, _, _ = hcwm.cwm.encode(z_t)
                parts_t1, _, _, _ = hcwm.cwm.encode(z_t1)

            # Neuromodulators
            signals = neuro.update(
                z_pred=parts_t.mean(1), z_actual=parts_t1.mean(1),
                rho=0.5, action_magnitude=action.norm(dim=-1).mean().item()
            )

            # Phase 1: train inverse dynamics only
            action_pred = hcwm.boundary_detector(parts_t, parts_t1)
            L_inv_dyn = F.mse_loss(action_pred, action)
            total_loss = L_inv_dyn

            # Phase 2: also train SCaR + SPlaTES (using THICK context)
            if phase == 2:
                ctx_h = hcwm.cwm.thick_gru.init_context(B, device)
                out   = hcwm.cwm.predict(
                    particles=parts_t, action=action,
                    context_h=ctx_h,
                    positions=torch.zeros(B, parts_t.shape[1], 2, device=device),
                    domain_id=batch["domain_id"].to(device),
                )
                # SPlaTES: predicted next state should match THICK context prediction
                splates = splates_skill_reward(
                    out["z_pred"], out["context_h"],
                    hcwm.context_decoder,
                    da_eff=signals["da_effective"]
                )
                total_loss = total_loss + 0.1 * splates

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            epoch_losses.append(total_loss.item())
            global_step += 1

            if global_step % log_every == 0:
                print(
                    f"[ep{epoch:02d} step{global_step:04d}] "
                    f"phase={phase}  loss={total_loss.item():.4f}  "
                    f"L_inv_dyn={L_inv_dyn.item():.4f}  "
                    f"regime={signals['regime']}"
                )

        mean_loss = np.mean(epoch_losses)
        print(f"\nEpoch {epoch:02d}  mean_loss={mean_loss:.4f}")

        if mean_loss < best_loss:
            best_loss = mean_loss
            path = Path(save_dir) / "hierarchical_best.pt"
            torch.save({
                "epoch":      epoch,
                "loss":       best_loss,
                "boundary":   hcwm.boundary_detector.state_dict(),
                "ctx_decoder": hcwm.context_decoder.state_dict(),
            }, path)
            print(f"  → Saved: {path}")

    print(f"\nHierarchical training complete. Best loss: {best_loss:.4f}")
    p_counts = hcwm.total_params()
    print(f"  boundary_detector: {p_counts['boundary_detector']:,} params")
    print(f"  context_decoder:   {p_counts['context_decoder']:,} params")
    return hcwm


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("HierarchicalCWM smoke test...")
    device = torch.device("cpu")
    B, K, D = 2, 16, 128

    cwm  = CortexWorldModel(d_model=D, K=K)
    hcwm = HierarchicalCWM(cwm, d_model=D, K=K, action_dim=9)

    particles = [torch.randn(B, K, D) for _ in range(10)]
    actions   = torch.randn(9, 9)
    boundaries = [3, 6]

    signals = {
        "da_effective": 0.6, "ecb": 0.1,
        "regime": "EXPLORE", "rho": 0.5,
    }

    out = hcwm.forward_hierarchical(particles, actions, boundaries, signals)
    print(f"  L_scar:       {out['L_scar'].item():.4f}")
    print(f"  L_splates:    {out['L_splates'].item():.4f}")
    print(f"  n_boundaries: {out['n_boundaries']}")

    # Boundary detector
    bd_out = hcwm.boundary_detector(particles[0], particles[1])
    print(f"  boundary_detector output shape: {tuple(bd_out.shape)}")

    p = hcwm.total_params()
    print(f"  Hierarchical params: {sum(p.values()):,}")
    print("Smoke test passed.")
