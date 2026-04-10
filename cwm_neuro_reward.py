"""
cwm_neuro_reward.py  —  CORTEX CWM v1.0
=========================================
Full seven-signal neuromodulator reward system for the CORTEX World Model.

Maps all signals from neuromodulator.py to CWM training rewards:

  DA   (dopamine)           Prediction error drive — scales L_predict
  5HT  (serotonin)          Latent stability — scales L_gaussian (collapse)
  NE   (norepinephrine/rho) Global arousal — sets planning temperature
  ACh  (acetylcholine)      Attention precision — weights domain-specific loss
  E/I  (excit/inhib)        Exploration width — scales action_std for MPC
  Ado  (adenosine)          Session fatigue — curriculum pacing
  eCB  (endocannabinoid)    Retrograde damping — prevents DA oscillation

Four regime states (DA × 5HT matrix):
  EXPLORE   high DA, high 5HT  — novel + stable, broad search
  WAIT      high DA, low 5HT   — novel + unstable, pause and observe
  REOBSERVE low DA, low 5HT   — known + unstable, re-examine
  EXPLOIT   low DA, high 5HT  — known + stable, commit and execute

Integration with CWM domains:
  OGBench-Cube   — DA drives skill reward, eCB prevents grasp oscillation
  PushT          — ACh weights contact precision, E/I sets exploration width
  RECON outdoor  — NE modulates GPS confidence, Ado paces curriculum
  Hexapod/Quad   — E/I sets gait frequency, 5HT gates stance stability
"""

import math
import time
from typing import Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Import the existing neuromodulator system ─────────────────────────────
# from neuromodulator import NeuromodulatorState, Regime, neuro_to_packet




# ── DreamerV3 tricks ──────────────────────────────────────────────────────

def symlog(x: torch.Tensor) -> torch.Tensor:
    """
    Symmetric log transform from DreamerV3.
    Compresses large magnitudes, identity near zero, preserves sign.
    symlog(x) = sign(x) * log(1 + |x|)
    Eliminates need for manual loss scale tuning across domains.
    """
    return torch.sign(x) * torch.log1p(torch.abs(x))


def free_bits(loss: torch.Tensor, min_val: float = 0.5) -> torch.Tensor:
    """
    Free bits from DreamerV3: clip loss below min_val so gradients
    only flow when the predictor is genuinely uncertain.
    Prevents encoder collapse by keeping L_jepa from trivialising to 0.
    """
    return torch.clamp(loss, min=min_val)

# ═══════════════════════════════════════════════════════════════════════════
# Signal reference card
# ═══════════════════════════════════════════════════════════════════════════
"""
Signal   Source                      Range    CWM role
──────   ──────────────────────────  ───────  ─────────────────────────────
DA       (1 - cos_sim(z_pred,z_act))/2  [0,1]    Reward prediction error scale
5HT      exp(-10 * std(z_history))   [0,1]    Collapse prevention strength
NE/rho   Allen Neuropixels spike rate [0,1]    Planning temperature divisor
ACh      (DA + (1-5HT)) / 2 (EMA)   [0,1]    Attention/precision weight
E/I      DA / (1 - 5HT + eps)       [0.5,2]  MeZO gaussian width / gait freq
Ado      elapsed_time / sat_time     [0,1]    Curriculum fatigue / pacing
eCB      DA * action_magnitude (EMA) [0,1]    Retrograde DA suppression
da_eff   DA * (1 - eCB * 0.4)       [0,1]    True DA after eCB damping
"""


# ═══════════════════════════════════════════════════════════════════════════
# NeuromodulatedCWMLoss
# ═══════════════════════════════════════════════════════════════════════════

class NeuromodulatedCWMLoss(nn.Module):
    """
    Full seven-signal CWM loss function.

    Each loss component is modulated by the biologically appropriate signal:

    L_predict   → DA_eff   Amplify gradient when world deviates from model
    L_gaussian  → 5HT      Strengthen collapse prevention when encoder is stable
    L_gps       → NE/rho   Weight GPS grounding with global arousal confidence
    L_contact   → ACh      Sharpen contact attention when surprising + unstable
    L_skill     → DA_eff × (1 - eCB)   Skill reward, eCB-damped to stop loops
    L_curv      → 5HT      Straightening loss stronger when representation stable
    L_fatigue   → Ado      Curriculum: weight easy examples late in session
    """

    def __init__(
        self,
        lambda_gaussian: float = 0.10,
        lambda_gps:      float = 0.05,
        lambda_contact:  float = 0.01,
        lambda_skill:    float = 0.05,
        lambda_curv:     float = 0.02,   # temporal straightening (from LPWM)
    ):
        super().__init__()
        self.lambda_gaussian = lambda_gaussian
        self.lambda_gps      = lambda_gps
        self.lambda_contact  = lambda_contact
        self.lambda_skill    = lambda_skill
        self.lambda_curv     = lambda_curv

        # Seven-signal state — update each training step
        self.da   = 0.50   # dopamine baseline
        self.sht  = 0.50   # serotonin baseline
        self.rho  = 0.50   # norepinephrine / fleet resonance
        self.ach  = 0.50   # acetylcholine baseline
        self.ei   = 1.00   # E/I ratio baseline
        self.ado  = 0.00   # adenosine (session age)
        self.ecb  = 0.00   # endocannabinoid (retrograde)
        self.da_eff = 0.50 # effective DA post-eCB suppression
        self.regime = "EXPLOIT"

    # ── Signal update ─────────────────────────────────────────────────────

    def update_from_neuro(self, signals: dict):
        """
        Accept the full signals dict returned by NeuromodulatorState.update().
        Call once per training batch before computing loss.

        signals = neuro.update(z_pred, z_actual, rho=rho, action_magnitude=a)
        loss_fn.update_from_neuro(signals)
        """
        self.da     = signals.get("da",           self.da)
        self.sht    = signals.get("sht",          self.sht)
        self.rho    = signals.get("rho",          self.rho)
        self.ach    = signals.get("ach",          self.ach)
        self.ei     = signals.get("ei",           self.ei)
        self.ado    = signals.get("ado",          self.ado)
        self.ecb    = signals.get("ecb",          self.ecb)
        self.da_eff = signals.get("da_effective", self.da)
        self.regime = signals.get("regime",       self.regime)

    def update_from_combat_log(self, da: float, rho: float,
                                ado_frac: float = 0.0):
        """
        Lightweight update from cortex_combat_log.csv values.
        Use for offline CWM training when full neuro pipeline not running.
        5HT, ACh, E/I, eCB estimated from DA and rho.
        """
        self.da   = da
        self.rho  = rho
        self.ado  = ado_frac
        # Estimate derived signals
        self.sht  = 1.0 - da          # when DA high, stability low
        self.ach  = (da + (1 - self.sht)) / 2.0
        self.ei   = max(0.5, min(2.0, da / (1.0 - self.sht + 0.1)))
        self.ecb  = 0.0               # conservative
        self.da_eff = da

    # ── Core loss components ──────────────────────────────────────────────

    def forward(
        self,
        z_pred:                torch.Tensor,    # (B, K, d) predicted particles
        z_target:              torch.Tensor,    # (B, K, d) actual particles
        signed_dist:           Optional[torch.Tensor] = None,  # (B,K,K)
        particle_positions:    Optional[torch.Tensor] = None,  # (B,K,2)
        gps_pred:              Optional[torch.Tensor] = None,  # (B,2)
        gps_target:            Optional[torch.Tensor] = None,  # (B,2)
        particles_skill_end:   Optional[torch.Tensor] = None,  # (B,K,d)
        particles_skill_start: Optional[torch.Tensor] = None,  # (B,K,d)
        z_trajectory:          Optional[torch.Tensor] = None,  # (B,T,K,d)
    ) -> Tuple[torch.Tensor, dict]:

        stats = {k: 0.0 for k in [
            "L_predict","L_gaussian","L_gps","L_contact",
            "L_skill","L_curv","L_fatigue",
            "da","sht","rho","ach","ei","ado","ecb","da_eff","regime"
        ]}
        stats.update({
            "da": self.da, "sht": self.sht, "rho": self.rho,
            "ach": self.ach, "ei": self.ei, "ado": self.ado,
            "ecb": self.ecb, "da_eff": self.da_eff, "regime": self.regime
        })

        # ── 1. Prediction loss — DA_eff modulated ─────────────────────────
        # DA encodes prediction error = (1 - cos_sim) / 2.
        # High DA: model was wrong → amplify gradient → learn faster.
        # Low DA: model was right → small gradient → don't overwrite.
        # eCB suppression prevents runaway amplification on repeated novelty.
        da_scale = 0.5 + self.da_eff   # [0.5, 1.5] — never zeros out
        L_predict = symlog(F.mse_loss(z_pred, z_target)) * da_scale
        stats["L_predict"] = L_predict.item()
        total = L_predict

        # ── 2. Collapse prevention — 5HT modulated ────────────────────────
        # 5HT = latent stability = exp(-10 * std(z_history)).
        # High 5HT: encoder is consistent → trust the representation space
        #           → stronger SIGReg to defend the geometry.
        # Low 5HT:  encoder is drifting → relax SIGReg → let it reorganise.
        sht_scale = 0.5 + self.sht   # [0.5, 1.5]
        z_pred_2d   = z_pred.mean(dim=1)   if z_pred.dim()   == 3 else z_pred
        z_target_2d = z_target.mean(dim=1) if z_target.dim() == 3 else z_target
        z_pred_2d   = torch.nan_to_num(z_pred_2d,   0.0).clamp(-10, 10)
        z_target_2d = torch.nan_to_num(z_target_2d, 0.0).clamp(-10, 10)
        try:
            L_gaussian = symlog(_sigreg_loss(z_pred_2d, z_target_2d))
            if not torch.isfinite(L_gaussian): L_gaussian = torch.tensor(0.0)
        except Exception:
            L_gaussian = torch.tensor(0.0)
        L_gaussian = L_gaussian * self.lambda_gaussian * sht_scale
        stats["L_gaussian"] = L_gaussian.item()
        total = total + L_gaussian

        # ── 3. GPS grounding — NE/rho modulated ──────────────────────────
        # NE/rho = Allen Neuropixels arousal.
        # High NE: strong neural signal → high-confidence spatial grounding.
        #          Weight GPS loss more heavily — trust the physical anchor.
        # Low NE:  quiet session → relax GPS pull → don't overcorrect.
        # This reproduces the CORTEX-16 formula: g_t = -L_cost / (1 + rho).
        if gps_pred is not None and gps_target is not None:
            rho_scale = 0.3 + 0.7 * self.rho   # [0.3, 1.0]
            L_gps = symlog(_gps_loss(z_pred, gps_pred, gps_target))
            L_gps = L_gps * self.lambda_gps * rho_scale
            stats["L_gps"] = L_gps.item()
            total = total + L_gps

        # ── 4. Contact auxiliary — ACh modulated ─────────────────────────
        # ACh = (DA + (1-5HT)) / 2 = high when surprising AND unstable.
        # "Pay attention to new input over stored model."
        # In CWM: pay attention to contact events (new physical transitions).
        # High ACh: something unexpected is happening at particle boundaries
        #           → sharpen contact head signal → amplify contact gradient.
        # Low ACh:  routine dynamics → normal contact penalty.
        if signed_dist is not None and particle_positions is not None:
            ach_scale = 0.5 + self.ach * 1.5   # [0.5, 2.0] — matches lr_scale formula
            L_contact = symlog(_contact_loss(signed_dist, particle_positions))
            L_contact = L_contact * self.lambda_contact * ach_scale
            stats["L_contact"] = L_contact.item()
            total = total + L_contact

        # ── 5. Skill transition — DA_eff × (1 - eCB) modulated ───────────
        # DA_eff: amplify skill reward proportional to prediction error.
        # eCB retrograde: if we've been trying the same skill repeatedly
        #                 in a high-DA state (stuck), suppress the reward
        #                 so the policy tries something different.
        # This is exactly what eCB does biologically: break loops.
        if particles_skill_end is not None and particles_skill_start is not None:
            ecb_damping  = 1.0 - self.ecb * 0.4   # eCB suppresses up to 40%
            skill_scale  = (0.5 + self.da_eff) * ecb_damping
            L_skill = symlog(_skill_transition_loss(particles_skill_end,
                                              particles_skill_start))
            L_skill = L_skill * self.lambda_skill * skill_scale
            stats["L_skill"] = L_skill.item()
            total = total + L_skill

        # ── 6. Temporal straightening — 5HT modulated ────────────────────
        # From Temporal Straightening (arXiv:2603.12231):
        # Straighter latent trajectories → better MPC convergence.
        # 5HT = stability → if encoder is stable, straighten more aggressively.
        # If encoder is reorganising (low 5HT), don't enforce straight lines.
        if z_trajectory is not None:
            sht_curv = 0.2 + self.sht * 0.8   # [0.2, 1.0]
            L_curv = symlog(_straightening_loss(z_trajectory))
            L_curv = L_curv * self.lambda_curv * sht_curv
            stats["L_curv"] = L_curv.item()
            total = total + L_curv

        # ── 7. Fatigue curriculum — Ado modulated ────────────────────────
        # Adenosine rises monotonically with session age.
        # Late in training: Ado high → shift training toward easy examples
        #                   (consolidation, not novelty).
        # Early in training: Ado low → normal sampling, all difficulties.
        # Implemented as a soft penalty on high-DA (hard) examples late in session.
        if self.ado > 0.3:  # only activates after 30% of session
            fatigue_penalty = self.ado * 0.1 * self.da   # small but nonzero
            L_fatigue = torch.tensor(fatigue_penalty, dtype=z_pred.dtype,
                                     device=z_pred.device)
            stats["L_fatigue"] = L_fatigue.item()
            total = total + L_fatigue

        return total, stats


# ═══════════════════════════════════════════════════════════════════════════
# Biological MPC with all seven signals
# ═══════════════════════════════════════════════════════════════════════════

class SevenSignalMPCPlanner:
    """
    CWM model-predictive controller driven by all seven neuromodulators.

    Extends ModulatedPlanner from neuromodulator.py to the CWM particle space.
    Directly uses the derived planner parameters already computed by
    NeuromodulatorState: n_candidates, action_std, action_scale, lr_scale.
    """

    def __init__(self, cwm, horizon: int = 8):
        self.cwm     = cwm
        self.horizon = horizon

    def plan(
        self,
        particles_0:    torch.Tensor,   # (1, K, d_model)
        goal_particles: torch.Tensor,   # (1, K, d_model)
        neuro_signals:  dict,           # from NeuromodulatorState.update()
        action_dim:     int = 9,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Plan using the neuromodulator-derived parameters directly.

        The NeuromodulatorState already computes:
          n_candidates : how many trajectories to sample
          action_std   : gaussian width for MeZO sampling (E/I modulated)
          action_scale : whether to act at all (regime-gated)
          eps_scale    : exploration epsilon (DA_eff + rho)

        We plug these straight into the Mirror Ascent planner.
        """
        # Extract planner params from neuro signals (already computed)
        n_cands     = neuro_signals.get("n_candidates", 32)
        action_std  = neuro_signals.get("action_std",   0.10)
        action_scl  = neuro_signals.get("action_scale", 1.00)
        eps_scale   = neuro_signals.get("eps_scale",    1.00)
        regime      = neuro_signals.get("regime",       "EXPLOIT")
        da_eff      = neuro_signals.get("da_effective", 0.50)
        rho         = neuro_signals.get("rho",          0.50)
        ado         = neuro_signals.get("ado",          0.00)

        # WAIT regime: don't act (neuro says something is wrong, observe first)
        if regime == "WAIT":
            return torch.zeros(action_dim), {"regime": "WAIT", "acted": False}

        # Sample n_cands candidate trajectories
        # action_std is E/I modulated: high E/I → broader exploration
        candidates = torch.randn(n_cands, self.horizon, action_dim) * action_std

        costs = []
        with torch.no_grad():
            for k in range(n_cands):
                particles = particles_0.clone()
                cum_cost  = 0.0
                for t in range(self.horizon):
                    action = candidates[k, t].unsqueeze(0)
                    particles, _ = self.cwm.particle_dynamics(particles, action)
                    step_cost = F.mse_loss(particles, goal_particles).item()
                    cum_cost += step_cost

                # Biologically-modulated cost (CORTEX-16 ARCHITECTURE.md formula):
                # g_t = -L_cost / (1.0 + rho)
                # High NE/rho: compress cost landscape → aggressive planning.
                modulated = -cum_cost / (1.0 + rho)
                costs.append(modulated)

        costs_t = torch.tensor(costs)

        # DA-modulated Mirror Ascent temperature:
        # High DA: unexpected outcomes → large policy update (exploit surprise).
        # Low DA:  expected outcomes → small update (don't disturb working policy).
        # eCB via da_eff: already suppressed, so oscillation loops are damped.
        # Adenosine: late-session conservatism → reduce eta.
        eta = 0.05 * (0.5 + da_eff) * (1.0 - ado * 0.3)
        kl  = 0.1

        q = torch.ones(n_cands) / n_cands
        w = torch.exp((costs_t - kl * torch.zeros(n_cands)) / max(eta, 1e-4))
        q = (q * w) / (q * w).sum()

        best_action = (candidates[:, 0] * q.unsqueeze(-1)).sum(0)
        best_action = best_action * action_scl  # regime-gated scale

        return best_action, {
            "regime":      regime,
            "n_candidates": n_cands,
            "action_std":  action_std,
            "da_eff":      da_eff,
            "eta":         eta,
            "acted":       True,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Domain-specific signal mappings
# ═══════════════════════════════════════════════════════════════════════════

class DomainNeuroMap:
    """
    Maps each neuromodulator to its most relevant CWM domain variable.
    All mappings derived from the biological function of each signal.
    """

    # ── OGBench-Cube (9-DOF arm, sequential skill composition) ────────────

    @staticmethod
    def ogbench_skill_phase(signals: dict) -> str:
        """
        Infer current skill phase from neuromodulator state.
        DA + 5HT + eCB together characterise where in a manipulation sequence
        the arm currently is.

        Returns: "approach" | "contact" | "carry" | "place" | "oscillating"
        """
        da_eff = signals.get("da_effective", 0.5)
        sht    = signals.get("sht",          0.5)
        ecb    = signals.get("ecb",          0.0)
        regime = signals.get("regime",       "EXPLOIT")

        if ecb > 0.5:          return "oscillating"  # stuck in a loop — eCB high
        if regime == "WAIT":   return "approach"     # observing, not committed
        if da_eff > 0.6 and sht < 0.4: return "contact"   # surprise + unstable
        if da_eff < 0.3 and sht > 0.6: return "carry"     # stable execution
        if regime == "EXPLOIT": return "place"             # committed, stable
        return "approach"

    @staticmethod
    def ogbench_grasp_reward(signals: dict,
                              contact_achieved: bool) -> float:
        """
        Grasp reward shaped by DA and eCB:
          - Contact achieved with high DA: large positive surprise → big reward.
          - Contact attempted many times (high eCB): dampen reward → try new approach.
        """
        da_eff = signals.get("da_effective", 0.5)
        ecb    = signals.get("ecb",          0.0)

        if not contact_achieved:
            return -0.1 * (1.0 + ecb)      # more negative when stuck looping

        base_reward = 1.0 + da_eff * 2.0   # [1.0, 3.0] proportional to surprise
        ecb_damping = 1.0 - ecb * 0.4      # eCB retrograde: max -40%
        return base_reward * ecb_damping

    # ── PushT (contact physics) ───────────────────────────────────────────

    @staticmethod
    def pusht_contact_attention(signals: dict) -> float:
        """
        ACh-modulated contact attention weight.
        High ACh = surprising + unstable = contact event is happening.
        Returns multiplier for contact head gradient.
        """
        return 0.5 + signals.get("ach", 0.5) * 1.5  # [0.5, 2.0]

    @staticmethod
    def pusht_exploration_width(signals: dict) -> float:
        """
        E/I modulated exploration width for PushT pusher.
        High E/I: broad search for the right push angle.
        Low E/I: fine-grained execution of known approach.
        """
        ei = signals.get("ei", 1.0)
        return max(0.05, min(0.20, 0.10 * ei))  # matches _action_std()

    # ── RECON outdoor navigation ──────────────────────────────────────────

    @staticmethod
    def recon_gps_confidence(signals: dict) -> float:
        """
        NE/rho modulated GPS grounding weight.
        High rho: strong Allen signal → confident spatial anchor.
        Low rho:  quiet session → relax GPS pull.
        Also decays with Ado (late-session conservatism).
        """
        rho = signals.get("rho", 0.5)
        ado = signals.get("ado", 0.0)
        return (0.3 + 0.7 * rho) * (1.0 - ado * 0.2)

    @staticmethod
    def recon_curriculum_difficulty(signals: dict) -> float:
        """
        Ado-modulated curriculum difficulty.
        Early session (Ado low): sample hard trajectories aggressively.
        Late session (Ado high): shift toward easier, stable trajectories.
        Returns preferred difficulty fraction [0=easy, 1=hard].
        """
        ado = signals.get("ado", 0.0)
        return max(0.2, 1.0 - ado * 0.8)   # [0.2, 1.0]

    # ── Hexapod locomotion (18-DOF) ───────────────────────────────────────

    @staticmethod
    def hexapod_gait_frequency(signals: dict,
                                base_freq: float = 1.0) -> float:
        """
        E/I modulated gait frequency.
        High E/I (exploration): faster gait, more aggressive locomotion.
        Low E/I (exploitation): slower, stable gait.
        Also NE/rho: high arousal → higher frequency.
        """
        ei  = signals.get("ei",  1.0)
        rho = signals.get("rho", 0.5)
        freq = base_freq * (0.5 + 0.5 * ei) * (0.7 + 0.3 * rho)
        return max(0.3, min(3.0, freq))

    @staticmethod
    def hexapod_step_margin(signals: dict,
                             base_margin: float = 0.05) -> float:
        """
        5HT-modulated step safety margin.
        High 5HT (stable): tight margin, confident footing.
        Low 5HT (unstable): large margin, cautious placement.
        Also eCB: if oscillating (loop), increase margin (back off).
        """
        sht = signals.get("sht", 0.5)
        ecb = signals.get("ecb", 0.0)
        margin = base_margin + 0.15 * (1.0 - sht) + 0.05 * ecb
        return max(0.02, min(0.30, margin))

    @staticmethod
    def hexapod_bilateral_symmetry_weight(signals: dict) -> float:
        """
        ACh-modulated symmetry enforcement.
        High ACh (attention/surprise): terrain is unusual → allow asymmetry.
        Low ACh (routine): enforce bilateral symmetry → stable tripod gait.
        """
        ach = signals.get("ach", 0.5)
        return max(0.0, 1.0 - ach * 1.5)   # [0.0, 1.0] → less symmetry when attention high

    # ── Quadruped locomotion (8–16-DOF) ──────────────────────────────────

    @staticmethod
    def quadruped_terrain_confidence(signals: dict) -> float:
        """
        DA + 5HT joint terrain confidence.
        EXPLOIT (low DA, high 5HT): known stable terrain → fast stride.
        REOBSERVE (low DA, low 5HT): drifting on unstable terrain → slow.
        WAIT (high DA, low 5HT): something unexpected → stop and scan.
        """
        regime = signals.get("regime", "EXPLOIT")
        ado    = signals.get("ado",    0.0)
        base = {"EXPLOIT": 1.0, "EXPLORE": 0.7, "REOBSERVE": 0.4, "WAIT": 0.0}
        return base.get(regime, 0.7) * (1.0 - ado * 0.15)


# ═══════════════════════════════════════════════════════════════════════════
# Regime-gated training strategy
# ═══════════════════════════════════════════════════════════════════════════

class RegimeGatedTrainer:
    """
    Adjusts CWM training strategy based on the four-regime state.

    EXPLORE   (high DA, high 5HT)  — novel + stable
      → Large learning rate (ACh lr_scale high), broad MeZO sampling.
      → Sample from diverse domains: push CWM into new territory.

    WAIT      (high DA, low 5HT)   — novel + unstable
      → Reduce lr: model is in flux, don't overcommit to current gradients.
      → Increase ACh: pay close attention to what just happened.
      → Re-sample the same batch: observe, don't update aggressively.

    REOBSERVE (low DA, low 5HT)   — known + unstable
      → Normal lr, focus on stability: strengthen 5HT (L_gaussian).
      → Sample harder variants of seen examples.

    EXPLOIT   (low DA, high 5HT)   — known + stable
      → Reduce lr (converged): do fine-tuning only.
      → Ado curriculum: shift toward consolidation examples.
      → High n_candidates in planner: commit to precise execution.
    """

    @staticmethod
    def get_training_config(signals: dict) -> dict:
        regime    = signals.get("regime",    "EXPLOIT")
        lr_scale  = signals.get("lr_scale",  1.0)   # already ACh-modulated
        n_cands   = signals.get("n_candidates", 32)
        action_std = signals.get("action_std", 0.1)
        ado       = signals.get("ado",       0.0)

        if regime == "EXPLORE":
            return {
                "lr_multiplier":     lr_scale,          # high (ACh-gated)
                "domain_diversity":  True,              # sample all domains
                "gradient_clip":     1.0,               # normal clip
                "n_candidates":      n_cands,           # high (up to 96)
                "action_std":        action_std,        # high E/I → wide
                "description": "EXPLORE: novel+stable, high lr, diverse domains"
            }
        elif regime == "WAIT":
            return {
                "lr_multiplier":     lr_scale * 0.3,   # damped: don't overcommit
                "domain_diversity":  False,             # repeat current domain
                "gradient_clip":     0.5,               # tighter clip
                "n_candidates":      max(16, n_cands // 2),
                "action_std":        action_std * 0.5, # cautious
                "description": "WAIT: novel+unstable, low lr, observe first"
            }
        elif regime == "REOBSERVE":
            return {
                "lr_multiplier":     lr_scale * 0.7,
                "domain_diversity":  False,
                "gradient_clip":     1.0,
                "n_candidates":      n_cands,
                "action_std":        action_std,
                "description": "REOBSERVE: known+unstable, normal lr, stabilise"
            }
        else:  # EXPLOIT
            return {
                "lr_multiplier":     lr_scale * (0.3 + 0.7 * (1 - ado)),
                "domain_diversity":  False,             # consolidate
                "gradient_clip":     0.3,               # fine-tuning clip
                "n_candidates":      n_cands,           # high precision
                "action_std":        action_std * 0.5, # precise execution
                "description": "EXPLOIT: known+stable, fine-tune, consolidate"
            }


# ═══════════════════════════════════════════════════════════════════════════
# Helper loss functions
# ═══════════════════════════════════════════════════════════════════════════

def _sigreg_loss(z1: torch.Tensor, z2: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """SIGReg (Epps-Pulley inspired) collapse prevention."""
    std_z1 = torch.sqrt(z1.var(dim=0) + eps)
    std_z2 = torch.sqrt(z2.var(dim=0) + eps)
    var_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))
    z1c = z1 - z1.mean(dim=0)
    z2c = z2 - z2.mean(dim=0)
    cov_z1 = (z1c.T @ z1c) / (z1.shape[0] - 1)
    cov_z2 = (z2c.T @ z2c) / (z2.shape[0] - 1)
    cov_loss = (
        cov_z1.fill_diagonal_(0).pow(2).sum() / z1.shape[1] +
        cov_z2.fill_diagonal_(0).pow(2).sum() / z2.shape[1]
    )
    return var_loss + cov_loss


def _gps_loss(z_pred: torch.Tensor,
              gps_pred: torch.Tensor,
              gps_target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(gps_pred, gps_target)


def _contact_loss(signed_dist: torch.Tensor,
                   particle_positions: torch.Tensor,
                   lambda_c: float = 1.0) -> torch.Tensor:
    B, K, _ = particle_positions.shape
    p_i = particle_positions.unsqueeze(2).expand(B, K, K, 2)
    p_j = particle_positions.unsqueeze(1).expand(B, K, K, 2)
    true_dist = torch.norm(p_i - p_j, dim=-1)
    return lambda_c * F.mse_loss(signed_dist, true_dist)


def _skill_transition_loss(end_a: torch.Tensor,
                            start_b: torch.Tensor,
                            lambda_s: float = 1.0) -> torch.Tensor:
    fwd = F.mse_loss(end_a, start_b.detach())
    bwd = F.mse_loss(start_b, end_a.detach())
    return lambda_s * (fwd + bwd)


def _straightening_loss(trajectory: torch.Tensor) -> torch.Tensor:
    """
    Temporal straightening from arXiv:2603.12231.
    trajectory: (B, T, K, d_model)
    Minimise curvature of latent paths → better MPC convergence.
    """
    B, T, K, D = trajectory.shape
    if T < 3:
        return torch.tensor(0.0, requires_grad=False)
    flat = trajectory.reshape(B, T, -1)   # (B, T, K*D)
    d1 = flat[:, 1:] - flat[:, :-1]       # velocities
    d2 = d1[:, 1:] - d1[:, :-1]           # accelerations (curvature)
    curvature = d2.norm(dim=-1).mean()
    return curvature


# ═══════════════════════════════════════════════════════════════════════════
# Integration guide
# ═══════════════════════════════════════════════════════════════════════════
"""
TRAIN LOOP USAGE
────────────────

from neuromodulator import NeuromodulatorState
from cwm_neuro_reward import NeuromodulatedCWMLoss, RegimeGatedTrainer

neuro    = NeuromodulatorState(session_start=time.time())
loss_fn  = NeuromodulatedCWMLoss()

for epoch in range(n_epochs):
    for batch in loader:
        z_t1_pred, signed_dist = dynamics(particles_t, action_padded)
        z_t1_true              = particle_enc(frames_t1)

        # Update neuromodulators from this transition
        signals = neuro.update(
            z_pred           = z_t1_pred.mean(dim=1),   # (B, d_model)
            z_actual         = z_t1_true.mean(dim=1),
            rho              = batch['rho'].mean().item(),
            action_magnitude = action_padded.norm(dim=-1).mean().item()
        )

        # Get training config for this regime
        config = RegimeGatedTrainer.get_training_config(signals)

        # Update loss function
        loss_fn.update_from_neuro(signals)

        # Compute modulated loss
        loss, stats = loss_fn(
            z_pred              = z_t1_pred,
            z_target            = z_t1_true,
            signed_dist         = signed_dist,
            particle_positions  = particle_enc.get_positions(particles_t),
            gps_pred            = gps_pred,
            gps_target          = gps_target,
        )

        # Apply regime-gated lr
        for pg in optimizer.param_groups:
            pg['lr'] = base_lr * config['lr_multiplier']

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, config['gradient_clip'])
        optimizer.step()

        # Log all signals
        log.info(
            "regime=%s DA=%.3f 5HT=%.3f NE=%.3f ACh=%.3f "
            "E/I=%.2f Ado=%.3f eCB=%.3f lr_x=%.2f",
            signals['regime'], signals['da'], signals['sht'],
            signals['rho'], signals['ach'], signals['ei'],
            signals['ado'], signals['ecb'], config['lr_multiplier']
        )


OFFLINE USAGE (from cortex_combat_log.csv)
──────────────────────────────────────────

profile = load_neuromodulator_profile('cortex_combat_log.csv', '2026-03-30')
for batch in loader:
    hour = batch['timestamp_et_hour']
    rho  = profile['rho_by_hour'].get(hour, 0.481)
    ado  = hour_to_ado_fraction(hour, session_start_hour=9.5)
    loss_fn.update_from_combat_log(da=0.500, rho=rho, ado_frac=ado)
    # DA stays at baseline 0.500 (not in combat log yet)
    # rho varies by hour from the U-shape profile
    # ado rises from 0 at 9:30 to ~0.6 at close
"""
