# CORTEX World Model (CWM) — Master Plan
# All pending ideas, architecture decisions, and SWM contributions in one place
# Last updated: 2026-03-31

---

## Status Overview

| Component | Status | Notes |
|---|---|---|
| StudentEncoder (56K) | ✅ Done | XINT8 NPU-ready, 0.34ms inference |
| ParticleEncoder | ✅ Designed | SpatialSoftmax, K particles |
| ParticleTransformer | ✅ Designed | d=128, 2L, 4H, action conditioning |
| CWMLoss | ✅ Designed | L_predict + L_gaussian + L_gps + L_contact + 7-signal neuro modulation |
| MoE-JEPA predictor | ✅ Designed | Sparse top-2 of 4 physics experts, neuro-temperature routing |
| train_cwm.py (v2) | 🔄 Training | Sprint 1 running — epoch 2, loss ~160, 30 epochs |
| gps_grounding_loss_fast | ✅ Done | Vectorised GPS anchor loss |
| RECON TemporalHead | ⏳ Waiting | Sprint 2 paused — needs CWM epoch 5 checkpoint |
| OGBench-Cube gap | ❌ Open | action_dim=9 designed, not validated |
| Contact physics (PushT) | ❌ New | Signed distance head + contact attention |
| Structured action module | ❌ New | Per-particle action relevance gate (LPWM) |
| Neuromodulator reward | ✅ Designed | 7-signal system — cwm_neuro_reward.py ready |
| VL Sink (SWM) | ❌ New | Sprint 4 — inference-only |
| Cross-temporal pairing | ❌ New | Sprint 2 — highest value, do first |
| GeoLatentDatabase | ❌ New | Sprint 4 — build-time prerequisite |

---

## Architecture — Locked Decisions

### Parameter Budget
| Component | Params | Runs on |
|---|---|---|
| StudentEncoder | 56,592 | AMD NPU XINT8 |
| Encoder MoE (MOERouterV2) | ~33,000 | CPU — spectral ribbon |
| ParticleEncoder | ~260,000 | CPU |
| MoE-JEPA Predictor (2L, 4E, k=2) | ~1,214,000 | CPU |
| ContactHead | +8,000 | CPU |
| THICK Context GRU | +25,000 | CPU |
| **Total CWM** | **~1.6M** | **GMKtec EVO-X2 NUC, no GPU** |
| LeWorldModel (baseline) | 15,000,000 | A100 |
| **Ratio** | **~9× smaller** | |

> **Note on MoE inference cost:** 4 experts exist but only 2 fire per
> particle per step (top-2 sparse routing). Effective FLOP cost is
> ~50% of a fully-dense 4-expert predictor. Capacity gain is 4×.
> MoE overhead vs original dense predictor: ~2.7× params, ~1.3× FLOPs.

### Action Conditioning — Locked
Action conditioning from day one. All domains use the same weights.
Observation-only domains pass `action = zeros`.

| Domain | action_dim | Source |
|---|---|---|
| RECON | 2 | Jackal cmd_vel in HDF5 |
| TwoRoom | 2 | OGBench format |
| PushT | 2 | Pusher velocity |
| OGBench-Cube | 9 | Robot arm joint torques |
| MVTec / SMAP / Cardiac / Bearing | 0 → zeros(9) | Passive sensing |

Max action_dim = 9. Pad smaller dims with zeros. Clean abstraction, no branching.

### ParticleTransformer — Locked
```python
class ParticleTransformer(nn.Module):
    """
    d_model=128, n_layers=2, n_heads=4, action_dim=9
    Action broadcast: action projected to 128-D, added to every particle token.
    Observation-only domains: action=torch.zeros(B, 9)
    """
    def __init__(self, d_model=128, n_layers=2, n_heads=4, action_dim=9):
        super().__init__()
        self.action_proj = nn.Linear(action_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model*4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, particles, action):
        """
        particles: (B, K, d_model)
        action:    (B, action_dim) — zeros for observation-only domains
        """
        a = self.action_proj(action).unsqueeze(1)  # (B, 1, d_model)
        x = particles + a                          # broadcast to all K particles
        x = self.transformer(x)
        return self.out_proj(x)
```

### CWMLoss — Locked
```python
class CWMLoss(nn.Module):
    """
    Two losses, matching LeWorldModel's design:
      L_predict: MSE between predicted and actual next particle states
      L_gaussian: SIGReg-style collapse prevention
      L_gps:      GPS grounding (CORTEX differentiator vs LeWM)
    """
    def __init__(self, lambda_gaussian=0.1, lambda_gps=0.05):
        super().__init__()
        self.lambda_gaussian = lambda_gaussian
        self.lambda_gps = lambda_gps

    def forward(self, z_pred, z_target, gps_pred=None, gps_target=None):
        L_predict = F.mse_loss(z_pred, z_target)
        L_gaussian = sigreg_loss(z_pred, z_target)
        loss = L_predict + self.lambda_gaussian * L_gaussian

        if gps_pred is not None and gps_target is not None:
            L_gps = gps_grounding_loss_fast(z_pred, gps_pred, gps_target)
            loss = loss + self.lambda_gps * L_gps

        return loss, {
            'L_predict': L_predict.item(),
            'L_gaussian': L_gaussian.item(),
            'L_gps': L_gps.item() if gps_pred is not None else 0.0,
        }
```

---

## Benchmark Targets and Honest Estimates

| Benchmark | LeWM | CWM Target | Status | Key mechanism |
|---|---|---|---|---|
| Two-Room | ❌ underperforms | > LeWM | 🔄 Sprint 3 | Particles handle bimodal |
| PushT | +18% vs PLDM | Match LeWM | 🔄 Sprint 3 | Contact head + ACh weight |
| OGBench-Cube | Strong | Competitive | 🔄 Sprint 3 | action_dim=9 + THICK GRU |
| RECON | ATE/RPE | AUROC > 0.70 | 🔄 Sprint 2 | Cross-temporal pairing |
| SMAP/MSL | — | AUROC > 0.80 | ✅ **0.8427** | eval_smap_combined.py |
| Cardiac | — | AUROC > 0.85 | ✅ **0.8894** | k=32, 400 clips |
| MVTec | — | AUROC > 0.80 | ✅ **0.8855** ensemble | DINOv2+student 512-D k-NN |

**CWM core differentiators vs LeWM:**
1. Multi-domain shared backbone (LeWM is single-domain)
2. Direct GPS grounding loss (LeWM is probe-based)
3. ~9× fewer parameters (1.6M vs 15M with MoE predictor)
4. 7-signal biological neuromodulator reward (unique)
5. AMD NPU XINT8 deployment (LeWM requires GPU)

---

## Sprint Plan

### Sprint 1 — train_cwm_v2.py 🔄 RUNNING — epoch 2, loss ~160

```powershell
conda activate ryzen-ai-1.7.0

# Smoke test first (no data needed, verifies all shapes)
python train_cwm.py --smoke

# Full training on RECON
python train_cwm.py --hdf5-dir recon_data\recon_release --epochs 30 --batch-size 16

# Reduced run for initial validation (5 files, 5 epochs)
python train_cwm.py --hdf5-dir recon_data\recon_release --max-files 5 --epochs 5
```

**Success criterion:** Loss < 0.01, GPS loss converging, DA oscillating (not
stuck at 0.5), regime cycling, expert specialisation emerging by epoch 10
(RECON particles → Expert 1 dominant).

### Two-Level MoE Architecture

```
Level 1 — Encoder MoE (MOERouterV2, existing)
  Input:  256-D StudentEncoder CLS token
  Experts: 4 dense linear blocks (Shape/Size/Depth/Velocity)
  Gate:   soft-gate, all experts fire, outputs concatenated
  Output: 128-D spectral ribbon
  Role:   spectral feature decomposition of the visual latent

Level 2 — Predictor MoE (SparseMoEFFN, new)
  Input:  (B, K=16, 128) particle embeddings
  Experts: 4 sparse FFN blocks (Contact/Navigation/Drift/Locomotion)
  Gate:   top-2 routing, only 2 of 4 fire per particle per step
  Output: (B, K, 128) transformed particles
  Role:   domain-specific physics modelling

Key insight: attention is shared (all particles interact) — only FFN is
specialised. Each particle routes independently → contact particles hit
Expert 0, navigation particles hit Expert 1, etc.
```

### Neuromodulator → Routing Temperature

| Regime   | DA   | 5HT  | Router τ | Behaviour                              |
|----------|------|------|----------|----------------------------------------|
| EXPLOIT  | low  | high | 0.3      | Sharp → one expert per particle        |
| EXPLORE  | high | high | 1.5      | Soft → spread probability              |
| WAIT     | high | low  | 3.0      | Near-uniform → don't commit            |
| REOBSERVE| low  | low  | 0.8      | Mild preference for known expert       |

### Expert Specialisation Diagnostic

Log every 5 epochs. Target after 30 epochs:

```
Domain        Expert0  Expert1  Expert2  Expert3
recon          12.3%   71.4%   10.1%    6.2%   ← Navigation dominant
ogbench        68.7%   14.2%    9.8%    7.3%   ← Contact dominant
smap           11.1%    8.9%   72.3%    7.7%   ← Drift dominant
```

If all domains route to same expert → increase `lambda_aux` (load balance weight).

```python
# train_cwm.py skeleton
from neuromodulator import NeuromodulatorState
from cwm_neuro_reward import NeuromodulatedCWMLoss, RegimeGatedTrainer

neuro   = NeuromodulatorState(session_start=time.time())
loss_fn = NeuromodulatedCWMLoss(
    lambda_gaussian=0.1, lambda_gps=0.05,
    lambda_contact=0.01, lambda_skill=0.05, lambda_curv=0.02
)

for batch in recon_loader:
    z_t1_pred, signed_dist = dynamics(particles_t, action_padded)
    z_t1_true              = particle_enc(frames_t1)

    # Update all 7 neuromodulators from this latent transition
    signals = neuro.update(
        z_pred           = z_t1_pred.mean(dim=1),
        z_actual         = z_t1_true.mean(dim=1),
        rho              = batch['rho'].mean().item(),
        action_magnitude = action_padded.norm(dim=-1).mean().item()
    )

    # Regime-gated training config (lr, gradient clip, domain diversity)
    config = RegimeGatedTrainer.get_training_config(signals)
    loss_fn.update_from_neuro(signals)

    loss, stats = loss_fn(
        z_pred=z_t1_pred, z_target=z_t1_true,
        signed_dist=signed_dist,
        particle_positions=particle_enc.get_positions(particles_t),
        gps_pred=gps_pred, gps_target=gps_target,
    )

    for pg in optimizer.param_groups:
        pg['lr'] = base_lr * config['lr_multiplier']

    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, config['gradient_clip'])
    optimizer.step()
```

**Success criterion:** Training loss < 0.01 on RECON, GPS loss converging,
neuromodulator log showing DA oscillating (not stuck at 0.5), regime cycling.

### Sprint 2 — RECON TemporalHead + Cross-Temporal Pairing (SWM Phase 1)
**Highest value, lowest risk. Do immediately after Sprint 1.**

Files to create:
- `cortex_geo_db.py` — GeoLatentDatabase
- `train_recon_cross_temporal.py` — cross-temporal TemporalHead training

**Success criterion:** RECON AUROC > 0.70

### Sprint 3 — Multi-Domain Training with Contact Encoding
**Extend train_cwm.py to all domains.**

Contact encoding applies to PushT and OGBench-Cube only. For observation-only
domains (MVTec, SMAP, Cardiac, Bearing), the contact head is inactive
(no particle pairs are in contact by definition — anomaly detection doesn't
involve physical interaction). The structured action module is most important
for PushT (pusher vs T-block) and OGBench-Cube (arm end-effector vs cubes).

| Domain | action_dim | Contact relevant | Notes |
|---|---|---|---|
| RECON | 2 | No | Navigation, no manipulation |
| TwoRoom | 2 | No | Doorway navigation |
| PushT | 2 | **Yes** | Pusher-block contact critical |
| OGBench-Cube | 9 | **Yes** | End-effector-cube contact critical |
| MVTec/SMAP/Cardiac/Bearing | 0 | No | Passive sensing |

**Success criterion:** All domains training without NaN, contact head producing
negative signed distances for PushT contact events.

### Sprint 4 — VL Sink + GeoLatentDatabase (SWM Phases 2 & 3)
**After multi-domain training is stable.**

Build `GeoLatentDatabase` from RECON traversals. Add VL Sink to CWM inference.
Expected gain: +0.03–0.05 RECON AUROC, improved long-horizon consistency.

### Sprint 5 — Full Eval vs LeWM
**Final sprint.**

Run full evaluation across all domains. Compare to LeWorldModel (arXiv:2603.19312).
Expected result: CWM > LeWM on Two-Room, matches LeWM on PushT (contact encoding),
competitive on OGBench-Cube (structured action + contact), RECON AUROC > 0.70.

---

## Files to Create

```
# Sprint 1 — Core CWM (COMPLETE)
train_cwm.py                    # ✅ Full training loop — run this
cwm_moe_jepa.py                 # ✅ MoE-JEPA predictor + contact head + THICK GRU
cwm_neuro_reward.py             # ✅ NeuromodulatedCWMLoss (7 signals)
neuromodulator.py               # ✅ SOURCE — 7-signal system
particle_encoder.py             # ParticleEncoder (SpatialSoftmax)
particle_transformer.py         # ContactAwareParticleTransformer
contact_head.py                 # SignedDistanceHead + ContactWeightedAttention
structured_action.py            # StructuredActionModule (LPWM-style)

# Sprint 2 — RECON + Cross-Temporal
cortex_geo_db.py                # GeoLatentDatabase (SWM-derived)
train_recon_cross_temporal.py   # Cross-temporal TemporalHead training
eval_recon_auroc.py             # RECON AUROC evaluation

# Sprint 3 — Multi-Domain
domain_loaders.py               # Per-domain data loaders with rho from HDF5
eval_cwm_all_domains.py         # Full domain evaluation

# Sprint 4 — VL Sink
eval_recon_vl_sink.py           # RECON eval with VL Sink

# Future — Hierarchical + Legged
cwm_hierarchical.py             # THICK context GRU (Sprint 3, OGBench-Cube)
cwm_bio_planner.py              # SevenSignalMPCPlanner
fog_surrogate.py                # FoG local surrogate (RL only)
```

---

## Neuromodulator Reward System

### Source file
`cwm_neuro_reward.py` — complete, ready to import. Built from `neuromodulator.py`
(the CORTEX-PE v16.11 six/seven-signal system).

### The seven signals and their CWM roles

| Signal | Computed from | Range | Loss component modulated |
|---|---|---|---|
| **DA** (dopamine) | `(1 - cos_sim(z_pred, z_actual)) / 2` | [0,1] | `L_predict` — amplify gradient when model was wrong |
| **5HT** (serotonin) | `exp(-10 × std(z_history))` | [0,1] | `L_gaussian` (SIGReg) — stronger collapse prevention when stable |
| **NE/rho** (norepinephrine) | Allen Neuropixels spike rate | [0,1] | `L_gps` + planning temperature: `g_t = -L / (1 + rho)` |
| **ACh** (acetylcholine) | `(DA + (1-5HT)) / 2` (EMA) | [0,1] | `L_contact` — sharpen when surprising AND unstable |
| **E/I** (excit/inhib) | `DA / (1 - 5HT + ε)` | [0.5,2] | MeZO action_std, hexapod gait frequency |
| **Ado** (adenosine) | `elapsed / saturation_time` | [0,1] | Curriculum pacing — easy examples late session |
| **eCB** (endocannabinoid) | `DA × action_magnitude` (EMA) | [0,1] | Damps skill reward: `(1 - eCB × 0.4)` — breaks grasp loops |

### The four regimes (DA × 5HT matrix)

```
             5HT low          5HT high
DA high  │  WAIT             EXPLORE
         │  novel+unstable   novel+stable
         │  pause, low lr    high lr, diverse
─────────┼──────────────────────────────────
DA low   │  REOBSERVE        EXPLOIT
         │  known+unstable   known+stable
         │  stabilise        fine-tune, commit
```

`RegimeGatedTrainer.get_training_config(signals)` returns the appropriate
lr_multiplier, gradient_clip, domain_diversity, and n_candidates for each regime.

### Domain-specific signal mappings

| Domain | Primary signal | Effect |
|---|---|---|
| OGBench-Cube | DA + eCB | `ogbench_grasp_reward()` — eCB breaks stuck grasp loops |
| PushT | ACh | `pusht_contact_attention()` — amplify contact head at surprise events |
| PushT | E/I | `pusht_exploration_width()` — wide search when E/I high |
| RECON outdoor | NE/rho | `recon_gps_confidence()` — GPS weight scales with arousal |
| RECON outdoor | Ado | `recon_curriculum_difficulty()` — hard examples early, easy late |
| Hexapod | E/I | `hexapod_gait_frequency()` — high E/I → faster gait |
| Hexapod | 5HT | `hexapod_step_margin()` — low 5HT → larger safety margin |
| Hexapod | ACh | `hexapod_bilateral_symmetry_weight()` — high ACh allows asymmetry |
| Quadruped | DA + 5HT | `quadruped_terrain_confidence()` — regime gates stride speed |

### Novelty claim

CWM is the first world model to use biological neuromodulator signals derived
from mouse V1 Neuropixels recordings as adaptive reward modulators during
training. The cosine similarity `cos_sim(z_pred, z_actual)` that computes DA
in `neuromodulator.py` is exactly the prediction error signal that CWM's
dynamics module minimises. The training reward and the biological arousal
signal are computed from the same latent transition. The same seven signals
also drive the CORTEX-16 trading system — one biological substrate, two
domains.

### Offline training (from cortex_combat_log.csv)

When the trading system is not running, load the rho profile from the session log:

```python
# rho U-shape observed 2026-03-30:
#   09:30–10:15 → rho ~0.624 (high arousal, EXPLORE regime)
#   11:00–16:00 → rho ~0.481 (low arousal, EXPLOIT regime)
loss_fn.update_from_combat_log(da=0.500, rho=rho_by_hour, ado_frac=ado)
```

---

## Contact Physics — Full Technical Plan

### The Problem

Contact is discontinuous. A pusher hits the T-block: momentum transfers
instantaneously. Standard transformers assume smooth latent evolution. Without
contact-specific inductive bias, the ParticleTransformer must learn from data
alone that particle i only affects particle j when they are physically co-located.
This is why PushT was estimated as "uncertain" and OGBench-Cube as "below LeWM"
in the original benchmark table.

### Method 1 — Signed Distance Head (ContactNets-inspired)

Add a lightweight pairwise head that predicts signed distance between all particle
pairs. Positive = separated, zero = contact, negative = penetrating. This head is
supervised for free from SpatialSoftmax particle positions (already computed in
ParticleEncoder — no new labels needed). The signed distance then gates inter-
particle attention: particles close in position attend heavily; particles far
apart barely communicate.

**Implementation:** ~8K params (two linear layers). Adds ~2ms per forward pass.

```python
class ContactAwareParticleTransformer(nn.Module):
    def __init__(self, d_model=128, n_layers=2, n_heads=4, K=16, action_dim=9):
        super().__init__()
        self.action_proj = nn.Linear(action_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model*4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)

        # Signed distance head: pairwise (particle_i, particle_j) -> scalar
        self.contact_head = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, particles, action):
        """
        particles: (B, K, d_model)
        action:    (B, action_dim)
        Returns:   next_particles (B, K, d_model), signed_dist (B, K, K)
        """
        B, K, D = particles.shape
        a = self.action_proj(action).unsqueeze(1)
        x = self.transformer(particles + a)

        # Pairwise signed distances
        p_i = x.unsqueeze(2).expand(B, K, K, D)
        p_j = x.unsqueeze(1).expand(B, K, K, D)
        signed_dist = self.contact_head(
            torch.cat([p_i, p_j], dim=-1)
        ).squeeze(-1)                          # (B, K, K)

        # Contact-weighted inter-particle coupling
        # Sigmoid(-dist * 10): near 1 when in contact, near 0 when separated
        contact_w = torch.sigmoid(-signed_dist * 10)
        contact_modulated = (contact_w.unsqueeze(-1) * x.unsqueeze(1)).sum(2)

        z_next = x + self.out_proj(x + contact_modulated)
        return z_next, signed_dist


def contact_auxiliary_loss(signed_dist_pred, particle_positions, lambda_contact=0.01):
    """
    Supervised from SpatialSoftmax positions — no extra labels needed.
    particle_positions: (B, K, 2) — 2D positions from ParticleEncoder
    signed_dist_pred:   (B, K, K) — predicted signed distances

    Loss: MSE between predicted signed distance and true Euclidean distance.
    Euclidean distance is always positive (not signed) — model learns to map
    large distance → large positive, zero distance → zero.
    """
    B, K, _ = particle_positions.shape
    p_i = particle_positions.unsqueeze(2).expand(B, K, K, 2)
    p_j = particle_positions.unsqueeze(1).expand(B, K, K, 2)
    true_dist = torch.norm(p_i - p_j, dim=-1)   # (B, K, K), always >= 0
    return lambda_contact * F.mse_loss(signed_dist_pred, true_dist)
```

**Expected impact on PushT:** The contact head gives the model a binary signal
for when coupling activates. Without it, the transformer must infer contact from
subtle latent correlations. With it, contact is an explicit gate.

### Method 2 — Structured Action Module (LPWM-inspired)

Replace the broadcast `action_proj` with a per-particle relevance gate. In PushT,
only the pusher particle should receive the action signal directly. The T-block
responds through contact coupling, not through the action. Without this,
broadcasting the action to all particles causes the T-block to spuriously respond
to the pusher's velocity even when they're not in contact, polluting the dynamics.

```python
class StructuredActionModule(nn.Module):
    """
    LPWM-inspired per-particle action relevance gate.
    Learned: which particles are most directly affected by the action.
    In PushT: pusher particle gets high relevance, T-block gets near-zero.
    In OGBench-Cube: end-effector particle gets high relevance, cubes get low.
    """
    def __init__(self, d_model=128, action_dim=9):
        super().__init__()
        self.relevance = nn.Sequential(
            nn.Linear(d_model + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()                 # [0, 1] relevance per particle
        )
        self.action_proj = nn.Linear(action_dim, d_model)

    def forward(self, particles, action):
        """
        particles: (B, K, d_model)
        action:    (B, action_dim)
        """
        B, K, D = particles.shape
        a_exp = action.unsqueeze(1).expand(B, K, -1)   # (B, K, action_dim)
        relevance = self.relevance(
            torch.cat([particles, a_exp], dim=-1)
        )                                               # (B, K, 1)
        a_proj = self.action_proj(action).unsqueeze(1)  # (B, 1, d_model)
        return particles + relevance * a_proj


# Replace the original one-liner in ParticleTransformer:
# BEFORE: a = self.action_proj(action).unsqueeze(1)  # broadcast to all K
# AFTER:  x = self.structured_action(particles, action)  # per-particle gate
```

**Parameter cost:** +16K params over the original broadcast projection. Fits
within the 800K budget.

### Method 3 — FoG Surrogate (Policy Gradient Only)

For RL-based planning experiments on PushT and OGBench-Cube, contact events
cause zero or undefined gradients (the transition is non-smooth). The FoG
approach from NYU/LAAS (2026) resolves this by learning a local linear
surrogate that approximates the ParticleTransformer's Jacobian around each
trajectory point. Policy gradient flows through the surrogate, not the
transformer.

This is not needed for the AUROC evaluation sprints. Implement only if doing
model-based RL experiments after Sprint 5.

```python
class FoGSurrogate(nn.Module):
    """
    Lightweight local surrogate for policy gradient through contact dynamics.
    Trained to match the Jacobian of ContactAwareParticleTransformer locally.
    """
    def __init__(self, d_model=128, K=16, action_dim=9):
        super().__init__()
        input_dim = K * d_model + action_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, K * d_model)
        )

    def forward(self, particles_flat, action):
        return self.net(torch.cat([particles_flat, action], dim=-1))
```

### Summary: PushT Gap Closure Plan

| Change | Params | Expected gain | Risk |
|---|---|---|---|
| Signed distance head + contact attention | +8K | +10–15pp PushT success | Low |
| Structured action module | +16K | +5–8pp (stops spurious action coupling) | Low |
| FoG surrogate (RL only) | +2M (separate, not in CWM) | Enables policy gradient | Medium |
| **Total with contact encoding** | **~822K** | **Competitive with LeWM on PushT** | |

Note: 822K slightly exceeds the 800K budget. Options: reduce K from 16 to 14
(saves ~30K), or keep 822K and update the invariant to ≤850K. The contact
encoding's value on PushT justifies this.

### OGBench-Cube Contact Specifics

For the 9-DOF arm picking and placing cubes, contact is end-effector-to-cube.
The structured action module ensures only the end-effector particle (the one
spatially closest to the gripper in the frame) receives the high-relevance action
signal. The other particles (cubes) respond only when their signed distance to
the end-effector particle crosses zero. This models pick-and-place as:

1. End-effector particle moves toward cube particle (action-driven)
2. Signed distance drops to ~0 (contact)
3. Contact weight activates → cube particle dynamics couple to end-effector
4. Cube particle moves with end-effector (carrying phase)
5. Signed distance increases (release)
6. Contact weight drops → cube particle decouples, free dynamics resume

This is the correct inductive bias for all cube manipulation sequences in OGBench.

---

## SWM Contributions — How They Fit

Three new techniques from arXiv:2603.15583 (Seoul World Model):

### 1. Cross-Temporal Pairing → Sprint 2
Force RECON TemporalHead to learn geometry not dynamics.
Pair `(z_session_A_location_L, z_session_B_location_L)` instead of same-session.
**Risk: Low. No architecture change.**

### 2. GeoLatentDatabase → Sprint 4
GPS-indexed (GPS, latent) pairs from all RECON traversals.
Prerequisite for VL Sink and GPS RAG.
**Risk: Low. Build-time only.**

### 3. Virtual Lookahead Sink → Sprint 4
At each ParticleTransformer chunk boundary, retrieve lookahead GPS latent
and inject as anchor particle. Prevents long-horizon error accumulation.
**Risk: Low. Inference-only.**

### 4. GPS RAG Training Signal → Future (post-Sprint 5)
Add retrieval-augmented GPS loss term to CWMLoss.
**Risk: Medium. Requires CWM retraining.**

---

## Key Invariants (Do Not Change)

1. StudentEncoder stays frozen during CWM training — no gradient through NPU path
2. Action conditioning uses zeros for observation-only domains — same weights, no branching
3. GPS grounding loss is direct (not probe-based) — this is the core differentiator vs LeWM
4. Total CWM params ≤ 850K — contact encoding adds ~24K, still 17× vs LeWM
5. NPU XINT8 export path remains valid — no operations incompatible with AMD Quark
6. Contact head supervised from SpatialSoftmax positions — no extra labels needed

---

## What the SWM Plan Was Missing

The `CWM_SWM_INTEGRATION_PLAN.md` document covers only the three new SWM
contributions. It does not include:

- train_cwm.py (Sprint 1, blocks everything)
- Multi-domain action conditioning design (Sprints 1 & 3)
- OGBench-Cube gap closure (action_dim=9, Sprint 3)
- Full benchmark comparison plan vs LeWM (Sprint 5)
- Existing pending RECON TemporalHead work (Sprint 2)
- Parameter budget and architecture lock decisions
- Contact physics solution (PushT gap closure)

This master plan supersedes both the SWM integration plan and the contact physics
research. Both remain valid as technical references for their respective sections.

---

## References

- LeWorldModel: arXiv:2603.19312 (baseline to beat)
- SWM: arXiv:2603.15583 (VL Sink, cross-temporal pairing)
- LPWM: arXiv:2603.04553 (structured action module, ICLR 2026 Oral)
- ContactNets: pmlr-v155-pfrommer21a (signed distance, complementarity loss)
- FoG: NYU/LAAS hal-05375426v3 (decoupled gradient for contact RL, PushT)
- FOCUS: 10.3389/fnbot.2025.1585386 (object-centric exploration)
- THICK: ICLR 2024 (hierarchical world model, context GRU)
- SPlaTES: RLC 2025 (skill-predictable world models)
- HiLAM: arXiv:2603.05815 (hierarchical latent action, ICLR 2026 Workshop)
- SCaR: NeurIPS 2024 (bidirectional skill chain regularisation)
- Temporal Straightening: arXiv:2603.12231 (L_curv, straighter latent paths)
- MuRF: arXiv:2603.25744 (multi-scale distillation)
- AVO: arXiv:2603.24517 (lineage tracking)
- Science Robotics survey: eadt1497 (representation hierarchy for manipulation)
- Neuromodulator system: neuromodulator.py (CORTEX-PE v16.11, 7-signal)
- Allen Brain Observatory: Neuropixels mouse V1 NE/rho source

---

*Updated: 2026-03-30 EOD | CORTEX-PE v16.17 | ~922K params | 16× vs LeWM*
*Neuromodulator reward system complete — cwm_neuro_reward.py ready for Sprint 1*

---

## GRASP Planner — Sprint 4 Integration (arXiv:2602.00475)

**Paper:** Parallel Stochastic Gradient-Based Planning for World Models  
**Authors:** Psenka, Rabbat, Krishnapriyan, LeCun, Bar (Meta FAIR / UC Berkeley / NYU, Jan 2026)  
**Status:** Adopted for Sprint 4 — replaces / augments SevenSignalMPCPlanner

### What GRASP is

Standard MPC samples K action sequences and picks the best (zero-order, CEM-style).
GRASP instead optimises *lifted virtual states* directly using gradient descent through
the world model, with Langevin noise for exploration and a stop-gradient on state inputs
to keep gradients stable.

Three components:

1. **Lifted states** — intermediate latents s1..sT treated as independent optimisation
   variables, not rolled out serially. All T world model calls run in parallel.
   Dynamics constraint becomes a soft penalty rather than a hard serial dependency.

2. **Langevin noise** — Gaussian noise injected into virtual states during optimisation.
   Enables escape from local minima without abandoning gradient information.
   Maps naturally to E/I neuromodulator signal — high E/I = more noise = broader search.

3. **Grad-cut** — stop-gradient on state inputs to world model during planning rollout.
   Only action-input gradients flow. Prevents adversarial exploitation of brittle Jacobians
   in high-dimensional latent spaces. Already designed into JEPA training — apply same
   principle to planning pass.

### Results (from paper)

- +10% success rate vs CEM on long-horizon tasks (T=50+)
- Less than half the compute time vs CEM
- Outperforms vanilla gradient descent and CEM on PushT, D4RL, DMControl

### Integration plan for CORTEX CWM Sprint 4

| GRASP component | CORTEX mapping |
|----------------|----------------|
| Virtual states | Particle sets (B, K, D) — already the right representation |
| Langevin noise magnitude | E/I neuromodulator signal — biological Langevin |
| Grad-cut on state inputs | Same stop-gradient as JEPA target path |
| Dense one-step goal loss | GPS grounding loss per step — physical anchor |
| Alternating GD synchronisation | EXPLOIT regime → fine-tune with gradient, EXPLORE → Langevin |

### Implementation strategy

Keep SevenSignalMPCPlanner as the fast zero-order baseline.
Add GRASPPlanner as the gradient-based alternative.
Gate on regime:
- EXPLOIT → GRASP (precise, gradient-based, slower)
- EXPLORE → Mirror Ascent sampler (fast, broad, zero-order)

**Critical constraint:** GRASP requires backprop through the MoE predictor at planning
time. On GMKtec EVO-X2 CPU, benchmark latency before committing. Target: one GRASP
planning step < 10ms to maintain 4Hz RECON frame rate.

### Key citation

Psenka et al. (2026) arXiv:2602.00475. ICLR 2026 World Models Workshop.

