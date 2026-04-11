# CWM Research Integration Map
> Five Feb–Mar 2026 papers mapped to CWM roadmap, ablations, and implementation priority
> Updated: 2026-04-01 EOD

---

## Quick Reference

| Paper | Core idea | CWM relevance | Priority | Sprint |
|-------|-----------|---------------|----------|--------|
| LeWorldModel | Gaussian regularizer replaces 6 tricks | Simplifies loss, better latent geometry for GRASP | HIGH | S3/S4 |
| Causal-JEPA | Object masking = causal intervention | Particle masking ablation, ContactHead validation | HIGH | S2/S3 |
| Probing Latent World | AIM probe reveals physical structure | Run on epoch 10 NOW — no training needed | IMMEDIATE | Now |
| GeoWorld | Hyperbolic geometry for hierarchical structure | k=8/16 AUROC ceiling, post-paper research | LOW | Post-S5 |
| PiJEPA | Policy-warm-started MPPI | GRASP <10ms fix, Sprint 4 planner | MEDIUM | S4 |

---

## 1. Probing Latent World — Run Tonight (No Training Required)

**Paper:** arXiv:2603.20327 — AIM discrete quantization probe on frozen V-JEPA 2 latents  
**Finding:** Physical quantities (grasp angle, object geometry, motion temporal structure)
are significantly encoded in frozen JEPA latents. χ² p < 10⁻⁴, MI up to 0.117 bits.

### What to do with CWM epoch 10 checkpoint

The AIM probe converts frozen continuous latents into discrete symbol sequences
without modifying the encoder. Run it on CWM's particle embeddings to answer:

> Do CWM's particles encode linear velocity, angular velocity, terrain type,
> and temporal structure — the quantities the training objective was designed to produce?

### Implementation (runs in ~20 mins on NUC)

```python
# probe_cwm_latents.py
# Simplified AIM probe for CWM particle embeddings
import torch, numpy as np
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score

def aim_probe(particle_embeddings, labels, n_clusters=16):
    """
    AIM: quantize continuous latents → discrete symbols, then test
    whether symbol distribution is non-random w.r.t. physical labels.
    
    particle_embeddings: (N, K, 128) — N samples, K=16 particles, 128-D
    labels: dict of {label_name: (N,) array}  e.g. linear_vel, angular_vel,
            terrain_class, time_gap_k
    """
    # Flatten to (N*K, 128)
    flat = particle_embeddings.reshape(-1, particle_embeddings.shape[-1])
    
    # Quantize to discrete symbols via K-means
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    symbols = km.fit_predict(flat)             # (N*K,)
    symbols = symbols.reshape(particle_embeddings.shape[:2])  # (N, K)
    
    # Per-sample: most common symbol across particles (consensus symbol)
    consensus = np.array([np.bincount(row, minlength=n_clusters).argmax()
                          for row in symbols])  # (N,)
    
    results = {}
    for label_name, label_vals in labels.items():
        # Discretize continuous labels into bins
        if label_vals.dtype == float:
            bins = np.percentile(label_vals, np.linspace(0, 100, 9))
            label_discrete = np.digitize(label_vals, bins[1:-1])
        else:
            label_discrete = label_vals.astype(int)
        
        # Chi-squared test: are symbols non-random w.r.t. label?
        contingency = np.zeros((n_clusters, label_discrete.max()+1))
        for s, l in zip(consensus, label_discrete):
            contingency[s, l] += 1
        chi2, p, dof, _ = chi2_contingency(contingency + 1e-6)
        
        # Mutual information
        mi = mutual_info_score(consensus, label_discrete)
        
        results[label_name] = {'chi2': chi2, 'p': p, 'mi': mi, 'dof': dof}
        print(f"  {label_name:25s}  χ²={chi2:.1f}  p={p:.2e}  MI={mi:.4f} bits")
    
    return results

# Usage with CWM checkpoint:
# 1. Load CWM, run inference on RECON test set
# 2. Collect particle_embeddings (N, K, 128)
# 3. Collect metadata: linear_vel, angular_vel, terrain_label, time_gap
# 4. Run aim_probe(embeddings, labels)

# Expected results if CWM is working:
# linear_velocity       χ² >> 0, p < 0.001, MI > 0.05
# angular_velocity      χ² >> 0, p < 0.001, MI > 0.05
# time_gap_k            χ² >> 0, p < 0.001, MI > 0.05  ← proves temporal structure
# terrain_roughness     χ² >> 0, p < 0.001, MI > 0.03
# random_label          χ² ≈ 0,  p >> 0.05, MI ≈ 0     ← null control
```

### Why this matters for the paper

If particles encode physical quantities with p < 0.001, it proves CWM has learned
structured representations of the physical world — not just visual similarity.
This is a standalone paper section: *"Probing CWM's Latent World"* with a table
matching the Probing paper's format but on RECON outdoor navigation.

Run it now. Results are independent of floor break.

---

## 2. Gaussian Regularizer (LeWorldModel) — Add to Sprint 3

**Paper:** arXiv:2603.19312 — L_total = L_pred + λ·L_reg where L_reg ~ N(0,I)  
**Finding:** Single Gaussian constraint replaces EMA + momentum + 6 hyperparameters.
48× faster planning vs foundation model baselines.

### Current CWM state

CWM currently uses four DreamerV3 anti-collapse tricks:
- Symlog on 6 loss components
- Free_bits floor on L_jepa (min=0.5)
- AGC adaptive gradient clipping (λ=0.01)
- Unimix 1% on MoE gates

These are not equivalent to LeWorldModel's Gaussian regularizer — they solve
different failure modes. But L_reg as an *addition* would:

1. Constrain particle embeddings to a well-shaped isotropic Gaussian
2. Give GRASP gradient-based planning better-conditioned latent space
3. Reduce the need for AGC (well-shaped latent = more stable gradients)

### Implementation for Sprint 3

```python
def gaussian_regularizer(z, lam=0.1):
    """
    LeWorldModel L_reg: force latent embeddings toward N(0,I).
    z: (B, K, 128) — particle embeddings
    Forces: mean → 0, covariance → I
    """
    B, K, D = z.shape
    z_flat = z.reshape(B*K, D)
    
    # Mean should be zero
    mean_loss = z_flat.mean(0).pow(2).mean()
    
    # Variance should be 1 per dimension
    var_loss = (z_flat.var(0) - 1).pow(2).mean()
    
    # Off-diagonal covariance should be zero (isotropy)
    z_norm = z_flat - z_flat.mean(0)
    cov = (z_norm.T @ z_norm) / (B*K - 1)
    off_diag = cov - torch.diag(cov.diag())
    cov_loss = off_diag.pow(2).mean()
    
    return lam * (mean_loss + var_loss + cov_loss)

# Add to total_loss in train_cwm_multidomain.py:
# total_loss = jepa_loss + gaussian_regularizer(particles)
```

### Ablation: LeWorldModel L_reg vs current DreamerV3 tricks

| Config | L_reg | DreamerV3 | Expected AUROC k=16 | GRASP latency |
|--------|-------|-----------|---------------------|---------------|
| A: Current | No | Yes (all 4) | baseline | baseline |
| B: + L_reg | Yes | Yes (all 4) | +0.01–0.02 | -20% (better conditioned) |
| C: L_reg only | Yes | No | TBD | TBD |
| D: Neither | No | No | Collapse risk | N/A |

Config B is the immediate addition. Config C is a long-run ablation for the paper.

---

## 3. Causal-JEPA Particle Masking — Sprint 2/3 Ablation

**Paper:** arXiv:2602.11389 — Object-level masking = latent causal intervention  
**Finding:** Mask one object slot, force prediction from remaining slots.
+20% on CLEVRER counterfactual reasoning, 8× faster MPC.

### Connection to CWM

CWM's K=16 particles are structurally equivalent to Causal-JEPA's object slots.
Each particle is a hypothesis about world state. The ContactHead already implements
interaction-dependent reasoning (ACh-gated contact between particles).

Causal-JEPA's masking strategy applied to CWM particles = mask a subset of
particles at time T and force the predictor to reconstruct them at T+1 from the
unmasked particles plus the action. This is stronger than the current JEPA objective
because it requires inter-particle interaction modelling, not just self-prediction.

### Implementation

```python
def particle_mask_loss(predictor, particles_t, particles_t1, action,
                       mask_ratio=0.25, stop_grad=True):
    """
    Causal-JEPA particle masking for CWM.
    
    Mask mask_ratio of particles at T.
    Predict masked particles at T+1 from unmasked particles + action.
    Stop-gradient on targets (as in standard JEPA).
    
    particles_t:   (B, K, D) — particle set at time T
    particles_t1:  (B, K, D) — target particle set at T+1
    action:        (B, action_dim)
    """
    B, K, D = particles_t.shape
    n_mask = max(1, int(K * mask_ratio))  # e.g. 4 of 16 particles
    
    # Random mask indices
    mask_idx = torch.randperm(K)[:n_mask]
    unmask_idx = torch.tensor([i for i in range(K) if i not in mask_idx])
    
    # Only visible particles fed to predictor
    visible = particles_t[:, unmask_idx, :]   # (B, K-n_mask, D)
    
    # Predict all K particles at T+1 from visible + action
    pred_t1 = predictor(visible, action, target_K=K)   # (B, K, D)
    
    # Loss only on masked positions (where prediction is hardest)
    target = particles_t1[:, mask_idx, :].detach() if stop_grad \
             else particles_t1[:, mask_idx, :]
    
    return F.mse_loss(pred_t1[:, mask_idx, :], target)
```

### Ablation design

| Config | Masking | Expected effect |
|--------|---------|-----------------|
| A (current) | None — predict all K from all K | Baseline |
| B | 25% particle mask | +inter-particle interaction signal |
| C | 50% particle mask | Stronger causal signal, harder task |
| D | Random + fixed mask | Tests robustness |

The Causal-JEPA paper used 1% of latent features vs patch models' 100%.
CWM at K=16 particles is already at ~6% of a patch-based 256-token model.
Masking 4 of 16 particles gives a clean ablation comparable to their setup.

### ContactHead validation via Causal-JEPA

The formal proof in Causal-JEPA that object-level masking induces causal bias
directly validates CWM's ContactHead design. If ContactHead is computing
genuine inter-particle interaction (not just self-dynamics), then masking
individual particles should force the other particles to use the ContactHead
signal more — visible as increased ACh activation during masked prediction.

This is a measurable ablation: run particle masking with and without ContactHead,
measure ACh activation delta. If ContactHead is working, ACh increases during
masked prediction. This would be a novel mechanistic result.

---

## 4. PiJEPA Policy Warm-start — Sprint 4 GRASP Fix

**Paper:** arXiv:2603.25981 — Policy-warm-started MPPI planning  
**Finding:** Initialise MPPI sampling from a learned policy rather than uniform Gaussian.
Dramatically improves convergence speed.

### Current GRASP problem

GRASP at H=5, iters=3 is currently borderline at ~10ms on NUC.
Current initialisation: actions sampled from uniform or Gaussian distribution.
PiJEPA shows that warm-starting from a policy prior converges in fewer iterations.

### Implementation for Sprint 4

```python
class PolicyWarmStartGRASP:
    """
    GRASP planner with PiJEPA-style policy warm-start.
    Instead of random action initialisation, use a small policy network
    to propose initial action sequences, then refine with GRASP gradient steps.
    """
    def __init__(self, world_model, policy_net, H=5, n_iters=2):
        # n_iters=2 instead of 3 because warm-start gives better initialisation
        self.wm = world_model
        self.policy = policy_net       # Small MLP: (128,) → (H, action_dim)
        self.H = H
        self.n_iters = n_iters
    
    def plan(self, z_current, z_goal):
        # Warm start: policy proposes initial action sequence
        with torch.no_grad():
            a_init = self.policy(z_current)     # (H, action_dim) — policy prior
        
        # GRASP refines from warm start, not from random
        a = a_init.clone().requires_grad_(True)
        opt = torch.optim.Adam([a], lr=0.1)
        
        for _ in range(self.n_iters):  # 2 instead of 3 = 33% faster
            opt.zero_grad()
            z = z_current
            for t in range(self.H):
                z = self.wm.predict(z, a[t])
            cost = (z - z_goal).pow(2).mean()
            cost.backward()
            opt.step()
        
        return a.detach()[0]  # First action

# Policy net is tiny: Linear(128, 256) → ReLU → Linear(256, H*action_dim)
# ~70K params, trains on RECON demonstrations in <1 hour
# Reduces effective GRASP iters from 3 to 2 → cuts latency ~33%
# If current GRASP is 12ms, warm-start version should reach ~8ms < 10ms target
```

### DINOv2 vs V-JEPA-2 backbone comparison

PiJEPA provides the only published comparison of these two backbones on
navigation planning. Their finding: DINOv2 is competitive with V-JEPA-2 on
navigation tasks when paired with a good world model predictor. This directly
validates CWM's choice to use DINOv2/StudentEncoder as the vision backbone
rather than a larger video encoder.

Cite PiJEPA for this when writing the paper's backbone choice justification.

---

## 5. GeoWorld Hyperbolic Geometry — Post-Paper Research

**Paper:** arXiv:2602.23058 — Hyperbolic JEPA for hierarchical world structure  
**Finding:** 2–3% improvement on multi-step planning. Geometric RL for stability.

### Why this is deferred

The improvement is modest (2–3%) and the implementation complexity is high
(Riemannian manifold operations, Poincaré ball model, hyperbolic exponential maps).
Not worth introducing before Sprint 5 paper deadline.

### Why it's relevant post-paper

RECON's outdoor navigation has natural hierarchical structure that Euclidean
space cannot capture well:

```
Level 0: Individual frames (0.25s apart) — smooth, local motion
Level 1: Trajectory segments (2-4s) — terrain-specific patterns
Level 2: Environment regions — paths vs open areas vs buildings
Level 3: Full campus topology — route planning
```

In hyperbolic space, this hierarchy maps naturally to a tree structure where:
- Nearby frames cluster tightly near the center
- Trajectory segments form branches
- Environment-level structure emerges at the manifold boundary

This would directly improve k=8 and k=16 AUROC beyond what the floor break
can achieve, because the geometry of the latent space would better reflect the
geometry of the physical navigation problem.

**Implement after paper acceptance as CWM v2 architecture upgrade.**

---

## Consolidated Ablation Table

For the Sprint 5 paper, run these ablations in order of value:

| # | Ablation | What it tests | Scripts needed | Runtime |
|---|----------|---------------|----------------|---------|
| 1 | Tab 1 vs Tab 2 | Encoder contribution | Already running | Done |
| 2 | AIM probe (Probing paper) | Physical structure in latents | probe_cwm_latents.py | 20 min |
| 3 | + Gaussian regularizer | LeWorldModel trick vs DreamerV3 | patch train_cwm_multidomain.py | S3 run |
| 4 | Particle masking 0% vs 25% vs 50% | Causal-JEPA contribution | patch train_cwm_v2.py | S3 run |
| 5 | GRASP warm-start vs cold | PiJEPA contribution | sprint4_plan.py | S4 |
| 6 | ContactHead on vs off | Interaction reasoning | eval_cwm_ablations.py | S5 |
| 7 | VL Sink on vs off | Long-horizon error reduction | eval_recon_auroc.py | S5 |
| 8 | Multi-domain vs single-domain | Generalisation claim | eval_cwm_all_domains.py | S5 |
| 9 | unimix vs no unimix | Router collapse contribution | test_moe_router_collapse.py | Done |

Ablations 1 and 9 are already complete.
Ablation 2 is runnable tonight.
Ablations 3–5 are Sprint 3/4 additions from new papers.
Ablations 6–8 are standard Sprint 5 ablations.

---

## Implementation Priority (What to Build and When)

### Tonight / before floor break
- [ ] `probe_cwm_latents.py` — AIM probe on epoch 10 checkpoint
  - No training needed, ~20 min runtime
  - Direct paper section if χ² p < 0.001

### Sprint 3 additions (patch train_cwm_multidomain.py)
- [ ] Add `gaussian_regularizer(particles, lam=0.1)` to total_loss
  - From LeWorldModel — 10 lines of code
  - Improves GRASP latent conditioning
- [ ] Add particle masking loss (25% mask ratio)
  - From Causal-JEPA — 30 lines of code
  - Ablation 4 in the table above
- [ ] Monitor ACh activation delta during masked vs unmasked batches
  - ContactHead validation from Causal-JEPA theory

### Sprint 4 additions
- [ ] Train tiny policy net on RECON demonstrations (~70K params)
- [ ] Replace GRASP cold-start with PiJEPA warm-start
- [ ] Target: reduce effective iters from 3 → 2, latency 12ms → <10ms

### Post-paper
- [ ] GeoWorld hyperbolic embedding for CWM v2
- [ ] CLIP dual-distillation for language-conditioned goals
- [ ] Causal-JEPA full object-slot architecture (replace particle set)

---

## Paper Section Map

| Paper section | Evidence source | Status |
|---------------|-----------------|--------|
| Encoder choice justified | PiJEPA DINOv2 vs V-JEPA-2 comparison | Citable now |
| Leakage-free design | VLA-JEPA formal analysis | Validates CWM objective |
| World model beats encoder | Floor break + k-sweep | Pending epoch 12/13 |
| Physical structure in latents | AIM probe (Probing paper) | Run tonight |
| Anti-collapse analysis | LeWorldModel vs DreamerV3 tricks | Sprint 3 ablation |
| Causal particle interaction | Causal-JEPA formal proof | ContactHead ablation |
| Edge deployment | Fast-WAM 190ms vs CWM 0.34ms NPU | Citable now |
| Planning efficiency | PiJEPA warm-start | Sprint 4 |
| Multi-domain generalisation | Sprint 3 results | Sprint 3 |
| Comparison vs SOTA | stable-worldmodel framework | Sprint 5 |

*CWM Research Integration · 2026-04-01 · Based on five Feb–Mar 2026 papers*
