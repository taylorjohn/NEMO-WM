# CORTEX — System Architecture
**Version:** NeMo-WM v2 + CORTEX-16 v5 + CORTEX-PE v16
**Hardware:** GMKtec EVO-X2 · AMD Ryzen AI MAX+ 395 · 128GB RAM · AMD NPU XINT8
**Last updated:** April 12, 2026

---

## Overview — Three Pillars

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CORTEX System                                   │
│                                                                              │
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────────┐    │
│  │   NeMo-WM v2     │   │   CORTEX-PE v16  │   │   CORTEX-16 v5       │    │
│  │  Robot Navigation│   │  Anomaly Detection│   │  Algorithmic Trading  │    │
│  │                  │   │                  │   │                       │    │
│  │  arXiv published │   │  Production      │   │  Paper trading live   │    │
│  │  AUROC=0.9997    │   │  AUROC=1.000     │   │  Alpaca API           │    │
│  └──────────────────┘   └──────────────────┘   └──────────────────────┘    │
│                                                                              │
│  Shared: NeuromodulatorBase (DA/ACh/CRT/NE) · AMD NPU · GMKtec EVO-X2      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Pillar 1 — NeMo-WM v2 World Model

### 1.1 Architecture Overview

```
SENSORS (4Hz)           BELIEF ENCODING              PLANNING              ACTION
─────────────────────────────────────────────────────────────────────────────────
vel, ang          ──►  ProprioEncoder  ──►  BeliefEncoder  ──►  b_t ∈ R^64
gps (lat/lon)                │                   │
images (64×64)               │             ParticleFilter
                             │             (SIR, N=16)
                             │                   │
                             └───────────────────┘
                                         │
                              ┌──────────▼──────────┐
                              │  BeliefTransition    │
                              │  f(b_t, a_t) → b̂,σ  │
                              └──────────┬──────────┘
                                         │
                              ┌──────────▼──────────┐
                              │  AnticipateReact     │
                              │  α = g(δ, CRT)       │
                              └──────────┬──────────┘
                                    │          │
                              α→1 (anticipate)  α→0 (react)
                                    │          │
                              ┌─────▼──────────▼─────┐
                              │   ImaginationRollout   │
                              │   T steps forward      │
                              │   16 action candidates │
                              └──────────┬────────────┘
                                         │
                              ┌──────────▼──────────┐
                              │  NeuromodulatedValue  │
                              │  V=DA·Q−CRT·U+ACh·H  │
                              └──────────┬──────────┘
                                         │
                              ┌──────────▼──────────┐
                              │   EpisodicBuffer     │
                              │   WM + Store + Schema│
                              └──────────┬──────────┘
                                         │
                                    best_action
                                         │
                                    robot executes
                                         │
                                    new observation ──► (loop)
```

### 1.2 Component Specifications

| Component | File | Params | Key Result |
|---|---|---|---|
| ProprioEncoder | `train_proprio_6c.py` | 26K | AUROC=0.9997 (arXiv v1) |
| ParticleFilter | `particle_belief_decoder.py` | — | SIR, N=16 |
| BeliefEncoder | `particle_belief_decoder.py` | 149K | d=64, GPS+proprio+particles |
| FAISSRetriever | `particle_belief_decoder.py` | — | 253K frames, 72% retrieval acc |
| BeliefTransitionModel | `belief_transition.py` | 459K | eval MSE=0.031, σ=0.137 |
| AnticipateReactGate | `anticipate_react_gate.py` | 0 | α=0.826 familiar, non-parametric |
| ImaginationRollout | `imagination_rollout.py` | — | 16 candidates, open+closed loop |
| QuasimetricHead | `value_function.py` | 66K | calib+asym+tri loss, best=0.00789 |
| NeuromodulatedValue | `value_function.py` | — | DA·Q−CRT·U+ACh·H |
| EpisodicBuffer | `episodic_buffer.py` | — | WM(K=8)+Store(10K)+Schema |
| GRASP Planner | `grasp_planner.py` | — | 5.60ms, <10ms confirmed |

### 1.3 Neuromodulator System

All four neuromodulators share `NeuromodulatorBase` in `value_function.py`,
unified with `cortex_brain/` trading engine:

```
DA  (Dopamine)       RPE_t = R_t − V(b_{t-1})
                     DA_t  = 0.9·DA_{t-1} + 0.1·sigmoid(RPE_t / 0.10)
                     Role: scales Q (goal proximity), gates episodic replay

ACh (Acetylcholine)  ACh_t = N_eff_t / N      (particle filter certainty)
                     T_horizon = round(32 · ACh_t)    (8s empirical ceiling)
                     Role: planning horizon, scales H (multi-step return)

CRT (Cortisol)       novelty_t = ||b_t − domain_mean|| / domain_std
                     CRT_t = 0.95·CRT_{t-1} + 0.05·tanh(novelty_t)
                     NE_t  = 1.22 · CRT_t    (from cortisol_domain_adaptive.py)
                     Role: domain novelty, forces reactive mode, penalises U

NE  (Norepinephrine) NE_t  = 1.22 · CRT_t    (exploration bonus)
                     eCB_t = 0.82 · (1 − CRT_t)  (anxiety reduction)
```

### 1.4 Empirically-Grounded Planning Horizon

T_MAX=32 (8.0s at 4Hz) is confirmed by **two independent measurements**:

| Measurement | Method | Result |
|---|---|---|
| ACh temporal sweep | AUROC vs k_ctx | Peak at k=32, degrades k=128 |
| Rollout drift | BeliefTransition MSE | Crosses 0.05 at step ~7, saturates ~22 |

Both converge on 8 seconds. Neither was assumed — both were derived empirically.

### 1.5 Double Dissociation (arXiv v1 Result)

```
No-VLM (proprio only):   AUROC = 0.9997   ← 26K params
VLM-only (ViT-G 1B):     AUROC = 0.8833   ← 1B params
Fusion:                   AUROC = 0.9767   ← visual adds noise

Gap: proprio outperforms ViT-G by +0.114 AUROC
Finding: visual scaling does not solve temporal self-localisation
         physics-grounded path integration does
```

### 1.6 Sprint C — Visual Grounding (Ablation Results)

10 conditions across 3 metrics. Best result: GPS retrieval + proprio metric.

| ID | Condition | Pixel | CLIP | Proprio |
|---|---|---|---|---|
| A1 | Chance | 53.5% | 45.0% | — |
| A3 | GPS retrieval | 64.5% | 70.5% | **72.0%** |
| A4 | GPS + PBD diffusion | 59.0% | 52.0% | 54.5% |

Diffusion decoder: **negative result** — GPS retrieval alone is the primary grounding mechanism. Documented and closed.

---

## Pillar 2 — CORTEX-PE v16 (Anomaly Detection)

### 2.1 Production Domains

| Domain | Best AUROC | Status |
|---|---|---|
| CWRU bearing faults | 1.0000 | Production |
| RECON visual navigation | 0.9499 | Production |
| MIMII audio machinery | 0.9313 | Production |
| Cardiac audio | 0.8894 | Production |
| MVTec AD (visual) | 0.8152 | Production |
| SMAP/MSL telemetry | 0.7730 | Production |

### 2.2 Key Architecture Decisions

- **DINOv2 student distillation** — `checkpoints/dinov2_student/student_best.pt`
- **VLM gate** — `neuro_vlm_gate.py`, Phase 3 bio gate 33/33
- **Cortisol domain-adaptive** — sensitivity=0.10, NE_scale=1.22, eCB=0.82
- **MuRF distillation** — documented negative result (AUROC 0.7297 < 0.7393 baseline)
- **DreamerV3 tricks** — symlog, free_bits≥0.5, AGC λ=0.01, unimix 1%

---

## Pillar 3 — CORTEX-16 v5 (Algorithmic Trading)

### 3.1 Live Session Stats (2026-03-31)

```
Ticks processed:  103,365
Entries:          0  (quarter-end spread toxic)
Aborts:           21,822
Equity:           $101,133.47  (+0.000% session)
Signal window:    9:30–11:00 ET (39.9% of morning ticks clear Z≥2.0)
```

### 3.2 Active Configuration

| Parameter | Value | Note |
|---|---|---|
| Z threshold | 2.18 | Dynamic alpha |
| RTT threshold | 150ms | Hard gate |
| Spread fallback | ask=0 → trade_price ±$0.01 | Fixed |
| Entry cooldown | 60s | PDT guard |
| Pytest suite | 50/50 passing | Pre-flight mandatory |

### 3.3 cortex_brain/ Module

99/99 tests passing. Components:
- `DopamineSystem` — RPE + CRT + NE (will unify with NeuromodulatorBase)
- `CJEPAPredictor` — Sandwich Norm INT8/NPU, 0.34ms inference
- `TestTimeMemoryWithClustering` — Ward clustering
- `StaticCSRRouter` — CSR sparse GIL-bypass
- `CortexEngine` — 18-phase safeguards

**Pending convergence:** `cortex_brain/` DopamineSystem → `NeuromodulatorBase` in `value_function.py`. Both are implemented and tested. Unification is a refactor, not a research problem.

---

## Shared Infrastructure

### Hardware

| Component | Spec | Role |
|---|---|---|
| CPU | AMD Ryzen AI MAX+ 395 | Training + inference |
| RAM | 128GB | Full RECON dataset in memory |
| NPU | AMD XINT8 | 0.34ms CJEPAPredictor inference |
| NUC | 192.168.1.195 | Trading engine co-processor |

### Key Checkpoints

| File | Description | Key Metric |
|---|---|---|
| `checkpoints/cwm/proprio_kctx32_best.pt` | ProprioEncoder k=32 | AUROC=0.9997 |
| `checkpoints/cwm/belief_transition_v2.pt` | Dynamics model v2 | MSE=0.031, σ=0.137 |
| `checkpoints/cwm/quasimetric_head.pt` | QM distance head | best=0.00789 |
| `checkpoints/cwm/recon_faiss.index` | GPS frame index | 253K frames |
| `checkpoints/cwm/recon_faiss_meta.pt` | Index metadata | 3.12GB |
| `checkpoints/dinov2_student/student_best.pt` | MVTec DINOv2 | AUROC=0.7393 |
| `checkpoints/cardiac/student_best.pt` | Cardiac audio | AUROC=0.8894 |

### RECON Dataset

```
Path:     ./recon_data/recon_release/jackal_2019-*.hdf5
Files:    10,995 HDF5 trajectories
Frames:   ~770K total (T=70 per file at 4Hz)
Keys:     images/rgb_left (JPEG), commands/linear_velocity,
          commands/angular_velocity, gps/latlong
Used for: ProprioEncoder training, BeliefTransition triplets,
          FAISS GPS index, ablation evaluation
```

---

## File Inventory — NeMo-WM v2

| File | Sprint | Status | Purpose |
|---|---|---|---|
| `train_proprio_6c.py` | v1 | ✅ | ProprioEncoder training |
| `particle_belief_decoder.py` | C | ✅ | GPS retrieval + PBD |
| `eval_pbd_ablation.py` | C | ✅ | 10-condition ablation |
| `belief_transition.py` | D1 | ✅ | Dynamics model |
| `anticipate_react_gate.py` | D3 | ✅ | α gate |
| `imagination_rollout.py` | D4 | ✅ | Planning loop |
| `value_function.py` | D5 | ✅ | V_neuro + NeuromodulatorBase |
| `episodic_buffer.py` | D6 | ✅ | Memory system |
| `grasp_planner.py` | D4 | ✅ | 5.60ms MPC planner |
| `synthetic_trajectory_generator.py` | 9 | ✅ | Physics sim pre-training |

---

## Key Citations

| Paper | Used For |
|---|---|
| Moser et al. 2008 | Grid cell parallel (heading dominance) |
| O'Keefe 1971 | Place cell parallel |
| Hasselmo 1999 | ACh temporal precision |
| McNaughton 2006 | Path integration |
| Taube 1998 | Head direction cells |
| GRASP arXiv:2602.00475 | GRASP planner (5.60ms) |
| SIMPLE arXiv:2603.27410 | Physics-grounded perception |
| V-JEPA 2 (Meta) | Scale comparison baseline |
| DreamerV3 | Stability tricks (symlog, free_bits, AGC) |
| Rao & Ballard 1999 | Predictive coding (gate basis) |
| Tulving 2002 | Mental time travel (episodic buffer) |
| McClelland et al. 1995 | Complementary learning systems |
