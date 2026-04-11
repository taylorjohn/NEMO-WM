# CORTEX System Architecture
## CORTEX-PE vs CORTEX World Model

```
CORTEX-PE v16.17  ·  GMKtec EVO-X2 NUC (GMKtec EVO-X2 (AMD Ryzen AI MAX+ 395, 128GB), 128GB) (192.168.1.195)
Python 3.12  ·  torch 2.10.0+cpu  ·  conda ryzen-ai-1.7.0
```

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CORTEX SYSTEM                                     │
│                                                                          │
│   ┌──────────────────┐   SHARED BACKBONE   ┌──────────────────────┐    │
│   │   CORTEX-PE      │◄───────────────────►│   CORTEX WM          │    │
│   │ Perception Engine│   StudentEncoder     │   World Model        │    │
│   │                  │   56K · XINT8 NPU   │                      │    │
│   │ Does X deviate   │                     │ What happens if      │    │
│   │ from normal?     │   7-Signal Neuro    │ I do action A?       │    │
│   │ (passive)        │◄───────────────────►│ (active)             │    │
│   └──────────────────┘   DA·5HT·NE·ACh    └──────────────────────┘    │
│                           E/I·Ado·eCB                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## CORTEX-PE — Perception Engine

### Purpose
Anomaly detection across heterogeneous sensor domains using a single shared backbone. No action, no dynamics — encode and score.

### Pipeline

```
INPUT DOMAINS (6)
├── Cardiac audio       PhysioNet 2016 WAV clips
├── SMAP/MSL telemetry  NASA spacecraft channels
├── MVTec AD            Industrial visual inspection
├── RECON outdoor       Jackal robot HDF5 4Hz frames
├── CWRU bearing        Vibration fault detection
└── OGBench-Cube        Robotic manipulation frames
        │
        ▼
DINOv2-small (22.1M)  ──MuRF distillation──►  StudentEncoder
                         scales=[0.75,1.0,1.5]  56K params
                                                 0.34ms/sample
                                                 AMD NPU XINT8
        │
        ▼  128-D class latent
PCA k-NN Anomaly Head
├── Fit PCA on normal class only (unsupervised)
├── k=32 nearest neighbour distance
└── AUROC on held-out set
        │
        ▼
ANOMALY SCORE → AUROC
```

### Results

| Domain        | AUROC  | Config            | Notes                       |
|---------------|--------|-------------------|-----------------------------|
| Cardiac audio | 0.8894 | k=32, 400 clips   | PhysioNet 2016 real data    |
| Cardiac audio | 0.7730 | k=32, 1000 clips  | Harder full-dataset eval    |
| SMAP/MSL      | 0.8427 | hybrid+semi done    | eval_smap_combined.py 71/81 |
| MVTec AD      | 0.7393 | k=32 standalone   | eval_mvtec.py               |
| MVTec ensemble| 0.8855 | DINOv2+student done  | eval_mvtec_ensemble.py 13/15|
| Efficiency    | 3.26×  | AUROC/MB          | vs TinyGLASS (11M params)   |

### Key Properties
- **Unsupervised** — normal class only, no anomaly labels needed
- **Zero-shot domains** — same weights, new domain in < 1 hour
- **NPU-deployed** — XINT8 via AMD Quark, opset 17 ONNX

---

## CORTEX WM — World Model

### Purpose
Action-conditioned latent dynamics prediction across manipulation, navigation, and locomotion domains. Predict next state, plan optimal action.

### Pipeline

```
INPUT DOMAINS (6+)
├── RECON outdoor       Jackal nav HDF5, action_dim=2
├── OGBench-Cube        9-DOF arm, action_dim=9
├── TwoRoom             2D nav, action_dim=2
├── PushT               Contact manipulation, action_dim=2
├── Hexapod (future)    18-DOF, CPG-encoded action_dim=4
└── Quadruped (future)  16-DOF, action_dim=12
        │
        ▼
StudentEncoder (FROZEN — no gradient through NPU path)
56K params · 128-D latent · XINT8 NPU
        │
        ▼
ParticleEncoder
296K params · SpatialSoftmax · K=16 keypoints
Discovers spatial structure without supervision
        │
        ▼  (B, K=16, d=128) particle set
ContactAwareParticleTransformer
446K params · d=128 · 2 layers · 4 heads
        ├── StructuredActionModule (LPWM-style per-particle relevance gate)
        │   Only pusher/end-effector particle receives action directly
        │   Object particles respond only through contact coupling
        ├── SignedDistanceHead (+8K params)
        │   Pairwise signed distance → contact-weighted attention
        │   Supervised free from SpatialSoftmax positions
        └── THICK Context GRU (+25K params)
            Slow-timescale skill-level dynamics
            Reduces OGBench-Cube horizon from 50→6 effective steps
        │
        ▼
NeuromodulatedCWMLoss  ←── 7-signal neuromodulator system
        ├── L_predict   × DA_eff     prediction error drive
        ├── L_gaussian  × 5HT        SIGReg collapse prevention
        ├── L_gps       × NE/rho     GPS physical grounding
        ├── L_contact   × ACh        contact attention sharpening
        ├── L_skill     × DA×(1-eCB) skill transition (eCB breaks loops)
        ├── L_curv      × 5HT        temporal straightening
        └── L_fatigue   × Ado        curriculum pacing
        │
        ▼
SevenSignalMPCPlanner
g_t = −L_cost / (1 + ρ)        Mirror Ascent, NE-modulated temperature
η   = 0.05 × (0.5 + DA_eff)    DA sets update rate
std = 0.1  × E/I                E/I sets action gaussian width
K   = 16–96 candidates          DA_eff × (1 − Ado×0.5) adapts count
        │
        ▼
NEXT PARTICLE STATE + BEST ACTION
```

### Parameter Budget

| Component             | Params  | Runs on  |
|-----------------------|---------|----------|
| StudentEncoder        | 56,592  | AMD NPU XINT8 |
| ParticleEncoder       | 295,808 | CPU      |
| ParticleDynamics (2L) | 446,464 | CPU      |
| ContactHead           | +8,000  | CPU      |
| THICK Context GRU     | +25,000 | CPU      |
| **Total CWM**         | **~831K** | **NUC, no GPU** |
| LeWorldModel          | 15,000,000 | A100  |
| **Advantage**         | **18× smaller** | |

### Benchmark Targets

| Benchmark    | LeWM         | CWM Target       | Key mechanism              |
|--------------|--------------|------------------|----------------------------|
| Two-Room     | ❌ fails      | > LeWM done        | Particles handle bimodal   |
| PushT        | +18% vs PLDM | Match LeWM       | Contact head + ACh weight  |
| OGBench-Cube | Strong       | Competitive      | action_dim=9 + THICK GRU  |
| RECON        | ATE/RPE      | AUROC > 0.70     | Cross-temporal pairing     |

---

## Shared Components

### StudentEncoder (shared backbone)
```
Role in CPE:  Feature extractor → PCA anomaly scoring
Role in CWM:  Frozen perception stage → particle dynamics input

Same 56K weights serve both systems.
In CWM: no gradient flows through this layer.
NPU handles inference asynchronously while CPU runs dynamics.
```

### Neuromodulator System (neuromodulator.py, CORTEX-PE v16.11)

All 7 signals derived from latent transitions. Same computation drives
the CORTEX-16 trading system and the CWM world model.

```
Signal  Source                        Range    CPE role              CWM role
──────  ────────────────────────────  ───────  ────────────────────  ─────────────────────────
DA      (1−cos_sim(z_pred,z_act))/2  [0,1]    Rho signal drive      L_predict scale
5HT     exp(−10 · std(z_history))   [0,1]    Encoder stability     L_gaussian (SIGReg) scale
NE/rho  Allen Neuropixels spike rate [0,1]    Z-score modulator     L_gps weight + plan temp
ACh     (DA + 1−5HT) / 2  (EMA)    [0,1]    Attention gate        L_contact sharpening
E/I     DA / (1 − 5HT + ε)         [0.5,2]  Exploration width     action_std, gait frequency
Ado     elapsed / saturation_time   [0,1]    Session fatigue       Curriculum pacing (easy→)
eCB     DA × ‖action‖  (EMA)        [0,1]    Loop suppression      Skill reward damping −40%
```

**DA_eff** = DA × (1 − eCB × 0.4) — effective dopamine after retrograde suppression.

#### Four Regime States (DA × 5HT matrix)

```
                 5HT LOW                5HT HIGH
DA HIGH  │  WAIT                   EXPLORE
         │  novel + unstable        novel + stable
         │  low lr, re-observe      high lr, diverse domains
─────────┼──────────────────────────────────────────
DA LOW   │  REOBSERVE              EXPLOIT
         │  known + unstable        known + stable
         │  stabilise SIGReg        fine-tune, commit, high K
```

`RegimeGatedTrainer.get_training_config(signals)` selects lr_multiplier,
gradient_clip, domain_diversity, and n_candidates per regime automatically.

---

## Architectural Comparison

| Dimension         | CORTEX-PE                          | CORTEX WM                                    |
|-------------------|------------------------------------|----------------------------------------------|
| **Goal**          | Anomaly detection (AUROC)          | Sequential control + world modelling         |
| **Task type**     | Passive — does X deviate?          | Active — what happens if I do A?             |
| **Output**        | Anomaly score per sample           | Next latent state + best action              |
| **Action input**  | None                               | action_dim=2→9 (zeros for passive domains)   |
| **Encoder**       | DINOv2 → StudentEncoder (56K)      | StudentEncoder FROZEN + ParticleEncoder      |
| **Representation**| Single 128-D class latent          | K=16 particle set (d=128 each)               |
| **Dynamics**      | None — static encoder              | ParticleTransformer (446K, 2L, 4H)           |
| **Contact**       | Not applicable                     | Signed distance head + ACh-gated attention   |
| **Skill planning**| Not applicable                     | THICK GRU + SCaR transition loss             |
| **GPS grounding** | Not used                           | gps_grounding_loss_fast (direct, not probe)  |
| **Reward signal** | AUROC on holdout set               | 7-signal neuromodulator (DA×5HT×NE×ACh…)    |
| **Training**      | Normal class only (unsupervised)   | State-action trajectories, all domains       |
| **NPU**           | XINT8 — 0.34ms/sample              | XINT8 student frozen, dynamics on CPU        |
| **Baseline**      | 27.7× AUROC/MB vs TinyGLASS       | 18× fewer params vs LeWorldModel             |

---

## Sprint Plan (CWM)

```
Sprint 1 — train_cwm_v2.py       🔄 RUNNING — epoch 3 done (loss 107), epoch 4 active
Sprint 2 — RECON TemporalHead    ⏳ WAITING — epoch 5 checkpoint needed (loss ~65)
Sprint 3 — Multi-domain          OGBench-Cube + PushT + TwoRoom + MVTec
Sprint 4 — GRASP planner + VL Sink  GRASP (arXiv:2602.00475) adopted — see below
Sprint 5 — Full eval vs LeWM     Paper submission target
```

## Key Files

```
neuromodulator.py               CORTEX-PE v16.11 — 7-signal system (SOURCE)
cwm_neuro_reward.py             done DONE — NeuromodulatedCWMLoss + SevenSignalMPCPlanner
CWM_MASTER_PLAN.md              Full sprint plan, architecture, references
CWM_SWM_INTEGRATION_PLAN.md    VL Sink, cross-temporal pairing, GeoLatentDB
CWM_SKILL_COMPOSITION_LEGGED_RESEARCH.md  THICK, SCaR, HiLAM, hexapod/quad
CWM_DOPAMINE_REWARD_SYSTEM.md  Domain signal mappings (superseded by cwm_neuro_reward.py)
```

## Novelty Claims

1. **Multi-domain shared backbone** — same 56K weights for cardiac audio,
   telemetry, visual inspection, outdoor navigation, and manipulation.
   No published system does this across this range of modalities.

2. **Direct GPS physical grounding** — `gps_grounding_loss_fast` uses GPS
   coordinates as a first-class training signal, not a downstream probe.
   LeWorldModel uses probe-based grounding.

3. **Biological neuromodulator reward shaping** — the same 7-signal system
   (derived from mouse V1 Neuropixels recordings) drives both the CORTEX-16
   trading system and the CWM world model training. DA = cos_sim prediction
   error is computed from the same latent transition that CWM minimises.

4. **18× parameter efficiency** — ~831K vs LeWM's 15M, AMD NPU deployment,
   no GPU required.

---
---

## GRASP Planner — Sprint 4 (arXiv:2602.00475)

**Paper:** Parallel Stochastic Gradient-Based Planning for World Models  
**Authors:** Psenka, Rabbat, Krishnapriyan, LeCun, Bar — Meta FAIR / UC Berkeley / NYU (Jan 2026)  
**Adopted:** Sprint 4, replaces/augments SevenSignalMPCPlanner

### Problem with current sampling planner

SevenSignalMPCPlanner is zero-order (Mirror Ascent + K random candidates). Performance
degrades at long horizons (T=25+) and high action dimensionality. GRASP solves both.

### GRASP — three components

**1. Lifted virtual states**
Instead of serial rollout , optimise intermediate states
 directly as independent variables alongside actions. Dynamics constraint
becomes a soft penalty. All T world model calls run in parallel → better conditioning,
faster convergence at long horizons.

**2. Langevin noise on states**
Gaussian noise injected into virtual states during optimisation. Enables escape from
local minima while retaining gradient information. Outperforms CEM by +10% success rate
at less than half the compute time (paper results on PushT/D4RL/DMControl).

**CORTEX extension:** Langevin noise magnitude gated by E/I neuromodulator signal.
High E/I → more noise → broader search (EXPLORE regime).
Low E/I → less noise → precise execution (EXPLOIT regime).
This is a novel biological interpretation not present in the GRASP paper.

**3. Grad-cut on state inputs**
Stop-gradient on state inputs to world model during planning rollout.
Only action-input gradients flow. Prevents adversarial exploitation of brittle Jacobians.
Same principle already used in JEPA target path — apply to planning pass.

### Mapping to CORTEX particle architecture

| GRASP | CORTEX CWM |
|-------|-----------|
| Virtual states s_1..s_T | Particle sets (B, K, D) |
| Langevin noise magnitude | E/I neuromodulator (biological) |
| Grad-cut on state inputs | JEPA stop-gradient principle |
| Dense one-step goal loss | GPS grounding loss per step |
| GD synchronisation step | EXPLOIT regime fine-tune pass |

### Implementation strategy



**Latency gate:** Benchmark one GRASP step on GMKtec EVO-X2 CPU before committing.
Target: < 10ms/step to maintain 4Hz RECON frame rate.

### Novelty claim

E/I-gated Langevin noise in GRASP is a CORTEX contribution — biologically-modulated
stochastic planning. Not in the GRASP paper, not in any existing neuromodulator work.

---

## NM-JEPA — Standalone Reference Implementation

`nm_jepa.py` is the publishable distillation of the CWM training algorithm.

**Verified on GMKtec EVO-X2:**
- encoder: 132,736 params
- predictor: 662,152 params  
- total: 794,888 params
- z_pred norm: 1.0000 (unit hypersphere)
- Self-test: all assertions pass, backward clean

NM-JEPA = CORTEX CWM training algorithm without particles, GPS, THICK GRU, or
domain-specific engineering. One file, no dependencies beyond PyTorch.

---

*CORTEX Architecture · 2026-03-31 · GMKtec EVO-X2 (AMD Ryzen AI MAX+ 395, 128GB) · no GPU*
