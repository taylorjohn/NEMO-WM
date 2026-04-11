# CORTEX — Claude Code Project Guide

> **Hardware:** AMD Ryzen AI MAX+ 395 NUC · IP 192.168.1.195  
> **Display:** Mac Glass Cockpit · IP 192.168.1.150  
> **Package:** cortex-brain v5.6.0  
> **Python:** 3.12 · conda ryzen-ai-1.7.0 · torch 2.10.0+cpu · numpy 1.26.4

---

## Critical Rules — Read Before Anything Else

```
NEVER run python without first activating the conda env:
  conda activate ryzen-ai-1.7.0

NEVER modify cortex_brain/trading/safeguards.py without running pytest after.

NEVER touch run_trading.py, test_run_trading.py, or cortex_alpaca_bridge.py
  during a live trading session (9:30–16:00 ET on weekdays).

NEVER lower the Z threshold below 2.0 or the circuit breaker below -2%.

NEVER commit .env — it contains live Alpaca API keys.

ALWAYS run pytest test_run_trading.py -v before any trading session.
ALWAYS run pytest after any change to cortex_brain/trading/*.
```

---

## Repository Structure

```
CORTEX/
│
├── cortex_brain/                   # Installed package (cortex-brain v5.6.0)
│   └── trading/
│       ├── safeguards.py           # 13-phase trading safeguard system
│       └── ...
│
├── # ── CORTEX-16 Trading ──────────────────────────────────────────────
├── run_trading.py                  # PRODUCTION — main trading loop
├── cortex_alpaca_bridge.py         # Alpaca REST API bridge
├── mac_glass_cockpit.py            # UDP HUD display → 192.168.1.150
├── test_run_trading.py             # 50-test pytest suite (gate for all changes)
├── market_replay.py                # Scenario replay engine
├── liquidate.py                    # Emergency position liquidation
│
├── # ── CORTEX-PE Perception Engine ────────────────────────────────────
├── eval_cardiac_audio.py           # Cardiac AUROC evaluation
├── eval_smap.py                    # SMAP/MSL telemetry AUROC evaluation
├── eval_mvtec.py                   # MVTec AD visual inspection evaluation
├── train_dinov2_distill.py         # MuRF multi-scale distillation training
├── export_onnx.py                  # XINT8 NPU export via AMD Quark
├── verify_npu.py                   # NPU hardware verification
├── verify_npu_hardwired.py         # Hardwired NPU smoke test
├── npu_stress_test.py              # NPU sustained-load test
│
├── # ── CWM World Model ─────────────────────────────────────────────────
├── neuromodulator.py               # SOURCE — 7-signal neuromodulator system (v16.11)
├── cwm_neuro_reward.py             # done NeuromodulatedCWMLoss + SevenSignalMPCPlanner
├── cwm_moe_jepa.py                 # done MoE-JEPA predictor — SparseMoEFFN + routing
├── train_cwm.py                    # done Sprint 1 — full training loop (smoke test first)
├── inspect_recon.py                # done Sprint 2 — RECON HDF5 pre-flight check
├── train_recon_temporal.py         # done Sprint 2 — TemporalHead InfoNCE training
├── eval_recon_auroc.py             # done Sprint 2 — RECON quasimetric AUROC
├── inspect_domains.py              # done Sprint 3 — all-domain pre-flight check
├── domain_loaders.py               # done Sprint 3 — unified multi-domain DataLoader
├── train_cwm_multidomain.py        # done Sprint 3 — multi-domain + OGBench hierarchical
├── cwm_hierarchical.py             # done Sprint 3 — THICK+SCaR+SPlaTES skill composition
├── eval_cwm_all_domains.py         # done Sprint 3/5 — full domain comparison table vs LeWM
├── cortex_geo_db.py                # done Sprint 4 — GeoLatentDatabase + VirtualLookaheadSink
├── eval_recon_vl_sink.py           # done Sprint 4 — VL Sink three-way ablation
├── eval_cwm_ablations.py           # done Sprint 5 — per-component ablation (A0–A6)
├── generate_paper_results.py       # done Sprint 5 — paper_results/ document generator
├── latent_predictor.py             # Legacy prototype (superseded by cwm_moe_jepa.py)
│
├── # ── Supporting ML ───────────────────────────────────────────────────
├── moe_router_v2.py                # Mixture-of-Experts router
├── manifold_projection.py          # Manifold projection utilities
├── active_sampler.py               # Active learning sampler
├── bok_planner.py                  # Best-of-K Mirror Ascent planner
├── engine.py                       # CortexEngine base class
├── hardware.py                     # Hardware detection utilities
│
├── # ── Config & Build ──────────────────────────────────────────────────
├── pyproject.toml                  # cortex-brain v5.6.0 package config
├── vaip_config.json                # Vitis AI Inference Provider config
├── launch_cortex.bat               # Windows startup batch script
├── cortex_init.sh                  # Session initialisation script
│
├── # ── Checkpoints ─────────────────────────────────────────────────────
├── checkpoints/
│   ├── cardiac/student_best.pt     # Cardiac StudentEncoder — AUROC 0.8894
│   ├── dinov2_student/student_best.pt  # MVTec standalone — AUROC 0.7393
│   └── dinov2_murf075/             # MuRF 0.75-scale — RETIRED (0.7297)
│
├── # ── Data ────────────────────────────────────────────────────────────
├── cardiac_data/                   # PhysioNet 2016 WAV clips (400 eval)
├── recon_data/recon_release/       # RECON HDF5 files (jackal_2019-*.hdf5)
├── allen_cache/                    # Allen Brain Observatory NWB files
│   └── session_715093703/          # Mouse V1 Neuropixels (256 neurons)
│
└── # ── Docs ─────────────────────────────────────────────────────────────
    ├── CLAUDE.md                   # This file
    ├── ARCHITECTURE.md             # CPE vs CWM architectural spec
    ├── amd_npu_layernorm_post.md   # Unpublished — publish this
    └── CWM_MASTER_PLAN.md          # Full CWM sprint plan
```

---

## System 1 — CORTEX-PE (Perception Engine)

### What it does
Anomaly detection across 6 heterogeneous sensor domains using a single shared
StudentEncoder backbone (56K params, AMD NPU XINT8). No action conditioning —
encode frames, score against normal distribution, report AUROC.

### Architecture
```
Input frame/signal
    → DINOv2-small teacher (22.1M, frozen reference only)
    → StudentEncoder (56K params, 128-D latent, XINT8 NPU, 0.34ms)
    → PCA fitted on normal-class training windows only
    → k-NN distance (k=32) as anomaly score
    → AUROC on held-out test set
```

### Confirmed Results

| Domain        | AUROC  | Command / Config                                      |
|---------------|--------|-------------------------------------------------------|
| Cardiac audio | 0.8894 | `eval_cardiac_audio.py --student checkpoints\cardiac\student_best.pt --data cardiac_data --max-per-class 200` |
| Cardiac audio | 0.7730 | Same + `--max-per-class 500` (harder full-dataset)    |
| SMAP/MSL      | 0.8427 | hybrid+semi done 71/81            | eval_smap_combined.py n_labeled=20 |
| MVTec AD      | 0.7393 | `eval_mvtec.py` standalone |
| MVTec ensemble| 0.8855 | `eval_mvtec_ensemble.py` DINOv2+student 512-D |

### Active TODO (in priority order)

~~**SMAP Fix**~~ done **DONE 2026-03-30 — AUROC 0.8427 (71/81, n_labeled=20)**
```
eval_smap_combined.py --data smap_data
Hybrid PCA+drift (56 channels) + semi-supervised LDA (25 hard channels).
T-2: 0.04→0.82  T-4: 0.33→0.92  M-7: 0.39→0.99  E-4: 0.15→0.99
Remaining failures (5): T-1, E-3, F-1, A-6, T-8
```

**1. MVTec Ensemble Eval — Recover 0.8152** ← DO NEXT
```
Problem:  0.8152 came from DINOv2 teacher + student ensemble path, now lost.
Fix:      Concat teacher features (384-D) + student (128-D) → 512-D k-NN.
Steps:
  1. Get-ChildItem -Filter "eval_mvtec*" | Select-Object FullName, LastWriteTime
  2. Get-ChildItem -Filter "*.py" -Recurse | Select-String -Pattern "ensemble"
  3. python eval_mvtec.py --student checkpoints\dinov2_student\student_best.pt --ensemble --k 32
```

**2. Publish amd_npu_layernorm_post.md**
```
File exists at CORTEX root. Just needs publishing.
Get-Content amd_npu_layernorm_post.md | Select-Object -First 5
```

### NPU Export Workflow
```powershell
# Export StudentEncoder to XINT8 ONNX for NPU
python export_onnx.py --checkpoint checkpoints\dinov2_student\student_best.pt --output cortex_encoder_int8.onnx

# Verify NPU can run it
python verify_npu.py

# Stress test sustained inference
python npu_stress_test.py
```

### MuRF Distillation — Reference
```powershell
# Multi-scale distillation (use scales=[1.0,1.5] only — 0.75 hurt 56K student)
python train_dinov2_distill.py --murf --scales 1.0 1.5 --epochs 30

# Results log:
#   scales=[0.75,1.0,1.5]  → final loss 0.0006 → MVTec AUROC 0.7297 (NEGATIVE — retired)
#   scales=[1.0,1.5]       → use this for next run
#   256K student experiment → pending (more capacity → expect >0.80)
```

---

## System 2 — CORTEX WM (World Model)

### What it does
Action-conditioned latent particle dynamics across manipulation, navigation, and
locomotion domains. Predicts next particle state given current particles + action.
Plans optimal action sequences using biologically-modulated MPC.

### Architecture
```
Input frame + action
    → StudentEncoder FROZEN (56K, XINT8 NPU) → 128-D latent
    → ParticleEncoder (296K) → K=16 SpatialSoftmax particles (B,16,128)
    → ContactAwareParticleTransformer (446K, d=128, 2L, 4H)
        ├── StructuredActionModule — per-particle action relevance gate
        ├── SignedDistanceHead (+8K) — contact-weighted attention
        └── THICK Context GRU (+25K) — slow-timescale skill dynamics
    → NeuromodulatedCWMLoss (7-signal biological reward)
    → SevenSignalMPCPlanner — Mirror Ascent, g_t = −L/(1+ρ)
```

### Parameter Budget

| Component          | Params  | Hardware      |
|--------------------|---------|---------------|
| StudentEncoder     | 56,592  | AMD NPU XINT8 |
| ParticleEncoder    | 295,808 | CPU           |
| ParticleDynamics   | 446,464 | CPU           |
| ContactHead        | +8,000  | CPU           |
| THICK Context GRU  | +25,000 | CPU           |
| **Total**          | **~831K** | **NUC, no GPU** |
| LeWorldModel       | 15M     | A100          |
| **Advantage**      | **18× smaller** |          |

### Domains and Action Dimensions

| Domain         | action_dim | K particles | Notes                        |
|----------------|-----------|-------------|------------------------------|
| RECON outdoor  | 2         | 16          | HDF5 4Hz, GPS grounding      |
| OGBench-Cube   | 9         | 16          | 9-DOF arm, contact critical  |
| TwoRoom        | 2         | 16          | 2D navigation                |
| PushT          | 2         | 16          | Contact manipulation         |
| Hexapod (future)| 4 (CPG)  | 18          | 6-leg, CPG-encoded           |
| Quadruped (future)| 12     | 8           | 4-leg + 4 terrain particles  |

Action zeros are used for observation-only domains (cardiac, SMAP, MVTec).

### Neuromodulator System (neuromodulator.py v16.11)

Seven biological signals, all derived from latent transitions:

| Signal | Source                           | Range   | CWM Loss Role              |
|--------|----------------------------------|---------|----------------------------|
| DA     | `(1−cos_sim(z_pred,z_act))/2`   | [0,1]   | L_predict scale            |
| 5HT    | `exp(−10·std(z_history))`       | [0,1]   | L_gaussian (SIGReg) scale  |
| NE/rho | Allen Neuropixels spike rate     | [0,1]   | L_gps weight + plan temp   |
| ACh    | `(DA + 1−5HT) / 2` EMA          | [0,1]   | L_contact sharpening       |
| E/I    | `DA / (1−5HT+ε)`                | [0.5,2] | action_std, gait frequency |
| Ado    | `elapsed / saturation_time`      | [0,1]   | Curriculum pacing          |
| eCB    | `DA × ‖action‖` EMA             | [0,1]   | L_skill damp (−40% max)    |

Four regimes (DA × 5HT):
- **EXPLORE** (DA↑ 5HT↑) — high lr, diverse domains
- **WAIT** (DA↑ 5HT↓) — low lr, pause and observe
- **REOBSERVE** (DA↓ 5HT↓) — stabilise, strengthen SIGReg
- **EXPLOIT** (DA↓ 5HT↑) — fine-tune, commit, high K candidates

### Sprint Plan

```
Sprint 1  train_cwm.py             done WRITTEN — python train_cwm.py --smoke
Sprint 2  Cross-temporal RECON     done WRITTEN — SPRINT2_EXECUTION.md
Sprint 3  Multi-domain             done WRITTEN — SPRINT3_EXECUTION.md
Sprint 4  VL Sink + GeoLatentDB   done WRITTEN — SPRINT4_EXECUTION.md
Sprint 5  Full eval vs LeWM        done WRITTEN — SPRINT5_EXECUTION.md
```

### Ready Files (all sprints)

```
# Sprint 1
neuromodulator.py          SOURCE — 7-signal system, import this
cwm_neuro_reward.py        NeuromodulatedCWMLoss + SevenSignalMPCPlanner
cwm_moe_jepa.py            MoEJEPAPredictor + SparseMoEFFN
train_cwm.py               Full Sprint 1 training loop

# Sprint 2
inspect_recon.py           Pre-flight HDF5 check
train_recon_temporal.py    TemporalHead InfoNCE training
eval_recon_auroc.py        Quasimetric AUROC + k-sweep

# Sprint 3
inspect_domains.py         All-domain data check
domain_loaders.py          Unified DataLoader (RECON/OGBench/PushT/TwoRoom/SMAP/MVTec)
train_cwm_multidomain.py   Multi-domain fine-tune + hierarchical OGBench
cwm_hierarchical.py        THICK + SCaR + SPlaTES

# Sprint 4
cortex_geo_db.py           GeoLatentDatabase + VirtualLookaheadSink
eval_recon_vl_sink.py      Three-way VL Sink ablation

# Sprint 5
eval_cwm_all_domains.py    Full comparison table vs LeWM
eval_cwm_ablations.py      A0–A6 per-component ablation
generate_paper_results.py  → paper_results/ (4 markdown docs + JSON)

# Execution guides
SPRINT2_EXECUTION.md  SPRINT3_EXECUTION.md
SPRINT4_EXECUTION.md  SPRINT5_EXECUTION.md
```

### RECON Data Structure
```python
# HDF5 file format — check before Sprint 1
import h5py
f = h5py.File('recon_data/recon_release/jackal_2019-XX.hdf5', 'r')
# Keys: images (JPEG bytes), actions, observations, GPS
# Decode images: PIL.Image.open(io.BytesIO(jpeg_bytes))
# Frame rate: 4Hz
# Action dim: 2 (forward velocity, angular velocity)
```

---

## System 3 — CORTEX-16 (Trading)

### What it does
Live algorithmic trading on Alpaca paper API. Signal from Allen Brain Observatory
Neuropixels (mouse V1 arousal) → Z-score → 13-phase safeguard system → surgical
market orders. Mac Glass Cockpit at 192.168.1.150 shows live telemetry via UDP.

### Daily Startup Sequence (9:30 ET)
```powershell
# Step 1 — Pre-flight (ALWAYS)
conda activate ryzen-ai-1.7.0
pytest test_run_trading.py -v
# Must be 50/50 before proceeding

# Step 2 — Cockpit (Terminal 1)
python mac_glass_cockpit.py

# Step 3 — Trading engine (Terminal 2, exactly 9:30 ET)
python run_trading.py --hud-ip 127.0.0.1
```

### Key Thresholds (as of 2026-03-30)

| Parameter         | Value    | Location                           |
|-------------------|----------|------------------------------------|
| Z threshold       | 2.18     | phase5_pdt_cooldown sigma_threshold |
| RTT threshold     | 150ms    | phase2_rtt_watchdog threshold_ms   |
| Spread max        | $0.05    | MAX_SPREAD (post-fix)              |
| Circuit breaker   | −2% vault| DailyCircuitBreaker max_loss_pct   |
| Entry cooldown    | 60s      | _last_entry_ts gate in act()       |
| Market hours      | 9:30–16:00 ET | MarketHoursGuard               |

### Signal Chain

```
Allen Neuropixels NWB file (256 mouse V1 neurons)
    → rolling 33ms spike window → population rate ρ
    → Z-score: (ρ − μ) / σ  via Rust math engine (sub-ms)
    → 13-phase safeguard filter (RTT, spread, PDT, flash crash…)
    → VWMP surgical order if Z ≥ 2.18 and all phases pass
    → 50/50 vault sweep on profit
    → UDP telemetry → Mac HUD (192.168.1.150:5005)
```

### 13-Phase Safeguard Summary

| Phase | Name                   | Threshold               | Current Status |
|-------|------------------------|-------------------------|----------------|
| P1    | VWMP Execution         | Inside spread           | done             |
| P2    | RTT Watchdog           | < 150ms                 | done (was 45ms)  |
| P3    | Order TTL              | 3s cancel               | done             |
| P4    | Stale Data             | < 1.5s old              | done             |
| P5    | PDT Cooldown / Z-gate  | Z ≥ 2.18, dt=3          | done (was 999)   |
| P6    | API Governor           | 200 req/min             | done             |
| P7    | Market Clock           | 9:30–16:00 ET           | done             |
| P8    | Buying Power           | DTBP available          | done             |
| P9    | Spread Gate            | ≤ $0.05                 | done (was $0.15) |
| P10   | Flash Crash            | < 2% drop in 60 ticks   | done             |
| P11   | EOD Liquidation        | 15:55 ET hard exit      | done             |
| P12   | Rust Math Engine       | Sub-ms Z calc           | done             |
| P13   | Toxic Flow (VPIN-lite) | OBI < 10×               | done             |

### Bugs Fixed (2026-03-30) — Do Not Reintroduce

```python
# Bug 1 — dt=999 sentinel (FIXED in run_trading.py line ~347)
# OLD: "dt": (999 if (pattern_day_trader and equity>=25000) else ...)
# NEW: "dt": (3 if float(getattr(a,"equity",0))>=25000 else max(0, 3-daytrade_count))

# Bug 2 — RTT threshold hardcoded 45ms (FIXED in cortex_brain/trading/safeguards.py)
# OLD: threshold_ms: float = 45.0
# NEW: threshold_ms: float = 150.0

# Bug 3 — Spread ap=0 fallback (FIXED in run_trading.py line ~193)
# When ask=0: ask = trade_price + 0.01; bid = trade_price - 0.01
```

### Danger Scenarios (NEVER_TRADE_SCENARIOS)
```python
BEARISH_SCENARIOS = ["FLASH_CRASH", "CIRCUIT_BREAKER", "EOD_SWEEP"]
# These must NEVER trigger a long entry — only exit logic
# SNIPER and HUNTER modes: position sizing capped to 0.5%
```

### Emergency Commands
```powershell
# Liquidate all positions immediately
python liquidate.py

# Kill trading loop safely (no open orders → just Ctrl+C)
# Only safe when: no position, no pending orders, paper account
```

### Vault Status (as of 2026-03-30)

| Metric          | Value         |
|-----------------|---------------|
| Equity          | $101,133.47   |
| Circuit breaker | −$2,022 (−2%) |
| Mode            | PAPER         |
| Symbol          | SPY + fleet   |

---

## Hardware & Environment

### NUC Setup
```powershell
# Activate environment (always first)
conda activate ryzen-ai-1.7.0

# Verify NPU is reachable
python verify_npu.py

# Check AMD Quark (XINT8 quantization tool)
python -c "import quark; print(quark.__version__)"

# Check torch
python -c "import torch; print(torch.__version__)"  # expects 2.10.0+cpu
```

### Network
```
NUC:     192.168.1.195  — all ML inference, trading engine
Mac HUD: 192.168.1.150  — Glass Cockpit display (UDP port 5005)
```

### Key External Data
```
Allen Brain Observatory: session_715093703
  Path: allen_cache/session_715093703/session_715093703.nwb
  Content: 256 mouse V1 neurons, Neuropixels recordings
  Use: NE/rho signal for trading + neuromodulator system

PhysioNet 2016:
  Path: cardiac_data/
  Use: Cardiac anomaly detection eval (400 clips = 200/class)

RECON outdoor nav:
  Path: recon_data/recon_release/jackal_2019-*.hdf5
  Use: CWM Sprint 1-2 training data
```

### NPU Export (AMD Quark XINT8)
```python
# Only XINT8 supports LayerNorm on AMD Ryzen AI NPU
# BF16 and A16W8 have LayerNorm CPU fallback — avoid
# Export requires opset 17 ONNX
# See: amd_npu_layernorm_post.md for full explanation
```

---

## Testing

### Trading System (mandatory gate)
```powershell
pytest test_run_trading.py -v
# Must pass 50/50 before any trading session
# Must pass 50/50 after any change to cortex_brain/trading/ or run_trading.py
```

### PE Smoke Tests
```powershell
# Quick sanity before full eval
python eval_cardiac_audio.py --student checkpoints\cardiac\student_best.pt --data cardiac_data --max-per-class 10 --epochs 2
# Expect: AUROC > 0.80 even on 20 clips
```

### NPU Verification
```powershell
python verify_npu.py         # basic
python npu_stress_test.py    # sustained load
python final_verify.py       # full system check
```

---

## Common Workflows

### Run a domain eval
```powershell
conda activate ryzen-ai-1.7.0

# Cardiac
python eval_cardiac_audio.py --student checkpoints\cardiac\student_best.pt --data cardiac_data --max-per-class 200

# MVTec
python eval_mvtec.py --student checkpoints\dinov2_student\student_best.pt --k 32

# SMAP (combined hybrid+semi — canonical command)
python eval_smap_combined.py --data smap_data
# → AUROC 0.8427, 71/81 channels, 4.7s
```

### Train a new student checkpoint
```powershell
# Standard distillation (no MuRF)
python train_dinov2_distill.py --epochs 30 --domain cardiac

# MuRF multi-scale (use scales 1.0 and 1.5 only — 0.75 hurts 56K model)
python train_dinov2_distill.py --murf --scales 1.0 1.5 --epochs 30
```

### Export to NPU
```powershell
python export_onnx.py --checkpoint checkpoints\dinov2_student\student_best.pt --output cortex_encoder_int8.onnx --quant xint8
python verify_npu.py --model cortex_encoder_int8.onnx
```

### Run CWM Sprint 1
```powershell
# 1. Verify RECON data
python inspect_recon.py --hdf5-dir recon_data\recon_release

# 2. Smoke test (no data needed, ~30s)
python train_cwm.py --smoke

# 3. Quick validation (5 files, 5 epochs)
python train_cwm.py --hdf5-dir recon_data\recon_release --max-files 5 --epochs 5

# 4. Full training
python train_cwm.py --hdf5-dir recon_data\recon_release --epochs 30

# See SPRINT2_EXECUTION.md, SPRINT3_EXECUTION.md, etc. for next steps
```

---

## Novelty Claims (for paper)

1. **Multi-domain shared backbone** — 56K StudentEncoder serves cardiac audio,
   telemetry, visual inspection, outdoor navigation, and manipulation. No
   published system does this across this range of modalities.

2. **Direct GPS grounding** — `gps_grounding_loss_fast` uses GPS as a first-class
   training signal, not a downstream probe (unlike LeWorldModel).

3. **Biological neuromodulator reward** — 7-signal system (mouse V1 Neuropixels)
   drives both CORTEX-16 trading and CWM world model training. DA = cos_sim
   prediction error is the same signal CWM's dynamics module minimises.

4. **Parameter efficiency** — ~831K vs LeWorldModel's 15M. 18× smaller, no GPU.

---

## Reference Numbers

| Metric              | Value     | Notes                          |
|---------------------|-----------|--------------------------------|
| StudentEncoder      | 56,592    | params, XINT8 NPU, 0.34ms      |
| CWM total           | ~831K     | no GPU                         |
| LeWorldModel        | 15M       | A100                           |
| Cardiac AUROC       | 0.8894    | 400 clips, k=32                |
| SMAP AUROC          | 0.8427    | hybrid+semi 71/81 done           |
| MVTec AUROC         | 0.7393    | standalone student             |
| MVTec ensemble      | 0.8855    | DINOv2+student 512-D done DONE       |
| Vault               | $101,133  | paper account                  |
| Z threshold         | 2.18      | Phase 5                        |
| pytest              | 50/50 done  | as of 2026-03-30               |

---

*Last updated: 2026-03-30 | cortex-brain v5.6.0 | CORTEX-PE v16.17*
