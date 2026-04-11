# CORTEX — AI Perception, World Model & Trading
**Hardware:** GMKtec EVO-X2 · AMD Ryzen AI MAX+ 395 · 128GB RAM · AMD NPU XINT8
**Repository:** github.com/taylorjohn

---

## Three Pillars

### NeMo-WM (Neuromodulated World Model)
Visual world model for outdoor robot navigation. Trained on RECON Berkeley campus (10,995 trajectories, 4Hz). Full computational dissociation achieved between VLM-grounded landmark system and proprioceptive path integration system.

**Headline results:**
- No-VLM AUROC=**0.9997** (k_ctx=32 ep11, top1_acc=0.9985, neg_dist=1.035, sep=+1.001)
- k_ctx=32 hard negatives past unit sphere (neg_dist=1.035, separation=+1.001, pos_dist=0.034)
- Synthetic zero-shot No-VLM=0.9697 (no real-world training data)
- HD:vel dominance ratio 9–43:1, timescale-invariant
- Grid cell AUROC 0.51→0.85 monotonic with GPS distance
- Paper v18 FINAL — arXiv-ready (cs.LG + cs.RO, CC BY 4.0)

### CORTEX-PE (Perception Engine)
Multi-domain anomaly detection. Single 56K-param encoder, no domain-specific retraining, 0.34ms NPU inference.

| Domain | Dataset | Score | Method |
|--------|---------|-------|--------|
| Bearing vibration | CWRU | **AUROC 1.0000** | PCA k=6, zero labels |
| Industrial audio | MIMII (4 machines) | **AUROC 0.9313** | Log-mel PCA k=32 |
| Outdoor navigation | RECON | **AUROC 0.9997** | Proprio k_ctx=32, 26K params |
| Visual inspection | MVTec AD (ensemble) | **AUROC 0.8923** | Student+DINOv2 512-D |
| Visual inspection | MVTec AD (patch) | **14/15** | DINOv2 patch DA |
| Cardiac audio | PhysioNet 2016 | **AUROC 0.7730** | Student encoder k=32 |
| Telemetry | SMAP/MSL (81 ch) | **AUROC 0.7730** | Hybrid PCA+drift w=128 |

26.3x better parameter efficiency than TinyGLASS (3.10 AUROC/MB vs 0.118).

### CORTEX-16 (Trading)
Live algorithmic trading on Alpaca paper API. 18-phase safeguard architecture. 99/99 pytest checks passing.

---

## Quick Start

### NeMo-WM evaluation
```powershell
# Dissociation eval (k_ctx=16 best)
python eval_recon_auroc.py `
    --head-ckpt checkpoints\cwm\temporal_head_sprint3.pt `
    --cwm-ckpt  checkpoints\cwm\cwm_multidomain_best.pt `
    --hdf5-dir  recon_data\recon_release `
    --n-pairs 1000 --k-pos 4 `
    --proprio-compare --hard-negatives --proprio-no-gps `
    --proprio-ckpt checkpoints\cwm\proprio_kctx16_best.pt

# Path integration ablation
python eval_path_integration_ablation.py `
    --proprio-ckpt checkpoints\cwm\proprio_kctx16_best.pt `
    --n-pairs 500

# Grid cell test
python eval_place_cell_receptive_fields.py `
    --grid-test --no-place `
    --cwm-ckpt checkpoints\cwm\cwm_multidomain_best.pt `
    --hdf5-dir recon_data\recon_release
```

### Sprint 9 synthetic pipeline
```powershell
# Generate synthetic trajectories
python synthetic_trajectory_generator.py `
    --n-trajectories 500 `
    --out-dir recon_data\synthetic_sprint9

# Pre-train on synthetic (physics-diverse)
python train_proprio_6c.py `
    --hdf5-dir recon_data\synthetic_sprint9 `
    --out-ckpt checkpoints\cwm\proprio_synth_pretrain.pt `
    --k-ctx 16 --epochs 10

# Fine-tune on real RECON
python train_proprio_6c.py `
    --hdf5-dir recon_data\recon_release `
    --out-ckpt checkpoints\cwm\proprio_kctx16_finetuned.pt `
    --k-ctx 16 --epochs 10 `
    --init-ckpt checkpoints\cwm\proprio_synth_pretrain.pt
```

### Trading
```powershell
python -m pytest cortex_brain/ -q   # 99/99 must pass
python cortex_live_v1_fixed.py      # launches engine, hibernates until 9:30 ET
```

---

## ACh Sweep Results (complete)

| k_ctx | Window | No-VLM AUROC | Training data |
|-------|--------|-------------|--------------|
| 2 | 0.5s | 0.925 | RECON |
| 4 | 1.0s | 0.961 | RECON |
| 8 | 2.0s | 0.977 | RECON |
| 16 | 4.0s | 0.9974 | RECON |
| **32** | **8.0s** | **0.9997** | **RECON** |
| 16 (synth) | 4.0s | **0.9697** | **500 synthetic files only** |

---

## Paper

**NeMo-WM v18 FINAL** — `NeMo_WM_Paper_v18_FINAL.docx`

| Contribution | Result |
|-------------|--------|
| Eight-signal neuromodulator replaces JEPA | AUROC 0.9837, zero L_jepa gradient |
| Cortisol as 8th signal | r=0.768 lag-1 loss prediction |
| Full computational dissociation | No-VLM=0.9974 > VLM-only=0.928 |
| Heading-dominant PI (timescale-invariant) | HD:vel ratio 9–43:1 |
| Synthetic zero-shot transfer | No-VLM=0.9697 without real-world data |
| Grid cell / spatial encoding | AUROC 0.51→0.85 monotonic |
| ACh temporal sweep | Superlinear improvement with window length |
| GRASP planner (DA-gated) | 3.60ms reactive / 6.59ms exploit |

**arXiv gate:** make github.com/taylorjohn public → submit cs.LG + cs.RO, CC BY 4.0.

---

## Documentation

| File | Contents |
|------|----------|
| `ARCHITECTURE.md` | Full component specs, ablation tables, checkpoints, references |
| `CWM_STATUS.md` | Sprint status, all experimental results, paper checklist |
| `CWM_NEXT_STEPS.md` | Prioritised action items, Sprint 9 roadmap, arXiv checklist |

---

## Blind Navigation — Operating Without Vision

**The fundamental finding:** NeMo-WM's proprioceptive encoder achieves
No-VLM AUROC=0.9997 using only five physical signals — velocity, angular
rate, heading (sin θ, cos θ), contact, and delta heading. No camera.
No GPS. No external signal of any kind.

### Sensor requirements for blind operation

| Signal | Sensor | Power | Cost |
|--------|--------|-------|------|
| Linear velocity | Wheel encoder | ~1mW | <$5 |
| Angular rate | IMU gyroscope | ~1mW | <$5 |
| Heading | IMU magnetometer | ~1mW | <$5 |
| Contact | Wheel current draw | 0mW | $0 |
| Delta heading | Computed from above | 0mW | $0 |

Total additional sensor cost: **~$10, ~3mW**. No camera required.

### What it can do blind

- Temporal self-localisation: AUROC **0.9997** (hard negatives k≥32)
- Know it is temporally close to 0-4 frames ago with near-perfect accuracy
- Distinguish frames 8+ seconds apart from nearby frames
- Dead-reckoning position estimate over 8-second windows
- Cortisol domain shift detection — knows environment changed without seeing it
- Navigate back to known physical states via GeoLatentDB

### What it cannot do blind

- Identify nearby objects or semantic content
- Correct long-horizon drift (visual landmarks needed for reset)
- Distinguish physically identical locations that look different (visual aliasing)
- Navigate new environments with no prior physical traversal

### The biological parallel

This is mammalian path integration. Rodents with visual cortex lesioned
still navigate mazes using head direction cells (heading), velocity
afferents (wheel encoder equivalent), and proprioception (contact).
McNaughton et al. 2006; Moser et al. 2008.

NeMo-WM's heading dominance ratio (HD:vel = ∞:1 at fine timescales,
9:1 at k_pos=4) mirrors head direction cell primacy in entorhinal cortex.
Timescale-invariant, as confirmed across k_pos ∈ {1, 4}.

### Deployment environments enabled

- Complete darkness (caves, night operations, power outages)
- Camera occlusion (dust, mud, smoke, water)
- Underwater and underground navigation
- Sensor failure fallback (camera fails → continue on IMU alone)
- Day/night transition without recalibration
- Environments where visual conditions change drastically

### The combined system

Blind operation (proprio) handles short-horizon self-localisation perfectly.
V-JEPA 2 visual pathway handles long-horizon drift correction and
environment generalisation. The two pathways are orthogonal — confirmed
by Full AUROC = No-VLM AUROC = 0.9997 exactly. Neither degrades the other.
The visual system adds zero overhead on a task the physical system has solved,
and the physical system adds zero overhead on tasks the visual system handles.

