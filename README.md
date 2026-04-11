<div align="center">

# 🧠 NeMo-WM
### Neuromodulated World Models for Edge-Deployed Robot Perception

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B-ee4c2c.svg)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/Hardware-AMD_NPU_XINT8-green.svg)](https://www.amd.com/)
[![arXiv](https://img.shields.io/badge/arXiv-cs.RO%20%7C%20cs.LG-b31b1b.svg)](https://arxiv.org/)
[![Last Updated](https://img.shields.io/badge/Updated-April_2026-blue.svg)]()

**A 26K-parameter proprioceptive encoder that outperforms V-JEPA 2 ViT-G (1034M parameters) on outdoor robot navigation.**

*Physics-grounded path integration beats internet-scale visual pretraining.*

</div>

---

## 🎯 Key Results

| Claim | Result | vs Best Alternative |
|-------|--------|-------------------|
| Temporal self-localisation | **AUROC 0.9997** | +0.117 vs ViT-G 1034M |
| Flow matching policy | **SR 100%** (350 eps, 6 seeds) | — |
| Full perception→action | **1.32ms** on $500 edge hardware | <2ms budget: 0.34% used |
| Synthetic pre-training | **top1_acc 0.9798** | → 0.9762 after 1 real ep |
| Multi-domain anomaly | **6 domains** | Single 56K encoder |

> **Key finding:** Visual scaling does not solve temporal self-localisation.  
> V-JEPA 2 ViT-G (1034M) scores 0.883 AUROC. NeMo-WM proprio (26K) scores **0.9997**.  
> The gap is +0.117 — physics-grounded path integration wins.

---

## 📊 Multi-Domain Anomaly Detection

Single 56K-param StudentEncoder · No domain retraining · 0.34ms NPU inference

```
Domain              Dataset              Score          Method
──────────────────────────────────────────────────────────────────────
⚙️  Bearing          CWRU                 AUROC 1.0000   PCA k=6
🏭  Industrial audio  MIMII (4 machines)   AUROC 0.9313   Log-mel k=32
🤖  Robot navigation  RECON (Berkeley)     AUROC 0.9997   Proprio k_ctx=32
👁️  Visual inspect.   MVTec AD (ensemble)  AUROC 0.8923   Student+DINOv2
👁️  Visual inspect.   MVTec AD (patch)     14/15          DINOv2 patch DA
❤️  Cardiac audio     PhysioNet 2016       AUROC 0.7730   Student k=32
🛰️  Telemetry         SMAP/MSL (81 ch)     AUROC 0.7730   Hybrid PCA+drift
```

### Parameter Efficiency (MVTec AD)

```
Method                    AUROC   Params    AUROC/MB   Edge
────────────────────────────────────────────────────────────
PatchCore (Roth 2021)     0.992   68M       0.015      ❌
TinyGLASS ResNet18 INT8   0.942   11M       0.118      ✅
CORTEX-PE Student (ours)  0.702   57K       3.10       ✅ NPU ← 26.3× TinyGLASS
CORTEX-PE Ensemble (ours) 0.892   22M       0.044      ✅
```

---

## 🧬 Architecture

NeMo-WM has three components forming a full perception-to-action pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                    NeMo-WM Architecture                         │
├──────────────┬──────────────────┬───────────────────────────────┤
│  Proprio     │  VLM Pathway     │  Flow Policy                  │
│  Encoder     │  (DINOv2)        │  (NeMoFlow)                   │
│              │                  │                               │
│  vel, ang    │  RGB frame       │  z + block_pos + DA + goal    │
│  heading     │      ↓           │           ↓                   │
│  contact     │  ViT-L/G         │  Flow matching ODE            │
│  delta_h     │      ↓           │           ↓                   │
│      ↓       │  z_visual        │  action chunk (H×2)           │
│  16-frame    │                  │                               │
│  attn pool   │                  │  SR=100% (n=350, 6 seeds)     │
│      ↓       │                  │  Latency: 0.46ms              │
│  z_proprio   │                  │                               │
│  AUROC=0.9997│  AUROC=0.907     │                               │
│  26K params  │  326M params     │                               │
└──────────────┴──────────────────┴───────────────────────────────┘
       ↑ Full computational dissociation confirmed ↑
```

### ACh Temporal Integration Sweep

The acetylcholine (ACh) signal controls temporal context window width k_ctx:

```
k_ctx   No-VLM AUROC   Interpretation
─────────────────────────────────────
2       0.925          1s context
4       0.961          2s context
8       0.977          4s context
16      0.9974         8s context
32      0.9997   ←★    16s context (production)
```

**Superlinear improvement** — mirrors Hasselmo 1999 ACh theory.  
Broader temporal integration = better path integration for slow 4Hz outdoor navigation.

### Scale Comparison (V-JEPA 2)

```
Model              Params    RECON AUROC   Notes
─────────────────────────────────────────────────────
V-JEPA 2 ViT-L     326M      0.907         Internet-scale
V-JEPA 2 ViT-G     1034M     0.883         Scaling HURTS
NeMo-WM Proprio    26K       0.9997   ←★   Physics-grounded
```

---

## ⚡ Hardware & Latency

```
Component           Latency    Throughput    Hardware
───────────────────────────────────────────────────────
Visual encoder      0.353ms    2,578 Hz      AMD NPU XINT8
Temporal head       0.146ms    8,389 Hz      AMD NPU XINT8
WM transition       0.356ms    2,916 Hz      AMD NPU XINT8
Total perception    0.855ms    1,169 Hz      AMD NPU XINT8
+ Flow policy       0.460ms    —             CPU
Full pipeline       1.315ms    763 Hz        AMD NPU + CPU
────────────────────────────────────────────────────────
4Hz navigation budget used: 0.34%
```

**Hardware:** GMKtec EVO-X2 · AMD Ryzen AI MAX+ 395 · 128GB RAM · ~$500  
**Minimum deployment:** Raspberry Pi 4 ($35), proprio-only, 28Hz

---

## 🧪 Neuromodulator Signals

Each biological signal maps to a computational function:

| Signal | Biological Role | NeMo-WM Function |
|--------|----------------|------------------|
| ACh | Temporal precision | k_ctx window width |
| Dopamine | Reward prediction error | GoalDA hot/cold temperature |
| Norepinephrine | Gain control | Heading dominance weight |
| Cortisol | Stress / domain shift | Cross-domain sensitivity |
| eCB | Inhibitory gating | Negative sampling suppression |

---

## 📁 Repository Structure

```
CORTEX/
├── train_proprio_6c.py          # Temporal encoder training (Sprint 6c)
├── train_nemo_wm_v2.py          # Unified v2 pipeline (Sprint F)
├── nemo_flow_policy.py          # Flow matching policy (SR=100%)
├── nemo_flow_policy_v2.py       # Orientation-aware policy (Sprint C)
├── stochastic_encoder.py        # VAE encoder, KL=DA signal (Sprint A)
├── proprio_decoder.py           # z→obs imagination (Sprint B, MAE=0.013)
├── cortisol_domain_adaptive.py  # Cortisol domain detector
├── neuro_vlm_gate.py            # VLM integration gate
├── eval_recon_auroc.py          # RECON dissociation eval
├── eval_mimii.py                # MIMII industrial eval (k=32)
├── eval_mvtec.py                # MVTec visual eval
├── eval_mvtec_ensemble.py       # MVTec ensemble (0.8923)
├── eval_cardiac_audio.py        # Cardiac audio eval
├── eval_smap_msl.py             # SMAP/MSL telemetry eval
├── checkpoints/
│   ├── cwm/
│   │   ├── proprio_kctx16_sprint9.pt    # Synthetic (top1=0.9798)
│   │   └── proprio_kctx16_recon_ft.pt  # RECON fine-tune (running)
│   ├── flow_policy/
│   │   └── nemo_flow_best.pt            # SR=100%
│   ├── nemo_v2/
│   │   ├── decoder_best.pt              # MAE=0.013
│   │   └── flow_v2_best.pt             # 3-dim orientation policy
│   └── dinov2_student/
│       └── student_best.pt             # MVTec AUROC=0.8923
└── figures/
    ├── auroc_sweep.png          # ACh k-sweep (300dpi)
    ├── dissociation.png         # No-VLM vs VLM-only
    ├── flow_comparison.png      # SR 2%→100%
    ├── pusht_flow_policy.gif    # PushT animation
    └── recon_neuromod.gif       # RECON + neuromod traces
```

---

## 🚀 Quick Start

```powershell
# Clone and setup
git clone https://github.com/taylorjohn/nemo-wm
cd nemo-wm
conda activate ryzen-ai-1.7.0

# Run RECON dissociation eval
python eval_recon_auroc.py `
    --proprio-ckpt checkpoints/cwm/proprio_kctx16_sprint9.pt `
    --hdf5-dir recon_data/recon_release `
    --n-pairs 1000 --k-pos 4 `
    --proprio-compare --hard-negatives --proprio-no-gps

# Run all domain benchmarks
python eval_mimii_perid.py --data mimii_data               # 0.9313
python eval_mvtec_ensemble.py --data data/mvtec            # 0.8923
python eval_cardiac_audio.py --student checkpoints/cardiac/student_best.pt
python eval_smap_msl.py --data smap_data --mode hybrid     # 0.7730

# Train proprio encoder (synthetic → real)
python train_proprio_6c.py `
    --hdf5-dir recon_data/synthetic_sprint9 `
    --k-ctx 16 --epochs 20 `
    --out-ckpt checkpoints/cwm/proprio_kctx16_sprint9.pt

# Fine-tune on real RECON data
python train_proprio_6c.py `
    --hdf5-dir recon_data/recon_release `
    --k-ctx 16 --epochs 10 `
    --init-ckpt checkpoints/cwm/proprio_kctx16_sprint9.pt `
    --out-ckpt checkpoints/cwm/proprio_kctx16_recon_ft.pt

# Train unified v2 pipeline
python train_nemo_wm_v2.py --mode decoder    # 5 min
python train_nemo_wm_v2.py --mode full       # 3 hrs
```

---

## 📄 Paper

**NeMo-WM: Neuromodulated World Models for Edge-Deployed Robot Perception**  
Taylor John · 2026  
*cs.RO + cs.LG · CC BY 4.0*

**arXiv:** Pending endorsement  
**GitHub:** [github.com/taylorjohn/nemo-wm](https://github.com/taylorjohn/nemo-wm)

### Key Contributions

1. Full computational dissociation between VLM and proprio pathways
2. 26K-param encoder outperforms ViT-G 1034M by +0.117 AUROC
3. ACh temporal integration confirmed superlinear (k=2→32)
4. Heading dominance 25-43:1 over velocity — timescale invariant
5. Flow matching policy SR=100% in 350 episodes, 6 seeds
6. Full pipeline 1.32ms on $500 edge hardware
7. Multi-domain: 6 anomaly detection domains, single encoder
8. Synthetic pre-training → real fine-tune: top1_acc 0.9798 → 0.9762+ (ep1, still training)
9. Stochastic encoder: KL divergence = biologically correct DA signal
10. Imagination viable: decoder MAE=0.013, z→obs reconstruction confirmed

---

## 📊 Training Progress (April 2026)

```
Sprint 9 synthetic pre-training:
  ep0=0.9137 → ep15=0.9666 → ep20=0.9798  ✅ FINAL

RECON fine-tune (10,995 real files, in progress):
  ep0=0.9650 → ep1=0.9762 → ep2-10 running...
  Expected ceiling: 0.980-0.988

Flow policy (position):
  SR=100% confirmed (n=350 episodes, 6 seeds, n_steps=10)
  Latency: 0.46ms full pipeline

Flow policy v2 (position + orientation):
  Training complete, eval pending
```

---

## 🏗️ Three Pillars

This repo contains all three CORTEX research pillars:

### NeMo-WM — Robot Navigation
Proprioceptive world model for outdoor navigation. Physics-grounded path integration. Full computational dissociation.

### CORTEX-PE — Perception Engine
Multi-domain anomaly detection across bearing, audio, visual, cardiac, and telemetry domains. Single encoder, no retraining.

### CORTEX-16 — Algorithmic Trading
Live trading on Alpaca paper API. 18-phase safeguard architecture. 99/99 pytest checks. Biological neuromodulation signals.

---

<div align="center">

**Hardware:** GMKtec EVO-X2 · AMD Ryzen AI MAX+ 395 · 128GB RAM · AMD NPU XINT8  
**Updated:** April 2026

</div>
