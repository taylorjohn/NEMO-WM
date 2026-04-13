# Reproducing NeMo-WM Paper Results

Every claim in the paper maps to a command below. Results are deterministic on CPU; GPU results may vary by ±0.001.

## Setup

```bash
git clone https://github.com/taylorjohn/nemo-wm.git
cd nemo-wm
pip install -r requirements.txt
```

### RECON Dataset
Download the RECON outdoor navigation dataset (Shah et al., 2021):
```bash
# ~12GB, 11,835 HDF5 files, Berkeley campus robot trajectories at 4Hz
# Each file: 70-130 frames with images, GPS, IMU, commands
mkdir -p recon_data/recon_release
# Download from: https://sites.google.com/view/recon-robot/dataset
# Place HDF5 files in recon_data/recon_release/
```

Verify:
```bash
python train_temporal_contrastive.py --data recon_data/recon_release --probe-only
# Should print: Files found: 11835, T = 70 frames @ 4Hz = 17.5s
```

---

## Table 1: Proprioceptive Encoder ACh Sweep

The core result — broader temporal context improves self-localisation superlinearly.

### Train each k_ctx variant
```bash
# k_ctx=2 (~30 min CPU)
python train_proprio_6c.py --hdf5-dir recon_data/recon_release \
    --out-ckpt checkpoints/cwm/proprio_kctx2_best.pt --k-ctx 2 --epochs 20

# k_ctx=4 (~45 min CPU)
python train_proprio_6c.py --hdf5-dir recon_data/recon_release \
    --out-ckpt checkpoints/cwm/proprio_kctx4_best.pt --k-ctx 4 --epochs 20

# k_ctx=8 (~1 hr CPU)
python train_proprio_6c.py --hdf5-dir recon_data/recon_release \
    --out-ckpt checkpoints/cwm/proprio_kctx8_best.pt --k-ctx 8 --epochs 20

# k_ctx=16 (~2 hr CPU)
python train_proprio_6c.py --hdf5-dir recon_data/recon_release \
    --out-ckpt checkpoints/cwm/proprio_kctx16_best.pt --k-ctx 16 --epochs 20

# k_ctx=32 (~4 hr CPU)
python train_proprio_6c.py --hdf5-dir recon_data/recon_release \
    --out-ckpt checkpoints/cwm/proprio_kctx32_best.pt --k-ctx 32 --epochs 40
```

### Evaluate AUROC for each
```bash
python eval_recon_auroc.py \
    --proprio-ckpt checkpoints/cwm/proprio_kctx16_best.pt \
    --hdf5-dir recon_data/recon_release \
    --n-pairs 1000 --k-pos 4 --proprio-compare --proprio-no-gps
```

**Expected results:**

| k_ctx | top1_acc | No-VLM AUROC | Parameters |
|-------|----------|-------------|------------|
| 2     | 0.769    | 0.925       | 26,561     |
| 4     | 0.850    | 0.953       | 26,561     |
| 8     | 0.931    | 0.977       | 26,561     |
| 16    | 0.992    | 0.9974      | 26,561     |
| 32    | 0.999    | 0.9997      | 26,561     |

---

## Table 2: Visual Encoder Comparison

### Train CWM + TemporalHead
```bash
# Train CWM particle encoder (~3 hr)
python train_cwm_multidomain.py --hdf5-dir recon_data/recon_release \
    --out-ckpt checkpoints/cwm/cwm_multidomain_best.pt --epochs 20

# Train temporal head on top (~1 hr)
python train_temporal_contrastive.py --data recon_data/recon_release \
    --encoder checkpoints/maze_weak_sigreg_straight/cortex_student_phase2_final.pt \
    --k 7 --epochs 20 --max-files 500
```

### Evaluate visual AUROC
```bash
python eval_recon_auroc.py \
    --head-ckpt checkpoints/cwm/temporal_head_sprint3.pt \
    --cwm-ckpt checkpoints/cwm/cwm_multidomain_best.pt \
    --hdf5-dir recon_data/recon_release \
    --n-pairs 1000 --k-pos 4
```

**Expected results:**

| Encoder | Params | AUROC |
|---------|--------|-------|
| StudentEncoder (DINOv2-S distilled) | 46K | 0.8912 |
| StudentEncoder + TemporalHead k=7 | 96K | 0.8879 (negative) |
| ProprioEncoder k_ctx=16 (ours) | 26K | 0.9974 |

---

## Table 3: Double Dissociation (Aphasia Ablation)

The language/proprioception independence finding.

```bash
python eval_recon_auroc.py \
    --head-ckpt checkpoints/cwm/temporal_head_sprint3.pt \
    --cwm-ckpt checkpoints/cwm/cwm_multidomain_best.pt \
    --hdf5-dir recon_data/recon_release \
    --n-pairs 500 --k-pos 4 \
    --aphasia-compare \
    --proprio-ckpt checkpoints/cwm/proprio_kctx16_best.pt
```

**Expected results:**

| Pathway | Full Model | Aphasia (lang=0) | Δ |
|---------|-----------|-----------------|-----|
| CWM visual | 0.9542 | 0.5000 (chance) | −0.4542 |
| Proprio | 0.9974 | 0.9974 | 0.0000 |

---

## Table 4: Synthetic Pre-training Transfer

### Generate synthetic trajectories
```bash
# Original 8 scenarios (500 files, <1 sec)
python synthetic_trajectory_generator.py \
    --n-trajectories 500 --out-dir recon_data/synthetic_sprint9

# Outdoor hard negatives (500 files, <1 sec)
python gen_outdoor_hard_negatives.py \
    --n-trajectories 500 --out-dir recon_data/synthetic_sprint9_outdoor
```

### Pre-train on synthetic, fine-tune on RECON
```bash
# Pre-train on synthetic
python train_proprio_6c.py --hdf5-dir recon_data/synthetic_sprint9 \
    --out-ckpt checkpoints/cwm/proprio_synth_pretrain.pt --k-ctx 16 --epochs 10

# Fine-tune on real RECON
python train_proprio_6c.py --hdf5-dir recon_data/recon_release \
    --out-ckpt checkpoints/cwm/proprio_kctx16_finetuned.pt \
    --init-ckpt checkpoints/cwm/proprio_synth_pretrain.pt --k-ctx 16 --epochs 10
```

**Expected results:**

| Init | ep0 top1_acc | No-VLM AUROC |
|------|-------------|-------------|
| Random (cold start) | 0.954 | 0.9974 |
| Synthetic pre-train | 0.966 (+0.012) | ~0.994 |
| Synthetic zero-shot | — | 0.9697 |

---

## Table 5: Multi-Domain Anomaly Detection

Uses the same encoder architecture across 6+ domains.

```bash
# CWRU bearings
python eval_cwru.py --ckpt checkpoints/cwm/cwm_multidomain_best.pt

# MIMII industrial audio
python eval_mimii.py --ckpt checkpoints/cwm/cwm_multidomain_best.pt

# MVTec visual inspection
python eval_mvtec_patchcore.py --ckpt checkpoints/cwm/cwm_multidomain_best.pt
```

**Expected results:**

| Domain | AUROC | Encoder |
|--------|-------|---------|
| CWRU bearings | 1.0000 | CWM 56K |
| MIMII fan (k=32) | 0.9313 | CWM 56K |
| MVTec (mean) | 0.8923 | CWM 56K |
| PhysioNet cardiac | 0.7241 | CWM 56K |
| RECON navigation | 0.9542 | CWM 56K |

---

## Figure: Visual Dreaming

```bash
# 8-second imagination rollout with GPS retrieval
python nemo_dream.py \
    --hdf5-dir recon_data/recon_release \
    --mode four_row --n-steps 32 --out dream_8sec.png
```

Produces 3-row figure: actual frames / GPS-retrieved (imagined) / belief state vectors.
Belief drift: cosine=0.27 at step 32, confirming 8-second planning horizon.

---

## Figure: RECON Quasimetric Evaluation

```bash
python eval_recon_quasimetric.py \
    --head-ckpt checkpoints/recon_contrastive/temporal_head_k7_best.pt \
    --encoder checkpoints/maze_weak_sigreg_straight/cortex_student_phase2_final.pt \
    --data recon_data/recon_release
```

---

## Hardware Requirements

| Component | Minimum | Used in Paper |
|-----------|---------|---------------|
| CPU | Any x86-64, 8+ cores | AMD Ryzen AI MAX+ 395 |
| RAM | 16GB | 128GB |
| GPU | Not required | Not used (CPU only) |
| NPU | Optional (AMD XINT8) | AMD Ryzen AI NPU |
| Disk | 20GB (code + data) | 256GB NVMe |

All training and evaluation runs on CPU. AMD NPU is used only for production inference (0.34ms/frame StudentEncoder).

---

## Script Index

| Script | Purpose |
|--------|---------|
| `train_proprio_6c.py` | ProprioceptiveEncoder training (all k_ctx) |
| `train_temporal_contrastive.py` | Visual TemporalHead training |
| `train_cwm_multidomain.py` | CWM particle encoder training |
| `train_dual_head_clip_nce_nr.py` | SemanticHead + CLIPBridge with null rejection |
| `eval_recon_auroc.py` | RECON AUROC evaluation (visual + proprio) |
| `eval_recon_quasimetric.py` | Quasimetric triplet/AUROC evaluation |
| `eval_double_dissociation.py` | Aphasia / stopped-frame ablation |
| `synthetic_trajectory_generator.py` | Sprint 9 synthetic trajectories (8 scenarios) |
| `gen_outdoor_hard_negatives.py` | Outdoor hard negative trajectories (6 scenarios) |
| `nav_text_pairs_sprint9.py` | 100 navigation-adjacent text pairs |
| `nemo_dream.py` | Visual dreaming demo |
| `neuro_vlm_gate.py` | Biological neuromodulator + VLM gate |
| `imagination_rollout.py` | Sprint D belief rollout + planning |
| `cortisol_domain_adaptive.py` | Domain-adaptive cortisol signal |
| `sprint9_phase2_runner.py` | Synth v2 pipeline orchestrator |

---

## Citation

```bibtex
@article{taylor2026nemowm,
  title={NeMo-WM: Neuromodulated World Models for Edge-Deployed Robot Perception},
  author={Taylor, John},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT
