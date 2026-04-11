# CORTEX-PE v1.0 — Perception Engine

> **Stable Release — 2026-03-31**  
> Edge-deployed multi-domain anomaly detection. AMD Ryzen AI MAX+ 395 NUC. No GPU.

---

## What It Is

CORTEX-PE is a multi-domain anomaly detection system built on a single shared 56K-parameter encoder running at 0.34ms on an AMD Ryzen AI NPU. One encoder backbone serves six sensor domains — cardiac audio, spacecraft telemetry, industrial visual inspection, outdoor navigation, bearing vibration, and manipulation.

The core claim: a 56K-parameter student model distilled from DINOv2-small achieves competitive anomaly detection across domains that typically require separate large models. 3.26× better AUROC/MB than TinyGLASS (11M params).

---

## v1.0 Results

| Domain | AUROC | Method | Script |
|--------|-------|--------|--------|
| Cardiac audio | **0.8894** | k=32, 400 PhysioNet 2016 clips | `eval_cardiac_audio.py` |
| SMAP/MSL telemetry | **0.8427** | Hybrid PCA+drift + semi-supervised | `eval_smap_combined.py` |
| MVTec AD (standalone) | **0.7393** | Student k-NN, k=32 | `eval_mvtec.py` |
| MVTec AD (ensemble) | **0.8855** | DINOv2+student 512-D k-NN | `eval_mvtec_ensemble.py` |
| Efficiency | **3.26×** | AUROC/MB vs TinyGLASS | — |

All results run on AMD Ryzen AI MAX+ 395 NUC, CPU inference, no GPU.

---

## Architecture

```
Input (audio / telemetry / image patch)
    ↓
DINOv2-small teacher (22.1M params)
    ↓  MuRF multi-scale distillation
StudentEncoder (56K params, XINT8, 0.34ms NPU)
    ↓
128-D L2-normalised latent
    ↓
Per-domain PCA  →  k-NN anomaly score  →  AUROC
```

**StudentEncoder** is a 3-layer CNN (Conv→BN→ReLU ×3, AdaptiveAvgPool, Linear proj) with 56,592 parameters. Exported to XINT8 ONNX via AMD Quark, opset 17. Runs at 0.34ms per sample on the Ryzen AI NPU.

**MuRF distillation** uses multi-scale teacher features at scales [0.5, 1.0, 1.5], converging to a loss floor of 0.0007. This is the key training improvement over single-scale distillation.

**SubspaceAD** fits a PCA subspace on normal training samples per domain, scores test samples by reconstruction error. Simple, interpretable, no fine-tuning per domain.

**MVTec ensemble** concatenates student (128-D) + DINOv2-small teacher (384-D) into a 512-D L2-normalised vector, then runs k-NN. This is the production path for visual inspection.

---

## Domain-Specific Notes

### Cardiac Audio
- Dataset: PhysioNet Challenge 2016 (400 clips, balanced normal/abnormal)
- Preprocessing: log-mel spectrogram, 128 bands, 2s window
- k=32 nearest neighbours in PCA subspace
- AUROC 0.8894 — competitive with published CNN classifiers at 1000× fewer params

### SMAP/MSL Telemetry
- Dataset: NASA SMAP/MSL spacecraft telemetry (81 channels)
- Key challenge: 17-20% anomaly contamination in training data corrupts naive PCA
- Solution: hybrid approach
  - 56 channels: hybrid PCA + drift adaptive mode (handles slow drift anomalies)
  - 25 hard channels: semi-supervised LDA with n_labeled=20 canonical samples
- AUROC 0.8427 (71/81 channels ≥ 0.70)
- Hard failures (5): T-1, E-3, F-1, A-6, T-8 — genuine edge cases, not fixable without more labeled data
- Runtime: 4.7s for full eval

### MVTec AD
- Dataset: MVTec Anomaly Detection (15 categories, 5 textures + 10 objects)
- Two paths:
  - **Standalone student**: 0.7393 mean AUROC, 0.34ms/sample, NPU-deployable
  - **Ensemble**: 0.8855 mean AUROC, DINOv2+student concat, CPU only
- Hard categories: screw (0.5142), capsule (0.7284) — require patch-level scoring to improve
- Textures mean: 0.9613 | Objects mean: 0.8476

---

## Competitive Context

| System | Params | AUROC | Hardware |
|--------|--------|-------|----------|
| PatchCore (WideResNet50) | 68M | 0.992 | GPU |
| PaDiM (WideResNet50) | 25M | 0.956 | GPU |
| TinyGLASS (ResNet18 INT8) | 11M | 0.942 | Edge |
| **CORTEX-PE ensemble** | **22.2M teacher + 56K student** | **0.8855** | **NPU, no GPU** |
| **CORTEX-PE standalone** | **56K** | **0.7393** | **NPU, 0.34ms** |

The standalone path is the novelty claim: 56K parameters, 0.34ms NPU inference, across 4+ domains.

---

## Eval Commands

```powershell
conda activate ryzen-ai-1.7.0

# Cardiac
python eval_cardiac_audio.py `
    --student checkpoints\cardiac\student_best.pt `
    --data cardiac_data `
    --max-per-class 200

# SMAP/MSL (canonical — hybrid + semi-supervised)
python eval_smap_combined.py --data smap_data

# MVTec standalone
python eval_mvtec.py `
    --student checkpoints\dinov2_student\student_best.pt `
    --k 32

# MVTec ensemble (target: 0.8855)
python eval_mvtec_ensemble.py --data data\mvtec
```

---

## Checkpoints

| File | Domain | AUROC | Notes |
|------|--------|-------|-------|
| `checkpoints\dinov2_student\student_best.pt` | MVTec (primary) | 0.7393 standalone | MuRF distilled, XINT8-exportable |
| `checkpoints\cardiac\student_best.pt` | Cardiac | 0.8894 | Fine-tuned on PhysioNet 2016 |

---

## NPU Export

```powershell
# Export to XINT8 ONNX for Ryzen AI NPU
python export_onnx.py `
    --checkpoint checkpoints\dinov2_student\student_best.pt `
    --output cortex_encoder_int8.onnx

# Verify NPU execution
python verify_npu.py

# Sustained inference stress test
python npu_stress_test.py
```

See `amd_npu_layernorm_post.md` for the LayerNorm workaround required for AMD NPU XINT8 export (BF16 has operator gaps; opset 17 + manual LayerNorm fusion is the fix).

---

## Known Limitations (v1.0)

- **5 SMAP channels unresolved**: T-1, E-3, F-1, A-6, T-8. These have ambiguous anomaly definitions in the labeled set. More canonical labels would help.
- **MVTec screw/capsule**: Global image-level scoring cannot resolve sub-pixel surface defects. Patch-level scoring (train with `--patch`) targets 0.88-0.92 but is slower.
- **CWRU bearing and OGBench-Cube**: Domains scaffolded but not yet evaluated with v1.0 encoder. Carried to CORTEX-WM Sprint 3.
- **StudentEncoder trained on MVTec distribution**: Cardiac and SMAP use the same encoder weights without domain-specific fine-tuning. Domain-specific fine-tuning (cardiac checkpoint) improves results significantly.

---

## Hardware

```
NUC:  AMD Ryzen AI MAX+ 395
      Python 3.12, conda ryzen-ai-1.7.0
      torch 2.10.0+cpu, numpy 1.26.4
      AMD Quark (XINT8 quantisation)
      Inference: 0.34ms/sample on NPU
```

---

## What's Next (CORTEX-WM)

CORTEX-PE v1.0 is the perception backbone for CORTEX World Model. The same StudentEncoder (frozen, XINT8) feeds particle-based latent dynamics in Sprint 1+ of CWM training. SMAP, cardiac, and MVTec anomaly detection continue to run in production during CWM development.

---

*CORTEX-PE v1.0 · 2026-03-31 · AMD Ryzen AI MAX+ 395 NUC · No GPU*
