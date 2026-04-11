# NeMo-WM: Neuromodulated World Model for Robot Navigation

[![arXiv](https://img.shields.io/badge/arXiv-pending-b31b1b.svg)](https://github.com/taylorjohn/NEMO-WM)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hardware](https://img.shields.io/badge/Hardware-AMD_Ryzen_AI_MAX%2B_395-ED1C24.svg)](https://www.amd.com/)
[![NPU](https://img.shields.io/badge/NPU-XINT8_0.34ms-00A86B.svg)](https://github.com/taylorjohn/strix-halo-vision-npu)
[![No GPU](https://img.shields.io/badge/GPU-None-lightgrey.svg)]()

Biologically-grounded navigation world model trained entirely on a GMKtec EVO-X2
(AMD Ryzen AI MAX+ 395, 128GB unified RAM). **No GPU. No cloud.**

Physics-grounded path integration outperforms visual scaling for temporal
self-localisation. A 26K parameter proprioceptive encoder beats V-JEPA 2 ViT-G
(1B parameters) by **+0.114 AUROC** on outdoor robot navigation benchmarks.

---

## Results

### RECON Navigation Benchmark

| Method | Params | No-VLM AUROC | VLM-only AUROC |
|---|---|---|---|
| **NeMo-WM proprio (k=64)** | **26K** | **0.9999** | -- |
| NeMo-WM proprio (k=32) | 26K | 0.9997 | -- |
| NeMo-WM proprio (k=16) | 26K | 0.9972 | -- |
| V-JEPA 2 ViT-G (zero-shot) | 1034M | -- | 0.8833 |
| V-JEPA 2 ViT-L (zero-shot) | 326M | -- | 0.9069 |
| NeMo-WM full (VLM + proprio) | 70K | 0.9998 | -- |

**Key finding:** Visual scaling does not solve temporal self-localisation.
Physics-grounded path integration (26K params) outperforms 1B-parameter vision models.

### Industrial Anomaly Detection (CORTEX-PE)

| Dataset | AUROC | Status |
|---|---|---|
| CWRU Bearing (all k) | **1.000** | PASS |
| MIMII Industrial Sound | 0.9313 | PASS |
| MVTec AD (p95 mode) | **15/15** | PASS |
| Cardiac PhysioNet | 0.7730 | PASS |
| SMAP/MSL | 0.7730 | PASS |
| RECON navigation | 0.9075 | PASS |

---

## ACh Sweep -- Temporal Context Window

Acetylcholine (ACh) in the biological system modulates the temporal integration
window of hippocampal place cells (Hasselmo 1999). NeMo-WM uses k_ctx as the
computational analogue: broader context = higher ACh = better path integration.

| k_ctx | Context | top1_acc | No-VLM AUROC | Notes |
|---|---|---|---|---|
| 2 | 1s | 0.925 | 0.925 | |
| 4 | 2s | 0.961 | 0.961 | |
| 8 | 4s | 0.977 | 0.977 | |
| 16 | 8s | 0.9874 | 0.9972 | RECON fine-tune |
| 32 | 16s | 0.9957 | 0.9997 | neg dist > unit sphere |
| **64** | **32s** | **1.0000** | **0.9999** | **saturation point** |

Superlinear scaling confirmed through k=64. No plateau observed at k=32.
Saturation range: 16-32 seconds of temporal context for 4Hz outdoor navigation.

---

## Robustness Ablations (k=64 vs k=16)

| Ablation | k=16 AUROC | k=64 AUROC | Delta |
|---|---|---|---|
| Baseline | 0.9721 | 0.9873 | +0.015 |
| Drop heading | 0.9026 | 0.9772 | +0.075 |
| Corrupt 10% frames | 0.5721 | **0.9812** | **+0.409** |
| Corrupt 30% frames | 0.5376 | **0.9599** | **+0.422** |
| Speed 2x | 0.9660 | 0.9876 | +0.022 |
| Velocity noise | 0.9722 | 0.9871 | +0.015 |

**k=64 is fault-tolerant.** 10% frame dropout is catastrophic at k=16 (0.57 AUROC)
but barely affects k=64 (0.98 AUROC). 32-second context buffers corruption.

---

## Dissociation Eval

Full computational dissociation confirms proprioceptive path integration is
independent of the visual language pathway:

```
[Full (VLM + proprio)]   AUROC = 0.9998
[No VLM (proprio only)]  AUROC = 0.9998   <-- dissociation confirmed
[VLM only]               AUROC = 0.9143
[Aphasia (VLM zeroed)]   AUROC = 0.5000   <-- chance without proprio
```

The proprioceptive encoder alone matches the full system. The visual pathway
contributes zero additional information for temporal self-localisation.
This mirrors the double dissociation in rodent hippocampus (Moser et al. 2008).

---

## Architecture

```
Observation (vel, ang, heading, contact, delta_h) x k_ctx frames
    |
    v
Sinusoidal PE --> Temporal ProprioEncoder (26K params)
    - d_per_frame=8, d_hidden=128, d_model=64
    - k_ctx frames, attention pooling
    - NO GPS
    |
    v
Contrastive head (InfoNCE, temp=0.05)
    - Positive: k<=4 steps apart (same-location pairs)
    - Negative: hard (same-file, k>=32 steps apart)
    |
    v
AUROC on held-out pairs
```

**Neuromodulators:**
- ACh: temporal context window (k_ctx)
- Dopamine: reward prediction error, trading signal
- Cortisol: domain-adaptive sensitivity (sensitivity=0.10)
- NE: scale factor 1.22 on novel domain entry
- eCB: retrograde suppression 0.82

---

## Heading Dominance (Timescale-Invariant)

Heading direction dominates velocity for path integration at all timescales:

| k_ctx | HD lesion effect | Velocity effect | Ratio |
|---|---|---|---|
| k=4 | 0.249 | 0.010 | **25:1** |
| k=1 | 0.354 | 0.008 | **43:1** |

Mirrors Moser et al. 2008 grid cell lesion experiments. Heading is the
primary signal for spatial self-localisation regardless of temporal scale.

---

## Hardware & Inference Speed

| Component | Latency | Hardware |
|---|---|---|
| ProprioEncoder (26K) | **1.31ms** | CPU (XINT8: 0.34ms NPU) |
| DINOv2-S/14 (21M) | 53.76ms CPU | **0.86ms NPU (XINT8)** |
| CLIP ViT-L/14 (428M) | 341.8ms CPU | ~5.7ms NPU (est.) |
| V-JEPA 2-L (326M) | **1849ms** | no NPU path |

**NeMo-WM is 1,411x faster than V-JEPA 2-L on identical hardware.**

Hardware: GMKtec EVO-X2, AMD Ryzen AI MAX+ 395, 128GB unified RAM.
NPU: AMD XINT8 via Quark quantization + VitisAI Execution Provider.

See [strix-halo-vision-npu](https://github.com/taylorjohn/strix-halo-vision-npu)
for the XINT8 quantization pipeline.

---

## Three Pillars

All three systems share the biological neuromodulation philosophy:

| System | Domain | Status |
|---|---|---|
| **NeMo-WM** | Robot navigation world model | Active research |
| **CORTEX-PE** | Multi-domain anomaly detection | v16.17+ production |
| **CORTEX-16** | Algorithmic trading (Alpaca) | Paper trading |

Shared signals: ACh (temporal context), DA (reward prediction), cortisol (domain shift),
NE (novelty response), eCB (retrograde gating).

---

## Key Files

| File | Description |
|---|---|
| `train_proprio_6c.py` | Sprint 6c temporal encoder training |
| `eval_recon_auroc.py` | Full dissociation eval (VLM + proprio + aphasia) |
| `neuro_vlm_gate.py` | Two-DA channel VLM gate (73/73 tests) |
| `cortisol_domain_adaptive.py` | Domain-adaptive cortisol signal |
| `eval_proprio_robustness.py` | 9-ablation robustness suite |
| `eval_imagination_rollout.py` | Imagination horizon eval (2 steps) |
| `run_vision_model.py` | Unified vision model benchmark launcher |
| `pusht_physics_registry.py` | Physics registry for manipulation |
| `grasp_planner.py` | GRASP planner arXiv:2602.00475 (<10ms) |
| `benchmark_vlm_npu.py` | Before/after NPU benchmark table |

---

## Negative Results

Documented explicitly per scientific practice:

- **MuRF distillation** at [0.75, 1.0, 1.5] scales: negative vs no-MuRF baseline
- **V-JEPA 2 ViT-G** (1034M): 0.8833 AUROC, worse than ViT-L 0.9069 -- scale hurts
- **Flow policy (H=8)**: 8% SR, corner collapse -- episodes too short for flow matching
- **Flow policy (H=2)**: 13% SR vs 61% scripted -- degenerate on short-horizon tasks
- **Place cell sampling (50 files)**: 7.16x enrichment was a sampling artifact

---

## Paper

**NeMo-WM: Neuromodulated World Model for Robot Navigation**
John Taylor (independent researcher)

arXiv submission in progress (cs.LG, cs.RO).
Endorsement received from Dhruv Shah (UC Berkeley).

Key references: Hasselmo 1999 (ACh), Moser et al. 2008 (grid cells),
O'Keefe 1971 (place cells), Taube 1998 (head direction),
McNaughton 2006 (path integration), DreamerV3, SIMPLE, GRASP, Seoul WM.

---

## Related Repos

- [strix-halo-vision-npu](https://github.com/taylorjohn/strix-halo-vision-npu)
  -- DINOv2/CLIP XINT8 pipeline for AMD Ryzen AI MAX+ NPU
- [amd-npu-vlm-compat](https://github.com/taylorjohn/amd-npu-vlm-compat)
  -- LLaVA, SigLIP, V-JEPA 2 fixes for AMD NPU

---

## Citation

```bibtex
@article{taylor2026nemowm,
  title={NeMo-WM: Neuromodulated World Model for Robot Navigation},
  author={Taylor, John},
  journal={arXiv preprint},
  year={2026}
}
```

---

## License

MIT. Contact: johntaylorcreative@gmail.com
