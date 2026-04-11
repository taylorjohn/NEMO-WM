# NeMo-WM: Neuromodulated World Model for Robot Navigation

Biologically-grounded navigation world model trained entirely on a GMKtec EVO-X2
(AMD Ryzen AI MAX+ 395, 128GB RAM). No GPU. No cloud.

## Key Results

| Result | Value |
|---|---|
| Proprioceptive encoder AUROC (k_ctx=64) | **0.9999** |
| RECON fine-tune AUROC | 0.9972 |
| V-JEPA 2 ViT-G (1034M params) AUROC | 0.8833 |
| NeMo-WM params | **26,561** |
| Speedup vs V-JEPA 2-L | **1411x** |
| MVTec AD | 15/15 PASS (p95 mode) |
| CWRU Bearing | AUROC 1.000 |
| MIMII | AUROC 0.9313 |

NeMo-WM outperforms V-JEPA 2 ViT-G (+0.114 AUROC) with 40,000x fewer parameters.
Physics-grounded path integration outperforms visual scaling for temporal self-localisation.

## ACh Sweep (Temporal Context Window)

| k_ctx | Context | top1_acc | No-VLM AUROC |
|---|---|---|---|
| 2 | 1s | 0.925 | 0.925 |
| 4 | 2s | 0.961 | 0.961 |
| 8 | 4s | 0.977 | 0.977 |
| 16 | 8s | 0.9874 | 0.9972 |
| 32 | 16s | 0.9957 | 0.9997 |
| 64 | 32s | 1.0000 | 0.9999 |

Superlinear scaling -- broader temporal context strictly better for 4Hz outdoor navigation.
Biological parallel: acetylcholine modulates temporal integration window in hippocampus (Hasselmo 1999).

## Three Pillars

- **NeMo-WM**: Neuromodulated world model for robot navigation (this repo)
- **CORTEX-PE**: Multi-domain anomaly detection (RECON, MVTec, CWRU, MIMII, cardiac, SMAP)
- **CORTEX-16**: Algorithmic trading engine (Alpaca paper trading)

All share a biological neuromodulation philosophy: ACh, dopamine, cortisol, NE, eCB signals.

## Hardware

GMKtec EVO-X2, AMD Ryzen AI MAX+ 395, 128GB unified RAM, AMD NPU XINT8 (0.34ms inference).
Trained without a discrete GPU.

## Key Files

- 	rain_proprio_6c.py -- Sprint 6c proprioceptive encoder training
- eval_recon_auroc.py -- Dissociation eval (VLM + proprio)
- 
euro_vlm_gate.py -- Two-DA channel VLM gate (73/73 tests)
- cortisol_domain_adaptive.py -- Domain-adaptive cortisol signal
- un_vision_model.py -- Unified vision model benchmark launcher
- pusht_physics_registry.py -- Physics registry for manipulation tasks
- grasp_planner.py -- GRASP planner (arXiv:2602.00475, <10ms)

## Related Repos

- [strix-halo-vision-npu](https://github.com/taylorjohn/strix-halo-vision-npu) -- DINOv2/CLIP on AMD NPU
- [amd-npu-vlm-compat](https://github.com/taylorjohn/amd-npu-vlm-compat) -- VLM fixes for AMD NPU

## Paper

arXiv submission pending endorsement. Independent researcher.
Contact: johntaylorcreative@gmail.com

## License

MIT
