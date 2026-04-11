# NeMo-WM Status Report
**Last updated:** 2026-04-08 (end of day)
**Hardware:** GMKtec EVO-X2 · AMD Ryzen AI MAX+ 395 · 128GB RAM · NPU XINT8

---

## ACh Temporal Window Sweep — COMPLETE ✅

| k_ctx | Window | top1_acc | No-VLM AUROC | VLM-only | Gap | Neg dist |
|-------|--------|----------|-------------|---------|-----|---------|
| 2 | 0.5s | 0.769 | 0.925 | 0.915 | +0.010 | — |
| 4 | 1.0s | 0.850 | 0.961 | 0.884 | +0.077 | — |
| 8 | 2.0s | 0.931 | 0.977 | 0.898 | +0.079 | 0.805 |
| 16 | 4.0s | 0.992 | 0.9974 | 0.928 | +0.069 | 1.019 |
| **32** | **8.0s** | **0.996** | **0.9997** | **0.907** | **+0.093** | **1.001** |

Superlinear through k=32. Hard negatives past unit sphere at k=16 and k=32.
Biological parallel: Hasselmo 1999 ACh temporal precision — low-ACh = broad integration = slow outdoor navigation.

---

## V-JEPA 2 Scale Comparison (2026-04-08)

| Model | Params | Zero-shot AUROC | Notes |
|-------|--------|----------------|-------|
| V-JEPA 2 ViT-G | 1034M | 0.883 | Worse than ViT-L |
| NeMo-WM StudentEncoder | 46K | 0.889 | Distilled |
| V-JEPA 2 ViT-L | 326M | 0.907 | Best visual zero-shot |
| NeMo-WM proprio k=32 | **26K** | **0.9997** | **+0.117 over ViT-G** |

Visual ceiling ~0.91 regardless of scale. Physics-grounded PI dominates.
ProjectionHead (ViT-L, ep1): top1_acc=0.844, training overnight.

---

## Path Integration Ablation — Complete

| k_ctx | k_pos | Full | HD drop | Vel drop | HD:vel |
|-------|-------|------|---------|---------|--------|
| 4 | 1 | 0.992 | −0.104 | −0.002 | 43:1 |
| 4 | 4 | 0.958 | −0.228 | −0.009 | 25:1 |
| 8 | 1 | 0.9996 | −0.030 | −0.001 | 38:1 |
| 8 | 4 | 0.982 | −0.122 | −0.010 | 13:1 |
| 16 | 1 | **1.000** | −0.007 | 0.000 | **∞:1** |
| 16 | 4 | 0.999 | −0.062 | −0.007 | 9:1 |

Five invariants confirmed. HD:vel ratio narrows with k_ctx (43→25→13→9:1, ∞:1 at fine scale).

---

## Dissociation Results

| Condition | k_ctx=4 | k_ctx=8 | k_ctx=16 | k_ctx=32 |
|-----------|---------|---------|---------|---------|
| Full (VLM + proprio) | 0.962 | 0.979 | 0.998 | — |
| **No VLM (proprio only)** | 0.961 | 0.977 | 0.9974 | **0.9997** |
| No proprio (VLM only) | 0.884 | 0.898 | 0.928 | 0.907 |
| Synth zero-shot | — | — | 0.9697 | — |

---

## Sprint 9 Results

| Encoder | Data | No-VLM AUROC |
|---------|------|-------------|
| k_ctx=16 RECON | 10,995 real | 0.9974 |
| k_ctx=32 RECON | 10,995 real | **0.9997** |
| Synthetic zero-shot | 500 synth | 0.9697 |
| Synth→RECON fine-tune | synth + RECON | 0.9965 |

---


## Final Dissociation Eval — CONFIRMED (2026-04-09)

Checkpoint: `proprio_kctx32_best.pt` (ep11, top1_acc=0.9985)
Protocol: n=1000, hard negatives k>=32, no GPS, k_pos<=4

| Condition | AUROC | Neg dist | Separation |
|-----------|-------|---------|------------|
| Full (VLM + proprio) | 0.9997 | 0.5308 ± 0.1662 | +0.5127 |
| **No VLM (proprio only)** | **0.9997** | **1.0352 ± 0.3184** | **+1.0011** |
| No proprio (VLM only) | 0.8905 | 0.0134 ± 0.0140 | +0.0115 |

Key finding: Full == No-VLM == 0.9997 exactly.
VLM pathway contributes zero additional discriminative power.
Neg dist = 1.035 — hard negatives 3.5% past unit sphere.
Separation = +1.001 — exceeds unit sphere radius.

### V-JEPA 2 Head Results (trained)
| Model | top1_acc | Epoch | AUROC |
|-------|---------|-------|-------|
| ViT-L + head | 0.9294 | 8 | TBD |
| **ViT-G + head** | **0.9931** | **9** | **TBD (pending eval)** |


## V-JEPA 2 Final Results (ALL CONFIRMED 2026-04-09)

| System | Params | AUROC | Notes |
|--------|--------|-------|-------|
| V-JEPA 2 ViT-G zero-shot | 1034M | 0.883 | Scaling hurts — worse than ViT-L |
| V-JEPA 2 ViT-L zero-shot | 326M | 0.907 | Best visual zero-shot |
| V-JEPA 2 ViT-L + head | 326M+304K | 0.9294 | ep8, top1=0.9294 |
| **V-JEPA 2 ViT-G + head** | **1034M+403K** | **0.9557** | **ep9, top1=0.9931** |
| **NeMo-WM proprio k=32** | **26K** | **0.9997** | **ep11, neg_dist=1.035** |
| Fusion ViT-G + proprio | 1034M+26K | 0.9767 | Visual adds noise — fusion < proprio alone |

**Critical finding:** Fusion (0.9767) < Proprio alone (0.9997).
The ViT-G visual pathway actively hurts performance at the proprio ceiling.
V-JEPA 2 ViT-G with trained head (0.9557) is 0.044 BELOW 26K proprio encoder.
Physics-grounded path integration dominates visual at every scale.

## Active Checkpoints

| Checkpoint | Epoch | Metric |
|-----------|-------|--------|
| `checkpoints/cwm/cwm_multidomain_best.pt` | 18 | loss=0.1620 |
| `checkpoints/cwm/temporal_head_sprint3.pt` | 9 | top1_acc=0.939 |
| `checkpoints/cwm/proprio_6c_best.pt` | 33 | top1_acc=0.850 |
| `checkpoints/cwm/proprio_kctx2_best.pt` | 18 | top1_acc=0.769 |
| `checkpoints/cwm/proprio_kctx8_best.pt` | 19 | top1_acc=0.931 |
| `checkpoints/cwm/proprio_kctx16_best.pt` | 9 | top1_acc=0.992 |
| `checkpoints/cwm/proprio_kctx32_best.pt` | 2 | top1_acc=0.996 |
| `checkpoints/cwm/proprio_synth_pretrain.pt` | 9 | top1_acc=0.961 |
| `checkpoints/cwm/proprio_kctx16_finetuned.pt` | 7 | top1_acc=0.986 |
| `checkpoints/cwm/vjepa2_head_best.pt` | 1 | top1_acc=0.844 |

---

## Overnight Training (running)

| Job | Status | Expected finish |
|-----|--------|----------------|
| k_ctx=32 (15 ep) | ep2 done, ep3+ running | ~7am |
| V-JEPA 2 ViT-L head (10 ep) | ep1 done | ~7am |
| V-JEPA 2 ViT-G head (10 ep, 200 files) | ep0 running | ~7am |

---

## Paper

**v18 FINAL** — all experiments complete, arXiv-ready.

| Item | Status |
|------|--------|
| Abstract (k_ctx=32, 0.9997) | ✅ |
| ACh sweep table (k=2..32) | ✅ |
| PI ablation table (k=16 k_pos=1, ∞:1) | ✅ |
| V-JEPA 2 scale comparison | ✅ |
| Contribution 10 | ✅ |
| Synthetic zero-shot + efficiency | ✅ |
| All citations (27) | ✅ |
| github.com/taylorjohn public | ⚠️ |
| arXiv submit (cs.LG + cs.RO) | ⚠️ |

---

## CORTEX-16

- Exit storm root cause fixed: `not self.active_order_id` guard on exit condition
- Today: entries/exits fired, flat by close, −$0.11 paper
- 9:25am tomorrow: restart engine

---

## Goal-Reaching Navigation Benchmark (2026-04-09)

Latent-space navigation eval on RECON (200 episodes, gap 8-32s, 16 steps max, 50% dist reduction required).

| System | SR | PLR | DDR/step | Init dist | Step time |
|--------|-----|-----|---------|---------|---------|
| VLM only (StudentEncoder) | 5% | 14.36 | −0.003 | 0.030 | 0.22ms |
| **Proprio k=32 (NeMo-WM)** | **100%** | **0.514** | **+0.029** | **0.397** | **0.22ms** |

Planning AUROC (VLM dist reduction → success): 0.9958

Key findings:
- VLM embedding space is geometrically flat for RECON navigation (init dist=0.030)
- Proprio embedding is strongly navigable (init dist=0.397, 20× SR improvement)
- Identical step time (0.22ms) — proprio adds zero latency overhead
- Consistent with AUROC dissociation: VLM=0.8905 vs proprio=0.9997

Note: planner uses latent-space proxy (simulated nudge toward goal).
True ATE/RPE requires action-conditioned CWM predictor (Sprint B).
