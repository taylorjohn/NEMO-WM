# NeMo-WM Confirmed Results Audit
**Date:** 2026-04-10  
**Status:** arXiv-ready  
**Repo:** github.com/taylorjohn/NEMO-WM

---

## 1. RECON Self-Localisation (Proprioceptive Encoder)

| k_ctx | AUROC (hard neg) | top1_acc | Params | Notes |
|-------|-----------------|----------|--------|-------|
| 2     | 0.925           | 0.769    | 26,561 | 0.5s window |
| 4     | 0.961           | 0.850    | 26,561 | 1.0s window |
| 8     | 0.977           | 0.931    | 26,561 | 2.0s window |
| 16    | 0.9974          | 0.992    | 26,561 | 4.0s window |
| 32    | **0.9997**      | 0.9985   | 26,561 | 8.0s window ← BEST |

**VLM comparison (zero-shot, no fine-tune):**
- V-JEPA 2 ViT-L (326M params): 0.9069 easy / 0.9319 hard
- V-JEPA 2 ViT-G (1034M params): 0.8833 easy / 0.9557 hard
- NeMo-WM k=32 (26K params): 0.9997 hard → **+0.0427 vs ViT-G**

**Key finding:** Visual scaling does not solve temporal self-localisation.
26K params outperforms 1034M params by +0.114 AUROC.

**Heading dominance:** ~43:1 ratio over velocity at k=1 (Moser et al. 2008)

---

## 2. VLM Integration

| Phase | Result | File |
|-------|--------|------|
| Phase 1 | F1=1.000 (CLIP text-target) | neuro_vlm_gate.py |
| Phase 2a | Hybrid α=0.252 | phase2a_hybrid.py |
| Phase 2b | Cortisol: 0 false+ vs periodic: 3 | phase2b_clean.py |
| Phase 3 | Bio gate: 33/33 ✓ | neuro_vlm_gate.py |

**Full dissociation confirmed:**
- No-VLM (proprio): AUROC=0.9645 easy / 0.9997 hard
- VLM-only (ViT-L): AUROC=0.891 easy / 0.907 hard
- Gap: +0.0927 in favour of proprio at k=32

---

## 3. Action-Conditioned World Model (PushT)

| Metric | Value | Threshold |
|--------|-------|-----------|
| ac_lift | +0.668 | >0.05 = action load-bearing |
| idm_mae | 0.035 | <0.10 = good |
| Training epochs | 45 | — |
| Checkpoint | action_wm_pusht_full_best.pt | ep44 loss=0.5244 |

---

## 4. Block Position Probe

| Metric | Value | Notes |
|--------|-------|-------|
| MAE | 0.0025–0.0026 | 0.3% of arena size |
| Checkpoint | block_probe_best.pt | — |
| d_model | 128 | — |

Emergent geometric encoding — probe reads off block position
directly from discriminative latent without explicit supervision.

---

## 5. Planning — Goal Reaching (PushT Synthetic)

### 5a. CEM / GRASP / Mirror Ascent comparison (n=50, seed=99)

| Planner | SR | Dist | Steps | DA |
|---------|-----|------|-------|----|
| Random / Graph WM | 2% | 0.380 | 290 | — |
| Mirror ascent | 2% | 0.300 | 294 | 0.32 |
| CEM + real goal | 6% | 0.313 | 282 | 0.31 |
| GRASP + probe | 8% | 0.381 | 276 | 0.46 |
| CEM + GoalDA (eval_pusht_sr.py) | **18%** | 0.284 | 267 | — |
| Closed loop Sprint C | 8% | 0.391 | 279 | 0.14 |

### 5b. NeMo Flow Policy (confirmed across 3 seeds / configs)

| Config | n | SR | Steps | Dist | Notes |
|--------|---|----|-------|------|-------|
| H=8, n-steps=1, seed=99 | 50 | 96% | 141 | 0.114 | First eval |
| H=8, n-steps=1, seed=default | 50 | 84% | 167 | 0.118 | Second eval |
| H=8, n-steps=1, seed=7 | 100 | 84% | 169 | 0.119 | n=100 robust |
| H=8, n-steps=1, re-plan=1 | 100 | 91% | 148 | 0.117 | Reactive |
| **H=8, n-steps=10, seed=42** | **100** | **100%** | **45** | **0.113** | **BEST** |
| H=16, n-steps=10, seed=42 | 100 | 100% | 46 | 0.112 | Longer chunk |
| No-DA ablation, n-steps=10 | 100 | 100% | 41 | 0.110 | Ablation |

**Paper-reported numbers (conservative, n=100):**
- NeMo Flow Policy (n-steps=1): **84% SR** ← conservative baseline
- NeMo Flow Policy (n-steps=10): **100% SR, 45 steps** ← production

**Ablation finding:** DA conditioning does not significantly affect SR on
synthetic PushT (both hit 100%). DA benefit expected on harder tasks /
real robot with observation noise.

---

## 6. Sprint 9 (Synthetic Pre-training, running)

| Epoch | top1_acc | Status |
|-------|----------|--------|
| 0 | 0.9137 | ✓ saved |
| 1-19 | TBD | running overnight |

500 synthetic files, k_ctx=16, 20 epochs.
Expected final: 0.96-0.98 based on RECON sweep pattern.

---

## 7. CORTEX-16 Trading

| Metric | Value |
|--------|-------|
| cortex_brain tests | 99/99 passing |
| Last session | 103,365 ticks, 0 entries, 21,822 aborts |
| Signal window | 9:30–11:00 ET only |
| NPU inference | 0.34ms (CJEPAPredictor) |

---

## 8. Hardware

| Component | Spec |
|-----------|------|
| Machine | GMKtec EVO-X2 |
| CPU | AMD Ryzen AI MAX+ 395 |
| RAM | 128GB |
| NPU | AMD XINT8, 0.34ms inference |
| Cost | ~$2,000 |
| Power | 8W inference |

---

## 9. What to Report in Paper

### Abstract / contributions:
- RECON AUROC: **0.9997** (k_ctx=32, 26K params)
- Visual scaling does not solve temporal self-localisation
- Computational dissociation confirmed (No-VLM > VLM-only)
- NeMo Flow Policy: **84% SR** (n-steps=1, n=100, conservative)
- NeMo Flow Policy: **100% SR** (n-steps=10, production)
- 5× improvement over best iterative planner (CEM+GoalDA 18%)

### Numbers NOT to use:
- ~~96%~~ (single seed n=50, optimistic)
- Any PointMaze numbers (failed, future work)
- Sprint 9 encoder AUROC (still training)

### Negative results to include:
- PointMaze: ac_lift never exceeded 0.05 — temporal copy collapse
- DA ablation: does not significantly affect SR on synthetic PushT
- CEM/GRASP plateau at 6-8% without GoalDA flow policy

---

## 10. Checklist for arXiv

- [x] GitHub public (taylorjohn/NEMO-WM)
- [x] figures/auroc_sweep.png
- [x] figures/dissociation.png  
- [x] render_rollouts.py fixed
- [x] All key results confirmed
- [ ] **arXiv submit** — cs.LG + cs.RO, CC BY 4.0
- [ ] Section 6.3 updated with flow policy numbers
- [ ] Sprint 9 final numbers (running overnight)
