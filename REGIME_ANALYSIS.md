# Sprint 8a — Regime Transition Analysis
**NeMo-WM Tab 1 (Random Encoder) vs Tab 2 (Real Encoder)**
**Date:** 2026-04-06

---

## Method

Extracted all `regime=EXPLOIT` and `regime=REOBSERVE` events from training
logs. Tab 1 = random encoder ablation. Tab 2 = real DINOv2-distilled encoder
(production checkpoint). Sampled every 500 steps (log-every=500).

---

## Tab 1 — Random Encoder EXPLOIT Timeline

| Epoch | Step | Duration | DA events |
|-------|------|----------|-----------|
| ep02 | s105,500–s113,500 | **14 steps** (7,000 steps sustained) | 1 (s108,000) |
| ep03 | s134,500 | 2 steps | 0 |
| ep03 | s142,500–s143,500 | 3 steps | 1 (s142,500) |
| ep06 | s255,000–s257,500 | 8 steps | 0 |
| ep08 | s324,500 | 1 step | 0 |
| ep08 | s332,000 | 1 step | 0 |

**Total EXPLOIT events: 29 sampled steps / ~768 total (~3.8%)**
**Total REOBSERVE: ~96.2%**

---

## Tab 2 — Real Encoder Regime Timeline

**0 EXPLOIT events across 158 sampled steps, ep0–ep29 (1.12M steps total)**
**100% REOBSERVE throughout all 30 epochs**

---

## Key Finding: Counterintuitive Regime Inversion

| | EXPLOIT | REOBSERVE |
|--|---------|-----------|
| Tab 1 (random encoder) | ~3.8% | ~96.2% |
| Tab 2 (real encoder) | **0%** | **100%** |

**Random encoder produces occasional EXPLOIT. Real encoder never does.**

This is counterintuitive — the random encoder appears "more confident" in
exploitation despite encoding no physical structure.

### Interpretation

**EXPLOIT** fires when the neuromodulator determines the predictor has
sufficient confidence to exploit its current policy rather than explore.
With a random encoder:
- The prediction targets are unstructured noise
- The predictor quickly finds statistical regularities (data distribution
  patterns, not physical structure)
- These trivial patterns produce locally high confidence → EXPLOIT
- But confidence collapses quickly → return to REOBSERVE
- Result: brief, episodic EXPLOIT clusters at ep2, ep3, ep6, ep8

With a real encoder:
- The prediction targets encode rich physical structure (velocity, heading,
  GPS — confirmed by AIM probe ***)
- The predictor always finds new structure to explore
- No trivial convergence → perpetually in REOBSERVE
- Result: 0 EXPLOIT across 1.12M training steps

**The real encoder's structured representations keep the predictor
permanently in exploratory mode. The random encoder's noise converges to
exploitable trivial patterns faster than it fails on them.**

---

## EXPLOIT Cluster Analysis

The largest cluster: **ep2 s105,500–s113,500 (14 consecutive log steps = 7,000 training steps).**

At ep2, the random encoder's GRU and predictor have seen ~70,000 steps.
The predictor found a statistical regularity in the data strong enough to
sustain 7,000 steps of EXPLOIT regime. This is the data distribution itself
(frame ordering, batch statistics) not physical navigation structure — the
AIM probe shows DA never reaches 0.003 and physical signals are null.

After ep3, EXPLOIT becomes increasingly rare (8 steps at ep6, 2 single
events at ep8). The predictor has exhausted the exploitable data statistics
and returns to permanent REOBSERVE. Loss is still compressing (0.94→0.65)
but through REOBSERVE-only gradient, not exploitation.

---

## Paper Contribution

This analysis provides a mechanistic distinction between the two encoders
that goes beyond "DA is zero" vs "DA is non-zero":

- **Random encoder**: episodic EXPLOIT from trivial statistical patterns,
  no sustained physical learning
- **Real encoder**: permanent REOBSERVE, perpetual novelty from structured
  physical representations, sustained learning across 30 epochs

The regime sequence is a leading indicator of representation quality:
if EXPLOIT appears early and sustained, the encoder is finding trivial
patterns. If EXPLOIT never appears, the encoder is generating perpetual
novelty — which is the correct behaviour for a physically-grounded world model.

---

## Figures for Paper

**Figure A:** Tab 1 regime timeline — bar chart showing EXPLOIT episodes
at ep2, ep3, ep6, ep8 against the REOBSERVE baseline

**Figure B:** Tab 1 vs Tab 2 EXPLOIT rate comparison — single bar chart
(3.8% vs 0%) with annotation explaining the inversion

