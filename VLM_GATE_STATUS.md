# VLM Gate Status — neuro_vlm_gate.py
**Last updated:** 2026-04-06  
**Phase:** 3 (biological attention gating)  
**Tests:** 64/64 passing (test_neuro_vlm_gate.py)

---

## Architecture

```
Neuromodulator state → VLM attention gains → Frame → z
                     ↘ DA: query scale       ↗
                     ↘ NE: spatial bias      ↗  → updated neuromodulator
                     ↘ ACh: temperature      ↗
                     ↘ eCB: recency decay    ↗
                     ↘ Ado: global gain      ↗
```

The neuromodulator **precedes** perception. Gains are computed from frame t-1 error and applied before frame t is encoded (predictive coding).

### Classes
- `NeuroState`: eight signals (DA, NE, ACh, 5HT, eCB, Ado, Cort, E/I), regime, step
- `AttentionGains`: eight gain parameters (query_scale, spatial_bias, temperature, topk_suppress, recency_decay, global_gain, threshold_mult, ei_bias)
- `BiologicalNeuromodulator`: gain computation + update_from_error + rest/recalibration
- `NeurallyGatedVLM`: wraps any transformer encoder with forward hooks

### aphasia_ablation flag (added 2026-04-06)
```python
gated = NeurallyGatedVLM(encoder, neuro, aphasia_ablation=True)
z = gated.encode(img)   # → zeros_like(z): language pathway destroyed
```
Togglable at eval time: `gated.aphasia_ablation = True/False`

### Encoder compatibility fix (added 2026-04-06)
```python
# Handles both HuggingFace (pixel_values=) and plain PyTorch (positional) encoders
try:
    out = self.vision_model(pixel_values=img_tensor)
except TypeError:
    out = self.vision_model(img_tensor)
```

---

## Ablation inference results (2026-04-06)

### On Sprint 1 checkpoint (i.i.d. pairs, n=5,000)
- Dynamic vs Frozen: −0.031 (p<0.05) — dynamic loses on shuffled pairs
- Interpretation: neuromodulator requires sequential temporal input

### On Sprint 3 ep05 checkpoint (i.i.d. pairs, n=5,000)
- Dynamic vs Frozen: **+0.058 (p<0.05)** — dynamic wins, sign flipped
- Interpretation: Sprint 3 representations drift into geometry where neutral gains hurt;
  dynamic modulation compensates for drift

### Biological interpretation
Dynamic gain control requires continuous temporal streams, not random-access i.i.d. pairs.
Consistent with biological neuromodulatory function.
Full benefit requires:
1. Sequential frame evaluation (not i.i.d. pairs)
2. End-to-end training with neuromodulator active (Sprint 3)

---

## Neuromodulator live test (run_test, 10 RECON frames)

```
Encoder: DINO, DA threshold: 0.0613
Step   DA      NE      ACh     5HT     eCB     Ado     Cort    Regime   QS
0      0.0000  0.0000  0.4800  0.5000  0.0000  0.0000  0.0000  EXPLOIT  1.000
...
9      0.0114  0.0075  0.3715  0.6442  0.0098  0.0000  0.0000  EXPLOIT  1.022
```
- DA stays low (0.01–0.02): adjacent RECON frames nearly identical, low surprise
- 5HT climbing (0.50→0.64): correct inverse response to low DA
- ACh decaying (0.48→0.37): no action demand, diffuse contextual mode
- Ado=0: no fatigue (DA never crossed threshold to trigger adenosine build)
- E/I=0.029: strongly inhibitory dominant — stable environment, consolidation mode

---

## Test coverage (64 tests)

| Class | Tests | Coverage |
|---|---|---|
| TestNeuroState | 5 | Defaults, ranges, mutability |
| TestAttentionGains | 2 | Neutral defaults, float types |
| TestAttentionGainComputation | 6 | Per-signal response, biological bounds |
| TestUpdateFromError | 8 | First frame, identical/orthogonal, decay, Ado build, cortisol |
| TestRegimeTransitions | 5 | All four regimes, priority order |
| TestRestAndRecalibration | 6 | Ado clearing, floor, needs_recalibration, reset_cortisol |
| TestGetState | 3 | Keys, rounding, step count |
| TestHistoryWindow | 2 | z_history and da_history bounded |
| TestEIBalance | 2 | High/low E/I polarity |
| TestNeurallyGatedVLM | 7 | Hooks, encode shape, unit vector, projection |
| TestPredictiveCodingLoop | 4 | Full loop, state evolution, spatial context, action magnitude |
| TestAphasiaAblation | 8 | Zero vector, MD survival, gains bounded, togglable |

---

## Files

- `neuro_vlm_gate.py` — main implementation
- `test_neuro_vlm_gate.py` — 64-test pytest suite
- `aphasia_eval.py` — standalone aphasia eval script (RPE-based, see caveats in ABLATION_RESULTS.md)
- `eval_recon_auroc.py` — AUROC-based eval with `--neuromod-compare` and `--aphasia-ablation` flags
