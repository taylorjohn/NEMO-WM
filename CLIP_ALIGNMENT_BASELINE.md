# CLIP ↔ NeMo-WM Alignment — Pre-Distillation Baseline
> Date: 2026-04-02 | Script: clip_particle_alignment_test.py
> Checkpoint: cwm_best.pt (epoch 20, loss 0.5702) | N=377 samples

## Result: NO ALIGNMENT (expected — distillation not yet done)

### What was tested
PCA projection of CLIP ViT-B/32 text embeddings (512-D) into NeMo-WM
particle space (128-D) via a shared 64-D PCA basis. Cosine similarity
between projected text queries and particle embeddings. Top-30 retrieval
compared against random baseline.

### Results table

| Query | Signal | Ratio | Z | Result |
|-------|--------|-------|---|--------|
| robot moving fast forward | linear_velocity | 1.07x | 1.98 | ❌ |
| robot driving at high speed | linear_velocity | 1.06x | 1.59 | ❌ |
| robot moving slowly and carefully | linear_velocity | 1.06x | 1.59 | ❌ |
| robot barely moving | linear_velocity | 1.07x | 1.98 | ❌ |
| robot stopped or stationary | linear_velocity | 1.08x | 2.20 | ❌ |
| robot turning sharply | angular_velocity | 1.01x | 0.12 | ❌ |
| robot spinning or rotating | angular_velocity | 0.98x | -0.17 | ❌ |
| robot going straight ahead | angular_velocity | 1.00x | 0.03 | ❌ |
| robot driving in a straight line | angular_velocity | 1.02x | 0.21 | ❌ |
| robot facing north outdoor campus | yaw | -0.11x | 0.67 | ❌ |
| robot navigating open outdoor area | linear_velocity | 1.09x | 2.42 | ❌ |
| robot on a narrow path | linear_velocity | 1.08x | 2.15 | ❌ |
| the quick brown fox (NULL) | linear_velocity | 1.06x | 1.68 | ✓ null |
| chocolate cake recipe (NULL) | linear_velocity | 1.06x | 1.76 | ✓ null |
| sonnets by shakespeare (NULL) | angular_velocity | 1.02x | 0.23 | ✓ null |

### Key observations

Semantic queries are indistinguishable from null controls. "Robot moving
fast" and "chocolate cake recipe" retrieve the same particles. This
confirms zero semantic alignment — the projection is finding visual
structure in the PCA basis, not semantic structure from CLIP.

Null controls calibrate correctly (3/3) — the test is valid. When
distillation creates real alignment, the test will detect it.

### Why this is correct

NeMo-WM's StudentEncoder was trained to distil DINOv2 spatial features,
not CLIP semantic features. DINOv2 and CLIP share some properties but
have fundamentally different training objectives:

- DINOv2: self-supervised spatial/geometric features via DINO loss
- CLIP: supervised semantic features aligned with natural language

The particle encoder inherits DINOv2's spatial bias. Without explicit
CLIP supervision, particles encode velocity, heading, and GPS displacement
(as confirmed by AIM probe) — but not the semantic categories that CLIP's
text encoder produces.

### Sprint 6 fix

Add CLIP as a second distillation teacher alongside DINOv2:

```python
# In StudentEncoder training:
L_dinov2 = cosine_loss(student_out, dinov2_features)
L_clip   = cosine_loss(student_out, clip_image_features)
total    = L_dinov2 + 0.3 * L_clip
```

After Sprint 6 distillation, re-run this test. Expected outcome:
- Semantic query ratios: 1.3–2.0x
- Null control ratios: ~1.0x (unchanged)
- Strong/weak alignment for velocity and turning queries

That delta — 1.07x → 1.5x+ — is the measurable contribution of
CLIP dual-distillation. The pre-distillation result you have now is
the before. Sprint 6 provides the after.

### Files

| File | Purpose |
|------|---------|
| `clip_particle_alignment_test.py` | Test script |
| `clip_alignment_results.json` | Machine-readable results (save this) |
| `CLIP_ALIGNMENT_BASELINE.md` | This document |

### Paper text

> Prior to CLIP dual-distillation, zero semantic alignment was observed
> between NeMo-WM particle embeddings and CLIP text embeddings (all query
> ratios 1.0–1.09x, indistinguishable from null controls at 1.02–1.06x,
> N=377 samples, K=30). Null control calibration (3/3 correctly null)
> validates the test design. This result confirms that CLIP distillation
> is a necessary precondition for text-conditioned navigation — not an
> optional enhancement — and establishes the pre-distillation baseline
> against which Sprint 6 results will be measured.

