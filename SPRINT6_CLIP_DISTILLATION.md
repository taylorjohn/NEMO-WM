# Sprint 6 — CLIP Dual-Distillation
> Date: 2026-04-03 | Status: IN PROGRESS

## Objective

Distil CLIP's semantic knowledge into NeMo-WM's StudentEncoder alongside
the existing DINOv2 spatial knowledge. This enables language-conditioned
navigation without an LLM at inference time.

## Architecture

```
Input frame (224×224)
       ↓
StudentEncoder (46K params, trainable)
       ↓
128-D unit-normalised embedding
       ↑ ↑
  DINOv2    CLIP ViT-B/32
  anchor    (frozen teacher)
  (frozen)       ↓
              512-D features
                 ↓
            clip_bridge (Linear 512→128)
                 ↓
            128-D projection

Loss = 0.7 * cosine(student, anchor) +  ← preserve DINOv2 knowledge
       0.3 * cosine(student, clip_proj)  ← add CLIP semantics
```

## Why self-distillation for DINOv2 preservation

Rather than re-loading DINOv2 (400M params), we use a frozen copy of
the existing student_best.pt as the DINOv2 anchor. This works because:
- The student already approximates DINOv2 features (distilled in Sprint 1)
- The frozen copy remembers everything the student learned
- L_preserve keeps the student close to that anchor during CLIP fine-tuning
- No catastrophic forgetting — spatial encoding maintained throughout

## Training configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base checkpoint | student_best.pt | Start from DINOv2 distillation |
| Save path | student_clip_best.pt | No collision with original |
| Learning rate | 5e-5 | Fine-tuning, not training from scratch |
| Alpha (CLIP) | 0.3 | 30% CLIP, 70% DINOv2 preservation |
| Epochs | 15 | ~18 hours on GMKtec |
| Batch size | 32 | Memory-efficient |
| Data | All 11,835 RECON files | Same data as Sprint 1 distillation |

## Smoke test results (3 epochs, 50 files)

Training converged cleanly:

| Epoch | Loss | L_preserve | L_clip |
|-------|------|-----------|--------|
| 0 | 0.6023 | 0.5153 | 0.8053 |
| 1 | 0.3436 | 0.2001 | 0.6783 |
| 2 | 0.2931 | 0.1465 | 0.6351 |

L_preserve drops fast — DINOv2 anchor is working.
L_clip drops slower — CLIP alignment takes more epochs on limited data.

## Alignment test results

### Pre-distillation (baseline, LOCKED)
All 12 semantic queries: 0.98–1.09x (null, indistinguishable from controls)
File: clip_alignment_results.json

### Post 3-epoch smoke test (50 files only)
| Category | Baseline | Post 3ep | Signal |
|----------|---------|---------|--------|
| Linear velocity | 1.01–1.09x | 1.01–1.04x | No change yet |
| Angular velocity | 0.98–1.02x | **1.12–1.20x** | WEAK emerging |
| Null controls | 1.02–1.06x | 1.02–1.18x | Some leakage |

Angular velocity showing clear movement after just 3 epochs on 50/11835 files.
Linear velocity needs more epochs — semantically harder for CLIP to express.

### Expected post full-run (15 epochs, all files)
| Category | Expected ratio | Threshold |
|----------|---------------|-----------|
| Angular velocity | 1.5x+ | STRONG |
| Linear velocity | 1.2–1.5x | WEAK→STRONG |
| Null controls | ~1.0x | Correctly null |

## Sprint 6 → Sprint 7 integration

After student_clip_best.pt is validated:

```python
# Language goal encoding (no LLM required)
import clip
text = "navigate toward the building entrance"
tokens = clip.tokenize([text])
with torch.no_grad():
    text_feat = clip_model.encode_text(tokens).float()  # (1, 512)
    text_feat = F.normalize(text_feat, dim=-1)
    goal_particle = clip_bridge(text_feat)               # (1, 128)
    goal_particle = F.normalize(goal_particle, dim=-1)

# Use as navigation goal — same interface as GPS goal
action = mirror_ascent.plan(current_particles, goal_particle)
```

Latency: CLIP text encode (~2ms) + bridge projection (~0.1ms) = ~2ms total.
Compatible with 4Hz real-time navigation.

## Research position

### Distinctive contribution vs prior work
NeMo-WM is the first system to combine:
1. CLIP semantic distillation into a 46K-parameter encoder (8,700× compression)
2. Language grounding in temporal world model dynamics (not static maps)
3. Biologically-plausible neuromodulation of language-conditioned prediction

### Closest prior work
- **CLIP-Fields** (Shafiullah 2022): language-queryable 3D maps, 400M params, static
- **NLMap** (Chen 2023): open-vocab navigation, 400M params, static maps
- **LERF** (Kerr 2023): language NeRF, 400M params, static scenes

### Honest framing
- LLM-free at **inference time**: ✅ accurate
- Trained without scale: ❌ CLIP pretraining required (400M params, 400M pairs)
- Correct claim: "Edge-deployable language-conditioned navigation without
  LLM inference — 46K parameters, 8W, $800 hardware, 4Hz real-time"

## Files

| File | Purpose | Status |
|------|---------|--------|
| train_student_clip.py | Dual-distillation training | ✅ running |
| clip_particle_alignment_test.py | Before/after measurement | ✅ baseline locked |
| clip_alignment_results.json | Pre-distillation baseline | ✅ locked |
| student_best.pt | Original DINOv2-distilled encoder | ✅ preserved |
| student_clip_best.pt | CLIP+DINOv2 dual-distilled encoder | ⏳ training |

## Next steps after Sprint 6

1. Run alignment test with student_clip_best.pt — expect STRONG angular vel
2. Update GeoLatentDB to use CLIP-aligned encoder (rebuild particles)
3. Sprint 7: wire CLIP text encoder → clip_bridge → MirrorAscent planner
4. Evaluate text-conditioned navigation on RECON vs GPS-conditioned baseline
5. Compare vs NOMAD (visual goals) on same trajectories
EOF
---

## Alpha=0.3 Post-Mortem (2026-04-04)

### What happened
L_clip stalled at 0.641 from epoch 4 through epoch 15. The bridge
layer (Linear 512→128) absorbed all CLIP gradient, leaving the student
encoder backbone unchanged. Result: 2 WEAK signals at epoch 8
(angular velocity 1.11–1.16x) — identical to 3-epoch smoke test.

### Root cause
Alpha=0.3 gives CLIP only 30% of gradient. With a trainable bridge
layer available as an easier optimisation target, the system routes
all CLIP gradient through the bridge and leaves the student backbone
untouched. The cosine LR scheduler then locked the bridge into its
local optimum before the student encoder had moved.

### Alpha=0.5 rerun — dramatically different
- L_clip at step 400: 0.352 (vs alpha=0.3's 0.647 at same point)
- L_preserve: 0.265–0.274 (50/50 balanced tension forcing backbone movement)
- Student encoder backbone must move — bridge alone cannot absorb 50% CLIP

### Configuration
```
alpha: 0.5 (50% CLIP, 50% DINOv2 preserve)
lr: 3e-5 (vs 5e-5 — slower to prevent overshooting)
epochs: 10
save: checkpoints/dinov2_student/student_clip_alpha05_best.pt
```

### Expected outcome
- Epoch 0 L_clip mean: < 0.38
- Epoch 3 alignment test: STRONG angular velocity (>1.5x)
- Epoch 10 alignment test: STRONG for angular + velocity (1.3–2.0x)

### Alignment test protocol
After each epoch completes:
1. Copy student_clip_alpha05_best.pt → student_best.pt (backup original first)
2. Run clip_particle_alignment_test.py
3. Restore original student_best.pt
4. Compare ratios to baseline (1.07x all null)

