
# RECON Domain — Complete Timeline
## StudentEncoder Temporal Geometry Learning

---

## Phase 0 — Failed Attempts (pre-Sunday 29 March)

### Attempt 1 — Wrong Architecture Checkpoint
- **What:** Loaded existing bearing/MIMII checkpoint into StudentEncoder for RECON eval
- **Result:** `strict=False` silently loaded 0/N tensors — key mismatch
  - Existing checkpoints: `backbone.stem.*` / `backbone.block*.*` keys (32-D multi-task)
  - Production StudentEncoder: `features.*` / `proj.*` keys (128-D, ONNX/NPU format)
- **Failure mode:** All evaluations returned near-random AUROC (0.50) despite no error
- **Lesson:** Always verify parameter count after `strict=False` checkpoint loading

### Attempt 2 — Cross-Trajectory InfoNCE
- **What:** Trained InfoNCE with positives = different frames from same trajectory,
  negatives = frames from different trajectories
- **Loss:** Converged well (near zero)
- **Result:** AUROC 0.10 — catastrophically worse than random
- **Diagnosis:** Encoder learned trajectory identity (which outdoor scene),
  not temporal ordering (how far along the scene)
- **Lesson:** Cross-trajectory InfoNCE teaches scene identity, not temporal geometry

### Attempt 3 — Projector-Output AUROC Evaluation  
- **What:** Evaluated AUROC on projector output (SimCLR/MoCo style 256-D head)
- **Result:** AUROC collapsed to 0.50 regardless of encoder quality
- **Diagnosis:** SimCLR projector pushes all embeddings apart equally —
  anomaly scoring on projector output is meaningless
- **Lesson:** Always evaluate on encoder latents, never projector output

---

## Phase 1 — Correct Formulation (Sunday 29 March 2026)

### Root Cause Identified
- **Problem:** All prior failures stemmed from wrong triplet formulation
- **Solution:** Same-trajectory triplet InfoNCE with hard negatives
  - Anchor: frame t from trajectory i
  - Positive: frame t+k_near (k_near ∈ [1,5]) from SAME trajectory
  - Negative: frame t+k_far (k_far ∈ [15,40]) from SAME trajectory
  - Hard negatives share identical scene, lighting, outdoor environment
  - Only temporal offset differs — forces encoder to learn temporal geometry

### Dataset
- 11,836 trajectories from RECON release dataset
- HDF5 format: `images/rgb_left` (JPEG bytes), `jackal/position` [T,3] metres
- T varies 6–82 frames, typical T=70 at 4Hz
- Train/val: 80/20 split by trajectory

### Training — Phase 1 (train_student_temporal.py)
- **Objective:** Triplet InfoNCE on encoder latents (projector discarded after)
- **Architecture:** StudentEncoder (56K params, `features.*`/`proj.*` keys)
- **Config:** 30 epochs, 8,000 triplets/epoch, batch=128, τ=0.07, lr=3e-4
- **Hardware:** AMD Ryzen AI MAX+ 395 NUC, CPU, ~80s/epoch
- **Loss:** 0.916 → 0.020 (converged epoch 21)
- **Metric:** Triplet ordering (fraction where pos_sim > neg_sim, random=0.50)

**Phase 1 Results:**
| Epoch | Loss | Triplet Ordering |
|---|---|---|
| 1 | 0.916 | ~0.52 |
| 10 | ~0.15 | ~0.72 |
| 21 | 0.020 | — |
| 30 | 0.020 | **0.818** |

- **Checkpoint:** `checkpoints/recon_student/student_best.pt`

---

## Phase 2 — RoPETemporalHead (Sunday 29 March 2026)

### Architecture
- StudentEncoder frozen (weights from Phase 1)
- RoPETemporalHead: 128-D latent → 96-D temporal embedding
  - Rotary Position Encoding (RoPE) for temporal geometry
  - Trained on frozen latents (one-time encode: 9,469 trajectories, ~1,500s)

### Training
- **Config:** 30 epochs on cached latents, lr=1e-3, cosine decay
- **Metric:** Close/Far AUROC (pair mode): fraction where far pair scores higher
- **Init AUROC:** 0.8316 (random head on good Phase 1 encoder)
- **Best:** Epoch 11, AUROC **0.9499**

**Phase 2 Training Curve:**
| Epoch | AUROC (pair) |
|---|---|
| 0 (init) | 0.8316 |
| 5 | ~0.88 |
| 11 | **0.9499** ★ |
| 30 | 0.9480 (slight overfit) |

- **Checkpoints:**
  - `checkpoints/recon_student/student_best.pt` — Phase 1 encoder
  - `checkpoints/recon_student/rope_head_best.pt` — Phase 2 RoPE head

---

## Phase 3 — Held-Out Evaluation (Sunday 29 March 2026)

### Protocol
- 200 trajectories never seen during training
- Three metrics computed simultaneously:
  1. **Triplet ordering** — same as training metric (random=0.50)
  2. **Close/Far AUROC** — pair discrimination (random=0.50)
  3. **Displacement Spearman ρ** — correlation with GPS distance in metres

### Results (eval_recon_quasimetric.py)

| Metric | Value | Significance |
|---|---|---|
| Triplet ordering (held-out) | **0.9420** | +12.2pp over training floor |
| AUROC encoder (single mode) | **0.8630** | Encoder alone, no head |
| AUROC head (pair mode) | **0.9499** | Encoder + RoPEHead |
| Spearman ρ vs GPS | **+0.4694** | p=3.8e-110 |

### Per-k Analysis (separation of concerns confirmed)

| k offset | Encoder sim | Head AUROC | Disp (m) |
|---|---|---|---|
| k=1–5 | 0.9936 | 0.8785 | 0.66m |
| k=6–15 | 0.9858 | 0.7404 | 2.33m |
| k=16+ | 0.9832 | 0.6433 | 5.25m |

**Key finding:** Encoder similarity is nearly flat across all k (correct — it captures
scene identity). RoPETemporalHead carries all positional geometry (AUROC drops with k
as temporal gaps become harder to resolve).

---

## Phase 4 — Infrastructure (Sunday 29 March 2026)

### AVO-Inspired Self-Improvement (arXiv:2603.24517)
- `lineage.py` — lineage logger, plateau detection, intervention table
- `domain_spec.py` — RECON DomainSpec registered with 7-step intervention table
- `domain_scaffold.py` — scaffold generator + intervention executor
- `ADDING_NEW_DOMAIN.md` — 11-step walkthrough

### RECON Intervention Table (in domain_spec.py)
```python
interventions=[
    "triplet_infonce",       # ✅ Phase 1 done — ordering 0.818
    "rope_head",             # ✅ Phase 2 done — AUROC 0.9499
    "increase_k_far",        # push k_far_max 40 → 70 (harder negatives)
    "colour_jitter_strong",  # brightness 0.4, contrast 0.4
    "dinov2_distillation",   # distil DINOv2 → StudentEncoder
    "coordinate_encoder",    # feed (x,y) into encoder as extra channels
    "larger_capacity",       # CNN 64 → 128 → 256 channels
]
```

---

## Pending — RECON (prioritised)

### P1 — Navigator Inference Wiring (this week)
The Phase 1 + Phase 2 checkpoints are validated but NOT yet connected to the live
navigation inference path. The old inference code may still load a broken checkpoint
or use the wrong architecture.

**Task:**
1. Find `recon_navigator.py` (or equivalent inference script)
2. Update to load `checkpoints/recon_student/student_best.pt` (`features.*`/`proj.*` keys)
3. Optionally load `rope_head_best.pt` for temporal scoring
4. Smoke test: 1 trajectory, confirm latent shape (1, 128), cos_sim > 0.99 vs reference

```powershell
# Verify checkpoint loads correctly
python -c "
import torch
ckpt = torch.load(r'checkpoints\\recon_student\\student_best.pt', map_location='cpu', weights_only=True)
print(list(ckpt['model'].keys())[:5])   # should be features.0.weight etc
print(f'AUROC: {ckpt.get(\"auroc\", \"not stored\")}')
"
```

### P2 — MuRF + RECON Init Distillation (after current MuRF run eval)
Use RECON checkpoint as student initialisation for the next DINOv2 + MuRF run.
Gives the student geometric prior before industrial texture distillation.

```powershell
python train_dinov2_distill.py `
  --mvtec-data .\data\mvtec `
  --recon-data .\recon_data\recon_release `
  --encoder-ckpt .\checkpoints\recon_student\student_best.pt `
  --epochs 30 --murf
```

Expected gain over random-init MuRF: +0.007 to +0.025pp on MVTec mean AUROC.

### P3 — GPS Grounding Loss (next sprint)
Add `--gps-grounding` flag to `train_dinov2_distill.py`:
- For each RECON frame pair `(t, t+k)` in a batch, read `jackal/position`
- Penalise violations of `||z_t - z_{t+k}|| < ||z_t - z_{t+j}||` when k < j physically
- λ_gps = 0.05 (small — grounding regulariser, not dominant loss)
- No new data needed — `jackal/position` already in all RECON HDF5 files

Expected gain: +0.01 to +0.04pp on MVTec texture categories (carpet, grid primarily).

### P4 — Increase k_far Range (next sprint)
`domain_scaffold.py` will suggest `increase_k_far` as next intervention.
Push k_far_max from 40 → 70 for harder same-trajectory negatives.
Expected: +0.5 to +2pp on held-out triplet ordering.

### P5 — Phase 2 Extended Training (if needed)
Current RoPEHead trained for 30 epochs. Best at epoch 11 (AUROC 0.9499).
Slight overfit after epoch 11. Could benefit from:
- Dropout in RoPEHead
- Lower LR cosine schedule
- More latent cache diversity (different trajectory subset per epoch)

---

## Key Learnings

| Finding | Impact |
|---|---|
| `strict=False` silently loads 0 params on key mismatch | Always verify param count |
| Cross-trajectory InfoNCE = scene identity, not geometry | Same-trajectory triplets required |
| Projector AUROC = always 0.50 | Evaluate on encoder latents only |
| Encoder ≈ flat similarity across k | Correct — encoder is scene-identity |
| RoPEHead carries all temporal geometry | Clean separation of concerns |
| Spearman ρ=0.47 vs GPS | Encoder is physically grounded |

---

*Script:* `train_student_temporal.py` (Phase 1+2) | `eval_recon_quasimetric.py` (eval)  
*Checkpoints:* `checkpoints/recon_student/student_best.pt` + `rope_head_best.pt`  
*Lineage:* `lineage/recon.jsonl` | Domain: `recon` in `domain_spec.py`  
*Last updated:* Sunday 29 March 2026
