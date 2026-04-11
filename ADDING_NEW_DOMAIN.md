# Adding a New Domain to CORTEX-PE
**Version:** v16.17 | **Date:** Saturday 28 March 2026

This walkthrough covers everything needed to bring a new sensing domain into
CORTEX-PE — from raw data to a production checkpoint with automatic improvement
tracking. Follow the steps in order. Each step has a verification command so
you know it worked before moving on.

---

## Overview

A CORTEX-PE domain consists of four things:

```
1. DomainSpec     — describes the task, data, head, loss, and improvement plan
2. Training script — train_{domain}.py, generated or written by hand
3. Eval script    — eval_{domain}.py, measures the primary metric on held-out data
4. Lineage file   — lineage/{domain}.jsonl, records every run automatically
```

The improvement loop is:

```
Define spec → Scaffold scripts → Run baseline → Commit to lineage
    → Plateau? → Execute next intervention → Repeat until target met
```

---

## Prerequisites

All files below should be in your CORTEX directory on the NUC:

```
CORTEX/
├── lineage.py          — lineage logger
├── domain_spec.py      — domain registry
├── domain_scaffold.py  — scaffold generator + intervention executor
├── train_student_temporal.py   — reference image training pipeline
└── eval_recon_quasimetric.py   — reference eval script
```

**Verify:**
```powershell
python domain_spec.py list
```
Expected output: 5 registered domains (recon, bearing, cardiac, smap, mimii).

---

## Step 1 — Understand Your Data

Before writing any code, answer these four questions. Everything else follows
from them.

| Question | Example answer |
|---|---|
| What is the input modality? | RGB images at 4Hz from a robot camera |
| What format is the data on disk? | HDF5 files, one per trajectory |
| What does a positive example look like? | Frames within 1 metre of each other |
| What metric defines success? | AUROC > 0.90 on close/far pair discrimination |

Also note any **data format gotchas** now — they will bite you later:

- RECON: `images/rgb_left` stores JPEG bytes as fixed-length HDF5 strings →
  requires `PIL.Image.open(io.BytesIO(bytes(raw)))`, not direct array read
- SMAP: training split has 17–20% anomaly contamination → need robust PCA cleaning
- MIMII: audio stored at 16kHz WAV, must resample for BEATs teacher

Write these down in the `data_notes` field of the spec. Future you will thank
present you.

---

## Step 2 — Define the DomainSpec

Open `domain_spec.py` and add your spec at the bottom of the file, above the
`if __name__ == "__main__"` block.

### 2a. Print the template

```powershell
python domain_spec.py template
```

This prints a filled-in skeleton. Copy it into `domain_spec.py`.

### 2b. Fill in the fields

```python
register(DomainSpec(
    name="my_domain",                    # short key, alphanumeric + underscore
    description="Detect X from Y data.", # one sentence
    modality="image",                    # "image" | "audio" | "telemetry" | "tabular"
    data_path=r".\data\my_domain",       # path on NUC relative to CORTEX dir
    data_format="hdf5",                  # "hdf5" | "wav" | "csv" | "npy" | "folder_of_images"
    data_notes="Describe key layout, decode gotchas here.",
    primary_metric="auroc",              # the number that defines improvement
    primary_metric_target=0.90,          # stop improving above this
    teacher=None,                        # None | "dinov2_small" | "beats" | "wav2vec2"
    head="ProjectionHead",               # see valid heads in domain_spec.py
    loss="triplet_infonce",              # see valid losses in domain_spec.py
    eval_script="eval_my_domain.py",     # name of the eval script
    checkpoint_dir=r"checkpoints\my_domain",
    baseline_result=None,                # fill in after first run
    interventions=[
        "baseline_run",                  # always first — no changes
        "increase_k_far",                # try if plateau after baseline
        "colour_jitter_strong",          # try next
        "dinov2_distillation",           # try if still below target
    ],
    notes="Any extra context, hardware requirements, known issues.",
    extra={},                            # domain-specific kwargs passed to scaffold
))
```

**Choosing the right head and loss:**

| Task | Head | Loss |
|---|---|---|
| Temporal ordering (navigation) | `RoPETemporalHead` | `triplet_infonce` |
| Anomaly detection (image/audio) | `SubspaceAD` or `ProjectionHead` | `distillation` or `subspace` |
| Anomaly detection (telemetry) | `SubspaceAD` | `subspace` |
| Classification | `ClassifierHead` | `crossentropy` |

**Choosing interventions:**

Order them from cheapest to most expensive. The system tries them in order
when it detects a plateau. Rules of thumb:

1. Baseline run (always first — establishes the floor)
2. Cheap data changes (k ranges, window sizes, augmentation strength)
3. Architecture changes (larger capacity, different head)
4. Teacher distillation (most expensive — run last)

### 2c. Validate the spec

```powershell
python domain_spec.py validate --domain my_domain
```

Expected: `✅ my_domain: OK`

If you see errors, fix the field values until validation passes before
continuing. Common mistakes: invalid modality string, empty interventions list,
non-alphanumeric name.

### 2d. Verify it appears in the registry

```powershell
python domain_spec.py list
```

Your domain should appear in the list with `🔄` status (no baseline yet).

```powershell
python domain_spec.py show --domain my_domain
```

Review the full spec summary. Check data_path and checkpoint_dir are correct.

---

## Step 3 — Generate the Training and Eval Scripts

```powershell
python domain_scaffold.py --domain my_domain --scaffold
```

This creates two files:
- `train_my_domain.py` — full training pipeline based on your modality
- `eval_my_domain.py` — placeholder eval script with TODO sections

**Expected output:**
```
Generated: .\train_my_domain.py
Generated: .\eval_my_domain.py
```

If the files already exist, add `--overwrite` to regenerate them.

### What the generated training script does

For **image** and **audio** domains: imports `StudentEncoder`,
`train_phase1()`, and `train_phase2()` from `train_student_temporal.py`, wires
in your data path, and calls `Lineage.commit()` at the end of each run.

For **telemetry** and **tabular** domains: scaffolds a `SubspaceAD` pipeline
with sliding windows and sklearn PCA. You only need to implement `load_data()`.

### 3a. Implement the data loader

Open `train_my_domain.py` and find the `# TODO` comment near the data loading
section. This is the only part you must write by hand — everything else is
generated.

**Image/HDF5 example:**
```python
def _load_hdf(path: Path) -> dict | None:
    try:
        import h5py
        with h5py.File(path, "r") as f:
            imgs = f["images/rgb_left"][:]   # JPEG bytes
            pos  = f["jackal/position"][:]   # [T, 3] metres
        return {"imgs": imgs, "pos": pos[:, :2]}
    except Exception:
        return None
```

**Telemetry/CSV example:**
```python
def load_data(data_path: str):
    import pandas as pd
    df = pd.read_csv(data_path)
    X_train = df[df["split"] == "train"].drop(columns=["label"]).values
    X_test  = df[df["split"] == "test"].drop(columns=["label"]).values
    y_test  = df[df["split"] == "test"]["label"].values
    return X_train, X_test, y_test
```

### 3b. Probe the data

Always run in probe mode before any training:

```powershell
python train_my_domain.py --data .\data\my_domain --probe-only
```

Expected output for image domain:
```
Found N HDF5 files
Checking first 3 train files...
  OK: file_001.hdf5  T=70  pos_shape=(70, 2)  img_size=(640, 480)
  OK: file_002.hdf5  T=82  ...
StudentEncoder  params=56,592  output_dim=128
Forward pass OK  output=(1, 128)  norm=1.0000
✅ Probe passed — ready to train
```

If probe fails, fix the data loader before continuing. Do not skip this step.

---

## Step 4 — Implement the Eval Script

Open `eval_my_domain.py`. It has `# TODO` sections that mirror the structure
of `eval_recon_quasimetric.py`. Implement:

1. **Data loading** — load held-out data (not seen during training)
2. **Encoder loading** — `StudentEncoder` + `encoder.load_state_dict(ckpt["model"])`
3. **Metric computation** — compute `primary_metric` on held-out set
4. **Summary print** — show result vs target

**Minimal working eval (image domain):**
```python
def main():
    # ... argparse ...
    encoder = StudentEncoder()
    ckpt = torch.load(args.checkpoint + "/student_best.pt",
                      map_location="cpu", weights_only=True)
    encoder.load_state_dict(ckpt["model"])
    encoder.eval()

    # Load held-out files, encode, compute AUROC
    trajs = load_trajectories(held_out_files)
    trajs = encode_trajectories(trajs, encoder, device)
    auc, _, _ = close_far_auroc(trajs, n_pairs=2000, seed=999)
    print(f"AUROC = {auc:.4f}  (target > {TARGET})")
```

**Verify the eval script runs** (will fail on NotImplementedError until
implemented, but should at least parse args without crashing):
```powershell
python eval_my_domain.py --data .\data\my_domain --help
```

---

## Step 5 — Run the Smoke Test

Before committing to a full training run, run a small smoke test. This catches
data pipeline errors, shape mismatches, and import failures within a few
minutes.

```powershell
python train_my_domain.py `
  --data .\data\my_domain `
  --max-files 5 `
  --epochs 3 `
  --n-pairs 500
```

**What to look for:**

| Signal | Meaning |
|---|---|
| Loss starts near `log(batch_size)` ≈ 4.85 | InfoNCE baseline — correct |
| Loss drops in epochs 1–3 | Gradients flowing — correct |
| Loss stays exactly flat | Data pipeline broken — check loader |
| `ValueError: low >= high` | Short trajectories — check k-sampling guard |
| Shape mismatch error | Check data dimensions vs encoder input |

If the smoke test completes without error and loss drops, proceed to full
training.

---

## Step 6 — Run Full Phase 1 Training

```powershell
python train_my_domain.py `
  --data .\data\my_domain `
  --epochs 30 `
  --n-pairs 8000 `
  --val-eval-files 100 `
  --eval-every 5 `
  --run-id my_domain_phase1_v1 `
  --notes "Phase 1 baseline, triplet InfoNCE from scratch"
```

The script automatically commits to lineage at the end. Watch for:

- **Loss < 3.0 by epoch 5** — encoder learning temporal structure
- **Loss < 1.0 by epoch 15** — strong convergence
- **Metric improving at eval checkpoints (epochs 5, 10, 15, 20, 25, 30)**

If loss stalls above 3.0 after epoch 10, stop and check:
- Are close/far pair stats balanced? (Expect close_disp ≈ 0.5–1.5m, far_disp > 3m)
- Is the data loader returning correct frames?
- Are k_near and k_far ranges appropriate for your trajectory length?

---

## Step 7 — Run Phase 2 (Head Training)

Phase 2 trains the RoPETemporalHead on frozen Phase 1 latents.

```powershell
python train_my_domain.py `
  --data .\data\my_domain `
  --epochs 30 `
  --n-pairs 8000 `
  --phase2 `
  --run-id my_domain_phase2_v1 `
  --notes "Phase 2 RoPE head on frozen encoder"
```

Phase 2 encoding time is a one-time cost (all trajectories encoded once, then
cached as latents). At ~0.3s per file, 1000 files takes ~5 minutes.

**Phase 2 success signals:**
- AUROC jumps at epoch 1 (head has coordinate information the encoder lacks)
- Separation (`far_sim - close_sim`) grows continuously
- AUROC > Phase 1 metric within 5 epochs

---

## Step 8 — Commit to Lineage

The training script calls `Lineage.commit()` automatically. Verify it worked:

```powershell
python lineage.py summary --domain my_domain
```

Expected:
```
────────────────────────────────────────────────────────────
Lineage: my_domain  (2 total runs)
────────────────────────────────────────────────────────────
  run_id                               auroc  notes
  ─────────────────────────────────  ───────  ──────────────────
  my_domain_phase1_v1                 0.8120  Phase 1 baseline...
  my_domain_phase2_v1                 0.9300  Phase 2 RoPE head ★

  Best: my_domain_phase2_v1  auroc=0.9300
  Status: ✅ progressing  (0 runs since best)
────────────────────────────────────────────────────────────
```

---

## Step 9 — Run Held-Out Evaluation

Always evaluate on data the model never saw during training:

```powershell
python eval_my_domain.py `
  --data .\data\my_domain `
  --checkpoint .\checkpoints\my_domain `
  --max-files 200 `
  --held-out-only
```

If the held-out AUROC is within ~3pp of the training AUROC, the model
generalises. A gap larger than 5pp suggests overfitting — try more data,
stronger augmentation, or reduce capacity.

Update the spec's `baseline_result` with the held-out number (not the training
number):

```python
# In domain_spec.py
baseline_result=0.9300,   # held-out AUROC, not training AUROC
```

---

## Step 10 — Check for Plateau and Next Intervention

```powershell
python domain_scaffold.py --domain my_domain --auto --dry-run
```

This prints the current status and the next intervention to try if the metric
has plateaued:

```
════════════════════════════════════════════════════════════
Auto-loop: my_domain  target=auroc>0.90
════════════════════════════════════════════════════════════
  Best: my_domain_phase2_v1  auroc=0.9300
  Status: ✅ progressing  (0 runs since best)
✅ Target reached (0.9300 ≥ 0.90). Done.
```

If the target is not yet reached:
```
⚠️  Plateau detected. Next intervention: increase_k_far
── Next command to run on NUC ─────────────────────────────────
  python train_my_domain.py \
  --data .\data\my_domain \
  --k-far-max 70 \
  --run-id my_domain_k_far70_v3 \
  ...
```

Copy and run the printed command. The intervention adjusts the config
automatically — you do not need to edit any files.

---

## Step 11 — Update the Domain Registry

Once you have a stable held-out result, update the spec in `domain_spec.py`:

```python
register(DomainSpec(
    ...
    baseline_result=0.9300,   # ← update with held-out result
    notes="Phase 2 AUROC 0.9300 held-out. Target 0.90 reached.",
    ...
))
```

Then verify the registry shows the correct status:

```powershell
python domain_spec.py list
```

Your domain should now show `✅` status.

---

## Quick Reference — All Commands

```powershell
# Registry
python domain_spec.py list
python domain_spec.py show   --domain my_domain
python domain_spec.py validate --domain my_domain
python domain_spec.py template

# Scaffold
python domain_scaffold.py --domain my_domain --scaffold
python domain_scaffold.py --domain my_domain --scaffold --overwrite

# Training
python train_my_domain.py --data .\data\my_domain --probe-only
python train_my_domain.py --data .\data\my_domain --max-files 5 --epochs 3 --n-pairs 500
python train_my_domain.py --data .\data\my_domain --epochs 30 --n-pairs 8000
python train_my_domain.py --data .\data\my_domain --epochs 30 --n-pairs 8000 --phase2

# Eval
python eval_my_domain.py --data .\data\my_domain --held-out-only --max-files 200

# Lineage
python lineage.py summary
python lineage.py summary  --domain my_domain
python lineage.py best     --domain my_domain
python lineage.py plateau  --domain my_domain

# Interventions
python domain_scaffold.py --domain my_domain --next-intervention --dry-run
python domain_scaffold.py --domain my_domain --next-intervention
python domain_scaffold.py --domain my_domain --auto --dry-run
python domain_scaffold.py --domain my_domain --auto --max-interventions 5
```

---

## Checklist

Use this before declaring a domain production-ready:

```
[ ] DomainSpec registered and validates cleanly
[ ] Probe passes (correct file count, shapes, image decode)
[ ] Smoke test completes (3 epochs, loss drops)
[ ] Phase 1 training: loss < 1.0 by epoch 20
[ ] Phase 2 training: AUROC > primary_metric_target
[ ] Held-out eval: AUROC within 3pp of training AUROC
[ ] Lineage committed (python lineage.py summary shows runs)
[ ] baseline_result updated in domain_spec.py
[ ] python domain_spec.py list shows ✅ for this domain
[ ] Eval script fully implemented (not just the scaffold TODO)
```

---

## Common Errors and Fixes

**`ValueError: low >= high` during pair sampling**
Short trajectories (T < k_near + k_far). Add a guard:
```python
if T < K_NEAR_MIN + K_FAR_MIN + 2:
    continue
```

**`AUROC = 0.2xxx` (below 0.5)**
Cosine similarity is inverted — encoder places far pairs closer than near
pairs. Pass `-sim` to `auroc_np()` (distance convention: low score = positive).

**`Loaded 0/N tensors` after checkpoint load**
Key name mismatch. Inspect the checkpoint:
```python
ckpt = torch.load("checkpoint.pt", map_location="cpu", weights_only=True)
print(list(ckpt.keys()))
print(list(ckpt.get("model", ckpt).keys())[:5])
```
The production StudentEncoder uses `features.*` / `proj.*` keys. Any other
naming means the checkpoint is from a different architecture.

**`AUROC stays at 0.5 across all epochs`**
Only 2 val files — not enough trajectories to form both close and far pairs.
Use `--val-eval-files 10` minimum.

**`Loss flat at log(batch_size)` past epoch 5**
InfoNCE not learning. Common causes:
- All pairs are from the same trajectory class (no contrast)
- k_near and k_far ranges overlap (ambiguous labels)
- Learning rate too low — try 3e-4 to 1e-3

**Phase 2 encoding takes > 30 minutes**
Normal for > 5000 trajectories. At ~0.3s/file, 10K files takes ~50 minutes.
This is a one-time cost per Phase 2 run.

---

## Architecture Decision Guide

**When to use triplet InfoNCE vs distillation:**
- Triplet InfoNCE: you have trajectory/sequence data and want temporal ordering
- Distillation: you have a good teacher (DINOv2, BEATs) and want to compress it

**When to use RoPETemporalHead vs SubspaceAD:**
- RoPETemporalHead: the task requires geometric distance (navigation, tracking)
- SubspaceAD: the task is anomaly detection (any modality)

**When to train from scratch vs use a pretrained teacher:**
- From scratch: when the domain is visually/acoustically unlike ImageNet
  (e.g., satellite telemetry, industrial vibration)
- Teacher: when the domain resembles natural images or speech
  (e.g., outdoor navigation, cardiac audio)

**Capacity rule of thumb:**
Start with the default 56K StudentEncoder. If AUROC plateaus below target
after 3 interventions, try `larger_capacity` (doubles CNN channels).
If still below target, distil from DINOv2.

---

*CORTEX-PE v16.17 — Saturday 28 March 2026*
