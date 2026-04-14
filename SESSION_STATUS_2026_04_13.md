# NeMo-WM Session Status — Monday 2026-04-13 (Final)

## Executive Summary

Sprint D complete (D1-D6). Aphasia double dissociation confirmed. Visual dreaming
demo produced. Synth v2 pre-training shows broader dynamics > higher ceiling.
PushT GRASP improved to 12% SR. PointMaze k=8 not breaking temporal copy.
arXiv submitted April 11 (on hold). GitHub public.

---

## Paper-Ready Results

### 1. Aphasia Double Dissociation
| Pathway | Full | Aphasia (lang=0) | Δ |
|---------|------|------------------|-----|
| CWM visual | 0.9542 | 0.5000 (chance) | −0.4542 |
| Proprio | 0.9974 | 0.9974 | 0.0000 |

### 2. Visual Temporal Head — Negative Result
| Config | AUROC | Separation |
|--------|-------|------------|
| Encoder only | 0.8912 | 0.0286 |
| Encoder + TemporalHead k=7 | 0.8879 | 0.0154 |

### 3. Synth v2 Transfer — Diversity > Ceiling
| Init | Synth top1 | RECON ep0 | Δ vs cold |
|------|-----------|-----------|-----------|
| Cold start | — | 0.9543 | — |
| v1 (500, 8 scenarios) | 0.961 | 0.9658 | +0.0115 |
| v2 (1000, 14 scenarios) | 0.9248 | 0.9677 | +0.0134 |

### 4. Visual Dreaming
- dream_8sec.png: 3-row (actual / GPS-retrieved / belief bars)
- 32-step drift cosine = 0.2656 confirms 8-second ACh planning horizon
- GPS retrieval IS the visual prediction (JEPA, no pixel generation)

### 5. Sprint D Complete
| Sprint | Component | Metric |
|--------|-----------|--------|
| D1 | BeliefTransitionModel | MSE=0.031 |
| D2 | NeuromodulatorBase | 32/32 shared |
| D3 | AnticipateReactGate | α switching |
| D4 | ImaginationRollout | Loop closed |
| D5 | NeuromodulatedValue | DA·Q−CRT·U+ACh·H |
| D6 | EpisodicBuffer | 35/36 tests |

---

## Training Jobs (Running at Session End)

| Tab | Job | Latest | Projected |
|-----|-----|--------|-----------|
| 1 | RECON fine-tune v2 | ep1=0.9736 | ~0.990 by ep7 |
| 2 | PointMaze k=8 | ep8, ac_lift<0.05 | Kill, pivot to flow |
| 3 | Sprint 6e SemanticHead | ep08, L_null~0 | Lower margin to 0.05 |

---

## All Benchmarks

| Benchmark | Result | Status |
|-----------|--------|--------|
| RECON navigation | AUROC 0.9997 | ✅ |
| PushT flow policy | 100% SR | ✅ |
| PushT GRASP | 12% SR | ⚠️ |
| TwoRoom proprio | AUROC 0.9697 | ✅ |
| TwoRoom visual | 0% SR | ❌ |
| PointMaze | ac_lift +0.013 | ❌ |
| CWRU bearings | AUROC 1.000 | ✅ |
| MIMII audio | AUROC 0.931 | ✅ |
| MVTec visual | AUROC 0.892 | ✅ |

---

## Files Created Today

| File | Purpose |
|------|---------|
| gen_outdoor_hard_negatives.py | 6 outdoor trajectory scenarios |
| nav_text_pairs_sprint9.py | 100 hard negative text pairs |
| sprint9_phase2_runner.py | Synth v2 pipeline orchestrator |
| nemo_dream.py | Visual dreaming demo |
| episodic_buffer.py | D6 EpisodicBuffer |
| test_episodic_buffer.py | 35/36 tests + benchmarks |
| REPRODUCE.md | Paper reproduction guide |
| requirements.txt | pip dependencies |
| NEMO_WM_SELF_QUESTIONS.md | 14 self-questions capability doc |

---

## Key Numbers

| Metric | Value |
|--------|-------|
| Proprio k=16 AUROC | 0.9974 |
| Proprio k=32 AUROC | 0.9997 |
| CWM aphasia Δ | +0.4542 |
| PushT flow SR | 100% |
| Synth v2→RECON ep0 | 0.9677 |
| Dream drift 32 steps | 0.2656 |
| D6 store latency | 0.055ms |
| D6 retrieve latency | 183ms (needs FAISS) |

---

## Next Steps

1. RECON fine-tune v2 eval when training finishes
2. Kill PointMaze, port flow policy
3. Sprint 6e: lower margin to 0.05
4. FAISS for EpisodicBuffer retrieve
5. Paper: add aphasia figure, dream figure, negative result row
