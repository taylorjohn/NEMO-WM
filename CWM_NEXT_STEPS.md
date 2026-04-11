# NeMo-WM Next Steps
**Last updated:** 2026-04-08

---

## Immediate (today / this session)

### 1. Synth→RECON fine-tune eval — when ep10 finishes
```powershell
python eval_recon_auroc.py `
    --head-ckpt checkpoints\cwm\temporal_head_sprint3.pt `
    --cwm-ckpt  checkpoints\cwm\cwm_multidomain_best.pt `
    --hdf5-dir  recon_data\recon_release `
    --n-pairs 1000 --k-pos 4 `
    --proprio-compare --hard-negatives --proprio-no-gps `
    --proprio-ckpt checkpoints\cwm\proprio_kctx16_finetuned.pt
```
Expected: No-VLM ~0.994–0.996. Comparison with cold-start (0.9974) quantifies synthetic pre-training benefit.

### 2. arXiv submission
```
□ Make github.com/taylorjohn public
□ Submit NeMo_WM_Paper_v18_FINAL.docx to arXiv
□ Categories: cs.LG (primary), cs.RO (secondary)
□ License: CC BY 4.0
```

---

## This Week

### Double dissociation on synthetic data (PI lesion)
RECON has no stationary frames (robot always moving). Synthetic stop_and_go has genuine stops.
Fix eval_double_dissociation.py to handle proprio-only HDF5 (no images), then:
```powershell
python eval_double_dissociation.py `
    --cwm-ckpt   checkpoints\cwm\cwm_multidomain_best.pt `
    --head-ckpt  checkpoints\cwm\temporal_head_sprint3.pt `
    --proprio-ckpt checkpoints\cwm\proprio_kctx16_best.pt `
    --hdf5-dir   recon_data\synthetic_sprint9 `
    --n-pairs 200 --vel-thresh 0.05
```
Expected: prop AUROC collapses on stopped frames (PI lesion confirmed). Paper-worthy if confirmed.

### ONNX export for k_ctx=16 proprio encoder
```powershell
python export_proprio_onnx.py `
    --ckpt checkpoints\cwm\proprio_kctx16_best.pt `
    --out  checkpoints\onnx\proprio_kctx16_xint8.onnx
```

### Rust math engine compilation
```powershell
pip install maturin
maturin develop --release
# Restores Phase 12 sub-ms math (currently Python fallback)
```

---

## Sprint 9 Roadmap

### Data collection (unblocks everything)
- 500+ TwoRoom trajectories (visual + proprio) — fixes zero-shot failure
- Outdoor hard negatives: basketball courts (CLIP score 0.273), urban canyons, parks
- 100 navigation-adjacent text pairs for SemanticHead null rejection (Sprint 6e)

### StudentEncoder re-distillation (requires data above)
Zero-shot failure (0% TwoRoom success) is a visual distribution mismatch.
Fix: re-distil StudentEncoder on RECON + TwoRoom + PushT combined → re-export XINT8.

### Sprint 6e — null rejection
Target: 3/3 null rejection (currently 2/3 at threshold=0.235).
Fix: null_weight=5.0 + hard navigation-adjacent negatives.

### Larger synthetic dataset
Current: 500 files, 8 scenarios. Expand to 5,000 files with:
- More geometric variety (urban canyons, enclosed spaces, circular paths)
- Higher curvature range (up to 3.0 rad/s)
- Longer trajectories (140 frames = 35s)

---

## CORTEX-16 Pending

### Wire cortex_brain/ into live engine
99/99 tests passing. DopamineSystem, CJEPAPredictor, StaticCSRRouter, CortexEngine ready.
Need: identify integration point in CortexGoldenMaster tick loop.

### NeuromodulatorBase shared base class
Convergence target: NeMo-WM neuromodulator + CORTEX-16 DopamineSystem derive from same base.

### Directory restructure
```
cortex_pe/    ← CORTEX-PE perception code
cortex_wm/    ← NeMo-WM world model code
```
Do when no training running.

---

## Completed This Session (2026-04-08)

| Task | Result |
|------|--------|
| k_ctx=16 training | top1_acc=0.992, No-VLM=0.9974 |
| k_ctx=32 training | top1_acc=0.996, No-VLM=**0.9997** (neg past unit sphere) |
| PI ablation k_ctx=16 | HD:vel=9:1, vel+ang load-bearing |
| Cortisol integration | sensitivity=0.10, domain reset, NE/eCB tuned |
| Synthetic trajectory generator | 500 files, 8 scenarios, 0.7s |
| Synthetic pre-train | top1_acc=0.961, zero-shot No-VLM=0.9697 |
| Synth→RECON fine-tune | ep5: 0.9832, +0.012 above cold start |
| Paper v18 FINAL | All numbers updated, arXiv-ready |
| Alpaca timeout fix | Exp backoff + stale-data fallback |
| Exit storm fix | Force sync after every order |
| eval_double_dissociation.py | Written (RECON has no stopped frames — use synthetic) |
