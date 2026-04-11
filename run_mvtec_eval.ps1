# CORTEX-PE MVTec AD Evaluation Commands
# Run from: C:\Users\MeteorAI\Desktop\CORTEX
# Updated: Sunday 29 March 2026

# ── Probe ────────────────────────────────────────────────────────────────────
python train_mvtec.py --data .\data\mvtec --probe-only

# ── Best result (ensemble k=32, random encoder) ───────────────────────────────
python train_mvtec.py `
  --data .\data\mvtec `
  --k 32 --ensemble `
  --run-id mvtec_k32_v1 `
  --notes "Ensemble k=32 random encoder"

# ── Per-defect breakdown (global k=32) ────────────────────────────────────────
python eval_mvtec.py `
  --data .\data\mvtec `
  --k 32 `
  --per-defect

# ── Per-defect breakdown with encoder (use after DINOv2 distillation) ─────────
# python eval_mvtec.py `
#   --data .\data\mvtec `
#   --k 32 --per-defect `
#   --encoder-ckpt .\checkpoints\dinov2_student\student_best.pt

# ── RECON encoder zero-shot transfer ─────────────────────────────────────────
python train_mvtec.py `
  --data .\data\mvtec `
  --k 32 --ensemble `
  --encoder-ckpt .\checkpoints\recon_student\student_best.pt `
  --run-id mvtec_recon_k32_v1 `
  --notes "Ensemble k=32, RECON encoder"

# ── Full eval with competitive table ─────────────────────────────────────────
python eval_mvtec.py `
  --data .\data\mvtec `
  --k 32 `
  --per-defect

# ── Lineage query ─────────────────────────────────────────────────────────────
python lineage.py summary --domain mvtec
