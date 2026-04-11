# CORTEX-16 Training Pipeline — PowerShell Commands
# Run these one at a time in order.
# Make sure all .py files are in C:\Users\MeteorAI\Desktop\CORTEX first.

# ── Step 0: Check you have the required packages ─────────────────────────────
pip install torch torchvision pillow matplotlib alpaca-trade-api python-dotenv

# ── Step 1: Phase 1 — Semantic Grounding (~2 hours CPU) ──────────────────────
python train_distillation.py --phase 1 --data ./tiny-imagenet-200/train --steps 12000 --out ./checkpoints

# ── Step 2: Evaluate Phase 1 ─────────────────────────────────────────────────
python evaluate_cortex.py --weights ./checkpoints/cortex_student_phase1_final.pt --out ./checkpoints/eval_phase1.json

# ── Step 3: Collect Phase 2 Domain Frames (needs Alpaca keys in .env) ────────
python collect_phase2_frames.py --symbol SPY --days 60 --out ./phase2_frames

# ── Step 4: Phase 2 — Dynamics Grounding ─────────────────────────────────────
python train_distillation.py --phase 2 --data ./phase2_frames --steps 6000 --resume ./checkpoints/cortex_student_phase1_final.pt --out ./checkpoints

# ── Step 5: Evaluate Phase 2 with regression guard ───────────────────────────
python evaluate_cortex.py --weights ./checkpoints/cortex_student_phase2_final.pt --baseline ./checkpoints/cortex_student_phase1_final.pt --out ./checkpoints/eval_phase2.json

# ── Step 6: Certify semantic axes ────────────────────────────────────────────
python certify_student.py --weights ./checkpoints/cortex_student_phase2_final.pt --out ./checkpoints/cortex_certificate.json

# ── Step 7: Export to NPU (run inside ryzen-ai conda env) ────────────────────
# First switch env: conda activate ryzen-ai-1.7.0
python export_student_xint8.py --weights ./checkpoints/cortex_student_phase2_final.pt --calib ./calib_images --out ./npu_models
