"""
beats_diagnostic.py
====================
Diagnoses why the BEATs distillation pipeline produces loss=0.0000.

Known causes of zero distillation loss:
  1. Detached tensors — teacher output used directly in loss with stop_grad on student
  2. Wrong branch — loss computed between teacher and teacher (not student)
  3. Collapsed student — student outputs constant vector → cosine_sim=1.0 always
  4. NaN teacher output — loss = 0 after nan_to_num
  5. Loss scale issue — lambda so small the displayed value rounds to 0.0000

Run this against your actual train_distillation.py BEATs setup:
    python beats_diagnostic.py --checkpoint ./checkpoints/cardiac_beats/best.pt
    python beats_diagnostic.py --smoke   # full smoke test without data

Outputs a report identifying which failure mode is active.
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ── Diagnostic checks ──────────────────────────────────────────────────────

def check_detached_tensors(student_out: torch.Tensor, teacher_out: torch.Tensor):
    """Check 1: Are gradients flowing through the student output?"""
    print("\n── Check 1: Gradient flow ──")

    if not student_out.requires_grad:
        print("  ❌ FAIL: student_out.requires_grad = False")
        print("     Root cause: student output was detached before loss computation.")
        print("     Fix: remove .detach() from student forward pass in loss calc.")
        return False

    loss = (1.0 - F.cosine_similarity(student_out, teacher_out.detach(), dim=-1)).mean()
    loss.backward(retain_graph=True)

    if student_out.grad is None and student_out.grad_fn is None:
        print("  ❌ FAIL: No gradient function on student_out")
        return False

    print("  ✅ PASS: Gradients flow through student output")
    return True


def check_teacher_output(teacher_out: torch.Tensor):
    """Check 2: Is the teacher producing valid non-constant output?"""
    print("\n── Check 2: Teacher output validity ──")

    if torch.isnan(teacher_out).any():
        print("  ❌ FAIL: Teacher output contains NaN")
        print("     Root cause: BEATs model producing NaN → nan_to_num → loss=0")
        print("     Fix: check BEATs input normalization and model weights")
        return False

    if torch.isinf(teacher_out).any():
        print("  ❌ FAIL: Teacher output contains Inf")
        return False

    std = teacher_out.std(dim=0).mean().item()
    if std < 1e-6:
        print(f"  ❌ FAIL: Teacher output collapsed (std={std:.2e})")
        print("     Root cause: Teacher outputting constant vector → cosine_sim=1.0 always → loss=0")
        print("     Fix: verify teacher forward pass is not returning zeros/constants")
        return False

    mean_norm = teacher_out.norm(dim=-1).mean().item()
    print(f"  ✅ PASS: Teacher output valid (std={std:.4f}, mean_norm={mean_norm:.4f})")
    return True


def check_student_collapse(student_out: torch.Tensor):
    """Check 3: Is the student collapsed to a constant output?"""
    print("\n── Check 3: Student collapse ──")

    std = student_out.std(dim=0).mean().item()
    if std < 1e-4:
        print(f"  ❌ FAIL: Student output collapsed (std={std:.2e})")
        print("     Root cause: student maps all inputs to same vector")
        print("     Fix: reinitialize student, check BatchNorm / weight init")
        return False

    print(f"  ✅ PASS: Student not collapsed (std={std:.4f})")
    return True


def check_loss_same_branch(student_out: torch.Tensor, teacher_out: torch.Tensor):
    """Check 4: Is loss accidentally computed between teacher and teacher?"""
    print("\n── Check 4: Loss branch identity ──")

    # If student and teacher outputs are identical, loss is computed wrong
    if torch.allclose(student_out.detach(), teacher_out.detach(), atol=1e-4):
        print("  ❌ FAIL: student_out ≈ teacher_out (same tensor or same branch)")
        print("     Root cause: loss computed between teacher and teacher outputs")
        print("     Fix: ensure student forward pass is separate from teacher")
        return False

    print("  ✅ PASS: Student and teacher outputs are distinct")
    return True


def check_loss_magnitude(student_out: torch.Tensor, teacher_out: torch.Tensor):
    """Check 5: Is the loss actually small or just not logged?"""
    print("\n── Check 5: Loss magnitude ──")

    with torch.no_grad():
        cos_sim = F.cosine_similarity(
            student_out.detach(), teacher_out.detach(), dim=-1
        )
        loss = (1.0 - cos_sim).mean().item()
        print(f"  Raw cosine distill loss: {loss:.6f}")

        if loss < 1e-5:
            print("  ❌ FAIL: Loss is genuinely near zero — student and teacher are identical")
            return False
        elif loss < 0.01:
            print("  ⚠️  WARNING: Loss < 0.01 — may display as 0.0000 if log format is %.4f")
            print(f"     Fix: log with {{:.6f}} instead of {{:.4f}}")
            return True  # Not broken, just display issue
        else:
            print(f"  ✅ PASS: Loss is non-trivial ({loss:.4f})")
            return True


def check_nan_masking(student_out: torch.Tensor, teacher_out: torch.Tensor):
    """Check 6: Is nan_to_num silently masking NaN loss?"""
    print("\n── Check 6: NaN masking ──")

    with torch.no_grad():
        cos_sim = F.cosine_similarity(
            student_out.detach(), teacher_out.detach(), dim=-1
        )
        loss_raw = (1.0 - cos_sim).mean()

        if torch.isnan(loss_raw):
            print("  ❌ FAIL: Raw loss is NaN — nan_to_num(loss) = 0.0")
            print("     Root cause: NaN in student or teacher → NaN cos_sim → NaN loss → 0 after masking")
            print("     Fix: add gradient clipping, check for zero-norm vectors in normalization")
            return False

    print("  ✅ PASS: Loss is not NaN")
    return True


# ── Smoke test with synthetic data ─────────────────────────────────────────

def run_smoke_test():
    """
    Full smoke test with synthetic student/teacher tensors.
    Tests all 6 failure modes with intentionally broken inputs.
    """
    print("=" * 60)
    print("  BEATs PIPELINE DIAGNOSTIC — SMOKE TEST")
    print("=" * 60)

    B, D = 16, 768
    teacher_out = F.normalize(torch.randn(B, D), dim=-1)

    # Scenario A: correct student
    print("\n▶ Scenario A: Healthy pipeline (should all pass)")
    student_a = F.normalize(torch.randn(B, D, requires_grad=True), dim=-1)
    _run_checks(student_a, teacher_out)

    # Scenario B: detached student
    print("\n▶ Scenario B: Detached student (should fail check 1)")
    student_b = F.normalize(torch.randn(B, D), dim=-1)  # no requires_grad
    _run_checks(student_b, teacher_out)

    # Scenario C: collapsed student
    print("\n▶ Scenario C: Collapsed student (should fail checks 3 and 5)")
    student_c = F.normalize(torch.ones(B, D) * 0.001, dim=-1).requires_grad_(True)
    _run_checks(student_c, teacher_out)

    # Scenario D: same branch (teacher == student)
    print("\n▶ Scenario D: Loss between teacher and teacher (should fail check 4)")
    student_d = teacher_out.clone().requires_grad_(True)
    _run_checks(student_d, teacher_out)

    # Scenario E: NaN teacher
    print("\n▶ Scenario E: NaN teacher output (should fail check 2)")
    nan_teacher = torch.full((B, D), float("nan"))
    student_e = F.normalize(torch.randn(B, D, requires_grad=True), dim=-1)
    _run_checks(student_e, nan_teacher)


def _run_checks(student_out, teacher_out):
    results = {}
    try:
        results["teacher_valid"] = check_teacher_output(teacher_out)
        results["student_not_collapsed"] = check_student_collapse(student_out)
        results["different_branches"] = check_loss_same_branch(student_out, teacher_out)
        results["loss_magnitude"] = check_loss_magnitude(student_out, teacher_out)
        results["no_nan_masking"] = check_nan_masking(student_out, teacher_out)
        results["grad_flow"] = check_detached_tensors(student_out, teacher_out)
    except Exception as e:
        print(f"  ⚠️  Check raised exception: {e}")

    failures = [k for k, v in results.items() if not v]
    if failures:
        print(f"\n  ❌ FAILING CHECKS: {failures}")
        print(f"  → Likely root cause: {_diagnose(failures)}")
    else:
        print("\n  ✅ All checks passed — pipeline is healthy")


def _diagnose(failures):
    if "teacher_valid" in failures:
        return "BEATs teacher producing NaN/Inf/constant output"
    if "grad_flow" in failures:
        return "Student output detached before loss — gradients not flowing"
    if "different_branches" in failures:
        return "Loss computed on same branch (teacher vs teacher)"
    if "student_not_collapsed" in failures:
        return "Student encoder collapsed to constant output"
    if "no_nan_masking" in failures:
        return "NaN in loss silently zeroed by nan_to_num"
    if "loss_magnitude" in failures:
        return "Loss is genuinely 0 — check all inputs are non-identical"
    return "Unknown — run individual checks manually"


# ── Load and test against actual checkpoint ────────────────────────────────

def diagnose_from_checkpoint(checkpoint_path: str):
    """
    Load a student checkpoint and run diagnostics against synthetic teacher output.
    Useful for checking student collapse after training with broken pipeline.
    """
    print(f"\n── Checkpoint Diagnostic: {checkpoint_path} ──")
    device = torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Try to extract student state dict
    state = ckpt.get("student") or ckpt.get("model") or ckpt
    if isinstance(state, dict) and "weight" not in str(list(state.keys())[:3]):
        # Nested dict — try common keys
        for key in ["student_state_dict", "encoder", "backbone"]:
            if key in state:
                state = state[key]
                break

    print(f"  Checkpoint keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else 'raw tensor'}")

    # Check if weights are non-trivial
    if isinstance(state, dict):
        total_norm = sum(v.norm().item() for v in state.values() if torch.is_tensor(v))
        print(f"  Total weight norm: {total_norm:.4f}")
        if total_norm < 1e-3:
            print("  ❌ Weights near zero — student never trained properly")
        else:
            print("  ✅ Weights have reasonable magnitude")


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BEATs pipeline diagnostic")
    parser.add_argument("--smoke", action="store_true",
                        help="Run smoke test with synthetic data")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to student checkpoint to diagnose")
    args = parser.parse_args()

    if args.smoke or not args.checkpoint:
        run_smoke_test()

    if args.checkpoint:
        diagnose_from_checkpoint(args.checkpoint)

    print("\n── Summary of common BEATs pipeline bugs ──")
    print("""
  1. Detached student:
       student_feat = student(x).detach()  ← removes grad
       Fix: student_feat = student(x)      ← keep grad

  2. Wrong branch:
       loss = cosine(teacher(x), teacher(x2))  ← teacher vs teacher
       Fix: loss = cosine(student(x), teacher(x).detach())

  3. NaN from zero-norm vector:
       z = z / z.norm()  ← if norm=0, produces NaN
       Fix: z = F.normalize(z, dim=-1)  ← handles zero safely

  4. Loss logging format:
       print(f"distill={loss:.4f}")  ← 0.0001 shows as 0.0000
       Fix: print(f"distill={loss:.6f}")

  5. Frozen student:
       for p in student.parameters(): p.requires_grad = False  ← forgot to unfreeze
       Fix: ensure student params are trainable before optimizer creation
    """)


if __name__ == "__main__":
    main()
