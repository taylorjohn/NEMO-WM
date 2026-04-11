# check_multidomain.py
# Preflight check for train_cwm_multidomain.py
# Verifies all imports, Sandwich Norm compatibility, and data paths
# Run this before train_cwm_multidomain.py to catch issues early
#
# Usage:
#   python check_multidomain.py
#   python check_multidomain.py --fix   (patches CortexWorldModel predictor)
#
# No unicode -- Windows cp1252 safe
import sys, argparse, importlib
import numpy as np
import torch
from pathlib import Path

PASS = "ok"
FAIL = "FAIL"
WARN = "warn"

results = []

def check(label, ok, detail=""):
    status = PASS if ok else FAIL
    results.append((label, status, detail))
    marker = "  [ok]" if ok else "  [FAIL]"
    print(f"{marker} {label}" + (f" -- {detail}" if detail else ""))
    return ok

def warn(label, detail=""):
    results.append((label, WARN, detail))
    print(f"  [warn] {label}" + (f" -- {detail}" if detail else ""))

def section(title):
    print(f"\n{title}")
    print("-" * 50)

# ---- 1. Core dependencies ---------------------------------------------------
section("1. Core imports")

for mod in ["torch","numpy","h5py","PIL","torchvision","sklearn"]:
    try:
        importlib.import_module(mod)
        check(mod, True)
    except ImportError as e:
        check(mod, False, str(e))

# ---- 2. NeMo-WM module imports ----------------------------------------------
section("2. NeMo-WM module imports")

modules_needed = [
    ("neuromodulator",   "NeuromodulatorState"),
    ("cwm_neuro_reward", "NeuromodulatedCWMLoss"),
    ("cwm_neuro_reward", "RegimeGatedTrainer"),
    ("cwm_moe_jepa",     "MoEJEPAPredictor"),
    ("cwm_moe_jepa",     "jepa_moe_loss"),
    ("train_cwm",        "CortexWorldModel"),
    ("train_cwm",        "DOMAIN_IDS"),
    ("domain_loaders",   "build_multi_domain_loader"),
    ("train_mvtec",      "StudentEncoder"),
]

import_ok = {}
for mod, attr in modules_needed:
    try:
        m = importlib.import_module(mod)
        has_attr = hasattr(m, attr)
        check(f"{mod}.{attr}", has_attr,
              "" if has_attr else f"module found but {attr} missing")
        import_ok[f"{mod}.{attr}"] = has_attr
    except ImportError as e:
        check(f"{mod}.{attr}", False, str(e))
        import_ok[f"{mod}.{attr}"] = False

# ---- 3. Sandwich Norm in latent_predictor -----------------------------------
section("3. Sandwich Norm compatibility")

try:
    from cortex_wm.latent_predictor import CJEPAPredictor, RMSNorm, DynamicTanh
    m = CJEPAPredictor(sandwich_norm=True); m.eval()
    z = torch.randn(2, 4, 128)
    with torch.no_grad():
        out = m(z, torch.zeros(2,2), torch.randn(2,2))
    bounded = bool(out.abs().max() < 1.0)
    check("CJEPAPredictor(sandwich_norm=True)", bounded,
          f"range=[{out.min():.3f},{out.max():.3f}]")
    check("RMSNorm importable", True)
    check("DynamicTanh importable", True)
except ImportError:
    # Try root latent_predictor
    try:
        from latent_predictor import CJEPAPredictor, RMSNorm, DynamicTanh
        m = CJEPAPredictor(sandwich_norm=True); m.eval()
        z = torch.randn(2,4,128)
        with torch.no_grad():
            out = m(z, torch.zeros(2,2), torch.randn(2,2))
        bounded = bool(out.abs().max() < 1.0)
        check("CJEPAPredictor(sandwich_norm=True) [root]", bounded,
              f"range=[{out.min():.3f},{out.max():.3f}]")
    except Exception as e:
        check("CJEPAPredictor Sandwich Norm", False, str(e))
except Exception as e:
    check("CJEPAPredictor Sandwich Norm", False, str(e))

# ---- 4. CortexWorldModel predictor check ------------------------------------
section("4. CortexWorldModel predictor")

try:
    from train_cwm import CortexWorldModel
    cwm = CortexWorldModel(d_model=128, K=16)
    pred = cwm.predictor

    # Check if it's Sandwich Norm capable
    has_sandwich = hasattr(pred, 'sandwich_norm')
    has_rms = any("RMSNorm" in type(m).__name__
                  for m in pred.modules())
    has_dtanh = any("DynamicTanh" in type(m).__name__
                    for m in pred.modules())

    check("CortexWorldModel instantiates", True)
    check("predictor has sandwich_norm attr", has_sandwich,
          "needs patch if False" if not has_sandwich else "")
    check("predictor has RMSNorm", has_rms,
          "sandwich_norm=False currently" if not has_rms else "")
    check("predictor has DynamicTanh", has_dtanh,
          "sandwich_norm=False currently" if not has_dtanh else "")

    # Forward pass smoke test
    try:
        cwm.eval()
        z = torch.randn(2, 128)
        particles_dummy = torch.randn(2, 16, 128)
        action_dummy    = torch.zeros(2, 9)  # placeholder, overridden below
        domain_id       = torch.zeros(2, dtype=torch.long)
        context_h       = cwm.thick_gru.init_context(2, torch.device("cpu"))
        pos_dummy       = torch.zeros(2, 16, 2)
        from train_cwm import MAX_ACTION_DIM
        action_dummy = torch.zeros(2, MAX_ACTION_DIM)
        out = cwm.predict(
            particles  = particles_dummy,
            action     = action_dummy,
            context_h  = context_h,
            positions  = pos_dummy,
            domain_id  = domain_id,
            regime     = "EXPLOIT",
            ach        = 0.3,
        )
        check("CortexWorldModel forward pass", True,
              f"keys={list(out.keys())}")
    except Exception as e:
        check("CortexWorldModel forward pass", False, str(e))

except ImportError as e:
    check("train_cwm import", False, str(e))
except Exception as e:
    check("CortexWorldModel init", False, str(e))

# ---- 5. Checkpoints ---------------------------------------------------------
section("5. Checkpoints")

ckpts = {
    "Sprint 1 CWM":     Path("checkpoints/cwm/cwm_best.pt"),
    "StudentEncoder":   Path("checkpoints/dinov2_student/student_best.pt"),
}
for name, path in ckpts.items():
    if path.exists():
        sz = path.stat().st_size / 1e6
        check(name, True, f"{sz:.1f}MB at {path}")
    else:
        check(name, False, f"not found: {path}")

# ---- 6. Data paths ----------------------------------------------------------
section("6. Data paths")

data_paths = {
    "RECON":    Path("recon_data/recon_release"),
    "SMAP":     Path("smap_data"),
    "MVTec":    Path("data/mvtec"),
    "OGBench":  Path("ogbench_data"),
    "PushT":    Path("recon_data/pushtoom"),
}
active = []
for name, path in data_paths.items():
    if path.exists():
        n_files = sum(1 for _ in path.rglob("*.hdf5")) + \
                  sum(1 for _ in path.rglob("*.png"))
        check(name, True, f"{n_files} files at {path}")
        active.append(name)
    else:
        warn(name, f"not found: {path} (will be skipped)")

# ---- 7. Summary -------------------------------------------------------------
section("Summary")

n_pass = sum(1 for _, s, _ in results if s == PASS)
n_fail = sum(1 for _, s, _ in results if s == FAIL)
n_warn = sum(1 for _, s, _ in results if s == WARN)

print(f"\n  {n_pass} passed, {n_fail} failed, {n_warn} warnings")
print(f"  Active data domains: {active}")

if n_fail == 0:
    print("\n  All checks passed. Ready to run:")
    print(f"  python train_cwm_multidomain.py \\")
    print(f"      --cwm-ckpt checkpoints\\cwm\\cwm_best.pt \\")
    print(f"      --recon-dir recon_data\\recon_release \\")
    print(f"      --smap-dir smap_data \\")
    print(f"      --mvtec-dir data\\mvtec \\")
    print(f"      --epochs 5 --log-every 20")
else:
    print(f"\n  {n_fail} issue(s) to fix before running.")
    fails = [(l, d) for l, s, d in results if s == FAIL]
    for label, detail in fails:
        print(f"    FAIL: {label}" + (f" ({detail})" if detail else ""))

    # Offer specific fixes
    missing_mods = [l.split(".")[0] for l, s, _ in results
                    if s == FAIL and "Import" in _]
    if missing_mods:
        print(f"\n  Missing modules: {set(missing_mods)}")
        print(f"  These files must exist in CORTEX working directory.")

    no_sandwich = not any(
        s == PASS and "RMSNorm" in l for l, s, _ in results
    )
    if no_sandwich:
        print(f"\n  CortexWorldModel predictor lacks Sandwich Norm.")
        print(f"  Run with --fix to patch it automatically.")

# ---- Optional: patch CortexWorldModel predictor ----------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--fix", action="store_true",
                help="Patch CortexWorldModel.predictor with Sandwich Norm")
args = ap.parse_args()

if args.fix:
    print("\n--- Applying Sandwich Norm patch to train_cwm.py ---")
    cwm_path = Path("train_cwm.py")
    if not cwm_path.exists():
        print("  train_cwm.py not found")
    else:
        with open(cwm_path, "rb") as f:
            code = f.read().decode("utf-8", errors="replace")

        if "sandwich_norm" in code:
            print("  Already patched -- sandwich_norm found in train_cwm.py")
        else:
            # Insert RMSNorm/DynamicTanh import and update CJEPAPredictor init
            old_import = "from latent_predictor import CJEPAPredictor"
            new_import = ("from cortex_wm.latent_predictor import "
                          "CJEPAPredictor, RMSNorm, DynamicTanh")
            if old_import in code:
                code = code.replace(old_import, new_import, 1)

            # Update CJEPAPredictor instantiation
            for old_init in [
                "CJEPAPredictor()",
                "CJEPAPredictor(latent_dim=128)",
                "CJEPAPredictor(latent_dim=d_model)",
            ]:
                new_init = old_init.rstrip(")") + ", sandwich_norm=True)"
                if old_init in code:
                    code = code.replace(old_init, new_init, 1)
                    print(f"  Patched: {old_init} -> {new_init}")

            with open(cwm_path, "wb") as f:
                f.write(code.encode("utf-8"))
            print("  train_cwm.py updated with Sandwich Norm")
