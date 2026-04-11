# restructure.py - CORTEX directory restructure
# Splits into cortex_pe/ cortex_wm/ cortex_16/ docs/
# Safe: copies files, does not delete originals until confirmed
# Run: python restructure.py          -- preview
# Run: python restructure.py --apply  -- execute

import shutil, os, argparse
from pathlib import Path

# -- File assignments ----------------------------------------------------------

CORTEX_WM = [
    # World model core
    "train_prior.py",
    "build_prior.py",
    "generate_prior.py",
    "latent_predictor.py",
    "latent_predictorOLD.py",
    "moe_router_v2.py",
    "active_sampler.py",
    "manifold_projection.py",
    "bok_planner.py",
    "architecture_check.py",
    "engine.py",
    "minimal_loop.py",
]

CORTEX_PE = [
    # Perception engine + NPU
    "export_onnx.py",
    "static_npu_mask.py",
    "verify_npu.py",
    "verify_npu_hardwired.py",
    "npu_stress_test.py",
    "hardware.py",
    "final_verify.py",
    "vaip_config.json",
]

CORTEX_16 = [
    # Trading system
    "cortex_v2_chaos_sim.py",
    "cortex_v2_nuc_engine.py",
    "cortex_v2_production_final.py",
    "cortex_v2_simulator.py",
    "cortex_v4_2_engine.py",
    "cortex_v4_3_engine.py",
    "cortex_v4_4_engine.py",
    "cortex_v4_5_engine.py",
    "cortex_v4_7_realistic.py",
    "cortex_v4_8_engine.py",
    "cortex_v4_nuc_engine.py",
    "cortex_v5_0_monday_live.py",
    "cortex_v5_1_paper_live.py",
    "cortex_v5_2_live_bridge.py",
    "cortex_v5_4_fleet_bridge.py",
    "cortex_v5_4_high_vol_test.py",
    "cortex_v5_4_simulation.py",
    "cortex_adapter_v2.py",
    "cortex_alpaca_bridge.py",
    "cortex_fixed_bridge_v4_pdt_shield.py",
    "cortex_gauntlet_30min.py",
    "cortex_gauntlet_v4_PORTFOLIO.py",
    "cortex_live_v1_fixed.py",
    "cortex_allen_neuropixels.py",
    "session_post_mortem.py",
    "liquidate.py",
    "unified_cortex_loop.py",
    "main_bridge.py",
    "launch_cortex.bat",
    "cortex_init.sh",
]

RUST = [
    # Rust bridge -- stays at root with Cargo files
    "cortex_bridge_v5.rs",
    "main.rs",
]

DOCS = [
    # Papers and documentation
    "2506_05495.pdf",
    "2507_12709.pdf",
    "2602_22647.pdf",
    "2602_23643v1.pdf",
    "2603_03276.pdf",
    "2603_04735.pdf",
    "2603_05498.pdf",
    "2603_09951.pdf",
    "2603_12228.pdf",
    "2603_12231.pdf",
    "06ExaminingSomeSeriousChallengescorrectedOK.pdf",
    "Attention_Residuals.pdf",
    "DINOv2_on_AMD_Ryzen_AI_NPU__Complete_Integration_Guide.pdf",
]

ROOT_KEEP = [
    # Stay at root -- shared infrastructure
    "__init__.py",
    "setup.py",
    "pyproject.toml",
    "requires.txt",
    "dependency_links.txt",
    "top_level.txt",
    "SOURCES.txt",
    "ARCHITECTURE.md",
    "README.md",
    "Cargo.toml",
    "Cargo.lock",
    "debug.log",
]

PLAN = {
    "cortex_wm": CORTEX_WM,
    "cortex_pe": CORTEX_PE,
    "cortex_16": CORTEX_16,
    "docs":      DOCS,
    "src/rust":  RUST,
}

# -- Preview and apply ---------------------------------------------------------

def preview():
    print("\nCORTEX Directory Restructure -- PREVIEW")
    print("="*55)
    total = 0
    for dest, files in PLAN.items():
        print(f"\n  {dest}/  ({len(files)} files)")
        for f in files:
            exists = Path(f).exists()
            marker = "ok" if exists else "MISSING"
            print(f"    [{marker}] {f}")
            if exists: total += 1
    print(f"\n  Root keep: {len(ROOT_KEEP)} files")
    print(f"\n  Total to move: {total}")
    print("\nRun with --apply to execute.")
    print("Originals preserved until --cleanup run separately.")

def apply():
    print("\nCORTEX Directory Restructure -- APPLYING")
    print("="*55)
    moved, missing = 0, 0
    for dest, files in PLAN.items():
        Path(dest).mkdir(parents=True, exist_ok=True)
        # Add __init__.py to Python dirs
        if dest not in ("docs", "src/rust"):
            init = Path(dest) / "__init__.py"
            if not init.exists():
                init.write_text(f'"""CORTEX {dest} module."""\n')
        for f in files:
            src = Path(f)
            dst = Path(dest) / src.name
            if src.exists():
                shutil.copy2(src, dst)
                print(f"  -> {dest}/{src.name}")
                moved += 1
            else:
                print(f"  MISSING: {f}")
                missing += 1

    print(f"\n  Moved: {moved}, Missing: {missing}")
    print("\n  Originals still at root -- verify then run:")
    print("  python restructure.py --cleanup")

def cleanup():
    """Remove originals from root after verifying copies are good."""
    print("\nCORTEX Directory Restructure -- CLEANUP")
    print("="*55)
    all_files = []
    for dest, files in PLAN.items():
        for f in files:
            all_files.append((f, dest))

    removed, skipped = 0, 0
    for f, dest in all_files:
        src = Path(f)
        dst = Path(dest) / Path(f).name
        if dst.exists() and src.exists():
            src.unlink()
            print(f"  removed: {f}")
            removed += 1
        elif not dst.exists():
            print(f"  SKIP (copy missing): {f}")
            skipped += 1
        else:
            skipped += 1

    print(f"\n  Removed: {removed}, Skipped: {skipped}")

def generate_powershell():
    """Generate a PowerShell script for running on CORTEX."""
    lines = ["# restructure.ps1 - Run on CORTEX to restructure directories"]
    for dest, files in PLAN.items():
        dest_ps = dest.replace("/", "\\")
        lines.append(f'\nNew-Item -ItemType Directory -Force -Path "{dest_ps}" | Out-Null')
        for f in files:
            lines.append(
                f'if (Test-Path "{f}") {{ Copy-Item "{f}" "{dest_ps}\\" }}'
            )
    script = "\n".join(lines)
    with open("restructure.ps1", "w") as out:
        out.write(script)
    print("Generated: restructure.ps1")
    print("Run on CORTEX: .\\restructure.ps1")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--apply",   action="store_true")
    p.add_argument("--cleanup", action="store_true")
    p.add_argument("--ps1",     action="store_true",
                   help="Generate PowerShell script for CORTEX")
    a = p.parse_args()

    if a.cleanup:
        cleanup()
    elif a.apply:
        apply()
    elif a.ps1:
        generate_powershell()
    else:
        preview()
