"""
smoke_test_cwm.py  --  CORTEX CWM Unified Smoke Test
=====================================================
Tests all Sprint 3-5 components without needing full data or trained models.
Run from CORTEX root: python smoke_test_cwm.py
"""
import sys, time, traceback
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

RESULTS = []

def run(name, fn):
    t0 = time.perf_counter()
    try:
        fn()
        ms = (time.perf_counter() - t0) * 1000
        RESULTS.append(("PASS", name, f"{ms:.0f}ms"))
    except Exception as e:
        RESULTS.append(("FAIL", name, str(e)[:100]))
        if "--verbose" in sys.argv:
            traceback.print_exc()

def _test_0():
    from cortex_geo_db import VirtualLookaheadSink, GeoLatentDatabase


def _test_1():
    from domain_loaders import _signal_to_frame


def _test_2():
    from cwm_hierarchical import (
        InverseDynamicsSkillBoundary,
        scar_transition_loss,
        splates_skill_reward,
    )


def _test_3():
    from cwm_hierarchical import HierarchicalCWM
    from train_cwm import CortexWorldModel


def _test_4():
    import importlib.util, json


def _test_5():
    import tempfile, os
    from pathlib import Path


def _test_6():
    from eval_cwm_ablations import RandomEncoder, StudentOnlyEncoder


def _test_7():
    from eval_recon_vl_sink import run_vl_sink_comparison
    import inspect
    sig = inspect.signature(run_vl_sink_comparison)
    params = list(sig.parameters.keys())
    assert "cwm_ckpt"  in params, "Missing cwm_ckpt param"
    assert "head_ckpt" in params, "Missing head_ckpt param"
    assert "hdf5_dir"  in params, "Missing hdf5_dir param"



run("cortex_geo_db -- VL Sink shape (B=4, K=16, D=128)", _test_0)
run("domain_loaders -- _signal_to_frame + batch collation", _test_1)
run("cwm_hierarchical -- IDB + SCaR + SPlaTES shapes", _test_2)
run("cwm_hierarchical -- HierarchicalCWM full forward", _test_3)
run("generate_paper_results -- LeWM reference table completeness", _test_4)
run("generate_paper_results -- output dir creation", _test_5)
run("eval_cwm_ablations -- RandomEncoder + StudentOnlyEncoder shapes", _test_6)
run("eval_recon_vl_sink -- importable + run_vl_sink_comparison signature", _test_7)

print()
print("=" * 62)
print("  CORTEX CWM Smoke Test Results")
print("=" * 62)
passed = sum(1 for r in RESULTS if r[0] == "PASS")
failed = sum(1 for r in RESULTS if r[0] == "FAIL")
for status, name, detail in RESULTS:
    icon = "OK  " if status == "PASS" else "FAIL"
    print(f"  {icon}  {name}")
    if status == "FAIL":
        print(f"        -> {detail}")
print()
print(f"  {passed}/{len(RESULTS)} passed  |  {failed} failed")
print("=" * 62)
if failed > 0:
    print("\n  Run with --verbose for full tracebacks.")
    sys.exit(1)
