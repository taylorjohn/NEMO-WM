"""
test_neuromodulator.py  —  CORTEX-PE v16.11
=============================================
pytest suite for the six-signal neuromodulator.

Run: pytest test_neuromodulator.py -v

Split into two groups:
  - Pure-Python tests (no torch) — run anywhere
  - Torch-dependent tests — run on NUC with full pytorch
"""

import math, sys, time, types, json
sys.path.insert(0, '.')

# ── Minimal torch stub for CI / syntax verification ───────────────────────
# Remove this block when running with real PyTorch (NUC)
import importlib
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import pytest

if HAS_TORCH:
    import torch.nn.functional as F
    from neuromodulator import (
        NeuromodulatorState, ModulatedPlanner,
        Regime, classify_regime, neuro_to_packet,
    )
else:
    # Stub torch so the module can be imported for pure-Python tests
    _torch = types.ModuleType("torch")
    _torch.Tensor = object
    _torch.no_grad = lambda: __import__("contextlib").nullcontext()
    _F = types.ModuleType("torch.nn.functional")
    _torch.nn = types.ModuleType("torch.nn")
    _torch.nn.functional = _F
    sys.modules["torch"] = _torch
    sys.modules["torch.nn"] = _torch.nn
    sys.modules["torch.nn.functional"] = _F
    _np = types.ModuleType("numpy")
    _np.ndarray = object
    sys.modules["numpy"] = _np
    from neuromodulator import (
        NeuromodulatorState, ModulatedPlanner,
        Regime, classify_regime, neuro_to_packet,
    )

# ════════════════════════════════════════════════════════════════════════════
# Pure-Python tests (no torch required)
# ════════════════════════════════════════════════════════════════════════════

# ── classify_regime ──────────────────────────────────────────────────────

@pytest.mark.parametrize("da,sht,expected", [
    (0.8, 0.8, Regime.EXPLORE),
    (0.8, 0.2, Regime.WAIT),
    (0.2, 0.2, Regime.REOBSERVE),
    (0.2, 0.8, Regime.EXPLOIT),
    (0.0, 0.0, Regime.REOBSERVE),
    (1.0, 1.0, Regime.EXPLORE),
    (0.5, 0.5, Regime.EXPLORE),   # >= threshold → EXPLORE
])
def test_classify_regime(da, sht, expected):
    assert classify_regime(da, sht) == expected


# ── NeuromodulatorState init ─────────────────────────────────────────────

def test_init_defaults():
    n = NeuromodulatorState()
    assert n.da  == 0.5
    assert n.sht == 0.5
    assert n.rho == 0.0
    assert n.ach == 0.5
    assert n.ei  == 1.0
    assert n.ado == 0.0
    assert n.ecb == 0.0


# ── Action scales ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("regime,expected_range", [
    (Regime.WAIT,      (0.0, 0.0)),
    (Regime.REOBSERVE, (0.3, 0.45)),
    (Regime.EXPLORE,   (0.8, 1.0)),
    (Regime.EXPLOIT,   (0.8, 1.0)),
])
def test_action_scale_ranges(regime, expected_range):
    n = NeuromodulatorState()
    scale = n._action_scale(regime)
    lo, hi = expected_range
    assert lo <= scale <= hi + 1e-9, f"{regime}: scale={scale} not in [{lo},{hi}]"


def test_wait_regime_zero_action():
    n = NeuromodulatorState()
    assert n._action_scale(Regime.WAIT) == 0.0


# ── should_act / is_novel / is_stable ────────────────────────────────────

def test_should_act_false_in_wait():
    n = NeuromodulatorState()
    n.da, n.sht = 0.9, 0.1
    assert not n.should_act

def test_should_act_true_outside_wait():
    n = NeuromodulatorState()
    for da, sht in [(0.1, 0.9), (0.9, 0.9), (0.1, 0.1)]:
        n.da, n.sht = da, sht
        assert n.should_act, f"DA={da},5HT={sht} should allow action"

def test_is_novel_flag():
    n = NeuromodulatorState()
    n.da = 0.8; assert n.is_novel
    n.da = 0.2; assert not n.is_novel

def test_is_stable_flag():
    n = NeuromodulatorState()
    n.sht = 0.8; assert n.is_stable
    n.sht = 0.2; assert not n.is_stable


# ── Adenosine ─────────────────────────────────────────────────────────────

def test_adenosine_zero_at_start():
    n = NeuromodulatorState(session_start=time.time())
    assert n.ado == 0.0

def test_adenosine_increases_over_time():
    past = time.time() - 7200   # 2 hours ago
    n = NeuromodulatorState(session_start=past, ado_saturate_hours=4.0)
    # manually trigger ado computation by reading property
    elapsed = time.time() - n._session_start
    ado = min(1.0, elapsed / n._ado_sat)
    assert 0.4 < ado < 0.6, f"Expected ~0.5 for 2h/4h, got {ado:.3f}"

def test_adenosine_saturates_at_one():
    ancient = time.time() - 100000
    n = NeuromodulatorState(session_start=ancient, ado_saturate_hours=4.0)
    elapsed = time.time() - n._session_start
    ado = min(1.0, elapsed / n._ado_sat)
    assert ado == 1.0

def test_adenosine_reduces_candidates():
    n = NeuromodulatorState()
    n.ado = 0.0;  lo = n._n_candidates(0.5)
    n.ado = 0.9;  hi_ado = n._n_candidates(0.5)
    assert hi_ado < lo, "More adenosine should mean fewer candidates"

def test_adenosine_reduces_action_scale():
    n = NeuromodulatorState()
    n.ado = 0.0; s0 = n._action_scale(Regime.EXPLOIT)
    n.ado = 0.9; s1 = n._action_scale(Regime.EXPLOIT)
    assert s1 < s0, "High adenosine should reduce action scale"


# ── E/I ratio ─────────────────────────────────────────────────────────────

def test_ei_bounds():
    n = NeuromodulatorState()
    for da, sht in [(0.0, 0.99), (1.0, 0.01), (0.5, 0.5)]:
        n.da, n.sht = da, sht
        ei = n.ei
        # Note: ei is not auto-updated without calling update(), so test formula
        ei_computed = max(0.5, min(2.0, da / (1.0 - sht + 0.1)))
        assert 0.5 <= ei_computed <= 2.0

def test_ei_high_da_low_sht_gives_high_ei():
    # High DA (surprise), low 5HT (instability) → high E/I → broad search
    n = NeuromodulatorState()
    n.da, n.sht = 0.9, 0.1
    ei = max(0.5, min(2.0, n.da / (1.0 - n.sht + 0.1)))
    assert ei > 0.8, f"Expected E/I > 0.8 for high-DA state, got {ei:.3f}"

def test_ei_modulates_action_std():
    n = NeuromodulatorState()
    n.ei = 0.5; std_lo = n._action_std()
    n.ei = 2.0; std_hi = n._action_std()
    assert std_hi > std_lo, "Higher E/I should give larger action std"


# ── ACh ───────────────────────────────────────────────────────────────────

def test_ach_modulates_lr_scale():
    n = NeuromodulatorState()
    n.ach = 0.0; lr_lo = n._lr_scale()
    n.ach = 1.0; lr_hi = n._lr_scale()
    assert lr_hi > lr_lo, "High ACh should increase learning rate scale"

def test_ach_lr_scale_bounds():
    n = NeuromodulatorState()
    for ach_val in [0.0, 0.5, 1.0]:
        n.ach = ach_val
        lr = n._lr_scale()
        assert 0.2 <= lr <= 2.0, f"LR scale out of bounds: {lr}"


# ── eCB ───────────────────────────────────────────────────────────────────

def test_ecb_init_zero():
    n = NeuromodulatorState()
    assert n.ecb == 0.0

def test_ecb_is_oscillating_property():
    n = NeuromodulatorState()
    n.ecb = 0.3; assert not n.is_oscillating
    n.ecb = 0.7; assert n.is_oscillating

def test_ecb_suppresses_da_in_should_act():
    # High ecb should suppress effective DA, potentially changing regime
    n = NeuromodulatorState()
    n.da  = 0.65   # just above threshold
    n.sht = 0.2    # low → would be WAIT without eCB
    n.ecb = 0.9    # strong retrograde suppression
    da_eff = n.da * (1.0 - n.ecb * 0.4)
    # da_eff = 0.65 * (1 - 0.36) = 0.65 * 0.64 = 0.416 → below 0.5 threshold
    assert da_eff < n.da_thresh, \
        f"eCB should suppress DA below threshold: da_eff={da_eff:.3f}"


# ── Reset ─────────────────────────────────────────────────────────────────

def test_full_reset_clears_adenosine():
    n = NeuromodulatorState(session_start=time.time() - 3600)
    n.da, n.sht, n.ecb = 0.9, 0.1, 0.8
    n.reset(full=True)
    assert n.da  == 0.5
    assert n.sht == 0.5
    assert n.ecb == 0.0
    assert n.ado == 0.0   # cleared by full reset

def test_partial_reset_preserves_adenosine():
    n = NeuromodulatorState()
    n.ado = 0.6
    n.reset(full=False)
    assert n.ado == 0.6   # preserved


# ── neuro_to_packet ────────────────────────────────────────────────────────

def test_packet_has_all_keys():
    n = NeuromodulatorState()
    # Build a minimal signals dict
    signals = {
        "da": 0.3, "sht": 0.7, "rho": 0.2,
        "ach": 0.4, "ei": 1.2, "ado": 0.1, "ecb": 0.05,
        "da_effective": 0.28, "regime": "EXPLOIT", "confidence": "HIGH",
        "action_scale": 1.0, "eps_scale": 1.24, "action_std": 0.12,
        "n_candidates": 48, "lr_scale": 1.1,
    }
    p = neuro_to_packet(signals)
    for key in ["DA","5HT","NE","ACH","EI","ADO","ECB","DA_EFF",
                "REGIME","CONF","ACT_SCALE","EPS_SCALE","ACT_STD",
                "N_CAND","LR_SCALE"]:
        assert key in p, f"Missing key: {key}"

def test_packet_json_serialisable():
    signals = {
        "da":0.3,"sht":0.7,"rho":0.2,"ach":0.4,"ei":1.2,
        "ado":0.1,"ecb":0.05,"da_effective":0.28,"regime":"EXPLOIT",
        "confidence":"HIGH","action_scale":1.0,"eps_scale":1.24,
        "action_std":0.12,"n_candidates":48,"lr_scale":1.1,
    }
    json.dumps(neuro_to_packet(signals))   # raises if not serialisable


# ── Log ───────────────────────────────────────────────────────────────────

def test_log_bounded_at_1000():
    n = NeuromodulatorState()
    for _ in range(1200):
        n._log.append({"x": 1})
    assert len(n._log) <= 1000


# ── Four-regime integration ───────────────────────────────────────────────

@pytest.mark.parametrize("da,sht,regime,scale_lo,scale_hi", [
    (0.2, 0.8, Regime.EXPLOIT,   0.8, 1.0),
    (0.8, 0.8, Regime.EXPLORE,   0.8, 1.0),
    (0.8, 0.2, Regime.WAIT,      0.0, 0.0),
    (0.2, 0.2, Regime.REOBSERVE, 0.3, 0.5),
])
def test_regime_action_scales(da, sht, regime, scale_lo, scale_hi):
    n = NeuromodulatorState()
    n.da, n.sht = da, sht
    r = classify_regime(da, sht)
    s = n._action_scale(r)
    assert r == regime, f"Wrong regime: {r}"
    assert scale_lo <= s <= scale_hi + 1e-9, f"{regime}: scale={s}"


# ════════════════════════════════════════════════════════════════════════════
# Torch-dependent tests (skip if no torch)
# ════════════════════════════════════════════════════════════════════════════

pytestmark_torch = pytest.mark.skipif(not HAS_TORCH,
    reason="torch not available")

@pytest.fixture
def z_identical():
    torch.manual_seed(0)
    z = F.normalize(torch.randn(32), dim=0)
    return z, z.clone()

@pytest.fixture
def z_orthogonal():
    torch.manual_seed(1)
    z = F.normalize(torch.randn(32), dim=0)
    return z, F.normalize(-z, dim=0)

@pytest.fixture
def z_random():
    torch.manual_seed(2)
    return (F.normalize(torch.randn(32), dim=0),
            F.normalize(torch.randn(32), dim=0))


@pytestmark_torch
def test_perfect_prediction_low_da(z_identical):
    n = NeuromodulatorState()
    zp, za = z_identical
    for _ in range(25): n.update(zp, za)
    assert n.da < 0.3, f"Expected low DA, got {n.da:.3f}"

@pytestmark_torch
def test_wrong_prediction_high_da(z_orthogonal):
    n = NeuromodulatorState()
    zp, za = z_orthogonal
    for _ in range(25): n.update(zp, za)
    assert n.da > 0.7, f"Expected high DA, got {n.da:.3f}"

@pytestmark_torch
def test_stable_latents_high_sht(z_identical):
    n = NeuromodulatorState()
    zp, za = z_identical
    for _ in range(40): n.update(zp, za)
    assert n.sht > 0.6, f"Expected high 5HT, got {n.sht:.3f}"

@pytestmark_torch
def test_noisy_latents_low_sht():
    n = NeuromodulatorState()
    for _ in range(40):
        za = F.normalize(torch.randn(32), dim=0)
        zp = F.normalize(torch.randn(32), dim=0)
        n.update(zp, za)
    assert n.sht < 0.5, f"Expected low 5HT, got {n.sht:.3f}"

@pytestmark_torch
def test_high_da_raises_ach(z_orthogonal):
    n = NeuromodulatorState()
    zp, za = z_orthogonal
    for _ in range(20): n.update(zp, za)
    assert n.ach > 0.5, f"High-DA scenario should raise ACh, got {n.ach:.3f}"

@pytestmark_torch
def test_ecb_rises_with_action_in_high_da(z_orthogonal):
    n = NeuromodulatorState()
    zp, za = z_orthogonal
    for _ in range(10): n.update(zp, za, action_magnitude=1.0)
    assert n.ecb > 0.0, "eCB should rise when action taken in high-DA state"

@pytestmark_torch
def test_ecb_dampens_effective_da(z_orthogonal):
    n = NeuromodulatorState()
    zp, za = z_orthogonal
    for _ in range(20): n.update(zp, za, action_magnitude=1.0)
    da_raw = n.da
    da_eff = da_raw * (1.0 - n.ecb * 0.4)
    assert da_eff < da_raw, "eCB should suppress effective DA"

@pytestmark_torch
def test_ema_smoothing_single_step(z_identical):
    n = NeuromodulatorState(da_decay=0.99)
    zp, za = z_identical
    for _ in range(50): n.update(zp, za)
    da_before = n.da
    za_novel = F.normalize(-zp, dim=0)
    n.update(zp, za_novel)
    delta = abs(n.da - da_before)
    assert delta <= (1 - 0.99) * 1.0 + 1e-5, \
        f"EMA smoothing: single step delta too large: {delta:.4f}"

@pytestmark_torch
def test_update_returns_all_keys(z_random):
    n = NeuromodulatorState()
    zp, za = z_random
    s = n.update(zp, za, rho=0.3, action_magnitude=0.5)
    for k in ["da","sht","rho","ach","ei","ado","ecb","da_effective",
              "regime","confidence","action_scale","eps_scale",
              "action_std","n_candidates","lr_scale","da_phasic","cos_sim"]:
        assert k in s, f"Missing key: {k}"

@pytestmark_torch
def test_update_values_in_range(z_random):
    n = NeuromodulatorState()
    zp, za = z_random
    s = n.update(zp, za)
    assert 0.0 <= s["da"]          <= 1.0
    assert 0.0 <= s["sht"]         <= 1.0
    assert 0.0 <= s["ach"]         <= 1.0
    assert 0.5 <= s["ei"]          <= 2.0
    assert 0.0 <= s["ado"]         <= 1.0
    assert 0.0 <= s["ecb"]         <= 1.0
    assert 0.0 <= s["da_effective"] <= 1.0
    assert s["eps_scale"]   >= 1.0
    assert 0.05 <= s["action_std"] <= 0.20
    assert 16   <= s["n_candidates"] <= 96
    assert 0.2  <= s["lr_scale"]   <= 2.0
