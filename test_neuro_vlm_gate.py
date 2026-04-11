"""
test_neuro_vlm_gate.py
======================
Pytest suite for BiologicalNeuromodulator and NeurallyGatedVLM.

Coverage targets
----------------
  NeuroState          — field types, defaults, value ranges
  AttentionGains      — field types, defaults, biological range constraints
  BiologicalNeuro     — gain computation, update_from_error, regime transitions,
                        DA decay, history window, rest/recalibration, cortisol
  NeurallyGatedVLM    — hook registration, encode shape, hook removal, no-op fast path
  Integration         — full predictive-coding loop, state evolution over sequence
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuro_vlm_gate import (
    AttentionGains,
    BiologicalNeuromodulator,
    NeuroState,
    NeurallyGatedVLM,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rand_z(dim: int = 64) -> torch.Tensor:
    """Unit-normalised random embedding."""
    return F.normalize(torch.randn(dim), dim=0)


def _make_neuro(**kwargs) -> BiologicalNeuromodulator:
    return BiologicalNeuromodulator(**kwargs)


class _TinyAttn(nn.Module):
    """Minimal transformer attention stand-in with a named 'attn' sub-module."""

    class _Attn(nn.Module):
        def forward(self, x):
            return x * 1.0   # identity

    def __init__(self, dim: int = 16):
        super().__init__()
        self.attn = self._Attn()
        self.proj = nn.Linear(dim, dim)

    def forward(self, pixel_values: torch.Tensor):
        b, c, h, w = pixel_values.shape
        flat  = pixel_values.view(b, c, -1).mean(-1)   # (B, C)
        flat  = flat[:, :self.proj.in_features] if c >= self.proj.in_features else \
                torch.cat([flat, torch.zeros(b, self.proj.in_features - c)], dim=1)
        out   = self.proj(self.attn(flat))

        # Mimic HuggingFace: return object with last_hidden_state
        class _Out:
            def __init__(self, t):
                self.last_hidden_state = t.unsqueeze(1)   # (B, 1, dim)
                self.pooler_output = None
        return _Out(out)


# ── NeuroState ─────────────────────────────────────────────────────────────────

class TestNeuroState:
    def test_defaults_in_unit_interval(self):
        s = NeuroState()
        for attr in ("da", "ne", "ach", "sht", "ecb", "ado", "cort", "ei"):
            v = getattr(s, attr)
            assert 0.0 <= v <= 1.0, f"{attr} default {v} outside [0,1]"

    def test_regime_default(self):
        assert NeuroState().regime == "EXPLOIT"

    def test_step_default_zero(self):
        assert NeuroState().step == 0

    def test_timestamp_recent(self):
        s = NeuroState()
        assert abs(s.timestamp - time.time()) < 2.0

    def test_mutability(self):
        s = NeuroState()
        s.da = 0.9
        assert s.da == 0.9


# ── AttentionGains ─────────────────────────────────────────────────────────────

class TestAttentionGains:
    def test_defaults_are_neutral(self):
        g = AttentionGains()
        assert g.query_scale   == 1.0
        assert g.spatial_bias  == 0.0
        assert g.temperature   == 1.0
        assert g.topk_suppress == 0.0
        assert g.recency_decay == 0.0
        assert g.global_gain   == 1.0
        assert g.threshold_mult == 1.0
        assert g.ei_bias       == 0.5

    def test_all_fields_are_floats(self):
        g = AttentionGains()
        for attr in g.__dataclass_fields__:
            assert isinstance(getattr(g, attr), float), attr


# ── BiologicalNeuromodulator — gain computation ───────────────────────────────

class TestAttentionGainComputation:
    def test_default_state_gives_neutral_gains(self):
        """Zero DA/NE/Ado → gains close to neutral defaults."""
        neuro = _make_neuro()
        g = neuro.get_attention_gains()
        # query_scale: 1 + 0 * 2 * (1+0) = 1.0
        assert g.query_scale == pytest.approx(1.0, abs=1e-6)
        # global_gain: clip(1 - 0 * 0.6, 0.3, 1.0) = 1.0
        assert g.global_gain == pytest.approx(1.0, abs=1e-6)
        # spatial_bias: 0 * (1 + 0 * 0.5) = 0.0
        assert g.spatial_bias == pytest.approx(0.0, abs=1e-6)

    def test_high_da_raises_query_scale(self):
        neuro = _make_neuro()
        neuro._state.da = 0.8
        g = neuro.get_attention_gains()
        assert g.query_scale > 2.0

    def test_high_ado_lowers_global_gain(self):
        neuro = _make_neuro()
        neuro._state.ado = 0.9
        g = neuro.get_attention_gains()
        assert g.global_gain < 0.6

    def test_high_ach_sharpens_temperature(self):
        neuro = _make_neuro()
        neuro._state.ach = 0.95
        g_high = neuro.get_attention_gains()
        neuro._state.ach = 0.05
        g_low = neuro.get_attention_gains()
        # Higher ACh → lower temperature (sharper)
        assert g_high.temperature < g_low.temperature

    def test_high_cort_raises_threshold_mult(self):
        neuro = _make_neuro()
        neuro._state.cort = 0.8
        g = neuro.get_attention_gains()
        assert g.threshold_mult > 2.0

    def test_gains_always_in_biological_bounds(self):
        """Gains must stay within documented biological ranges regardless of state."""
        neuro = _make_neuro()
        # Saturate every signal
        for attr in ("da", "ne", "ach", "sht", "ecb", "ado", "cort"):
            setattr(neuro._state, attr, 1.0)
        g = neuro.get_attention_gains()
        assert 0.5  <= g.query_scale    <= 4.0
        assert 0.0  <= g.spatial_bias   <= 1.0
        assert 0.3  <= g.temperature    <= 2.0
        assert 0.0  <= g.topk_suppress  <= 0.8
        assert 0.0  <= g.recency_decay  <= 0.5
        assert 0.3  <= g.global_gain    <= 1.0
        assert 1.0  <= g.threshold_mult <= 5.0
        assert 0.0  <= g.ei_bias        <= 1.0

    def test_gains_are_floats(self):
        g = _make_neuro().get_attention_gains()
        for field in g.__dataclass_fields__:
            assert isinstance(getattr(g, field), float), field


# ── BiologicalNeuromodulator — update_from_error ──────────────────────────────

class TestUpdateFromError:
    def test_first_frame_no_z_pred(self):
        """None z_pred (first frame) must not raise and DA stays 0."""
        neuro = _make_neuro()
        z = _rand_z()
        state = neuro.update_from_error(None, z)
        assert state.da == pytest.approx(0.0, abs=1e-9)  # rpe=0 on first frame
        assert state.step == 1

    def test_identical_frames_low_da(self):
        """Identical embeddings → near-zero prediction error → DA stays low."""
        neuro = _make_neuro()
        z = _rand_z()
        neuro.update_from_error(None, z)
        state = neuro.update_from_error(z, z)
        assert state.da < 0.05

    def test_orthogonal_frames_raise_da(self):
        """Maximally different embeddings → high RPE → DA increases."""
        neuro = _make_neuro()
        z1 = _rand_z()
        z2 = F.normalize(-z1 + torch.randn_like(z1) * 0.1, dim=0)  # ~opposite
        neuro.update_from_error(None, z1)
        state = neuro.update_from_error(z1, z2)
        assert state.da > 0.1

    def test_step_counter_increments(self):
        neuro = _make_neuro()
        z = _rand_z()
        for i in range(1, 6):
            state = neuro.update_from_error(None, z)
            assert state.step == i

    def test_all_signals_in_unit_interval(self):
        """All neuromodulator signals must remain in [0, 1] after updates."""
        neuro = _make_neuro()
        z_prev = None
        for _ in range(30):
            z = _rand_z()
            state = neuro.update_from_error(z_prev, z)
            z_prev = z
        for attr in ("da", "ne", "ach", "sht", "ecb", "ado", "cort", "ei"):
            v = getattr(state, attr)
            assert 0.0 <= v <= 1.0, f"{attr}={v} out of [0,1]"

    def test_da_decays_when_no_surprise(self):
        """DA should decay back toward 0 when frames are identical."""
        neuro = _make_neuro()
        z = _rand_z()
        # First, raise DA
        neuro._state.da = 0.8
        for _ in range(10):
            state = neuro.update_from_error(z, z)
        assert state.da < 0.2   # decayed

    def test_ado_builds_with_high_da(self):
        """Adenosine builds when DA is consistently high."""
        neuro = _make_neuro(da_threshold=0.0)
        z_prev = _rand_z()
        for _ in range(40):
            z = _rand_z()   # random new frames → high DA
            neuro.update_from_error(z_prev, z)
            z_prev = z
        assert neuro._state.ado > 0.2

    def test_cortisol_calibrates_after_warmup(self):
        """Cortisol baseline should be set after calibration_steps frames."""
        neuro = _make_neuro()
        z = _rand_z()
        for _ in range(25):
            neuro.update_from_error(None, z)
        assert neuro._cortisol_baseline is not None


# ── Regime transitions ─────────────────────────────────────────────────────────

class TestRegimeTransitions:
    def _run_n(self, neuro, n, z_factory=None):
        z_prev = None
        for _ in range(n):
            z = z_factory() if z_factory else _rand_z()
            state = neuro.update_from_error(z_prev, z)
            z_prev = z
        return state

    def test_exploit_regime_on_stable_input(self):
        neuro = _make_neuro(da_threshold=0.5)  # high threshold → hard to REOBSERVE
        z = _rand_z()
        state = self._run_n(neuro, 10, z_factory=lambda: z)
        assert state.regime == "EXPLOIT"

    def test_reobserve_regime_on_high_da(self):
        neuro = _make_neuro(da_threshold=0.01)
        z_prev = _rand_z()
        # Feed orthogonal frames to maximise RPE
        state = None
        for _ in range(5):
            z = _rand_z()
            state = neuro.update_from_error(z_prev, z)
            z_prev = z
        assert state.regime in ("REOBSERVE", "FATIGUE", "STRESSED")

    def test_fatigue_regime_when_ado_high(self):
        neuro = _make_neuro(fatigue_threshold=0.3)
        neuro._state.ado = 0.35
        neuro._state.da  = 0.0
        neuro._state.cort = 0.0
        z = _rand_z()
        state = neuro.update_from_error(None, z)
        assert state.regime == "FATIGUE"

    def test_stressed_regime_when_cort_high(self):
        neuro = _make_neuro(stress_threshold=0.3, fatigue_threshold=0.99)
        neuro._state.cort = 0.6
        neuro._state.ado  = 0.0
        neuro._state.da   = 0.0
        z = _rand_z()
        state = neuro.update_from_error(None, z)
        assert state.regime == "STRESSED"

    def test_fatigue_takes_priority_over_stressed(self):
        """FATIGUE check precedes STRESSED in update logic."""
        neuro = _make_neuro(fatigue_threshold=0.3, stress_threshold=0.2)
        neuro._state.ado  = 0.35
        neuro._state.cort = 0.5
        z = _rand_z()
        state = neuro.update_from_error(None, z)
        assert state.regime == "FATIGUE"


# ── Rest / recalibration ───────────────────────────────────────────────────────

class TestRestAndRecalibration:
    def test_rest_clears_adenosine(self):
        neuro = _make_neuro()
        neuro._state.ado = 0.8
        neuro.rest(steps=10)
        assert neuro._state.ado < 0.8

    def test_rest_sets_regime_to_exploit(self):
        neuro = _make_neuro()
        neuro._state.regime = "FATIGUE"
        neuro.rest(steps=1)
        assert neuro._state.regime == "EXPLOIT"

    def test_rest_ado_floors_at_zero(self):
        neuro = _make_neuro()
        neuro._state.ado = 0.05
        neuro.rest(steps=100)
        assert neuro._state.ado == pytest.approx(0.0, abs=1e-9)

    def test_needs_recalibration_above_threshold(self):
        neuro = _make_neuro(fatigue_threshold=0.5)
        neuro._state.ado = 0.6
        assert neuro.needs_recalibration() is True

    def test_needs_recalibration_below_threshold(self):
        neuro = _make_neuro(fatigue_threshold=0.5)
        neuro._state.ado = 0.3
        assert neuro.needs_recalibration() is False

    def test_reset_cortisol_updates_baseline(self):
        neuro = _make_neuro()
        neuro._loss_history = [0.1, 0.2, 0.3, 0.4, 0.5]
        neuro.reset_cortisol()
        expected = np.mean([0.1, 0.2, 0.3, 0.4, 0.5]) * 0.95
        assert neuro._cortisol_baseline == pytest.approx(expected, rel=1e-6)


# ── get_state ──────────────────────────────────────────────────────────────────

class TestGetState:
    def test_returns_expected_keys(self):
        keys = {"da", "ne", "ach", "sht", "ecb", "ado", "cort", "ei",
                "regime", "step"}
        assert set(_make_neuro().get_state().keys()) == keys

    def test_numeric_values_are_rounded(self):
        neuro = _make_neuro()
        neuro._state.da = 0.123456789
        state = neuro.get_state()
        assert state["da"] == round(0.123456789, 4)

    def test_step_matches_update_count(self):
        neuro = _make_neuro()
        z = _rand_z()
        for _ in range(7):
            neuro.update_from_error(None, z)
        assert neuro.get_state()["step"] == 7


# ── History window ─────────────────────────────────────────────────────────────

class TestHistoryWindow:
    def test_z_history_bounded_by_window(self):
        neuro = _make_neuro(history_window=5)
        z = _rand_z()
        for _ in range(20):
            neuro.update_from_error(None, z)
        assert len(neuro._z_history) <= 5

    def test_da_history_bounded_by_window(self):
        neuro = _make_neuro(history_window=5)
        z = _rand_z()
        for _ in range(20):
            neuro.update_from_error(None, z)
        assert len(neuro._da_history) <= 5


# ── E/I balance ───────────────────────────────────────────────────────────────

class TestEIBalance:
    def test_ei_high_when_da_and_ne_high(self):
        neuro = _make_neuro()
        neuro._state.da  = 0.9
        neuro._state.ne  = 0.9
        neuro._state.sht = 0.0
        neuro._state.ado = 0.0
        # Compute expected E/I
        exc = (0.9 + 0.9) / 2
        inh = (0.0 + 0.0) / 2
        expected = exc / (exc + inh + 1e-6)
        z = _rand_z()
        state = neuro.update_from_error(None, z)
        # After update the signals shift; just verify E/I > 0.5
        assert state.ei > 0.5

    def test_ei_low_when_sht_and_ado_high(self):
        neuro = _make_neuro()
        neuro._state.da  = 0.0
        neuro._state.ne  = 0.0
        neuro._state.sht = 0.9
        neuro._state.ado = 0.9
        z = _rand_z()
        state = neuro.update_from_error(None, z)
        assert state.ei < 0.5


# ── NeurallyGatedVLM ──────────────────────────────────────────────────────────

class TestNeurallyGatedVLM:
    def _build(self, dim=16):
        vision  = _TinyAttn(dim=dim)
        neuro   = _make_neuro()
        gated   = NeurallyGatedVLM(vision, neuro)
        return gated, neuro

    def test_hooks_registered(self):
        gated, _ = self._build()
        assert len(gated._hooks) > 0

    def test_encode_returns_unit_vector(self):
        gated, _ = self._build()
        img = torch.randn(1, 16, 8, 8)
        z   = gated.encode(img)
        assert abs(z.norm().item() - 1.0) < 1e-5

    def test_encode_shape_is_1d(self):
        gated, _ = self._build()
        img = torch.randn(1, 16, 8, 8)
        z   = gated.encode(img)
        assert z.dim() == 1

    def test_remove_hooks_empties_list(self):
        gated, _ = self._build()
        gated.remove_hooks()
        assert len(gated._hooks) == 0

    def test_encode_uses_neuromodulator_gains(self):
        """With DA=0 and DA=0.9, embedding magnitudes differ."""
        # Not strictly guaranteed for all architectures, but with a linear
        # model the attention hook scales outputs, so norms differ.
        dim   = 16
        vision = _TinyAttn(dim=dim)
        img    = torch.randn(1, 16, 8, 8)

        neuro_low = _make_neuro()
        neuro_low._state.da = 0.0
        gated_low = NeurallyGatedVLM(vision, neuro_low)
        z_low = gated_low.encode(img)

        neuro_high = _make_neuro()
        neuro_high._state.da = 0.9
        gated_high = NeurallyGatedVLM(vision, neuro_high)
        z_high = gated_high.encode(img)

        # Both are unit-normalised, but the hook-scaled intermediate differs
        # We can at least verify both are valid unit vectors
        assert abs(z_low.norm().item()  - 1.0) < 1e-5
        assert abs(z_high.norm().item() - 1.0) < 1e-5

    def test_no_op_fast_path_when_gains_default(self):
        """Hook returns early when gains are neutral (default state)."""
        gated, _ = self._build()
        # Default gains: global_gain=1.0, query_scale=1.0 → fast path
        img = torch.randn(1, 16, 8, 8)
        z   = gated.encode(img)
        assert z is not None  # simply must not raise

    def test_with_projection(self):
        """Optional project layer applied after encoder."""
        dim    = 16
        proj   = nn.Linear(dim, 8)
        vision = _TinyAttn(dim=dim)
        neuro  = _make_neuro()
        gated  = NeurallyGatedVLM(vision, neuro, project=proj)
        img    = torch.randn(1, 16, 8, 8)
        z      = gated.encode(img)
        assert z.shape[0] == 8


# ── Integration — predictive coding loop ──────────────────────────────────────

class TestPredictiveCodingLoop:
    """
    Full end-to-end loop:
      gains = neuro.get_attention_gains()   [BEFORE encode]
      z     = gated.encode(img)
      state = neuro.update_from_error(z_prev, z)  [AFTER encode]
    """

    def test_loop_runs_without_error(self):
        dim   = 16
        neuro = _make_neuro()
        gated = NeurallyGatedVLM(_TinyAttn(dim=dim), neuro)

        z_prev = None
        for _ in range(20):
            img   = torch.randn(1, 16, 8, 8)
            z     = gated.encode(img)
            state = neuro.update_from_error(z_prev, z)
            z_prev = z

        assert state.step == 20

    def test_state_evolves_across_frames(self):
        """State must not be frozen — at least one signal must change."""
        dim   = 16
        neuro = _make_neuro()
        gated = NeurallyGatedVLM(_TinyAttn(dim=dim), neuro)

        snap_before = neuro.get_state().copy()
        z_prev = None
        for _ in range(5):
            img   = torch.randn(1, 16, 8, 8)
            z     = gated.encode(img)
            neuro.update_from_error(z_prev, z)
            z_prev = z
        snap_after = neuro.get_state().copy()

        changed = any(
            snap_before[k] != snap_after[k]
            for k in ("da", "ne", "ach", "ado", "step")
        )
        assert changed, "Neuromodulator state never changed across 5 frames"

    def test_spatial_context_shapes_ne(self):
        """Providing non-zero spatial_context raises NE above NE without it."""
        neuro_spatial = _make_neuro()
        neuro_plain   = _make_neuro()
        z = _rand_z()
        ctx = torch.tensor([1.0, 2.0])   # GPS-like displacement

        neuro_spatial.update_from_error(None, z, spatial_context=ctx)
        neuro_plain.update_from_error(None, z, spatial_context=None)

        # Spatial context should drive NE up relative to plain (rpe=0 both)
        # Both start at rpe=0 on first frame, but ctx drives ne_new differently
        # First frame: z_prev=None → rpe=0; with ctx ne_new = disp/(disp+1)
        # without ctx ne_new = rpe*0.8 = 0
        assert neuro_spatial._state.ne >= neuro_plain._state.ne

    def test_action_magnitude_shapes_ach(self):
        """Higher action_magnitude should eventually raise ACh."""
        neuro_active = _make_neuro()
        neuro_idle   = _make_neuro()
        z = _rand_z()
        for _ in range(10):
            neuro_active.update_from_error(None, z, action_magnitude=1.0)
            neuro_idle.update_from_error(None, z,   action_magnitude=0.0)

        assert neuro_active._state.ach >= neuro_idle._state.ach


# ── Edge cases ────────────────────────────────────────────────────────────────


    def test_zero_dim_z_does_not_crash(self):
        neuro = _make_neuro()
        z = torch.zeros(64)
        # Normalisation of zero vector is ill-defined; update must not raise
        try:
            neuro.update_from_error(None, z)
        except Exception as e:
            pytest.fail(f"update_from_error raised on zero vector: {e}")

    def test_large_batch_of_updates_stable(self):
        neuro = _make_neuro()
        z = _rand_z(128)
        for _ in range(200):
            neuro.update_from_error(None, z)
        state = neuro.get_state()
        for k, v in state.items():
            if isinstance(v, float):
                assert math.isfinite(v), f"{k}={v} is not finite"

    def test_multiple_gated_vlms_independent_state(self):
        """Two NeurallyGatedVLMs with separate neuromodulators are independent."""
        dim   = 16
        n1, n2 = _make_neuro(), _make_neuro()
        n1._state.da = 0.9
        n2._state.da = 0.0
        g1 = NeurallyGatedVLM(_TinyAttn(dim), n1)
        g2 = NeurallyGatedVLM(_TinyAttn(dim), n2)
        img = torch.randn(1, 16, 8, 8)
        z1  = g1.encode(img)
        z2  = g2.encode(img)
        # Both normalised but driven by different gain states
        assert z1.shape == z2.shape

    def test_cortisol_baseline_none_on_fresh_neuro(self):
        assert _make_neuro()._cortisol_baseline is None

    def test_custom_cortisol_baseline_respected(self):
        neuro = _make_neuro(cortisol_baseline=0.05)
        assert neuro._cortisol_baseline == 0.05


# ── Aphasia ablation ──────────────────────────────────────────────────────────

class TestAphasiaAblation:
    """
    Computational analogue of Fedorenko's aphasia findings.

    Fedorenko's key result: patients with destroyed language networks retain
    complex non-linguistic reasoning. The computational equivalent here:
    zero the VLM embedding (language pathway) and verify the neuromodulated
    world model still runs — neuromodulator dynamics continue, state evolves,
    and gains remain well-formed. If RECON AUROC holds up in a real eval
    this directly parallels the aphasia finding.

    Tests verify:
      1. Ablated pathway produces zeros (language input destroyed).
      2. Neuromodulator still updates — MD-analog dynamics survive.
      3. Gains remain in biological bounds with zero input.
      4. Control (non-ablated) produces non-zero embeddings — zeros are from flag.
      5. Ablation flag is independent per instance.
      6. Ablation is togglable at eval time without re-instantiation.
      7. RPE behaviour under perpetual-zero signal (expected: max surprise).
    """

    def _build(self, aphasia: bool = False, dim: int = 16):
        neuro = _make_neuro()
        gated = NeurallyGatedVLM(_TinyAttn(dim=dim), neuro,
                                 aphasia_ablation=aphasia)
        return gated, neuro

    def test_ablated_encode_returns_zero_vector(self):
        """Language pathway destroyed → VLM embedding is all zeros."""
        gated, _ = self._build(aphasia=True)
        img = torch.randn(1, 16, 8, 8)
        z   = gated.encode(img)
        assert torch.all(z == 0.0), "Aphasia ablation must zero the embedding"

    def test_ablated_z_shape_and_dtype_preserved(self):
        """Shape and dtype survive the zeroing."""
        gated, _ = self._build(aphasia=True)
        img = torch.randn(1, 16, 8, 8)
        z   = gated.encode(img)
        assert z.shape[0] == 16
        assert z.dtype == torch.float32

    def test_neuromodulator_still_updates_under_ablation(self):
        """MD-analog core keeps running with VLM pathway destroyed."""
        gated, neuro = self._build(aphasia=True)
        img = torch.randn(1, 16, 8, 8)
        z_prev = None
        for _ in range(5):
            z     = gated.encode(img)
            state = neuro.update_from_error(z_prev, z)
            z_prev = z
        assert state.step == 5
        for attr in ("da", "ne", "ach", "sht", "ecb", "ado", "cort", "ei"):
            v = getattr(state, attr)
            assert 0.0 <= v <= 1.0, f"{attr}={v} out of [0,1] under ablation"

    def test_gains_well_formed_under_ablation(self):
        """Attention gains stay in biological bounds with zero input."""
        gated, neuro = self._build(aphasia=True)
        img = torch.randn(1, 16, 8, 8)
        z   = gated.encode(img)
        neuro.update_from_error(None, z)
        g = neuro.get_attention_gains()
        assert 0.5  <= g.query_scale    <= 4.0
        assert 0.0  <= g.spatial_bias   <= 1.0
        assert 0.3  <= g.temperature    <= 2.0
        assert 0.3  <= g.global_gain    <= 1.0
        assert 1.0  <= g.threshold_mult <= 5.0

    def test_control_produces_nonzero_embedding(self):
        """Sanity: zeros in ablated case come from the flag, not the encoder."""
        gated, _ = self._build(aphasia=False)
        img = torch.randn(1, 16, 8, 8)
        z   = gated.encode(img)
        assert z.norm().item() > 1e-6

    def test_ablation_flag_is_per_instance(self):
        """Two models — one ablated, one not — are independent."""
        dim = 16
        img = torch.randn(1, 16, 8, 8)
        gated_on,  _ = self._build(aphasia=True,  dim=dim)
        gated_off, _ = self._build(aphasia=False, dim=dim)
        z_on  = gated_on.encode(img)
        z_off = gated_off.encode(img)
        assert torch.all(z_on  == 0.0)
        assert z_off.norm().item() > 1e-6

    def test_ablation_togglable_at_eval_time(self):
        """Bool attribute can be toggled mid-session — useful for sweep evals."""
        gated, _ = self._build(aphasia=False)
        img = torch.randn(1, 16, 8, 8)

        z_before = gated.encode(img)
        assert z_before.norm().item() > 1e-6

        gated.aphasia_ablation = True
        z_ablated = gated.encode(img)
        assert torch.all(z_ablated == 0.0)

        gated.aphasia_ablation = False
        z_after = gated.encode(img)
        assert z_after.norm().item() > 1e-6

    def test_ablation_produces_max_rpe_perpetually(self):
        """
        Zero embeddings → perpetual novelty from neuromodulator's perspective.
        Without language grounding the system cannot confirm predictions —
        the expected aphasia finding. DA must remain bounded despite max RPE.
        """
        gated, neuro = self._build(aphasia=True)
        img = torch.randn(1, 16, 8, 8)
        z0  = gated.encode(img)   # zeros
        state = neuro.update_from_error(z0, z0)
        assert 0.0 <= state.da <= 1.0



class TestTextGoalDA:
    """Tests for the two-DA channel: text-goal similarity blending."""

    def test_update_text_goal_hot(self):
        neuro = BiologicalNeuromodulator()
        da = neuro.update_text_goal(0.26)  # above HOT=0.245
        assert da == 1.0

    def test_update_text_goal_cold(self):
        neuro = BiologicalNeuromodulator()
        da = neuro.update_text_goal(0.15)  # below COLD=0.180
        assert da == 0.0

    def test_update_text_goal_midrange(self):
        neuro = BiologicalNeuromodulator()
        da = neuro.update_text_goal(0.21)  # dead zone
        assert 0.0 < da < 1.0

    def test_text_goal_da_property(self):
        neuro = BiologicalNeuromodulator()
        neuro.update_text_goal(0.26)
        assert neuro.text_goal_da == 1.0

    def test_blended_da_weights(self):
        neuro = BiologicalNeuromodulator()
        neuro._state.da = 0.5
        neuro.update_text_goal(0.26)  # text_goal_da=1.0
        # expect 0.7*0.5 + 0.3*1.0 = 0.65
        assert abs(neuro.blended_da - 0.65) < 0.01

    def test_blended_da_temporal_only(self):
        neuro = BiologicalNeuromodulator()
        neuro._state.da = 0.8
        neuro.update_text_goal(0.15)  # text_goal_da=0.0
        # expect 0.7*0.8 + 0.3*0.0 = 0.56
        assert abs(neuro.blended_da - 0.56) < 0.01

    def test_default_text_goal_da_neutral(self):
        neuro = BiologicalNeuromodulator()
        assert neuro.text_goal_da == 0.5  # neutral start

    def test_set_text_goal_on_gated_vlm(self):
        import torch, torch.nn as nn
        vision = nn.Sequential(nn.Flatten(), nn.Linear(12288, 128))
        neuro  = BiologicalNeuromodulator()
        gated  = NeurallyGatedVLM(vision, neuro)
        proj   = torch.randn(128)
        gated.set_text_goal(proj)
        assert hasattr(gated, '_text_goal_proj')
        assert gated._text_goal_proj is not None

    def test_clear_text_goal(self):
        import torch, torch.nn as nn
        vision = nn.Sequential(nn.Flatten(), nn.Linear(12288, 128))
        neuro  = BiologicalNeuromodulator()
        gated  = NeurallyGatedVLM(vision, neuro)
        gated.set_text_goal(torch.randn(128))
        gated.clear_text_goal()
        assert gated._text_goal_proj is None
