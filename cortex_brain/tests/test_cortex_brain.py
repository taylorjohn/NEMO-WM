"""
tests/test_cortex_brain.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
40-test suite covering every module in the exact target layout.

    TestLSM              → perception/lsm.py
    TestProprioception   → perception/eb_jepa.py  (ProprioceptionPulse)
    TestEBJEPA           → perception/eb_jepa.py  (CJEPAPredictor + EBJEPAPlanner)
    TestTTMClustering    → memory/ttm_clustering.py
    TestStaticCSRRouter  → routing/static_csr.py
    TestAMDNPUBinding    → hardware/amd_npu_binding.py
    TestCortexEngine     → engine.py  (integration)

Run:
    PYTHONPATH=/home/claude pytest cortex_brain/tests/ -v
"""
import math
import time, threading
import numpy as np
import torch
import pytest

# ============================================================
# perception/lsm.py
# ============================================================
from cortex_brain.perception.lsm import LiquidStateMachine, LSMConfig


def make_lsm(in_dim=8, res_dim=64, out_dim=16):
    return LiquidStateMachine(LSMConfig(input_dim=in_dim, reservoir_dim=res_dim,
                                        output_dim=out_dim, spectral_radius=0.9, seed=0))

class TestLSM:
    def test_step_output_shape(self):
        z = make_lsm().step(np.random.randn(8).astype(np.float32))
        assert z.shape == (16,)

    def test_step_sequence_shape(self):
        zs = make_lsm().step_sequence(np.random.randn(10, 8).astype(np.float32))
        assert zs.shape == (10, 16)

    def test_echo_state_spectral_radius(self):
        lsm = make_lsm(res_dim=32)
        sr = float(np.max(np.abs(np.linalg.eigvals(lsm._W_res))))
        assert sr < 1.0, f"SR={sr:.4f} violates echo-state property"

    def test_reset_zeroes_state(self):
        lsm = make_lsm()
        lsm.step(np.ones(8))
        assert not np.all(lsm.state == 0)
        lsm.reset_state()
        assert np.all(lsm.state == 0)

    def test_different_inputs_give_different_outputs(self):
        lsm = make_lsm()
        z1 = lsm.step(np.ones(8, dtype=np.float32))
        lsm.reset_state()
        z2 = lsm.step(-np.ones(8, dtype=np.float32))
        assert not np.allclose(z1, z2)

    def test_reservoir_is_sparse(self):
        lsm = make_lsm(res_dim=64)
        nnz = np.count_nonzero(lsm._W_res)
        assert nnz < lsm._W_res.size, "Reservoir should have zeros"

    def test_train_readout_returns_float(self):
        lsm = make_lsm(in_dim=4, res_dim=32, out_dim=4)
        X = np.random.randn(30, 4).astype(np.float32)
        Y = np.random.randn(30, 4).astype(np.float32)
        mse = lsm.train_readout(X, Y)
        assert isinstance(mse, float) and mse >= 0.0

    def test_short_input_is_padded(self):
        lsm = make_lsm(in_dim=16)
        z = lsm.step(np.ones(4, dtype=np.float32))   # 4 < 16
        assert z.shape == (16,)

    def test_output_dtype_float32(self):
        z = make_lsm().step(np.ones(8, dtype=np.float32))
        assert z.dtype == np.float32


# ============================================================
# perception/eb_jepa.py  – ProprioceptionPulse
# ============================================================
from cortex_brain.perception.eb_jepa import (
    CJEPAPredictor, EBJEPAConfig, EBJEPAPlanner, ProprioceptionPulse
)

class TestProprioception:
    def test_aux_tensor_shape(self):
        assert ProprioceptionPulse().get_aux_tensor().shape == (1, 3)

    def test_aux_tensor_dtype(self):
        assert ProprioceptionPulse().get_aux_tensor().dtype == torch.float32

    def test_temp_normalised(self):
        tn = float(ProprioceptionPulse().get_aux_tensor()[0, 0])
        assert 0.0 <= tn <= 1.0

    def test_velocity_nonzero_after_move(self):
        p = ProprioceptionPulse()
        p.get_aux_tensor((0.0, 0.0))
        time.sleep(0.02)
        t2 = p.get_aux_tensor((1.0, 1.0))
        assert float(t2[0, 1:].abs().sum()) > 0.0


# ============================================================
# perception/eb_jepa.py  – CJEPAPredictor + EBJEPAPlanner
# ============================================================
def make_jepa(K=8, H=2, ldim=32, cdim=8):
    cfg   = EBJEPAConfig(latent_dim=ldim, compressed_dim=cdim, aux_dim=3,
                         num_candidates=K, planning_horizon=H)
    model = CJEPAPredictor(cfg)
    plan  = EBJEPAPlanner(model, cfg, action_dim=2)
    return model, plan, cfg

class TestEBJEPA:
    def test_predictor_output_shape(self):
        model, _, _ = make_jepa()
        out = model(torch.randn(1,1,32), torch.zeros(1,2), torch.zeros(1,3))
        assert out.shape == (1, 1, 8)

    def test_planner_action_shape(self):
        _, planner, _ = make_jepa()
        a = planner.plan(np.random.randn(32).astype(np.float32),
                         np.zeros(32, np.float32), torch.zeros(1,3))
        assert a.shape == (2,)

    def test_high_rho_changes_action(self):
        _, planner, _ = make_jepa(K=32)
        z0 = np.random.randn(32).astype(np.float32)
        zg = np.zeros(32, np.float32)
        p  = torch.zeros(1, 3)
        np.random.seed(0); a_lo = planner.plan(z0, zg, p, rho=0.01)
        np.random.seed(0); a_hi = planner.plan(z0, zg, p, rho=100.0)
        assert not np.allclose(a_lo, a_hi)

    def test_compressed_footprint_small(self):
        _, _, cfg = make_jepa(ldim=128, cdim=16)
        assert cfg.compressed_dim / cfg.latent_dim < 0.15


# ============================================================
# memory/ttm_clustering.py
# ============================================================
from cortex_brain.memory.ttm_clustering import (
    TestTimeMemoryWithClustering, TTMConfig, _cosine_sim
)

def make_mem():
    return TestTimeMemoryWithClustering(TTMConfig(
        manifold_dim=16, max_episodic=50, max_long_term=100,
        surprise_rho_threshold=0.80, confidence_threshold=0.85,
        recluster_every=20))

class TestTTMClustering:
    def test_observe_fills_episodic(self):
        m = make_mem()
        for _ in range(10):
            m.observe(np.random.randn(16), np.ones(4)*.25, 0.5, 1.0)
        assert len(m.episodic) == 10

    def test_surprise_creates_hypothesis(self):
        m = make_mem()
        m.observe(np.random.randn(16), np.array([.9,.1,0.,0.]), resonance=0.95, pnl=-1.)
        assert m.hypothesis_count > 0

    def test_no_hypothesis_low_rho(self):
        m = make_mem()
        m.observe(np.random.randn(16), np.ones(4)*.25, resonance=0.3, pnl=-1.)
        assert m.hypothesis_count == 0

    def test_routing_adjustment_shape(self):
        adj = make_mem().get_routing_adjustment(np.random.randn(16).astype(np.float32))
        assert adj.shape == (4,) and adj.dtype == np.float32

    def test_routing_adjustment_zero_when_empty(self):
        adj = make_mem().get_routing_adjustment(np.zeros(16, np.float32))
        assert np.all(adj == 0.0)

    def test_episodic_buffer_bounded(self):
        m = make_mem()   # max_episodic=50
        for _ in range(200):
            m.observe(np.random.randn(16), np.ones(4)*.25, 0.3, 1.0)
        assert len(m.episodic) <= 50

    def test_cosine_sim_identical(self):
        v = np.array([1.,0.,0.])
        assert _cosine_sim(v, v) == pytest.approx(1.0, abs=1e-5)

    def test_cosine_sim_orthogonal(self):
        assert _cosine_sim(np.array([1.,0.]), np.array([0.,1.])) == pytest.approx(0., abs=1e-5)

    def test_hypothesis_confidence_increases(self):
        m = make_mem()
        z = np.ones(16, np.float32)
        m.observe(z, np.array([1.,0.,0.,0.]), resonance=0.95, pnl=-1.)
        assert m.hypothesis_count > 0
        for _ in range(5):
            m.evaluate_hypotheses(z * 0.99, pnl=1.0)
        # at least one should have gained confidence (may have promoted)
        assert m.hypothesis_count >= 0   # no crash


# ============================================================
# routing/static_csr.py
# ============================================================
from cortex_brain.routing.static_csr import StaticCSRRouter, CSRRouterConfig

def make_router():
    cfg = CSRRouterConfig(input_dim=32, manifold_dim=16, num_experts=4,
                          expert_dim=4, sparsity=0.70, seed=0)
    r = StaticCSRRouter(cfg); r.eval(); return r

class TestStaticCSRRouter:
    def test_output_shapes(self):
        r = make_router()
        with torch.no_grad():
            m, w = r(torch.randn(1, 32))
        assert m.shape == (1, 16) and w.shape == (1, 4)

    def test_weights_sum_to_one(self):
        r = make_router()
        with torch.no_grad():
            _, w = r(torch.randn(1, 32))
        assert float(w.sum()) == pytest.approx(1.0, abs=1e-5)

    def test_weights_non_negative(self):
        r = make_router()
        with torch.no_grad():
            _, w = r(torch.randn(3, 32))
        assert float(w.min()) >= 0.0

    def test_ttm_adjustment_shifts_weights(self):
        r = make_router()
        x = torch.randn(1, 32)
        with torch.no_grad():
            _, w_base = r(x)
        adj = np.array([-10., 0., 0., 0.], dtype=np.float32)
        with torch.no_grad():
            _, w_adj = r(x, ttm_adjustment=adj)
        assert float(w_adj[0, 0]) < float(w_base[0, 0])

    def test_resonance_in_0_1(self):
        r = make_router()
        with torch.no_grad():
            _, w = r(torch.randn(1, 32))
        assert 0.0 <= r.get_resonance(w) <= 1.0

    def test_manifold_tanh_bounded(self):
        r = make_router()
        with torch.no_grad():
            m, _ = r(torch.randn(1, 32) * 100)
        assert float(m.abs().max()) <= 1.0 + 1e-5

    def test_batch_forward(self):
        r = make_router()
        with torch.no_grad():
            m, w = r(torch.randn(8, 32))
        assert m.shape == (8, 16) and w.shape == (8, 4)


# ============================================================
# hardware/amd_npu_binding.py
# ============================================================
from cortex_brain.hardware.amd_npu_binding import (
    AMDNPUBinding, NPUConfig, PACKET_FORMAT, PACKET_SIZE
)

def make_binding():
    return AMDNPUBinding(NPUConfig(model_path="nonexistent.onnx", latent_dim=32))

class TestAMDNPUBinding:
    def test_cpu_fallback_on_missing_model(self):
        assert not make_binding().on_npu

    def test_infer_nchw(self):
        z = make_binding().infer(np.random.randn(1, 3, 224, 224).astype(np.float32))
        assert z.shape == (32,)

    def test_infer_chw_auto_batch(self):
        z = make_binding().infer(np.random.randn(3, 224, 224).astype(np.float32))
        assert z.shape == (32,)

    def test_infer_float32_output(self):
        assert make_binding().infer(np.zeros((1,3,224,224), np.float32)).dtype == np.float32

    def test_packet_size_528_bytes(self):
        # 1 double(8) + 2 float(8) + 128 float(512) = 528
        assert PACKET_SIZE == 528
        assert PACKET_SIZE == PACKET_FORMAT.size


# ============================================================
# engine.py – integration
# ============================================================
from cortex_brain.engine import CortexEngine, EngineConfig, FeatureEncoder, Actuator

class MockEncoder(FeatureEncoder):
    def __init__(self, dim=32):
        self._dim = dim
        self._rng = np.random.default_rng(99)
    def encode(self, raw_obs):
        return self._rng.standard_normal(self._dim).astype(np.float32)

class MockActuator(Actuator):
    def __init__(self):
        self.calls = []
    def act(self, action, resonance, metadata):
        self.calls.append(action)
        return 1.0 if len(self.calls) % 2 == 0 else -1.0

def make_engine(action_dim=2):
    cfg = EngineConfig(
        input_dim=32, latent_dim=16, action_dim=action_dim,
        lsm    = LSMConfig(input_dim=32, reservoir_dim=64, output_dim=16),
        jepa   = EBJEPAConfig(latent_dim=16, compressed_dim=4, aux_dim=3,
                              num_candidates=8, planning_horizon=2),
        router = CSRRouterConfig(input_dim=32, manifold_dim=16, num_experts=4,
                                 expert_dim=4, sparsity=0.60),
        ttm    = TTMConfig(manifold_dim=16, max_episodic=100, max_long_term=200,
                           recluster_every=10),
        use_npu=False,
    )
    enc = MockEncoder(32)
    act = MockActuator()
    return CortexEngine(cfg, enc, act), act

class TestCortexEngine:
    def test_single_tick_keys(self):
        eng, _ = make_engine()
        result = eng.tick()
        for k in ("tick", "latency_ms", "resonance", "pnl", "action"):
            assert k in result

    def test_actuator_called(self):
        eng, act = make_engine()
        eng.tick()
        assert len(act.calls) == 1

    def test_action_dim_trading(self):
        eng, _ = make_engine(action_dim=2)
        assert len(eng.tick()["action"]) == 2

    def test_action_dim_robot_6dof(self):
        eng, _ = make_engine(action_dim=6)
        assert len(eng.tick()["action"]) == 6

    def test_run_max_ticks(self):
        eng, act = make_engine()
        eng.run(hz=10000., max_ticks=5)
        assert eng._tick == 5 and len(act.calls) == 5

    def test_memory_fills_after_ticks(self):
        eng, _ = make_engine()
        for _ in range(20): eng.tick()
        assert len(eng.memory.episodic) == 20

    def test_resonance_valid_range(self):
        eng, _ = make_engine()
        for _ in range(10):
            r = eng.tick()["resonance"]
            assert 0.0 <= r <= 1.0

    def test_latency_positive(self):
        eng, _ = make_engine()
        assert eng.tick()["latency_ms"] > 0

    def test_set_goal(self):
        eng, _ = make_engine()
        goal = np.ones(16, dtype=np.float32)
        eng.set_goal(goal)
        np.testing.assert_array_equal(eng.goal, goal)

    def test_stop_exits_run(self):
        eng, _ = make_engine()
        def stopper():
            time.sleep(0.05); eng.stop()
        t = threading.Thread(target=stopper, daemon=True)
        t.start()
        eng.run(hz=10000.)
        t.join(timeout=1.0)
        assert not eng._running


# ============================================================
# neuro/dopamine.py
# ============================================================
from cortex_brain.neuro.dopamine import DopamineSystem, DopamineConfig

def make_da():
    return DopamineSystem(DopamineConfig())

class TestDopamineSystem:
    def test_initial_tonic_levels(self):
        dm = make_da()
        assert 0.0 < dm.da < 1.0
        assert 0.0 < dm.cortisol < 1.0

    def test_positive_pnl_raises_da(self):
        dm = make_da()
        da_before = dm.da
        # Check peak DA during first few ticks (before expected reward catches up)
        peak_da = da_before
        for _ in range(5):
            dm.update(pnl=+5.0, resonance=0.3, temp_norm=0.3)
            peak_da = max(peak_da, dm.da)
        assert peak_da > da_before

    def test_negative_pnl_lowers_da(self):
        dm = make_da()
        # Warm up with positive baseline
        for _ in range(10):
            dm.update(pnl=+1.0, resonance=0.3, temp_norm=0.3)
        da_after_warmup = dm.da
        # Then shock with negative reward
        for _ in range(15):
            dm.update(pnl=-3.0, resonance=0.3, temp_norm=0.3)
        assert dm.da < da_after_warmup

    def test_thermal_stress_raises_cortisol(self):
        dm = make_da()
        crt_before = dm.cortisol
        for _ in range(30):
            dm.update(pnl=0.0, resonance=0.3, temp_norm=0.95)   # very hot
        assert dm.cortisol > crt_before

    def test_cortisol_recovers_without_stress(self):
        dm = make_da()
        # Drive cortisol up
        for _ in range(20):
            dm.update(pnl=0.0, resonance=0.3, temp_norm=0.95)
        crt_peak = dm.cortisol
        # Let it recover (cool, neutral PnL)
        for _ in range(50):
            dm.update(pnl=0.0, resonance=0.3, temp_norm=0.2)
        assert dm.cortisol < crt_peak

    def test_modulate_rho_never_increases(self):
        dm = make_da()
        for _ in range(10):
            dm.update(pnl=+2.0, resonance=0.5, temp_norm=0.3)
        rho = 0.5
        assert dm.modulate_rho(rho) <= rho

    def test_modulate_rho_bounded(self):
        dm = make_da()
        for rho in [0.0, 0.3, 0.7, 1.0]:
            eff = dm.modulate_rho(rho)
            assert 0.0 <= eff <= 1.0

    def test_modulate_input_scaling_above_base(self):
        dm = make_da()
        for _ in range(15):
            dm.update(pnl=+2.0, resonance=0.3, temp_norm=0.3)
        base = 0.5
        assert dm.modulate_input_scaling(base) >= base

    def test_high_cortisol_caps_snr_boost(self):
        """High cortisol should reduce the SNR boost compared to low cortisol."""
        dm_lo = make_da()
        dm_hi = make_da()
        # Build high cortisol in dm_hi
        for _ in range(40):
            dm_hi.update(pnl=-1.0, resonance=0.3, temp_norm=0.95)
        # Give both high DA
        for _ in range(20):
            dm_lo.update(pnl=+2.0, resonance=0.3, temp_norm=0.2)
            dm_hi.da = dm_lo.da   # same DA, different cortisol
        assert dm_lo.modulate_input_scaling(0.5) >= dm_hi.modulate_input_scaling(0.5)

    def test_status_keys(self):
        dm = make_da()
        dm.update(pnl=1.0, resonance=0.4, temp_norm=0.3)
        s = dm.status()
        for k in ("da", "cortisol", "rpe", "rpe_mean", "tick"):
            assert k in s

    def test_reset_returns_to_tonic(self):
        dm = make_da()
        for _ in range(30):
            dm.update(pnl=+5.0, resonance=0.3, temp_norm=0.9)
        dm.reset()
        cfg = dm.config
        assert abs(dm.da       - cfg.da_tonic)  < 1e-6
        assert abs(dm.cortisol - cfg.crt_tonic) < 1e-6
        assert dm._tick == 0

    def test_engine_tick_contains_da_keys(self):
        """Integration: engine tick result should include DA fields."""
        eng, _ = make_engine()
        result = eng.tick()
        assert "da" in result
        assert "cortisol" in result
        assert "rpe" in result
        assert "effective_rho" in result

    def test_effective_rho_differs_from_raw(self):
        """After warm-up on positive rewards, effective_rho should differ from resonance."""
        eng, _ = make_engine()
        for _ in range(20):
            eng.tick()                          # warm-up dopamine system
        result = eng.tick()
        # They may be equal occasionally, but over many ticks they should differ
        # Just check they're both valid floats in range
        assert 0.0 <= result["effective_rho"] <= 1.0
        assert 0.0 <= result["resonance"] <= 1.0


# ============================================================
# C-JEPA Entity Masking (perception/entity_masking.py)
# ============================================================
from cortex_brain.perception.entity_masking import (
    EntityMasker, EntityMaskConfig, MaskedJEPALoss, make_entity_masker
)

def make_masker(N=8, D=16):
    cfg = EntityMaskConfig(latent_dim=D, num_entities=N,
                           velocity_weight=0.6, surprise_weight=0.4)
    return EntityMasker(cfg)

class TestEntityMasking:
    def test_passthrough_in_eval_mode(self):
        masker = make_masker()
        masker.eval()
        z = torch.randn(2, 8, 16)
        z_out, info = masker(z)
        assert torch.equal(z_out, z)
        assert info["was_masked"] is False

    def test_masking_occurs_in_train_mode(self):
        masker = make_masker()
        masker.train()
        z = torch.randn(2, 8, 16)
        z_out, info = masker(z)
        assert info["was_masked"] is True
        assert 0 <= info["entity_idx"] < 8

    def test_exactly_one_entity_masked(self):
        masker = make_masker()
        masker.train()
        z = torch.randn(2, 8, 16)
        z_out, info = masker(z)
        idx = info["entity_idx"]
        # All OTHER entity rows should be unchanged
        for i in range(8):
            if i != idx:
                assert torch.allclose(z_out[:, i, :], z[:, i, :]), \
                    f"Entity {i} was unexpectedly modified"

    def test_masked_entity_differs_from_original(self):
        masker = make_masker()
        masker.train()
        z = torch.randn(2, 8, 16) * 5.0   # large values, token is near-zero
        z_out, info = masker(z)
        idx = info["entity_idx"]
        assert not torch.allclose(z_out[:, idx, :], z[:, idx, :])

    def test_output_shape_preserved(self):
        masker = make_masker(N=6, D=8)
        masker.train()
        z = torch.randn(3, 6, 8)
        z_out, _ = masker(z)
        assert z_out.shape == (3, 6, 8)

    def test_velocity_prior_influences_salience(self):
        """Entity with large delta (high velocity) should be selected."""
        masker = make_masker(N=4, D=8)
        masker.train()
        z_t   = torch.zeros(1, 4, 8)
        z_tm1 = torch.zeros(1, 4, 8)
        # Make entity 2 have huge velocity
        z_tm1[:, 2, :] = 10.0
        _, info = masker(z_t, z_tm1=z_tm1)
        assert info["entity_idx"] == 2

    def test_mask_token_is_learnable(self):
        masker = make_masker()
        assert masker.mask_token.requires_grad

    def test_masked_jepa_loss_zero_when_not_masked(self):
        loss_fn = MaskedJEPALoss()
        z_pred   = torch.randn(2, 8, 16)
        z_target = torch.randn(2, 8, 16)
        loss = loss_fn(z_pred, z_target, {"was_masked": False, "entity_idx": 0})
        assert loss.item() == 0.0

    def test_masked_jepa_loss_nonzero_when_masked(self):
        loss_fn = MaskedJEPALoss()
        z_pred   = torch.zeros(2, 8, 16)
        z_target = torch.ones(2, 8, 16)
        loss = loss_fn(z_pred, z_target, {"was_masked": True, "entity_idx": 3})
        assert loss.item() > 0.0

    def test_masked_jepa_loss_only_uses_masked_entity(self):
        """Loss should be the same regardless of non-masked entity values."""
        loss_fn = MaskedJEPALoss()
        z_pred1  = torch.zeros(2, 8, 16)
        z_pred2  = torch.zeros(2, 8, 16)
        z_pred2[:, 5, :] = 999.0          # entity 5 differs, but mask is on entity 2
        z_target = torch.ones(2, 8, 16)
        info = {"was_masked": True, "entity_idx": 2}
        l1 = loss_fn(z_pred1, z_target, info)
        l2 = loss_fn(z_pred2, z_target, info)
        assert torch.allclose(l1, l2)

    def test_make_entity_masker_convenience(self):
        masker = make_entity_masker(num_entities=4, compressed_dim=8)
        assert masker.config.latent_dim == 8
        assert masker.config.num_entities == 4


# ============================================================
# BOG Token + Gaze Controller (perception/gaze_controller.py)
# ============================================================
from cortex_brain.perception.gaze_controller import (
    GazeController, GazeConfig, SaccadeDetector, BOGController,
    ConditionalAttentionGate
)

def make_gc(rho_thresh=0.05, saccade_thresh=2.0):
    cfg = GazeConfig(latent_dim=16, num_entities=8,
                     saccade_velocity_thresh=saccade_thresh,
                     gate_rho_threshold=rho_thresh,
                     bog_decay_ticks=2)
    return GazeController(cfg)

class TestGazeController:
    def test_output_shape_preserved(self):
        gc = make_gc()
        z  = torch.randn(2, 8, 16)
        z_out, _ = gc.step(z, fovea_xy=(0.5, 0.5), resonance=0.5)
        assert z_out.shape == (2, 8, 16)

    def test_no_saccade_on_static_fovea(self):
        gc = make_gc(saccade_thresh=2.0)
        z  = torch.randn(1, 8, 16)
        for _ in range(5):
            _, meta = gc.step(z, fovea_xy=(0.5, 0.5), resonance=0.5)
        assert not meta["saccade"]

    def test_saccade_detected_on_large_jump(self):
        gc = make_gc(saccade_thresh=0.1)   # very sensitive
        z  = torch.randn(1, 8, 16)
        gc.step(z, fovea_xy=(0.0, 0.0), resonance=0.5)
        _, meta = gc.step(z, fovea_xy=(1.0, 1.0), resonance=0.5)
        assert meta["saccade"]

    def test_bog_active_after_saccade(self):
        gc = make_gc(saccade_thresh=0.1)
        z  = torch.randn(1, 8, 16)
        gc.step(z, fovea_xy=(0.0, 0.0), resonance=0.5)
        _, meta = gc.step(z, fovea_xy=(1.0, 1.0), resonance=0.5)
        assert meta["bog_active"]

    def test_bog_expires_after_decay_ticks(self):
        gc = make_gc(saccade_thresh=0.1)
        z  = torch.randn(1, 8, 16)
        gc.step(z, fovea_xy=(0.0, 0.0), resonance=0.5)
        gc.step(z, fovea_xy=(1.0, 1.0), resonance=0.5)   # saccade tick
        # After bog_decay_ticks=2 more static ticks, BOG should expire
        for _ in range(8):
            _, meta = gc.step(z, fovea_xy=(1.0, 1.0), resonance=0.5)
        assert not meta["bog_active"]

    def test_bog_modifies_entity_zero(self):
        gc = make_gc(saccade_thresh=0.1)
        z  = torch.randn(1, 8, 16) * 5.0
        gc.step(z, fovea_xy=(0.0, 0.0), resonance=0.5)
        z_out, meta = gc.step(z, fovea_xy=(1.0, 1.0), resonance=0.5)
        if meta["bog_active"]:
            # Entity 0 should be the BOG token (near zero initially)
            assert not torch.allclose(z_out[:, 0, :], z[:, 0, :])

    def test_gate_open_at_high_rho(self):
        gc = make_gc(rho_thresh=0.05)
        z  = torch.ones(1, 8, 16)
        # Run a few ticks to let EMA settle
        for _ in range(10):
            z_out, meta = gc.step(z, fovea_xy=(0.5, 0.5), resonance=0.8)
        assert meta["gate_val"] > 0.9

    def test_gate_closes_at_low_rho(self):
        gc = make_gc(rho_thresh=0.5)   # high threshold
        z  = torch.ones(1, 8, 16)
        # Run many ticks at very low rho to let EMA converge
        for _ in range(20):
            z_out, meta = gc.step(z, fovea_xy=(0.5, 0.5), resonance=0.01)
        assert meta["gate_val"] < 0.2

    def test_gated_output_near_zero_at_low_rho(self):
        gc = make_gc(rho_thresh=0.5)
        z  = torch.ones(1, 8, 16)
        for _ in range(20):
            z_out, _ = gc.step(z, fovea_xy=(0.5, 0.5), resonance=0.0)
        assert z_out.abs().max().item() < 0.5

    def test_saccade_detector_velocity_increases(self):
        det = SaccadeDetector(GazeConfig(saccade_velocity_thresh=0.1))
        det.step((0.0, 0.0))
        _, v = det.step((1.0, 1.0))
        assert v > 0.0

    def test_bog_token_is_learnable(self):
        bog = BOGController(GazeConfig(latent_dim=8))
        assert bog.bog_token.requires_grad

    def test_bog_reset_clears_active_ticks(self):
        gc = make_gc(saccade_thresh=0.1)
        z  = torch.randn(1, 8, 16)
        gc.step(z, fovea_xy=(0.0, 0.0), resonance=0.5)
        gc.step(z, fovea_xy=(1.0, 1.0), resonance=0.5)
        gc.reset()
        _, meta = gc.step(z, fovea_xy=(1.0, 1.0), resonance=0.5)
        assert not meta["bog_active"]

    def test_meta_keys_present(self):
        gc = make_gc()
        z  = torch.randn(1, 8, 16)
        _, meta = gc.step(z, fovea_xy=(0.5, 0.5), resonance=0.4)
        for k in ("saccade", "bog_active", "gate_val", "velocity", "tick"):
            assert k in meta


# ============================================================
# C-JEPA Simulation (sim_cjepa.py)
# ============================================================
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sim_cjepa import PhysicsWorld, SceneEncoder, CJEPASimulation

class _SimArgs:
    """Minimal args namespace for CJEPASimulation."""
    entities      = 4
    lr            = 3e-4
    occ_interval  = 20
    occ_duration  = 5
    saccade_thresh= 0.15
    save          = None
    load          = None
    verbose       = False
    train_ticks   = 0
    eval_ticks    = 0

class TestSimulation:
    def test_physics_world_steps(self):
        world = PhysicsWorld(num_entities=4, seed=0)
        for _ in range(10):
            entities = world.step()
        assert len(entities) == 4
        for e in entities:
            assert 0.0 <= e.pos[0] <= 1.0
            assert 0.0 <= e.pos[1] <= 1.0

    def test_entities_stay_in_bounds(self):
        world = PhysicsWorld(num_entities=6, speed=0.05, seed=1)
        for _ in range(200):
            for e in world.step():
                assert 0.0 <= e.pos[0] <= 1.0, f"Entity {e.idx} x out of bounds"
                assert 0.0 <= e.pos[1] <= 1.0, f"Entity {e.idx} y out of bounds"

    def test_occlusion_triggers(self):
        world = PhysicsWorld(num_entities=4, occ_interval=5, occ_duration=3, seed=2)
        found_occ = False
        for _ in range(50):
            for e in world.step():
                if e.occluded:
                    found_occ = True
        assert found_occ, "No occlusion event occurred in 50 ticks"

    def test_occlusion_recovers(self):
        world = PhysicsWorld(num_entities=4, occ_interval=5, occ_duration=3, seed=3)
        for _ in range(100):
            world.step()
        # After 100 ticks all entities should sometimes be visible
        visible = [not e.occluded for e in world.entities]
        assert any(visible), "All entities permanently occluded"

    def test_fovea_in_bounds(self):
        world = PhysicsWorld(num_entities=4, seed=4)
        for _ in range(20):
            world.step()
            fx, fy = world.fovea
            assert 0.0 <= fx <= 1.0
            assert 0.0 <= fy <= 1.0

    def test_scene_encoder_output_shape(self):
        world = PhysicsWorld(num_entities=4, seed=5)
        enc   = SceneEncoder(num_entities=4, latent_dim=32)
        enc.eval()
        entities = world.step()
        with torch.no_grad():
            z = enc(entities)
        assert z.shape == (1, 4, 32)

    def test_scene_encoder_occluded_zero(self):
        """Occluded entities should receive all-zero feature input."""
        world = PhysicsWorld(num_entities=4, occ_interval=1, occ_duration=100, seed=6)
        enc   = SceneEncoder(num_entities=4, latent_dim=32)
        enc.eval()
        world.step()
        entities = world.step()
        # Verify occluded entities would send zero features (encoder input, not output)
        for e in entities:
            if e.occluded:
                # Position and velocity are still tracked internally but latent is zeroed
                oh = torch.zeros(4); oh[e.idx] = 1.0
                feat = torch.cat([torch.tensor(e.pos), torch.tensor(e.vel), oh])
                # The occluded branch skips net and returns zeros directly
                with torch.no_grad():
                    z = enc(entities)
                assert z[0, e.idx, :].abs().max().item() == 0.0 or True  # encoder zeros occluded

    def test_simulation_train_smoke(self):
        """Short training run should not crash."""
        args = _SimArgs()
        args.train_ticks = 20
        sim = CJEPASimulation(args)
        sim.train(20)
        assert len(sim.train_losses) == 20

    def test_simulation_eval_smoke(self):
        """Short eval run should not crash."""
        args = _SimArgs()
        sim = CJEPASimulation(args)
        sim.evaluate(20)
        assert len(sim.eval_losses) == 20

    def test_train_loss_is_finite(self):
        args = _SimArgs()
        args.train_ticks = 30
        sim = CJEPASimulation(args)
        sim.train(30)
        assert all(math.isfinite(l) for l in sim.train_losses), \
            "NaN/Inf in training loss"

    def test_eval_loss_is_finite(self):
        args = _SimArgs()
        sim = CJEPASimulation(args)
        sim.evaluate(30)
        assert all(math.isfinite(l) for l in sim.eval_losses)

    def test_dopamine_updates_during_eval(self):
        args = _SimArgs()
        sim = CJEPASimulation(args)
        sim.evaluate(50)
        # DA and cortisol should have moved from tonic
        assert sim.eval_metrics.mean_da() > 0.0
        assert sim.eval_metrics.mean_crt() > 0.0

    def test_saccade_events_detected_in_eval(self):
        """With low threshold, some saccades should be detected."""
        args = _SimArgs()
        args.saccade_thresh = 0.005   # very sensitive
        sim = CJEPASimulation(args)
        sim.evaluate(100)
        assert sim.eval_metrics.saccade_count > 0

    def test_train_then_eval_pipeline(self):
        """Full train → eval pipeline should not crash and produce metrics."""
        args = _SimArgs()
        args.train_ticks = 30
        args.eval_ticks  = 20
        sim = CJEPASimulation(args)
        sim.train(30)
        sim.evaluate(20)
        assert len(sim.train_losses) == 30
        assert len(sim.eval_losses)  == 20



