"""
cortex_brain.engine
~~~~~~~~~~~~~~~~~~~~
CortexEngine – universal wrapper composing all subsystems.

Source: unified_cortex_v5_5_final.py orchestration pattern.

Tick pipeline
-------------
1. raw_obs → FeatureEncoder → x          (input_dim,)
2. x       → LSM.step      → z_lsm       (latent_dim,)   temporal memory
3. z_lsm   → StaticCSRRouter→ manifold, weights           sparse routing
4. weights → TTM.get_routing_adjustment → adj             long-term memory
5. manifold→ EBJEPAPlanner.plan → action                  predictive planning
6. action  → Actuator → pnl                               execution
7. (manifold, weights, resonance, pnl) → TTM.observe      memory update

Usage – Trading
---------------
    engine = CortexEngine(EngineConfig(), MarketEncoder(), AlpacaActuator())
    engine.run(hz=0.5)

Usage – Robotics
----------------
    engine = CortexEngine(EngineConfig(action_dim=6), CameraEncoder(), JointActuator())
    engine.run(hz=50.0)
"""
from __future__ import annotations
import logging, time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional
import numpy as np
import torch

from .hardware.amd_npu_binding import AMDNPUBinding, NPUConfig
from .memory.ttm_clustering    import TestTimeMemoryWithClustering, TTMConfig
from .neuro.dopamine           import DopamineSystem, DopamineConfig
from .perception.eb_jepa       import CJEPAPredictor, EBJEPAConfig, EBJEPAPlanner, ProprioceptionPulse
from .perception.lsm           import LiquidStateMachine, LSMConfig
from .routing.static_csr       import StaticCSRRouter, CSRRouterConfig

logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    input_dim:  int = 256
    latent_dim: int = 128
    action_dim: int = 2

    lsm: LSMConfig = field(default_factory=lambda: LSMConfig(
        input_dim=256, reservoir_dim=512, output_dim=128, spectral_radius=0.9))
    jepa: EBJEPAConfig = field(default_factory=lambda: EBJEPAConfig(
        latent_dim=128, compressed_dim=16, aux_dim=3, num_candidates=64, planning_horizon=5))
    router: CSRRouterConfig = field(default_factory=lambda: CSRRouterConfig(
        input_dim=256, manifold_dim=128, num_experts=4, sparsity=0.80))
    ttm: TTMConfig = field(default_factory=lambda: TTMConfig(
        manifold_dim=128, max_episodic=500, max_long_term=1000))
    npu: NPUConfig = field(default_factory=NPUConfig)

    use_npu:        bool            = False
    use_dopamine:   bool            = True
    telemetry_ip:   Optional[str]  = None
    telemetry_port: int             = 5005
    dopamine: DopamineConfig = field(default_factory=DopamineConfig)


class FeatureEncoder:
    """Override encode() to convert any domain's raw_obs → (input_dim,) float32."""
    def encode(self, raw_obs: Any) -> np.ndarray:
        raise NotImplementedError


class Actuator:
    """Override act() to execute in the real/simulated world. Returns PnL/reward."""
    def act(self, action: np.ndarray, resonance: float, metadata: Dict) -> float:
        raise NotImplementedError


class CortexEngine:
    def __init__(self, config: EngineConfig, feature_encoder: FeatureEncoder,
                 actuator: Actuator, goal_latent: Optional[np.ndarray] = None) -> None:
        self.config  = config
        self.encoder = feature_encoder
        self.actuator = actuator
        self.goal = goal_latent if goal_latent is not None \
                    else np.zeros(config.latent_dim, dtype=np.float32)

        self.lsm    = LiquidStateMachine(config.lsm)
        self.router = StaticCSRRouter(config.router)
        self.router.eval()
        self.memory = TestTimeMemoryWithClustering(config.ttm)
        self.npu    = AMDNPUBinding(config.npu) if config.use_npu else None
        self.jepa   = EBJEPAPlanner(CJEPAPredictor(config.jepa), config.jepa,
                                    action_dim=config.action_dim)
        self.proprio = ProprioceptionPulse()
        self.dopamine = DopamineSystem(config.dopamine) if config.use_dopamine else None

        self._tick    = 0
        self._running = False

        logger.info("CortexEngine │ LSM %d→%d │ CSR %d experts │ JEPA K=%d H=%d │ DA=%s",
                    config.lsm.reservoir_dim, config.lsm.output_dim,
                    config.router.num_experts,
                    config.jepa.num_candidates, config.jepa.planning_horizon,
                    "on" if config.use_dopamine else "off")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def tick(self, raw_obs: Any = None) -> Dict[str, Any]:
        t0 = time.perf_counter()

        x       = self.encoder.encode(raw_obs).astype(np.float32)      # (input_dim,)

        # ── DA: SNR enhancement — scale LSM input before reservoir step ──
        if self.dopamine is not None:
            eff_scale = self.dopamine.modulate_input_scaling(self.config.lsm.input_scaling)
            x = x * (eff_scale / max(self.config.lsm.input_scaling, 1e-8))

        z_lsm   = self.lsm.step(x)                                     # (latent_dim,)

        x_t     = torch.from_numpy(x).unsqueeze(0)                     # (1, input_dim)
        ttm_adj = self.memory.get_routing_adjustment(z_lsm)
        with torch.no_grad():
            manifold, weights = self.router(x_t, ttm_adj)              # (1,128), (1,4)

        resonance  = self.router.get_resonance(weights)
        z_manifold = manifold.squeeze().numpy()                         # (128,)

        # ── DA: Arousal modulation — lower ρ on positive RPE ─────────────
        effective_rho = self.dopamine.modulate_rho(resonance) \
                        if self.dopamine is not None else resonance

        proprio = self.proprio.get_aux_tensor()                         # (1, 3)
        action  = self.jepa.plan(z_manifold, self.goal, proprio, rho=effective_rho)

        pnl = self.actuator.act(
            action, resonance,
            {"manifold": z_manifold, "weights": weights.squeeze().detach().numpy()})

        # ── DA: Update neuromodulators AFTER reward is observed ──────────
        da_status = {}
        if self.dopamine is not None:
            temp_norm = float(proprio[0, 0])                            # cpu_temp normalised
            self.dopamine.update(pnl=pnl, resonance=resonance, temp_norm=temp_norm)
            da_status = self.dopamine.status()

        moe_w = weights.squeeze().detach().numpy()
        self.memory.observe(z_manifold, moe_w, resonance, pnl)
        self.memory.evaluate_hypotheses(z_manifold, pnl)

        if self.config.telemetry_ip and self.npu:
            self.npu.broadcast_telemetry(z_manifold, (0.5, 0.5),
                                         self.config.telemetry_ip, self.config.telemetry_port)

        self._tick += 1
        da_status.pop("tick", None)   # avoid collision with result's own tick counter
        result = dict(tick=self._tick, latency_ms=(time.perf_counter()-t0)*1000,
                      resonance=resonance, effective_rho=effective_rho, pnl=pnl,
                      prior_size=self.memory.prior_size,
                      hypotheses=self.memory.hypothesis_count,
                      action=action.tolist(), **da_status)
        logger.debug("tick=%d %.1fms ρ=%.4f ρ_eff=%.4f pnl=%.2f DA=%.3f CRT=%.3f",
                     self._tick, result["latency_ms"], resonance, effective_rho, pnl,
                     da_status.get("da", 0), da_status.get("cortisol", 0))
        return result

    def run(self, hz: float = 0.5, max_ticks: Optional[int] = None,
            obs_source: Optional[Callable[[], Any]] = None) -> None:
        self._running = True
        interval = 1.0 / hz
        logger.info("CortexEngine loop at %.2fHz.", hz)
        try:
            while self._running:
                if max_ticks and self._tick >= max_ticks:
                    break
                t0 = time.perf_counter()
                self.tick(obs_source() if obs_source else None)
                time.sleep(max(0.0, interval - (time.perf_counter() - t0)))
        except KeyboardInterrupt:
            logger.info("Interrupted.")
        finally:
            self._running = False
            logger.info("Stopped at tick %d.", self._tick)

    def stop(self) -> None:
        self._running = False

    def set_goal(self, goal_latent: np.ndarray) -> None:
        self.goal = goal_latent.astype(np.float32)
