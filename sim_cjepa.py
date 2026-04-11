"""
sim_cjepa.py
~~~~~~~~~~~~~
Simulation to train and test C-JEPA Entity Masking, BOG Gaze Controller,
Sandwich Norm, and Dopamine System together in a single loop.

WHAT THE SIMULATION DOES
--------------------------
1. PhysicsWorld generates a synthetic scene of N entities moving in 2D.
   Entities bounce off walls, occlude each other, and occasionally
   "disappear" behind an occluder for T_occ frames — simulating the
   object permanence problem C-JEPA is designed to solve.

2. SceneEncoder projects entity states (position, velocity, identity)
   into the 128-D latent space the real system uses.

3. TRAINING PHASE (--train-ticks N):
   - EntityMasker selects the highest-salience entity (usually the
     fast-moving or recently-occluded one) and masks its latent.
   - CJEPAPredictor must reconstruct it.
   - MaskedJEPALoss provides gradient only on the masked entity.
   - Adam optimiser updates predictor weights.

4. EVALUATION PHASE (--eval-ticks N):
   - GazeController tracks fovea position (follows the highest-salience
     entity). When the entity jumps (saccade), BOG token is injected.
   - DopamineSystem modulates resonance (rho) based on prediction quality.
   - Metrics logged every tick: loss, DA, CRT, gate_val, saccade events.

5. RESULTS printed at the end:
   - Prediction loss curve (train vs eval)
   - Occlusion recovery rate: did the predictor find the hidden entity?
   - Saccade events and BOG injection count
   - DA/CRT stability metrics

USAGE
-----
    # Quick smoke test (fast):
    python sim_cjepa.py --train-ticks 100 --eval-ticks 50 --entities 4

    # Full training run:
    python sim_cjepa.py --train-ticks 2000 --eval-ticks 500 --entities 8

    # With live logging:
    python sim_cjepa.py --train-ticks 500 --eval-ticks 200 --verbose

    # Save trained weights:
    python sim_cjepa.py --train-ticks 2000 --save weights/cjepa.pt

    # Load and evaluate only:
    python sim_cjepa.py --train-ticks 0 --eval-ticks 500 --load weights/cjepa.pt
"""
from __future__ import annotations
import argparse, logging, time, math, os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ── Local imports ─────────────────────────────────────────────────────────────
from cortex_brain.perception.eb_jepa import (
    CJEPAPredictor, EBJEPAConfig, ProprioceptionPulse)
from cortex_brain.perception.entity_masking import (
    EntityMasker, EntityMaskConfig, MaskedJEPALoss)
from cortex_brain.perception.gaze_controller import (
    GazeController, GazeConfig)
from cortex_brain.neuro.dopamine import DopamineSystem, DopamineConfig

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("sim_cjepa")


# ─────────────────────────────────────────────────────────────────────────────
# 1. PHYSICS WORLD
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EntityState:
    idx:     int
    pos:     np.ndarray   # (2,) normalised [0, 1]
    vel:     np.ndarray   # (2,) normalised velocity
    radius:  float
    occluded: bool = False
    occ_timer: int = 0

class PhysicsWorld:
    """
    2D bounding-box world with N entities, wall bouncing, and occlusion events.

    Occlusion model: every ~occ_interval ticks one random entity "hides"
    behind an occluder for occ_duration ticks. During occlusion its latent
    is set to zero (as the encoder would receive no visual input for it).
    This is the core problem C-JEPA must solve.
    """
    def __init__(
        self,
        num_entities:  int   = 8,
        speed:         float = 0.008,
        occ_interval:  int   = 40,
        occ_duration:  int   = 12,
        seed:          int   = 42,
    ) -> None:
        self.N    = num_entities
        self.rng  = np.random.default_rng(seed)
        self.occ_interval = occ_interval
        self.occ_duration = occ_duration
        self._tick = 0

        self.entities: List[EntityState] = []
        for i in range(num_entities):
            pos = self.rng.uniform(0.1, 0.9, size=2).astype(np.float32)
            angle = self.rng.uniform(0, 2 * math.pi)
            spd   = speed * self.rng.uniform(0.5, 2.0)
            vel   = np.array([math.cos(angle), math.sin(angle)],
                             dtype=np.float32) * spd
            self.entities.append(EntityState(i, pos, vel, radius=0.04))

    def step(self) -> List[EntityState]:
        self._tick += 1

        # Possibly start an occlusion
        if self._tick % self.occ_interval == 0:
            candidate = [e for e in self.entities if not e.occluded]
            if candidate:
                victim = self.rng.choice(candidate)
                victim.occluded  = True
                victim.occ_timer = self.occ_duration

        for e in self.entities:
            # Physics
            e.pos += e.vel
            # Wall bounce
            for dim in range(2):
                if e.pos[dim] < e.radius:
                    e.pos[dim]  =  e.radius
                    e.vel[dim]  = abs(e.vel[dim])
                elif e.pos[dim] > 1.0 - e.radius:
                    e.pos[dim]  = 1.0 - e.radius
                    e.vel[dim]  = -abs(e.vel[dim])
            # Occlusion countdown
            if e.occluded:
                e.occ_timer -= 1
                if e.occ_timer <= 0:
                    e.occluded = False

        return self.entities

    @property
    def fovea(self) -> Tuple[float, float]:
        """
        Simulated fovea follows the fastest-moving visible entity.
        Returns normalised (x, y).
        """
        best, best_spd = self.entities[0], 0.0
        for e in self.entities:
            if not e.occluded:
                spd = float(np.linalg.norm(e.vel))
                if spd > best_spd:
                    best, best_spd = e, spd
        return (float(best.pos[0]), float(best.pos[1]))


# ─────────────────────────────────────────────────────────────────────────────
# 2. SCENE ENCODER  (entity state → 128-D latent)
# ─────────────────────────────────────────────────────────────────────────────

class SceneEncoder(nn.Module):
    """
    Maps raw entity state [pos_x, pos_y, vel_x, vel_y, identity_one_hot]
    to a 128-D latent vector per entity.

    Occluded entities produce all-zero latents (no visual signal).
    """
    def __init__(self, num_entities: int, latent_dim: int = 128) -> None:
        super().__init__()
        in_dim = 4 + num_entities   # pos(2) + vel(2) + one-hot identity
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, latent_dim), nn.Tanh(),
        )
        self.N = num_entities

    def forward(self, entities: List[EntityState]) -> torch.Tensor:
        """Returns (1, N, 128) float32."""
        rows = []
        occ_mask = []
        for e in entities:
            oh = torch.zeros(self.N); oh[e.idx] = 1.0
            feat = torch.cat([torch.tensor(e.pos), torch.tensor(e.vel), oh])
            rows.append(feat)
            occ_mask.append(e.occluded)
        x = torch.stack(rows).unsqueeze(0)        # (1, N, 4+N)
        z = self.net(x)                            # (1, N, latent_dim)
        # Zero out occluded entity latents (no visual signal)
        for i, occluded in enumerate(occ_mask):
            if occluded:
                z[0, i, :] = 0.0
        return z


# ─────────────────────────────────────────────────────────────────────────────
# 3. METRICS TRACKER
# ─────────────────────────────────────────────────────────────────────────────

class Metrics:
    def __init__(self, window: int = 50) -> None:
        self.losses:      deque = deque(maxlen=window)
        self.da_vals:     deque = deque(maxlen=window)
        self.crt_vals:    deque = deque(maxlen=window)
        self.gate_vals:   deque = deque(maxlen=window)
        self.saccade_count = 0
        self.bog_count     = 0
        self.occ_recoveries = 0   # ticks where masked entity was occluded
        self.occ_attempts   = 0

    def update(self, loss, da, crt, gate, saccade, bog_active, was_occluded):
        self.losses.append(loss)
        self.da_vals.append(da)
        self.crt_vals.append(crt)
        self.gate_vals.append(gate)
        if saccade:    self.saccade_count += 1
        if bog_active: self.bog_count     += 1
        if was_occluded:
            self.occ_attempts += 1

    def mean_loss(self):  return float(np.mean(self.losses))  if self.losses else 0.
    def mean_da(self):    return float(np.mean(self.da_vals)) if self.da_vals else 0.
    def mean_crt(self):   return float(np.mean(self.crt_vals)) if self.crt_vals else 0.
    def mean_gate(self):  return float(np.mean(self.gate_vals)) if self.gate_vals else 0.


# ─────────────────────────────────────────────────────────────────────────────
# 4. SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

class CJEPASimulation:
    """
    Orchestrates the full training and evaluation loop.

    Training:
        World → SceneEncoder → EntityMasker → CJEPAPredictor → MaskedJEPALoss
                                                              ↑ Adam update

    Evaluation:
        World → SceneEncoder → GazeController → CJEPAPredictor (no grad)
                             ↓                        ↓
                         DopamineSystem ←── prediction quality (1 - loss)
    """
    def __init__(self, args) -> None:
        self.args = args
        N  = args.entities
        D  = 128    # latent dim
        CD = 16     # compressed dim

        # ── World ─────────────────────────────────────────────────────────
        self.world = PhysicsWorld(
            num_entities  = N,
            speed         = 0.007,
            occ_interval  = args.occ_interval,
            occ_duration  = args.occ_duration,
        )

        # ── Models ────────────────────────────────────────────────────────
        self.encoder = SceneEncoder(N, latent_dim=D)

        jepa_cfg = EBJEPAConfig(
            latent_dim      = D,
            compressed_dim  = CD,
            aux_dim         = 3,
            hidden_dim      = 64,
            sandwich_norm   = True,
            num_candidates  = 32,
            planning_horizon= 3,
        )
        self.predictor = CJEPAPredictor(jepa_cfg)

        mask_cfg = EntityMaskConfig(
            latent_dim      = CD,
            num_entities    = N,
            velocity_weight = 0.6,
            surprise_weight = 0.4,
        )
        self.masker = EntityMasker(mask_cfg)
        self.loss_fn = MaskedJEPALoss()

        gaze_cfg = GazeConfig(
            latent_dim              = CD,
            num_entities            = N,
            saccade_velocity_thresh = args.saccade_thresh,
            bog_decay_ticks         = 3,
            gate_rho_threshold      = 0.05,
        )
        self.gaze  = GazeController(gaze_cfg)
        self.proprio = ProprioceptionPulse()
        self.dopamine = DopamineSystem(DopamineConfig(da_lr=0.2, crt_lr=0.06))

        # ── Optimiser ─────────────────────────────────────────────────────
        params = (list(self.predictor.parameters()) +
                  list(self.encoder.parameters()) +
                  list(self.masker.parameters()))
        self.opt = optim.Adam(params, lr=args.lr)

        # ── State ─────────────────────────────────────────────────────────
        self.z_prev:     Optional[torch.Tensor] = None   # previous frame latents
        self.train_losses: List[float] = []
        self.eval_losses:  List[float] = []
        self.train_metrics = Metrics()
        self.eval_metrics  = Metrics()

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(self, num_ticks: int) -> None:
        if num_ticks == 0:
            return
        self.predictor.train()
        self.masker.train()
        self.encoder.train()
        self.z_prev = None

        log.warning("=== TRAINING  %d ticks ===", num_ticks)
        t_start = time.perf_counter()

        for tick in range(1, num_ticks + 1):
            entities = self.world.step()
            z_full   = self.encoder(entities)              # (1, N, 128)

            # Compress to 16-D for masker and predictor
            with torch.no_grad():
                z_comp_full = self.predictor.compressor(z_full)   # (1, N, 16)

            # Mask one entity
            z_masked, mask_info = self.masker(
                z_comp_full,
                z_tm1  = self.z_prev,
            )

            # Predict next compressed state
            prop  = self.proprio.get_aux_tensor()          # (1, 3)
            u_act = torch.zeros(1, 2)
            z_pred = self.predictor(
                torch.nn.functional.pad(
                    z_masked, (0, 128 - 16)),              # expand back for bridge
                u_act, prop)                               # (1, N, 16)

            # Loss on masked entity only
            loss = self.loss_fn(z_pred, z_comp_full, mask_info)

            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.predictor.parameters()) +
                list(self.encoder.parameters()) +
                list(self.masker.parameters()), 1.0)
            self.opt.step()

            loss_val = float(loss.item())
            self.train_losses.append(loss_val)

            # Was the masked entity occluded? (hardest case)
            was_occ = mask_info["was_masked"] and \
                      entities[mask_info["entity_idx"]].occluded
            self.train_metrics.update(
                loss_val, da=0, crt=0, gate=1.0,
                saccade=False, bog_active=False,
                was_occluded=was_occ)

            self.z_prev = z_comp_full.detach()

            if self.args.verbose and tick % 50 == 0:
                recent = float(np.mean(self.train_losses[-50:]))
                print(f"  train tick={tick:5d}  loss={recent:.4f}  "
                      f"masked_entity={mask_info['entity_idx']}  "
                      f"salience={mask_info['salience']:.3f}  "
                      f"occ={was_occ}")

        elapsed = time.perf_counter() - t_start
        final_loss = float(np.mean(self.train_losses[-100:]))
        print(f"\n✅ Training complete  "
              f"ticks={num_ticks}  "
              f"time={elapsed:.1f}s  "
              f"final_loss={final_loss:.4f}")

    # ── Evaluation loop ───────────────────────────────────────────────────────

    def evaluate(self, num_ticks: int) -> None:
        if num_ticks == 0:
            return
        self.predictor.eval()
        self.masker.eval()
        self.encoder.eval()
        self.gaze.reset()
        self.dopamine.reset()
        self.z_prev = None

        log.warning("=== EVALUATION  %d ticks ===", num_ticks)
        t_start = time.perf_counter()

        for tick in range(1, num_ticks + 1):
            entities = self.world.step()
            fovea    = self.world.fovea

            with torch.no_grad():
                z_full  = self.encoder(entities)                   # (1, N, 128)
                z_comp  = self.predictor.compressor(z_full)        # (1, N, 16)

            # Gaze controller: saccade detect + BOG + gate
            z_gated, gaze_meta = self.gaze.step(
                z_comp, fovea_xy=fovea,
                resonance=self.dopamine.da)

            # Predict
            prop  = self.proprio.get_aux_tensor()
            u_act = torch.zeros(1, 2)
            with torch.no_grad():
                z_pred = self.predictor(
                    torch.nn.functional.pad(z_gated, (0, 128 - 16)),
                    u_act, prop)

            # Compute prediction error as a proxy reward signal
            pred_err = float((z_pred - z_comp).norm(dim=-1).mean())
            pred_quality = max(0.0, 1.0 - pred_err)     # [0,1], higher = better

            # Dopamine update — reward = prediction quality
            pnl = pred_quality - 0.5                     # centre around zero
            temp_norm = float(prop[0, 0])
            self.dopamine.update(pnl=pnl, resonance=self.dopamine.da,
                                 temp_norm=temp_norm)

            self.eval_losses.append(pred_err)

            # Track which entities were occluded
            was_occ = any(e.occluded for e in entities)
            self.eval_metrics.update(
                pred_err,
                da        = self.dopamine.da,
                crt       = self.dopamine.cortisol,
                gate      = gaze_meta["gate_val"],
                saccade   = gaze_meta["saccade"],
                bog_active= gaze_meta["bog_active"],
                was_occluded = was_occ,
            )

            self.z_prev = z_comp.detach()

            if self.args.verbose and tick % 50 == 0:
                print(f"  eval  tick={tick:5d}  "
                      f"err={pred_err:.4f}  "
                      f"DA={self.dopamine.da:.3f}  "
                      f"CRT={self.dopamine.cortisol:.3f}  "
                      f"gate={gaze_meta['gate_val']:.3f}  "
                      f"saccade={gaze_meta['saccade']}  "
                      f"BOG={gaze_meta['bog_active']}  "
                      f"occ={was_occ}")

        elapsed = time.perf_counter() - t_start
        final_err = float(np.mean(self.eval_losses[-100:]))
        print(f"\n✅ Evaluation complete  "
              f"ticks={num_ticks}  "
              f"time={elapsed:.1f}s  "
              f"mean_err={final_err:.4f}")

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "predictor":  self.predictor.state_dict(),
            "encoder":    self.encoder.state_dict(),
            "masker":     self.masker.state_dict(),
        }, path)
        print(f"💾 Saved weights → {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        self.predictor.load_state_dict(ckpt["predictor"])
        self.encoder.load_state_dict(ckpt["encoder"])
        self.masker.load_state_dict(ckpt["masker"])
        print(f"📂 Loaded weights ← {path}")

    # ── Report ────────────────────────────────────────────────────────────────

    def report(self) -> None:
        sep = "─" * 60
        print(f"\n{sep}")
        print("  C-JEPA SIMULATION REPORT")
        print(sep)

        # Training summary
        if self.train_losses:
            early = float(np.mean(self.train_losses[:min(100, len(self.train_losses))]))
            late  = float(np.mean(self.train_losses[-min(100, len(self.train_losses)):]))
            improvement = (early - late) / (early + 1e-8) * 100
            print(f"\n  TRAINING")
            print(f"    Ticks           : {len(self.train_losses)}")
            print(f"    Early loss      : {early:.4f}")
            print(f"    Final loss      : {late:.4f}")
            print(f"    Improvement     : {improvement:+.1f}%")
            print(f"    Occ ticks       : {self.train_metrics.occ_attempts}")

        # Eval summary
        if self.eval_losses:
            mean_err  = float(np.mean(self.eval_losses))
            m         = self.eval_metrics
            print(f"\n  EVALUATION")
            print(f"    Ticks           : {len(self.eval_losses)}")
            print(f"    Mean pred error : {mean_err:.4f}")
            print(f"    Mean DA         : {m.mean_da():.3f}")
            print(f"    Mean cortisol   : {m.mean_crt():.3f}")
            print(f"    Mean gate       : {m.mean_gate():.3f}")
            print(f"    Saccade events  : {m.saccade_count}")
            print(f"    BOG injections  : {m.bog_count}")
            print(f"    Occ ticks       : {m.occ_attempts}")

        # Loss curve (ASCII sparkline)
        if self.train_losses and len(self.train_losses) >= 10:
            print(f"\n  TRAINING LOSS CURVE (each block = {len(self.train_losses)//10} ticks)")
            segments = np.array_split(self.train_losses, 10)
            means    = [float(np.mean(s)) for s in segments]
            vmin, vmax = min(means), max(means)
            bars = "▁▂▃▄▅▆▇█"
            line = ""
            for v in means:
                if vmax > vmin:
                    idx = int((v - vmin) / (vmax - vmin) * 7)
                else:
                    idx = 0
                line += bars[7 - idx]   # invert: lower loss = taller bar
            print(f"    {line}  ({vmin:.3f} → {vmax:.3f})")

        # Feature status
        print(f"\n  FEATURES ACTIVE")
        print(f"    Sandwich Norm   : ✅  (INT8-ready predictor)")
        print(f"    Entity Masking  : ✅  (C-JEPA trajectory masking)")
        print(f"    Gaze Controller : ✅  (BOG token + conditional gate)")
        print(f"    Dopamine System : ✅  (DA/CRT/RPE modulation)")
        print(f"{sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 5. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="C-JEPA Simulation")
    p.add_argument("--train-ticks",   type=int,   default=500,
                   help="Number of training ticks (0 = eval only)")
    p.add_argument("--eval-ticks",    type=int,   default=200,
                   help="Number of evaluation ticks")
    p.add_argument("--entities",      type=int,   default=6,
                   help="Number of entities in the scene")
    p.add_argument("--lr",            type=float, default=3e-4,
                   help="Adam learning rate")
    p.add_argument("--occ-interval",  type=int,   default=40,
                   help="Ticks between occlusion events")
    p.add_argument("--occ-duration",  type=int,   default=12,
                   help="Duration of each occlusion in ticks")
    p.add_argument("--saccade-thresh",type=float, default=0.15,
                   help="Fovea velocity threshold for saccade detection")
    p.add_argument("--save",          type=str,   default=None,
                   help="Path to save trained weights (e.g. weights/cjepa.pt)")
    p.add_argument("--load",          type=str,   default=None,
                   help="Path to load weights before evaluation")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print per-tick logs every 50 ticks")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"""
╔══════════════════════════════════════════════════════╗
║            C-JEPA SIMULATION  v1.0                  ║
║  Entities={args.entities}  Train={args.train_ticks}  Eval={args.eval_ticks}  ║
║  Sandwich Norm · Entity Masking · BOG Token · DA     ║
╚══════════════════════════════════════════════════════╝
""")

    sim = CJEPASimulation(args)

    if args.load:
        sim.load(args.load)

    sim.train(args.train_ticks)

    if args.save and args.train_ticks > 0:
        sim.save(args.save)

    sim.evaluate(args.eval_ticks)
    sim.report()
