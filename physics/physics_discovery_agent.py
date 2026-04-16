"""
physics_discovery_agent.py — Autonomous Physics Discovery
============================================================
A world model that does science.

Architecture:
  1. SimplePhysicsSim — configurable 2D physics with hidden forces
  2. PhysicsBeliefModel — F=ma prior + learnable force terms
  3. AnomalyClassifier — noise vs parameter error vs missing law
  4. PhysicsKnowledgeBase — local queryable physics concepts
  5. HypothesisEngine — formulate + test candidate explanations
  6. DiscoveryAgent — the full observe→predict→detect→query→revise loop

The agent starts knowing only F=ma. You introduce gravity, friction,
magnetic fields, buoyancy. It discovers each one autonomously.

Usage:
    python physics_discovery_agent.py
    python physics_discovery_agent.py --scenario magnetic
    python physics_discovery_agent.py --scenario all

Author: John Taylor
Sprint: Physics Discovery Agent (Paper 4 prototype)
"""

import math
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# 1. Simple Physics Simulation
# ──────────────────────────────────────────────────────────────────────────────

class ForceType(Enum):
    GRAVITY = "gravity"
    FRICTION = "friction"
    MAGNETIC = "magnetic"
    BUOYANCY = "buoyancy"
    SPRING = "spring"
    DRAG = "drag"


@dataclass
class PhysicsState:
    """State of a point mass in 2D."""
    pos: np.ndarray      # (2,) position [x, y]
    vel: np.ndarray      # (2,) velocity [vx, vy]
    mass: float = 1.0
    time: float = 0.0


class SimplePhysicsSim:
    """
    Configurable 2D physics simulation.

    Start with just F=ma (no forces = constant velocity).
    Then secretly add forces the agent must discover.
    """

    def __init__(self, dt: float = 0.1):
        self.dt = dt
        self.active_forces: Dict[str, dict] = {}

    def add_force(self, name: str, force_type: ForceType, **params):
        """Add a hidden force the agent must discover."""
        self.active_forces[name] = {
            'type': force_type,
            'params': params,
        }

    def remove_force(self, name: str):
        if name in self.active_forces:
            del self.active_forces[name]

    def compute_forces(self, state: PhysicsState) -> np.ndarray:
        """Compute total force on the object."""
        total = np.zeros(2)

        for name, force in self.active_forces.items():
            ft = force['type']
            p = force['params']

            if ft == ForceType.GRAVITY:
                # F = m * g (downward)
                g = p.get('g', -9.81)
                total[1] += state.mass * g

            elif ft == ForceType.FRICTION:
                # F = -mu * m * g * v_hat (opposes motion)
                mu = p.get('mu', 0.3)
                speed = np.linalg.norm(state.vel)
                if speed > 0.01:
                    v_hat = state.vel / speed
                    total -= mu * state.mass * 9.81 * v_hat

            elif ft == ForceType.MAGNETIC:
                # F toward a fixed point (like a magnet)
                center = np.array(p.get('center', [5.0, 5.0]))
                strength = p.get('strength', 2.0)
                diff = center - state.pos
                dist = np.linalg.norm(diff) + 0.1
                total += strength * diff / (dist ** 2)

            elif ft == ForceType.BUOYANCY:
                # F = rho * V * g (upward, depends on height)
                rho = p.get('rho', 1.0)
                # Stronger at lower y (submerged more)
                depth = max(0, p.get('surface_y', 3.0) - state.pos[1])
                total[1] += rho * depth * 9.81

            elif ft == ForceType.SPRING:
                # F = -k * (pos - anchor)
                k = p.get('k', 1.0)
                anchor = np.array(p.get('anchor', [0.0, 0.0]))
                total -= k * (state.pos - anchor)

            elif ft == ForceType.DRAG:
                # F = -0.5 * Cd * rho * A * |v|^2 * v_hat
                cd = p.get('cd', 0.5)
                speed = np.linalg.norm(state.vel)
                if speed > 0.01:
                    total -= cd * speed * state.vel

        return total

    def step(self, state: PhysicsState) -> PhysicsState:
        """Advance physics by dt using semi-implicit Euler."""
        forces = self.compute_forces(state)
        acc = forces / state.mass

        new_vel = state.vel + acc * self.dt
        new_pos = state.pos + new_vel * self.dt

        return PhysicsState(
            pos=new_pos.copy(),
            vel=new_vel.copy(),
            mass=state.mass,
            time=state.time + self.dt,
        )

    def rollout(self, state: PhysicsState, n_steps: int) -> List[PhysicsState]:
        """Run n_steps of simulation."""
        trajectory = [state]
        for _ in range(n_steps):
            state = self.step(state)
            trajectory.append(state)
        return trajectory


# ──────────────────────────────────────────────────────────────────────────────
# 2. Physics Belief Model — the agent's understanding of physics
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LearnedForce:
    """A force the agent has discovered."""
    name: str
    description: str
    # The force function: (state) → force vector
    compute: callable
    confidence: float = 0.0
    discovery_time: float = 0.0
    test_count: int = 0
    test_successes: int = 0


class PhysicsBeliefModel:
    """
    The agent's current understanding of physics.

    Starts with F=ma (Newton's second law) as the only prior.
    Can add discovered forces that modify predictions.
    Tracks confidence in each discovered law.
    """

    def __init__(self, dt: float = 0.1):
        self.dt = dt
        self.discovered_forces: Dict[str, LearnedForce] = {}
        self._prediction_history: List[float] = []

    def predict(self, state: PhysicsState, n_steps: int = 1) -> List[PhysicsState]:
        """Predict future states using current belief."""
        trajectory = [state]
        current = PhysicsState(
            pos=state.pos.copy(),
            vel=state.vel.copy(),
            mass=state.mass,
            time=state.time,
        )

        for _ in range(n_steps):
            # Sum all discovered forces
            total_force = np.zeros(2)
            for name, force in self.discovered_forces.items():
                if force.confidence > 0.3:  # only use confident forces
                    total_force += force.compute(current)

            # F = ma
            acc = total_force / current.mass
            new_vel = current.vel + acc * self.dt
            new_pos = current.pos + new_vel * self.dt

            current = PhysicsState(
                pos=new_pos.copy(),
                vel=new_vel.copy(),
                mass=current.mass,
                time=current.time + self.dt,
            )
            trajectory.append(current)

        return trajectory

    def prediction_error(self, predicted: PhysicsState,
                          actual: PhysicsState) -> dict:
        """Compute structured prediction error."""
        pos_err = np.linalg.norm(predicted.pos - actual.pos)
        vel_err = np.linalg.norm(predicted.vel - actual.vel)

        # Direction of error (which way are we wrong?)
        pos_diff = actual.pos - predicted.pos
        vel_diff = actual.vel - predicted.vel

        # Infer missing acceleration
        # If we predicted vel_p but got vel_a, the missing acc is:
        missing_acc = vel_diff / self.dt

        return {
            'pos_error': pos_err,
            'vel_error': vel_err,
            'pos_diff': pos_diff,
            'vel_diff': vel_diff,
            'missing_acc': missing_acc,
            'missing_force': missing_acc * actual.mass,
        }

    def add_force(self, force: LearnedForce):
        """Add a discovered force to the belief model."""
        self.discovered_forces[force.name] = force

    def remove_force(self, name: str):
        if name in self.discovered_forces:
            del self.discovered_forces[name]

    @property
    def known_forces(self) -> List[str]:
        return list(self.discovered_forces.keys())


# ──────────────────────────────────────────────────────────────────────────────
# 3. Anomaly Classifier — is it noise, parameter error, or missing law?
# ──────────────────────────────────────────────────────────────────────────────

class AnomalyType(Enum):
    NOISE = "noise"              # random, not systematic
    PARAMETER = "parameter"      # known force, wrong parameter
    MISSING_LAW = "missing_law"  # systematic, unexplained
    NONE = "none"                # prediction is fine


@dataclass
class AnomalyReport:
    anomaly_type: AnomalyType
    confidence: float
    missing_force_estimate: np.ndarray  # estimated missing force
    direction: str                       # human-readable direction
    magnitude: float
    is_systematic: bool
    pattern: str                         # description of the pattern


class AnomalyClassifier:
    """
    Classifies prediction errors into noise, parameter errors, or
    missing physical laws.

    Key insight: systematic errors in a consistent direction indicate
    a missing force. Random errors indicate noise. Errors that scale
    with a known quantity indicate a parameter error.
    """

    def __init__(self, noise_threshold: float = 0.001,
                 systematic_threshold: float = 0.01,
                 window_size: int = 10):
        self.noise_threshold = noise_threshold
        self.systematic_threshold = systematic_threshold
        self.window_size = window_size
        self._error_history: List[dict] = []

    def classify(self, errors: List[dict]) -> AnomalyReport:
        """
        Classify a sequence of prediction errors.

        Args:
            errors: list of prediction_error() dicts from PhysicsBeliefModel
        """
        if not errors:
            return AnomalyReport(
                AnomalyType.NONE, 0.0, np.zeros(2), "none", 0.0, False, "no data")

        self._error_history.extend(errors)

        # Recent errors
        recent = errors[-self.window_size:]
        pos_errors = [e['vel_error'] for e in recent]
        missing_forces = [e['missing_force'] for e in recent]

        mean_error = np.mean(pos_errors)
        mean_force = np.mean(missing_forces, axis=0)
        force_std = np.std(missing_forces, axis=0)
        force_mag = np.linalg.norm(mean_force)

        # Classify
        if (mean_error < self.noise_threshold and np.linalg.norm(mean_force) < 0.01):
            return AnomalyReport(
                AnomalyType.NONE, 1.0 - mean_error / self.noise_threshold,
                mean_force, "none", force_mag, False, "predictions are accurate")

        # Check if errors are systematic (consistent direction)
        if len(recent) >= 3:
            # Consistency: how aligned are the missing forces?
            norms = [np.linalg.norm(f) for f in missing_forces]
            if min(norms) > 0.01:
                unit_forces = [f / (np.linalg.norm(f) + 1e-8)
                               for f in missing_forces]
                # Average alignment with mean direction
                mean_dir = mean_force / (force_mag + 1e-8)
                alignments = [np.dot(u, mean_dir) for u in unit_forces]
                mean_alignment = np.mean(alignments)
            else:
                mean_alignment = 0.0
        else:
            mean_alignment = 0.0

        is_systematic = mean_alignment > 0.7

        # Determine direction
        if force_mag > 0.01:
            angle = math.atan2(mean_force[1], mean_force[0])
            if abs(angle - math.pi/2) < 0.3:
                direction = "upward"
            elif abs(angle + math.pi/2) < 0.3:
                direction = "downward"
            elif abs(angle) < 0.3:
                direction = "rightward"
            elif abs(abs(angle) - math.pi) < 0.3:
                direction = "leftward"
            else:
                direction = f"at {math.degrees(angle):.0f} degrees"
        else:
            direction = "none"

        # Pattern description
        if is_systematic and force_mag > self.systematic_threshold:
            # Check if force is constant (gravity-like) or position-dependent
            force_variance = np.mean(force_std)
            if force_variance < force_mag * 0.3:
                pattern = f"constant {direction} force ~{force_mag:.2f}N"
            else:
                pattern = f"position-dependent {direction} force ~{force_mag:.2f}N"

            return AnomalyReport(
                AnomalyType.MISSING_LAW, mean_alignment,
                mean_force, direction, force_mag,
                True, pattern)

        elif mean_error > self.noise_threshold and not is_systematic:
            return AnomalyReport(
                AnomalyType.NOISE, 1.0 - mean_alignment,
                mean_force, direction, force_mag,
                False, "random prediction errors (noise)")

        else:
            return AnomalyReport(
                AnomalyType.PARAMETER, 0.5,
                mean_force, direction, force_mag,
                False, f"moderate {direction} bias, possibly wrong parameter")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Physics Knowledge Base — local queryable physics concepts
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PhysicsConcept:
    name: str
    description: str
    direction: str           # typical force direction
    depends_on: List[str]    # what it depends on
    force_template: callable  # (state, params) → force
    default_params: dict
    keywords: List[str]


class PhysicsKnowledgeBase:
    """
    Local knowledge base of physics concepts.
    The agent queries this when it detects a missing law.

    This is the "distilled oracle" — in Phase 2, replace with
    Claude API calls for open-ended discovery.
    """

    def __init__(self):
        self.concepts: Dict[str, PhysicsConcept] = {}
        self._populate()

    def _populate(self):
        """Seed with fundamental physics concepts."""

        self.concepts['gravity'] = PhysicsConcept(
            name='gravity',
            description='Constant downward force proportional to mass. '
                        'F = m * g, where g ≈ 9.81 m/s². '
                        'Acts on all objects with mass.',
            direction='downward',
            depends_on=['mass'],
            force_template=lambda s, p: np.array([0, s.mass * p.get('g', -9.81)]),
            default_params={'g': -9.81},
            keywords=['downward', 'constant', 'falling', 'weight', 'drop'],
        )

        self.concepts['friction'] = PhysicsConcept(
            name='friction',
            description='Force opposing motion, proportional to normal force. '
                        'F = -mu * N * v_hat. Slows objects down.',
            direction='opposing motion',
            depends_on=['velocity', 'mass'],
            force_template=lambda s, p: (
                -p.get('mu', 0.3) * s.mass * 9.81 *
                (s.vel / (np.linalg.norm(s.vel) + 1e-8))
                if np.linalg.norm(s.vel) > 0.01 else np.zeros(2)
            ),
            default_params={'mu': 0.3},
            keywords=['slowing', 'opposing', 'deceleration', 'resistance'],
        )

        self.concepts['buoyancy'] = PhysicsConcept(
            name='buoyancy',
            description='Upward force in fluid, proportional to displaced volume. '
                        'F = rho * V * g. Stronger when deeper submerged.',
            direction='upward',
            depends_on=['position', 'depth'],
            force_template=lambda s, p: np.array([
                0, p.get('rho', 1.0) *
                max(0, p.get('surface_y', 3.0) - s.pos[1]) * 9.81
            ]),
            default_params={'rho': 1.0, 'surface_y': 3.0},
            keywords=['upward', 'floating', 'fluid', 'rising', 'depth'],
        )

        self.concepts['magnetic'] = PhysicsConcept(
            name='magnetic attraction',
            description='Force toward a fixed point, inversely proportional '
                        'to distance squared. F = k * q / r². '
                        'Pulls objects toward a center.',
            direction='toward center',
            depends_on=['position', 'distance'],
            force_template=lambda s, p: (
                p.get('strength', 2.0) *
                (np.array(p.get('center', [5, 5])) - s.pos) /
                (np.linalg.norm(np.array(p.get('center', [5, 5])) - s.pos) + 0.1) ** 2
            ),
            default_params={'strength': 2.0, 'center': [5.0, 5.0]},
            keywords=['attraction', 'toward', 'center', 'pull', 'radial'],
        )

        self.concepts['spring'] = PhysicsConcept(
            name='spring/elastic',
            description='Restoring force proportional to displacement. '
                        'F = -k * x. Pulls objects back to equilibrium.',
            direction='toward anchor',
            depends_on=['position', 'displacement'],
            force_template=lambda s, p: (
                -p.get('k', 1.0) *
                (s.pos - np.array(p.get('anchor', [0, 0])))
            ),
            default_params={'k': 1.0, 'anchor': [0.0, 0.0]},
            keywords=['restoring', 'oscillation', 'elastic', 'spring', 'pull back'],
        )

        self.concepts['drag'] = PhysicsConcept(
            name='aerodynamic drag',
            description='Force opposing motion, proportional to velocity squared. '
                        'F = -0.5 * Cd * rho * A * v^2. Stronger at high speed.',
            direction='opposing motion',
            depends_on=['velocity'],
            force_template=lambda s, p: (
                -p.get('cd', 0.5) * np.linalg.norm(s.vel) * s.vel
                if np.linalg.norm(s.vel) > 0.01 else np.zeros(2)
            ),
            default_params={'cd': 0.5},
            keywords=['drag', 'air resistance', 'slowing', 'quadratic', 'speed'],
        )

    def query(self, anomaly: AnomalyReport,
              already_known: List[str]) -> List[PhysicsConcept]:
        """
        Query the knowledge base given an anomaly report.
        Returns ranked candidate concepts.

        This is the "oracle consultation" step.
        """
        candidates = []

        for name, concept in self.concepts.items():
            if concept.name in already_known:
                continue

            score = 0.0

            # Direction match
            if anomaly.direction in concept.direction:
                score += 3.0
            elif anomaly.direction == 'upward' and 'upward' in concept.direction:
                score += 3.0
            elif anomaly.direction == 'downward' and 'downward' in concept.direction:
                score += 3.0

            # Keyword match with pattern
            for kw in concept.keywords:
                if kw in anomaly.pattern.lower():
                    score += 1.0

            # Systematic = more likely a real force
            if anomaly.is_systematic:
                score += 1.0

            # Constant force → gravity/friction; variable → spring/magnetic
            if 'constant' in anomaly.pattern:
                if concept.name in ['gravity', 'friction']:
                    score += 2.0
            elif 'position-dependent' in anomaly.pattern:
                if concept.name in ['magnetic attraction', 'spring/elastic',
                                     'buoyancy']:
                    score += 2.0

            if score > 0:
                candidates.append((score, concept))

        candidates.sort(key=lambda x: -x[0])
        return [c for _, c in candidates]


# ──────────────────────────────────────────────────────────────────────────────
# 5. Hypothesis Engine — formulate and test candidate explanations
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class HypothesisResult:
    concept_name: str
    error_before: float
    error_after: float
    improvement: float
    improvement_pct: float
    accepted: bool
    params_used: dict


class HypothesisEngine:
    """
    Test candidate physics concepts against observed data.

    For each candidate:
      1. Add its force to the belief model
      2. Re-predict the observed trajectory
      3. Measure if prediction error decreases
      4. Accept if improvement > threshold
    """

    def __init__(self, acceptance_threshold: float = 0.1):
        self.acceptance_threshold = acceptance_threshold
        self._test_history: List[HypothesisResult] = []

    def test(self, concept: PhysicsConcept,
             belief: PhysicsBeliefModel,
             observations: List[PhysicsState]) -> HypothesisResult:
        """
        Test a candidate concept with parameter fitting.

        Instead of testing only default params, try multiple scales
        to find the best fit. This handles cases like gravity+buoyancy
        where the net force is less than default g.
        """
        if len(observations) < 3:
            return HypothesisResult(
                concept.name, 0, 0, 0, 0, False, concept.default_params)

        # Baseline error (current belief, no new force)
        baseline_errors = []
        for i in range(len(observations) - 1):
            pred = belief.predict(observations[i], n_steps=1)
            err = belief.prediction_error(pred[-1], observations[i + 1])
            baseline_errors.append(err['pos_error'])
        baseline_mean = np.mean(baseline_errors)

        # Try multiple parameter scales
        scales = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        best_error = baseline_mean
        best_scale = 1.0
        best_params = concept.default_params.copy()

        for scale in scales:
            # Scale the primary parameter
            scaled_params = concept.default_params.copy()
            for k, v in scaled_params.items():
                if isinstance(v, (int, float)) and k not in ('surface_y',):
                    scaled_params[k] = v * scale

            test_force = LearnedForce(
                name=concept.name,
                description=concept.description,
                compute=lambda s, c=concept, p=scaled_params: c.force_template(s, p),
                confidence=1.0,
            )
            belief.add_force(test_force)

            test_errors = []
            for i in range(len(observations) - 1):
                pred = belief.predict(observations[i], n_steps=1)
                err = belief.prediction_error(pred[-1], observations[i + 1])
                test_errors.append(err['pos_error'])
            test_mean = np.mean(test_errors)

            belief.remove_force(concept.name)

            if test_mean < best_error:
                best_error = test_mean
                best_scale = scale
                best_params = scaled_params

        improvement = baseline_mean - best_error
        improvement_pct = (improvement / (baseline_mean + 1e-8)) * 100
        accepted = improvement_pct > self.acceptance_threshold * 100

        result = HypothesisResult(
            concept_name=concept.name,
            error_before=baseline_mean,
            error_after=best_error,
            improvement=improvement,
            improvement_pct=improvement_pct,
            accepted=accepted,
            params_used=best_params,
        )
        self._test_history.append(result)
        return result


# ──────────────────────────────────────────────────────────────────────────────
# 6. Discovery Agent — the full loop
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Discovery:
    """A discovered physical law."""
    name: str
    description: str
    confidence: float
    error_reduction: float
    discovery_step: int
    observations_used: int


class DiscoveryAgent:
    """
    The full observe → predict → detect → introspect → query → revise → consolidate loop.

    This is the "Autonomous AI Scientist" prototype.
    """

    def __init__(self, dt: float = 0.1):
        self.dt = dt
        self.belief = PhysicsBeliefModel(dt=dt)
        self.classifier = AnomalyClassifier(
            noise_threshold=0.001,
            systematic_threshold=0.01,
        )
        self.knowledge_base = PhysicsKnowledgeBase()
        self.hypothesis_engine = HypothesisEngine(acceptance_threshold=0.4)

        self.discoveries: List[Discovery] = []
        self._observation_buffer: List[PhysicsState] = []
        self._error_buffer: List[dict] = []
        self._step_count = 0
        self._cortisol = 0.0

    def observe(self, state: PhysicsState):
        """Feed one observation to the agent."""
        self._observation_buffer.append(state)

        if len(self._observation_buffer) < 2:
            return

        # Predict what we expected
        prev = self._observation_buffer[-2]
        predicted = self.belief.predict(prev, n_steps=1)
        error = self.belief.prediction_error(predicted[-1], state)

        self._error_buffer.append(error)
        self._step_count += 1

        # Update cortisol (stress signal)
        self._cortisol = (0.9 * self._cortisol +
                          0.1 * error['pos_error'])

    def introspect(self) -> Optional[AnomalyReport]:
        """
        Q3: Is something wrong?
        Q8: Should I trust my prediction?

        Classify recent prediction errors.
        """
        if len(self._error_buffer) < 5:
            return None

        report = self.classifier.classify(self._error_buffer[-20:])
        return report

    def discover(self, verbose: bool = True) -> Optional[Discovery]:
        """
        Run one discovery cycle.

        Returns a Discovery if a new law was found, None otherwise.
        """
        report = self.introspect()
        if report is None:
            return None

        if report.anomaly_type == AnomalyType.NONE:
            if verbose:
                print(f"    [introspect] predictions accurate, no anomaly")
            return None

        if report.anomaly_type == AnomalyType.NOISE:
            if verbose:
                print(f"    [introspect] noise detected, not systematic")
            return None

        if report.anomaly_type != AnomalyType.MISSING_LAW:
            if verbose:
                print(f"    [introspect] anomaly type: {report.anomaly_type.value}")
            return None

        # ── Missing law detected! ──
        if verbose:
            print(f"    [ANOMALY] {report.pattern}")
            print(f"    [ANOMALY] direction={report.direction}, "
                  f"magnitude={report.magnitude:.3f}, "
                  f"systematic={report.is_systematic}")

        # Query knowledge base
        already_known = self.belief.known_forces
        candidates = self.knowledge_base.query(report, already_known)

        if not candidates:
            if verbose:
                print(f"    [query] no candidates in knowledge base")
            return None

        if verbose:
            print(f"    [query] {len(candidates)} candidates: "
                  f"{[c.name for c in candidates]}")

        # Test each candidate
        best_result = None
        best_concept = None

        recent_obs = self._observation_buffer[-30:]

        for concept in candidates[:6]:  # test top 3
            result = self.hypothesis_engine.test(
                concept, self.belief, recent_obs)

            if verbose:
                tag = "ACCEPTED" if result.accepted else "rejected"
                print(f"    [test] {concept.name}: "
                      f"error {result.error_before:.4f} -> {result.error_after:.4f} "
                      f"({result.improvement_pct:+.1f}%) [{tag}]")

            if result.accepted:
                if (best_result is None or
                        result.improvement_pct > best_result.improvement_pct):
                    best_result = result
                    best_concept = concept

        if best_concept is None:
            if verbose:
                print(f"    [discover] no hypothesis accepted")
            return None

        # ── Accept and consolidate ──
        fitted_params = best_result.params_used
        learned = LearnedForce(
            name=best_concept.name,
            description=best_concept.description,
            compute=lambda s, c=best_concept, p=fitted_params: c.force_template(s, p),
            confidence=min(1.0, best_result.improvement_pct / 100),
            discovery_time=self._step_count * self.dt,
        )
        self.belief.add_force(learned)

        discovery = Discovery(
            name=best_concept.name,
            description=best_concept.description,
            confidence=learned.confidence,
            error_reduction=best_result.improvement_pct,
            discovery_step=self._step_count,
            observations_used=len(recent_obs),
        )
        self.discoveries.append(discovery)

        # Clear error buffer (fresh start with new belief)
        self._error_buffer = []

        if verbose:
            print(f"    [DISCOVERY] '{best_concept.name}' — "
                  f"error reduced {best_result.improvement_pct:.1f}% "
                  f"(params={best_result.params_used})")

        return discovery


# ──────────────────────────────────────────────────────────────────────────────
# Scenarios
# ──────────────────────────────────────────────────────────────────────────────

def run_scenario(name: str, sim: SimplePhysicsSim,
                 initial_state: PhysicsState,
                 n_observe: int = 100,
                 discover_every: int = 20):
    """Run a discovery scenario."""
    print(f"\n{'='*65}")
    print(f"  Scenario: {name}")
    print(f"  Hidden forces: {list(sim.active_forces.keys())}")
    print(f"{'='*65}")

    agent = DiscoveryAgent(dt=sim.dt)

    # Observe the world
    trajectory = sim.rollout(initial_state, n_observe)

    for i, state in enumerate(trajectory):
        agent.observe(state)

        # Periodically try to discover
        if (i + 1) % discover_every == 0 and i > 10:
            print(f"\n  Step {i+1}/{n_observe} "
                  f"(cortisol={agent._cortisol:.4f})")
            agent.discover(verbose=True)

    # Final discovery attempt
    print(f"\n  Final discovery attempt (step {n_observe}):")
    agent.discover(verbose=True)

    # Summary
    print(f"\n  {'─'*50}")
    print(f"  Summary:")
    print(f"    Known forces: {agent.belief.known_forces}")
    print(f"    Discoveries: {len(agent.discoveries)}")
    for d in agent.discoveries:
        print(f"      - {d.name} (confidence={d.confidence:.2f}, "
              f"error reduction={d.error_reduction:.1f}%, "
              f"step={d.discovery_step})")

    # Test: are all hidden forces discovered?
    hidden = set(sim.active_forces.keys())
    found = set(agent.belief.known_forces)
    missed = hidden - found
    extra = found - hidden
    if not missed:
        print(f"    ALL FORCES DISCOVERED")
    else:
        print(f"    Missed: {missed}")
    if extra:
        print(f"    False positives: {extra}")

    return agent


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="all",
                    choices=["gravity", "friction", "magnetic",
                             "buoyancy", "combined", "all"])
    args = ap.parse_args()

    print("=" * 65)
    print("  Physics-Informed Autonomous Discovery Agent")
    print("  'A world model that does science'")
    print("=" * 65)

    dt = 0.05
    scenarios = []

    if args.scenario in ("gravity", "all"):
        sim = SimplePhysicsSim(dt=dt)
        sim.add_force("gravity", ForceType.GRAVITY, g=-9.81)
        s0 = PhysicsState(pos=np.array([0.0, 10.0]),
                          vel=np.array([2.0, 0.0]), mass=1.0)
        scenarios.append(("Falling ball (discover gravity)", sim, s0))

    if args.scenario in ("friction", "all"):
        sim = SimplePhysicsSim(dt=dt)
        sim.add_force("friction", ForceType.FRICTION, mu=0.3)
        s0 = PhysicsState(pos=np.array([0.0, 0.0]),
                          vel=np.array([5.0, 0.0]), mass=1.0)
        scenarios.append(("Sliding block (discover friction)", sim, s0))

    if args.scenario in ("magnetic", "all"):
        sim = SimplePhysicsSim(dt=dt)
        sim.add_force("magnetic attraction", ForceType.MAGNETIC,
                      center=[5.0, 5.0], strength=3.0)
        s0 = PhysicsState(pos=np.array([0.0, 0.0]),
                          vel=np.array([1.0, 1.0]), mass=1.0)
        scenarios.append(("Magnetic pull (discover magnetism)", sim, s0))

    if args.scenario in ("buoyancy", "all"):
        sim = SimplePhysicsSim(dt=dt)
        sim.add_force("gravity", ForceType.GRAVITY, g=-9.81)
        sim.add_force("buoyancy", ForceType.BUOYANCY,
                      rho=1.5, surface_y=5.0)
        s0 = PhysicsState(pos=np.array([0.0, 3.0]),
                          vel=np.array([1.0, 0.0]), mass=1.0)
        scenarios.append(("Sinking + floating (discover gravity + buoyancy)",
                          sim, s0))

    if args.scenario in ("combined", "all"):
        sim = SimplePhysicsSim(dt=dt)
        sim.add_force("gravity", ForceType.GRAVITY, g=-9.81)
        sim.add_force("drag", ForceType.DRAG, cd=0.3)
        s0 = PhysicsState(pos=np.array([0.0, 20.0]),
                          vel=np.array([3.0, 0.0]), mass=1.0)
        scenarios.append(("Projectile with drag (discover gravity + drag)",
                          sim, s0))

    all_discoveries = []
    for name, sim, s0 in scenarios:
        agent = run_scenario(name, sim, s0, n_observe=100, discover_every=20)
        all_discoveries.extend(agent.discoveries)

    print(f"\n{'='*65}")
    print(f"  Total discoveries: {len(all_discoveries)}")
    for d in all_discoveries:
        print(f"    {d.name}: error -{d.error_reduction:.1f}%, "
              f"step {d.discovery_step}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
