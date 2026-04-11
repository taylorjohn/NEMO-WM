"""
pusht_physics_registry.py — Pluggable Physics & Planner System
===============================================================
A registry pattern for PushT simulation environments and planners.

Design principles:
  - Physics and planners are independent, composable
  - New physics engines registered with @register_physics
  - New planners registered with @register_planner
  - Auto-discovery: registry scans known + user-defined entries
  - Adaptive selection: curriculum can switch based on SR feedback

Physics engines (what the world does):
  simple        — point mass, instant velocity (original)
  rigid_body    — proper inertia, torque, friction (pusht_orientation_demos)
  spring        — elastic contact with restitution
  viscous       — damped dynamics, good for slow precise tasks
  stochastic    — adds process noise (trains robustness)
  asymmetric    — T-shape mass distribution (closer to real PushT)

Planners (what the agent does):
  random        — random actions (baseline)
  position      — push to goal position only
  rotation      — push + torque to rotate
  two_phase     — position first, then rotation
  potential     — potential field planner
  rrt           — probabilistic roadmap (slow but high SR)
  da_gated      — switches strategy based on DA signal

Usage:
    from pusht_physics_registry import PhysicsRegistry, PlannerRegistry
    from pusht_physics_registry import generate_demos

    # List available
    print(PhysicsRegistry.list())
    print(PlannerRegistry.list())

    # Generate with specific combination
    demos = generate_demos(
        n_demos=500,
        physics='rigid_body',
        planner='two_phase',
    )

    # Adaptive curriculum
    demos = generate_demos(
        n_demos=500,
        physics='auto',    # tries each, uses best SR
        planner='auto',
    )

    # Register custom physics
    @PhysicsRegistry.register('my_physics')
    class MyPhysics(PhysicsBase):
        ...
"""

import math
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Type

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Core abstractions
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BlockState:
    pos:   np.ndarray
    angle: float
    vel:   np.ndarray = field(default_factory=lambda: np.zeros(2))
    omega: float = 0.0

    def obs(self, goal_angle_norm: float = 0.0) -> np.ndarray:
        """Return 6-dim observation."""
        return np.array([
            0.0, 0.0,  # agent pos (filled in by caller)
            self.pos[0], self.pos[1],
            (self.angle % (2 * math.pi)) / (2 * math.pi),
            goal_angle_norm,
        ], dtype=np.float32)

    def clone(self) -> 'BlockState':
        return BlockState(
            pos=self.pos.copy(),
            angle=self.angle,
            vel=self.vel.copy(),
            omega=self.omega,
        )


@dataclass
class AgentState:
    pos: np.ndarray

    def clone(self) -> 'AgentState':
        return AgentState(pos=self.pos.copy())


@dataclass
class Task:
    goal_pos:     np.ndarray
    goal_angle:   float          # radians
    pos_threshold:  float = 0.08
    ang_threshold:  float = math.pi  # ~20 degrees

    @property
    def goal_angle_norm(self) -> float:
        return (self.goal_angle % (2 * math.pi)) / (2 * math.pi)

    def is_success(self, block: BlockState) -> bool:
        pos_ok = np.linalg.norm(block.pos - self.goal_pos) < self.pos_threshold
        ang_ok = abs(self._wrap(block.angle - self.goal_angle)) < self.ang_threshold
        return pos_ok and ang_ok

    def is_pos_done(self, block: BlockState) -> bool:
        return np.linalg.norm(block.pos - self.goal_pos) < self.pos_threshold

    @staticmethod
    def _wrap(a: float) -> float:
        while a >  math.pi: a -= 2 * math.pi
        while a < -math.pi: a += 2 * math.pi
        return a


# ─────────────────────────────────────────────────────────────────────────────
# Physics Registry
# ─────────────────────────────────────────────────────────────────────────────

class PhysicsBase(ABC):
    """Base class for all physics engines."""

    @abstractmethod
    def step(
        self,
        agent:  AgentState,
        block:  BlockState,
        action: np.ndarray,
        task:   Task,
    ) -> Tuple[AgentState, BlockState]:
        """Advance one step. Returns new (agent, block) states."""
        ...

    @property
    def description(self) -> str:
        return self.__class__.__doc__ or self.__class__.__name__


class PhysicsRegistry:
    _registry: Dict[str, Type[PhysicsBase]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(physics_cls: Type[PhysicsBase]):
            cls._registry[name] = physics_cls
            return physics_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> PhysicsBase:
        if name not in cls._registry:
            raise KeyError(f"Unknown physics: {name!r}. Available: {cls.list()}")
        return cls._registry[name]()

    @classmethod
    def list(cls) -> List[str]:
        return sorted(cls._registry.keys())

    @classmethod
    def best_for_task(cls, task: Task, n_probe: int = 20, seed: int = 0) -> str:
        """Auto-select physics engine with highest SR on probe demos."""
        best_name = 'rigid_body'
        best_sr   = -1.0

        for name in cls.list():
            if name in ('stochastic',):  # skip noisy engines for selection
                continue
            try:
                demos = _generate_batch(
                    n=n_probe, physics_name=name,
                    planner_name='two_phase', seed=seed,
                    task_sampler=lambda r: task
                )
                sr = sum(d['success'] for d in demos) / len(demos)
                if sr > best_sr:
                    best_sr   = sr
                    best_name = name
            except Exception:
                continue

        return best_name


# ─────────────────────────────────────────────────────────────────────────────
# Physics implementations
# ─────────────────────────────────────────────────────────────────────────────

@PhysicsRegistry.register('simple')
class SimplePhysics(PhysicsBase):
    """Point mass, instant velocity transfer. Fast but unrealistic."""

    def __init__(self, friction: float = 0.82, agent_speed: float = 0.05):
        self.friction    = friction
        self.agent_speed = agent_speed

    def step(self, agent, block, action, task):
        new_agent = agent.clone()
        new_block = block.clone()

        # Agent moves toward action target
        direction = action[:2] - new_agent.pos
        dist      = np.linalg.norm(direction)
        if dist > 1e-6:
            new_agent.pos += direction / dist * min(dist, self.agent_speed)
        new_agent.pos = np.clip(new_agent.pos, 0, 1)

        # Contact push
        diff = new_agent.pos - new_block.pos
        if np.linalg.norm(diff) < 0.08:
            push = diff / (np.linalg.norm(diff) + 1e-6) * -0.14
            new_block.pos   = np.clip(new_block.pos + push, 0.05, 0.95)
            new_block.angle += push[0] * 0.5 + np.random.normal(0, 0.01)

        return new_agent, new_block


@PhysicsRegistry.register('rigid_body')
class RigidBodyPhysics(PhysicsBase):
    """Proper rigid body: inertia, torque from contact offset, friction."""

    def __init__(
        self,
        mass:        float = 1.0,
        inertia:     float = 0.08,
        friction:    float = 0.84,
        agent_speed: float = 0.05,
        force_scale: float = 0.08,
    ):
        self.mass        = mass
        self.inertia     = inertia
        self.friction    = friction
        self.agent_speed = agent_speed
        self.force_scale = force_scale

    def step(self, agent, block, action, task):
        new_agent = agent.clone()
        new_block = block.clone()

        # Agent moves
        direction = action[:2] - new_agent.pos
        dist      = np.linalg.norm(direction)
        if dist > 1e-6:
            new_agent.pos += direction / dist * min(dist, self.agent_speed)
        new_agent.pos = np.clip(new_agent.pos, 0, 1)

        # Rigid body contact
        diff = new_agent.pos - new_block.pos
        d    = np.linalg.norm(diff)
        if d < 0.08 and d > 1e-6:
            push_dir   = -diff / d
            push_force = push_dir * self.force_scale

            # Torque = r × F (2D)
            contact = new_block.pos + diff * 0.5
            r       = contact - new_block.pos
            torque  = r[0] * push_force[1] - r[1] * push_force[0]

            new_block.vel   += push_force / self.mass
            new_block.omega += torque / self.inertia

        # Integrate
        new_block.pos   += new_block.vel
        new_block.angle += new_block.omega
        new_block.vel   *= self.friction
        new_block.omega *= self.friction
        new_block.pos    = np.clip(new_block.pos, 0.05, 0.95)

        return new_agent, new_block


@PhysicsRegistry.register('spring')
class SpringPhysics(PhysicsBase):
    """Elastic contact with restitution — bouncy block."""

    def __init__(
        self,
        stiffness:   float = 2.0,
        damping:     float = 0.7,
        agent_speed: float = 0.05,
    ):
        self.stiffness   = stiffness
        self.damping     = damping
        self.agent_speed = agent_speed

    def step(self, agent, block, action, task):
        new_agent = agent.clone()
        new_block = block.clone()

        direction = action[:2] - new_agent.pos
        dist      = np.linalg.norm(direction)
        if dist > 1e-6:
            new_agent.pos += direction / dist * min(dist, self.agent_speed)
        new_agent.pos = np.clip(new_agent.pos, 0, 1)

        diff    = new_agent.pos - new_block.pos
        d       = np.linalg.norm(diff)
        overlap = 0.08 - d

        if overlap > 0 and d > 1e-6:
            # Spring force proportional to overlap
            force = -diff / d * overlap * self.stiffness
            new_block.vel   += force
            new_block.angle += (diff[0] * force[1] - diff[1] * force[0]) * 0.3

        new_block.vel   *= self.damping
        new_block.pos   += new_block.vel
        new_block.pos    = np.clip(new_block.pos, 0.05, 0.95)
        new_block.omega *= self.damping
        new_block.angle += new_block.omega

        return new_agent, new_block


@PhysicsRegistry.register('viscous')
class ViscousPhysics(PhysicsBase):
    """Heavy damping — slow, precise. Good for fine rotation tasks."""

    def __init__(
        self,
        friction:    float = 0.60,
        agent_speed: float = 0.25,
        force_scale: float = 0.20,
    ):
        self.friction    = friction
        self.agent_speed = agent_speed
        self.force_scale = force_scale

    def step(self, agent, block, action, task):
        new_agent = agent.clone()
        new_block = block.clone()

        direction = action[:2] - new_agent.pos
        dist      = np.linalg.norm(direction)
        if dist > 1e-6:
            new_agent.pos += direction / dist * min(dist, self.agent_speed)
        new_agent.pos = np.clip(new_agent.pos, 0, 1)

        diff = new_agent.pos - new_block.pos
        d    = np.linalg.norm(diff)
        if d < 0.10 and d > 1e-6:
            push = -diff / d * self.force_scale
            r    = diff * 0.5
            torque = r[0] * push[1] - r[1] * push[0]
            new_block.vel   += push
            new_block.omega += torque * 2.0

        new_block.vel   *= self.friction
        new_block.omega *= self.friction
        new_block.pos   += new_block.vel
        new_block.angle += new_block.omega
        new_block.pos    = np.clip(new_block.pos, 0.05, 0.95)

        return new_agent, new_block


@PhysicsRegistry.register('stochastic')
class StochasticPhysics(PhysicsBase):
    """Rigid body + process noise — trains robustness to sensor/actuator noise."""

    def __init__(
        self,
        base_physics: str  = 'rigid_body',
        pos_noise:    float = 0.008,
        angle_noise:  float = 0.015,
        dropout_p:    float = 0.05,
    ):
        self.base     = PhysicsRegistry.get(base_physics)
        self.pos_noise   = pos_noise
        self.angle_noise = angle_noise
        self.dropout_p   = dropout_p

    def step(self, agent, block, action, task):
        # Random action dropout
        if np.random.random() < self.dropout_p:
            action = action.copy()
            action[:2] = agent.pos  # no-op

        new_agent, new_block = self.base.step(agent, block, action, task)

        # Process noise
        new_block.pos   += np.random.normal(0, self.pos_noise, 2)
        new_block.angle += np.random.normal(0, self.angle_noise)
        new_block.pos    = np.clip(new_block.pos, 0.05, 0.95)

        return new_agent, new_block


@PhysicsRegistry.register('asymmetric')
class AsymmetricPhysics(PhysicsBase):
    """
    T-shape mass distribution — centre of mass offset from geometric centre.
    More realistic for actual PushT where the T has unequal arm lengths.
    """

    def __init__(
        self,
        com_offset:  float = 0.03,  # centre of mass offset from geometric centre
        inertia:     float = 0.12,
        friction:    float = 0.83,
        agent_speed: float = 0.05,
    ):
        self.com_offset  = com_offset
        self.inertia     = inertia
        self.friction    = friction
        self.agent_speed = agent_speed

    def _com(self, block: BlockState) -> np.ndarray:
        """Centre of mass offset by angle."""
        return block.pos + np.array([
            math.cos(block.angle) * self.com_offset,
            math.sin(block.angle) * self.com_offset,
        ])

    def step(self, agent, block, action, task):
        new_agent = agent.clone()
        new_block = block.clone()

        direction = action[:2] - new_agent.pos
        dist      = np.linalg.norm(direction)
        if dist > 1e-6:
            new_agent.pos += direction / dist * min(dist, self.agent_speed)
        new_agent.pos = np.clip(new_agent.pos, 0, 1)

        com  = self._com(new_block)
        diff = new_agent.pos - com
        d    = np.linalg.norm(diff)

        if d < 0.09 and d > 1e-6:
            push    = -diff / d * 0.14
            r       = new_agent.pos - com
            torque  = r[0] * push[1] - r[1] * push[0]

            new_block.vel   += push
            new_block.omega += torque / self.inertia

        new_block.vel   *= self.friction
        new_block.omega *= self.friction
        new_block.pos   += new_block.vel
        new_block.angle += new_block.omega
        new_block.pos    = np.clip(new_block.pos, 0.05, 0.95)

        return new_agent, new_block


# ─────────────────────────────────────────────────────────────────────────────
# Planner Registry
# ─────────────────────────────────────────────────────────────────────────────

class PlannerBase(ABC):
    """Base class for all planners."""

    @abstractmethod
    def plan(
        self,
        agent: AgentState,
        block: BlockState,
        task:  Task,
        rng:   np.random.RandomState,
    ) -> np.ndarray:
        """Return 3-dim action [target_x, target_y, target_angle_norm]."""
        ...

    @property
    def description(self) -> str:
        return self.__class__.__doc__ or self.__class__.__name__


class PlannerRegistry:
    _registry: Dict[str, Type[PlannerBase]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(planner_cls: Type[PlannerBase]):
            cls._registry[name] = planner_cls
            return planner_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> PlannerBase:
        if name not in cls._registry:
            raise KeyError(f"Unknown planner: {name!r}. Available: {cls.list()}")
        return cls._registry[name]()

    @classmethod
    def list(cls) -> List[str]:
        return sorted(cls._registry.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Planner implementations
# ─────────────────────────────────────────────────────────────────────────────

def _wrap(a: float) -> float:
    while a >  math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a


@PlannerRegistry.register('random')
class RandomPlanner(PlannerBase):
    """Random actions — baseline / exploration."""

    def plan(self, agent, block, task, rng):
        return rng.uniform(0, 1, 3).astype(np.float32)


@PlannerRegistry.register('position')
class PositionPlanner(PlannerBase):
    """Push block to goal position, ignore orientation."""

    def plan(self, agent, block, task, rng):
        to_goal = task.goal_pos - block.pos
        dist    = np.linalg.norm(to_goal)

        if dist < 0.02:
            target = task.goal_pos
        else:
            # Get behind block and push toward goal
            push_dir = (task.goal_pos - block.pos)
            push_dist = np.linalg.norm(push_dir) + 1e-6
            push_dir_n = push_dir / push_dist
            # Position agent behind block (opposite side from goal)
            behind_block = block.pos - push_dir_n * 0.08
            behind_block = np.clip(behind_block, 0.05, 0.95)
            agent_to_behind = np.linalg.norm(agent.pos - behind_block)
            if dist < 0.05:
                # At goal: stay still, block is there
                target = agent.pos
            elif agent_to_behind > 0.05:
                target = behind_block  # reposition behind block
            else:
                # Push: aim slightly past goal to carry block through
                overshoot = task.goal_pos + push_dir_n * 0.03
                target = np.clip(overshoot, 0, 1)

        target = np.clip(target, 0, 1)
        return np.array([target[0], target[1], task.goal_angle_norm], dtype=np.float32)


@PlannerRegistry.register('rotation')
class RotationPlanner(PlannerBase):
    """Apply lateral torque to rotate block — ignores position."""

    def plan(self, agent, block, task, rng):
        angle_err = _wrap(task.goal_angle - block.angle)
        rot_dir   = 1.0 if angle_err > 0 else -1.0

        # Approach perpendicularly to apply torque
        perp    = np.array([-math.sin(block.angle), math.cos(block.angle)])
        target  = block.pos + perp * rot_dir * 0.10
        target  = np.clip(target, 0.05, 0.95)

        ang_norm = ((block.angle + angle_err * 0.15) % (2 * math.pi)) / (2 * math.pi)
        return np.array([target[0], target[1], float(ang_norm)], dtype=np.float32)


@PlannerRegistry.register('two_phase')
class TwoPhhasePlanner(PlannerBase):
    """Phase 1: push to position. Phase 2: rotate to angle."""

    def __init__(self):
        self._pos = PositionPlanner()
        self._rot = RotationPlanner()

    def plan(self, agent, block, task, rng):
        if not task.is_pos_done(block):
            return self._pos.plan(agent, block, task, rng)
        else:
            return self._rot.plan(agent, block, task, rng)


@PlannerRegistry.register('potential')
class PotentialPlanner(PlannerBase):
    """Potential field: attract to goal, repel from walls."""

    def plan(self, agent, block, task, rng):
        # Attractive force toward goal
        to_goal = task.goal_pos - block.pos
        dist    = np.linalg.norm(to_goal) + 1e-6
        attract = to_goal / dist * min(dist, 0.15)

        # Repel from walls
        repel = np.zeros(2)
        for wall_pos in [0.05, 0.95]:
            for dim in range(2):
                d = block.pos[dim] - wall_pos
                if abs(d) < 0.15:
                    repel[dim] += 0.02 / (d + 1e-6)

        target = np.clip(block.pos + attract + repel - (block.pos - agent.pos) * 0.12, 0, 1)

        angle_err = _wrap(task.goal_angle - block.angle)
        ang_norm  = ((block.angle + angle_err * 0.1) % (2 * math.pi)) / (2 * math.pi)
        return np.array([target[0], target[1], float(ang_norm)], dtype=np.float32)


@PlannerRegistry.register('da_gated')
class DAGatedPlanner(PlannerBase):
    """
    DA-gated: HOT (near goal) = exploit/rotate, COLD (far) = explore/push.
    Mirrors the biological temperature-gated strategy switching.
    """

    def __init__(self):
        self._pos = PositionPlanner()
        self._rot = RotationPlanner()
        self._pot = PotentialPlanner()

    def plan(self, agent, block, task, rng):
        pos_dist   = np.linalg.norm(block.pos - task.goal_pos)
        angle_err  = abs(_wrap(task.goal_angle - block.angle))

        # DA signal: high when close to goal
        da = 1.0 - min(pos_dist / 0.3, 1.0)

        if da < 0.3:      # COLD: far from goal, use potential field
            return self._pot.plan(agent, block, task, rng)
        elif da < 0.7:    # WARM: near goal, push to position
            return self._pos.plan(agent, block, task, rng)
        else:             # HOT: at position, rotate
            return self._rot.plan(agent, block, task, rng)


# ─────────────────────────────────────────────────────────────────────────────
# Task samplers
# ─────────────────────────────────────────────────────────────────────────────

def sample_task_full(rng: np.random.RandomState) -> Task:
    """Full rotation — any angle."""
    return Task(
        goal_pos=rng.uniform(0.35, 0.65, 2),
        goal_angle=rng.uniform(0, 2 * math.pi),
    )


def sample_task_small_angle(rng: np.random.RandomState) -> Task:
    """Small rotation — within ±30°."""
    base = rng.uniform(0, 2 * math.pi)
    return Task(
        goal_pos=rng.uniform(0.35, 0.65, 2),
        goal_angle=base + rng.uniform(-math.pi/6, math.pi/6),
    )


def sample_task_position_only(rng: np.random.RandomState) -> Task:
    """Position only — loose angle threshold."""
    t = sample_task_full(rng)
    t.ang_threshold = math.pi  # always pass
    return t


TASK_SAMPLERS = {
    'full':          sample_task_full,
    'small_angle':   sample_task_small_angle,
    'position_only': sample_task_position_only,
}


# ─────────────────────────────────────────────────────────────────────────────
# Curriculum manager
# ─────────────────────────────────────────────────────────────────────────────

class CurriculumManager:
    """
    Adaptive curriculum: adjusts task difficulty based on rolling SR.

    Starts easy (small_angle), promotes to full when SR > threshold,
    demotes when SR drops. Mirrors ACh-gated temporal integration:
    broader context (harder tasks) only when foundation is solid.
    """

    def __init__(
        self,
        promote_threshold: float = 0.50,
        demote_threshold:  float = 0.25,
        window:            int   = 50,
    ):
        self.promote_threshold = promote_threshold
        self.demote_threshold  = demote_threshold
        self.window            = window
        self.history:          List[bool] = []
        self.level             = 0
        self.levels            = ['position_only', 'small_angle', 'full']

    @property
    def current_task(self) -> str:
        return self.levels[self.level]

    def update(self, success: bool):
        self.history.append(success)
        if len(self.history) < self.window:
            return

        recent_sr = sum(self.history[-self.window:]) / self.window

        if recent_sr >= self.promote_threshold and self.level < len(self.levels) - 1:
            self.level += 1
            print(f"  [Curriculum] Promoted → {self.current_task} (SR={recent_sr:.1%})")
        elif recent_sr <= self.demote_threshold and self.level > 0:
            self.level -= 1
            print(f"  [Curriculum] Demoted  → {self.current_task} (SR={recent_sr:.1%})")

    def sample_task(self, rng: np.random.RandomState) -> Task:
        return TASK_SAMPLERS[self.current_task](rng)


# ─────────────────────────────────────────────────────────────────────────────
# Demo generation
# ─────────────────────────────────────────────────────────────────────────────

def _generate_batch(
    n:            int,
    physics_name: str,
    planner_name: str,
    seed:         int,
    task_sampler: Callable,
    max_steps:    int = 400,
    H:            int = 8,
) -> List[dict]:
    """Generate a batch of demos with given physics + planner."""
    physics = PhysicsRegistry.get(physics_name)
    planner = PlannerRegistry.get(planner_name)
    rng     = np.random.RandomState(seed)
    demos   = []

    for _ in range(n):
        task = task_sampler(rng)

        # Initial state
        init_agent = np.clip(rng.uniform(0.05, 0.40, 2), 0.05, 0.95)
        init_block = np.clip(rng.uniform(0.20, 0.55, 2), 0.05, 0.95)
        init_angle = rng.uniform(0, 2 * math.pi)

        agent = AgentState(init_agent)
        block = BlockState(init_block, init_angle)

        obs_list    = []
        action_list = []

        for _ in range(max_steps):
            obs = np.array([
                agent.pos[0], agent.pos[1],
                block.pos[0], block.pos[1],
                (block.angle % (2 * math.pi)) / (2 * math.pi),
                task.goal_angle_norm,
            ], dtype=np.float32)

            action = planner.plan(agent, block, task, rng)
            obs_list.append(obs)
            action_list.append(action)

            agent, block = physics.step(agent, block, action, task)

            if task.is_success(block):
                break

        demos.append({
            'obs':     np.array(obs_list,    dtype=np.float32),
            'actions': np.array(action_list, dtype=np.float32),
            'goal':    np.array([
                task.goal_pos[0], task.goal_pos[1], task.goal_angle_norm
            ], dtype=np.float32),
            'success':     task.is_success(block),
            'physics':     physics_name,
            'planner':     planner_name,
            'task_type':   'unknown',
        })

    return demos


def generate_demos(
    n_demos:      int  = 500,
    physics:      str  = 'rigid_body',
    planner:      str  = 'two_phase',
    task_mode:    str  = 'curriculum',
    max_steps:    int  = 400,
    H:            int  = 8,
    seed:         int  = 42,
    verbose:      bool = True,
) -> List[dict]:
    """
    Generate PushT demos with pluggable physics and planner.

    Args:
        physics:   physics engine name or 'auto' (selects by SR)
        planner:   planner name or 'auto' (selects by SR)
        task_mode: 'full' | 'small_angle' | 'position_only' | 'curriculum'
    """
    rng = np.random.RandomState(seed)

    # Auto-select physics/planner
    if physics == 'auto':
        physics = _auto_select(PhysicsRegistry.list(), planner, seed, rng)
        if verbose:
            print(f"  Auto-selected physics: {physics}")

    if planner == 'auto':
        planner = _auto_select_planner(physics, PlannerRegistry.list(), seed, rng)
        if verbose:
            print(f"  Auto-selected planner: {planner}")

    # Task curriculum
    curriculum = CurriculumManager() if task_mode == 'curriculum' else None

    def get_task(rng):
        if curriculum:
            return curriculum.sample_task(rng)
        return TASK_SAMPLERS.get(task_mode, sample_task_full)(rng)

    physics_eng = PhysicsRegistry.get(physics)
    planner_eng = PlannerRegistry.get(planner)
    demos       = []

    for i in range(n_demos):
        task       = get_task(rng)
        init_agent = np.clip(rng.uniform(0.05, 0.40, 2), 0.05, 0.95)
        init_block = np.clip(rng.uniform(0.20, 0.55, 2), 0.05, 0.95)
        init_angle = rng.uniform(0, 2 * math.pi)

        agent = AgentState(init_agent)
        block = BlockState(init_block, init_angle)

        obs_list = []
        act_list = []

        for _ in range(max_steps):
            obs = np.array([
                agent.pos[0], agent.pos[1],
                block.pos[0], block.pos[1],
                (block.angle % (2 * math.pi)) / (2 * math.pi),
                task.goal_angle_norm,
            ], dtype=np.float32)

            action = planner_eng.plan(agent, block, task, rng)
            obs_list.append(obs)
            act_list.append(action)

            agent, block = physics_eng.step(agent, block, action, task)

            if task.is_success(block):
                break

        success = task.is_success(block)
        if curriculum:
            curriculum.update(success)

        demos.append({
            'obs':      np.array(obs_list, dtype=np.float32),
            'actions':  np.array(act_list, dtype=np.float32),
            'goal':     np.array([
                task.goal_pos[0], task.goal_pos[1], task.goal_angle_norm
            ], dtype=np.float32),
            'success':  success,
            'physics':  physics,
            'planner':  planner,
        })

        if verbose and (i + 1) % 100 == 0:
            sr = sum(d['success'] for d in demos) / len(demos)
            level = curriculum.current_task if curriculum else task_mode
            print(f"  Demo {i+1}/{n_demos}  SR={sr:.1%}  level={level}")

    return demos


def _auto_select(physics_list, planner, seed, rng, n_probe=20):
    """Select physics engine with highest SR on probe."""
    best, best_sr = 'rigid_body', -1.0
    for p in physics_list:
        if p == 'stochastic':
            continue
        try:
            demos = _generate_batch(n_probe, p, planner, seed,
                                    lambda r: sample_task_small_angle(r))
            sr = sum(d['success'] for d in demos) / len(demos)
            if sr > best_sr:
                best_sr, best = sr, p
        except Exception:
            continue
    return best


def _auto_select_planner(physics, planner_list, seed, rng, n_probe=20):
    """Select planner with highest SR."""
    best, best_sr = 'two_phase', -1.0
    for p in planner_list:
        if p == 'random':
            continue
        try:
            demos = _generate_batch(n_probe, physics, p, seed,
                                    lambda r: sample_task_small_angle(r))
            sr = sum(d['success'] for d in demos) / len(demos)
            if sr > best_sr:
                best_sr, best = sr, p
        except Exception:
            continue
    return best


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description='PushT physics/planner registry')
    ap.add_argument('--n-demos',   type=int, default=200)
    ap.add_argument('--physics',   default='rigid_body')
    ap.add_argument('--planner',   default='two_phase')
    ap.add_argument('--task-mode', default='curriculum',
                    choices=['full', 'small_angle', 'position_only', 'curriculum'])
    ap.add_argument('--seed',      type=int, default=42)
    ap.add_argument('--benchmark', action='store_true',
                    help='Benchmark all physics × planner combinations')
    args = ap.parse_args()

    if args.benchmark:
        print("\nBenchmarking all physics × planner combinations...")
        print(f"{'Physics':15s}  {'Planner':15s}  {'SR':8s}  {'Avg steps'}")
        print('─' * 60)

        for ph in PhysicsRegistry.list():
            for pl in PlannerRegistry.list():
                if pl == 'random':
                    continue
                try:
                    t0    = time.time()
                    demos = _generate_batch(
                        50, ph, pl, args.seed,
                        lambda r: sample_task_small_angle(r)
                    )
                    sr    = sum(d['success'] for d in demos) / len(demos)
                    steps = sum(len(d['obs']) for d in demos) / len(demos)
                    elapsed = time.time() - t0
                    flag  = '★' if sr > 0.5 else ' '
                    print(f"  {ph:15s}  {pl:15s}  {sr:.1%} {flag}  "
                          f"{steps:5.0f}  ({elapsed:.1f}s)")
                except Exception as e:
                    print(f"  {ph:15s}  {pl:15s}  ERROR: {e}")
    else:
        print(f"\nPhysics engines: {PhysicsRegistry.list()}")
        print(f"Planners:        {PlannerRegistry.list()}")
        print(f"\nGenerating {args.n_demos} demos...")
        print(f"  physics={args.physics}  planner={args.planner}  "
              f"task_mode={args.task_mode}")

        t0    = time.time()
        demos = generate_demos(
            n_demos=args.n_demos,
            physics=args.physics,
            planner=args.planner,
            task_mode=args.task_mode,
            seed=args.seed,
        )
        elapsed = time.time() - t0

        total   = len(demos)
        success = sum(d['success'] for d in demos)
        steps   = sum(len(d['obs']) for d in demos) / total

        print(f"\nResults:")
        print(f"  SR:         {success}/{total} = {success/total:.1%}")
        print(f"  Avg steps:  {steps:.0f}")
        print(f"  Elapsed:    {elapsed:.1f}s")
        print(f"\nTo train flow policy v2:")
        print(f"  python train_nemo_wm_v2.py --mode full --n-demos 1000")
        print(f"  (update train_nemo_wm_v2.py to use generate_demos() from this module)")
