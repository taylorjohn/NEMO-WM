"""
adaptive_learner.py — Three Modes of Autonomous Learning
==========================================================
Extends the physics discovery agent with:

1. InstructionLearner — "investigate friction at different speeds"
   Parses natural language instructions into experiment designs,
   runs targeted observations, and discovers the specified phenomenon.

2. AutoLearner — when the knowledge base has no answer
   Fits a general force model (polynomial basis) to the residual
   error. Discovers forces with NO prior knowledge of what they
   might be. The agent writes its own physics.

3. OracleLearner — ask an LLM when stuck
   Formats the anomaly as a natural language question, queries
   a local knowledge base or Claude API, parses the response
   into a testable force template.

Usage:
    from adaptive_learner import InstructionLearner, AutoLearner, OracleLearner

    # Mode 1: instruction-driven
    learner = InstructionLearner(agent, sim)
    learner.investigate("what force acts on objects near y=3?")

    # Mode 2: auto-learn
    auto = AutoLearner(agent)
    auto.fit_unknown_force(observations)

    # Mode 3: LLM oracle
    oracle = OracleLearner(agent, use_api=False)
    oracle.consult(anomaly_report)
"""

import math
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from physics_discovery_agent import (
    ForceType,
    PhysicsState, SimplePhysicsSim, PhysicsBeliefModel,
    LearnedForce, AnomalyClassifier, AnomalyReport, AnomalyType,
    PhysicsConcept, HypothesisEngine, HypothesisResult,
    DiscoveryAgent, Discovery,
)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Instruction-Driven Learning
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Experiment:
    """A designed experiment to test a hypothesis."""
    name: str
    description: str
    initial_states: List[PhysicsState]
    variable: str        # what we're varying
    prediction: str      # what we expect if hypothesis is true


class InstructionLearner:
    """
    Parses natural language instructions into experiments.

    Examples:
      "investigate what happens at different heights"
        → vary y position, measure force at each height
      "test if friction depends on speed"
        → launch at different speeds, measure deceleration
      "check for forces near position (5, 5)"
        → place object at varying distances from (5,5)

    The instruction parser maps keywords to experiment designs.
    Phase 2 replaces this with LLM-based instruction parsing.
    """

    def __init__(self, agent: DiscoveryAgent, sim: SimplePhysicsSim):
        self.agent = agent
        self.sim = sim
        self._experiments_run = 0

    def parse_instruction(self, instruction: str) -> List[Experiment]:
        """Parse instruction into experiment designs."""
        instruction = instruction.lower()
        experiments = []

        if any(w in instruction for w in ['height', 'depth', 'vertical',
                                            'fall', 'drop', 'y pos']):
            # Vary vertical position
            experiments.append(Experiment(
                name="height_sweep",
                description="Test at different heights",
                initial_states=[
                    PhysicsState(np.array([1.0, h]), np.array([0.5, 0.0]))
                    for h in [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
                ],
                variable="y_position",
                prediction="force varies with height → position-dependent",
            ))

        if any(w in instruction for w in ['speed', 'velocity', 'fast',
                                            'slow', 'friction', 'drag']):
            # Vary speed
            experiments.append(Experiment(
                name="speed_sweep",
                description="Test at different speeds",
                initial_states=[
                    PhysicsState(np.array([0.0, 0.0]), np.array([v, 0.0]))
                    for v in [0.5, 1.0, 2.0, 5.0, 10.0]
                ],
                variable="speed",
                prediction="force proportional to speed → friction/drag",
            ))

        if any(w in instruction for w in ['near', 'position', 'distance',
                                            'attract', 'pull', 'center']):
            # Vary distance from a point
            center = [5.0, 5.0]
            experiments.append(Experiment(
                name="distance_sweep",
                description=f"Test at different distances from {center}",
                initial_states=[
                    PhysicsState(np.array([5.0 - d, 5.0 - d]),
                                np.array([0.0, 0.0]))
                    for d in [0.5, 1.0, 2.0, 3.0, 5.0]
                ],
                variable="distance",
                prediction="force increases with proximity → attraction",
            ))

        if any(w in instruction for w in ['mass', 'weight', 'heavy',
                                            'light']):
            # Vary mass
            experiments.append(Experiment(
                name="mass_sweep",
                description="Test with different masses",
                initial_states=[
                    PhysicsState(np.array([0.0, 5.0]), np.array([1.0, 0.0]),
                                mass=m)
                    for m in [0.5, 1.0, 2.0, 5.0, 10.0]
                ],
                variable="mass",
                prediction="force proportional to mass → gravitational",
            ))

        if not experiments:
            # Default: general sweep
            experiments.append(Experiment(
                name="general_sweep",
                description="General exploration at multiple positions",
                initial_states=[
                    PhysicsState(np.array([x, y]), np.array([1.0, 0.0]))
                    for x in [0, 3, 6] for y in [0, 3, 6]
                ],
                variable="position",
                prediction="look for any systematic forces",
            ))

        return experiments

    def investigate(self, instruction: str,
                     n_steps: int = 30,
                     verbose: bool = True) -> List[dict]:
        """
        Run instruction-driven investigation.

        Returns list of findings per experiment.
        """
        experiments = self.parse_instruction(instruction)
        findings = []

        if verbose:
            print(f"\n  Instruction: '{instruction}'")
            print(f"  Experiments designed: {len(experiments)}")

        for exp in experiments:
            if verbose:
                print(f"\n  ── Experiment: {exp.name} ──")
                print(f"  {exp.description}")
                print(f"  Variable: {exp.variable}")
                print(f"  Trials: {len(exp.initial_states)}")

            trial_results = []

            for i, s0 in enumerate(exp.initial_states):
                # Reset agent for each trial
                trial_agent = DiscoveryAgent(dt=self.sim.dt)

                # Run observation
                trajectory = self.sim.rollout(s0, n_steps)
                for state in trajectory:
                    trial_agent.observe(state)

                # Measure residual force
                errors = trial_agent._error_buffer
                if errors:
                    mean_force = np.mean(
                        [e['missing_force'] for e in errors], axis=0)
                    mean_error = np.mean([e['vel_error'] for e in errors])
                else:
                    mean_force = np.zeros(2)
                    mean_error = 0.0

                var_val = {
                    'y_position': s0.pos[1],
                    'speed': np.linalg.norm(s0.vel),
                    'distance': np.linalg.norm(s0.pos - np.array([5, 5])),
                    'mass': s0.mass,
                    'position': tuple(s0.pos),
                }.get(exp.variable, i)

                trial_results.append({
                    'variable_value': var_val,
                    'mean_force': mean_force,
                    'force_magnitude': np.linalg.norm(mean_force),
                    'force_direction': math.degrees(
                        math.atan2(mean_force[1], mean_force[0])),
                    'mean_error': mean_error,
                })

                if verbose:
                    fm = np.linalg.norm(mean_force)
                    fd = math.degrees(math.atan2(mean_force[1], mean_force[0]))
                    print(f"    {exp.variable}={var_val:.1f}: "
                          f"force={fm:.3f}N at {fd:.0f} deg")

            # Analyze: does force depend on the variable?
            vals = [t['variable_value'] for t in trial_results
                    if isinstance(t['variable_value'], (int, float))]
            mags = [t['force_magnitude'] for t in trial_results]

            if len(vals) >= 3 and len(mags) >= 3:
                correlation = abs(np.corrcoef(vals, mags)[0, 1])
            else:
                correlation = 0.0

            finding = {
                'experiment': exp.name,
                'variable': exp.variable,
                'correlation': correlation,
                'force_depends_on_variable': correlation > 0.5,
                'trials': trial_results,
                'prediction': exp.prediction,
            }
            findings.append(finding)

            if verbose:
                dep = "YES" if correlation > 0.5 else "no"
                print(f"  Correlation({exp.variable}, force): "
                      f"{correlation:.3f} → depends: {dep}")

            self._experiments_run += 1

        return findings


# ──────────────────────────────────────────────────────────────────────────────
# 2. Auto-Learn — fit unknown forces mathematically
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FittedForce:
    """A force discovered by basis function fitting."""
    name: str
    coefficients: np.ndarray
    basis_names: List[str]
    r_squared: float
    equation: str


class AutoLearner:
    """
    When the knowledge base has no answer, fit a general force model.

    Uses polynomial basis functions over state variables:
      F_x = a0 + a1*x + a2*y + a3*vx + a4*vy + a5*x^2 + a6*y^2 + ...
      F_y = b0 + b1*x + b2*y + ...

    The agent discovers the mathematical form of the force without
    knowing what it's called. It writes its own physics.

    This is the "genuinely novel discovery" mode.
    """

    def __init__(self, max_degree: int = 2):
        self.max_degree = max_degree
        self._discoveries: List[FittedForce] = []

    def _build_basis(self, state: PhysicsState) -> np.ndarray:
        """Build polynomial basis features from state."""
        x, y = state.pos
        vx, vy = state.vel
        m = state.mass

        features = [1.0]  # constant
        names = ['const']

        # Linear
        for val, name in [(x, 'x'), (y, 'y'), (vx, 'vx'), (vy, 'vy')]:
            features.append(val)
            names.append(name)

        # Quadratic
        if self.max_degree >= 2:
            for val, name in [(x*x, 'x^2'), (y*y, 'y^2'),
                              (vx*vx, 'vx^2'), (vy*vy, 'vy^2'),
                              (x*y, 'x*y'), (vx*vy, 'vx*vy')]:
                features.append(val)
                names.append(name)

        # Special: 1/r^2 from origin and from (5,5)
        r_origin = math.sqrt(x*x + y*y) + 0.1
        r_center = math.sqrt((x-5)**2 + (y-5)**2) + 0.1
        features.extend([1/r_origin, 1/r_center,
                          1/(r_origin**2), 1/(r_center**2)])
        names.extend(['1/r_origin', '1/r_center',
                       '1/r_origin^2', '1/r_center^2'])

        # Speed
        speed = math.sqrt(vx*vx + vy*vy) + 0.01
        features.extend([speed, speed**2])
        names.extend(['|v|', '|v|^2'])

        self._basis_names = names
        return np.array(features)

    def fit_unknown_force(self, observations: List[PhysicsState],
                           belief: PhysicsBeliefModel,
                           verbose: bool = True) -> Optional[FittedForce]:
        """
        Fit a general polynomial force model to prediction errors.

        Uses least squares: find coefficients c such that
        F_missing ≈ Phi(state) @ c
        """
        if len(observations) < 10:
            return None

        # Collect (state, missing_force) pairs
        X_list = []
        Fx_list = []
        Fy_list = []

        for i in range(len(observations) - 1):
            pred = belief.predict(observations[i], n_steps=1)
            err = belief.prediction_error(pred[-1], observations[i + 1])

            phi = self._build_basis(observations[i])
            X_list.append(phi)
            Fx_list.append(err['missing_force'][0])
            Fy_list.append(err['missing_force'][1])

        X = np.array(X_list)
        Fx = np.array(Fx_list)
        Fy = np.array(Fy_list)

        # Least squares fit for each component
        try:
            cx, res_x, _, _ = np.linalg.lstsq(X, Fx, rcond=None)
            cy, res_y, _, _ = np.linalg.lstsq(X, Fy, rcond=None)
        except np.linalg.LinAlgError:
            return None

        # R-squared
        Fx_pred = X @ cx
        Fy_pred = X @ cy
        ss_res_x = np.sum((Fx - Fx_pred) ** 2)
        ss_res_y = np.sum((Fy - Fy_pred) ** 2)
        ss_tot_x = np.sum((Fx - np.mean(Fx)) ** 2) + 1e-8
        ss_tot_y = np.sum((Fy - np.mean(Fy)) ** 2) + 1e-8
        r2_x = 1 - ss_res_x / ss_tot_x
        r2_y = 1 - ss_res_y / ss_tot_y
        r2 = (r2_x + r2_y) / 2

        # Build equation string (only significant terms)
        threshold = 0.01 * max(np.max(np.abs(cx)), np.max(np.abs(cy)))
        eq_parts_x = []
        eq_parts_y = []

        for i, name in enumerate(self._basis_names):
            if abs(cx[i]) > threshold:
                eq_parts_x.append(f"{cx[i]:+.3f}*{name}")
            if abs(cy[i]) > threshold:
                eq_parts_y.append(f"{cy[i]:+.3f}*{name}")

        equation = (f"Fx = {' '.join(eq_parts_x[:5]) if eq_parts_x else '0'}\n"
                    f"    Fy = {' '.join(eq_parts_y[:5]) if eq_parts_y else '0'}")

        if verbose:
            print(f"\n  ── Auto-Learn: Basis Function Fitting ──")
            print(f"  Samples: {len(X_list)}")
            print(f"  Basis functions: {len(self._basis_names)}")
            print(f"  R-squared: {r2:.4f}")
            print(f"  Equation: {equation}")

        if r2 < 0.5:
            if verbose:
                print(f"  R^2 too low ({r2:.3f}) — force model not reliable")
            return None

        # Create a force function from fitted coefficients
        basis_names = self._basis_names.copy()
        cx_copy = cx.copy()
        cy_copy = cy.copy()

        def fitted_force(state):
            phi = self._build_basis(state)
            return np.array([phi @ cx_copy, phi @ cy_copy])

        result = FittedForce(
            name=f"auto_force_{len(self._discoveries)}",
            coefficients=np.stack([cx, cy]),
            basis_names=basis_names,
            r_squared=r2,
            equation=equation,
        )
        self._discoveries.append(result)

        # Add to belief model
        learned = LearnedForce(
            name=result.name,
            description=f"Auto-discovered force (R²={r2:.3f}): {equation}",
            compute=fitted_force,
            confidence=r2,
        )
        belief.add_force(learned)

        if verbose:
            print(f"  DISCOVERED: '{result.name}' (R²={r2:.3f})")

        return result


# ──────────────────────────────────────────────────────────────────────────────
# 3. Oracle Learner — ask an LLM when stuck
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class OracleQuery:
    question: str
    context: dict
    response: Optional[str] = None


@dataclass
class OracleResponse:
    concept_name: str
    description: str
    force_type: str        # constant, linear, quadratic, inverse_square
    direction: str
    depends_on: List[str]
    equation_hint: str


class OracleLearner:
    """
    Query an external oracle (LLM or local KB) when the agent is stuck.

    Phase 1 (local): Pattern-match anomaly to extended knowledge base
    Phase 2 (API): Send natural language query to Claude API

    The oracle consultation follows a structured protocol:
      1. Format anomaly as natural language question
      2. Include context: what forces are already known,
         what's been tried, what the error pattern looks like
      3. Parse response into a testable force template
      4. Test the template against observations
      5. Accept or reject based on error reduction
    """

    def __init__(self, agent: DiscoveryAgent, use_api: bool = False):
        self.agent = agent
        self.use_api = use_api
        self._query_history: List[OracleQuery] = []

    def format_question(self, anomaly: AnomalyReport) -> str:
        """Format anomaly as a natural language question."""
        known = self.agent.belief.known_forces
        known_str = ', '.join(known) if known else 'none'

        question = (
            f"I observe an object with a systematic unexplained "
            f"{anomaly.direction} force of magnitude {anomaly.magnitude:.2f}N. "
            f"The force pattern is: {anomaly.pattern}. "
            f"Forces already accounted for: {known_str}. "
            f"What physical phenomenon could explain this? "
            f"Provide the force equation and what variables it depends on."
        )
        return question

    def consult_local(self, anomaly: AnomalyReport) -> Optional[OracleResponse]:
        """
        Phase 1: Local pattern-matching oracle.
        Uses extended rules beyond the basic knowledge base.
        """
        direction = anomaly.direction
        magnitude = anomaly.magnitude
        pattern = anomaly.pattern
        known = set(self.agent.belief.known_forces)

        # Rule-based extended oracle
        responses = []

        # Constant downward + not gravity → might be gravity with buoyancy
        if ('downward' in direction and 'gravity' not in known
                and magnitude > 5.0):
            responses.append(OracleResponse(
                concept_name='gravity',
                description='Gravitational acceleration',
                force_type='constant',
                direction='downward',
                depends_on=['mass'],
                equation_hint='F = m * g, try g from -5 to -12',
            ))

        # Upward force → buoyancy or lift
        if 'upward' in direction and 'buoyancy' not in known:
            responses.append(OracleResponse(
                concept_name='buoyancy',
                description='Fluid buoyancy force, depends on depth',
                force_type='linear_position',
                direction='upward',
                depends_on=['position_y', 'depth'],
                equation_hint='F = rho * depth * g',
            ))

        # Opposing motion → friction or drag
        if ('opposing' in pattern or 'slowing' in pattern or
                'leftward' in direction):
            if 'friction' not in known:
                responses.append(OracleResponse(
                    concept_name='friction',
                    description='Kinetic friction opposing motion',
                    force_type='velocity_dependent',
                    direction='opposing_motion',
                    depends_on=['velocity'],
                    equation_hint='F = -mu * m * g * v_hat',
                ))
            if 'aerodynamic drag' not in known:
                responses.append(OracleResponse(
                    concept_name='drag',
                    description='Aerodynamic drag proportional to v^2',
                    force_type='quadratic_velocity',
                    direction='opposing_motion',
                    depends_on=['velocity_squared'],
                    equation_hint='F = -Cd * |v| * v',
                ))

        # Toward a point → attraction
        if ('degrees' in direction or 'toward' in pattern):
            if 'magnetic attraction' not in known:
                responses.append(OracleResponse(
                    concept_name='attraction',
                    description='Attractive force toward a center point',
                    force_type='inverse_square',
                    direction='toward_center',
                    depends_on=['position', 'distance'],
                    equation_hint='F = k * (center - pos) / |center - pos|^2',
                ))

        # Position-dependent → spring or field
        if 'position-dependent' in pattern and 'spring/elastic' not in known:
            responses.append(OracleResponse(
                concept_name='restoring_force',
                description='Restoring force proportional to displacement',
                force_type='linear_position',
                direction='toward_equilibrium',
                depends_on=['position'],
                equation_hint='F = -k * (pos - anchor)',
            ))

        return responses[0] if responses else None

    def consult_api(self, anomaly: AnomalyReport) -> Optional[OracleResponse]:
        """
        Phase 2: Query Claude API for open-ended discovery.

        Requires:
            pip install anthropic

        The LLM response is parsed into an OracleResponse with
        structured fields the agent can test.
        """
        question = self.format_question(anomaly)

        try:
            import anthropic
            client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var

            system_prompt = (
                "You are a physics expert helping an autonomous agent "
                "discover physical laws. The agent has detected an "
                "unexplained force and needs your help identifying it. "
                "Respond with EXACTLY this format:\n"
                "CONCEPT: <name>\n"
                "DESCRIPTION: <one sentence>\n"
                "FORCE_TYPE: constant|linear|quadratic|inverse_square\n"
                "DIRECTION: <direction>\n"
                "DEPENDS_ON: <comma-separated variables>\n"
                "EQUATION: <force equation>"
            )

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                system=system_prompt,
                messages=[{"role": "user", "content": question}],
            )

            text = response.content[0].text

            # Parse structured response
            lines = text.strip().split('\n')
            parsed = {}
            for line in lines:
                if ':' in line:
                    key, val = line.split(':', 1)
                    parsed[key.strip().upper()] = val.strip()

            return OracleResponse(
                concept_name=parsed.get('CONCEPT', 'unknown'),
                description=parsed.get('DESCRIPTION', ''),
                force_type=parsed.get('FORCE_TYPE', 'unknown'),
                direction=parsed.get('DIRECTION', ''),
                depends_on=parsed.get('DEPENDS_ON', '').split(','),
                equation_hint=parsed.get('EQUATION', ''),
            )

        except ImportError:
            print("    [oracle] anthropic package not installed, "
                  "using local oracle")
            return self.consult_local(anomaly)
        except Exception as e:
            print(f"    [oracle] API error: {e}, falling back to local")
            return self.consult_local(anomaly)

    def consult(self, anomaly: AnomalyReport,
                verbose: bool = True) -> Optional[OracleResponse]:
        """
        Consult the oracle (local or API based on configuration).
        """
        question = self.format_question(anomaly)

        query = OracleQuery(
            question=question,
            context={
                'known_forces': self.agent.belief.known_forces,
                'direction': anomaly.direction,
                'magnitude': anomaly.magnitude,
                'pattern': anomaly.pattern,
            },
        )
        self._query_history.append(query)

        if verbose:
            print(f"    [oracle] Question: {question[:100]}...")

        if self.use_api:
            response = self.consult_api(anomaly)
        else:
            response = self.consult_local(anomaly)

        if response and verbose:
            print(f"    [oracle] Answer: {response.concept_name} — "
                  f"{response.description}")
            print(f"    [oracle] Equation hint: {response.equation_hint}")

        return response


# ──────────────────────────────────────────────────────────────────────────────
# Integration: Enhanced Discovery Agent
# ──────────────────────────────────────────────────────────────────────────────

class EnhancedDiscoveryAgent(DiscoveryAgent):
    """
    Extended agent with all three learning modes.

    Discovery cascade:
      1. Try knowledge base (existing behavior)
      2. If KB fails → consult oracle
      3. If oracle fails → auto-learn (basis fitting)
      4. If auto-learn fails → design experiments (instruction mode)
    """

    def __init__(self, sim: SimplePhysicsSim, dt: float = 0.1,
                 use_api: bool = False):
        super().__init__(dt=dt)
        self.sim = sim
        self.auto_learner = AutoLearner(max_degree=2)
        self.oracle = OracleLearner(self, use_api=use_api)
        self.instruction_learner = InstructionLearner(self, sim)

    def discover_enhanced(self, verbose: bool = True) -> Optional[Discovery]:
        """
        Enhanced discovery with cascade:
          KB → Oracle → Auto-Learn
        """
        # Step 1: Try standard KB discovery
        result = self.discover(verbose=verbose)
        if result is not None:
            return result

        # Step 2: If KB failed but anomaly exists, consult oracle
        report = self.introspect()
        if report is None or report.anomaly_type != AnomalyType.MISSING_LAW:
            return None

        if verbose:
            print(f"    [cascade] KB failed, consulting oracle...")

        oracle_response = self.oracle.consult(report, verbose=verbose)
        if oracle_response:
            # TODO: convert oracle response to testable force and test it
            if verbose:
                print(f"    [cascade] Oracle suggests: "
                      f"{oracle_response.concept_name}")

        # Step 3: If oracle didn't resolve, try auto-learn
        if verbose:
            print(f"    [cascade] Trying auto-learn (basis fitting)...")

        recent_obs = self._observation_buffer[-50:]
        fitted = self.auto_learner.fit_unknown_force(
            recent_obs, self.belief, verbose=verbose)

        if fitted:
            discovery = Discovery(
                name=fitted.name,
                description=f"Auto-discovered: {fitted.equation}",
                confidence=fitted.r_squared,
                error_reduction=fitted.r_squared * 100,
                discovery_step=self._step_count,
                observations_used=len(recent_obs),
            )
            self.discoveries.append(discovery)
            return discovery

        return None


# ──────────────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────────────

def demo():
    print("=" * 65)
    print("  Adaptive Learner Demo")
    print("=" * 65)

    dt = 0.05

    # ── Demo 1: Instruction-driven ──
    print("\n── Demo 1: Instruction-Driven Learning ──")
    sim = SimplePhysicsSim(dt=dt)
    sim.add_force("gravity", ForceType.GRAVITY, g=-9.81)
    agent = DiscoveryAgent(dt=dt)
    learner = InstructionLearner(agent, sim)

    findings = learner.investigate(
        "investigate what happens at different heights",
        n_steps=20, verbose=True)

    # ── Demo 2: Auto-learn on unknown force ──
    print("\n── Demo 2: Auto-Learn (Basis Fitting) ──")
    sim2 = SimplePhysicsSim(dt=dt)
    sim2.add_force("mystery", ForceType.SPRING, k=2.0, anchor=[3.0, 3.0])

    agent2 = DiscoveryAgent(dt=dt)
    auto = AutoLearner(max_degree=2)

    # Observe
    s0 = PhysicsState(np.array([1.0, 1.0]), np.array([0.5, 0.3]))
    traj = sim2.rollout(s0, 60)
    for s in traj:
        agent2.observe(s)

    # Auto-learn
    result = auto.fit_unknown_force(traj, agent2.belief, verbose=True)

    # ── Demo 3: Oracle consultation ──
    print("\n── Demo 3: Oracle Consultation (Local) ──")
    sim3 = SimplePhysicsSim(dt=dt)
    sim3.add_force("buoyancy", ForceType.BUOYANCY, rho=1.5, surface_y=5.0)

    agent3 = DiscoveryAgent(dt=dt)
    oracle = OracleLearner(agent3, use_api=False)

    s0 = PhysicsState(np.array([1.0, 2.0]), np.array([0.5, 0.0]))
    traj3 = sim3.rollout(s0, 40)
    for s in traj3:
        agent3.observe(s)

    report = agent3.introspect()
    if report and report.anomaly_type == AnomalyType.MISSING_LAW:
        response = oracle.consult(report, verbose=True)

    # ── Demo 4: Enhanced cascade ──
    print("\n── Demo 4: Enhanced Discovery Cascade ──")
    sim4 = SimplePhysicsSim(dt=dt)
    sim4.add_force("gravity", ForceType.GRAVITY, g=-9.81)
    sim4.add_force("buoyancy", ForceType.BUOYANCY, rho=1.0, surface_y=5.0)

    enhanced = EnhancedDiscoveryAgent(sim4, dt=dt)
    s0 = PhysicsState(np.array([1.0, 3.0]), np.array([1.0, 0.0]))
    traj4 = sim4.rollout(s0, 80)

    for i, s in enumerate(traj4):
        enhanced.observe(s)
        if (i + 1) % 20 == 0 and i > 10:
            print(f"\n  Step {i+1}/{len(traj4)}")
            enhanced.discover_enhanced(verbose=True)

    print(f"\n  Final discoveries: {enhanced.belief.known_forces}")
    print(f"  Auto-discoveries: {len(enhanced.auto_learner._discoveries)}")

    print(f"\n{'='*65}")


if __name__ == "__main__":
    demo()
