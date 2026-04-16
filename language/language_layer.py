"""
language_layer.py — Self-Generated Language Without LLM
=========================================================
The system explains itself using its own signals.
No external model. No API. No tokens. Just templates
grounded in the actual neuromodulatory state.

Components:
  1. SelfNarrator — 17 questions answered in natural language
  2. PhysicsExplainer — auto-learned equations → human descriptions
  3. ConsolidationReporter — what changed after sleep
  4. AnomalyExplainer — structured natural language anomaly reports
  5. DiscoveryJournal — running log of all discoveries with context

Paper claim: "The first world model that explains its own reasoning
in natural language at every step — without an LLM."

Usage:
    from language_layer import SelfNarrator, PhysicsExplainer

    narrator = SelfNarrator()
    text = narrator.narrate(signals)
    print(text)  # "Heading north at 0.8 m/s. Planning 6 steps ahead..."

Author: John Taylor
"""

import math
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# 1. Self-Narrator — 17 questions in natural language
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class NarrationSignals:
    """All signals needed for narration."""
    # Position & motion
    pos: np.ndarray = field(default_factory=lambda: np.zeros(2))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    heading: float = 0.0
    speed: float = 0.0

    # Neuromodulators
    da: float = 0.0
    ach: float = 0.5
    crt: float = 0.0
    ne: float = 0.3
    sht: float = 0.5

    # Cognitive state
    gate_alpha: float = 0.5
    k_eff: int = 8
    horizon_steps: int = 8
    n_retrieved: int = 0
    dist_to_goal: float = 1.0
    explore_score: float = 0.5
    adenosine: float = 0.0

    # Memory
    ep_count: int = 0
    schema_steps: int = 0
    novelty: float = 0.5

    # Step info
    step: int = 0
    episode: int = 0


class SelfNarrator:
    """
    Generates natural language narration from neuromodulatory signals.

    Each of the 17 introspective questions produces a sentence fragment.
    The narrator selects the most relevant 2-4 based on current state
    and composes a coherent thought.

    No LLM. No external model. Templates grounded in real signals.
    """

    def _heading_text(self, heading: float) -> str:
        """Convert heading radians to cardinal direction."""
        deg = math.degrees(heading) % 360
        dirs = ['east', 'northeast', 'north', 'northwest',
                'west', 'southwest', 'south', 'southeast']
        idx = int((deg + 22.5) / 45) % 8
        return dirs[idx]

    def _speed_text(self, speed: float) -> str:
        if speed < 0.1:
            return "stationary"
        elif speed < 0.5:
            return "moving slowly"
        elif speed < 1.5:
            return "at moderate speed"
        elif speed < 3.0:
            return "moving quickly"
        else:
            return "at high speed"

    def _da_text(self, da: float) -> str:
        if da > 0.8:
            return "highly surprising"
        elif da > 0.5:
            return "somewhat novel"
        elif da > 0.2:
            return "mostly familiar"
        else:
            return "completely routine"

    def _crt_text(self, crt: float) -> str:
        if crt > 0.7:
            return "under high stress"
        elif crt > 0.4:
            return "moderately stressed"
        elif crt > 0.1:
            return "slightly tense"
        else:
            return "calm"

    def _gate_text(self, alpha: float) -> str:
        if alpha > 0.7:
            return "trusting my predictions"
        elif alpha > 0.4:
            return "partially trusting predictions"
        else:
            return "relying on direct observation"

    def _explore_text(self, score: float) -> str:
        if score > 0.7:
            return "exploring new territory"
        elif score > 0.3:
            return "balancing exploration and exploitation"
        else:
            return "exploiting known strategies"

    def q1_where(self, s: NarrationSignals) -> str:
        """Q1: Where am I?"""
        direction = self._heading_text(s.heading)
        speed = self._speed_text(s.speed)
        return f"Heading {direction} {speed} at ({s.pos[0]:.1f}, {s.pos[1]:.1f})"

    def q3_anomaly(self, s: NarrationSignals) -> Optional[str]:
        """Q3: Is something wrong?"""
        if s.crt > 0.4:
            return f"Something feels off. Cortisol elevated at {s.crt:.2f}"
        return None

    def q5_predict(self, s: NarrationSignals) -> str:
        """Q5: What happens if I continue?"""
        if s.speed > 0.1:
            dx = s.vel[0] * 0.5
            dy = s.vel[1] * 0.5
            return f"If I continue, I'll be at ({s.pos[0]+dx:.1f}, {s.pos[1]+dy:.1f}) in 0.5s"
        return "Standing still, nothing will change"

    def q7_horizon(self, s: NarrationSignals) -> str:
        """Q7: How far ahead should I plan?"""
        seconds = s.horizon_steps * 0.25
        if s.ach > 0.7:
            confidence = "High confidence"
        elif s.ach > 0.4:
            confidence = "Moderate confidence"
        else:
            confidence = "Low confidence"
        return f"{confidence}, planning {s.horizon_steps} steps ({seconds:.1f}s) ahead"

    def q8_trust(self, s: NarrationSignals) -> str:
        """Q8: Should I trust my prediction?"""
        return self._gate_text(s.gate_alpha)

    def q9_action(self, s: NarrationSignals) -> str:
        """Q9: What's the best action?"""
        if s.dist_to_goal < 0.3:
            return "Almost at the goal, fine-tuning approach"
        elif s.dist_to_goal < 1.0:
            return "Goal is close, heading straight for it"
        else:
            return f"Goal is {s.dist_to_goal:.1f} units away, navigating toward it"

    def q10_memory(self, s: NarrationSignals) -> Optional[str]:
        """Q10: Have I been here before?"""
        if s.n_retrieved > 0:
            return f"Found {s.n_retrieved} similar past episodes in memory"
        elif s.ep_count > 0:
            return "No similar episodes found, this is new territory"
        return None

    def q11_novelty(self, s: NarrationSignals) -> Optional[str]:
        """Q11: Is this a new kind of place?"""
        if s.novelty > 1.5:
            return f"Highly novel situation (novelty={s.novelty:.2f})"
        elif s.novelty > 0.8:
            return None  # not interesting enough to mention
        else:
            return "Familiar territory, schemas cover this well"

    def q12_language(self, s: NarrationSignals) -> Optional[str]:
        """Q12: Does language help here?"""
        if s.ach > 0.7 and s.speed > 0.5:
            return "Heading signal strong, skipping visual processing"
        return None

    def q13_capacity(self, s: NarrationSignals) -> Optional[str]:
        """Q13: How much working memory available?"""
        if s.k_eff < 6:
            return f"Working memory reduced to {s.k_eff}/8 under stress"
        return None

    def q14_dream(self, s: NarrationSignals) -> Optional[str]:
        """Q14: What would I dream about?"""
        if s.speed < 0.05 and s.adenosine > 0.5:
            return "Idle and fatigued, ready to consolidate memories"
        return None

    def q16_explore(self, s: NarrationSignals) -> Optional[str]:
        """Q16: Should I explore or exploit?"""
        if s.explore_score > 0.7:
            return "High novelty, switching to exploration mode"
        elif s.explore_score < 0.3:
            return "Familiar ground, exploiting known paths"
        return None

    def q17_fatigue(self, s: NarrationSignals) -> Optional[str]:
        """Q17: Am I getting fatigued?"""
        if s.adenosine > 0.6:
            return f"Fatigue building (adenosine={s.adenosine:.2f}), need rest soon"
        return None

    def narrate(self, s: NarrationSignals) -> str:
        """
        Generate a natural language narration of the current state.
        Selects the most relevant 2-4 sentences from 17 questions.
        """
        sentences = []

        # Always include location
        sentences.append(self.q1_where(s))

        # Priority: anomalies and stress
        a = self.q3_anomaly(s)
        if a:
            sentences.append(a)

        # Cognitive state (pick most relevant)
        if s.dist_to_goal < 0.5:
            sentences.append(self.q9_action(s))
        elif s.n_retrieved > 0:
            m = self.q10_memory(s)
            if m:
                sentences.append(m)

        # Planning
        sentences.append(self.q7_horizon(s))

        # Conditional additions
        for q in [self.q11_novelty, self.q12_language, self.q13_capacity,
                   self.q16_explore, self.q17_fatigue, self.q14_dream]:
            result = q(s)
            if result:
                sentences.append(result)

        # Cap at 4 sentences for readability
        return ". ".join(sentences[:4]) + "."

    def narrate_full(self, s: NarrationSignals) -> dict:
        """
        Generate all 17 answers as a structured dict.
        Useful for logging and debugging.
        """
        return {
            'Q1_where': self.q1_where(s),
            'Q3_anomaly': self.q3_anomaly(s),
            'Q5_predict': self.q5_predict(s),
            'Q7_horizon': self.q7_horizon(s),
            'Q8_trust': self.q8_trust(s),
            'Q9_action': self.q9_action(s),
            'Q10_memory': self.q10_memory(s),
            'Q11_novelty': self.q11_novelty(s),
            'Q12_language': self.q12_language(s),
            'Q13_capacity': self.q13_capacity(s),
            'Q14_dream': self.q14_dream(s),
            'Q16_explore': self.q16_explore(s),
            'Q17_fatigue': self.q17_fatigue(s),
            'narration': self.narrate(s),
        }


# ──────────────────────────────────────────────────────────────────────────────
# 2. Physics Explainer — equations → human descriptions
# ──────────────────────────────────────────────────────────────────────────────

class PhysicsExplainer:
    """
    Converts auto-learned physics equations into natural language.
    No LLM — pattern matches on basis function names and coefficients.

    Examples:
      Fy = -9.81*const → "Constant downward force of 9.81 N (gravity)"
      Fy = +11.2 - 9.81*y → "Downward force that weakens with height,
                               reverses to upward below y=1.14 (buoyancy)"
      Fx = -0.3*vx → "Force opposing horizontal motion (friction)"
    """

    def explain_coefficients(self, coefficients: np.ndarray,
                              basis_names: List[str]) -> str:
        """Explain a fitted force from coefficients and basis names."""
        cx, cy = coefficients[0], coefficients[1]
        explanations = []

        # Analyze each component
        for axis, coeffs, axis_name in [(0, cx, 'horizontal'),
                                         (1, cy, 'vertical')]:
            significant = []
            for i, (c, name) in enumerate(zip(coeffs, basis_names)):
                if abs(c) > 0.05:
                    significant.append((c, name))

            if not significant:
                continue

            # Sort by magnitude
            significant.sort(key=lambda x: -abs(x[0]))

            for coeff, name in significant[:3]:
                explanation = self._explain_term(coeff, name, axis_name)
                if explanation:
                    explanations.append(explanation)

        if not explanations:
            return "No significant forces detected."

        # Synthesize
        summary = self._synthesize(explanations, cx, cy, basis_names)
        return summary

    def _explain_term(self, coeff: float, basis_name: str,
                       axis: str) -> Optional[str]:
        """Explain a single term in the force equation."""
        direction = "positive" if coeff > 0 else "negative"
        mag = abs(coeff)

        if basis_name == 'const':
            if axis == 'vertical':
                if coeff < -5:
                    return f"constant downward force of {mag:.1f}N (like gravity)"
                elif coeff > 5:
                    return f"constant upward force of {mag:.1f}N (like buoyancy or lift)"
                elif abs(coeff) > 0.5:
                    d = "upward" if coeff > 0 else "downward"
                    return f"constant {d} force of {mag:.1f}N"
            else:
                if abs(coeff) > 0.5:
                    d = "rightward" if coeff > 0 else "leftward"
                    return f"constant {d} force of {mag:.1f}N"

        elif basis_name in ('x', 'y'):
            coord = basis_name
            if abs(coeff) > 0.5:
                if coeff < 0:
                    return f"restoring force proportional to {coord} position (like a spring, k={mag:.2f})"
                else:
                    return f"force that increases with {coord} position (magnitude {mag:.2f} per unit)"

        elif basis_name in ('vx', 'vy'):
            comp = 'horizontal' if basis_name == 'vx' else 'vertical'
            if coeff < -0.1:
                return f"force opposing {comp} velocity (like friction/drag, coefficient {mag:.2f})"
            elif coeff > 0.1:
                return f"force amplifying {comp} velocity (acceleration, coefficient {mag:.2f})"

        elif '1/r' in basis_name:
            if abs(coeff) > 0.1:
                source = "origin" if "origin" in basis_name else "center (5,5)"
                d = "toward" if coeff > 0 else "away from"
                return f"inverse-distance force {d} {source} (like magnetic/gravitational attraction)"

        elif basis_name in ('|v|', '|v|^2'):
            if coeff < -0.1:
                kind = "linear" if basis_name == '|v|' else "quadratic"
                return f"{kind} drag opposing motion (coefficient {mag:.2f})"

        elif basis_name in ('x^2', 'y^2'):
            if abs(coeff) > 0.05:
                coord = basis_name[0]
                return f"nonlinear {coord}-dependent force (quadratic, coefficient {mag:.3f})"

        return None

    def _synthesize(self, explanations: List[str],
                     cx: np.ndarray, cy: np.ndarray,
                     basis_names: List[str]) -> str:
        """Synthesize individual explanations into a coherent description."""
        if len(explanations) == 1:
            return f"Discovered force: {explanations[0]}."

        # Check for equilibrium point
        const_y = cy[basis_names.index('const')] if 'const' in basis_names else 0
        y_coeff = cy[basis_names.index('y')] if 'y' in basis_names else 0

        if abs(y_coeff) > 0.1 and abs(const_y) > 0.1:
            # There's an equilibrium where const + y_coeff * y_eq = 0
            y_eq = -const_y / y_coeff
            if 0 < y_eq < 20:
                explanations.append(
                    f"Forces balance at y={y_eq:.2f} (equilibrium point)")

        result = "Discovered forces:\n"
        for i, exp in enumerate(explanations):
            result += f"  {i+1}. {exp}\n"
        return result.strip()

    def explain_discovery(self, name: str, equation: str,
                           r_squared: float) -> str:
        """Generate a plain English discovery announcement."""
        return (f"Discovery: '{name}'\n"
                f"  Equation: {equation}\n"
                f"  Fit quality: R²={r_squared:.4f}\n"
                f"  Confidence: {'high' if r_squared > 0.95 else 'moderate' if r_squared > 0.8 else 'low'}")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Consolidation Reporter — what changed after sleep
# ──────────────────────────────────────────────────────────────────────────────

class ConsolidationReporter:
    """
    Generates a verbal report after memory consolidation.
    Summarizes what was learned, what was compressed, and
    what questions remain.
    """

    def report(self, before: dict, after: dict,
                consolidation_result: dict) -> str:
        """
        Generate consolidation report from before/after stats.

        Args:
            before: buf.stats() before consolidation
            after: buf.stats() after consolidation
            consolidation_result: from SleepConsolidation.consolidate()
        """
        lines = ["Consolidation report:"]

        # Episodes
        ep_before = before.get('ep_count', 0)
        ep_after = after.get('ep_count', 0)
        ep_diff = ep_before - ep_after
        if ep_diff > 0:
            lines.append(f"  Pruned {ep_diff} redundant episodes "
                         f"({ep_before} -> {ep_after})")
        else:
            lines.append(f"  Episodes: {ep_after} (no pruning needed)")

        # Schemas
        schema_before = before.get('schema_steps', 0)
        schema_after = after.get('schema_steps', 0)
        if schema_after > schema_before:
            lines.append(f"  Schemas grew: {schema_before} -> {schema_after} "
                         f"(+{schema_after - schema_before} prototypes)")

        # Replayed
        replayed = consolidation_result.get('episodes_replayed', 0)
        if replayed > 0:
            lines.append(f"  Replayed {replayed} high-priority episodes")

        # Priority
        prio_before = before.get('ep_mean_priority', 0)
        prio_after = after.get('ep_mean_priority', 0)
        if prio_after > prio_before:
            lines.append(f"  Mean priority increased: {prio_before:.3f} -> "
                         f"{prio_after:.3f} (low-value episodes removed)")

        # Duration
        duration = consolidation_result.get('duration_ms', 0)
        lines.append(f"  Duration: {duration:.1f}ms")

        # Assessment
        if ep_diff > 5:
            lines.append("  Assessment: significant compression achieved")
        elif schema_after > schema_before:
            lines.append("  Assessment: new schemas formed from experience")
        else:
            lines.append("  Assessment: memory stable, no major changes")

        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Anomaly Explainer — structured NL anomaly reports
# ──────────────────────────────────────────────────────────────────────────────

class AnomalyExplainer:
    """
    Converts AnomalyReport into natural language explanation.
    No LLM — template-based with cause reasoning.
    """

    def explain(self, direction: str, magnitude: float,
                pattern: str, is_systematic: bool,
                known_forces: List[str]) -> str:
        """Generate natural language anomaly explanation."""
        lines = []

        # What happened
        if is_systematic:
            lines.append(f"Systematic anomaly detected: {pattern}")
        else:
            lines.append(f"Irregular anomaly: {pattern}")

        # Force description
        if magnitude > 5:
            strength = "strong"
        elif magnitude > 1:
            strength = "moderate"
        else:
            strength = "weak"
        lines.append(f"A {strength} {direction} force "
                     f"of approximately {magnitude:.1f}N is unexplained")

        # Possible causes
        causes = self._suggest_causes(direction, magnitude,
                                        pattern, known_forces)
        if causes:
            lines.append("Possible causes:")
            for cause in causes:
                lines.append(f"  - {cause}")

        # Recommendation
        if is_systematic and magnitude > 0.5:
            lines.append("Recommendation: investigate this force systematically")
        elif not is_systematic:
            lines.append("Recommendation: may be noise, continue observing")

        return "\n".join(lines)

    def _suggest_causes(self, direction: str, magnitude: float,
                         pattern: str, known: List[str]) -> List[str]:
        """Suggest possible physical causes."""
        causes = []

        if 'downward' in direction and 'gravity' not in known:
            causes.append("gravitational acceleration (F = mg)")
        if 'upward' in direction and 'buoyancy' not in known:
            causes.append("buoyancy in fluid (F = rho * V * g)")
        if 'opposing' in pattern and 'friction' not in known:
            causes.append("friction or drag (opposes motion)")
        if 'position-dependent' in pattern:
            causes.append("spring-like restoring force (F = -kx)")
            causes.append("magnetic or electric field (F ~ 1/r²)")
        if 'degrees' in direction and 'magnetic' not in known:
            causes.append("attraction toward a nearby object or field")
        if 'constant' in pattern and magnitude > 8:
            causes.append("gravitational field")
        if 'constant' in pattern and magnitude < 3:
            causes.append("wind or current")

        if not causes:
            causes.append("unknown force — consider auto-learning")

        return causes[:4]


# ──────────────────────────────────────────────────────────────────────────────
# 5. Discovery Journal — running log
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class JournalEntry:
    timestamp: float
    step: int
    event_type: str    # discovery, anomaly, consolidation, experiment
    title: str
    details: str


class DiscoveryJournal:
    """
    Running log of all discoveries, anomalies, and consolidations
    in natural language. The agent's lab notebook.
    """

    def __init__(self):
        self.entries: List[JournalEntry] = []

    def log_discovery(self, step: int, name: str,
                       equation: str, r_squared: float,
                       method: str):
        self.entries.append(JournalEntry(
            timestamp=time.time(),
            step=step,
            event_type='discovery',
            title=f"Discovered: {name}",
            details=(f"Method: {method}\n"
                     f"Equation: {equation}\n"
                     f"Confidence: R²={r_squared:.4f}"),
        ))

    def log_anomaly(self, step: int, direction: str,
                     magnitude: float, resolved: bool):
        status = "resolved" if resolved else "unresolved"
        self.entries.append(JournalEntry(
            timestamp=time.time(),
            step=step,
            event_type='anomaly',
            title=f"Anomaly ({status}): {direction} force {magnitude:.2f}N",
            details=f"Systematic {direction} force detected at step {step}",
        ))

    def log_consolidation(self, step: int, pruned: int,
                            schemas_added: int):
        self.entries.append(JournalEntry(
            timestamp=time.time(),
            step=step,
            event_type='consolidation',
            title=f"Memory consolidation",
            details=(f"Pruned {pruned} episodes, "
                     f"added {schemas_added} schemas"),
        ))

    def log_experiment(self, step: int, instruction: str,
                        finding: str):
        self.entries.append(JournalEntry(
            timestamp=time.time(),
            step=step,
            event_type='experiment',
            title=f"Experiment: {instruction}",
            details=finding,
        ))

    def summary(self, last_n: int = 10) -> str:
        """Generate a natural language summary of recent events."""
        recent = self.entries[-last_n:]
        if not recent:
            return "No events recorded yet."

        lines = [f"Journal summary ({len(recent)} recent entries):"]
        for e in recent:
            lines.append(f"  [{e.event_type}] Step {e.step}: {e.title}")
        return "\n".join(lines)

    def discoveries_summary(self) -> str:
        """Summarize only discoveries."""
        discoveries = [e for e in self.entries
                       if e.event_type == 'discovery']
        if not discoveries:
            return "No discoveries yet."

        lines = [f"Discoveries ({len(discoveries)} total):"]
        for d in discoveries:
            lines.append(f"  - {d.title}")
            lines.append(f"    {d.details}")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────────────

def demo():
    print("=" * 65)
    print("  Language Layer Demo — No LLM Required")
    print("=" * 65)

    narrator = SelfNarrator()
    explainer = PhysicsExplainer()
    anomaly_exp = AnomalyExplainer()
    reporter = ConsolidationReporter()
    journal = DiscoveryJournal()

    # ── Demo 1: Self-Narration at different states ──
    print("\n── Self-Narration ──")

    states = [
        ("Exploring new territory", NarrationSignals(
            pos=np.array([3.2, 7.1]), vel=np.array([0.8, 0.3]),
            heading=0.36, speed=0.85, da=0.7, ach=0.8, crt=0.1,
            gate_alpha=0.85, k_eff=8, horizon_steps=7,
            n_retrieved=0, dist_to_goal=4.2, explore_score=0.8,
            novelty=1.8, ep_count=50, step=14,
        )),
        ("Under stress, near goal", NarrationSignals(
            pos=np.array([5.1, 4.8]), vel=np.array([0.3, -0.1]),
            heading=-0.32, speed=0.32, da=0.2, ach=0.4, crt=0.6,
            gate_alpha=0.35, k_eff=4, horizon_steps=3,
            n_retrieved=3, dist_to_goal=0.4, explore_score=0.15,
            novelty=0.3, ep_count=200, step=87,
        )),
        ("Fatigued, idle", NarrationSignals(
            pos=np.array([2.0, 2.0]), vel=np.array([0.01, 0.0]),
            heading=0.0, speed=0.01, da=0.05, ach=0.2, crt=0.2,
            gate_alpha=0.6, k_eff=7, horizon_steps=2,
            n_retrieved=0, dist_to_goal=5.0, adenosine=0.75,
            novelty=0.4, ep_count=500, schema_steps=1200, step=300,
        )),
    ]

    for label, signals in states:
        print(f"\n  Scenario: {label}")
        narration = narrator.narrate(signals)
        print(f"  >> {narration}")

    # Full narration
    print(f"\n  Full Q&A for last state:")
    full = narrator.narrate_full(states[-1][1])
    for key, val in full.items():
        if val and key != 'narration':
            print(f"    {key}: {val}")

    # ── Demo 2: Physics Explanation ──
    print("\n── Physics Explanation ──")

    # Gravity
    coeffs_gravity = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-9.81, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])
    names = ['const', 'x', 'y', 'vx', 'vy', 'x^2', 'y^2',
             'vx^2', 'vy^2', 'x*y', 'vx*vy',
             '1/r_origin', '1/r_center', '1/r_origin^2', '1/r_center^2',
             '|v|', '|v|^2']

    print("\n  Equation: Fy = -9.81")
    explanation = explainer.explain_coefficients(coeffs_gravity, names)
    print(f"  {explanation}")

    # Gravity + buoyancy
    coeffs_gb = np.zeros((2, 17))
    coeffs_gb[1, 0] = 11.2   # const
    coeffs_gb[1, 2] = -9.81  # y
    print("\n  Equation: Fy = 11.2 - 9.81*y")
    explanation = explainer.explain_coefficients(coeffs_gb, names)
    print(f"  {explanation}")

    # Friction
    coeffs_fric = np.zeros((2, 17))
    coeffs_fric[0, 3] = -2.94  # vx
    print("\n  Equation: Fx = -2.94*vx")
    explanation = explainer.explain_coefficients(coeffs_fric, names)
    print(f"  {explanation}")

    # Spring
    coeffs_spring = np.zeros((2, 17))
    coeffs_spring[0, 1] = -2.0  # x
    coeffs_spring[1, 2] = -2.0  # y
    print("\n  Equation: Fx = -2.0*x, Fy = -2.0*y")
    explanation = explainer.explain_coefficients(coeffs_spring, names)
    print(f"  {explanation}")

    # ── Demo 3: Anomaly Explanation ──
    print("\n── Anomaly Explanation ──")

    explanation = anomaly_exp.explain(
        direction="downward",
        magnitude=9.81,
        pattern="constant downward force ~9.81N",
        is_systematic=True,
        known_forces=[],
    )
    print(f"  {explanation}")

    explanation2 = anomaly_exp.explain(
        direction="at 45 degrees",
        magnitude=2.4,
        pattern="position-dependent at 45 degrees force",
        is_systematic=True,
        known_forces=['gravity'],
    )
    print(f"\n  {explanation2}")

    # ── Demo 4: Consolidation Report ──
    print("\n── Consolidation Report ──")

    before = {'ep_count': 1688, 'schema_steps': 1800,
              'ep_mean_priority': 0.42}
    after = {'ep_count': 1450, 'schema_steps': 2027,
             'ep_mean_priority': 0.55}
    consol = {'episodes_replayed': 32, 'episodes_pruned': 238,
              'duration_ms': 1.14}

    report = reporter.report(before, after, consol)
    print(f"  {report}")

    # ── Demo 5: Discovery Journal ──
    print("\n── Discovery Journal ──")

    journal.log_anomaly(19, "downward", 9.81, True)
    journal.log_discovery(19, "gravity", "Fy = -9.81", 1.0, "knowledge_base")
    journal.log_anomaly(19, "leftward", 2.94, True)
    journal.log_discovery(19, "friction", "Fx = -mu*m*g*v_hat", 1.0,
                          "knowledge_base")
    journal.log_experiment(50, "investigate heights",
                           "Force is constant at all heights → not position-dependent")
    journal.log_consolidation(100, 238, 227)
    journal.log_discovery(120, "auto_force_0", "Fy = 11.2 - 9.81*y", 1.0,
                          "auto_learn")

    print(f"  {journal.summary()}")
    print()
    print(f"  {journal.discoveries_summary()}")

    print(f"\n{'='*65}")
    print(f"  All language generated without any LLM")
    print(f"  Templates grounded in actual neuromodulatory signals")
    print(f"{'='*65}")


if __name__ == "__main__":
    demo()
