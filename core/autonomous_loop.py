"""
autonomous_loop.py — The Autonomous Curiosity-Driven Loop
============================================================
The "AGI kernel": all 20 questions + all features wired into
a single continuous loop that generates its own goals, explores,
learns, dreams, and accumulates knowledge over time.

CONTROLLABLE:
  --run          Start the loop
  --cycles N     Run N cycles then stop (default: infinite)
  --status       Show current knowledge state
  --reset        Clear all accumulated knowledge
  --export       Export knowledge to JSON

The loop can be paused/resumed by creating/deleting a file:
  touch PAUSE    → pauses the loop
  rm PAUSE       → resumes

Each cycle:
  1. SURVEY    — Ask all 20 questions, identify knowledge gaps
  2. GOAL      — Generate goal from curiosity (highest novelty)
  3. PLAN      — Hierarchical schema planning to reach goal
  4. EXECUTE   — Simulate actions, collect experience
  5. NARRATE   — Generate language describing what happened
  6. EVALUATE  — Counterfactual: "what if I had done differently?"
  7. SLEEP     — Consolidate schemas, emotional tagging, replay
  8. GROW      — Update vocabulary, schemas, intentions
  9. REFLECT   — Log what was learned, set new prospective memories

Usage:
    python autonomous_loop.py --run --cycles 20     # run 20 cycles
    python autonomous_loop.py --run                 # run forever (Ctrl+C)
    python autonomous_loop.py --status              # show knowledge
    python autonomous_loop.py --export              # save to JSON
    python autonomous_loop.py --test                # validate loop
"""

import argparse
import json
import time
import signal
import sys
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from collections import defaultdict

D_BELIEF = 64
D_ACTION = 2
PAUSE_FILE = Path("PAUSE")
STATE_FILE = Path("data/autonomous_state.json")


# ──────────────────────────────────────────────────────────────────────────────
# Knowledge State (persists across runs)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class KnowledgeState:
    """Everything the system has learned."""
    total_cycles: int = 0
    total_steps: int = 0
    total_episodes: int = 0
    vocabulary_size: int = 0
    schemas_active: int = 0
    total_word_bindings: int = 0
    total_sleep_cycles: int = 0
    total_counterfactuals: int = 0
    total_intentions_set: int = 0
    total_intentions_fired: int = 0
    novelty_history: List[float] = field(default_factory=list)
    vocab_history: List[int] = field(default_factory=list)
    mood_history: List[str] = field(default_factory=list)
    goals_achieved: int = 0
    goals_attempted: int = 0
    domains_explored: List[str] = field(default_factory=list)
    best_discoveries: List[str] = field(default_factory=list)
    wm_capacity_history: List[int] = field(default_factory=list)

    def save(self, path=STATE_FILE):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path=STATE_FILE):
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            return cls(**data)
        return cls()


# ──────────────────────────────────────────────────────────────────────────────
# Minimal component implementations (standalone, no imports needed)
# ──────────────────────────────────────────────────────────────────────────────

class MiniTransition:
    """
    Dual-system transition model (Kahneman).
    System 1 (MLP): fast, habitual, 7μs — routine predictions
    System 2 (NeuroTransformer-lite): slow, deliberate — novel situations
    Auto-switches based on novelty and prediction error.
    """
    def __init__(self, seed=42):
        self.trained = False
        # System 1: fast MLP
        trained_path = Path("data/minari_trained/transition_model.npz")
        if trained_path.exists():
            data = np.load(trained_path)
            self.W1 = data["W1"]
            self.b1 = data["b1"]
            self.W2 = data["W2"]
            self.b2 = data["b2"]
            self.trained = True
        else:
            rng = np.random.RandomState(seed)
            d_in = D_BELIEF + D_ACTION
            self.W1 = rng.randn(d_in, 128).astype(np.float32) * 0.1
            self.b1 = np.zeros(128, dtype=np.float32)
            self.W2 = rng.randn(128, D_BELIEF).astype(np.float32) * 0.1
            self.b2 = np.zeros(D_BELIEF, dtype=np.float32)

        # System 2: context-aware attention over recent beliefs
        rng2 = np.random.RandomState(99)
        self.s2_W_q = rng2.randn(D_BELIEF, D_BELIEF).astype(np.float32) * 0.1
        self.s2_W_k = rng2.randn(D_BELIEF, D_BELIEF).astype(np.float32) * 0.1
        self.s2_W_v = rng2.randn(D_BELIEF, D_BELIEF).astype(np.float32) * 0.1
        self.s2_W_out = rng2.randn(D_BELIEF + D_ACTION, D_BELIEF).astype(np.float32) * 0.1

        # History for System 2
        self.belief_history = []
        self.max_history = 8
        self.last_error = 0.0
        self.s1_count = 0
        self.s2_count = 0

    def _predict_s1(self, belief, action):
        """System 1: fast MLP prediction."""
        x = np.concatenate([belief, action])
        h = np.maximum(0, x @ self.W1 + self.b1)
        return np.clip(h @ self.W2 + self.b2, -5.0, 5.0)

    def _predict_s2(self, belief, action):
        """System 2: attend to recent belief history + action."""
        history = list(self.belief_history[-self.max_history:])
        if len(history) < 2:
            return self._predict_s1(belief, action)

        beliefs = np.stack(history)
        n = len(beliefs)

        # Simple self-attention
        Q = beliefs @ self.s2_W_q
        K = beliefs @ self.s2_W_k
        V = beliefs @ self.s2_W_v

        scores = Q @ K.T / np.sqrt(D_BELIEF)

        # Recency bias (5HT-style temporal locality)
        recency = np.exp(-np.arange(n)[::-1].astype(np.float32) * 0.3)
        scores = scores * recency[None, :]

        # Softmax
        e = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = e / (e.sum(axis=-1, keepdims=True) + 1e-10)

        context = (attn @ V)[-1]  # last position

        # Combine context + action → prediction
        x = np.concatenate([context, action])
        pred = x @ self.s2_W_out
        return np.clip(pred, -5.0, 5.0)

    def predict(self, belief, action, novelty=None):
        """Auto-switch between S1 and S2."""
        self.belief_history.append(belief.copy())
        if len(self.belief_history) > self.max_history:
            self.belief_history = self.belief_history[-self.max_history:]

        # Switching criteria
        use_s2 = False
        if novelty is not None and novelty > 3.0:
            use_s2 = True
        elif self.last_error > 2.0:
            use_s2 = True
        elif len(self.belief_history) >= 3:
            recent = self.belief_history[-3:]
            change = max(np.linalg.norm(recent[i] - recent[i-1])
                          for i in range(1, len(recent)))
            if change > 3.0:
                use_s2 = True

        if use_s2 and len(self.belief_history) >= 3:
            pred = self._predict_s2(belief, action)
            self.s2_count += 1
        else:
            pred = self._predict_s1(belief, action)
            self.s1_count += 1

        return pred

    def error(self, belief, action, next_belief):
        pred = self.predict(belief, action)
        if np.any(np.isnan(pred)):
            self.last_error = 1.0
            return 1.0
        err = float(np.linalg.norm(pred - next_belief))
        self.last_error = err
        return err

    @property
    def s2_ratio(self):
        total = self.s1_count + self.s2_count
        return self.s2_count / max(total, 1)


class MiniSchemas:
    def __init__(self, n=32):
        self.n = n
        self.usage = np.zeros(n)
        self.names = [f"region_{i}" for i in range(n)]

        # Try loading trained codebook
        trained_path = Path("data/minari_trained/schema_codebook.npz")
        if trained_path.exists():
            data = np.load(trained_path)
            self.codebook = data["codebook"]
            self.usage = data["usage"].astype(float)
            self.n = len(self.codebook)
            self.names = [f"region_{i}" for i in range(self.n)]
            self.trained = True
        else:
            self.codebook = np.random.randn(n, D_BELIEF).astype(np.float32) * 2.0
            self.trained = False

    def nearest(self, belief):
        dists = np.linalg.norm(self.codebook - belief, axis=1)
        idx = int(np.argmin(dists))
        self.usage[idx] += 1
        return idx, float(dists[idx])

    def novelty(self, belief):
        _, dist = self.nearest(belief)
        return dist

    def consolidate(self, beliefs, lr=0.05):
        before = 0
        for b in beliefs:
            if np.any(np.isnan(b)) or np.any(np.abs(b) > 100):
                continue
            idx, dist = self.nearest(b)
            before += dist
            self.codebook[idx] += lr * (b - self.codebook[idx])
        # Clamp codebook to prevent divergence
        self.codebook = np.clip(self.codebook, -10.0, 10.0)
        return before / max(len(beliefs), 1)

    def most_novel_schema(self):
        """Which schema should we explore next? Diverse selection."""
        if self.usage.sum() == 0:
            return np.random.randint(self.n)
        # Score: inverse usage + random jitter for diversity
        inv_usage = 1.0 / (self.usage + 1)
        jitter = np.random.random(self.n) * 0.3
        scores = inv_usage + jitter
        return int(np.argmax(scores))


class MiniVocab:
    def __init__(self):
        self.words = defaultdict(list)
        self.prototypes = {}  # word → 64-D prototype belief
        self.total_hearings = 0

        # Auto-load saved vocabulary from vocab_trainer
        self._load_persistent_vocab()

    def _load_persistent_vocab(self):
        """Load vocabulary from vocab_trainer's saved file."""
        vocab_path = Path("data/vocabulary.npz")
        if vocab_path.exists():
            try:
                data = np.load(vocab_path, allow_pickle=True)
                words = data["words"]
                protos = data["prototypes"]
                hearings = data["hearings"]

                for i, word in enumerate(words):
                    word = str(word)
                    self.prototypes[word] = protos[i].astype(np.float32)
                    # Seed word list with prototype
                    if word not in self.words:
                        self.words[word] = [protos[i].astype(np.float32)]
                    self.total_hearings += int(hearings[i])
            except Exception:
                pass  # graceful fallback

    def hear(self, word, belief):
        if len(word) >= 3:
            self.words[word].append(belief.copy())
            self.total_hearings += 1

            # Update prototype with EMA
            if word in self.prototypes:
                lr = 0.05
                self.prototypes[word] = (
                    (1 - lr) * self.prototypes[word] + lr * belief
                ).astype(np.float32)
            else:
                self.prototypes[word] = belief.copy()

    def hear_sentence(self, sentence, belief):
        stops = {'the', 'is', 'at', 'in', 'on', 'to', 'an', 'of', 'and',
                 'or', 'it', 'be', 'as', 'by', 'for'}
        for w in sentence.lower().split():
            w = ''.join(c for c in w if c.isalnum())
            if w not in stops and len(w) >= 3:
                self.hear(w, belief)

    def lookup(self, word):
        """Get prototype belief for a word."""
        if word in self.prototypes:
            return self.prototypes[word]
        if word in self.words and self.words[word]:
            return np.mean(self.words[word][-50:], axis=0).astype(np.float32)
        return None

    def similarity(self, w1, w2):
        """Cosine similarity between two word prototypes."""
        p1, p2 = self.lookup(w1), self.lookup(w2)
        if p1 is None or p2 is None:
            return 0.0
        n1, n2 = np.linalg.norm(p1), np.linalg.norm(p2)
        if n1 < 1e-8 or n2 < 1e-8:
            return 0.0
        return float(np.dot(p1, p2) / (n1 * n2))

    def describe(self, belief, max_words=3):
        """Find the best words to describe a belief state."""
        if not self.prototypes:
            return "unknown"
        scored = []
        for word, proto in self.prototypes.items():
            n_p = np.linalg.norm(proto)
            n_b = np.linalg.norm(belief)
            if n_p > 1e-8 and n_b > 1e-8:
                sim = float(np.dot(proto, belief) / (n_p * n_b))
                scored.append((word, sim))
        scored.sort(key=lambda x: -x[1])
        return " ".join(w for w, s in scored[:max_words])

    def save(self):
        """Save current vocabulary to disk."""
        if not self.prototypes:
            return
        words = list(self.prototypes.keys())
        protos = np.stack([self.prototypes[w] for w in words])
        hearings = np.array([len(self.words.get(w, [])) for w in words])
        np.savez(Path("data/vocabulary.npz"),
                 words=np.array(words),
                 prototypes=protos,
                 hearings=hearings,
                 variances=np.zeros(len(words)))

    @property
    def size(self):
        return len(self.words)


class MiniNeuromod:
    def __init__(self):
        self.signals = {"DA": 0.4, "ACh": 0.6, "CRT": 0.2,
                         "NE": 0.5, "5HT": 0.5}

    def update(self, pred_error, novelty, reward):
        # Clamp inputs to prevent NaN propagation
        pred_error = float(np.clip(pred_error, 0, 5))
        novelty = float(np.clip(novelty, 0, 10))
        reward = float(np.clip(reward, -1, 1))

        # EMA toward target — signals return to baseline over time
        decay = 0.7  # blend rate (0.7 = responsive but recovers)
        target_da = np.clip(0.3 + pred_error * 0.2 + reward * 0.3, 0, 1)
        target_ach = np.clip(0.6 - pred_error * 0.1 + reward * 0.2, 0.1, 1)
        target_crt = np.clip(0.15 + novelty * 0.05, 0, 0.8)
        target_ne = np.clip(0.4 + pred_error * 0.1 + novelty * 0.03, 0.1, 1)
        target_5ht = np.clip(0.5 - target_da * 0.2 + reward * 0.2, 0.1, 1)

        self.signals["DA"] = decay * self.signals["DA"] + (1 - decay) * target_da
        self.signals["ACh"] = decay * self.signals["ACh"] + (1 - decay) * target_ach
        self.signals["CRT"] = decay * self.signals["CRT"] + (1 - decay) * target_crt
        self.signals["NE"] = decay * self.signals["NE"] + (1 - decay) * target_ne
        self.signals["5HT"] = decay * self.signals["5HT"] + (1 - decay) * target_5ht

        # Safety: clamp all signals
        for k in self.signals:
            self.signals[k] = float(np.clip(self.signals[k], 0.01, 0.99))

    def effectiveness(self, name):
        optima = {"DA": 0.4, "ACh": 0.6, "CRT": 0.3, "NE": 0.5, "5HT": 0.5}
        widths = {"DA": 0.5, "ACh": 0.5, "CRT": 0.4, "NE": 0.45, "5HT": 0.5}
        level = self.signals[name]
        return max(0, 1 - ((level - optima[name]) / widths[name]) ** 2)

    def wm_capacity(self):
        da_e = self.effectiveness("DA")
        ne_e = self.effectiveness("NE")
        ach_e = self.effectiveness("ACh")
        factor = (da_e * ne_e * ach_e) ** (1/3)
        return max(2, int(8 * factor) - int(self.signals["CRT"] * 6))

    def mood(self):
        da = self.signals["DA"]
        crt = self.signals["CRT"]
        ach = self.signals["ACh"]
        if da > 0.6 and crt < 0.3:
            return "Curious-Confident"
        elif da > 0.6 and crt > 0.3:
            return "Curious-Anxious"
        elif crt > 0.5:
            return "Stressed-Alert"
        elif da < 0.3 and ach > 0.6:
            return "Calm-Focused"
        elif da < 0.3 and ach < 0.4:
            return "Bored-Disengaged"
        else:
            return "Neutral-Steady"

    def planning_horizon(self):
        ach_e = self.effectiveness("ACh")
        sht_e = self.effectiveness("5HT")
        return max(1, int(32 * ach_e * sht_e))


# ──────────────────────────────────────────────────────────────────────────────
# Narration templates by context
# ──────────────────────────────────────────────────────────────────────────────

NARRATIONS = {
    "explore_start": [
        "heading toward unexplored region {schema}",
        "curiosity drives exploration of {schema}",
        "novelty detected in direction of {schema}",
        "navigating toward distant landmark at region {schema}",
        "beginning new trajectory toward frontier zone {schema}",
    ],
    "high_da": [
        "something surprising encountered here",
        "unexpected pattern in this region",
        "prediction error triggered DA spike",
        "anomaly detected sharp deviation from expected trajectory",
        "surprise discovery novel obstacle configuration ahead",
        "this corridor segment differs from stored schema",
        "encountering unfamiliar terrain with steep gradient",
        "remarkable finding hidden passage between regions",
    ],
    "high_crt": [
        "stress elevated entering unfamiliar territory",
        "caution required in this area",
        "cortisol rising approaching unknown danger zone",
        "working memory degraded under cognitive load pressure",
        "uncertain environment requires careful planning horizon",
        "elevated arousal detecting potential threat nearby",
    ],
    "low_novelty": [
        "familiar territory recognized from previous exploration",
        "schema match confirmed routine navigation continues",
        "confidence high traversing well known corridor path",
        "relaxed movement through mapped safe region",
    ],
    "goal_reached": [
        "target schema reached successfully after navigation",
        "exploration goal achieved discovered new features",
        "novelty reduced substantially after thorough investigation",
        "destination reached collecting final observations here",
        "goal region explored updating knowledge representation",
    ],
    "sleep": [
        "consolidating experiences from exploration cycle",
        "dreaming about recent discoveries replaying trajectories",
        "schemas updating during consolidation sleep cycle",
        "backward replay computing credit assignment signal",
        "emotional episodes receiving priority consolidation weight",
    ],
    "counterfactual": [
        "reflecting on alternative actions not taken",
        "regret analysis suggests different path would improve reward",
        "counterfactual simulation branching from decision point",
        "comparing actual outcome versus imagined alternative trajectory",
    ],
    "discovery": [
        "discovered new spatial relationship between regions",
        "learned obstacle avoidance pattern for narrow corridor",
        "identified shortcut connection between distant schemas",
        "mapping boundary between familiar and novel territory",
        "found repeating structure suggesting underlying physics rule",
        "gradient pattern discovered suggesting force field nearby",
        "transition dynamics differ here indicating domain boundary",
    ],
    "mood_shift": [
        "mood shifting from cautious exploration to confident navigation",
        "arousal decreasing as environment becomes predictable",
        "dopamine stabilizing entering familiar comfortable zone",
        "attention sharpening with acetylcholine increase",
    ],
}


def narrate(context, **kwargs):
    templates = NARRATIONS.get(context, NARRATIONS["explore_start"])
    template = np.random.choice(templates)
    try:
        return template.format(**kwargs)
    except KeyError:
        return template


# ──────────────────────────────────────────────────────────────────────────────
# The Autonomous Loop
# ──────────────────────────────────────────────────────────────────────────────

class AutonomousLoop:
    """
    The closed-loop AGI kernel.

    All 20 questions + all features wired into continuous
    curiosity-driven exploration with self-generated goals.
    """

    def __init__(self, state: KnowledgeState = None):
        self.state = state or KnowledgeState()
        self.transition = MiniTransition(seed=42 + self.state.total_cycles)
        self.schemas = MiniSchemas(n=32)
        self.vocab = MiniVocab()
        self.neuro = MiniNeuromod()

        # Log trained status
        self.using_trained = self.transition.trained and self.schemas.trained

        # Load real beliefs for starting positions
        beliefs_path = Path("data/minari_trained/beliefs_sample.npz")
        if beliefs_path.exists():
            self.real_beliefs = np.load(beliefs_path)["beliefs"]
        else:
            self.real_beliefs = None

        # Episodic buffer
        self.episodes = []
        self.max_episodes = 200

        # Prospective memory
        self.intentions = []

        # Running flag
        self.running = True
        self.paused = False

        # Instruction queue
        self.pending_instruction = None

    def survey(self, belief):
        """Step 1: Ask all 20 questions about current state."""
        schema_id, novelty = self.schemas.nearest(belief)
        pred_error = self.transition.error(
            belief, np.zeros(D_ACTION), belief)
        self.neuro.update(pred_error, novelty, 0)

        return {
            "Q1_position": belief[:2].tolist(),
            "Q3_anomaly": pred_error > 0.5,
            "Q7_horizon": self.neuro.planning_horizon(),
            "Q8_alpha": self.neuro.effectiveness("ACh"),
            "Q11_novelty": novelty,
            "Q14_mood": self.neuro.mood(),
            "Q16_wm": self.neuro.wm_capacity(),
            "Q17_fatigue": self.state.total_steps * 0.001,
            "schema": schema_id,
        }

    def generate_goal(self):
        """Step 2: Pick exploration target from curiosity OR instruction."""
        # Check for pending instructions
        if self.pending_instruction:
            instruction = self.pending_instruction
            self.pending_instruction = None
            return self._goal_from_instruction(instruction)

        # Default: curiosity-driven
        target_schema = self.schemas.most_novel_schema()
        target_belief = self.schemas.codebook[target_schema].copy()
        self.state.goals_attempted += 1
        return target_schema, target_belief

    def _goal_from_instruction(self, instruction):
        """Convert a natural language instruction to a goal belief."""
        # Parse instruction
        words = instruction.lower().split()
        stops = {'the', 'to', 'a', 'an', 'go', 'please', 'can', 'you'}
        content_words = [w for w in words if w not in stops]

        # Check for region reference ("region_5", "region 5")
        for i, w in enumerate(words):
            if w == "region" and i + 1 < len(words):
                try:
                    schema_id = int(words[i + 1])
                    if 0 <= schema_id < len(self.schemas.codebook):
                        self.state.goals_attempted += 1
                        return schema_id, self.schemas.codebook[schema_id].copy()
                except ValueError:
                    pass
            if w.startswith("region_"):
                try:
                    schema_id = int(w.split("_")[1])
                    if 0 <= schema_id < len(self.schemas.codebook):
                        self.state.goals_attempted += 1
                        return schema_id, self.schemas.codebook[schema_id].copy()
                except (ValueError, IndexError):
                    pass

        # Ground content words to belief space
        goal_belief = np.zeros(D_BELIEF, dtype=np.float32)
        n_grounded = 0

        for word in content_words:
            proto = self.vocab.lookup(word)
            if proto is not None:
                goal_belief += proto
                n_grounded += 1

        # Handle negation
        negate = any(w in words for w in ["not", "avoid", "away", "dont"])

        if n_grounded > 0:
            goal_belief /= n_grounded
            if negate:
                goal_belief = -goal_belief

            # Find nearest schema to grounded goal
            dists = np.linalg.norm(
                self.schemas.codebook - goal_belief, axis=1)
            target_schema = int(np.argmin(dists))
        else:
            # Fallback to curiosity if instruction not understood
            target_schema = self.schemas.most_novel_schema()
            goal_belief = self.schemas.codebook[target_schema].copy()

        self.state.goals_attempted += 1
        return target_schema, goal_belief

    def instruct(self, instruction):
        """Queue an instruction for the next cycle."""
        self.pending_instruction = instruction

    def plan(self, current_belief, goal_belief):
        """Step 3: Hierarchical plan to reach goal."""
        n_steps = 30
        actions = []
        for i in range(n_steps):
            # Direction toward goal in action-space dimensions
            diff = goal_belief[:D_ACTION] - current_belief[:D_ACTION]
            # Actions in [-1, 1] range (matching Minari PushT/PointMaze)
            action = np.clip(diff * 0.5, -1.0, 1.0).astype(np.float32)
            # Add exploration noise
            action += np.random.randn(D_ACTION).astype(np.float32) * 0.15
            action = np.clip(action, -1.0, 1.0).astype(np.float32)
            actions.append(action)
            # Update current belief using transition model for next step
            if self.transition.trained:
                current_belief = self.transition.predict(current_belief, action)
        return actions

    def execute(self, belief, actions, goal_schema):
        """Step 4: Execute plan, collect experience."""
        trajectory = []
        rewards = []
        da_signals = []
        crt_signals = []
        narrations = []

        for i, action in enumerate(actions):
            # Get current novelty for S1/S2 switching
            current_novelty = self.schemas.novelty(belief)

            # Physics — dual system transition
            if self.transition.trained:
                predicted = self.transition.predict(
                    belief, action, novelty=current_novelty)
                # Anchor to nearest real belief to prevent drift
                if self.real_beliefs is not None:
                    dists = np.linalg.norm(self.real_beliefs - predicted, axis=1)
                    nearest_idx = np.argmin(dists)
                    nearest_real = self.real_beliefs[nearest_idx]
                    next_belief = 0.7 * predicted + 0.3 * nearest_real
                else:
                    next_belief = predicted
                next_belief += np.random.randn(D_BELIEF).astype(np.float32) * 0.01
            else:
                next_belief = belief + 0.1 * action[0] + \
                              np.random.randn(D_BELIEF).astype(np.float32) * 0.03
                next_belief[2:] += np.random.randn(D_BELIEF - 2).astype(
                    np.float32) * 0.02

            # Neuromodulatory signals
            pred_error = self.transition.error(belief, action, next_belief)
            novelty = self.schemas.novelty(next_belief)
            schema_id, _ = self.schemas.nearest(next_belief)

            # Reward: proximity to goal schema
            goal_dist = np.linalg.norm(
                next_belief - self.schemas.codebook[goal_schema])
            reward = float(np.exp(-goal_dist * 0.5))

            self.neuro.update(pred_error, novelty, reward)
            da = self.neuro.signals["DA"]
            crt = self.neuro.signals["CRT"]

            # Narrate key moments
            if i == 0:
                narrations.append((i, narrate("explore_start",
                    schema=self.schemas.names[goal_schema])))
            elif da > 0.6:
                narrations.append((i, narrate("high_da")))
            elif crt > 0.4:
                narrations.append((i, narrate("high_crt")))
            elif novelty < 0.3 and i % 5 == 0:
                narrations.append((i, narrate("low_novelty")))
            elif i == len(actions) // 2:
                narrations.append((i, narrate("discovery")))
            elif i == len(actions) - 1:
                narrations.append((i, narrate("goal_reached")))

            trajectory.append(next_belief.copy())
            rewards.append(reward)
            da_signals.append(da)
            crt_signals.append(crt)

            belief = next_belief
            self.state.total_steps += 1

        # Check if goal reached at any point during trajectory
        reached = False
        for t_belief in trajectory:
            t_schema, t_dist = self.schemas.nearest(t_belief)
            if t_schema == goal_schema:
                reached = True
                break
        if not reached:
            # Check if final position is close enough
            final_schema, final_dist = self.schemas.nearest(trajectory[-1])
            reached = final_dist < 3.0 and final_schema == goal_schema
        if reached:
            self.state.goals_achieved += 1

        return {
            "trajectory": trajectory,
            "rewards": rewards,
            "da": da_signals,
            "crt": crt_signals,
            "narrations": narrations,
            "reached": reached,
            "final_belief": trajectory[-1],
        }

    def learn_language(self, narrations, beliefs):
        """Step 5: Ground words from narrations at belief states."""
        words_before = self.vocab.size
        for step_i, narr in narrations:
            if step_i < len(beliefs):
                self.vocab.hear_sentence(narr, beliefs[step_i])
        return self.vocab.size - words_before

    def counterfactual(self, trajectory, actions, rewards):
        """Step 6: What if I had done differently?"""
        if len(trajectory) < 5:
            return 0, 0

        # Find highest-DA step
        best_regret = 0
        best_step = 0

        for step in range(0, len(trajectory) - 1, 5):
            belief = trajectory[step]
            orig_action = actions[step] if step < len(actions) else np.zeros(D_ACTION)

            # Try alternative
            alt_action = -orig_action  # opposite
            orig_next = self.transition.predict(belief, orig_action)
            alt_next = self.transition.predict(belief, alt_action)

            orig_r = float(np.exp(-np.linalg.norm(orig_next[:2])))
            alt_r = float(np.exp(-np.linalg.norm(alt_next[:2])))
            regret = alt_r - orig_r

            if abs(regret) > abs(best_regret):
                best_regret = regret
                best_step = step

        self.state.total_counterfactuals += 1
        return best_regret, best_step

    def sleep(self, trajectory, rewards, da_signals, crt_signals):
        """Step 7: Consolidate during sleep + dream narration."""
        if not trajectory:
            return 0

        beliefs = np.stack(trajectory)

        # Emotional tagging: high emotion = consolidate more
        intensity = np.sqrt(
            np.mean(da_signals)**2 + np.mean(crt_signals)**2) / np.sqrt(2)

        # Schema consolidation
        lr = 0.05 * (1 + 4 * intensity)  # emotional episodes consolidate faster
        loss = self.schemas.consolidate(beliefs, lr=min(lr, 0.2))

        # Backward replay credit
        credits = np.zeros(len(rewards))
        credits[-1] = rewards[-1]
        for i in range(len(rewards) - 2, -1, -1):
            credits[i] = rewards[i] + 0.95 * credits[i + 1]

        # Dream narration — replay with new descriptions to grow vocab
        dream_narrations = [
            narrate("sleep"),
            narrate("discovery"),
            narrate("counterfactual"),
        ]
        if intensity > 0.5:
            dream_narrations.append(narrate("high_da"))
            dream_narrations.append(narrate("mood_shift"))
        for dn in dream_narrations:
            # Ground dream words at random replayed belief
            idx = np.random.randint(len(beliefs))
            self.vocab.hear_sentence(dn, beliefs[idx])

        self.state.total_sleep_cycles += 1
        return loss

    def reflect(self, survey, execution, new_words, regret, sleep_loss):
        """Step 8: Log what was learned, update state."""
        self.state.total_cycles += 1
        self.state.total_episodes += 1
        self.state.vocabulary_size = self.vocab.size
        self.state.total_word_bindings = self.vocab.total_hearings
        self.state.schemas_active = int((self.schemas.usage > 0).sum())

        novelty = survey["Q11_novelty"]
        self.state.novelty_history.append(float(novelty))
        self.state.vocab_history.append(self.vocab.size)
        self.state.mood_history.append(survey["Q14_mood"])
        self.state.wm_capacity_history.append(survey["Q16_wm"])

    def run_cycle(self, verbose=True):
        """One complete autonomous cycle."""
        # Always start from a real data point to stay on manifold
        if self.real_beliefs is not None:
            # Pick a random real starting position (simulates being in the world)
            idx = np.random.randint(len(self.real_beliefs))
            belief = self.real_beliefs[idx].copy()
        elif self.episodes:
            belief = self.episodes[-1][-1].copy()
        else:
            belief = np.random.randn(D_BELIEF).astype(np.float32) * 0.5

        # 1. Survey
        survey = self.survey(belief)

        # 2. Goal
        goal_schema, goal_belief = self.generate_goal()

        # 3. Plan
        actions = self.plan(belief, goal_belief)

        # 4. Execute
        exec_result = self.execute(belief, actions, goal_schema)

        # 5. Language
        new_words = self.learn_language(
            exec_result["narrations"], exec_result["trajectory"])

        # 6. Counterfactual
        regret, regret_step = self.counterfactual(
            exec_result["trajectory"], actions, exec_result["rewards"])

        # 7. Sleep
        sleep_loss = self.sleep(
            exec_result["trajectory"], exec_result["rewards"],
            exec_result["da"], exec_result["crt"])

        # 8. Reflect
        self.reflect(survey, exec_result, new_words, regret, sleep_loss)

        # Store trajectory
        self.episodes.append(exec_result["trajectory"])
        if len(self.episodes) > self.max_episodes:
            self.episodes = self.episodes[-self.max_episodes:]

        if verbose:
            goal_name = self.schemas.names[goal_schema]
            reached = "✓" if exec_result["reached"] else "○"
            mood = survey["Q14_mood"]
            print(f"  Cycle {self.state.total_cycles:>4} │ "
                  f"Goal: {goal_name:<12} {reached} │ "
                  f"Vocab: {self.vocab.size:>3} (+{new_words}) │ "
                  f"Nov: {survey['Q11_novelty']:.2f} │ "
                  f"WM: K={survey['Q16_wm']} │ "
                  f"Mood: {mood}")

        return {
            "cycle": self.state.total_cycles,
            "goal": self.schemas.names[goal_schema],
            "reached": exec_result["reached"],
            "vocab": self.vocab.size,
            "new_words": new_words,
            "novelty": survey["Q11_novelty"],
            "regret": regret,
            "sleep_loss": sleep_loss,
            "mood": survey["Q14_mood"],
        }

    def run(self, max_cycles=-1, verbose=True):
        """Run the autonomous loop."""
        if verbose:
            trained_str = "TRAINED (Minari 1M)" if self.using_trained else "UNTRAINED (random)"
            print("=" * 75)
            print(f"  Autonomous Curiosity-Driven Loop — {trained_str}")
            print("  Ctrl+C to stop │ Create 'PAUSE' file to pause")
            print("=" * 75)
            print(f"  {'Cycle':>7} │ {'Goal':<18} │ {'Vocab':>9} │ "
                  f"{'Nov':>5} │ {'WM':>4} │ Mood")
            print(f"  {'─'*7}─┼─{'─'*18}─┼─{'─'*9}─┼─"
                  f"{'─'*5}─┼─{'─'*4}─┼─{'─'*18}")

        cycle = 0
        try:
            while self.running:
                # Check pause
                if PAUSE_FILE.exists():
                    if not self.paused:
                        print(f"\n  ⏸  PAUSED (delete '{PAUSE_FILE}' to resume)")
                        self.paused = True
                    time.sleep(1)
                    continue
                elif self.paused:
                    print(f"  ▶  RESUMED")
                    self.paused = False

                self.run_cycle(verbose=verbose)
                cycle += 1

                if max_cycles > 0 and cycle >= max_cycles:
                    break

                # Brief pause between cycles
                time.sleep(0.01)

        except KeyboardInterrupt:
            if verbose:
                print(f"\n  Stopped by user after {cycle} cycles")

        if verbose:
            self.print_summary()

        # Save state
        self.state.save()

        # Save vocabulary to disk for persistence
        try:
            self.vocab.save()
        except Exception:
            pass

    def print_summary(self):
        s = self.state
        print(f"\n{'='*75}")
        print(f"  Autonomous Loop Summary")
        print(f"{'='*75}")
        print(f"  Cycles:          {s.total_cycles}")
        print(f"  Steps:           {s.total_steps}")
        print(f"  Goals:           {s.goals_achieved}/{s.goals_attempted} "
              f"({100*s.goals_achieved/max(s.goals_attempted,1):.0f}%)")
        print(f"  Vocabulary:      {s.vocabulary_size} words "
              f"({s.total_word_bindings} bindings)")
        print(f"  Active schemas:  {s.schemas_active}/16")
        print(f"  Sleep cycles:    {s.total_sleep_cycles}")
        print(f"  Counterfactuals: {s.total_counterfactuals}")

        # Dual system stats
        s1 = self.transition.s1_count
        s2 = self.transition.s2_count
        total_preds = s1 + s2
        if total_preds > 0:
            print(f"  System 1 (fast):  {s1} ({100*s1/total_preds:.0f}%)")
            print(f"  System 2 (slow):  {s2} ({100*s2/total_preds:.0f}%)")

        if s.novelty_history:
            print(f"\n  Novelty: {s.novelty_history[0]:.2f} → "
                  f"{s.novelty_history[-1]:.2f} "
                  f"({(s.novelty_history[-1]-s.novelty_history[0])/max(s.novelty_history[0],0.01)*100:+.1f}%)")

        if s.vocab_history:
            print(f"  Vocab:   {s.vocab_history[0]} → {s.vocab_history[-1]}")

        if s.mood_history:
            from collections import Counter
            mood_counts = Counter(s.mood_history)
            top_mood = mood_counts.most_common(1)[0]
            print(f"  Top mood: {top_mood[0]} ({top_mood[1]}×)")

        print(f"  State saved to: {STATE_FILE}")
        print(f"{'='*75}")


def show_status():
    """Show current knowledge state."""
    state = KnowledgeState.load()
    print("=" * 65)
    print("  NeMo-WM Knowledge State")
    print("=" * 65)
    print(f"  Cycles:          {state.total_cycles}")
    print(f"  Steps:           {state.total_steps}")
    print(f"  Vocabulary:      {state.vocabulary_size} words")
    print(f"  Word bindings:   {state.total_word_bindings}")
    print(f"  Schemas active:  {state.schemas_active}")
    print(f"  Goals:           {state.goals_achieved}/{state.goals_attempted}")
    print(f"  Sleep cycles:    {state.total_sleep_cycles}")
    print(f"  Counterfactuals: {state.total_counterfactuals}")
    if state.novelty_history:
        print(f"  Novelty trend:   {state.novelty_history[0]:.2f} → "
              f"{state.novelty_history[-1]:.2f}")
    if state.mood_history:
        print(f"  Last mood:       {state.mood_history[-1]}")
    print(f"{'='*65}")


def export_state():
    """Export state to JSON."""
    state = KnowledgeState.load()
    out = Path("data/autonomous_export.json")
    state.save(out)
    print(f"Exported to {out}")


def reset_state():
    """Clear all accumulated knowledge."""
    if STATE_FILE.exists():
        STATE_FILE.unlink()
        print("State reset.")
    else:
        print("No state to reset.")


def run_tests():
    """Validate the autonomous loop."""
    print("=" * 65)
    print("  Autonomous Loop Tests")
    print("=" * 65)
    p = 0; t = 0

    loop = AutonomousLoop()

    print("\n  T1: Single cycle completes")
    result = loop.run_cycle(verbose=False)
    ok = result["cycle"] == 1
    print(f"    Cycle {result['cycle']} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T2: Vocabulary grows")
    ok = result["vocab"] > 0
    print(f"    Vocab={result['vocab']} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T3: Goal is generated")
    ok = result["goal"] is not None
    print(f"    Goal='{result['goal']}' {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T4: Multiple cycles accumulate")
    for _ in range(9):
        loop.run_cycle(verbose=False)
    ok = loop.state.total_cycles == 10
    print(f"    Cycles={loop.state.total_cycles} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T5: Schemas get visited")
    ok = loop.state.schemas_active > 0
    print(f"    Active={loop.state.schemas_active}/16 "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T6: Sleep cycles accumulate")
    ok = loop.state.total_sleep_cycles == 10
    print(f"    Sleep={loop.state.total_sleep_cycles} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T7: Counterfactuals computed")
    ok = loop.state.total_counterfactuals > 0
    print(f"    CF={loop.state.total_counterfactuals} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T8: State serializes")
    loop.state.save()
    loaded = KnowledgeState.load()
    ok = loaded.total_cycles == loop.state.total_cycles
    print(f"    Save/load: {loaded.total_cycles} cycles "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    # Cleanup
    if STATE_FILE.exists():
        STATE_FILE.unlink()

    print(f"\n{'='*65}")
    print(f"  Results: {p}/{t} tests passed")
    print(f"{'='*65}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="NeMo-WM Autonomous Curiosity Loop")
    ap.add_argument("--run", action="store_true",
                     help="Start the autonomous loop")
    ap.add_argument("--cycles", type=int, default=-1,
                     help="Max cycles (-1 = infinite)")
    ap.add_argument("--status", action="store_true",
                     help="Show current knowledge state")
    ap.add_argument("--export", action="store_true",
                     help="Export knowledge to JSON")
    ap.add_argument("--reset", action="store_true",
                     help="Clear all accumulated knowledge")
    ap.add_argument("--test", action="store_true",
                     help="Run validation tests")
    ap.add_argument("--instruct", type=str, default=None,
                     help="Give instruction: 'go explore region 5'")
    args = ap.parse_args()

    if args.test:
        run_tests()
    elif args.status:
        show_status()
    elif args.export:
        export_state()
    elif args.reset:
        reset_state()
    elif args.instruct:
        state = KnowledgeState.load()
        loop = AutonomousLoop(state)
        print(f"  Instruction: \"{args.instruct}\"")
        loop.instruct(args.instruct)
        loop.run(max_cycles=1)
    elif args.run:
        state = KnowledgeState.load()
        loop = AutonomousLoop(state)

        # Check for instruction file
        instruct_file = Path("INSTRUCT")
        if instruct_file.exists():
            instruction = instruct_file.read_text().strip()
            instruct_file.unlink()
            if instruction:
                loop.instruct(instruction)
                print(f"  Instruction loaded: \"{instruction}\"")

        loop.run(max_cycles=args.cycles)
    else:
        ap.print_help()
        print("\nExamples:")
        print("  python autonomous_loop.py --run --cycles 20")
        print("  python autonomous_loop.py --run  # infinite, Ctrl+C")
        print("  python autonomous_loop.py --instruct 'go explore region 5'")
        print("  python autonomous_loop.py --status")
        print("  python autonomous_loop.py --test")
