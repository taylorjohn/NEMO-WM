"""
language_v2.py — Predictive Language Comprehension
=====================================================
Five upgrades to WordGrounder that move from pattern matching
to genuine comprehension via world model simulation.

1. Negation/Modifiers     — "not dangerous" flips the belief vector
2. BeliefAccumulator      — word order matters, Kalman-style updates
3. PredictiveGrounder     — sentences understood by simulating them
4. ContrastiveLearning    — meanings defined by what they're NOT
5. EpisodicReplayLearning — self-teaching from narrated memories

All biologically inspired. No LLM. No parser. No pretrained anything.

Usage:
    python language_v2.py
    python language_v2.py --demo predictive
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

from word_grounder import WordGrounder, SentenceComprehender, D_BELIEF


# ──────────────────────────────────────────────────────────────────────────────
# 1. Negation & Modifiers — vector arithmetic on beliefs
# ──────────────────────────────────────────────────────────────────────────────

class ModifierProcessor:
    """
    Handle negation, intensifiers, and diminishers via belief vector ops.
    
    "not dangerous" = -dangerous_belief
    "very fast" = fast_belief * 1.5
    "slightly warm" = warm_belief * 0.5
    
    No parser. Just detect modifier words and apply vector transforms.
    """

    NEGATION = {'not', 'no', 'never', "dont", 'without', 'none',
                'neither', 'nor', 'nothing', 'nowhere', 'nobody',
                'impossible', 'unable', 'cannot'}

    INTENSIFIERS = {'very', 'extremely', 'really', 'highly', 'super',
                    'incredibly', 'absolutely', 'totally', 'completely',
                    'massive', 'enormous', 'huge'}

    DIMINISHERS = {'slightly', 'barely', 'somewhat', 'little', 'bit',
                   'partly', 'mildly', 'faintly', 'weakly', 'minor'}

    COMPARATIVES = {'more', 'less', 'most', 'least', 'bigger', 'smaller',
                    'faster', 'slower', 'stronger', 'weaker'}

    def __init__(self):
        self._modifier_count = 0

    def process(self, words: List[str],
                grounder: WordGrounder) -> np.ndarray:
        """
        Process a word list with modifier awareness.
        Returns composed belief with modifiers applied.
        """
        beliefs = []
        pending_modifier = None

        for word in words:
            word_lower = word.lower()

            # Check for modifiers
            if word_lower in self.NEGATION:
                pending_modifier = 'negate'
                self._modifier_count += 1
                continue
            elif word_lower in self.INTENSIFIERS:
                pending_modifier = 'intensify'
                self._modifier_count += 1
                continue
            elif word_lower in self.DIMINISHERS:
                pending_modifier = 'diminish'
                self._modifier_count += 1
                continue

            # Get word belief
            meaning = grounder.understand(word_lower)
            if meaning is None:
                pending_modifier = None
                continue

            # Apply pending modifier
            if pending_modifier == 'negate':
                meaning = -meaning
            elif pending_modifier == 'intensify':
                meaning = meaning * 1.5
            elif pending_modifier == 'diminish':
                meaning = meaning * 0.5

            beliefs.append(meaning)
            pending_modifier = None

        if not beliefs:
            return np.zeros(grounder.d_belief)

        return np.mean(beliefs, axis=0)

    def has_negation(self, words: List[str]) -> bool:
        return any(w.lower() in self.NEGATION for w in words)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Belief Accumulator — word order matters
# ──────────────────────────────────────────────────────────────────────────────

class BeliefAccumulator:
    """
    Running belief state across a sentence.
    Like a Kalman filter: each word UPDATES the belief, not replaces it.
    
    "dog bites man" != "man bites dog" because:
      after "dog": belief = dog_prototype
      after "dog bites": belief shifts toward action+aggression
      after "dog bites man": belief adds human-target
    vs:
      after "man": belief = human_prototype  
      after "man bites": belief shifts toward action+aggression
      after "man bites dog": belief adds dog-target
    
    The final beliefs differ because the starting point differs.
    """

    def __init__(self, d_belief: int = D_BELIEF,
                 gain: float = 0.3,
                 decay: float = 0.95):
        self.d_belief = d_belief
        self.gain = gain        # how much each new word shifts belief
        self.decay = decay      # how much old context fades
        self.modifier = ModifierProcessor()

    def process_sentence(self, sentence: str,
                          grounder: WordGrounder) -> Dict:
        """
        Process sentence word by word, building running belief.
        
        Returns:
            dict with accumulated belief, word contributions,
            confidence, and processing trace
        """
        words = grounder._tokenize(sentence)
        state = np.zeros(self.d_belief, dtype=np.float32)
        trace = []
        grounded_count = 0
        pending_modifier = None

        for i, word in enumerate(words):
            word_lower = word.lower()

            # Check modifiers
            if word_lower in self.modifier.NEGATION:
                pending_modifier = 'negate'
                trace.append({'word': word, 'type': 'modifier',
                              'effect': 'negate'})
                continue
            elif word_lower in self.modifier.INTENSIFIERS:
                pending_modifier = 'intensify'
                trace.append({'word': word, 'type': 'modifier',
                              'effect': 'intensify'})
                continue
            elif word_lower in self.modifier.DIMINISHERS:
                pending_modifier = 'diminish'
                trace.append({'word': word, 'type': 'modifier',
                              'effect': 'diminish'})
                continue

            meaning = grounder.understand(word_lower)
            if meaning is None:
                trace.append({'word': word, 'type': 'unknown'})
                pending_modifier = None
                continue

            # Apply modifier
            if pending_modifier == 'negate':
                meaning = -meaning
            elif pending_modifier == 'intensify':
                meaning = meaning * 1.5
            elif pending_modifier == 'diminish':
                meaning = meaning * 0.5
            pending_modifier = None

            # Kalman-style update
            old_state = state.copy()
            state = self.decay * state + self.gain * meaning

            # Normalize to prevent drift
            norm = np.linalg.norm(state)
            if norm > 3.0:
                state = state * 3.0 / norm

            grounded_count += 1
            shift = float(np.linalg.norm(state - old_state))
            trace.append({
                'word': word,
                'type': 'grounded',
                'modifier': pending_modifier,
                'shift': shift,
            })

        confidence = grounded_count / max(len(words), 1)

        return {
            'belief': state,
            'confidence': confidence,
            'grounded_count': grounded_count,
            'total_words': len(words),
            'trace': trace,
            'understood': confidence > 0.3,
        }


# ──────────────────────────────────────────────────────────────────────────────
# 3. Predictive Grounder — comprehension IS simulation
# ──────────────────────────────────────────────────────────────────────────────

class SimpleTransitionModel(nn.Module):
    """Minimal transition model for predictive grounding."""
    def __init__(self, d_belief=D_BELIEF, d_action=D_BELIEF):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_belief + d_action, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, d_belief),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class PredictiveGrounder:
    """
    Understand sentences by SIMULATING them in the world model.
    
    "if I push the block left it moves left"
      1. Ground "push" → action belief (from ActionPrimitives)
      2. Ground "block" → object belief
      3. Run transition model: predict(block, push_left)
      4. Ground "moves left" → expected outcome belief
      5. Compare prediction vs expectation
      6. High match = sentence understood through simulation
    
    This is predictive processing: comprehension = prediction + verification.
    The brain doesn't parse syntax — it simulates meaning.
    """

    # Words that signal actions (cause)
    ACTION_WORDS = {'push', 'pull', 'move', 'drop', 'throw', 'hit',
                    'roll', 'slide', 'turn', 'stop', 'catch', 'grab',
                    'release', 'jump', 'climb', 'fall', 'spin', 'break'}

    # Words that signal results (effect)
    RESULT_WORDS = {'moves', 'falls', 'stops', 'breaks', 'bounces',
                    'slides', 'rolls', 'spins', 'changes', 'appears',
                    'disappears', 'accelerates'}

    # Words that signal conditions
    CONDITION_WORDS = {'if', 'when', 'unless', 'because', 'since',
                       'after', 'before', 'while', 'until'}

    def __init__(self, grounder: WordGrounder,
                 transition_model: nn.Module = None):
        self.grounder = grounder
        self.accumulator = BeliefAccumulator()

        if transition_model is None:
            self.transition = SimpleTransitionModel()
            self.transition.eval()
            self._train_transition_from_vocabulary()
        else:
            self.transition = transition_model

    def _train_transition_from_vocabulary(self):
        """Bootstrap transition model from word-belief associations."""
        # Create cause-effect pairs from vocabulary
        pairs = []
        for action_word in self.ACTION_WORDS:
            action_belief = self.grounder.understand(action_word)
            if action_belief is None:
                continue
            # Find associated result words
            for result_word in self.RESULT_WORDS:
                result_belief = self.grounder.understand(result_word)
                if result_belief is None:
                    continue
                # If action and result have positive similarity,
                # they might be a cause-effect pair
                sim = np.dot(action_belief, result_belief) / (
                    np.linalg.norm(action_belief) *
                    np.linalg.norm(result_belief) + 1e-8)
                if sim > 0.2:
                    # Create training pair: state + action → result
                    state = np.zeros(D_BELIEF, dtype=np.float32)
                    pairs.append((state, action_belief, result_belief))

        if len(pairs) < 3:
            return

        # Quick training
        optimizer = torch.optim.Adam(self.transition.parameters(), lr=1e-3)
        self.transition.train()

        for epoch in range(50):
            total_loss = 0
            for state, action, result in pairs:
                s = torch.from_numpy(state).unsqueeze(0)
                a = torch.from_numpy(action).unsqueeze(0)
                r = torch.from_numpy(result).unsqueeze(0)

                pred = self.transition(s, a)
                loss = nn.functional.mse_loss(pred, r)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        self.transition.eval()

    def comprehend_predictive(self, sentence: str) -> Dict:
        """
        Understand a sentence by simulating it.
        
        Returns:
            dict with:
              prediction_match: how well simulation matches stated outcome
              causal_chain: detected cause → effect pairs
              simulated: whether simulation was possible
              confidence: overall comprehension confidence
        """
        words = self.grounder._tokenize(sentence)

        # Find action words and split into cause/effect
        action_idx = None
        for i, w in enumerate(words):
            if w.lower() in self.ACTION_WORDS:
                action_idx = i
                break

        if action_idx is None:
            # No action word — fall back to accumulator
            result = self.accumulator.process_sentence(
                sentence, self.grounder)
            result['simulated'] = False
            result['prediction_match'] = 0.0
            return result

        # Split: context before action, action, expected result after
        context_words = words[:action_idx]
        action_word = words[action_idx]
        result_words = words[action_idx + 1:]

        # Ground each part
        context_belief = np.zeros(D_BELIEF, dtype=np.float32)
        for w in context_words:
            m = self.grounder.understand(w)
            if m is not None:
                context_belief += m

        action_belief = self.grounder.understand(action_word)
        if action_belief is None:
            action_belief = np.zeros(D_BELIEF, dtype=np.float32)

        # Simulate: what does the world model predict?
        with torch.no_grad():
            s = torch.from_numpy(context_belief).unsqueeze(0)
            a = torch.from_numpy(action_belief).unsqueeze(0)
            predicted = self.transition(s, a).squeeze(0).numpy()

        # Ground the expected result
        expected = np.zeros(D_BELIEF, dtype=np.float32)
        grounded_results = 0
        for w in result_words:
            m = self.grounder.understand(w)
            if m is not None:
                expected += m
                grounded_results += 1

        # Compare prediction vs expectation
        if np.linalg.norm(predicted) > 1e-8 and np.linalg.norm(expected) > 1e-8:
            match = float(np.dot(predicted, expected) / (
                np.linalg.norm(predicted) * np.linalg.norm(expected)))
        else:
            match = 0.0

        # Overall confidence
        total_grounded = sum(1 for w in words
                             if self.grounder.understand(w) is not None)
        confidence = total_grounded / max(len(words), 1)

        return {
            'belief': predicted,
            'confidence': confidence,
            'prediction_match': match,
            'simulated': True,
            'understood': confidence > 0.3 and match > 0.1,
            'action': action_word,
            'context': context_words,
            'expected_result': result_words,
            'grounded_count': total_grounded,
            'total_words': len(words),
            'causal_chain': f"{'→'.join(context_words)} + {action_word} → {'→'.join(result_words)}",
        }


# ──────────────────────────────────────────────────────────────────────────────
# 4. Contrastive Learning — meanings by what they're NOT
# ──────────────────────────────────────────────────────────────────────────────

class ContrastiveGrounder:
    """
    Learn word meanings by contrast, not just co-occurrence.
    
    "gravity" isn't just "downward force" — it's:
      "downward force AND NOT friction AND NOT magnetic"
    
    Contrastive vectors make meanings more discriminative.
    This mirrors how children learn: "that's NOT a dog, it's a cat."
    """

    def __init__(self, grounder: WordGrounder):
        self.grounder = grounder
        self.contrastive: Dict[str, Dict[str, np.ndarray]] = {}

    def build_contrasts(self, word_groups: Dict[str, List[str]] = None):
        """
        Build contrastive meanings within semantic groups.
        
        word_groups maps category → list of words that should be
        distinguished from each other.
        """
        if word_groups is None:
            word_groups = {
                'forces': ['gravity', 'friction', 'magnetic', 'spring',
                           'buoyancy', 'drag'],
                'directions': ['left', 'right', 'up', 'down', 'forward',
                               'backward'],
                'emotions': ['danger', 'safe', 'calm', 'stressed',
                             'curious', 'bored'],
                'speeds': ['fast', 'slow', 'quickly', 'slowly'],
                'actions': ['push', 'pull', 'move', 'stop', 'turn',
                            'grab'],
                'sizes': ['big', 'small', 'heavy', 'light'],
            }

        for category, words in word_groups.items():
            prototypes = {}
            for w in words:
                proto = self.grounder.understand(w)
                if proto is not None:
                    prototypes[w] = proto

            # For each word, compute "what makes it different"
            for word, proto in prototypes.items():
                self.contrastive[word] = {}
                others = [p for w2, p in prototypes.items() if w2 != word]
                if not others:
                    continue

                # Contrastive = prototype - mean(others)
                mean_others = np.mean(others, axis=0)
                self.contrastive[word]['_discriminative'] = proto - mean_others

                # Pairwise contrasts
                for other_word, other_proto in prototypes.items():
                    if other_word != word:
                        self.contrastive[word][other_word] = proto - other_proto

    def discriminative_similarity(self, word_a: str, word_b: str) -> float:
        """
        Similarity using contrastive (discriminative) vectors
        instead of raw prototypes.
        """
        if word_a not in self.contrastive or word_b not in self.contrastive:
            return self.grounder.similarity(word_a, word_b)

        a = self.contrastive[word_a].get('_discriminative')
        b = self.contrastive[word_b].get('_discriminative')

        if a is None or b is None:
            return self.grounder.similarity(word_a, word_b)

        norm_a = np.linalg.norm(a) + 1e-8
        norm_b = np.linalg.norm(b) + 1e-8
        return float(np.dot(a, b) / (norm_a * norm_b))

    def explain_difference(self, word_a: str, word_b: str) -> str:
        """Why are these words different? In belief-space terms."""
        if word_a not in self.contrastive:
            return f"Don't know enough about '{word_a}'"
        if word_b not in self.contrastive.get(word_a, {}):
            return f"Haven't compared '{word_a}' with '{word_b}'"

        diff = self.contrastive[word_a][word_b]
        top_dims = np.argsort(np.abs(diff))[-3:][::-1]
        return (f"'{word_a}' differs from '{word_b}' most in "
                f"dimensions {list(top_dims)} "
                f"(magnitudes: {[f'{diff[d]:+.2f}' for d in top_dims]})")


# ──────────────────────────────────────────────────────────────────────────────
# 5. Episodic Replay Learning — self-teaching from memories
# ──────────────────────────────────────────────────────────────────────────────

class EpisodicReplayLearner:
    """
    During sleep: replay memories, narrate them, learn words.
    
    The system teaches itself language by:
      1. Sample episode from episodic buffer
      2. Describe the episode using known words
      3. Bind description words to the replayed belief
      4. Vocabulary grows from own experience
    
    This is how children learn in their sleep:
    memory consolidation + language rehearsal.
    """

    # Simple templates for generating narration from belief properties
    TEMPLATES = {
        'speed': [
            (0.7, "moving fast"),
            (0.3, "moving at moderate speed"),
            (-0.3, "moving slowly"),
            (-0.7, "nearly stopped"),
        ],
        'vertical': [
            (0.5, "going uphill"),
            (-0.5, "going downhill"),
            (0.0, "on flat ground"),
        ],
        'threat': [
            (0.7, "in a dangerous area"),
            (0.3, "in a risky situation"),
            (-0.3, "in a safe area"),
        ],
        'curiosity': [
            (0.7, "exploring something new"),
            (0.3, "seeing something interesting"),
            (-0.3, "in familiar territory"),
        ],
    }

    # Map belief dimensions to semantic categories
    DIM_MAP = {
        'speed': 59,
        'vertical': 0,
        'threat': 34,
        'curiosity': 36,
    }

    def __init__(self, grounder: WordGrounder):
        self.grounder = grounder
        self.replay_count = 0
        self.words_learned = 0

    def narrate_belief(self, belief: np.ndarray) -> str:
        """Generate narration from belief state without LLM."""
        parts = []
        for category, templates in self.TEMPLATES.items():
            dim = self.DIM_MAP.get(category)
            if dim is None or dim >= len(belief):
                continue
            val = belief[dim]
            # Find closest template
            best_template = min(templates,
                                key=lambda t: abs(t[0] - val))
            if abs(val) > 0.2:  # only describe if signal is strong
                parts.append(best_template[1])

        return ". ".join(parts) if parts else "quiet moment"

    def replay_and_learn(self, beliefs: List[np.ndarray],
                          das: List[float] = None,
                          moods: List[str] = None,
                          n_episodes: int = 50,
                          verbose: bool = True) -> Dict:
        """
        Replay episodes and learn language from self-narration.
        
        Args:
            beliefs: list of belief states (from episodic buffer)
            das: dopamine values for each
            moods: mood names for each
            n_episodes: how many to replay
        """
        if das is None:
            das = [0.3] * len(beliefs)
        if moods is None:
            moods = [''] * len(beliefs)

        new_words_before = self.grounder.vocab_size
        n = min(n_episodes, len(beliefs))

        for i in range(n):
            belief = beliefs[i]
            da = das[i] if i < len(das) else 0.3
            mood = moods[i] if i < len(moods) else ''

            # Narrate the replayed memory
            narration = self.narrate_belief(belief)

            # Bind narration words to the replayed belief
            self.grounder.hear_sentence(
                narration, belief, da=da, mood=mood,
                source="episodic_replay")

            self.replay_count += 1

        new_words = self.grounder.vocab_size - new_words_before
        self.words_learned += new_words

        result = {
            'episodes_replayed': n,
            'new_words': new_words,
            'total_vocab': self.grounder.vocab_size,
            'total_replays': self.replay_count,
        }

        if verbose:
            print(f"  Replayed {n} episodes, learned {new_words} new words, "
                  f"vocab={self.grounder.vocab_size}")

        return result


# ──────────────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────────────

def build_trained_grounder():
    """Build a grounder with vocabulary from curriculum."""
    from train_language_curriculum import build_curriculum
    grounder = WordGrounder(d_belief=D_BELIEF)
    curriculum = build_curriculum(d_belief=D_BELIEF)
    for word, belief, da, mood, physics, domain in curriculum:
        grounder.hear(word, belief, da=max(0, da), mood=mood,
                       physics=physics, source=f"curriculum_{domain}")
    return grounder


def demo_negation():
    print("\n" + "=" * 60)
    print("  1. NEGATION & MODIFIERS")
    print("  'not dangerous' = flip the belief vector")
    print("=" * 60)

    grounder = build_trained_grounder()
    modifier = ModifierProcessor()

    pairs = [
        ("dangerous", "not dangerous"),
        ("safe", "not safe"),
        ("fast", "not fast"),
        ("steep corridor", "not steep corridor"),
        ("very dangerous", "slightly dangerous"),
        ("extremely fast", "barely moving"),
    ]

    for pos, neg in pairs:
        pos_words = grounder._tokenize(pos)
        neg_words = grounder._tokenize(neg)

        pos_belief = modifier.process(pos_words, grounder)
        neg_belief = modifier.process(neg_words, grounder)

        # Similarity between positive and negated
        pos_norm = np.linalg.norm(pos_belief) + 1e-8
        neg_norm = np.linalg.norm(neg_belief) + 1e-8
        sim = float(np.dot(pos_belief, neg_belief) / (pos_norm * neg_norm))

        print(f"\n  '{pos}' vs '{neg}'")
        print(f"    Similarity: {sim:+.3f}")
        if sim < -0.3:
            print(f"    OPPOSITE (correct)")
        elif sim < 0.3:
            print(f"    DIFFERENT (modifier changed meaning)")
        else:
            print(f"    SIMILAR (modifier had little effect)")


def demo_accumulator():
    print("\n" + "=" * 60)
    print("  2. BELIEF ACCUMULATOR")
    print("  'dog bites man' != 'man bites dog'")
    print("=" * 60)

    grounder = build_trained_grounder()
    acc = BeliefAccumulator()

    sentence_pairs = [
        ("push block left", "block push left"),
        ("danger near corridor", "corridor near danger"),
        ("fast heavy ball falls", "falls fast heavy ball"),
        ("not safe steep path", "safe not steep path"),
    ]

    for a, b in sentence_pairs:
        result_a = acc.process_sentence(a, grounder)
        result_b = acc.process_sentence(b, grounder)

        belief_a = result_a['belief']
        belief_b = result_b['belief']

        norm_a = np.linalg.norm(belief_a) + 1e-8
        norm_b = np.linalg.norm(belief_b) + 1e-8
        sim = float(np.dot(belief_a, belief_b) / (norm_a * norm_b))

        print(f"\n  '{a}' vs '{b}'")
        print(f"    Similarity: {sim:.3f}")
        print(f"    A grounded: {result_a['grounded_count']}/{result_a['total_words']}")
        print(f"    B grounded: {result_b['grounded_count']}/{result_b['total_words']}")
        if sim < 0.95:
            print(f"    DIFFERENT meanings (word order matters)")
        else:
            print(f"    Similar meanings (word order didn't matter here)")


def demo_predictive():
    print("\n" + "=" * 60)
    print("  3. PREDICTIVE GROUNDING")
    print("  'comprehension IS simulation'")
    print("=" * 60)

    grounder = build_trained_grounder()
    predictor = PredictiveGrounder(grounder)

    sentences = [
        "push the block and it moves",
        "drop the ball and it falls",
        "hit the wall and it breaks",
        "pull the heavy object left",
        "gravity makes the ball fall down",
        "the corridor is safe and quiet",
        "quantum entanglement causes decoherence",
    ]

    for sent in sentences:
        result = predictor.comprehend_predictive(sent)
        status = "SIMULATED" if result.get('simulated') else "ACCUMULATED"
        understood = "UNDERSTOOD" if result.get('understood') else "NOT UNDERSTOOD"

        print(f"\n  '{sent}'")
        print(f"    [{status}] [{understood}]")
        print(f"    Confidence: {result['confidence']:.0%}")
        if result.get('simulated'):
            print(f"    Prediction match: {result['prediction_match']:.3f}")
            print(f"    Causal chain: {result.get('causal_chain', '')}")
        print(f"    Grounded: {result['grounded_count']}/{result['total_words']}")


def demo_contrastive():
    print("\n" + "=" * 60)
    print("  4. CONTRASTIVE LEARNING")
    print("  'gravity is what friction is NOT'")
    print("=" * 60)

    grounder = build_trained_grounder()
    contrastive = ContrastiveGrounder(grounder)
    contrastive.build_contrasts()

    print("\n  Standard similarity vs Contrastive similarity:")
    pairs = [
        ("gravity", "friction"),
        ("gravity", "falling"),
        ("push", "pull"),
        ("left", "right"),
        ("danger", "safe"),
        ("fast", "slow"),
        ("gravity", "corridor"),
    ]

    print(f"  {'Pair':<25} {'Standard':>10} {'Contrastive':>12}")
    print(f"  {'-'*49}")

    for a, b in pairs:
        std = grounder.similarity(a, b)
        con = contrastive.discriminative_similarity(a, b)
        print(f"  {a+' / '+b:<25} {std:>+10.3f} {con:>+12.3f}")

    print(f"\n  Explanations:")
    for a, b in [("gravity", "friction"), ("push", "pull"), ("left", "right")]:
        explanation = contrastive.explain_difference(a, b)
        print(f"    {explanation}")


def demo_replay():
    print("\n" + "=" * 60)
    print("  5. EPISODIC REPLAY LEARNING")
    print("  'I teach myself by narrating my memories'")
    print("=" * 60)

    grounder = build_trained_grounder()
    learner = EpisodicReplayLearner(grounder)

    vocab_before = grounder.vocab_size

    # Simulate 100 belief states from experience
    rng = np.random.RandomState(42)
    beliefs = []
    das = []
    moods = []

    for i in range(100):
        b = np.zeros(D_BELIEF, dtype=np.float32)
        # Simulate different scenarios
        if i < 25:  # fast movement
            b[59] = 0.8 + rng.randn() * 0.1
            b[34] = -0.3
        elif i < 50:  # dangerous area
            b[34] = 0.8 + rng.randn() * 0.1
            b[36] = 0.5
        elif i < 75:  # exploration
            b[36] = 0.8 + rng.randn() * 0.1
            b[59] = 0.3
        else:  # uphill
            b[0] = 0.7 + rng.randn() * 0.1
            b[34] = 0.2

        beliefs.append(b)
        das.append(float(np.abs(rng.randn() * 0.3)))
        moods.append('')

    print(f"  Vocab before replay: {vocab_before}")
    result = learner.replay_and_learn(beliefs, das, moods,
                                       n_episodes=100)
    print(f"  Vocab after replay: {result['total_vocab']}")

    # Show what was learned
    print(f"\n  Self-narration examples:")
    for i in [0, 25, 50, 75]:
        narration = learner.narrate_belief(beliefs[i])
        print(f"    Episode {i}: '{narration}'")

    # Test: can the system now understand replay-generated phrases?
    test_phrases = [
        "moving fast",
        "dangerous area",
        "exploring something new",
        "going uphill",
        "familiar territory",
    ]

    comprehender = SentenceComprehender(grounder)
    print(f"\n  Comprehension after replay:")
    for phrase in test_phrases:
        result = comprehender.comprehend(phrase)
        conf = result['confidence']
        understood = "OK" if result.get('understood') else "NO"
        print(f"    [{understood}] '{phrase}' (conf={conf:.0%})")


def demo_all_improvements():
    """Show the full pipeline improvement."""
    print("\n" + "=" * 60)
    print("  FULL PIPELINE COMPARISON")
    print("  Old (pattern matching) vs New (predictive + modifiers)")
    print("=" * 60)

    grounder = build_trained_grounder()

    # Old method
    old = SentenceComprehender(grounder)
    # New method
    new_acc = BeliefAccumulator()
    new_pred = PredictiveGrounder(grounder)

    test_sentences = [
        "not dangerous safe corridor",
        "push the heavy block left",
        "very fast ball falls down",
        "if steep then move slowly",
        "gravity makes objects fall",
        "the corridor is not steep",
        "extremely dangerous dark edge",
    ]

    print(f"\n  {'Sentence':<35} {'Old':>5} {'New':>5} {'Simulated':>10}")
    print(f"  {'-'*57}")

    for sent in test_sentences:
        old_result = old.comprehend(sent)
        new_result = new_pred.comprehend_predictive(sent)

        old_conf = old_result['confidence']
        new_conf = new_result['confidence']
        simulated = "Yes" if new_result.get('simulated') else "No"

        short = sent[:33] + '..' if len(sent) > 35 else sent
        print(f"  {short:<35} {old_conf:>4.0%} {new_conf:>4.0%} {simulated:>10}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", default="all",
                    choices=["negation", "accumulator", "predictive",
                             "contrastive", "replay", "comparison", "all"])
    args = ap.parse_args()

    print("=" * 60)
    print("  NeMo-WM Language v2: Predictive Comprehension")
    print("  From pattern matching to world model simulation")
    print("=" * 60)

    demos = {
        "negation": demo_negation,
        "accumulator": demo_accumulator,
        "predictive": demo_predictive,
        "contrastive": demo_contrastive,
        "replay": demo_replay,
        "comparison": demo_all_improvements,
    }

    if args.demo == "all":
        for name, fn in demos.items():
            fn()
    else:
        demos[args.demo]()

    print(f"\n{'=' * 60}")
    print(f"  All upgrades complete. No LLM. No parser.")
    print(f"  Comprehension = world model simulation.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
