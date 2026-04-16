"""
word_grounder.py — Grounded Language Without LLM
===================================================
Words mean what the robot experienced when it heard them.

The system learns language like a toddler:
  1. Experience something (see, feel, discover)
  2. Hear a word at the same time
  3. Bind word → belief state
  4. Repeat across many experiences
  5. Word meaning = prototype belief pattern

No LLM. No pretrained embeddings. No parser.
No separation of language from perception.

Components:
  1. WordGrounder — core word-to-belief binding
  2. SentenceComprehender — compose word beliefs for understanding
  3. VideoLanguageLearner — learn from narrated video
  4. InstructionFollower — convert NL goals to belief targets
  5. QuestionAnswerer — answer questions about experiences

Usage:
    from word_grounder import WordGrounder, VideoLanguageLearner

    grounder = WordGrounder()
    grounder.hear("gravity", belief_state, da=0.8, mood="Curious")
    meaning = grounder.understand("gravity")
    episodes = grounder.comprehend("the ball falls due to gravity")

Author: John Taylor
"""

import time
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict

import numpy as np
import torch

D_BELIEF = 64


# ──────────────────────────────────────────────────────────────────────────────
# 1. Word Memory — what the system experienced when it heard each word
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class WordExperience:
    """A single experience associated with a word."""
    belief: np.ndarray           # (D_BELIEF,) belief state when word was heard
    da: float = 0.0              # dopamine — how surprising
    mood: str = ""               # mood state name
    neuromod: np.ndarray = None  # (5,) [DA, ACh, CRT, NE, 5HT]
    physics: str = ""            # active physics discovery
    action: np.ndarray = None    # (2,) action being taken
    context: str = ""            # what else was happening
    timestamp: float = field(default_factory=time.time)
    source: str = ""             # "video", "narration", "textbook", "experience"


class WordGrounder:
    """
    Bind words to belief states via co-occurrence.

    Like a toddler: hear "hot" while touching stove →
    "hot" = the belief pattern of burning + high CRT + DA spike.

    Word meaning is NOT a definition.
    Word meaning IS the prototype sensorimotor pattern.

    Includes morphological stemming so "falls" → "falling",
    "moves" → "move", etc. Function words (the, is, from, that)
    are grounded as near-zero beliefs so they don't tank comprehension.
    """

    # Suffix rules: (suffix_to_strip, replacement_to_add)
    SUFFIX_RULES = [
        ("ing", ""),
        ("ing", "e"),
        ("s", ""),
        ("es", ""),
        ("es", "e"),
        ("ed", ""),
        ("ed", "e"),
        ("ly", ""),
        ("tion", ""),
        ("ment", ""),
    ]

    # Explicit irregular mappings
    IRREGULAR_STEMS = {
        "falls": "falling", "fell": "falling",
        "moves": "move", "moved": "move",
        "causes": "cause", "caused": "cause",
        "pulls": "pull", "pulled": "pull",
        "pushes": "push", "pushed": "push",
        "drops": "drop", "dropped": "drop",
        "hits": "hit",
        "effects": "effect",
        "results": "result",
        "patterns": "pattern",
        "examples": "example",
        "closed": "close", "closes": "close",
        "repeated": "repeat", "repeats": "repeat",
        "emerges": "emerge", "emerged": "emerge",
        "carefully": "careful",
        "slowly": "slow",
        "quickly": "quick",
        "breaks": "break",
        "bounces": "bounce",
        "slides": "sliding",
        "rolls": "roll",
        "spins": "spin",
        "stops": "stop",
        "changes": "change",
        "appears": "appear",
        "disappears": "disappear",
    }

    # Function words — semantically light, grounded as near-zero beliefs
    FUNCTION_WORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be",
        "to", "of", "in", "on", "at", "by", "for", "with",
        "from", "that", "this", "it", "its", "and", "or",
        "but", "not", "no", "do", "does", "did", "has", "have",
        "had", "will", "would", "can", "could", "may", "might",
        "shall", "should", "due", "around", "beyond", "about",
        "into", "onto", "upon", "through", "across", "along",
    }

    def __init__(self, d_belief: int = D_BELIEF):
        self.d_belief = d_belief
        self.vocabulary: Dict[str, List[WordExperience]] = defaultdict(list)
        self._prototype_cache: Dict[str, np.ndarray] = {}
        self._cache_dirty: set = set()
        self._total_hearings = 0
        self._stem_cache: Dict[str, str] = {}
        self._function_words_added = False

    def hear(self, word: str, belief: np.ndarray,
             da: float = 0.0, mood: str = "",
             neuromod: np.ndarray = None,
             physics: str = "", action: np.ndarray = None,
             context: str = "", source: str = "experience"):
        """
        The system hears a word while experiencing a belief state.
        This is the fundamental learning operation — co-occurrence.
        """
        word = word.lower().strip()
        if not word or len(word) < 2:
            return

        exp = WordExperience(
            belief=belief.copy() if isinstance(belief, np.ndarray) else belief.detach().cpu().numpy().copy(),
            da=da,
            mood=mood,
            neuromod=neuromod.copy() if neuromod is not None else None,
            physics=physics,
            action=action.copy() if action is not None else None,
            context=context,
            source=source,
        )
        self.vocabulary[word].append(exp)
        self._cache_dirty.add(word)
        self._total_hearings += 1

    def hear_sentence(self, sentence: str, belief: np.ndarray,
                       **kwargs):
        """Hear all words in a sentence at the same belief state."""
        # Simple tokenization — no parser needed
        words = self._tokenize(sentence)
        for word in words:
            self.hear(word, belief, **kwargs)

    def _tokenize(self, text: str) -> List[str]:
        """Ultra-simple tokenization. No parser, no NLP library."""
        # Ensure function words are grounded
        self._ensure_function_words()
        # Remove punctuation, split on spaces
        clean = ''.join(c if c.isalnum() or c == ' ' else ' ' for c in text.lower())
        words = [w for w in clean.split() if len(w) >= 2]
        # Minimal stopwords — only ultra-grammatical words with no semantic content
        stops = {'is', 'at', 'be', 'as', 'was', 'are'}
        return [w for w in words if w not in stops]

    def _ensure_function_words(self):
        """Add function words as near-zero beliefs on first use."""
        if self._function_words_added:
            return
        rng = np.random.RandomState(99)
        for fw in self.FUNCTION_WORDS:
            if fw not in self.vocabulary:
                belief = rng.randn(self.d_belief).astype(np.float32) * 0.01
                self.hear(fw, belief, da=0.01, mood="Calm-Relaxed",
                          source="function_word")
        self._function_words_added = True

    def _stem(self, word: str) -> Optional[str]:
        """
        Try to find a known stem for an unknown word.
        Uses irregular mappings first, then suffix stripping.
        Returns the stem if found in vocabulary, else None.
        """
        # Check cache
        if word in self._stem_cache:
            return self._stem_cache[word]

        # Irregular mapping
        if word in self.IRREGULAR_STEMS:
            stem = self.IRREGULAR_STEMS[word]
            if stem in self.vocabulary:
                self._stem_cache[word] = stem
                return stem

        # Suffix stripping
        for suffix, replacement in self.SUFFIX_RULES:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                stem = word[:-len(suffix)] + replacement
                if stem in self.vocabulary:
                    self._stem_cache[word] = stem
                    return stem

        return None

    def understand(self, word: str) -> Optional[np.ndarray]:
        """
        What does this word mean?
        Returns the prototype belief pattern — the centroid
        of all experiences associated with this word.

        Falls back to stemming: "falls" → "falling", "moves" → "move".
        """
        word = word.lower().strip()

        # Direct match
        if word in self.vocabulary:
            if word in self._cache_dirty:
                beliefs = np.stack([exp.belief for exp in self.vocabulary[word]])
                self._prototype_cache[word] = beliefs.mean(axis=0)
                self._cache_dirty.discard(word)
            return self._prototype_cache.get(word)

        # Stemming fallback
        stem = self._stem(word)
        if stem is not None and stem in self.vocabulary:
            if stem in self._cache_dirty:
                beliefs = np.stack([exp.belief for exp in self.vocabulary[stem]])
                self._prototype_cache[stem] = beliefs.mean(axis=0)
                self._cache_dirty.discard(stem)
            return self._prototype_cache.get(stem)

        return None

    def understand_rich(self, word: str) -> Optional[Dict]:
        """
        Rich understanding: not just the belief prototype but
        the full experiential meaning — mood, DA, physics, action.
        """
        word = word.lower().strip()
        if word not in self.vocabulary:
            return None

        exps = self.vocabulary[word]
        beliefs = np.stack([e.belief for e in exps])
        das = [e.da for e in exps]
        moods = [e.mood for e in exps if e.mood]
        physics_tags = [e.physics for e in exps if e.physics]
        sources = [e.source for e in exps]

        # Most common mood when this word was heard
        mood_counts = defaultdict(int)
        for m in moods:
            mood_counts[m] += 1
        top_mood = max(mood_counts, key=mood_counts.get) if mood_counts else ""

        return {
            'word': word,
            'prototype': beliefs.mean(axis=0),
            'variance': beliefs.var(axis=0),
            'n_experiences': len(exps),
            'mean_da': np.mean(das),
            'typical_mood': top_mood,
            'physics_associations': list(set(physics_tags)),
            'sources': list(set(sources)),
            'confidence': min(1.0, len(exps) / 20.0),
        }

    def similarity(self, word_a: str, word_b: str) -> float:
        """How similar are two words in belief space?"""
        a = self.understand(word_a)
        b = self.understand(word_b)
        if a is None or b is None:
            return 0.0
        norm_a = np.linalg.norm(a) + 1e-8
        norm_b = np.linalg.norm(b) + 1e-8
        return float(np.dot(a, b) / (norm_a * norm_b))

    def nearest_words(self, belief: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """
        What words describe this belief state?
        The inverse of understanding — from experience to language.
        """
        if not self.vocabulary:
            return []

        scored = []
        b_norm = belief / (np.linalg.norm(belief) + 1e-8)

        for word in self.vocabulary:
            proto = self.understand(word)
            if proto is not None:
                p_norm = proto / (np.linalg.norm(proto) + 1e-8)
                sim = float(np.dot(b_norm, p_norm))
                scored.append((word, sim))

        scored.sort(key=lambda x: -x[1])
        return scored[:k]

    def describe(self, belief: np.ndarray) -> str:
        """Generate a natural language description from belief state."""
        words = self.nearest_words(belief, k=5)
        if not words:
            return "Unknown situation."
        relevant = [w for w, s in words if s > 0.3]
        if not relevant:
            relevant = [words[0][0]]
        return "Situation involves: " + ", ".join(relevant)

    @property
    def vocab_size(self) -> int:
        return len(self.vocabulary)

    def stats(self) -> Dict:
        word_counts = {w: len(exps) for w, exps in self.vocabulary.items()}
        top_words = sorted(word_counts.items(), key=lambda x: -x[1])[:10]
        return {
            'vocab_size': self.vocab_size,
            'total_hearings': self._total_hearings,
            'top_words': top_words,
            'avg_experiences_per_word': (
                np.mean(list(word_counts.values())) if word_counts else 0),
        }


# ──────────────────────────────────────────────────────────────────────────────
# 2. Sentence Comprehension — compose word beliefs
# ──────────────────────────────────────────────────────────────────────────────

class SentenceComprehender:
    """
    Understand sentences by composing word beliefs.

    "the steep corridor near building 3"
      = compose(steep_belief, corridor_belief, building_belief)
      = weighted sum in belief space
      → retrieve matching episodes

    No parser. No grammar. Comprehension IS pattern completion.
    """

    def __init__(self, grounder: WordGrounder,
                 episodic_buffer=None):
        self.grounder = grounder
        self.episodic_buffer = episodic_buffer

    def comprehend(self, sentence: str) -> Dict:
        """
        Understand a sentence by composing word beliefs.

        Returns:
          composed_belief: the combined belief pattern
          word_meanings: individual word contributions
          matching_episodes: retrieved similar experiences
          confidence: how well-grounded the sentence is
        """
        words = self.grounder._tokenize(sentence)
        word_meanings = {}
        grounded_beliefs = []
        ungrounded_words = []

        for word in words:
            meaning = self.grounder.understand(word)
            if meaning is not None:
                word_meanings[word] = meaning
                grounded_beliefs.append(meaning)
            else:
                ungrounded_words.append(word)

        if not grounded_beliefs:
            return {
                'composed_belief': np.zeros(self.grounder.d_belief),
                'word_meanings': {},
                'grounded_words': [],
                'ungrounded': words,
                'confidence': 0.0,
                'understood': False,
            }

        # Compose: weighted sum of word beliefs
        composed = np.mean(grounded_beliefs, axis=0)
        composed_norm = np.linalg.norm(composed)
        if composed_norm > 1e-8:
            composed = composed / composed_norm

        # Confidence: fraction of words grounded
        confidence = len(grounded_beliefs) / max(len(words), 1)

        result = {
            'composed_belief': composed,
            'word_meanings': {w: m.tolist() for w, m in
                              list(word_meanings.items())[:5]},
            'grounded_words': list(word_meanings.keys()),
            'ungrounded': ungrounded_words,
            'confidence': confidence,
            'understood': confidence > 0.3,
        }

        # Retrieve matching episodes if buffer available
        if self.episodic_buffer is not None:
            try:
                episodes = self.episodic_buffer.retrieve(composed, k=5)
                result['matching_episodes'] = len(episodes)
            except Exception:
                pass

        return result

    def answer(self, question: str) -> str:
        """
        Answer a question by comprehending it and describing
        the retrieved belief pattern.
        """
        result = self.comprehend(question)
        if not result['understood']:
            unknown = ', '.join(result['ungrounded'][:3])
            return f"I don't understand: {unknown}. I haven't experienced those concepts."

        description = self.grounder.describe(result['composed_belief'])
        confidence = result['confidence']

        response = description
        if result.get('matching_episodes', 0) > 0:
            response += f" I've experienced something similar {result['matching_episodes']} times before."
        if confidence < 0.5:
            response += " (Low confidence — some words are unfamiliar.)"

        return response


# ──────────────────────────────────────────────────────────────────────────────
# 3. Video Language Learner — learn from narrated video
# ──────────────────────────────────────────────────────────────────────────────

class VideoLanguageLearner:
    """
    Learn word meanings from video with subtitles/narration.

    The pipeline:
      1. Process video frame through CNN → belief state
      2. Extract words from subtitle/narration at that timestamp
      3. Compute neuromodulatory response (DA, mood, etc.)
      4. Bind words to (belief, neuromod, physics) tuple
      5. Repeat for entire video

    This is how children learn: see + hear simultaneously.
    """

    def __init__(self, grounder: WordGrounder,
                 visual_encoder=None,
                 physics_agent=None,
                 mood_detector=None):
        self.grounder = grounder
        self.visual_encoder = visual_encoder
        self.physics_agent = physics_agent
        self.mood_detector = mood_detector
        self._videos_processed = 0
        self._words_learned = 0

    def learn_from_subtitled_video(self, frames: np.ndarray,
                                     subtitles: List[Tuple[int, str]],
                                     window: int = 5,
                                     verbose: bool = True) -> Dict:
        """
        Learn from video with timestamped subtitles.

        Args:
            frames: (N, H, W, 3) uint8 video frames
            subtitles: [(frame_idx, "text"), ...] timestamped text
            window: bind words to frames within ±window of subtitle

        Returns:
            dict with learning stats
        """
        n_frames = len(frames)
        new_words = 0
        new_hearings = 0

        if verbose:
            print(f"  Processing {n_frames} frames, "
                  f"{len(subtitles)} subtitle segments")

        # Build subtitle index: which words are active at each frame
        frame_words = defaultdict(list)
        for frame_idx, text in subtitles:
            words = self.grounder._tokenize(text)
            for offset in range(-window, window + 1):
                t = frame_idx + offset
                if 0 <= t < n_frames:
                    frame_words[t].extend(words)

        # Process frames
        prev_belief = None
        for t in range(n_frames):
            # Encode frame
            if self.visual_encoder is not None:
                frame_tensor = torch.from_numpy(
                    frames[t].astype(np.float32) / 255.0
                ).permute(2, 0, 1).unsqueeze(0)
                with torch.no_grad():
                    belief = self.visual_encoder(frame_tensor).squeeze().numpy()
            else:
                # Fallback: use raw pixel statistics as belief
                frame = frames[t].astype(np.float32) / 255.0
                belief = np.concatenate([
                    frame.mean(axis=(0, 1)),   # mean color (3,)
                    frame.std(axis=(0, 1)),    # color variance (3,)
                    np.array([frame.mean()]),   # brightness (1,)
                ]).astype(np.float32)
                # Pad to D_BELIEF
                belief = np.pad(belief, (0, max(0, D_BELIEF - len(belief))))[:D_BELIEF]

            # Compute DA (surprise from frame change)
            if prev_belief is not None:
                da = float(np.linalg.norm(belief - prev_belief))
            else:
                da = 0.5

            # Detect mood
            mood = ""
            if self.mood_detector is not None:
                neuromod = np.array([da, 0.5, 0.1, 0.3, 0.5])
                mood_result = self.mood_detector(
                    torch.from_numpy(neuromod).float())
                mood = mood_result.get('mood_name', '')

            # Detect physics
            physics = ""
            # (Would integrate with physics_discovery_agent here)

            # Bind words at this frame to belief state
            if t in frame_words:
                for word in frame_words[t]:
                    was_new = word not in self.grounder.vocabulary
                    self.grounder.hear(
                        word, belief, da=da, mood=mood,
                        physics=physics, source="video",
                        context=f"frame_{t}")
                    new_hearings += 1
                    if was_new:
                        new_words += 1

            prev_belief = belief

        self._videos_processed += 1
        self._words_learned += new_words

        result = {
            'frames_processed': n_frames,
            'subtitles_processed': len(subtitles),
            'new_words': new_words,
            'new_hearings': new_hearings,
            'vocab_size': self.grounder.vocab_size,
        }

        if verbose:
            print(f"  Learned {new_words} new words, "
                  f"{new_hearings} word-experience bindings")
            print(f"  Total vocabulary: {self.grounder.vocab_size} words")

        return result

    def learn_from_physics_demo(self, frames: np.ndarray,
                                  narration: List[Tuple[int, str]],
                                  verbose: bool = True) -> Dict:
        """
        Learn from a physics demonstration video.
        Specialised for binding physics concepts to observations.
        """
        return self.learn_from_subtitled_video(
            frames, narration, window=10, verbose=verbose)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Instruction Follower — NL goals to belief targets
# ──────────────────────────────────────────────────────────────────────────────

class InstructionFollower:
    """
    Convert natural language instructions to belief-space goals.

    "Go to where it was steep" →
      1. understand("steep") → belief prototype
      2. retrieve episodes matching steep-belief
      3. extract goal position from matched episodes
      4. pass to flow policy as target

    "Be careful near the edge" →
      1. understand("careful") → high-CRT belief pattern
      2. understand("edge") → belief pattern near boundaries
      3. set CRT bias to match "careful" neuromod pattern
      4. plan toward "edge" beliefs with caution
    """

    def __init__(self, grounder: WordGrounder,
                 comprehender: SentenceComprehender):
        self.grounder = grounder
        self.comprehender = comprehender

    def parse_instruction(self, instruction: str) -> Dict:
        """
        Parse instruction into belief-space target + neuromod adjustments.
        No grammar. Just word-to-belief mapping.
        """
        result = self.comprehender.comprehend(instruction)

        # Detect instruction type from grounded words
        action_words = {'go', 'move', 'push', 'turn', 'stop',
                        'explore', 'find', 'avoid', 'approach'}
        emotion_words = {'careful', 'fast', 'slow', 'cautious',
                         'brave', 'gentle', 'aggressive'}
        query_words = {'where', 'what', 'why', 'how', 'when'}

        grounded = set(result['grounded_words'])

        instruction_type = 'unknown'
        if grounded & action_words:
            instruction_type = 'action'
        elif grounded & emotion_words:
            instruction_type = 'modulation'
        elif grounded & query_words:
            instruction_type = 'query'

        # Extract neuromodulatory adjustments from emotion words
        neuromod_adjustments = {}
        for word in grounded & emotion_words:
            rich = self.grounder.understand_rich(word)
            if rich and rich.get('typical_mood'):
                neuromod_adjustments[word] = rich['typical_mood']

        return {
            'type': instruction_type,
            'target_belief': result['composed_belief'],
            'confidence': result['confidence'],
            'neuromod_adjustments': neuromod_adjustments,
            'grounded_words': result['grounded_words'],
            'ungrounded': result['ungrounded'],
        }

    def follow(self, instruction: str, verbose: bool = True) -> Dict:
        """Follow an instruction and return the plan."""
        parsed = self.parse_instruction(instruction)

        if verbose:
            print(f"  Instruction: '{instruction}'")
            print(f"  Type: {parsed['type']}")
            print(f"  Grounded: {parsed['grounded_words']}")
            if parsed['ungrounded']:
                print(f"  Unknown words: {parsed['ungrounded']}")
            print(f"  Confidence: {parsed['confidence']:.1%}")
            if parsed['neuromod_adjustments']:
                print(f"  Mood adjustments: {parsed['neuromod_adjustments']}")

        return parsed


# ──────────────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────────────

def demo():
    print("=" * 65)
    print("  Grounded Language Without LLM")
    print("  'Words mean what I experienced when I heard them'")
    print("=" * 65)

    grounder = WordGrounder(d_belief=D_BELIEF)
    rng = np.random.RandomState(42)

    # ── Stage 1: Learn words from simulated experiences ──
    print("\n── Stage 1: Learning Words from Experience ──")

    # Simulate experiences with words
    experiences = [
        # (word, belief_pattern, da, mood, physics)
        ("gravity", "downward", 0.8, "Curious-Alert", "Fy=-9.81"),
        ("gravity", "downward", 0.6, "Curious-Confident", "Fy=-9.81"),
        ("gravity", "downward", 0.3, "Calm-Confident", "Fy=-9.81"),
        ("falling", "downward", 0.9, "Curious-Alert", "Fy=-9.81"),
        ("falling", "downward", 0.7, "Alert-Cautious", ""),
        ("friction", "opposing", 0.7, "Curious-Uncertain", "Fx=-mu*v"),
        ("friction", "opposing", 0.5, "Calm-Confident", "Fx=-mu*v"),
        ("sliding", "opposing", 0.6, "Curious-Alert", ""),
        ("corridor", "straight", 0.1, "Calm-Relaxed", ""),
        ("corridor", "straight", 0.1, "Familiar-Relaxed", ""),
        ("corridor", "straight", 0.2, "Calm-Confident", ""),
        ("steep", "uphill", 0.4, "Alert-Cautious", ""),
        ("steep", "uphill", 0.5, "Stressed-Alert", ""),
        ("danger", "threat", 0.9, "Stressed-Alert", ""),
        ("danger", "threat", 0.95, "Stressed-Reactive", ""),
        ("safe", "calm_area", 0.05, "Calm-Relaxed", ""),
        ("safe", "calm_area", 0.1, "Familiar-Confident", ""),
        ("magnetic", "attraction", 0.8, "Curious-Alert", "F=k/r^2"),
        ("push", "contact_force", 0.6, "Alert-Confident", ""),
        ("push", "contact_force", 0.5, "Confident-Calm", ""),
        ("turn", "rotation", 0.3, "Calm-Confident", ""),
        ("turn", "rotation", 0.4, "Alert-Cautious", ""),
        ("fast", "high_speed", 0.5, "Alert-Bold", ""),
        ("slow", "low_speed", 0.1, "Calm-Relaxed", ""),
        ("careful", "cautious", 0.2, "Cautious-Alert", ""),
    ]

    # Generate belief patterns for each concept
    concept_beliefs = {
        "downward": rng.randn(D_BELIEF) * 0.3 + np.array([0, -1] + [0]*(D_BELIEF-2)),
        "opposing": rng.randn(D_BELIEF) * 0.3 + np.array([-1, 0] + [0]*(D_BELIEF-2)),
        "straight": rng.randn(D_BELIEF) * 0.2 + np.array([1, 0] + [0]*(D_BELIEF-2)),
        "uphill": rng.randn(D_BELIEF) * 0.3 + np.array([0.5, 1] + [0]*(D_BELIEF-2)),
        "threat": rng.randn(D_BELIEF) * 0.4 + np.array([0, 0, 1] + [0]*(D_BELIEF-3)),
        "calm_area": rng.randn(D_BELIEF) * 0.1,
        "attraction": rng.randn(D_BELIEF) * 0.3 + np.array([0.7, 0.7] + [0]*(D_BELIEF-2)),
        "contact_force": rng.randn(D_BELIEF) * 0.3 + np.array([1, 0, 0.5] + [0]*(D_BELIEF-3)),
        "rotation": rng.randn(D_BELIEF) * 0.3 + np.array([0, 0, 0, 1] + [0]*(D_BELIEF-4)),
        "high_speed": rng.randn(D_BELIEF) * 0.3 + np.array([2, 0] + [0]*(D_BELIEF-2)),
        "low_speed": rng.randn(D_BELIEF) * 0.1 + np.array([0.1, 0] + [0]*(D_BELIEF-2)),
        "cautious": rng.randn(D_BELIEF) * 0.2 + np.array([0, 0, 0.8] + [0]*(D_BELIEF-3)),
    }

    for word, concept, da, mood, physics in experiences:
        belief = concept_beliefs[concept] + rng.randn(D_BELIEF) * 0.1
        grounder.hear(word, belief, da=da, mood=mood,
                       physics=physics, source="simulation")

    stats = grounder.stats()
    print(f"  Vocabulary: {stats['vocab_size']} words")
    print(f"  Total hearings: {stats['total_hearings']}")
    print(f"  Top words: {stats['top_words'][:5]}")

    # ── Stage 2: Understanding words ──
    print("\n── Stage 2: Word Understanding ──")

    test_words = ["gravity", "friction", "corridor", "danger",
                   "magnetic", "push", "careful"]
    for word in test_words:
        rich = grounder.understand_rich(word)
        if rich:
            print(f"  '{word}': {rich['n_experiences']} experiences, "
                  f"DA={rich['mean_da']:.2f}, "
                  f"mood='{rich['typical_mood']}', "
                  f"physics={rich['physics_associations']}, "
                  f"confidence={rich['confidence']:.2f}")

    # ── Stage 3: Word similarity ──
    print("\n── Stage 3: Word Similarity (in belief space) ──")

    pairs = [
        ("gravity", "falling"),
        ("gravity", "friction"),
        ("gravity", "corridor"),
        ("danger", "careful"),
        ("danger", "safe"),
        ("push", "friction"),
        ("steep", "danger"),
        ("fast", "slow"),
    ]

    for a, b in pairs:
        sim = grounder.similarity(a, b)
        print(f"  sim({a}, {b}) = {sim:.3f}")

    # ── Stage 4: Sentence comprehension ──
    print("\n── Stage 4: Sentence Comprehension ──")

    comprehender = SentenceComprehender(grounder)

    sentences = [
        "the ball is falling due to gravity",
        "there is danger on the steep corridor",
        "push the block carefully",
        "the magnetic force pulls objects",
        "quantum entanglement superposition",  # unknown words
    ]

    for sent in sentences:
        result = comprehender.comprehend(sent)
        status = "UNDERSTOOD" if result['understood'] else "NOT UNDERSTOOD"
        print(f"\n  '{sent}'")
        print(f"    {status} (confidence={result['confidence']:.1%})")
        print(f"    Grounded: {result['grounded_words']}")
        if result['ungrounded']:
            print(f"    Unknown: {result['ungrounded']}")

    # ── Stage 5: Answering questions ──
    print("\n── Stage 5: Answering Questions ──")

    questions = [
        "what is gravity",
        "where is danger",
        "what happens when falling",
        "explain quantum mechanics",
    ]

    for q in questions:
        answer = comprehender.answer(q)
        print(f"  Q: '{q}'")
        print(f"  A: {answer}")

    # ── Stage 6: Instruction following ──
    print("\n── Stage 6: Instruction Following ──")

    follower = InstructionFollower(grounder, comprehender)

    instructions = [
        "go to the steep corridor",
        "push the block carefully",
        "be careful near danger",
        "explore fast",
    ]

    for inst in instructions:
        follower.follow(inst, verbose=True)
        print()

    # ── Stage 7: Describe a belief state ──
    print("── Stage 7: Belief → Language (the reverse) ──")

    test_beliefs = [
        ("Downward force", concept_beliefs["downward"]),
        ("Safe calm area", concept_beliefs["calm_area"]),
        ("High speed", concept_beliefs["high_speed"]),
        ("Threat detected", concept_beliefs["threat"]),
    ]

    for label, belief in test_beliefs:
        description = grounder.describe(belief)
        nearest = grounder.nearest_words(belief, k=3)
        print(f"  [{label}] → {description}")
        print(f"    Nearest: {[(w, f'{s:.3f}') for w, s in nearest]}")

    # ── Stage 8: Learn from simulated narrated video ──
    print("\n── Stage 8: Learning from Video ──")

    # Simulate 100 frames of a physics video
    video_frames = rng.randint(0, 255, (100, 16, 16, 3), dtype=np.uint8)

    # Subtitles at specific frames
    subtitles = [
        (10, "the ball begins to fall"),
        (25, "gravity pulls it downward"),
        (40, "it accelerates faster and faster"),
        (55, "friction slows the sliding block"),
        (70, "the magnetic force attracts"),
        (85, "the ball bounces and stops"),
    ]

    learner = VideoLanguageLearner(grounder)
    result = learner.learn_from_subtitled_video(
        video_frames, subtitles, window=5, verbose=True)

    # ── Final stats ──
    print(f"\n── Final Vocabulary ──")
    final_stats = grounder.stats()
    print(f"  Total words: {final_stats['vocab_size']}")
    print(f"  Total hearings: {final_stats['total_hearings']}")
    print(f"  Avg experiences/word: {final_stats['avg_experiences_per_word']:.1f}")
    print(f"  Top 10 words:")
    for word, count in final_stats['top_words']:
        rich = grounder.understand_rich(word)
        confidence = rich['confidence'] if rich else 0
        print(f"    '{word}': {count} experiences, "
              f"confidence={confidence:.1%}")

    print(f"\n{'='*65}")
    print(f"  Language learned from experience. No LLM. No parser.")
    print(f"  No pretrained embeddings. No separation from perception.")
    print(f"{'='*65}")


if __name__ == "__main__":
    demo()
