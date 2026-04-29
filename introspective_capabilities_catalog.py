"""
introspective_capabilities_catalog.py
======================================
Single source of truth for NeMo-WM's introspective capabilities.

Usage:
    python introspective_capabilities_catalog.py
    python introspective_capabilities_catalog.py --output FILENAME.md
    python introspective_capabilities_catalog.py --json
    python introspective_capabilities_catalog.py --check  (validates entries)

Categorization
--------------
Each capability has TWO independent properties:

  GROUNDED:   has biological/cognitive science citation linking it to
              a known mechanism in the human brain or animal cognition

  TESTED:     has executable test code that runs and asserts a result
              (more than just "function exists" — actual measurement)

This gives 4 quadrants:

  GROUNDED + TESTED   = "production-grade" — the strongest claim
  GROUNDED            = paper-grade theory, implementation pending
  TESTED              = working code, citation/grounding pending
  (neither)           = designed but not built; future work

Adding a capability
-------------------
Append a Capability(...) entry below. Re-run to regenerate the catalog.

Author: John Taylor + Claude collaboration
Date: April 29, 2026
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List
from collections import Counter
import argparse
import json


# ===========================================================================
# Data structure
# ===========================================================================

@dataclass
class Capability:
    """One introspective capability of NeMo-WM."""

    cap_id: str                    # canonical ID, e.g. "C01"
    question: str                  # the question this capability answers
    short_name: str                # 1-3 word label
    component: str                 # which class/module implements it
    source_file: str               # primary file path
    mechanism: str                 # 1-2 sentence description
    category: str                  # Perception | Imagination | Decision |
                                   # Language | Meta | Memory | Emotion | Future
    grounded: bool                 # has citation/biology link
    grounding_citation: str        # citation if grounded, else ""
    tested: bool                   # has executable test
    test_location: str             # where test is defined, else ""
    benchmark_id: str              # ID in benchmark_20q.py if any (Q1-Q20)
    paper_id: str                  # ID in paper2 MD if any
    latency_us: Optional[float]    # measured latency in microseconds
    human_parallel: str            # 1-sentence human cognitive analog
    notes: str = ""

    @property
    def status(self) -> str:
        if self.grounded and self.tested:
            return "PRODUCTION"
        elif self.grounded:
            return "GROUNDED_ONLY"
        elif self.tested:
            return "TESTED_ONLY"
        else:
            return "DESIGNED"

    @property
    def status_emoji(self) -> str:
        return {
            "PRODUCTION":   "[GT]",  # Grounded + Tested
            "GROUNDED_ONLY": "[G]",
            "TESTED_ONLY":   "[T]",
            "DESIGNED":      "[D]",
        }[self.status]


# ===========================================================================
# Capability catalog
# ===========================================================================
# Order: by category, then by canonical cap_id.
# Status is derived from grounded + tested booleans.

CAPABILITIES: List[Capability] = [
    # =====================================================================
    # PERCEPTION (Q1-Q4 — confirmed match between MD and benchmark)
    # =====================================================================
    Capability(
        cap_id="C01",
        question="Where am I right now?",
        short_name="Self-localization",
        component="ProprioEncoder",
        source_file="core/proprioceptive_encoder.py",
        mechanism="26K-param contrastive temporal encoder integrates 8s of "
                  "velocity, heading, contact into 64-D belief state.",
        category="Perception",
        grounded=True,
        grounding_citation="Moser et al. (2008); McNaughton (2006); O'Keefe (1971)",
        tested=True,
        test_location="benchmark_20q.py:101 (Q1); eval_recon_auroc.py",
        benchmark_id="Q1",
        paper_id="Q1",
        latency_us=0.14,
        human_parallel="Closed-eye spatial awareness via proprioception.",
    ),
    Capability(
        cap_id="C02",
        question="What does this place look like?",
        short_name="Place recognition",
        component="GPSRetriever",
        source_file="planning/gps_retrieval.py",
        mechanism="Dead-reckoned GPS to nearest real frame from prior "
                  "traversals via FAISS over 215K-frame index.",
        category="Perception",
        grounded=True,
        grounding_citation="O'Keefe & Nadel (1978); Schacter et al. (2012)",
        tested=True,
        test_location="benchmark_20q.py:108 (Q2)",
        benchmark_id="Q2",
        paper_id="Q2",
        latency_us=15.38,
        human_parallel="Returning to a familiar street: 'I've been here'.",
    ),
    Capability(
        cap_id="C03",
        question="Is something wrong here?",
        short_name="Anomaly detection",
        component="CWM multidomain encoder",
        source_file="core/cwm_multidomain.py",
        mechanism="Reconstruction error in latent space flags anomalies "
                  "across 6 production domains.",
        category="Perception",
        grounded=True,
        grounding_citation="Botvinick et al. (2004) — ACC error monitoring",
        tested=True,
        test_location="benchmark_20q.py:116 (Q3); MIMII/MVTec eval scripts",
        benchmark_id="Q3",
        paper_id="Q3",
        latency_us=1.38,
        human_parallel="Walking into a room and sensing 'something is off'.",
    ),
    Capability(
        cap_id="C04",
        question="Where is [text description]?",
        short_name="Semantic search",
        component="SemanticHead + CLIPBridge",
        source_file="language/semantic_head.py",
        mechanism="CLIP text -> bridge -> latent space -> nearest node "
                  "in GeoLatentDB.",
        category="Perception",
        grounded=True,
        grounding_citation="Epstein & Kanwisher (1998) — parahippocampal place area",
        tested=True,
        test_location="benchmark_20q.py:124 (Q4)",
        benchmark_id="Q4",
        paper_id="Q4",
        latency_us=14.36,
        human_parallel="'The coffee shop near the bridge' triggers spatial recall.",
    ),

    # =====================================================================
    # IMAGINATION (Q5-Q8 — confirmed match)
    # =====================================================================
    Capability(
        cap_id="C05",
        question="If I do X, where do I end up?",
        short_name="Forward prediction",
        component="BeliefTransitionModel",
        source_file="core/belief_transition.py",
        mechanism="f(b_t, a_t) -> b_{t+1} with calibrated uncertainty sigma=0.137.",
        category="Imagination",
        grounded=True,
        grounding_citation="Wolpert & Ghahramani (2000)",
        tested=True,
        test_location="benchmark_20q.py:135 (Q5)",
        benchmark_id="Q5",
        paper_id="Q5",
        latency_us=3.58,
        human_parallel="Pre-movement prediction of sensory consequences.",
    ),
    Capability(
        cap_id="C06",
        question="What happens over the next 8 seconds?",
        short_name="Mental simulation",
        component="ImaginationRollout",
        source_file="planning/imagination_rollout.py",
        mechanism="32-step forward simulation, 8 action candidates evaluated.",
        category="Imagination",
        grounded=True,
        grounding_citation="Schacter et al. (2007) — constructive episodic simulation",
        tested=True,
        test_location="benchmark_20q.py:146 (Q6)",
        benchmark_id="Q6",
        paper_id="Q6",
        latency_us=116.60,
        human_parallel="Mentally walking through a crowded room before moving.",
    ),
    Capability(
        cap_id="C07",
        question="How far ahead should I plan?",
        short_name="Adaptive horizon",
        component="ACh-gated planning horizon",
        source_file="core/neuromodulator.py",
        mechanism="T_horizon = max(1, round(32 * ACh_t)). ACh = N_eff/N.",
        category="Imagination",
        grounded=True,
        grounding_citation="Hasselmo (2006) — ACh controls temporal binding",
        tested=True,
        test_location="benchmark_20q.py:153 (Q7)",
        benchmark_id="Q7",
        paper_id="Q7",
        latency_us=None,
        human_parallel="Confident on familiar ground, plan-step-by-step when lost.",
    ),
    Capability(
        cap_id="C08",
        question="Should I trust my prediction or look again?",
        short_name="Anticipate-react gating",
        component="AnticipateReactGate",
        source_file="core/anticipate_react.py",
        mechanism="alpha = g(delta, CRT). High alpha = open-loop, low alpha = "
                  "closed-loop verification.",
        category="Imagination",
        grounded=True,
        grounding_citation="Friston (2010) — free energy principle",
        tested=True,
        test_location="benchmark_20q.py:162 (Q8)",
        benchmark_id="Q8",
        paper_id="Q8",
        latency_us=None,
        human_parallel="Familiar drive: barely look. Construction zone: snap to attention.",
    ),

    # =====================================================================
    # DECISION (Q9-Q11 — confirmed match)
    # =====================================================================
    Capability(
        cap_id="C09",
        question="Which action is best given my current state?",
        short_name="Action selection",
        component="NeuromodulatedValue",
        source_file="core/value_function.py",
        mechanism="Q-value = DA*Q - CRT*U + ACh*H. QM best=0.00789.",
        category="Decision",
        grounded=True,
        grounding_citation="Schultz (1998) — DA encodes RPE; "
                           "Yu & Dayan (2005) — ACh signals expected uncertainty",
        tested=True,
        test_location="benchmark_20q.py:177 (Q9)",
        benchmark_id="Q9",
        paper_id="Q9",
        latency_us=None,
        human_parallel="Choosing route through traffic — value + uncertainty + caution.",
    ),
    Capability(
        cap_id="C10",
        question="Have I been in a similar situation before?",
        short_name="Episodic retrieval",
        component="EpisodicBuffer",
        source_file="core/episodic_buffer.py",
        mechanism="10K |DA|-priority buffer; nearest-neighbor retrieval over past beliefs.",
        category="Decision",
        grounded=True,
        grounding_citation="Tulving (1985); Lisman & Grace (2005) — VTA-hippocampal loop",
        tested=True,
        test_location="benchmark_20q.py:188 (Q10)",
        benchmark_id="Q10",
        paper_id="Q10",
        latency_us=7.05,
        human_parallel="'I've been in this kind of meeting before' triggers retrieval.",
    ),
    Capability(
        cap_id="C11",
        question="Is this a new kind of place?",
        short_name="Schema novelty",
        component="DiVeQSchemaStore",
        source_file="core/diveq_schema.py",
        mechanism="32-codebook discrete-vector quantization; novelty = "
                  "min distance to any prototype.",
        category="Decision",
        grounded=True,
        grounding_citation="Bartlett (1932) — schema theory; "
                           "Norman & Shallice (1986)",
        tested=True,
        test_location="benchmark_20q.py:203 (Q11)",
        benchmark_id="Q11",
        paper_id="Q11",
        latency_us=5.01,
        human_parallel="First time in a new culture: nothing maps to existing categories.",
    ),

    # =====================================================================
    # LANGUAGE (paper2 has Q12-Q15 with these topics; benchmark has different)
    # Treated as separate capabilities — both are real
    # =====================================================================
    Capability(
        cap_id="C12",
        question="Does language help me understand this scene?",
        short_name="Language-vision binding",
        component="PredictiveGrounder",
        source_file="language_v2.py",
        mechanism="Tests whether adding text predictions improves visual reconstruction.",
        category="Language",
        grounded=True,
        grounding_citation="Fedorenko et al. — language-vision dissociation; "
                           "aphasia preserves complex perception",
        tested=False,
        test_location="",
        benchmark_id="",
        paper_id="Q12 (paper2)",
        latency_us=None,
        human_parallel="Hearing 'red barn' before seeing it — does the cue help?",
        notes="Paper2-style framing. Code-side uses different Q12.",
    ),
    Capability(
        cap_id="C13",
        question="What does this word mean in my world?",
        short_name="Word grounding",
        component="GroundedVocabulary",
        source_file="language/word_grounding.py",
        mechanism="380-word vocabulary mapped to belief-space centroids.",
        category="Language",
        grounded=False,
        grounding_citation="",
        tested=True,
        test_location="benchmark_20q.py:219 (Q12-code)",
        benchmark_id="Q12 (code)",
        paper_id="",
        latency_us=0.06,
        human_parallel="'Apple' triggers a sensory complex of red+round+sweet.",
        notes="Code's Q12. Tested but ungrounded vs Lakoff/Johnson body grounding.",
    ),
    Capability(
        cap_id="C14",
        question="Can I understand this sentence?",
        short_name="Compositional comprehension",
        component="SentenceComprehension",
        source_file="language/sentence_comp.py",
        mechanism="Multi-word sequences compose into action+object slots.",
        category="Language",
        grounded=False,
        grounding_citation="",
        tested=True,
        test_location="benchmark_20q.py:231 (Q13-code)",
        benchmark_id="Q13 (code)",
        paper_id="",
        latency_us=4.58,
        human_parallel="'The cat is on the mat' decomposes into known relations.",
        notes="Code's Q13. Could be grounded via Frege/Pinker/Marcus compositionality.",
    ),
    Capability(
        cap_id="C15",
        question="What does this sound look like across modalities?",
        short_name="Audio-visual binding",
        component="AudioBeliefEncoder",
        source_file="audio_encoder.py",
        mechanism="Audio features projected into shared belief space with vision.",
        category="Language",
        grounded=True,
        grounding_citation="McGurk & MacDonald (1976); "
                           "Stein & Meredith (1993) — multisensory integration",
        tested=False,
        test_location="",
        benchmark_id="",
        paper_id="Q15 (paper2)",
        latency_us=None,
        human_parallel="Hearing thunder while seeing lightning binds them as one event.",
    ),

    # =====================================================================
    # META-COGNITION (mood, WM, fatigue, language-help)
    # =====================================================================
    Capability(
        cap_id="C16",
        question="What mood am I in?",
        short_name="Mood self-report",
        component="MoodStates",
        source_file="diveq_novel.py",
        mechanism="16-mood codebook over neuromodulator scalar tuple "
                  "(DA, NE, ACh, 5HT, eCB, Adenosine, Cortisol).",
        category="Meta",
        grounded=True,
        grounding_citation="Russell (1980) — circumplex model of affect",
        tested=True,
        test_location="benchmark_20q.py:246 (Q14-code)",
        benchmark_id="Q14 (code)",
        paper_id="",
        latency_us=None,
        human_parallel="'I feel restless and curious today' — multidimensional self-state.",
    ),
    Capability(
        cap_id="C17",
        question="Does language help my decisions?",
        short_name="Aphasia ablation",
        component="AphasiaAblation",
        source_file="eval/aphasia_ablation.py",
        mechanism="Zero language input; measure performance drop. "
                  "Quantifies language's contribution.",
        category="Meta",
        grounded=True,
        grounding_citation="Fedorenko et al. (2024) — aphasia preserves reasoning",
        tested=True,
        test_location="benchmark_20q.py:255 (Q15-code)",
        benchmark_id="Q15 (code)",
        paper_id="",
        latency_us=None,
        human_parallel="Stroke patients with aphasia: complex reasoning preserved.",
    ),
    Capability(
        cap_id="C18",
        question="How much working memory is available?",
        short_name="WM capacity",
        component="WorkingMemory",
        source_file="diveq_integration.py",
        mechanism="Inverted-U capacity around K=7 items; tracks utilization.",
        category="Meta",
        grounded=True,
        grounding_citation="Miller (1956) — magical number 7; "
                           "Cowan (2001) — chunk-based capacity",
        tested=True,
        test_location="benchmark_20q.py:268 (Q16-code)",
        benchmark_id="Q16 (code)",
        paper_id="Q13 (paper2)",
        latency_us=0.54,
        human_parallel="Phone numbers chunked into 3-3-4 to fit memory span.",
    ),
    Capability(
        cap_id="C19",
        question="Should I explore or stick with what I know?",
        short_name="Explore/exploit",
        component="ExploreExploit",
        source_file="q16_q17_wiring.py",
        mechanism="Tracks confidence in current schema; suggests "
                  "exploration when novelty>threshold.",
        category="Meta",
        grounded=True,
        grounding_citation="Daw et al. (2006) — exploration/exploitation in PFC",
        tested=True,
        test_location="q16_q17_wiring.py selftest",
        benchmark_id="",
        paper_id="Q16 (paper2)",
        latency_us=None,
        human_parallel="Familiar restaurant vs new place — which is the signal?",
    ),
    Capability(
        cap_id="C20",
        question="Am I getting fatigued?",
        short_name="Fatigue monitor",
        component="FatigueMonitor / Adenosine",
        source_file="q16_q17_wiring.py",
        mechanism="Adenosine accumulates with cognitive load; triggers "
                  "recalibration above threshold.",
        category="Meta",
        grounded=True,
        grounding_citation="Porkka-Heiskanen (1999) — adenosine sleep regulation",
        tested=True,
        test_location="benchmark_20q.py:275 (Q17); q16_q17_wiring.py",
        benchmark_id="Q17",
        paper_id="Q17 (paper2)",
        latency_us=None,
        human_parallel="Mid-afternoon mental slowdown — adenosine, not laziness.",
        notes="Q17 is the only number that matches between MD and benchmark.",
    ),

    # =====================================================================
    # FUTURE / COUNTERFACTUAL / PROSPECTIVE (Q18-Q20 in code)
    # =====================================================================
    Capability(
        cap_id="C21",
        question="What if I had done X instead?",
        short_name="Counterfactual reasoning",
        component="CounterfactualEngine",
        source_file="counterfactual_reasoning.py",
        mechanism="Rollback belief, replay alternate action through "
                  "TransitionModel, compute regret = V(actual) - V(counterfactual).",
        category="Future",
        grounded=True,
        grounding_citation="Roese (1997); Camille et al. (2004) — OFC encodes regret",
        tested=True,
        test_location="benchmark_20q.py:289 (Q18); counterfactual_reasoning.py demo",
        benchmark_id="Q18",
        paper_id="",
        latency_us=9.51,
        human_parallel="'If I had taken the other route...' — OFC regret circuit.",
    ),
    Capability(
        cap_id="C22",
        question="What is my high-level plan?",
        short_name="Hierarchical planning",
        component="SchemaCodebook + SchemaGraph",
        source_file="hierarchical_schema_planning.py",
        mechanism="A* over 32-node schema graph; returns sequence of subgoals.",
        category="Future",
        grounded=True,
        grounding_citation="Botvinick (2008); Badre & D'Esposito (2009)",
        tested=True,
        test_location="benchmark_20q.py:302 (Q19)",
        benchmark_id="Q19",
        paper_id="",
        latency_us=8.60,
        human_parallel="'Drive downtown then take highway' — abstracted plan.",
    ),
    Capability(
        cap_id="C23",
        question="What do I need to remember to do later?",
        short_name="Prospective memory",
        component="ProspectiveMemory",
        source_file="cognitive_extensions.py",
        mechanism="Intention store with cue-trigger pairs; scans state for matches.",
        category="Future",
        grounded=True,
        grounding_citation="Einstein & McDaniel (1990); Burgess et al. (2003) — BA10",
        tested=True,
        test_location="benchmark_20q.py:308 (Q20); cognitive_extensions.py tests",
        benchmark_id="Q20",
        paper_id="",
        latency_us=None,
        human_parallel="'Call mom before bed' surfaces at the right cue.",
    ),

    # =====================================================================
    # IMAGINATION / DREAM
    # =====================================================================
    Capability(
        cap_id="C24",
        question="What would I dream about this route?",
        short_name="Dream interpolation",
        component="DreamInterpolation",
        source_file="diveq_novel.py",
        mechanism="Interpolates between two episodic memories in belief space "
                  "to generate novel paths.",
        category="Imagination",
        grounded=True,
        grounding_citation="Stickgold (2005) — REM sleep memory consolidation",
        tested=False,
        test_location="",
        benchmark_id="",
        paper_id="Q14 (paper2)",
        latency_us=None,
        human_parallel="Dreams blend recent experiences into novel sequences.",
    ),
    Capability(
        cap_id="C25",
        question="What story does this dream tell?",
        short_name="Dream narration",
        component="DreamNarrator",
        source_file="cognitive_quickwins.py",
        mechanism="Captions a dream sequence using grounded vocabulary.",
        category="Imagination",
        grounded=False,
        grounding_citation="",
        tested=True,
        test_location="cognitive_quickwins.py demo",
        benchmark_id="",
        paper_id="",
        latency_us=None,
        human_parallel="Waking and saying 'I dreamed I was flying through a forest'.",
        notes="Could be grounded via Hobson (2009) AIM model of dream content.",
    ),

    # =====================================================================
    # MEMORY DYNAMICS
    # =====================================================================
    Capability(
        cap_id="C26",
        question="How emotionally salient was this experience?",
        short_name="Emotional tagging",
        component="EmotionalTagging",
        source_file="cognitive_extensions.py",
        mechanism="High-DA episodes tagged for 3x consolidation priority.",
        category="Emotion",
        grounded=True,
        grounding_citation="McGaugh (2000) — emotional arousal enhances memory",
        tested=True,
        test_location="benchmark_20q.py:352 (integrated)",
        benchmark_id="",
        paper_id="",
        latency_us=None,
        human_parallel="You remember your wedding day; not last Tuesday's lunch.",
    ),
    Capability(
        cap_id="C27",
        question="Should I rewrite this memory?",
        short_name="Reconsolidation",
        component="ReconsolidationEngine",
        source_file="cognitive_extensions.py",
        mechanism="50-step labile window after retrieval; updates can rewrite memory.",
        category="Emotion",
        grounded=True,
        grounding_citation="Nader, Schafe, LeDoux (2000) — reconsolidation",
        tested=True,
        test_location="benchmark_20q.py:357 (integrated)",
        benchmark_id="",
        paper_id="",
        latency_us=None,
        human_parallel="Therapy works partly via memory reconsolidation.",
    ),
    Capability(
        cap_id="C28",
        question="What should I consolidate during sleep?",
        short_name="Sleep consolidation",
        component="SleepConsolidation",
        source_file="eval/introspective_extensions.py",
        mechanism="Idle-triggered replay; high-priority episodes compressed into schemas.",
        category="Emotion",
        grounded=True,
        grounding_citation="Walker & Stickgold (2004) — sleep-dependent consolidation",
        tested=True,
        test_location="introspective_extensions.py selftest",
        benchmark_id="",
        paper_id="",
        latency_us=None,
        human_parallel="A good night's sleep solidifies what you learned that day.",
    ),

    # =====================================================================
    # UNCERTAINTY / SELF-CORRECTION
    # =====================================================================
    Capability(
        cap_id="C29",
        question="How certain am I?",
        short_name="Disagreement-DA uncertainty",
        component="DisagreementDA",
        source_file="eval/introspective_extensions.py",
        mechanism="Compares flow policy action with retrieved past action; "
                  "disagreement -> high DA -> novelty signal.",
        category="Meta",
        grounded=True,
        grounding_citation="Lisman & Grace (2005) — VTA-hippocampal prediction error",
        tested=True,
        test_location="introspective_extensions.py selftest",
        benchmark_id="",
        paper_id="",
        latency_us=None,
        human_parallel="'This doesn't match what I remember' triggers DA + attention.",
    ),
    Capability(
        cap_id="C30",
        question="Which sensor should I trust right now?",
        short_name="Modality routing",
        component="ModalityRouter",
        source_file="eval/introspective_extensions.py",
        mechanism="Per-step routing between proprio, VLM, or both based on "
                  "neuromodulator state.",
        category="Meta",
        grounded=True,
        grounding_citation="Shams & Beierholm (2010) — Bayesian causal inference "
                           "in multisensory integration",
        tested=True,
        test_location="introspective_extensions.py selftest",
        benchmark_id="",
        paper_id="",
        latency_us=None,
        human_parallel="In dim light, trust touch over vision.",
    ),
    Capability(
        cap_id="C31",
        question="What did I get wrong? Should I retrain?",
        short_name="Self-correction",
        component="AnomalyRetrainer",
        source_file="eval/introspective_extensions.py",
        mechanism="Detected failures generate (state, correct_action) pairs "
                  "for online retraining.",
        category="Meta",
        grounded=True,
        grounding_citation="Holroyd & Coles (2002) — error-related negativity",
        tested=True,
        test_location="introspective_extensions.py selftest",
        benchmark_id="",
        paper_id="",
        latency_us=None,
        human_parallel="Mistake at work -> mental note to handle differently next time.",
    ),

    # =====================================================================
    # CURIOSITY / ATTENTION
    # =====================================================================
    Capability(
        cap_id="C32",
        question="What is interesting right now?",
        short_name="Curiosity engine",
        component="CuriosityEngine",
        source_file="nemo_agi3_agent.py",
        mechanism="Information-gain proxy guides exploration; high gain -> high curiosity.",
        category="Decision",
        grounded=True,
        grounding_citation="Schmidhuber (1991); Kidd & Hayden (2015)",
        tested=True,
        test_location="nemo_agi3_agent.py runtime; ARC-AGI-3 deployment",
        benchmark_id="",
        paper_id="",
        latency_us=None,
        human_parallel="A novel object draws attention more than a familiar one.",
    ),
    Capability(
        cap_id="C33",
        question="Have I seen enough of this?",
        short_name="Curiosity decay",
        component="CuriosityDecay",
        source_file="cognitive_quickwins.py",
        mechanism="Per-stimulus repetition counter decays curiosity weight.",
        category="Decision",
        grounded=True,
        grounding_citation="Sokolov (1963) — orienting response habituation",
        tested=True,
        test_location="cognitive_quickwins.py demo",
        benchmark_id="",
        paper_id="",
        latency_us=None,
        human_parallel="The new poster grabs you week 1, fades by week 4.",
    ),

    # =====================================================================
    # SCHEMAS / CATEGORIZATION
    # =====================================================================
    Capability(
        cap_id="C34",
        question="What category does this fit?",
        short_name="Schema naming",
        component="SchemaNamer",
        source_file="cognitive_quickwins.py",
        mechanism="Assigns the closest schema label from grounded vocabulary.",
        category="Language",
        grounded=False,
        grounding_citation="",
        tested=True,
        test_location="cognitive_quickwins.py demo",
        benchmark_id="",
        paper_id="",
        latency_us=None,
        human_parallel="Looking at a new bird and thinking 'a kind of sparrow'.",
        notes="Could ground via Rosch (1973) prototype theory.",
    ),

    # =====================================================================
    # PROCEDURAL MEMORY / SKILL
    # =====================================================================
    Capability(
        cap_id="C35",
        question="How do I do this skill?",
        short_name="Procedural memory",
        component="ProceduralMemory",
        source_file="agi_gap_closure.py",
        mechanism="Stores skill graphs (state -> action sequences) for reuse.",
        category="Memory",
        grounded=True,
        grounding_citation="Squire (1992) — declarative vs procedural memory; "
                           "basal ganglia in skill learning",
        tested=True,
        test_location="agi_gap_closure.py demo",
        benchmark_id="",
        paper_id="",
        latency_us=None,
        human_parallel="Riding a bike — automatic, doesn't require recall.",
    ),

    # =====================================================================
    # FUTURE / DESIGNED CAPABILITIES (not yet built or partially built)
    # =====================================================================
    Capability(
        cap_id="C36",
        question="What time is it (subjective)?",
        short_name="Subjective timing",
        component="(planned)",
        source_file="(future)",
        mechanism="Tracks accumulated DA pulses as a subjective clock signal.",
        category="Meta",
        grounded=True,
        grounding_citation="Buhusi & Meck (2005) — interval timing in DA system",
        tested=False,
        test_location="",
        benchmark_id="",
        paper_id="",
        latency_us=None,
        human_parallel="Boring meetings 'feel longer' than fun ones — DA modulates time perception.",
        notes="Designed but not implemented.",
    ),
    Capability(
        cap_id="C37",
        question="How does this make me feel in terms of valence?",
        short_name="Valence assignment",
        component="(planned valence head)",
        source_file="(future)",
        mechanism="Affective dimension orthogonal to mood state.",
        category="Emotion",
        grounded=True,
        grounding_citation="Russell (1980) — valence/arousal circumplex",
        tested=False,
        test_location="",
        benchmark_id="",
        paper_id="",
        latency_us=None,
        human_parallel="Pleasant vs unpleasant labels appear pre-conscious.",
        notes="Mood states (C16) are partial implementation; valence head separate.",
    ),
    Capability(
        cap_id="C38",
        question="What is happening in someone else's mind?",
        short_name="Theory of mind",
        component="(planned)",
        source_file="(future)",
        mechanism="Model another agent's belief from their observed actions.",
        category="Future",
        grounded=True,
        grounding_citation="Premack & Woodruff (1978); Saxe & Kanwisher (2003) — TPJ",
        tested=False,
        test_location="",
        benchmark_id="",
        paper_id="",
        latency_us=None,
        human_parallel="Watching someone reach the wrong way — 'they think it's there'.",
        notes="Designed for multi-agent extensions.",
    ),
]


# ===========================================================================
# Output generators
# ===========================================================================

def generate_markdown(caps: List[Capability]) -> str:
    """Generate the canonical Markdown catalog."""
    lines = []

    # Header
    lines.append("# NeMo-WM Introspective Capabilities Catalog")
    lines.append("")
    lines.append("> Generated by `introspective_capabilities_catalog.py`. "
                 "Re-run after adding capabilities.")
    lines.append("")
    lines.append("**This is the single source of truth for what NeMo-WM can "
                 "introspect about itself.**")
    lines.append("")

    # Counts
    n_total = len(caps)
    by_status = Counter(c.status for c in caps)
    n_prod = by_status.get("PRODUCTION", 0)
    n_grd  = by_status.get("GROUNDED_ONLY", 0)
    n_tst  = by_status.get("TESTED_ONLY", 0)
    n_des  = by_status.get("DESIGNED", 0)

    lines.append("## Headline Numbers")
    lines.append("")
    lines.append(f"| Status | Count | Meaning |")
    lines.append(f"|---|---|---|")
    lines.append(f"| **PRODUCTION** (grounded + tested) | **{n_prod}** | "
                 "Citation in literature AND running test code |")
    lines.append(f"| GROUNDED only | {n_grd} | Cited grounding, "
                 "implementation pending |")
    lines.append(f"| TESTED only | {n_tst} | Working code, "
                 "grounding pending |")
    lines.append(f"| DESIGNED | {n_des} | Designed in architecture, not built |")
    lines.append(f"| **TOTAL** | **{n_total}** | Distinct introspective capabilities |")
    lines.append("")

    # Methodology
    lines.append("## Methodology")
    lines.append("")
    lines.append("Each capability has two independent properties:")
    lines.append("")
    lines.append("- **GROUNDED**: has a citation linking it to a known mechanism "
                 "in cognitive science / neuroscience.")
    lines.append("- **TESTED**: has executable code that runs and asserts "
                 "(not just 'function exists').")
    lines.append("")
    lines.append("Status is the cross-product. The strongest claim NeMo-WM can "
                 "make is **PRODUCTION** — both grounded AND tested.")
    lines.append("")

    # Comparison
    lines.append("## Comparison with Other World Models")
    lines.append("")
    lines.append("Estimated PRODUCTION-grade introspective capabilities:")
    lines.append("")
    lines.append("| Model | Production capabilities |")
    lines.append("|---|---|")
    lines.append(f"| **NeMo-WM** | **{n_prod}** |")
    lines.append("| DreamerV3 | ~2 (latent state, rollout) |")
    lines.append("| TD-MPC2 | ~2 |")
    lines.append("| DINO-WM | ~1 (visual prediction) |")
    lines.append("| Diffusion Policy | 0 (no introspective layer) |")
    lines.append("")

    # Master table by category
    lines.append("## Master Catalog (by category)")
    lines.append("")
    by_cat = {}
    for c in caps:
        by_cat.setdefault(c.category, []).append(c)

    for cat in ["Perception", "Imagination", "Decision", "Language",
                "Meta", "Emotion", "Memory", "Future"]:
        cat_caps = by_cat.get(cat, [])
        if not cat_caps:
            continue
        lines.append(f"### {cat} ({len(cat_caps)})")
        lines.append("")
        lines.append("| ID | Status | Question | Component | Grounding | Test |")
        lines.append("|---|---|---|---|---|---|")
        for c in cat_caps:
            grd = "Y" if c.grounded else "-"
            tst = "Y" if c.tested else "-"
            grounding = (c.grounding_citation[:40] + "...") if len(c.grounding_citation) > 40 \
                        else c.grounding_citation
            test = (c.test_location[:30] + "...") if len(c.test_location) > 30 \
                   else c.test_location
            lines.append(f"| {c.cap_id} | {c.status_emoji} | {c.question} | "
                         f"`{c.component}` | {grounding or '-'} | {test or '-'} |")
        lines.append("")

    # Detailed entries
    lines.append("## Detailed Entries")
    lines.append("")
    for c in caps:
        lines.append(f"### {c.cap_id}: {c.short_name}")
        lines.append("")
        lines.append(f"**Question:** {c.question}")
        lines.append("")
        lines.append(f"**Status:** {c.status_emoji} {c.status}")
        lines.append("")
        lines.append(f"**Component:** `{c.component}`")
        lines.append("")
        lines.append(f"**Source:** `{c.source_file}`")
        lines.append("")
        lines.append(f"**Mechanism:** {c.mechanism}")
        lines.append("")
        if c.grounded:
            lines.append(f"**Grounding:** {c.grounding_citation}")
            lines.append("")
        if c.tested:
            lines.append(f"**Test:** `{c.test_location}`")
            lines.append("")
        if c.latency_us is not None:
            lines.append(f"**Latency:** {c.latency_us} us")
            lines.append("")
        lines.append(f"**Human parallel:** {c.human_parallel}")
        lines.append("")
        if c.benchmark_id or c.paper_id:
            ids = []
            if c.benchmark_id:
                ids.append(f"benchmark={c.benchmark_id}")
            if c.paper_id:
                ids.append(f"paper={c.paper_id}")
            lines.append(f"**External IDs:** {', '.join(ids)}")
            lines.append("")
        if c.notes:
            lines.append(f"**Notes:** {c.notes}")
            lines.append("")
        lines.append("---")
        lines.append("")

    # Footer
    lines.append("")
    lines.append("## Adding a Capability")
    lines.append("")
    lines.append("Edit `introspective_capabilities_catalog.py`, append a "
                 "`Capability(...)` entry, re-run the script. The Markdown "
                 "regenerates automatically.")
    lines.append("")

    return "\n".join(lines)


def validate(caps: List[Capability]) -> int:
    """Validate the catalog. Return number of issues found."""
    issues = 0
    seen_ids = set()
    for c in caps:
        if c.cap_id in seen_ids:
            print(f"  DUPLICATE ID: {c.cap_id}")
            issues += 1
        seen_ids.add(c.cap_id)
        if c.grounded and not c.grounding_citation:
            print(f"  {c.cap_id} marked grounded but no citation")
            issues += 1
        if c.tested and not c.test_location:
            print(f"  {c.cap_id} marked tested but no test_location")
            issues += 1
        if not c.question.endswith("?"):
            print(f"  {c.cap_id} question missing '?': {c.question[:60]}")
            issues += 1
    return issues


# ===========================================================================
# Main
# ===========================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="INTROSPECTIVE_CAPABILITIES_CATALOG.md")
    p.add_argument("--json", action="store_true")
    p.add_argument("--check", action="store_true")
    args = p.parse_args()

    if args.check:
        print(f"Validating {len(CAPABILITIES)} capabilities...")
        n_issues = validate(CAPABILITIES)
        if n_issues == 0:
            print(f"OK -- all entries valid")
        else:
            print(f"{n_issues} issues found")
        return n_issues

    if args.json:
        out = json.dumps([asdict(c) for c in CAPABILITIES], indent=2)
        if args.output.endswith(".md"):
            args.output = args.output.replace(".md", ".json")
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"Wrote {len(CAPABILITIES)} entries to {args.output}")
        return 0

    md = generate_markdown(CAPABILITIES)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(md)

    by_status = Counter(c.status for c in CAPABILITIES)
    print(f"Wrote {len(CAPABILITIES)} capabilities to {args.output}")
    print(f"  PRODUCTION (grounded+tested): {by_status.get('PRODUCTION', 0)}")
    print(f"  GROUNDED only:                {by_status.get('GROUNDED_ONLY', 0)}")
    print(f"  TESTED only:                  {by_status.get('TESTED_ONLY', 0)}")
    print(f"  DESIGNED:                     {by_status.get('DESIGNED', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
