"""
test_predictive_v2.py — Break the 65.1% Ceiling
=================================================
Rebuilds 395-word vocabulary fresh (3 seconds) and tests
three grounding methods on identical sentences.

No checkpoint loading — avoids MemoryError from 37M hearings.
"""

import numpy as np
from word_grounder import WordGrounder, SentenceComprehender, D_BELIEF
from language_v2 import (PredictiveGrounder, BeliefAccumulator,
                          ModifierProcessor)
from overnight_language_curriculum import build_extended_curriculum


def build_saturated_grounder():
    """Rebuild 395-word vocab fresh — same words as overnight."""
    print("Building 395-word vocabulary (same as overnight)...")
    grounder = WordGrounder(d_belief=D_BELIEF)
    curriculum, domains = build_extended_curriculum(d_belief=D_BELIEF)

    for word, belief, da, mood, physics, domain in curriculum:
        grounder.hear(word, belief, da=da, mood=mood,
                      physics=physics, source=domain)

    print(f"  Vocab: {grounder.vocab_size} words")
    print(f"  Hearings: {grounder._total_hearings:,}")
    return grounder


# Same test sentences as overnight
TEST_SENTENCES = [
    # Simple (expect high all methods)
    ("the ball falls", "simple"),
    ("push left", "simple"),
    ("danger ahead", "simple"),
    # Compound
    ("the ball falls due to gravity", "compound"),
    ("push the heavy block to the left", "compound"),
    ("danger on the steep dark corridor", "compound"),
    # Complex
    ("if the path is steep then move slowly", "complex"),
    ("push the block and it moves forward", "complex"),
    ("drop the ball and it falls down", "complex"),
    # Multi-clause
    ("first observe the obstacle then plan around it", "multi"),
    ("when danger is near move slowly and be careful", "multi"),
    ("the target is far ahead beyond the closed door", "multi"),
    # Abstract (these plateau at 0% with prototype)
    ("gravity is a type of force that pulls down", "abstract"),
    ("causes lead to effects and results", "abstract"),
    ("patterns emerge from repeated examples", "abstract"),
]


def test_all_methods(grounder):
    """Run prototype, accumulator, and predictive on all sentences."""
    comp = SentenceComprehender(grounder)
    acc = BeliefAccumulator()
    mod = ModifierProcessor()
    pred = PredictiveGrounder(grounder)

    print(f"\n{'='*90}")
    print(f"  {'Sentence':<48} {'Type':<9} {'Proto':>6} {'Accum':>6} {'Pred':>6} {'Sim':>4}")
    print(f"{'='*90}")

    proto_scores = []
    accum_scores = []
    pred_scores = []

    by_type = {}

    for sent, stype in TEST_SENTENCES:
        # Prototype
        p_result = comp.comprehend(sent)
        p_conf = p_result["confidence"]

        # Accumulator
        a_result = acc.process_sentence(sent, grounder)
        a_conf = a_result["confidence"]

        # Predictive
        pr_result = pred.comprehend_predictive(sent)
        pr_conf = pr_result["confidence"]
        sim = "Y" if pr_result.get("simulated") else ""

        short = sent[:46] + ".." if len(sent) > 48 else sent
        print(f"  {short:<48} {stype:<9} "
              f"{p_conf*100:>5.0f}% "
              f"{a_conf*100:>5.0f}% "
              f"{pr_conf*100:>5.0f}% "
              f"{sim:>4}")

        proto_scores.append(p_conf)
        accum_scores.append(a_conf)
        pred_scores.append(pr_conf)

        if stype not in by_type:
            by_type[stype] = {"proto": [], "accum": [], "pred": []}
        by_type[stype]["proto"].append(p_conf)
        by_type[stype]["accum"].append(a_conf)
        by_type[stype]["pred"].append(pr_conf)

    print(f"{'='*90}")

    # Averages
    p_avg = np.mean(proto_scores)
    a_avg = np.mean(accum_scores)
    pr_avg = np.mean(pred_scores)

    print(f"\n  Overall averages:")
    print(f"    Prototype:    {p_avg*100:.1f}%")
    print(f"    Accumulator:  {a_avg*100:.1f}%")
    print(f"    Predictive:   {pr_avg*100:.1f}%")

    # By type
    print(f"\n  By sentence type:")
    print(f"  {'Type':<12} {'Proto':>8} {'Accum':>8} {'Pred':>8}")
    print(f"  {'-'*38}")
    for stype in ["simple", "compound", "complex", "multi", "abstract"]:
        if stype in by_type:
            p = np.mean(by_type[stype]["proto"])
            a = np.mean(by_type[stype]["accum"])
            pr = np.mean(by_type[stype]["pred"])
            print(f"  {stype:<12} {p*100:>7.0f}% {a*100:>7.0f}% {pr*100:>7.0f}%")

    # Word coverage check
    print(f"\n  Vocabulary coverage check:")
    all_words = set()
    grounded_words = set()
    for sent, _ in TEST_SENTENCES:
        words = grounder._tokenize(sent)
        for w in words:
            all_words.add(w)
            if grounder.understand(w) is not None:
                grounded_words.add(w)

    missing = all_words - grounded_words
    print(f"    Total unique words in tests: {len(all_words)}")
    print(f"    Grounded: {len(grounded_words)}")
    print(f"    Missing: {len(missing)}")
    if missing:
        print(f"    Missing words: {sorted(missing)}")

    # Verdict
    print(f"\n{'='*60}")
    improvement = pr_avg - p_avg
    if improvement > 0.05:
        print(f"  BREAKTHROUGH: Predictive beats prototype by "
              f"{improvement*100:.1f}%")
    elif improvement > 0:
        print(f"  MARGINAL: Predictive slightly better (+{improvement*100:.1f}%)")
    else:
        print(f"  PARITY: Methods similar ({improvement*100:+.1f}%)")

    if a_avg > p_avg + 0.05:
        print(f"  Accumulator also improves: +{(a_avg-p_avg)*100:.1f}%")

    print(f"  Note: Overnight plateau at 65.1% was on same vocab")
    print(f"  37M hearings did not improve. Architecture is the limit.")
    print(f"{'='*60}")


def main():
    print("=" * 65)
    print("  Paper 2 Decisive Experiment")
    print("  Three grounding methods on 395-word vocabulary")
    print("  Can predictive grounding break the 65.1% ceiling?")
    print("=" * 65)

    grounder = build_saturated_grounder()
    test_all_methods(grounder)


if __name__ == "__main__":
    main()
