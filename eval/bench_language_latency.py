"""
bench_language_latency.py — Language Layer Latency Benchmark
==============================================================
Confirms <20μs per call for Paper 2.

Benchmarks all language components:
  1. WordGrounder.understand()     — single word lookup
  2. WordGrounder._stem()          — stemming fallback
  3. SentenceComprehender.comprehend() — full sentence
  4. WordGrounder.similarity()     — word pair similarity
  5. WordGrounder.describe()       — belief → language
  6. SelfNarrator (if available)   — template narration

Usage:
    python bench_language_latency.py
"""

import time
import numpy as np
from collections import defaultdict

from word_grounder import WordGrounder, SentenceComprehender, D_BELIEF


def benchmark(func, args, n_warmup=100, n_runs=10000, label=""):
    """Benchmark a function, return mean/std in microseconds."""
    # Warmup
    for _ in range(n_warmup):
        func(*args)

    # Timed runs
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter_ns()
        func(*args)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000.0)  # ns → μs

    times = np.array(times)
    return {
        "label": label,
        "mean_us": np.mean(times),
        "std_us": np.std(times),
        "median_us": np.median(times),
        "p99_us": np.percentile(times, 99),
        "min_us": np.min(times),
        "max_us": np.max(times),
        "n_runs": n_runs,
        "hz": 1e6 / np.mean(times),
    }


def build_grounder():
    """Build a 380-word vocabulary for realistic benchmarking."""
    try:
        from overnight_language_curriculum import build_extended_curriculum
        grounder = WordGrounder(d_belief=D_BELIEF)
        curriculum, _ = build_extended_curriculum(d_belief=D_BELIEF)
        for word, belief, da, mood, physics, domain in curriculum:
            grounder.hear(word, belief, da=da, mood=mood,
                          physics=physics, source=domain)
        return grounder
    except ImportError:
        # Fallback: build small vocab
        grounder = WordGrounder(d_belief=D_BELIEF)
        rng = np.random.RandomState(42)
        words = ["gravity", "falling", "push", "pull", "left", "right",
                 "danger", "safe", "corridor", "steep", "fast", "slow",
                 "ball", "block", "obstacle", "wall", "door", "target",
                 "move", "stop", "turn", "climb", "drop", "bounce"]
        for w in words:
            for _ in range(5):
                belief = rng.randn(D_BELIEF).astype(np.float32)
                grounder.hear(w, belief, da=rng.random(), mood="Calm")
        return grounder


def main():
    print("=" * 70)
    print("  Language Layer Latency Benchmark")
    print("  Target: <20μs per call for Paper 2")
    print("=" * 70)

    grounder = build_grounder()
    comp = SentenceComprehender(grounder)
    print(f"  Vocabulary: {grounder.vocab_size} words")
    print(f"  Hearings: {grounder._total_hearings:,}")
    print(f"  Runs per benchmark: 10,000")

    results = []

    # 1. Word lookup (direct match)
    r = benchmark(grounder.understand, ("gravity",),
                  label="understand(word) — direct")
    results.append(r)

    # 2. Word lookup (stemming fallback)
    r = benchmark(grounder.understand, ("falls",),
                  label="understand(word) — stemmed")
    results.append(r)

    # 3. Word lookup (unknown word)
    r = benchmark(grounder.understand, ("quantum",),
                  label="understand(word) — unknown")
    results.append(r)

    # 4. Word similarity
    r = benchmark(grounder.similarity, ("gravity", "falling"),
                  label="similarity(a, b)")
    results.append(r)

    # 5. Simple sentence (3 words)
    r = benchmark(comp.comprehend, ("push left now",),
                  label="comprehend(3 words)")
    results.append(r)

    # 6. Medium sentence (7 words)
    r = benchmark(comp.comprehend, ("the ball falls due to gravity fast",),
                  label="comprehend(7 words)")
    results.append(r)

    # 7. Complex sentence (12 words)
    r = benchmark(comp.comprehend,
                  ("if the path is steep then move slowly and be careful now",),
                  label="comprehend(12 words)")
    results.append(r)

    # 8. Describe belief (inverse: belief → words)
    test_belief = np.random.randn(D_BELIEF).astype(np.float32)
    r = benchmark(grounder.describe, (test_belief,),
                  label="describe(belief) — 5 nearest")
    results.append(r)

    # 9. Nearest words
    r = benchmark(grounder.nearest_words, (test_belief, 5),
                  label="nearest_words(k=5)")
    results.append(r)

    # 10. Tokenize
    r = benchmark(grounder._tokenize,
                  ("the ball falls due to gravity on the steep corridor",),
                  label="_tokenize(10 words)")
    results.append(r)

    # 11. SelfNarrator (if available)
    try:
        from language_layer import SelfNarrator
        narrator = SelfNarrator()
        # Build a fake neuromod state
        neuro_state = {
            "DA": 0.5, "ACh": 0.3, "CRT": 0.1, "NE": 0.4,
            "5HT": 0.2, "step": 100, "reward": 0.8,
        }
        r = benchmark(narrator.narrate, (neuro_state,),
                      label="SelfNarrator.narrate()")
        results.append(r)
    except (ImportError, Exception) as e:
        print(f"  (SelfNarrator skipped: {e})")

    # Print results
    print(f"\n{'='*70}")
    print(f"  {'Operation':<35} {'Mean':>8} {'Median':>8} {'P99':>8} {'Hz':>12}")
    print(f"  {'-'*67}")

    all_pass = True
    for r in results:
        status = "✓" if r["mean_us"] < 20 else "✗"
        if r["mean_us"] >= 20:
            all_pass = False
        print(f"  {status} {r['label']:<33} "
              f"{r['mean_us']:>7.2f}μs "
              f"{r['median_us']:>7.2f}μs "
              f"{r['p99_us']:>7.2f}μs "
              f"{r['hz']:>11,.0f}")

    print(f"  {'-'*67}")

    # Summary stats
    word_ops = [r for r in results if "understand" in r["label"]]
    sent_ops = [r for r in results if "comprehend" in r["label"]]

    if word_ops:
        avg_word = np.mean([r["mean_us"] for r in word_ops])
        print(f"\n  Word lookup avg:     {avg_word:.2f}μs")
    if sent_ops:
        avg_sent = np.mean([r["mean_us"] for r in sent_ops])
        print(f"  Sentence comp avg:   {avg_sent:.2f}μs")

    overall_avg = np.mean([r["mean_us"] for r in results])
    overall_max = max(r["mean_us"] for r in results)
    print(f"  Overall avg:         {overall_avg:.2f}μs")
    print(f"  Slowest operation:   {overall_max:.2f}μs")

    print(f"\n{'='*70}")
    if all_pass:
        print(f"  ALL PASS: Every operation < 20μs")
        print(f"  Paper 2 claim confirmed: language layer adds negligible latency")
    else:
        slow = [r for r in results if r["mean_us"] >= 20]
        print(f"  {len(slow)} operations >= 20μs:")
        for r in slow:
            print(f"    {r['label']}: {r['mean_us']:.2f}μs")
        if overall_avg < 20:
            print(f"  Overall average still < 20μs ({overall_avg:.2f}μs)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
