"""
vocab_trainer.py — Train Vocabulary from Real Navigation Data
===============================================================
Feeds real Minari PointMaze beliefs through the compositional
narrator to grow vocabulary grounded in actual physics.

Saves vocabulary to disk so it persists across runs.
Loads existing vocabulary to continue growing.

Usage:
    python vocab_trainer.py                          # train on all 3 mazes
    python vocab_trainer.py --cycles 1000            # more cycles
    python vocab_trainer.py --status                 # show current vocab
    python vocab_trainer.py --test                   # quick validation
"""

import argparse
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, List

D_BELIEF = 64

VOCAB_PATH = Path("data/vocabulary.npz")
VOCAB_STATS_PATH = Path("data/vocabulary_stats.json")
Path("data").mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Vocabulary Persistence
# ══════════════════════════════════════════════════════════════════════════════

def save_vocabulary(narrator):
    """Save full vocabulary to disk."""
    from compositional_narrator import GroundedWord

    words = list(narrator.words.keys())
    prototypes = np.stack([narrator.words[w].prototype for w in words])
    hearings = np.array([narrator.words[w].n_hearings for w in words])
    variances = np.array([narrator.words[w].variance for w in words])

    np.savez(VOCAB_PATH,
             words=np.array(words),
             prototypes=prototypes,
             hearings=hearings,
             variances=variances)

    # Save compositions and stats
    stats = narrator.get_stats()
    stats["compositions"] = {k: v for k, v in narrator.compositions.items()}
    stats["word_list"] = words

    with open(VOCAB_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"  Saved {len(words)} words to {VOCAB_PATH}")


def load_vocabulary():
    """Load vocabulary from disk into a narrator."""
    from compositional_narrator import CompositionalNarrator, GroundedWord

    narrator = CompositionalNarrator()

    if not VOCAB_PATH.exists():
        print(f"  No saved vocabulary found at {VOCAB_PATH}")
        return narrator

    data = np.load(VOCAB_PATH, allow_pickle=True)
    words = data["words"]
    prototypes = data["prototypes"]
    hearings = data["hearings"]

    for i, word in enumerate(words):
        word = str(word)
        narrator.words[word] = GroundedWord(
            word, prototypes[i], n_hearings=int(hearings[i]))

    # Load compositions
    if VOCAB_STATS_PATH.exists():
        with open(VOCAB_STATS_PATH) as f:
            stats = json.load(f)
        narrator.compositions = stats.get("compositions", {})
        narrator.words_invented = stats.get("invented", 0)
        narrator.words_composed = stats.get("composed", 0)
        narrator.words_split = stats.get("split", 0)
        narrator.total_narrations = stats.get("total_narrations", 0)

    print(f"  Loaded {len(narrator.words)} words from {VOCAB_PATH}")
    return narrator


# ══════════════════════════════════════════════════════════════════════════════
# Belief Projection (same as train_from_minari.py)
# ══════════════════════════════════════════════════════════════════════════════

def project_to_belief(obs, d_belief=D_BELIEF):
    """Project low-D observations to belief space."""
    d_obs = obs.shape[-1]
    rng = np.random.RandomState(42)

    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False

    N = obs.shape[0]
    belief = np.zeros((N, d_belief), dtype=np.float32)
    belief[:, :d_obs] = obs / (np.abs(obs).max(axis=0, keepdims=True) + 1e-8)

    n_quad = min(d_obs * (d_obs + 1) // 2, d_belief - d_obs)
    idx = d_obs
    for i in range(d_obs):
        for j in range(i, d_obs):
            if idx >= d_belief:
                break
            belief[:, idx] = obs[:, i] * obs[:, j] * 0.1
            idx += 1

    W_rand = rng.randn(d_obs, d_belief - idx).astype(np.float32) * 0.3
    if idx < d_belief:
        belief[:, idx:] = np.tanh(obs @ W_rand)

    if squeeze:
        return belief[0]
    return belief


# ══════════════════════════════════════════════════════════════════════════════
# Load Minari Data
# ══════════════════════════════════════════════════════════════════════════════

def load_minari_beliefs(dataset_name, max_episodes=200):
    """Load beliefs from Minari dataset."""
    import minari

    print(f"  Loading {dataset_name}...")
    dataset = minari.load_dataset(dataset_name, download=True)

    all_obs = []
    episode_info = []
    count = 0

    for ep in dataset.iterate_episodes():
        obs = ep.observations
        if isinstance(obs, dict):
            obs_arr = obs.get("observation", None)
        else:
            obs_arr = obs
        if obs_arr is None:
            continue

        all_obs.append(obs_arr)
        episode_info.append({
            "episode": count,
            "length": len(obs_arr),
        })
        count += 1
        if max_episodes and count >= max_episodes:
            break

    all_obs = np.concatenate(all_obs, axis=0).astype(np.float32)
    beliefs = project_to_belief(all_obs)

    print(f"    {count} episodes, {len(beliefs)} beliefs")
    return beliefs, episode_info


# ══════════════════════════════════════════════════════════════════════════════
# Seed Vocabulary from Spatial Structure
# ══════════════════════════════════════════════════════════════════════════════

def create_seed_vocab(beliefs):
    """
    Create seed vocabulary from data distribution.
    These are the 'first words' the system learns from experience.
    """
    vocab = {}
    mean = beliefs.mean(axis=0)
    std = beliefs.std(axis=0)

    # Position words — from data extremes
    x_sorted = beliefs[beliefs[:, 0].argsort()]
    vocab["left"] = x_sorted[:100].mean(axis=0)
    vocab["right"] = x_sorted[-100:].mean(axis=0)
    vocab["center"] = mean.copy()

    y_sorted = beliefs[beliefs[:, 1].argsort()]
    vocab["lower"] = y_sorted[:100].mean(axis=0)
    vocab["upper"] = y_sorted[-100:].mean(axis=0)

    # Velocity words — from speed
    if beliefs.shape[1] > 2:
        speeds = np.sqrt(beliefs[:, 2]**2 + beliefs[:, 3]**2) \
                  if beliefs.shape[1] > 3 else np.abs(beliefs[:, 2])
        speed_sorted = beliefs[speeds.argsort()]
        vocab["slow"] = speed_sorted[:100].mean(axis=0)
        vocab["fast"] = speed_sorted[-100:].mean(axis=0)
        vocab["still"] = speed_sorted[:20].mean(axis=0)

    # Direction words — from velocity sign
    if beliefs.shape[1] > 3:
        vocab["eastward"] = beliefs[beliefs[:, 2] > 0.5].mean(axis=0) \
                             if (beliefs[:, 2] > 0.5).sum() > 10 else mean + std * 0.5
        vocab["westward"] = beliefs[beliefs[:, 2] < -0.5].mean(axis=0) \
                             if (beliefs[:, 2] < -0.5).sum() > 10 else mean - std * 0.5
        vocab["northward"] = beliefs[beliefs[:, 3] > 0.5].mean(axis=0) \
                              if (beliefs[:, 3] > 0.5).sum() > 10 else mean + std * 0.3
        vocab["southward"] = beliefs[beliefs[:, 3] < -0.5].mean(axis=0) \
                              if (beliefs[:, 3] < -0.5).sum() > 10 else mean - std * 0.3

    # Cluster-based words — schema regions
    from pathlib import Path
    schema_path = Path("data/minari_trained/schema_codebook.npz")
    if schema_path.exists():
        codebook = np.load(schema_path)["codebook"]
        for i in range(min(8, len(codebook))):
            vocab[f"region_{i}"] = codebook[i]

    # Change words — from consecutive differences
    n_diff = min(999, len(beliefs) - 1)
    if n_diff > 10:
        diffs = np.linalg.norm(np.diff(beliefs[:n_diff+1], axis=0), axis=1)
        high_change = beliefs[:n_diff][diffs > np.percentile(diffs, 90)]
        low_change = beliefs[:n_diff][diffs < np.percentile(diffs, 10)]
        if len(high_change) > 5:
            vocab["change"] = high_change.mean(axis=0)
            vocab["turn"] = high_change[:50].mean(axis=0)
        if len(low_change) > 5:
            vocab["steady"] = low_change.mean(axis=0)
            vocab["straight"] = low_change[:50].mean(axis=0)

    # Boundary words
    vocab["start"] = beliefs[0].copy()
    vocab["goal"] = beliefs[-1].copy()

    return vocab


# ══════════════════════════════════════════════════════════════════════════════
# Training Loop
# ══════════════════════════════════════════════════════════════════════════════

def train_vocabulary(n_cycles=500, maze_names=None):
    """
    Feed real navigation beliefs through compositional narrator.
    Grows vocabulary grounded in actual physics.
    """
    from compositional_narrator import CompositionalNarrator

    if maze_names is None:
        maze_names = [
            "D4RL/pointmaze/umaze-v2",
            "D4RL/pointmaze/medium-v2",
            "D4RL/pointmaze/large-v2",
        ]

    print("=" * 70)
    print("  Vocabulary Training from Real Navigation Data")
    print("=" * 70)

    # Load or create narrator
    if VOCAB_PATH.exists():
        narrator = load_vocabulary()
        print(f"  Resuming from {len(narrator.words)} words")
    else:
        narrator = None

    # Load beliefs from all mazes
    all_beliefs = []
    all_maze_labels = []

    for maze_name in maze_names:
        try:
            beliefs, ep_info = load_minari_beliefs(maze_name, max_episodes=100)
            all_beliefs.append(beliefs)
            all_maze_labels.extend([maze_name.split("/")[-1]] * len(beliefs))
            print(f"    {maze_name}: {len(beliefs)} beliefs loaded")
        except Exception as e:
            print(f"    {maze_name}: SKIP ({e})")

    if not all_beliefs:
        print("  No data loaded!")
        return

    beliefs = np.concatenate(all_beliefs, axis=0)
    print(f"\n  Total beliefs: {len(beliefs)} across {len(maze_names)} mazes")

    # Create seed vocab if starting fresh
    if narrator is None:
        seed = create_seed_vocab(beliefs)
        narrator = CompositionalNarrator(initial_vocab=seed)
        print(f"  Created seed vocabulary: {len(seed)} words")

    # Schema codebook for context
    schema_path = Path("data/minari_trained/schema_codebook.npz")
    if schema_path.exists():
        codebook = np.load(schema_path)["codebook"]
    else:
        codebook = None

    # Training loop
    print(f"\n  Training for {n_cycles} cycles...")
    print(f"  {'Cycle':>7} │ {'Vocab':>6} │ {'Quality':>7} │ "
          f"{'New':>4} │ {'Compounds':>9} │ Event")
    print(f"  {'─'*7}─┼─{'─'*6}─┼─{'─'*7}─┼─{'─'*4}─┼─{'─'*9}─┼─{'─'*20}")

    t0 = time.time()
    vocab_curve = []
    quality_curve = []

    for cycle in range(n_cycles):
        # Pick a random belief from real data
        idx = np.random.randint(len(beliefs))
        belief = beliefs[idx]

        # Context from data
        schema_id = 0
        if codebook is not None:
            dists = np.linalg.norm(codebook - belief, axis=1)
            schema_id = int(np.argmin(dists))

        maze_label = all_maze_labels[idx] if idx < len(all_maze_labels) else "unknown"

        context = {
            "schema": schema_id,
            "step": cycle,
            "maze": maze_label,
        }

        result = narrator.narrate(belief, context)
        vocab_curve.append(result["vocab_size"])
        quality_curve.append(result["quality"])

        # Print milestones
        event = ""
        if result["new_words"]:
            for nw in result["new_words"][:1]:
                if "_" in nw and nw not in ["left_a", "left_b",
                                              "right_a", "right_b"]:
                    event = f"COMPOSE: {nw}"
                elif nw.startswith("exp_"):
                    event = f"INVENT: {nw}"
                else:
                    event = f"SPLIT: {nw}"

        if cycle % (n_cycles // 20) == 0 or (event and cycle % 50 == 0):
            print(f"  {cycle:>7} │ {result['vocab_size']:>6} │ "
                  f"{result['quality']:>7.3f} │ "
                  f"{len(result['new_words']):>4} │ "
                  f"{len(narrator.compositions):>9} │ {event}")

        # Save periodically
        if cycle > 0 and cycle % 500 == 0:
            save_vocabulary(narrator)

    elapsed = time.time() - t0

    # Final save
    save_vocabulary(narrator)

    # Summary
    stats = narrator.get_stats()
    print(f"\n{'='*70}")
    print(f"  Training Summary")
    print(f"{'='*70}")
    print(f"  Cycles:       {n_cycles}")
    print(f"  Time:         {elapsed:.1f}s ({n_cycles/elapsed:.0f} cycles/s)")
    print(f"  Vocabulary:   {stats['vocab_size']} words")
    print(f"  Base words:   {stats['base_words']}")
    print(f"  Compounds:    {stats['compounds']}")
    print(f"  Invented:     {stats['invented']}")
    print(f"  Split:        {stats['split']}")
    print(f"  Composed:     {stats['composed']}")
    print(f"  Avg quality:  {np.mean(quality_curve):.3f}")
    print(f"  Avg hearings: {stats['avg_hearings']:.1f}")
    print(f"  Data source:  {', '.join(maze_names)}")

    # Growth milestones
    print(f"\n  Vocabulary Growth:")
    checkpoints = np.linspace(0, n_cycles - 1, 10, dtype=int)
    for cp in checkpoints:
        if cp < len(vocab_curve):
            print(f"    Cycle {cp:>6}: {vocab_curve[cp]:>5} words")

    # Top compounds
    if narrator.compositions:
        print(f"\n  Top Compounds:")
        for compound, parts in list(narrator.compositions.items())[:10]:
            n = narrator.words[compound].n_hearings if compound in narrator.words else 0
            print(f"    {compound:<30} = {' + '.join(parts):>25}  "
                  f"(heard {n}×)")

    # Word similarity sanity check
    print(f"\n  Word Similarity Check:")
    pairs = [("left", "right"), ("fast", "slow"), ("upper", "lower")]
    for w1, w2 in pairs:
        if w1 in narrator.words and w2 in narrator.words:
            sim = narrator.words[w1].similarity(narrator.words[w2].prototype)
            print(f"    sim({w1}, {w2}) = {sim:+.3f}")

    print(f"\n  Saved to: {VOCAB_PATH}")
    print(f"{'='*70}")


def show_status():
    """Show current vocabulary status."""
    if not VOCAB_PATH.exists():
        print("No vocabulary saved yet.")
        return

    data = np.load(VOCAB_PATH, allow_pickle=True)
    words = data["words"]
    hearings = data["hearings"]

    print(f"=" * 65)
    print(f"  Vocabulary Status")
    print(f"=" * 65)
    print(f"  Total words: {len(words)}")

    if VOCAB_STATS_PATH.exists():
        with open(VOCAB_STATS_PATH) as f:
            stats = json.load(f)
        print(f"  Compounds:   {stats.get('compounds', 0)}")
        print(f"  Invented:    {stats.get('invented', 0)}")
        print(f"  Split:       {stats.get('split', 0)}")
        print(f"  Narrations:  {stats.get('total_narrations', 0)}")

    # Top words by hearings
    sorted_idx = np.argsort(hearings)[::-1]
    print(f"\n  Top 20 words by usage:")
    for i in sorted_idx[:20]:
        print(f"    {str(words[i]):<30} heard {hearings[i]:>6}×")

    # Base vs compound
    base = sum(1 for w in words if "_" not in str(w))
    compound = sum(1 for w in words if "_" in str(w))
    print(f"\n  Base words: {base}")
    print(f"  Compound words: {compound}")
    print(f"=" * 65)


def run_tests():
    print("=" * 65)
    print("  Vocabulary Trainer Tests")
    print("=" * 65)
    p = 0; t = 0

    print("\n  T1: Seed vocabulary from data")
    rng = np.random.RandomState(42)
    beliefs = rng.randn(500, D_BELIEF).astype(np.float32)
    seed = create_seed_vocab(beliefs)
    ok = len(seed) >= 5
    print(f"    Seed words: {len(seed)} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T2: Narrator with seed vocab works")
    from compositional_narrator import CompositionalNarrator
    narrator = CompositionalNarrator(initial_vocab=seed)
    result = narrator.narrate(beliefs[0], {"schema": 0, "step": 0})
    ok = result["vocab_size"] >= len(seed)
    print(f"    Vocab: {result['vocab_size']} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T3: Save vocabulary")
    save_vocabulary(narrator)
    ok = VOCAB_PATH.exists()
    print(f"    Saved: {VOCAB_PATH.exists()} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T4: Load vocabulary")
    narrator2 = load_vocabulary()
    ok = len(narrator2.words) == len(narrator.words)
    print(f"    Loaded {len(narrator2.words)} words "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T5: Training grows vocabulary")
    for i in range(100):
        belief = rng.randn(D_BELIEF).astype(np.float32) * (0.5 + i * 0.01)
        narrator2.narrate(belief, {"schema": i % 32, "step": i})
    ok = len(narrator2.words) > len(narrator.words)
    print(f"    {len(narrator.words)} → {len(narrator2.words)} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T6: Save updated vocabulary")
    save_vocabulary(narrator2)
    narrator3 = load_vocabulary()
    ok = len(narrator3.words) == len(narrator2.words)
    print(f"    Persistence: {len(narrator3.words)} words "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print(f"\n{'='*65}")
    print(f"  Results: {p}/{t} tests passed")
    print(f"{'='*65}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--cycles", type=int, default=500)
    args = ap.parse_args()

    if args.test:
        run_tests()
    elif args.status:
        show_status()
    else:
        train_vocabulary(n_cycles=args.cycles)
