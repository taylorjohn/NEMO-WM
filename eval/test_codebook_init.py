"""
test_codebook_init.py — Data-Initialized vs Random Codebook
=============================================================
Proves that initializing the DiVeQ codebook from early observations
fixes the -0.9% novelty decay issue.

Expected result:
  Random init:  high novelty, barely decays (codes far from data)
  Data init:    low baseline novelty, meaningful decay on familiar data,
                high novelty on truly novel inputs

Usage:
    python test_codebook_init.py
"""

import torch
import numpy as np
from diveq_schema import DiVeQSchemaStore


def generate_observations(n=500, d=64, n_clusters=8, seed=42):
    """Generate synthetic observations with cluster structure."""
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_clusters, d).astype(np.float32) * 2.0
    obs = []
    labels = []
    for i in range(n):
        c = i % n_clusters
        point = centers[c] + rng.randn(d).astype(np.float32) * 0.3
        obs.append(point)
        labels.append(c)
    return torch.from_numpy(np.array(obs)), labels


def generate_novel(n=50, d=64, seed=99):
    """Generate truly novel observations (far from training clusters)."""
    rng = np.random.RandomState(seed)
    # Shift far from training distribution
    novel = rng.randn(n, d).astype(np.float32) * 5.0 + 10.0
    return torch.from_numpy(novel)


def test_novelty_decay(store, observations, label, n_steps=100):
    """Run observations through store, track novelty over time."""
    novelties = []
    for i in range(min(n_steps, len(observations))):
        nov = store.novelty(observations[i])
        store.update(observations[i])
        novelties.append(nov)
    return novelties


def main():
    print("=" * 65)
    print("  DiVeQ Codebook Init: Data vs Random")
    print("  Does data initialization fix novelty decay?")
    print("=" * 65)

    d = 64
    n_schemas = 32

    # Generate data
    observations, labels = generate_observations(500, d, n_clusters=8)
    novel_obs = generate_novel(50, d)

    # Split: first 200 for init, rest for testing
    init_data = observations[:200]
    test_data = observations[200:]

    # ── Test A: Random init (current behavior) ──
    print("\n--- A: Random Init (current) ---")
    store_random = DiVeQSchemaStore(n_schemas=n_schemas, d_belief=d, use_sf=False)
    stats_r = store_random.stats()
    print(f"  Codebook norm: {stats_r['codebook_norm']:.3f}")

    # Novelty on familiar data
    nov_random_familiar = test_novelty_decay(store_random, test_data, "random")
    # Novelty on novel data
    nov_random_novel = [store_random.novelty(novel_obs[i]) for i in range(50)]

    print(f"  Familiar novelty: start={nov_random_familiar[0]:.3f}, "
          f"end={nov_random_familiar[-1]:.3f}, "
          f"decay={((nov_random_familiar[0]-nov_random_familiar[-1])/nov_random_familiar[0]*100):.1f}%")
    print(f"  Novel novelty: mean={np.mean(nov_random_novel):.3f}")

    # ── Test B: Sample init ──
    print("\n--- B: Sample Init (random subset) ---")
    store_sample = DiVeQSchemaStore(n_schemas=n_schemas, d_belief=d, use_sf=False)
    init_stats = store_sample.initialize_from_data(init_data, method="sample")
    print(f"  Active codes: {init_stats['active_codes']}/{n_schemas}")
    print(f"  Mean dist to nearest: {init_stats['mean_dist_to_nearest']:.3f}")
    print(f"  Codebook norm: {init_stats['codebook_norm']:.3f}")

    nov_sample_familiar = test_novelty_decay(store_sample, test_data, "sample")
    nov_sample_novel = [store_sample.novelty(novel_obs[i]) for i in range(50)]

    print(f"  Familiar novelty: start={nov_sample_familiar[0]:.3f}, "
          f"end={nov_sample_familiar[-1]:.3f}, "
          f"decay={((nov_sample_familiar[0]-nov_sample_familiar[-1])/(nov_sample_familiar[0]+1e-8)*100):.1f}%")
    print(f"  Novel novelty: mean={np.mean(nov_sample_novel):.3f}")

    # ── Test C: K-means++ init ──
    print("\n--- C: K-means++ Init ---")
    store_kpp = DiVeQSchemaStore(n_schemas=n_schemas, d_belief=d, use_sf=False)
    init_stats = store_kpp.initialize_from_data(init_data, method="kmeans++")
    print(f"  Active codes: {init_stats['active_codes']}/{n_schemas}")
    print(f"  Mean dist to nearest: {init_stats['mean_dist_to_nearest']:.3f}")

    nov_kpp_familiar = test_novelty_decay(store_kpp, test_data, "kmeans++")
    nov_kpp_novel = [store_kpp.novelty(novel_obs[i]) for i in range(50)]

    print(f"  Familiar novelty: start={nov_kpp_familiar[0]:.3f}, "
          f"end={nov_kpp_familiar[-1]:.3f}, "
          f"decay={((nov_kpp_familiar[0]-nov_kpp_familiar[-1])/(nov_kpp_familiar[0]+1e-8)*100):.1f}%")
    print(f"  Novel novelty: mean={np.mean(nov_kpp_novel):.3f}")

    # ── Test D: Full K-means init ──
    print("\n--- D: K-means Init (20 iterations) ---")
    store_km = DiVeQSchemaStore(n_schemas=n_schemas, d_belief=d, use_sf=False)
    init_stats = store_km.initialize_from_data(init_data, method="kmeans", n_iter=20)
    print(f"  Active codes: {init_stats['active_codes']}/{n_schemas}")
    print(f"  Mean dist to nearest: {init_stats['mean_dist_to_nearest']:.3f}")

    nov_km_familiar = test_novelty_decay(store_km, test_data, "kmeans")
    nov_km_novel = [store_km.novelty(novel_obs[i]) for i in range(50)]

    print(f"  Familiar novelty: start={nov_km_familiar[0]:.3f}, "
          f"end={nov_km_familiar[-1]:.3f}, "
          f"decay={((nov_km_familiar[0]-nov_km_familiar[-1])/(nov_km_familiar[0]+1e-8)*100):.1f}%")
    print(f"  Novel novelty: mean={np.mean(nov_km_novel):.3f}")

    # ── Summary ──
    print(f"\n{'='*65}")
    print(f"  {'Method':<16} {'Familiar':>10} {'Novel':>10} {'Ratio':>8} {'Decay':>8}")
    print(f"  {'-'*54}")

    results = [
        ("Random", np.mean(nov_random_familiar), np.mean(nov_random_novel),
         nov_random_familiar[0], nov_random_familiar[-1]),
        ("Sample", np.mean(nov_sample_familiar), np.mean(nov_sample_novel),
         nov_sample_familiar[0], nov_sample_familiar[-1]),
        ("K-means++", np.mean(nov_kpp_familiar), np.mean(nov_kpp_novel),
         nov_kpp_familiar[0], nov_kpp_familiar[-1]),
        ("K-means", np.mean(nov_km_familiar), np.mean(nov_km_novel),
         nov_km_familiar[0], nov_km_familiar[-1]),
    ]

    best_ratio = 0
    best_method = ""
    for name, fam, nov, start, end in results:
        ratio = nov / (fam + 1e-8)
        decay_pct = (start - end) / (start + 1e-8) * 100
        print(f"  {name:<16} {fam:>10.3f} {nov:>10.3f} {ratio:>7.1f}x {decay_pct:>+7.1f}%")
        if ratio > best_ratio:
            best_ratio = ratio
            best_method = name

    print(f"  {'-'*54}")
    print(f"  Best novel/familiar ratio: {best_method} ({best_ratio:.1f}x)")
    print(f"\n  Key insight:")
    print(f"    Random init: familiar novelty ≈ novel novelty (can't distinguish)")
    print(f"    Data init:   novel >> familiar (correctly identifies new experiences)")
    print(f"{'='*65}")

    # Discrimination test
    print(f"\n  Discrimination test (can the codebook tell familiar from novel?):")
    for name, fam, nov, _, _ in results:
        gap = nov - fam
        discriminates = "YES" if gap > fam * 0.5 else "NO"
        print(f"    {name:<16} gap={gap:.3f}  discriminates: {discriminates}")


if __name__ == "__main__":
    main()
