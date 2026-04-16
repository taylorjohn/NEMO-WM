"""
benchmark_diveq.py — DiVeQ vs EMA Schema Benchmark
=====================================================
Runs the same RECON curiosity loop with two buffer implementations:
  1. Original EpisodicBuffer (k-means/EMA schemas)
  2. DiVeQEpisodicBuffer (differentiable schemas)

Measures:
  - Novelty decay rate (how fast schemas learn)
  - Retrieval latency
  - Schema coverage (codebook utilization)
  - Consolidation quality (loss reduction)

Usage:
    python benchmark_diveq.py --max-files 50
"""

import argparse
import glob
import time
import numpy as np
import torch
import torch.nn as nn
import h5py

from episodic_buffer import EpisodicBuffer, D_BELIEF
from diveq_integration import DiVeQEpisodicBuffer


class ProprioFeatureExtractor:
    def __init__(self):
        # Fixed seed so both buffers get identical features
        torch.manual_seed(42)
        self.encoder = nn.Sequential(
            nn.Linear(8, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, D_BELIEF), nn.LayerNorm(D_BELIEF), nn.Tanh(),
        )
        self.encoder.eval()

    @torch.no_grad()
    def extract(self, hdf5_path, stride=4):
        beliefs = []
        try:
            with h5py.File(hdf5_path, "r") as hf:
                lin_vel = np.array(hf["jackal"]["linear_velocity"])
                ang_vel = np.array(hf["jackal"]["angular_velocity"])
                yaw = np.array(hf["jackal"]["yaw"])
                contact = np.array(hf["collision"]["any"]).astype(np.float32)

                T = len(lin_vel)
                if T < stride * 2:
                    return []

                for t in range(0, T - stride, stride):
                    heading = float(yaw[t])
                    features = np.array([
                        float(lin_vel[t]), float(ang_vel[t]),
                        np.sin(heading), np.cos(heading),
                        float(contact[t]), heading / np.pi,
                        float(lin_vel[t]) * np.cos(heading),
                        float(lin_vel[t]) * np.sin(heading),
                    ], dtype=np.float32)
                    feat_t = torch.from_numpy(features).unsqueeze(0)
                    b = self.encoder(feat_t).squeeze(0)
                    beliefs.append(b)
        except Exception:
            pass
        return beliefs


def run_benchmark(buf, buf_name, files, extractor, sleep_every=10):
    """Run curiosity loop on one buffer implementation."""
    file_stats = []
    t0 = time.perf_counter()
    store_times = []
    retrieve_times = []
    novelty_times = []

    for fi, fpath in enumerate(files):
        beliefs = extractor.extract(fpath, stride=4)
        if len(beliefs) < 4:
            continue

        file_novelties = []
        file_retrievals = 0

        for t in range(len(beliefs) - 1):
            b_t = beliefs[t]
            b_t1 = beliefs[t + 1]
            delta = b_t1 - b_t
            a_t = delta[:2].clone()

            # Retrieve
            t_ret = time.perf_counter()
            retrieved = buf.retrieve(b_t, k=3)
            retrieve_times.append(time.perf_counter() - t_ret)
            file_retrievals += len(retrieved)

            da = 0.5  # fixed for fair comparison

            # Store
            t_store = time.perf_counter()
            buf.store(b_t, a_t, b_t1, da=da, crt=0.1)
            store_times.append(time.perf_counter() - t_store)

            # Novelty
            t_nov = time.perf_counter()
            novelty = buf.novelty(b_t)
            novelty_times.append(time.perf_counter() - t_nov)
            file_novelties.append(float(novelty))

        stats = buf.stats()
        mean_novelty = np.mean(file_novelties) if file_novelties else 0.0

        file_stats.append({
            'file_idx': fi,
            'mean_novelty': mean_novelty,
            'ep_count': stats.get('ep_count', stats.get('episodic_size', 0)),
            'retrievals': file_retrievals,
        })

    elapsed = time.perf_counter() - t0
    return {
        'name': buf_name,
        'file_stats': file_stats,
        'elapsed': elapsed,
        'store_mean_us': np.mean(store_times) * 1e6,
        'retrieve_mean_us': np.mean(retrieve_times) * 1e6,
        'novelty_mean_us': np.mean(novelty_times) * 1e6,
        'final_stats': buf.stats(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5-dir", default="recon_data/recon_release")
    ap.add_argument("--max-files", type=int, default=50)
    args = ap.parse_args()

    print("=" * 65)
    print("  DiVeQ vs EMA Schema Benchmark on RECON")
    print("=" * 65)

    files = sorted(glob.glob(f"{args.hdf5_dir}/*.hdf5"))[:args.max_files]
    print(f"  Files: {len(files)}")

    extractor = ProprioFeatureExtractor()

    # ── Run 1: Original EpisodicBuffer ──
    print(f"\n-- Running: Original EMA Buffer --")
    buf_ema = EpisodicBuffer(k_wm=8, capacity=5000)
    result_ema = run_benchmark(buf_ema, "EMA", files, extractor)
    print(f"  Done in {result_ema['elapsed']:.1f}s")

    # ── Run 2: DiVeQ Buffer ──
    print(f"\n-- Running: DiVeQ Buffer --")
    buf_diveq = DiVeQEpisodicBuffer(d_belief=64, d_action=2,
                                      k_wm=8, capacity=5000, n_schemas=64)
    result_diveq = run_benchmark(buf_diveq, "DiVeQ", files, extractor)
    print(f"  Done in {result_diveq['elapsed']:.1f}s")

    # ── Run consolidation on DiVeQ ──
    print(f"\n-- Running DiVeQ Consolidation --")
    consol_result = buf_diveq.consolidate_differentiable(n_steps=20, lr=1e-3)
    print(f"  Status: {consol_result.get('status', 'unknown')}")
    if 'loss_start' in consol_result:
        print(f"  Loss: {consol_result['loss_start']:.4f} -> "
              f"{consol_result['loss_end']:.4f}")

    # ── Comparison ──
    print(f"\n{'='*65}")
    print(f"  Results Comparison")
    print(f"{'='*65}")

    n_ema = len(result_ema['file_stats'])
    n_diveq = len(result_diveq['file_stats'])

    # Novelty over time
    def quarter_novelty(stats):
        n = len(stats)
        if n < 4:
            return [0] * 4
        q_size = n // 4
        return [
            np.mean([s['mean_novelty'] for s in stats[:q_size]]),
            np.mean([s['mean_novelty'] for s in stats[q_size:2*q_size]]),
            np.mean([s['mean_novelty'] for s in stats[2*q_size:3*q_size]]),
            np.mean([s['mean_novelty'] for s in stats[3*q_size:]]),
        ]

    ema_q = quarter_novelty(result_ema['file_stats'])
    diveq_q = quarter_novelty(result_diveq['file_stats'])

    print(f"\n  Novelty over time (lower = schemas learned better):")
    print(f"  {'Quarter':<12} {'EMA':>10} {'DiVeQ':>10} {'Winner':>10}")
    print(f"  {'-'*44}")
    for i, label in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
        winner = "DiVeQ" if diveq_q[i] < ema_q[i] else "EMA"
        print(f"  {label:<12} {ema_q[i]:>10.4f} {diveq_q[i]:>10.4f} {winner:>10}")

    # Novelty decay
    ema_decay = ema_q[0] - ema_q[3]
    diveq_decay = diveq_q[0] - diveq_q[3]
    ema_pct = (ema_decay / ema_q[0] * 100) if ema_q[0] > 0 else 0
    diveq_pct = (diveq_decay / diveq_q[0] * 100) if diveq_q[0] > 0 else 0

    print(f"\n  Novelty decay Q1->Q4:")
    print(f"    EMA:   {ema_decay:+.4f} ({ema_pct:+.1f}%)")
    print(f"    DiVeQ: {diveq_decay:+.4f} ({diveq_pct:+.1f}%)")

    # Latency
    print(f"\n  Latency (microseconds):")
    print(f"  {'Operation':<12} {'EMA':>12} {'DiVeQ':>12}")
    print(f"  {'-'*38}")
    print(f"  {'store':<12} {result_ema['store_mean_us']:>12.1f} "
          f"{result_diveq['store_mean_us']:>12.1f}")
    print(f"  {'retrieve':<12} {result_ema['retrieve_mean_us']:>12.1f} "
          f"{result_diveq['retrieve_mean_us']:>12.1f}")
    print(f"  {'novelty':<12} {result_ema['novelty_mean_us']:>12.1f} "
          f"{result_diveq['novelty_mean_us']:>12.1f}")

    # Schema stats
    print(f"\n  Schema stats:")
    ema_stats = result_ema['final_stats']
    diveq_stats = result_diveq['final_stats']
    print(f"    EMA schemas: {ema_stats.get('schema_size', ema_stats.get('schema_steps', 'N/A'))}")
    print(f"    DiVeQ active: {diveq_stats.get('schema_active', 'N/A')}/{diveq_stats.get('schema_count', 'N/A')}")
    print(f"    DiVeQ usage: {diveq_stats.get('schema_usage', 0):.1%}")

    # Overall wall time
    print(f"\n  Wall time:")
    print(f"    EMA:   {result_ema['elapsed']:.1f}s")
    print(f"    DiVeQ: {result_diveq['elapsed']:.1f}s")

    print(f"\n{'='*65}")


if __name__ == "__main__":
    main()
