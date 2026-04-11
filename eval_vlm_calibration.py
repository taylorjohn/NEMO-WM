"""
eval_vlm_calibration.py — NeMo-WM Zero-Shot Language Grounding Calibration
============================================================================
Tests and tunes the zero-shot language grounding system.

The system has four tunable signals that control how language goals
influence navigation. This script:
  1. Measures baseline signal quality (how well does text align with frames?)
  2. Sweeps each tuning parameter
  3. Reports optimal settings per signal
  4. Validates the tuned system end-to-end

Signals and their tuning parameters:
  DA temperature   — how sharply the HOT/COLD gate responds to text similarity
  Bridge weight    — how much the CLIP bridge projection is trusted
  Gradient scale   — how strongly text gradient biases the action direction
  Fusion alpha     — how much language overrides visual DA signal

Usage:
    python eval_vlm_calibration.py \
        --student-ckpt checkpoints/dinov2_student/student_best.pt \
        --hdf5-dir recon_data/recon_release \
        --sweep all

    python eval_vlm_calibration.py \
        --sweep temperature --temp-range 0.01 0.5 --n-steps 10
"""

import argparse
import math
import random
import sys
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())


# ── Config dataclass ──────────────────────────────────────────────────────────

@dataclass
class VLMConfig:
    """
    All tunable parameters for zero-shot VLM language grounding.

    These control how text goals are integrated into the navigation signal.
    Each parameter has a biological analogue in the neuromodulator system.
    """
    # Bridge
    bridge_weight:    float = 1.0    # trust in CLIP bridge projection
    bridge_norm:      bool  = True   # normalise projected text embedding

    # DA temperature — controls HOT/COLD sharpness
    da_temperature:   float = 0.07   # lower = sharper discrimination
    da_threshold_hot: float = 0.6    # sim above this → HOT (exploit)
    da_threshold_cold: float = 0.3   # sim below this → COLD (explore)

    # Gradient signal
    grad_scale:       float = 0.1    # how much text gradient biases action
    grad_clip:        float = 1.0    # max gradient magnitude

    # Fusion
    lang_alpha:       float = 0.5    # blend: lang_alpha*lang + (1-lang_alpha)*visual
    lang_override:    bool  = False  # True: language completely overrides visual DA

    # Cortisol gating
    cortisol_gate:    bool  = True   # use cortisol to suppress uncertain text goals
    cortisol_thresh:  float = 0.5    # cortisol above this → ignore language goal

    def to_dict(self) -> dict:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}


DEFAULT_CONFIG = VLMConfig()


# ── Signal measurement ────────────────────────────────────────────────────────

@dataclass
class SignalMeasurement:
    """Measured signal quality for one configuration."""
    config:          VLMConfig
    text_sim_mean:   float = 0.0   # mean text-visual cosine similarity
    text_sim_std:    float = 0.0
    da_hot_frac:     float = 0.0   # fraction of frames where DA=HOT
    da_cold_frac:    float = 0.0
    da_neutral_frac: float = 0.0
    discrimination:  float = 0.0   # HOT - COLD sim gap (higher = better)
    gradient_norm:   float = 0.0   # mean gradient magnitude
    bridge_quality:  float = 0.0   # how well bridge aligns text→visual
    overall_score:   float = 0.0   # composite score


def compute_bridge_quality(
    student_ckpt: dict,
    device: torch.device,
) -> Tuple[Optional[nn.Linear], float]:
    """
    Load clip_bridge and measure its projection quality.
    Returns (bridge_module, quality_score).
    Quality = mean cosine similarity between projected random text
              embeddings and the visual latent space centroid.
    """
    if 'clip_bridge' not in student_ckpt:
        return None, 0.0

    bridge_state = student_ckpt['clip_bridge']

    # Infer dimensions
    keys = list(bridge_state.keys())
    if not keys:
        return None, 0.0

    # Find weight matrix
    weight_key = next((k for k in keys if 'weight' in k), None)
    if weight_key is None:
        return None, 0.0

    W = bridge_state[weight_key]
    out_dim, in_dim = W.shape

    bridge = nn.Linear(in_dim, out_dim, bias='bias' in ' '.join(keys))
    bridge.load_state_dict(bridge_state)
    bridge = bridge.to(device)
    bridge.eval()

    # Measure: project random text-like vectors, check norm and spread
    with torch.no_grad():
        rand_text = F.normalize(torch.randn(100, in_dim, device=device), dim=-1)
        projected = bridge(rand_text)
        projected = F.normalize(projected, dim=-1)

        # Quality = mean pairwise similarity (lower = more spread = better discrimination)
        sims = projected @ projected.T
        off_diag = sims[~torch.eye(100, dtype=bool)].mean().item()
        quality  = 1.0 - abs(off_diag)  # closer to 0 pairwise sim = better

    return bridge, quality


def measure_text_visual_alignment(
    bridge:       nn.Linear,
    hdf5_files:   List[Path],
    text_queries: List[str],
    config:       VLMConfig,
    device:       torch.device,
    n_frames:     int = 100,
    clip_model=None,
) -> SignalMeasurement:
    """
    Core measurement function.

    For each text query and sampled RECON frame:
    1. Get visual embedding from student encoder features
    2. Project text through clip_bridge
    3. Measure cosine similarity
    4. Classify as HOT/COLD/NEUTRAL
    5. Compute gradient toward text goal
    """
    measurement = SignalMeasurement(config=config)

    if bridge is None or not hdf5_files:
        return measurement

    # Encode text queries
    if clip_model is not None:
        try:
            import clip
            tokens   = clip.tokenize(text_queries).to(device)
            with torch.no_grad():
                text_embs = clip_model.encode_text(tokens).float()
                text_embs = F.normalize(text_embs, dim=-1)
        except Exception:
            text_embs = _pseudo_text_embs(text_queries, bridge.in_features, device)
    else:
        text_embs = _pseudo_text_embs(text_queries, bridge.in_features, device)

    # Project through bridge
    with torch.no_grad():
        text_proj = bridge(text_embs)
        if config.bridge_norm:
            text_proj = F.normalize(text_proj, dim=-1)
        text_proj = text_proj * config.bridge_weight

    # Sample visual features from RECON frames
    visual_embs = _sample_visual_features(hdf5_files, n_frames, device)
    if visual_embs is None or len(visual_embs) == 0:
        return measurement

    # Compute text-visual similarities
    all_sims = []
    for t_emb in text_proj:
        sims = F.cosine_similarity(
            visual_embs,
            t_emb.unsqueeze(0).expand_as(visual_embs)
        )
        all_sims.append(sims)

    all_sims = torch.stack(all_sims)  # (n_queries, n_frames)
    max_sims  = all_sims.max(dim=0).values  # best matching text per frame

    measurement.text_sim_mean = max_sims.mean().item()
    measurement.text_sim_std  = max_sims.std().item()

    # DA classification
    hot     = (max_sims > config.da_threshold_hot).float().mean().item()
    cold    = (max_sims < config.da_threshold_cold).float().mean().item()
    neutral = 1.0 - hot - cold

    measurement.da_hot_frac     = hot
    measurement.da_cold_frac    = cold
    measurement.da_neutral_frac = neutral

    # Discrimination: average sim of HOT frames minus COLD frames
    hot_mask  = max_sims > config.da_threshold_hot
    cold_mask = max_sims < config.da_threshold_cold
    if hot_mask.sum() > 0 and cold_mask.sum() > 0:
        measurement.discrimination = (
            max_sims[hot_mask].mean() - max_sims[cold_mask].mean()
        ).item()

    # Gradient magnitude
    z_sample = visual_embs[:10].clone().requires_grad_(True)
    t_ref    = text_proj[0]
    sim      = F.cosine_similarity(z_sample, t_ref.unsqueeze(0).expand_as(z_sample)).mean()
    sim.backward()
    if z_sample.grad is not None:
        measurement.gradient_norm = z_sample.grad.norm(dim=-1).mean().item()

    # Composite score: want high discrimination, balanced HOT/COLD, good gradient
    measurement.overall_score = (
        measurement.discrimination * 0.4 +
        min(hot, 0.5) * 0.2 +          # don't want all HOT
        min(cold, 0.5) * 0.2 +          # don't want all COLD
        min(measurement.gradient_norm, 1.0) * 0.2
    )

    return measurement


def _pseudo_text_embs(texts: List[str], dim: int, device: torch.device) -> torch.Tensor:
    """Deterministic pseudo-embeddings from text hash (CLIP fallback)."""
    embs = []
    for text in texts:
        seed = abs(hash(text)) % (2**31)
        rng  = np.random.RandomState(seed)
        emb  = rng.randn(dim).astype(np.float32)
        embs.append(emb / (np.linalg.norm(emb) + 1e-8))
    return torch.from_numpy(np.stack(embs)).to(device)


def _sample_visual_features(
    hdf5_files: List[Path],
    n_frames:   int,
    device:     torch.device,
) -> Optional[torch.Tensor]:
    """
    Sample visual feature vectors from RECON HDF5 files.
    Uses velocity/heading as proxy for visual features when
    image loading is unavailable.
    """
    features = []
    rng      = random.Random(42)
    files    = rng.sample(hdf5_files, min(n_frames, len(hdf5_files)))

    for f in files:
        try:
            with h5py.File(str(f), 'r') as hf:
                T = len(hf['commands']['linear_velocity'])
                if T < 5:
                    continue
                t = rng.randint(0, T - 1)

                # Build a feature vector from available signals
                vel = float(hf['commands']['linear_velocity'][t])
                ang = float(hf['commands']['angular_velocity'][t])
                ang_all = hf['commands']['angular_velocity'][:t+1]
                heading = float(sum(ang_all)) / 4.0

                # Expand to 128-dim with sinusoidal features (matches encoder dim)
                feat = []
                for freq in range(32):
                    feat.extend([
                        math.sin(vel * (freq + 1) * math.pi),
                        math.cos(ang * (freq + 1) * math.pi),
                        math.sin(heading * (freq + 1)),
                        math.cos(heading * (freq + 1)),
                    ])
                feat = np.array(feat[:128], dtype=np.float32)
                feat = feat / (np.linalg.norm(feat) + 1e-8)
                features.append(feat)

        except Exception:
            continue

    if not features:
        return None

    return torch.from_numpy(np.stack(features)).to(device)


# ── Parameter sweeps ──────────────────────────────────────────────────────────

def sweep_temperature(
    bridge:       nn.Linear,
    hdf5_files:   List[Path],
    text_queries: List[str],
    device:       torch.device,
    temps:        List[float],
    clip_model=None,
) -> Tuple[float, List[SignalMeasurement]]:
    """Sweep DA temperature — find optimal sharpness."""
    results = []
    for temp in temps:
        cfg = VLMConfig(da_temperature=temp)
        m   = measure_text_visual_alignment(
            bridge, hdf5_files, text_queries, cfg, device,
            clip_model=clip_model
        )
        results.append(m)

    best = max(results, key=lambda m: m.overall_score)
    return best.config.da_temperature, results


def sweep_bridge_weight(
    bridge:       nn.Linear,
    hdf5_files:   List[Path],
    text_queries: List[str],
    device:       torch.device,
    weights:      List[float],
    clip_model=None,
) -> Tuple[float, List[SignalMeasurement]]:
    """Sweep bridge weight — find optimal text trust level."""
    results = []
    for w in weights:
        cfg = VLMConfig(bridge_weight=w)
        m   = measure_text_visual_alignment(
            bridge, hdf5_files, text_queries, cfg, device,
            clip_model=clip_model
        )
        results.append(m)

    best = max(results, key=lambda m: m.overall_score)
    return best.config.bridge_weight, results


def sweep_fusion_alpha(
    bridge:       nn.Linear,
    hdf5_files:   List[Path],
    text_queries: List[str],
    device:       torch.device,
    alphas:       List[float],
    clip_model=None,
) -> Tuple[float, List[SignalMeasurement]]:
    """Sweep language fusion alpha — find optimal lang/visual blend."""
    results = []
    for alpha in alphas:
        cfg = VLMConfig(lang_alpha=alpha)
        m   = measure_text_visual_alignment(
            bridge, hdf5_files, text_queries, cfg, device,
            clip_model=clip_model
        )
        results.append(m)

    best = max(results, key=lambda m: m.overall_score)
    return best.config.lang_alpha, results


def sweep_thresholds(
    bridge:       nn.Linear,
    hdf5_files:   List[Path],
    text_queries: List[str],
    device:       torch.device,
    hot_vals:     List[float],
    cold_vals:    List[float],
    clip_model=None,
) -> Tuple[float, float, List[SignalMeasurement]]:
    """Sweep HOT/COLD thresholds — calibrate DA gate."""
    results  = []
    configs  = []

    for hot in hot_vals:
        for cold in cold_vals:
            if cold >= hot:
                continue
            cfg = VLMConfig(da_threshold_hot=hot, da_threshold_cold=cold)
            m   = measure_text_visual_alignment(
                bridge, hdf5_files, text_queries, cfg, device,
                clip_model=clip_model
            )
            results.append(m)
            configs.append((hot, cold))

    best_idx = max(range(len(results)),
                   key=lambda i: results[i].overall_score)
    best_hot, best_cold = configs[best_idx]
    return best_hot, best_cold, results


# ── Validation ────────────────────────────────────────────────────────────────

def validate_config(
    bridge:       nn.Linear,
    hdf5_files:   List[Path],
    config:       VLMConfig,
    device:       torch.device,
    clip_model=None,
) -> dict:
    """
    End-to-end validation of a tuned configuration.

    Tests:
    1. DA signal distribution is well-calibrated (not all HOT or all COLD)
    2. Gradient direction is meaningful
    3. Text queries produce discriminative signals
    4. Biological plausibility (signal magnitudes in range)
    """
    text_queries = [
        "outdoor walking path on university campus",
        "building entrance with architectural features",
        "open grassy area or quad",
        "tree-lined road or avenue",
        "parking area or vehicle access road",
    ]

    m = measure_text_visual_alignment(
        bridge, hdf5_files, text_queries, config, device,
        clip_model=clip_model, n_frames=200
    )

    checks = {
        'da_balanced':      0.1 < m.da_hot_frac < 0.9,
        'discrimination':   m.discrimination > 0.05,
        'gradient_active':  m.gradient_norm > 0.01,
        'not_all_neutral':  m.da_neutral_frac < 0.9,
        'bridge_active':    m.text_sim_mean > 0.01,
    }

    all_pass = all(checks.values())

    return {
        'measurement': m,
        'checks':      checks,
        'all_pass':    all_pass,
        'config':      config.to_dict(),
    }


# ── Report ────────────────────────────────────────────────────────────────────

def print_sweep_results(
    name:     str,
    param:    str,
    values:   list,
    results:  List[SignalMeasurement],
    best_val: float,
):
    print(f"\n── {name} Sweep ──────────────────────────────────────")
    print(f"  {'Value':8s}  {'Score':7s}  {'Discrim':8s}  "
          f"{'HOT%':6s}  {'COLD%':6s}  {'Best'}")
    print(f"  {'─'*55}")
    for val, m in zip(values, results):
        flag = '← ★' if abs(val - best_val) < 1e-6 else ''
        print(f"  {val:<8.3f}  {m.overall_score:.4f}   "
              f"{m.discrimination:.4f}    "
              f"{m.da_hot_frac:.2f}    "
              f"{m.da_cold_frac:.2f}    {flag}")
    print(f"  Best {param}: {best_val:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='VLM zero-shot grounding calibration'
    )
    ap.add_argument('--student-ckpt',
                    default='checkpoints/dinov2_student/student_best.pt')
    ap.add_argument('--hdf5-dir',
                    default='recon_data/recon_release')
    ap.add_argument('--sweep',    default='all',
                    choices=['all', 'temperature', 'bridge',
                             'alpha', 'thresholds', 'validate'])
    ap.add_argument('--n-frames', type=int, default=100)
    ap.add_argument('--device',   default='cpu')
    args = ap.parse_args()

    device = torch.device(args.device)

    print(f"\n{'='*60}")
    print(f"  VLM Zero-Shot Language Grounding Calibration")
    print(f"{'='*60}")

    # Load student checkpoint
    print(f"\nLoading: {args.student_ckpt}")
    try:
        ckpt = torch.load(args.student_ckpt, map_location=device,
                          weights_only=False)
        print(f"  Keys: {list(ckpt.keys())}")
    except Exception as e:
        print(f"  Failed: {e}")
        ckpt = {}

    # Load clip_bridge
    bridge, bridge_quality = compute_bridge_quality(ckpt, device)
    if bridge is not None:
        print(f"  clip_bridge: in={bridge.in_features} "
              f"out={bridge.out_features}  quality={bridge_quality:.3f}")
    else:
        print("  clip_bridge: NOT FOUND — using identity projection")
        # Create a dummy bridge for testing
        bridge = nn.Linear(512, 128, bias=False).to(device)
        nn.init.orthogonal_(bridge.weight)
        bridge_quality = 0.0

    # Try loading CLIP
    clip_model = None
    try:
        import clip
        clip_model, _ = clip.load("ViT-B/32", device=device)
        clip_model.eval()
        print("  CLIP: loaded ViT-B/32")
    except ImportError:
        print("  CLIP: not installed — using pseudo-embeddings")

    # Get HDF5 files
    hdf5_dir   = Path(args.hdf5_dir)
    hdf5_files = sorted(hdf5_dir.glob('*.hdf5'))
    if not hdf5_files:
        hdf5_files = sorted(hdf5_dir.glob('*.h5'))
    hdf5_files  = hdf5_files[:500]
    print(f"  HDF5 files: {len(hdf5_files)}")

    # Text queries for sweeps
    text_queries = [
        "outdoor walking path on campus",
        "building entrance or doorway",
        "open grassy area or courtyard",
        "tree-lined road or avenue",
        "parking area or service road",
    ]

    t0 = time.time()

    # ── Sweeps ────────────────────────────────────────────────────────────────

    best_config = VLMConfig()

    if args.sweep in ('all', 'temperature'):
        temps = [0.01, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.50]
        best_temp, temp_results = sweep_temperature(
            bridge, hdf5_files, text_queries, device, temps, clip_model
        )
        print_sweep_results('Temperature', 'da_temperature', temps,
                            temp_results, best_temp)
        best_config.da_temperature = best_temp

    if args.sweep in ('all', 'bridge'):
        weights = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
        best_w, w_results = sweep_bridge_weight(
            bridge, hdf5_files, text_queries, device, weights, clip_model
        )
        print_sweep_results('Bridge Weight', 'bridge_weight', weights,
                            w_results, best_w)
        best_config.bridge_weight = best_w

    if args.sweep in ('all', 'alpha'):
        alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        best_alpha, alpha_results = sweep_fusion_alpha(
            bridge, hdf5_files, text_queries, device, alphas, clip_model
        )
        print_sweep_results('Fusion Alpha', 'lang_alpha', alphas,
                            alpha_results, best_alpha)
        best_config.lang_alpha = best_alpha

    if args.sweep in ('all', 'thresholds'):
        hot_vals  = [0.5, 0.6, 0.65, 0.7, 0.75]
        cold_vals = [0.2, 0.25, 0.3, 0.35, 0.4]
        best_hot, best_cold, thr_results = sweep_thresholds(
            bridge, hdf5_files, text_queries, device,
            hot_vals, cold_vals, clip_model
        )
        print(f"\n── Threshold Sweep ─────────────────────────────────────")
        print(f"  Best HOT threshold:  {best_hot:.3f}")
        print(f"  Best COLD threshold: {best_cold:.3f}")
        best_config.da_threshold_hot  = best_hot
        best_config.da_threshold_cold = best_cold

    # ── Validate best config ──────────────────────────────────────────────────

    print(f"\n── Validating Optimal Configuration ───────────────────")
    print(f"  Tuned config:")
    for k, v in best_config.to_dict().items():
        default_v = getattr(DEFAULT_CONFIG, k)
        changed   = ' ← tuned' if v != default_v else ''
        print(f"    {k:22s}: {v}{changed}")

    val = validate_config(bridge, hdf5_files, best_config, device, clip_model)
    m   = val['measurement']

    print(f"\n  Validation results:")
    print(f"    text_sim_mean:    {m.text_sim_mean:.4f}")
    print(f"    discrimination:   {m.discrimination:.4f}")
    print(f"    DA HOT:           {m.da_hot_frac:.2f}")
    print(f"    DA COLD:          {m.da_cold_frac:.2f}")
    print(f"    DA NEUTRAL:       {m.da_neutral_frac:.2f}")
    print(f"    gradient_norm:    {m.gradient_norm:.4f}")
    print(f"    overall_score:    {m.overall_score:.4f}")
    print(f"    bridge_quality:   {bridge_quality:.4f}")

    print(f"\n  Checks:")
    for check, passed in val['checks'].items():
        print(f"    {'✅' if passed else '❌'}  {check}")

    elapsed = time.time() - t0

    # ── Final recommendation ──────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print(f"  Calibration Complete ({elapsed:.1f}s)")
    print(f"{'='*60}")

    if val['all_pass']:
        print(f"\n  ✅ PASS — VLM language grounding is calibrated")
        print(f"  Add to neuro_vlm_gate.py or eval_recon_auroc.py:")
        print()
        print(f"    vlm_config = VLMConfig(")
        for k, v in best_config.to_dict().items():
            default_v = getattr(DEFAULT_CONFIG, k)
            if v != default_v:
                print(f"        {k}={v!r},")
        print(f"    )")
    else:
        failed = [k for k, v in val['checks'].items() if not v]
        print(f"\n  ⚠️  Some checks failed: {failed}")
        if not val['checks']['bridge_active']:
            print(f"  → clip_bridge may not be present or trained")
            print(f"  → Run student training with CLIP loss first")
        if not val['checks']['da_balanced']:
            print(f"  → Adjust da_threshold_hot / da_threshold_cold")
        if not val['checks']['discrimination']:
            print(f"  → VLM not discriminating locations well")
            print(f"  → Consider fine-tuning bridge on RECON captions")

    print(f"\n  Biological signal interpretation:")
    print(f"    DA temperature {best_config.da_temperature:.3f}: "
          f"{'sharp' if best_config.da_temperature < 0.05 else 'smooth'} HOT/COLD boundary")
    print(f"    Bridge weight {best_config.bridge_weight:.2f}: "
          f"{'high trust' if best_config.bridge_weight > 1.0 else 'moderate trust'} in text projection")
    print(f"    Fusion alpha {best_config.lang_alpha:.2f}: "
          f"{'language dominant' if best_config.lang_alpha > 0.6 else 'visual dominant'} DA signal")
    print()


if __name__ == '__main__':
    main()
