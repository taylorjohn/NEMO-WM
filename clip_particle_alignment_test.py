"""
clip_particle_alignment_test.py
================================
Tests whether NeMo-WM particle embeddings are already partially
aligned with CLIP text embeddings — WITHOUT any new training.

This is the feasibility test for Sprint 6 CLIP dual-distillation.

Method:
  1. Load NeMo-WM particle embeddings from RECON HDF5 files
  2. Record ground-truth physical labels (velocity, yaw, angular vel)
  3. Encode text queries through CLIP text encoder
  4. Project CLIP 512-D → 128-D (particle space) via PCA
  5. Measure cosine similarity: do "fast robot" queries retrieve
     high-velocity particles above the random baseline?

Success criteria:
  - ratio >= 1.3x  → strong alignment, CLIP distillation straightforward
  - ratio 1.1-1.3x → weak alignment, distillation needed but will work
  - ratio ~1.0x    → no alignment, distillation load-bearing

Install CLIP first:
  pip install git+https://github.com/openai/CLIP.git

Run:
  python clip_particle_alignment_test.py \
      --cwm-ckpt  checkpoints\cwm\cwm_best.pt \
      --hdf5-dir  recon_data\recon_release \
      --n-files   50

Author: John Taylor — github.com/taylorjohn
Date:   2026-04-02
"""

import argparse
import glob
import io
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA

# ── Imports that may not be installed ────────────────────────────────────────
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("CLIP not installed. Run:")
    print("  pip install git+https://github.com/openai/CLIP.git")
    print("Then re-run this script.\n")

try:
    from train_mvtec import StudentEncoder
    from train_cwm_DEPRECATED import CortexWorldModel
    CORTEX_AVAILABLE = True
except ImportError:
    CORTEX_AVAILABLE = False
    print("CORTEX modules not found. Run from CORTEX root directory.")

# ── Text queries mapped to expected physical signal ───────────────────────────
# Format: "text query" → (signal_name, expected_direction)
# direction: "high" means top-K retrieved should be above baseline
#            "low"  means top-K retrieved should be below baseline
QUERIES = {
    # Velocity queries
    "robot moving fast forward":          ("linear_velocity", "high"),
    "robot driving at high speed":        ("linear_velocity", "high"),
    "robot moving slowly and carefully":  ("linear_velocity", "low"),
    "robot barely moving":                ("linear_velocity", "low"),
    "robot stopped or stationary":        ("linear_velocity", "low"),

    # Angular / turning queries
    "robot turning sharply":              ("angular_velocity", "high"),
    "robot spinning or rotating":         ("angular_velocity", "high"),
    "robot going straight ahead":         ("angular_velocity", "low"),
    "robot driving in a straight line":   ("angular_velocity", "low"),

    # Heading / yaw queries  
    "robot facing north outdoor campus":  ("yaw", "high"),
    "robot navigating open outdoor area": ("linear_velocity", "high"),
    "robot on a narrow path":             ("linear_velocity", "low"),
}

# ── Null control queries — should show ratio ~1.0 ────────────────────────────
NULL_QUERIES = {
    "the quick brown fox":                ("linear_velocity", "null"),
    "a recipe for chocolate cake":        ("linear_velocity", "null"),
    "sonnets by shakespeare":             ("angular_velocity", "null"),
}


def decode_jpeg(data: bytes, size: int = 224) -> torch.Tensor:
    img = Image.open(io.BytesIO(data)).convert("RGB").resize((size, size))
    return torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0


def load_particles_and_labels(cwm_ckpt: str, hdf5_dir: str,
                               n_files: int = 50, n_per_file: int = 8,
                               k_gap: int = 4):
    """
    Load particle embeddings and ground-truth physical labels from RECON HDF5.

    Returns:
        particles : (N, 128)  mean-pooled particle embeddings
        labels    : dict of (N,) arrays for each physical quantity
    """
    if not CORTEX_AVAILABLE:
        raise RuntimeError("CORTEX modules not available")

    device = torch.device("cpu")

    # Load StudentEncoder
    encoder = StudentEncoder().to(device)
    ckpt = torch.load(r"checkpoints\dinov2_student\student_best.pt",
                      map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
    encoder.load_state_dict(sd, strict=False)
    encoder.eval()
    print(f"  StudentEncoder loaded")

    # Load CWM
    cwm = CortexWorldModel(d_model=128, K=16).to(device)
    ckpt2 = torch.load(cwm_ckpt, map_location="cpu", weights_only=False)
    sd2 = ckpt2.get("model", ckpt2.get("state_dict", ckpt2))
    cwm.load_state_dict(sd2, strict=False)
    cwm.eval()
    ep = ckpt2.get("epoch", "?")
    loss = ckpt2.get("loss", "?")
    print(f"  CWM loaded (epoch {ep}, loss {loss:.4f})")

    files = sorted(glob.glob(f"{hdf5_dir}/*.hdf5"))[:n_files]
    print(f"  Collecting from {len(files)} files, {n_per_file} samples each")

    all_particles  = []
    all_lin_vel    = []
    all_ang_vel    = []
    all_yaw        = []
    all_gps_disp   = []

    for fi, fpath in enumerate(files):
        if fi % 10 == 0:
            print(f"  File {fi+1}/{len(files)}...")
        try:
            with h5py.File(fpath, "r") as hf:
                imgs = hf["images"]["rgb_left"]
                T = len(imgs)
                if T < k_gap + 1:
                    continue

                lin_vel = np.array(hf["commands"]["linear_velocity"])
                ang_vel = np.array(hf["commands"]["angular_velocity"])
                gps     = np.array(hf["gps"]["latlong"]) if "gps" in hf \
                          else np.zeros((T, 2))

                # Sample n_per_file random frames
                idxs = np.random.choice(T - k_gap, size=min(n_per_file, T - k_gap),
                                        replace=False)

                for i in idxs:
                    try:
                        ft = decode_jpeg(bytes(imgs[i])).unsqueeze(0)

                        with torch.no_grad():
                            z = encoder(ft)
                            p, _, _, _ = cwm.encode(z)
                            # p: (1, K, 128) → mean pool → (128,)
                            p_mean = p.mean(1).squeeze(0)

                        # Physical labels
                        lv = float(lin_vel[i])
                        av = abs(float(ang_vel[i]))

                        # Yaw: cumulative angular displacement as proxy
                        yaw_proxy = float(np.sum(ang_vel[max(0, i-10):i]))

                        # GPS displacement from previous frame
                        if i > 0:
                            dg = gps[i] - gps[i-1]
                            gps_disp = float(np.linalg.norm(dg) * 111000)
                        else:
                            gps_disp = 0.0

                        all_particles.append(p_mean.numpy())
                        all_lin_vel.append(lv)
                        all_ang_vel.append(av)
                        all_yaw.append(yaw_proxy)
                        all_gps_disp.append(gps_disp)

                    except Exception:
                        continue
        except OSError:
            continue

    N = len(all_particles)
    print(f"  Collected {N} samples")

    particles = np.stack(all_particles)   # (N, 128)
    labels = {
        "linear_velocity":  np.array(all_lin_vel),
        "angular_velocity": np.array(all_ang_vel),
        "yaw":              np.array(all_yaw),
        "gps_displacement": np.array(all_gps_disp),
    }
    return particles, labels


def run_alignment_test(particles: np.ndarray, labels: dict,
                        queries: dict, null_queries: dict,
                        top_k: int = 30):
    """
    Project CLIP text embeddings into particle space via PCA.
    Measure cosine similarity and compute retrieval ratios.
    """
    if not CLIP_AVAILABLE:
        raise RuntimeError("CLIP not installed")

    print("\n  Loading CLIP ViT-B/32...")
    clip_model, _ = clip.load("ViT-B/32", device="cpu")
    clip_model.eval()

    # Strategy: project BOTH particles and text into a shared 64-D space
    # Step 1: encode particles → 512-D via a random but fixed linear map
    #         (same map for particles and text so they share a basis)
    # Better strategy: use CLIP image encoder on the same frames to get
    # a shared 512-D space, then compare text vs particle in that space.
    #
    # Practical approach here: 
    # - Upsample particles 128→512 via a fixed random projection
    # - Fit PCA on the upsampled particles to get 64-D basis
    # - Project both text (512→64) and particles (128→512→64) into that basis
    # - Measure cosine similarity in 64-D shared space

    print("  Fitting shared projection space (PCA 512→64)...")
    np.random.seed(42)
    # Fixed random uplift matrix: 128 → 512
    uplift = np.random.randn(128, 512).astype(np.float32)
    uplift /= np.linalg.norm(uplift, axis=1, keepdims=True)

    # Upsample particles to 512-D
    p_up = particles @ uplift  # (N, 512)

    # Fit PCA on upsampled particles → 64-D shared space
    n_components = min(64, p_up.shape[0] - 1)
    pca = PCA(n_components=n_components)
    pca.fit(p_up)

    # Project particles into shared space
    p_shared = pca.transform(p_up)  # (N, 64)
    p_norm = p_shared / (np.linalg.norm(p_shared, axis=1, keepdims=True) + 1e-8)

    all_queries = {**queries, **null_queries}
    query_texts = list(all_queries.keys())

    print(f"  Encoding {len(query_texts)} text queries through CLIP...")
    with torch.no_grad():
        tokens = clip.tokenize(query_texts)
        text_emb = clip_model.encode_text(tokens).float().numpy()  # (Q, 512)

    # Project text into same 64-D shared space via same PCA
    text_proj = pca.transform(text_emb)  # (Q, 64)
    t_norm = text_proj / (np.linalg.norm(text_proj, axis=1, keepdims=True) + 1e-8)

    # Cosine similarity matrix (Q, N)
    sim = t_norm @ p_norm.T

    results = []
    for qi, (query, (signal, direction)) in enumerate(all_queries.items()):
        top_idx = np.argsort(sim[qi])[::-1][:top_k]

        if signal in labels:
            sig_vals = labels[signal]
            top_mean = sig_vals[top_idx].mean()
            base_mean = sig_vals.mean()
            base_std  = sig_vals.std()
            ratio = top_mean / (base_mean + 1e-8)

            # Z-score: how many std above/below baseline
            z = (top_mean - base_mean) / (base_std / np.sqrt(top_k) + 1e-8)
        else:
            top_mean = base_mean = ratio = z = 0.0

        results.append({
            "query":     query,
            "signal":    signal,
            "direction": direction,
            "top_mean":  float(top_mean),
            "base_mean": float(base_mean),
            "ratio":     float(ratio),
            "z_score":   float(z),
        })

    return results


def print_results(results: list):
    print("\n" + "=" * 90)
    print("  CLIP ↔ NeMo-WM Particle Alignment Test")
    print("  NeMo-WM Sprint 6 Feasibility Check")
    print("=" * 90)
    print(f"\n  {'Query':45s} {'Signal':18s} {'Top-K mean':>10} {'Baseline':>10} {'Ratio':>7} {'Z':>6}  Result")
    print(f"  {'-'*45} {'-'*18} {'-'*10} {'-'*10} {'-'*7} {'-'*6}  ------")

    semantic_results = []
    null_results     = []

    for r in results:
        if r["direction"] == "null":
            null_results.append(r)
        else:
            semantic_results.append(r)

    def verdict(r):
        ratio, direction, z = r["ratio"], r["direction"], r["z_score"]
        if direction == "null":
            if 0.85 <= ratio <= 1.15:
                return "✓ correct null"
            else:
                return "⚠ null leaking"
        if direction == "high":
            if ratio >= 1.3 and z > 1.5:   return "✅ STRONG"
            if ratio >= 1.1 and z > 0.5:   return "🟡 WEAK"
            return "❌ not aligned"
        if direction == "low":
            if ratio <= 0.85 and z < -1.5: return "✅ STRONG"
            if ratio <= 0.90 and z < -0.5: return "🟡 WEAK"
            return "❌ not aligned"
        return "?"

    print("\n  — Semantic queries —")
    strong = weak = miss = 0
    for r in semantic_results:
        v = verdict(r)
        if "STRONG" in v: strong += 1
        elif "WEAK"  in v: weak   += 1
        else:              miss   += 1
        print(f"  {r['query'][:45]:45s} {r['signal']:18s} "
              f"{r['top_mean']:10.4f} {r['base_mean']:10.4f} "
              f"{r['ratio']:7.2f}x {r['z_score']:6.2f}  {v}")

    print(f"\n  — Null control queries —")
    null_ok = 0
    for r in null_results:
        v = verdict(r)
        if "correct" in v: null_ok += 1
        print(f"  {r['query'][:45]:45s} {r['signal']:18s} "
              f"{r['top_mean']:10.4f} {r['base_mean']:10.4f} "
              f"{r['ratio']:7.2f}x {r['z_score']:6.2f}  {v}")

    total = len(semantic_results)
    print("\n" + "=" * 90)
    print(f"  SUMMARY")
    print(f"  Semantic queries:  {total} total | {strong} STRONG | {weak} WEAK | {miss} not aligned")
    print(f"  Null controls:     {null_ok}/{len(null_results)} correctly null")
    print()

    # Overall verdict
    strong_pct = strong / total if total > 0 else 0
    weak_pct   = (strong + weak) / total if total > 0 else 0

    if strong_pct >= 0.4 and null_ok == len(null_results):
        verdict_str = "✅ STRONG ALIGNMENT — CLIP distillation is viable NOW"
        verdict_detail = ("Particles already partially encode CLIP semantics. "
                         "Sprint 6 CLIP dual-distillation will reinforce existing "
                         "alignment rather than building from scratch. "
                         "Text-conditioned navigation is feasible without architectural changes.")
    elif weak_pct >= 0.5 and null_ok >= len(null_results) - 1:
        verdict_str = "🟡 WEAK ALIGNMENT — CLIP distillation needed but will work"
        verdict_detail = ("Some semantic structure is present but not strong enough "
                         "for direct text-to-particle retrieval. Sprint 6 CLIP "
                         "dual-distillation is load-bearing — required before "
                         "text-conditioned navigation is reliable.")
    else:
        verdict_str = "❌ NO ALIGNMENT — Full distillation required"
        verdict_detail = ("Particle embeddings are not semantically aligned with CLIP. "
                         "Sprint 6 requires full CLIP dual-distillation before "
                         "text conditioning will work. The distillation step is "
                         "non-optional.")

    print(f"  OVERALL VERDICT: {verdict_str}")
    print(f"  {verdict_detail}")
    print("=" * 90)

    return {
        "strong": strong, "weak": weak, "miss": miss,
        "null_ok": null_ok, "verdict": verdict_str,
        "strong_pct": strong_pct, "weak_pct": weak_pct,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test CLIP ↔ NeMo-WM particle alignment (Sprint 6 feasibility)"
    )
    parser.add_argument("--cwm-ckpt",  default=r"checkpoints\cwm\cwm_best.pt")
    parser.add_argument("--hdf5-dir",  default=r"recon_data\recon_release")
    parser.add_argument("--n-files",   type=int, default=50)
    parser.add_argument("--n-per-file",type=int, default=8)
    parser.add_argument("--k-gap",     type=int, default=4)
    parser.add_argument("--top-k",     type=int, default=30,
                        help="Number of top retrieved particles to evaluate")
    parser.add_argument("--save-json", action="store_true",
                        help="Save results to clip_alignment_results.json")
    args = parser.parse_args()

    if not CLIP_AVAILABLE:
        return
    if not CORTEX_AVAILABLE:
        return

    print("\nCLIP ↔ NeMo-WM Particle Alignment Test")
    print("=" * 50)
    print(f"  CWM checkpoint : {args.cwm_ckpt}")
    print(f"  HDF5 directory : {args.hdf5_dir}")
    print(f"  Files          : {args.n_files}")
    print(f"  Samples/file   : {args.n_per_file}")
    print(f"  Top-K          : {args.top_k}")
    print(f"  Method         : PCA projection of CLIP text embeddings → particle space")
    print()

    t0 = time.time()

    print("Step 1: Loading particle embeddings from RECON...")
    particles, labels = load_particles_and_labels(
        args.cwm_ckpt, args.hdf5_dir,
        args.n_files, args.n_per_file, args.k_gap
    )

    print("\nStep 2: Running CLIP alignment test...")
    results = run_alignment_test(
        particles, labels, QUERIES, NULL_QUERIES, args.top_k
    )

    print_results(results)

    print(f"\n  Total time: {time.time()-t0:.1f}s")

    if args.save_json:
        out = {
            "config": vars(args),
            "n_particles": len(particles),
            "label_stats": {k: {"mean": float(v.mean()), "std": float(v.std())}
                           for k, v in labels.items()},
            "results": results,
        }
        with open("clip_alignment_results.json", "w") as f:
            json.dump(out, f, indent=2)
        print("  Results saved to clip_alignment_results.json")


if __name__ == "__main__":
    main()
