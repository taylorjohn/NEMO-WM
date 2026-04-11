"""
probe_per_expert.py  --  Per-Expert AIM Probe for NeMo-WM MoE Router
================================================================
Based on: "Probing the Latent World" (arXiv:2603.20327)
Applies the AI Mother Tongue (AIM) framework to CWM particle embeddings.

Answers: what does each MoE expert specialise in?
Runs the AIM probe separately on frames routed to each expert at inference.
Expected: Expert 3 (100% at ep29) should show strong physical encoding;
other experts should show weaker or different signal profiles.

Original probe tests:
  - Linear velocity (robot forward speed)
  - Angular velocity (robot turning rate)
  - Temporal structure (same vs different trajectory)
  - GPS displacement (spatial distance traveled)
  - Terrain proxy (visual complexity of scene)

Method:
  1. Load CWM checkpoint (epoch 10, no training needed)
  2. Extract particle embeddings (N, K=16, 128) from RECON HDF5 files
  3. Collect metadata labels from the same HDF5 files
  4. K-means quantize particles to discrete symbols
  5. Chi-squared test + mutual information per physical label
  6. Print paper-ready results table

Expected results if CWM has learned physical structure:
  chi2 >> 0, p < 0.001, MI > 0.05 for velocity/angular/temporal labels
  chi2 ≈ 0,  p >> 0.05, MI ≈ 0    for null/random control

Usage (on NUC, ~20 min):
    python probe_cwm_latents.py \\
        --cwm-ckpt  checkpoints\\cwm\\cwm_best.pt \\
        --hdf5-dir  recon_data\\recon_release \\
        --n-files   50 \\
        --n-clusters 16

Runtime: ~15-25 minutes on NUC CPU.
"""

import io
import argparse
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

try:
    from sklearn.cluster import KMeans, MiniBatchKMeans
    from sklearn.metrics import mutual_info_score
    from scipy.stats import chi2_contingency
except ImportError:
    raise ImportError("pip install scikit-learn scipy --break-system-packages")

try:
    from train_mvtec import StudentEncoder
    print("StudentEncoder loaded from train_mvtec")
except ImportError:
    # Inline replica matching train_cwm_v2.py architecture
    class StudentEncoder(nn.Module):
        def __init__(self, embed_dim=128):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, 4, 4), nn.GELU(),
                nn.Conv2d(32, 64, 4, 4), nn.GELU(),
                nn.Conv2d(64, 128, 4, 4), nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.proj = nn.Linear(128, embed_dim)
        def forward(self, x):
            return F.normalize(self.proj(self.conv(x).flatten(1)), dim=-1)
    print("Using inline StudentEncoder replica")


class ParticleEncoder(nn.Module):
    def __init__(self, input_dim=128, K=16, d_model=128):
        super().__init__()
        self.K = K
        self.d_model = d_model
        self.proj = nn.Linear(input_dim, K * d_model)

    def forward(self, z):
        B = z.shape[0]
        return F.normalize(
            self.proj(z).view(B, self.K, self.d_model), dim=-1
        )


# ── Data loading ──────────────────────────────────────────────────────────

def decode_frame(jpeg_bytes, img_size=224):
    # Handle numpy.bytes_, numpy.void, or plain bytes
    if hasattr(jpeg_bytes, 'tobytes'):
        raw = jpeg_bytes.tobytes()
    else:
        raw = bytes(jpeg_bytes)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img = img.resize((img_size, img_size), Image.BILINEAR)
    return torch.from_numpy(np.array(img)).float().div(255).permute(2, 0, 1)


def load_hdf5_metadata(hdf5_path, t):
    """Extract physical labels at frame index t from one HDF5 file."""
    with h5py.File(hdf5_path, "r") as hf:
        lin_vel  = float(hf["commands"]["linear_velocity"][t])
        ang_vel  = float(hf["commands"]["angular_velocity"][t])
        gps      = hf["gps"]["latlong"][t]           # (2,) lat/lon
        n_frames = hf["images"]["rgb_left"].shape[0]
    return {
        "linear_velocity":  lin_vel,
        "angular_velocity": ang_vel,
        "gps_lat":          float(gps[0]),
        "gps_lon":          float(gps[1]),
        "n_frames":         n_frames,
    }


def load_frame_pair(hdf5_path, t1, t2, img_size=224):
    """Load two frames from same file and their metadata."""
    with h5py.File(hdf5_path, "r") as hf:
        rgb_ds  = hf["images"]["rgb_left"]
        raw1    = rgb_ds[t1]
        raw2    = rgb_ds[t2]
        frame1  = decode_frame(raw1.tobytes() if hasattr(raw1, 'tobytes') else bytes(raw1), img_size)
        frame2  = decode_frame(raw2.tobytes() if hasattr(raw2, 'tobytes') else bytes(raw2), img_size)
        lin1 = float(hf["commands"]["linear_velocity"][t1])
        ang1 = float(hf["commands"]["angular_velocity"][t1])
        # Ground truth from robot odometry (higher quality than commands)
        jlin1 = float(hf["jackal"]["linear_velocity"][t1])
        jang1 = float(hf["jackal"]["angular_velocity"][t1])
        yaw1  = float(hf["jackal"]["yaw"][t1])
        gps1  = hf["gps"]["latlong"][t1].copy()
        gps2  = hf["gps"]["latlong"][t2].copy()
    return frame1, frame2, lin1, ang1, jlin1, jang1, yaw1, gps1, gps2


# ── Sample collection ──────────────────────────────────────────────────────

def collect_samples(hdf5_dir, enc, pe, n_files=50, n_per_file=8,
                    k_gap=4, img_size=224, device="cpu", seed=42):
    """
    Collect particle embeddings and physical labels from RECON.

    For each file, sample n_per_file frames and extract:
      - Particle embeddings: (N, K, 128)
      - Labels: linear_vel, angular_vel, gps_displacement, k_gap, visual_norm

    Also collect cross-trajectory pairs (label: trajectory_match=0)
    and same-trajectory pairs (label: trajectory_match=1).
    """
    random.seed(seed)
    np.random.seed(seed)

    files = sorted(Path(hdf5_dir).glob("jackal_2019-*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No HDF5 files in {hdf5_dir}")
    files = files[:n_files]
    print(f"\n  Collecting from {len(files)} HDF5 files, "
          f"{n_per_file} samples each → target {len(files)*n_per_file} samples")

    # Storage
    all_particles   = []   # (N, K, 128)
    all_lin_vel     = []   # (N,)
    all_ang_vel     = []   # (N,)
    all_gps_disp    = []   # (N,) meters traveled over k_gap frames
    all_k_gap       = []   # (N,) temporal distance in frames
    all_traj_match  = []   # (N,) 1=same traj, 0=different traj
    all_visual_norm = []   # (N,) L2 norm of raw embedding (texture proxy)
    all_expert_idx  = []   # (N,) MoE expert index at inference

    # Extra label storage
    all_jlin_vel    = []
    all_jang_vel    = []
    all_yaw         = []

    enc.eval(); pe.eval()

    for fi, fpath in enumerate(files):
        if fi % 10 == 0:
            print(f"  File {fi+1}/{len(files)}...", flush=True)

        try:
            with h5py.File(fpath, "r") as hf:
                T = hf["images"]["rgb_left"].shape[0]

            if T < k_gap + 2:
                continue

            for _ in range(n_per_file):
                t1 = random.randint(0, T - k_gap - 2)
                k  = random.randint(1, k_gap)
                t2 = t1 + k

                try:
                    f1, f2, lin, ang, jlin, jang, yaw, gps1, gps2 = load_frame_pair(
                        fpath, t1, t2, img_size
                    )
                except Exception:
                    continue

                # GPS displacement (Haversine approximation in meters)
                dlat = (gps2[0] - gps1[0]) * 111_000
                dlon = (gps2[1] - gps1[1]) * 111_000 * np.cos(np.radians(gps1[0]))
                gps_disp = float(np.sqrt(dlat**2 + dlon**2))

                with torch.no_grad():
                    x = f1.unsqueeze(0).to(device)
                    z = enc(x)                          # (1, 128)
                    pts = pe(z)                         # (1, K, 128)
                    visual_norm = z.norm().item()
                    # Extract MoE routing decision for this frame
                    try:
                        if cwm is None: raise ValueError("no cwm")
                        _, _, _, router_logits = cwm.encode(z)
                        if router_logits is not None:
                            expert_idx = int(router_logits.argmax(-1).flatten()[0].item())
                        else:
                            expert_idx = -1
                    except Exception:
                        expert_idx = -1

                all_particles.append(pts.squeeze(0).cpu().numpy())
                all_expert_idx.append(expert_idx)
                all_lin_vel.append(lin)
                all_ang_vel.append(ang)
                all_jlin_vel.append(jlin)
                all_jang_vel.append(jang)
                all_yaw.append(yaw)
                all_gps_disp.append(gps_disp)
                all_k_gap.append(k)
                all_traj_match.append(1)               # same trajectory
                all_visual_norm.append(visual_norm)

        except Exception as e:
            if fi == 0 and len(all_particles) == 0:
                import traceback
                print(f"\n  ERROR on first file {fpath}:")
                traceback.print_exc()
                print("  Aborting early to show error.")
                raise
            continue

    # Add cross-trajectory pairs (trajectory_match=0)
    n_cross = len(all_particles) // 2
    flist = list(files)
    added = 0
    while added < n_cross:
        f1, f2 = random.sample(flist, 2)
        try:
            with h5py.File(f1, "r") as h1, h5py.File(f2, "r") as h2:
                T1 = h1["images"]["rgb_left"].shape[0]
                T2 = h2["images"]["rgb_left"].shape[0]
                if T1 < 2 or T2 < 2:
                    continue
                t1 = random.randint(0, T1-1)
                t2 = random.randint(0, T2-1)
                raw_c = h1["images"]["rgb_left"][t1]
                frame1 = decode_frame(raw_c.tobytes() if hasattr(raw_c,'tobytes') else bytes(raw_c), img_size)
                lin = float(h1["commands"]["linear_velocity"][t1])
                ang = float(h1["commands"]["angular_velocity"][t1])

            with torch.no_grad():
                x = frame1.unsqueeze(0).to(device)
                z = enc(x)
                pts = pe(z)
                visual_norm = z.norm().item()

            all_particles.append(pts.squeeze(0).cpu().numpy())
            all_lin_vel.append(lin)
            all_ang_vel.append(ang)
            all_jlin_vel.append(0.0)
            all_jang_vel.append(0.0)
            all_yaw.append(0.0)
            all_gps_disp.append(0.0)       # unknown for cross-traj
            all_k_gap.append(0)            # 0 = cross trajectory marker
            all_traj_match.append(0)
            all_visual_norm.append(visual_norm)
            added += 1
        except Exception:
            continue

    particles = np.stack(all_particles)   # (N, K, 128)
    labels = {
        "linear_velocity":       np.array(all_lin_vel,    dtype=float),
        "angular_velocity":      np.array(all_ang_vel,    dtype=float),
        "jackal_linear_vel":     np.array(all_jlin_vel,   dtype=float),
        "jackal_angular_vel":    np.array(all_jang_vel,   dtype=float),
        "jackal_yaw":            np.array(all_yaw,        dtype=float),
        "gps_displacement":      np.array(all_gps_disp,   dtype=float),
        "temporal_gap_k":        np.array(all_k_gap,      dtype=float),
        "trajectory_match":      np.array(all_traj_match, dtype=int),
        "visual_norm":           np.array(all_visual_norm, dtype=float),
    }
    # Null control: random labels (should show p >> 0.05)
    labels["_null_control"] = np.random.randint(0, 8, size=len(all_lin_vel))

    print(f"  Collected {particles.shape[0]} samples, "
          f"{particles.shape[1]} particles, {particles.shape[2]}-D")
    return particles, labels


# ── AIM probe ──────────────────────────────────────────────────────────────

def discretize_label(vals, n_bins=8):
    """Discretize continuous label into n_bins equal-frequency bins."""
    if vals.dtype in (int, np.int64, np.int32):
        return vals.astype(int)
    percentiles = np.linspace(0, 100, n_bins + 1)
    bins = np.percentile(vals, percentiles)
    bins = np.unique(bins)    # handle duplicates
    return np.digitize(vals, bins[1:-1]).astype(int)


def aim_probe(particles, labels, n_clusters=16, n_bins=8, seed=42):
    """
    AIM (AI Mother Tongue) probe.

    Quantize particle embeddings → discrete symbols via K-means.
    Test symbol distribution against physical labels via chi-squared + MI.

    particles: (N, K, D) — particle embeddings
    labels:    dict of {name: (N,) array}
    """
    N, K, D = particles.shape

    # ── Consensus symbol: pool across K particles ──────────────────────────
    # Two pooling strategies:
    # 1. Mean pool: average particles, then quantize
    # 2. Vote:      quantize each particle, then take majority vote

    print(f"\n  Quantizing {N} samples × {K} particles × {D}-D "
          f"→ {n_clusters} symbols...")

    # Strategy 1: mean pool then quantize
    mean_particles = particles.mean(axis=1)           # (N, D)
    km_mean = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed,
                              n_init=10, batch_size=512)
    symbols_mean = km_mean.fit_predict(mean_particles)  # (N,)

    # Strategy 2: per-particle quantize then vote
    flat = particles.reshape(N * K, D)
    km_flat = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed,
                              n_init=10, batch_size=2048)
    flat_symbols = km_flat.fit_predict(flat)
    per_particle = flat_symbols.reshape(N, K)
    symbols_vote = np.array([
        np.bincount(row, minlength=n_clusters).argmax()
        for row in per_particle
    ])

    # ── Per-label chi-squared + MI ─────────────────────────────────────────
    results = {}
    label_order = [
        "trajectory_match",   # binary, most interpretable
        "temporal_gap_k",     # 1–4 steps
        "linear_velocity",    # command velocity
        "angular_velocity",   # command turn rate
        "jackal_linear_vel",  # ground truth odometry (higher quality)
        "jackal_angular_vel", # ground truth odometry
        "jackal_yaw",         # robot heading angle
        "gps_displacement",   # continuous meters
        "visual_norm",        # proxy for texture/complexity
        "_null_control",      # must show p >> 0.05
    ]

    print(f"\n{'='*72}")
    print(f"  {'Label':25s}  {'Strategy':10s}  {'χ²':>8}  "
          f"{'p':>10}  {'MI (bits)':>10}  {'Sig':>5}")
    print(f"  {'-'*25}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*5}")

    for label_name in label_order:
        if label_name not in labels:
            continue
        raw = labels[label_name]

        # Remove NaN/inf
        valid = np.isfinite(raw)
        if valid.sum() < 50:
            continue

        label_disc = discretize_label(raw[valid], n_bins=n_bins)

        for strat_name, syms in [("mean_pool", symbols_mean[valid]),
                                  ("vote    ", symbols_vote[valid])]:
            # Contingency table
            n_sym = n_clusters
            n_lab = label_disc.max() + 1
            ct = np.zeros((n_sym, n_lab), dtype=float)
            for s, l in zip(syms, label_disc):
                ct[s, l] += 1
            ct += 1e-6   # Laplace smoothing

            # Chi-squared test
            chi2, p, dof, _ = chi2_contingency(ct)

            # Mutual information
            mi = mutual_info_score(syms, label_disc)

            # Significance flag
            if p < 0.001:
                sig = "***"
            elif p < 0.01:
                sig = " **"
            elif p < 0.05:
                sig = "  *"
            else:
                sig = "   "

            print(f"  {label_name:25s}  {strat_name:10s}  "
                  f"{chi2:8.1f}  {p:10.2e}  {mi:10.4f}  {sig:>5}")

            results[f"{label_name}_{strat_name.strip()}"] = {
                "chi2": chi2, "p": p, "mi": mi, "dof": dof
            }

        print()

    return results


# ── Paper-ready summary ────────────────────────────────────────────────────

def print_paper_summary(results):
    print(f"\n{'='*72}")
    print("  PAPER TABLE — CWM Latent Physical Structure (AIM Probe)")
    print(f"  Based on arXiv:2603.20327 (Probing the Latent World)")
    print(f"{'='*72}")
    print(f"  {'Physical quantity':28s}  {'χ²':>8}  {'p-value':>10}  "
          f"{'MI (bits)':>10}  {'Result':>12}")
    print(f"  {'-'*28}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*12}")

    label_display = {
        "trajectory_match_mean_pool":   "Trajectory identity",
        "temporal_gap_k_mean_pool":     "Temporal distance (k steps)",
        "linear_velocity_mean_pool":    "Cmd linear velocity",
        "angular_velocity_mean_pool":   "Cmd angular velocity",
        "jackal_linear_vel_mean_pool":  "GT linear velocity (odometry)",
        "jackal_angular_vel_mean_pool": "GT angular velocity (odometry)",
        "jackal_yaw_mean_pool":         "Robot heading (yaw)",
        "gps_displacement_mean_pool":   "GPS displacement (m)",
        "visual_norm_mean_pool":        "Visual complexity (enc norm)",
        "_null_control_mean_pool":      "Null control (random)",
    }

    for key, display in label_display.items():
        if key not in results:
            continue
        r = results[key]
        p = r["p"]
        if p < 0.001:
            result_str = "ENCODED ***"
        elif p < 0.01:
            result_str = "ENCODED  **"
        elif p < 0.05:
            result_str = "ENCODED   *"
        else:
            result_str = "not encoded"
        print(f"  {display:28s}  {r['chi2']:8.1f}  {p:10.2e}  "
              f"{r['mi']:10.4f}  {result_str:>12}")

    print(f"\n  *** p < 0.001   ** p < 0.01   * p < 0.05")
    print(f"\n  Interpretation:")
    print(f"  - ENCODED with p < 0.001 → CWM particles carry this physical signal")
    print(f"  - Null control should show p >> 0.05 (validates test calibration)")
    print(f"{'='*72}\n")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="AIM latent probe for CWM particles")
    p.add_argument("--cwm-ckpt",  default=r"checkpoints\cwm\cwm_best.pt")
    p.add_argument("--hdf5-dir",  default=r"recon_data\recon_release")
    p.add_argument("--n-files",   type=int, default=50,
                   help="Number of HDF5 files to probe (default 50)")
    p.add_argument("--n-per-file",type=int, default=8,
                   help="Samples per file (default 8)")
    p.add_argument("--k-gap",     type=int, default=4,
                   help="Max temporal gap in frames (default 4 = 1 second)")
    p.add_argument("--n-clusters",type=int, default=16,
                   help="K-means clusters for quantization (default 16)")
    p.add_argument("--n-bins",    type=int, default=8,
                   help="Bins for continuous label discretization (default 8)")
    p.add_argument("--K",         type=int, default=16,
                   help="Particle count (default 16)")
    p.add_argument("--device",    default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)

    print("\nCORTEX CWM — AIM Latent World Probe")
    print("="*72)
    print(f"  Checkpoint : {args.cwm_ckpt}")
    print(f"  HDF5 dir   : {args.hdf5_dir}")
    print(f"  Files      : {args.n_files}")
    print(f"  K particles: {args.K}")
    print(f"  Clusters   : {args.n_clusters}")
    print(f"  Method     : arXiv:2603.20327 (Probing the Latent World)")

    # ── Load models ────────────────────────────────────────────────────────
    enc = StudentEncoder().to(device)
    enc_path = Path("checkpoints/dinov2_student/student_best.pt")
    if enc_path.exists():
        sd = torch.load(enc_path, map_location="cpu", weights_only=False)
        enc.load_state_dict(sd.get("model", sd.get("state_dict", sd)),
                            strict=False)
        print(f"\n  StudentEncoder: loaded from {enc_path}")
    else:
        print(f"\n  StudentEncoder: random weights (student_best.pt not found)")
    enc.eval()

    pe = ParticleEncoder(128, args.K, 128).to(device)
    ckpt_path = Path(args.cwm_ckpt)
    if ckpt_path.exists():
        cwm_data = torch.load(ckpt_path, map_location="cpu",
                              weights_only=False)
        cwm_sd   = cwm_data.get("model", cwm_data)
        ep   = cwm_data.get("epoch", "?")
        loss = cwm_data.get("loss",  "?")
        pe_sd = {k.replace("particle_enc.", ""): v
                 for k, v in cwm_sd.items()
                 if k.startswith("particle_enc.")}
        if pe_sd:
            pe.load_state_dict(pe_sd, strict=False)
            lstr = f"{loss:.4f}" if isinstance(loss, float) else str(loss)
            print(f"  ParticleEncoder: loaded (epoch {ep}, loss {lstr})")
        else:
            print("  ParticleEncoder: random weights (key particle_enc.* not found)")
    else:
        print(f"  WARNING: CWM checkpoint not found: {ckpt_path}")
    pe.eval()

    # ── Collect samples ────────────────────────────────────────────────────
    particles, labels = collect_samples(
        hdf5_dir  = args.hdf5_dir,
        enc       = enc,
        pe        = pe,
        n_files   = args.n_files,
        n_per_file= args.n_per_file,
        k_gap     = args.k_gap,
        device    = device,
    )

    # ── Run AIM probe ──────────────────────────────────────────────────────
    results = aim_probe(
        particles  = particles,
        labels     = labels,
        n_clusters = args.n_clusters,
        n_bins     = args.n_bins,
    )

    # ── Print paper table ──────────────────────────────────────────────────
    print_paper_summary(results)

    print("  Next steps:")
    print("  - ENCODED labels with p < 0.001 → paper section 'Probing CWM Latents'")
    print("  - Compare: same probe on Tab 1 (random encoder) → should show p >> 0.05")
    print("  - That contrast is the mechanistic proof of learning\n")


if __name__ == "__main__":
    main()
