"""
geo_latent_db.py — Sprint 4 GeoLatentDB
=========================================
GPS-indexed particle embedding database for goal-conditioned navigation.

Maps physical GPS coordinates to NeMo-WM particle embeddings extracted
from RECON training trajectories. Enables goal-conditioned navigation:
  1. Give a GPS target coordinate
  2. Look up the nearest particle embedding (0.027ms)
  3. Use as GRASP planner target distribution

Coverage: 65,476 entries, Berkeley campus RECON dataset
Spatial resolution: median gap 0.1m, 95th pct 0.3m, max 3.3m
Query latency: 0.027ms (KD-tree, scipy)

Usage:
    db = GeoLatentDB('geo_latent_db_gps.npy',
                     'geo_latent_db_particles.npy')
    goal_particle, dist_m = db.query(lat=37.8719, lon=-122.2585)

Author: John Taylor — github.com/taylorjohn
Date:   2026-04-03
"""

import numpy as np
import time
from pathlib import Path
from scipy.spatial import KDTree
from typing import Tuple, Optional


METRES_PER_DEG = 111000.0  # approximate, good for campus scale


class GeoLatentDB:
    """
    GPS-to-particle embedding lookup database.

    Attributes:
        gps       : (N, 2) GPS coordinates in degrees (lat, lon)
        particles : (N, 128) mean-pooled particle embeddings
        tree      : KDTree for fast nearest-neighbour queries
        n_entries : number of valid entries
    """

    def __init__(self, gps_path: str = "geo_latent_db_gps.npy",
                 particles_path: str = "geo_latent_db_particles_norm.npy"):
        """
        Load the GeoLatentDB from numpy arrays.

        Args:
            gps_path:       path to GPS coordinates array (N, 2) in degrees
            particles_path: path to particle embeddings array (N, 128)
        """
        if not Path(gps_path).exists():
            raise FileNotFoundError(
                f"GeoLatentDB not found: {gps_path}\n"
                "Run the GeoLatentDB builder script first:\n"
                "  python -c \"import geo_latent_db; geo_latent_db.build()\""
            )

        t0 = time.time()
        self.gps       = np.load(gps_path)          # (N, 2) degrees
        self.particles = np.load(particles_path)     # (N, 128)
        self.n_entries = len(self.gps)

        # Build KD-tree in metre space for meaningful distance queries
        gps_m  = self.gps * METRES_PER_DEG          # (N, 2) metres
        self.tree = KDTree(gps_m)

        load_time = time.time() - t0
        print(f"GeoLatentDB loaded: {self.n_entries:,} entries in {load_time:.2f}s")
        print(f"  GPS range: lat {self.gps[:,0].min():.4f}–{self.gps[:,0].max():.4f}")
        print(f"             lon {self.gps[:,1].min():.4f}–{self.gps[:,1].max():.4f}")

    def query(self, lat: float, lon: float,
              k: int = 1) -> Tuple[np.ndarray, float]:
        """
        Find the nearest particle embedding for a GPS goal coordinate.

        Args:
            lat: target latitude in degrees
            lon: target longitude in degrees
            k:   number of nearest neighbours (default 1)

        Returns:
            particle: (128,) or (k, 128) mean-pooled particle embedding
            dist_m:   distance to nearest entry in metres
        """
        query_m = np.array([lat * METRES_PER_DEG, lon * METRES_PER_DEG])
        dists, idxs = self.tree.query(query_m, k=k)

        if k == 1:
            return self.particles[idxs], float(dists)
        else:
            return self.particles[idxs], dists.tolist()

    def query_radius(self, lat: float, lon: float,
                     radius_m: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find all entries within a radius of a GPS coordinate.

        Args:
            lat:      centre latitude in degrees
            lon:      centre longitude in degrees
            radius_m: search radius in metres

        Returns:
            particles: (M, 128) particle embeddings within radius
            dists_m:   (M,) distances in metres
        """
        query_m = np.array([lat * METRES_PER_DEG, lon * METRES_PER_DEG])
        idxs = self.tree.query_ball_point(query_m, radius_m)

        if not idxs:
            return np.zeros((0, 128)), np.zeros(0)

        idxs = np.array(idxs)
        dists = np.linalg.norm(
            self.gps[idxs] * METRES_PER_DEG - query_m, axis=1
        )
        return self.particles[idxs], dists

    def mean_goal_particle(self, lat: float, lon: float,
                           radius_m: float = 5.0) -> np.ndarray:
        """
        Return mean particle embedding within radius — more robust than
        single nearest neighbour for noisy GPS or sparse areas.

        Args:
            lat:      goal latitude in degrees
            lon:      goal longitude in degrees
            radius_m: averaging radius in metres

        Returns:
            particle: (128,) normalised mean particle embedding
        """
        particles, dists = self.query_radius(lat, lon, radius_m)

        if len(particles) == 0:
            # Fallback to single nearest neighbour
            particle, dist = self.query(lat, lon)
            return particle / (np.linalg.norm(particle) + 1e-8)

        # Distance-weighted average
        weights = 1.0 / (dists + 1.0)
        weighted = (particles * weights[:, None]).sum(0) / weights.sum()
        return weighted / (np.linalg.norm(weighted) + 1e-8)

    def coverage_stats(self) -> dict:
        """
        Compute spatial coverage statistics.

        Returns:
            dict with median_gap_m, p95_gap_m, max_gap_m
        """
        dists, _ = self.tree.query(self.gps * METRES_PER_DEG, k=2)
        nearest = dists[:, 1]
        return {
            "n_entries":    self.n_entries,
            "median_gap_m": float(np.median(nearest)),
            "p95_gap_m":    float(np.percentile(nearest, 95)),
            "max_gap_m":    float(np.max(nearest)),
        }

    def benchmark_query(self, n: int = 1000) -> float:
        """
        Benchmark query latency.

        Returns:
            mean latency in milliseconds
        """
        idxs = np.random.randint(0, self.n_entries, size=n)
        queries = self.gps[idxs]

        t0 = time.time()
        for q in queries:
            self.tree.query(q * METRES_PER_DEG)
        elapsed = time.time() - t0

        ms_per_query = elapsed * 1000 / n
        print(f"Query benchmark: {ms_per_query:.4f}ms per query ({n} trials)")
        return ms_per_query


def build(hdf5_dir: str = "recon_data/recon_release",
          out_gps: str = "geo_latent_db_gps.npy",
          out_particles: str = "geo_latent_db_particles.npy",
          frame_stride: int = 10):
    """
    Build GeoLatentDB from RECON HDF5 files.

    Args:
        hdf5_dir:     directory containing RECON HDF5 files
        out_gps:      output path for GPS numpy array
        out_particles: output path for particles numpy array
        frame_stride: sample every Nth frame (default 10 = 2.5s at 4Hz)
    """
    import glob, h5py, torch, io
    from PIL import Image
    from train_mvtec import StudentEncoder
    from train_cwm_DEPRECATED import CortexWorldModel

    print(f"Building GeoLatentDB from {hdf5_dir}...")

    encoder = StudentEncoder()
    ckpt = torch.load(
        r"checkpoints\dinov2_student\student_best.pt",
        map_location="cpu", weights_only=False
    )
    sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
    encoder.load_state_dict(sd, strict=False)
    encoder.eval()

    cwm = CortexWorldModel(d_model=128, K=16)
    ckpt2 = torch.load(
        r"checkpoints\cwm\cwm_best.pt",
        map_location="cpu", weights_only=False
    )
    cwm.load_state_dict(ckpt2["model"], strict=False)
    cwm.eval()

    files = sorted(glob.glob(f"{hdf5_dir}/*.hdf5"))
    all_gps, all_particles = [], []

    for fi, fpath in enumerate(files):
        if fi % 500 == 0:
            print(f"  File {fi}/{len(files)}... ({len(all_gps)} entries)")
        try:
            with h5py.File(fpath, "r") as hf:
                imgs = hf["images"]["rgb_left"]
                gps  = hf["gps"]["latlong"]
                for i in range(0, len(imgs) - 1, frame_stride):
                    ft = Image.open(
                        io.BytesIO(bytes(imgs[i]))
                    ).convert("RGB").resize((224, 224))
                    ft = torch.from_numpy(
                        __import__("numpy").array(ft)
                    ).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                    with torch.no_grad():
                        z = encoder(ft)
                        p, _, _, _ = cwm.encode(z)
                        p_mean = p.mean(1).squeeze(0).numpy()
                    gps_val = list(map(float, gps[i]))
                    if abs(gps_val[0]) > 0.001 and abs(gps_val[1]) > 0.001:
                        all_gps.append(gps_val)
                        all_particles.append(p_mean)
        except Exception:
            continue

    gps_arr = np.array(all_gps, dtype=np.float64)
    par_arr = np.array(all_particles, dtype=np.float32)

    # Normalise particles — raw CWM outputs are not unit-normalised
    par_norm = par_arr / (np.linalg.norm(par_arr, axis=1, keepdims=True) + 1e-8)
    np.save(out_gps, gps_arr)
    np.save(out_particles, par_norm.astype(np.float32))
    print(f"GeoLatentDB saved: {len(all_gps):,} entries (particles unit-normalised)")
    print(f"  {out_gps}: {gps_arr.nbytes / 1e6:.1f}MB")
    print(f"  {out_particles}: {par_arr.nbytes / 1e6:.1f}MB")
    return gps_arr, par_arr


if __name__ == "__main__":
    # Quick test
    db = GeoLatentDB()
    stats = db.coverage_stats()
    print(f"\nCoverage stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    db.benchmark_query(1000)
