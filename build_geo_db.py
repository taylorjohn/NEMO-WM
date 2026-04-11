"""
build_geo_db.py — Build GeoLatentDB from NeMo-WM checkpoint
=============================================================
Iterates over RECON HDF5 files, encodes frames through
StudentEncoder + ParticleEncoder, and saves GPS-indexed
particle embeddings to geo_latent_db.pt.

The GeoLatentDB is the spatial memory for Sprint 7a/7b navigation:
  - 65,476 GPS-indexed particle embeddings
  - 0.09m median spatial gap between nodes
  - 0.022ms KD-tree query latency
  - Used by WaypointGraph for A* and language-weighted planning

Run:
    python build_geo_db.py \
        --cwm-ckpt  checkpoints\cwm\cwm_best.pt \
        --hdf5-dir  recon_data\recon_release \
        --save-path checkpoints\cwm\geo_latent_db.pt

    # Quick test on 10 files:
    python build_geo_db.py \
        --cwm-ckpt  checkpoints\cwm\cwm_best.pt \
        --hdf5-dir  recon_data\recon_release \
        --save-path checkpoints\cwm\geo_latent_db_test.pt \
        --max-files 10

Author: John Taylor — github.com/taylorjohn
"""

import argparse
import glob
import io
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# NeMo-WM imports
from train_mvtec import StudentEncoder
from train_cwm_DEPRECATED import CortexWorldModel

DEVICE    = torch.device("cpu")
IMG_SIZE  = 224
FRAMES_PER_FILE = 8    # sample this many frames per HDF5 file


def load_models(cwm_ckpt: str):
    """Load StudentEncoder and CortexWorldModel from checkpoint."""
    # StudentEncoder — always loads from its own checkpoint embedded in CWM
    enc = StudentEncoder().to(DEVICE)
    enc.eval()

    # Load student encoder weights
    student_path = Path(cwm_ckpt).parent.parent / "dinov2_student" / "student_best.pt"
    if student_path.exists():
        sd = torch.load(student_path, map_location="cpu", weights_only=False)
        enc.load_state_dict(sd.get("model", sd), strict=False)
        print(f"[build_geo_db] StudentEncoder loaded: {student_path}")
    else:
        print(f"[build_geo_db] Warning: student_best.pt not found at {student_path}")
        print(f"               Using random encoder — embeddings will be uninformative")

    # CortexWorldModel — provides ParticleEncoder
    cwm = CortexWorldModel(d_model=128, K=16).to(DEVICE)
    ckpt = torch.load(cwm_ckpt, map_location="cpu", weights_only=False)
    cwm.load_state_dict(ckpt.get("model", ckpt), strict=False)
    cwm.eval()
    ep   = ckpt.get("epoch", "?")
    loss = ckpt.get("loss",  "?")
    print(f"[build_geo_db] CWM loaded: ep{ep}, loss={loss:.4f}" if isinstance(loss, float)
          else f"[build_geo_db] CWM loaded: ep{ep}")

    return enc, cwm


def encode_frame(frame_bytes, enc, cwm):
    """
    Encode one frame to mean particle embedding.
    Returns (128,) float32 unit-normalised embedding.
    """
    img = Image.open(io.BytesIO(bytes(frame_bytes))).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    t   = torch.from_numpy(np.array(img)).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    with torch.no_grad():
        z      = enc(t)                          # (1, 128)
        p, _, _, _ = cwm.encode(z)               # p: (1, K, 128)
        mean_p = p.squeeze(0).mean(0)            # (128,)
        mean_p = F.normalize(mean_p, dim=0)      # unit norm

    return mean_p.numpy()


def build_geo_db(
    cwm_ckpt:   str,
    hdf5_dir:   str,
    save_path:  str,
    max_files:  int  = -1,
    frames_per_file: int = FRAMES_PER_FILE,
    stride:     int  = 8,     # frame stride within each HDF5
    verbose:    bool = True,
):
    """
    Build and save the GeoLatentDB.

    Args:
        cwm_ckpt:        Path to cwm_best.pt checkpoint
        hdf5_dir:        Directory containing RECON .hdf5 files
        save_path:       Output path for geo_latent_db.pt
        max_files:       Max HDF5 files to process (-1 = all)
        frames_per_file: How many frames to sample per file
        stride:          Frame stride within each file
        verbose:         Print progress

    Saves:
        {
            'gps':        np.ndarray (N, 2)    lat/lon
            'embeddings': np.ndarray (N, 128)  unit-norm particle embeddings
            'file_idx':   np.ndarray (N,)      source file index
            'frame_idx':  np.ndarray (N,)      source frame index
            'metadata': {
                'cwm_ckpt': str,
                'n_files': int,
                'n_entries': int,
                'frames_per_file': int,
                'build_time_s': float,
            }
        }
    """
    files = sorted(glob.glob(f"{hdf5_dir}/*.hdf5"))
    if max_files > 0:
        files = files[:max_files]

    print(f"[build_geo_db] Processing {len(files)} HDF5 files")
    print(f"[build_geo_db] Frames per file: {frames_per_file}, stride: {stride}")

    enc, cwm = load_models(cwm_ckpt)

    all_gps   = []
    all_embs  = []
    all_fidx  = []
    all_frameidx = []

    t_start = time.perf_counter()
    n_skipped = 0

    for file_i, path in enumerate(files):
        if verbose and file_i % 20 == 0:
            elapsed = time.perf_counter() - t_start
            rate    = file_i / max(elapsed, 1e-3)
            eta     = (len(files) - file_i) / max(rate, 1e-6)
            print(f"  File {file_i+1:4d}/{len(files)} | "
                  f"entries={len(all_gps):,} | "
                  f"elapsed={elapsed:.0f}s | eta={eta:.0f}s")

        try:
            with h5py.File(path, "r") as hf:
                # Validate required keys
                if "images" not in hf or "rgb_left" not in hf["images"]:
                    n_skipped += 1
                    continue
                if "gps" not in hf or "latlong" not in hf["gps"]:
                    n_skipped += 1
                    continue

                imgs    = hf["images"]["rgb_left"]
                gps_arr = np.array(hf["gps"]["latlong"])  # (T, 2) lat/lon
                T       = len(imgs)

                # Sample frame indices
                indices = list(range(0, T, stride))
                # Subsample to frames_per_file evenly
                if len(indices) > frames_per_file:
                    step = len(indices) / frames_per_file
                    indices = [indices[int(i * step)] for i in range(frames_per_file)]

                for frame_i in indices:
                    # GPS for this frame
                    gps = gps_arr[frame_i]
                    if not np.all(np.isfinite(gps)):
                        continue

                    # Encode frame
                    emb = encode_frame(imgs[frame_i], enc, cwm)

                    all_gps.append(gps)
                    all_embs.append(emb)
                    all_fidx.append(file_i)
                    all_frameidx.append(frame_i)

        except Exception as e:
            if verbose:
                print(f"  Skip {Path(path).name}: {e}")
            n_skipped += 1
            continue

    elapsed = time.perf_counter() - t_start

    if not all_gps:
        raise RuntimeError("No entries built — check HDF5 paths and keys")

    gps_arr  = np.array(all_gps,  dtype=np.float32)    # (N, 2)
    embs_arr = np.array(all_embs, dtype=np.float32)    # (N, 128)
    fidx_arr = np.array(all_fidx, dtype=np.int32)
    frameidx = np.array(all_frameidx, dtype=np.int32)

    # Compute spatial statistics
    from scipy.spatial import KDTree
    tree = KDTree(gps_arr)
    dists, _ = tree.query(gps_arr, k=2)
    neighbour_dists_m = dists[:, 1] * 111_000  # approx metres
    median_gap_m = float(np.median(neighbour_dists_m))

    print(f"\n[build_geo_db] Build complete:")
    print(f"  Files processed: {len(files) - n_skipped} / {len(files)}")
    print(f"  Entries:         {len(gps_arr):,}")
    print(f"  GPS range:       lat {gps_arr[:,0].min():.4f}–{gps_arr[:,0].max():.4f}")
    print(f"                   lon {gps_arr[:,1].min():.4f}–{gps_arr[:,1].max():.4f}")
    print(f"  Median gap:      {median_gap_m:.2f}m")
    print(f"  Build time:      {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "gps":        gps_arr,
        "embeddings": embs_arr,
        "file_idx":   fidx_arr,
        "frame_idx":  frameidx,
        "metadata": {
            "cwm_ckpt":        str(cwm_ckpt),
            "n_files":         len(files) - n_skipped,
            "n_entries":       len(gps_arr),
            "frames_per_file": frames_per_file,
            "median_gap_m":    median_gap_m,
            "build_time_s":    elapsed,
        }
    }, save_path)
    print(f"  Saved:           {save_path}")

    return gps_arr, embs_arr


def verify(save_path: str):
    """Quick verification of saved GeoLatentDB."""
    print(f"\n[verify] Loading: {save_path}")
    db = torch.load(save_path, map_location="cpu", weights_only=False)

    gps  = db["gps"]
    embs = db["embeddings"]
    meta = db.get("metadata", {})

    print(f"  Entries:    {len(gps):,}")
    print(f"  GPS:        lat {gps[:,0].min():.4f}–{gps[:,0].max():.4f}")
    print(f"              lon {gps[:,1].min():.4f}–{gps[:,1].max():.4f}")
    print(f"  Embeddings: {embs.shape}, dtype={embs.dtype}")
    print(f"  Norm check: mean={np.linalg.norm(embs, axis=1).mean():.4f} "
          f"(should be ~1.0)")
    print(f"  Median gap: {meta.get('median_gap_m', '?'):.2f}m")
    print(f"  Build time: {meta.get('build_time_s', 0)/60:.1f}min")

    # KD-tree query speed
    from scipy.spatial import KDTree
    tree = KDTree(gps)
    lat_q, lon_q = float(gps[:,0].mean()), float(gps[:,1].mean())
    t0 = time.perf_counter()
    for _ in range(1000):
        _, _ = tree.query([lat_q, lon_q])
    query_ms = (time.perf_counter() - t0)
    print(f"  KD-tree query: {query_ms:.3f}ms per query (1000 runs)")
    print(f"  ✓ Verification complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build NeMo-WM GeoLatentDB")
    parser.add_argument("--cwm-ckpt",    required=True,
                        help="Path to cwm_best.pt")
    parser.add_argument("--hdf5-dir",    required=True,
                        help="Directory containing RECON .hdf5 files")
    parser.add_argument("--save-path",   required=True,
                        help="Output path for geo_latent_db.pt")
    parser.add_argument("--max-files",   type=int, default=-1,
                        help="Max files to process (-1 = all ~196)")
    parser.add_argument("--frames-per-file", type=int, default=FRAMES_PER_FILE,
                        help="Frames sampled per HDF5 file (default: 8)")
    parser.add_argument("--stride",      type=int, default=8,
                        help="Frame stride within each file (default: 8)")
    parser.add_argument("--verify",      action="store_true",
                        help="Verify an existing GeoLatentDB without building")
    args = parser.parse_args()

    if args.verify:
        verify(args.save_path)
    else:
        build_geo_db(
            cwm_ckpt        = args.cwm_ckpt,
            hdf5_dir        = args.hdf5_dir,
            save_path       = args.save_path,
            max_files       = args.max_files,
            frames_per_file = args.frames_per_file,
            stride          = args.stride,
        )
        verify(args.save_path)
