"""
download_pushtoom.py — Download PushT + TwoRoom benchmark datasets
==================================================================
Downloads the datasets used by DINO-WM for standard world model benchmarks.
Required for Sprint 9e — training NeMo-WM on PushT/TwoRoom to produce
comparable AUROC numbers against DINO-WM's published results.

Sources:
    PushT:   HuggingFace lerobot/pusht (HDF5 format, ~2GB)
    TwoRoom: HuggingFace minigrid/tworoom or DINO-WM release

Run:
    python download_pushtoom.py --dest recon_data\pushtoom
    python download_pushtoom.py --dest recon_data\pushtoom --dataset pusht
    python download_pushtoom.py --dest recon_data\pushtoom --dataset tworoom
    python download_pushtoom.py --list  # show available datasets

Author: John Taylor — github.com/taylorjohn
Date:   2026-04-06
"""

import argparse
import os
import sys
from pathlib import Path

DATASETS = {
    "pusht": {
        "description": "PushT manipulation task (DINO-WM benchmark)",
        "hf_repo":     "lerobot/pusht",
        "hf_split":    "train",
        "size_est":    "~2.1GB",
        "format":      "HDF5 episodes",
        "notes":       "T-shaped block pushing task, 200x200 frames",
    },
    "tworoom": {
        "description": "TwoRoom navigation task (DINO-WM benchmark)",
        "hf_repo":     "lerobot/minigrid_tworooms",
        "hf_split":    "train",
        "size_est":    "~800MB",
        "format":      "HDF5 episodes",
        "notes":       "Two-room navigation, grid observations",
    },
    "scand": {
        "description": "SCAND outdoor navigation (UT Austin campus)",
        "url":         "https://dataverse.tdl.org/api/access/datafile/68061",
        "size_est":    "~15GB",
        "format":      "ROS bag → HDF5",
        "notes":       "Multi-campus GeoLatentDB extension (Sprint 9c)",
    },
}


def list_datasets():
    print("\nAvailable datasets:")
    print("=" * 60)
    for name, info in DATASETS.items():
        print(f"\n  {name}")
        print(f"    {info['description']}")
        print(f"    Size:   {info['size_est']}")
        print(f"    Format: {info['format']}")
        print(f"    Notes:  {info['notes']}")
    print()


def download_hf(repo: str, split: str, dest: Path, dataset_name: str):
    """Download from HuggingFace datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing HuggingFace datasets...")
        os.system(f"{sys.executable} -m pip install datasets --break-system-packages -q")
        from datasets import load_dataset

    print(f"\nDownloading {dataset_name} from HuggingFace: {repo}")
    print(f"  Split: {split}")
    print(f"  Destination: {dest}")
    dest.mkdir(parents=True, exist_ok=True)

    try:
        ds = load_dataset(repo, split=split, cache_dir=str(dest / "hf_cache"))
        print(f"  Downloaded: {len(ds)} episodes")

        # Convert to HDF5 format compatible with NeMo-WM
        print("  Converting to NeMo-WM HDF5 format...")
        _convert_to_hdf5(ds, dest, dataset_name)

    except Exception as e:
        print(f"\n  HuggingFace download failed: {e}")
        print("\n  Manual download alternative:")
        print(f"    pip install huggingface_hub")
        print(f"    python -c \"from huggingface_hub import snapshot_download; "
              f"snapshot_download(repo_id='{repo}', repo_type='dataset', "
              f"local_dir='{dest}')\"")


def _convert_to_hdf5(ds, dest: Path, dataset_name: str):
    """Convert HuggingFace dataset to NeMo-WM HDF5 format."""
    try:
        import h5py
        import numpy as np
        from PIL import Image
        import io

        out_dir = dest / dataset_name
        out_dir.mkdir(exist_ok=True)

        episode_id = 0
        for item in ds:
            # Each item is one timestep — group into episodes by episode_index
            ep_idx = item.get("episode_index", episode_id)
            out_path = out_dir / f"{dataset_name}_{ep_idx:05d}.hdf5"

            if out_path.exists():
                continue

            # Build episode buffer — accumulate until episode changes
            # (simplified: write one HDF5 per sample for compatibility)
            with h5py.File(out_path, 'w') as hf:
                imgs_grp = hf.create_group("images")

                # Extract image
                if "observation.image" in item:
                    img_data = item["observation.image"]
                    if hasattr(img_data, 'tobytes'):
                        buf = io.BytesIO()
                        img_data.save(buf, format='JPEG')
                        img_bytes = np.frombuffer(buf.getvalue(), dtype=np.uint8)
                        imgs_grp.create_dataset("rgb_left",
                            data=img_bytes.reshape(1, -1))

                # Extract actions
                if "action" in item:
                    action = item["action"]
                    if hasattr(action, '__iter__'):
                        hf.create_dataset("commands/linear_velocity",
                            data=[float(action[0]) if len(action) > 0 else 0.0])
                        hf.create_dataset("commands/angular_velocity",
                            data=[float(action[1]) if len(action) > 1 else 0.0])

                # Dummy GPS (PushT/TwoRoom don't have GPS)
                hf.create_dataset("gps/latlong", data=[[0.0, 0.0]])

            episode_id += 1
            if episode_id % 100 == 0:
                print(f"    Converted {episode_id} episodes...", flush=True)

        print(f"  Converted {episode_id} episodes to {out_dir}")
        print(f"  Run eval with: --hdf5-dir {out_dir}")

    except Exception as e:
        print(f"  Conversion error: {e}")
        print("  Raw HuggingFace data is in the cache directory")


def download(dataset_name: str, dest: Path):
    if dataset_name not in DATASETS:
        print(f"Unknown dataset: {dataset_name}")
        print(f"Available: {list(DATASETS.keys())}")
        return

    info = DATASETS[dataset_name]
    print(f"\nDataset: {dataset_name}")
    print(f"  {info['description']}")
    print(f"  Estimated size: {info['size_est']}")

    if "hf_repo" in info:
        download_hf(info["hf_repo"], info["hf_split"], dest, dataset_name)
    else:
        print(f"\n  Direct download URL: {info.get('url', 'see notes')}")
        print(f"  Notes: {info['notes']}")
        print(f"  Manual download required for this dataset")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download PushT/TwoRoom benchmarks for NeMo-WM"
    )
    parser.add_argument("--dest",    default="recon_data/pushtoom",
                        help="Download destination directory")
    parser.add_argument("--dataset", default="all",
                        choices=["all", "pusht", "tworoom", "scand"],
                        help="Which dataset to download (default: all)")
    parser.add_argument("--list",    action="store_true",
                        help="List available datasets and exit")
    args = parser.parse_args()

    if args.list:
        list_datasets()
        sys.exit(0)

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Destination: {dest.resolve()}")

    if args.dataset == "all":
        for name in ["pusht", "tworoom"]:
            download(name, dest)
    else:
        download(args.dataset, dest)

    print("\nDone. Next step:")
    print("  python train_cwm_v2.py \\")
    print(f"      --hdf5-dir {dest / args.dataset} \\")
    print("      --epochs 10 \\")
    print("      --save-path checkpoints\\cwm\\cwm_pushtoom_best.pt")
