"""
domain_loaders.py  —  CORTEX CWM Sprint 3
==========================================
Unified DataLoader factory for all CWM training domains.

Each domain returns the same dict schema:
    frame_t    (3, 224, 224)  current frame
    frame_t1   (3, 224, 224)  next frame
    action     (MAX_ACTION_DIM,)  zero-padded
    gps        (2,)  lat/lon or zeros
    domain_id  ()  int from DOMAIN_IDS
    domain     str  domain name

Domains implemented:
    recon      RECON outdoor nav HDF5 (Sprint 1 — already done)
    ogbench    OGBench-Cube 9-DOF manipulation
    pusht      PushT contact manipulation
    tworoom    TwoRoom 2D navigation
    mvtec      MVTec AD (observation-only, action=zeros)
    smap       SMAP telemetry (observation-only, 1-D signal → tiled frame)

Usage:
    from domain_loaders import build_multi_domain_loader

    loader = build_multi_domain_loader(
        domains=["recon", "ogbench", "pusht"],
        domain_roots={
            "recon":   "recon_data/recon_release",
            "ogbench": "ogbench_data",
            "pusht":   "pusht_data",
        },
        batch_size=16,
    )
    for batch in loader:
        # batch["domain"] tells you which domain this batch came from
"""

import io
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import (
    ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
)

MAX_ACTION_DIM = 9

DOMAIN_IDS = {
    "recon":     0,
    "ogbench":   1,
    "tworoom":   2,
    "pusht":     3,
    "smap":      4,
    "cardiac":   5,
    "hexapod":   6,
    "quadruped": 7,
    "mvtec":     8,
}

DOMAIN_ACTION_DIM = {
    "recon":     2,
    "ogbench":   9,
    "tworoom":   2,
    "pusht":     2,
    "smap":      0,
    "cardiac":   0,
    "mvtec":     0,
    "hexapod":   4,
    "quadruped": 12,
}


# ── Shared utilities ──────────────────────────────────────────────────────

def _decode_jpeg(jpeg_bytes: bytes, img_size: int = 224) -> torch.Tensor:
    img = Image.open(io.BytesIO(jpeg_bytes)).convert('RGB')
    img = img.resize((img_size, img_size))
    t   = torch.from_numpy(np.array(img)).float() / 255.0
    return t.permute(2, 0, 1)   # (3, H, W)

def _pad_action(action: torch.Tensor) -> torch.Tensor:
    return F.pad(action, (0, MAX_ACTION_DIM - action.shape[-1]))

def _zeros_action() -> torch.Tensor:
    return torch.zeros(MAX_ACTION_DIM)

def _signal_to_frame(signal: np.ndarray, img_size: int = 224) -> torch.Tensor:
    """
    Convert 1-D time-series signal to (3, H, W) pseudo-frame.
    Tile signal into image columns — preserves temporal structure.
    Used for SMAP telemetry and Cardiac audio.
    """
    L = len(signal)
    sig_norm = (signal - signal.min()) / (signal.ptp() + 1e-6)
    frame    = np.tile(sig_norm, (img_size, 1))  # (H, L)
    if frame.shape[1] > img_size:
        frame = frame[:, :img_size]
    else:
        frame = np.pad(frame, ((0,0),(0, img_size-frame.shape[1])))
    frame = frame[:img_size, :img_size]
    tensor = torch.from_numpy(frame).float()
    return tensor.unsqueeze(0).expand(3, -1, -1)  # (3, H, W)


# ═══════════════════════════════════════════════════════════════════════════
# Domain 0 — RECON
# ═══════════════════════════════════════════════════════════════════════════

class RECONDomain(Dataset):
    """RECON outdoor navigation. action_dim=2."""

    def __init__(self, data_root: str, max_files: int = None,
                 img_size: int = 224):
        self.img_size  = img_size
        self.domain    = "recon"
        self.domain_id = DOMAIN_IDS["recon"]
        files = sorted(Path(data_root).glob("jackal_2019-*.hdf5"))
        if max_files:
            files = files[:max_files]
        self.samples = []
        for f in files:
            try:
                with h5py.File(f, "r") as hf:
                    for i in range(min(hf['images']['rgb_left'].shape[0],
                               hf['commands']['linear_velocity'].shape[0]) - 1):
                        self.samples.append((str(f), i))
            except OSError:
                continue  # skip corrupted/truncated HDF5 files

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, i = self.samples[idx]
        with h5py.File(path, 'r') as hf:
            rgb = hf['images']['rgb_left']
            frame_t  = _decode_jpeg(bytes(rgb[i]),   self.img_size)
            frame_t1 = _decode_jpeg(bytes(rgb[i+1]), self.img_size)
            action   = torch.tensor(
                [float(hf['commands']['linear_velocity'][i]),
                 float(hf['commands']['angular_velocity'][i])],
                dtype=torch.float32)
            gps_raw  = hf['gps']['latlong'][i] if 'gps' in hf else [0., 0.]
            gps      = torch.tensor(gps_raw, dtype=torch.float32)
        return {
            "frame_t":   frame_t,
            "frame_t1":  frame_t1,
            "action":    _pad_action(action),
            "gps":       gps,
            "domain_id": torch.tensor(self.domain_id, dtype=torch.long),
            "domain":    self.domain,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Domain 1 — OGBench-Cube
# ═══════════════════════════════════════════════════════════════════════════

class OGBenchDomain(Dataset):
    """
    OGBench-Cube 9-DOF arm manipulation.
    Expects HDF5 files with keys: images, actions, observations.
    action_dim=9 (joint torques).
    """

    def __init__(self, data_root: str, max_files: int = None,
                 img_size: int = 224):
        self.img_size  = img_size
        self.domain    = "ogbench"
        self.domain_id = DOMAIN_IDS["ogbench"]
        files = sorted(Path(data_root).glob("*.hdf5"))
        if max_files:
            files = files[:max_files]
        self.samples = []
        for f in files:
            try:
                with h5py.File(f, "r") as hf:
                    n = len(hf['images']) - 1 if 'images' in hf else 0
                    for i in range(n):
                        self.samples.append((str(f), i))
            except OSError:
                continue  # skip corrupted/truncated HDF5 files

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, i = self.samples[idx]
        with h5py.File(path, 'r') as hf:
            if 'images' in hf:
                frame_t  = _decode_jpeg(bytes(hf['images'][i:i+1][0]),   self.img_size)
                frame_t1 = _decode_jpeg(bytes(hf['images'][i+1:i+2][0]), self.img_size)
            else:
                # Fallback: black frames if no images (use proprio only)
                frame_t  = torch.zeros(3, self.img_size, self.img_size)
                frame_t1 = torch.zeros(3, self.img_size, self.img_size)

            action = torch.tensor(hf['actions'][i], dtype=torch.float32)
            if action.shape[0] > MAX_ACTION_DIM:
                action = action[:MAX_ACTION_DIM]
            elif action.shape[0] < MAX_ACTION_DIM:
                action = _pad_action(action)

        return {
            "frame_t":   frame_t,
            "frame_t1":  frame_t1,
            "action":    action,
            "gps":       torch.zeros(2),
            "domain_id": torch.tensor(self.domain_id, dtype=torch.long),
            "domain":    self.domain,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Domain 2 — TwoRoom
# ═══════════════════════════════════════════════════════════════════════════

class TwoRoomDomain(Dataset):
    """
    TwoRoom 2D navigation. action_dim=2 (dx, dy).
    Expects npz or hdf5 files with frames and actions.
    """

    def __init__(self, data_root: str, max_files: int = None,
                 img_size: int = 224):
        self.img_size  = img_size
        self.domain    = "tworoom"
        self.domain_id = DOMAIN_IDS["tworoom"]
        files = sorted(Path(data_root).glob("*.npz"))
        if max_files:
            files = files[:max_files]
        self.samples = []
        for f in files:
            data = np.load(f)
            n    = len(data['observations']) - 1
            for i in range(n):
                self.samples.append((str(f), i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, i = self.samples[idx]
        data     = np.load(path)
        obs_t    = data['observations'][i]    # (H, W, 3) or (obs_dim,)
        obs_t1   = data['observations'][i+1]
        action   = torch.tensor(data['actions'][i], dtype=torch.float32)

        if obs_t.ndim == 3:
            frame_t  = torch.from_numpy(obs_t).float().permute(2,0,1) / 255.0
            frame_t1 = torch.from_numpy(obs_t1).float().permute(2,0,1) / 255.0
        else:
            frame_t  = _signal_to_frame(obs_t,  self.img_size)
            frame_t1 = _signal_to_frame(obs_t1, self.img_size)

        return {
            "frame_t":   frame_t,
            "frame_t1":  frame_t1,
            "action":    _pad_action(action),
            "gps":       torch.zeros(2),
            "domain_id": torch.tensor(self.domain_id, dtype=torch.long),
            "domain":    self.domain,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Domain 3 — PushT
# ═══════════════════════════════════════════════════════════════════════════

class PushTDomain(Dataset):
    """
    PushT contact manipulation. action_dim=2 (pusher XY).
    Expects zarr or hdf5 trajectory files.
    """

    def __init__(self, data_root: str, max_files: int = None,
                 img_size: int = 224):
        self.img_size  = img_size
        self.domain    = "pusht"
        self.domain_id = DOMAIN_IDS["pusht"]
        files = sorted(Path(data_root).glob("*.hdf5"))
        if max_files:
            files = files[:max_files]
        self.samples = []
        for f in files:
            try:
                with h5py.File(f, "r") as hf:
                    n = len(hf.get('images', hf.get('obs', []))) - 1
                    for i in range(n):
                        self.samples.append((str(f), i))
            except OSError:
                continue  # skip corrupted/truncated HDF5 files

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, i = self.samples[idx]
        with h5py.File(path, 'r') as hf:
            img_key = 'images' if 'images' in hf else 'obs'
            frame_t  = _decode_jpeg(bytes(hf[img_key][i]),   self.img_size) \
                       if isinstance(hf[img_key][i], bytes) else \
                       torch.from_numpy(np.array(hf[img_key][i])).float().permute(2,0,1)/255.0
            frame_t1 = _decode_jpeg(bytes(hf[img_key][i+1]), self.img_size) \
                       if isinstance(hf[img_key][i+1], bytes) else \
                       torch.from_numpy(np.array(hf[img_key][i+1])).float().permute(2,0,1)/255.0
            action = torch.tensor(hf['actions'][i], dtype=torch.float32)

        return {
            "frame_t":   frame_t,
            "frame_t1":  frame_t1,
            "action":    _pad_action(action),
            "gps":       torch.zeros(2),
            "domain_id": torch.tensor(self.domain_id, dtype=torch.long),
            "domain":    self.domain,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Domain 4 — SMAP Telemetry (observation-only)
# ═══════════════════════════════════════════════════════════════════════════

class SMAPDomain(Dataset):
    """
    SMAP/MSL spacecraft telemetry. action=zeros (observation-only).
    Loads from the standard smap/ directory structure:
        smap/train/<channel_id>.npy  — shape (T,)
        smap/test/<channel_id>.npy
    Frames are signal-to-image converted (tiled time series).
    """

    def __init__(self, data_root: str, split: str = "train",
                 window: int = 128, stride: int = 1, img_size: int = 224):
        self.window    = window
        self.img_size  = img_size
        self.domain    = "smap"
        self.domain_id = DOMAIN_IDS["smap"]
        self.samples   = []

        split_dir = Path(data_root) / split
        for npy_file in sorted(split_dir.glob("*.npy")):
            data = np.load(npy_file)
            T    = data.shape[0]
            for t in range(0, T - window - 1, stride):
                self.samples.append((data, t))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data, t = self.samples[idx]
        window_t  = data[t   : t + self.window, 0]   # first channel
        window_t1 = data[t+1 : t + self.window + 1, 0]
        frame_t  = _signal_to_frame(window_t,  self.img_size)
        frame_t1 = _signal_to_frame(window_t1, self.img_size)
        return {
            "frame_t":   frame_t,
            "frame_t1":  frame_t1,
            "action":    _zeros_action(),
            "gps":       torch.zeros(2),
            "domain_id": torch.tensor(self.domain_id, dtype=torch.long),
            "domain":    self.domain,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Domain 8 — MVTec AD (observation-only, normal class only)
# ═══════════════════════════════════════════════════════════════════════════

class MVTecDomain(Dataset):
    """
    MVTec AD visual inspection. action=zeros (observation-only).
    Only loads 'good' (normal) training images. Used to train
    the CWM encoder to represent normal visual states.
    """

    def __init__(self, data_root: str, categories: Optional[List[str]] = None,
                 img_size: int = 224):
        self.img_size  = img_size
        self.domain    = "mvtec"
        self.domain_id = DOMAIN_IDS["mvtec"]

        all_cats = ["bottle","cable","capsule","carpet","grid","hazelnut",
                    "leather","metal_nut","pill","screw","tile","toothbrush",
                    "transistor","wood","zipper"]
        cats = categories or all_cats
        self.image_paths = []
        for cat in cats:
            good_dir = Path(data_root) / cat / "train" / "good"
            if good_dir.exists():
                self.image_paths.extend(sorted(good_dir.glob("*.png"))
                                        + sorted(good_dir.glob("*.jpg")))

    def __len__(self):
        return max(1, len(self.image_paths) - 1)

    def __getitem__(self, idx):
        # Consecutive image pairs (t, t+1) within same category
        # For MVTec: t and t+1 are not temporally related but that's fine —
        # CWM treats action=0, so this is pure representation learning
        p_t  = self.image_paths[idx % len(self.image_paths)]
        p_t1 = self.image_paths[(idx + 1) % len(self.image_paths)]

        def load_img(p):
            img = Image.open(p).convert('RGB').resize(
                (self.img_size, self.img_size))
            return torch.from_numpy(np.array(img)).float().permute(2,0,1)/255.0

        return {
            "frame_t":   load_img(p_t),
            "frame_t1":  load_img(p_t1),
            "action":    _zeros_action(),
            "gps":       torch.zeros(2),
            "domain_id": torch.tensor(self.domain_id, dtype=torch.long),
            "domain":    self.domain,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Multi-domain loader factory
# ═══════════════════════════════════════════════════════════════════════════

DOMAIN_CLASSES = {
    "recon":   RECONDomain,
    "ogbench": OGBenchDomain,
    "tworoom": TwoRoomDomain,
    "pusht":   PushTDomain,
    "smap":    SMAPDomain,
    "mvtec":   MVTecDomain,
}

# Weight each domain equally regardless of dataset size
DOMAIN_WEIGHTS = {
    "recon":   1.0,
    "ogbench": 1.0,
    "tworoom": 1.0,
    "pusht":   1.0,
    "smap":    1.0,
    "mvtec":   1.0,
}


def build_multi_domain_loader(
    domains:      List[str],
    domain_roots: Dict[str, str],
    batch_size:   int   = 16,
    num_workers:  int   = 2,
    max_files:    Optional[Dict[str, int]] = None,
    balance:      bool  = True,
) -> DataLoader:
    """
    Build a combined DataLoader that samples uniformly across all domains.

    Parameters
    ----------
    domains       : list of domain names to include
    domain_roots  : {domain_name: path_to_data_dir}
    batch_size    : samples per batch (mixed across domains)
    balance       : if True, use WeightedRandomSampler to equalise domains
    max_files     : {domain: max_files} to limit dataset size per domain

    Returns
    -------
    DataLoader yielding batches from all domains mixed
    """
    datasets, weights = [], []
    max_files = max_files or {}

    for domain in domains:
        if domain not in domain_roots:
            print(f"WARNING: No root path for domain '{domain}' — skipping")
            continue
        if domain not in DOMAIN_CLASSES:
            print(f"WARNING: No dataset class for '{domain}' — skipping")
            continue

        root = domain_roots[domain]
        if not Path(root).exists():
            print(f"WARNING: {root} does not exist — skipping '{domain}'")
            continue

        cls = DOMAIN_CLASSES[domain]
        kw  = {"data_root": root}
        if max_files.get(domain):
            kw["max_files"] = max_files[domain]

        ds = cls(**kw)
        datasets.append(ds)
        weights.extend([DOMAIN_WEIGHTS.get(domain, 1.0)] * len(ds))
        print(f"  {domain:12s}: {len(ds):>7,} samples  ({root})")

    if not datasets:
        raise ValueError("No valid domains found. Check domain_roots paths.")

    combined = ConcatDataset(datasets)
    print(f"\nTotal: {len(combined):,} samples across {len(datasets)} domains")

    sampler = None
    if balance and len(datasets) > 1:
        w = torch.tensor(weights, dtype=torch.float)
        sampler = WeightedRandomSampler(w, num_samples=len(combined),
                                         replacement=True)

    return DataLoader(
        combined,
        batch_size  = batch_size,
        sampler     = sampler,
        shuffle     = (sampler is None),
        num_workers = num_workers,
        pin_memory  = False,
        drop_last   = True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Quick check
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--recon-dir",   default="recon_data/recon_release")
    p.add_argument("--ogbench-dir", default="ogbench_data")
    p.add_argument("--pusht-dir",   default="pusht_data")
    p.add_argument("--smap-dir",    default="smap")
    p.add_argument("--mvtec-dir",   default="mvtec")
    args = p.parse_args()

    domain_roots = {
        "recon":   args.recon_dir,
        "ogbench": args.ogbench_dir,
        "pusht":   args.pusht_dir,
        "smap":    args.smap_dir,
        "mvtec":   args.mvtec_dir,
    }

    # Only load domains whose paths exist
    active = [d for d, r in domain_roots.items() if Path(r).exists()]
    print(f"Active domains: {active}")

    if active:
        loader = build_multi_domain_loader(
            domains      = active,
            domain_roots = domain_roots,
            batch_size   = 2,
            max_files    = {d: 2 for d in active},
        )
        batch = next(iter(loader))
        print(f"\nBatch shapes:")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k:12s}: {tuple(v.shape)}")
            else:
                print(f"  {k:12s}: {v}")
