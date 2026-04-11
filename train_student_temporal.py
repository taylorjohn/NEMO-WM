"""train_student_temporal.py — CORTEX-PE v16.17
═══════════════════════════════════════════════════════════════════════════════
Phase 1  Train StudentEncoder from scratch with temporal InfoNCE on RECON data.
Phase 2  Freeze encoder, train RoPETemporalHead on frozen 128-D latents.

Architecture
────────────
StudentEncoder (~46K params, features/proj naming):
  features: Conv2d(3→16,s2)+BN+ReLU → Conv2d(16→32,s2)+BN+ReLU
          → Conv2d(32→64,s2)+BN+ReLU → AdaptiveAvgPool2d(2,2)
  proj:     Linear(256→128)
  Output:   128-D L2-normalised latent

ProjectionHead (training only, discarded after Phase 1):
  Linear(128→256) + BN + ReLU + Linear(256→128)

Data
────
RECON HDF5: images/rgb_left (JPEG bytes), jackal/position (metres)
Pairs: (frame_t, frame_{t+k}),  k ∈ [k_min, k_max]
Raw pixels loaded per batch — encoder trained end-to-end.

Loss
────
Phase 1: InfoNCE (NT-Xent), τ=0.07
Phase 2: x-prediction (EMA target) + VICReg variance term

Checkpoint format
─────────────────
checkpoints/recon_student/student_best.pt
  {"step": int, "model": {"features.*": ..., "proj.*": ...}, "auroc": float}

Run sequence (recommended)
──────────────────────────
# 1. Probe only — verify data loads correctly
python train_student_temporal.py --data ./recon_data/recon_release --probe-only

# 2. Smoke test — 5 files, 3 epochs
python train_student_temporal.py --data ./recon_data/recon_release --max-files 5 --epochs 3

# 3. Full Phase 1 — 30 epochs, all files
python train_student_temporal.py --data ./recon_data/recon_release --epochs 30

# 4. Phase 2 — train RoPE head on frozen encoder
python train_student_temporal.py --data ./recon_data/recon_release --phase2 --epochs 20 \\
  --encoder-ckpt ./checkpoints/recon_student/student_best.pt

Success criteria
────────────────
Phase 1: InfoNCE loss < 3.0 by epoch 5, < 1.5 by epoch 20
Phase 2: AUROC > 0.80, loss < 1.0 by epoch 10
"""

from __future__ import annotations

import argparse
import io
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

# ── numpy compat (1.x vs 2.x) ──────────────────────────────────────────────
_np_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)

# ─────────────────────────────────────────────────────────────────────────────
# Architecture
# ─────────────────────────────────────────────────────────────────────────────

class StudentEncoder(nn.Module):
    """Lightweight 128-D CNN encoder for NPU deployment.

    Naming follows features/proj convention for ONNX/XINT8 export
    compatibility. No bias on conv layers (matches Quark quantisation path).

    Parameter count  ~46K:
      Conv 3→16:     3 × 16 × 9              =   432
      BN 16:         16 × 2                  =    32
      Conv 16→32:    16 × 32 × 9             = 4,608
      BN 32:         32 × 2                  =    64
      Conv 32→64:    32 × 64 × 9             = 18,432
      BN 64:         64 × 2                  =   128
      AvgPool(2,2) → 64 × 4 = 256
      Linear 256→128 + bias                  = 33,024
      ─────────────────────────────────────────────────
      Total                                  ≈ 56,720
    """
    LATENT_DIM = 128

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,  16, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(16),  nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.proj = nn.Linear(64 * 2 * 2, self.LATENT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(self.features(x).flatten(1))
        return F.normalize(z, dim=-1)   # unit-sphere: cosine sim = dot product

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ProjectionHead(nn.Module):
    """2-layer MLP projector — used only during Phase 1, discarded after.

    SimCLR/MoCo finding: projecting to a higher-dim space then discarding
    the projector preserves more useful information in the encoder itself.
    """
    def __init__(self, in_dim: int = 128, hidden: int = 256, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(z), dim=-1)


class RoPETemporalHead(nn.Module):
    """Phase 2 head — 3D RoPE attention on (z_t, z_{t+k}) pairs.

    Imported from rope_temporal_head.py if available, otherwise
    falls back to the inline MLP baseline.
    """
    pass   # populated at runtime — see _build_phase2_head()


# ─────────────────────────────────────────────────────────────────────────────
# InfoNCE loss
# ─────────────────────────────────────────────────────────────────────────────

def infonce_loss(z_a: torch.Tensor, z_b: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """NT-Xent loss over a batch of (anchor, positive) pairs.

    Both z_a and z_b must be L2-normalised.
    For a batch of N pairs:
      logits[i,j] = dot(z_a[i], z_b[j]) / tau
      loss = -mean(log-softmax on diagonal)
    """
    N = z_a.size(0)
    logits = torch.mm(z_a, z_b.T) / tau          # [N, N]
    labels = torch.arange(N, device=z_a.device)  # diagonal is the positive
    return F.cross_entropy(logits, labels)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset — Phase 1 (raw pixels, encoder trains end-to-end)
# ─────────────────────────────────────────────────────────────────────────────

# Augmentation: mild spatial resize + colour jitter on both views.
# We do NOT use heavy augmentations (random crops, grayscale) because
# the temporal gap *is* the supervision signal — the model must learn
# what changes between frames, not be invariant to everything.
_PIXEL_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_EVAL_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _load_hdf(path: Path) -> dict | None:
    """Load one RECON HDF5 file → raw JPEG bytes + positions."""
    try:
        import h5py
        with h5py.File(path, "r") as f:
            imgs = f["images/rgb_left"][:]   # JPEG bytes as fixed-len strings
            pos  = f["jackal/position"][:]   # [T, 3]
        return {"imgs": imgs, "pos": pos[:, :2]}
    except Exception:
        return None


class RECONPixelDataset(Dataset):
    """Phase 1 dataset — returns same-trajectory triplets for temporal InfoNCE.

    Each item: (anchor_t, pos_{t+k_near}, neg_{t+k_far}) from the SAME trajectory.

    Why triplets, not pairs?
    ────────────────────────
    Cross-trajectory InfoNCE teaches trajectory identity, not temporal ordering.
    The encoder learns "traj A looks like X" and gets loss ≈ 0 without ever
    learning that frame_5 should be closer to frame_6 than to frame_60.

    Same-trajectory triplets force temporal ordering: the encoder MUST learn
    that nearby frames are more similar than distant ones because the hard
    negative (k_far) comes from the same scene/lighting/texture as the anchor.

    Loss: NT-Xent where positive = pos_near, negatives = all other neg_far in batch.
    Concretely, a batch of B triplets gives:
      z_anchor [B, D], z_pos [B, D], z_neg [B, D]
    Loss = InfoNCE(z_anchor, z_pos) with z_neg as additional in-batch negatives.

    AUROC evaluation: cosine_sim(z_anchor, z_pos) vs cosine_sim(z_anchor, z_neg).
    """

    T_MAX        = 70
    XY_SCALE     = 10.0
    CLOSE_THRESH = 1.0    # metres — positive if disp < this
    FAR_THRESH   = 2.5    # metres — hard negative if disp > this

    # k ranges: near = temporal positive, far = hard negative
    K_NEAR_MIN = 1
    K_NEAR_MAX = 5    # ≤5 steps = ~1.25s at 4Hz = close in time
    K_FAR_MIN  = 15   # ≥15 steps = ~3.75s at 4Hz = far in time
    K_FAR_MAX  = 40   # allow up to 40 steps for harder negatives

    def __init__(
        self,
        hdf_files: list[Path],
        k_min: int   = 1,    # kept for API compat, overridden by K_NEAR_MIN
        k_max: int   = 15,   # kept for API compat, overridden by K_NEAR_MAX
        n_pairs: int = 8000,
        augment: bool = True,
        seed: int    = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        self._transform = _PIXEL_TRANSFORM if augment else _EVAL_TRANSFORM

        # Load all trajectories
        trajs: list[dict] = []
        for p in hdf_files:
            raw = _load_hdf(p)
            if raw is not None:
                trajs.append(raw)
        self._trajs = trajs

        # Sample triplets — store (traj_idx, t_anchor, t_pos, t_neg)
        self._triplets: list[tuple] = []
        attempts = 0
        while len(self._triplets) < n_pairs and attempts < n_pairs * 20:
            attempts += 1
            if not trajs:
                break
            ti = int(rng.integers(len(trajs)))
            T  = len(trajs[ti]["imgs"])
            # Need enough room for anchor + k_far_max steps
            if T < self.K_NEAR_MIN + self.K_FAR_MIN + 2:
                continue
            t_anchor = int(rng.integers(0, max(1, T - self.K_FAR_MIN - 1)))
            # Positive: near step
            k_near_hi = min(self.K_NEAR_MAX, T - t_anchor - self.K_FAR_MIN - 1)
            if k_near_hi < self.K_NEAR_MIN:
                continue
            k_near = int(rng.integers(self.K_NEAR_MIN, k_near_hi + 1))
            t_pos  = t_anchor + k_near
            # Negative: far step from anchor (not from positive)
            t_neg_min = t_anchor + self.K_FAR_MIN
            t_neg_max = min(t_anchor + self.K_FAR_MAX, T - 1)
            if t_neg_max < t_neg_min:
                continue
            t_neg = int(rng.integers(t_neg_min, t_neg_max + 1))
            self._triplets.append((ti, t_anchor, t_pos, t_neg))

        # Compute displacement stats
        pos_disps = []; neg_disps = []
        for ti, ta, tp, tn in self._triplets:
            pos_xy = trajs[ti]["pos"]
            pos_disps.append(float(np.linalg.norm(pos_xy[ta] - pos_xy[tp])))
            neg_disps.append(float(np.linalg.norm(pos_xy[ta] - pos_xy[tn])))
        pd_m = float(np.mean(pos_disps)) if pos_disps else 0.
        nd_m = float(np.mean(neg_disps)) if neg_disps else 0.
        print(f"  {len(self._triplets)} triplets | "
              f"pos_disp={pd_m:.2f}m  neg_disp={nd_m:.2f}m  "
              f"(same-trajectory hard negatives)")

        # Pre-cache all unique frames as uint8 numpy
        needed = set()
        for ti, ta, tp, tn in self._triplets:
            needed.add((ti, ta)); needed.add((ti, tp)); needed.add((ti, tn))
        from PIL import Image as _PIL_Image
        print(f"  Pre-caching {len(needed)} unique frames...", end=" ", flush=True)
        t0 = time.time()
        self._frame_cache: dict[tuple, np.ndarray] = {}
        for ti, t in needed:
            raw = bytes(trajs[ti]["imgs"][t])
            self._frame_cache[(ti, t)] = np.array(
                _PIL_Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8
            )
        print(f"{time.time()-t0:.1f}s")

    def _decode(self, ti: int, t: int) -> torch.Tensor:
        from PIL import Image
        cached = self._frame_cache.get((ti, t))
        img = Image.fromarray(cached) if cached is not None else               Image.open(io.BytesIO(bytes(self._trajs[ti]["imgs"][t]))).convert("RGB")
        return self._transform(img)

    def __len__(self) -> int:
        return len(self._triplets)

    def __getitem__(self, i: int):
        ti, ta, tp, tn = self._triplets[i]
        return self._decode(ti, ta), self._decode(ti, tp), self._decode(ti, tn)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset — Phase 2 (pre-encoded latents, head trains on frozen encoder)
# ─────────────────────────────────────────────────────────────────────────────

class RECONLatentDataset(Dataset):
    """Phase 2 dataset — pre-encodes frames with frozen encoder.

    Same structure as RECONPairDataset in train_rope_temporal.py
    but uses the new StudentEncoder architecture.
    """

    T_MAX        = 70
    XY_SCALE     = 10.0
    CLOSE_THRESH = 1.0
    FAR_THRESH   = 3.0

    def __init__(
        self,
        hdf_files: list[Path],
        encoder: nn.Module,
        k_min: int   = 1,
        k_max: int   = 15,
        n_pairs: int = 8000,
        device: torch.device = torch.device("cpu"),
        seed: int    = 42,
    ) -> None:
        rng  = np.random.default_rng(seed)
        enc  = encoder.to(device).eval()
        self._trajs: list[dict] = []

        from PIL import Image
        print(f"  Encoding {len(hdf_files)} trajectories...")
        t0 = time.time()

        for path in hdf_files:
            raw = _load_hdf(path)
            if raw is None:
                continue
            T        = len(raw["imgs"])
            origin   = raw["pos"][0].copy()
            pos_xy   = (raw["pos"] - origin) / self.XY_SCALE

            latents = np.zeros((T, StudentEncoder.LATENT_DIM), dtype=np.float32)
            for t in range(T):
                try:
                    img = Image.open(io.BytesIO(bytes(raw["imgs"][t]))).convert("RGB")
                    inp = _EVAL_TRANSFORM(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        latents[t] = enc(inp).squeeze(0).cpu().numpy()
                except Exception:
                    latents[t] = latents[t - 1] if t > 0 else 0.0

            self._trajs.append({"latents": latents, "pos_xy": pos_xy, "T": T})

        print(f"  Encoded {len(self._trajs)} trajectories in {time.time()-t0:.1f}s")

        self._pairs: list[dict] = []
        for _ in range(n_pairs):
            if not self._trajs:
                break
            tr  = self._trajs[rng.integers(len(self._trajs))]
            T   = tr["T"]
            t   = int(rng.integers(0, max(1, T - k_min)))
            k_hi = min(k_max, T - t - 1)
            if k_hi < k_min:
                continue
            k   = int(rng.integers(k_min, k_hi + 1))
            tk  = t + k

            pos_t  = tr["pos_xy"][t]
            pos_tk = tr["pos_xy"][tk]
            disp   = float(np.linalg.norm(pos_t - pos_tk) * self.XY_SCALE)

            c_t  = np.array([t  / self.T_MAX, pos_t[0],  pos_t[1]],  dtype=np.float32)
            c_tk = np.array([tk / self.T_MAX, pos_tk[0], pos_tk[1]], dtype=np.float32)

            label = (1 if disp < self.CLOSE_THRESH else
                     0 if disp > self.FAR_THRESH   else -1)

            self._pairs.append({
                "z_t":   tr["latents"][t].copy(),
                "z_tk":  tr["latents"][tk].copy(),
                "c_t":   c_t, "c_tk": c_tk,
                "eval_label": label, "disp": disp, "k": k,
            })

        labels = np.array([p["eval_label"] for p in self._pairs])
        cd = np.mean([p["disp"] for p in self._pairs if p["eval_label"] == 1]) if (labels == 1).any() else 0.
        fd = np.mean([p["disp"] for p in self._pairs if p["eval_label"] == 0]) if (labels == 0).any() else 0.
        print(f"  {len(self._pairs)} pairs | "
              f"close={(labels==1).sum()} far={(labels==0).sum()} ambig={(labels==-1).sum()} | "
              f"close_disp={cd:.2f}m far_disp={fd:.2f}m")

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, i: int):
        p = self._pairs[i]
        return (torch.from_numpy(p["z_t"]),  torch.from_numpy(p["z_tk"]),
                torch.from_numpy(p["c_t"]),  torch.from_numpy(p["c_tk"]),
                torch.tensor(p["eval_label"], dtype=torch.float32))


# ─────────────────────────────────────────────────────────────────────────────
# AUROC
# ─────────────────────────────────────────────────────────────────────────────

def auroc_np(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under ROC curve. scores = cosine similarity (higher = closer)."""
    labels = np.asarray(labels, dtype=float)
    mask   = labels >= 0
    scores = np.asarray(scores)[mask]
    labels = labels[mask]
    if len(labels) == 0 or labels.sum() == 0 or (1 - labels).sum() == 0:
        return 0.5
    order  = np.argsort(scores)
    tpr = [0.]; fpr = [0.]; tp = fp = 0
    n_pos = labels.sum(); n_neg = len(labels) - n_pos
    for l in labels[order]:
        if l == 1: tp += 1
        else:      fp += 1
        tpr.append(tp / n_pos); fpr.append(fp / n_neg)
    return float(_np_trapz(tpr, fpr))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 evaluation — encode held-out frames, measure cosine AUROC
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def build_val_frame_cache(
    hdf_files: list[Path], max_files: int = 100, seed: int = 77
) -> list[dict]:
    """Pre-decode val frames as uint8 numpy — done ONCE per training run.

    Returns a list of trajectory dicts:
        {"frames": [np.ndarray uint8, ...], "pos": np.ndarray [T,2]}
    Limited to max_files for speed.
    """
    from PIL import Image as _PIL_Image
    rng = np.random.default_rng(seed)
    files = list(hdf_files)
    if len(files) > max_files:
        idx = rng.choice(len(files), max_files, replace=False)
        files = [files[i] for i in idx]

    trajs = []
    for p in files:
        raw = _load_hdf(p)
        if raw is None:
            continue
        T = len(raw["imgs"])
        frames = []
        for t in range(T):
            try:
                arr = np.array(
                    _PIL_Image.open(io.BytesIO(bytes(raw["imgs"][t]))).convert("RGB"),
                    dtype=np.uint8,
                )
            except Exception:
                arr = frames[-1] if frames else np.zeros((224, 224, 3), np.uint8)
            frames.append(arr)
        trajs.append({"frames": frames, "pos": raw["pos"], "T": T})
    return trajs


@torch.no_grad()
def eval_phase1(
    encoder: nn.Module,
    projector: nn.Module,
    val_cache: list[dict],
    k_min: int, k_max: int, n_pairs: int,
    device: torch.device, seed: int = 99,
) -> float:
    """AUROC on projector output using pre-cached val frames.

    Re-encodes frames with the *current* encoder each call (weights change
    each epoch), but avoids HDF5 seek + JPEG decode by using the frame cache.
    AUROC is measured on projector output — what InfoNCE actually trains.
    """
    from PIL import Image as _PIL_Image
    encoder.eval(); projector.eval()
    rng = np.random.default_rng(seed)

    CLOSE = RECONPixelDataset.CLOSE_THRESH
    FAR   = RECONPixelDataset.FAR_THRESH
    scores: list[float] = []
    labels_list: list[int] = []

    attempts = 0
    while len(scores) < n_pairs and attempts < n_pairs * 10:
        attempts += 1
        tr  = val_cache[rng.integers(len(val_cache))]
        T   = tr["T"]
        t   = int(rng.integers(0, max(1, T - k_min)))
        k_hi = min(k_max, T - t - 1)
        if k_hi < k_min:
            continue
        k   = int(rng.integers(k_min, k_hi + 1))
        tk  = t + k
        disp = float(np.linalg.norm(tr["pos"][t] - tr["pos"][tk]))
        # Sample hard negative: far frame from same trajectory
        t_neg_min = t + RECONPixelDataset.K_FAR_MIN
        t_neg_max = min(t + RECONPixelDataset.K_FAR_MAX, T - 1)
        if t_neg_max < t_neg_min:
            continue
        t_neg = int(rng.integers(t_neg_min, t_neg_max + 1))
        try:
            img_a = _EVAL_TRANSFORM(_PIL_Image.fromarray(tr["frames"][t])).unsqueeze(0).to(device)
            img_p = _EVAL_TRANSFORM(_PIL_Image.fromarray(tr["frames"][tk])).unsqueeze(0).to(device)
            img_n = _EVAL_TRANSFORM(_PIL_Image.fromarray(tr["frames"][t_neg])).unsqueeze(0).to(device)
            z_a = encoder(img_a)
            z_p = encoder(img_p)
            z_n = encoder(img_n)
            # Score = pos_sim - neg_sim: positive = encoder learned ordering
            sim = float(F.cosine_similarity(z_a, z_p).item()) - float(F.cosine_similarity(z_a, z_n).item())
        except Exception:
            continue
        scores.append(sim)
        labels_list.append(1)   # always a valid triplet; score > 0 = correct

    # Fraction of triplets where pos_sim > neg_sim (score > 0)
    # Random encoder: ~0.5; perfect: ~1.0
    scores_arr = np.array(scores)
    return float((scores_arr > 0).mean()) if len(scores_arr) > 0 else 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — MLP baseline head (no RoPE dependency required)
# ─────────────────────────────────────────────────────────────────────────────

class MLPHead(nn.Module):
    """Simple MLP temporal head for Phase 2 baseline."""
    def __init__(self, in_dim: int = StudentEncoder.LATENT_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, 64),
        )

    def forward(self, z: torch.Tensor, *_, **__) -> torch.Tensor:
        return self.net(z)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _build_phase2_head(args, device: torch.device) -> tuple[nn.Module, nn.Module | None]:
    """Try to import RoPETemporalHead; fall back to MLPHead."""
    try:
        import sys
        sys.path.insert(0, str(Path(args.data).parent.parent))
        from rope_temporal_head import RoPETemporalHead as _RoPE, infonce_loss as _inf
        rope = _RoPE(latent_dim=StudentEncoder.LATENT_DIM,
                     embed_dim=args.embed_dim).to(device)
        print(f"RoPETemporalHead loaded  params={rope.n_params:,}")
        return rope, _inf
    except ImportError:
        mlp = MLPHead().to(device)
        print(f"RoPETemporalHead unavailable — using MLPHead  params={mlp.n_params:,}")
        return mlp, None


def _phase2_infonce(head, z_t, z_tk, c_t, c_tk, loss_fn):
    """InfoNCE via cosine distance on head embeddings."""
    if loss_fn is not None:
        try:
            e_t  = head(z_t,  c_t)
            e_tk = head(z_tk, c_tk)
            return loss_fn(e_t, e_tk)
        except Exception:
            pass
    e_t  = F.normalize(head(z_t),  dim=-1)
    e_tk = F.normalize(head(z_tk), dim=-1)
    return infonce_loss(e_t, e_tk)


@torch.no_grad()
def eval_phase2(head, loader, device) -> tuple[float, float, float]:
    head.eval()
    scores: list[float] = []; labels: list[int] = []
    close_d: list[float] = []; far_d: list[float] = []
    for z_t, z_tk, c_t, c_tk, lab in loader:
        z_t  = z_t.to(device); z_tk = z_tk.to(device)
        c_t  = c_t.to(device); c_tk = c_tk.to(device)
        try:
            e_t  = head(z_t,  c_t)
            e_tk = head(z_tk, c_tk)
        except Exception:
            e_t  = head(z_t)
            e_tk = head(z_tk)
        e_t  = F.normalize(e_t,  dim=-1)
        e_tk = F.normalize(e_tk, dim=-1)
        sim  = -(e_t * e_tk).sum(-1).cpu().numpy()  # distance: low=close, matches auroc_np convention
        lab_np = lab.numpy().astype(int)
        for s, l in zip(sim, lab_np):
            if l == 1:  close_d.append(float(s))
            elif l == 0: far_d.append(float(s))
            labels.append(l); scores.append(float(s))
    auc = auroc_np(np.array(scores), np.array(labels))
    close_m = float(np.mean(close_d)) if close_d else 0.
    far_m   = float(np.mean(far_d))   if far_d   else 0.
    return auc, close_m, far_m


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_phase1(args, encoder: nn.Module, train_files: list[Path],
                 val_files: list[Path], device: torch.device,
                 out_dir: Path) -> None:
    """Train encoder + projection head with temporal InfoNCE."""

    print("\n── Phase 1: Temporal InfoNCE encoder training ──────────────────────")
    projector = ProjectionHead(in_dim=StudentEncoder.LATENT_DIM).to(device)

    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=args.lr,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    print("Building train dataset (raw pixels):")
    train_ds = RECONPixelDataset(
        train_files, k_min=args.k_min, k_max=args.k_max,
        n_pairs=args.n_pairs, augment=True, seed=args.seed,
    )
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=0, drop_last=True)

    # ── Build val frame cache (done once, re-encoded each eval epoch) ────
    n_val_eval = min(getattr(args, "val_eval_files", 100), len(val_files))
    eval_every = getattr(args, "eval_every", 5)
    print(f"\nBuilding val frame cache ({n_val_eval} files, eval every {eval_every} epochs)...")
    t_cache = time.time()
    val_cache = build_val_frame_cache(val_files, max_files=n_val_eval, seed=77)
    print(f"Val cache ready: {len(val_cache)} trajs  ({time.time()-t_cache:.1f}s)")

    n_eval_pairs = min(500, args.n_pairs // 4)
    init_auroc = eval_phase1(encoder, projector, val_cache,
                             args.k_min, args.k_max, n_eval_pairs, device)
    print(f"Init encoder AUROC = {init_auroc:.4f}  (measured on encoder, projector discarded after training)")

    best_auroc = init_auroc
    best_path  = out_dir / "student_best.pt"

    print(f"\n{'Ep':>4}  {'Loss':>8}  {'LR':>8}  {'AUROC':>8}")
    print("─" * 38)

    for epoch in range(1, args.epochs + 1):
        encoder.train(); projector.train()
        t0 = time.time(); total_loss = 0.; n_batches = 0

        for img_anchor, img_pos, img_neg in train_ld:
            img_anchor = img_anchor.to(device)
            img_pos    = img_pos.to(device)
            img_neg    = img_neg.to(device)

            z_a = projector(encoder(img_anchor))
            z_p = projector(encoder(img_pos))
            z_n = projector(encoder(img_neg))

            B = z_a.size(0)
            logits_pos = (z_a * z_p).sum(-1, keepdim=True)
            logits_neg = torch.mm(z_a, z_n.T)
            logits  = torch.cat([logits_pos, logits_neg], dim=1) / args.tau
            targets = torch.zeros(B, dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, targets)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(projector.parameters()), 1.0
            )
            opt.step()

            total_loss += loss.item()
            n_batches  += 1

        sched.step()
        avg_loss = total_loss / max(n_batches, 1)
        cur_lr   = sched.get_last_lr()[0]
        dt  = time.time() - t0

        # AUROC on projector output (eval_every epochs, or final epoch)
        if epoch % eval_every == 0 or epoch == args.epochs:
            auc = eval_phase1(encoder, projector, val_cache,
                              args.k_min, args.k_max, n_eval_pairs, device)
        else:
            auc = best_auroc   # carry forward between evals

        marker = " ★" if auc > best_auroc else ""
        eval_marker = " [eval]" if (epoch % eval_every == 0 or epoch == args.epochs) else ""
        print(f"{epoch:>4}  {avg_loss:>8.4f}  {cur_lr:>8.6f}  {auc:>8.4f}  ({dt:.1f}s){marker}{eval_marker}")

        if auc > best_auroc:
            best_auroc = auc
            _save_encoder(encoder, epoch, best_auroc, best_path)

    # Always save final
    final_path = out_dir / "student_final.pt"
    _save_encoder(encoder, args.epochs, best_auroc, final_path)

    print(f"\nPhase 1 complete — best AUROC={best_auroc:.4f}")
    print(f"Checkpoint: {best_path}")


def train_phase2(args, encoder: nn.Module, train_files: list[Path],
                 val_files: list[Path], device: torch.device,
                 out_dir: Path) -> None:
    """Freeze encoder, train RoPETemporalHead on 128-D latents."""

    print("\n── Phase 2: RoPE head training (frozen encoder) ────────────────────")
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    head, rope_loss_fn = _build_phase2_head(args, device)

    print("\nBuilding train dataset (pre-encoded latents):")
    train_ds = RECONLatentDataset(
        train_files, encoder, k_min=args.k_min, k_max=args.k_max,
        n_pairs=args.n_pairs, device=device, seed=args.seed,
    )
    print("Building val dataset:")
    val_ds = RECONLatentDataset(
        val_files, encoder, k_min=args.k_min, k_max=args.k_max,
        n_pairs=max(500, args.n_pairs // 8), device=device, seed=args.seed + 1,
    )
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=0, drop_last=True)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                          num_workers=0)

    opt   = torch.optim.Adam(head.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    init_auc, close0, far0 = eval_phase2(head, val_ld, device)
    print(f"\nInit head AUROC={init_auc:.4f}  close={close0:.4f}  far={far0:.4f}")

    best_auroc = init_auc
    best_path  = out_dir / "rope_head_best.pt"

    print(f"\n{'Ep':>4}  {'Loss':>8}  {'AUROC':>8}  {'Sep':>8}")
    print("─" * 38)

    for epoch in range(1, args.epochs + 1):
        head.train()
        t0 = time.time(); total_loss = 0.; n_batches = 0

        for z_t, z_tk, c_t, c_tk, _lab in train_ld:
            z_t  = z_t.to(device);  z_tk = z_tk.to(device)
            c_t  = c_t.to(device);  c_tk = c_tk.to(device)

            loss = _phase2_infonce(head, z_t, z_tk, c_t, c_tk, rope_loss_fn)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            n_batches  += 1

        sched.step()
        avg_loss = total_loss / max(n_batches, 1)

        auc, close_m, far_m = eval_phase2(head, val_ld, device)
        sep = far_m - close_m
        dt  = time.time() - t0

        marker = " ★" if auc > best_auroc else ""
        print(f"{epoch:>4}  {avg_loss:>8.4f}  {auc:>8.4f}  {sep:>+8.4f}  ({dt:.1f}s){marker}")

        if auc > best_auroc:
            best_auroc = auc
            torch.save({"epoch": epoch, "auroc": auc,
                        "model": head.state_dict()}, best_path)

    print(f"\nPhase 2 complete — best AUROC={best_auroc:.4f}")
    print(f"Checkpoint: {best_path}")


def _save_encoder(encoder: nn.Module, step: int, auroc: float, path: Path) -> None:
    torch.save({
        "step":  step,
        "auroc": auroc,
        "model": encoder.state_dict(),   # keys: features.*, proj.*
    }, path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="CORTEX-PE StudentEncoder temporal training")

    # Data
    ap.add_argument("--data",       required=True,    help="Path to recon_release/ directory")
    ap.add_argument("--max-files",  type=int,         default=None,  help="Limit HDF5 files (smoke test)")
    ap.add_argument("--val-frac",   type=float,       default=0.2,   help="Fraction of files for val")

    # Pair sampling
    ap.add_argument("--k-min",      type=int,         default=1)
    ap.add_argument("--k-max",      type=int,         default=15)
    ap.add_argument("--n-pairs",    type=int,         default=8000)

    # Training
    ap.add_argument("--epochs",     type=int,         default=30)
    ap.add_argument("--batch",      type=int,         default=128)
    ap.add_argument("--lr",         type=float,       default=3e-4)
    ap.add_argument("--tau",        type=float,       default=0.07,  help="InfoNCE temperature")
    ap.add_argument("--seed",       type=int,         default=42)

    # Phase 2
    ap.add_argument("--phase2",     action="store_true",  help="Run Phase 2 after Phase 1")
    ap.add_argument("--phase2-only", action="store_true", help="Skip Phase 1, only run Phase 2")
    ap.add_argument("--encoder-ckpt", type=str,       default=None,
                    help="Pre-trained encoder for Phase 2 only")
    ap.add_argument("--embed-dim",  type=int,         default=96,    help="RoPE head embed dim")

    # Misc
    ap.add_argument("--out-dir",    type=str,         default="checkpoints/recon_student")
    ap.add_argument("--probe-only", action="store_true", help="Load data, print stats, exit")
    ap.add_argument("--val-eval-files", type=int,     default=100,
                    help="Val files to cache for AUROC eval (default 100)")
    ap.add_argument("--eval-every",     type=int,     default=5,
                    help="Evaluate AUROC every N epochs (default 5)")

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Discover files ────────────────────────────────────────────────────
    data_dir = Path(args.data)
    all_files = sorted(data_dir.rglob("*.hdf5")) + sorted(data_dir.rglob("*.h5"))
    if not all_files:
        raise FileNotFoundError(f"No HDF5 files found in {data_dir}")
    if args.max_files:
        all_files = all_files[:args.max_files]

    rng       = np.random.default_rng(args.seed)
    perm      = rng.permutation(len(all_files))
    n_val     = max(2, int(len(all_files) * args.val_frac))
    val_files  = [all_files[i] for i in perm[:n_val]]
    train_files = [all_files[i] for i in perm[n_val:]]

    print(f"Found {len(all_files)} HDF5 files")
    print(f"Train: {len(train_files)} files | Val: {len(val_files)} files")

    # ── Probe mode ────────────────────────────────────────────────────────
    if args.probe_only:
        print("\n── Probe mode ───────────────────────────────────────────────────────")
        print("Checking first 3 train files...")
        from PIL import Image
        for p in train_files[:3]:
            raw = _load_hdf(p)
            if raw is None:
                print(f"  FAIL: {p.name}")
                continue
            T = len(raw["imgs"])
            img = Image.open(io.BytesIO(bytes(raw["imgs"][0]))).convert("RGB")
            print(f"  OK: {p.name}  T={T}  pos_shape={raw['pos'].shape}  "
                  f"img_size={img.size}")
        enc = StudentEncoder()
        print(f"\nStudentEncoder  params={enc.n_params:,}  output_dim={enc.LATENT_DIM}")
        dummy = torch.zeros(1, 3, 224, 224)
        out   = enc(dummy)
        print(f"Forward pass OK  output={tuple(out.shape)}  "
              f"norm={out.norm().item():.4f} (should be ~1.0)")
        print("\n✅ Probe passed — ready to train")
        return

    # ── Build encoder ─────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder = StudentEncoder().to(device)
    print(f"\nStudentEncoder  params={encoder.n_params:,}")

    if args.phase2_only or (args.phase2 and args.encoder_ckpt):
        ckpt_path = args.encoder_ckpt or str(out_dir / "student_best.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        encoder.load_state_dict(ckpt["model"])
        prev_auroc = ckpt.get("auroc", "?")
        print(f"Loaded encoder from {ckpt_path}  (auroc={prev_auroc})")

    # ── Phase 1 ───────────────────────────────────────────────────────────
    if not args.phase2_only:
        train_phase1(args, encoder, train_files, val_files, device, out_dir)

    # ── Phase 2 ───────────────────────────────────────────────────────────
    if args.phase2 or args.phase2_only:
        # Reload best checkpoint from Phase 1 for Phase 2
        if not args.phase2_only:
            best = out_dir / "student_best.pt"
            if best.exists():
                ckpt = torch.load(best, map_location="cpu", weights_only=True)
                encoder.load_state_dict(ckpt["model"])
                print(f"\nPhase 2: loaded best encoder (auroc={ckpt.get('auroc','?'):.4f})")
        train_phase2(args, encoder, train_files, val_files, device, out_dir)

    print("\n══════════════════════════════════════════════════")
    print(f"Done. Checkpoints in: {out_dir}")
    print(f"  student_best.pt   — best Phase 1 encoder (features.* / proj.*)")
    print(f"  student_final.pt  — final Phase 1 encoder")
    if args.phase2 or args.phase2_only:
        print(f"  rope_head_best.pt — best Phase 2 head")
    print("══════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
