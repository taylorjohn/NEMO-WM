"""eval_recon_quasimetric.py — CORTEX-PE v16.17
═══════════════════════════════════════════════════════════════════════════════
Held-out evaluation of the RECON StudentEncoder (Phase 1) and RoPETemporalHead
(Phase 2) trained by train_student_temporal.py.

Reports:
  1. Triplet ordering score — fraction of (anchor, pos_near, neg_far) triplets
     where encoder places pos closer than neg. Random = 0.50.
  2. Close/Far AUROC — cosine similarity discriminates close (<1m) from far (>3m)
     pairs. Both raw encoder and head-projected versions reported.
  3. Per-k breakdown — AUROC at short (k1-5), medium (k6-15), long (k16+) offsets.
  4. Displacement correlation — Spearman ρ between quasimetric distance and metres.

Usage:
  # Evaluate encoder only
  python eval_recon_quasimetric.py --data ./recon_data/recon_release

  # Evaluate encoder + RoPE head
  python eval_recon_quasimetric.py \\
    --data ./recon_data/recon_release \\
    --encoder-ckpt ./checkpoints/recon_student/student_best.pt \\
    --head-ckpt    ./checkpoints/recon_student/rope_head_best.pt

  # Held-out files only (last N files, sorted by name — not seen during training)
  python eval_recon_quasimetric.py \\
    --data ./recon_data/recon_release --max-files 200 --held-out-only
"""

from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

# ── numpy compat ────────────────────────────────────────────────────────────
_np_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)

# ─────────────────────────────────────────────────────────────────────────────
# StudentEncoder — must match train_student_temporal.py exactly
# ─────────────────────────────────────────────────────────────────────────────

class StudentEncoder(nn.Module):
    """128-D L2-normalised CNN encoder.  features/proj naming for NPU export."""
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
        return F.normalize(self.proj(self.features(x).flatten(1)), dim=-1)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

_EVAL_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

CLOSE_THRESH = 1.0   # metres
FAR_THRESH   = 3.0
K_NEAR_MAX   = 5
K_FAR_MIN    = 15
K_FAR_MAX    = 40


def _load_hdf(path: Path) -> dict | None:
    try:
        import h5py
        with h5py.File(path, "r") as f:
            imgs = f["images/rgb_left"][:]
            pos  = f["jackal/position"][:]
        return {"imgs": imgs, "pos": pos[:, :2]}
    except Exception:
        return None


def _decode_frame(arr: np.ndarray) -> torch.Tensor:
    from PIL import Image
    return _EVAL_TRANSFORM(Image.fromarray(arr))


def load_trajectories(hdf_files: list[Path], verbose: bool = True) -> list[dict]:
    """Load and pre-cache all frames as uint8 numpy."""
    from PIL import Image as _PIL
    trajs = []
    t0 = time.time()
    for p in hdf_files:
        raw = _load_hdf(p)
        if raw is None:
            continue
        T = len(raw["imgs"])
        frames = []
        for t in range(T):
            try:
                arr = np.array(
                    _PIL.open(io.BytesIO(bytes(raw["imgs"][t]))).convert("RGB"),
                    dtype=np.uint8,
                )
            except Exception:
                arr = frames[-1] if frames else np.zeros((224, 224, 3), np.uint8)
            frames.append(arr)
        trajs.append({"frames": frames, "pos": raw["pos"], "T": T})
    if verbose:
        print(f"  Loaded {len(trajs)} trajectories in {time.time()-t0:.1f}s")
    return trajs


# ─────────────────────────────────────────────────────────────────────────────
# Encode
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_trajectories(
    trajs: list[dict], encoder: nn.Module, device: torch.device
) -> list[dict]:
    """Encode all frames; return trajs with added 'latents' key."""
    encoder.eval()
    t0 = time.time()
    out = []
    for tr in trajs:
        T = tr["T"]
        lats = np.zeros((T, StudentEncoder.LATENT_DIM), dtype=np.float32)
        for t in range(T):
            try:
                inp = _decode_frame(tr["frames"][t]).unsqueeze(0).to(device)
                lats[t] = encoder(inp).squeeze(0).cpu().numpy()
            except Exception:
                lats[t] = lats[t - 1] if t > 0 else 0.0
        out.append({**tr, "latents": lats})
    print(f"  Encoded {len(out)} trajectories in {time.time()-t0:.1f}s")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def auroc_np(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=float)
    mask   = labels >= 0
    scores = np.asarray(scores)[mask]
    labels = labels[mask]
    if len(labels) == 0 or labels.sum() == 0 or (1 - labels).sum() == 0:
        return 0.5
    order = np.argsort(scores)
    tpr = [0.]; fpr = [0.]; tp = fp = 0
    n_pos = labels.sum(); n_neg = len(labels) - n_pos
    for l in labels[order]:
        if l == 1: tp += 1
        else:      fp += 1
        tpr.append(tp / n_pos); fpr.append(fp / n_neg)
    return float(_np_trapz(tpr, fpr))


def triplet_ordering_score(
    trajs: list[dict], n_triplets: int, seed: int
) -> float:
    """Fraction of (anchor, pos_near, neg_far) where pos_sim > neg_sim.
    Uses raw encoder latents (already L2-normalised — dot = cosine sim).
    """
    rng = np.random.default_rng(seed)
    correct = 0; total = 0; attempts = 0

    while total < n_triplets and attempts < n_triplets * 20:
        attempts += 1
        tr = trajs[rng.integers(len(trajs))]
        T  = tr["T"]
        if T < K_NEAR_MAX + K_FAR_MIN + 2:
            continue
        ta     = int(rng.integers(0, max(1, T - K_FAR_MIN - 1)))
        k_near_hi = min(K_NEAR_MAX, T - ta - K_FAR_MIN - 1)
        if k_near_hi < 1:
            continue
        tp     = ta + int(rng.integers(1, k_near_hi + 1))
        tn_max = min(ta + K_FAR_MAX, T - 1)
        if tn_max < ta + K_FAR_MIN:
            continue
        tn     = int(rng.integers(ta + K_FAR_MIN, tn_max + 1))

        z_a = tr["latents"][ta]
        z_p = tr["latents"][tp]
        z_n = tr["latents"][tn]
        sim_pos = float(np.dot(z_a, z_p))
        sim_neg = float(np.dot(z_a, z_n))
        if sim_pos > sim_neg:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.5


def close_far_auroc(
    trajs: list[dict], n_pairs: int, seed: int,
    head: nn.Module | None = None, device: torch.device = torch.device("cpu"),
    k_min: int = 1, k_max: int = 40,
) -> tuple[float, float, float]:
    """AUROC, mean_close_sim, mean_far_sim for close/far pairs.

    If head is None, uses raw encoder cosine sim (dot product on unit vectors).
    If head is provided, projects latents through head then measures cosine sim.
    Coords fed to head as [t/70, x/10, y/10] — head must accept (z, coord) or just (z).
    """
    rng = np.random.default_rng(seed)
    T_MAX   = 70
    XY_SCALE = 10.0
    origin = None

    scores: list[float] = []; labels: list[int] = []
    close_sims: list[float] = []; far_sims: list[float] = []
    attempts = 0

    while len(scores) < n_pairs and attempts < n_pairs * 20:
        attempts += 1
        tr  = trajs[rng.integers(len(trajs))]
        T   = tr["T"]
        t   = int(rng.integers(0, max(1, T - k_min)))
        k_hi = min(k_max, T - t - 1)
        if k_hi < k_min:
            continue
        k   = int(rng.integers(k_min, k_hi + 1))
        tk  = t + k
        disp = float(np.linalg.norm(tr["pos"][t] - tr["pos"][tk]))
        label = (1 if disp < CLOSE_THRESH else 0 if disp > FAR_THRESH else -1)
        if label < 0:
            continue

        z_t  = tr["latents"][t]
        z_tk = tr["latents"][tk]

        if head is not None:
            # Normalise position to trajectory origin
            origin = tr["pos"][0]
            pos_xy = (tr["pos"] - origin) / XY_SCALE
            c_t  = torch.tensor([t  / T_MAX, pos_xy[t,  0], pos_xy[t,  1]], dtype=torch.float32).unsqueeze(0).to(device)
            c_tk = torch.tensor([tk / T_MAX, pos_xy[tk, 0], pos_xy[tk, 1]], dtype=torch.float32).unsqueeze(0).to(device)
            zt_  = torch.from_numpy(z_t).unsqueeze(0).to(device)
            ztk_ = torch.from_numpy(z_tk).unsqueeze(0).to(device)
            with torch.no_grad():
                # RoPETemporalHead.forward(z_t, z_tk=None, c_t=None, c_tk=None)
                # Pair mode: returns out_proj(attn(z_t, z_tk)) → [B, out_dim]
                # Single mode (z_tk=None): returns out_proj(encoder(z_t)) → [B, out_dim]
                # For AUROC: embed both frames in single mode, measure cosine sim.
                e_t  = F.normalize(head(zt_),  dim=-1)   # [1, out_dim]
                e_tk = F.normalize(head(ztk_), dim=-1)   # [1, out_dim]
                sim = float(F.cosine_similarity(e_t, e_tk).item())
        else:
            sim = float(np.dot(z_t, z_tk))  # already unit vectors

        scores.append(-sim)  # auroc_np: ascending sort, low score = positive (close)
        labels.append(label)
        if label == 1:
            close_sims.append(sim)   # raw sim for reporting only
        else:
            far_sims.append(sim)

    auc       = auroc_np(np.array(scores), np.array(labels))
    close_m   = float(np.mean(close_sims)) if close_sims else 0.
    far_m     = float(np.mean(far_sims))   if far_sims   else 0.
    return auc, close_m, far_m


def per_k_auroc(
    trajs: list[dict], n_pairs: int, seed: int,
    head: nn.Module | None = None, device: torch.device = torch.device("cpu"),
) -> dict[str, dict]:
    """Mean cosine similarity by temporal offset k.

    AUROC degenerates within tight k windows (k1-5 pairs are almost all close;
    k16+ pairs are almost all far) because only one class is present.
    Instead report mean_sim and the count — the trend across k buckets is the signal:
    a well-trained encoder should show decreasing sim as k increases.
    """
    buckets = {"k1-5": (1, 5), "k6-15": (6, 15), "k16+": (16, 40)}
    results = {}
    rng = np.random.default_rng(seed + 77)

    for name, (k_lo, k_hi) in buckets.items():
        sims: list[float] = []; disps: list[float] = []
        attempts = 0
        target = n_pairs // len(buckets)
        while len(sims) < target and attempts < target * 20:
            attempts += 1
            tr  = trajs[rng.integers(len(trajs))]
            T   = tr["T"]
            t   = int(rng.integers(0, max(1, T - k_lo)))
            k_hi2 = min(k_hi, T - t - 1)
            if k_hi2 < k_lo:
                continue
            k   = int(rng.integers(k_lo, k_hi2 + 1))
            tk  = t + k
            disp = float(np.linalg.norm(tr["pos"][t] - tr["pos"][tk]))

            if head is not None:
                T_MAX = 70; XY_SCALE = 10.0
                origin  = tr["pos"][0]
                pos_xy  = (tr["pos"] - origin) / XY_SCALE
                zt_  = torch.from_numpy(tr["latents"][t]).unsqueeze(0).to(device)
                ztk_ = torch.from_numpy(tr["latents"][tk]).unsqueeze(0).to(device)
                with torch.no_grad():
                    e_t  = F.normalize(head(zt_),  dim=-1)
                    e_tk = F.normalize(head(ztk_), dim=-1)
                sim = float(F.cosine_similarity(e_t, e_tk).item())
            else:
                sim = float(np.dot(tr["latents"][t], tr["latents"][tk]))

            sims.append(sim); disps.append(disp)

        results[name] = {
            "mean_sim": float(np.mean(sims)) if sims else 0.,
            "mean_disp": float(np.mean(disps)) if disps else 0.,
            "n": len(sims),
        }
    return results


def displacement_correlation(trajs: list[dict], n_pairs: int, seed: int) -> tuple[float, float]:
    """Spearman ρ: quasimetric distance (1 - cosine sim) vs actual metres."""
    try:
        from scipy.stats import spearmanr
    except ImportError:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    dists: list[float] = []; disps: list[float] = []
    attempts = 0

    while len(dists) < n_pairs and attempts < n_pairs * 10:
        attempts += 1
        tr  = trajs[rng.integers(len(trajs))]
        T   = tr["T"]
        if T <= 1:
            continue
        t   = int(rng.integers(0, max(1, T - 1)))
        k_hi = T - t - 1
        if k_hi < 1:
            continue
        tk  = t + 1 + int(rng.integers(0, k_hi))
        disp = float(np.linalg.norm(tr["pos"][t] - tr["pos"][tk]))
        sim  = float(np.dot(tr["latents"][t], tr["latents"][tk]))
        dists.append(1.0 - sim)   # cosine distance
        disps.append(disp)

    rho, pval = spearmanr(dists, disps)
    return float(rho), float(pval)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="RECON quasimetric evaluation")
    ap.add_argument("--data",         required=True,   help="recon_release/ directory")
    ap.add_argument("--encoder-ckpt", default=None,    help="student_best.pt path")
    ap.add_argument("--head-ckpt",    default=None,    help="rope_head_best.pt path")
    ap.add_argument("--max-files",    type=int, default=200)
    ap.add_argument("--held-out-only", action="store_true",
                    help="Use last --max-files files (not seen during default training split)")
    ap.add_argument("--n-pairs",      type=int, default=2000)
    ap.add_argument("--n-triplets",   type=int, default=2000)
    ap.add_argument("--embed-dim",    type=int, default=96,  help="RoPEHead embed dim")
    ap.add_argument("--seed",         type=int, default=999)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Discover files ────────────────────────────────────────────────────
    data_dir  = Path(args.data)
    all_files = sorted(data_dir.rglob("*.hdf5")) + sorted(data_dir.rglob("*.h5"))
    if not all_files:
        raise FileNotFoundError(f"No HDF5 files in {data_dir}")

    if args.held_out_only:
        # Take last N (default training split uses a shuffled 80/20; last by name = unseen)
        eval_files = all_files[-args.max_files:]
        print(f"Held-out eval: {len(eval_files)} files (last {args.max_files} by name)")
    else:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(all_files), min(args.max_files, len(all_files)), replace=False)
        eval_files = [all_files[i] for i in sorted(idx)]
        print(f"Random sample: {len(eval_files)} files")

    # ── Load encoder ──────────────────────────────────────────────────────
    encoder = StudentEncoder().to(device)
    if args.encoder_ckpt:
        ckpt  = torch.load(args.encoder_ckpt, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        missing, unexpected = encoder.load_state_dict(state, strict=False)
        loaded = len(state) - len(missing)
        print(f"Encoder: {args.encoder_ckpt}")
        print(f"  Loaded {loaded}/{len(state)} tensors  "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
        if ckpt.get("auroc"):
            print(f"  Checkpoint AUROC: {ckpt['auroc']:.4f}")
    else:
        print("Encoder: random weights (no checkpoint)")
    encoder.eval()

    # ── Load RoPE head ────────────────────────────────────────────────────
    head = None
    if args.head_ckpt:
        try:
            import sys; sys.path.insert(0, str(Path(__file__).parent))
            from rope_temporal_head import RoPETemporalHead
            head = RoPETemporalHead(
                latent_dim=StudentEncoder.LATENT_DIM,
                embed_dim=args.embed_dim,
            ).to(device)
            ckpt = torch.load(args.head_ckpt, map_location="cpu", weights_only=False)
            head.load_state_dict(ckpt.get("model", ckpt))
            head.eval()
            print(f"RoPEHead: {args.head_ckpt}  "
                  f"(epoch {ckpt.get('epoch','?')}  AUROC {ckpt.get('auroc',0):.4f})")
        except Exception as e:
            print(f"RoPEHead load failed: {e} — evaluating encoder only")
            head = None

    # ── Load and encode trajectories ──────────────────────────────────────
    print(f"\nLoading {len(eval_files)} trajectories...")
    trajs = load_trajectories(eval_files)
    print("Encoding with StudentEncoder...")
    trajs = encode_trajectories(trajs, encoder, device)

    print(f"\n{'═'*58}")
    print(f"RECON Quasimetric Evaluation  —  {len(trajs)} trajectories")
    print(f"{'═'*58}")

    # ── 1. Triplet ordering score ─────────────────────────────────────────
    print(f"\n── 1. Triplet Ordering Score (encoder only) ────────────────")
    print(f"   k_near ∈ [1,{K_NEAR_MAX}]  k_far ∈ [{K_FAR_MIN},{K_FAR_MAX}]  "
          f"n={args.n_triplets}")
    t0 = time.time()
    tri_score = triplet_ordering_score(trajs, args.n_triplets, args.seed)
    print(f"   Score = {tri_score:.4f}  (random=0.50, target>0.80)  "
          f"({time.time()-t0:.1f}s)")

    # ── 2. Close/Far AUROC ────────────────────────────────────────────────
    print(f"\n── 2. Close/Far AUROC  (close<{CLOSE_THRESH}m, far>{FAR_THRESH}m) ──────────────")
    print(f"   n_pairs={args.n_pairs}")

    t0 = time.time()
    enc_auc, enc_close, enc_far = close_far_auroc(trajs, args.n_pairs, args.seed)
    enc_sep = enc_close - enc_far  # positive = close pairs more similar than far
    print(f"\n   Encoder only:")
    print(f"     AUROC     = {enc_auc:.4f}")
    print(f"     close sim = {enc_close:+.4f}   far sim = {enc_far:+.4f}   sep = {enc_sep:+.4f}")

    if head is not None:
        t0 = time.time()
        head_auc, head_close, head_far = close_far_auroc(
            trajs, args.n_pairs, args.seed, head=head, device=device
        )
        head_sep = head_close - head_far  # positive = head correctly orders close > far
        gain = head_auc - enc_auc
        print(f"\n   Encoder + RoPEHead:")
        print(f"     AUROC     = {head_auc:.4f}  (Δ={gain:+.4f} vs encoder)")
        print(f"     close sim = {head_close:+.4f}   far sim = {head_far:+.4f}   sep = {head_sep:+.4f}")

    # ── 3. Per-k breakdown ────────────────────────────────────────────────
    print(f"\n── 3. Per-k AUROC Breakdown ────────────────────────────────")
    enc_k = per_k_auroc(trajs, args.n_pairs, args.seed)
    if head is not None:
        head_k = per_k_auroc(trajs, args.n_pairs, args.seed, head=head, device=device)

    # Header
    header = f"  {'Bucket':<10}  {'EncSim':>8}  {'Disp(m)':>8}"
    if head is not None:
        header += f"  {'HeadSim':>8}  {'Δ':>7}"
    print(header)
    print("  " + "─" * (len(header) - 2))
    for bucket, vals in enc_k.items():
        row = f"  {bucket:<10}  {vals['mean_sim']:>8.4f}  {vals['mean_disp']:>8.2f}m"
        if head is not None:
            hv = head_k.get(bucket, {}).get("mean_sim", 0.)
            row += f"  {hv:>8.4f}  {hv - vals['mean_sim']:>+7.4f}"
        print(row)
    print("  (Decreasing sim with k = encoder learned temporal ordering)")

    # ── 4. Displacement correlation ───────────────────────────────────────
    print(f"\n── 4. Displacement Correlation (Spearman ρ) ────────────────")
    rho, pval = displacement_correlation(trajs, args.n_pairs, args.seed)
    if not np.isnan(rho):
        print(f"   ρ = {rho:+.4f}  p = {pval:.2e}  "
              f"({'significant' if pval < 0.01 else 'not significant'})")
        print(f"   (positive ρ = quasimetric distance tracks physical metres)")
    else:
        print("   (scipy not installed — skipping)")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'═'*58}")
    print(f"SUMMARY")
    print(f"{'═'*58}")
    print(f"  Triplet ordering score : {tri_score:.4f}  "
          f"({'✅ above 0.80' if tri_score > 0.80 else '❌ below 0.80'})")
    print(f"  Encoder AUROC          : {enc_auc:.4f}  "
          f"({'✅ above 0.70' if enc_auc > 0.70 else '❌ below 0.70'})")
    if head is not None:
        print(f"  Head AUROC             : {head_auc:.4f}  "
              f"({'✅ above 0.90' if head_auc > 0.90 else '⚠️  below 0.90'})")
    if not np.isnan(rho):
        print(f"  Displacement ρ         : {rho:+.4f}")
    print(f"{'═'*58}")


if __name__ == "__main__":
    main()
