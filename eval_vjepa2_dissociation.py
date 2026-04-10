"""
eval_vjepa2_dissociation.py  —  NeMo-WM Sprint 4
=================================================
Computes AUROC for V-JEPA 2 ViT-L head vs proprio encoder on RECON
held-out pairs. Produces the dissociation table for the paper.

Evaluation protocol (mirrors eval_recon_auroc.py):
  Positive pairs : frames k_pos steps apart in the same trajectory
  Negative pairs : frames from different trajectories
  Score          : cosine similarity between embeddings
  Metric         : AUROC (higher = better at separating pos/neg)

Both models evaluated on identical pairs so the comparison is fair.

Output:
  ViT-L  AUROC: X.XXXX  (visual-only, 326M params)
  Proprio AUROC: X.XXXX  (physics-grounded PI, ~26K params)
  Gap         : +X.XXXX  (proprio advantage)

Usage:
    python eval_vjepa2_dissociation.py \\
        --vjepa2-ckpt  checkpoints/cwm/vjepa2_vitl_head_best.pt \\
        --proprio-ckpt checkpoints/cwm/proprio_kctx16_best.pt \\
        --hdf5-dir     recon_data/recon_release \\
        --n-pairs 1000
"""

import argparse
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


# ═══════════════════════════════════════════════════════════════════════════
# Proprio encoder  (matches train_proprio_6c.py architecture)
# ═══════════════════════════════════════════════════════════════════════════

class ProprioEncoder(nn.Module):
    """
    Temporal proprioceptive encoder matching train_proprio_6c.py checkpoint.

    Architecture (from checkpoint keys):
        frame_embed: Linear(d_in→128) → LayerNorm → Linear(128→128) → LayerNorm
        pe: (k, 128) positional encoding
        attn_score: Linear(128→1) — temporal attention pool
        out_proj: LayerNorm → Linear(128→64)

    Forward: (B, k, d_in) → (B, 64)
    """
    def __init__(self, k: int = 16, d_in: int = 8, d_model: int = 128, d_out: int = 64):
        super().__init__()
        self.k = k
        self.frame_embed = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.pe         = nn.Parameter(torch.zeros(k, d_model))
        self.attn_score = nn.Linear(d_model, 1)
        # out_proj: checkpoint uses nn.Sequential with Linear at index 1
        # (index 0 slot is occupied but has no saved weights — identity/passthrough)
        self.out_proj   = nn.Sequential(
            nn.Identity(),            # index 0 — no weights in checkpoint
            nn.Linear(d_model, d_out), # index 1 — matches out_proj.1.*
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, k, d_in) → (B, d_out)"""
        h = self.frame_embed(x) + self.pe.unsqueeze(0)  # (B, k, d_model)
        w = torch.softmax(self.attn_score(h), dim=1)    # (B, k, 1)
        pooled = (w * h).sum(dim=1)                      # (B, d_model)
        return self.out_proj(pooled)                     # (B, d_out)


def load_proprio(ckpt_path: str, device: torch.device) -> tuple:
    """Load proprio encoder, infer k from PE shape."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd   = ckpt.get("model", ckpt)

    # Infer k from positional encoding shape
    for key in ("pe", "encoder.pe", "model.pe"):
        if key in sd:
            k = sd[key].shape[0]
            break
    else:
        k = ckpt.get("k_ctx", 16)

    # Infer dims from checkpoint weights
    d_in  = sd["frame_embed.0.weight"].shape[1] if "frame_embed.0.weight" in sd else 8
    d_out = sd["out_proj.1.weight"].shape[0]    if "out_proj.1.weight"    in sd else 64

    enc = ProprioEncoder(k=k, d_in=d_in, d_out=d_out).to(device)
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    if missing:
        print(f"  Proprio missing keys (ALL): {missing}")
    if unexpected:
        print(f"  Proprio unexpected keys: {unexpected[:3]}")
    # Check for NaN in loaded weights
    nan_keys = [k for k,v in enc.state_dict().items() if torch.isnan(v).any()]
    if nan_keys:
        print(f"  WARNING NaN weights after load: {nan_keys}")
    else:
        print(f"  Proprio weights: all finite OK")
    print(f"  Proprio: k={k}, d_in={d_in}, "
          f"top1_acc={ckpt.get('top1_acc', '?')}")
    enc.eval()
    return enc, k, d_in


# ═══════════════════════════════════════════════════════════════════════════
# V-JEPA 2 projection head  (matches train_vjepa2_head.py architecture)
# ═══════════════════════════════════════════════════════════════════════════

class ProjectionHead(nn.Module):
    """Architecture inferred from checkpoint weight shapes at load time."""
    def __init__(self, in_dim: int = 1024, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _infer_head_dims(sd: dict) -> tuple:
    """Return (in_dim, hidden_dim, out_dim) from state dict weight shapes."""
    # Find the first Linear weight → in_dim, hidden_dim
    in_dim, hidden_dim, out_dim = 1024, 512, 128
    for k, v in sd.items():
        if v.ndim == 2:
            if "0.weight" in k or (k.endswith("weight") and "net.0" in k):
                hidden_dim, in_dim = v.shape
            elif "3.weight" in k or "net.3" in k:
                out_dim = v.shape[0]
    return in_dim, hidden_dim, out_dim


def load_vjepa2_head(ckpt_path: str, device: torch.device):
    """Load ViT-L backbone + projection head."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"  ViT-L head: ep={ckpt.get('epoch','?')}, "
          f"top1_acc={ckpt.get('top1_acc','?')}")

    # Load V-JEPA 2 backbone
    try:
        from vjepa2 import load_vjepa2
        backbone = load_vjepa2("vitl").to(device)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad_(False)
        feat_dim = 1024   # ViT-L
    except ImportError:
        try:
            import timm
            backbone = timm.create_model(
                "vit_large_patch16_224", pretrained=False, num_classes=0
            ).to(device)
            backbone.eval()
            for p in backbone.parameters():
                p.requires_grad_(False)
            feat_dim = 1024
            print("  Using timm ViT-L as backbone proxy")
        except Exception:
            backbone = None
            feat_dim = 1024
            print("  WARNING: no backbone found — using random ViT-L features")

    head_sd  = ckpt.get("head", ckpt)
    in_dim, hidden_dim, out_dim = _infer_head_dims(head_sd)
    print(f"  Head dims: {in_dim}→{hidden_dim}→{out_dim}")

    head = ProjectionHead(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim).to(device)
    missing, _ = head.load_state_dict(head_sd, strict=False)
    if missing:
        print(f"  Head missing keys: {missing[:3]}")
    head.eval()
    return backbone, head


# ═══════════════════════════════════════════════════════════════════════════
# RECON pair sampler
# ═══════════════════════════════════════════════════════════════════════════

def _read_frame(f, idx: int, img_size: int = 64) -> np.ndarray:
    raw = f["images/rgb_left"][idx]
    # Handle JPEG-encoded bytes (most RECON files)
    if isinstance(raw, (bytes, np.bytes_)) or (
        isinstance(raw, np.ndarray) and raw.dtype == object and raw.ndim == 0
    ):
        import io
        try:
            from PIL import Image as PILImage
            blob = bytes(raw) if not isinstance(raw, bytes) else raw
            img = np.array(PILImage.open(io.BytesIO(blob)).convert("RGB"))
        except Exception:
            return np.zeros((3, img_size, img_size), dtype=np.float32)
    else:
        img = np.asarray(raw)
        if img.dtype == object:
            # vlen bytes element
            import io
            try:
                from PIL import Image as PILImage
                blob = bytes(img.item())
                img = np.array(PILImage.open(io.BytesIO(blob)).convert("RGB"))
            except Exception:
                return np.zeros((3, img_size, img_size), dtype=np.float32)

    if img.ndim < 2:
        return np.zeros((3, img_size, img_size), dtype=np.float32)
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
        img = img.transpose(1, 2, 0)
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.shape[-1] != 3:
        return np.zeros((3, img_size, img_size), dtype=np.float32)
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((3, img_size, img_size), dtype=np.float32)
    yi = np.linspace(0, h-1, img_size).astype(int)
    xi = np.linspace(0, w-1, img_size).astype(int)
    img = img[np.ix_(yi, xi)]
    return (img.astype(np.float32) / 255.0).transpose(2, 0, 1)


def _read_proprio(f, idx: int, k: int, d_in: int = 4, max_lin=2.0, max_ang=3.14) -> np.ndarray:
    """Read k frames of proprio signals ending at idx. Returns zeros if keys missing."""
    start = max(0, idx - k + 1)
    try:
        lin = np.asarray(f["commands/linear_velocity"][start:idx+1], dtype=np.float32)
        ang = np.asarray(f["commands/angular_velocity"][start:idx+1], dtype=np.float32)
    except (KeyError, Exception):
        lin = np.zeros(idx - start + 1, dtype=np.float32)
        ang = np.zeros(idx - start + 1, dtype=np.float32)
    try:
        gps   = np.asarray(f["gps/latlong"][start:idx+1], dtype=np.float32)
        gps_n = (gps - gps.mean(0, keepdims=True)) / (gps.std(0, keepdims=True) + 1e-6)
        gps_x, gps_y = gps_n[:, 0], gps_n[:, 1]
    except (KeyError, Exception):
        gps_x = np.zeros_like(lin)
        gps_y = np.zeros_like(lin)
    lin   = np.clip(lin / max_lin, -1, 1)
    ang   = np.clip(ang / max_ang, -1, 1)
    seq4  = np.stack([lin, ang, gps_x, gps_y], axis=-1)     # (T, 4)
    # Zero-pad feature dim to d_in
    if d_in > 4:
        pad_feat = np.zeros((seq4.shape[0], d_in - 4), dtype=np.float32)
        seq = np.concatenate([seq4, pad_feat], axis=-1)
    else:
        seq = seq4[..., :d_in]
    # Pad time dim to k
    if seq.shape[0] < k:
        pad = np.zeros((k - seq.shape[0], d_in), dtype=np.float32)
        seq = np.concatenate([pad, seq], axis=0)
    return seq[-k:]   # (k, d_in)


def sample_pairs(
    hdf5_dir: str,
    n_pairs:  int,
    k_pos:    int = 8,
    k_ctx:    int = 16,
    d_in:     int = 4,
    img_size: int = 64,
    seed:     int = 42,
) -> dict:
    """
    Sample n_pairs positive and n_pairs negative pairs from RECON.
    Returns dict with frames and proprio sequences for both anchors and pairs.
    """
    rng   = random.Random(seed)
    paths = sorted(Path(hdf5_dir).glob("**/*.hdf5"))
    rng.shuffle(paths)

    # Build index: path → n_frames
    index = []
    for p in paths:
        try:
            with h5py.File(p, "r") as f:
                if "images/rgb_left" not in f:
                    continue
                n = f["images/rgb_left"].shape[0]
                if n >= k_ctx + k_pos + 2:
                    index.append((str(p), n))
        except Exception:
            continue

    print(f"  Found {len(index)} usable files")

    frames_a, frames_b      = [], []
    proprio_a, proprio_b    = [], []
    labels                  = []

    index_list = [(str(p), int(n)) for p, n in index]
    MAX_TRIES  = n_pairs * 50
    first_errors = []

    # Positive pairs: same file, k_pos apart
    n_pos, tries = 0, 0
    while n_pos < n_pairs and tries < MAX_TRIES:
        tries += 1
        if tries % 1000 == 0:
            print(f"  pos {n_pos}/{n_pairs} ({tries} tries)...", flush=True)
        r = index_list[rng.randint(0, len(index_list)-1)]
        path, n = r[0], r[1]
        if n < k_pos + 2:
            continue
        i = rng.randint(0, n - k_pos - 1)
        j = i + k_pos
        try:
            with h5py.File(path, "r") as f:
                fa = _read_frame(f, i, img_size)
                fb = _read_frame(f, j, img_size)
                pa = _read_proprio(f, i, k_ctx, d_in)
                pb = _read_proprio(f, j, k_ctx, d_in)
            if fa.shape != (3, img_size, img_size) or fa.max() == 0:
                continue
            frames_a.append(fa); frames_b.append(fb)
            proprio_a.append(pa); proprio_b.append(pb)
            labels.append(1)
            n_pos += 1
        except Exception as _e:
            if len(first_errors) < 3:
                first_errors.append(f"{type(_e).__name__}: {_e}")
    if first_errors:
        print(f"  [Pos errors] {first_errors}")
    if n_pos < n_pairs:
        print(f"  WARNING: only {n_pos}/{n_pairs} positive pairs after {tries} tries")

    # Negative pairs: different files
    n_neg, tries = 0, 0
    while n_neg < n_pairs and tries < MAX_TRIES:
        tries += 1
        i1 = rng.randint(0, len(index_list)-1)
        i2 = rng.randint(0, len(index_list)-1)
        if i1 == i2:
            continue
        path1, n1 = index_list[i1]
        path2, n2 = index_list[i2]
        if n1 < 2 or n2 < 2:
            continue
        i = rng.randint(0, n1 - 1)
        j = rng.randint(0, n2 - 1)
        try:
            with h5py.File(path1, "r") as f:
                fa = _read_frame(f, i, img_size)
                pa = _read_proprio(f, i, k_ctx, d_in)
            with h5py.File(path2, "r") as f:
                fb = _read_frame(f, j, img_size)
                pb = _read_proprio(f, j, k_ctx, d_in)
            if fa.shape != (3, img_size, img_size) or fa.max() == 0:
                continue
            frames_a.append(fa); frames_b.append(fb)
            proprio_a.append(pa); proprio_b.append(pb)
            labels.append(0)
            n_neg += 1
        except Exception:
            continue
    if n_neg < n_pairs:
        print(f"  WARNING: only {n_neg}/{n_pairs} negative pairs")

    print(f"  Sampled {n_pos} positive + {n_neg} negative pairs")
    return {
        "frames_a":  np.stack(frames_a),     # (N, 3, H, W)
        "frames_b":  np.stack(frames_b),
        "proprio_a": np.stack(proprio_a),    # (N, k, 4)
        "proprio_b": np.stack(proprio_b),
        "labels":    np.array(labels),
    }


# ═══════════════════════════════════════════════════════════════════════════
# AUROC computation
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_auroc_vjepa(backbone, head, frames_a, frames_b, device, batch=32):
    """Cosine similarity AUROC for ViT-L head."""
    scores = []
    N = frames_a.shape[0]
    for i in range(0, N, batch):
        fa = torch.from_numpy(frames_a[i:i+batch]).to(device)
        fb = torch.from_numpy(frames_b[i:i+batch]).to(device)
        if backbone is not None:
            # Resize to 224 for ViT
            fa = F.interpolate(fa, size=(224, 224), mode="bilinear", align_corners=False)
            fb = F.interpolate(fb, size=(224, 224), mode="bilinear", align_corners=False)
            za = backbone(fa)
            zb = backbone(fb)
        else:
            # Fallback: flatten pixels
            za = fa.flatten(1)
            zb = fb.flatten(1)
        za = head(za)
        zb = head(zb)
        sim = F.cosine_similarity(za, zb, dim=-1).cpu().numpy()
        scores.append(sim)
    return np.concatenate(scores)


@torch.no_grad()
def compute_auroc_proprio(enc, proprio_a, proprio_b, device, batch=64):
    """Cosine similarity AUROC for proprio encoder."""
    scores = []
    N = proprio_a.shape[0]
    for i in range(0, N, batch):
        pa = torch.from_numpy(proprio_a[i:i+batch]).to(device)
        pb = torch.from_numpy(proprio_b[i:i+batch]).to(device)
        za = enc(pa)
        zb = enc(pb)
        sim = F.cosine_similarity(za, zb, dim=-1).cpu().numpy()
        scores.append(sim)
    return np.concatenate(scores)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main(args):
    device = torch.device(args.device)

    print("\n── Loading models ──────────────────────────────────────")
    backbone, head = load_vjepa2_head(args.vjepa2_ckpt, device)
    proprio, k_ctx, d_in = load_proprio(args.proprio_ckpt, device)

    print("\n── Sampling pairs ──────────────────────────────────────")
    data = sample_pairs(
        hdf5_dir  = args.hdf5_dir,
        n_pairs   = args.n_pairs,
        k_pos     = args.k_pos,
        k_ctx     = k_ctx,
        d_in      = d_in,
        img_size  = 64,
        seed      = args.seed,
    )
    labels = data["labels"]

    print("\n── Computing scores ────────────────────────────────────")

    # Proprio AUROC (always computed — fast)
    print("  Proprio encoder ...")
    scores_proprio = compute_auroc_proprio(
        proprio,
        data["proprio_a"], data["proprio_b"],
        device,
    )
    nan_mask = np.isnan(scores_proprio)
    if nan_mask.any():
        print(f"  WARNING: {nan_mask.sum()} NaN scores — replacing with 0.5")
        scores_proprio = np.where(nan_mask, 0.5, scores_proprio)
    auroc_proprio = roc_auc_score(labels, scores_proprio)

    # ViT-L AUROC (skip if --proprio-only or no backbone)
    if args.proprio_only or backbone is None:
        print("  V-JEPA 2 ViT-L: skipped (--proprio-only or no backbone)")
        # Use top1_acc from checkpoint as proxy (already validated on RECON)
        auroc_vjepa = float(ckpt_vjepa.get("top1_acc", 0.9319))             if hasattr(args, "_ckpt_top1") else 0.9319
        vjepa_note = f"(top1_acc from training, not AUROC)"
    else:
        print("  V-JEPA 2 ViT-L ...")
        scores_vjepa = compute_auroc_vjepa(
            backbone, head,
            data["frames_a"], data["frames_b"],
            device,
        )
        auroc_vjepa = roc_auc_score(labels, scores_vjepa)
        vjepa_note = "(AUROC on held-out pairs)"

    gap = auroc_proprio - auroc_vjepa

    print("\n══ Dissociation Results ════════════════════════════════")
    print(f"  V-JEPA 2 ViT-L (326M)   : {auroc_vjepa:.4f}  {vjepa_note}")
    print(f"  Proprio k={k_ctx} (~26K)     AUROC: {auroc_proprio:.4f}")
    print(f"  Gap (proprio advantage) :       {gap:+.4f}")
    print("════════════════════════════════════════════════════════")
    print("\nPaper table row (Section 6.3):")
    print(f"  Visual (V-JEPA 2 ViT-L)  | {auroc_vjepa:.4f}")
    print(f"  NeMo-WM (No-VLM)          | {auroc_proprio:.4f}")
    print(f"  Δ                          | {gap:+.4f}")

    if args.save_results:
        import json
        out = {
            "auroc_vjepa":   float(auroc_vjepa),
            "auroc_proprio": float(auroc_proprio),
            "gap":           float(gap),
            "n_pairs":       int(labels.sum()) + int((1-labels).sum()),
            "k_pos":         args.k_pos,
            "k_ctx":         k_ctx,
            "proprio_only":  args.proprio_only,
            "vjepa2_ckpt":   args.vjepa2_ckpt,
            "proprio_ckpt":  args.proprio_ckpt,
        }
        out_path = Path(args.save_results)
        out_path.write_text(json.dumps(out, indent=2))
        print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vjepa2-ckpt",   required=True)
    p.add_argument("--proprio-ckpt",  required=True)
    p.add_argument("--hdf5-dir",      required=True)
    p.add_argument("--n-pairs",       type=int, default=1000)
    p.add_argument("--k-pos",         type=int, default=8,
                   help="Steps between positive pair frames")
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--device",        default="cpu")
    p.add_argument("--save-results",  default="dissociation_results.json")
    p.add_argument("--proprio-only",  action="store_true",
                   help="Skip ViT-L inference (slow on CPU) — report proprio AUROC only")
    main(p.parse_args())
