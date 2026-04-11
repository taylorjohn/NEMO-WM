"""
test_vjepa2_e1.py  —  NeMo-WM × V-JEPA 2  Quick Integration Test (E1)
======================================================================
Fastest possible test: run RECON quasimetric eval with V-JEPA 2's
visual encoder instead of NeMo-WM's StudentEncoder.

This answers the key question BEFORE committing to the NeurIPS paper:
  Does V-JEPA 2's encoder leave room for the proprio pathway to add value?

Three scenarios:
  A) V-JEPA 2 alone >> NeMo-WM visual:  proprio fusion adds less, story is "efficiency"
  B) V-JEPA 2 alone ~= NeMo-WM visual:  proprio still adds +0.07 gap as before
  C) V-JEPA 2 alone < NeMo-WM visual:   surprising, V-JEPA 2 not suited for RECON quasimetric

Setup (run once):
  pip install transformers huggingface_hub timm einops
  python test_vjepa2_e1.py --download          # downloads ~5GB ViT-g weights
  python test_vjepa2_e1.py --eval              # runs eval (30 min)

For comparison baseline, also run:
  python eval_recon_auroc.py \\
      --head-ckpt checkpoints/cwm/temporal_head_sprint3.pt \\
      --cwm-ckpt  checkpoints/cwm/cwm_multidomain_best.pt \\
      --hdf5-dir  recon_data/recon_release \\
      --n-pairs 1000 --k-pos 4

Expected result (hypothesis):
  V-JEPA 2 + NeMo-WM head: AUROC ~0.95-0.97 (better visual features)
  V-JEPA 2 + proprio:       AUROC ~0.99+ (dual system)
  NeMo-WM standalone:       AUROC 0.9837 (existing result)
"""

import argparse
import io
import math
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import transforms


# ── V-JEPA 2 loader ───────────────────────────────────────────────────────────

def load_vjepa2_encoder(model_size="small", device="cpu"):
    """
    Load V-JEPA 2 visual encoder from HuggingFace.

    model_size options (trade-off speed vs quality):
      "small"  — ViT-S/16, ~22M params, fastest, good baseline
      "base"   — ViT-B/16, ~86M params, recommended for testing
      "large"  — ViT-L/16, ~307M params
      "giant"  — ViT-g/14, ~1.1B params (full V-JEPA 2, needs ~20GB RAM)

    For quick testing on GMKtec (128GB RAM): use "large" or "giant"
    For fast iteration: use "base"
    """
    try:
        from transformers import AutoModel, AutoProcessor
    except ImportError:
        raise ImportError("pip install transformers huggingface_hub")

    model_ids = {
        "small":  "facebook/vjepa2-vits-fpc64-256",
        "base":   "facebook/vjepa2-vitb-fpc64-256",
        "large":  "facebook/vjepa2-vitl-fpc64-256",
        "giant":  "facebook/vjepa2-vitg-fpc64-256",
    }
    model_id = model_ids[model_size]
    print(f"Loading V-JEPA 2 encoder: {model_id}")
    print(f"  (First run downloads weights — be patient)")

    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model = model.to(device).eval()

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded: {params:.1f}M params")
    return model


def vjepa2_embed_frame(model, frame_tensor, device="cpu"):
    """
    Extract a 64-D quasimetric embedding from V-JEPA 2.

    frame_tensor: (3, 224, 224) normalised tensor

    V-JEPA 2 outputs (B, num_patches, d_model). We mean-pool patches
    then project to 64-D to match the NeMo-WM TemporalHead output space.
    """
    with torch.no_grad():
        # V-JEPA 2 expects (B, T, C, H, W) — single frame = T=1
        x = frame_tensor.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 3, 224, 224)
        out = model(x)
        # Extract patch embeddings: (B, T, N, D) or (B, N, D)
        if hasattr(out, "last_hidden_state"):
            feats = out.last_hidden_state  # (B, N, D)
        else:
            feats = out[0]
        # Mean pool over patches and time
        z = feats.mean(dim=tuple(range(1, feats.dim() - 1)))  # (B, D)
        return F.normalize(z.squeeze(0), dim=-1)  # (D,)


# ── Lightweight projection head ───────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """
    64-D projection from V-JEPA 2's d_model to quasimetric space.
    Trained on RECON pairs with InfoNCE — same as TemporalHead.
    """
    def __init__(self, d_in=768, d_out=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.GELU(), nn.LayerNorm(256),
            nn.Linear(256, d_out),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


# ── Frame decoder (same as NeMo-WM) ──────────────────────────────────────────

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def decode_frame(jpeg_bytes):
    return TRANSFORM(Image.open(io.BytesIO(jpeg_bytes)).convert("RGB"))


# ── RECON pair sampler ────────────────────────────────────────────────────────

def sample_pairs(hdf5_dir, n_pairs=500, k_pos=4, k_hard_min=32, seed=42):
    rng = np.random.default_rng(seed)
    files = sorted(Path(hdf5_dir).glob("jackal_2019-*.hdf5"))
    rng.shuffle(files := list(files))

    pairs = []
    for fp in files:
        if len(pairs) >= n_pairs:
            break
        try:
            with h5py.File(fp, "r") as hf:
                T = hf["images"]["rgb_left"].shape[0]
                if T < k_hard_min + k_pos + 5:
                    continue
                for t in range(k_hard_min, T - k_pos - 1):
                    t_pos = t + rng.integers(1, k_pos + 1)
                    t_neg = rng.integers(0, max(1, t - k_hard_min))
                    pairs.append((str(fp), t, int(t_pos), int(t_neg)))
                    if len(pairs) >= n_pairs:
                        break
        except Exception:
            continue

    return pairs


# ── Zero-shot eval (no projection head, raw cosine) ──────────────────────────

@torch.no_grad()
def eval_vjepa2_zeroshot(model, pairs, device="cpu"):
    """
    Zero-shot: use raw V-JEPA 2 embeddings with cosine similarity.
    No finetuning needed. This is the fastest test.
    """
    scores, labels = [], []
    t0 = time.perf_counter()

    for i, (fp, t_a, t_p, t_n) in enumerate(pairs):
        try:
            with h5py.File(fp, "r") as hf:
                img_a = hf["images"]["rgb_left"][t_a]
                img_p = hf["images"]["rgb_left"][t_p]
                img_n = hf["images"]["rgb_left"][t_n]

            f_a = decode_frame(bytes(img_a)).to(device)
            f_p = decode_frame(bytes(img_p)).to(device)
            f_n = decode_frame(bytes(img_n)).to(device)

            z_a = vjepa2_embed_frame(model, f_a, device)
            z_p = vjepa2_embed_frame(model, f_p, device)
            z_n = vjepa2_embed_frame(model, f_n, device)

            scores.append(float(F.cosine_similarity(z_a.unsqueeze(0), z_p.unsqueeze(0))))
            labels.append(1)
            scores.append(float(F.cosine_similarity(z_a.unsqueeze(0), z_n.unsqueeze(0))))
            labels.append(0)

        except Exception:
            continue

        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            eta = elapsed / (i + 1) * (len(pairs) - i - 1)
            print(f"  {i+1}/{len(pairs)}  ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

    auroc = float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else float("nan")
    return auroc


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5-dir",    default="recon_data/recon_release")
    p.add_argument("--n-pairs",     type=int, default=500)
    p.add_argument("--k-pos",       type=int, default=4)
    p.add_argument("--model-size",  default="base",
                   choices=["small", "base", "large", "giant"],
                   help="V-JEPA 2 model size (base recommended for quick test)")
    p.add_argument("--device",      default="cpu")
    p.add_argument("--download",    action="store_true",
                   help="Download weights and exit")
    args = p.parse_args()

    dev = torch.device(args.device)

    print("\n" + "="*65)
    print("  NeMo-WM × V-JEPA 2 — E1 Baseline Test")
    print("="*65)
    print(f"  Model:    V-JEPA 2 {args.model_size}")
    print(f"  Dataset:  {args.hdf5_dir}")
    print(f"  Pairs:    {args.n_pairs} (hard negatives, k>={32})")
    print(f"  Device:   {args.device}")
    print("="*65 + "\n")

    # Load encoder
    encoder = load_vjepa2_encoder(args.model_size, dev)

    if args.download:
        print("Weights downloaded. Run without --download to evaluate.")
        return

    # Sample pairs
    print(f"Sampling {args.n_pairs} hard-negative RECON pairs...")
    pairs = sample_pairs(args.hdf5_dir, args.n_pairs, args.k_pos)
    print(f"  Sampled {len(pairs)} pairs\n")

    # Zero-shot eval
    print("Running zero-shot eval (raw V-JEPA 2 cosine similarity)...")
    auroc = eval_vjepa2_zeroshot(encoder, pairs, dev)

    print(f"\n{'='*65}")
    print(f"  V-JEPA 2 {args.model_size} — zero-shot RECON quasimetric")
    print(f"  Hard negatives (same-file, k>=32), k_pos={args.k_pos}")
    print(f"{'='*65}")
    print(f"  AUROC: {auroc:.4f}")
    print(f"{'='*65}")
    print(f"\n  NeMo-WM reference (VLM-only, same protocol): 0.8893")
    print(f"  NeMo-WM full model:                          0.9978")
    print(f"  NeMo-WM No-VLM (proprio k_ctx=16):           0.9974")

    if auroc > 0.95:
        print(f"\n  → V-JEPA 2 visual features strong on RECON.")
        print(f"    Proprio pathway story becomes: complementary efficiency.")
        print(f"    NeurIPS angle: dual-system robustness + domain adaptation.")
    elif auroc > 0.88:
        print(f"\n  → V-JEPA 2 comparable to NeMo-WM VLM-only (0.889).")
        print(f"    Proprio pathway adds same +0.10 gap as on StudentEncoder.")
        print(f"    NeurIPS angle: full dissociation story holds at scale.")
    else:
        print(f"\n  → V-JEPA 2 weaker than NeMo-WM VLM-only on RECON quasimetric.")
        print(f"    V-JEPA 2 features not suited for temporal navigation metric.")
        print(f"    Consider: add a TemporalHead fine-tuned on RECON first.")

    print(f"\n  Next: train a lightweight ProjectionHead on top of frozen")
    print(f"  V-JEPA 2 features (same as TemporalHead in NeMo-WM), then")
    print(f"  run full dissociation eval with proprio_kctx16_best.pt.")


if __name__ == "__main__":
    main()
