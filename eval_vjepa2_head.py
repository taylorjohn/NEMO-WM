"""
eval_vjepa2_head.py — CORTEX-NAV
=================================
Evaluate V-JEPA 2 ProjectionHead AUROC on RECON hard-negative protocol
and compare against proprio k=32 and fusion.

Usage:
  # ViT-G head only
  python eval_vjepa2_head.py \
      --vjepa2-ckpt  checkpoints\cwm\vjepa2_vitg_head_best.pt \
      --hdf5-dir     recon_data\recon_release \
      --model-size   giant \
      --n-pairs 1000 --k-pos 4

  # ViT-G head + proprio fusion
  python eval_vjepa2_head.py \
      --vjepa2-ckpt  checkpoints\cwm\vjepa2_vitg_head_best.pt \
      --proprio-ckpt checkpoints\cwm\proprio_kctx32_best.pt \
      --hdf5-dir     recon_data\recon_release \
      --model-size   giant \
      --n-pairs 1000 --k-pos 4 \
      --hard-negatives --proprio-no-gps

  # Full comparison: ViT-L head vs ViT-G head vs proprio vs fusion
  python eval_vjepa2_head.py \
      --vjepa2-ckpt      checkpoints\cwm\vjepa2_vitg_head_best.pt \
      --vjepa2-ckpt-l    checkpoints\cwm\vjepa2_head_best.pt \
      --proprio-ckpt     checkpoints\cwm\proprio_kctx32_best.pt \
      --hdf5-dir         recon_data\recon_release \
      --n-pairs 1000 --k-pos 4 \
      --hard-negatives --proprio-no-gps \
      --full-comparison
"""

import argparse
import io
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import transforms
from tqdm import tqdm

# ── Constants ────────────────────────────────────────────────

MODEL_IDS = {
    "base":  "facebook/vjepa2-vitb-fpc64-256",
    "large": "facebook/vjepa2-vitl-fpc64-256",
    "giant": "facebook/vjepa2-vitg-fpc64-256",
}

FEAT_DIMS = {"base": 768, "large": 1024, "giant": 1408}

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Models ────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    def __init__(self, d_in: int = 1024, d_out: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.GELU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.GELU(), nn.LayerNorm(128),
            nn.Linear(128, d_out),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class ProprioEncoderTemporal(nn.Module):
    """Minimal wrapper matching Sprint 6c architecture."""
    def __init__(self, d_per_frame=8, k_ctx=32,
                 d_hidden=128, d_model=64):
        super().__init__()
        self.k_ctx   = k_ctx
        self.d_model = d_model
        self.frame_embed = nn.Linear(d_per_frame, d_hidden)
        pe = torch.zeros(k_ctx, d_hidden)
        pos = torch.arange(k_ctx).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_hidden, 2).float() *
            (-np.log(10000.0) / d_hidden)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
        self.attn = nn.MultiheadAttention(d_hidden, num_heads=4,
                                          batch_first=True)
        self.pool = nn.Linear(d_hidden, d_model)

    def forward(self, x):
        # x: (B, k_ctx, d_per_frame)
        h = self.frame_embed(x) + self.pe.unsqueeze(0)
        h, _ = self.attn(h, h, h)
        return F.normalize(self.pool(h.mean(dim=1)), dim=-1)


# ── Feature extraction ────────────────────────────────────────

@torch.no_grad()
def extract_vjepa2(encoder, jpeg_bytes: bytes, device: str) -> torch.Tensor:
    img = TRANSFORM(
        Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    ).unsqueeze(0).to(device)

    # V-JEPA 2 needs (B, C, T, H, W)
    vid = img.unsqueeze(1).repeat(1, 2, 1, 1, 1)  # (B,T,C,H,W) for ViT-G kernel_t=2
    out = encoder(pixel_values_videos=vid)

    if hasattr(out, "last_hidden_state"):
        feats = out.last_hidden_state
    elif isinstance(out, (tuple, list)):
        feats = out[0]
    else:
        feats = out

    while feats.dim() > 2:
        feats = feats.mean(dim=1)

    return F.normalize(feats.squeeze(0), dim=-1).cpu()


# ── RECON HDF5 helpers ────────────────────────────────────────

def list_hdf5_files(hdf5_dir: str, max_files: int = None):
    files = sorted(Path(hdf5_dir).glob("jackal_2019-*.hdf5"))
    if max_files:
        files = files[:max_files]
    return files


def load_proprio_context(hf, t: int, k_ctx: int) -> np.ndarray:
    """Load k_ctx frames of proprioceptive signals ending at t."""
    frames = []
    for i in range(max(0, t - k_ctx + 1), t + 1):
        try:
            vel = float(hf["commands/linear_velocity"][i])
            ang = float(hf["commands/angular_velocity"][i])
            try:
                gps = hf["gps/latlong"][i]
                lat, lon = float(gps[0]), float(gps[1])
            except Exception:
                lat, lon = 0.0, 0.0
            contact = 0.0
            # Compute heading proxy from angular velocity integration
            heading = ang * i * 0.25  # 4Hz
            sin_h, cos_h = np.sin(heading), np.cos(heading)
            delta_h = ang * 0.25
            frame = np.array([vel, ang, lat, lon, contact,
                               sin_h, cos_h, delta_h], dtype=np.float32)
        except Exception:
            frame = np.zeros(8, dtype=np.float32)
        frames.append(frame)

    # Pad to k_ctx if needed
    while len(frames) < k_ctx:
        frames.insert(0, np.zeros(8, dtype=np.float32))

    arr = np.stack(frames[-k_ctx:])  # (k_ctx, 8)
    # Zero GPS if no-gps mode
    arr[:, 2] = 0.0
    arr[:, 3] = 0.0
    return arr


# ── Eval core ─────────────────────────────────────────────────

def eval_condition(
    name: str,
    scores_pos: list,
    scores_neg: list,
) -> dict:
    labels = [1] * len(scores_pos) + [0] * len(scores_neg)
    preds  = scores_pos + scores_neg
    auroc  = roc_auc_score(labels, preds)

    pos_a = np.array(scores_pos)
    neg_a = np.array(scores_neg)

    return {
        "name":      name,
        "auroc":     auroc,
        "pos_mean":  float(pos_a.mean()),
        "pos_std":   float(pos_a.std()),
        "neg_mean":  float(neg_a.mean()),
        "neg_std":   float(neg_a.std()),
        "sep":       float(neg_a.mean() - pos_a.mean()),
        "n":         len(labels),
    }


def run_eval(
    files, encoder_vg, head_vg,
    encoder_vl, head_vl,
    proprio_enc,
    n_pairs, k_pos, k_hard_min,
    hard_negatives, proprio_no_gps,
    device, k_ctx_proprio=32,
):
    """
    For each pair: compute similarity scores under each condition.
    Returns dict of condition → (pos_scores, neg_scores).
    """
    conditions = {
        "vjepa2_vitg_head": ([], []),
        "vjepa2_vitl_head": ([], []),
        "proprio_only":     ([], []),
        "fusion_vitg":      ([], []),
        "fusion_vitl":      ([], []),
    }

    if encoder_vl is None:
        del conditions["vjepa2_vitl_head"]
        del conditions["fusion_vitl"]

    if proprio_enc is None:
        del conditions["proprio_only"]
        del conditions["fusion_vitg"]
        if "fusion_vitl" in conditions:
            del conditions["fusion_vitl"]

    collected = 0
    pbar = tqdm(total=n_pairs, desc="Collecting pairs")

    while collected < n_pairs:
        fp = random.choice(files)
        try:
            with h5py.File(fp, "r") as hf:
                imgs = hf["images/rgb_left"]
                T = imgs.shape[0]
                if T < k_hard_min + k_pos + 10:
                    continue

                t_a = random.randint(k_hard_min, T - k_pos - 2)
                t_p = t_a + random.randint(1, k_pos)

                if hard_negatives:
                    neg_pool = [t for t in range(T)
                                if abs(t - t_a) >= k_hard_min]
                    if not neg_pool:
                        continue
                    t_n = random.choice(neg_pool)
                else:
                    t_n = random.randint(0, T - 1)
                    while abs(t_n - t_a) < k_hard_min:
                        t_n = random.randint(0, T - 1)

                b_a = bytes(imgs[t_a])
                b_p = bytes(imgs[t_p])
                b_n = bytes(imgs[t_n])

                # V-JEPA 2 ViT-G
                if "vjepa2_vitg_head" in conditions and encoder_vg:
                    z_a = head_vg(extract_vjepa2(encoder_vg, b_a, device).unsqueeze(0).to(device))
                    z_p = head_vg(extract_vjepa2(encoder_vg, b_p, device).unsqueeze(0).to(device))
                    z_n = head_vg(extract_vjepa2(encoder_vg, b_n, device).unsqueeze(0).to(device))
                    s_p = float((z_a * z_p).sum())
                    s_n = float((z_a * z_n).sum())
                    conditions["vjepa2_vitg_head"][0].append(s_p)
                    conditions["vjepa2_vitg_head"][1].append(s_n)

                # V-JEPA 2 ViT-L
                if "vjepa2_vitl_head" in conditions and encoder_vl:
                    z_a = head_vl(extract_vjepa2(encoder_vl, b_a, device).unsqueeze(0).to(device))
                    z_p = head_vl(extract_vjepa2(encoder_vl, b_p, device).unsqueeze(0).to(device))
                    z_n = head_vl(extract_vjepa2(encoder_vl, b_n, device).unsqueeze(0).to(device))
                    s_p = float((z_a * z_p).sum())
                    s_n = float((z_a * z_n).sum())
                    conditions["vjepa2_vitl_head"][0].append(s_p)
                    conditions["vjepa2_vitl_head"][1].append(s_n)

                # Proprio only
                if "proprio_only" in conditions and proprio_enc:
                    ctx_a = torch.tensor(
                        load_proprio_context(hf, t_a, k_ctx_proprio),
                        dtype=torch.float32
                    ).unsqueeze(0).to(device)
                    ctx_p = torch.tensor(
                        load_proprio_context(hf, t_p, k_ctx_proprio),
                        dtype=torch.float32
                    ).unsqueeze(0).to(device)
                    ctx_n = torch.tensor(
                        load_proprio_context(hf, t_n, k_ctx_proprio),
                        dtype=torch.float32
                    ).unsqueeze(0).to(device)
                    with torch.no_grad():
                        z_a = proprio_enc(ctx_a)
                        z_p = proprio_enc(ctx_p)
                        z_n = proprio_enc(ctx_n)
                    s_p = float((z_a * z_p).sum())
                    s_n = float((z_a * z_n).sum())
                    conditions["proprio_only"][0].append(s_p)
                    conditions["proprio_only"][1].append(s_n)

                # Fusion ViT-G + proprio
                if "fusion_vitg" in conditions:
                    # Already computed z_a, z_p, z_n for both — fuse
                    vg_a = head_vg(extract_vjepa2(encoder_vg, b_a, device).unsqueeze(0).to(device))
                    vg_p = head_vg(extract_vjepa2(encoder_vg, b_p, device).unsqueeze(0).to(device))
                    vg_n = head_vg(extract_vjepa2(encoder_vg, b_n, device).unsqueeze(0).to(device))
                    pr_a = proprio_enc(torch.tensor(load_proprio_context(hf, t_a, k_ctx_proprio), dtype=torch.float32).unsqueeze(0).to(device))
                    pr_p = proprio_enc(torch.tensor(load_proprio_context(hf, t_p, k_ctx_proprio), dtype=torch.float32).unsqueeze(0).to(device))
                    pr_n = proprio_enc(torch.tensor(load_proprio_context(hf, t_n, k_ctx_proprio), dtype=torch.float32).unsqueeze(0).to(device))
                    # Concatenate and re-normalise
                    fa = F.normalize(torch.cat([vg_a, pr_a], dim=-1), dim=-1)
                    fp_ = F.normalize(torch.cat([vg_p, pr_p], dim=-1), dim=-1)
                    fn = F.normalize(torch.cat([vg_n, pr_n], dim=-1), dim=-1)
                    conditions["fusion_vitg"][0].append(float((fa * fp_).sum()))
                    conditions["fusion_vitg"][1].append(float((fa * fn).sum()))

                collected += 1
                pbar.update(1)

        except Exception:
            continue

    pbar.close()
    return conditions


# ── Main ──────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vjepa2-ckpt",    required=True,
                   help="V-JEPA 2 head checkpoint (ViT-G or ViT-L)")
    p.add_argument("--vjepa2-ckpt-l",  default=None,
                   help="Optional ViT-L head checkpoint for comparison")
    p.add_argument("--proprio-ckpt",   default=None,
                   help="Proprio k=32 checkpoint for fusion eval")
    p.add_argument("--hdf5-dir",       required=True)
    p.add_argument("--model-size",     default="giant",
                   choices=["base", "large", "giant"])
    p.add_argument("--n-pairs",        type=int, default=1000)
    p.add_argument("--k-pos",          type=int, default=4)
    p.add_argument("--k-hard-min",     type=int, default=32)
    p.add_argument("--max-files",      type=int, default=None)
    p.add_argument("--hard-negatives", action="store_true", default=True)
    p.add_argument("--proprio-no-gps", action="store_true", default=True)
    p.add_argument("--device",         default="cpu")
    p.add_argument("--full-comparison", action="store_true")
    args = p.parse_args()

    from transformers import AutoModel

    device = args.device
    d_in   = FEAT_DIMS[args.model_size]

    # ── Load V-JEPA 2 ViT-G encoder + head ──────────────────
    print(f"Loading V-JEPA 2 {args.model_size}...")
    encoder_vg = AutoModel.from_pretrained(
        MODEL_IDS[args.model_size], trust_remote_code=True
    ).to(device).eval()

    ckpt_vg = torch.load(args.vjepa2_ckpt, map_location="cpu")
    head_vg = ProjectionHead(d_in=d_in, d_out=64).to(device).eval()
    sd = ckpt_vg.get("head", ckpt_vg)
    head_vg.load_state_dict(sd, strict=False)
    top1_vg  = ckpt_vg.get("top1_acc", "?")
    epoch_vg = ckpt_vg.get("epoch", "?")
    print(f"  ViT-G head loaded: epoch={epoch_vg}, top1_acc={top1_vg}")

    # ── Load V-JEPA 2 ViT-L (optional) ──────────────────────
    encoder_vl, head_vl = None, None
    if args.vjepa2_ckpt_l:
        print("Loading V-JEPA 2 large (for comparison)...")
        encoder_vl = AutoModel.from_pretrained(
            MODEL_IDS["large"], trust_remote_code=True
        ).to(device).eval()
        ckpt_vl = torch.load(args.vjepa2_ckpt_l, map_location="cpu")
        head_vl = ProjectionHead(d_in=1024, d_out=64).to(device).eval()
        head_vl.load_state_dict(ckpt_vl.get("head", ckpt_vl), strict=False)
        top1_vl = ckpt_vl.get("top1_acc", "?")
        print(f"  ViT-L head loaded: top1_acc={top1_vl}")

    # ── Load proprio encoder (optional) ─────────────────────
    proprio_enc = None
    k_ctx_proprio = 32
    if args.proprio_ckpt:
        print("Loading proprio k=32 encoder...")
        ckpt_pr = torch.load(args.proprio_ckpt, map_location="cpu")
        k_ctx_proprio = ckpt_pr.get("k_ctx", 32)
        proprio_enc = ProprioEncoderTemporal(k_ctx=k_ctx_proprio).to(device).eval()
        proprio_enc.load_state_dict(ckpt_pr.get("model", ckpt_pr), strict=False)
        top1_pr = ckpt_pr.get("top1_acc", "?")
        print(f"  Proprio k={k_ctx_proprio} loaded: top1_acc={top1_pr}")

    # ── Files ────────────────────────────────────────────────
    files = list_hdf5_files(args.hdf5_dir, args.max_files)
    print(f"\n  {len(files)} HDF5 files")
    print(f"  {args.n_pairs} pairs | k_pos<={args.k_pos} | "
          f"hard_neg k>={args.k_hard_min}")

    # ── Run eval ─────────────────────────────────────────────
    conditions = run_eval(
        files=files,
        encoder_vg=encoder_vg, head_vg=head_vg,
        encoder_vl=encoder_vl, head_vl=head_vl,
        proprio_enc=proprio_enc,
        n_pairs=args.n_pairs,
        k_pos=args.k_pos,
        k_hard_min=args.k_hard_min,
        hard_negatives=args.hard_negatives,
        proprio_no_gps=args.proprio_no_gps,
        device=device,
        k_ctx_proprio=k_ctx_proprio,
    )

    # ── Print results ─────────────────────────────────────────
    print(f"\n{'='*66}")
    print(f"  V-JEPA 2 Head Dissociation Eval")
    print(f"  n={args.n_pairs} | k_pos<={args.k_pos} | "
          f"hard_neg k>={args.k_hard_min}")
    print(f"{'='*66}")

    label_map = {
        "vjepa2_vitg_head": f"[V-JEPA 2 ViT-G head  (trained, top1={top1_vg:.4f})]",
        "vjepa2_vitl_head": "[V-JEPA 2 ViT-L head  (trained)]",
        "proprio_only":     "[Proprio k=32 only    (No VLM)] ",
        "fusion_vitg":      "[Fusion: ViT-G + proprio]       ",
        "fusion_vitl":      "[Fusion: ViT-L + proprio]       ",
    }

    results = {}
    for key, (pos, neg) in conditions.items():
        if not pos:
            continue
        r = eval_condition(label_map.get(key, key), pos, neg)
        results[key] = r
        status = "PASS" if r["auroc"] > 0.95 else "watch"
        print(f"\n  {r['name']}")
        print(f"    AUROC:      {r['auroc']:.4f}  {status}")
        print(f"    Pos dist:   {r['pos_mean']:.4f} +/- {r['pos_std']:.4f}")
        print(f"    Neg dist:   {r['neg_mean']:.4f} +/- {r['neg_std']:.4f}")
        print(f"    Separation: {r['sep']:+.4f}")

    print(f"\n{'='*66}")

    # ── Summary comparison ───────────────────────────────────
    print("\n  Summary:")
    print(f"  {'System':<35} {'AUROC':>8}  {'Neg dist':>10}")
    print(f"  {'─'*55}")
    for key, r in sorted(results.items(),
                          key=lambda x: x[1]["auroc"], reverse=True):
        name = key.replace("_", " ").replace("vjepa2 ", "V-JEPA 2 ")
        print(f"  {name:<35} {r['auroc']:>8.4f}  {r['neg_mean']:>10.4f}")

    # ── NeurIPS headline ─────────────────────────────────────
    if "fusion_vitg" in results and "proprio_only" in results:
        f_auroc = results["fusion_vitg"]["auroc"]
        p_auroc = results["proprio_only"]["auroc"]
        v_auroc = results["vjepa2_vitg_head"]["auroc"]
        print(f"\n  NeurIPS headline:")
        print(f"  ViT-G head {v_auroc:.4f} + proprio {p_auroc:.4f}")
        print(f"  → Fusion {f_auroc:.4f} "
              f"({'↑' if f_auroc > max(v_auroc, p_auroc) else '='}"
              f"{abs(f_auroc - max(v_auroc, p_auroc)):.4f} over best single)")

    print(f"{'='*66}\n")


if __name__ == "__main__":
    main()
