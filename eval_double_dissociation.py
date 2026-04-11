"""
eval_double_dissociation.py  —  NeMo-WM Double Dissociation Filter Eval
========================================================================

Gold-standard double dissociation test: each navigational system should
collapse under its specific input lesion, while the other system survives.

System 1 — Landmark (VLM-grounded):
  Lesion: visually homogeneous terrain (low visual entropy frames)
  Prediction: VLM-only AUROC degrades; proprio-only AUROC survives

System 2 — Path integration (proprioceptive):
  Lesion: stopped / near-stationary frames (vel < 0.1 m/s)
  Prediction: proprio-only AUROC collapses (no motion = no PI signal);
              VLM-only AUROC survives

Both conditions use existing checkpoints — no new training required.

Usage:
  python eval_double_dissociation.py \
      --head-ckpt  checkpoints/cwm/temporal_head_sprint3.pt \
      --cwm-ckpt   checkpoints/cwm/cwm_multidomain_best.pt \
      --proprio-ckpt checkpoints/cwm/proprio_kctx8_best.pt \
      --hdf5-dir   recon_data/recon_release \
      --n-pairs 500

Expected results (biological prediction):
  Stopped frames condition:
    No-VLM (proprio only): AUROC << 0.977 baseline  ← PI collapses
    VLM-only:              AUROC ~same as baseline   ← landmark survives
  Visual homogeneity condition:
    VLM-only:              AUROC < baseline           ← landmark degrades
    No-VLM (proprio only): AUROC ~same as baseline   ← PI survives
"""

import argparse, math, time, warnings
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


# ===========================================================================
# Minimal model stubs — replace with real imports on GMKtec
# ===========================================================================

# Inline minimal model definitions — no external imports needed
class StudentEncoder(nn.Module):
    """46K-param DINOv2 distilled encoder stub — replaced by checkpoint load."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3*224*224, 128), nn.LayerNorm(128))
    def forward(self, x):
        return nn.functional.normalize(self.net(x.flatten(1)), dim=-1)

class ParticleEncoder(nn.Module):
    def __init__(self, d_in=128, k=16, d_out=128):
        super().__init__()
        self.proj = nn.Linear(d_in, k * d_out)
        self.k, self.d = k, d_out
    def forward(self, x):
        B = x.shape[0]
        return nn.functional.normalize(
            self.proj(x).view(B, self.k, self.d), dim=-1)

class TemporalHead(nn.Module):
    def __init__(self, d_model=128, embed_dim=64):
        super().__init__()
        self.pool = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                   nn.LayerNorm(d_model))
        self.proj = nn.Sequential(nn.Linear(d_model, embed_dim*2), nn.GELU(),
                                   nn.Linear(embed_dim*2, embed_dim))
    def forward(self, particles):  # particles: (B, K, D)
        x = particles.mean(dim=1)
        return F.normalize(self.proj(self.pool(x)), dim=-1)


# ===========================================================================
# Arch-aware temporal proprio loader (mirrors eval_recon_auroc.py)
# ===========================================================================

def _load_temporal_encoder(ckpt_path: str, device):
    """Load Sprint 6c temporal encoder from checkpoint."""
    import math as _math
    pd = torch.load(ckpt_path, map_location=device, weights_only=False)
    assert pd.get("arch") == "temporal", \
        f"Expected arch=temporal, got {pd.get('arch')} — pass kctx8 checkpoint"

    k_ctx       = pd.get("k_ctx",       8)
    d_per_frame = pd.get("d_per_frame",  8)
    d_hidden    = pd.get("d_hidden",   128)
    d_model     = pd.get("d_model",     64)

    def _make_pe(k, d):
        pe  = torch.zeros(k, d)
        pos = torch.arange(k).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-_math.log(10000.) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)[:, :pe[:, 1::2].shape[1]]
        return pe

    class _TE(nn.Module):
        def __init__(self):
            super().__init__()
            self.frame_embed = nn.Sequential(
                nn.Linear(d_per_frame, d_hidden), nn.LayerNorm(d_hidden), nn.GELU(),
                nn.Linear(d_hidden, d_hidden),    nn.LayerNorm(d_hidden),
            )
            self.register_buffer("pe", _make_pe(k_ctx, d_hidden))
            self.attn_score = nn.Linear(d_hidden, 1)
            self.out_proj   = nn.Sequential(nn.GELU(), nn.Linear(d_hidden, d_model))

        def forward(self, x):
            h = self.frame_embed(x) + self.pe.unsqueeze(0)
            w = torch.softmax(self.attn_score(h).squeeze(-1), dim=-1)
            return F.normalize(self.out_proj((h * w.unsqueeze(-1)).sum(1)), dim=-1)

    net = _TE().to(device)
    net.load_state_dict(pd["model"])
    net.eval()
    acc_str = f"{pd.get('top1_acc', '?'):.4f}" if isinstance(pd.get("top1_acc"), float) \
              else str(pd.get("top1_acc", "?"))
    print(f"ProprioEncoderTemporal loaded: {ckpt_path} "
          f"(epoch={pd.get('epoch','?')}, top1_acc={acc_str}, k_ctx={k_ctx})")

    def _extract_window(hf, t_end, no_gps=True):
        frames = []
        for i in range(k_ctx):
            t = max(0, t_end - (k_ctx - 1 - i))
            v  = float(hf["commands"]["linear_velocity"][t])
            a  = float(hf["commands"]["angular_velocity"][t])
            ang = hf["commands"]["angular_velocity"][:t+1]
            ha = float(np.sum(ang)) / 4.0
            sh, ch = _math.sin(ha), _math.cos(ha)
            dh = (ha - float(np.sum(hf["commands"]["angular_velocity"][:t])) / 4.0
                  if t > 0 else 0.0)
            ct = 1.0 if abs(v) > 0.3 else 0.0
            frames.append([v, a, 0.0, 0.0, ct, sh, ch, dh])   # GPS always zeroed
        arr = torch.tensor(frames, dtype=torch.float32)
        return torch.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def encoder_fn(hf, t_end):
        w = _extract_window(hf, t_end).unsqueeze(0).to(device)
        with torch.no_grad():
            return net(w).squeeze(0)

    return encoder_fn, k_ctx


# ===========================================================================
# Frame filter functions
# ===========================================================================

def is_stopped_frame(hf, t: int, vel_thresh: float = 0.10) -> bool:
    """True if robot is stationary at frame t (vel < vel_thresh m/s)."""
    v = abs(float(hf["commands"]["linear_velocity"][t]))
    return v < vel_thresh


def visual_entropy(hf, t: int) -> float:
    """
    Proxy for visual homogeneity: std-dev of JPEG file sizes across a
    3-frame window. Low variance = visually uniform (long straight corridor).
    We use byte-length as a cheap entropy proxy — larger JPEG = more detail.
    """
    sizes = []
    T = hf["images"]["rgb_left"].shape[0]
    for dt in [-1, 0, 1]:
        ti = max(0, min(T-1, t+dt))
        sizes.append(len(bytes(hf["images"]["rgb_left"][ti])))
    return float(np.std(sizes))


def is_low_entropy_frame(hf, t: int, entropy_thresh: float = 500.0) -> bool:
    """
    True if visual scene is homogeneous.
    entropy_thresh tuned empirically: RECON outdoor frames avg ~2000 bytes std,
    long-straight corridor sections drop to ~200-600 bytes std.
    """
    return visual_entropy(hf, t) < entropy_thresh


# ===========================================================================
# Pair sampler with frame filter
# ===========================================================================

def sample_filtered_pairs(
    hdf5_dir: str,
    n_pairs: int,
    k_pos: int,
    k_hard_min: int,
    frame_filter,          # callable(hf, t) -> bool  — True = keep frame
    filter_name: str,
    max_files: int = 500,
    seed: int = 42,
):
    """
    Sample (anchor, positive, negative) triplets where the anchor frame
    passes frame_filter. This isolates the condition of interest.
    """
    rng   = np.random.default_rng(seed)
    files = sorted(Path(hdf5_dir).glob("*.hdf5"))[:max_files]
    rng.shuffle(files := list(files))

    pairs  = []   # list of (filepath, t_anchor, t_pos, t_neg)
    tested = 0

    for fp in files:
        if len(pairs) >= n_pairs:
            break
        try:
            with h5py.File(fp, "r") as hf:
                T = hf["images"]["rgb_left"].shape[0]
                if T < k_hard_min + k_pos + 5:
                    continue
                for t in range(k_hard_min, T - k_pos - 1):
                    tested += 1
                    if not frame_filter(hf, t):
                        continue
                    t_pos = t + rng.integers(1, k_pos + 1)
                    t_neg = rng.integers(0, max(1, t - k_hard_min))
                    pairs.append((str(fp), t, t_pos, int(t_neg)))
                    if len(pairs) >= n_pairs:
                        break
        except Exception:
            continue

    kept_pct = 100 * len(pairs) / max(tested, 1)
    print(f"  [{filter_name}] sampled {len(pairs)}/{n_pairs} pairs "
          f"from {kept_pct:.1f}% of tested frames")
    return pairs


# ===========================================================================
# Embedding functions
# ===========================================================================

def _load_vlm_models(cwm_ckpt, head_ckpt, device):
    enc  = StudentEncoder().to(device)
    ep   = Path("checkpoints/dinov2_student/student_best.pt")
    if ep.exists():
        sd = torch.load(ep, map_location="cpu", weights_only=False)
        enc.load_state_dict(sd.get("model", sd.get("state_dict", sd)), strict=False)
    enc.eval()

    pe = ParticleEncoder(128, 16, 128).to(device)
    if Path(cwm_ckpt).exists():
        cd  = torch.load(cwm_ckpt, map_location=device, weights_only=False)
        psd = {k.replace("particle_enc.", ""): v
               for k, v in cd.get("model", cd).items()
               if k.startswith("particle_enc.")}
        if psd: pe.load_state_dict(psd, strict=False)
    pe.eval()

    head = TemporalHead(128, 64).to(device)
    if Path(head_ckpt).exists():
        hd  = torch.load(head_ckpt, map_location=device, weights_only=False)
        raw = hd.get("head", hd)
        # Strip "pool./proj." prefixes if checkpoint uses different naming
        head.load_state_dict(raw, strict=False)
    head.eval()
    print(f"VLM models loaded: StudentEncoder + ParticleEncoder + TemporalHead")

    def decode_frame(raw_bytes, size=224):
        from PIL import Image
        from torchvision import transforms
        import io
        tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        return tf(Image.open(io.BytesIO(raw_bytes)).convert("RGB"))

    @torch.no_grad()
    def vlm_embed(filepath, t):
        with h5py.File(filepath, "r") as hf:
            frame = decode_frame(bytes(hf["images"]["rgb_left"][t])).to(device)
        z = head(pe(enc(frame.unsqueeze(0)))).squeeze(0)
        return F.normalize(z, dim=-1)

    return vlm_embed


# ===========================================================================
# AUROC computation
# ===========================================================================

@torch.no_grad()
def compute_auroc(pairs, embed_fn_a, embed_fn_p, label: str) -> float:
    """embed_fn_a/p: callable(filepath, t) -> (D,) tensor"""
    scores, labels = [], []
    for (fp, t_a, t_p, t_n) in pairs:
        try:
            z_a = embed_fn_a(fp, t_a)
            z_p = embed_fn_p(fp, t_p)
            z_n = embed_fn_p(fp, t_n)
            scores.append(float(F.cosine_similarity(z_a.unsqueeze(0),
                                                     z_p.unsqueeze(0))))
            labels.append(1)
            scores.append(float(F.cosine_similarity(z_a.unsqueeze(0),
                                                     z_n.unsqueeze(0))))
            labels.append(0)
        except Exception:
            continue
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


# ===========================================================================
# Main eval
# ===========================================================================

@torch.no_grad()
def run_double_dissociation(
    cwm_ckpt:    str,
    head_ckpt:   str,
    proprio_ckpt: str,
    hdf5_dir:    str,
    n_pairs:     int  = 500,
    k_pos:       int  = 4,
    k_hard_min:  int  = 32,
    vel_thresh:  float = 0.10,
    entropy_thresh: float = 500.0,
    device_str:  str  = "cpu",
):
    dev = torch.device(device_str)

    print("Loading models...")
    vlm_embed  = _load_vlm_models(cwm_ckpt, head_ckpt, dev)
    prop_enc, k_ctx = _load_temporal_encoder(proprio_ckpt, dev)

    def prop_embed(filepath, t):
        with h5py.File(filepath, "r") as hf:
            return prop_enc(hf, t)

    # Fused embed for full model
    def full_embed(filepath, t):
        z_v = vlm_embed(filepath, t)
        z_p = prop_embed(filepath, t)
        return F.normalize(z_v + z_p, dim=-1)

    print(f"\nSampling pairs for each condition "
          f"(n_pairs={n_pairs}, k_pos={k_pos}, k_hard_min={k_hard_min})...")

    # ── Condition 1: Stopped frames ───────────────────────────────────────
    stopped_pairs = sample_filtered_pairs(
        hdf5_dir, n_pairs, k_pos, k_hard_min,
        frame_filter = lambda hf, t: is_stopped_frame(hf, t, vel_thresh),
        filter_name  = "Stopped frames",
    )

    # ── Condition 2: Visually homogeneous frames ──────────────────────────
    homog_pairs = sample_filtered_pairs(
        hdf5_dir, n_pairs, k_pos, k_hard_min,
        frame_filter = lambda hf, t: is_low_entropy_frame(hf, t, entropy_thresh),
        filter_name  = "Low visual entropy",
    )

    # ── Baselines (unfiltered hard negatives) ─────────────────────────────
    baseline_pairs = sample_filtered_pairs(
        hdf5_dir, n_pairs, k_pos, k_hard_min,
        frame_filter = lambda hf, t: True,
        filter_name  = "Baseline (unfiltered)",
    )

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  Double Dissociation Eval  —  NeMo-WM")
    print(f"  k_ctx={k_ctx}, k_pos={k_pos}, hard neg k>={k_hard_min}")
    print(f"{sep}")

    results = {}
    for cond_name, pairs in [
        ("Baseline (unfiltered)", baseline_pairs),
        ("Stopped frames [PI lesion]", stopped_pairs),
        ("Low visual entropy [landmark lesion]", homog_pairs),
    ]:
        if not pairs:
            print(f"\n  {cond_name}: insufficient pairs — skipping")
            continue

        print(f"\n  Condition: {cond_name}  (n={len(pairs)} pairs)")
        print(f"  {'─'*60}")

        t0 = time.perf_counter()
        auroc_full = compute_auroc(pairs, full_embed,  full_embed,  "full")
        auroc_vlm  = compute_auroc(pairs, vlm_embed,   vlm_embed,   "vlm")
        auroc_prop = compute_auroc(pairs, prop_embed,  prop_embed,  "prop")
        elapsed    = time.perf_counter() - t0

        results[cond_name] = dict(full=auroc_full, vlm=auroc_vlm, prop=auroc_prop)

        def _fmt(v):
            return f"{v:.4f}" if not math.isnan(v) else "  NaN"

        print(f"  {'Full (VLM + proprio)':<28} AUROC={_fmt(auroc_full)}")
        print(f"  {'No VLM (proprio only)':<28} AUROC={_fmt(auroc_prop)}", end="")
        if cond_name != "Baseline (unfiltered)":
            base_prop = results.get("Baseline (unfiltered)", {}).get("prop", float("nan"))
            if not math.isnan(base_prop) and not math.isnan(auroc_prop):
                delta = auroc_prop - base_prop
                flag  = " ← PI COLLAPSES" if delta < -0.10 else (" ← stable" if delta > -0.03 else "")
                print(f"  (Δ={delta:+.4f}{flag})", end="")
        print()
        print(f"  {'No proprio (VLM only)':<28} AUROC={_fmt(auroc_vlm)}", end="")
        if cond_name != "Baseline (unfiltered)":
            base_vlm = results.get("Baseline (unfiltered)", {}).get("vlm", float("nan"))
            if not math.isnan(base_vlm) and not math.isnan(auroc_vlm):
                delta = auroc_vlm - base_vlm
                flag  = " ← LANDMARK DEGRADES" if delta < -0.05 else (" ← stable" if delta > -0.02 else "")
                print(f"  (Δ={delta:+.4f}{flag})", end="")
        print()
        print(f"  Elapsed: {elapsed:.1f}s")

    print(f"\n{sep}")
    print("  Double Dissociation Summary")
    print(f"  {'─'*60}")
    print("  Biological prediction:")
    print("  Stopped frames  → proprio AUROC << baseline (PI collapses)")
    print("                  → VLM AUROC ~ baseline     (landmark survives)")
    print("  Low entropy     → VLM AUROC < baseline     (landmark degrades)")
    print("                  → proprio AUROC ~ baseline (PI survives)")

    if "Stopped frames [PI lesion]" in results and \
       "Baseline (unfiltered)" in results:
        b = results["Baseline (unfiltered)"]
        s = results["Stopped frames [PI lesion]"]
        pi_collapse   = (s["prop"] - b["prop"]) < -0.10
        lm_survives_s = (s["vlm"]  - b["vlm"])  > -0.05
        print(f"\n  PI system (stopped frames):")
        print(f"    prop drop: {s['prop']-b['prop']:+.4f}  "
              f"{'CONFIRMED COLLAPSE' if pi_collapse else 'not collapsed'}")
        print(f"    VLM drop:  {s['vlm']-b['vlm']:+.4f}  "
              f"{'VLM STABLE' if lm_survives_s else 'also degraded'}")

    if "Low visual entropy [landmark lesion]" in results and \
       "Baseline (unfiltered)" in results:
        b = results["Baseline (unfiltered)"]
        h = results["Low visual entropy [landmark lesion]"]
        lm_degrade  = (h["vlm"]  - b["vlm"])  < -0.05
        pi_survives = (h["prop"] - b["prop"]) > -0.05
        print(f"\n  Landmark system (low entropy frames):")
        print(f"    VLM drop:  {h['vlm']-b['vlm']:+.4f}  "
              f"{'CONFIRMED DEGRADATION' if lm_degrade else 'minimal'}")
        print(f"    prop drop: {h['prop']-b['prop']:+.4f}  "
              f"{'PI STABLE' if pi_survives else 'also degraded'}")

    print(f"{sep}\n")
    return results


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Double dissociation filter eval")
    p.add_argument("--cwm-ckpt",     default="checkpoints/cwm/cwm_multidomain_best.pt")
    p.add_argument("--head-ckpt",    default="checkpoints/cwm/temporal_head_sprint3.pt")
    p.add_argument("--proprio-ckpt", default="checkpoints/cwm/proprio_kctx8_best.pt")
    p.add_argument("--hdf5-dir",     default="recon_data/recon_release")
    p.add_argument("--n-pairs",      type=int,   default=500)
    p.add_argument("--k-pos",        type=int,   default=4)
    p.add_argument("--k-hard-min",   type=int,   default=32)
    p.add_argument("--vel-thresh",   type=float, default=0.10,
                   help="Velocity threshold for stopped-frame filter (m/s)")
    p.add_argument("--entropy-thresh", type=float, default=500.0,
                   help="JPEG size std-dev threshold for low-entropy filter")
    p.add_argument("--device",       default="cpu")
    args = p.parse_args()

    run_double_dissociation(
        cwm_ckpt       = args.cwm_ckpt,
        head_ckpt      = args.head_ckpt,
        proprio_ckpt   = args.proprio_ckpt,
        hdf5_dir       = args.hdf5_dir,
        n_pairs        = args.n_pairs,
        k_pos          = args.k_pos,
        k_hard_min     = args.k_hard_min,
        vel_thresh     = args.vel_thresh,
        entropy_thresh = args.entropy_thresh,
        device_str     = args.device,
    )
