"""
eval_path_integration_ablation.py  --  NeMo-WM Neuroscience Ablation Series
=============================================================================
Path Integration Channel Ablation.

Biological parallel:
    Lesion studies in rodents selectively disrupt velocity signals (muscle
    spindles, optic flow) vs. head direction signals (anterior thalamus,
    entorhinal cortex) to identify which inputs drive path integration.
    McNaughton et al. (2006), Moser et al. (2008).

Computational analog:
    Zero individual signal channels in the Sprint 6c temporal encoder
    and measure the drop in navigational quasimetric AUROC.

    If velocity is load-bearing  -> zeroing vel causes large AUROC drop
    If heading is load-bearing   -> zeroing heading causes large AUROC drop
    If both are required         -> double dissociation (strongest claim)
    If contact is load-bearing   -> novel finding (contact gating navigation)

Channel groups (matching ProprioEncoderTemporal d_per_frame=8 layout):
    [0]    linear_velocity
    [1]    angular_velocity
    [2:4]  gps displacement (zeroed by --proprio-no-gps, always 0 here)
    [4]    contact
    [5:6]  heading [sin, cos]
    [7]    delta_heading (instantaneous turn rate)

Usage:
    python eval_path_integration_ablation.py ^
        --head-ckpt checkpoints/cwm/temporal_head_sprint3.pt ^
        --cwm-ckpt  checkpoints/cwm/cwm_multidomain_best.pt ^
        --hdf5-dir  recon_data/recon_release ^
        --proprio-ckpt checkpoints/cwm/proprio_6c_best.pt ^
        --n-pairs 500 --k-pos 4

Output:
    Table showing AUROC for each ablation condition.
    Identifies which channels are load-bearing for path integration.
"""

import io, argparse, random, math
from pathlib import Path
import h5py, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    raise ImportError("pip install scikit-learn")
from train_mvtec import StudentEncoder


# ===========================================================================
# Channel ablation definitions
# ===========================================================================

# Each entry: (label, description, channels_to_zero)
# Channels: [vel=0, ang=1, gps_n=2, gps_e=3, contact=4, sin_h=5, cos_h=6, delta_h=7]
ABLATION_CONDITIONS = [
    # Baseline
    ("Full proprio",       "All channels active",                        []),

    # Single-channel knockouts
    ("No velocity",        "Zero linear_velocity [0]",                   [0]),
    ("No angular_vel",     "Zero angular_velocity [1]",                  [1]),
    ("No contact",         "Zero contact [4]",                           [4]),
    ("No heading",         "Zero sin_h + cos_h [5,6]",                  [5, 6]),
    ("No delta_heading",   "Zero instantaneous turn rate [7]",           [7]),

    # Group knockouts
    ("No motion",          "Zero vel + ang [0,1]",                       [0, 1]),
    ("No heading (full)",  "Zero heading + delta_h [5,6,7]",             [5, 6, 7]),
    ("No motion+heading",  "Zero vel+ang+heading+delta_h [0,1,5,6,7]",   [0, 1, 5, 6, 7]),
    ("Contact only",       "Only contact active — zero all else",
                                                                          [0, 1, 2, 3, 5, 6, 7]),
    ("Vel+ang only",       "Zero heading+contact+delta_h",               [4, 5, 6, 7]),
    ("Heading only",       "Zero vel+ang+contact+delta_h",               [0, 1, 4, 7]),

    # Biological parallels
    ("Velocity lesion",    "McNaughton: remove velocity (muscle spindle)", [0]),
    ("HD lesion",          "Moser: remove head direction (ant. thalamus)", [5, 6, 7]),
    ("Velocity+HD lesion", "Complete path integration disruption",         [0, 1, 5, 6, 7]),
]


# ===========================================================================
# Model components (same as eval_recon_auroc.py)
# ===========================================================================

class TemporalHead(nn.Module):
    def __init__(self, d_model=128, embed_dim=64):
        super().__init__()
        self.pool = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        self.proj = nn.Sequential(
            nn.Linear(d_model, embed_dim*2), nn.GELU(),
            nn.Linear(embed_dim*2, embed_dim)
        )
    def forward(self, particles):
        return F.normalize(self.proj(self.pool(particles.mean(dim=1))), dim=-1)


class ParticleEncoder(nn.Module):
    def __init__(self, input_dim=128, K=16, d_model=128):
        super().__init__()
        self.K = K; self.d_model = d_model
        self.proj = nn.Linear(input_dim, K * d_model)
    def forward(self, z):
        B = z.shape[0]
        return F.normalize(self.proj(z).view(B, self.K, self.d_model), dim=-1)


def decode_frame(jpeg_bytes, img_size=224):
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB").resize((img_size, img_size))
    return torch.from_numpy(np.array(img)).float().div(255).permute(2, 0, 1)


# ===========================================================================
# Temporal encoder loader (Sprint 6c)
# ===========================================================================

def load_temporal_encoder(ckpt_path, device):
    """Load Sprint 6c ProprioEncoderTemporal from checkpoint."""
    pd = torch.load(ckpt_path, map_location=device, weights_only=False)
    assert pd.get("arch") == "temporal", \
        f"Expected temporal arch checkpoint, got: {pd.get('arch')}"

    k_ctx       = pd.get("k_ctx",       4)
    d_per_frame = pd.get("d_per_frame", 8)
    d_hidden    = pd.get("d_hidden",   128)
    d_model     = pd.get("d_model",     64)

    def _make_pe(k, d):
        pe  = torch.zeros(k, d)
        pos = torch.arange(k).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)[:, :pe[:, 1::2].shape[1]]
        return pe

    class _Enc(nn.Module):
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

    net = _Enc().to(device)
    net.load_state_dict(pd["model"])
    net.eval()

    acc     = pd.get("top1_acc", "?")
    acc_str = f"{acc:.4f}" if isinstance(acc, float) else str(acc)
    print(f"ProprioEncoderTemporal loaded: {ckpt_path} "
          f"(epoch={pd.get('epoch','?')}, top1_acc={acc_str}, k_ctx={k_ctx})")
    return net, k_ctx, d_per_frame


# ===========================================================================
# Signal extraction
# ===========================================================================

def extract_window(hf, t_end, k_ctx, zero_channels=None):
    """
    Extract k_ctx-frame window ending at t_end.
    zero_channels: list of channel indices to zero (ablation).
    Returns (k_ctx, 8) tensor.
    """
    frames = []
    for i in range(k_ctx):
        t = max(0, t_end - (k_ctx - 1 - i))
        v   = float(hf["commands"]["linear_velocity"][t])
        a   = float(hf["commands"]["angular_velocity"][t])
        ang = hf["commands"]["angular_velocity"][:t+1]
        ha  = float(np.sum(ang)) / 4.0
        sh, ch = math.sin(ha), math.cos(ha)
        dh  = (ha - float(np.sum(hf["commands"]["angular_velocity"][:t])) / 4.0
               if t > 0 else 0.0)
        ct  = 1.0 if abs(v) > 0.3 else 0.0
        frames.append([v, a, 0.0, 0.0, ct, sh, ch, dh])  # GPS always 0

    arr = torch.tensor(frames, dtype=torch.float32)
    arr = torch.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if zero_channels:
        arr[:, zero_channels] = 0.0

    return arr


# ===========================================================================
# Eval dataset (hard negatives, same-file)
# ===========================================================================

class HardNegDataset:
    def __init__(self, hdf5_dir, k_pos=4, n_pairs=500, k_hard_min=32,
                 max_files=None, seed=42):
        random.seed(seed); np.random.seed(seed)
        files = sorted(Path(hdf5_dir).glob("jackal_2019-*.hdf5"))
        if max_files:
            files = files[:max_files]
        self.files = [str(f) for f in files]
        self.k_pos = k_pos
        self.k_hard_min = k_hard_min
        self.pairs = self._build(n_pairs)
        n_pos = sum(1 for p in self.pairs if p["label"] == 1)
        print(f"  HardNegDataset: {n_pos} pos, {len(self.pairs)-n_pos} neg "
              f"(same-file, k>={k_hard_min}), {len(self.files)} files")

    def _build(self, n):
        pairs = []; half = n // 2; att = 0
        # Positive pairs: same file, k <= k_pos
        while len(pairs) < half and att < half * 20:
            att += 1
            f = random.choice(self.files)
            try:
                with h5py.File(f, "r") as hf:
                    L = hf["images"]["rgb_left"].shape[0]
                    if L < self.k_pos + 4: continue
                    t = random.randint(1, L - self.k_pos - 2)
                    k = random.randint(1, self.k_pos)
                    pairs.append({"label": 1, "f": f, "t1": t, "t2": t + k})
            except: continue
        att = 0
        # Hard negatives: same file, gap >= k_hard_min
        while len(pairs) < n and att < half * 20:
            att += 1
            f = random.choice(self.files)
            try:
                with h5py.File(f, "r") as hf:
                    L = hf["images"]["rgb_left"].shape[0]
                    if L < self.k_hard_min + 4: continue
                    t1 = random.randint(1, L - self.k_hard_min - 1)
                    t2 = random.randint(t1 + self.k_hard_min, L - 1)
                    pairs.append({"label": 0, "f": f, "t1": t1, "t2": t2})
            except: continue
        return pairs


# ===========================================================================
# Main ablation eval
# ===========================================================================

@torch.no_grad()
def run_ablation(
    cwm_ckpt, head_ckpt, hdf5_dir, proprio_ckpt,
    n_pairs=500, k_pos=4, k_hard_min=32,
    max_files=None, device_str="cpu",
):
    dev = torch.device(device_str)

    # Load VLM + particle models
    enc = StudentEncoder().to(dev)
    enc_path = Path("checkpoints/dinov2_student/student_best.pt")
    if enc_path.exists():
        sd = torch.load(enc_path, map_location="cpu", weights_only=False)
        enc.load_state_dict(sd.get("model", sd.get("state_dict", sd)), strict=False)
    enc.eval()

    pe = ParticleEncoder(128, 16, 128).to(dev)
    if Path(cwm_ckpt).exists():
        cwm = torch.load(cwm_ckpt, map_location=dev, weights_only=False)
        pe_sd = {k.replace("particle_enc.", ""): v
                 for k, v in cwm.get("model", cwm).items()
                 if k.startswith("particle_enc.")}
        if pe_sd: pe.load_state_dict(pe_sd, strict=False)
    pe.eval()

    head = TemporalHead(128, 64).to(dev)
    if Path(head_ckpt).exists():
        hd = torch.load(head_ckpt, map_location=dev, weights_only=False)
        head.load_state_dict(hd.get("head", hd))
    head.eval()

    # Load 6c temporal encoder
    proprio_net, k_ctx, d_per_frame = load_temporal_encoder(proprio_ckpt, dev)

    # Build eval dataset
    ds = HardNegDataset(hdf5_dir, k_pos=k_pos, n_pairs=n_pairs,
                        k_hard_min=k_hard_min, max_files=max_files)

    # VLM-only baseline (no proprio)
    def embed_vlm(filepath, t):
        with h5py.File(filepath, "r") as hf:
            frame = decode_frame(bytes(hf["images"]["rgb_left"][t])).to(dev)
        z    = enc(frame.unsqueeze(0))
        pts  = pe(z)
        return head(pts).squeeze(0)

    print(f"\n{'='*70}")
    print(f"  Path Integration Channel Ablation (Sprint 6c)")
    print(f"  Hard negatives: same-file, k>={k_hard_min}")
    print(f"  n_pairs={n_pairs}, k_pos<={k_pos}")
    print(f"{'='*70}")

    # Compute VLM-only baseline
    vlm_dists, vlm_labels, errs = [], [], 0
    for pair in ds.pairs:
        try:
            e1 = embed_vlm(pair["f"], pair["t1"])
            e2 = embed_vlm(pair["f"], pair["t2"])
            vlm_dists.append(
                1.0 - F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item()
            )
            vlm_labels.append(pair["label"])
        except Exception:
            errs += 1
    vlm_auroc = roc_auc_score(vlm_labels, [-d for d in vlm_dists])
    print(f"\n  {'VLM-only (no proprio)':<40} AUROC={vlm_auroc:.4f}  "
          f"[biological: landmark navigation]")
    print(f"  {'─'*65}")

    # Run each ablation condition
    results = {}
    for label, desc, zero_chs in ABLATION_CONDITIONS:
        dists, labels, errs = [], [], 0
        for pair in ds.pairs:
            try:
                with h5py.File(pair["f"], "r") as hf:
                    w1 = extract_window(hf, pair["t1"], k_ctx, zero_chs).unsqueeze(0).to(dev)
                    w2 = extract_window(hf, pair["t2"], k_ctx, zero_chs).unsqueeze(0).to(dev)
                z1 = proprio_net(w1).squeeze(0)
                z2 = proprio_net(w2).squeeze(0)
                if z1.isnan().any() or z2.isnan().any():
                    raise ValueError("NaN")
                dists.append(
                    1.0 - F.cosine_similarity(z1.unsqueeze(0), z2.unsqueeze(0)).item()
                )
                labels.append(pair["label"])
            except Exception:
                errs += 1

        if len(labels) < 10:
            print(f"  {label:<40} insufficient pairs")
            continue

        auroc = roc_auc_score(labels, [-d for d in dists])
        delta = auroc - results.get("Full proprio", auroc)
        results[label] = auroc

        # Flag load-bearing channels (drop > 0.05)
        is_full = label == "Full proprio"
        drop    = results["Full proprio"] - auroc if "Full proprio" in results else 0.0
        flag    = " ← LOAD-BEARING" if drop > 0.05 and not is_full else ""
        flag    = " ← CRITICAL"     if drop > 0.15 and not is_full else flag
        errs_str = f"  [{errs} skipped]" if errs > 0 else ""
        print(f"  {label:<40} AUROC={auroc:.4f}  drop={drop:+.4f}{flag}{errs_str}")

    # Summary
    full = results.get("Full proprio", 0.0)
    vl   = results.get("Velocity lesion", full)
    hl   = results.get("HD lesion", full)
    cl   = results.get("Velocity+HD lesion", full)

    print(f"\n{'='*70}")
    print(f"  Path Integration Dissociation Summary")
    print(f"{'='*70}")
    print(f"  Full proprio (path integrator):  {full:.4f}")
    print(f"  VLM-only (landmark navigator):   {vlm_auroc:.4f}")
    print(f"  Velocity lesion:                 {vl:.4f}  (drop={full-vl:+.4f})")
    print(f"  Head direction lesion:           {hl:.4f}  (drop={full-hl:+.4f})")
    print(f"  Complete PI lesion:              {cl:.4f}  (drop={full-cl:+.4f})")

    if full - vl > 0.05 and full - hl > 0.05:
        print(f"\n  DOUBLE DISSOCIATION: both velocity and heading are")
        print(f"  independently load-bearing for path integration.")
        print(f"  Mirrors biological: McNaughton (velocity) + Moser (HD).")
    elif full - vl > full - hl:
        print(f"\n  VELOCITY-DOMINANT: linear velocity is the primary")
        print(f"  path integration signal (cf. McNaughton et al. 2006).")
    else:
        print(f"\n  HEADING-DOMINANT: head direction is the primary")
        print(f"  path integration signal (cf. Moser et al. 2008).")

    print(f"{'='*70}")
    return results, vlm_auroc


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Path integration channel ablation -- NeMo-WM neuroscience series"
    )
    p.add_argument("--cwm-ckpt",     default="checkpoints/cwm/cwm_multidomain_best.pt")
    p.add_argument("--head-ckpt",    default="checkpoints/cwm/temporal_head_sprint3.pt")
    p.add_argument("--hdf5-dir",     default="recon_data/recon_release")
    p.add_argument("--proprio-ckpt", default="checkpoints/cwm/proprio_6c_best.pt")
    p.add_argument("--n-pairs",      type=int, default=500)
    p.add_argument("--k-pos",        type=int, default=4)
    p.add_argument("--k-hard-min",   type=int, default=32)
    p.add_argument("--max-files",    type=int, default=None)
    p.add_argument("--device",       default="cpu")
    args = p.parse_args()

    run_ablation(
        cwm_ckpt    = args.cwm_ckpt,
        head_ckpt   = args.head_ckpt,
        hdf5_dir    = args.hdf5_dir,
        proprio_ckpt= args.proprio_ckpt,
        n_pairs     = args.n_pairs,
        k_pos       = args.k_pos,
        k_hard_min  = args.k_hard_min,
        max_files   = args.max_files,
        device_str  = args.device,
    )
