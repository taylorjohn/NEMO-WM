# train_vit_temporal_split.py
# ViT-Tiny with three improvements over baseline:
#   1. Temporal pairs (t, t+k) instead of augmentation pairs
#   2. Per-signal split projection heads (velocity / spatial / general)
#   3. Fixed NeuroLoss (positive signals, inverse 5HT, no runaway negatives)
#
# Usage:
#   python train_vit_temporal_split.py --max-files 200 --epochs 10 --log-every 200
#   python train_vit_temporal_split.py --max-files 50  --epochs 20 --log-every 100
#
# No unicode -- Windows cp1252 safe
import glob, io, time, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
TRANSFORM_AUG = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---- Architecture --------------------------------------------------------

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=192):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, patch_size)
        n = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        return torch.cat([cls, x], dim=1) + self.pos_embed

class Attention(nn.Module):
    def __init__(self, dim, heads=3):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv  = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        x = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class Block(nn.Module):
    def __init__(self, dim, heads=3):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim))
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ViTTinySplit(nn.Module):
    """
    ViT-Tiny with three split projection heads.

    Instead of one 128-dim output competing for all signals:
      velocity_head  (32-dim): supervised by lin_vel + ang_vel loss
      spatial_head   (32-dim): supervised by NE/GPS loss
      general_head   (64-dim): supervised by DA + 5HT + contrastive

    Total output: 128-dim (same as baseline for fair AIM probe comparison)
    Each dimension specialised for its physical signal.
    """
    def __init__(self, img_size=224, patch_size=16, embed_dim=192,
                 depth=12, heads=3):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        self.blocks      = nn.Sequential(*[Block(embed_dim, heads)
                                           for _ in range(depth)])
        self.norm        = nn.LayerNorm(embed_dim)

        # Split projection heads
        self.vel_head     = nn.Linear(embed_dim, 32)   # velocity signals
        self.spatial_head = nn.Linear(embed_dim, 32)   # GPS / spatial
        self.general_head = nn.Linear(embed_dim, 64)   # general representation

    def forward(self, x):
        x   = self.patch_embed(x)
        x   = self.blocks(x)
        cls = self.norm(x)[:, 0]  # CLS token

        z_vel     = F.normalize(self.vel_head(cls),     dim=-1)
        z_spatial = F.normalize(self.spatial_head(cls), dim=-1)
        z_general = F.normalize(self.general_head(cls), dim=-1)

        # Concatenate for AIM probe (128-dim total, same as baseline)
        return torch.cat([z_vel, z_spatial, z_general], dim=-1)

    def forward_split(self, x):
        """Returns (z_vel, z_spatial, z_general) separately for loss computation."""
        x   = self.patch_embed(x)
        x   = self.blocks(x)
        cls = self.norm(x)[:, 0]
        return (F.normalize(self.vel_head(cls),     dim=-1),
                F.normalize(self.spatial_head(cls), dim=-1),
                F.normalize(self.general_head(cls), dim=-1))

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


# ---- Temporal data loader -----------------------------------------------

def temporal_batches(hdf5_dir, batch_size=16, max_files=None, k=4):
    """
    Yield (img_t, img_t+k, labels_t) batches using temporal pairs.

    Instead of two augmented views of the same frame (SimCLR),
    use frames k steps apart. The encoder must learn what changes
    between frames -- directly velocity and heading.

    k=4 at 4Hz = 1 second gap. Enough motion to create signal,
    not so much that scenes are completely different.
    """
    import h5py
    files = sorted(glob.glob(f"{hdf5_dir}/*.hdf5"))
    if max_files:
        files = files[:max_files]
    b_t, b_tk, bl = [], [], []
    for f in files:
        try:
            with h5py.File(f) as hf:
                imgs = hf["images"]["rgb_left"]
                N    = len(imgs)
                if N < k + 2:
                    continue
                for i in range(N - k - 1):
                    try:
                        # Frame t and frame t+k
                        img_t  = Image.open(
                            io.BytesIO(bytes(imgs[i]))).convert("RGB")
                        img_tk = Image.open(
                            io.BytesIO(bytes(imgs[i + k]))).convert("RGB")

                        # Light augmentation on temporal pairs
                        # (not heavy -- temporal signal should be preserved)
                        b_t.append(TRANSFORM_AUG(img_t))
                        b_tk.append(TRANSFORM(img_tk))

                        lin = float(hf["commands"]["linear_velocity"][i])
                        ang = float(hf["commands"]["angular_velocity"][i])
                        gps = list(hf["gps"]["latlong"][i])
                        bl.append([lin, ang, gps[0], gps[1]])

                        if len(b_t) == batch_size:
                            yield (torch.stack(b_t),
                                   torch.stack(b_tk),
                                   torch.tensor(bl, dtype=torch.float32))
                            b_t, b_tk, bl = [], [], []
                    except:
                        pass
        except:
            pass


# ---- Split NeuroLoss ----------------------------------------------------

class SplitNeuroLoss(nn.Module):
    """
    Fixed NeuroLoss adapted for split projection heads.

    Each loss term targets the head it corresponds to:
      vel_head:     ACh (linear velocity) + eCB (angular velocity)
      spatial_head: NE (GPS consistency)
      general_head: DA (prediction error) + 5HT (diversity) + temporal contrastive

    All terms positive. No runaway negative loss.
    """
    def __init__(self):
        super().__init__()
        # Learnable weights per signal group
        self.w_vel     = nn.Parameter(torch.ones(2))   # ACh, eCB
        self.w_spatial = nn.Parameter(torch.ones(1))   # NE
        self.w_general = nn.Parameter(torch.ones(3))   # DA, 5HT, temporal

    def forward(self, z_vel_t, z_vel_tk,
                z_sp_t,  z_sp_tk,
                z_gen_t, z_gen_tk,
                labels):

        wv = F.softmax(self.w_vel,     dim=0) * 2
        ws = F.softmax(self.w_spatial, dim=0) * 1
        wg = F.softmax(self.w_general, dim=0) * 3

        # === Velocity head losses ===
        lin_vel = labels[:, 0:1].float().abs()
        ang_vel = labels[:, 1:2].float().abs()
        lin_n   = lin_vel / (lin_vel.max() + 1e-6)
        ang_n   = ang_vel / (ang_vel.max() + 1e-6)

        # Temporal embedding change should be proportional to velocity
        vel_diff = (z_vel_t - z_vel_tk).norm(dim=-1, keepdim=True)
        # ACh: linear velocity consistency
        L_ach = F.mse_loss(vel_diff, lin_n)
        # eCB: angular velocity consistency
        L_ecb = F.mse_loss(vel_diff, ang_n)
        L_velocity = wv[0] * L_ach + wv[1] * L_ecb

        # === Spatial head losses ===
        gps    = labels[:, 2:].float()
        gps_n  = F.normalize(gps, dim=1)
        sp_n   = F.normalize(z_sp_t.float(), dim=1)
        # NE: GPS-nearby frames should have similar spatial embeddings
        gps_sim = torch.mm(gps_n, gps_n.t())
        emb_sim = torch.mm(sp_n,  sp_n.t())
        L_ne    = F.mse_loss(emb_sim, gps_sim.clamp(-1, 1))
        L_spatial = ws[0] * L_ne

        # === General head losses ===
        # DA: temporal prediction error (how different are t and t+k)
        L_da = (1.0 - (z_gen_t * z_gen_tk).sum(dim=-1)).clamp(0, 2).mean()

        # 5HT: inverse to DA -- high surprise = low 5HT = penalise low variance
        var   = z_gen_t.var(dim=0).mean()
        da_lv = L_da.detach().clamp(0, 1)
        L_sht = F.relu((1.0 - da_lv) * 0.5 - var)

        # Temporal contrastive: t and t+k of same trajectory should be similar
        # t and t+k from different trajectories should differ
        # Using in-batch negatives within general head
        sim = torch.mm(F.normalize(z_gen_t, dim=1),
                       F.normalize(z_gen_tk, dim=1).t()) / 0.07
        labels_c = torch.arange(z_gen_t.shape[0])
        L_temp = F.cross_entropy(sim, labels_c) * 0.3

        L_general = wg[0]*L_da + wg[1]*L_sht + wg[2]*L_temp

        # Total loss
        L_total = L_velocity + L_spatial + L_general

        signals = {
            "da":   round(L_da.item(),   4),
            "sht":  round(L_sht.item(),  4),
            "ne":   round(L_ne.item(),   4),
            "ach":  round(L_ach.item(),  4),
            "ecb":  round(L_ecb.item(),  4),
            "temp": round(L_temp.item(), 4),
        }
        return L_total, signals


# ---- AIM probe ----------------------------------------------------------

def aim_probe(enc, hdf5_dir, max_files=40, n=800):
    import h5py
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    enc.eval()
    files = sorted(glob.glob(f"{hdf5_dir}/*.hdf5"))[:max_files]
    Z, Y = [], []
    for path in files:
        if len(Z) >= n: break
        try:
            with h5py.File(path) as hf:
                imgs = hf["images"]["rgb_left"]
                for i in range(min(10, len(imgs))):
                    if len(Z) >= n: break
                    img = Image.open(io.BytesIO(bytes(imgs[i]))).convert("RGB")
                    x = TRANSFORM(img).unsqueeze(0)
                    with torch.no_grad():
                        z = enc(x).squeeze(0).numpy()
                    lin = float(hf["commands"]["linear_velocity"][i])
                    ang = float(hf["commands"]["angular_velocity"][i])
                    gps = list(hf["gps"]["latlong"][i])
                    Z.append(z); Y.append([lin, ang, gps[0], gps[1]])
        except: pass
    if len(Z) < 20: return {"n": len(Z), "error": "too few samples"}
    Z = np.array(Z); Y = np.array(Y)
    Zs = StandardScaler().fit_transform(Z)
    sp = int(0.8 * len(Z))
    labels = ["linear_vel", "angular_vel", "gps_lat", "gps_lon"]
    out = {"n": len(Z)}
    for i, lb in enumerate(labels):
        y = Y[:, i]
        r = Ridge(alpha=1.0)
        r.fit(Zs[:sp], y[:sp])
        out[lb] = round(float(r.score(Zs[sp:], y[sp:])), 4)
    return out


# ---- Main ---------------------------------------------------------------

import argparse
p = argparse.ArgumentParser()
p.add_argument("--hdf5-dir",   default="recon_data/recon_release")
p.add_argument("--epochs",     type=int, default=10)
p.add_argument("--max-files",  type=int, default=200)
p.add_argument("--log-every",  type=int, default=200)
p.add_argument("--k",          type=int, default=4,
               help="Temporal gap in frames (default 4 = 1 second at 4Hz)")
p.add_argument("--batch-size", type=int, default=16)
a = p.parse_args()

ckpt = "checkpoints/cwm/vit_tiny_temporal_split.pt"
Path("checkpoints/cwm").mkdir(parents=True, exist_ok=True)

print("\nViT-Tiny Temporal Split -- Phase 2a Next Step")
print("="*60)
print(f"Improvements over baseline:")
print(f"  1. Temporal pairs (t, t+{a.k}) instead of augmentation pairs")
print(f"  2. Split projection heads (vel=32 spatial=32 general=64)")
print(f"  3. Fixed NeuroLoss (positive signals, inverse 5HT)")
print(f"Files: {a.max_files}, Epochs: {a.epochs}, k={a.k}")
print()

enc = ViTTinySplit()
lf  = SplitNeuroLoss()
print(f"Params: {enc.param_count():,}")

params = list(enc.parameters()) + list(lf.parameters())
opt    = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-4)

best, step = float("inf"), 0
t0 = time.perf_counter()

for ep in range(a.epochs):
    losses = []
    for v_t, v_tk, lab in temporal_batches(
            a.hdf5_dir, a.batch_size, a.max_files, a.k):
        zv_t,  zs_t,  zg_t  = enc.forward_split(v_t)
        zv_tk, zs_tk, zg_tk = enc.forward_split(v_tk)

        loss, sig = lf(zv_t, zv_tk, zs_t, zs_tk, zg_t, zg_tk, lab)
        if not torch.isfinite(loss): continue

        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); step += 1; losses.append(loss.item())

        if step % a.log_every == 0:
            elapsed = time.perf_counter() - t0
            sstr = " ".join(f"{k}={v}" for k, v in sig.items())
            print(f"  [ep{ep:02d} s{step:05d}] loss={loss.item():.4f} "
                  f"{sstr} ({elapsed:.0f}s)")

    mean = float(np.mean(losses)) if losses else float("inf")
    print(f"  Epoch {ep:02d} mean={mean:.4f}")
    if mean < best:
        best = mean
        torch.save({"epoch": ep, "loss": best,
                    "model": enc.state_dict()}, ckpt)
        print(f"  -> saved ({best:.4f})")

# AIM probe
print("\n" + "="*60)
print("AIM PROBE (in-distribution, split encoder)")
print("="*60)

# Load all available encoders for comparison
# ViTTinyBase loaded inline

enc_results = {}

# This encoder (split + temporal)
enc.eval()
print("Probing ViT-Tiny Temporal+Split (this run)...")
enc_results["D_temporal_split"] = aim_probe(enc, a.hdf5_dir)
print(f"  n={enc_results['D_temporal_split'].get('n')} samples")

# Load A for comparison if available
ckpt_a = "checkpoints/cwm/vit_tiny_contrastive.pt"
if Path(ckpt_a).exists():
    ea = ViTTinyBase()
    ea.load_state_dict(
        torch.load(ckpt_a, map_location="cpu", weights_only=False)["model"])
    ea.eval()
    print("Probing A_contrastive...")
    enc_results["A_contrastive"] = aim_probe(ea, a.hdf5_dir)

# Load B fixed if available
ckpt_bf = "checkpoints/cwm/vit_tiny_neuromodulated_fixed.pt"
if Path(ckpt_bf).exists():
    ebf = ViTTinyBase()
    ebf.load_state_dict(
        torch.load(ckpt_bf, map_location="cpu", weights_only=False)["model"])
    ebf.eval()
    print("Probing B_neuro_fixed...")
    enc_results["B_neuro_fixed"] = aim_probe(ebf, a.hdf5_dir)

labels = ["linear_vel", "angular_vel", "gps_lat", "gps_lon"]
wins   = {k: 0 for k in enc_results}

print(f"\n{'='*70}")
header = "  Signal              " + "".join(f"{k:>16}" for k in enc_results)
print(header)
print(f"  {'-'*65}")

for lb in labels:
    vals    = {k: enc_results[k].get(lb, float("nan")) for k in enc_results}
    best_k  = max(vals, key=lambda k: vals[k] if not np.isnan(vals[k]) else -999)
    wins[best_k] = wins.get(best_k, 0) + 1
    row = f"  {lb:<20}" + "".join(f"{vals[k]:>16.4f}" for k in enc_results)
    row += f"  <- {best_k}"
    print(row)

print(f"\n  Wins: " + " ".join(f"{k}={v}" for k, v in wins.items()))
best_enc = max(wins, key=wins.get)
print(f"  Overall winner: {best_enc}")

if "D_temporal_split" == best_enc or wins.get("D_temporal_split", 0) >= 3:
    print("\n  TEMPORAL+SPLIT WINS")
    print("  Temporal pairs + per-signal heads = better physical semantics")
    print("  Ready to scale to 500 files or ViT-Small")
elif wins.get("D_temporal_split", 0) >= 2:
    print("\n  PARTIAL WIN -- temporal+split competitive, tune further")
else:
    print("\n  Baseline wins -- check temporal gap k or loss weights")

import json
with open("vlm_phase2a_temporal_split_results.json", "w") as f:
    json.dump({"k": a.k, "max_files": a.max_files,
               "epochs": a.epochs, "results": enc_results,
               "wins": wins}, f, indent=2)
print("  Saved: vlm_phase2a_temporal_split_results.json")
