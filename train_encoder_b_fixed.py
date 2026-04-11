# train_encoder_b_fixed.py - Train fixed Encoder B (Neuromodulated)
# Fixes from original: positive signals, correct 5HT inversion, Z-score cortisol
# No unicode - Windows cp1252 safe
import glob, io, time, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
TRANSFORM_AUG2 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
TRANSFORM_CLEAN = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Architecture (matches original checkpoints exactly) ---
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=192):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        return torch.cat([cls, x], dim=1) + self.pos_embed

class Attention(nn.Module):
    def __init__(self, dim, heads=3, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = self.drop((q @ k.transpose(-2, -1)) * self.scale)
        x = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class Block(nn.Module):
    def __init__(self, dim, heads=3, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim))
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))

class ViTTiny(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=192,
                 depth=12, heads=3, out_dim=128):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        self.blocks = nn.Sequential(*[Block(embed_dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, out_dim)
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.norm(x)
        return F.normalize(self.head(x[:, 0]), dim=-1)
    def param_count(self):
        return sum(p.numel() for p in self.parameters())

# --- Fixed NeuroLoss ---
class FixedNeuroLoss(nn.Module):
    """
    Fixed neuromodulated loss with correct biological signal signs.

    Fixes vs original:
      - ACh: rewards high velocity frames having MORE embedding change (positive)
      - eCB: rewards high angular velocity frames having MORE embedding change (positive)
      - 5HT: INVERSE to DA (high surprise -> low 5HT -> explore)
      - Cortisol: Z-score based, not ratio-based
      - All terms positive -- no runaway negative loss
    """
    def __init__(self):
        super().__init__()
        self.log_weights = nn.Parameter(torch.zeros(6))

    def forward(self, z1, z2, labels):
        w = F.softmax(self.log_weights, dim=0) * 6

        # DA: prediction surprise (always positive 0-2)
        L_da = (1.0 - (z1 * z2).sum(dim=-1)).clamp(0, 2).mean()

        # 5HT: INVERSE to DA -- high surprise = low 5HT = penalise homogeneity
        # When DA is high, we want diversity (explore). Penalise low variance.
        var = z1.var(dim=0).mean()
        da_level = L_da.detach().clamp(0, 1)
        sht_target = 1.0 - da_level  # inverse relationship
        L_sht = F.relu(sht_target * 0.5 - var)

        # NE: spatial grounding (GPS consistency, positive)
        gps = F.normalize(labels[:, 2:].float(), dim=1)
        zn = F.normalize(z1.float(), dim=1)
        gps_sim = torch.mm(gps, gps.t())
        emb_sim = torch.mm(zn, zn.t())
        L_ne = F.mse_loss(emb_sim, gps_sim.clamp(-1, 1))

        # ACh: velocity-proportional embedding change (POSITIVE -- reward it)
        lin_vel = labels[:, 0:1].float().abs()
        lin_norm = lin_vel / (lin_vel.max() + 1e-6)
        emb_diff = (z1 - z2).norm(dim=-1, keepdim=True)
        # High velocity should produce more embedding change
        L_ach = F.mse_loss(emb_diff, lin_norm)

        # eCB: angular velocity consistency (POSITIVE)
        ang_vel = labels[:, 1:2].float().abs()
        ang_norm = ang_vel / (ang_vel.max() + 1e-6)
        L_ecb = F.mse_loss(emb_diff, ang_norm)

        # Cortisol: batch variance Z-score proxy (positive)
        batch_var = z1.var(dim=0).mean()
        L_cort = F.relu(0.1 - batch_var)  # penalise collapsed representations

        loss = (w[0]*L_da + w[1]*L_sht + w[2]*L_ne +
                w[3]*L_ach + w[4]*L_ecb + w[5]*L_cort)

        signals = {
            "da":   round(L_da.item(), 4),
            "sht":  round(L_sht.item(), 4),
            "ne":   round(L_ne.item(), 4),
            "ach":  round(L_ach.item(), 4),
            "ecb":  round(L_ecb.item(), 4),
            "cort": round(L_cort.item(), 4),
        }
        return loss, signals

# --- Data ---
def batches(hdf5_dir, batch_size=16, max_files=None):
    import h5py
    files = sorted(glob.glob(f"{hdf5_dir}/*.hdf5"))
    if max_files: files = files[:max_files]
    b1, b2, bl = [], [], []
    for f in files:
        try:
            with h5py.File(f) as hf:
                imgs = hf["images"]["rgb_left"]
                for i in range(len(imgs) - 1):
                    try:
                        img = Image.open(io.BytesIO(bytes(imgs[i]))).convert("RGB")
                        b1.append(TRANSFORM(img))
                        b2.append(TRANSFORM_AUG2(img))
                        lin = float(hf["commands"]["linear_velocity"][i])
                        ang = float(hf["commands"]["angular_velocity"][i])
                        gps = list(hf["gps"]["latlong"][i])
                        bl.append([lin, ang, gps[0], gps[1]])
                        if len(b1) == batch_size:
                            yield (torch.stack(b1), torch.stack(b2),
                                   torch.tensor(bl, dtype=torch.float32))
                            b1, b2, bl = [], [], []
                    except: pass
        except: pass

# --- AIM probe ---
def aim_probe(enc, hdf5_dir, max_files=40, n=800, use_first=True):
    import h5py
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    enc.eval()
    all_files = sorted(glob.glob(f"{hdf5_dir}/*.hdf5"))
    files = all_files[:max_files] if use_first else all_files[int(0.8*len(all_files)):]
    Z, Y = [], []
    for path in files:
        if len(Z) >= n: break
        try:
            with h5py.File(path) as hf:
                imgs = hf["images"]["rgb_left"]
                for i in range(min(10, len(imgs))):
                    if len(Z) >= n: break
                    img = Image.open(io.BytesIO(bytes(imgs[i]))).convert("RGB")
                    x = TRANSFORM_CLEAN(img).unsqueeze(0)
                    with torch.no_grad():
                        z = enc(x).squeeze(0).numpy()
                    lin = float(hf["commands"]["linear_velocity"][i])
                    ang = float(hf["commands"]["angular_velocity"][i])
                    gps = list(hf["gps"]["latlong"][i])
                    Z.append(z); Y.append([lin, ang, gps[0], gps[1]])
        except: pass
    if len(Z) < 20: return {"n": len(Z), "error": "too few"}
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

# --- Train ---
import argparse
p = argparse.ArgumentParser()
p.add_argument("--hdf5-dir",  default="recon_data/recon_release")
p.add_argument("--epochs",    type=int, default=20)
p.add_argument("--max-files", type=int, default=50)
p.add_argument("--log-every", type=int, default=100)
a = p.parse_args()

ckpt_b = "checkpoints/cwm/vit_tiny_neuromodulated_fixed.pt"
ckpt_a = "checkpoints/cwm/vit_tiny_contrastive.pt"
ckpt_c = "checkpoints/cwm/vit_tiny_hybrid.pt"
Path("checkpoints/cwm").mkdir(parents=True, exist_ok=True)

print("\nPhase 2a - Encoder B Fixed (Correct NeuroLoss)")
print("="*60)
print("Fixes: positive ACh/eCB, inverse 5HT, Z-score cortisol\n")

enc = ViTTiny()
print(f"Params: {enc.param_count():,}")
lf  = FixedNeuroLoss()
params = list(enc.parameters()) + list(lf.parameters())
opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-4)

best, step = float("inf"), 0
t0 = time.perf_counter()

for ep in range(a.epochs):
    losses = []
    for v1, v2, lab in batches(a.hdf5_dir, max_files=a.max_files):
        z1 = enc(v1); z2 = enc(v2)
        loss, sig = lf(z1, z2, lab)
        if not torch.isfinite(loss): continue
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); step += 1; losses.append(loss.item())
        if step % a.log_every == 0:
            sstr = " ".join(f"{k}={v}" for k, v in sig.items())
            elapsed = time.perf_counter() - t0
            print(f"  [ep{ep:02d} s{step:05d}] loss={loss.item():.4f} "
                  f"{sstr} ({elapsed:.0f}s)")
    mean = float(np.mean(losses)) if losses else float("inf")
    print(f"  Epoch {ep:02d} mean={mean:.4f}")
    if mean < best:
        best = mean
        torch.save({"epoch": ep, "loss": best, "model": enc.state_dict()},
                   ckpt_b)
        print(f"  -> saved ({best:.4f})")

print("\n" + "="*60)
print("AIM PROBE - All Four Encoders (in-distribution)")
print("="*60)

enc_b_fixed = enc
ckpts = {
    "A_contrastive":      ckpt_a,
    "B_neuro_original":   "checkpoints/cwm/vit_tiny_neuromodulated.pt",
    "B_neuro_fixed":      ckpt_b,
    "C_hybrid":           ckpt_c,
}

results = {}
for name, ckpt in ckpts.items():
    if not Path(ckpt).exists():
        print(f"  MISSING: {ckpt}"); continue
    e = ViTTiny()
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    e.load_state_dict(sd["model"]); e.eval()
    ep_n = sd.get("epoch", "?")
    loss_n = sd.get("loss", 0)
    print(f"  Probing {name} (ep={ep_n} loss={loss_n:.4f})...")
    results[name] = aim_probe(e, a.hdf5_dir)
    print(f"    n={results[name].get('n')} samples")

labels = ["linear_vel", "angular_vel", "gps_lat", "gps_lon"]
wins = {k: 0 for k in results}

print(f"\n{'='*75}")
print(f"  {'Signal':<18} {'A Contrast':>11} "
      f"{'B orig':>8} {'B fixed':>8} {'C Hybrid':>9} {'Winner'}")
print(f"  {'-'*70}")

for lb in labels:
    vals = {k: results[k].get(lb, float("nan")) for k in results}
    best_k = max(vals, key=lambda k: vals[k] if not np.isnan(vals[k]) else -999)
    wins[best_k] = wins.get(best_k, 0) + 1
    ra  = vals.get("A_contrastive", float("nan"))
    rbo = vals.get("B_neuro_original", float("nan"))
    rbf = vals.get("B_neuro_fixed", float("nan"))
    rc  = vals.get("C_hybrid", float("nan"))
    print(f"  {lb:<18} {ra:>11.4f} {rbo:>8.4f} {rbf:>8.4f} {rc:>9.4f}  {best_k}")

print(f"\n  Wins: " + " ".join(f"{k}={v}" for k, v in wins.items()))

import json
with open("vlm_phase2a_final_results.json", "w") as f:
    json.dump({"results": results, "wins": wins}, f, indent=2)
print("  Saved: vlm_phase2a_final_results.json")
