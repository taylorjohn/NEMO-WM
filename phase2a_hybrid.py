# phase2a_hybrid.py - Train Encoder C (Hybrid) using saved A+B checkpoints
# Run after vlm_phase2a_pretrain.py finishes A and B
# No unicode characters - Windows cp1252 safe
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

# --- ViT-Tiny ---
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=192):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        n = (img_size // patch_size) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = nn.Parameter(torch.zeros(1, n+1, embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1,2)
        x = torch.cat([self.cls.expand(x.shape[0],-1,-1), x], dim=1) + self.pos
        return x

class Attn(nn.Module):
    def __init__(self, dim, heads=3):
        super().__init__()
        self.h = heads; self.s = (dim//heads)**-0.5
        self.qkv = nn.Linear(dim, dim*3)
        self.proj = nn.Linear(dim, dim)
    def forward(self, x):
        B,N,C = x.shape
        qkv = self.qkv(x).reshape(B,N,3,self.h,C//self.h).permute(2,0,3,1,4)
        q,k,v = qkv.unbind(0)
        a = (q @ k.transpose(-2,-1)) * self.s
        x = (a.softmax(-1) @ v).transpose(1,2).reshape(B,N,C)
        return self.proj(x)

class Block(nn.Module):
    def __init__(self, dim, heads=3):
        super().__init__()
        self.n1 = nn.LayerNorm(dim); self.a = Attn(dim, heads)
        self.n2 = nn.LayerNorm(dim)
        mlp = int(dim*4)
        self.m = nn.Sequential(nn.Linear(dim,mlp), nn.GELU(), nn.Linear(mlp,dim))
    def forward(self, x):
        x = x + self.a(self.n1(x))
        return x + self.m(self.n2(x))

class ViTTiny(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.pe = PatchEmbed()
        self.blocks = nn.Sequential(*[Block(192) for _ in range(12)])
        self.norm = nn.LayerNorm(192)
        self.head = nn.Linear(192, out_dim)
    def forward(self, x):
        x = self.pe(x); x = self.blocks(x); x = self.norm(x)
        return F.normalize(self.head(x[:,0]), dim=-1)
    def n_params(self): return sum(p.numel() for p in self.parameters())

# --- Data ---
def batches(hdf5_dir, batch_size=16, max_files=None):
    import h5py
    files = sorted(glob.glob(f"{hdf5_dir}/*.hdf5"))
    if max_files: files = files[:max_files]
    b1,b2,bl = [],[],[]
    for f in files:
        try:
            with h5py.File(f) as hf:
                imgs = hf["images"]["rgb_left"]
                for i in range(len(imgs)-1):
                    try:
                        img = Image.open(io.BytesIO(bytes(imgs[i]))).convert("RGB")
                        v1 = TRANSFORM(img); v2 = TRANSFORM_AUG2(img)
                        lin = float(hf["commands"]["linear_velocity"][i])
                        ang = float(hf["commands"]["angular_velocity"][i])
                        gps = list(hf["gps"]["latlong"][i])
                        b1.append(v1); b2.append(v2); bl.append([lin,ang,gps[0],gps[1]])
                        if len(b1)==batch_size:
                            yield torch.stack(b1), torch.stack(b2), torch.tensor(bl, dtype=torch.float32)
                            b1,b2,bl=[],[],[]
                    except: pass
        except: pass

# --- Losses ---
def nt_xent(z1, z2, t=0.07):
    B = z1.shape[0]
    z = torch.cat([z1,z2],0)
    sim = torch.mm(z, z.t()) / t
    sim.masked_fill_(torch.eye(2*B, dtype=torch.bool), -9e15)
    labels = torch.cat([torch.arange(B,2*B), torch.arange(B)])
    return F.cross_entropy(sim, labels)

class NeuroLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.lw = nn.Parameter(torch.zeros(6))
    def forward(self, z1, z2, labels):
        w = F.softmax(self.lw, dim=0) * 6
        L_da  = (1.0 - (z1*z2).sum(-1)).mean()
        L_sht = F.relu(0.5 - z1.var(0).mean())
        gps = F.normalize(labels[:,2:].float(), dim=1)
        zn  = F.normalize(z1.float(), dim=1)
        L_ne  = F.mse_loss(torch.mm(zn,zn.t()), torch.mm(gps,gps.t()).clamp(-1,1))
        lv = F.normalize(labels[:,0:1].float(), dim=0)
        L_ach = -(lv.abs() * (z1-z2).norm(dim=-1,keepdim=True)).mean()
        av = F.normalize(labels[:,1:2].float().abs(), dim=0)
        L_ecb = -(av * (z1-z2).norm(dim=-1,keepdim=True)).mean()
        L_cort= -z1.var(0).mean()
        loss = w[0]*L_da + w[1]*L_sht + w[2]*L_ne + w[3]*L_ach + w[4]*L_ecb + w[5]*L_cort
        return loss, {"da":round(L_da.item(),4),"ne":round(L_ne.item(),4)}

class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_a = nn.Parameter(torch.tensor(0.0))  # sigmoid -> alpha
        self.neuro = NeuroLoss()
    def forward(self, z1, z2, labels):
        alpha = torch.sigmoid(self.log_a)
        Lc = nt_xent(z1, z2)
        Ln, sig = self.neuro(z1, z2, labels)
        loss = alpha * Lc + (1-alpha) * Ln
        sig["alpha"] = round(float(alpha.item()), 3)
        sig["L_c"] = round(float(Lc.item()), 4)
        sig["L_n"] = round(float(Ln.item()), 4)
        return loss, sig

# --- Train ---
def train(hdf5_dir, epochs, max_files, log_every, ckpt_path):
    enc = ViTTiny()
    print(f"  Params: {enc.n_params():,}")
    lf = HybridLoss()
    params = list(enc.parameters()) + list(lf.parameters())
    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-4)
    best, step = float("inf"), 0
    t0 = time.perf_counter()
    for ep in range(epochs):
        losses = []
        for v1,v2,lab in batches(hdf5_dir, max_files=max_files):
            z1 = enc(v1); z2 = enc(v2)
            loss, sig = lf(z1, z2, lab)
            if not torch.isfinite(loss): continue
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); step += 1; losses.append(loss.item())
            if step % log_every == 0:
                sstr = " ".join(f"{k}={v}" for k,v in sig.items())
                print(f"  [ep{ep:02d} s{step:05d}] loss={loss.item():.4f} {sstr} ({time.perf_counter()-t0:.0f}s)")
        mean = np.mean(losses) if losses else float("inf")
        print(f"  Epoch {ep:02d} mean={mean:.4f}")
        if mean < best:
            best = mean
            torch.save({"epoch":ep,"loss":best,"model":enc.state_dict()}, ckpt_path)
            print(f"  -> saved ({best:.4f})")
    return enc

# --- AIM probe ---
def aim_probe(enc, hdf5_dir, max_files=30, n=1000):
    import h5py
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    enc.eval()
    Z, Y = [], []
    all_files = sorted(glob.glob(f"{hdf5_dir}/*.hdf5"))
    split = int(0.8 * len(all_files))
    probe_files = all_files[split:split+max_files]
    print(f"  AIM probe: {len(probe_files)} held-out files")
    for path in probe_files:
        if len(Z) >= n: break
        try:
            with h5py.File(path) as hf:
                imgs = hf["images"]["rgb_left"]
                for i in range(min(10, len(imgs))):
                    if len(Z) >= n: break
                    img = Image.open(io.BytesIO(bytes(imgs[i]))).convert("RGB")
                    x = TRANSFORM_CLEAN(img).unsqueeze(0)
                    with torch.no_grad(): z = enc(x).squeeze(0).numpy()
                    lin = float(hf["commands"]["linear_velocity"][i])
                    ang = float(hf["commands"]["angular_velocity"][i])
                    gps = list(hf["gps"]["latlong"][i])
                    Z.append(z); Y.append([lin, ang, gps[0], gps[1]])
        except: pass
    if len(Z) < 20: return {"n": len(Z), "error": "too few"}
    Z = np.array(Z); Y = np.array(Y)
    Zs = StandardScaler().fit_transform(Z)
    split = int(0.8*len(Z))
    labels = ["linear_vel","angular_vel","gps_lat","gps_lon"]
    out = {"n": len(Z)}
    for i,lb in enumerate(labels):
        y = Y[:,i]
        r = Ridge(alpha=1.0)
        r.fit(Zs[:split], y[:split])
        out[lb] = round(float(r.score(Zs[split:], y[split:])), 4)
    return out

# --- Main ---
import argparse
p = argparse.ArgumentParser()
p.add_argument("--hdf5-dir",   default="recon_data/recon_release")
p.add_argument("--epochs",     type=int, default=20)
p.add_argument("--max-files",  type=int, default=50)
p.add_argument("--log-every",  type=int, default=100)
p.add_argument("--skip-probe", action="store_true")
a = p.parse_args()

ckpt_c = "checkpoints/cwm/vit_tiny_hybrid.pt"
ckpt_a = "checkpoints/cwm/vit_tiny_contrastive.pt"
ckpt_b = "checkpoints/cwm/vit_tiny_neuromodulated.pt"
Path("checkpoints/cwm").mkdir(parents=True, exist_ok=True)

print("\nPhase 2a - Encoder C: Hybrid (Contrastive + NeMo-WM)")
print("="*60)
print("Loads existing A+B checkpoints, trains C from scratch")
print("Then runs AIM probe on all three\n")

print("Training Encoder C - Hybrid...")
enc_c = train(a.hdf5_dir, a.epochs, a.max_files, a.log_every, ckpt_c)

if not a.skip_probe:
    try:
        from sklearn.linear_model import Ridge
    except ImportError:
        print("sklearn not found - skipping probe")
        a.skip_probe = True

if not a.skip_probe:
    print("\n" + "="*60)
    print("AIM PROBE - All Three Encoders (held-out frames)")
    print("="*60)

    encs = {}
    for name, ckpt in [("A_contrastive", ckpt_a),
                        ("B_neuromodulated", ckpt_b),
                        ("C_hybrid", ckpt_c)]:
        e = ViTTiny()
        if Path(ckpt).exists():
            e.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
            e.eval()
            encs[name] = e
            print(f"  Loaded {name}")
        else:
            print(f"  Missing {name} - skipping")

    results = {}
    for name, e in encs.items():
        print(f"\n  Probing {name}...")
        results[name] = aim_probe(e, a.hdf5_dir)

    print(f"\n{'='*65}")
    print(f"  {'Signal':<18} {'A Contrast':>11} {'B Neuro':>8} {'C Hybrid':>9} {'Winner'}")
    print(f"  {'-'*60}")
    labels = ["linear_vel","angular_vel","gps_lat","gps_lon"]
    wins = {"A_contrastive":0, "B_neuromodulated":0, "C_hybrid":0}
    for lb in labels:
        vals = {k: results[k].get(lb, float("nan")) for k in results}
        best_k = max(vals, key=lambda k: vals[k])
        wins[best_k] += 1
        w = {"A_contrastive":"A", "B_neuromodulated":"B", "C_hybrid":"C Hybrid"}[best_k]
        ra = vals.get("A_contrastive", float("nan"))
        rb = vals.get("B_neuromodulated", float("nan"))
        rc = vals.get("C_hybrid", float("nan"))
        print(f"  {lb:<18} {ra:>11.4f} {rb:>8.4f} {rc:>9.4f}  {w}")

    print(f"\n  Wins: A={wins['A_contrastive']} B={wins['B_neuromodulated']} C={wins['C_hybrid']}")
    best = max(wins, key=wins.get)
    if wins["C_hybrid"] >= 3:
        print("\n  HYBRID WINS - contrastive base + neuromodulated grounding")
        print("  Mirrors NeMo-WM + JEPA result at encoder level")
    elif wins["B_neuromodulated"] >= 3:
        print("\n  NEUROMODULATED WINS - biological signals sufficient alone")
    elif wins["C_hybrid"] + wins["B_neuromodulated"] >= 3:
        print("\n  NEUROMODULATION HELPS - hybrid or pure both beat contrastive")
    else:
        print("\n  CONTRASTIVE WINS - needs more epochs")

    import json
    with open("vlm_phase2a_results.json","w") as f:
        json.dump({"epochs":a.epochs,"results":results,"wins":wins}, f, indent=2)
    print("\n  Saved: vlm_phase2a_results.json")
