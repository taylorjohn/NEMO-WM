# final_aim_probe.py - AIM probe on all four ViT-Tiny checkpoints
# Handles both key naming conventions (abbreviated and full)
# cp1252 safe, weights_only=False
import glob, io, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms

CLEAN = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Full-name architecture (matches A, B_orig, C_hybrid checkpoints)
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=192):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        n = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n+1, embed_dim))
    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1,2)
        cls = self.cls_token.expand(x.shape[0],-1,-1)
        return torch.cat([cls,x],dim=1) + self.pos_embed

class Attention(nn.Module):
    def __init__(self, dim, heads=3):
        super().__init__()
        self.heads=heads; self.scale=(dim//heads)**-0.5
        self.qkv=nn.Linear(dim,dim*3); self.proj=nn.Linear(dim,dim)
    def forward(self, x):
        B,N,C=x.shape
        qkv=self.qkv(x).reshape(B,N,3,self.heads,C//self.heads).permute(2,0,3,1,4)
        q,k,v=qkv.unbind(0)
        # (B, heads, N, head_dim) -> (B, N, heads, head_dim) -> (B, N, C)
        out=(q@k.transpose(-2,-1)*self.scale).softmax(-1)@v
        out=out.transpose(1,2).contiguous().reshape(B,N,C)
        return self.proj(out)

class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1=nn.LayerNorm(dim); self.attn=Attention(dim)
        self.norm2=nn.LayerNorm(dim)
        self.mlp=nn.Sequential(nn.Linear(dim,dim*4),nn.GELU(),nn.Linear(dim*4,dim))
    def forward(self, x):
        x=x+self.attn(self.norm1(x)); return x+self.mlp(self.norm2(x))

class ViTTiny(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.patch_embed=PatchEmbed()
        self.blocks=nn.Sequential(*[Block(192) for _ in range(12)])
        self.norm=nn.LayerNorm(192); self.head=nn.Linear(192,out_dim)
    def forward(self, x):
        x=self.patch_embed(x); x=self.blocks(x); x=self.norm(x)
        return F.normalize(self.head(x[:,0]),dim=-1)

def remap_keys(sd):
    """Remap abbreviated key names to full names if needed."""
    if "pe.cls" not in sd:
        return sd  # already full names
    remap = {}
    for k,v in sd.items():
        nk = k
        nk = nk.replace("pe.cls","patch_embed.cls_token")
        nk = nk.replace("pe.pos","patch_embed.pos_embed")
        nk = nk.replace("pe.proj","patch_embed.proj")
        nk = nk.replace(".n1.",".norm1.").replace(".n2.",".norm2.")
        nk = nk.replace(".a.qkv",".attn.qkv").replace(".a.proj",".attn.proj")
        nk = nk.replace(".m.0",".mlp.0").replace(".m.2",".mlp.2")
        remap[nk] = v
    return remap

def load_enc(ckpt):
    sd_raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    model_sd = sd_raw.get("model", sd_raw)
    sd = remap_keys(model_sd)
    e = ViTTiny()
    missing, unexpected = e.load_state_dict(sd, strict=False)
    if missing:
        print(f"    Warning: {len(missing)} missing keys")
    e.eval()
    info = {"epoch": sd_raw.get("epoch","?"),
            "loss":  sd_raw.get("loss", 0)}
    return e, info

def aim_probe(enc, hdf5_dir, max_files=40, n=800):
    import h5py
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    enc.eval()
    files = sorted(glob.glob(f"{hdf5_dir}/*.hdf5"))[:max_files]
    print(f"    Found {len(files)} files in {hdf5_dir}")
    Z,Y = [],[]
    for path in files:
        if len(Z)>=n: break
        try:
            with h5py.File(path) as hf:
                imgs = hf["images"]["rgb_left"]
                for i in range(min(10,len(imgs))):
                    if len(Z)>=n: break
                    img=Image.open(io.BytesIO(bytes(imgs[i]))).convert("RGB")
                    x=CLEAN(img).unsqueeze(0)
                    with torch.no_grad(): z=enc(x).squeeze(0).numpy()
                    lin=float(hf["commands"]["linear_velocity"][i])
                    ang=float(hf["commands"]["angular_velocity"][i])
                    gps=list(hf["gps"]["latlong"][i])
                    Z.append(z); Y.append([lin,ang,gps[0],gps[1]])
        except Exception as ex:
            print(f"    HDF5 ERR: {ex}"); pass
    if len(Z)<20: return {"n":len(Z),"error":"too few"}
    Z=np.array(Z); Y=np.array(Y)
    Zs=StandardScaler().fit_transform(Z)
    sp=int(0.8*len(Z))
    labels=["linear_vel","angular_vel","gps_lat","gps_lon"]
    out={"n":len(Z)}
    for i,lb in enumerate(labels):
        y=Y[:,i]; r=Ridge(alpha=1.0)
        r.fit(Zs[:sp],y[:sp])
        out[lb]=round(float(r.score(Zs[sp:],y[sp:])),4)
    return out

import argparse
p=argparse.ArgumentParser()
p.add_argument("--hdf5-dir",default="recon_data/recon_release")
p.add_argument("--n",type=int,default=800)
a=p.parse_args()

ckpts = {
    "A_contrastive":       "checkpoints/cwm/vit_tiny_contrastive.pt",
    "B_neuro_original":    "checkpoints/cwm/vit_tiny_neuromodulated.pt",
    "B_neuro_fixed":       "checkpoints/cwm/vit_tiny_neuromodulated_fixed.pt",
    "C_hybrid":            "checkpoints/cwm/vit_tiny_hybrid.pt",
    "D_temporal_split":    "checkpoints/cwm/vit_tiny_temporal_split.pt",
}

print("\nFinal AIM Probe -- All Four ViT-Tiny Encoders")
print("="*65)
print(f"In-distribution (first 40 files), n={a.n}\n")

results = {}
for name, ckpt in ckpts.items():
    if not Path(ckpt).exists():
        print(f"  MISSING: {ckpt}"); continue
    enc, info = load_enc(ckpt)
    loss_str = f"{info['loss']:.4f}" if isinstance(info['loss'],float) else str(info['loss'])
    print(f"  Probing {name} (ep={info['epoch']} loss={loss_str})...")
    results[name] = aim_probe(enc, a.hdf5_dir, n=a.n)
    print(f"    n={results[name].get('n')} samples")

labels = ["linear_vel","angular_vel","gps_lat","gps_lon"]
wins = {k:0 for k in results}

print(f"\n{'='*75}")
print(f"  {'Signal':<18}",end="")
for k in results: print(f"  {k[:14]:>14}",end="")
print()
print(f"  {'-'*70}")

for lb in labels:
    vals = {k: results[k].get(lb,float("nan")) for k in results}
    best = max(vals, key=lambda k: vals[k] if not np.isnan(vals[k]) else -999)
    wins[best] = wins.get(best,0)+1
    print(f"  {lb:<18}",end="")
    for k in results:
        v=vals[k]
        marker = " *" if k==best else "  "
        print(f"  {v:>12.4f}{marker}",end="")
    print()

print(f"\n  Wins: "+" ".join(f"{k}={v}" for k,v in wins.items()))
best_enc=max(wins,key=wins.get)
labels_map={"A_contrastive":"Contrastive","B_neuro_original":"Neuro(broken)",
            "B_neuro_fixed":"Neuro(fixed)","C_hybrid":"Hybrid"}
print(f"  Overall winner: {labels_map.get(best_enc,best_enc)}")

if wins.get("B_neuro_fixed",0) >= 2:
    print("\n  FIXED NEURO wins or ties -- positive signals encode physics")
    print("  Compare fixed vs original B to confirm sign correction matters")
elif wins.get("C_hybrid",0) >= 2:
    print("\n  HYBRID wins -- contrastive + neuromodulated complement each other")
elif wins.get("A_contrastive",0) >= 3:
    print("\n  CONTRASTIVE wins -- need more data or epochs for neuro signals")

# Key comparison: fixed vs original B
print(f"\n  Key comparison -- Fixed vs Original NeuroLoss:")
for lb in labels:
    vo = results.get("B_neuro_original",{}).get(lb,float("nan"))
    vf = results.get("B_neuro_fixed",  {}).get(lb,float("nan"))
    delta = vf-vo if not (np.isnan(vo) or np.isnan(vf)) else float("nan")
    direction = "improved" if delta>0 else "worse" if delta<0 else "same"
    print(f"    {lb:<18} original={vo:.4f}  fixed={vf:.4f}  "
          f"delta={delta:+.4f} ({direction})")

import json
with open("final_aim_probe_results.json","w") as f:
    json.dump({"results":results,"wins":wins},f,indent=2)
print("\n  Saved: final_aim_probe_results.json")
