"""
diag_recon.py -- Quick HDF5 diagnostic for probe_cwm_latents
Run this first to see exactly what's failing.
"""
import io, sys
from pathlib import Path
import h5py, numpy as np
from PIL import Image
import torch, torch.nn.functional as F

HDF5_DIR = r"recon_data\recon_release"

files = sorted(Path(HDF5_DIR).glob("jackal_2019-*.hdf5"))
if not files:
    print(f"ERROR: No HDF5 files found in {HDF5_DIR}")
    sys.exit(1)

f = files[0]
print(f"Testing: {f.name}\n")

# ── Step 1: Print HDF5 tree ───────────────────────────────────────────────
def print_tree(hf, prefix=""):
    for key in hf.keys():
        item = hf[key]
        if hasattr(item, 'keys'):
            print(f"{prefix}{key}/")
            print_tree(item, prefix + "  ")
        else:
            print(f"{prefix}{key:30s}  shape={item.shape}  dtype={item.dtype}")

print("=== HDF5 structure ===")
with h5py.File(f, "r") as hf:
    print_tree(hf)

# ── Step 2: Read one frame ────────────────────────────────────────────────
print("\n=== Reading frame 0 ===")
with h5py.File(f, "r") as hf:
    # Try to get the raw bytes
    raw = hf["images"]["rgb_left"][0]
    print(f"raw type    : {type(raw)}")
    print(f"raw dtype   : {getattr(raw, 'dtype', 'N/A')}")
    print(f"raw shape   : {getattr(raw, 'shape', 'N/A')}")
    print(f"has tobytes : {hasattr(raw, 'tobytes')}")

    # Try decode
    if hasattr(raw, 'tobytes'):
        jpeg = raw.tobytes()
    else:
        jpeg = bytes(raw)
    print(f"jpeg len    : {len(jpeg)} bytes")

    try:
        img = Image.open(io.BytesIO(jpeg)).convert("RGB")
        print(f"PIL decode  : OK — {img.size}")
        arr = np.array(img.resize((224,224)))
        frame = torch.from_numpy(arr).float().div(255).permute(2,0,1)
        print(f"frame shape : {frame.shape}  min={frame.min():.3f}  max={frame.max():.3f}")
    except Exception as e:
        print(f"PIL decode FAILED: {e}")
        print(f"First 16 bytes: {jpeg[:16]}")

# ── Step 3: Read velocity + GPS ───────────────────────────────────────────
print("\n=== Reading metadata frame 0 ===")
with h5py.File(f, "r") as hf:
    try:
        lin = float(hf["commands"]["linear_velocity"][0, 0])
        ang = float(hf["commands"]["angular_velocity"][0, 0])
        gps = hf["gps"]["latlong"][0]
        print(f"linear_vel  : {lin:.4f}")
        print(f"angular_vel : {ang:.4f}")
        print(f"gps latlong : {gps}")
    except Exception as e:
        print(f"Metadata read FAILED: {e}")
        # Try alternative key names
        print("\nAvailable keys under 'commands':")
        try:
            for k in hf["commands"].keys():
                print(f"  commands/{k}  shape={hf['commands'][k].shape}")
        except Exception as e2:
            print(f"  No 'commands' group: {e2}")

# ── Step 4: Try StudentEncoder + ParticleEncoder ──────────────────────────
print("\n=== Model forward pass ===")
try:
    from train_mvtec import StudentEncoder
    print("StudentEncoder: imported OK")
except Exception as e:
    print(f"StudentEncoder import FAILED: {e}")
    sys.exit(1)

class ParticleEncoder(torch.nn.Module):
    def __init__(self, input_dim=128, K=16, d_model=128):
        super().__init__()
        self.K = K; self.d_model = d_model
        self.proj = torch.nn.Linear(input_dim, K*d_model)
    def forward(self, z):
        B = z.shape[0]
        return F.normalize(self.proj(z).view(B,self.K,self.d_model), dim=-1)

enc = StudentEncoder()
pe  = ParticleEncoder()

ckpt = Path(r"checkpoints\cwm\cwm_best.pt")
if ckpt.exists():
    data = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd   = data.get("model", data)
    pe_sd = {k.replace("particle_enc.",""):v
             for k,v in sd.items() if k.startswith("particle_enc.")}
    if pe_sd:
        pe.load_state_dict(pe_sd, strict=False)
        print(f"ParticleEncoder: loaded (epoch {data.get('epoch','?')})")
    else:
        print("ParticleEncoder: random (particle_enc.* not found in checkpoint)")
        print(f"Checkpoint keys sample: {list(sd.keys())[:10]}")
else:
    print(f"Checkpoint not found: {ckpt}")

enc.eval(); pe.eval()

# Test forward pass
with h5py.File(f, "r") as hf:
    raw = hf["images"]["rgb_left"][0]
    jpeg = raw.tobytes() if hasattr(raw,'tobytes') else bytes(raw)

img = Image.open(io.BytesIO(jpeg)).convert("RGB").resize((224,224))
x   = torch.from_numpy(np.array(img)).float().div(255).permute(2,0,1).unsqueeze(0)

with torch.no_grad():
    z   = enc(x)
    pts = pe(z)

print(f"enc output  : {z.shape}  norm={z.norm():.4f}")
print(f"particles   : {pts.shape}")
print(f"\nAll checks passed. probe_cwm_latents.py should work now.")
print(f"If it still fails, paste the output of this script.")
