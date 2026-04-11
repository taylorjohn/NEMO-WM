"""train_recon_temporal_v2.py - Sprint 2 TemporalHead InfoNCE training."""
import io, glob, random, time
import numpy as np, h5py, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from train_mvtec import StudentEncoder
from train_cwm import CortexWorldModel

class TemporalHead(nn.Module):
    def __init__(self, d_model=128, embed_dim=64):
        super().__init__()
        self.pool = nn.Sequential(nn.Linear(d_model,d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.proj = nn.Sequential(nn.Linear(d_model,embed_dim*2), nn.GELU(), nn.Linear(embed_dim*2,embed_dim))
        self.temperature = torch.tensor(0.07)
    def forward(self, particles):
        x = self.pool(particles.mean(dim=1))
        return F.normalize(self.proj(x), dim=-1)

def infonce(z1, z2, temp):
    B = z1.shape[0]
    sim = torch.mm(z1, z2.T) / temp.clamp(min=0.01)
    labels = torch.arange(B, device=z1.device)
    loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
    acc  = (sim.argmax(dim=1)==labels).float().mean().item()
    return loss, acc

def decode(jpeg, sz=224):
    img = Image.open(io.BytesIO(jpeg)).convert("RGB").resize((sz,sz))
    return torch.from_numpy(np.array(img)).float().permute(2,0,1)/255.0

def get_pairs(hdf5_dir, max_files=None, batch_size=32, k_max=8):
    files = sorted(glob.glob(f"{hdf5_dir}/*.hdf5"))
    if max_files: files = files[:max_files]
    buf_t, buf_tk = [], []
    for f in files:
        try:
            with h5py.File(f,"r") as hf:
                imgs = hf["images"]["rgb_left"]
                N = len(imgs)
                for t in range(N - k_max):
                    k = random.randint(1, k_max)
                    try:
                        buf_t.append(decode(bytes(imgs[t])))
                        buf_tk.append(decode(bytes(imgs[t+k])))
                        if len(buf_t) == batch_size:
                            yield torch.stack(buf_t), torch.stack(buf_tk)
                            buf_t, buf_tk = [], []
                    except: pass
        except: pass

def train(cwm_ckpt="checkpoints/cwm/cwm_best.pt",
          hdf5_dir="recon_data/recon_release",
          epochs=20, batch_size=32, lr=1e-3,
          max_files=None, log_every=50, save_dir="checkpoints/cwm"):

    import pathlib; pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    encoder = StudentEncoder().eval().to(device)
    cwm = CortexWorldModel(d_model=128, K=16).to(device)
    if pathlib.Path(cwm_ckpt).exists():
        ckpt = torch.load(cwm_ckpt, map_location=device, weights_only=False)
        cwm.load_state_dict(ckpt["model"])
        print(f"CWM loaded: {cwm_ckpt} (epoch {ckpt['epoch']}, loss {ckpt['loss']:.2f})")
    for p in cwm.parameters(): p.requires_grad_(False)
    cwm.eval()

    head = TemporalHead(d_model=128, embed_dim=64).to(device)
    opt  = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    print(f"TemporalHead: {sum(p.numel() for p in head.parameters()):,} params")

    best, step = float("inf"), 0
    for ep in range(epochs):
        losses, accs = [], []
        for frame_t, frame_tk in get_pairs(hdf5_dir, max_files, batch_size):
            with torch.no_grad():
                z_t  = encoder(frame_t)
                z_tk = encoder(frame_tk)
                p_t,  _, _, _ = cwm.encode(z_t)
                p_tk, _, _, _ = cwm.encode(z_tk)
            e_t  = head(p_t)
            e_tk = head(p_tk)
            loss, acc = infonce(e_t, e_tk, head.temperature)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step(); step += 1
            losses.append(loss.item()); accs.append(acc)
            if step % log_every == 0:
                print(f"[ep{ep:02d} s{step:05d}] loss={loss.item():.4f} top1_acc={acc:.3f} temp={head.temperature.item():.3f}")
        mean_loss = np.mean(losses); mean_acc = np.mean(accs)
        print(f"Epoch {ep:02d} loss={mean_loss:.4f} top1_acc={mean_acc:.3f}")
        if mean_loss < best:
            best = mean_loss
            torch.save({"epoch":ep,"loss":best,"acc":mean_acc,"head":head.state_dict(),"embed_dim":64},
                       f"{save_dir}/temporal_head_best.pt")
            print(f"  -> saved")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cwm-ckpt",   default=r"checkpoints\cwm\cwm_best.pt")
    p.add_argument("--hdf5-dir",   default="recon_data/recon_release")
    p.add_argument("--epochs",     type=int,   default=20)
    p.add_argument("--batch-size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--max-files",  type=int,   default=None)
    p.add_argument("--log-every",  type=int,   default=50)
    a = p.parse_args()
    train(a.cwm_ckpt, a.hdf5_dir, a.epochs, a.batch_size, a.lr, a.max_files, a.log_every)
