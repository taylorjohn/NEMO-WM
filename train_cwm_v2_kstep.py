"""train_cwm_v2.py - Sprint 1, clean rewrite, real StudentEncoder."""
import io, glob, time
import numpy as np, h5py, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from train_mvtec import StudentEncoder
from neuromodulator import NeuromodulatorState
from cwm_neuro_reward import NeuromodulatedCWMLoss
from cwm_moe_jepa import jepa_moe_loss
from train_cwm_DEPRECATED import CortexWorldModel, MAX_ACTION_DIM

DEVICE = torch.device("cpu")

def decode(jpeg, sz=224):
    img = Image.open(io.BytesIO(jpeg)).convert("RGB").resize((sz,sz))
    return torch.from_numpy(np.array(img)).float().permute(2,0,1)/255.0

def get_batches(hdf5_dir, max_files=None, batch_size=16, k_step=4):
    """
    k_step: predict k_step frames ahead (default 4 = 1 second at 4Hz).
    Using k_step>1 ensures DINOv2 features differ enough for the predictor
    to have genuine dynamics to learn. k_step=1 (adjacent frames) is too
    easy with a frozen encoder — MSE stays well below free_bits floor.
    """
    files = sorted(glob.glob(f"{hdf5_dir}/*.hdf5"))
    if max_files: files = files[:max_files]
    buf_t, buf_t1, buf_a, buf_g = [], [], [], []
    for f in files:
        try:
            with h5py.File(f,"r") as hf:
                imgs = hf["images"]["rgb_left"]
                N = len(imgs) - k_step  # ensure t+k_step exists
                for i in range(N):
                    try:
                        ft  = decode(bytes(imgs[i]))
                        ft1 = decode(bytes(imgs[i + k_step]))  # k_step ahead
                        lin = float(hf["commands"]["linear_velocity"][i])
                        ang = float(hf["commands"]["angular_velocity"][i])
                        gps = list(hf["gps"]["latlong"][i]) if "gps" in hf else [0.,0.]
                        buf_t.append(ft); buf_t1.append(ft1)
                        buf_a.append([lin,ang]); buf_g.append(gps)
                        if len(buf_t)==batch_size:
                            yield (torch.stack(buf_t), torch.stack(buf_t1),
                                   torch.tensor(buf_a,dtype=torch.float32),
                                   torch.tensor(buf_g,dtype=torch.float32))
                            buf_t,buf_t1,buf_a,buf_g=[],[],[],[]
                    except: pass
        except: pass



def agc_clip_(parameters, lam: float = 0.01, eps: float = 1e-6):
    """
    Adaptive Gradient Clipping from DreamerV3.
    Clips each parameter's gradient by the ratio of its weight norm
    to its gradient norm: clip = lam * ||w|| / ||g||
    Scale-invariant unlike global clip_grad_norm_.
    Large parameters get more headroom; small ones aren't overwhelmed.
    """
    for p in parameters:
        if p.grad is None or p.data.numel() == 0:
            continue
        p_norm = p.data.norm(2.0)
        g_norm = p.grad.norm(2.0)
        max_norm = lam * p_norm.clamp(min=eps)
        if g_norm > max_norm:
            p.grad.mul_(max_norm / (g_norm + eps))

# FREE_BITS set per run via free_bits arg: adjacent DINOv2 frames have MSE ~0.05-0.15,
             # far below 0.5 floor. k_step=4 raises MSE to 0.3-0.6, floor at 0.1 is reachable.

def train(hdf5_dir="recon_data/recon_release", epochs=30,
          lr=1e-4, batch_size=16, max_files=None, log_every=500,
          k_step=4, free_bits=0.1):

    print(f"\ntrain_cwm_v2.py: k_step={k_step} free_bits={free_bits}")
    print(f"  Predicting {k_step} frame(s) ahead = {k_step/4:.2f}s at 4Hz")
    print(f"  Free bits floor: {free_bits} (was 0.5 — lowered because DINOv2 adj MSE ~0.05-0.15)\n")
    encoder = StudentEncoder().to(DEVICE)
    _ckpt = torch.load(r'checkpoints\dinov2_student\student_best.pt', map_location='cpu')
    _sd = _ckpt.get('model', _ckpt.get('state_dict', _ckpt))
    missing, unexpected = encoder.load_state_dict(_sd, strict=False)
    print(f'StudentEncoder loaded: {len(missing)} missing, {len(unexpected)} unexpected')
    del _ckpt, _sd
    encoder.eval()
    model   = CortexWorldModel(d_model=128, K=16).to(DEVICE)

    params = list(model.parameters())
    opt    = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    neuro  = NeuromodulatorState(session_start=time.time())
    loss_fn= NeuromodulatedCWMLoss()

    best, step = float("inf"), 0
    for ep in range(epochs):
        losses = []
        for frame_t, frame_t1, action, gps in get_batches(hdf5_dir, max_files, batch_size, k_step):
            B = frame_t.shape[0]
            action_pad = F.pad(action, (0, MAX_ACTION_DIM-2))

            with torch.no_grad():
                z_t   = encoder(frame_t)
                z_t1  = encoder(frame_t1)
                p_t1, _, _, _ = model.encode(z_t1)
                z_target = p_t1.detach()

            p_t, pos_t, _, _ = model.encode(z_t)
            ctx_h  = model.thick_gru.init_context(B, DEVICE)
            domain = torch.zeros(B, dtype=torch.long)
            out    = model.predict(p_t, action_pad, ctx_h, pos_t, domain)

            signals = neuro.update(p_t.mean(1), z_target.mean(1),
                                   rho=0.5, action_magnitude=action.norm(dim=-1).mean().item())
            loss_fn.update_from_neuro(signals)

            L_j,_ = jepa_moe_loss(out["z_pred"], z_target,
                                   out["moe_aux_loss"], signals["da_effective"])
            L_n,_ = loss_fn(out["z_pred"], z_target,
                            signed_dist=out["signed_dist"],
                            particle_positions=pos_t,
                            gps_pred=out["gps_pred"], gps_target=gps)
            L_j = torch.clamp(L_j, min=free_bits)  # free bits
            total = L_j + L_n + out["contact_loss"]

            if not torch.isfinite(total): continue
            opt.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); step += 1

            losses.append(total.item())
            if step % log_every == 0:
                exp = out["attn_weights"][0] if "attn_weights" in out else {}
                print(f"[ep{ep:02d} s{step:05d}] loss={total.item():.4f} "
                      f"L_jepa={L_j.item():.4f} regime={signals['regime']} "
                      f"DA={signals['da']:.3f}")

        mean = np.mean(losses) if losses else float("inf")
        print(f"Epoch {ep:02d} mean={mean:.4f}")
        if mean < best:
            best = mean
            torch.save({"epoch":ep,"loss":best,"model":model.state_dict()},
                       "checkpoints/cwm/cwm_best.pt")
            print(f"  -> saved (loss={best:.4f})")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5-dir",   default="recon_data/recon_release")
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch-size", type=int,   default=16)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--max-files",  type=int,   default=None)
    p.add_argument("--log-every",  type=int,   default=500)
    p.add_argument("--k-step",     type=int,   default=4,    help="Predict k frames ahead (default 4 = 1 second)")
    p.add_argument("--free-bits",  type=float, default=0.1,  help="Free bits floor for L_jepa (default 0.1)")
    a = p.parse_args()
    train(a.hdf5_dir, a.epochs, a.lr, a.batch_size, a.max_files, a.log_every, a.k_step, a.free_bits)
