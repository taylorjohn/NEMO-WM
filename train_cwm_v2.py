"""train_cwm_v2.py - Sprint 1, clean rewrite, real StudentEncoder."""
# -*- coding: utf-8 -*-
import io, glob, time
import numpy as np, h5py, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from train_mvtec import StudentEncoder
from neuromodulator import NeuromodulatorState
try:
    from domain_topology import get_topology, auto_calibrate_cortisol
    TOPOLOGY_AVAILABLE = True
except ImportError:
    TOPOLOGY_AVAILABLE = False
    print("[domain_topology] Not found - using defaults")
from cwm_neuro_reward import NeuromodulatedCWMLoss
from cwm_moe_jepa import jepa_moe_loss
from train_cwm_DEPRECATED import CortexWorldModel, MAX_ACTION_DIM

DEVICE = torch.device("cpu")

def decode(jpeg, sz=224):
    img = Image.open(io.BytesIO(jpeg)).convert("RGB").resize((sz,sz))
    return torch.from_numpy(np.array(img)).float().permute(2,0,1)/255.0

def get_batches(hdf5_dir, max_files=None, batch_size=16):
    files = sorted(glob.glob(f"{hdf5_dir}/*.hdf5"))
    if max_files: files = files[:max_files]
    buf_t, buf_t1, buf_a, buf_g = [], [], [], []
    for f in files:
        try:
            with h5py.File(f,"r") as hf:
                imgs = hf["images"]["rgb_left"]
                N = len(imgs)-1
                for i in range(N):
                    try:
                        ft  = decode(bytes(imgs[i]))
                        ft1 = decode(bytes(imgs[i+1]))
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

def train(hdf5_dir="recon_data/recon_release", epochs=30,
          lr=1e-4, batch_size=16, max_files=None, log_every=500,
          random_encoder=False, no_cortisol=False, save_path=None,
          domain="recon"):

    encoder = StudentEncoder().to(DEVICE)
    if random_encoder:
        print('StudentEncoder: RANDOM WEIGHTS (ablation control - no checkpoint loaded)')
    else:
        _ckpt = torch.load(r'checkpoints\dinov2_student\student_best.pt', map_location='cpu', weights_only=False)
        _sd = _ckpt.get('model', _ckpt.get('state_dict', _ckpt))
        missing, unexpected = encoder.load_state_dict(_sd, strict=False)
        print(f'StudentEncoder loaded: {len(missing)} missing, {len(unexpected)} unexpected')
        del _ckpt, _sd
    encoder.eval()
    model   = CortexWorldModel(d_model=128, K=16).to(DEVICE)

    params = list(model.parameters())
    opt    = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    neuro  = NeuromodulatorState(session_start=time.time())
    if no_cortisol:
        neuro.cortisol_weight = 0.0
        print("  Cortisol: DISABLED (Sprint 8d ablation)")
    else:
        print("  Cortisol: ENABLED (v16.12)")

    # Load domain topology
    topo = None
    if TOPOLOGY_AVAILABLE:
        topo = get_topology(domain)
        print(f"  Domain:   {topo.name} | spatial={topo.spatial_type} | expected_regime={topo.expected_regime}")
    epoch0_losses = []
    loss_fn= NeuromodulatedCWMLoss()

    best, step = float("inf"), 0
    for ep in range(epochs):
        losses = []
        for frame_t, frame_t1, action, gps in get_batches(hdf5_dir, max_files, batch_size):
            B = frame_t.shape[0]
            action_pad = F.pad(action, (0, MAX_ACTION_DIM-2))

            with torch.no_grad():
                z_t   = encoder(frame_t)
                z_t1  = encoder(frame_t1)
                p_t1, _, _, _ = model.encode(z_t1)
                z_target = p_t1.detach()

            p_t, pos_t, _, _ = model.encode(z_t)
            ctx_h  = model.thick_gru.init_context(B, DEVICE)
            domain_ids = torch.zeros(B, dtype=torch.long)
            out    = model.predict(p_t, action_pad, ctx_h, pos_t, domain_ids)

            signals = neuro.update(p_t.mean(1), z_target.mean(1),
                                   rho=0.5, action_magnitude=action.norm(dim=-1).mean().item())
            loss_fn.update_from_neuro(signals)

            L_j,_ = jepa_moe_loss(out["z_pred"], z_target,
                                   out["moe_aux_loss"], signals["da_effective"])
            L_n,_ = loss_fn(out["z_pred"], z_target,
                            signed_dist=out["signed_dist"],
                            particle_positions=pos_t,
                            gps_pred=out["gps_pred"], gps_target=gps)
            L_j = torch.clamp(L_j, min=0.5)  # free bits
            total = L_j + L_n + out["contact_loss"]

            if not torch.isfinite(total): continue
            opt.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); step += 1

            losses.append(total.item())
            if ep == 0: epoch0_losses.append(total.item())
            if step % log_every == 0:
                exp = out["attn_weights"][0] if "attn_weights" in out else {}
                print(f"[ep{ep:02d} s{step:05d}] loss={total.item():.4f} "
                      f"L_jepa={L_j.item():.4f} regime={signals['regime']} "
                      f"DA={signals['da']:.3f}")

        mean = np.mean(losses) if losses else float("inf")
        print(f"Epoch {ep:02d} mean={mean:.4f}")
        # Auto-calibrate cortisol after epoch 0
        if ep == 0 and topo is not None and not no_cortisol and epoch0_losses:
            ep0_mean = float(np.mean(epoch0_losses))
            auto_calibrate_cortisol(neuro, topo, ep0_mean)
        if mean < best:
            best = mean
            _sp = save_path or "checkpoints/cwm/cwm_best.pt"
            torch.save({"epoch":ep,"loss":best,"model":model.state_dict()}, _sp)
            print(f"  -> saved (loss={best:.4f})")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5-dir",   default="recon_data/recon_release")
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch-size", type=int,   default=16)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--max-files",  type=int,   default=None)
    p.add_argument("--log-every",      type=int,   default=500)
    p.add_argument("--random-encoder", action="store_true", help="Use random encoder weights (ablation)")
    p.add_argument("--no-cortisol",    action="store_true", help="Disable cortisol (Sprint 8d ablation)")
    p.add_argument("--save-path",      default=None,        help="Override checkpoint save path")
    p.add_argument("--domain",          default="recon",     help="Domain topology: recon, pusht, cwru, mimii, cardiac, smap_msl, mvtec, scand, pointmaze (see domain_topology.py)")
    a = p.parse_args()
    train(a.hdf5_dir, a.epochs, a.lr, a.batch_size, a.max_files,
          a.log_every, a.random_encoder,
          no_cortisol=a.no_cortisol, save_path=a.save_path,
          domain=a.domain)
