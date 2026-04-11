"""
train_rope_vs_mlp.py — A/B: RoPETemporalHead v2 vs MLP Baseline
"""

import sys, math, time, argparse
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from rope_temporal_head import RoPETemporalHead, infonce_loss, xpred_loss, vicreg_loss


class MLPTemporalHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128,256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256,64),
        )
    def forward(self, z, *a, **k): return self.net(z)
    @property
    def n_params(self): return sum(p.numel() for p in self.parameters())


class SyntheticRECON(Dataset):
    """
    Geometric + visual structure:
    - Close frames (k<5, disp<1.5m):  high latent cosine similarity
    - Far frames  (k>10, disp>3m):   low latent cosine similarity
    Both position and visual signal are available — RoPE uses both.
    """
    def __init__(self, n_traj=200, T=70, n_pairs=8000, seed=42):
        rng = np.random.default_rng(seed)
        pairs = []
        trajs = []

        for _ in range(n_traj):
            pos = np.zeros((T, 2))
            vel = rng.standard_normal(2) * 0.4
            for t in range(1, T):
                vel = vel * 0.88 + rng.standard_normal(2) * 0.12
                pos[t] = pos[t-1] + vel

            # Latents: slowly drifting local scene + noise
            local = rng.standard_normal(128).astype(np.float32)
            lats  = np.zeros((T, 128), dtype=np.float32)
            for t in range(T):
                local = local * 0.94 + rng.standard_normal(128).astype(np.float32) * 0.06
                z = local + rng.standard_normal(128).astype(np.float32) * 0.4
                lats[t] = z / np.linalg.norm(z)
            trajs.append({"pos": pos, "lats": lats})

        for _ in range(n_pairs):
            tr = trajs[rng.integers(n_traj)]
            if rng.random() > 0.5:
                t  = rng.integers(0, T-6); k = rng.integers(1, 5)
            else:
                t  = rng.integers(0, T-12); k = rng.integers(10, T-t)
            tk = min(t+k, T-1)
            pt = tr["pos"][t]; ptk = tr["pos"][tk]
            disp = float(np.linalg.norm(pt-ptk))
            pairs.append({
                "z_t":  tr["lats"][t].copy(), "z_tk": tr["lats"][tk].copy(),
                "c_t":  np.array([t/70., pt[0]/10.,  pt[1]/10.],  dtype=np.float32),
                "c_tk": np.array([tk/70., ptk[0]/10., ptk[1]/10.], dtype=np.float32),
                "label": float(disp < 1.5), "disp": disp,
            })

        self.pairs = pairs
        labels = np.array([p["label"] for p in pairs])
        cd = np.array([p["disp"] for p in pairs if p["label"]==1]).mean()
        fd = np.array([p["disp"] for p in pairs if p["label"]==0]).mean()
        print(f"  {len(pairs)} pairs | close={labels.sum():.0f} far={(1-labels).sum():.0f} "
              f"| close_disp={cd:.2f}m far_disp={fd:.2f}m")

    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        p = self.pairs[i]
        return (torch.from_numpy(p["z_t"]),  torch.from_numpy(p["z_tk"]),
                torch.from_numpy(p["c_t"]),  torch.from_numpy(p["c_tk"]),
                torch.tensor(p["label"], dtype=torch.float32))


def auroc_np(scores, labels):
    neg = -np.asarray(scores); labels = np.asarray(labels)
    n_pos = labels.sum(); n_neg = len(labels)-n_pos
    if n_pos==0 or n_neg==0: return 0.5
    order = np.argsort(-neg)
    tpr=[0.]; fpr=[0.]; tp=fp=0
    for l in labels[order]:
        if l==1: tp+=1
        else:    fp+=1
        tpr.append(tp/n_pos); fpr.append(fp/n_neg)
    return float(_np_trapz(tpr, fpr))


def evaluate(head, loader, device, is_rope=False):
    """Single-frame eval for both heads — fair comparison."""
    head.eval()
    scores=[]; labels=[]; close_d=[]; far_d=[]
    with torch.no_grad():
        for z_t, z_tk, *_, lbl in loader:
            z_t=z_t.to(device); z_tk=z_tk.to(device)
            # Both heads: single-frame embeddings
            e_t  = head(z_t)
            e_tk = head(z_tk)
            d = 1.0 - F.cosine_similarity(
                F.normalize(e_t,dim=-1), F.normalize(e_tk,dim=-1), dim=-1)
            d_np=d.cpu().numpy(); l_np=lbl.numpy()
            scores.extend(d_np); labels.extend(l_np)
            close_d.extend(d_np[l_np==1]); far_d.extend(d_np[l_np==0])
    s=np.array(scores); l=np.array(labels)
    return {"auroc": auroc_np(s,l),
            "close": np.mean(close_d) if close_d else 0.,
            "far":   np.mean(far_d)   if far_d   else 0.,
            "sep":   np.mean(far_d)-np.mean(close_d) if close_d and far_d else 0.}


def train_mlp_epoch(head, loader, opt, device, T=0.07):
    head.train(); losses=[]
    for z_t, z_tk, *_ , __ in loader:
        z_t=z_t.to(device); z_tk=z_tk.to(device)
        e_t=head(z_t); e_tk=head(z_tk)
        loss = infonce_loss(e_t, e_tk, T) + 0.1*vicreg_loss(e_t)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step(); losses.append(loss.item())
    return float(np.mean(losses))


def train_rope_epoch(head, loader, opt, device, T=0.07, ema_decay=0.996):
    head.train(); losses=[]
    for z_t, z_tk, c_t, c_tk, _ in loader:
        z_t=z_t.to(device); z_tk=z_tk.to(device)
        c_t=c_t.to(device); c_tk=c_tk.to(device)

        # x-prediction: predict EMA(z_tk) from pair context
        pred   = head(z_t, z_tk, c_t, c_tk)
        target = head.target_embed(z_tk)          # EMA, no grad
        loss, _ = xpred_loss(pred, target, T)

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        head.update_ema(ema_decay)                # keep EMA fresh
        losses.append(loss.item())
    return float(np.mean(losses))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",  type=int,   default=30)
    p.add_argument("--batch",   type=int,   default=128)
    p.add_argument("--lr",      type=float, default=1e-3)
    p.add_argument("--n-pairs", type=int,   default=8000)
    p.add_argument("--T",       type=float, default=0.07)
    p.add_argument("--seed",    type=int,   default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}\n\nBuilding datasets...")

    train_ds = SyntheticRECON(n_pairs=args.n_pairs, seed=args.seed)
    val_ds   = SyntheticRECON(n_pairs=1000, seed=args.seed+7)
    tl = DataLoader(train_ds, args.batch, shuffle=True,  num_workers=0)
    vl = DataLoader(val_ds,   args.batch, shuffle=False, num_workers=0)

    mlp  = MLPTemporalHead().to(device)
    rope = RoPETemporalHead().to(device)
    print(f"\nMLP  {mlp.n_params:,} params | RoPE {rope.n_params:,} params\n")

    mo = torch.optim.AdamW(mlp.parameters(),  lr=args.lr, weight_decay=1e-4)
    ro = torch.optim.AdamW(rope.parameters(), lr=args.lr, weight_decay=1e-4)
    ms = torch.optim.lr_scheduler.CosineAnnealingLR(mo, args.epochs)
    rs = torch.optim.lr_scheduler.CosineAnnealingLR(ro, args.epochs)

    r0m = evaluate(mlp,  vl, device)
    r0r = evaluate(rope, vl, device)
    print(f"Init:  MLP={r0m['auroc']:.4f}  RoPE={r0r['auroc']:.4f}\n")
    print(f"{'Ep':>3}  {'MLoss':>7} {'MAuc':>6} {'MSep':>6}  "
          f"{'RLoss':>7} {'RAuc':>6} {'RSep':>6}  {'ΔAUC':>6}")
    print("─"*65)

    best = {"mlp": 0., "rope": 0.}

    for ep in range(1, args.epochs+1):
        t0 = time.time()
        ml = train_mlp_epoch(mlp,  tl, mo, device, args.T)
        rl = train_rope_epoch(rope, tl, ro, device, args.T)
        ms.step(); rs.step()

        rm = evaluate(mlp,  vl, device)
        rr = evaluate(rope, vl, device)
        best["mlp"]  = max(best["mlp"],  rm["auroc"])
        best["rope"] = max(best["rope"], rr["auroc"])

        print(f"{ep:3d}  {ml:7.4f} {rm['auroc']:6.4f} {rm['sep']:+6.4f}  "
              f"{rl:7.4f} {rr['auroc']:6.4f} {rr['sep']:+6.4f}  "
              f"{rr['auroc']-rm['auroc']:+6.4f}  ({time.time()-t0:.1f}s)")

    print("─"*65)
    fm = evaluate(mlp,  vl, device)
    fr = evaluate(rope, vl, device)
    print(f"\nFINAL ({args.epochs} epochs)")
    print(f"MLP   AUROC={fm['auroc']:.4f}  close={fm['close']:.4f}  far={fm['far']:.4f}  sep={fm['sep']:+.4f}")
    print(f"RoPE  AUROC={fr['auroc']:.4f}  close={fr['close']:.4f}  far={fr['far']:.4f}  sep={fr['sep']:+.4f}")
    print(f"Δ AUROC = {fr['auroc']-fm['auroc']:+.4f}  "
          f"({'✅ RoPE wins' if fr['auroc'] > fm['auroc'] else 'MLP wins'})")
    print(f"Best MLP={best['mlp']:.4f}  Best RoPE={best['rope']:.4f}  "
          f"Best Δ={best['rope']-best['mlp']:+.4f}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv += ["--epochs","30","--n-pairs","8000"]
    main()
