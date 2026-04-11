"""
train_proprio.py  --  NeMo-WM Sprint 6b
========================================
Improvements over Sprint 6:
  1. Larger model: d_hidden=128, d_model=64 (vs d_hidden=64)
  2. Relative heading: delta heading t→t+1 added as 8th signal (d_in=8)
  3. Lower temperature: 0.05 (vs 0.07) — sharper contrastive objective
  4. Longer training: 40 epochs (vs 20)

Sprint 6 result:  Full=0.892, No-VLM=0.850, VLM-only=0.862
Sprint 6b target: No-VLM >= 0.862 (full dissociation — proprio matches VLM-only)

Usage:
    python train_proprio.py ^
        --hdf5-dir recon_data/recon_release ^
        --out-ckpt checkpoints/cwm/proprio_6b_best.pt ^
        --epochs 40 --d-hidden 128 --temperature 0.05
"""

import argparse, random, math, time
from pathlib import Path
import h5py, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR


# ===========================================================================
# Dataset — adds relative heading as 8th signal
# ===========================================================================

class ProprioPairDataset:
    """
    Sprint 6b: d_in=8 — adds relative heading delta (heading_t+1 - heading_t).
    This gives the model instantaneous turning rate, which varies more within
    a trajectory than absolute heading, helping discriminate temporal position.

    Signals per frame (d_in=8):
        [0] linear_velocity
        [1] angular_velocity
        [2] gps_zero (always 0 — no GPS)
        [3] gps_zero (always 0)
        [4] contact
        [5] sin(heading)   -- accumulated heading from t=0
        [6] cos(heading)
        [7] delta_heading  -- heading_t - heading_{t-1} (relative, instantaneous)
    """

    def __init__(self, hdf5_dir, k_max=4, k_hard_min=32,
                 n_neg=7, max_files=None, seed=42):
        random.seed(seed); np.random.seed(seed)
        files = sorted(Path(hdf5_dir).glob("jackal_2019-*.hdf5"))
        if max_files:
            files = files[:max_files]
        self.files      = [str(f) for f in files]
        self.k_max      = k_max
        self.k_hard_min = k_hard_min
        self.n_neg      = n_neg
        assert len(self.files) >= 1
        print(f"ProprioPairDataset: {len(self.files)} files, "
              f"k_max={k_max}, k_hard_min={k_hard_min}, n_neg={n_neg}, d_in=8")

    def _extract(self, hf, t):
        """Extract 8-D signal vector for frame t. No GPS."""
        vel_val = float(hf["commands"]["linear_velocity"][t])
        ang_val = float(hf["commands"]["angular_velocity"][t])

        # Accumulated heading from t=0
        ang_all       = hf["commands"]["angular_velocity"][:t+1]
        heading_angle = float(np.sum(ang_all)) / 4.0
        sin_h         = math.sin(heading_angle)
        cos_h         = math.cos(heading_angle)

        # Relative heading: delta from previous frame (instantaneous turn rate)
        if t > 0:
            prev_ang  = hf["commands"]["angular_velocity"][:t]
            prev_head = float(np.sum(prev_ang)) / 4.0
            delta_h   = heading_angle - prev_head
        else:
            delta_h   = 0.0

        contact = 1.0 if abs(vel_val) > 0.3 else 0.0

        return torch.tensor(
            [vel_val, ang_val, 0.0, 0.0, contact, sin_h, cos_h, delta_h],
            dtype=torch.float32
        )

    def _safe(self, t):
        return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

    def sample_batch(self, batch_size=64):
        anchors, positives, negatives_list = [], [], []
        attempts = 0
        while len(anchors) < batch_size and attempts < batch_size * 20:
            attempts += 1
            filepath = random.choice(self.files)
            try:
                with h5py.File(filepath, "r") as hf:
                    L = hf["images"]["rgb_left"].shape[0]
                    if L < self.k_hard_min + self.k_max + 2:
                        continue
                    t_anchor = random.randint(1, L - self.k_max - 2)  # t>=1 for delta_h
                    k        = random.randint(1, self.k_max)
                    t_pos    = t_anchor + k

                    neg_candidates = [
                        t for t in range(1, L)
                        if abs(t - t_anchor) >= self.k_hard_min
                    ]
                    if len(neg_candidates) < self.n_neg:
                        continue
                    neg_ts = random.sample(neg_candidates, self.n_neg)

                    z_anchor = self._safe(self._extract(hf, t_anchor))
                    z_pos    = self._safe(self._extract(hf, t_pos))
                    z_negs   = torch.stack([
                        self._safe(self._extract(hf, t_n)) for t_n in neg_ts
                    ])
                    anchors.append(z_anchor)
                    positives.append(z_pos)
                    negatives_list.append(z_negs)
            except Exception:
                continue

        if not anchors:
            return None
        return {
            "anchor":    torch.stack(anchors),
            "positive":  torch.stack(positives),
            "negatives": torch.stack(negatives_list),
        }


# ===========================================================================
# Encoder
# ===========================================================================

class ProprioEncoderDirect(nn.Module):
    """
    Sprint 6b: d_in=8, d_hidden=128, d_model=64.
    Larger hidden layer + relative heading signal.
    """
    def __init__(self, d_in=8, d_hidden=128, d_model=64, n_layers=3):
        super().__init__()
        self.d_in     = d_in
        self.d_hidden = d_hidden
        self.d_model  = d_model

        layers = []
        din = d_in
        for i in range(n_layers):
            dout = d_hidden if i < n_layers - 1 else d_model
            layers.append(nn.Linear(din, dout))
            if i < n_layers - 1:
                layers.append(nn.LayerNorm(dout))
                layers.append(nn.GELU())
            din = dout
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)

    def save(self, path, epoch=None, loss=None, acc=None):
        torch.save({
            "model":    self.state_dict(),
            "d_in":     self.d_in,
            "d_hidden": self.d_hidden,
            "d_model":  self.d_model,
            "epoch":    epoch,
            "loss":     loss,
            "top1_acc": acc,
        }, path)
        print(f"  → Saved (best top1_acc={acc:.4f}): {path}")

    @classmethod
    def load(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        enc  = cls(
            d_in     = ckpt.get("d_in",     8),
            d_hidden = ckpt.get("d_hidden", 128),
            d_model  = ckpt.get("d_model",  64),
        )
        enc.load_state_dict(ckpt["model"])
        print(f"ProprioEncoder 6b loaded: {path} "
              f"(epoch={ckpt.get('epoch','?')}, top1_acc={ckpt.get('top1_acc','?')})")
        return enc


# ===========================================================================
# InfoNCE loss
# ===========================================================================

def infonce_loss(z_anchor, z_positive, z_negatives, temperature=0.05):
    B, D   = z_anchor.shape
    n_neg  = z_negatives.shape[1]

    sim_pos = F.cosine_similarity(z_anchor, z_positive, dim=-1) / temperature

    z_anc_rep  = z_anchor.unsqueeze(1).expand(-1, n_neg, -1).reshape(B * n_neg, D)
    z_neg_flat = z_negatives.reshape(B * n_neg, D)
    sim_neg    = F.cosine_similarity(z_anc_rep, z_neg_flat, dim=-1).view(B, n_neg) / temperature

    logits = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1)
    labels = torch.zeros(B, dtype=torch.long, device=z_anchor.device)
    loss   = F.cross_entropy(logits, labels)
    top1   = (logits.argmax(dim=1) == 0).float().mean().item()
    return loss, top1


# ===========================================================================
# Training loop
# ===========================================================================

def train(
    hdf5_dir, out_ckpt, epochs=40, steps_epoch=500,
    batch_size=64, lr=3e-4, temperature=0.05,
    n_neg=7, k_max=4, k_hard_min=32,
    d_hidden=128, d_in=8, d_model=64,
    device_str="cpu",
):
    dev = torch.device(device_str)
    Path(out_ckpt).parent.mkdir(parents=True, exist_ok=True)

    enc = ProprioEncoderDirect(d_in=d_in, d_hidden=d_hidden, d_model=d_model).to(dev)
    n_params = sum(p.numel() for p in enc.parameters())

    print(f"\nSprint 6b — ProprioceptiveEncoder Training")
    print(f"  d_in={d_in} (+rel_heading), d_hidden={d_hidden}, d_model={d_model}")
    print(f"  Parameters: {n_params:,}  (Sprint 6: 9,088)")
    print(f"  Epochs: {epochs}  |  Steps/epoch: {steps_epoch}")
    print(f"  LR: {lr}  |  Temp: {temperature} (Sprint 6: 0.07)")
    print(f"  k_max: {k_max}  |  k_hard_min: {k_hard_min}")
    print(f"  Output: {out_ckpt}\n")

    ds    = ProprioPairDataset(hdf5_dir, k_max=k_max, k_hard_min=k_hard_min, n_neg=n_neg)
    opt   = torch.optim.Adam(enc.parameters(), lr=lr)
    sched = CosineAnnealingLR(opt, T_max=epochs * steps_epoch, eta_min=lr * 0.01)

    best_acc  = 0.0
    best_loss = float("inf")

    print(f"{'Epoch':>6}  {'Loss':>8}  {'top1_acc':>10}  {'LR':>12}  {'Elapsed':>10}")
    print(f"{'─'*55}")

    for epoch in range(epochs):
        enc.train()
        losses, accs = [], []
        t0 = time.perf_counter()

        for _ in range(steps_epoch):
            batch = ds.sample_batch(batch_size)
            if batch is None:
                continue
            z_a = enc(batch["anchor"].to(dev))
            z_p = enc(batch["positive"].to(dev))
            neg = batch["negatives"].to(dev)
            z_n = torch.stack([enc(neg[:, i]) for i in range(n_neg)], dim=1)

            loss, acc = infonce_loss(z_a, z_p, z_n, temperature)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), max_norm=1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
            accs.append(acc)

        mean_loss = float(np.mean(losses)) if losses else float("inf")
        mean_acc  = float(np.mean(accs))   if accs   else 0.0
        elapsed   = time.perf_counter() - t0
        cur_lr    = sched.get_last_lr()[0]

        print(f"  {epoch:>4}  {mean_loss:>8.4f}  {mean_acc:>10.4f}  "
              f"{cur_lr:>12.2e}  {elapsed:>9.1f}s")

        if mean_acc > best_acc or (mean_acc == best_acc and mean_loss < best_loss):
            best_acc  = mean_acc
            best_loss = mean_loss
            enc.save(out_ckpt, epoch=epoch, loss=mean_loss, acc=mean_acc)

    print(f"\nTraining complete. Best top1_acc={best_acc:.4f}")
    print(f"Checkpoint: {out_ckpt}")
    print(f"\nRun dissociation eval:")
    print(f"  python eval_recon_auroc.py \\")
    print(f"      --head-ckpt checkpoints/cwm/temporal_head_sprint3.pt \\")
    print(f"      --cwm-ckpt  checkpoints/cwm/cwm_multidomain_best.pt \\")
    print(f"      --hdf5-dir  recon_data/recon_release \\")
    print(f"      --n-pairs 1000 --k-pos 4 \\")
    print(f"      --proprio-compare --hard-negatives --proprio-no-gps \\")
    print(f"      --proprio-ckpt {out_ckpt}")
    return best_acc


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Sprint 6b ProprioceptiveEncoder training")
    p.add_argument("--hdf5-dir",    default="recon_data/recon_release")
    p.add_argument("--out-ckpt",    default="checkpoints/cwm/proprio_6b_best.pt")
    p.add_argument("--epochs",      type=int,   default=40)
    p.add_argument("--steps",       type=int,   default=500)
    p.add_argument("--batch-size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--n-neg",       type=int,   default=7)
    p.add_argument("--k-max",       type=int,   default=4)
    p.add_argument("--k-hard-min",  type=int,   default=32)
    p.add_argument("--d-hidden",    type=int,   default=128)
    p.add_argument("--d-in",        type=int,   default=8)
    p.add_argument("--d-model",     type=int,   default=64)
    p.add_argument("--device",      default="cpu")
    args = p.parse_args()

    train(
        hdf5_dir    = args.hdf5_dir,
        out_ckpt    = args.out_ckpt,
        epochs      = args.epochs,
        steps_epoch = args.steps,
        batch_size  = args.batch_size,
        lr          = args.lr,
        temperature = args.temperature,
        n_neg       = args.n_neg,
        k_max       = args.k_max,
        k_hard_min  = args.k_hard_min,
        d_hidden    = args.d_hidden,
        d_in        = args.d_in,
        d_model     = args.d_model,
        device_str  = args.device,
    )
