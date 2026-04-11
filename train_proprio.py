"""
train_proprio.py  --  NeMo-WM Sprint 6
=======================================
Train ProprioceptiveEncoder on RECON dataset using contrastive InfoNCE.

Objective: learn temporal proximity from vel/ang/heading/contact signals
WITHOUT GPS (removes the GPS-scale shortcut from the baseline eval).

After training, run the dissociation eval:
    python eval_recon_auroc.py ^
        --head-ckpt checkpoints/cwm/temporal_head_sprint3.pt ^
        --cwm-ckpt  checkpoints/cwm/cwm_multidomain_best.pt ^
        --hdf5-dir  recon_data/recon_release ^
        --n-pairs 1000 --k-pos 4 ^
        --proprio-compare --hard-negatives --proprio-no-gps ^
        --proprio-ckpt checkpoints/cwm/proprio_best.pt

Sprint 6 PASSED when:
  Full (VLM + trained proprio) AUROC > 0.865  (beats VLM-only baseline)
  No-VLM (trained proprio only) AUROC > 0.792  (beats random baseline)

Usage:
    python train_proprio.py ^
        --hdf5-dir recon_data/recon_release ^
        --out-ckpt checkpoints/cwm/proprio_best.pt ^
        --epochs 20

Design mirrors train_temporal_head.py (Sprint 3b):
  - Same contrastive InfoNCE loss (temperature=0.07)
  - Same cosine LR schedule
  - Same n_neg=7, k_max=4, k_hard_min=32
  - Same checkpoint-on-best logic
"""

import argparse, random, math, time
from pathlib import Path
import h5py, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from proprioceptive_encoder import ProprioceptiveEncoder, ProprioConfig


# ===========================================================================
# Proprioceptive pair dataset
# ===========================================================================

class ProprioPairDataset:
    """
    Samples (anchor, positive, [negatives]) triples from RECON HDF5.

    Anchor:    frame t
    Positive:  frame t + k, where k in [1, k_max]  (temporally close)
    Negatives: frames t + k_neg where k_neg >= k_hard_min  (temporally far)
               sampled from the SAME file — no GPS cross-file shortcut

    Signals extracted per frame (no GPS):
        linear_velocity, angular_velocity, heading (sin/cos), contact
    """

    def __init__(
        self,
        hdf5_dir:     str,
        k_max:        int   = 4,
        k_hard_min:   int   = 32,
        n_neg:        int   = 7,
        max_files:    int   = None,
        seed:         int   = 42,
    ):
        random.seed(seed); np.random.seed(seed)
        files = sorted(Path(hdf5_dir).glob("jackal_2019-*.hdf5"))
        if max_files:
            files = files[:max_files]
        self.files      = [str(f) for f in files]
        self.k_max      = k_max
        self.k_hard_min = k_hard_min
        self.n_neg      = n_neg
        assert len(self.files) >= 1, "No RECON files found"
        print(f"ProprioPairDataset: {len(self.files)} files, "
              f"k_max={k_max}, k_hard_min={k_hard_min}, n_neg={n_neg}")

    def _extract_signals(self, hf, t):
        """Extract (vel, ang, heading, contact) — NO GPS."""
        vel_val = float(hf["commands"]["linear_velocity"][t])
        ang_val = float(hf["commands"]["angular_velocity"][t])

        # Heading: integrated angular velocity from frame 0
        ang_all       = hf["commands"]["angular_velocity"][:t+1]
        heading_angle = float(np.sum(ang_all)) / 4.0   # 4Hz
        sin_h         = math.sin(heading_angle)
        cos_h         = math.cos(heading_angle)

        contact = 1.0 if abs(vel_val) > 0.3 else 0.0

        # Pack into (7,) vector: vel, ang, gps_zeros(2), contact, sin, cos
        # Match ProprioceptiveEncoder input layout (d_in=7)
        return torch.tensor(
            [vel_val, ang_val, 0.0, 0.0, contact, sin_h, cos_h],
            dtype=torch.float32
        )

    def _safe(self, t):
        return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

    def sample_batch(self, batch_size: int = 64):
        """
        Returns dict with:
            anchor   : (B, 7)
            positive : (B, 7)
            negatives: (B, n_neg, 7)
        """
        anchors, positives, negatives_list = [], [], []
        attempts = 0
        while len(anchors) < batch_size and attempts < batch_size * 20:
            attempts += 1
            filepath = random.choice(self.files)
            try:
                with h5py.File(filepath, "r") as hf:
                    L = hf["images"]["rgb_left"].shape[0]
                    # Need room for anchor + positive + negatives
                    if L < self.k_hard_min + self.k_max + 2:
                        continue

                    # Anchor
                    t_anchor = random.randint(0, L - self.k_max - 2)

                    # Positive: k in [1, k_max]
                    k = random.randint(1, self.k_max)
                    t_pos = t_anchor + k

                    # Negatives: same file, k_neg >= k_hard_min
                    neg_signals = []
                    neg_candidates = [
                        t for t in range(L)
                        if abs(t - t_anchor) >= self.k_hard_min
                    ]
                    if len(neg_candidates) < self.n_neg:
                        continue
                    neg_ts = random.sample(neg_candidates, self.n_neg)

                    z_anchor = self._safe(self._extract_signals(hf, t_anchor))
                    z_pos    = self._safe(self._extract_signals(hf, t_pos))
                    z_negs   = torch.stack([
                        self._safe(self._extract_signals(hf, t_n))
                        for t_n in neg_ts
                    ])  # (n_neg, 7)

                    anchors.append(z_anchor)
                    positives.append(z_pos)
                    negatives_list.append(z_negs)
            except Exception:
                continue

        if len(anchors) == 0:
            return None

        return {
            "anchor":    torch.stack(anchors),                       # (B, 7)
            "positive":  torch.stack(positives),                     # (B, 7)
            "negatives": torch.stack(negatives_list),                # (B, n_neg, 7)
        }


# ===========================================================================
# ProprioEncoder wrapper that takes (B, 7) raw input
# ===========================================================================

class ProprioEncoderDirect(nn.Module):
    """
    Wraps ProprioceptiveEncoder to accept pre-packed (B, 7) tensors
    instead of the five separate tensors the original expects.
    
    Input layout: [vel, ang, gps_n, gps_e, contact, sin_h, cos_h]
    GPS channels are always 0 in the no-GPS training regime.
    """
    def __init__(self, config=None):
        super().__init__()
        cfg = config or ProprioConfig(d_model=64)
        # Build the MLP directly — same as ProprioceptiveEncoder but accepts (B,7)
        d_in     = cfg.d_in      # 7
        d_hidden = cfg.d_hidden  # 64
        d_out    = cfg.d_model   # 64

        layers = []
        din = d_in
        for i in range(cfg.n_layers):
            dout = d_hidden if i < cfg.n_layers - 1 else d_out
            layers.append(nn.Linear(din, dout))
            if cfg.use_layernorm and i < cfg.n_layers - 1:
                layers.append(nn.LayerNorm(dout))
            if i < cfg.n_layers - 1:
                layers.append(nn.GELU())
                if cfg.dropout > 0:
                    layers.append(nn.Dropout(cfg.dropout))
            din = dout
        self.net = nn.Sequential(*layers)
        self.cfg = cfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 7) raw signals → (B, d_model) unit-normalised."""
        return F.normalize(self.net(x), dim=-1)

    def save(self, path, epoch=None, loss=None, acc=None):
        torch.save({
            "model":    self.state_dict(),
            "config":   self.cfg,
            "epoch":    epoch,
            "loss":     loss,
            "top1_acc": acc,
        }, path)
        print(f"  → Saved: {path} (epoch={epoch}, loss={loss:.4f}, top1_acc={acc:.4f})")

    @classmethod
    def load(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg  = ckpt.get("config", ProprioConfig(d_model=64))
        enc  = cls(cfg)
        enc.load_state_dict(ckpt["model"])
        acc = ckpt.get("top1_acc", "?")
        print(f"ProprioEncoder loaded: {path} "
              f"(epoch={ckpt.get('epoch','?')}, top1_acc={acc})")
        return enc


# ===========================================================================
# InfoNCE contrastive loss
# ===========================================================================

def infonce_loss(
    z_anchor:    torch.Tensor,   # (B, D) unit-normalised
    z_positive:  torch.Tensor,   # (B, D)
    z_negatives: torch.Tensor,   # (B, n_neg, D)
    temperature: float = 0.07,
) -> tuple[torch.Tensor, float]:
    """
    InfoNCE loss with in-batch negatives.
    Returns (loss, top1_accuracy).
    """
    B, D = z_anchor.shape
    n_neg = z_negatives.shape[1]

    # Positive similarity
    sim_pos = F.cosine_similarity(z_anchor, z_positive, dim=-1) / temperature  # (B,)

    # Negative similarities
    z_neg_flat = z_negatives.view(B * n_neg, D)                    # (B*n_neg, D)
    z_anc_rep  = z_anchor.unsqueeze(1).expand(-1, n_neg, -1)       # (B, n_neg, D)
    sim_neg    = F.cosine_similarity(
        z_anc_rep.reshape(B * n_neg, D), z_neg_flat, dim=-1
    ).view(B, n_neg) / temperature                                 # (B, n_neg)

    # Logits: [pos | neg1 | neg2 | ...]
    logits = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1)     # (B, 1+n_neg)
    labels = torch.zeros(B, dtype=torch.long, device=z_anchor.device)  # positive is idx 0

    loss    = F.cross_entropy(logits, labels)
    top1    = (logits.argmax(dim=1) == 0).float().mean().item()

    return loss, top1


# ===========================================================================
# Training loop
# ===========================================================================

def train(
    hdf5_dir:     str,
    out_ckpt:     str,
    epochs:       int   = 20,
    steps_epoch:  int   = 500,
    batch_size:   int   = 64,
    lr:           float = 3e-4,
    temperature:  float = 0.07,
    n_neg:        int   = 7,
    k_max:        int   = 4,
    k_hard_min:   int   = 32,
    device_str:   str   = "cpu",
):
    dev = torch.device(device_str)
    out_path = Path(out_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = ProprioConfig(d_in=7, d_hidden=64, d_model=64, n_layers=3)
    enc = ProprioEncoderDirect(cfg).to(dev)
    n_params = sum(p.numel() for p in enc.parameters())
    print(f"\nSprint 6 — ProprioceptiveEncoder Training")
    print(f"  d_in={cfg.d_in}, d_hidden={cfg.d_hidden}, d_model={cfg.d_model}")
    print(f"  Parameters: {n_params:,}")
    print(f"  Epochs: {epochs}  |  Steps/epoch: {steps_epoch}")
    print(f"  LR: {lr}  |  Temp: {temperature}  |  n_neg: {n_neg}")
    print(f"  k_max: {k_max}  |  k_hard_min: {k_hard_min}")
    print(f"  No GPS (vel+ang+heading+contact only)")
    print(f"  Output: {out_ckpt}\n")

    ds = ProprioPairDataset(
        hdf5_dir, k_max=k_max, k_hard_min=k_hard_min,
        n_neg=n_neg,
    )

    opt = torch.optim.Adam(enc.parameters(), lr=lr)
    sched = CosineAnnealingLR(opt, T_max=epochs * steps_epoch, eta_min=lr * 0.01)

    best_acc  = 0.0
    best_loss = float("inf")

    print(f"{'Epoch':>6}  {'Loss':>8}  {'top1_acc':>10}  {'LR':>12}  {'Elapsed':>10}")
    print(f"{'─'*55}")

    for epoch in range(epochs):
        enc.train()
        epoch_losses, epoch_accs = [], []
        t0 = time.perf_counter()

        for step in range(steps_epoch):
            batch = ds.sample_batch(batch_size)
            if batch is None:
                continue

            anchor    = batch["anchor"].to(dev)
            positive  = batch["positive"].to(dev)
            negatives = batch["negatives"].to(dev)

            z_a = enc(anchor)
            z_p = enc(positive)
            z_n = torch.stack([enc(negatives[:, i]) for i in range(n_neg)], dim=1)

            loss, acc = infonce_loss(z_a, z_p, z_n, temperature)

            opt.zero_grad()
            loss.backward()
            # Gradient clipping (mirrors TemporalHead training)
            torch.nn.utils.clip_grad_norm_(enc.parameters(), max_norm=1.0)
            opt.step()
            sched.step()

            epoch_losses.append(loss.item())
            epoch_accs.append(acc)

        mean_loss = np.mean(epoch_losses) if epoch_losses else float("inf")
        mean_acc  = np.mean(epoch_accs)  if epoch_accs  else 0.0
        elapsed   = time.perf_counter() - t0
        current_lr = sched.get_last_lr()[0]

        print(f"  {epoch:>4}  {mean_loss:>8.4f}  {mean_acc:>10.4f}  {current_lr:>12.2e}  {elapsed:>9.1f}s")

        if mean_acc > best_acc or (mean_acc == best_acc and mean_loss < best_loss):
            best_acc  = mean_acc
            best_loss = mean_loss
            enc.save(out_path, epoch=epoch, loss=mean_loss, acc=mean_acc)

    print(f"\nTraining complete. Best top1_acc={best_acc:.4f}")
    print(f"Checkpoint: {out_ckpt}")
    print(f"\nNext: run dissociation eval with --proprio-ckpt {out_ckpt}")
    print(f"  Sprint 6 PASSED when Full AUROC > 0.865 (VLM-only baseline)")
    return best_acc


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Sprint 6 ProprioceptiveEncoder training")
    p.add_argument("--hdf5-dir",    default="recon_data/recon_release")
    p.add_argument("--out-ckpt",    default="checkpoints/cwm/proprio_best.pt")
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--steps",       type=int,   default=500)
    p.add_argument("--batch-size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--n-neg",       type=int,   default=7)
    p.add_argument("--k-max",       type=int,   default=4)
    p.add_argument("--k-hard-min",  type=int,   default=32)
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
        device_str  = args.device,
    )
