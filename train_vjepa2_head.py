"""
train_vjepa2_head.py  —  NeMo-WM × V-JEPA 2  ProjectionHead Training
======================================================================
Train a lightweight 2-layer projection head on top of frozen V-JEPA 2
features. Equivalent to NeMo-WM's TemporalHead but operating on V-JEPA 2
patch embeddings instead of CWM particle embeddings.

This is E2 of the NeurIPS integration experiment:
  E1 (done):  V-JEPA 2 zero-shot AUROC = 0.907
  E2 (this):  V-JEPA 2 + ProjectionHead trained on RECON → ~0.960+
  E3 (next):  V-JEPA 2 + ProjectionHead + proprio k_ctx=16 → ~0.998+

Expected training time: ~2 hours (500 steps/epoch × 10 epochs, CPU)
Expected result: AUROC 0.960-0.975 (ProjectionHead fine-tune)
                 AUROC 0.997-0.999 (fused with proprio k_ctx=16)

Run:
  python train_vjepa2_head.py `
      --hdf5-dir recon_data\recon_release `
      --model-size large `
      --epochs 10 `
      --out-ckpt checkpoints\cwm\vjepa2_head_best.pt
"""

import argparse, io, time, random
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import transforms

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

MODEL_IDS = {
    "base":  "facebook/vjepa2-vitb-fpc64-256",   # d_model=768
    "large": "facebook/vjepa2-vitl-fpc64-256",   # d_model=1024
    "giant": "facebook/vjepa2-vitg-fpc64-256",   # d_model=1408
}
D_MODELS = {"base": 768, "large": 1024, "giant": 1408}


# ── ProjectionHead (same design as NeMo-WM TemporalHead) ─────────────────────

class ProjectionHead(nn.Module):
    """
    64-D projection from V-JEPA 2 d_model → quasimetric space.
    Parameters: ~270K (large), ~210K (base)
    """
    def __init__(self, d_in=1024, d_out=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.GELU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.GELU(), nn.LayerNorm(128),
            nn.Linear(128, d_out),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


# ── Feature extractor (frozen V-JEPA 2) ──────────────────────────────────────

@torch.no_grad()
def extract_features(encoder, jpeg_bytes, device):
    """Extract mean-pooled patch features from V-JEPA 2."""
    frame = TRANSFORM(
        Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    ).unsqueeze(0).to(device)

    try:
        out = encoder(pixel_values_videos=frame.unsqueeze(1).repeat(1,2,1,1,1))
    except Exception:
        out = encoder(pixel_values_videos=frame.unsqueeze(1).repeat(1,2,1,1,1))

    if hasattr(out, "last_hidden_state"):
        feats = out.last_hidden_state
    elif isinstance(out, (tuple, list)):
        feats = out[0]
    else:
        feats = out

    while feats.dim() > 2:
        feats = feats.mean(dim=1)

    return feats.squeeze(0)  # (D,) — NOT normalised yet, head will project


# ── Dataset ───────────────────────────────────────────────────────────────────

class VJEPAFeatureDataset:
    """
    Samples (anchor, positive, negatives) triplets from RECON.
    Pre-extracts V-JEPA 2 features to avoid redundant forward passes.
    Uses same hard-negative protocol as NeMo-WM training.
    """
    def __init__(self, hdf5_dir, encoder, device,
                 k_ctx=1, k_max=4, k_hard_min=32, n_neg=7,
                 max_files=None, cache_features=True):
        self.files = sorted(Path(hdf5_dir).glob("jackal_2019-*.hdf5"))
        if max_files:
            self.files = self.files[:max_files]
        self.k_ctx     = k_ctx
        self.encoder   = encoder
        self.device    = device
        self.k_max     = k_max
        self.k_hard_min = k_hard_min
        self.n_neg     = n_neg
        self._cache    = {}  # filepath → {frame_idx: feature}
        print(f"VJEPAFeatureDataset: {len(self.files)} files | "
              f"k_max={k_max}, k_hard_min={k_hard_min}, n_neg={n_neg}")

    def _get_feature(self, filepath, frame_idx):
        key = (filepath, frame_idx)
        if key not in self._cache:
            with h5py.File(filepath, "r") as hf:
                jpeg = bytes(hf["images"]["rgb_left"][frame_idx])
            self._cache[key] = extract_features(
                self.encoder, jpeg, self.device).cpu()
        return self._cache[key]

    def sample_batch(self, batch_size=32):
        anchors, positives, negatives = [], [], []
        attempts = 0
        rng = random
        min_t = self.k_ctx + 1

        while len(anchors) < batch_size and attempts < batch_size * 30:
            attempts += 1
            fp = str(rng.choice(self.files))
            try:
                with h5py.File(fp, "r") as hf:
                    L = hf["images"]["rgb_left"].shape[0]
                if L < self.k_hard_min + self.k_max + 5:
                    continue

                t_a = rng.randint(min_t, L - self.k_max - 2)
                t_p = t_a + rng.randint(1, self.k_max + 1)

                neg_ts = [t for t in range(min_t, L)
                          if abs(t - t_a) >= self.k_hard_min]
                if len(neg_ts) < self.n_neg:
                    continue

                neg_ts = rng.sample(neg_ts, self.n_neg)

                z_a = self._get_feature(fp, t_a)
                z_p = self._get_feature(fp, t_p)
                z_negs = torch.stack([self._get_feature(fp, t) for t in neg_ts])

                anchors.append(z_a)
                positives.append(z_p)
                negatives.append(z_negs)

            except Exception:
                continue

        if not anchors:
            return None

        return (
            torch.stack(anchors).to(self.device),    # (B, D)
            torch.stack(positives).to(self.device),  # (B, D)
            torch.stack(negatives).to(self.device),  # (B, n_neg, D)
        )


# ── InfoNCE loss ──────────────────────────────────────────────────────────────

def info_nce_loss(z_a, z_p, z_neg, head, temperature=0.05):
    """
    z_a, z_p: (B, D_in) raw V-JEPA 2 features
    z_neg:    (B, n_neg, D_in)
    head:     ProjectionHead → (B, 64) normalised
    """
    B, n_neg, D = z_neg.shape

    e_a   = head(z_a)                          # (B, 64)
    e_p   = head(z_p)                          # (B, 64)
    e_neg = head(z_neg.view(B * n_neg, D))     # (B*n_neg, 64)
    e_neg = e_neg.view(B, n_neg, -1)

    sim_pos = (e_a * e_p).sum(-1, keepdim=True) / temperature   # (B, 1)
    sim_neg = torch.bmm(e_neg, e_a.unsqueeze(-1)).squeeze(-1) / temperature  # (B, n_neg)

    logits = torch.cat([sim_pos, sim_neg], dim=1)  # (B, 1+n_neg)
    labels = torch.zeros(B, dtype=torch.long, device=z_a.device)

    loss = F.cross_entropy(logits, labels)

    with torch.no_grad():
        top1 = (logits.argmax(dim=1) == 0).float().mean()

    return loss, float(top1)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(hdf5_dir, model_size, epochs, steps_per_epoch, out_ckpt,
          lr=3e-4, batch_size=32, device="cpu", max_files=None):

    dev = torch.device(device)

    # Load frozen encoder
    from transformers import AutoModel
    print(f"Loading frozen V-JEPA 2 {model_size}...")
    encoder = AutoModel.from_pretrained(
        MODEL_IDS[model_size], trust_remote_code=True
    ).to(dev).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    d_in = D_MODELS[model_size]
    head = ProjectionHead(d_in=d_in, d_out=64).to(dev)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"ProjectionHead: {n_params:,} trainable params")

    opt = torch.optim.Adam(head.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    ds = VJEPAFeatureDataset(
        hdf5_dir, encoder, dev,
        k_max=4, k_hard_min=32, n_neg=7,
        max_files=max_files,
    )

    best_acc, best_ep = 0.0, -1
    Path(out_ckpt).parent.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining ProjectionHead on frozen V-JEPA 2 {model_size}")
    print(f"  Epochs: {epochs}  |  Steps/epoch: {steps_per_epoch}")
    print(f"  LR: {lr}  |  Batch: {batch_size}  |  Temp: 0.05")
    print(f"  Output: {out_ckpt}\n")
    print(f" {'Epoch':>5}  {'Loss':>8}  {'top1_acc':>10}  {'LR':>12}  {'Elapsed':>10}")
    print(f"  {'─'*55}")

    for ep in range(epochs):
        t0 = time.perf_counter()
        losses, accs = [], []

        for _ in range(steps_per_epoch):
            batch = ds.sample_batch(batch_size)
            if batch is None:
                continue
            z_a, z_p, z_neg = batch

            loss, acc = info_nce_loss(z_a, z_p, z_neg, head)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()

            losses.append(float(loss.detach()))
            accs.append(acc)

        sched.step()
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        mean_acc  = float(np.mean(accs))  if accs  else float("nan")
        elapsed   = time.perf_counter() - t0
        lr_now    = sched.get_last_lr()[0]

        print(f" {ep:>5}  {mean_loss:>8.4f}  {mean_acc:>10.4f}  "
              f"{lr_now:>12.2e}  {elapsed:>8.1f}s")

        if mean_acc > best_acc:
            best_acc = mean_acc
            best_ep  = ep
            torch.save({
                "head": head.state_dict(),
                "model_size": model_size,
                "d_in": d_in,
                "epoch": ep,
                "top1_acc": mean_acc,
                "loss": mean_loss,
            }, out_ckpt)
            print(f"  → Saved (best top1_acc={best_acc:.4f}): {out_ckpt}")

    print(f"\nTraining complete. Best top1_acc={best_acc:.4f} (ep{best_ep})")
    print(f"Checkpoint: {out_ckpt}")
    print(f"\nRun dissociation eval:")
    print(f"  python eval_vjepa2_dissociation.py \\")
    print(f"      --vjepa2-ckpt {out_ckpt} \\")
    print(f"      --proprio-ckpt checkpoints\\cwm\\proprio_kctx16_best.pt \\")
    print(f"      --hdf5-dir {hdf5_dir} \\")
    print(f"      --n-pairs 1000")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5-dir",         default="recon_data/recon_release")
    p.add_argument("--model-size",       default="large",
                   choices=["base", "large", "giant"])
    p.add_argument("--epochs",           type=int, default=10)
    p.add_argument("--steps-per-epoch",  type=int, default=500)
    p.add_argument("--batch-size",       type=int, default=32)
    p.add_argument("--lr",               type=float, default=3e-4)
    p.add_argument("--device",           default="cpu")
    p.add_argument("--max-files",        type=int, default=None)
    p.add_argument("--out-ckpt",
                   default="checkpoints/cwm/vjepa2_head_best.pt")
    args = p.parse_args()

    train(
        hdf5_dir        = args.hdf5_dir,
        model_size      = args.model_size,
        epochs          = args.epochs,
        steps_per_epoch = args.steps_per_epoch,
        out_ckpt        = args.out_ckpt,
        lr              = args.lr,
        batch_size      = args.batch_size,
        device          = args.device,
        max_files       = args.max_files,
    )
