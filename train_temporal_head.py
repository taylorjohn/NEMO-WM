"""
train_temporal_head.py  --  CORTEX Sprint 3b
=============================================
Retrain the TemporalHead on Sprint 3 CWM particle embeddings.

The TemporalHead learns to discriminate temporally-adjacent frame pairs
from random pairs, giving the AUROC metric its short-horizon (k=1)
discriminability. Sprint 1's TemporalHead (top1_acc=0.048) was trained
on Sprint 1 CWM particles (mean loss 0.567). Sprint 3 CWM particles
(mean loss ~0.165) are richer and require a re-matched head.

Training: contrastive pairs from RECON HDF5 files.
  Positive: frames t, t+k within the same trajectory (k sampled 1..k_max)
  Negative: frames from different HDF5 files (random)

Loss: InfoNCE (NT-Xent) over cosine similarity.
  top1_acc tracks whether the positive is the nearest neighbour among
  N negatives per batch — the same metric used by eval_recon_auroc.py.

Usage:
    python train_temporal_head.py ^
        --cwm-ckpt  checkpoints\\cwm\\cwm_multidomain_best.pt ^
        --out-ckpt  checkpoints\\cwm\\temporal_head_sprint3.pt ^
        --hdf5-dir  recon_data\\recon_release ^
        --epochs 20

Quick sanity check (3 epochs):
    python train_temporal_head.py --epochs 3 --n-pairs 2000

Sprint 3b target: top1_acc >= 0.15 (3× Sprint 1), k=1 AUROC >= 0.975
"""
import io, argparse, random, time
from pathlib import Path

import h5py, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from train_mvtec import StudentEncoder


# ── Architecture (must match eval_recon_auroc.py exactly) ─────────────────────

class TemporalHead(nn.Module):
    def __init__(self, d_model=128, embed_dim=64):
        super().__init__()
        self.pool = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        self.proj = nn.Sequential(
            nn.Linear(d_model, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, particles):
        x = particles.mean(dim=1)           # (B, d_model)
        return F.normalize(self.proj(self.pool(x)), dim=-1)


class ParticleEncoder(nn.Module):
    def __init__(self, input_dim=128, K=16, d_model=128):
        super().__init__()
        self.K = K
        self.d_model = d_model
        self.proj = nn.Linear(input_dim, K * d_model)

    def forward(self, z):
        B = z.shape[0]
        return F.normalize(self.proj(z).view(B, self.K, self.d_model), dim=-1)


# ── Data ───────────────────────────────────────────────────────────────────────

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def decode_frame(jpeg_bytes):
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    return TRANSFORM(img)


class TemporalPairDataset:
    """
    On-the-fly pair sampler from RECON HDF5 files.
    Returns (anchor_frame, positive_frame, [negative_frames...]).
    """

    def __init__(self, hdf5_dir, k_max=4, n_neg=7, seed=42):
        self.k_max  = k_max
        self.n_neg  = n_neg
        self.files  = sorted(Path(hdf5_dir).glob("jackal_2019-*.hdf5"))
        assert len(self.files) >= 2, f"Need ≥2 HDF5 files, found {len(self.files)}"
        random.seed(seed)
        np.random.seed(seed)
        print(f"  TemporalPairDataset: {len(self.files)} files, "
              f"k_max={k_max}, n_neg={n_neg}")

    def _load_frame(self, filepath, t):
        with h5py.File(filepath, "r") as hf:
            return decode_frame(bytes(hf["images"]["rgb_left"][t]))

    def sample_batch(self, batch_size=32):
        """
        Returns:
            anchors:   (B, 3, 224, 224)
            positives: (B, 3, 224, 224)
            negatives: (B, n_neg, 3, 224, 224)
        """
        anchors, positives, negatives = [], [], []
        attempts = 0
        while len(anchors) < batch_size and attempts < batch_size * 20:
            attempts += 1
            try:
                # Sample anchor + positive (same file, adjacent frames)
                f_pos = random.choice(self.files)
                with h5py.File(f_pos, "r") as hf:
                    L = hf["images"]["rgb_left"].shape[0]
                if L < self.k_max + 2:
                    continue
                t0 = random.randint(0, L - self.k_max - 2)
                k  = random.randint(1, self.k_max)
                anchor   = self._load_frame(f_pos, t0)
                positive = self._load_frame(f_pos, t0 + k)

                # Sample negatives (different files)
                neg_files = random.sample(
                    [f for f in self.files if f != f_pos],
                    min(self.n_neg, len(self.files) - 1)
                )
                neg_frames = []
                for nf in neg_files:
                    with h5py.File(nf, "r") as hf:
                        L_n = hf["images"]["rgb_left"].shape[0]
                    t_n = random.randint(0, L_n - 1)
                    neg_frames.append(self._load_frame(nf, t_n))

                if len(neg_frames) < self.n_neg:
                    continue

                anchors.append(anchor)
                positives.append(positive)
                negatives.append(torch.stack(neg_frames))   # (n_neg, 3, H, W)

            except Exception:
                continue

        if not anchors:
            return None, None, None

        return (
            torch.stack(anchors),                    # (B, 3, H, W)
            torch.stack(positives),                  # (B, 3, H, W)
            torch.stack(negatives),                  # (B, n_neg, 3, H, W)
        )


# ── Loss ───────────────────────────────────────────────────────────────────────

def infonce_loss(anchor_emb, positive_emb, negative_embs, temperature=0.07):
    """
    InfoNCE loss.
    anchor_emb:    (B, dim)
    positive_emb:  (B, dim)
    negative_embs: (B, n_neg, dim)

    Returns:
        loss:     scalar
        top1_acc: fraction of anchors where positive is nearest neighbour
    """
    B, dim = anchor_emb.shape
    n_neg  = negative_embs.shape[1]

    # Similarity to positive: (B,)
    sim_pos = (anchor_emb * positive_emb).sum(dim=-1, keepdim=True) / temperature

    # Similarity to negatives: (B, n_neg)
    sim_neg = torch.bmm(
        negative_embs,                          # (B, n_neg, dim)
        anchor_emb.unsqueeze(-1)                # (B, dim, 1)
    ).squeeze(-1) / temperature                 # (B, n_neg)

    # Logits: positive is class 0, negatives are classes 1..n_neg
    logits = torch.cat([sim_pos, sim_neg], dim=1)   # (B, 1+n_neg)
    labels = torch.zeros(B, dtype=torch.long, device=anchor_emb.device)

    loss = F.cross_entropy(logits, labels)

    # top1_acc: positive rank 1
    with torch.no_grad():
        top1_acc = (logits.argmax(dim=1) == 0).float().mean().item()

    return loss, top1_acc


# ── Training loop ──────────────────────────────────────────────────────────────

def train(
    cwm_ckpt:   str  = r"checkpoints\cwm\cwm_multidomain_best.pt",
    out_ckpt:   str  = r"checkpoints\cwm\temporal_head_sprint3.pt",
    hdf5_dir:   str  = "recon_data/recon_release",
    epochs:     int  = 20,
    lr:         float = 3e-4,
    batch_size: int  = 32,
    n_neg:      int  = 7,
    k_max:      int  = 4,
    temperature: float = 0.07,
    steps_per_epoch: int = 500,
    K:          int  = 16,
    d_model:    int  = 128,
    embed_dim:  int  = 64,
    device_str: str  = "cpu",
    seed:       int  = 42,
):
    torch.manual_seed(seed)
    random.seed(seed)
    dev = torch.device(device_str)

    print(f"\nSprint 3b — TemporalHead retrain")
    print(f"  CWM ckpt:   {cwm_ckpt}")
    print(f"  Output:     {out_ckpt}")
    print(f"  Epochs:     {epochs}  |  Steps/epoch: {steps_per_epoch}")
    print(f"  LR: {lr}  |  Temp: {temperature}  |  n_neg: {n_neg}  |  k_max: {k_max}")

    # ── Load frozen StudentEncoder + ParticleEncoder ──────────────────────────
    enc = StudentEncoder().to(dev)
    enc_path = Path("checkpoints/dinov2_student/student_best.pt")
    if enc_path.exists():
        sd = torch.load(enc_path, map_location="cpu", weights_only=False)
        enc.load_state_dict(sd.get("model", sd.get("state_dict", sd)), strict=False)
        print(f"  StudentEncoder: {enc_path}")
    else:
        print("  WARNING: student_best.pt not found — random encoder")
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)

    pe = ParticleEncoder(128, K, d_model).to(dev)
    if Path(cwm_ckpt).exists():
        cwm_data = torch.load(cwm_ckpt, map_location=dev, weights_only=False)
        cwm_sd   = cwm_data.get("model", cwm_data)
        ep_saved = cwm_data.get("epoch", "?")
        loss_saved = cwm_data.get("loss", "?")
        pe_sd = {
            k.replace("particle_enc.", ""): v
            for k, v in cwm_sd.items()
            if k.startswith("particle_enc.")
        }
        if pe_sd:
            pe.load_state_dict(pe_sd, strict=False)
            print(f"  ParticleEncoder: {cwm_ckpt} (ep{ep_saved}, loss {loss_saved:.4f})")
        else:
            print(f"  WARNING: no particle_enc.* keys found in {cwm_ckpt}")
    else:
        print(f"  WARNING: CWM ckpt not found: {cwm_ckpt}")
    pe.eval()
    for p in pe.parameters():
        p.requires_grad_(False)

    # ── TemporalHead — trainable ──────────────────────────────────────────────
    # Load Sprint 1 head as initialisation if available (warm start)
    sprint1_head = Path("checkpoints/cwm/temporal_head_best.pt")
    head = TemporalHead(d_model, embed_dim).to(dev)
    if sprint1_head.exists():
        hd = torch.load(sprint1_head, map_location=dev, weights_only=False)
        head.load_state_dict(hd.get("head", hd), strict=False)
        s1_acc = hd.get("acc", hd.get("top1_acc", "?"))
        print(f"  TemporalHead init: {sprint1_head} (Sprint1 top1_acc={s1_acc})")
    else:
        print("  TemporalHead: random init")
    head.train()

    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)

    # ── Data sampler ──────────────────────────────────────────────────────────
    ds = TemporalPairDataset(hdf5_dir, k_max=k_max, n_neg=n_neg, seed=seed)

    best_acc = 0.0
    out_path = Path(out_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'Epoch':>5}  {'Loss':>8}  {'top1_acc':>10}  {'LR':>10}  {'Elapsed':>8}")
    print("─" * 55)

    for ep in range(epochs):
        t0 = time.time()
        losses, accs = [], []

        for step in range(steps_per_epoch):
            anc_imgs, pos_imgs, neg_imgs = ds.sample_batch(batch_size)
            if anc_imgs is None:
                continue

            anc_imgs = anc_imgs.to(dev)
            pos_imgs = pos_imgs.to(dev)
            neg_imgs = neg_imgs.to(dev)   # (B, n_neg, 3, H, W)
            B = anc_imgs.shape[0]

            with torch.no_grad():
                # Encode anchors
                z_anc = enc(anc_imgs)               # (B, 128)
                pts_anc = pe(z_anc)                 # (B, K, d_model)

                # Encode positives
                z_pos = enc(pos_imgs)
                pts_pos = pe(z_pos)

                # Encode negatives: reshape to (B*n_neg, 3, H, W)
                B_n, n_neg_actual, C, H, W = neg_imgs.shape
                neg_flat = neg_imgs.view(B_n * n_neg_actual, C, H, W)
                z_neg    = enc(neg_flat)            # (B*n_neg, 128)
                pts_neg  = pe(z_neg)               # (B*n_neg, K, d_model)

            # Head forward (trainable)
            emb_anc = head(pts_anc)                 # (B, embed_dim)
            emb_pos = head(pts_pos)                 # (B, embed_dim)
            emb_neg = head(pts_neg)                 # (B*n_neg, embed_dim)
            emb_neg = emb_neg.view(B, n_neg_actual, -1)   # (B, n_neg, embed_dim)

            loss, top1 = infonce_loss(emb_anc, emb_pos, emb_neg, temperature)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()

            losses.append(loss.item())
            accs.append(top1)

        sched.step()

        mean_loss = np.mean(losses) if losses else float("nan")
        mean_acc  = np.mean(accs)   if accs   else 0.0
        elapsed   = time.time() - t0
        lr_now    = opt.param_groups[0]["lr"]

        print(f"{ep:>5}  {mean_loss:>8.4f}  {mean_acc:>10.4f}  {lr_now:>10.2e}  {elapsed:>7.1f}s")

        # Save if improved
        if mean_acc > best_acc:
            best_acc = mean_acc
            torch.save({
                "epoch":    ep,
                "top1_acc": mean_acc,
                "acc":      mean_acc,
                "loss":     mean_loss,
                "head":     head.state_dict(),
                "cwm_ckpt": cwm_ckpt,
                "sprint":   "3b",
            }, out_path)
            print(f"       → Saved (best top1_acc={best_acc:.4f}): {out_path}")

    print(f"\nSprint 3b complete.")
    print(f"  Best top1_acc: {best_acc:.4f}")
    print(f"  Checkpoint:    {out_path}")
    print(f"\nEval command:")
    print(f"  python eval_recon_auroc.py --k-sweep --n-pairs 10000 "
          f"--cwm-ckpt {cwm_ckpt} --head-ckpt {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Sprint 3b — TemporalHead retrain")
    p.add_argument("--cwm-ckpt",        default=r"checkpoints\cwm\cwm_multidomain_best.pt")
    p.add_argument("--out-ckpt",        default=r"checkpoints\cwm\temporal_head_sprint3.pt")
    p.add_argument("--hdf5-dir",        default="recon_data/recon_release")
    p.add_argument("--epochs",          type=int,   default=20)
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--batch-size",      type=int,   default=32)
    p.add_argument("--n-neg",           type=int,   default=7)
    p.add_argument("--k-max",           type=int,   default=4)
    p.add_argument("--temperature",     type=float, default=0.07)
    p.add_argument("--steps-per-epoch", type=int,   default=500)
    p.add_argument("--device",          default="cpu")
    p.add_argument("--seed",            type=int,   default=42)
    a = p.parse_args()
    train(
        cwm_ckpt        = a.cwm_ckpt,
        out_ckpt        = a.out_ckpt,
        hdf5_dir        = a.hdf5_dir,
        epochs          = a.epochs,
        lr              = a.lr,
        batch_size      = a.batch_size,
        n_neg           = a.n_neg,
        k_max           = a.k_max,
        temperature     = a.temperature,
        steps_per_epoch = a.steps_per_epoch,
        device_str      = a.device,
        seed            = a.seed,
    )
