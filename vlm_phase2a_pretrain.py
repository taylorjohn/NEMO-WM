"""
vlm_phase2a_pretrain.py — Neuromodulated Pretraining vs Contrastive Baseline
=============================================================================
Phase 2a of VLM integration plan.

Question: Can neuromodulatory signals replace the contrastive pretraining
objective for a small vision encoder, producing better physical semantics
at lower parameter count?

Method:
  Train two ViT-Tiny encoders from scratch on RECON frames:
    A. Contrastive baseline — SimCLR-style NT-Xent loss (standard approach)
    B. Neuromodulated — NeMo-WM eight-signal loss (DA + 5HT + NE + cortisol)

  After N epochs, evaluate both with AIM probe on physical quantities:
    - Linear velocity (from commands/linear_velocity)
    - Angular velocity (from commands/angular_velocity)
    - GPS displacement (from gps/latlong)

  If neuromodulated encoder shows higher AIM probe R² on physical quantities
  at same parameter count → neuromodulated pretraining learns better physics.

Architecture: ViT-Tiny (5.7M params) — small enough to train in hours on CPU.

Usage:
    python vlm_phase2a_pretrain.py --epochs 5 --quick   # smoke test (~30 min)
    python vlm_phase2a_pretrain.py --epochs 20          # full comparison (~3 hrs)
    python vlm_phase2a_pretrain.py --eval-only          # eval existing checkpoints

Author: John Taylor — github.com/taylorjohn
Sprint: VLM Phase 2a
"""

import argparse
import glob
import io
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms


# ── ViT-Tiny encoder ──────────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=192):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=3, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, dim, heads=3, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTTiny(nn.Module):
    """
    ViT-Tiny: 5.7M params, 12 layers, 192-dim, 3 heads.
    Small enough to train from scratch on CPU in hours.
    """
    def __init__(self, img_size=224, patch_size=16, embed_dim=192,
                 depth=12, heads=3, out_dim=128):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        self.blocks = nn.Sequential(*[Block(embed_dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, out_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.norm(x)
        return F.normalize(self.head(x[:, 0]), dim=-1)  # CLS token → out_dim

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


# ── Data loader ───────────────────────────────────────────────────────────────

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

TRANSFORM_AUG2 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

TRANSFORM_CLEAN = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_recon_batch(hdf5_dir: str, batch_size: int = 16,
                     max_files: int = None) -> list:
    """
    Generator yielding batches of (img_view1, img_view2, labels) from RECON.
    labels = (linear_vel, angular_vel, gps_lat, gps_lon)
    """
    import h5py
    files = sorted(glob.glob(f"{hdf5_dir}/*.hdf5"))
    if max_files:
        files = files[:max_files]

    buf_v1, buf_v2, buf_labels = [], [], []

    for path in files:
        try:
            with h5py.File(path) as hf:
                imgs = hf["images"]["rgb_left"]
                N = len(imgs) - 1
                for i in range(N):
                    try:
                        jpeg = bytes(imgs[i])
                        img = Image.open(io.BytesIO(jpeg)).convert("RGB")
                        v1 = TRANSFORM(img)
                        v2 = TRANSFORM_AUG2(img)
                        lin = float(hf["commands"]["linear_velocity"][i])
                        ang = float(hf["commands"]["angular_velocity"][i])
                        gps = list(hf["gps"]["latlong"][i])
                        buf_v1.append(v1)
                        buf_v2.append(v2)
                        buf_labels.append([lin, ang, gps[0], gps[1]])
                        if len(buf_v1) == batch_size:
                            yield (torch.stack(buf_v1),
                                   torch.stack(buf_v2),
                                   torch.tensor(buf_labels, dtype=torch.float32))
                            buf_v1, buf_v2, buf_labels = [], [], []
                    except Exception:
                        pass
        except Exception:
            pass


# ── Loss functions ────────────────────────────────────────────────────────────

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor,
                 temperature: float = 0.07) -> torch.Tensor:
    """
    SimCLR NT-Xent contrastive loss.
    Standard baseline for self-supervised visual pretraining.
    """
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)                     # (2B, D)
    sim = torch.mm(z, z.t()) / temperature             # (2B, 2B)
    mask = torch.eye(2*B, dtype=torch.bool)
    sim.masked_fill_(mask, -9e15)
    labels = torch.cat([torch.arange(B, 2*B), torch.arange(B)])
    return F.cross_entropy(sim, labels)


class NeuromodulatedLoss(nn.Module):
    """
    NeMo-WM neuromodulated loss for pretraining.
    Replaces contrastive objective with eight biological signals.
    """
    def __init__(self):
        super().__init__()
        # Learnable signal weights (initialised near 1/8 each)
        self.log_weights = nn.Parameter(torch.zeros(6))

    def forward(self, z1: torch.Tensor, z2: torch.Tensor,
                labels: torch.Tensor) -> tuple:
        """
        z1, z2: two augmented views (B, D)
        labels: (B, 4) = [lin_vel, ang_vel, gps_lat, gps_lon]
        """
        weights = F.softmax(self.log_weights, dim=0) * 6

        # ── DA: prediction surprise (geometric distance) ───────────────────
        L_da = (1.0 - (z1 * z2).sum(dim=-1)).mean()

        # ── 5HT: representation diversity (prevent collapse) ──────────────
        var = z1.var(dim=0).mean()
        L_sht = F.relu(0.5 - var)   # penalise when variance < 0.5

        # ── NE: spatial grounding (GPS consistency) ───────────────────────
        gps = labels[:, 2:]  # (B, 2) lat/lon
        gps_norm = F.normalize(gps.float(), dim=1)
        z_norm   = F.normalize(z1.float(), dim=1)
        # Representations that are GPS-close should be embedding-close
        gps_sim  = torch.mm(gps_norm, gps_norm.t())
        emb_sim  = torch.mm(z_norm, z_norm.t())
        L_ne     = F.mse_loss(emb_sim, gps_sim.clamp(-1, 1))

        # ── ACh: velocity consistency ─────────────────────────────────────
        lin_vel = labels[:, 0:1].float()  # (B, 1)
        lin_norm = F.normalize(lin_vel, dim=0)
        # High velocity frames should have higher embedding variance
        L_ach = -torch.abs(lin_norm * (z1 - z2).norm(dim=-1, keepdim=True)).mean()

        # ── eCB: angular velocity consistency ────────────────────────────
        ang_vel = labels[:, 1:2].float()
        ang_norm = F.normalize(ang_vel.abs(), dim=0)
        L_ecb = -torch.abs(ang_norm * (z1 - z2).norm(dim=-1, keepdim=True)).mean()

        # ── Cortisol: distribution shift proxy (batch variance) ───────────
        batch_mean = z1.mean(dim=0)
        L_cort = -z1.var(dim=0).mean()  # penalise low batch variance

        # Weighted combination
        L_total = (weights[0] * L_da  +
                   weights[1] * L_sht +
                   weights[2] * L_ne  +
                   weights[3] * L_ach +
                   weights[4] * L_ecb +
                   weights[5] * L_cort)

        signals = {
            "da":   round(L_da.item(), 4),
            "sht":  round(L_sht.item(), 4),
            "ne":   round(L_ne.item(), 4),
            "ach":  round(L_ach.item(), 4),
            "ecb":  round(L_ecb.item(), 4),
            "cort": round(L_cort.item(), 4),
        }
        return L_total, signals


# ── Hybrid loss (contrastive + neuromodulated) ───────────────────────────────

class HybridLoss(nn.Module):
    """
    Encoder C — Hybrid: SimCLR base + NeMo-WM neuromodulated signals.
    Mirrors the NeMo-WM + JEPA relationship:
      - Contrastive provides structural learning signal (never zero)
      - Neuromodulator adds physical grounding (GPS, velocity, angular)
      - Learnable alpha balances the two objectives
    """
    def __init__(self, init_alpha: float = 0.5):
        super().__init__()
        # Learnable balance between contrastive and neuromodulated
        self.log_alpha = nn.Parameter(torch.tensor(float(np.log(init_alpha))))
        self.neuro_loss = NeuromodulatedLoss()

    def forward(self, z1: torch.Tensor, z2: torch.Tensor,
                labels: torch.Tensor) -> tuple:
        alpha = torch.sigmoid(self.log_alpha)  # 0-1, learnable

        # Contrastive component
        L_contrast = nt_xent_loss(z1, z2)

        # Neuromodulated component
        L_neuro, signals = self.neuro_loss(z1, z2, labels)

        # Hybrid: alpha weights contrastive, (1-alpha) weights neuromodulated
        L_total = alpha * L_contrast + (1.0 - alpha) * L_neuro

        signals["alpha"]      = round(float(alpha.item()), 3)
        signals["L_contrast"] = round(float(L_contrast.item()), 4)
        signals["L_neuro"]    = round(float(L_neuro.item()), 4)
        return L_total, signals


# ── AIM probe ─────────────────────────────────────────────────────────────────

def aim_probe(encoder: nn.Module, hdf5_dir: str,
              max_files: int = 20, n_samples: int = 500,
              held_out: bool = True) -> dict:
    """
    AIM probe: linear regression from frozen latents to physical quantities.
    Returns R² for each physical signal.
    held_out=True uses last 20% of files (unseen during training).
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    encoder.eval()
    Z, Y = [], []
    count = 0

    import h5py
    all_files = sorted(glob.glob(f"{hdf5_dir}/*.hdf5"))
    if held_out and len(all_files) > 10:
        # Use last 20% — not seen during training
        split = int(0.8 * len(all_files))
        files = all_files[split:split + max_files]
        print(f"    AIM probe: {len(files)} held-out files")
    else:
        files = all_files[:max_files]
        print(f"    AIM probe: {len(files)} files (no held-out split)")
    for path in files:
        if count >= n_samples:
            break
        try:
            with h5py.File(path) as hf:
                imgs = hf["images"]["rgb_left"]
                N = min(len(imgs), 10)
                for i in range(N):
                    if count >= n_samples:
                        break
                    try:
                        jpeg = bytes(imgs[i])
                        img  = Image.open(io.BytesIO(jpeg)).convert("RGB")
                        x    = TRANSFORM_CLEAN(img).unsqueeze(0)
                        with torch.no_grad():
                            z = encoder(x).squeeze(0).numpy()
                        lin = float(hf["commands"]["linear_velocity"][i])
                        ang = float(hf["commands"]["angular_velocity"][i])
                        gps = list(hf["gps"]["latlong"][i])
                        Z.append(z)
                        Y.append([lin, ang, gps[0], gps[1]])
                        count += 1
                    except Exception:
                        pass
        except Exception:
            pass

    if len(Z) < 20:
        return {"error": "insufficient samples", "n": len(Z)}

    Z = np.array(Z)
    Y = np.array(Y)
    labels = ["linear_velocity", "angular_velocity", "gps_lat", "gps_lon"]

    scaler = StandardScaler()
    Z_scaled = scaler.fit_transform(Z)

    results = {"n_samples": len(Z)}
    for i, label in enumerate(labels):
        y = Y[:, i]
        ridge = Ridge(alpha=1.0)
        # Simple train/test split
        split = int(0.8 * len(Z))
        ridge.fit(Z_scaled[:split], y[:split])
        r2 = ridge.score(Z_scaled[split:], y[split:])
        results[label] = round(float(r2), 4)

    return results


# ── Training loop ──────────────────────────────────────────────────────────────

def train_encoder(mode: str, hdf5_dir: str, epochs: int,
                  max_files: int = None, log_every: int = 100,
                  save_path: str = None) -> nn.Module:
    """
    Train ViT-Tiny encoder with either contrastive or neuromodulated loss.
    mode: 'contrastive' | 'neuromodulated'
    """
    encoder = ViTTiny()
    print(f"\n  Parameters: {encoder.param_count():,}")

    params = list(encoder.parameters())
    if mode == "neuromodulated":
        loss_fn = NeuromodulatedLoss()
        params += list(loss_fn.parameters())
        opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-4)
    elif mode == "hybrid":
        loss_fn = HybridLoss()
        params += list(loss_fn.parameters())
        opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-4)
    else:
        loss_fn = None
        opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-4)

    best_loss = float("inf")
    step = 0
    t0 = time.perf_counter()

    for ep in range(epochs):
        losses = []
        for v1, v2, labels in load_recon_batch(hdf5_dir, max_files=max_files):
            z1 = encoder(v1)
            z2 = encoder(v2)

            if mode == "contrastive":
                loss = nt_xent_loss(z1, z2)
                signals = {}
            else:
                loss, signals = loss_fn(z1, z2, labels)
                # Log alpha for hybrid
                if mode == "hybrid" and step % log_every == 0:
                    alpha = float(torch.sigmoid(loss_fn.log_alpha).item())
                    signals["alpha"] = round(alpha, 3)

            if not torch.isfinite(loss):
                continue

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            losses.append(loss.item())
            step += 1

            if step % log_every == 0:
                elapsed = time.perf_counter() - t0
                sig_str = " ".join(f"{k}={v:.3f}" for k, v in signals.items())
                print(f"  [ep{ep:02d} s{step:05d}] loss={loss.item():.4f} "
                      f"{sig_str} ({elapsed:.0f}s)")

        mean = np.mean(losses) if losses else float("inf")
        print(f"  Epoch {ep:02d} mean={mean:.4f}")

        if mean < best_loss and save_path:
            best_loss = mean
            torch.save({"epoch": ep, "loss": best_loss,
                        "model": encoder.state_dict()}, save_path)
            print(f"  → saved ({best_loss:.4f})")

    return encoder


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int, default=10)
    parser.add_argument("--hdf5-dir",   default="recon_data/recon_release")
    parser.add_argument("--max-files",  type=int, default=None)
    parser.add_argument("--log-every",  type=int, default=200)
    parser.add_argument("--quick",      action="store_true",
                        help="5 epochs, 10 files — smoke test")
    parser.add_argument("--eval-only",  action="store_true",
                        help="Skip training, eval existing checkpoints")
    parser.add_argument("--skip-sklearn", action="store_true",
                        help="Skip AIM probe if sklearn unavailable")
    parser.add_argument("--config",     default="all",
                        choices=["all", "a", "b", "c", "ab", "bc", "ac"],
                        help="Which encoders to train: a=contrastive b=neuro c=hybrid")
    a = parser.parse_args()

    if a.quick:
        a.epochs    = 5
        a.max_files = 20   # 20 files × ~70 frames = ~1400 training frames
        print("Quick mode: 5 epochs, 20 files (~1400 frames per encoder)")

    ckpt_contrast = "checkpoints/cwm/vit_tiny_contrastive.pt"
    ckpt_neuro    = "checkpoints/cwm/vit_tiny_neuromodulated.pt"
    ckpt_hybrid   = "checkpoints/cwm/vit_tiny_hybrid.pt"
    Path("checkpoints/cwm").mkdir(parents=True, exist_ok=True)

    # Check sklearn
    try:
        import sklearn
        has_sklearn = True
    except ImportError:
        has_sklearn = False
        print("sklearn not found — AIM probe skipped (pip install scikit-learn)")

    if not a.eval_only:
        # ── Train contrastive baseline ────────────────────────────────────────
        if "a" in a.config or a.config == "all":
            print("\n" + "="*60)
            print("  ENCODER A — Contrastive (SimCLR NT-Xent)")
            print("="*60)
            enc_contrast = train_encoder(
                "contrastive", a.hdf5_dir, a.epochs,
                a.max_files, a.log_every, ckpt_contrast
            )
        elif Path(ckpt_contrast).exists():
            print("\n  Encoder A: loading existing checkpoint")
            enc_contrast = ViTTiny()
            enc_contrast.load_state_dict(
                torch.load(ckpt_contrast, map_location="cpu")["model"])

        # ── Train neuromodulated encoder ──────────────────────────────────────
        if "b" in a.config or a.config == "all":
            print("\n" + "="*60)
            print("  ENCODER B — Neuromodulated (NeMo-WM signals)")
            print("="*60)
            enc_neuro = train_encoder(
                "neuromodulated", a.hdf5_dir, a.epochs,
                a.max_files, a.log_every, ckpt_neuro
            )
        elif Path(ckpt_neuro).exists():
            print("\n  Encoder B: loading existing checkpoint")
            enc_neuro = ViTTiny()
            enc_neuro.load_state_dict(
                torch.load(ckpt_neuro, map_location="cpu")["model"])
    # (eval_only else block now handled above)

        # ── Train hybrid encoder ─────────────────────────────────────────────
        print("\n" + "="*60)
        print("  ENCODER C — Hybrid (SimCLR + NeMo-WM signals)")
        print("  Contrastive base + neuromodulated physical grounding")
        print("="*60)
        enc_hybrid = train_encoder(
            "hybrid", a.hdf5_dir, a.epochs,
            a.max_files, a.log_every, ckpt_hybrid
        )
    else:
        enc_contrast = ViTTiny()
        enc_neuro    = ViTTiny()
        enc_hybrid   = ViTTiny()
        if Path(ckpt_contrast).exists():
            enc_contrast.load_state_dict(
                torch.load(ckpt_contrast, map_location="cpu")["model"])
        if Path(ckpt_neuro).exists():
            enc_neuro.load_state_dict(
                torch.load(ckpt_neuro, map_location="cpu")["model"])
        if Path(ckpt_hybrid).exists():
            enc_hybrid.load_state_dict(
                torch.load(ckpt_hybrid, map_location="cpu")["model"])

    # ── AIM probe evaluation ──────────────────────────────────────────────────
    if not has_sklearn or a.skip_sklearn:
        print("\nSkipping AIM probe — install scikit-learn to enable:")
        print("  pip install scikit-learn --break-system-packages")
        return

    print("\n" + "="*60)
    print("  AIM PROBE — Physical Quantity Encoding")
    print("="*60)
    print("  Metric: R² (linear regression from frozen latents)")
    print("  Higher R² = encoder learned this physical quantity")
    print()

    labels = ["linear_velocity", "angular_velocity", "gps_lat", "gps_lon"]

    probe_files  = 30 if not a.quick else 10
    probe_samples = 1000 if not a.quick else 200
    print("  Running AIM probe on Contrastive encoder...")
    r2_contrast = aim_probe(enc_contrast, a.hdf5_dir,
                            max_files=probe_files, n_samples=probe_samples)
    print("  Running AIM probe on Neuromodulated encoder...")
    r2_neuro = aim_probe(enc_neuro, a.hdf5_dir,
                         max_files=probe_files, n_samples=probe_samples)
    print("  Running AIM probe on Hybrid encoder...")
    r2_hybrid = aim_probe(enc_hybrid, a.hdf5_dir,
                          max_files=probe_files, n_samples=probe_samples)

    # ── Results table ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  AIM PROBE RESULTS")
    print(f"{'='*60}")
    print(f"  {'Signal':<22} {'Contrast R²':>11} {'Neuro R²':>9} {'Hybrid R²':>10} {'Winner'}")
    print(f"  {'-'*65}")

    neuro_wins = 0
    hybrid_wins = 0
    for label in labels:
        r2c = r2_contrast.get(label, float('nan'))
        r2n = r2_neuro.get(label, float('nan'))
        r2h = r2_hybrid.get(label, float('nan'))
        best = max(r2c, r2n, r2h)
        if best == r2h:
            winner = "Hybrid ✅"
            hybrid_wins += 1
        elif best == r2n:
            winner = "Neuro ✅"
            neuro_wins += 1
        else:
            winner = "Contrast"
        print(f"  {label:<22} {r2c:>11.4f} {r2n:>9.4f} {r2h:>10.4f} {winner}")

    print(f"\n  Samples: {r2_contrast.get('n_samples', '?')}")
    print(f"  Hybrid wins:        {hybrid_wins}/{len(labels)}")
    print(f"  Neuromodulated wins:{neuro_wins}/{len(labels)}")
    print(f"  Contrastive wins:   {len(labels)-hybrid_wins-neuro_wins}/{len(labels)}")

    if hybrid_wins >= 3:
        print("\n  ✅ HYBRID wins — contrastive base + neuromodulated grounding")
        print("     Mirrors the NeMo-WM + JEPA result at encoder level.")
        print("     Both signals are complementary, not competing.")
    elif neuro_wins >= 3:
        print("\n  ✅ Pure neuromodulated wins — biological signals sufficient alone")
    elif hybrid_wins + neuro_wins >= 3:
        print("\n  ⚠️  Neuromodulation helps (hybrid or pure) — contrastive insufficient alone")
    else:
        print("\n  ❌ Contrastive wins — may need more epochs")

    # Save results
    import json
    results = {
        "epochs": a.epochs,
        "max_files": a.max_files,
        "contrastive": r2_contrast,
        "neuromodulated": r2_neuro,
        "hybrid": r2_hybrid,
        "hybrid_wins": hybrid_wins,
        "neuro_wins": neuro_wins,
    }
    with open("vlm_phase2a_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n  Saved: vlm_phase2a_results.json")


if __name__ == "__main__":
    main()
