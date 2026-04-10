"""
train_proprio_6c.py  --  NeMo-WM Sprint 6c
==========================================
Temporal context window for proprioceptive encoding.

Sprint 6/6b finding: single-frame signals (vel, ang, heading, contact)
have a hard accuracy ceiling at top1_acc ~0.718 regardless of model size
or temperature. The discriminative limit is in the data, not the model.

Sprint 6c fix: stack a window of k_ctx=4 consecutive frames into a single
(k_ctx * d_per_frame,) = (4*8,) = 32-D input vector with sinusoidal
positional encoding injected per frame. The trajectory pattern over 1
second (4 frames at 4Hz) is far more discriminative than any single
timestep -- the robot's turning sequence distinguishes t=5 from t=60.

Architecture (ProprioEncoderTemporal):
    Input:  (B, k_ctx, d_per_frame) -- k_ctx=4, d_per_frame=8
    PosEnc: sinusoidal, added to each frame embedding
    Trunk:  frame MLP -> (B, k_ctx, d_hidden)
            temporal pooling (attention-weighted mean)
            output MLP -> (B, d_model)
    Output: (B, d_model=64) unit-normalised

Contrastive setup:
    Anchor:   frames [t-3, t-2, t-1, t]       (window ending at t)
    Positive: frames [t+1, t+2, t+3, t+k]     (window ending at t+k, k<=4)
    Negative: frames [t+k_hard-3, ..., t+k_hard] (window ending >= t+32)

Sprint 6c target:
    No-VLM AUROC >= 0.862 (matching VLM-only) = full dissociation
    Full AUROC > 0.892 (Sprint 6 best)

Usage:
    python train_proprio_6c.py ^
        --hdf5-dir recon_data/recon_release ^
        --out-ckpt checkpoints/cwm/proprio_6c_best.pt ^
        --epochs 40
"""

import argparse, random, math, time
from pathlib import Path
import h5py, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

# Cortisol domain-adaptive signal (Sprint 9)
try:
    from cortisol_domain_adaptive import CortisolSignalAdaptive, CortisolConfig
    _CORTISOL_AVAILABLE = True
except ImportError:
    _CORTISOL_AVAILABLE = False


# ===========================================================================
# Constants
# ===========================================================================

K_CTX       = 4    # frames per context window
D_PER_FRAME = 8    # signals per frame: vel, ang, gps_n, gps_e, contact, sin_h, cos_h, delta_h


# ===========================================================================
# Positional encoding
# ===========================================================================

def make_sinusoidal_pe(k_ctx: int, d_model: int) -> torch.Tensor:
    """
    Sinusoidal positional encoding for k_ctx positions.
    Returns (k_ctx, d_model) tensor.
    """
    pe  = torch.zeros(k_ctx, d_model)
    pos = torch.arange(k_ctx).unsqueeze(1).float()
    div = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)[:, :pe[:, 1::2].shape[1]]
    return pe   # (k_ctx, d_model)


# ===========================================================================
# Temporal proprioceptive encoder
# ===========================================================================

class ProprioEncoderTemporal(nn.Module):
    """
    Sprint 6c: encodes a window of k_ctx proprioceptive frames.

    Frame embedding:
        Linear(d_per_frame -> d_hidden) + LayerNorm + GELU
        + sinusoidal positional encoding

    Temporal pooling:
        Learned attention weights (1D softmax over k_ctx positions)
        -> weighted mean -> (B, d_hidden)

    Output MLP:
        Linear(d_hidden -> d_model) -> unit-normalised (B, d_model)
    """

    def __init__(
        self,
        k_ctx:       int = K_CTX,
        d_per_frame: int = D_PER_FRAME,
        d_hidden:    int = 128,
        d_model:     int = 64,
    ):
        super().__init__()
        self.k_ctx       = k_ctx
        self.d_per_frame = d_per_frame
        self.d_hidden    = d_hidden
        self.d_model     = d_model

        # Frame embedding: each of the k_ctx frames -> d_hidden
        self.frame_embed = nn.Sequential(
            nn.Linear(d_per_frame, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.LayerNorm(d_hidden),
        )

        # Sinusoidal PE (fixed, registered as buffer)
        pe = make_sinusoidal_pe(k_ctx, d_hidden)  # (k_ctx, d_hidden)
        self.register_buffer("pe", pe)

        # Temporal attention pooling: scalar score per position
        self.attn_score = nn.Linear(d_hidden, 1)

        # Output projection
        self.out_proj = nn.Sequential(
            nn.GELU(),
            nn.Linear(d_hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, k_ctx, d_per_frame) -- window of k_ctx frames
        returns: (B, d_model) unit-normalised
        """
        B = x.shape[0]

        # Embed each frame: (B, k_ctx, d_hidden)
        h = self.frame_embed(x)

        # Add positional encoding
        h = h + self.pe.unsqueeze(0)          # (B, k_ctx, d_hidden)

        # Attention pooling
        scores = self.attn_score(h).squeeze(-1)  # (B, k_ctx)
        weights = torch.softmax(scores, dim=-1)   # (B, k_ctx)
        pooled  = (h * weights.unsqueeze(-1)).sum(dim=1)  # (B, d_hidden)

        # Output
        out = self.out_proj(pooled)            # (B, d_model)
        return F.normalize(out, dim=-1)

    def save(self, path, epoch=None, loss=None, acc=None):
        torch.save({
            "model":       self.state_dict(),
            "k_ctx":       self.k_ctx,
            "d_per_frame": self.d_per_frame,
            "d_hidden":    self.d_hidden,
            "d_model":     self.d_model,
            "epoch":       epoch,
            "loss":        loss,
            "top1_acc":    acc,
            "arch":        "temporal",
        }, path)
        print(f"  → Saved (best top1_acc={acc:.4f}): {path}")

    @classmethod
    def load(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        enc  = cls(
            k_ctx       = ckpt.get("k_ctx",       K_CTX),
            d_per_frame = ckpt.get("d_per_frame", D_PER_FRAME),
            d_hidden    = ckpt.get("d_hidden",    128),
            d_model     = ckpt.get("d_model",     64),
        )
        enc.load_state_dict(ckpt["model"])
        acc = ckpt.get("top1_acc", "?")
        astr = f"{acc:.4f}" if isinstance(acc, float) else str(acc)
        print(f"ProprioEncoderTemporal loaded: {path} "
              f"(epoch={ckpt.get('epoch','?')}, top1_acc={astr})")
        return enc


# ===========================================================================
# Dataset — extracts context windows
# ===========================================================================

class ProprioPairDatasetTemporal:
    """
    Samples (anchor_window, positive_window, [negative_windows]) from RECON.

    Anchor window:   frames [t-k_ctx+1, ..., t]   (most recent = t)
    Positive window: frames [t+1, ..., t+k]        (k <= k_max, shifted window)
    Negative window: frames far ahead in same file  (gap >= k_hard_min)

    All windows are k_ctx=4 frames of 8-D signals. No GPS.
    """

    def __init__(
        self,
        hdf5_dir:   str,
        k_ctx:      int = K_CTX,
        k_max:      int = 4,
        k_hard_min: int = 32,
        n_neg:      int = 7,
        max_files:  int = None,
        seed:       int = 42,
    ):
        random.seed(seed); np.random.seed(seed)
        files = sorted(Path(hdf5_dir).glob("jackal_2019-*.hdf5"))
        if not files:   # fallback: accept any .hdf5 (synthetic data, custom datasets)
            files = sorted(Path(hdf5_dir).glob("*.hdf5"))
        if max_files:
            files = files[:max_files]
        self.files      = [str(f) for f in files]
        self.k_ctx      = k_ctx
        self.k_max      = k_max
        self.k_hard_min = k_hard_min
        self.n_neg      = n_neg
        assert len(self.files) >= 1
        print(f"ProprioPairDatasetTemporal: {len(self.files)} files | "
              f"k_ctx={k_ctx}, k_max={k_max}, k_hard_min={k_hard_min}, n_neg={n_neg}")

    def _extract_frame(self, hf, t: int) -> np.ndarray:
        """Extract 8-D signal vector for frame t. No GPS."""
        vel_val = float(hf["commands"]["linear_velocity"][t])
        ang_val = float(hf["commands"]["angular_velocity"][t])

        ang_arr       = hf["commands"]["angular_velocity"][:t+1]
        heading_angle = float(np.sum(ang_arr)) / 4.0
        sin_h = math.sin(heading_angle)
        cos_h = math.cos(heading_angle)

        if t > 0:
            prev_ang  = hf["commands"]["angular_velocity"][:t]
            prev_head = float(np.sum(prev_ang)) / 4.0
            delta_h   = heading_angle - prev_head
        else:
            delta_h = 0.0

        contact = 1.0 if abs(vel_val) > 0.3 else 0.0
        return np.array(
            [vel_val, ang_val, 0.0, 0.0, contact, sin_h, cos_h, delta_h],
            dtype=np.float32
        )

    def _extract_window(self, hf, t_end: int) -> torch.Tensor:
        """
        Extract k_ctx-frame window ending at t_end.
        Returns (k_ctx, D_PER_FRAME) tensor, nan-cleaned.
        """
        frames = []
        for i in range(self.k_ctx):
            t = max(0, t_end - (self.k_ctx - 1 - i))
            frames.append(self._extract_frame(hf, t))
        arr = np.stack(frames, axis=0)                  # (k_ctx, D_PER_FRAME)
        t   = torch.from_numpy(arr)
        return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

    def sample_batch(self, batch_size: int = 64):
        anchors, positives, negatives_list = [], [], []
        attempts = 0
        min_t = self.k_ctx - 1          # need enough frames for window
        min_L = self.k_hard_min + self.k_max + self.k_ctx + 2

        while len(anchors) < batch_size and attempts < batch_size * 20:
            attempts += 1
            filepath = random.choice(self.files)
            try:
                with h5py.File(filepath, "r") as hf:
                    # Support both RECON (has images) and synthetic (commands only)
                    if "images" in hf and "rgb_left" in hf["images"]:
                        L = hf["images"]["rgb_left"].shape[0]
                    else:
                        L = hf["commands"]["linear_velocity"].shape[0]
                    if L < min_L:
                        continue

                    # Anchor: window ending at t_anchor
                    t_anchor = random.randint(min_t, L - self.k_max - 2)

                    # Positive: window ending at t_anchor + k
                    k = random.randint(1, self.k_max)
                    t_pos = t_anchor + k

                    # Negatives: windows ending >= t_anchor + k_hard_min
                    neg_ends = [
                        t for t in range(min_t, L)
                        if t - t_anchor >= self.k_hard_min
                    ]
                    if len(neg_ends) < self.n_neg:
                        continue
                    neg_ts = random.sample(neg_ends, self.n_neg)

                    w_anchor = self._extract_window(hf, t_anchor)
                    w_pos    = self._extract_window(hf, t_pos)
                    w_negs   = torch.stack([
                        self._extract_window(hf, t_n) for t_n in neg_ts
                    ])                                  # (n_neg, k_ctx, D)

                    anchors.append(w_anchor)
                    positives.append(w_pos)
                    negatives_list.append(w_negs)

            except Exception:
                continue

        if not anchors:
            return None

        return {
            "anchor":    torch.stack(anchors),          # (B, k_ctx, D)
            "positive":  torch.stack(positives),        # (B, k_ctx, D)
            "negatives": torch.stack(negatives_list),   # (B, n_neg, k_ctx, D)
        }


# ===========================================================================
# InfoNCE loss (unchanged from Sprint 6b)
# ===========================================================================

def infonce_loss(z_anchor, z_positive, z_negatives, temperature=0.05):
    B, D  = z_anchor.shape
    n_neg = z_negatives.shape[1]

    sim_pos = F.cosine_similarity(z_anchor, z_positive, dim=-1) / temperature

    z_anc_rep  = z_anchor.unsqueeze(1).expand(-1, n_neg, -1).reshape(B * n_neg, D)
    z_neg_flat = z_negatives.reshape(B * n_neg, D)
    sim_neg    = F.cosine_similarity(z_anc_rep, z_neg_flat, dim=-1
                                     ).view(B, n_neg) / temperature

    logits = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1)
    labels = torch.zeros(B, dtype=torch.long, device=z_anchor.device)
    loss   = F.cross_entropy(logits, labels)
    top1   = (logits.argmax(dim=1) == 0).float().mean().item()
    return loss, top1


# ===========================================================================
# Training loop
# ===========================================================================

def train(
    hdf5_dir,
    out_ckpt,
    epochs             = 40,
    steps_epoch        = 500,
    batch_size         = 64,
    lr                 = 3e-4,
    temperature        = 0.05,
    n_neg              = 7,
    k_ctx              = K_CTX,
    k_max              = 4,
    k_hard_min         = 32,
    d_hidden           = 128,
    d_model            = 64,
    device_str         = "cpu",
    use_cortisol         = True,
    cortisol_sensitivity = 0.10,
    init_ckpt            = None,   # optional checkpoint to init from (sim-to-real)
):
    dev = torch.device(device_str)
    Path(out_ckpt).parent.mkdir(parents=True, exist_ok=True)

    # ── Cortisol domain-adaptive signal ──────────────────────────────────────
    cort_signal = None
    if use_cortisol and _CORTISOL_AVAILABLE:
        cort_cfg    = CortisolConfig(
            sensitivity  = cortisol_sensitivity,
            ne_scale_high  = 1.22,
            ecb_scale_high = 0.82,
            log_transitions = True,
        )
        cort_signal = CortisolSignalAdaptive(cfg=cort_cfg)
        print(f"  Cortisol: ACTIVE (sensitivity={cortisol_sensitivity})")
    elif use_cortisol:
        print(f"  Cortisol: DISABLED (cortisol_domain_adaptive.py not found)")

    enc = ProprioEncoderTemporal(
        k_ctx=k_ctx, d_per_frame=D_PER_FRAME,
        d_hidden=d_hidden, d_model=d_model,
    ).to(dev)
    n_params = sum(p.numel() for p in enc.parameters())

    # Optional: initialise from pre-trained checkpoint (sim-to-real fine-tune)
    if init_ckpt and Path(init_ckpt).exists():
        pd_init = torch.load(init_ckpt, map_location=dev, weights_only=False)
        enc.load_state_dict(pd_init["model"])
        prev_acc = pd_init.get("top1_acc", "?")
        print(f"  Init from: {init_ckpt} (top1_acc={prev_acc})")
    elif init_ckpt:
        print(f"  Warning: --init-ckpt {init_ckpt} not found — training from scratch")

    print(f"\nSprint 6c — Temporal ProprioceptiveEncoder Training")
    print(f"  Architecture: {k_ctx}-frame window, sinusoidal PE, attn pooling")
    print(f"  d_per_frame={D_PER_FRAME}, d_hidden={d_hidden}, d_model={d_model}")
    print(f"  Parameters: {n_params:,}  (Sprint 6b: 26,432)")
    print(f"  Epochs: {epochs}  |  Steps/epoch: {steps_epoch}")
    print(f"  LR: {lr}  |  Temp: {temperature}")
    print(f"  k_ctx: {k_ctx}  |  k_max: {k_max}  |  k_hard_min: {k_hard_min}")
    print(f"  No GPS (vel+ang+heading+contact+delta_h only)")
    print(f"  Output: {out_ckpt}\n")

    ds    = ProprioPairDatasetTemporal(
        hdf5_dir, k_ctx=k_ctx, k_max=k_max,
        k_hard_min=k_hard_min, n_neg=n_neg,
    )
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

            # Encode anchor and positive windows
            z_a = enc(batch["anchor"].to(dev))       # (B, d_model)
            z_p = enc(batch["positive"].to(dev))     # (B, d_model)

            # Encode negative windows: (B, n_neg, k_ctx, D) -> (B, n_neg, d_model)
            neg = batch["negatives"].to(dev)         # (B, n_neg, k_ctx, D)
            B, n_neg_b, kc, df = neg.shape
            z_n = enc(neg.view(B * n_neg_b, kc, df)
                      ).view(B, n_neg_b, d_model)    # (B, n_neg, d_model)

            loss, acc = infonce_loss(z_a, z_p, z_n, temperature)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), max_norm=1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
            accs.append(acc)

            # Cortisol step — detect domain shifts, modulate NE/eCB
            if cort_signal is not None:
                cort_signal.step(loss=loss.item())

        mean_loss = float(np.mean(losses)) if losses else float("inf")
        mean_acc  = float(np.mean(accs))   if accs   else 0.0
        elapsed   = time.perf_counter() - t0
        cur_lr    = sched.get_last_lr()[0]

        print(f"  {epoch:>4}  {mean_loss:>8.4f}  {mean_acc:>10.4f}  "
              f"{cur_lr:>12.2e}  {elapsed:>9.1f}s")

        # Cortisol epoch summary
        if cort_signal is not None:
            cs = cort_signal.summary()
            if cs.get("domain_resets", 0) > 0 or cs.get("cort_max", 0) > 0.15:
                print(f"         cort_max={cs['cort_max']:.3f}  "
                      f"ne={cs['ne_scale_mean']:.3f}  ecb={cs['ecb_scale_mean']:.3f}  "
                      f"regime={cs['regime']}  resets={cs['domain_resets']}")
            cort_signal.reset_epoch()

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
    print(f"\nSprint 6c target: No-VLM AUROC >= 0.862 (full dissociation)")
    return best_acc


# ===========================================================================
# Self-test
# ===========================================================================

def self_test():
    print("Sprint 6c self-test...")
    enc = ProprioEncoderTemporal(k_ctx=4, d_per_frame=8, d_hidden=64, d_model=32)
    enc.eval()

    B = 8
    x = torch.randn(B, 4, 8)
    with torch.no_grad():
        z = enc(x)

    assert z.shape == (B, 32), f"Wrong shape: {z.shape}"
    norms = z.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(B), atol=1e-5), \
        f"Not unit-normalised: {norms}"
    print(f"  Forward pass: {z.shape} -- unit-normalised OK")

    # Test InfoNCE
    z_a = F.normalize(torch.randn(B, 32), dim=-1)
    z_p = F.normalize(torch.randn(B, 32), dim=-1)
    z_n = F.normalize(torch.randn(B, 7, 32), dim=-1)
    loss, acc = infonce_loss(z_a, z_p, z_n, temperature=0.05)
    assert loss.item() > 0
    print(f"  InfoNCE loss={loss.item():.4f}, top1_acc={acc:.4f}")

    # Test positional encoding shape
    pe = make_sinusoidal_pe(4, 64)
    assert pe.shape == (4, 64)
    assert not pe.isnan().any()
    print(f"  Positional encoding: {pe.shape} -- no NaN OK")

    n_params = sum(p.numel() for p in enc.parameters())
    print(f"  Parameters: {n_params:,}")
    print("  All assertions passed.\n")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Sprint 6c temporal proprio encoder")
    p.add_argument("--hdf5-dir",    default="recon_data/recon_release")
    p.add_argument("--out-ckpt",    default="checkpoints/cwm/proprio_6c_best.pt")
    p.add_argument("--epochs",      type=int,   default=40)
    p.add_argument("--steps",       type=int,   default=500)
    p.add_argument("--batch-size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--n-neg",       type=int,   default=7)
    p.add_argument("--k-ctx",       type=int,   default=4,
                   help="Frames per context window (default 4 = 1s at 4Hz)")
    p.add_argument("--k-max",       type=int,   default=4)
    p.add_argument("--k-hard-min",  type=int,   default=32)
    p.add_argument("--d-hidden",    type=int,   default=128)
    p.add_argument("--d-model",     type=int,   default=64)
    p.add_argument("--device",      default="cpu")
    p.add_argument("--test",        action="store_true")
    p.add_argument("--no-cortisol", action="store_true",
                   help="Disable cortisol domain-adaptive signal")
    p.add_argument("--init-ckpt",       default=None,
                   help="Init encoder weights from checkpoint (sim-to-real fine-tune)")
    p.add_argument("--cortisol-sensitivity", type=float, default=0.10,
                   help="Cortisol sensitivity (default 0.10, was 0.05 in v16.12)")
    args = p.parse_args()

    if args.test:
        self_test()
    else:
        train(
            hdf5_dir             = args.hdf5_dir,
            out_ckpt             = args.out_ckpt,
            epochs               = args.epochs,
            steps_epoch          = args.steps,
            batch_size           = args.batch_size,
            lr                   = args.lr,
            temperature          = args.temperature,
            n_neg                = args.n_neg,
            k_ctx                = args.k_ctx,
            k_max                = args.k_max,
            k_hard_min           = args.k_hard_min,
            d_hidden             = args.d_hidden,
            use_cortisol         = not args.no_cortisol,
            cortisol_sensitivity = args.cortisol_sensitivity,
            init_ckpt            = args.init_ckpt,
            d_model     = args.d_model,
            device_str  = args.device,
        )
