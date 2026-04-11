"""
train_cwm_multidomain.py  —  CORTEX CWM Sprint 3
=================================================
Multi-domain fine-tuning of the CWM predictor.

Extends Sprint 1 (train_cwm.py) to train across all domains simultaneously
using the domain_loaders.py WeightedRandomSampler.

Strategy:
  - Load Sprint 1 CWM checkpoint (trained on RECON)
  - Fine-tune predictor + contact head across all domains
  - StudentEncoder stays FROZEN (XINT8 NPU path unchanged)
  - Domain embedding in MoEJEPAPredictor gates expert routing per domain
  - MoE expert specialisation validated per domain after training

Expert routing expected after Sprint 3:
  RECON/TwoRoom  → Expert 1 (smooth navigation)
  OGBench/PushT  → Expert 0 (contact rigid-body)
  SMAP/Cardiac   → Expert 2 (temporal drift)
  Hexapod/Quad   → Expert 3 (periodic locomotion, future)

Usage:
    # Start from Sprint 1 checkpoint
    python train_cwm_multidomain.py \
        --cwm-ckpt    checkpoints\cwm\cwm_best.pt \
        --recon-dir   recon_data\recon_release \
        --ogbench-dir ogbench_data \
        --smap-dir    smap \
        --mvtec-dir   mvtec \
        --epochs      20
"""

import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# DreamerV3 tricks active: symlog, free_bits, AGC, unimix (from cwm_neuro_reward.py)
from neuromodulator import NeuromodulatorState
from cwm_neuro_reward import NeuromodulatedCWMLoss, RegimeGatedTrainer
from cwm_moe_jepa import MoEJEPAPredictor, jepa_moe_loss, log_expert_utilisation
from train_cwm import CortexWorldModel, MAX_ACTION_DIM, DOMAIN_IDS
from domain_loaders import build_multi_domain_loader


# ═══════════════════════════════════════════════════════════════════════════
# Expert specialisation tracker
# ═══════════════════════════════════════════════════════════════════════════

class ExpertSpecialisationTracker:
    """
    Tracks which experts each domain routes to over training.
    Logs per-domain expert usage every N steps.
    Alerts if load balance collapses (one expert > 70%).
    """

    def __init__(self, n_experts: int = 4, domains: List[str] = None):
        self.n_experts = n_experts
        self.domains   = domains or list(DOMAIN_IDS.keys())
        # {domain: {expert_id: count}}
        self.counts: Dict[str, Dict[int, int]] = {
            d: {e: 0 for e in range(n_experts)} for d in self.domains
        }
        self.steps = 0

    def update(self, domain: str, expert_probs: torch.Tensor):
        """
        expert_probs: (B*K, n_experts) router probabilities
        """
        if domain not in self.counts:
            self.counts[domain] = {e: 0 for e in range(self.n_experts)}
        top1 = expert_probs.argmax(dim=-1).cpu()
        for e in range(self.n_experts):
            self.counts[domain][e] += (top1 == e).sum().item()
        self.steps += 1

    def report(self, epoch: int):
        print(f"\n[Epoch {epoch}] Expert specialisation by domain:")
        print(f"  {'Domain':12s}  " +
              "  ".join(f"Expert{e}" for e in range(self.n_experts)))
        print(f"  {'─'*12}  " + "  ".join(["───────"] * self.n_experts))

        for domain in self.domains:
            counts = self.counts[domain]
            total  = sum(counts.values()) + 1e-9
            pcts   = [f"{100*counts[e]/total:6.1f}%" for e in range(self.n_experts)]
            dom_max = max(counts[e]/total for e in range(self.n_experts))
            flag = " ⚠️  collapse" if dom_max > 0.70 else ""
            print(f"  {domain:12s}  " + "  ".join(pcts) + flag)

        # Check for load collapse across all domains
        all_counts = {e: sum(self.counts[d][e]
                             for d in self.domains)
                      for e in range(self.n_experts)}
        total      = sum(all_counts.values()) + 1e-9
        global_max = max(all_counts[e]/total for e in range(self.n_experts))
        if global_max > 0.60:
            print(f"\n  ⚠️  Global expert collapse detected ({global_max:.1%})")
            print(f"     → Increase lambda_aux in SparseMoEFFN (currently 0.01)")
            print(f"     → Try lambda_aux=0.05 or 0.10")

        # Reset counts
        self.counts = {d: {e: 0 for e in range(self.n_experts)}
                       for d in self.domains}


# ═══════════════════════════════════════════════════════════════════════════
# Domain-aware loss weighting
# ═══════════════════════════════════════════════════════════════════════════

# Per-domain loss weights — contact domains get higher contact loss weight
DOMAIN_LOSS_WEIGHTS = {
    "recon":   {"L_jepa": 1.0, "L_contact": 0.5, "L_gps": 1.0},
    "ogbench": {"L_jepa": 1.0, "L_contact": 2.0, "L_gps": 0.0},
    "pusht":   {"L_jepa": 1.0, "L_contact": 2.0, "L_gps": 0.0},
    "tworoom": {"L_jepa": 1.0, "L_contact": 0.3, "L_gps": 0.5},
    "smap":    {"L_jepa": 0.8, "L_contact": 0.0, "L_gps": 0.0},
    "mvtec":   {"L_jepa": 0.8, "L_contact": 0.0, "L_gps": 0.0},
    "cardiac": {"L_jepa": 0.8, "L_contact": 0.0, "L_gps": 0.0},
}


def get_domain_loss_weights(domain: str) -> dict:
    return DOMAIN_LOSS_WEIGHTS.get(domain, {
        "L_jepa": 1.0, "L_contact": 1.0, "L_gps": 0.5
    })


# ═══════════════════════════════════════════════════════════════════════════
# Multi-domain training loop
# ═══════════════════════════════════════════════════════════════════════════

def train_multidomain(
    cwm_ckpt:      str,
    domain_roots:  Dict[str, str],
    n_epochs:      int   = 20,
    batch_size:    int   = 16,
    base_lr:       float = 1e-4,    # lower than Sprint 1 — fine-tuning
    save_dir:      str   = r"checkpoints\cwm",
    log_every:     int   = 50,
    report_every:  int   = 5,       # epochs between specialisation reports
    max_files:     Optional[Dict[str, int]] = None,
    device_str:    str   = "cpu",
):
    device = torch.device(device_str)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # ── Load Sprint 1 checkpoint ───────────────────────────────────────────
    cwm = CortexWorldModel(d_model=128, K=16).to(device)
    if Path(cwm_ckpt).exists():
        ckpt = torch.load(cwm_ckpt, map_location=device, weights_only=False)
        cwm.load_state_dict(ckpt["model"])
        sprint1_loss = ckpt.get("loss", "?")
        print(f"Sprint 1 checkpoint loaded: {cwm_ckpt}")
        print(f"  Sprint 1 best loss: {sprint1_loss}")
    else:
        print(f"WARNING: {cwm_ckpt} not found — training from scratch")

    # StudentEncoder stays frozen — never updated after Sprint 1
    # In production: encoder runs on NPU, not in this training graph
    # Real StudentEncoder -- frozen, XINT8 NPU path unchanged
    from train_mvtec import StudentEncoder as _SE
    student_enc = _SE().to(device)
    _enc_path = Path("checkpoints/dinov2_student/student_best.pt")
    if _enc_path.exists():
        _sd = torch.load(_enc_path, map_location="cpu", weights_only=False)
        _sd = _sd.get("model", _sd.get("state_dict", _sd))
        student_enc.load_state_dict(_sd, strict=False)
        print(f"StudentEncoder loaded: {_enc_path}")
    else:
        print("WARNING: student_best.pt not found -- random encoder weights")
    student_enc.eval()
    for _p in student_enc.parameters():
        _p.requires_grad_(False)

    # ── Multi-domain loader ────────────────────────────────────────────────
    active_domains = [d for d, p in domain_roots.items()
                      if p and Path(p).exists()]
    print(f"\nActive domains: {active_domains}")

    loader = build_multi_domain_loader(
        domains      = active_domains,
        domain_roots = domain_roots,
        batch_size   = batch_size,
        max_files    = max_files,
        balance      = True,
    )

    # ── Neuromodulators & loss ─────────────────────────────────────────────
    neuro       = NeuromodulatorState(session_start=time.time(),
                                       ado_saturate_hours=float(n_epochs))
    loss_fn     = NeuromodulatedCWMLoss(
        lambda_gaussian=0.10, lambda_gps=0.05,
        lambda_contact=0.01,  lambda_skill=0.05, lambda_curv=0.02,
    )
    tracker     = ExpertSpecialisationTracker(
        n_experts=4, domains=active_domains
    )

    # ── Optimiser — fine-tune predictor + contact head only ───────────────
    # Encoder MoE and ParticleEncoder inherit RECON representations — freeze
    trainable_modules = [
        cwm.predictor,
        cwm.contact_head,
        cwm.thick_gru,
        cwm.gps_head,
    ]
    trainable_params = []
    for m in trainable_modules:
        trainable_params.extend(m.parameters())

    optimizer = torch.optim.AdamW(
        trainable_params, lr=base_lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr      = base_lr * 3,
        total_steps = n_epochs * len(loader),
        pct_start   = 0.1,
    )

    best_loss    = float("inf")
    global_step  = 0
    domain_stats: Dict[str, list] = {d: [] for d in active_domains}

    for epoch in range(n_epochs):
        cwm.predictor.train()
        cwm.contact_head.train()
        cwm.thick_gru.train()
        cwm.gps_head.train()

        # Freeze encoder-side
        cwm.encoder_moe.eval()
        cwm.particle_enc.eval()

        epoch_losses: Dict[str, list] = {d: [] for d in active_domains}
        epoch_losses["all"] = []

        for batch in loader:
            frame_t   = batch["frame_t"].to(device)
            frame_t1  = batch["frame_t1"].to(device)
            action    = batch["action"].to(device)
            gps       = batch["gps"].to(device)
            domain_id = batch["domain_id"].to(device)

            # domain is a list of strings (one per sample in batch)
            # Use the first element for domain-specific weighting
            domain    = batch["domain"][0] if isinstance(batch["domain"], list) \
                        else batch["domain"]
            B         = frame_t.shape[0]

            # ── Context path ───────────────────────────────────────────────
            z_t   = student_enc(frame_t)
            z_t1c = student_enc(frame_t1)

            with torch.no_grad():
                cwm.encoder_moe.eval()
                cwm.particle_enc.eval()
            particles_t,  pos_t,  _, _ = cwm.encode(z_t)
            particles_t1, pos_t1, _, _ = cwm.encode(z_t1c)

            # ── Target path (stop-gradient) ────────────────────────────────
            with torch.no_grad():
                z_t1_tgt        = student_enc(frame_t1)
                p_t1_tgt, _, _, _ = cwm.encode(z_t1_tgt)
                z_target         = p_t1_tgt.detach()

            # ── Neuromodulators ────────────────────────────────────────────
            signals = neuro.update(
                z_pred           = particles_t.mean(dim=1),
                z_actual         = z_target.mean(dim=1),
                rho              = 0.5,
                action_magnitude = action.norm(dim=-1).mean().item(),
            )
            loss_fn.update_from_neuro(signals)
            config  = RegimeGatedTrainer.get_training_config(signals)

            # Skip domain if regime=WAIT (neuro says: observe, don't update)
            if signals["regime"] == "WAIT":
                continue

            # ── Forward ───────────────────────────────────────────────────
            context_h = cwm.thick_gru.init_context(B, device)
            out = cwm.predict(
                particles  = particles_t,
                action     = action,
                context_h  = context_h,
                positions  = pos_t,
                domain_id  = domain_id,
                regime     = signals["regime"],
                ach        = signals["ach"],
            )
            z_pred      = out["z_pred"]
            signed_dist = out["signed_dist"]
            gps_pred    = out["gps_pred"]
            moe_aux     = out["moe_aux_loss"]

            # ── Track expert routing ───────────────────────────────────────
            with torch.no_grad():
                layer0 = cwm.predictor.layers[0]
                flat   = particles_t.reshape(-1, cwm.d_model)
                probs  = F.softmax(layer0.moe_ffn.router(flat), dim=-1)
                tracker.update(domain, probs)

            # ── Domain-weighted losses ─────────────────────────────────────
            dw = get_domain_loss_weights(domain)

            L_jepa, jepa_stats = jepa_moe_loss(
                z_pred, z_target, moe_aux,
                neuro_da_eff = signals["da_effective"]
            )
            L_jepa = L_jepa * dw["L_jepa"]

            L_neuro, _ = loss_fn(
                z_pred             = z_pred,
                z_target           = z_target,
                signed_dist        = signed_dist        if dw["L_contact"] > 0 else None,
                particle_positions = pos_t              if dw["L_contact"] > 0 else None,
                gps_pred           = gps_pred           if dw["L_gps"]     > 0 else None,
                gps_target         = gps                if dw["L_gps"]     > 0 else None,
            )

            L_contact_sup = out["contact_loss"] * dw["L_contact"]

            # ── MoE router anti-collapse (Method D) ────────────────────────
            # Switch load balancing + z-loss to prevent expert collapse.
            # Triggered because epoch 0 probe showed Expert3=100%.
            # alpha=0.05 (higher than default 0.01 — collapse already occurred)
            # beta=0.001 (z-loss, penalises large router logits)
            try:
                layer0   = cwm.predictor.layers[0]
                flat     = particles_t.reshape(-1, cwm.d_model)
                logits   = layer0.moe_ffn.router(flat)          # (T, N_experts)
                probs    = F.softmax(logits, dim=-1)
                top_idx  = probs.argmax(-1)
                N_exp    = probs.shape[-1]
                f_i      = torch.zeros(N_exp, device=probs.device).scatter_add(
                               0, top_idx, torch.ones(flat.shape[0], device=probs.device)
                           ) / flat.shape[0]
                P_i      = probs.mean(0)
                L_lb     = 0.01 * N_exp * (f_i * P_i).sum()
                L_z      = 0.0002 * (torch.logsumexp(logits, dim=-1) ** 2).mean()
                L_router = L_lb + L_z
            except Exception:
                L_router = torch.tensor(0.0)

            total_loss = L_jepa + L_neuro + L_contact_sup + L_router

            # ── Optimise ───────────────────────────────────────────────────
            if not torch.isfinite(total_loss):
                optimizer.zero_grad()
                continue  # skip NaN/inf — never call backward on NaN

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainable_params, config["gradient_clip"]
            )
            # NOTE: removed manual lr override — OneCycleLR controls lr
            optimizer.step()
            scheduler.step()

            loss_val = total_loss.item()
            epoch_losses[domain].append(loss_val)
            epoch_losses["all"].append(loss_val)
            domain_stats[domain].append(loss_val)
            global_step += 1

            if global_step % log_every == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                print(
                    f"[ep{epoch:02d} s{global_step:05d}] "
                    f"domain={domain:8s} "
                    f"loss={loss_val:.4f}  "
                    f"regime={signals['regime']:10s} "
                    f"DA={signals['da']:.3f} "
                    f"5HT={signals['sht']:.3f} "
                    f"ACh={signals['ach']:.3f} "
                    f"lr={lr_now:.2e}"
                )

        # ── Epoch summary ──────────────────────────────────────────────────
        mean_all = np.mean(epoch_losses["all"]) if epoch_losses["all"] else 0
        print(f"\nEpoch {epoch:02d}  mean_loss={mean_all:.4f}")
        print("  Per-domain losses:")
        for d in active_domains:
            dl = epoch_losses.get(d, [])
            if dl:
                print(f"    {d:12s}: {np.mean(dl):.4f} ({len(dl)} batches)")

        # ── Expert specialisation report ───────────────────────────────────
        if (epoch + 1) % report_every == 0:
            tracker.report(epoch)

        # ── Checkpoint ────────────────────────────────────────────────────
        if mean_all < best_loss:
            best_loss = mean_all
            path = Path(save_dir) / "cwm_multidomain_best.pt"
            torch.save({
                "epoch":          epoch,
                "loss":           best_loss,
                "model":          cwm.state_dict(),
                "active_domains": active_domains,
                "signals":        signals,
            }, path)
            print(f"  → Saved: {path}")

    # ── Final specialisation report ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Multi-domain training complete")
    print(f"Best loss: {best_loss:.4f}")
    print("=" * 60)
    tracker.report(n_epochs)

    return cwm


# ═══════════════════════════════════════════════════════════════════════════
# OGBench-Cube hierarchical fine-tune  (Sprint 3 — requires cwm_hierarchical)
# ═══════════════════════════════════════════════════════════════════════════

def train_ogbench_hierarchical(
    cwm_ckpt:      str   = r"checkpoints\cwm\cwm_multidomain_best.pt",
    ogbench_dir:   str   = "ogbench_data",
    n_epochs:      int   = 20,
    base_lr:       float = 5e-5,
    save_dir:      str   = r"checkpoints\cwm",
    device_str:    str   = "cpu",
):
    """
    Hierarchical fine-tune for OGBench-Cube long-horizon skill composition.
    Runs AFTER multi-domain training. Trains boundary detector and SCaR/SPlaTES.
    """
    from cwm_hierarchical import HierarchicalCWM, train_hierarchical_cwm
    from domain_loaders import OGBenchDomain
    from torch.utils.data import DataLoader

    device = torch.device(device_str)

    cwm = CortexWorldModel(d_model=128, K=16).to(device)
    if Path(cwm_ckpt).exists():
        ckpt = torch.load(cwm_ckpt, map_location=device, weights_only=False)
        cwm.load_state_dict(ckpt["model"])
        print(f"Multi-domain checkpoint loaded: {cwm_ckpt}")

    hcwm = HierarchicalCWM(cwm, d_model=128, K=16, action_dim=9).to(device)

    if not Path(ogbench_dir).exists():
        print(f"OGBench data not found at {ogbench_dir} — skipping")
        return hcwm

    ds     = OGBenchDomain(ogbench_dir)
    loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=2)
    print(f"OGBench dataset: {len(ds):,} samples")

    train_hierarchical_cwm(
        hcwm     = hcwm,
        ogbench_loader = loader,
        n_epochs = n_epochs,
        base_lr  = base_lr,
        save_dir = save_dir,
        device_str = device_str,
    )
    return hcwm


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cwm-ckpt",      default=r"checkpoints\cwm\cwm_best.pt")
    p.add_argument("--recon-dir",     default="recon_data/recon_release")
    p.add_argument("--ogbench-dir",   default="ogbench_data")
    p.add_argument("--pusht-dir",     default="pusht_data")
    p.add_argument("--tworoom-dir",   default="tworoom_data")
    p.add_argument("--smap-dir",      default="smap")
    p.add_argument("--mvtec-dir",     default="mvtec")
    p.add_argument("--epochs",        type=int,   default=20)
    p.add_argument("--batch-size",    type=int,   default=16)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--save-dir",      default=r"checkpoints\cwm")
    p.add_argument("--log-every",     type=int,   default=50)
    p.add_argument("--report-every",  type=int,   default=5)
    p.add_argument("--hierarchical",  action="store_true",
                   help="Run OGBench hierarchical fine-tune after multi-domain")
    args = p.parse_args()

    domain_roots = {
        "recon":   args.recon_dir,
        "ogbench": args.ogbench_dir,
        "pusht":   args.pusht_dir,
        "tworoom": args.tworoom_dir,
        "smap":    args.smap_dir,
        "mvtec":   args.mvtec_dir,
    }

    cwm = train_multidomain(
        cwm_ckpt     = args.cwm_ckpt,
        domain_roots = domain_roots,
        n_epochs     = args.epochs,
        batch_size   = args.batch_size,
        base_lr      = args.lr,
        save_dir     = args.save_dir,
        log_every    = args.log_every,
        report_every = args.report_every,
    )

    if args.hierarchical:
        print("\nStarting OGBench hierarchical fine-tune...")
        train_ogbench_hierarchical(
            cwm_ckpt    = str(Path(args.save_dir) / "cwm_multidomain_best.pt"),
            ogbench_dir = args.ogbench_dir,
            save_dir    = args.save_dir,
        )
