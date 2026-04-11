"""
vlm_phase2b_cortisol_adapt.py — Cortisol-Triggered Continual Adaptation
========================================================================
Phase 2b of VLM integration plan.

Problem: VLMs forget when deployed in new environments. Standard approach
is periodic full fine-tuning — expensive, disrupts stable representations,
requires human-curated new data.

NeMo-WM solution: cortisol detects distribution shift one epoch before
loss moves. When cortisol rises above calibrated threshold, trigger a
targeted 10-step fine-tuning pass on the vision encoder only.
Language model weights stay frozen throughout.

Key properties:
  - Triggered automatically by cortisol — no human intervention
  - Fine-tunes vision encoder only — language weights preserved
  - 10-step micro-update — fast, targeted, minimal forgetting
  - Cortisol resets after adaptation — prevents runaway updates
  - Auditable: every adaptation event logged with cortisol level + trigger reason

Comparison:
  A. Static model — no adaptation (baseline)
  B. Periodic fine-tuning — every N steps regardless (standard approach)
  C. Cortisol-triggered — adapts only when cortisol threshold exceeded

Evaluation:
  - Inject domain shift at step 100 (outdoor → industrial)
  - Measure DA recovery rate after shift
  - Measure stable-phase false positive rate (spurious adaptations)
  - Compare adaptation speed and stability

Usage:
    python vlm_phase2b_cortisol_adapt.py
    python vlm_phase2b_cortisol_adapt.py --shift-step 50 --n-steps 200
    python vlm_phase2b_cortisol_adapt.py --encoder clip  # use CLIP instead of DINOv2

Author: John Taylor — github.com/taylorjohn
Sprint: VLM Phase 2b
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
from PIL import Image, ImageDraw
from torchvision import transforms


# ── Image generators ───────────────────────────────────────────────────────────

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_recon_frames(hdf5_dir: str, n: int = 150) -> list:
    """Load N consecutive RECON frames (stable domain)."""
    import h5py
    frames = []
    files = sorted(glob.glob(f"{hdf5_dir}/*.hdf5"))
    for path in files:
        if len(frames) >= n:
            break
        try:
            with h5py.File(path) as hf:
                imgs = hf["images"]["rgb_left"]
                for i in range(min(10, len(imgs))):
                    if len(frames) >= n:
                        break
                    jpeg = bytes(imgs[i])
                    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
                    frames.append(("recon", img))
        except Exception:
            pass
    return frames


def make_shift_frame(scene: str = "industrial", size: int = 224,
                     seed: int = None) -> Image.Image:
    """
    Generate domain-shift frames WITH random variation.
    Each call produces a slightly different frame so consecutive
    shift frames have non-zero DA (like real environments).
    seed=None for random variation, seed=N for reproducible.
    """
    if seed is not None:
        np.random.seed(seed)
    img = Image.new("RGB", (size, size))
    d = ImageDraw.Draw(img)
    # Add random jitter to colours and positions
    jitter = lambda v, r=15: int(np.clip(v + np.random.randint(-r, r), 0, 255))
    if scene == "industrial":
        bg = (jitter(30), jitter(30), jitter(30))
        d.rectangle([0, 0, size, size], fill=bg)
        # Moving machinery — random stripe positions
        for i in range(0, size, np.random.randint(30, 50)):
            stripe_col = (jitter(60), jitter(60), jitter(60))
            d.rectangle([i, 0, i + np.random.randint(10, 25), size],
                        fill=stripe_col)
        # Object at random position within region
        ox = np.random.randint(size//5, size//3)
        oy = np.random.randint(size//4, size//2)
        obj_col = (jitter(200, 20), jitter(80, 20), 0)
        d.rectangle([ox, oy, ox + size//2, oy + size//3], fill=obj_col)
    elif scene == "medical":
        d.rectangle([0, 0, size, size], fill=(jitter(240), jitter(240), jitter(240)))
        cx = np.random.randint(size//3, 2*size//3)
        cy = np.random.randint(size//3, 2*size//3)
        d.ellipse([cx-size//4, cy-size//4, cx+size//4, cy+size//4],
                  fill=(jitter(200, 10), jitter(150, 10), jitter(150, 10)))
    elif scene == "underwater":
        d.rectangle([0, 0, size, size],
                    fill=(jitter(0, 5), jitter(60, 10), jitter(120, 10)))
        for i in range(np.random.randint(3, 8)):
            x = np.random.randint(10, size-10)
            y = np.random.randint(10, size-10)
            r = np.random.randint(8, 20)
            d.ellipse([x-r, y-r, x+r, y+r],
                      fill=(0, jitter(120, 20), jitter(80, 20)))
    return img


# ── Cortisol monitor ───────────────────────────────────────────────────────────

class CortisolMonitor:
    """
    Standalone cortisol signal for deployment monitoring.
    Tracks rolling loss and fires when distribution shift detected.
    Calibrated threshold from Phase 1c.
    """

    def __init__(self, baseline: float = None, sensitivity: float = 1.0,
                 window: int = 10, threshold: float = None):
        self.baseline     = baseline      # None = auto from first N steps
        self.sensitivity  = sensitivity
        self.window       = window
        self.threshold    = threshold     # DA threshold (from Phase 1c calibration)
        self.loss_history = []
        self.da_history   = []
        self.events       = []            # log of adaptation triggers
        self.step         = 0
        self._calibration_steps = 20

    def update(self, da: float, loss: float) -> dict:
        """
        Update cortisol with current DA and loss.
        Returns signal dict including whether adaptation should trigger.
        """
        self.step += 1
        self.loss_history.append(loss)
        self.da_history.append(da)

        # Auto-calibrate baseline from first N steps
        if (self.baseline is None and
                len(self.loss_history) >= self._calibration_steps):
            self.baseline = float(np.mean(
                self.loss_history[:self._calibration_steps])) * 0.95
            print(f"    [Cortisol] Baseline calibrated: {self.baseline:.4f}")

        # Cortisol: rolling loss vs baseline
        if self.baseline and len(self.loss_history) >= 3:
            recent = float(np.mean(self.loss_history[-3:]))
            std    = getattr(self, '_stable_std', self.baseline * 0.1) + 1e-8
            # Z-score: how many standard deviations above stable baseline?
            # Fires when recent loss is >2 sigma above stable distribution
            z_score = (recent - self.baseline) / std
            cort = float(np.clip(z_score / 3.0, 0, 1))  # 3-sigma = full cortisol
        else:
            cort = 0.0

        # DA rolling average
        da_mean = float(np.mean(self.da_history[-self.window:])) \
            if len(self.da_history) >= self.window else da

        # Adaptation trigger: cortisol high OR DA above calibrated threshold
        da_trigger   = (self.threshold is not None and da_mean > self.threshold)
        cort_trigger = cort > 0.6   # stricter — only genuine stress events
        cooldown     = getattr(self, '_cooldown', 0)
        if cooldown > 0:
            self._cooldown = cooldown - 1
        should_adapt = (da_trigger or cort_trigger) and                        self.step > self._calibration_steps and cooldown == 0

        return {
            "step":         self.step,
            "da":           round(da, 4),
            "da_mean":      round(da_mean, 4),
            "loss":         round(loss, 4),
            "cort":         round(cort, 4),
            "baseline":     round(self.baseline, 4) if self.baseline else None,
            "should_adapt": should_adapt,
            "trigger":      "da" if da_trigger else ("cort" if cort_trigger else "none"),
        }

    def log_event(self, step: int, trigger: str, cort: float, da: float):
        self.events.append({"step": step, "trigger": trigger,
                           "cort": cort, "da": da})

    def reset_after_adapt(self, cooldown_steps: int = 10):
        """Reset rolling window after adaptation to prevent runaway updates."""
        if self.loss_history:
            # Soft reset — keep history but update baseline to current level
            recent = float(np.mean(self.loss_history[-5:]))
            self.baseline = recent * 0.95
            self.da_history = self.da_history[-5:]  # keep last 5 only
            # Cooldown: don't trigger again for N steps
            self._cooldown = cooldown_steps


# ── Encoder ────────────────────────────────────────────────────────────────────

def load_dino_encoder():
    import sys; sys.path.insert(0, '.')
    from train_mvtec import StudentEncoder
    model = StudentEncoder()
    ckpt = torch.load(
        r'checkpoints\dinov2_student\student_best.pt',
        map_location='cpu', weights_only=False
    )
    sd = ckpt.get('model', ckpt.get('state_dict', ckpt))
    model.load_state_dict(sd, strict=False)
    return model


def load_clip_encoder():
    from transformers import CLIPProcessor, CLIPModel
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    return proc, model


def encode_dino(model, img: Image.Image) -> torch.Tensor:
    x = TRANSFORM(img).unsqueeze(0)
    with torch.no_grad():
        z = model(x)
    return F.normalize(z.squeeze(0).float(), dim=0)


def encode_clip(proc, model, img: Image.Image) -> torch.Tensor:
    inputs = proc(images=img, return_tensors="pt")
    with torch.no_grad():
        out = model.vision_model(pixel_values=inputs["pixel_values"])
        z = out.pooler_output if out.pooler_output is not None \
            else out.last_hidden_state[:, 0, :]
        z = model.visual_projection(z)
    return F.normalize(z.squeeze(0).float(), dim=0)


# ── Micro fine-tuning ──────────────────────────────────────────────────────────

def micro_finetune(encoder, recent_frames: list, n_steps: int = 10,
                   lr: float = 1e-5) -> float:
    """
    10-step targeted fine-tuning on recent frames.
    Uses self-supervised consistency loss — no labels needed.
    Returns mean loss after adaptation.
    """
    encoder.train()
    opt = torch.optim.AdamW(encoder.parameters(), lr=lr)
    losses = []

    for step in range(n_steps):
        if len(recent_frames) < 2:
            break
        # Random consecutive pair from recent frames
        idx = np.random.randint(0, len(recent_frames) - 1)
        img_t  = recent_frames[idx]
        img_t1 = recent_frames[idx + 1]

        z_t  = encoder(TRANSFORM(img_t).unsqueeze(0))
        z_t1 = encoder(TRANSFORM(img_t1).unsqueeze(0))

        # Consistency loss: consecutive frames should be similar
        # (adapted to new domain statistics without forgetting old ones)
        z_t  = F.normalize(z_t.squeeze(0), dim=0)
        z_t1 = F.normalize(z_t1.squeeze(0), dim=0)

        L_consist = 1.0 - torch.dot(z_t, z_t1)
        L_diverse = F.relu(0.3 - z_t.var())   # prevent collapse

        loss = L_consist + 0.1 * L_diverse

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 0.5)
        opt.step()
        losses.append(loss.item())

    encoder.eval()
    return float(np.mean(losses)) if losses else 0.0


# ── Simulation ─────────────────────────────────────────────────────────────────

def run_simulation(encoder_name: str, encode_fn,
                   stable_frames: list, shift_scene: str,
                   shift_step: int, n_steps: int,
                   mode: str,  # "static" | "periodic" | "cortisol"
                   da_threshold: float = 0.0613,
                   periodic_interval: int = 20) -> dict:
    """
    Simulate deployment with injected domain shift.

    Modes:
      static   — no adaptation (baseline)
      periodic — adapt every periodic_interval steps
      cortisol — adapt only when cortisol triggers
    """
    monitor = CortisolMonitor(threshold=da_threshold)
    log = []
    adaptations = []
    recent_frames = []
    z_prev = None

    # Normalise both to (domain, img) tuples
    stable_tuples = [(d, img) if isinstance(d, str) else ("recon", d)
                     for d, img in (stable_frames[:shift_step]
                     if stable_frames and isinstance(stable_frames[0], tuple)
                     else [("recon", f) for f in stable_frames[:shift_step]])]
    # Each shift frame is different — random variation for realistic dynamics
    shift_tuples  = [("shift", make_shift_frame(shift_scene))
                     for _ in range(n_steps - shift_step)]
    all_frames    = stable_tuples + shift_tuples

    for i, (domain, img) in enumerate(all_frames[:n_steps]):
        z = encode_fn(img)
        recent_frames.append(img)
        if len(recent_frames) > 30:
            recent_frames.pop(0)

        if z_prev is not None:
            da   = float(1.0 - torch.dot(z, z_prev).clamp(-1, 1))
            mse  = float(F.mse_loss(z, z_prev).item())
            sig  = monitor.update(da, mse)

            # Adaptation decision
            adapted = False
            if mode == "periodic" and i % periodic_interval == 0 and i > 20:
                micro_finetune(encoder_name, recent_frames)
                monitor.reset_after_adapt()
                adapted = True
                adaptations.append({"step": i, "trigger": "periodic"})
            elif mode == "cortisol" and sig["should_adapt"]:
                micro_finetune(encoder_name, recent_frames)
                monitor.reset_after_adapt()
                adapted = True
                adaptations.append({"step": i, "trigger": sig["trigger"],
                                   "cort": sig["cort"]})
                monitor.log_event(i, sig["trigger"], sig["cort"], da)

            actual_domain = "shift" if i >= shift_step else "stable"
            log.append({
                "step":        i,
                "domain":      actual_domain,
                "da":          round(da, 4),
                "cort":        round(sig["cort"], 4),
                "adapted":     adapted,
            })

        z_prev = z.detach()

    # Compute metrics
    stable_log = [r for r in log if r["domain"] == "stable"]
    shift_log  = [r for r in log if r["domain"] == "shift"]

    # False positive rate: adaptations during stable phase
    stable_adapts = [a for a in adaptations if a["step"] < shift_step]
    shift_adapts  = [a for a in adaptations if a["step"] >= shift_step]

    # DA recovery: steps until DA returns below threshold after shift
    recovery_step = None
    if shift_log:
        for r in shift_log:
            if r["da"] < da_threshold and recovery_step is None:
                recovery_step = r["step"] - shift_step

    return {
        "mode":              mode,
        "n_adaptations":     len(adaptations),
        "false_positives":   len(stable_adapts),
        "shift_adaptations": len(shift_adapts),
        "stable_da_mean":    round(float(np.mean([r["da"] for r in stable_log])), 4)
                             if stable_log else 0,
        "shift_da_mean":     round(float(np.mean([r["da"] for r in shift_log])), 4)
                             if shift_log else 0,
        "shift_da_peak":     round(float(np.max([r["da"] for r in shift_log])), 4)
                             if shift_log else 0,
        "recovery_steps":    recovery_step,
        "log":               log,
        "adaptations":       adaptations,
    }


def print_results(results: dict):
    print(f"\n  Mode: {results['mode'].upper()}")
    print(f"  ─────────────────────────────────────────")
    print(f"  Stable DA mean:    {results['stable_da_mean']:.4f}")
    print(f"  Shift  DA mean:    {results['shift_da_mean']:.4f}")
    print(f"  Shift  DA peak:    {results['shift_da_peak']:.4f}")
    print(f"  Total adaptations: {results['n_adaptations']}")
    print(f"  False positives:   {results['false_positives']} "
          f"(during stable phase)")
    print(f"  Shift adaptations: {results['shift_adaptations']}")
    rec = results['recovery_steps']
    if rec is not None:
        print(f"  DA recovery:       {rec} steps after shift")
    else:
        print(f"  DA recovery:       never (DA stayed above threshold)")


def print_comparison(all_results: list):
    print(f"\n{'='*65}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Mode':<12} {'False+':>7} {'Adaptations':>12} "
          f"{'Shift DA':>9} {'Recovery':>10}")
    print(f"  {'-'*60}")
    for r in all_results:
        rec = f"{r['recovery_steps']}steps" if r['recovery_steps'] else "never"
        print(f"  {r['mode']:<12} {r['false_positives']:>7} "
              f"{r['n_adaptations']:>12} "
              f"{r['shift_da_mean']:>9.4f} {rec:>10}")

    # Find winner
    cortisol = next((r for r in all_results if r["mode"] == "cortisol"), None)
    periodic  = next((r for r in all_results if r["mode"] == "periodic"), None)
    static    = next((r for r in all_results if r["mode"] == "static"), None)

    if cortisol:
        print(f"\n  Cortisol vs Static:")
        if static:
            da_imp = static["shift_da_mean"] - cortisol["shift_da_mean"]
            print(f"    DA improvement: {da_imp:+.4f} "
                  f"({'✅ cortisol adapts faster' if da_imp > 0 else '❌ no improvement'})")
        if periodic:
            fp_diff = periodic["false_positives"] - cortisol["false_positives"]
            print(f"    False positive reduction vs periodic: {fp_diff:+d} "
                  f"({'✅ fewer spurious adaptations' if fp_diff > 0 else '—'})")

        if cortisol["false_positives"] == 0 and \
           (static is None or cortisol["shift_da_mean"] < static["shift_da_mean"]):
            print("\n  ✅ CORTISOL WINS: zero false positives + faster recovery")
            print("     Paper claim: cortisol-triggered adaptation is more precise")
            print("     than periodic fine-tuning and more responsive than static.")
        else:
            print("\n  ⚠️  Mixed result — review per-mode metrics above")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder",          default="dino",
                        choices=["dino", "clip"])
    parser.add_argument("--shift-step",       type=int, default=100)
    parser.add_argument("--n-steps",          type=int, default=200)
    parser.add_argument("--shift-scene",      default="industrial",
                        choices=["industrial", "medical", "underwater"])
    parser.add_argument("--periodic-interval",type=int, default=20)
    parser.add_argument("--hdf5-dir",         default="recon_data/recon_release")
    parser.add_argument("--da-threshold",     type=float, default=None,
                        help="DA threshold (default: 0.0613 DINOv2, 0.1424 CLIP)")
    a = parser.parse_args()

    # Default thresholds from Phase 1c calibration
    if a.da_threshold is None:
        a.da_threshold = 0.0613 if a.encoder == "dino" else 0.1424

    print(f"\nNeMo-WM Phase 2b — Cortisol-Triggered Continual Adaptation")
    print("="*65)
    print(f"  Encoder:      {a.encoder.upper()}")
    print(f"  Shift at:     step {a.shift_step} → {a.shift_scene}")
    print(f"  Total steps:  {a.n_steps}")
    print(f"  DA threshold: {a.da_threshold:.4f} (Phase 1c calibrated)")

    # Load encoder
    print(f"\nLoading {a.encoder.upper()} encoder...")
    if a.encoder == "dino":
        try:
            model = load_dino_encoder()
            model.eval()
            encode_fn = lambda img: encode_dino(model, img)
            finetune_model = model
            print("  ✅ DINOv2-Student loaded")
        except Exception as e:
            print(f"  ❌ {e}")
            return
    else:
        proc, model = load_clip_encoder()
        model.eval()
        encode_fn = lambda img: encode_clip(proc, model, img)
        finetune_model = model.vision_model
        print("  ✅ CLIP loaded")

    # Load stable frames
    print(f"\nLoading {a.shift_step + 20} stable RECON frames...")
    stable_frames = load_recon_frames(a.hdf5_dir, n=a.shift_step + 20)
    if len(stable_frames) < 20:
        print("  ❌ Not enough RECON frames")
        return
    print(f"  ✅ Loaded {len(stable_frames)} frames")

    # Wrap encode_fn to return (domain, img) tuples for simulation
    stable_labeled = stable_frames[:a.shift_step]

    print(f"\nRunning three-way comparison...")
    print(f"  Stable phase: steps 0-{a.shift_step}")
    print(f"  Shift phase:  steps {a.shift_step}-{a.n_steps} ({a.shift_scene})\n")

    all_results = []

    for mode in ["static", "periodic", "cortisol"]:
        print(f"  [{mode.upper()}]", end=" ", flush=True)
        # Reload encoder state for fair comparison
        if a.encoder == "dino":
            m2 = load_dino_encoder(); m2.eval()
            ef = lambda img, m=m2: encode_dino(m, img)
        else:
            p2, m2 = load_clip_encoder(); m2.eval()
            ef = lambda img, m=m2, p=p2: encode_clip(p, m, img)

        r = run_simulation(
            encoder_name  = m2,
            encode_fn     = ef,
            stable_frames = stable_labeled,
            shift_scene   = a.shift_scene,
            shift_step    = a.shift_step,
            n_steps       = a.n_steps,
            mode          = mode,
            da_threshold  = a.da_threshold,
            periodic_interval = a.periodic_interval,
        )
        all_results.append(r)
        print(f"done — {r['n_adaptations']} adaptations")

    # Print results
    print(f"\n{'='*65}")
    print("  PER-MODE RESULTS")
    print(f"{'='*65}")
    for r in all_results:
        print_results(r)

    print_comparison(all_results)

    # Timeline printout
    print(f"\n\n  DA TIMELINE (every 10 steps):")
    print(f"  {'Step':>6} {'Domain':<10} {'DA':>7} {'Cort':>7} "
          f"{'Static':>8} {'Periodic':>9} {'Cortisol':>9}")
    print(f"  {'-'*60}")
    for i in range(0, a.n_steps - 1, 10):
        rows = [r["log"][i] if i < len(r["log"]) else None
                for r in all_results]
        if all(r is None for r in rows):
            continue
        ref = next(r for r in rows if r)
        adapt_flags = ["  A" if r and r.get("adapted") else "   "
                       for r in rows]
        domain_str = f"{'SHIFT→' if ref['domain']=='shift' else 'stable':<10}"
        print(f"  {i:>6} {domain_str} "
              f"{ref['da']:>7.4f} {ref['cort']:>7.4f} "
              f"{'':>8}{'':>9}{'':>9}"
              f"{''.join(adapt_flags)}")

    # Save
    import json
    out = {
        "encoder":    a.encoder,
        "shift_step": a.shift_step,
        "shift_scene": a.shift_scene,
        "da_threshold": a.da_threshold,
        "results": [{k: v for k, v in r.items() if k != "log"}
                    for r in all_results]
    }
    with open("vlm_phase2b_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n  Saved: vlm_phase2b_results.json")


if __name__ == "__main__":
    main()
