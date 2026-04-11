"""
vlm_neuro_test.py — NeMo-WM Neuromodulator + VLM Integration Test
==================================================================
Tests three vision encoder configurations with the NeMo-WM neuromodulator
to compare how each responds to:
  1. Stable input (same scene repeated)          → expect low DA, low cortisol
  2. Novel input (scene change)                  → expect high DA
  3. Domain shift (outdoor → indoor → manip)     → expect cortisol to rise
  4. Representation collapse (noise images)      → expect high 5HT response
  5. Semantic similarity (similar scenes)        → expect moderate DA

Three configs:
  A. DINOv2 StudentEncoder (already on CORTEX — no download)
  B. CLIP ViT-B/32 (400MB download — vision+language bridge)
  C. SmolVLM-256M (256M params — smallest full VLM)

Ablation:
  - With neuromodulator (full NeMo-WM signals)
  - Without neuromodulator (raw encoder loss only)
  - Cortisol only (single signal ablation)

Usage:
    python vlm_neuro_test.py                    # all three configs
    python vlm_neuro_test.py --config dino      # DINOv2 only (fastest)
    python vlm_neuro_test.py --config clip      # CLIP only
    python vlm_neuro_test.py --config smolvlm   # SmolVLM only
    python vlm_neuro_test.py --no-download      # skip any downloads

Author: John Taylor — github.com/taylorjohn
Sprint: VLM integration test
"""

import argparse
import io
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from pathlib import Path

# ── Synthetic test images ──────────────────────────────────────────────────────

def make_image(scene_type: str, size: int = 224) -> Image.Image:
    """Generate synthetic test images for each scenario."""
    img = Image.new("RGB", (size, size), (0, 0, 0))
    d = ImageDraw.Draw(img)

    if scene_type == "outdoor_1":
        # Sky + ground + horizon
        d.rectangle([0, 0, size, size//2], fill=(135, 206, 235))   # sky blue
        d.rectangle([0, size//2, size, size], fill=(34, 139, 34))  # grass green
        d.ellipse([size//3, size//4, 2*size//3, size//2-10], fill=(255, 255, 0))  # sun
    elif scene_type == "outdoor_2":
        # Similar outdoor, different time
        d.rectangle([0, 0, size, size//2], fill=(255, 140, 0))     # sunset
        d.rectangle([0, size//2, size, size], fill=(34, 100, 34))  # darker grass
        d.ellipse([size//4, size//4, size//2, size//2], fill=(255, 60, 0))
    elif scene_type == "indoor":
        # Room — walls, floor, furniture
        d.rectangle([0, 0, size, size], fill=(245, 245, 220))      # beige wall
        d.rectangle([0, 3*size//4, size, size], fill=(139, 90, 43)) # brown floor
        d.rectangle([size//4, size//4, 3*size//4, 2*size//3], fill=(100, 100, 180))  # sofa
    elif scene_type == "manipulation":
        # Robot arm + object
        d.rectangle([0, 0, size, size], fill=(50, 50, 50))         # dark background
        d.rectangle([size//2-5, size//4, size//2+5, 3*size//4], fill=(200, 200, 200))  # arm
        d.rectangle([size//3, 2*size//3, 2*size//3, size-10], fill=(255, 100, 0))  # object
    elif scene_type == "noise":
        # Random noise — should trigger 5HT (collapse prevention)
        arr = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        return img
    elif scene_type == "outdoor_similar":
        # Very similar to outdoor_1 — small perturbation
        d.rectangle([0, 0, size, size//2], fill=(130, 200, 230))   # slightly different sky
        d.rectangle([0, size//2, size, size], fill=(30, 135, 30))  # slightly different grass
        d.ellipse([size//3+5, size//4+5, 2*size//3+5, size//2-5], fill=(255, 250, 0))

    return img


SCENARIOS = [
    ("stable",          ["outdoor_1", "outdoor_1", "outdoor_1"]),  # same scene x3
    ("novel",           ["outdoor_1", "indoor", "manipulation"]),  # scene changes
    ("domain_shift",    ["outdoor_1", "outdoor_2", "indoor"]),     # gradual shift
    ("collapse_risk",   ["noise", "noise", "noise"]),              # noise = collapse test
    ("similar_scenes",  ["outdoor_1", "outdoor_similar", "outdoor_2"]),  # semantically close
]


# ── Encoder wrappers ───────────────────────────────────────────────────────────

class DinoStudentEncoder:
    """Uses existing CORTEX StudentEncoder — no download."""
    name = "DINOv2-Student (CORTEX)"
    size_mb = 0

    def __init__(self):
        try:
            import sys
            sys.path.insert(0, '.')
            from train_mvtec import StudentEncoder
            self.model = StudentEncoder()
            ckpt = torch.load(
                r'checkpoints\dinov2_student\student_best.pt',
                map_location='cpu', weights_only=False
            )
            sd = ckpt.get('model', ckpt.get('state_dict', ckpt))
            self.model.load_state_dict(sd, strict=False)
            self.model.eval()
            self.available = True
            print(f"  ✅ {self.name} loaded")
        except Exception as e:
            self.available = False
            print(f"  ❌ {self.name} failed: {e}")

    def encode(self, img: Image.Image) -> torch.Tensor:
        from torchvision import transforms
        t = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        x = t(img).unsqueeze(0)
        with torch.no_grad():
            z = self.model(x)
        return z.squeeze(0)


class CLIPEncoder:
    """CLIP ViT-B/32 — vision+language bridge, ~400MB."""
    name = "CLIP ViT-B/32"
    size_mb = 400

    def __init__(self, no_download=False):
        try:
            from transformers import CLIPProcessor, CLIPModel
            model_id = "openai/clip-vit-base-patch32"
            if no_download:
                self.available = False
                print(f"  ⏭  {self.name} skipped (--no-download)")
                return
            print(f"  ⬇  {self.name} downloading (~400MB)...")
            self.processor = CLIPProcessor.from_pretrained(model_id)
            self.model = CLIPModel.from_pretrained(model_id)
            self.model.eval()
            self.available = True
            print(f"  ✅ {self.name} loaded")
        except Exception as e:
            self.available = False
            print(f"  ❌ {self.name} failed: {e}")

    def encode(self, img: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=img, return_tensors="pt")
        with torch.no_grad():
            # Use vision_model directly — works across transformers versions
            out = self.model.vision_model(pixel_values=inputs["pixel_values"])
            # pooler_output is (B, D), last_hidden_state[:,0] is CLS token
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                z = out.pooler_output
            else:
                z = out.last_hidden_state[:, 0, :]  # CLS token
        return z.squeeze(0)


class SmolVLMEncoder:
    """SmolVLM-256M — smallest full VLM, vision encoder only."""
    name = "SmolVLM-256M"
    size_mb = 500

    def __init__(self, no_download=False):
        try:
            from transformers import AutoProcessor, AutoModel
            model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
            if no_download:
                self.available = False
                print(f"  ⏭  {self.name} skipped (--no-download)")
                return
            print(f"  ⬇  {self.name} downloading (~500MB)...")
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModel.from_pretrained(model_id)
            self.model.eval()
            self.available = True
            print(f"  ✅ {self.name} loaded")
        except Exception as e:
            self.available = False
            print(f"  ❌ {self.name} failed: {e}")

    def encode(self, img: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=img, return_tensors="pt")
        with torch.no_grad():
            # SmolVLM / Idefics3 — access vision model
            try:
                vision = self.model.model.vision_model
            except AttributeError:
                try:
                    vision = self.model.vision_model
                except AttributeError:
                    vision = self.model.model.vision_model

            # Idefics3 processor returns pixel_values as list of tensors (tiles)
            # Flatten if needed
            pv = inputs.get("pixel_values")
            if pv is None:
                raise RuntimeError("No pixel_values in processor output")
            if isinstance(pv, list):
                pv = torch.cat([p if p.dim() == 4 else p.unsqueeze(0) for p in pv], dim=0)
            if pv.dim() == 5:
                # (B, tiles, C, H, W) -> process first tile only
                pv = pv[:, 0]

            # Cast to model dtype (SmolVLM loads as BFloat16)
            model_dtype = next(vision.parameters()).dtype
            pv = pv.to(dtype=model_dtype)

            out = vision(pixel_values=pv)
            # BaseModelOutputWithPooling — use last_hidden_state mean pooling
            z = out.last_hidden_state.mean(dim=1).squeeze(0)
        return z


# ── Minimal neuromodulator (self-contained, no imports needed) ─────────────────

class MinimalNeuromodulator:
    """
    Self-contained neuromodulator for integration testing.
    Does not import from CORTEX — standalone for VLM testing.
    Eight signals matching NeMo-WM architecture.
    """
    def __init__(self, baseline: float = None):
        self.baseline = baseline  # None = auto-calibrate
        self.loss_history = []
        self.z_history = []
        self.step = 0

    def update(self, z_pred: torch.Tensor, z_target: torch.Tensor) -> dict:
        self.step += 1

        # Normalise
        zp = F.normalize(z_pred.float().flatten(), dim=0)
        zt = F.normalize(z_target.float().flatten(), dim=0)

        # ── Dopamine: prediction surprise (geometric distance) ────────────────
        da = float(1.0 - torch.dot(zp, zt).clamp(-1, 1))

        # ── Serotonin: representation diversity (variance across batch) ───────
        if len(self.z_history) >= 3:
            stack = torch.stack(self.z_history[-8:])
            var = stack.var(dim=0).mean().item()
            sht = float(np.clip(1.0 - var * 10, 0, 1))  # high when diversity LOW
        else:
            sht = 0.5

        # ── Norepinephrine: uncertainty / spatial novelty ────────────────────
        ne = float(torch.norm(zp - zt).item() / (torch.norm(zt).item() + 1e-6))
        ne = float(np.clip(ne, 0, 1))

        # ── Acetylcholine: contact / sharp transition detection ───────────────
        if len(self.z_history) >= 2:
            prev = F.normalize(self.z_history[-1].float().flatten(), dim=0)
            ach = float(1.0 - torch.dot(zp, prev).clamp(-1, 1))
        else:
            ach = 0.0

        # ── Endocannabinoid: context novelty (rolling mean distance) ─────────
        ecb = float(da * 0.3 + ne * 0.7)

        # ── Adenosine: fatigue (inverse of recent activity) ──────────────────
        if len(self.loss_history) >= 5:
            recent = np.mean(self.loss_history[-5:])
            older  = np.mean(self.loss_history[-10:-5]) if len(self.loss_history) >= 10 else recent
            ado = float(np.clip((older - recent) / (older + 1e-6), 0, 1))
        else:
            ado = 0.0

        # ── E/I balance: arousal (ratio of excitatory to inhibitory) ─────────
        ei = float(np.clip((da + ne) / (sht + ado + 1e-6), 0, 2)) / 2.0

        # ── Cortisol: distribution shift detector ────────────────────────────
        mse = float(F.mse_loss(zp, zt).item())
        self.loss_history.append(mse)

        if self.baseline is None and len(self.loss_history) >= 5:
            self.baseline = float(np.mean(self.loss_history[:5])) * 0.95

        if self.baseline is not None and len(self.loss_history) >= 3:
            recent_loss = float(np.mean(self.loss_history[-3:]))
            cort = float(np.clip((recent_loss - self.baseline) / (self.baseline + 1e-6), 0, 1))
        else:
            cort = 0.0

        # ── Regime ────────────────────────────────────────────────────────────
        regime = "REOBSERVE" if da > 0.01 else "EXPLOIT"

        signals = {
            "da": round(da, 4),
            "sht": round(sht, 4),
            "ne": round(ne, 4),
            "ach": round(ach, 4),
            "ecb": round(ecb, 4),
            "ado": round(ado, 4),
            "ei": round(ei, 4),
            "cort": round(cort, 4),
            "regime": regime,
            "mse": round(mse, 4),
        }

        self.z_history.append(zt.detach())
        if len(self.z_history) > 20:
            self.z_history.pop(0)

        return signals


# ── Run one scenario ──────────────────────────────────────────────────────────

def run_scenario(encoder, scenario_name: str, scene_types: list,
                 with_neuro: bool = True, cortisol_only: bool = False) -> dict:
    """Run a scenario and return signal averages."""
    neuro = MinimalNeuromodulator()
    results = []

    images = [make_image(s) for s in scene_types]

    for i in range(len(images) - 1):
        z_t   = encoder.encode(images[i])
        z_t1  = encoder.encode(images[i + 1])

        if with_neuro:
            signals = neuro.update(z_t, z_t1)
            if cortisol_only:
                signals = {k: (v if k in ['cort', 'regime', 'mse'] else 0.0)
                          for k, v in signals.items()}
        else:
            # No neuromodulator — just raw MSE
            mse = float(F.mse_loss(
                F.normalize(z_t.float().flatten(), dim=0),
                F.normalize(z_t1.float().flatten(), dim=0)
            ).item())
            signals = {"da": 0, "sht": 0, "ne": 0, "ach": 0,
                      "ecb": 0, "ado": 0, "ei": 0, "cort": 0,
                      "regime": "N/A", "mse": round(mse, 4)}

        results.append(signals)

    # Average numeric signals
    avg = {}
    for key in ["da", "sht", "ne", "ach", "ecb", "ado", "ei", "cort", "mse"]:
        avg[key] = round(float(np.mean([r[key] for r in results])), 4)

    regimes = [r["regime"] for r in results]
    avg["regime"] = "REOBSERVE" if regimes.count("REOBSERVE") > len(regimes) / 2 else "EXPLOIT"
    avg["scenario"] = scenario_name

    return avg


# ── Print results table ───────────────────────────────────────────────────────

def print_table(encoder_name: str, all_results: list):
    print(f"\n{'='*80}")
    print(f"  {encoder_name}")
    print(f"{'='*80}")
    print(f"  {'Scenario':<20} {'DA':>6} {'5HT':>6} {'NE':>6} {'ACh':>6} "
          f"{'Cort':>6} {'MSE':>6} {'Regime':<12}")
    print(f"  {'-'*76}")
    for r in all_results:
        print(f"  {r['scenario']:<20} {r['da']:>6.4f} {r['sht']:>6.4f} "
              f"{r['ne']:>6.4f} {r['ach']:>6.4f} {r['cort']:>6.4f} "
              f"{r['mse']:>6.4f} {r['regime']:<12}")
    print()


def print_ablation(scenario: str, results_by_config: dict):
    print(f"\n{'='*80}")
    print(f"  ABLATION — scenario: {scenario}")
    print(f"{'='*80}")
    print(f"  {'Config':<35} {'DA':>6} {'Cort':>6} {'MSE':>6} {'Regime':<12}")
    print(f"  {'-'*65}")
    for config_name, r in results_by_config.items():
        print(f"  {config_name:<35} {r['da']:>6.4f} {r['cort']:>6.4f} "
              f"{r['mse']:>6.4f} {r['regime']:<12}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       default="all",
                        choices=["all", "dino", "clip", "smolvlm"])
    parser.add_argument("--no-download",  action="store_true")
    parser.add_argument("--scenario",     default="all")
    a = parser.parse_args()

    print("\nNeMo-WM VLM Integration Test")
    print("="*50)
    print("Loading encoders...\n")

    # Load encoders
    encoders = []
    if a.config in ("all", "dino"):
        encoders.append(DinoStudentEncoder())
    if a.config in ("all", "clip"):
        encoders.append(CLIPEncoder(no_download=a.no_download))
    if a.config in ("all", "smolvlm"):
        encoders.append(SmolVLMEncoder(no_download=a.no_download))

    available = [e for e in encoders if e.available]
    if not available:
        print("\nNo encoders available. Check paths and try --config dino")
        return

    print(f"\nRunning {len(SCENARIOS)} scenarios × {len(available)} encoders × 3 ablation configs\n")

    # ── Per-encoder full scenario test ────────────────────────────────────────
    for enc in available:
        all_results = []
        for scenario_name, scene_types in SCENARIOS:
            r = run_scenario(enc, scenario_name, scene_types,
                           with_neuro=True, cortisol_only=False)
            all_results.append(r)
        print_table(enc.name, all_results)

    # ── Ablation on domain_shift scenario ────────────────────────────────────
    domain_shift_scenes = dict(SCENARIOS)["domain_shift"]

    for enc in available:
        ablation = {}
        ablation[f"{enc.name} — full neuromodulator"] = run_scenario(
            enc, "domain_shift", domain_shift_scenes, with_neuro=True)
        ablation[f"{enc.name} — cortisol only"] = run_scenario(
            enc, "domain_shift", domain_shift_scenes,
            with_neuro=True, cortisol_only=True)
        ablation[f"{enc.name} — no neuromodulator"] = run_scenario(
            enc, "domain_shift", domain_shift_scenes, with_neuro=False)

        print_ablation("domain_shift", ablation)

    # ── Cross-encoder comparison on novel scenario ────────────────────────────
    if len(available) > 1:
        novel_scenes = dict(SCENARIOS)["novel"]
        print(f"\n{'='*80}")
        print(f"  CROSS-ENCODER COMPARISON — scenario: novel")
        print(f"{'='*80}")
        print(f"  {'Encoder':<35} {'DA':>6} {'5HT':>6} {'Cort':>6} {'MSE':>6}")
        print(f"  {'-'*65}")
        for enc in available:
            r = run_scenario(enc, "novel", novel_scenes, with_neuro=True)
            print(f"  {enc.name:<35} {r['da']:>6.4f} {r['sht']:>6.4f} "
                  f"{r['cort']:>6.4f} {r['mse']:>6.4f}")
        print()

    print("\nInterpretation guide:")
    print("  DA   high on 'novel', 'domain_shift' → neuromodulator detecting change")
    print("  5HT  high on 'collapse_risk' (noise)  → collapse prevention active")
    print("  Cort rising on 'domain_shift'          → distribution shift detection")
    print("  DA   low on 'stable'                   → correct quiescence")
    print("  MSE  compare across configs             → raw encoder sensitivity baseline")
    print()
    print("Expected REOBSERVE: novel, domain_shift, collapse_risk")
    print("Expected EXPLOIT:   stable, similar_scenes")


if __name__ == "__main__":
    main()
