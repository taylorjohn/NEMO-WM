"""
vlm_phase1c_calibrate.py — Per-Encoder DA Threshold Calibration
================================================================
Phase 1c of VLM integration plan.

Problem identified in Phase 1b:
  - All DA values cluster 0.65-0.83 — threshold 0.75 too coarse
  - Each encoder needs its own DA threshold for REOBSERVE
  - SmolVLM DA ~0.026, DINOv2 DA ~0.215, CLIP DA ~0.330 on same input
  - Text-as-target CLIP DA clusters 0.67-0.83

Method:
  1. Run N=50 consecutive RECON frames through each encoder
  2. Compute DA distribution (stable = within-domain baseline)
  3. Inject synthetic domain shift frames (indoor, manipulation)
  4. Compute DA distribution (shifted = out-of-domain signal)
  5. Set threshold = midpoint between stable 90th pct and shifted 10th pct
  6. Store as domain_topology entry for each encoder

Output:
  - Calibrated DA threshold per encoder
  - Precision/recall at threshold
  - Updated domain_topology.py entries

Usage:
    python vlm_phase1c_calibrate.py
    python vlm_phase1c_calibrate.py --n-frames 100

Author: John Taylor — github.com/taylorjohn
Sprint: VLM Phase 1c
"""

import argparse
import glob
import io
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image, ImageDraw


# ── Image loaders ──────────────────────────────────────────────────────────────

def load_recon_frames(hdf5_dir: str, n: int = 50) -> list:
    """Load N frames from RECON HDF5 files."""
    import h5py
    frames = []
    files = sorted(glob.glob(f"{hdf5_dir}/*.hdf5"))
    for path in files:
        if len(frames) >= n:
            break
        try:
            with h5py.File(path) as hf:
                imgs = hf["images"]["rgb_left"]
                for i in range(min(5, len(imgs))):
                    if len(frames) >= n:
                        break
                    jpeg = bytes(imgs[i])
                    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
                    frames.append(("recon", img))
        except Exception:
            continue
    print(f"  Loaded {len(frames)} RECON frames")
    return frames


def make_shift_frames(n: int = 20) -> list:
    """Make synthetic domain-shift frames (indoor, manipulation)."""
    frames = []
    types = ["indoor", "manipulation", "noise"]
    for i in range(n):
        scene = types[i % len(types)]
        size = 224
        img = Image.new("RGB", (size, size))
        d = ImageDraw.Draw(img)
        if scene == "indoor":
            d.rectangle([0, 0, size, size], fill=(245, 245, 220))
            d.rectangle([0, 3*size//4, size, size], fill=(139, 90, 43))
            d.rectangle([size//4, size//4, 3*size//4, 2*size//3], fill=(100, 100, 180))
        elif scene == "manipulation":
            d.rectangle([0, 0, size, size], fill=(50, 50, 50))
            d.rectangle([size//2-5, size//4, size//2+5, 3*size//4], fill=(200, 200, 200))
            d.rectangle([size//3, 2*size//3, 2*size//3, size-10], fill=(255, 100, 0))
        else:
            arr = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
            img = Image.fromarray(arr)
        frames.append((scene, img))
    return frames


# ── Encoders ───────────────────────────────────────────────────────────────────

def load_dino(ckpt_path: str = r'checkpoints\dinov2_student\student_best.pt'):
    import sys; sys.path.insert(0, '.')
    from train_mvtec import StudentEncoder
    from torchvision import transforms
    model = StudentEncoder()
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('model', ckpt.get('state_dict', ckpt))
    model.load_state_dict(sd, strict=False)
    model.eval()
    t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    def encode(img):
        with torch.no_grad():
            return F.normalize(model(t(img).unsqueeze(0)).squeeze(0).float(), dim=0)
    return "DINOv2-Student", encode


def load_clip():
    from transformers import CLIPProcessor, CLIPModel
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    def encode(img):
        inputs = proc(images=img, return_tensors="pt")
        with torch.no_grad():
            out = model.vision_model(pixel_values=inputs["pixel_values"])
            z = out.pooler_output if out.pooler_output is not None \
                else out.last_hidden_state[:, 0, :]
            z = model.visual_projection(z)
        return F.normalize(z.squeeze(0).float(), dim=0)
    return "CLIP ViT-B/32", encode


def load_clip_text_target(text: str = "outdoor robot navigation grass sky path"):
    """CLIP with text-as-target: DA = distance from text embedding."""
    from transformers import CLIPProcessor, CLIPModel
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    # Pre-compute text target
    inputs = proc(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        zt = out.pooler_output if out.pooler_output is not None \
            else out.last_hidden_state[:, -1, :]
        zt = model.text_projection(zt)
    z_text = F.normalize(zt.squeeze(0).float(), dim=0)

    def encode(img):
        inputs2 = proc(images=img, return_tensors="pt")
        with torch.no_grad():
            out2 = model.vision_model(pixel_values=inputs2["pixel_values"])
            z = out2.pooler_output if out2.pooler_output is not None \
                else out2.last_hidden_state[:, 0, :]
            z = model.visual_projection(z)
        return F.normalize(z.squeeze(0).float(), dim=0), z_text

    print(f"  Text target: '{text}'")
    return f"CLIP text-target", encode


# ── DA computation ─────────────────────────────────────────────────────────────

def compute_da_sequence(encode_fn, frames: list,
                        is_text_target: bool = False) -> list:
    """Compute DA for consecutive frame pairs."""
    das = []
    for i in range(len(frames) - 1):
        _, img_t  = frames[i]
        _, img_t1 = frames[i + 1]

        if is_text_target:
            z_t,  z_text = encode_fn(img_t)
            z_t1, _      = encode_fn(img_t1)
            # DA = distance from text target, averaged over two frames
            da_t  = float(1.0 - torch.dot(z_t,  z_text).clamp(-1, 1))
            da_t1 = float(1.0 - torch.dot(z_t1, z_text).clamp(-1, 1))
            da = (da_t + da_t1) / 2
        else:
            z_t  = encode_fn(img_t)
            z_t1 = encode_fn(img_t1)
            da = float(1.0 - torch.dot(z_t, z_t1).clamp(-1, 1))

        das.append(da)
    return das


def calibrate_threshold(stable_das: list, shifted_das: list,
                        percentile_stable: float = 90.0,
                        percentile_shifted: float = 10.0) -> dict:
    """
    Find optimal threshold between stable and shifted distributions.
    threshold = midpoint between stable 90th pct and shifted 10th pct.
    """
    s90 = float(np.percentile(stable_das, percentile_stable))
    sh10 = float(np.percentile(shifted_das, percentile_shifted))

    if sh10 > s90:
        threshold = (s90 + sh10) / 2
    else:
        # Distributions overlap — use ROC optimal
        all_das = [(d, 0) for d in stable_das] + [(d, 1) for d in shifted_das]
        all_das.sort()
        best_f1, best_thresh = 0, s90
        for thresh, _ in all_das:
            tp = sum(1 for d, l in all_das if d >= thresh and l == 1)
            fp = sum(1 for d, l in all_das if d >= thresh and l == 0)
            fn = sum(1 for d, l in all_das if d < thresh and l == 1)
            p = tp / (tp + fp + 1e-6)
            r = tp / (tp + fn + 1e-6)
            f1 = 2*p*r / (p + r + 1e-6)
            if f1 > best_f1:
                best_f1, best_thresh = f1, thresh
        threshold = best_thresh

    # Compute precision/recall at threshold
    tp = sum(1 for d in shifted_das if d >= threshold)
    fp = sum(1 for d in stable_das  if d >= threshold)
    tn = sum(1 for d in stable_das  if d < threshold)
    fn = sum(1 for d in shifted_das if d < threshold)

    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)

    return {
        "threshold":        round(threshold, 4),
        "stable_mean":      round(float(np.mean(stable_das)), 4),
        "stable_std":       round(float(np.std(stable_das)), 4),
        "stable_p90":       round(s90, 4),
        "shifted_mean":     round(float(np.mean(shifted_das)), 4),
        "shifted_std":      round(float(np.std(shifted_das)), 4),
        "shifted_p10":      round(sh10, 4),
        "precision":        round(precision, 3),
        "recall":           round(recall, 3),
        "f1":               round(f1, 3),
        "overlap":          sh10 <= s90,
    }


def print_calibration(name: str, cal: dict):
    print(f"\n  {'='*60}")
    print(f"  {name}")
    print(f"  {'='*60}")
    print(f"  Stable  distribution: mean={cal['stable_mean']:.4f} "
          f"std={cal['stable_std']:.4f} p90={cal['stable_p90']:.4f}")
    print(f"  Shifted distribution: mean={cal['shifted_mean']:.4f} "
          f"std={cal['shifted_std']:.4f} p10={cal['shifted_p10']:.4f}")
    print(f"  Overlap: {'YES — distributions overlap' if cal['overlap'] else 'NO — clean separation'}")
    print(f"  ──────────────────────────────────────────────────────────")
    print(f"  Calibrated threshold: {cal['threshold']:.4f}")
    print(f"  Precision: {cal['precision']:.3f}  "
          f"Recall: {cal['recall']:.3f}  F1: {cal['f1']:.3f}")
    status = "✅ Good" if cal['f1'] > 0.7 else \
             "⚠️  Moderate" if cal['f1'] > 0.4 else "❌ Poor"
    print(f"  Status: {status}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-frames",  type=int, default=50)
    parser.add_argument("--n-shift",   type=int, default=20)
    parser.add_argument("--hdf5-dir",  default="recon_data/recon_release")
    parser.add_argument("--text",      default="outdoor robot navigation grass sky path")
    a = parser.parse_args()

    print("\nNeMo-WM Phase 1c — Per-Encoder Threshold Calibration")
    print("="*60)

    # Load frames
    print(f"\nLoading {a.n_frames} stable RECON frames...")
    stable_frames = load_recon_frames(a.hdf5_dir, a.n_frames)
    if len(stable_frames) < 10:
        print("  Not enough RECON frames — using synthetic outdoor")
        stable_frames = [("outdoor", _make_outdoor()) for _ in range(a.n_frames)]

    print(f"Generating {a.n_shift} domain-shift frames...")
    shift_frames = make_shift_frames(a.n_shift)

    results = {}

    # ── Encoder 1: DINOv2-Student ─────────────────────────────────────────────
    print("\n[1/3] DINOv2-Student...")
    try:
        name, encode = load_dino()
        stable_das  = compute_da_sequence(encode, stable_frames)
        shifted_das = compute_da_sequence(encode, shift_frames)
        cal = calibrate_threshold(stable_das, shifted_das)
        print_calibration(name, cal)
        results["dino"] = {"name": name, **cal}
    except Exception as e:
        print(f"  ❌ DINOv2 failed: {e}")

    # ── Encoder 2: CLIP visual ────────────────────────────────────────────────
    print("\n[2/3] CLIP ViT-B/32 (visual only)...")
    try:
        name, encode = load_clip()
        stable_das  = compute_da_sequence(encode, stable_frames)
        shifted_das = compute_da_sequence(encode, shift_frames)
        cal = calibrate_threshold(stable_das, shifted_das)
        print_calibration(name, cal)
        results["clip_visual"] = {"name": name, **cal}
    except Exception as e:
        print(f"  ❌ CLIP failed: {e}")

    # ── Encoder 3: CLIP text-as-target ───────────────────────────────────────
    print(f"\n[3/3] CLIP text-as-target...")
    try:
        name, encode = load_clip_text_target(a.text)
        stable_das  = compute_da_sequence(encode, stable_frames,
                                          is_text_target=True)
        shifted_das = compute_da_sequence(encode, shift_frames,
                                          is_text_target=True)
        cal = calibrate_threshold(stable_das, shifted_das)
        print_calibration(name, cal)
        results["clip_text"] = {"name": name, **cal}
    except Exception as e:
        print(f"  ❌ CLIP text-target failed: {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("  CALIBRATION SUMMARY — domain_topology.py DA thresholds")
    print(f"{'='*60}")
    print(f"  {'Encoder':<30} {'Threshold':>9} {'F1':>6} {'Status'}")
    print(f"  {'-'*58}")
    for key, r in results.items():
        status = "✅" if r['f1'] > 0.7 else "⚠️ " if r['f1'] > 0.4 else "❌"
        print(f"  {r['name']:<30} {r['threshold']:>9.4f} "
              f"{r['f1']:>6.3f} {status}")

    # Save results
    out_path = Path("vlm_calibration_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")

    # Print domain_topology.py update snippet
    print(f"\n{'='*60}")
    print("  ADD TO domain_topology.py:")
    print(f"{'='*60}")
    for key, r in results.items():
        print(f"\n  # {r['name']}")
        print(f"  da_threshold = {r['threshold']:.4f}  "
              f"# F1={r['f1']:.3f}, stable_mean={r['stable_mean']:.4f}")


def _make_outdoor():
    """Fallback synthetic outdoor frame."""
    size = 224
    img = Image.new("RGB", (size, size))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, size, size//2], fill=(135, 206, 235))
    d.rectangle([0, size//2, size, size], fill=(34, 139, 34))
    d.ellipse([size//3, size//4, 2*size//3, size//2-10], fill=(255, 255, 0))
    return img


if __name__ == "__main__":
    main()
