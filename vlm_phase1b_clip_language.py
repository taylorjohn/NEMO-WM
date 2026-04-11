"""
vlm_phase1b_clip_language.py — Language-Conditioned CLIP Neuromodulation
=========================================================================
Phase 1b of VLM integration plan.

Tests whether text grounding changes the DA (surprise) signal in CLIP.
Key question: if the text query MISMATCHES the image, does DA rise?

If yes: CLIP acts as a language-grounded anomaly detector.
        Text = "outdoor navigation" + image of robot arm → high DA
        Text = "outdoor navigation" + image of outdoor scene → low DA
        This is a new capability: language-conditioned regime switching.

If no: CLIP visual features dominate over text grounding.
       Text conditioning has no effect on neuromodulator signals.

Either result is informative for VLM integration design.

Test matrix:
  4 image types × 4 text queries × (matched + mismatched) = 32 combinations
  Plus: pure visual (no text) baseline for each image

Usage:
    python vlm_phase1b_clip_language.py
    python vlm_phase1b_clip_language.py --real-frames   # use RECON HDF5 frames

Author: John Taylor — github.com/taylorjohn
Sprint: VLM Phase 1b
"""

import argparse
import glob
import io
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

# ── Image generators (reused from phase1) ─────────────────────────────────────

def make_image(scene_type: str, size: int = 224) -> Image.Image:
    img = Image.new("RGB", (size, size), (0, 0, 0))
    d = ImageDraw.Draw(img)
    if scene_type == "outdoor":
        d.rectangle([0, 0, size, size//2], fill=(135, 206, 235))
        d.rectangle([0, size//2, size, size], fill=(34, 139, 34))
        d.ellipse([size//3, size//4, 2*size//3, size//2-10], fill=(255, 255, 0))
    elif scene_type == "indoor":
        d.rectangle([0, 0, size, size], fill=(245, 245, 220))
        d.rectangle([0, 3*size//4, size, size], fill=(139, 90, 43))
        d.rectangle([size//4, size//4, 3*size//4, 2*size//3], fill=(100, 100, 180))
    elif scene_type == "manipulation":
        d.rectangle([0, 0, size, size], fill=(50, 50, 50))
        d.rectangle([size//2-5, size//4, size//2+5, 3*size//4], fill=(200, 200, 200))
        d.rectangle([size//3, 2*size//3, 2*size//3, size-10], fill=(255, 100, 0))
    elif scene_type == "noise":
        arr = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        return Image.fromarray(arr)
    return img


def load_recon_frames(hdf5_dir: str, n: int = 4) -> dict:
    """Load real frames from RECON HDF5 files."""
    import h5py
    files = sorted(glob.glob(f"{hdf5_dir}/*.hdf5"))[:n]
    frames = {}
    for i, path in enumerate(files):
        try:
            with h5py.File(path) as hf:
                jpeg = bytes(hf["images"]["rgb_left"][0])
                img = Image.open(io.BytesIO(jpeg)).convert("RGB")
                frames[f"recon_{i}"] = img
        except Exception as e:
            print(f"  Skipping {path}: {e}")
    return frames


# ── CLIP multimodal encoder ────────────────────────────────────────────────────

class CLIPMultimodalEncoder:
    """
    CLIP encoder with both vision and language paths.
    Produces three types of embeddings:
      - visual_only: image features only (baseline)
      - text_only:   text features only
      - joint:       average of visual and text (multimodal)
    """

    def __init__(self):
        from transformers import CLIPProcessor, CLIPModel
        model_id = "openai/clip-vit-base-patch32"
        print("Loading CLIP ViT-B/32 (cached)...")
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id)
        self.model.eval()
        print("  ✅ CLIP loaded\n")

    def encode_visual(self, img: Image.Image) -> torch.Tensor:
        """Visual features projected into shared CLIP space (512-dim)."""
        inputs = self.processor(images=img, return_tensors="pt")
        with torch.no_grad():
            out = self.model.vision_model(pixel_values=inputs["pixel_values"])
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                z = out.pooler_output
            else:
                z = out.last_hidden_state[:, 0, :]  # CLS token
            # Project into shared embedding space (768 → 512)
            z = self.model.visual_projection(z)
        return F.normalize(z.squeeze(0).float(), dim=0)

    def encode_text(self, text: str) -> torch.Tensor:
        """Text features projected into shared CLIP space (512-dim)."""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        with torch.no_grad():
            out = self.model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                z = out.pooler_output
            else:
                z = out.last_hidden_state[:, -1, :]  # EOS token
            # Project into shared embedding space (512 → 512)
            z = self.model.text_projection(z)
        return F.normalize(z.squeeze(0).float(), dim=0)

    def encode_joint(self, img: Image.Image, text: str) -> torch.Tensor:
        """Joint vision+language embedding — normalised average."""
        zv = self.encode_visual(img)
        zt = self.encode_text(text)
        return F.normalize((zv + zt) / 2.0, dim=0)

    def similarity(self, img: Image.Image, text: str) -> float:
        """CLIP similarity score between image and text."""
        zv = self.encode_visual(img)
        zt = self.encode_text(text)
        # Apply CLIP projection layers for proper similarity
        with torch.no_grad():
            try:
                zv_proj = self.model.visual_projection(zv.unsqueeze(0)).squeeze(0)
                zt_proj = self.model.text_projection(zt.unsqueeze(0)).squeeze(0)
                zv_proj = F.normalize(zv_proj, dim=0)
                zt_proj = F.normalize(zt_proj, dim=0)
                return float(torch.dot(zv_proj, zt_proj).item())
            except Exception:
                return float(torch.dot(zv, zt).item())


# ── Minimal neuromodulator (standalone) ───────────────────────────────────────

class Neuromodulator:
    def __init__(self):
        self.z_history = []
        self.loss_history = []
        self.baseline = None

    def update(self, z_pred: torch.Tensor, z_target: torch.Tensor) -> dict:
        zp = F.normalize(z_pred.float().flatten(), dim=0)
        zt = F.normalize(z_target.float().flatten(), dim=0)

        da  = float(1.0 - torch.dot(zp, zt).clamp(-1, 1))
        ne  = float(np.clip(torch.norm(zp - zt).item() /
                            (torch.norm(zt).item() + 1e-6), 0, 1))
        mse = float(F.mse_loss(zp, zt).item())

        if len(self.z_history) >= 3:
            stack = torch.stack(self.z_history[-8:])
            var = stack.var(dim=0).mean().item()
            sht = float(np.clip(1.0 - var * 10, 0, 1))
        else:
            sht = 0.5

        self.loss_history.append(mse)
        if self.baseline is None and len(self.loss_history) >= 5:
            self.baseline = np.mean(self.loss_history[:5]) * 0.95
        if self.baseline and len(self.loss_history) >= 3:
            cort = float(np.clip(
                (np.mean(self.loss_history[-3:]) - self.baseline) /
                (self.baseline + 1e-6), 0, 1))
        else:
            cort = 0.0

        self.z_history.append(zt.detach())
        if len(self.z_history) > 20:
            self.z_history.pop(0)

        regime = "REOBSERVE" if da > 0.01 else "EXPLOIT"
        return {"da": round(da, 4), "ne": round(ne, 4),
                "sht": round(sht, 4), "cort": round(cort, 4),
                "mse": round(mse, 4), "regime": regime}


# ── Test matrix ───────────────────────────────────────────────────────────────

TEXT_QUERIES = {
    "outdoor_nav":   "outdoor robot navigation grass sky",
    "indoor_room":   "indoor room furniture sofa floor",
    "robot_arm":     "robot arm manipulation dark background",
    "random_noise":  "random noise static pixels",
}

IMAGE_TYPES = ["outdoor", "indoor", "manipulation", "noise"]

# Which text matches which image (ground truth)
MATCHES = {
    "outdoor":     "outdoor_nav",
    "indoor":      "indoor_room",
    "manipulation":"robot_arm",
    "noise":       "random_noise",
}


def run_language_test(clip: CLIPMultimodalEncoder) -> list:
    """
    For each image type:
      - Compute CLIP similarity score vs each text query
      - Compute DA for visual_only, text_only, joint embeddings
      - Compare matched vs mismatched text
    """
    results = []

    images = {t: make_image(t) for t in IMAGE_TYPES}

    for img_type, img in images.items():
        matched_text_key = MATCHES[img_type]
        z_visual = clip.encode_visual(img)

        for text_key, text in TEXT_QUERIES.items():
            z_text  = clip.encode_text(text)
            z_joint = clip.encode_joint(img, text)
            sim     = clip.similarity(img, text)
            match   = (text_key == matched_text_key)

            # DA: visual vs text (how surprising is this text given this image?)
            da_vis_vs_text = float(1.0 - torch.dot(z_visual, z_text).clamp(-1, 1))

            # DA: visual vs joint (how much does text change the visual embedding?)
            da_text_effect = float(1.0 - torch.dot(z_visual, z_joint).clamp(-1, 1))

            results.append({
                "image":         img_type,
                "text":          text_key,
                "match":         match,
                "clip_sim":      round(sim, 4),
                "da_vis_text":   round(da_vis_vs_text, 4),
                "da_text_effect":round(da_text_effect, 4),
                "regime":        "REOBSERVE" if da_vis_vs_text > 0.05 else "EXPLOIT",
            })

    return results


def run_sequential_test(clip: CLIPMultimodalEncoder,
                        images: dict, text_query: str) -> list:
    """
    Sequential test: feed frames in order with a fixed text query.
    Measures how DA changes as scenes change relative to the text anchor.
    """
    neuro = Neuromodulator()
    results = []
    img_list = list(images.items())
    z_text = clip.encode_text(text_query)

    for i in range(len(img_list) - 1):
        name_t,  img_t  = img_list[i]
        name_t1, img_t1 = img_list[i + 1]

        # Three modes
        zv_t  = clip.encode_visual(img_t)
        zv_t1 = clip.encode_visual(img_t1)
        zj_t  = clip.encode_joint(img_t, text_query)
        zj_t1 = clip.encode_joint(img_t1, text_query)

        s_visual = neuro.update(zv_t, zv_t1)
        neuro2 = Neuromodulator()
        s_joint  = neuro2.update(zj_t, zj_t1)

        results.append({
            "step":      f"{name_t}→{name_t1}",
            "text":      text_query[:30],
            "da_visual": s_visual["da"],
            "da_joint":  s_joint["da"],
            "da_delta":  round(s_joint["da"] - s_visual["da"], 4),
            "regime_v":  s_visual["regime"],
            "regime_j":  s_joint["regime"],
        })

    return results


# ── Print functions ───────────────────────────────────────────────────────────

def print_language_matrix(results: list):
    print(f"\n{'='*80}")
    print("  LANGUAGE-CONDITIONED DA MATRIX")
    print("  DA_vis_text = surprise of text given image (high = mismatch)")
    print("  DA_text_fx  = how much text shifts the visual embedding")
    print(f"{'='*80}")
    print(f"  {'Image':<14} {'Text Query':<16} {'Match':<6} "
          f"{'CLIP_sim':>8} {'DA_vis_text':>11} {'DA_text_fx':>10} {'Regime':<12}")
    print(f"  {'-'*78}")

    for r in results:
        match_str = "✓ YES" if r["match"] else "✗ NO "
        print(f"  {r['image']:<14} {r['text']:<16} {match_str:<6} "
              f"{r['clip_sim']:>8.4f} {r['da_vis_text']:>11.4f} "
              f"{r['da_text_effect']:>10.4f} {r['regime']:<12}")

    # Summary
    matched = [r for r in results if r["match"]]
    mismatched = [r for r in results if not r["match"]]
    print(f"\n  Matched    avg DA_vis_text: {np.mean([r['da_vis_text'] for r in matched]):.4f}")
    print(f"  Mismatched avg DA_vis_text: {np.mean([r['da_vis_text'] for r in mismatched]):.4f}")
    delta = np.mean([r['da_vis_text'] for r in mismatched]) - \
            np.mean([r['da_vis_text'] for r in matched])
    print(f"  Mismatch DA premium:        {delta:+.4f}  "
          f"{'✅ language grounding works' if delta > 0.02 else '⚠️  weak or no effect'}")


def print_sequential(results: list, text_query: str):
    print(f"\n{'='*80}")
    print(f"  SEQUENTIAL TEST — text anchor: '{text_query[:40]}'")
    print(f"  DA_delta = joint - visual  (positive = text amplifies surprise)")
    print(f"{'='*80}")
    print(f"  {'Step':<25} {'DA_visual':>9} {'DA_joint':>9} "
          f"{'DA_delta':>9} {'Regime_V':<12} {'Regime_J':<12}")
    print(f"  {'-'*78}")
    for r in results:
        delta_str = f"{r['da_delta']:+.4f}"
        print(f"  {r['step']:<25} {r['da_visual']:>9.4f} {r['da_joint']:>9.4f} "
              f"{delta_str:>9} {r['regime_v']:<12} {r['regime_j']:<12}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-frames", action="store_true",
                        help="Use RECON HDF5 frames instead of synthetic")
    parser.add_argument("--hdf5-dir", default="recon_data/recon_release",
                        help="RECON HDF5 directory for --real-frames")
    a = parser.parse_args()

    clip = CLIPMultimodalEncoder()

    # ── Test 1: Language matrix ───────────────────────────────────────────────
    print("Test 1: Language-conditioned DA matrix (4 images × 4 text queries)")
    results = run_language_test(clip)
    print_language_matrix(results)

    # ── Test 2: Sequential with text anchor ──────────────────────────────────
    print("\n\nTest 2: Sequential scene changes with fixed text anchor")

    if a.real_frames:
        print(f"  Loading real RECON frames from {a.hdf5_dir}...")
        images = load_recon_frames(a.hdf5_dir, n=5)
        if not images:
            print("  No frames loaded — falling back to synthetic")
            images = {t: make_image(t) for t in IMAGE_TYPES}
    else:
        images = {t: make_image(t) for t in IMAGE_TYPES}

    # Run with each text query as anchor
    for text_key, text in TEXT_QUERIES.items():
        seq = run_sequential_test(clip, images, text)
        print_sequential(seq, text)

    # ── Test 3: Text-as-target (correct architecture) ────────────────────────
    print("\n\nTest 3: Text-as-TARGET neuromodulation (correct architecture)")
    print("  z_pred = current visual frame")
    print("  z_target = text description of expected scene")
    print("  DA = how surprising is this frame given the text expectation?\n")

    image_sequence = list({t: make_image(t) for t in IMAGE_TYPES}.items())

    for text_key, text in TEXT_QUERIES.items():
        z_text = clip.encode_text(text)
        print(f"  Text anchor: '{text}'")
        print(f"  {'Frame':<16} {'DA_text_target':>14} {'CLIP_sim':>8} {'Regime':<12}")
        print(f"  {'-'*55}")
        for img_name, img in image_sequence:
            z_vis = clip.encode_visual(img)
            # DA: how far is visual from text expectation
            da = float(1.0 - torch.dot(z_vis, z_text).clamp(-1, 1))
            sim = float(torch.dot(z_vis, z_text).item())
            regime = "REOBSERVE" if da > 0.75 else "EXPLOIT"
            match = "✓" if MATCHES[img_name] == text_key else " "
            print(f"  {match} {img_name:<14} {da:>14.4f} {sim:>8.4f} {regime:<12}")
        print()

    # ── Test 4: Matched vs mismatched summary ─────────────────────────────────
    print(f"\n\n{'='*80}")
    print("  KEY FINDING SUMMARY")
    print(f"{'='*80}")

    matched    = [r for r in results if r["match"]]
    mismatched = [r for r in results if not r["match"]]
    avg_m  = np.mean([r["da_vis_text"] for r in matched])
    avg_mm = np.mean([r["da_vis_text"] for r in mismatched])
    delta  = avg_mm - avg_m

    print(f"\n  Matched text+image pairs:     avg DA = {avg_m:.4f}")
    print(f"  Mismatched text+image pairs:  avg DA = {avg_mm:.4f}")
    print(f"  Mismatch premium:             Δ DA = {delta:+.4f}")
    print()

    if delta > 0.05:
        print("  ✅ STRONG language grounding effect")
        print("     Text mismatch reliably raises DA — language-conditioned")
        print("     anomaly detection confirmed.")
        print("     Implication: NeMo-WM + CLIP = text-queryable world model")
    elif delta > 0.02:
        print("  ⚠️  MODERATE language grounding effect")
        print("     Text conditioning has some effect but needs stronger signal.")
        print("     Try: use joint embedding as z_target instead of baseline.")
    else:
        print("  ❌ WEAK language grounding effect")
        print("     Visual features dominate. Text conditioning not effective")
        print("     at the embedding level with this CLIP variant.")
        print("     Try: larger CLIP (ViT-L/14) or SigLIP.")

    print()
    print("Next step:")
    print("  python vlm_phase1b_clip_language.py --real-frames")
    print("  (uses actual RECON navigation frames for ground truth)")


if __name__ == "__main__":
    main()
