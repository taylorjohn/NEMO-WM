"""
certify_student.py — CORTEX-16 Semantic Axis Certification

Ports the CORTEX-12 certification toolchain to the StudentEncoder architecture.
Produces a human-readable JSON certificate that makes the semantic axis claims
in ARCHITECTURE.md and README.md formally defensible.

What it certifies:
  Each of the four 32-D semantic axis subspaces is tested against synthetic
  validation data that varies one attribute at a time (shape, size, depth,
  velocity proxy). A certification score is computed per axis and an overall
  pass/fail verdict is issued.

  The certificate is a standalone JSON file that:
    - Documents the tested model checkpoint hash
    - Records per-axis scores and thresholds
    - Records overall pass/fail
    - Is portable — can be diff'd between checkpoints to track axis drift

Certificate format (matches CORTEX-12 JSON structure):
  {
    "model_checkpoint": "cortex_student_phase2_final.pt",
    "checkpoint_hash":  "abc123...",
    "timestamp":        "2026-03-16T...",
    "overall_pass":     true,
    "axes": {
      "shape":    {"score": 0.87, "threshold": 0.70, "pass": true},
      "size":     {"score": 0.81, "threshold": 0.65, "pass": true},
      "depth":    {"score": 0.72, "threshold": 0.60, "pass": true},
      "velocity": {"score": 0.68, "threshold": 0.55, "pass": true}
    },
    "stability": {
      "same_mean": 0.9887, "same_std": 0.0027,
      "diff_mean": 0.5720, "diff_std": 0.0100
    }
  }

Usage:
  python certify_student.py --weights cortex_student_phase2_final.pt
  python certify_student.py --weights cortex_student_phase2_final.pt \\
      --out cortex_certificate_v15.json
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw

from student_encoder import StudentEncoder


# =============================================================================
# Transforms (must match training)
# =============================================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CERT_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# =============================================================================
# Synthetic Certification Data Generator
# Each generator varies only the target attribute, holding others fixed.
# =============================================================================
def render(shape="circle", color=(180, 60, 60), size=80, pos=(112, 112)):
    img  = Image.new("RGB", (224, 224), (128, 128, 128))
    draw = ImageDraw.Draw(img)
    cx, cy = pos
    r = size // 2
    if shape == "circle":
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=color)
    elif shape == "square":
        draw.rectangle([cx-r, cy-r, cx+r, cy+r], fill=color)
    elif shape == "triangle":
        draw.polygon([(cx, cy-r), (cx-r, cy+r), (cx+r, cy+r)], fill=color)
    return CERT_TRANSFORM(img)


def generate_shape_pairs(n=30):
    """Pairs that differ only in shape — should be separated by shape axis."""
    positives, negatives = [], []
    for _ in range(n):
        c = tuple(np.random.randint(80, 200, 3).tolist())
        s = int(np.random.randint(60, 100))
        positives.append((render("circle",   c, s), render("circle",   c, s)))
        negatives.append((render("circle",   c, s), render("square",   c, s)))
        negatives.append((render("circle",   c, s), render("triangle", c, s)))
    return positives, negatives


def generate_size_pairs(n=30):
    """Pairs that differ only in size — should be separated by size axis."""
    positives, negatives = [], []
    for _ in range(n):
        c = tuple(np.random.randint(80, 200, 3).tolist())
        positives.append((render("circle", c, 80), render("circle", c, 80)))
        negatives.append((render("circle", c, 80), render("circle", c, 40)))
        negatives.append((render("circle", c, 80), render("circle", c, 100)))
    return positives, negatives


def generate_depth_pairs(n=30):
    """
    Depth proxy: simulate depth via brightness/contrast variation.
    Brighter objects = closer; darker = farther.
    """
    positives, negatives = [], []
    for _ in range(n):
        bright = tuple(np.random.randint(160, 220, 3).tolist())
        dark   = tuple(np.clip(np.array(bright) // 3, 20, 80).tolist())
        positives.append((render("circle", bright, 80), render("circle", bright, 80)))
        negatives.append((render("circle", bright, 80), render("circle", dark,   80)))
    return positives, negatives


def generate_velocity_pairs(n=30):
    """
    Velocity proxy: simulate motion via position shift.
    Same position = zero velocity; large shift = high velocity.
    """
    positives, negatives = [], []
    for _ in range(n):
        c = tuple(np.random.randint(80, 200, 3).tolist())
        p1 = (112, 112)
        p2 = (int(np.random.randint(140, 180)), int(np.random.randint(140, 180)))
        positives.append((render("circle", c, 70, p1), render("circle", c, 70, p1)))
        negatives.append((render("circle", c, 70, p1), render("circle", c, 70, p2)))
    return positives, negatives


# =============================================================================
# Axis Subspace Extractor
# Slices the 128-D latent into the four 32-D semantic subspaces.
# =============================================================================
AXIS_SLICES = {
    "shape":    (0,  32),
    "size":     (32, 64),
    "depth":    (64, 96),
    "velocity": (96, 128),
}


def axis_subspace(z: torch.Tensor, axis: str) -> torch.Tensor:
    """Extract and re-normalise the 32-D subspace for a given axis."""
    s, e = AXIS_SLICES[axis]
    sub  = z[:, s:e]
    return F.normalize(sub, dim=-1)


# =============================================================================
# Axis Certification Score
# Measures discriminability: how much better the axis separates
# positive pairs (same concept) vs negative pairs (differing on that attribute).
#
# Score = mean_sim(positives) - mean_sim(negatives)
# Range: [-1, 1]. Higher = better axis discrimination.
# Threshold: axis-specific (see AXIS_THRESHOLDS below).
# =============================================================================
AXIS_THRESHOLDS = {
    "shape":    0.70,   # Shape encodes first in JEPA training — higher threshold
    "size":     0.65,
    "depth":    0.55,   # Depth proxy is noisier — lower threshold
    "velocity": 0.50,   # Velocity proxy is noisiest
}


@torch.no_grad()
def certify_axis(
    model: StudentEncoder,
    axis: str,
    pair_generator,
    n_pairs: int = 30,
) -> dict:
    positives, negatives = pair_generator(n=n_pairs)

    pos_sims, neg_sims = [], []

    for (a, b) in positives:
        za = axis_subspace(model(a.unsqueeze(0)), axis)
        zb = axis_subspace(model(b.unsqueeze(0)), axis)
        pos_sims.append(F.cosine_similarity(za, zb).item())

    for (a, b) in negatives:
        za = axis_subspace(model(a.unsqueeze(0)), axis)
        zb = axis_subspace(model(b.unsqueeze(0)), axis)
        neg_sims.append(F.cosine_similarity(za, zb).item())

    score     = np.mean(pos_sims) - np.mean(neg_sims)
    threshold = AXIS_THRESHOLDS[axis]
    passed    = score >= threshold

    return {
        "score":         round(float(score), 4),
        "threshold":     threshold,
        "pass":          passed,
        "pos_mean":      round(float(np.mean(pos_sims)), 4),
        "neg_mean":      round(float(np.mean(neg_sims)), 4),
        "pos_std":       round(float(np.std(pos_sims)),  4),
        "neg_std":       round(float(np.std(neg_sims)),  4),
        "n_positive_pairs": len(positives),
        "n_negative_pairs": len(negatives),
    }


# =============================================================================
# Stability Metrics (same as evaluate_cortex.py Test 2)
# Recorded in the certificate for cross-checkpoint comparison.
# =============================================================================
@torch.no_grad()
def compute_stability(model: StudentEncoder, n_per_concept: int = 12) -> dict:
    concepts = [
        {"shape": "circle",   "color": (220, 50,  50),  "size": 80},
        {"shape": "square",   "color": (50,  220, 50),  "size": 80},
        {"shape": "triangle", "color": (50,  50,  220), "size": 80},
        {"shape": "circle",   "color": (220, 50,  50),  "size": 45},
    ]

    embeddings = []
    for c in concepts:
        frames = torch.stack([
            render(c["shape"], c["color"], c["size"],
                   pos=(112 + np.random.randint(-4, 4), 112 + np.random.randint(-4, 4)))
            for _ in range(n_per_concept)
        ])
        embeddings.append(model(frames))

    same_sims, diff_sims = [], []
    for z in embeddings:
        for i in range(len(z)):
            for j in range(i+1, len(z)):
                same_sims.append(F.cosine_similarity(z[i:i+1], z[j:j+1]).item())
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            for zi in embeddings[i]:
                for zj in embeddings[j]:
                    diff_sims.append(
                        F.cosine_similarity(zi.unsqueeze(0), zj.unsqueeze(0)).item()
                    )

    return {
        "same_mean": round(float(np.mean(same_sims)), 4),
        "same_std":  round(float(np.std(same_sims)),  4),
        "diff_mean": round(float(np.mean(diff_sims)), 4),
        "diff_std":  round(float(np.std(diff_sims)),  4),
    }


# =============================================================================
# Main Certification Run
# =============================================================================
def certify(weights_path: str, output_path: str = "cortex_certificate.json") -> dict:
    print("\n" + "="*60)
    print("  CORTEX-16 SEMANTIC AXIS CERTIFICATION")
    print(f"  Checkpoint: {weights_path}")
    print("="*60 + "\n")

    # Load model
    model = StudentEncoder()
    ckpt  = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    # Checkpoint hash for reproducibility
    with open(weights_path, "rb") as f:
        ckpt_hash = hashlib.sha256(f.read()).hexdigest()[:16]

    # Run axis certifications
    axis_generators = {
        "shape":    generate_shape_pairs,
        "size":     generate_size_pairs,
        "depth":    generate_depth_pairs,
        "velocity": generate_velocity_pairs,
    }

    axes_results = {}
    for axis, generator in axis_generators.items():
        print(f"  Certifying axis: {axis}...")
        result = certify_axis(model, axis, generator, n_pairs=40)
        axes_results[axis] = result
        status = "✅ PASS" if result["pass"] else "❌ FAIL"
        print(f"    score={result['score']:.4f}  threshold={result['threshold']}  {status}")

    # Stability metrics
    print("\n  Computing stability metrics...")
    stability = compute_stability(model)
    print(f"    SAME mean = {stability['same_mean']}  std = {stability['same_std']}")
    print(f"    DIFF mean = {stability['diff_mean']}  std = {stability['diff_std']}")

    # Overall verdict
    all_passed    = all(r["pass"] for r in axes_results.values())
    stability_ok  = (stability["same_mean"] > 0.97 and stability["diff_mean"] < 0.65)
    overall_pass  = all_passed and stability_ok

    certificate = {
        "model_checkpoint": Path(weights_path).name,
        "checkpoint_hash":  ckpt_hash,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "overall_pass":     overall_pass,
        "axes":             axes_results,
        "stability":        stability,
        "axis_layout": {
            "dims_0_31":   "shape",
            "dims_32_63":  "size",
            "dims_64_95":  "depth",
            "dims_96_127": "velocity",
        },
    }

    with open(output_path, "w") as f:
        json.dump(certificate, f, indent=2)

    print("\n" + "="*60)
    print(f"  OVERALL: {'✅ CERTIFIED' if overall_pass else '❌ NOT CERTIFIED'}")
    print(f"  Certificate saved → {output_path}")
    print("="*60)

    return certificate


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CORTEX-16 Semantic Axis Certification")
    p.add_argument("--weights", required=True,
                   help="Path to trained StudentEncoder .pt checkpoint")
    p.add_argument("--out",     default="cortex_certificate.json",
                   help="Output path for JSON certificate")
    args = p.parse_args()

    certify(args.weights, args.out)
