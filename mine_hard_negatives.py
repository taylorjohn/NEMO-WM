"""
mine_hard_negatives.py — Sprint 9a: Automatic Hard Negative Mining
==================================================================
Finds the best null texts for language grounding by scoring a diverse
sentence corpus against the RECON mean visual embedding via CLIP.

The problem with manual null texts (chocolate cake, Shakespeare):
  These are obviously non-navigation. RECON's CLIPBridge easily separates
  them in absolute similarity space but the test metric (ratio) explodes
  because base similarities are near zero.

Hard negatives are texts that CLIP maps NEAR navigation visuals:
  "a car drives down the highway" — outdoor, motion, road
  "satellite view of a city" — outdoor, spatial, aerial
  These are genuinely confusable and test real boundary learning.

Method:
  1. Load CLIP ViT-B/32 (already cached)
  2. Encode RECON frames → compute mean visual embedding
  3. Encode diverse sentence corpus → cosine sim vs mean embedding
  4. Top-K = hard negatives (confusable with navigation)
  5. Bottom-K = easy negatives (clearly non-navigation)
  6. Write ranked list to hard_negatives.json

Run:
    python mine_hard_negatives.py \
        --hdf5-dir recon_data\recon_release \
        --n-files  30 \
        --top-k    20 \
        --output   hard_negatives.json

Author: John Taylor — github.com/taylorjohn
Date:   2026-04-06
"""

import argparse
import glob
import io
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import clip as clip_module
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

DEVICE = torch.device("cpu")

# ── Diverse sentence corpus ───────────────────────────────────────────────────
# 200 sentences across: outdoor scenes, vehicles, urban environments,
# nature, abstract concepts, food, literature, science, sports, home
CORPUS = [
    # Outdoor / potentially confusable
    "a car drives down the highway at high speed",
    "a bicycle lane on a city street",
    "a parking lot with rows of vehicles",
    "a satellite map view of urban streets",
    "drone footage of a city from above",
    "traffic lights changing at an intersection",
    "a pedestrian crossing a busy road",
    "a bus stop on the side of the road",
    "a highway overpass with vehicles underneath",
    "a dirt path through a forest",
    "a sidewalk next to a building",
    "a crosswalk painted on asphalt",
    "a park with trees and grass",
    "a sports field seen from above",
    "a construction site with machinery",
    "a loading dock behind a warehouse",
    "an airport runway with planes",
    "a golf course with green fairways",
    "a river running through a city",
    "a bridge over water",
    "a shopping mall parking garage",
    "a residential street with houses",
    "a fire truck parked on a street",
    "a delivery truck making a stop",
    "a motorcycle on a country road",
    "a train station platform",
    "a boat harbor with docks",
    "a solar panel array on a rooftop",
    "a swimming pool in a backyard",
    "an outdoor basketball court",
    # Nature
    "a mountain landscape with snow",
    "a beach with ocean waves",
    "a dense forest with tall trees",
    "a desert with sand dunes",
    "a field of wheat in summer",
    "a waterfall in a jungle",
    "a sunset over the ocean",
    "a storm cloud over a plain",
    "a volcano with smoke rising",
    "an arctic landscape with ice",
    # Urban / indoor
    "a crowded shopping mall interior",
    "a restaurant kitchen with chefs",
    "an office building lobby",
    "a hospital hallway",
    "a school gymnasium",
    "a factory floor with machines",
    "a library with rows of books",
    "a supermarket aisle",
    "a subway station platform",
    "a hotel lobby",
    # Vehicles / motion
    "a Formula 1 race car on a track",
    "a cargo ship at sea",
    "a fighter jet in flight",
    "a hot air balloon over fields",
    "a submarine underwater",
    "a space shuttle launching",
    "a helicopter flying over a city",
    "a sailboat on a lake",
    "a tank on a military training ground",
    "an ambulance with sirens",
    # Abstract / clearly non-navigation
    "a recipe for chocolate cake with frosting",
    "sonnets by william shakespeare",
    "the quick brown fox jumps over the lazy dog",
    "a mathematical formula on a blackboard",
    "a musical score for a symphony",
    "the periodic table of elements",
    "a chess board mid-game",
    "abstract oil painting with bright colors",
    "a microscope image of cells",
    "a bar chart showing quarterly revenue",
    "a medical diagram of the human heart",
    "a world map showing continents",
    "stock market prices fluctuating",
    "a computer circuit board closeup",
    "ancient hieroglyphics on stone",
    "a DNA double helix structure",
    "a galaxy photographed by telescope",
    "a coral reef underwater",
    "a polar bear on ice",
    "a flock of birds in formation",
    # Food / domestic
    "a bowl of pasta with tomato sauce",
    "a kitchen counter with ingredients",
    "a birthday cake with candles",
    "a cup of coffee on a desk",
    "a fresh salad with vegetables",
    "a pizza coming out of an oven",
    "a thanksgiving turkey dinner",
    "a smoothie with fruit",
    "a barbecue grill with meat",
    "a sushi plate with chopsticks",
    # Sports / activity
    "a soccer match in a stadium",
    "a basketball player dunking",
    "a swimmer in a pool",
    "a tennis match on clay court",
    "a marathon race on a city street",
    "a ski slope with skiers",
    "a boxing match in a ring",
    "a yoga class in a studio",
    "a rock climbing wall",
    "a cycling race in the mountains",
    # Technology / science
    "a server room with blinking lights",
    "a robot arm in a factory",
    "a 3D printer making an object",
    "a telescope observatory at night",
    "a laboratory with test tubes",
    "a wind turbine farm",
    "a nuclear power plant cooling tower",
    "a satellite in orbit around earth",
    "a data center with cooling systems",
    "an MRI machine in a hospital",
]


def encode_corpus(clip_model, clip_module, corpus: list[str],
                  batch_size: int = 32) -> np.ndarray:
    """Encode all sentences. Returns (N, 512) unit-normalised."""
    all_embs = []
    for i in range(0, len(corpus), batch_size):
        batch = corpus[i:i+batch_size]
        with torch.no_grad():
            tokens = clip_module.tokenize(batch).to(DEVICE)
            embs   = clip_model.encode_text(tokens).float()
            embs   = F.normalize(embs, dim=-1)
        all_embs.append(embs.numpy())
    return np.concatenate(all_embs, axis=0)   # (N, 512)


def compute_recon_mean(clip_model, hdf5_dir: str, n_files: int = 30) -> np.ndarray:
    """
    Compute mean CLIP visual embedding over RECON frames.
    Returns (512,) unit-normalised mean.
    """
    import h5py
    try:
        import torchvision.transforms as T
        preprocess = T.Compose([
            T.Resize(224), T.CenterCrop(224), T.ToTensor(),
            T.Normalize([0.48145466,0.4578275,0.40821073],
                        [0.26862954,0.26130258,0.27577711])
        ])
    except ImportError:
        # Fallback: simple resize + normalize
        def preprocess(img):
            img = img.resize((224, 224))
            arr = np.array(img, dtype=np.float32) / 255.0
            mean = np.array([0.48145466, 0.4578275, 0.40821073])
            std  = np.array([0.26862954, 0.26130258, 0.27577711])
            arr  = (arr - mean) / std
            return torch.from_numpy(arr).permute(2,0,1).float()

    files = sorted(glob.glob(f"{hdf5_dir}/*.hdf5"))[:n_files]
    if not files:
        files = sorted(glob.glob(f"{hdf5_dir}/**/*.hdf5", recursive=True))[:n_files]

    print(f"  Computing RECON mean embedding from {len(files)} files...")
    all_vis = []

    for path in files:
        try:
            with h5py.File(path, 'r') as hf:
                imgs = hf['images']['rgb_left']
                idx  = random.randint(0, len(imgs)-1)
                img  = Image.open(io.BytesIO(bytes(imgs[idx]))).convert('RGB')
                x    = preprocess(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    vis_emb = clip_model.encode_image(x).float()
                    vis_emb = F.normalize(vis_emb, dim=-1)
                all_vis.append(vis_emb.squeeze(0).numpy())
        except Exception:
            pass

    if not all_vis:
        raise RuntimeError("No RECON frames loaded — check hdf5_dir")

    mean_emb = np.mean(all_vis, axis=0)
    mean_emb /= (np.linalg.norm(mean_emb) + 1e-8)
    print(f"  Mean embedding computed from {len(all_vis)} frames")
    return mean_emb


def mine(args):
    if not CLIP_AVAILABLE:
        raise ImportError(
            "Install CLIP: pip install git+https://github.com/openai/CLIP.git"
        )

    print("\nSprint 9a — Hard Negative Mining")
    print("=" * 60)
    print(f"  Corpus:    {len(CORPUS)} sentences")
    print(f"  RECON:     {args.n_files} files")
    print(f"  Top-K:     {args.top_k} hard negatives")
    print("=" * 60)

    t0 = time.perf_counter()

    # Load CLIP
    print("\n  Loading CLIP ViT-B/32...")
    clip_model, _ = clip_module.load("ViT-B/32", device=DEVICE)
    clip_model.eval()
    clip_ms = (time.perf_counter() - t0) * 1000
    print(f"  Loaded in {clip_ms:.0f}ms")

    # Compute RECON mean visual embedding
    recon_mean = compute_recon_mean(clip_model, args.hdf5_dir, args.n_files)

    # Encode corpus
    print(f"\n  Encoding {len(CORPUS)} sentences...")
    t1 = time.perf_counter()
    text_embs = encode_corpus(clip_model, clip_module, CORPUS)  # (N, 512)
    enc_ms = (time.perf_counter() - t1) * 1000
    print(f"  Encoded in {enc_ms:.0f}ms")

    # Score against RECON mean
    scores = text_embs @ recon_mean    # (N,) cosine similarities

    # Rank
    ranked_idx  = np.argsort(scores)[::-1]
    hard_neg    = [(CORPUS[i], float(scores[i])) for i in ranked_idx[:args.top_k]]
    easy_neg    = [(CORPUS[i], float(scores[i])) for i in ranked_idx[-args.top_k:]]
    mid_idx     = ranked_idx[len(ranked_idx)//2-3:len(ranked_idx)//2+3]
    mid_neg     = [(CORPUS[i], float(scores[i])) for i in mid_idx]

    # Print results
    print(f"\n{'='*60}")
    print(f"  TOP {args.top_k} HARD NEGATIVES (most confusable with navigation)")
    print(f"  These score highest against RECON visual embeddings")
    print(f"{'='*60}")
    for i, (txt, score) in enumerate(hard_neg):
        print(f"  {i+1:2d}. [{score:+.3f}] {txt}")

    print(f"\n  MIDDLE {len(mid_neg)} (moderate similarity)")
    for txt, score in mid_neg:
        print(f"       [{score:+.3f}] {txt}")

    print(f"\n  BOTTOM {args.top_k} EASY NEGATIVES (clearly non-navigation)")
    for i, (txt, score) in enumerate(easy_neg[:5]):
        print(f"  {i+1:2d}. [{score:+.3f}] {txt}")
    print(f"  ...")

    print(f"\n  Score statistics:")
    print(f"    Max (hardest): {scores.max():.3f} — '{CORPUS[scores.argmax()]}'")
    print(f"    Min (easiest): {scores.min():.3f} — '{CORPUS[scores.argmin()]}'")
    print(f"    Mean: {scores.mean():.3f}")
    print(f"    Std:  {scores.std():.3f}")

    # Save
    output = {
        "recon_files_used":  args.n_files,
        "corpus_size":       len(CORPUS),
        "top_k":             args.top_k,
        "hard_negatives":    hard_neg,
        "easy_negatives":    easy_neg,
        "full_ranking":      [(CORPUS[i], float(scores[i]))
                              for i in ranked_idx],
    }
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"\n  Saved: {args.output}")
    print(f"\n  Use top-{args.top_k} hard negatives in Sprint 6f:")
    print(f"  NULL_TEXTS = [")
    for txt, score in hard_neg[:8]:
        print(f'      "{txt}",  # score={score:.3f}')
    print(f"  ]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sprint 9a — Hard Negative Mining")
    parser.add_argument("--hdf5-dir",  default="recon_data/recon_release")
    parser.add_argument("--n-files",   type=int,   default=30)
    parser.add_argument("--top-k",     type=int,   default=20)
    parser.add_argument("--output",    default="hard_negatives.json")
    args = parser.parse_args()
    mine(args)
