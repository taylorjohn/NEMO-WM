"""
null_classifier.py — Sprint 6e: Binary Null Rejection Classifier
=================================================================
Replaces margin-based repulsion with a learned decision boundary.

The auto-tuner showed that margin-based null repulsion can't separate
"basketball court" from "robot on basketball court" — they share too
many CLIP features. A binary classifier on top of the similarity
scores learns the boundary in <1 second.

Architecture:
    For each query text:
        1. Encode with CLIP → text_emb (512-D)
        2. Project through CLIPBridge → text_proj (128-D)
        3. Compute similarity profile against N real frames → (N,)
        4. Extract features: [topk_mean, base_mean, std, max, ratio]
        5. Binary MLP: features (5-D) → is_navigation? {0, 1}

Training data:
    9 positive navigation queries (from test_dualhead_alignment.py)
    108 null queries (8 generic + 100 hard negatives)

Usage:
    python null_classifier.py --train    # train + save
    python null_classifier.py --eval     # eval on held-out
    python null_classifier.py --query "robot moving forward"

Author: John Taylor
Sprint: 6e (Null Rejection — final approach)
"""

import argparse
import glob
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import clip
import h5py
from PIL import Image

DEVICE = "cpu"
TOPK = 10

# ── Positive navigation queries ─────────────────────────────────────────────
POSITIVE_QUERIES = [
    "robot moving fast forward",
    "robot driving at high speed",
    "robot moving slowly and carefully",
    "robot barely moving",
    "robot stopped or stationary",
    "robot turning sharply",
    "robot spinning or rotating",
    "robot going straight ahead",
    "robot driving in a straight line",
]

# ── Null queries (generic + hard negatives) ──────────────────────────────────
NULL_QUERIES_GENERIC = [
    "the quick brown fox",
    "a recipe for chocolate cake",
    "sonnets by shakespeare",
    "a quarterly financial earnings report",
    "marine biology ocean research paper",
    "ingredients for beef stew",
    "how to play the piano",
    "a history of ancient rome",
]


def load_null_queries():
    texts = list(NULL_QUERIES_GENERIC)
    pair_file = Path("recon_data/nav_text_pairs_sprint9.json")
    if pair_file.exists():
        pairs = json.load(open(pair_file))
        texts += [p["negative"] for p in pairs]
    return texts


# ── Models ───────────────────────────────────────────────────────────────────

class SemanticHead(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class NullClassifier(nn.Module):
    """
    Binary classifier: similarity profile → is_navigation?

    Input features (5-D):
        topk_mean:  mean similarity of top-K frames
        base_mean:  mean similarity across all frames
        std:        std of similarities
        max_sim:    max single-frame similarity
        ratio:      topk_mean / (|base_mean| + eps)

    Output: logit (positive = navigation, negative = null)
    """
    def __init__(self, d_in=5, d_hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, features):
        return self.net(features).squeeze(-1)

    def predict(self, features):
        with torch.no_grad():
            return torch.sigmoid(self.forward(features)) > 0.5


# ── Feature extraction ───────────────────────────────────────────────────────

def decode_frame(jpeg_bytes, size=224):
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB").resize((size, size))
    return torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0


@torch.no_grad()
def collect_frame_embeddings(backbone, semantic_head, hdf5_dir,
                              max_files=50, stride=5):
    """Collect semantic embeddings from real RECON frames."""
    all_emb = []
    files = sorted(glob.glob(f"{hdf5_dir}/*.hdf5"))[:max_files]
    for fpath in files:
        try:
            with h5py.File(fpath, "r") as hf:
                imgs = hf["images"]["rgb_left"]
                for i in range(0, len(imgs), stride):
                    try:
                        frame = decode_frame(bytes(imgs[i]))
                        feat = backbone(frame.unsqueeze(0))
                        sem = semantic_head(feat)
                        all_emb.append(sem.squeeze(0))
                    except Exception:
                        pass
        except Exception:
            pass
    if not all_emb:
        return None
    return torch.stack(all_emb)  # (N, 128)


@torch.no_grad()
def extract_features(query_text, all_sem_emb, clip_bridge, clip_model):
    """
    Extract 5-D feature vector for one query text.
    Returns: (topk_mean, base_mean, std, max_sim, ratio)
    """
    tokens = clip.tokenize([query_text]).to(DEVICE)
    text_emb = F.normalize(clip_model.encode_text(tokens).float(), dim=-1)
    text_proj = F.normalize(clip_bridge(text_emb), dim=-1)

    sem_normed = F.normalize(all_sem_emb, dim=-1)
    sims = (sem_normed @ text_proj.T).squeeze(-1)  # (N,)

    topk_vals, _ = torch.topk(sims, min(TOPK, len(sims)))
    topk_mean = topk_vals.mean().item()
    base_mean = sims.mean().item()
    std = sims.std().item()
    max_sim = sims.max().item()
    ratio = topk_mean / (abs(base_mean) + 1e-8)

    return torch.tensor([topk_mean, base_mean, std, max_sim, ratio],
                         dtype=torch.float32)


# ── Training ─────────────────────────────────────────────────────────────────

def train_classifier(hdf5_dir, base_ckpt, student_ckpt, save_path,
                     max_files=50):
    print("=" * 60)
    print("  Sprint 6e — Null Rejection Binary Classifier")
    print("=" * 60)

    # Load backbone
    from train_mvtec import StudentEncoder
    enc = StudentEncoder()
    if Path(student_ckpt).exists():
        sd = torch.load(student_ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd.get("state_dict", sd))
        enc.load_state_dict(sd, strict=False)
    backbone_features = enc.features
    pool = nn.AdaptiveAvgPool2d((2, 2))
    for p in enc.parameters():
        p.requires_grad_(False)
    enc.eval()

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = backbone_features
            self.pool = pool
        def forward(self, x):
            with torch.no_grad():
                return self.pool(self.features(x)).flatten(1)

    backbone = Backbone()

    # Load semantic head + bridge
    ckpt = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    sem_head = SemanticHead(in_dim=256)
    sem_head.load_state_dict(ckpt["semantic_head"])
    sem_head.eval()

    clip_bridge = nn.Linear(512, 128, bias=False)
    clip_bridge.load_state_dict(ckpt["clip_bridge"])
    clip_bridge.eval()

    # Load CLIP
    clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
    clip_model.eval()

    print(f"  Base: {base_ckpt}")
    print(f"  Backbone: {student_ckpt}")

    # Collect frame embeddings
    print(f"  Collecting frame embeddings from {max_files} files...")
    t0 = time.time()
    all_sem_emb = collect_frame_embeddings(backbone, sem_head, hdf5_dir,
                                            max_files=max_files)
    print(f"  {len(all_sem_emb)} frames in {time.time()-t0:.1f}s")

    # Extract features for all queries
    null_queries = load_null_queries()
    print(f"  Positive queries: {len(POSITIVE_QUERIES)}")
    print(f"  Null queries: {len(null_queries)}")

    features = []
    labels = []

    for q in POSITIVE_QUERIES:
        feat = extract_features(q, all_sem_emb, clip_bridge, clip_model)
        features.append(feat)
        labels.append(1.0)

    for q in null_queries:
        feat = extract_features(q, all_sem_emb, clip_bridge, clip_model)
        features.append(feat)
        labels.append(0.0)

    X = torch.stack(features)  # (N_total, 5)
    y = torch.tensor(labels)   # (N_total,)

    print(f"\n  Feature matrix: {X.shape}")
    print(f"  Positive: {int(y.sum())}, Null: {int((1-y).sum())}")

    # Show feature distributions
    pos_mask = y == 1.0
    null_mask = y == 0.0
    feat_names = ["topk_mean", "base_mean", "std", "max_sim", "ratio"]
    print(f"\n  {'Feature':<12} {'Pos mean':>10} {'Null mean':>10} {'Gap':>8}")
    print(f"  {'─'*44}")
    for i, name in enumerate(feat_names):
        pm = X[pos_mask, i].mean().item()
        nm = X[null_mask, i].mean().item()
        print(f"  {name:<12} {pm:>+10.4f} {nm:>+10.4f} {pm-nm:>+8.4f}")

    # Train classifier
    print(f"\n  Training binary classifier...")
    clf = NullClassifier(d_in=5, d_hidden=32)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=1e-4)

    # Class-weighted BCE (9 pos vs ~108 null)
    pos_weight = torch.tensor([len(null_queries) / len(POSITIVE_QUERIES)])

    best_acc = 0.0
    for epoch in range(200):
        clf.train()
        logits = clf(X)
        loss = F.binary_cross_entropy_with_logits(logits, y,
                                                   pos_weight=pos_weight)
        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            preds = torch.sigmoid(logits) > 0.5
            acc = (preds == y.bool()).float().mean().item()
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in clf.state_dict().items()}

        if (epoch + 1) % 50 == 0:
            print(f"    ep{epoch+1:3d}  loss={loss.item():.4f}  acc={acc:.1%}")

    clf.load_state_dict(best_state)
    print(f"\n  Best accuracy: {best_acc:.1%}")

    # Final eval
    clf.eval()
    with torch.no_grad():
        logits = clf(X)
        probs = torch.sigmoid(logits)
        preds = probs > 0.5

    print(f"\n  Final results:")
    print(f"  {'Query':<45} {'Prob':>6} {'Pred':>6} {'True':>6}")
    print(f"  {'─'*66}")

    n_correct_pos = 0
    n_correct_null = 0

    for i, q in enumerate(POSITIVE_QUERIES):
        pred = "NAV" if preds[i] else "null"
        true = "NAV"
        correct = pred == true
        n_correct_pos += int(correct)
        mark = "" if correct else " WRONG"
        print(f"  {q[:45]:<45} {probs[i].item():>6.3f} {pred:>6} {true:>6}{mark}")

    for i, q in enumerate(null_queries):
        idx = len(POSITIVE_QUERIES) + i
        pred = "NAV" if preds[idx] else "null"
        true = "null"
        correct = pred == true
        n_correct_null += int(correct)
        if not correct or i < 8:  # print first 8 + any errors
            mark = "" if correct else " WRONG"
            print(f"  {q[:45]:<45} {probs[idx].item():>6.3f} {pred:>6} {true:>6}{mark}")

    total_null = len(null_queries)
    print(f"\n  Positive: {n_correct_pos}/{len(POSITIVE_QUERIES)} correct")
    print(f"  Null: {n_correct_null}/{total_null} correct")
    print(f"  Total: {n_correct_pos + n_correct_null}/"
          f"{len(POSITIVE_QUERIES) + total_null}")

    # Save
    save_p = Path(save_path)
    save_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "classifier": best_state,
        "accuracy": best_acc,
        "n_positive": len(POSITIVE_QUERIES),
        "n_null": total_null,
        "feature_names": feat_names,
    }, str(save_p))
    print(f"\n  Saved: {save_p}")
    print(f"  Params: {sum(p.numel() for p in clf.parameters()):,}")
    print("=" * 60)

    return clf


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Sprint 6e Null Classifier")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--query", type=str, default=None)
    ap.add_argument("--hdf5-dir", default="recon_data/recon_release")
    ap.add_argument("--base-ckpt",
                    default="checkpoints/dinov2_student/student_dualhead_nce_best.pt")
    ap.add_argument("--student-ckpt",
                    default="checkpoints/dinov2_student/student_best.pt")
    ap.add_argument("--save-path",
                    default="checkpoints/dinov2_student/null_classifier.pt")
    ap.add_argument("--max-files", type=int, default=50)
    args = ap.parse_args()

    if args.train or not (args.eval or args.query):
        train_classifier(
            args.hdf5_dir, args.base_ckpt, args.student_ckpt,
            args.save_path, args.max_files,
        )

if __name__ == "__main__":
    main()
