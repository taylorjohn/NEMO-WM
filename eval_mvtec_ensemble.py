from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from PIL import Image
from train_mvtec import (
    StudentEncoder, ALL_CATEGORIES, TEXTURES, OBJECTS,
    load_category, _TRANSFORM_256,
)

@torch.no_grad()
def encode(model, images, transform, batch=32):
    feats = []
    for i in range(0, len(images), batch):
        tensors = torch.stack([transform(img) for img in images[i:i+batch]])
        out = model(tensors)
        if isinstance(out, torch.Tensor):
            feats.append(F.normalize(out.flatten(1), dim=-1).cpu().numpy())
    return np.concatenate(feats, axis=0) if feats else np.empty((0, 1))

def load_imgs(paths):
    imgs = []
    for p in paths:
        try: imgs.append(Image.open(p).convert("RGB"))
        except: pass
    return imgs

def eval_category(cat, data_root, student, dinov2, k=32):
    cat_dir = data_root / cat
    if not cat_dir.exists(): return None
    d = load_category(cat_dir)
    train_imgs  = load_imgs(d["train_good"])
    test_imgs   = load_imgs(d["test_good"] + d["test_defect"])
    test_labels = [0]*len(d["test_good"]) + [1]*len(d["test_defect"])
    if not train_imgs or not test_imgs: return None
    labels = np.array(test_labels)
    if labels.sum() == 0 or labels.sum() == len(labels): return None

    train_s = encode(student, train_imgs, _TRANSFORM_256)
    test_s  = encode(student, test_imgs,  _TRANSFORM_256)
    train_d = encode(dinov2,  train_imgs, _TRANSFORM_256)
    test_d  = encode(dinov2,  test_imgs,  _TRANSFORM_256)

    train_e = np.concatenate([train_s, train_d], axis=1)
    test_e  = np.concatenate([test_s,  test_d],  axis=1)
    train_e /= np.linalg.norm(train_e, axis=1, keepdims=True) + 1e-8
    test_e  /= np.linalg.norm(test_e,  axis=1, keepdims=True) + 1e-8

    knn = NearestNeighbors(n_neighbors=min(k, len(train_e)))
    knn.fit(train_e)
    dists, _ = knn.kneighbors(test_e)
    return float(roc_auc_score(labels, dists.mean(axis=1)))

def run(args):
    print("=" * 60)
    print("  MVTec Ensemble  (student 128-D + DINOv2 384-D = 512-D)")
    print("=" * 60)
    data_root = Path(args.data)
    cats = args.categories or ALL_CATEGORIES

    student = StudentEncoder()
    ckpt_path = Path(args.encoder_ckpt)
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), weights_only=False, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt.get("model", ckpt))
        student.load_state_dict(sd, strict=False)
        print("  Student : " + str(ckpt_path))
    student.eval()

    print("  Loading DINOv2-small (cached)...")
    dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                              pretrained=True, verbose=False)
    dinov2.eval()
    print("  DINOv2  : OK")

    t0, results = time.perf_counter(), {}
    for cat in cats:
        auroc = eval_category(cat, data_root, student, dinov2, k=args.k)
        if auroc is None:
            print("  " + cat.ljust(14) + "  skipped")
            continue
        flag = "OK" if auroc >= 0.80 else "--"
        print("  " + cat.ljust(14) + "  " + str(round(auroc, 4)) + "  " + flag)
        results[cat] = auroc

    if not results:
        print("No results -- check --data path")
        return

    mean    = float(np.mean(list(results.values())))
    n_pass  = sum(1 for v in results.values() if v >= 0.80)
    elapsed = time.perf_counter() - t0
    tex = float(np.mean([results[c] for c in TEXTURES if c in results])) if any(c in results for c in TEXTURES) else 0.0
    obj = float(np.mean([results[c] for c in OBJECTS  if c in results])) if any(c in results for c in OBJECTS)  else 0.0

    print("=" * 60)
    print("  Categories    : " + str(len(results)))
    print("  Pass (>=0.80) : " + str(n_pass) + "/" + str(len(results)))
    print("  Textures mean : " + str(round(tex, 4)))
    print("  Objects mean  : " + str(round(obj, 4)))
    met = "MET" if mean >= 0.8152 else "below 0.8152"
    print("  ENSEMBLE AUROC: " + str(round(mean, 4)) + "  [" + met + "]")
    print("  Student-only  : 0.7393")
    print("  Delta         : " + str(round(mean - 0.7393, 4)))
    print("  Elapsed       : " + str(round(elapsed, 1)) + "s")
    print("=" * 60)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data",         required=True)
    p.add_argument("--encoder-ckpt", default="checkpoints/dinov2_student/student_best.pt")
    p.add_argument("--k",            type=int, default=32)
    p.add_argument("--categories",   nargs="+", default=None)
    args = p.parse_args()
    run(args)
