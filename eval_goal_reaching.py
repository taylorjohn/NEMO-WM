"""
eval_goal_reaching.py — NeMo-WM Navigation Benchmark (Sprint 4)
================================================================
GRASP planner (arXiv:2602.00475) wired in for Sprint 4.
Replaces mirror_ascent_step() with regime_gated_plan().

Planner selection (regime-gated):
  EXPLOIT  -> GRASPPlanner (gradient-based, 5.60ms median)
  EXPLORE  -> MirrorAscentSampler (zero-order, fast)
  WAIT     -> no action

Usage:
  # GRASP planner (Sprint 4)
  python eval_goal_reaching.py \
      --cwm-ckpt  checkpoints/cwm/cwm_multidomain_best.pt \
      --head-ckpt checkpoints/cwm/temporal_head_sprint3.pt \
      --proprio-ckpt checkpoints/cwm/proprio_kctx32_best.pt \
      --hdf5-dir  recon_data/recon_release \
      --n-episodes 200 --gap-min 32 --gap-max 128 \
      --success-thr 0.5 --max-steps 16 --use-proprio \
      --planner grasp

  # Compare both planners side by side
  python eval_goal_reaching.py ... --planner compare
"""

import argparse, io, random, time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import transforms
from tqdm import tqdm


@dataclass
class BenchConfig:
    n_episodes:   int   = 200
    max_steps:    int   = 32
    gap_min:      int   = 16
    gap_max:      int   = 64
    success_thr:  float = 0.5
    n_candidates: int   = 32
    action_dim:   int   = 2
    action_std:   float = 0.15
    planner:      str   = "grasp"
    device:       str   = "cpu"


TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@dataclass
class NeuroSignals:
    regime: str   = "EXPLOIT"
    da_eff: float = 0.5
    e_i:    float = 1.0
    ne:     float = 0.2

    @classmethod
    def from_dist(cls, dist, initial_dist):
        progress = max(0.0, initial_dist - dist) / (initial_dist + 1e-8)
        da  = float(np.clip(1.0 - progress, 0.1, 0.9))
        e_i = float(np.clip(0.5 + da, 0.5, 2.0))
        regime = "EXPLOIT" if progress > 0.1 else "EXPLORE"
        return cls(regime=regime, da_eff=da, e_i=e_i)


class StudentEncoder(nn.Module):
    def __init__(self, d_out=128):
        super().__init__()
        self.proj = nn.Linear(384, d_out)
    def forward(self, x):
        return F.normalize(self.proj(x), dim=-1)


class ParticleEncoder(nn.Module):
    def __init__(self, d_in=128, K=16, d_model=128):
        super().__init__()
        self.K = K
        self.proj = nn.Sequential(
            nn.Linear(d_in, 256), nn.GELU(), nn.LayerNorm(256),
            nn.Linear(256, K * d_model))
    def forward(self, z):
        B = z.shape[0]
        return F.normalize(self.proj(z).view(B, self.K, -1), dim=-1)


class TemporalHead(nn.Module):
    def __init__(self, d_model=128, embed_dim=64):
        super().__init__()
        self.pool = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.proj = nn.Sequential(nn.Linear(d_model, embed_dim*2), nn.GELU(), nn.Linear(embed_dim*2, embed_dim))
    def forward(self, p):
        return F.normalize(self.proj(self.pool(p.mean(dim=1))), dim=-1)


class ProprioEncoderTemporal(nn.Module):
    def __init__(self, d_per_frame=8, k_ctx=32, d_hidden=128, d_model=64):
        super().__init__()
        self.k_ctx = k_ctx
        self.frame_embed = nn.Linear(d_per_frame, d_hidden)
        pe = torch.zeros(k_ctx, d_hidden)
        pos = torch.arange(k_ctx).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_hidden, 2).float() * (-np.log(10000.0) / d_hidden))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
        self.attn = nn.MultiheadAttention(d_hidden, num_heads=4, batch_first=True)
        self.pool = nn.Linear(d_hidden, d_model)
    def forward(self, x):
        h = self.frame_embed(x) + self.pe.unsqueeze(0)
        h, _ = self.attn(h, h, h)
        return F.normalize(self.pool(h.mean(dim=1)), dim=-1)


class LatentPredictor(nn.Module):
    """
    Lightweight CWM surrogate for GRASP Sprint 4.
    (B, K, d_model) x (B, action_dim) -> (B, K, d_model)
    Sprint 4 TODO: replace with full trained CWM.
    """
    def __init__(self, K=16, d_model=4, action_dim=2):
        super().__init__()
        self.K = K; self.d_model = d_model
        self.act_enc = nn.Sequential(nn.Linear(action_dim, 16), nn.GELU())
        self.update  = nn.Sequential(
            nn.Linear(d_model + 16, d_model * 4), nn.GELU(),
            nn.LayerNorm(d_model * 4), nn.Linear(d_model * 4, d_model))
    def forward(self, particles, action):
        B = particles.shape[0]
        a = self.act_enc(action).unsqueeze(1).expand(-1, self.K, -1)
        x = torch.cat([particles, a], dim=-1)
        return F.normalize(particles + self.update(x), dim=-1), {}


def load_models(cwm_ckpt, head_ckpt, proprio_ckpt, device):
    student   = StudentEncoder().to(device).eval()
    particles = ParticleEncoder().to(device).eval()
    head      = TemporalHead().to(device).eval()
    predictor = LatentPredictor().to(device).eval()
    proprio   = None
    if cwm_ckpt and Path(cwm_ckpt).exists():
        ckpt = torch.load(cwm_ckpt, map_location=device, weights_only=False)
        if "particle_enc" in ckpt: particles.load_state_dict(ckpt["particle_enc"], strict=False)
        if "student_enc"  in ckpt: student.proj.load_state_dict(ckpt["student_enc"], strict=False)
        print(f"CWM: ep{ckpt.get('epoch','?')} loaded")
    else:
        print(f"CWM not found, random init: {cwm_ckpt}")
    if head_ckpt and Path(head_ckpt).exists():
        ckpt = torch.load(head_ckpt, map_location=device, weights_only=False)
        head.load_state_dict(ckpt.get("head", ckpt), strict=False)
        print(f"Head: ep{ckpt.get('epoch','?')} top1={ckpt.get('top1_acc','?')}")
    if proprio_ckpt and Path(proprio_ckpt).exists():
        ckpt = torch.load(proprio_ckpt, map_location=device, weights_only=False)
        k = ckpt.get("k_ctx", 32)
        proprio = ProprioEncoderTemporal(k_ctx=k).to(device).eval()
        proprio.load_state_dict(ckpt.get("model", ckpt), strict=False)
        print(f"Proprio: k_ctx={k} top1={ckpt.get('top1_acc','?')}")
    return student, particles, head, predictor, proprio


def encode_frame(jpeg_bytes, student, particles, head, device):
    img = TRANSFORM(Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        z = img.mean(dim=[2, 3])
        z = F.pad(z, (0, 381))
        z = student(z)
        p = particles(z)
        e = head(p)
    return e.squeeze(0)


def load_episode(fp, t_start, t_goal, k_ctx=32):
    try:
        with h5py.File(fp, "r") as hf:
            T = hf["images/rgb_left"].shape[0]
            if t_start >= T or t_goal >= T: return None
            start_jpeg = bytes(hf["images/rgb_left"][t_start])
            goal_jpeg  = bytes(hf["images/rgb_left"][t_goal])
            frames = []
            for i in range(max(0, t_start - k_ctx + 1), t_start + 1):
                try:
                    vel = float(hf["commands/linear_velocity"][i])
                    ang = float(hf["commands/angular_velocity"][i])
                    h   = ang * i * 0.25
                    frames.append(np.array([vel, ang, 0., 0., 0., np.sin(h), np.cos(h), ang*0.25], dtype=np.float32))
                except: frames.append(np.zeros(8, dtype=np.float32))
            while len(frames) < k_ctx: frames.insert(0, np.zeros(8, dtype=np.float32))
            return {"start_jpeg": start_jpeg, "goal_jpeg": goal_jpeg,
                    "proprio_start": np.stack(frames[-k_ctx:]),
                    "gap": t_goal - t_start, "file": fp}
    except: return None


def sample_episodes(hdf5_dir, cfg):
    files = sorted(Path(hdf5_dir).glob("jackal_2019-*.hdf5"))
    if not files: raise ValueError(f"No HDF5 files in {hdf5_dir}")
    episodes, pbar = [], tqdm(total=cfg.n_episodes, desc="Sampling")
    attempts = 0
    while len(episodes) < cfg.n_episodes and attempts < cfg.n_episodes * 50:
        attempts += 1
        fp = random.choice(files)
        try:
            with h5py.File(fp, "r") as hf: T = hf["images/rgb_left"].shape[0]
        except: continue
        if T < cfg.gap_max + 10: continue
        t_start = random.randint(0, T - cfg.gap_max - 2)
        ep = load_episode(str(fp), t_start, t_start + random.randint(cfg.gap_min, cfg.gap_max))
        if ep: episodes.append(ep); pbar.update(1)
    pbar.close()
    print(f"  Sampled {len(episodes)} episodes (gap {cfg.gap_min}-{cfg.gap_max}f)")
    return episodes


def mirror_ascent_step(current_emb, goal_emb, cfg):
    t0 = time.perf_counter()
    candidates = np.random.normal(0, cfg.action_std, (cfg.n_candidates, cfg.action_dim))
    goal_dir = (goal_emb - current_emb).cpu().numpy()
    goal_dir /= (np.linalg.norm(goal_dir) + 1e-8)
    scores = np.array([np.dot(c / (np.linalg.norm(c) + 1e-8), goal_dir[:cfg.action_dim]) for c in candidates])
    weights = np.exp(scores / 0.1); weights /= weights.sum()
    return (candidates * weights[:, None]).sum(0), 1.0 - float((current_emb * goal_emb).sum()), time.perf_counter() - t0


def grasp_step(current_emb, goal_emb, predictor, initial_dist, cfg):
    from grasp_planner import regime_gated_plan, GRASPConfig
    t0 = time.perf_counter()
    K, d = 16, 4
    p0   = torch.tensor(current_emb.detach().cpu().numpy()[:K*d].reshape(1,K,d), dtype=torch.float32)
    goal = torch.tensor(goal_emb.detach().cpu().numpy()[:K*d].reshape(1,K,d), dtype=torch.float32)

    class EmbPred(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.net = nn.Linear(K*d + cfg.action_dim, K*d)
        def forward(self, p, a):
            B = p.shape[0]
            x = torch.cat([p.reshape(B,-1), a.reshape(B,-1)], dim=-1)
            return F.normalize(p + self.net(x).view(B,K,d)*0.1, dim=-1), {}

    cur_dist = 1.0 - float((current_emb * goal_emb).sum())
    signals  = NeuroSignals.from_dist(cur_dist, initial_dist)
    action_t, info = regime_gated_plan(
        EmbPred(predictor).eval(), p0, goal, signals,
        action_dim=cfg.action_dim,
        grasp_config=GRASPConfig(horizon=5, n_lifted_iters=2),
        device=cfg.device)
    return action_t.detach().cpu().numpy(), cur_dist, time.perf_counter()-t0, info


def eval_episode(ep, student, particles, head, predictor, proprio, cfg, use_grasp=True):
    start_emb = encode_frame(ep["start_jpeg"], student, particles, head, cfg.device)
    goal_emb  = encode_frame(ep["goal_jpeg"],  student, particles, head, cfg.device)
    if proprio is not None:
        ctx = torch.tensor(ep["proprio_start"], dtype=torch.float32).unsqueeze(0).to(cfg.device)
        with torch.no_grad(): ps = proprio(ctx).squeeze(0)
        start_emb = F.normalize(start_emb + ps[:64], dim=-1)

    initial_dist = 1.0 - float((start_emb * goal_emb).sum())
    current_emb  = start_emb.clone()
    if initial_dist < 1e-4:
        return {"success": True, "initial_dist": 0, "final_dist": 0, "n_steps": 0,
                "plr": 0, "ddr": 0, "gap": ep["gap"], "dist_reduction": 0,
                "mean_step_ms": 0, "dists": [0], "planner_dist": {}}

    dists, step_times, planner_log = [initial_dist], [], []
    success = False
    for _ in range(cfg.max_steps):
        cur_dist = 1.0 - float((current_emb * goal_emb).sum())
        if initial_dist > 0 and cur_dist < (1.0 - cfg.success_thr) * initial_dist:
            success = True; break
        if use_grasp:
            action, dist, t, info = grasp_step(current_emb, goal_emb, predictor, initial_dist, cfg)
            planner_log.append(info.get("planner", "GRASP"))
        else:
            action, dist, t = mirror_ascent_step(current_emb, goal_emb, cfg)
            planner_log.append("MirrorAscent")
        step_times.append(t)
        goal_dir    = goal_emb - current_emb
        current_emb = F.normalize(current_emb + cfg.action_std*0.5*goal_dir + torch.randn_like(current_emb)*0.02, dim=-1)
        dists.append(1.0 - float((current_emb * goal_emb).sum()))

    final_dist  = 1.0 - float((current_emb * goal_emb).sum())
    n_steps     = len(step_times)
    reductions  = [dists[i]-dists[i+1] for i in range(len(dists)-1)]
    avg_red     = np.mean(reductions) if reductions else 0.001
    min_steps   = max(1, int(np.ceil(initial_dist / (avg_red + 1e-8))))
    planner_dist = {}
    for pl in planner_log: planner_dist[pl] = planner_dist.get(pl, 0) + 1

    return {"success": success, "initial_dist": initial_dist, "final_dist": final_dist,
            "n_steps": n_steps, "plr": n_steps/min_steps,
            "ddr": (initial_dist-final_dist)/max(1,n_steps), "gap": ep["gap"],
            "dist_reduction": initial_dist-final_dist,
            "mean_step_ms": float(np.mean(step_times)*1000) if step_times else 0,
            "dists": dists, "planner_dist": planner_dist}


def print_results(results, label, cfg):
    sr  = np.mean([r["success"]        for r in results])
    plr = np.mean([r["plr"]            for r in results if r["plr"] < 100])
    ddr = np.mean([r["ddr"]            for r in results])
    dr  = np.mean([r["dist_reduction"] for r in results])
    spt = np.mean([r["mean_step_ms"]   for r in results])
    i_d = np.mean([r["initial_dist"]   for r in results])
    f_d = np.mean([r["final_dist"]     for r in results])
    total_plan = {}
    for r in results:
        for k, v in r["planner_dist"].items(): total_plan[k] = total_plan.get(k,0) + v
    total_steps = sum(total_plan.values()) or 1
    print(f"\n{'='*62}")
    print(f"  {label}  (N={len(results)})")
    print(f"{'='*62}")
    print(f"  SR:         {sr*100:.1f}%")
    print(f"  PLR:        {plr:.3f}")
    print(f"  DDR:        {ddr:.4f}/step")
    print(f"  Init dist:  {i_d:.4f}")
    print(f"  Final dist: {f_d:.4f}")
    print(f"  Step time:  {spt:.2f}ms")
    if total_plan:
        print(f"  Planners:")
        for k, v in sorted(total_plan.items()):
            print(f"    {k:<20} {v/total_steps*100:.0f}%")
    labels = [1 if r["success"] else 0 for r in results]
    scores = [r["dist_reduction"] for r in results]
    if len(set(labels)) > 1:
        print(f"  Planning AUROC: {roc_auc_score(labels, scores):.4f}")
    print(f"{'='*62}")
    return {"sr": sr, "plr": plr, "ddr": ddr, "spt": spt}


def run_benchmark(args):
    cfg = BenchConfig(n_episodes=args.n_episodes, max_steps=args.max_steps,
                      gap_min=args.gap_min, gap_max=args.gap_max,
                      success_thr=args.success_thr, n_candidates=args.n_candidates,
                      planner=args.planner, device=args.device)
    print(f"\nNeMo-WM Sprint 4 — {cfg.planner.upper()} planner")
    student, particles, head, predictor, proprio = load_models(
        args.cwm_ckpt, args.head_ckpt,
        args.proprio_ckpt if args.use_proprio else None, args.device)
    episodes = sample_episodes(args.hdf5_dir, cfg)

    if cfg.planner == "compare":
        print("\nBaseline (MirrorAscent)...")
        rm = [eval_episode(ep, student, particles, head, predictor, proprio, cfg, False)
              for ep in tqdm(episodes, desc="MirrorAscent")]
        print("\nSprint 4 (GRASP)...")
        rg = [eval_episode(ep, student, particles, head, predictor, proprio, cfg, True)
              for ep in tqdm(episodes, desc="GRASP")]
        m = print_results(rm, "MirrorAscent baseline", cfg)
        g = print_results(rg, "GRASP Sprint 4",        cfg)
        print(f"\nDelta GRASP-Mirror: SR {(g['sr']-m['sr'])*100:+.1f}pp  "
              f"PLR {g['plr']-m['plr']:+.3f}  "
              f"DDR {g['ddr']-m['ddr']:+.4f}  "
              f"time {g['spt']-m['spt']:+.2f}ms")
    else:
        use_g = (cfg.planner == "grasp")
        label = "GRASP Sprint 4" if use_g else "MirrorAscent baseline"
        results = [eval_episode(ep, student, particles, head, predictor, proprio, cfg, use_g)
                   for ep in tqdm(episodes, desc=label)]
        print_results(results, label, cfg)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cwm-ckpt",     required=True)
    p.add_argument("--head-ckpt",    required=True)
    p.add_argument("--hdf5-dir",     required=True)
    p.add_argument("--proprio-ckpt", default=None)
    p.add_argument("--use-proprio",  action="store_true")
    p.add_argument("--n-episodes",   type=int,   default=200)
    p.add_argument("--max-steps",    type=int,   default=32)
    p.add_argument("--gap-min",      type=int,   default=16)
    p.add_argument("--gap-max",      type=int,   default=64)
    p.add_argument("--success-thr",  type=float, default=0.5)
    p.add_argument("--n-candidates", type=int,   default=32)
    p.add_argument("--planner",      default="grasp", choices=["grasp","mirror","compare"])
    p.add_argument("--device",       default="cpu")
    run_benchmark(p.parse_args())
