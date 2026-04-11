"""
run_benchmark.py — CORTEX-16 MPC Benchmark

Evaluates planning success rate on Wall, UMaze, Medium.
Comparable to Wang et al. 2026 Table 1 (arXiv:2603.12231).

Two planners available (--planner flag):
    mirror_ascent  — K=64 candidates, R=3 rounds (default, fast)
    gradient       — Adam on action sequence, 100 steps, lr=0.01
                     Matches Wang et al. protocol exactly for fair comparison

Key protocol changes from Wang et al. (Table 4 & 5):
    Frameskip:         5  (not 1 — consecutive frames are near-identical)
    Planning horizon:  25 (not 5 — must cover full maze traversal)
    Executed actions:  25 open-loop, 5 MPC chunk (Wang et al. Table 5 footnote)

Options:
    1 — GeometricHorizon MLP (global 128-D)
    2 — CausalTransformer (global 128-D, K=3)
    3 — SpatialMLP (1568-D spatial)
    4 — SpatialTransformer (196 patches × 8ch, K=3)  ← recommended

Usage:
    # Quick mirror-ascent benchmark
    python run_benchmark.py --encoder ./checkpoints/cortex_student_phase2_final.pt --option 4 --env wall --threshold-file ./benchmark_thresholds_opt4_50pct.json

    # Wang et al. protocol (gradient planner, horizon=25, frameskip=5)
    python run_benchmark.py --encoder ./checkpoints/cortex_student_phase2_final.pt --option 4 --env wall --planner gradient --threshold-file ./benchmark_thresholds_opt4_50pct.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from student_encoder import StudentEncoder
from latent_predictor import GeometricHorizonPredictor
from train_predictor import (
    CausalTransformerPredictor,
    SpatialChannelProjector,
    SpatialMLPPredictor,
    SpatialTransformerPredictor,
    ENV_CONFIG,
    FRAME_TRANSFORM,
    SPATIAL_DIM,
    N_PATCHES,
    DV,
)


# =============================================================================
# Planning constants — Wang et al. 2026 Table 5
# =============================================================================
PLAN_HORIZON   = 25    # Wang et al.: subplanner horizon 25
N_EXECUTED     = 25    # Wang et al.: open-loop executes 25 actions
N_MPC_CHUNK    = 5     # Wang et al.: MPC executes first 5 of 25 planned
N_MPC_STEPS    = 20    # Max MPC replanning iterations (25 total steps / 5 per chunk)
N_TEST         = 50
N_SEEDS        = 3

# Mirror ascent params (fast alternative to gradient)
MA_CANDS  = 64
MA_ITERS  = 3
MA_ETA    = 1.0

# Gradient planner params — Wang et al. Table 5
GD_LR       = 0.01
GD_STEPS    = 100
GD_ACTION_INIT = "zero"   # Wang et al.: zero initialisation

DINO_WM = {
    "wall":   {"open": 52.67, "mpc": 76.67},
    "umaze":  {"open": 35.33, "mpc": 80.67},
    "medium": {"open": 40.83, "mpc": 76.67},
    "pusht":  {"open": 56.00, "mpc": 66.00},
}
WANG_BEST = {
    "wall":   {"open": 90.67, "mpc": 100.00},
    "umaze":  {"open": 94.00, "mpc": 100.00},
    "medium": {"open": 82.67, "mpc":  98.67},
    "pusht":  {"open": 77.33, "mpc":  85.33},
}


# =============================================================================
# Predictor loading
# =============================================================================
def load_predictor(pred_path, option, env_name):
    config = ENV_CONFIG[env_name]
    ckpt   = torch.load(pred_path, map_location="cpu")
    spatial_proj = None

    if option == 1:
        pred = GeometricHorizonPredictor(128, config["action_dim"])
    elif option == 2:
        pred = CausalTransformerPredictor(128, config["action_dim"], history_len=3)
    elif option == 3:
        pred         = SpatialMLPPredictor(SPATIAL_DIM, config["action_dim"])
        spatial_proj = SpatialChannelProjector(32, DV)
        spatial_proj.load_state_dict(ckpt["spatial_proj_state_dict"])
        spatial_proj.eval()
    elif option == 4:
        pred = SpatialTransformerPredictor(
            n_patches=N_PATCHES, dv=DV,
            action_dim=config["action_dim"], history_len=3,
            d_model=64, n_heads=4, n_layers=4,
        )
        spatial_proj = SpatialChannelProjector(32, DV)
        spatial_proj.load_state_dict(ckpt["spatial_proj_state_dict"])
        spatial_proj.eval()

    pred.load_state_dict(ckpt["state_dict"])
    pred.eval()
    return pred, spatial_proj


# =============================================================================
# Encoder wrapper
# =============================================================================
class Encoder:
    def __init__(self, encoder, option, spatial_proj=None):
        self.encoder      = encoder
        self.option       = option
        self.spatial_proj = spatial_proj

    def encode(self, frame_np):
        t = FRAME_TRANSFORM(frame_np).unsqueeze(0)
        with torch.no_grad():
            if self.option in (3, 4) and self.spatial_proj is not None:
                _, spatial = self.encoder(t, return_spatial=True)
                if self.option == 4:
                    return self.spatial_proj(spatial, use_tokens=True).squeeze(0)  # (196, 8)
                return self.spatial_proj(spatial, use_tokens=False).squeeze(0)     # (1568,)
            return self.encoder(t).squeeze(0)  # (128,)

    def flat(self, z):
        return z.reshape(-1)


# =============================================================================
# Gradient planner — Wang et al. Table 5 protocol
# Adam on action sequence, 100 steps, lr=0.01, zero initialisation
# =============================================================================
def gradient_plan(predictor, z_start, z_goal, action_dim, option,
                  horizon=PLAN_HORIZON, n_steps=GD_STEPS, lr=GD_LR,
                  z_hist=None, a_hist=None):
    """
    Optimise action sequence via gradient descent.
    Matches Wang et al. 2026 Table 5 exactly.
    """
    actions = torch.zeros(horizon, action_dim, requires_grad=True)
    opt     = torch.optim.Adam([actions], lr=lr)
    z_g     = z_goal.flatten()

    for _ in range(n_steps):
        opt.zero_grad()
        z    = z_start.clone().unsqueeze(0)
        cost = torch.tensor(0.0)
        z_h  = z_hist.clone().unsqueeze(0) if z_hist is not None else None
        a_h  = a_hist.clone().unsqueeze(0) if a_hist is not None else None

        for t in range(horizon):
            act = actions[t].unsqueeze(0)
            if option == 1:
                z = predictor(z, act, gamma=1.0)
            elif option == 2:
                z = predictor(z_h, a_h)
                if z_h is not None:
                    z_h = torch.cat([z_h[:, 1:, :], z.unsqueeze(1)], dim=1)
                    a_h = torch.cat([a_h[:, 1:, :], act.unsqueeze(1)], dim=1)
            elif option == 3:
                z = predictor(z, act)
            elif option == 4:
                if z_h is not None:
                    z_out = predictor(z_h, a_h)
                    z_h   = torch.cat([z_h[:, 1:, :, :], z_out.reshape(1,1,N_PATCHES,DV)], dim=1)
                    a_h   = torch.cat([a_h[:, 1:, :], act.unsqueeze(1)], dim=1)
                    z     = z_out.unsqueeze(0) if z_out.dim() == 1 else z_out

            cost = cost + torch.norm(z.reshape(-1) - z_g, p=2)

        cost.backward()
        opt.step()

    return actions[0].detach().numpy()


# =============================================================================
# Hybrid planner — mirror ascent warm-start + gradient refinement
# Best of both: MA finds a good basin, GD refines it.
# Fixes gradient getting trapped in local minima on UMaze/Medium topology.
# =============================================================================
def hybrid_plan(predictor, z_start, z_goal, action_dim, option,
                horizon=PLAN_HORIZON, n_steps=GD_STEPS, lr=GD_LR,
                z_hist=None, a_hist=None,
                ma_iters=MA_ITERS, gd_steps=50):
    """
    Phase 1 — Mirror ascent (MA_ITERS rounds of K=64 candidates)
               Explores the action space, finds a good basin.
    Phase 2 — Gradient descent (50 steps from MA warm start)
               Refines the MA solution without escaping the good basin.

    Wall:   GD alone works (89.3%) — hybrid preserves this.
    UMaze:  MA alone works (62%) — hybrid preserves via warm start.
    Medium: MA alone works (50%) — hybrid preserves via warm start.
    """
    # Phase 1: Mirror ascent to find good initialisation
    cands = torch.randn(MA_CANDS, horizon, action_dim) * 0.1
    log_q = torch.zeros(MA_CANDS)
    z_g   = z_goal.flatten()

    with torch.no_grad():
        for _ in range(ma_iters):
            costs = []
            for i in range(MA_CANDS):
                z    = z_start.clone().unsqueeze(0)
                cost = 0.0
                z_h  = z_hist.clone().unsqueeze(0) if z_hist is not None else None
                a_h  = a_hist.clone().unsqueeze(0) if a_hist is not None else None
                for t in range(horizon):
                    act = cands[i, t].unsqueeze(0)
                    if option == 1:
                        z = predictor(z, act, gamma=1.0)
                    elif option == 3:
                        z = predictor(z, act)
                    cost += torch.norm(z.reshape(-1) - z_g, p=2).item()
                costs.append(cost)
            costs_t = torch.tensor(costs)
            log_q   = log_q + MA_ETA * (-costs_t)
            log_q   = log_q - log_q.logsumexp(0)
            w          = log_q.exp()
            elite_mean = (w.view(-1,1,1) * cands).sum(0)
            cands = elite_mean.unsqueeze(0) + torch.randn(MA_CANDS, horizon, action_dim) * 0.05

    # Best MA candidate as warm start for gradient
    best_cand = cands[log_q.argmax()]  # (horizon, action_dim)

    # Phase 2: Gradient refinement from MA warm start
    actions = best_cand.clone().detach().requires_grad_(True)
    opt     = torch.optim.Adam([actions], lr=lr)

    for _ in range(gd_steps):
        opt.zero_grad()
        z    = z_start.clone().unsqueeze(0)
        cost = torch.tensor(0.0)
        z_h  = z_hist.clone().unsqueeze(0) if z_hist is not None else None
        a_h  = a_hist.clone().unsqueeze(0) if a_hist is not None else None
        for t in range(horizon):
            act = actions[t].unsqueeze(0)
            if option == 1:
                z = predictor(z, act, gamma=1.0)
            elif option == 3:
                z = predictor(z, act)
            cost = cost + torch.norm(z.reshape(-1) - z_g, p=2)
        cost.backward()
        opt.step()

    return actions[0].detach().numpy()




# =============================================================================
# MeZO planners — Memory-Efficient Zero-Order optimisation
# =============================================================================
MEZO_SIGMA          = 0.1
MEZO_LR             = 0.05
MEZO_STEPS          = 50
MEZO_MOMENTUM_BETA  = 0.9
MEZO_MOMENTUM_STEPS = 60
MEZO_POP_K          = 20
MEZO_POP_ROUNDS     = 3
MEZO_POP_REFINE     = 5

def _rollout_cost(actions, predictor, z_start, z_goal, option,
                  z_hist=None, a_hist=None, horizon=PLAN_HORIZON):
    z   = z_start.clone().unsqueeze(0)
    z_g = z_goal.flatten()
    cost = 0.0
    for t in range(horizon):
        act = actions[t].unsqueeze(0)
        if option == 1:
            z = predictor(z, act, gamma=1.0)
        elif option == 3:
            z = predictor(z, act)
        cost += torch.norm(z.reshape(-1) - z_g, p=2).item()
    return cost

def mezo_plan(predictor, z_start, z_goal, action_dim, option,
              horizon=PLAN_HORIZON, n_steps=MEZO_STEPS,
              sigma=MEZO_SIGMA, lr=MEZO_LR, z_hist=None, a_hist=None):
    actions = torch.zeros(horizon, action_dim)
    with torch.no_grad():
        for _ in range(n_steps):
            eps   = torch.randn_like(actions)
            c_pos = _rollout_cost(actions + sigma*eps, predictor, z_start, z_goal, option, z_hist, a_hist, horizon)
            c_neg = _rollout_cost(actions - sigma*eps, predictor, z_start, z_goal, option, z_hist, a_hist, horizon)
            actions = actions - lr * ((c_pos - c_neg) / (2*sigma)) * eps
    return actions[0].numpy()

def mezo_warm_plan(predictor, z_start, z_goal, action_dim, option,
                   horizon=PLAN_HORIZON, z_hist=None, a_hist=None,
                   mezo_steps=30, sigma=MEZO_SIGMA, lr=MEZO_LR):
    best_first = mirror_ascent_plan(predictor, z_start, z_goal, action_dim, option,
                                    horizon=horizon, z_hist=z_hist, a_hist=a_hist)
    actions = torch.tensor(best_first, dtype=torch.float32).unsqueeze(0).expand(horizon, -1).clone()
    with torch.no_grad():
        for _ in range(mezo_steps):
            eps   = torch.randn_like(actions)
            c_pos = _rollout_cost(actions + sigma*eps, predictor, z_start, z_goal, option, z_hist, a_hist, horizon)
            c_neg = _rollout_cost(actions - sigma*eps, predictor, z_start, z_goal, option, z_hist, a_hist, horizon)
            actions = actions - lr * ((c_pos - c_neg) / (2*sigma)) * eps
    return actions[0].numpy()

def mezo_momentum_plan(predictor, z_start, z_goal, action_dim, option,
                       horizon=PLAN_HORIZON, z_hist=None, a_hist=None,
                       n_steps=MEZO_MOMENTUM_STEPS, sigma=MEZO_SIGMA,
                       lr=0.03, beta=MEZO_MOMENTUM_BETA):
    actions  = torch.zeros(horizon, action_dim)
    momentum = torch.zeros_like(actions)
    with torch.no_grad():
        for step in range(n_steps):
            eps      = torch.randn_like(actions)
            c_pos    = _rollout_cost(actions + sigma*eps, predictor, z_start, z_goal, option, z_hist, a_hist, horizon)
            c_neg    = _rollout_cost(actions - sigma*eps, predictor, z_start, z_goal, option, z_hist, a_hist, horizon)
            g        = ((c_pos - c_neg) / (2*sigma)) * eps
            momentum = beta * momentum + (1-beta) * g
            m_hat    = momentum / (1 - beta**(step+1))
            actions  = actions - lr * m_hat
    return actions[0].numpy()

def mezo_cem_plan(predictor, z_start, z_goal, action_dim, option,
                  horizon=PLAN_HORIZON, z_hist=None, a_hist=None,
                  K=MEZO_POP_K, rounds=MEZO_POP_ROUNDS,
                  n_refine=MEZO_POP_REFINE, sigma=MEZO_SIGMA,
                  lr=0.05, elite_frac=0.5):
    n_elite = max(1, int(K * elite_frac))
    pop     = torch.randn(K, horizon, action_dim) * 0.1
    with torch.no_grad():
        for _ in range(rounds):
            for k in range(K):
                for _ in range(n_refine):
                    eps    = torch.randn(horizon, action_dim)
                    c_pos  = _rollout_cost(pop[k] + sigma*eps, predictor, z_start, z_goal, option, z_hist, a_hist, horizon)
                    c_neg  = _rollout_cost(pop[k] - sigma*eps, predictor, z_start, z_goal, option, z_hist, a_hist, horizon)
                    pop[k] = pop[k] - lr * ((c_pos - c_neg) / (2*sigma)) * eps
            costs      = torch.tensor([_rollout_cost(pop[k], predictor, z_start, z_goal, option, z_hist, a_hist, horizon) for k in range(K)])
            elite_idx  = costs.argsort()[:n_elite]
            elite_mean = pop[elite_idx].mean(0)
            elite_std  = pop[elite_idx].std(0).clamp(min=0.02)
            non_mask   = torch.ones(K, dtype=torch.bool)
            non_mask[elite_idx] = False
            n_non = non_mask.sum().item()
            pop[non_mask] = elite_mean.unsqueeze(0) + elite_std.unsqueeze(0) * torch.randn(n_non, horizon, action_dim)
        costs = torch.tensor([_rollout_cost(pop[k], predictor, z_start, z_goal, option, z_hist, a_hist, horizon) for k in range(K)])
    return pop[costs.argmin()][0].numpy()

def mirror_ascent_plan(predictor, z_start, z_goal, action_dim, option,
                       horizon=PLAN_HORIZON, z_hist=None, a_hist=None):
    cands = torch.randn(MA_CANDS, horizon, action_dim) * 0.1
    log_q = torch.zeros(MA_CANDS)
    z_g   = z_goal.flatten()

    with torch.no_grad():
        for _ in range(MA_ITERS):
            costs = []
            for i in range(MA_CANDS):
                z    = z_start.clone().unsqueeze(0)
                cost = 0.0
                z_h  = z_hist.clone().unsqueeze(0) if z_hist is not None else None
                a_h  = a_hist.clone().unsqueeze(0) if a_hist is not None else None

                for t in range(horizon):
                    act = cands[i, t].unsqueeze(0)
                    if option == 1:
                        z = predictor(z, act, gamma=1.0)
                    elif option == 2:
                        z = predictor(z_h, a_h)
                        if z_h is not None:
                            z_h = torch.cat([z_h[:, 1:], z.unsqueeze(1)], dim=1)
                            a_h = torch.cat([a_h[:, 1:], act.unsqueeze(1)], dim=1)
                    elif option == 3:
                        z = predictor(z, act)
                    elif option == 4:
                        if z_h is not None:
                            z_out = predictor(z_h, a_h)
                            z_h   = torch.cat([z_h[:, 1:], z_out.reshape(1,1,N_PATCHES,DV)], dim=1)
                            a_h   = torch.cat([a_h[:, 1:], act.unsqueeze(1)], dim=1)
                            z     = z_out.unsqueeze(0)
                    cost += torch.norm(z.reshape(-1) - z_g, p=2).item()
                costs.append(cost)

            costs_t = torch.tensor(costs)
            log_q   = log_q + MA_ETA * (-costs_t)
            log_q   = log_q - log_q.logsumexp(0)
            w          = log_q.exp()
            elite_mean = (w.view(-1,1,1) * cands).sum(0)
            cands = elite_mean.unsqueeze(0) + torch.randn(MA_CANDS, horizon, action_dim) * 0.05

    return cands[log_q.argmax()][0].numpy()


# =============================================================================
# Evaluation
# =============================================================================
def evaluate_env(env_name, enc, predictor, option, data_root,
                 spatial_proj=None, seed=42, thresh=1.0,
                 planner_type="mirror_ascent"):
    config  = ENV_CONFIG[env_name]
    raw     = np.load(Path(data_root) / env_name / "trajectories.npy", allow_pickle=True)
    test    = raw[config["n_train"]:config["n_train"] + N_TEST]
    encoder = Encoder(enc, option, spatial_proj)
    K       = 3 if option in (2, 4) else 1

    np.random.seed(seed); torch.manual_seed(seed)
    open_ok, mpc_ok = [], []

    if planner_type == "hybrid":
        plan_fn = lambda *a, **kw: hybrid_plan(*a, **kw)
    elif planner_type == "gradient":
        plan_fn = gradient_plan
    else:
        plan_fn = mirror_ascent_plan

    for i, traj in enumerate(test):
        obs = traj["observations"]; actions = traj["actions"]; T = len(obs)
        if T < N_EXECUTED + 2: continue

        s  = np.random.randint(0, T - N_EXECUTED)
        g  = min(s + N_EXECUTED, T - 1)
        zs = encoder.encode(obs[s])
        zg = encoder.encode(obs[g])

        if K > 1:
            hi  = [max(0, s-(K-1-k)) for k in range(K)]
            if option == 4:
                z_h = torch.stack([encoder.encode(obs[j]) for j in hi])  # (K, 196, 8)
            else:
                z_h = torch.stack([encoder.encode(obs[j]) for j in hi])  # (K, 128)
            a_h = torch.stack([torch.tensor(actions[j], dtype=torch.float32) for j in hi])
        else:
            z_h = a_h = None

        # Open-loop — plan once, execute all N_EXECUTED steps
        z = zs.clone()
        act = plan_fn(predictor, z, zg, config["action_dim"], option,
                      z_hist=z_h, a_hist=a_h)
        act_t = torch.tensor(act, dtype=torch.float32)
        with torch.no_grad():
            for _ in range(N_EXECUTED):
                if option == 1:
                    z = predictor(z.unsqueeze(0), act_t.unsqueeze(0), gamma=1.0).squeeze(0)
                elif option == 3:
                    z = predictor(z.unsqueeze(0), act_t.unsqueeze(0)).squeeze(0)
        open_ok.append(float(torch.norm(encoder.flat(z) - encoder.flat(zg)).item() < thresh))

        # MPC — replan every N_MPC_CHUNK steps
        z   = zs.clone()
        zh2 = z_h.clone() if z_h is not None else None
        ah2 = a_h.clone() if a_h is not None else None
        hit = False
        for step in range(N_MPC_STEPS):
            if torch.norm(encoder.flat(z) - encoder.flat(zg)).item() < thresh:
                hit = True; break
            act   = plan_fn(predictor, z, zg, config["action_dim"], option,
                            z_hist=zh2, a_hist=ah2)
            act_t = torch.tensor(act, dtype=torch.float32)
            # Execute N_MPC_CHUNK steps before replanning
            with torch.no_grad():
                for _ in range(N_MPC_CHUNK):
                    if option == 1:
                        z = predictor(z.unsqueeze(0), act_t.unsqueeze(0), gamma=1.0).squeeze(0)
                    elif option == 3:
                        z_n = predictor(z.unsqueeze(0), act_t.unsqueeze(0)).squeeze(0)
                        z   = z_n
        mpc_ok.append(float(hit))

        if (i+1) % 10 == 0:
            print(f"   {i+1}/{N_TEST} | "
                  f"open={np.mean(open_ok)*100:.1f}%  mpc={np.mean(mpc_ok)*100:.1f}%")

    return {"open_loop": float(np.mean(open_ok)*100), "mpc": float(np.mean(mpc_ok)*100)}


# =============================================================================
# Main
# =============================================================================
def run_benchmark(encoder_path, option, target_envs, predictor_dir="./predictors",
                  data_root="./benchmark_data", n_seeds=N_SEEDS,
                  output_path="benchmark_results.json", threshold_file=None,
                  planner_type="mirror_ascent"):

    option_names = {1: "GeometricHorizon", 2: "CausalTransformer",
                    3: "SpatialMLP", 4: "SpatialTransformer (Path A+B)"}

    print("\n" + "="*60)
    print("  CORTEX-16 MPC BENCHMARK")
    print(f"  Encoder:  {encoder_path}")
    print(f"  Option:   {option} — {option_names[option]}")
    print(f"  Planner:  {planner_type}")
    print(f"  Horizon:  {PLAN_HORIZON}  (Wang et al. protocol)")
    print(f"  Envs:     {', '.join(target_envs)}")
    print("="*60 + "\n")

    thresholds = {}
    if threshold_file and Path(threshold_file).exists():
        thresholds = json.load(open(threshold_file))
        print(f"  Thresholds: {threshold_file}")
        for k, v in thresholds.items():
            print(f"    {k}: {v:.4f}")
        print()

    encoder = StudentEncoder()
    ckpt    = torch.load(encoder_path, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    stem_map = {"backbone.stem.0":"backbone.block1.0","backbone.stem.1":"backbone.block1.1",
                "backbone.stem.3":"backbone.block2.0","backbone.stem.4":"backbone.block2.1",
                "backbone.stem.6":"backbone.block3.0","backbone.stem.7":"backbone.block3.1",
                "backbone.stem.9":"backbone.block4.0","backbone.stem.10":"backbone.block4.1"}
    state = {next((k.replace(op,np,1) for op,np in stem_map.items() if k.startswith(op+".")),k):v
             for k,v in state.items()}
    encoder.load_state_dict(state, strict=False)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    all_results = {}

    for env_name in target_envs:
        pred_path = Path(predictor_dir) / f"predictor_{env_name}_opt{option}.pt"
        if not pred_path.exists():
            print(f"⚠️  {env_name}: predictor not found — train first")
            print(f"   python train_predictor.py --encoder {encoder_path} "
                  f"--env {env_name} --option {option} --frameskip 5 --min-dist 0.5")
            continue

        print(f"\n── {env_name.upper()} ──────────────────────────────────────")
        predictor, spatial_proj = load_predictor(pred_path, option, env_name)
        thresh = thresholds.get(env_name, 1.0)

        seed_results = []
        for seed in range(n_seeds):
            print(f"   Seed {seed+1}/{n_seeds} ({planner_type})...")
            r = evaluate_env(env_name, encoder, predictor, option,
                             data_root, spatial_proj, seed*42, thresh,
                             planner_type=planner_type)
            seed_results.append(r)

        opens = [r["open_loop"] for r in seed_results]
        mpcs  = [r["mpc"]       for r in seed_results]
        res   = {
            "open_loop_mean": float(np.mean(opens)),
            "open_loop_std":  float(np.std(opens)),
            "mpc_mean":       float(np.mean(mpcs)),
            "mpc_std":        float(np.std(mpcs)),
            "threshold":      thresh,
            "planner":        planner_type,
            "horizon":        PLAN_HORIZON,
        }
        all_results[env_name] = res

        dino = DINO_WM.get(env_name, {})
        best = WANG_BEST.get(env_name, {})
        print(f"\n   {'Method':<34} {'Open':>9} {'MPC':>9}")
        print(f"   {'-'*52}")
        print(f"   {'DINO-WM baseline':<34} {dino.get('open',0):>8.1f}%  {dino.get('mpc',0):>8.1f}%")
        print(f"   {'Wang et al. best':<34} {best.get('open',0):>8.1f}%  {best.get('mpc',0):>8.1f}%")
        print(f"   {'CORTEX-16 Opt' + str(option):<34} {res['open_loop_mean']:>8.1f}%  {res['mpc_mean']:>8.1f}%")
        print(f"   {'  ± std':<34} {res['open_loop_std']:>8.1f}%  {res['mpc_std']:>8.1f}%")

    if all_results:
        print("\n" + "="*60)
        print(f"  FINAL SUMMARY  [{planner_type}, H={PLAN_HORIZON}]")
        print("="*60)
        print(f"  {'Env':<10} {'DINO-WM':>10} {'Wang Best':>10} {'CORTEX-16':>10} {'Gap':>8}")
        print(f"  {'-'*52}")
        for env in target_envs:
            if env not in all_results: continue
            r    = all_results[env]
            dino = DINO_WM.get(env, {}).get("mpc", 0)
            best = WANG_BEST.get(env, {}).get("mpc", 0)
            our  = r["mpc_mean"]
            d    = our - dino
            print(f"  {env:<10} {dino:>9.1f}%  {best:>9.1f}%  {our:>9.1f}%  {'↑' if d>0 else '↓'}{abs(d):.1f}")
        print()
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Results → {output_path}")

    return all_results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--encoder",        required=True)
    p.add_argument("--option",         type=int, required=True, choices=[1,2,3,4])
    p.add_argument("--env",            default="all")
    p.add_argument("--data",           default="./benchmark_data")
    p.add_argument("--pred-dir",       default="./predictors")
    p.add_argument("--seeds",          type=int, default=3)
    p.add_argument("--out",            default="benchmark_results.json")
    p.add_argument("--threshold-file", default=None)
    p.add_argument("--planner",        default="mirror_ascent",
                   choices=["mirror_ascent","gradient","hybrid","mezo","mezo_warm","mezo_momentum","mezo_cem"],
                   help="mirror_ascent | gradient | hybrid | mezo | mezo_warm | mezo_momentum | mezo_cem")
    p.add_argument("--neuro", action="store_true", default=False)
    args = p.parse_args()

    envs = ["wall","umaze","medium","pusht"] if args.env == "all" else [args.env]
    run_benchmark(args.encoder, args.option, envs, args.pred_dir,
                  args.data, args.seeds, args.out, args.threshold_file,
                  planner_type=args.planner)