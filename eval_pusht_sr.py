"""
eval_pusht_sr.py  —  NeMo-WM PushT Planning Eval
=================================================
MPC planning loop using the trained ActionConditionedTransition world model.

Plans by rolling out K candidate action sequences through the world model,
scoring each by latent-space distance to goal, executing the best first action.
Replans every step (closed-loop MPC).

Outputs:
  - Success Rate (SR) over N episodes
  - MP4 video comparing scripted baseline vs world model planner
  - Per-episode stats: steps-to-success, final distance

Usage:
    # Eval trained checkpoint
    python eval_pusht_sr.py \\
        --ckpt  checkpoints/action_wm/action_wm_pusht_full_best.pt \\
        --n-episodes 20 \\
        --out   figures/pusht_planned.mp4

    # Compare baseline vs WM planner side by side
    python eval_pusht_sr.py \\
        --ckpt  checkpoints/action_wm/action_wm_pusht_full_best.pt \\
        --compare \\
        --n-episodes 10 \\
        --out   figures/pusht_compare.mp4

    # Just SR number, no video
    python eval_pusht_sr.py \\
        --ckpt  checkpoints/action_wm/action_wm_pusht_full_best.pt \\
        --n-episodes 50 \\
        --no-video
"""

import argparse
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# Model definitions (must match train_action_wm.py exactly)
# ═══════════════════════════════════════════════════════════════════════════

class AdaLNZero(nn.Module):
    def __init__(self, action_dim: int, d_model: int):
        super().__init__()
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 3 * d_model),
        )
        nn.init.zeros_(self.action_mlp[-1].weight)
        nn.init.zeros_(self.action_mlp[-1].bias)
        self.d_model = d_model

    def forward(self, h, a_emb):
        gamma, beta, alpha = self.action_mlp(a_emb).chunk(3, dim=-1)
        if h.dim() == 3:
            gamma = gamma.unsqueeze(1)
            beta  = beta.unsqueeze(1)
            alpha = alpha.unsqueeze(1)
        h_norm = F.layer_norm(h, [self.d_model])
        return h + 0.1 * alpha * ((1 + gamma) * h_norm + beta)


class ActionEmbedder(nn.Module):
    def __init__(self, action_dim: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
    def forward(self, a):
        return self.net(a)


class StateEncoder(nn.Module):
    def __init__(self, obs_dim: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
    def forward(self, x):
        return self.net(x)


class ActionConditionedTransition(nn.Module):
    def __init__(self, d_model: int, action_dim: int, n_layers: int = 3):
        super().__init__()
        self.action_emb = ActionEmbedder(action_dim, d_model)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            for _ in range(n_layers)
        ])
        self.adaln = nn.ModuleList([
            AdaLNZero(d_model, d_model) for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        a_emb = self.action_emb(a)
        h = z
        for layer, adaln in zip(self.layers, self.adaln):
            h = h + adaln(layer(h), a_emb)
        return self.out_norm(h)

    def rollout(self, z0, actions):
        preds = []
        z = z0
        for k in range(actions.shape[1]):
            z = self.forward(z, actions[:, k])
            preds.append(z)
        return preds


# ═══════════════════════════════════════════════════════════════════════════
# Load checkpoint
# ═══════════════════════════════════════════════════════════════════════════

def load_checkpoint(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    obs_dim    = ckpt.get("obs_dim",    5)
    action_dim = ckpt.get("action_dim", 2)
    k_steps    = ckpt.get("k_steps",   4)
    D = 128

    encoder    = StateEncoder(obs_dim, D).to(device)
    transition = ActionConditionedTransition(D, action_dim).to(device)

    encoder.load_state_dict(ckpt["encoder"])
    transition.load_state_dict(ckpt["transition"])
    encoder.eval()
    transition.eval()

    print(f"Loaded: {ckpt_path}")
    print(f"  obs_dim={obs_dim} action_dim={action_dim} k_steps={k_steps}")
    print(f"  epoch={ckpt.get('epoch','?')} loss={ckpt.get('loss',0):.4f}")
    print(f"  ac_lift={ckpt.get('diagnostics',{}).get('ac_lift',0):+.4f}")
    return encoder, transition, obs_dim, action_dim


# ═══════════════════════════════════════════════════════════════════════════
# MPC planner
# ═══════════════════════════════════════════════════════════════════════════

class MPCPlanner:
    """
    Random-shooting MPC using the trained world model.

    At each step:
      1. Encode current obs → z_current
      2. Encode goal obs   → z_goal  (or use goal state directly)
      3. Sample K random action sequences of length H
      4. Rollout each through transition model
      5. Score by ||z_H - z_goal||₂
      6. Execute first action of best sequence
      7. Replan next step

    This is the same CEM/random-shooting approach used by DINO-WM,
    simplified to random shooting for CPU tractability.
    """

    def __init__(
        self,
        encoder:    nn.Module,
        transition: nn.Module,
        action_dim: int,
        device:     torch.device,
        n_candidates: int = 512,
        horizon:    int = 8,
        action_low: float = 0.0,
        action_high: float = 1.0,
        n_elite:    int = 64,
        n_iters:    int = 3,
    ):
        self.enc      = encoder
        self.trans    = transition
        self.act_dim  = action_dim
        self.device   = device
        self.K        = n_candidates
        self.H        = horizon
        self.a_low    = action_low
        self.a_high   = action_high
        self.n_elite  = n_elite
        self.n_iters  = n_iters

    @torch.no_grad()
    def plan(
        self,
        obs:      np.ndarray,   # (obs_dim,) current observation
        goal_obs: np.ndarray,   # (obs_dim,) goal observation
    ) -> np.ndarray:
        """Returns best action (action_dim,) for current step."""
        # Skip /512 if obs already normalised (synthetic: max≤1.0, gym: max≈512)
        if obs[:4].max() > 1.5:  # gym pixel space
            obs_n  = obs.copy();      obs_n[:4]  /= 512.0
            goal_n = goal_obs.copy(); goal_n[:4] /= 512.0
        else:                        # already normalised [0,1]
            obs_n  = obs.copy()
            goal_n = goal_obs.copy()
        z_cur  = self.enc(self._to_tensor(obs_n).unsqueeze(0))   # (1, D)
        z_goal = self.enc(self._to_tensor(goal_n).unsqueeze(0))  # (1, D)

        # CEM-style: start with uniform, refine with elite samples
        mu  = torch.full((self.H, self.act_dim), 0.5, device=self.device)
        std = torch.full((self.H, self.act_dim), 0.3, device=self.device)

        for _ in range(self.n_iters):
            # Sample K action sequences: (K, H, act_dim)
            actions = torch.clamp(
                mu.unsqueeze(0) + std.unsqueeze(0) * torch.randn(
                    self.K, self.H, self.act_dim, device=self.device
                ),
                self.a_low, self.a_high
            )

            # Rollout: expand z_cur to (K, D)
            z_batch = z_cur.expand(self.K, -1)
            preds   = self.trans.rollout(z_batch, actions)
            z_final = preds[-1]   # (K, D)

            # Score: negative distance to goal (lower = closer)
            z_goal_exp = z_goal.expand(self.K, -1)
            scores = -torch.norm(z_final - z_goal_exp, dim=-1)  # (K,)

            # Elite update
            elite_idx = scores.topk(self.n_elite).indices
            elite_actions = actions[elite_idx]   # (n_elite, H, act_dim)
            mu  = elite_actions.mean(0)
            std = elite_actions.std(0).clamp(min=0.05)

        # Return first action of best sequence
        best_idx    = scores.argmax()
        best_act = actions[best_idx, 0].cpu().numpy()
        # Scale to pixel space only if obs was pixel space
        best_action = best_act * 512.0 if obs[:4].max() > 1.5 else best_act
        return best_action

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x.astype(np.float32)).to(self.device)


# ═══════════════════════════════════════════════════════════════════════════
# Scripted baseline policy
# ═══════════════════════════════════════════════════════════════════════════

def scripted_policy(obs: np.ndarray, goal_pos: np.ndarray = None) -> np.ndarray:
    """
    Naive scripted policy: move toward block, then push toward goal.
    Used as baseline comparison.
    """
    if goal_pos is None:
        goal_pos = np.array([333.0, 333.0])
    agent = obs[:2]
    block = obs[2:4]
    if np.linalg.norm(agent - block) > 60:
        target = block + np.random.normal(0, 10, 2)
    else:
        push_dir = goal_pos - block
        target   = agent + push_dir * 0.25 + np.random.normal(0, 10, 2)
    return np.clip(target, 25, 487)


# ═══════════════════════════════════════════════════════════════════════════
# PushT success criterion
# ═══════════════════════════════════════════════════════════════════════════

def is_success(obs: np.ndarray, goal_pos: np.ndarray, thr: float = 0.12) -> bool:
    """Block position within thr of goal."""
    block = obs[2:4] / 512.0
    goal_n = goal_pos / 512.0
    return float(np.linalg.norm(block - goal_n)) < thr


# ═══════════════════════════════════════════════════════════════════════════
# Episode runner
# ═══════════════════════════════════════════════════════════════════════════

def run_episode(
    env,
    planner,
    goal_obs:    np.ndarray,
    goal_pos:    np.ndarray,
    use_wm:      bool = True,
    max_steps:   int = 300,
    render:      bool = True,
    success_thr: float = 0.10,
) -> dict:
    """Run one episode, return stats and frames."""
    obs, _ = env.reset()
    frames = []
    success = False
    steps   = 0

    for step in range(max_steps):
        if render:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        if use_wm:
            action = planner.plan(obs, goal_obs)
        else:
            action = scripted_policy(obs, goal_pos)

        obs, _, terminated, truncated, _ = env.step(action)
        steps += 1

        if is_success(obs, goal_pos, success_thr):
            success = True
            # Hold success frame
            if render:
                for _ in range(20):
                    frames.append(env.render())
            break

        if terminated or truncated:
            break

    return {
        "success":    success,
        "steps":      steps,
        "final_dist": float(np.linalg.norm(obs[2:4] - goal_pos)),
        "frames":     frames,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main eval loop
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic eval (within-distribution, matches training physics)
# ═══════════════════════════════════════════════════════════════════════════

def eval_pusht_synthetic(args):
    """
    Eval on synthetic simple-Euler physics — same distribution as training.
    No gym needed. Actions and obs are in [0,1] normalised space.
    """
    import math
    device = torch.device("cpu")
    encoder, transition, obs_dim, action_dim = load_checkpoint(args.ckpt, device)
    # Load block probe for obs-space scoring if available
    _probe = None
    if hasattr(args, "probe") and args.probe and Path(args.probe).exists():
        from block_probe import load_probe, ProbeMPCPlanner
        _probe = load_probe(args.probe, device)
        print(f"  Probe scorer active — obs-space GoalDA scoring")

    # Dummy goal — updated per episode
    _dummy_goal = np.array([0.65, 0.65])
    if _probe is not None:
        from block_probe import ProbeMPCPlanner
        planner = ProbeMPCPlanner(
            encoder=encoder, transition=transition,
            action_dim=action_dim, device=device,
            probe=_probe, goal=_dummy_goal,
            n_candidates=args.n_candidates, horizon=args.horizon,
            n_elite=args.n_elite, n_iters=args.n_iters,
        )
    else:
        planner = MPCPlanner(
            encoder=encoder, transition=transition, action_dim=action_dim,
            device=device, n_candidates=args.n_candidates, horizon=args.horizon,
            n_elite=args.n_elite, n_iters=args.n_iters,
            action_low=0.0, action_high=1.0,
        )

    rng = np.random.RandomState(99)
    goals = [np.array([0.65 + rng.uniform(-0.1, 0.1),
                        0.65 + rng.uniform(-0.1, 0.1)])
             for _ in range(args.n_episodes)]

    print(f"\n── WM Planner Synthetic (K={args.n_candidates}, H={args.horizon}) ──")
    results = []

    for ep in range(args.n_episodes):
        goal = goals[ep]
        if _probe is not None:
            planner.scorer.goal = goal   # update per episode
        # Start state
        agent = rng.uniform(0.1, 0.4, 2)
        block = rng.uniform(0.3, 0.6, 2)
        angle = rng.uniform(0, 2 * math.pi)
        vel   = np.zeros(2)

        # Goal obs: agent at START, block at goal
        # Keeps agent dims unchanged so z_goal gradient points at block only
        goal_obs = np.array([agent[0], agent[1],
                             goal[0], goal[1], 0.0], dtype=np.float32)
        success = False

        # GoalDA: hot/cold guidance signal
        try:
            from hybrid_predictor import GoalDA
            goal_da = GoalDA()
        except ImportError:
            goal_da = None

        for step in range(args.max_steps):
            obs = np.array([agent[0], agent[1], block[0], block[1],
                            angle / (2 * math.pi)], dtype=np.float32)

            # Update DA and scale planning horizon (hotter = look further ahead)
            if goal_da is not None:
                da = goal_da.update(obs, goal)
                H  = max(4, min(16, int(args.horizon * (0.5 + da))))
                planner.H = H   # dynamic horizon

            # Update goal_obs with current agent pos — keeps gradient on block
            goal_obs = np.array([obs[0], obs[1],
                                 goal[0], goal[1], 0.0], dtype=np.float32)

            # Plan with WM
            action = planner.plan(obs, goal_obs)
            action = np.clip(action, 0.0, 1.0)

            # Synthetic physics step (matches training)
            agent += (action - agent) * 0.4 + rng.normal(0, 0.01, 2)
            agent  = np.clip(agent, 0, 1)
            if np.linalg.norm(agent - block) < 0.1:
                push   = (agent - block) * 0.2
                block  = np.clip(block - push, 0, 1)
                angle += rng.normal(0, 0.05)

            dist = np.linalg.norm(block - goal)
            if dist < args.success_thr:
                success = True
                break

        da_info = goal_da.summary() if goal_da else {}
        results.append({"success": success, "steps": step+1,
                        "final_dist": float(dist),
                        "da_mean": da_info.get("da_mean", 0.5)})
        temp = goal_da.temperature() if goal_da else ""
        print(f"  ep{ep+1:02d} {'✓' if success else '✗'}  "
              f"steps={step+1:3d}  dist={dist:.3f}  "
              f"DA={da_info.get('da_mean',0.5):.2f}  {temp}")

    sr    = np.mean([r["success"] for r in results])
    steps = np.mean([r["steps"]   for r in results])
    dist  = np.mean([r["final_dist"] for r in results])
    print(f"\n══ Synthetic SR: {sr:.1%}  steps={steps:.1f}  dist={dist:.3f} ══")
    return sr


def eval_pusht(args):
    device = torch.device("cpu")

    # Load world model
    encoder, transition, obs_dim, action_dim = load_checkpoint(
        args.ckpt, device
    )
    planner = MPCPlanner(
        encoder    = encoder,
        transition = transition,
        action_dim = action_dim,
        device     = device,
        n_candidates = args.n_candidates,
        horizon      = args.horizon,
        n_elite      = args.n_elite,
        n_iters      = args.n_iters,
    )

    # Init gym environment
    try:
        import gymnasium as gym
        import gym_pusht  # noqa
        env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array")
        print("gym-pusht loaded — using real pymunk physics")
    except ImportError:
        print("ERROR: gym-pusht not installed. Run: pip install gym-pusht gymnasium")
        return

    # Fixed goals per episode for fair comparison
    rng = np.random.RandomState(42)
    goals = [np.array([(0.65 + rng.uniform(-0.1, 0.1)) * 512,
                        (0.65 + rng.uniform(-0.1, 0.1)) * 512])
             for _ in range(args.n_episodes)]

    # ── World model planner eval ──────────────────────────────────────────
    print(f"\n── WM Planner (K={args.n_candidates}, H={args.horizon}, "
          f"iters={args.n_iters}) ─────────────────")
    wm_results   = []
    wm_frames_all = []

    t0 = time.time()
    for ep in range(args.n_episodes):
        # Get a goal observation by running scripted policy briefly
        obs_tmp, _ = env.reset(seed=ep * 13 + 1)
        goal_pos   = goals[ep]
        # Use a synthetic goal obs: place block at goal
        goal_obs = np.array([goal_pos[0], goal_pos[1],  # agent at goal
                              goal_pos[0], goal_pos[1],  # block at goal
                              0.0])                       # angle=0

        env.reset(seed=ep * 13)
        result = run_episode(
            env, planner, goal_obs, goal_pos,
            use_wm      = True,
            max_steps   = args.max_steps,
            render      = not args.no_video,
            success_thr = args.success_thr,
        )
        wm_results.append(result)
        if not args.no_video and result["frames"]:
            wm_frames_all.extend(result["frames"])

        status = "✓" if result["success"] else "✗"
        print(f"  ep{ep+1:02d} {status}  steps={result['steps']:3d}  "
              f"dist={result['final_dist']:.3f}")

    wm_sr    = np.mean([r["success"] for r in wm_results])
    wm_steps = np.mean([r["steps"]   for r in wm_results])
    wm_dist  = np.mean([r["final_dist"] for r in wm_results])
    wm_time  = time.time() - t0

    print(f"\n  WM Planner SR:    {wm_sr:.1%}  ({sum(r['success'] for r in wm_results)}"
          f"/{args.n_episodes})")
    print(f"  Mean steps:       {wm_steps:.1f}")
    print(f"  Mean final dist:  {wm_dist:.3f}")
    print(f"  Total time:       {wm_time:.1f}s")

    # ── Scripted baseline ─────────────────────────────────────────────────
    if args.compare:
        print(f"\n── Scripted Baseline ──────────────────────────────────────────")
        bl_results    = []
        bl_frames_all = []

        for ep in range(args.n_episodes):
            goal_pos = goals[ep]
            goal_obs = np.zeros(obs_dim)

            env.reset(seed=ep * 13)
            result = run_episode(
                env, planner, goal_obs, goal_pos,
                use_wm      = False,
                max_steps   = args.max_steps,
                render      = not args.no_video,
                success_thr = args.success_thr,
            )
            bl_results.append(result)
            if not args.no_video and result["frames"]:
                bl_frames_all.extend(result["frames"])

            status = "✓" if result["success"] else "✗"
            print(f"  ep{ep+1:02d} {status}  steps={result['steps']:3d}  "
                  f"dist={result['final_dist']:.3f}")

        bl_sr    = np.mean([r["success"] for r in bl_results])
        bl_steps = np.mean([r["steps"]   for r in bl_results])
        bl_dist  = np.mean([r["final_dist"] for r in bl_results])

        print(f"\n  Scripted SR:      {bl_sr:.1%}  ({sum(r['success'] for r in bl_results)}"
              f"/{args.n_episodes})")
        print(f"  Mean steps:       {bl_steps:.1f}")
        print(f"  Mean final dist:  {bl_dist:.3f}")

    env.close()

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n══ Sprint C PushT Results ══════════════════════════════════════")
    print(f"  WM Planner SR:  {wm_sr:.1%}")
    if args.compare:
        print(f"  Scripted SR:    {bl_sr:.1%}")
        delta = wm_sr - bl_sr
        print(f"  Δ SR:           {delta:+.1%}")
    print(f"  K={args.n_candidates} H={args.horizon} "
          f"iters={args.n_iters} thr={args.success_thr}")
    print("════════════════════════════════════════════════════════════════")

    # ── Save video ────────────────────────────────────────────────────────
    if not args.no_video:
        import imageio
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)

        if args.compare and bl_frames_all and wm_frames_all:
            # Side-by-side: baseline | WM planner
            n = min(len(bl_frames_all), len(wm_frames_all))
            combined = []
            for i in range(n):
                bl_f  = bl_frames_all[i]
                wm_f  = wm_frames_all[i]
                # Resize to same height if needed
                h = max(bl_f.shape[0], wm_f.shape[0])
                if bl_f.shape[0] != h:
                    bl_f = np.pad(bl_f, ((0, h-bl_f.shape[0]),(0,0),(0,0)))
                if wm_f.shape[0] != h:
                    wm_f = np.pad(wm_f, ((0, h-wm_f.shape[0]),(0,0),(0,0)))
                # Add divider
                div = np.zeros((h, 4, 3), dtype=np.uint8)
                frame = np.concatenate([bl_f, div, wm_f], axis=1)
                combined.append(frame)
            imageio.mimsave(args.out, combined, fps=10, macro_block_size=1)
            print(f"\nSide-by-side video: {args.out}")
            print("  Left=Scripted  Right=WM Planner")
        else:
            imageio.mimsave(args.out, wm_frames_all, fps=10, macro_block_size=1)
            print(f"\nWM planner video: {args.out}")

    return {
        "wm_sr":   wm_sr,
        "bl_sr":   bl_sr if args.compare else None,
        "wm_dist": wm_dist,
        "wm_steps": wm_steps,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",          required=True,
                   help="Path to action_wm_pusht_full_best.pt")
    p.add_argument("--n-episodes",    type=int,   default=20)
    p.add_argument("--max-steps",     type=int,   default=300)
    p.add_argument("--success-thr",   type=float, default=0.12)
    p.add_argument("--n-candidates",  type=int,   default=512,
                   help="Random shooting candidates K")
    p.add_argument("--horizon",       type=int,   default=8,
                   help="Planning horizon H")
    p.add_argument("--n-elite",       type=int,   default=64,
                   help="CEM elite samples")
    p.add_argument("--n-iters",       type=int,   default=3,
                   help="CEM iterations")
    p.add_argument("--compare",       action="store_true",
                   help="Also run scripted baseline for comparison")
    p.add_argument("--no-video",      action="store_true")
    p.add_argument("--probe",          default=None,
                   help="Path to block_probe_best.pt for obs-space scoring")
    p.add_argument("--synthetic",     action="store_true",
                   help="Eval on synthetic physics (matches training dist)")
    p.add_argument("--out",           default="figures/pusht_planned.mp4")
    args = p.parse_args()

    if args.synthetic:
        eval_pusht_synthetic(args)
    else:
        eval_pusht(args)
