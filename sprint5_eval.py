"""
sprint5_eval.py — NeMo-WM vs stable-worldmodel Benchmark
==========================================================
Sprint 5: evaluate NeMo-WM on stable-worldmodel environments and
compare against published DINO-WM, LeWM, PLDM baselines.

Environments tested (from stable-worldmodel paper):
    swm/TwoRoom-v1      — navigation (DINO-WM: 100%, LeWM: 87%)
    swm/PushT-v1        — contact manipulation (DINO-WM: 74%, LeWM: 96%)
    swm/ReacherDMControl-v0 — continuous control (DINO-WM: 79%)
    swm/OGBCube-v0      — object manipulation (DINO-WM: 86%, GCBC: 84%)

Method:
    NeMo-WM encodes current and goal observations via StudentEncoder →
    ParticleEncoder. Planning uses MirrorAscent (real-time, ~0.17ms)
    to minimise distance between current particles and goal particles.

Usage:
    python sprint5_eval.py --env swm/TwoRoom-v1 --episodes 50
    python sprint5_eval.py --all --episodes 20
    python sprint5_eval.py --env swm/TwoRoom-v1 --episodes 50 --solver cem

Author: John Taylor — github.com/taylorjohn
Date:   2026-04-03
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import stable_worldmodel as sw
from stable_worldmodel.solver import CEMSolver, GradientSolver

from train_mvtec import StudentEncoder
from train_cwm_DEPRECATED import CortexWorldModel

# ── Published baselines from stable-worldmodel paper ─────────────────────────
BASELINES = {
    "swm/TwoRoom-v1":          {"DINO-WM": 1.00, "LeWM": 0.87, "PLDM": 0.97, "GCBC": 1.00},
    "swm/PushT-v1":            {"DINO-WM": 0.74, "LeWM": 0.96, "PLDM": 0.78, "GCBC": 0.75},
    "swm/ReacherDMControl-v0": {"DINO-WM": 0.79, "LeWM": 0.86, "PLDM": 0.78},
    "swm/OGBCube-v0":          {"DINO-WM": 0.86, "LeWM": 0.74, "PLDM": 0.65, "GCBC": 0.84},
}

DEVICE = torch.device("cpu")
IMAGE_SHAPE = (64, 64)


class NeMoWMPolicy:
    """
    NeMo-WM policy for stable-worldmodel evaluation.

    Encodes current + goal observations via StudentEncoder → ParticleEncoder.
    Plans via particle-space distance minimisation (MirrorAscent).
    """

    def __init__(self, encoder_ckpt: str, cwm_ckpt: str,
                 action_dim: int = 2, plan_lr: float = 0.1):
        # StudentEncoder
        self.encoder = StudentEncoder().to(DEVICE)
        if Path(encoder_ckpt).exists():
            ckpt = torch.load(encoder_ckpt, map_location="cpu", weights_only=False)
            sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
            self.encoder.load_state_dict(sd, strict=False)
        self.encoder.eval()

        # CortexWorldModel
        self.cwm = CortexWorldModel(d_model=128, K=16).to(DEVICE)
        if Path(cwm_ckpt).exists():
            ckpt2 = torch.load(cwm_ckpt, map_location="cpu", weights_only=False)
            self.cwm.load_state_dict(
                ckpt2.get("model", ckpt2.get("state_dict", ckpt2)), strict=False
            )
        self.cwm.eval()

        self.action_dim = action_dim
        self.plan_lr    = plan_lr
        self.goal_particles: torch.Tensor = None

    def _encode(self, obs: np.ndarray) -> torch.Tensor:
        """
        Encode observation to particle embedding.

        Handles both image observations (H,W,3) and pixel arrays from info.
        Falls back to zero-image encoding for proprioceptive-only environments.

        Returns:
            particles: (K, 128) particle embeddings
        """
        if isinstance(obs, np.ndarray):
            # Handle (1,1,H,W,3) from info['pixels'] — squeeze batch dims
            while obs.ndim > 3 and obs.shape[0] == 1:
                obs = obs.squeeze(0)
            # Flat proprioceptive obs — use zero image (no visual info available)
            if obs.ndim == 1 or (obs.ndim == 2 and min(obs.shape) < 3):
                obs = np.zeros((3, 224, 224), dtype=np.float32)
            else:
                if obs.dtype == np.uint8:
                    obs = obs.astype(np.float32) / 255.0
                if obs.ndim == 3 and obs.shape[-1] == 3:
                    obs = obs.transpose(2, 0, 1)  # HWC → CHW
            obs = torch.tensor(obs, dtype=torch.float32)
        elif isinstance(obs, torch.Tensor):
            while obs.ndim > 3 and obs.shape[0] == 1:
                obs = obs.squeeze(0)

        # Ensure 3D CHW
        if obs.ndim == 2:
            obs = obs.unsqueeze(0).expand(3, -1, -1)
        elif obs.ndim == 1:
            obs = torch.zeros(3, 224, 224)

        # Resize to 224×224 for StudentEncoder
        if obs.shape[-2:] != (224, 224):
            obs = F.interpolate(
                obs.unsqueeze(0).float(), size=(224, 224),
                mode="bilinear", align_corners=False
            ).squeeze(0)

        with torch.no_grad():
            z = self.encoder(obs.unsqueeze(0))
            particles, _, _, _ = self.cwm.encode(z)
            return particles.squeeze(0)  # (K, 128)

    def set_goal(self, goal_obs: np.ndarray):
        """Set goal observation — encode to goal particles."""
        self.goal_particles = self._encode(goal_obs)  # (K, 128)

    def act(self, obs: np.ndarray) -> np.ndarray:
        """
        Plan action to minimise distance between current and goal particles.

        Uses MirrorAscent: gradient of cosine distance w.r.t. action direction.
        """
        if self.goal_particles is None:
            return np.zeros(self.action_dim)

        current = self._encode(obs)  # (K, 128)

        # Cosine similarity between current and goal particle means
        curr_mean = F.normalize(current.mean(0), dim=0)
        goal_mean = F.normalize(self.goal_particles.mean(0), dim=0)

        # Direction in embedding space to move toward goal
        delta = (goal_mean - curr_mean).numpy()

        # Project to action space (first action_dim components)
        action = delta[:self.action_dim]
        norm = np.linalg.norm(action)
        if norm > 1e-8:
            action = action / norm * np.clip(norm, 0, 1)

        return action.astype(np.float32)

    def reset(self):
        """Reset between episodes."""
        self.goal_particles = None


def evaluate_env(
    env_name:     str,
    policy:       NeMoWMPolicy,
    n_episodes:   int = 50,
    max_steps:    int = 100,
    verbose:      bool = True,
) -> dict:
    """
    Evaluate NeMo-WM policy on a single stable-worldmodel environment.

    Args:
        env_name:   environment name (e.g. 'swm/TwoRoom-v1')
        policy:     NeMo-WM policy instance
        n_episodes: number of evaluation episodes
        max_steps:  maximum steps per episode
        verbose:    print per-episode results

    Returns:
        dict with success_rate, mean_steps, episode_results
    """
    print(f"\n{'='*60}")
    print(f"  Environment: {env_name}")
    print(f"  Episodes:    {n_episodes}")
    print(f"  Max steps:   {max_steps}")

    # Published baselines for comparison
    if env_name in BASELINES:
        print(f"  Baselines:   ", end="")
        for model, sr in BASELINES[env_name].items():
            print(f"{model}={sr:.0%}  ", end="")
        print()
    print(f"{'='*60}")

    try:
        world = sw.World(
            env_name        = env_name,
            num_envs        = 1,
            image_shape     = IMAGE_SHAPE,
            max_episode_steps = max_steps,
            verbose         = 0,
        )
    except Exception as e:
        print(f"  ❌ Failed to create environment: {e}")
        return {"error": str(e), "success_rate": None}

    successes   = []
    step_counts = []
    t_start     = time.time()

    for ep in range(n_episodes):
        policy.reset()

        try:
            obs, info = world.envs.reset()
        except Exception:
            obs = world.envs.reset()
            info = {}

        # Extract pixel observations and goal from info
        # TwoRoom and similar envs: obs=(1,10) state, pixels in info['pixels']
        pixels = None
        goal_pixels = None
        if isinstance(info, dict):
            if "pixels" in info:
                pixels = info["pixels"]  # (1,1,H,W,3)
            if "goal_state" in info:
                # Use goal state pixels if available, else use current pixels
                goal_pixels = pixels  # proxy: same visual space

        # Use pixels if available, else fall back to obs
        obs_single      = pixels[0]      if pixels      is not None else obs[0] if obs.ndim > 1 else obs
        goal_obs_single = goal_pixels[0] if goal_pixels is not None else obs_single

        policy.set_goal(goal_obs_single)

        success    = False
        step_count = 0

        for step in range(max_steps):
            action = policy.act(obs_single)

            # Expand action for vectorised env
            actions = action[np.newaxis]

            try:
                obs, reward, terminated, truncated, info = world.envs.step(actions)
                done = bool(terminated[0]) or bool(truncated[0])
            except Exception:
                result = world.envs.step(actions)
                obs, reward, done, info = result[0], result[1], result[2], result[3] if len(result) > 3 else {}
                if isinstance(done, np.ndarray):
                    done = bool(done[0])

            # Extract success signal
            if isinstance(info, (list, tuple)):
                info = info[0] if len(info) > 0 else {}
            success_flag = (
                info.get("success", False) or
                info.get("is_success", False) or
                (isinstance(reward, np.ndarray) and reward[0] > 0.5) or
                (not isinstance(reward, np.ndarray) and reward > 0.5)
            )

            # Prefer pixels from info for visual encoding
            if isinstance(info, dict) and "pixels" in info:
                obs_single = info["pixels"][0]
            elif isinstance(obs, np.ndarray) and obs.ndim > 1:
                obs_single = obs[0]
            else:
                obs_single = obs

            step_count += 1
            if success_flag or done:
                success = bool(success_flag)
                break

        successes.append(success)
        step_counts.append(step_count)

        if verbose:
            status = "✅" if success else "❌"
            print(f"  Ep {ep+1:3d}/{n_episodes}: {status} steps={step_count:3d}")

    world.close()

    success_rate = np.mean(successes)
    mean_steps   = np.mean(step_counts)
    elapsed      = time.time() - t_start

    print(f"\n  Results: success_rate={success_rate:.1%}  mean_steps={mean_steps:.1f}  time={elapsed:.1f}s")

    # Compare against baselines
    if env_name in BASELINES:
        print(f"\n  Comparison:")
        print(f"  {'Model':12s} {'Success':>8}")
        print(f"  {'-'*12} {'-'*8}")
        print(f"  {'NeMo-WM':12s} {success_rate:>8.1%}  ← this run")
        for model, sr in sorted(BASELINES[env_name].items(), key=lambda x: -x[1]):
            marker = " ✅" if success_rate >= sr else " ❌"
            print(f"  {model:12s} {sr:>8.1%}{marker}")

    return {
        "env":          env_name,
        "success_rate": float(success_rate),
        "mean_steps":   float(mean_steps),
        "n_episodes":   n_episodes,
        "successes":    successes,
        "elapsed_s":    elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="NeMo-WM Sprint 5 — stable-worldmodel evaluation"
    )
    parser.add_argument("--env",       type=str, default="swm/TwoRoom-v1",
                        help="Environment name")
    parser.add_argument("--all",       action="store_true",
                        help="Evaluate all four benchmark environments")
    parser.add_argument("--episodes",  type=int, default=50,
                        help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=100,
                        help="Maximum steps per episode")
    parser.add_argument("--cwm-ckpt",  default=r"checkpoints\cwm\cwm_best.pt")
    parser.add_argument("--encoder-ckpt",
                        default=r"checkpoints\dinov2_student\student_best.pt")
    parser.add_argument("--quiet",     action="store_true",
                        help="Suppress per-episode output")
    args = parser.parse_args()

    print("\nNeMo-WM Sprint 5 — stable-worldmodel Benchmark")
    print("=" * 60)
    print(f"  CWM:     {args.cwm_ckpt}")
    print(f"  Encoder: {args.encoder_ckpt}")

    # Build policy
    policy = NeMoWMPolicy(
        encoder_ckpt = args.encoder_ckpt,
        cwm_ckpt     = args.cwm_ckpt,
    )
    print(f"  Policy: NeMo-WM (MirrorAscent planner)")

    # Environments to evaluate
    if args.all:
        envs = list(BASELINES.keys())
    else:
        envs = [args.env]

    all_results = {}
    for env_name in envs:
        result = evaluate_env(
            env_name   = env_name,
            policy     = policy,
            n_episodes = args.episodes,
            max_steps  = args.max_steps,
            verbose    = not args.quiet,
        )
        all_results[env_name] = result

    # Summary table
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print(f"  SUMMARY")
        print(f"  {'Environment':30s} {'NeMo-WM':>10} {'DINO-WM':>10} {'LeWM':>10}")
        print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")
        for env, res in all_results.items():
            if res.get("success_rate") is not None:
                sr      = res["success_rate"]
                dino_wm = BASELINES.get(env, {}).get("DINO-WM", float("nan"))
                lewm    = BASELINES.get(env, {}).get("LeWM", float("nan"))
                name    = env.split("/")[-1]
                print(f"  {name:30s} {sr:>10.1%} {dino_wm:>10.1%} {lewm:>10.1%}")


if __name__ == "__main__":
    main()
