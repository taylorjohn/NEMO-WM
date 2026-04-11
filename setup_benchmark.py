"""
setup_benchmark.py — Install environments and generate trajectory data

Generates trajectory datasets for the CORTEX-16 benchmark.
Matches the format used by Wang et al. 2026 (arXiv:2603.12231).

Key fixes vs previous version:
  - render() called AFTER env.step() so frames show post-action state
  - Action amplitude amplified (×3) for diverse trajectories
  - Wall renderer uses larger motion noise for distinct consecutive frames
  - Verification step checks obs[0] != obs[1] before saving

Usage:
    python setup_benchmark.py
    python setup_benchmark.py --env umaze --skip-install
    python setup_benchmark.py --skip-install   # skip pip, regenerate all
"""

import argparse
import subprocess
import sys
from pathlib import Path


# =============================================================================
# Environment configuration
# =============================================================================
ENVS = {
    "wall": {
        "desc":    "2-room navigation with narrow door (synthetic renderer)",
        "n_train": 1920,
        "n_test":  50,
        "steps":   50,
        "epochs":  20,
    },
    "umaze": {
        "desc":    "PointMaze U-shaped maze (MuJoCo)",
        "n_train": 2000,
        "n_test":  50,
        "steps":   100,
        "epochs":  20,
    },
    "medium": {
        "desc":    "PointMaze medium maze (MuJoCo)",
        "n_train": 4000,
        "n_test":  50,
        "steps":   100,
        "epochs":  20,
    },
    "pusht": {
        "desc":    "PushT contact-rich manipulation",
        "n_train": 18500,
        "n_test":  50,
        "steps":   200,
        "epochs":  2,
    },
}

PACKAGES = [
    "gymnasium>=0.29.0",
    "gymnasium-robotics>=1.2.4",
    "gym-pusht>=0.1.0",
    "mujoco>=3.0.0",
    "imageio",
    "tqdm",
    "requests",
]


# =============================================================================
# Package installation
# =============================================================================
def install_packages():
    print("── Installing environment packages ──────────────────────────────")
    for pkg in PACKAGES:
        print(f"   pip install {pkg}")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg, "-q"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"   ⚠️  {pkg} failed: {result.stderr.strip()[:100]}")
        else:
            print(f"   ✅ {pkg}")
    print()


# =============================================================================
# Wall environment — synthetic renderer with larger action noise
# =============================================================================
def _render_wall_frame(pos, size=64):
    import numpy as np
    frame = np.ones((size, size, 3), dtype=np.uint8) * 200

    # Wall
    wx     = size // 2
    door_y1 = int(0.45 * size)
    door_y2 = int(0.55 * size)
    frame[:door_y1, wx-1:wx+1] = [50, 50, 50]
    frame[door_y2:,  wx-1:wx+1] = [50, 50, 50]

    # Agent — larger dot for more visible movement
    ax = int(pos[0] * size)
    ay = int(pos[1] * size)
    ax = int(max(3, min(size-4, ax)))
    ay = int(max(3, min(size-4, ay)))
    frame[ay-3:ay+3, ax-3:ax+3] = [220, 50, 50]

    return frame


def generate_wall_trajectories(n_trajectories: int, n_steps: int, out_dir: Path):
    import numpy as np
    out_dir.mkdir(parents=True, exist_ok=True)

    trajectories = []
    for i in range(n_trajectories):
        traj = {"observations": [], "actions": []}
        pos = np.array([np.random.uniform(0.1, 0.4), np.random.uniform(0.1, 0.9)])
        vel = np.random.randn(2) * 0.05   # larger initial velocity

        for t in range(n_steps):
            # Apply action FIRST, then render — fixes identical frame bug
            vel  = vel * 0.7 + np.random.randn(2) * 0.08   # larger noise
            vel  = np.clip(vel, -0.15, 0.15)                # larger max speed
            action = vel.copy()

            new_pos = pos + vel
            new_pos = np.clip(new_pos, 0.05, 0.95)

            # Wall collision
            if pos[0] < 0.5 and new_pos[0] >= 0.5:
                if not (0.45 <= new_pos[1] <= 0.55):
                    new_pos[0] = 0.49
                    vel[0] = -vel[0] * 0.5  # bounce
            elif pos[0] > 0.5 and new_pos[0] <= 0.5:
                if not (0.45 <= new_pos[1] <= 0.55):
                    new_pos[0] = 0.51
                    vel[0] = -vel[0] * 0.5

            pos = new_pos

            # Render AFTER moving — frame shows new position
            obs = _render_wall_frame(pos)
            traj["observations"].append(obs)
            traj["actions"].append(action)

        trajectories.append(traj)
        if (i + 1) % 200 == 0:
            print(f"   Wall: {i+1}/{n_trajectories} trajectories")

    # Verify frames are distinct
    t0 = trajectories[0]["observations"]
    import numpy as np
    if (np.array(t0[0]) == np.array(t0[1])).all():
        print("   ⚠️  Warning: consecutive wall frames still identical")
    else:
        print("   ✅ Wall frames verified distinct")

    np.save(out_dir / "trajectories.npy",
            np.array(trajectories, dtype=object), allow_pickle=True)
    print(f"   ✅ Wall: {n_trajectories} trajectories → {out_dir}")


# =============================================================================
# MuJoCo maze environments
# Key fix: render() called AFTER step(), not before
# =============================================================================
def generate_maze_trajectories(env_name: str, n_trajectories: int, n_steps: int, out_dir: Path):
    import numpy as np
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import gymnasium as gym
        import gymnasium_robotics   # noqa — registers PointMaze envs

        env_id = "PointMaze_UMaze-v3" if env_name == "umaze" else "PointMaze_Medium-v3"
        env    = gym.make(env_id, render_mode="rgb_array",
                          max_episode_steps=n_steps + 10)

        # Verify renderer produces non-identical frames
        obs, _ = env.reset()
        env.step(env.action_space.sample())
        f0 = env.render()
        env.step(env.action_space.sample() * 3.0)
        f1 = env.render()
        if (f0 == f1).all():
            print(f"   ⚠️  {env_name}: renderer may produce identical frames")
        else:
            print(f"   ✅ {env_name}: renderer produces distinct frames")

        trajectories = []
        for i in range(n_trajectories):
            obs, _ = env.reset()
            traj   = {"observations": [], "actions": []}

            for t in range(n_steps):
                # Step FIRST with amplified action
                action = env.action_space.sample() * 3.0   # amplify exploration
                obs, reward, terminated, truncated, _ = env.step(action)

                # Render AFTER step — shows post-action state
                frame = env.render()
                frame_resized = _resize_frame(frame, 64)

                traj["observations"].append(frame_resized)
                traj["actions"].append(action)

                if terminated or truncated:
                    obs, _ = env.reset()

            trajectories.append(traj)
            if (i + 1) % 500 == 0:
                print(f"   {env_name}: {i+1}/{n_trajectories} trajectories")

        env.close()

        # Final verification
        t0 = trajectories[0]["observations"]
        if (np.array(t0[0]) == np.array(t0[1])).all():
            print(f"   ⚠️  {env_name}: consecutive frames still identical after fix")
        else:
            print(f"   ✅ {env_name}: consecutive frames verified distinct")

        np.save(out_dir / "trajectories.npy",
                np.array(trajectories, dtype=object), allow_pickle=True)
        print(f"   ✅ {env_name}: {n_trajectories} trajectories → {out_dir}")

    except Exception as e:
        print(f"   ⚠️  {env_name} generation failed: {e}")
        print(f"   Falling back to synthetic trajectories (random noise)...")
        _generate_synthetic_maze(env_name, n_trajectories, n_steps, out_dir)


# =============================================================================
# PushT environment
# =============================================================================
def generate_pusht_trajectories(n_trajectories: int, n_steps: int, out_dir: Path):
    import numpy as np
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import gymnasium as gym
        import gym_pusht   # noqa

        env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array",
                       obs_type="pixels")

        trajectories = []
        for i in range(n_trajectories):
            obs, _ = env.reset()
            traj   = {"observations": [], "actions": []}

            for t in range(n_steps):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, _ = env.step(action)
                frame = env.render()   # render AFTER step
                traj["observations"].append(_resize_frame(frame, 64))
                traj["actions"].append(action)

                if terminated or truncated:
                    obs, _ = env.reset()

            trajectories.append(traj)
            if (i + 1) % 2000 == 0:
                print(f"   PushT: {i+1}/{n_trajectories} trajectories")

        env.close()
        np.save(out_dir / "trajectories.npy",
                np.array(trajectories, dtype=object), allow_pickle=True)
        print(f"   ✅ PushT: {n_trajectories} trajectories → {out_dir}")

    except Exception as e:
        print(f"   ⚠️  PushT generation failed: {e}")
        _generate_synthetic_pusht(n_trajectories, n_steps, out_dir)


# =============================================================================
# Utilities
# =============================================================================
def _resize_frame(frame, size):
    from PIL import Image
    import numpy as np
    img = Image.fromarray(frame).resize((size, size), Image.BILINEAR)
    return np.array(img)


def _generate_synthetic_maze(env_name, n_traj, n_steps, out_dir):
    import numpy as np
    # Random noise — not useful for training but avoids a crash
    trajectories = [{
        "observations": [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                         for _ in range(n_steps)],
        "actions":      [np.random.randn(2) * 0.3 for _ in range(n_steps)],
    } for _ in range(n_traj)]
    np.save(out_dir / "trajectories.npy",
            np.array(trajectories, dtype=object), allow_pickle=True)
    print(f"   ⚠️  {env_name}: synthetic fallback saved")


def _generate_synthetic_pusht(n_traj, n_steps, out_dir):
    import numpy as np
    trajectories = [{
        "observations": [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                         for _ in range(n_steps)],
        "actions":      [np.random.randn(2) * 0.5 for _ in range(n_steps)],
    } for _ in range(n_traj)]
    np.save(out_dir / "trajectories.npy",
            np.array(trajectories, dtype=object), allow_pickle=True)
    print(f"   ⚠️  PushT: synthetic fallback saved")


# =============================================================================
# Main
# =============================================================================
def setup(target_envs, skip_install):
    data_root = Path("./benchmark_data")
    data_root.mkdir(exist_ok=True)

    if not skip_install:
        install_packages()

    print("── Generating trajectory datasets ───────────────────────────────")
    print("   (This takes 5–30 minutes depending on environment)\n")

    generators = {
        "wall":   lambda out: generate_wall_trajectories(
                      ENVS["wall"]["n_train"] + ENVS["wall"]["n_test"],
                      ENVS["wall"]["steps"], out),
        "umaze":  lambda out: generate_maze_trajectories(
                      "umaze",
                      ENVS["umaze"]["n_train"] + ENVS["umaze"]["n_test"],
                      ENVS["umaze"]["steps"], out),
        "medium": lambda out: generate_maze_trajectories(
                      "medium",
                      ENVS["medium"]["n_train"] + ENVS["medium"]["n_test"],
                      ENVS["medium"]["steps"], out),
        "pusht":  lambda out: generate_pusht_trajectories(
                      ENVS["pusht"]["n_train"] + ENVS["pusht"]["n_test"],
                      ENVS["pusht"]["steps"], out),
    }

    for env_name in target_envs:
        out_dir   = data_root / env_name
        traj_file = out_dir / "trajectories.npy"
        if traj_file.exists():
            print(f"   ⏭️  {env_name}: already exists ({traj_file})")
            continue
        print(f"   Generating {env_name}...")
        generators[env_name](out_dir)

    print("\n── Summary ──────────────────────────────────────────────────────")
    for env_name in target_envs:
        traj_file = data_root / env_name / "trajectories.npy"
        status    = "✅ ready" if traj_file.exists() else "❌ missing"
        print(f"   {env_name:8s}: {status}")

    print(f"\n✅ Setup complete. Data root: {data_root.absolute()}")
    print("\nNext steps:")
    print("  python train_predictor.py --encoder ./checkpoints/cortex_student_phase1_final.pt --env umaze --option 1")
    print("  python train_predictor.py --encoder ./checkpoints/cortex_student_phase1_final.pt --env medium --option 1")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CORTEX-16 Benchmark Setup")
    p.add_argument("--env",          default="all",
                   help="wall | umaze | medium | pusht | all")
    p.add_argument("--skip-install", action="store_true")
    args = p.parse_args()

    target = list(ENVS.keys()) if args.env == "all" else [args.env]
    setup(target, args.skip_install)
