"""
generate_pusht_data.py — Generate PushT HDF5 data for NeMo-WM
==============================================================
Generates PushT episodes directly from gym_pusht environment,
saving in the exact HDF5 format expected by train_cwm_v2.py:

    images/rgb_left  — JPEG bytes (same as RECON)
    commands/linear_velocity  — action dim 0
    commands/angular_velocity — action dim 1
    gps/latlong      — block XY position (used as spatial signal for NE)

Install:
    pip install gym-pusht pygame --break-system-packages

Run:
    python generate_pusht_data.py --n-episodes 500 --dest recon_data\pushtoom\pusht
    python generate_pusht_data.py --n-episodes 100 --dest recon_data\pushtoom\pusht --quick

Author: John Taylor — github.com/taylorjohn
Date:   2026-04-06
Sprint: 9e (PushT benchmark)
"""

import argparse
import io
import random
import time
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

try:
    import gym_pusht  # noqa
    import gymnasium as gym
    GYM_OK = True
except ImportError:
    GYM_OK = False

IMG_SIZE = 224   # resize to match RECON frames


def generate_episode(env, max_steps: int = 200, render_size: int = 96) -> dict:
    """
    Run one random-action episode and return dict of arrays.
    Returns None if episode has zero valid frames.
    """
    obs, _ = env.reset()
    frames, actions, positions = [], [], []

    for _ in range(max_steps):
        # Random action in PushT action space
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Get rendered frame
        frame = env.render()  # (H, W, 3) uint8
        if frame is None:
            # Fallback: extract pixels from obs dict
            if isinstance(obs, dict) and "pixels" in obs:
                frame = obs["pixels"]
            elif isinstance(obs, dict) and "observation" in obs:
                frame = obs["observation"]
            else:
                continue  # skip frame if no image available
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            continue
        img = Image.fromarray(frame.astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE))

        # JPEG encode (matches RECON format)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=85)
        jpeg_bytes = np.frombuffer(buf.getvalue(), dtype=np.uint8)

        frames.append(jpeg_bytes)
        actions.append(action.astype(np.float32))

        # Block XY position as spatial proxy for GPS signal
        block_pos = obs.get("block_pose", np.zeros(2))[:2] \
            if isinstance(obs, dict) else np.zeros(2)
        positions.append(block_pos.astype(np.float64))

        if terminated or truncated:
            break

    if len(frames) < 5:
        return None

    return {
        "frames":    frames,
        "actions":   np.stack(actions),     # (T, 2)
        "positions": np.stack(positions),   # (T, 2)
    }


def write_hdf5(path: Path, episode: dict):
    """Write episode to HDF5 in NeMo-WM format."""
    T = len(episode["frames"])
    with h5py.File(path, 'w') as hf:
        imgs_grp = hf.create_group("images")

        # Variable-length JPEG bytes — same as RECON
        try:
            dt = h5py.vlen_dtype(np.dtype('uint8'))
        except AttributeError:
            dt = h5py.special_dtype(vlen=np.dtype('uint8'))
        ds = imgs_grp.create_dataset("rgb_left", shape=(T,), dtype=dt)
        for i, jpeg in enumerate(episode["frames"]):
            ds[i] = jpeg

        # Verify images were written
        assert "rgb_left" in imgs_grp, "rgb_left write failed"

        # Actions: use dim 0 as linear_vel, dim 1 as angular_vel
        actions = episode["actions"]
        hf.create_dataset("commands/linear_velocity",
                          data=actions[:, 0])
        hf.create_dataset("commands/angular_velocity",
                          data=actions[:, 1] if actions.shape[1] > 1
                          else np.zeros(T))

        # Block XY as GPS proxy (NE signal grounding)
        hf.create_dataset("gps/latlong", data=episode["positions"])


def verify_hdf5(path: Path) -> bool:
    """Quick sanity check on written file."""
    try:
        with h5py.File(path, 'r') as hf:
            T = hf["images"]["rgb_left"].shape[0]
            assert T >= 5, f"Too few frames: {T}"
            # Verify JPEG decode works
            jpeg = bytes(hf["images"]["rgb_left"][0])
            img = Image.open(io.BytesIO(jpeg))
            assert img.size == (IMG_SIZE, IMG_SIZE)
            assert "commands" in hf
            assert "gps" in hf
        return True
    except Exception as e:
        print(f"  Verify failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate PushT HDF5 data for NeMo-WM"
    )
    parser.add_argument("--n-episodes",  type=int, default=500)
    parser.add_argument("--max-steps",   type=int, default=200)
    parser.add_argument("--dest",        default="recon_data/pushtoom/pusht")
    parser.add_argument("--quick",       action="store_true",
                        help="50 episodes only (smoke test)")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    if not GYM_OK:
        print("gym-pusht not installed. Run:")
        print("  pip install gym-pusht pygame --break-system-packages")
        return

    n = 50 if args.quick else args.n_episodes
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating PushT data: {n} episodes → {dest}")
    print(f"  Frame size: {IMG_SIZE}×{IMG_SIZE} (JPEG, NeMo-WM format)")
    print(f"  Max steps per episode: {args.max_steps}")
    print()

    # Check existing files
    existing = list(dest.glob("pusht_*.hdf5"))
    start_idx = len(existing)
    if start_idx > 0:
        print(f"  Found {start_idx} existing episodes — resuming from {start_idx}")

    try:
        env = gym.make(
            "gym_pusht/PushT-v0",
            obs_type="pixels_agent_pos",
            render_mode="rgb_array",
        )
    except Exception as e:
        print(f"Failed to create PushT env: {e}")
        print("Try: pip install gym-pusht --break-system-packages")
        return

    random.seed(args.seed)
    np.random.seed(args.seed)

    t0 = time.perf_counter()
    written = 0
    failed  = 0

    for ep_idx in range(start_idx, start_idx + n):
        episode = generate_episode(env, args.max_steps)
        if episode is None:
            failed += 1
            continue

        path = dest / f"pusht_{ep_idx:05d}.hdf5"
        write_hdf5(path, episode)

        if not verify_hdf5(path):
            path.unlink()
            failed += 1
            continue

        written += 1
        T = len(episode["frames"])

        if written % 25 == 0 or written == 1:
            elapsed = time.perf_counter() - t0
            rate = written / elapsed
            eta  = (n - written) / rate if rate > 0 else 0
            print(f"  Episode {ep_idx:04d} | T={T:3d} frames | "
                  f"{written}/{n} written | "
                  f"{rate:.1f} ep/s | eta {eta:.0f}s",
                  flush=True)

    env.close()
    elapsed = time.perf_counter() - t0

    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Written: {written} episodes")
    print(f"  Failed:  {failed} episodes")
    print(f"  Location: {dest.resolve()}")
    print()

    # Verify a sample
    samples = sorted(dest.glob("pusht_*.hdf5"))[-3:]  # newest files
    print("Sample verification:")
    for p in samples:
        with h5py.File(p) as hf:
            T = hf["images"]["rgb_left"].shape[0]
            lin = hf["commands"]["linear_velocity"][0]
            print(f"  {p.name}: T={T}, lin_vel[0]={lin:.3f}")

    print()
    print("Train NeMo-WM on PushT:")
    print(f"  python train_cwm_v2.py \\")
    print(f"      --hdf5-dir  {dest} \\")
    print(f"      --epochs    10 \\")
    print(f"      --save-path checkpoints\\cwm\\cwm_pushtoom_best.pt")


if __name__ == "__main__":
    main()
