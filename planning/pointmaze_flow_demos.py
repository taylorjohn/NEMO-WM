"""
pointmaze_flow_demos.py — PointMaze Demo Generator for Flow Policy
===================================================================
Generates scripted demos for PointMaze-UMaze using A* waypoints + PD controller.
Same format as generate_orientation_demos() for direct use with NeMoV2Dataset.

UMaze layout (3x3 grid, walls marked #):
    # # #
    . . #
    # . #
    # . .
    # # #

Coordinate system: x ∈ [0, 3], y ∈ [0, 3] (grid units)
Normalized to [-1, 1] for training.

Demo format:
    {'obs': (T, 4), 'actions': (T, 2), 'goal': (2,), 'success': bool}
    obs = (x, y, vx, vy) normalized to [-1, 1]
    actions = (force_x, force_y) in [-1, 1]
    goal = (goal_x, goal_y) normalized to [-1, 1]

Usage:
    from pointmaze_flow_demos import generate_pointmaze_demos
    demos = generate_pointmaze_demos(n_demos=1000, H=8)

    # Or with Minari real data:
    from pointmaze_flow_demos import load_minari_demos
    demos = load_minari_demos(n_demos=2000)

Author: John Taylor
Sprint: PointMaze flow policy port
"""

import math
from typing import List, Dict, Tuple, Optional

import numpy as np


# ── UMaze geometry ───────────────────────────────────────────────────────────

# Walls: cells that are blocked (row, col) in a 5x3 grid
# UMaze shape:
#   row 0: wall wall wall
#   row 1: open open wall
#   row 2: wall open wall
#   row 3: wall open open
#   row 4: wall wall wall
UMAZE_WALLS = {
    (0, 0), (0, 1), (0, 2),
    (1, 2),
    (2, 0), (2, 2),
    (3, 0),
    (4, 0), (4, 1), (4, 2),
}

UMAZE_OPEN = [
    (1, 0), (1, 1),
    (2, 1),
    (3, 1), (3, 2),
]

# Waypoints for navigating the U (in grid coords, cell centers)
UMAZE_WAYPOINTS_FORWARD = [
    (1, 0), (1, 1), (2, 1), (3, 1), (3, 2),
]
UMAZE_WAYPOINTS_REVERSE = list(reversed(UMAZE_WAYPOINTS_FORWARD))

# Convert grid cell (row, col) to continuous coords
# Cell (r, c) has center at (c + 0.5, r + 0.5) in (x, y)
def cell_to_pos(row, col):
    return np.array([col + 0.5, row + 0.5], dtype=np.float32)

# Position bounds for normalization
POS_MIN = np.array([0.0, 0.0], dtype=np.float32)
POS_MAX = np.array([3.0, 5.0], dtype=np.float32)


def normalize_pos(pos):
    """Normalize position to [-1, 1]."""
    return 2.0 * (pos - POS_MIN) / (POS_MAX - POS_MIN) - 1.0

def denormalize_pos(pos_norm):
    """Denormalize from [-1, 1] to world coords."""
    return (pos_norm + 1.0) / 2.0 * (POS_MAX - POS_MIN) + POS_MIN

def normalize_vel(vel, max_vel=2.0):
    """Normalize velocity to [-1, 1]."""
    return np.clip(vel / max_vel, -1.0, 1.0)


# ── Collision detection ──────────────────────────────────────────────────────

def is_wall(pos):
    """Check if position is inside a wall cell."""
    col = int(pos[0])
    row = int(pos[1])
    col = max(0, min(col, 2))
    row = max(0, min(row, 4))
    return (row, col) in UMAZE_WALLS

def clip_to_open(pos, prev_pos):
    """If pos is in a wall, return prev_pos (bounce)."""
    if is_wall(pos) or pos[0] < 0.1 or pos[0] > 2.9 or pos[1] < 0.1 or pos[1] > 4.9:
        return prev_pos.copy()
    return pos


# ── Simple physics ───────────────────────────────────────────────────────────

def step_physics(pos, vel, force, dt=0.1, friction=0.5, max_vel=2.0):
    """
    Simple 2D point mass with friction.
    force ∈ [-1, 1]^2 → acceleration
    """
    acc = force * 5.0  # scale force to acceleration
    vel_new = vel + acc * dt - friction * vel * dt
    vel_new = np.clip(vel_new, -max_vel, max_vel)
    pos_new = pos + vel_new * dt
    # Wall collision
    pos_clipped = clip_to_open(pos_new, pos)
    if np.any(pos_clipped != pos_new):
        vel_new *= -0.2  # bounce with damping
    return pos_clipped, vel_new


# ── PD Controller ────────────────────────────────────────────────────────────

def pd_controller(pos, vel, target, kp=3.0, kd=1.5):
    """PD controller: force = kp * (target - pos) - kd * vel."""
    err = target - pos
    force = kp * err - kd * vel
    return np.clip(force, -1.0, 1.0)


# ── A* waypoint planning ────────────────────────────────────────────────────

def plan_waypoints(start_pos, goal_pos):
    """
    Plan waypoints through the UMaze from start to goal.
    Uses the known U-shape topology — no general A* needed.
    """
    # Find nearest open cells
    start_cell = min(UMAZE_OPEN,
                     key=lambda c: np.linalg.norm(cell_to_pos(*c) - start_pos))
    goal_cell = min(UMAZE_OPEN,
                    key=lambda c: np.linalg.norm(cell_to_pos(*c) - goal_pos))

    start_idx = UMAZE_WAYPOINTS_FORWARD.index(start_cell)
    goal_idx = UMAZE_WAYPOINTS_FORWARD.index(goal_cell)

    if start_idx <= goal_idx:
        path = UMAZE_WAYPOINTS_FORWARD[start_idx:goal_idx + 1]
    else:
        path = UMAZE_WAYPOINTS_FORWARD[goal_idx:start_idx + 1][::-1]

    waypoints = [cell_to_pos(r, c) for r, c in path]
    # Replace last waypoint with exact goal
    waypoints[-1] = goal_pos.copy()
    return waypoints


# ── Demo Generator ───────────────────────────────────────────────────────────

def generate_pointmaze_demos(
    n_demos:    int = 1000,
    H:          int = 8,
    max_steps:  int = 300,
    pos_thresh: float = 0.15,
    seed:       int = 42,
) -> List[Dict]:
    """
    Generate scripted PointMaze-UMaze demos.

    Each demo:
        1. Random start and goal in open cells
        2. A* waypoints through the U
        3. PD controller follows waypoints
        4. Success if within pos_thresh of goal

    Returns list of dicts:
        {'obs': (T, 4), 'actions': (T, 2), 'goal': (2,), 'success': bool}
        All normalized to [-1, 1].
    """
    rng = np.random.RandomState(seed)
    demos = []

    for demo_idx in range(n_demos):
        # Random start and goal (different cells)
        cells = rng.choice(len(UMAZE_OPEN), 2, replace=False)
        start_cell = UMAZE_OPEN[cells[0]]
        goal_cell = UMAZE_OPEN[cells[1]]

        # Add noise within cell
        start_pos = cell_to_pos(*start_cell) + rng.uniform(-0.3, 0.3, 2)
        goal_pos = cell_to_pos(*goal_cell) + rng.uniform(-0.3, 0.3, 2)
        start_pos = np.clip(start_pos, [0.2, 0.2], [2.8, 4.8])
        goal_pos = np.clip(goal_pos, [0.2, 0.2], [2.8, 4.8])

        # Plan waypoints
        waypoints = plan_waypoints(start_pos, goal_pos)
        wp_idx = 0

        pos = start_pos.copy()
        vel = np.zeros(2, dtype=np.float32)

        obs_list = []
        action_list = []
        success = False

        for step in range(max_steps):
            # Current observation (normalized)
            obs = np.concatenate([
                normalize_pos(pos),
                normalize_vel(vel),
            ]).astype(np.float32)
            obs_list.append(obs)

            # Target: current waypoint
            target = waypoints[wp_idx]

            # Advance waypoint if close enough
            if np.linalg.norm(pos - target) < 0.3 and wp_idx < len(waypoints) - 1:
                wp_idx += 1
                target = waypoints[wp_idx]

            # PD control with noise
            force = pd_controller(pos, vel, target)
            noise = rng.randn(2).astype(np.float32) * 0.05
            force = np.clip(force + noise, -1.0, 1.0)
            action_list.append(force.copy())

            # Physics step
            pos, vel = step_physics(pos, vel, force)

            # Check success
            if np.linalg.norm(pos - goal_pos) < pos_thresh:
                success = True
                # Pad remaining steps with zero actions at goal
                for _ in range(H):
                    obs_final = np.concatenate([
                        normalize_pos(pos),
                        normalize_vel(vel * 0.9),  # friction
                    ]).astype(np.float32)
                    obs_list.append(obs_final)
                    action_list.append(np.zeros(2, dtype=np.float32))
                    vel *= 0.9
                break

        demos.append({
            'obs': np.array(obs_list, dtype=np.float32),
            'actions': np.array(action_list, dtype=np.float32),
            'goal': normalize_pos(goal_pos).astype(np.float32),
            'success': success,
        })

    sr = sum(d['success'] for d in demos) / len(demos)
    print(f"  PointMaze demos: {n_demos} generated, SR={sr:.1%}")
    return demos


# ── Load from Minari ─────────────────────────────────────────────────────────

def load_minari_demos(n_demos: int = 2000, H: int = 8) -> List[Dict]:
    """
    Load real PointMaze demos from Minari D4RL dataset.
    Auto-downloads if not cached.
    """
    import minari
    ds = minari.load_dataset("D4RL/pointmaze/umaze-v2", download=True)
    demos = []
    count = 0

    for ep in ds.iterate_episodes():
        if count >= n_demos:
            break

        obs_raw = ep.observations["observation"].astype(np.float32)
        actions = ep.actions.astype(np.float32)
        T = min(len(obs_raw), len(actions))

        if T < H + 2:
            continue

        # obs_raw: (x, y, vx, vy) — normalize
        pos = obs_raw[:T, :2]
        vel = obs_raw[:T, 2:4]

        # Normalize using D4RL bounds
        pos_norm = 2.0 * (pos - np.array([-2, -2])) / np.array([12, 12]) - 1.0
        vel_norm = np.clip(vel / 2.0, -1.0, 1.0)
        obs_norm = np.concatenate([pos_norm, vel_norm], axis=1)

        # Goal = last position
        goal = pos_norm[-1, :2].copy()

        demos.append({
            'obs': obs_norm[:T],
            'actions': np.clip(actions[:T], -1.0, 1.0),
            'goal': goal,
            'success': True,  # D4RL demos are expert trajectories
        })
        count += 1

    print(f"  Minari demos: {len(demos)} loaded")
    return demos


# ── Eval: open-loop flow policy on PointMaze ─────────────────────────────────

def eval_pointmaze_flow(policy, encoder, n_episodes=50, H=8, seed=42):
    """
    Evaluate flow policy on PointMaze with simple physics.
    Returns SR and mean steps.
    """
    import torch
    import torch.nn.functional as F

    rng = np.random.RandomState(seed)
    successes = 0
    total_steps = 0

    for ep in range(n_episodes):
        cells = rng.choice(len(UMAZE_OPEN), 2, replace=False)
        start_pos = cell_to_pos(*UMAZE_OPEN[cells[0]]) + rng.uniform(-0.3, 0.3, 2)
        goal_pos = cell_to_pos(*UMAZE_OPEN[cells[1]]) + rng.uniform(-0.3, 0.3, 2)
        start_pos = np.clip(start_pos, [0.2, 0.2], [2.8, 4.8])
        goal_pos = np.clip(goal_pos, [0.2, 0.2], [2.8, 4.8])

        pos = start_pos.copy()
        vel = np.zeros(2, dtype=np.float32)
        goal_norm = normalize_pos(goal_pos)

        for step in range(300):
            obs = np.concatenate([normalize_pos(pos), normalize_vel(vel)])
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            goal_t = torch.from_numpy(goal_norm).float().unsqueeze(0)

            with torch.no_grad():
                z = encoder(obs_t)
                # Generate action trajectory via flow
                x = torch.randn(1, H * 2)
                da = torch.tensor([[0.5]])
                for t_step in range(10):
                    t_val = torch.tensor([[t_step / 10.0]])
                    v = policy(z, obs_t, da, goal_t, t_val, x)
                    x = x - v * 0.1
                actions = x.view(1, H, 2).squeeze(0).numpy()

            # Execute first action
            force = np.clip(actions[0], -1.0, 1.0)
            pos, vel = step_physics(pos, vel, force)

            if np.linalg.norm(pos - goal_pos) < 0.15:
                successes += 1
                total_steps += step + 1
                break
        else:
            total_steps += 300

    sr = successes / n_episodes
    avg_steps = total_steps / n_episodes
    print(f"  PointMaze Flow Policy: SR={sr:.1%} ({successes}/{n_episodes}) "
          f"avg_steps={avg_steps:.0f}")
    return sr, avg_steps


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-demos", type=int, default=1000)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--minari", action="store_true",
                    help="Load from Minari instead of generating")
    args = ap.parse_args()

    print("=" * 60)
    print("  PointMaze Flow Policy — Demo Generator")
    print("=" * 60)

    if args.minari:
        demos = load_minari_demos(n_demos=args.n_demos, H=args.H)
    else:
        demos = generate_pointmaze_demos(
            n_demos=args.n_demos, H=args.H, seed=args.seed)

    # Stats
    sr = sum(d['success'] for d in demos) / len(demos)
    lens = [len(d['obs']) for d in demos]
    print(f"\n  Summary:")
    print(f"    Demos: {len(demos)}")
    print(f"    SR: {sr:.1%}")
    print(f"    Trajectory length: {np.mean(lens):.0f} mean, "
          f"{np.min(lens)}-{np.max(lens)} range")
    print(f"    obs shape: {demos[0]['obs'].shape}")
    print(f"    action shape: {demos[0]['actions'].shape}")
    print(f"    goal shape: {demos[0]['goal'].shape}")
    print("=" * 60)
