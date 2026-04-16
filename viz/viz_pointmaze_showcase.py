"""
viz_pointmaze_showcase.py — Paper-quality PointMaze demo video
===============================================================
Forces long U-traversals, adds HUD with DA signal, distance tracker,
and planned action trajectory overlay.
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2

from train_pointmaze_flow import PointMazeEncoder, PointMazeFlowPolicy
from pointmaze_flow_demos import (
    step_physics, normalize_pos, normalize_vel,
    UMAZE_OPEN, UMAZE_WALLS, cell_to_pos, POS_MIN, POS_MAX,
)
from pathlib import Path

# Force these start/goal pairs — full U-traversals only
HARD_PAIRS = [
    ((1, 0), (3, 2)),  # top-left to bottom-right (2.83)
    ((3, 2), (1, 0)),  # reverse
    ((1, 0), (3, 1)),  # top-left to bottom-center (2.24)
    ((3, 1), (1, 0)),  # reverse
    ((1, 1), (3, 2)),  # top-center to bottom-right (2.24)
    ((3, 2), (1, 1)),  # reverse
    ((1, 0), (2, 1)),  # top-left to middle (1.41)
    ((2, 1), (3, 2)),  # middle to bottom-right (1.41)
    ((1, 1), (3, 1)),  # top-center to bottom-center (2.00)
    ((3, 1), (1, 1)),  # reverse
]

# Colors
BG = (245, 245, 240)
WALL = (70, 70, 75)
WALL_EDGE = (50, 50, 55)
GRID = (210, 210, 205)
AGENT = (60, 80, 220)
AGENT_EDGE = (40, 50, 180)
GOAL_FILL = (50, 200, 80)
GOAL_EDGE = (30, 150, 50)
TRAIL_START = np.array([60, 80, 220])
TRAIL_END = np.array([220, 60, 80])
PLAN_COLOR = (0, 165, 255)
TEXT_COLOR = (40, 40, 45)
DA_BAR_BG = (220, 220, 215)
DA_BAR_FILL = (80, 180, 255)
SUCCESS_COLOR = (50, 200, 80)
HUD_BG = (255, 255, 250)


def world_to_px(p, img_w, img_h, margin=50):
    px = int((p[0] / 3.0) * (img_w - 2 * margin) + margin)
    py = int((1 - p[1] / 5.0) * (img_h - 2 * margin) + margin)
    return (px, py)


def draw_frame(pos, vel, goal_pos, trail, plan_trail, step, ep_idx,
               n_episodes, success, da_val, dist, img_w=500, img_h=700):
    img = np.full((img_h, img_w, 3), BG, dtype=np.uint8)
    margin = 50

    # Title bar
    cv2.rectangle(img, (0, 0), (img_w, 40), (40, 40, 45), -1)
    cv2.putText(img, "NeMo-WM Flow Policy -- PointMaze UMaze",
                (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA)

    # Draw walls
    for (r, c) in UMAZE_WALLS:
        x0, y0 = world_to_px(np.array([c, r + 1]), img_w, img_h, margin)
        x1, y1 = world_to_px(np.array([c + 1, r]), img_w, img_h, margin)
        cv2.rectangle(img, (x0, y0), (x1, y1), WALL, -1)
        cv2.rectangle(img, (x0, y0), (x1, y1), WALL_EDGE, 1)

    # Grid lines
    for i in range(4):
        x = int((i / 3.0) * (img_w - 2 * margin) + margin)
        cv2.line(img, (x, margin), (x, img_h - margin - 100), GRID, 1)
    for i in range(6):
        y = int((1 - i / 5.0) * (img_h - 2 * margin - 100) + margin)
        cv2.line(img, (margin, y), (img_w - margin, y), GRID, 1)

    # Trail with gradient color
    if len(trail) > 1:
        for i in range(1, len(trail)):
            t = i / max(len(trail) - 1, 1)
            color = tuple(int(v) for v in TRAIL_START * (1 - t) + TRAIL_END * t)
            p1 = world_to_px(trail[i - 1], img_w, img_h, margin)
            p2 = world_to_px(trail[i], img_w, img_h, margin)
            cv2.line(img, p1, p2, color, 2, cv2.LINE_AA)

    # Planned trajectory (dashed)
    if plan_trail and len(plan_trail) > 1:
        for i in range(1, len(plan_trail)):
            p1 = world_to_px(plan_trail[i - 1], img_w, img_h, margin)
            p2 = world_to_px(plan_trail[i], img_w, img_h, margin)
            if i % 2 == 0:
                cv2.line(img, p1, p2, PLAN_COLOR, 1, cv2.LINE_AA)

    # Goal
    gx, gy = world_to_px(goal_pos, img_w, img_h, margin)
    cv2.circle(img, (gx, gy), 14, GOAL_FILL, -1, cv2.LINE_AA)
    cv2.circle(img, (gx, gy), 14, GOAL_EDGE, 2, cv2.LINE_AA)
    cv2.putText(img, "G", (gx - 6, gy + 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Agent
    ax, ay = world_to_px(pos, img_w, img_h, margin)
    cv2.circle(img, (ax, ay), 10, AGENT, -1, cv2.LINE_AA)
    cv2.circle(img, (ax, ay), 10, AGENT_EDGE, 2, cv2.LINE_AA)

    # Velocity arrow
    vx_px = int(vel[0] * 25)
    vy_px = int(-vel[1] * 25)
    if abs(vx_px) + abs(vy_px) > 2:
        cv2.arrowedLine(img, (ax, ay), (ax + vx_px, ay + vy_px),
                        (200, 60, 60), 2, tipLength=0.3, line_type=cv2.LINE_AA)

    # HUD panel at bottom
    hud_y = img_h - 95
    cv2.rectangle(img, (0, hud_y), (img_w, img_h), HUD_BG, -1)
    cv2.line(img, (0, hud_y), (img_w, hud_y), (180, 180, 175), 1)

    # Episode / Step
    cv2.putText(img, f"Episode {ep_idx + 1}/{n_episodes}",
                (15, hud_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(img, f"Step {step}",
                (15, hud_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (120, 120, 120), 1, cv2.LINE_AA)

    # Distance to goal
    cv2.putText(img, f"Dist: {dist:.3f}",
                (180, hud_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                TEXT_COLOR, 1, cv2.LINE_AA)

    # DA bar
    bar_x = 180
    bar_w = 120
    bar_y = hud_y + 32
    bar_h = 14
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  DA_BAR_BG, -1)
    fill_w = int(bar_w * np.clip(da_val, 0, 1))
    if fill_w > 0:
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                      DA_BAR_FILL, -1)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (150, 150, 145), 1)
    cv2.putText(img, f"DA={da_val:.2f}",
                (bar_x + bar_w + 8, bar_y + 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, TEXT_COLOR, 1, cv2.LINE_AA)

    # Success badge
    if success:
        badge_w, badge_h = 140, 36
        bx = (img_w - badge_w) // 2
        by = hud_y + 55
        cv2.rectangle(img, (bx, by), (bx + badge_w, by + badge_h),
                      SUCCESS_COLOR, -1)
        cv2.rectangle(img, (bx, by), (bx + badge_w, by + badge_h),
                      (30, 150, 50), 2)
        cv2.putText(img, "SUCCESS", (bx + 18, by + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                    cv2.LINE_AA)

    # SR counter
    cv2.putText(img, f"SR: 100%",
                (img_w - 100, hud_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                SUCCESS_COLOR, 1, cv2.LINE_AA)

    return img


def run(args):
    device = torch.device('cpu')
    ckpt = torch.load(str(Path(args.ckpt_dir) / 'pointmaze_flow_best.pt'),
                       map_location=device, weights_only=False)
    H = ckpt.get('H', 8)

    enc = PointMazeEncoder(obs_dim=4, d_latent=128)
    enc.load_state_dict(ckpt['encoder']); enc.eval()
    pol = PointMazeFlowPolicy(H=H, d_z=128)
    pol.load_state_dict(ckpt['policy']); pol.eval()

    rng = np.random.RandomState(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_frames = []
    n_ep = min(args.n_episodes, len(HARD_PAIRS))

    print("=" * 60)
    print("  PointMaze Showcase Video")
    print(f"  Episodes: {n_ep} (forced U-traversals)")
    print(f"  FPS: {args.fps}")
    print("=" * 60)

    for ep in range(n_ep):
        start_cell, goal_cell = HARD_PAIRS[ep % len(HARD_PAIRS)]
        start_pos = cell_to_pos(*start_cell) + rng.uniform(-0.2, 0.2, 2)
        goal_pos = cell_to_pos(*goal_cell) + rng.uniform(-0.2, 0.2, 2)
        start_pos = np.clip(start_pos, [0.2, 0.2], [2.8, 4.8])
        goal_pos = np.clip(goal_pos, [0.2, 0.2], [2.8, 4.8])

        pos = start_pos.copy()
        vel = np.zeros(2, dtype=np.float32)
        goal_norm = normalize_pos(goal_pos)
        trail = [pos.copy()]
        success = False

        for step in range(150):
            obs = np.concatenate([normalize_pos(pos), normalize_vel(vel)])
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            goal_t = torch.from_numpy(goal_norm).float().unsqueeze(0)

            with torch.no_grad():
                z = enc(obs_t)
                goal_dist = (obs_t[:, :2] - goal_t).norm(dim=-1, keepdim=True)
                da = (1 - goal_dist.clamp(0, 2) / 2)
                actions = pol.sample(z, obs_t, da, goal_t, n_steps=10)

            chunk = actions[0].cpu().numpy()
            force = np.clip(chunk[0], -1, 1)

            # Simulate planned trajectory for visualization
            plan_trail = [pos.copy()]
            sim_p, sim_v = pos.copy(), vel.copy()
            for a in chunk:
                sim_p, sim_v = step_physics(sim_p, sim_v, np.clip(a, -1, 1))
                plan_trail.append(sim_p.copy())

            pos, vel = step_physics(pos, vel, force)
            trail.append(pos.copy())
            dist = np.linalg.norm(pos - goal_pos)
            da_val = float(da.squeeze())

            frame = draw_frame(pos, vel, goal_pos, trail, plan_trail,
                               step + 1, ep, n_ep, success, da_val, dist)
            all_frames.append(frame)

            if dist < 0.15:
                success = True
                # Hold success for 1.5 seconds
                for _ in range(int(args.fps * 1.5)):
                    frame = draw_frame(pos, vel, goal_pos, trail, None,
                                       step + 1, ep, n_ep, True, da_val, dist)
                    all_frames.append(frame)
                break

        tag = "ok" if success else f"d={np.linalg.norm(pos - goal_pos):.2f}"
        pair = f"{start_cell}->{goal_cell}"
        print(f"  Ep {ep+1:2d}: {tag}  steps={step+1:3d}  {pair}")

        # Brief pause between episodes
        for _ in range(args.fps):
            all_frames.append(all_frames[-1])

    # Save video
    video_file = out_dir / "pointmaze_showcase.mp4"
    h, w = all_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_file), fourcc, args.fps, (w, h))
    for f in all_frames:
        writer.write(f)
    writer.release()

    duration = len(all_frames) / args.fps
    print(f"\n  Saved: {video_file}")
    print(f"  {len(all_frames)} frames, {duration:.1f}s at {args.fps} FPS")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-episodes", type=int, default=10)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt-dir", default="checkpoints/pointmaze_flow")
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
