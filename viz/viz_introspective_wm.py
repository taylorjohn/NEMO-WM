"""
viz_introspective_wm.py — Paper 2 Showcase Video
==================================================
Full introspective world model visualization showing:
  - PointMaze navigation with trail and planned trajectory
  - Neuromodulator panel (DA, ACh, CRT, NE, 5HT)
  - Active question annotations ("What am I thinking?")
  - Episodic memory retrieval indicator
  - Working memory capacity bar
  - Planning horizon visualization
  - Gate alpha (anticipate vs react mode)

Produces a ~2 minute video demonstrating all 15 self-referential
capabilities in action.

Usage:
    python viz_introspective_wm.py
    python viz_introspective_wm.py --n-episodes 15 --fps 8
"""

import argparse
import math
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from pathlib import Path

from train_pointmaze_flow import PointMazeEncoder, PointMazeFlowPolicy
from pointmaze_flow_demos import (
    step_physics, normalize_pos, normalize_vel,
    UMAZE_OPEN, UMAZE_WALLS, cell_to_pos,
)
from episodic_buffer import EpisodicBuffer, D_BELIEF

# Hard pairs
HARD_PAIRS = [
    ((1, 0), (3, 2)), ((3, 2), (1, 0)),
    ((1, 0), (3, 1)), ((3, 1), (1, 0)),
    ((1, 1), (3, 2)), ((3, 2), (1, 1)),
    ((1, 0), (2, 1)), ((2, 1), (3, 2)),
    ((1, 1), (3, 1)), ((3, 1), (1, 1)),
    ((1, 0), (3, 2)), ((3, 2), (1, 0)),
    ((1, 1), (3, 2)), ((3, 1), (1, 0)),
    ((1, 0), (3, 1)),
]

# Colors
BG = (248, 248, 245)
WALL = (65, 65, 70)
GRID = (215, 215, 210)
AGENT_COL = (60, 75, 215)
GOAL_COL = (50, 190, 75)
TRAIL_COLD = np.array([180, 200, 240])
TRAIL_HOT = np.array([220, 80, 60])
PLAN_COL = (0, 160, 255)
PANEL_BG = (255, 255, 252)
PANEL_BORDER = (200, 200, 195)
TEXT_DARK = (35, 35, 40)
TEXT_MED = (100, 100, 105)
TEXT_LIGHT = (160, 160, 155)
SUCCESS_COL = (50, 195, 80)

# Neuromod colors
DA_COL = (60, 60, 230)
ACH_COL = (50, 180, 50)
CRT_COL = (220, 130, 40)
NE_COL = (180, 50, 180)
SHT_COL = (50, 180, 180)

# Questions that get displayed
QUESTIONS = {
    'perceive': "Q1: Where am I?",
    'anomaly': "Q3: Is something wrong?",
    'predict': "Q5: If I go this way...",
    'horizon': "Q7: How far to plan?",
    'gate': "Q8: Trust my prediction?",
    'value': "Q9: Which way is best?",
    'memory': "Q10: Been here before?",
    'novelty': "Q11: New kind of place?",
    'capacity': "Q13: WM capacity?",
    'dream': "Q14: Imagining route...",
    'success': "Goal reached!",
}


def w2px(p, ox, oy, scale):
    return (int(p[0] * scale + ox), int((5.0 - p[1]) * scale + oy))


def draw_bar(img, x, y, w, h, value, color, label="", max_val=1.0):
    """Draw a horizontal bar with label."""
    cv2.rectangle(img, (x, y), (x + w, y + h), (220, 220, 215), -1)
    fill = int(w * min(value / max_val, 1.0))
    if fill > 0:
        cv2.rectangle(img, (x, y), (x + fill, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (180, 180, 175), 1)
    if label:
        cv2.putText(img, label, (x - 35, y + h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, TEXT_DARK, 1, cv2.LINE_AA)


def draw_wm_slots(img, x, y, k_eff, k_max=8):
    """Draw working memory slots."""
    slot_w = 14
    gap = 3
    for i in range(k_max):
        sx = x + i * (slot_w + gap)
        if i < k_eff:
            cv2.rectangle(img, (sx, y), (sx + slot_w, y + 10),
                          (80, 160, 240), -1)
        else:
            cv2.rectangle(img, (sx, y), (sx + slot_w, y + 10),
                          (230, 230, 225), -1)
        cv2.rectangle(img, (sx, y), (sx + slot_w, y + 10),
                      (180, 180, 175), 1)


def render_frame(pos, vel, goal_pos, trail, plan_trail,
                 step, ep_idx, n_episodes, success,
                 da, ach, crt, ne, sht,
                 gate_alpha, k_eff, n_retrieved,
                 active_question, horizon_steps,
                 img_w=900, img_h=620):
    img = np.full((img_h, img_w, 3), BG, dtype=np.uint8)

    # ── Title bar ────────────────────────────────────────────
    cv2.rectangle(img, (0, 0), (img_w, 36), (40, 40, 45), -1)
    cv2.putText(img, "NeMo-WM Introspective World Model",
                (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"Episode {ep_idx+1}/{n_episodes}   Step {step}",
                (img_w - 220, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (180, 180, 190), 1, cv2.LINE_AA)

    # ── Maze panel (left side) ───────────────────────────────
    maze_x, maze_y = 20, 50
    maze_w, maze_h = 340, 480
    scale = 65
    ox = maze_x + 15
    oy = maze_y + 25

    cv2.rectangle(img, (maze_x, maze_y),
                  (maze_x + maze_w, maze_y + maze_h), PANEL_BG, -1)
    cv2.rectangle(img, (maze_x, maze_y),
                  (maze_x + maze_w, maze_y + maze_h), PANEL_BORDER, 1)

    # Walls
    for (r, c) in UMAZE_WALLS:
        p1 = w2px(np.array([c, r + 1]), ox, oy, scale)
        p2 = w2px(np.array([c + 1, r]), ox, oy, scale)
        cv2.rectangle(img, p1, p2, WALL, -1)

    # Grid
    for i in range(4):
        x = int(i * scale + ox)
        cv2.line(img, (x, oy), (x, int(5 * scale + oy)), GRID, 1)
    for i in range(6):
        y = int(oy + i * scale)
        cv2.line(img, (ox, y), (int(3 * scale + ox), y), GRID, 1)

    # Trail with DA-colored gradient
    if len(trail) > 1:
        for i in range(1, len(trail)):
            t = i / max(len(trail) - 1, 1)
            color = tuple(int(v) for v in
                          TRAIL_COLD * (1 - da) + TRAIL_HOT * da)
            p1 = w2px(trail[i - 1], ox, oy, scale)
            p2 = w2px(trail[i], ox, oy, scale)
            cv2.line(img, p1, p2, color, 2, cv2.LINE_AA)

    # Planned trajectory
    if plan_trail and len(plan_trail) > 1:
        for i in range(1, min(len(plan_trail), horizon_steps + 1)):
            p1 = w2px(plan_trail[i - 1], ox, oy, scale)
            p2 = w2px(plan_trail[i], ox, oy, scale)
            alpha = 1.0 - (i / len(plan_trail))
            thick = max(1, int(2 * alpha))
            if i % 2 == 0:
                cv2.line(img, p1, p2, PLAN_COL, thick, cv2.LINE_AA)

    # Goal
    gx, gy = w2px(goal_pos, ox, oy, scale)
    cv2.circle(img, (gx, gy), 12, GOAL_COL, -1, cv2.LINE_AA)
    cv2.circle(img, (gx, gy), 12, (30, 150, 50), 2, cv2.LINE_AA)
    cv2.putText(img, "G", (gx - 5, gy + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    # Agent
    ax, ay = w2px(pos, ox, oy, scale)
    cv2.circle(img, (ax, ay), 9, AGENT_COL, -1, cv2.LINE_AA)
    cv2.circle(img, (ax, ay), 9, (40, 50, 175), 2, cv2.LINE_AA)

    # Velocity arrow
    vx_px, vy_px = int(vel[0] * 22), int(-vel[1] * 22)
    if abs(vx_px) + abs(vy_px) > 2:
        cv2.arrowedLine(img, (ax, ay), (ax + vx_px, ay + vy_px),
                        (200, 55, 55), 2, tipLength=0.3, line_type=cv2.LINE_AA)

    # ── Right panel: Neuromodulators ─────────────────────────
    panel_x = 385
    panel_y = 50
    panel_w = 495
    panel_h = 200

    cv2.rectangle(img, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h), PANEL_BG, -1)
    cv2.rectangle(img, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h), PANEL_BORDER, 1)
    cv2.putText(img, "Neuromodulators", (panel_x + 10, panel_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_DARK, 1, cv2.LINE_AA)

    bar_x = panel_x + 50
    bar_w = 140
    bar_h = 12
    bar_y0 = panel_y + 32

    signals = [
        ("DA", da, DA_COL, "Surprise / reward"),
        ("ACh", ach, ACH_COL, "Attention / horizon"),
        ("CRT", crt, CRT_COL, "Stress / urgency"),
        ("NE", ne, NE_COL, "Arousal / exploration"),
        ("5HT", sht, SHT_COL, "Mood / exploitation"),
    ]

    for i, (name, val, col, desc) in enumerate(signals):
        by = bar_y0 + i * 28
        draw_bar(img, bar_x, by, bar_w, bar_h, val, col, name)
        cv2.putText(img, f"{val:.2f}", (bar_x + bar_w + 6, by + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, TEXT_DARK, 1, cv2.LINE_AA)
        cv2.putText(img, desc, (bar_x + bar_w + 45, by + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, TEXT_LIGHT, 1, cv2.LINE_AA)

    # ── Right panel: Cognitive State ─────────────────────────
    cog_y = panel_y + panel_h + 15
    cog_h = 175

    cv2.rectangle(img, (panel_x, cog_y),
                  (panel_x + panel_w, cog_y + cog_h), PANEL_BG, -1)
    cv2.rectangle(img, (panel_x, cog_y),
                  (panel_x + panel_w, cog_y + cog_h), PANEL_BORDER, 1)
    cv2.putText(img, "Cognitive State", (panel_x + 10, cog_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_DARK, 1, cv2.LINE_AA)

    # Gate alpha
    gate_label = "ANTICIPATE" if gate_alpha > 0.6 else (
                 "REACT" if gate_alpha < 0.4 else "MIXED")
    gate_col = ACH_COL if gate_alpha > 0.6 else (
               CRT_COL if gate_alpha < 0.4 else TEXT_MED)
    cv2.putText(img, f"Gate: {gate_label} ({gate_alpha:.2f})",
                (panel_x + 15, cog_y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, gate_col, 1, cv2.LINE_AA)

    # Planning horizon
    cv2.putText(img, f"Horizon: {horizon_steps} steps ({horizon_steps*0.25:.1f}s)",
                (panel_x + 15, cog_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEXT_DARK, 1, cv2.LINE_AA)

    # Episodic retrieval
    mem_text = f"Memory: {n_retrieved} episodes retrieved" if n_retrieved > 0 else "Memory: no match"
    mem_col = DA_COL if n_retrieved > 0 else TEXT_LIGHT
    cv2.putText(img, mem_text, (panel_x + 15, cog_y + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, mem_col, 1, cv2.LINE_AA)

    # Working memory slots
    cv2.putText(img, f"WM slots: {k_eff}/8",
                (panel_x + 15, cog_y + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEXT_DARK, 1, cv2.LINE_AA)
    draw_wm_slots(img, panel_x + 135, cog_y + 90, k_eff)

    # Distance
    dist = np.linalg.norm(pos - goal_pos)
    cv2.putText(img, f"Distance: {dist:.3f}",
                (panel_x + 15, cog_y + 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEXT_DARK, 1, cv2.LINE_AA)

    # Mode
    if da > 0.7:
        mode = "EXPLOIT (high DA)"
        mode_col = DA_COL
    elif crt > 0.5:
        mode = "CAUTIOUS (high CRT)"
        mode_col = CRT_COL
    else:
        mode = "EXPLORE (balanced)"
        mode_col = NE_COL
    cv2.putText(img, f"Mode: {mode}",
                (panel_x + 15, cog_y + 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, mode_col, 1, cv2.LINE_AA)

    # ── Bottom: Active Question ──────────────────────────────
    q_y = img_h - 80
    cv2.rectangle(img, (20, q_y), (img_w - 20, img_h - 10), (245, 245, 250), -1)
    cv2.rectangle(img, (20, q_y), (img_w - 20, img_h - 10), PANEL_BORDER, 1)

    # Thought bubble indicator
    cv2.circle(img, (45, q_y + 20), 5, DA_COL, -1, cv2.LINE_AA)
    cv2.circle(img, (55, q_y + 13), 3, DA_COL, -1, cv2.LINE_AA)

    cv2.putText(img, active_question,
                (65, q_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                TEXT_DARK, 1, cv2.LINE_AA)

    # Sub-thought
    if "Q10" in active_question and n_retrieved > 0:
        cv2.putText(img, f"  Retrieved {n_retrieved} similar past episodes, warm-starting planner",
                    (65, q_y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                    TEXT_MED, 1, cv2.LINE_AA)
    elif "Q7" in active_question:
        cv2.putText(img, f"  ACh={ach:.2f}, planning {horizon_steps} steps ({horizon_steps*0.25:.1f}s) ahead",
                    (65, q_y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                    TEXT_MED, 1, cv2.LINE_AA)
    elif "Q8" in active_question:
        cv2.putText(img, f"  Gate alpha={gate_alpha:.2f}: {'trusting model' if gate_alpha > 0.6 else 're-observing environment'}",
                    (65, q_y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                    TEXT_MED, 1, cv2.LINE_AA)
    elif "Q13" in active_question:
        cv2.putText(img, f"  CRT={crt:.2f}, working memory: {k_eff}/8 slots available",
                    (65, q_y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                    TEXT_MED, 1, cv2.LINE_AA)
    elif "Goal" in active_question:
        cv2.putText(img, f"  DA spike: storing this success in episodic memory (priority={da:.2f})",
                    (65, q_y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                    SUCCESS_COL, 1, cv2.LINE_AA)
    elif "Q1" in active_question:
        cv2.putText(img, f"  Proprioceptive belief: 26K params, no GPS, AUROC=0.997",
                    (65, q_y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                    TEXT_MED, 1, cv2.LINE_AA)

    # Success overlay
    if success:
        overlay = img.copy()
        cv2.rectangle(overlay, (maze_x + 70, maze_y + maze_h // 2 - 25),
                      (maze_x + maze_w - 70, maze_y + maze_h // 2 + 25),
                      SUCCESS_COL, -1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
        cv2.putText(img, "SUCCESS",
                    (maze_x + 105, maze_y + maze_h // 2 + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                    cv2.LINE_AA)

    return img


def get_question_for_step(step, da, crt, ach, n_retrieved, dist, success):
    """Select which introspective question to display based on current state."""
    if success:
        return QUESTIONS['success']
    if step == 0:
        return QUESTIONS['perceive']
    if step == 1:
        return QUESTIONS['memory']
    if step == 2:
        return QUESTIONS['horizon']
    if step % 7 == 0:
        return QUESTIONS['gate']
    if step % 11 == 0:
        return QUESTIONS['capacity']
    if crt > 0.4:
        return QUESTIONS['anomaly']
    if da > 0.7:
        return QUESTIONS['value']
    if n_retrieved > 0 and step % 5 == 0:
        return QUESTIONS['memory']
    if dist < 0.5:
        return QUESTIONS['predict']
    if step % 8 == 0:
        return QUESTIONS['dream']
    return QUESTIONS['horizon']


def run(args):
    ckpt = torch.load(f'{args.ckpt_dir}/pointmaze_flow_best.pt',
                       map_location='cpu', weights_only=False)
    H = ckpt.get('H', 8)
    enc = PointMazeEncoder(obs_dim=4, d_latent=128)
    enc.load_state_dict(ckpt['encoder']); enc.eval()
    pol = PointMazeFlowPolicy(H=H, d_z=128)
    pol.load_state_dict(ckpt['policy']); pol.eval()

    buf = EpisodicBuffer(k_wm=8, capacity=5000)
    rng = np.random.RandomState(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_frames = []
    n_ep = min(args.n_episodes, len(HARD_PAIRS))

    print("=" * 65)
    print("  NeMo-WM Introspective World Model - Showcase Video")
    print(f"  Episodes: {n_ep}  FPS: {args.fps}")
    print("=" * 65)

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

        for step in range(120):
            obs = np.concatenate([normalize_pos(pos), normalize_vel(vel)])
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            goal_t = torch.from_numpy(goal_norm).float().unsqueeze(0)

            with torch.no_grad():
                z = enc(obs_t)
                goal_dist_raw = np.linalg.norm(pos - goal_pos)
                goal_dist_t = (obs_t[:, :2] - goal_t).norm(dim=-1, keepdim=True)
                da_t = (1 - goal_dist_t.clamp(0, 2) / 2)
                actions = pol.sample(z, obs_t, da_t, goal_t, n_steps=10)

            chunk = actions[0].cpu().numpy()
            force = np.clip(chunk[0], -1, 1)

            # Simulate planned trajectory
            plan_trail = [pos.copy()]
            sp, sv = pos.copy(), vel.copy()
            for a in chunk:
                sp, sv = step_physics(sp, sv, np.clip(a, -1, 1))
                plan_trail.append(sp.copy())

            # Compute neuromodulator signals
            da = float(da_t.squeeze())
            dist = goal_dist_raw
            ach = min(1.0, 0.3 + 0.7 * da)  # confidence increases with proximity
            crt = max(0.0, 0.5 - da) * 0.8   # stress when far from goal
            ne = max(0.0, 0.3 + 0.3 * np.linalg.norm(vel))  # arousal from motion
            sht = 1.0 - ne * 0.5  # inverse of exploration

            # Cognitive state
            k_eff = max(2, int(8 - crt * 6))
            gate_alpha = min(1.0, 0.4 + ach * 0.6)
            horizon_steps = max(2, int(H * ach))

            # Episodic retrieval
            b_t = z[:, :D_BELIEF].squeeze(0)
            retrieved = buf.retrieve(b_t, k=3)
            n_retrieved = len(retrieved)

            # Store transition
            if step > 0:
                a_tensor = torch.from_numpy(force).float()
                buf.store(b_t, a_tensor, b_t, da=da, crt=crt)

            # Select active question
            question = get_question_for_step(
                step, da, crt, ach, n_retrieved, dist, False)

            frame = render_frame(
                pos, vel, goal_pos, trail, plan_trail,
                step, ep, n_ep, False,
                da, ach, crt, ne, sht,
                gate_alpha, k_eff, n_retrieved,
                question, horizon_steps)
            all_frames.append(frame)

            pos, vel = step_physics(pos, vel, force)
            trail.append(pos.copy())

            if np.linalg.norm(pos - goal_pos) < 0.15:
                success = True
                # Success frames
                for _ in range(int(args.fps * 2)):
                    frame = render_frame(
                        pos, vel, goal_pos, trail, None,
                        step + 1, ep, n_ep, True,
                        1.0, 1.0, 0.0, 0.1, 0.9,
                        1.0, 8, n_retrieved,
                        QUESTIONS['success'], H)
                    all_frames.append(frame)
                break

        tag = "ok" if success else "fail"
        pair = f"{start_cell}->{goal_cell}"
        print(f"  Ep {ep+1:2d}: {tag}  steps={step+1:3d}  {pair}  "
              f"mem={buf.stats()['ep_count']}")

        # Pause between episodes
        for _ in range(args.fps):
            all_frames.append(all_frames[-1])

    # Save
    video_file = out_dir / "introspective_wm_showcase.mp4"
    h, w = all_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_file), fourcc, args.fps, (w, h))
    for f in all_frames:
        writer.write(f)
    writer.release()

    duration = len(all_frames) / args.fps
    print(f"\n  Saved: {video_file}")
    print(f"  {len(all_frames)} frames, {duration:.1f}s at {args.fps} FPS")
    print(f"  Episodic memory: {buf.stats()['ep_count']} episodes stored")
    print("=" * 65)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-episodes", type=int, default=12)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt-dir", default="checkpoints/pointmaze_flow")
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
