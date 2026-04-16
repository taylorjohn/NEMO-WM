"""
viz_language_demos.py — Video demos of the language layer
==========================================================
Three demo videos showing self-narration in real-time:

1. PointMaze with live thought narration (Q1-Q17)
2. Physics discovery with verbal explanation
3. Combined: introspective maze + discovery journal

Usage:
    python viz_language_demos.py --demo maze
    python viz_language_demos.py --demo physics
    python viz_language_demos.py --demo all
"""

import argparse
import math
import numpy as np
import torch
import cv2
from pathlib import Path

from language_layer import (
    SelfNarrator, NarrationSignals, PhysicsExplainer,
    AnomalyExplainer, DiscoveryJournal,
)
from train_pointmaze_flow import PointMazeEncoder, PointMazeFlowPolicy
from pointmaze_flow_demos import (
    step_physics, normalize_pos, normalize_vel,
    UMAZE_OPEN, UMAZE_WALLS, cell_to_pos,
)
from physics_discovery_agent import (
    SimplePhysicsSim, PhysicsState, DiscoveryAgent, ForceType,
)

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# Colors
BG = (248, 248, 245)
WALL = (65, 65, 70)
AGENT_COL = (60, 75, 215)
GOAL_COL = (50, 190, 75)
PANEL_BG = (255, 255, 252)
PANEL_BORDER = (200, 200, 195)
TEXT_DARK = (35, 35, 40)
TEXT_MED = (100, 100, 105)
TEXT_LIGHT = (150, 150, 145)
THOUGHT_BG = (240, 242, 255)
THOUGHT_BORDER = (180, 185, 220)
DA_COL = (60, 60, 230)
ACH_COL = (50, 180, 50)
CRT_COL = (220, 130, 40)
SUCCESS_COL = (50, 195, 80)
DISCOVERY_BG = (240, 255, 240)
DISCOVERY_BORDER = (100, 200, 100)
JOURNAL_BG = (255, 252, 240)


def w2px(p, ox, oy, scale):
    return (int(p[0] * scale + ox), int((5.0 - p[1]) * scale + oy))


def draw_text_wrapped(img, text, x, y, max_width, font_scale=0.38,
                       color=TEXT_DARK, line_height=18):
    """Draw text with word wrapping."""
    words = text.split()
    lines = []
    current = ""
    for word in words:
        test = current + " " + word if current else word
        (tw, _), _ = cv2.getTextSize(test, cv2.FONT_HERSHEY_SIMPLEX,
                                      font_scale, 1)
        if tw > max_width and current:
            lines.append(current)
            current = word
        else:
            current = test
    if current:
        lines.append(current)

    for i, line in enumerate(lines):
        cv2.putText(img, line, (x, y + i * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1,
                    cv2.LINE_AA)
    return len(lines) * line_height


def draw_bar_small(img, x, y, w, h, value, color, label=""):
    cv2.rectangle(img, (x, y), (x + w, y + h), (220, 220, 215), -1)
    fill = int(w * min(value, 1.0))
    if fill > 0:
        cv2.rectangle(img, (x, y), (x + fill, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (180, 180, 175), 1)
    if label:
        cv2.putText(img, label, (x - 30, y + h - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, TEXT_DARK, 1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
# Demo 1: PointMaze with live narration
# ──────────────────────────────────────────────────────────────────────────────

def demo_maze(args):
    print("\n  Generating: PointMaze + Live Narration")

    ckpt = torch.load('checkpoints/pointmaze_flow/pointmaze_flow_best.pt',
                       map_location='cpu', weights_only=False)
    H = ckpt.get('H', 8)
    enc = PointMazeEncoder(obs_dim=4, d_latent=128)
    enc.load_state_dict(ckpt['encoder']); enc.eval()
    pol = PointMazeFlowPolicy(H=H, d_z=128)
    pol.load_state_dict(ckpt['policy']); pol.eval()

    narrator = SelfNarrator()
    rng = np.random.RandomState(args.seed)

    HARD_PAIRS = [
        ((1, 0), (3, 2)), ((3, 2), (1, 0)),
        ((1, 0), (3, 1)), ((3, 1), (1, 0)),
        ((1, 1), (3, 2)), ((3, 2), (1, 1)),
        ((1, 0), (2, 1)), ((2, 1), (3, 2)),
    ]

    all_frames = []
    n_ep = min(args.n_episodes, len(HARD_PAIRS))
    img_w, img_h = 900, 580

    for ep in range(n_ep):
        start_cell, goal_cell = HARD_PAIRS[ep]
        start_pos = cell_to_pos(*start_cell) + rng.uniform(-0.2, 0.2, 2)
        goal_pos = cell_to_pos(*goal_cell) + rng.uniform(-0.2, 0.2, 2)
        start_pos = np.clip(start_pos, [0.2, 0.2], [2.8, 4.8])
        goal_pos = np.clip(goal_pos, [0.2, 0.2], [2.8, 4.8])

        pos = start_pos.copy()
        vel = np.zeros(2, dtype=np.float32)
        goal_norm = normalize_pos(goal_pos)
        trail = [pos.copy()]
        success = False

        for step in range(100):
            obs = np.concatenate([normalize_pos(pos), normalize_vel(vel)])
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            goal_t = torch.from_numpy(goal_norm).float().unsqueeze(0)

            with torch.no_grad():
                z = enc(obs_t)
                goal_dist = float((obs_t[:, :2] - goal_t).norm())
                da_t = 1 - min(goal_dist, 2) / 2
                actions = pol.sample(z, obs_t,
                                     torch.tensor([[da_t]]),
                                     goal_t, n_steps=10)

            chunk = actions[0].cpu().numpy()
            force = np.clip(chunk[0], -1, 1)

            # Compute signals
            dist = np.linalg.norm(pos - goal_pos)
            da = da_t
            ach = min(1.0, 0.3 + 0.7 * da)
            crt = max(0.0, 0.5 - da) * 0.8
            ne = min(1.0, 0.3 + 0.3 * np.linalg.norm(vel))
            heading = math.atan2(vel[1], vel[0]) if np.linalg.norm(vel) > 0.01 else 0
            gate_alpha = min(1.0, 0.4 + ach * 0.6)
            k_eff = max(2, int(8 - crt * 6))
            horizon = max(2, int(H * ach))
            explore_score = max(0, min(1, (1 - da) * 1.5))

            # Generate narration
            signals = NarrationSignals(
                pos=pos, vel=vel, heading=heading,
                speed=float(np.linalg.norm(vel)),
                da=da, ach=ach, crt=crt, ne=ne, sht=1-ne*0.5,
                gate_alpha=gate_alpha, k_eff=k_eff,
                horizon_steps=horizon, n_retrieved=0,
                dist_to_goal=dist, explore_score=explore_score,
                novelty=max(0.2, 2.0 - ep * 0.2),
                ep_count=ep * 25, step=step, episode=ep,
            )
            narration = narrator.narrate(signals)

            # ── Draw frame ──
            img = np.full((img_h, img_w, 3), BG, dtype=np.uint8)

            # Title
            cv2.rectangle(img, (0, 0), (img_w, 32), (40, 40, 45), -1)
            cv2.putText(img, "NeMo-WM Self-Narrating World Model",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, f"Ep {ep+1}/{n_ep}  Step {step}",
                        (img_w - 160, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (180, 180, 190), 1, cv2.LINE_AA)

            # Maze (left)
            mx, my = 15, 42
            scale = 55
            ox, oy = mx + 10, my + 10

            cv2.rectangle(img, (mx, my), (mx + 280, my + 340),
                          PANEL_BG, -1)
            cv2.rectangle(img, (mx, my), (mx + 280, my + 340),
                          PANEL_BORDER, 1)

            for (r, c) in UMAZE_WALLS:
                p1 = w2px(np.array([c, r+1]), ox, oy, scale)
                p2 = w2px(np.array([c+1, r]), ox, oy, scale)
                cv2.rectangle(img, p1, p2, WALL, -1)

            # Trail
            if len(trail) > 1:
                for i in range(1, len(trail)):
                    t = i / max(len(trail) - 1, 1)
                    c = (int(180 - 120*t), int(200 - 120*t), int(240 - 20*t))
                    p1 = w2px(trail[i-1], ox, oy, scale)
                    p2 = w2px(trail[i], ox, oy, scale)
                    cv2.line(img, p1, p2, c, 2, cv2.LINE_AA)

            # Goal + Agent
            gx, gy = w2px(goal_pos, ox, oy, scale)
            cv2.circle(img, (gx, gy), 10, GOAL_COL, -1, cv2.LINE_AA)
            ax, ay = w2px(pos, ox, oy, scale)
            cv2.circle(img, (ax, ay), 8, AGENT_COL, -1, cv2.LINE_AA)

            # Neuromod bars (right of maze)
            bx = 310
            by = 50
            cv2.rectangle(img, (bx, by), (bx + 260, by + 130),
                          PANEL_BG, -1)
            cv2.rectangle(img, (bx, by), (bx + 260, by + 130),
                          PANEL_BORDER, 1)
            cv2.putText(img, "Neuromodulators", (bx + 8, by + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEXT_DARK, 1,
                        cv2.LINE_AA)

            sigs = [("DA", da, DA_COL), ("ACh", ach, ACH_COL),
                    ("CRT", crt, CRT_COL), ("NE", ne, (180, 50, 180)),
                    ("5HT", 1-ne*0.5, (50, 180, 180))]
            for i, (name, val, col) in enumerate(sigs):
                draw_bar_small(img, bx + 40, by + 26 + i * 20,
                               100, 10, val, col, name)
                cv2.putText(img, f"{val:.2f}", (bx + 145, by + 35 + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, TEXT_DARK, 1,
                            cv2.LINE_AA)

            # State info
            cv2.rectangle(img, (bx, by + 140), (bx + 260, by + 240),
                          PANEL_BG, -1)
            cv2.rectangle(img, (bx, by + 140), (bx + 260, by + 240),
                          PANEL_BORDER, 1)
            cv2.putText(img, "Cognitive State", (bx + 8, by + 155),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEXT_DARK, 1,
                        cv2.LINE_AA)

            gate_label = "ANTICIPATE" if gate_alpha > 0.6 else "REACT"
            gate_c = ACH_COL if gate_alpha > 0.6 else CRT_COL
            cv2.putText(img, f"Gate: {gate_label} ({gate_alpha:.2f})",
                        (bx + 12, by + 173), cv2.FONT_HERSHEY_SIMPLEX,
                        0.3, gate_c, 1, cv2.LINE_AA)
            cv2.putText(img, f"Horizon: {horizon} steps",
                        (bx + 12, by + 190), cv2.FONT_HERSHEY_SIMPLEX,
                        0.3, TEXT_DARK, 1, cv2.LINE_AA)
            cv2.putText(img, f"WM: {k_eff}/8 slots",
                        (bx + 12, by + 207), cv2.FONT_HERSHEY_SIMPLEX,
                        0.3, TEXT_DARK, 1, cv2.LINE_AA)
            cv2.putText(img, f"Dist: {dist:.2f}",
                        (bx + 12, by + 224), cv2.FONT_HERSHEY_SIMPLEX,
                        0.3, TEXT_DARK, 1, cv2.LINE_AA)

            # Thought bubble (bottom)
            tb_y = img_h - 170
            cv2.rectangle(img, (15, tb_y), (img_w - 15, img_h - 10),
                          THOUGHT_BG, -1)
            cv2.rectangle(img, (15, tb_y), (img_w - 15, img_h - 10),
                          THOUGHT_BORDER, 1)

            # Thought bubble dots
            cv2.circle(img, (35, tb_y - 5), 4, THOUGHT_BORDER, -1)
            cv2.circle(img, (42, tb_y - 14), 3, THOUGHT_BORDER, -1)

            cv2.putText(img, "Internal Monologue",
                        (50, tb_y + 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (100, 100, 160), 1, cv2.LINE_AA)

            draw_text_wrapped(img, narration, 30, tb_y + 40,
                              max_width=img_w - 60, font_scale=0.38,
                              color=TEXT_DARK, line_height=20)

            # Success overlay
            if success:
                cv2.rectangle(img, (80, 180), (230, 220), SUCCESS_COL, -1)
                cv2.putText(img, "SUCCESS", (100, 207),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2, cv2.LINE_AA)

            all_frames.append(img)

            # Step physics
            pos, vel = step_physics(pos, vel, force)
            trail.append(pos.copy())

            if np.linalg.norm(pos - goal_pos) < 0.15:
                success = True
                for _ in range(int(args.fps * 1.5)):
                    # Update narration for success
                    signals.dist_to_goal = 0.0
                    signals.da = 1.0
                    success_narration = "Goal reached! DA spike, storing this success in episodic memory. Planning horizon maxed, high confidence."
                    simg = img.copy()
                    cv2.rectangle(simg, (80, 180), (230, 220),
                                  SUCCESS_COL, -1)
                    cv2.putText(simg, "SUCCESS", (100, 207),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 255, 255), 2, cv2.LINE_AA)
                    # Update thought bubble
                    cv2.rectangle(simg, (16, tb_y + 1),
                                  (img_w - 16, img_h - 11),
                                  THOUGHT_BG, -1)
                    cv2.putText(simg, "Internal Monologue",
                                (50, tb_y + 18), cv2.FONT_HERSHEY_SIMPLEX,
                                0.38, (100, 100, 160), 1, cv2.LINE_AA)
                    draw_text_wrapped(simg, success_narration,
                                      30, tb_y + 40, img_w - 60,
                                      color=SUCCESS_COL)
                    all_frames.append(simg)
                break

        # Pause between episodes
        for _ in range(args.fps):
            all_frames.append(all_frames[-1])

        print(f"    Ep {ep+1}: {'ok' if success else 'fail'}  "
              f"steps={step+1}  {start_cell}->{goal_cell}")

    # Save
    video_file = OUT / "narrated_maze.mp4"
    h, w = all_frames[0].shape[:2]
    writer = cv2.VideoWriter(str(video_file),
                              cv2.VideoWriter_fourcc(*'mp4v'),
                              args.fps, (w, h))
    for f in all_frames:
        writer.write(f)
    writer.release()
    duration = len(all_frames) / args.fps
    print(f"    Saved: {video_file} ({len(all_frames)} frames, {duration:.1f}s)")


# ──────────────────────────────────────────────────────────────────────────────
# Demo 2: Physics discovery with verbal explanation
# ──────────────────────────────────────────────────────────────────────────────

def demo_physics(args):
    print("\n  Generating: Physics Discovery + Verbal Explanation")

    explainer = PhysicsExplainer()
    anomaly_exp = AnomalyExplainer()
    journal = DiscoveryJournal()

    scenarios = [
        ("Gravity", ForceType.GRAVITY, {'g': -9.81},
         PhysicsState(np.array([0.0, 10.0]), np.array([2.0, 0.0]))),
        ("Friction", ForceType.FRICTION, {'mu': 0.3},
         PhysicsState(np.array([0.0, 0.0]), np.array([5.0, 0.0]))),
        ("Magnetic", ForceType.MAGNETIC, {'center': [5, 5], 'strength': 3.0},
         PhysicsState(np.array([0.0, 0.0]), np.array([1.0, 1.0]))),
    ]

    all_frames = []
    img_w, img_h = 900, 580
    dt = 0.05

    for sci, (scenario_name, ftype, fparams, s0) in enumerate(scenarios):
        sim = SimplePhysicsSim(dt=dt)
        sim.add_force(scenario_name.lower(), ftype, **fparams)
        agent = DiscoveryAgent(dt=dt)

        trajectory = sim.rollout(s0, 100)
        discovered = False
        discovery_text = ""
        anomaly_text = ""

        for i, state in enumerate(trajectory):
            agent.observe(state)

            # Try discovery every 20 steps
            if (i + 1) % 20 == 0 and i > 5 and not discovered:
                result = agent.discover(verbose=False)
                if result:
                    discovered = True
                    discovery_text = (
                        f"DISCOVERED: {result.name}\n"
                        f"Error reduced by {result.error_reduction:.1f}%\n"
                        f"Confidence: {result.confidence:.2f}")
                    journal.log_discovery(i, result.name,
                                          f"Found via KB", result.confidence,
                                          "knowledge_base")
                else:
                    report = agent.introspect()
                    if report and report.anomaly_type.value == 'missing_law':
                        anomaly_text = anomaly_exp.explain(
                            report.direction, report.magnitude,
                            report.pattern, report.is_systematic,
                            agent.belief.known_forces)

            # ── Draw frame ──
            img = np.full((img_h, img_w, 3), BG, dtype=np.uint8)

            # Title
            cv2.rectangle(img, (0, 0), (img_w, 32), (40, 40, 45), -1)
            cv2.putText(img, f"Physics Discovery: {scenario_name}",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, f"Step {i+1}/100",
                        (img_w - 120, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (180, 180, 190), 1, cv2.LINE_AA)

            # Physics visualization (left) — show trajectory
            cv2.rectangle(img, (15, 42), (400, 340), PANEL_BG, -1)
            cv2.rectangle(img, (15, 42), (400, 340), PANEL_BORDER, 1)
            cv2.putText(img, "Trajectory", (25, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEXT_DARK, 1,
                        cv2.LINE_AA)

            # Scale trajectory to fit
            positions = np.array([s.pos for s in trajectory[:i+1]])
            if len(positions) > 1:
                pmin = positions.min(axis=0) - 1
                pmax = positions.max(axis=0) + 1
                prange = pmax - pmin
                prange = np.maximum(prange, 1.0)

                for j in range(1, len(positions)):
                    t = j / len(positions)
                    px1 = int(20 + (positions[j-1, 0] - pmin[0]) / prange[0] * 370)
                    py1 = int(335 - (positions[j-1, 1] - pmin[1]) / prange[1] * 280)
                    px2 = int(20 + (positions[j, 0] - pmin[0]) / prange[0] * 370)
                    py2 = int(335 - (positions[j, 1] - pmin[1]) / prange[1] * 280)
                    c = (int(60 + 160*t), int(75 - 40*t), int(215 - 150*t))
                    cv2.line(img, (px1, py1), (px2, py2), c, 2, cv2.LINE_AA)

                # Current position
                cpx = int(20 + (positions[-1, 0] - pmin[0]) / prange[0] * 370)
                cpy = int(335 - (positions[-1, 1] - pmin[1]) / prange[1] * 280)
                cv2.circle(img, (cpx, cpy), 6, AGENT_COL, -1, cv2.LINE_AA)

            # Cortisol gauge
            cortisol = agent._cortisol
            cv2.putText(img, f"Cortisol: {cortisol:.4f}",
                        (25, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        CRT_COL if cortisol > 0.01 else TEXT_LIGHT,
                        1, cv2.LINE_AA)

            # Known forces
            known_text = f"Known: {', '.join(agent.belief.known_forces) or 'F=ma only'}"
            cv2.putText(img, known_text, (220, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        ACH_COL if agent.belief.known_forces else TEXT_LIGHT,
                        1, cv2.LINE_AA)

            # Anomaly panel (right top)
            cv2.rectangle(img, (410, 42), (img_w - 15, 220),
                          PANEL_BG, -1)
            cv2.rectangle(img, (410, 42), (img_w - 15, 220),
                          PANEL_BORDER, 1)
            cv2.putText(img, "Anomaly Analysis", (420, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEXT_DARK, 1,
                        cv2.LINE_AA)

            if anomaly_text and not discovered:
                draw_text_wrapped(img, anomaly_text, 420, 78,
                                  max_width=460, font_scale=0.3,
                                  color=CRT_COL, line_height=15)
            elif not discovered:
                cv2.putText(img, "Observing...", (420, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            TEXT_LIGHT, 1, cv2.LINE_AA)

            # Discovery panel (right bottom)
            if discovered:
                cv2.rectangle(img, (410, 230), (img_w - 15, 340),
                              DISCOVERY_BG, -1)
                cv2.rectangle(img, (410, 230), (img_w - 15, 340),
                              DISCOVERY_BORDER, 2)
            else:
                cv2.rectangle(img, (410, 230), (img_w - 15, 340),
                              PANEL_BG, -1)
                cv2.rectangle(img, (410, 230), (img_w - 15, 340),
                              PANEL_BORDER, 1)

            cv2.putText(img, "Discovery", (420, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        SUCCESS_COL if discovered else TEXT_DARK, 1,
                        cv2.LINE_AA)

            if discovered:
                draw_text_wrapped(img, discovery_text, 420, 270,
                                  max_width=460, font_scale=0.35,
                                  color=SUCCESS_COL, line_height=18)

            # Journal (bottom)
            cv2.rectangle(img, (15, 350), (img_w - 15, img_h - 10),
                          JOURNAL_BG, -1)
            cv2.rectangle(img, (15, 350), (img_w - 15, img_h - 10),
                          PANEL_BORDER, 1)
            cv2.putText(img, "Discovery Journal", (25, 370),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEXT_DARK, 1,
                        cv2.LINE_AA)

            journal_text = journal.summary(last_n=5)
            draw_text_wrapped(img, journal_text, 25, 388,
                              max_width=img_w - 60, font_scale=0.3,
                              color=TEXT_MED, line_height=16)

            all_frames.append(img)

        # Hold final frame
        for _ in range(args.fps * 2):
            all_frames.append(all_frames[-1])

        tag = "DISCOVERED" if discovered else "missed"
        print(f"    {scenario_name}: {tag}")

    # Save
    video_file = OUT / "physics_discovery_narrated.mp4"
    h, w = all_frames[0].shape[:2]
    writer = cv2.VideoWriter(str(video_file),
                              cv2.VideoWriter_fourcc(*'mp4v'),
                              args.fps, (w, h))
    for f in all_frames:
        writer.write(f)
    writer.release()
    duration = len(all_frames) / args.fps
    print(f"    Saved: {video_file} ({len(all_frames)} frames, {duration:.1f}s)")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", default="all",
                    choices=["maze", "physics", "all"])
    ap.add_argument("--n-episodes", type=int, default=6)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("=" * 65)
    print("  Language Layer Video Demos")
    print("=" * 65)

    if args.demo in ("maze", "all"):
        demo_maze(args)

    if args.demo in ("physics", "all"):
        demo_physics(args)

    print(f"\n{'='*65}")
    print(f"  All videos generated in {OUT}/")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
