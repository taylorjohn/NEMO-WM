"""
gen_narrated_maze.py — Narrated Maze Navigation Video
=======================================================
Simulates PointMaze navigation with neuromodulatory HUD,
self-narration overlay, and belief state visualization.

No pygame needed — uses matplotlib animation + ffmpeg.

Usage:
    python gen_narrated_maze.py

Output: outputs/narrated_maze.mp4
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

Path("outputs").mkdir(exist_ok=True)

# Maze layout (U-maze)
WALLS = [
    ((0, 0), (8, 0)),   # bottom
    ((0, 8), (8, 8)),   # top
    ((0, 0), (0, 8)),   # left
    ((8, 0), (8, 8)),   # right
    ((2, 2), (6, 2)),   # inner bottom
    ((2, 2), (2, 6)),   # inner left
    ((6, 2), (6, 6)),   # inner right
]

START = np.array([1.0, 1.0])
GOAL = np.array([7.0, 1.0])
N_FRAMES = 200


def generate_trajectory():
    """Generate a U-maze trajectory with neuromod signals."""
    # Waypoints: bottom-left → top-left → top-right → bottom-right
    waypoints = [
        np.array([1.0, 1.0]),
        np.array([1.0, 7.0]),
        np.array([7.0, 7.0]),
        np.array([7.0, 1.0]),
    ]

    trajectory = []
    neuro = []
    narrations = []

    steps_per_segment = N_FRAMES // len(waypoints)
    rng = np.random.RandomState(42)

    for seg_i in range(len(waypoints)):
        start = waypoints[seg_i]
        end = waypoints[(seg_i + 1) % len(waypoints)]

        for step in range(steps_per_segment):
            t = step / steps_per_segment
            pos = start + t * (end - start)
            pos += rng.randn(2) * 0.05  # noise

            # Neuromodulatory signals
            da = 0.3 + 0.4 * float(step == 0)  # spike at waypoint
            ach = 0.5 + 0.3 * t  # confidence grows
            crt = 0.2 + 0.3 * float(seg_i == 2)  # stress at turn
            ne = 0.4 + 0.2 * rng.random()
            sht = 0.5

            # Narration
            if step == 0:
                narr = ["Turning at waypoint — DA spike",
                        "Heading north along corridor",
                        "Crossing to east side — stress elevated",
                        "Approaching goal — confidence high"][seg_i]
            elif step == steps_per_segment // 2:
                narr = ["Path clear, ACh rising",
                        "Halfway through corridor",
                        "Unfamiliar territory — CRT elevated",
                        "Goal in sight — planning horizon extends"][seg_i]
            else:
                narr = ""

            # WM capacity
            k_eff = max(2, 8 - int(crt * 6))

            # Planning horizon
            horizon = max(1, int(32 * ach))

            trajectory.append(pos.copy())
            neuro.append({
                "DA": da, "ACh": ach, "CRT": crt, "NE": ne, "5HT": sht,
                "K_eff": k_eff, "horizon": horizon,
            })
            narrations.append(narr)

    return trajectory, neuro, narrations


def make_video():
    print("Generating narrated maze video...")

    trajectory, neuro, narrations = generate_trajectory()

    fig = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor('#0d1117')

    ax_maze = fig.add_axes([0.02, 0.05, 0.45, 0.9])
    ax_neuro = fig.add_axes([0.52, 0.55, 0.45, 0.4])
    ax_narr = fig.add_axes([0.52, 0.05, 0.45, 0.45])

    last_narration = [""]

    def animate(frame_idx):
        if frame_idx >= len(trajectory):
            return

        pos = trajectory[frame_idx]
        nm = neuro[frame_idx]
        narr = narrations[frame_idx]
        if narr:
            last_narration[0] = narr

        # ── Maze panel ──
        ax_maze.clear()
        ax_maze.set_facecolor('#161b22')
        ax_maze.set_xlim(-0.5, 8.5)
        ax_maze.set_ylim(-0.5, 8.5)
        ax_maze.set_aspect('equal')
        ax_maze.set_title('PointMaze Navigation', color='white',
                          fontsize=13, fontweight='bold')
        ax_maze.tick_params(colors='#8b949e')
        for spine in ax_maze.spines.values():
            spine.set_color('#30363d')

        # Draw walls
        for (x1, y1), (x2, y2) in WALLS:
            ax_maze.plot([x1, x2], [y1, y2], color='#6b7280', linewidth=3)

        # Trail
        trail_start = max(0, frame_idx - 30)
        trail = trajectory[trail_start:frame_idx + 1]
        if len(trail) > 1:
            xs = [p[0] for p in trail]
            ys = [p[1] for p in trail]
            for i in range(len(xs) - 1):
                alpha = 0.1 + 0.9 * (i / len(xs))
                ax_maze.plot(xs[i:i+2], ys[i:i+2], color='#3b82f6',
                             alpha=alpha, linewidth=2)

        # Agent
        agent_color = '#ef4444' if nm['CRT'] > 0.4 else '#3b82f6'
        ax_maze.plot(pos[0], pos[1], 'o', color=agent_color,
                     markersize=12, zorder=10)

        # Goal
        ax_maze.plot(GOAL[0], GOAL[1], '*', color='#10b981',
                     markersize=15, zorder=10)
        ax_maze.text(GOAL[0], GOAL[1] + 0.4, 'GOAL', color='#10b981',
                     ha='center', fontsize=8)

        # Start
        ax_maze.plot(START[0], START[1], 's', color='#f59e0b',
                     markersize=8, zorder=5, alpha=0.5)

        # Distance to goal
        dist = np.linalg.norm(pos - GOAL)
        ax_maze.text(0, 8.2, f'Dist: {dist:.1f}  Step: {frame_idx}',
                     color='#8b949e', fontsize=9)

        # ── Neuromod panel ──
        ax_neuro.clear()
        ax_neuro.set_facecolor('#161b22')
        ax_neuro.set_title('Neuromodulatory State', color='white',
                           fontsize=11, fontweight='bold')
        for spine in ax_neuro.spines.values():
            spine.set_color('#30363d')

        signals = ['DA', 'ACh', 'CRT', 'NE', '5HT']
        values = [nm[s] for s in signals]
        colors = ['#f59e0b', '#3b82f6', '#ef4444', '#8b5cf6', '#06b6d4']
        optimals = [0.4, 0.6, 0.3, 0.5, 0.5]

        bars = ax_neuro.barh(range(len(signals)), values, color=colors,
                              alpha=0.8, height=0.6)

        # Optimal markers
        for i, opt in enumerate(optimals):
            ax_neuro.plot(opt, i, '|', color='white', markersize=15,
                          markeredgewidth=2)

        ax_neuro.set_yticks(range(len(signals)))
        ax_neuro.set_yticklabels(signals, color='white', fontsize=10)
        ax_neuro.set_xlim(0, 1)
        ax_neuro.tick_params(colors='#8b949e')

        # WM and horizon
        ax_neuro.text(0.7, 4.5, f"WM K={nm['K_eff']}",
                      color='#10b981', fontsize=10, fontweight='bold',
                      transform=ax_neuro.transData)
        ax_neuro.text(0.7, 3.7, f"H={nm['horizon']}",
                      color='#58a6ff', fontsize=10,
                      transform=ax_neuro.transData)

        # ── Narration panel ──
        ax_narr.clear()
        ax_narr.set_facecolor('#161b22')
        ax_narr.set_xlim(0, 10)
        ax_narr.set_ylim(0, 10)
        ax_narr.axis('off')
        ax_narr.set_title('Self-Narration', color='white',
                          fontsize=11, fontweight='bold')

        # Current narration
        if last_narration[0]:
            ax_narr.text(5, 7, f'"{last_narration[0]}"',
                         color='#58a6ff', fontsize=11, ha='center',
                         style='italic', fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.5',
                                   facecolor='#1a3a5c', edgecolor='#58a6ff',
                                   alpha=0.8))

        # Status
        status_lines = [
            f"Q1: Where am I? → ({pos[0]:.1f}, {pos[1]:.1f})",
            f"Q7: Plan horizon → {nm['horizon']} steps",
            f"Q13: WM capacity → K={nm['K_eff']}",
            f"Q8: Trust prediction → α={'high' if nm['ACh'] > 0.5 else 'low'}",
        ]
        for i, line in enumerate(status_lines):
            ax_narr.text(0.5, 4.5 - i * 1.2, line,
                         color='#8b949e', fontsize=9, fontfamily='monospace')

        # Mood
        if nm['DA'] > 0.5:
            mood = "Curious-Alert"
        elif nm['CRT'] > 0.4:
            mood = "Stressed-Cautious"
        elif nm['ACh'] > 0.6:
            mood = "Confident-Focused"
        else:
            mood = "Calm-Relaxed"

        ax_narr.text(5, 0.5, f"Mood: {mood}",
                     color='#f59e0b', fontsize=11, ha='center',
                     fontweight='bold')

    print(f"  Frames: {N_FRAMES}")
    print(f"  Rendering...")

    anim = animation.FuncAnimation(fig, animate, frames=N_FRAMES,
                                     interval=50, blit=False)

    out_path = "outputs/narrated_maze.mp4"
    try:
        anim.save(out_path, writer='ffmpeg', fps=20, dpi=100,
                  savefig_kwargs={'facecolor': '#0d1117'})
        print(f"  Saved: {out_path}")
        size_mb = Path(out_path).stat().st_size / 1e6
        print(f"  Size: {size_mb:.1f} MB")
    except Exception as e:
        print(f"  ffmpeg failed: {e}")
        try:
            gif_path = out_path.replace('.mp4', '.gif')
            anim.save(gif_path, writer='pillow', fps=15)
            print(f"  Saved as GIF: {gif_path}")
        except:
            print(f"  Install ffmpeg: conda install ffmpeg")

    plt.close()
    print("Done!")


if __name__ == "__main__":
    make_video()
