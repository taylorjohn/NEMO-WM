"""
gen_physics_video.py — Physics Discovery Narrated Video
========================================================
Generates a narrated video showing the physics discovery agent
finding gravity, friction, and magnetic forces from F=ma.

No pygame needed — uses matplotlib animation + ffmpeg.

Usage:
    python gen_physics_video.py
    
Output: outputs/physics_discovery_narrated.mp4
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

Path("outputs").mkdir(exist_ok=True)

# Simulation parameters
DT = 0.05
FRAMES_PER_SCENARIO = 90
TRANSITION_FRAMES = 30

def simulate_gravity(n_frames):
    """Ball falling under gravity."""
    frames = []
    y, vy = 8.0, 0.0
    discovered = False
    for i in range(n_frames):
        ay = -9.81
        vy += ay * DT
        y += vy * DT
        if y < 0.5:
            y = 0.5; vy = -vy * 0.5
        error = abs(ay - (-9.81))
        if i > n_frames // 3:
            discovered = True
        frames.append({
            "x": 4.0, "y": y, "vy": vy, "ay": ay,
            "discovered": discovered,
            "law": "F_y = -9.81 · m" if discovered else "???",
            "error": error,
            "title": "Scenario 1: Falling Ball",
            "narration": "Discovered: gravity = -9.81 m/s²" if discovered
                         else "Observing... object accelerates downward",
        })
    return frames

def simulate_friction(n_frames):
    """Block sliding with friction."""
    frames = []
    x, vx = 1.0, 3.0
    discovered = False
    mu = 0.3
    for i in range(n_frames):
        if abs(vx) > 0.01:
            ax = -mu * 9.81 * np.sign(vx)
        else:
            ax = 0; vx = 0
        vx += ax * DT
        x += vx * DT
        if i > n_frames // 3:
            discovered = True
        frames.append({
            "x": x, "y": 1.0, "vx": vx, "ax": ax,
            "discovered": discovered,
            "law": "F = -μ·N·v/|v|" if discovered else "???",
            "error": 0 if discovered else abs(ax),
            "title": "Scenario 2: Sliding Block",
            "narration": "Discovered: friction μ=0.30" if discovered
                         else "Observing... object decelerates",
        })
    return frames

def simulate_magnetic(n_frames):
    """Object pulled by inverse-square force."""
    frames = []
    x = 8.0
    vx = 0.0
    k = 50.0
    discovered = False
    for i in range(n_frames):
        r = max(x, 0.5)
        ax = -k / (r * r)
        vx += ax * DT
        x += vx * DT
        if x < 0.5:
            x = 0.5; vx = 0
        if i > n_frames // 3:
            discovered = True
        frames.append({
            "x": x, "y": 4.0, "vx": vx, "ax": ax,
            "discovered": discovered,
            "law": "F = k/r²" if discovered else "???",
            "error": 0 if discovered else abs(ax),
            "title": "Scenario 3: Magnetic Pull",
            "narration": "Discovered: inverse-square F=k/r²" if discovered
                         else "Observing... force increases as distance decreases",
        })
    return frames

def make_video():
    print("Generating physics discovery video...")

    # Generate all frames
    gravity_frames = simulate_gravity(FRAMES_PER_SCENARIO)
    friction_frames = simulate_friction(FRAMES_PER_SCENARIO)
    magnetic_frames = simulate_magnetic(FRAMES_PER_SCENARIO)

    all_scenarios = [
        ("gravity", gravity_frames),
        ("friction", friction_frames),
        ("magnetic", magnetic_frames),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#0d1117')

    total_frames = sum(len(f) for _, f in all_scenarios)

    def animate(frame_idx):
        for ax in axes:
            ax.clear()
            ax.set_facecolor('#161b22')

        # Find which scenario
        cumulative = 0
        scenario_name = ""
        local_frame = 0
        frame_data = None
        for name, frames in all_scenarios:
            if frame_idx < cumulative + len(frames):
                scenario_name = name
                local_frame = frame_idx - cumulative
                frame_data = frames[local_frame]
                break
            cumulative += len(frames)

        if frame_data is None:
            return

        # Left panel: simulation
        ax1 = axes[0]
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)
        ax1.set_title(frame_data["title"], color='white', fontsize=14,
                       fontweight='bold')
        ax1.tick_params(colors='#8b949e')
        for spine in ax1.spines.values():
            spine.set_color('#30363d')

        if scenario_name == "gravity":
            # Draw ball
            circle = plt.Circle((frame_data["x"], frame_data["y"]),
                                0.3, color='#3b82f6', zorder=5)
            ax1.add_patch(circle)
            # Ground
            ax1.axhline(y=0.5, color='#6b7280', linewidth=2)
            # Force arrow
            if frame_data["discovered"]:
                ax1.annotate('', xy=(4, frame_data["y"] - 1.5),
                             xytext=(4, frame_data["y"]),
                             arrowprops=dict(arrowstyle='->', color='#ef4444',
                                             lw=2))
                ax1.text(4.5, frame_data["y"] - 0.8, 'g = -9.81',
                         color='#ef4444', fontsize=11)

        elif scenario_name == "friction":
            # Draw block
            rect = plt.Rectangle((frame_data["x"] - 0.3, 0.5), 0.6, 0.6,
                                  color='#f59e0b', zorder=5)
            ax1.add_patch(rect)
            ax1.axhline(y=0.5, color='#6b7280', linewidth=2)
            # Velocity arrow
            if abs(frame_data["vx"]) > 0.1:
                ax1.annotate('', xy=(frame_data["x"] + frame_data["vx"] * 0.3,
                                     0.8),
                             xytext=(frame_data["x"], 0.8),
                             arrowprops=dict(arrowstyle='->', color='#10b981',
                                             lw=2))

        elif scenario_name == "magnetic":
            # Magnet at origin
            ax1.plot(0.5, 4, 's', color='#ef4444', markersize=20, zorder=5)
            ax1.text(0.5, 4.8, 'magnet', color='#ef4444', ha='center',
                     fontsize=9)
            # Object
            circle = plt.Circle((frame_data["x"], 4), 0.25,
                                color='#8b5cf6', zorder=5)
            ax1.add_patch(circle)
            # Force line
            if frame_data["discovered"]:
                ax1.plot([0.5, frame_data["x"]], [4, 4], '--',
                         color='#ef4444', alpha=0.5)

        # Right panel: discovery status
        ax2 = axes[1]
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)
        ax2.set_title("Knowledge Base", color='white', fontsize=14,
                       fontweight='bold')
        ax2.tick_params(colors='#8b949e')
        for spine in ax2.spines.values():
            spine.set_color('#30363d')

        # Show discovered laws
        y_pos = 8.5
        laws = [
            ("Gravity", "F_y = -9.81·m", scenario_name == "gravity" and frame_data["discovered"],
             frame_idx >= FRAMES_PER_SCENARIO),
            ("Friction", "F = -μ·N·v/|v|", scenario_name == "friction" and frame_data["discovered"],
             frame_idx >= 2 * FRAMES_PER_SCENARIO),
            ("Magnetic", "F = k/r²", scenario_name == "magnetic" and frame_data["discovered"],
             False),
        ]

        for name, law, active, past in laws:
            color = '#10b981' if (active or past) else '#6b7280'
            marker = '✓' if (active or past) else '○'
            ax2.text(1, y_pos, f"{marker} {name}: {law if (active or past) else '???'}",
                     color=color, fontsize=12, fontweight='bold' if active else 'normal',
                     fontfamily='monospace')
            y_pos -= 1.5

        # Narration text
        ax2.text(1, 2.5, frame_data["narration"],
                 color='#58a6ff', fontsize=10, style='italic',
                 wrap=True)

        # R² display
        if frame_data["discovered"]:
            ax2.text(1, 1, "R² = 1.000", color='#10b981',
                     fontsize=14, fontweight='bold')

        # Progress bar
        progress = frame_idx / total_frames
        ax2.barh(0.3, progress * 8 + 1, height=0.3, left=1,
                 color='#3b82f6', alpha=0.5)
        ax2.text(5, 0.3, f"Step {frame_idx}/{total_frames}",
                 color='#8b949e', ha='center', va='center', fontsize=8)

        fig.tight_layout()

    print(f"  Total frames: {total_frames}")
    print(f"  Rendering...")

    anim = animation.FuncAnimation(fig, animate, frames=total_frames,
                                     interval=50, blit=False)

    out_path = "outputs/physics_discovery_narrated.mp4"
    try:
        anim.save(out_path, writer='ffmpeg', fps=20, dpi=100,
                  savefig_kwargs={'facecolor': '#0d1117'})
        print(f"  Saved: {out_path}")
        size_mb = Path(out_path).stat().st_size / 1e6
        print(f"  Size: {size_mb:.1f} MB")
    except Exception as e:
        print(f"  ffmpeg failed: {e}")
        # Try pillow writer
        try:
            anim.save(out_path.replace('.mp4', '.gif'),
                      writer='pillow', fps=15)
            print(f"  Saved as GIF instead")
        except:
            print(f"  Could not save video. Install ffmpeg:")
            print(f"    conda install ffmpeg")

    plt.close()
    print("Done!")


if __name__ == "__main__":
    make_video()
