"""
make_pusht_video.py — NeMo Flow Policy PushT visualisation
Proper T-shape block + target zone + agent trajectory trails.

Usage:
    python make_pusht_video.py
Output:
    figures/pusht_flow_policy.gif   (full quality)
    figures/pusht_web.gif           (web-optimised, smaller)
    figures/pusht_flow_policy.mp4   (if ffmpeg available)
"""

import numpy as np, math, imageio, torch
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from train_action_wm import StateEncoder
from block_probe import BlockProbe
from nemo_flow_policy import NeMoFlowPolicy, NeMoFlowAgent


# ── Load models ──────────────────────────────────────────────────────────────

device = torch.device('cpu')

enc_ckpt = torch.load('checkpoints/action_wm/action_wm_pusht_full_best.pt',
                      map_location=device, weights_only=False)
encoder = StateEncoder(5, 128).to(device)
encoder.load_state_dict(enc_ckpt['encoder']); encoder.eval()

p_ckpt = torch.load('checkpoints/action_wm/block_probe_best.pt',
                    map_location=device, weights_only=False)
probe = BlockProbe(128).to(device)
probe.load_state_dict(p_ckpt['probe']); probe.eval()

pol_ckpt = torch.load('checkpoints/flow_policy/nemo_flow_best.pt',
                      map_location=device, weights_only=False)
policy = NeMoFlowPolicy(H=8).to(device)
policy.load_state_dict(pol_ckpt['policy']); policy.eval()

agent = NeMoFlowAgent(encoder, probe, policy, device, n_steps=10)


# ── T-shape drawing ───────────────────────────────────────────────────────────

def t_verts(cx, cy, angle, size=0.10):
    """Return (hbar_verts, stem_verts) for a T centred at cx,cy rotated by angle."""
    s = size
    hbar_local = np.array([[-s,    s*0.28],
                            [ s,    s*0.28],
                            [ s,   -s*0.08],
                            [-s,   -s*0.08]])
    stem_local = np.array([[-s*0.28, -s*0.08],
                            [ s*0.28, -s*0.08],
                            [ s*0.28, -s*0.80],
                            [-s*0.28, -s*0.80]])
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    centre = np.array([cx, cy])
    hbar = (R @ hbar_local.T).T + centre
    stem = (R @ stem_local.T).T + centre
    return hbar, stem


def draw_T(ax, cx, cy, angle, size=0.10,
           fc='#2563EB', ec='#93c5fd', alpha=0.92, lw=1.8, zorder=4):
    hbar, stem = t_verts(cx, cy, angle, size)
    for verts in [hbar, stem]:
        ax.add_patch(patches.Polygon(
            verts, closed=True, fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=zorder
        ))


def draw_T_target(ax, cx, cy, angle, size=0.115):
    hbar, stem = t_verts(cx, cy, angle, size)
    for verts in [hbar, stem]:
        ax.add_patch(patches.Polygon(
            verts, closed=True, fc='#10b981', ec='none', alpha=0.15, zorder=2
        ))
        ax.add_patch(patches.Polygon(
            verts, closed=True, fc='none', ec='#34d399', lw=2.0, alpha=0.7,
            linestyle='--', zorder=3
        ))


# ── Render ────────────────────────────────────────────────────────────────────

Path('figures').mkdir(exist_ok=True)
rng = np.random.RandomState(42)
all_frames = []
TARGET_ANGLE = math.pi / 5   # 36 deg — looks clean on screen

N_EPISODES  = 4
MAX_STEPS   = 250
TRAIL_LEN   = 14             # number of past positions to show

for ep in range(N_EPISODES):
    goal      = np.array([0.65 + rng.uniform(-0.08, 0.08),
                           0.65 + rng.uniform(-0.08, 0.08)])
    agent.reset()
    agent_pos = rng.uniform(0.10, 0.35, 2)
    block     = rng.uniform(0.30, 0.55, 2)
    angle     = rng.uniform(0, 2 * math.pi)

    agent_history: list = []   # trail of past agent positions
    ep_frames:    list = []

    for step in range(MAX_STEPS):
        obs = np.array([agent_pos[0], agent_pos[1],
                        block[0],     block[1],
                        angle / (2 * math.pi)], dtype=np.float32)

        action, da, _ = agent.act(obs, goal)
        action = np.clip(action, 0, 1)

        # Physics
        agent_pos += (action - agent_pos) * 0.4 + rng.normal(0, 0.01, 2)
        agent_pos  = np.clip(agent_pos, 0, 1)

        contact = np.linalg.norm(agent_pos - block) < 0.10
        if contact:
            push   = (agent_pos - block) * 0.20
            block  = np.clip(block - push, 0, 1)
            push_dir = math.atan2(-push[1], -push[0])
            angle    = 0.88 * angle + 0.12 * push_dir + rng.normal(0, 0.025)

        dist    = np.linalg.norm(block - goal)
        success = dist < 0.12

        # Store trail
        agent_history.append(agent_pos.copy())
        if len(agent_history) > TRAIL_LEN:
            agent_history.pop(0)

        # ── Draw frame ────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(4.5, 4.5), facecolor='#060B14')
        ax.set_facecolor('#060B14')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)

        # Grid
        for v in np.arange(0.2, 1.0, 0.2):
            ax.axhline(v, color='#1E293B', lw=0.5, alpha=0.35)
            ax.axvline(v, color='#1E293B', lw=0.5, alpha=0.35)

        # Target T-zone
        draw_T_target(ax, goal[0], goal[1], TARGET_ANGLE)

        # T-block
        draw_T(ax, block[0], block[1], angle, size=0.10,
               fc='#2563EB', ec='#93c5fd', alpha=0.93, lw=1.8)

        # DA colour
        da_c = '#ef4444' if da > 0.6 else '#f59e0b' if da > 0.3 else '#60a5fa'

        # Agent trajectory trail (fading circles)
        n_hist = len(agent_history)
        for ti, tpos in enumerate(agent_history[:-1]):
            alpha = ((ti + 1) / n_hist) * 0.35
            r     = 0.012 + 0.008 * (ti / n_hist)
            ax.add_patch(plt.Circle(tpos, r, color=da_c,
                                    alpha=alpha, zorder=6))

        # Agent (end-effector)
        ax.add_patch(plt.Circle(agent_pos, 0.033, color=da_c,
                                zorder=8, alpha=0.95))
        ax.add_patch(plt.Circle(agent_pos, 0.033, color='white',
                                fill=False, lw=1.5, zorder=9, alpha=0.55))

        # Contact flash
        if contact:
            ax.add_patch(plt.Circle(agent_pos, 0.055, color=da_c,
                                    fill=False, lw=1.0, zorder=7, alpha=0.35))

        # DA bar strip
        ax.add_patch(patches.Rectangle((0, 0), 1, 0.048,
                     fc='#0D1117', ec='none', zorder=10))
        ax.add_patch(patches.Rectangle((0.055, 0.007), da * 0.93, 0.033,
                     fc=da_c, ec='none', alpha=0.82, zorder=11))
        ax.text(0.006, 0.023, 'DA', color='#64748b', fontsize=6.2,
                fontfamily='monospace', va='center', zorder=12)
        ax.text(0.058 + da * 0.93 + 0.01, 0.023,
                f'{da:.2f}', color=da_c, fontsize=6.2,
                fontfamily='monospace', va='center', zorder=12)

        # Info overlay
        temp_str = 'HOT ' if da > 0.6 else 'WARM' if da > 0.3 else 'COLD'
        status   = 'SUCCESS' if success else f'dist={dist:.3f}'
        s_color  = '#10b981' if success else '#475569'

        ax.text(0.02, 0.995, 'NeMo-WM  Flow Policy',
                color='#e2e8f0', fontsize=8.2, fontfamily='monospace',
                transform=ax.transAxes, va='top', fontweight='bold')
        ax.text(0.02, 0.945,
                f'ep {ep+1}/{N_EPISODES}   step {step+1:3d}   {temp_str}',
                color='#94a3b8', fontsize=6.8, fontfamily='monospace',
                transform=ax.transAxes, va='top')
        ax.text(0.02, 0.895, status,
                color=s_color, fontsize=7.0, fontfamily='monospace',
                transform=ax.transAxes, va='top', fontweight='bold')

        # Legend
        for i, (lc, lbl) in enumerate([('#2563EB','block'),
                                        (da_c,    'agent'),
                                        ('#10b981','target')]):
            y = 0.995 - i * 0.048
            ax.add_patch(plt.Circle((0.83, y), 0.011, color=lc,
                                    transform=ax.transAxes, zorder=13,
                                    clip_on=False))
            ax.text(0.855, y, lbl, color='#64748b', fontsize=5.8,
                    fontfamily='monospace', transform=ax.transAxes, va='center')

        fig.tight_layout(pad=0)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(),
                            dtype=np.uint8).reshape(h, w, 4)
        ep_frames.append(buf[:, :, :3].copy())
        plt.close(fig)

        if success:
            all_frames.extend(ep_frames)
            all_frames.extend([ep_frames[-1]] * 30)  # hold on success
            print(f'  ep{ep+1}: SUCCESS in {step+1} steps  DA={da:.2f}')
            break
    else:
        all_frames.extend(ep_frames)
        print(f'  ep{ep+1}: {len(ep_frames)} steps (timeout)')

# ── Save outputs ──────────────────────────────────────────────────────────────

# Full quality GIF (every frame, 20fps)
gif_path = 'figures/pusht_flow_policy.gif'
imageio.mimsave(gif_path, all_frames, fps=20, loop=0)
size_mb = Path(gif_path).stat().st_size / 1e6
print(f'\nSaved {gif_path}  ({len(all_frames)} frames, {size_mb:.1f} MB)')

# Web-optimised GIF (every 2nd frame, 12fps)
web_path = 'figures/pusht_web.gif'
imageio.mimsave(web_path, all_frames[::2], fps=12, loop=0)
web_mb = Path(web_path).stat().st_size / 1e6
print(f'Saved {web_path}  ({len(all_frames)//2} frames, {web_mb:.1f} MB)')

# MP4
try:
    mp4_path = 'figures/pusht_flow_policy.mp4'
    imageio.mimsave(mp4_path, all_frames, fps=30, macro_block_size=1)
    print(f'Saved {mp4_path}')
except Exception as e:
    print(f'MP4 skipped: {e}')
