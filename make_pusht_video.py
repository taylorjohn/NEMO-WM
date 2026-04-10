import numpy as np, math, imageio, torch
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from train_action_wm import StateEncoder
from block_probe import BlockProbe
from nemo_flow_policy import NeMoFlowPolicy, NeMoFlowAgent

device = torch.device('cpu')
enc_ckpt = torch.load('checkpoints/action_wm/action_wm_pusht_full_best.pt', map_location=device, weights_only=False)
encoder = StateEncoder(5, 128).to(device); encoder.load_state_dict(enc_ckpt['encoder']); encoder.eval()
p_ckpt = torch.load('checkpoints/action_wm/block_probe_best.pt', map_location=device, weights_only=False)
probe = BlockProbe(128).to(device); probe.load_state_dict(p_ckpt['probe']); probe.eval()
pol_ckpt = torch.load('checkpoints/flow_policy/nemo_flow_best.pt', map_location=device, weights_only=False)
policy = NeMoFlowPolicy(H=8).to(device); policy.load_state_dict(pol_ckpt['policy']); policy.eval()
agent = NeMoFlowAgent(encoder, probe, policy, device, n_steps=10)

Path('figures').mkdir(exist_ok=True)
rng = np.random.RandomState(42)
all_frames = []

TEMP_LABELS = {True: 'HOT', False: 'WARM'}

for ep in range(4):
    goal = np.array([0.65+rng.uniform(-0.08,0.08), 0.65+rng.uniform(-0.08,0.08)])
    agent.reset()
    agent_pos = rng.uniform(0.1, 0.35, 2)
    block = rng.uniform(0.3, 0.55, 2)
    angle = rng.uniform(0, 2*math.pi)
    ep_frames = []

    for step in range(200):
        obs = np.array([agent_pos[0],agent_pos[1],block[0],block[1],angle/(2*math.pi)], dtype=np.float32)
        action, da, temp = agent.act(obs, goal)
        action = np.clip(action, 0, 1)
        agent_pos += (action-agent_pos)*0.4 + rng.normal(0,0.01,2)
        agent_pos = np.clip(agent_pos, 0, 1)
        if np.linalg.norm(agent_pos-block) < 0.1:
            push = (agent_pos-block)*0.2
            block = np.clip(block-push, 0, 1)
            angle += rng.normal(0, 0.05)
        dist = np.linalg.norm(block-goal)
        success = dist < 0.12

        fig, ax = plt.subplots(figsize=(4,4), facecolor='#0F172A')
        ax.set_facecolor('#0F172A'); ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)

        # Goal zone
        ax.add_patch(plt.Circle(goal, 0.12, color='#10b981', alpha=0.18))
        ax.add_patch(plt.Circle(goal, 0.12, color='#10b981', fill=False, lw=2, ls='--'))
        ax.plot(*goal, 'x', color='#10b981', ms=10, mew=2.5)

        # T-block
        bx,by = block
        ax.add_patch(patches.FancyBboxPatch((bx-0.07,by-0.04),0.14,0.08,
            boxstyle='round,pad=0.01',fc='#3b82f6',ec='#60a5fa',lw=2,alpha=0.9))
        ax.add_patch(patches.FancyBboxPatch((bx-0.03,by-0.09),0.06,0.05,
            boxstyle='round,pad=0.01',fc='#3b82f6',ec='#60a5fa',lw=2,alpha=0.9))

        # Agent
        ax.add_patch(plt.Circle(agent_pos, 0.035, color='#ef4444', zorder=5))
        ax.add_patch(plt.Circle(agent_pos, 0.035, color='#fca5a5', fill=False, lw=1.5, zorder=6))

        # DA bar
        da_c = '#ef4444' if da>0.6 else '#f59e0b' if da>0.3 else '#3b82f6'
        ax.barh(0.025, da, height=0.03, left=0, color=da_c, alpha=0.7)
        ax.text(0.01, 0.025, 'DA', color='white', fontsize=6, fontfamily='monospace', va='center')

        # Labels — ASCII only, no emoji
        temp_str = 'HOT' if da>0.6 else 'WARM' if da>0.3 else 'COLD'
        status = 'SUCCESS' if success else f'dist={dist:.3f}'
        ax.text(0.02,0.97, f'NeMo Flow  ep{ep+1}  step {step+1}',
                color='white', fontsize=7.5, fontfamily='monospace', transform=ax.transAxes, va='top')
        ax.text(0.02,0.91, f'DA={da:.2f}  {temp_str}',
                color=da_c, fontsize=7, fontfamily='monospace', transform=ax.transAxes, va='top')
        ax.text(0.02,0.85, status,
                color='#10b981' if success else '#94a3b8', fontsize=7,
                fontfamily='monospace', transform=ax.transAxes, va='top')

        fig.tight_layout(pad=0)
        fig.canvas.draw()
        w,h = fig.canvas.get_width_height()
        # Fix: use buffer_rgba then drop alpha channel
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h,w,4)
        img = buf[:,:,:3].copy()
        ep_frames.append(img)
        plt.close(fig)

        if success:
            all_frames.extend(ep_frames)
            all_frames.extend([ep_frames[-1]]*20)
            print(f'ep{ep+1}: SUCCESS in {step+1} steps')
            break
    else:
        all_frames.extend(ep_frames)
        print(f'ep{ep+1}: {len(ep_frames)} steps (no success)')

imageio.mimsave('figures/pusht_flow_policy.gif', all_frames[::2], fps=15, loop=0)
print(f'Saved figures/pusht_flow_policy.gif ({len(all_frames)} frames total)')
