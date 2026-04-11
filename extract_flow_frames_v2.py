import numpy as np, json
from pathlib import Path
from PIL import Image

data    = np.load('./benchmark_data/umaze/trajectories.npy', allow_pickle=True)
out_dir = Path('./phase2_flow_frames')
out_dir.mkdir(exist_ok=True)

# Check action range first
all_acts = np.concatenate([np.array(d['actions']) for d in data[:50]])
print(f'Action range: min={all_acts.min():.3f} max={all_acts.max():.3f} mean={np.abs(all_acts).mean():.3f}')
act_scale = float(np.abs(all_acts).max())
print(f'Scale factor: {act_scale:.3f}')

n_traj    = 400
frame_idx = 0
meta      = []
print(f'Extracting {n_traj} trajectories...')
for t_idx in range(n_traj):
    obs  = data[t_idx]['observations']
    acts = np.array(data[t_idx]['actions'])
    for step_i, obs_frame in enumerate(obs):
        img = Image.fromarray(obs_frame.astype('uint8'), 'RGB').resize((224,224), Image.BILINEAR)
        fname = f'frame_{frame_idx:07d}.png'
        img.save(out_dir / fname)
        # Save normalised action for this step
        act = acts[step_i].tolist() if step_i < len(acts) else [0.0, 0.0]
        act_norm = [a / (act_scale + 1e-8) for a in act]
        meta.append({'frame': fname, 'action': act_norm, 'traj': t_idx, 'step': step_i})
        frame_idx += 1
    if t_idx % 50 == 0:
        print(f'  {t_idx}/{n_traj} — {frame_idx} frames')

json.dump({'act_scale': act_scale, 'frames': meta},
          open(out_dir / 'flow_meta.json', 'w'), indent=2)
print(f'Done — {frame_idx} frames + flow_meta.json')
