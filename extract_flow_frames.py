import numpy as np
from pathlib import Path
from PIL import Image

data    = np.load('./benchmark_data/umaze/trajectories.npy', allow_pickle=True)
out_dir = Path('./phase2_flow_frames')
out_dir.mkdir(exist_ok=True)

n_traj  = 400
frame_idx = 0
print(f'Extracting frames from {n_traj} trajectories...')
for t_idx in range(n_traj):
    obs = data[t_idx]['observations']
    for obs_frame in obs:
        img = Image.fromarray(obs_frame.astype('uint8'), 'RGB').resize((224,224), Image.BILINEAR)
        img.save(out_dir / f'frame_{frame_idx:07d}.png')
        frame_idx += 1
    if t_idx % 50 == 0:
        print(f'  traj {t_idx}/{n_traj} — {frame_idx} frames saved')

print(f'Done — {frame_idx} frames in {out_dir}')
