import numpy as np, json
from pathlib import Path
from PIL import Image

data    = np.load('./benchmark_data/pusht/trajectories.npy', allow_pickle=True)
out_dir = Path('./phase2_pusht_frames')
out_dir.mkdir(exist_ok=True)

all_acts = np.concatenate([np.array(data[i]['actions']) for i in range(50)])
act_scale = float(np.abs(all_acts).max())
print(f'Action scale: {act_scale:.3f}')

n_traj = 200; frame_idx = 0; meta = []
for t_idx in range(n_traj):
    obs  = data[t_idx]['observations']
    acts = np.array(data[t_idx]['actions'])
    for step_i, frame in enumerate(obs):
        arr = np.array(frame)
        if arr.ndim == 3 and arr.shape[2] == 3:
            img = Image.fromarray(arr.astype('uint8'), 'RGB').resize((224,224), Image.BILINEAR)
        elif arr.size % 3 == 0:
            s = int((arr.size//3)**0.5)
            img = Image.fromarray(arr.reshape(s,s,3).astype('uint8')).resize((224,224), Image.BILINEAR)
        else:
            continue
        fname = f'frame_{frame_idx:07d}.png'
        img.save(out_dir / fname)
        act = acts[step_i].tolist() if step_i < len(acts) else [0.0, 0.0]
        meta.append({'frame': fname, 'action': [a/(act_scale+1e-8) for a in act], 'traj': t_idx, 'step': step_i})
        frame_idx += 1
    if t_idx % 50 == 0:
        print(f'  {t_idx}/{n_traj} — {frame_idx} frames')

json.dump({'act_scale': act_scale, 'frames': meta}, open(out_dir/'flow_meta.json','w'), indent=2)
print(f'Done — {frame_idx} frames')
