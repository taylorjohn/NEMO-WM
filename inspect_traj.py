import numpy as np
from pathlib import Path
from PIL import Image

data  = np.load('./benchmark_data/umaze/trajectories.npy', allow_pickle=True)
traj  = data[0]
obs   = traj['observations']
print(f'Traj 0 obs: type={type(obs)}, len={len(obs)}')
o0 = obs[0]
print(f'  obs[0]: type={type(o0)}, ', end='')
if hasattr(o0, 'shape'):
    print(f'shape={o0.shape}, dtype={o0.dtype}, min={o0.min():.2f}, max={o0.max():.2f}')
else:
    print(f'value={o0}')

# Show a few trajectories
print(f'Total trajectories: {len(data)}')
for i in [0,1,2]:
    obs_i = data[i]['observations']
    act_i = data[i]['actions']
    print(f'  Traj {i}: obs={len(obs_i)} steps, acts={len(act_i)}')
