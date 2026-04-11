import numpy as np
data = np.load('./benchmark_data/umaze/trajectories.npy', allow_pickle=True)
traj = data[0]
print('Keys:', list(traj.keys()))
obs = traj['observations']
# Check if any key has position/state info
for k in traj.keys():
    v = traj[k]
    if hasattr(v, '__len__'):
        sample = v[0] if hasattr(v[0], '__len__') else v[0]
        print(f'  {k}: len={len(v)}, sample shape={np.array(sample).shape}, sample={np.array(sample)[:4]}')
