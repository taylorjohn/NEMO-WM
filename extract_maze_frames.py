import numpy as np
from PIL import Image
from pathlib import Path

for env in ['wall', 'umaze', 'medium']:
    traj_path = f'./benchmark_data/{env}/trajectories.npy'
    if not Path(traj_path).exists():
        print(f'Skip {env}')
        continue

    out_dir = Path(f'./maze_frames/{env}_cls/maze')
    out_dir.mkdir(parents=True, exist_ok=True)

    trajs = np.load(traj_path, allow_pickle=True)
    frame_idx = 0
    n_trajs = min(200, len(trajs))   # 200 trajectories = up to 10000 frames

    for i in range(n_trajs):
        obs = np.array(trajs[i]['observations'])  # (T, 64, 64, 3)
        for t in range(0, len(obs), 5):           # every 5th frame
            frame = Image.fromarray(obs[t].astype(np.uint8))
            frame = frame.resize((224, 224), Image.BILINEAR)
            frame.save(out_dir / f'frame_{frame_idx:06d}.png')
            frame_idx += 1

    print(f'{env}: {frame_idx} frames saved to {out_dir}')
