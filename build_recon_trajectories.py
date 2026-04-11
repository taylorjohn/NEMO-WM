import h5py, numpy as np, json, io
from pathlib import Path
from PIL import Image
from torchvision import transforms

recon_dir = Path('./recon_data/recon_release')
out_file  = './benchmark_data/recon/trajectories.npy'
Path('./benchmark_data/recon').mkdir(parents=True, exist_ok=True)

T = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224)])
files = sorted(recon_dir.glob('*.hdf5'))[:200]

trajectories = []
for fi, fpath in enumerate(files):
    h = h5py.File(str(fpath), 'r')
    imgs_raw   = h['images/rgb_left'][:]
    lin_vels   = h['commands/linear_velocity'][:]
    ang_vels   = h['commands/angular_velocity'][:]
    collisions = h['collision/any'][:]
    
    obs = []; actions = []
    for step in range(len(imgs_raw)):
        if collisions[step]: continue
        img = np.array(T(Image.open(io.BytesIO(imgs_raw[step])).convert('RGB')))
        obs.append(img)
        actions.append([float(lin_vels[step]), float(ang_vels[step])])
    
    if len(obs) >= 10:
        trajectories.append({'observations': np.array(obs), 'actions': np.array(actions)})
    h.close()
    if fi % 50 == 0: print(f'{fi}/{len(files)} — {len(trajectories)} trajectories')

np.save(out_file, np.array(trajectories, dtype=object), allow_pickle=True)
print(f'Saved {len(trajectories)} trajectories → {out_file}')
