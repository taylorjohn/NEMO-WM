import h5py, numpy as np, json, io
from pathlib import Path
from PIL import Image

recon_dir = Path('./recon_data/recon_release')
out_dir   = Path('./phase2_recon_frames')
out_dir.mkdir(exist_ok=True)

files = sorted(recon_dir.glob('*.hdf5'))
print(f'HDF5 files: {len(files)}')

# Use 200 files for Phase 2 training (~14,000 frames)
N_FILES   = 200
files     = files[:N_FILES]

# Compute action scale from first 50 files
all_lin, all_ang = [], []
for f in files[:50]:
    h = h5py.File(str(f), 'r')
    all_lin.extend(h['commands/linear_velocity'][:].tolist())
    all_ang.extend(h['commands/angular_velocity'][:].tolist())
    h.close()

lin_scale = float(np.abs(all_lin).max())
ang_scale = float(np.abs(all_ang).max())
act_scale = max(lin_scale, ang_scale)
print(f'Action scale: lin={lin_scale:.3f} ang={ang_scale:.3f} max={act_scale:.3f}')

frame_idx = 0
meta = []

for f_idx, fpath in enumerate(files):
    h = h5py.File(str(fpath), 'r')
    
    imgs_raw  = h['images/rgb_left'][:]
    lin_vels  = h['commands/linear_velocity'][:]
    ang_vels  = h['commands/angular_velocity'][:]
    collisions = h['collision/any'][:]
    
    for step in range(len(imgs_raw)):
        # Skip collision frames
        if collisions[step]:
            continue
        
        # Decode JPEG
        img = Image.open(io.BytesIO(imgs_raw[step])).resize((224, 224), Image.BILINEAR)
        fname = f'frame_{frame_idx:07d}.png'
        img.save(out_dir / fname)
        
        # Normalise actions: [lin_vel, ang_vel]
        action = [
            float(lin_vels[step]) / (lin_scale + 1e-8),
            float(ang_vels[step]) / (ang_scale + 1e-8)
        ]
        meta.append({'frame': fname, 'action': action, 'file': fpath.name, 'step': step})
        frame_idx += 1
    
    h.close()
    if f_idx % 50 == 0:
        print(f'  {f_idx}/{N_FILES} files — {frame_idx} frames')

json.dump({'act_scale': act_scale, 'lin_scale': lin_scale, 'ang_scale': ang_scale,
           'frames': meta}, 
          open(out_dir / 'flow_meta.json', 'w'), indent=2)
print(f'Done — {frame_idx} frames saved to {out_dir}')
