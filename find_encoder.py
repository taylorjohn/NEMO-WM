import torch, pathlib, os

# Search all .pt files for the one with StudentEncoder architecture
for pt in pathlib.Path('./checkpoints').rglob('*.pt'):
    try:
        ckpt  = torch.load(pt, map_location='cpu', weights_only=True)
        keys  = list(ckpt.keys()) if isinstance(ckpt, dict) else []
        # Look for state dict with features/proj keys (StudentEncoder)
        state = ckpt
        if 'model' in ckpt: state = ckpt['model']
        if 'model_state_dict' in ckpt: state = ckpt['model_state_dict']
        if 'state_dict' in ckpt: state = ckpt['state_dict']
        skeys = list(state.keys()) if isinstance(state, dict) else []
        has_features = any('features' in k or 'block1' in k or 'proj' in k for k in skeys)
        size_kb = pt.stat().st_size // 1024
        print(f'{str(pt):70s}  {size_kb:6d}KB  keys={skeys[:3]}')
    except Exception as e:
        print(f'{str(pt):70s}  ERROR: {e}')
