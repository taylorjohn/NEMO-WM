import torch, pathlib

for path_str in [
    './checkpoints/recon_transition/transition_best.pt',
    './checkpoints/cardiac/student_best.pt',
]:
    p = pathlib.Path(path_str)
    if not p.exists():
        print(f'NOT FOUND: {path_str}')
        continue
    ckpt = torch.load(p, map_location='cpu', weights_only=True)
    print(f'\n=== {p.name} ===')
    print(f'Top keys: {list(ckpt.keys())}')
    state = ckpt
    for k in ['model','model_state_dict','state_dict','student','predictor','net']:
        if k in ckpt:
            state = ckpt[k]
            print(f'State key: {k}')
            break
    if isinstance(state, dict):
        for k,v in list(state.items())[:8]:
            print(f'  {k:50s} {str(v.shape):20s} {v.dtype}')
        print(f'  ... ({len(state)} total keys)')
