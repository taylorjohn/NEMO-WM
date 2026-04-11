import torch, pathlib
p = pathlib.Path('./checkpoints/recon_contrastive/temporal_head_k7_best.pt')
ckpt = torch.load(p, map_location='cpu', weights_only=True)
print('Top keys:', list(ckpt.keys()))
state = ckpt.get('head', ckpt.get('model_state_dict', ckpt))
for k,v in state.items():
    print(f'  {k:50s} {str(v.shape):25s} {v.dtype}')
