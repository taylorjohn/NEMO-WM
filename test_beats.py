import torch
from BEATs import BEATs, BEATsConfig

checkpoint = torch.load('./BEATs_iter3_plus_AS2M.pt', map_location='cpu')
cfg = BEATsConfig(checkpoint['cfg'])
model = BEATs(cfg)
model.load_state_dict(checkpoint['model'])
model.eval()

audio = torch.randn(1, 16000)
padding_mask = torch.zeros(1, 16000).bool()

with torch.no_grad():
    out = model.extract_features(audio, padding_mask=padding_mask)
    print(f'outputs: {len(out)}')
    for i, r in enumerate(out):
        if hasattr(r, 'shape'):
            print(f'  [{i}] shape={r.shape}')
        else:
            print(f'  [{i}] type={type(r)}')
