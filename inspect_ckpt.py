import torch
ckpt = torch.load('./checkpoints/maze_weak_sigreg_straight/cortex_student_phase2_final.pt', map_location='cpu', weights_only=True)
print('Top-level keys:', list(ckpt.keys()))
if 'model' in ckpt:
    m = ckpt['model']
    if isinstance(m, dict):
        keys = list(m.keys())
        print('model keys (first 10):', keys[:10])
        print('model keys (last 5):', keys[-5:])
        for k,v in list(m.items())[:3]:
            print(f'  {k}: {v.shape}')
    else:
        print('model type:', type(m))
if 'step' in ckpt:
    print('step:', ckpt['step'])
