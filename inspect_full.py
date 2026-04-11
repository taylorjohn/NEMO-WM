import torch
ckpt  = torch.load('./checkpoints/maze_weak_sigreg_straight/cortex_student_phase2_final.pt', map_location='cpu', weights_only=True)
state = ckpt['model']
print('ALL KEYS:')
for k, v in state.items():
    print(f'  {k:60s} {str(v.shape):30s} {v.dtype}')
