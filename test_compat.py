import torch, re

def load_encoder_compat(path):
    """Load encoder checkpoint regardless of old/new key naming."""
    from student_encoder import StudentEncoder
    ckpt = torch.load(path, map_location='cpu')
    state = ckpt['model'] if 'model' in ckpt else ckpt
    model = StudentEncoder()

    # Try strict load first
    try:
        model.load_state_dict(state, strict=True)
        return model
    except RuntimeError:
        pass

    # Map old flat stem keys to new block keys
    # Old: backbone.stem.0  → block1 (conv)
    #      backbone.stem.1  → block1 (bn)
    #      backbone.stem.3  → block2 (conv)  [after ReLU at index 2]
    #      backbone.stem.4  → block2 (bn)
    #      backbone.stem.6  → block3 (conv)
    #      backbone.stem.7  → block3 (bn)
    #      backbone.stem.9  → block4 (conv)
    #      backbone.stem.10 → block4 (bn)
    stem_to_block = {
        'backbone.stem.0':  'backbone.block1.0',
        'backbone.stem.1':  'backbone.block1.1',
        'backbone.stem.3':  'backbone.block2.0',
        'backbone.stem.4':  'backbone.block2.1',
        'backbone.stem.6':  'backbone.block3.0',
        'backbone.stem.7':  'backbone.block3.1',
        'backbone.stem.9':  'backbone.block4.0',
        'backbone.stem.10': 'backbone.block4.1',
    }
    new_state = {}
    for k, v in state.items():
        new_k = k
        for old_prefix, new_prefix in stem_to_block.items():
            if k.startswith(old_prefix + '.') or k == old_prefix:
                new_k = k.replace(old_prefix, new_prefix, 1)
                break
        new_state[new_k] = v

    # Load with strict=False to handle any remaining mismatches
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print(f'  Warning: missing keys: {missing[:3]}...')
    if unexpected:
        print(f'  Warning: unexpected keys: {unexpected[:3]}...')
    return model

# Test
m = load_encoder_compat('./checkpoints/pusht/cortex_student_flow_final.pt')
import torch
z = m(torch.randn(1,3,224,224))
print(f'OK — z shape: {z.shape}')
