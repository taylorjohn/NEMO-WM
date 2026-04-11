import shutil, re
from pathlib import Path
src = Path('train_distillation.py').read_text(encoding='utf-8', errors='replace')

# Fix 1: add --gc to parse_args
src = src.replace('    return p.parse_args()', '    p.add_argument("--gc", default="none", choices=["none","standard","gcc2","moment"])\n    return p.parse_args()', 1)

# Fix 2: add gc param to train_phase1 signature - find and fix double comma
src = src.replace('    output_dir=".",\n):', '    output_dir=".",\n    gc="none",\n):')

# Fix 3: pass gc to train_phase1 call
src = src.replace('            output_dir    = args.out,\n        )', '            output_dir    = args.out,\n            gc            = args.gc,\n        )')

# Fix 4: inject GC wrapper after Phase 1 scheduler line
anchor = '    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)'
wrapper = '''
    if gc != 'none':
        try:
            from gc_optimizer import GCAdamW as _GC
            optimizer = _GC(optimizer.param_groups, lr=optimizer.defaults.get('lr',1e-3), weight_decay=optimizer.defaults.get('weight_decay',0), use_gc=(gc in ('standard','gcc2')), use_mc=(gc=='moment'))
            print(f'  Optimizer: GCAdamW gc={gc}')
        except Exception as e:
            print(f'  GC failed {e}, using AdamW')
    else:
        print('  Optimizer: AdamW baseline')
'''
src = src.replace(anchor, wrapper + anchor, 1)

Path('train_distillation.py').write_text(src, encoding='utf-8')
print('Done. Checks:')
print('--gc in file:', '--gc' in src)
print('gc param in sig:', 'gc="none"' in src)
print('GC wrapper in body:', 'gc_optimizer' in src)
