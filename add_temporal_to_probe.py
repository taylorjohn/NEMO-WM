# add_temporal_to_probe.py - adds D_temporal_split checkpoint to final_aim_probe.py
with open('final_aim_probe.py') as f:
    code = f.read()

old = '    "C_hybrid":            "checkpoints/cwm/vit_tiny_hybrid.pt",'
new = ('    "C_hybrid":            "checkpoints/cwm/vit_tiny_hybrid.pt",\n'
       '    "D_temporal_split":    "checkpoints/cwm/vit_tiny_temporal_split.pt",')

if 'D_temporal_split' in code:
    print('Already added')
elif old in code:
    code = code.replace(old, new)
    with open('final_aim_probe.py', 'w') as f:
        f.write(code)
    print('Added D_temporal_split to probe')
else:
    print('ERROR: anchor string not found -- check final_aim_probe.py')
