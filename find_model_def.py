import pathlib
# Search for the training script that defines this model
for py in pathlib.Path('.').rglob('*.py'):
    try:
        txt = py.read_text(encoding='utf-8', errors='ignore')
        if 'backbone.block' in txt and 'heads' in txt and 'class' in txt:
            print(f'FOUND: {py}')
            # Print the class definition
            lines = txt.splitlines()
            for i,l in enumerate(lines):
                if 'class' in l and ('Student' in l or 'Encoder' in l or 'Backbone' in l or 'Model' in l or 'Cortex' in l):
                    print(f'  Line {i}: {l}')
    except: pass
