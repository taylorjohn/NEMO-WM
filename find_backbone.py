import pathlib
# Search all py files for the backbone/stem class
for py in sorted(pathlib.Path('.').rglob('*.py')):
    try:
        txt = py.read_text(encoding='utf-8', errors='ignore')
        if 'backbone' in txt and 'stem' in txt and 'block' in txt and 'class' in txt:
            print(f'\n=== {py} ===')
            lines = txt.splitlines()
            for i, l in enumerate(lines):
                if any(kw in l for kw in ['class ', 'def forward', 'block', 'stem', 'AdaptiveAvg', 'flatten', 'cat']):
                    print(f'  {i:4d}: {l}')
    except: pass
