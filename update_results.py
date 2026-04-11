import json
from pathlib import Path

Path('./results').mkdir(exist_ok=True)
r = json.load(open('./results/cortex_pe_results.json'))

# Add cross-domain baseline finding
r['results']['heart_sound_circor']['cross_domain_baseline'] = {
    'encoder': 'bearing_phase2',
    'auroc': 0.5082,
    'normal_norm_mean': 7.380,
    'murmur_norm_mean': 7.297,
    'norm_separation_pct': 1.1,
    'interpretation': 'Near-random — bearing encoder has no useful heart sound features. Domain-specific training required.'
}

json.dump(r, open('./results/cortex_pe_results.json', 'w'), indent=2)
print('ok')
