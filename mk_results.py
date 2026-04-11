import json, pathlib
pathlib.Path('./results').mkdir(exist_ok=True)
data = {
    'cross_domain_baseline': {
        'bearing_encoder_on_heart': {
            'auroc': 0.5082,
            'normal_norm': 7.380,
            'murmur_norm': 7.297,
            'separation_pct': 1.1,
            'note': 'Near-random. Domain-specific training required.'
        }
    }
}
json.dump(data, open('./results/heart_cross_domain.json','w'), indent=2)
print('saved')
