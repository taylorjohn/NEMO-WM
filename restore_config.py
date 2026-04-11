src = open('train_predictor.py', encoding='utf-8').read()
src = src.replace(
    '"umaze":  {"n_train": 200,   "steps": 100, "epochs": 2,  "action_dim": 2}',
    '"umaze":  {"n_train": 2000,  "steps": 100, "epochs": 20, "action_dim": 2}'
)
open('train_predictor.py', 'w', encoding='utf-8').write(src)
print('restored')
