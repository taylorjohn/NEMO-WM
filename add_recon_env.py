src = open('train_predictor.py', encoding='utf-8').read()
src2 = src.replace(old, new)
open('train_predictor.py', 'w', encoding='utf-8').write(src2)
print('recon added:', 'recon' in src2)
