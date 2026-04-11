import scipy.io.wavfile as wf, pathlib, numpy as np
p = next(pathlib.Path('./mimii_data/fan/id_00/normal').glob('*.wav'))
print('File:', p.name)
sr, sig = wf.read(str(p))
print('SR:', sr, 'Shape:', sig.shape, 'Dtype:', sig.dtype)
print('Min/Max:', sig.min(), sig.max())
