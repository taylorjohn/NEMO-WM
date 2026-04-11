import json, torch, numpy as np, csv, wave, time
from pathlib import Path
from PIL import Image
from scipy.signal import stft
from scipy.ndimage import median_filter
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
from student_encoder import StudentEncoder

# Load normal patient WAV files
PCG_FS = 4000; WIN = 1000; HOP = 500; FRAME = 224
T = transforms.Compose([transforms.Resize(FRAME), transforms.CenterCrop(FRAME),
    transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

def wav_to_frame(seg):
    _,_,Z = stft(seg, fs=PCG_FS, nperseg=256, noverlap=192, nfft=256)
    S = np.log1p(np.abs(Z)**2)
    h = median_filter(S,(1,15)); p = median_filter(S,(15,1))
    def _img(a):
        mn,mx=a.min(),a.max()
        if mx>mn: a=(a-mn)/(mx-mn)
        return np.array(Image.fromarray((a*255).astype('uint8'),'L').resize((FRAME,FRAME)))
    return np.stack([_img(S),_img(h),_img(p)],axis=-1)

patients = {}
with open('./heart_data/training_data.csv') as f:
    for row in csv.DictReader(f):
        pid = row.get('Patient ID','').strip()
        patients[pid] = row.get('Murmur','Unknown').strip()
normal_pids = [p for p,m in patients.items() if m=='Absent']
print(f'Normal patients: {len(normal_pids)}')

# Build consecutive pairs
pairs = []
data_dir = Path('./heart_data/training_data')
for pid in normal_pids[:300]:
    for wav_path in sorted(data_dir.glob(f'{pid}_*.wav'))[:1]:
        try:
            with wave.open(str(wav_path),'rb') as wf:
                n=wf.getnframes(); sw=wf.getsampwidth()
                raw=wf.readframes(n)
            audio = np.frombuffer(raw,dtype=np.int16).astype(np.float32)/32768.0
            audio = audio/(np.abs(audio).max()+1e-8)
            for start in range(0, len(audio)-WIN*2, HOP):
                pairs.append((audio[start:start+WIN], audio[start+HOP:start+HOP+WIN]))
        except: pass
print(f'Pairs: {len(pairs)}')

# Load cardiac phase encoder
ckpt = torch.load('./checkpoints/heart_phase/cardiac_phase_final.pt', map_location='cpu')
model = StudentEncoder()
model.load_state_dict(ckpt['encoder'])
# Add transition predictor head
trans_head = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Linear(64,128))
opt = torch.optim.Adam(list(model.parameters())+list(trans_head.parameters()), lr=3e-4)

print('Training transition predictor (10 epochs)...')
idx = list(range(len(pairs))); 
for epoch in range(10):
    np.random.shuffle(idx)
    total_loss = 0; n = 0
    for i in idx[:5000]:
        s0, s1 = pairs[i]
        f0 = T(Image.fromarray(wav_to_frame(s0))).unsqueeze(0)
        f1 = T(Image.fromarray(wav_to_frame(s1))).unsqueeze(0)
        with torch.no_grad(): z1 = model(f1)
        z0 = model(f0)
        z1_pred = trans_head(z0)
        loss = F.mse_loss(z1_pred, z1.detach())
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item(); n += 1
    print(f'Epoch {epoch+1}/10 | loss={total_loss/n:.4f}')

torch.save({'encoder': model.state_dict(), 'trans_head': trans_head.state_dict()},
           './checkpoints/heart_phase/cardiac_phase2_final.pt')
print('Saved: ./checkpoints/heart_phase/cardiac_phase2_final.pt')
