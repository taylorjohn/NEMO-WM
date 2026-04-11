import json, torch, numpy as np, csv, wave
from pathlib import Path
from PIL import Image
from scipy.signal import stft
from scipy.ndimage import median_filter
from torchvision import transforms
from torch import nn
from student_encoder import StudentEncoder
from sklearn.metrics import roc_auc_score

PCG_FS=4000; WIN=1000; HOP=500; FRAME=224
T = transforms.Compose([transforms.Resize(FRAME), transforms.CenterCrop(FRAME),
    transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

def wav_to_frame(seg):
    _,_,Z = stft(seg,fs=PCG_FS,nperseg=256,noverlap=192,nfft=256)
    S=np.log1p(np.abs(Z)**2); h=median_filter(S,(1,15)); p=median_filter(S,(15,1))
    def _img(a):
        mn,mx=a.min(),a.max()
        if mx>mn: a=(a-mn)/(mx-mn)
        return np.array(Image.fromarray((a*255).astype('uint8'),'L').resize((FRAME,FRAME)))
    return np.stack([_img(S),_img(h),_img(p)],axis=-1)

ckpt = torch.load('./checkpoints/heart_phase/cardiac_phase2_final.pt', map_location='cpu')
model = StudentEncoder(); model.load_state_dict(ckpt['encoder'])
trans_head = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Linear(64,128))
trans_head.load_state_dict(ckpt['trans_head'])
model.eval(); trans_head.eval()

patients = {}
with open('./heart_data/training_data.csv') as f:
    for row in csv.DictReader(f):
        pid=row.get('Patient ID','').strip()
        patients[pid] = row.get('Murmur','Unknown').strip()

results = []
data_dir = Path('./heart_data/training_data')
print('Evaluating transition prediction error per patient...')
for pid, murmur in list(patients.items())[:400]:
    if murmur == 'Unknown': continue
    label = 1 if murmur=='Present' else 0
    errors = []
    for wav_path in sorted(data_dir.glob(f'{pid}_*.wav'))[:2]:
        try:
            with wave.open(str(wav_path),'rb') as wf:
                raw=wf.readframes(wf.getnframes())
            audio=np.frombuffer(raw,dtype=np.int16).astype(np.float32)/32768.0
            audio=audio/(np.abs(audio).max()+1e-8)
            for start in range(0,min(len(audio)-WIN*2,WIN*20),HOP*2):
                f0=T(Image.fromarray(wav_to_frame(audio[start:start+WIN]))).unsqueeze(0)
                f1=T(Image.fromarray(wav_to_frame(audio[start+HOP:start+HOP+WIN]))).unsqueeze(0)
                with torch.no_grad():
                    z0=model(f0); z1=model(f1)
                    z1_pred=trans_head(z0)
                    err=float(((z1_pred-z1)**2).mean())
                errors.append(err)
        except: pass
    if errors:
        results.append({'label':label,'error':np.mean(errors),'murmur':murmur})

labels=[r['label'] for r in results]; scores=[r['error'] for r in results]
mn=np.mean([r['error'] for r in results if r['label']==0])
mm=np.mean([r['error'] for r in results if r['label']==1])
auroc=roc_auc_score(labels,scores)
print(f'Patients: {len(results)} ({sum(l==0 for l in labels)} normal, {sum(l==1 for l in labels)} murmur)')
print(f'Normal transition error:  {mn:.5f}')
print(f'Murmur transition error:  {mm:.5f}')
print(f'AUROC: {auroc:.4f}')
json.dump({'auroc':round(auroc,4),'normal_err':round(mn,5),'murmur_err':round(mm,5)},
          open('./results/heart_phase2_auroc.json','w'),indent=2)
