import json, torch, numpy as np, csv
from pathlib import Path
from train_cardiac_phase import CardiacPhaseDataset, PhaseHead, signal_to_frame, N_PHASES, FRAME_TRANSFORM
from student_encoder import StudentEncoder
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from PIL import Image

ckpt  = torch.load('./checkpoints/heart_phase/cardiac_phase_final.pt', map_location='cpu')
model = StudentEncoder(); head = PhaseHead()
state = ckpt['encoder']
stem_map = {"backbone.stem.0":"backbone.block1.0","backbone.stem.1":"backbone.block1.1",
            "backbone.stem.3":"backbone.block2.0","backbone.stem.4":"backbone.block2.1",
            "backbone.stem.6":"backbone.block3.0","backbone.stem.7":"backbone.block3.1",
            "backbone.stem.9":"backbone.block4.0","backbone.stem.10":"backbone.block4.1"}
state = {next((k.replace(op,np,1) for op,np in stem_map.items() if k.startswith(op+".")),k):v
         for k,v in state.items()}
model.load_state_dict(state, strict=False); head.load_state_dict(ckpt['head'])
model.eval(); head.eval()

# Load all patients, 5 windows per phase per patient = 20 windows max, balanced
patients = {}
with open('./heart_data/training_data.csv', newline='') as f:
    for row in csv.DictReader(f):
        pid = row.get('Patient ID','').strip()
        patients[pid] = row.get('Murmur','Unknown').strip()

data_dir = Path('./heart_data/training_data')
results = []
print('Evaluating...')

for pid, murmur in patients.items():
    if murmur == 'Unknown': continue
    label = 1 if murmur == 'Present' else 0
    wavs = sorted(data_dir.glob(f'{pid}_*.wav'))
    entropies = []
    for wav_path in wavs[:2]:  # max 2 locations per patient
        tsv_path = wav_path.with_suffix('.tsv')
        if not tsv_path.exists(): continue
        import wave
        with wave.open(str(wav_path),'rb') as wf:
            n = wf.getnframes()
        # Sample 10 windows evenly across recording
        for start in np.linspace(0, max(0,n-1000), 10, dtype=int):
            import numpy as np2
            with wave.open(str(wav_path),'rb') as wf:
                sw = wf.getsampwidth(); wf.setpos(int(start))
                raw = wf.readframes(1000)
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)/32768.0
            if len(audio) < 1000: audio = np.pad(audio,(0,1000-len(audio)))
            frame = signal_to_frame(audio)
            t = FRAME_TRANSFORM(Image.fromarray(frame)).unsqueeze(0)
            with torch.no_grad():
                z = model(t); logits = head(z)
                probs = torch.softmax(logits,dim=-1)
                ent = -(probs * torch.log(probs+1e-8)).sum().item()
            entropies.append(ent)
    if entropies:
        results.append({'label':label,'entropy':np.mean(entropies),'murmur':murmur})

labels  = [r['label']   for r in results]
entropies = [r['entropy'] for r in results]
auroc = roc_auc_score(labels, entropies)
mn = np.mean([r['entropy'] for r in results if r['label']==0])
mm = np.mean([r['entropy'] for r in results if r['label']==1])
print(f'Patients: {len(results)} ({sum(l==0 for l in labels)} normal, {sum(l==1 for l in labels)} murmur)')
print(f'Normal entropy:  {mn:.4f}')
print(f'Murmur entropy:  {mm:.4f}')
print(f'AUROC: {auroc:.4f}')


