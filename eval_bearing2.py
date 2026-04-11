import numpy as np, json, torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from student_encoder import StudentEncoder
from sklearn.metrics import roc_auc_score

T = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

ckpt  = torch.load('./checkpoints/bearing/cortex_student_phase2_final.pt', map_location='cpu')
model = StudentEncoder()
model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
model.eval()

meta      = json.load(open('./bearing_data/cwru_all/metadata.json'))
frame_dir = Path('./bearing_data/cwru_all')
results   = []

with torch.no_grad():
    for i, m in enumerate(meta):
        img  = T(Image.open(frame_dir / m['frame'])).unsqueeze(0)
        z    = model(img).squeeze(0).numpy()
        norm = float(np.linalg.norm(z))
        results.append({'z': z, 'norm': norm, 'label': m['label'],
                        'fault_type': m['fault_type'], 'diameter': m['diameter']})

labels = [r['label'] for r in results]
norms  = [r['norm']  for r in results]

# Healthy frames have HIGHER norm — use negative norm as anomaly score
neg_norms = [-n for n in norms]

print(f'Healthy norm mean: {np.mean([r["norm"] for r in results if r["label"]==0]):.3f}')
print(f'Fault   norm mean: {np.mean([r["norm"] for r in results if r["label"]==1]):.3f}')
print()
print(f'AUROC (norm-based):     {roc_auc_score(labels, neg_norms):.4f}')

for ft in ['ball', 'inner_race', 'outer_race']:
    sub = [r for r in results if r['fault_type'] in ['normal', ft]]
    l   = [r['label'] for r in sub]
    s   = [-r['norm'] for r in sub]
    print(f'  {ft}: AUROC={roc_auc_score(l, s):.4f}')

print('By diameter:')
for diam in [0.007, 0.014, 0.021]:
    sub = [r for r in results if r['label']==0 or r['diameter']==diam]
    l   = [r['label'] for r in sub]
    s   = [-r['norm'] for r in sub]
    mh  = np.mean([r['norm'] for r in sub if r['label']==0])
    mf  = np.mean([r['norm'] for r in sub if r['label']==1])
    print(f'  {diam}: AUROC={roc_auc_score(l,s):.4f}  healthy_norm={mh:.3f}  fault_norm={mf:.3f}')
