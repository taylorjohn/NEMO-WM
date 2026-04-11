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

print(f'Encoding {len(meta)} frames...')
with torch.no_grad():
    for i, m in enumerate(meta):
        img = T(Image.open(frame_dir / m['frame'])).unsqueeze(0)
        z   = model(img).squeeze(0).numpy()
        results.append({'z': z, 'label': m['label'],
                        'fault_type': m['fault_type'], 'diameter': m['diameter']})
        if i % 500 == 0:
            print(f'  {i}/{len(meta)}')

z_healthy = np.stack([r['z'] for r in results if r['label']==0]).mean(axis=0)
z_healthy /= np.linalg.norm(z_healthy)
for r in results:
    r['dist'] = float(np.linalg.norm(r['z'] - z_healthy))

labels = [r['label'] for r in results]
dists  = [r['dist']  for r in results]
print(f'AUROC overall: {roc_auc_score(labels, dists):.4f}')

for ft in ['ball', 'inner_race', 'outer_race']:
    sub = [r for r in results if r['fault_type'] in ['normal', ft]]
    l   = [r['label'] for r in sub]
    d   = [r['dist']  for r in sub]
    print(f'  {ft}: AUROC={roc_auc_score(l, d):.4f}')

print('By diameter:')
for diam in [0.007, 0.014, 0.021]:
    sub = [r for r in results if r['label']==0 or r['diameter']==diam]
    l   = [r['label'] for r in sub]
    d   = [r['dist']  for r in sub]
    mh  = np.mean([r['dist'] for r in sub if r['label']==0])
    mf  = np.mean([r['dist'] for r in sub if r['label']==1])
    print(f'  {diam}: AUROC={roc_auc_score(l,d):.4f}  healthy={mh:.3f}  fault={mf:.3f}')
