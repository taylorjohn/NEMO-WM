import torch, torch.nn as nn, torchaudio
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import sys, json

# ── Config ────────────────────────────────────────────────────
CSV       = './heart_data/training_data.csv'
DATA_DIR  = './heart_data/training_data'
BEATS_PT  = './BEATs_iter3_plus_AS2M.pt'
OUT_DIR   = './checkpoints/beats_cardiac'
SR        = 4000
WIN_SEC   = 3
EPOCHS    = 20
LR        = 1e-3
BATCH     = 32
LABEL_FRAC = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
print(f'Label fraction: {LABEL_FRAC}')

# ── Dataset ───────────────────────────────────────────────────
class CirCorDataset(Dataset):
    def __init__(self, csv_path, data_dir, label_frac=1.0, split='train'):
        df = pd.read_csv(csv_path)
        df = df[df['Murmur'].isin(['Present', 'Absent'])].reset_index(drop=True)
        # Patient-level split 80/20
        pids = df['Patient ID'].unique()
        np.random.seed(42)
        np.random.shuffle(pids)
        n_train = int(len(pids) * 0.8)
        train_pids = set(pids[:n_train])
        val_pids   = set(pids[n_train:])
        mask = df['Patient ID'].isin(train_pids if split=='train' else val_pids)
        df = df[mask]
        if split == 'train' and label_frac < 1.0:
            df = df.sample(frac=label_frac, random_state=42)
        self.records = []
        n = SR * WIN_SEC
        for _, row in df.iterrows():
            pid   = str(row['Patient ID'])
            label = 1 if row['Murmur'] == 'Present' else 0
            for wav_path in Path(data_dir).glob(f'{pid}_*.wav'):
                import soundfile as sf; data, sr = sf.read(str(wav_path)); import torch; waveform = torch.tensor(data, dtype=torch.float32).unsqueeze(0) if data.ndim==1 else torch.tensor(data.T, dtype=torch.float32)
                if sr != SR:
                    waveform = torchaudio.functional.resample(waveform, sr, SR)
                waveform = waveform.mean(0)  # mono
                # Slice into WIN_SEC windows
                for start in range(0, waveform.shape[0] - n, n // 2):
                    self.records.append((waveform[start:start+n].clone(), label))
        print(f'  {split}: {len(self.records)} windows, label_frac={label_frac}')

    def __len__(self): return len(self.records)
    def __getitem__(self, i):
        wav, label = self.records[i]
        return wav, label

# ── BEATs wrapper ─────────────────────────────────────────────
# BEATs uses custom loader — load checkpoint and use as feature extractor
ckpt = torch.load(BEATS_PT, map_location='cpu')
# BEATs checkpoint stores cfg and model weights
# Use simple approach: load via fairseq if available, else use raw features
try:
    from BEATs import BEATs, BEATsConfig
    cfg = BEATsConfig(ckpt['cfg'])
    beats = BEATs(cfg)
    beats.load_state_dict(ckpt['model'])
    beats.eval()
    FEAT_DIM = 768
    USE_BEATS = True
    print('BEATs loaded via BEATsConfig')
except Exception as e:
    print(f'BEATs custom loader failed ({e}), falling back to mel+CNN')
    USE_BEATS = False
    FEAT_DIM  = 128

# ── Model ─────────────────────────────────────────────────────
class CardiacClassifier(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
    def forward(self, x): return self.head(x)

model = CardiacClassifier(FEAT_DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

train_ds = CirCorDataset(CSV, DATA_DIR, label_frac=LABEL_FRAC, split='train')
val_ds   = CirCorDataset(CSV, DATA_DIR, label_frac=1.0,         split='val')
train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)

def extract_features(wavs):
    if USE_BEATS:
        with torch.no_grad():
            padding_mask = torch.zeros(wavs.shape[0], wavs.shape[1]).bool()
            feats, _ = beats.extract_features(wavs, padding_mask=padding_mask)
            return feats.mean(1)  # (B, 768) mean pool over time
    else:
        # Fallback: simple mel features
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=512, hop_length=128, n_mels=128)
        m = mel(wavs).mean(-1)  # (B, 128)
        return m

best_auroc = 0.0
results = []
for epoch in range(EPOCHS):
    model.train()
    for wavs, labels in train_dl:
        feats = extract_features(wavs)
        logits = model(feats)
        loss = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0]))(logits, labels)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    # Validation
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for wavs, labels in val_dl:
            feats  = extract_features(wavs)
            probs  = torch.softmax(model(feats), dim=-1)[:, 1]
            all_scores.extend(probs.tolist())
            all_labels.extend(labels.tolist())
    auroc = roc_auc_score(all_labels, all_scores)
    results.append({'epoch': epoch+1, 'auroc': auroc})
    scheduler.step()
    print(f'Epoch {epoch+1:>3}/{EPOCHS} | AUROC={auroc:.4f} | lr={scheduler.get_last_lr()[0]:.2e}')
    if auroc > best_auroc:
        best_auroc = auroc
        torch.save({'model': model.state_dict(), 'feat_dim': FEAT_DIM,
                    'label_frac': LABEL_FRAC, 'auroc': auroc},
                   f'{OUT_DIR}/beats_head_frac{LABEL_FRAC:.2f}.pt')

print(f'Best AUROC: {best_auroc:.4f} at label_frac={LABEL_FRAC}')
json.dump({'label_frac': LABEL_FRAC, 'best_auroc': best_auroc, 'history': results},
          open(f'{OUT_DIR}/results_frac{LABEL_FRAC:.2f}.json', 'w'), indent=2)
