from quark.onnx import ModelQuantizer, QConfig
import numpy as np, onnxruntime as ort, torch, torch.nn as nn
import scipy.io.wavfile as wavfile, pathlib

# Load real cardiac WAV files for calibration
calib_clips = []
TARGET_SR = 2000
CLIP_LEN  = 2000
for wav in list(pathlib.Path('./cardiac_data').rglob('*.wav'))[:200]:
    try:
        sr, sig = wavfile.read(str(wav))
        sig = sig.astype(np.float32)
        if sig.ndim > 1: sig = sig[:,0]
        if sr != TARGET_SR:
            n = int(len(sig)*TARGET_SR/sr)
            sig = np.interp(np.linspace(0,len(sig)-1,n),np.arange(len(sig)),sig).astype(np.float32)
        peak = np.abs(sig).max()
        if peak > 1e-6: sig /= peak
        mid = len(sig)//2
        clip = sig[max(0,mid-CLIP_LEN//2):max(0,mid-CLIP_LEN//2)+CLIP_LEN]
        if len(clip) < CLIP_LEN: clip = np.pad(clip,(0,CLIP_LEN-len(clip)))
        calib_clips.append(clip.astype(np.float32))
    except: pass

print(f'Loaded {len(calib_clips)} real cardiac clips for calibration')

class CalibReader:
    def __init__(self):
        self.data = [{'audio': c.reshape(1,1,-1)} for c in calib_clips]
        self.idx  = 0
    def get_next(self):
        if self.idx >= len(self.data): return None
        r = self.data[self.idx]; self.idx += 1; return r

config    = QConfig.get_default_config('XINT8')
quantizer = ModelQuantizer(config)
quantizer.quantize_model(
    './npu_models/cardiac_student_fp32.onnx',
    './npu_models/cardiac_student_xint8.onnx',
    CalibReader()
)

# Verify cosine
class CardiacStudentEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv1d(1,32,10,stride=5,padding=2),torch.nn.GELU(),
            torch.nn.Conv1d(32,64,3,stride=2,padding=1),torch.nn.GELU(),
            torch.nn.Conv1d(64,128,3,stride=2,padding=1),torch.nn.GELU(),
            torch.nn.Conv1d(128,256,3,stride=2,padding=1),torch.nn.GELU(),
            torch.nn.AdaptiveAvgPool1d(1),
        )
        self.proj = torch.nn.Sequential(torch.nn.Linear(256,768),torch.nn.LayerNorm(768))
    def forward(self,x): return self.proj(self.encoder(x).squeeze(-1))

import torch
ckpt  = torch.load('./checkpoints/cardiac/student_best.pt',map_location='cpu',weights_only=True)
model = CardiacStudentEncoder()
model.load_state_dict(ckpt['model_state_dict'],strict=True)
model.eval()

x_np = calib_clips[0].reshape(1,1,-1)
with torch.no_grad():
    pt_out = model(torch.from_numpy(x_np)).numpy().flatten()
sess     = ort.InferenceSession('./npu_models/cardiac_student_xint8.onnx',providers=['CPUExecutionProvider'])
onnx_out = sess.run(None,{'audio':x_np})[0].flatten()
cos = float(np.dot(pt_out,onnx_out)/(np.linalg.norm(pt_out)*np.linalg.norm(onnx_out)+1e-8))
print(f'Cosine XINT8 vs PT: {cos:.4f}  {"OK" if cos>0.95 else "WARNING"}')
