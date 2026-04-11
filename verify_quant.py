import torch, torch.nn as nn, onnxruntime as ort, numpy as np

class StudentEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,stride=2,padding=1), nn.GELU(),
            nn.Conv2d(32,64,3,stride=2,padding=1), nn.GELU(),
            nn.Conv2d(64,128,3,stride=2,padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((4,4)),
        )
        self.proj = nn.Sequential(
            nn.Linear(2048,512), nn.GELU(), nn.Linear(512,128)
        )
    def forward(self,x): return self.proj(self.features(x).flatten(1))

ckpt  = torch.load('./checkpoints/maze_weak_sigreg_straight/cortex_student_phase2_final.pt', map_location='cpu', weights_only=True)
state = ckpt.get('model', ckpt)
state = {k.replace('encoder.','',1):v for k,v in state.items() if not k.startswith('projector')}
model = StudentEncoder()
model.load_state_dict(state, strict=False)
model.eval()

x_np = np.random.randn(1,3,224,224).astype(np.float32)
x_pt = torch.from_numpy(x_np)

with torch.no_grad():
    pt_out = model(x_pt).numpy()

sess     = ort.InferenceSession('./npu_models/cortex_student_xint8.onnx', providers=['CPUExecutionProvider'])
onnx_out = sess.run(None, {'pixel_values': x_np})[0]

cosine = float(np.dot(pt_out.flatten(), onnx_out.flatten()) /
               (np.linalg.norm(pt_out) * np.linalg.norm(onnx_out) + 1e-8))
mae    = float(np.abs(pt_out - onnx_out).mean())

print('Cosine similarity (PT vs XINT8):', round(cosine, 4))
print('MAE                             :', round(mae, 6))
print('Status: OK' if cosine > 0.95 else 'Status: WARNING - quantization error too high')
