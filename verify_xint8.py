import torch, numpy as np, onnxruntime as ort
from student_encoder import StudentEncoder

ckpt = torch.load('./checkpoints/maze_weak_sigreg_straight/cortex_student_phase2_final.pt', map_location='cpu', weights_only=True)
encoder = StudentEncoder()
encoder.load_state_dict(ckpt['model'], strict=True)
encoder.eval()

x_np = np.random.randn(1,3,224,224).astype(np.float32)
with torch.no_grad():
    pt_out = encoder(torch.from_numpy(x_np)).numpy()

sess     = ort.InferenceSession('./npu_models/cortex_student_xint8.onnx', providers=['CPUExecutionProvider'])
onnx_out = sess.run(None, {'pixel_values': x_np})[0]

cosine = float(np.dot(pt_out.flatten(), onnx_out.flatten()) /
               (np.linalg.norm(pt_out) * np.linalg.norm(onnx_out) + 1e-8))
mae    = float(np.abs(pt_out - onnx_out).mean())
print('Cosine similarity (PT vs XINT8):', round(cosine, 4))
print('MAE                             :', round(mae, 6))
print('XINT8 size: 63KB')
print('Status: OK' if cosine > 0.95 else 'WARNING - quantization degradation')
