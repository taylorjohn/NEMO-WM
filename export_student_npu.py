"""
export_student_npu.py - CORTEX-PE v16.15
Correct export using student_encoder.py architecture.
"""
import torch, hashlib, json, datetime, onnx
from pathlib import Path
from student_encoder import StudentEncoder, StudentEncoderONNX

CKPT = Path('./checkpoints/maze_weak_sigreg_straight/cortex_student_phase2_final.pt')
OUT  = Path('./npu_models')
OUT.mkdir(exist_ok=True)

# Load checkpoint
ckpt  = torch.load(CKPT, map_location='cpu', weights_only=True)
state = ckpt['model']  # backbone.block1... keys

encoder = StudentEncoder()
missing, unexpected = encoder.load_state_dict(state, strict=True)
print(f'Missing : {missing}')
print(f'Unexpected: {unexpected[:3]}')
encoder.eval()

# Verify output
with torch.no_grad():
    dummy = torch.randn(1, 3, 224, 224)
    z = encoder(dummy)
print(f'Output shape : {z.shape}')
print(f'L2 norm      : {z.norm(dim=-1).item():.6f}  (should be ~1.0)')

# Export ONNX
onnx_model = StudentEncoderONNX(encoder)
fp32_path  = OUT / 'cortex_student_fp32.onnx'

torch.onnx.export(
    onnx_model, (dummy,), str(fp32_path),
    opset_version=17,
    do_constant_folding=True,
    input_names=['pixel_values'],
    output_names=['latent'],
    dynamic_axes=None,
)
onnx.checker.check_model(str(fp32_path))
print(f'FP32 ONNX: {fp32_path}  ({fp32_path.stat().st_size/1024:.1f}KB)')

# SHA256
h = hashlib.sha256()
CKPT.read_bytes()
for chunk in iter(lambda: open(CKPT,'rb').read(8192), b''):
    h.update(chunk)
sha = h.hexdigest()

# Embed metadata
model = onnx.load(str(fp32_path))
for k,v in {
    'cortex_version': 'v16.15',
    'checkpoint_sha256': sha,
    'export_timestamp': datetime.datetime.now().isoformat(),
    'architecture': 'CortexCNNBackbone+ShatteredLatentHead',
    'output': 'L2-normalized 128-D latent',
}.items():
    e = model.metadata_props.add(); e.key=k; e.value=v
onnx.save(model, str(fp32_path))

print(f'SHA256: {sha[:32]}...')
print('FP32 export complete. Now quantize with:')
print('python export_student_npu.py --quantize')
