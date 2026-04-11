# -*- coding: utf-8 -*-
"""
export_temporal_head.py - CORTEX-PE v16.15
Export TemporalHead (visual quasimetric) to XINT8 ONNX for NPU deployment.

Architecture (from checkpoint weight shapes):
    Linear(128->256) + BatchNorm1d(256) + ReLU + Linear(256->64)
    No LayerNorm -> any quantization format works
    Input:  (1, 128) student encoder latent
    Output: (1, 64)  temporal distance embedding
"""
import hashlib, datetime, json
import numpy as np, onnx, torch, torch.nn as nn
import onnxruntime as ort
from pathlib import Path
from quark.onnx import ModelQuantizer, QConfig

CKPT_PATH = Path('./checkpoints/recon_contrastive/temporal_head_k7_best.pt')
OUT_DIR   = Path('./npu_models')
OUT_DIR.mkdir(exist_ok=True)

FP32_PATH  = OUT_DIR / 'temporal_head_fp32.onnx'
XINT8_PATH = OUT_DIR / 'temporal_head_xint8.onnx'


class TemporalHead(nn.Module):
    """
    Visual quasimetric projection head.
    Input:  (B, 128) student latent
    Output: (B, 64)  temporal distance embedding
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
        )
    def forward(self, x): return self.net(x)


# Load checkpoint
ckpt  = torch.load(CKPT_PATH, map_location='cpu', weights_only=True)
state = ckpt['head']
model = TemporalHead()
missing, unexpected = model.load_state_dict(state, strict=True)
assert not missing,    f'Missing: {missing}'
assert not unexpected, f'Unexpected: {unexpected}'
model.eval()

n_params  = sum(p.numel() for p in model.parameters())
ckpt_hash = hashlib.sha256(CKPT_PATH.read_bytes()).hexdigest()
print(f'TemporalHead: {n_params:,} params')
print(f'SHA256: {ckpt_hash[:16]}...{ckpt_hash[-8:]}')

dummy = torch.zeros(1, 128)
with torch.no_grad():
    out = model(dummy)
print(f'Output shape: {tuple(out.shape)}')

# FP32 ONNX export
torch.onnx.export(
    model, (dummy,), str(FP32_PATH),
    opset_version=17, do_constant_folding=True,
    input_names=['latent'], output_names=['temporal_embedding'],
    dynamic_axes=None,
)
onnx.checker.check_model(str(FP32_PATH))
print(f'FP32: {FP32_PATH.name} ({FP32_PATH.stat().st_size/1024:.1f}KB)')

# FP32 cosine verify
with torch.no_grad():
    pt_out = model(dummy).numpy().flatten()
sess     = ort.InferenceSession(str(FP32_PATH), providers=['CPUExecutionProvider'])
onnx_out = sess.run(None, {'latent': dummy.numpy()})[0].flatten()
cos_fp32 = float(np.dot(pt_out, onnx_out) /
                 (np.linalg.norm(pt_out) * np.linalg.norm(onnx_out) + 1e-8))
print(f'Cosine FP32 vs PT: {cos_fp32:.6f}  {"OK" if cos_fp32 > 0.9999 else "WARNING"}')

# XINT8 quantization — synthetic calibration fine for simple Linear+BN
class CalibReader:
    def __init__(self, n=200):
        self.data = [{'latent': np.random.randn(1, 128).astype(np.float32)}
                     for _ in range(n)]
        self.idx = 0
    def get_next(self):
        if self.idx >= len(self.data): return None
        r = self.data[self.idx]; self.idx += 1; return r

print('Quantizing to XINT8...')
ModelQuantizer(QConfig.get_default_config('XINT8')).quantize_model(
    str(FP32_PATH), str(XINT8_PATH), CalibReader()
)
print(f'XINT8: {XINT8_PATH.name} ({XINT8_PATH.stat().st_size/1024:.1f}KB)')

# XINT8 cosine verify
sess2    = ort.InferenceSession(str(XINT8_PATH), providers=['CPUExecutionProvider'])
xint8_out = sess2.run(None, {'latent': dummy.numpy()})[0].flatten()
cos_xint8 = float(np.dot(pt_out, xint8_out) /
                  (np.linalg.norm(pt_out) * np.linalg.norm(xint8_out) + 1e-8))
status = 'OK' if cos_xint8 > 0.95 else 'WARNING'
print(f'Cosine XINT8 vs PT: {cos_xint8:.4f}  {status}')

# Embed metadata
model_onnx = onnx.load(str(XINT8_PATH))
for k, v in {
    'cortex_version':    'v16.15',
    'model_name':        'temporal_head',
    'checkpoint_sha256': ckpt_hash,
    'export_timestamp':  datetime.datetime.now().isoformat(),
    'quantization':      'XINT8',
    'input_shape':       '1x128',
    'output_shape':      '1x64',
    'n_params':          str(n_params),
    'target_hardware':   'AMD XDNA2 Ryzen AI MAX+ 395',
    'cos_fp32_vs_pt':    str(round(cos_fp32, 6)),
    'cos_xint8_vs_pt':   str(round(cos_xint8, 4)),
}.items():
    e = model_onnx.metadata_props.add()
    e.key, e.value = k, v
onnx.save(model_onnx, str(XINT8_PATH))

# Update manifest
manifest_path = OUT_DIR / 'npu_stack_manifest.json'
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
manifest['temporal_head'] = {
    'name': 'temporal_head', 'cos_fp32': cos_fp32,
    'cos_xint8': cos_xint8, 'sha256': ckpt_hash,
    'xint8_path': str(XINT8_PATH),
}
manifest_path.write_text(json.dumps(manifest, indent=2))

print(f'\nManifest updated: {manifest_path}')
print(f'NPU stack complete:')
for name, r in manifest.items():
    cx = r.get('cos_xint8', 'N/A')
    print(f'  {name:30s} cosine={cx}')
