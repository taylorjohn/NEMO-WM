from quark.onnx import ModelQuantizer, QConfig
import numpy as np, onnx, json, datetime, hashlib
from pathlib import Path

fp32  = './npu_models/cortex_student_fp32.onnx'
xint8 = './npu_models/cortex_student_xint8.onnx'

# Calibration reader - synthetic is fine for this architecture
class CalibReader:
    def __init__(self, n=200):
        self.data = [{'pixel_values': np.random.randn(1,3,224,224).astype(np.float32)} for _ in range(n)]
        self.idx  = 0
    def get_next(self):
        if self.idx >= len(self.data): return None
        r = self.data[self.idx]; self.idx += 1; return r

config    = QConfig.get_default_config('XINT8')
quantizer = ModelQuantizer(config)
print('Quantizing...')
quantizer.quantize_model(fp32, xint8, CalibReader())

# Embed metadata
sha = hashlib.sha256(open('./checkpoints/maze_weak_sigreg_straight/cortex_student_phase2_final.pt','rb').read()).hexdigest()
model = onnx.load(xint8)
for k,v in {
    'cortex_version':     'v16.15',
    'checkpoint_sha256':  sha,
    'export_timestamp':   datetime.datetime.now().isoformat(),
    'architecture':       'CortexCNNBackbone+ShatteredLatentHead',
    'quantization':       'XINT8',
    'input':              '1x3x224x224',
    'output':             'L2-normalized 128-D latent',
    'target':             'AMD XDNA2 Ryzen AI MAX+ 395',
}.items():
    e = model.metadata_props.add(); e.key=k; e.value=v
onnx.save(model, xint8)

size = Path(xint8).stat().st_size
print(f'XINT8 ONNX: {xint8}  ({size/1024:.1f}KB)')
print(f'SHA256: {sha[:32]}...')
json.dump({'checkpoint_sha256':sha,'export_timestamp':datetime.datetime.now().isoformat(),'xint8_size_kb':size/1024},
          open('./npu_models/export_manifest.json','w'), indent=2)
print('Done - manifest saved')
