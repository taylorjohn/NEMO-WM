# test_all.py
# Benchmark all VLM fixes on AMD Ryzen AI MAX+ NPU
#
# Usage:
#   python test_all.py --benchmark
#   python test_all.py --export-all
#   python test_all.py --check

import argparse
import os
import time
import sys
import numpy as np

FIXES = {
    'llava':       {'script': 'fix_llava/run.py',       'model': 'CLIP ViT-L/14 (LLaVA)',    'cpu_ms': 375.0},
    'siglip-base': {'script': 'fix_siglip/run.py',      'model': 'SigLIP ViT-B/16',           'cpu_ms': 80.0},
    'siglip-so':   {'script': 'fix_siglip/run.py',      'model': 'SigLIP SO400M',             'cpu_ms': 200.0},
    'vjepa2-l':    {'script': 'fix_vjepa2/run.py',      'model': 'V-JEPA 2 ViT-L (8 frames)','cpu_ms': 1849.0},
}

XINT8_PATHS = {
    'llava':       'fix_llava/clip_vitl14_xint8.onnx',
    'siglip-base': 'fix_siglip/siglip_base_xint8.onnx',
    'siglip-so':   'fix_siglip/siglip_so400m_xint8.onnx',
    'vjepa2-l':    'fix_vjepa2/vjepa2_l_frame_xint8.onnx',
}


def check_status():
    """Show which fixes are exported and ready."""
    print('\nFix Status')
    print('=' * 60)
    print(f"  {'Fix':15s} {'ONNX':8s} {'XINT8':8s} {'Status'}")
    print('  ' + '-' * 50)

    for name, info in FIXES.items():
        xint8 = XINT8_PATHS[name]
        onnx  = xint8.replace('_xint8.onnx', '.onnx')
        has_onnx  = os.path.exists(onnx)
        has_xint8 = os.path.exists(xint8)

        if has_xint8:
            status = 'Ready for NPU'
        elif has_onnx:
            status = 'Need quantization'
        else:
            status = 'Need --export'

        o = 'yes' if has_onnx  else 'no'
        x = 'yes' if has_xint8 else 'no'
        print(f"  {name:15s} {o:8s} {x:8s} {status}")

    print()
    print('To export all:  python test_all.py --export-all')
    print('To benchmark:   python test_all.py --benchmark')


def export_all(device='cpu'):
    """Export and quantize all supported models."""
    print('\nExporting all fixes...')
    print('This downloads models from HuggingFace and may take a while.')
    print()

    steps = [
        ('LLaVA (CLIP ViT-L/14)', 'python fix_llava/run.py --export'),
        ('SigLIP base',           'python fix_siglip/run.py --model siglip-base --export'),
        ('V-JEPA 2-L',            'python fix_vjepa2/run.py --model vjepa2-l --export'),
    ]

    for name, cmd in steps:
        print(f'--- {name} ---')
        print(f'  {cmd}')
        ret = os.system(cmd)
        if ret != 0:
            print(f'  Failed (exit code {ret})')
        print()


def benchmark_all(vaip_config='vaip_config.json', n=30):
    """Benchmark all exported fixes."""
    try:
        import onnxruntime as ort
    except ImportError:
        print('pip install onnxruntime'); return

    print('\nBenchmark -- AMD Ryzen AI MAX+ NPU')
    print('=' * 65)
    print(f"  {'Model':28s} {'CPU ms':8s} {'NPU ms':8s} {'Speedup':8s} {'Status'}")
    print('  ' + '-' * 62)

    results = {}
    for name, info in FIXES.items():
        xint8 = XINT8_PATHS[name]
        if not os.path.exists(xint8):
            print(f"  {info['model']:28s} {info['cpu_ms']:7.0f}  {'--':8s} {'--':8s} not exported")
            continue

        try:
            providers = (['VitisAIExecutionProvider'] if os.path.exists(vaip_config)
                         else ['CPUExecutionProvider'])
            opts = ([{'config_file': vaip_config}] if os.path.exists(vaip_config)
                    else [{}])
            sess = ort.InferenceSession(xint8, providers=providers, provider_options=opts)

            # Get input shape
            inp = sess.get_inputs()[0]
            shape = [d if isinstance(d, int) else 1 for d in inp.shape]
            dummy = {inp.name: np.random.randn(*shape).astype(np.float32)}

            # Warmup
            for _ in range(5): sess.run(None, dummy)

            # Measure
            lats = []
            for _ in range(n):
                t0 = time.perf_counter()
                sess.run(None, dummy)
                lats.append((time.perf_counter() - t0) * 1000)

            med     = float(np.median(lats))
            cpu_ms  = info['cpu_ms']
            speedup = cpu_ms / med if med > 0 else 0
            provider = 'NPU' if os.path.exists(vaip_config) else 'CPU'

            results[name] = {'npu_ms': med, 'cpu_ms': cpu_ms, 'speedup': speedup}
            print(f"  {info['model']:28s} {cpu_ms:7.0f}  {med:7.1f}  {speedup:6.1f}x  {provider}")

        except Exception as e:
            print(f"  {info['model']:28s} {info['cpu_ms']:7.0f}  error: {e}")

    if results:
        print()
        best = max(results.items(), key=lambda x: x[1]['speedup'])
        print(f"  Best speedup: {best[0]} -- {best[1]['speedup']:.0f}x faster on NPU")


def main():
    ap = argparse.ArgumentParser(
        description='AMD NPU VLM fix -- test and benchmark all models'
    )
    ap.add_argument('--check',       action='store_true', help='Check export status')
    ap.add_argument('--export-all',  action='store_true', help='Export all models')
    ap.add_argument('--benchmark',   action='store_true', help='Benchmark all exported')
    ap.add_argument('--vaip-config', default='vaip_config.json')
    ap.add_argument('--device',      default='cpu')
    args = ap.parse_args()

    print('\namd-npu-vlm-compat -- Fix popular VLMs for AMD Ryzen AI MAX+ NPU')

    if args.check or not any([args.export_all, args.benchmark]):
        check_status()

    if args.export_all:
        export_all(args.device)

    if args.benchmark:
        benchmark_all(args.vaip_config)

    if not any([args.check, args.export_all, args.benchmark]):
        print()
        print('Usage:')
        print('  python test_all.py --check          # see what is ready')
        print('  python test_all.py --export-all     # download + export all')
        print('  python test_all.py --benchmark      # measure NPU speedup')


if __name__ == '__main__':
    main()
