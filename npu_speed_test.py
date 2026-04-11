import onnxruntime as ort
import numpy as np
import time

MODELS = [
    ('StudentEncoder',      './npu_models/cortex_student_xint8.onnx',      {'pixel_values': np.zeros((1,3,224,224), np.float32)}),
    ('TransitionPredictor', './npu_models/transition_predictor_xint8.onnx', {'state':        np.zeros((1,8),         np.float32)}),
    ('TemporalHead',        './npu_models/temporal_head_xint8.onnx',        {'latent':       np.zeros((1,128),        np.float32)}),
    ('CardiacStudent',      './npu_models/cardiac_student_xint8.onnx',      {'audio':        np.zeros((1,1,2000),     np.float32)}),
]

PROVIDER_OPTS = [{
    'cache_dir':   './npu_cache',
    'cache_key':   'cortex_stack',
    'target':      'X2',
}]

WARMUP = 10
RUNS   = 100

print('\nCORTEX-PE NPU Stack Speed Test')
print('=' * 55)
print(f'{"Model":25s}  {"Warmup":8s}  {"Mean":8s}  {"P99":8s}  {"Throughput"}')
print('-' * 55)

for name, path, inputs in MODELS:
    try:
        sess = ort.InferenceSession(
            path,
            providers=['VitisAIExecutionProvider', 'CPUExecutionProvider'],
            provider_options=PROVIDER_OPTS + [{}],
        )

        # Warmup
        for _ in range(WARMUP):
            sess.run(None, inputs)

        # Timed runs
        times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            sess.run(None, inputs)
            times.append((time.perf_counter() - t0) * 1000)

        mean_ms = np.mean(times)
        p99_ms  = np.percentile(times, 99)
        fps     = 1000 / mean_ms

        print(f'  {name:23s}  {mean_ms:6.3f}ms  {mean_ms:6.3f}ms  {p99_ms:6.3f}ms  {fps:6.0f}Hz')

    except Exception as e:
        print(f'  {name:23s}  ERROR: {e}')

print('=' * 55)
print('\nFull stack (all 4 sequential):')
sessions = []
for name, path, inputs in MODELS:
    try:
        sessions.append((name, ort.InferenceSession(
            path,
            providers=['VitisAIExecutionProvider','CPUExecutionProvider'],
            provider_options=PROVIDER_OPTS + [{}],
        ), inputs))
    except: pass

for _ in range(WARMUP):
    for _, sess, inp in sessions: sess.run(None, inp)

stack_times = []
for _ in range(RUNS):
    t0 = time.perf_counter()
    for _, sess, inp in sessions: sess.run(None, inp)
    stack_times.append((time.perf_counter() - t0) * 1000)

print(f'  Total stack mean : {np.mean(stack_times):.3f}ms')
print(f'  Total stack P99  : {np.percentile(stack_times,99):.3f}ms')
print(f'  Stack throughput : {1000/np.mean(stack_times):.0f}Hz')
