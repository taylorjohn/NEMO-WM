from neuromodulator import NeuromodulatorState, CortisolSignal
import torch, time

# Test 1: CortisolSignal standalone
print("Test 1: CortisolSignal standalone")
cs = CortisolSignal(baseline=0.567)
for loss in [0.571, 0.572, 0.573, 0.574, 0.575]:
    c = cs.update(loss)
    print(f"  loss={loss:.3f} cortisol={c:.4f}")

# Test 2: Integrated in NeuromodulatorState
print("\nTest 2: Integrated in NeuromodulatorState")
neuro = NeuromodulatorState(session_start=time.time())
z = torch.randn(128)
sig = neuro.update(z, z * 0.9, rho=0.5)
print(f"  cortisol in signals: {sig['cortisol']}")
print(f"  all keys: {list(sig.keys())}")

# Test 3: Modulation effect
print("\nTest 3: Cortisol modulation")
cs2 = CortisolSignal(baseline=0.567)
for _ in range(50):
    cs2.update(0.575)  # sustained elevation
cort = cs2.update(0.575)
signals = {"rho": 0.3, "ecb": 0.4}
modulated = cs2.modulate(signals, cort)
print(f"  cortisol: {cort:.4f}")
print(f"  NE before: 0.300  after: {modulated['rho']:.4f}  (should be higher)")
print(f"  eCB before: 0.400  after: {modulated['ecb']:.4f}  (should be lower)")

# Test 4: Reset
print("\nTest 4: Reset")
neuro.reset(full=True)
print(f"  cortisol after reset: {neuro.cortisol}")

print("\nAll OK")
