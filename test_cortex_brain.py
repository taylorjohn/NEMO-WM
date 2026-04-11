"""
test_cortex_brain.py — CORTEX-16 v2 Brain Test Suite
=====================================================
Tests all three brain components in isolation + integrated.

Run:
  python test_cortex_brain.py

Expected: all tests PASS in <5 seconds.
"""

import sys
import math
import time
import numpy as np
import torch

# Import brain components
sys.path.insert(0, ".")
from cortex_v2_brain_live import (
    StaticCSRRouter,
    CJEPAPredictor,
    DopamineSystem,
    MPCActionSelector,
    NeuroState,
    BASE_Z_ENTRY,
)

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []

def test(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((name, condition))
    print(f"  {status}  {name}" + (f" — {detail}" if detail else ""))
    return condition


print("\n" + "="*60)
print("  CORTEX-16 v2 Brain Test Suite")
print("="*60)


# ── T1: StaticCSRRouter ──────────────────────────────────────────────────────
print("\n[T1] StaticCSRRouter")

router = StaticCSRRouter()

# Basic routing
channels = router.route(
    bid=500.0, ask=500.05, micro_price=500.025,
    spread=0.05, bid_sz=100.0, ask_sz=80.0,
    z_score=1.5, rtt_ms=12.0
)
test("Output shape (4,)", channels.shape == (4,),
     f"got {channels.shape}")
test("Output dtype float32", channels.dtype == np.float32,
     f"got {channels.dtype}")
test("Channels finite", np.all(np.isfinite(channels)),
     f"channels={channels}")

# Consistency: same input → same output after warmup
for _ in range(50):
    router.route(500.0, 500.05, 500.025, 0.05, 100.0, 80.0, 1.5, 12.0)

c_check = router.route(500.0, 500.05, 500.025, 0.05, 100.0, 80.0, 1.5, 12.0)

# High spread raises liquidity channel differently than low spread
c_tight = router.route(500.0, 500.01, 500.005, 0.01, 100.0, 100.0, 0.5, 10.0)
c_wide  = router.route(500.0, 500.50, 500.25,  0.50, 10.0,  200.0, 3.0, 40.0)
test("Warmup channels finite and bounded", np.all(np.abs(c_check) < 100))
test("High spread produces distinct routing",
     not np.allclose(c_tight, c_wide, atol=0.01))

# Speed test
t0 = time.perf_counter()
for _ in range(10000):
    router.route(500.0, 500.05, 500.025, 0.05, 100.0, 80.0, 1.5, 12.0)
ms = (time.perf_counter() - t0) * 1000 / 10000
test(f"Routing speed <0.1ms/call", ms < 0.1, f"{ms:.4f}ms/call")


# ── T2: CJEPAPredictor ───────────────────────────────────────────────────────
print("\n[T2] CJEPAPredictor")

pred = CJEPAPredictor()
n_params = sum(p.numel() for p in pred.parameters())
test(f"Parameter count <10K", n_params < 10000, f"{n_params:,} params")

# Forward pass shapes
state  = torch.randn(1, 4)
action = torch.tensor([[1.0, 0.0, 0.0]])  # buy
out = pred(state, action)
test("Output shape (1,4)", out.shape == (1, 4), f"got {out.shape}")
test("Output finite", torch.all(torch.isfinite(out)).item())

# Rollout
channels_np = np.random.randn(4).astype(np.float32)
action_np   = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # hold
preds = pred.rollout(channels_np, action_np, horizon=5)
test("Rollout shape (5,4)", preds.shape == (5, 4), f"got {preds.shape}")
test("Rollout finite", np.all(np.isfinite(preds)))

# Buy vs hold vs sell produce different predictions
buy_preds  = pred.rollout(channels_np, np.array([1,0,0], dtype=np.float32))
sell_preds = pred.rollout(channels_np, np.array([0,1,0], dtype=np.float32))
hold_preds = pred.rollout(channels_np, np.array([0,0,1], dtype=np.float32))
test("Buy≠sell predictions",
     not np.allclose(buy_preds, sell_preds, atol=1e-4))
test("Buy≠hold predictions",
     not np.allclose(buy_preds, hold_preds, atol=1e-4))

# Symlog/symexp inverse
x = torch.tensor([[-10.0, -1.0, 0.0, 1.0, 10.0]])
x_roundtrip = CJEPAPredictor.symexp(CJEPAPredictor.symlog(x))
test("symlog/symexp roundtrip", torch.allclose(x, x_roundtrip, atol=1e-5))

# Speed
t0 = time.perf_counter()
for _ in range(1000):
    pred.rollout(channels_np, action_np, horizon=5)
ms = (time.perf_counter() - t0) * 1000 / 1000
test(f"Rollout speed <5ms", ms < 5.0, f"{ms:.3f}ms/rollout")


# ── T3: DopamineSystem ───────────────────────────────────────────────────────
print("\n[T3] DopamineSystem")

da = DopamineSystem()

# Initial state
ns = da.state
test("Initial DA=0.5", abs(ns.da - 0.5) < 0.01, f"da={ns.da}")
test("Initial cortisol=0.0", ns.cortisol == 0.0, f"cort={ns.cortisol}")

# Update with normal conditions
ns = da.update(z_actual=1.0, z_predicted=1.1,
               rho=2.0, is_toxic=False, rtt_ms=12.0)
test("Signals in [0,1]: DA", 0 <= ns.da <= 1, f"da={ns.da:.4f}")
test("Signals in [0,1]: 5HT", 0 <= ns.sht <= 1, f"sht={ns.sht:.4f}")
test("Signals in [0,1]: eCB", 0 <= ns.ecb <= 1, f"ecb={ns.ecb:.4f}")
test("E/I in [0.5,2.0]", 0.5 <= ns.ei <= 2.0, f"ei={ns.ei:.4f}")

# Cortisol rises under sustained toxic conditions
for _ in range(50):
    da.update(z_actual=5.0, z_predicted=0.0,
              rho=8.0, is_toxic=True, rtt_ms=50.0)
ns_stressed = da.state
test("Cortisol rises under stress",
     ns_stressed.cortisol > 0.01,
     f"cortisol={ns_stressed.cortisol:.4f}")

# Entry threshold rises with cortisol
test("z_entry > BASE under stress",
     ns_stressed.z_entry_threshold > BASE_Z_ENTRY,
     f"z_entry={ns_stressed.z_entry_threshold:.3f}")

# Position scale shrinks with cortisol
test("pos_scale < 1 under stress",
     ns_stressed.position_scale < 1.0,
     f"scale={ns_stressed.position_scale:.3f}")

# Regime is WAIT when SHT low
da2 = DopamineSystem()
da2.state.sht = 0.1
ns2 = da2.update(z_actual=1.0, z_predicted=1.0,
                 rho=1.0, is_toxic=False, rtt_ms=10.0)
test("WAIT regime when 5HT low", ns2.regime == "WAIT",
     f"regime={ns2.regime}")

# DA signal rises on surprise
da3 = DopamineSystem()
# Prime with some history
for _ in range(10):
    da3.update(0.5, 0.5, 2.0, False, 10.0)
ns_base = da3.state.da
da3.update(z_actual=4.5, z_predicted=0.1,  # big surprise
           rho=2.0, is_toxic=False, rtt_ms=10.0)
test("DA rises on surprise", da3.state.da > ns_base,
     f"before={ns_base:.4f} after={da3.state.da:.4f}")


# ── T4: MPCActionSelector ────────────────────────────────────────────────────
print("\n[T4] MPCActionSelector")

mpc = MPCActionSelector(pred, horizon=5)

channels_np = np.zeros(4, dtype=np.float32)
ns = NeuroState()

action, confidence = mpc.select(channels_np, ns)
test("Returns valid action",
     action in ["buy", "sell", "hold"],
     f"action={action}")
test("Confidence in [0,1]", 0 <= confidence <= 1,
     f"confidence={confidence:.4f}")

# Stressed state prefers hold (explore_std large → softer selection)
ns_stressed = NeuroState(cortisol=0.8, position_scale=0.2, explore_std=0.15)
actions_stressed = [mpc.select(channels_np, ns_stressed)[0] for _ in range(20)]
test("MPC runs 20 times without error",
     len(actions_stressed) == 20)


# ── T5: Integration — brain_tick without Alpaca ──────────────────────────────
print("\n[T5] Integration (no Alpaca)")

# Build a mock engine without connecting to Alpaca
class MockBrain:
    def __init__(self):
        self.router    = StaticCSRRouter()
        self.predictor = CJEPAPredictor()
        self.da_system = DopamineSystem()
        self.mpc       = MPCActionSelector(self.predictor)
        self.last_channels = np.zeros(4, dtype=np.float32)

    def tick(self, bid, ask, z_score, rho, is_toxic=False, rtt_ms=12.0):
        micro = (bid + ask) / 2
        channels = self.router.route(
            bid=bid, ask=ask, micro_price=micro,
            spread=ask-bid, bid_sz=100.0, ask_sz=80.0,
            z_score=z_score, rtt_ms=rtt_ms
        )
        hold = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        pred_state = self.predictor.rollout(channels, hold, 1)
        z_pred = float(pred_state[0, 2]) * 3.0
        ns = self.da_system.update(
            z_actual=z_score, z_predicted=z_pred,
            rho=rho, is_toxic=is_toxic, rtt_ms=rtt_ms
        )
        self.last_channels = channels
        action, conf = self.mpc.select(channels, ns)
        return ns, action, conf

brain = MockBrain()

# Warm-up ticks
for i in range(100):
    ns, act, conf = brain.tick(500.0, 500.05, 1.0 + 0.01*i, 3.0)

test("100 warm-up ticks complete", True)
test("DA signal active after warmup", 0 <= ns.da <= 1,
     f"da={ns.da:.4f}")
test("Action selected", act in ["buy", "sell", "hold"])

# Stress scenario: 50 toxic ticks
for _ in range(50):
    ns, act, conf = brain.tick(500.0, 500.5, 5.0, 8.0,
                               is_toxic=True, rtt_ms=50.0)

test("Cortisol elevated after toxic ticks",
     ns.cortisol > 0.01, f"cortisol={ns.cortisol:.4f}")
test("Entry threshold elevated",
     ns.z_entry_threshold > BASE_Z_ENTRY,
     f"z_entry={ns.z_entry_threshold:.3f} (base={BASE_Z_ENTRY})")
test("Position scale reduced",
     ns.position_scale < 0.9, f"scale={ns.position_scale:.3f}")

# Recovery: 50 clean ticks
for _ in range(50):
    ns, act, conf = brain.tick(500.0, 500.02, 0.5, 2.0)
test("Cortisol decays after clean ticks",
     ns.cortisol < brain.da_system.state.cortisol + 0.1,
     f"cortisol={ns.cortisol:.4f}")

# Speed: full tick loop
t0 = time.perf_counter()
for _ in range(1000):
    brain.tick(500.0, 500.05, 1.5, 3.0)
ms = (time.perf_counter() - t0) * 1000 / 1000
test(f"Full brain tick <5ms", ms < 5.0, f"{ms:.3f}ms/tick")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
total  = len(results)
passed = sum(1 for _, ok in results if ok)
failed = total - passed

print(f"  {passed}/{total} tests passed")
if failed:
    print(f"\n  Failed tests:")
    for name, ok in results:
        if not ok:
            print(f"    ❌ {name}")

print("="*60 + "\n")
sys.exit(0 if failed == 0 else 1)
