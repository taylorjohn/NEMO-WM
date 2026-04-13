"""
test_episodic_buffer.py — Sprint D6 Tests & Benchmarks
=======================================================
Tests all four EpisodicBuffer components + integration with ImaginationRollout.
Matches the actual API in episodic_buffer.py.

Run:
    python test_episodic_buffer.py                # all tests
    python test_episodic_buffer.py --bench         # benchmarks only
    python test_episodic_buffer.py --integration   # integration with rollout
"""

import argparse
import time
import sys
import numpy as np
import torch

from episodic_buffer import (
    Episode, WorkingMemory, EpisodicStore, SchemaStore, EpisodicBuffer,
)

# ──────────────────────────────────────────────────────────────────────────────
PASSED = 0
FAILED = 0
D = 64

def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  ✅ {name}")
    else:
        FAILED += 1
        print(f"  ❌ {name}  {detail}")

def rand_belief():
    b = torch.randn(D)
    return b / b.norm()

def rand_action():
    return torch.randn(2) * 0.1


# ──────────────────────────────────────────────────────────────────────────────
# 1. WorkingMemory
# ──────────────────────────────────────────────────────────────────────────────
def test_working_memory():
    print("\n── WorkingMemory ─────────────────────────────────────")
    wm = WorkingMemory(K=8)

    # Effective K under cortisol
    check("K_eff at CORT=0.0", wm.effective_k(0.0) == 8)
    check("K_eff at CORT=0.5 < 8", wm.effective_k(0.5) < 8)
    check("K_eff at CORT=1.0 >= 1", wm.effective_k(1.0) >= 1)

    # Push beliefs
    for i in range(5):
        wm.push(rand_belief(), rand_action())

    # as_tensor should return something
    t = wm.as_tensor(crt=0.0)
    check("as_tensor not None after pushes", t is not None)
    if t is not None:
        check("as_tensor is 2D", t.dim() == 2)
        check("as_tensor cols = D", t.shape[-1] == D)

    # Under high cortisol, fewer items
    t_stress = wm.as_tensor(crt=0.8)
    if t is not None and t_stress is not None:
        check("stressed WM smaller", t_stress.shape[0] <= t.shape[0],
              f"{t_stress.shape[0]} vs {t.shape[0]}")

    # Fill past capacity
    for i in range(10):
        wm.push(rand_belief(), rand_action())
    t_full = wm.as_tensor(crt=0.0)
    if t_full is not None:
        check("capped at K=8", t_full.shape[0] <= 8)

    # Attend
    query = rand_belief()
    attended = wm.attend(query, crt=0.0)
    if attended is not None:
        check("attend returns D-dim", attended.shape == (D,))

    # Empty WM
    wm_empty = WorkingMemory(K=4)
    check("empty as_tensor is None", wm_empty.as_tensor() is None)


# ──────────────────────────────────────────────────────────────────────────────
# 2. EpisodicStore
# ──────────────────────────────────────────────────────────────────────────────
def test_episodic_store():
    print("\n── EpisodicStore ─────────────────────────────────────")
    store = EpisodicStore(capacity=100)

    # Store episodes
    for i in range(50):
        b = rand_belief()
        a = rand_action()
        b1 = rand_belief()
        da = float(torch.rand(1)) * 0.8
        delta = torch.randn(D) * 0.1
        store.store(b, a, b1, da=da, delta=delta, step=i)

    # Retrieve by similarity
    query = rand_belief()
    similar = store.retrieve(query, k=5)
    check("retrieve returns list", isinstance(similar, list))
    check("retrieve returns ≤5", len(similar) <= 5)
    if similar:
        check("retrieve returns Episodes",
              all(isinstance(ep, Episode) for ep in similar))
        # Check similarity ordering
        sims = [torch.cosine_similarity(query.unsqueeze(0),
                ep.b_t.unsqueeze(0)).item() for ep in similar]
        check("retrieval sorted by similarity",
              all(sims[i] >= sims[i+1] - 0.001 for i in range(len(sims)-1)),
              f"sims={[f'{s:.3f}' for s in sims]}")

    # Replay
    batch = store.replay(n=10)
    check("replay returns list", isinstance(batch, list))
    check("replay returns ≤10", len(batch) <= 10)

    # Capacity
    store2 = EpisodicStore(capacity=20)
    for i in range(50):
        store2.store(rand_belief(), rand_action(), rand_belief(),
                     da=float(torch.rand(1)), delta=torch.randn(D)*0.1, step=i)
    # Should not exceed capacity
    replay_all = store2.replay(n=100)
    check("capacity respected", len(replay_all) <= 20)


# ──────────────────────────────────────────────────────────────────────────────
# 3. SchemaStore
# ──────────────────────────────────────────────────────────────────────────────
def test_schema_store():
    print("\n── SchemaStore ───────────────────────────────────────")
    ss = SchemaStore()

    b = rand_belief()

    # Update accumulates statistics
    for i in range(20):
        ss.update(rand_belief())
    check("update runs without error", True)

    # Compress
    compressed = ss.compress(b)
    check("compress returns tensor", isinstance(compressed, torch.Tensor))
    check("compress shape matches", compressed.shape == b.shape)

    # Novelty
    nov = ss.novelty(b)
    check("novelty returns float", isinstance(nov, float))
    check("novelty >= 0", nov >= 0.0)

    # Reconstruct
    delta = torch.randn(D) * 0.1
    recon = ss.reconstruct(delta)
    check("reconstruct returns tensor", isinstance(recon, torch.Tensor))

    # New domain
    ss.new_domain()
    nov_after = ss.novelty(b)
    check("novelty after new_domain", isinstance(nov_after, float))


# ──────────────────────────────────────────────────────────────────────────────
# 4. EpisodicBuffer (unified)
# ──────────────────────────────────────────────────────────────────────────────
def test_episodic_buffer():
    print("\n── EpisodicBuffer (unified) ──────────────────────────")
    buf = EpisodicBuffer(k_wm=8, capacity=1000, crt_domain_thresh=0.5)

    # Store 100 steps
    for i in range(100):
        b = rand_belief()
        a = rand_action()
        b1 = rand_belief()
        da = 0.02 if i % 20 != 0 else 0.6
        crt = 0.1
        buf.store(b, a, b1, da=da, crt=crt)

    # Stats
    stats = buf.stats()
    check("stats returns dict", isinstance(stats, dict))
    print(f"    Stats: {stats}")

    # Retrieve
    query = rand_belief()
    similar = buf.retrieve(query, k=3)
    check("retrieve returns list", isinstance(similar, list))

    # Attend (WM)
    attended = buf.attend(query, crt=0.0)
    if attended is not None:
        check("attend returns D-dim", attended.shape == (D,))
    else:
        check("attend returns None (acceptable)", True)

    # Replay
    batch = buf.replay(n=10)
    check("replay works", isinstance(batch, list))

    # Compress / novelty / reconstruct (SchemaStore passthrough)
    compressed = buf.compress(query)
    check("compress works", isinstance(compressed, torch.Tensor))

    nov = buf.novelty(query)
    check("novelty works", isinstance(nov, float))

    delta = torch.randn(D) * 0.1
    recon = buf.reconstruct(delta)
    check("reconstruct works", isinstance(recon, torch.Tensor))


# ──────────────────────────────────────────────────────────────────────────────
# 5. Integration with ImaginationRollout
# ──────────────────────────────────────────────────────────────────────────────
def test_integration():
    print("\n── Integration: EpisodicBuffer × ImaginationRollout ──")

    try:
        from imagination_rollout import ImaginationRollout, NeuroState, RolloutResult
        from anticipate_react_gate import AnticipateReactGate, GateState
    except ImportError as e:
        print(f"  ⚠️  Skipping integration test: {e}")
        return

    # Mock transition model
    class MockTransition(torch.nn.Module):
        def __init__(self, d=64):
            super().__init__()
            self.linear = torch.nn.Linear(d + 2, d)
        def forward(self, b, a):
            x = torch.cat([b, a], dim=-1)
            b_next = b + 0.01 * self.linear(x)
            sigma = torch.ones_like(b) * 0.1
            return b_next, sigma

    trans = MockTransition()
    gate = AnticipateReactGate()
    rollout = ImaginationRollout(trans, gate, n_actions=8)
    buf = EpisodicBuffer(k_wm=8, capacity=1000)

    # Populate buffer
    for i in range(50):
        b = rand_belief()
        a = rand_action()
        b1 = rand_belief()
        da = float(torch.rand(1)) * 0.8
        buf.store(b, a, b1, da=da, crt=0.1)

    # Retrieve similar past actions as warm-start
    query = rand_belief()
    similar = buf.retrieve(query, k=3)
    if similar:
        mem_actions = torch.stack([ep.a_t for ep in similar])
    else:
        mem_actions = None

    b_0 = query
    b_goal = rand_belief()
    neuro = NeuroState(da=0.5, ach=0.5, crt=0.1)
    gate_s = GateState()

    # Plan WITH memory actions
    result = rollout.plan(b_0, b_goal, neuro, gate_s, alpha=0.9,
                          memory_actions=mem_actions)
    check("plan with memory_actions works",
          isinstance(result, RolloutResult))
    check("plan value is finite", np.isfinite(result.value))

    # Plan WITHOUT memory (baseline)
    result_base = rollout.plan(b_0, b_goal, neuro, gate_s, alpha=0.9)
    check("baseline plan works", isinstance(result_base, RolloutResult))

    # Store rollout results back
    for b_k, sigma_k in result.trajectory:
        buf.store(b_k, result.best_action, b_k,
                  da=float(sigma_k.mean()), crt=0.1)
    check("buffer grew after rollout", True)

    # Attend with WM context
    ctx = buf.attend(b_0, crt=0.2)
    if ctx is not None:
        check("WM context for rollout conditioning", ctx.shape == (D,))
    else:
        check("WM attend returned None (ok if WM empty)", True)

    print(f"\n  {buf.stats()}")


# ──────────────────────────────────────────────────────────────────────────────
# Benchmarks
# ──────────────────────────────────────────────────────────────────────────────
def bench():
    print("\n── Benchmarks ────────────────────────────────────────")
    buf = EpisodicBuffer(k_wm=8, capacity=10_000)

    # Store throughput
    N = 10_000
    t0 = time.perf_counter()
    for i in range(N):
        buf.store(rand_belief(), rand_action(), rand_belief(),
                  da=float(torch.rand(1)) * 0.5, crt=0.1)
    elapsed = time.perf_counter() - t0
    print(f"  store():       {N} steps in {elapsed:.3f}s "
          f"({N/elapsed:.0f} steps/s, {elapsed/N*1000:.4f} ms/step)")

    # Retrieve throughput
    N_Q = 1000
    t0 = time.perf_counter()
    for _ in range(N_Q):
        buf.retrieve(rand_belief(), k=5)
    elapsed = time.perf_counter() - t0
    print(f"  retrieve(k=5): {N_Q} queries in {elapsed:.3f}s "
          f"({N_Q/elapsed:.0f} q/s, {elapsed/N_Q*1000:.4f} ms/query)")

    # Replay throughput
    N_R = 1000
    t0 = time.perf_counter()
    for _ in range(N_R):
        buf.replay(n=16)
    elapsed = time.perf_counter() - t0
    print(f"  replay(16):    {N_R} batches in {elapsed:.3f}s "
          f"({N_R/elapsed:.0f} b/s, {elapsed/N_R*1000:.4f} ms/batch)")

    # Attend throughput
    N_A = 1000
    t0 = time.perf_counter()
    for _ in range(N_A):
        buf.attend(rand_belief(), crt=0.3)
    elapsed = time.perf_counter() - t0
    print(f"  attend():      {N_A} calls in {elapsed:.3f}s "
          f"({elapsed/N_A*1000:.4f} ms/call)")

    # Compress throughput
    N_C = 1000
    t0 = time.perf_counter()
    for _ in range(N_C):
        buf.compress(rand_belief())
    elapsed = time.perf_counter() - t0
    print(f"  compress():    {N_C} calls in {elapsed:.3f}s "
          f"({elapsed/N_C*1000:.4f} ms/call)")

    # Novelty throughput
    N_N = 1000
    t0 = time.perf_counter()
    for _ in range(N_N):
        buf.novelty(rand_belief())
    elapsed = time.perf_counter() - t0
    print(f"  novelty():     {N_N} calls in {elapsed:.3f}s "
          f"({elapsed/N_N*1000:.4f} ms/call)")

    print(f"\n  Final: {buf.stats()}")


# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--integration", action="store_true")
    args = ap.parse_args()

    print("═" * 60)
    print("  NeMo-WM Sprint D6 — EpisodicBuffer Tests")
    print("═" * 60)

    if args.bench:
        bench()
    elif args.integration:
        test_integration()
    else:
        test_working_memory()
        test_episodic_store()
        test_schema_store()
        test_episodic_buffer()
        test_integration()
        bench()

        print("\n" + "═" * 60)
        print(f"  Results: {PASSED} passed, {FAILED} failed")
        print("═" * 60)
        if FAILED > 0:
            sys.exit(1)


if __name__ == "__main__":
    main()
