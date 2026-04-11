# Sprint 4 Results — GRASP Planner
**Date:** 2026-04-06  
**Status:** PASS ✓  
**Committed config:** H=8, iters=2, median 9.10ms

---

## Implementation

**File:** `grasp_planner.py`  
**Reference:** arXiv:2602.00475 — Psenka, Rabbat, Krishnapriyan, LeCun, Bar (Meta FAIR 2026)

### CORTEX contributions beyond paper
1. **E/I-gated Langevin noise**: noise_std = base × (1 + e_i_scale × (E/I − 1))
   - High E/I (exploration) → wider stochastic search
   - Low E/I (exploitation) → precise gradient-based planning
2. **Regime-gated planner selection**:
   - EXPLOIT → GRASPPlanner (gradient-based, lifted states)
   - EXPLORE/REOBSERVE → MirrorAscentSampler (zero-order, 32 candidates)
   - WAIT → no action

### Architecture
- Lifted states: H+1 virtual states s_0..s_H optimised in parallel
- Grad-cut: stop-gradient on state inputs to predictor (stability trick from paper Sec 3.3)
- Loss: L_goal + 0.3 × L_dense + L_dyn
- Optimisers: Adam on actions (lr=0.05) and virtual states (lr=0.10)

---

## Benchmark Results

**Hardware:** GMKtec EVO-X2, AMD Ryzen AI MAX+ 395, 128GB, no GPU  
**Environment:** conda ryzen-ai-1.7.0, Python 3.12  
**MockPredictor:** MLP (K×d_model + action_dim → 256 → K×d_model)  
**Trials:** 20 timed after 3 warmup runs

| Config | Mean (ms) | Median (ms) | P95 (ms) | Min/Max (ms) | Result |
|---|---|---|---|---|---|
| H=5, iters=3 | 14.02 | 12.57 | 20.44 | 10.18/23.45 | FAIL |
| H=8, iters=5 | 35.30 | 29.08 | 66.52 | 22.48/73.10 | FAIL |
| **H=8, iters=2** | **11.09** | **9.10** | **21.53** | **8.10/23.16** | **PASS** |

**Target:** median < 10ms for 4Hz (250ms budget)  
**Margin:** P95 21.53ms = 8.6% of per-frame budget — comfortable

**Key finding:** Optimizer step cost dominates over rollout length.
H=8 iters=2 outperforms H=5 iters=3 (9.10ms vs 12.57ms) despite longer horizon,
because 2 Adam steps < 3 Adam steps × the cost difference.

---

## Self-test results

```
GRASP: action=[0. 0.], loss=~1.0, time=24–28ms  ← self-test uses different config
MirrorAscent: action=[~0.01, ~-0.003]
regime=EXPLOIT:     planner=GRASP,        acted=True
regime=EXPLORE:     planner=MirrorAscent, acted=True
regime=REOBSERVE:   planner=MirrorAscent, acted=True
regime=WAIT:        planner=none,         acted=False
All assertions passed.
```

Note: self-test times (24–28ms) reflect default H=8, iters=5 — not the committed config.
Benchmark times (9.10ms) use committed H=8, iters=2.

---

## Integration into Sprint 4 loop

```python
from grasp_planner import regime_gated_plan, GRASPConfig

config = GRASPConfig(horizon=8, n_lifted_iters=2)  # benchmarked config

action, info = regime_gated_plan(
    cwm_predictor  = predictor,       # CWM forward: (particles, action) -> next_particles
    particles_0    = particles_t,     # (1, K, d_model) current state
    goal_particles = goal,            # (1, K, d_model) goal state
    signals        = neuro.signals,   # NeuroState with .regime, .da, .e_i, .ne
    action_dim     = 2,
    grasp_config   = config,
    device         = "cpu",
)
# info keys: planner, regime, acted, elapsed_ms, best_loss, n_iters
```

---

## Files changed

- `grasp_planner.py`: `GRASPConfig` defaults updated to `horizon=8, n_lifted_iters=2`
  - Comment documents benchmark date, median, P95
  - CLI `--iters` default updated to 2
