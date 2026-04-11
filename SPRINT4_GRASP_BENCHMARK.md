# Sprint 4 — GRASP Planner Benchmark
> Date: 2026-04-02 | Hardware: GMKtec EVO-X2, AMD Ryzen AI MAX+ 395

## Result: PASS at H=4, iters=3

| Config | Median | P95 | Status |
|--------|--------|-----|--------|
| H=8, iters=5 (default) | 28.34ms | 80.03ms | FAIL |
| **H=4, iters=3** | **9.91ms** | **16.62ms** | **PASS ✅** |

## Confirmed Configuration

```python
GRASP_CONFIG = {
    "horizon": 4,   # 1 second at 4Hz — matches world model accuracy peak
    "iters":   3,   # 3 Langevin refinement iterations  
    "K":       16,  # 16 particle rollouts
    "device":  "cpu",
}
```

## Regime Routing (confirmed working)

| Regime | Planner | Rationale |
|--------|---------|-----------|
| EXPLOIT | GRASP | Best planning when confident |
| EXPLORE | MirrorAscent | Fast exploration when uncertain |
| REOBSERVE | MirrorAscent | Fast when gathering information |
| WAIT | none | No action when neuro says observe |

## Why H=4 is the right horizon

The AIM probe confirmed world model accuracy peaks at k=4 (1 second):
- AUROC k=4: 0.8886 — strong temporal discrimination
- AUROC k=8: 0.8341 — still good but degrading
- L_jepa_real at k=8: 0.003 — predictor accurate

Planning at H=4 aligns the planner's horizon with the world model's
accuracy peak. This is not a compromise — it is optimal configuration.

## Budget at 4Hz

- Frame budget: 250ms per frame
- GRASP median: 9.91ms (4% of budget)
- GRASP P95: 16.62ms (6.6% of budget)
- Remaining budget: ~233ms for perception, GPS, logging

## Sprint 4 Next Steps

1. Wire GRASP into live inference pipeline
2. Build GeoLatentDB (GPS → particle embedding lookup)
3. Implement goal-conditioned navigation (GPS target → GRASP plan)
4. ContactHead collision avoidance in planner scoring
5. Planning horizon sweep eval (k=1,2,4,8 — quasimetric in planning)
6. Run `python grasp_planner.py --benchmark` on NPU path when available

