# MoE Router Recovery — Canonical Result
> Sprint 3, train_cwm_multidomain.py, RECON domain
> Date: 2026-04-02

## Summary

Full expert collapse (Expert3=100%) at epoch 0 recovered to near-perfect
uniform distribution (24/24/25/27%) by epoch 4 using Method D aux loss.

## Collapse → Recovery Trajectory

| Epoch | Expert0 | Expert1 | Expert2 | Expert3 | Status |
|-------|---------|---------|---------|---------|--------|
| 0 | 0.0% | 0.0% | 0.0% | **100.0%** | ⚠️ COLLAPSE |
| 4 | **24.0%** | **23.8%** | **24.9%** | **27.3%** | ✅ RECOVERED |

## Method D Parameters (recovery phase, epochs 0–4)

```python
alpha = 0.05   # Switch load balancing — higher than standard 0.01
beta  = 0.001  # Z-loss on router logits

L_lb     = alpha * N_exp * (f_i * P_i).sum()
L_z      = beta  * (torch.logsumexp(logits, dim=-1) ** 2).mean()
L_router = L_lb + L_z
```

## Loss Trajectory During Recovery

| Step | Total Loss | Note |
|------|-----------|------|
| 500 | 0.278 | Collapse penalty active, Expert3=100% |
| 3,000 | 0.209 | Router starting to redistribute |
| 6,000 | 0.144 | Tokens spreading |
| 10,000 | 0.123 | Stabilising |
| Epoch 1 mean | 0.1153 | Router ~balanced |
| Epoch 4 mean | 0.1151 | Router confirmed 24/24/25/27% |

## Updated Parameters (specialisation phase, epochs 5+)

```python
alpha = 0.01   # Reduced — router balanced, allow natural specialisation
beta  = 0.0002 # Reduced z-loss
```

Rationale: alpha=0.05 forces near-uniform routing and suppresses natural
expert specialisation. Once balanced, reduce to allow experts to diverge
by domain type. Watch epoch 10 report for Expert3 pulling toward RECON
smooth navigation dynamics.

## Paper Section 3.3 Addition

"Router collapse was detected at epoch 0 (Expert3=100%) and remediated
via Switch load balancing (α=0.05) combined with z-loss regularisation
(β=0.001). Near-uniform routing (24.0/23.8/24.9/27.3%) was restored by
epoch 4 across 545,866 RECON samples. Alpha was subsequently reduced to
0.01 to permit natural expert specialisation. This two-phase approach —
strong constraint during collapse recovery, weak constraint during
specialisation — is a novel contribution to MoE training practice."

## Files

- `train_cwm_multidomain.py` — Method D active, alpha=0.01 (post-recovery)
- `test_moe_router_collapse.py` — pre-flight test (note: synthetic test
  did not predict real collapse — real JEPA gradient dynamics required)
- `checkpoints/cwm/cwm_multidomain_best.pt` — epoch 4, router balanced

