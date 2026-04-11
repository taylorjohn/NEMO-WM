"""
test_moe_router_collapse.py  --  Router collapse risk test for Sprint 3
=======================================================================
Tests the current moe_router_v2.py against synthetic multi-domain batches
and compares multiple mitigation strategies.

What we currently use:
  - Dense soft-gate: weights = softmax(gate(x))
  - Unimix 1%:       weights = 0.99 * softmax + 0.01/4  (applied in training)
  - No aux loss, no z-loss, no temperature control

Mitigation methods tested (A-F):
  A. Baseline -- current setup (unimix only)
  B. Unimix + Switch load balancing loss (alpha=0.01)
  C. Unimix + z-loss (beta=0.001)
  D. Unimix + load balancing + z-loss  [recommended by research]
  E. Temperature annealing (T=2.0 -> 1.0) + load balancing
  F. Small init (std=0.001) + load balancing + z-loss  [best practice]

Synthetic domains (no real data needed):
  RECON  -- image-like:  (B, 128) from DINOv2-style CLS features
  SMAP   -- telemetry:   (B, 128) from 1D time-series embedding
  MVTec  -- texture:     (B, 128) from texture patch embedding

Run: python test_moe_router_collapse.py
Expected runtime: ~3 minutes on NUC CPU
"""

import sys
import time
import math
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Import the actual router ─────────────────────────────────────────────────
try:
    from moe_router_v2 import MOERouterV2
    print("Loaded moe_router_v2.py from CORTEX root")
except ImportError:
    # Fallback: define inline (matches current file exactly)
    class MOERouterV2(nn.Module):
        def __init__(self, input_dim=256, manifold_dim=128):
            super().__init__()
            self.experts = nn.ModuleList([nn.Linear(input_dim, 32) for _ in range(4)])
            self.gate = nn.Linear(input_dim, 4)

        def forward(self, x):
            weights = torch.softmax(self.gate(x), dim=-1)
            expert_outputs = [expert(x) for expert in self.experts]
            manifold = torch.cat(expert_outputs, dim=-1)
            return torch.tanh(manifold), weights
    print("Warning: moe_router_v2.py not found, using inline replica")

NUM_EXPERTS = 4
INPUT_DIM   = 128      # CWM uses 128-D student encoder output
MANIFOLD_DIM = 128
BATCH_SIZE  = 256
N_STEPS     = 200      # pretest steps
LOG_EVERY   = 20
SEED        = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ===========================================================================
# Synthetic domain generators
# Match statistical properties of real domain embeddings
# ===========================================================================

def make_recon_batch(n=BATCH_SIZE, d=INPUT_DIM):
    """
    RECON outdoor navigation -- DINOv2 CLS token features.
    Natural image features: roughly Gaussian, moderate variance.
    """
    return F.normalize(torch.randn(n, d), dim=-1)


def make_smap_batch(n=BATCH_SIZE, d=INPUT_DIM):
    """
    SMAP satellite telemetry -- 1D time-series embedded to d-dim.
    Time-series features: sparse, low variance, often near-zero.
    Simulated with a sparse mixture of Gaussians.
    """
    # Most dims near zero (sparse signal), a few active
    base = torch.randn(n, d) * 0.1
    # Activate ~20% of dims with a different distribution
    mask = (torch.rand(n, d) < 0.2).float()
    active = torch.randn(n, d) * 0.8
    x = base + mask * active
    return F.normalize(x, dim=-1)


def make_mvtec_batch(n=BATCH_SIZE, d=INPUT_DIM):
    """
    MVTec industrial textures -- patch embedding.
    Texture features: high-frequency, more uniform distribution.
    """
    # Approximately uniform-like distribution on the sphere
    x = torch.randn(n, d)
    # Add some structure: correlated dims from texture patterns
    corr = torch.randn(d, d // 4) @ torch.randn(d // 4, d)
    x = x + 0.3 * (torch.randn(n, d) @ corr)
    return F.normalize(x, dim=-1)


DOMAIN_FACTORIES = {
    "RECON":  make_recon_batch,
    "SMAP":   make_smap_batch,
    "MVTec":  make_mvtec_batch,
}


# ===========================================================================
# Routing metrics
# ===========================================================================

def routing_metrics(weights: torch.Tensor, num_experts=NUM_EXPERTS) -> dict:
    """
    Compute all standard MoE routing health metrics.
    weights: (B, E) -- router softmax output (pre or post unimix)
    """
    B, E = weights.shape

    # 1. Normalized entropy (per-sample, averaged)
    ent = -(weights * torch.log(weights + 1e-10)).sum(-1)    # (B,)
    norm_ent = (ent.mean() / math.log(E)).item()

    # 2. Token fractions (soft, averaged across batch)
    fracs = weights.mean(0)                                   # (E,)

    # 3. Load Imbalance Factor
    lif = (E * fracs.max()).item()

    # 4. Coefficient of Variation
    cv = (fracs.std() / fracs.mean()).item()

    # 5. Dead experts (fraction < 1/(2E))
    dead = (fracs < 1 / (2 * E)).sum().item()

    # 6. Max expert dominance
    max_frac = fracs.max().item()

    return {
        "norm_entropy": norm_ent,
        "fracs":        fracs.detach().cpu().numpy(),
        "lif":          lif,
        "cv":           cv,
        "dead":         dead,
        "max_frac":     max_frac,
    }


def apply_unimix(weights, eps=0.01, num_experts=NUM_EXPERTS):
    """Apply DreamerV3-style unimix: 99% router + 1% uniform."""
    return (1 - eps) * weights + eps / num_experts


def status_emoji(norm_ent):
    if norm_ent > 0.85: return "OK  "
    if norm_ent > 0.70: return "WARN"
    return "CRIT"


# ===========================================================================
# Loss functions
# ===========================================================================

def switch_load_balance_loss(gate_logits, alpha=0.01, num_experts=NUM_EXPERTS):
    """Switch Transformer auxiliary load balancing loss."""
    probs = F.softmax(gate_logits, dim=-1)          # (B, E)
    # f_i: fraction hard-routed to expert i (non-differentiable)
    top_idx = probs.argmax(-1)                       # (B,)
    mask = F.one_hot(top_idx, num_experts).float()  # (B, E)
    f_i = mask.mean(0)                              # (E,)
    # P_i: mean softmax probability (differentiable)
    P_i = probs.mean(0)                             # (E,)
    return alpha * num_experts * (f_i * P_i).sum()


def router_z_loss(gate_logits, beta=0.001):
    """ST-MoE router z-loss: penalizes large router logits."""
    return beta * (torch.logsumexp(gate_logits, dim=-1) ** 2).mean()


# ===========================================================================
# Augmented router wrappers for each mitigation method
# ===========================================================================

class RouterWithAux(nn.Module):
    """Wrap MOERouterV2 gate to also return raw logits for aux losses."""
    def __init__(self, base: MOERouterV2, method: str, init_std: float = None,
                 init_temp: float = 1.0):
        super().__init__()
        self.gate    = copy.deepcopy(base.gate)
        self.experts = copy.deepcopy(base.experts)
        self.method  = method
        self.temp    = init_temp
        self.step    = 0

        if init_std is not None:
            nn.init.normal_(self.gate.weight, mean=0, std=init_std)
            nn.init.zeros_(self.gate.bias)

    def forward(self, x, anneal_steps=100):
        logits = self.gate(x)                        # (B, E)

        # Temperature (method E)
        if self.method in ("E",) and self.step < anneal_steps:
            t = self.temp - (self.temp - 1.0) * (self.step / anneal_steps)
            logits = logits / t

        weights = F.softmax(logits, dim=-1)          # (B, E)
        weights_mix = apply_unimix(weights)          # always apply unimix

        expert_outputs = [e(x) for e in self.experts]
        manifold = torch.cat(expert_outputs, dim=-1)

        return torch.tanh(manifold), weights_mix, logits

    def aux_loss(self, logits):
        loss = torch.tensor(0.0, requires_grad=True)
        if self.method in ("B", "D", "E", "F"):
            loss = loss + switch_load_balance_loss(logits)
        if self.method in ("C", "D", "F"):
            loss = loss + router_z_loss(logits)
        return loss


# ===========================================================================
# Phase 0: Static synthetic probe (no training)
# ===========================================================================

def static_probe():
    """
    Pass synthetic domain batches through a freshly initialized router.
    Tests whether the embedding geometry creates collapse risk at init.
    Runs in <10 seconds.
    """
    print("\n" + "="*65)
    print("  PHASE 0: STATIC PROBE — No training, fresh router")
    print("="*65)
    print("  Tests whether domain embedding geometry creates collapse risk")
    print("  at initialization, before any gradient updates.\n")

    router = MOERouterV2(input_dim=INPUT_DIM, manifold_dim=MANIFOLD_DIM)
    router.eval()

    domain_metrics = {}
    with torch.no_grad():
        for domain, factory in DOMAIN_FACTORIES.items():
            x = factory()
            _, weights = router(x)
            weights_mix = apply_unimix(weights)
            m = routing_metrics(weights_mix)
            domain_metrics[domain] = m

            fracs_str = " ".join(f"{f:.3f}" for f in m["fracs"])
            print(f"  {domain:6s}  H={m['norm_entropy']:.3f} [{status_emoji(m['norm_entropy'])}]"
                  f"  LIF={m['lif']:.2f}  fracs=[{fracs_str}]"
                  f"  dead={m['dead']}")

    # Cross-domain routing divergence
    all_fracs = np.stack([m["fracs"] for m in domain_metrics.values()])
    cross_std = all_fracs.std(axis=0)
    print(f"\n  Cross-domain expert std: {' '.join(f'{s:.4f}' for s in cross_std)}")
    print(f"  (Near-zero = domains route identically = healthy at init)")

    # Overall risk
    min_ent = min(m['norm_entropy'] for m in domain_metrics.values())
    if min_ent > 0.85:
        print(f"\n  Static probe: LOW RISK (min entropy {min_ent:.3f})")
    elif min_ent > 0.70:
        print(f"\n  Static probe: MEDIUM RISK (min entropy {min_ent:.3f})")
    else:
        print(f"\n  Static probe: HIGH RISK (min entropy {min_ent:.3f})")

    return domain_metrics


# ===========================================================================
# Phase 1: 200-step pretest for each mitigation method
# ===========================================================================

METHODS = {
    "A": "Baseline (unimix only, current setup)",
    "B": "Unimix + load balancing (alpha=0.01)",
    "C": "Unimix + z-loss (beta=0.001)",
    "D": "Unimix + load balancing + z-loss  [RECOMMENDED]",
    "E": "Temp anneal (2.0->1.0) + load balancing",
    "F": "Small init (std=0.001) + load balance + z-loss  [BEST PRACTICE]",
}


def run_method_pretest(method_key: str, n_steps=N_STEPS) -> dict:
    """Run 200-step pretest for a single mitigation method."""
    torch.manual_seed(SEED)

    base = MOERouterV2(input_dim=INPUT_DIM, manifold_dim=MANIFOLD_DIM)
    init_std = 0.001 if method_key == "F" else None
    init_temp = 2.0  if method_key == "E" else 1.0
    router = RouterWithAux(base, method=method_key,
                           init_std=init_std, init_temp=init_temp)

    opt = torch.optim.Adam(router.parameters(), lr=1e-3)

    history = []
    domains = list(DOMAIN_FACTORIES.keys())

    for step in range(n_steps):
        # Alternating multi-domain batches
        domain = domains[step % len(domains)]
        x = DOMAIN_FACTORIES[domain]()

        opt.zero_grad()
        manifold, weights, logits = router(x, anneal_steps=n_steps // 2)
        router.step = step

        # Task loss proxy: predict zero manifold (stand-in for JEPA loss)
        task_loss = (manifold ** 2).mean()
        aux = router.aux_loss(logits)
        loss = task_loss + aux
        loss.backward()
        opt.step()

        if step % LOG_EVERY == 0:
            with torch.no_grad():
                m = routing_metrics(weights)
                history.append({
                    "step":         step,
                    "domain":       domain,
                    "norm_entropy": m["norm_entropy"],
                    "lif":          m["lif"],
                    "cv":           m["cv"],
                    "dead":         m["dead"],
                    "max_frac":     m["max_frac"],
                    "fracs":        m["fracs"].copy(),
                    "task_loss":    task_loss.item(),
                    "aux_loss":     aux.item(),
                })

    # Final multi-domain evaluation
    router.eval()
    final_domain_metrics = {}
    with torch.no_grad():
        for domain, factory in DOMAIN_FACTORIES.items():
            x = factory()
            _, weights, logits = router(x)
            m = routing_metrics(weights)
            final_domain_metrics[domain] = m

    # Collapse verdict
    entropies  = [h["norm_entropy"] for h in history]
    min_fracs  = [h["fracs"].min() for h in history]
    final_ent  = np.mean([m["norm_entropy"] for m in final_domain_metrics.values()])
    final_dead = sum(m["dead"] for m in final_domain_metrics.values())

    # Monotone decline = entropy DROPS over first 5 checkpoints by >0.01
    monotone_decline = (
        len(entropies) >= 3 and
        entropies[0] - entropies[min(4, len(entropies)-1)] > 0.01
    )

    if final_ent < 0.70 or final_dead > 0 or monotone_decline:
        risk = "HIGH"
    elif final_ent < 0.85:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        "method":        method_key,
        "history":       history,
        "final_domains": final_domain_metrics,
        "final_entropy": final_ent,
        "final_dead":    final_dead,
        "risk":          risk,
        "monotone":      monotone_decline,
    }


def run_all_pretests():
    print("\n" + "="*65)
    print("  PHASE 1: 200-STEP PRETEST — Six mitigation methods")
    print("="*65)
    print(f"  Domains: RECON (image), SMAP (telemetry), MVTec (texture)")
    print(f"  Steps: {N_STEPS}  |  Batch: {BATCH_SIZE}  |  Experts: {NUM_EXPERTS}")
    print(f"  Input dim: {INPUT_DIM}  |  Seed: {SEED}\n")

    results = {}
    for key, desc in METHODS.items():
        t0 = time.perf_counter()
        r = run_method_pretest(key)
        elapsed = time.perf_counter() - t0
        results[key] = r

        ent = r["final_entropy"]
        risk_str = {"LOW": "LOW  ", "MEDIUM": "MED  ", "HIGH": "HIGH "}[r["risk"]]
        print(f"  [{key}] {risk_str} H={ent:.3f}  dead={r['final_dead']}  "
              f"{elapsed:.1f}s  | {desc}")

    return results


# ===========================================================================
# Phase 2: Per-domain entropy curves for the best methods
# ===========================================================================

def print_domain_breakdown(results: dict):
    print("\n" + "="*65)
    print("  PHASE 2: PER-DOMAIN EXPERT UTILISATION (final state)")
    print("="*65)

    for key in METHODS:
        r = results[key]
        print(f"\n  [{key}] {METHODS[key]}")
        print(f"       {'Domain':8s}  {'H':>6}  {'LIF':>5}  {'dead':>4}  fracs")
        for domain, m in r["final_domains"].items():
            fracs_str = " ".join(f"{f:.3f}" for f in m["fracs"])
            print(f"       {domain:8s}  {m['norm_entropy']:6.3f}  "
                  f"{m['lif']:5.2f}  {m['dead']:4d}  [{fracs_str}]")


# ===========================================================================
# Phase 3: Recommendation
# ===========================================================================

def recommend(results: dict):
    print("\n" + "="*65)
    print("  PHASE 3: RECOMMENDATION FOR SPRINT 3")
    print("="*65)

    # Score each method: lower is better
    scores = {}
    for key, r in results.items():
        ent = r["final_entropy"]
        dead = r["final_dead"]
        ent_score = max(0, 0.85 - ent) * 10    # penalty for low entropy
        dead_score = dead * 5                   # penalty per dead expert
        scores[key] = ent_score + dead_score

    ranked = sorted(scores.items(), key=lambda x: x[1])

    print("\n  Ranking (lower score = better):\n")
    for rank, (key, score) in enumerate(ranked, 1):
        r = results[key]
        risk  = r["risk"]
        ent   = r["final_entropy"]
        stars = "*** RECOMMENDED ***" if rank == 1 else ""
        print(f"  #{rank}  [{key}] score={score:.2f}  H={ent:.3f}  "
              f"risk={risk:6s}  {stars}")
        print(f"       {METHODS[key]}")

    best = ranked[0][0]
    worst = ranked[-1][0]

    print(f"\n  Best:    [{best}] -- {METHODS[best]}")
    print(f"  Current: [A]   -- {METHODS['A']}")

    if best == "A":
        print("\n  Current setup (unimix only) is sufficient.")
        print("  No changes needed before Sprint 3.")
    else:
        print(f"\n  Upgrade recommended: replace [A] with [{best}]")
        print(f"  Expected entropy improvement: "
              f"{results['A']['final_entropy']:.3f} -> {results[best]['final_entropy']:.3f}")

    print("\n  Code change for moe_router_v2.py:")
    print("""
  # In training loop (train_cwm_multidomain.py), add after computing logits:
  def moe_aux_loss(logits, alpha=0.01, beta=0.001, N=4):
      probs = torch.softmax(logits, dim=-1)
      top_idx = probs.argmax(-1)
      f_i = torch.zeros(N).scatter_add(0, top_idx,
              torch.ones(logits.shape[0])) / logits.shape[0]
      P_i = probs.mean(0)
      lb_loss = alpha * N * (f_i * P_i).sum()
      z_loss  = beta * (torch.logsumexp(logits, dim=-1)**2).mean()
      return lb_loss + z_loss

  # Add to total_loss:
  # total_loss = jepa_loss + moe_aux_loss(router_logits)
""")

    return best


# ===========================================================================
# Phase 4: Entropy trajectory plot (ASCII)
# ===========================================================================

def plot_entropy_curves(results: dict):
    print("\n" + "="*65)
    print("  PHASE 4: ENTROPY TRAJECTORY (ASCII, steps 0-200)")
    print("="*65)
    print("  Format: each row = one method, columns = steps 0,20,..200")
    print("  H values: 1.00=perfect, 0.85=healthy, 0.70=warning\n")

    print(f"  {'Method':2s}  " + "  ".join(f"s{h['step']:>3}" for h in results["A"]["history"]))
    print("  " + "-"*60)

    for key in METHODS:
        r = results[key]
        vals = [h["norm_entropy"] for h in r["history"]]
        bar = "  ".join(f"{v:.3f}" if v > 0.85 else
                        f"\033[33m{v:.3f}\033[0m" if v > 0.70 else
                        f"\033[31m{v:.3f}\033[0m"
                        for v in vals)
        print(f"  [{key}]  {bar}")

    print("\n  Green=OK(>0.85)  Yellow=WARN(0.70-0.85)  Red=CRIT(<0.70)")


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("\nCORTEX CWM -- MoE Router Collapse Risk Assessment")
    print("=" * 65)
    print(f"Current router: MOERouterV2 | 4 experts | input_dim={INPUT_DIM}")
    print(f"Current mitigation: unimix 1% only (no aux loss, no z-loss)")
    print(f"Testing: 6 mitigation methods over {N_STEPS} steps each")

    t_total = time.perf_counter()

    # Phase 0: static probe
    static_probe()

    # Phase 1: 200-step pretests
    results = run_all_pretests()

    # Phase 2: per-domain breakdown
    print_domain_breakdown(results)

    # Phase 3: recommendation
    best = recommend(results)

    # Phase 4: entropy curves
    plot_entropy_curves(results)

    elapsed = time.perf_counter() - t_total
    print(f"\n  Total runtime: {elapsed:.1f}s")
    print("="*65)
    print(f"  Run before Sprint 3 to confirm router is healthy.")
    print(f"  If [A] is not best, patch train_cwm_multidomain.py with aux loss.")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
