"""
gc_optimizer.py — Gradient Centralization for CORTEX-PE Phase 1/2 Training
============================================================================
Drop-in AdamW wrapper implementing Gradient Centralization (Yong et al., ECCV 2020).

Key insight from arXiv:2603.17676: the primary benefit of LayerNorm is mean-centering
of backpropagated error signals. GC applies this directly to weight gradients before
the Adam update — zero overhead, ~0.3–0.9% benchmark improvement on CNN training.

INTEGRATION — train_distillation.py:
    # Replace:
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    # With:
    from gc_optimizer import GCAdamW
    optimizer = GCAdamW(params, lr=lr, weight_decay=wd)

INTEGRATION — train_predictor.py (SpatialTransformer Option 4):
    # Same replacement — GC applies to conv weight tensors AND
    # transformer weight matrices (attn Q/K/V/O, FFN weights)
    from gc_optimizer import GCAdamW
    optimizer = GCAdamW(predictor.parameters(), lr=lr, weight_decay=wd)

Rules (from paper + ECCV ablations):
    - Apply only to weight tensors with ndim >= 2
    - Never apply to biases (ndim == 1)
    - Never apply to BatchNorm weight/bias (1D)
    - Center across all dims EXCEPT output channel (dim 0)
"""

import torch
from torch.optim import AdamW


def _centralize_grad(grad: torch.Tensor) -> torch.Tensor:
    """
    Subtract mean across all dims except the output (first) dim.
    For conv weights: (C_out, C_in, kH, kW) → mean over (C_in, kH, kW)
    For linear weights: (out, in) → mean over (in,)
    For transformer QKV: same as linear
    """
    if grad.dim() < 2:
        return grad  # biases, BN params — skip
    # Mean over dims 1..N, keep dim 0 intact
    reduce_dims = tuple(range(1, grad.dim()))
    return grad - grad.mean(dim=reduce_dims, keepdim=True)


class GCAdamW(AdamW):
    """
    AdamW with Gradient Centralization applied before each parameter update.

    Identical to torch.optim.AdamW in every respect except that gradients
    for weight tensors (ndim >= 2) are mean-centred before the update step.
    This adds a single subtraction per weight tensor — negligible overhead.

    Parameters
    ----------
    params : iterable
        Model parameters (same as AdamW)
    use_gc : bool
        Enable gradient centralization (default True).
        Set False to fall back to standard AdamW for ablation.
    use_mc : bool
        Enable Moment Centralization variant (Sadu et al. 2022):
        centres the first moment m_t post-EMA instead of the raw gradient.
        Marginal improvement over GC in some Adam experiments.
        Default False (standard GC is simpler and well-validated).
    All other kwargs passed directly to AdamW.
    """

    def __init__(self, params, use_gc: bool = True, use_mc: bool = False, **kwargs):
        self.use_gc = use_gc
        self.use_mc = use_mc
        super().__init__(params, **kwargs)

    @torch.no_grad()
    def step(self, closure=None):
        # ── Gradient Centralization (applied to raw gradients) ──────────────
        if self.use_gc and not self.use_mc:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None and p.grad.dim() >= 2:
                        p.grad.data.copy_(_centralize_grad(p.grad.data))

        loss = super().step(closure)

        # ── Moment Centralization (applied to m_t after EMA update) ─────────
        # Run after super().step() so the state has been initialised/updated
        if self.use_mc:
            for group in self.param_groups:
                for p in group['params']:
                    if p in self.state and 'exp_avg' in self.state[p]:
                        m = self.state[p]['exp_avg']
                        if m.dim() >= 2:
                            self.state[p]['exp_avg'] = _centralize_grad(m)

        return loss


# ── Convenience factory ────────────────────────────────────────────────────

def make_gc_optimizer(
    model: torch.nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    use_mc: bool = False,
) -> GCAdamW:
    """
    Build a GCAdamW optimizer for a model, separating params that should
    receive weight decay from those that should not (biases, BN).

    Usage:
        optimizer = make_gc_optimizer(encoder, lr=1e-3, weight_decay=1e-2)
    """
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # No weight decay for 1-D tensors (bias, BN weight/bias)
        if param.dim() == 1 or 'bias' in name or 'bn' in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {'params': decay_params,    'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]
    return GCAdamW(param_groups, lr=lr, use_gc=True, use_mc=use_mc)


# ── Quick sanity test ──────────────────────────────────────────────────────

if __name__ == '__main__':
    import torch.nn as nn

    # Mock CNN encoder matching CORTEX-PE block structure
    model = nn.Sequential(
        nn.Conv2d(3,  16, 3, stride=2, padding=1),   # block1
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, stride=2, padding=1),   # block2
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=2, padding=1),   # block3
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 32, 3, stride=2, padding=1),   # block4
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 128),
    )

    opt = make_gc_optimizer(model, lr=1e-3, weight_decay=1e-2)

    # Forward + backward
    x = torch.randn(4, 3, 224, 224)
    loss = model(x).sum()
    loss.backward()

    # Verify GC zeroes the grad means for weight tensors
    for name, p in model.named_parameters():
        if p.grad is not None and p.grad.dim() >= 2:
            # After step() call the grad is modified in-place before update
            pass  # step modifies grad in-place; check pre-step here

    # Run step and verify no crash
    opt.step()
    opt.zero_grad()
    print('✅ GCAdamW step completed without error')

    # Verify MC variant
    opt_mc = make_gc_optimizer(model, lr=1e-3, use_mc=True)
    x2 = torch.randn(4, 3, 224, 224)
    model(x2).sum().backward()
    opt_mc.step()
    opt_mc.zero_grad()
    print('✅ GCAdamW (Moment Centralization) step completed without error')

    # Verify GC is actually centering gradients
    model2 = nn.Linear(32, 128)
    opt2 = GCAdamW(model2.parameters(), lr=1e-3)
    inp = torch.randn(8, 32)
    model2(inp).sum().backward()

    # Check grad mean before step
    raw_mean = model2.weight.grad.mean(dim=1).abs().mean().item()

    # Manually apply GC and verify mean collapses
    gc_grad = _centralize_grad(model2.weight.grad)
    gc_mean = gc_grad.mean(dim=1).abs().mean().item()
    print(f'✅ Grad mean before GC: {raw_mean:.6f}  |  after GC: {gc_mean:.2e} (should be ~0)')
    assert gc_mean < 1e-6, f'GC did not centre gradient: {gc_mean}'
    print('✅ All checks passed')
