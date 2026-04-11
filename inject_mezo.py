src = open('run_benchmark.py', encoding='utf-8').read()

mezo_code = '''
# =============================================================================
# MeZO planners — Memory-Efficient Zero-Order optimisation
# =============================================================================
MEZO_SIGMA          = 0.1
MEZO_LR             = 0.05
MEZO_STEPS          = 50
MEZO_MOMENTUM_BETA  = 0.9
MEZO_MOMENTUM_STEPS = 60
MEZO_POP_K          = 20
MEZO_POP_ROUNDS     = 3
MEZO_POP_REFINE     = 5

def _rollout_cost(actions, predictor, z_start, z_goal, option,
                  z_hist=None, a_hist=None, horizon=PLAN_HORIZON):
    z   = z_start.clone().unsqueeze(0)
    z_g = z_goal.flatten()
    cost = 0.0
    for t in range(horizon):
        act = actions[t].unsqueeze(0)
        if option == 1:
            z = predictor(z, act, gamma=1.0)
        elif option == 3:
            z = predictor(z, act)
        cost += torch.norm(z.reshape(-1) - z_g, p=2).item()
    return cost

def mezo_plan(predictor, z_start, z_goal, action_dim, option,
              horizon=PLAN_HORIZON, n_steps=MEZO_STEPS,
              sigma=MEZO_SIGMA, lr=MEZO_LR, z_hist=None, a_hist=None):
    actions = torch.zeros(horizon, action_dim)
    with torch.no_grad():
        for _ in range(n_steps):
            eps   = torch.randn_like(actions)
            c_pos = _rollout_cost(actions + sigma*eps, predictor, z_start, z_goal, option, z_hist, a_hist, horizon)
            c_neg = _rollout_cost(actions - sigma*eps, predictor, z_start, z_goal, option, z_hist, a_hist, horizon)
            actions = actions - lr * ((c_pos - c_neg) / (2*sigma)) * eps
    return actions[0].numpy()

def mezo_warm_plan(predictor, z_start, z_goal, action_dim, option,
                   horizon=PLAN_HORIZON, z_hist=None, a_hist=None,
                   mezo_steps=30, sigma=MEZO_SIGMA, lr=MEZO_LR):
    best_first = mirror_ascent_plan(predictor, z_start, z_goal, action_dim, option,
                                    horizon=horizon, z_hist=z_hist, a_hist=a_hist)
    actions = torch.tensor(best_first, dtype=torch.float32).unsqueeze(0).expand(horizon, -1).clone()
    with torch.no_grad():
        for _ in range(mezo_steps):
            eps   = torch.randn_like(actions)
            c_pos = _rollout_cost(actions + sigma*eps, predictor, z_start, z_goal, option, z_hist, a_hist, horizon)
            c_neg = _rollout_cost(actions - sigma*eps, predictor, z_start, z_goal, option, z_hist, a_hist, horizon)
            actions = actions - lr * ((c_pos - c_neg) / (2*sigma)) * eps
    return actions[0].numpy()

def mezo_momentum_plan(predictor, z_start, z_goal, action_dim, option,
                       horizon=PLAN_HORIZON, z_hist=None, a_hist=None,
                       n_steps=MEZO_MOMENTUM_STEPS, sigma=MEZO_SIGMA,
                       lr=0.03, beta=MEZO_MOMENTUM_BETA):
    actions  = torch.zeros(horizon, action_dim)
    momentum = torch.zeros_like(actions)
    with torch.no_grad():
        for step in range(n_steps):
            eps      = torch.randn_like(actions)
            c_pos    = _rollout_cost(actions + sigma*eps, predictor, z_start, z_goal, option, z_hist, a_hist, horizon)
            c_neg    = _rollout_cost(actions - sigma*eps, predictor, z_start, z_goal, option, z_hist, a_hist, horizon)
            g        = ((c_pos - c_neg) / (2*sigma)) * eps
            momentum = beta * momentum + (1-beta) * g
            m_hat    = momentum / (1 - beta**(step+1))
            actions  = actions - lr * m_hat
    return actions[0].numpy()

def mezo_cem_plan(predictor, z_start, z_goal, action_dim, option,
                  horizon=PLAN_HORIZON, z_hist=None, a_hist=None,
                  K=MEZO_POP_K, rounds=MEZO_POP_ROUNDS,
                  n_refine=MEZO_POP_REFINE, sigma=MEZO_SIGMA,
                  lr=0.05, elite_frac=0.5):
    n_elite = max(1, int(K * elite_frac))
    pop     = torch.randn(K, horizon, action_dim) * 0.1
    with torch.no_grad():
        for _ in range(rounds):
            for k in range(K):
                for _ in range(n_refine):
                    eps    = torch.randn(horizon, action_dim)
                    c_pos  = _rollout_cost(pop[k] + sigma*eps, predictor, z_start, z_goal, option, z_hist, a_hist, horizon)
                    c_neg  = _rollout_cost(pop[k] - sigma*eps, predictor, z_start, z_goal, option, z_hist, a_hist, horizon)
                    pop[k] = pop[k] - lr * ((c_pos - c_neg) / (2*sigma)) * eps
            costs      = torch.tensor([_rollout_cost(pop[k], predictor, z_start, z_goal, option, z_hist, a_hist, horizon) for k in range(K)])
            elite_idx  = costs.argsort()[:n_elite]
            elite_mean = pop[elite_idx].mean(0)
            elite_std  = pop[elite_idx].std(0).clamp(min=0.02)
            non_mask   = torch.ones(K, dtype=torch.bool)
            non_mask[elite_idx] = False
            n_non = non_mask.sum().item()
            pop[non_mask] = elite_mean.unsqueeze(0) + elite_std.unsqueeze(0) * torch.randn(n_non, horizon, action_dim)
        costs = torch.tensor([_rollout_cost(pop[k], predictor, z_start, z_goal, option, z_hist, a_hist, horizon) for k in range(K)])
    return pop[costs.argmin()][0].numpy()

'''

# Insert before mirror_ascent_plan definition
anchor = 'def mirror_ascent_plan('
assert anchor in src, 'anchor not found'
src2 = src.replace(anchor, mezo_code + anchor, 1)

import ast
ast.parse(src2)
open('run_benchmark.py', 'w', encoding='utf-8').write(src2)

for fn in ['mezo_plan','mezo_warm_plan','mezo_momentum_plan','mezo_cem_plan','_rollout_cost']:
    print(fn, 'OK' if ('def ' + fn) in src2 else 'MISSING')
print(f'Total lines: {src2.count(chr(10))}')
