import numpy as np, torch, json
from student_encoder import StudentEncoder
from train_predictor import SpatialChannelProjector, SpatialMLPPredictor, SPATIAL_DIM, DV, ENV_CONFIG, FRAME_TRANSFORM
from pathlib import Path

thresholds = json.load(open('./benchmark_thresholds_opt3_50pct.json'))
encoder = StudentEncoder()
ckpt = torch.load('./checkpoints/cortex_student_phase2_final.pt', map_location='cpu')
encoder.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
encoder.eval()
for p in encoder.parameters(): p.requires_grad = False

def run_env(env_name, n_mpc=8):
    config = ENV_CONFIG[env_name]
    thresh = thresholds[env_name]
    pckpt = torch.load(f'./predictors/predictor_{env_name}_opt3.pt', map_location='cpu')
    pred = SpatialMLPPredictor(SPATIAL_DIM, config['action_dim'])
    pred.load_state_dict(pckpt['state_dict']); pred.eval()
    sp = SpatialChannelProjector(32, DV)
    sp.load_state_dict(pckpt['spatial_proj_state_dict']); sp.eval()

    def enc(o):
        t = FRAME_TRANSFORM(o).unsqueeze(0)
        with torch.no_grad():
            _, s = encoder(t, return_spatial=True)
            return sp(s).squeeze(0)

    def predict(z, action):
        with torch.no_grad():
            return pred(z.unsqueeze(0), action.unsqueeze(0)).squeeze(0)

    def plan(z_s, z_g, n_cand=64, iters=3, horizon=5):
        cands = torch.randn(n_cand, horizon, config['action_dim']) * 0.1
        log_q = torch.zeros(n_cand)
        with torch.no_grad():
            for _ in range(iters):
                costs = []
                for i in range(n_cand):
                    z = z_s.clone(); c = 0.0
                    for k in range(horizon):
                        z = predict(z, cands[i,k])
                        c += torch.norm(z - z_g, p=2).item()
                    costs.append(c)
                costs = torch.tensor(costs)
                log_q = log_q + (-costs); log_q -= log_q.logsumexp(0)
                w = log_q.exp()
                elite = (w.view(-1,1,1)*cands).sum(0)
                cands = elite.unsqueeze(0) + torch.randn(n_cand, horizon, config['action_dim'])*0.05
        return cands[log_q.argmax()][0].numpy()

    trajs = np.load(f'./benchmark_data/{env_name}/trajectories.npy', allow_pickle=True)
    test = trajs[config['n_train']:config['n_train']+50]
    open_ok, mpc_ok = [], []
    np.random.seed(0); torch.manual_seed(0)

    for i, traj in enumerate(test):
        obs = traj['observations']; T = len(obs)
        if T < n_mpc + 2: continue
        s = np.random.randint(0, T - n_mpc); g = min(s + n_mpc, T-1)
        z_s = enc(obs[s]); z_g = enc(obs[g])
        z = z_s.clone()
        for _ in range(n_mpc):
            a = plan(z, z_g); z = predict(z, torch.tensor(a, dtype=torch.float32))
        open_ok.append(float(torch.norm(z-z_g,p=2).item() < thresh))
        z = z_s.clone(); hit = False
        for _ in range(n_mpc):
            if torch.norm(z-z_g,p=2).item() < thresh: hit=True; break
            a = plan(z, z_g); z = predict(z, torch.tensor(a, dtype=torch.float32))
        mpc_ok.append(float(hit))

    print(f'{env_name:8s}: open={np.mean(open_ok)*100:.1f}%  mpc={np.mean(mpc_ok)*100:.1f}%  (thresh={thresh:.2f})')
    return np.mean(open_ok)*100, np.mean(mpc_ok)*100

print('Running spatial benchmark (option 3, 50pct thresholds)...')
print()
results = {}
for env in ['wall', 'umaze', 'medium']:
    o, m = run_env(env)
    results[env] = {'open': o, 'mpc': m}

print()
print('Summary vs DINO-WM MPC baseline:')
baselines = {'wall': 76.7, 'umaze': 80.7, 'medium': 76.7}
for env in ['wall','umaze','medium']:
    m = results[env]['mpc']
    b = baselines[env]
    print(f'  {env:8s}: CORTEX-16={m:.1f}%  DINO-WM={b:.1f}%  gap={m-b:+.1f}')
