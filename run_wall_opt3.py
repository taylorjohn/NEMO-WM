import torch, numpy as np
from pathlib import Path
from train_predictor import SpatialChannelProjector, SpatialMLPPredictor, SPATIAL_DIM, DV, ENV_CONFIG, FRAME_TRANSFORM
from student_encoder import StudentEncoder

encoder_path = './checkpoints/cortex_student_phase2_final.pt'
pred_path    = './predictors/predictor_wall_opt3.pt'
data_root    = './benchmark_data'
thresh       = 13.01

encoder = StudentEncoder()
ckpt = torch.load(encoder_path, map_location='cpu')
encoder.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
encoder.eval()
for p in encoder.parameters(): p.requires_grad = False

pckpt = torch.load(pred_path, map_location='cpu')
pred = SpatialMLPPredictor(SPATIAL_DIM, 2)
pred.load_state_dict(pckpt['state_dict'])
pred.eval()
sp = SpatialChannelProjector(32, DV)
sp.load_state_dict(pckpt['spatial_proj_state_dict'])
sp.eval()

def encode(obs):
    t = FRAME_TRANSFORM(obs).unsqueeze(0)
    with torch.no_grad():
        _, spatial = encoder(t, return_spatial=True)
        return sp(spatial).squeeze(0)

def predict(z, action):
    with torch.no_grad():
        return pred(z.unsqueeze(0), action.unsqueeze(0)).squeeze(0)

def plan(z_start, z_goal, n_cand=64, iters=3, horizon=1):
    cands = torch.randn(n_cand, horizon, 2) * 0.1
    log_q = torch.zeros(n_cand)
    with torch.no_grad():
        for _ in range(iters):
            costs = []
            for i in range(n_cand):
                z = z_start.clone(); c = 0.0
                for k in range(horizon):
                    z = predict(z, cands[i,k])
                    c += torch.norm(z - z_goal, p=2).item()
                costs.append(c)
            costs = torch.tensor(costs)
            log_q = log_q + (-costs); log_q -= log_q.logsumexp(0)
            w = log_q.exp()
            elite = (w.view(-1,1,1)*cands).sum(0)
            cands = elite.unsqueeze(0) + torch.randn(n_cand, horizon, 2)*0.05
    return cands[log_q.argmax()][0].numpy()

config = ENV_CONFIG['wall']
trajs = np.load(Path(data_root)/'wall'/'trajectories.npy', allow_pickle=True)
test  = trajs[config['n_train']:config['n_train']+50]

open_ok, mpc_ok = [], []
np.random.seed(0); torch.manual_seed(0)
for i, traj in enumerate(test):
    obs = traj['observations']; T = len(obs)
    if T < 10: continue
    s = np.random.randint(0, T//2); g = min(s + 40, T-1)
    z_s = encode(obs[s]); z_g = encode(obs[g])
    # open loop
    z = z_s.clone()
    for _ in range(8):
        a = plan(z, z_g); z = predict(z, torch.tensor(a, dtype=torch.float32))
    open_ok.append(float(torch.norm(z-z_g,p=2).item() < thresh))
    # mpc
    z = z_s.clone(); hit = False
    for _ in range(8):
        if torch.norm(z-z_g,p=2).item() < thresh: hit=True; break
        a = plan(z, z_g); z = predict(z, torch.tensor(a, dtype=torch.float32))
    mpc_ok.append(float(hit))
    if (i+1)%10==0: print(f'{i+1}/50 open={np.mean(open_ok)*100:.1f}% mpc={np.mean(mpc_ok)*100:.1f}%')

print(f'WALL opt3: open={np.mean(open_ok)*100:.1f}%  mpc={np.mean(mpc_ok)*100:.1f}%')
print(f'DINO-WM baseline: mpc=76.7%')



