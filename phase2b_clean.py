# phase2b_clean.py - Clean Phase 2b with correct cortisol baseline
# No unicode characters, no encoding issues
import glob, io, numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def make_shift(size=224):
    img = Image.new("RGB", (size, size))
    d = ImageDraw.Draw(img)
    r = lambda v: int(np.clip(v + np.random.randint(-15, 15), 0, 255))
    d.rectangle([0,0,size,size], fill=(r(30),r(30),r(30)))
    for i in range(0, size, np.random.randint(30,50)):
        d.rectangle([i,0,i+np.random.randint(10,25),size], fill=(r(60),r(60),r(60)))
    ox, oy = np.random.randint(30,80), np.random.randint(50,100)
    d.rectangle([ox,oy,ox+size//2,oy+size//3], fill=(r(200),r(80),0))
    return img

def load_recon(hdf5_dir, n=120):
    import h5py
    frames = []
    for path in sorted(glob.glob(f"{hdf5_dir}/*.hdf5")):
        if len(frames) >= n: break
        try:
            with h5py.File(path) as hf:
                imgs = hf["images"]["rgb_left"]
                for i in range(min(10, len(imgs))):
                    if len(frames) >= n: break
                    img = Image.open(io.BytesIO(bytes(imgs[i]))).convert("RGB")
                    frames.append(img)
        except: pass
    return frames

def load_dino():
    import sys; sys.path.insert(0, '.')
    from train_mvtec import StudentEncoder
    m = StudentEncoder()
    ckpt = torch.load(r'checkpoints\dinov2_student\student_best.pt',
                      map_location='cpu', weights_only=False)
    m.load_state_dict(ckpt.get('model', ckpt), strict=False)
    m.eval()
    return m

def encode(model, img):
    with torch.no_grad():
        z = model(TRANSFORM(img).unsqueeze(0))
    return F.normalize(z.squeeze(0).float(), dim=0)

def micro_finetune(model, recent, n=10, lr=1e-5):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(n):
        if len(recent) < 2: break
        i = np.random.randint(0, len(recent)-1)
        z1 = F.normalize(model(TRANSFORM(recent[i]).unsqueeze(0)).squeeze(0), dim=0)
        z2 = F.normalize(model(TRANSFORM(recent[i+1]).unsqueeze(0)).squeeze(0), dim=0)
        loss = (1.0 - torch.dot(z1,z2)) + F.relu(0.3 - z1.var())
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
    model.eval()

def run(model, stable_frames, shift_step, n_steps, mode,
        periodic_interval=20, da_thresh=0.0613):
    
    # Build frame sequence
    all_frames = stable_frames[:shift_step] + \
                 [make_shift() for _ in range(n_steps - shift_step)]
    
    # Calibrate cortisol baseline on first 20 stable frames
    cal_losses = []
    z_cal = encode(model, all_frames[0])
    for img in all_frames[1:21]:
        z = encode(model, img)
        cal_losses.append(float(F.mse_loss(z_cal, z).item()))
        z_cal = z
    
    mu  = float(np.mean(cal_losses))
    std = float(np.std(cal_losses)) + 1e-8
    # Baseline = mean + 3*std, hard floor at 0.005
    baseline = max(mu + 3*std, 0.005)
    print(f"    Cortisol baseline: {baseline:.6f} (mu={mu:.6f} std={std:.6f})")
    
    log = []
    adaptations = []
    recent = []
    z_prev = None
    cooldown = 0
    loss_hist = []
    
    for i, img in enumerate(all_frames[:n_steps]):
        domain = "stable" if i < shift_step else "shift"
        z = encode(model, img)
        recent.append(img)
        if len(recent) > 30: recent.pop(0)
        
        da, cort, adapted = 0.0, 0.0, False
        
        if z_prev is not None:
            da  = float(1.0 - torch.dot(z, z_prev).clamp(-1,1))
            mse = float(F.mse_loss(z, z_prev).item())
            loss_hist.append(mse)
            
            # Z-score cortisol
            if len(loss_hist) >= 3:
                recent_loss = float(np.mean(loss_hist[-3:]))
                z_score = (recent_loss - baseline) / std
                cort = float(np.clip(z_score / 5.0, 0, 1))
            
            if cooldown > 0:
                cooldown -= 1
            
            # Adaptation decision
            if mode == "periodic" and i % periodic_interval == 0 and i > 21:
                micro_finetune(model, recent)
                adaptations.append({"step": i, "domain": domain})
                adapted = True
                cooldown = 10
            elif mode == "cortisol" and cort > 0.5 and i > 21 and cooldown == 0:
                micro_finetune(model, recent)
                adaptations.append({"step": i, "domain": domain})
                adapted = True
                cooldown = 10
        
        log.append({"step":i, "domain":domain, "da":round(da,4),
                    "cort":round(cort,4), "adapted":adapted})
        z_prev = z.detach()
    
    stable_log = [r for r in log if r["domain"]=="stable"]
    shift_log  = [r for r in log if r["domain"]=="shift"]
    fp = [a for a in adaptations if a["domain"]=="stable"]
    sp = [a for a in adaptations if a["domain"]=="shift"]
    
    rec = None
    for r in shift_log:
        if r["da"] < da_thresh and rec is None:
            rec = r["step"] - shift_step
    
    return {
        "mode": mode,
        "n_adapt": len(adaptations),
        "false_pos": len(fp),
        "shift_adapt": len(sp),
        "stable_da": round(float(np.mean([r["da"] for r in stable_log])), 4),
        "shift_da":  round(float(np.mean([r["da"] for r in shift_log])), 4),
        "shift_peak":round(float(np.max([r["da"] for r in shift_log])) if shift_log else 0, 4),
        "recovery":  rec,
        "log": log,
    }

print("\nPhase 2b - Cortisol-Triggered Continual Adaptation (clean)")
print("="*60)

hdf5_dir = "recon_data/recon_release"
shift_step, n_steps = 100, 200

print("Loading DINOv2...")
stable = load_recon(hdf5_dir, 120)
print(f"Loaded {len(stable)} stable frames")
print(f"\nStable phase: steps 0-{shift_step}")
print(f"Shift phase:  steps {shift_step}-{n_steps} (industrial)\n")

results = []
for mode in ["static", "periodic", "cortisol"]:
    print(f"[{mode.upper()}]", end=" ", flush=True)
    from train_mvtec import StudentEncoder
    m = StudentEncoder()
    ckpt = torch.load(r'checkpoints\dinov2_student\student_best.pt',
                      map_location='cpu', weights_only=False)
    m.load_state_dict(ckpt.get('model', ckpt), strict=False)
    m.eval()
    r = run(m, stable, shift_step, n_steps, mode)
    results.append(r)
    print(f"done - {r['n_adapt']} adaptations, {r['false_pos']} false+")

print(f"\n{'='*60}")
print(f"  {'Mode':<10} {'False+':>7} {'Adapt':>6} {'Shift DA':>9} {'Peak DA':>8} {'Recovery':>10}")
print(f"  {'-'*55}")
for r in results:
    rec = f"{r['recovery']}steps" if r['recovery'] is not None else "never"
    print(f"  {r['mode']:<10} {r['false_pos']:>7} {r['n_adapt']:>6} "
          f"{r['shift_da']:>9.4f} {r['shift_peak']:>8.4f} {rec:>10}")

print(f"\n  DA TIMELINE (every 10 steps):")
print(f"  {'Step':>5} {'Domain':<10} {'DA':>7} {'Cort':>7}")
print(f"  {'-'*35}")
ref_log = results[0]["log"]
for r in ref_log[::10]:
    print(f"  {r['step']:>5} {r['domain']:<10} {r['da']:>7.4f} {r['cort']:>7.4f}")

# Verdict
static   = next(r for r in results if r["mode"]=="static")
cortisol = next(r for r in results if r["mode"]=="cortisol")
periodic = next(r for r in results if r["mode"]=="periodic")

print(f"\n{'='*60}")
print("  VERDICT")
print(f"{'='*60}")
print(f"  Cortisol false positives: {cortisol['false_pos']} "
      f"vs periodic: {periodic['false_pos']}")
print(f"  Cortisol shift DA: {cortisol['shift_da']:.4f} "
      f"vs static: {static['shift_da']:.4f}")

if (cortisol['false_pos'] <= periodic['false_pos'] and
    cortisol['shift_da'] <= static['shift_da'] + 0.01):
    print("\n  CORTISOL WINS: targeted adaptation with controlled false positives")
elif cortisol['false_pos'] < periodic['false_pos']:
    print("\n  PARTIAL WIN: fewer false positives than periodic")
else:
    print("\n  NEEDS TUNING: check cortisol threshold or baseline")
