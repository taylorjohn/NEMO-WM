# aphasia_eval.py  — run from C:\Users\MeteorAI\Desktop\CORTEX\
import glob, io, torch, torch.nn.functional as F
import h5py
from PIL import Image
from torchvision import transforms
from neuro_vlm_gate import BiologicalNeuromodulator, NeurallyGatedVLM

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

import sys; sys.path.insert(0, '.')
from train_mvtec import StudentEncoder

def load_frames(hdf5_dir, n=30):
    frames = []
    for path in sorted(glob.glob(f"{hdf5_dir}/*.hdf5"))[:10]:
        with h5py.File(path) as hf:
            imgs = hf["images"]["rgb_left"]
            for i in range(min(5, len(imgs))):
                img = Image.open(io.BytesIO(bytes(imgs[i]))).convert("RGB")
                frames.append(TRANSFORM(img).unsqueeze(0))
                if len(frames) >= n: return frames
    return frames

def run_condition(frames, ablation: bool):
    model = StudentEncoder()
    ckpt  = torch.load(r'checkpoints\dinov2_student\student_best.pt',
                       map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt.get('model', ckpt), strict=False)
    model.eval()

    neuro = BiologicalNeuromodulator(da_threshold=0.0613)
    gated = NeurallyGatedVLM(model, neuro, aphasia_ablation=ablation)

    z_prev, errors = None, []
    for x in frames:
        z = gated.encode(x)
        if z_prev is not None:
            rpe = float(1.0 - torch.dot(z, z_prev).clamp(-1, 1))
            errors.append(rpe)
        neuro.update_from_error(z_prev, z)
        z_prev = z

    state = neuro.get_state()
    return {
        "mean_rpe":  round(sum(errors) / len(errors), 6) if errors else 0,
        "final_da":  state["da"],
        "final_sht": state["sht"],
        "final_ei":  state["ei"],
        "regime":    state["regime"],
    }

frames = load_frames("recon_data/recon_release", n=30)
print(f"Loaded {len(frames)} frames\n")

full    = run_condition(frames, ablation=False)
aphasia = run_condition(frames, ablation=True)

print(f"{'Condition':<20} {'mean_RPE':>10} {'DA':>8} {'5HT':>8} {'E/I':>8} {'Regime'}")
print("-" * 68)
print(f"{'Full model':<20} {full['mean_rpe']:>10.6f} {full['final_da']:>8.4f} {full['final_sht']:>8.4f} {full['final_ei']:>8.4f} {full['regime']}")
print(f"{'Aphasia ablation':<20} {aphasia['mean_rpe']:>10.6f} {aphasia['final_da']:>8.4f} {aphasia['final_sht']:>8.4f} {aphasia['final_ei']:>8.4f} {aphasia['regime']}")
print(f"\nRPE delta: {aphasia['mean_rpe'] - full['mean_rpe']:+.6f}")
print(f"\nInterpretation:")
print(f"  RPE delta ≈ 0  → VLM gate not load-bearing → aphasia claim supported")
print(f"  RPE delta >> 0 → VLM gate contributes → report as VLM-dependent finding")