"""
run_curvature.py — measure latent trajectory curvature on real PNG frames
Run from CORTEX root:
    python run_curvature.py --encoder ./checkpoints/maze_weak_sigreg_straight/cortex_student_phase2_final.pt --traj-dir ./phase2_frames
"""
import argparse, torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def curvature(traj):
    """Mean cosine similarity of consecutive difference vectors. 1=straight, 0=random."""
    if len(traj) < 3: return float('nan')
    t = torch.stack(traj).float()
    d = t[1:] - t[:-1]
    d = torch.nn.functional.normalize(d, dim=1)
    return float((d[:-1] * d[1:]).sum(dim=1).mean())

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder",  required=True)
    p.add_argument("--traj-dir", default="./phase2_frames")
    p.add_argument("--n-traj",   type=int, default=20,
                   help="Number of consecutive-frame trajectories to sample")
    p.add_argument("--traj-len", type=int, default=25,
                   help="Frames per trajectory")
    args = p.parse_args()

    device = "cpu"
    from student_encoder import StudentEncoder
    enc = StudentEncoder()
    state = torch.load(args.encoder, map_location=device)
    if isinstance(state, dict):
        state = state.get("model_state_dict", state.get("student", state))
    enc.load_state_dict(state, strict=False)
    enc.eval()
    print(f"Encoder loaded")

    frames = sorted(Path(args.traj_dir).glob("frame_*.png"))
    if not frames:
        frames = sorted(Path(args.traj_dir).glob("*.png"))
    print(f"Found {len(frames)} PNG frames in {args.traj_dir}")

    if len(frames) < args.traj_len:
        print("Not enough frames"); return

    # Extract all latents once
    print("Encoding frames...")
    latents = []
    with torch.no_grad():
        for f in frames[:args.n_traj * args.traj_len]:
            img = TRANSFORM(Image.open(f).convert("RGB")).unsqueeze(0)
            latents.append(enc(img).squeeze(0).cpu())
    print(f"Encoded {len(latents)} frames")

    # Split into trajectories
    curvs = []
    for i in range(args.n_traj):
        traj = latents[i*args.traj_len:(i+1)*args.traj_len]
        if len(traj) == args.traj_len:
            curvs.append(curvature(traj))

    mean_c = np.mean(curvs)
    std_c  = np.std(curvs)

    print(f"\n{'='*50}")
    print(f"  Trajectories:     {len(curvs)}")
    print(f"  Curvature mean:   {mean_c:.4f}  (1.0=perfectly straight)")
    print(f"  Curvature std:    {std_c:.4f}")
    print(f"  Target:           > 0.5")
    print(f"  Result:           {'✅ PASS' if mean_c > 0.5 else '⚠️  BELOW TARGET'}")
    print(f"{'='*50}")
    print(f"\nInterpretation:")
    if mean_c > 0.8:
        print("  Excellent — near-linear latent trajectories confirm")
        print("  temporal straightening is working on real data.")
    elif mean_c > 0.5:
        print("  Good — straightening is working, some curvature remains.")
    else:
        print("  Low — encoder trajectories are still curved.")
        print("  Consider more straightening regularization.")

if __name__ == "__main__":
    main()
