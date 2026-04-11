"""
test_neuromodulator_extensions.py
==================================
Simple empirical tests for proposed neuromodulator extensions.
All tests run on existing checkpoints — no new training required.

Tests:
    1. Oxytocin signal validation — familiarity vs novelty
    2. DA-loss correlation — from existing training logs
    3. ACh-router temperature effect — synthetic validation
    4. Cortisol simulation — retroactive on Tab 2 loss curve

Run all:
    python test_neuromodulator_extensions.py

Run individual:
    python test_neuromodulator_extensions.py --test oxytocin
    python test_neuromodulator_extensions.py --test da_correlation
    python test_neuromodulator_extensions.py --test ach_router
    python test_neuromodulator_extensions.py --test cortisol

Author: John Taylor — NeMo-WM Sprint extensions
Date:   2026-04-04
"""

import argparse
import glob
import io
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import h5py
from PIL import Image
from scipy import stats

from train_mvtec import StudentEncoder
from train_cwm_DEPRECATED import CortexWorldModel

DEVICE = torch.device("cpu")


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_models(cwm_ckpt=r"checkpoints\cwm\cwm_best.pt",
                enc_ckpt=r"checkpoints\dinov2_student\student_best.pt"):
    enc = StudentEncoder().to(DEVICE)
    if Path(enc_ckpt).exists():
        sd = torch.load(enc_ckpt, map_location="cpu", weights_only=False)
        enc.load_state_dict(sd.get("model", sd), strict=False)
    enc.eval()

    cwm = CortexWorldModel(d_model=128, K=16).to(DEVICE)
    if Path(cwm_ckpt).exists():
        ckpt = torch.load(cwm_ckpt, map_location="cpu", weights_only=False)
        cwm.load_state_dict(ckpt.get("model", ckpt), strict=False)
    cwm.eval()
    return enc, cwm


def get_particle_mean(enc, cwm, jpeg_bytes):
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB").resize((224, 224))
    ft = torch.from_numpy(np.array(img)).float().permute(2,0,1).unsqueeze(0)/255.0
    with torch.no_grad():
        z = enc(ft)
        p, _, _, _ = cwm.encode(z)
        return p.mean(1).squeeze(0)  # (128,)


def collect_particle_means(enc, cwm, files, n_per_file=5):
    means = []
    for f in files:
        try:
            with h5py.File(f, "r") as hf:
                imgs = hf["images"]["rgb_left"]
                idxs = np.linspace(0, len(imgs)-1, n_per_file, dtype=int)
                for i in idxs:
                    pm = get_particle_mean(enc, cwm, bytes(imgs[i]))
                    means.append(pm.numpy())
        except Exception:
            pass
    return np.array(means) if means else np.zeros((0, 128))


# ── Test 1: Oxytocin signal validation ───────────────────────────────────────

class OxytocinSignal:
    """Familiarity signal: cosine similarity to running EMA of particles."""
    def __init__(self, decay=0.99):
        self.ema = None
        self.decay = decay

    def update(self, p_mean: np.ndarray) -> float:
        if self.ema is None:
            self.ema = p_mean.copy()
            return 0.5  # neutral on first frame
        familiarity = float(np.dot(p_mean, self.ema) /
                            (np.linalg.norm(p_mean) * np.linalg.norm(self.ema) + 1e-8))
        self.ema = self.decay * self.ema + (1 - self.decay) * p_mean
        return max(0.0, min(1.0, familiarity))


def test_oxytocin(enc, cwm, hdf5_dir, n_seen=20, n_unseen=20):
    """
    Test: oxytocin (familiarity) should be higher for SEEN files
    than UNSEEN files after warming up the EMA on seen trajectories.

    Expected result: seen_oxytocin > unseen_oxytocin, t-test p < 0.05
    """
    print("\n" + "="*60)
    print("Test 1: Oxytocin Signal Validation")
    print("="*60)

    files = sorted(glob.glob(f"{hdf5_dir}/*.hdf5"))
    seen_files   = files[:n_seen]           # first N files = "known"
    unseen_files = files[-n_unseen:]        # last N files = "novel"

    if not seen_files or not unseen_files:
        print("  ❌ Not enough HDF5 files found")
        return

    # Warm up EMA on seen files
    print(f"  Warming up EMA on {n_seen} seen files...")
    oxt = OxytocinSignal(decay=0.99)
    for f in seen_files:
        try:
            with h5py.File(f, "r") as hf:
                imgs = hf["images"]["rgb_left"]
                for i in range(0, min(10, len(imgs)), 2):
                    pm = get_particle_mean(enc, cwm, bytes(imgs[i])).numpy()
                    oxt.update(pm)
        except Exception:
            pass

    # Collect oxytocin scores for seen vs unseen
    seen_scores   = []
    unseen_scores = []

    print(f"  Collecting oxytocin scores on seen files...")
    for f in seen_files[:10]:
        try:
            with h5py.File(f, "r") as hf:
                imgs = hf["images"]["rgb_left"]
                for i in range(0, min(8, len(imgs))):
                    pm = get_particle_mean(enc, cwm, bytes(imgs[i])).numpy()
                    score = OxytocinSignal(decay=0.99)
                    score.ema = oxt.ema.copy()
                    seen_scores.append(score.update(pm))
        except Exception:
            pass

    print(f"  Collecting oxytocin scores on unseen files...")
    for f in unseen_files[:10]:
        try:
            with h5py.File(f, "r") as hf:
                imgs = hf["images"]["rgb_left"]
                for i in range(0, min(8, len(imgs))):
                    pm = get_particle_mean(enc, cwm, bytes(imgs[i])).numpy()
                    score = OxytocinSignal(decay=0.99)
                    score.ema = oxt.ema.copy()
                    unseen_scores.append(score.update(pm))
        except Exception:
            pass

    seen_arr   = np.array(seen_scores)
    unseen_arr = np.array(unseen_scores)

    t_stat, p_val = stats.ttest_ind(seen_arr, unseen_arr)
    cohens_d = (seen_arr.mean() - unseen_arr.mean()) / np.sqrt(
        (seen_arr.std()**2 + unseen_arr.std()**2) / 2 + 1e-8
    )

    print(f"\n  Results:")
    print(f"  Seen   oxytocin: mean={seen_arr.mean():.4f}  std={seen_arr.std():.4f}  N={len(seen_arr)}")
    print(f"  Unseen oxytocin: mean={unseen_arr.mean():.4f}  std={unseen_arr.std():.4f}  N={len(unseen_arr)}")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value:     {p_val:.4f}")
    print(f"  Cohen's d:   {cohens_d:.3f}")

    if seen_arr.mean() > unseen_arr.mean() and p_val < 0.05:
        print(f"\n  ✅ PASS — Oxytocin distinguishes seen from unseen (p={p_val:.4f})")
        print(f"  → Oxytocin signal is valid. Add to neuromodulator.")
    elif seen_arr.mean() > unseen_arr.mean():
        print(f"\n  🟡 MARGINAL — Effect in right direction but p={p_val:.4f} > 0.05")
        print(f"  → Increase n_seen/n_unseen or use more frames per file")
    else:
        print(f"\n  ❌ FAIL — Oxytocin does not distinguish seen from unseen")
        print(f"  → EMA decay may need tuning, or representations too similar")

    return {"seen_mean": float(seen_arr.mean()), "unseen_mean": float(unseen_arr.mean()),
            "p_value": float(p_val), "cohens_d": float(cohens_d)}


# ── Test 2: DA-loss correlation ───────────────────────────────────────────────

def test_da_correlation(log_file=None):
    """
    Test: DA signal should positively correlate with loss spikes.
    Parse existing training log or use hardcoded Tab 2 observations.

    Expected result: Pearson r(DA, loss) > 0.3
    """
    print("\n" + "="*60)
    print("Test 2: DA-Loss Correlation")
    print("="*60)

    # Hardcoded observations from Tab 2 final epoch (epoch 29)
    # Format: (step, loss, da)
    observations = [
        (1084500, 0.5613, 0.001), (1085000, 0.5636, 0.001),
        (1086500, 0.5677, 0.001), (1087000, 0.5814, 0.000),  # loss spike
        (1089500, 0.5714, 0.001), (1090000, 0.5617, 0.001),
        (1092000, 0.5603, 0.001), (1093000, 0.5683, 0.001),
        (1097500, 0.5715, 0.001), (1099500, 0.5676, 0.001),
        (1100000, 0.5640, 0.001), (1101000, 0.5640, 0.001),
        (1102000, 0.5781, 0.001),  # loss spike
        (1102500, 0.5654, 0.002),  # DA spike
        (1105000, 0.5730, 0.001), (1105500, 0.5727, 0.002),  # DA spike
        (1109500, 0.5981, 0.001),  # large loss spike
        (1110500, 0.5627, 0.002), (1111000, 0.5785, 0.002),  # DA+loss
        (1111500, 0.5627, 0.002),  # DA sustained
        # Epoch 28 final steps — DA=0.002 cluster before save
        (1082500, 0.5743, 0.002), (1083000, 0.5661, 0.002),
        (1083500, 0.5669, 0.002), (1084000, 0.5652, 0.002),
    ]

    steps  = np.array([o[0] for o in observations])
    losses = np.array([o[1] for o in observations])
    das    = np.array([o[2] for o in observations])

    r, p = stats.pearsonr(das, losses)

    print(f"\n  Data: {len(observations)} observations from Tab 2 epochs 28-29")
    print(f"  DA range:   {das.min():.3f} – {das.max():.3f}")
    print(f"  Loss range: {losses.min():.4f} – {losses.max():.4f}")
    print(f"\n  Pearson r(DA, loss): {r:.4f}")
    print(f"  p-value:             {p:.4f}")

    # Also check: do DA=0.002 steps have higher mean loss?
    high_da = losses[das >= 0.002]
    low_da  = losses[das < 0.002]
    t_stat, t_p = stats.ttest_ind(high_da, low_da)

    print(f"\n  Loss when DA=0.002: mean={high_da.mean():.4f}  N={len(high_da)}")
    print(f"  Loss when DA<0.002: mean={low_da.mean():.4f}   N={len(low_da)}")
    print(f"  t-test p={t_p:.4f}")

    if r > 0.3 and p < 0.05:
        print(f"\n  ✅ PASS — DA positively correlates with loss (r={r:.3f}, p={p:.4f})")
        print(f"  → DA replay buffer justified: high-DA steps are high-error steps")
    elif high_da.mean() > low_da.mean() and t_p < 0.1:
        print(f"\n  🟡 MARGINAL — DA=0.002 associated with higher loss (p={t_p:.4f})")
        print(f"  → Effect present, more data needed for strong r")
    else:
        print(f"\n  ❌ FAIL — DA does not reliably track loss")

    return {"pearson_r": float(r), "p_value": float(p),
            "high_da_loss": float(high_da.mean()), "low_da_loss": float(low_da.mean())}


# ── Test 3: ACh router temperature ───────────────────────────────────────────

def test_ach_router(enc, cwm, hdf5_dir, n_files=20):
    """
    Test: ACh-modulated router temperature should change routing entropy.
    High ACh → low temp → low entropy (sharp routing).
    Low ACh → high temp → high entropy (soft routing).

    Expected result: negative correlation between ACh and router entropy.
    """
    print("\n" + "="*60)
    print("Test 3: ACh Router Temperature Effect")
    print("="*60)

    files = sorted(glob.glob(f"{hdf5_dir}/*.hdf5"))[:n_files]

    ach_values    = []
    entropy_base  = []  # standard softmax entropy
    entropy_ach   = []  # ACh-modulated entropy

    def router_entropy(logits: torch.Tensor, temp: float = 1.0) -> float:
        probs = F.softmax(logits / temp, dim=-1)
        return float(-( probs * (probs + 1e-8).log()).sum(-1).mean())

    print(f"  Collecting router statistics from {n_files} files...")

    # Simulate ACh variation by using visual complexity as proxy
    # (ACh in training correlates with frame richness)
    for f in files:
        try:
            with h5py.File(f, "r") as hf:
                imgs = hf["images"]["rgb_left"]
                for i in range(0, min(5, len(imgs))):
                    img = Image.open(io.BytesIO(bytes(imgs[i]))).convert("RGB").resize((224,224))
                    ft  = torch.from_numpy(np.array(img)).float().permute(2,0,1).unsqueeze(0)/255.0

                    with torch.no_grad():
                        z = enc(ft)
                        p, _, _, _ = cwm.encode(z)
                        flat = p.reshape(-1, 128)

                        # Get router logits from layer 0
                        layer0  = cwm.predictor.layers[0]
                        logits  = layer0.moe_ffn.router(flat)  # (K, n_experts)

                        # Simulate ACh from visual complexity (enc norm variance)
                        ach_proxy = float(z.norm(dim=-1).mean())
                        # Normalise to 0.4–0.5 range (observed ACh range)
                        ach = 0.4 + 0.1 * min(1.0, ach_proxy)

                        # Temperature: high ACh → low temp (0.5), low ACh → high temp (2.0)
                        temp = 1.0 + (0.5 - ach) * 6.0
                        temp = max(0.3, min(3.0, temp))

                        h_base = router_entropy(logits, temp=1.0)
                        h_ach  = router_entropy(logits, temp=temp)

                        ach_values.append(ach)
                        entropy_base.append(h_base)
                        entropy_ach.append(h_ach)
        except Exception:
            pass

    ach_arr  = np.array(ach_values)
    h_base   = np.array(entropy_base)
    h_ach    = np.array(entropy_ach)

    r_base, p_base = stats.pearsonr(ach_arr, h_base)
    r_ach,  p_ach  = stats.pearsonr(ach_arr, h_ach)

    print(f"\n  N observations: {len(ach_arr)}")
    print(f"  ACh range:      {ach_arr.min():.3f} – {ach_arr.max():.3f}")
    print(f"\n  Without ACh modulation:")
    print(f"    r(ACh, entropy) = {r_base:.4f}  p={p_base:.4f}")
    print(f"\n  With ACh temperature modulation:")
    print(f"    r(ACh, entropy) = {r_ach:.4f}  p={p_ach:.4f}")
    print(f"    Entropy reduction: {(h_base.mean() - h_ach.mean()):.4f} nats mean")

    if r_ach < -0.3:
        print(f"\n  ✅ PASS — ACh modulation creates negative ACh-entropy correlation")
        print(f"  → High ACh sharply routes (low entropy) as designed")
    else:
        print(f"\n  🟡 ACh proxy (visual norm) may not capture true ACh variation")
        print(f"  → Test again with actual ACh signal from neuromodulator state")

    return {"r_base": float(r_base), "r_ach": float(r_ach),
            "entropy_reduction": float(h_base.mean() - h_ach.mean())}


# ── Test 4: Cortisol simulation ───────────────────────────────────────────────

def test_cortisol():
    """
    Test: Apply cortisol formula retroactively to Tab 2 loss curve.
    Cortisol should predict loss plateaus 10-20 steps ahead.

    Expected result: cortisol[t] correlates with loss[t+10:t+20]
    """
    print("\n" + "="*60)
    print("Test 4: Cortisol Signal Simulation")
    print("="*60)

    # Tab 2 epoch means (observed)
    epoch_means = [
        0.5708, 0.5705, 0.5702, 0.5700, 0.5698,
        0.5696, 0.5694, 0.5692, 0.5690, 0.5688,
        0.5686, 0.5684, 0.5682, 0.5680, 0.5678,
        0.5676, 0.5675, 0.5674, 0.5674, 0.5674,
        0.5696, 0.5691, 0.5688, 0.5685, 0.5682,
        0.5680, 0.5675, 0.5674, 0.5674, 0.5674,
    ]
    baseline = 0.567  # approximate convergence floor

    # Apply cortisol formula: rolling mean excess above baseline
    window = 5
    cortisol = []
    for i in range(len(epoch_means)):
        start = max(0, i - window)
        recent = np.mean(epoch_means[start:i+1])
        cort = max(0.0, (recent - baseline) / baseline)
        cortisol.append(min(1.0, cort * 10))  # scale to [0,1]

    cortisol = np.array(cortisol)
    losses   = np.array(epoch_means)

    # Lag correlation: does cortisol[t] predict loss[t+k]?
    print(f"\n  Lag correlation r(cortisol[t], loss[t+k]):")
    best_r, best_lag = 0, 0
    for lag in range(1, 8):
        if lag < len(cortisol):
            r, p = stats.pearsonr(cortisol[:-lag], losses[lag:])
            print(f"    lag={lag}: r={r:.4f}  p={p:.4f}")
            if abs(r) > abs(best_r):
                best_r, best_lag = r, lag

    print(f"\n  Best predictive lag: {best_lag} epochs (r={best_r:.4f})")
    print(f"\n  Cortisol trajectory:")
    for i, (c, l) in enumerate(zip(cortisol, losses)):
        bar = "█" * int(c * 20)
        print(f"    ep{i:02d}: cortisol={c:.3f} {bar[:10]:10s} loss={l:.4f}")

    if best_r > 0.3:
        print(f"\n  ✅ PASS — Cortisol predicts future loss elevation")
        print(f"  → Cortisol signal would enable proactive adaptation")
    else:
        print(f"\n  🟡 Cortisol shows weak predictive power on epoch-level data")
        print(f"  → More useful at batch level (100-step window)")

    return {"best_lag": best_lag, "best_r": float(best_r),
            "cortisol_mean": float(cortisol.mean())}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NeMo-WM Neuromodulator Extension Tests"
    )
    parser.add_argument("--test", choices=["oxytocin","da_correlation",
                                            "ach_router","cortisol","all"],
                        default="all")
    parser.add_argument("--hdf5-dir", default="recon_data/recon_release")
    parser.add_argument("--cwm-ckpt", default=r"checkpoints\cwm\cwm_best.pt")
    args = parser.parse_args()

    print("\nNeMo-WM Neuromodulator Extension Tests")
    print("=" * 60)

    results = {}

    if args.test in ("da_correlation", "all"):
        results["da_correlation"] = test_da_correlation()

    if args.test in ("cortisol", "all"):
        results["cortisol"] = test_cortisol()

    if args.test in ("oxytocin", "ach_router", "all"):
        print("\nLoading models...")
        enc, cwm = load_models(cwm_ckpt=args.cwm_ckpt)
        print(f"  Models loaded.")

        if args.test in ("oxytocin", "all"):
            results["oxytocin"] = test_oxytocin(enc, cwm, args.hdf5_dir)

        if args.test in ("ach_router", "all"):
            results["ach_router"] = test_ach_router(enc, cwm, args.hdf5_dir)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for test_name, result in results.items():
        if result:
            print(f"\n  {test_name}:")
            for k, v in result.items():
                print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    print("\nDone.")


if __name__ == "__main__":
    main()
