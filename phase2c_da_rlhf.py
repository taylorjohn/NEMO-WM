# phase2c_da_rlhf.py
# Phase 2c: DA as RLHF replacement
#
# Standard RLHF:  x -> policy -> y -> reward_model(x,y) -> r -> PPO_update
# Phase 2c:       x -> encoder -> z -> DA(z,baseline) -> RPE -> REINFORCE_update
#
# DA replaces the reward model. No human annotations. No separate reward training.
# The biological RPE (DA_t - mean(DA_{t-N})) is the only training signal.
#
# Two modes:
#   --unsupervised  RPE = DA_t - rolling_mean(DA)
#                   Calibrates encoder to maintain stable anomaly distribution
#   --supervised    r = +1 correct classification, -1 miss
#                   RPE = r - rolling_mean(r)
#
# Key difference from standard RLHF:
#   - No reward model training needed
#   - No human preference labels needed
#   - DA signal IS the reward signal (biological RPE)
#   - Online learning -- updates happen during inference
#
# Usage:
#   python phase2c_da_rlhf.py --mode unsupervised --data-dir data/mvtec --category bottle
#   python phase2c_da_rlhf.py --mode supervised   --data-dir data/mvtec --steps 200
#   python phase2c_da_rlhf.py --mode compare      --data-dir data/mvtec (vs frozen encoder)
#
# No unicode -- Windows cp1252 safe
import argparse, json, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from pathlib import Path
from PIL import Image
from torchvision import transforms

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---- DA Reward Signal -------------------------------------------------------

class DARewardSignal:
    """
    Computes RPE (reward prediction error) from the DA anomaly signal.

    Biological grounding:
        DA encodes prediction error, not reward itself.
        RPE = actual_outcome - expected_outcome
        Positive RPE: frame was MORE anomalous than rolling expectation
        Negative RPE: frame was LESS anomalous than rolling expectation

    Usage (unsupervised):
        signal = DARewardSignal(window=20)
        rpe = signal.update(da_score)         # RPE from rolling DA mean

    Usage (supervised):
        signal = DARewardSignal(window=20)
        rpe = signal.update_supervised(da_score, label, threshold)
    """
    def __init__(self, window: int = 20, baseline_window: int = 50):
        self.window          = window
        self.da_history      = deque(maxlen=window)
        self.r_history       = deque(maxlen=window)
        self.baseline_da     = deque(maxlen=baseline_window)
        self._tick           = 0
        self.da_mean         = 0.5  # prior
        self.r_mean          = 0.0

    def update(self, da_score: float) -> float:
        """
        Unsupervised RPE: how surprising is this DA value given recent history?
        RPE = DA_t - mean(DA_{t-N:t})
        """
        self.da_history.append(da_score)
        self.baseline_da.append(da_score)
        self.da_mean = float(np.mean(self.da_history)) if self.da_history else 0.5
        rpe = da_score - self.da_mean
        self._tick += 1
        return float(rpe)

    def update_supervised(self, da_score: float, label: int,
                          threshold: float) -> float:
        """
        Supervised RPE: r = +1 correct, -1 wrong.
        RPE = r - mean(r over window)
        label: 1=anomaly, 0=normal
        """
        pred = 1 if da_score >= threshold else 0
        r = 1.0 if pred == label else -1.0
        self.r_history.append(r)
        self.da_history.append(da_score)
        self.r_mean = float(np.mean(self.r_history)) if self.r_history else 0.0
        rpe = r - self.r_mean
        self._tick += 1
        return float(rpe)

    @property
    def expected_da(self) -> float:
        return self.da_mean

    @property
    def expected_reward(self) -> float:
        return self.r_mean

    def status(self) -> dict:
        return {
            "tick":        self._tick,
            "da_mean":     round(self.da_mean, 4),
            "r_mean":      round(self.r_mean, 4),
            "buffer_size": len(self.da_history),
        }


# ---- REINFORCE Encoder Updater ----------------------------------------------

class REINFORCEEncoderUpdater:
    """
    REINFORCE-style update for the encoder using DA-RPE as reward signal.

    Policy gradient theorem (simplified):
        gradient = RPE * grad(log p(z | x))
        loss      = -RPE * log_prob(z)

    For a deterministic encoder, we approximate log p(z|x) via:
        log_prob(z) = -|| z - mean_z ||^2 / (2 * sigma^2)
    This is the Gaussian log-likelihood of the embedding under the
    current distribution of normal-frame embeddings.

    Positive RPE -> reduce loss -> push encoder toward current embedding
    Negative RPE -> increase loss -> push encoder away from current embedding

    In practice this means:
        - High surprise (anomaly detected correctly): reinforce the sensitivity
        - Low surprise (anomaly missed): penalise the embedding similarity
    """
    def __init__(self, encoder: torch.nn.Module, lr: float = 1e-5,
                 baseline_decay: float = 0.99, entropy_coef: float = 0.01):
        self.encoder       = encoder
        self.optimizer     = torch.optim.Adam(encoder.parameters(), lr=lr)
        self.baseline      = 0.0
        self.baseline_decay= baseline_decay
        self.entropy_coef  = entropy_coef
        self._step         = 0
        self._losses       = []

    def update(self, z: torch.Tensor, rpe: float,
               z_baseline: torch.Tensor) -> float:
        """
        One REINFORCE step.

        Args:
            z:          current embedding (1, dim)
            rpe:        reward prediction error from DARewardSignal
            z_baseline: mean embedding of normal frames (stable baseline)

        Returns:
            loss value (float)
        """
        self.encoder.train()
        self.optimizer.zero_grad()

        # Exponential moving average baseline (variance reduction)
        self.baseline = (self.baseline_decay * self.baseline
                         + (1.0 - self.baseline_decay) * rpe)
        advantage = rpe - self.baseline

        # Log-probability: cosine similarity to baseline as a proxy
        z_norm      = F.normalize(z.float(), dim=-1)
        b_norm      = F.normalize(z_baseline.float(), dim=-1)
        cos_sim     = torch.clamp((z_norm * b_norm).sum(), -1.0, 1.0)
        da_score    = 1.0 - cos_sim                     # distance from normal
        log_prob    = torch.log(da_score.clamp(1e-6, 1.0))

        # REINFORCE loss: -advantage * log_prob
        # Entropy regularisation: keeps encoder from collapsing
        entropy     = -(da_score * log_prob + (1-da_score) * torch.log((1-da_score).clamp(1e-6)))
        loss        = -advantage * log_prob - self.entropy_coef * entropy

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
        self.optimizer.step()

        self._step += 1
        self._losses.append(float(loss.item()))
        return float(loss.item())

    def mean_loss(self, n: int = 20) -> float:
        if not self._losses:
            return 0.0
        return float(np.mean(self._losses[-n:]))


# ---- Encoder loader ---------------------------------------------------------

def load_encoder(frozen: bool = False):
    """Load StudentEncoder. frozen=True for baseline comparison."""
    sys.path.insert(0, '.')
    try:
        from train_mvtec import StudentEncoder
        m = StudentEncoder()
        ckpt = torch.load(
            r'checkpoints\dinov2_student\student_best.pt',
            map_location='cpu', weights_only=False)
        m.load_state_dict(ckpt.get('model', ckpt), strict=False)
        if frozen:
            for p in m.parameters():
                p.requires_grad = False
            m.eval()
        else:
            m.train()
        return m
    except Exception as e:
        raise RuntimeError(f"StudentEncoder load failed: {e}")


def encode(model: torch.nn.Module, img: Image.Image,
           grad: bool = False) -> torch.Tensor:
    x = TRANSFORM(img).unsqueeze(0)
    if grad:
        z = model(x)
    else:
        with torch.no_grad():
            z = model(x)
    return z.squeeze(0)


# ---- Data loader ------------------------------------------------------------

def load_images(mvtec_dir: str, category: str,
                max_n: int = 60, max_a: int = 20):
    base = Path(mvtec_dir) / category
    test = base / "test"
    if not test.exists():
        test = base
    normal, anomaly = [], []
    good = test / "good"
    if good.exists():
        for f in sorted(good.glob("*.png"))[:max_n]:
            try: normal.append(Image.open(f).convert("RGB"))
            except: pass
    for dd in sorted(test.iterdir()):
        if dd.name == "good" or not dd.is_dir(): continue
        for f in list(dd.glob("*.png"))[:max(1, max_a // 5)]:
            try: anomaly.append(Image.open(f).convert("RGB"))
            except: pass
        if len(anomaly) >= max_a: break
    return normal[:max_n], anomaly[:max_a]


def build_baseline(model, normal_imgs, n_base: int = 10) -> torch.Tensor:
    zs = []
    for img in normal_imgs[:n_base]:
        zs.append(encode(model, img).detach())
    return F.normalize(torch.stack(zs).mean(0), dim=0)


def score(stable_das, shift_das):
    if not stable_das or not shift_das:
        return {"f1": 0.0, "fp": 0}
    all_p = [(d, 0) for d in stable_das] + [(d, 1) for d in shift_das]
    all_p.sort()
    best_f1, best_t = 0.0, float(np.median(stable_das))
    for t, _ in all_p:
        tp = sum(1 for d, l in all_p if d >= t and l == 1)
        fp = sum(1 for d, l in all_p if d >= t and l == 0)
        fn = sum(1 for d, l in all_p if d < t and l == 1)
        p = tp / (tp + fp + 1e-6)
        r = tp / (tp + fn + 1e-6)
        f1 = 2 * p * r / (p + r + 1e-6)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    tp = sum(1 for d in shift_das if d >= best_t)
    fp = sum(1 for d in stable_das if d >= best_t)
    fn = sum(1 for d in shift_das if d < best_t)
    p = tp / (tp + fp + 1e-6)
    r = tp / (tp + fn + 1e-6)
    f1 = 2 * p * r / (p + r + 1e-6)
    return {"f1": round(f1, 3), "fp": fp, "threshold": round(best_t, 4)}


def compute_f1(model, normal_imgs, anomaly_imgs,
               z_baseline: torch.Tensor) -> dict:
    model.eval()
    stable_das = [
        float(1.0 - torch.dot(
            F.normalize(encode(model, img), dim=0),
            z_baseline).clamp(-1, 1))
        for img in normal_imgs[10:]
    ]
    shift_das = [
        float(1.0 - torch.dot(
            F.normalize(encode(model, img), dim=0),
            z_baseline).clamp(-1, 1))
        for img in anomaly_imgs
    ]
    return score(stable_das, shift_das)


# ---- Training loops ---------------------------------------------------------

def run_unsupervised(model, normal_imgs, anomaly_imgs, steps: int,
                     lr: float, log_every: int):
    """
    Unsupervised Phase 2c: RPE = DA_t - rolling_mean(DA).
    No labels. Encoder updates to maintain calibrated DA distribution.
    """
    print("\n  Mode: UNSUPERVISED (no labels)")
    print(f"  RPE = DA_t - rolling_mean(DA, window=20)")

    z_baseline = build_baseline(model, normal_imgs, n_base=10)
    da_signal  = DARewardSignal(window=20)
    updater    = REINFORCEEncoderUpdater(model, lr=lr)

    # Mix normal (label=0) and anomaly (label=1) randomly
    all_imgs   = [(img, 0) for img in normal_imgs[10:]] + \
                 [(img, 1) for img in anomaly_imgs]
    np.random.shuffle(all_imgs)

    results = []
    for step in range(steps):
        img, label = all_imgs[step % len(all_imgs)]

        # Encode WITH gradient for update
        z = encode(model, img, grad=True)

        # DA score
        z_norm   = F.normalize(z.float(), dim=-1)
        da_score = float(1.0 - torch.dot(z_norm.detach(), z_baseline).clamp(-1, 1))

        # RPE from rolling mean (no labels used)
        rpe = da_signal.update(da_score)

        # REINFORCE update
        loss = updater.update(z.unsqueeze(0), rpe, z_baseline.unsqueeze(0))

        # Rebuild baseline periodically
        if step % 50 == 49:
            z_baseline = build_baseline(model, normal_imgs, n_base=10)

        if step % log_every == 0 or step == steps - 1:
            cal = compute_f1(model, normal_imgs, anomaly_imgs, z_baseline)
            results.append({"step": step, **cal, "loss": round(loss, 4),
                            "rpe": round(rpe, 4),
                            "da_mean": round(da_signal.da_mean, 4)})
            print(f"  step {step:>4} | F1={cal['f1']:.3f} FP={cal['fp']:>3} | "
                  f"loss={loss:+.4f} rpe={rpe:+.4f} "
                  f"da_mean={da_signal.da_mean:.4f}")

    return results


def run_supervised(model, normal_imgs, anomaly_imgs, steps: int,
                   lr: float, log_every: int):
    """
    Supervised Phase 2c: r = +1 correct, -1 wrong.
    RPE = r - rolling_mean(r). No reward model -- DA IS the reward signal.
    """
    print("\n  Mode: SUPERVISED (labels used, no reward model)")
    print(f"  RPE = correctness - rolling_mean(correctness, window=20)")

    z_baseline = build_baseline(model, normal_imgs, n_base=10)
    da_signal  = DARewardSignal(window=20)
    updater    = REINFORCEEncoderUpdater(model, lr=lr)

    # Compute initial threshold
    init_cal   = compute_f1(model, normal_imgs, anomaly_imgs, z_baseline)
    threshold  = init_cal["threshold"]

    all_imgs   = [(img, 0) for img in normal_imgs[10:]] + \
                 [(img, 1) for img in anomaly_imgs]
    np.random.shuffle(all_imgs)

    results = []
    for step in range(steps):
        img, label = all_imgs[step % len(all_imgs)]

        z      = encode(model, img, grad=True)
        z_norm = F.normalize(z.float(), dim=-1)
        da     = float(1.0 - torch.dot(z_norm.detach(), z_baseline).clamp(-1, 1))

        # Supervised RPE: correctness signal
        rpe = da_signal.update_supervised(da, label, threshold)

        loss = updater.update(z.unsqueeze(0), rpe, z_baseline.unsqueeze(0))

        if step % 50 == 49:
            z_baseline = build_baseline(model, normal_imgs, n_base=10)
            cal2 = compute_f1(model, normal_imgs, anomaly_imgs, z_baseline)
            threshold = cal2["threshold"]

        if step % log_every == 0 or step == steps - 1:
            cal = compute_f1(model, normal_imgs, anomaly_imgs, z_baseline)
            results.append({"step": step, **cal, "loss": round(loss, 4),
                            "rpe": round(rpe, 4),
                            "r_mean": round(da_signal.r_mean, 4)})
            print(f"  step {step:>4} | F1={cal['f1']:.3f} FP={cal['fp']:>3} | "
                  f"loss={loss:+.4f} rpe={rpe:+.4f} "
                  f"r_mean={da_signal.r_mean:+.4f}")

    return results


def run_compare(model_da, model_frozen,
                normal_imgs, anomaly_imgs, steps: int,
                lr: float, log_every: int):
    """
    Compare Phase 2c (DA-updated) vs frozen encoder.
    Shows whether DA-RPE updates improve classification.
    """
    print("\n  Mode: COMPARE (DA-updated vs frozen encoder)")

    z_base_da     = build_baseline(model_da,     normal_imgs, n_base=10)
    z_base_frozen = build_baseline(model_frozen, normal_imgs, n_base=10)

    da_signal = DARewardSignal(window=20)
    updater   = REINFORCEEncoderUpdater(model_da, lr=lr)

    all_imgs  = [(img, 0) for img in normal_imgs[10:]] + \
                [(img, 1) for img in anomaly_imgs]
    np.random.shuffle(all_imgs)

    results = []
    for step in range(steps):
        img, label = all_imgs[step % len(all_imgs)]
        z = encode(model_da, img, grad=True)
        z_norm = F.normalize(z.float(), dim=-1)
        da = float(1.0 - torch.dot(z_norm.detach(), z_base_da).clamp(-1, 1))
        rpe = da_signal.update(da)
        updater.update(z.unsqueeze(0), rpe, z_base_da.unsqueeze(0))

        if step % 50 == 49:
            z_base_da = build_baseline(model_da, normal_imgs, n_base=10)

        if step % log_every == 0 or step == steps - 1:
            cal_da  = compute_f1(model_da,     normal_imgs, anomaly_imgs, z_base_da)
            cal_frz = compute_f1(model_frozen, normal_imgs, anomaly_imgs, z_base_frozen)
            delta   = round(cal_da["f1"] - cal_frz["f1"], 3)
            results.append({"step": step,
                            "f1_da": cal_da["f1"], "f1_frozen": cal_frz["f1"],
                            "delta": delta})
            marker = " +" if delta > 0 else ("  " if delta == 0 else " -")
            print(f"  step {step:>4} | "
                  f"DA-RLHF F1={cal_da['f1']:.3f}  "
                  f"Frozen F1={cal_frz['f1']:.3f}  "
                  f"delta={delta:+.3f}{marker}")

    return results


# ---- Main -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",       default="supervised",
                    choices=["unsupervised","supervised","compare"])
    ap.add_argument("--data-dir",   default="data/mvtec")
    ap.add_argument("--category",   default="bottle")
    ap.add_argument("--steps",      type=int,   default=200)
    ap.add_argument("--lr",         type=float, default=1e-5)
    ap.add_argument("--log-every",  type=int,   default=20)
    ap.add_argument("--max-normal", type=int,   default=50)
    ap.add_argument("--max-anomaly",type=int,   default=20)
    a = ap.parse_args()

    print("\nNeMo-WM Phase 2c -- DA as RLHF Replacement")
    print("="*60)
    print(f"Category: {a.category} | Steps: {a.steps} | LR: {a.lr}")
    print(f"Standard RLHF: human labels -> reward_model -> PPO")
    print(f"Phase 2c:      DA signal    -> RPE           -> REINFORCE")

    print(f"\nLoading images from {a.data_dir}/{a.category}...")
    norm, anom = load_images(a.data_dir, a.category,
                             a.max_normal, a.max_anomaly)
    if len(norm) < 15:
        print(f"ERROR: only {len(norm)} normal images"); return
    if len(anom) < 3:
        print(f"ERROR: only {len(anom)} anomaly images"); return
    print(f"  Normal: {len(norm)}, Anomaly: {len(anom)}")

    print("Loading encoder...")
    try:
        if a.mode == "compare":
            model_da     = load_encoder(frozen=False)
            model_frozen = load_encoder(frozen=True)
            print("  DA-updated + frozen encoders loaded")
        else:
            model = load_encoder(frozen=False)
            print("  Encoder loaded")
    except Exception as e:
        print(f"  ERROR: {e}"); return

    # Baseline F1 before any updates
    if a.mode != "compare":
        z_b = build_baseline(model, norm, n_base=10)
        baseline_cal = compute_f1(model, norm, anom, z_b)
        print(f"\nBaseline (frozen): F1={baseline_cal['f1']:.3f} "
              f"FP={baseline_cal['fp']}")

    print(f"\n{'='*60}")
    t0 = time.perf_counter()

    if a.mode == "unsupervised":
        results = run_unsupervised(model, norm, anom,
                                   a.steps, a.lr, a.log_every)
    elif a.mode == "supervised":
        results = run_supervised(model, norm, anom,
                                 a.steps, a.lr, a.log_every)
    else:
        results = run_compare(model_da, model_frozen, norm, anom,
                              a.steps, a.lr, a.log_every)

    elapsed = time.perf_counter() - t0
    print(f"\n{'='*60}")
    print(f"Completed {a.steps} steps in {elapsed:.1f}s "
          f"({a.steps/elapsed:.1f} steps/s)")

    # Final summary
    if results:
        last = results[-1]
        if a.mode == "compare":
            print(f"\nFinal: DA-RLHF F1={last['f1_da']:.3f}  "
                  f"Frozen F1={last['f1_frozen']:.3f}  "
                  f"delta={last['delta']:+.3f}")
        else:
            print(f"\nFinal F1: {last['f1']:.3f}  FP: {last['fp']}")
            if a.mode != "compare" and len(results) > 1:
                f1_start = results[0]["f1"]
                f1_end   = last["f1"]
                delta    = round(f1_end - f1_start, 3)
                print(f"F1 change: {f1_start:.3f} -> {f1_end:.3f} "
                      f"({delta:+.3f})")

    print(f"\nPaper claim:")
    print(f"  DA-RPE replaces the reward model in RLHF.")
    print(f"  No human annotations. No reward model training.")
    print(f"  Biological RPE signal drives encoder adaptation online.")

    fname = f"phase2c_{a.mode}_{a.category}_results.json"
    with open(fname, "w") as f:
        json.dump({"config": vars(a), "results": results}, f, indent=2)
    print(f"  Saved: {fname}")


if __name__ == "__main__":
    main()
