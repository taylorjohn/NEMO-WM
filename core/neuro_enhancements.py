"""
neuro_enhancements.py — Neuroscience-Grounded Enhancements
=============================================================
Two novel contributions for Paper 2:

1. INVERTED-U NEUROMODULATION
   Every neuromodulator has an optimal level. Too much or too little
   degrades performance. This is the most replicated finding in
   cognitive neuroscience (Arnsten 2021) but NO AI world model
   implements it.
   
   References:
   - Arnsten (2021) "Neuromodulation of PFC cognitive function"
   - Yerkes & Dodson (1908) original inverted-U
   - Goldman-Rakic (1995) DA D1 receptor inverted-U

2. BIDIRECTIONAL REPLAY
   Hippocampal replay goes both forward AND backward:
   - Forward replay: planning (what will happen next?)
   - Backward replay: credit assignment (what caused this outcome?)
   
   References:
   - Foster & Wilson (2006) "Reverse replay of behavioural sequences"
   - Diba & Buzsaki (2007) "Forward and reverse hippocampal place-cell
     sequences during ripples"

Usage:
    python neuro_enhancements.py          # full demo
    python neuro_enhancements.py --test   # run tests
"""

import argparse
import numpy as np
from typing import List, Dict

D_BELIEF = 64


class InvertedUNeuromodulator:
    """
    Neuromodulatory signals with inverted-U dose-response curves.
    
    Effectiveness = 1 - ((signal - optimal) / width)^2
    Clamped to [0, 1].
    
    Arnsten (2021): "Most neuromodulators have a narrow inverted U
    dose response, which coordinates arousal state with cognitive
    state, and contributes to cognitive deficits with fatigue or
    uncontrollable stress."
    """

    PROFILES = {
        "DA":  (0.4, 0.5, "reward_sensitivity"),
        "ACh": (0.6, 0.5, "learning_rate"),
        "NE":  (0.5, 0.45, "sensory_gain"),
        "5HT": (0.5, 0.5, "patience"),
        "CRT": (0.3, 0.4, "vigilance"),
    }

    DEFICIT_LOW = {
        "DA": "apathy, amotivation (Parkinson's-like)",
        "ACh": "poor encoding, no new learning (Alzheimer's-like)",
        "NE": "drowsiness, inattention (fatigue)",
        "5HT": "impulsivity, poor inhibition",
        "CRT": "complacency, missed threats",
    }

    DEFICIT_HIGH = {
        "DA": "distractibility, psychosis-like (schizophrenia-like)",
        "ACh": "hallucination, over-binding (confabulation)",
        "NE": "panic, tunnel vision (PTSD-like)",
        "5HT": "rigidity, perseveration (OCD-like)",
        "CRT": "paralysis, cognitive shutdown (learned helplessness)",
    }

    def __init__(self):
        self.signals = {name: 0.5 for name in self.PROFILES}

    def set(self, name, value):
        self.signals[name] = np.clip(value, 0.0, 1.0)

    def effectiveness(self, name):
        optimal, width, _ = self.PROFILES[name]
        level = self.signals[name]
        eff = 1.0 - ((level - optimal) / width) ** 2
        return float(np.clip(eff, 0.0, 1.0))

    def all_effectiveness(self):
        return {n: self.effectiveness(n) for n in self.PROFILES}

    def overall_cognitive_state(self):
        effs = [self.effectiveness(n) for n in self.PROFILES]
        return float(np.prod(effs) ** (1.0 / len(effs)))

    def diagnose(self):
        deficits = []
        for name in self.PROFILES:
            optimal, _, _ = self.PROFILES[name]
            level = self.signals[name]
            eff = self.effectiveness(name)
            if eff < 0.5:
                if level < optimal:
                    deficits.append(f"{name} too low ({level:.2f}): {self.DEFICIT_LOW[name]}")
                else:
                    deficits.append(f"{name} too high ({level:.2f}): {self.DEFICIT_HIGH[name]}")
        return deficits

    def compute_wm_capacity(self):
        da_eff = self.effectiveness("DA")
        ne_eff = self.effectiveness("NE")
        ach_eff = self.effectiveness("ACh")
        crt_level = self.signals["CRT"]
        base_k = 8
        neuro_factor = (da_eff * ne_eff * ach_eff) ** (1/3)
        stress_penalty = int(crt_level * 6)
        return max(2, int(base_k * neuro_factor) - stress_penalty)

    def compute_planning_horizon(self):
        ach_eff = self.effectiveness("ACh")
        sht_eff = self.effectiveness("5HT")
        return max(1, int(32 * ach_eff * sht_eff))

    def compute_exploration_drive(self):
        da = self.signals["DA"]
        sht = self.signals["5HT"]
        return float(np.clip(da - sht * 0.5 + 0.25, 0, 1))


class BidirectionalReplay:
    """
    Hippocampal replay in both forward and backward directions.
    
    Foster & Wilson (2006): "Reverse replay of behavioural sequences
    in hippocampal place cells during the awake state."
    """

    def __init__(self):
        self.forward_replays = 0
        self.backward_replays = 0

    def forward_replay(self, episode, transition_fn=None):
        self.forward_replays += 1
        beliefs = np.stack(episode)
        pred_errors = []
        for i in range(1, len(episode)):
            if transition_fn:
                predicted = transition_fn(episode[i-1])
                error = np.linalg.norm(predicted - episode[i])
            else:
                error = np.linalg.norm(episode[i] - episode[i-1])
            pred_errors.append(error)

        return {
            "direction": "forward",
            "steps": len(episode),
            "mean_belief": beliefs.mean(axis=0),
            "total_pred_error": sum(pred_errors),
            "peak_surprise_step": int(np.argmax(pred_errors)) if pred_errors else 0,
        }

    def backward_replay(self, episode, rewards=None, da_signals=None):
        self.backward_replays += 1
        n = len(episode)

        if rewards is None:
            rewards = [float(np.exp(-np.linalg.norm(b[:2]))) for b in episode]

        gamma = 0.95
        credits = np.zeros(n)
        credits[-1] = rewards[-1]
        for t in range(n - 2, -1, -1):
            credits[t] = rewards[t] + gamma * credits[t + 1]

        if credits.max() > credits.min():
            credit_norm = (credits - credits.min()) / (credits.max() - credits.min())
        else:
            credit_norm = np.ones(n) * 0.5

        credit_gradient = np.abs(np.diff(credits))
        critical_steps = np.argsort(credit_gradient)[-3:][::-1]

        return {
            "direction": "backward",
            "steps": n,
            "credits": credits,
            "credit_norm": credit_norm,
            "critical_steps": critical_steps.tolist(),
            "total_credit": float(credits.sum()),
            "peak_credit_step": int(np.argmax(credits)),
            "causal_chain_length": int(np.sum(credit_gradient > 0.01)),
        }

    def sleep_cycle(self, episodes, rewards_per_ep=None, da_per_ep=None):
        forward_results = []
        backward_results = []
        for i, ep in enumerate(episodes):
            rewards = rewards_per_ep[i] if rewards_per_ep else None
            da = da_per_ep[i] if da_per_ep else None
            forward_results.append(self.forward_replay(ep))
            backward_results.append(self.backward_replay(ep, rewards, da))

        return {
            "episodes_replayed": len(episodes),
            "forward_replays": len(forward_results),
            "backward_replays": len(backward_results),
            "mean_credit": np.mean([r["total_credit"] for r in backward_results]),
            "n_critical_steps": len(set(s for r in backward_results for s in r["critical_steps"])),
            "forward_pred_error": np.mean([r["total_pred_error"] for r in forward_results]),
            "causal_chains": np.mean([r["causal_chain_length"] for r in backward_results]),
        }


def demo():
    print("=" * 65)
    print("  Neuroscience-Grounded Enhancements")
    print("  1. Inverted-U Neuromodulation (Arnsten 2021)")
    print("  2. Bidirectional Replay (Foster & Wilson 2006)")
    print("=" * 65)

    neuro = InvertedUNeuromodulator()

    print("\n  Dose-response curves:")
    print(f"  {'Level':<8}", end="")
    for name in neuro.PROFILES:
        print(f"{name:>8}", end="")
    print()
    for level in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
        print(f"  {level:<8.1f}", end="")
        for name in neuro.PROFILES:
            neuro.set(name, level)
            print(f"{neuro.effectiveness(name):>8.2f}", end="")
        print()

    print(f"\n  Stress sweep:")
    print(f"  {'CRT':<6} {'WM K':>5} {'Horizon':>8} {'Deficits'}")
    for crt in [0.0, 0.3, 0.5, 0.7, 1.0]:
        neuro.set("DA", 0.4); neuro.set("ACh", 0.6)
        neuro.set("NE", 0.5); neuro.set("5HT", 0.5)
        neuro.set("CRT", crt)
        k = neuro.compute_wm_capacity()
        h = neuro.compute_planning_horizon()
        d = neuro.diagnose()
        print(f"  {crt:<6.1f} {k:>5} {h:>8} {d[0][:40] if d else 'none'}")

    print(f"\n  DA sweep (others optimal):")
    for da in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        neuro.set("DA", da); neuro.set("CRT", 0.2)
        eff = neuro.effectiveness("DA")
        bar = "█" * int(eff * 15)
        d = [x for x in neuro.diagnose() if x.startswith("DA")]
        print(f"    DA={da:.1f}  {bar:<15} {eff:.2f}  "
              f"{d[0][15:] if d else 'optimal'}")

    # Bidirectional replay
    print(f"\n{'─'*65}")
    print(f"  BIDIRECTIONAL REPLAY")
    replay = BidirectionalReplay()
    rng = np.random.RandomState(42)

    episode, rewards = [], []
    b = rng.randn(D_BELIEF).astype(np.float32) * 0.3
    for _ in range(25):
        episode.append(b.copy())
        rewards.append(float(np.exp(-np.linalg.norm(b[:2]))))
        b += rng.randn(D_BELIEF).astype(np.float32) * 0.1
        b[:2] *= 0.95

    fwd = replay.forward_replay(episode)
    bwd = replay.backward_replay(episode, rewards)

    print(f"\n  Forward: peak surprise step {fwd['peak_surprise_step']}, "
          f"error={fwd['total_pred_error']:.3f}")
    print(f"  Backward: peak credit step {bwd['peak_credit_step']}, "
          f"credit={bwd['total_credit']:.3f}")
    print(f"  Critical decisions: {bwd['critical_steps']}")
    print(f"  Causal chain: {bwd['causal_chain_length']} steps")

    print(f"\n  Credit assignment:")
    for i in range(0, len(episode), 4):
        bar = "█" * int(bwd['credit_norm'][i] * 15)
        crit = " ← CRITICAL" if i in bwd['critical_steps'] else ""
        print(f"    step {i:>3}: credit={bwd['credits'][i]:>6.3f} {bar}{crit}")

    # Sleep cycle
    eps = []
    rws = []
    for _ in range(5):
        e, r = [], []
        b = rng.randn(D_BELIEF).astype(np.float32) * 0.5
        for _ in range(20):
            e.append(b.copy()); r.append(float(np.exp(-np.linalg.norm(b[:2]))))
            b += rng.randn(D_BELIEF).astype(np.float32) * 0.1
        eps.append(e); rws.append(r)

    sr = replay.sleep_cycle(eps, rws)
    print(f"\n  Sleep cycle: {sr['forward_replays']} fwd + "
          f"{sr['backward_replays']} bwd, "
          f"credit={sr['mean_credit']:.3f}, "
          f"chains={sr['causal_chains']:.1f}")

    print(f"\n{'='*65}")
    print(f"  Novel contributions:")
    print(f"  1. Inverted-U: first AI world model with dose-response curves")
    print(f"  2. Bidirectional replay: forward (plan) + backward (credit)")
    print(f"{'='*65}")


def run_tests():
    print("=" * 65)
    print("  Neuroscience Enhancement Tests")
    print("=" * 65)
    p = 0; t = 0
    neuro = InvertedUNeuromodulator()

    print("\n  T1: Effectiveness peaks at optimal")
    neuro.set("DA", 0.4); peak = neuro.effectiveness("DA")
    neuro.set("DA", 0.0); low = neuro.effectiveness("DA")
    neuro.set("DA", 1.0); high = neuro.effectiveness("DA")
    ok = peak > low and peak > high
    print(f"    low={low:.2f} optimal={peak:.2f} high={high:.2f} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T2: Inverted-U approximately symmetric")
    neuro.set("DA", 0.2); below = neuro.effectiveness("DA")
    neuro.set("DA", 0.6); above = neuro.effectiveness("DA")
    ok = abs(below - above) < 0.15
    print(f"    DA=0.2:{below:.2f} DA=0.6:{above:.2f} diff={abs(below-above):.3f} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T3: WM degrades under stress")
    for n in ["DA","ACh","NE","5HT"]:
        neuro.set(n, [0.4,0.6,0.5,0.5][list(neuro.PROFILES).index(n)])
    neuro.set("CRT", 0.0); k0 = neuro.compute_wm_capacity()
    neuro.set("CRT", 0.5); k5 = neuro.compute_wm_capacity()
    neuro.set("CRT", 1.0); k10 = neuro.compute_wm_capacity()
    ok = k0 > k5 > k10
    print(f"    CRT=0:K={k0} CRT=0.5:K={k5} CRT=1.0:K={k10} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T4: Diagnose detects extremes")
    neuro.set("DA", 0.0); neuro.set("CRT", 0.9)
    d = neuro.diagnose()
    ok = len(d) >= 2
    print(f"    Deficits: {len(d)} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T5: No deficits at optimal")
    neuro.set("DA",0.4); neuro.set("ACh",0.6); neuro.set("NE",0.5)
    neuro.set("5HT",0.5); neuro.set("CRT",0.3)
    ok = len(neuro.diagnose()) == 0
    print(f"    Deficits: {len(neuro.diagnose())} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T6: Planning horizon from ACh × 5HT")
    neuro.set("ACh",0.6); neuro.set("5HT",0.5)
    h_opt = neuro.compute_planning_horizon()
    neuro.set("ACh",0.1)
    h_low = neuro.compute_planning_horizon()
    ok = h_opt > h_low
    print(f"    Optimal:{h_opt} low_ACh:{h_low} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    replay = BidirectionalReplay()
    rng = np.random.RandomState(42)
    ep, rw = [], []
    b = rng.randn(D_BELIEF).astype(np.float32) * 0.3
    for _ in range(20):
        ep.append(b.copy())
        rw.append(float(np.exp(-np.linalg.norm(b[:2]))))
        b += rng.randn(D_BELIEF).astype(np.float32) * 0.1
        b[:2] *= 0.95

    print("\n  T7: Forward replay prediction errors")
    fwd = replay.forward_replay(ep)
    ok = fwd["total_pred_error"] > 0
    print(f"    Error: {fwd['total_pred_error']:.3f} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T8: Backward replay credit assignment")
    bwd = replay.backward_replay(ep, rw)
    ok = len(bwd["credits"]) == len(ep) and bwd["total_credit"] > 0
    print(f"    Credits: {len(bwd['credits'])} total={bwd['total_credit']:.3f} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T9: Credit increases toward end")
    c = bwd["credits"]
    ok = c[len(c)//2:].mean() > c[:len(c)//2].mean()
    print(f"    First:{c[:len(c)//2].mean():.3f} Second:{c[len(c)//2:].mean():.3f} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T10: Critical steps identified")
    ok = len(bwd["critical_steps"]) > 0
    print(f"    Steps: {bwd['critical_steps']} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T11: Forward ≠ backward signals")
    ok = fwd["peak_surprise_step"] != bwd["peak_credit_step"]
    print(f"    Fwd peak:{fwd['peak_surprise_step']} Bwd peak:{bwd['peak_credit_step']} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T12: Sleep cycle bidirectional")
    eps, rws = [], []
    for _ in range(5):
        e, r = [], []
        b = rng.randn(D_BELIEF).astype(np.float32) * 0.5
        for _ in range(15):
            e.append(b.copy()); r.append(float(np.exp(-np.linalg.norm(b[:2]))))
            b += rng.randn(D_BELIEF).astype(np.float32) * 0.1
        eps.append(e); rws.append(r)
    sr = replay.sleep_cycle(eps, rws)
    ok = sr["forward_replays"] == 5 and sr["backward_replays"] == 5
    print(f"    Fwd:{sr['forward_replays']} Bwd:{sr['backward_replays']} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print(f"\n{'='*65}")
    print(f"  Results: {p}/{t} tests passed")
    print(f"{'='*65}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        run_tests()
    else:
        demo()
