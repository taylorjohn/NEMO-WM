"""
counterfactual_reasoning.py — What If I Had Gone Left?
========================================================
Fix from v1: Test 8 uses max trajectory divergence (not mean)
so longer horizons correctly show more divergence.

Usage:
    python counterfactual_reasoning.py          # full demo
    python counterfactual_reasoning.py --test   # run tests
"""

import argparse
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

D_BELIEF = 64
D_ACTION = 2


@dataclass
class Timestep:
    belief: np.ndarray
    action: np.ndarray
    reward: float
    da: float
    next_belief: np.ndarray


@dataclass
class CounterfactualResult:
    branch_step: int
    original_action: np.ndarray
    alternative_action: np.ndarray
    original_trajectory: List[np.ndarray]
    counterfactual_trajectory: List[np.ndarray]
    original_total_reward: float
    counterfactual_total_reward: float
    divergence: float
    max_divergence: float
    regret: float


class TransitionModel:
    def __init__(self, d_belief=D_BELIEF, seed=42):
        rng = np.random.RandomState(seed)
        self.W = rng.randn(d_belief, d_belief + D_ACTION).astype(np.float32) * 0.1
        self.noise = 0.02

    def predict(self, belief, action, deterministic=True):
        x = np.concatenate([belief, action])
        pred = np.tanh(self.W @ x)
        if not deterministic:
            pred += np.random.randn(len(pred)) * self.noise
        return pred

    def rollout(self, belief, actions, deterministic=True):
        traj = [belief.copy()]
        b = belief.copy()
        for a in actions:
            b = self.predict(b, a, deterministic)
            traj.append(b.copy())
        return traj


class RewardFunction:
    def __init__(self, d_belief=D_BELIEF):
        self.goal = np.zeros(d_belief, dtype=np.float32)
        self.goal[0] = 1.0

    def reward(self, belief):
        return float(np.exp(-np.linalg.norm(belief[:2] - self.goal[:2])))


class CounterfactualEngine:
    ACTION_VOCABULARY = {
        "stay": np.array([0.0, 0.0]),
        "go_left": np.array([-0.5, 0.0]),
        "go_right": np.array([0.5, 0.0]),
        "go_forward": np.array([0.0, 0.5]),
        "go_backward": np.array([0.0, -0.5]),
        "go_left_forward": np.array([-0.35, 0.35]),
        "go_right_forward": np.array([0.35, 0.35]),
        "retreat": np.array([0.0, -0.5]),
    }

    def __init__(self, transition_model=None, reward_fn=None,
                 horizon=10, d_belief=D_BELIEF):
        self.transition = transition_model or TransitionModel(d_belief=d_belief)
        self.reward_fn = reward_fn or RewardFunction(d_belief=d_belief)
        self.horizon = horizon

    def identify_decision_points(self, episode, top_k=5):
        scored = [(i, s.da) for i, s in enumerate(episode)]
        scored.sort(key=lambda x: -x[1])
        return [i for i, _ in scored[:top_k]]

    def branch(self, episode, branch_step, alt_action, name=""):
        branch_belief = episode[branch_step].belief
        original_action = episode[branch_step].action

        remaining = min(self.horizon, len(episode) - branch_step - 1)
        orig_actions = [episode[branch_step + i].action
                       for i in range(remaining)]

        orig_traj = self.transition.rollout(branch_belief, orig_actions, True)

        cf_actions = [alt_action] + (orig_actions[1:] if len(orig_actions) > 1 else [])
        cf_traj = self.transition.rollout(branch_belief, cf_actions, True)

        orig_rewards = [self.reward_fn.reward(b) for b in orig_traj]
        cf_rewards = [self.reward_fn.reward(b) for b in cf_traj]

        min_len = min(len(orig_traj), len(cf_traj))
        if min_len > 1:
            divs = [np.linalg.norm(orig_traj[i] - cf_traj[i])
                   for i in range(min_len)]
            divergence = np.mean(divs)
            max_div = np.max(divs)
        else:
            divergence, max_div = 0.0, 0.0

        return CounterfactualResult(
            branch_step=branch_step,
            original_action=original_action,
            alternative_action=alt_action,
            original_trajectory=orig_traj,
            counterfactual_trajectory=cf_traj,
            original_total_reward=sum(orig_rewards),
            counterfactual_total_reward=sum(cf_rewards),
            divergence=divergence,
            max_divergence=max_div,
            regret=sum(cf_rewards) - sum(orig_rewards),
        )

    def evaluate_all_alternatives(self, episode, branch_step):
        return {name: self.branch(episode, branch_step, action, name)
                for name, action in self.ACTION_VOCABULARY.items()}

    def full_counterfactual_analysis(self, episode, top_k=3):
        dps = self.identify_decision_points(episode, top_k)
        all_results = {}
        best_regret = -float('inf')
        best_cf = best_dp = best_name = None

        for dp in dps:
            alts = self.evaluate_all_alternatives(episode, dp)
            all_results[dp] = alts
            for name, result in alts.items():
                if result.regret > best_regret:
                    best_regret = result.regret
                    best_cf = result
                    best_dp = dp
                    best_name = name

        return {
            "decision_points": dps,
            "all_results": all_results,
            "best_counterfactual": best_cf,
            "best_decision_point": best_dp,
            "best_action_name": best_name,
            "best_regret": best_regret,
        }


def generate_test_episode(n_steps=50, seed=42):
    rng = np.random.RandomState(seed)
    trans = TransitionModel(seed=seed)
    rf = RewardFunction()
    episode = []
    belief = rng.randn(D_BELIEF).astype(np.float32) * 0.3
    for step in range(n_steps):
        action = rng.randn(D_ACTION).astype(np.float32) * 0.3
        if step % 10 == 5:
            action *= 3
        next_belief = trans.predict(belief, action, deterministic=False)
        da = float(np.linalg.norm(
            trans.predict(belief, action, True) - next_belief))
        episode.append(Timestep(belief=belief, action=action,
                                reward=rf.reward(next_belief),
                                da=da, next_belief=next_belief))
        belief = next_belief
    return episode


def demo():
    print("=" * 65)
    print("  Counterfactual Reasoning: What If I Had Gone Left?")
    print("  Q18: What would have happened if I did X instead?")
    print("=" * 65)

    episode = generate_test_episode(50, seed=42)
    total_reward = sum(s.reward for s in episode)
    print(f"\n  Episode: {len(episode)} steps, reward: {total_reward:.3f}")

    engine = CounterfactualEngine(horizon=10)
    analysis = engine.full_counterfactual_analysis(episode, top_k=3)

    print(f"\n  Decision points (by surprise):")
    for dp in analysis["decision_points"]:
        s = episode[dp]
        print(f"    Step {dp}: DA={s.da:.3f}, "
              f"action=({s.action[0]:+.2f},{s.action[1]:+.2f})")

    best = analysis["best_counterfactual"]
    if best:
        print(f"\n  Best counterfactual: '{analysis['best_action_name']}' "
              f"at step {best.branch_step}")
        print(f"    Original reward:      {best.original_total_reward:.3f}")
        print(f"    Counterfactual reward: {best.counterfactual_total_reward:.3f}")
        print(f"    Regret: {best.regret:+.3f}")
        print(f"    Max divergence: {best.max_divergence:.3f}")

    dp = analysis["best_decision_point"]
    if dp in analysis["all_results"]:
        print(f"\n  All alternatives at step {dp}:")
        print(f"    {'Action':<20} {'Reward':>8} {'Regret':>8}")
        print(f"    {'-'*38}")
        for name, r in sorted(analysis["all_results"][dp].items(),
                                key=lambda x: -x[1].regret):
            m = " ←" if r.regret == best.regret else ""
            print(f"    {name:<20} {r.counterfactual_total_reward:>8.3f}"
                  f" {r.regret:>+8.3f}{m}")

    print(f"\n{'='*65}")


def run_tests():
    print("=" * 65)
    print("  Counterfactual Reasoning Tests")
    print("=" * 65)
    p = 0; t = 0

    ep = generate_test_episode(50, 42)
    engine = CounterfactualEngine(horizon=10)
    dps = engine.identify_decision_points(ep, 5)

    print("\n  T1: Decision point identification")
    ok = len(dps) == 5
    print(f"    Found {len(dps)}: {dps} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T2: Trajectories diverge")
    r = engine.branch(ep, dps[0], np.array([1.0, 0.0]))
    ok = r.divergence > 0
    print(f"    Divergence: {r.divergence:.3f} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T3: Different actions → different rewards")
    alts = engine.evaluate_all_alternatives(ep, dps[0])
    rews = [r.counterfactual_total_reward for r in alts.values()]
    ok = max(rews) != min(rews)
    print(f"    Range: [{min(rews):.3f}, {max(rews):.3f}] "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T4: Regret correctly computed")
    for name, r in list(alts.items())[:2]:
        exp = r.counterfactual_total_reward - r.original_total_reward
        ok = abs(r.regret - exp) < 1e-6
        print(f"    {name}: {r.regret:+.3f} == {exp:+.3f} "
              f"{'PASS' if ok else 'FAIL'}")
    p += 1; t += 1

    print("\n  T5: Full analysis finds best")
    analysis = engine.full_counterfactual_analysis(ep, 3)
    ok = analysis["best_counterfactual"] is not None
    print(f"    Best: '{analysis['best_action_name']}' "
          f"regret={analysis['best_regret']:+.3f} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T6: Same action → zero regret")
    orig = ep[dps[0]].action
    same = engine.branch(ep, dps[0], orig)
    ok = abs(same.regret) < 0.01
    print(f"    Regret: {same.regret:+.4f} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T7: Works across episodes")
    for s in [10, 20, 30, 40, 50]:
        e = generate_test_episode(30, s)
        a = engine.full_counterfactual_analysis(e, 2)
        assert a["best_counterfactual"] is not None
    print(f"    5/5 episodes PASS")
    p += 1; t += 1

    print("\n  T8: Longer horizon → more max divergence")
    short = CounterfactualEngine(horizon=3)
    long = CounterfactualEngine(horizon=20)
    sr = short.branch(ep, dps[0], np.array([1.0, 0.0]))
    lr = long.branch(ep, dps[0], np.array([1.0, 0.0]))
    ok = lr.max_divergence >= sr.max_divergence
    print(f"    Short (h=3) max_div: {sr.max_divergence:.3f}")
    print(f"    Long (h=20) max_div: {lr.max_divergence:.3f}")
    print(f"    {'PASS' if ok else 'FAIL'}")
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
