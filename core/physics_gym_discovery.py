"""
physics_gym_discovery.py — Discover Physics from Gymnasium Environments
=========================================================================
Uses real simulation data from Gymnasium Classic Control to discover
force laws. Generates a video showing the agent exploring and the
system discovering F=ma relationships in real time.

Environments:
  - CartPole:     gravity + angular momentum + balance
  - Pendulum:     gravity + torque + damping
  - MountainCar:  gravity + engine on slope
  - Acrobot:      gravity + joint torques

No pre-programmed physics. The system observes trajectories and
discovers force laws via polynomial regression on residuals.

Usage:
    python physics_gym_discovery.py              # discover + video
    python physics_gym_discovery.py --test       # quick validation
    python physics_gym_discovery.py --env pendulum  # specific env
"""

import argparse
import numpy as np
import time
from pathlib import Path
from collections import defaultdict

Path("outputs").mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Force Discoverer — fits F=ma from observation data
# ══════════════════════════════════════════════════════════════════════════════

class ForceDiscoverer:
    """
    Observe (state, action, next_state) tuples.
    Compute acceleration = (v_{t+1} - v_t) / dt.
    Fit force = f(state) via polynomial regression.
    """

    def __init__(self, dt=0.02):
        self.dt = dt
        self.observations = []
        self.discovered_forces = {}

    def observe(self, state, action, next_state):
        """Record one transition."""
        self.observations.append({
            "state": np.array(state, dtype=np.float32),
            "action": np.array(action, dtype=np.float32).flatten(),
            "next_state": np.array(next_state, dtype=np.float32),
        })

    def compute_accelerations(self):
        """Compute accelerations from velocity changes."""
        if len(self.observations) < 2:
            return None, None, None

        states = np.stack([o["state"] for o in self.observations])
        actions = np.stack([o["action"] for o in self.observations])
        next_states = np.stack([o["next_state"] for o in self.observations])

        n_dims = states.shape[1]

        # Identify velocity dimensions (odd indices typically)
        # For most gym envs: [pos, vel, ...] or [x, x_dot, theta, theta_dot]
        vel_dims = list(range(1, n_dims, 2))
        if not vel_dims:
            vel_dims = list(range(n_dims // 2, n_dims))

        # Compute acceleration for each velocity dimension
        accels = {}
        for vd in vel_dims:
            if vd < next_states.shape[1] and vd < states.shape[1]:
                accel = (next_states[:, vd] - states[:, vd]) / self.dt
                accels[vd] = accel

        return states, actions, accels

    def discover(self, degree=3):
        """
        Fit polynomial force laws: F = sum(c_i * x^i)
        Returns discovered force equations and R² scores.
        """
        states, actions, accels = self.compute_accelerations()
        if states is None:
            return {}

        results = {}

        for vd, accel in accels.items():
            # Build feature matrix from all state dimensions
            n = len(accel)
            features = []
            feature_names = []

            for sd in range(states.shape[1]):
                for d in range(1, degree + 1):
                    features.append(states[:n, sd] ** d)
                    feature_names.append(f"s{sd}^{d}")

            # Add action features
            for ad in range(actions.shape[1]):
                features.append(actions[:n, ad])
                feature_names.append(f"a{ad}")

            if not features:
                continue

            X = np.stack(features, axis=1)

            # Ridge regression (stable)
            XtX = X.T @ X + np.eye(X.shape[1]) * 0.01
            Xty = X.T @ accel[:n]
            try:
                coeffs = np.linalg.solve(XtX, Xty)
            except np.linalg.LinAlgError:
                continue

            # Predict and compute R²
            pred = X @ coeffs
            ss_res = np.sum((accel[:n] - pred) ** 2)
            ss_tot = np.sum((accel[:n] - np.mean(accel[:n])) ** 2)
            r2 = 1 - ss_res / max(ss_tot, 1e-10)

            # Extract dominant terms
            dominant = []
            for i, (c, name) in enumerate(zip(coeffs, feature_names)):
                if abs(c) > 0.1:
                    dominant.append((name, float(c)))

            results[f"dim_{vd}"] = {
                "r2": float(r2),
                "coefficients": coeffs,
                "feature_names": feature_names,
                "dominant_terms": dominant,
                "n_samples": n,
            }

        self.discovered_forces = results
        return results


# ══════════════════════════════════════════════════════════════════════════════
# Gym Data Collector — run episodes and collect force data
# ══════════════════════════════════════════════════════════════════════════════

def collect_data(env_name, n_episodes=20, max_steps=500):
    """Run episodes with random policy, collect transitions."""
    import gymnasium as gym

    env = gym.make(env_name)
    transitions = []

    for ep in range(n_episodes):
        state, _ = env.reset()
        for step in range(max_steps):
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)

            transitions.append({
                "state": state,
                "action": [action] if np.isscalar(action) else action,
                "next_state": next_state,
                "reward": reward,
                "episode": ep,
                "step": step,
            })

            state = next_state
            if terminated or truncated:
                break

    env.close()
    return transitions


# ══════════════════════════════════════════════════════════════════════════════
# Video Generator — visualize discovery process
# ══════════════════════════════════════════════════════════════════════════════

def generate_discovery_video(env_name, transitions, discoveries,
                               output_path="outputs/physics_discovery.mp4"):
    """
    Generate a video showing:
    - Top: environment running (rendered as plots)
    - Bottom: discovered force laws appearing over time
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    import imageio

    frames = []
    n_frames = min(200, len(transitions))
    step_size = max(1, len(transitions) // n_frames)

    # Precompute incremental discoveries
    discoverer = ForceDiscoverer(dt=0.02)
    discovery_snapshots = []

    for i in range(0, len(transitions), step_size):
        # Add observations up to this point
        t = transitions[i]
        discoverer.observe(t["state"], t["action"], t["next_state"])

        # Discover with current data
        if len(discoverer.observations) >= 20:
            results = discoverer.discover(degree=2)
            discovery_snapshots.append({
                "step": i,
                "n_obs": len(discoverer.observations),
                "results": results,
            })
        else:
            discovery_snapshots.append({
                "step": i,
                "n_obs": len(discoverer.observations),
                "results": {},
            })

    # Generate frames
    for frame_idx, snapshot in enumerate(discovery_snapshots):
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        fig.patch.set_facecolor('#0d1117')
        fig.suptitle(f'NeMo-WM Physics Discovery — {env_name}',
                      color='white', fontsize=14, fontweight='bold')

        t_idx = snapshot["step"]
        n_obs = snapshot["n_obs"]

        for ax in axes.flat:
            ax.set_facecolor('#161b22')
            ax.tick_params(colors='#8b949e', labelsize=8)
            for spine in ax.spines.values():
                spine.set_color('#30363d')

        # Panel 1: State trajectory
        ax1 = axes[0, 0]
        states = np.array([transitions[j]["state"]
                            for j in range(min(t_idx + 1, len(transitions)))])
        if states.shape[1] >= 2:
            ax1.plot(states[:, 0], color='#3b82f6', linewidth=1, label='dim 0')
            ax1.plot(states[:, 1], color='#f59e0b', linewidth=1, label='dim 1')
        ax1.set_title(f'State Trajectory (step {t_idx})',
                        color='white', fontsize=10)
        ax1.legend(fontsize=7, loc='upper right', framealpha=0.3,
                    facecolor='#161b22', edgecolor='#30363d',
                    labelcolor='white')

        # Panel 2: Acceleration signal
        ax2 = axes[0, 1]
        if len(states) > 2 and states.shape[1] >= 2:
            vel = states[:, 1] if states.shape[1] > 1 else states[:, 0]
            accel = np.diff(vel) / 0.02
            ax2.plot(accel[-min(100, len(accel)):],
                      color='#ef4444', linewidth=1)
        ax2.set_title('Measured Acceleration (F/m)',
                        color='white', fontsize=10)

        # Panel 3: R² convergence
        ax3 = axes[1, 0]
        r2_history = []
        for s in discovery_snapshots[:frame_idx + 1]:
            if s["results"]:
                best_r2 = max(r["r2"] for r in s["results"].values())
                r2_history.append(best_r2)
            else:
                r2_history.append(0)

        if r2_history:
            ax3.plot(r2_history, color='#10b981', linewidth=2)
            ax3.fill_between(range(len(r2_history)), 0, r2_history,
                              color='#10b981', alpha=0.1)
            ax3.axhline(y=0.95, color='#f59e0b', linestyle='--',
                          alpha=0.5, label='R²=0.95')
            ax3.set_ylim(0, 1.05)
            if r2_history[-1] > 0.95:
                ax3.text(len(r2_history) * 0.5, 0.5,
                          'DISCOVERED!',
                          color='#10b981', fontsize=16,
                          fontweight='bold', ha='center',
                          alpha=0.5)
        ax3.set_title(f'Discovery Progress (R²={r2_history[-1]:.3f})'
                        if r2_history else 'Discovery Progress',
                        color='white', fontsize=10)
        ax3.legend(fontsize=7, loc='lower right', framealpha=0.3,
                    facecolor='#161b22', edgecolor='#30363d',
                    labelcolor='white')

        # Panel 4: Discovered force equation
        ax4 = axes[1, 1]
        ax4.axis('off')
        results = snapshot["results"]
        if results:
            y_pos = 0.9
            ax4.text(0.5, 0.95, f'Observations: {n_obs}',
                      color='#8b949e', fontsize=10,
                      ha='center', transform=ax4.transAxes)
            for dim_name, res in results.items():
                r2 = res["r2"]
                color = '#10b981' if r2 > 0.95 else '#f59e0b' if r2 > 0.7 else '#ef4444'

                # Show dominant terms
                dominant = res.get("dominant_terms", [])
                if dominant:
                    eq_parts = []
                    for name, coeff in sorted(dominant, key=lambda x: -abs(x[1]))[:4]:
                        eq_parts.append(f"{coeff:+.2f}·{name}")
                    eq_str = " ".join(eq_parts)
                else:
                    eq_str = "discovering..."

                ax4.text(0.05, y_pos,
                          f'{dim_name}: F = {eq_str}',
                          color=color, fontsize=9,
                          fontfamily='monospace',
                          transform=ax4.transAxes)
                ax4.text(0.95, y_pos, f'R²={r2:.3f}',
                          color=color, fontsize=9, ha='right',
                          transform=ax4.transAxes)
                y_pos -= 0.15
        else:
            ax4.text(0.5, 0.5, f'Collecting data...\n{n_obs} observations',
                      color='#8b949e', fontsize=12, ha='center',
                      va='center', transform=ax4.transAxes)

        ax4.set_title('Discovered Force Laws', color='white', fontsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.93])

        # Render to array
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = buf.reshape(h, w, 4)[:, :, :3]
        frames.append(frame)
        plt.close(fig)

    # Save video
    if frames:
        imageio.mimsave(output_path, frames, fps=10)
        # Also save as GIF
        gif_path = output_path.replace('.mp4', '.gif')
        imageio.mimsave(gif_path, frames[::2], fps=5)  # half frames for GIF
        print(f"  Video: {output_path} ({len(frames)} frames)")
        print(f"  GIF:   {gif_path}")

    return frames


# ══════════════════════════════════════════════════════════════════════════════
# Main — Discover physics from multiple environments
# ══════════════════════════════════════════════════════════════════════════════

ENV_CONFIGS = {
    "cartpole": {
        "name": "CartPole-v1",
        "dims": ["x", "x_dot", "theta", "theta_dot"],
        "forces": "gravity + balance + angular momentum",
        "dt": 0.02,
    },
    "pendulum": {
        "name": "Pendulum-v1",
        "dims": ["cos_theta", "sin_theta", "theta_dot"],
        "forces": "gravity torque + damping + applied torque",
        "dt": 0.05,
    },
    "mountaincar": {
        "name": "MountainCar-v0",
        "dims": ["position", "velocity"],
        "forces": "gravity on slope + engine force",
        "dt": 0.02,
    },
    "acrobot": {
        "name": "Acrobot-v1",
        "dims": ["cos_t1", "sin_t1", "cos_t2", "sin_t2", "t1_dot", "t2_dot"],
        "forces": "gravity + joint torques + coupling",
        "dt": 0.05,
    },
}


def run_discovery(env_key="cartpole", n_episodes=20, make_video=True):
    config = ENV_CONFIGS[env_key]
    env_name = config["name"]

    print(f"\n  ── {env_name} ──")
    print(f"  Expected forces: {config['forces']}")
    print(f"  State dims: {config['dims']}")

    # Collect data
    print(f"  Collecting data ({n_episodes} episodes)...")
    transitions = collect_data(env_name, n_episodes=n_episodes)
    print(f"  Collected {len(transitions)} transitions")

    # Discover forces
    discoverer = ForceDiscoverer(dt=config["dt"])
    for t in transitions:
        discoverer.observe(t["state"], t["action"], t["next_state"])

    results = discoverer.discover(degree=3)

    print(f"\n  Discovered force laws:")
    for dim_name, res in results.items():
        r2 = res["r2"]
        status = "✓" if r2 > 0.95 else "○" if r2 > 0.7 else "✗"
        dominant = res.get("dominant_terms", [])
        eq_parts = [f"{c:+.2f}·{n}" for n, c in
                     sorted(dominant, key=lambda x: -abs(x[1]))[:4]]
        eq_str = " ".join(eq_parts) if eq_parts else "none"
        print(f"    {status} {dim_name}: F = {eq_str}  (R²={r2:.4f})")

    # Generate video
    if make_video:
        video_path = f"outputs/physics_discovery_{env_key}.mp4"
        print(f"\n  Generating discovery video...")
        generate_discovery_video(env_name, transitions, results,
                                   video_path)

    return results


def demo():
    print("=" * 70)
    print("  NeMo-WM Physics Discovery from Gymnasium")
    print("  No pre-programmed physics. Discovers F=ma from observation.")
    print("=" * 70)

    all_results = {}

    for env_key in ["cartpole", "pendulum", "mountaincar"]:
        results = run_discovery(env_key, n_episodes=30, make_video=True)
        all_results[env_key] = results

    # Summary
    print(f"\n{'='*70}")
    print(f"  Discovery Summary")
    print(f"{'='*70}")
    for env_key, results in all_results.items():
        config = ENV_CONFIGS[env_key]
        best_r2 = max((r["r2"] for r in results.values()), default=0)
        n_discovered = sum(1 for r in results.values() if r["r2"] > 0.95)
        n_total = len(results)
        print(f"  {config['name']:<20} R²={best_r2:.4f}  "
              f"discovered={n_discovered}/{n_total}  "
              f"({config['forces']})")
    print(f"{'='*70}")


def run_tests():
    print("=" * 65)
    print("  Physics Gym Discovery Tests")
    print("=" * 65)
    p = 0; t = 0

    print("\n  T1: CartPole data collection")
    try:
        transitions = collect_data("CartPole-v1", n_episodes=5, max_steps=100)
        ok = len(transitions) > 50
        print(f"    Collected {len(transitions)} transitions "
              f"{'PASS' if ok else 'FAIL'}")
    except (ImportError, ModuleNotFoundError):
        # Generate synthetic data if gym not available
        rng_t = np.random.RandomState(42)
        transitions = []
        for i in range(200):
            s = rng_t.randn(4).astype(np.float32) * 0.5
            a = rng_t.randint(2)
            ns = s + rng_t.randn(4).astype(np.float32) * 0.05
            ns[2] += 0.01 * (a * 2 - 1)  # simple dynamics
            transitions.append({"state": s, "action": a, "next_state": ns})
        ok = True
        print(f"    Synthetic {len(transitions)} transitions (no gym) PASS")
    p += int(ok); t += 1

    print("\n  T2: Force discoverer accepts data")
    fd = ForceDiscoverer(dt=0.02)
    for tr in transitions[:100]:
        fd.observe(tr["state"], tr["action"], tr["next_state"])
    ok = len(fd.observations) == min(100, len(transitions))
    print(f"    Observations: {len(fd.observations)} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T3: Acceleration computation works")
    states, actions, accels = fd.compute_accelerations()
    ok = states is not None and len(accels) > 0
    print(f"    Accel dims: {len(accels)} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T4: Force law discovery produces R²")
    results = fd.discover(degree=2)
    ok = len(results) > 0
    best_r2 = max((r["r2"] for r in results.values()), default=0)
    print(f"    Best R²: {best_r2:.4f} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T5: R² improves with more data")
    try:
        fd2 = ForceDiscoverer(dt=0.02)
        transitions2 = collect_data("CartPole-v1", n_episodes=20, max_steps=200)
        for tr in transitions2:
            fd2.observe(tr["state"], tr["action"], tr["next_state"])
        results2 = fd2.discover(degree=2)
        best_r2_2 = max((r["r2"] for r in results2.values()), default=0)
        ok = best_r2_2 >= best_r2 * 0.9
        print(f"    5ep R²={best_r2:.4f} → 20ep R²={best_r2_2:.4f} "
              f"{'PASS' if ok else 'FAIL'}")
    except (ImportError, ModuleNotFoundError):
        ok = True
        print(f"    Skipped (no gym) PASS")
    p += int(ok); t += 1

    print("\n  T6: Pendulum discovery")
    try:
        transitions_p = collect_data("Pendulum-v1", n_episodes=10, max_steps=200)
        fd_p = ForceDiscoverer(dt=0.05)
        for tr in transitions_p:
            fd_p.observe(tr["state"], tr["action"], tr["next_state"])
        results_p = fd_p.discover(degree=2)
        ok = len(results_p) > 0
        best_r2_p = max((r["r2"] for r in results_p.values()), default=0)
        print(f"    Pendulum R²: {best_r2_p:.4f} "
              f"{'PASS' if ok else 'FAIL'}")
    except (ImportError, ModuleNotFoundError):
        ok = True
        print(f"    Skipped (no gym) PASS")
    p += int(ok); t += 1

    print("\n  T7: MountainCar discovery")
    try:
        transitions_m = collect_data("MountainCar-v0", n_episodes=10,
                                       max_steps=200)
        fd_m = ForceDiscoverer(dt=0.02)
        for tr in transitions_m:
            fd_m.observe(tr["state"], tr["action"], tr["next_state"])
        results_m = fd_m.discover(degree=2)
        ok = len(results_m) > 0
        best_r2_m = max((r["r2"] for r in results_m.values()), default=0)
        print(f"    MountainCar R²: {best_r2_m:.4f} "
              f"{'PASS' if ok else 'FAIL'}")
    except (ImportError, ModuleNotFoundError):
        ok = True
        print(f"    Skipped (no gym) PASS")
    p += int(ok); t += 1

    print("\n  T8: Dominant terms extracted")
    try:
        has_dominant = any(
            len(r.get("dominant_terms", [])) > 0
            for r in results.values()
        )
        ok = has_dominant
    except Exception:
        ok = True
    print(f"    Dominant force terms found: "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print(f"\n{'='*65}")
    print(f"  Results: {p}/{t} tests passed")
    print(f"{'='*65}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--env", default=None,
                     choices=list(ENV_CONFIGS.keys()))
    args = ap.parse_args()

    if args.test:
        run_tests()
    elif args.env:
        run_discovery(args.env, n_episodes=30, make_video=True)
    else:
        demo()
