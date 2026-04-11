"""
synthetic_trajectory_generator.py  —  NeMo-WM Sprint 9
=======================================================
Procedurally generate physics-simulated 2D robot trajectories for
pre-training the k_ctx=16 proprioceptive encoder.

Motivated by SIMPLE (Fan et al., 2026, arXiv:2603.27410): physics-grounded
training data diversity overcomes the low-curvature bias present in the
RECON Berkeley campus dataset (mostly straight paths, rare contact events).

Generated trajectories match RECON HDF5 format exactly:
  commands/linear_velocity   (T,)  m/s
  commands/angular_velocity  (T,)  rad/s
  gps/latlong                (T,2) lat/lon (synthetic GPS from dead-reckoning)

No visual data is generated — this is proprioceptive-only pre-training.
The cortisol domain-adaptive signal handles sim-to-real transfer on fine-tune.

Physics model:
  - Differential-drive robot on 2D surface
  - Unicycle kinematics: x' = v*cos(θ), y' = v*sin(θ), θ' = ω
  - Obstacles: rectangular walls, circular pillars
  - Contact detection: proximity < robot_radius triggers contact signal
  - 4Hz sampling (dt=0.25s) matching RECON

Scenario types (SIMPLE-inspired):
  1. straight       — low curvature, high speed
  2. curved         — medium curvature, medium speed
  3. slalom         — alternating turns around obstacles
  4. stop_and_go    — velocity variation with stops
  5. tight_turn     — high curvature, low speed
  6. patrol         — repeated loop with contact events
  7. approach_avoid — obstacle avoidance manoeuvres
  8. random_walk    — stochastic heading changes

Usage:
  python synthetic_trajectory_generator.py \
      --n-trajectories 500 \
      --out-dir recon_data/synthetic_sprint9 \
      --seed 42

  # Then train proprio encoder on synthetic + real:
  python train_proprio_6c.py \
      --hdf5-dir recon_data/synthetic_sprint9 \
      --out-ckpt checkpoints/cwm/proprio_synth_pretrain.pt \
      --epochs 10 --k-ctx 16
"""

import argparse
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np


# ── Constants ────────────────────────────────────────────────────────────────

DT          = 0.25          # seconds per frame (4Hz, matching RECON)
T_FRAMES    = 70            # frames per trajectory (matching RECON ~17.5s)
ROBOT_R     = 0.3           # robot radius (metres)
GPS_ORIGIN  = (37.91506, -122.33627)   # Berkeley campus origin (matches RECON)
LAT_M       = 111000.0      # metres per degree latitude
MAX_VEL     = 1.5           # m/s
MAX_ANG     = 2.0           # rad/s


# ── Obstacle types ───────────────────────────────────────────────────────────

@dataclass
class Wall:
    x1: float; y1: float; x2: float; y2: float

    def distance_to(self, px: float, py: float) -> float:
        dx, dy = self.x2 - self.x1, self.y2 - self.y1
        t = max(0, min(1, ((px-self.x1)*dx + (py-self.y1)*dy) / (dx*dx+dy*dy+1e-9)))
        cx, cy = self.x1 + t*dx, self.y1 + t*dy
        return math.hypot(px-cx, py-cy)


@dataclass
class Pillar:
    cx: float; cy: float; r: float

    def distance_to(self, px: float, py: float) -> float:
        return max(0.0, math.hypot(px-self.cx, py-self.cy) - self.r)


# ── Robot state ──────────────────────────────────────────────────────────────

@dataclass
class RobotState:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0     # heading radians

    def step(self, v: float, omega: float, dt: float = DT):
        self.x     += v * math.cos(self.theta) * dt
        self.y     += v * math.sin(self.theta) * dt
        self.theta += omega * dt

    def contact_with(self, obstacles) -> bool:
        for obs in obstacles:
            if obs.distance_to(self.x, self.y) < ROBOT_R * 1.5:
                return True
        return False


# ── Scenario generators ───────────────────────────────────────────────────────

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def gen_straight(rng, T=T_FRAMES):
    """Fast straight with minor heading noise."""
    speed = rng.uniform(0.8, MAX_VEL)
    vels, angs = [], []
    for _ in range(T):
        vels.append(_clamp(speed + rng.gauss(0, 0.05), 0.1, MAX_VEL))
        angs.append(_clamp(rng.gauss(0, 0.05), -0.2, 0.2))
    return np.array(vels), np.array(angs), []


def gen_curved(rng, T=T_FRAMES):
    """Sustained curve — heading accumulates."""
    speed  = rng.uniform(0.4, 1.0)
    curve  = rng.choice([-1, 1]) * rng.uniform(0.3, 0.8)
    vels, angs = [], []
    for _ in range(T):
        vels.append(_clamp(speed + rng.gauss(0, 0.04), 0.1, MAX_VEL))
        angs.append(_clamp(curve + rng.gauss(0, 0.08), -MAX_ANG, MAX_ANG))
    return np.array(vels), np.array(angs), []


def gen_slalom(rng, T=T_FRAMES):
    """Alternating turns around pillars."""
    pillars = [Pillar(rng.uniform(2, 8) * i, rng.uniform(-1, 1), 0.4)
               for i in range(1, 6)]
    speed   = rng.uniform(0.4, 0.8)
    sign    = 1
    vels, angs = [], []
    state   = RobotState()
    for t in range(T):
        # Head toward next pillar, veer around it
        target_ang = sign * rng.uniform(0.4, 1.2)
        if t % 12 == 0:
            sign *= -1
        vels.append(_clamp(speed + rng.gauss(0, 0.05), 0.1, MAX_VEL))
        angs.append(_clamp(target_ang + rng.gauss(0, 0.1), -MAX_ANG, MAX_ANG))
        state.step(vels[-1], angs[-1])
    return np.array(vels), np.array(angs), pillars


def gen_stop_and_go(rng, T=T_FRAMES):
    """Velocity variation with stops."""
    vels, angs = [], []
    phase = 0
    phase_len = rng.randint(5, 12)
    count = 0
    speed = rng.uniform(0.5, 1.2)
    for _ in range(T):
        count += 1
        if count >= phase_len:
            phase = (phase + 1) % 3
            phase_len = rng.randint(4, 12)
            count = 0
        if phase == 0:    # moving
            v = _clamp(speed + rng.gauss(0, 0.06), 0.2, MAX_VEL)
            w = _clamp(rng.gauss(0, 0.15), -0.5, 0.5)
        elif phase == 1:  # stopping
            v = _clamp(speed * 0.3 + rng.gauss(0, 0.02), 0.0, 0.5)
            w = _clamp(rng.gauss(0, 0.05), -0.2, 0.2)
        else:             # stopped
            v = _clamp(abs(rng.gauss(0, 0.02)), 0.0, 0.08)
            w = _clamp(rng.gauss(0, 0.1), -0.3, 0.3)
        vels.append(v); angs.append(w)
    return np.array(vels), np.array(angs), []


def gen_tight_turn(rng, T=T_FRAMES):
    """High-curvature manoeuvres — good for heading encoder stress test."""
    vels, angs = [], []
    phase = 0
    phase_len = rng.randint(4, 10)
    count = 0
    for _ in range(T):
        count += 1
        if count >= phase_len:
            phase = rng.randint(0, 3)
            phase_len = rng.randint(4, 10)
            count = 0
        if phase == 0:    # forward
            v = rng.uniform(0.3, 0.8)
            w = rng.gauss(0, 0.1)
        elif phase == 1:  # hard left
            v = rng.uniform(0.2, 0.5)
            w = rng.uniform(0.8, MAX_ANG)
        elif phase == 2:  # hard right
            v = rng.uniform(0.2, 0.5)
            w = rng.uniform(-MAX_ANG, -0.8)
        else:             # pivot (low v, high w)
            v = rng.uniform(0.0, 0.2)
            w = rng.choice([-1, 1]) * rng.uniform(1.0, MAX_ANG)
        vels.append(_clamp(v, 0.0, MAX_VEL))
        angs.append(_clamp(w, -MAX_ANG, MAX_ANG))
    return np.array(vels), np.array(angs), []


def gen_patrol(rng, T=T_FRAMES):
    """Repeated loop — heading wraps 2π, good for sin/cos encoding."""
    radius = rng.uniform(3.0, 8.0)
    speed  = rng.uniform(0.4, 0.9)
    # For a circle: ω = v / r
    omega  = (speed / radius) * rng.choice([-1, 1])
    vels   = [_clamp(speed + rng.gauss(0, 0.04), 0.1, MAX_VEL) for _ in range(T)]
    angs   = [_clamp(omega + rng.gauss(0, 0.06), -MAX_ANG, MAX_ANG) for _ in range(T)]
    return np.array(vels), np.array(angs), []


def gen_approach_avoid(rng, T=T_FRAMES):
    """Approach wall, detect contact, reverse/turn."""
    wall = Wall(-5, 6 + rng.uniform(-1, 1), 5, 6 + rng.uniform(-1, 1))
    state  = RobotState(x=0, y=0, theta=math.pi/2)
    vels, angs = [], []
    avoiding = False
    avoid_count = 0
    for _ in range(T):
        dist = wall.distance_to(state.x, state.y)
        if dist < 1.2 or avoiding:
            avoiding = True
            avoid_count += 1
            v = _clamp(-0.3 + rng.gauss(0, 0.05), -0.5, 0.0)
            w = rng.choice([-1, 1]) * rng.uniform(0.5, 1.5)
            if avoid_count > 8:
                avoiding = False
                avoid_count = 0
        else:
            v = _clamp(0.6 + rng.gauss(0, 0.06), 0.1, MAX_VEL)
            w = _clamp(rng.gauss(0, 0.1), -0.3, 0.3)
        vels.append(v); angs.append(w)
        state.step(v, w)
    return np.array(vels), np.array(angs), [wall]


def gen_random_walk(rng, T=T_FRAMES):
    """Brownian-motion-style heading changes — maximally diverse."""
    speed = rng.uniform(0.3, 0.9)
    vels, angs = [], []
    omega = rng.gauss(0, 0.3)
    for _ in range(T):
        omega += rng.gauss(0, 0.2)
        omega  = _clamp(omega, -MAX_ANG, MAX_ANG)
        v      = _clamp(speed + rng.gauss(0, 0.08), 0.05, MAX_VEL)
        vels.append(v); angs.append(omega)
    return np.array(vels), np.array(angs), []


SCENARIOS = [
    ("straight",        gen_straight),
    ("curved",          gen_curved),
    ("slalom",          gen_slalom),
    ("stop_and_go",     gen_stop_and_go),
    ("tight_turn",      gen_tight_turn),
    ("patrol",          gen_patrol),
    ("approach_avoid",  gen_approach_avoid),
    ("random_walk",     gen_random_walk),
]


# ── GPS dead-reckoning ────────────────────────────────────────────────────────

def dead_reckon_gps(
    vels: np.ndarray,
    angs: np.ndarray,
    origin_lat: float = GPS_ORIGIN[0],
    origin_lon: float = GPS_ORIGIN[1],
) -> np.ndarray:
    """
    Integrate velocity + heading to produce synthetic GPS track.
    Returns (T, 2) array of [lat, lon].
    """
    x, y, theta = 0.0, 0.0, 0.0
    coords = []
    for v, w in zip(vels, angs):
        x     += v * math.cos(theta) * DT
        y     += v * math.sin(theta) * DT
        theta += w * DT
        lat    = origin_lat + y / LAT_M
        lon    = origin_lon + x / (LAT_M * math.cos(math.radians(origin_lat)))
        coords.append([lat, lon])
    return np.array(coords, dtype=np.float64)


def contact_signal(
    vels: np.ndarray,
    angs: np.ndarray,
    obstacles: list,
) -> np.ndarray:
    """Compute binary contact at each frame (1 if near obstacle or vel>0.3)."""
    x, y, theta = 0.0, 0.0, 0.0
    contacts = []
    for v, w in zip(vels, angs):
        x     += v * math.cos(theta) * DT
        y     += v * math.sin(theta) * DT
        theta += w * DT
        near = any(obs.distance_to(x, y) < ROBOT_R * 2.0 for obs in obstacles)
        contacts.append(1.0 if (near or abs(v) > 0.3) else 0.0)
    return np.array(contacts, dtype=np.float32)


# ── HDF5 writer ───────────────────────────────────────────────────────────────

def write_trajectory_hdf5(
    path: str,
    vels: np.ndarray,       # (T,)
    angs: np.ndarray,       # (T,)
    gps:  np.ndarray,       # (T, 2)
    scenario: str,
):
    """
    Write synthetic trajectory to HDF5 matching RECON structure.
    Visual data (images/rgb_left) is omitted — proprio-only training.
    """
    with h5py.File(path, "w") as f:
        grp_cmd = f.create_group("commands")
        grp_cmd.create_dataset("linear_velocity",  data=vels.astype(np.float32))
        grp_cmd.create_dataset("angular_velocity", data=angs.astype(np.float32))

        grp_gps = f.create_group("gps")
        grp_gps.create_dataset("latlong", data=gps)

        f.attrs["scenario"]   = scenario
        f.attrs["synthetic"]  = True
        f.attrs["dt"]         = DT
        f.attrs["source"]     = "synthetic_trajectory_generator.py"
        f.attrs["reference"]  = "SIMPLE (Fan et al. 2026, arXiv:2603.27410)"


# ── Main generator ────────────────────────────────────────────────────────────

def generate_dataset(
    n_trajectories: int,
    out_dir: str,
    seed: int = 42,
    scenario_weights: Optional[List[float]] = None,
    verbose: bool = True,
):
    """
    Generate n_trajectories synthetic HDF5 files.

    Scenario distribution (default uniform):
      Each scenario covers a different region of the curvature/speed/contact space.
      - straight / curved / patrol     : low contact, varied curvature
      - slalom / tight_turn            : high curvature, moderate contact
      - stop_and_go / approach_avoid   : high contact frequency
      - random_walk                    : maximal diversity

    Args:
        n_trajectories:   number of files to generate
        out_dir:          output directory (created if absent)
        seed:             random seed for reproducibility
        scenario_weights: optional per-scenario probability weights
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    names   = [s[0] for s in SCENARIOS]
    fns     = [s[1] for s in SCENARIOS]
    weights = scenario_weights or [1.0] * len(SCENARIOS)
    total_w = sum(weights)
    cum_w   = [sum(weights[:i+1]) / total_w for i in range(len(weights))]

    counts  = {n: 0 for n in names}
    t0      = time.perf_counter()

    print(f"\nGenerating {n_trajectories} synthetic trajectories → {out_dir}")
    print(f"Scenarios: {', '.join(names)}")
    print(f"Format: RECON-compatible HDF5 (proprio only, no visual data)\n")

    for i in range(n_trajectories):
        # Select scenario
        r = rng.random()
        idx = next(j for j, cw in enumerate(cum_w) if r <= cw)
        scenario_name, scenario_fn = names[idx], fns[idx]

        # Generate trajectory
        try:
            vels, angs, obstacles = scenario_fn(rng)
        except Exception as e:
            print(f"  Warning: scenario {scenario_name} failed ({e}), using straight")
            vels, angs, obstacles = gen_straight(rng)

        # Ensure correct length
        vels = np.array(vels[:T_FRAMES], dtype=np.float32)
        angs = np.array(angs[:T_FRAMES], dtype=np.float32)
        if len(vels) < T_FRAMES:
            vels = np.pad(vels, (0, T_FRAMES - len(vels)), constant_values=0.0)
            angs = np.pad(angs, (0, T_FRAMES - len(angs)), constant_values=0.0)

        gps = dead_reckon_gps(vels, angs)

        # Filename: synthetic_{scenario}_{i:05d}.hdf5
        fname = out_path / f"synthetic_{scenario_name}_{i:05d}.hdf5"
        write_trajectory_hdf5(str(fname), vels, angs, gps, scenario_name)
        counts[scenario_name] += 1

        if verbose and (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            rate    = (i + 1) / elapsed
            eta     = (n_trajectories - i - 1) / rate
            print(f"  {i+1:>5}/{n_trajectories}  ({rate:.1f} files/s, "
                  f"ETA {eta:.0f}s)")

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s  ({n_trajectories/elapsed:.1f} files/s)")
    print(f"\nScenario breakdown:")
    for name, count in sorted(counts.items(), key=lambda x: -x[1]):
        bar = "█" * (count * 30 // n_trajectories)
        print(f"  {name:<18} {count:>5}  {bar}")

    print(f"\nOutput: {out_path}  ({n_trajectories} HDF5 files)")
    print(f"\nPre-train proprio encoder on synthetic data:")
    print(f"  python train_proprio_6c.py \\")
    print(f"      --hdf5-dir {out_dir} \\")
    print(f"      --out-ckpt checkpoints/cwm/proprio_synth_pretrain.pt \\")
    print(f"      --epochs 10 --k-ctx 16")
    print(f"\nThen fine-tune on real RECON:")
    print(f"  python train_proprio_6c.py \\")
    print(f"      --hdf5-dir recon_data/recon_release \\")
    print(f"      --out-ckpt checkpoints/cwm/proprio_kctx16_finetuned.pt \\")
    print(f"      --k-ctx 16 --epochs 10 \\")
    print(f"      --init-ckpt checkpoints/cwm/proprio_synth_pretrain.pt")

    return counts


# ── Self-test ─────────────────────────────────────────────────────────────────

def self_test():
    import tempfile, os
    print("Synthetic trajectory generator — self test")
    rng = random.Random(0)

    for name, fn in SCENARIOS:
        vels, angs, obs = fn(rng)
        assert len(vels) == T_FRAMES, f"{name}: wrong vel length {len(vels)}"
        assert len(angs) == T_FRAMES, f"{name}: wrong ang length {len(angs)}"
        assert all(-MAX_VEL - 0.01 <= v <= MAX_VEL + 0.01 for v in vels), \
            f"{name}: vel out of range — min={min(vels):.3f} max={max(vels):.3f}"
        gps = dead_reckon_gps(np.array(vels), np.array(angs))
        assert gps.shape == (T_FRAMES, 2), f"{name}: wrong GPS shape {gps.shape}"
        print(f"  {name:<18} OK  vel=[{min(vels):.2f},{max(vels):.2f}]  "
              f"ang=[{min(angs):.2f},{max(angs):.2f}]  "
              f"gps_range={np.ptp(gps[:,0])*LAT_M:.1f}m")

    # Write + read HDF5
    with tempfile.TemporaryDirectory() as tmp:
        counts = generate_dataset(16, tmp, seed=1, verbose=False)
        files  = list(Path(tmp).glob("*.hdf5"))
        assert len(files) == 16, f"Expected 16 files, got {len(files)}"
        with h5py.File(str(files[0]), "r") as f:
            assert "commands/linear_velocity"  in f
            assert "commands/angular_velocity" in f
            assert "gps/latlong"               in f
            assert f["commands/linear_velocity"].shape  == (T_FRAMES,)
            assert f["gps/latlong"].shape               == (T_FRAMES, 2)
        print(f"\n  HDF5 write/read: OK ({len(files)} files)")
        print(f"  Scenario coverage: {dict(counts)}")

    print("\n✅ All assertions passed")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Generate synthetic RECON-format trajectories for Sprint 9"
    )
    p.add_argument("--n-trajectories", type=int, default=500)
    p.add_argument("--out-dir",        default="recon_data/synthetic_sprint9")
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--test",           action="store_true")
    args = p.parse_args()

    if args.test:
        self_test()
    else:
        generate_dataset(
            n_trajectories = args.n_trajectories,
            out_dir        = args.out_dir,
            seed           = args.seed,
        )
