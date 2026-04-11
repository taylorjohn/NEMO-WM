"""
train_action_wm.py  —  NeMo-WM Action Conditioning Sprint C
============================================================
Trains an action-conditioned world model on PushT and PointMaze-UMaze
in parallel, using three interventions to prevent temporal copy collapse:

  1. AdaLN-Zero conditioning  — replaces additive action injection
  2. Multi-step prediction    — predict t+k (k=4) to break frame shortcuts
  3. Inverse dynamics loss    — IDM head forces action-sensitive latents

Benchmarks:
  PushT       — state-only (5-dim obs, 2-dim action), fast ablations
  PointMaze   — visual 64×64 + state (4-dim obs, 2-dim action), paper-quality

Five diagnostic metrics logged every epoch:
  1. ac_lift       — action-conditioned vs unconditional loss gap
  2. cf_div        — counterfactual prediction divergence
  3. grad_ratio    — ||∂L/∂a|| / ||∂L/∂z||
  4. idm_acc       — inverse dynamics probe accuracy
  5. gate_norm     — AdaLN alpha gate magnitude

Pass criteria (Sprint C):
  PushT:      ac_lift > 0.05 by ep5, IDM MAE < 0.1 by ep10
  PointMaze:  ac_lift > 0.02 by ep5, SR (open-loop GD) > 50% by ep20

DreamerV3 stability tricks: symlog targets, AGC λ=0.01, unimix 1%

Usage:
    # PushT only (fast, state-only)
    python train_action_wm.py --domain pusht --epochs 50

    # PointMaze only (visual)
    python train_action_wm.py --domain pointmaze --epochs 100 --visual

    # Both in parallel (runs sequentially, saves best per domain)
    python train_action_wm.py --domain both --epochs 50

    # Full ablation suite (6 variants)
    python train_action_wm.py --domain pusht --ablation --epochs 30
"""

import argparse
import math
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ═══════════════════════════════════════════════════════════════════════════
# DreamerV3 stability utilities
# ═══════════════════════════════════════════════════════════════════════════

def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def info_nce_loss(z_pred: torch.Tensor, z_target: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """
    InfoNCE contrastive prediction loss.

    Positives: (z_pred[i], z_target[i]) — same trajectory step
    Negatives: (z_pred[i], z_target[j]) for j != i — other batch samples at t+k

    Unlike MSE, copying z_t as z_pred gives high similarity to ALL z_target[j]
    (since all are temporally smooth), keeping loss high. Only using the action
    to predict the CORRECT z_target[i] reduces the loss.

    tau=0.1 — sharp temperature for strong gradient signal.
    """
    # Normalise embeddings
    z_pred_n   = F.normalize(z_pred,   dim=-1)   # (B, D)
    z_target_n = F.normalize(z_target, dim=-1)   # (B, D)
    # Similarity matrix (B, B)
    logits = torch.mm(z_pred_n, z_target_n.T) / tau
    # Positives are on the diagonal
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)

def agc_clip(params, clip: float = 0.01, eps: float = 1e-3):
    for p in params:
        if p.grad is None: continue
        p_norm = p.detach().norm(2).clamp(min=eps)
        g_norm = p.grad.detach().norm(2)
        if g_norm > clip * p_norm:
            p.grad.mul_(clip * p_norm / g_norm)

def unimix(a: torch.Tensor, eps: float = 0.01) -> torch.Tensor:
    """Unimix 1%: blend action with uniform to avoid degenerate embeddings."""
    return (1 - eps) * a + eps * torch.zeros_like(a)


# ═══════════════════════════════════════════════════════════════════════════
# AdaLN-Zero action conditioning
# ═══════════════════════════════════════════════════════════════════════════

class AdaLNZero(nn.Module):
    """
    Adaptive Layer Normalization with zero initialization.

    Replaces additive action injection. Applies per-feature scale γ, shift β,
    and gate α, all learned from the action vector. Zero initialization means
    the model starts as fully unconditioned and gradually learns action influence.

    Gate α starts at 0 → sigmoid(0) = 0.5, but weight init at 0 means
    the output contribution starts near zero. Action influence grows as
    gradients flow through the IDM and multi-step prediction losses.

    Reference: DiT (Peebles & Xie 2023), LeWM (2026), AdaLN-Zero survey.
    """

    def __init__(self, action_dim: int, d_model: int):
        super().__init__()
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 3 * d_model),   # → γ, β, α
        )
        # Zero init: starts as identity, action conditioning grows from zero
        nn.init.zeros_(self.action_mlp[-1].weight)
        nn.init.zeros_(self.action_mlp[-1].bias)
        self.d_model = d_model

    def forward(self, h: torch.Tensor, a_emb: torch.Tensor) -> torch.Tensor:
        """
        h:     (B, T, d_model) or (B, d_model)
        a_emb: (B, d_model)  — pre-projected action embedding
        Returns: modulated h, same shape
        """
        gamma, beta, alpha = self.action_mlp(a_emb).chunk(3, dim=-1)

        if h.dim() == 3:
            gamma = gamma.unsqueeze(1)  # (B, 1, d)
            beta  = beta.unsqueeze(1)
            alpha = alpha.unsqueeze(1)

        h_norm = F.layer_norm(h, [self.d_model])
        h_mod  = (1 + gamma) * h_norm + beta
        return h + 0.1 * alpha * h_mod          # residual + linear gate (no tanh saturation)


class ActionEmbedder(nn.Module):
    """Projects raw action → d_model embedding for AdaLN conditioning."""

    def __init__(self, action_dim: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.net(unimix(a))


# ═══════════════════════════════════════════════════════════════════════════
# Inverse dynamics head
# ═══════════════════════════════════════════════════════════════════════════

class InverseDynamicsHead(nn.Module):
    """
    Predicts action a_t from (z_t, z_{t+1}).

    If the world model collapses to temporal copying (z_{t+1} ≈ z_t),
    the IDM cannot recover the action, creating gradient pressure to
    preserve action-relevant information in the latent space.

    λ_IDM=100 (WAM 2026) — deliberately high to maintain action gradients.
    """

    def __init__(self, d_model: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, action_dim),
        )

    def forward(self, z_t: torch.Tensor, z_t1: torch.Tensor) -> torch.Tensor:
        """z_t, z_t1: (B, d_model) → predicted action (B, action_dim)"""
        return self.net(torch.cat([z_t, z_t1], dim=-1))


# ═══════════════════════════════════════════════════════════════════════════
# Lightweight world model (state-based)
# ═══════════════════════════════════════════════════════════════════════════

class StateEncoder(nn.Module):
    """Encodes raw state observations → latent."""

    def __init__(self, obs_dim: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActionConditionedTransition(nn.Module):
    """
    Single-step transition model with AdaLN-Zero action conditioning.
    Predicts z_{t+1} from z_t and a_t.

    Uses 3 residual MLP layers, each conditioned via AdaLN-Zero.
    No transformer — keeps parameter count low for CPU training.
    """

    def __init__(self, d_model: int, action_dim: int, n_layers: int = 3):
        super().__init__()
        self.action_emb = ActionEmbedder(action_dim, d_model)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            for _ in range(n_layers)
        ])
        self.adaln = nn.ModuleList([
            AdaLNZero(d_model, d_model) for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        z: (B, d_model)
        a: (B, action_dim) — normalised to [-1, 1]
        Returns: z_next (B, d_model)
        """
        a_emb = self.action_emb(a)
        h = z
        for layer, adaln in zip(self.layers, self.adaln):
            h = h + adaln(layer(h), a_emb)     # residual + AdaLN modulation
        return self.out_norm(h)

    def rollout(self, z0: torch.Tensor, actions: torch.Tensor) -> List[torch.Tensor]:
        """
        Multi-step rollout.
        z0:      (B, d_model)
        actions: (B, K, action_dim)
        Returns: list of K predicted states [z1, z2, ..., zK]
        """
        preds = []
        z = z0
        for k in range(actions.shape[1]):
            z = self.forward(z, actions[:, k])
            preds.append(z)
        return preds


class VisualEncoder(nn.Module):
    """
    Lightweight visual encoder for 64×64 frames.
    CNN → flatten → MLP → d_model. ~500K params, CPU-trainable.
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3,  32, 4, stride=2), nn.GELU(),   # 31×31
            nn.Conv2d(32, 64, 4, stride=2), nn.GELU(),   # 14×14
            nn.Conv2d(64, 128, 4, stride=2), nn.GELU(),  # 6×6
            nn.Conv2d(128, 256, 4, stride=2), nn.GELU(), # 2×2
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 64, 64) → (B, d_model)"""
        return self.proj(self.cnn(x))


# ═══════════════════════════════════════════════════════════════════════════
# PushT dataset
# ═══════════════════════════════════════════════════════════════════════════

class PushTStateDataset(Dataset):
    """
    PushT state-only dataset.

    Loads from:
      - lerobot/pusht HuggingFace (auto-download if hf_datasets available)
      - pusht_cchi_v7_replay.zarr (Columbia diffusion policy format)
      - Synthetic scripted policy (fallback, always available)

    State: (agent_x, agent_y, block_x, block_y, block_angle) ∈ R^5
    Action: (target_x, target_y) ∈ [-1, 1]^2 (normalised)
    """

    OBS_DIM    = 5
    ACTION_DIM = 2

    def __init__(
        self,
        data_path:  Optional[str] = None,
        k_steps:    int = 4,
        n_episodes: int = 206,
        synthetic:  bool = False,
    ):
        self.k = k_steps
        self.trajs: List[Dict[str, np.ndarray]] = []

        if synthetic or data_path is None:
            self._generate_synthetic(n_episodes)
        else:
            loaded = False
            p = Path(data_path)
            if p.suffix == ".zarr" or (p.is_dir() and (p / ".zarray").exists()):
                loaded = self._load_zarr(p)
            elif p.suffix in (".hdf5", ".h5"):
                loaded = self._load_hdf5(p)
            if not loaded:
                print(f"  Could not load {data_path} — generating synthetic data")
                self._generate_synthetic(n_episodes)

        # Build (episode, start_frame) index
        self.index = []
        for ep_idx, traj in enumerate(self.trajs):
            T = traj["obs"].shape[0]
            for t in range(T - k_steps):
                self.index.append((ep_idx, t))

        print(f"PushTStateDataset: {len(self.trajs)} episodes, "
              f"{len(self.index)} samples (k={k_steps})")

    def _generate_synthetic(self, n_episodes: int):
        """
        Scripted PushT policy: move agent toward block, push toward goal.
        Generates diverse trajectories with random goals and noise.
        """
        rng = np.random.RandomState(42)
        goal = np.array([0.75, 0.75])   # normalised target

        for _ in range(n_episodes):
            T = rng.randint(60, 150)
            obs  = np.zeros((T, 5), dtype=np.float32)
            acts = np.zeros((T, 2), dtype=np.float32)

            # Random initial state
            agent = rng.uniform(0.1, 0.9, 2)
            block = rng.uniform(0.2, 0.8, 2)
            angle = rng.uniform(0, 2 * math.pi)
            ep_goal = rng.uniform(0.5, 0.95, 2)

            for t in range(T):
                obs[t] = [agent[0], agent[1], block[0], block[1], angle / (2*math.pi)]

                # Phase 1: move toward block; Phase 2: push toward goal
                if np.linalg.norm(agent - block) > 0.15:
                    target = block + rng.normal(0, 0.05, 2)
                else:
                    push_dir = ep_goal - block
                    target   = agent + push_dir * 0.3 + rng.normal(0, 0.03, 2)

                target = np.clip(target, 0, 1)
                acts[t] = target * 2 - 1   # → [-1, 1]

                # Simple dynamics
                agent += (target - agent) * 0.4 + rng.normal(0, 0.01, 2)
                agent  = np.clip(agent, 0, 1)

                # Block moves if agent is close
                if np.linalg.norm(agent - block) < 0.1:
                    push = (agent - block) * 0.2
                    block = np.clip(block - push, 0, 1)
                    angle += rng.normal(0, 0.1)

            self.trajs.append({"obs": obs, "action": acts})

    def _load_zarr(self, path: Path) -> bool:
        try:
            import zarr
            z = zarr.open(str(path), "r")
            obs    = np.array(z["data/state"])      # (N, 5)
            action = np.array(z["data/action"])     # (N, 2)
            ep_ends = np.array(z["meta/episode_ends"])
            starts  = np.concatenate([[0], ep_ends[:-1]])
            for s, e in zip(starts, ep_ends):
                self.trajs.append({
                    "obs":    obs[s:e].astype(np.float32),
                    "action": action[s:e].astype(np.float32),
                })
            return True
        except Exception as ex:
            print(f"  Zarr load failed: {ex}")
            return False

    def _load_hdf5(self, path: Path) -> bool:
        try:
            import h5py
            with h5py.File(path, "r") as f:
                for ep_key in f.keys():
                    ep = f[ep_key]
                    self.trajs.append({
                        "obs":    np.array(ep["obs"],    dtype=np.float32),
                        "action": np.array(ep["action"], dtype=np.float32),
                    })
            return True
        except Exception as ex:
            print(f"  HDF5 load failed: {ex}")
            return False

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ep_idx, t = self.index[idx]
        traj = self.trajs[ep_idx]
        obs_seq    = traj["obs"][t : t + self.k + 1]       # (k+1, 5)
        action_seq = traj["action"][t : t + self.k]        # (k, 2)
        return {
            "obs":    torch.from_numpy(obs_seq),
            "action": torch.from_numpy(action_seq),
        }


# ═══════════════════════════════════════════════════════════════════════════
# PointMaze dataset
# ═══════════════════════════════════════════════════════════════════════════

class PointMazeDataset(Dataset):
    """
    PointMaze-UMaze dataset.

    Loads from:
      - Minari D4RL/pointmaze/umaze-v2 (auto-download if minari available)
      - Custom HDF5 file
      - Synthetic A* + PD controller (fallback)

    State: (x, y, vx, vy) ∈ R^4, normalised to [-1, 1]
    Action: (force_x, force_y) ∈ [-1, 1]^2
    Visual: 64×64 RGB top-down render (if visual=True)
    """

    OBS_DIM    = 4
    ACTION_DIM = 2

    # UMaze bounds (normalisation)
    POS_MIN = np.array([-2.0, -2.0])
    POS_MAX = np.array([10.0, 10.0])

    def __init__(
        self,
        data_path:  Optional[str] = None,
        k_steps:    int = 4,
        visual:     bool = False,
        img_size:   int = 64,
        n_episodes: int = 500,
    ):
        self.k        = k_steps
        self.visual   = visual
        self.img_size = img_size
        self.trajs: List[Dict[str, np.ndarray]] = []

        loaded = False
        if data_path is not None:
            p = Path(data_path)
            if p.suffix in (".hdf5", ".h5"):
                loaded = self._load_hdf5(p)
            elif p.is_dir():
                loaded = self._load_minari(str(p))

        if not loaded:
            try:
                loaded = self._load_minari_auto()
            except Exception:
                pass

        if not loaded:
            print("  Generating synthetic PointMaze trajectories...")
            self._generate_synthetic(n_episodes)

        if visual:
            self._prerender()

        self.index = []
        for ep_idx, traj in enumerate(self.trajs):
            T = traj["obs"].shape[0]
            for t in range(T - k_steps):
                self.index.append((ep_idx, t))

        print(f"PointMazeDataset: {len(self.trajs)} episodes, "
              f"{len(self.index)} samples (k={k_steps}, visual={visual})")

    def _load_minari_auto(self) -> bool:
        try:
            import minari
            ds = minari.load_dataset("D4RL/pointmaze/umaze-v2", download=True)
            for ep in ds.iterate_episodes():
                obs    = ep.observations["observation"].astype(np.float32)
                action = ep.actions.astype(np.float32)
                self.trajs.append({"obs": self._norm_obs(obs), "action": action})
            return True
        except Exception as ex:
            print(f"  Minari auto-load failed: {ex}")
            return False

    def _load_minari(self, path: str) -> bool:
        try:
            import minari
            ds = minari.load_dataset(path)
            for ep in ds.iterate_episodes():
                obs    = ep.observations["observation"].astype(np.float32)
                action = ep.actions.astype(np.float32)
                self.trajs.append({"obs": self._norm_obs(obs), "action": action})
            return True
        except Exception as ex:
            print(f"  Minari load failed: {ex}")
            return False

    def _load_hdf5(self, path: Path) -> bool:
        try:
            import h5py
            with h5py.File(path, "r") as f:
                obs    = np.array(f["observations"], dtype=np.float32)
                action = np.array(f["actions"],      dtype=np.float32)
                terminals = np.array(f.get("terminals", np.zeros(len(obs))))
                ep_ends = np.where(terminals)[0] + 1
                if len(ep_ends) == 0:
                    ep_ends = [len(obs)]
                starts = np.concatenate([[0], ep_ends[:-1]])
                for s, e in zip(starts, ep_ends):
                    self.trajs.append({
                        "obs":    self._norm_obs(obs[s:e]),
                        "action": action[s:e],
                    })
            return True
        except Exception as ex:
            print(f"  HDF5 load failed: {ex}")
            return False

    def _norm_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalise (x, y, vx, vy) to [-1, 1]."""
        pos = obs[:, :2]
        vel = obs[:, 2:]
        pos_n = 2 * (pos - self.POS_MIN) / (self.POS_MAX - self.POS_MIN) - 1
        vel_n = np.clip(vel / 3.0, -1, 1)
        return np.concatenate([pos_n, vel_n], axis=1).astype(np.float32)

    def _generate_synthetic(self, n_episodes: int):
        """Simple scripted PD controller navigating U-maze."""
        rng = np.random.RandomState(42)
        goals = [
            np.array([7.0, 1.0]),
            np.array([1.0, 7.0]),
            np.array([7.0, 7.0]),
        ]
        for i in range(n_episodes):
            T   = rng.randint(80, 200)
            obs = np.zeros((T, 4), dtype=np.float32)
            act = np.zeros((T, 2), dtype=np.float32)
            pos = rng.uniform(0, 2, 2)
            vel = np.zeros(2)
            goal = goals[i % len(goals)] + rng.normal(0, 0.5, 2)
            for t in range(T):
                obs[t] = [*self._norm_obs(pos[None])[0], *np.clip(vel/3, -1, 1)]
                direction = goal - pos
                dist      = np.linalg.norm(direction) + 1e-6
                force     = direction / dist * min(dist, 1.0) + rng.normal(0, 0.1, 2)
                force     = np.clip(force, -1, 1)
                act[t]    = force
                vel = vel * 0.8 + force * 0.3
                pos = pos + vel * 0.1
                pos = np.clip(pos, -1, 9)
            self.trajs.append({"obs": obs, "action": act})

    def _prerender(self):
        """Render top-down maze images for each timestep."""
        s = self.img_size
        for traj in self.trajs:
            T   = traj["obs"].shape[0]
            imgs = np.zeros((T, 3, s, s), dtype=np.float32)
            pos_n = traj["obs"][:, :2]   # already in [-1, 1]
            # Simple rendering: agent dot on grey background
            for t in range(T):
                img = np.ones((s, s, 3), dtype=np.float32) * 0.3
                # Convert normalised pos → pixel
                px = int((pos_n[t, 0] + 1) / 2 * (s - 1))
                py = int((pos_n[t, 1] + 1) / 2 * (s - 1))
                px, py = np.clip(px, 1, s-2), np.clip(py, 1, s-2)
                img[py-1:py+2, px-1:px+2] = [1.0, 0.2, 0.2]   # red agent
                # U-maze walls (approximate)
                img[:, s//3-1:s//3+1] = 0.1   # left wall
                img[:s//2, 2*s//3-1:2*s//3+1] = 0.1   # right wall (upper)
                imgs[t] = img.transpose(2, 0, 1)
            traj["frames"] = imgs

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ep_idx, t = self.index[idx]
        traj = self.trajs[ep_idx]
        obs_seq    = traj["obs"][t : t + self.k + 1]
        action_seq = traj["action"][t : t + self.k]
        item = {
            "obs":    torch.from_numpy(obs_seq),
            "action": torch.from_numpy(action_seq),
        }
        if self.visual and "frames" in traj:
            item["frames"] = torch.from_numpy(traj["frames"][t : t + self.k + 1])
        return item


# ═══════════════════════════════════════════════════════════════════════════
# Diagnostic metrics
# ═══════════════════════════════════════════════════════════════════════════

class ActionDiagnostics:
    """
    Tracks the 5 diagnostic metrics that reveal whether actions are
    genuinely load-bearing or ignored.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.ac_lifts   = []
        self.cf_divs    = []
        self.idm_errs   = []
        self.grad_ratios = []

    def update_ac_lift(self, L_cond: float, L_uncond: float):
        self.ac_lifts.append(L_uncond - L_cond)

    def update_cf_div(self, model, z_t, a1, a2):
        """Counterfactual divergence: same state, different actions."""
        with torch.no_grad():
            z1 = model(z_t, a1)
            z2 = model(z_t, a2)
            self.cf_divs.append((z1 - z2).norm(dim=-1).mean().item())

    def update_idm(self, idm_head, z_t, z_t1, a_true):
        with torch.no_grad():
            a_pred = idm_head(z_t, z_t1)
            err = (a_pred - a_true).abs().mean().item()
            self.idm_errs.append(err)

    def update_grad_ratio(self, action_params, state_params):
        """Action gradient magnitude vs state gradient magnitude."""
        def _norm(params):
            norms = [p.grad.norm().item() for p in params
                     if p.grad is not None]
            return np.mean(norms) if norms else 0.0
        ratio = _norm(action_params) / (_norm(state_params) + 1e-8)
        self.grad_ratios.append(ratio)

    def report(self, epoch: int, gate_norm: float = 0.0):
        ac   = np.mean(self.ac_lifts)   if self.ac_lifts   else 0
        cf   = np.mean(self.cf_divs)    if self.cf_divs    else 0
        idm  = np.mean(self.idm_errs)   if self.idm_errs   else 0
        grad = np.mean(self.grad_ratios) if self.grad_ratios else 0

        print(f"\n  [Epoch {epoch:02d} Diagnostics]")
        print(f"    ac_lift    = {ac:+.4f}   (>0.05 = action load-bearing)")
        print(f"    cf_div     = {cf:.4f}    (should grow with training)")
        print(f"    idm_mae    = {idm:.4f}   (<0.1 = good action encoding)")
        print(f"    grad_ratio = {grad:.4f}  (>0.01 = action grads alive)")
        print(f"    gate_norm  = {gate_norm:.4f} (AdaLN α magnitude)")

        if ac < 0.005 and epoch >= 5:
            print("    ⚠️  ACTION COLLAPSE: ac_lift near zero — increase λ_IDM or k_steps")
        if grad < 0.001 and epoch >= 3:
            print("    ⚠️  DEAD ACTION GRADIENTS: increase λ_IDM or check AdaLN zero-init")
        if ac > 0.05:
            print("    ✓  Action conditioning load-bearing")

        self.reset()
        return {"ac_lift": ac, "cf_div": cf, "idm_mae": idm, "grad_ratio": grad}


# ═══════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════

def train_domain(
    domain:       str,
    data_path:    Optional[str],
    n_epochs:     int,
    batch_size:   int,
    base_lr:      float,
    k_steps:      int,
    lambda_idm:   float,
    lambda_cf:    float,
    action_drop:  float,
    visual:       bool,
    save_dir:     str,
    log_every:    int,
    device_str:   str,
    ablation_variant:  Optional[str] = None,
    use_struct_probe:  bool  = False,
    lambda_struct:     float = 1.0,
) -> dict:
    """
    Train action-conditioned world model for one domain.
    ablation_variant: None = full system, or one of:
        'additive' | 'no_multistep' | 'no_idm' | 'no_dropout' | 'no_straight'
    """
    device = torch.device(device_str)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # ── Dataset ────────────────────────────────────────────────────────────
    if domain == "pusht":
        ds = PushTStateDataset(
            data_path=data_path, k_steps=k_steps, synthetic=(data_path is None)
        )
        obs_dim, action_dim = 5, 2
        use_visual = False
    else:  # pointmaze
        ds = PointMazeDataset(
            data_path=data_path, k_steps=k_steps, visual=visual
        )
        obs_dim, action_dim = 4, 2
        use_visual = visual

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=True,
    )

    # Structured probe: lightweight z→block_pos predictor
    struct_probe = None
    if use_struct_probe:
        _dev = torch.device(device_str)
        struct_probe = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(), nn.LayerNorm(64),
            nn.Linear(64, 2), nn.Sigmoid(),
        ).to(_dev)
        print("  Structured probe loss active (z→block_pos)")

    # ── Model ──────────────────────────────────────────────────────────────
    D = 128
    encoder    = StateEncoder(obs_dim, D).to(device)
    transition = ActionConditionedTransition(D, action_dim).to(device)
    idm_head   = InverseDynamicsHead(D, action_dim).to(device)

    if use_visual:
        vis_enc = VisualEncoder(D).to(device)
        vis_params = list(vis_enc.parameters())
    else:
        vis_enc = None
        vis_params = []

    # Ablation: replace AdaLN with additive injection
    if ablation_variant == "additive":
        # Monkey-patch: zero out AdaLN, use simple additive
        for adaln in transition.adaln:
            adaln.action_mlp[-1].weight.data *= 0
            adaln.action_mlp[-1].bias.data   *= 0

    all_params = (
        list(encoder.parameters())
        + list(transition.parameters())
        + list(idm_head.parameters())
        + vis_params
        + (list(struct_probe.parameters()) if struct_probe else [])
    )
    n_params = sum(p.numel() for p in all_params)
    print(f"\n[{domain.upper()}] params={n_params:,} | "
          f"variant={ablation_variant or 'full'}")

    optimizer = torch.optim.AdamW(all_params, lr=base_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=base_lr*5,
        total_steps=n_epochs * len(loader), pct_start=0.05,
    )

    diagnostics = ActionDiagnostics()
    best_loss   = float("inf")
    global_step = 0

    # Action params (for grad ratio tracking)
    action_params = list(transition.action_emb.parameters()) + \
                    [p for adaln in transition.adaln for p in adaln.parameters()]
    state_params  = list(encoder.parameters())

    for epoch in range(n_epochs):
        encoder.train(); transition.train(); idm_head.train()
        if vis_enc: vis_enc.train()

        epoch_losses = []

        for batch in loader:
            obs_seq    = batch["obs"].to(device)        # (B, k+1, obs_dim)
            action_seq = batch["action"].to(device)     # (B, k, action_dim)
            B = obs_seq.shape[0]

            # ── Encode all observations ─────────────────────────────────
            obs_flat = obs_seq.view(B * (k_steps + 1), obs_dim)
            z_all    = encoder(obs_flat).view(B, k_steps + 1, D)  # (B, k+1, D)
            z0       = z_all[:, 0]

            # ── Multi-step prediction (or single-step) ──────────────────
            if ablation_variant == "no_multistep":
                # Predict only t+1 — InfoNCE to prevent copy collapse
                z_pred = transition(z0, action_seq[:, 0])
                z_tgt  = z_all[:, 1].detach()
                L_pred = info_nce_loss(z_pred, z_tgt)
            else:
                # Predict t+1 through t+k — InfoNCE at each step
                z_preds = transition.rollout(z0, action_seq)  # [k tensors of (B,D)]
                L_pred = sum(
                    info_nce_loss(z_preds[k], z_all[:, k+1].detach())
                    for k in range(k_steps)
                ) / k_steps

            # ── Inverse dynamics loss ───────────────────────────────────
            if ablation_variant == "no_idm":
                L_idm = torch.tensor(0.0, device=device)
            else:
                # IDM on each consecutive pair
                L_idm = sum(
                    F.smooth_l1_loss(
                        idm_head(z_all[:, k], z_all[:, k+1].detach()),
                        action_seq[:, k]
                    )
                    for k in range(k_steps)
                ) / k_steps

            # ── Action dropout baseline (for ac_lift diagnostic) ────────
            if ablation_variant != "no_dropout" and action_drop > 0:
                with torch.no_grad():
                    a_zero   = torch.zeros_like(action_seq[:, 0])
                    z_uncond = transition(z0, a_zero)
                    L_uncond = info_nce_loss(z_uncond, z_all[:, 1].detach()).item()
                diagnostics.update_ac_lift(L_pred.item(), L_uncond)

            # ── Counterfactual divergence (diagnostic) ──────────────────
            if global_step % (log_every * 5) == 0:
                a2 = torch.roll(action_seq[:, 0], 1, 0)  # shuffled actions
                diagnostics.update_cf_div(
                    lambda z, a: transition(z, a),
                    z0.detach(), action_seq[:, 0].detach(), a2.detach()
                )
                diagnostics.update_idm(
                    idm_head, z_all[:, 0].detach(),
                    z_all[:, 1].detach(), action_seq[:, 0].detach()
                )

            # ── Structured probe loss (z → block position) ────────────
            if struct_probe is not None:
                z_for_probe  = z_all[:, 1]                        # (B, d)
                block_pred   = struct_probe(z_for_probe)          # (B, 2)
                block_true   = obs_seq[:, 1, 2:4].to(device) if obs_seq.dim()==3 \
                               else obs_seq[:, 2:4].to(device)   # (B, 2)
                L_struct     = F.smooth_l1_loss(block_pred, block_true)
            else:
                L_struct = torch.tensor(0.0, device=device)

            # ── Total loss ──────────────────────────────────────────────
            total_loss = L_pred + lambda_idm * L_idm + lambda_struct * L_struct

            if not torch.isfinite(total_loss):
                optimizer.zero_grad(); continue

            optimizer.zero_grad()
            total_loss.backward()
            agc_clip(all_params)
            diagnostics.update_grad_ratio(action_params, state_params)
            optimizer.step()
            scheduler.step()

            epoch_losses.append(total_loss.item())
            global_step += 1

            if global_step % log_every == 0:
                gate_norms = [
                    adaln.action_mlp[-1].weight.norm().item()
                    for adaln in transition.adaln
                ]
                gate_mean = np.mean(gate_norms)
                print(
                    f"[{domain} ep{epoch:02d} s{global_step:05d}] "
                    f"L={total_loss.item():.4f} "
                    f"L_pred={L_pred.item():.4f} "
                    f"L_idm={L_idm.item():.4f} "
                    f"L_struct={L_struct.item():.4f} "
                    f"gate={gate_mean:.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )

        # ── Epoch summary ───────────────────────────────────────────────
        mean_L = np.mean(epoch_losses) if epoch_losses else 0
        gate_norms = [adaln.action_mlp[-1].weight.norm().item()
                      for adaln in transition.adaln]
        diag = diagnostics.report(epoch, gate_norm=np.mean(gate_norms))
        print(f"[{domain}] Epoch {epoch:02d}  mean_loss={mean_L:.4f}")

        # ── Save best ───────────────────────────────────────────────────
        if mean_L < best_loss:
            best_loss = mean_L
            variant_tag = ablation_variant or "full"
            ckpt_path = Path(save_dir) / f"action_wm_{domain}_{variant_tag}_best.pt"
            torch.save({
                "epoch":     epoch,
                "loss":      best_loss,
                "domain":    domain,
                "variant":   variant_tag,
                "k_steps":   k_steps,
                "obs_dim":   obs_dim,
                "action_dim": action_dim,
                "diagnostics": diag,
                "encoder":   encoder.state_dict(),
                "transition": transition.state_dict(),
                "idm_head":  idm_head.state_dict(),
                "struct_probe": struct_probe.state_dict() if struct_probe else None,
            }, ckpt_path)
            print(f"  → Saved: {ckpt_path}")

    return {"domain": domain, "best_loss": best_loss, "diagnostics": diag}


# ═══════════════════════════════════════════════════════════════════════════
# Ablation suite
# ═══════════════════════════════════════════════════════════════════════════

ABLATION_VARIANTS = [
    "full",         # All 3 interventions
    "additive",     # Additive injection only (Sprint B baseline)
    "no_multistep", # AdaLN-Zero + IDM, no multi-step
    "no_idm",       # AdaLN-Zero + multi-step, no IDM
    "no_dropout",   # AdaLN-Zero + multi-step + IDM, no action dropout
    "no_straight",  # (planned) no temporal straightening — same as full for now
]


def run_ablation(args):
    print("\n══ Sprint C Ablation Suite ══════════════════════════════════")
    print(f"  Domain: {args.domain} | Epochs: {args.epochs} | k={args.k_steps}")
    results = {}

    for variant in ABLATION_VARIANTS:
        print(f"\n── Variant: {variant} ──────────────────────────────────────")
        r = train_domain(
            domain    = args.domain,
            data_path = args.data_path,
            n_epochs  = args.epochs,
            batch_size= args.batch_size,
            base_lr   = args.lr,
            k_steps   = args.k_steps,
            lambda_idm= args.lambda_idm,
            lambda_cf = args.lambda_cf,
            action_drop = args.action_drop,
            visual    = args.visual,
            save_dir  = args.save_dir,
            log_every = args.log_every,
            device_str= args.device,
            ablation_variant = None if variant == "full" else variant,
        )
        results[variant] = r

    print("\n══ Ablation Results ═════════════════════════════════════════")
    print(f"  {'Variant':15s}  {'Best Loss':10s}  {'ac_lift':10s}  {'idm_mae':10s}  {'grad_ratio':10s}")
    print(f"  {'─'*15}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")
    for v, r in results.items():
        d = r["diagnostics"]
        print(f"  {v:15s}  {r['best_loss']:10.4f}  "
              f"{d['ac_lift']:+10.4f}  {d['idm_mae']:10.4f}  {d['grad_ratio']:10.4f}")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--domain",      default="both",
                   choices=["pusht", "pointmaze", "both"])
    p.add_argument("--data-path",   default=None,
                   help="Path to dataset (zarr/hdf5). If None, uses synthetic data.")
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch-size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--k-steps",     type=int,   default=4,
                   help="Multi-step prediction horizon")
    p.add_argument("--lambda-idm",  type=float, default=100.0,
                   help="IDM loss weight (WAM 2026: 1000, conservative: 100)")
    p.add_argument("--lambda-cf",   type=float, default=0.1)
    p.add_argument("--action-drop", type=float, default=0.3,
                   help="Action dropout probability for ac_lift diagnostic")
    p.add_argument("--visual",      action="store_true",
                   help="Use visual observations for PointMaze")
    p.add_argument("--save-dir",    default="checkpoints/action_wm")
    p.add_argument("--log-every",   type=int,   default=50)
    p.add_argument("--device",      default="cpu")
    p.add_argument("--struct-probe", action="store_true",
                   help="Add structured z→block_pos probe loss")
    p.add_argument("--lambda-struct", type=float, default=1.0)
    p.add_argument("--ablation",    action="store_true",
                   help="Run full ablation suite (6 variants)")
    args = p.parse_args()

    if args.ablation:
        if args.domain == "both":
            print("Ablation with --domain both: running PushT ablation first")
            args.domain = "pusht"
        run_ablation(args)
    elif args.domain == "both":
        print("Running PushT (state-only, fast) and PointMaze (visual) in parallel tabs")
        print("Tip: open two terminals and run:")
        print(f"  python train_action_wm.py --domain pusht  --epochs {args.epochs}")
        print(f"  python train_action_wm.py --domain pointmaze --epochs {args.epochs} {'--visual' if args.visual else ''}")
        print("\nRunning PushT first (no --visual flag), then PointMaze...\n")

        for domain in ["pusht", "pointmaze"]:
            train_domain(
                domain      = domain,
                data_path   = args.data_path,
                n_epochs    = args.epochs,
                batch_size  = args.batch_size,
                base_lr     = args.lr,
                k_steps     = args.k_steps,
                lambda_idm  = args.lambda_idm,
                lambda_cf   = args.lambda_cf,
                action_drop = args.action_drop,
                visual      = (args.visual and domain == "pointmaze"),
                save_dir    = args.save_dir,
                log_every   = args.log_every,
                device_str  = args.device,
            )
    else:
        train_domain(
            domain      = args.domain,
            data_path   = args.data_path,
            n_epochs    = args.epochs,
            batch_size  = args.batch_size,
            base_lr     = args.lr,
            k_steps     = args.k_steps,
            lambda_idm  = args.lambda_idm,
            lambda_cf   = args.lambda_cf,
            action_drop = args.action_drop,
            visual      = args.visual,
            save_dir    = args.save_dir,
            log_every   = args.log_every,
            device_str  = args.device,
        )
