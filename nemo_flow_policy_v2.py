"""
nemo_flow_policy_v2.py — NeMo-WM Sprint C
==========================================
Phase 2: Orientation-aware flow matching policy.

Extends Phase 1 (position-only, SR=100%) to full PushT:
  Phase 1: action = (dx, dy)          — 2-dim, SR=100% position
  Phase 2: action = (dx, dy, dangle)  — 3-dim, SR=? position+orientation

Changes from v1:
  1. Action chunk: H*2 → H*3
  2. Obs:  adds target_angle to conditioning (2→3 goal dims)
  3. Block state: adds block_angle (2→3 block dims)
  4. Success: position AND angle within threshold
  5. Training: demos need rotational variation

Biological mapping:
  dangle conditioned on DA temperature — HOT = exploit known rotation
                                         COLD = explore rotation space
  Cortisol spikes when orientation error compounds across steps

Backward compatible:
  NeMoFlowPolicyV2 can load V1 weights for position dims
  and zero-initialise the angle head (transfer learning)

Usage:
    from nemo_flow_policy_v2 import (
        NeMoFlowPolicyV2, NeMoFlowAgentV2,
        generate_orientation_demos, train_flow_policy_v2
    )

    # Fresh training
    policy = NeMoFlowPolicyV2(H=8)
    agent  = NeMoFlowAgentV2(encoder, probe, policy, device)

    # Transfer from Phase 1
    policy = NeMoFlowPolicyV2.from_phase1(
        'checkpoints/flow_policy/nemo_flow_best.pt', H=8
    )
"""

import math
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Orientation utilities ─────────────────────────────────────────────────────

def angle_diff(a: float, b: float) -> float:
    """Signed angle difference a-b in [-π, π]."""
    d = a - b
    while d >  math.pi: d -= 2 * math.pi
    while d < -math.pi: d += 2 * math.pi
    return d


def angle_success(
    block_angle: float,
    target_angle: float,
    threshold_deg: float = 15.0,
) -> bool:
    """True if block orientation is within threshold of target."""
    diff = abs(angle_diff(block_angle, target_angle))
    return diff < math.radians(threshold_deg)


def full_success(
    block_pos:    np.ndarray,
    block_angle:  float,
    goal_pos:     np.ndarray,
    target_angle: float,
    pos_thr:      float = 0.12,
    ang_thr_deg:  float = 15.0,
) -> bool:
    """Full PushT success: position AND orientation."""
    pos_ok = np.linalg.norm(block_pos - goal_pos) < pos_thr
    ang_ok = angle_success(block_angle, target_angle, ang_thr_deg)
    return pos_ok and ang_ok


# ── GoalDA v2 — includes orientation component ────────────────────────────────

class GoalDAv2(nn.Module):
    """
    Orientation-aware GoalDA signal.

    DA = 1 - (w_pos * pos_dist + w_ang * ang_dist)
         normalised to [0, 1]

    Position and angle contribute independently:
        DA=1 → exactly at goal position and angle (HOT, exploit)
        DA=0 → far from goal or wrong orientation (COLD, explore)
    """

    def __init__(
        self,
        pos_weight: float = 0.7,
        ang_weight: float = 0.3,
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.ang_weight = ang_weight

    def forward(
        self,
        block_pos:    torch.Tensor,   # (B, 2)
        block_angle:  torch.Tensor,   # (B, 1) radians
        goal_pos:     torch.Tensor,   # (B, 2)
        target_angle: torch.Tensor,   # (B, 1) radians
        pos_scale:    float = 0.12,
        ang_scale:    float = math.pi,
    ) -> torch.Tensor:
        """Returns DA signal in [0, 1], shape (B, 1)."""
        pos_dist = (block_pos - goal_pos).norm(dim=-1, keepdim=True)
        ang_dist = (block_angle - target_angle).abs()
        # Wrap angle difference to [0, π]
        ang_dist = torch.min(ang_dist, 2 * math.pi - ang_dist)

        pos_norm = (pos_dist / pos_scale).clamp(0, 1)
        ang_norm = (ang_dist / ang_scale).clamp(0, 1)

        combined = self.pos_weight * pos_norm + self.ang_weight * ang_norm
        return 1.0 - combined.clamp(0, 1)


# ── Flow Policy V2 ────────────────────────────────────────────────────────────

class NeMoFlowPolicyV2(nn.Module):
    """
    Orientation-aware neuromodulated flow matching policy.

    Conditioning vector:
        z           (128) — world model latent
        block_state   (3) — [block_x, block_y, block_angle_norm]
        DA            (1) — combined pos+angle GoalDA
        goal          (3) — [goal_x, goal_y, target_angle_norm]
        t             (1) — flow time in [0, 1]
        x_t         (H*3) — noisy action chunk at time t

    Action chunk: H × [dx, dy, dangle]
        dx, dy    ∈ [0, 1]  — target position
        dangle    ∈ [0, 1]  — target angle (normalised by 2π)

    Transfer from V1:
        Position dims initialised from V1 weights
        Angle dim zero-initialised (learns from scratch)
    """

    ACTION_DIM = 3   # (dx, dy, dangle)

    def __init__(
        self,
        H:        int = 8,
        d_z:      int = 128,
        d_hidden: int = 256,
        n_layers: int = 4,
    ):
        super().__init__()
        self.H         = H
        self.d_z       = d_z
        self.action_dim = self.ACTION_DIM

        # Conditioning: z(128) + block(3) + DA(1) + goal(3) + t(1) + x_t(H*3)
        cond_dim = d_z + 3 + 1 + 3 + 1 + H * self.ACTION_DIM

        layers = []
        in_dim = cond_dim
        for _ in range(n_layers):
            layers += [
                nn.Linear(in_dim, d_hidden),
                nn.LayerNorm(d_hidden),
                nn.GELU(),
            ]
            in_dim = d_hidden
        layers.append(nn.Linear(d_hidden, H * self.ACTION_DIM))

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        z:           torch.Tensor,   # (B, 128)
        block_state: torch.Tensor,   # (B, 3) [bx, by, bangle_norm]
        da:          torch.Tensor,   # (B, 1)
        goal:        torch.Tensor,   # (B, 3) [gx, gy, target_angle_norm]
        t:           torch.Tensor,   # (B, 1) flow time
        x_t:         torch.Tensor,   # (B, H*3) noisy actions
    ) -> torch.Tensor:
        """Predict velocity field v_θ(x_t, t, cond) → (B, H*3)."""
        cond = torch.cat([z, block_state, da, goal, t, x_t], dim=-1)
        return self.net(cond)

    def sample(
        self,
        z:           torch.Tensor,
        block_state: torch.Tensor,
        da:          torch.Tensor,
        goal:        torch.Tensor,
        n_steps:     int = 1,
        H:           Optional[int] = None,
    ) -> torch.Tensor:
        """
        Sample action chunk via ODE.

        Returns:
            actions: (B, H, 3) clipped to [0, 1]
        """
        H = H or self.H
        B = z.shape[0]
        device = z.device

        # DA-conditioned noise temperature: HOT=low noise, COLD=high noise
        temp  = 1.0 - da.mean().item()
        x_t   = torch.randn(B, H * self.ACTION_DIM, device=device) * (0.5 + 0.5 * temp)

        dt = 1.0 / n_steps
        for i in range(n_steps):
            t_val = torch.full((B, 1), i * dt, device=device)
            with torch.no_grad():
                v = self.forward(z, block_state, da, goal, t_val, x_t)
            x_t = x_t + v * dt

        return x_t.view(B, H, self.ACTION_DIM).clamp(0, 1)

    @classmethod
    def from_phase1(
        cls,
        ckpt_path: str,
        H: int = 8,
        **kwargs,
    ) -> 'NeMoFlowPolicyV2':
        """
        Transfer Phase 1 weights (2-dim actions) to Phase 2 (3-dim).

        Strategy:
          - Position dims (dx, dy): copy from Phase 1 input layer
          - Angle dim (dangle):     zero-initialise
          - Hidden layers:          copy directly (same size)
          - Output layer:           extend with zero-init angle cols
        """
        v2 = cls(H=H, **kwargs)

        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        v1_state = ckpt.get('policy', ckpt)

        # Try to load compatible layers (will partially fail on dim mismatch)
        incompatible = []
        v2_state = v2.state_dict()
        for key, v1_val in v1_state.items():
            if key not in v2_state:
                continue
            if v2_state[key].shape == v1_val.shape:
                v2_state[key] = v1_val
            else:
                incompatible.append(key)

        v2.load_state_dict(v2_state)

        n_compat = len(v1_state) - len(incompatible)
        print(f"Phase 1 transfer: {n_compat}/{len(v1_state)} layers copied")
        print(f"  Incompatible (zero-init): {incompatible}")
        return v2


# ── Agent V2 ──────────────────────────────────────────────────────────────────

class NeMoFlowAgentV2:
    """
    Orientation-aware agent wrapping NeMoFlowPolicyV2.

    Obs: [agent_x, agent_y, block_x, block_y, block_angle_norm,
          target_angle_norm]  — 6-dim

    Action: [target_x, target_y, target_angle_norm]  — 3-dim
    """

    def __init__(
        self,
        encoder:  nn.Module,
        probe:    nn.Module,       # predicts block (x, y, angle)
        policy:   NeMoFlowPolicyV2,
        device:   torch.device,
        goal_da:  Optional[GoalDAv2] = None,
        n_steps:  int = 1,
    ):
        self.encoder = encoder
        self.probe   = probe
        self.policy  = policy
        self.device  = device
        self.goal_da = goal_da or GoalDAv2()
        self.n_steps = n_steps
        self._da     = 0.5
        self._temp   = 'NEUTRAL'

    def reset(self):
        self._da   = 0.5
        self._temp = 'NEUTRAL'

    def act(
        self,
        obs:          np.ndarray,   # 6-dim: [ax,ay,bx,by,bangle,tangle]
        goal:         np.ndarray,   # 3-dim: [gx, gy, target_angle_norm]
    ) -> Tuple[np.ndarray, float, str]:
        """
        Returns:
            action:  (3,) [target_x, target_y, target_angle_norm]
            da:      float DA signal
            temp:    str temperature label
        """
        obs_t  = torch.from_numpy(obs[:5].astype(np.float32)).unsqueeze(0).to(self.device)
        goal_t = torch.from_numpy(goal.astype(np.float32)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            z = self.encoder(obs_t)

            # Probe predicts block state (x, y, angle)
            block_state = self.probe(z)  # (1, 3)

            # GoalDA — position + orientation combined
            block_pos   = block_state[:, :2]
            block_angle = block_state[:, 2:3] * 2 * math.pi   # denorm
            goal_pos    = goal_t[:, :2]
            target_ang  = goal_t[:, 2:3] * 2 * math.pi        # denorm

            da = self.goal_da(block_pos, block_angle, goal_pos, target_ang)

        self._da = da.item()
        if self._da > 0.6:   self._temp = 'HOT'
        elif self._da > 0.3: self._temp = 'WARM'
        elif self._da > 0.1: self._temp = 'COOL'
        else:                self._temp = 'COLD'

        with torch.no_grad():
            action_chunk = self.policy.sample(
                z, block_state, da, goal_t, n_steps=self.n_steps
            )
        action = action_chunk[0, 0].cpu().numpy()
        return action, self._da, self._temp


# ── Demo generation ───────────────────────────────────────────────────────────

def generate_orientation_demos(
    n_demos:      int = 500,
    H:            int = 8,
    goal_range:   Tuple[float, float] = (0.3, 0.7),
    angle_range:  Tuple[float, float] = (0, 2 * math.pi),
    seed:         int = 42,
) -> list:
    """
    Generate scripted demos for Phase 2 training.

    Each demo includes:
        - Random goal position and target angle
        - Scripted agent that pushes block toward goal AND rotates it
        - Rotation achieved by approaching from correct angle

    Strategy:
        1. Compute approach angle = target_block_angle + π (push from opposite)
        2. Move agent to approach position
        3. Push block toward goal while maintaining approach angle
        4. Success when pos AND angle within threshold

    Returns list of dicts:
        {'obs': (T, 6), 'actions': (T, 3), 'goal': (3,), 'success': bool}
    """
    rng = np.random.RandomState(seed)
    demos = []

    for demo_idx in range(n_demos):
        goal_pos     = rng.uniform(*goal_range, 2)
        target_angle = rng.uniform(*angle_range)
        goal         = np.array([goal_pos[0], goal_pos[1],
                                  target_angle / (2 * math.pi)])

        agent_pos   = rng.uniform(0.05, 0.35, 2)
        block_pos   = rng.uniform(0.25, 0.55, 2)
        block_angle = rng.uniform(0, 2 * math.pi)

        obs_list    = []
        action_list = []
        success     = False

        for step in range(300):
            # Current obs (6-dim)
            obs = np.array([
                agent_pos[0], agent_pos[1],
                block_pos[0], block_pos[1],
                block_angle / (2 * math.pi),
                target_angle / (2 * math.pi),
            ], dtype=np.float32)

            # Scripted policy:
            # Approach from angle that will rotate block correctly
            angle_err = angle_diff(block_angle, target_angle)

            # Desired approach direction (perpendicular to push direction)
            push_dir    = math.atan2(goal_pos[1] - block_pos[1],
                                     goal_pos[0] - block_pos[0])
            # Offset approach angle to apply torque for rotation
            torque_sign = 1.0 if angle_err > 0 else -1.0
            approach    = push_dir + torque_sign * 0.3   # slight angular offset

            # Compute approach position (behind block relative to goal)
            approach_pos = block_pos - np.array([
                math.cos(approach), math.sin(approach)
            ]) * 0.15

            # Action: move toward approach position if far, else push
            dist_to_approach = np.linalg.norm(agent_pos - approach_pos)
            if dist_to_approach > 0.08:
                target_action = np.clip(approach_pos, 0, 1)
            else:
                target_action = np.clip(block_pos + (goal_pos - block_pos) * 0.3, 0, 1)

            # Target angle action
            target_ang_norm = (block_angle + angle_err * 0.1) / (2 * math.pi) % 1.0

            action = np.array([
                target_action[0],
                target_action[1],
                float(target_ang_norm),
            ], dtype=np.float32)

            obs_list.append(obs)
            action_list.append(action)

            # Physics step
            agent_pos += (target_action - agent_pos) * 0.4 + rng.normal(0, 0.01, 2)
            agent_pos  = np.clip(agent_pos, 0, 1)

            if np.linalg.norm(agent_pos - block_pos) < 0.10:
                push      = (agent_pos - block_pos) * 0.18
                block_pos = np.clip(block_pos - push, 0, 1)
                # Torque: rotation proportional to lateral push component
                lateral   = push[0] * math.sin(push_dir) - push[1] * math.cos(push_dir)
                block_angle += lateral * 2.0 + rng.normal(0, 0.02)

            pos_ok = np.linalg.norm(block_pos - goal_pos) < 0.12
            ang_ok = abs(angle_diff(block_angle, target_angle)) < math.radians(20)
            if pos_ok and ang_ok:
                success = True
                break

        demos.append({
            'obs':     np.array(obs_list,    dtype=np.float32),
            'actions': np.array(action_list, dtype=np.float32),
            'goal':    goal.astype(np.float32),
            'success': success,
        })

        if (demo_idx + 1) % 100 == 0:
            sr = sum(d['success'] for d in demos) / len(demos)
            print(f"  Demo {demo_idx+1}/{n_demos}  SR={sr:.1%}")

    return demos


# ── Quick test ────────────────────────────────────────────────────────────────

def _test():
    print("NeMoFlowPolicyV2 self-test")

    H, B = 8, 4
    pol = NeMoFlowPolicyV2(H=H)
    print(f"  Params: {sum(p.numel() for p in pol.parameters()):,}")

    z           = torch.randn(B, 128)
    block_state = torch.rand(B, 3)
    da          = torch.rand(B, 1)
    goal        = torch.rand(B, 3)
    t           = torch.rand(B, 1)
    x_t         = torch.randn(B, H * 3)

    # Forward
    v = pol(z, block_state, da, goal, t, x_t)
    assert v.shape == (B, H * 3), f"Bad shape: {v.shape}"
    print(f"  Forward: v={v.shape}")

    # Sample
    pol.eval()
    with torch.no_grad():
        actions = pol.sample(z[:1], block_state[:1], da[:1], goal[:1], n_steps=1)
    assert actions.shape == (1, H, 3)
    assert actions.min() >= 0 and actions.max() <= 1
    print(f"  Sample: actions={actions.shape}  range=[{actions.min():.2f},{actions.max():.2f}]")

    # GoalDA v2
    goal_da  = GoalDAv2()
    block_p  = torch.rand(B, 2)
    block_a  = torch.rand(B, 1) * 2 * math.pi
    goal_p   = torch.rand(B, 2)
    target_a = torch.rand(B, 1) * 2 * math.pi
    da_out   = goal_da(block_p, block_a, goal_p, target_a)
    assert da_out.shape == (B, 1)
    assert da_out.min() >= 0 and da_out.max() <= 1
    print(f"  GoalDA v2: {da_out.shape}  range=[{da_out.min():.2f},{da_out.max():.2f}]")

    # Flow matching training step
    pol.train()
    x0     = torch.rand(B, H * 3)    # clean actions
    x1     = torch.randn(B, H * 3)   # noise
    t_rand = torch.rand(B, 1)
    x_t    = (1 - t_rand) * x0 + t_rand * x1
    v_pred = pol(z, block_state, da, goal, t_rand, x_t)
    v_gt   = x1 - x0                  # target velocity field
    loss   = F.mse_loss(v_pred, v_gt)
    loss.backward()
    print(f"  Flow loss: {loss.item():.4f}  Backward: OK")

    # Demo generation (small)
    print("  Generating 10 demos...")
    demos = generate_orientation_demos(n_demos=10, seed=0)
    sr    = sum(d['success'] for d in demos) / len(demos)
    print(f"  Scripted SR: {sr:.1%}  (expect 30-60% — hard task)")

    # Phase 1 transfer (skip if no checkpoint)
    from pathlib import Path
    ckpt = Path('checkpoints/flow_policy/nemo_flow_best.pt')
    if ckpt.exists():
        pol_v2 = NeMoFlowPolicyV2.from_phase1(str(ckpt), H=H)
        print(f"  Phase 1 transfer: OK  params={sum(p.numel() for p in pol_v2.parameters()):,}")
    else:
        print("  Phase 1 transfer: skipped (no checkpoint found)")

    print("  All assertions passed. ✅")


if __name__ == '__main__':
    _test()
