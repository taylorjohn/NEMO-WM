"""
hybrid_predictor.py  —  NeMo-WM Hybrid World Model
===================================================
DA-gated router that selects between flat MLP (Sprint C) and
graph GNN (Sprint D) predictors based on neuromodulator state.

Core principle:
  High DA (surprise/contact) → graph mode (contact-aware, relational)
  Low DA  (smooth motion)    → flat mode  (fast, efficient)
  ACh                        → controls context window k in both modes

The routing threshold θ is tunable and can be learned. Contact detection
fires the DA spike before the predictor is called, so the router adapts
within the same timestep — no lag.

Integration:
  - Drop-in replacement for ActionConditionedTransition in train_action_wm.py
  - Drop-in replacement for GraphDynamicsPredictor in train_graph_wm.py
  - GRASP planner calls .forward() and .rollout() unchanged

Usage:
    from hybrid_predictor import HybridPredictor, ContactDetector, NeuroRouter

    hybrid = HybridPredictor(
        flat_ckpt  = 'checkpoints/action_wm/action_wm_pusht_full_best.pt',
        graph_ckpt = 'checkpoints/graph_wm/graph_wm_pusht_best.pt',
        da_threshold = 0.6,
    )

    # Same interface as both individual models
    z_next = hybrid(obs, action, neuro_state)
    preds  = hybrid.rollout(obs, actions, neuro_state)
"""

import math
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# Contact detector
# ═══════════════════════════════════════════════════════════════════════════

class ContactDetector:
    """
    Detects contact events from raw observations.
    Returns a DA-modulated surprise signal in [0, 1].

    Contact phases for PushT:
      Phase 0: agent far from block → smooth navigation → DA low
      Phase 1: agent reaches block  → contact spike → DA high
      Phase 2: pushing block        → sustained contact → DA medium
      Phase 3: block near goal      → task completion spike → DA high

    The spike shape mirrors biological DA: sharp onset, exponential decay.
    """

    def __init__(
        self,
        contact_thr:  float = 0.15,   # normalised distance threshold
        goal_thr:     float = 0.08,   # goal proximity threshold
        decay:        float = 0.85,   # DA decay per step
        spike_height: float = 1.0,    # DA spike magnitude on contact
    ):
        self.contact_thr  = contact_thr
        self.goal_thr     = goal_thr
        self.decay        = decay
        self.spike_height = spike_height
        self._da          = 0.0
        self._prev_contact = False
        self._prev_near_goal = False

    def update(self, obs: np.ndarray, goal: np.ndarray) -> float:
        """
        obs:  (5,) normalised PushT obs — (ax, ay, bx, by, angle_norm)
        goal: (2,) normalised goal position
        Returns: DA signal in [0, 1]
        """
        agent = obs[:2]
        block = obs[2:4]

        agent_block_dist = float(np.linalg.norm(agent - block))
        block_goal_dist  = float(np.linalg.norm(block - goal))

        contact   = agent_block_dist < self.contact_thr
        near_goal = block_goal_dist  < self.goal_thr

        # Spike on contact onset or goal proximity onset
        spike = 0.0
        if contact and not self._prev_contact:
            spike = self.spike_height          # contact onset
        elif near_goal and not self._prev_near_goal:
            spike = self.spike_height * 0.8    # goal proximity onset
        elif contact:
            spike = self.spike_height * 0.3    # sustained contact (lower)

        self._da = min(1.0, self._da * self.decay + spike)
        self._prev_contact   = contact
        self._prev_near_goal = near_goal
        return self._da

    def reset(self):
        self._da = 0.0
        self._prev_contact    = False
        self._prev_near_goal  = False

    @property
    def da(self) -> float:
        return self._da



# ═══════════════════════════════════════════════════════════════════════════
# Goal-directed DA — hot/cold shaping signal
# ═══════════════════════════════════════════════════════════════════════════

class GoalDA:
    """
    DA as continuous hot/cold guidance signal toward goal.

    Maps block-to-goal distance delta onto DA:
      Getting closer  → DA rises  (warm/hot)
      Getting further → DA falls  (cold)
      Contact onset   → DA spike  (hot — found block)
      Goal reached    → DA spike  (found it)
      Neutral         → slow decay

    This is reward prediction error (RPE) — the core biological DA signal.
    Not the reward itself, but the *change* in expected reward.

    Hierarchy of events:
      dist_delta > 0.01  (approaching block/goal) → +DA
      contact onset                                → +1.0 spike
      block near goal                              → +0.8
      goal reached                                 → +1.0 spike
      dist_delta < -0.01 (moving away)            → -DA

    ACh coupling: high DA → extend k_ctx (plan longer when warmer).
    This gives the planner a continuous gradient signal throughout
    the episode, not just at discrete contact events.
    """

    def __init__(
        self,
        decay:          float = 0.92,
        approach_scale: float = 4.0,   # DA gain for approaching
        retreat_scale:  float = 3.0,   # DA penalty for retreating
        contact_thr:    float = 0.15,
        goal_thr:       float = 0.10,
        spike_contact:  float = 1.0,
        spike_goal:     float = 1.0,
    ):
        self.decay          = decay
        self.approach_scale = approach_scale
        self.retreat_scale  = retreat_scale
        self.contact_thr    = contact_thr
        self.goal_thr       = goal_thr
        self.spike_contact  = spike_contact
        self.spike_goal     = spike_goal

        self._da             = 0.5   # start neutral
        self._prev_agent_block = None
        self._prev_block_goal  = None
        self._prev_contact     = False
        self._prev_near_goal   = False
        self._history: List[float] = []

    def update(self, obs: np.ndarray, goal: np.ndarray) -> float:
        """
        obs:  (5,) normalised — (ax, ay, bx, by, angle_norm)
        goal: (2,) normalised goal position
        Returns: DA signal in [0, 1]
        """
        agent = obs[:2]
        block = obs[2:4]

        dist_ab = float(np.linalg.norm(agent - block))
        dist_bg = float(np.linalg.norm(block - goal))

        contact   = dist_ab < self.contact_thr
        near_goal = dist_bg < self.goal_thr

        # ── Spike events (discrete) ───────────────────────────────────────
        spike = 0.0
        if contact and not self._prev_contact:
            spike = self.spike_contact          # contact onset — hot
        if near_goal and not self._prev_near_goal:
            spike = max(spike, self.spike_goal) # goal proximity onset

        # ── Continuous shaping (delta-based) ──────────────────────────────
        shaping = 0.0

        # Phase 1: agent approaching block (before contact)
        if not contact and self._prev_agent_block is not None:
            delta_ab = self._prev_agent_block - dist_ab   # + = closer
            if delta_ab > 0.005:
                shaping += delta_ab * self.approach_scale * 0.5
            elif delta_ab < -0.005:
                shaping += delta_ab * self.retreat_scale  * 0.5  # negative

        # Phase 2: block approaching goal (after contact)
        if contact and self._prev_block_goal is not None:
            delta_bg = self._prev_block_goal - dist_bg    # + = closer
            if delta_bg > 0.003:
                shaping += delta_bg * self.approach_scale
            elif delta_bg < -0.003:
                shaping += delta_bg * self.retreat_scale  # negative

        # ── DA update ─────────────────────────────────────────────────────
        # Decay baseline, add spike, add shaping
        self._da = self._da * self.decay + spike + shaping
        self._da = float(np.clip(self._da, 0.0, 1.0))

        # Store state
        self._prev_agent_block = dist_ab
        self._prev_block_goal  = dist_bg
        self._prev_contact     = contact
        self._prev_near_goal   = near_goal
        self._history.append(self._da)

        return self._da

    def reset(self):
        self._da             = 0.5
        self._prev_agent_block = None
        self._prev_block_goal  = None
        self._prev_contact     = False
        self._prev_near_goal   = False
        self._history.clear()

    @property
    def da(self) -> float:
        return self._da

    def temperature(self) -> str:
        """Human-readable hot/cold label for logging."""
        if self._da > 0.8:   return "🔥 HOT"
        if self._da > 0.6:   return "♨  WARM"
        if self._da > 0.4:   return "〜 NEUTRAL"
        if self._da > 0.2:   return "❄  COOL"
        return                       "🧊 COLD"

    def summary(self) -> dict:
        if not self._history:
            return {}
        h = np.array(self._history)
        return {
            "da_mean":  float(h.mean()),
            "da_max":   float(h.max()),
            "da_min":   float(h.min()),
            "hot_pct":  float((h > 0.6).mean()),
            "cold_pct": float((h < 0.3).mean()),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Neuromodulator state (lightweight, compatible with neuromodulator_base.py)
# ═══════════════════════════════════════════════════════════════════════════

class NeuroState:
    """
    Minimal neuromodulator state for the hybrid predictor.
    Compatible with the full NeuromodulatorBase interface.
    """
    def __init__(self):
        self.da       = 0.5    # dopamine — surprise / novelty
        self.ach      = 0.5    # acetylcholine — context window width
        self.sht      = 1.0    # serotonin — stability
        self.ne       = 0.2    # norepinephrine — gain
        self.cortisol = 0.0    # stress / adversity
        self.regime   = "EXPLOIT"
        self.k_ctx    = 16     # ACh-controlled context window


class NeuroRouter:
    """
    Routes between flat and graph predictors based on DA state.

    Routing logic:
      DA > da_threshold  → graph mode (contact, surprise)
      DA <= da_threshold → flat mode  (smooth motion)

    ACh controls k_ctx in both modes:
      high ACh → large k_ctx (broad temporal context)
      low ACh  → small k_ctx (local context)

    The threshold θ can be learned by backpropagating through
    a soft Gumbel-sigmoid mixture instead of hard routing.
    """

    K_CTX_SCHEDULE = {
        # ACh → k_ctx mapping (mirrors the validated sweep)
        0.0: 2,
        0.2: 4,
        0.4: 8,
        0.6: 16,
        0.8: 32,
        1.0: 32,
    }

    def __init__(
        self,
        da_threshold:  float = 0.6,
        soft_routing:  bool  = False,   # if True, use sigmoid mixture
        temperature:   float = 0.1,     # Gumbel temperature for soft routing
    ):
        self.θ        = da_threshold
        self.soft     = soft_routing
        self.temp     = temperature
        self._history: List[str] = []

    def route(self, neuro: NeuroState) -> Tuple[str, float, int]:
        """
        Returns (mode, da, k_ctx).
        mode: 'flat' | 'graph' | 'soft' (when soft_routing=True)
        """
        # ACh → k_ctx
        k_ctx = self._ach_to_k(neuro.ach)

        if self.soft:
            # Soft mixture weight: w_graph = sigmoid((DA - θ) / τ)
            w_graph = torch.sigmoid(
                torch.tensor((neuro.da - self.θ) / self.temp)
            ).item()
            mode = "soft"
        else:
            w_graph = 1.0 if neuro.da > self.θ else 0.0
            mode = "graph" if neuro.da > self.θ else "flat"

        self._history.append(mode)
        return mode, w_graph, k_ctx

    def _ach_to_k(self, ach: float) -> int:
        """Interpolate ACh → k_ctx from validated sweep."""
        keys = sorted(self.K_CTX_SCHEDULE.keys())
        for i in range(len(keys) - 1):
            if keys[i] <= ach <= keys[i+1]:
                t = (ach - keys[i]) / (keys[i+1] - keys[i])
                k0 = self.K_CTX_SCHEDULE[keys[i]]
                k1 = self.K_CTX_SCHEDULE[keys[i+1]]
                return int(round(k0 + t * (k1 - k0)))
        return self.K_CTX_SCHEDULE[keys[-1]]

    def stats(self) -> dict:
        if not self._history:
            return {}
        total = len(self._history)
        return {
            "flat_pct":  sum(1 for m in self._history if m == "flat")  / total,
            "graph_pct": sum(1 for m in self._history if m == "graph") / total,
            "total":     total,
        }

    def reset_stats(self):
        self._history.clear()


# ═══════════════════════════════════════════════════════════════════════════
# Flat model interface (Sprint C)
# ═══════════════════════════════════════════════════════════════════════════

class FlatModelWrapper(nn.Module):
    """
    Wraps the Sprint C ActionConditionedTransition for use in HybridPredictor.
    Loads from checkpoint, exposes .predict(z, action) → z_next.
    """

    def __init__(self, ckpt_path: str, device: torch.device):
        super().__init__()
        # Import here to avoid circular dependency
        from train_action_wm import StateEncoder, ActionConditionedTransition
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        obs_dim    = ckpt.get("obs_dim",    5)
        action_dim = ckpt.get("action_dim", 2)
        D = 128
        self.encoder    = StateEncoder(obs_dim, D)
        self.transition = ActionConditionedTransition(D, action_dim)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.transition.load_state_dict(ckpt["transition"])
        self.d_model = D
        self.obs_dim = obs_dim
        print(f"  Flat model loaded: ep={ckpt.get('epoch','?')} "
              f"ac_lift={ckpt.get('diagnostics',{}).get('ac_lift',0):+.4f}")

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, obs_dim) → (B, d_model)"""
        return self.encoder(obs)

    def predict(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """z: (B, d_model), action: (B, 2) → (B, d_model)"""
        return self.transition(z, action)

    def rollout(self, z0: torch.Tensor, actions: torch.Tensor) -> List[torch.Tensor]:
        """z0: (B, d), actions: (B, K, 2) → list of K tensors"""
        return self.transition.rollout(z0, actions)


# ═══════════════════════════════════════════════════════════════════════════
# Graph model interface (Sprint D)
# ═══════════════════════════════════════════════════════════════════════════

class GraphModelWrapper(nn.Module):
    """
    Wraps the Sprint D GraphDynamicsPredictor for use in HybridPredictor.
    Loads from checkpoint, exposes .predict(nodes, action) → nodes_next.
    Also computes per-node uncertainty for adaptive horizon.
    """

    def __init__(self, ckpt_path: str, device: torch.device):
        super().__init__()
        from train_graph_wm import GraphDynamicsPredictor, SelectiveAttentionPlanner
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.predictor = GraphDynamicsPredictor()
        self.attn_plan = SelectiveAttentionPlanner()
        self.predictor.load_state_dict(ckpt["predictor"])
        self.attn_plan.load_state_dict(ckpt["attn_plan"])
        print(f"  Graph model loaded: ep={ckpt.get('epoch','?')} "
              f"ac_lift={ckpt.get('ac_lift',0):+.4f}")

    def encode(self, raw_nodes: torch.Tensor) -> torch.Tensor:
        """raw_nodes: (B, N, 4) → (B, N, d_node)"""
        return self.predictor.encode(raw_nodes)

    def predict(self, nodes: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """nodes: (B, N, d), action: (B, 2) → (B, N, d)"""
        return self.predictor(nodes, action)

    def rollout(self, nodes0: torch.Tensor, actions: torch.Tensor) -> List[torch.Tensor]:
        return self.predictor.rollout(nodes0, actions)

    def get_edge_attention(self) -> torch.Tensor:
        return self.predictor.get_edge_attention()

    def node_uncertainty(
        self,
        nodes: torch.Tensor,
        action: torch.Tensor,
        n_samples: int = 5,
    ) -> torch.Tensor:
        """
        Monte Carlo uncertainty via dropout.
        Returns per-node variance (B, N) — used for adaptive horizon.
        Requires dropout layers (add to GraphDynamicsPredictor for production).
        """
        preds = []
        with torch.no_grad():
            self.predictor.train()   # enable dropout
            for _ in range(n_samples):
                preds.append(self.predictor(nodes, action))
            self.predictor.eval()
        stacked = torch.stack(preds, dim=0)  # (n_samples, B, N, d)
        return stacked.var(dim=0).mean(dim=-1)  # (B, N)


# ═══════════════════════════════════════════════════════════════════════════
# Adaptive horizon
# ═══════════════════════════════════════════════════════════════════════════

class AdaptiveHorizon:
    """
    Maps node uncertainty → planning horizon H.

    High uncertainty (contact, phase change) → longer horizon (explore).
    Low uncertainty (smooth) → shorter horizon (exploit, cheaper).

    Mirrors the ACh temporal integration finding:
    broader context window = better localisation at low frequencies.
    Here: higher uncertainty = longer planning horizon needed.
    """

    def __init__(
        self,
        base_h:   int   = 8,
        max_h:    int   = 16,
        scale:    float = 2.0,
        ema_decay: float = 0.9,
    ):
        self.base_h   = base_h
        self.max_h    = max_h
        self.scale    = scale
        self._mean_unc = 0.1   # running mean uncertainty

    def update(self, uncertainty: float) -> int:
        """
        uncertainty: scalar epistemic uncertainty from graph model
        Returns: planning horizon H
        """
        self._mean_unc = 0.9 * self._mean_unc + 0.1 * uncertainty
        if self._mean_unc < 1e-6:
            return self.base_h
        ratio = uncertainty / (self._mean_unc + 1e-8)
        h = self.base_h + self.scale * max(0.0, ratio - 1.0)
        return int(np.clip(round(h), self.base_h, self.max_h))


# ═══════════════════════════════════════════════════════════════════════════
# Hybrid predictor
# ═══════════════════════════════════════════════════════════════════════════

class HybridPredictor(nn.Module):
    """
    DA-gated hybrid world model.

    Routes each prediction through either the flat MLP (Sprint C)
    or the graph GNN (Sprint D) based on the current DA signal.

    For soft routing (differentiable): output = w_graph * graph_pred
                                              + (1 - w_graph) * flat_pred
    For hard routing (inference):      output = graph_pred  if DA > θ
                                              = flat_pred   if DA <= θ

    The flat model operates on flattened obs vectors.
    The graph model operates on node-structured observations.
    Both are projected to d_model=128 for the mixer.

    Pass neuro_state=None to use hard routing with DA=0 (always flat).
    """

    def __init__(
        self,
        flat_ckpt:    Optional[str] = None,
        graph_ckpt:   Optional[str] = None,
        da_threshold: float = 0.6,
        soft_routing: bool  = False,
        device_str:   str   = "cpu",
    ):
        super().__init__()
        self.device   = torch.device(device_str)
        self.router   = NeuroRouter(da_threshold, soft_routing)
        self.adaptive = AdaptiveHorizon()
        self.D        = 128

        self._flat_available  = False
        self._graph_available = False

        if flat_ckpt and Path(flat_ckpt).exists():
            self.flat = FlatModelWrapper(flat_ckpt, self.device).to(self.device)
            self._flat_available = True
        else:
            print(f"  Flat ckpt not found: {flat_ckpt} — flat mode unavailable")
            self.flat = None

        if graph_ckpt and Path(graph_ckpt).exists():
            self.graph = GraphModelWrapper(graph_ckpt, self.device).to(self.device)
            self._graph_available = True
        else:
            print(f"  Graph ckpt not found: {graph_ckpt} — graph mode unavailable")
            self.graph = None

        # Mixer: projects graph node mean → flat latent space for unified output
        if self._graph_available:
            self.graph_to_flat = nn.Linear(64, 128).to(self.device)

        self._last_mode = "flat"
        self._mode_counts = {"flat": 0, "graph": 0, "soft": 0}

    @torch.no_grad()
    def forward(
        self,
        obs:         np.ndarray,    # (obs_dim,) normalised flat observation
        action:      np.ndarray,    # (action_dim,) action
        neuro:       Optional[NeuroState] = None,
        raw_nodes:   Optional[np.ndarray] = None,  # (N, 4) if graph available
    ) -> np.ndarray:
        """
        Single-step prediction.
        Returns: next latent z (128-dim) or node embeddings (N, 64) depending on mode.
        """
        if neuro is None:
            neuro = NeuroState()

        mode, w_graph, k_ctx = self.router.route(neuro)
        self._last_mode = mode
        self._mode_counts[mode] = self._mode_counts.get(mode, 0) + 1

        obs_t    = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        action_t = torch.from_numpy(action.astype(np.float32)).unsqueeze(0).to(self.device)

        if mode == "flat" or not self._graph_available:
            z = self.flat.encode(obs_t)
            z_next = self.flat.predict(z, action_t)
            return z_next.squeeze(0).cpu().numpy()

        elif mode == "graph" or not self._flat_available:
            if raw_nodes is None:
                # Build graph from flat obs (PushT only)
                raw_nodes = self._obs_to_nodes(obs)
            nodes_t = torch.from_numpy(raw_nodes.astype(np.float32)).unsqueeze(0).to(self.device)
            nodes   = self.graph.encode(nodes_t)
            # Adaptive horizon based on uncertainty
            unc     = self.graph.node_uncertainty(nodes, action_t).mean().item()
            h       = self.adaptive.update(unc)
            nodes_next = self.graph.predict(nodes, action_t)
            # Project to flat latent for unified downstream use
            z_next = self.graph_to_flat(nodes_next.mean(dim=1))
            return z_next.squeeze(0).cpu().numpy()

        else:
            # Soft routing: weighted mixture
            z_flat  = self.flat.predict(self.flat.encode(obs_t), action_t)
            if raw_nodes is None:
                raw_nodes = self._obs_to_nodes(obs)
            nodes_t  = torch.from_numpy(raw_nodes.astype(np.float32)).unsqueeze(0).to(self.device)
            nodes    = self.graph.encode(nodes_t)
            z_graph  = self.graph_to_flat(self.graph.predict(nodes, action_t).mean(1))
            z_next   = (1 - w_graph) * z_flat + w_graph * z_graph
            return z_next.squeeze(0).cpu().numpy()

    def _obs_to_nodes(self, obs: np.ndarray) -> np.ndarray:
        """Convert flat PushT obs (5,) to graph nodes (3, 4)."""
        import math
        angle = obs[4] * 2 * math.pi
        nodes = np.zeros((3, 4), dtype=np.float32)
        nodes[0, :2] = obs[:2]                                   # agent
        nodes[1, :2] = obs[2:4]                                  # block
        nodes[1, 2]  = math.sin(angle)
        nodes[1, 3]  = math.cos(angle)
        nodes[2, :2] = np.array([0.65, 0.65])                    # goal
        return nodes

    def report(self):
        total = sum(self._mode_counts.values())
        if total == 0:
            return
        print("\n── Hybrid predictor routing report ─────────────────")
        for mode, count in self._mode_counts.items():
            pct = 100 * count / total
            print(f"  {mode:6s}: {count:5d} steps  ({pct:.1f}%)")
        ea = self.graph.get_edge_attention() if self._graph_available else None
        if ea is not None:
            print(f"  Edge attention — agent:{ea[0].mean():.3f}  "
                  f"block:{ea[1].mean():.3f}  goal:{ea[2].mean():.3f}")
        print(f"  DA threshold: θ={self.router.θ:.2f}")


# ═══════════════════════════════════════════════════════════════════════════
# Training: joint hybrid training
# ═══════════════════════════════════════════════════════════════════════════

def train_hybrid(
    domain:       str   = "pusht",
    flat_ckpt:    Optional[str] = None,
    graph_ckpt:   Optional[str] = None,
    data_path:    Optional[str] = None,
    n_epochs:     int   = 20,
    batch_size:   int   = 64,
    base_lr:      float = 1e-4,
    da_threshold: float = 0.6,
    save_dir:     str   = "checkpoints/hybrid",
    log_every:    int   = 100,
    device_str:   str   = "cpu",
):
    """
    Fine-tunes the hybrid predictor end-to-end.

    Uses the routing weight w_graph as a soft label — batches where
    contact is detected get higher w_graph, routing more gradient
    to the graph branch. This lets both models specialise on their
    respective regimes while sharing the encoder.
    """
    from train_action_wm import PushTStateDataset
    import numpy as np
    from torch.utils.data import DataLoader

    device = torch.device(device_str)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Build hybrid model
    hybrid = HybridPredictor(
        flat_ckpt    = flat_ckpt,
        graph_ckpt   = graph_ckpt,
        da_threshold = da_threshold,
        soft_routing = True,   # differentiable during training
        device_str   = device_str,
    )

    ds = PushTStateDataset(data_path=data_path, k_steps=4)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=0, drop_last=True)

    # Only train the router threshold and mixer — keep both models frozen
    trainable = []
    if hasattr(hybrid, "graph_to_flat"):
        trainable.extend(hybrid.graph_to_flat.parameters())

    print(f"\nHybrid fine-tune: {sum(p.numel() for p in trainable):,} trainable params")
    optimizer = torch.optim.AdamW(trainable, lr=base_lr)
    contact_det = ContactDetector()
    best_loss = float("inf")

    for epoch in range(n_epochs):
        epoch_losses = []
        graph_calls = flat_calls = 0

        for batch in loader:
            obs_seq = batch["obs"].to(device)    # (B, k+1, 5)
            action  = batch["action"].to(device) # (B, k, 2)
            B = obs_seq.shape[0]

            # Compute per-sample DA from contact detection
            da_vals = []
            for b in range(B):
                contact_det.reset()
                da = contact_det.update(
                    obs_seq[b, 0].cpu().numpy(),
                    np.array([0.65, 0.65])
                )
                da_vals.append(da)
            da_tensor = torch.tensor(da_vals, device=device)

            # Routing weights per sample
            w_graph = torch.sigmoid(
                (da_tensor - da_threshold) / hybrid.router.temp
            )  # (B,)

            # Flat prediction
            z_flat = None
            if hybrid._flat_available:
                flat_enc = hybrid.flat.encoder
                z0_flat  = flat_enc(obs_seq[:, 0])
                z_flat   = hybrid.flat.transition(z0_flat, action[:, 0])
                flat_calls += B

            # Graph prediction
            z_graph = None
            if hybrid._graph_available:
                import math
                nodes_raw = torch.zeros(B, 3, 4, device=device)
                o = obs_seq[:, 0]
                nodes_raw[:, 0, :2] = o[:, :2]
                nodes_raw[:, 1, :2] = o[:, 2:4]
                angle = o[:, 4] * 2 * math.pi
                nodes_raw[:, 1, 2] = torch.sin(angle)
                nodes_raw[:, 1, 3] = torch.cos(angle)
                nodes_raw[:, 2, :2] = 0.65
                nodes_enc  = hybrid.graph.encode(nodes_raw)
                nodes_next = hybrid.graph.predict(nodes_enc, action[:, 0])
                z_graph    = hybrid.graph_to_flat(nodes_next.mean(dim=1))
                graph_calls += B

            # Target: encode next obs with flat model
            with torch.no_grad():
                z_target = hybrid.flat.encoder(obs_seq[:, 1]).detach()

            # Mixture loss
            if z_flat is not None and z_graph is not None:
                w = w_graph.unsqueeze(-1)   # (B, 1)
                z_mix = (1 - w) * z_flat + w * z_graph
            elif z_flat is not None:
                z_mix = z_flat
            else:
                z_mix = z_graph

            # InfoNCE
            zp = F.normalize(z_mix,    dim=-1)
            zt = F.normalize(z_target, dim=-1)
            logits = torch.mm(zp, zt.T) / 0.1
            labels = torch.arange(B, device=device)
            loss = F.cross_entropy(logits, labels)

            if not torch.isfinite(loss):
                optimizer.zero_grad(); continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        mean_L = np.mean(epoch_losses) if epoch_losses else 0
        print(f"Epoch {epoch:02d}  loss={mean_L:.4f}  "
              f"flat={flat_calls} graph={graph_calls}")

        if mean_L < best_loss:
            best_loss = mean_L
            path = Path(save_dir) / f"hybrid_{domain}_best.pt"
            torch.save({
                "epoch":        epoch,
                "loss":         best_loss,
                "da_threshold": da_threshold,
                "graph_to_flat": hybrid.graph_to_flat.state_dict()
                    if hasattr(hybrid, "graph_to_flat") else None,
            }, path)
            print(f"  → Saved: {path}")

    return hybrid


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--flat-ckpt",
                   default="checkpoints/action_wm/action_wm_pusht_full_best.pt")
    p.add_argument("--graph-ckpt",
                   default="checkpoints/graph_wm/graph_wm_pusht_best.pt")
    p.add_argument("--da-threshold", type=float, default=0.6)
    p.add_argument("--train",        action="store_true")
    p.add_argument("--epochs",       type=int, default=20)
    p.add_argument("--device",       default="cpu")
    p.add_argument("--demo",         action="store_true")
    args = p.parse_args()

    if args.train:
        train_hybrid(
            flat_ckpt    = args.flat_ckpt,
            graph_ckpt   = args.graph_ckpt,
            da_threshold = args.da_threshold,
            n_epochs     = args.epochs,
            device_str   = args.device,
        )
    elif args.demo:
        # Quick demo: run 20 steps, show routing decisions
        print("\nHybrid predictor demo (synthetic PushT)...")
        hybrid = HybridPredictor(
            flat_ckpt    = args.flat_ckpt,
            graph_ckpt   = args.graph_ckpt,
            da_threshold = args.da_threshold,
            device_str   = args.device,
        )
        detector = ContactDetector()
        neuro    = NeuroState()
        rng      = np.random.RandomState(0)

        agent = np.array([0.2, 0.2])
        block = np.array([0.5, 0.5])
        goal  = np.array([0.75, 0.75])

        for step in range(20):
            obs    = np.array([agent[0], agent[1], block[0], block[1], 0.0], dtype=np.float32)
            action = np.clip(block - agent + rng.normal(0, 0.05, 2), -1, 1).astype(np.float32)

            neuro.da = detector.update(obs, goal)
            mode, w_graph, k_ctx = hybrid.router.route(neuro)

            print(f"  step {step:02d}  DA={neuro.da:.3f}  mode={mode:5s}  "
                  f"k_ctx={k_ctx:2d}  dist={np.linalg.norm(agent-block):.3f}")

            # Move agent toward block, then push block toward goal
            if np.linalg.norm(agent - block) > 0.12:
                agent += (block - agent) * 0.4 + rng.normal(0, 0.02, 2)
            else:
                push = goal - block
                agent += push * 0.3 + rng.normal(0, 0.02, 2)
                block = np.clip(block - (agent - block) * 0.25, 0, 1)
            agent = np.clip(agent, 0, 1)

        hybrid.report()
    else:
        print("Use --demo for routing demo, --train for fine-tuning")
        print("Example:")
        print("  python hybrid_predictor.py --demo")
        print("  python hybrid_predictor.py --train --epochs 20")
