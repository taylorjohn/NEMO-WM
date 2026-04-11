"""
train_graph_wm.py  —  NeMo-WM Sprint D: Lifted Graph World Model
================================================================
Graph-structured world model for PushT and contact-rich 2D tasks.

Instead of a flat latent vector, the world is represented as a graph
of entities (nodes) with relational edges. The GNN dynamics predictor
propagates action effects along edges, capturing contact structure.

PushT graph (3 nodes):
  Node 0 — Agent        (x, y)              → 2-dim
  Node 1 — T-Block      (x, y, sin θ, cos θ) → 4-dim
  Node 2 — Goal region  (x, y)              → 2-dim (static)

Edges: fully connected (all pairs), learned edge weights.

Three interventions (same as Sprint C, now at node level):
  1. AdaLN-Zero per node — action modulates each node's dynamics
  2. Multi-step InfoNCE — contrastive prediction, prevents copy collapse
  3. Inverse dynamics per edge — which edge changed given the transition?

Selective attention planner:
  At plan time, compute attention over nodes relative to goal node.
  Focus MPC rollouts on task-relevant subgraph (agent + block, not goal).

Sprint D pass criteria:
  - Per-node prediction InfoNCE < 1.5 by ep5
  - Edge attention selects agent+block subgraph (not goal) ≥ 80% of steps
  - Synthetic SR > 60% with H=8

Usage:
    python train_graph_wm.py --domain pusht --epochs 50
    python train_graph_wm.py --domain pusht --eval --ckpt checkpoints/graph_wm/pusht_best.pt
"""

import argparse
import math
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ═══════════════════════════════════════════════════════════════════════════
# DreamerV3 stability
# ═══════════════════════════════════════════════════════════════════════════

def symlog(x):
    return torch.sign(x) * torch.log1p(x.abs())

def agc_clip(params, clip=0.01, eps=1e-3):
    for p in params:
        if p.grad is None: continue
        pn = p.detach().norm(2).clamp(min=eps)
        gn = p.grad.detach().norm(2)
        if gn > clip * pn:
            p.grad.mul_(clip * pn / gn)

def info_nce(z_pred, z_target, tau=0.1):
    """InfoNCE over batch — positives on diagonal."""
    z_pred_n   = F.normalize(z_pred,   dim=-1)
    z_target_n = F.normalize(z_target, dim=-1)
    logits = torch.mm(z_pred_n, z_target_n.T) / tau
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)


# ═══════════════════════════════════════════════════════════════════════════
# PushT graph definition
# ═══════════════════════════════════════════════════════════════════════════

# Node feature dimensions for PushT
NODE_DIMS = {
    "pusht": [2, 4, 2],     # agent(x,y), block(x,y,sinθ,cosθ), goal(x,y)
}
N_NODES   = 3
ACTION_DIM = 2
D_NODE    = 64    # node embedding dimension


def obs_to_graph(obs: np.ndarray, goal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert PushT obs [0,1]^5 to graph node features.

    obs: (agent_x, agent_y, block_x, block_y, block_angle_norm)
    goal: (goal_x, goal_y)

    Returns:
        nodes: (N_NODES, max_node_dim) zero-padded
        mask:  (N_NODES, node_dim) valid feature mask
    """
    angle = obs[4] * 2 * math.pi    # denormalise angle
    nodes = np.zeros((N_NODES, 4), dtype=np.float32)
    # Node 0: agent
    nodes[0, :2] = obs[:2]
    # Node 1: block
    nodes[1, :2] = obs[2:4]
    nodes[1, 2]  = math.sin(angle)
    nodes[1, 3]  = math.cos(angle)
    # Node 2: goal
    nodes[2, :2] = goal
    return nodes


def obs_seq_to_graph_seq(
    obs_seq: np.ndarray,
    goal:    np.ndarray,
) -> np.ndarray:
    """(k+1, 5) → (k+1, N_NODES, 4)"""
    return np.stack([obs_to_graph(obs_seq[t], goal)
                     for t in range(len(obs_seq))])


# ═══════════════════════════════════════════════════════════════════════════
# AdaLN-Zero for node-level action conditioning
# ═══════════════════════════════════════════════════════════════════════════

class NodeAdaLN(nn.Module):
    """
    Per-node AdaLN-Zero conditioning.
    Action modulates each node's embedding independently via a shared MLP.
    Gate starts at zero — conditioning grows during training.
    """
    def __init__(self, action_dim: int, d_node: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(action_dim, d_node),
            nn.SiLU(),
            nn.Linear(d_node, 3 * d_node),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.d = d_node

    def forward(self, h: torch.Tensor, a_emb: torch.Tensor) -> torch.Tensor:
        """
        h:     (B, N, d_node)
        a_emb: (B, d_node) — projected action
        Returns: modulated (B, N, d_node)
        """
        gamma, beta, alpha = self.mlp(a_emb).chunk(3, dim=-1)  # (B, d)
        gamma = gamma.unsqueeze(1)   # (B, 1, d) — broadcasts over N
        beta  = beta.unsqueeze(1)
        alpha = alpha.unsqueeze(1)
        h_norm = F.layer_norm(h, [self.d])
        return h + 0.1 * alpha * ((1 + gamma) * h_norm + beta)


# ═══════════════════════════════════════════════════════════════════════════
# Node encoder
# ═══════════════════════════════════════════════════════════════════════════

class NodeEncoder(nn.Module):
    """Encodes raw node features → d_node embedding per node."""
    def __init__(self, raw_dim: int = 4, d_node: int = D_NODE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(raw_dim, d_node),
            nn.LayerNorm(d_node),
            nn.GELU(),
            nn.Linear(d_node, d_node),
            nn.LayerNorm(d_node),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, raw_dim) → (B, N, d_node)"""
        B, N, D = x.shape
        return self.net(x.view(B * N, D)).view(B, N, -1)


# ═══════════════════════════════════════════════════════════════════════════
# Graph Neural Network dynamics predictor
# ═══════════════════════════════════════════════════════════════════════════

class EdgeNetwork(nn.Module):
    """
    Computes edge messages: concat (node_i, node_j) → message.
    Used for both message passing and edge-level inverse dynamics.
    """
    def __init__(self, d_node: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d_node, d_node),
            nn.GELU(),
            nn.Linear(d_node, d_node),
        )
        # Learned edge attention weights (N×N)
        # Asymmetric init: bias agent(0)↔block(1) contact edge
        _init = torch.tensor([
            [0.0, 1.5, 0.5],
            [1.5, 0.0, 0.8],
            [0.3, 0.8, 0.0],
        ])
        self.edge_attn = nn.Parameter(_init)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        """
        nodes: (B, N, d_node)
        Returns: aggregated messages (B, N, d_node)
        """
        B, N, D = nodes.shape
        # Pairwise messages
        ni = nodes.unsqueeze(2).expand(-1, -1, N, -1)   # (B, N, N, D)
        nj = nodes.unsqueeze(1).expand(-1, N, -1, -1)   # (B, N, N, D)
        msg = self.net(torch.cat([ni, nj], dim=-1))      # (B, N, N, D)
        # Attention-weighted aggregation
        attn = torch.softmax(self.edge_attn, dim=-1)     # (N, N)
        agg  = (attn.unsqueeze(0).unsqueeze(-1) * msg).sum(dim=2)  # (B, N, D)
        return agg


class GraphDynamicsPredictor(nn.Module):
    """
    GNN-based dynamics predictor.

    Given current node embeddings and action, predicts next node embeddings.
    Architecture: NodeEncoder → EdgeNetwork (message passing) → AdaLN-Zero → output nodes

    Two message-passing rounds before action injection give the GNN
    time to compute relational structure before being modulated.
    """
    def __init__(self, d_node: int = D_NODE, action_dim: int = ACTION_DIM,
                 n_mp_rounds: int = 2):
        super().__init__()
        self.node_enc   = NodeEncoder(raw_dim=4, d_node=d_node)
        self.edge_nets  = nn.ModuleList([EdgeNetwork(d_node)
                                          for _ in range(n_mp_rounds)])
        self.action_emb = nn.Sequential(
            nn.Linear(action_dim, d_node), nn.SiLU(),
            nn.Linear(d_node, d_node), nn.LayerNorm(d_node),
        )
        self.adaln = NodeAdaLN(d_node, d_node)
        self.node_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_node, d_node), nn.GELU(), nn.Linear(d_node, d_node)
            )
            for _ in range(n_mp_rounds)
        ])
        self.out_norm = nn.LayerNorm(d_node)
        self.d = d_node

    def encode(self, raw_nodes: torch.Tensor) -> torch.Tensor:
        """raw_nodes: (B, N, 4) → (B, N, d_node)"""
        return self.node_enc(raw_nodes)

    def forward(
        self,
        nodes: torch.Tensor,    # (B, N, d_node) — already encoded
        action: torch.Tensor,   # (B, action_dim)
    ) -> torch.Tensor:
        """Predict next node embeddings (B, N, d_node)."""
        a_emb = self.action_emb(action)     # (B, d_node)
        h = nodes

        # Message passing rounds
        for edge_net, mlp in zip(self.edge_nets, self.node_mlps):
            msg = edge_net(h)
            h   = h + mlp(msg)              # residual update

        # Action conditioning via AdaLN-Zero
        h = self.adaln(h, a_emb)
        return self.out_norm(h)

    def rollout(
        self,
        nodes0:  torch.Tensor,     # (B, N, d_node)
        actions: torch.Tensor,     # (B, K, action_dim)
    ) -> List[torch.Tensor]:
        """K-step rollout. Returns list of K node tensors."""
        preds = []
        h = nodes0
        for k in range(actions.shape[1]):
            h = self.forward(h, actions[:, k])
            preds.append(h)
        return preds

    def get_edge_attention(self) -> torch.Tensor:
        """Returns (N, N) edge attention weights — for diagnostics."""
        return torch.softmax(self.edge_nets[0].edge_attn, dim=-1).detach()


# ═══════════════════════════════════════════════════════════════════════════
# Inverse dynamics head (edge-level)
# ═══════════════════════════════════════════════════════════════════════════

class GraphIDM(nn.Module):
    """
    Predicts action from graph transition (nodes_t, nodes_t1).
    Operates on the agent-block edge specifically (most action-relevant).
    """
    def __init__(self, d_node: int = D_NODE, action_dim: int = ACTION_DIM):
        super().__init__()
        # Focus on agent node (0) and block node (1) delta
        self.net = nn.Sequential(
            nn.Linear(4 * d_node, d_node), nn.GELU(),
            nn.Linear(d_node, d_node // 2), nn.GELU(),
            nn.Linear(d_node // 2, action_dim),
        )

    def forward(
        self,
        nodes_t:  torch.Tensor,   # (B, N, d_node)
        nodes_t1: torch.Tensor,   # (B, N, d_node)
    ) -> torch.Tensor:
        """Returns predicted action (B, action_dim)."""
        # Agent and block nodes before and after
        agent_t  = nodes_t[:, 0]    # (B, d)
        block_t  = nodes_t[:, 1]
        agent_t1 = nodes_t1[:, 0]
        block_t1 = nodes_t1[:, 1]
        return self.net(torch.cat([agent_t, block_t, agent_t1, block_t1], dim=-1))


# ═══════════════════════════════════════════════════════════════════════════
# Selective attention planner
# ═══════════════════════════════════════════════════════════════════════════

class SelectiveAttentionPlanner(nn.Module):
    """
    Computes per-node attention relative to goal, selects task-relevant subgraph.

    For PushT:
      - Goal node (2) has highest relevance initially
      - Block node (1) becomes relevant once agent is close
      - Agent node (0) always relevant

    Attention is used to:
      1. Weight the planning cost (closer to goal = more weight)
      2. Select which nodes to use for uncertainty estimation
    """
    def __init__(self, d_node: int = D_NODE):
        super().__init__()
        self.query_proj = nn.Linear(d_node, d_node)
        self.key_proj   = nn.Linear(d_node, d_node)

    def forward(
        self,
        nodes: torch.Tensor,     # (B, N, d_node)
        goal_node_idx: int = 2,  # which node is the goal
    ) -> torch.Tensor:
        """Returns attention weights (B, N) over nodes relative to goal."""
        goal = nodes[:, goal_node_idx].unsqueeze(1)   # (B, 1, d)
        q = self.query_proj(goal)                      # (B, 1, d)
        k = self.key_proj(nodes)                       # (B, N, d)
        scores = (q * k).sum(-1) / math.sqrt(nodes.shape[-1])  # (B, N)
        return torch.softmax(scores, dim=-1)


# ═══════════════════════════════════════════════════════════════════════════
# PushT graph dataset
# ═══════════════════════════════════════════════════════════════════════════

class PushTGraphDataset(Dataset):
    """
    PushT dataset returning graph sequences.
    Same synthetic physics as train_action_wm.py but converted to graph format.
    """

    def __init__(
        self,
        data_path:  Optional[str] = None,
        k_steps:    int = 4,
        n_episodes: int = 206,
    ):
        self.k = k_steps
        self.trajs: List[dict] = []

        loaded = False
        if data_path:
            p = Path(data_path)
            if p.suffix == ".zarr" or p.is_dir():
                loaded = self._load_zarr(p)
            elif p.suffix in (".hdf5", ".h5"):
                loaded = self._load_hdf5(p)

        if not loaded:
            self._generate_synthetic(n_episodes)

        self.index = []
        for ep_idx, traj in enumerate(self.trajs):
            T = traj["obs"].shape[0]
            for t in range(T - k_steps):
                self.index.append((ep_idx, t))

        print(f"PushTGraphDataset: {len(self.trajs)} episodes, "
              f"{len(self.index):,} samples (k={k_steps})")

    def _generate_synthetic(self, n_episodes: int):
        rng = np.random.RandomState(42)
        for ep in range(n_episodes):
            T     = rng.randint(80, 180)
            obs   = np.zeros((T, 5), dtype=np.float32)
            acts  = np.zeros((T, 2), dtype=np.float32)
            goal  = rng.uniform(0.55, 0.90, 2)

            agent = rng.uniform(0.1, 0.4, 2)
            block = rng.uniform(0.3, 0.6, 2)
            angle = rng.uniform(0, 2 * math.pi)

            for t in range(T):
                obs[t] = [agent[0], agent[1], block[0], block[1],
                          angle / (2 * math.pi)]
                if np.linalg.norm(agent - block) > 0.12:
                    target = block + rng.normal(0, 0.03, 2)
                else:
                    push_dir = goal - block
                    target   = agent + push_dir * 0.3 + rng.normal(0, 0.02, 2)
                target = np.clip(target, 0, 1)
                acts[t] = target * 2 - 1

                agent += (target - agent) * 0.4 + rng.normal(0, 0.01, 2)
                agent  = np.clip(agent, 0, 1)
                if np.linalg.norm(agent - block) < 0.1:
                    push   = (agent - block) * 0.2
                    block  = np.clip(block - push, 0, 1)
                    angle += rng.normal(0, 0.05)

            # Convert to graph sequences
            graphs = obs_seq_to_graph_seq(obs, goal)
            self.trajs.append({"obs": obs, "action": acts,
                                "graphs": graphs, "goal": goal})

    def _load_zarr(self, path: Path) -> bool:
        try:
            import zarr
            z = zarr.open(str(path), "r")
            obs    = np.array(z["data/state"])
            action = np.array(z["data/action"])
            ep_ends = np.array(z["meta/episode_ends"])
            starts  = np.concatenate([[0], ep_ends[:-1]])
            for s, e in zip(starts, ep_ends):
                obs_ep = obs[s:e].astype(np.float32)
                # Normalise pixel coords
                obs_ep[:, :4] /= 512.0
                obs_ep[:, 4]  /= (2 * math.pi)
                act_ep = (action[s:e].astype(np.float32) / 512.0) * 2 - 1
                goal   = obs_ep[-1, 2:4]   # last block pos as proxy goal
                graphs = obs_seq_to_graph_seq(obs_ep, goal)
                self.trajs.append({"obs": obs_ep, "action": act_ep,
                                    "graphs": graphs, "goal": goal})
            return True
        except Exception as e:
            print(f"  Zarr load failed: {e}")
            return False

    def _load_hdf5(self, path: Path) -> bool:
        try:
            import h5py
            with h5py.File(path, "r") as f:
                obs_all = np.array(f["observations"], dtype=np.float32)
                act_all = np.array(f["actions"], dtype=np.float32)
                # Normalise
                obs_all[:, :4] /= 512.0
                obs_all[:, 4]  /= (2 * math.pi)
                act_all        /= 512.0
                act_all         = act_all * 2 - 1   # → [-1, 1]
                goal = np.array([0.65, 0.65])
                graphs = obs_seq_to_graph_seq(obs_all, goal)
                self.trajs.append({"obs": obs_all, "action": act_all,
                                    "graphs": graphs, "goal": goal})
            return True
        except Exception as e:
            print(f"  HDF5 load failed: {e}")
            return False

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ep_idx, t = self.index[idx]
        traj   = self.trajs[ep_idx]
        graphs = traj["graphs"][t : t + self.k + 1]     # (k+1, N, 4)
        action = traj["action"][t : t + self.k]          # (k, 2)
        goal   = traj["goal"]
        return {
            "graphs": torch.from_numpy(graphs.astype(np.float32)),
            "action": torch.from_numpy(action.astype(np.float32)),
            "goal":   torch.from_numpy(goal.astype(np.float32)),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════

def train_graph_wm(
    domain:     str   = "pusht",
    data_path:  Optional[str] = None,
    n_epochs:   int   = 50,
    batch_size: int   = 64,
    base_lr:    float = 3e-4,
    k_steps:    int   = 4,
    lambda_idm: float = 50.0,
    save_dir:   str   = "checkpoints/graph_wm",
    log_every:  int   = 50,
    device_str: str   = "cpu",
):
    device = torch.device(device_str)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # ── Dataset ────────────────────────────────────────────────────────────
    ds = PushTGraphDataset(data_path=data_path, k_steps=k_steps)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=0, drop_last=True)

    # ── Models ─────────────────────────────────────────────────────────────
    predictor = GraphDynamicsPredictor(d_node=D_NODE, action_dim=ACTION_DIM).to(device)
    idm_head  = GraphIDM(d_node=D_NODE, action_dim=ACTION_DIM).to(device)
    attn_plan = SelectiveAttentionPlanner(d_node=D_NODE).to(device)

    all_params = (list(predictor.parameters())
                + list(idm_head.parameters())
                + list(attn_plan.parameters()))
    n_params = sum(p.numel() for p in all_params)
    print(f"\n[GRAPH-WM / {domain.upper()}]  params={n_params:,}")

    optimizer = torch.optim.AdamW(all_params, lr=base_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=base_lr*5,
        total_steps=n_epochs * len(loader), pct_start=0.05,
    )

    best_loss   = float("inf")
    global_step = 0

    for epoch in range(n_epochs):
        predictor.train(); idm_head.train(); attn_plan.train()
        epoch_losses, ac_lifts, idm_errs = [], [], []

        for batch in loader:
            graphs = batch["graphs"].to(device)   # (B, k+1, N, 4)
            action = batch["action"].to(device)   # (B, k, 2)
            B = graphs.shape[0]

            # ── Encode all graph snapshots ──────────────────────────────
            # (B, k+1, N, 4) → (B, k+1, N, d_node)
            g_flat = graphs.view(B * (k_steps+1), N_NODES, 4)
            z_flat = predictor.encode(g_flat).view(B, k_steps+1, N_NODES, D_NODE)
            z0 = z_flat[:, 0]    # (B, N, d_node)

            # Contact detection from raw graph (agent=node0, block=node1)
            agent_pos = graphs[:, 0, 0, :2]
            block_pos = graphs[:, 0, 1, :2]
            contact   = ((agent_pos - block_pos).norm(dim=-1) < 0.15).float()

            # ── Multi-step prediction (InfoNCE per node) ────────────────
            z_preds = predictor.rollout(z0, action)   # k tensors of (B, N, d)
            L_pred = torch.tensor(0.0, device=device)
            for k in range(k_steps):
                z_p = z_preds[k]                      # (B, N, d)
                z_t = z_flat[:, k+1].detach()         # (B, N, d)
                # InfoNCE per node — flatten N into batch for contrast
                z_p_flat = z_p.reshape(B * N_NODES, D_NODE)
                z_t_flat = z_t.reshape(B * N_NODES, D_NODE)
                L_pred = L_pred + info_nce(z_p_flat, z_t_flat)
            L_pred = L_pred / k_steps

            # ── Action-conditioned vs unconditional (ac_lift) ───────────
            with torch.no_grad():
                a_zero = torch.zeros_like(action[:, 0])
                z_uncond = predictor(z0, a_zero)
                z_tgt    = z_flat[:, 1].detach()
                L_uncond = info_nce(
                    z_uncond.reshape(B*N_NODES, D_NODE),
                    z_tgt.reshape(B*N_NODES, D_NODE)
                ).item()
            ac_lifts.append(L_uncond - L_pred.item())

            # ── Inverse dynamics (edge-level) ───────────────────────────
            L_idm = sum(
                F.smooth_l1_loss(
                    idm_head(z_flat[:, k], z_flat[:, k+1].detach()),
                    action[:, k]
                )
                for k in range(k_steps)
            ) / k_steps
            idm_errs.append(L_idm.item())

            # Edge supervision: pull contact edges high during contact
            ea_raw = predictor.edge_nets[0].edge_attn
            ea_ab  = torch.softmax(ea_raw, dim=-1)[0, 1]
            ea_ba  = torch.softmax(ea_raw, dim=-1)[1, 0]
            L_edge = (contact * (1 - ea_ab).pow(2)).mean() +                      (contact * (1 - ea_ba).pow(2)).mean()

            total = L_pred + lambda_idm * L_idm + 5.0 * L_edge
            if not torch.isfinite(total):
                optimizer.zero_grad(); continue

            optimizer.zero_grad()
            total.backward()
            agc_clip(all_params)
            optimizer.step()
            scheduler.step()

            epoch_losses.append(total.item())
            global_step += 1

            if global_step % log_every == 0:
                ea = predictor.get_edge_attention()
                gate = predictor.adaln.mlp[-1].weight.norm().item()
                print(
                    f"[ep{epoch:02d} s{global_step:05d}] "
                    f"L={total.item():.4f} "
                    f"L_pred={L_pred.item():.4f} "
                    f"L_idm={L_idm.item():.4f} "
                    f"L_edge={L_edge.item():.4f} "
                    f"ac_lift={np.mean(ac_lifts[-20:]):+.4f} "
                    f"gate={gate:.3f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )

        # ── Epoch summary ───────────────────────────────────────────────
        mean_L   = np.mean(epoch_losses) if epoch_losses else 0
        mean_ac  = np.mean(ac_lifts)     if ac_lifts     else 0
        mean_idm = np.mean(idm_errs)     if idm_errs     else 0

        # Edge attention diagnostic
        ea = predictor.get_edge_attention()
        agent_attn = ea[0].mean().item()
        block_attn = ea[1].mean().item()
        goal_attn  = ea[2].mean().item()

        print(f"\nEpoch {epoch:02d}  loss={mean_L:.4f}  "
              f"ac_lift={mean_ac:+.4f}  idm={mean_idm:.4f}")
        print(f"  Edge attention: agent={agent_attn:.3f}  "
              f"block={block_attn:.3f}  goal={goal_attn:.3f}")

        if mean_ac > 0.05:
            print("  ✓ Action conditioning load-bearing")
        if block_attn > goal_attn:
            print("  ✓ Block edge dominates goal edge (contact-aware routing)")

        # ── Save ────────────────────────────────────────────────────────
        if mean_L < best_loss:
            best_loss = mean_L
            path = Path(save_dir) / f"graph_wm_{domain}_best.pt"
            torch.save({
                "epoch":     epoch,
                "loss":      best_loss,
                "domain":    domain,
                "k_steps":   k_steps,
                "ac_lift":   float(mean_ac),
                "idm_mae":   float(mean_idm),
                "predictor": predictor.state_dict(),
                "idm_head":  idm_head.state_dict(),
                "attn_plan": attn_plan.state_dict(),
                "edge_attn": ea.cpu().numpy(),
            }, path)
            print(f"  → Saved: {path}")

    return predictor, idm_head, attn_plan


# ═══════════════════════════════════════════════════════════════════════════
# Graph MPC planner eval (synthetic)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def plan_graph_mpc(
    predictor:   GraphDynamicsPredictor,
    attn_plan:   SelectiveAttentionPlanner,
    obs:         np.ndarray,    # (5,) normalised PushT obs
    goal:        np.ndarray,    # (2,) normalised goal pos
    n_cands:     int = 256,
    horizon:     int = 8,
    n_elite:     int = 32,
    n_iters:     int = 3,
    device:      torch.device = torch.device("cpu"),
) -> np.ndarray:
    """CEM graph-based planning. Returns action in [0,1]^2."""
    raw_nodes = obs_to_graph(obs, goal)
    nodes = predictor.encode(
        torch.from_numpy(raw_nodes).unsqueeze(0).to(device)
    )  # (1, N, d_node)

    # Selective attention: weight cost by node relevance
    attn_w = attn_plan(nodes, goal_node_idx=2)   # (1, N)

    # Goal in latent: encode goal-at-goal state
    goal_raw = obs_to_graph(
        np.array([goal[0], goal[1], goal[0], goal[1], 0.0]), goal
    )
    z_goal = predictor.encode(
        torch.from_numpy(goal_raw).unsqueeze(0).to(device)
    )  # (1, N, d_node)

    # CEM
    mu  = torch.full((horizon, ACTION_DIM), 0.0, device=device)
    std = torch.full((horizon, ACTION_DIM), 0.4, device=device)

    for _ in range(n_iters):
        # Sample candidates: (K, H, 2)
        actions = torch.clamp(
            mu + std * torch.randn(n_cands, horizon, ACTION_DIM, device=device),
            -1.0, 1.0
        )
        # Rollout
        z_batch = nodes.expand(n_cands, -1, -1)
        preds   = predictor.rollout(z_batch, actions)
        z_final = preds[-1]    # (K, N, d_node)

        # Score: attention-weighted node distance to goal
        z_goal_exp = z_goal.expand(n_cands, -1, -1)         # (K, N, d)
        node_dists = (z_final - z_goal_exp).norm(dim=-1)    # (K, N)
        w_exp      = attn_w.expand(n_cands, -1)             # (K, N)
        scores     = -(w_exp * node_dists).sum(-1)           # (K,)

        # Elite update
        elite_idx = scores.topk(n_elite).indices
        elite_ac  = actions[elite_idx]
        mu  = elite_ac.mean(0)
        std = elite_ac.std(0).clamp(min=0.05)

    # Best action (convert from [-1,1] to [0,1])
    best  = actions[scores.argmax(), 0].cpu().numpy()
    return (best + 1.0) / 2.0    # → [0, 1]


def eval_graph_synthetic(
    predictor:  GraphDynamicsPredictor,
    attn_plan:  SelectiveAttentionPlanner,
    n_episodes: int = 50,
    max_steps:  int = 200,
    success_thr: float = 0.08,
    device:     torch.device = torch.device("cpu"),
):
    """Eval on synthetic PushT physics. Returns SR."""
    import math
    rng = np.random.RandomState(123)
    successes = []

    for ep in range(n_episodes):
        goal  = rng.uniform(0.55, 0.90, 2)
        agent = rng.uniform(0.1, 0.4, 2)
        block = rng.uniform(0.3, 0.6, 2)
        angle = rng.uniform(0, 2 * math.pi)
        success = False

        for step in range(max_steps):
            obs = np.array([agent[0], agent[1], block[0], block[1],
                            angle / (2 * math.pi)], dtype=np.float32)
            action = plan_graph_mpc(predictor, attn_plan, obs, goal,
                                    device=device)
            action_raw = action * 2 - 1   # → [-1, 1] for synthetic physics

            agent += (action - agent) * 0.4 + rng.normal(0, 0.01, 2)
            agent  = np.clip(agent, 0, 1)
            if np.linalg.norm(agent - block) < 0.1:
                push   = (agent - block) * 0.2
                block  = np.clip(block - push, 0, 1)
                angle += rng.normal(0, 0.05)

            if np.linalg.norm(block - goal) < success_thr:
                success = True
                break

        successes.append(success)
        print(f"  ep{ep+1:02d} {'✓' if success else '✗'}  "
              f"dist={np.linalg.norm(block-goal):.3f}")

    sr = np.mean(successes)
    print(f"\n══ Graph WM SR: {sr:.1%} ({sum(successes)}/{n_episodes}) ══")
    return sr


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--domain",     default="pusht")
    p.add_argument("--data-path",  default=None)
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch-size", type=int,   default=64)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--k-steps",    type=int,   default=4)
    p.add_argument("--lambda-idm", type=float, default=50.0)
    p.add_argument("--save-dir",   default="checkpoints/graph_wm")
    p.add_argument("--log-every",  type=int,   default=50)
    p.add_argument("--device",     default="cpu")
    p.add_argument("--eval",       action="store_true",
                   help="Run synthetic SR eval after training")
    p.add_argument("--ckpt",       default=None,
                   help="Load checkpoint for eval only")
    p.add_argument("--n-episodes", type=int, default=50)
    args = p.parse_args()

    device = torch.device(args.device)

    if args.ckpt and Path(args.ckpt).exists():
        # Eval only
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        predictor = GraphDynamicsPredictor().to(device)
        attn_plan = SelectiveAttentionPlanner().to(device)
        predictor.load_state_dict(ckpt["predictor"])
        attn_plan.load_state_dict(ckpt["attn_plan"])
        predictor.eval(); attn_plan.eval()
        print(f"Loaded: {args.ckpt}  (ac_lift={ckpt.get('ac_lift',0):+.4f})")
        eval_graph_synthetic(predictor, attn_plan,
                             n_episodes=args.n_episodes, device=device)
    else:
        predictor, idm_head, attn_plan = train_graph_wm(
            domain     = args.domain,
            data_path  = args.data_path,
            n_epochs   = args.epochs,
            batch_size = args.batch_size,
            base_lr    = args.lr,
            k_steps    = args.k_steps,
            lambda_idm = args.lambda_idm,
            save_dir   = args.save_dir,
            log_every  = args.log_every,
            device_str = args.device,
        )
        if args.eval:
            predictor.eval(); attn_plan.eval()
            eval_graph_synthetic(predictor, attn_plan,
                                 n_episodes=args.n_episodes, device=device)
