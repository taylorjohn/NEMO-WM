"""
sprint7b_language_nav.py — Language-Weighted Waypoint Planning (R2F + CLIPBridge)
==================================================================================
Extends Sprint 7a's WaypointGraph with language-weighted A* search.

R2F framing: each GeoLatentDB node is treated as a "language frontier" — a
candidate location scored by its relevance to a text query. The A* cost
function is modified so high-relevance nodes are attracted toward the search
path. Language-conditioned navigation requires no new training.

Architecture:
    text_query
        -> CLIP ViT-B/32 -> CLIPBridge (Sprint 6c checkpoint)
        -> language_emb (128-D)
        -> score all 65,476 GeoLatentDB nodes: s_i = cos(node_i, language_emb)
        -> language scores pre-computed once per query — O(N) dot products

    Language-weighted A*:
        g_cost(edge) = distance_m * (1 - λ * relevance_score[neighbor])
        h_cost(node) = haversine(node, goal) * (1 - λ * relevance_score[node])
        λ = language_weight [0, 1] — controls GPS vs language tradeoff

    Goal selection:
        1. Text query → top-K relevant nodes by language score
        2. Filter to nodes reachable from start (connected subgraph)
        3. Plan weighted A* path toward highest-scoring reachable node

    Execution:
        WaypointExecutor from Sprint 7a — unchanged
        4Hz loop, cosine switch, replan timeout

Performance:
    Language scoring: O(N×D) = 65,476 × 128 dot products ≈ 0.3ms
    Weighted A* plan: < 5ms (same graph, modified cost)
    Total latency: < 6ms (well within 250ms 4Hz budget)

Run:
    # Self-test (no hardware needed)
    python sprint7b_language_nav.py --self-test

    # Benchmark
    python sprint7b_language_nav.py --benchmark

    # Live navigation with text goal
    python sprint7b_language_nav.py \
        --text "navigate to open outdoor area near the road" \
        --start-lat 37.9105 --start-lon -122.3395 \
        --clip-ckpt checkpoints/dinov2_student/student_dualhead_nce_best.pt \
        --geo-db   checkpoints/cwm/geo_latent_db.pt

Author: John Taylor — github.com/taylorjohn
Date:   2026-04-06
Sprint: 7b
"""

import argparse
import heapq
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import KDTree

# ── Sprint 7a core (inlined) ──────────────────────────────────────
import argparse
import heapq
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import KDTree

DEVICE = torch.device("cpu")

# ── Constants ─────────────────────────────────────────────────────────────────

SWITCH_THRESH    = 0.92    # cosine similarity to declare waypoint reached
REPLAN_TIMEOUT   = 60      # steps before forcing replan (15s at 4Hz)
ARRIVAL_THRESH   = 0.97    # cosine similarity to declare goal reached
MAX_GRAPH_RADIUS = 0.25    # max metres between connected graph nodes
MIN_GRAPH_RADIUS = 0.05    # min metres — skip duplicate nodes
N_WAYPOINTS_MAX  = 12      # max waypoints per plan
EARTH_RADIUS_M   = 6_371_000.0


# ── Coordinate helpers ────────────────────────────────────────────────────────

def gps_to_metres(lat1: float, lon1: float,
                  lat2: float, lon2: float) -> tuple[float, float]:
    """Convert GPS delta to approximate Cartesian metres (flat Earth)."""
    dlat = (lat2 - lat1) * np.pi / 180.0
    dlon = (lon2 - lon1) * np.pi / 180.0
    lat_m = np.radians((lat1 + lat2) / 2)
    dx = dlon * EARTH_RADIUS_M * np.cos(lat_m)
    dy = dlat * EARTH_RADIUS_M
    return dx, dy


def gps_distance_m(lat1: float, lon1: float,
                   lat2: float, lon2: float) -> float:
    """Haversine distance in metres."""
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat/2)**2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2))
         * np.sin(dlon/2)**2)
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


# ── Waypoint graph ────────────────────────────────────────────────────────────

@dataclass
class WaypointNode:
    idx:       int
    lat:       float
    lon:       float
    embedding: np.ndarray   # (128,) mean particle embedding


@dataclass(order=True)
class _AStarItem:
    f_score:  float
    idx:      int = field(compare=False)
    g_score:  float = field(compare=False)
    parent:   Optional[int] = field(compare=False, default=None)


class WaypointGraph:
    """
    Builds a navigable graph from GeoLatentDB entries.
    Nodes are GPS-indexed particle embeddings. Edges connect
    nodes within MAX_GRAPH_RADIUS metres of each other.

    A* search uses latent cosine distance as heuristic:
    physically close nodes tend to have similar embeddings.
    """

    def __init__(
        self,
        max_radius_m: float = MAX_GRAPH_RADIUS,
        min_radius_m: float = MIN_GRAPH_RADIUS,
    ):
        self.max_radius = max_radius_m
        self.min_radius = min_radius_m
        self.nodes:  list[WaypointNode] = []
        self._kdtree: Optional[KDTree] = None
        self._coords: Optional[np.ndarray] = None   # (N, 2) lat/lon

    def build_from_geodict(self, geodict: dict) -> int:
        """
        Build graph from GeoLatentDB save dict.
        Expected keys: 'gps' (N, 2) and 'embeddings' (N, 128).
        Returns number of nodes added.
        """
        gps  = np.array(geodict['gps'])        # (N, 2) lat, lon
        embs = np.array(geodict['embeddings']) # (N, 128)

        # Deduplicate by minimum spatial distance
        kept = []
        tree = KDTree(gps)
        mask = np.ones(len(gps), dtype=bool)

        for i in range(len(gps)):
            if not mask[i]:
                continue
            nn = tree.query_ball_point(gps[i], r=self.min_radius / 111_000)
            for j in nn:
                if j != i:
                    mask[j] = False

        for i in np.where(mask)[0]:
            self.nodes.append(WaypointNode(
                idx       = len(self.nodes),
                lat       = float(gps[i, 0]),
                lon       = float(gps[i, 1]),
                embedding = embs[i].astype(np.float32),
            ))

        self._coords  = np.array([[n.lat, n.lon] for n in self.nodes])
        self._kdtree  = KDTree(self._coords)
        return len(self.nodes)

    def nearest_node(self, lat: float, lon: float) -> WaypointNode:
        """Return closest node to a GPS coordinate."""
        _, idx = self._kdtree.query([lat, lon])
        return self.nodes[idx]

    def nearest_node_by_embedding(self, embedding: np.ndarray) -> WaypointNode:
        """Return node whose embedding is most similar to query."""
        embs = np.stack([n.embedding for n in self.nodes])
        q    = embedding / (np.linalg.norm(embedding) + 1e-8)
        e    = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
        sims = e @ q
        return self.nodes[int(np.argmax(sims))]

    def _neighbours(self, node: WaypointNode) -> list[tuple[int, float]]:
        """Return (idx, distance_m) for all nodes within max_radius."""
        radius_deg = self.max_radius / 111_000
        idxs = self._kdtree.query_ball_point(
            [node.lat, node.lon], r=radius_deg
        )
        out = []
        for i in idxs:
            if i == node.idx:
                continue
            d = gps_distance_m(node.lat, node.lon,
                                self.nodes[i].lat, self.nodes[i].lon)
            if d >= self.min_radius:
                out.append((i, d))
        return out

    def plan(
        self,
        start_lat:   float,
        start_lon:   float,
        goal_lat:    float,
        goal_lon:    float,
        max_waypoints: int = N_WAYPOINTS_MAX,
    ) -> list[WaypointNode]:
        """
        A* search from start GPS to goal GPS.
        Returns ordered list of WaypointNodes to follow.
        Heuristic: Haversine distance to goal.
        """
        if self._kdtree is None:
            raise RuntimeError("Graph not built — call build_from_geodict() first")

        start_node = self.nearest_node(start_lat, start_lon)
        goal_node  = self.nearest_node(goal_lat, goal_lon)

        if start_node.idx == goal_node.idx:
            return [goal_node]

        open_heap: list[_AStarItem] = []
        came_from: dict[int, Optional[int]] = {}
        g_score:   dict[int, float] = {start_node.idx: 0.0}

        h0 = gps_distance_m(start_node.lat, start_node.lon,
                             goal_node.lat,  goal_node.lon)
        heapq.heappush(open_heap, _AStarItem(h0, start_node.idx, 0.0, None))
        came_from[start_node.idx] = None

        while open_heap:
            item = heapq.heappop(open_heap)
            curr_idx, g = item.idx, item.g_score

            if curr_idx == goal_node.idx:
                break

            if g > g_score.get(curr_idx, float('inf')):
                continue  # stale entry

            for nb_idx, dist in self._neighbours(self.nodes[curr_idx]):
                new_g = g + dist
                if new_g < g_score.get(nb_idx, float('inf')):
                    g_score[nb_idx] = new_g
                    came_from[nb_idx] = curr_idx
                    h = gps_distance_m(self.nodes[nb_idx].lat, self.nodes[nb_idx].lon,
                                       goal_node.lat, goal_node.lon)
                    heapq.heappush(open_heap, _AStarItem(new_g + h, nb_idx, new_g))

        # Reconstruct path
        path = []
        cur = goal_node.idx
        while cur is not None:
            path.append(self.nodes[cur])
            cur = came_from.get(cur)
        path.reverse()

        # Subsample to max_waypoints (keep start and goal)
        if len(path) > max_waypoints:
            step = len(path) / (max_waypoints - 1)
            idxs = [int(i * step) for i in range(max_waypoints - 1)] + [len(path) - 1]
            path = [path[i] for i in sorted(set(idxs))]

        return path


# ── Waypoint executor ─────────────────────────────────────────────────────────

class MirrorAscentPlanner:
    """
    Fast gradient-based 2-D planner.
    Computes action direction from current particle mean to goal embedding.
    """
    def __init__(self, action_dim: int = 2, lr: float = 0.12, steps: int = 3):
        self.action_dim = action_dim
        self.lr         = lr
        self.steps      = steps

    def plan(self, current: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """
        current: (K, 128) particle embeddings or (128,) mean
        goal:    (128,) target embedding
        Returns: (action_dim,) action in [-1, 1]
        """
        cur_mean = current.mean(0) if current.ndim == 2 else current
        delta    = goal - cur_mean
        norm     = np.linalg.norm(delta) + 1e-8
        # Project to 2-D action space (use first 2 dims after PCA-like direction)
        action   = (delta[:self.action_dim] / norm).clip(-1.0, 1.0)
        return (action * self.lr).clip(-1.0, 1.0)


@dataclass
class ExecutionState:
    waypoints:        list[WaypointNode]
    waypoint_idx:     int   = 0
    steps_on_wp:      int   = 0
    total_steps:      int   = 0
    waypoints_reached: int  = 0
    replans:          int   = 0
    arrived:          bool  = False


class WaypointExecutor:
    """
    4Hz execution loop that follows a waypoint plan.

    At each step:
      1. Compute cosine similarity between current particles and active waypoint.
      2. If sim > SWITCH_THRESH: advance to next waypoint.
      3. If steps_on_wp > REPLAN_TIMEOUT: request replan.
      4. If waypoint is final and sim > ARRIVAL_THRESH: mark arrived.
      5. Return MirrorAscent action toward active waypoint.
    """

    def __init__(
        self,
        planner:        MirrorAscentPlanner,
        switch_thresh:  float = SWITCH_THRESH,
        replan_timeout: int   = REPLAN_TIMEOUT,
        arrival_thresh: float = ARRIVAL_THRESH,
    ):
        self.planner        = planner
        self.switch_thresh  = switch_thresh
        self.replan_timeout = replan_timeout
        self.arrival_thresh = arrival_thresh
        self.state:         Optional[ExecutionState] = None

    def reset(self, waypoints: list[WaypointNode]):
        """Start executing a new waypoint plan."""
        self.state = ExecutionState(waypoints=waypoints)

    def step(
        self,
        current_particles: np.ndarray,   # (K, 128) or (128,)
        current_gps:       Optional[tuple[float, float]] = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Execute one 4Hz step.

        Returns:
            action: (2,) numpy array in [-1, 1]
            info:   dict with execution state for logging
        """
        if self.state is None or self.state.arrived:
            return np.zeros(2), {"status": "idle"}

        s = self.state
        s.total_steps += 1
        s.steps_on_wp += 1

        cur_mean = (current_particles.mean(0)
                    if current_particles.ndim == 2
                    else current_particles)
        cur_norm = cur_mean / (np.linalg.norm(cur_mean) + 1e-8)

        active_wp  = s.waypoints[s.waypoint_idx]
        wp_emb     = active_wp.embedding
        wp_norm    = wp_emb / (np.linalg.norm(wp_emb) + 1e-8)
        sim        = float(np.dot(cur_norm, wp_norm))
        is_last    = s.waypoint_idx == len(s.waypoints) - 1

        status = "navigating"

        # ── Arrival at final waypoint ─────────────────────────────────────
        if is_last and sim > self.arrival_thresh:
            s.arrived = True
            status = "ARRIVED"
            return np.zeros(2), {
                "status":     status,
                "sim":        sim,
                "waypoint":   s.waypoint_idx,
                "total_steps": s.total_steps,
                "replans":    s.replans,
            }

        # ── Waypoint switch ───────────────────────────────────────────────
        if not is_last and sim > self.switch_thresh:
            s.waypoint_idx    += 1
            s.waypoints_reached += 1
            s.steps_on_wp     = 0
            active_wp  = s.waypoints[s.waypoint_idx]
            wp_emb     = active_wp.embedding
            status     = f"SWITCHED -> wp {s.waypoint_idx}"

        # ── Replan trigger ────────────────────────────────────────────────
        needs_replan = s.steps_on_wp > self.replan_timeout
        if needs_replan:
            status = "REPLAN_NEEDED"
            s.replans    += 1
            s.steps_on_wp = 0

        # ── Plan action toward active waypoint ────────────────────────────
        action = self.planner.plan(current_particles, active_wp.embedding)

        info = {
            "status":          status,
            "sim":             sim,
            "waypoint":        s.waypoint_idx,
            "n_waypoints":     len(s.waypoints),
            "steps_on_wp":     s.steps_on_wp,
            "total_steps":     s.total_steps,
            "waypoints_reached": s.waypoints_reached,
            "replans":         s.replans,
            "needs_replan":    needs_replan,
            "active_gps":      (active_wp.lat, active_wp.lon),
        }
        return action, info


# ── Language goal resolver (Sprint 6d) ────────────────────────────────────────

class LanguageGoalResolver:
    """
    Resolves a text query to a GPS goal by:
      1. Encoding text via CLIP ViT-B/32 -> CLIPBridge -> SemanticHead space
      2. Finding nearest GeoLatentDB node by cosine similarity in semantic space

    Requires Sprint 6d checkpoint (student_dualhead_nce_nr_best.pt).
    Falls back to None if checkpoint not available.
    """

    def __init__(
        self,
        dualhead_ckpt:  str,
        backbone_ckpt:  str = "checkpoints/dinov2_student/student_best.pt",
    ):
        self.available = False
        try:
            import clip as clip_module
            self._clip = clip_module
            self._clip_model, _ = clip_module.load("ViT-B/32", device=DEVICE)
            self._clip_model.eval()

            # Load SemanticHead + CLIPBridge from Sprint 6d checkpoint
            from train_mvtec import StudentEncoder
            ckpt = torch.load(dualhead_ckpt, map_location="cpu", weights_only=False)

            # Reproduce SemanticHead architecture
            self.semantic_head = nn.Sequential(
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.LayerNorm(128),
            ).to(DEVICE)
            self.clip_bridge = nn.Linear(512, 128, bias=False).to(DEVICE)
            self.semantic_head.load_state_dict(ckpt["semantic_head"])
            self.clip_bridge.load_state_dict(ckpt["clip_bridge"])
            self.semantic_head.eval()
            self.clip_bridge.eval()
            self.available = True
            print(f"[LanguageGoalResolver] Loaded: ep{ckpt.get('epoch','?')}, "
                  f"L_clip={ckpt.get('loss','?'):.4f}")
        except Exception as e:
            print(f"[LanguageGoalResolver] Not available: {e}")

    def resolve(self, text: str, graph: WaypointGraph) -> Optional[WaypointNode]:
        """
        Resolve text query to nearest WaypointNode.
        Returns None if not available.
        """
        if not self.available:
            return None

        with torch.no_grad():
            tokens   = self._clip.tokenize([text]).to(DEVICE)
            txt_emb  = self._clip_model.encode_text(tokens).float()
            txt_emb  = F.normalize(txt_emb, dim=-1)
            sem_emb  = F.normalize(self.clip_bridge(txt_emb), dim=-1)

        sem_np = sem_emb.squeeze(0).numpy()
        node   = graph.nearest_node_by_embedding(sem_np)
        print(f"[LanguageGoalResolver] '{text}' -> "
              f"({node.lat:.5f}, {node.lon:.5f})")
        return node


# ── Full waypoint planner ─────────────────────────────────────────────────────

class WaypointPlanner:
    """
    Complete Sprint 7a system. Wraps WaypointGraph + WaypointExecutor +
    optional LanguageGoalResolver into a single interface.

    Usage:
        planner = WaypointPlanner(geo_db_path="checkpoints/cwm/geo_latent_db.pt")
        planner.set_goal_gps(37.9150, -122.3354)

        # 4Hz loop:
        action, info = planner.step(current_particles, current_gps=(lat, lon))
        if info.get("needs_replan"):
            planner.replan(current_gps=(lat, lon))
        if info.get("status") == "ARRIVED":
            print("Goal reached!")
    """

    def __init__(
        self,
        geo_db_path:     Optional[str] = None,
        dualhead_ckpt:   Optional[str] = None,
        backbone_ckpt:   str = "checkpoints/dinov2_student/student_best.pt",
        max_radius_m:    float = MAX_GRAPH_RADIUS,
        switch_thresh:   float = SWITCH_THRESH,
        replan_timeout:  int   = REPLAN_TIMEOUT,
        arrival_thresh:  float = ARRIVAL_THRESH,
    ):
        self.graph    = WaypointGraph(max_radius_m=max_radius_m)
        self.planner  = MirrorAscentPlanner()
        self.executor = WaypointExecutor(
            self.planner, switch_thresh, replan_timeout, arrival_thresh
        )
        self._goal_node: Optional[WaypointNode] = None
        self._last_gps:  Optional[tuple[float, float]] = None
        self._total_replans: int = 0

        # Language goal resolver (optional)
        self.lang_resolver: Optional[LanguageGoalResolver] = None
        if dualhead_ckpt and Path(dualhead_ckpt).exists():
            self.lang_resolver = LanguageGoalResolver(dualhead_ckpt, backbone_ckpt)

        # Load GeoLatentDB
        if geo_db_path and Path(geo_db_path).exists():
            self._load_geo_db(geo_db_path)
        else:
            print("[WaypointPlanner] No GeoLatentDB loaded — run in mock mode")

    def _load_geo_db(self, path: str):
        print(f"[WaypointPlanner] Loading GeoLatentDB: {path}")
        data = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            geodict = data
        else:
            # GeoLatentDatabase object
            geodict = {"gps": data.gps, "embeddings": data.embeddings}
        n = self.graph.build_from_geodict(geodict)
        print(f"[WaypointPlanner] Graph built: {n:,} nodes")

    def set_goal_gps(
        self,
        goal_lat:    float,
        goal_lon:    float,
        start_lat:   Optional[float] = None,
        start_lon:   Optional[float] = None,
    ):
        """Set navigation goal by GPS coordinate."""
        self._goal_node = self.graph.nearest_node(goal_lat, goal_lon)
        if start_lat is not None and start_lon is not None:
            self._plan(start_lat, start_lon)
            self._last_gps = (start_lat, start_lon)

    def set_goal_text(self, text: str) -> bool:
        """Set navigation goal by text query (requires Sprint 6d)."""
        if self.lang_resolver is None:
            print("[WaypointPlanner] Language resolver not available")
            return False
        node = self.lang_resolver.resolve(text, self.graph)
        if node is None:
            return False
        self._goal_node = node
        print(f"[WaypointPlanner] Language goal -> "
              f"({node.lat:.5f}, {node.lon:.5f})")
        return True

    def _plan(self, start_lat: float, start_lon: float):
        if self._goal_node is None:
            raise RuntimeError("Set a goal before planning")
        t0 = time.perf_counter()
        waypoints = self.graph.plan(
            start_lat, start_lon,
            self._goal_node.lat, self._goal_node.lon,
        )
        dt_ms = (time.perf_counter() - t0) * 1000
        self.executor.reset(waypoints)
        dist_m = gps_distance_m(start_lat, start_lon,
                                self._goal_node.lat, self._goal_node.lon)
        print(f"[WaypointPlanner] Plan: {len(waypoints)} waypoints, "
              f"dist={dist_m:.1f}m, latency={dt_ms:.2f}ms")

    def replan(self, current_gps: tuple[float, float]):
        """Replan from current GPS position."""
        self._last_gps = current_gps
        self._total_replans += 1
        self._plan(*current_gps)

    def step(
        self,
        current_particles: np.ndarray,
        current_gps:       Optional[tuple[float, float]] = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Execute one 4Hz navigation step.
        Returns (action, info).
        """
        if current_gps:
            self._last_gps = current_gps

        action, info = self.executor.step(current_particles, current_gps)

        # Auto-replan if executor requests it and we have GPS
        if info.get("needs_replan") and self._last_gps:
            self.replan(self._last_gps)
            # Re-run step with fresh plan
            action, info = self.executor.step(current_particles, current_gps)

        return action, info

    @property
    def is_arrived(self) -> bool:
        return (self.executor.state is not None
                and self.executor.state.arrived)

    @property
    def n_nodes(self) -> int:
        return len(self.graph.nodes)

DEVICE = torch.device("cpu")

# ── Hyperparameters ───────────────────────────────────────────────────────────

LANGUAGE_WEIGHT     = 0.35   # λ — how much language pulls vs GPS pushes
                              # 0 = pure GPS A*, 1 = pure language greedy
MIN_SCORE_THRESHOLD = 0.10   # ignore nodes with language score below this
TOP_K_GOALS         = 5      # candidate goal nodes from language scoring
SCORE_TEMPERATURE   = 0.5    # softmax temperature for goal selection


# ── CLIPBridge loader ─────────────────────────────────────────────────────────

class CLIPBridge(nn.Module):
    """Sprint 6c CLIPBridge architecture — 512→128."""
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(512, 128, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), dim=-1)


def load_clip_bridge(ckpt_path: str) -> tuple:
    """
    Load CLIPBridge from Sprint 6c/6e checkpoint.
    Returns (clip_model, clip_bridge) both frozen and eval.
    """
    try:
        import clip as clip_module
    except ImportError:
        raise ImportError(
            "Install CLIP: pip install git+https://github.com/openai/CLIP.git"
        )

    clip_model, _ = clip_module.load("ViT-B/32", device=DEVICE)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    bridge = CLIPBridge().to(DEVICE)
    if Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cb_sd = ckpt.get("clip_bridge", {})
        # Remap bare Linear keys from Sprint 6c format
        if "weight" in cb_sd and "proj.weight" not in cb_sd:
            cb_sd = {"proj." + k: v for k, v in cb_sd.items()}
        missing, _ = bridge.load_state_dict(cb_sd, strict=False)
        if missing:
            print(f"[CLIPBridge] Reinitialised: {missing}")
        ep   = ckpt.get("epoch", "?")
        loss = ckpt.get("loss",  "?")
        print(f"[CLIPBridge] Loaded: ep{ep}, L_clip={loss:.4f}" if isinstance(loss, float)
              else f"[CLIPBridge] Loaded: ep{ep}")
    else:
        print(f"[CLIPBridge] Checkpoint not found — random weights")

    bridge.eval()
    for p in bridge.parameters():
        p.requires_grad_(False)

    return clip_module, clip_model, bridge


# ── Language scorer ───────────────────────────────────────────────────────────

class LanguageScorer:
    """
    Scores all GeoLatentDB nodes against a text query.

    Pre-computes scores once per query — O(N×D) dot products.
    Scores are cosine similarities in the CLIPBridge embedding space.
    """

    def __init__(
        self,
        clip_module,
        clip_model,
        clip_bridge:     CLIPBridge,
        min_score:       float = MIN_SCORE_THRESHOLD,
        score_temperature: float = SCORE_TEMPERATURE,
    ):
        self.clip_module   = clip_module
        self.clip_model    = clip_model
        self.clip_bridge   = clip_bridge
        self.min_score     = min_score
        self.temperature   = score_temperature
        self._scores:      Optional[np.ndarray] = None   # (N,) current scores
        self._query:       Optional[str]         = None
        self._nodes_embs:  Optional[np.ndarray] = None   # (N, 128) cached

    def build_node_matrix(self, nodes: list[WaypointNode]):
        """Pre-compute normalised node embedding matrix. Call once after graph build."""
        embs = np.stack([n.embedding for n in nodes])   # (N, 128)
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
        self._nodes_embs = embs / norms                  # (N, 128) unit-norm
        print(f"[LanguageScorer] Node matrix: {self._nodes_embs.shape}")

    def encode_query(self, text: str) -> np.ndarray:
        """Encode a text query via CLIP → CLIPBridge. Returns (128,) unit-norm."""
        with torch.no_grad():
            tokens  = self.clip_module.tokenize([text]).to(DEVICE)
            txt_emb = self.clip_model.encode_text(tokens).float()
            txt_emb = F.normalize(txt_emb, dim=-1)
            sem_emb = self.clip_bridge(txt_emb)
        return sem_emb.squeeze(0).numpy()

    def score(self, text: str) -> np.ndarray:
        """
        Score all N nodes against text query.
        Returns (N,) cosine similarities, cached until query changes.
        """
        if self._nodes_embs is None or len(self._nodes_embs) == 0:
            raise RuntimeError(
                "Node matrix not built — call build_node_matrix(nodes) first. "
                "Requires a loaded GeoLatentDB."
            )
        if text == self._query and self._scores is not None:
            return self._scores

        t0       = time.perf_counter()
        lang_emb = self.encode_query(text)                    # (128,)
        scores   = self._nodes_embs @ lang_emb               # (N,) dot products
        # Clamp negatives — below zero means anti-correlated, treat as zero
        scores   = np.clip(scores, 0.0, 1.0)
        dt_ms    = (time.perf_counter() - t0) * 1000

        self._scores = scores
        self._query  = text
        print(f"[LanguageScorer] '{text[:50]}' → "
              f"max={scores.max():.3f} mean={scores.mean():.3f} "
              f"p95={np.percentile(scores, 95):.3f} ({dt_ms:.1f}ms)")
        return scores

    def top_goal_nodes(
        self,
        nodes:  list[WaypointNode],
        text:   str,
        top_k:  int = TOP_K_GOALS,
    ) -> list[tuple[WaypointNode, float]]:
        """Return top-K nodes sorted by language score."""
        scores  = self.score(text)
        top_idx = np.argsort(scores)[-top_k:][::-1]
        return [(nodes[i], float(scores[i])) for i in top_idx
                if scores[i] >= self.min_score]


# ── Language-weighted A* ──────────────────────────────────────────────────────

class LanguageWeightedGraph(WaypointGraph):
    """
    Extends WaypointGraph with language-weighted A*.

    Cost modification:
        edge_cost(u→v) = dist_m(u,v) × (1 - λ × score[v])
        heuristic(v)   = dist_m(v,goal) × (1 - λ × score[v])

    λ=0: identical to standard GPS A* (Sprint 7a)
    λ=1: path completely guided by language scores, ignoring distance
    λ=0.35: balanced — language pulls path toward relevant regions
             while GPS keeps the route geometrically coherent

    The effect: nodes with high language relevance become "cheaper" to
    traverse. The planner naturally threads through high-relevance regions
    on the way to the goal, rather than taking the pure shortest path.
    """

    def __init__(self, *args, language_weight: float = LANGUAGE_WEIGHT, **kwargs):
        super().__init__(*args, **kwargs)
        self.language_weight = language_weight
        self._lang_scores:  Optional[np.ndarray] = None   # (N,) current scores

    def set_language_scores(self, scores: np.ndarray):
        """Set pre-computed language scores for current query."""
        self._lang_scores = scores

    def _edge_cost(self, dist_m: float, node_idx: int) -> float:
        """Distance modified by language relevance of the destination node."""
        if self._lang_scores is None or self.language_weight == 0.0:
            return dist_m
        score = float(self._lang_scores[node_idx])
        return dist_m * (1.0 - self.language_weight * score)

    def _heuristic(self, node: WaypointNode, goal: WaypointNode) -> float:
        """Haversine distance modified by language relevance of current node."""
        h = gps_distance_m(node.lat, node.lon, goal.lat, goal.lon)
        if self._lang_scores is None or self.language_weight == 0.0:
            return h
        score = float(self._lang_scores[node.idx])
        return h * (1.0 - self.language_weight * score)

    def plan_weighted(
        self,
        start_lat:     float,
        start_lon:     float,
        goal_lat:      float,
        goal_lon:      float,
        max_waypoints: int = N_WAYPOINTS_MAX,
    ) -> list[WaypointNode]:
        """
        Language-weighted A* from start GPS to goal GPS.
        Requires set_language_scores() to have been called first.
        Falls back to standard A* if no scores set.
        """
        if self._kdtree is None:
            raise RuntimeError("Graph not built — call build_from_geodict() first")

        start_node = self.nearest_node(start_lat, start_lon)
        goal_node  = self.nearest_node(goal_lat, goal_lon)

        if start_node.idx == goal_node.idx:
            return [goal_node]

        open_heap: list[_AStarItem] = []
        came_from: dict[int, Optional[int]] = {}
        g_score:   dict[int, float] = {start_node.idx: 0.0}

        h0 = self._heuristic(start_node, goal_node)
        heapq.heappush(open_heap, _AStarItem(h0, start_node.idx, 0.0, None))
        came_from[start_node.idx] = None

        while open_heap:
            item = heapq.heappop(open_heap)
            curr_idx, g = item.idx, item.g_score

            if curr_idx == goal_node.idx:
                break

            if g > g_score.get(curr_idx, float("inf")):
                continue

            for nb_idx, dist in self._neighbours(self.nodes[curr_idx]):
                edge_cost = self._edge_cost(dist, nb_idx)
                new_g     = g + edge_cost
                if new_g < g_score.get(nb_idx, float("inf")):
                    g_score[nb_idx]   = new_g
                    came_from[nb_idx] = curr_idx
                    h = self._heuristic(self.nodes[nb_idx], goal_node)
                    heapq.heappush(open_heap,
                                   _AStarItem(new_g + h, nb_idx, new_g))

        # Reconstruct
        path, cur = [], goal_node.idx
        while cur is not None:
            path.append(self.nodes[cur])
            cur = came_from.get(cur)
        path.reverse()

        # Subsample
        if len(path) > max_waypoints:
            step = len(path) / (max_waypoints - 1)
            idxs = [int(i * step) for i in range(max_waypoints - 1)] + [len(path) - 1]
            path = [path[i] for i in sorted(set(idxs))]

        return path


# ── Full language navigation system ──────────────────────────────────────────

class LanguageNavigator:
    """
    Complete Sprint 7b system.
    Text query → language-scored A* → WaypointExecutor → 4Hz actions.

    Usage:
        nav = LanguageNavigator(
            geo_db_path  = "checkpoints/cwm/geo_latent_db.pt",
            clip_ckpt    = "checkpoints/dinov2_student/student_dualhead_nce_best.pt",
        )

        # Set a text goal
        nav.set_goal("navigate to the open road area near the trees")

        # 4Hz loop
        while not nav.is_arrived:
            action, info = nav.step(current_particles, current_gps=(lat, lon))
            send_action(action)
            print(info["status"], info["active_score"])
    """

    def __init__(
        self,
        geo_db_path:      Optional[str] = None,
        clip_ckpt:        Optional[str] = None,
        language_weight:  float = LANGUAGE_WEIGHT,
        max_radius_m:     float = MAX_GRAPH_RADIUS,
        switch_thresh:    float = SWITCH_THRESH,
        replan_timeout:   int   = REPLAN_TIMEOUT,
        arrival_thresh:   float = ARRIVAL_THRESH,
    ):
        self.graph    = LanguageWeightedGraph(
            max_radius_m   = max_radius_m,
            language_weight = language_weight,
        )
        self.planner  = MirrorAscentPlanner()
        self.executor = WaypointExecutor(
            self.planner, switch_thresh, replan_timeout, arrival_thresh
        )
        self.scorer:   Optional[LanguageScorer] = None
        self._query:   Optional[str]            = None
        self._goal_node: Optional[WaypointNode] = None
        self._last_gps:  Optional[tuple]        = None
        self._total_replans = 0

        # Load GeoLatentDB
        if geo_db_path and Path(geo_db_path).exists():
            self._load_geo_db(geo_db_path)
        else:
            print("[LanguageNavigator] No GeoLatentDB — mock mode")

        # Load CLIPBridge
        if clip_ckpt:
            try:
                clip_module, clip_model, bridge = load_clip_bridge(clip_ckpt)
                self.scorer = LanguageScorer(clip_module, clip_model, bridge)
                if self.graph.nodes:
                    self.scorer.build_node_matrix(self.graph.nodes)
            except Exception as e:
                print(f"[LanguageNavigator] CLIPBridge unavailable: {e}")

    def _load_geo_db(self, path: str):
        print(f"[LanguageNavigator] Loading GeoLatentDB: {path}")
        data = torch.load(path, map_location="cpu", weights_only=False)
        geodict = data if isinstance(data, dict) else {
            "gps": data.gps, "embeddings": data.embeddings
        }
        n = self.graph.build_from_geodict(geodict)
        print(f"[LanguageNavigator] Graph: {n:,} nodes")

    def set_goal(
        self,
        text:      str,
        start_gps: Optional[tuple] = None,
    ) -> Optional[WaypointNode]:
        """
        Set navigation goal by text query.
        Returns the selected goal node, or None if scorer unavailable.
        """
        self._query = text

        if self.scorer is None:
            print("[LanguageNavigator] No CLIPBridge — cannot resolve text goal")
            return None

        if not self.graph.nodes:
            print("[LanguageNavigator] No GeoLatentDB nodes — cannot score text goal. "
                  "Load a GeoLatentDB with --geo-db.")
            return None

        if self.scorer._nodes_embs is None:
            self.scorer.build_node_matrix(self.graph.nodes)

        # Score all nodes
        scores = self.scorer.score(text)
        self.graph.set_language_scores(scores)

        # Select goal: highest scoring node
        candidates = self.scorer.top_goal_nodes(self.graph.nodes, text, top_k=TOP_K_GOALS)
        if not candidates:
            print("[LanguageNavigator] No nodes above score threshold")
            return None

        # Log top candidates
        print(f"[LanguageNavigator] Top {len(candidates)} goal candidates:")
        for node, score in candidates[:3]:
            print(f"  ({node.lat:.5f}, {node.lon:.5f}) score={score:.3f}")

        self._goal_node = candidates[0][0]

        if start_gps:
            self._last_gps = start_gps
            self._plan(*start_gps)

        return self._goal_node

    def set_goal_gps(self, lat: float, lon: float, start_gps=None):
        """Set GPS goal without language scoring (Sprint 7a compatibility)."""
        self._goal_node = self.graph.nearest_node(lat, lon)
        if start_gps:
            self._last_gps = start_gps
            self._plan(*start_gps)

    def _plan(self, start_lat: float, start_lon: float):
        if self._goal_node is None:
            raise RuntimeError("Set a goal before planning")

        t0 = time.perf_counter()
        waypoints = self.graph.plan_weighted(
            start_lat, start_lon,
            self._goal_node.lat, self._goal_node.lon,
        )
        dt_ms = (time.perf_counter() - t0) * 1000

        self.executor.reset(waypoints)
        dist_m = gps_distance_m(start_lat, start_lon,
                                self._goal_node.lat, self._goal_node.lon)

        # Compute mean language score along planned path
        if self.graph._lang_scores is not None:
            path_scores = [float(self.graph._lang_scores[n.idx]) for n in waypoints]
            mean_score  = float(np.mean(path_scores))
            max_score   = float(np.max(path_scores))
        else:
            mean_score = max_score = 0.0

        print(f"[LanguageNavigator] Plan: {len(waypoints)} waypoints, "
              f"dist={dist_m:.1f}m, {dt_ms:.2f}ms | "
              f"lang_score mean={mean_score:.3f} max={max_score:.3f}")

    def replan(self, current_gps: tuple):
        self._last_gps = current_gps
        self._total_replans += 1
        self._plan(*current_gps)

    def step(
        self,
        current_particles: np.ndarray,
        current_gps:       Optional[tuple] = None,
    ) -> tuple[np.ndarray, dict]:
        """4Hz navigation step. Returns (action, info)."""
        if current_gps:
            self._last_gps = current_gps

        action, info = self.executor.step(current_particles, current_gps)

        if info.get("needs_replan") and self._last_gps:
            self.replan(self._last_gps)
            action, info = self.executor.step(current_particles, current_gps)

        # Attach language score of active waypoint to info
        if (self.graph._lang_scores is not None
                and self.executor.state is not None
                and not self.executor.state.arrived):
            wp_idx = self.executor.state.waypoint_idx
            if wp_idx < len(self.executor.state.waypoints):
                wp = self.executor.state.waypoints[wp_idx]
                info["active_score"] = float(self.graph._lang_scores[wp.idx])
                info["query"]        = self._query
        return action, info

    @property
    def is_arrived(self) -> bool:
        return (self.executor.state is not None
                and self.executor.state.arrived)

    @property
    def n_nodes(self) -> int:
        return len(self.graph.nodes)


# ── Self-test ─────────────────────────────────────────────────────────────────

def self_test():
    print("\n" + "=" * 60)
    print("  Sprint 7b — Language Navigation Self-Test")
    print("=" * 60)

    rng = np.random.default_rng(42)
    n   = 500

    # Build synthetic graph
    lats = rng.uniform(37.910, 37.920, n)
    lons = rng.uniform(-122.340, -122.330, n)
    embs = rng.standard_normal((n, 128)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)

    # Build language-weighted graph
    graph = LanguageWeightedGraph(max_radius_m=0.30)
    graph.build_from_geodict({"gps": np.stack([lats, lons], 1), "embeddings": embs})

    print(f"  Graph: {len(graph.nodes):,} nodes")

    # ── Test 1: Pure GPS plan (λ=0) baseline ────────────────────────────
    print("\n── Test 1: GPS baseline (language_weight=0)")
    graph.language_weight = 0.0
    t0 = time.perf_counter()
    path_gps = graph.plan(37.911, -122.339, 37.919, -122.331)
    dt_gps   = (time.perf_counter() - t0) * 1000
    print(f"  GPS path: {len(path_gps)} waypoints, {dt_gps:.2f}ms")
    assert 1 <= len(path_gps) <= N_WAYPOINTS_MAX
    print("  PASS")

    # ── Test 2: Language-weighted plan (λ=0.35) ─────────────────────────
    print("\n── Test 2: Language-weighted plan (λ=0.35)")
    # Inject synthetic language scores — make nodes in middle-lat region score high
    synth_scores = np.zeros(n, dtype=np.float32)
    for i, (lat, lon) in enumerate(zip(lats, lons)):
        # High score near lat=37.915, lon=-122.335 (middle of graph)
        d = ((lat - 37.915)**2 + (lon + 122.335)**2) ** 0.5
        synth_scores[i] = max(0.0, 1.0 - d * 500)

    graph.language_weight = LANGUAGE_WEIGHT
    graph.set_language_scores(synth_scores)

    t0 = time.perf_counter()
    path_lang = graph.plan_weighted(37.911, -122.339, 37.919, -122.331)
    dt_lang   = (time.perf_counter() - t0) * 1000
    print(f"  Language path: {len(path_lang)} waypoints, {dt_lang:.2f}ms")

    # Weighted path should thread through high-score region
    lang_scores_on_path = [float(synth_scores[n.idx]) for n in path_lang]
    gps_scores_on_path  = [float(synth_scores[n.idx]) for n in path_gps]
    print(f"  Mean score — language path: {np.mean(lang_scores_on_path):.4f}")
    print(f"  Mean score — GPS path:      {np.mean(gps_scores_on_path):.4f}")
    # Language path should have higher mean score
    assert np.mean(lang_scores_on_path) >= np.mean(gps_scores_on_path) - 0.001, \
        "Language path should thread through higher-scoring nodes"
    print("  PASS — language path scores ≥ GPS path scores")

    # ── Test 3: LanguageNavigator execution ─────────────────────────────
    print("\n── Test 3: Navigator 4Hz execution (50 steps)")
    nav = LanguageNavigator(max_radius_m=0.30)
    nav.graph.build_from_geodict({"gps": np.stack([lats, lons], 1), "embeddings": embs})
    nav.graph.set_language_scores(synth_scores)
    # Manually set goal without CLIPBridge
    nav._goal_node = nav.graph.nodes[int(synth_scores.argmax())]
    nav._last_gps  = (37.911, -122.339)
    nav._plan(37.911, -122.339)

    step_times = []
    for _ in range(50):
        particles = rng.standard_normal((16, 128)).astype(np.float32)
        particles /= np.linalg.norm(particles, axis=1, keepdims=True)
        t0 = time.perf_counter()
        action, info = nav.step(particles, current_gps=(37.911, -122.339))
        step_times.append((time.perf_counter() - t0) * 1000)
        assert action.shape == (2,)

    print(f"  Step latency: median={np.median(step_times):.3f}ms "
          f"p95={np.percentile(step_times,95):.3f}ms")
    print("  PASS")

    # ── Test 4: Score query (mock — no CLIP) ────────────────────────────
    print("\n── Test 4: Top goal candidates from scores")
    # Build a fake scorer that returns synth_scores
    class MockScorer:
        def score(self, text):
            return synth_scores
        def top_goal_nodes(self, nodes, text, top_k=5):
            scores = self.score(text)
            top_idx = np.argsort(scores)[-top_k:][::-1]
            return [(nodes[i], float(scores[i])) for i in top_idx]

    mock_scorer = MockScorer()
    candidates = mock_scorer.top_goal_nodes(nav.graph.nodes, "open road area")
    best_node, best_score = candidates[0]
    print(f"  Best goal: ({best_node.lat:.5f}, {best_node.lon:.5f}) score={best_score:.3f}")
    assert best_score == float(synth_scores.max())
    print("  PASS")

    # ── Test 5: Latency benchmark ────────────────────────────────────────
    print("\n── Test 5: Planning latency")
    graph.set_language_scores(synth_scores)
    times = []
    for _ in range(30):
        sl, sg = rng.uniform(37.910,37.914), rng.uniform(-122.340,-122.336)
        gl, gg = rng.uniform(37.916,37.920), rng.uniform(-122.334,-122.330)
        t0 = time.perf_counter()
        graph.plan_weighted(sl, sg, gl, gg)
        times.append((time.perf_counter() - t0) * 1000)
    print(f"  Weighted A*: median={np.median(times):.2f}ms "
          f"p95={np.percentile(times,95):.2f}ms")
    assert np.median(times) < 50.0
    print("  PASS")

    print("\n" + "=" * 60)
    print("  All 5 tests PASSED")
    print("=" * 60 + "\n")


# ── Benchmark ─────────────────────────────────────────────────────────────────

def benchmark(geo_db_path=None, clip_ckpt=None):
    print("\n── Sprint 7b Benchmark")
    rng = np.random.default_rng(0)

    nav = LanguageNavigator(geo_db_path=geo_db_path, clip_ckpt=clip_ckpt,
                            max_radius_m=0.30)

    if not nav.n_nodes:
        n = 5000
        lats = rng.uniform(37.910, 37.920, n)
        lons = rng.uniform(-122.340, -122.330, n)
        embs = rng.standard_normal((n, 128)).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
        nav.graph.build_from_geodict(
            {"gps": np.stack([lats, lons], 1), "embeddings": embs}
        )

    print(f"  Nodes: {nav.n_nodes:,}")

    # Scoring benchmark
    if nav.scorer:
        nav.scorer.build_node_matrix(nav.graph.nodes)
        # Separate CLIP text encoding (one-time, ~40ms CPU) from matrix multiply
        tok = nav.scorer.clip_module.tokenize(
            ["robot driving toward open outdoor area"]).to(DEVICE)
        t_clip = time.perf_counter()
        with torch.no_grad():
            txt_e = nav.scorer.clip_model.encode_text(tok).float()
            txt_e = F.normalize(txt_e, dim=-1)
        clip_ms = (time.perf_counter() - t_clip) * 1000

        t_mat = time.perf_counter()
        lang_emb = nav.scorer.clip_bridge(txt_e)
        lang_np  = F.normalize(lang_emb, dim=-1).squeeze(0).numpy()
        scores   = nav.scorer._nodes_embs @ lang_np
        scores   = np.clip(scores, 0.0, 1.0)
        mat_ms   = (time.perf_counter() - t_mat) * 1000
        score_ms = clip_ms + mat_ms

        print(f"\n  Language scoring ({nav.n_nodes:,} nodes):")
        print(f"    CLIP encode: {clip_ms:.1f}ms  (one-time per query, CPU-bound)")
        print(f"    Matrix mul:  {mat_ms:.2f}ms  ({nav.n_nodes:,}×128 dot products)")
        print(f"    Total:       {score_ms:.1f}ms  | max={scores.max():.3f} mean={scores.mean():.3f}")
        nav.graph.set_language_scores(scores)
        nav.graph.set_language_scores(scores)
    else:
        # Inject random scores for benchmark
        scores = rng.random(nav.n_nodes).astype(np.float32)
        nav.graph.set_language_scores(scores)
        score_ms = 0.0

    # Planning benchmark
    plan_times_gps, plan_times_lang = [], []
    for _ in range(30):
        sl,sg = rng.uniform(37.910,37.914), rng.uniform(-122.340,-122.336)
        gl,gg = rng.uniform(37.916,37.920), rng.uniform(-122.334,-122.330)

        nav.graph.language_weight = 0.0
        t0 = time.perf_counter()
        nav.graph.plan(sl, sg, gl, gg)
        plan_times_gps.append((time.perf_counter()-t0)*1000)

        nav.graph.language_weight = LANGUAGE_WEIGHT
        t0 = time.perf_counter()
        nav.graph.plan_weighted(sl, sg, gl, gg)
        plan_times_lang.append((time.perf_counter()-t0)*1000)

    # Step benchmark
    nav._goal_node = nav.graph.nodes[int(scores.argmax())]
    nav._plan(37.911, -122.339)
    step_times = []
    for _ in range(200):
        particles = rng.standard_normal((16,128)).astype(np.float32)
        particles /= np.linalg.norm(particles, axis=1, keepdims=True)
        t0 = time.perf_counter()
        nav.step(particles, current_gps=(37.911, -122.339))
        step_times.append((time.perf_counter()-t0)*1000)

    print(f"\n  GPS A*:          median={np.median(plan_times_gps):.2f}ms")
    print(f"  Language A*:     median={np.median(plan_times_lang):.2f}ms")
    print(f"  Step (4Hz loop): median={np.median(step_times):.3f}ms "
          f"p95={np.percentile(step_times,95):.3f}ms")

    per_step_ms = np.median(plan_times_lang) + np.median(step_times)
    first_query_ms = score_ms + per_step_ms
    budget_ok = per_step_ms < 1.0   # 4Hz budget: 250ms/frame, target <1ms overhead
    print(f"\n  First query latency: {first_query_ms:.1f}ms "
          f"(CLIP encode {clip_ms:.0f}ms one-time + plan {np.median(plan_times_lang):.2f}ms)")
    print(f"  Per-step latency:    {per_step_ms:.3f}ms "
          f"({'PASS ✓' if budget_ok else 'FAIL ✗'} vs 1ms target)")
    print(f"  CLIP encode is CPU-bound (~40ms). GPU/NPU would reduce to <2ms.")
    print(f"  Subsequent queries to same goal: {per_step_ms:.3f}ms (scores cached)")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sprint 7b — Language Navigation")
    parser.add_argument("--self-test",   action="store_true")
    parser.add_argument("--benchmark",   action="store_true")
    parser.add_argument("--text",        default=None,
                        help="Text navigation goal")
    parser.add_argument("--start-lat",   type=float, default=37.9105)
    parser.add_argument("--start-lon",   type=float, default=-122.3395)
    parser.add_argument("--geo-db",      default="checkpoints/cwm/geo_latent_db.pt")
    parser.add_argument("--clip-ckpt",   default="checkpoints/dinov2_student/student_dualhead_nce_best.pt")
    parser.add_argument("--lang-weight", type=float, default=LANGUAGE_WEIGHT)
    parser.add_argument("--n-steps",     type=int,   default=100)
    args = parser.parse_args()

    if args.self_test:
        self_test()

    elif args.benchmark:
        benchmark(
            geo_db_path = args.geo_db if Path(args.geo_db).exists() else None,
            clip_ckpt   = args.clip_ckpt if Path(args.clip_ckpt).exists() else None,
        )

    elif args.text:
        nav = LanguageNavigator(
            geo_db_path     = args.geo_db,
            clip_ckpt       = args.clip_ckpt,
            language_weight = args.lang_weight,
        )
        goal = nav.set_goal(args.text, start_gps=(args.start_lat, args.start_lon))
        if goal is None:
            print("No goal set — check CLIPBridge checkpoint")
        else:
            rng = np.random.default_rng(0)
            print(f"\nSimulating {args.n_steps} steps toward '{args.text[:50]}'...\n")
            for i in range(args.n_steps):
                particles = rng.standard_normal((16, 128)).astype(np.float32)
                particles /= np.linalg.norm(particles, axis=1, keepdims=True)
                action, info = nav.step(
                    particles,
                    current_gps=(args.start_lat, args.start_lon)
                )
                if i % 10 == 0 or info["status"] != "navigating":
                    score_str = f"lang={info.get('active_score', 0):.3f}"
                    print(f"  step {i:03d} | wp {info['waypoint']}/{info['n_waypoints']-1} "
                          f"| sim={info['sim']:.3f} | {score_str} | {info['status']}")
                if nav.is_arrived:
                    print("\n  ✓ ARRIVED at language goal")
                    break
    else:
        parser.print_help()
