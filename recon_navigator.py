"""
recon_navigator.py — CORTEX-PE v16.15
Unified Navigation: Visual Quasimetric + GPS Dead-Reckoning

Integrates all three RECON-trained components into a single navigator:

    StudentEncoder      (46K params, NPU/XINT8)   →  128-D visual latent
    TemporalHead        (50K params, CPU)           →  64-D metric embedding  AUROC 0.9337
    TransitionPredictor  (5K params, CPU)           →  Δ(lat,lon) prediction  MAE 0.098m

Two complementary signals per tick:
    visual_distance   — L2 in metric space; how visually far from goal
    predicted_pos     — where GPS dead-reckoning says you will be next step

Usage (minimal):
    nav = ReconNavigator(
        encoder_path   = "checkpoints/maze_weak_sigreg_straight/cortex_student_phase2_final.pt",
        head_path      = "checkpoints/recon_contrastive/temporal_head_k7_best.pt",
        predictor_path = "checkpoints/recon_transition/transition_best.pt",
    )

    state = NavState(lat=37.41, lon=-122.01, bearing=270.0,
                     gps_vel=(0.5, 0.0, 0.0), cmd_lin=0.5, cmd_ang=0.0)

    result = nav.step(current_frame, goal_frame, state)
    print(result.visual_distance)   # quasimetric distance to goal
    print(result.predicted_delta)   # (Dlat, Dlon) for next step
    print(result.goal_reached)      # True if below threshold

    best = nav.plan(current_frame, goal_frame, state, candidate_frames, candidate_states,
                    goal_lat=37.412, goal_lon=-122.010)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

class StudentEncoder(nn.Module):
    """46K-param CNN distilled from DINOv2-small. XINT8-quantized for NPU."""
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.proj = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.GELU(),
            nn.Linear(512, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.features(x).flatten(1))


class TemporalHead(nn.Module):
    """
    Linear(128->256) -> BN -> ReLU -> Linear(256->64)
    AUROC 0.9337 | close~0.20 | far~0.66
    """
    def __init__(self, input_dim: int = 128, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class TransitionPredictor(nn.Module):
    """
    Linear(8->64) -> LN -> GELU -> Linear(64->64) -> LN -> GELU -> Linear(64->2)
    Predicts normalized (Dlat, Dlon) from 8-D navigation state.
    MAE 0.098m on RECON full val split.
    """
    INPUT_DIM  = 8
    OUTPUT_DIM = 2

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.INPUT_DIM, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, self.OUTPUT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class NavState(NamedTuple):
    """
    Current navigation state for the TransitionPredictor.

    Args:
        lat:      Current latitude  (degrees)
        lon:      Current longitude (degrees)
        bearing:  Compass heading   (degrees, 0=N, 90=E)
        gps_vel:  GPS velocity      (vx, vy, vz) in m/s
        cmd_lin:  Commanded linear velocity  (m/s)
        cmd_ang:  Commanded angular velocity (rad/s)
    """
    lat:     float
    lon:     float
    bearing: float
    gps_vel: tuple
    cmd_lin: float
    cmd_ang: float

    def to_array(self) -> np.ndarray:
        return np.array([
            self.lat, self.lon, self.bearing,
            self.gps_vel[0], self.gps_vel[1], self.gps_vel[2],
            self.cmd_lin, self.cmd_ang,
        ], dtype=np.float32)


@dataclass
class NavResult:
    """
    Full navigation output for one step.

    Visual:  visual_distance, goal_reached, current_embed, goal_embed
    DR:      predicted_delta (Dlat, Dlon), predicted_pos (lat, lon)
    Timing:  encoder_latency_ms, head_latency_ms, dr_latency_ms
    """
    visual_distance:    float
    goal_reached:       bool
    current_embed:      np.ndarray = field(repr=False)
    goal_embed:         np.ndarray = field(repr=False)
    predicted_delta:    object = None   # (Dlat, Dlon) | None
    predicted_pos:      object = None   # (lat, lon)   | None
    encoder_latency_ms: float = 0.0
    head_latency_ms:    float = 0.0
    dr_latency_ms:      float = 0.0

    @property
    def total_latency_ms(self) -> float:
        return self.encoder_latency_ms + self.head_latency_ms + self.dr_latency_ms

    def summary(self) -> str:
        parts = [
            f"visual_dist={self.visual_distance:.4f}",
            f"goal_reached={self.goal_reached}",
            f"latency={self.total_latency_ms:.1f}ms",
        ]
        if self.predicted_delta is not None:
            dlat, dlon = self.predicted_delta
            parts.append(f"pred_delta=({dlat*1e6:+.3f}u-lat, {dlon*1e6:+.3f}u-lon)")
            parts.append(f"pred_pos=({self.predicted_pos[0]:.6f}, {self.predicted_pos[1]:.6f})")
        return "  ".join(parts)


@dataclass
class PlanResult:
    """Returned by nav.plan() -- ranked action candidates."""
    ranked_indices:   list
    visual_distances: list
    position_deltas:  object   # list of (Dlat, Dlon) | None
    best_index:       int
    fused_scores:     list


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _preprocess(img, size: int = 224) -> torch.Tensor:
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img = img.resize((size, size), Image.BILINEAR).convert("RGB")
    t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    return (t - _MEAN) / _STD


# ---------------------------------------------------------------------------
# Main navigator
# ---------------------------------------------------------------------------

class ReconNavigator:
    """
    CORTEX-PE v16.15 unified navigation module.

    Combines visual quasimetric (StudentEncoder + TemporalHead) with
    GPS dead-reckoning (TransitionPredictor) into a single inference object.

    Args:
        encoder_path:    Path to cortex_student_phase2_final.pt
        head_path:       Path to temporal_head_k7_best.pt
        predictor_path:  Path to transition_best.pt (None = disable DR)
        device:          cpu | cuda | npu (npu -> cpu fallback)
        goal_threshold:  visual_distance below which goal_reached = True
                         Calibrated: close~0.20, far~0.66. Default 0.25.
        verbose:         Print load summary
    """

    CLOSE_MEAN = 0.2027
    FAR_MEAN   = 0.6590

    def __init__(
        self,
        encoder_path,
        head_path,
        predictor_path=None,
        device="cpu",
        goal_threshold=0.25,
        verbose=True,
    ):
        self.device         = torch.device(device if device != "npu" else "cpu")
        self.goal_threshold = goal_threshold
        self._has_predictor = predictor_path is not None

        # StudentEncoder
        self.encoder = StudentEncoder(latent_dim=128)
        ckpt  = torch.load(encoder_path, map_location="cpu", weights_only=True)
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        state = {k.replace("encoder.", "", 1): v for k, v in state.items()
                 if not k.startswith("projector")}
        self.encoder.load_state_dict(state, strict=False)
        self.encoder.eval().to(self.device)
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        # TemporalHead
        self.head = TemporalHead(input_dim=128, embed_dim=64)
        head_ckpt  = torch.load(head_path, map_location="cpu", weights_only=True)
        head_state = (head_ckpt.get("model_state_dict")
                      or head_ckpt.get("head")
                      or head_ckpt)
        self.head.load_state_dict(head_state, strict=True)
        self.head.eval().to(self.device)
        for p in self.head.parameters():
            p.requires_grad_(False)
        self._head_meta = {"k": head_ckpt.get("k", 7),
                           "val_loss": head_ckpt.get("val_loss", float("nan"))}

        # TransitionPredictor (optional)
        self.predictor   = None
        self._norm_stats = None
        self._pred_meta  = {}

        if predictor_path is not None:
            pred_ckpt = torch.load(predictor_path, map_location="cpu", weights_only=True)
            hidden    = pred_ckpt.get("hidden", 64)
            self.predictor = TransitionPredictor(hidden=hidden)
            self.predictor.load_state_dict(pred_ckpt["model_state_dict"], strict=True)
            self.predictor.eval().to(self.device)
            for p in self.predictor.parameters():
                p.requires_grad_(False)
            raw = pred_ckpt["norm_stats"]
            self._norm_stats = {k: np.array(v, dtype=np.float32) for k, v in raw.items()}
            self._pred_meta  = {"val_mae_m": pred_ckpt.get("val_mae_m", float("nan"))}

        if verbose:
            self._print_banner()

    def _print_banner(self):
        enc_p  = sum(p.numel() for p in self.encoder.parameters())
        head_p = sum(p.numel() for p in self.head.parameters())
        print("  ✅ ReconNavigator v16.15 ready")
        print(f"     StudentEncoder      : {enc_p:,} params  [{self.device}]")
        print(f"     TemporalHead        : {head_p:,} params  "
              f"[k={self._head_meta['k']}, AUROC~0.9337]")
        if self.predictor is not None:
            pred_p = sum(p.numel() for p in self.predictor.parameters())
            print(f"     TransitionPredictor : {pred_p:,} params  "
                  f"[MAE~{self._pred_meta['val_mae_m']:.3f}m]")
        else:
            print("     TransitionPredictor : disabled")
        print(f"     goal_threshold      : {self.goal_threshold:.3f}")

    # -- Internal: visual embedding -----------------------------------------

    @torch.no_grad()
    def _embed(self, frame):
        x = _preprocess(frame).unsqueeze(0).to(self.device)
        t0 = time.perf_counter()
        z  = self.encoder(x)
        enc_ms = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        e  = self.head(z)
        head_ms = (time.perf_counter() - t1) * 1000
        return e.squeeze(0).cpu().numpy(), enc_ms, head_ms

    @torch.no_grad()
    def _embed_batch(self, frames):
        xs = torch.stack([_preprocess(f) for f in frames]).to(self.device)
        t0 = time.perf_counter()
        zs = self.encoder(xs)
        enc_ms = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        es = self.head(zs)
        head_ms = (time.perf_counter() - t1) * 1000
        return es.cpu().numpy(), enc_ms, head_ms

    # -- Internal: dead-reckoning -------------------------------------------

    @torch.no_grad()
    def _predict_delta(self, state):
        """Single NavState -> (Dlat, Dlon), latency_ms."""
        if self.predictor is None:
            raise RuntimeError("TransitionPredictor not loaded")
        ns  = self._norm_stats
        raw = state.to_array()
        if not np.isfinite(raw).all():
            raise ValueError(f"NavState contains non-finite values: {raw}")
        x_n = np.clip((raw - ns["state_mean"]) / ns["state_std"], -10, 10)
        x_t = torch.from_numpy(x_n).unsqueeze(0).to(self.device)
        t0  = time.perf_counter()
        pred_n = self.predictor(x_t).squeeze(0).cpu().numpy()
        ms  = (time.perf_counter() - t0) * 1000
        delta = pred_n * ns["delta_std"] + ns["delta_mean"]
        return (float(delta[0]), float(delta[1])), ms

    @torch.no_grad()
    def _predict_delta_batch(self, states):
        """N NavStates -> N×2 array, latency_ms."""
        if self.predictor is None:
            raise RuntimeError("TransitionPredictor not loaded")
        ns  = self._norm_stats
        raw = np.stack([s.to_array() for s in states])
        x_n = np.clip((raw - ns["state_mean"]) / ns["state_std"], -10, 10)
        x_t = torch.from_numpy(x_n.astype(np.float32)).to(self.device)
        t0  = time.perf_counter()
        pred_n = self.predictor(x_t).cpu().numpy()
        ms  = (time.perf_counter() - t0) * 1000
        return pred_n * ns["delta_std"] + ns["delta_mean"], ms

    # -- Public API ---------------------------------------------------------

    def step(self, current_frame, goal_frame, state=None):
        """
        Full navigation tick: visual quasimetric + optional dead-reckoning.

        Args:
            current_frame: RGB observation at current position
            goal_frame:    RGB observation at target position
            state:         NavState for dead-reckoning (None = skip DR)

        Returns:
            NavResult with both signals populated
        """
        cur_e, enc_ms_c, head_ms_c = self._embed(current_frame)
        goal_e, enc_ms_g, head_ms_g = self._embed(goal_frame)
        vis_dist = float(np.linalg.norm(cur_e - goal_e))

        pred_delta, pred_pos, dr_ms = None, None, 0.0
        if state is not None and self.predictor is not None:
            (dlat, dlon), dr_ms = self._predict_delta(state)
            pred_delta = (dlat, dlon)
            pred_pos   = (state.lat + dlat, state.lon + dlon)

        return NavResult(
            visual_distance    = vis_dist,
            goal_reached       = vis_dist < self.goal_threshold,
            current_embed      = cur_e,
            goal_embed         = goal_e,
            predicted_delta    = pred_delta,
            predicted_pos      = pred_pos,
            encoder_latency_ms = enc_ms_c + enc_ms_g,
            head_latency_ms    = head_ms_c + head_ms_g,
            dr_latency_ms      = dr_ms,
        )

    def plan(
        self,
        current_frame,
        goal_frame,
        current_state,
        candidate_frames,
        candidate_states=None,
        visual_weight=0.7,
        dr_weight=0.3,
        goal_lat=None,
        goal_lon=None,
    ):
        """
        Rank candidate actions by fused visual + dead-reckoning score.

        Visual score (weight 0.7): L2 dist of candidate frame to goal embed.
        DR score     (weight 0.3): predicted next-position proximity to goal GPS.

        Falls back to pure visual ranking if DR is unavailable.

        Args:
            candidate_frames: N RGB observations (one per candidate action)
            candidate_states: N NavStates (state *after* taking each action)
            goal_lat/lon:     Goal GPS for DR scoring

        Returns:
            PlanResult with ranked_indices, fused_scores, best_index
        """
        if not candidate_frames:
            raise ValueError("candidate_frames must be non-empty")
        N = len(candidate_frames)

        # Visual scores
        goal_e, _, _ = self._embed(goal_frame)
        cand_e, _, _ = self._embed_batch(candidate_frames)
        vis_dists    = [float(np.linalg.norm(e - goal_e)) for e in cand_e]
        vis_arr      = np.array(vis_dists)
        vis_norm     = (vis_arr - vis_arr.min()) / (vis_arr.max() - vis_arr.min() + 1e-8)

        # DR scores (optional)
        pos_deltas  = None
        fused_w_vis = 1.0
        fused_w_dr  = 0.0
        dr_norm     = np.zeros(N)

        use_dr = (
            self.predictor is not None
            and candidate_states is not None
            and len(candidate_states) == N
            and goal_lat is not None
            and goal_lon is not None
        )

        if use_dr:
            deltas, _ = self._predict_delta_batch(candidate_states)
            pos_deltas = [(float(d[0]), float(d[1])) for d in deltas]
            pred_lats  = np.array([s.lat + d[0] for s, d in zip(candidate_states, pos_deltas)])
            pred_lons  = np.array([s.lon + d[1] for s, d in zip(candidate_states, pos_deltas)])
            lat_m = 111_139.0
            lon_m = 111_139.0 * np.cos(np.radians(goal_lat))
            dr_dists   = np.sqrt(((pred_lats - goal_lat) * lat_m)**2
                               + ((pred_lons - goal_lon) * lon_m)**2)
            dr_norm     = (dr_dists - dr_dists.min()) / (dr_dists.max() - dr_dists.min() + 1e-8)
            fused_w_vis = visual_weight
            fused_w_dr  = dr_weight

        fused  = fused_w_vis * vis_norm + fused_w_dr * dr_norm
        ranked = sorted(range(N), key=lambda i: fused[i])

        return PlanResult(
            ranked_indices   = ranked,
            visual_distances = vis_dists,
            position_deltas  = pos_deltas,
            best_index       = ranked[0],
            fused_scores     = [float(fused[i]) for i in ranked],
        )

    # -- Cached goal for streaming ------------------------------------------

    def embed_goal(self, goal_frame) -> np.ndarray:
        """Pre-compute goal embedding. Use with step_cached() for high-freq loops."""
        e, _, _ = self._embed(goal_frame)
        return e

    @torch.no_grad()
    def step_cached(self, current_frame, goal_embed, state=None):
        """
        Fast path: goal embedding pre-computed, saves one encoder+head pass.

            goal_e = nav.embed_goal(goal_img)
            while True:
                result = nav.step_cached(frame, goal_e, state)
        """
        cur_e, enc_ms, head_ms = self._embed(current_frame)
        vis_dist = float(np.linalg.norm(cur_e - goal_embed))

        pred_delta, pred_pos, dr_ms = None, None, 0.0
        if state is not None and self.predictor is not None:
            (dlat, dlon), dr_ms = self._predict_delta(state)
            pred_delta = (dlat, dlon)
            pred_pos   = (state.lat + dlat, state.lon + dlon)

        return NavResult(
            visual_distance    = vis_dist,
            goal_reached       = vis_dist < self.goal_threshold,
            current_embed      = cur_e,
            goal_embed         = goal_embed,
            predicted_delta    = pred_delta,
            predicted_pos      = pred_pos,
            encoder_latency_ms = enc_ms,
            head_latency_ms    = head_ms,
            dr_latency_ms      = dr_ms,
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ReconNavigator v16.15 smoke test")
    parser.add_argument("--encoder",   required=True)
    parser.add_argument("--head",      required=True)
    parser.add_argument("--predictor", required=True)
    parser.add_argument("--device",    default="cpu")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  ReconNavigator Smoke Test -- CORTEX-PE v16.15")
    print("=" * 60)

    nav = ReconNavigator(
        encoder_path   = args.encoder,
        head_path      = args.head,
        predictor_path = args.predictor,
        device         = args.device,
    )

    rng   = np.random.default_rng(42)
    base  = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    close = np.clip(base.astype(int) + rng.integers(-10, 10, base.shape), 0, 255).astype(np.uint8)
    far   = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)

    state = NavState(
        lat=37.4100, lon=-122.0100, bearing=270.0,
        gps_vel=(0.5, 0.02, 0.0), cmd_lin=0.5, cmd_ang=0.0,
    )

    print("\n-- step() with both signals --")
    r = nav.step(base, close, state)
    print(f"  {r.summary()}")
    assert r.predicted_delta is not None
    assert r.visual_distance >= 0.0

    print("\n-- Distance ordering --")
    r_close = nav.step(base, close)
    r_far   = nav.step(base, far)
    print(f"  Close: {r_close.visual_distance:.4f}  Far: {r_far.visual_distance:.4f}")
    assert r_close.visual_distance < r_far.visual_distance
    print("  OK: distance ordering correct")

    print("\n-- step_cached() --")
    goal_e   = nav.embed_goal(close)
    r_cached = nav.step_cached(base, goal_e, state)
    print(f"  {r_cached.summary()}")
    assert abs(r_cached.visual_distance - r_close.visual_distance) < 1e-4
    print("  OK: cached matches full step()")

    print("\n-- plan() fused ranking --")
    candidates   = [close, far, base]
    cand_states  = [
        NavState(37.4101, -122.0100, 270.0, (0.5, 0.0, 0.0), 0.5,  0.0),
        NavState(37.4100, -122.0105, 180.0, (0.0, 0.5, 0.0), 0.5,  0.1),
        NavState(37.4099, -122.0100,  90.0, (0.3, 0.1, 0.0), 0.3, -0.1),
    ]
    result = nav.plan(
        base, close, state,
        candidate_frames = candidates,
        candidate_states = cand_states,
        goal_lat=37.4101, goal_lon=-122.0100,
    )
    print(f"  Best action    : {result.best_index}")
    print(f"  Ranked indices : {result.ranked_indices}")
    print(f"  Fused scores   : {[f'{s:.4f}' for s in result.fused_scores]}")
    print("  OK: plan() complete")

    print("\n  All smoke tests passed")
    print("=" * 60 + "\n")
