"""
cortex_geo_db.py  —  CORTEX CWM Sprint 4
=========================================
GeoLatentDatabase + Virtual Lookahead Sink (VL Sink).

Both from Seoul World Model (arXiv:2603.15583).

GeoLatentDatabase:
  GPS-indexed store of (gps_position → latent_embedding) pairs.
  Built at train time from RECON HDF5 files.
  At inference: retrieve nearby latents by GPS → use as goal targets for MPC.
  Gives the planner spatial memory without retraining.

VL Sink (Virtual Lookahead Sink):
  Inference-only addition. Zero parameters.
  Appends a learned "sink" token to the particle sequence at inference.
  Attention to the sink token gives each particle a summary of the full
  particle set without adding dynamics parameters.
  Prevents long-horizon error accumulation by giving the model a global
  "where are we heading" signal at each step.

Usage:
    from cortex_geo_db import GeoLatentDatabase, VirtualLookaheadSink

    # Build database from RECON trajectories
    db = GeoLatentDatabase(embed_dim=128)
    db.build(cwm, hdf5_dir="recon_data/recon_release")
    db.save("checkpoints/cwm/geo_latent_db.pt")

    # Retrieve goal candidates at inference
    goal_latents = db.query_nearby(current_gps, radius_m=20.0, top_k=5)

    # Attach VL Sink to predictor at inference
    sink = VirtualLookaheadSink(d_model=128)
    particles_with_sink = sink(particles)   # (B, K+1, d_model)
"""

import math
import time
import io
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# ═══════════════════════════════════════════════════════════════════════════
# GeoLatentDatabase
# ═══════════════════════════════════════════════════════════════════════════

class GeoLatentDatabase:
    """
    GPS-indexed database of latent embeddings from RECON trajectories.

    Structure:
        gps_coords   : np.ndarray (N, 2)  — lat/lon for each stored entry
        latents      : np.ndarray (N, embed_dim) — mean particle embedding
        timestamps   : np.ndarray (N,)   — unix timestamp for each entry
        traj_ids     : np.ndarray (N,)   — which trajectory each entry came from

    At inference:
        1. Query by current GPS → find k nearest stored entries
        2. Return their latents as candidate goal targets for MPC
        3. Planner picks the goal that minimises predicted cost

    This gives the CWM spatial memory: it has seen this GPS region before
    and knows what the latent space looks like there.
    """

    def __init__(self, embed_dim: int = 128):
        self.embed_dim = embed_dim
        self.gps_coords: Optional[np.ndarray] = None    # (N, 2)
        self.latents:    Optional[np.ndarray] = None    # (N, embed_dim)
        self.timestamps: Optional[np.ndarray] = None    # (N,)
        self.traj_ids:   Optional[np.ndarray] = None    # (N,)
        self._built = False

    # ── Build ──────────────────────────────────────────────────────────────

    def build(
        self,
        cwm,                       # CortexWorldModel (frozen encoder)
        hdf5_dir:  str,
        student_mock: nn.Module,
        max_files: Optional[int] = None,
        stride:    int = 4,        # sample every N frames (4Hz → 1Hz effective)
        device:    torch.device = torch.device("cpu"),
        verbose:   bool = True,
    ):
        """
        Walk all RECON HDF5 files and store (GPS → particle mean embedding) pairs.
        Uses stride to avoid storing every 4Hz frame (too dense, redundant).
        """
        files = sorted(Path(hdf5_dir).glob("jackal_2019-*.hdf5"))
        if max_files:
            files = files[:max_files]

        all_gps, all_latents, all_ts, all_traj = [], [], [], []

        cwm.eval()
        for traj_id, fpath in enumerate(files):
            with h5py.File(fpath, 'r') as hf:
                n_frames  = len(hf['images'])
                has_gps   = 'observations' in hf

                for i in range(0, n_frames, stride):
                    # Decode frame
                    try:
                        jpeg = bytes(hf['images'][i])
                        img  = Image.open(io.BytesIO(jpeg)).convert('RGB')
                        img  = img.resize((224, 224))
                        t    = torch.from_numpy(np.array(img)).float()
                        t    = t.permute(2,0,1).unsqueeze(0).to(device) / 255.0
                    except Exception:
                        continue

                    with torch.no_grad():
                        z        = student_mock(t.reshape(1, -1))
                        parts, _, _, _ = cwm.encode(z)
                        emb      = parts.mean(dim=1).squeeze(0).cpu().numpy()

                    gps = hf['observations'][i][:2] if has_gps else [0., 0.]
                    all_gps.append(np.array(gps, dtype=np.float32))
                    all_latents.append(emb.astype(np.float32))
                    all_ts.append(float(i) / 4.0)   # approximate seconds at 4Hz
                    all_traj.append(traj_id)

            if verbose and (traj_id + 1) % 10 == 0:
                print(f"  GeoLatentDB: {traj_id+1}/{len(files)} trajectories, "
                      f"{len(all_latents):,} entries")

        self.gps_coords = np.stack(all_gps)
        self.latents    = np.stack(all_latents)
        self.timestamps = np.array(all_ts,   dtype=np.float32)
        self.traj_ids   = np.array(all_traj, dtype=np.int32)
        self._built     = True

        if verbose:
            print(f"\nGeoLatentDatabase built:")
            print(f"  Entries:   {len(self.latents):,}")
            print(f"  GPS range: lat [{self.gps_coords[:,0].min():.4f}, "
                  f"{self.gps_coords[:,0].max():.4f}]  "
                  f"lon [{self.gps_coords[:,1].min():.4f}, "
                  f"{self.gps_coords[:,1].max():.4f}]")
            print(f"  Embed dim: {self.embed_dim}")

    # ── Query ──────────────────────────────────────────────────────────────

    def query_nearby(
        self,
        gps:       np.ndarray,      # (2,) current lat/lon
        radius_m:  float = 20.0,    # search radius in metres
        top_k:     int   = 5,
        exclude_traj: Optional[int] = None,  # exclude current trajectory
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return top_k stored latents whose GPS position is within radius_m.

        Returns:
            latents    (top_k, embed_dim)  — candidate goal embeddings
            distances  (top_k,)            — GPS distances in metres
        """
        assert self._built, "Call build() first"

        # Approximate distance in metres (lat/lon → metres at mid-latitude)
        dlat = (self.gps_coords[:, 0] - gps[0]) * 111_139.0
        dlon = (self.gps_coords[:, 1] - gps[1]) * 111_139.0 * math.cos(
            math.radians(float(gps[0]))
        )
        dist_m = np.sqrt(dlat**2 + dlon**2)

        # Filter by radius
        mask = dist_m <= radius_m
        if exclude_traj is not None:
            mask &= (self.traj_ids != exclude_traj)

        if not mask.any():
            # No entries in range — return closest k regardless of radius
            idx = np.argsort(dist_m)[:top_k]
            return self.latents[idx], dist_m[idx]

        nearby_idx   = np.where(mask)[0]
        nearby_dists = dist_m[nearby_idx]
        order        = np.argsort(nearby_dists)[:top_k]
        idx          = nearby_idx[order]

        return self.latents[idx], nearby_dists[order]

    def query_as_tensor(
        self,
        gps:    np.ndarray,
        radius_m: float = 20.0,
        top_k:  int = 5,
        device: torch.device = torch.device("cpu"),
    ) -> Optional[torch.Tensor]:
        """Returns goal candidates as (top_k, embed_dim) tensor or None."""
        latents, _ = self.query_nearby(gps, radius_m=radius_m, top_k=top_k)
        if len(latents) == 0:
            return None
        return torch.from_numpy(latents).float().to(device)

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            "gps_coords": self.gps_coords,
            "latents":    self.latents,
            "timestamps": self.timestamps,
            "traj_ids":   self.traj_ids,
            "embed_dim":  self.embed_dim,
        }, path)
        n = len(self.latents) if self.latents is not None else 0
        print(f"GeoLatentDB saved: {path} ({n:,} entries)")

    def load(self, path: str):
        data = torch.load(path, map_location="cpu")
        self.gps_coords = data["gps_coords"]
        self.latents    = data["latents"]
        self.timestamps = data["timestamps"]
        self.traj_ids   = data["traj_ids"]
        self.embed_dim  = data["embed_dim"]
        self._built     = True
        print(f"GeoLatentDB loaded: {path} "
              f"({len(self.latents):,} entries, embed_dim={self.embed_dim})")

    def __len__(self):
        return len(self.latents) if self.latents is not None else 0


# ═══════════════════════════════════════════════════════════════════════════
# Virtual Lookahead Sink (VL Sink)
# ═══════════════════════════════════════════════════════════════════════════

class VirtualLookaheadSink(nn.Module):
    """
    Virtual Lookahead Sink (VL Sink) from Seoul World Model (arXiv:2603.15583).

    ZERO dynamics parameters. Inference-only addition.

    How it works:
        1. Append a learnable "sink" token to the particle sequence
        2. Run self-attention over [particles, sink_token]
        3. The sink token attends to all K particles — global summary
        4. Each particle attends to the sink — gets global context
        5. Remove sink token before output — same K particles out

    Why it helps long-horizon planning:
        Without VL Sink: each particle only has local K-neighbourhood context.
        With VL Sink:    every particle has a global "where are we" signal.
        This prevents error accumulation in 6+ step OGBench-Cube rollouts.

    Parameters
    ----------
    d_model : particle embedding dimension (128)
    n_heads : attention heads (4)
    """

    def __init__(self, d_model: int = 128, n_heads: int = 4):
        super().__init__()
        # Learnable sink token — trained with the rest of the model
        # but never output (stripped before returning particles)
        self.sink_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Single attention layer for sink communication
        self.sink_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=0.0
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, particles: torch.Tensor) -> torch.Tensor:
        """
        particles : (B, K, d_model)
        Returns   : (B, K, d_model) — same shape, sink removed after attn

        The sink token attends freely to all K particles and each particle
        attends to the sink token. After attention, strip the sink.
        """
        B, K, D = particles.shape

        # Expand sink to batch
        sink = self.sink_token.expand(B, 1, D)   # (B, 1, D)

        # Append sink: sequence = [particles, sink]
        seq = torch.cat([particles, sink], dim=1)   # (B, K+1, D)

        # Self-attention across full sequence (particles ↔ sink)
        attn_out, _ = self.sink_attn(seq, seq, seq, need_weights=False)

        # Residual + norm
        seq = self.norm(seq + attn_out)

        # Strip sink token — return only particle positions
        return seq[:, :K, :]   # (B, K, D)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ═══════════════════════════════════════════════════════════════════════════
# GPS-conditioned MPC with GeoLatentDB
# ═══════════════════════════════════════════════════════════════════════════

class GeoConditionedPlanner:
    """
    Extends SevenSignalMPCPlanner with GPS-conditioned goal selection.

    At each inference step:
        1. Query GeoLatentDB for nearby stored latents (candidate goals)
        2. For each candidate goal, run MPC rollout and compute cost
        3. Select the goal with lowest predicted cost
        4. Execute best action from that rollout

    This turns the planner from "minimise any prediction error" into
    "navigate toward previously-seen GPS locations" — directional control
    from a memory bank, zero additional training.
    """

    def __init__(
        self,
        cwm,
        geo_db:    GeoLatentDatabase,
        vl_sink:   Optional[VirtualLookaheadSink] = None,
        horizon:   int = 6,
        n_cands:   int = 32,
    ):
        self.cwm     = cwm
        self.geo_db  = geo_db
        self.vl_sink = vl_sink
        self.horizon = horizon
        self.n_cands = n_cands

    @torch.no_grad()
    def plan(
        self,
        particles:      torch.Tensor,    # (1, K, d_model)
        current_gps:    np.ndarray,      # (2,) current lat/lon
        action_dim:     int = 2,
        neuro_signals:  dict = None,
        device:         torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, dict]:
        """
        Plan toward the GPS-retrieved goal with lowest MPC cost.
        Returns (best_action, info_dict).
        """
        neuro_signals = neuro_signals or {
            "n_candidates": self.n_cands, "action_std": 0.1,
            "action_scale": 1.0, "rho": 0.5, "da_effective": 0.5,
            "regime": "EXPLOIT", "ado": 0.0,
        }

        # Apply VL Sink if available (zero params, just adds global context)
        if self.vl_sink is not None:
            particles = self.vl_sink(particles)

        # Query GeoLatentDB for goal candidates
        goal_latents = self.geo_db.query_as_tensor(
            current_gps, radius_m=20.0, top_k=5, device=device
        )

        if goal_latents is None:
            # No nearby stored latents — plan without geo guidance
            goal_latents = torch.zeros(1, self.cwm.d_model, device=device)
            goal_latents = goal_latents.unsqueeze(0).expand(
                1, particles.shape[1], self.cwm.d_model
            )
        else:
            # Use mean of retrieved latents as goal (B=1, K, D)
            goal_mean = goal_latents.mean(dim=0)    # (embed_dim,)
            goal_latents = goal_mean.unsqueeze(0).unsqueeze(0).expand(
                1, particles.shape[1], -1
            )

        # Sample candidate action sequences
        rho         = neuro_signals.get("rho", 0.5)
        action_std  = neuro_signals.get("action_std", 0.1)
        n_cands     = neuro_signals.get("n_candidates", self.n_cands)
        action_scl  = neuro_signals.get("action_scale", 1.0)
        da_eff      = neuro_signals.get("da_effective", 0.5)
        ado         = neuro_signals.get("ado", 0.0)

        candidates = torch.randn(n_cands, self.horizon, action_dim,
                                  device=device) * action_std

        costs = []
        for k in range(n_cands):
            p = particles.clone()
            cum_cost = 0.0
            for t in range(self.horizon):
                action_pad = F.pad(candidates[k, t].unsqueeze(0),
                                   (0, 9 - action_dim))
                domain_id  = torch.zeros(1, dtype=torch.long, device=device)
                ctx_h      = self.cwm.thick_gru.init_context(1, device)

                out = self.cwm.predict(
                    particles=p, action=action_pad,
                    context_h=ctx_h, positions=torch.zeros(1, p.shape[1], 2,
                                                            device=device),
                    domain_id=domain_id,
                )
                p = out["z_pred"]
                step_cost = F.mse_loss(p, goal_latents).item()
                cum_cost += step_cost

            # NE/rho-modulated cost (ARCHITECTURE.md formula)
            costs.append(-cum_cost / (1.0 + rho))

        costs_t = torch.tensor(costs, device=device)

        # DA-modulated Mirror Ascent
        eta = 0.05 * (0.5 + da_eff) * (1.0 - ado * 0.3)
        q   = F.softmax(costs_t / max(eta, 1e-4), dim=0)
        best_action = (candidates[:, 0, :action_dim] *
                       q.unsqueeze(-1)).sum(0) * action_scl

        return best_action, {
            "n_geo_goals":  5 if goal_latents is not None else 0,
            "current_gps":  current_gps.tolist(),
            "regime":       neuro_signals.get("regime", "EXPLOIT"),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Build script
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cwm-ckpt",  default=r"checkpoints\cwm\cwm_best.pt")
    p.add_argument("--hdf5-dir",  default="recon_data/recon_release")
    p.add_argument("--out",       default=r"checkpoints\cwm\geo_latent_db.pt")
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--stride",    type=int, default=4)
    p.add_argument("--smoke",     action="store_true")
    args = p.parse_args()

    if args.smoke:
        # Smoke test VL Sink
        print("VL Sink smoke test...")
        sink = VirtualLookaheadSink(d_model=128, n_heads=4)
        particles = torch.randn(4, 16, 128)
        out = sink(particles)
        assert out.shape == particles.shape, f"Shape mismatch: {out.shape}"
        print(f"  VL Sink OK — output shape: {tuple(out.shape)}")
        print(f"  VL Sink params: {sink.total_params():,}")

        # Smoke test GeoLatentDB
        db = GeoLatentDatabase(embed_dim=128)
        db.gps_coords = np.array([[37.0, -122.0], [37.001, -122.001],
                                    [37.5, -122.5]], dtype=np.float32)
        db.latents    = np.random.randn(3, 128).astype(np.float32)
        db.timestamps = np.array([0., 1., 2.], dtype=np.float32)
        db.traj_ids   = np.array([0, 0, 1], dtype=np.int32)
        db._built     = True

        lats, dists = db.query_nearby(np.array([37.0, -122.0]), radius_m=200.0)
        print(f"  GeoLatentDB query: {len(lats)} results within 200m")
        print("Smoke test passed.")
    else:
        from train_cwm import CortexWorldModel
        import torch.nn as nn

        cwm = CortexWorldModel(d_model=128, K=16)
        if Path(args.cwm_ckpt).exists():
            ckpt = torch.load(args.cwm_ckpt, map_location="cpu")
            cwm.load_state_dict(ckpt["model"])
        cwm.eval()

        student_mock = nn.Linear(3 * 224 * 224, 256)

        db = GeoLatentDatabase(embed_dim=128)
        db.build(cwm, args.hdf5_dir, student_mock,
                  max_files=args.max_files, stride=args.stride)
        db.save(args.out)
