"""
proprioceptive_encoder.py  --  NeMo-WM Sprint 6
================================================
Proprioceptive second encoding pathway for biological dissociation.

Motivation (from aphasia ablation, Sprint 5):
    Zeroing the VLM gate collapses AUROC to 0.500 at ALL temporal horizons
    (k=1 through k=16). The particle encoder is fully VLM-grounded -- it has
    no autonomous representation of physical quantities.

    Fedorenko's aphasia finding: language networks destroyed -> complex
    reasoning PRESERVED (biological dissociation).
    NeMo-WM finding: VLM gate destroyed -> world modelling ELIMINATED.

    To achieve dissociation, particles must learn physical structure from a
    pathway that does NOT depend on VLM embeddings. This file implements that
    pathway: a proprioceptive encoder that fuses velocity, GPS displacement,
    angular velocity, and contact events into a 128-D embedding that can be
    combined with (or substituted for) the VLM embedding in the particle encoder.

Architecture:
    ProprioceptiveEncoder:
        Inputs:  linear_vel (1,), angular_vel (1,), gps_disp (2,),
                 contact (1,), heading (2,) [sin/cos]
        Output:  z_proprio (128,) -- unit-normalised, same space as VLM embedding

    FusionGate:
        Fuses z_vlm and z_proprio into z_fused (128,) via learned alpha gate.
        alpha=1.0 -> full VLM (current system)
        alpha=0.0 -> full proprio (dissociated system -- Sprint 6 target)
        alpha=learned -> hybrid

    AblationWrapper:
        Drop-in replacement for the VLM pathway. Accepts proprio signals and
        returns z_proprio as if it were z_vlm. Used to test dissociation
        in eval_recon_auroc.py via --proprio-ablation flag.

Usage:
    # Training
    from proprioceptive_encoder import ProprioceptiveEncoder, FusionGate
    proprio_enc = ProprioceptiveEncoder()
    fusion      = FusionGate()

    z_vlm    = vlm_gate.encode(frame)           # (128,) -- current pathway
    z_proprio = proprio_enc(vel, ang, gps, contact, heading)  # (128,)
    z_fused  = fusion(z_vlm, z_proprio)         # (128,) -- combined

    # Dissociation test (aphasia-equivalent for VLM path)
    z_fused_no_vlm = fusion(torch.zeros(128), z_proprio)  # proprio only

Run tests:
    python proprioceptive_encoder.py --test
    python proprioceptive_encoder.py --test --dissociation-eval
"""

import argparse
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Config
# ===========================================================================

@dataclass
class ProprioConfig:
    """
    Configuration for proprioceptive encoder.

    Input signals from RECON HDF5:
        linear_vel   : commands/linear_velocity     (1,)   m/s
        angular_vel  : commands/angular_velocity    (1,)   rad/s
        gps_disp     : gps/latlong delta            (2,)   degrees -> metres
        contact      : synthetic from vel>0.3       (1,)   binary proxy
        heading      : from accumulated angular_vel (2,)   [sin, cos]

    d_model: must match particle embedding dim (128 in production).
    """
    d_in:          int   = 7       # 1+1+2+1+2
    d_hidden:      int   = 64
    d_model:       int   = 128
    n_layers:      int   = 3
    dropout:       float = 0.0     # off at inference; on during training
    use_layernorm: bool  = True

    # Fusion gate
    fusion_init_alpha: float = 0.5   # start halfway between VLM and proprio
    fusion_learnable:  bool  = True  # False = fixed alpha (ablation)


# ===========================================================================
# Proprioceptive encoder
# ===========================================================================

class ProprioceptiveEncoder(nn.Module):
    """
    Encodes physical proprioceptive signals into a 128-D embedding.

    The embedding lives in the same unit-normalised space as the VLM
    embedding (z_vlm), enabling direct combination via FusionGate.

    Inputs (all floats, normalised to ~[-1, 1] range):
        linear_vel  (B, 1)   -- commanded linear velocity
        angular_vel (B, 1)   -- commanded angular velocity
        gps_disp    (B, 2)   -- GPS displacement (north, east) in metres
        contact     (B, 1)   -- contact / non-zero velocity indicator
        heading     (B, 2)   -- robot heading as [sin(theta), cos(theta)]

    Output:
        z_proprio (B, d_model) -- unit-normalised proprioceptive embedding
    """

    def __init__(self, config: Optional[ProprioConfig] = None):
        super().__init__()
        self.cfg = config or ProprioConfig()
        cfg = self.cfg

        layers = []
        d_in = cfg.d_in
        for i in range(cfg.n_layers):
            d_out = cfg.d_hidden if i < cfg.n_layers - 1 else cfg.d_model
            layers.append(nn.Linear(d_in, d_out))
            if cfg.use_layernorm and i < cfg.n_layers - 1:
                layers.append(nn.LayerNorm(d_out))
            if i < cfg.n_layers - 1:
                layers.append(nn.GELU())
                if cfg.dropout > 0:
                    layers.append(nn.Dropout(cfg.dropout))
            d_in = d_out

        self.net = nn.Sequential(*layers)

        # Input normalisation statistics (set from RECON dataset)
        # These are reasonable defaults; tune from data stats
        self.register_buffer("vel_mean",  torch.tensor([0.4]))    # m/s
        self.register_buffer("vel_std",   torch.tensor([0.3]))
        self.register_buffer("ang_mean",  torch.tensor([0.0]))    # rad/s
        self.register_buffer("ang_std",   torch.tensor([0.5]))
        self.register_buffer("gps_mean",  torch.zeros(2))         # metres
        self.register_buffer("gps_std",   torch.tensor([2.0, 2.0]))

    def normalise(
        self,
        linear_vel:  torch.Tensor,   # (B, 1)
        angular_vel: torch.Tensor,   # (B, 1)
        gps_disp:    torch.Tensor,   # (B, 2)
        contact:     torch.Tensor,   # (B, 1)
        heading:     torch.Tensor,   # (B, 2)  already in [-1,1] via sin/cos
    ) -> torch.Tensor:
        vel_n = (linear_vel  - self.vel_mean) / (self.vel_std  + 1e-6)
        ang_n = (angular_vel - self.ang_mean) / (self.ang_std  + 1e-6)
        gps_n = (gps_disp    - self.gps_mean) / (self.gps_std  + 1e-6)
        # contact and heading are already normalised / bounded
        return torch.cat([vel_n, ang_n, gps_n, contact, heading], dim=-1)

    def forward(
        self,
        linear_vel:  torch.Tensor,
        angular_vel: torch.Tensor,
        gps_disp:    torch.Tensor,
        contact:     torch.Tensor,
        heading:     torch.Tensor,
    ) -> torch.Tensor:
        x = self.normalise(linear_vel, angular_vel, gps_disp, contact, heading)
        z = self.net(x)
        return F.normalize(z, dim=-1)     # unit-normalised, same space as z_vlm

    @staticmethod
    def from_recon_batch(batch: dict) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """
        Extract proprioceptive signals from a RECON HDF5 batch dict.
        Computes heading from accumulated angular velocity (no magnetometer).

        batch keys (from RECON HDF5):
            commands/linear_velocity    (T,)
            commands/angular_velocity   (T,)
            gps/latlong                 (T, 2)
        """
        B = batch["commands/linear_velocity"].shape[0]

        vel  = batch["commands/linear_velocity"].unsqueeze(-1).float()   # (B, 1)
        ang  = batch["commands/angular_velocity"].unsqueeze(-1).float()  # (B, 1)

        # GPS displacement: delta from first frame, converted to metres
        # 1 degree lat ~ 111,000m; 1 degree lon ~ 111,000 * cos(lat)m
        gps_raw = batch["gps/latlong"].float()          # (B, 2)
        gps_origin = gps_raw[0:1].expand_as(gps_raw)
        gps_delta = gps_raw - gps_origin                # (B, 2) in degrees
        lat_ref = gps_raw[0, 0]
        metres_per_deg_lat = 111000.0
        metres_per_deg_lon = 111000.0 * math.cos(math.radians(float(lat_ref)))
        gps_disp = torch.stack([
            gps_delta[:, 0] * metres_per_deg_lat,
            gps_delta[:, 1] * metres_per_deg_lon,
        ], dim=-1)                                      # (B, 2) in metres

        # Heading: accumulated angular velocity integration
        heading_angle = torch.cumsum(ang.squeeze(-1), dim=0) * (1.0 / 4.0)  # 4Hz
        heading = torch.stack([
            torch.sin(heading_angle),
            torch.cos(heading_angle),
        ], dim=-1)                                      # (B, 2)

        # Contact proxy: linear velocity > 0.3 m/s (moving forward)
        contact = (vel.squeeze(-1) > 0.3).float().unsqueeze(-1)  # (B, 1)

        return vel, ang, gps_disp, contact, heading


# ===========================================================================
# Fusion gate
# ===========================================================================

class FusionGate(nn.Module):
    """
    Fuses VLM embedding and proprioceptive embedding into a single embedding.

    alpha=1.0: full VLM   (current NeMo-WM behaviour)
    alpha=0.0: full proprio (dissociated -- Sprint 6 target)
    alpha=learned: hybrid, converging toward whatever drives lower loss

    The alpha is applied per-dimension if learnable (allows the model to
    discover which dimensions are better driven by VLM vs proprio).
    """

    def __init__(self, config: Optional[ProprioConfig] = None):
        super().__init__()
        cfg = config or ProprioConfig()
        self.d_model = cfg.d_model

        if cfg.fusion_learnable:
            # Per-dimension alpha, initialised to fusion_init_alpha
            init = torch.full((cfg.d_model,), cfg.fusion_init_alpha)
            self.alpha_logit = nn.Parameter(
                torch.logit(init.clamp(0.01, 0.99))
            )
        else:
            self.register_buffer(
                "alpha_fixed",
                torch.full((cfg.d_model,), cfg.fusion_init_alpha)
            )
        self.learnable = cfg.fusion_learnable

    @property
    def alpha(self) -> torch.Tensor:
        """Returns per-dimension alpha in [0, 1]."""
        if self.learnable:
            return torch.sigmoid(self.alpha_logit)
        return self.alpha_fixed

    def forward(
        self,
        z_vlm:    torch.Tensor,   # (B, d_model) -- VLM embedding (or zeros for ablation)
        z_proprio: torch.Tensor,  # (B, d_model) -- proprioceptive embedding
    ) -> torch.Tensor:
        """Returns fused embedding, unit-normalised."""
        alpha = self.alpha
        z = alpha * z_vlm + (1.0 - alpha) * z_proprio
        return F.normalize(z, dim=-1)

    def alpha_stats(self) -> dict:
        """Diagnostic: mean/min/max of alpha across dimensions."""
        a = self.alpha.detach()
        return {
            "alpha_mean": float(a.mean()),
            "alpha_min":  float(a.min()),
            "alpha_max":  float(a.max()),
            "vlm_dims":   int((a > 0.7).sum()),    # dims dominated by VLM
            "prop_dims":  int((a < 0.3).sum()),    # dims dominated by proprio
            "mixed_dims": int(((a >= 0.3) & (a <= 0.7)).sum()),
        }


# ===========================================================================
# Dissociation evaluator (Sprint 6 test harness)
# ===========================================================================

class DissociationEvaluator:
    """
    Measures how much AUROC is preserved when VLM is zeroed but
    proprioceptive encoding is preserved.

    Three conditions (mirrors the aphasia ablation from Sprint 4/5):
        Full:       z_fused = FusionGate(z_vlm, z_proprio)
        No VLM:     z_fused = FusionGate(zeros, z_proprio)  -- target: AUROC > 0.50
        No proprio: z_fused = FusionGate(z_vlm, zeros)      -- should be < full

    If "No VLM" AUROC > 0.50, dissociation has begun. This is the Sprint 6
    success criterion.
    """

    def __init__(self, proprio_enc: ProprioceptiveEncoder, fusion: FusionGate):
        self.proprio_enc = proprio_enc
        self.fusion      = fusion

    @torch.no_grad()
    def eval_pair(
        self,
        z_vlm_a:     torch.Tensor,   # (d_model,) VLM embedding, frame A
        z_vlm_b:     torch.Tensor,   # (d_model,) VLM embedding, frame B
        proprio_a:   Tuple,          # (vel, ang, gps, contact, heading) for A
        proprio_b:   Tuple,          # same for B
    ) -> dict:
        """
        Compute distances under all three conditions for a pair (A, B).
        Returns dict with distances per condition.
        """
        d = self.proprio_enc.cfg.d_model

        def _encode(z_vlm, proprio, ablate_vlm=False, ablate_proprio=False):
            z_v = torch.zeros(d) if ablate_vlm   else z_vlm
            z_p = self.proprio_enc(*proprio)
            z_p = torch.zeros(d) if ablate_proprio else z_p
            return self.fusion(z_v.unsqueeze(0), z_p.unsqueeze(0)).squeeze(0)

        z_full_a   = _encode(z_vlm_a, proprio_a)
        z_full_b   = _encode(z_vlm_b, proprio_b)
        z_novlm_a  = _encode(z_vlm_a, proprio_a, ablate_vlm=True)
        z_novlm_b  = _encode(z_vlm_b, proprio_b, ablate_vlm=True)
        z_noprop_a = _encode(z_vlm_a, proprio_a, ablate_proprio=True)
        z_noprop_b = _encode(z_vlm_b, proprio_b, ablate_proprio=True)

        return {
            "full":       float(torch.norm(z_full_a   - z_full_b)),
            "no_vlm":     float(torch.norm(z_novlm_a  - z_novlm_b)),
            "no_proprio": float(torch.norm(z_noprop_a - z_noprop_b)),
        }


# ===========================================================================
# Self-test
# ===========================================================================

def run_tests(dissociation_eval: bool = False):
    print("proprioceptive_encoder.py self-test...\n")

    cfg = ProprioConfig(d_model=128, n_layers=3)
    enc = ProprioceptiveEncoder(cfg)
    enc.eval()

    B = 4
    vel     = torch.rand(B, 1) * 0.8
    ang     = torch.randn(B, 1) * 0.3
    gps     = torch.randn(B, 2) * 2.0
    contact = (vel > 0.3).float()
    heading = torch.stack([torch.sin(torch.randn(B)), torch.cos(torch.randn(B))], dim=-1)

    z = enc(vel, ang, gps, contact, heading)

    assert z.shape == (B, 128), f"Wrong shape: {z.shape}"
    norms = z.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(B), atol=1e-5), \
        f"Not unit-normalised: {norms}"
    print(f"  ProprioceptiveEncoder: output shape {z.shape}, unit-normalised ✓")
    print(f"  z mean={float(z.mean()):.4f}, std={float(z.std()):.4f}")

    # FusionGate
    fusion = FusionGate(cfg)
    z_vlm   = F.normalize(torch.randn(B, 128), dim=-1)
    z_fused = fusion(z_vlm, z)
    assert z_fused.shape == (B, 128), f"Fusion wrong shape: {z_fused.shape}"
    norms_f = z_fused.norm(dim=-1)
    assert torch.allclose(norms_f, torch.ones(B), atol=1e-5)
    stats = fusion.alpha_stats()
    print(f"  FusionGate: output shape {z_fused.shape}, unit-normalised ✓")
    print(f"  Alpha stats: mean={stats['alpha_mean']:.3f}, "
          f"vlm_dims={stats['vlm_dims']}, prop_dims={stats['prop_dims']}, "
          f"mixed={stats['mixed_dims']}")

    # VLM ablation (aphasia-equivalent)
    z_novlm = fusion(torch.zeros(B, 128), z)
    norms_nv = z_novlm.norm(dim=-1)
    assert torch.allclose(norms_nv, torch.ones(B), atol=1e-5)
    dist_full  = float(torch.norm(z_fused[0]  - z_fused[1]))
    dist_novlm = float(torch.norm(z_novlm[0]  - z_novlm[1]))
    print(f"  VLM ablation: dist_full={dist_full:.4f}, "
          f"dist_novlm={dist_novlm:.4f} (>0 = proprio encodes something) ✓")

    # from_recon_batch interface (mock batch)
    mock_batch = {
        "commands/linear_velocity":  torch.rand(8) * 0.8,
        "commands/angular_velocity": torch.randn(8) * 0.3,
        "gps/latlong":               torch.randn(8, 2) * 0.001 + torch.tensor([37.87, -122.26]),
    }
    vel2, ang2, gps2, contact2, heading2 = ProprioceptiveEncoder.from_recon_batch(mock_batch)
    z2 = enc(vel2, ang2, gps2, contact2, heading2)
    assert z2.shape == (8, 128)
    print(f"  from_recon_batch: shape {z2.shape} ✓")

    if dissociation_eval:
        print("\n  Dissociation eval (mock -- real eval needs temporal pairs):")
        evaluator = DissociationEvaluator(enc, fusion)
        z_vlm_a = F.normalize(torch.randn(128), dim=-1)
        z_vlm_b = F.normalize(torch.randn(128), dim=-1)
        pa = (torch.rand(1,1)*0.8, torch.randn(1,1)*0.3,
              torch.randn(1,2)*2, (torch.rand(1,1)>0.3).float(),
              torch.stack([torch.sin(torch.zeros(1)), torch.cos(torch.zeros(1))], -1))
        pb = (torch.rand(1,1)*0.8, torch.randn(1,1)*0.3,
              torch.randn(1,2)*2, (torch.rand(1,1)>0.3).float(),
              torch.stack([torch.sin(torch.ones(1)*0.5), torch.cos(torch.ones(1)*0.5)], -1))
        result = evaluator.eval_pair(z_vlm_a, z_vlm_b, pa, pb)
        print(f"    full dist:       {result['full']:.4f}")
        print(f"    no_vlm dist:     {result['no_vlm']:.4f}  "
              f"({'> 0 — proprio encodes structure' if result['no_vlm'] > 0.001 else '≈ 0 — no structure yet'})")
        print(f"    no_proprio dist: {result['no_proprio']:.4f}")
        print(f"\n  Sprint 6 criterion: no_vlm AUROC > 0.50 on RECON eval pairs.")
        print(f"  If no_vlm dist separation matches pos/neg pairs -> dissociation achieved.")

    print("\n  All assertions passed.")
    print("\nNext step: wire ProprioceptiveEncoder into eval_recon_auroc.py")
    print("  --proprio-ablation flag: zero VLM, use proprio only")
    print("  --proprio-fused flag:    use FusionGate(z_vlm, z_proprio)")
    print("  Success criterion: --proprio-ablation AUROC > 0.50")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sprint 6 proprioceptive encoder")
    parser.add_argument("--test",               action="store_true")
    parser.add_argument("--dissociation-eval",  action="store_true")
    args = parser.parse_args()

    if args.test or args.dissociation_eval:
        run_tests(dissociation_eval=args.dissociation_eval)
    else:
        print("Run with --test to verify all components.")
        print("Run with --test --dissociation-eval for full dissociation harness.")
