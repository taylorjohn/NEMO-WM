"""
domain_topology.py — NeMo-WM Domain Topology Metadata (Sprint 9f)
=================================================================
Defines what kind of world each domain presents to the neuromodulator.

The core insight from PushT cross-domain training (2026-04-06):
  RECON GPS (persistent_2d)  → permanent REOBSERVE → loss 0.0743
  PushT block XY (episodic_2d) → permanent EXPLOIT → loss 0.663 stagnant

The neuromodulator treated both identically through NE (spatial grounding).
But they have fundamentally different geometry:
  - RECON GPS: robot physically moves through space, displacement accumulates
  - PushT XY: block resets each episode, no persistent map possible

This module gives each domain a topology declaration that tells the
neuromodulator how to interpret its signals, what cortisol baseline to use,
and when EXPLOIT vs REOBSERVE is the correct regime.

Usage:
    from domain_topology import TOPOLOGIES, get_topology, auto_calibrate_cortisol

    topo = get_topology("recon")
    ne_signal = topo.compute_ne_signal(gps_current, gps_prev)
    baseline  = topo.cortisol_baseline or epoch_0_loss_mean

Adding a new domain:
    See "HOW TO ADD A NEW DOMAIN" at the bottom of this file.

Author: John Taylor — github.com/taylorjohn
Date:   2026-04-06
Sprint: 9f
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ── Spatial type constants ────────────────────────────────────────────────────

SPATIAL_NONE        = "none"         # No spatial signal (audio, telemetry)
SPATIAL_EPISODIC_2D = "episodic_2d"  # 2D coordinates, reset each episode
SPATIAL_PERSISTENT_2D = "persistent_2d"  # 2D GPS, accumulates across session
SPATIAL_PERSISTENT_3D = "persistent_3d"  # 3D point cloud / volumetric
SPATIAL_GRID_2D     = "grid_2d"      # Discrete grid (MiniGrid, maze)
SPATIAL_PATCH_GRID  = "patch_grid"   # 2D patch positions (MVTec inspection)

# ── Metric constants ──────────────────────────────────────────────────────────

METRIC_NONE        = "none"
METRIC_EUCLIDEAN   = "euclidean"
METRIC_HAVERSINE   = "haversine"    # GPS lat/lon → meters
METRIC_MANHATTAN   = "manhattan"    # Grid navigation
METRIC_COSINE      = "cosine"       # Embedding similarity


# ── Topology dataclass ────────────────────────────────────────────────────────

@dataclass
class DomainTopology:
    """
    Full description of a domain's world geometry and signal structure.

    The neuromodulator uses this to:
    1. Scale NE (spatial grounding) correctly for the domain's metric
    2. Set cortisol baseline (None = auto-calibrate from epoch-0 loss)
    3. Enable/disable signals that aren't meaningful for this domain
    4. Choose correct REOBSERVE trigger sensitivity
    """

    # Identity
    name:        str
    description: str

    # ── Spatial topology ──────────────────────────────────────────────────────
    spatial_type:       str   = SPATIAL_NONE
    spatial_metric:     str   = METRIC_NONE
    spatial_persistent: bool  = False  # does map accumulate across episodes?
    spatial_dim:        int   = 0      # dimensionality (0=none, 2=2D, 3=3D)

    # Typical displacement magnitude per step (used to scale NE sensitivity)
    # RECON: ~0.5m/step at 4Hz, PushT: ~10px/step in [0,512] space
    spatial_scale_per_step: float = 0.0

    # ── Temporal properties ───────────────────────────────────────────────────
    frame_rate_hz:          float = 4.0   # observation frequency
    episode_length_typical: int   = 200   # typical steps per episode
    inter_episode_gap:      bool  = True  # does state reset between episodes?

    # ── Neuromodulator signal configuration ───────────────────────────────────
    ne_mode:         str  = "disabled"   # "gps", "pixel", "grid", "disabled"
    ach_enabled:     bool = False         # contact detection meaningful?
    ecb_enabled:     bool = True          # skill/novelty signal
    ado_enabled:     bool = True          # fatigue signal
    ei_enabled:      bool = True          # arousal/curvature

    # Cortisol: None = auto-calibrate from epoch-0 loss mean
    # Set explicitly if you know the expected loss scale
    cortisol_baseline:   Optional[float] = None
    cortisol_sensitivity: float = 1.0    # multiplier on cortisol response

    # ── Available sensor signals ──────────────────────────────────────────────
    has_velocity:         bool = False
    has_angular_velocity: bool = False
    has_contact:          bool = False
    has_audio:            bool = False
    has_depth:            bool = False

    # ── Expected learning regime ──────────────────────────────────────────────
    # "reobserve" = REOBSERVE expected (rich novel environment)
    # "exploit"   = EXPLOIT expected and correct (simple/episodic)
    # "mixed"     = both regimes expected
    expected_regime: str = "reobserve"

    # ── Notes ─────────────────────────────────────────────────────────────────
    notes: str = ""

    # ── Computed properties ───────────────────────────────────────────────────

    def compute_ne_signal(
        self,
        current_spatial:  np.ndarray,
        previous_spatial: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute NE (norepinephrine / spatial grounding) signal for this step.
        Returns displacement normalised to [0, 1] relative to typical scale.
        """
        if self.ne_mode == "disabled" or previous_spatial is None:
            return 0.0

        d = current_spatial - previous_spatial

        if self.spatial_metric == METRIC_HAVERSINE:
            # GPS lat/lon → meters
            dlat = d[0] * 111_000
            dlon = d[1] * 111_000 * np.cos(np.radians(current_spatial[0]))
            dist = float(np.sqrt(dlat**2 + dlon**2))
        elif self.spatial_metric in (METRIC_EUCLIDEAN, METRIC_MANHATTAN):
            dist = float(np.linalg.norm(d))
        else:
            dist = 0.0

        scale = self.spatial_scale_per_step if self.spatial_scale_per_step > 0 else 1.0
        return float(np.clip(dist / scale, 0.0, 1.0))

    def effective_cortisol_baseline(self, epoch0_loss: Optional[float] = None) -> float:
        """
        Return the cortisol baseline to use for this domain.
        If cortisol_baseline is None, auto-calibrate from epoch-0 loss.
        """
        if self.cortisol_baseline is not None:
            return self.cortisol_baseline
        if epoch0_loss is not None:
            # Auto-calibrate: baseline = epoch-0 mean - 1 sigma headroom
            # Cortisol should activate when loss rises above this
            return epoch0_loss * 0.95   # 5% below starting loss
        return 0.567   # RECON default (fallback)

    def reobserve_threshold(self) -> float:
        """
        DA threshold for switching from EXPLOIT to REOBSERVE.
        Episodic domains need a higher threshold (expect more EXPLOIT).
        """
        if self.expected_regime == "exploit":
            return 0.010    # higher threshold — don't fight expected EXPLOIT
        elif self.expected_regime == "mixed":
            return 0.003
        else:
            return 0.001    # RECON default — very sensitive


# ── Domain registry ───────────────────────────────────────────────────────────

TOPOLOGIES: dict[str, DomainTopology] = {}

def _register(t: DomainTopology):
    TOPOLOGIES[t.name] = t
    return t


# ── RECON — Outdoor Robot Navigation ─────────────────────────────────────────
_register(DomainTopology(
    name        = "recon",
    description = "Berkeley outdoor robot navigation (Jackal, 4Hz, GPS)",
    spatial_type       = SPATIAL_PERSISTENT_2D,
    spatial_metric     = METRIC_HAVERSINE,
    spatial_persistent = True,
    spatial_dim        = 2,
    spatial_scale_per_step = 0.5,   # ~0.5m per step at 4Hz walking speed
    frame_rate_hz          = 4.0,
    episode_length_typical = 70,
    inter_episode_gap      = True,
    ne_mode       = "gps",
    ach_enabled   = False,
    cortisol_baseline    = 0.567,   # calibrated from Sprint 3 training
    cortisol_sensitivity = 1.0,
    has_velocity         = True,
    has_angular_velocity = True,
    has_contact          = False,
    has_audio            = False,
    expected_regime = "reobserve",
    notes = "Production domain. Cortisol r=0.768 lag-1 validated. "
            "REOBSERVE 100% across 1.12M steps with cortisol enabled. "
            "EXPLOIT 100% without cortisol (Sprint 8d ablation).",
))

# ── PushT — Block Pushing Manipulation ───────────────────────────────────────
_register(DomainTopology(
    name        = "pusht",
    description = "PushT block pushing — gym_pusht, episodic, pixel space",
    spatial_type       = SPATIAL_EPISODIC_2D,
    spatial_metric     = METRIC_EUCLIDEAN,
    spatial_persistent = False,     # block resets each episode
    spatial_dim        = 2,
    spatial_scale_per_step = 15.0,  # ~15px/step in [0,512] space
    frame_rate_hz          = 10.0,
    episode_length_typical = 200,
    inter_episode_gap      = True,
    ne_mode       = "pixel",
    ach_enabled   = True,           # contact with T-block is meaningful
    cortisol_baseline    = None,    # AUTO-CALIBRATE from epoch-0 loss (~0.663)
    cortisol_sensitivity = 0.5,     # lower sensitivity for episodic resets
    has_velocity         = True,
    has_angular_velocity = False,
    has_contact          = True,    # block contact detection
    has_audio            = False,
    expected_regime = "exploit",    # episodic domain — EXPLOIT is correct
    notes = "Observed: EXPLOIT 100% with cortisol enabled (epoch 0 loss 0.663). "
            "Root cause: episodic resets prevent persistent spatial map. "
            "Fix: auto-calibrate cortisol_baseline to epoch-0 loss, "
            "reduce cortisol_sensitivity for episodic domains.",
))

# ── CWRU — Bearing Fault Detection (Audio/Vibration) ─────────────────────────
_register(DomainTopology(
    name        = "cwru",
    description = "CWRU bearing dataset — accelerometer vibration, fault detection",
    spatial_type       = SPATIAL_NONE,
    spatial_metric     = METRIC_NONE,
    spatial_persistent = False,
    spatial_dim        = 0,
    frame_rate_hz      = 12_000.0,  # 12kHz accelerometer
    episode_length_typical = 1024,
    inter_episode_gap  = False,     # continuous signal, window-sliced
    ne_mode       = "disabled",
    ach_enabled   = False,
    cortisol_baseline    = None,    # auto-calibrate
    cortisol_sensitivity = 2.0,     # high sensitivity — faults are rare events
    has_velocity         = False,
    has_angular_velocity = True,    # shaft rotation
    has_contact          = False,
    has_audio            = True,    # vibration as audio proxy
    expected_regime = "mixed",
    notes = "Primary signal: vibration spectrum novelty via Ado (fatigue). "
            "Fault events should trigger REOBSERVE. Normal operation EXPLOIT.",
))

# ── MIMII — Industrial Equipment Audio ───────────────────────────────────────
_register(DomainTopology(
    name        = "mimii",
    description = "MIMII industrial audio — fan, pump, slider, valve",
    spatial_type       = SPATIAL_NONE,
    spatial_metric     = METRIC_NONE,
    spatial_persistent = False,
    spatial_dim        = 0,
    frame_rate_hz      = 16_000.0,  # 16kHz audio
    episode_length_typical = 160_000,  # 10 second clips
    inter_episode_gap  = True,
    ne_mode       = "disabled",
    ach_enabled   = False,
    cortisol_baseline    = None,
    cortisol_sensitivity = 2.0,
    has_velocity         = False,
    has_angular_velocity = False,
    has_contact          = False,
    has_audio            = True,
    expected_regime = "mixed",
    notes = "Four equipment types. Anomaly detection AUROC validated. "
            "5HT (gaussian diversity) is primary signal for anomaly detection.",
))

# ── Cardiac — Heart Sound Classification ─────────────────────────────────────
_register(DomainTopology(
    name        = "cardiac",
    description = "Cardiac audio — PCG heart sounds, anomaly detection",
    spatial_type       = SPATIAL_NONE,
    spatial_metric     = METRIC_NONE,
    spatial_persistent = False,
    spatial_dim        = 0,
    frame_rate_hz      = 4_000.0,
    episode_length_typical = 4_000,
    inter_episode_gap  = True,
    ne_mode       = "disabled",
    ach_enabled   = False,
    cortisol_baseline    = None,
    cortisol_sensitivity = 3.0,     # very high — cardiac anomalies critical
    has_velocity         = False,
    has_angular_velocity = False,
    has_contact          = True,    # valve contact proxy
    has_audio            = True,
    expected_regime = "mixed",
    notes = "AUROC 0.8894 validated. High cortisol_sensitivity — "
            "distribution shift in cardiac signals is clinically significant.",
))

# ── SMAP/MSL — Satellite Telemetry ───────────────────────────────────────────
_register(DomainTopology(
    name        = "smap_msl",
    description = "NASA SMAP/MSL satellite telemetry — multivariate time series",
    spatial_type       = SPATIAL_PERSISTENT_3D,  # orbital position in 3D space
    spatial_metric     = METRIC_EUCLIDEAN,
    spatial_persistent = True,
    spatial_dim        = 3,
    spatial_scale_per_step = 1000.0,  # km/step in orbital space
    frame_rate_hz          = 1.0 / 60,  # 1 sample per minute
    episode_length_typical = 8_640,    # 6 days at 1/min
    inter_episode_gap      = False,
    ne_mode       = "disabled",         # orbital position not yet integrated
    ach_enabled   = False,
    cortisol_baseline    = None,
    cortisol_sensitivity = 4.0,         # anomalies are rare and critical
    has_velocity         = True,        # orbital velocity
    has_angular_velocity = True,        # attitude control
    has_contact          = False,
    has_audio            = False,
    expected_regime = "mixed",
    notes = "AUROC 0.8427 validated. Orbital 3D position available but not "
            "yet integrated as NE signal. Sprint 9g: integrate orbital ephemeris.",
))

# ── MVTec AD — Visual Inspection ─────────────────────────────────────────────
_register(DomainTopology(
    name        = "mvtec",
    description = "MVTec AD — visual inspection, 15 object categories",
    spatial_type       = SPATIAL_PATCH_GRID,
    spatial_metric     = METRIC_EUCLIDEAN,
    spatial_persistent = False,
    spatial_dim        = 2,
    spatial_scale_per_step = 32.0,  # patch size in pixels
    frame_rate_hz          = 1.0,   # static images
    episode_length_typical = 1,     # one image per sample
    inter_episode_gap      = True,
    ne_mode       = "disabled",     # no meaningful spatial traversal
    ach_enabled   = False,
    cortisol_baseline    = None,
    cortisol_sensitivity = 1.5,
    has_velocity         = False,
    has_angular_velocity = False,
    has_contact          = False,
    has_audio            = False,
    expected_regime = "mixed",
    notes = "Patch-level spatial structure available via DINOv2 attention maps. "
            "5HT (representation diversity) is primary anomaly signal.",
))

# ── SCAND — Multi-Campus Outdoor Navigation ───────────────────────────────────
_register(DomainTopology(
    name        = "scand",
    description = "SCAND — UT Austin campus outdoor navigation, GPS",
    spatial_type       = SPATIAL_PERSISTENT_2D,
    spatial_metric     = METRIC_HAVERSINE,
    spatial_persistent = True,
    spatial_dim        = 2,
    spatial_scale_per_step = 0.4,   # similar to RECON
    frame_rate_hz          = 5.0,
    episode_length_typical = 100,
    inter_episode_gap      = True,
    ne_mode       = "gps",
    ach_enabled   = False,
    cortisol_baseline    = None,    # auto-calibrate — different campus, different visual stats
    cortisol_sensitivity = 1.0,
    has_velocity         = True,
    has_angular_velocity = True,
    has_contact          = False,
    has_audio            = False,
    expected_regime = "reobserve",
    notes = "Sprint 9c: multi-campus GeoLatentDB. Same topology as RECON. "
            "Cortisol auto-calibrates because visual distribution differs from Berkeley.",
))

# ── PointMaze — DINO-WM benchmark ────────────────────────────────────────────
_register(DomainTopology(
    name        = "pointmaze",
    description = "PointMaze — 2D maze navigation, DINO-WM benchmark",
    spatial_type       = SPATIAL_EPISODIC_2D,
    spatial_metric     = METRIC_EUCLIDEAN,
    spatial_persistent = False,
    spatial_dim        = 2,
    spatial_scale_per_step = 0.05,  # normalised maze coordinates
    frame_rate_hz          = 10.0,
    episode_length_typical = 500,
    inter_episode_gap      = True,
    ne_mode       = "pixel",
    ach_enabled   = False,
    cortisol_baseline    = None,
    cortisol_sensitivity = 0.7,
    has_velocity         = True,
    has_angular_velocity = False,
    has_contact          = False,   # walls detected implicitly via velocity
    has_audio            = False,
    expected_regime = "mixed",      # maze has structure → some REOBSERVE expected
    notes = "DINO-WM benchmark. Episodic but has persistent maze structure "
            "within episode — mixed regime expected unlike PushT.",
))


# ── VLM Encoder Topologies (calibrated Phase 1c, 2026-04-06) ─────────────────
# Thresholds calibrated on 50 RECON frames (stable) vs 20 shift frames
# All three encoders achieve F1 > 0.88 with clean distributional separation

_register(DomainTopology(
    name        = "recon_clip_visual",
    description = "RECON navigation with CLIP ViT-B/32 visual encoder",
    spatial_type       = SPATIAL_PERSISTENT_2D,
    spatial_metric     = METRIC_HAVERSINE,
    spatial_persistent = True,
    spatial_dim        = 2,
    spatial_scale_per_step = 0.5,
    frame_rate_hz          = 4.0,
    episode_length_typical = 70,
    ne_mode       = "gps",
    cortisol_baseline    = None,  # auto-calibrate
    cortisol_sensitivity = 1.0,
    has_velocity         = True,
    has_angular_velocity = True,
    expected_regime = "reobserve",
    notes = "CLIP ViT-B/32 visual encoder on RECON. "
            "Calibrated DA threshold=0.1424 (F1=0.884). "
            "Stable mean=0.0628, shifted mean=0.2530. Clean separation.",
))

_register(DomainTopology(
    name        = "recon_clip_text_target",
    description = "RECON navigation with CLIP text-as-target (language-grounded)",
    spatial_type       = SPATIAL_PERSISTENT_2D,
    spatial_metric     = METRIC_HAVERSINE,
    spatial_persistent = True,
    spatial_dim        = 2,
    spatial_scale_per_step = 0.5,
    frame_rate_hz          = 4.0,
    episode_length_typical = 70,
    ne_mode       = "gps",
    cortisol_baseline    = None,
    cortisol_sensitivity = 1.0,
    has_velocity         = True,
    has_angular_velocity = True,
    expected_regime = "reobserve",
    notes = "CLIP text-as-target: z_pred=visual, z_target=text description. "
            "Calibrated DA threshold=0.7712 (F1=1.000 — PERFECT). "
            "Stable mean=0.7427, shifted mean=0.7812. "
            "Text: outdoor robot navigation grass sky path. "
            "Language-grounded domain shift detection: zero false positives, "
            "zero false negatives on 50+20 frame calibration set.",
))

# ── Lookup and utilities ──────────────────────────────────────────────────────

def get_topology(domain_name: str) -> DomainTopology:
    """Get topology by name. Returns generic fallback if unknown."""
    if domain_name in TOPOLOGIES:
        return TOPOLOGIES[domain_name]
    print(f"[DomainTopology] Unknown domain '{domain_name}' — using generic fallback")
    return _GENERIC_FALLBACK

def list_domains() -> list[str]:
    return sorted(TOPOLOGIES.keys())


_GENERIC_FALLBACK = DomainTopology(
    name        = "generic",
    description = "Generic fallback — minimal assumptions",
    spatial_type  = SPATIAL_NONE,
    ne_mode       = "disabled",
    cortisol_baseline    = None,      # always auto-calibrate
    cortisol_sensitivity = 1.0,
    expected_regime = "mixed",
)


# ── Cortisol auto-calibration ─────────────────────────────────────────────────

def auto_calibrate_cortisol(
    neuro,
    topology:        DomainTopology,
    epoch0_loss_mean: float,
) -> float:
    """
    Set cortisol baseline from epoch-0 loss mean.
    Call this at the end of epoch 0 training.

    Usage in train_cwm_v2.py:
        from domain_topology import get_topology, auto_calibrate_cortisol
        topo = get_topology("recon")   # or "pusht", "cwru", etc.

        # After epoch 0 completes:
        new_baseline = auto_calibrate_cortisol(neuro, topo, epoch0_mean_loss)

    Returns the new baseline for logging.
    """
    baseline = topology.effective_cortisol_baseline(epoch0_loss_mean)

    # neuro.cortisol may be a float or a CortisolSignal object
    # Try object attribute first, fall back to module-level patch
    try:
        neuro.cortisol.baseline = baseline
    except AttributeError:
        # cortisol is a float — store as override for next update cycle
        neuro._cortisol_baseline_override = baseline
        # Also try common attribute names used in different neuromodulator versions
        for attr in ["cortisol_baseline", "cort_baseline", "baseline"]:
            if hasattr(neuro, attr):
                setattr(neuro, attr, baseline)
                break

    # Adjust NE weight for non-GPS domains
    if topology.ne_mode == "disabled":
        for attr in ["ne_weight", "ne_scale", "_ne_weight"]:
            if hasattr(neuro, attr):
                setattr(neuro, attr, 0.0)
                break
    elif topology.ne_mode == "pixel":
        for attr in ["ne_weight", "ne_scale", "_ne_weight"]:
            if hasattr(neuro, attr):
                setattr(neuro, attr, getattr(neuro, attr) * 0.5)
                break

    print(f"[DomainTopology] '{topology.name}' cortisol calibrated:")
    print(f"  epoch-0 loss: {epoch0_loss_mean:.4f}")
    print(f"  baseline set: {baseline:.4f}")
    print(f"  NE mode:      {topology.ne_mode}")
    print(f"  Expected regime: {topology.expected_regime}")

    return baseline


# ── Print summary ──────────────────────────────────────────────────────────────

def print_topology(name: str):
    t = get_topology(name)
    print(f"\n{'='*60}")
    print(f"  {t.name} — {t.description}")
    print(f"{'='*60}")
    print(f"  Spatial:    {t.spatial_type} ({t.spatial_metric})")
    print(f"  Persistent: {t.spatial_persistent}")
    print(f"  NE mode:    {t.ne_mode}")
    print(f"  Cortisol:   {'auto' if t.cortisol_baseline is None else t.cortisol_baseline}"
          f" (sensitivity {t.cortisol_sensitivity}x)")
    print(f"  Signals:    vel={t.has_velocity} ang={t.has_angular_velocity} "
          f"contact={t.has_contact} audio={t.has_audio}")
    print(f"  Expected:   {t.expected_regime}")
    if t.notes:
        print(f"  Notes:      {t.notes[:80]}...")


# ═══════════════════════════════════════════════════════════════════════════════
# HOW TO ADD A NEW DOMAIN
# ═══════════════════════════════════════════════════════════════════════════════
#
# Step 1 — Characterise your domain (answer these questions):
#   a) Does it have spatial coordinates? (GPS, pixel XY, 3D position?)
#   b) Do spatial coords reset each episode or accumulate across sessions?
#   c) What's the typical displacement per step? (metres, pixels, grid cells)
#   d) What's the frame rate? (Hz)
#   e) Does the system make contact with objects? (relevant for ACh)
#   f) Is there audio? (relevant for Ado/fatigue)
#   g) What regime do you expect? (simple env → exploit, rich env → reobserve)
#
# Step 2 — Choose spatial_type:
#   SPATIAL_NONE         → audio only, telemetry, tabular (no spatial coords)
#   SPATIAL_EPISODIC_2D  → 2D coords that reset each episode (PushT, MiniGrid)
#   SPATIAL_PERSISTENT_2D → 2D GPS accumulating across session (RECON, SCAND)
#   SPATIAL_PERSISTENT_3D → 3D position accumulating (drone, 3D mapping)
#   SPATIAL_GRID_2D      → discrete grid (MiniGrid, maze, chess)
#   SPATIAL_PATCH_GRID   → 2D image patches (MVTec, visual inspection)
#
# Step 3 — Choose ne_mode:
#   "gps"      → lat/lon coordinates, haversine metric
#   "pixel"    → pixel XY in some image space, euclidean metric
#   "grid"     → discrete grid cell (i, j), manhattan metric
#   "disabled" → no spatial signal (audio, telemetry)
#
# Step 4 — Set cortisol_baseline:
#   None  → auto-calibrate from epoch-0 loss (RECOMMENDED for new domains)
#   float → hardcode if you know the expected loss scale
#
# Step 5 — Add to this file:
#
#   _register(DomainTopology(
#       name        = "my_domain",
#       description = "What this domain is",
#       spatial_type       = SPATIAL_EPISODIC_2D,   # your choice
#       spatial_metric     = METRIC_EUCLIDEAN,
#       spatial_persistent = False,
#       spatial_dim        = 2,
#       spatial_scale_per_step = 20.0,              # typical step size
#       frame_rate_hz          = 10.0,
#       episode_length_typical = 100,
#       inter_episode_gap      = True,
#       ne_mode            = "pixel",
#       ach_enabled        = True,                  # if contact matters
#       cortisol_baseline  = None,                  # auto-calibrate
#       cortisol_sensitivity = 1.0,
#       has_velocity         = True,
#       expected_regime    = "mixed",
#       notes = "Brief description of expected behaviour",
#   ))
#
# Step 6 — Add to train_cwm_v2.py:
#
#   from domain_topology import get_topology, auto_calibrate_cortisol
#   topo = get_topology("my_domain")
#
#   # End of epoch 0:
#   new_baseline = auto_calibrate_cortisol(neuro, topo, epoch0_mean_loss)
#
# Step 7 — Validate:
#   - Run 1 epoch and check regime (EXPLOIT vs REOBSERVE)
#   - If EXPLOIT when you expected REOBSERVE: lower cortisol_baseline
#   - If REOBSERVE when you expected EXPLOIT: raise cortisol_sensitivity
#   - If NE not responding: check spatial_scale_per_step is realistic
#
# ═══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    print("NeMo-WM Domain Topology Registry")
    print(f"Registered domains: {list_domains()}\n")

    for name in list_domains():
        print_topology(name)

    print("\n--- Auto-calibration example ---")
    topo = get_topology("pusht")
    epoch0_loss = 0.663   # observed
    baseline = topo.effective_cortisol_baseline(epoch0_loss)
    print(f"PushT auto-calibrated baseline: {baseline:.4f}")
    print(f"  (was hardcoded 0.567 for RECON — too low for PushT)")
    print(f"  Expected regime: {topo.expected_regime}")
    print(f"  EXPLOIT is correct for episodic domains")
