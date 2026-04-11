"""
CORTEX-PE Eval Integration Patches
=====================================
Exactly two lines per eval script. Remove them to revert.
Run this file to verify each patch is syntactically valid.
"""

import textwrap

# ── The 2-line patches ────────────────────────────────────────────────────────

IMPORTS = {
    "eval_cwru_bearing.py":  'from perception_annotator import PerceptionAnnotator; _ann = PerceptionAnnotator("bearing", annotation_rate=0.15)',
    "eval_cardiac_audio.py": 'from perception_annotator import PerceptionAnnotator; _ann = PerceptionAnnotator("cardiac", annotation_rate=0.20)',
    "eval_smap_msl.py":      'from perception_annotator import PerceptionAnnotator; _ann = PerceptionAnnotator("smap",    annotation_rate=0.10)',
    "recon_navigator.py":    'from perception_annotator import PerceptionAnnotator; _ann = PerceptionAnnotator("recon",   annotation_rate=0.05)',
}

OBSERVE_CALLS = {
    "eval_cwru_bearing.py":  '_ann.observe(feat, anomaly_score=float(score), label=fault_type)',
    "eval_cardiac_audio.py": '_ann.observe(z.numpy(), anomaly_score=float(score), label=ground_truth_label)',
    "eval_smap_msl.py":      '_ann.observe(window_features, anomaly_score=float(anomaly_score), label=channel_id)',
    "recon_navigator.py":    '_ann.observe(z.numpy(), anomaly_score=float(qm_dist), label="obstacle" if qm_dist > 0.75 else "clear")',
}

LOCATIONS = {
    "eval_cwru_bearing.py": {
        "import": "Top of file, after existing imports",
        "observe": "After 'anom_scores = pca_score(...)' — inside fault-type loop",
        "note": "PCA features are 6–12D; FeatureProjector lifts to 128D automatically",
    },
    "eval_cardiac_audio.py": {
        "import": "Top of file, after existing imports",
        "observe": "After 'score = anomaly_scorer(z)' — StudentEncoder output is already 128D",
        "note": "No projection needed — StudentEncoder output passed directly",
    },
    "eval_smap_msl.py": {
        "import": "Top of file, after existing imports",
        "observe": "After anomaly score computed per window — 25D telemetry features",
        "note": "FeatureProjector: 25D → 128D via online PCA after 32 samples",
    },
    "recon_navigator.py": {
        "import": "Top of file, after existing imports",
        "observe": "After 'qm_dist = temporal_head(z, z_goal)' — repurpose as novelty score",
        "note": "annotation_rate=0.05 because RECON runs at 4Hz (many frames)",
    },
}

FULL_EXAMPLES = {
    "eval_cwru_bearing.py": """
# ── BEFORE (existing code) ──────────────────────────────────────────────────

import numpy as np
import scipy.io
# ... other imports ...

def run(args):
    ...
    for fault_type in ANOMALY_TYPES:
        anom_feats  = np.concatenate(type_feats)
        anom_scores = pca_score(anom_feats, pca_mean, pca_comps)
        auc = auroc(...)
    print(f"AUROC = {auc:.4f}")

# ── AFTER (2 lines added, marked ← NEW) ────────────────────────────────────

import numpy as np
import scipy.io
from perception_annotator import PerceptionAnnotator           # ← NEW (line 1)
_ann = PerceptionAnnotator("bearing", annotation_rate=0.15)    # ← NEW (line 2)

def run(args):
    ...
    for fault_type in ANOMALY_TYPES:
        anom_feats  = np.concatenate(type_feats)
        anom_scores = pca_score(anom_feats, pca_mean, pca_comps)
        for feat, score in zip(anom_feats, anom_scores):       # ← NEW (observe)
            _ann.observe(feat, float(score), label=fault_type)  # ← NEW (observe)
        auc = auroc(...)
    print(f"AUROC = {auc:.4f}")
    _ann.close()                                               # ← NEW (cleanup)
""",

    "eval_cardiac_audio.py": """
# ── AFTER ────────────────────────────────────────────────────────────────────

from perception_annotator import PerceptionAnnotator           # ← NEW
_ann = PerceptionAnnotator("cardiac", annotation_rate=0.20)    # ← NEW

for batch in dataloader:
    audio, label = batch
    z = student_encoder(audio)            # 128-D NPU latent (existing)
    score = anomaly_scorer(z)             # anomaly score   (existing)
    _ann.observe(z.numpy(), float(score), label=str(label))  # ← NEW

_ann.close()                                                   # ← NEW
""",

    "eval_smap_msl.py": """
# ── AFTER ────────────────────────────────────────────────────────────────────

from perception_annotator import PerceptionAnnotator           # ← NEW
_ann = PerceptionAnnotator("smap", annotation_rate=0.10)       # ← NEW

for channel_id, windows in telemetry_channels.items():
    for w in windows:
        score = scorer(w)                         # existing
        _ann.observe(w, float(score), channel_id) # ← NEW (25D → 128D auto)

_ann.close()                                                   # ← NEW
""",

    "recon_navigator.py": """
# ── AFTER ────────────────────────────────────────────────────────────────────

from perception_annotator import PerceptionAnnotator              # ← NEW
_ann = PerceptionAnnotator("recon", annotation_rate=0.05)         # ← NEW

for frame in trajectory:
    z = student_encoder(frame)                  # 128-D NPU (existing)
    qm_dist = temporal_head(z, z_goal)          # quasimetric (existing)
    label = "obstacle" if qm_dist > 0.75 else "clear"
    _ann.observe(z.numpy(), float(qm_dist), label)  # ← NEW

_ann.close()                                                      # ← NEW
""",
}

# ── Verification ──────────────────────────────────────────────────────────────

def verify_all() -> bool:
    print("=== Eval Integration Patches — Verification ===\n")
    all_ok = True

    for fname in IMPORTS:
        import_line  = IMPORTS[fname]
        observe_line = OBSERVE_CALLS[fname]
        loc          = LOCATIONS[fname]

        # Verify both lines compile
        errors = []
        for line, tag in [(import_line, "import"), (observe_line, "observe")]:
            try:
                compile(line, fname, "single")
            except SyntaxError as e:
                errors.append(f"{tag}: {e}")

        if errors:
            print(f"  ❌  {fname}")
            for e in errors:
                print(f"       {e}")
            all_ok = False
        else:
            print(f"  ✅  {fname}")
            print(f"       Location : {loc['import']}")
            print(f"       Note     : {loc['note']}")

    print()
    return all_ok


def print_patch(fname: str):
    """Print the full before/after example for a given file."""
    if fname not in FULL_EXAMPLES:
        print(f"No example for {fname}")
        return
    print(f"\n{'─'*60}")
    print(f"  {fname}")
    print(f"{'─'*60}")
    print(FULL_EXAMPLES[fname])
    print(f"\n  Line 1 (top of file):")
    print(f"    {IMPORTS[fname]}")
    print(f"\n  Line 2 (inside eval loop, after score is computed):")
    print(f"    {OBSERVE_CALLS[fname]}")
    print(f"\n  Cleanup (after final print):")
    print(f"    _ann.close()")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in FULL_EXAMPLES:
        print_patch(sys.argv[1])
    else:
        ok = verify_all()
        print("  All patches verified ✅" if ok else "  ❌ Errors above")
        print()
        print("  Show full example:  python eval_integration_patches.py eval_cwru_bearing.py")
        print("  Show for cardiac:   python eval_integration_patches.py eval_cardiac_audio.py")
        print("  Show for SMAP:      python eval_integration_patches.py eval_smap_msl.py")
        print("  Show for RECON:     python eval_integration_patches.py recon_navigator.py")
