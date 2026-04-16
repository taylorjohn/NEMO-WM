"""
overnight_language_curriculum.py — Long-Form Language Training
=================================================================
Run overnight. Teaches 500+ words across 15 domains with:
  - Multiple passes (not just one curriculum run)
  - Sleep consolidation cycles (replay + narration)
  - Contrastive refinement
  - Progressive complexity
  - Checkpoint saves every 30 min
  - Resume from checkpoint

Expected runtime: 8-12 hours
Expected vocabulary: 500+ words
Expected cognitive age: 6-8 years

Usage:
    python overnight_language_curriculum.py
    python overnight_language_curriculum.py --hours 10
    python overnight_language_curriculum.py --resume
"""

import argparse
import time
import pickle
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta

from word_grounder import WordGrounder, SentenceComprehender, D_BELIEF
from language_v2 import (BeliefAccumulator, PredictiveGrounder,
                          ContrastiveGrounder, EpisodicReplayLearner,
                          ModifierProcessor)
from curriculum_generator import CurriculumGenerator, BeliefFactory

CHECKPOINT_DIR = Path("outputs/overnight_checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Extended Vocabulary — 15 domains, 500+ words
# ──────────────────────────────────────────────────────────────────────────────

def build_extended_curriculum(d_belief=D_BELIEF, seed=42):
    """
    Build a massive curriculum of 500+ grounded concepts across
    15 semantic domains with 8-12 variations each.
    """
    bf = BeliefFactory(d_belief=d_belief, seed=seed)
    rng = np.random.RandomState(seed)
    curriculum = []

    # ── Domain 1: Physics (40 concepts) ──
    physics = {
        "gravity": (dict(vertical=-1), 0.7, "Curious-Alert", "Fy=-9.81*m"),
        "weight": (dict(vertical=-0.5, weight=1), 0.4, "Calm-Confident", "W=mg"),
        "falling": (dict(vertical=-1, speed=0.5), 0.8, "Alert-Cautious", ""),
        "dropping": (dict(vertical=-0.8, speed=0.3), 0.6, "Alert-Cautious", ""),
        "sinking": (dict(vertical=-0.5, buoyancy=-0.5), 0.5, "Calm-Confident", ""),
        "floating": (dict(vertical=0.2, buoyancy=1), 0.3, "Calm-Relaxed", "F=rho*V*g"),
        "rising": (dict(vertical=0.8, speed=0.3), 0.5, "Curious-Alert", ""),
        "friction": (dict(friction=-1), 0.6, "Curious-Confident", "F=-mu*N"),
        "sliding": (dict(friction=-0.5, locomotion=0.5), 0.5, "Alert-Confident", ""),
        "drag": (dict(friction=-0.5, speed=-0.3), 0.5, "Calm-Confident", "F=-Cd*v^2"),
        "force": (dict(force=1), 0.5, "Curious-Confident", "F=ma"),
        "energy": (dict(energy=1), 0.4, "Calm-Confident", "E=0.5mv^2"),
        "momentum": (dict(momentum=1), 0.3, "Calm-Relaxed", "p=mv"),
        "acceleration": (dict(speed=0.5, force=1), 0.6, "Curious-Alert", "a=F/m"),
        "deceleration": (dict(speed=-0.5, friction=0.5), 0.4, "Alert-Cautious", ""),
        "velocity": (dict(speed=0.8, forward=0.5), 0.3, "Calm-Confident", ""),
        "collision": (dict(collision=1, force=1), 0.9, "Stressed-Alert", ""),
        "impact": (dict(collision=1, force=0.8), 0.9, "Stressed-Alert", ""),
        "bounce": (dict(vertical=0.5, spring=0.5), 0.8, "Curious-Alert", ""),
        "spring": (dict(spring=1), 0.5, "Curious-Confident", "F=-kx"),
        "elastic": (dict(spring=0.8), 0.4, "Calm-Confident", ""),
        "rigid": (dict(spring=-0.5), 0.3, "Calm-Confident", ""),
        "magnetic": (dict(magnetic=1), 0.7, "Curious-Alert", "F=k/r^2"),
        "attract": (dict(magnetic=0.8, proximity=0.5), 0.6, "Curious-Confident", ""),
        "repel": (dict(magnetic=-0.8), 0.6, "Cautious-Alert", ""),
        "pressure": (dict(force=0.8, weight=0.3), 0.4, "Calm-Confident", "P=F/A"),
        "tension": (dict(spring=0.3, force=0.5), 0.4, "Alert-Confident", ""),
        "torque": (dict(rotation=1, force=0.5), 0.4, "Curious-Alert", "tau=rF"),
        "rotation": (dict(rotation=1), 0.3, "Calm-Confident", ""),
        "oscillation": (dict(vertical=0.3, time=0.5), 0.5, "Curious-Alert", ""),
        "wave": (dict(speed=0.5, time=0.5), 0.4, "Curious-Confident", ""),
        "resonance": (dict(spring=0.5, time=0.8), 0.6, "Curious-Alert", ""),
        "inertia": (dict(momentum=0.8, weight=0.5), 0.4, "Calm-Relaxed", ""),
        "mass": (dict(weight=1, size=0.3), 0.3, "Calm-Confident", ""),
        "density": (dict(weight=0.8, size=-0.3), 0.3, "Calm-Confident", ""),
        "buoyancy": (dict(buoyancy=1, vertical=0.3), 0.5, "Curious-Confident", ""),
        "viscosity": (dict(friction=0.8, speed=-0.5), 0.4, "Calm-Confident", ""),
        "turbulence": (dict(friction=0.8, curiosity=0.5), 0.7, "Alert-Cautious", ""),
        "equilibrium": (dict(force=0, stress=-0.5), 0.2, "Calm-Relaxed", ""),
        "unstable": (dict(force=0.5, stress=0.5), 0.7, "Cautious-Alert", ""),
    }

    # ── Domain 2: Materials (25 concepts) ──
    materials = {
        "wood": (dict(weight=0.3, friction=0.3), 0.2, "Calm-Relaxed"),
        "metal": (dict(weight=0.8, magnetic=0.3), 0.3, "Calm-Confident"),
        "plastic": (dict(weight=-0.3, spring=0.3), 0.2, "Calm-Relaxed"),
        "glass": (dict(weight=0.3, friction=-0.3), 0.5, "Cautious-Alert"),
        "rubber": (dict(spring=0.8, friction=0.5), 0.3, "Calm-Confident"),
        "stone": (dict(weight=1, spring=-0.5), 0.3, "Calm-Confident"),
        "cloth": (dict(weight=-0.5, friction=0.5), 0.1, "Calm-Relaxed"),
        "paper": (dict(weight=-0.7, friction=0.3), 0.1, "Calm-Relaxed"),
        "ice": (dict(weight=0.3, friction=-0.8), 0.5, "Cautious-Alert"),
        "water": (dict(buoyancy=0.5, friction=-0.3), 0.2, "Calm-Relaxed"),
        "sand": (dict(friction=0.5, weight=0.5), 0.3, "Calm-Confident"),
        "mud": (dict(friction=0.8, weight=0.5), 0.4, "Cautious-Alert"),
        "soft": (dict(spring=0.5, weight=-0.3), 0.1, "Calm-Relaxed"),
        "hard": (dict(spring=-0.5, weight=0.5), 0.3, "Calm-Confident"),
        "smooth": (dict(friction=-0.5), 0.1, "Calm-Relaxed"),
        "rough": (dict(friction=0.5), 0.3, "Cautious-Alert"),
        "sharp": (dict(collision=0.5, threat=0.5), 0.6, "Cautious-Alert"),
        "blunt": (dict(collision=0.3, threat=-0.3), 0.2, "Calm-Confident"),
        "hot": (dict(collision=0.3, threat=0.3), 0.6, "Alert-Cautious"),
        "cold": (dict(collision=-0.3), 0.3, "Alert-Cautious"),
        "warm": (dict(threat=-0.2), 0.2, "Calm-Relaxed"),
        "cool": (dict(threat=-0.1), 0.1, "Calm-Relaxed"),
        "dry": (dict(buoyancy=-0.3, friction=0.3), 0.1, "Calm-Relaxed"),
        "wet": (dict(buoyancy=0.3, friction=-0.3), 0.3, "Alert-Cautious"),
        "dense": (dict(weight=0.8, size=-0.2), 0.3, "Calm-Confident"),
    }

    # ── Domain 3: Navigation (35 concepts) ──
    navigation = {
        "corridor": (dict(path=1), 0.1, "Calm-Relaxed"),
        "hallway": (dict(path=0.9), 0.1, "Calm-Relaxed"),
        "passage": (dict(path=0.8, enclosure=0.3), 0.2, "Calm-Confident"),
        "tunnel": (dict(path=0.8, enclosure=0.8), 0.4, "Cautious-Alert"),
        "path": (dict(path=0.7), 0.2, "Calm-Confident"),
        "road": (dict(path=0.8), 0.2, "Calm-Confident"),
        "street": (dict(path=0.7), 0.2, "Calm-Confident"),
        "trail": (dict(path=0.6), 0.3, "Curious-Confident"),
        "bridge": (dict(path=0.8, elevation=0.5), 0.5, "Alert-Cautious"),
        "intersection": (dict(junction=1), 0.5, "Alert-Cautious"),
        "junction": (dict(junction=0.9), 0.4, "Alert-Cautious"),
        "crossroads": (dict(junction=0.8, curiosity=0.3), 0.5, "Curious-Alert"),
        "corner": (dict(junction=0.5, proximity=0.3), 0.4, "Cautious-Alert"),
        "bend": (dict(junction=0.3, rotation=0.5), 0.3, "Cautious-Alert"),
        "uphill": (dict(slope=1), 0.4, "Alert-Bold"),
        "downhill": (dict(slope=-1), 0.3, "Calm-Confident"),
        "steep": (dict(slope=1.5), 0.5, "Stressed-Alert"),
        "gentle": (dict(slope=0.3), 0.1, "Calm-Relaxed"),
        "flat": (dict(slope=0), 0.1, "Calm-Relaxed"),
        "landmark": (dict(landmark=1), 0.6, "Curious-Alert"),
        "building": (dict(landmark=0.8, wall=0.3), 0.3, "Calm-Confident"),
        "structure": (dict(landmark=0.5, wall=0.5), 0.3, "Calm-Confident"),
        "monument": (dict(landmark=1, curiosity=0.5), 0.5, "Curious-Alert"),
        "wall": (dict(wall=1), 0.3, "Cautious-Alert"),
        "barrier": (dict(wall=0.8, threat=0.3), 0.5, "Cautious-Alert"),
        "fence": (dict(wall=0.5), 0.3, "Cautious-Alert"),
        "gate": (dict(door=0.8), 0.4, "Curious-Confident"),
        "door": (dict(door=1), 0.4, "Curious-Confident"),
        "doorway": (dict(door=0.9, enclosure=0.3), 0.4, "Curious-Confident"),
        "entrance": (dict(door=0.7, curiosity=0.3), 0.4, "Curious-Confident"),
        "exit": (dict(door=0.7, curiosity=-0.3), 0.3, "Calm-Confident"),
        "opening": (dict(door=0.6), 0.3, "Calm-Confident"),
        "obstacle": (dict(wall=0.8, threat=0.5), 0.6, "Stressed-Cautious"),
        "hazard": (dict(threat=0.8, wall=0.3), 0.8, "Stressed-Alert"),
        "pit": (dict(vertical=-1, threat=0.8), 0.8, "Stressed-Alert"),
    }

    # ── Domain 4: Emotion/State (35 concepts) ──
    emotion = {
        "danger": (dict(threat=1, stress=0.8), 0.9, "Stressed-Alert"),
        "threat": (dict(threat=0.9, stress=0.7), 0.85, "Stressed-Reactive"),
        "risk": (dict(threat=0.6, stress=0.3), 0.6, "Cautious-Alert"),
        "safe": (dict(threat=-0.5, stress=-0.5), 0.05, "Calm-Relaxed"),
        "secure": (dict(threat=-0.4, stress=-0.6), 0.05, "Calm-Confident"),
        "protected": (dict(threat=-0.5, stress=-0.3), 0.1, "Calm-Confident"),
        "curious": (dict(curiosity=1), 0.7, "Curious-Alert"),
        "interested": (dict(curiosity=0.7), 0.5, "Curious-Confident"),
        "surprised": (dict(curiosity=1.2, confidence=-0.3), 0.9, "Curious-Uncertain"),
        "amazed": (dict(curiosity=1.5, confidence=0.3), 0.9, "Curious-Bold"),
        "familiar": (dict(curiosity=-0.5, confidence=0.5), 0.05, "Familiar-Relaxed"),
        "novel": (dict(curiosity=1), 0.8, "Curious-Alert"),
        "strange": (dict(curiosity=0.8, confidence=-0.5), 0.6, "Curious-Uncertain"),
        "stressed": (dict(stress=1, confidence=-0.3), 0.6, "Stressed-Alert"),
        "anxious": (dict(stress=0.8, threat=0.5), 0.5, "Stressed-Cautious"),
        "worried": (dict(stress=0.6, threat=0.3), 0.5, "Cautious-Alert"),
        "calm": (dict(stress=-0.5), 0.05, "Calm-Relaxed"),
        "relaxed": (dict(stress=-0.6), 0.03, "Calm-Relaxed"),
        "peaceful": (dict(stress=-0.7, threat=-0.5), 0.02, "Calm-Relaxed"),
        "confident": (dict(confidence=1), 0.3, "Confident-Calm"),
        "certain": (dict(confidence=0.8), 0.2, "Confident-Calm"),
        "sure": (dict(confidence=0.7), 0.2, "Confident-Calm"),
        "uncertain": (dict(confidence=-0.5), 0.5, "Uncertain-Cautious"),
        "unsure": (dict(confidence=-0.4), 0.4, "Uncertain-Cautious"),
        "confused": (dict(confidence=-0.8, curiosity=0.5), 0.5, "Uncertain-Curious"),
        "excited": (dict(curiosity=0.8, confidence=0.7), 0.7, "Curious-Bold"),
        "eager": (dict(curiosity=0.7, confidence=0.5), 0.6, "Curious-Confident"),
        "bored": (dict(curiosity=-0.8), 0.02, "Familiar-Relaxed"),
        "tired": (dict(stress=0.3, confidence=-0.3), 0.2, "Stressed-Fatigued"),
        "alert": (dict(stress=0.3, confidence=0.3), 0.5, "Alert-Cautious"),
        "focused": (dict(confidence=0.5, curiosity=0.3), 0.4, "Confident-Alert"),
        "distracted": (dict(confidence=-0.3, curiosity=0.5), 0.4, "Uncertain-Curious"),
        "determined": (dict(confidence=0.8, stress=0.2), 0.5, "Confident-Bold"),
        "hesitant": (dict(confidence=-0.5, stress=0.3), 0.4, "Uncertain-Cautious"),
        "bold": (dict(confidence=0.8, threat=-0.3), 0.7, "Confident-Bold"),
        "careful": (dict(confidence=0.3, stress=0.3, threat=0.2), 0.3, "Cautious-Alert"),
    }

    # ── Domain 5: Action (40 concepts) ──
    action = {
        "push": (dict(push_pull=1, force=0.5), 0.5, "Alert-Confident"),
        "pull": (dict(push_pull=-1, force=0.5), 0.5, "Alert-Confident"),
        "shove": (dict(push_pull=1.5, force=0.8), 0.7, "Alert-Bold"),
        "nudge": (dict(push_pull=0.3, force=0.2), 0.2, "Calm-Confident"),
        "drag": (dict(push_pull=-0.5, friction=0.5), 0.5, "Alert-Confident"),
        "lift": (dict(vertical=0.8, force=0.8), 0.5, "Alert-Bold"),
        "lower": (dict(vertical=-0.5, force=0.5), 0.3, "Calm-Confident"),
        "drop": (dict(vertical=-1, grip=-1), 0.6, "Alert-Cautious"),
        "grab": (dict(grip=1, force=0.5), 0.6, "Alert-Confident"),
        "release": (dict(grip=-1), 0.3, "Calm-Confident"),
        "hold": (dict(grip=0.8, locomotion=-0.3), 0.3, "Calm-Confident"),
        "throw": (dict(force=1, momentum=0.8, speed=1), 0.7, "Alert-Bold"),
        "catch": (dict(grip=1, momentum=-0.5), 0.6, "Alert-Confident"),
        "move": (dict(locomotion=0.5), 0.3, "Calm-Confident"),
        "stop": (dict(locomotion=-1, friction=0.3), 0.2, "Calm-Relaxed"),
        "pause": (dict(locomotion=-0.5), 0.2, "Calm-Confident"),
        "continue": (dict(locomotion=0.5, time=0.3), 0.2, "Calm-Confident"),
        "turn": (dict(rotation=1), 0.3, "Alert-Cautious"),
        "rotate": (dict(rotation=1.2), 0.4, "Curious-Alert"),
        "spin": (dict(rotation=1.5, speed=0.5), 0.5, "Curious-Alert"),
        "walk": (dict(locomotion=0.5, speed=0.3), 0.2, "Calm-Confident"),
        "run": (dict(locomotion=0.8, speed=1), 0.5, "Alert-Bold"),
        "jump": (dict(vertical=0.8, speed=0.5), 0.7, "Alert-Bold"),
        "climb": (dict(slope=0.5, vertical=0.5), 0.5, "Alert-Bold"),
        "descend": (dict(slope=-0.5, vertical=-0.3), 0.3, "Calm-Confident"),
        "explore": (dict(locomotion=0.3, curiosity=0.8), 0.6, "Curious-Bold"),
        "avoid": (dict(wall=0.3, threat=0.5), 0.5, "Cautious-Alert"),
        "approach": (dict(forward=0.5, curiosity=0.3), 0.4, "Curious-Confident"),
        "retreat": (dict(forward=-0.5, threat=0.3), 0.5, "Cautious-Alert"),
        "follow": (dict(forward=0.3, curiosity=0.3), 0.3, "Curious-Confident"),
        "lead": (dict(forward=0.5, confidence=0.5), 0.4, "Confident-Bold"),
        "hide": (dict(enclosure=0.5, threat=0.3), 0.5, "Stressed-Cautious"),
        "seek": (dict(locomotion=0.3, curiosity=0.7), 0.6, "Curious-Bold"),
        "wait": (dict(locomotion=-0.5, urgency=-0.5), 0.2, "Calm-Relaxed"),
        "rest": (dict(locomotion=-0.8, stress=-0.5), 0.1, "Calm-Relaxed"),
        "build": (dict(force=0.5, curiosity=0.3), 0.4, "Confident-Bold"),
        "break": (dict(collision=0.8, force=0.5), 0.7, "Alert-Bold"),
        "fix": (dict(force=0.3, confidence=0.5), 0.4, "Confident-Calm"),
        "open": (dict(door=0.8, grip=0.5), 0.4, "Curious-Confident"),
        "close": (dict(door=-0.5, grip=0.3), 0.3, "Calm-Confident"),
    }

    # ── Domain 6: Spatial (30 concepts) ──
    spatial = {
        "left": (dict(lateral=-1), 0.2, "Calm-Confident"),
        "right": (dict(lateral=1), 0.2, "Calm-Confident"),
        "up": (dict(vertical_pos=1), 0.2, "Calm-Confident"),
        "down": (dict(vertical_pos=-1), 0.2, "Calm-Confident"),
        "forward": (dict(forward=1), 0.2, "Calm-Confident"),
        "backward": (dict(forward=-1), 0.2, "Calm-Confident"),
        "north": (dict(forward=0.8, lateral=0), 0.2, "Calm-Confident"),
        "south": (dict(forward=-0.8, lateral=0), 0.2, "Calm-Confident"),
        "east": (dict(lateral=0.8, forward=0), 0.2, "Calm-Confident"),
        "west": (dict(lateral=-0.8, forward=0), 0.2, "Calm-Confident"),
        "near": (dict(proximity=1), 0.3, "Alert-Cautious"),
        "far": (dict(proximity=-1), 0.1, "Calm-Relaxed"),
        "close": (dict(proximity=0.8), 0.3, "Alert-Cautious"),
        "distant": (dict(proximity=-0.8), 0.1, "Calm-Relaxed"),
        "inside": (dict(enclosure=1), 0.2, "Calm-Confident"),
        "outside": (dict(enclosure=-1), 0.2, "Calm-Confident"),
        "within": (dict(enclosure=0.8), 0.2, "Calm-Confident"),
        "above": (dict(elevation=1), 0.2, "Calm-Confident"),
        "below": (dict(elevation=-1), 0.2, "Calm-Confident"),
        "beneath": (dict(elevation=-0.8), 0.2, "Calm-Confident"),
        "over": (dict(elevation=0.8), 0.2, "Calm-Confident"),
        "under": (dict(elevation=-0.8), 0.2, "Calm-Confident"),
        "between": (dict(centrality=0.5), 0.3, "Alert-Cautious"),
        "among": (dict(centrality=0.3, proximity=0.5), 0.2, "Calm-Confident"),
        "center": (dict(centrality=1), 0.2, "Calm-Confident"),
        "edge": (dict(centrality=-1, threat=0.3), 0.4, "Cautious-Alert"),
        "top": (dict(elevation=1.5), 0.2, "Calm-Confident"),
        "bottom": (dict(elevation=-1.5), 0.2, "Calm-Confident"),
        "behind": (dict(forward=-0.5), 0.2, "Calm-Confident"),
        "ahead": (dict(forward=1), 0.3, "Alert-Confident"),
    }

    # ── Domain 7: Temporal (25 concepts) ──
    temporal = {
        "before": (dict(time=-1), 0.2, "Calm-Confident"),
        "after": (dict(time=1), 0.2, "Calm-Confident"),
        "during": (dict(time=0, frequency=0.5), 0.2, "Calm-Confident"),
        "now": (dict(time=0, urgency=0.8), 0.3, "Alert-Confident"),
        "currently": (dict(time=0, urgency=0.5), 0.3, "Alert-Confident"),
        "soon": (dict(time=0.3, urgency=0.5), 0.3, "Alert-Cautious"),
        "later": (dict(time=0.8, urgency=-0.3), 0.1, "Calm-Relaxed"),
        "yesterday": (dict(time=-1, urgency=-0.5), 0.1, "Calm-Relaxed"),
        "today": (dict(time=0), 0.2, "Calm-Confident"),
        "tomorrow": (dict(time=0.8, urgency=0.3), 0.2, "Calm-Confident"),
        "always": (dict(frequency=1), 0.1, "Calm-Confident"),
        "never": (dict(frequency=-1), 0.2, "Calm-Confident"),
        "sometimes": (dict(frequency=0.3), 0.2, "Uncertain-Cautious"),
        "often": (dict(frequency=0.7), 0.2, "Calm-Confident"),
        "rarely": (dict(frequency=-0.7), 0.1, "Calm-Confident"),
        "quickly": (dict(urgency=1, speed=1), 0.4, "Alert-Bold"),
        "slowly": (dict(urgency=-0.5, speed=-1), 0.1, "Calm-Relaxed"),
        "suddenly": (dict(urgency=1.5, curiosity=0.5), 0.8, "Alert-Reactive"),
        "gradually": (dict(urgency=-0.3, time=0.3), 0.1, "Calm-Relaxed"),
        "first": (dict(time=-1, urgency=0.3), 0.3, "Alert-Confident"),
        "last": (dict(time=1, urgency=-0.2), 0.2, "Calm-Confident"),
        "next": (dict(time=0.3, curiosity=0.3), 0.3, "Curious-Confident"),
        "previous": (dict(time=-0.5), 0.2, "Calm-Confident"),
        "begin": (dict(time=-1, curiosity=0.5), 0.4, "Curious-Bold"),
        "finish": (dict(time=1, confidence=0.5), 0.3, "Confident-Calm"),
    }

    # ── Domain 8: Magnitude (20 concepts) ──
    magnitude = {
        "fast": (dict(speed=1), 0.4, "Alert-Bold"),
        "slow": (dict(speed=-1), 0.1, "Calm-Relaxed"),
        "quick": (dict(speed=0.8, urgency=0.5), 0.4, "Alert-Bold"),
        "sluggish": (dict(speed=-0.8), 0.1, "Calm-Relaxed"),
        "big": (dict(size=1), 0.3, "Alert-Cautious"),
        "small": (dict(size=-1), 0.1, "Calm-Confident"),
        "large": (dict(size=0.8), 0.3, "Alert-Cautious"),
        "tiny": (dict(size=-0.8), 0.1, "Calm-Confident"),
        "huge": (dict(size=1.5), 0.5, "Alert-Bold"),
        "massive": (dict(size=1.5, weight=1), 0.5, "Alert-Bold"),
        "strong": (dict(strength=1), 0.4, "Alert-Bold"),
        "weak": (dict(strength=-1), 0.2, "Calm-Relaxed"),
        "powerful": (dict(strength=0.9), 0.5, "Confident-Bold"),
        "feeble": (dict(strength=-0.8), 0.2, "Calm-Relaxed"),
        "heavy": (dict(weight=1), 0.3, "Alert-Cautious"),
        "light": (dict(weight=-1), 0.1, "Calm-Relaxed"),
        "high": (dict(vertical_pos=1.5), 0.3, "Alert-Cautious"),
        "low": (dict(vertical_pos=-1.5), 0.2, "Calm-Relaxed"),
        "tall": (dict(vertical_pos=1.2, size=0.5), 0.3, "Alert-Cautious"),
        "short": (dict(vertical_pos=-0.5, size=-0.3), 0.1, "Calm-Confident"),
    }

    # ── Domain 9: Sensory (25 concepts) ──
    sensory = {
        "bright": (dict(brightness=1), 0.3, "Alert-Curious"),
        "dark": (dict(brightness=-1), 0.4, "Cautious-Alert"),
        "dim": (dict(brightness=-0.5), 0.3, "Cautious-Alert"),
        "radiant": (dict(brightness=1.2), 0.5, "Curious-Bold"),
        "shadowy": (dict(brightness=-0.7), 0.5, "Cautious-Alert"),
        "loud": (dict(brightness=0.5, strength=0.5), 0.5, "Alert-Stressed"),
        "quiet": (dict(brightness=-0.3, strength=-0.5), 0.1, "Calm-Relaxed"),
        "silent": (dict(brightness=-0.5, strength=-0.8), 0.05, "Calm-Relaxed"),
        "noisy": (dict(strength=0.5, stress=0.3), 0.5, "Alert-Stressed"),
        "rough": (dict(spring=0.5, friction=0.3), 0.3, "Alert-Cautious"),
        "smooth": (dict(friction=-0.3), 0.1, "Calm-Relaxed"),
        "soft": (dict(spring=-0.5, stress=-0.3), 0.1, "Calm-Relaxed"),
        "hard": (dict(spring=-0.5, weight=0.5), 0.3, "Calm-Confident"),
        "sharp": (dict(threat=0.5, force=0.8), 0.5, "Cautious-Alert"),
        "dull": (dict(threat=-0.3, force=0.3), 0.2, "Calm-Confident"),
        "warm": (dict(stress=-0.2), 0.2, "Calm-Relaxed"),
        "cold": (dict(threat=0.3), 0.3, "Alert-Cautious"),
        "hot": (dict(threat=0.5, collision=0.3), 0.6, "Alert-Cautious"),
        "wet": (dict(friction=-0.3, buoyancy=0.3), 0.3, "Alert-Cautious"),
        "dry": (dict(friction=0.3), 0.1, "Calm-Relaxed"),
        "sweet": (dict(stress=-0.3), 0.3, "Calm-Relaxed"),
        "bitter": (dict(stress=0.3), 0.4, "Cautious-Alert"),
        "fragrant": (dict(curiosity=0.3), 0.3, "Curious-Confident"),
        "pungent": (dict(curiosity=0.5, stress=0.3), 0.5, "Alert-Cautious"),
        "fresh": (dict(stress=-0.3, curiosity=0.3), 0.2, "Calm-Confident"),
    }

    # ── Domain 10: Abstract Relations (25 concepts) ──
    relations = {
        "part": (dict(size=-0.3, confidence=0.3), 0.2, "Calm-Confident"),
        "whole": (dict(size=0.5, confidence=0.5), 0.3, "Calm-Confident"),
        "piece": (dict(size=-0.5), 0.2, "Calm-Confident"),
        "group": (dict(proximity=0.5, confidence=0.3), 0.2, "Calm-Confident"),
        "set": (dict(proximity=0.3, confidence=0.5), 0.2, "Calm-Confident"),
        "type": (dict(confidence=0.5), 0.2, "Calm-Confident"),
        "kind": (dict(confidence=0.4), 0.2, "Calm-Confident"),
        "category": (dict(confidence=0.5), 0.2, "Calm-Confident"),
        "example": (dict(confidence=0.4, curiosity=0.3), 0.3, "Curious-Confident"),
        "similar": (dict(confidence=0.5), 0.3, "Calm-Confident"),
        "different": (dict(curiosity=0.6, confidence=-0.3), 0.4, "Curious-Uncertain"),
        "same": (dict(confidence=0.8), 0.2, "Calm-Confident"),
        "opposite": (dict(curiosity=0.5, confidence=-0.5), 0.4, "Curious-Uncertain"),
        "like": (dict(confidence=0.3, curiosity=0.3), 0.3, "Calm-Confident"),
        "unlike": (dict(curiosity=0.5, confidence=-0.3), 0.4, "Curious-Uncertain"),
        "equal": (dict(confidence=0.7), 0.2, "Calm-Confident"),
        "unequal": (dict(confidence=-0.3, curiosity=0.3), 0.4, "Uncertain-Curious"),
        "connected": (dict(proximity=0.7), 0.3, "Calm-Confident"),
        "separate": (dict(proximity=-0.5), 0.3, "Calm-Confident"),
        "joined": (dict(proximity=0.8), 0.3, "Calm-Confident"),
        "apart": (dict(proximity=-0.7), 0.2, "Calm-Confident"),
        "together": (dict(proximity=0.8, confidence=0.3), 0.3, "Calm-Confident"),
        "alone": (dict(proximity=-0.8), 0.3, "Cautious-Alert"),
        "linked": (dict(proximity=0.6, confidence=0.3), 0.2, "Calm-Confident"),
        "isolated": (dict(proximity=-1, stress=0.3), 0.5, "Cautious-Alert"),
    }

    # ── Domain 11: Cognitive (20 concepts) ──
    cognitive = {
        "think": (dict(curiosity=0.4, confidence=0.3), 0.3, "Curious-Confident"),
        "know": (dict(confidence=0.9, curiosity=-0.3), 0.2, "Confident-Calm"),
        "learn": (dict(curiosity=0.7, confidence=0.2), 0.5, "Curious-Confident"),
        "remember": (dict(time=-0.5, confidence=0.5), 0.4, "Calm-Confident"),
        "forget": (dict(time=-0.3, confidence=-0.5), 0.5, "Uncertain-Cautious"),
        "understand": (dict(confidence=0.8, curiosity=-0.3), 0.3, "Confident-Calm"),
        "believe": (dict(confidence=0.6), 0.2, "Calm-Confident"),
        "doubt": (dict(confidence=-0.6, curiosity=0.3), 0.5, "Uncertain-Cautious"),
        "notice": (dict(curiosity=0.7, brightness=0.3), 0.5, "Curious-Alert"),
        "observe": (dict(curiosity=0.5, confidence=0.3), 0.3, "Curious-Confident"),
        "imagine": (dict(curiosity=0.8, confidence=-0.2), 0.5, "Curious-Bold"),
        "predict": (dict(time=0.5, confidence=0.3), 0.4, "Confident-Alert"),
        "expect": (dict(time=0.3, confidence=0.5), 0.3, "Confident-Calm"),
        "suppose": (dict(curiosity=0.5, confidence=-0.3), 0.4, "Uncertain-Curious"),
        "wonder": (dict(curiosity=0.8, confidence=-0.3), 0.6, "Curious-Uncertain"),
        "realize": (dict(curiosity=0.5, confidence=0.7), 0.6, "Confident-Alert"),
        "recognize": (dict(confidence=0.8, curiosity=-0.3), 0.4, "Confident-Calm"),
        "identify": (dict(confidence=0.7, curiosity=0.3), 0.4, "Confident-Alert"),
        "discover": (dict(curiosity=1, confidence=0.5), 0.8, "Curious-Bold"),
        "invent": (dict(curiosity=0.9, confidence=0.7), 0.7, "Curious-Bold"),
        "emerge": (dict(curiosity=0.6, confidence=0.3), 0.5, "Curious-Alert"),
        "repeated": (dict(frequency=0.8, confidence=0.5), 0.2, "Calm-Confident"),
    }

    # ── Domain 12: Quantifiers (15 concepts) ──
    quantifiers = {
        "all": (dict(size=1, confidence=0.8), 0.2, "Calm-Confident"),
        "none": (dict(size=-1, confidence=0.5), 0.2, "Calm-Confident"),
        "some": (dict(size=0.3, confidence=0.3), 0.1, "Calm-Confident"),
        "many": (dict(size=0.8), 0.2, "Calm-Confident"),
        "few": (dict(size=-0.5), 0.1, "Calm-Confident"),
        "most": (dict(size=0.8, confidence=0.5), 0.2, "Calm-Confident"),
        "several": (dict(size=0.3), 0.1, "Calm-Confident"),
        "each": (dict(size=0.2, confidence=0.5), 0.2, "Calm-Confident"),
        "every": (dict(size=1, confidence=0.8), 0.2, "Calm-Confident"),
        "any": (dict(size=0.3, confidence=0.3), 0.1, "Calm-Confident"),
        "much": (dict(size=0.7), 0.2, "Calm-Confident"),
        "little": (dict(size=-0.7), 0.1, "Calm-Relaxed"),
        "one": (dict(size=-0.5, confidence=0.5), 0.2, "Calm-Confident"),
        "two": (dict(size=-0.3), 0.1, "Calm-Confident"),
        "many": (dict(size=0.8), 0.2, "Calm-Confident"),
    }

    # ── Domain 13: Conditionals (15 concepts) ──
    conditionals = {
        "if": (dict(curiosity=0.5, confidence=-0.2), 0.4, "Curious-Cautious"),
        "when": (dict(time=0, curiosity=0.3), 0.3, "Alert-Cautious"),
        "unless": (dict(curiosity=0.5, threat=0.2), 0.4, "Cautious-Alert"),
        "until": (dict(time=0.5, urgency=0.3), 0.3, "Alert-Cautious"),
        "while": (dict(time=0, frequency=0.5), 0.2, "Calm-Confident"),
        "because": (dict(confidence=0.5, curiosity=0.3), 0.4, "Confident-Curious"),
        "therefore": (dict(confidence=0.6, time=0.3), 0.3, "Confident-Calm"),
        "so": (dict(confidence=0.5, time=0.3), 0.3, "Confident-Calm"),
        "but": (dict(curiosity=0.3, confidence=-0.2), 0.3, "Cautious-Alert"),
        "however": (dict(curiosity=0.3, confidence=-0.2), 0.3, "Cautious-Alert"),
        "although": (dict(curiosity=0.3, confidence=-0.3), 0.4, "Cautious-Alert"),
        "might": (dict(confidence=-0.5, curiosity=0.4), 0.3, "Uncertain-Curious"),
        "should": (dict(confidence=0.5, urgency=0.3), 0.3, "Confident-Alert"),
        "must": (dict(confidence=0.8, urgency=0.8), 0.4, "Confident-Bold"),
        "could": (dict(confidence=-0.3), 0.2, "Uncertain-Cautious"),
    }

    # ── Domain 14: Composition/Explanation (20 concepts) ──
    composition = {
        "first": (dict(time=-1, urgency=0.3), 0.3, "Alert-Confident"),
        "then": (dict(time=0.5), 0.2, "Calm-Confident"),
        "finally": (dict(time=1, urgency=0.3), 0.3, "Confident-Calm"),
        "step": (dict(time=0.3, locomotion=0.3), 0.3, "Calm-Confident"),
        "plan": (dict(confidence=0.5, time=-0.3), 0.4, "Confident-Alert"),
        "goal": (dict(landmark=0.8, confidence=0.5), 0.5, "Confident-Alert"),
        "purpose": (dict(confidence=0.6, landmark=0.5), 0.4, "Confident-Alert"),
        "result": (dict(time=0.5, confidence=0.4), 0.3, "Confident-Calm"),
        "effect": (dict(time=0.5, force=0.3), 0.4, "Confident-Alert"),
        "cause": (dict(time=-0.5, force=0.5), 0.4, "Confident-Alert"),
        "reason": (dict(curiosity=0.5, confidence=0.3), 0.4, "Curious-Confident"),
        "means": (dict(confidence=0.5), 0.3, "Confident-Calm"),
        "method": (dict(confidence=0.5, time=0), 0.3, "Confident-Calm"),
        "way": (dict(path=0.3, confidence=0.3), 0.2, "Calm-Confident"),
        "process": (dict(time=0, frequency=0.5), 0.3, "Calm-Confident"),
        "sequence": (dict(time=0.5, frequency=0.5), 0.3, "Calm-Confident"),
        "order": (dict(time=0.3, confidence=0.5), 0.3, "Confident-Calm"),
        "example": (dict(confidence=0.4, curiosity=0.3), 0.3, "Curious-Confident"),
        "instance": (dict(confidence=0.3, curiosity=0.2), 0.2, "Calm-Confident"),
        "pattern": (dict(frequency=0.7, confidence=0.5), 0.4, "Confident-Alert"),
    }

    # ── Domain 15: Objects (25 concepts) ──
    objects = {
        "ball": (dict(size=-0.3, spring=0.3), 0.3, "Calm-Confident"),
        "block": (dict(size=0.3, weight=0.3), 0.3, "Calm-Confident"),
        "box": (dict(size=0.5, enclosure=0.3), 0.2, "Calm-Confident"),
        "cube": (dict(size=0.3, confidence=0.5), 0.2, "Calm-Confident"),
        "sphere": (dict(size=0.3, rotation=0.3), 0.3, "Curious-Confident"),
        "cylinder": (dict(size=0.3, rotation=0.2), 0.2, "Calm-Confident"),
        "object": (dict(size=0.3), 0.2, "Calm-Confident"),
        "thing": (dict(size=0.2, confidence=0.2), 0.2, "Calm-Confident"),
        "item": (dict(size=0.2, confidence=0.3), 0.2, "Calm-Confident"),
        "tool": (dict(size=0.3, grip=0.3), 0.4, "Confident-Alert"),
        "container": (dict(size=0.5, enclosure=0.8), 0.3, "Calm-Confident"),
        "vehicle": (dict(size=0.8, locomotion=0.8), 0.5, "Alert-Confident"),
        "machine": (dict(size=0.5, force=0.5), 0.4, "Alert-Confident"),
        "device": (dict(size=0.2, force=0.3), 0.4, "Confident-Alert"),
        "target": (dict(landmark=0.8, curiosity=0.5), 0.5, "Curious-Alert"),
        "goal": (dict(landmark=0.8, confidence=0.5), 0.5, "Confident-Alert"),
        "person": (dict(size=0.5, curiosity=0.3), 0.4, "Curious-Alert"),
        "robot": (dict(size=0.3, locomotion=0.5), 0.5, "Alert-Confident"),
        "creature": (dict(size=0.3, curiosity=0.5), 0.5, "Curious-Alert"),
        "animal": (dict(size=0.3, curiosity=0.6), 0.5, "Curious-Alert"),
        "food": (dict(size=-0.3, stress=-0.3), 0.4, "Calm-Relaxed"),
        "water": (dict(buoyancy=0.5), 0.2, "Calm-Relaxed"),
        "fire": (dict(threat=0.5, brightness=1), 0.8, "Alert-Cautious"),
        "light": (dict(brightness=1), 0.3, "Alert-Curious"),
        "shadow": (dict(brightness=-0.7), 0.4, "Cautious-Alert"),
    }

    # Combine all domains
    all_domains = {
        'physics': physics,
        'materials': materials,
        'navigation': navigation,
        'emotion': emotion,
        'action': action,
        'spatial': spatial,
        'temporal': temporal,
        'magnitude': magnitude,
        'sensory': sensory,
        'relations': relations,
        'cognitive': cognitive,
        'quantifiers': quantifiers,
        'conditionals': conditionals,
        'composition': composition,
        'objects': objects,
    }

    # Generate 8-12 variations per word
    for domain_name, domain_words in all_domains.items():
        for word, info in domain_words.items():
            if len(info) == 4:
                dims, da, mood, physics = info
            else:
                dims, da, mood = info
                physics = ""
            n_variations = rng.randint(8, 13)
            for _ in range(n_variations):
                belief = bf.make(**dims)
                da_var = da + rng.randn() * 0.1
                curriculum.append((word, belief, max(0, da_var),
                                   mood, physics, domain_name))

    rng.shuffle(curriculum)
    return curriculum, all_domains


# ──────────────────────────────────────────────────────────────────────────────
# Overnight Training Loop
# ──────────────────────────────────────────────────────────────────────────────

class OvernightTrainer:
    """Trains language system for hours with checkpoints."""

    def __init__(self, d_belief=D_BELIEF, resume=False):
        self.d_belief = d_belief
        self.grounder = WordGrounder(d_belief=d_belief)
        self.accumulator = BeliefAccumulator(d_belief=d_belief)
        self.modifier = ModifierProcessor()
        self.contrastive = ContrastiveGrounder(self.grounder)
        self.replay = EpisodicReplayLearner(self.grounder)
        self.comprehender = SentenceComprehender(self.grounder)

        self.metrics = {
            'total_pass_count': 0,
            'total_hearings': 0,
            'vocab_trajectory': [],
            'comprehension_trajectory': [],
            'sleep_cycles': 0,
            'start_time': time.time(),
            'checkpoint_times': [],
        }

        if resume:
            self.load_checkpoint()

    def save_checkpoint(self):
        """Save grounder state."""
        state = {
            'vocabulary': dict(self.grounder.vocabulary),
            'hearings': self.grounder.hearings,
            'metrics': self.metrics,
        }
        path = CHECKPOINT_DIR / f"checkpoint_latest.pkl"
        with open(path, 'wb') as f:
            pickle.dump(state, f)

        # Also timestamped backup
        ts = datetime.now().strftime("%H%M")
        with open(CHECKPOINT_DIR / f"checkpoint_{ts}.pkl", 'wb') as f:
            pickle.dump(state, f)

        self.metrics['checkpoint_times'].append(time.time())

    def load_checkpoint(self):
        """Resume from last checkpoint."""
        path = CHECKPOINT_DIR / "checkpoint_latest.pkl"
        if not path.exists():
            print("  No checkpoint found, starting fresh")
            return False
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.grounder.vocabulary = defaultdict(list, state['vocabulary'])
        self.grounder.hearings = state['hearings']
        self.metrics = state['metrics']
        print(f"  Resumed: vocab={self.grounder.vocab_size}, "
              f"hearings={self.grounder.hearings}")
        return True

    def train_pass(self, curriculum, pass_num, verbose=True):
        """One pass through the curriculum."""
        if verbose:
            print(f"\n-- Pass {pass_num} ({len(curriculum)} examples) --")

        for i, (word, belief, da, mood, physics, domain) in enumerate(curriculum):
            self.grounder.hear(word, belief, da=da, mood=mood,
                                physics=physics,
                                source=f"pass{pass_num}_{domain}")

        self.metrics['total_pass_count'] += 1
        self.metrics['total_hearings'] = self.grounder.hearings

    def sleep_consolidation(self, n_episodes=200, verbose=True):
        """Replay and narrate memories to grow vocab."""
        if verbose:
            print(f"-- Sleep cycle {self.metrics['sleep_cycles'] + 1} "
                  f"({n_episodes} replays) --")

        # Sample beliefs from vocabulary
        all_beliefs = []
        for word, experiences in self.grounder.vocabulary.items():
            for exp in experiences[:3]:  # sample from each word
                all_beliefs.append((exp['belief'], exp['da'], exp['mood']))

        if len(all_beliefs) < 10:
            return

        rng = np.random.RandomState(int(time.time()))
        rng.shuffle(all_beliefs)

        beliefs = [b for b, d, m in all_beliefs[:n_episodes]]
        das = [d for b, d, m in all_beliefs[:n_episodes]]
        moods = [m for b, d, m in all_beliefs[:n_episodes]]

        self.replay.replay_and_learn(
            beliefs, das, moods, n_episodes=len(beliefs), verbose=verbose)

        self.metrics['sleep_cycles'] += 1

    def test_comprehension(self):
        """Test on increasingly complex sentences."""
        test_sentences = [
            # Simple
            "the ball falls",
            "push left",
            "danger ahead",
            # Compound
            "the ball falls due to gravity",
            "push the heavy block to the left",
            "danger on the steep dark corridor",
            # Complex
            "if the path is steep then move slowly because of gravity",
            "imagine pushing the heavy block left because the corridor is blocked",
            "remember that friction stops rolling objects on rough surfaces",
            # Multi-clause
            "first observe the obstacle then plan the safe route around it",
            "when you see danger move slowly and be careful",
            "the target is far ahead beyond the closed door near the corner",
            # Abstract
            "gravity is a type of force that pulls objects down",
            "understanding causes comes from observing effects",
            "patterns emerge from repeated examples",
        ]

        scores = []
        for sent in test_sentences:
            result = self.comprehender.comprehend(sent)
            scores.append(result['confidence'])

        avg = float(np.mean(scores))
        return avg, scores, test_sentences

    def run_overnight(self, hours=10, save_every_minutes=30):
        """Main training loop."""
        print("=" * 65)
        print(f"  NeMo-WM Overnight Language Training")
        print(f"  Duration: {hours} hours")
        print(f"  Save every: {save_every_minutes} min")
        print(f"  Started: {datetime.now().strftime('%H:%M:%S')}")
        print(f"  Expected end: "
              f"{(datetime.now() + timedelta(hours=hours)).strftime('%H:%M:%S')}")
        print("=" * 65)

        # Build curriculum
        print("\n-- Building extended curriculum --")
        curriculum, all_domains = build_extended_curriculum(
            d_belief=self.d_belief)
        print(f"  Total examples: {len(curriculum)}")
        print(f"  Domains: {len(all_domains)}")
        for name, words in all_domains.items():
            print(f"    {name}: {len(words)} concepts")

        # Build contrastive meanings
        print("\n-- Building contrastive meanings --")
        self.contrastive.build_contrasts()

        end_time = time.time() + hours * 3600
        last_save = time.time()
        last_test = time.time()
        pass_num = 0

        while time.time() < end_time:
            pass_num += 1
            print(f"\n{'='*65}")
            elapsed = (time.time() - self.metrics['start_time']) / 3600
            remaining = (end_time - time.time()) / 3600
            print(f"  Pass {pass_num}  "
                  f"elapsed: {elapsed:.1f}h  remaining: {remaining:.1f}h")
            print(f"  Vocab: {self.grounder.vocab_size}  "
                  f"Hearings: {self.grounder.hearings}")
            print(f"{'='*65}")

            # Train on curriculum
            self.train_pass(curriculum, pass_num, verbose=True)

            # Sleep consolidation every 3 passes
            if pass_num % 3 == 0:
                self.sleep_consolidation(n_episodes=300)

            # Rebuild contrastive meanings periodically
            if pass_num % 5 == 0:
                print("-- Refreshing contrastive meanings --")
                self.contrastive.build_contrasts()

            # Test comprehension every 5 passes
            if pass_num % 5 == 0:
                avg, scores, sents = self.test_comprehension()
                self.metrics['comprehension_trajectory'].append({
                    'pass': pass_num,
                    'avg': avg,
                    'time': time.time() - self.metrics['start_time'],
                })
                print(f"-- Comprehension test: avg confidence = {avg:.1%}")
                best_idx = int(np.argmax(scores))
                worst_idx = int(np.argmin(scores))
                print(f"   Best:  '{sents[best_idx][:50]}' "
                      f"({scores[best_idx]:.0%})")
                print(f"   Worst: '{sents[worst_idx][:50]}' "
                      f"({scores[worst_idx]:.0%})")

            self.metrics['vocab_trajectory'].append({
                'pass': pass_num,
                'vocab': self.grounder.vocab_size,
                'time': time.time() - self.metrics['start_time'],
            })

            # Save checkpoint
            if time.time() - last_save > save_every_minutes * 60:
                self.save_checkpoint()
                print(f"-- Checkpoint saved at "
                      f"{datetime.now().strftime('%H:%M:%S')}")
                last_save = time.time()

            # Shuffle curriculum for next pass
            rng = np.random.RandomState(int(time.time()))
            rng.shuffle(curriculum)

        # Final save
        self.save_checkpoint()

        # Final report
        self.final_report()

    def final_report(self):
        """Print comprehensive final report."""
        print(f"\n{'='*65}")
        print(f"  Overnight Training Complete")
        print(f"{'='*65}")

        elapsed = (time.time() - self.metrics['start_time']) / 3600
        print(f"\n  Duration: {elapsed:.2f} hours")
        print(f"  Total passes: {self.metrics['total_pass_count']}")
        print(f"  Total hearings: {self.grounder.hearings}")
        print(f"  Sleep cycles: {self.metrics['sleep_cycles']}")
        print(f"  Vocabulary: {self.grounder.vocab_size} words")

        avg_exp = self.grounder.hearings / max(self.grounder.vocab_size, 1)
        print(f"  Avg experiences/word: {avg_exp:.1f}")

        # Final comprehension test
        print(f"\n-- Final Comprehension Test --")
        avg, scores, sents = self.test_comprehension()
        print(f"  Average confidence: {avg:.0%}")

        for sent, score in zip(sents, scores):
            status = "OK" if score > 0.5 else "WEAK" if score > 0.3 else "FAIL"
            short = sent[:55] + '..' if len(sent) > 55 else sent
            print(f"  [{status:4}] {score:>4.0%}  {short}")

        # Cognitive age estimate
        if avg > 0.85:
            age = "6-8 years (adult-like comprehension)"
        elif avg > 0.75:
            age = "4-6 years (story comprehension + teaching)"
        elif avg > 0.6:
            age = "3-4 years (multi-step + explanation)"
        elif avg > 0.5:
            age = "2-3 years (conditionals + analogy)"
        else:
            age = "18-24 months (spatial + temporal)"

        print(f"\n  Cognitive age equivalent: {age}")

        # Save final metrics
        with open(CHECKPOINT_DIR / "final_metrics.json", 'w') as f:
            metrics_copy = {k: v for k, v in self.metrics.items()
                            if not isinstance(v, list) or len(v) < 1000}
            # Convert non-serializable values
            metrics_copy['elapsed_hours'] = elapsed
            metrics_copy['vocab_size'] = self.grounder.vocab_size
            metrics_copy['avg_comprehension'] = avg
            metrics_copy['cognitive_age'] = age
            json.dump(metrics_copy, f, indent=2, default=str)

        print(f"\n  Metrics saved: {CHECKPOINT_DIR / 'final_metrics.json'}")
        print(f"  Checkpoint: {CHECKPOINT_DIR / 'checkpoint_latest.pkl'}")
        print(f"{'='*65}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=10.0,
                    help="Training duration in hours")
    ap.add_argument("--save-every", type=int, default=30,
                    help="Save checkpoint every N minutes")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from last checkpoint")
    args = ap.parse_args()

    trainer = OvernightTrainer(resume=args.resume)
    trainer.run_overnight(hours=args.hours,
                           save_every_minutes=args.save_every)


if __name__ == "__main__":
    main()
