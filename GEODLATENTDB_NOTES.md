# GeoLatentDB — Design Notes & Expansion Plan
> Built: 2026-04-02 | Prototype: 803 entries (100 files)

## What it is
A GPS-to-particle embedding lookup table. Maps physical coordinates
(latitude/longitude) to NeMo-WM particle embeddings extracted from
the same location during training traversals. Enables goal-conditioned
navigation: give a GPS coordinate, retrieve the associated particle
state, use as GRASP planning target.

## Current prototype
- 803 entries from 100 RECON files
- Sampling: every 10th frame (2.5 second intervals)
- Embedding: 128-D mean-pooled particle (K=16 → mean → 128-D)
- Storage: JSON (suitable for prototype, switch to numpy/FAISS at scale)

## Expansion to full dataset
```powershell
# Run on all 11,835 files for full campus coverage
python -c "
# Same script but files = sorted(glob.glob(...))  # no [:100] limit
# Expected: ~94,000 entries at every 10th frame
# Or ~470,000 entries at every 2nd frame for finer resolution
"
```

## KD-tree query interface
```python
from scipy.spatial import KDTree
import json, numpy as np

db = json.load(open('geo_latent_db.json'))
gps   = np.array([e['gps'] for e in db])
parts = np.array([e['particle'] for e in db])

# Convert degrees to metres (approximate, good enough for campus scale)
METRES_PER_DEG = 111000
tree = KDTree(gps * METRES_PER_DEG)

def get_goal_particle(lat, lon, k=1):
    dists, idxs = tree.query([lat * METRES_PER_DEG,
                               lon * METRES_PER_DEG], k=k)
    if k == 1:
        return parts[idxs], float(dists) / METRES_PER_DEG
    return parts[idxs], dists / METRES_PER_DEG  # k nearest

def get_coverage_radius():
    # Check how dense the GPS coverage is
    dists, _ = tree.query(gps * METRES_PER_DEG, k=2)
    nearest = dists[:, 1]  # distance to nearest neighbour
    print(f"Median gap: {np.median(nearest):.1f}m")
    print(f"Max gap: {np.max(nearest):.1f}m")
    print(f"95th pct gap: {np.percentile(nearest, 95):.1f}m")
```

## Sprint 4 integration points

### 1. Goal specification
```python
# User provides GPS goal
goal_lat, goal_lon = 37.8719, -122.2585  # example: Sather Gate
goal_particle, dist = get_goal_particle(goal_lat, goal_lon)
print(f"Nearest DB entry: {dist*111000:.1f}m away")
```

### 2. GRASP planning
```python
# Use goal particle as GRASP target
action = grasp_planner.plan(
    current_particles=particles_t,
    target_particles=goal_particle,
    horizon=4, iters=3
)
```

### 3. Progress monitoring
```python
# Track distance to goal using GPS
current_lat, current_lon = gps_reading()
dist_to_goal = haversine(current_lat, current_lon, goal_lat, goal_lon)
if dist_to_goal < 5.0:  # within 5 metres
    print("Goal reached")
```

## Limitations of current prototype
- 803 entries may not cover full campus — some GPS coordinates will
  have no nearby entry. Need full dataset expansion for production.
- Mean-pooled particle loses per-particle diversity. Consider storing
  full (K, 128) particle set for richer goal specification.
- No temporal context stored — particle is instantaneous state,
  not trajectory. Consider storing THICK GRU context alongside.

## Relationship to paper
GPS grounding was confirmed by AIM probe (p=2.3e-3). GeoLatentDB
exploits that grounding — the fact that particles encode GPS displacement
means GPS-indexed particle retrieval should find semantically similar
states. This is tested empirically in Sprint 4 navigation eval.

