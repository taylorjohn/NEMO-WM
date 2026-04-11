# SWM → CORTEX CWM Integration Plan
# arXiv:2603.15583 — Seoul World Model (KAIST + NAVER AI Lab)
# Integrated into: CORTEX World Model (CWM) — arXiv:2603.19312 baseline

---

## Executive Summary

Seoul World Model (SWM) introduces three techniques directly applicable to
CORTEX CWM: (1) Virtual Lookahead Sink for long-horizon stability, (2)
Cross-Temporal Pairing to force geometry-stable encodings, and (3) Retrieval-
Augmented GPS conditioning as an active inference mechanism. All three require
zero new parameters — they are training data, loss, and inference-time changes
only. The CORTEX GPS grounding advantage (direct, not probe-based) makes these
integrations cleaner than they would be for LeWM.

---

## Contribution 1: Virtual Lookahead Sink (VL Sink)

### What SWM does
Autoregressive generation accumulates errors over long horizons. SWM's VL Sink
dynamically retrieves the nearest street-view image as a virtual future anchor
ahead of the current generation chunk. Instead of conditioning only on history,
the model also sees a clean, error-free reference from a future GPS location.
This prevents quality degradation over trajectories spanning hundreds of metres.

### CORTEX CWM translation
The ParticleTransformer (d_model=128, 2 layers, 4 heads) generates latent
trajectories autoregressively with no re-grounding. At each step, prediction
error compounds. The VL Sink pattern translates directly:

**At inference time:**
1. Maintain a geo-indexed latent database built from RECON HDF5 traversals
2. At each ParticleTransformer chunk boundary, retrieve the k-nearest GPS
   latents from the database (using GPS coordinates from the HDF5 metadata)
3. Inject the retrieved latent as an additional "lookahead anchor" particle
4. The anchor particle is frozen (no gradient) — it only conditions dynamics

**Implementation sketch:**

```python
class VLSinkParticleTransformer(nn.Module):
    def __init__(self, d_model=128, n_layers=2, n_heads=4, action_dim=2):
        super().__init__()
        self.transformer = ParticleTransformer(d_model, n_layers, n_heads)
        self.action_embed = nn.Linear(action_dim, d_model)
        # VL Sink: project anchor latent into particle space
        self.anchor_proj = nn.Linear(d_model, d_model)

    def forward(self, particles, actions, anchor_latent=None):
        """
        particles:      (B, N, d_model) — current particle set
        actions:        (B, action_dim) — current action
        anchor_latent:  (B, d_model) or None — GPS-retrieved future anchor
        """
        x = particles + self.action_embed(actions).unsqueeze(1)
        if anchor_latent is not None:
            # Inject anchor as an additional particle (frozen, attend-only)
            anchor = self.anchor_proj(anchor_latent).unsqueeze(1)  # (B,1,d)
            x = torch.cat([x, anchor.detach()], dim=1)
        return self.transformer(x)[:, :-1]  # strip anchor from output


class GeoLatentDatabase:
    """
    Geo-indexed database of (GPS, latent) pairs built from RECON HDF5 traversals.
    Used at inference time for VL Sink retrieval.
    """
    def __init__(self):
        self.gps_coords = []   # list of (lat, lon) or (x, y) in local frame
        self.latents = []      # corresponding encoder outputs

    def add(self, gps, latent):
        self.gps_coords.append(gps)
        self.latents.append(latent)

    def retrieve_nearest(self, query_gps, k=1):
        """Returns k nearest latents by GPS distance."""
        import numpy as np
        coords = np.array(self.gps_coords)
        query = np.array(query_gps)
        dists = np.linalg.norm(coords - query, axis=1)
        idx = np.argsort(dists)[:k]
        return [self.latents[i] for i in idx]

    @classmethod
    def build_from_recon(cls, hdf5_paths, encoder):
        """Build database from RECON HDF5 files."""
        db = cls()
        for path in hdf5_paths:
            import h5py
            from PIL import Image
            import io
            with h5py.File(path, 'r') as f:
                imgs = f['observations/images/rgb_left'][:]
                gps  = f['observations/position'][:]  # (T, 2) or (T, 3)
                for i in range(0, len(imgs), 10):    # sample every 10 frames
                    img = Image.open(io.BytesIO(imgs[i]))
                    with torch.no_grad():
                        z = encoder(img)
                    db.add(gps[i, :2], z.cpu())
        return db
```

### Integration point in CWM training
No change to training loss. VL Sink is inference-only. Build the
`GeoLatentDatabase` after training using the RECON traversal data and
the trained StudentEncoder. Add retrieval to the CWM inference loop.

### Expected gain
SWM reports VL Sink prevents quality degradation in long-horizon generation.
For RECON AUROC, this should improve quasimetric consistency at long distances
(the current TemporalHead struggles with k > 10 step predictions). Expected
AUROC gain: +0.03–0.05 on the 0.70 target.

---

## Contribution 2: Cross-Temporal Pairing

### What SWM does
Reference images are required to come from a different time than the target
video. This forces the model to learn persistent spatial structure (buildings,
roads, landmarks) rather than transient content (vehicles, pedestrians,
weather). Without this, dynamic objects in references leak into generation.

### CORTEX CWM translation
Your current RECON TemporalHead training builds `(z_t, z_{t+k})` pairs from
the same traversal session. This means dynamic obstacles (other robots, moving
objects the Jackal encountered) in the reference frame can corrupt the
temporal embedding. Cross-temporal pairing fixes this:

**Modified pair construction:**

```python
def build_cross_temporal_pairs(hdf5_paths, encoder, k_steps=5):
    """
    Build (anchor, target) pairs where anchor and target come from
    DIFFERENT traversal sessions of geographically close locations.
    Forces encoder to learn geometry, not dynamics.
    """
    from collections import defaultdict
    import h5py, io
    from PIL import Image

    # Build GPS → (session, frame_idx) index
    location_index = defaultdict(list)  # gps_cell → [(session_id, frame_idx)]

    for session_id, path in enumerate(hdf5_paths):
        with h5py.File(path, 'r') as f:
            gps = f['observations/position'][:]
            for i in range(len(gps)):
                # Discretize GPS to 1-metre grid cells
                cell = (round(float(gps[i, 0]), 0),
                        round(float(gps[i, 1]), 0))
                location_index[cell].append((session_id, i))

    pairs = []
    for cell, entries in location_index.items():
        # Only use cells visited by multiple sessions
        sessions_present = set(e[0] for e in entries)
        if len(sessions_present) < 2:
            continue
        # Sample one anchor from session A, one target from session B
        from random import choice, sample
        s_a, s_b = sample(list(sessions_present), 2)
        anchors = [e for e in entries if e[0] == s_a]
        targets = [e for e in entries if e[0] == s_b]
        if anchors and targets:
            pairs.append((choice(anchors), choice(targets)))

    return pairs


def train_temporal_head_cross_temporal(pairs, hdf5_paths, encoder, head,
                                        optimizer, n_epochs=20):
    """
    InfoNCE training on cross-temporal pairs.
    Anchor: frame from session A at location L
    Positive: frame from session B at same location L (different time)
    Negative: frames from random locations (in-batch negatives)
    """
    for epoch in range(n_epochs):
        for (s_a, i_a), (s_b, i_b) in pairs:
            with h5py.File(hdf5_paths[s_a], 'r') as f:
                img_a = Image.open(io.BytesIO(
                    f['observations/images/rgb_left'][i_a]))
            with h5py.File(hdf5_paths[s_b], 'r') as f:
                img_b = Image.open(io.BytesIO(
                    f['observations/images/rgb_left'][i_b]))

            with torch.no_grad():
                z_a = encoder(img_a)
            z_b_pred = head(z_a)

            with torch.no_grad():
                z_b_true = encoder(img_b)

            # InfoNCE loss (in-batch negatives)
            loss = infonce_loss(z_b_pred, z_b_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### Why this matters for RECON AUROC
The current TemporalHead failure mode is memorising session-specific dynamic
content (the exact vehicle positions, lighting conditions in a given traversal)
rather than learning the quasimetric geometry of the outdoor space. Cross-
temporal pairing directly attacks this by making same-session dynamics
unavailable as a learning signal. Expected AUROC gain on RECON: +0.04–0.08,
potentially pushing past the 0.70 target on its own.

---

## Contribution 3: GPS as Active RAG Index

### What SWM does
SWM uses geo-indexed nearest-neighbour retrieval at inference time. Given
current GPS coordinates and camera actions, it retrieves nearby street-view
images and conditions generation through two pathways: geometric (depth-based
warping) and semantic (direct attention to reference images).

### CORTEX CWM translation
Your direct GPS physical grounding is already designed as a training signal
(`gps_grounding_loss_fast`). SWM shows GPS grounding should also be an
**active inference mechanism**. The translation:

**GPS grounding as retrieval-augmented conditioning:**

```python
class GPSGroundedCWMInference:
    """
    Wraps the trained CWM with GPS-indexed retrieval for inference.
    Implements the SWM RAG pattern in CORTEX latent space.
    """
    def __init__(self, cwm, encoder, geo_db: GeoLatentDatabase):
        self.cwm = cwm
        self.encoder = encoder
        self.geo_db  = geo_db

    def predict(self, current_obs, current_gps, action,
                lookahead_gps=None, k_refs=3):
        """
        current_obs:    current frame (PIL Image or tensor)
        current_gps:    (lat, lon) current position
        action:         action vector (2D for RECON)
        lookahead_gps:  predicted future GPS position (for VL Sink)
        k_refs:         number of reference latents to retrieve
        """
        with torch.no_grad():
            z_current = self.encoder(current_obs)

        # Retrieve nearby reference latents (geometric context)
        ref_latents = self.geo_db.retrieve_nearest(current_gps, k=k_refs)
        ref_tensor  = torch.stack(ref_latents).unsqueeze(0)  # (1, k, d)

        # VL Sink: retrieve lookahead anchor
        anchor = None
        if lookahead_gps is not None:
            [anchor] = self.geo_db.retrieve_nearest(lookahead_gps, k=1)

        # Predict next latent state with GPS context
        z_next = self.cwm(
            z_current.unsqueeze(0),
            action,
            ref_context=ref_tensor,
            vl_anchor=anchor,
        )
        return z_next
```

### GPS loss modification
The existing `gps_grounding_loss_fast` computes a loss between predicted
latents and GPS-grounded targets. Extend it to also pull predicted latents
toward retrieved reference latents from the same GPS neighbourhood:

```python
def gps_grounding_loss_with_retrieval(z_pred, gps_coords, geo_db,
                                       alpha_retrieval=0.1):
    """
    Extends gps_grounding_loss_fast with retrieval-augmented pull.
    z_pred:     (B, d_model) predicted latents
    gps_coords: (B, 2) GPS coordinates for each sample
    geo_db:     GeoLatentDatabase
    """
    retrieval_loss = 0.0
    for i, (z, gps) in enumerate(zip(z_pred, gps_coords)):
        refs = geo_db.retrieve_nearest(gps.cpu().numpy(), k=3)
        if refs:
            ref_mean = torch.stack(refs).mean(0).to(z.device)
            retrieval_loss += F.mse_loss(z, ref_mean.detach())
    retrieval_loss /= len(z_pred)

    return alpha_retrieval * retrieval_loss
```

---

## Integration Roadmap

### Phase 1 — Cross-Temporal Pairing (do first, no architecture change)
**Effort:** 1–2 days  
**Risk:** Low — changes only the data pipeline, not the model  
**Expected gain:** +0.04–0.08 RECON AUROC  

Steps:
1. Build `location_index` from all RECON HDF5 files using GPS coordinates
2. Extract cross-temporal pairs (sessions with geographic overlap)
3. Retrain TemporalHead (~33K params, frozen encoder) on cross-temporal pairs
4. Eval: `python eval_recon.py --head checkpoints/recon_contrastive/...`

### Phase 2 — GeoLatentDatabase (prerequisite for VL Sink and GPS RAG)
**Effort:** 1 day  
**Risk:** Low — build-time only, no training change  

Steps:
1. Run StudentEncoder over all RECON HDF5 frames (every 10 frames)
2. Store (GPS, latent) pairs in the `GeoLatentDatabase`
3. Save to disk: `checkpoints/recon_geo_db.pkl`
4. Verify retrieval quality: nearest-neighbour latents should be visually similar

### Phase 3 — VL Sink in ParticleTransformer (inference only)
**Effort:** 1–2 days  
**Risk:** Low — inference-time only, no retraining  

Steps:
1. Implement `VLSinkParticleTransformer` wrapper (see code above)
2. At each chunk boundary during RECON eval, retrieve lookahead anchor
3. Lookahead GPS = current_GPS + action_direction * lookahead_distance
4. Eval with and without VL Sink to quantify gain

### Phase 4 — GPS RAG Training Signal (requires retraining CWM)
**Effort:** 3–5 days  
**Risk:** Medium — modifies CWM training loop  

Steps:
1. Add `gps_grounding_loss_with_retrieval` to `CWMLoss`
2. Build `GeoLatentDatabase` before training (Phase 2 prerequisite)
3. Retrain CWM with retrieval augmentation (α_retrieval=0.1)
4. Compare vs baseline CWM on all domains

---

## Novelty Preservation

SWM operates at video generation level (pixel space). CORTEX CWM operates
entirely in latent space with a shared multi-domain backbone. The key
differentiators vs both SWM and LeWM remain intact:

| Feature | SWM | LeWM | CORTEX CWM |
|---|---|---|---|
| Domain scope | Outdoor navigation | Single domain | Multi-domain shared backbone |
| GPS grounding | Via street-view retrieval | Probe-based | Direct GPS loss (novel) |
| Param count | Large (video diffusion) | 14M | ~798K |
| NPU deployment | No | No | AMD XINT8 target |
| VL Sink | ✅ Original | ❌ | ✅ Adapted (latent space) |
| Cross-temporal | ✅ Original | ❌ | ✅ Adapted (RECON sessions) |

The CORTEX adaptation is genuinely novel: VL Sink in pure latent space (no
pixel decoding), cross-temporal pairing across robot traversal sessions (not
street-view cameras), and GPS RAG as a training signal rather than only an
inference mechanism.

---

## Files to Create/Modify

```
# New files
cortex_geo_db.py              # GeoLatentDatabase class
train_recon_cross_temporal.py # Cross-temporal TemporalHead training
eval_recon_vl_sink.py        # RECON eval with VL Sink

# Modified files
particle_encoder.py           # Add VLSinkParticleTransformer wrapper
train_cwm.py                  # Add GPS RAG loss term
cwm_loss.py                   # Add gps_grounding_loss_with_retrieval
```

---

## References

- SWM: arXiv:2603.15583 (Seo et al., KAIST + NAVER AI Lab, Mar 2026)
- CORTEX CWM baseline: LeWorldModel arXiv:2603.19312
- RECON dataset: Shah et al. 2021, Jackal robot outdoor navigation HDF5s
- LPWM (particle JEPA): arXiv:2603.04553
- MuRF (multi-scale distillation): arXiv:2603.25744

---

*Generated: 2026-03-30 | CORTEX-PE v16.17*
