# NeMo-WM — Neuromodulator Extensions
> Date: 2026-04-04 | Status: Research plan + feasibility assessment

## Current seven signals (production)

| Signal | Gates | Biological function |
|--------|-------|-------------------|
| Dopamine (DA) | L_predict | Surprise amplification |
| Serotonin (5HT) | L_gaussian | Representation diversity |
| Norepinephrine (NE) | L_gps | Spatial grounding |
| Acetylcholine (ACh) | L_contact | Contact detection |
| Endocannabinoid (eCB) | L_skill | Context novelty |
| Adenosine (Ado) | L_fatigue | Fatigue regulation |
| E/I Balance | L_curvature | Overall arousal |

---

## Tier 1 — Immediate additions (days, high confidence)

### Oxytocin (familiarity signal)
**Biological basis:** Social bonding and familiarity recognition in mammals.
**Implementation:** Cosine similarity between current particle mean and
exponential moving average of recent particle means.

```python
class OxytocinSignal:
    def __init__(self, decay=0.99):
        self.ema_particles = None
        self.decay = decay

    def update(self, p_mean: torch.Tensor) -> float:
        if self.ema_particles is None:
            self.ema_particles = p_mean.detach().clone()
        familiarity = F.cosine_similarity(
            p_mean, self.ema_particles, dim=-1
        ).mean().item()
        self.ema_particles = (
            self.decay * self.ema_particles +
            (1 - self.decay) * p_mean.detach()
        )
        return float(torch.clamp(torch.tensor(familiarity), 0, 1))
```

**Effect:** High oxytocin (familiar terrain) → reduce L_n weight.
Low oxytocin (new territory) → amplify L_n. Robot learns harder
in novel environments and consolidates in known ones.

**Test:** Run probe on two conditions — trajectories from files seen
in training vs held-out files. Oxytocin should be higher for seen
trajectories. Simple two-sample t-test.

---

### DA Replay Buffer (high-surprise episode replay)
**Biological basis:** Hippocampal replay preferentially replays
high-dopamine (surprising) experiences during consolidation.

```python
class DAReplayBuffer:
    def __init__(self, capacity=1000, replay_alpha=0.6):
        self.buffer = []  # list of (frame_t, frame_t1, action, gps, da)
        self.capacity = capacity
        self.replay_alpha = replay_alpha  # priority exponent

    def push(self, batch, da_value: float):
        if len(self.buffer) >= self.capacity:
            # Remove lowest-DA entry
            self.buffer.sort(key=lambda x: x[-1])
            self.buffer.pop(0)
        self.buffer.append((*batch, da_value))

    def sample(self, n=16) -> list:
        # Prioritised sampling weighted by DA^alpha
        das = torch.tensor([b[-1] for b in self.buffer])
        weights = (das ** self.replay_alpha)
        weights /= weights.sum()
        idxs = torch.multinomial(weights, min(n, len(self.buffer)))
        return [self.buffer[i] for i in idxs]

    def should_replay(self, da_threshold=0.002) -> bool:
        return len(self.buffer) > 32
```

**Effect:** Steps where DA >= 0.002 (the threshold observed in Tab 2
final epoch) are stored and replayed in subsequent batches. The system
spends proportionally more gradient on genuinely surprising frames.

**Test:** Compare loss curves with and without replay buffer on
the same held-out RECON files. Expect faster convergence and higher
final AUROC with replay.

---

### ACh Router Temperature
**Biological basis:** Acetylcholine controls cortical signal-to-noise
ratio — high ACh sharpens processing, low ACh diffuses it.

```python
def ach_modulated_router(logits: torch.Tensor, ach: float,
                          base_temp: float = 1.0) -> torch.Tensor:
    """
    High ACh → lower temperature → sharper expert selection.
    Low ACh → higher temperature → softer routing.
    ACh range: 0.4–0.5 (observed). Map to temp range: 0.5–2.0.
    """
    temp = base_temp * (1.0 + (0.5 - ach) * 4.0)
    temp = max(0.3, min(3.0, temp))
    return F.softmax(logits / temp, dim=-1)
```

**Effect:** In familiar terrain (high ACh) the router sharpens expert
boundaries → more specialisation. In novel terrain (low ACh) routing
softens → more generalisation. Automatic domain adaptation.

**Test:** Log router entropy alongside ACh signal across a full epoch.
Expect negative correlation: higher ACh → lower router entropy
(sharper routing). Pearson r should be < -0.3 to confirm the effect.

---

## Tier 2 — Medium effort (1-2 weeks)

### Cortisol (sustained stress signal)
Slow-timescale signal tracking cumulative loss over last N=100 batches.

```python
class CortisolSignal:
    def __init__(self, window=100, baseline=0.57):
        self.loss_history = []
        self.window = window
        self.baseline = baseline  # expected loss at convergence

    def update(self, loss: float) -> float:
        self.loss_history.append(loss)
        if len(self.loss_history) > self.window:
            self.loss_history.pop(0)
        recent_mean = np.mean(self.loss_history)
        cortisol = max(0.0, (recent_mean - self.baseline) / self.baseline)
        return min(1.0, cortisol)

    def modulate(self, signals: dict, cortisol: float) -> dict:
        # High cortisol → re-anchor to GPS, forget skill habits
        signals['ne'] = min(1.0, signals['ne'] + 0.3 * cortisol)
        signals['ecb'] = max(0.0, signals['ecb'] - 0.3 * cortisol)
        return signals
```

**Test:** Inject a distribution shift mid-training (switch to different
RECON campus area). Cortisol should spike within 50 batches. Measure
time-to-cortisol-spike vs time-to-loss-spike.

---

### Histamine (motion-gated context update)
Gate THICK GRU update rate by robot motion magnitude.

```python
def histamine_signal(action: torch.Tensor) -> float:
    """
    High motion → high histamine → fast context updates.
    Stationary → low histamine → slow context decay.
    """
    motion = action.norm(dim=-1).mean().item()
    return float(torch.sigmoid(torch.tensor(motion * 5.0 - 1.0)))

def histamine_gated_gru(h_prev, x, histamine, gru_module):
    h_new = gru_module(x, h_prev)
    # Blend: high histamine → mostly new state
    #        low histamine → mostly old state (slow decay)
    h_blended = histamine * h_new + (1 - histamine) * h_prev
    return h_blended
```

**Test:** Compare temporal context coherence (cosine similarity of
successive h_t vectors) for moving vs stationary robot segments.
Expect higher coherence during stationary periods with histamine gating.

---

## Tier 3 — Research (months, architectural changes)

### BDNF — Selective weight decay
Neuromodulated regularisation: apply strong weight decay to idle
expert pathways, zero decay to frequently-used ones.

```python
def bdnf_weight_decay(model, routing_counts: dict, base_decay=1e-4):
    for expert_id, count in routing_counts.items():
        usage_rate = count / sum(routing_counts.values())
        # High usage → low decay (protect), low usage → high decay (prune)
        expert_decay = base_decay * (1.0 - usage_rate) * 10
        expert_params = model.predictor.layers[0].moe_ffn.experts[expert_id]
        for p in expert_params.parameters():
            p.data -= expert_decay * p.data
```

### Growth hormone — Dynamic K expansion
Trigger K=16 → K=32 particle expansion when 5HT consistently high
(representation at capacity limit). Requires architecture changes.

### Endorphin — Closed-loop language grounding
After Sprint 6: when particle-space distance to text goal decreases,
amplify L_clip for involved semantic categories. Requires Sprint 7.

### Vasopressin — Spatial memory consolidation
Amplify GPS-grounded components during high-displacement steps.
Correlates with cortisol and NE for a richer spatial memory system.

---

## Simple empirical tests (can run now)

### Test 1: Oxytocin signal validation (30 minutes)
```python
# Run on existing checkpoint — no training needed
import torch, glob, h5py, numpy as np, io
from PIL import Image
from train_mvtec import StudentEncoder
from train_cwm_DEPRECATED import CortexWorldModel

enc = StudentEncoder()
# ... load checkpoints ...

# Collect particle means from SEEN files (first 100)
# vs UNSEEN files (files 10000+)
# Compute oxytocin for each
# Expect: seen_oxytocin > unseen_oxytocin (t-test)
```

### Test 2: DA-loss correlation (already observable in logs)
From existing Tab 2 logs: DA=0.002 appeared at steps where loss
was measurably higher (0.573, 0.578, 0.598 vs baseline 0.562).
Compute Pearson r(DA, loss) across all logged steps.
**Expected: r > 0.3 (DA tracks loss spikes)**

### Test 3: ACh-router correlation
From existing Sprint 3 logs: ACh ranges 0.441–0.452.
Cross-reference with expert routing distribution.
**Expected: lower ACh → higher router entropy (softer routing)**

### Test 4: Cortisol simulation
Take Tab 2 loss curve. Apply cortisol formula retroactively.
Plot cortisol vs loss trajectory.
**Expected: cortisol predicts loss plateaus 10-20 steps ahead**

---

## Implementation priority

| Signal | Effort | Expected improvement | Implement when |
|--------|--------|---------------------|---------------|
| DA replay buffer | 2 days | +0.02–0.05 AUROC | After Sprint 6 |
| Oxytocin | 1 day | Faster domain adaptation | After Tab 1 |
| ACh router temp | 4 hours | Better expert specialisation | Sprint 3 ep30 |
| Cortisol | 3 days | Distribution shift detection | Sprint 7 |
| Histamine | 2 days | Better temporal context | Sprint 7 |
| BDNF weight decay | 1 week | Cleaner expert pruning | Post-paper |
| Growth hormone | 3 weeks | Dynamic capacity scaling | Post-paper |
| Endorphin | 2 weeks | Closed-loop language | Sprint 7 |

---

## Key empirical observation motivating replay buffer

**Tab 2 final epoch DA=0.002 clusters:**
- Steps 1,082,500–1,084,000: sustained DA=0.002 (4 consecutive readings)
  → Epoch 28 saved at loss=0.5674 immediately after
- Steps 1,102,500, 1,105,500: two DA=0.002 spikes in epoch 29
- Steps 1,110,500–1,111,500: three consecutive DA=0.002 readings

After 30 epochs and 1.1M training steps, DA=0.002 clusters in the
final epoch confirm the system never fully saturated. A replay buffer
would have amplified these high-DA batches, potentially reducing the
number of epochs needed to reach the same loss level by 20-30%.


---

## Empirical Test Results (2026-04-04)

### Test 1: Oxytocin — MARGINAL
- Seen mean: 0.9323, Unseen mean: 0.9233
- p=0.278, Cohen's d=0.173
- Effect in correct direction but underpowered
- Root cause: 20 warm-up files insufficient; unit-norm particles
  produce tight geometric clustering across all RECON frames
- **Action:** Retest with n_seen=200, n_unseen=200

### Test 2: DA-Loss Correlation — FAIL (wrong formulation)
- Pearson r=-0.205, p=0.337
- High DA (0.002) associated with slightly LOWER loss: 0.5683 vs 0.5701
- Root cause: DA tracks RELATIVE surprise (loss vs running mean),
  not absolute loss. In a converged system, DA fires on batches
  slightly above recent average regardless of absolute magnitude.
- **Action:** Reformulate as r(DA, loss - running_mean_loss).
  Replay buffer should use: priority = max(0, loss - ema_loss)
  rather than DA value directly.

### Test 3: ACh Router Temperature — INCONCLUSIVE
- ACh proxy (visual norm) was exactly 0.500 for all frames
- Root cause: StudentEncoder has perfect unit normalisation (1.000±0.000)
  so norm-based proxy is constant. Zero ACh variation → zero effect.
- **Action:** Retest using real ACh from NeuromodulatorState.update()
  on live training batches.

### Test 4: Cortisol Signal — STRONG PASS
- r=0.768 at lag=1, p<0.0001
- Cortisol correctly predicted the Sprint 3 distribution shift
  (epoch 20 loss rise from 0.5674→0.5696)
- Signal remains predictive through lag=5 (r=0.429, p=0.032)
- **Action:** Implement in neuromodulator. Wire to L_gps (NE) and
  L_skill (eCB) amplification during high-cortisol periods.

### Revised implementation plan
1. **Cortisol** — implement immediately, empirically validated (r=0.768)
2. **DA replay** — reformulate as relative-error priority, retest
3. **Oxytocin** — retest with 200 warm-up files
4. **ACh router** — retest with real ACh from live neuromodulator

### Cortisol implementation (ready to add to neuromodulator.py)
```python
class CortisolSignal:
    def __init__(self, window=100, baseline=None, decay=0.99):
        self.loss_ema = None
        self.decay = decay
        self.baseline = baseline

    def update(self, loss: float) -> float:
        if self.loss_ema is None:
            self.loss_ema = loss
        self.loss_ema = self.decay * self.loss_ema + (1-self.decay) * loss
        ref = self.baseline if self.baseline else self.loss_ema
        cortisol = max(0.0, (loss - ref) / (ref + 1e-8))
        return min(1.0, cortisol * 10)

    def modulate(self, signals: dict, cortisol: float) -> dict:
        # High cortisol → re-anchor GPS, suppress habits
        signals['ne']  = min(1.0, signals['ne']  + 0.3 * cortisol)
        signals['ecb'] = max(0.0, signals['ecb'] - 0.3 * cortisol)
        return signals
```

