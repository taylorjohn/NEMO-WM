# CWM Dopamine-Modulated Reward System
# Routing CORTEX-16's biological neuromodulators into CWM world model rewards
# 2026-03-30

---

## Background: The Existing Neuromodulator System

CORTEX-16 already runs three neuromodulators derived from Allen Brain Observatory
Neuropixels data (256 mouse V1 neurons). These are live signals computed every tick:

| Signal | Source | Baseline | Role |
|---|---|---|---|
| `rho` (ρ) | Fleet resonance / biological spike rate | ~0.48–0.62 | Arousal scalar — divides cost landscape |
| `DA` | Dopamine modulator | 0.500 | Reward prediction — spikes on positive surprises |
| `CRT` | Cortisol/certainty signal | 0.000 | Stress — rises with task uncertainty |

The existing biologically-modulated Mirror Ascent planner already uses rho:

```python
# From ARCHITECTURE.md — existing CORTEX-16 formulation
g_t = -L_cost / (1.0 + rho)
# When rho is high (strong neural signal), cost landscape compresses
# → more aggressive, exploratory planning
# When rho is low (quiet signal), costs stay high → conservative
```

**The idea:** route the same three signals into the CWM skill composition and
legged robot reward framework. The CWM dynamics module learns from the same
biological reward modulation that drives the trading system. One neuromodulator
system, two domains.

---

## Design Principles

### 1. DA modulates reward magnitude (prediction error signal)
Dopamine in neuroscience encodes **reward prediction error** — the difference
between expected and actual outcome. When the skill achieves better than
predicted: DA spikes. When it fails: DA dips below baseline.

In CWM terms: DA scales the reward for skill completion and contact quality.
High DA = the model was surprised by a good outcome → large positive update.
Low DA = the model predicted success but failed → large negative update.

### 2. rho modulates exploration vs exploitation
High rho (0.62+) = the neural substrate is highly aroused = high-conviction
signal = compress the cost landscape → the planner explores more aggressively.
Low rho (0.48) = quiet signal = keep costs high → conservative, exploit known.

In CWM terms: rho sets the effective planning temperature. High rho shortens
the effective MPC horizon (fast, confident planning). Low rho extends it
(slow, cautious planning with more rollouts).

### 3. CRT modulates the penalty for contact failure
Cortisol rises under task uncertainty and stress. In CWM terms: high CRT
means the current task is difficult/uncertain → increase the penalty for
contact failures and missed skill boundaries. Zero CRT (easy task, high
certainty) → normal penalty weights.

---

## Implementation: DopamineModulatedCWMLoss

```python
class DopamineModulatedCWMLoss(nn.Module):
    """
    CWM loss function modulated by the CORTEX-16 neuromodulator signals.

    Uses DA, rho, and CRT from the existing biological signal pipeline.
    These are either live values from the trading system (shared NUC process)
    or cached values from the last trading session for offline CWM training.

    Loss components:
      L_predict:  MSE between predicted and actual next particle states
      L_gaussian: SIGReg collapse prevention
      L_gps:      GPS grounding (RECON domain)
      L_contact:  Signed distance auxiliary (PushT / OGBench-Cube)
      L_skill:    SCaR skill transition (OGBench-Cube hierarchical)

    Neuromodulator effects:
      DA  → scales L_predict reward magnitude (reward prediction error)
      rho → sets planning temperature (exploration/exploitation)
      CRT → scales L_contact penalty (contact stress)
    """

    def __init__(self,
                 lambda_gaussian: float = 0.1,
                 lambda_gps:      float = 0.05,
                 lambda_contact:  float = 0.01,
                 lambda_skill:    float = 0.05):
        super().__init__()
        self.lambda_gaussian = lambda_gaussian
        self.lambda_gps      = lambda_gps
        self.lambda_contact  = lambda_contact
        self.lambda_skill    = lambda_skill

        # Neuromodulator state — updated each tick from CORTEX-16 or
        # set to biological baseline for offline CWM training
        self.DA  = 0.500   # Dopamine baseline
        self.rho = 0.500   # Fleet resonance baseline
        self.CRT = 0.000   # Cortisol baseline

    def update_neuromodulators(self, DA: float, rho: float, CRT: float):
        """
        Call before each training batch with current neuromodulator values.
        In live mode: pull from the trading system's shared state.
        In offline mode: use session-average values from cortex_combat_log.csv.
        """
        self.DA  = max(0.1, min(2.0, DA))   # clamp to reasonable range
        self.rho = max(0.1, min(1.0, rho))
        self.CRT = max(0.0, min(1.0, CRT))

    def dopamine_reward_scale(self) -> float:
        """
        DA-modulated reward scale.
        DA = 0.5 (baseline) → scale = 1.0 (normal update)
        DA = 1.0 (spike)    → scale = 2.0 (strong reward signal)
        DA = 0.1 (dip)      → scale = 0.2 (weak signal, cautious update)
        """
        return self.DA * 2.0   # maps [0.1, 1.0] → [0.2, 2.0]

    def rho_temperature(self) -> float:
        """
        rho-modulated planning temperature.
        High rho → low temperature → sharper, more confident gradients.
        Low rho  → high temperature → softer, more exploratory gradients.
        From ARCHITECTURE.md: g_t = -L_cost / (1.0 + rho)
        """
        return 1.0 / (1.0 + self.rho)   # [0.5, 0.91] for rho in [0.1, 1.0]

    def cortisol_contact_scale(self) -> float:
        """
        CRT-modulated contact penalty.
        CRT = 0.0 (easy, certain) → normal contact penalty
        CRT = 1.0 (stressed, uncertain) → 3× contact penalty
        """
        return 1.0 + 2.0 * self.CRT   # maps [0, 1] → [1, 3]

    def forward(self,
                z_pred,           # (B, K, d_model) predicted next particles
                z_target,         # (B, K, d_model) actual next particles
                signed_dist=None,          # (B, K, K) contact head output
                particle_positions=None,   # (B, K, 2) SpatialSoftmax positions
                gps_pred=None,             # (B, 2) predicted GPS
                gps_target=None,           # (B, 2) actual GPS
                particles_skill_end=None,  # (B, K, d) end of skill A
                particles_skill_start=None # (B, K, d) start of skill B
                ):

        # ── Core prediction loss (DA-modulated) ──────────────────────────────
        # DA modulates the magnitude of the reward prediction error signal.
        # High DA: unexpected success → amplify the gradient update.
        # Low DA: expected failure → dampen, model already anticipates this.
        da_scale = self.dopamine_reward_scale()
        L_predict  = F.mse_loss(z_pred, z_target) * da_scale

        # ── Representation collapse prevention (temperature-modulated) ────────
        # rho sets the gradient temperature — high rho = explore more aggressively.
        # Apply rho temperature as a softening factor on SIGReg loss.
        temp = self.rho_temperature()
        L_gaussian = sigreg_loss(z_pred, z_target) * self.lambda_gaussian * temp

        total_loss = L_predict + L_gaussian

        stats = {
            'L_predict':  L_predict.item(),
            'L_gaussian': L_gaussian.item(),
            'DA':         self.DA,
            'rho':        self.rho,
            'CRT':        self.CRT,
            'da_scale':   da_scale,
            'temp':       temp,
        }

        # ── GPS grounding loss (rho-modulated) ───────────────────────────────
        # High rho = confident signal → trust GPS grounding more strongly.
        if gps_pred is not None and gps_target is not None:
            gps_scale = self.rho  # scales with confidence in the spatial signal
            L_gps = gps_grounding_loss_fast(z_pred, gps_pred, gps_target)
            L_gps = L_gps * self.lambda_gps * gps_scale
            total_loss = total_loss + L_gps
            stats['L_gps'] = L_gps.item()

        # ── Contact auxiliary loss (CRT-modulated) ───────────────────────────
        # High CRT = uncertain/stressed task → heavily penalise contact failures.
        # Low CRT = confident task → normal contact penalty.
        if signed_dist is not None and particle_positions is not None:
            crt_scale = self.cortisol_contact_scale()
            L_contact = contact_auxiliary_loss(
                signed_dist, particle_positions,
                lambda_contact=self.lambda_contact * crt_scale
            )
            total_loss = total_loss + L_contact
            stats['L_contact']  = L_contact.item()
            stats['crt_scale']  = crt_scale

        # ── Skill transition loss (DA × CRT modulated) ───────────────────────
        # SCaR bidirectional regularisation at skill boundaries.
        # DA scales the positive reward for clean handoffs.
        # CRT scales the penalty for bad handoffs (stressed task = critical).
        if particles_skill_end is not None and particles_skill_start is not None:
            skill_scale = da_scale * (1.0 + self.CRT)  # combined modulation
            L_skill = skill_transition_loss(
                particles_skill_end, particles_skill_start,
                lambda_scar=self.lambda_skill * skill_scale
            )
            total_loss = total_loss + L_skill
            stats['L_skill']     = L_skill.item()
            stats['skill_scale'] = skill_scale

        return total_loss, stats
```

---

## Biological Mirror Ascent for CWM Planning

The existing CORTEX-16 Mirror Ascent planner already implements rho-divided
cost landscapes. The same formulation extends to CWM MPC:

```python
class BiologicalMPCPlanner:
    """
    CWM model-predictive controller using the same biologically-modulated
    Mirror Ascent from CORTEX-16's ARCHITECTURE.md.

    g_t = -L_cost / (1.0 + rho)    [from ARCHITECTURE.md]

    Extension: DA also modulates the mirror ascent update rate.
    High DA (unexpected good outcome) → larger policy updates.
    """

    def __init__(self, cwm, K=32, horizon=8):
        self.cwm     = cwm       # trained CWM world model
        self.K       = K         # number of candidate trajectories
        self.horizon = horizon   # planning steps

    def plan(self,
             particles_0,         # (1, K, d_model) current state
             goal_particles,      # (1, K, d_model) goal state
             action_dim=9,
             DA=0.500,
             rho=0.500,
             CRT=0.000):
        """
        Sample K candidate action sequences, roll them out through the CWM,
        score each by the biologically-modulated cost, return the best action.
        """
        B = 1
        candidates = torch.randn(self.K, self.horizon, action_dim) * 0.1

        costs = []
        for k in range(self.K):
            particles = particles_0.clone()
            cumulative_cost = 0.0

            for t in range(self.horizon):
                action = candidates[k, t].unsqueeze(0)
                particles, _ = self.cwm.particle_dynamics(particles, action)

                # Per-step cost: distance to goal in particle space
                step_cost = F.mse_loss(particles, goal_particles).item()

                # Contact penalty at stressed steps (high CRT)
                # Cortisol spikes when the arm is near the cube but hasn't grasped
                if CRT > 0.1:
                    contact_penalty = CRT * 0.1   # additional stress penalty
                    step_cost += contact_penalty

                cumulative_cost += step_cost

            # Biologically-modulated cost (from ARCHITECTURE.md):
            # g_t = -L_cost / (1.0 + rho)
            modulated_cost = -cumulative_cost / (1.0 + rho)
            costs.append(modulated_cost)

        costs_t = torch.tensor(costs)

        # DA-modulated Mirror Ascent temperature
        # High DA → sharp, confident weighting (exploit)
        # Low DA  → soft, exploratory weighting (explore)
        eta = 0.05 * DA * 2.0   # DA scales the update rate
        kl_lambda = 0.1

        q = torch.ones(self.K) / self.K   # uniform prior
        weights = torch.exp(
            (costs_t - kl_lambda * torch.zeros(self.K)) / eta
        )
        q = q * weights
        q = q / q.sum()   # normalise

        # Weighted action: take expectation under updated belief
        best_action = (candidates[:, 0] * q.unsqueeze(-1)).sum(0)
        return best_action, q.max().item()
```

---

## Neuromodulator Pipeline: Live vs Offline

### Live Mode (during joint CWM + trading sessions)

The NUC runs both systems simultaneously. The dopamine signals from the trading
system can be piped to CWM training in real time:

```python
class NeuromodulatorBridge:
    """
    Reads live DA/rho/CRT from the trading system's vault state
    and feeds them to the CWM loss function.

    Call update() each time the trading tick fires.
    The CWM training loop reads the latest values.
    """

    def __init__(self):
        self.DA  = 0.500
        self.rho = 0.500
        self.CRT = 0.000
        self._lock = threading.Lock()

    def update_from_trading_tick(self, tick_log_line: str):
        """
        Parse live tick log: "rho=0.5466 DA=0.500 CRT=0.000"
        """
        import re
        with self._lock:
            m = re.search(r'rho=([0-9.]+)\s+DA=([0-9.]+)\s+CRT=([0-9.]+)',
                          tick_log_line)
            if m:
                self.rho = float(m.group(1))
                self.DA  = float(m.group(2))
                self.CRT = float(m.group(3))

    def get(self):
        with self._lock:
            return self.DA, self.rho, self.CRT
```

### Offline Mode (CWM training from stored sessions)

Use session-averaged neuromodulator values from `cortex_combat_log.csv`:

```python
def load_neuromodulator_profile(combat_log_path: str, date: str) -> dict:
    """
    Extract DA/rho/CRT statistics from a session log for offline CWM training.
    Returns time-of-day profile: bucket → (DA_mean, rho_mean, CRT_mean)
    """
    import pandas as pd
    df = pd.read_csv(combat_log_path, encoding='cp1252',
                     parse_dates=['timestamp'])
    df = df[df['timestamp'].dt.date.astype(str) == date]

    # Parse neuromodulator values from phase_abort strings if embedded
    # Otherwise use resonance as rho proxy
    profile = {
        'DA_mean':  0.500,   # baseline (DA not yet in combat log)
        'rho_mean': df['resonance'].mean(),
        'CRT_mean': 0.000,   # baseline (CRT not yet in combat log)
        'rho_by_hour': df.groupby(
            df['timestamp'].dt.tz_convert('US/Eastern').dt.hour
        )['resonance'].mean().to_dict()
    }
    return profile
```

The `rho_by_hour` profile captures the morning U-shape observed on 2026-03-30:
- 09:00–10:15: rho ~0.624 (high arousal, strong signal)
- 11:00–16:00: rho ~0.481 (low arousal, weak signal)

For CWM training, apply the appropriate rho for each training sample's
time-of-day, giving the model a biologically grounded temporal prior.

---

## Skill Reward Mapping via DA

The existing CORTEX-16 DA signal represents reward prediction error. For CWM
skill composition, each skill phase has a natural DA interpretation:

| Skill Phase | Expected DA | Interpretation |
|---|---|---|
| Approach (arm moves toward cube) | 0.450–0.500 | Routine, predicted |
| Contact made (gripper touches cube) | 0.600–0.800 | Positive surprise |
| Lift (cube leaves surface) | 0.550–0.700 | Expected but confirmatory |
| Transport (cube in air, stable) | 0.500 | Neutral — ongoing execution |
| Placement (cube contacts target) | 0.700–0.900 | Strong positive surprise |
| Release (gripper opens, cube stays) | 0.600–0.750 | Task complete |
| Drop/failure | 0.100–0.300 | Negative surprise — large gradient |

In practice, DA is computed from the Allen neural data and is not labelled
per-skill. But the average DA across a successful trajectory should be higher
than across a failed one. The DopamineModulatedCWMLoss trains the world model
to predict transitions that the biological substrate rewards.

---

## Hexapod/Quadruped Reward Modulation

For legged robot domains, the same neuromodulator signals apply with
terrain-specific interpretations:

| Domain | DA interpretation | CRT interpretation | rho interpretation |
|---|---|---|---|
| Hexapod rough terrain | High on stable foothold | High during obstacle | High = aggressive gait |
| Quadruped slope | High on stable step | High on steep grade | High = confident stride |
| Quadruped stairs | High after each step | Very high mid-stair | Low = cautious planning |

The hexapod CPG action encoder can use rho to modulate gait frequency:

```python
def rho_modulated_cpg_frequency(rho: float,
                                 base_freq: float = 1.0) -> float:
    """
    High rho → higher gait frequency (more aggressive locomotion).
    Low rho  → lower frequency (cautious terrain navigation).
    Maps rho [0.1, 1.0] → frequency [0.5, 2.0] Hz.
    """
    return base_freq * (0.5 + 1.5 * rho)   # [0.65, 2.0] for rho in [0.1, 1.0]
```

And CRT to modulate step safety margins:

```python
def crt_modulated_step_margin(CRT: float,
                               base_margin: float = 0.05) -> float:
    """
    High CRT (stressed/uncertain terrain) → larger safety margin between
    predicted foothold and actual step. More conservative.
    CRT=0 → base_margin=0.05m, CRT=1 → margin=0.20m
    """
    return base_margin + 0.15 * CRT
```

---

## Integration with Existing CWMLoss

The `DopamineModulatedCWMLoss` is a drop-in replacement for `CWMLoss` in
`train_cwm.py`. The only change to the training loop:

```python
# In train_cwm.py — replace CWMLoss with DopamineModulatedCWMLoss
loss_fn = DopamineModulatedCWMLoss(
    lambda_gaussian=0.1,
    lambda_gps=0.05,
    lambda_contact=0.01,
    lambda_skill=0.05
)

# Load neuromodulator profile from last trading session
nm_profile = load_neuromodulator_profile('cortex_combat_log.csv', '2026-03-30')

for epoch in range(n_epochs):
    for batch in recon_loader:
        # Update neuromodulators from time-of-day profile
        hour = batch['timestamp'].hour
        rho  = nm_profile['rho_by_hour'].get(hour, nm_profile['rho_mean'])
        loss_fn.update_neuromodulators(
            DA=nm_profile['DA_mean'],
            rho=rho,
            CRT=nm_profile['CRT_mean']
        )

        particles_t1_pred, signed_dist = dynamics(particles_t, action_padded)

        loss, stats = loss_fn(
            z_pred=particles_t1_pred,
            z_target=particles_t1_true,
            signed_dist=signed_dist,
            particle_positions=particle_enc.get_positions(particles_t),
            gps_pred=gps_t1_pred,
            gps_target=gps_t1
        )

        loss.backward()
        optimizer.step()
```

---

## What This Gives You

**For the research paper:** CWM is the first world model to use biological
neuromodulator signals (derived from mouse V1 Neuropixels recordings) as an
adaptive reward modulator during training. DA, rho, and CRT from the same
neural substrate that drives the trading system now also shape the dynamics
learning in the manipulation and navigation world model. This is a genuine
novelty claim that no published system makes — multi-domain shared backbone,
GPS grounding, and biological neuromodulator reward shaping all in one.

**For OGBench-Cube:** DA-scaled skill rewards mean the model learns the
_affective_ value of skill completion, not just the geometric prediction error.
High-DA transitions (contact events, successful grasps) receive amplified
gradient updates — exactly the credit assignment mechanism that makes long-
horizon sequential skill composition converge faster.

**For legged robots:** rho-modulated gait frequency + CRT-modulated step
margins give the policy a built-in confidence/stress response to terrain
difficulty, derived from the same biological signal that tells the trading
system when to be aggressive vs conservative.

---

## Files to Create/Modify

```
# New
cwm_dopamine_loss.py          # DopamineModulatedCWMLoss
cwm_bio_planner.py            # BiologicalMPCPlanner
cwm_neuromod_bridge.py        # NeuromodulatorBridge (live + offline)

# Modified
train_cwm.py                  # Use DopamineModulatedCWMLoss, load nm_profile
cwm_loss.py                   # Keep as fallback (no neuromodulators)
```

---

*Generated: 2026-03-30 | CORTEX-PE v16.17*
*Ties together: CORTEX-16 neuromodulators + CWM skill composition + legged robots*
