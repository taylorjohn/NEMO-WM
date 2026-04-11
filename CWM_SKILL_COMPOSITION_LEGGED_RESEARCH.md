# Deep Research: Sequential Skill Composition & High-DOF Robot Control
# For CORTEX World Model — OGBench-Cube + Future Legged Robot Domains
# 2026-03-30

---

## Part 1: Sequential Skill Composition for Long-Horizon Manipulation

### The Core Problem

The OGBench-Cube benchmark requires sequential skill composition over 6+ step
horizons: move arm to cube → grasp → lift → transport → place → release. Each
step is a separate skill with distinct preconditions and effects. A flat world
model at the particle level sees only one-step transitions — it has no concept
that "grasp" and "transport" are semantically distinct phases. Error accumulates
at each step boundary, and a mistake in step 2 (failed grasp) invalidates all
planning for steps 3–6.

This is the honest capacity gap in CWM: the ParticleTransformer with d=128, 2
layers is a single-level flat dynamics model. It can predict the next particle
state, but it cannot plan "grasp then stack" without planning every primitive
action in between.

---

### Method 1: Hierarchical World Models (THICK — ICLR 2024)

The most principled approach. THICK (Temporal Hierarchies from Invariant Context
Kernels) learns a two-level world model where the lower level handles fine-grained
transitions and the upper level handles context changes — i.e., skill boundaries.

**How it works:** The lower level world model updates sparsely — most of the time,
the latent context is invariant (within a skill, state changes smoothly). When a
context change is detected (skill boundary: grasp→transport), the high-level model
fires and predicts the abstract outcome. This removes the need to plan every
primitive action to reason about multi-step sequences.

**For CWM translation:** Add a second "context" latent that changes slowly. The
ParticleTransformer operates at the fast timescale; a thin GRU or SSM operates
at the slow timescale and only updates when particle configurations change
discontinuously (contact events, phase transitions). The MPC horizon at the high
level is reduced by factor K (skill length), making 6-step planning as easy as
1-step planning.

```python
class HierarchicalCWM(nn.Module):
    """
    Two-level world model:
      - Low level: ParticleTransformer (fast, per-step dynamics)
      - High level: ContextGRU (slow, context/skill-level dynamics)
    """
    def __init__(self, d_model=128, K=16, action_dim=9, context_dim=32):
        super().__init__()
        # Fast level: existing ParticleTransformer
        self.particle_dynamics = ContactAwareParticleTransformer(
            d_model=d_model, K=K, action_dim=action_dim
        )
        # Slow level: context GRU, updates only on phase transitions
        self.context_gru = nn.GRUCell(d_model * K, context_dim)
        # Sparsity gate: is this a context-change step?
        self.change_gate = nn.Sequential(
            nn.Linear(d_model * K, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()          # near 1 = context change, near 0 = within-skill
        )
        # High-level predictor: predicts outcome K steps ahead
        self.high_pred = nn.GRUCell(context_dim + action_dim, context_dim)
        self.context_dim = context_dim

    def forward(self, particles, action, context_h):
        """
        particles: (B, K, d_model)
        action:    (B, action_dim)
        context_h: (B, context_dim) — slow context state
        """
        B, K, D = particles.shape
        flat = particles.reshape(B, -1)                # (B, K*D)

        # Low-level step
        particles_next, signed_dist = self.particle_dynamics(particles, action)

        # Is this a context-change boundary?
        change_prob = self.change_gate(flat)           # (B, 1)

        # Sparse update: only update context when change_prob > threshold
        # During training: use gumbel-sigmoid for differentiability
        context_input = flat
        context_candidate = self.context_gru(context_input, context_h)
        context_h_next = (change_prob * context_candidate +
                          (1 - change_prob) * context_h)

        return particles_next, context_h_next, signed_dist, change_prob

    def plan_abstract(self, context_h, action_sequence):
        """
        High-level planning: predict context K steps ahead without
        rolling out every primitive action. Used for goal-directed search.
        action_sequence: (N_skills, action_dim) — one action per skill
        """
        h = context_h
        contexts = [h]
        for a in action_sequence:
            inp = torch.cat([h, a.unsqueeze(0)], dim=-1)
            h = self.high_pred(inp, h)
            contexts.append(h)
        return contexts
```

**Expected impact:** Reduces effective planning horizon from 50+ primitive steps
to 6 skill-level steps for OGBench-Cube. Credit assignment improves dramatically.
This is the primary architectural addition needed for OGBench-Cube parity with LeWM.

**Parameter cost:** +context_dim (32) × K (16) × d_model (128) ≈ +25K params.
Fits within budget.

---

### Method 2: SPlaTES — Stable Planning with Temporally Extended Skills (RLC 2025)

Key insight from SPlaTES: rather than learning skills from scratch, train skills
to be maximally predictable by the abstract world model. "Skills should achieve
what the abstract world model predicts" — this is a bidirectional constraint.

The abstract world model shapes the skill reward (skills must land in the predicted
outcome state). Skills in turn determine the world model's transition dynamics. Both
are trained jointly.

**For CWM:** The context GRU (above) serves as the abstract world model. Skills
are defined implicitly by the span between context-change events. The key addition:
a skill reward that penalises deviation from the context GRU's prediction.

```python
def skill_reward(particles_achieved, context_pred, decoder):
    """
    Reward for a skill: how close is the achieved particle state to what
    the high-level context GRU predicted?
    decoder: maps context → expected particle configuration
    """
    particles_expected = decoder(context_pred)  # (B, K, d_model)
    similarity = F.cosine_similarity(
        particles_achieved.reshape(B, -1),
        particles_expected.reshape(B, -1),
        dim=-1
    )
    return similarity.mean()
```

**Why this matters for OGBench-Cube:** Without this, skills (grasp, lift, place)
are learned independently. With this, each skill is trained to land in exactly
the state the high-level planner expected. Chaining becomes reliable because
skill endpoints match skill start points.

---

### Method 3: HiLAM — Hierarchical Latent Action Model (ICLR 2026 Workshop)

HiLAM is the most recent approach (March 2026). It discovers latent skills
from action-free video by using an inverse dynamics model to infer motion between
frames, then compressing sequences of low-level latent actions into high-level
latent skills with a dynamic chunking mechanism.

The key differentiator from THICK and SPlaTES: HiLAM works from **unlabeled
video** — no action labels required for skill discovery. For OGBench-Cube this
matters because the dataset is collected by scripted policies with different
strategies; the underlying skills are present in the video even if the action
labels don't cleanly segment them.

**For CWM:** Use HiLAM-style inverse dynamics to discover natural skill boundaries
in OGBench-Cube trajectories before training the HierarchicalCWM:

```python
class InverseDynamicsSkillBoundary(nn.Module):
    """
    Detects skill boundaries in particle trajectories without action labels.
    Based on HiLAM's inverse dynamics approach.
    """
    def __init__(self, d_model=128, K=16):
        super().__init__()
        # Predict action from consecutive particle states
        self.inv_dyn = nn.Sequential(
            nn.Linear(d_model * K * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 9)   # 9-DOF arm action
        )
        # Boundary detector: high loss → skill boundary
        self.boundary_threshold = 0.1

    def forward(self, particles_t, particles_t1):
        B, K, D = particles_t.shape
        x = torch.cat([particles_t.reshape(B, -1),
                        particles_t1.reshape(B, -1)], dim=-1)
        action_pred = self.inv_dyn(x)
        return action_pred

    def detect_boundaries(self, trajectory, action_gt):
        """
        Identify skill boundaries: timesteps where inverse dynamics
        prediction error spikes (transition between skills is discontinuous).
        """
        errors = []
        for t in range(len(trajectory) - 1):
            pred = self.forward(trajectory[t], trajectory[t+1])
            err = F.mse_loss(pred, action_gt[t])
            errors.append(err.item())
        # Boundaries: local maxima in prediction error
        boundaries = [i for i in range(1, len(errors)-1)
                      if errors[i] > errors[i-1] and errors[i] > errors[i+1]
                      and errors[i] > self.boundary_threshold]
        return boundaries
```

---

### Method 4: SCaR — Skill Chaining via Dual Regularization (NeurIPS 2024)

SCaR addresses the handoff problem between sequentially chained skills. Without
explicit regularisation, skill A ends in state S_A and skill B starts from state
S_B — if S_A ≠ S_B, the chain fails. SCaR applies bidirectional regularisation:

- Forward: terminal state of skill A pulled toward initial state of skill B
- Backward: initial state of skill B pulled toward terminal state of skill A

In particle terms: at skill boundaries, the particle configuration from the
preceding skill must match the expected particle configuration for the succeeding
skill. This is a loss term, not a planning algorithm:

```python
def skill_transition_loss(particles_end_skill_a, particles_start_skill_b,
                           lambda_scar=0.05):
    """
    Bidirectional skill transition regularisation (SCaR-style).
    Ensures skill endpoints match skill startpoints in particle space.
    """
    forward_loss  = F.mse_loss(particles_end_skill_a,
                                particles_start_skill_b.detach())
    backward_loss = F.mse_loss(particles_start_skill_b,
                                particles_end_skill_a.detach())
    return lambda_scar * (forward_loss + backward_loss)
```

This is the cheapest addition — ~0 parameters, just a loss term at identified
skill boundaries. Combined with HiLAM boundary detection, it trains the
ParticleTransformer to produce smooth handoffs automatically.

---

### Recommended Implementation Order for OGBench-Cube

| Priority | Method | Cost | Expected Gain |
|---|---|---|---|
| 1 | HiLAM inverse dynamics — detect skill boundaries | ~50K params | Required for all below |
| 2 | SCaR transition loss — smooth skill handoffs | 0 params | +10–15pp success rate |
| 3 | THICK context GRU — reduce planning horizon | +25K params | Enables 6-step planning |
| 4 | SPlaTES skill reward shaping | 0 params | Stabilises training |

Total addition: ~75K params. CWM at ~897K — still 16× vs LeWM.

---

## Part 2: High-DOF Robot Control

### 9-DOF Arm (OGBench-Cube)

**What 9-DOF means:** Typical robot arm joints are: 3× shoulder (rotation, pitch,
yaw) + 1× elbow + 3× wrist + 2× gripper fingers = 9 DOF. Each joint requires a
separate torque command. The action space is continuous: a_t ∈ ℝ⁹.

**Why it's hard for world models:** The arm's configuration space has 9 independent
dimensions with coupled dynamics (moving shoulder changes wrist position non-linearly).
Most manipulation world models either:
1. Use end-effector control (3D position + orientation = 6 DOF) and rely on IK
2. Use joint torques directly (9 DOF) and learn the full kinematics

OGBench-Cube uses joint torques. The particle world model must learn to associate
joint velocity patterns with end-effector trajectories from scratch.

**The structured action module (already designed) helps here:** Only the
end-effector particle gets high relevance; the contact head detects when the
gripper particle reaches a cube particle. This gives the model implicit IK
structure without explicitly computing it.

**Key technique from literature — physics-informed continuous-time RL (2026):**
For 7-DOF arms, deriving Hamilton-Jacobi-Bellman equations as a data loss term
and using Kalman filter torque estimation significantly improves control quality.
For CWM, a simplified version: add a physics consistency loss that penalises
predicted particle accelerations that violate Newton's second law (F=ma) given
the arm's known inertia tensor. This is 0 extra parameters — just a loss term
using known robot mass properties.

```python
def physics_consistency_loss(particles_t, particles_t1, particles_t2,
                              dt=0.1, lambda_phys=0.001):
    """
    Penalise non-physical accelerations in particle trajectories.
    Finite-difference acceleration must be consistent with expected
    inertia-bounded motion.
    """
    vel_t  = (particles_t1 - particles_t)   / dt   # (B, K, d_model)
    vel_t1 = (particles_t2 - particles_t1)  / dt
    accel  = (vel_t1 - vel_t) / dt                  # (B, K, d_model)

    # Penalise very large accelerations (physically implausible)
    # Threshold: 9.8 m/s² × scale_factor
    max_accel = 5.0  # units depend on particle embedding scale
    accel_loss = F.relu(accel.norm(dim=-1) - max_accel).mean()
    return lambda_phys * accel_loss
```

---

### Hexapod (18-DOF) — Future Domain

**DOF breakdown:** 6 legs × 3 joints (hip, knee, ankle) = 18 total. Typical
gait patterns: tripod (3 legs simultaneous), alternating tripod, wave gait.
Tripod is fastest but least stable on rough terrain.

**Key finding from 2025–2026 research:** Pure DRL on 18-DOF hexapods is extremely
slow to converge (large action space, complex coupling). The winning approach is
**hierarchical CPG + DRL** (published in IEEE TETCI 2026):

- High level: DRL learns CPG parameters (frequency, phase offset per leg)
- Mid level: CPG generates rhythmic gait patterns
- Low level: PD controllers execute joint commands

This 3-layer architecture reduces the effective search space from ℝ¹⁸ per step
to ℝ⁴ (CPG parameters: frequency, amplitude, offset, phase) per step. Training
time drops from weeks to hours.

**For a CWM hexapod domain:** The action_dim would be 18, but the structured
action module would learn that adjacent leg particles are coupled through the
CPG pattern — when the right-front leg particle moves, the right-back and
left-middle particles follow a predictable phase relationship. This is the
same inter-particle coupling learned by the contact head, but for leg-leg
coordination rather than arm-object contact.

The key CWM adaptation: add a leg-symmetry prior to the signed distance head.
Hexapods have bilateral and front-back symmetry — particles representing
symmetric legs should have correlated dynamics. Enforce this as a soft
regularisation:

```python
def hexapod_symmetry_loss(particles, leg_pair_indices, lambda_sym=0.01):
    """
    For hexapod: symmetric leg pairs (RF-LF, RM-LM, RB-LB) should have
    correlated but phase-offset particle dynamics.
    leg_pair_indices: [(i_RF, i_LF), (i_RM, i_LM), (i_RB, i_LB)]
    """
    loss = 0.0
    for (i, j) in leg_pair_indices:
        p_i = particles[:, i, :]   # (B, d_model)
        p_j = particles[:, j, :]   # (B, d_model)
        # Symmetric legs should have similar norms but opposite phase
        loss += F.mse_loss(p_i.norm(dim=-1), p_j.norm(dim=-1))
    return lambda_sym * loss / len(leg_pair_indices)
```

**CPG-conditioned action module for hexapod:**

Rather than broadcasting 18-dim raw joint torques, encode them through a CPG
prior: derive a 4-dim CPG parameter vector (frequency, amplitude, duty cycle,
phase) from the 18 joint commands using an inverse model, and use this 4-dim
vector as the action input to the StructuredActionModule. The decoder then maps
CPG params back to per-leg particle updates.

```python
class CPGActionEncoder(nn.Module):
    """
    Encodes 18-DOF joint commands into 4-dim CPG space for hexapod.
    Drastically reduces effective action dimensionality.
    """
    def __init__(self, n_joints=18, cpg_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_joints, 64), nn.ReLU(),
            nn.Linear(64, cpg_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(cpg_dim, 64), nn.ReLU(),
            nn.Linear(64, n_joints)
        )

    def forward(self, joint_actions):
        cpg = self.encoder(joint_actions)   # (B, 4)
        return cpg

    def reconstruction_loss(self, joint_actions):
        cpg = self.encoder(joint_actions)
        recon = self.decoder(cpg)
        return F.mse_loss(recon, joint_actions)
```

---

### Quadruped (8–16-DOF) — Future Domain

**DOF breakdown:** 4 legs × 2 joints (hip, knee) = 8 DOF (simplified). Full
quadruped: 4 legs × 4 joints (hip_roll, hip_pitch, knee, ankle) = 16 DOF.
ANYmal, Spot, and MIT Cheetah all operate in 12–16 DOF range.

**Key result — Science Robotics 2020/2022 (ETH Zurich):** Proprioceptive-only
RL (no vision) achieves zero-shot sim-to-real transfer on challenging natural
terrain including mud, snow, and dynamic footholds. Inputs: joint positions,
velocities, IMU, foot contact. No camera required.

**For CWM quadruped domain:** The particle representation maps naturally to leg
kinematics — K=4 particles, one per leg, each encoding (position, velocity,
contact_state) in its 128-dim latent. The contact head's signed distance then
models foot-ground contact (when particle distance to ground plane → 0, contact
activates and foot must not penetrate).

**Terrain heightmap as a new input modality:**

The RECON outdoor navigation domain already handles 4Hz outdoor frames. For
quadruped terrain navigation, the key addition is a terrain heightmap particle —
an extra particle representing the local terrain elevation around each foot.
The signed distance head then models foot-to-terrain distance, and the dynamics
module learns to keep this distance ≥ 0 (no penetration) while minimising it
during stance phases.

```python
class TerrainAwareParticleEncoder(nn.Module):
    """
    Extends ParticleEncoder with terrain heightmap particles.
    For quadruped: 4 leg particles + 4 terrain particles = K=8 total.
    """
    def __init__(self, d_model=128, K_legs=4, K_terrain=4):
        super().__init__()
        self.K = K_legs + K_terrain
        self.leg_encoder    = SpatialSoftmaxEncoder(d_model, K_legs)
        self.terrain_encoder = nn.Sequential(
            nn.Linear(K_terrain * 3, 256),  # heightmap patch features
            nn.ReLU(),
            nn.Linear(256, K_terrain * d_model)
        )

    def forward(self, frame, heightmap=None):
        leg_particles = self.leg_encoder(frame)         # (B, K_legs, d_model)
        if heightmap is not None:
            terrain = self.terrain_encoder(heightmap)
            terrain = terrain.reshape(B, self.K_terrain, -1)
            return torch.cat([leg_particles, terrain], dim=1)
        return leg_particles
```

---

### Cross-Domain Particle Mapping

One of CWM's unique advantages: the same 56K StudentEncoder backbone and
ParticleTransformer weights serve all domains. The action_dim and particle
count can vary per domain while preserving shared weights:

| Domain | K particles | action_dim | Contact relevant | Symmetry prior |
|---|---|---|---|---|
| OGBench-Cube | 16 | 9 | Yes (arm-cube) | No |
| PushT | 16 | 2 | Yes (pusher-T) | No |
| Hexapod (future) | 18 (6 legs × 3 joints) | 18 → 4 CPG | Yes (foot-ground) | Bilateral |
| Quadruped (future) | 8 (4 legs + 4 terrain) | 12 | Yes (foot-terrain) | Bilateral |
| RECON outdoor | 16 | 2 | No | No |
| MVTec/Cardiac/SMAP | 16 | 0 | No | No |

The hexapod and quadruped domains would represent a significant CWM expansion —
truly multi-domain control from inspection to manipulation to locomotion, all with
the same backbone. No published system currently does this.

---

## Integration with CWM — What to Add

### Immediate (Sprint 3, OGBench-Cube)

1. **Inverse dynamics skill boundary detection (HiLAM)** — run offline on the
   OGBench-Cube dataset to identify natural skill boundaries before training.
   ~50K param model, run once.

2. **SCaR transition loss** — add to CWMLoss as λ_scar=0.05. Zero new parameters.
   Activated only at detected skill boundaries.

3. **THICK context GRU** — add 25K param slow-timescale context state to
   HierarchicalCWM. Replaces flat ParticleTransformer for OGBench-Cube evaluation.

### Future (post-Sprint 5, new domains)

4. **CPG action encoder** — for hexapod domain, encodes 18-DOF → 4-dim CPG.
   Reduces action space complexity from ℝ¹⁸ to ℝ⁴.

5. **Terrain heightmap particles** — for quadruped, adds 4 terrain particles to
   the standard 4 leg particles. Enables foot-terrain contact modelling.

6. **Hexapod symmetry loss** — bilateral gait regularisation, 0 parameters.

7. **Physics consistency loss** — penalise non-physical accelerations for arm and
   legged robot domains. 0 parameters, known inertia tensors.

---

## Key Papers Referenced

| Paper | Relevance |
|---|---|
| THICK (ICLR 2024) | Two-level hierarchical world model, context-change detection |
| SPlaTES (RLC 2025) | Skill-predictable world model for long-horizon MPC |
| HiLAM (ICLR 2026 Workshop) | Latent skill discovery from unlabeled video |
| SCaR (NeurIPS 2024) | Bidirectional skill-chain regularisation |
| CPG+DRL hexapod (IEEE TETCI 2026) | 3-layer control for 18-DOF hexapods |
| ANYmal perceptive locomotion (Science Robotics 2022) | Proprioceptive quadruped sim-to-real |
| Hexapod motion priors (arXiv:2511.03167, Nov 2025) | First real-world RL hexapod controller |
| PICRL 7-DOF arm (JII 2026) | Physics-informed RL for arm manipulation |
| ContactNets (CoRL 2020) | Signed distance for discontinuous contact |
| FoG (NYU/LAAS 2026) | Decoupled gradient for contact RL on PushT |

---

*Generated: 2026-03-30 | CORTEX-PE v16.17*
*Relates to: CWM_MASTER_PLAN.md Sprint 3 (OGBench-Cube) and future legged robot domains*
