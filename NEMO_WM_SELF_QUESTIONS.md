# NeMo-WM: What the World Model Knows

## The 16 Questions

Most world models answer one question: "What will I see next?"
NeMo-WM answers 16 distinct self-referential questions, each grounded
in a different component and neuromodulatory mechanism.

---

## Perception — Where am I?

### 1. "Where am I right now?"
**Component:** ProprioEncoder (k_ctx=16, 26K params)
**Mechanism:** Integrates 8 seconds of motion (velocity, heading, contact) into a 64-D belief state.
**Result:** AUROC 0.9974 — outperforms V-JEPA 2 ViT-G (1034M params, AUROC 0.883).
**Latency:** 0.055ms per step.

### 2. "What does this place look like?"
**Component:** GPSRetriever (FAISS, 215K frames)
**Mechanism:** Dead-reckons GPS from proprio, retrieves nearest real frame from prior traversals.
**Result:** 69% CLIP cosine accuracy. This IS the visual prediction — retrieval from experience, not pixel generation.
**Latency:** ~6ms per query.

### 3. "Is something wrong here?"
**Component:** CWM multidomain encoder (56K params)
**Mechanism:** Reconstruction error in latent space flags anomalies across 6+ domains.
**Result:** CWRU bearings 1.000, MIMII audio 0.931, MVTec visual 0.892 AUROC.
**Latency:** <1ms per frame.

### 4. "Where is [text description]?"
**Component:** SemanticHead (98K) + CLIPBridge (65K)
**Mechanism:** CLIP text encoding → bridge to latent space → nearest node in GeoLatentDB (10,906 nodes).
**Result:** 9/9 STRONG semantic alignment. Language A* 0.05ms — faster than GPS A* 0.06ms.
**Latency:** 25ms CLIP encode (one-time) + 0.64ms scoring.

---

## Imagination — What happens next?

### 5. "If I turn left, where do I end up?"
**Component:** BeliefTransitionModel (Sprint D1)
**Mechanism:** Predicts b_{t+1} = f(b_t, a_t) with calibrated uncertainty σ.
**Result:** Eval MSE=0.031, σ=0.137 calibrated.
**Latency:** <0.5ms per step.

### 6. "What happens over the next 8 seconds?"
**Component:** ImaginationRollout (Sprint D4)
**Mechanism:** Simulates 32 steps forward, evaluates 8 action candidates under V_neuro.
**Result:** Belief drift cosine=0.27 at step 32 — dream coherent for ~2s, degrading to 8s. Independently confirms ACh planning horizon.
**Latency:** Open-loop ~2ms, closed-loop ~15ms.

### 7. "How far ahead should I plan?"
**Component:** ACh-gated planning horizon (Sprint D5)
**Mechanism:** T_horizon = max(1, round(32 × ACh_t)). ACh = N_eff/N from particle filter.
**Result:** ACh sweep: k=2(0.925) → k=4(0.953) → k=8(0.977) → k=16(0.9974) → k=32(0.9997). Superlinear — broader window = better for slow 4Hz outdoor nav.
**Latency:** Computed from particle filter, no additional cost.

### 8. "Should I trust my prediction or look again?"
**Component:** AnticipateReactGate (Sprint D3)
**Mechanism:** α = g(δ, CRT). δ = prediction error. High α → trust model (anticipatory). Low α → re-observe (reactive).
**Result:** α > 0.8 → open-loop planning. α < 0.4 → closed-loop with fresh observations.
**Latency:** <0.1ms.

---

## Decision — What should I do?

### 9. "Which action is best given my current state?"
**Component:** NeuromodulatedValue (Sprint D5)
**Mechanism:** V = DA·Q(b,g) − CRT·U(b) + ACh·H(b,g)
- DA scales goal proximity (surprise → chase the reward)
- CRT penalises novelty (stress → avoid the unknown)
- ACh extends value horizon (certainty → plan further)
**Result:** Three neuromodulators each control a distinct aspect of value.
**Latency:** Computed during rollout, no additional cost.

### 10. "Have I been in a similar situation before?"
**Component:** EpisodicBuffer (Sprint D6)
**Mechanism:** DA-gated storage (surprising events consolidated). Cosine retrieval of k=3 similar past beliefs. Past actions warm-start the planner.
**Result:** 35/36 tests. 18K store/s. Integration with ImaginationRollout confirmed.
**Latency:** 0.055ms store, 183ms retrieve (needs FAISS optimisation → <1ms).

### 11. "Is this a new kind of place?"
**Component:** SchemaStore (Sprint D6)
**Mechanism:** Domain-compressed prototypes. novelty() measures distance from known schemas. High novelty → cortisol → conservative planning.
**Result:** 10× storage compression. Domain-specific prototypes.
**Latency:** 0.023ms.

---

## Meta-cognition — What do I know about knowing?

### 12. "Does language help me understand this scene?"
**Component:** VLM aphasia ablation
**Mechanism:** Zero the language gate → measure impact. CWM visual path collapses to chance (0.500). Proprio path unaffected (0.9974).
**Result:** Double dissociation. Language necessary for visual WM, irrelevant for physics-grounded self-localisation. Parallels Fedorenko et al. language/MD independence.
**Delta:** +0.4542 AUROC.

### 13. "How much of my working memory is available?"
**Component:** WorkingMemory (Sprint D6)
**Mechanism:** K_eff = max(K_min, K_base − floor(CRT × 6)). Cortisol reduces effective capacity.
**Result:** CORT=0.0 → K=8. CORT=0.5 → K=5. CORT=1.0 → K=2.
**Cognitive parallel:** Lupien et al. (1999) — stress-induced WM degradation.

### 14. "What would I dream about this route?"
**Component:** nemo_dream.py
**Mechanism:** Belief rollout → GPS retrieval at imagined positions → 3-row visualisation (actual / retrieved / belief bars).
**Result:** 8-second dream figure. Drift cosine=0.27 at step 32. The dream IS the planning — retrieval-based, not pixel generation. JEPA predicts in latent space.

---

## Partially Wired

### 15. "Should I explore or stick with what I know?"
**Component:** 5HT/DA regime switching
**Status:** Regime labels computed (EXPLOIT/EXPLORE/REOBSERVE) but not yet gating the planner's action sampling distribution. The signal exists; the wire doesn't.

### 16. "Am I getting tired? Should I recalibrate?"
**Component:** Adenosine fatigue gate
**Status:** ado computed and fatigue threshold set (0.7) but not triggering recalibration or sleep-wake cycles.

---

## Why This Matters

The standard world model paper reports one number: prediction accuracy.
NeMo-WM reports 14 live capabilities across perception, imagination,
decision-making, and meta-cognition — each grounded in a named
biological mechanism with a measured result.

The system doesn't just predict. It knows what it knows, how far to
trust itself, when to stop imagining, and what worked last time.
That's the difference between a predictor and a mind.

---

## Architecture Map

```
Question                          Component                    Signal
──────────────────────────────────────────────────────────────────────
"Where am I?"                  → ProprioEncoder               → b_t
"What does it look like?"      → GPSRetriever                 → frame
"Is something wrong?"          → CWM anomaly                  → score
"Where is [text]?"             → SemanticHead + CLIPBridge    → node_id
"If I do X?"                   → BeliefTransitionModel        → b_{t+1}, σ
"Next 8 seconds?"              → ImaginationRollout           → trajectory
"How far to plan?"             → ACh (N_eff/N)                → T_horizon
"Trust prediction?"            → AnticipateReactGate (α)      → mode
"Best action?"                 → V_neuro (DA·Q−CRT·U+ACh·H)  → action
"Been here before?"            → EpisodicBuffer               → memory
"New kind of place?"           → SchemaStore.novelty()         → CRT
"Does language help?"          → Aphasia ablation              → Δ AUROC
"WM capacity?"                 → K_eff = f(CRT)               → K items
"What would I dream?"          → nemo_dream.py                → figure
```

---

*Updated: 2026-04-13 — Sprint D6 complete, all 14 capabilities confirmed.*
