# Failure modes and fixes for world model and JEPA training

Five specific failure modes in world model and JEPA-based training each have published mitigation strategies, though practical coverage varies sharply. **MoE router collapse has the richest literature** with exact loss formulations and threshold values. GRASP planner ablations exist for horizon scaling but not for latent/action dimension scaling — a significant gap. The stable-worldmodel evaluation harness introduces a 50-step budget that silently degrades performance versus original benchmarks. Cross-domain JEPA with MoE routing has no single published system but can be assembled from well-documented components (PCGrad, DSBN, ETF loss). Quasimetric AUROC on navigation data is a genuinely novel metric — no published baselines exist, requiring you to construct your own from related contrastive RL work.

---

## 1. MoE router collapse: three metrics catch it early, five losses prevent it

Router collapse — where one expert absorbs all tokens — is detectable well before it becomes catastrophic. **Three complementary metrics** provide early warning at different sensitivities.

**Normalized router entropy** is the earliest signal. Computed as H(P) = −Σ pₑ·log(pₑ) divided by log(E) where E is expert count, it ranges from 0 (total collapse) to 1 (uniform). Flag at **< 0.85**, intervene at **< 0.7**, per Brenndoerfer (2025). This metric declines before token fractions become obviously imbalanced. **Load Imbalance Factor** (LIF = max(fᵢ) / (1/N)) catches worst-case severity: flag at **> 1.5**, intervene at **> 2.0**. The **coefficient of variation** of expert importance scores (CV = std/mean of Σₓ gᵢ(x) across experts) tracks whether imbalance is concentrated or diffuse. Monitor all three per-layer every 100–500 training steps.

Two additional diagnostics provide structural insight. Track **unique experts activated** across a full validation set — this should remain at E throughout training. Compute **pairwise cosine similarity** between expert weight matrices every 1–5K steps; rising average similarity signals functional redundancy preceding collapse.

The canonical prevention loss is the **Switch Transformer auxiliary load balance loss**: L_balance = α·N·Σ fᵢ·Pᵢ, where fᵢ is the fraction of tokens dispatched to expert i and Pᵢ is the fraction of router probability allocated to expert i. The recommended coefficient is **α = 0.01**, swept from 10⁻¹ to 10⁻⁵ in the original paper. The **ST-MoE router z-loss** L_z = (1/B)·Σ [log(Σ exp(hᵢ))]² penalizes large router logits and was the only intervention that improved stability without degrading quality. Use coefficient **β = 0.001–0.01** alongside the balance loss. A combined formulation is:

> **L_total = L_task + 0.01·L_balance + 0.001·L_z**

**DeepSeek-V3's loss-free balancing** offers an alternative that avoids auxiliary gradient interference entirely. It adds per-expert bias terms to routing scores before top-K selection, then computes gate weights from the original unbiased scores. Biases update outside backpropagation: increment γ for underloaded experts, decrement for overloaded ones. This achieved both better performance and better balance than auxiliary-loss approaches up to 3B parameters.

For **small-scale pretesting**, the ST-MoE protocol is the gold standard: train a FLOP-matched proxy at ~1/10th scale for 5–10K steps across 3+ seeds. Pass criteria: normalized entropy > 0.85 at all layers, LIF < 1.5, all experts activated. Then run domain-pure batches through the trained proxy and compute a **Domain-Expert Specificity Matrix** — different domains should show distinct but overlapping expert preferences, not identical distributions and not complete segregation.

Two papers are especially relevant for multi-domain world models. **M3-JEPA** (ICML 2025) uses a Multi-Gate MoE predictor within a JEPA framework, with gating that disentangles modality-specific and shared information. **DES-MoE** (EMNLP 2025) provides a 3-phase progressive specialization schedule — warm-up (all parameters trainable), stabilization (adaptive router + domain-relevant experts only), consolidation (only domain-specific experts trainable) — reducing expert overlap between domains from 0.47 to 0.18. DreamerV3 does not use MoE but its **1% unimix** technique (mixing 99% network output with 1% uniform distribution on all categorical outputs) is directly applicable to router softmax as cheap anti-collapse insurance.

---

## 2. GRASP planner: horizon ablations exist, but dimension-scaling studies do not

The GRASP planner (Psenka et al., arXiv:2602.00475, submitted January 2026, accepted ICLR 2026 Workshop on World Models) introduces a "lifted" optimization approach that treats planning as a collocation problem. Instead of serial rollouts through a world model F_θ, GRASP introduces **virtual states** s₁...s_{T−1} as independent optimization variables alongside actions. Pairwise dynamics consistency is enforced via a penalty L_dyn = Σ ‖F_θ(s̄_t, a_t) − s_{t+1}‖² with stop-gradient on state inputs. Langevin noise injects exploration into state updates, and periodic GD sync steps (every **K_sync = 100** stochastic iterations) perform a full serial rollout for refinement.

The paper provides **horizon-scaling ablations** on Push-T across H = 10, 20, 30, 40, 50. GRASP maintains success rate as horizon increases while CEM and vanilla gradient descent degrade significantly at longer horizons. Component ablations at H = 40 (500 trials per setting) confirm all three core innovations contribute: removing GD sync degrades performance, removing Langevin noise (σ_state = 0) increases local minima trapping, and removing gradient detachment lets the optimizer exploit brittle state-input gradients. The theoretical convergence analysis shows the lifted method has O(T) per-iteration cost with well-conditioned pairwise terms, versus exponentially bad conditioning for shooting methods as T grows.

**Critical gap: no ablations exist for latent dimension or action dimension scaling.** The paper tests exclusively on Push-T (2D action space) and does not vary latent dimension, report wall-clock timing, or provide CPU benchmarks. The term "lifted iterations" in the query maps to the total optimization step count between sync steps (controlled by K_sync = 100). There are no minimum viable configuration recommendations.

For the specific target of **128-dim latent, 2-dim action, < 10ms on CPU**, a rough estimation based on method structure: each GRASP iteration requires ~H parallel forward passes of F_θ plus gradient computation (~2H effective evaluations). With a small MLP world model on 128-dim inputs, each forward pass takes ~0.01–0.1ms on CPU. At H = 15, one iteration costs ~3ms sequentially. To fit within 10ms, you'd get ~3–5 lifted iterations — likely insufficient for convergence. **Practical recommendation**: use H = 10–12 with 5–8 lifted iterations plus 1 sync step at iteration 5, or move to GPU where the parallelism advantage of GRASP over CEM is fully realized. The code repository (github.com/michael-psenka/grasp) is listed on the project page (michaelpsenka.io/grasp) but may not yet be publicly available.

---

## 3. Stable-worldmodel's 50-step budget silently breaks models tuned for infinite planning

The stable-worldmodel framework (Maes, Le Lidec et al., arXiv:2602.08968, GalilAI Group, Brown University) provides a unified evaluation harness at github.com/galilai-group/stable-worldmodel (v0.0.5 as of February 2026, 275 stars, 8 open issues). Several failure modes have been documented or can be inferred from the codebase.

**The most impactful issue is the 50-step budget.** The paper explicitly states: "Unlike the original work [DINO-WM], which had an infinite planning budget, we fixed the steps budget to 50, which corresponds to 2× the minimum number of steps required to succeed (25)." Models benchmarked under DINO-WM's original infinite-horizon protocol will show dramatically lower success rates under this constraint. This is not a bug — it's a deliberate evaluation choice — but wrapping a custom model without adjusting planning parameters will produce misleadingly poor results. With horizon = 10 and receding_horizon = 5 (the standard CEM configuration), you get ~10 replanning cycles in 50 steps. Ensure your model predicts well at this horizon scale.

The `get_cost()` interface requires models to expose a cost function over candidate action sequences. For DINO-WM-style models, this is MSE between predicted and goal latent states: C(a_{0:H−1}) = ‖ẑ_H − z_g‖². Pre-trained cost models can be loaded via `swm.policy.AutoCostModel('pusht/lewm')`. **The exact tensor shape requirements are only loosely documented through examples**, which is a primary source of wrapping difficulty. Study the built-in implementations (dinowm.py, dreamer.py, tdmpc.py in `stable_worldmodel/wm/`) for the exact interface contract.

**CEM versus gradient-based solvers** is a critical configuration choice. The DINO-WM paper found CEM outperforms gradient descent, hypothesizing this is "due to our choice to not constrain the terrain smoothness of the world model during training, potentially leading to issues with the gradient." If your world model's latent dynamics are non-smooth (common for JEPA-style models without explicit smoothness constraints), gradient-based solvers (GradientSolver, PGD, LagrangianSolver) will underperform or fail. **Start with CEMSolver(num_samples=300)** as the default.

A severe **distribution shift sensitivity** is documented: DINO-WM achieves **94.0% success** on expert demonstrations but drops to **12.0%** on random policy trajectories. Custom models will face the same sensitivity — always evaluate on both in-distribution and out-of-distribution goal states. Additional practical traps include a **path inconsistency** (README says `~/.stable-wm/`, documentation says `~/.stable_worldmodel/` — set `$STABLEWM_HOME` explicitly), incomplete base installation (use `pip install stable-worldmodel[env, train]`), and checkpoint naming confusion (use `pusht/lewm` not `pusht/lewm_object.ckpt`).

---

## 4. Cross-domain JEPA with MoE needs assembled components, not an off-the-shelf solution

**No published system directly combines JEPA + MoE + multi-domain training** across visually dissimilar domains. However, well-documented techniques from adjacent fields can be composed into a working architecture. The key challenge — preventing negative transfer between outdoor navigation RGB, satellite telemetry time-series, and industrial inspection images through a shared frozen backbone — requires intervention at three levels: gradient flow, normalization, and routing.

**Gradient surgery** prevents conflicting domain gradients from degrading shared parameters. **PCGrad** (Yu et al., NeurIPS 2020) projects away interfering gradient components: for each domain pair, if gradients conflict (negative cosine similarity), replace gᵢ with gᵢ − (gᵢ·gⱼ/‖gⱼ‖²)·gⱼ. This yields >30% improvement in multi-task RL. **CAGrad** (Liu et al., NeurIPS 2021) is more principled — it finds the update maximizing worst-case local improvement within a ball around the average gradient, parameterized by c (c = 0 gives standard GD, c → ∞ gives MGDA). Apply either method to shared parameters (router, predictor) while allowing domain-specific expert parameters to update freely with their own domain loss.

**Domain-Specific Batch Normalization** (DSBN, Chang et al., CVPR 2019) maintains separate BN statistics (μ, σ², γ, β) per domain while sharing all other parameters. Place DSBN after each MoE expert layer to ensure RGB outdoor features, satellite telemetry features, and inspection image features maintain appropriately normalized distributions. This is documented as "the single most impactful component" for multi-source domain adaptation.

For **MoE routing with a frozen DINOv2 encoder**, the MoECLIP pattern (arXiv:2603.03101) is directly applicable. It inserts MoE-LoRA adapters at selected layers of a frozen vision encoder with two enforcing mechanisms:

- **Frozen Orthogonal Feature Separation (FOFS)**: orthogonally separates feature space at LoRA input, forcing experts to attend to different subspaces
- **Simplex ETF Loss**: regulates expert outputs to form an Equiangular Tight Frame (maximally separated representations), with L_etf = (1/LK²)·Σ (Gᵢ − G_ideal)² where G_ideal has 1 on diagonal and −1/(K−1) off-diagonal

For multi-modal routing, **FuseMoE** (NeurIPS 2024) provides per-modality routers with Laplace gating (L2-distance-based rather than dot-product, avoiding magnitude bias) and entropy regularization. This handles the modality mismatch between RGB images and time-series inputs.

A practical training schedule should use **three phases**: (1) domain isolation (epochs 1–20, hard domain routing, no cross-domain gradient flow), (2) soft mixing (epochs 20–50, linear interpolation from hard to learned routing, enable PCGrad), (3) full integration (epochs 50–100, learned routing with CAGrad, gradually decrease router temperature). The composite loss is:

> **L_total = Σ_d w_d·L_JEPA_d + 0.1·L_etf + 0.01·L_balance + 0.01·L_entropy**

where L_JEPA_d is per-domain prediction loss and w_d uses dynamic task weighting (GradNorm or FAMO). One important caveat from ERMoE (2025): **strong load-balancing losses can hurt performance** by fostering over-uniform routing. Prefer ETF loss for expert differentiation and use minimal balancing constraints.

---

## 5. Quasimetric AUROC has no published baselines — you are defining the metric

The specific evaluation metric — temporal contrastive discrimination AUROC using a quasimetric distance on robot navigation latent representations — appears to be **a novel evaluation approach with no published baselines on RECON or comparable datasets**. Existing quasimetric and contrastive RL work evaluates via success rate and value function error, not AUROC.

The closest related methods are **QRL** (Wang et al., ICML 2023), which learns quasimetric value functions using Interval Quasimetric Embeddings (IQE) or Metric Residual Networks (MRN), and **Temporal Metric Distillation** (Myers et al., 2025), which combines contrastive successor features with MRN parameterization and outperforms both QRL and Contrastive RL by >3× on stitching environments. Neither reports AUROC. **ATC** (Augmented Temporal Contrast, Stooke et al., 2021) achieves >90% accuracy on temporal discrimination in Atari but does not report AUROC on navigation data.

Based on published discrimination accuracy values across adjacent domains, expected AUROC ranges for your evaluation are:

| Configuration | Expected AUROC |
|---|---|
| Random encoder (baseline) | ~0.50 |
| Frozen DINOv2, L2 distance in feature space | 0.60–0.75 |
| Frozen DINOv2 + learned quasimetric head (no triangle inequality) | 0.75–0.85 |
| Frozen DINOv2 + trained MRN/IQE model | 0.80–0.92 |
| End-to-end trained contrastive encoder | 0.85–0.95 |

These estimates draw on DINOv2 linear probe results (**0.763 AUROC** on NIH Chest X-rays, frozen), large reward model temporal contrast scores (~0.950 in manipulation), and ATC's >90% temporal discrimination accuracy. The **RECON dataset** (Shah et al., CoRL 2021) contains 5,000+ trajectories across 9 real-world outdoor environments around Berkeley, ~50GB, with significant seasonal/lighting variation. **RAE-NWM** (2025) is the most relevant DINOv2-on-RECON baseline, reporting ATE = 1.36 and RPE = 0.37 in DINOv2 representation space, but no AUROC.

Discrimination performance should vary with temporal distance Δ: easiest at very small Δ (clearly reachable) and very large Δ (clearly unreachable), hardest at intermediate values near the reachability boundary. RECON's dense off-road vegetation creates stochastic textures that DINOv2 struggles with (documented in RAE-NWM), which will suppress AUROC relative to structured indoor environments. **Asymmetry is a key test**: quasimetric AUROC should show d(A→B) ≠ d(B→A) for uphill/downhill or directional paths — report AUROC separately for forward and reversed pairs.

For a recommended evaluation protocol: sample positive pairs from trajectory segments at varying Δ (1s, 5s, 10s, 30s), sample negatives from different trajectories, report AUROC as a function of Δ, and compare frozen DINOv2 cosine similarity versus learned linear quasimetric head versus full MRN model. This will produce the first published baselines for this metric.

---

## Conclusion

The five failure modes sit on a spectrum from well-characterized to genuinely novel. MoE router collapse has production-grade solutions: the Switch Transformer balance loss (α = 0.01) plus ST-MoE z-loss (β = 0.001), monitored via normalized entropy > 0.85, validated through small-scale proxy pretests. The GRASP planner's horizon scaling is documented but its compute-performance tradeoffs at specific latent dimensions remain unmapped — the < 10ms CPU target likely requires H ≤ 12 with a lightweight MLP world model. Stable-worldmodel's 50-step budget is the single most important configuration detail to understand when wrapping custom models; default to CEM with 300 samples and test on both expert and random goal distributions. Cross-domain JEPA training requires assembling gradient surgery (PCGrad/CAGrad), domain-specific normalization (DSBN), and expert differentiation (ETF loss with FOFS) into a phased training schedule — no off-the-shelf solution exists. Quasimetric AUROC on navigation data is a frontier metric where you will be establishing the baselines, with frozen DINOv2 likely scoring 0.60–0.75 before task-specific training.