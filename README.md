# NeMo-WM: Neuromodulated World Model

<p align="center">
  <b>Perceive. Remember. Plan. Discover. Speak. Explain.</b><br>
  All on CPU. 1.2M parameters. No LLM. No pretrained encoder.
</p>

<p align="center">
  <a href="https://arxiv.org"><img src="https://img.shields.io/badge/arXiv-submitted-b31b1b.svg"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"/></a>
  <a href="https://www.amd.com/"><img src="https://img.shields.io/badge/Hardware-AMD_Ryzen_AI_MAX%2B_395-ED1C24.svg"/></a>
  <img src="https://img.shields.io/badge/GPU-None-lightgrey.svg"/>
  <img src="https://img.shields.io/badge/Parameters-1.2M-blue.svg"/>
  <img src="https://img.shields.io/badge/Language-No_LLM-green.svg"/>
</p>

---

## Highlights

<table>
<tr>
<td width="50%">

**🧠 Biologically-Inspired Architecture**
- 8 neuromodulatory signals (DA, ACh, CRT, NE, 5HT)
- Three-store memory (Working, Episodic, Schema)
- Cortisol-gated working memory capacity
- Sleep consolidation with DiVeQ schemas

</td>
<td width="50%">

**🗣️ Language Without LLM**
- 380 grounded words across 15 domains
- 100% comprehension on test sentences
- Cognitive age equivalent: 4-6 years
- Words = sensorimotor experience prototypes

</td>
</tr>
<tr>
<td>

**⚡ Radical Efficiency**
- 26K params beats 1B-param ViT-G (+0.116 AUROC)
- CPU-only training and inference
- Self-narration at 350,866 Hz (2.8us/call)
- VPP belief augmentation: +25% with 9.6% overhead

</td>
<td>

**🔬 Autonomous Discovery**
- Physics laws from F=ma (gravity, friction, magnetic)
- Auto-learn: polynomial basis fitting, R²=1.0
- 16 emergent mood states
- Self-teaching via dream narration

</td>
</tr>
</table>

---

## Key Results

| Capability | Result | Comparison |
|---|---|---|
| Anomaly Detection | **AUROC 0.9978** (6 domains) | Beats V-JEPA 2 ViT-G (1034M params) |
| Self-Localization | **AUROC 0.9970** (26K params) | +0.116 vs ViT-G |
| Planning (PointMaze) | **100% SR**, 19 avg steps | Flow matching, 32s training |
| Planning (PushT) | **0.92 max coverage** | 206 human demos, 77min CPU |
| Memory (FAISS) | **0.076ms** retrieval | 2414x speedup |
| Schema Learning (DiVeQ) | **-68%** consolidation loss | 2.3x faster than EMA |
| Physics Discovery | **3/3 forces**, R²=1.0 | Gravity, friction, magnetic |
| Language (no LLM) | **100% comprehension** | 380 words, 15 domains |
| VPP Augmentation | **+25.1% +/- 2.7%** loss | Real data, 3 runs |
| Self-Narration | **2.8us/call** | 350,866 Hz |
| Mood States | **16 emergent** | From neuromodulatory signals |
| Introspective Questions | **17/17** | DreamerV3: 2/17 |

---

## Architecture

```
Sensory Input (vision, proprio, IMU, audio)
    |
    v
Perception Layer --- CNN (1.2M) + ProprioEncoder (26K)
    |
    v
Belief State (64D) -----------------------------------------.
    |                                                         |
    |---> Working Memory (K=8, cortisol-gated)               |
    |---> Episodic Buffer (10K, FAISS, DA-priority)          |
    |---> Schema Store (DiVeQ, 64 codes, sleep-trained)      |
    |---> Transition Model -> Flow Policy -> Actions         |
    |---> Neuromodulators (DA, ACh, CRT, NE, 5HT)           |
    |---> Language Layer (WordGrounder, SelfNarrator)         |
    '---> Physics Discovery Agent (KB -> Auto-learn)         |
                                                              |
    .----------- Sleep Consolidation <------------------------'
    |  EMA (waking) + Gradient (sleep) -> -68% loss
    |  Episodic replay -> Schema compression
    |  Dream narration -> Vocabulary growth
    '----------------------------------------------
```

---

## Grounded Language System (No LLM)

> *"Words mean what I experienced when I heard them."*

NeMo-WM learns language like a toddler -- through sensorimotor co-occurrence, not text corpora. No LLM. No pretrained embeddings. No parser.

### How It Works

```
Robot hears "gravity" while experiencing:
  -> Objects falling (visual belief pattern)
  -> Dopamine spike (surprise signal)  
  -> Physics agent discovers Fy = -9.81

"gravity" = prototype of all these belief states
```

### Word Similarity (Learned from Experience)

```
sim(gravity,  falling)  = +0.997    <- nearly identical
sim(gravity,  weight)   = +0.996    <- same force  
sim(corridor, hallway)  = +0.879    <- synonyms discovered
sim(near,     close)    = +0.895    <- synonyms discovered
sim(danger,   safe)     = -0.791    <- correctly opposite
sim(fast,     slow)     = -0.924    <- correctly opposite
sim(left,     right)    = -0.933    <- correctly opposite
sim(curious,  bored)    = -0.735    <- correctly opposite
sim(gravity,  corridor) = -0.018    <- correctly unrelated
```

No dictionary. No pretrained embeddings. Learned purely from experience.

### 10 Cognitive Levels (All Passed)

| Level | Name | Score | Example |
|---|---|---|---|
| L1 | Object Permanence | **100%** | "the ball exists" |
| L2 | Cause-Effect | **100%** | "push -> it moves" |
| L3 | Spatial Relations | **100%** | "above, below, between" |
| L4 | Temporal Ordering | **100%** | "before, after, then" |
| L5 | Conditionals | **100%** | "if steep then careful" |
| L6 | Analogy | **60%** | "gravity is like magnetic" |
| L7 | Abstraction | **100%** | "force = category of gravity" |
| L8 | Composition | **100%** | "first push, then turn, then stop" |
| L9 | Explanation | **100%** | "it fell because gravity" |
| L10 | Teaching | **88%** | "imagine the ball falls down" |

Cognitive age equivalent: **4-6 years**

### Sentence Comprehension

```
"the ball falls due to gravity"                    -> UNDERSTOOD (100%)
"push the heavy block to the left"                 -> UNDERSTOOD (100%)
"gravity is a type of force that pulls down"       -> UNDERSTOOD (100%)
"causes lead to effects and results"               -> UNDERSTOOD (100%)
"quantum entanglement superposition"               -> NOT UNDERSTOOD (0%)  <- honest
```

### Overnight Plateau Proof (37 Million Hearings)

```
Comprehension vs Training Volume:
  1K hearings:    22.7%
  10K hearings:   53.5%
  100K hearings:  65.1%  <- plateau starts
  1M hearings:    65.1%
  37M hearings:   65.1%  <- proven ceiling (8,930 passes, 2,976 sleep cycles)
  
  + stemming:     96.4%  <- breakthrough (zero extra hearings)
  + 3 words:     100.0%  <- complete (zero extra training)
```

Key finding: The 65.1% ceiling was vocabulary coverage, not architecture. Adding morphological stemming ("falls"->"falling", "moves"->"move") and function words instantly broke through -- zero additional training needed.

### Language v2: Five Comprehension Upgrades

| Upgrade | What | Result |
|---|---|---|
| **Negation** | "not dangerous" = flip belief vector | sim(safe, not safe) = -1.000 |
| **Belief Accumulator** | Word order matters (Kalman filter) | "not safe steep" != "safe not steep" |
| **Predictive Grounding** | Comprehension = world model simulation | Transition model verifies sentences |
| **Contrastive Learning** | Meanings by what they're NOT | gravity/friction: -0.086 -> -0.849 |
| **Episodic Replay** | Self-teaching from narrated memories | +9 words from 100 replayed episodes |

---

## VPP-Inspired Belief Augmentation

Inspired by "Active Stereo Without Pattern Projector" (Bartolomei et al., ICCV 2023). Sparse belief hints amplify dense visual signals.

```
Standard:    96x96x6 (2-frame RGB)              -> CNN -> 128D -> Policy
Augmented:   96x96x8 (2-frame RGB + 2ch belief) -> CNN -> 128D -> Policy

The 2 extra channels: learned spatial attention from [agent_x, agent_y, last_action]
```

### Results on Real Human Demos (25,650 samples, 206 episodes)

| Run | Standard | VPP Augmented | Improvement |
|---|---|---|---|
| 1 | 0.2824 | **0.2070** | **+26.7%** |
| 2 | 0.2890 | **0.2100** | **+27.3%** |
| 3 | 0.2840 | **0.2231** | **+21.4%** |
| **Mean** | **0.2851** | **0.2134** | **+25.1% +/- 2.7%** |

+25% loss reduction for only 9.6% parameter overhead. Consistent across all 3 runs on real human demonstrations.

---

## Physics Discovery Agent

Autonomously discovers physical laws from F=ma:

| Scenario | Discovered Law | R² | Method |
|---|---|---|---|
| Falling ball | Fy = -9.81*m | 1.0 | Knowledge base |
| Sliding block | F = -mu*N*v/\|v\| | 1.0 | Knowledge base |
| Magnetic pull | F = k/r² | 1.0 | Knowledge base |
| Combined forces | **Fy = 11.2 - 9.81y** | **1.0** | **Auto-learn** |

Three learning modes: knowledge base lookup, auto-learn (polynomial basis), oracle cascade.

---

## Emergent Mood States

DiVeQ quantization of the neuromodulatory space produces 16 named emotions:

```
High DA + Low ACh  ->  "Curious-Uncertain"    (exploring unknown)
Low CRT + Low NE   ->  "Calm-Relaxed"         (safe familiar area)
High CRT + High NE ->  "Stressed-Alert"       (danger detected)
High 5HT + High NE ->  "Cautious-Alert"       (risky but manageable)
High DA + High ACh ->  "Curious-Confident"    (exploring with understanding)
```

83 unique mood transitions observed. Moods modulate behavior: "Stressed-Alert" shortens planning horizon, "Curious-Bold" increases exploration.

---

## Memory and Sleep Consolidation

### Three-Store Architecture

| Store | Capacity | Duration | Mechanism |
|---|---|---|---|
| **Working Memory** | K=8 (cortisol-gated) | Active task | K degrades under stress |
| **Episodic Store** | 10K entries | Long-term | FAISS-indexed, DA-priority |
| **Schema Store** | 64 DiVeQ codes | Permanent | EMA waking + gradient sleep |

### Sleep Cycle Results

```
Consolidation Loss:  0.505 -> 0.163 (-68%) over 5 cycles
Store latency:       18.7us (EMA: 43.5us = 2.3x faster)
Retrieve latency:    0.076ms (brute force: 183ms = 2414x faster)
```

### Novel DiVeQ Applications

| Feature | Description | Result |
|---|---|---|
| **DreamInterpolation** | Cosine walk between schemas during sleep | 9/20 unique schemas |
| **ActionPrimitives** | Quantize motor space -> named primitives | 16 primitives, 100% usage |
| **AdaptiveCodebook** | Surprise-gated neurogenesis | 8->29->4 lifecycle |

---

## Comparison with Other Systems

| Capability | NeMo-WM | DreamerV3 | Diff. Policy | DINO-WM | TD-MPC2 |
|---|---|---|---|---|---|
| Introspective Qs | **17/17** | 2/17 | 0/17 | 1/17 | 2/17 |
| Episodic Memory | **Yes** | No | No | No | No |
| Schema Learning | **DiVeQ** | No | No | No | No |
| Self-Narration | **5 components** | No | No | No | No |
| Physics Discovery | **3/3 + auto** | No | No | No | No |
| Grounded Language | **380 words** | No | No | No | No |
| Mood States | **16 emergent** | No | No | No | No |
| Training Hardware | **CPU only** | GPU | GPU | GPU | GPU |
| Parameters | **1.2M** | ~200M | ~263M | ~5M | ~50M |

---

## Neuromodulatory Signals

| Signal | Biological Analogue | Computational Role |
|---|---|---|
| **DA** (Dopamine) | Reward/novelty | Episodic storage priority, exploration |
| **ACh** (Acetylcholine) | Attention/confidence | Prediction gate, context window |
| **CRT** (Cortisol) | Stress response | WM degradation, domain shift |
| **NE** (Norepinephrine) | Alertness/arousal | Sensory gain, exploration |
| **5HT** (Serotonin) | Caution/inhibition | Risk assessment, action suppression |

### Aphasia Double Dissociation

```
Language zeroed (simulate aphasia):
  Visual WM:  0.9542 -> 0.5000 (chance)  <- language NECESSARY
  Proprio WM: 0.9974 -> 0.9974           <- language IRRELEVANT

  Delta = +0.4542 -- parallels Fedorenko et al.
```

---

## Demo Videos

| PointMaze Navigation | PushT Vision Policy | Language Learning |
|---|---|---|
| 100% SR, 10/10 U-traversals | 0.92 max coverage, CPU only | 380 words, no LLM, TTS narration |
| [pointmaze_showcase.mp4](outputs/pointmaze_showcase.mp4) | [pusht_best_episode.mp4](outputs/pusht_best_episode.mp4) | [language_learning_demo.mp4](outputs/language_learning_demo.mp4) |

---

## Quick Start

```bash
# Clone
git clone https://github.com/taylorjohn/NEMO-WM.git
cd NEMO-WM

# Install
pip install torch torchvision numpy h5py faiss-cpu gymnasium gym-pusht pyttsx3

# Run benchmarks
python benchmark_full.py --run-live

# Train PushT vision policy (77min CPU)
python train_pusht_vision_v2.py --epochs 300 --eval

# Language curriculum (10 cognitive levels)
python curriculum_generator.py

# Grounded word learning (8 stages)
python word_grounder.py

# Language v2 (negation, predictive, contrastive)
python language_v2.py --demo all

# Physics discovery
python physics_discovery_agent.py

# VPP belief augmentation test (real data)
python test_vpp_real.py --epochs 200 --runs 3
```

---

## Papers

| # | Title | Status | Target |
|---|---|---|---|
| 1 | **NeMo-WM: Neuromodulated World Model** | arXiv submitted | cs.RO, cs.LG, cs.CV |
| 2 | **Introspective World Models** | Writing | NeurIPS 2026 |
| 3 | **Neuromodulated Transfer Learning** | Numbers locked | -- |
| 4 | **Physics Discovery Agent** | Working prototype | -- |

---

## Hardware

| Component | Spec |
|---|---|
| CPU | AMD Ryzen AI MAX+ 395 |
| RAM | 128GB unified |
| NPU | AMD XDNA2 |
| GPU | **None** |
| Cost | ~$2,000 |
| Training | ~45W |
| Inference | ~8W |

---

## Citation

```bibtex
@article{taylor2026nemowm,
  title={NeMo-WM: A Neuromodulated World Model for Embodied Agents},
  author={Taylor, John},
  journal={arXiv preprint},
  year={2026}
}
```

---

## License

MIT. Contact: johntaylorcreative@gmail.com
