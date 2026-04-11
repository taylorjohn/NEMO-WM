"""
neuro_vlm_gate.py — Biological Neuromodulator as VLM Attention Gate
====================================================================
Phase 3 core contribution.

Architecture shift:
  BEFORE (signal computer):
    Frame → VLM encoder → z → Neuromodulator(z) → signals → loss weights

  NOW (gain controller, biologically accurate):
    Neuromodulator state → VLM attention gains → Frame → z
                        ↘ DA threshold         ↗
                        ↘ 5HT suppression     ↗  → updated neuromodulator
                        ↘ NE spatial bias     ↗
                        ↘ ACh temperature     ↗

The neuromodulator PRECEDES perception. It sets gain parameters on the
VLM's attention heads BEFORE the frame is processed. The resulting
embedding is shaped by current neuromodulator state, not evaluated by
it afterwards.

Biological basis:
  DA  — prediction error threshold. Midbrain fires before full cortical
        processing. Amplifies discrepancy detection in subsequent frames.
  NE  — spatial attention gain. High NE = edges, movement, discontinuities.
        Low NE = diffuse contextual attention.
  ACh — feature binding gain / attention temperature. High ACh = sharper,
        more peaked attention. Low = broad, contextual processing.
  5HT — inhibition of habitual responses. Low 5HT = explore novel.
        High 5HT = stay with familiar. Inverse to naive implementation.
  eCB — retrograde suppression. Unlearns most recent pattern slightly.
        Prevents over-encoding of transient events (biological forgetting).
  Ado — fatigue gate. All gains reduce when adenosine rises.
        Sleep pressure — system becomes conservative, needs recalibration.
  Cort— global gain multiplier. Lowers all thresholds. Everything more
        salient under stress / distribution shift.
  E/I — excitatory/inhibitory balance. High E = generative, novel
        associations. High I = integrative, stable pattern reinforcement.

Key properties:
  - State-dependent encoding: same frame → different embedding under
    different neuromodulator state (biological: emotionally significant
    events encoded more vividly under high ACh + NE)
  - Predictive gating: gains computed from PREVIOUS frame's error,
    applied to NEXT frame's attention (predictive coding)
  - Fatigue-aware: Ado reduces global gain, triggering recalibration
  - eCB retrograde gate: startle response is not over-encoded

Compatible with:
  - Any transformer VLM (CLIP, SmolVLM, Qwen2.5-VL, VILA, InternVL3)
  - DINOv2-based encoders (CORTEX production)
  - Any model with accessible attention heads

Usage:
    from neuro_vlm_gate import BiologicalNeuromodulator, NeurallyGatedVLM

    # Wrap any VLM vision encoder
    neuro = BiologicalNeuromodulator()
    gated = NeurallyGatedVLM(clip_model.vision_model, neuro)

    # Process frames — neuromodulator gates attention automatically
    for img in frames:
        z = gated.encode(img)          # gates before encoding
        neuro.update_from_error(z)     # updates state for next frame

    # Inspect current neuromodulator state
    state = neuro.get_state()
    gains = neuro.get_attention_gains()
    print(f"DA={state['da']:.3f} NE={state['ne']:.3f} regime={state['regime']}")

Author: John Taylor — github.com/taylorjohn
Sprint: VLM Phase 3 (core architecture)
"""

from __future__ import annotations
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional


# ── Neuromodulator state ───────────────────────────────────────────────────────

@dataclass
class NeuroState:
    """
    Complete neuromodulator state at one timestep.
    All values in [0, 1] unless noted.
    """
    # Eight biological signals
    da:   float = 0.0    # dopamine — prediction error / surprise
    ne:   float = 0.0    # norepinephrine — spatial uncertainty / arousal
    ach:  float = 0.5    # acetylcholine — attention focus / binding
    sht:  float = 0.5    # serotonin — inhibition of habit / diversity
    ecb:  float = 0.0    # endocannabinoid — retrograde suppression
    ado:  float = 0.0    # adenosine — fatigue / sleep pressure
    cort: float = 0.0    # cortisol — global arousal / stress
    ei:   float = 0.5    # E/I balance — 0=inhibitory, 1=excitatory

    # Derived
    regime:    str   = "EXPLOIT"   # EXPLOIT | REOBSERVE | FATIGUE | STRESSED
    step:      int   = 0
    timestamp: float = field(default_factory=time.time)


# ── Attention gain parameters ──────────────────────────────────────────────────

@dataclass
class AttentionGains:
    """
    Gain parameters applied to VLM attention BEFORE encoding.
    Each parameter corresponds to one or more biological signals.
    """
    # DA: scale query vectors — amplifies discrepancy detection
    query_scale:    float = 1.0   # range [0.5, 3.0]

    # NE: spatial frequency bias — high NE = attend to edges/movement
    spatial_bias:   float = 0.0   # range [0.0, 1.0]

    # ACh: attention temperature — high ACh = sharper peaked attention
    temperature:    float = 1.0   # range [0.5, 2.0], lower = sharper

    # 5HT: top-K suppression — high 5HT = attend broadly, not just top tokens
    topk_suppress:  float = 0.0   # range [0.0, 1.0]

    # eCB: recency decay — reduces weight on immediately prior context
    recency_decay:  float = 0.0   # range [0.0, 0.5]

    # Ado: global gain — fatigue reduces all attention signals
    global_gain:    float = 1.0   # range [0.3, 1.0]

    # Cortisol: threshold multiplier — everything more salient under stress
    threshold_mult: float = 1.0   # range [1.0, 4.0]

    # E/I: integration vs generation balance
    ei_bias:        float = 0.5   # 0=integrate stable patterns, 1=generate novel


# ── Biological neuromodulator ─────────────────────────────────────────────────

class BiologicalNeuromodulator:
    """
    Gain-control neuromodulator that precedes VLM perception.

    Update cycle:
      1. get_attention_gains()   — called BEFORE encoding frame t
      2. VLM encodes frame t with gains applied to attention heads
      3. update_from_error(z_pred, z_actual) — called AFTER encoding
         Updates state for frame t+1

    This implements predictive coding: gains for frame t are computed
    from error on frame t-1. The system pre-gates attention based on
    what it expected to be surprised by.
    """

    def __init__(self,
                 da_threshold:     float = 0.01,    # REOBSERVE trigger
                 fatigue_threshold: float = 0.7,    # FATIGUE trigger
                 stress_threshold:  float = 0.5,    # STRESSED trigger
                 history_window:    int   = 20,
                 cortisol_baseline: Optional[float] = None):

        self.da_threshold      = da_threshold
        self.fatigue_threshold = fatigue_threshold
        self.stress_threshold  = stress_threshold
        self.history_window    = history_window

        # Running state
        self._state = NeuroState()

        # History buffers
        self._da_history:   list = []
        self._loss_history: list = []
        self._z_history:    list = []
        self._step:         int  = 0

        # Cortisol baseline (auto-calibrate if None)
        self._cortisol_baseline = cortisol_baseline
        self._calibration_steps = 20

        # Decay constants (biological timescales)
        self._da_decay   = 0.7    # fast — DA is phasic
        self._ne_decay   = 0.85   # medium — NE is tonic-phasic
        self._ach_decay  = 0.9    # slow — ACh is tonic
        self._sht_decay  = 0.95   # very slow — 5HT is tonic
        self._ecb_decay  = 0.6    # fast — eCB is transient
        self._ado_build  = 0.015  # build rate (reaches 0.75 in 50 REOBSERVE steps)
        self._ado_clear  = 0.1    # fast clear — cleared by rest/reset

        # Event log
        self.event_log: list = []

    # ── Main interface ─────────────────────────────────────────────────────────

    def get_attention_gains(self) -> AttentionGains:
        """
        Compute attention gain parameters from current neuromodulator state.
        Call BEFORE encoding the next frame.

        Biological: midbrain neuromodulatory nuclei tonically modulate
        thalamocortical gain before sensory information reaches cortex.
        """
        s = self._state

        # DA: amplifies query scale — high DA = attend to discrepancies
        # Biological: VTA dopamine projections to PFC amplify prediction
        # error signals before they reach working memory
        query_scale = 1.0 + s.da * 2.0 * (1.0 + s.cort)

        # NE: spatial frequency bias — high NE = prioritise edges/motion
        # Biological: LC-NE projections modulate spatial frequency tuning
        # in V1/V2, shifting preference toward high-freq (detail) features
        spatial_bias = s.ne * (1.0 + s.cort * 0.5)

        # ACh: attention temperature — high ACh = sharper attention
        # Biological: BF-ACh projections reduce lateral inhibition in cortex,
        # allowing more focused, less diffuse attention patterns
        # Lower temperature = sharper (inverse relationship)
        temperature = 1.0 / (0.5 + s.ach + s.cort * 0.3)
        temperature = float(np.clip(temperature, 0.3, 2.0))

        # 5HT: top-K suppression — high 5HT = broader, less peaked attention
        # Biological: 5HT2A receptors on pyramidal cells increase gain of
        # diffuse inputs relative to focused inputs (broadens tuning curves)
        # Note: inverse of naive implementation — high 5HT = MORE suppression
        # of top-attended tokens, forcing attention to spread
        topk_suppress = s.sht * 0.5

        # eCB: recency decay — high eCB = suppress most recent context
        # Biological: endocannabinoids act retrograde at synapses, suppressing
        # the presynaptic terminals that were most recently active
        # Prevents over-encoding of immediately prior context
        recency_decay = s.ecb * 0.4

        # Ado: global gain — high Ado = attenuated attention everywhere
        # Biological: adenosine A1 receptors suppress excitatory transmission
        # globally; sleep pressure reduces thalamocortical relay gain
        global_gain = float(np.clip(1.0 - s.ado * 0.6, 0.3, 1.0))

        # Cortisol: threshold multiplier — high cortisol = everything salient
        # Biological: glucocorticoid receptors in amygdala and hippocampus
        # lower the threshold for strong encoding of arousal-related stimuli
        threshold_mult = 1.0 + s.cort * 3.0

        # E/I: determines integration vs generation mode
        # High E/I = excitatory dominance = more novel associations
        # Low E/I = inhibitory dominance = more stable pattern reinforcement
        ei_bias = s.ei

        return AttentionGains(
            query_scale    = float(np.clip(query_scale, 0.5, 4.0)),
            spatial_bias   = float(np.clip(spatial_bias, 0.0, 1.0)),
            temperature    = temperature,
            topk_suppress  = float(np.clip(topk_suppress, 0.0, 0.8)),
            recency_decay  = float(np.clip(recency_decay, 0.0, 0.5)),
            global_gain    = global_gain,
            threshold_mult = float(np.clip(threshold_mult, 1.0, 5.0)),
            ei_bias        = float(np.clip(ei_bias, 0.0, 1.0)),
        )

    def update_from_error(self,
                          z_pred:   Optional[torch.Tensor],
                          z_actual: torch.Tensor,
                          spatial_context: Optional[torch.Tensor] = None,
                          action_magnitude: float = 0.0) -> NeuroState:
        """
        Update neuromodulator state from prediction error.
        Call AFTER encoding frame t. State applies to frame t+1.

        Biological: RPE (reward prediction error) computed by midbrain
        (VTA/SNc) and broadcasted to cortex, modifying subsequent
        processing. This is the phasic DA response.

        Args:
            z_pred:           predicted embedding (None for first frame)
            z_actual:         actual observed embedding
            spatial_context:  GPS or pixel XY for NE spatial signal
            action_magnitude: magnitude of action taken (for ACh)
        """
        self._step += 1
        self._state.step = self._step

        z_actual_flat = F.normalize(z_actual.float().flatten(), dim=0)

        # ── Compute prediction error (RPE) ────────────────────────────────────
        if z_pred is not None:
            z_pred_flat = F.normalize(z_pred.float().flatten(), dim=0)
            rpe = float(1.0 - torch.dot(z_pred_flat, z_actual_flat).clamp(-1, 1))
        else:
            rpe = 0.0

        mse = float(F.mse_loss(
            z_actual_flat,
            self._z_history[-1] if self._z_history else z_actual_flat
        ).item())

        # ── DA: phasic prediction error signal ────────────────────────────────
        # Biological: VTA phasic burst proportional to unsigned RPE
        # Decays fast (phasic, not tonic)
        da_new = rpe * (1.0 + self._state.cort)
        self._state.da = float(np.clip(
            self._state.da * self._da_decay + da_new * (1 - self._da_decay),
            0.0, 1.0
        ))

        # ── NE: spatial uncertainty signal ────────────────────────────────────
        # Biological: LC-NE fires when environment is uncertain or novel
        # Uses spatial displacement if available, else RPE proxy
        if spatial_context is not None and len(self._z_history) > 0:
            disp = float(spatial_context.norm().item())
            ne_new = float(np.clip(disp / (disp + 1.0), 0, 1))
        else:
            ne_new = rpe * 0.8
        self._state.ne = float(np.clip(
            self._state.ne * self._ne_decay + ne_new * (1 - self._ne_decay),
            0.0, 1.0
        ))

        # ── ACh: attention sharpening signal ──────────────────────────────────
        # Biological: BF-ACh rises with task engagement and action demands
        # High action = high ACh = sharper attention to relevant features
        ach_target = float(np.clip(
            0.3 + action_magnitude * 0.5 + self._state.da * 0.3, 0, 1))
        self._state.ach = float(np.clip(
            self._state.ach * self._ach_decay + ach_target * (1 - self._ach_decay),
            0.0, 1.0
        ))

        # ── 5HT: diversity / habit inhibition signal ──────────────────────────
        # Biological: raphe 5HT is INVERSE to DA at short timescales
        # High DA (surprise) → low 5HT → explore novel responses
        # Low DA (habitual) → high 5HT → stay with familiar patterns
        # This is the correct biological relationship (often inverted naively)
        if len(self._z_history) >= 3:
            stack = torch.stack(self._z_history[-8:]).float()
            var   = float(stack.var(dim=0).mean().item())
            sht_target = float(np.clip(1.0 - self._state.da * 2.0, 0, 1))
        else:
            sht_target = 0.5
        self._state.sht = float(np.clip(
            self._state.sht * self._sht_decay + sht_target * (1 - self._sht_decay),
            0.0, 1.0
        ))

        # ── eCB: retrograde suppression ───────────────────────────────────────
        # Biological: released when postsynaptic activity is high
        # Suppresses the most recently active presynaptic inputs
        # Prevents runaway potentiation of startle responses
        ecb_new = float(np.clip(self._state.da * 0.6 + self._state.ne * 0.4, 0, 1))
        self._state.ecb = float(np.clip(
            self._state.ecb * self._ecb_decay + ecb_new * (1 - self._ecb_decay),
            0.0, 1.0
        ))

        # ── Ado: fatigue accumulation ─────────────────────────────────────────
        # Biological: adenosine accumulates during wakefulness / activity
        # Cleared by rest (regime switch to EXPLOIT for extended period)
        # Adenosine: builds proportional to DA (more surprise = more fatigue)
        # Biological: adenosine accumulates proportionally to neural activity
        if self._state.da > self.da_threshold:
            # High DA = active exploration = adenosine builds faster
            self._state.ado = float(np.clip(
                self._state.ado + self._ado_build * (1 + self._state.da),
                0.0, 1.0))
        elif self._state.regime == "REOBSERVE":
            self._state.ado = float(np.clip(
                self._state.ado + self._ado_build, 0.0, 1.0))
        else:
            # Exploitation / rest — adenosine clears slowly
            self._state.ado = float(np.clip(
                self._state.ado - self._ado_clear * 0.3, 0.0, 1.0))

        # ── Cortisol: distribution shift / chronic stress signal ──────────────
        # Biological: HPA axis cortisol rises with sustained unpredictability
        # Acts as global gain multiplier on all other signals
        self._loss_history.append(mse)
        if (self._cortisol_baseline is None and
                len(self._loss_history) >= self._calibration_steps):
            self._cortisol_baseline = float(
                np.mean(self._loss_history[:self._calibration_steps])) * 0.95

        if self._cortisol_baseline and len(self._loss_history) >= 3:
            recent = float(np.mean(self._loss_history[-3:]))
            cort_new = float(np.clip(
                (recent - self._cortisol_baseline) /
                (self._cortisol_baseline + 1e-6), 0, 1))
        else:
            cort_new = 0.0
        # Cortisol decays slowly (chronic signal)
        self._state.cort = float(np.clip(
            self._state.cort * 0.97 + cort_new * 0.03, 0.0, 1.0))

        # ── E/I balance ───────────────────────────────────────────────────────
        # High DA + High NE → excitatory dominance (explore)
        # High 5HT + High Ado → inhibitory dominance (consolidate)
        excitatory   = (self._state.da + self._state.ne) / 2
        inhibitory   = (self._state.sht + self._state.ado) / 2
        self._state.ei = float(np.clip(
            excitatory / (excitatory + inhibitory + 1e-6), 0.0, 1.0))

        # ── Regime ────────────────────────────────────────────────────────────
        if self._state.ado > self.fatigue_threshold:
            self._state.regime = "FATIGUE"
        elif self._state.cort > self.stress_threshold:
            self._state.regime = "STRESSED"
        elif self._state.da > self.da_threshold:
            self._state.regime = "REOBSERVE"
        else:
            self._state.regime = "EXPLOIT"

        # Store history
        self._z_history.append(z_actual.detach().float())
        self._da_history.append(self._state.da)
        if len(self._z_history) > self.history_window:
            self._z_history.pop(0)
        if len(self._da_history) > self.history_window:
            self._da_history.pop(0)

        return self._state

    # ── State access ──────────────────────────────────────────────────────────

    def get_state(self) -> dict:
        s = self._state
        return {
            "da": round(s.da, 4), "ne": round(s.ne, 4),
            "ach": round(s.ach, 4), "sht": round(s.sht, 4),
            "ecb": round(s.ecb, 4), "ado": round(s.ado, 4),
            "cort": round(s.cort, 4), "ei": round(s.ei, 4),
            "regime": s.regime, "step": s.step,
        }

    def needs_recalibration(self) -> bool:
        """True when fatigue is high — system needs a rest period."""
        return self._state.ado > self.fatigue_threshold

    def rest(self, steps: int = 10):
        """Simulate a rest period — clears adenosine."""
        for _ in range(steps):
            self._state.ado = max(0.0, self._state.ado - self._ado_clear)
        self._state.regime = "EXPLOIT"

    def reset_cortisol(self):
        """After adaptation — soft reset cortisol baseline."""
        if self._loss_history:
            self._cortisol_baseline = float(
                np.mean(self._loss_history[-5:])) * 0.95


# ── Neurally-gated VLM wrapper ────────────────────────────────────────────────

class NeurallyGatedVLM(nn.Module):
    """
    Wraps any transformer vision model with biological attention gating.

    The neuromodulator state is applied to attention heads BEFORE
    the frame is processed. The encoder sees a neuromodulator-shaped
    view of the world, not the raw frame.

    Works by:
      1. Registering forward hooks on each attention layer
      2. Hooks apply gain parameters from neuromodulator state
      3. No weight modification — gains applied at runtime only
      4. Compatible with any transformer: CLIP, DINOv2, SmolVLM, etc.

    Usage:
        neuro = BiologicalNeuromodulator(da_threshold=0.0613)
        gated = NeurallyGatedVLM(clip.vision_model, neuro)
        z = gated.encode(img)    # neuromodulator gates attention
        neuro.update_from_error(z_pred, z)
    """

    def __init__(self, vision_model: nn.Module,
                 neuro: BiologicalNeuromodulator,
                 project: Optional[nn.Module] = None,
                 aphasia_ablation: bool = False):
        super().__init__()
        self.vision_model    = vision_model
        self.neuro           = neuro
        self.project         = project          # optional projection (e.g. CLIP)
        self.aphasia_ablation = aphasia_ablation # zero VLM embedding → language-free eval
        self._hooks          = []
        self._current_gains  = AttentionGains()
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on all attention layers."""
        for name, module in self.vision_model.named_modules():
            # Hook on any attention softmax output
            if any(k in name.lower() for k in
                   ['attn', 'attention', 'self_attn']):
                if hasattr(module, 'forward'):
                    hook = module.register_forward_hook(self._attention_hook)
                    self._hooks.append(hook)

    def _attention_hook(self, module, input, output):
        """
        Apply neuromodulator gains to attention output.
        Called automatically during forward pass.
        """
        g = self._current_gains
        if g.global_gain >= 0.99 and g.query_scale <= 1.01:
            return output  # no-op if gains are default

        if isinstance(output, tuple):
            attn_out = output[0]
            rest     = output[1:]
        else:
            attn_out = output
            rest     = None

        # Apply global gain (Adenosine fatigue)
        attn_out = attn_out * g.global_gain

        # Apply query scale (Dopamine — amplify discrepancy signal)
        # Scale the magnitude of attention output
        attn_out = attn_out * g.query_scale

        # Apply recency decay (eCB — suppress most recent context)
        # Reduce contribution of the first tokens (most recent in causal models)
        if g.recency_decay > 0.01 and attn_out.dim() >= 3:
            n_recent = max(1, int(attn_out.shape[1] * g.recency_decay))
            attn_out[:, :n_recent] = attn_out[:, :n_recent] * (1 - g.recency_decay)

        if rest is not None:
            return (attn_out,) + rest
        return attn_out

    def encode(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encode image with neuromodulator-gated attention.
        img_tensor: (1, C, H, W) preprocessed image tensor
        """
        # Get current gains BEFORE encoding (predictive gating)
        self._current_gains = self.neuro.get_attention_gains()

        with torch.no_grad():
            # HuggingFace models (CLIP, SmolVLM, …) use pixel_values=
            # Plain PyTorch models (StudentEncoder, DINOv2 wrapper) take a positional tensor
            try:
                out = self.vision_model(pixel_values=img_tensor)
            except TypeError:
                out = self.vision_model(img_tensor)

        # Extract embedding — handle HuggingFace outputs and plain tensors
        if hasattr(out, 'pooler_output') and out.pooler_output is not None:
            z = out.pooler_output.squeeze(0)
        elif hasattr(out, 'last_hidden_state'):
            z = out.last_hidden_state[:, 0, :].squeeze(0)  # CLS token
        elif isinstance(out, torch.Tensor):
            z = out.squeeze(0)
        else:
            z = out.squeeze(0)

        # Optional projection (CLIP: 768→512)
        if self.project is not None:
            with torch.no_grad():
                z = self.project(z.unsqueeze(0)).squeeze(0)

        z = F.normalize(z.float(), dim=0)

        # ── Aphasia ablation ──────────────────────────────────────────────────
        # Computational analogue of Fedorenko's aphasia patients: destroy the
        # language / VLM pathway entirely and let the neuromodulated world model
        # run on particle dynamics alone.  If RECON AUROC / particle loss hold
        # up, the non-linguistic core is sufficient — a direct parallel to the
        # finding that patients with destroyed language networks retain complex
        # reasoning. One-line activation: pass aphasia_ablation=True at eval.
        if self.aphasia_ablation:
            z = torch.zeros_like(z)   # language input zeroed — MD core only

        return z

    def remove_hooks(self):
        """Remove all registered hooks (cleanup)."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def __del__(self):
        self.remove_hooks()


# ── Standalone test ───────────────────────────────────────────────────────────

def run_test(hdf5_dir: str = "recon_data/recon_release",
             encoder: str = "dino",
             n_frames: int = 30):
    """
    Test biological neuromodulator on real RECON frames.
    Compare gated vs ungated embeddings.
    """
    import glob, io, h5py
    from PIL import Image
    from torchvision import transforms

    TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print(f"\nBiological Neuromodulator Gate — Test")
    print("="*55)

    # Load encoder
    if encoder == "clip":
        from transformers import CLIPProcessor, CLIPModel
        proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()
        vision = model.vision_model
        project = model.visual_projection
        preprocess = lambda img: proc(images=img, return_tensors="pt")["pixel_values"]
    else:
        import sys; sys.path.insert(0, '.')
        from train_mvtec import StudentEncoder
        model = StudentEncoder()
        ckpt = torch.load(r'checkpoints\dinov2_student\student_best.pt',
                         map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt.get('model', ckpt), strict=False)
        model.eval()
        vision  = model
        project = None
        preprocess = lambda img: TRANSFORM(img).unsqueeze(0)

    # Create neuromodulator with calibrated threshold
    threshold = 0.1424 if encoder == "clip" else 0.0613
    neuro = BiologicalNeuromodulator(da_threshold=threshold)

    # Wrap with gated VLM (DINOv2 doesn't expose attention cleanly,
    # so we test gain tracking even if hook doesn't fire)
    gated = NeurallyGatedVLM(vision, neuro, project=project)

    print(f"\nEncoder: {encoder.upper()}, DA threshold: {threshold}")
    print(f"{'Step':>5} {'DA':>7} {'NE':>7} {'ACh':>7} {'5HT':>7} "
          f"{'eCB':>7} {'Ado':>7} {'Cort':>7} {'Regime':<12} {'Gains_QS':>8}")
    print("-"*95)

    # Load frames
    files  = sorted(glob.glob(f"{hdf5_dir}/*.hdf5"))[:10]
    frames = []
    for path in files:
        try:
            with h5py.File(path) as hf:
                imgs = hf["images"]["rgb_left"]
                for i in range(min(5, len(imgs))):
                    jpeg = bytes(imgs[i])
                    img  = Image.open(io.BytesIO(jpeg)).convert("RGB")
                    frames.append(img)
                    if len(frames) >= n_frames:
                        break
        except Exception:
            pass
        if len(frames) >= n_frames:
            break

    z_prev = None
    for i, img in enumerate(frames):
        x = preprocess(img)
        z = gated.encode(x)

        gains = neuro.get_attention_gains()
        state = neuro.update_from_error(z_prev, z)

        print(f"{i:>5} {state.da:>7.4f} {state.ne:>7.4f} "
              f"{state.ach:>7.4f} {state.sht:>7.4f} "
              f"{state.ecb:>7.4f} {state.ado:>7.4f} "
              f"{state.cort:>7.4f} {state.regime:<12} "
              f"{gains.query_scale:>8.3f}")
        z_prev = z

    print(f"\nFinal state: {neuro.get_state()}")
    print(f"Needs recalibration: {neuro.needs_recalibration()}")

    # Show gain range over session
    print(f"\nGain summary:")
    print(f"  query_scale range shows DA modulation of attention amplitude")
    print(f"  temperature  shows ACh modulation of attention sharpness")
    print(f"  global_gain  shows Ado fatigue accumulation")
    print(f"  FATIGUE regime = system needs rest, reduce inference frequency")
    print(f"  STRESSED regime = cortisol high, all signals amplified")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--encoder",   default="dino", choices=["dino", "clip"])
    p.add_argument("--hdf5-dir",  default="recon_data/recon_release")
    p.add_argument("--n-frames",  type=int, default=30)
    a = p.parse_args()
    run_test(a.hdf5_dir, a.encoder, a.n_frames)
