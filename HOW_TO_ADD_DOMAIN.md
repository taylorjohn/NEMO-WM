# How to Add a New Domain to NeMo-WM
**For:** Developers extending CORTEX-PE or NeMo-WM to a new data source
**Prerequisites:** Working NeMo-WM environment, student_best.pt checkpoint

---

## Overview

NeMo-WM supports multi-domain training through domain-gated MoE routing. Each domain contributes batches during training, and the MoE router learns domain-specific expert preferences at inference. Adding a new domain requires five steps:

1. Write a domain loader
2. Choose or adapt an encoder
3. Decide which neuromodulator signals apply
4. Register the domain with the training script
5. Validate with AIM probe and AUROC

---

## Step 1 — Write a Domain Loader

All domain loaders follow the same interface used in `domain_loaders.py`:

```python
class MyDomainLoader:
    """
    Returns batches compatible with the NeMo-WM training loop.
    """
    def __init__(self, data_dir: str, batch_size: int = 16):
        self.data_dir = data_dir
        self.batch_size = batch_size

    def __iter__(self):
        """
        Yields dicts with at minimum:
            images:  (B, 3, 224, 224) float32 in [0,1]
            actions: (B, action_dim) float32

        Optional but recommended:
            gps:     (B, 2) float32 lat/lon — enables NE signal
            contact: (B,)   float32 in [0,1] — enables ACh signal
            labels:  (B,)   int — for supervised AUROC evaluation only
        """
        for batch in self._load_batches():
            yield {
                "images":  batch["frames"],     # (B, 3, 224, 224)
                "actions": batch["commands"],   # (B, action_dim)
                # optional:
                "gps":     batch.get("gps"),
                "contact": batch.get("contact"),
            }
```

**Data format requirements:**

| Field | Shape | Required | Notes |
|-------|-------|----------|-------|
| images | (B, 3, 224, 224) | Yes | RGB, float32, [0,1] |
| actions | (B, D) | Yes | Any dimension D |
| gps | (B, 2) | No | Enables NE spatial signal |
| contact | (B,) | No | Enables ACh contact signal |

**Existing loaders to reference:**
- `ReconLoader` — HDF5 with GPS and IMU (full signals)
- `MVTecLoader` — visual only, no physical metadata
- `SMAPLoader` — telemetry, no visual input

---

## Step 2 — Choose an Encoder

**Decision tree:**

```
Is your data visual (images/video)?
    YES -> Use existing StudentEncoder (student_best.pt)
           Fastest path. 0.34ms on NPU.

    NO -> Is it audio?
              YES -> Use MEL spectrogram -> treat as (1, H, W) image
                    Modify first Conv2d: Conv2d(1->16, ...)
                    See train_mvtec.py AudioStudentEncoder

          -> Is it telemetry/time-series?
              YES -> Use 1D encoder:
                    Conv1d(channels, 64) x3 + AdaptiveAvgPool1d(4)
                    Linear(256->128) + L2-norm
                    Same output shape as StudentEncoder

          -> Other modality?
              Build encoder that outputs (B, 128) unit-normalised.
              That is the only interface requirement.
```

**Encoder interface requirement:**
```python
z = encoder(x)  # x: any shape -> z: (B, 128), unit-normalised
assert z.shape == (B, 128)
assert torch.allclose(z.norm(dim=-1), torch.ones(B), atol=1e-3)
```

---

## Step 3 — Which Neuromodulator Signals Apply

| Signal | Requires | Skip if |
|--------|----------|---------|
| DA | Particle predictions | Never skip |
| 5HT | Particle history | Never skip |
| NE | GPS coordinates | No GPS available |
| ACh | Contact / event signal | No event metadata |
| E/I | DA + 5HT | Never skip |
| Ado | Step counter | Never skip |
| eCB | Actions + DA | No action magnitude |
| Cortisol | Rolling loss | Never skip |

**Minimal viable configuration (visual only, no metadata):**
- DA, 5HT, E/I, Ado, Cortisol active
- NE, ACh, eCB disabled (set weight to 0.0 in neuromodulator config)

**Full configuration (with GPS and contact):**
All eight signals active. Matches RECON domain.

In `train_cwm_multidomain.py`, set per-domain signal weights:

```python
DOMAIN_SIGNAL_CONFIG = {
    "recon":    {"ne": 1.0, "ach": 1.0, "ecb": 1.0},  # full
    "mvtec":    {"ne": 0.0, "ach": 0.0, "ecb": 0.5},  # visual only
    "my_domain":{"ne": 0.0, "ach": 1.0, "ecb": 0.5},  # contact but no GPS
}
```

---

## Step 4 — Register with Training Script

In `train_cwm_multidomain.py`, add your domain:

```python
# 1. Add loader
from my_domain_loader import MyDomainLoader

# 2. Add to domain registry
DOMAIN_LOADERS = {
    "recon":     lambda: ReconLoader(args.recon_dir),
    "mvtec":     lambda: MVTecLoader(args.mvtec_dir),
    "my_domain": lambda: MyDomainLoader(args.my_domain_dir),  # add this
}

# 3. Add CLI argument
parser.add_argument("--my-domain-dir", default=None)

# 4. Skip if not provided
if args.my_domain_dir:
    loaders["my_domain"] = DOMAIN_LOADERS["my_domain"]()
```

**Run smoke test first (2 epochs, 20 files):**
```powershell
python train_cwm_multidomain.py ^
    --cwm-ckpt   checkpoints\cwm\cwm_best.pt ^
    --recon-dir  recon_data\recon_release ^
    --my-domain-dir path\to\my\data ^
    --max-files  20 ^
    --epochs     2
```

Watch for:
- No NaN losses
- Router not collapsing to 100% one expert immediately
- DA firing on at least one domain

---

## Step 5 — Validate

### 5a. AIM probe — what does your domain encode?

```powershell
python probe_cwm_latents.py ^
    --cwm-ckpt  checkpoints\cwm\cwm_multidomain_best.pt ^
    --hdf5-dir  path\to\my\data ^
    --n-files   50 --n-per-file 8 ^
    --k-gap 4 --n-clusters 16
```

Adapt the probe signals to your domain. If your domain has frequency instead of velocity, add a `frequency` signal to `probe_cwm_latents.py` following the existing pattern.

**Minimum validation bar:**
- At least one domain-specific signal encodes at p < 0.05
- Null control stays p > 0.05

### 5b. AUROC k-sweep

```powershell
python eval_recon_auroc.py ^
    --cwm-ckpt  checkpoints\cwm\cwm_multidomain_best.pt ^
    --head-ckpt checkpoints\cwm\temporal_head_best.pt ^
    --hdf5-dir  path\to\my\data ^
    --k-sweep
```

**Expected AUROC ranges by domain type:**

| Domain type | k=1 AUROC target | Notes |
|-------------|-----------------|-------|
| Structured visual (navigation) | 0.90-0.99 | Spatial features rich |
| Industrial audio (MIMII) | 0.88-0.95 | Good with spectrogram |
| Bearing vibration (CWRU) | 0.95-1.00 | Strong periodic signal |
| Telemetry (SMAP) | 0.82-0.90 | Temporal dependencies |
| Medical audio (cardiac) | 0.85-0.92 | Rhythm-dependent |

---

## Common Failure Modes

**Router collapses immediately (100% one expert):**
- Increase aux loss weight: `--aux-alpha 0.05`
- Run 4-epoch recovery phase before reducing to 0.01
- Check batch size — too small gives unstable routing gradients

**DA=0.000 throughout on new domain:**
- Encoder output is too uniform — check normalisation
- Domain may need its own encoder rather than reusing StudentEncoder
- Run AIM probe — if all signals null, encoder is not learning

**AUROC near 0.50 (random):**
- Check if labels are balanced in test set
- Verify temporal_head_best.pt was trained on your domain (may need fine-tuning)
- Check that AIM probe shows at least some signal first

**L_jepa stays at 0.5000 (expected for low fps):**
- This is normal for slow visual change domains
- Neuromodulator carries the learning — check DA is non-zero
- If DA=0.000 AND L_jepa=0.5000, the encoder is not producing useful features

---

## Checklist

- [ ] Domain loader returns `(images, actions)` minimum
- [ ] Encoder outputs `(B, 128)` unit-normalised
- [ ] Smoke test passes (no NaN, 2 epochs)
- [ ] Router not collapsing at epoch 0
- [ ] AIM probe shows at least one domain-specific signal p < 0.05
- [ ] AUROC k=1 above 0.70
- [ ] Null control p > 0.05 in probe


---

## Domain Templates

### Template A — Visual Domain (images, no physical metadata)

Use this for: MVTec-style inspection, surveillance cameras, medical imaging, satellite imagery.

```python
# my_visual_loader.py
import glob, io
import numpy as np
import torch
import h5py
from PIL import Image


class VisualDomainLoader:
    """
    Minimal visual domain loader. No GPS, no contact.
    Active neuromodulator signals: DA, 5HT, E/I, Ado, Cortisol.
    """
    def __init__(self, data_dir: str, batch_size: int = 16,
                 image_size: int = 224, stride: int = 1):
        self.files = sorted(glob.glob(f"{data_dir}/**/*.png", recursive=True) +
                            glob.glob(f"{data_dir}/**/*.jpg", recursive=True))
        self.batch_size = batch_size
        self.image_size = image_size
        self.stride = stride

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB").resize(
            (self.image_size, self.image_size))
        return torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0

    def __iter__(self):
        buf = []
        for i, path in enumerate(self.files):
            if i % self.stride != 0:
                continue
            try:
                buf.append(self._load_image(path))
                if len(buf) == self.batch_size:
                    yield {
                        "images":  torch.stack(buf),        # (B, 3, 224, 224)
                        "actions": torch.zeros(len(buf), 2), # dummy action
                        # gps: omitted — NE signal disabled
                        # contact: omitted — ACh signal disabled
                    }
                    buf = []
            except Exception:
                pass
        if buf:
            yield {
                "images":  torch.stack(buf),
                "actions": torch.zeros(len(buf), 2),
            }
```

**Register in training script:**
```python
DOMAIN_SIGNAL_CONFIG["my_visual"] = {
    "ne": 0.0,   # no GPS
    "ach": 0.0,  # no contact
    "ecb": 0.3,  # mild habit suppression
}
```

**Expected probe signals:**
- Visual complexity (enc norm): may encode *
- Spatial structure: encodes ***
- Null control: must stay null

---

### Template B — Audio Domain (waveforms / spectrograms)

Use this for: MIMII industrial machines, cardiac audio, DCASE environmental audio, speech anomaly detection.

```python
# my_audio_loader.py
import glob
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T


class AudioDomainLoader:
    """
    Converts audio waveforms to MEL spectrograms treated as single-channel images.
    Requires AudioStudentEncoder (1-channel variant of StudentEncoder).
    Active neuromodulator signals: DA, 5HT, E/I, Ado, Cortisol.
    """
    def __init__(self, data_dir: str, batch_size: int = 16,
                 sample_rate: int = 16000, n_mels: int = 128,
                 segment_len: float = 1.0):
        self.files = sorted(glob.glob(f"{data_dir}/**/*.wav", recursive=True))
        self.batch_size = batch_size
        self.segment_samples = int(sample_rate * segment_len)
        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024, hop_length=512, n_mels=n_mels,
        )
        self.db = T.AmplitudeToDB()

    def _load_segment(self, path: str) -> torch.Tensor:
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)  # mono
        # Pad or trim to fixed length
        if waveform.shape[1] < self.segment_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.segment_samples - waveform.shape[1]))
        else:
            waveform = waveform[:, :self.segment_samples]
        # MEL spectrogram -> (1, n_mels, T) -> resize to (1, 224, 224)
        spec = self.db(self.mel(waveform))                # (1, n_mels, T)
        spec = torch.nn.functional.interpolate(
            spec.unsqueeze(0), size=(224, 224), mode="bilinear",
            align_corners=False).squeeze(0)               # (1, 224, 224)
        # Normalise to [0, 1]
        spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
        return spec

    def __iter__(self):
        buf = []
        for path in self.files:
            try:
                seg = self._load_segment(path)            # (1, 224, 224)
                # Repeat channel to RGB for StudentEncoder compatibility
                buf.append(seg.repeat(3, 1, 1))           # (3, 224, 224)
                if len(buf) == self.batch_size:
                    yield {
                        "images":  torch.stack(buf),      # (B, 3, 224, 224)
                        "actions": torch.zeros(len(buf), 1),
                    }
                    buf = []
            except Exception:
                pass
        if buf:
            yield {"images": torch.stack(buf),
                   "actions": torch.zeros(len(buf), 1)}
```

**Register:**
```python
DOMAIN_SIGNAL_CONFIG["my_audio"] = {
    "ne": 0.0,   # no GPS
    "ach": 0.5,  # use event-level anomaly score as contact proxy
    "ecb": 0.3,
}
```

**Acoustic contact proxy (for ACh):** If you have anomaly labels or event timestamps, use them as the contact signal:
```python
# In your loader, if you have event labels:
"contact": torch.tensor(is_anomaly_frame, dtype=torch.float32),
```

**Expected probe signals:**
- Frequency content (custom signal): encodes ***
- Temporal rhythm: may encode *
- Null control: must stay null

---

### Template C — Telemetry / Time-Series Domain

Use this for: SMAP/MSL satellite data, industrial sensor streams, IoT monitoring, EEG/ECG.

```python
# my_telemetry_loader.py
import glob
import numpy as np
import torch


class TelemetryDomainLoader:
    """
    Converts multivariate time-series to pseudo-images via sliding window.
    Uses a 1D convolutional encoder rather than StudentEncoder.
    Active neuromodulator signals: DA, 5HT, Cortisol (+ ACh if events available).
    """
    def __init__(self, data_dir: str, batch_size: int = 16,
                 window_size: int = 128, n_channels: int = 25,
                 stride: int = 32):
        self.files = sorted(glob.glob(f"{data_dir}/*.npy"))
        self.batch_size = batch_size
        self.window_size = window_size
        self.n_channels = n_channels
        self.stride = stride

    def _windows(self, data: np.ndarray):
        """Slide window across time dimension. data: (T, C)"""
        T = data.shape[0]
        for start in range(0, T - self.window_size, self.stride):
            yield data[start:start + self.window_size]  # (W, C)

    def __iter__(self):
        buf, buf_raw = [], []
        for path in self.files:
            try:
                data = np.load(path)  # (T, C)
                if data.shape[1] != self.n_channels:
                    data = data[:, :self.n_channels]
                for window in self._windows(data):
                    # Normalise window
                    w = (window - window.mean(0)) / (window.std(0) + 1e-8)
                    # Treat as (C, W) -> interpolate to (3, 224, 224)
                    t = torch.from_numpy(w).float().T.unsqueeze(0)  # (1, C, W)
                    img = torch.nn.functional.interpolate(
                        t.unsqueeze(0), size=(224, 224), mode="bilinear",
                        align_corners=False).squeeze()               # (224, 224)
                    img = img.unsqueeze(0).repeat(3, 1, 1)           # (3, 224, 224)
                    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                    buf.append(img)
                    buf_raw.append(torch.from_numpy(window).float())
                    if len(buf) == self.batch_size:
                        yield {
                            "images":  torch.stack(buf),
                            "actions": torch.zeros(len(buf), 1),
                            # No GPS for telemetry
                        }
                        buf, buf_raw = [], []
            except Exception:
                pass
```

**Register:**
```python
DOMAIN_SIGNAL_CONFIG["my_telemetry"] = {
    "ne": 0.0,   # no GPS
    "ach": 0.0,  # no contact (unless you have event labels)
    "ecb": 0.2,
}
```

**Expected probe signals:**
- Trend / drift: encodes ***
- Seasonality: may encode *
- Null control: must stay null

---

## Signal Selection — Q&A Decision Chart

Work through these questions in order. Stop at the first match.

---

**Q1: Do you have GPS or absolute position coordinates?**

```
YES → Enable NE signal (ne: 1.0)
      NE amplifies gradient when GPS prediction diverges.
      Strongly recommended for any geo-referenced data.
      Signals: DA ✓  5HT ✓  NE ✓  ACh ?  E/I ✓  Ado ✓  eCB ?  Cortisol ✓

NO  → Disable NE (ne: 0.0)
      No position signal available, NE would be uninformative.
      Continue to Q2.
```

---

**Q2: Do you have discrete events, contacts, or anomaly timestamps?**

```
YES → Enable ACh signal (ach: 0.5-1.0)
      ACh gates inter-particle interaction on contact/event frames.
      Examples: robot contact, machine fault onset, cardiac beat peak,
                valve opening, cell division, network packet arrival.
      Higher ach weight if events are rare and important.

NO  → Do you have a continuous signal that proxies contact?
          YES (e.g. force sensor, vibration amplitude) →
              Use it as a soft contact proxy, ach: 0.3-0.5
          NO → Disable ACh (ach: 0.0)
      Continue to Q3.
```

---

**Q3: Do you have action commands or control inputs?**

```
YES → Enable eCB signal (ecb: 0.5-1.0)
      eCB gates habit suppression: DA × ||action|| EMA.
      High eCB suppresses learning on familiar action patterns.
      Recommended whenever the agent has explicit actions.

NO  → Is there an implicit action proxy?
          YES (e.g. motor current, control voltage, flow rate) →
              Use as action magnitude, ecb: 0.3
          NO → Disable eCB (ecb: 0.0) or set very low (ecb: 0.1)
      Continue to Q4.
```

---

**Q4: Is your data likely to have distribution shift during training?**

```
YES → Cortisol always active (always enabled by default)
      Cortisol detects sustained loss elevation above rolling baseline.
      Especially important for: multi-session data, varying environments,
      seasonal data, data from multiple machines/subjects.
      Validated: r=0.768 lag-1 prediction of future loss (p<0.0001).

UNSURE → Keep Cortisol active. It adds negligible compute and
          the r=0.768 validation means it's informative even when
          you don't expect shift.
```

---

**Q5: Is your domain low frame-rate or slow-changing?**

```
YES (e.g. 4Hz outdoor nav, 1Hz telemetry, static inspection images) →
    L_jepa WILL be clamped at 0.5000 throughout training.
    This is expected and correct. The neuromodulator carries all learning.
    Do NOT increase free_bits threshold to try to "fix" this.
    Check: is DA > 0.000 after epoch 3? If yes, training is working.

NO (e.g. 30fps video, high-frequency vibration) →
    L_jepa may break the free_bits floor and contribute gradient.
    Both L_jepa and neuromodulator will train simultaneously.
    This is the ideal combined regime described in the paper.
```

---

**Q6: Do you have temporal structure (sequences matter)?**

```
YES (video, time-series, trajectories) →
    Use k-sweep AUROC: k=1,2,4,8,16.
    Watch for k=2 inversion (evidence of temporal encoding).
    AIM probe: add temporal_gap_k as a probe signal.

NO (independent images, i.i.d. samples) →
    Use k=1 AUROC only.
    AIM probe: skip temporal_gap_k signal.
    THICK GRU context still runs but may not contribute.
```

---

## Quick Reference — Signal Selection by Domain Type

| Domain | DA | 5HT | NE | ACh | E/I | Ado | eCB | Cortisol |
|--------|----|----|-----|-----|-----|-----|-----|---------|
| Outdoor robot nav (RECON) | ✓ | ✓ | ✓ GPS | ✓ Contact | ✓ | ✓ | ✓ | ✓ |
| Indoor manipulation | ✓ | ✓ | ✗ | ✓ Grasp | ✓ | ✓ | ✓ | ✓ |
| Industrial audio (MIMII) | ✓ | ✓ | ✗ | ✓ Fault | ✓ | ✓ | ✗ | ✓ |
| Bearing vibration (CWRU) | ✓ | ✓ | ✗ | ✓ Fault | ✓ | ✓ | ✗ | ✓ |
| Cardiac audio | ✓ | ✓ | ✗ | ✓ Beat | ✓ | ✓ | ✗ | ✓ |
| Satellite telemetry (SMAP) | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ |
| Visual inspection (MVTec) | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ |
| Autonomous driving | ✓ | ✓ | ✓ GPS | ✓ Collision | ✓ | ✓ | ✓ | ✓ |
| EEG/ECG medical | ✓ | ✓ | ✗ | ✓ Event | ✓ | ✓ | ✗ | ✓ |
| Generic i.i.d. images | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ |

**Minimum viable configuration (always safe):** DA + 5HT + E/I + Ado + Cortisol

