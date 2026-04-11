# amd-npu-vlm-compat

Fix popular vision-language models to run on AMD Ryzen AI MAX+ (Strix Halo) NPU.

Most VLMs fail on AMD NPU not because of the vision encoder but because of
graph fragmentation in the connector layers. This repo isolates and fixes each
component so the vision side runs on NPU even when the LLM stays on CPU.

---

## Compatibility table

| Model | Vision Enc | Connector | LLM | Fix Available |
|---|---|---|---|---|
| LLaVA-1.5 | CLIP ViT-L | MLP 2-layer | Vicuna/Llama | yes -- fix_llava/ |
| InstructBLIP | ViT-G | Q-Former | Vicuna | yes -- fix_instructblip/ |
| PaliGemma | SigLIP | linear | Gemma | yes -- fix_siglip/ |
| Qwen-VL | ViT custom | resampler | Qwen | yes -- fix_qwenvl/ |
| Qwen2-VL | ViT custom | MRope | Qwen2 | partial -- fixed res only |
| V-JEPA 2-L | ViT-L | -- | -- | yes -- fix_vjepa2/ (frame wrapper) |
| V-JEPA 2-G | ViT-G | -- | -- | partial -- OOM on calibration |
| CogVLM | ViT-E | expert router | Vicuna | no -- scatter/gather not in XDNA2 |
| MiniGPT-4 | ViT-G | Q-Former | Vicuna | no -- same as CogVLM |

Status: NPU = vision encoder runs on NPU with XINT8. LLM always stays on CPU.

---

## The pattern

Every VLM has three parts:

    Image -> [Vision Encoder] -> [Connector] -> [LLM backbone] -> Text

Vision encoders are ViTs -- they work great on NPU with XINT8.
LLM backbones always run on CPU -- autoregressive decoding does not benefit from NPU.
Connectors are the problem -- small layers that fragment the NPU graph.

The fix in every case is the same:
1. Export vision encoder separately to ONNX
2. XINT8 quantize it
3. Run it on NPU
4. Pass the output to the connector + LLM running normally on CPU

---

## Quickstart

    conda activate ryzen-ai-1.7.0
    pip install -r requirements.txt

    # Fix LLaVA (easiest, 2 hrs)
    python fix_llava/run.py --model llava-1.5-7b --image photo.jpg --prompt "describe this"

    # Fix SigLIP / PaliGemma
    python fix_siglip/run.py --model google/paligemma-3b-pt-224

    # Fix V-JEPA 2 (frame wrapper)
    python fix_vjepa2/run.py --model facebook/vjepa2-vitl-fpc64-256 --video clip.mp4

    # Benchmark all fixed models
    python test_all.py --benchmark

---

## Why this matters

V-JEPA 2-L on AMD NPU (broken):   1849ms per 8-frame clip
V-JEPA 2-L on AMD NPU (fixed):    ~230ms per frame (frame wrapper, NPU per frame)

LLaVA vision encoder (broken):    375ms (CLIP-L/14 on CPU)
LLaVA vision encoder (fixed):     ~50ms (CLIP-L/14 XINT8 on NPU)

LLM backbone: always CPU, not changed.

---

## Key insight

XINT8 is the only quantization mode where LayerNorm runs on AMD NPU.
BF16/A8W8/A16W8 all trigger CPU fallbacks for LayerNorm.
Every ViT has 2 LayerNorms per block.

See also: github.com/taylorjohn/strix-halo-vision-npu

---

## Hardware tested

GMKtec EVO-X2, Ryzen AI MAX+ 395, 128GB unified RAM, XDNA2 NPU
No discrete GPU.

---

MIT License

## Measured CPU baselines (GMKtec EVO-X2, Ryzen AI MAX+ 395)

Model               CPU PyTorch    ONNX CPU    NPU XINT8    Speedup
DINOv2-S/14         41.5ms         --          0.85ms       49x  (measured)
CLIP ViT-B/32       47.9ms         13.6ms      ~1.0ms       ~50x (estimated)
CLIP ViT-L/14       341.8ms        164.5ms     ~5.7ms       ~60x (estimated)
V-JEPA 2-L          1849ms         --          ~400ms       ~5x  (wrapper)

Measured April 11 2026.

## Measured CPU baselines (GMKtec EVO-X2, Ryzen AI MAX+ 395)

Model               CPU PyTorch    ONNX CPU    NPU XINT8    Speedup
DINOv2-S/14 (21M)   41.5ms         --          0.85ms       49x measured
CLIP ViT-B/32       47.9ms         13.6ms      ~1.0ms       ~50x estimated
CLIP ViT-L/14       341.8ms        164.5ms     ~5.7ms       ~60x estimated
V-JEPA 2-L          1849ms         --          ~400ms       ~5x wrapper

Measured April 11 2026 on GMKtec EVO-X2.
ONNX/CPU = PyTorch FP32 exported to ONNX, no quantization.
NPU XINT8 = AMD Quark quantization + VitisAI Execution Provider.
