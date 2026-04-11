"""
export_onnx.py
~~~~~~~~~~~~~~
Export the CJEPAPredictor encoder stage to ONNX for AMD Ryzen AI NPU.

What gets exported
------------------
The visual encoder path:  (1, 3, 224, 224) image → (128,) latent vector
This is the CPU→NPU offload path used by AMDNPUBinding.infer().

The full CJEPAPredictor (latent→latent prediction) stays on CPU/GPU
because it depends on dynamic K-candidate sampling at runtime.

Output files
------------
cortex_v13_npu.onnx      — NPU encoder model (loaded by amd_npu_binding.py)

Usage
-----
    python export_onnx.py                      # export with random weights
    python export_onnx.py --weights my.pt      # load saved weights first
    python export_onnx.py --verify             # run a quick ONNX Runtime check
    python export_onnx.py --opset 17           # change ONNX opset (default 17)

Vitis AI next steps
-------------------
After export, quantise for the NPU:
    vai_q_onnx quantize_static     \\
        --input_model  cortex_v13_npu.onnx \\
        --output_model cortex_v13_npu_quantised.onnx \\
        --calibration_data_reader <your_reader>

Then point NPUConfig.model_path at the quantised file and add --npu to your run command.
"""
from __future__ import annotations
import argparse, logging, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lightweight CNN encoder  (same architecture as AMDNPUBinding CPU fallback)
# ---------------------------------------------------------------------------

class NPUEncoder(nn.Module):
    """
    Minimal ResNet-style encoder: (B, 3, 224, 224) → (B, 128)

    This is intentionally small (~680 KB) so it fits in the Ryzen AI
    SRAM budget without spilling to DDR.

    Layer budget
    ------------
    conv1  :  3→16  k=7 s=2  →  112×112
    conv2  : 16→32  k=3 s=2  →   56×56
    conv3  : 32→64  k=3 s=2  →   28×28
    conv4  : 64→128 k=3 s=2  →   14×14
    gap    : global average pool → (B, 128)
    """

    def __init__(self, out_dim: int = 128) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(3,  16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            # Block 4
            nn.Conv2d(64, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True),
            # Global average pool
            nn.AdaptiveAvgPool2d(1),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 224, 224) → (B, out_dim)"""
        return self.backbone(x).flatten(1)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export(
    weights_path: str | None,
    output_path: str,
    opset: int,
    verify: bool,
) -> None:
    model = NPUEncoder(out_dim=128)
    model.eval()

    if weights_path:
        p = Path(weights_path)
        if p.exists():
            state = torch.load(p, map_location="cpu")
            # support both raw state_dict and checkpoint dicts
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            elif "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)
            log.info("Loaded weights from %s", p)
        else:
            log.warning("Weights file not found: %s — exporting random weights.", p)
    else:
        log.info("No weights file specified — exporting random initialisation.")

    dummy = torch.randn(1, 3, 224, 224)

    # Warm-up
    with torch.no_grad():
        out = model(dummy)
    log.info("Forward pass OK: input %s → output %s", tuple(dummy.shape), tuple(out.shape))

    # Export
    out_p = Path(output_path)
    torch.onnx.export(
        model,
        dummy,
        str(out_p),
        export_params       = True,
        opset_version       = opset,
        do_constant_folding = True,
        input_names         = ["image_nchw"],
        output_names        = ["latent"],
        dynamic_axes        = {
            "image_nchw": {0: "batch"},
            "latent":     {0: "batch"},
        },
    )
    size_kb = out_p.stat().st_size / 1024
    log.info("Exported → %s  (%.1f KB)", out_p, size_kb)

    # Optional ONNX Runtime verify
    if verify:
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(out_p),
                                        providers=["CPUExecutionProvider"])
            feeds = {"image_nchw": dummy.numpy()}
            result = sess.run(["latent"], feeds)[0]
            assert result.shape == (1, 128), f"Unexpected shape: {result.shape}"

            # Compare to torch output
            with torch.no_grad():
                torch_out = model(dummy).numpy()
            max_diff = float(np.abs(result - torch_out).max())
            log.info("ORT verify OK — max diff vs torch: %.2e", max_diff)
            assert max_diff < 1e-4, "ORT / torch mismatch too large"
            log.info("✅ Export verified — ready for Vitis AI quantisation.")
        except ImportError:
            log.warning("onnxruntime not installed — skipping verify (pip install onnxruntime)")
        except Exception as exc:
            log.error("Verify FAILED: %s", exc)
            sys.exit(1)
    else:
        log.info("✅ Export complete. Run with --verify to check ONNX Runtime.")

    print(f"\n{'─'*55}")
    print(f"  Model :  {out_p}  ({size_kb:.0f} KB)")
    print(f"  Input :  (B, 3, 224, 224)  float32  NCHW")
    print(f"  Output:  (B, 128)           float32")
    print(f"  Opset :  {opset}")
    print(f"{'─'*55}")
    print(f"\nNext step (Vitis AI quantise):")
    print(f"  vai_q_onnx quantize_static \\")
    print(f"      --input_model  {out_p} \\")
    print(f"      --output_model cortex_v13_npu_quant.onnx \\")
    print(f"      --calibration_data_reader <your_reader>")
    print(f"\nThen run with:  python run_trading.py --sim --npu")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export CortexBrain NPU encoder to ONNX")
    parser.add_argument("--weights", default=None,
                        help="Path to .pt weights file (optional)")
    parser.add_argument("--output",  default="cortex_v13_npu.onnx",
                        help="Output ONNX file (default: cortex_v13_npu.onnx)")
    parser.add_argument("--opset",   type=int, default=17,
                        help="ONNX opset version (default: 17)")
    parser.add_argument("--verify",  action="store_true",
                        help="Run ONNX Runtime inference check after export")
    args = parser.parse_args()

    export(
        weights_path = args.weights,
        output_path  = args.output,
        opset        = args.opset,
        verify       = args.verify,
    )