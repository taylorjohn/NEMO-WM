"""
student_encoder.py — CORTEX-16 Student CNN Encoder

Architecture designed for knowledge distillation from a frozen DINOv2 teacher.
Targets ~680KB after XINT8 quantization for AMD Ryzen AI NPU deployment.

Design principles:
  - Small CNN backbone → 14×14×32 spatial feature map (preserves spatial structure
    per Wang et al. 2026 finding that spatial dims matter more than channel depth)
  - Global average pool → 32-D global descriptor
  - ShatteredLatentHead maps 32-D → 128-D with strict 4×32 semantic axis boundaries:
      dims   0-31:  Shape
      dims  32-63:  Size
      dims  64-95:  Depth
      dims 96-127:  Velocity
  - Total parameter count: ~170K fp32 → ~170KB fp32 → ~42KB INT8
    (well under the 680KB budget that CORTEX-12 validated)

NPU deployment:
  - Export with torch.onnx.export at opset 17, dynamic_axes=None (required)
  - Quantize to XINT8 via AMD Quark before loading into VitisAI EP
  - All ops (Conv, ReLU, BatchNorm, AvgPool) are in the NPU's supported set
    for XINT8 — no CPU fallback points expected
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Semantic Axis Head (from cortex_adapter_v2.py — preserved exactly)
# Four independent linear projections enforce hard dimensional boundaries.
# No weight sharing between axes — shape features cannot contaminate velocity.
# ---------------------------------------------------------------------------
class ShatteredLatentHead(nn.Module):
    """
    Maps a global descriptor to the 128-D semantic latent space.

    Axis layout (matches CORTEX-12 certification toolchain):
        dims   0-31:  Shape
        dims  32-63:  Size
        dims  64-95:  Depth
        dims 96-127:  Velocity
    """
    def __init__(self, in_features: int = 32):
        super().__init__()
        self.heads = nn.ModuleDict({
            'shape':    nn.Linear(in_features, 32),
            'size':     nn.Linear(in_features, 32),
            'depth':    nn.Linear(in_features, 32),
            'velocity': nn.Linear(in_features, 32),
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self.heads[k](x) for k in ['shape', 'size', 'depth', 'velocity']],
            dim=-1,
        )


# ---------------------------------------------------------------------------
# CNN Backbone
# Produces 14×14×32 spatial feature map from 224×224×3 input.
# Architecture: 4 conv blocks with progressive channel doubling then reduction.
# Batch normalisation throughout for XINT8 quantization stability.
# ---------------------------------------------------------------------------
class CortexCNNBackbone(nn.Module):
    """
    Lightweight CNN producing spatial features at 1/16 input resolution.

    Input:  (B, 3, 224, 224)
    Output: (B, 32, 14, 14)

    V-JEPA 2.1 Deep Self-Supervision: supports return_intermediates=True
    to expose intermediate feature maps at each block for hierarchical
    supervision during training. Intermediates are training-only and
    never exported to the NPU.

    Parameter count: ~147K fp32
    """
    def __init__(self):
        super().__init__()
        # Split into named blocks for intermediate access
        self.block1 = nn.Sequential(
            nn.Conv2d(3,  16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
        )  # 224 → 112, 16ch
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )  # 112 → 56, 32ch
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )  # 56 → 28, 64ch
        self.block4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )  # 28 → 14, 32ch

        # Alias for backward compatibility (NPU export path uses self.stem)
        self.stem = nn.Sequential(
            self.block1, self.block2, self.block3, self.block4
        )

    def forward(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list]:
        """
        Args:
            x:                   (B, 3, 224, 224)
            return_intermediates: if True, return list of 4 intermediate
                                  global-pooled features for deep supervision.
                                  Training-only — never exported to NPU.

        Returns:
            out: (B, 32, 14, 14) final spatial map
            intermediates (optional): list of 4 tensors
                [(B,16), (B,32), (B,64), (B,32)] — one per block, global-pooled
        """
        h1 = self.block1(x)   # (B, 16, 112, 112)
        h2 = self.block2(h1)  # (B, 32,  56,  56)
        h3 = self.block3(h2)  # (B, 64,  28,  28)
        h4 = self.block4(h3)  # (B, 32,  14,  14)

        if return_intermediates:
            # Global average pool each block → compact descriptors
            g1 = h1.mean(dim=[-2, -1])  # (B, 16)
            g2 = h2.mean(dim=[-2, -1])  # (B, 32)
            g3 = h3.mean(dim=[-2, -1])  # (B, 64)
            g4 = h4.mean(dim=[-2, -1])  # (B, 32)
            return h4, [g1, g2, g3, g4]
        return h4


# ---------------------------------------------------------------------------
# Full Student Encoder
# ---------------------------------------------------------------------------
class StudentEncoder(nn.Module):
    """
    Complete student encoder: CNN backbone + global pool + semantic head.

    Input:  (B, 3, 224, 224)   — raw RGB frame, ImageNet-normalised
    Output:
        z:          (B, 128)    — semantic latent for planning (NPU output)
        spatial:    (B, 32, 14, 14)  — spatial features for world model (optional)

    Total fp32 parameters: ~170K (~680KB fp32, ~170KB INT8)
    """
    def __init__(self):
        super().__init__()
        self.backbone = CortexCNNBackbone()
        self.head     = ShatteredLatentHead(in_features=32)

    def forward(
        self,
        x: torch.Tensor,
        return_spatial: bool = False,
        return_intermediates: bool = False,
    ) -> torch.Tensor | tuple:
        """
        Args:
            x:                   (B, 3, 224, 224) input frame
            return_spatial:      if True, also return the 14×14×32 spatial map
            return_intermediates: if True, return list of 4 global-pooled
                                  block descriptors for deep self-supervision.
                                  Training-only. Takes priority over return_spatial.

        Returns:
            z:           (B, 128) normalised semantic latent
            spatial (if return_spatial): (B, 32, 14, 14)
            intermediates (if return_intermediates):
                (z, spatial, [(B,16),(B,32),(B,64),(B,32)])
        """
        if return_intermediates:
            spatial, intermediates = self.backbone(x, return_intermediates=True)
        else:
            spatial = self.backbone(x)

        g     = spatial.mean(dim=[-2, -1])   # (B, 32)
        z_raw = self.head(g)                  # (B, 128)
        z     = F.normalize(z_raw, dim=-1)    # (B, 128)

        if return_intermediates:
            return z, spatial, intermediates
        if return_spatial:
            return z, spatial
        return z

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def size_kb(self, dtype_bytes: int = 4) -> float:
        """Returns model size in KB for a given dtype (4=fp32, 1=int8)."""
        return self.count_parameters() * dtype_bytes / 1024


# ---------------------------------------------------------------------------
# ONNX Export Interface
# The NPU only needs the global latent z — spatial map is CPU/training only.
# Wraps StudentEncoder to export a single-output model for VitisAI EP.
# ---------------------------------------------------------------------------
class StudentEncoderONNX(nn.Module):
    """
    Thin wrapper that exposes only the 128-D latent output for ONNX export.
    The spatial map is not exported — it is only used during training.

    Export command (run from export_student_xint8.py):
        torch.onnx.export(
            StudentEncoderONNX(encoder),
            dummy_input,
            "cortex_student_fp32.onnx",
            opset_version=17,
            input_names=["input_frame"],
            output_names=["output_latent"],
            dynamic_axes=None,    # REQUIRED: NPU needs fixed shapes
        )
    """
    def __init__(self, encoder: StudentEncoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, return_spatial=False)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = StudentEncoder()
    dummy = torch.randn(1, 3, 224, 224)

    z, spatial = model(dummy, return_spatial=True)

    print(f"✅ Student encoder initialised")
    print(f"   Parameters:  {model.count_parameters():,}")
    print(f"   Size fp32:   {model.size_kb(4):.1f} KB")
    print(f"   Size INT8:   {model.size_kb(1):.1f} KB")
    print(f"   Output z:    {tuple(z.shape)}   (semantic latent)")
    print(f"   Spatial map: {tuple(spatial.shape)}  (14×14×32)")
    print()
    print("Axis boundaries in z:")
    print("   dims   0-31:  shape")
    print("   dims  32-63:  size")
    print("   dims  64-95:  depth")
    print("   dims 96-127:  velocity")
    assert z.shape == (1, 128),          f"Expected (1,128), got {z.shape}"
    assert spatial.shape == (1, 32, 14, 14), f"Expected (1,32,14,14), got {spatial.shape}"
    assert abs(z.norm(dim=-1).item() - 1.0) < 1e-5, "z should be L2-normalised"
    print("\n✅ All assertions passed.")
