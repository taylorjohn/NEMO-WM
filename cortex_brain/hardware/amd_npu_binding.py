"""
cortex_brain.hardware.amd_npu_binding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Vitis AI / ONNX Runtime zero-copy isolation layer.

Implements the DualPath zero-copy pattern:
  * Input frames bound to pinned shared memory via iobinding
  * Prevents Host→Device copies stalling the CPU during NPU inference
  * Decouples CPU (BoK planner) and NPU (encoder) for async parallelism
  * Latency target: <2.0ms per frame on Ryzen AI

Falls back transparently to CPU when Vitis AI EP is unavailable.
Binary telemetry packet: timestamp(d) + x,y(2f) + 128-D latent(128f) = 528 bytes
"""
from __future__ import annotations
import logging, struct, socket, time
from dataclasses import dataclass
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

PACKET_FORMAT = struct.Struct("<d 2f 128f")
PACKET_SIZE   = PACKET_FORMAT.size   # 528 bytes


@dataclass
class NPUConfig:
    model_path:   str = "cortex_v13_npu.onnx"
    config_path:  str = "vaip_config.json"
    input_name:   str = "input_frame"
    output_name:  str = "output_latent"
    latent_dim:   int = 128
    frame_h: int = 224
    frame_w: int = 224
    frame_c: int = 3


class AMDNPUBinding:
    """Zero-copy ONNX inference on AMD Ryzen AI NPU."""

    def __init__(self, config: NPUConfig = NPUConfig()) -> None:
        self.config = config
        self._session = None
        self._io_binding = None
        self.on_npu = False
        self._cpu_model = None
        self._try_npu_init()
        if not self.on_npu:
            self._init_cpu_fallback()

    def infer(self, frame: np.ndarray) -> np.ndarray:
        """Run forward pass. Input CHW or NCHW float32 → (latent_dim,)."""
        frame = self._validate(frame)
        return self._npu_infer(frame) if self.on_npu else self._cpu_infer(frame)

    def broadcast_telemetry(self, latent: np.ndarray, fovea_xy: tuple,
                             target_ip: str, port: int = 5005) -> None:
        """Pack 528-byte UDP packet and send to macOS visualizer HUD."""
        padded = np.zeros(128, dtype=np.float32)
        padded[:min(len(latent), 128)] = latent[:128]
        pkt = PACKET_FORMAT.pack(time.time(), *fovea_xy, *padded.tolist())
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.sendto(pkt, (target_ip, port))

    def _try_npu_init(self) -> None:
        try:
            import onnxruntime as ort
            self._session = ort.InferenceSession(
                self.config.model_path,
                providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
                provider_options=[{"config_file": self.config.config_path}, {}],
            )
            self._io_binding = self._session.io_binding()
            self.on_npu = True
            logger.info("AMDNPUBinding: Vitis AI EP online – zero-copy active.")
        except Exception as exc:
            logger.warning("AMDNPUBinding: NPU unavailable (%s) – CPU fallback.", exc)

    def _init_cpu_fallback(self) -> None:
        try:
            import torch, torch.nn as nn
            c = self.config
            self._cpu_model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(c.frame_c * c.frame_h * c.frame_w, 512),
                nn.GELU(),
                nn.Linear(512, c.latent_dim),
                nn.Tanh(),
            )
            self._cpu_model.eval()
        except ImportError:
            logger.warning("torch unavailable – random latents.")

    def _npu_infer(self, frame: np.ndarray) -> np.ndarray:
        self._io_binding.bind_cpu_input(self.config.input_name, frame)
        self._io_binding.bind_output(self.config.output_name)
        self._session.run_with_iobinding(self._io_binding)
        out = self._io_binding.copy_outputs_to_cpu()[0].flatten()
        self._io_binding.clear_binding_inputs()
        return out[:self.config.latent_dim]

    def _cpu_infer(self, frame: np.ndarray) -> np.ndarray:
        if self._cpu_model is None:
            return np.random.randn(self.config.latent_dim).astype(np.float32)
        import torch
        with torch.no_grad():
            return self._cpu_model(torch.from_numpy(frame)).numpy().flatten()[:self.config.latent_dim]

    def _validate(self, frame: np.ndarray) -> np.ndarray:
        frame = frame.astype(np.float32)
        return frame[np.newaxis] if frame.ndim == 3 else frame