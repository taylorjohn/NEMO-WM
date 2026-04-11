"""
run_robot.py
~~~~~~~~~~~~~
Quickstart: run CortexBrain for robotic control.

Usage
-----
    # Headless simulation (no camera, no hardware):
    python run_robot.py --sim --joints 6 --ticks 200

    # With webcam:
    python run_robot.py --camera --joints 6 --hz 50

    # With AMD NPU encoder:
    python run_robot.py --camera --joints 6 --npu --hz 50
"""
import argparse, logging, time
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

parser = argparse.ArgumentParser(description="CortexBrain – Robotics")
parser.add_argument("--joints",  type=int,   default=6)
parser.add_argument("--hz",      type=float, default=50.0)
parser.add_argument("--ticks",   type=int,   default=None)
parser.add_argument("--camera",  action="store_true",
                    help="Use real webcam (requires opencv-python)")
parser.add_argument("--sim",     action="store_true",
                    help="Fully simulated – no camera or hardware required")
parser.add_argument("--npu",     action="store_true",
                    help="Enable AMD Ryzen AI NPU encoder")
parser.add_argument("--hud-ip",  default=None,
                    help="Mac IP for UDP telemetry HUD (e.g. 192.168.1.150)")
args = parser.parse_args()

from cortex_brain.engine import CortexEngine, EngineConfig, FeatureEncoder, Actuator
from cortex_brain.perception.lsm      import LSMConfig
from cortex_brain.perception.eb_jepa  import EBJEPAConfig
from cortex_brain.routing.static_csr  import CSRRouterConfig
from cortex_brain.memory.ttm_clustering import TTMConfig
from cortex_brain.hardware.amd_npu_binding import NPUConfig, AMDNPUBinding


# ── Camera / sensor encoder ───────────────────────────────────────────────────
class CameraEncoder(FeatureEncoder):
    """
    Captures a frame from webcam, runs NPU / CPU encoder → 256-D latent.
    Falls back to random noise in --sim mode.
    """
    def __init__(self, use_npu: bool, sim: bool):
        self._sim     = sim
        self._cap     = None
        self._binding = AMDNPUBinding(NPUConfig()) if use_npu else None
        self._rng     = np.random.default_rng(0)

        if not sim and args.camera:
            try:
                import cv2
                self._cap = cv2.VideoCapture(0)
                self._cv2 = cv2
                logging.info("Camera opened.")
            except ImportError:
                logging.warning("opencv-python not installed – using noise frames.")

    def encode(self, raw_obs) -> np.ndarray:
        frame = self._capture()
        if self._binding is not None:
            latent = self._binding.infer(frame)          # (128,)
        else:
            import torch, torch.nn as nn
            # tiny CNN projection for CPU path
            if not hasattr(self, "_cpu_enc"):
                self._cpu_enc = nn.Sequential(
                    nn.Flatten(), nn.Linear(3*224*224, 256), nn.Tanh())
                self._cpu_enc.eval()
            with torch.no_grad():
                latent = self._cpu_enc(
                    torch.from_numpy(frame)).numpy().flatten()[:256]
        return latent.astype(np.float32)

    def _capture(self) -> np.ndarray:
        if self._cap is not None and self._cap.isOpened():
            ret, bgr = self._cap.read()
            if ret:
                rgb = self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)
                rgb = self._cv2.resize(rgb, (224, 224))
                arr = rgb.astype(np.float32) / 127.5 - 1.0
                return arr.transpose(2, 0, 1)[np.newaxis]
        return self._rng.standard_normal((1, 3, 224, 224)).astype(np.float32)


# ── Joint actuator ────────────────────────────────────────────────────────────
class JointActuator(Actuator):
    """
    Converts planned action → joint velocity commands.
    Override _send() to talk to your robot SDK (ROS, dynamixel, etc.).
    """
    def __init__(self, num_joints: int, max_velocity: float = 1.0):
        self.num_joints   = num_joints
        self.max_velocity = max_velocity

    def act(self, action: np.ndarray, resonance: float, metadata: dict) -> float:
        padded   = np.zeros(self.num_joints, dtype=np.float32)
        padded[:len(action)] = action[:self.num_joints]
        commands = np.clip(padded, -1.0, 1.0) * self.max_velocity
        self._send(commands, resonance)
        return 0.0

    def _send(self, joint_velocities: np.ndarray, resonance: float):
        """
        Replace this with your hardware SDK call.
        Examples:
            ROS:        pub.publish(Float64MultiArray(data=joint_velocities))
            Dynamixel:  dxl.set_goal_velocity(joint_velocities)
        """
        logging.debug("joints=%s  ρ=%.4f",
                      np.round(joint_velocities, 3).tolist(), resonance)


# ── Build & run ───────────────────────────────────────────────────────────────
cfg = EngineConfig(
    input_dim=256, latent_dim=128, action_dim=args.joints,
    lsm    = LSMConfig(input_dim=256, reservoir_dim=512, output_dim=128,
                       spectral_radius=0.9, leak_rate=0.2),
    jepa   = EBJEPAConfig(latent_dim=128, compressed_dim=16, aux_dim=3,
                          num_candidates=32, planning_horizon=3,
                          gamma_horizon=1.0),
    router = CSRRouterConfig(input_dim=256, manifold_dim=128, num_experts=4),
    ttm    = TTMConfig(manifold_dim=128, max_episodic=500, max_long_term=1000),
    npu    = NPUConfig(),
    use_npu      = args.npu,
    telemetry_ip = args.hud_ip,
)

engine = CortexEngine(
    config          = cfg,
    feature_encoder = CameraEncoder(use_npu=args.npu, sim=args.sim),
    actuator        = JointActuator(num_joints=args.joints),
    goal_latent     = np.zeros(128, dtype=np.float32),  # set a real goal here
)

print(f"\n🤖 CortexBrain – Robotics")
print(f"   joints={args.joints}  hz={args.hz}  camera={args.camera}  npu={args.npu}")
print(f"   ticks={'∞' if args.ticks is None else args.ticks}")
print(f"   Press Ctrl+C to stop\n")

engine.run(hz=args.hz, max_ticks=args.ticks)