"""
unified_cortex_loop.py — CORTEX-16 Unified Execution Engine

Encoder inference:   AMD Radeon iGPU via DirectML (0.12ms avg)
Planning:            Mirror Ascent MPC (K=64, R=3, gamma=2.0)
Modulation:          Allen Brain Observatory Neuropixels spike rate rho
Telemetry:           UDP broadcast port 5005 to macOS visualiser HUD

Biological modulation:
    rho in [0, 1] from neuropixels_reader.py (session 715093703, V1 mouse)
    High rho = elevated arousal -> flatten cost landscape -> exploration
    Low rho  = calm             -> sharpen cost landscape -> exploitation
    rho = sigmoid( z-score of 33ms spike count, rolling 300-sample history )

Supports three predictor types (auto-detected from checkpoint):
    Option 1 — GeometricHorizonPredictor (128-D global, gamma-conditioned)
    Option 2 — CausalTransformerPredictor (128-D global, K=3 history)
    Option 3 — SpatialMLPPredictor (1568-D spatial 14x14x8)
    Option 4 — SpatialTransformerPredictor (196 patches x 8ch, K=3)

Usage:
    python unified_cortex_loop.py --mock
    python unified_cortex_loop.py --symbol SPY --hz 1
    python unified_cortex_loop.py --symbol SPY --hz 1 --predictor ./predictors/predictor_wall_opt1.pt
"""

import argparse
import socket
import struct
import time
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort
from torchvision import transforms

from cortex_wm.latent_predictor import GeometricHorizonPredictor
from neuropixels_reader import get_neuropixels_rho


# =============================================================================
# Goal State Manager
# =============================================================================
class GoalStateManager:
    """
    Loads bull/bear goal latents from compute_goal_states.py output.
    Updates z_goal each tick based on recent 5-bar price momentum.
    Falls back to zeros if goal files not found.
    """
    def __init__(self, goals_dir: str = "./goals"):
        self.z_bull = None
        self.z_bear = None
        self.z_zero = np.zeros(128, dtype=np.float32)
        self._load(goals_dir)

    def _load(self, goals_dir: str) -> None:
        bull_path = Path(goals_dir) / "z_goal_bull.npy"
        bear_path = Path(goals_dir) / "z_goal_bear.npy"
        if bull_path.exists() and bear_path.exists():
            self.z_bull = np.load(bull_path).astype(np.float32)
            self.z_bear = np.load(bear_path).astype(np.float32)
            stats_path  = Path(goals_dir) / "goal_stats.json"
            cosine = "?"
            if stats_path.exists():
                import json as _json
                stats  = _json.load(open(stats_path))
                cosine = f"{stats.get('cosine_sim', 0):.4f}"
            print(f"✅ Goal states loaded | cosine sim bull/bear: {cosine}")
        else:
            print(f"⚠️  Goal states not found in {goals_dir} — using zeros")
            print(f"   Run: python compute_goal_states.py --days 25")

    @property
    def available(self) -> bool:
        return self.z_bull is not None

    def get_goal(self, bars_df=None) -> tuple:
        """Returns (z_goal, regime_str) based on 5-bar momentum."""
        if not self.available or bars_df is None or len(bars_df) < 5:
            return self.z_zero, "neutral"
        closes   = bars_df["close"].values
        momentum = (closes[-1] - closes[-5]) / closes[-5]
        if momentum > 0.0005:
            return self.z_bull, f"bull {momentum*100:+.3f}%"
        elif momentum < -0.0005:
            return self.z_bear, f"bear {momentum*100:+.3f}%"
        else:
            return self.z_zero, f"flat {momentum*100:+.3f}%"


# =============================================================================
# Transforms — must match Phase 2 training exactly
# =============================================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

FRAME_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# =============================================================================
# Inference Engine — DirectML primary, VitisAI NPU secondary, CPU fallback
# =============================================================================
class InferenceEngine:
    """
    ONNX inference engine. Provider priority:
        1. DmlExecutionProvider    (AMD Radeon iGPU, 0.12ms)
        2. VitisAIExecutionProvider (AMD Ryzen AI NPU, <2ms, requires conda)
        3. CPUExecutionProvider    (fallback)

    Input:  (1, 3, 224, 224) fp32 RGB frame, ImageNet-normalised
    Output: (1, 128) fp32 semantic latent
    """

    # Default vaip config path — override with VAIP_CONFIG env var
    VAIP_CONFIG = (
        r"C:\Users\MeteorAI\Desktop\cortex-12v15\vaip_config.json"
    )

    def __init__(self, model_path: str):
        import os
        cache_dir   = str(Path(model_path).parent)
        vaip_config = os.environ.get("VAIP_CONFIG", self.VAIP_CONFIG)

        # VitisAI NPU — try first if xint8 model
        if "xint8" in model_path.lower() or "npu" in model_path.lower():
            try:
                vai_opts = {
                    "cache_dir":   cache_dir,
                    "cache_key":   "cortex16",
                    "log_level":   "warning",
                }
                self.session = ort.InferenceSession(
                    model_path,
                    providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
                    provider_options=[vai_opts, {}],
                )
                if "VitisAI" in self.session.get_providers()[0]:
                    self.provider = "VitisAI (AMD Ryzen AI NPU)"
                    print(f"⚡ Encoder: {self.provider}")
                    return
            except Exception as e:
                print(f"   VitisAI init failed: {e}")

        # DirectML (AMD Radeon iGPU)
        try:
            self.session = ort.InferenceSession(
                model_path,
                providers=["DmlExecutionProvider", "CPUExecutionProvider"],
            )
            if "Dml" in self.session.get_providers()[0]:
                self.provider = "DirectML (AMD Radeon iGPU)"
                print(f"⚡ Encoder: {self.provider}")
                return
        except Exception:
            pass

        # CPU fallback
        self.session  = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.provider = "CPU (fallback)"
        print(f"⚠️  Encoder: {self.provider}")

    def encode(self, frame_np: np.ndarray) -> np.ndarray:
        """Encode (H, W, 3) uint8 RGB frame to (128,) float32 latent."""
        t   = FRAME_TRANSFORM(frame_np).unsqueeze(0).numpy()
        out = self.session.run(None, {"input_frame": t})
        return out[0].squeeze(0)


# =============================================================================
# Predictor loader — auto-detects option from checkpoint metadata
# =============================================================================
def load_predictor_from_checkpoint(path: str):
    """
    Load predictor and optional spatial projector from checkpoint.
    Auto-detects option from ckpt['option']. Falls back to option 1.
    Returns (predictor, spatial_proj_or_None).
    """
    ckpt = torch.load(path, map_location="cpu")
    opt  = ckpt.get("option", 1)

    if opt == 4:
        from train_predictor import (
            SpatialTransformerPredictor, SpatialChannelProjector,
            N_PATCHES, DV,
        )
        config = ckpt.get("config", {"action_dim": 2})
        pred   = SpatialTransformerPredictor(
            n_patches=N_PATCHES, dv=DV,
            action_dim=config["action_dim"], history_len=3,
            d_model=64, n_heads=4, n_layers=4,
        )
        pred.load_state_dict(ckpt["state_dict"])
        pred.eval()
        sp = SpatialChannelProjector(32, DV)
        sp.load_state_dict(ckpt["spatial_proj_state_dict"])
        sp.eval()
        print(f"✅ Predictor opt4 (SpatialTransformer 196×8, K=3) loaded: {path}")
        return pred, sp

    elif opt == 3:
        from train_predictor import SpatialMLPPredictor, SpatialChannelProjector, SPATIAL_DIM, DV
        pred = SpatialMLPPredictor(SPATIAL_DIM, 2)
        pred.load_state_dict(ckpt["state_dict"])
        pred.eval()
        sp = SpatialChannelProjector(32, DV)
        sp.load_state_dict(ckpt["spatial_proj_state_dict"])
        sp.eval()
        print(f"✅ Predictor opt3 (SpatialMLP 14x14x8) loaded: {path}")
        return pred, sp

    elif opt == 2:
        from train_predictor import CausalTransformerPredictor
        pred = CausalTransformerPredictor(128, 2, history_len=3)
        pred.load_state_dict(ckpt["state_dict"])
        pred.eval()
        print(f"✅ Predictor opt2 (CausalTransformer) loaded: {path}")
        return pred, None

    else:
        pred = GeometricHorizonPredictor(latent_dim=128, action_dim=2)
        sd   = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        pred.load_state_dict(sd)
        pred.eval()
        print(f"✅ Predictor opt1 (GeometricHorizon) loaded: {path}")
        return pred, None


# =============================================================================
# Predictor call — dispatches correctly by type
# =============================================================================
def call_predictor(predictor, z: torch.Tensor, action: torch.Tensor,
                   gamma: float = 1.0) -> torch.Tensor:
    if isinstance(predictor, GeometricHorizonPredictor):
        return predictor(z, action, gamma=gamma)
    return predictor(z, action)


# =============================================================================
# Mirror Ascent Planner
# K=64 candidates, R=3 rounds, eta=1.0
# Biologically modulated by Neuropixels rho
# =============================================================================
class MirrorAscentPlanner:
    """
    Iterative softmax reweighting over candidate action sequences.

    Cost per candidate — Wang et al. 2026 §B.1 (linear ramp):
        c_k = mean_{t=1}^{H}  w_t * ||z_goal - z_hat_{k,t}||_2
        w_t = t / H  (linear ramp; terminal step has full weight 1.0)

    Biological modulation (arousal):
        g_t = -c / (1 + rho)
    High rho flattens cost landscape (exploration).
    Low rho sharpens cost landscape (exploitation).

    Mirror ascent update:
        log_q += eta * g_t
        log_q -= logsumexp(log_q)   (normalise on simplex)
    """

    def __init__(self, predictor, spatial_proj=None, action_dim=2,
                 n_candidates=64, n_iters=3, horizon=5, eta=1.0):
        self.predictor    = predictor
        self.spatial_proj = spatial_proj
        self.action_dim   = action_dim
        self.n_candidates = n_candidates
        self.n_iters      = n_iters
        self.horizon      = horizon
        self.eta          = eta

    def plan(self, z_current: np.ndarray, z_goal: np.ndarray,
             rho: float = 0.0, gamma: float = 1.0) -> np.ndarray:
        """
        Args:
            z_current: (128,) or (1568,) or (196,8) current latent
            z_goal:    same shape as z_current
            rho:       Neuropixels arousal scalar in [0, 1]
            gamma:     Temporal horizon scalar (1=local, 2=moderate)

        Returns:
            action: (action_dim,) optimal first action

        Cost weighting — Wang et al. 2026 §B.1 (linear ramp):
            w_k = (k+1) / H
        Earlier steps contribute less; terminal step has full weight.
        Provides denser planning gradient vs uniform accumulation,
        improving navigation around corners and through narrow passages.
        """
        z_c = torch.tensor(z_current, dtype=torch.float32).unsqueeze(0)
        z_g = torch.tensor(z_goal,    dtype=torch.float32).unsqueeze(0)

        # Linear-ramp step weights: w_k = (k+1)/H  for k in 0..H-1
        # Precomputed once per plan() call — not to be confused with
        # the mirror-ascent candidate weights (log_q.exp()) below.
        H            = self.horizon
        step_weights = torch.arange(1, H + 1, dtype=torch.float32) / H  # (H,)

        candidates = torch.randn(self.n_candidates, H, self.action_dim) * 0.1
        log_q      = torch.zeros(self.n_candidates)

        with torch.no_grad():
            for _ in range(self.n_iters):
                costs = []
                for i in range(self.n_candidates):
                    z    = z_c.clone()
                    cost = 0.0
                    for k in range(H):
                        act  = candidates[i, k].unsqueeze(0)
                        z    = call_predictor(self.predictor, z, act, gamma)
                        # Weighted step cost — replaces uniform accumulation
                        step_dist = torch.norm(z.flatten() - z_g.flatten(), p=2).item()
                        cost += step_weights[k].item() * step_dist
                    costs.append((cost / H) / (1.0 + rho))  # mean + rho modulation

                costs_t = torch.tensor(costs)
                log_q   = log_q + self.eta * (-costs_t)
                log_q   = log_q - log_q.logsumexp(dim=0)
                weights    = log_q.exp()
                elite_mean = (weights.view(-1, 1, 1) * candidates).sum(dim=0)
                candidates = (
                    elite_mean.unsqueeze(0)
                    + torch.randn(self.n_candidates, self.horizon, self.action_dim) * 0.05
                )

        best = log_q.argmax().item()
        return candidates[best][0].numpy()


# =============================================================================
# UDP Telemetry (macOS HUD)
# 528-byte struct: <d 2f 128f>
# timestamp(8) + action_xy(8) + latent_z(512) = 528 bytes
# =============================================================================
class UDPTelemetry:
    def __init__(self, host: str = "255.255.255.255", port: int = 5005):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.addr = (host, port)

    def send(self, timestamp: float, x: float, y: float, latent: np.ndarray):
        payload = struct.pack("<d2f128f", timestamp, x, y, *latent[:128].tolist())
        try:
            self.sock.sendto(payload, self.addr)
        except Exception:
            pass


# =============================================================================
# Chart Frame Renderer — matches Phase 2 training renderer exactly
# =============================================================================
def render_chart_frame(bars_df) -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BG, UP, DOWN, WICK, VOL = "#0d0d0d", "#00e5b0", "#ff3a5c", "#555555", "#1a3a4a"
    fig, (ax_p, ax_v) = plt.subplots(
        2, 1, figsize=(2.24, 2.24), dpi=100,
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0},
        facecolor=BG,
    )
    ax_p.set_facecolor(BG)
    ax_v.set_facecolor(BG)

    n      = len(bars_df)
    xs     = np.arange(n)
    opens  = bars_df["open"].values
    highs  = bars_df["high"].values
    lows   = bars_df["low"].values
    closes = bars_df["close"].values
    vols   = bars_df["volume"].values

    for i in xs:
        color  = UP if closes[i] >= opens[i] else DOWN
        bottom = min(opens[i], closes[i])
        height = abs(closes[i] - opens[i]) or (highs[i] - lows[i]) * 0.01
        ax_p.bar(i, height, bottom=bottom, color=color, width=0.8, linewidth=0)
        ax_p.plot([i, i], [lows[i], highs[i]], color=WICK, linewidth=0.6, zorder=0)

    pad = (highs.max() - lows.min()) * 0.05
    ax_p.set_ylim(lows.min() - pad, highs.max() + pad)
    ax_p.set_xlim(-0.5, n - 0.5)
    ax_p.axis("off")
    ax_v.bar(xs, vols, color=VOL, width=0.8, linewidth=0)
    ax_v.set_xlim(-0.5, n - 0.5)
    ax_v.axis("off")

    fig.canvas.draw()
    buf   = fig.canvas.buffer_rgba()
    image = np.frombuffer(buf, dtype=np.uint8).reshape(224, 224, 4)
    plt.close(fig)
    return image[:, :, :3]


# =============================================================================
# Main Loop
# =============================================================================
def run(
    model_path:     str   = "./npu_models/cortex_student_fp32.onnx",
    symbol:         str   = "SPY",
    mock:           bool  = False,
    hz:             float = 1.0,
    predictor_path: str   = None,
):
    print("\n" + "=" * 60)
    print("  CORTEX-16 UNIFIED ENGINE")
    print(f"  Encoder:   {model_path}")
    print(f"  Symbol:    {symbol}")
    print(f"  Mode:      {'MOCK' if mock else 'LIVE PAPER'}")
    print(f"  Rate:      {hz}Hz")
    print("=" * 60 + "\n")

    # Encoder
    engine = InferenceEngine(model_path)

    # Predictor
    spatial_proj = None
    if predictor_path and Path(predictor_path).exists():
        predictor, spatial_proj = load_predictor_from_checkpoint(predictor_path)
    else:
        predictor = GeometricHorizonPredictor(latent_dim=128, action_dim=2)
        predictor.eval()
        print("⚠️  No predictor checkpoint — using random weights")

    planner   = MirrorAscentPlanner(predictor, spatial_proj)
    telemetry = UDPTelemetry()
    goal_mgr  = GoalStateManager("./goals")
    z_goal    = np.zeros(128, dtype=np.float32)
    regime    = "neutral"

    # Alpaca connection
    api = None
    if not mock:
        try:
            import os
            from dotenv import load_dotenv
            load_dotenv()
            from alpaca_trade_api.rest import REST, TimeFrame
            api = REST(
                os.getenv("ALPACA_PAPER_KEY"),
                os.getenv("ALPACA_PAPER_SECRET"),
                "https://paper-api.alpaca.markets",
            )
            tf = TimeFrame
            print("✅ Alpaca connected (paper)")
        except Exception as e:
            print(f"⚠️  Alpaca unavailable: {e} — mock mode")
            mock = True

    # Neuropixels — lazy init happens on first get_neuropixels_rho() call
    # Prints status message (loaded or fallback) on first call only
    print("🧠 Initialising Neuropixels reader...")
    rho_init = get_neuropixels_rho()
    print(f"   Initial rho={rho_init:.4f} (0.0 expected until history accumulates)")

    print(f"\n🟢 Entering loop at {hz}Hz. Ctrl+C to stop.\n")
    tick = 0

    try:
        while True:
            t0 = time.perf_counter()

            # Acquire frame
            bars_df = None
            if mock:
                frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            else:
                try:
                    raw   = api.get_bars(symbol, tf.Minute, limit=60, feed="iex").df
                    raw.columns = [c.lower() for c in raw.columns]
                    bars_df = raw.tail(60)
                    frame   = (
                        render_chart_frame(bars_df)
                        if len(bars_df) >= 60
                        else np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                    )
                except Exception:
                    frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

            # Encode
            z_current = engine.encode(frame)

            # Neuropixels arousal
            rho = get_neuropixels_rho()

            # Update goal state from momentum
            z_goal, regime = goal_mgr.get_goal(bars_df)

            # Plan
            action = planner.plan(z_current, z_goal, rho=rho, gamma=2.0)

            # Telemetry
            telemetry.send(time.time(), float(action[0]), float(action[1]), z_current)

            # Timing
            elapsed_ms = (time.perf_counter() - t0) * 1000
            time.sleep(max(0.0, (1.0 / hz) - elapsed_ms / 1000.0))

            if tick % 10 == 0:
                print(
                    f"Tick {tick:>6} | {elapsed_ms:>7.1f}ms | "
                    f"rho={rho:.3f} | "
                    f"{regime:<18} | "
                    f"action=[{action[0]:+.3f},{action[1]:+.3f}]"
                )
            tick += 1

    except KeyboardInterrupt:
        print("\n🛑 CORTEX-16 loop terminated.")


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CORTEX-16 Unified Execution Engine")
    p.add_argument("--model",     default="./npu_models/cortex_student_fp32.onnx")
    p.add_argument("--predictor", default=None)
    p.add_argument("--symbol",    default="SPY")
    p.add_argument("--hz",        type=float, default=1.0)
    p.add_argument("--mock",      action="store_true")
    args = p.parse_args()

    run(args.model, args.symbol, args.mock, args.hz, args.predictor)
