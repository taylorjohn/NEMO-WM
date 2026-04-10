"""
make_recon_video.py — NeMo-WM RECON Trajectory Visualisation
=============================================================
Replays RECON HDF5 episodes with live overlays showing:
  - GPS trajectory through space (top-left panel)
  - Proprio encoder latent similarity score (localisation confidence)
  - Live ACh / DA / Cortisol / NE signal traces (bottom panel)
  - Heading vector (where the robot is pointing)
  - Hard negative detection events (spike overlays)

Works with synthetic_sprint9 data (velocity + GPS structure).
Swap --hdf5-dir for recon_data/recon_release when accessible.

Usage:
    python make_recon_video.py \
        --hdf5-dir recon_data/synthetic_sprint9 \
        --ckpt checkpoints/cwm/proprio_kctx16_sprint9.pt \
        --n-episodes 3 --fps 15
"""

import argparse, math, glob, os, random
import numpy as np
import torch
import torch.nn.functional as F
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from pathlib import Path

try:
    import h5py
    HAS_H5 = True
except ImportError:
    HAS_H5 = False
    print("pip install h5py")
    exit(1)


# ── Minimal proprio encoder loader ───────────────────────────────────────────

def load_encoder(ckpt_path: str, device: torch.device):
    """Load ProprioEncoderTemporal from Sprint 6c checkpoint."""
    import sys, os
    sys.path.insert(0, os.getcwd())
    from train_proprio_6c import ProprioEncoderTemporal
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    k_ctx   = ckpt.get('k_ctx', 16)
    d_model = ckpt.get('d_model', 64)
    enc = ProprioEncoderTemporal(k_ctx=k_ctx, d_model=d_model).to(device)
    enc.load_state_dict(ckpt['model'])
    enc.eval()
    print(f"Encoder: k_ctx={k_ctx} top1_acc={ckpt.get('top1_acc',0):.4f}")
    return enc, k_ctx


# ── Neuromodulator signals from velocity stream ───────────────────────────────

class NeuroStream:
    """
    Computes rolling neuromodulator signals from raw velocity commands.

    ACh   — temporal precision: low = broad integration
    DA    — reward prediction error: spikes on surprise / direction change
    NE    — gain: amplifies unusual velocities
    Cortisol — domain shift: triggers on abrupt speed/heading change
    eCB   — smoothing: low-pass filter on DA
    """

    def __init__(self):
        self._da   = 0.3
        self._ach  = 0.2
        self._ne   = 0.5
        self._cort = 0.05
        self._ecb  = 0.3
        self._prev_lin = 0.0
        self._prev_ang = 0.0
        self._baseline_lin = None
        self._history = {s: [] for s in ['da','ach','ne','cortisol','ecb']}

    def update(self, lin_vel: float, ang_vel: float) -> dict:
        # DA: surprise = deviation from recent mean
        d_lin = abs(lin_vel - self._prev_lin)
        d_ang = abs(ang_vel - self._prev_ang)
        surprise = float(np.clip(d_lin * 3.0 + d_ang * 1.5, 0, 1))

        # Domain baseline
        if self._baseline_lin is None:
            self._baseline_lin = abs(lin_vel)

        domain_drift = abs(abs(lin_vel) - self._baseline_lin)
        if domain_drift > 0.3:
            self._cort = min(1.0, self._cort + 0.25)
            self._baseline_lin = abs(lin_vel)
        else:
            self._cort = max(0.02, self._cort * 0.96)

        self._da  = float(np.clip(self._da * 0.88 + surprise * 0.5, 0, 1))
        self._ne  = float(np.clip(0.3 + abs(lin_vel) * 0.8 + abs(ang_vel) * 0.3, 0, 1))
        self._ecb = float(self._ecb * 0.85 + self._da * 0.15)
        # ACh stays low (broad integration) — only rises when speed is high
        speed = math.sqrt(lin_vel**2 + ang_vel**2)
        self._ach = float(np.clip(self._ach * 0.95 + speed * 0.15, 0.05, 0.9))

        self._prev_lin = lin_vel
        self._prev_ang = ang_vel

        out = {
            'da':       self._da,
            'ach':      self._ach,
            'ne':       self._ne,
            'cortisol': self._cort,
            'ecb':      self._ecb,
        }
        for k, v in out.items():
            self._history[k].append(v)
        return out

    def history(self, key: str, window: int = 60) -> np.ndarray:
        h = self._history.get(key, [])
        arr = np.array(h[-window:]) if len(h) >= 1 else np.zeros(window)
        if len(arr) < window:
            arr = np.pad(arr, (window - len(arr), 0))
        return arr


# ── Compute proprio latent similarity ────────────────────────────────────────

@torch.no_grad()
def compute_similarity_stream(enc, obs_seq: np.ndarray, k_ctx: int,
                               device: torch.device) -> np.ndarray:
    """
    Compute rolling self-similarity scores along the trajectory.
    High similarity = robot is in a familiar-feeling state.
    Low similarity = robot is somewhere new (potential anomaly).
    Returns array of length T.
    """
    T = len(obs_seq)
    sims = np.full(T, 0.5)

    if T < k_ctx * 2:
        return sims

    # Encode windows
    zs = []
    for t in range(k_ctx, T):
        window = obs_seq[t - k_ctx:t]          # (k_ctx, d_obs)
        x = torch.from_numpy(window.astype(np.float32)).unsqueeze(0).to(device)
        z = enc(x)
        zs.append(z.squeeze(0).cpu().numpy())

    zs = np.array(zs)   # (T-k_ctx, D)

    # Rolling cosine similarity to k_ctx-steps-ago embedding
    for i in range(len(zs) - k_ctx):
        a = zs[i]
        b = zs[i + k_ctx]
        sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
        sims[i + k_ctx + k_ctx] = (sim + 1) / 2   # map [-1,1] → [0,1]

    return sims


# ── Load HDF5 episode ─────────────────────────────────────────────────────────

def load_episode(path: str):
    with h5py.File(path, 'r') as f:
        lin = f['commands/linear_velocity'][:]
        ang = f['commands/angular_velocity'][:]
        gps = f['gps/latlong'][:]
    return lin, ang, gps


def build_obs_seq(lin, ang):
    import math as _math
    heading_rad  = __import__('numpy').cumsum(ang) * 0.25
    heading_norm = (heading_rad % (2 * _math.pi)) / (2 * _math.pi)
    sin_h        = __import__('numpy').sin(heading_rad)
    cos_h        = __import__('numpy').cos(heading_rad)
    d_heading    = __import__('numpy').diff(heading_norm, prepend=heading_norm[0])
    contact      = (__import__('numpy').abs(__import__('numpy').diff(lin, prepend=lin[0])) > 0.15).astype('float32')
    delta_h      = __import__('numpy').diff(heading_rad, prepend=heading_rad[0])
    return __import__('numpy').stack([lin, ang/(_math.pi+1e-8), heading_norm,
                     sin_h, cos_h, d_heading, contact, delta_h],
                    axis=1).astype('float32')


COLORS = {
    'da':       '#ef4444',
    'ach':      '#3b82f6',
    'ne':       '#f59e0b',
    'cortisol': '#f97316',
    'ecb':      '#10b981',
    'sim':      '#a78bfa',
}

SIG_LABELS = {
    'da':       'DA   dopamine',
    'ach':      'ACh  temporal',
    'ne':       'NE   gain',
    'cortisol': 'Cort domain',
    'ecb':      'eCB  smooth',
    'sim':      'Sim  localise',
}

TRACE_WINDOW = 80

def draw_frame(
    t: int,
    gps: np.ndarray,
    neuro: dict,
    neuro_stream: NeuroStream,
    sim_scores: np.ndarray,
    lin_vel: float,
    ang_vel: float,
    heading: float,
    ep_idx: int,
    n_ep: int,
    hdf5_name: str,
) -> np.ndarray:
    """Render one frame. Returns HxWx3 uint8 array."""

    fig = plt.figure(figsize=(9, 5), facecolor='#060B14')
    gs  = gridspec.GridSpec(3, 3, figure=fig,
                            left=0.06, right=0.98,
                            top=0.93, bottom=0.06,
                            wspace=0.35, hspace=0.55)

    # ── Panel A: GPS trajectory ───────────────────────────────────────────
    ax_map = fig.add_subplot(gs[:, 0])
    ax_map.set_facecolor('#0D1117')
    for sp in ax_map.spines.values():
        sp.set_color('#1E293B')

    T_show = t + 1
    lats = gps[:T_show, 0]
    lons = gps[:T_show, 1]

    # Normalise to [0,1] for consistent plotting
    lat_min, lat_max = gps[:, 0].min(), gps[:, 0].max()
    lon_min, lon_max = gps[:, 1].min(), gps[:, 1].max()
    lat_r = max(lat_max - lat_min, 1e-6)
    lon_r = max(lon_max - lon_min, 1e-6)

    lats_n = (lats - lat_min) / lat_r
    lons_n = (lons - lon_min) / lon_r
    all_lats_n = (gps[:, 0] - lat_min) / lat_r
    all_lons_n = (gps[:, 1] - lon_min) / lon_r

    # Ghost full path
    ax_map.plot(all_lons_n, all_lats_n, color='#1E293B', lw=1.2, zorder=1)

    # Colour trajectory by sim score
    if T_show > 1:
        for i in range(1, T_show):
            s = sim_scores[i]
            c = plt.cm.RdYlGn(s)
            ax_map.plot([lons_n[i-1], lons_n[i]],
                        [lats_n[i-1], lats_n[i]],
                        color=c, lw=2.0, zorder=2)

    # Robot position
    cx, cy = lons_n[-1], lats_n[-1]
    ax_map.add_patch(plt.Circle((cx, cy), 0.04, color='white', zorder=5))

    # Heading arrow
    hx = math.cos(heading * 2 * math.pi) * 0.08
    hy = math.sin(heading * 2 * math.pi) * 0.08
    ax_map.annotate('', xy=(cx + hx, cy + hy), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle='->', color='#f59e0b',
                                   lw=1.8), zorder=6)

    ax_map.set_xlim(-0.1, 1.1); ax_map.set_ylim(-0.1, 1.1)
    ax_map.set_xticks([]); ax_map.set_yticks([])
    ax_map.set_title('GPS Trajectory', color='#94a3b8', fontsize=7,
                     fontfamily='monospace', pad=4)
    ax_map.text(0.02, 0.97, f'ep {ep_idx+1}/{n_ep}', color='#64748b',
                fontsize=6, fontfamily='monospace', transform=ax_map.transAxes,
                va='top')

    # Localisation quality colourbar indicator
    sim_now = float(sim_scores[t])
    qual = 'LOCALISED' if sim_now > 0.65 else 'SEARCHING' if sim_now > 0.4 else 'LOST'
    qual_c = '#10b981' if sim_now > 0.65 else '#f59e0b' if sim_now > 0.4 else '#ef4444'
    ax_map.text(0.5, 0.03, qual, color=qual_c, fontsize=6.5,
                fontfamily='monospace', transform=ax_map.transAxes,
                ha='center', va='bottom', fontweight='bold')

    # ── Panel B: Neuromodulator signal cards ──────────────────────────────
    signals = ['da', 'ach', 'ne', 'cortisol', 'ecb', 'sim']
    vals    = {**neuro, 'sim': sim_now}

    for idx, sig in enumerate(signals):
        row = idx // 2
        col = (idx % 2) + 1
        ax_s = fig.add_subplot(gs[row, col])
        ax_s.set_facecolor('#0D1117')
        for sp in ax_s.spines.values():
            sp.set_color('#1E293B')

        c = COLORS[sig]

        # Trace history
        if sig == 'sim':
            hist = sim_scores[max(0, t - TRACE_WINDOW):t + 1]
            if len(hist) < TRACE_WINDOW:
                hist = np.pad(hist, (TRACE_WINDOW - len(hist), 0))
        else:
            hist = neuro_stream.history(sig, TRACE_WINDOW)

        x = np.arange(len(hist))
        ax_s.fill_between(x, hist, alpha=0.18, color=c)
        ax_s.plot(x, hist, color=c, lw=1.2)

        # Current value marker
        val = float(vals[sig])
        ax_s.axhline(val, color=c, lw=0.7, ls='--', alpha=0.5)
        ax_s.scatter([len(hist) - 1], [val], color=c, s=18, zorder=5)

        ax_s.set_xlim(0, TRACE_WINDOW)
        ax_s.set_ylim(-0.05, 1.12)
        ax_s.set_xticks([]); ax_s.set_yticks([0, 0.5, 1])
        ax_s.tick_params(colors='#475569', labelsize=5)

        label = SIG_LABELS[sig]
        ax_s.set_title(f'{label}  {val:.2f}',
                       color=c, fontsize=6.2,
                       fontfamily='monospace', pad=2)

    # ── Title bar ─────────────────────────────────────────────────────────
    fig.text(0.5, 0.975,
             f'NeMo-WM  ·  RECON Navigation  ·  '
             f't={t:4d}  lin={lin_vel:+.2f}  ang={ang_vel:+.2f}',
             ha='center', va='top', color='#e2e8f0',
             fontsize=7.5, fontfamily='monospace', fontweight='bold')

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(),
                        dtype=np.uint8).reshape(h, w, 4)
    img = buf[:, :, :3].copy()
    plt.close(fig)
    return img


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hdf5-dir',   default='recon_data/synthetic_sprint9')
    ap.add_argument('--ckpt',       default='checkpoints/cwm/proprio_kctx16_sprint9.pt')
    ap.add_argument('--n-episodes', type=int, default=3)
    ap.add_argument('--fps',        type=int, default=15)
    ap.add_argument('--skip',       type=int, default=2,
                    help='Render every N-th frame (1=all, 2=half)')
    ap.add_argument('--seed',       type=int, default=42)
    ap.add_argument('--device',     default='cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    Path('figures').mkdir(exist_ok=True)

    # Load encoder
    enc, k_ctx = load_encoder(args.ckpt, device)

    # Select episodes
    files = sorted(glob.glob(os.path.join(args.hdf5_dir, '*.hdf5')))
    if not files:
        print(f"No HDF5 files in {args.hdf5_dir}")
        return
    random.seed(args.seed)
    selected = random.sample(files, min(args.n_episodes, len(files)))
    print(f"Rendering {len(selected)} episodes from {args.hdf5_dir}")

    all_frames = []

    for ep_idx, fpath in enumerate(selected):
        name = os.path.basename(fpath)
        print(f"\n  ep{ep_idx+1}: {name}")
        lin, ang, gps = load_episode(fpath)
        obs_seq = build_obs_seq(lin, ang)
        T = len(lin)

        # Pre-compute similarity stream
        print(f"    Computing latent similarities ({T} steps)...")
        sim_scores = compute_similarity_stream(enc, obs_seq, k_ctx, device)

        # Build heading array
        headings = np.cumsum(ang) * 0.25
        headings = (headings % (2 * math.pi)) / (2 * math.pi)

        neuro_stream = NeuroStream()
        ep_frames = []

        for t in range(T):
            neuro = neuro_stream.update(float(lin[t]), float(ang[t]))

            if t % args.skip != 0:
                continue

            frame = draw_frame(
                t=t,
                gps=gps,
                neuro=neuro,
                neuro_stream=neuro_stream,
                sim_scores=sim_scores,
                lin_vel=float(lin[t]),
                ang_vel=float(ang[t]),
                heading=float(headings[t]),
                ep_idx=ep_idx,
                n_ep=len(selected),
                hdf5_name=name,
            )
            ep_frames.append(frame)

        # Hold last frame
        if ep_frames:
            ep_frames.extend([ep_frames[-1]] * 15)

        all_frames.extend(ep_frames)
        print(f"    {len(ep_frames)} frames rendered")

    if not all_frames:
        print("No frames rendered.")
        return

    # Save
    gif_path = 'figures/recon_neuromod.gif'
    imageio.mimsave(gif_path, all_frames, fps=args.fps, loop=0)
    size_mb = Path(gif_path).stat().st_size / 1e6
    print(f"\nSaved {gif_path}  ({len(all_frames)} frames, {size_mb:.1f} MB)")

    web_path = 'figures/recon_neuromod_web.gif'
    imageio.mimsave(web_path, all_frames[::2], fps=args.fps // 2, loop=0)
    print(f"Saved {web_path}  ({len(all_frames)//2} frames)")

    try:
        mp4_path = 'figures/recon_neuromod.mp4'
        imageio.mimsave(mp4_path, all_frames, fps=args.fps * 2,
                        macro_block_size=1)
        print(f"Saved {mp4_path}")
    except Exception as e:
        print(f"MP4 skipped: {e}")


if __name__ == '__main__':
    main()
