"""
teleop_pusht.py — Mouse + Keyboard Teleoperation for PushT
=============================================================
MOUSE CONTROL: Click/drag to set agent target position directly.
This is much more intuitive than keyboard for pushing tasks.

Controls:
  MOUSE CLICK/DRAG — set agent target (most intuitive!)
  WASD / Arrow Keys — nudge agent (alternative)
  SPACE — pause/resume
  R — reset episode
  Q — quit and save
  F — toggle fast mode
  +/- — adjust keyboard speed

The gym-pusht action space expects absolute [x, y] target positions
in [0, 512] coordinates. Mouse mapping is 1:1 with the render window.

Saves to: data/teleop_pusht/
  - episodes.npz (states, actions, frames, episode boundaries)

Usage:
    python teleop_pusht.py                    # collect demos (mouse)
    python teleop_pusht.py --episodes 20      # collect 20 episodes
    python teleop_pusht.py --review           # review saved demos
    python teleop_pusht.py --train            # train vision policy
"""

import argparse
import time
import numpy as np
from pathlib import Path

MAX_STEPS = 1000
RENDER_SIZE = 96
SAVE_DIR = Path("data/teleop_pusht")


def collect_demos(n_episodes=10):
    """Collect human demonstrations via mouse + keyboard."""
    try:
        import gymnasium
        import gym_pusht
        import pygame
    except ImportError as e:
        print(f"Missing: {e}")
        print("Install: pip install gymnasium gym-pusht pygame")
        return

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    pygame.init()
    WINDOW = 512
    screen = pygame.display.set_mode((WINDOW, WINDOW))
    pygame.display.set_caption(
        "PushT — CLICK to move agent | R=reset | Q=save&quit")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 14)
    big_font = pygame.font.SysFont("consolas", 28)

    env = gymnasium.make("gym_pusht/PushT-v0",
                          obs_type="pixels",
                          render_mode="rgb_array")

    all_states = []
    all_actions = []
    all_frames = []
    episode_lengths = []
    coverages = []

    obs_test, _ = env.reset(seed=0)
    obs_h, obs_w = obs_test.shape[:2]
    print(f"  Observation size: {obs_w}x{obs_h}")

    ep = 0
    quit_all = False

    while ep < n_episodes and not quit_all:
        obs, info = env.reset(seed=42 + ep)
        target_pos = np.array([256.0, 400.0])

        ep_states = []
        ep_actions = []
        ep_frames = []
        best_coverage = 0
        step = 0
        paused = False
        fast_mode = False
        kbd_speed = 25.0
        mouse_active = False

        print(f"\n--- Episode {ep+1}/{n_episodes} ---")
        print("  CLICK/DRAG to move | R=reset | Q=save&quit | F=fast")

        while step < MAX_STEPS:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_all = True
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        quit_all = True
                        break
                    elif event.key == pygame.K_r:
                        print("  Reset")
                        ep_states.clear()
                        ep_actions.clear()
                        ep_frames.clear()
                        step = 0
                        best_coverage = 0
                        obs, info = env.reset(seed=42 + ep)
                        target_pos = np.array([256.0, 400.0])
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_f:
                        fast_mode = not fast_mode
                        print(f"  Fast: {'ON' if fast_mode else 'OFF'}")
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                        kbd_speed = min(100, kbd_speed + 5)
                    elif event.key == pygame.K_MINUS:
                        kbd_speed = max(5, kbd_speed - 5)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_active = True
                    mx, my = event.pos
                    target_pos = np.array([float(mx), float(my)])
                elif event.type == pygame.MOUSEBUTTONUP:
                    mouse_active = False
                elif event.type == pygame.MOUSEMOTION and mouse_active:
                    mx, my = event.pos
                    target_pos = np.array([float(mx), float(my)])

            if quit_all:
                break

            if paused:
                render = env.render()
                surf = pygame.surfarray.make_surface(
                    np.transpose(render, (1, 0, 2)))
                screen.blit(pygame.transform.scale(surf, (WINDOW, WINDOW)),
                            (0, 0))
                txt = big_font.render("PAUSED", True, (255, 255, 0))
                screen.blit(txt, (WINDOW//2 - 60, WINDOW//2))
                pygame.display.flip()
                clock.tick(30)
                continue

            keys = pygame.key.get_pressed()
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                target_pos[1] -= kbd_speed
            if keys[pygame.K_s] or keys[pygame.K_DOWN]:
                target_pos[1] += kbd_speed
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                target_pos[0] -= kbd_speed
            if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                target_pos[0] += kbd_speed

            target_pos = np.clip(target_pos, 0, 512)

            ep_states.append(target_pos.copy())
            ep_actions.append(target_pos.copy())

            obs, reward, terminated, truncated, info = env.step(target_pos)
            coverage = info.get("coverage", 0)
            best_coverage = max(best_coverage, coverage)

            if obs.shape[0] != RENDER_SIZE or obs.shape[1] != RENDER_SIZE:
                try:
                    from PIL import Image
                    img = Image.fromarray(obs)
                    img = img.resize((RENDER_SIZE, RENDER_SIZE),
                                     Image.BILINEAR)
                    frame = np.array(img)
                except ImportError:
                    sy = obs.shape[0] // RENDER_SIZE
                    sx = obs.shape[1] // RENDER_SIZE
                    frame = obs[::sy, ::sx][:RENDER_SIZE, :RENDER_SIZE]
            else:
                frame = obs
            ep_frames.append(frame)

            step += 1

            render = env.render()
            surf = pygame.surfarray.make_surface(
                np.transpose(render, (1, 0, 2)))
            if render.shape[0] != WINDOW:
                surf = pygame.transform.scale(surf, (WINDOW, WINDOW))
            screen.blit(surf, (0, 0))

            tx, ty = int(target_pos[0]), int(target_pos[1])
            sx = WINDOW / 512
            dtx, dty = int(tx * sx), int(ty * sx)
            pygame.draw.circle(screen, (255, 0, 0), (dtx, dty), 8, 2)
            pygame.draw.line(screen, (255, 0, 0),
                             (dtx-12, dty), (dtx+12, dty), 1)
            pygame.draw.line(screen, (255, 0, 0),
                             (dtx, dty-12), (dtx, dty+12), 1)

            cov_color = (0, 255, 100) if coverage > 0.5 else (
                (255, 200, 0) if coverage > 0.2 else (200, 200, 200))
            lines = [
                f"Ep {ep+1}/{n_episodes}  Step {step}/{MAX_STEPS}",
                f"Coverage: {coverage:.3f}  Best: {best_coverage:.3f}",
                f"Target: ({target_pos[0]:.0f}, {target_pos[1]:.0f})",
                f"{'FAST ' if fast_mode else ''}Click to move agent",
            ]
            for i, line in enumerate(lines):
                txt = font.render(line, True, cov_color)
                screen.blit(txt, (8, 8 + i * 18))

            bar_w = int(coverage * 300)
            pygame.draw.rect(screen, (40, 40, 40), (8, 85, 300, 10))
            pygame.draw.rect(screen, cov_color, (8, 85, bar_w, 10))
            best_x = int(best_coverage * 300) + 8
            pygame.draw.line(screen, (255, 255, 255),
                             (best_x, 83), (best_x, 97), 2)

            pygame.display.flip()
            clock.tick(30 if not fast_mode else 60)

            if coverage >= 0.95:
                print(f"  SUCCESS! Coverage {coverage:.3f} at step {step}")
                break
            if terminated or truncated:
                break

        if ep_states and not quit_all:
            all_states.extend(ep_states)
            all_actions.extend(ep_actions)
            all_frames.extend(ep_frames)
            episode_lengths.append(len(ep_states))
            coverages.append(best_coverage)
            print(f"  Done: {len(ep_states)} steps, coverage={best_coverage:.3f}")
            ep += 1
        elif quit_all and ep_states:
            all_states.extend(ep_states)
            all_actions.extend(ep_actions)
            all_frames.extend(ep_frames)
            episode_lengths.append(len(ep_states))
            coverages.append(best_coverage)
            ep += 1

    env.close()
    pygame.quit()

    if not all_states:
        print("No demos collected!")
        return

    states = np.array(all_states, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.float32)
    frames = np.array(all_frames, dtype=np.uint8)
    ep_lens = np.array(episode_lengths)

    np.savez_compressed(
        str(SAVE_DIR / "episodes.npz"),
        states=states,
        actions=actions,
        frames=frames,
        episode_lengths=ep_lens,
    )

    print(f"\n{'='*60}")
    print(f"  Saved {len(episode_lengths)} episodes to {SAVE_DIR}/")
    print(f"  Total steps: {len(states)}")
    print(f"  Frames: {frames.shape}")
    print(f"  Coverages: {[f'{c:.3f}' for c in coverages]}")
    print(f"  Mean coverage: {np.mean(coverages):.3f}")
    print(f"  Max coverage: {np.max(coverages):.3f}")
    print(f"  File: {(SAVE_DIR / 'episodes.npz').stat().st_size / 1e6:.1f} MB")
    print(f"{'='*60}")
    print(f"\nTrain: python teleop_pusht.py --train")


def review_demos():
    """Review collected demos."""
    npz_path = SAVE_DIR / "episodes.npz"
    if not npz_path.exists():
        print(f"No demos at {npz_path}")
        return

    data = np.load(str(npz_path))
    states = data["states"]
    ep_lens = data["episode_lengths"]

    print(f"  Episodes: {len(ep_lens)}")
    print(f"  Total steps: {len(states)}")

    idx = 0
    for i, elen in enumerate(ep_lens):
        ep_s = states[idx:idx+elen]
        dist = np.sqrt(np.sum(np.diff(ep_s, axis=0)**2, axis=1)).sum()
        print(f"  Ep {i+1}: {elen} steps, dist={dist:.0f}px")
        idx += elen


def train_on_demos(epochs=300):
    """Train vision policy on collected demos."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    npz_path = SAVE_DIR / "episodes.npz"
    if not npz_path.exists():
        print(f"No demos at {npz_path}")
        return

    data = np.load(str(npz_path))
    frames = data["frames"]
    actions = data["actions"]
    ep_lens = data["episode_lengths"]

    print(f"  Episodes: {len(ep_lens)}, Frames: {len(frames)}")
    actions_norm = actions / 512.0

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(6, 32, 5, 2, 2), nn.BatchNorm2d(32), nn.GELU(),
                nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.GELU(),
                nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.GELU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 128))
        def forward(self, x): return self.net(x)

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(131, 256), nn.GELU(),
                nn.Linear(256, 256), nn.GELU(),
                nn.Linear(256, 2))
        def forward(self, z, x_t, t):
            return self.net(torch.cat([z, x_t, t.unsqueeze(-1)], -1))
        def sample(self, z, n=10):
            B = z.shape[0]; x = torch.randn(B, 2); dt = 1.0/n
            for i in range(n):
                x = x + self.forward(z, x, torch.full((B,), i*dt)) * dt
            return x

    enc = Encoder(); pol = Policy()
    params = sum(p.numel() for p in enc.parameters()) + \
             sum(p.numel() for p in pol.parameters())
    print(f"  Params: {params:,}")
    opt = torch.optim.AdamW(list(enc.parameters())+list(pol.parameters()),
                             lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    n = len(frames) - 1
    best = float("inf")
    for ep in range(1, epochs+1):
        enc.train(); pol.train()
        idx = np.random.choice(n, min(256, n), replace=False)
        cur = torch.from_numpy(frames[idx+1].astype(np.float32)/255).permute(0,3,1,2)
        pre = torch.from_numpy(frames[idx].astype(np.float32)/255).permute(0,3,1,2)
        x = torch.cat([cur, pre], 1)
        a = torch.from_numpy(actions_norm[idx+1])
        z = enc(x); t = torch.rand(len(idx))
        noise = torch.randn_like(a)
        x_t = t.unsqueeze(-1)*a + (1-t.unsqueeze(-1))*noise
        loss = F.mse_loss(pol(z, x_t, t), a - noise)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        best = min(best, loss.item())
        if ep % 50 == 0: print(f"  ep {ep}/{epochs} loss={loss.item():.4f} best={best:.4f}")

    Path("checkpoints/teleop_pusht").mkdir(parents=True, exist_ok=True)
    torch.save({"enc": enc.state_dict(), "pol": pol.state_dict(),
                "loss": best, "n_demos": len(ep_lens)},
               "checkpoints/teleop_pusht/best.pt")
    print(f"\n  Best loss: {best:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--review", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()

    if args.review: review_demos()
    elif args.train: train_on_demos(args.epochs)
    else: collect_demos(args.episodes)
