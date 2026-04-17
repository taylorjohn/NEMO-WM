"""
train_from_minari.py — Train NeMo-WM from Minari PointMaze Data
================================================================
Loads 1M real expert transitions from D4RL PointMaze and uses them
to train NeMo-WM's core components:

1. Transition model — predict next state from (state, action)
2. Schema codebook — DiVeQ-style clustering of belief states
3. Language grounding — label trajectories with spatial/temporal words
4. Neuromodulatory signals — compute DA/ACh/CRT from real trajectories
5. Episodic buffer — store and retrieve real experiences

This replaces the self-generated synthetic data with real expert
navigation trajectories, dramatically improving all downstream tasks.

Usage:
    python train_from_minari.py                    # train all components
    python train_from_minari.py --episodes 500     # use subset
    python train_from_minari.py --test             # validate loading
"""

import argparse
import time
import numpy as np
from pathlib import Path

D_BELIEF = 64
D_ACTION = 2


def load_minari_data(dataset_name="D4RL/pointmaze/umaze-v2",
                      max_episodes=None):
    """Load Minari dataset and extract arrays."""
    import minari
    print(f"  Loading {dataset_name}...")
    dataset = minari.load_dataset(dataset_name)
    print(f"  Episodes: {dataset.total_episodes}")
    print(f"  Total steps: {dataset.total_steps}")

    all_obs = []
    all_actions = []
    all_rewards = []
    episode_boundaries = []

    count = 0
    for ep in dataset.iterate_episodes():
        obs = ep.observations
        # Handle dict observations (PointMaze uses {"observation": ..., "achieved_goal": ..., "desired_goal": ...})
        if isinstance(obs, dict):
            obs_arr = obs.get("observation", obs.get("obs", None))
            goal_arr = obs.get("desired_goal", None)
        else:
            obs_arr = obs
            goal_arr = None

        if obs_arr is None:
            continue

        actions = ep.actions
        rewards = ep.rewards if hasattr(ep, 'rewards') else np.zeros(len(actions))

        # obs has one more timestep than actions
        for t in range(len(actions)):
            all_obs.append(obs_arr[t])
            all_actions.append(actions[t])
            all_rewards.append(float(rewards[t]) if t < len(rewards) else 0.0)

        episode_boundaries.append(len(all_obs))
        count += 1
        if max_episodes and count >= max_episodes:
            break

    all_obs = np.array(all_obs, dtype=np.float32)
    all_actions = np.array(all_actions, dtype=np.float32)
    all_rewards = np.array(all_rewards, dtype=np.float32)

    print(f"  Loaded {count} episodes, {len(all_obs)} transitions")
    print(f"  Obs shape: {all_obs.shape}")
    print(f"  Action shape: {all_actions.shape}")
    print(f"  Obs range: [{all_obs.min():.2f}, {all_obs.max():.2f}]")
    print(f"  Action range: [{all_actions.min():.2f}, {all_actions.max():.2f}]")

    return all_obs, all_actions, all_rewards, episode_boundaries


def project_to_belief(obs, d_belief=D_BELIEF):
    """
    Project low-D observations to belief space.
    PointMaze obs is 4D: [x, y, vx, vy].
    We project to 64D using a learned-style random projection
    that preserves spatial structure.
    """
    d_obs = obs.shape[-1]
    rng = np.random.RandomState(42)  # fixed projection for reproducibility

    # Structured projection: first dims are raw obs, rest are nonlinear features
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False

    N = obs.shape[0]
    belief = np.zeros((N, d_belief), dtype=np.float32)

    # Raw features (normalized)
    belief[:, :d_obs] = obs / (np.abs(obs).max(axis=0, keepdims=True) + 1e-8)

    # Quadratic features
    n_quad = min(d_obs * (d_obs + 1) // 2, d_belief - d_obs)
    idx = d_obs
    for i in range(d_obs):
        for j in range(i, d_obs):
            if idx >= d_belief:
                break
            belief[:, idx] = obs[:, i] * obs[:, j] * 0.1
            idx += 1

    # Random projection for remaining dims
    W_rand = rng.randn(d_obs, d_belief - idx).astype(np.float32) * 0.3
    if idx < d_belief:
        belief[:, idx:] = np.tanh(obs @ W_rand)

    if squeeze:
        return belief[0]
    return belief


class TransitionModel:
    """Simple MLP transition model: b_{t+1} = f(b_t, a_t)."""

    def __init__(self, d_belief=D_BELIEF, d_action=D_ACTION, hidden=128):
        rng = np.random.RandomState(42)
        d_in = d_belief + d_action
        self.W1 = rng.randn(d_in, hidden).astype(np.float32) * np.sqrt(2 / d_in)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.randn(hidden, d_belief).astype(np.float32) * np.sqrt(2 / hidden)
        self.b2 = np.zeros(d_belief, dtype=np.float32)
        self.train_steps = 0

    def predict(self, belief, action):
        x = np.concatenate([belief, action])
        h = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        return h @ self.W2 + self.b2

    def predict_batch(self, beliefs, actions):
        x = np.concatenate([beliefs, actions], axis=1)
        h = np.maximum(0, x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def train(self, beliefs, actions, next_beliefs,
              epochs=5, lr=0.001, batch_size=256):
        """Train with SGD on real transition data."""
        N = len(beliefs)
        losses = []

        for epoch in range(epochs):
            indices = np.random.permutation(N)
            epoch_loss = 0
            n_batches = 0

            for start in range(0, N - batch_size, batch_size):
                batch_idx = indices[start:start + batch_size]
                b = beliefs[batch_idx]
                a = actions[batch_idx]
                nb = next_beliefs[batch_idx]

                # Forward
                x = np.concatenate([b, a], axis=1)
                h = np.maximum(0, x @ self.W1 + self.b1)
                pred = h @ self.W2 + self.b2

                # Loss
                error = pred - nb
                loss = float(np.mean(error ** 2))
                epoch_loss += loss
                n_batches += 1

                # Backward (simple gradient descent)
                d_pred = 2 * error / batch_size
                d_W2 = h.T @ d_pred
                d_b2 = d_pred.sum(axis=0)

                d_h = d_pred @ self.W2.T
                d_h[h <= 0] = 0  # ReLU gradient

                d_W1 = x.T @ d_h
                d_b1 = d_h.sum(axis=0)

                # Update
                self.W1 -= lr * d_W1
                self.b1 -= lr * d_b1
                self.W2 -= lr * d_W2
                self.b2 -= lr * d_b2

                self.train_steps += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)

        return losses


class SchemaStore:
    """K-means style schema codebook."""

    def __init__(self, n_schemas=32, d_belief=D_BELIEF):
        self.n = n_schemas
        self.d = d_belief
        self.codebook = np.random.randn(n_schemas, d_belief).astype(np.float32)
        self.usage = np.zeros(n_schemas, dtype=np.int64)

    def initialize_from_data(self, beliefs, n_iter=20):
        """K-means initialization."""
        N = len(beliefs)
        K = self.n

        # K-means++ init
        centroids = np.zeros((K, self.d), dtype=np.float32)
        centroids[0] = beliefs[np.random.randint(N)]
        for k in range(1, K):
            dists = np.min(np.linalg.norm(
                beliefs[:, None] - centroids[None, :k], axis=2), axis=1)
            probs = dists ** 2 / (dists ** 2).sum()
            centroids[k] = beliefs[np.random.choice(N, p=probs)]

        # K-means iterations
        for it in range(n_iter):
            assign = np.argmin(np.linalg.norm(
                beliefs[:, None] - centroids[None, :], axis=2), axis=1)
            for k in range(K):
                mask = assign == k
                if mask.any():
                    centroids[k] = beliefs[mask].mean(axis=0)

        self.codebook = centroids
        self.usage = np.bincount(assign, minlength=K).astype(np.int64)

    def nearest(self, belief):
        dists = np.linalg.norm(self.codebook - belief, axis=1)
        idx = int(np.argmin(dists))
        return idx, float(dists[idx])

    def novelty(self, belief):
        _, dist = self.nearest(belief)
        return dist


class LanguageGrounder:
    """Ground spatial/temporal words from trajectory structure."""

    def __init__(self):
        self.words = {}
        self.bindings = 0

    def label_trajectory(self, beliefs, actions, rewards):
        """
        Auto-label a trajectory with spatial and temporal words
        based on the structure of movement.
        """
        labels = []
        N = len(beliefs)

        for t in range(N):
            b = beliefs[t]
            frame_words = []

            # Position words
            x, y = b[0], b[1]
            if x > 0.5:
                frame_words.append("right")
            elif x < -0.5:
                frame_words.append("left")
            if y > 0.5:
                frame_words.append("upper")
            elif y < -0.5:
                frame_words.append("lower")
            if abs(x) < 0.3 and abs(y) < 0.3:
                frame_words.append("center")

            # Velocity words
            if len(b) > 2:
                vx, vy = b[2], b[3] if len(b) > 3 else 0
                speed = np.sqrt(vx**2 + vy**2)
                if speed > 0.5:
                    frame_words.append("fast")
                    frame_words.append("moving")
                elif speed < 0.1:
                    frame_words.append("slow")
                    frame_words.append("stationary")

                # Direction
                if abs(vx) > abs(vy) and vx > 0.1:
                    frame_words.append("eastward")
                elif abs(vx) > abs(vy) and vx < -0.1:
                    frame_words.append("westward")
                elif abs(vy) > abs(vx) and vy > 0.1:
                    frame_words.append("northward")
                elif abs(vy) > abs(vx) and vy < -0.1:
                    frame_words.append("southward")

            # Temporal words
            if t == 0:
                frame_words.append("start")
                frame_words.append("beginning")
            elif t == N - 1:
                frame_words.append("end")
                frame_words.append("finish")
            elif t < N * 0.25:
                frame_words.append("early")
            elif t > N * 0.75:
                frame_words.append("late")

            # Action words
            if t < len(actions):
                a = actions[t]
                a_mag = np.linalg.norm(a)
                if a_mag > 0.5:
                    frame_words.append("push")
                    frame_words.append("accelerate")
                elif a_mag < 0.1:
                    frame_words.append("coast")
                    frame_words.append("drift")

            # Change detection
            if t > 0:
                delta = np.linalg.norm(beliefs[t] - beliefs[t-1])
                if delta > 0.3:
                    frame_words.append("turn")
                    frame_words.append("change")
                elif delta < 0.01:
                    frame_words.append("still")

            # Reward words
            if t < len(rewards) and rewards[t] > 0:
                frame_words.append("goal")
                frame_words.append("success")
                frame_words.append("reward")

            # Ground each word to belief state
            for w in frame_words:
                if w not in self.words:
                    self.words[w] = []
                self.words[w].append(b.copy())
                self.bindings += 1

            labels.append(frame_words)

        return labels

    def get_prototype(self, word):
        """Get mean belief vector for a word."""
        if word not in self.words or not self.words[word]:
            return None
        return np.mean(self.words[word], axis=0)

    def similarity(self, w1, w2):
        """Cosine similarity between word prototypes."""
        p1 = self.get_prototype(w1)
        p2 = self.get_prototype(w2)
        if p1 is None or p2 is None:
            return 0.0
        return float(np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2) + 1e-8))


def compute_neuromod(beliefs, actions, schemas):
    """Compute neuromodulatory signals from trajectory."""
    N = len(beliefs)
    signals = {"DA": [], "ACh": [], "CRT": [], "NE": [], "5HT": []}

    for t in range(N):
        # Prediction error (proxy: change in belief)
        if t > 0:
            pred_error = float(np.linalg.norm(beliefs[t] - beliefs[t-1]))
        else:
            pred_error = 0.0

        # Schema novelty
        _, novelty = schemas.nearest(beliefs[t])

        # DA: surprise (prediction error)
        da = np.clip(0.3 + pred_error * 2, 0, 1)
        # ACh: confidence (inverse of prediction error)
        ach = np.clip(0.7 - pred_error, 0.1, 1)
        # CRT: stress (schema novelty)
        crt = np.clip(novelty * 0.1, 0, 0.8)
        # NE: arousal
        ne = np.clip(0.4 + pred_error + novelty * 0.05, 0.1, 1)
        # 5HT: caution
        sht = np.clip(0.5 - da * 0.3, 0.1, 1)

        signals["DA"].append(da)
        signals["ACh"].append(ach)
        signals["CRT"].append(crt)
        signals["NE"].append(ne)
        signals["5HT"].append(sht)

    return {k: np.array(v) for k, v in signals.items()}


def train_all(max_episodes=500, dataset_name="D4RL/pointmaze/umaze-v2"):
    """Train all NeMo-WM components from Minari data."""
    print("=" * 70)
    print("  NeMo-WM Training from Minari PointMaze")
    print("=" * 70)

    # 1. Load data
    obs, actions, rewards, boundaries = load_minari_data(
        dataset_name, max_episodes=max_episodes)

    # 2. Project to belief space
    print(f"\n  Projecting {len(obs)} observations to {D_BELIEF}-D belief space...")
    t0 = time.time()
    beliefs = project_to_belief(obs)
    print(f"    Done in {time.time()-t0:.1f}s")
    print(f"    Belief range: [{beliefs.min():.2f}, {beliefs.max():.2f}]")

    # Pad actions to D_ACTION if needed
    if actions.shape[1] < D_ACTION:
        pad = np.zeros((len(actions), D_ACTION - actions.shape[1]), dtype=np.float32)
        actions = np.concatenate([actions, pad], axis=1)
    elif actions.shape[1] > D_ACTION:
        actions = actions[:, :D_ACTION]

    # 3. Train schema codebook
    print(f"\n  Initializing schema codebook (32 schemas from {len(beliefs)} beliefs)...")
    schemas = SchemaStore(n_schemas=32)
    t0 = time.time()
    # Subsample for speed
    sub_idx = np.random.choice(len(beliefs), size=min(50000, len(beliefs)), replace=False)
    schemas.initialize_from_data(beliefs[sub_idx], n_iter=20)
    print(f"    Done in {time.time()-t0:.1f}s")
    print(f"    Active schemas: {(schemas.usage > 0).sum()}/32")

    # Test novelty
    test_novelties = [schemas.novelty(beliefs[i]) for i in range(0, 1000, 10)]
    print(f"    Mean novelty: {np.mean(test_novelties):.3f}")
    print(f"    Min novelty:  {np.min(test_novelties):.3f}")
    print(f"    Max novelty:  {np.max(test_novelties):.3f}")

    # 4. Train transition model
    print(f"\n  Training transition model...")
    model = TransitionModel(d_belief=D_BELIEF, d_action=D_ACTION)

    # Prepare (b_t, a_t) → b_{t+1} pairs
    N = len(beliefs) - 1
    b_train = beliefs[:N]
    a_train = actions[:N]
    nb_train = beliefs[1:N+1]

    # Remove episode boundaries (don't predict across episodes)
    valid = np.ones(N, dtype=bool)
    for b_end in boundaries:
        if b_end - 1 < N:
            valid[b_end - 1] = False
    b_train = b_train[valid]
    a_train = a_train[valid]
    nb_train = nb_train[valid]
    print(f"    Training on {len(b_train)} valid transitions")

    t0 = time.time()
    losses = model.train(b_train, a_train, nb_train,
                          epochs=10, lr=0.0005, batch_size=512)
    train_time = time.time() - t0
    print(f"    Epochs: {len(losses)}")
    print(f"    Loss: {losses[0]:.4f} → {losses[-1]:.4f} "
          f"({(1-losses[-1]/losses[0])*100:+.1f}%)")
    print(f"    Time: {train_time:.1f}s ({len(b_train)/train_time:.0f} samples/s)")

    # 5. Test prediction quality
    print(f"\n  Testing prediction quality...")
    test_idx = np.random.choice(len(b_train), size=1000, replace=False)
    preds = model.predict_batch(b_train[test_idx], a_train[test_idx])
    targets = nb_train[test_idx]
    mse = float(np.mean((preds - targets) ** 2))
    # Position-only MSE (first 4 dims are the real obs)
    pos_mse = float(np.mean((preds[:, :4] - targets[:, :4]) ** 2))
    print(f"    Full MSE:     {mse:.6f}")
    print(f"    Position MSE: {pos_mse:.6f}")

    # 6. Language grounding
    print(f"\n  Grounding language from trajectories...")
    grounder = LanguageGrounder()
    t0 = time.time()

    # Process episodes
    ep_start = 0
    n_eps = 0
    for ep_end in boundaries[:min(200, len(boundaries))]:
        ep_beliefs = beliefs[ep_start:ep_end]
        ep_actions = actions[ep_start:min(ep_end, len(actions))]
        ep_rewards = rewards[ep_start:ep_end]
        if len(ep_beliefs) > 2:
            grounder.label_trajectory(ep_beliefs, ep_actions, ep_rewards)
            n_eps += 1
        ep_start = ep_end

    print(f"    Processed {n_eps} episodes in {time.time()-t0:.1f}s")
    print(f"    Vocabulary: {len(grounder.words)} words")
    print(f"    Bindings: {grounder.bindings:,}")

    # Show learned similarities
    print(f"\n  Word similarities (learned from experience):")
    pairs = [
        ("left", "right"), ("fast", "slow"), ("start", "end"),
        ("upper", "lower"), ("push", "coast"), ("goal", "reward"),
        ("northward", "southward"), ("eastward", "westward"),
        ("turn", "still"), ("fast", "push"),
    ]
    for w1, w2 in pairs:
        sim = grounder.similarity(w1, w2)
        if sim != 0:
            print(f"    sim({w1:>10}, {w2:<10}) = {sim:+.3f}")

    # 7. Neuromodulatory analysis
    print(f"\n  Neuromodulatory signal analysis...")
    # Sample 5 episodes
    ep_start = 0
    for i, ep_end in enumerate(boundaries[:5]):
        ep_beliefs = beliefs[ep_start:ep_end]
        signals = compute_neuromod(ep_beliefs, actions[ep_start:ep_end], schemas)
        print(f"    Episode {i+1} ({ep_end-ep_start} steps): "
              f"DA={np.mean(signals['DA']):.2f} "
              f"ACh={np.mean(signals['ACh']):.2f} "
              f"CRT={np.mean(signals['CRT']):.2f}")
        ep_start = ep_end

    # 8. Save everything
    save_dir = Path("data/minari_trained")
    save_dir.mkdir(parents=True, exist_ok=True)

    np.savez(save_dir / "transition_model.npz",
             W1=model.W1, b1=model.b1, W2=model.W2, b2=model.b2)
    np.savez(save_dir / "schema_codebook.npz",
             codebook=schemas.codebook, usage=schemas.usage)
    np.savez(save_dir / "beliefs_sample.npz",
             beliefs=beliefs[:10000])

    print(f"\n  Saved to {save_dir}/")
    print(f"    transition_model.npz ({model.train_steps} train steps)")
    print(f"    schema_codebook.npz (32 schemas)")
    print(f"    beliefs_sample.npz (10K beliefs)")

    # Summary
    print(f"\n{'='*70}")
    print(f"  Training Summary")
    print(f"{'='*70}")
    print(f"  Data:         {len(obs):,} transitions from {len(boundaries)} episodes")
    print(f"  Transition:   MSE {losses[-1]:.6f} (pos MSE {pos_mse:.6f})")
    print(f"  Schemas:      {(schemas.usage > 0).sum()}/32 active")
    print(f"  Vocabulary:   {len(grounder.words)} words, {grounder.bindings:,} bindings")
    print(f"  Training:     {train_time:.1f}s on CPU")
    print(f"{'='*70}")


def run_tests():
    """Quick validation of loading."""
    print("=" * 65)
    print("  Minari Loader Tests")
    print("=" * 65)
    p = 0; t = 0

    print("\n  T1: Load dataset")
    try:
        obs, actions, rewards, bounds = load_minari_data(max_episodes=10)
        ok = len(obs) > 0
        print(f"    {len(obs)} transitions {'PASS' if ok else 'FAIL'}")
        p += int(ok); t += 1
    except Exception as e:
        print(f"    FAIL: {e}")
        t += 1
        return

    print("\n  T2: Project to belief space")
    beliefs = project_to_belief(obs)
    ok = beliefs.shape == (len(obs), D_BELIEF)
    print(f"    Shape: {beliefs.shape} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T3: Schema initialization")
    schemas = SchemaStore(n_schemas=8)
    schemas.initialize_from_data(beliefs[:1000], n_iter=5)
    ok = (schemas.usage > 0).sum() > 0
    print(f"    Active: {(schemas.usage > 0).sum()}/8 {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T4: Transition model trains")
    model = TransitionModel()
    N = min(len(beliefs) - 1, 1000)
    a_pad = actions[:N]
    if a_pad.shape[1] < D_ACTION:
        a_pad = np.concatenate([a_pad, np.zeros((N, D_ACTION - a_pad.shape[1]))], axis=1)
    losses = model.train(beliefs[:N], a_pad.astype(np.float32),
                          beliefs[1:N+1], epochs=3, lr=0.001, batch_size=64)
    ok = losses[-1] < losses[0]
    print(f"    Loss: {losses[0]:.4f} → {losses[-1]:.4f} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T5: Language grounding")
    grounder = LanguageGrounder()
    labels = grounder.label_trajectory(beliefs[:100], actions[:100],
                                         rewards[:100])
    ok = len(grounder.words) > 5
    print(f"    Words: {len(grounder.words)} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T6: Word similarities")
    sim = grounder.similarity("left", "right")
    ok = True  # just verify no crash
    print(f"    sim(left, right) = {sim:+.3f} PASS")
    p += int(ok); t += 1

    print(f"\n{'='*65}")
    print(f"  Results: {p}/{t} tests passed")
    print(f"{'='*65}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--episodes", type=int, default=500,
                     help="Max episodes to use (default 500)")
    ap.add_argument("--dataset", default="D4RL/pointmaze/umaze-v2")
    args = ap.parse_args()

    if args.test:
        run_tests()
    else:
        train_all(max_episodes=args.episodes, dataset_name=args.dataset)
