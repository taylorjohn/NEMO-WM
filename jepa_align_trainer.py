"""
CORTEX-PE JEPA Alignment Trainer
==================================

Trains the LatentProjector to align the NPU embedding space with the
LLM's concept space. Run offline after market close / during downtime.

This is the LLM-JEPA training objective applied to perception:
  L = InfoNCE(Projector(z_npu), LLM_embed(text_description))

Since we cannot access the LLM's internal embeddings directly (black-box API),
we use a self-supervised proxy called Paired Contrastive Alignment (PCA-align):

  Positive pair: (z_i, z_j) where description_i and description_j
                  are semantically similar (cosine sim of TF-IDF > threshold)
  Negative pair: (z_i, z_k) where descriptions are dissimilar

This teaches the projector to cluster perceptually similar states together
without needing the LLM's actual embedding weights.

Two training modes
------------------
  1. Self-contrastive (default): pairs within same anomaly band
     High anomaly latents cluster together, low anomaly apart.
     Fast. Works with 50+ samples.

  2. Text-similarity-guided (--text-guided): pairs via TF-IDF similarity
     of the LLM-generated descriptions. More precise. Needs 200+ samples.
     Requires: pip install scikit-learn

Usage
-----
  python jepa_align_trainer.py --domain bearing
  python jepa_align_trainer.py --domain all --epochs 20
  python jepa_align_trainer.py --domain cardiac --text-guided --batch 64
  python jepa_align_trainer.py --eval-only   # just run embedding eval
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    print("ERROR: PyTorch required. pip install torch")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
from perception_llm import LatentProjector, LanguageMemoryDB, DOMAIN_CONTEXTS


# ─────────────────────────────────────────────────────────────────────────────
# Text similarity (optional TF-IDF, falls back to keyword overlap)
# ─────────────────────────────────────────────────────────────────────────────

def text_similarity_matrix(texts: list[str]) -> np.ndarray:
    """
    Compute pairwise text similarity. Uses TF-IDF if sklearn available,
    otherwise keyword overlap.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        tfidf = TfidfVectorizer(max_features=500, stop_words="english")
        vecs = tfidf.fit_transform(texts)
        return cosine_similarity(vecs)
    except ImportError:
        # Fallback: Jaccard on word sets
        N = len(texts)
        mat = np.eye(N, dtype=np.float32)
        word_sets = [set(t.lower().split()) for t in texts]
        for i in range(N):
            for j in range(i + 1, N):
                inter = len(word_sets[i] & word_sets[j])
                union = len(word_sets[i] | word_sets[j])
                sim = inter / union if union > 0 else 0.0
                mat[i, j] = mat[j, i] = sim
        return mat


# ─────────────────────────────────────────────────────────────────────────────
# Contrastive loss variants
# ─────────────────────────────────────────────────────────────────────────────

def infonce_loss(
    anchors: torch.Tensor,      # [B, D] — projected latents
    positives: torch.Tensor,    # [B, D] — matched descriptions / similar latents
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Standard InfoNCE (NT-Xent) contrastive loss.
    Anchors should predict their corresponding positive,
    treating all other samples in the batch as negatives.
    """
    a = F.normalize(anchors, dim=-1)
    p = F.normalize(positives, dim=-1)
    logits = (a @ p.T) / temperature        # [B, B]
    labels = torch.arange(len(a), device=a.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_p = F.cross_entropy(logits.T, labels)
    return (loss_a + loss_p) / 2.0


def anomaly_band_loss(
    proj: torch.Tensor,        # [B, D]
    scores: torch.Tensor,      # [B] — anomaly scores 0..1
    temperature: float = 0.1,
    band_width: float = 0.15,
) -> torch.Tensor:
    """
    Pull together latents with similar anomaly scores,
    push apart latents in different anomaly bands.
    Simple proxy for text-similarity when text embeddings aren't available.
    """
    p_norm = F.normalize(proj, dim=-1)                  # [B, D]
    sim = p_norm @ p_norm.T                             # [B, B]

    # Soft positive mask: similar anomaly score → should cluster
    score_diff = (scores.unsqueeze(0) - scores.unsqueeze(1)).abs()  # [B, B]
    pos_mask = (score_diff < band_width).float()
    pos_mask.fill_diagonal_(0)

    # Negative mask: different anomaly band
    neg_mask = (score_diff > band_width * 2).float()

    # Pull positives together, push negatives apart
    pos_loss = -(sim * pos_mask).sum() / (pos_mask.sum() + 1e-8)
    neg_loss = F.relu(sim * neg_mask - 0.1).sum() / (neg_mask.sum() + 1e-8)
    return pos_loss + neg_loss


def vicreg_loss(
    proj: torch.Tensor,
    sim_coeff: float = 25.0,
    var_coeff: float = 25.0,
    cov_coeff: float = 1.0,
) -> torch.Tensor:
    """
    VICReg-style collapse prevention.
    Ensures the projected space doesn't collapse to constant output.
    Combined with contrastive loss for stable training.
    """
    # Variance: each dimension should have std > 1
    std = torch.sqrt(proj.var(dim=0) + 1e-4)
    var_loss = F.relu(1.0 - std).mean()

    # Covariance: off-diagonal covariance should be zero
    proj_c = proj - proj.mean(dim=0)
    cov = (proj_c.T @ proj_c) / (proj.shape[0] - 1)
    cov_loss = (cov ** 2).fill_diagonal_(0.0).sum() / proj.shape[1]

    return var_coeff * var_loss + cov_coeff * cov_loss


# ─────────────────────────────────────────────────────────────────────────────
# Dataset from LanguageMemoryDB
# ─────────────────────────────────────────────────────────────────────────────

class PerceptionDataset:
    """Loads (latent, text, anomaly_score) triples from LanguageMemoryDB."""

    def __init__(self, db: LanguageMemoryDB, domain: str):
        self.domain = domain
        rows = db._conn.execute("""
            SELECT latent_blob, answer, anomaly_score
            FROM descriptions WHERE domain=?
            ORDER BY timestamp
        """, (domain,)).fetchall()

        self.latents = []
        self.texts = []
        self.scores = []

        for row in rows:
            emb = np.frombuffer(row[0], dtype=np.float32).copy()
            if emb.shape == (128,):
                self.latents.append(emb)
                self.texts.append(row[1] or "")
                self.scores.append(float(row[2] or 0.0))

        self.N = len(self.latents)

    def to_tensors(self) -> tuple[torch.Tensor, np.ndarray, list[str]]:
        z = torch.from_numpy(np.stack(self.latents))        # [N, 128]
        scores = np.array(self.scores)
        return z, scores, self.texts

    def random_batch(self, batch_size: int) -> tuple[torch.Tensor, np.ndarray, list[str]]:
        idx = np.random.choice(self.N, min(batch_size, self.N), replace=False)
        z = torch.from_numpy(np.stack([self.latents[i] for i in idx]))
        scores = np.array([self.scores[i] for i in idx])
        texts = [self.texts[i] for i in idx]
        return z, scores, texts


# ─────────────────────────────────────────────────────────────────────────────
# Embedding quality evaluator
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_alignment(
    projector: LatentProjector,
    dataset: PerceptionDataset,
) -> dict:
    """
    Measure how well the projected latent space captures anomaly structure.
    Metrics:
      - alignment_gap:  avg(sim_same_band) - avg(sim_diff_band)
                        Higher = better. Target > 0.2
      - intra_cluster_std: std of embeddings in same anomaly band
                           Lower = tighter clusters
      - collapse_check: mean std per dimension. < 0.01 = collapsed
    """
    if dataset.N < 8:
        return {"error": "Not enough data for evaluation (need ≥8 samples)"}

    projector.eval()
    z, scores, _ = dataset.to_tensors()

    with torch.no_grad():
        proj = projector(z)[:, 0, :]          # [N, llm_dim]
        proj_norm = F.normalize(proj, dim=-1)  # [N, llm_dim]

    sim = (proj_norm @ proj_norm.T).numpy()    # [N, N]
    scores_np = scores

    # Same-band vs different-band similarity
    score_diff = np.abs(scores_np[:, None] - scores_np[None, :])
    same_band = sim[score_diff < 0.15]
    diff_band = sim[score_diff > 0.30]

    alignment_gap = float(same_band.mean() - diff_band.mean()) \
        if len(same_band) > 0 and len(diff_band) > 0 else 0.0

    collapse_check = float(proj.std(dim=0).mean().item())

    return {
        "n_samples": dataset.N,
        "alignment_gap": round(alignment_gap, 4),
        "same_band_sim": round(float(same_band.mean()), 4) if len(same_band) > 0 else 0.0,
        "diff_band_sim": round(float(diff_band.mean()), 4) if len(diff_band) > 0 else 0.0,
        "collapse_check": round(collapse_check, 4),
        "status": (
            "GOOD" if alignment_gap > 0.2 and collapse_check > 0.05 else
            "COLLAPSED" if collapse_check < 0.01 else
            "TRAINING"
        )
    }


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class JEPAAlignTrainer:
    """
    Trains LatentProjector using accumulated (latent, text) pairs.

    The training loop:
      for each batch:
        proj = Projector(z_batch)          # [B, n_tokens, llm_dim]
        use proj[:, 0, :] as anchor
        
        if text_guided:
          sim_matrix = TF-IDF(texts)
          positives  = proj[top-1-most-similar-text]
          loss = InfoNCE(anchor, positive)
        else:
          loss = anomaly_band_loss(anchor, scores)
        
        loss += VICReg(anchor)             # collapse prevention always on
        loss.backward()
    """

    def __init__(
        self,
        db: LanguageMemoryDB,
        domain: str,
        llm_dim: int = 896,
        n_tokens: int = 4,
        lr: float = 3e-4,
        weight_decay: float = 1e-5,
        temperature: float = 0.07,
        text_guided: bool = False,
        checkpoint_dir: str = "checkpoints/projector",
    ):
        self.domain = domain
        self.text_guided = text_guided
        self.temperature = temperature
        self.ckpt_dir = Path(checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.dataset = PerceptionDataset(db, domain)
        print(f"[JEPAAlignTrainer/{domain}] Dataset: {self.dataset.N} samples")

        self.projector = LatentProjector(
            latent_dim=128, llm_dim=llm_dim, n_tokens=n_tokens
        )
        self.optimizer = torch.optim.AdamW(
            self.projector.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=lr * 0.1
        )

        # Load existing checkpoint if available
        latest = self._find_latest_checkpoint()
        if latest:
            self.projector.load_state_dict(torch.load(latest))
            print(f"[JEPAAlignTrainer/{domain}] Resumed from {latest}")

    def _find_latest_checkpoint(self) -> Optional[str]:
        checkpoints = sorted(self.ckpt_dir.glob(f"{self.domain}_projector_*.pt"))
        return str(checkpoints[-1]) if checkpoints else None

    def _text_guided_loss(
        self, proj: torch.Tensor, texts: list[str]
    ) -> torch.Tensor:
        """InfoNCE with positive pairs from text similarity."""
        sim_mat = text_similarity_matrix(texts)          # [B, B]
        B = proj.shape[0]

        # For each anchor, the positive is the most textually similar other sample
        np.fill_diagonal(sim_mat, -1)
        pos_idx = sim_mat.argmax(axis=1)                 # [B]

        anchors = proj[:, 0, :]                          # [B, llm_dim]
        positives = proj[pos_idx, 0, :]                  # [B, llm_dim]
        return infonce_loss(anchors, positives, self.temperature)

    def _anomaly_band_loss(
        self, proj: torch.Tensor, scores: np.ndarray
    ) -> torch.Tensor:
        scores_t = torch.from_numpy(scores).float()
        return anomaly_band_loss(proj[:, 0, :], scores_t)

    def train_epoch(self, batch_size: int = 128) -> dict:
        if self.dataset.N < 16:
            return {"error": "need ≥16 samples", "loss": 0.0}

        self.projector.train()
        n_batches = max(1, self.dataset.N // batch_size)
        losses = {"total": [], "contrastive": [], "vicreg": []}

        for _ in range(n_batches):
            z, scores, texts = self.dataset.random_batch(batch_size)

            proj = self.projector(z)                     # [B, n_tokens, llm_dim]

            if self.text_guided and all(t for t in texts):
                cont_loss = self._text_guided_loss(proj, texts)
            else:
                cont_loss = self._anomaly_band_loss(proj, scores)

            vic_loss = vicreg_loss(proj[:, 0, :])
            total = cont_loss + 0.1 * vic_loss

            self.optimizer.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(self.projector.parameters(), 1.0)
            self.optimizer.step()

            losses["total"].append(total.item())
            losses["contrastive"].append(cont_loss.item())
            losses["vicreg"].append(vic_loss.item())

        self.scheduler.step()

        return {
            "loss":        round(float(np.mean(losses["total"])), 5),
            "contrastive": round(float(np.mean(losses["contrastive"])), 5),
            "vicreg":      round(float(np.mean(losses["vicreg"])), 5),
            "n_batches":   n_batches,
        }

    def run(self, epochs: int = 30, batch_size: int = 128,
            eval_every: int = 5) -> str:
        print(f"\n[JEPAAlignTrainer/{self.domain}] Training {epochs} epochs | "
              f"mode={'text-guided' if self.text_guided else 'anomaly-band'}")

        history = []
        for epoch in range(1, epochs + 1):
            result = self.train_epoch(batch_size)
            history.append(result)

            line = (f"  epoch {epoch:3d}/{epochs} | "
                    f"loss={result['loss']:.5f} | "
                    f"contrastive={result['contrastive']:.5f} | "
                    f"vicreg={result['vicreg']:.5f}")
            print(line)

            if epoch % eval_every == 0:
                eval_result = evaluate_alignment(self.projector, self.dataset)
                print(f"  ── eval: gap={eval_result.get('alignment_gap', 'n/a')} | "
                      f"collapse={eval_result.get('collapse_check', 'n/a')} | "
                      f"status={eval_result.get('status', 'n/a')}")

        ckpt = self.ckpt_dir / f"{self.domain}_projector_{int(time.time())}.pt"
        torch.save(self.projector.state_dict(), str(ckpt))

        # Save a "latest" symlink / copy for easy loading
        latest = self.ckpt_dir / f"{self.domain}_projector_latest.pt"
        torch.save(self.projector.state_dict(), str(latest))

        print(f"\n[JEPAAlignTrainer/{self.domain}] Saved → {ckpt}")
        return str(ckpt)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CORTEX-PE JEPA Alignment Trainer")
    parser.add_argument("--domain", default="bearing",
                        help=f"Domain to train: {list(DOMAIN_CONTEXTS.keys())} or 'all'")
    parser.add_argument("--db", default="language_memory.db")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--llm-dim", type=int, default=896,
                        help="LLM embedding dimension (Qwen2.5-0.5B=896, 1.5B=1536)")
    parser.add_argument("--n-tokens", type=int, default=4,
                        help="Number of perception tokens to project")
    parser.add_argument("--text-guided", action="store_true",
                        help="Use TF-IDF text similarity for pairing (needs sklearn)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, just evaluate current checkpoint")
    parser.add_argument("--checkpoint-dir", default="checkpoints/projector")
    args = parser.parse_args()

    db = LanguageMemoryDB(args.db)

    domains = list(DOMAIN_CONTEXTS.keys()) if args.domain == "all" else [args.domain]

    for domain in domains:
        trainer = JEPAAlignTrainer(
            db=db,
            domain=domain,
            llm_dim=args.llm_dim,
            n_tokens=args.n_tokens,
            lr=args.lr,
            text_guided=args.text_guided,
            checkpoint_dir=args.checkpoint_dir,
        )

        if args.eval_only:
            result = evaluate_alignment(trainer.projector, trainer.dataset)
            print(f"\n[{domain}] Eval: {json.dumps(result, indent=2)}")
        else:
            if trainer.dataset.N < 16:
                print(f"[{domain}] Skipping — only {trainer.dataset.N} samples "
                      f"(need ≥16). Run the REPL first to accumulate observations.")
                continue
            trainer.run(epochs=args.epochs, batch_size=args.batch)

    db.close()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Smoke test — synthetic data
        import tempfile, os
        print("=== JEPAAlignTrainer Smoke Test ===")
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            tmp_db = f.name

        db = LanguageMemoryDB(tmp_db)
        rng = np.random.default_rng(7)

        # Seed DB with 40 synthetic observations
        for i in range(40):
            z = rng.standard_normal(128).astype(np.float32)
            score = float(rng.uniform(0, 1))
            db.record(
                domain="bearing", latent=z,
                answer=("Outer race fault detected" if score > 0.6 else "Normal operation"),
                anomaly_score=score, model_used="mock",
            )
        trainer = JEPAAlignTrainer(db, "bearing", llm_dim=64, n_tokens=2)
        result = trainer.run(epochs=5, batch_size=20, eval_every=2)
        print(f"\nCheckpoint: {result}")

        eval_r = evaluate_alignment(trainer.projector, trainer.dataset)
        print(f"Final eval: {json.dumps(eval_r, indent=2)}")

        db.close()
        os.unlink(tmp_db)
        import shutil
        shutil.rmtree("checkpoints", ignore_errors=True)
        print("\n✅ Smoke test passed")
    else:
        main()
