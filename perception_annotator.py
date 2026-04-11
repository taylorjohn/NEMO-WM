"""
CORTEX-PE Perception Annotator
================================

Auto-generates language annotations as eval scripts run — the glue layer
that populates LanguageMemoryDB without any extra user effort.

Two integration modes
---------------------
  1. Callback hook  — attach to any eval loop, fires on anomalous samples
  2. Batch annotate — post-hoc annotation of saved anomaly score arrays

Works with any CORTEX-PE domain:
  • bearing (PCA features, 6–12D → projected to 128-D)
  • cardiac (MelSpec features or raw StudentEncoder 128-D)
  • smap    (25-channel telemetry segments → encoded to 128-D)
  • recon   (frame latents, already 128-D from NPU)

Feature projection
------------------
The eval scripts produce features of varying dimensions (PCA is 6-D, raw
segments are 4096-D). A lightweight FeatureProjector lifts them to 128-D
so PerceptionLLM always receives a consistent-shaped embedding.

Usage — as callback during eval
---------------------------------
  ann = PerceptionAnnotator(domain="bearing")
  
  # Inside your eval loop:
  score = pca_score(features)
  ann.observe(features, anomaly_score=score, label="outer_race")

Usage — post-hoc batch from saved arrays
-----------------------------------------
  ann = PerceptionAnnotator(domain="smap")
  ann.annotate_batch(
      features=np.load("smap_features.npy"),   # [N, D]
      scores=np.load("smap_scores.npy"),        # [N]
      labels=["normal"]*50 + ["anomaly"]*20,
      threshold=0.60,
  )

Usage — decorator on existing eval function
--------------------------------------------
  @ann.wrap_eval("bearing")
  def run_bearing_eval(args):
      ...
      return {"auroc": 0.9993, "features": feats, "scores": scores}
"""

import sys
import time
import json
import hashlib
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Any
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent))
from perception_llm import (
    PerceptionLLM, LanguageMemoryDB, MockBackend, DOMAIN_CONTEXTS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Feature Projector — lifts arbitrary-dim features to 128-D
# ─────────────────────────────────────────────────────────────────────────────

class FeatureProjector:
    """
    Dimensionality adapter: maps D-dimensional feature vectors to 128-D.
    
    Strategy:
      D < 128  → pad with zeros, apply learned linear (PCA-style)
      D = 128  → passthrough (NPU StudentEncoder output)
      D > 128  → PCA reduction via online covariance estimate
    
    No PyTorch required — pure numpy.
    """

    def __init__(self, input_dim: int, output_dim: int = 128):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._W: Optional[np.ndarray] = None       # [input_dim, output_dim]
        self._fitted = False
        self._n_seen = 0
        self._mean = np.zeros(input_dim, dtype=np.float64)
        self._M2 = np.zeros((input_dim, input_dim), dtype=np.float64)

    def _init_random(self):
        """Random orthogonal projection — works without training data."""
        rng = np.random.default_rng(42)
        raw = rng.standard_normal((self.input_dim, self.output_dim))
        # QR decomposition for orthonormal columns
        if self.input_dim >= self.output_dim:
            Q, _ = np.linalg.qr(raw)
            self._W = Q[:, :self.output_dim].astype(np.float32)
        else:
            Q, _ = np.linalg.qr(raw.T)
            self._W = Q.T.astype(np.float32)

    def update(self, x: np.ndarray):
        """Online Welford update for covariance — call with each new sample."""
        x = x.astype(np.float64).flatten()[:self.input_dim]
        self._n_seen += 1
        delta = x - self._mean
        self._mean += delta / self._n_seen
        delta2 = x - self._mean
        self._M2 += np.outer(delta, delta2)

        if self._n_seen >= max(32, self.output_dim * 2) and not self._fitted:
            cov = self._M2 / (self._n_seen - 1)
            eigvals, eigvecs = np.linalg.eigh(cov)
            idx = np.argsort(eigvals)[::-1][:self.output_dim]
            self._W = eigvecs[:, idx].astype(np.float32)
            self._fitted = True

    def project(self, x: np.ndarray) -> np.ndarray:
        """Project x to 128-D. x can be any shape; will be flattened."""
        x = x.astype(np.float32).flatten()

        if self.input_dim == self.output_dim:
            z = x[:self.output_dim]
            norm = np.linalg.norm(z)
            return z / (norm + 1e-8)

        # Trim or pad
        if len(x) < self.input_dim:
            x = np.pad(x, (0, self.input_dim - len(x)))
        else:
            x = x[:self.input_dim]

        if self._W is None:
            self._init_random()

        z = (x - self._mean.astype(np.float32)) @ self._W   # [output_dim]

        # L2-normalise to unit sphere (consistent with NPU StudentEncoder)
        norm = np.linalg.norm(z)
        return z / (norm + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Observation buffer — batches DB writes for speed
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Observation:
    z128: np.ndarray           # 128-D projected latent
    raw_features: np.ndarray   # original features (any dim)
    anomaly_score: float
    label: str                 # "normal", "outer_race", "inner_race", etc.
    timestamp: float = field(default_factory=time.time)
    answered: bool = False
    answer: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Main Annotator
# ─────────────────────────────────────────────────────────────────────────────

class PerceptionAnnotator:
    """
    Hooks into CORTEX-PE eval scripts and auto-generates language annotations.
    
    Completely non-invasive: add two lines to any eval loop, remove two lines
    to go back to the original behaviour.
    """

    # Per-domain alert thresholds (mirrors perception_llm.py)
    THRESHOLDS = {
        "bearing": 0.65,
        "cardiac": 0.70,
        "smap":    0.60,
        "recon":   0.75,
    }

    # How many samples to annotate per eval run (LLM calls are slow)
    MAX_ANNOTATIONS_PER_RUN = 20

    def __init__(
        self,
        domain: str,
        db_path: str = "language_memory.db",
        input_dim: Optional[int] = None,   # auto-detected on first observe()
        prefer_local: bool = True,
        verbose: bool = False,
        dry_run: bool = False,             # log but don't call LLM
        annotation_rate: float = 0.1,     # annotate this fraction of anomalous samples
        seed: int = 42,
    ):
        assert domain in DOMAIN_CONTEXTS, f"Unknown domain: {domain}"
        self.domain = domain
        self.verbose = verbose
        self.dry_run = dry_run
        self.annotation_rate = annotation_rate
        self._rng = np.random.default_rng(seed)

        self.db = LanguageMemoryDB(db_path)
        self.plm = PerceptionLLM(
            domain=domain,
            db_path=db_path,
            prefer_local=prefer_local,
            verbose=False,
        )
        if dry_run:
            self.plm.llm = MockBackend()
            self.plm._backend_name = "mock/dry_run"

        self._projector: Optional[FeatureProjector] = None
        self._input_dim: Optional[int] = input_dim

        # Counters
        self._n_observed = 0
        self._n_above_threshold = 0
        self._n_annotated = 0
        self._n_skipped = 0
        self._session_id = f"annotate_{domain}_{int(time.time())}"

        # Dedup: don't annotate near-identical observations
        self._seen_hashes: set[str] = set()

        if verbose:
            backend = self.plm._backend_name
            print(f"[PerceptionAnnotator/{domain}] init | "
                  f"backend={backend} | threshold={self.THRESHOLDS[domain]}")

    def _ensure_projector(self, raw_dim: int):
        if self._projector is None:
            self._input_dim = raw_dim
            self._projector = FeatureProjector(raw_dim, output_dim=128)
            if self.verbose:
                print(f"[PerceptionAnnotator/{self.domain}] "
                      f"FeatureProjector: {raw_dim}→128")

    def _should_annotate(self, score: float, z: np.ndarray) -> bool:
        """Decide whether this observation warrants an LLM call."""
        threshold = self.THRESHOLDS[self.domain]

        # Must be above threshold
        if score < threshold:
            return False

        # Rate-limit annotations per session
        if self._n_annotated >= self.MAX_ANNOTATIONS_PER_RUN:
            return False

        # Stochastic subsampling
        if self._rng.random() > self.annotation_rate:
            return False

        # Dedup: skip if we've seen a very similar embedding recently
        z_hash = hashlib.md5(np.round(z, 2).tobytes()).hexdigest()[:8]
        if z_hash in self._seen_hashes:
            self._n_skipped += 1
            return False
        self._seen_hashes.add(z_hash)

        return True

    def _question_for_label(self, label: str, score: float) -> str:
        """Generate a targeted question based on the fault label."""
        label_lc = label.lower()
        templates = {
            # Bearing
            "outer": f"Score {score:.3f}: Describe the outer race fault characteristics in this signal.",
            "inner": f"Score {score:.3f}: What inner race fault pattern does this embedding represent?",
            "ball":  f"Score {score:.3f}: Is this a ball fault signature or healthy baseline?",
            "normal": f"Score {score:.3f}: This is labelled normal. Confirm or identify any subtle anomaly.",
            # Cardiac
            "murmur": f"Score {score:.3f}: Characterise the murmur type from this cardiac embedding.",
            "artifact": f"Score {score:.3f}: Is this a genuine cardiac anomaly or recording artifact?",
            # SMAP
            "a": f"Score {score:.3f}: Mode switch or point anomaly in channel A telemetry?",
            "b": f"Score {score:.3f}: Characterise this anomaly in channel B telemetry.",
            # RECON
            "obstacle": f"Score {score:.3f}: What obstacle or terrain change caused this scene shift?",
        }
        for key, tmpl in templates.items():
            if key in label_lc:
                return tmpl
        return (f"Score {score:.3f} on label '{label}': "
                f"Describe what the perception engine is observing.")

    def observe(
        self,
        features: np.ndarray,
        anomaly_score: float,
        label: str = "unknown",
        extra_context: str = "",
    ) -> Optional[str]:
        """
        Main API: call once per sample in your eval loop.
        
        Returns the LLM-generated annotation if one was created, else None.
        Does not block if annotation is skipped (> 95% of calls return None fast).

        Args:
            features:      raw features of any dimension (or 128-D NPU latent)
            anomaly_score: pre-computed anomaly score from PCA/SubspaceAD
            label:         ground-truth or predicted label (for question generation)
            extra_context: additional context string injected into the prompt

        Returns:
            annotation string if generated, None otherwise
        """
        self._n_observed += 1

        # Ensure projector
        raw = features.astype(np.float32).flatten()
        self._ensure_projector(raw.shape[0])
        self._projector.update(raw)
        z = self._projector.project(raw)

        threshold = self.THRESHOLDS[self.domain]
        if anomaly_score >= threshold:
            self._n_above_threshold += 1

        if not self._should_annotate(anomaly_score, z):
            return None

        # Generate annotation
        question = self._question_for_label(label, anomaly_score)
        answer = self.plm.ask(
            z,
            question=question,
            anomaly_score=anomaly_score,
            save=True,
            extra_context=extra_context,
        )
        self._n_annotated += 1

        if self.verbose:
            print(f"  [annotate/{self.domain}] score={anomaly_score:.3f} | "
                  f"label={label} | {answer[:80]}...")

        return answer

    def annotate_batch(
        self,
        features: np.ndarray,          # [N, D]
        scores: np.ndarray,            # [N]
        labels: Optional[list[str]] = None,
        threshold: Optional[float] = None,
        n_top: int = 20,
        question_override: Optional[str] = None,
    ) -> list[dict]:
        """
        Post-hoc annotation of saved eval outputs.
        Picks the n_top highest-scoring samples and annotates them.
        
        Args:
            features:          [N, D] feature array
            scores:            [N] anomaly scores
            labels:            optional list of N label strings
            threshold:         min score to annotate (default: domain threshold)
            n_top:             max annotations to generate
            question_override: use this question for all samples

        Returns:
            list of {"idx": int, "score": float, "label": str, "answer": str}
        """
        if threshold is None:
            threshold = self.THRESHOLDS[self.domain]
        if labels is None:
            labels = ["unknown"] * len(scores)

        # Select candidates above threshold, sorted by score descending
        above = [(i, scores[i]) for i in range(len(scores)) if scores[i] >= threshold]
        above.sort(key=lambda x: -x[1])
        candidates = above[:n_top]

        results = []
        for idx, score in candidates:
            raw = features[idx].astype(np.float32).flatten()
            self._ensure_projector(raw.shape[0])
            self._projector.update(raw)
            z = self._projector.project(raw)

            label = labels[idx]
            question = question_override or self._question_for_label(label, score)

            answer = self.plm.ask(
                z, question=question, anomaly_score=score, save=True
            )
            self._n_annotated += 1

            results.append({
                "idx": int(idx),
                "score": float(score),
                "label": label,
                "answer": answer,
            })
            if self.verbose:
                print(f"  [{idx:4d}] score={score:.3f} | {label:12s} | {answer[:70]}...")

        return results

    def summary(self) -> dict:
        """Return session statistics."""
        return {
            "session_id": self._session_id,
            "domain": self.domain,
            "n_observed": self._n_observed,
            "n_above_threshold": self._n_above_threshold,
            "n_annotated": self._n_annotated,
            "n_skipped_dedup": self._n_skipped,
            "db_stats": self.db.stats(),
        }

    def wrap_eval(self, domain_hint: str = "") -> Callable:
        """
        Decorator: auto-annotates top anomalies from eval function return value.
        
        The decorated function must return a dict with keys:
            "features": np.ndarray [N, D]
            "scores":   np.ndarray [N]
            "labels":   list[str]  (optional)
            "auroc":    float      (optional, logged)

        Example:
            @ann.wrap_eval()
            def run(args):
                ...
                return {"features": feats, "scores": scores, "labels": lbls}
        """
        def decorator(fn: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                result = fn(*args, **kwargs)
                if isinstance(result, dict):
                    feats  = result.get("features")
                    scores = result.get("scores")
                    labels = result.get("labels")
                    auroc  = result.get("auroc")
                    if feats is not None and scores is not None:
                        print(f"\n[PerceptionAnnotator] Auto-annotating top anomalies "
                              f"from {fn.__name__}...")
                        annotations = self.annotate_batch(
                            feats, scores, labels, n_top=10
                        )
                        result["annotations"] = annotations
                        if auroc:
                            print(f"  AUROC={auroc:.4f} | "
                                  f"{len(annotations)} annotations saved to DB")
                return result
            return wrapper
        return decorator

    def close(self):
        self.db.close()


# ─────────────────────────────────────────────────────────────────────────────
# CLI — batch annotate from saved numpy arrays
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="CORTEX-PE Perception Annotator — batch language annotation"
    )
    parser.add_argument("--domain", choices=list(DOMAIN_CONTEXTS), required=True)
    parser.add_argument("--features", type=str, required=True,
                        help=".npy file: [N, D] feature array")
    parser.add_argument("--scores", type=str, required=True,
                        help=".npy file: [N] anomaly scores")
    parser.add_argument("--labels", type=str, default=None,
                        help=".json file: list of N label strings")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Min anomaly score to annotate")
    parser.add_argument("--n-top", type=int, default=20,
                        help="Max annotations to generate")
    parser.add_argument("--db", default="language_memory.db")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use mock backend (no LLM API calls)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    features = np.load(args.features)
    scores   = np.load(args.scores)
    labels   = None
    if args.labels:
        with open(args.labels) as f:
            labels = json.load(f)

    ann = PerceptionAnnotator(
        domain=args.domain, db_path=args.db,
        dry_run=args.dry_run, verbose=args.verbose
    )

    print(f"\n[PerceptionAnnotator/{args.domain}] "
          f"features={features.shape} scores={scores.shape}")

    results = ann.annotate_batch(
        features=features, scores=scores, labels=labels,
        threshold=args.threshold, n_top=args.n_top,
    )

    print(f"\n── Annotations ({len(results)}) ────────────────────────")
    for r in results:
        print(f"  [{r['idx']:4d}] score={r['score']:.3f} | {r['label']:12s}")
        print(f"         {r['answer'][:120]}")
    print()

    summary = ann.summary()
    print(f"── Summary ──────────────────────────────────────────")
    print(json.dumps({k: v for k, v in summary.items() if k != "db_stats"}, indent=2))
    ann.close()


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, tempfile, json

    if "--" in sys.argv or len(sys.argv) == 1:
        print("=== PerceptionAnnotator Smoke Test ===\n")

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            tmp_db = f.name

        rng = np.random.default_rng(0)

        # --- Test 1: observe() hook
        print("Test 1: observe() — simulating bearing eval loop")
        ann = PerceptionAnnotator("bearing", db_path=tmp_db, dry_run=True,
                                  annotation_rate=1.0, verbose=True)
        ann.MAX_ANNOTATIONS_PER_RUN = 5

        fault_labels = ["ball", "ball", "inner", "outer", "outer", "inner", "outer"]
        for i, label in enumerate(fault_labels):
            feats = rng.standard_normal(12).astype(np.float32)   # 12-D PCA features
            score = 0.3 + 0.6 * (i / len(fault_labels))
            ann.observe(feats, anomaly_score=score, label=label)

        print(f"  Summary: {json.dumps(ann.summary(), default=str)[:300]}")

        # --- Test 2: annotate_batch()
        print("\nTest 2: annotate_batch() — SMAP telemetry")
        ann2 = PerceptionAnnotator("smap", db_path=tmp_db, dry_run=True,
                                   annotation_rate=1.0, verbose=True)
        feats_batch = rng.standard_normal((30, 25)).astype(np.float32)  # 25-D SMAP
        scores_batch = rng.uniform(0, 1, 30).astype(np.float32)
        scores_batch[10:15] = 0.85  # inject high-anomaly samples
        labels_batch = ["nominal"] * 25 + ["mode_switch"] * 5
        results = ann2.annotate_batch(feats_batch, scores_batch, labels_batch, n_top=5)
        print(f"  Generated {len(results)} annotations")

        # --- Test 3: FeatureProjector dimensions
        print("\nTest 3: FeatureProjector — various input dims")
        for dim in [6, 12, 25, 128, 512, 4096]:
            proj = FeatureProjector(dim, output_dim=128)
            x = rng.standard_normal(dim).astype(np.float32)
            for _ in range(50):           # warm up online covariance
                proj.update(rng.standard_normal(dim).astype(np.float32))
            z = proj.project(x)
            assert z.shape == (128,), f"Expected (128,), got {z.shape}"
            assert abs(np.linalg.norm(z) - 1.0) < 0.01, "Output not unit-norm"
            print(f"  dim={dim:5d} → 128 ✅  norm={np.linalg.norm(z):.4f}")

        ann.close()
        ann2.close()
        os.unlink(tmp_db)
        print("\n✅ All smoke tests passed")
    else:
        main()
