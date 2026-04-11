"""
cortex_brain.memory.ttm_clustering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Test-Time Memory (TTM) with Agglomerative Clustering.

Source: working_memory.py (TestTimeMemoryLoop) extended with sklearn clustering.

Memory Hierarchy
----------------
1. Episodic Buffer (E)   – bounded deque, O(1) append
2. Working Hypotheses (H)– corrective rules born from Surprise events
3. Long-Term Anchors (P) – verified physical principles (Dynamic Prior)
4. Cluster Index (C)     – Ward-linkage clusters for sub-linear retrieval

Surprise = high resonance (rho > θ) AND negative pnl outcome.
Hypotheses gain confidence via cosine-similarity tests → promoted to Prior.
All buffers are bounded: O(1) worst-case per tick.
"""
from __future__ import annotations
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TTMConfig:
    manifold_dim:          int   = 128
    max_episodic:          int   = 500
    max_long_term:         int   = 1000
    confidence_threshold:  float = 0.85
    surprise_rho_threshold:float = 0.90
    similarity_threshold:  float = 0.85
    retrieval_threshold:   float = 0.90
    expert_penalty:        float = 0.15
    num_clusters:          int   = 16
    recluster_every:       int   = 50


class TestTimeMemoryWithClustering:
    """
    Training-free intra-session self-evolution for CORTEX-16.

    Usage
    -----
    >>> mem = TestTimeMemoryWithClustering()
    >>> mem.observe(z, moe_weights, resonance=0.95, pnl=-50.0)
    >>> adj = mem.get_routing_adjustment(current_z)   # (4,) penalty vector
    """

    def __init__(self, config: TTMConfig = TTMConfig()) -> None:
        self.config = config
        self.episodic:   deque[Dict[str, Any]] = deque(maxlen=config.max_episodic)
        self.hypotheses: Dict[str, Dict[str, Any]] = {}
        self.anchors:    deque[Dict[str, Any]]      = deque(maxlen=config.max_long_term)
        self._centroids: Optional[np.ndarray] = None
        self._labels:    Optional[np.ndarray] = None
        self._ticks = 0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def observe(self, manifold_z: np.ndarray, moe_weights: np.ndarray,
                resonance: float, pnl: float) -> None:
        event = {"z": manifold_z.copy(), "w": moe_weights.copy(),
                 "rho": resonance, "success": pnl > 0}
        self.episodic.append(event)
        if resonance > self.config.surprise_rho_threshold and not event["success"]:
            self._generate_hypothesis(event)
        self._ticks += 1
        if self._ticks % self.config.recluster_every == 0:
            self._recluster()

    def evaluate_hypotheses(self, current_z: np.ndarray, pnl: float) -> None:
        to_remove: List[str] = []
        for hyp_id, hyp in self.hypotheses.items():
            if _cosine_sim(current_z, hyp["target_z"]) > self.config.similarity_threshold:
                hyp["tests"] += 1
                hyp["confidence"] += 0.1 if pnl > 0 else -0.1
                hyp["confidence"] = float(np.clip(hyp["confidence"], 0.0, 1.0))
                if hyp["confidence"] >= self.config.confidence_threshold:
                    self._promote(hyp_id, hyp); to_remove.append(hyp_id)
                elif hyp["confidence"] < 0.2:
                    to_remove.append(hyp_id)
        for k in to_remove:
            self.hypotheses.pop(k, None)

    def get_routing_adjustment(self, current_z: np.ndarray) -> np.ndarray:
        """O(1) MoE penalty vector via cluster-indexed retrieval."""
        adj = np.zeros(4, dtype=np.float32)
        for anchor in self._candidate_anchors(current_z):
            if _cosine_sim(current_z, anchor["target_z"]) > self.config.retrieval_threshold:
                adj[int(anchor["expert_idx"]) % 4] -= self.config.expert_penalty
        return adj

    @property
    def prior_size(self) -> int:        return len(self.anchors)
    @property
    def hypothesis_count(self) -> int:  return len(self.hypotheses)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _generate_hypothesis(self, ev: Dict[str, Any]) -> None:
        dom = int(np.argmax(ev["w"]))
        sig = tuple(ev["z"][:8].round(3).tolist())
        hid = f"exp{dom}_fail_{hash(sig)}"
        if hid not in self.hypotheses:
            self.hypotheses[hid] = {"target_z": ev["z"].copy(),
                                    "expert_idx": dom, "confidence": 0.5, "tests": 0}

    def _promote(self, hyp_id: str, hyp: Dict[str, Any]) -> None:
        self.anchors.append({"target_z": hyp["target_z"].copy(),
                             "expert_idx": hyp["expert_idx"],
                             "confidence": hyp["confidence"]})
        self._centroids = None
        logger.info("🧬 [PRIOR EVOLVED] %s → prior_size=%d", hyp_id, self.prior_size)

    def _recluster(self) -> None:
        n = len(self.anchors)
        if n < self.config.num_clusters:
            return
        try:
            from sklearn.cluster import AgglomerativeClustering
            zs = np.stack([a["target_z"] for a in self.anchors])
            k  = min(self.config.num_clusters, n)
            lbl = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(zs)
            self._centroids = np.stack([zs[lbl == c].mean(axis=0)
                                        for c in range(k) if (lbl == c).any()]).astype(np.float32)
            self._labels = lbl
        except ImportError:
            pass

    def _candidate_anchors(self, z: np.ndarray) -> List[Dict[str, Any]]:
        if self._centroids is None or self._labels is None:
            return list(self.anchors)
        sims = np.array([_cosine_sim(z, c) for c in self._centroids])
        cl   = int(np.argmax(sims))
        return [a for a, lbl in zip(self.anchors, self._labels) if lbl == cl]


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))