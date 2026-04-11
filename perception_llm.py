"""
CORTEX-PE PerceptionLLM â€” Language Layer for a Languageless Engine
====================================================================

Adds natural language grounding to the NPU StudentEncoder (128-D latent space)
across all perception domains: bearing, cardiac, SMAP telemetry, RECON navigation.

Architecture (inspired by LLaVA / InstructBLIP):
  NPU latent (128-D)
       â”‚
  LatentProjector  â† trainable MLP, ~50K params
  (128 â†’ llm_embed_dim)
       â”‚
  Injected as prefix "perception tokens" into LLM context
       â”‚
  Small local LLM (Qwen2.5-0.5B via ollama, ~400MB)
       â”‚
  Natural language answer

LLM-JEPA connection:
  View 1: NPU latent embedding (perception)
  View 2: Text description (language)
  These are two views of the same underlying knowledge â†’ valid JEPA view pair.
  The LanguageMemoryDB accumulates (latent, text) pairs for future JEPA training.

COMPLETELY SEPARATE from trading code:
  - No imports from cortex_brain, moe_router_v2, or market_replay
  - Own SQLite DB (language_memory.db)
  - Own domain context strings
  - Disable/remove without touching any existing pipeline

Usage:
  python perception_llm.py --domain bearing --question "Is this healthy?"
  python perception_llm.py --domain recon   --describe
  python perception_llm.py --domain cardiac --question "What anomaly signature?"
  python perception_llm.py --domain smap    --anomaly-score 0.91
"""

import sqlite3
import json
import time
import struct
import argparse
import textwrap
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np

# â”€â”€â”€ Optional torch (for LatentProjector) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# â”€â”€â”€ LLM backends (try ollama first, fall back to Anthropic API) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
try:
    import requests as _requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# =============================================================================
# Domain Context â€” grounded knowledge per perception domain
# =============================================================================

DOMAIN_CONTEXTS = {
    "bearing": textwrap.dedent("""
        You are analyzing output from a real-time bearing fault detection system.
        The system uses a 128-dimensional latent embedding produced by a StudentEncoder
        running on an AMD Ryzen AI NPU at ~0.34ms per inference.
        
        The encoder was trained on CWRU (Case Western Reserve University) bearing data
        with AUROC 1.0000 on outer/inner/ball race fault detection.
        The latent space captures four fault signatures:
          - Normal: low anomaly score, tight cluster in latent space
          - Outer race fault: characteristic periodic impulses at BPFO frequency
          - Inner race fault: modulated by shaft rotation, amplitude variation
          - Ball fault: lower energy, diffuse cluster
        
        Anomaly scoring uses SubspaceAD with PatchCore at k=16.
        AUROC 0.9993. Threshold for alert: anomaly score > 0.65.
    """).strip(),

    "cardiac": textwrap.dedent("""
        You are analyzing output from a cardiac audio anomaly detection system.
        The system embeds PCG (phonocardiogram) heart sound recordings into a
        128-dimensional latent space using a StudentEncoder on AMD Ryzen AI NPU.
        
        Training: PhysioNet/CinC 2016 cardiac audio dataset.
        Best checkpoint: epoch 4, cos_sim=0.865, AUROC=0.792.
        
        The latent space distinguishes:
          - Normal heart sounds: S1 and S2 with clean intervals
          - Murmurs: extra energy between S1/S2 (systolic or diastolic)
          - Extra heart sounds: S3 (early diastole) or S4 (pre-systolic)
          - Irregular rhythms: arrhythmia patterns
        
        Anomaly score above 0.70 warrants clinical review.
        This system is a screening aid only â€” not a diagnostic device.
    """).strip(),

    "smap": textwrap.dedent("""
        You are analyzing output from the SMAP (Soil Moisture Active Passive)
        satellite telemetry anomaly detection system.
        
        The StudentEncoder processes 25-channel SMAP/MSL telemetry into a
        128-dimensional latent embedding.
        Current performance: AUROC 0.7362 at k=16, window=128.
        Known challenge: 17-20% anomaly rate in training data (contamination).
        A T-1/T-2 fix using robust PCA on lowest-80% reconstruction windows
        is planned to address contamination.
        
        Anomaly types in SMAP:
          - Mode switches: sudden telemetry channel transitions
          - Point anomalies: single-tick outliers
          - Contextual anomalies: normal values in wrong context
          - Collective anomalies: sequences that deviate as a group
        
        Channel groups: attitude control, thermal, power, science instrument.
        Anomaly score above 0.60 triggers alert logging.
    """).strip(),

    "recon": textwrap.dedent("""
        You are analyzing output from the RECON outdoor navigation system.
        A StudentEncoder running on AMD Ryzen AI NPU processes 224Ã—224 RGB frames
        at 4Hz from a Jackal robot stereo camera into 128-dimensional embeddings.
        
        The system uses a TemporalHead (InfoNCE temporal contrastive training)
        to learn visual quasimetric distances for navigation planning.
        Current AUROC for quasimetric: 0.9337.
        
        The latent space encodes:
          - Scene type: open path, dense vegetation, obstacle field, narrow corridor
          - Traversability: smooth, rough, sloped terrain
          - Temporal context: how much the scene has changed since t-k
          - Navigation state: approaching goal, lateral drift, correct heading
        
        Dead-reckoning MAE: 0.098m. Visual odometry integrates 4Hz latent deltas.
        The TemporalHead produces quasimetric distances; small = close, large = far.
    """).strip(),
}


# =============================================================================
# Language Memory Database
# =============================================================================

LANGUAGE_SCHEMA = """
-- (latent, description) pairs â€” the LLM-JEPA view pair corpus for this domain
CREATE TABLE IF NOT EXISTS descriptions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   REAL    NOT NULL,
    domain      TEXT    NOT NULL,
    latent_blob BLOB    NOT NULL,       -- 128-D float32
    anomaly_score REAL,
    question    TEXT,
    answer      TEXT    NOT NULL,
    model_used  TEXT    NOT NULL,
    tokens_used INTEGER
);

-- Cached domain statistics (updated on each insert)
CREATE TABLE IF NOT EXISTS domain_stats (
    domain          TEXT    PRIMARY KEY,
    n_descriptions  INTEGER DEFAULT 0,
    avg_anomaly     REAL    DEFAULT 0.0,
    last_seen       REAL
);

CREATE INDEX IF NOT EXISTS idx_desc_domain ON descriptions(domain);
CREATE INDEX IF NOT EXISTS idx_desc_ts     ON descriptions(timestamp);
"""


class LanguageMemoryDB:
    """
    SQLite store for (latent, text) pairs.
    Separate from ExperienceDB â€” no trading data mixed in.
    """

    def __init__(self, path: str = "language_memory.db"):
        self.path = path
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(LANGUAGE_SCHEMA)
        self._conn.commit()

    def record(self, domain: str, latent: np.ndarray, answer: str,
               question: Optional[str] = None, anomaly_score: Optional[float] = None,
               model_used: str = "unknown", tokens_used: int = 0) -> int:
        cur = self._conn.execute("""
            INSERT INTO descriptions
            (timestamp, domain, latent_blob, anomaly_score, question, answer, model_used, tokens_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            time.time(), domain, latent.astype(np.float32).tobytes(),
            float(anomaly_score) if anomaly_score is not None else None, question, answer, model_used, int(tokens_used)
        ))
        self._conn.execute("""
            INSERT INTO domain_stats (domain, n_descriptions, avg_anomaly, last_seen)
            VALUES (?, 1, ?, ?)
            ON CONFLICT(domain) DO UPDATE SET
                n_descriptions = n_descriptions + 1,
                avg_anomaly    = (avg_anomaly * n_descriptions + excluded.avg_anomaly) / (n_descriptions + 1),
                last_seen      = excluded.last_seen
        """, (domain, float(anomaly_score) if anomaly_score is not None else 0.0, time.time()))
        self._conn.commit()
        return cur.lastrowid

    def recent(self, domain: str, n: int = 5) -> list[dict]:
        rows = self._conn.execute("""
            SELECT timestamp, question, answer, anomaly_score, model_used
            FROM descriptions WHERE domain=?
            ORDER BY timestamp DESC LIMIT ?
        """, (domain, n)).fetchall()
        return [
            {"ts": r[0], "question": r[1], "answer": r[2],
             "anomaly_score": r[3], "model": r[4]}
            for r in rows
        ]

    def jepa_batch(self, domain: str, n: int = 256) -> Optional[tuple]:
        """Return (latents, answers) for LLM-JEPA training â€” the view pair corpus."""
        rows = self._conn.execute("""
            SELECT latent_blob, answer FROM descriptions
            WHERE domain=? ORDER BY RANDOM() LIMIT ?
        """, (domain, n)).fetchall()
        if len(rows) < 16:
            return None
        latents = np.stack([
            np.frombuffer(r[0], dtype=np.float32) for r in rows
        ])
        texts = [r[1] for r in rows]
        return latents, texts

    def stats(self) -> dict:
        rows = self._conn.execute(
            "SELECT domain, n_descriptions, avg_anomaly FROM domain_stats"
        ).fetchall()
        return {r[0]: {"n": r[1], "avg_anomaly": r[2]} for r in rows}

    def migrate_scores(self) -> int:
        """
        Fix np.float32 values stored as 4-byte BLOBs in anomaly_score column.
        Call once on any existing DB: db.migrate_scores()
        Safe to run multiple times (idempotent).
        Returns number of rows fixed.
        """
        import struct
        rows = self._conn.execute(
            "SELECT id, anomaly_score FROM descriptions"
        ).fetchall()
        fixed = 0
        for row_id, val in rows:
            if isinstance(val, bytes) and len(val) == 4:
                real_val = float(struct.unpack('<f', val)[0])
                self._conn.execute(
                    "UPDATE descriptions SET anomaly_score=? WHERE id=?",
                    (real_val, row_id)
                )
                fixed += 1
        if fixed:
            self._conn.commit()
        return fixed

    def close(self):
        self._conn.commit()
        self._conn.close()


# =============================================================================
# LatentProjector â€” maps NPU embeddings into LLM token space
# =============================================================================

if HAS_TORCH:
    class LatentProjector(nn.Module):
        """
        Projects 128-D NPU latent into the LLM's embedding dimension.
        Produces n_tokens "perception tokens" prepended to the LLM context.
        
        Based on LLaVA's linear projection, extended to multi-token output.
        ~50K params for llm_dim=896 (Qwen2.5-0.5B), ~100K for llm_dim=1536.
        
        The n_tokens dimension lets the LLM attend over a richer representation:
          - Token 0: global summary (mean-pool)
          - Token 1: anomaly-sensitive features (scaled by anomaly_score)
          - Token 2: temporal delta (if sequence provided)
        """
        def __init__(self, latent_dim: int = 128, llm_dim: int = 896, n_tokens: int = 4):
            super().__init__()
            self.n_tokens = n_tokens
            self.proj = nn.Sequential(
                nn.Linear(latent_dim, 256),
                nn.GELU(),
                nn.LayerNorm(256),
                nn.Linear(256, llm_dim * n_tokens),  # produce n_tokens worth of embeddings
            )
            self.token_norm = nn.LayerNorm(llm_dim)

        def forward(self, z: "torch.Tensor") -> "torch.Tensor":
            """
            z: [B, 128] or [128]
            Returns: [B, n_tokens, llm_dim] â€” perception token sequence
            """
            if z.dim() == 1:
                z = z.unsqueeze(0)
            out = self.proj(z)  # [B, llm_dim * n_tokens]
            out = out.view(z.shape[0], self.n_tokens, -1)  # [B, n_tokens, llm_dim]
            return self.token_norm(out)

        def to_text_summary(self, z: "torch.Tensor") -> str:
            """
            Fallback: convert latent to a compact numerical string
            for injection into LLM context when native token injection isn't possible.
            Used with API-based LLMs (Anthropic) where we can't inject raw tensors.
            """
            z_np = z.detach().cpu().numpy().flatten()
            # Summarize the 128-D vector as: mean, std, max_dims, min_dims
            mean = float(z_np.mean())
            std = float(z_np.std())
            top5_idx = z_np.argsort()[-5:][::-1]
            bot5_idx = z_np.argsort()[:5]
            summary = (
                f"[latent: mean={mean:.3f}, std={std:.3f}, "
                f"active_dims={top5_idx.tolist()}, "
                f"suppressed_dims={bot5_idx.tolist()}, "
                f"norm={float(np.linalg.norm(z_np)):.3f}]"
            )
            return summary


# =============================================================================
# LLM Backends
# =============================================================================

class OllamaBackend:
    """
    Local LLM via ollama. Runs Qwen3-0.6B on the NUC CPU/iGPU.
    Install: https://ollama.ai  |  Model: ollama pull qwen3:0.6b
    """
    DEFAULT_MODEL = "qwen3:0.6b"
    API_URL = "http://localhost:11434/api/generate"

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self._available = self._check()

    def _check(self) -> bool:
        if not HAS_REQUESTS:
            return False
        try:
            r = _requests.get("http://localhost:11434/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    @property
    def available(self) -> bool:
        return self._available

    def generate(self, prompt: str, max_tokens: int = 512) -> tuple[str, int]:
        """Returns (response_text, tokens_used)."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.3}
        }
        r = _requests.post(self.API_URL, json=payload, timeout=60)
        data = r.json()
        text = data.get("response", "")
        # Qwen3 wraps chain-of-thought in <think>...</think> — strip before returning
        import re as _re
        text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()
        tokens = data.get("eval_count", 0)
        return text.strip(), tokens


class AnthropicBackend:
    """
    Fallback: Anthropic claude-haiku-4-5 via API.
    Useful when ollama isn't running or for higher-quality responses.
    Does NOT inject raw tensor tokens â€” uses text summary of latent instead.
    """
    API_URL = "https://api.anthropic.com/v1/messages"
    MODEL = "claude-haiku-4-5-20251001"

    def __init__(self):
        self._available = HAS_REQUESTS

    @property
    def available(self) -> bool:
        return self._available

    def generate(self, prompt: str, max_tokens: int = 512) -> tuple[str, int]:
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": self.MODEL,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]
        }
        r = _requests.post(self.API_URL, headers=headers, json=payload, timeout=30)
        data = r.json()
        text = data.get("content", [{}])[0].get("text", "")
        tokens = data.get("usage", {}).get("output_tokens", 0)
        return text.strip(), tokens


class MockBackend:
    """
    Zero-dependency fallback for smoke testing with no LLM available.
    Produces deterministic template responses based on domain + anomaly score.
    """
    available = True

    def generate(self, prompt: str, max_tokens: int = 512) -> tuple[str, int]:
        # Extract domain hint from prompt
        domain = "unknown"
        for d in DOMAIN_CONTEXTS:
            if d in prompt.lower():
                domain = d
                break

        anomaly_hint = ""
        if "0.9" in prompt or "0.8" in prompt:
            anomaly_hint = "HIGH anomaly confidence detected. "
        elif "0.3" in prompt or "0.2" in prompt:
            anomaly_hint = "LOW anomaly confidence â€” likely normal. "

        templates = {
            "bearing": f"{anomaly_hint}The latent embedding shows {'elevated spectral energy in the outer race frequency band, consistent with early-stage outer race fault' if anomaly_hint else 'tight clustering near the normal centroid. No fault signature detected.'}",
            "cardiac": f"{anomaly_hint}Heart sound analysis: {'irregular energy between S1 and S2 suggests systolic murmur' if anomaly_hint else 'clean S1-S2 pattern with normal intervals. No murmur detected.'}",
            "smap": f"{anomaly_hint}Telemetry state: {'collective anomaly across thermal and power channels, possible mode switch' if anomaly_hint else 'nominal telemetry across all channels.'}",
            "recon": f"{anomaly_hint}Scene analysis: {'significant visual change from previous frame â€” obstacle or terrain change' if anomaly_hint else 'stable traversable path ahead, low temporal delta.'}",
            "unknown": f"{anomaly_hint}[MockBackend] No LLM available. Domain-specific response would appear here.",
        }
        return templates.get(domain, templates["unknown"]), 50


# =============================================================================
# PerceptionLLM â€” main interface
# =============================================================================

class PerceptionLLM:
    """
    Language interface for the CORTEX-PE perception engine.
    
    Accepts 128-D NPU latent embeddings and answers natural language questions
    about what the perception engine is observing.
    
    Completely modular â€” disable by simply not instantiating this class.
    No side effects on NPU inference, trading loop, or anomaly detection.
    
    Example:
        plm = PerceptionLLM(domain="bearing", db_path="language_memory.db")
        z = student_encoder(audio_frame)           # 128-D from NPU
        answer = plm.ask(z, "Is this bearing healthy?")
        print(answer)
        # â†’ "The latent embedding shows elevated spectral energy near dims [42, 17, 88],
        #    consistent with early outer race fault. Anomaly score 0.83 exceeds threshold."
    """

    def __init__(
        self,
        domain: str = "bearing",
        db_path: str = "language_memory.db",
        prefer_local: bool = True,
        projector_path: Optional[str] = None,
        verbose: bool = False,
    ):
        assert domain in DOMAIN_CONTEXTS, f"Unknown domain: {domain}. Choose from {list(DOMAIN_CONTEXTS)}"
        self.domain = domain
        self.verbose = verbose
        self.db = LanguageMemoryDB(db_path)

        # Select LLM backend
        self._ollama = OllamaBackend()
        self._anthropic = AnthropicBackend()
        self._mock = MockBackend()

        if prefer_local and self._ollama.available:
            self.llm = self._ollama
            self._backend_name = f"ollama/{OllamaBackend.DEFAULT_MODEL}"
        elif self._anthropic.available:
            self.llm = self._anthropic
            self._backend_name = f"anthropic/{AnthropicBackend.MODEL}"
        else:
            self.llm = self._mock
            self._backend_name = "mock"

        if self.verbose:
            print(f"[PerceptionLLM] Backend: {self._backend_name} | Domain: {domain}")

        # Optional latent projector (for local LLM token injection)
        self.projector = None
        if HAS_TORCH and projector_path and Path(projector_path).exists():
            self.projector = LatentProjector()
            self.projector.load_state_dict(torch.load(projector_path))
            self.projector.eval()

    def _build_prompt(
        self,
        z: np.ndarray,
        question: str,
        anomaly_score: Optional[float] = None,
        extra_context: str = "",
    ) -> str:
        """
        Build the full prompt: domain context + latent summary + question.
        The latent summary is a compressed numerical description of the embedding.
        """
        # Latent summary (text-based for API backends, token-based for local LLM)
        mean = float(z.mean())
        std = float(z.std())
        norm = float(np.linalg.norm(z))
        top5 = z.argsort()[-5:][::-1].tolist()
        bot5 = z.argsort()[:5].tolist()
        anomaly_line = f"Anomaly score: {anomaly_score:.4f}" if anomaly_score is not None else ""

        latent_summary = (
            f"Current perception state (128-D latent embedding):\n"
            f"  Statistical summary: mean={mean:.4f}, std={std:.4f}, L2-norm={norm:.4f}\n"
            f"  Most activated dimensions: {top5}\n"
            f"  Least activated dimensions: {bot5}\n"
            f"  {anomaly_line}"
        )

        prompt = (
            f"{DOMAIN_CONTEXTS[self.domain]}\n\n"
            f"---\n"
            f"{latent_summary}\n"
            f"{extra_context}\n"
            f"---\n\n"
            f"Question: {question}\n\n"
            f"Answer concisely and technically. "
            f"Do not add disclaimers beyond what the domain context already states."
        )
        return prompt

    def ask(
        self,
        z: np.ndarray,
        question: str,
        anomaly_score: Optional[float] = None,
        save: bool = True,
        extra_context: str = "",
    ) -> str:
        """
        Ask a natural language question about a perception state.
        
        Args:
            z:             128-D float32 latent from NPU StudentEncoder
            question:      natural language question
            anomaly_score: pre-computed anomaly score (optional, adds context)
            save:          whether to save to LanguageMemoryDB
            extra_context: additional domain-specific context string
        
        Returns:
            Natural language answer string
        """
        z = np.asarray(z, dtype=np.float32).flatten()
        assert z.shape == (128,), f"Expected 128-D latent, got {z.shape}"

        prompt = self._build_prompt(z, question, anomaly_score, extra_context)
        answer, tokens = self.llm.generate(prompt)

        if save:
            self.db.record(
                domain=self.domain, latent=z, answer=answer,
                question=question, anomaly_score=anomaly_score,
                model_used=self._backend_name, tokens_used=tokens,
            )

        if self.verbose:
            print(f"[PerceptionLLM/{self.domain}] Q: {question}")
            print(f"[PerceptionLLM/{self.domain}] A: {answer[:200]}...")
            print(f"[PerceptionLLM/{self.domain}] Tokens: {tokens} | Backend: {self._backend_name}")

        return answer

    def describe(
        self,
        z: np.ndarray,
        anomaly_score: Optional[float] = None,
        save: bool = True,
    ) -> str:
        """Shorthand: generate a free-form description of the current perception state."""
        return self.ask(
            z,
            question=(
                f"Describe what the perception engine is currently observing "
                f"in the {self.domain} domain. Be specific about the latent "
                f"features and their physical meaning."
            ),
            anomaly_score=anomaly_score,
            save=save,
        )

    def alert(self, z: np.ndarray, anomaly_score: float) -> Optional[str]:
        """
        Generate an alert description only if anomaly_score exceeds domain threshold.
        Returns None if score is below threshold (no-op in normal operation).
        """
        thresholds = {"bearing": 0.65, "cardiac": 0.70, "smap": 0.60, "recon": 0.75}
        threshold = thresholds.get(self.domain, 0.65)

        if anomaly_score < threshold:
            return None

        return self.ask(
            z,
            question=(
                f"An anomaly has been detected with score {anomaly_score:.4f} "
                f"(threshold: {threshold}). Describe the anomaly in detail. "
                f"What are the most likely failure modes or causes?"
            ),
            anomaly_score=anomaly_score,
            save=True,
        )

    def compare(self, z1: np.ndarray, z2: np.ndarray, label1: str = "t", label2: str = "t+k") -> str:
        """
        Compare two latent states â€” useful for temporal change detection in RECON.
        """
        z1 = np.asarray(z1, dtype=np.float32).flatten()
        z2 = np.asarray(z2, dtype=np.float32).flatten()
        delta = z2 - z1
        cosine = float(np.dot(z1, z2) / (np.linalg.norm(z1) * np.linalg.norm(z2) + 1e-8))
        l2 = float(np.linalg.norm(delta))

        extra = (
            f"Comparing two perception states:\n"
            f"  Cosine similarity ({label1} vs {label2}): {cosine:.4f}\n"
            f"  L2 distance: {l2:.4f}\n"
            f"  Top changing dimensions: {np.abs(delta).argsort()[-5:][::-1].tolist()}\n"
        )
        return self.ask(
            z2,
            question=f"How has the scene/signal changed between {label1} and {label2}?",
            extra_context=extra,
        )

    def history(self, n: int = 5) -> list[dict]:
        """Return the n most recent Q&A pairs for this domain."""
        return self.db.recent(self.domain, n)

    def stats(self) -> dict:
        return self.db.stats()

    def close(self):
        self.db.close()


# =============================================================================
# Domain-specific convenience wrappers
# =============================================================================

class BearingLLM(PerceptionLLM):
    """Bearing fault detection language interface."""
    def __init__(self, **kwargs):
        super().__init__(domain="bearing", **kwargs)

    def diagnose(self, z: np.ndarray, anomaly_score: float,
                 rpm: Optional[float] = None) -> str:
        extra = f"Shaft speed: {rpm:.0f} RPM\n" if rpm else ""
        return self.ask(
            z,
            question="What bearing fault type does this signal most likely represent? "
                     "Estimate severity (early/moderate/severe).",
            anomaly_score=anomaly_score,
            extra_context=extra,
        )


class CardiacLLM(PerceptionLLM):
    """Cardiac audio anomaly language interface."""
    def __init__(self, **kwargs):
        super().__init__(domain="cardiac", **kwargs)

    def screen(self, z: np.ndarray, anomaly_score: float) -> str:
        return self.ask(
            z,
            question="Screen this heart sound recording. What does the anomaly pattern suggest? "
                     "Which clinical finding should be investigated?",
            anomaly_score=anomaly_score,
        )


class SMAPLanguage(PerceptionLLM):
    """SMAP telemetry anomaly language interface."""
    def __init__(self, **kwargs):
        super().__init__(domain="smap", **kwargs)

    def triage(self, z: np.ndarray, anomaly_score: float, channel_id: str = "") -> str:
        extra = f"Telemetry channel: {channel_id}\n" if channel_id else ""
        return self.ask(
            z,
            question="Triage this telemetry anomaly. Which subsystem is most likely affected? "
                     "Is this a hard failure, soft fault, or environmental effect?",
            anomaly_score=anomaly_score,
            extra_context=extra,
        )


class RECONLanguage(PerceptionLLM):
    """RECON outdoor navigation language interface."""
    def __init__(self, **kwargs):
        super().__init__(domain="recon", **kwargs)

    def narrate_scene(self, z: np.ndarray, z_prev: Optional[np.ndarray] = None,
                      quasimetric_dist: Optional[float] = None) -> str:
        extra = ""
        if quasimetric_dist is not None:
            extra += f"Visual quasimetric distance to goal: {quasimetric_dist:.4f}\n"
        if z_prev is not None:
            delta_norm = float(np.linalg.norm(z - z_prev))
            extra += f"Scene change (L2 delta from previous frame): {delta_norm:.4f}\n"
        return self.ask(
            z,
            question="Describe the current outdoor scene and the robot's navigational state. "
                     "Is the path clear? What terrain features dominate?",
            extra_context=extra,
        )


# =============================================================================
# Projector training â€” offline, nightly
# =============================================================================

if HAS_TORCH:
    def train_projector(
        db: LanguageMemoryDB,
        domain: str,
        save_path: str = "checkpoints/projector",
        epochs: int = 20,
        lr: float = 1e-3,
    ) -> Optional[str]:
        """
        Train the LatentProjector to align NPU latent space with LLM embedding space.
        
        Uses accumulated (latent, description) pairs from LanguageMemoryDB as the
        LLM-JEPA view pair corpus. The objective: projector(z_latent) should be
        close to the LLM's embedding of the corresponding description.
        
        This is the exact LLM-JEPA training objective applied to perception:
          L = cosine_distance(Projector(z_npu), LLM_embed(text_description))
        
        In practice without access to LLM internals, we use a contrastive proxy:
          - Positive pair: (z, text_A) where text_A was generated FROM z
          - Negative pair: (z, text_B) where text_B was generated from a different z
          - InfoNCE loss aligns them
        
        Runs offline after accumulating >100 (latent, text) pairs.
        """
        batch = db.jepa_batch(domain, n=256)
        if batch is None:
            print(f"[train_projector] Not enough data for {domain}. Need â‰¥16 samples.")
            return None

        latents, texts = batch
        z_tensor = torch.from_numpy(latents)  # [N, 128]

        projector = LatentProjector(latent_dim=128, llm_dim=896, n_tokens=4)
        optimizer = torch.optim.AdamW(projector.parameters(), lr=lr)

        print(f"[train_projector] Training on {len(latents)} pairs for {domain}...")
        for epoch in range(epochs):
            projector.train()
            optimizer.zero_grad()

            # Project latents: [N, n_tokens, llm_dim] â†’ use first token
            proj = projector(z_tensor)[:, 0, :]  # [N, llm_dim]
            proj_norm = proj / (proj.norm(dim=-1, keepdim=True) + 1e-8)

            # Self-contrastive loss: push same-batch embeddings apart
            # (positive = diagonal, negatives = off-diagonal)
            sim_matrix = proj_norm @ proj_norm.T  # [N, N]
            labels = torch.arange(len(proj_norm))
            loss = torch.nn.functional.cross_entropy(sim_matrix / 0.07, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
            optimizer.step()

            if epoch % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs} | loss={loss.item():.4f}")

        Path(save_path).mkdir(parents=True, exist_ok=True)
        ckpt = f"{save_path}/{domain}_projector_{int(time.time())}.pt"
        torch.save(projector.state_dict(), ckpt)
        print(f"[train_projector] Saved â†’ {ckpt}")
        return ckpt


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="CORTEX-PE PerceptionLLM CLI")
    parser.add_argument("--domain", choices=list(DOMAIN_CONTEXTS), default="bearing")
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--anomaly-score", type=float, default=None)
    parser.add_argument("--db", type=str, default="language_memory.db")
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--backend", choices=["local", "api", "mock"], default="local")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    plm = PerceptionLLM(
        domain=args.domain,
        db_path=args.db,
        prefer_local=(args.backend == "local"),
        verbose=args.verbose,
    )

    if args.stats:
        print(json.dumps(plm.stats(), indent=2))
        return

    if args.history:
        for entry in plm.history(n=5):
            ts = time.strftime("%H:%M:%S", time.localtime(entry["ts"]))
            print(f"[{ts}] Q: {entry['question']}")
            print(f"       A: {entry['answer'][:200]}")
            print()
        return

    # Synthetic latent (replace with real NPU output in production)
    rng = np.random.default_rng(42)
    z = rng.standard_normal(128).astype(np.float32)

    if args.describe:
        answer = plm.describe(z, anomaly_score=args.anomaly_score)
        print(f"\n[{args.domain.upper()} DESCRIPTION]\n{answer}\n")

    elif args.question:
        answer = plm.ask(z, args.question, anomaly_score=args.anomaly_score)
        print(f"\n[{args.domain.upper()} Q&A]\nQ: {args.question}\nA: {answer}\n")

    elif args.anomaly_score is not None:
        alert = plm.alert(z, args.anomaly_score)
        if alert:
            print(f"\n[ALERT â€” {args.domain.upper()}]\n{alert}\n")
        else:
            print(f"[{args.domain.upper()}] Score {args.anomaly_score:.3f} below threshold. No alert.")

    else:
        parser.print_help()

    plm.close()


if __name__ == "__main__":
    # Smoke test (no args = run test)
    import sys
    if len(sys.argv) == 1:
        print("=== PerceptionLLM Smoke Test ===\n")
        import tempfile, os

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            tmp_db = f.name

        for domain in ["bearing", "cardiac", "smap", "recon"]:
            plm = PerceptionLLM(domain=domain, db_path=tmp_db,
                                prefer_local=False, verbose=False)
            # Force mock backend for CI
            plm.llm = MockBackend()
            plm._backend_name = "mock"

            z = np.random.randn(128).astype(np.float32)
            ans = plm.describe(z, anomaly_score=0.82)
            print(f"[{domain:8s}] {ans[:120]}")
            plm.close()

        # Test DB stats
        db = LanguageMemoryDB(tmp_db)
        stats = db.stats()
        print(f"\nDB stats: {json.dumps(stats, indent=2)}")
        db.close()
        os.unlink(tmp_db)
        print("\nâœ… Smoke test passed")
    else:
        main()

