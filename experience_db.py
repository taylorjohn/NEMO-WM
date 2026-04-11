"""
CORTEX-PE Experience Database — Retrieval-Augmented Trading Memory
===================================================================

Architecture analogy to LLMs:

  LLM pretraining corpus   →  ExperienceDB (SQLite, grows every session)
  LLM token embeddings     →  128-D market embeddings (MarketEncoder / NPU StudentEncoder)
  RAG retrieval            →  cosine search over experience index
  In-context few-shot      →  ContextualAdapter conditioned on retrieved episodes
  Fine-tuning              →  nightly retraining of adapter on accumulated experiences

The system learns in two ways:
  1. Online: each trade outcome is written to the DB immediately
  2. Offline: nightly retraining of ContextualAdapter on JEPA corpus

Usage in the trading loop:
  db = ExperienceDB("cortex_memory.db")
  db.load_index()

  # At decision time:
  context = db.retrieve(z_current, k=5)
  confidence_adj = adapter(z_current, context)

  # After trade closes:
  db.record_trade(session_id, scenario, z, action, entry, exit, pnl, rho)
"""

import sqlite3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import json
import time


# ─────────────────────────────────────────────────────────────────────────────
# Database Schema
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA = """
-- Every trade outcome, forever
CREATE TABLE IF NOT EXISTS experiences (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       REAL    NOT NULL,
    session_id      TEXT    NOT NULL,
    scenario        TEXT    NOT NULL,      -- TRENDING_UP, FLASH_CRASH, etc.
    market_emb      BLOB    NOT NULL,      -- 128-D float32 (market encoder output)
    market_features BLOB    NOT NULL,      -- 8-D float32 (raw: price,vol,spread,rtt,...)
    action          TEXT    NOT NULL,      -- LONG / SHORT / HOLD
    entry_price     REAL,
    exit_price      REAL,
    pnl             REAL,
    duration_ticks  INTEGER,
    rho             REAL,                  -- resonance at entry
    outcome         TEXT                   -- WIN / LOSS / NEUTRAL
);

-- Per-session summary for trend analysis
CREATE TABLE IF NOT EXISTS sessions (
    id              TEXT    PRIMARY KEY,
    date            TEXT    NOT NULL,
    start_vault     REAL,
    end_vault       REAL,
    total_pnl       REAL,
    win_rate        REAL,
    num_trades      INTEGER,
    scenarios_hit   TEXT,                  -- JSON list of scenarios encountered
    checkpoint_path TEXT                   -- path to model weights snapshot
);

-- Market-JEPA training corpus (view pairs)
-- View 1: market_features at tick t
-- View 2: scenario at tick t+k
-- Grows every tick, used for nightly JEPA retraining
CREATE TABLE IF NOT EXISTS jepa_corpus (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       REAL    NOT NULL,
    session_id      TEXT    NOT NULL,
    market_features BLOB    NOT NULL,      -- 8-D raw features at tick t
    market_emb      BLOB    NOT NULL,      -- 128-D embedding at tick t
    scenario_future TEXT    NOT NULL,      -- scenario label at tick t+k
    scenario_onehot BLOB    NOT NULL,      -- 18-D one-hot at tick t+k
    k_offset        INTEGER NOT NULL       -- actual k used
);

-- Scenario statistics for monitoring
CREATE TABLE IF NOT EXISTS scenario_stats (
    scenario        TEXT    PRIMARY KEY,
    total_trades    INTEGER DEFAULT 0,
    wins            INTEGER DEFAULT 0,
    losses          INTEGER DEFAULT 0,
    total_pnl       REAL    DEFAULT 0.0,
    avg_rho         REAL    DEFAULT 0.0,
    last_seen       REAL
);

CREATE INDEX IF NOT EXISTS idx_exp_scenario ON experiences(scenario);
CREATE INDEX IF NOT EXISTS idx_exp_session  ON experiences(session_id);
CREATE INDEX IF NOT EXISTS idx_exp_outcome  ON experiences(outcome);
CREATE INDEX IF NOT EXISTS idx_jepa_session ON jepa_corpus(session_id);
"""


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrievedExperience:
    scenario: str
    action: str
    pnl: float
    outcome: str
    rho: float
    similarity: float          # cosine similarity to query embedding
    market_emb: np.ndarray     # 128-D


@dataclass
class TradeRecord:
    session_id: str
    scenario: str
    market_emb: np.ndarray     # 128-D float32
    market_features: np.ndarray  # 8-D float32
    action: str
    entry_price: float
    exit_price: Optional[float]
    pnl: Optional[float]
    duration_ticks: Optional[int]
    rho: float
    outcome: Optional[str]


# ─────────────────────────────────────────────────────────────────────────────
# In-memory vector index (numpy cosine search)
# Fast enough for 10K–500K experiences on AMD NUC CPU
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingIndex:
    """
    Flat cosine similarity index over stored market embeddings.
    Loaded from SQLite at startup, updated incrementally during session.
    
    At 100K experiences: ~50MB RAM, ~2ms per query (128-D, numpy vectorized).
    At 500K experiences: ~250MB RAM, ~10ms per query — still within 50ms tick budget.
    """
    def __init__(self):
        self._embeddings: np.ndarray = np.empty((0, 128), dtype=np.float32)
        self._ids: list[int] = []
        self._scenarios: list[str] = []
        self._outcomes: list[str] = []
        self._actions: list[str] = []
        self._pnls: list[float] = []
        self._rhos: list[float] = []

    def add(self, exp_id: int, emb: np.ndarray, scenario: str,
            outcome: str, action: str, pnl: float, rho: float):
        norm = emb / (np.linalg.norm(emb) + 1e-8)
        self._embeddings = np.vstack([self._embeddings, norm[None]]) \
            if len(self._embeddings) > 0 else norm[None].copy()
        self._ids.append(exp_id)
        self._scenarios.append(scenario)
        self._outcomes.append(outcome)
        self._actions.append(action)
        self._pnls.append(pnl)
        self._rhos.append(rho)

    def query(self, z: np.ndarray, k: int = 5,
              scenario_filter: Optional[str] = None) -> list[RetrievedExperience]:
        """
        Find k most similar past market states by cosine similarity.
        Optional scenario_filter restricts to a specific scenario class.
        """
        if len(self._ids) == 0:
            return []

        z_norm = z / (np.linalg.norm(z) + 1e-8)
        sims = self._embeddings @ z_norm           # [N] cosine similarities

        if scenario_filter:
            mask = np.array([s == scenario_filter for s in self._scenarios])
            sims = np.where(mask, sims, -2.0)      # mask non-matching scenarios

        top_k = np.argsort(sims)[::-1][:k]

        results = []
        for idx in top_k:
            if sims[idx] < -1.5:  # all masked out
                continue
            results.append(RetrievedExperience(
                scenario=self._scenarios[idx],
                action=self._actions[idx],
                pnl=self._pnls[idx],
                outcome=self._outcomes[idx],
                rho=self._rhos[idx],
                similarity=float(sims[idx]),
                market_emb=self._embeddings[idx],
            ))
        return results

    def __len__(self):
        return len(self._ids)


# ─────────────────────────────────────────────────────────────────────────────
# Experience Database
# ─────────────────────────────────────────────────────────────────────────────

class ExperienceDB:
    """
    Persistent trading memory. SQLite backend + in-memory cosine index.
    
    This is the 'training corpus' equivalent for CORTEX-PE.
    Every trade outcome becomes a training example for future sessions.
    """

    # 18 scenarios from CORTEX-16 safeguard architecture
    SCENARIOS = [
        "TRENDING_UP", "TRENDING_DOWN", "FLASH_CRASH", "RECOVERY",
        "FRACTURE", "CALM", "HIGH_VOL", "LOW_VOL", "STALE_DATA",
        "GHOST_RTT", "PDT_LOCK", "EOD", "TOXIC_SPREAD", "CIRCUIT_BREAK",
        "SNIPER", "HUNTER", "SCOUT", "UNKNOWN"
    ]
    SCENARIO_IDX = {s: i for i, s in enumerate(SCENARIOS)}

    def __init__(self, db_path: str = "cortex_memory.db"):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")   # concurrent reads during trading
        self._conn.execute("PRAGMA synchronous=NORMAL") # safe but not fsync-on-every-write
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        self._index = EmbeddingIndex()

    def load_index(self, max_rows: int = 100_000):
        """
        Load recent experiences into the in-memory index at startup.
        Loads the most recent max_rows experiences (recency bias).
        """
        rows = self._conn.execute("""
            SELECT id, market_emb, scenario, outcome, action, pnl, rho
            FROM experiences
            WHERE outcome IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT ?
        """, (max_rows,)).fetchall()

        for row in rows:
            exp_id, emb_blob, scenario, outcome, action, pnl, rho = row
            emb = np.frombuffer(emb_blob, dtype=np.float32).copy()
            if emb.shape == (128,):
                self._index.add(exp_id, emb, scenario,
                                outcome or "NEUTRAL", action or "HOLD",
                                pnl or 0.0, rho or 0.0)

        print(f"[ExperienceDB] Loaded {len(self._index)} experiences into index.")
        return len(self._index)

    def retrieve(self, z: np.ndarray, k: int = 5,
                 scenario_filter: Optional[str] = None) -> list[RetrievedExperience]:
        """
        Retrieve k most similar past market states.
        This is the RAG retrieval step — called every tick before decision.
        """
        return self._index.query(z, k=k, scenario_filter=scenario_filter)

    def record_trade_open(self, record: TradeRecord) -> int:
        """Write trade to DB at entry. Returns row id."""
        cur = self._conn.execute("""
            INSERT INTO experiences
            (timestamp, session_id, scenario, market_emb, market_features,
             action, entry_price, rho, outcome)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            time.time(), record.session_id, record.scenario,
            record.market_emb.tobytes(), record.market_features.tobytes(),
            record.action, record.entry_price, record.rho, None
        ))
        self._conn.commit()
        return cur.lastrowid

    def record_trade_close(self, row_id: int, exit_price: float,
                           pnl: float, duration_ticks: int):
        """Update trade record at exit. Adds to in-memory index."""
        outcome = "WIN" if pnl > 0 else ("LOSS" if pnl < 0 else "NEUTRAL")
        self._conn.execute("""
            UPDATE experiences
            SET exit_price=?, pnl=?, duration_ticks=?, outcome=?
            WHERE id=?
        """, (exit_price, pnl, duration_ticks, outcome, row_id))
        self._conn.commit()

        # Add completed trade to live index for same-session retrieval
        row = self._conn.execute(
            "SELECT market_emb, scenario, action, rho FROM experiences WHERE id=?",
            (row_id,)
        ).fetchone()
        if row:
            emb = np.frombuffer(row[0], dtype=np.float32).copy()
            self._index.add(row_id, emb, row[1], outcome, row[2], pnl, row[3])

        # Update scenario statistics
        self._conn.execute("""
            INSERT INTO scenario_stats (scenario, total_trades, wins, losses, total_pnl, avg_rho, last_seen)
            VALUES (?, 1, ?, ?, ?, ?, ?)
            ON CONFLICT(scenario) DO UPDATE SET
                total_trades = total_trades + 1,
                wins         = wins + excluded.wins,
                losses       = losses + excluded.losses,
                total_pnl    = total_pnl + excluded.total_pnl,
                avg_rho      = (avg_rho * total_trades + excluded.avg_rho) / (total_trades + 1),
                last_seen    = excluded.last_seen
        """, (
            row[1] if row else "UNKNOWN",
            1 if pnl > 0 else 0,
            1 if pnl < 0 else 0,
            pnl, row[3] if row else 0.0, time.time()
        ))
        self._conn.commit()

    def record_jepa_tick(self, session_id: str, market_features: np.ndarray,
                         market_emb: np.ndarray, scenario_future: str, k: int):
        """
        Record a (market_t, scenario_{t+k}) pair for JEPA corpus.
        Called every tick — provides the 'training data' for Market-JEPA retraining.
        """
        scenario_oh = np.zeros(18, dtype=np.float32)
        idx = self.SCENARIO_IDX.get(scenario_future, 17)
        scenario_oh[idx] = 1.0

        self._conn.execute("""
            INSERT INTO jepa_corpus
            (timestamp, session_id, market_features, market_emb, scenario_future, scenario_onehot, k_offset)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            time.time(), session_id,
            market_features.tobytes(), market_emb.tobytes(),
            scenario_future, scenario_oh.tobytes(), k
        ))
        # Batch commit every 100 ticks to avoid write amplification
        # Caller should call conn.commit() periodically

    def record_session(self, session_id: str, date: str, start_vault: float,
                       end_vault: float, win_rate: float, num_trades: int,
                       scenarios_hit: list[str], checkpoint_path: str = ""):
        self._conn.execute("""
            INSERT OR REPLACE INTO sessions
            (id, date, start_vault, end_vault, total_pnl, win_rate, num_trades, scenarios_hit, checkpoint_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id, date, start_vault, end_vault,
            end_vault - start_vault, win_rate, num_trades,
            json.dumps(scenarios_hit), checkpoint_path
        ))
        self._conn.commit()

    def get_scenario_stats(self) -> dict:
        """Return win rate and avg PnL per scenario — for HUD display."""
        rows = self._conn.execute("""
            SELECT scenario, total_trades, wins, total_pnl, avg_rho
            FROM scenario_stats ORDER BY total_trades DESC
        """).fetchall()
        return {
            row[0]: {
                "trades": row[1],
                "win_rate": row[2] / row[1] if row[1] > 0 else 0.0,
                "total_pnl": row[3],
                "avg_rho": row[4],
            }
            for row in rows
        }

    def get_jepa_batch(self, batch_size: int = 256) -> Optional[tuple]:
        """Sample a random batch from the JEPA corpus for retraining."""
        rows = self._conn.execute("""
            SELECT market_features, scenario_onehot
            FROM jepa_corpus
            ORDER BY RANDOM() LIMIT ?
        """, (batch_size,)).fetchall()

        if len(rows) < batch_size // 4:
            return None  # not enough data yet

        features = np.array([
            np.frombuffer(r[0], dtype=np.float32) for r in rows
        ])
        onehots = np.array([
            np.frombuffer(r[1], dtype=np.float32) for r in rows
        ])
        return (
            torch.from_numpy(features),
            torch.from_numpy(onehots)
        )

    def summary(self) -> str:
        total_trades = self._conn.execute(
            "SELECT COUNT(*) FROM experiences WHERE outcome IS NOT NULL"
        ).fetchone()[0]
        total_sessions = self._conn.execute(
            "SELECT COUNT(*) FROM sessions"
        ).fetchone()[0]
        jepa_ticks = self._conn.execute(
            "SELECT COUNT(*) FROM jepa_corpus"
        ).fetchone()[0]
        return (
            f"ExperienceDB: {total_trades} trades | "
            f"{total_sessions} sessions | "
            f"{jepa_ticks} JEPA ticks | "
            f"{len(self._index)} in index"
        )

    def commit(self):
        self._conn.commit()

    def close(self):
        self._conn.commit()
        self._conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Contextual Adapter — the "in-context learning" module
# ─────────────────────────────────────────────────────────────────────────────

class ContextualAdapter(nn.Module):
    """
    Conditions the trading decision on retrieved past experiences.
    
    This is the 'in-context learning' equivalent to LLM few-shot prompting.
    
    Input:  current market embedding (128-D)
          + mean of k retrieved embeddings (128-D)
          + retrieved outcome statistics (4-D: win_rate, avg_pnl, avg_rho, avg_sim)
    Output: confidence adjustment (-1 to +1) and action bias (SHORT/HOLD/LONG)
    
    Small enough to retrain nightly on accumulated JEPA corpus in <5 minutes.
    """

    def __init__(self, latent_dim: int = 128, k_retrieved: int = 5):
        super().__init__()
        # Context summarizer: compress k retrieved embeddings into one vector
        self.context_summarizer = nn.Sequential(
            nn.Linear(latent_dim + 4, 64),  # 4 = [win_rate, avg_pnl, avg_rho, avg_sim]
            nn.GELU(),
            nn.LayerNorm(64),
        )
        # Fusion: combine current state + context summary
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim + 64, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.GELU(),
        )
        # Output heads
        self.confidence_head = nn.Linear(64, 1)   # sigmoid → [0,1] confidence multiplier
        self.action_bias_head = nn.Linear(64, 3)   # softmax → [SHORT, HOLD, LONG] bias

    def summarize_context(
        self, retrieved: list[RetrievedExperience]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert retrieved episodes into a fixed-size context vector."""
        if not retrieved:
            # No history: neutral context
            ctx_emb = torch.zeros(128)
            ctx_stats = torch.tensor([0.5, 0.0, 0.5, 0.0])  # neutral stats
            return ctx_emb, ctx_stats

        embeddings = torch.tensor(
            np.stack([r.market_emb for r in retrieved]), dtype=torch.float32
        )
        ctx_emb = embeddings.mean(dim=0)  # [128]

        wins = sum(1 for r in retrieved if r.outcome == "WIN")
        win_rate = wins / len(retrieved)
        avg_pnl = np.mean([r.pnl for r in retrieved])
        avg_rho = np.mean([r.rho for r in retrieved])
        avg_sim = np.mean([r.similarity for r in retrieved])

        ctx_stats = torch.tensor(
            [win_rate, avg_pnl / 100.0, avg_rho, avg_sim],  # normalize pnl to ~[-1,1]
            dtype=torch.float32
        )
        return ctx_emb, ctx_stats

    def forward(
        self,
        z_current: torch.Tensor,           # [1, 128] current market embedding
        retrieved: list[RetrievedExperience]
    ) -> dict:
        """
        Returns confidence multiplier and action bias given current state + retrieved context.
        
        Usage in trading loop:
            ctx = db.retrieve(z_np, k=5, scenario_filter=current_scenario)
            out = adapter(z_tensor, ctx)
            
            # Multiply existing confidence by adapter output
            final_confidence = base_confidence * out["confidence_mult"]
            
            # Add action bias to existing action logits
            action_logits += out["action_bias"] * 0.3  # small weight
        """
        ctx_emb, ctx_stats = self.summarize_context(retrieved)

        # Summarize context
        ctx_input = torch.cat([ctx_emb.unsqueeze(0), ctx_stats.unsqueeze(0)], dim=-1)  # [1, 132]
        ctx_summary = self.context_summarizer(ctx_input)  # [1, 64]

        # Fuse with current state
        fused = self.fusion(
            torch.cat([z_current, ctx_summary], dim=-1)  # [1, 192]
        )  # [1, 64]

        confidence_mult = torch.sigmoid(self.confidence_head(fused)).squeeze()  # scalar [0,1]
        action_bias = torch.softmax(self.action_bias_head(fused), dim=-1).squeeze()  # [3]

        return {
            "confidence_mult": confidence_mult.item(),
            "action_bias": action_bias.detach().numpy(),   # [SHORT, HOLD, LONG]
            "context_win_rate": ctx_stats[0].item(),
            "context_avg_pnl": ctx_stats[1].item() * 100,
            "n_retrieved": len(retrieved),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Nightly Retrainer
# ─────────────────────────────────────────────────────────────────────────────

class NightlyRetrainer:
    """
    Offline retraining of ContextualAdapter on accumulated ExperienceDB.
    Run after market close (4:01 PM ET), before next session.
    
    This is the 'fine-tuning' equivalent: each day's trades become new training data.
    The JEPA corpus provides the self-supervised pairs; trade outcomes provide
    supervised signal for the confidence head.
    """

    def __init__(self, db: ExperienceDB, adapter: ContextualAdapter,
                 market_encoder: nn.Module):
        self.db = db
        self.adapter = adapter
        self.encoder = market_encoder
        self.optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4, weight_decay=1e-5)

    def train_epoch(self, n_batches: int = 50) -> dict:
        """Train adapter for one epoch on JEPA corpus + outcome supervision."""
        self.adapter.train()
        losses = []

        for _ in range(n_batches):
            batch = self.db.get_jepa_batch(batch_size=256)
            if batch is None:
                break

            market_features, scenario_oh = batch

            # Encode market features
            with torch.no_grad():
                z = self.encoder(market_features)  # [B, 128]

            # JEPA loss: predict scenario from market embedding
            # (using adapter's fusion network as the predictor here)
            # In practice: a separate JEPAPredictor head is used
            # This simplified version uses the scenario cross-entropy
            # as a proxy for "can the adapter predict what scenario follows?"
            action_logits = self.adapter.action_bias_head(
                self.adapter.fusion(
                    torch.cat([
                        z,
                        self.adapter.context_summarizer(
                            torch.cat([z, torch.zeros(z.shape[0], 4)], dim=-1)
                        )
                    ], dim=-1)
                )
            )  # [B, 3]

            # Outcome supervision: wins should produce high confidence
            # We use the scenario one-hot as a proxy target for the action bias
            # (first 3 scenario classes map to SHORT, HOLD, LONG tendencies)
            target_scenario = scenario_oh.argmax(dim=-1) % 3  # [B] mod 3 as proxy
            loss = F.cross_entropy(action_logits, target_scenario)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), 1.0)
            self.optimizer.step()
            losses.append(loss.item())

        avg_loss = np.mean(losses) if losses else 0.0
        self.adapter.eval()
        return {"avg_loss": avg_loss, "n_batches": len(losses)}

    def run(self, epochs: int = 10, checkpoint_dir: str = "checkpoints/adapter") -> str:
        """Full nightly retraining run. Returns checkpoint path."""
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        print(f"[NightlyRetrainer] Starting. DB: {self.db.summary()}")

        history = []
        for epoch in range(epochs):
            result = self.train_epoch()
            history.append(result)
            print(f"  Epoch {epoch+1}/{epochs} | loss={result['avg_loss']:.4f} "
                  f"| batches={result['n_batches']}")

        # Save checkpoint
        ckpt_path = f"{checkpoint_dir}/adapter_{int(time.time())}.pt"
        torch.save(self.adapter.state_dict(), ckpt_path)
        print(f"[NightlyRetrainer] Saved → {ckpt_path}")
        return ckpt_path


# ─────────────────────────────────────────────────────────────────────────────
# Integration snippet — trading loop patch
# ─────────────────────────────────────────────────────────────────────────────

INTEGRATION_EXAMPLE = '''
# ── Add to trading loop startup ───────────────────────────────────────────────

db = ExperienceDB("cortex_memory.db")
n_loaded = db.load_index()
print(f"Loaded {n_loaded} past experiences")

adapter = ContextualAdapter()
try:
    adapter.load_state_dict(torch.load("checkpoints/adapter/latest.pt"))
    adapter.eval()
    print("ContextualAdapter: nightly weights loaded")
except FileNotFoundError:
    print("ContextualAdapter: no checkpoint found, using random init")

SESSION_ID = f"session_{int(time.time())}"
open_trade_db_id = None

# ── Inside tick loop ──────────────────────────────────────────────────────────

# 1. Build market feature vector
market_features = np.array([
    price / 600.0,
    vol / 10.0,
    spread / 0.05,
    rtt / 50.0,
    delta_1m / 5.0,
    delta_5m / 10.0,
    rho,
    lsm_pulse / 100.0,
], dtype=np.float32)

# 2. Encode (existing market encoder or NPU StudentEncoder)
z_np = market_encoder(market_features)  # 128-D numpy

# 3. JEPA corpus: record (market_t, scenario_{t+k})
# scenario_buffer is a deque of length k
if len(scenario_buffer) == K_FUTURE:
    db.record_jepa_tick(SESSION_ID, market_features, z_np, scenario_buffer[0], K_FUTURE)

# 4. Retrieve similar past experiences (RAG step)
retrieved = db.retrieve(z_np, k=5, scenario_filter=current_scenario)

# 5. Get contextual adjustment
z_tensor = torch.from_numpy(z_np).unsqueeze(0)
ctx = adapter(z_tensor, retrieved)

# 6. Apply confidence multiplier to existing entry gate
adjusted_confidence = base_confidence * ctx["confidence_mult"]
# If past similar situations in this scenario had 0% WR → mult is low → stay out

# ── On trade entry ────────────────────────────────────────────────────────────
open_trade_db_id = db.record_trade_open(TradeRecord(
    session_id=SESSION_ID, scenario=current_scenario,
    market_emb=z_np, market_features=market_features,
    action=action, entry_price=price, exit_price=None,
    pnl=None, duration_ticks=None, rho=rho, outcome=None,
))

# ── On trade close ────────────────────────────────────────────────────────────
db.record_trade_close(open_trade_db_id, exit_price=price,
                      pnl=realized_pnl, duration_ticks=ticks_held)

# ── Commit JEPA ticks in batch (every 100 ticks) ─────────────────────────────
if tick % 100 == 0:
    db.commit()

# ── After session end ─────────────────────────────────────────────────────────
db.record_session(SESSION_ID, date=today, start_vault=25000,
                  end_vault=vault.balance, win_rate=session_wr,
                  num_trades=n_trades, scenarios_hit=scenarios_seen)
db.close()

# ── Nightly retraining (run at 4:05 PM ET or on cron) ────────────────────────
retrainer = NightlyRetrainer(db, adapter, market_encoder)
ckpt = retrainer.run(epochs=10)
print(f"Ready for tomorrow: {ckpt}")
'''


if __name__ == "__main__":
    # Smoke test
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        tmp_path = f.name

    db = ExperienceDB(tmp_path)

    # Fake a few trades
    for i in range(5):
        z = np.random.randn(128).astype(np.float32)
        feats = np.random.randn(8).astype(np.float32)
        rec = TradeRecord(
            session_id="test_session", scenario="TRENDING_UP",
            market_emb=z, market_features=feats, action="LONG",
            entry_price=500.0 + i, exit_price=None, pnl=None,
            duration_ticks=None, rho=0.85, outcome=None
        )
        row_id = db.record_trade_open(rec)
        db.record_trade_close(row_id, 502.0 + i, pnl=2.0 - i * 0.5, duration_ticks=10)

    print(db.summary())
    stats = db.get_scenario_stats()
    print("Scenario stats:", stats)

    # Test retrieval
    db.load_index()
    query_z = np.random.randn(128).astype(np.float32)
    results = db.retrieve(query_z, k=3)
    print(f"Retrieved {len(results)} experiences")
    for r in results:
        print(f"  {r.scenario} | {r.outcome} | pnl={r.pnl:.2f} | sim={r.similarity:.4f}")

    # Test adapter
    adapter = ContextualAdapter()
    z_t = torch.randn(1, 128)
    out = adapter(z_t, results)
    print(f"Adapter output: conf_mult={out['confidence_mult']:.4f} | "
          f"action_bias={out['action_bias']}")

    db.close()
    os.unlink(tmp_path)
    print("✅ Smoke test passed")
