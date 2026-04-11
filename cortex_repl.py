"""
CORTEX-PE Conversational REPL
==============================

An interactive multi-turn session that gives CORTEX-PE a language interface.
Works like an LLM chat — but grounded in real NPU perception states.

Features
--------
  • Multi-turn conversation with full history (proper context window)
  • RAG: retrieves similar past observations from LanguageMemoryDB
  • Domain switching mid-conversation: /bearing, /cardiac, /smap, /recon
  • Load real saved latents from checkpoint files
  • Streams answers token-by-token when using ollama backend
  • /save  — commit conversation to DB
  • /diff  — compare current vs previous latent state
  • /alert — run threshold check on current state
  • /stats — show domain statistics
  • /clear — reset context window
  • /quit  — exit

Architecture
------------
  ConversationEngine wraps PerceptionLLM and adds:
    - context_window: list of (role, content) turns
    - rag_retriever:  query LanguageMemoryDB for similar past observations
    - latent_buffer:  rolling deque of recent NPU latents for /diff

Completely separate from trading code. Import perception_llm only.

Usage
-----
  python cortex_repl.py --domain bearing
  python cortex_repl.py --domain recon --latent-file saved_latents/session_001.npy
  python cortex_repl.py --demo          # runs a scripted demo with mock latents
"""

import sys
import os
import json
import time
import textwrap
import argparse
import numpy as np
from pathlib import Path
from collections import deque
from typing import Optional

# Local module — perception_llm must be in same directory or PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))
from perception_llm import (
    PerceptionLLM, LanguageMemoryDB, MockBackend, OllamaBackend,
    DOMAIN_CONTEXTS, BearingLLM, CardiacLLM, SMAPLanguage, RECONLanguage,
)


# ─────────────────────────────────────────────────────────────────────────────
# Terminal colours (cross-platform via ANSI, degrades gracefully on Windows)
# ─────────────────────────────────────────────────────────────────────────────

class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    CYAN   = "\033[36m"
    GREEN  = "\033[32m"
    YELLOW = "\033[33m"
    RED    = "\033[31m"
    BLUE   = "\033[34m"
    MAG    = "\033[35m"

    @staticmethod
    def strip() -> bool:
        """Return True if we should suppress colour codes."""
        return not sys.stdout.isatty() or os.name == "nt"


def fmt(text: str, colour: str, bold: bool = False) -> str:
    if C.strip():
        return text
    prefix = (C.BOLD if bold else "") + colour
    return f"{prefix}{text}{C.RESET}"


def wrap(text: str, width: int = 88, indent: str = "  ") -> str:
    """Word-wrap a block of text with indent."""
    return textwrap.fill(text, width=width, initial_indent=indent,
                         subsequent_indent=indent)


# ─────────────────────────────────────────────────────────────────────────────
# RAG Retriever — semantic search over LanguageMemoryDB
# ─────────────────────────────────────────────────────────────────────────────

class RAGRetriever:
    """
    Retrieves the k most similar past observations from the language memory.
    Used to inject relevant history into the LLM context ("few-shot" grounding).
    
    Similar to ExperienceDB's EmbeddingIndex but operates on text descriptions
    rather than trade outcomes.
    """

    def __init__(self, db: LanguageMemoryDB, domain: str, max_cache: int = 5000):
        self.db = db
        self.domain = domain
        self._embs: np.ndarray = np.empty((0, 128), dtype=np.float32)
        self._answers: list[str] = []
        self._questions: list[str] = []
        self._scores: list[float] = []
        self._load(max_cache)

    def _load(self, n: int):
        rows = self.db._conn.execute("""
            SELECT latent_blob, answer, question, anomaly_score
            FROM descriptions WHERE domain=?
            ORDER BY timestamp DESC LIMIT ?
        """, (self.domain, n)).fetchall()
        for row in rows:
            emb = np.frombuffer(row[0], dtype=np.float32).copy()
            if emb.shape == (128,):
                norm = emb / (np.linalg.norm(emb) + 1e-8)
                self._embs = np.vstack([self._embs, norm[None]]) \
                    if len(self._embs) > 0 else norm[None].copy()
                self._answers.append(row[1])
                self._questions.append(row[2] or "")
                self._scores.append(row[3] or 0.0)

    def add(self, z: np.ndarray, answer: str, question: str = "",
            anomaly_score: float = 0.0):
        norm = z / (np.linalg.norm(z) + 1e-8)
        self._embs = np.vstack([self._embs, norm[None]]) \
            if len(self._embs) > 0 else norm[None].copy()
        self._answers.append(answer)
        self._questions.append(question)
        self._scores.append(anomaly_score)

    def retrieve(self, z: np.ndarray, k: int = 3) -> list[dict]:
        if len(self._answers) == 0:
            return []
        z_norm = z / (np.linalg.norm(z) + 1e-8)
        sims = self._embs @ z_norm
        top_k = np.argsort(sims)[::-1][:k]
        return [
            {
                "similarity": float(sims[i]),
                "question": self._questions[i],
                "answer": self._answers[i],
                "anomaly_score": self._scores[i],
            }
            for i in top_k if sims[i] > 0.1   # ignore very dissimilar
        ]

    def format_for_context(self, retrieved: list[dict]) -> str:
        if not retrieved:
            return ""
        lines = ["Relevant past observations (retrieved from memory):"]
        for i, r in enumerate(retrieved, 1):
            sim_pct = int(r["similarity"] * 100)
            lines.append(
                f"  [{i}] (similarity {sim_pct}%)"
                + (f" anomaly={r['anomaly_score']:.2f}" if r["anomaly_score"] else "")
            )
            if r["question"]:
                lines.append(f"      Q: {r['question'][:80]}")
            lines.append(f"      A: {r['answer'][:200]}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._answers)


# ─────────────────────────────────────────────────────────────────────────────
# Conversation Engine
# ─────────────────────────────────────────────────────────────────────────────

MAX_CONTEXT_TURNS = 12   # keep last N turns in context window
MAX_CONTEXT_CHARS = 6000 # trim if context exceeds this (approximate token budget)


class ConversationEngine:
    """
    Multi-turn conversation layer over PerceptionLLM.
    
    Maintains:
      - context_window: list of {"role": "user"|"assistant", "content": str}
      - current_z:      the active 128-D latent state
      - latent_history: rolling buffer for /diff
      - rag:            retriever for past similar observations
    """

    def __init__(self, plm: PerceptionLLM, db: LanguageMemoryDB):
        self.plm = plm
        self.db = db
        self.rag = RAGRetriever(db, plm.domain)
        self.context_window: list[dict] = []
        self.current_z: Optional[np.ndarray] = None
        self.latent_history: deque = deque(maxlen=32)
        self.turn_count = 0
        self.session_id = f"repl_{int(time.time())}"

    def set_latent(self, z: np.ndarray, anomaly_score: Optional[float] = None):
        """Update current perception state."""
        self.current_z = z.astype(np.float32).flatten()
        self.latent_history.append((time.time(), self.current_z.copy(), anomaly_score))

    def _build_system_context(self) -> str:
        """Build the full system context for this turn."""
        lines = [DOMAIN_CONTEXTS[self.plm.domain], ""]

        # Current latent summary
        if self.current_z is not None:
            z = self.current_z
            mean, std, norm_ = z.mean(), z.std(), np.linalg.norm(z)
            top5 = z.argsort()[-5:][::-1].tolist()
            lines += [
                "Current perception state:",
                f"  mean={mean:.4f}  std={std:.4f}  norm={norm_:.4f}",
                f"  active dims: {top5}",
                "",
            ]

        # RAG context
        if self.current_z is not None:
            retrieved = self.rag.retrieve(self.current_z, k=3)
            rag_block = self.rag.format_for_context(retrieved)
            if rag_block:
                lines += [rag_block, ""]

        return "\n".join(lines)

    def _build_full_prompt(self, user_message: str) -> str:
        """
        Construct the complete prompt:
          [system context]
          [conversation history]
          User: <message>
          Assistant:
        """
        system = self._build_system_context()

        # Conversation history (trim if too long)
        history_parts = []
        for turn in self.context_window[-MAX_CONTEXT_TURNS:]:
            role = "User" if turn["role"] == "user" else "Assistant"
            history_parts.append(f"{role}: {turn['content']}")
        history = "\n".join(history_parts)

        # Trim if over budget
        if len(system) + len(history) > MAX_CONTEXT_CHARS:
            history = history[-(MAX_CONTEXT_CHARS - len(system)):]

        prompt = (
            f"{system}\n"
            f"---\n"
            f"Conversation history:\n{history}\n"
            f"---\n"
            f"User: {user_message}\n"
            f"Assistant:"
        )
        return prompt

    def chat(self, user_message: str,
             anomaly_score: Optional[float] = None) -> str:
        """Process one turn. Returns assistant response."""
        if self.current_z is None:
            # No latent loaded — answer from domain knowledge only
            prompt = (
                f"{DOMAIN_CONTEXTS[self.plm.domain]}\n\n"
                f"User: {user_message}\nAssistant:"
            )
        else:
            prompt = self._build_full_prompt(user_message)

        response, tokens = self.plm.llm.generate(prompt, max_tokens=600)

        # Update context window
        self.context_window.append({"role": "user", "content": user_message})
        self.context_window.append({"role": "assistant", "content": response})
        self.turn_count += 1

        # Save to DB and update RAG index
        if self.current_z is not None:
            record_id = self.db.record(
                domain=self.plm.domain,
                latent=self.current_z,
                answer=response,
                question=user_message,
                anomaly_score=anomaly_score,
                model_used=self.plm._backend_name,
                tokens_used=tokens,
            )
            self.rag.add(self.current_z, response, user_message, anomaly_score or 0.0)

        return response

    def diff(self) -> str:
        """Compare current latent vs the one before it."""
        if len(self.latent_history) < 2:
            return "Not enough latent history for comparison."
        _, z_prev, _ = self.latent_history[-2]
        _, z_curr, score_curr = self.latent_history[-1]
        return self.plm.compare(z_prev, z_curr,
                                label1="previous", label2="current")

    def alert_check(self) -> Optional[str]:
        if self.current_z is None:
            return "No latent loaded."
        _, _, score = self.latent_history[-1] if self.latent_history else (None, None, None)
        if score is None:
            return "No anomaly score available. Load a latent with --anomaly-score."
        result = self.plm.alert(self.current_z, score)
        return result or f"Score {score:.4f} is below alert threshold. System nominal."

    def clear_context(self):
        self.context_window.clear()

    def switch_domain(self, new_domain: str) -> bool:
        if new_domain not in DOMAIN_CONTEXTS:
            return False
        self.plm = PerceptionLLM(
            domain=new_domain,
            db_path=self.db.path,
            prefer_local=isinstance(self.plm.llm, OllamaBackend),
        )
        # Force same backend as before
        self.plm.llm = self.db and self.plm.llm  # keep existing
        self.rag = RAGRetriever(self.db, new_domain)
        self.context_window.clear()
        return True


# ─────────────────────────────────────────────────────────────────────────────
# REPL
# ─────────────────────────────────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════════╗
║          CORTEX-PE  Perception Language Interface        ║
║          Language layer for the languageless NPU engine  ║
╚══════════════════════════════════════════════════════════╝
"""

HELP_TEXT = """
Commands
────────
  /domain <name>   Switch domain: bearing | cardiac | smap | recon
  /load <file>     Load latent from .npy file  (shape: [128] or [N,128])
  /random          Generate a random latent (for testing)
  /anomaly <score> Set anomaly score for current latent (0.0–1.0)
  /diff            Compare current vs previous latent
  /alert           Run domain-specific alert check
  /describe        Free-form description of current state
  /history [n]     Show last n Q&A pairs from DB (default 5)
  /stats           Show domain statistics from DB
  /clear           Clear conversation context window
  /verbose         Toggle verbose mode
  /help            Show this help
  /quit or Ctrl-C  Exit
"""

DOMAIN_COLORS = {
    "bearing": C.YELLOW,
    "cardiac": C.RED,
    "smap":    C.BLUE,
    "recon":   C.GREEN,
}


def print_banner(domain: str):
    print(fmt(BANNER, C.CYAN, bold=True))
    color = DOMAIN_COLORS.get(domain, C.MAG)
    print(fmt(f"  Active domain: {domain.upper()}", color, bold=True))
    print(fmt("  Type /help for commands, or ask a question directly.", C.DIM))
    print()


def print_response(text: str, domain: str):
    color = DOMAIN_COLORS.get(domain, C.MAG)
    print()
    print(fmt(f"  ● {domain.upper()}", color, bold=True))
    for line in text.split("\n"):
        print(wrap(line, width=88, indent="    ") if line.strip() else "")
    print()


def run_repl(engine: ConversationEngine, initial_z: Optional[np.ndarray] = None,
             initial_score: Optional[float] = None):
    domain = engine.plm.domain
    current_score: Optional[float] = initial_score
    verbose = False

    print_banner(domain)

    if initial_z is not None:
        engine.set_latent(initial_z, initial_score)
        print(fmt(f"  Latent loaded: shape={initial_z.shape}, "
                  f"norm={np.linalg.norm(initial_z):.3f}", C.DIM))
        if initial_score is not None:
            print(fmt(f"  Anomaly score: {initial_score:.4f}", C.DIM))
        print()

    rag_count = len(engine.rag)
    if rag_count > 0:
        print(fmt(f"  RAG memory: {rag_count} past observations loaded.", C.DIM))
        print()

    while True:
        domain = engine.plm.domain
        color = DOMAIN_COLORS.get(domain, C.MAG)
        prompt_str = fmt(f"[{domain}] ", color, bold=True) + fmt("▶ ", C.DIM)

        try:
            user_input = input(prompt_str).strip()
        except (EOFError, KeyboardInterrupt):
            print(fmt("\n  Session ended.", C.DIM))
            break

        if not user_input:
            continue

        # ── Commands ──────────────────────────────────────────────────────────

        if user_input.lower() in ("/quit", "/exit", "/q"):
            print(fmt("  Goodbye.", C.DIM))
            break

        elif user_input.lower() == "/help":
            print(fmt(HELP_TEXT, C.DIM))

        elif user_input.lower() == "/clear":
            engine.clear_context()
            print(fmt("  Context window cleared.", C.DIM))

        elif user_input.lower() == "/verbose":
            verbose = not verbose
            print(fmt(f"  Verbose: {'on' if verbose else 'off'}", C.DIM))
            engine.plm.verbose = verbose

        elif user_input.lower() == "/random":
            z = np.random.randn(128).astype(np.float32)
            engine.set_latent(z, current_score)
            print(fmt(f"  Random latent loaded. norm={np.linalg.norm(z):.3f}", C.DIM))

        elif user_input.lower().startswith("/anomaly "):
            try:
                score = float(user_input.split()[1])
                current_score = max(0.0, min(1.0, score))
                if engine.current_z is not None:
                    engine.set_latent(engine.current_z, current_score)
                print(fmt(f"  Anomaly score set: {current_score:.4f}", C.DIM))
            except (IndexError, ValueError):
                print(fmt("  Usage: /anomaly <0.0–1.0>", C.RED))

        elif user_input.lower().startswith("/load "):
            path = user_input.split(maxsplit=1)[1].strip()
            try:
                data = np.load(path)
                if data.ndim == 2:
                    z = data[-1].astype(np.float32)  # take last frame
                    print(fmt(f"  Loaded {data.shape[0]} latents, using last frame.", C.DIM))
                else:
                    z = data.astype(np.float32).flatten()[:128]
                engine.set_latent(z, current_score)
                print(fmt(f"  Latent loaded from {path}. norm={np.linalg.norm(z):.3f}", C.DIM))
            except Exception as e:
                print(fmt(f"  Error loading {path}: {e}", C.RED))

        elif user_input.lower().startswith("/domain "):
            new_domain = user_input.split()[1].lower()
            if engine.switch_domain(new_domain):
                # Re-apply same LLM backend
                engine.plm.llm = engine.plm._ollama \
                    if engine.plm._ollama.available else engine.plm._mock
                engine.plm._backend_name = (
                    f"ollama/{OllamaBackend.DEFAULT_MODEL}"
                    if engine.plm._ollama.available else "mock"
                )
                print(fmt(f"  Switched to domain: {new_domain.upper()}", C.DIM))
                print_banner(new_domain)
            else:
                print(fmt(f"  Unknown domain: {new_domain}. "
                           f"Choose: {list(DOMAIN_CONTEXTS.keys())}", C.RED))

        elif user_input.lower() == "/diff":
            print(fmt("  Computing latent diff...", C.DIM))
            result = engine.diff()
            print_response(result, engine.plm.domain)

        elif user_input.lower() == "/alert":
            print(fmt("  Running alert check...", C.DIM))
            result = engine.alert_check()
            if result:
                print_response(result, engine.plm.domain)

        elif user_input.lower() == "/describe":
            if engine.current_z is None:
                print(fmt("  No latent loaded. Use /random or /load.", C.RED))
                continue
            print(fmt("  Generating description...", C.DIM))
            result = engine.plm.describe(engine.current_z, current_score, save=True)
            engine.rag.add(engine.current_z, result, "describe", current_score or 0.0)
            print_response(result, engine.plm.domain)

        elif user_input.lower().startswith("/history"):
            parts = user_input.split()
            n = int(parts[1]) if len(parts) > 1 else 5
            history = engine.db.recent(engine.plm.domain, n=n)
            if not history:
                print(fmt("  No history yet in this domain.", C.DIM))
            for entry in history:
                ts = time.strftime("%H:%M:%S", time.localtime(entry["ts"]))
                score_str = (f" [{entry['anomaly_score']:.2f}]"
                             if entry.get("anomaly_score") else "")
                print(fmt(f"\n  [{ts}]{score_str}", C.DIM))
                if entry.get("question"):
                    print(fmt(f"  Q: {entry['question']}", C.CYAN))
                print(wrap(entry["answer"][:300], width=84, indent="  A: "))

        elif user_input.lower() == "/stats":
            stats = engine.db.stats()
            print()
            for dom, s in sorted(stats.items()):
                col = DOMAIN_COLORS.get(dom, C.MAG)
                print(fmt(f"  {dom:10s}", col, bold=True)
                      + fmt(f"  {s['n']:4d} observations  "
                            f"avg_anomaly={s['avg_anomaly']:.3f}", C.DIM))
            rag_info = f"\n  RAG index: {len(engine.rag)} vectors loaded"
            print(fmt(rag_info, C.DIM))
            print()

        # ── Natural language question ─────────────────────────────────────────

        else:
            if engine.current_z is None:
                # Answer from domain knowledge only
                print(fmt("  (No latent loaded — answering from domain knowledge only)", C.DIM))

            thinking = fmt("  Thinking...", C.DIM)
            print(thinking, end="\r", flush=True)

            response = engine.chat(user_input, anomaly_score=current_score)
            print(" " * len(thinking), end="\r")  # clear "Thinking..."
            print_response(response, engine.plm.domain)

    engine.db.close()


# ─────────────────────────────────────────────────────────────────────────────
# Demo script — scripted conversation showing all capabilities
# ─────────────────────────────────────────────────────────────────────────────

def run_demo(db_path: str = "language_memory_demo.db"):
    """
    Scripted demo session with synthetic latents.
    Shows what the REPL would look like with real NPU data.
    """
    print(fmt("\n=== CORTEX-PE PerceptionLLM Demo ===\n", C.CYAN, bold=True))

    db = LanguageMemoryDB(db_path)
    rng = np.random.default_rng(99)

    scenarios = [
        # (domain, anomaly_score, question, description)
        ("bearing", 0.83,
         "What fault type does this look like?",
         "High anomaly — outer race fault signature"),
        ("bearing", 0.12,
         "Is this bearing healthy?",
         "Low anomaly — normal operation"),
        ("cardiac", 0.79,
         "Does this heart sound show any murmur?",
         "Elevated cardiac anomaly"),
        ("smap", 0.67,
         "Which satellite subsystem is affected?",
         "SMAP telemetry anomaly"),
        ("recon", 0.35,
         "Can the robot proceed on this path?",
         "Low scene change — safe navigation"),
    ]

    for domain, score, question, desc in scenarios:
        plm = PerceptionLLM(domain=domain, db_path=db_path, prefer_local=False)
        plm.llm = MockBackend()
        plm._backend_name = "mock"

        z = rng.standard_normal(128).astype(np.float32)
        z = z / np.linalg.norm(z)

        color = DOMAIN_COLORS.get(domain, C.MAG)
        print(fmt(f"\n[{domain.upper()}] score={score:.2f}  —  {desc}", color, bold=True))
        print(fmt(f"  Q: {question}", C.CYAN))

        answer = plm.ask(z, question, anomaly_score=score, save=True)
        print(wrap(answer, width=84, indent="  A: "))
        plm.close()

    # Show accumulated stats
    print(fmt("\n\n--- Database Stats After Demo ---", C.DIM))
    stats = db.stats()
    for dom, s in sorted(stats.items()):
        col = DOMAIN_COLORS.get(dom, C.MAG)
        print(fmt(f"  {dom:10s}", col, bold=True)
              + fmt(f"  {s['n']} observations, avg_anomaly={s['avg_anomaly']:.3f}", C.DIM))

    db.close()
    import os
    os.unlink(db_path)
    print(fmt("\n✅ Demo complete. Run without --demo to start interactive REPL.\n", C.GREEN))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CORTEX-PE Conversational REPL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python cortex_repl.py --domain bearing
          python cortex_repl.py --domain cardiac --anomaly-score 0.85
          python cortex_repl.py --domain recon --latent-file saved.npy
          python cortex_repl.py --demo
        """)
    )
    parser.add_argument("--domain", choices=list(DOMAIN_CONTEXTS),
                        default="bearing", help="Perception domain")
    parser.add_argument("--db", default="language_memory.db",
                        help="Path to LanguageMemoryDB")
    parser.add_argument("--latent-file", type=str, default=None,
                        help=".npy file containing 128-D latent(s)")
    parser.add_argument("--anomaly-score", type=float, default=None,
                        help="Initial anomaly score (0.0–1.0)")
    parser.add_argument("--backend", choices=["local", "api", "mock"],
                        default="local", help="LLM backend preference")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--demo", action="store_true",
                        help="Run scripted demo instead of interactive REPL")
    args = parser.parse_args()

    if args.demo:
        run_demo()
        return

    # Build PerceptionLLM
    plm = PerceptionLLM(
        domain=args.domain,
        db_path=args.db,
        prefer_local=(args.backend != "api"),
        verbose=args.verbose,
    )
    if args.backend == "mock":
        plm.llm = MockBackend()
        plm._backend_name = "mock"

    db = plm.db
    engine = ConversationEngine(plm, db)

    # Load initial latent
    initial_z = None
    if args.latent_file:
        try:
            data = np.load(args.latent_file)
            initial_z = data[-1] if data.ndim == 2 else data.flatten()[:128]
        except Exception as e:
            print(fmt(f"Warning: could not load {args.latent_file}: {e}", C.RED))
    else:
        # Start with a random latent so the REPL is immediately useful
        initial_z = np.random.randn(128).astype(np.float32)
        print(fmt("  No latent file specified — using random latent. "
                  "Use /load <file> or /random to change.", C.DIM))

    run_repl(engine, initial_z=initial_z, initial_score=args.anomaly_score)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No args: run demo
        run_demo()
    else:
        main()
