"""
CORTEX-PE Language Stack Startup
==================================
Single entry point for the full language layer.

Checks dependencies, starts the browser in background,
then drops into the domain REPL. Works on Windows (NUC).

Usage:
    python cortex_language_startup.py                  # bearing REPL (default)
    python cortex_language_startup.py --domain cardiac
    python cortex_language_startup.py --browser-only   # just the web UI
    python cortex_language_startup.py --train bearing  # run JEPA trainer only
    python cortex_language_startup.py --check          # dependency check only
"""

import sys, os, time, argparse, subprocess, threading, socket
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# â”€â”€ colour helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
G = "\033[32m"; Y = "\033[33m"; R = "\033[31m"; C = "\033[36m"; D = "\033[2m"; X = "\033[0m"
def ok(m):   print(f"  {G}âœ…{X}  {m}")
def warn(m): print(f"  {Y}âš ï¸ {X}  {m}")
def err(m):  print(f"  {R}âŒ{X}  {m}")
def info(m): print(f"  {C}â„¹ï¸ {X}  {m}")

BANNER = f"""
{C}â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
â•‘     CORTEX-PE  Language Stack  v1.0              â•‘
â•‘     NPU perception â†’ natural language interface  â•‘
â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•{X}
"""

# â”€â”€ dependency check â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def check_deps() -> dict:
    results = {}

    # numpy
    try:
        import numpy; results["numpy"] = numpy.__version__; ok(f"numpy {numpy.__version__}")
    except ImportError:
        results["numpy"] = None; err("numpy missing  â†’  pip install numpy")

    # torch
    try:
        import torch; results["torch"] = torch.__version__; ok(f"torch {torch.__version__}")
    except ImportError:
        results["torch"] = None; warn("torch missing (JEPA trainer disabled)  â†’  pip install torch")

    # sqlite3 (stdlib)
    import sqlite3; results["sqlite3"] = sqlite3.sqlite_version
    ok(f"sqlite3 {sqlite3.sqlite_version}")

    # perception_llm module
    try:
        from perception_llm import PerceptionLLM
        results["perception_llm"] = True; ok("perception_llm.py")
    except ImportError as e:
        results["perception_llm"] = None; err(f"perception_llm.py not found: {e}")

    # ollama (optional local LLM)
    ollama_url = "http://localhost:11434"
    try:
        import urllib.request
        urllib.request.urlopen(f"{ollama_url}/api/tags", timeout=2)
        results["ollama"] = True; ok("ollama running  â†’  local LLM active")
    except Exception:
        results["ollama"] = False
        warn("ollama not running  â†’  will use MockBackend (template responses)")
        info("  To enable: winget install Ollama.Ollama && ollama pull qwen3:0.6b")

    # anthropic API key (optional fallback)
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key:
        results["anthropic"] = True; ok("ANTHROPIC_API_KEY set  â†’  API fallback available")
    else:
        results["anthropic"] = False
        info("No ANTHROPIC_API_KEY  (optional â€” ollama preferred)")

    return results

# â”€â”€ browser startup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def port_free(port: int) -> bool:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False

def start_browser(db_path: str, port: int = 7860):
    """Start the language memory browser in a daemon thread."""
    from language_memory_browser import Handler, HTTPServer
    if not port_free(port):
        warn(f"Port {port} in use â€” browser not started (may already be running)")
        return None

    Handler.db_path = db_path
    server = HTTPServer(("0.0.0.0", port), Handler)

    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    ok(f"Language Memory Browser  â†’  http://localhost:{port}")
    return server

# â”€â”€ JEPA trainer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_trainer(domain: str, db_path: str, epochs: int = 30):
    try:
        from jepa_align_trainer import JEPAAlignTrainer, evaluate_alignment
        from perception_llm import LanguageMemoryDB
    except ImportError as e:
        err(f"Trainer unavailable: {e}"); return

    db = LanguageMemoryDB(db_path)
    trainer = JEPAAlignTrainer(db, domain)

    if trainer.dataset.N < 16:
        warn(f"Only {trainer.dataset.N} samples for '{domain}' â€” need â‰¥16.")
        info("Run the REPL first to accumulate observations, then retrain.")
        db.close(); return

    print(f"\n  Training projector for '{domain}'  ({trainer.dataset.N} samples)...")
    ckpt = trainer.run(epochs=epochs, batch_size=min(128, trainer.dataset.N))
    result = evaluate_alignment(trainer.projector, trainer.dataset)
    print(f"\n  Alignment: gap={result['alignment_gap']}  status={result['status']}")
    ok(f"Checkpoint saved â†’ {ckpt}")
    db.close()

# â”€â”€ main REPL launcher â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def launch_repl(domain: str, db_path: str, deps: dict,
                projector_path: str = None, anomaly_score: float = None):
    from perception_llm import PerceptionLLM, MockBackend, OllamaBackend
    from cortex_repl import ConversationEngine, run_repl
    import numpy as np

    # Pick LLM backend
    plm = PerceptionLLM(domain=domain, db_path=db_path,
                        prefer_local=deps.get("ollama", False),
                        projector_path=projector_path)

    if not deps.get("ollama"):
        plm.llm = MockBackend()
        plm._backend_name = "mock"
        warn("Using MockBackend â€” answers are template-based.")
        info("For real answers: ollama pull qwen3:0.6b")

    db = plm.db
    engine = ConversationEngine(plm, db)

    # Start with random latent (user can /load a real one)
    z0 = np.random.randn(128).astype("f4")
    run_repl(engine, initial_z=z0, initial_score=anomaly_score)

# â”€â”€ entry point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main():
    parser = argparse.ArgumentParser(description="CORTEX-PE Language Stack")
    parser.add_argument("--domain", default="bearing",
                        choices=["bearing","cardiac","smap","recon"],
                        help="Perception domain for the REPL")
    parser.add_argument("--db", default="language_memory.db")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--browser-only", action="store_true")
    parser.add_argument("--train", metavar="DOMAIN",
                        help="Run JEPA trainer for DOMAIN then exit")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--anomaly-score", type=float, default=None)
    parser.add_argument("--projector", type=str, default=None,
                        help="Path to trained projector checkpoint")
    parser.add_argument("--check", action="store_true",
                        help="Dependency check only, then exit")
    args = parser.parse_args()

    print(BANNER)

    print(f"{D}  Checking dependencies...{X}\n")
    deps = check_deps()
    print()

    # Auto-migrate any existing DB with BLOB scores (safe no-op if already clean)
    if Path(args.db).exists():
        try:
            from perception_llm import LanguageMemoryDB
            _mdb = LanguageMemoryDB(args.db)
            n_fixed = _mdb.migrate_scores()
            if n_fixed:
                ok(f"DB migration: fixed {n_fixed} BLOB scores â†’ REAL")
            _mdb.close()
        except Exception:
            pass

    if args.check:
        return

    # Trainer mode
    if args.train:
        run_trainer(args.train, args.db, args.epochs)
        return

    # Start browser
    server = None
    if not args.no_browser:
        server = start_browser(args.db, args.port)
        if server:
            # Open in browser after 0.5s
            def _open():
                time.sleep(0.5)
                import webbrowser
                webbrowser.open(f"http://localhost:{args.port}")
            threading.Thread(target=_open, daemon=True).start()

    if args.browser_only:
        if server:
            info(f"Browser running at http://localhost:{args.port}  (Ctrl-C to stop)")
            try:
                while True: time.sleep(1)
            except KeyboardInterrupt:
                print("\n  Stopped.")
        return

    # Auto-detect latest projector checkpoint
    projector = args.projector
    if projector is None:
        latest = Path(f"checkpoints/projector/{args.domain}_projector_latest.pt")
        if latest.exists():
            projector = str(latest)
            ok(f"Projector loaded â†’ {projector}")

    print()
    info(f"Starting {args.domain.upper()} REPL...")
    print()

    try:
        launch_repl(args.domain, args.db, deps,
                    projector_path=projector,
                    anomaly_score=args.anomaly_score)
    except KeyboardInterrupt:
        pass

    if server:
        server.shutdown()

if __name__ == "__main__":
    main()

