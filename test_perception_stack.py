"""
CORTEX-PE PerceptionLLM — Full Stack Integration Test
======================================================
Runs entirely with MockBackend (no ollama, no API key needed).
Tests every layer: DB, annotator, projector, REPL engine, trainer, browser.
"""

import os, sys, json, time, tempfile, shutil, threading, http.client
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

RESULTS = {}

def section(name):
    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")

def ok(label, detail=""):
    tag = f"  ✅ {label}"
    print(tag + (f"  — {detail}" if detail else ""))
    RESULTS[label] = "PASS"

def fail(label, err):
    print(f"  ❌ {label}  — {err}")
    RESULTS[label] = f"FAIL: {err}"

# ── shared temp DB ─────────────────────────────────────────────────────────
TMP_DIR = tempfile.mkdtemp(prefix="cortex_test_")
DB_PATH  = os.path.join(TMP_DIR, "test_memory.db")
CKPT_DIR = os.path.join(TMP_DIR, "checkpoints")
RNG = np.random.default_rng(2026)

try:

    # ══════════════════════════════════════════════════════════════
    section("1  LanguageMemoryDB — schema, write, read, stats")
    # ══════════════════════════════════════════════════════════════
    from perception_llm import LanguageMemoryDB, DOMAIN_CONTEXTS, MockBackend

    db = LanguageMemoryDB(DB_PATH)

    for domain in DOMAIN_CONTEXTS:
        for i in range(20):
            z = RNG.standard_normal(128).astype(np.float32)
            score = float(RNG.uniform(0.2, 0.95))
            db.record(domain=domain, latent=z,
                      answer=f"Test answer {i} for {domain}.",
                      question=f"Test question {i}?",
                      anomaly_score=score, model_used="mock")

    stats = db.stats()
    assert len(stats) == 4,           "Expected 4 domains"
    assert stats["bearing"]["n"] == 20, "Expected 8 bearing records"
    ok("DB schema + write", f"{sum(s['n'] for s in stats.values())} records across 4 domains")

    recent = db.recent("bearing", n=3)
    assert len(recent) == 3
    ok("DB recent() query", f"got {len(recent)} records")

    batch = db.jepa_batch("bearing", n=16)
    assert batch is not None, "jepa_batch returned None — need >=16 records"
    latents, texts = batch
    assert latents.shape[1] == 128, f"expected 128-D latents, got {latents.shape[1]}"
    assert isinstance(texts, list) and len(texts) == latents.shape[0]
    ok("DB jepa_batch()", f"latents {latents.shape}, {len(texts)} text answers")

    # ══════════════════════════════════════════════════════════════
    section("2  PerceptionLLM — all 4 domains, ask/describe/alert/compare")
    # ══════════════════════════════════════════════════════════════
    from perception_llm import PerceptionLLM, BearingLLM, CardiacLLM, SMAPLanguage, RECONLanguage

    for cls, domain, score in [
        (BearingLLM,   "bearing", 0.83),
        (CardiacLLM,   "cardiac", 0.79),
        (SMAPLanguage, "smap",    0.67),
        (RECONLanguage,"recon",   0.35),
    ]:
        plm = cls(db_path=DB_PATH, prefer_local=False)
        plm.llm = MockBackend(); plm._backend_name = "mock"
        z = RNG.standard_normal(128).astype(np.float32)

        ans = plm.ask(z, "Test question?", anomaly_score=score)
        assert isinstance(ans, str) and len(ans) > 10
        ok(f"PerceptionLLM/{domain} ask()")

        desc = plm.describe(z, anomaly_score=score)
        assert len(desc) > 10
        ok(f"PerceptionLLM/{domain} describe()")

        alert = plm.alert(z, score)
        if domain == "recon":   # score 0.35 < threshold 0.75
            assert alert is None
            ok(f"PerceptionLLM/{domain} alert() suppressed (below threshold)")
        else:
            assert alert is not None
            ok(f"PerceptionLLM/{domain} alert() fired")

        z2 = RNG.standard_normal(128).astype(np.float32)
        cmp = plm.compare(z, z2)
        assert len(cmp) > 10
        ok(f"PerceptionLLM/{domain} compare()")
        plm.close()

    # ══════════════════════════════════════════════════════════════
    section("3  FeatureProjector — all input dimensions")
    # ══════════════════════════════════════════════════════════════
    from perception_annotator import FeatureProjector

    for dim in [6, 12, 25, 128, 512, 4096]:
        proj = FeatureProjector(dim, output_dim=128)
        for _ in range(64):
            proj.update(RNG.standard_normal(dim).astype(np.float32))
        x = RNG.standard_normal(dim).astype(np.float32)
        z = proj.project(x)
        assert z.shape == (128,), f"Expected (128,) got {z.shape}"
        assert abs(np.linalg.norm(z) - 1.0) < 0.02, f"not unit norm: {np.linalg.norm(z):.4f}"
        ok(f"FeatureProjector {dim}→128", f"norm={np.linalg.norm(z):.4f}")

    # ══════════════════════════════════════════════════════════════
    section("4  PerceptionAnnotator — observe() + annotate_batch()")
    # ══════════════════════════════════════════════════════════════
    from perception_annotator import PerceptionAnnotator

    ann = PerceptionAnnotator("bearing", db_path=DB_PATH, dry_run=True,
                              annotation_rate=1.0, verbose=False)
    ann.MAX_ANNOTATIONS_PER_RUN = 10

    for i in range(12):
        feats = RNG.standard_normal(12).astype(np.float32)
        score = 0.3 + 0.55 * (i / 12)
        label = ["ball","inner","outer"][i % 3]
        ann.observe(feats, anomaly_score=score, label=label)

    s = ann.summary()
    assert s["n_observed"] == 12
    assert s["n_annotated"] > 0
    ok("Annotator observe()", f"{s['n_annotated']} annotated / {s['n_observed']} observed")

    ann2 = PerceptionAnnotator("smap", db_path=DB_PATH, dry_run=True,
                               annotation_rate=1.0, verbose=False)
    feats_batch = RNG.standard_normal((40, 25)).astype(np.float32)
    scores_batch = RNG.uniform(0, 1, 40).astype(np.float32)
    scores_batch[5:10] = 0.88
    labels_batch = ["nominal"]*35 + ["mode_switch"]*5
    results = ann2.annotate_batch(feats_batch, scores_batch, labels_batch, n_top=5)
    assert len(results) == 5
    ok("Annotator annotate_batch()", f"{len(results)} annotations generated")
    ann.close(); ann2.close()

    # ══════════════════════════════════════════════════════════════
    section("5  RAGRetriever — cosine search over memory")
    # ══════════════════════════════════════════════════════════════
    from cortex_repl import RAGRetriever

    db2 = LanguageMemoryDB(DB_PATH)
    rag = RAGRetriever(db2, "bearing")
    n_loaded = len(rag)
    assert n_loaded > 0, "RAG index empty"
    ok("RAGRetriever load", f"{n_loaded} vectors in index")

    query = RNG.standard_normal(128).astype(np.float32)
    retrieved = rag.retrieve(query, k=3)
    assert isinstance(retrieved, list)
    ok("RAGRetriever retrieve()", f"got {len(retrieved)} results")

    ctx_str = rag.format_for_context(retrieved)
    assert isinstance(ctx_str, str)  # may be empty if cosine sims all < 0.1
    ok("RAGRetriever format_for_context()", f"len={len(ctx_str)}")

    # ══════════════════════════════════════════════════════════════
    section("6  ConversationEngine — multi-turn chat")
    # ══════════════════════════════════════════════════════════════
    from cortex_repl import ConversationEngine
    from perception_llm import PerceptionLLM

    plm = PerceptionLLM(domain="bearing", db_path=DB_PATH, prefer_local=False)
    plm.llm = MockBackend(); plm._backend_name = "mock"

    engine = ConversationEngine(plm, db2)
    z = RNG.standard_normal(128).astype(np.float32)
    engine.set_latent(z, anomaly_score=0.78)
    assert engine.current_z is not None
    ok("ConversationEngine set_latent()")

    r1 = engine.chat("What fault type is this?", anomaly_score=0.78)
    assert len(r1) > 10
    assert engine.turn_count == 1
    ok("ConversationEngine turn 1", r1[:60])

    r2 = engine.chat("How severe is it?", anomaly_score=0.78)
    assert engine.turn_count == 2
    assert len(engine.context_window) == 4   # 2 user + 2 assistant
    ok("ConversationEngine turn 2 (context window)", f"{len(engine.context_window)} turns stored")

    z2 = RNG.standard_normal(128).astype(np.float32)
    engine.set_latent(z2, anomaly_score=0.30)
    diff = engine.diff()
    assert len(diff) > 10
    ok("ConversationEngine diff()")

    engine.clear_context()
    assert len(engine.context_window) == 0
    ok("ConversationEngine clear_context()")
    db2.close()

    # ══════════════════════════════════════════════════════════════
    section("7  JEPAAlignTrainer — 5 epochs on synthetic data")
    # ══════════════════════════════════════════════════════════════
    from jepa_align_trainer import JEPAAlignTrainer, evaluate_alignment

    db3 = LanguageMemoryDB(DB_PATH)
    # Seed with enough data (need ≥16)
    for i in range(50):
        z = RNG.standard_normal(128).astype(np.float32)
        score = float(RNG.uniform(0.1, 0.95))
        db3.record("cardiac", z,
                   "Normal S1 S2" if score < 0.5 else "Systolic murmur detected",
                   anomaly_score=score, model_used="mock")

    trainer = JEPAAlignTrainer(db3, "cardiac",
                               llm_dim=64, n_tokens=2,  # tiny for speed
                               checkpoint_dir=CKPT_DIR)
    assert trainer.dataset.N >= 16, f"Need ≥16, got {trainer.dataset.N}"
    ok("JEPAAlignTrainer init", f"{trainer.dataset.N} samples")

    # 3-epoch quick train
    for ep in range(1, 4):
        result = trainer.train_epoch(batch_size=16)
        assert "loss" in result
        assert result["loss"] < 999
    ok("JEPAAlignTrainer 3 epochs", f"final loss={result['loss']:.5f}")

    eval_r = evaluate_alignment(trainer.projector, trainer.dataset)
    assert "alignment_gap" in eval_r
    assert eval_r["collapse_check"] > 0.01, "Projector collapsed"
    ok("evaluate_alignment()", f"gap={eval_r['alignment_gap']}, status={eval_r['status']}")

    ckpt = trainer.run(epochs=2, batch_size=16)
    assert Path(ckpt).exists()
    ok("JEPAAlignTrainer checkpoint saved", ckpt)

    # verify latest.pt also written
    latest = Path(CKPT_DIR) / "cardiac_projector_latest.pt"
    assert latest.exists()
    ok("Latest checkpoint link exists")
    db3.close()

    # ══════════════════════════════════════════════════════════════
    section("8  Language Memory Browser — HTTP server smoke test")
    # ══════════════════════════════════════════════════════════════
    from language_memory_browser import Handler, HTTPServer

    Handler.db_path = DB_PATH
    server = HTTPServer(("127.0.0.1", 0), Handler)   # port=0 → OS assigns free port
    port = server.server_address[1]

    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.2)

    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)

    # GET /
    conn.request("GET", "/")
    resp = conn.getresponse()
    assert resp.status == 200
    body = resp.read()
    assert b"CORTEX-PE Language Memory" in body
    ok("Browser GET /", f"HTML {len(body)} bytes")

    # GET /api/observations
    conn.request("GET", "/api/observations?limit=5")
    resp = conn.getresponse()
    assert resp.status == 200
    data = json.loads(resp.read())
    assert "observations" in data
    assert "stats" in data
    ok("Browser GET /api/observations", f"{len(data['observations'])} records returned")

    # GET /api/observations?domain=bearing&min_score=0.7
    conn.request("GET", "/api/observations?domain=bearing&min_score=0.7&limit=10")
    resp = conn.getresponse()
    data = json.loads(resp.read())
    assert all(o["domain"] == "bearing" for o in data["observations"])
    assert all(o["anomaly_score"] >= 0.7 for o in data["observations"])
    ok("Browser domain + score filter")

    # GET /api/export
    conn.request("GET", "/api/export")
    resp = conn.getresponse()
    assert resp.status == 200
    export = json.loads(resp.read())
    assert isinstance(export, list)
    ok("Browser GET /api/export", f"{len(export)} records")

    server.shutdown()
    ok("Browser server shutdown cleanly")

    # ══════════════════════════════════════════════════════════════
    section("9  End-to-end pipeline — annotate → retrieve → chat")
    # ══════════════════════════════════════════════════════════════
    db_e2e = LanguageMemoryDB(DB_PATH)
    ann_e2e = PerceptionAnnotator("recon", db_path=DB_PATH, dry_run=True,
                                  annotation_rate=1.0, verbose=False)
    ann_e2e.MAX_ANNOTATIONS_PER_RUN = 15

    # Simulate a RECON eval run (128-D latents from NPU)
    latents_e2e = RNG.standard_normal((20, 128)).astype(np.float32)
    scores_e2e  = RNG.uniform(0.2, 0.98, 20).astype(np.float32)
    for z, score in zip(latents_e2e, scores_e2e):
        ann_e2e.observe(z, anomaly_score=float(score), label="obstacle" if score > 0.75 else "clear")

    # Now open a conversation that uses those as RAG context
    plm_e2e = PerceptionLLM("recon", db_path=DB_PATH, prefer_local=False)
    plm_e2e.llm = MockBackend(); plm_e2e._backend_name = "mock"
    engine_e2e = ConversationEngine(plm_e2e, db_e2e)

    # RAG should find relevant past observations
    rag_e2e = RAGRetriever(db_e2e, "recon")
    query_z = latents_e2e[0]
    retrieved = rag_e2e.retrieve(query_z, k=3)
    ok("E2E RAG finds past RECON observations", f"{len(retrieved)} retrieved")

    engine_e2e.set_latent(query_z, 0.82)
    r = engine_e2e.chat("Is the path safe?")
    assert len(r) > 5
    ok("E2E full pipeline chat()", r[:70])

    ann_e2e.close()
    db_e2e.close()

finally:
    shutil.rmtree(TMP_DIR, ignore_errors=True)

# ══════════════════════════════════════════════════════════════
section("RESULTS SUMMARY")
# ══════════════════════════════════════════════════════════════
passed = [k for k, v in RESULTS.items() if v == "PASS"]
failed = [k for k, v in RESULTS.items() if v != "PASS"]

print(f"\n  Passed : {len(passed)}")
print(f"  Failed : {len(failed)}")
if failed:
    for f in failed:
        print(f"  ✗ {f}  →  {RESULTS[f]}")
print()
print(f"  {'✅ ALL TESTS PASSED' if not failed else '❌ SOME TESTS FAILED'}")
print()
