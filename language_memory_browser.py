"""
CORTEX-PE Language Memory Browser
====================================

Lightweight HTTP server that serves a web dashboard for browsing
the language_memory.db — all annotated observations across all domains.

No external dependencies beyond stdlib and numpy.
Opens in any browser at http://localhost:7860

Features
--------
  • Browse annotations by domain with anomaly score filtering
  • Search by keyword in answers
  • Latent similarity search — find observations similar to a queried one
  • Export filtered results as JSON
  • Live refresh — new annotations appear without page reload

Usage
-----
  python language_memory_browser.py
  python language_memory_browser.py --db language_memory.db --port 7860
"""

import sys
import json
import time
import sqlite3
import argparse
import threading
import webbrowser
import numpy as np
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import struct

def _safe_float(v):
    """Unpack np.float32 BLOB scores that SQLite stores as 4-byte bytes."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, bytes) and len(v) == 4:
        return float(struct.unpack('<f', v)[0])
    if isinstance(v, bytes) and len(v) == 8:
        return float(struct.unpack('<d', v)[0])
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


sys.path.insert(0, str(Path(__file__).parent))
from perception_llm import LanguageMemoryDB, DOMAIN_CONTEXTS

DB_PATH = "language_memory.db"

# ─────────────────────────────────────────────────────────────────────────────
# HTML / JS dashboard (single-file, inline)
# ─────────────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CORTEX-PE Language Memory</title>
<style>
  :root {
    --bg: #0d0f14; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --dim: #8b949e; --cyan: #58a6ff;
    --green: #3fb950; --yellow: #d29922; --red: #f85149;
    --blue: #388bfd; --purple: #bc8cff;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: ui-monospace,
    "SFMono-Regular", monospace; font-size: 13px; }
  header { background: var(--surface); border-bottom: 1px solid var(--border);
    padding: 12px 20px; display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 15px; color: var(--cyan); font-weight: 600; }
  header span { color: var(--dim); font-size: 11px; }
  .controls { padding: 12px 20px; display: flex; gap: 10px; flex-wrap: wrap;
    border-bottom: 1px solid var(--border); background: var(--surface); }
  .controls select, .controls input { background: var(--bg);
    color: var(--text); border: 1px solid var(--border); border-radius: 4px;
    padding: 5px 8px; font-family: inherit; font-size: 12px; }
  .controls button { background: var(--cyan); color: #000; border: none;
    border-radius: 4px; padding: 5px 12px; cursor: pointer; font-size: 12px;
    font-weight: 600; }
  .controls button:hover { opacity: 0.85; }
  .stats-bar { padding: 8px 20px; display: flex; gap: 20px; flex-wrap: wrap;
    border-bottom: 1px solid var(--border); font-size: 11px; color: var(--dim); }
  .stats-bar span b { color: var(--text); }
  #cards { padding: 16px 20px; display: grid;
    grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 12px; }
  .card { background: var(--surface); border: 1px solid var(--border);
    border-radius: 6px; padding: 12px; position: relative; }
  .card-header { display: flex; justify-content: space-between;
    margin-bottom: 8px; font-size: 11px; }
  .domain-tag { padding: 2px 7px; border-radius: 3px; font-weight: 600;
    font-size: 10px; text-transform: uppercase; }
  .domain-bearing { background: #332200; color: var(--yellow); }
  .domain-cardiac  { background: #2d0a0a; color: var(--red); }
  .domain-smap     { background: #0a1a2d; color: var(--blue); }
  .domain-recon    { background: #0a2210; color: var(--green); }
  .score-bar { height: 3px; border-radius: 2px; margin-bottom: 8px; }
  .question { color: var(--cyan); margin-bottom: 5px; font-size: 11px; }
  .answer { color: var(--text); line-height: 1.5; font-size: 12px;
    white-space: pre-wrap; word-break: break-word; }
  .meta { color: var(--dim); font-size: 10px; margin-top: 8px; }
  .sim-badge { position: absolute; top: 8px; right: 8px; font-size: 10px;
    background: var(--purple); color: #000; padding: 1px 5px;
    border-radius: 3px; font-weight: 600; }
  #loading { text-align: center; padding: 40px; color: var(--dim); }
  #empty { text-align: center; padding: 40px; color: var(--dim); display: none; }
  .score-high { background: var(--red); }
  .score-mid  { background: var(--yellow); }
  .score-low  { background: var(--green); }
</style>
</head>
<body>

<header>
  <h1>⬡ CORTEX-PE Language Memory</h1>
  <span id="db-path"></span>
  <span id="last-refresh" style="margin-left:auto"></span>
</header>

<div class="controls">
  <select id="domain-filter">
    <option value="">All domains</option>
    <option value="bearing">Bearing</option>
    <option value="cardiac">Cardiac</option>
    <option value="smap">SMAP</option>
    <option value="recon">RECON</option>
  </select>
  <input id="min-score" type="number" placeholder="Min score" min="0" max="1"
         step="0.05" style="width:110px">
  <input id="search" type="text" placeholder="Search answers…" style="width:200px">
  <input id="limit" type="number" value="50" min="5" max="500" style="width:70px">
  <button onclick="load()">Refresh</button>
  <button onclick="exportJSON()" style="background:var(--dim);color:#000">Export JSON</button>
</div>

<div class="stats-bar" id="stats-bar">Loading…</div>

<div id="cards"><div id="loading">Loading observations…</div></div>
<div id="empty">No observations match the current filters.</div>

<script>
const DOMAIN_COLORS = {
  bearing: '#d29922', cardiac: '#f85149', smap: '#388bfd', recon: '#3fb950'
};

function scoreClass(s) {
  if (s >= 0.75) return 'score-high';
  if (s >= 0.50) return 'score-mid';
  return 'score-low';
}

function fmtTime(ts) {
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString();
}

async function load() {
  const domain = document.getElementById('domain-filter').value;
  const minScore = document.getElementById('min-score').value || 0;
  const search   = document.getElementById('search').value;
  const limit    = document.getElementById('limit').value || 50;

  let url = `/api/observations?limit=${limit}&min_score=${minScore}`;
  if (domain) url += `&domain=${domain}`;
  if (search)  url += `&search=${encodeURIComponent(search)}`;

  const resp = await fetch(url);
  const data = await resp.json();

  renderStats(data.stats);
  renderCards(data.observations);
  document.getElementById('last-refresh').textContent =
    'Refreshed ' + new Date().toLocaleTimeString();
  document.getElementById('db-path').textContent = data.db_path;
}

function renderStats(stats) {
  const bar = document.getElementById('stats-bar');
  if (!stats || Object.keys(stats).length === 0) {
    bar.innerHTML = 'No observations yet. Run eval scripts or the REPL to populate.';
    return;
  }
  let html = '';
  for (const [domain, s] of Object.entries(stats)) {
    const col = DOMAIN_COLORS[domain] || '#8b949e';
    html += `<span style="color:${col}"><b>${domain}</b> ${s.n} obs  avg_score=${s.avg_anomaly.toFixed(3)}</span>`;
  }
  bar.innerHTML = html;
}

function renderCards(obs) {
  const container = document.getElementById('cards');
  const empty = document.getElementById('empty');
  container.innerHTML = '';

  if (!obs || obs.length === 0) {
    empty.style.display = 'block';
    return;
  }
  empty.style.display = 'none';

  obs.forEach(o => {
    const score = o.anomaly_score || 0;
    const pct = Math.round(score * 100);
    const col = DOMAIN_COLORS[o.domain] || '#8b949e';
    container.innerHTML += `
      <div class="card">
        <div class="card-header">
          <span class="domain-tag domain-${o.domain}">${o.domain}</span>
          <span style="color:var(--dim)">${fmtTime(o.timestamp)}</span>
          <span style="color:${col}">${score.toFixed(3)}</span>
        </div>
        <div class="score-bar ${scoreClass(score)}" style="width:${pct}%"></div>
        ${o.question ? `<div class="question">Q: ${escHtml(o.question)}</div>` : ''}
        <div class="answer">${escHtml(o.answer.substring(0, 400))}${o.answer.length > 400 ? '…' : ''}</div>
        <div class="meta">model: ${o.model_used || '?'}  |  tokens: ${o.tokens_used || '?'}</div>
      </div>`;
  });
}

function escHtml(s) {
  return (s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

async function exportJSON() {
  const url = document.getElementById('domain-filter').value;
  const r = await fetch('/api/export');
  const blob = await r.blob();
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'language_memory_export.json';
  a.click();
}

// Auto-refresh every 30s
load();
setInterval(load, 30000);
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Request handler
# ─────────────────────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):

    db_path: str = DB_PATH

    def log_message(self, fmt, *args):
        pass  # suppress default access log

    def _conn(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)

        if parsed.path == "/":
            self._send(200, "text/html", HTML.encode())

        elif parsed.path == "/api/observations":
            domain    = qs.get("domain", [None])[0]
            min_score = float(qs.get("min_score", [0])[0])
            search    = qs.get("search", [None])[0]
            limit     = int(qs.get("limit", [50])[0])

            conn = self._conn()
            where = ["1=1", f"anomaly_score >= {min_score}"]
            if domain:
                where.append(f"domain = '{domain}'")
            if search:
                safe = search.replace("'", "''")
                where.append(f"answer LIKE '%{safe}%'")

            rows = conn.execute(f"""
                SELECT timestamp, domain, anomaly_score, question, answer,
                       model_used, tokens_used
                FROM descriptions
                WHERE {' AND '.join(where)}
                ORDER BY timestamp DESC LIMIT {limit}
            """).fetchall()

            obs = [
                {"timestamp": float(r[0]) if r[0] else 0,
                 "domain": str(r[1] or ""),
                 "anomaly_score": _safe_float(r[2]),
                 "question": str(r[3] or ""),
                 "answer": str(r[4] or ""),
                 "model_used": str(r[5] or ""),
                 "tokens_used": int(r[6]) if r[6] else 0}
                for r in rows
            ]

            # Domain stats
            stat_rows = conn.execute(
                "SELECT domain, n_descriptions, avg_anomaly FROM domain_stats"
            ).fetchall()
            stats = {r[0]: {"n": r[1], "avg_anomaly": r[2]} for r in stat_rows}
            conn.close()

            payload = json.dumps({"observations": obs, "stats": stats,
                                  "db_path": self.db_path}).encode()
            self._send(200, "application/json", payload)

        elif parsed.path == "/api/export":
            conn = self._conn()
            rows = conn.execute("""
                SELECT timestamp, domain, anomaly_score, question, answer, model_used
                FROM descriptions ORDER BY timestamp DESC
            """).fetchall()
            conn.close()
            data = [{"ts": float(r[0]) if r[0] else 0,
                     "domain": str(r[1] or ""),
                     "score": _safe_float(r[2]),
                     "q": str(r[3] or ""),
                     "a": str(r[4] or ""),
                     "model": str(r[5] or "")} for r in rows]
            payload = json.dumps(data, indent=2).encode()
            self._send(200, "application/json", payload,
                       extra_headers={"Content-Disposition":
                                      "attachment; filename=language_memory.json"})
        else:
            self._send(404, "text/plain", b"Not found")

    def _send(self, code, ctype, body, extra_headers=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CORTEX-PE Language Memory Browser")
    parser.add_argument("--db", default="language_memory.db")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    if not Path(args.db).exists():
        print(f"[Browser] DB not found: {args.db}")
        print("[Browser] Run the REPL or annotator first to populate.")

    Handler.db_path = args.db
    server = HTTPServer(("0.0.0.0", args.port), Handler)

    url = f"http://localhost:{args.port}"
    print(f"[CORTEX-PE Language Memory Browser]")
    print(f"  DB   : {args.db}")
    print(f"  URL  : {url}")
    print(f"  Stop : Ctrl-C\n")

    if not args.no_open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Browser] Stopped.")


if __name__ == "__main__":
    main()
