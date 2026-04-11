"""
contradiction.py — NeMo-WM status document contradiction detector
==================================================================
Compares claims in CWM_STATUS.md against actual evidence
(checkpoint metadata, log files, probe results).
Writes CONTRADICTIONS.md with any discrepancies found.

Author: John Taylor — github.com/taylorjohn
"""

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Claim:
    source:  str           # which doc/line
    text:    str           # original claim text
    key:     str           # what it claims (e.g. "auroc_k1")
    value:   float         # claimed numeric value
    unit:    str = ""


@dataclass
class Evidence:
    source:  str           # where evidence comes from
    key:     str           # what it evidences
    value:   float         # actual value found
    note:    str = ""


@dataclass
class Contradiction:
    key:        str
    claimed:    float
    found:      float
    claim_src:  str
    evid_src:   str
    delta_pct:  float
    severity:   str        # 'ERROR', 'WARNING', 'INFO'


# ── Claim extractor ──────────────────────────────────────────────────────────

# Patterns to extract numeric claims from status docs
CLAIM_PATTERNS = [
    # AUROC
    (r'AUROC\s+k=1.*?([\d.]{4,})',    'auroc_k1',    ''),
    (r'AUROC\s+k=2.*?([\d.]{4,})',    'auroc_k2',    ''),
    (r'AUROC\s+k=4.*?([\d.]{4,})',    'auroc_k4',    ''),
    (r'AUROC\s+k=8.*?([\d.]{4,})',    'auroc_k8',    ''),
    (r'AUROC\s+k=16.*?([\d.]{4,})',   'auroc_k16',   ''),
    # Loss
    (r'loss\s+0\.(5\d{3})',           'cwm_best_loss', ''),
    (r'cwm_best\.pt.*?loss\s+([\d.]+)', 'cwm_best_loss', ''),
    # DA
    (r'DA\s*peak.*?([\d.]{5,})',      'da_peak',     ''),
    (r'peak\s+DA.*?([\d.]{5,})',      'da_peak',     ''),
    # Cortisol
    (r'r=0\.([\d]{3})',               'cortisol_r',  ''),
    (r'cortisol.*?r=([\d.]+)',        'cortisol_r',  ''),
    # Navigation
    (r'([\d.]+)ms\s+end.to.end',      'nav_latency_ms', 'ms'),
    (r'4\.93ms',                       'nav_latency_ms', 'ms'),
    # GeoLatentDB
    (r'(65,\d+|65\d{3})\s+GPS',      'geodb_entries', ''),
    # Epoch
    (r'epoch\s+29.*?loss\s+([\d.]+)', 'cwm_ep29_loss', ''),
]

def extract_claims(text: str, source: str) -> list[Claim]:
    claims = []
    for pattern, key, unit in CLAIM_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            raw = m.group(1).replace(',', '')
            try:
                val = float(raw)
                claims.append(Claim(source=source, text=m.group(0), key=key, value=val, unit=unit))
            except ValueError:
                pass
    # Deduplicate by key — keep first occurrence
    seen = set()
    out = []
    for c in claims:
        if c.key not in seen:
            seen.add(c.key)
            out.append(c)
    return out


# ── Evidence gatherers ───────────────────────────────────────────────────────

def gather_checkpoint_evidence(ckpt_path: Path) -> list[Evidence]:
    """Load epoch/loss from a checkpoint file."""
    evidence = []
    if not ckpt_path.exists():
        return evidence
    try:
        import torch
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict):
            epoch = ckpt.get('epoch')
            loss  = ckpt.get('loss')
            if epoch is not None:
                evidence.append(Evidence(
                    source=str(ckpt_path),
                    key='cwm_best_epoch',
                    value=float(epoch),
                    note=f"epoch from checkpoint"
                ))
            if loss is not None:
                evidence.append(Evidence(
                    source=str(ckpt_path),
                    key='cwm_best_loss',
                    value=float(loss),
                    note=f"loss from checkpoint"
                ))
    except Exception as e:
        evidence.append(Evidence(
            source=str(ckpt_path), key='_error', value=0.0,
            note=f"Could not load: {e}"
        ))
    return evidence


def gather_probe_evidence(probe_json: Path) -> list[Evidence]:
    """Read latest probe results JSON."""
    evidence = []
    if not probe_json.exists():
        return evidence
    try:
        data = json.loads(probe_json.read_text())
        # Probe JSON structure: {"auroc": {"k1": 0.9837, ...}, "probe": {...}}
        auroc = data.get('auroc', {})
        for k, v in auroc.items():
            evidence.append(Evidence(
                source=str(probe_json),
                key=f'auroc_{k.replace("=", "")}',
                value=float(v),
                note=f"from probe results JSON"
            ))
    except Exception:
        pass
    return evidence


def gather_log_evidence(log_path: Path) -> list[Evidence]:
    """Scan training log for peak DA and final loss."""
    evidence = []
    if not log_path.exists():
        return evidence

    max_da = 0.0
    max_da_step = 0
    last_loss = None

    try:
        with open(log_path, encoding='utf-8', errors='replace') as f:
            for line in f:
                # DA
                da_m = re.search(r'DA=([\d.]+)', line)
                step_m = re.search(r's(\d+)', line)
                if da_m and step_m:
                    da = float(da_m[1])
                    if da > max_da:
                        max_da = da
                        max_da_step = int(step_m[1])
                # Epoch mean
                loss_m = re.search(r'mean(?:_loss)?=([\d.]+)', line)
                if loss_m:
                    last_loss = float(loss_m[1])

        if max_da > 0:
            evidence.append(Evidence(
                source=str(log_path), key='da_peak', value=max_da,
                note=f"peak at step {max_da_step:,}"
            ))
        if last_loss is not None:
            evidence.append(Evidence(
                source=str(log_path), key='log_final_loss', value=last_loss,
                note="last epoch mean in log"
            ))
    except Exception:
        pass
    return evidence


# ── Comparator ───────────────────────────────────────────────────────────────

TOLERANCE = {
    'auroc_k1':      0.001,   # AUROC must match to 3 decimal places
    'auroc_k2':      0.001,
    'auroc_k4':      0.002,
    'auroc_k8':      0.002,
    'auroc_k16':     0.003,
    'cwm_best_loss': 0.001,
    'da_peak':       0.001,
    'cortisol_r':    0.01,
    'nav_latency_ms': 0.1,
    '_default':      0.01,
}


def compare(claims: list[Claim], evidence: list[Evidence]) -> list[Contradiction]:
    contradictions = []
    ev_map: dict[str, Evidence] = {e.key: e for e in evidence}

    for c in claims:
        if c.key not in ev_map:
            continue
        e = ev_map[c.key]
        tol = TOLERANCE.get(c.key, TOLERANCE['_default'])
        delta = abs(c.value - e.value)
        delta_pct = delta / max(abs(c.value), 1e-9) * 100

        if delta > tol:
            severity = 'ERROR' if delta_pct > 5 else 'WARNING'
            contradictions.append(Contradiction(
                key=c.key, claimed=c.value, found=e.value,
                claim_src=c.source, evid_src=e.source,
                delta_pct=delta_pct, severity=severity
            ))

    return contradictions


# ── Report writer ─────────────────────────────────────────────────────────────

def write_report(
    contradictions: list[Contradiction],
    claims: list[Claim],
    evidence: list[Evidence],
    output_path: Path,
):
    ts = time.strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# NeMo-WM Contradiction Report",
        f"> Generated: {ts}",
        f"> Claims checked: {len(claims)} | Evidence sources: {len(evidence)}",
        "",
    ]

    if not contradictions:
        lines += [
            "## ✅ No contradictions found",
            "",
            "All checked claims match their evidence sources.",
            "",
        ]
    else:
        errors   = [c for c in contradictions if c.severity == 'ERROR']
        warnings = [c for c in contradictions if c.severity == 'WARNING']
        lines += [
            f"## ❌ {len(errors)} ERROR(s), ⚠ {len(warnings)} WARNING(s)",
            "",
        ]
        for c in contradictions:
            icon = "❌" if c.severity == 'ERROR' else "⚠"
            lines += [
                f"### {icon} {c.key}",
                f"- Claimed: **{c.claimed}** (in {c.claim_src})",
                f"- Found:   **{c.found}** (in {c.evid_src})",
                f"- Delta:   {c.delta_pct:.1f}%",
                "",
            ]

    # Verified claims
    lines += ["## ✅ Verified claims", ""]
    ev_keys = {e.key for e in evidence}
    contra_keys = {c.key for c in contradictions}
    for cl in claims:
        if cl.key in ev_keys and cl.key not in contra_keys:
            ev = next(e for e in evidence if e.key == cl.key)
            lines.append(f"- {cl.key}: {cl.value} ✓ (matches {ev.source})")
    lines.append("")

    output_path.write_text("\n".join(lines))
    print(f"[contradiction] Report written: {output_path} ({len(contradictions)} issues)")


# ── Main ─────────────────────────────────────────────────────────────────────

def run_check(
    status_doc:   str = "CWM_STATUS.md",
    ckpt_path:    str = "checkpoints/cwm/cwm_best.pt",
    log_path:     str = "logs/training.log",
    probe_json:   str = "PROBE_RESULTS_EP12_N1752.json",
    output:       str = "CONTRADICTIONS.md",
):
    status_text = Path(status_doc).read_text(encoding='utf-8') if Path(status_doc).exists() else ""
    claims   = extract_claims(status_text, status_doc)

    evidence: list[Evidence] = []
    evidence += gather_checkpoint_evidence(Path(ckpt_path))
    evidence += gather_log_evidence(Path(log_path))
    evidence += gather_probe_evidence(Path(probe_json))

    contradictions = compare(claims, evidence)
    write_report(contradictions, claims, evidence, Path(output))
    return contradictions


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--status",     default="CWM_STATUS.md")
    parser.add_argument("--ckpt",       default="checkpoints/cwm/cwm_best.pt")
    parser.add_argument("--log",        default="logs/training.log")
    parser.add_argument("--probe-json", default="PROBE_RESULTS_EP12_N1752.json")
    parser.add_argument("--output",     default="CONTRADICTIONS.md")
    args = parser.parse_args()

    issues = run_check(
        status_doc = args.status,
        ckpt_path  = args.ckpt,
        log_path   = args.log,
        probe_json = args.probe_json,
        output     = args.output,
    )
    print(f"Done. {len(issues)} contradiction(s) found.")
