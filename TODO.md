# CORTEX-PE TODO — v16.15
**Updated:** Post-MIMII eval, end of week  
**Priority:** High → Low within each section

---

## 🔴 Monday — First Priority

### 1. Trading System — Fix Trade-Blocking Bug
The current trading system blocks all trades. Must be diagnosed and fixed before next paper trading session.
- Run pytest pre-flight, identify which safeguard phase is blocking
- Fix and resume paper trading: T1 → `mac_glass_cockpit.py`, T2 → `run_trading.py --hud-ip 127.0.0.1`

### 2. Cardiac Eval — Run After Training Completes (~1:30am Saturday)
```powershell
python eval_cardiac_audio.py --student checkpoints\cardiac\student_best.pt
```
Targets: cos_sim ≥ 0.80, AUROC ≥ 0.75. Currently tracking: epoch 4 cos_sim=0.865, AUROC=0.792 ✅

### 3. MIMII Valve — Add Delta Features
Valve AUROC 0.6645 (below 0.70 target). Root cause: log-mel mean+std discards temporal dynamics.
Fix: add delta coefficients to `log_mel()` in `eval_mimii_perid.py`:
```python
delta = np.diff(log, axis=1, prepend=log[:, :1])
return np.concatenate([log.mean(1), log.std(1),
                       delta.mean(1), delta.std(1)]).astype(np.float32)  # 256-D
```
Then rerun: `python eval_mimii_perid.py` — expected to push valve above 0.70.

---

## 🟡 This Week

### 4. NPU Full Stack Deployment
Deploy TransitionPredictor and CardiacStudentEncoder to NPU via XINT8 pipeline.
Both have LayerNorm → XINT8 required (confirmed by npu_coverage_audit.py pattern).
```powershell
# Export each model using export_student_npu.py pattern
# Verify cosine > 0.95 for each
# Update export_manifest.json with SHA256
```

### 5. AMD Community Post — Publish
Post `amd_npu_layernorm_post.md` to:
- AMD Developer Community: https://community.amd.com/t5/ai/bd-p/amd-ai
- Ryzen AI GitHub Discussions: https://github.com/amd/RyzenAI-SW/discussions
- ONNX Runtime GitHub Discussions: https://github.com/microsoft/onnxruntime/discussions

Add footer note: "In SDK 1.7.1, vitisai_ep_report.json generation requires manual enablement."

### 6. Ryzen AI SDK 1.7.1 Upgrade
Currently on 1.7.0. 1.7.1 is available.
- Wait until trading system is fixed and live first
- Clean conda env upgrade, re-run full export pipeline
- Key change: vitisai_ep_report.json no longer auto-generated — update npu_coverage_audit.py

### 7. MIMII — Update Documentation with Per-ID Protocol
Update STATUS_REPORT.md and README.md with correct MIMII results:
```
Fan     AUROC 0.9332  ✅
Pump    AUROC 0.9357  ✅  
Slider  AUROC 0.9318  ✅
Valve   AUROC 0.6645  ❌ (pending delta feature fix)
Protocol: Per-ID (own normal → own abnormal), log-mel 128-D PCA k=8
```

---

## 🟢 Backlog

### 8. SMAP/MSL NASA Telemetry Domain
Next domain after MIMII. Spacecraft + Mars Science Lab sensor anomaly data.
Free download from NASA GitHub. Multivariate time series, labeled anomalies.
Directly relevant to navigation/perception framing.

### 9. Cardiac — WavLM Student MIMII Transfer Test
Run `eval_mimii_perid.py --student checkpoints/cardiac/student_best.pt` after cardiac training finishes.
Currently cardiac student scored 0.376 on fan (worse than log-mel).
Re-test after full 20-epoch training — richer WavLM embeddings may generalise better.

### 10. ReconNavigator — Integration into Live Navigation Loop
Wire `recon_navigator.py` into a live camera feed loop.
`step_cached()` at 1.2ms supports >800Hz — far faster than camera frame rate.
Integration point: replace the stub in `engine.py`.

### 11. Benchmark Curvature Diagnostics
Curvature diagnostic patch built for `run_benchmark.py` — run and document results.
Measures cosine similarity of adjacent latent trajectory differences (straightening metric).

### 12. npu_coverage_audit.py — Add XINT8 Operator Database
Current audit flags unknown operators (DequantizeLinear, QuantizeLinear, Erf, Div).
Add these to OPERATOR_COVERAGE dict as XINT8-native operators.
Also fix false-positive LayerNorm detection on XINT8 models.

---

## ✅ Completed This Week

- StudentEncoder NPU: re-exported with correct architecture, cosine 0.9997, SHA256 pinned
- RECON TemporalHead: AUROC 0.9337 ✅
- RECON TransitionPredictor: MAE 0.098m ✅
- ReconNavigator v16.15: unified step()/plan()/step_cached() ✅
- CWRU Bearing: AUROC 1.0000 ✅
- MIMII Industrial: Fan/Pump/Slider AUROC 0.93+ ✅
- npu_coverage_audit.py: built and validated ✅
- export_student_npu.py: correct pipeline with version pinning ✅
- AMD community post: drafted ✅
- Documentation: README, ARCHITECTURE, STATUS_REPORT all updated ✅
