import argparse
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

def make_auroc_sweep(out="figures/auroc_sweep.png"):
    Path("figures").mkdir(exist_ok=True)
    k_vals = [2, 4, 8, 16, 32]
    auroc  = [0.925, 0.961, 0.977, 0.9974, 0.9997]
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="#0F172A")
    ax.set_facecolor("#0F172A")
    ax.plot(k_vals, auroc, color="#2563EB", lw=2.5, marker="o", ms=8, label="NeMo-WM proprio (26K)")
    for k, v in zip(k_vals, auroc):
        ax.annotate(f"{v:.4f}", (k, v), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=8, color="#93C5FD")
    ax.axhline(0.9319, color="#DC2626", lw=1.5, ls="--", label="V-JEPA 2 ViT-L 326M")
    ax.axhline(0.9557, color="#F97316", lw=1.5, ls=":", label="V-JEPA 2 ViT-G 1034M")
    ax.set_xlabel("ACh context window k_ctx", color="white", fontsize=12)
    ax.set_ylabel("RECON AUROC", color="white", fontsize=12)
    ax.set_title("Superlinear ACh scaling: 26K params outperforms 1034M", color="white", fontsize=11)
    ax.set_xticks(k_vals); ax.tick_params(colors="white")
    [s.set_color("#1E293B") for s in ax.spines.values()]
    ax.grid(True, color="#1E293B", alpha=0.5); ax.set_ylim(0.91, 1.003)
    ax.legend(facecolor="#1E293B", labelcolor="white", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="#0F172A")
    print(f"Saved {out}")

def make_dissociation(out="figures/dissociation.png"):
    Path("figures").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="#0F172A")
    ax.set_facecolor("#0F172A")
    methods = ["VLM-only (ViT-L)", "No-VLM (proprio)"]
    easy = [0.891, 0.9645]; hard = [0.907, 0.9997]
    x = np.arange(2); w = 0.35
    b1 = ax.bar(x - w/2, easy, w, label="Easy negatives", color="#3B82F6", alpha=0.85)
    b2 = ax.bar(x + w/2, hard, w, label="Hard negatives", color="#10B981", alpha=0.85)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003, f"{bar.get_height():.4f}", ha="center", fontsize=9, color="white")
    ax.set_ylabel("AUROC", color="white", fontsize=12)
    ax.set_title("Full computational dissociation: proprio vs VLM-only", color="white", fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(methods, color="white", fontsize=10)
    ax.tick_params(colors="white")
    [s.set_color("#1E293B") for s in ax.spines.values()]
    ax.grid(True, color="#1E293B", alpha=0.5, axis="y"); ax.set_ylim(0.85, 1.02)
    ax.legend(facecolor="#1E293B", labelcolor="white", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="#0F172A")
    print(f"Saved {out}")

def make_flow_comparison(out="figures/flow_comparison.png"):
    Path("figures").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="#0F172A")
    ax.set_facecolor("#0F172A")
    methods = ["Random", "CEM\nlatent", "GRASP\n+probe", "CEM\n+GoalDA", "Closed\nloop", "NeMo Flow\n(n=1)", "NeMo Flow\n(n=10)"]
    sr      = [2, 2, 8, 18, 8, 84, 100]
    colors  = ["#374151","#374151","#374151","#F59E0B","#374151","#3B82F6","#10B981"]
    bars = ax.bar(methods, sr, color=colors, alpha=0.85, width=0.6, edgecolor="none")
    for bar, v in zip(bars, sr):
        ax.text(bar.get_x()+bar.get_width()/2, v+1.5, f"{v}%", ha="center", fontsize=9, color="white", fontfamily="monospace")
    ax.set_ylabel("Success Rate (%)", color="white", fontsize=12)
    ax.set_title("Perception-to-action: NeMo Flow Policy closes the gap", color="white", fontsize=11)
    ax.tick_params(colors="white"); ax.set_ylim(0, 115)
    [s.set_color("#1E293B") for s in ax.spines.values()]
    ax.grid(True, color="#1E293B", alpha=0.4, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="#0F172A")
    print(f"Saved {out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["auroc_sweep","dissociation","flow_comparison","all"])
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.mode in ("auroc_sweep", "all"):
        make_auroc_sweep(args.out or "figures/auroc_sweep.png")
    if args.mode in ("dissociation", "all"):
        make_dissociation(args.out or "figures/dissociation.png")
    if args.mode in ("flow_comparison", "all"):
        make_flow_comparison(args.out or "figures/flow_comparison.png")
(ryzen-ai-1.7.0) PS C:\Users\MeteorAI\Desktop\CORTEX> python eval_pusht_sr.py `
>>     --ckpt checkpoints/action_wm/action_wm_pusht_full_best.pt `
>>     --probe checkpoints/action_wm/block_probe_best.pt `
>>     --n-episodes 100 --no-video --synthetic
Loaded: checkpoints/action_wm/action_wm_pusht_full_best.pt
  obs_dim=5 action_dim=2 k_steps=4
  epoch=44 loss=0.5244
  ac_lift=+0.8286
Block probe loaded: ep=18 MAE=0.0026 (0.3% of arena)
  Probe scorer active — obs-space GoalDA scoring

── WM Planner Synthetic (K=512, H=8) ──
  ep01 ✗  steps=300  dist=0.190  DA=0.31  🧊 COLD
  ep02 ✗  steps=300  dist=0.317  DA=0.09  🧊 COLD
  ep03 ✗  steps=300  dist=0.518  DA=0.82  ♨  WARM
  ep04 ✗  steps=300  dist=0.516  DA=0.31  🧊 COLD
  ep05 ✗  steps=300  dist=0.595  DA=0.34  ❄  COOL
  ep06 ✗  steps=300  dist=0.489  DA=0.39  🧊 COLD
  ep07 ✗  steps=300  dist=0.349  DA=0.41  ❄  COOL
  ep08 ✗  steps=300  dist=0.533  DA=0.36  ♨  WARM
  ep09 ✗  steps=300  dist=0.380  DA=0.20  🧊 COLD
  ep10 ✗  steps=300  dist=0.533  DA=0.83  🔥 HOT
  ep11 ✗  steps=300  dist=0.243  DA=0.14  🧊 COLD
  ep12 ✗  steps=300  dist=0.247  DA=0.36  ❄  COOL
  ep13 ✗  steps=300  dist=0.274  DA=0.18  🧊 COLD
  ep14 ✗  steps=300  dist=0.309  DA=0.17  ❄  COOL
  ep15 ✗  steps=300  dist=0.530  DA=0.31  🧊 COLD
  ep16 ✗  steps=300  dist=0.153  DA=0.22  ❄  COOL
  ep17 ✓  steps=  1  dist=0.049  DA=1.00  🔥 HOT
  ep18 ✗  steps=300  dist=0.379  DA=0.80  🔥 HOT
  ep19 ✗  steps=300  dist=0.578  DA=0.35  🧊 COLD
  ep20 ✗  steps=300  dist=0.264  DA=0.25  🧊 COLD
  ep21 ✗  steps=300  dist=0.208  DA=0.19  🧊 COLD
  ep22 ✓  steps=  1  dist=0.117  DA=0.46  〜 NEUTRAL
  ep23 ✗  steps=300  dist=0.476  DA=0.51  ❄  COOL
  ep24 ✗  steps=300  dist=0.614  DA=0.10  🧊 COLD
  ep25 ✗  steps=300  dist=0.491  DA=0.35  🧊 COLD
  ep26 ✗  steps=300  dist=0.326  DA=0.20  🧊 COLD
  ep27 ✗  steps=300  dist=0.414  DA=0.50  ❄  COOL
  ep28 ✗  steps=300  dist=0.329  DA=0.21  ❄  COOL
  ep29 ✗  steps=300  dist=0.475  DA=0.48  ♨  WARM
  ep30 ✗  steps=300  dist=0.550  DA=0.49  🧊 COLD
  ep31 ✗  steps=300  dist=0.596  DA=0.36  🧊 COLD
  ep32 ✗  steps=300  dist=0.481  DA=0.39  🧊 COLD
  ep33 ✓  steps=  1  dist=0.088  DA=1.00  🔥 HOT
  ep34 ✗  steps=300  dist=0.521  DA=0.26  ❄  COOL
  ep35 ✗  steps=300  dist=0.294  DA=0.12  🧊 COLD
  ep36 ✗  steps=300  dist=0.212  DA=0.21  🧊 COLD
  ep37 ✗  steps=300  dist=0.430  DA=0.35  🧊 COLD
  ep38 ✗  steps=300  dist=0.423  DA=0.32  ♨  WARM
  ep39 ✗  steps=300  dist=0.704  DA=0.48  🧊 COLD
  ep40 ✓  steps=  1  dist=0.119  DA=0.46  〜 NEUTRAL
  ep41 ✗  steps=300  dist=0.584  DA=0.48  🧊 COLD
  ep42 ✗  steps=300  dist=0.428  DA=0.48  ❄  COOL
  ep43 ✗  steps=300  dist=0.456  DA=0.35  〜 NEUTRAL
  ep44 ✗  steps=300  dist=0.433  DA=0.39  🔥 HOT
  ep45 ✗  steps=300  dist=0.601  DA=0.27  🧊 COLD
  ep46 ✗  steps=300  dist=0.596  DA=0.32  🧊 COLD
  ep47 ✗  steps=300  dist=0.441  DA=0.31  🧊 COLD
  ep48 ✓  steps=  1  dist=0.040  DA=1.00  🔥 HOT
  ep49 ✗  steps=300  dist=0.139  DA=0.35  ❄  COOL
  ep50 ✗  steps=300  dist=0.574  DA=0.58  ❄  COOL
  ep51 ✗  steps=300  dist=0.616  DA=0.34  🧊 COLD
  ep52 ✗  steps=300  dist=0.546  DA=0.75  ❄  COOL
  ep53 ✗  steps=300  dist=0.574  DA=0.63  ♨  WARM
  ep54 ✓  steps=  1  dist=0.078  DA=1.00  🔥 HOT
  ep55 ✗  steps=300  dist=0.645  DA=0.46  🧊 COLD
  ep56 ✗  steps=300  dist=0.511  DA=0.60  🔥 HOT
  ep57 ✗  steps=300  dist=0.511  DA=0.39  🧊 COLD
  ep58 ✗  steps=300  dist=0.416  DA=0.18  🧊 COLD
  ep59 ✗  steps=300  dist=0.382  DA=0.47  ❄  COOL
  ep60 ✗  steps=300  dist=0.511  DA=0.32  🧊 COLD
  ep61 ✗  steps=300  dist=0.536  DA=0.33  〜 NEUTRAL
  ep62 ✗  steps=300  dist=0.228  DA=0.20  ❄  COOL
  ep63 ✗  steps=300  dist=0.607  DA=0.44  〜 NEUTRAL
  ep64 ✗  steps=300  dist=0.500  DA=0.38  🧊 COLD
  ep65 ✗  steps=300  dist=0.324  DA=0.19  🧊 COLD
  ep66 ✗  steps=300  dist=0.560  DA=0.30  🧊 COLD
  ep67 ✗  steps=300  dist=0.264  DA=0.25  🧊 COLD
  ep68 ✗  steps=300  dist=0.489  DA=0.26  ❄  COOL
  ep69 ✗  steps=300  dist=0.499  DA=0.48  🧊 COLD
  ep70 ✗  steps=300  dist=0.390  DA=0.52  ❄  COOL
  ep71 ✗  steps=300  dist=0.358  DA=0.17  🧊 COLD
  ep72 ✗  steps=300  dist=0.266  DA=0.25  ❄  COOL
  ep73 ✗  steps=300  dist=0.428  DA=0.56  🧊 COLD
  ep74 ✗  steps=300  dist=0.192  DA=0.30  🧊 COLD
  ep75 ✗  steps=300  dist=0.590  DA=0.54  🧊 COLD
  ep76 ✓  steps=  1  dist=0.095  DA=1.00  🔥 HOT
  ep77 ✗  steps=300  dist=0.695  DA=0.25  〜 NEUTRAL
  ep78 ✗  steps=300  dist=0.371  DA=0.18  🧊 COLD
  ep79 ✗  steps=300  dist=0.553  DA=0.31  ❄  COOL
  ep80 ✗  steps=300  dist=0.483  DA=0.35  🧊 COLD
  ep81 ✗  steps=300  dist=0.630  DA=0.28  🧊 COLD
  ep82 ✗  steps=300  dist=0.152  DA=0.30  ❄  COOL
  ep83 ✓  steps=  1  dist=0.120  DA=0.46  〜 NEUTRAL
  ep84 ✗  steps=300  dist=0.219  DA=0.34  🧊 COLD
  ep85 ✗  steps=300  dist=0.299  DA=0.29  🧊 COLD
  ep86 ✗  steps=300  dist=0.340  DA=0.55  ❄  COOL
  ep87 ✗  steps=300  dist=0.581  DA=0.32  🧊 COLD
  ep88 ✗  steps=300  dist=0.262  DA=0.41  🔥 HOT
  ep89 ✗  steps=300  dist=0.424  DA=0.48  🧊 COLD
  ep90 ✗  steps=300  dist=0.210  DA=0.53  🔥 HOT
  ep91 ✗  steps=300  dist=0.470  DA=0.54  🧊 COLD
  ep92 ✗  steps=300  dist=0.524  DA=0.25  🧊 COLD
  ep93 ✗  steps=300  dist=0.214  DA=0.17  ❄  COOL
  ep94 ✗  steps=300  dist=0.394  DA=0.57  🔥 HOT
  ep95 ✗  steps=300  dist=0.414  DA=0.74  🔥 HOT
  ep96 ✗  steps=300  dist=0.582  DA=0.45  ❄  COOL
  ep97 ✗  steps=300  dist=0.306  DA=0.18  🧊 COLD
  ep98 ✗  steps=300  dist=0.330  DA=0.21  ❄  COOL
  ep99 ✓  steps=  1  dist=0.102  DA=0.46  〜 NEUTRAL
  ep100 ✗  steps=300  dist=0.529  DA=0.47  🧊 COLD

══ Synthetic SR: 9.0%  steps=273.1  dist=0.399 ══
(ryzen-ai-1.7.0) PS C:\Users\MeteorAI\Desktop\CORTEX>

(ryzen-ai-1.7.0) PS C:\Users\MeteorAI\Desktop\CORTEX> python eval_pusht_sr.py `
>>     --ckpt checkpoints/action_wm/action_wm_pusht_full_best.pt `
>>     --probe checkpoints/action_wm/block_probe_best.pt `
>>     --n-episodes 100 --no-video --synthetic
Loaded: checkpoints/action_wm/action_wm_pusht_full_best.pt
  obs_dim=5 action_dim=2 k_steps=4
  epoch=44 loss=0.5244
  ac_lift=+0.8286
Block probe loaded: ep=18 MAE=0.0026 (0.3% of arena)
  Probe scorer active — obs-space GoalDA scoring

── WM Planner Synthetic (K=512, H=8) ──
  ep01 ✗  steps=300  dist=0.193  DA=0.26  🧊 COLD
  ep02 ✗  steps=300  dist=0.317  DA=0.08  🧊 COLD
  ep03 ✗  steps=300  dist=0.529  DA=0.49  ❄  COOL
  ep04 ✗  steps=300  dist=0.400  DA=0.38  🧊 COLD
  ep05 ✗  steps=300  dist=0.294  DA=0.41  〜 NEUTRAL
  ep06 ✗  steps=300  dist=0.628  DA=0.26  〜 NEUTRAL
  ep07 ✗  steps=300  dist=0.426  DA=0.47  🧊 COLD
  ep08 ✗  steps=300  dist=0.626  DA=0.26  🧊 COLD
  ep09 ✗  steps=300  dist=0.533  DA=0.61  🧊 COLD
  ep10 ✗  steps=300  dist=0.450  DA=0.37  ♨  WARM
  ep11 ✗  steps=300  dist=0.442  DA=0.36  ❄  COOL
  ep12 ✗  steps=300  dist=0.452  DA=0.38  🧊 COLD
  ep13 ✗  steps=300  dist=0.403  DA=0.61  🔥 HOT
  ep14 ✗  steps=300  dist=0.439  DA=0.38  🧊 COLD
  ep15 ✓  steps=  1  dist=0.082  DA=1.00  🔥 HOT
  ep16 ✗  steps=300  dist=0.352  DA=0.39  🧊 COLD
  ep17 ✗  steps=300  dist=0.424  DA=0.35  〜 NEUTRAL
  ep18 ✗  steps=300  dist=0.386  DA=0.32  ❄  COOL
  ep19 ✗  steps=300  dist=0.226  DA=0.24  〜 NEUTRAL
  ep20 ✗  steps=300  dist=0.523  DA=0.37  ❄  COOL
  ep21 ✗  steps=300  dist=0.492  DA=0.69  🧊 COLD
  ep22 ✗  steps=300  dist=0.378  DA=0.18  🧊 COLD
  ep23 ✗  steps=300  dist=0.610  DA=0.27  ❄  COOL
  ep24 ✗  steps=300  dist=0.664  DA=0.52  ❄  COOL
  ep25 ✗  steps=300  dist=0.452  DA=0.50  〜 NEUTRAL
  ep26 ✗  steps=300  dist=0.422  DA=0.47  🔥 HOT
  ep27 ✗  steps=300  dist=0.259  DA=0.23  ❄  COOL
  ep28 ✗  steps=300  dist=0.371  DA=0.22  🧊 COLD
  ep29 ✗  steps=300  dist=0.355  DA=0.17  ❄  COOL
  ep30 ✗  steps=300  dist=0.582  DA=0.36  🧊 COLD
  ep31 ✗  steps=300  dist=0.450  DA=0.38  ♨  WARM
  ep32 ✗  steps=300  dist=0.400  DA=0.38  🧊 COLD
  ep33 ✓  steps=122  dist=0.119  DA=0.23  〜 NEUTRAL
  ep34 ✗  steps=300  dist=0.491  DA=0.67  ♨  WARM
  ep35 ✗  steps=300  dist=0.174  DA=0.33  ❄  COOL
  ep36 ✗  steps=300  dist=0.572  DA=0.42  ♨  WARM
  ep37 ✗  steps=300  dist=0.503  DA=0.62  🧊 COLD
  ep38 ✗  steps=300  dist=0.640  DA=0.28  🧊 COLD
  ep39 ✗  steps=300  dist=0.491  DA=0.33  ❄  COOL
  ep40 ✗  steps=300  dist=0.564  DA=0.52  ❄  COOL
  ep41 ✗  steps=300  dist=0.183  DA=0.20  🧊 COLD
  ep42 ✗  steps=300  dist=0.493  DA=0.35  🧊 COLD
  ep43 ✗  steps=300  dist=0.158  DA=0.22  🧊 COLD
  ep44 ✗  steps=300  dist=0.497  DA=0.49  🔥 HOT
  ep45 ✗  steps=300  dist=0.573  DA=0.77  🧊 COLD
  ep46 ✗  steps=300  dist=0.221  DA=0.22  ❄  COOL
  ep47 ✗  steps=300  dist=0.329  DA=0.36  ❄  COOL
  ep48 ✗  steps=300  dist=0.212  DA=0.25  🧊 COLD
  ep49 ✗  steps=300  dist=0.560  DA=0.63  ❄  COOL
  ep50 ✗  steps=300  dist=0.395  DA=0.45  ❄  COOL
  ep51 ✗  steps=300  dist=0.345  DA=0.17  🧊 COLD
  ep52 ✗  steps=300  dist=0.534  DA=0.35  ❄  COOL
  ep53 ✗  steps=300  dist=0.140  DA=0.45  〜 NEUTRAL
  ep54 ✗  steps=300  dist=0.207  DA=0.20  🧊 COLD
  ep55 ✗  steps=300  dist=0.198  DA=0.43  ♨  WARM
  ep56 ✗  steps=300  dist=0.159  DA=0.23  ❄  COOL
  ep57 ✗  steps=300  dist=0.459  DA=0.27  🧊 COLD
  ep58 ✗  steps=300  dist=0.278  DA=0.49  ❄  COOL
  ep59 ✗  steps=300  dist=0.484  DA=0.36  ❄  COOL
  ep60 ✗  steps=300  dist=0.396  DA=0.33  〜 NEUTRAL
  ep61 ✗  steps=300  dist=0.397  DA=0.16  〜 NEUTRAL
  ep62 ✗  steps=300  dist=0.436  DA=0.42  ❄  COOL
  ep63 ✗  steps=300  dist=0.476  DA=0.44  〜 NEUTRAL
  ep64 ✗  steps=300  dist=0.556  DA=0.26  🧊 COLD
  ep65 ✗  steps=300  dist=0.509  DA=0.39  〜 NEUTRAL
  ep66 ✗  steps=300  dist=0.423  DA=0.65  〜 NEUTRAL
  ep67 ✗  steps=300  dist=0.607  DA=0.47  🧊 COLD
  ep68 ✗  steps=300  dist=0.147  DA=0.19  🧊 COLD
  ep69 ✗  steps=300  dist=0.637  DA=0.21  〜 NEUTRAL
  ep70 ✗  steps=300  dist=0.488  DA=0.43  ❄  COOL
  ep71 ✗  steps=300  dist=0.359  DA=0.57  🔥 HOT
  ep72 ✗  steps=300  dist=0.200  DA=0.22  ❄  COOL
  ep73 ✗  steps=300  dist=0.576  DA=0.38  ❄  COOL
  ep74 ✓  steps= 16  dist=0.109  DA=0.57  🔥 HOT
  ep75 ✗  steps=300  dist=0.613  DA=0.25  ❄  COOL
  ep76 ✗  steps=300  dist=0.469  DA=0.49  🧊 COLD
  ep77 ✗  steps=300  dist=0.638  DA=0.34  🧊 COLD
  ep78 ✗  steps=300  dist=0.517  DA=0.32  ❄  COOL
  ep79 ✗  steps=300  dist=0.513  DA=0.52  🧊 COLD
  ep80 ✗  steps=300  dist=0.511  DA=0.34  ❄  COOL
  ep81 ✗  steps=300  dist=0.573  DA=0.36  ♨  WARM
  ep82 ✗  steps=300  dist=0.392  DA=0.18  🧊 COLD
  ep83 ✗  steps=300  dist=0.225  DA=0.14  🧊 COLD
  ep84 ✗  steps=300  dist=0.453  DA=0.58  〜 NEUTRAL
  ep85 ✗  steps=300  dist=0.550  DA=0.31  🧊 COLD
  ep86 ✗  steps=300  dist=0.557  DA=0.54  🧊 COLD
  ep87 ✗  steps=300  dist=0.482  DA=0.48  🧊 COLD
  ep88 ✗  steps=300  dist=0.539  DA=0.36  ❄  COOL
  ep89 ✗  steps=300  dist=0.473  DA=0.46  ❄  COOL
  ep90 ✗  steps=300  dist=0.393  DA=0.33  🧊 COLD
  ep91 ✗  steps=300  dist=0.297  DA=0.32  🧊 COLD
  ep92 ✗  steps=300  dist=0.188  DA=0.38  🧊 COLD
  ep93 ✗  steps=300  dist=0.519  DA=0.29  🧊 COLD
  ep94 ✗  steps=300  dist=0.551  DA=0.43  🧊 COLD
  ep95 ✗  steps=300  dist=0.526  DA=0.36  🧊 COLD
  ep96 ✗  steps=300  dist=0.354  DA=0.17  ❄  COOL
  ep97 ✗  steps=300  dist=0.439  DA=0.43  🧊 COLD
  ep98 ✗  steps=300  dist=0.574  DA=0.28  ❄  COOL
  ep99 ✗  steps=300  dist=0.580  DA=0.43  🧊 COLD
  ep100 ✗  steps=300  dist=0.551  DA=0.32  🧊 COLD

══ Synthetic SR: 3.0%  steps=292.4  dist=0.423 ══
(ryzen-ai-1.7.0) PS C:\Users\MeteorAI\Desktop\CORTEX>

(ryzen-ai-1.7.0) PS C:\Users\MeteorAI\Desktop\CORTEX> python -c @"
>> import onnxruntime as ort, numpy as np, time
>>
>> # Load all three perception models
>> visual = ort.InferenceSession('npu_models/cortex_student_xint8.onnx',
>>     providers=['VitisAIExecutionProvider','CPUExecutionProvider'])
>> temporal = ort.InferenceSession('npu_models/temporal_head_xint8.onnx',
>>     providers=['VitisAIExecutionProvider','CPUExecutionProvider'])
>> transition = ort.InferenceSession('npu_models/transition_predictor_xint8.onnx',
>>     providers=['VitisAIExecutionProvider','CPUExecutionProvider'])
>>
>> # Dummy inputs
>> img = np.random.randn(1, 3, 224, 224).astype(np.float32)
>> z   = np.random.randn(1, 128).astype(np.float32)
>> obs = np.random.randn(1, 8).astype(np.float32)
>>
>> # Warmup
>> for _ in range(20):
>>     v_out = visual.run(None, {visual.get_inputs()[0].name: img})[0]
>>     t_out = temporal.run(None, {temporal.get_inputs()[0].name: z})[0]
>>     tr_out = transition.run(None, {transition.get_inputs()[0].name: obs})[0]
>>
>> # Benchmark full pipeline
>> times_full = []
>> times_vis  = []
>> times_temp = []
>> times_trans = []
>>
>> for _ in range(500):
>>     t0 = time.perf_counter()
>>     v_out  = visual.run(None, {visual.get_inputs()[0].name: img})[0]
>>     t1 = time.perf_counter()
>>     t_out  = temporal.run(None, {temporal.get_inputs()[0].name: z})[0]
>>     t2 = time.perf_counter()
>>     tr_out = transition.run(None, {transition.get_inputs()[0].name: obs})[0]
>>     t3 = time.perf_counter()
>>     times_vis.append((t1-t0)*1000)
>>     times_temp.append((t2-t1)*1000)
>>     times_trans.append((t3-t2)*1000)
>>     times_full.append((t3-t0)*1000)
>>
>> print('═══ NPU Pipeline Benchmark (AMD Ryzen AI MAX+ 395 XINT8) ═══')
>> print(f'Visual encoder  : {np.mean(times_vis):.3f}ms  p95={np.percentile(times_vis,95):.3f}ms')
>> print(f'Temporal head   : {np.mean(times_temp):.3f}ms  p95={np.percentile(times_temp,95):.3f}ms')
>> print(f'WM transition   : {np.mean(times_trans):.3f}ms  p95={np.percentile(times_trans,95):.3f}ms')
>> print(f'─────────────────────────────────────────────────')
>> print(f'Total perception: {np.mean(times_full):.3f}ms  p95={np.percentile(times_full,95):.3f}ms')
>> print(f'Max control rate: {1000/np.mean(times_full):.0f} Hz')
>> print(f'4Hz budget used : {np.mean(times_full)/250*100:.2f}%')
>> print(f'10Hz budget used: {np.mean(times_full)/100*100:.2f}%')
>> print(f'')
>> print(f'+ Flow policy (CPU n=1): ~0.46ms')
>> print(f'Full perception-to-action: ~{np.mean(times_full)+0.46:.2f}ms')
>> print(f'═════════════════════════════════════════════════')
>> "@ 2>$null
INFO: [aiecompiler 77-749] Reading logical device aie2p_8x4_device
Using TXN FORMAT 0.1
INFO: [aiecompiler 77-749] Reading logical device aie2p_8x4_device
INFO: [aiecompiler 77-749] Reading logical device aie2p_8x4_device
INFO: [aiecompiler 77-749] Reading logical device aie2p_8x4_device
═══ NPU Pipeline Benchmark (AMD Ryzen AI MAX+ 395 XINT8) ═══
Visual encoder  : 0.353ms  p95=0.380ms
Temporal head   : 0.146ms  p95=0.170ms
WM transition   : 0.356ms  p95=0.432ms
─────────────────────────────────────────────────
Total perception: 0.855ms  p95=0.959ms
Max control rate: 1169 Hz
4Hz budget used : 0.34%
10Hz budget used: 0.86%

+ Flow policy (CPU n=1): ~0.46ms
Full perception-to-action: ~1.32ms
═════════════════════════════════════════════════
(ryzen-ai-1.7.0) PS C:\Users\MeteorAI\Desktop\CORTEX>

(ryzen-ai-1.7.0) PS C:\Users\MeteorAI\Desktop\CORTEX> python train_proprio_6c.py `
>>     --hdf5-dir recon_data/synthetic_sprint9 `
>>     --k-ctx 16 --epochs 20 `
>>     --out-ckpt checkpoints/cwm/proprio_kctx16_sprint9.pt
  Cortisol: ACTIVE (sensitivity=0.1)

Sprint 6c — Temporal ProprioceptiveEncoder Training
  Architecture: 16-frame window, sinusoidal PE, attn pooling
  d_per_frame=8, d_hidden=128, d_model=64
  Parameters: 26,561  (Sprint 6b: 26,432)
  Epochs: 20  |  Steps/epoch: 500
  LR: 0.0003  |  Temp: 0.05
  k_ctx: 16  |  k_max: 4  |  k_hard_min: 32
  No GPS (vel+ang+heading+contact+delta_h only)
  Output: checkpoints/cwm/proprio_kctx16_sprint9.pt

ProprioPairDatasetTemporal: 500 files | k_ctx=16, k_max=4, k_hard_min=32, n_neg=7
 Epoch      Loss    top1_acc            LR     Elapsed
───────────────────────────────────────────────────────
     0    0.3603      0.9137      2.98e-04     1863.6s
  → Saved (best top1_acc=0.9137): checkpoints/cwm/proprio_kctx16_sprint9.pt
     1    0.2405      0.9308      2.93e-04     1098.3s
  → Saved (best top1_acc=0.9308): checkpoints/cwm/proprio_kctx16_sprint9.pt
     2    0.1980      0.9408      2.84e-04      868.5s
  → Saved (best top1_acc=0.9408): checkpoints/cwm/proprio_kctx16_sprint9.pt
     3    0.1834      0.9439      2.72e-04      897.9s
  → Saved (best top1_acc=0.9439): checkpoints/cwm/proprio_kctx16_sprint9.pt
     4    0.1668      0.9488      2.57e-04      859.4s
  → Saved (best top1_acc=0.9488): checkpoints/cwm/proprio_kctx16_sprint9.pt
     5    0.1481      0.9529      2.39e-04     1291.4s
  → Saved (best top1_acc=0.9529): checkpoints/cwm/proprio_kctx16_sprint9.pt
     6    0.1389      0.9567      2.19e-04     1224.1s
  → Saved (best top1_acc=0.9567): checkpoints/cwm/proprio_kctx16_sprint9.pt
     7    0.1314      0.9573      1.97e-04      876.9s
  → Saved (best top1_acc=0.9573): checkpoints/cwm/proprio_kctx16_sprint9.pt
     8    0.1285      0.9581      1.75e-04      865.0s
  → Saved (best top1_acc=0.9581): checkpoints/cwm/proprio_kctx16_sprint9.pt
     9    0.1156      0.9610      1.52e-04      860.1s
  → Saved (best top1_acc=0.9610): checkpoints/cwm/proprio_kctx16_sprint9.pt
    10    0.1121      0.9605      1.28e-04      861.6s
