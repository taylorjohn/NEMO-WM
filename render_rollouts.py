import argparse
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def make_auroc_sweep(out='figures/auroc_sweep.png'):
    Path('figures').mkdir(exist_ok=True)
    k_vals = [2, 4, 8, 16, 32]
    auroc  = [0.925, 0.961, 0.977, 0.9974, 0.9997]
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor='#0F172A')
    ax.set_facecolor('#0F172A')
    ax.plot(k_vals, auroc, color='#2563EB', lw=2.5, marker='o', ms=8, label='NeMo-WM proprio (26K)')
    for k, v in zip(k_vals, auroc):
        ax.annotate(f'{v:.4f}', (k, v), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=8, color='#93C5FD')
    ax.axhline(0.9319, color='#DC2626', lw=1.5, ls='--', label='V-JEPA 2 ViT-L 326M')
    ax.axhline(0.9557, color='#F97316', lw=1.5, ls=':', label='V-JEPA 2 ViT-G 1034M')
    ax.set_xlabel('ACh context window k_ctx', color='white', fontsize=12)
    ax.set_ylabel('RECON AUROC', color='white', fontsize=12)
    ax.set_title('Superlinear ACh scaling: 26K params outperforms 1034M', color='white', fontsize=11)
    ax.set_xticks(k_vals); ax.tick_params(colors='white')
    [s.set_color('#1E293B') for s in ax.spines.values()]
    ax.grid(True, color='#1E293B', alpha=0.5); ax.set_ylim(0.91, 1.003)
    ax.legend(facecolor='#1E293B', labelcolor='white', fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='#0F172A')
    print(f'Saved {out}')

def make_dissociation(out='figures/dissociation.png'):
    Path('figures').mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor='#0F172A')
    ax.set_facecolor('#0F172A')
    methods = ['VLM-only (ViT-L)', 'No-VLM (proprio)']
    easy = [0.891, 0.9645]; hard = [0.907, 0.9997]
    x = np.arange(2); w = 0.35
    b1 = ax.bar(x - w/2, easy, w, label='Easy negatives', color='#3B82F6', alpha=0.85)
    b2 = ax.bar(x + w/2, hard, w, label='Hard negatives', color='#10B981', alpha=0.85)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003, f'{bar.get_height():.4f}', ha='center', fontsize=9, color='white')
    ax.set_ylabel('AUROC', color='white', fontsize=12)
    ax.set_title('Full computational dissociation: proprio vs VLM-only', color='white', fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(methods, color='white', fontsize=10)
    ax.tick_params(colors='white')
    [s.set_color('#1E293B') for s in ax.spines.values()]
    ax.grid(True, color='#1E293B', alpha=0.5, axis='y'); ax.set_ylim(0.85, 1.02)
    ax.legend(facecolor='#1E293B', labelcolor='white', fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='#0F172A')
    print(f'Saved {out}')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['auroc_sweep','dissociation','all'])
    p.add_argument('--out', default=None)
    args = p.parse_args()
    if args.mode in ('auroc_sweep', 'all'):
        make_auroc_sweep(args.out or 'figures/auroc_sweep.png')
    if args.mode in ('dissociation', 'all'):
        make_dissociation(args.out or 'figures/dissociation.png')
