"""
paper2_new_figures.py — Generate Sprint D6++ Figures for Paper 2
================================================================
Creates publication-quality figures for the new results:
  fig7_plateau_proof.png    — Comprehension vs training volume
  fig8_vpp_comparison.png   — Standard vs VPP bar chart
  fig9_word_similarity.png  — Learned word similarity heatmap
  fig10_codebook_init.png   — Random vs K-means codebook discrimination
  fig11_language_levels.png — 10 cognitive levels bar chart
  fig12_comparison.png      — 17 questions radar chart vs baselines

Usage:
    python paper2_new_figures.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'blue': '#3b82f6',
    'green': '#10b981',
    'red': '#ef4444',
    'orange': '#f59e0b',
    'purple': '#8b5cf6',
    'cyan': '#06b6d4',
    'gray': '#6b7280',
}


def fig7_plateau_proof():
    """Comprehension vs training volume — the plateau breakthrough."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Data points
    hearings = [1e3, 1e4, 1e5, 1e6, 3.7e7]
    comprehension = [22.7, 53.5, 65.1, 65.1, 65.1]
    labels = ['1K', '10K', '100K', '1M', '37M']

    # Plot plateau line
    ax.semilogx(hearings, comprehension, 'o-',
                color=COLORS['orange'], linewidth=2.5, markersize=10,
                label='Prototype grounding', zorder=5)

    # Annotate plateau region
    ax.axhspan(64, 66.5, alpha=0.1, color=COLORS['orange'])
    ax.annotate('65.1% plateau\n(unchanged for 3+ orders of magnitude)',
                xy=(1e6, 65.1), xytext=(3e3, 75),
                fontsize=9, color=COLORS['orange'],
                arrowprops=dict(arrowstyle='->', color=COLORS['orange'], lw=1.5))

    # Stemming breakthrough
    ax.plot([3.7e7], [96.4], 's', color=COLORS['blue'],
            markersize=14, zorder=6, label='+ Stemming (96.4%)')
    ax.plot([3.7e7], [100], '*', color=COLORS['green'],
            markersize=18, zorder=7, label='+ 3 words (100%)')

    # Arrow from plateau to breakthrough
    ax.annotate('', xy=(3.7e7, 96.4), xytext=(3.7e7, 65.1),
                arrowprops=dict(arrowstyle='->', color=COLORS['blue'],
                                lw=2, linestyle='--'))
    ax.annotate('Stemming\n(0 extra\nhearings)',
                xy=(3.7e7, 80), fontsize=8, color=COLORS['blue'],
                ha='right')

    ax.set_xlabel('Training volume (hearings)')
    ax.set_ylabel('Comprehension (%)')
    ax.set_title('Plateau proof: 37M hearings cannot break 65.1% ceiling\n'
                 'Stemming breaks through instantly — architecture, not data',
                 fontsize=12, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.set_xlim(5e2, 1e8)
    ax.legend(loc='lower right', framealpha=0.9)

    # Add data labels
    for h, c, l in zip(hearings, comprehension, labels):
        ax.annotate(f'{c}%', xy=(h, c), xytext=(0, 12),
                    textcoords='offset points', ha='center',
                    fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig('fig7_plateau_proof.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  Saved fig7_plateau_proof.png')


def fig8_vpp_comparison():
    """VPP Standard vs Augmented bar chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5),
                                     gridspec_kw={'width_ratios': [2, 1]})

    # Left: per-run comparison
    runs = [1, 2, 3]
    standard = [0.2824, 0.2890, 0.2840]
    vpp = [0.2070, 0.2100, 0.2231]
    improvements = [26.7, 27.3, 21.4]

    x = np.arange(len(runs))
    w = 0.35

    bars1 = ax1.bar(x - w/2, standard, w, color=COLORS['gray'],
                     label='Standard (6ch, 214K)', alpha=0.8)
    bars2 = ax1.bar(x + w/2, vpp, w, color=COLORS['green'],
                     label='VPP Augmented (8ch, 235K)', alpha=0.9)

    # Add improvement labels
    for i, imp in enumerate(improvements):
        ax1.annotate(f'+{imp}%', xy=(x[i] + w/2, vpp[i]),
                     xytext=(0, 8), textcoords='offset points',
                     ha='center', fontsize=10, fontweight='bold',
                     color=COLORS['green'])

    ax1.set_xlabel('Run')
    ax1.set_ylabel('Best Flow Matching Loss')
    ax1.set_title('VPP on real human demos (25,650 samples)',
                  fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Run 1', 'Run 2', 'Run 3'])
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 0.35)

    # Right: mean comparison
    mean_std = np.mean(standard)
    mean_vpp = np.mean(vpp)
    ax2.bar(['Standard', 'VPP'], [mean_std, mean_vpp],
            color=[COLORS['gray'], COLORS['green']], alpha=0.9,
            edgecolor='white', linewidth=2)
    ax2.annotate(f'+25.1%\n±2.7%', xy=(1, mean_vpp),
                 xytext=(0, 10), textcoords='offset points',
                 ha='center', fontsize=12, fontweight='bold',
                 color=COLORS['green'])

    # Add param overhead note
    ax2.text(0.5, 0.02, '9.6% param overhead only',
             transform=ax2.transAxes, ha='center', fontsize=8,
             color=COLORS['gray'])

    ax2.set_ylabel('Mean Loss')
    ax2.set_title('Mean ± std', fontweight='bold')
    ax2.set_ylim(0, 0.35)

    plt.tight_layout()
    plt.savefig('fig8_vpp_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  Saved fig8_vpp_comparison.png')


def fig9_word_similarity():
    """Word similarity heatmap — learned from experience."""
    words = ['gravity', 'falling', 'weight', 'danger', 'safe',
             'fast', 'slow', 'left', 'right', 'corridor']

    # Similarity matrix (from actual results)
    sim = np.array([
        [ 1.000,  0.997,  0.996, -0.050, -0.122, 0.100, -0.100,  0.050, -0.050, -0.018],
        [ 0.997,  1.000,  0.990, -0.040, -0.110, 0.080, -0.080,  0.040, -0.040, -0.015],
        [ 0.996,  0.990,  1.000, -0.060, -0.130, 0.090, -0.090,  0.055, -0.055, -0.020],
        [-0.050, -0.040, -0.060,  1.000, -0.791, 0.200, -0.150,  0.100, -0.100,  0.300],
        [-0.122, -0.110, -0.130, -0.791,  1.000,-0.180,  0.200, -0.080,  0.080, -0.250],
        [ 0.100,  0.080,  0.090,  0.200, -0.180, 1.000, -0.924,  0.150, -0.150,  0.100],
        [-0.100, -0.080, -0.090, -0.150,  0.200,-0.924,  1.000, -0.120,  0.120, -0.080],
        [ 0.050,  0.040,  0.055,  0.100, -0.080, 0.150, -0.120,  1.000, -0.933,  0.050],
        [-0.050, -0.040, -0.055, -0.100,  0.080,-0.150,  0.120, -0.933,  1.000, -0.050],
        [-0.018, -0.015, -0.020,  0.300, -0.250, 0.100, -0.080,  0.050, -0.050,  1.000],
    ])

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(sim, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')

    ax.set_xticks(range(len(words)))
    ax.set_yticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(words, fontsize=10)

    # Add value annotations
    for i in range(len(words)):
        for j in range(len(words)):
            val = sim[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            if i == j:
                continue  # skip diagonal
            if abs(val) > 0.3:
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=7, color=color, fontweight='bold')

    ax.set_title('Word similarity in belief space\n'
                 '(learned from sensorimotor experience, no LLM)',
                 fontweight='bold', fontsize=12)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Cosine similarity', fontsize=10)

    plt.tight_layout()
    plt.savefig('fig9_word_similarity.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  Saved fig9_word_similarity.png')


def fig10_codebook_init():
    """Random vs K-means codebook initialization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: familiar novelty comparison
    methods = ['Random', 'Sample', 'K-means++', 'K-means']
    familiar = [14.73, 3.11, 3.15, 2.49]
    colors = [COLORS['red'], COLORS['orange'], COLORS['blue'], COLORS['green']]

    bars = ax1.bar(methods, familiar, color=colors, alpha=0.85,
                    edgecolor='white', linewidth=2)

    for bar, val in zip(bars, familiar):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{val:.1f}', ha='center', fontweight='bold', fontsize=11)

    ax1.set_ylabel('Mean familiar novelty (↓ better)')
    ax1.set_title('Familiar data novelty\n(lower = codebook covers data)',
                  fontweight='bold')
    ax1.axhline(y=3, color=COLORS['gray'], linestyle='--', alpha=0.5,
                label='Good coverage threshold')

    # Right: novel/familiar ratio
    ratios = [6.2, 29.1, 28.7, 36.4]
    bars2 = ax2.bar(methods, ratios, color=colors, alpha=0.85,
                     edgecolor='white', linewidth=2)

    for bar, val in zip(bars2, ratios):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.1f}×', ha='center', fontweight='bold', fontsize=11)

    ax2.set_ylabel('Novel / familiar ratio (↑ better)')
    ax2.set_title('Discrimination ratio\n(higher = better novelty detection)',
                  fontweight='bold')

    # Annotate improvement
    ax2.annotate('5.9× better\ndiscrimination',
                 xy=(3, 36.4), xytext=(1.5, 38),
                 fontsize=9, color=COLORS['green'], fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=COLORS['green']))

    plt.tight_layout()
    plt.savefig('fig10_codebook_init.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  Saved fig10_codebook_init.png')


def fig11_language_levels():
    """10 cognitive levels bar chart."""
    fig, ax = plt.subplots(figsize=(9, 4.5))

    levels = ['L1\nObject\nPermanence', 'L2\nCause-\nEffect',
              'L3\nSpatial', 'L4\nTemporal',
              'L5\nCondi-\ntionals', 'L6\nAnalogy',
              'L7\nAbstrac-\ntion', 'L8\nCompo-\nsition',
              'L9\nExpla-\nnation', 'L10\nTeaching']
    scores = [100, 100, 100, 100, 100, 60, 100, 100, 100, 88]

    colors_list = [COLORS['green'] if s == 100
                   else COLORS['orange'] if s >= 60
                   else COLORS['red'] for s in scores]

    bars = ax.bar(range(len(levels)), scores, color=colors_list,
                   alpha=0.85, edgecolor='white', linewidth=2)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{score}%', ha='center', fontweight='bold', fontsize=11,
                color=COLORS['green'] if score == 100 else COLORS['orange'])

    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(levels, fontsize=8)
    ax.set_ylabel('Score (%)')
    ax.set_ylim(0, 115)
    ax.set_title('10 cognitive levels — all passed\n'
                 '380 words, 15 domains, cognitive age 4–6 years',
                 fontweight='bold', fontsize=12)

    # Pass threshold
    ax.axhline(y=50, color=COLORS['red'], linestyle='--', alpha=0.4,
               label='Pass threshold (50%)')
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig('fig11_language_levels.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  Saved fig11_language_levels.png')


def fig12_comparison_radar():
    """17 questions comparison: NeMo-WM vs baselines."""
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    categories = ['Perception\n(Q1-4)', 'Imagination\n(Q5-8)',
                   'Decision\n(Q9-11)', 'Language\n(Q12-14)',
                   'Meta-cog\n(Q15-17)']
    N = len(categories)

    # Scores (out of max per category)
    nemo =     [4, 4, 3, 3, 3]  # 17/17
    dreamer =  [0, 2, 0, 0, 0]  # 2/17
    dino =     [1, 0, 0, 0, 0]  # 1/17
    tdmpc =    [0, 1, 1, 0, 0]  # 2/17
    diffpol =  [0, 0, 0, 0, 0]  # 0/17

    # Normalize to [0, 1]
    maxes = [4, 4, 3, 3, 3]
    nemo_n = [n/m for n, m in zip(nemo, maxes)]
    dreamer_n = [n/m for n, m in zip(dreamer, maxes)]
    dino_n = [n/m for n, m in zip(dino, maxes)]
    tdmpc_n = [n/m for n, m in zip(tdmpc, maxes)]

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # close the polygon

    def add_line(data, label, color, lw=2):
        values = data + data[:1]
        ax.plot(angles, values, 'o-', color=color, linewidth=lw, label=label)
        ax.fill(angles, values, alpha=0.1, color=color)

    add_line(nemo_n, 'NeMo-WM (17/17)', COLORS['blue'], lw=3)
    add_line(dreamer_n, 'DreamerV3 (2/17)', COLORS['orange'], lw=1.5)
    add_line(dino_n, 'DINO-WM (1/17)', COLORS['purple'], lw=1.5)
    add_line(tdmpc_n, 'TD-MPC2 (2/17)', COLORS['cyan'], lw=1.5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=8)

    ax.set_title('Introspective capabilities\n'
                 'NeMo-WM answers all 17 questions',
                 fontsize=13, fontweight='bold', y=1.08)
    ax.legend(loc='lower right', bbox_to_anchor=(1.3, -0.05), fontsize=9)

    plt.tight_layout()
    plt.savefig('fig12_comparison_radar.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  Saved fig12_comparison_radar.png')


def fig13_latency():
    """Language latency benchmark results."""
    fig, ax = plt.subplots(figsize=(8, 4))

    operations = [
        'Word lookup\n(direct)', 'Word lookup\n(stemmed)',
        'Similarity', 'Tokenize',
        'Comprehend\n(3 words)', 'Comprehend\n(7 words)',
        'Comprehend\n(12 words)'
    ]
    latencies = [0.21, 0.27, 3.80, 3.79, 11.18, 15.42, 17.85]

    colors_list = [COLORS['green'] if l < 5
                   else COLORS['blue'] if l < 15
                   else COLORS['orange'] for l in latencies]

    bars = ax.barh(range(len(operations)), latencies,
                    color=colors_list, alpha=0.85,
                    edgecolor='white', linewidth=2)

    for bar, val in zip(bars, latencies):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}μs', va='center', fontsize=10, fontweight='bold')

    ax.set_yticks(range(len(operations)))
    ax.set_yticklabels(operations, fontsize=9)
    ax.set_xlabel('Latency (μs)')
    ax.set_title('Language layer latency — all operations < 20μs\n'
                 'Word lookup at 4.7 million Hz',
                 fontweight='bold', fontsize=12)

    # 20μs threshold line
    ax.axvline(x=20, color=COLORS['red'], linestyle='--', alpha=0.5,
               label='20μs target')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 25)

    plt.tight_layout()
    plt.savefig('fig13_latency.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  Saved fig13_latency.png')


if __name__ == '__main__':
    print('=' * 60)
    print('  Paper 2: Generating Sprint D6++ Figures')
    print('=' * 60)

    fig7_plateau_proof()
    fig8_vpp_comparison()
    fig9_word_similarity()
    fig10_codebook_init()
    fig11_language_levels()
    fig12_comparison_radar()
    fig13_latency()

    print(f'\n  All 7 figures saved.')
    print(f'  Copy to paper2 directory for LaTeX compilation.')
