"""Figures for the few-shot runs on a held-out species (queue_2026-11-16_fewshot_loso.sh).

Reads the <name>.json files that fewshot_probe.py writes (no GPU, no data) and the
full-data test scores in results/cv/512_dev_<encoder>/cv_runs.csv as a reference.

    python scripts/plot_fewshot.py [--roots results/fewshot/cv3_bdavidii results/fewshot/loso]
                                   [--out results/fewshot/loso/figures]

Writes (a figure is skipped while its results are missing):
    1_loso_k10_<fold>.png      F1 of each species when it is the new one (10 tiles):
                               PlantCLEF vs DINOv2 probe, bars = mean, dots = draws
    2_k_sweep_<fold>.png       the same F1 against the number of labelled tiles k
    3_bdavidii_methods.png     B. davidii, 10 tiles: probes and base + fine-tune, MiT-b2
                               vs PlantCLEF; F1 of the new species and of the known ones
    3_methods_<species>.png    the same for every other species with base-model results
                               (queue_2026-11-23_fewshot_next.sh), incl. the combined model
    4_random_vs_flight.png     10 tiles from all training sites vs from one flight (and from
                               2 / 5 flights once those draws exist)
    5_labelling_budget.png     where to spend the labels: 10 tiles over all sites / 5 / 2 / 1
                               flights vs 30 tiles from one flight
    fewshot_scores.csv         one row per head / checkpoint (all roots)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from fewshot_probe import summary_rows  # noqa: E402

# display order (configs/data/neophytes_names_colors.yaml)
SPECIES = ['A. altissima', 'R. typhina', 'B. davidii', 'R. japonica', 'B. orientalis', 'S. inaequidens']
# one colour per backbone in every figure (validated: dataviz validate_palette.js, light)
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
PROBES = {'probe_pc24': ('PlantCLEF ViT', BLUE), 'probe_dinov2': ('DINOv2 ViT', ORANGE)}
# figure 3: method -> (label, colour), top to bottom
METHODS = {
    'probe_mit_b2': ('MiT-b2 ImageNet · linear probe', AQUA),
    'probe_A': ('MiT-b2 base encoder · linear probe', AQUA),
    'ft_A_r3': ('MiT-b2 U-Net · base + fine-tune', AQUA),
    'ft_C_r3': ('MiT-b2 + nDSM U-Net · base + fine-tune', AQUA),
    'probe_dinov2': ('DINOv2 · linear probe', ORANGE),
    'probe_pc24': ('PlantCLEF · linear probe', BLUE),
    'probe_P': ('PlantCLEF base ViT · linear probe', BLUE),
    'ft_P_frozen': ('PlantCLEF U-Net · base + fine-tune, ViT frozen', BLUE),
    'ft_P_r3': ('PlantCLEF U-Net · base + fine-tune', BLUE),
    'combo_A_pc24': ('MiT-b2 base (known) + PlantCLEF probe (new)', YELLOW),
}
BACKBONE = {AQUA: 'MiT-b2', ORANGE: 'DINOv2 ViT', BLUE: 'PlantCLEF ViT', YELLOW: 'combined'}
# figure 3 is drawn for a species once it has one of these (base-model results)
BASE_METHODS = {'probe_A', 'ft_A_r3', 'probe_P', 'ft_P_r3', 'ft_P_frozen', 'combo_A_pc24'}
SURFACE, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
FULL_DATA = ('mit_b2', 'U-Net MiT-b2 trained on all labels')

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.size': 10, 'text.color': INK, 'axes.labelcolor': INK2, 'axes.edgecolor': AXIS,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.spines.left': False,
    'axes.grid': True, 'axes.grid.axis': 'y', 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.axisbelow': True, 'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    'legend.frameon': False, 'ytick.major.size': 0, 'xtick.major.size': 0,
})


def load(roots):
    rows = [dict(r, root=root) for root in roots if os.path.isdir(root) for r in summary_rows(root)]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # # a head that was run again in a later root replaces the earlier one; no F1 = the
    # # species has no pixel in what was scored (smoke tests with --max_test_batches)
    df = df.drop_duplicates(['method', 'fold', 'draw', 'species', 'k', 'seed'], keep='last')
    return df[df['F1'].notna()]


def full_data_f1(fold, encoder=FULL_DATA[0]):
    """{species: test F1 of the U-Net trained on all labels} (mean over its seeds)."""
    path = f'results/cv/512_dev_{encoder}/cv_runs.csv'
    if not os.path.exists(path):
        return {}
    runs = pd.read_csv(path)
    runs = runs[runs['fold'] == fold]
    return {s: runs[f'F1_{s}'].mean() for s in SPECIES if len(runs) and f'F1_{s}' in runs}


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('wrote', path)


def grouped_bars(ax, data, series, by, ref=None):
    """Bars = mean F1 per species and series, dots = the draws, tick = full-data reference.

    series: {value of column `by`: (label, colour)}.
    """
    width = 0.8 / len(series)
    for j, (key, (label, colour)) in enumerate(series.items()):
        for i, sp in enumerate(SPECIES):
            f1 = data[(data[by] == key) & (data['species'] == sp)]['F1'].to_numpy()
            if not len(f1):
                continue
            x = i - 0.4 + (j + 0.5) * width
            ax.bar(x, f1.mean(), width=width - 0.04, color=colour, linewidth=0)
            ax.scatter(np.full(len(f1), x), f1, s=16, color=SURFACE, edgecolor=INK2, linewidth=0.9, zorder=3)
            ax.text(x, max(f1.max(), f1.mean()) + 0.025, f'{f1.mean():.2f}', ha='center', va='bottom',
                    fontsize=8.5, color=INK2)
    for i, sp in enumerate(SPECIES):
        if ref and sp in ref:
            ax.hlines(ref[sp], i - 0.44, i + 0.44, color=INK, linewidth=1.4, linestyle=(0, (4, 2)), zorder=2)
    ax.set_xticks(range(len(SPECIES)), SPECIES, fontstyle='italic')
    ax.set_xlim(-0.6, len(SPECIES) - 0.4)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel('test F1 of the new species')
    handles = [Patch(color=c, label=l) for l, c in series.values()]
    handles.append(Line2D([], [], marker='o', linestyle='', markersize=4.5, markerfacecolor=SURFACE,
                          markeredgecolor=INK2, label='one draw of the tiles'))
    if ref:
        handles.append(Line2D([], [], color=INK, linewidth=1.4, linestyle=(0, (4, 2)), label=FULL_DATA[1]))
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=len(handles),
              fontsize=9, handlelength=1.6, columnspacing=1.6)


def fig_loso(df, fold, out, k=10):
    # # the ImageNet MiT-b2 probe joins figure 1 once it has more than the October B. davidii head
    series = dict(PROBES)
    b2 = df[(df['fold'] == fold) & (df['draw'] == 'random') & (df['k'] == k) & (df['method'] == 'probe_mit_b2')]
    if b2['species'].nunique() > 1:
        series['probe_mit_b2'] = ('MiT-b2 ImageNet', AQUA)
    fusion = df[(df['fold'] == fold) & (df['draw'] == 'random') & (df['k'] == k) & (df['method'] == 'probe_fusion')]
    if not fusion.empty:
        series['probe_fusion'] = ('PlantCLEF + MiT-b2 fusion', YELLOW)
    d = df[(df['fold'] == fold) & (df['draw'] == 'random') & (df['k'] == k) & df['method'].isin(series)]
    if d.empty:
        return print(f'figure 1 ({fold}): no probe results with k = {k} yet')
    fig, ax = plt.subplots(figsize=(10, 4.3))
    grouped_bars(ax, d, series, 'method', full_data_f1(fold))
    encoder = 'encoder' if 'probe_mit_b2' in series else 'ViT'
    ax.set_title(f'Each species as the new one: frozen {encoder} + linear head, {k} labelled tiles ({fold})')
    save(fig, out, f'1_loso_k{k}_{fold}.png')


def fig_k_sweep(df, fold, out):
    d = df[(df['fold'] == fold) & (df['draw'] == 'random') & df['method'].isin(PROBES)]
    ks = sorted(d['k'].unique())
    if len(ks) < 2:
        return print(f'figure 2 ({fold}): needs results for at least two k')
    ref = full_data_f1(fold)
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), sharex=True, sharey=True)
    for ax, sp in zip(axes.flat, SPECIES):
        ends = []
        for method, (label, colour) in PROBES.items():
            m = d[(d['method'] == method) & (d['species'] == sp)]
            if m.empty:
                continue
            mean = m.groupby('k')['F1'].mean()
            x = [ks.index(k) for k in mean.index]
            ax.plot(x, mean.to_numpy(), color=colour, linewidth=2, marker='o', markersize=5.5,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
            ax.scatter([ks.index(k) for k in m['k']], m['F1'], s=9, color=colour, alpha=0.45, linewidth=0, zorder=2)
            ends.append([x[-1], mean.iloc[-1], mean.iloc[-1]])
        # # value of the last k next to each line; two labels closer than 0.08 are pushed apart
        ends.sort(key=lambda e: e[1])
        if len(ends) == 2 and ends[1][1] - ends[0][1] < 0.08:
            mid = (ends[0][1] + ends[1][1]) / 2
            ends[0][2], ends[1][2] = mid - 0.04, mid + 0.04
        for x_end, value, y_text in ends:
            ax.text(x_end + 0.13, y_text, f'{value:.2f}', va='center', fontsize=8.5, color=INK2)
        if sp in ref:
            ax.axhline(ref[sp], color=INK, linewidth=1.2, linestyle=(0, (4, 2)), zorder=1)
        ax.set_title(sp, fontstyle='italic', fontweight='normal')
        ax.set_xticks(range(len(ks)), [str(k) for k in ks])
        ax.set_xlim(-0.3, len(ks) - 0.45)
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.01, 0.25))
    for ax in axes[-1]:
        ax.set_xlabel('labelled tiles of the new species (k)')
    for ax in axes[:, 0]:
        ax.set_ylabel('test F1')
    handles = [Line2D([], [], color=c, linewidth=2, marker='o', markersize=5.5, label=f'{l} (mean of the draws)')
               for l, c in PROBES.values()]
    handles.append(Line2D([], [], color=INK, linewidth=1.2, linestyle=(0, (4, 2)), label=FULL_DATA[1]))
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(f'How many labelled tiles does a new species need? Linear probes, {fold}', x=0.07, ha='left',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    save(fig, out, f'2_k_sweep_{fold}.png')


def fig_methods(df, out, species='B. davidii', fold='cv3', k=10):
    d = df[(df['fold'] == fold) & (df['draw'] == 'random') & (df['k'] == k) & (df['species'] == species)
           & df['method'].isin(METHODS)]
    methods = [m for m in METHODS if m in set(d['method'])]
    if not methods:
        return print('figure 3: no results')
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 0.42 * len(methods) + 1.5), sharey=True)
    for ax, (col, title) in zip(axes, [('F1', f'F1 of the new species ({species})'),
                                       ('known_F1', 'mean F1 of the five known species')]):
        for i, method in enumerate(methods):
            v = d[d['method'] == method][col].dropna().to_numpy()
            if not len(v):
                continue
            ax.barh(i, v.mean(), height=0.62, color=METHODS[method][1], linewidth=0)
            ax.scatter(v, np.full(len(v), i), s=16, color=SURFACE, edgecolor=INK2, linewidth=0.9, zorder=3)
            ax.text(max(v.max(), v.mean()) + 0.02, i, f'{v.mean():.2f}', va='center', fontsize=9, color=INK2)
        ax.set_title(title, fontweight='normal', fontsize=10)
        ax.set_xlim(0, 1.0)
        ax.set_xticks(np.arange(0, 1.01, 0.2))
        ax.grid(axis='x')
        ax.grid(axis='y', visible=False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(True)
    axes[0].set_yticks(range(len(methods)), [METHODS[m][0] for m in methods])
    axes[0].invert_yaxis()
    colours = list(dict.fromkeys(METHODS[m][1] for m in methods))
    handles = [Patch(color=c, label=BACKBONE[c]) for c in colours]
    handles.append(Line2D([], [], marker='o', linestyle='', markersize=4.5, markerfacecolor=SURFACE,
                          markeredgecolor=INK2, label='one draw of the tiles'))
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), fontsize=9, bbox_to_anchor=(0.6, -0.08))
    fig.suptitle(f'{species} as a new species with {k} labelled tiles ({fold} test sites)', x=0.02, ha='left',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    short = species.replace('. ', '').replace(' ', '').lower()
    save(fig, out, '3_bdavidii_methods.png' if short == 'bdavidii' else f'3_methods_{short}.png')


def fig_draws(df, out, fold='cv3', k=10, method='probe_pc24'):
    d = df[(df['fold'] == fold) & (df['k'] == k) & (df['method'] == method)]
    if 'flight' not in set(d['draw']):
        return print('figure 4: no single-flight results yet')
    series = {'random': (f'{k} tiles from all training sites', BLUE),
              'flights5': (f'{k} tiles from 5 flights', AQUA), 'flights2': (f'{k} tiles from 2 flights', ORANGE),
              'flight': (f'{k} tiles from one flight', YELLOW)}
    series = {key: v for key, v in series.items() if key in set(d['draw'])}
    fig, ax = plt.subplots(figsize=(10, 4.3))
    grouped_bars(ax, d, series, 'draw', full_data_f1(fold))
    ax.set_title(f'Where the labelled tiles come from: PlantCLEF ViT + linear head ({fold})')
    save(fig, out, '4_random_vs_flight.png')


def fig_budget(df, out, fold='cv3', method='probe_pc24'):
    """Where to spend the labels: 10 tiles spread over n flights vs 30 tiles from one flight."""
    d = df[(df['fold'] == fold) & (df['method'] == method)].copy()
    d['setting'] = d['draw'] + '_k' + d['k'].astype(str)
    if 'flight_k30' not in set(d['setting']):
        return print('figure 5: no 30-tile single-flight results yet')
    series = {'random_k10': ('10 tiles, all training sites', BLUE), 'flights5_k10': ('10 tiles, 5 flights', AQUA),
              'flights2_k10': ('10 tiles, 2 flights', ORANGE), 'flight_k10': ('10 tiles, 1 flight', YELLOW),
              'flight_k30': ('30 tiles, 1 flight', INK2)}
    series = {key: v for key, v in series.items() if key in set(d['setting'])}
    fig, ax = plt.subplots(figsize=(11, 4.3))
    grouped_bars(ax, d, series, 'setting', full_data_f1(fold))
    # # 7 legend entries: in 4 columns instead of one row
    old = ax.get_legend()
    handles = getattr(old, 'legend_handles', None) or old.legendHandles
    labels = [t.get_text() for t in old.get_texts()]
    old.remove()
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=9)
    ax.set_title(f'Labelling budget: few tiles at many places or many tiles at one place? PlantCLEF probe ({fold})')
    save(fig, out, '5_labelling_budget.png')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--roots', nargs='+', default=['results/fewshot/cv3_bdavidii', 'results/fewshot/loso'])
    p.add_argument('--out', default='results/fewshot/loso/figures')
    args = p.parse_args()

    df = load(args.roots)
    if df.empty:
        return print('no results under', args.roots)
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, 'fewshot_scores.csv'), index=False)
    for fold in sorted(df['fold'].unique()):
        fig_loso(df, fold, args.out)
        fig_k_sweep(df, fold, args.out)
    fig_methods(df, args.out)
    for sp in SPECIES:
        if sp != 'B. davidii' and BASE_METHODS & set(df[df['species'] == sp]['method']):
            fig_methods(df, args.out, species=sp)
    fig_draws(df, args.out)
    fig_budget(df, args.out)


if __name__ == '__main__':
    main()
