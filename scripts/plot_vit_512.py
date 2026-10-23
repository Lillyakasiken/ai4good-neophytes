"""Figures and summary table for the ViT encoder queue (queue_2026-10-20_vit_pc24.sh).

Compares U-Net + MiT-b2, PlantCLEF24 ViT and DINOv2 ViT at 512 crops (cv1, cv3).
Reads the test.py outputs in lightning_logs/ and the training curves from the local
wandb run files (matched to a run directory by start time), so it needs no GPU.

    python scripts/plot_vit_512.py [--out docs/plantclef2024_vit_results]

Writes:
    1_test_f1_per_species.png   test F1 per species, bars = mean, markers = runs
    2_val_f1_curves.png         validation species F1 per epoch (ViT frozen in 0-5)
    3_missed_as_background.png  share of each species' test pixels predicted as background
    summary.csv                 per run: test F1-avg-wo0, and the per-site mean F1 under
                                the old (has_gt) and the new (TP+FP+FN > 0) NaN rule
"""
import argparse
import glob
import json
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore

ENC = {'mit_b2': ('MiT-b2', 'C0'),
       'vitb14_pc24': ('PlantCLEF', 'C1'),
       'vitb14_dinov2': ('DINOv2', 'C2')}
SPECIES = ['A. altissima', 'R. typhina', 'B. davidii', 'R. japonica', 'B. orientalis', 'S. inaequidens']


def load_runs():
    runs = {}
    for d in sorted(glob.glob('lightning_logs/*_np_cv[13]_Unet_*_512_s4[23]')):
        name = os.path.basename(d)
        fold, enc, seed = re.search(r'np_(cv\d)_Unet_(.+)_512_s(\d+)', name).groups()
        if enc not in ENC:
            continue
        qdir = sorted(glob.glob(f'{d}/test_*best*/quantitative'))[-1]
        runs[name] = dict(fold=fold, enc=enc, seed=int(seed),
                          semseg=json.load(open(f'{qdir}/metrics_semseg.json')),
                          site=json.load(open(f'{qdir}/metrics_semseg_site.json')),
                          conf=np.load(f'{qdir}/confusion_matrix.npy').astype(float),
                          history=[])
    return runs


def load_val_curves(runs):
    """val_f1_species per epoch from the wandb run that started within 2 min of the run dir."""
    start = lambda s: datetime.strptime(s, '%Y%m%d_%H%M%S')
    for w in sorted(glob.glob('wandb/run-*')):
        wt = start(os.path.basename(w)[4:19])
        hit = [n for n in runs if 0 <= (wt - start(n[:15])).total_seconds() < 120]
        files = glob.glob(f'{w}/*.wandb')
        if not hit or not files:
            continue
        ds = datastore.DataStore()
        ds.open_for_scan(files[0])
        while (data := ds.scan_data()) is not None:
            rec = pb.Record()
            rec.ParseFromString(data)
            if rec.WhichOneof('record_type') != 'history':
                continue
            items = {(it.key or '/'.join(it.nested_key)): it.value_json for it in rec.history.item}
            row = {k: json.loads(items[k]) for k in ('epoch', 'val_f1_species') if k in items}
            if 'val_f1_species' in row:
                runs[hit[0]]['history'].append((row['epoch'], row['val_f1_species']))


def species_f1(conf, rule):
    """F1-avg-wo0 from a confusion matrix; rule 'old' = NaN without reference pixels,
    'new' = NaN only if the class is absent from reference and prediction."""
    TP = np.diag(conf)
    FP, FN = conf.sum(0) - TP, conf.sum(1) - TP
    scored = (TP + FN) != 0 if rule == 'old' else (TP + FP + FN) != 0
    f1 = np.divide(TP, TP + 0.5 * (FP + FN), out=np.full_like(TP, np.nan), where=scored)
    return np.nanmean(f1[1:])


def by_enc(runs, enc):
    return [r for r in runs.values() if r['enc'] == enc]


def plot_test_f1(runs, out):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    labels = SPECIES + ['Mean']
    x, w = np.arange(len(labels)), 0.26
    for i, (enc, (name, col)) in enumerate(ENC.items()):
        rs = by_enc(runs, enc)
        vals = np.array([r['semseg']['F1'][1:7] + [r['semseg']['F1-avg-wo0']] for r in rs])
        xs = x + (i - 1) * w
        ax.bar(xs, vals.mean(0), w * 0.92, color=col, label=name)
        for r, v in zip(rs, vals):
            ax.scatter(xs, v, s=14, color='k', zorder=3, linewidths=0, marker='o' if r['seed'] == 42 else '^')
    ax.set_xticks(x, labels, fontsize=9.5)
    ax.set_ylabel('F1')
    ax.set_ylim(0, 1)
    ax.set_title('Test F1', loc='left')
    h, l = ax.get_legend_handles_labels()
    h += [Line2D([], [], color='k', marker=m, ls='') for m in 'o^']
    fig.legend(h, l + ['s42', 's43'], frameon=False, ncol=5, loc='lower center')
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(f'{out}/1_test_f1_per_species.png', dpi=150)


def plot_val_curves(runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, fold in zip(axes, ['cv3', 'cv1']):
        ax.axvspan(-0.5, 5.5, color='lightgray', alpha=0.4, zorder=0)
        ax.text(2.5, 0.37, 'frozen', ha='center', color='gray')
        for enc, (name, col) in ENC.items():
            for r in by_enc(runs, enc):
                if r['fold'] == fold and r['history']:
                    ax.plot(*zip(*r['history']), color=col, lw=2, marker='o', ms=4,
                            ls='-' if r['seed'] == 42 else '--')
        ax.set_title(fold, loc='left')
        ax.set_xlabel('Epoch')
        ax.set_ylim(0.35, 0.85)
    axes[0].set_ylabel('Val F1')
    h = [Line2D([], [], color=col, lw=2) for _, col in ENC.values()]
    h += [Line2D([], [], color='k', lw=2, ls=ls) for ls in ('-', '--')]
    fig.legend(h, [n for n, _ in ENC.values()] + ['s42', 's43'], frameon=False, ncol=5, loc='lower center')
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(f'{out}/2_val_f1_curves.png', dpi=150)


def plot_missed(runs, out):
    fig, ax = plt.subplots(figsize=(10, 4))
    x, w = np.arange(len(SPECIES)), 0.26
    for i, (enc, (name, col)) in enumerate(ENC.items()):
        conf = sum(r['conf'] for r in by_enc(runs, enc))
        ax.bar(x + (i - 1) * w, 100 * conf[1:, 0] / conf[1:].sum(1), w * 0.92, color=col, label=name)
    ax.set_xticks(x, SPECIES, fontsize=9.5)
    ax.set_ylabel('Missed (%)')
    ax.set_ylim(0, 100)
    ax.set_title('Pixels predicted as background', loc='left')
    ax.legend(frameon=False, ncol=3, loc='upper left')
    fig.tight_layout()
    fig.savefig(f'{out}/3_missed_as_background.png', dpi=150)


def summary(runs, out):
    rows = []
    for name, r in runs.items():
        site_confs = [np.array(v['conf'], dtype=float) for v in r['site'].values()]
        rows.append({'encoder': ENC[r['enc']][0], 'fold': r['fold'], 'seed': r['seed'],
                     'test_F1_species': species_f1(r['conf'], 'new'),
                     'site_mean_F1_old_rule': np.mean([species_f1(c, 'old') for c in site_confs]),
                     'site_mean_F1_new_rule': np.mean([species_f1(c, 'new') for c in site_confs]),
                     'run': name})
    df = pd.DataFrame(rows).sort_values(['encoder', 'fold', 'seed'])
    df.to_csv(f'{out}/summary.csv', index=False, float_format='%.4f')
    print(df.drop(columns='run').round(3).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='docs/plantclef2024_vit_results')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    runs = load_runs()
    load_val_curves(runs)
    plot_test_f1(runs, args.out)
    plot_val_curves(runs, args.out)
    plot_missed(runs, args.out)
    summary(runs, args.out)
    print('saved to', args.out)
