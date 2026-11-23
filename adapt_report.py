#!/usr/bin/env python3
"""Score few-shot site adaption runs against the received model (task 1).

An adaption run fine-tunes the received model on k labelled tiles per species of one
held-out flight (train.py data=neophytes_split_cv3_adapt_512; the k tiles come from
the flight's /test and /val folders). test.py then scores it on the fold's test set,
the /train folders of the held-out sites, and with eval_site=True it saves one
confusion matrix per flight (metrics_semseg_site.json). From these, per run and species:

    rest          the adapted flight itself (its /train tiles; the k tiles are not among them)
    other_flight  the other flights of the same site, confusion matrices summed
    other_site    the flights of the fold's other held-out sites, summed

The received model (k = 0) is scored on exactly the same tiles, from the
metrics_semseg_site.json of its own test.py run. A (set, species) cell is kept only
if the set has at least --min-tiles tiles containing the species; empty cells do not
enter any mean. A set that a run was not evaluated on (test set restricted to the
adapted site) is skipped for that run.

    python adapt_report.py --baseline <received run>/test_<...>/quantitative/metrics_semseg_site.json
    python adapt_report.py --baseline <...> --runs '*np_cv3_adapt_*' --out results/adapt_cv3

Outputs (--out): runs.csv, one row per run x set x species (F1, precision, recall of
both models, number of tiles with the species, number of k tiles with it), and
summary.csv, the mean and std over the draws (seeds) per flight x k x set x species.
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import yaml

from utils.eval_utils import metrics_from_confusion

# # run directory name: <date>_<time>_np_cv3_adapt_<flight>_k<k>_<model>_<crop>_s<seed>
RUN_RE = re.compile(r'_np_cv\d+_adapt_(?P<flight>.+)_k(?P<k>\d+)_.*_s(?P<seed>\d+)$')


def site_of(flight):
    """Site of a flight, e.g. ``Basel_2_3`` -> ``Basel_2``."""
    return flight.rsplit('_', 1)[0]


def load_site_confs(path):
    """{flight: confusion matrix} from a metrics_semseg_site.json (rows = reference,
    classes in class-id order; only the per-class score arrays there are reordered)."""
    with open(path) as f:
        return {flight: np.array(m['conf'], dtype=np.int64) for flight, m in json.load(f).items()}


def run_site_confs(run_dir):
    """Per-flight confusion matrices of a run, merged over its test.py passes of the
    final checkpoint (e.g. one pass on the adapted site, one on the whole fold)."""
    confs = {}
    for test_dir in sorted(glob.glob(os.path.join(run_dir, 'test_*final*'))):
        path = os.path.join(test_dir, 'quantitative', 'metrics_semseg_site.json')
        if os.path.isfile(path):
            confs.update(load_site_confs(path))
    return confs


def tiles_with_species(stats, species):
    """{class id: number of tiles in stats that contain the species}."""
    return {c: int((stats[f'{s}_px'] > 0).sum()) for c, s in species.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--baseline', required=True, nargs='+',
                        help="metrics_semseg_site.json of the received model's test.py run(s), merged per flight")
    parser.add_argument('--runs', nargs='+', default=['*np_cv3_adapt_*'], help='glob patterns of run directories')
    parser.add_argument('--log-dir', default='lightning_logs')
    parser.add_argument('--stats', default='../data/Neophytes/stats_imagewise.csv')
    parser.add_argument('--classes', default='configs/data/neophytes_names_colors.yaml')
    parser.add_argument('--min-tiles', type=int, default=5, help='min. tiles with the species for a cell to count')
    parser.add_argument('--out', default='results/adapt_cv3')
    args = parser.parse_args()

    # %% inputs
    with open(args.classes) as f:
        classes = yaml.safe_load(f)['classes']
    species = {int(c): v['stats_name'] for c, v in classes.items() if int(c) > 0}
    stats = pd.read_csv(args.stats)
    stats['tile'] = stats['img_path'].map(os.path.basename)
    baseline = {}
    for path in args.baseline:
        baseline.update(load_site_confs(path))

    run_dirs = sorted({d for p in args.runs for d in glob.glob(os.path.join(args.log_dir, p)) if os.path.isdir(d)})

    # %% one row per run x set x species
    rows = []
    for run_dir in run_dirs:
        name = os.path.basename(run_dir)
        m = RUN_RE.search(name)
        adapted = run_site_confs(run_dir)
        if m is None or not adapted:
            print(f'  skipped (no adaption run name or no test.py pass of the final checkpoint): {name}')
            continue
        flight, k, seed = m['flight'], int(m['k']), int(m['seed'])

        # # which species the k tiles contain (train.py writes the drawn tiles)
        with open(os.path.join(run_dir, 'train_tiles.txt')) as f:
            drawn = {os.path.basename(line.strip()) for line in f if line.strip()}
        k_tiles = tiles_with_species(stats[stats['tile'].isin(drawn)], species)

        sets = {
            'rest': [flight],
            'other_flight': [f for f in baseline if site_of(f) == site_of(flight) and f != flight],
            'other_site': [f for f in baseline if site_of(f) != site_of(flight)],
        }
        for set_name, flights in sets.items():
            if not flights or any(f not in adapted for f in flights):
                continue
            base = metrics_from_confusion(sum(baseline[f] for f in flights))
            adapt = metrics_from_confusion(sum(adapted[f] for f in flights))
            # # same tiles as the cv3 test set: the /train folders of these flights
            n_tiles = tiles_with_species(stats[(stats['split'] == 'train') & stats['site'].isin(flights)], species)
            for c, s in species.items():
                if n_tiles[c] < args.min_tiles:
                    continue
                rows.append({'run': name, 'flight': flight, 'k': k, 'seed': seed, 'set': set_name,
                             'species': s, 'n_tiles': n_tiles[c], 'k_tiles': k_tiles[c],
                             'F1_base': base['F1'][c], 'F1_adapt': adapt['F1'][c],
                             'P_base': base['Precision'][c], 'P_adapt': adapt['Precision'][c],
                             'R_base': base['Recall'][c], 'R_adapt': adapt['Recall'][c]})
    if not rows:
        raise SystemExit('Nothing to report.')

    runs = pd.DataFrame(rows)
    runs['dF1'] = runs['F1_adapt'] - runs['F1_base']

    # %% mean and std over the draws (seeds), per flight x k x set x species
    keys = ['flight', 'k', 'set', 'species']
    summary = runs.groupby(keys).agg(
        n_draws=('seed', 'nunique'), n_tiles=('n_tiles', 'first'), k_tiles=('k_tiles', 'mean'),
        F1_base=('F1_base', 'first'), F1_adapt=('F1_adapt', 'mean'), F1_adapt_std=('F1_adapt', 'std'),
        dF1=('dF1', 'mean'), dF1_min=('dF1', 'min'), dF1_max=('dF1', 'max'),
        P_base=('P_base', 'first'), P_adapt=('P_adapt', 'mean'),
        R_base=('R_base', 'first'), R_adapt=('R_adapt', 'mean')).reset_index()

    os.makedirs(args.out, exist_ok=True)
    runs.to_csv(os.path.join(args.out, 'runs.csv'), index=False)
    summary.to_csv(os.path.join(args.out, 'summary.csv'), index=False)
    with pd.option_context('display.width', 200, 'display.max_rows', None, 'display.float_format', '{:.3f}'.format):
        print(summary)
    print(f"\nSaved runs.csv and summary.csv to {args.out}")


if __name__ == '__main__':
    main()
