"""CPU analyses of the few-shot results asked for in docs/plantclef2024_fewshot_en.md.

Reads the <name>.json files of fewshot_probe.py (no GPU, no data) and writes
<root>/analysis.txt:

    python scripts/analyze_fewshot.py [--root results/fewshot/loso]

  A. Confusions of the new species (section 4.3): which true classes the false positives
     come from, and what the missed pixels are predicted as (R. typhina: precision 0.22).
  B. k = 10 -> 30 on the same draws (section 3.4, the R. japonica drop): the draws are
     nested, so the change is paired; F1, precision and recall.
  C. PlantCLEF - DINOv2 with the species as the unit (limitation 4): the 6 draws of one
     species are not independent, so: the per-species mean differences, an exact sign-flip
     test over the species, and a bootstrap CI that resamples species, then draws.
  D. Calibration (section 6.4, fewshot_probe.py probe --calibrate): F1 at the argmax, at
     the bias chosen on the few-shot subset, and at the best bias in hindsight.
  E. Combined model (section 6.2): combo_A_pc24 against its parts on the same draws.
  F. Labelling budget (section 6.5): mean F1 per species for every draw type and k.
  G. Fusion probe (section 6.3): probe_fusion against probe_pc24 on the same draws.

A section is skipped while its results are missing.
"""
import argparse
import glob
import itertools
import json
import os

import numpy as np
import pandas as pd

SPECIES = ['A. altissima', 'R. typhina', 'B. davidii', 'R. japonica', 'B. orientalis', 'S. inaequidens']


def load(root):
    """One row per scored head with metadata, keeping the confusion matrix."""
    rows = []
    for path in sorted(glob.glob(os.path.join(root, '*', '*.json'))):
        with open(path) as f:
            r = json.load(f)
        if 'F1' not in r or 'new_class' not in r:
            continue
        c = r['new_class']
        rows.append(dict(method=r.get('method'), fold=r.get('fold'), draw=r.get('draw', 'random'), k=r.get('k'),
                         species=r['class_names'][c], new_class=c, seed=r.get('seed'), dir=os.path.dirname(path),
                         F1=r['F1'][c], precision=r['Precision'][c], recall=r['Recall'][c],
                         known_F1=float(np.nanmean([v for i, v in enumerate(r['F1']) if i not in (0, c)])),
                         conf=np.asarray(r['conf'], dtype=np.float64), class_names=r['class_names'],
                         sweep=r.get('bias_sweep'), calib_delta=r.get('calib_delta')))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # # a head scored again in a later directory replaces the earlier one
    return df.drop_duplicates(['method', 'fold', 'draw', 'species', 'k', 'seed'], keep='last')


def fmt(df):
    return df.to_string(float_format=lambda v: f'{v:.3f}', na_rep='-')


def new_f1(conf, c):
    tp = conf[c, c]
    fp, fn = conf[:, c].sum() - tp, conf[c, :].sum() - tp
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else np.nan


def sec_confusions(df, methods=('probe_pc24', 'probe_dinov2'), fold='cv3', k=10):
    d = df[(df['fold'] == fold) & (df['draw'] == 'random') & (df['k'] == k) & df['method'].isin(methods)]
    if d.empty:
        return []
    lines = [f'A. Confusions of the new species ({fold}, k = {k}, random draws; confusion matrices summed over '
             'the draws). FP from: share of the false-positive pixels by true class. Missed as: share of the '
             'missed pixels by predicted class.']
    for method, g in d.groupby('method'):
        fp_rows, fn_rows = {}, {}
        for sp, h in g.groupby('species'):
            conf = np.sum(np.stack(h['conf'].to_list()), axis=0)
            c, names = h['new_class'].iloc[0], h['class_names'].iloc[0]
            fp = np.delete(conf[:, c], c)
            fn = np.delete(conf[c, :], c)
            other = [n for i, n in enumerate(names) if i != c]
            fp_rows[sp] = pd.Series(fp / max(fp.sum(), 1), index=other)
            fn_rows[sp] = pd.Series(fn / max(fn.sum(), 1), index=other)
        order = [s for s in SPECIES if s in fp_rows]
        lines += ['', f'{method}: FP from (rows: new species)', fmt(pd.DataFrame(fp_rows).T.loc[order]),
                  '', f'{method}: missed as', fmt(pd.DataFrame(fn_rows).T.loc[order])]
    return lines


def sec_k_pairs(df, method='probe_pc24', fold='cv3', k0=10, k1=30):
    d = df[(df['fold'] == fold) & (df['draw'] == 'random') & (df['method'] == method) & df['k'].isin([k0, k1])]
    p = d.pivot_table(index=['species', 'seed'], columns='k', values=['F1', 'precision', 'recall'])
    if p.empty or k0 not in p['F1'] or k1 not in p['F1']:
        return []
    p = p.dropna()
    diff = pd.DataFrame({m: p[m][k1] - p[m][k0] for m in ['F1', 'precision', 'recall']})
    out = diff.groupby('species').agg(['mean', 'min', 'max'])
    out.columns = [f'{m} {s}' for m, s in out.columns]
    out['F1 up'] = (diff['F1'] > 0).groupby('species').sum()
    out['draws'] = diff.groupby('species').size()
    return [f'B. k = {k0} -> {k1} on the same (nested) draws, {method}, {fold}: change of the new species '
            f'(k{k1} - k{k0})', fmt(out.reindex([s for s in SPECIES if s in out.index]))]


def sign_flip_p(means):
    """Exact two-sided sign-flip test of mean(means) = 0 over all 2^n sign patterns."""
    means = np.asarray(means)
    obs = abs(means.mean())
    flips = np.array(list(itertools.product([1, -1], repeat=len(means))))
    return float(np.mean(np.abs((flips * means).mean(axis=1)) >= obs - 1e-12))


def hier_bootstrap(groups, n=20000, seed=0):
    """95% CI of the mean paired difference: resample species, then draws within each."""
    rng = np.random.RandomState(seed)
    groups = [np.asarray(g) for g in groups]
    stats = np.empty(n)
    for b in range(n):
        picked = rng.randint(len(groups), size=len(groups))
        stats[b] = np.mean([rng.choice(groups[i], size=len(groups[i])).mean() for i in picked])
    return np.percentile(stats, [2.5, 97.5])


def sec_paired_species(df, a='probe_pc24', b='probe_dinov2'):
    d = df[(df['draw'] == 'random') & df['method'].isin([a, b])]
    p = d.pivot_table(index=['fold', 'k', 'species', 'seed'], columns='method', values='F1').dropna()
    if p.empty or a not in p or b not in p:
        return []
    lines = [f'C. {a} - {b}, new-species F1, with the species as the unit (sign-flip test over the species, '
             'bootstrap CI over species then draws)']
    for (fold, k), g in p.groupby(level=[0, 1]):
        diff = (g[a] - g[b]).groupby(level='species')
        means = diff.mean()
        lo, hi = hier_bootstrap([v.to_numpy() for _, v in diff])
        per = ', '.join(f'{s} {means[s]:+.3f}' for s in SPECIES if s in means)
        lines.append(f'{fold} k={k:>2}: mean {means.mean():+.3f} [95% CI {lo:+.3f}, {hi:+.3f}], '
                     f'{int((means > 0).sum())}/{len(means)} species higher, sign-flip p = {sign_flip_p(means):.3f} '
                     f'(minimum possible {2 / 2 ** len(means):.3f}); {per}')
    return lines


def sec_calibration(df):
    d = df[df['sweep'].notna()].copy()
    if d.empty:
        return []
    rows = []
    for _, r in d.iterrows():
        deltas, c = r['sweep']['deltas'], r['new_class']
        f1 = np.array([new_f1(np.asarray(conf, dtype=np.float64), c) for conf in r['sweep']['conf']])
        cal = f1[deltas.index(r['calib_delta'])] if r['calib_delta'] is not None else np.nan
        rows.append(dict(method=r['method'], fold=r['fold'], k=r['k'], species=r['species'], seed=r['seed'],
                         F1_argmax=f1[deltas.index(0.0)], F1_cal=cal, F1_best=np.nanmax(f1),
                         delta_cal=r['calib_delta'], delta_best=deltas[int(np.nanargmax(f1))]))
    t = pd.DataFrame(rows)
    g = t.groupby(['method', 'fold', 'k', 'species']).agg(
        draws=('seed', 'size'), F1_argmax=('F1_argmax', 'mean'), F1_cal=('F1_cal', 'mean'), F1_best=('F1_best', 'mean'),
        delta_cal=('delta_cal', 'mean'), delta_best=('delta_best', 'mean'),
        cal_better=('F1_cal', lambda v: int((v.to_numpy() > t.loc[v.index, 'F1_argmax'].to_numpy()).sum())))
    mean = t.groupby(['method', 'fold', 'k'])[['F1_argmax', 'F1_cal', 'F1_best']].mean()
    return ['D. Calibration of the new-species bias: F1 at the argmax, at the bias chosen on the few-shot '
            'subset (no test labels), and at the best bias in hindsight (an upper bound, uses test labels)',
            fmt(g), '', 'means over species and draws', fmt(mean)]


def sec_combo(df, combo='combo_A_pc24', head='probe_pc24'):
    d = df[(df['method'].isin([combo, head, 'probe_A', 'ft_A_r3'])) & (df['draw'] == 'random') & (df['k'] == 10)]
    if combo not in set(d['method']):
        return []
    lines = [f'E. Combined model ({combo}: known species from the MiT-b2 base, new species from the {head} head) '
             'against its parts, same draws']
    for sp in [s for s in SPECIES if s in set(d[d['method'] == combo]['species'])]:
        seeds = set(d[(d['method'] == combo) & (d['species'] == sp)]['seed'])
        g = d[(d['species'] == sp) & d['seed'].isin(seeds)]
        t = g.groupby('method')[['F1', 'known_F1']].agg(['mean', 'size'])
        lines += ['', f'{sp} (draws {sorted(seeds)})', fmt(t)]
    return lines


def sec_budget(df, method='probe_pc24', fold='cv3'):
    d = df[(df['fold'] == fold) & (df['method'] == method)]
    if d['draw'].nunique() < 2:
        return []
    p = d.pivot_table(index=['draw', 'k'], columns='species', values='F1', aggfunc='mean')
    p = p.reindex(columns=[s for s in SPECIES if s in p.columns])
    p['mean'] = p.mean(axis=1, skipna=False)
    p['draws'] = d.groupby(['draw', 'k']).size() / d.groupby(['draw', 'k'])['species'].nunique()
    return [f'F. Labelling budget, {method}, {fold}: mean new-species F1 per draw type and k '
            '(flight = one flight, flights<n> = n flights, random = all training sites)', fmt(p)]


def sec_fusion(df, a='probe_fusion', b='probe_pc24'):
    d = df[(df['draw'] == 'random') & df['method'].isin([a, b])]
    p = d.pivot_table(index=['fold', 'k', 'species', 'seed'], columns='method', values=['F1', 'known_F1']).dropna()
    if p.empty or a not in p['F1']:
        return []
    diff = pd.DataFrame({'F1 ' + a: p['F1'][a], 'F1 ' + b: p['F1'][b], 'diff': p['F1'][a] - p['F1'][b],
                         'known diff': p['known_F1'][a] - p['known_F1'][b]})
    t = diff.groupby(level=[0, 1, 2]).mean()
    t['wins'] = (diff['diff'] > 0).groupby(level=[0, 1, 2]).sum()
    return [f'G. {a} - {b} on the same draws (new-species F1; known diff = known-species F1)', fmt(t)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', default='results/fewshot/loso')
    args = p.parse_args()

    pd.set_option('display.width', 250, 'display.max_rows', 500, 'display.max_columns', 50)
    df = load(args.root)
    if df.empty:
        return print('no results under', args.root)
    sections = [sec_confusions(df), sec_k_pairs(df), sec_paired_species(df), sec_calibration(df), sec_combo(df),
                sec_budget(df), sec_fusion(df)]
    text = '\n\n\n'.join('\n'.join(s) for s in sections if s)
    print(text)
    with open(os.path.join(args.root, 'analysis.txt'), 'w') as f:
        f.write(text + '\n')


if __name__ == '__main__':
    main()
