#!/usr/bin/env python3
"""Aggregate the five cross-validation folds into the headline metric.

A single fold says as much about which four sites were held out as about the
model. The number that can be compared between methods is therefore the mean
over the folds, with the spread as the honest error bar:

    F1 per class and overall, mean ± std over the five folds.

This script collects the ``metrics_semseg.json`` that ``test.py`` writes for each
run, prints the table, saves it as CSV and draws the bar chart. Runs are keyed by
(fold, seed): the seeds of one fold are averaged first, then mean and sample std
(ddof=1) are taken over the folds. Every single run is also written to
``cv_runs.csv``, the input for paired comparisons between methods on identical
(fold, seed) pairs.

    # train and test the five folds first
    for i in 1 2 3 4 5; do python train.py data=neophytes_split_cv${i}_train_1024; done
    for run in lightning_logs/*np_cv*; do python test.py exp_name=$(basename $run); done

    python evaluate_cv.py --runs '*np_cv*_mit_b2_*'
    python evaluate_cv.py --runs '*np_cv1*' '*np_cv2*' --out results/cv/mit_b2
"""

import argparse
import glob
import json
import os
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np

METRICS = ["F1", "IoU", "Precision", "Recall"]


def find_test_dir(run_dir, ckpt_type=None):
    """Newest ``test_*`` directory of a run that holds a quantitative evaluation."""
    candidates = sorted(glob.glob(os.path.join(run_dir, "test_*")))
    if ckpt_type:
        candidates = [c for c in candidates if ckpt_type in os.path.basename(c)]
    candidates = [c for c in candidates
                  if os.path.isfile(os.path.join(c, "quantitative", "metrics_semseg.json"))]
    return candidates[-1] if candidates else None


def fold_of(run_name):
    """Fold label from a run directory name, e.g. ``..._np_cv3_Unet...`` -> ``cv3``."""
    m = re.search(r'_np_(cv\d+)', run_name)
    return m.group(1) if m else run_name


def seed_of(run_name):
    """Seed from a run directory name, e.g. ``..._1024_s42`` -> ``42`` (None if absent)."""
    m = re.search(r'_s(\d+)$', run_name)
    return int(m.group(1)) if m else None


def collect(log_dir, patterns, ckpt_type=None):
    """{(fold, seed): metrics dict} for every run matching one of the glob patterns."""
    run_dirs = sorted({d for p in patterns for d in glob.glob(os.path.join(log_dir, p))
                       if os.path.isdir(d)})
    if not run_dirs:
        raise SystemExit(f"No run directories in {log_dir} matching: {' '.join(patterns)}")

    runs = {}
    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        test_dir = find_test_dir(run_dir, ckpt_type)
        if test_dir is None:
            print(f"  skipped (no evaluation found, run test.py first): {run_name}")
            continue

        with open(os.path.join(test_dir, "quantitative", "metrics_semseg.json")) as f:
            metrics = json.load(f)

        key = (fold_of(run_name), seed_of(run_name))
        if key in runs:
            print(f"  WARNING: two runs for fold {key[0]} seed {key[1]}, "
                  f"keeping the newer one ({run_name})")
        metrics["run"] = run_name
        metrics["test_dir"] = test_dir
        runs[key] = metrics
        print(f"  {key[0]:<5} s{key[1]}  {run_name}")

    if not runs:
        raise SystemExit("Nothing to aggregate.")
    return runs


def _std(values, axis=None):
    """Sample std over folds (ddof=1); nan when fewer than two folds are available."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # "degrees of freedom <= 0"
        return np.nanstd(values, axis=axis, ddof=1)


def aggregate(runs):
    """Mean and sample std over folds, per class and for the class average without
    background. The seeds of a fold are averaged first, so every fold counts once."""
    fold_names = sorted({fold for fold, _ in runs})
    class_names = next(iter(runs.values()))["class_names"]

    def fold_mean(key):
        """(folds, ...) array: per fold, the mean over its seeds."""
        return np.array([np.nanmean([np.asarray(m[key], dtype=float)
                                     for (f, _), m in runs.items() if f == fold], axis=0)
                         for fold in fold_names], dtype=float)

    per_class = {}
    for metric in METRICS:
        values = fold_mean(metric)  # (folds, classes)
        per_class[metric] = {"mean": np.nanmean(values, axis=0), "std": _std(values, axis=0)}

    overall = {}
    for metric in METRICS:
        values = fold_mean(f"{metric}-avg-wo0")  # (folds,)
        overall[metric] = {"mean": float(np.nanmean(values)), "std": float(_std(values)),
                           "per_fold": dict(zip(fold_names, values.tolist()))}

    return fold_names, class_names, per_class, overall


def print_table(fold_names, class_names, per_class, overall):
    width = max(len(n) for n in class_names) + 2

    print(f"\nPer-class scores, mean ± sample std over {len(fold_names)} folds "
          f"({', '.join(fold_names)}; seeds averaged within each fold)")
    print(f"{'class':<{width}}" + "".join(f"{m:>18}" for m in METRICS))
    for i, name in enumerate(class_names):
        row = "".join(f"{per_class[m]['mean'][i]:>11.3f} ±{per_class[m]['std'][i]:5.3f}"
                      for m in METRICS)
        print(f"{name:<{width}}{row}")

    print(f"\n{'mean (wo background)':<{width}}" +
          "".join(f"{overall[m]['mean']:>11.3f} ±{overall[m]['std']:5.3f}" for m in METRICS))

    print("\nF1 (wo background) per fold (mean over its seeds):")
    for fold, value in overall["F1"]["per_fold"].items():
        print(f"  {fold:<5} {value:.4f}")


def save_csv(path, fold_names, class_names, per_class, overall):
    lines = ["scope,metric,mean,std"]
    for i, name in enumerate(class_names):
        for metric in METRICS:
            lines.append(f"{name},{metric},{per_class[metric]['mean'][i]:.6f},"
                         f"{per_class[metric]['std'][i]:.6f}")
    for metric in METRICS:
        lines.append(f"avg-wo-background,{metric},{overall[metric]['mean']:.6f},"
                     f"{overall[metric]['std']:.6f}")
    for fold, value in overall["F1"]["per_fold"].items():
        lines.append(f"{fold},F1-avg-wo0,{value:.6f},")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def save_runs_csv(path, runs, class_names):
    """One row per (fold, seed) run with its overall and per-class scores -- the
    input for paired tests (mean ΔF1 over identical fold/seed pairs)."""
    header = ["fold", "seed", "run"] + [f"{m}-avg-wo0" for m in METRICS] + \
             [f"{m}_{name}" for m in METRICS for name in class_names]
    lines = [",".join(header)]
    for (fold, seed), m in sorted(runs.items(), key=lambda kv: (kv[0][0], kv[0][1] or -1)):
        row = [fold, "" if seed is None else str(seed), m["run"]]
        row += [f"{m[f'{metric}-avg-wo0']:.6f}" for metric in METRICS]
        row += [f"{v:.6f}" for metric in METRICS for v in m[metric]]
        lines.append(",".join(row))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def plot(path, class_names, per_class, overall):
    """Grouped bars per class with the fold spread as error bars; the fold-averaged
    mean without background is drawn as a horizontal reference line."""
    x = np.arange(len(class_names))
    width = 0.5 / len(METRICS)
    group_offset = (len(METRICS) - 1) * width / 2
    colors = plt.get_cmap("Set2")(np.linspace(0, 1, len(METRICS)))

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (metric, color) in enumerate(zip(METRICS, colors)):
        ax.bar(x - group_offset + i * width, per_class[metric]["mean"], width,
               yerr=per_class[metric]["std"], capsize=2, label=metric, color=color,
               error_kw={"linewidth": 0.8})

    ax.axhline(overall["F1"]["mean"], color="0.3", linestyle="--", linewidth=1)
    ax.text(len(class_names) - 0.5, overall["F1"]["mean"],
            f" mF1 {overall['F1']['mean']:.3f}", va="bottom", ha="right", fontsize=8, color="0.3")

    ax.set_ylim(0, 1)
    ax.set_ylabel("score (mean ± std over folds)")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=len(METRICS))
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["*np_cv*"],
                    help="glob pattern(s) for run directories inside --log-dir "
                         "(default: %(default)s)")
    ap.add_argument("--log-dir", default="lightning_logs")
    ap.add_argument("--ckpt-type", default=None,
                    help="only use evaluations of this checkpoint type (best / last)")
    ap.add_argument("--out", default="results/cv",
                    help="output directory for the CSV and the figure (default: %(default)s)")
    args = ap.parse_args()

    print("Collecting run evaluations:")
    runs = collect(args.log_dir, args.runs, args.ckpt_type)
    fold_names, class_names, per_class, overall = aggregate(runs)
    print_table(fold_names, class_names, per_class, overall)

    os.makedirs(args.out, exist_ok=True)
    save_csv(os.path.join(args.out, "cv_scores.csv"), fold_names, class_names, per_class, overall)
    save_runs_csv(os.path.join(args.out, "cv_runs.csv"), runs, class_names)
    plot(os.path.join(args.out, "cv_scores.png"), class_names, per_class, overall)
    print(f"\nSaved {os.path.join(args.out, 'cv_scores.csv')}, "
          f"{os.path.join(args.out, 'cv_runs.csv')} and "
          f"{os.path.join(args.out, 'cv_scores.png')}")


if __name__ == "__main__":
    main()
