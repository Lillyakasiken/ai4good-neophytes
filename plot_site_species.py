#!/usr/bin/env python3
"""Plot how the six species are distributed over the sites and CV folds.

Most species are labelled at only a few sites, often one dominant one. A model
can then learn the site (background, ballast, native vegetation) as a cue for
the species, and that cue breaks on a held-out site. These figures show how
strong the coupling is and what each CV fold removes from training.

Everything is computed from ``stats_imagewise.csv`` (labelled area in m², so
sites with different GSD are comparable); no tile is opened. Sites are the
``<site>`` part of ``<site>_<flight>``; by default the flights of a site are
summed, which is the area the model sees during training.

    python plot_site_species.py
    python plot_site_species.py --flight-agg mean --out-dir results/dataset_viz

Figures written to ``--out-dir`` (PNG + PDF each):

1. ``site_species_share``    site x species, % of each species' total area per site
2. ``site_composition``      site x species, % of each site's labelled area per species
3. ``fold_holdout``          fold x species, % of each species' area held out by the fold
4. ``species_concentration`` effective number of sites per species + top-site share

The figures are derived from the confidential dataset: keep them inside the
course (slides for the supervisors are fine, public repos are not).
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import LinearSegmentedColormap

# # Sequential single-hue ramp (light -> dark blue), steps 100..700 of the
# # validated reference palette. Exact zeros are drawn as white, so "absent"
# # stays distinguishable from "small".
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
INK = "#1f1f1e"
INK_MUTED = "#6b6b68"
GRID = "#d9d8d4"


def load_config(path):
    """Species names in display order, their stats column names and the CV folds."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    classes = cfg["classes"]
    order = cfg["display_order"]
    names = [classes[c]["name"] for c in order]
    cols = [f"{classes[c]['stats_name']}_m2" for c in order]
    return names, cols, cfg["cv_folds"]


def site_table(stats_csv, cols, names, flight_agg):
    """Labelled area [m²] per site (rows) and species (columns), plus flights per site."""
    df = pd.read_csv(stats_csv)
    df["site_id"] = df["site"].str.rsplit("_", n=1).str[0]
    per_flight = df.groupby(["site_id", "site"])[cols].sum()
    area = per_flight.groupby("site_id").agg(flight_agg)
    area.columns = names
    n_flights = per_flight.reset_index().groupby("site_id")["site"].nunique()
    return area, n_flights


def order_by_fold(area, folds):
    """Sites grouped by CV fold (fold order, site order as in the config); unassigned last."""
    rows, fold_of = [], {}
    for fold, sites in folds.items():
        for s in sites:
            if s in area.index:
                rows.append(s)
                fold_of[s] = fold
    for s in sorted(set(area.index) - set(rows)):
        rows.append(s)
        fold_of[s] = "none"
    return area.loc[rows], fold_of


def heatmap(ax, values, row_labels, col_labels, vmax=100):
    """Annotated % heatmap; zero cells white, text in ink tokens chosen for contrast."""
    shown = np.ma.masked_where(values <= 0, values)
    cmap = CMAP.copy()
    cmap.set_bad("white")
    im = ax.imshow(shown, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if v <= 0:
                continue
            text = "<1" if v < 1 else f"{v:.0f}"
            color = "white" if v > 0.55 * vmax else INK
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontstyle="italic")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    # # thin white cell gaps instead of a grid
    ax.set_xticks(np.arange(-0.5, values.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, values.shape[0]), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0, colors=INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def mark_folds(ax, sites, fold_of):
    """Separator lines between folds and the fold name left of each block."""
    start = 0
    for i in range(1, len(sites) + 1):
        if i == len(sites) or fold_of[sites[i]] != fold_of[sites[start]]:
            if i < len(sites):
                ax.axhline(i - 0.5, color=INK, linewidth=1.2)
            ax.text(1.02, (start + i - 1) / 2, fold_of[sites[start]], transform=ax.get_yaxis_transform(),
                    ha="left", va="center", fontsize=9, fontweight="bold", color=INK_MUTED)
            start = i


def titles(ax, title, subtitle):
    """Bold title + muted subtitle, left-aligned above the axes at fixed point offsets."""
    kw = dict(xy=(0, 1), xycoords="axes fraction", textcoords="offset points", ha="left", va="bottom")
    ax.annotate(title, xytext=(0, 24), fontsize=12, fontweight="bold", color=INK, **kw)
    ax.annotate(subtitle, xytext=(0, 8), fontsize=9, color=INK_MUTED, **kw)


def save(fig, out_dir, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_dir}/{name}.png|pdf")


def plot_site_heatmap(area, n_flights, fold_of, normalise, title, subtitle, out_dir, name):
    """normalise='species': each column sums to 100. normalise='site': each row sums to 100."""
    vals = area.to_numpy(dtype=float)
    if normalise == "species":
        vals = 100 * vals / np.where(vals.sum(0) > 0, vals.sum(0), 1)
    else:
        vals = 100 * vals / np.where(vals.sum(1, keepdims=True) > 0, vals.sum(1, keepdims=True), 1)
    sites = list(area.index)
    row_labels = [f"{s} ({n_flights[s]}×)" for s in sites]
    col_labels = list(area.columns)
    if normalise == "species":
        col_labels = [f"{c}\n{t:,.0f} m²" for c, t in zip(col_labels, area.sum(0))]
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(sites) + 2))
    im = heatmap(ax, vals, row_labels, col_labels)
    mark_folds(ax, sites, fold_of)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.025, pad=0.12, aspect=40)
    cbar.set_label("%", color=INK)
    cbar.outline.set_visible(False)
    titles(ax, title, subtitle)
    save(fig, out_dir, name)


def plot_fold_holdout(area, folds, out_dir):
    """% of each species' area that lies in the held-out sites of each fold."""
    total = area.sum(0)
    rows = {f: area.loc[[s for s in sites if s in area.index]].sum(0) for f, sites in folds.items()}
    held = pd.DataFrame(rows).T
    vals = 100 * held.to_numpy(dtype=float) / np.where(total.to_numpy() > 0, total.to_numpy(), 1)
    labels = [f"{f}: {', '.join(folds[f])}" for f in held.index]
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    im = heatmap(ax, vals, labels, list(area.columns))
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("% held out", color=INK)
    cbar.outline.set_visible(False)
    titles(ax, "What each CV fold takes away from training", "% of each species' labelled area located in the fold's 4 held-out sites "
                 "(high = the model trains on little of that species)")
    save(fig, out_dir, "fold_holdout")
    return pd.DataFrame(vals, index=held.index, columns=area.columns)


def plot_concentration(area, out_dir):
    """Effective number of sites exp(H) per species (1 = all area at one site), top-site share as label."""
    share = area / area.sum(0)
    p = share.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.nansum(np.where(p > 0, p * np.log(p), 0), axis=0)
    eff = np.exp(h)
    n_sites = (area > 0).sum(0).to_numpy()
    top_site = share.idxmax(0)
    top_share = 100 * share.max(0)

    species = list(area.columns)
    y = np.arange(len(species))[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.barh(y, eff, height=0.5, color=BLUE_RAMP[4], zorder=2)
    for yi, e, n, ts, tv in zip(y, eff, n_sites, top_site, top_share):
        ax.text(e + 0.15, yi, f"{e:.1f} of {n} sites  ·  top: {ts} {tv:.0f}%",
                va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(species, fontstyle="italic")
    ax.set_xlim(0, max(eff.max(), 1) * 2.1)
    ax.set_xlabel("effective number of sites  exp(entropy of area over sites)", color=INK)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(length=0, colors=INK)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    titles(ax, "How concentrated is each species?", "1 = all labelled area at one site; n = area spread evenly over n sites")
    save(fig, out_dir, "species_concentration")
    return pd.DataFrame({"n_sites": n_sites, "eff_sites": eff, "top_site": top_site, "top_share_%": top_share},
                        index=species)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stats-csv", default="../data/Neophytes/stats_imagewise.csv")
    parser.add_argument("--classes-yaml", default="configs/data/neophytes_names_colors.yaml")
    parser.add_argument("--out-dir", default="results/dataset_viz")
    parser.add_argument("--flight-agg", choices=["sum", "mean"], default="sum",
                        help="combine the flights of a site: sum = area seen in training, mean = area per flight")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "figure.facecolor": "white"})

    # %% data
    names, cols, folds = load_config(args.classes_yaml)
    area, n_flights = site_table(args.stats_csv, cols, names, args.flight_agg)
    area, fold_of = order_by_fold(area, folds)
    area.assign(flights=n_flights).to_csv(os.path.join(args.out_dir, "site_species_area_m2.csv"))
    agg_note = "flights summed" if args.flight_agg == "sum" else "mean over flights"

    # %% figures
    plot_site_heatmap(area, n_flights, fold_of, "species",
                      "Where does each species come from?",
                      f"% of each species' labelled area per site (columns sum to 100; {agg_note}; "
                      "(n×) = flights; total area below)",
                      args.out_dir, "site_species_share")
    plot_site_heatmap(area, n_flights, fold_of, "site",
                      "What is labelled at each site?",
                      f"% of each site's labelled area per species (rows sum to 100; {agg_note})",
                      args.out_dir, "site_composition")
    holdout = plot_fold_holdout(area, folds, args.out_dir)
    conc = plot_concentration(area, args.out_dir)

    # %% tables
    pd.set_option("display.width", 160)
    print("\n% of species area held out per fold:\n", holdout.round(1))
    print("\nConcentration per species:\n", conc.round(2))


if __name__ == "__main__":
    main()
