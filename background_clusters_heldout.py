#!/usr/bin/env python3
"""Do the held-out sites of a fold have the same background types as its training sites?

The training clusters of ``background_clusters.py --pool ... --fold <fold>`` are refitted
here with the same PCA, k-means and seed (the labels are checked against pool_<fold>.csv),
and every pool window of the held-out sites is assigned to its nearest training centroid.
Nothing is fitted on the held-out sites.

    python background_clusters_heldout.py --pool results/background_pool --fold cv3

Outputs (``--out-dir``/pool_<fold>_k<k>_heldout/):

    summary.md           per cluster: type, share of the training and of the held-out windows,
                         median distance to the centroid (training vs held-out) and the share of
                         held-out windows farther away than 95% of the training windows
    heldout_<fold>.csv   the pool rows of the held-out sites + cluster + dist
    montages/cluster_XX.png    20 held-out windows per cluster (12 nearest the centre + 8 random)
    compare/cluster_XX.png     the training montage above the held-out montage

The montages show confidential data: view them on the node, keep them inside the course.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.windows import Window

from background_clusters import montages


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", default="results/background_pool", help="output dir of background_pool.py")
    parser.add_argument("--fold", default="cv3")
    parser.add_argument("--clusters-dir", default="results/background_clusters/pool_cv3_k25",
                        help="output of background_clusters.py for this fold (pool_<fold>.csv, montages/)")
    parser.add_argument("--types-csv", default="configs/synthesis/cluster_types_pool_cv3_k25.csv")
    parser.add_argument("--classes-yaml", default="configs/data/neophytes_names_colors.yaml")
    parser.add_argument("--out-dir", default="results/background_clusters")
    parser.add_argument("--k", type=int, default=25)
    parser.add_argument("--n-show", type=int, default=20, help="windows per montage")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    out = os.path.join(args.out_dir, f"pool_{args.fold}_k{args.k}_heldout")
    os.makedirs(os.path.join(out, "montages"), exist_ok=True)
    os.makedirs(os.path.join(out, "compare"), exist_ok=True)

    # %% refit the training clusters exactly as background_clusters.py does
    with open(args.classes_yaml) as f:
        held_out = set(yaml.safe_load(f)["cv_folds"][args.fold])
    pool = pd.read_csv(os.path.join(args.pool, "pool.csv"))
    is_ho = pool["site"].isin(held_out).to_numpy()
    x = np.load(os.path.join(args.pool, "emb.npz"))["emb"].astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    pca = PCA(50, random_state=args.seed).fit(x[~is_ho])
    x_tr, x_ho = pca.transform(x[~is_ho]), pca.transform(x[is_ho])
    km = KMeans(args.k, n_init=10, random_state=args.seed).fit(x_tr)
    train = pd.read_csv(os.path.join(args.clusters_dir, f"pool_{args.fold}.csv"),
                        usecols=["img_path", "y0", "x0", "site", "cluster"])
    assert len(train) == len(x_tr) and (train["img_path"].to_numpy() == pool.loc[~is_ho, "img_path"].to_numpy()).all()
    agree = (km.labels_ == train["cluster"].to_numpy()).mean()
    print(f"refit vs pool_{args.fold}.csv: {agree:.4%} of training windows get the same cluster")
    assert agree > 0.999, "the refit does not reproduce the training clusters; the held-out assignment would not match"

    # %% assign the held-out windows to the nearest training centroid
    ho = pool[is_ho].reset_index(drop=True)
    ho["cluster"] = km.predict(x_ho)
    ho["dist"] = np.linalg.norm(x_ho - km.cluster_centers_[ho["cluster"]], axis=1)
    train["dist"] = np.linalg.norm(x_tr - km.cluster_centers_[km.labels_], axis=1)
    ho.to_csv(os.path.join(out, f"heldout_{args.fold}.csv"), index=False)
    print(f"{len(ho)} held-out windows from {ho['site'].nunique()} sites ({', '.join(sorted(held_out))})")

    types = pd.read_csv(args.types_csv).set_index("cluster")
    summary_md(train, ho, types, args, out)

    # %% montages of the held-out windows: n-show per cluster, 60% nearest the centre + 40% random
    n_near = round(0.6 * args.n_show)
    picks = {}
    for c, g in ho.groupby("cluster"):
        near = g.nsmallest(n_near, "dist")
        rest = g.drop(near.index)
        picks[c] = pd.concat([near, rest.sample(min(args.n_show - n_near, len(rest)), random_state=args.seed)])
    size = int(ho["size"].iloc[0])
    step = max(1, size // 256)
    crops = {}
    for i, r in pd.concat(picks.values()).iterrows():
        with rasterio.open(r["img_path"]) as src:
            img = src.read([1, 2, 3], window=Window(int(r["x0"]), int(r["y0"]), size, size))
        crops[i] = img.transpose(1, 2, 0)[::step, ::step]
    montages(picks, crops, ho, n_near, out)

    # %% training montage above the held-out montage, one page per cluster
    for c in range(args.k):
        panels = [("TRAINING sites", os.path.join(args.clusters_dir, "montages", f"cluster_{c:02d}.png")),
                  ("HELD-OUT sites", os.path.join(out, "montages", f"cluster_{c:02d}.png"))]
        fig, axes = plt.subplots(2, 1, figsize=(15, 26))
        for ax, (title, path) in zip(axes, panels):
            ax.axis("off")
            if os.path.exists(path):
                ax.imshow(plt.imread(path))
            else:
                ax.text(0.5, 0.5, "no windows", ha="center", va="center", fontsize=20)
            ax.set_title(title, fontsize=16, weight="bold")
        t = types.loc[c]
        fig.suptitle(f"cluster {c}: {t['type']} ({'usable' if t['usable'] else 'not usable'})", fontsize=20)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "compare", f"cluster_{c:02d}.png"), dpi=80)
        plt.close(fig)
    print(f"written to {out}")


def summary_md(train, ho, types, args, out):
    """summary.md: cluster and type shares and distances, training vs held-out sites."""
    rows = []
    for c in range(args.k):
        d_tr = train.loc[train["cluster"] == c, "dist"]
        d_ho = ho.loc[ho["cluster"] == c, "dist"]
        p95 = d_tr.quantile(0.95)
        rows.append({"cluster": c, "type": types.loc[c, "type"], "usable": types.loc[c, "usable"],
                     "train share": f"{len(d_tr) / len(train):.1%}", "held-out share": f"{len(d_ho) / len(ho):.1%}",
                     "held-out sites": ho.loc[ho["cluster"] == c, "site"].nunique(),
                     "median dist train": round(d_tr.median(), 3),
                     "median dist held-out": round(d_ho.median(), 3) if len(d_ho) else None,
                     "held-out > train p95": f"{(d_ho > p95).mean():.0%}" if len(d_ho) else None})
    by_cluster = pd.DataFrame(rows).to_markdown(index=False)

    tr_t = train["cluster"].map(types["type"]).value_counts(normalize=True)
    ho_t = ho["cluster"].map(types["type"]).value_counts(normalize=True)
    by_type = pd.DataFrame({"train share": tr_t, "held-out share": ho_t}).fillna(0) \
        .sort_values("train share", ascending=False).map(lambda v: f"{v:.1%}").to_markdown()

    # # per held-out site: type shares, to see whether one site drives the match
    by_site = pd.crosstab(ho["cluster"].map(types["type"]), ho["site"], normalize="columns") \
        .map(lambda v: f"{v:.1%}").to_markdown()

    # # overall: is a held-out window as close to its centroid as a training window?
    p95 = train.groupby("cluster")["dist"].quantile(0.95)
    far = (ho["dist"] > ho["cluster"].map(p95)).mean()
    text = (f"# Held-out sites in the training clusters ({args.fold}, k={args.k})\n\n"
            f"{len(train)} training windows ({train['site'].nunique()} sites), {len(ho)} held-out windows "
            f"({ho['site'].nunique()} sites), each assigned to its nearest training centroid (50-d PCA space).\n\n"
            f"Median distance to the centroid: training {train['dist'].median():.3f}, held-out {ho['dist'].median():.3f}. "
            f"Held-out windows farther away than 95% of their cluster's training windows: {far:.1%} "
            f"(5% expected if the held-out sites look like the training sites).\n\n"
            f"## By type\n\n{by_type}\n\n## By cluster\n\n{by_cluster}\n\n## Held-out type shares per site\n\n{by_site}\n")
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
