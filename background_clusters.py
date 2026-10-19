#!/usr/bin/env python3
"""Cluster background windows of the training sites: surface types or sites?

Copy-paste synthesis should put every species on every background type. The
types are found here: DINOv2 CLS embeddings of plant-free windows are grouped
with k-means, and three outputs show whether a cluster is a surface type
(ballast, grass, soil, ...) spread over many sites, or just one site / its
lighting.

The embeddings are not recomputed: they come from the train cache of
plantclef_probe.py (branch plantclef-experiments, emb_train.npz). Its tile table
and window order are rebuilt here exactly as there (same split, seed, bg
fraction and pos-thr); the asserts check this against the cache.

    python background_clusters.py --emb ../ai4good-neophytes/results/plantclef_probe/emb_train.npz
    python background_clusters.py ... --size 1024 --k 20

Outputs (``--out-dir``/<backbone>_<size>_k<k>/):

    tsne.png             one dot per window: left coloured by cluster (number at its centre),
                         right by site. A cluster in many colours = type, in one colour = site.
    clusters.md          per cluster: windows, number of sites, largest site and its share
    montages/cluster_XX.png   20 windows per cluster (12 nearest the centre + 8 random)
    clusters.csv         per window: cluster, tile, site, flight (for the synthesis loader)

The montages show confidential data: view them on the node, keep them inside the course.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import yaml

from preview_paste import IGNORE, read_tile

N_CLS = 7  # background + 6 species, as in plantclef_probe.py


def split_tiles(stats_csv, split_yaml, bg_fraction, rng):
    """Train tile table, identical to plantclef_probe.split_tiles (first rng draw, no max-tiles)."""
    with open(split_yaml) as f:
        dirs = yaml.safe_load(f)["image_dir"]["train"]
    df = pd.read_csv(stats_csv)
    d = df[df["img_path"].str.startswith(tuple(p.rstrip("/") + "/" for p in dirs))]
    lab, bg = d[d["only_background"] == 0], d[d["only_background"] == 1]
    n_bg = min(len(bg), round(bg_fraction / (1 - bg_fraction) * len(lab)))
    return pd.concat([lab, bg.sample(n_bg, random_state=int(rng.integers(1 << 31)))]).reset_index(drop=True)


def window_label(gt, pos_thr):
    """As plantclef_probe.window_label: -1 = skipped, 0 = background, else dominant species."""
    px = np.bincount(gt.ravel(), minlength=N_CLS + 1)[:N_CLS]
    if px.sum() < 0.5 * gt.size:
        return -1
    if px[1:].sum() == 0:
        return 0
    c = int(px[1:].argmax()) + 1
    return c if px[c] >= pos_thr * px.sum() else -1


def kept_windows(row, size, pos_thr):
    """RGB and the (y0, x0) of every window of one size that plantclef_probe kept, in its order."""
    image, mask, _ = read_tile(row["img_path"], row["mask_path"])
    with rasterio.open(row["img_path"]) as src:
        alpha = src.read(4) if src.count >= 4 else np.full(mask.shape, 255, np.uint8)
    gt = np.where((mask != IGNORE) & (alpha > 0), mask, N_CLS)
    pos = [(y0, x0) for y0 in range(0, gt.shape[0] - size + 1, size)
           for x0 in range(0, gt.shape[1] - size + 1, size)
           if window_label(gt[y0:y0 + size, x0:x0 + size], pos_thr) >= 0]
    return image, pos


def colours(n):
    """n distinguishable colours (tab20 + tab20b)."""
    pal = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)
    return [pal[i % len(pal)] for i in range(n)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--emb", default="../ai4good-neophytes/results/plantclef_probe/emb_train.npz",
                        help="train cache of plantclef_probe.py")
    parser.add_argument("--stats-csv", default="../data/Neophytes/stats_imagewise.csv")
    parser.add_argument("--split-yaml", default="configs/data/split_cv3.yaml")
    parser.add_argument("--out-dir", default="results/background_clusters")
    parser.add_argument("--backbone", default="dino", choices=["dino", "pc"], help="DINOv2 or PlantCLEF embeddings")
    parser.add_argument("--size", type=int, default=512, help="window size (256, 512 or 1024)")
    parser.add_argument("--k", type=int, default=25, help="clusters; merge them into types by hand afterwards")
    parser.add_argument("--n-show", type=int, default=20, help="windows per montage")
    parser.add_argument("--max-tsne", type=int, default=20000, help="windows shown in the t-SNE plot")
    # # must match the plantclef_probe.py run that wrote --emb
    parser.add_argument("--bg-fraction", type=float, default=0.10)
    parser.add_argument("--pos-thr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    out = os.path.join(args.out_dir, f"{args.backbone}_{args.size}_k{args.k}")
    os.makedirs(os.path.join(out, "montages"), exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # %% background windows of the cache + their tile, site and flight
    tiles = split_tiles(args.stats_csv, args.split_yaml, args.bg_fraction, rng)
    emb = dict(np.load(args.emb))
    assert emb["tile"].max() < len(tiles), "tile table does not match the cache (different split/seed?)"
    win = pd.DataFrame({"tile": emb["tile"], "size": emb["size"], "label": emb["label"]})
    # # rank of each window among the kept windows of its tile and size = its position in the grid scan
    win["rank"] = win.groupby(["tile", "size"]).cumcount()
    sel = np.flatnonzero((win["label"] == 0) & (win["size"] == args.size))
    win = win.iloc[sel].reset_index(drop=True)
    win["flight"] = tiles["site"].to_numpy()[win["tile"]]
    win["site"] = win["flight"].str.rsplit("_", n=1).str[0]
    print(f"{len(tiles)} train tiles; {len(win)} background windows of {args.size} px from {win['tile'].nunique()} tiles, "
          f"{win['site'].nunique()} sites, {win['flight'].nunique()} flights")

    # %% k-means on L2-normalised CLS embeddings (cosine geometry), PCA to 50 dims against noise
    x = emb[args.backbone][sel].astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    x = PCA(50, random_state=args.seed).fit_transform(x)
    km = KMeans(args.k, n_init=10, random_state=args.seed).fit(x)
    win["cluster"] = km.labels_
    win["dist"] = np.linalg.norm(x - km.cluster_centers_[km.labels_], axis=1)
    win[["cluster", "tile", "rank", "site", "flight"]].assign(img_path=tiles["img_path"].to_numpy()[win["tile"]]) \
        .to_csv(os.path.join(out, "clusters.csv"), index=False)

    # %% clusters.md: is a cluster spread over sites (type) or concentrated on one (site)?
    rows = []
    for c, g in win.groupby("cluster"):
        top = g["site"].value_counts()
        rows.append({"cluster": c, "windows": len(g), "sites": len(top),
                     "largest site": top.index[0], "share": f"{top.iloc[0] / len(g):.0%}"})
    table = pd.DataFrame(rows).to_markdown(index=False)
    with open(os.path.join(out, "clusters.md"), "w") as f:
        f.write(f"# Background clusters ({args.backbone}, {args.size} px, k={args.k})\n\n"
                f"{len(win)} windows, {win['site'].nunique()} sites. Many sites and a low share = surface type; "
                f"few sites and a high share = site/lighting cluster.\n\n{table}\n")
    print(table)

    # %% tsne.png: same dots, coloured by cluster (left) and by site (right)
    show = rng.choice(len(win), min(args.max_tsne, len(win)), replace=False)
    xy = TSNE(2, init="pca", perplexity=30, random_state=args.seed).fit_transform(x[show])
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    col_c = colours(args.k)
    axes[0].scatter(*xy.T, s=3, c=[col_c[c] for c in win["cluster"].to_numpy()[show]])
    for c in range(args.k):
        m = win["cluster"].to_numpy()[show] == c
        if m.any():
            axes[0].text(*np.median(xy[m], 0), str(c), fontsize=13, weight="bold", ha="center", va="center")
    sites = sorted(win["site"].unique())
    col_s = dict(zip(sites, colours(len(sites))))
    for s in sites:
        m = win["site"].to_numpy()[show] == s
        axes[1].scatter(*xy[m].T, s=3, color=col_s[s], label=s)
    axes[1].legend(markerscale=5, fontsize=9, loc="center left", bbox_to_anchor=(1, 0.5))
    axes[0].set_title("coloured by k-means cluster")
    axes[1].set_title("coloured by site")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(out, "tsne.png"), dpi=120)
    plt.close(fig)

    # %% montages: n-show windows per cluster, 60% nearest the centre + 40% random
    n_near = round(0.6 * args.n_show)
    picks = {}
    for c, g in win.groupby("cluster"):
        near = g.nsmallest(n_near, "dist")
        rest = g.drop(near.index)
        picks[c] = pd.concat([near, rest.sample(min(args.n_show - n_near, len(rest)), random_state=args.seed)])
    crops = {}
    for t, g in pd.concat(picks.values()).groupby("tile"):
        image, pos = kept_windows(tiles.iloc[t], args.size, args.pos_thr)
        assert len(pos) == (emb["tile"][emb["size"] == args.size] == t).sum(), f"window order differs on tile {t}"
        for i, r in g.iterrows():
            y0, x0 = pos[int(r["rank"])]
            step = max(1, args.size // 256)
            crops[i] = image[y0:y0 + args.size:step, x0:x0 + args.size:step]
    ncol = 5
    for c, g in picks.items():
        nrow = int(np.ceil(len(g) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 3.3 * nrow), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (i, r) in zip(axes.ravel(), g.iterrows()):
            ax.imshow(crops[i])
            ax.set_title(("centre  " if i in g.index[:n_near] else "random  ") + r["flight"], fontsize=9)
        fig.suptitle(f"cluster {c}: {(win['cluster'] == c).sum()} windows, "
                     f"{win.loc[win['cluster'] == c, 'site'].nunique()} sites", fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "montages", f"cluster_{c:02d}.png"), dpi=100)
        plt.close(fig)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
