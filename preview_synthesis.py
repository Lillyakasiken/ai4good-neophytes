#!/usr/bin/env python3
"""Preview synthesis.synthesize(): every species pasted onto one background per cluster.

A look-only tool to judge the compositing before it goes into training. One random
background window per cluster of a background_clusters.py run; onto it, one random
plant of each of the six species, each from a training tile of another site
(preview_paste.pick_instance: connected mask region, scaled to the background GSD,
rotated/flipped, placed mostly on background). Only training sites of --fold are used.

One figure per cluster, one column per species:

    row 1  source: the plant outlined in its own tile
    row 2  hard paste (red box = pasted plant)
    row 3  soft paste
    row 4  label of the soft paste (black background, class colour, grey ignore ring)

The first column shows the background window alone.

    python preview_synthesis.py
    python preview_synthesis.py --clusters results/background_clusters/dino_512_k25/clusters.csv --seed 1

Output (``--out-dir``, default results/synthesis_preview): cluster_XX.png and index.csv.
The figures show confidential data: view them on the node, keep them inside the course.
"""

import argparse
import os
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from background_clusters import kept_windows
from preview_paste import load_config, pick_instance, place, read_tile, show_outlined, training_tiles
from synthesis import IGNORE, synthesize


def label_rgb(mask, classes):
    """Label mask -> RGB: class colours, ignore grey."""
    cmap = ListedColormap(["#000000"] + [classes[c]["color"] for c in range(1, 7)])
    rgb = cmap(np.clip(mask, 0, 6))[..., :3]
    rgb[mask == IGNORE] = 0.6
    return rgb


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clusters", default="results/background_clusters/dino_512_k25/clusters.csv",
                        help="clusters.csv of a background_clusters.py run (probe-cache mode)")
    parser.add_argument("--stats-csv", default="../data/Neophytes/stats_imagewise.csv")
    parser.add_argument("--classes-yaml", default="configs/data/neophytes_names_colors.yaml")
    parser.add_argument("--out-dir", default="results/synthesis_preview")
    parser.add_argument("--fold", default="cv3", help="plants only from training sites of this fold")
    parser.add_argument("--size", type=int, default=512, help="window size of the clustering run")
    parser.add_argument("--pos-thr", type=float, default=0.01, help="as in the clustering run")
    parser.add_argument("--min-area-m2", type=float, default=0.002, help="smallest usable plant region")
    parser.add_argument("--max-size-frac", type=float, default=0.6, help="largest plant side as fraction of the window")
    parser.add_argument("--min-bg-frac", type=float, default=0.8, help="share of the plant that must land on background")
    parser.add_argument("--sigma", type=float, default=2.0, help="soft paste: Gaussian blur of the mask edge [px]")
    parser.add_argument("--ring", type=int, default=3, help="ignore ring around the pasted plant [px]")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    classes, folds = load_config(args.classes_yaml)
    tiles = training_tiles(args.stats_csv, folds, args.fold, classes)
    mask_of = pd.read_csv(args.stats_csv).set_index("img_path")["mask_path"]
    clusters = pd.read_csv(args.clusters)
    inst_args = SimpleNamespace(min_area_m2=args.min_area_m2, max_size_frac=args.max_size_frac)

    rows = []
    for c, g in clusters.groupby("cluster"):
        # %% one random background window of the cluster
        w = g.iloc[rng.integers(len(g))]
        tile = {"img_path": w["img_path"], "mask_path": mask_of[w["img_path"]]}
        image, pos = kept_windows(tile, args.size, args.pos_thr)
        by, bx = pos[int(w["rank"])]
        _, full_mask, gsd = read_tile(tile["img_path"], tile["mask_path"])
        background = image[by:by + args.size, bx:bx + args.size]
        bg_mask = full_mask[by:by + args.size, bx:bx + args.size]

        # %% one random plant per species, hard + soft
        fig, axes = plt.subplots(4, 7, figsize=(28, 16.5))
        for ax in axes.ravel():
            ax.set_axis_off()
        axes[0, 0].imshow(background)
        axes[0, 0].set_title(f"background {w['flight']}", fontsize=10)
        for col, cls in enumerate(range(1, 7), start=1):
            name, stats_name = classes[cls]["name"], classes[cls]["stats_name"]
            inst = pick_instance(tiles, cls, stats_name, w["site"], gsd, args.size, inst_args, rng)
            yx = None if inst is None else place(bg_mask, inst[2], args.min_bg_frac, rng)
            if yx is None:
                axes[0, col].text(0.5, 0.5, f"no {name}\ncould be placed", ha="center", va="center",
                                  fontsize=11, transform=axes[0, col].transAxes)
                print(f"cluster {c}: no {name} placed")
                continue
            s_row, patch, alpha, (src_img, src_outline), scale = inst
            show_outlined(axes[0, col], src_img, src_outline, f"{name} @ {s_row['site_id']} (×{scale:.2f})")
            soft_mask = None
            for r, method in ((1, "hard"), (2, "soft")):
                out, out_mask = synthesize(background, bg_mask, patch, alpha, cls, *yx, method=method,
                                           sigma=args.sigma, ring=args.ring)
                axes[r, col].imshow(out)
                axes[r, col].add_patch(plt.Rectangle((yx[1], yx[0]), alpha.shape[1], alpha.shape[0], fill=False,
                                                     edgecolor="red", linewidth=1, linestyle="--"))
                axes[r, col].set_title(f"{method} paste" + (f" (σ={args.sigma:g} px)" if method == "soft" else ""),
                                       fontsize=10)
                soft_mask = out_mask
            axes[3, col].imshow(label_rgb(soft_mask, classes))
            axes[3, col].set_title(f"label (ignore ring {args.ring} px)", fontsize=10)
            rows.append({"cluster": c, "background_tile": w["img_path"], "bg_y0": by, "bg_x0": bx,
                         "background_flight": w["flight"], "class": name, "source_site": s_row["site_id"],
                         "source_tile": s_row["img_path"], "scale": round(scale, 3), "y": yx[0], "x": yx[1],
                         "h": alpha.shape[0], "w": alpha.shape[1]})
        fig.suptitle(f"cluster {c} ({len(g)} windows) · background {w['flight']} at ({by}, {bx}), "
                     f"{gsd:.1f} mm/px", x=0.01, ha="left", fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"cluster_{c:02d}.png"), dpi=80)
        plt.close(fig)
        print(f"cluster {c}: done", flush=True)

    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "index.csv"), index=False)
    print(f"wrote {clusters['cluster'].nunique()} figures and index.csv to {args.out_dir}")


if __name__ == "__main__":
    main()
