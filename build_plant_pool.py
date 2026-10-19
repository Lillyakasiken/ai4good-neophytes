#!/usr/bin/env python3
"""Plant pool for copy-paste synthesis: every usable plant instance of a fold's training tiles.

An instance is a connected region of one class in a mask (instances from semantic
masks, as preview_paste.find_region), kept if it does not touch the tile border,
covers >= --min-area-m2 and is < 20% ignore pixels -- the same rules as the preview.
Only the tiles of the fold's training split (train + test folders of the training
sites, split_<fold>.yaml) are read, so no held-out site and no val tile enters.

Per instance only references are stored; the synthetic dataset reads just the
bounding box at training time and recovers the region from the seed pixel.

    python build_plant_pool.py --fold cv3

Outputs (``--out-dir``/<fold>/): plants.csv (img_path, mask_path, cls, site, flight,
gsd_mm, y0, x0, y1, x1, seed_y, seed_x, area_px, area_m2) and counts.md (instances per
species x site). Derived from the confidential dataset: keep it on the node.
"""

import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import yaml
from scipy import ndimage

from preview_paste import IGNORE, read_tile

MIN_AREA_M2 = 0.002


def tile_instances(row):
    """All usable instances of one tile (one dict per instance)."""
    image_path, mask_path = row
    _, mask, gsd = read_tile(image_path, mask_path)
    out = []
    for cls in range(1, 7):
        comp, n = ndimage.label(mask == cls)
        if n == 0:
            continue
        areas = np.bincount(comp.ravel())
        for i, sl in enumerate(ndimage.find_objects(comp), start=1):
            touches = (sl[0].start == 0 or sl[1].start == 0
                       or sl[0].stop == mask.shape[0] or sl[1].stop == mask.shape[1])
            area_m2 = areas[i] * (gsd / 1000.0) ** 2
            if touches or area_m2 < MIN_AREA_M2 or (mask[sl] == IGNORE).mean() >= 0.2:
                continue
            ys, xs = np.nonzero(comp[sl] == i)
            out.append({"img_path": image_path, "mask_path": mask_path, "cls": cls, "gsd_mm": gsd,
                        "y0": sl[0].start, "x0": sl[1].start, "y1": sl[0].stop, "x1": sl[1].stop,
                        "seed_y": sl[0].start + int(ys[0]), "seed_x": sl[1].start + int(xs[0]),
                        "area_px": int(areas[i]), "area_m2": area_m2})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fold", default="cv3")
    parser.add_argument("--stats-csv", default="../data/Neophytes/stats_imagewise.csv")
    parser.add_argument("--classes-yaml", default="configs/data/neophytes_names_colors.yaml")
    parser.add_argument("--out-dir", default="results/plant_pool")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    out = os.path.join(args.out_dir, args.fold)
    os.makedirs(out, exist_ok=True)
    with open(f"configs/data/split_{args.fold}.yaml") as f:
        train_dirs = tuple(p.rstrip("/") + "/" for p in yaml.safe_load(f)["image_dir"]["train"])
    with open(args.classes_yaml) as f:
        classes = yaml.safe_load(f)["classes"]

    # %% labelled tiles of the fold's training split
    stats = pd.read_csv(args.stats_csv)
    tiles = stats[stats["img_path"].str.startswith(train_dirs) & (stats["only_background"] == 0)]
    print(f"{args.fold}: {len(tiles)} labelled training tiles", flush=True)
    with Pool(args.workers) as pool:
        rows = [r for t in pool.imap(tile_instances, zip(tiles["img_path"], tiles["mask_path"]), chunksize=8)
                for r in t]
    plants = pd.DataFrame(rows)
    plants["flight"] = plants["img_path"].map(tiles.set_index("img_path")["site"])
    plants["site"] = plants["flight"].str.rsplit("_", n=1).str[0]
    plants.to_csv(os.path.join(out, "plants.csv"), index=False)

    # %% counts.md: instances per species x site
    plants["species"] = plants["cls"].map(lambda c: classes[c]["name"])
    tab = plants.groupby(["species", "site"]).size().unstack(fill_value=0)
    tab["total"] = tab.sum(1)
    tab["sites"] = (tab.drop(columns="total") > 0).sum(1)
    lines = ["| species | " + " | ".join(tab.columns) + " |", "|---" * (len(tab.columns) + 1) + "|"]
    lines += ["| " + s + " | " + " | ".join(str(v) for v in r) + " |" for s, r in zip(tab.index, tab.to_numpy())]
    with open(os.path.join(out, "counts.md"), "w") as f:
        f.write(f"# Plant instances, {args.fold} training split\n\n{len(plants)} instances from "
                f"{plants['img_path'].nunique()} tiles (min area {MIN_AREA_M2} m², not touching the tile "
                f"border, < 20% ignore).\n\n" + "\n".join(lines) + "\n")
    print(open(os.path.join(out, "counts.md")).read())


if __name__ == "__main__":
    main()
