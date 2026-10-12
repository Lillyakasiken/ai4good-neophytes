#!/usr/bin/env python3
"""Preview copy-paste synthesis: one real plant pasted onto a crop of another site.

A look-only tool to judge whether cut-and-paste composites look realistic before
anything goes into training. Nothing here is used by train.py.

Per example, one class is drawn uniformly. A 1024 crop of a training tile is the
canvas, and one plant of that class is pasted into it. The plant is a connected
region of the class in the mask of a training tile from a *different* site
(instances from semantic masks, as in Motoi et al. 2024), scaled to the canvas
GSD, rotated/flipped, and placed mostly on background. Two canvas types
alternate:

    O2  a normal training crop with its own labelled plants (Ghiasi et al. 2021)
    O1  a crop of a plant-free tile (background bank)

Each panel shows, left to right:

    1. real reference: a real plant of the same species at the canvas site, if
       that species-site combination exists in the training data
    2. source: the plant that is cut out, outlined in its own tile
    3. background: the canvas crop before pasting
    4. hard paste (binary mask), pasted plant in a red box
    5. soft paste (Gaussian-blurred mask edge), pasted plant in a red box

Only training sites of the chosen fold are used, never the held-out ones.

    python preview_paste.py                          # cv3, 30 examples
    python preview_paste.py --fold cv1 --n-examples 10 --seed 1

Output (``--out-dir``, default results/paste_preview): one PNG panel per example
and ``index.csv`` with one row per example. The panels show confidential data:
view them on the node, keep them inside the course.
"""

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import yaml
from scipy import ndimage

IGNORE = -1


def load_config(path):
    """Classes 1..6 (name, colour, stats name) and CV folds from the classes yaml."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    classes = {c: v for c, v in cfg["classes"].items() if c != 0}
    return classes, cfg["cv_folds"]


def training_tiles(stats_csv, folds, fold, classes):
    """Tiles of the fold's training sites (train + test folders; val stays out)."""
    df = pd.read_csv(stats_csv)
    df["site_id"] = df["site"].str.rsplit("_", n=1).str[0]
    held_out = set(folds[fold])
    df = df[~df["site_id"].isin(held_out) & df["split"].isin(["train", "test"])].copy()
    px_cols = [f"{v['stats_name']}_px" for v in classes.values()]
    df["fg_px"] = df[px_cols].sum(1)
    df["nodata_frac"] = df["nodata_px"] / df["total_px"]
    return df.reset_index(drop=True)


def read_tile(img_path, mask_path):
    """RGB uint8 (H, W, 3), class mask int64 with 254/255 -> IGNORE, GSD in mm/px."""
    with rasterio.open(img_path) as src:
        image = np.transpose(src.read([1, 2, 3]), (1, 2, 0))
        gsd = abs(src.transform.a) * 1000.0
    with rasterio.open(mask_path) as src:
        mask = src.read(1).astype(np.int64)
    mask[(mask == 254) | (mask == 255)] = IGNORE
    return image, mask, gsd


def find_region(row, cls, min_area_m2, rng, max_side_at_gsd=None):
    """A random connected region of class `cls` in one tile, or None.

    Usable regions don't touch the tile border, cover at least min_area_m2 and are
    less than 20% ignore pixels. max_side_at_gsd(gsd) optionally caps the bbox side
    in source pixels. Returns (image, gsd, bbox slices, region mask inside the bbox).
    """
    image, mask, gsd = read_tile(row["img_path"], row["mask_path"])
    comp, n = ndimage.label(mask == cls)
    if n == 0:
        return None
    areas = np.bincount(comp.ravel())
    max_side = max_side_at_gsd(gsd) if max_side_at_gsd else np.inf
    candidates = []
    for i, sl in enumerate(ndimage.find_objects(comp), start=1):
        touches = (sl[0].start == 0 or sl[1].start == 0
                   or sl[0].stop == mask.shape[0] or sl[1].stop == mask.shape[1])
        side = max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start)
        if (not touches and areas[i] * (gsd / 1000.0) ** 2 >= min_area_m2
                and (mask[sl] == IGNORE).mean() < 0.2 and side <= max_side):
            candidates.append((i, sl))
    if not candidates:
        return None
    i, sl = candidates[rng.integers(len(candidates))]
    return image, gsd, sl, comp[sl] == i


def context_crop(image, sl, region):
    """The region's surroundings (half its size on each side + 32 px) and its outline mask."""
    pad = max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) // 2 + 32
    y0, x0 = max(sl[0].start - pad, 0), max(sl[1].start - pad, 0)
    y1, x1 = min(sl[0].stop + pad, image.shape[0]), min(sl[1].stop + pad, image.shape[1])
    outline = np.zeros((y1 - y0, x1 - x0), bool)
    outline[sl[0].start - y0:sl[0].stop - y0, sl[1].start - x0:sl[1].stop - x0] = region
    return image[y0:y1, x0:x1], outline


def pick_canvas(tiles, kind, crop, rng, max_tries=20):
    """A crop x crop window of a random training tile. O1: plant-free tile, O2: tile with plants,
    window centred on a random labelled pixel so the canvas shows its own plants."""
    pool = tiles[(tiles["fg_px"] == 0) if kind == "O1" else (tiles["fg_px"] > 0)]
    pool = pool[pool["nodata_frac"] < 0.1]
    if pool.empty:
        raise RuntimeError(f"no {kind} canvas tiles in this fold")
    for _ in range(max_tries):
        row = pool.iloc[rng.integers(len(pool))]
        image, mask, gsd = read_tile(row["img_path"], row["mask_path"])
        h, w = mask.shape
        if kind == "O2":
            ys, xs = np.nonzero(mask > 0)
            k = rng.integers(len(ys))
            y0 = int(np.clip(ys[k] - crop // 2, 0, h - crop))
            x0 = int(np.clip(xs[k] - crop // 2, 0, w - crop))
        else:
            y0, x0 = rng.integers(0, h - crop + 1), rng.integers(0, w - crop + 1)
        img_c = image[y0:y0 + crop, x0:x0 + crop].copy()
        # # skip windows in the empty orthophoto margin
        if (img_c.sum(-1) == 0).mean() < 0.05:
            return row, img_c, mask[y0:y0 + crop, x0:x0 + crop].copy(), gsd
    raise RuntimeError(f"no usable {kind} canvas found")


def pick_instance(tiles, cls, stats_name, canvas_site, canvas_gsd, crop, args, rng, max_tries=15):
    """A plant of class `cls` from a training tile of another site, scaled to the canvas GSD and
    randomly rotated/flipped. Returns (row, patch RGB, binary mask, source context, scale) or None."""
    pool = tiles[(tiles[f"{stats_name}_px"] > 0) & (tiles["site_id"] != canvas_site)]
    if pool.empty:
        return None
    for _ in range(max_tries):
        row = pool.iloc[rng.integers(len(pool))]
        jitter = rng.uniform(0.9, 1.1)
        found = find_region(row, cls, args.min_area_m2, rng,
                            max_side_at_gsd=lambda g: args.max_size_frac * crop / (g / canvas_gsd * jitter))
        if found is None:
            continue
        image, gsd, sl, region = found
        scale = gsd / canvas_gsd * jitter
        new_wh = (max(1, round(region.shape[1] * scale)), max(1, round(region.shape[0] * scale)))
        patch = cv2.resize(np.ascontiguousarray(image[sl]), new_wh, interpolation=cv2.INTER_CUBIC)
        alpha = cv2.resize(region.astype(np.float32), new_wh, interpolation=cv2.INTER_LINEAR) > 0.5
        # # random 90° rotation + flip, as in the training augmentation
        k = rng.integers(4)
        patch, alpha = np.rot90(patch, k), np.rot90(alpha, k)
        if rng.random() < 0.5:
            patch, alpha = patch[:, ::-1], alpha[:, ::-1]
        return (row, np.ascontiguousarray(patch), np.ascontiguousarray(alpha),
                context_crop(image, sl, region), scale)
    return None


def pick_reference(tiles, cls, stats_name, canvas_site, args, rng, max_tries=10):
    """A real plant of class `cls` at the canvas site (native resolution), or None if that
    species-site combination has no usable region in the fold's training data."""
    pool = tiles[(tiles[f"{stats_name}_px"] > 0) & (tiles["site_id"] == canvas_site)]
    for _ in range(min(max_tries, len(pool))):
        row = pool.iloc[rng.integers(len(pool))]
        found = find_region(row, cls, args.min_area_m2, rng)
        if found is not None:
            image, _, sl, region = found
            return row, context_crop(image, sl, region)
    return None


def place(canvas_mask, alpha, min_bg_frac, rng, max_tries=50):
    """Top-left corner where at least min_bg_frac of the pasted plant lands on background."""
    H, W = canvas_mask.shape
    h, w = alpha.shape
    for _ in range(max_tries):
        y, x = rng.integers(0, H - h + 1), rng.integers(0, W - w + 1)
        if (canvas_mask[y:y + h, x:x + w][alpha] == 0).mean() >= min_bg_frac:
            return y, x
    return None


def composite(canvas, patch, alpha, y, x, sigma):
    """Hard paste (binary mask) and soft paste (Gaussian-blurred mask edge) of one plant."""
    h, w = alpha.shape
    win = (slice(y, y + h), slice(x, x + w))
    hard, soft = canvas.copy(), canvas.copy()
    hard[win] = np.where(alpha[..., None], patch, hard[win])
    a = np.clip(ndimage.gaussian_filter(alpha.astype(np.float32), sigma), 0, 1)[..., None]
    soft[win] = (a * patch + (1 - a) * soft[win]).astype(np.uint8)
    return hard, soft


def show_outlined(ax, image, outline, title):
    ax.imshow(image)
    ax.contour(outline, levels=[0.5], colors="yellow", linewidths=1)
    ax.set_title(title, fontsize=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stats-csv", default="../data/Neophytes/stats_imagewise.csv")
    parser.add_argument("--classes-yaml", default="configs/data/neophytes_names_colors.yaml")
    parser.add_argument("--out-dir", default="results/paste_preview")
    parser.add_argument("--fold", default="cv3", help="only training sites of this fold are used")
    parser.add_argument("--n-examples", type=int, default=30)
    parser.add_argument("--crop", type=int, default=1024, help="canvas size in px (training crop size)")
    parser.add_argument("--min-area-m2", type=float, default=0.002, help="smallest usable plant region")
    parser.add_argument("--max-size-frac", type=float, default=0.6, help="largest plant side as fraction of the crop")
    parser.add_argument("--min-bg-frac", type=float, default=0.8, help="share of the plant that must land on background")
    parser.add_argument("--sigma", type=float, default=2.0, help="Gaussian blur of the mask edge for the soft paste [px]")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # %% data
    classes, folds = load_config(args.classes_yaml)
    tiles = training_tiles(args.stats_csv, folds, args.fold, classes)
    print(f"{args.fold}: {len(tiles)} training tiles from {tiles['site_id'].nunique()} sites "
          f"(held out: {', '.join(folds[args.fold])})")

    # %% examples
    rows = []
    for ex in range(args.n_examples):
        kind = "O2" if ex % 2 == 0 else "O1"
        cls = int(rng.integers(1, 7))
        name, stats_name = classes[cls]["name"], classes[cls]["stats_name"]
        c_row, canvas, c_mask, c_gsd = pick_canvas(tiles, kind, args.crop, rng)
        c_site = c_row["site_id"]

        inst = pick_instance(tiles, cls, stats_name, c_site, c_gsd, args.crop, args, rng)
        pos = None if inst is None else place(c_mask, inst[2], args.min_bg_frac, rng)
        if pos is None:
            print(f"example {ex}: no {name} could be pasted onto {c_site}, skipped")
            continue
        s_row, patch, alpha, (src_img, src_outline), scale = inst
        hard, soft = composite(canvas, patch, alpha, *pos, args.sigma)
        ref = pick_reference(tiles, cls, stats_name, c_site, args, rng)

        # %% panel: real reference | source | background | hard | soft
        fig, axes = plt.subplots(1, 5, figsize=(25, 5.6))
        if ref is not None:
            show_outlined(axes[0], *ref[1], f"real {name} @ {c_site}")
        else:
            axes[0].text(0.5, 0.5, f"no real {name}\nat {c_site}\nin the training data",
                         ha="center", va="center", fontsize=11, transform=axes[0].transAxes)
            axes[0].set_title("real reference", fontsize=10)
        show_outlined(axes[1], src_img, src_outline, f"source: {name} @ {s_row['site_id']}")
        axes[2].imshow(canvas)
        axes[2].set_title(f"background: {c_site} ({kind})", fontsize=10)
        for ax, img, title in ((axes[3], hard, "hard paste"), (axes[4], soft, f"soft paste (σ={args.sigma:g} px)")):
            ax.imshow(img)
            ax.add_patch(plt.Rectangle((pos[1], pos[0]), alpha.shape[1], alpha.shape[0],
                                       fill=False, edgecolor="red", linewidth=1, linestyle="--"))
            ax.set_title(title, fontsize=10)
        for ax in axes:
            ax.set_axis_off()
        fig.suptitle(f"#{ex:02d} {kind} · {name}: {s_row['site_id']} → {c_site} · "
                     f"scaled ×{scale:.2f} to {c_gsd:.1f} mm/px",
                     x=0.01, ha="left", fontsize=11)
        fig.savefig(os.path.join(args.out_dir, f"paste_{ex:02d}_{kind}.png"), dpi=110, bbox_inches="tight")
        plt.close(fig)

        rows.append({"example": ex, "canvas_type": kind, "class": name,
                     "source_site": s_row["site_id"], "source_tile": s_row["img_path"],
                     "canvas_site": c_site, "canvas_tile": c_row["img_path"], "canvas_gsd_mm": round(c_gsd, 2),
                     "scale": round(scale, 3), "y": pos[0], "x": pos[1], "h": alpha.shape[0], "w": alpha.shape[1],
                     "reference_tile": ref[0]["img_path"] if ref is not None else ""})
        print(f"example {ex}: {kind}, {name} {s_row['site_id']} -> {c_site}, "
              f"real reference {'yes' if ref is not None else 'none'}")

    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "index.csv"), index=False)
    print(f"wrote {len(rows)} panels and index.csv to {args.out_dir}")


if __name__ == "__main__":
    main()
