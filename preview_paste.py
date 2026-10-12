#!/usr/bin/env python3
"""Preview copy-paste synthesis: real plants pasted onto crops of other sites.

A look-only tool to judge whether cut-and-paste composites look realistic before
anything goes into training. Nothing here is used by train.py.

Per example, a 1024 crop of a training tile is the canvas, and 1-3 plants are
pasted into it. Each plant is a connected region of one class in the mask of a
training tile from a *different* site (instances from semantic masks, as in
Motoi et al. 2024), scaled to the canvas GSD, rotated/flipped, and placed mostly
on background. Two canvas types are shown, half each:

    O2  a normal training crop with its own labelled plants (Ghiasi et al. 2021)
    O1  a crop of a plant-free tile (background bank)

Every composite is rendered twice, hard paste and soft paste (Gaussian-blurred
mask edge); the label is identical for both. Only training sites of the chosen
fold are used, never the held-out ones.

    python preview_paste.py                          # cv3, 30 examples
    python preview_paste.py --fold cv1 --n-examples 10 --seed 1

Output (``--out-dir``, default results/paste_preview): one PNG panel per example
and ``index.csv`` with one row per pasted plant. The panels show confidential
data: view them on the node, keep them inside the course.
"""

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import yaml
from matplotlib.colors import ListedColormap
from scipy import ndimage

IGNORE = -1


def load_config(path):
    """Class names, colours, stats column names and CV folds from the classes yaml."""
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


def pick_instance(tiles, cls, stats_name, exclude_site, canvas_gsd, crop, args, rng, max_tries=15):
    """A connected region of class `cls` from a training tile of another site, already scaled
    to the canvas GSD. Returns (row, patch RGB, binary mask, (context crop, context mask), scale),
    or None if no usable region was found."""
    pool = tiles[(tiles[f"{stats_name}_px"] > 0) & (tiles["site_id"] != exclude_site)]
    if pool.empty:
        return None
    for _ in range(max_tries):
        row = pool.iloc[rng.integers(len(pool))]
        image, mask, gsd = read_tile(row["img_path"], row["mask_path"])
        scale = gsd / canvas_gsd * rng.uniform(0.9, 1.1)
        comp, n = ndimage.label(mask == cls)
        if n == 0:
            continue
        areas = np.bincount(comp.ravel())
        candidates = []
        for i, sl in enumerate(ndimage.find_objects(comp), start=1):
            hh, ww = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
            touches = sl[0].start == 0 or sl[1].start == 0 or sl[0].stop == mask.shape[0] or sl[1].stop == mask.shape[1]
            area_m2 = areas[i] * (gsd / 1000.0) ** 2
            ignore_frac = (mask[sl] == IGNORE).mean()
            if (not touches and area_m2 >= args.min_area_m2 and ignore_frac < 0.2
                    and max(hh, ww) * scale <= args.max_size_frac * crop):
                candidates.append((i, sl))
        if not candidates:
            continue
        i, sl = candidates[rng.integers(len(candidates))]
        patch = np.ascontiguousarray(image[sl])
        alpha = (comp[sl] == i).astype(np.float32)
        new_hw = (max(1, round(alpha.shape[1] * scale)), max(1, round(alpha.shape[0] * scale)))
        patch = cv2.resize(patch, new_hw, interpolation=cv2.INTER_CUBIC)
        alpha = cv2.resize(alpha, new_hw, interpolation=cv2.INTER_LINEAR) > 0.5
        # # random 90° rotation + flip, as in the training augmentation
        k = rng.integers(4)
        patch, alpha = np.rot90(patch, k), np.rot90(alpha, k)
        if rng.random() < 0.5:
            patch, alpha = patch[:, ::-1], alpha[:, ::-1]
        # # context around the instance in its source tile, for the panel
        pad = max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) // 2 + 32
        cy0, cx0 = max(sl[0].start - pad, 0), max(sl[1].start - pad, 0)
        cy1, cx1 = min(sl[0].stop + pad, mask.shape[0]), min(sl[1].stop + pad, mask.shape[1])
        context = (image[cy0:cy1, cx0:cx1], comp[cy0:cy1, cx0:cx1] == i)
        return row, np.ascontiguousarray(patch), np.ascontiguousarray(alpha), context, scale
    return None


def place(canvas_mask, alpha, min_bg_frac, rng, max_tries=50):
    """Top-left corner where at least min_bg_frac of the pasted plant lands on background."""
    H, W = canvas_mask.shape
    h, w = alpha.shape
    for _ in range(max_tries):
        y, x = rng.integers(0, H - h + 1), rng.integers(0, W - w + 1)
        under = canvas_mask[y:y + h, x:x + w][alpha]
        if (under == 0).mean() >= min_bg_frac:
            return y, x
    return None


def paste(hard, soft, label, patch, alpha, y, x, cls, sigma):
    """Paste in place: hard = binary mask, soft = Gaussian-blurred mask edge. Same label for both."""
    h, w = alpha.shape
    win = (slice(y, y + h), slice(x, x + w))
    a = alpha[..., None]
    hard[win] = np.where(a, patch, hard[win])
    a_soft = np.clip(ndimage.gaussian_filter(alpha.astype(np.float32), sigma), 0, 1)[..., None]
    soft[win] = (a_soft * patch + (1 - a_soft) * soft[win]).astype(np.uint8)
    label[win][alpha] = cls


def label_overlay(ax, image, label, cmap):
    ax.imshow(image)
    shown = np.ma.masked_where(label == 0, label)
    ax.imshow(shown, cmap=cmap, vmin=-1, vmax=6, alpha=0.5, interpolation="nearest")


def zoom_box(y, x, h, w, size):
    """Square window around a pasted plant, padded so its edge is visible."""
    side = int(min(max(h, w) * 1.6 + 64, size))
    cy, cx = y + h // 2, x + w // 2
    y0 = int(np.clip(cy - side // 2, 0, size - side))
    x0 = int(np.clip(cx - side // 2, 0, size - side))
    return slice(y0, y0 + side), slice(x0, x0 + side)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stats-csv", default="../data/Neophytes/stats_imagewise.csv")
    parser.add_argument("--classes-yaml", default="configs/data/neophytes_names_colors.yaml")
    parser.add_argument("--out-dir", default="results/paste_preview")
    parser.add_argument("--fold", default="cv3", help="only training sites of this fold are used")
    parser.add_argument("--n-examples", type=int, default=30)
    parser.add_argument("--max-pastes", type=int, default=3, help="1..max plants per canvas")
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
    colors = ["#808080", "#000000"] + [classes[c]["color"] for c in range(1, 7)]
    cmap = ListedColormap(colors)  # # -1 ignore (grey), 0 background, 1..6 species
    print(f"{args.fold}: {len(tiles)} training tiles from {tiles['site_id'].nunique()} sites "
          f"(held out: {', '.join(folds[args.fold])})")

    # %% examples
    rows = []
    for ex in range(args.n_examples):
        kind = "O2" if ex % 2 == 0 else "O1"
        c_row, canvas, c_mask, c_gsd = pick_canvas(tiles, kind, args.crop, rng)
        hard, soft, label = canvas.copy(), canvas.copy(), c_mask.copy()
        pasted = []
        for _ in range(rng.integers(1, args.max_pastes + 1)):
            cls = int(rng.integers(1, 7))
            inst = pick_instance(tiles, cls, classes[cls]["stats_name"], c_row["site_id"], c_gsd, args.crop, args, rng)
            if inst is None:
                continue
            s_row, patch, alpha, context, scale = inst
            pos = place(label, alpha, args.min_bg_frac, rng)
            if pos is None:
                continue
            paste(hard, soft, label, patch, alpha, *pos, cls, args.sigma)
            pasted.append((cls, s_row, context, scale, pos, alpha.shape))
            rows.append({"example": ex, "canvas_type": kind, "canvas_tile": c_row["img_path"],
                         "canvas_site": c_row["site_id"], "canvas_gsd_mm": round(c_gsd, 2),
                         "class": classes[cls]["name"], "source_tile": s_row["img_path"],
                         "source_site": s_row["site_id"], "scale": round(scale, 3),
                         "y": pos[0], "x": pos[1], "h": alpha.shape[0], "w": alpha.shape[1],
                         "area_px": int(alpha.sum())})
        if not pasted:
            print(f"example {ex}: nothing could be pasted, skipped")
            continue

        # %% panel: source | hard | soft | label | zoom hard + soft
        cls, s_row, (ctx_img, ctx_mask), scale, (y, x), (h, w) = pasted[0]
        zy, zx = zoom_box(y, x, h, w, args.crop)
        fig, axes = plt.subplots(1, 5, figsize=(25, 5.6))
        axes[0].imshow(ctx_img)
        axes[0].contour(ctx_mask, levels=[0.5], colors="yellow", linewidths=1)
        axes[0].set_title(f"source: {classes[cls]['name']} @ {s_row['site_id']}", fontsize=10)
        axes[1].imshow(hard)
        axes[1].set_title("hard paste", fontsize=10)
        axes[2].imshow(soft)
        axes[2].set_title(f"soft paste (σ={args.sigma:g} px)", fontsize=10)
        label_overlay(axes[3], soft, label, cmap)
        axes[3].set_title("label (pasted + canvas's own)", fontsize=10)
        axes[4].imshow(np.concatenate([hard[zy, zx], np.full((hard[zy, zx].shape[0], 8, 3), 255, np.uint8), soft[zy, zx]], 1))
        axes[4].set_title("zoom on 1st paste: hard | soft", fontsize=10)
        for ax in axes[1:4]:
            for (_, _, _, _, (py, px), (ph, pw)) in pasted:
                ax.add_patch(plt.Rectangle((px, py), pw, ph, fill=False, edgecolor="red", linewidth=0.8, linestyle="--"))
        for ax in axes:
            ax.set_axis_off()
        names = ", ".join(f"{classes[p[0]]['name']} ({p[1]['site_id']}, ×{p[3]:.2f})" for p in pasted)
        fig.suptitle(f"#{ex:02d} {kind} · canvas {c_row['site_id']} ({c_gsd:.1f} mm/px) ← {names}",
                     x=0.01, ha="left", fontsize=11)
        fig.savefig(os.path.join(args.out_dir, f"paste_{ex:02d}_{kind}.png"), dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"example {ex}: {kind}, {len(pasted)} pasted")

    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "index.csv"), index=False)
    print(f"wrote {args.out_dir}/paste_*.png and index.csv")


if __name__ == "__main__":
    main()
