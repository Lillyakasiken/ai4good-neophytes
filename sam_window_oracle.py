#!/usr/bin/env python3
"""Window oracle: can SAM segment our plants if it only sees one grid window at a time?

Ceiling of a "detect windows -> SAM -> classify cut-outs" pipeline, with both
classifiers assumed perfect, compared against the U-Net baseline on the same tiles.

Per tile and window size (exclusive grid, no overlap):
  1. Perfect window classifier: a window is positive if >= --pos-thr of its pixels
     carry a species label.
  2. SAM automatic masks on each positive window (SAM resizes the window to 1024).
  3. Inside each window the masks become non-overlapping regions: every pixel goes
     to the smallest mask covering it (as in sam_oracle.py).
  4. Perfect cut-out classifier: a region whose valid pixels are > --majority one
     species gets that species, otherwise background. Pixels without a region
     (negative windows, gaps) are background.
  5. The windows are put back into the full tile, which is scored and drawn.

Regions are not stitched across window borders: with a perfect majority-vote
classifier, joining two regions can only keep or lower the score, so the
unstitched map is the ceiling of SAM's masks.

Tiles: --n-tiles labelled tiles of the cv3 test split (the /train folders of the
held-out sites, as in test.py), drawn per species round robin over flights.
Background-only tiles are left out: with perfect classifiers the oracle cannot
make mistakes there, while the baseline can, so leaving them out only favours
the baseline.

Both the oracle and the baseline are scored with utils.eval_utils.metrics_from_confusion,
the function test.py uses, on the same pixels (254/255 = ignore).

    python sam_window_oracle.py --sam-ckpt <sam2.1_hiera_large.pt> --baseline-ckpt <run>/checkpoints/best_*.ckpt
    python sam_window_oracle.py ... --n-tiles 4 --n-vis 4            # quick timing check

Outputs in --out-dir: summary.md, confusion.json, tiles.csv and vis/*.png. They are
derived from the confidential dataset: keep them on the node.
Needs an env with sam2 + the project stack (smp, lightning), e.g. the personal sam2 env.
"""

import argparse
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib.colors import to_rgb
from scipy import ndimage

from sam_oracle import IGNORE, N_CLS, load_config, read_tile


def cv_test_tiles(stats_csv, split_yaml, classes, n_tiles, min_class_m2, rng):
    """Labelled tiles of the split's test folders, per species round robin over flights."""
    with open(split_yaml) as f:
        test_dirs = [d.rstrip("/") + "/" for d in yaml.safe_load(f)["image_dir"]["test"]]
    df = pd.read_csv(stats_csv)
    df = df[df["img_path"].apply(lambda p: any(p.startswith(d) for d in test_dirs))].copy()
    has = {c: df[f"{classes[c]['stats_name']}_m2"] >= min_class_m2 for c in range(1, N_CLS)}
    chosen, drawn_for = [], {}
    # # rarest species first; a species with too few tiles leaves its quota to the others
    for i, c in enumerate(sorted(has, key=lambda c: has[c].sum())):
        per_class = int(np.ceil((n_tiles - len(chosen)) / (N_CLS - 1 - i)))
        pool = df[has[c] & ~df.index.isin(chosen)]
        by_flight = [g.sample(frac=1, random_state=int(rng.integers(1 << 31))).index.tolist()
                     for _, g in pool.groupby("site")]
        rng.shuffle(by_flight)
        picked = []
        while len(picked) < per_class and any(by_flight):
            for lst in by_flight:
                if lst and len(picked) < per_class:
                    picked.append(lst.pop(0))
        chosen += picked
        drawn_for.update({i: c for i in picked})
    out = df.loc[chosen].copy()
    out["drawn_for"] = [drawn_for[i] for i in out.index]
    return out.reset_index(drop=True)


def window_regions(segs, size):
    """Smallest-mask-wins partition of one window: region id per pixel (-1 = no mask)."""
    reg = np.full((size, size), -1, np.int32)
    for j in sorted(range(len(segs)), key=lambda j: -segs[j].sum()):
        reg[segs[j]] = j
    # # renumber to 0..n-1 (masks fully covered by smaller ones disappear)
    ids = np.unique(reg[reg >= 0])
    lut = np.full(len(segs) + 1, -1, np.int32)
    lut[ids] = np.arange(len(ids))
    return lut[reg]


def oracle_labels(seg, gt, valid, majority):
    """Perfect cut-out classifier: segment -> species if > majority of its valid pixels are it."""
    pred = np.zeros(gt.shape, np.int64)
    n = int(seg.max()) + 1
    if n == 0:
        return pred
    m = valid & (seg >= 0)
    votes = np.bincount(seg[m] * N_CLS + gt[m], minlength=n * N_CLS).reshape(n, N_CLS)
    share = votes / np.maximum(votes.sum(1, keepdims=True), 1)
    lut = np.where(share.max(1) > majority, share.argmax(1), 0)
    pred[seg >= 0] = lut[seg[seg >= 0]]
    return pred


def confusion(gt, pred, valid):
    return np.bincount(gt[valid] * N_CLS + pred[valid], minlength=N_CLS * N_CLS).reshape(N_CLS, N_CLS)


def baseline_predict(model, image, mean, std):
    """U-Net prediction on the whole tile, as in utils.eval_utils.evaluate_model."""
    x = (image.astype(np.float32) / 255.0 - mean) / std
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).cuda()
    with torch.no_grad():
        return torch.argmax(model(x), dim=1)[0].cpu().numpy().astype(np.int64)


def save_vis(path, image, gt, base_pred, per_size, colors, title):
    """Row 0: RGB + labels | labels | baseline. One row per window size: windows | segments | oracle."""
    s = slice(None, None, 2)
    base = image[s, s] / 255.0

    def species_rgb(lab):
        out = np.zeros((*lab.shape, 3))
        for c in range(1, N_CLS):
            out[lab == c] = to_rgb(colors[c])
        return out

    def with_outlines(img):
        out = img.copy()
        for c in range(1, N_CLS):
            m = gt[s, s] == c
            out[m & ~ndimage.binary_erosion(m, iterations=2)] = to_rgb(colors[c])
        return out

    def blend(lab):
        rgb = species_rgb(lab[s, s])
        return np.where(rgb.any(-1, keepdims=True), 0.4 * base + 0.6 * rgb, base)

    rows = [[(with_outlines(base), "RGB + label outlines"), (blend(gt), "labels"),
             (with_outlines(blend(base_pred)), "baseline U-Net (+ label outlines)")]]
    for size, d in per_size.items():
        win = base.copy()
        for (y0, x0) in d["pos"]:
            ys, xs = slice(y0 // 2, (y0 + size) // 2), slice(x0 // 2, (x0 + size) // 2)
            win[ys, xs] = 0.6 * win[ys, xs] + 0.4 * np.array([1.0, 1.0, 0.0])
        win[:, ::size // 2] = win[::size // 2, :] = 1.0
        rng = np.random.default_rng(0)
        seg = d["seg"][s, s]
        pal = rng.uniform(0.2, 1, (max(int(seg.max()) + 1, 1), 3))
        seg_rgb = np.where((seg >= 0)[..., None], 0.35 * base + 0.65 * pal[np.maximum(seg, 0)], base)
        rows.append([(win, f"{size} px windows: {len(d['pos'])} positive"),
                     (seg_rgb, f"SAM regions per window ({d['n_seg']})"),
                     (with_outlines(blend(d["pred"])), "oracle (+ label outlines)")])
    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 5.1 * len(rows)))
    for r, row in enumerate(rows):
        for c, (img, t) in enumerate(row):
            axes[r, c].imshow(img)
            axes[r, c].set_title(t, fontsize=10)
            axes[r, c].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def score_table(confs, names, metrics_from_confusion):
    """One row per setting: F1 per species + F1-avg-wo0, as test.py computes it."""
    rows = {}
    for key, conf in confs.items():
        m = metrics_from_confusion(np.array(conf))
        rows[key] = {**{names[c]: m["F1"][c] for c in range(1, N_CLS)}, "F1-avg-wo0": m["F1-avg-wo0"],
                     "Precision-avg-wo0": m["Precision-avg-wo0"], "Recall-avg-wo0": m["Recall-avg-wo0"]}
    return pd.DataFrame(rows).T


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stats-csv", default="../data/Neophytes/stats_imagewise.csv")
    parser.add_argument("--split-yaml", default="configs/data/split_cv3.yaml")
    parser.add_argument("--classes-yaml", default="configs/data/neophytes_names_colors.yaml")
    parser.add_argument("--sam-ckpt", default="../weights/sam2.1_hiera_large.pt")
    parser.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--baseline-ckpt", required=True, help="U-Net checkpoint of a cv3 run")
    parser.add_argument("--norm-mean", type=float, nargs=3, default=[0.485, 0.456, 0.406],
                        help="as data.norm_mean in configs/data/transforms_*.yaml")
    parser.add_argument("--norm-std", type=float, nargs=3, default=[0.229, 0.224, 0.225])
    parser.add_argument("--out-dir", default="results/sam_window_oracle")
    parser.add_argument("--n-tiles", type=int, default=100)
    parser.add_argument("--min-class-m2", type=float, default=0.005, help="min species area in a drawn tile")
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--pos-thr", type=float, nargs="+", default=[0.001, 0.01, 0.05],
                        help="min labelled share of a positive window")
    parser.add_argument("--vis-thr", type=float, default=0.01, help="threshold shown in the figures")
    parser.add_argument("--majority", type=float, default=0.5)
    parser.add_argument("--pps", type=int, default=32, help="AMG points per side")
    parser.add_argument("--iou-thr", type=float, default=0.7, help="AMG pred_iou_thresh")
    parser.add_argument("--stab-thr", type=float, default=0.9, help="AMG stability_score_thresh")
    parser.add_argument("--n-vis", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    from models.semseg_plm import SegmentationModel
    from utils.eval_utils import metrics_from_confusion

    os.makedirs(os.path.join(args.out_dir, "vis"), exist_ok=True)
    rng = np.random.default_rng(args.seed)
    classes = load_config(args.classes_yaml)
    names = [classes[c]["name"] for c in range(N_CLS)]
    colors = {c: classes[c]["color"] for c in range(N_CLS)}
    thr_min = min(args.pos_thr)

    # %% models
    sam = build_sam2(args.model_cfg, args.sam_ckpt, device="cuda", apply_postprocessing=False)
    amg = SAM2AutomaticMaskGenerator(sam, points_per_side=args.pps, points_per_batch=64,
                                     pred_iou_thresh=args.iou_thr, stability_score_thresh=args.stab_thr,
                                     crop_n_layers=0, output_mode="binary_mask")
    unet = SegmentationModel.load_from_checkpoint(args.baseline_ckpt, weights_only=False, map_location="cuda")
    unet.set_patch_2_img_size(False)
    unet.cuda().eval()
    mean, std = np.array(args.norm_mean, np.float32), np.array(args.norm_std, np.float32)

    # %% tiles
    tiles = cv_test_tiles(args.stats_csv, args.split_yaml, classes, args.n_tiles, args.min_class_m2, rng)
    print(f"{len(tiles)} tiles from {tiles.site.nunique()} flights")

    confs = {"baseline": np.zeros((N_CLS, N_CLS), np.int64)}
    confs.update({f"oracle_{s}_thr{t}": np.zeros((N_CLS, N_CLS), np.int64)
                  for s in args.sizes for t in args.pos_thr})
    tile_rows, sam_time, n_windows = [], 0.0, 0
    for t, row in tiles.iterrows():
        image, mask, gsd = read_tile(row["img_path"], row["mask_path"])
        valid = mask != IGNORE
        gt = np.where(valid, mask, 0)
        h, w = gt.shape
        tile_id = os.path.basename(row["img_path"])

        base_pred = baseline_predict(unet, image, mean, std)
        confs["baseline"] += confusion(gt, base_pred, valid)
        info = dict(tile=tile_id, flight=row["site"], drawn_for=names[row["drawn_for"]])

        per_size = {}
        for size in args.sizes:
            assert h % size == 0 and w % size == 0, f"tile {h}x{w} not divisible by window {size}"
            # # SAM once per window positive at the lowest threshold; higher thresholds only drop windows
            regions, share = {}, {}
            for y0 in range(0, h, size):
                for x0 in range(0, w, size):
                    lab = (gt[y0:y0 + size, x0:x0 + size] > 0).mean()
                    if lab < thr_min:
                        continue
                    t0 = time.time()
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        segs = [m["segmentation"] for m in amg.generate(image[y0:y0 + size, x0:x0 + size])]
                    sam_time += time.time() - t0
                    n_windows += 1
                    regions[(y0, x0)], share[(y0, x0)] = window_regions(segs, size), lab
            for thr in args.pos_thr:
                pos = [k for k, v in share.items() if v >= thr]
                # # windows back into the full tile; region ids made unique across windows
                seg = np.full((h, w), -1, np.int32)
                offset = 0
                for (y0, x0) in pos:
                    r = regions[(y0, x0)]
                    seg[y0:y0 + size, x0:x0 + size] = np.where(r >= 0, r + offset, -1)
                    offset += int(r.max()) + 1
                pred = oracle_labels(seg, gt, valid, args.majority)
                confs[f"oracle_{size}_thr{thr}"] += confusion(gt, pred, valid)
                info.update({f"pos_{size}_thr{thr}": len(pos), f"seg_{size}_thr{thr}": offset})
                if thr == args.vis_thr:
                    per_size[size] = dict(pos=pos, seg=seg, pred=pred, n_seg=offset)
        tile_rows.append(info)

        if t < args.n_vis and per_size:
            save_vis(os.path.join(args.out_dir, "vis", f"{t:03d}_{row['site']}_{names[row['drawn_for']]}.png"),
                     image, gt, base_pred, per_size, colors,
                     f"{tile_id}  ({gsd:.2f} mm/px; drawn for {names[row['drawn_for']]}; "
                     f"windows positive at >= {args.vis_thr:.1%} label)")
        print(f"[{t + 1}/{len(tiles)}] {tile_id}: SAM {sam_time / max(n_windows, 1):.1f} s/window, "
              f"{n_windows} windows so far", flush=True)

    # %% outputs
    pd.DataFrame(tile_rows).to_csv(os.path.join(args.out_dir, "tiles.csv"), index=False)
    with open(os.path.join(args.out_dir, "confusion.json"), "w") as f:
        json.dump({"classes": names, **{k: v.tolist() for k, v in confs.items()}}, f, indent=1)
    table = score_table(confs, names, metrics_from_confusion)
    label_px = confs["baseline"].sum(1)
    md = [f"# SAM window oracle vs. baseline (cv3 test, {len(tiles)} labelled tiles)\n",
          f"Baseline: `{args.baseline_ckpt}`. SAM 2.1 `{os.path.basename(args.sam_ckpt)}`, "
          f"points_per_side={args.pps}, pred_iou_thresh={args.iou_thr}, stability={args.stab_thr}. "
          f"No stitching across windows; majority > {args.majority}. "
          f"SAM {sam_time / max(n_windows, 1):.2f} s/window over {n_windows} windows.\n",
          "Oracle rows assume a perfect window classifier and a perfect cut-out classifier: "
          "they are ceilings, the baseline row is a real model. F1 as in test.py "
          "(`metrics_from_confusion`), only on these tiles.\n",
          "Labelled pixels per species: " + ", ".join(f"{names[c]} {label_px[c]:,}" for c in range(1, N_CLS)) + "\n",
          table.round(3).to_markdown()]
    with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
