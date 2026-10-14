#!/usr/bin/env python3
"""Oracle test: does SAM 2 cut out our labelled plants?

A segment-then-classify pipeline (class-agnostic SAM masks -> species classifier)
can only be as good as SAM's masks. This script measures that ceiling on labelled
tiles before anything is trained: the labels decide what the best possible use of
SAM's masks would be, so every number here is an upper bound.

Tiles: up to --per-class labelled tiles per species, spread over sites (round
robin), from the train + test folders of all sites (SAM is not trained on our
data, so held-out vs training sites does not matter here). Every labelled plant
in a selected tile is evaluated, not only the species the tile was drawn for.

Plants are connected regions of one class in the mask (>= --min-inst-m2).
Ignore pixels (254/255) are left out of every count.

Automatic mode (SAM2AutomaticMaskGenerator, a grid of point prompts), two settings:
  tile        the 2048 tile, resized by SAM to 1024 (~5 mm/px)
  tile+crops  plus 4 overlapping half-tile crops (~native 2.4 mm/px), merged by NMS

Per plant and setting:
  best_iou     IoU of the single best SAM mask (can one mask be the plant?)
  best_prec    share of that mask that lies on the plant (< 1: mask leaks out)
  best_cover   share of the plant covered by that mask (< 1: mask is a part)
  union_iou    IoU of the union of all masks lying >= 50% on the plant
               (can the plant be assembled from parts by a perfect classifier?)
  n_parts      number of those masks
  covered      share of the plant covered by any SAM mask

Oracle label map per setting: every pixel goes to the smallest SAM mask covering
it, every mask gets the majority label of its pixels, uncovered pixels become
background. Per-class pixel F1/IoU of that map = what a perfect mask classifier
on top of SAM would score.

Prompted mode (SAM2ImagePredictor on a native-resolution 1024 crop around the
plant), up to --prompt-per-class plants per class and tile:
  point_best   one click at the plant's most interior point, best of SAM's 3 masks
  point_top    same click, the mask SAM itself scores highest
  box_top      the plant's bounding box as prompt (detector + SAM upper bound)

    python sam_oracle.py --ckpt <sam2.1_hiera_large.pt>
    python sam_oracle.py --per-class 4 --n-vis 4            # quick check

Outputs in --out-dir: instances.csv, prompts.csv, confusion.json, summary.md and
vis/*.png. They are derived from the confidential dataset: keep them on the node.
SAM 2 is not in the shared env; run with a personal env that has `sam2`.
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
import rasterio
import torch
import yaml
from scipy import ndimage

IGNORE = -1
N_CLS = 7  # background + 6 species


def load_config(path):
    """Classes 0..6 (name, colour, stats name) from the classes yaml."""
    with open(path) as f:
        return yaml.safe_load(f)["classes"]


def labelled_tiles(stats_csv, classes, per_class, min_class_m2, max_nodata, rng):
    """Up to per_class tiles per species, round robin over sites; each tile once."""
    df = pd.read_csv(stats_csv)
    df = df[df["split"].isin(["train", "test"])].copy()
    df["site_id"] = df["site"].str.rsplit("_", n=1).str[0]
    df = df[df["nodata_px"] / df["total_px"] <= max_nodata]
    chosen, drawn_for = [], {}
    for c in range(1, N_CLS):
        pool = df[(df[f"{classes[c]['stats_name']}_m2"] >= min_class_m2) & ~df.index.isin(chosen)]
        by_site = [g.sample(frac=1, random_state=int(rng.integers(1 << 31))).index.tolist()
                   for _, g in pool.groupby("site_id")]
        rng.shuffle(by_site)
        picked = []
        while len(picked) < per_class and any(by_site):
            for lst in by_site:
                if lst and len(picked) < per_class:
                    picked.append(lst.pop(0))
        chosen += picked
        drawn_for.update({i: c for i in picked})
    out = df.loc[chosen].copy()
    out["drawn_for"] = [drawn_for[i] for i in chosen]
    return out.reset_index(drop=True)


def read_tile(img_path, mask_path):
    """RGB uint8 (H, W, 3), class mask int64 with 254/255 -> IGNORE, GSD in mm/px."""
    with rasterio.open(img_path) as src:
        image = np.ascontiguousarray(np.transpose(src.read([1, 2, 3]), (1, 2, 0)))
        gsd = abs(src.transform.a) * 1000.0
    with rasterio.open(mask_path) as src:
        mask = src.read(1).astype(np.int64)
    mask[(mask == 254) | (mask == 255)] = IGNORE
    return image, mask, gsd


def instances(mask, min_px):
    """Label image of all plants (0 = none) and per plant (class, area in px, bbox slices)."""
    inst = np.zeros(mask.shape, np.int32)
    info = []
    for c in range(1, N_CLS):
        comp, n = ndimage.label(mask == c)
        if n == 0:
            continue
        areas = np.bincount(comp.ravel())
        for i, sl in enumerate(ndimage.find_objects(comp), start=1):
            if areas[i] >= min_px:
                inst[sl][comp[sl] == i] = len(info) + 1
                info.append((c, int(areas[i]), sl))
    return inst, info


def amg_metrics(segs, inst, info, valid):
    """Per-plant metrics (see module docstring) for one list of SAM masks."""
    k = len(info)
    inst_v = np.where(valid, inst, 0).ravel()
    area = np.bincount(inst_v, minlength=k + 1).astype(float)
    inter = np.zeros((len(segs), k + 1))
    m_area = np.zeros(len(segs))
    covered = np.zeros(inst_v.shape, bool)
    for j, s in enumerate(segs):
        s = s.ravel() & valid.ravel()
        inter[j] = np.bincount(inst_v[s], minlength=k + 1)
        m_area[j] = s.sum()
        covered |= s
    cov = np.bincount(inst_v[covered], minlength=k + 1)
    rows = []
    for p in range(1, k + 1):
        if len(segs) == 0:
            rows.append(dict(best_iou=0, best_prec=0, best_cover=0, union_iou=0, n_parts=0, covered=0))
            continue
        iou = inter[:, p] / np.maximum(m_area + area[p] - inter[:, p], 1)
        b = int(iou.argmax())
        parts = np.where(inter[:, p] >= 0.5 * np.maximum(m_area, 1))[0]
        if len(parts):
            u = np.zeros(inst_v.shape, bool)
            for j in parts:
                u |= segs[j].ravel()
            u &= valid.ravel()
            ui = (u & (inst_v == p)).sum()
            union_iou = ui / (u.sum() + area[p] - ui)
        else:
            union_iou = 0.0
        rows.append(dict(best_iou=iou[b], best_prec=inter[b, p] / max(m_area[b], 1),
                         best_cover=inter[b, p] / max(area[p], 1), union_iou=union_iou,
                         n_parts=len(parts), covered=cov[p] / max(area[p], 1)))
    return rows


def oracle_map(segs, gt, valid):
    """Smallest-mask-wins assignment, majority label per mask; returns predicted label map."""
    assign = np.full(gt.shape, -1, np.int32)
    order = sorted(range(len(segs)), key=lambda j: -segs[j].sum())
    for j in order:
        assign[segs[j]] = j
    pred = np.zeros(gt.shape, np.int64)
    if len(segs):
        a = assign[valid & (assign >= 0)]
        g = gt[valid & (assign >= 0)]
        votes = np.bincount(a * N_CLS + g, minlength=len(segs) * N_CLS).reshape(len(segs), N_CLS)
        lut = votes.argmax(1)
        pred[assign >= 0] = lut[assign[assign >= 0]]
    return pred


def prompt_metrics(predictor, image, inst, info, per_class, crop, rng):
    """Point and box prompts on a native crop around up to per_class plants per class."""
    rows = []
    h, w = inst.shape
    for c in range(1, N_CLS):
        ids = [p for p, (cc, _, _) in enumerate(info, start=1) if cc == c]
        for p in rng.permutation(ids)[:per_class]:
            c_, _, sl = info[p - 1]
            cy, cx = (sl[0].start + sl[0].stop) // 2, (sl[1].start + sl[1].stop) // 2
            y0 = int(np.clip(cy - crop // 2, 0, max(h - crop, 0)))
            x0 = int(np.clip(cx - crop // 2, 0, max(w - crop, 0)))
            gt = inst[y0:y0 + crop, x0:x0 + crop] == p
            dist = ndimage.distance_transform_edt(gt)
            py, px = np.unravel_index(dist.argmax(), dist.shape)
            ys, xs = np.where(gt)
            box = np.array([xs.min(), ys.min(), xs.max(), ys.max()])
            predictor.set_image(image[y0:y0 + crop, x0:x0 + crop])
            pm, ps, _ = predictor.predict(point_coords=np.array([[px, py]]), point_labels=np.array([1]),
                                          multimask_output=True)
            bm, bs, _ = predictor.predict(box=box, multimask_output=False)
            ious = [(m.astype(bool) & gt).sum() / max((m.astype(bool) | gt).sum(), 1) for m in pm]
            b_iou = (bm[0].astype(bool) & gt).sum() / max((bm[0].astype(bool) | gt).sum(), 1)
            rows.append(dict(inst=p, cls=c, area_px=info[p - 1][1], clipped=bool(gt.sum() < info[p - 1][1]),
                             point_best=max(ious), point_top=ious[int(np.argmax(ps))], box_top=b_iou))
    return rows


def save_vis(path, image, mask, segs, pred, colors, title):
    """RGB + label outlines | SAM masks | oracle label map, downscaled 2x."""
    s = slice(None, None, 2)
    rng = np.random.default_rng(0)
    sam_rgb = np.zeros((*mask.shape, 3), float)
    for seg in sorted(segs, key=lambda m: -m.sum()):
        sam_rgb[seg] = rng.uniform(0.2, 1, 3)
    lab = np.zeros((*mask.shape, 3), float)
    pred_rgb = np.zeros((*mask.shape, 3), float)
    for c in range(1, N_CLS):
        rgb = np.array(matplotlib.colors.to_rgb(colors[c]))
        edge = (mask == c) & ~ndimage.binary_erosion(mask == c, iterations=3)
        lab[edge] = rgb
        pred_rgb[pred == c] = rgb
    base = image[s, s] / 255.0
    outl = np.where(lab[s, s].any(-1, keepdims=True), lab[s, s], base)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.4))
    for ax, im, t in zip(axes, [outl, 0.45 * base + 0.55 * sam_rgb[s, s], 0.5 * base + 0.5 * pred_rgb[s, s]],
                         ["RGB + label outlines", f"SAM masks ({len(segs)})", "oracle: SAM masks, majority label"]):
        ax.imshow(im)
        ax.set_title(t, fontsize=10)
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def f1_table(conf, names):
    """Per-class pixel F1 and IoU from a confusion matrix (rows = label, cols = prediction)."""
    tp = np.diag(conf).astype(float)
    fp, fn = conf.sum(0) - tp, conf.sum(1) - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    iou = tp / np.maximum(tp + fp + fn, 1)
    return pd.DataFrame({"F1": f1, "IoU": iou, "label_px": conf.sum(1)}, index=names)


def summarise(inst_df, prm_df, confs, names, args, n_tiles, runtimes):
    """Markdown summary: per class and setting."""
    out = [f"# SAM 2 oracle on labelled plants\n",
           f"{n_tiles} tiles, SAM 2.1 {os.path.basename(args.ckpt)}, points_per_side={args.pps}, "
           f"pred_iou_thresh={args.iou_thr}, stability={args.stab_thr}, plants >= {args.min_inst_m2} m².\n"]
    for setting, g in inst_df.groupby("setting", sort=False):
        out.append(f"\n## Automatic masks, setting `{setting}` (mean {runtimes[setting]:.1f} s/tile)\n")
        t = g.groupby("cls_name").apply(lambda d: pd.Series({
            "plants": len(d),
            "med best IoU": d.best_iou.median(),
            "best IoU>=0.5 %": 100 * (d.best_iou >= 0.5).mean(),
            "best IoU>=0.75 %": 100 * (d.best_iou >= 0.75).mean(),
            "area-wt best IoU": np.average(d.best_iou, weights=d.area_px),
            "med prec": d.best_prec.median(),
            "med cover": d.best_cover.median(),
            "med union IoU": d.union_iou.median(),
            "union IoU>=0.5 %": 100 * (d.union_iou >= 0.5).mean(),
            "med parts": d.n_parts.median(),
            "med covered": d.covered.median(),
        }))
        out.append(t.round(2).to_markdown())
        ft = f1_table(np.array(confs[setting]), names)
        out.append(f"\n\nOracle label map (perfect classifier on SAM masks), pixel level:\n")
        out.append(ft.round(3).to_markdown())
        out.append(f"\n\nmean F1 without background: **{ft.F1.iloc[1:][ft.label_px.iloc[1:] > 0].mean():.3f}**\n")
    if len(prm_df):
        out.append("\n## Prompted (native 1024 crop around the plant)\n")
        t = prm_df.groupby("cls_name").agg(plants=("inst", "size"), point_best=("point_best", "median"),
                                           point_top=("point_top", "median"), box_top=("box_top", "median"),
                                           box_ge_075=("box_top", lambda x: 100 * (x >= 0.75).mean()))
        out.append("Median IoU; box_ge_075 = % plants with box IoU >= 0.75\n\n" + t.round(2).to_markdown())
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stats-csv", default="../data/Neophytes/stats_imagewise.csv")
    parser.add_argument("--classes-yaml", default="configs/data/neophytes_names_colors.yaml")
    parser.add_argument("--ckpt", default="../weights/sam2.1_hiera_large.pt")
    parser.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--out-dir", default="results/sam_oracle")
    parser.add_argument("--per-class", type=int, default=10, help="tiles drawn per species")
    parser.add_argument("--min-class-m2", type=float, default=0.05, help="min area of the species in a drawn tile")
    parser.add_argument("--min-inst-m2", type=float, default=0.0025, help="min plant area (5 x 5 cm)")
    parser.add_argument("--max-nodata", type=float, default=0.3)
    parser.add_argument("--pps", type=int, default=32, help="AMG points per side")
    parser.add_argument("--iou-thr", type=float, default=0.7, help="AMG pred_iou_thresh (SAM default 0.8)")
    parser.add_argument("--stab-thr", type=float, default=0.9, help="AMG stability_score_thresh (SAM default 0.95)")
    parser.add_argument("--prompt-per-class", type=int, default=5)
    parser.add_argument("--crop", type=int, default=1024, help="crop for prompted mode")
    parser.add_argument("--n-vis", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    os.makedirs(os.path.join(args.out_dir, "vis"), exist_ok=True)
    rng = np.random.default_rng(args.seed)
    classes = load_config(args.classes_yaml)
    names = [classes[c]["name"] for c in range(N_CLS)]
    colors = {c: classes[c]["color"] for c in range(N_CLS)}

    # %% model
    model = build_sam2(args.model_cfg, args.ckpt, device="cuda", apply_postprocessing=False)
    amg_kw = dict(points_per_side=args.pps, points_per_batch=64, pred_iou_thresh=args.iou_thr,
                  stability_score_thresh=args.stab_thr, output_mode="binary_mask")
    generators = {"tile": SAM2AutomaticMaskGenerator(model, crop_n_layers=0, **amg_kw),
                  "tile+crops": SAM2AutomaticMaskGenerator(model, crop_n_layers=1, **amg_kw)}
    predictor = SAM2ImagePredictor(model)

    # %% tiles
    tiles = labelled_tiles(args.stats_csv, classes, args.per_class, args.min_class_m2, args.max_nodata, rng)
    print(f"{len(tiles)} tiles from {tiles.site_id.nunique()} sites")

    inst_rows, prm_rows = [], []
    confs = {s: np.zeros((N_CLS, N_CLS), np.int64) for s in generators}
    times = {s: [] for s in generators}
    for t, row in tiles.iterrows():
        image, mask, gsd = read_tile(row["img_path"], row["mask_path"])
        valid = mask != IGNORE
        gt = np.where(valid, mask, 0)
        inst, info = instances(mask, args.min_inst_m2 / (gsd / 1000.0) ** 2)
        tile_id = os.path.basename(row["img_path"])
        for setting, gen in generators.items():
            t0 = time.time()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                segs = [m["segmentation"] for m in gen.generate(image)]
            times[setting].append(time.time() - t0)
            for p, r in enumerate(amg_metrics(segs, inst, info, valid), start=1):
                c, a, _ = info[p - 1]
                inst_rows.append(dict(tile=tile_id, site=row["site_id"], setting=setting, inst=p, cls=c,
                                      cls_name=names[c], area_px=a, area_m2=a * (gsd / 1000) ** 2,
                                      n_masks=len(segs), **r))
            pred = oracle_map(segs, gt, valid)
            confs[setting] += np.bincount(gt[valid] * N_CLS + pred[valid],
                                          minlength=N_CLS * N_CLS).reshape(N_CLS, N_CLS)
            if setting == "tile+crops" and t < args.n_vis:
                save_vis(os.path.join(args.out_dir, "vis", f"{t:03d}_{row['site']}_{names[row['drawn_for']]}.png"),
                         image, mask, segs, pred, colors, f"{tile_id}  ({gsd:.2f} mm/px, {len(info)} plants)")
            del segs
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for r in prompt_metrics(predictor, image, inst, info, args.prompt_per_class, args.crop, rng):
                prm_rows.append(dict(tile=tile_id, site=row["site_id"], cls_name=names[r["cls"]], **r))
        print(f"[{t + 1}/{len(tiles)}] {tile_id}: {len(info)} plants, "
              + ", ".join(f"{s} {times[s][-1]:.0f}s" for s in generators), flush=True)

    # %% outputs
    inst_df, prm_df = pd.DataFrame(inst_rows), pd.DataFrame(prm_rows)
    inst_df.to_csv(os.path.join(args.out_dir, "instances.csv"), index=False)
    prm_df.to_csv(os.path.join(args.out_dir, "prompts.csv"), index=False)
    with open(os.path.join(args.out_dir, "confusion.json"), "w") as f:
        json.dump({"classes": names, **{s: c.tolist() for s, c in confs.items()}}, f, indent=1)
    md = summarise(inst_df, prm_df, confs, names, args, len(tiles), {s: np.mean(v) for s, v in times.items()})
    with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
