#!/usr/bin/env python3
"""Linear probe: are frozen PlantCLEF 2024 embeddings of drone windows separable by species?

Every window of an exclusive grid (--sizes) is resized to 518 and passed through two
frozen backbones: PlantCLEF 2024 ViT-B/14 reg4 "onlyclassifier_then_all" (DINOv2
fine-tuned on 1.4M close-up plant photos, Zenodo 10.5281/zenodo.10848263) and plain
DINOv2 ViT-B/14 reg4 (the same backbone before the plant fine-tuning). A linear head
(6 species + background) is trained on the CLS embeddings, per window size.

Window label: the dominant species if it covers >= --pos-thr of the valid pixels,
background if the window holds no species pixel. Windows in between, or with < 50%
valid pixels (ignore 254/255 and nodata excluded), are skipped.

Split as for the cv3 baseline (split_cv3.yaml):
  train  all labelled tiles of the training sites + background-only tiles making up
         --bg-fraction of the tiles (as bg_fraction in data.yaml); class-balanced loss
  val    the val folders of the training sites, sampled the same way; picks the
         regularisation C (as the baseline picks its best epoch on val)
  test   every window of every tile of the held-out sites (as test.py)

Rows per window size, all scored on test with metrics_from_confusion (as test.py):
  linear PlantCLEF / linear DINOv2   the trained head on each backbone
  zero-shot top1                     PlantCLEF's own head: top-1 over 7806 species if
                                     it is one of our 6, else background
  zero-shot thr                      argmax over our 6 if max p >= tau (tau picked on val)

    python plantclef_probe.py --weights-dir ../weights/plantclef2024/pretrained_models
    python plantclef_probe.py ... --max-tiles 20 --out-dir results/plantclef_probe_debug   # quick check

Embeddings are cached in --out-dir (emb_<split>.npz) and reused on a rerun.
Outputs are derived from the confidential dataset: keep them on the node.
"""

import argparse
import os

import numpy as np
import pandas as pd
import rasterio
import torch
import yaml

from sam_oracle import IGNORE, N_CLS, load_config, read_tile

PC_MODEL = "vit_base_patch14_reg4_dinov2.lvd142m"
PC_DIR = "vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all"
# # PlantCLEF species of our class ids 1..6 (configs/data/neophytes_names_colors.yaml)
SPECIES = ["Buddleja davidii", "Rhus typhina", "Reynoutria japonica", "Bunias orientalis",
           "Senecio inaequidens", "Ailanthus altissima"]


def split_tiles(stats_csv, split_yaml, bg_fraction, max_tiles, rng):
    """Tile table per split: train/val = labelled + background-only tiles at bg_fraction, test = all."""
    with open(split_yaml) as f:
        dirs = yaml.safe_load(f)["image_dir"]
    df = pd.read_csv(stats_csv)
    out = {}
    for split in ["train", "val", "test"]:
        d = df[df["img_path"].str.startswith(tuple(p.rstrip("/") + "/" for p in dirs[split]))]
        if split != "test":
            lab, bg = d[d["only_background"] == 0], d[d["only_background"] == 1]
            n_bg = min(len(bg), round(bg_fraction / (1 - bg_fraction) * len(lab)))
            d = pd.concat([lab, bg.sample(n_bg, random_state=int(rng.integers(1 << 31)))])
        if max_tiles:
            d = d.sample(min(max_tiles, len(d)), random_state=int(rng.integers(1 << 31)))
        out[split] = d.reset_index(drop=True)
    return out


class Tiles(torch.utils.data.Dataset):
    """One tile -> RGB uint8 and reference mask (N_CLS = not counted: ignore or nodata)."""

    def __init__(self, tiles):
        self.tiles = tiles

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, i):
        row = self.tiles.iloc[i]
        image, mask, _ = read_tile(row["img_path"], row["mask_path"])
        with rasterio.open(row["img_path"]) as src:
            alpha = src.read(4) if src.count >= 4 else np.full(mask.shape, 255, np.uint8)
        return image, np.where((mask != IGNORE) & (alpha > 0), mask, N_CLS), i


def window_label(gt, pos_thr):
    px = np.bincount(gt.ravel(), minlength=N_CLS + 1)[:N_CLS]
    if px.sum() < 0.5 * gt.size:
        return -1
    if px[1:].sum() == 0:
        return 0
    c = int(px[1:].argmax()) + 1
    return c if px[c] >= pos_thr * px.sum() else -1


@torch.no_grad()
def embed_split(tiles, models, target_idx, sizes, pos_thr, batch, workers):
    """CLS embeddings of both backbones + PlantCLEF zero-shot outputs for every labelled window."""
    import torch.nn.functional as F

    pc, dino, mean, std, res = models
    to_ours = torch.zeros(pc.num_classes, dtype=torch.long, device="cuda")
    to_ours[torch.from_numpy(target_idx).cuda()] = torch.arange(1, N_CLS, device="cuda")
    t6 = torch.from_numpy(target_idx).cuda()
    out = {k: [] for k in ["size", "label", "tile", "pc", "dino", "p6", "top1"]}
    loader = torch.utils.data.DataLoader(Tiles(tiles), batch_size=None, num_workers=workers)
    for n, (image, gt, i) in enumerate(loader):
        gt = gt.numpy()
        x = image.cuda().permute(2, 0, 1).unsqueeze(0).float().div(255)
        wins = []
        for size in sizes:
            for y0 in range(0, gt.shape[0] - size + 1, size):
                for x0 in range(0, gt.shape[1] - size + 1, size):
                    label = window_label(gt[y0:y0 + size, x0:x0 + size], pos_thr)
                    if label < 0:
                        continue
                    out["size"].append(size)
                    out["label"].append(label)
                    out["tile"].append(int(i))
                    wins.append(F.interpolate(x[:, :, y0:y0 + size, x0:x0 + size], size=(res, res),
                                              mode="bicubic", antialias=True, align_corners=False))
        if not wins:
            continue
        xb = (torch.cat(wins).clamp(0, 1) - mean) / std
        for s in range(0, len(xb), batch):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                cls_pc = pc.forward_head(pc.forward_features(xb[s:s + batch]), pre_logits=True)
                logits = pc.head(cls_pc).float()
                cls_dino = dino(xb[s:s + batch])
            out["pc"].append(cls_pc.half().cpu())
            out["dino"].append(cls_dino.half().cpu())
            out["p6"].append(torch.softmax(logits, 1)[:, t6].cpu())
            out["top1"].append(to_ours[logits.argmax(1)].cpu())
        if (n + 1) % 200 == 0:
            print(f"  {n + 1}/{len(tiles)} tiles, {len(out['label'])} windows", flush=True)
    return {k: (torch.cat(v).numpy() if isinstance(v[0], torch.Tensor) else np.array(v)) for k, v in out.items()}


def confusion(y, pred):
    return np.bincount(y * N_CLS + pred, minlength=N_CLS * N_CLS).reshape(N_CLS, N_CLS)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights-dir", default="../weights/plantclef2024/pretrained_models",
                        help="unpacked Zenodo archive (class_mapping.txt, species_id_to_name.txt, model dirs)")
    parser.add_argument("--stats-csv", default="../data/Neophytes/stats_imagewise.csv")
    parser.add_argument("--split-yaml", default="configs/data/split_cv3.yaml")
    parser.add_argument("--classes-yaml", default="configs/data/neophytes_names_colors.yaml")
    parser.add_argument("--out-dir", default="results/plantclef_probe")
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--pos-thr", type=float, default=0.01, help="min share of the dominant species")
    parser.add_argument("--bg-fraction", type=float, default=0.10, help="share of background-only train/val tiles")
    parser.add_argument("--Cs", type=float, nargs="+", default=[0.001, 0.01, 0.1, 1.0])
    parser.add_argument("--max-tiles", type=int, default=0, help="tiles per split for a quick check (0 = all)")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import timm
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from utils.eval_utils import metrics_from_confusion

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    classes = load_config(args.classes_yaml)
    names = [classes[c]["name"] for c in range(N_CLS)]

    # %% PlantCLEF class index of each of our species
    cid_to_spid = [line.strip() for line in open(os.path.join(args.weights_dir, "class_mapping.txt"))]
    spid_to_sp = pd.read_csv(os.path.join(args.weights_dir, "species_id_to_name.txt"), sep=";", quoting=1,
                             dtype={"species_id": str}).set_index("species_id")["species"].to_dict()
    binomial = {" ".join(spid_to_sp[s].split()[:2]): i for i, s in enumerate(cid_to_spid)}
    target_idx = np.array([binomial[s] for s in SPECIES])

    # %% embeddings per split (cached)
    tiles = split_tiles(args.stats_csv, args.split_yaml, args.bg_fraction, args.max_tiles, rng)
    emb, models = {}, None
    for split, t in tiles.items():
        path = os.path.join(args.out_dir, f"emb_{split}.npz")
        if os.path.exists(path):
            emb[split] = dict(np.load(path))
            print(f"{split}: loaded {path}")
            continue
        if models is None:
            pc = timm.create_model(PC_MODEL, pretrained=False, num_classes=len(cid_to_spid),
                                   checkpoint_path=os.path.join(args.weights_dir, PC_DIR, "model_best.pth.tar"))
            dino = timm.create_model(PC_MODEL, pretrained=True, num_classes=0)
            cfg = timm.data.resolve_model_data_config(pc)
            models = (pc.cuda().eval(), dino.cuda().eval(),
                      torch.tensor(cfg["mean"], device="cuda").view(1, 3, 1, 1),
                      torch.tensor(cfg["std"], device="cuda").view(1, 3, 1, 1), cfg["input_size"][-1])
        print(f"{split}: {len(t)} tiles ({(t['only_background'] == 0).sum()} labelled)", flush=True)
        emb[split] = embed_split(t, models, target_idx, args.sizes, args.pos_thr, args.batch, args.workers)
        np.savez(path, **emb[split])

    # %% heads per window size, scored on the held-out sites
    md = ["# Linear probe on frozen embeddings, cv3 held-out sites\n",
          f"Windows: exclusive grid, resized to 518; species window = dominant species >= {args.pos_thr:.0%} "
          f"of valid pixels. Train/val: labelled tiles + {args.bg_fraction:.0%} background-only tiles; "
          f"test: every tile of the held-out sites ({len(tiles['test'])} tiles). F1 as in test.py.\n"]
    for size in args.sizes:
        sel = {s: emb[s]["size"] == size for s in emb}
        y = {s: emb[s]["label"][sel[s]] for s in emb}
        rows, notes = {}, []
        for key, name in [("pc", "linear PlantCLEF"), ("dino", "linear DINOv2")]:
            scaler = StandardScaler().fit(emb["train"][key][sel["train"]].astype(np.float32))
            X = {s: scaler.transform(emb[s][key][sel[s]].astype(np.float32)) for s in emb}
            best = None
            for C in args.Cs:
                head = LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(X["train"], y["train"])
                f1_val = metrics_from_confusion(confusion(y["val"], head.predict(X["val"])))["F1-avg-wo0"]
                if best is None or f1_val > best[0]:
                    best = (f1_val, C, head)
            rows[name] = metrics_from_confusion(confusion(y["test"], best[2].predict(X["test"])))
            notes.append(f"{name}: C = {best[1]:g} (val F1-avg-wo0 {best[0]:.3f})")
        rows["zero-shot top1"] = metrics_from_confusion(confusion(y["test"], emb["test"]["top1"][sel["test"]]))
        p6 = {s: emb[s]["p6"][sel[s]] for s in emb}
        thr_pred = lambda s, tau: np.where(p6[s].max(1) >= tau, p6[s].argmax(1) + 1, 0)
        taus = np.r_[0.0, np.logspace(-5, 0, 51)]
        tau = taus[int(np.nanargmax([metrics_from_confusion(confusion(y["val"], thr_pred("val", t)))["F1-avg-wo0"]
                                     for t in taus]))]
        rows["zero-shot thr"] = metrics_from_confusion(confusion(y["test"], thr_pred("test", tau)))
        notes.append(f"zero-shot thr: tau = {tau:.2e}")
        table = pd.DataFrame({k: {**{names[c]: m["F1"][c] for c in range(1, N_CLS)}, "F1-avg-wo0": m["F1-avg-wo0"],
                                  "P-avg-wo0": m["Precision-avg-wo0"], "R-avg-wo0": m["Recall-avg-wo0"]}
                              for k, m in rows.items()}).T
        n = np.bincount(y["test"], minlength=N_CLS)
        md += [f"\n## {size} px windows\n", "Test windows: " + ", ".join(f"{names[c]} {n[c]}" for c in range(N_CLS))
               + ". " + "; ".join(notes) + ".\n", table.to_markdown(floatfmt=".3f")]
    with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
