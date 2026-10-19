#!/usr/bin/env python3
"""Background pool for copy-paste synthesis: every label-free window of every site.

Each tile of the ``train`` and ``test`` folders of all 20 sites (``val`` stays out,
it picks the best epoch) is cut into an exclusive grid of --size windows. A window
enters the pool if it holds no labelled species pixel and >= 50% valid pixels
(alpha > 0, not ignore), the same rule as the background windows of
plantclef_probe.py. Each pool window gets a DINOv2 ViT-B/14 reg4 CLS embedding
(resized to 518, as in the probe). No pixels are stored, only references.

Per fold, background_clusters.py --pool keeps the fold's training sites and
clusters them, so held-out sites never shape the clusters.

    python background_pool.py
    python background_pool.py --max-tiles 20 --out-dir results/background_pool_debug   # quick check

Outputs in --out-dir:

    pool.csv   one row per window: img_path, mask_path, y0, x0, size, site, flight, year,
               date, folder, gsd_mm, valid_frac, n_ignore (uncertain/ignore pixels),
               only_background_tile, brightness, exg (excess green (2G-R-B)/(R+G+B))
    emb.npz    DINOv2 CLS embeddings (float16), same row order as pool.csv
    parts/     per-chunk results; a rerun skips finished chunks

Derived from the confidential dataset: keep everything on the node.
"""

import argparse
import os

import numpy as np
import pandas as pd
import rasterio
import torch

from preview_paste import IGNORE, read_tile

DINO_MODEL = "vit_base_patch14_reg4_dinov2.lvd142m"  # plain DINOv2 of plantclef_probe.py


class PoolTiles(torch.utils.data.Dataset):
    """One tile -> its label-free windows (uint8, n x 3 x size x size) and their metadata."""

    def __init__(self, tiles, size):
        self.tiles, self.size = tiles, size

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, i):
        row, s = self.tiles.iloc[i], self.size
        image, mask, gsd = read_tile(row["img_path"], row["mask_path"])
        with rasterio.open(row["img_path"]) as src:
            alpha = src.read(4) if src.count >= 4 else np.full(mask.shape, 255, np.uint8)
        valid = (mask != IGNORE) & (alpha > 0)
        crops, meta = [], []
        for y0 in range(0, mask.shape[0] - s + 1, s):
            for x0 in range(0, mask.shape[1] - s + 1, s):
                m, v = mask[y0:y0 + s, x0:x0 + s], valid[y0:y0 + s, x0:x0 + s]
                # # same rule as plantclef_probe.window_label == 0: >= 50% valid, no species pixel
                if v.mean() < 0.5 or ((m >= 1) & (m <= 6) & v).any():
                    continue
                rgb = image[y0:y0 + s, x0:x0 + s].astype(np.float32)
                r, g, b = rgb[v, 0], rgb[v, 1], rgb[v, 2]
                crops.append(image[y0:y0 + s, x0:x0 + s].transpose(2, 0, 1))
                meta.append({"row": i, "y0": y0, "x0": x0, "valid_frac": float(v.mean()),
                             "n_ignore": int(((m == IGNORE) & (alpha[y0:y0 + s, x0:x0 + s] > 0)).sum()),
                             "brightness": float((r + g + b).mean() / 765),
                             "exg": float(((2 * g - r - b) / (r + g + b + 1e-6)).mean()), "gsd_mm": gsd})
        crops = torch.from_numpy(np.stack(crops)) if crops else torch.zeros(0, 3, s, s, dtype=torch.uint8)
        return crops, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stats-csv", default="../data/Neophytes/stats_imagewise.csv")
    parser.add_argument("--out-dir", default="results/background_pool")
    parser.add_argument("--size", type=int, default=512, help="window size in tile px")
    parser.add_argument("--chunk", type=int, default=2000, help="tiles per saved part")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-tiles", type=int, default=0, help="tiles for a quick check (0 = all)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import timm
    import torch.nn.functional as F

    os.makedirs(os.path.join(args.out_dir, "parts"), exist_ok=True)

    # %% tiles: train + test folders of all sites
    tiles = pd.read_csv(args.stats_csv)
    tiles = tiles[tiles["split"].isin(["train", "test"])].reset_index(drop=True)
    if args.max_tiles:
        tiles = tiles.sample(args.max_tiles, random_state=args.seed).reset_index(drop=True)
    print(f"{len(tiles)} tiles ({(tiles['only_background'] == 1).sum()} background-only), "
          f"{tiles['site'].nunique()} flights", flush=True)

    # %% embeddings per chunk
    model = timm.create_model(DINO_MODEL, pretrained=True, num_classes=0).cuda().eval()
    cfg = timm.data.resolve_model_data_config(model)
    mean = torch.tensor(cfg["mean"], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device="cuda").view(1, 3, 1, 1)
    res = cfg["input_size"][-1]
    for start in range(0, len(tiles), args.chunk):
        part = os.path.join(args.out_dir, "parts", f"part_{start // args.chunk:03d}")
        if os.path.exists(part + ".npz"):
            print(f"tiles {start}-: loaded {part}.npz", flush=True)
            continue
        loader = torch.utils.data.DataLoader(PoolTiles(tiles.iloc[start:start + args.chunk], args.size),
                                             batch_size=None, num_workers=args.workers)
        embs, metas = [], []
        with torch.no_grad():
            for n, (crops, meta) in enumerate(loader):
                for b in range(0, len(crops), args.batch):
                    x = crops[b:b + args.batch].cuda().float().div(255)
                    x = F.interpolate(x, size=(res, res), mode="bicubic", antialias=True, align_corners=False)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        embs.append(model((x.clamp(0, 1) - mean) / std).half().cpu())
                for m in meta:
                    m["row"] += start
                metas += meta
                if (n + 1) % 500 == 0:
                    print(f"  tiles {start + n + 1}/{len(tiles)}, {len(metas)} windows in this part", flush=True)
        emb = torch.cat(embs).numpy() if embs else np.zeros((0, model.num_features), np.float16)
        np.savez(part + ".npz", emb=emb)
        pd.DataFrame(metas).to_csv(part + ".csv", index=False)
        print(f"tiles {start}-{start + len(loader) - 1}: {len(metas)} windows -> {part}", flush=True)

    # %% pool.csv + emb.npz
    n_parts = (len(tiles) + args.chunk - 1) // args.chunk
    parts = [os.path.join(args.out_dir, "parts", f"part_{p:03d}") for p in range(n_parts)]
    meta = pd.concat([pd.read_csv(p + ".csv") for p in parts], ignore_index=True)
    emb = np.concatenate([np.load(p + ".npz")["emb"] for p in parts])
    t = tiles.iloc[meta["row"]].reset_index(drop=True)
    pool = pd.DataFrame({
        "img_path": t["img_path"], "mask_path": t["mask_path"], "y0": meta["y0"], "x0": meta["x0"],
        "size": args.size, "site": t["site"].str.rsplit("_", n=1).str[0], "flight": t["site"], "year": t["year"],
        "date": t["img_path"].map(lambda p: "-".join(os.path.basename(p).split("_")[:3])), "folder": t["split"],
        "gsd_mm": meta["gsd_mm"], "valid_frac": meta["valid_frac"], "n_ignore": meta["n_ignore"],
        "only_background_tile": t["only_background"], "brightness": meta["brightness"], "exg": meta["exg"]})
    pool.to_csv(os.path.join(args.out_dir, "pool.csv"), index=False)
    np.savez(os.path.join(args.out_dir, "emb.npz"), emb=emb)
    print(f"pool: {len(pool)} windows from {pool['img_path'].nunique()} tiles, {pool['site'].nunique()} sites "
          f"({(pool['only_background_tile'] == 1).mean():.0%} from background-only tiles) -> {args.out_dir}")


if __name__ == "__main__":
    main()
