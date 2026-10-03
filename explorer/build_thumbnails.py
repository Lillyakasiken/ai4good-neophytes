#!/usr/bin/env python3
"""Precompute 256px WebP thumbnails for every tile in the catalog.

Two files per tile: an RGB thumbnail (empty orthophoto margin painted flat)
and a class-mask thumbnail with a transparent background, so the viewer can
lay the labels over the photo. Width, height, and ground sampling distance
are written to ``cache/raster_meta.parquet`` while each image is open.

The run is resumable. A tile whose thumb, mask, and meta row already exist
is skipped.

    python build_thumbnails.py              # all tiles
    python build_thumbnails.py --limit 20   # trial run
    python build_thumbnails.py --workers 8
"""

import argparse
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
from PIL import Image
from rasterio.enums import Resampling

from common import (
    CACHE,
    EMPTY_RGB,
    IGNORE_COLORS,
    MASKS_DIR,
    META_PARQUET,
    THUMB_PX,
    THUMBS_DIR,
    TILES_PARQUET,
    load_names,
    resolve_data_path,
)

_LUT = None


def _init_worker():
    global _LUT
    _LUT = _lut()

META_SCHEMA = pa.schema([
    ("tile_id", pa.string()),
    ("width", pa.int32()),
    ("height", pa.int32()),
    ("gsd_mm", pa.float64()),
])


def _lut():
    """Index by mask value: RGB plus alpha. Background stays transparent."""
    lut = np.zeros((256, 4), dtype=np.uint8)
    classes, _, _ = load_names()
    for spec in classes:
        if spec["id"] == 0:
            continue
        lut[spec["id"], :3] = spec["rgb"]
        lut[spec["id"], 3] = 255
    for value, rgb in IGNORE_COLORS.items():
        lut[value, :3] = rgb
        lut[value, 3] = 220
    return lut


def _thumb_hw(height, width):
    scale = THUMB_PX / max(height, width)
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


def _gsd_mm(src):
    try:
        gsd = abs(src.transform.a) * 1000.0
    except Exception:
        return None
    return float(gsd) if gsd > 0 else None


def _rgb_thumb(src, out_h, out_w):
    data = src.read(
        [1, 2, 3],
        out_shape=(3, out_h, out_w),
        resampling=Resampling.bilinear,
    )
    if data.dtype != np.uint8:
        peak = float(np.max(data)) if data.size else 0.0
        if peak <= 0:
            data = np.zeros_like(data, dtype=np.uint8)
        elif peak <= 255:
            data = data.astype(np.uint8)
        else:
            data = np.clip(data.astype(np.float32) / peak * 255.0, 0, 255).astype(np.uint8)
    rgb = np.moveaxis(data, 0, -1)
    # Band 4 is the orthophoto alpha. Some tiles also set nodata=0, and
    # read_masks() then ignores the alpha band, so read the band directly.
    if src.count >= 4:
        alpha = src.read(4, out_shape=(out_h, out_w), resampling=Resampling.nearest)
    else:
        alpha = src.read_masks(1, out_shape=(out_h, out_w), resampling=Resampling.nearest)
    rgb[alpha == 0] = EMPTY_RGB
    return rgb


def _mask_thumb(path, out_h, out_w, lut):
    with rasterio.open(path) as src:
        cls = src.read(1, out_shape=(out_h, out_w), resampling=Resampling.nearest)
    return lut[cls]


def render(task):
    """Write the missing files for one tile and return its raster meta row."""
    tile_id = task["tile_id"]
    thumb_path = task["thumb_path"]
    mask_path = task["mask_path"]
    try:
        with rasterio.open(task["img_path"]) as src:
            width, height = int(src.width), int(src.height)
            gsd = _gsd_mm(src)
            if task["write_thumb"]:
                out_h, out_w = _thumb_hw(height, width)
                rgb = _rgb_thumb(src, out_h, out_w)
            else:
                out_h = out_w = None
        if task["write_mask"]:
            if out_h is None:
                out_h, out_w = _thumb_hw(height, width)
            rgba = _mask_thumb(task["mask_src"], out_h, out_w, _LUT)
            Image.fromarray(rgba, mode="RGBA").save(mask_path, "WEBP", quality=80, method=4)
        if task["write_thumb"]:
            Image.fromarray(rgb, mode="RGB").save(thumb_path, "WEBP", quality=80, method=4)
        return {"tile_id": tile_id, "width": width, "height": height, "gsd_mm": gsd, "error": None}
    except Exception as exc:
        return {"tile_id": tile_id, "width": None, "height": None, "gsd_mm": None, "error": str(exc)}


def _load_meta():
    if not META_PARQUET.is_file():
        return []
    table = pq.read_table(META_PARQUET)
    return table.to_pylist()


def _write_meta(rows):
    table = pa.Table.from_pylist(rows, schema=META_SCHEMA)
    tmp = META_PARQUET.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, META_PARQUET)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="process at most this many tiles that still need work")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()

    if not TILES_PARQUET.is_file():
        raise SystemExit(f"Catalog missing: {TILES_PARQUET}. Run build_catalog.py first.")

    THUMBS_DIR.mkdir(parents=True, exist_ok=True)
    MASKS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    existing = _load_meta()
    have_meta = {row["tile_id"] for row in existing}
    catalog = pq.read_table(TILES_PARQUET, columns=["tile_id", "img_path", "mask_path"]).to_pylist()

    tasks = []
    for row in catalog:
        tid = row["tile_id"]
        thumb_path = THUMBS_DIR / f"{tid}.webp"
        mask_out = MASKS_DIR / f"{tid}.webp"
        write_thumb = not thumb_path.is_file()
        write_mask = not mask_out.is_file()
        need_meta = tid not in have_meta
        if not (write_thumb or write_mask or need_meta):
            continue
        tasks.append({
            "tile_id": tid,
            "img_path": str(resolve_data_path(row["img_path"])),
            "mask_src": str(resolve_data_path(row["mask_path"])),
            "thumb_path": str(thumb_path),
            "mask_path": str(mask_out),
            "write_thumb": write_thumb,
            "write_mask": write_mask,
        })
        if args.limit is not None and len(tasks) >= args.limit:
            break

    if not tasks:
        print(f"Nothing to do. {len(have_meta)} tiles already have thumbnails and meta.")
        return

    print(f"{len(tasks)} tiles to process with {args.workers} workers")
    rows = list(existing)
    errors = []
    done = 0
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx, initializer=_init_worker) as pool:
        futures = [pool.submit(render, task) for task in tasks]
        for fut in as_completed(futures):
            result = fut.result()
            done += 1
            if result["error"]:
                errors.append(f"{result['tile_id']}: {result['error']}")
            elif result["tile_id"] not in have_meta:
                rows.append({
                    "tile_id": result["tile_id"],
                    "width": result["width"],
                    "height": result["height"],
                    "gsd_mm": result["gsd_mm"],
                })
                have_meta.add(result["tile_id"])
            if done % 100 == 0 or done == len(tasks):
                _write_meta(rows)
                print(f"  {done}/{len(tasks)}")

    _write_meta(rows)
    if errors:
        err_path = CACHE / "thumb_errors.txt"
        err_path.write_text("\n".join(errors) + "\n")
        print(f"{len(errors)} tiles failed. See {err_path}")
    print(f"Meta rows: {len(rows)}")


if __name__ == "__main__":
    main()
