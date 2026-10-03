#!/usr/bin/env python3
"""Read each tile's georeferencing and write cache/grid.parquet.

The orthophoto of one flight is cut into a regular grid of 2048 px tiles.
The map view places those tiles by the GeoTIFF origin, so this pass only
reads headers.

    python build_grid.py
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pyarrow as pa
import pyarrow.parquet as pq
import rasterio

from common import GRID_PARQUET, TILES_PARQUET, resolve_data_path

SCHEMA = pa.schema([
    ("tile_id", pa.string()),
    ("origin_x", pa.float64()),
    ("origin_y", pa.float64()),
    ("gsd_m", pa.float64()),
])


def _origin(task):
    tile_id, path = task
    with rasterio.open(path) as src:
        transform = src.transform
        return {
            "tile_id": tile_id,
            "origin_x": float(transform.c),
            "origin_y": float(transform.f),
            "gsd_m": float(abs(transform.a)),
        }


def main():
    table = pq.read_table(TILES_PARQUET, columns=["tile_id", "img_path"])
    tasks = [
        (row["tile_id"], str(resolve_data_path(row["img_path"])))
        for row in table.to_pylist()
    ]
    rows = []
    done = 0
    workers = min(16, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_origin, task) for task in tasks]
        for future in as_completed(futures):
            rows.append(future.result())
            done += 1
            if done % 4000 == 0 or done == len(tasks):
                print(f"{done}/{len(tasks)}", flush=True)
    out = pa.Table.from_pylist(rows, schema=SCHEMA)
    tmp = GRID_PARQUET.with_suffix(".parquet.tmp")
    pq.write_table(out, tmp)
    os.replace(tmp, GRID_PARQUET)
    print(f"Wrote {len(rows)} origins to {GRID_PARQUET}")


if __name__ == "__main__":
    main()
