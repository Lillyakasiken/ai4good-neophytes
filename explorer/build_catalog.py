#!/usr/bin/env python3
"""Build the one-row-per-tile Parquet catalog from stats_imagewise.csv.

The CSV already holds pixel counts, areas, and phenology buckets. This script
adds the fields the viewer filters on: site, flight, acquisition date, how
much of the tile is empty, and the role of the tile in each CV fold. Role
comes from the same ``split_cv*.yaml`` files training uses, so a tile marked
``cv3 / train`` is a tile the baseline dataloader would see.

    python build_catalog.py
"""

import csv
import datetime as dt
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq

from common import (
    CACHE,
    DATA_CSV,
    FOLDS,
    TILES_PARQUET,
    load_names,
    load_role_maps,
    norm_rel,
    parse_acq,
    species_columns,
    split_key,
    split_site_flight,
    tile_id,
)

# Spot-checks against configs/data/split_cv3.yaml. The on-disk test folder of
# a training site is still a training tile; the on-disk train folder of a
# held-out site is a test tile; its val and test folders are unused.
CV3_EXPECT = {
    "2024/Aarau_1_1/train": "train",
    "2024/Aarau_1_1/test": "train",
    "2024/Aarau_1_1/val": "val",
    "2024/Basel_2_1/train": "test",
    "2024/Basel_2_1/val": "unused",
    "2024/Basel_2_1/test": "unused",
}


def _blank(value):
    return value is None or str(value).strip() == ""


def _as_int(value):
    if _blank(value):
        return None
    return int(float(value))


def _as_float(value):
    if _blank(value):
        return None
    return float(value)


def _arrow_type(name):
    if name == "acq_date":
        return pa.date32()
    if name == "labelled":
        return pa.bool_()
    if name.endswith("_m2") or name in ("nodata_frac", "label_m2"):
        return pa.float64()
    if (
        name.endswith("_px")
        or name.startswith("only_")
        or name.startswith("exactly_")
        or name in ("year", "month", "n_species", "nodata_px", "total_px")
    ):
        return pa.int64()
    return pa.string()


def _convert(name, value):
    kind = _arrow_type(name)
    if kind == pa.date32():
        if _blank(value):
            return None
        return dt.date.fromisoformat(value)
    if kind == pa.bool_():
        return bool(value)
    if kind == pa.float64():
        return _as_float(value)
    if kind == pa.int64():
        return _as_int(value)
    if _blank(value):
        return None
    return str(value)


def build():
    if not DATA_CSV.is_file():
        raise SystemExit(f"Stats CSV not found: {DATA_CSV}")

    classes, display_order, cv_folds = load_names()
    species = species_columns(classes, display_order)
    species_names = [s["stats_name"] for s in species]
    sites = sorted({s for names in cv_folds.values() for s in names}, key=len, reverse=True)
    site_set = set(sites)
    role_maps = load_role_maps()

    with DATA_CSV.open(newline="") as fh:
        reader = csv.DictReader(fh)
        csv_fields = list(reader.fieldnames)
        raw_rows = list(reader)

    derived = [
        "tile_id", "site_name", "flight", "acq_date", "month",
        "nodata_frac", "n_species", "labelled", "label_m2",
    ]
    role_fields = [f"{fold}_role" for fold in FOLDS]
    fields = derived + csv_fields + role_fields
    columns = {name: [] for name in fields}

    unmatched = set()
    missing_key = 0
    role_counts = {fold: Counter() for fold in FOLDS}
    cv3_seen = {key: set() for key in CV3_EXPECT}
    no_date = 0

    for raw in raw_rows:
        img = norm_rel(raw["img_path"])
        mask = norm_rel(raw["mask_path"])
        key = split_key(img)
        if key is None:
            missing_key += 1
        folder = raw["site"]
        site_name, flight = split_site_flight(folder, sites)
        if site_name not in site_set:
            unmatched.add(folder)
        acq, month = parse_acq(img)
        if acq is None:
            no_date += 1

        px = {name: _as_int(raw[f"{name}_px"]) or 0 for name in species_names}
        areas = {name: _as_float(raw[f"{name}_m2"]) or 0.0 for name in species_names}
        total_px = _as_int(raw["total_px"]) or 0
        nodata_px = _as_int(raw["nodata_px"]) or 0
        present = [name for name in species_names if px[name] > 0]

        values = {
            "tile_id": tile_id(img),
            "site_name": site_name,
            "flight": flight,
            "acq_date": acq,
            "month": month,
            "nodata_frac": (nodata_px / total_px) if total_px else 0.0,
            "n_species": len(present),
            "labelled": bool(present),
            "label_m2": float(sum(areas.values())),
        }
        for name in csv_fields:
            if name in ("img_path", "mask_path"):
                values[name] = img if name == "img_path" else mask
            else:
                values[name] = raw[name]
        for fold in FOLDS:
            role = role_maps[fold].get(key, "unused") if key else "unused"
            values[f"{fold}_role"] = role
            role_counts[fold][role] += 1
        if key in cv3_seen:
            cv3_seen[key].add(values["cv3_role"])

        for name in fields:
            columns[name].append(_convert(name, values[name]))

    ids = columns["tile_id"]
    if len(ids) != len(set(ids)):
        raise SystemExit("tile_id collision; widen the hash before shipping")

    for key, expected in CV3_EXPECT.items():
        got = cv3_seen[key]
        if got != {expected}:
            raise SystemExit(f"cv3 role for {key}: expected {expected}, got {got or 'no tiles'}")

    arrays = [pa.array(columns[name], type=_arrow_type(name)) for name in fields]
    table = pa.Table.from_arrays(arrays, names=fields)
    CACHE.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, TILES_PARQUET)

    print(f"Wrote {table.num_rows} tiles to {TILES_PARQUET}")
    for fold in FOLDS:
        counts = ", ".join(f"{role} {role_counts[fold][role]}" for role in ("train", "val", "test", "unused"))
        print(f"  {fold}: {counts}")
    labelled = sum(columns["labelled"])
    print(f"  labelled tiles: {labelled}    background-only: {table.num_rows - labelled}")
    if unmatched:
        print(f"  site folders not in cv_folds: {sorted(unmatched)}")
    if missing_key:
        print(f"  tiles with no year/site/split key: {missing_key}")
    if no_date:
        print(f"  tiles with no acquisition date in the filename: {no_date}")


if __name__ == "__main__":
    build()
