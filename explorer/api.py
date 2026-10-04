"""Tile explorer API. Queried with DuckDB over the Parquet catalog.

Run from this directory, with its virtualenv:

    uvicorn api:app --port 8000
"""

import math
import threading
from datetime import date, datetime

import duckdb
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from common import (
    CACHE,
    EMPTY_RGB,
    FOLDS,
    GRID_PARQUET,
    MASKS_DIR,
    META_PARQUET,
    PHASE_LABELS,
    PHASES,
    ROLES,
    THUMB_PX,
    TILES_PARQUET,
    THUMBS_DIR,
    load_names,
    resolve_data_path,
    species_columns,
)
from segment import (
    SegmentFailed,
    SegmentUnavailable,
    catalog as segment_catalog,
    run as segment_run,
)
from vision import catalog, load_source, render

CLASSES, DISPLAY_ORDER, CV_FOLDS = load_names()
SPECIES = species_columns(CLASSES, DISPLAY_ORDER)
SPECIES_NAMES = [s["stats_name"] for s in SPECIES]
SPECIES_BY_NAME = {s["stats_name"]: s for s in SPECIES}

NODATA_BINS = (
    ("lt_1", "under 1%"),
    ("p1_5", "1–5%"),
    ("p5_10", "5–10%"),
    ("p10_25", "10–25%"),
    ("p25_50", "25–50%"),
    ("ge_50", "50% or more"),
)

SORTS = {
    "label_m2": "label_m2 DESC, tile_id",
    "nodata_frac": "nodata_frac DESC, tile_id",
    "acq_date": "acq_date NULLS LAST, tile_id",
    "site_name": "site_name, flight, acq_date, tile_id",
}
for _name in SPECIES_NAMES:
    SORTS[f"{_name}_m2"] = f"{_name}_m2 DESC, tile_id"

app = FastAPI(title="Neophyte tile explorer")
THUMBS_DIR.mkdir(parents=True, exist_ok=True)
MASKS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/thumbs", StaticFiles(directory=THUMBS_DIR), name="thumbs")
app.mount("/masks", StaticFiles(directory=MASKS_DIR), name="masks")


def _require_catalog():
    if not TILES_PARQUET.is_file():
        raise HTTPException(status_code=503, detail="Catalog missing. Run explorer/build_catalog.py.")


def _fold(fold):
    if fold not in FOLDS:
        raise HTTPException(status_code=400, detail=f"fold must be one of {', '.join(FOLDS)}")
    return fold


def _jsonify(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _fetch(sql, params):
    con = duckdb.connect()
    try:
        con.execute(sql, params)
        columns = [d[0] for d in con.description]
        return [
            {key: _jsonify(value) for key, value in zip(columns, row)}
            for row in con.fetchall()
        ]
    finally:
        con.close()


def _source():
    """FROM clause. Raster meta is joined when the thumbnail pass has written it."""
    if META_PARQUET.is_file():
        return (
            "FROM read_parquet(?) t LEFT JOIN read_parquet(?) m USING (tile_id)",
            [str(TILES_PARQUET), str(META_PARQUET)],
        )
    return (
        "FROM read_parquet(?) t",
        [str(TILES_PARQUET)],
    )


def _where(
    fold, role, split, site, flight, year, month, species, labelled,
    nodata_min, nodata_max, min_m2, min_m2_species, include_role=True,
    background_only=False,
):
    clauses = []
    params = []
    if include_role and role and role != "all":
        if role not in ROLES:
            raise HTTPException(status_code=400, detail=f"role must be one of {', '.join(ROLES)} or all")
        clauses.append(f"{fold}_role = ?")
        params.append(role)
    if split:
        if split not in ("train", "val", "test"):
            raise HTTPException(status_code=400, detail="split must be train, val, or test")
        clauses.append("split = ?")
        params.append(split)
    if site:
        clauses.append("site_name = ?")
        params.append(site)
    if flight:
        clauses.append("flight = ?")
        params.append(flight)
    if year is not None:
        clauses.append("year = ?")
        params.append(year)
    if month is not None:
        clauses.append("month = ?")
        params.append(month)
    if species:
        parts = []
        for name in species:
            if name not in SPECIES_BY_NAME:
                raise HTTPException(status_code=400, detail=f"unknown species {name}")
            parts.append(f"{name}_px > 0")
        clauses.append("(" + " OR ".join(parts) + ")")
    if labelled:
        clauses.append("labelled")
    if background_only:
        clauses.append("NOT labelled")
    if nodata_min is not None:
        clauses.append("nodata_frac >= ?")
        params.append(nodata_min)
    if nodata_max is not None:
        clauses.append("nodata_frac <= ?")
        params.append(nodata_max)
    if min_m2 is not None:
        if min_m2_species not in SPECIES_BY_NAME:
            raise HTTPException(status_code=400, detail="min_m2_species must be a species stats name")
        clauses.append(f"{min_m2_species}_m2 >= ?")
        params.append(min_m2)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


def _present(row):
    return [name for name in SPECIES_NAMES if (row.get(f"{name}_px") or 0) > 0]


def _tile_brief(row):
    return {
        "tile_id": row["tile_id"],
        "site_name": row["site_name"],
        "flight": row["flight"],
        "site": row["site"],
        "year": row["year"],
        "month": row["month"],
        "acq_date": row["acq_date"],
        "split": row["split"],
        "cv_role": row["cv_role"],
        "labelled": row["labelled"],
        "label_m2": row["label_m2"],
        "nodata_frac": row["nodata_frac"],
        "n_species": row["n_species"],
        "species": _present(row),
        "thumb": f"/thumbs/{row['tile_id']}.webp",
        "mask": f"/masks/{row['tile_id']}.webp",
    }


@app.get("/api/facets")
def facets():
    _require_catalog()
    flights = _fetch(
        """
        SELECT site_name, flight, COUNT(*)::BIGINT AS tiles
        FROM read_parquet(?)
        GROUP BY 1, 2
        ORDER BY 1, 2
        """,
        [str(TILES_PARQUET)],
    )
    years = _fetch(
        "SELECT DISTINCT year FROM read_parquet(?) WHERE year IS NOT NULL ORDER BY 1",
        [str(TILES_PARQUET)],
    )
    months = _fetch(
        "SELECT DISTINCT month FROM read_parquet(?) WHERE month IS NOT NULL ORDER BY 1",
        [str(TILES_PARQUET)],
    )
    return {
        "sites": sorted({row["site_name"] for row in flights}),
        "flights": flights,
        "years": [row["year"] for row in years],
        "months": [row["month"] for row in months],
        "species": [
            {"id": s["id"], "stats_name": s["stats_name"], "name": s["name"], "color": s["color"]}
            for s in SPECIES
        ],
        "folds": {fold: {"test_sites": sites} for fold, sites in CV_FOLDS.items()},
        "sorts": list(SORTS),
    }


@app.get("/api/tiles")
def tiles(
    fold: str = "cv3",
    role: str = "train",
    split: str | None = None,
    site: str | None = None,
    flight: str | None = None,
    year: int | None = None,
    month: int | None = None,
    species: list[str] = Query(default=[]),
    labelled: bool = False,
    background_only: bool = False,
    nodata_min: float | None = None,
    nodata_max: float | None = None,
    min_m2: float | None = None,
    min_m2_species: str | None = None,
    sort: str = "label_m2",
    limit: int = Query(default=60, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    _require_catalog()
    fold = _fold(fold)
    if sort not in SORTS:
        raise HTTPException(status_code=400, detail=f"sort must be one of {', '.join(SORTS)}")
    where, params = _where(
        fold, role, split, site, flight, year, month, species, labelled,
        nodata_min, nodata_max, min_m2, min_m2_species,
        background_only=background_only,
    )
    px_cols = ", ".join(f"{name}_px" for name in SPECIES_NAMES)
    source, source_params = "FROM read_parquet(?) t", [str(TILES_PARQUET)]
    count_sql = f"SELECT COUNT(*)::BIGINT AS total {source} {where}"
    total = _fetch(count_sql, source_params + params)[0]["total"]
    list_sql = f"""
        SELECT tile_id, site_name, flight, site, year, month, acq_date, split,
               {fold}_role AS cv_role, labelled, label_m2, nodata_frac, n_species,
               {px_cols}
        {source}
        {where}
        ORDER BY {SORTS[sort]}
        LIMIT ? OFFSET ?
    """
    rows = _fetch(list_sql, source_params + params + [limit, offset])
    return {"total": total, "limit": limit, "offset": offset, "tiles": [_tile_brief(row) for row in rows]}


@app.get("/api/tiles/{tile_id}")
def tile_detail(tile_id: str):
    _require_catalog()
    if len(tile_id) != 16 or any(c not in "0123456789abcdef" for c in tile_id):
        raise HTTPException(status_code=400, detail="tile_id must be 16 hex characters")
    source, source_params = _source()
    width_cols = (
        "m.width AS width, m.height AS height, m.gsd_mm AS gsd_mm"
        if META_PARQUET.is_file()
        else "NULL::INTEGER AS width, NULL::INTEGER AS height, NULL::DOUBLE AS gsd_mm"
    )
    rows = _fetch(
        f"SELECT t.*, {width_cols} {source} WHERE t.tile_id = ?",
        source_params + [tile_id],
    )
    if not rows:
        raise HTTPException(status_code=404, detail="tile not found")
    row = rows[0]
    species = []
    for spec in SPECIES:
        name = spec["stats_name"]
        species.append({
            "stats_name": name,
            "name": spec["name"],
            "color": spec["color"],
            "px": row[f"{name}_px"],
            "m2": row[f"{name}_m2"],
            "phases": [
                {
                    "phase": phase,
                    "label": PHASE_LABELS[phase],
                    "px": row[f"{phase}_{name}_px"],
                    "m2": row[f"{phase}_{name}_m2"],
                }
                for phase in PHASES
            ],
        })
    return {
        "tile_id": row["tile_id"],
        "img_path": row["img_path"],
        "mask_path": row["mask_path"],
        "site_name": row["site_name"],
        "flight": row["flight"],
        "site": row["site"],
        "year": row["year"],
        "month": row["month"],
        "acq_date": row["acq_date"],
        "split": row["split"],
        "nodata_px": row["nodata_px"],
        "total_px": row["total_px"],
        "nodata_frac": row["nodata_frac"],
        "width": row["width"],
        "height": row["height"],
        "gsd_mm": row["gsd_mm"],
        "labelled": row["labelled"],
        "n_species": row["n_species"],
        "label_m2": row["label_m2"],
        "background": {"px": row["background_px"], "m2": row["background_m2"]},
        "roles": {fold: row[f"{fold}_role"] for fold in FOLDS},
        "species": species,
        "thumb": f"/thumbs/{row['tile_id']}.webp",
        "mask": f"/masks/{row['tile_id']}.webp",
    }


class PreviewIn(BaseModel):
    max_edge: int = 512
    ops: list[dict] = Field(default_factory=list)


def _tile_image_path(tile_id):
    if len(tile_id) != 16 or any(c not in "0123456789abcdef" for c in tile_id):
        raise HTTPException(status_code=400, detail="tile_id must be 16 hex characters")
    _require_catalog()
    rows = _fetch(
        "SELECT img_path FROM read_parquet(?) WHERE tile_id = ?",
        [str(TILES_PARQUET), tile_id],
    )
    if not rows:
        raise HTTPException(status_code=404, detail="tile not found")
    path = resolve_data_path(rows[0]["img_path"])
    if not path.is_file():
        raise HTTPException(status_code=404, detail="tile image is not on disk")
    return path


@app.get("/api/vision/ops")
def vision_ops():
    """Whitelist of preview filters and their slider ranges."""
    return catalog()


@app.post("/api/tiles/{tile_id}/preview")
def tile_preview(tile_id: str, body: PreviewIn):
    """Filtered preview and histogram. The JPEG is not written to disk."""
    path = _tile_image_path(tile_id)
    try:
        rgb, alpha, elev = load_source(tile_id, path, body.max_edge)
        return render(rgb, alpha, body.ops, elev=elev)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class SegmentIn(BaseModel):
    model: str = "yoloe-26s"
    max_edge: int = 512
    conf: float = 0.25
    texts: list[str] = Field(default_factory=list)
    points: list[dict] = Field(default_factory=list)
    boxes: list[dict] = Field(default_factory=list)


@app.get("/api/segment/models")
def segment_models():
    """Whitelist of zero-shot segmentation models and species phrases."""
    return segment_catalog()


@app.post("/api/tiles/{tile_id}/segment")
def tile_segment(tile_id: str, body: SegmentIn):
    """Instance overlay for one tile. The PNG is not written to disk."""
    path = _tile_image_path(tile_id)
    try:
        rgb, alpha, _elev = load_source(tile_id, path, body.max_edge)
        return segment_run(
            rgb,
            alpha,
            model=body.model,
            max_edge=body.max_edge,
            conf=body.conf,
            texts=body.texts,
            points=body.points,
            boxes=body.boxes,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SegmentUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except SegmentFailed as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/stats")
def stats(
    fold: str = "cv3",
    role: str = "train",
    split: str | None = None,
    site: str | None = None,
    flight: str | None = None,
    year: int | None = None,
    month: int | None = None,
    species: list[str] = Query(default=[]),
    labelled: bool = False,
    background_only: bool = False,
    nodata_min: float | None = None,
    nodata_max: float | None = None,
    min_m2: float | None = None,
    min_m2_species: str | None = None,
):
    _require_catalog()
    fold = _fold(fold)
    common = dict(
        fold=fold, split=split, site=site, flight=flight, year=year, month=month,
        species=species, labelled=labelled, background_only=background_only,
        nodata_min=nodata_min, nodata_max=nodata_max,
        min_m2=min_m2, min_m2_species=min_m2_species,
    )
    where, params = _where(role=role, include_role=True, **common)
    where_roles, params_roles = _where(role=role, include_role=False, **common)
    source, source_params = "FROM read_parquet(?) t", [str(TILES_PARQUET)]

    totals = _fetch(
        f"""
        SELECT COUNT(*)::BIGINT AS total,
               COALESCE(SUM(CASE WHEN labelled THEN 1 ELSE 0 END), 0)::BIGINT AS labelled
        {source} {where}
        """,
        source_params + params,
    )[0]

    species_exprs = []
    for name in SPECIES_NAMES:
        species_exprs.append(f"COALESCE(SUM(CASE WHEN {name}_px > 0 THEN 1 ELSE 0 END), 0)::BIGINT AS {name}_tiles")
        species_exprs.append(f"COALESCE(SUM({name}_m2), 0)::DOUBLE AS {name}_m2")
        for phase in PHASES:
            species_exprs.append(
                f"COALESCE(SUM({phase}_{name}_m2), 0)::DOUBLE AS {phase}_{name}_m2"
            )
    species_exprs.append("COALESCE(SUM(background_m2), 0)::DOUBLE AS background_m2")
    wide = _fetch(
        f"SELECT {', '.join(species_exprs)} {source} {where}",
        source_params + params,
    )[0]

    by_site = _fetch(
        f"""
        SELECT site_name,
               COUNT(*)::BIGINT AS tiles,
               COALESCE(SUM(CASE WHEN labelled THEN 1 ELSE 0 END), 0)::BIGINT AS labelled
        {source} {where}
        GROUP BY 1
        ORDER BY tiles DESC, site_name
        """,
        source_params + params,
    )
    by_month = _fetch(
        f"""
        SELECT year, month,
               COUNT(*)::BIGINT AS tiles,
               COALESCE(SUM(CASE WHEN labelled THEN 1 ELSE 0 END), 0)::BIGINT AS labelled
        {source} {where}
        GROUP BY 1, 2
        ORDER BY 1, 2
        """,
        source_params + params,
    )
    nodata_exprs = [
        "COALESCE(SUM(CASE WHEN nodata_frac < 0.01 THEN 1 ELSE 0 END), 0)::BIGINT AS lt_1",
        "COALESCE(SUM(CASE WHEN nodata_frac >= 0.01 AND nodata_frac < 0.05 THEN 1 ELSE 0 END), 0)::BIGINT AS p1_5",
        "COALESCE(SUM(CASE WHEN nodata_frac >= 0.05 AND nodata_frac < 0.10 THEN 1 ELSE 0 END), 0)::BIGINT AS p5_10",
        "COALESCE(SUM(CASE WHEN nodata_frac >= 0.10 AND nodata_frac < 0.25 THEN 1 ELSE 0 END), 0)::BIGINT AS p10_25",
        "COALESCE(SUM(CASE WHEN nodata_frac >= 0.25 AND nodata_frac < 0.50 THEN 1 ELSE 0 END), 0)::BIGINT AS p25_50",
        "COALESCE(SUM(CASE WHEN nodata_frac >= 0.50 THEN 1 ELSE 0 END), 0)::BIGINT AS ge_50",
    ]
    nodata_row = _fetch(
        f"SELECT {', '.join(nodata_exprs)} {source} {where}",
        source_params + params,
    )[0]
    by_role_rows = _fetch(
        f"""
        SELECT {fold}_role AS role, COUNT(*)::BIGINT AS tiles
        {source} {where_roles}
        GROUP BY 1
        """,
        source_params + params_roles,
    )
    by_role = {role_name: 0 for role_name in ROLES}
    for row in by_role_rows:
        by_role[row["role"]] = row["tiles"]

    return {
        "total": totals["total"],
        "labelled": totals["labelled"],
        "background_only": totals["total"] - totals["labelled"],
        "background_m2": wide["background_m2"],
        "by_role": by_role,
        "species": [
            {
                "stats_name": spec["stats_name"],
                "name": spec["name"],
                "color": spec["color"],
                "tiles": wide[f"{spec['stats_name']}_tiles"],
                "m2": wide[f"{spec['stats_name']}_m2"],
                "phases": [
                    {
                        "phase": phase,
                        "label": PHASE_LABELS[phase],
                        "m2": wide[f"{phase}_{spec['stats_name']}_m2"],
                    }
                    for phase in PHASES
                ],
            }
            for spec in SPECIES
        ],
        "by_site": by_site,
        "by_month": by_month,
        "nodata": [{"id": key, "label": label, "tiles": nodata_row[key]} for key, label in NODATA_BINS],
    }


MOSAICS_DIR = CACHE / "mosaics"
OVERVIEW_PX = 40
SHEET_TILES = 8
SHEET_PAD = 16
_overview_lock = threading.Lock()
_layout_lock = threading.Lock()
_layout_cache = {}
_sheet_locks_guard = threading.Lock()
_sheet_locks = {}


def _flight_key(site, flight):
    if not site or not flight or "/" in site or "/" in flight or ".." in site or ".." in flight:
        raise HTTPException(status_code=400, detail="invalid site or flight")
    return site, flight


def _place_flight(site, flight, fold):
    """One flight, every folder, on the grid it was cut from."""
    _require_catalog()
    fold = _fold(fold)
    site, flight = _flight_key(site, flight)
    if not GRID_PARQUET.is_file():
        raise HTTPException(status_code=503, detail="Grid missing. Run explorer/build_grid.py.")
    rows = _fetch(
        f"""
        SELECT t.tile_id, t.labelled, t.split, t.acq_date, t.{fold}_role AS cv_role,
               g.origin_x, g.origin_y, g.gsd_m
        FROM read_parquet(?) t
        JOIN read_parquet(?) g USING (tile_id)
        WHERE t.site_name = ? AND t.flight = ?
        """,
        [str(TILES_PARQUET), str(GRID_PARQUET), site, flight],
    )
    if not rows:
        raise HTTPException(status_code=404, detail="No tiles for that flight")
    step = 2048.0 * sorted(row["gsd_m"] for row in rows)[len(rows) // 2]
    min_x = min(row["origin_x"] for row in rows)
    max_y = max(row["origin_y"] for row in rows)
    placed = []
    for row in rows:
        placed.append({
            "tile_id": row["tile_id"],
            "col": int(round((row["origin_x"] - min_x) / step)),
            "row": int(round((max_y - row["origin_y"]) / step)),
            "labelled": row["labelled"],
            "split": row["split"],
            "cv_role": row["cv_role"],
            "thumb": f"/thumbs/{row['tile_id']}.webp",
            "mask": f"/masks/{row['tile_id']}.webp",
        })
    return {
        "site": site,
        "flight": flight,
        "acq_date": rows[0]["acq_date"],
        "ncols": max(item["col"] for item in placed) + 1,
        "nrows": max(item["row"] for item in placed) + 1,
        "sheet": SHEET_TILES,
        "thumb_px": THUMB_PX,
        "sheet_pad": SHEET_PAD,
        "tiles": placed,
    }


def _overview_paths(site, flight):
    return (
        MOSAICS_DIR / f"{site}_{flight}.webp",
        MOSAICS_DIR / f"{site}_{flight}_mask.webp",
    )


def _ensure_overview(site, flight):
    rgb_path, mask_path = _overview_paths(site, flight)
    with _overview_lock:
        if rgb_path.is_file() and mask_path.is_file():
            return rgb_path, mask_path
        placed = _place_flight(site, flight, "cv3")
        MOSAICS_DIR.mkdir(parents=True, exist_ok=True)
        width = placed["ncols"] * OVERVIEW_PX
        height = placed["nrows"] * OVERVIEW_PX
        rgb = Image.new("RGB", (width, height), EMPTY_RGB)
        mask = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        for tile in placed["tiles"]:
            box = (tile["col"] * OVERVIEW_PX, tile["row"] * OVERVIEW_PX)
            thumb = THUMBS_DIR / f"{tile['tile_id']}.webp"
            if thumb.is_file():
                with Image.open(thumb) as im:
                    rgb.paste(im.convert("RGB").resize((OVERVIEW_PX, OVERVIEW_PX), Image.Resampling.BOX), box)
            mask_file = MASKS_DIR / f"{tile['tile_id']}.webp"
            if mask_file.is_file():
                with Image.open(mask_file) as im:
                    chip = im.convert("RGBA").resize((OVERVIEW_PX, OVERVIEW_PX), Image.Resampling.NEAREST)
                    mask.paste(chip, box, chip)
        rgb.save(rgb_path, "WEBP", quality=75)
        mask.save(mask_path, "WEBP", quality=75)
        return rgb_path, mask_path


@app.get("/api/mosaic")
def mosaic(site: str, flight: str, fold: str = "cv3"):
    """Tiles of one flight, placed on the orthophoto grid they were cut from."""
    return _place_flight(site, flight, fold)


@app.get("/api/mosaic/{site}/{flight}/overview.webp")
def mosaic_overview(site: str, flight: str):
    rgb_path, _mask_path = _ensure_overview(site, flight)
    return FileResponse(rgb_path, media_type="image/webp")


@app.get("/api/mosaic/{site}/{flight}/overview-mask.webp")
def mosaic_overview_mask(site: str, flight: str):
    _rgb_path, mask_path = _ensure_overview(site, flight)
    return FileResponse(mask_path, media_type="image/webp")


def _flight_layout(site, flight):
    key = (site, flight)
    with _layout_lock:
        hit = _layout_cache.get(key)
    if hit is not None:
        return hit
    placed = _place_flight(site, flight, "cv3")
    with _layout_lock:
        _layout_cache[key] = placed
    return placed


def _sheet_lock(key):
    with _sheet_locks_guard:
        lock = _sheet_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _sheet_locks[key] = lock
        return lock


def _fit_chip(path, resample):
    with Image.open(path) as im:
        chip = im.convert("RGBA")
        if chip.size != (THUMB_PX, THUMB_PX):
            chip = chip.resize((THUMB_PX, THUMB_PX), resample)
        return chip


def _ensure_sheet(site, flight, cc, cr):
    """One WebP for an 8×8 block: photo on top, class mask below, with a pad between them."""
    site, flight = _flight_key(site, flight)
    if cc < 0 or cr < 0:
        raise HTTPException(status_code=404, detail="sheet out of range")
    path = MOSAICS_DIR / "sheets" / f"{site}_{flight}_{cc}_{cr}.webp"
    with _sheet_lock((site, flight, cc, cr)):
        if path.is_file():
            return path
        placed = _flight_layout(site, flight)
        col0 = cc * SHEET_TILES
        row0 = cr * SHEET_TILES
        if col0 >= placed["ncols"] or row0 >= placed["nrows"]:
            raise HTTPException(status_code=404, detail="sheet out of range")
        cols = min(SHEET_TILES, placed["ncols"] - col0)
        rows = min(SHEET_TILES, placed["nrows"] - row0)
        tiles = [
            tile
            for tile in placed["tiles"]
            if col0 <= tile["col"] < col0 + cols and row0 <= tile["row"] < row0 + rows
        ]
        if not tiles:
            raise HTTPException(status_code=404, detail="empty sheet")
        photo_h = rows * THUMB_PX
        image = Image.new("RGBA", (cols * THUMB_PX, photo_h * 2 + SHEET_PAD), (0, 0, 0, 0))
        image.paste(Image.new("RGBA", (cols * THUMB_PX, photo_h), (*EMPTY_RGB, 255)), (0, 0))
        for tile in tiles:
            box_x = (tile["col"] - col0) * THUMB_PX
            box_y = (tile["row"] - row0) * THUMB_PX
            thumb = THUMBS_DIR / f"{tile['tile_id']}.webp"
            if thumb.is_file():
                image.paste(_fit_chip(thumb, Image.Resampling.BOX), (box_x, box_y))
            mask = MASKS_DIR / f"{tile['tile_id']}.webp"
            if mask.is_file():
                chip = _fit_chip(mask, Image.Resampling.NEAREST)
                image.paste(chip, (box_x, box_y + photo_h + SHEET_PAD), chip)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        image.save(tmp, "WEBP", quality=80, method=0)
        tmp.replace(path)
        return path


@app.get("/api/mosaic/{site}/{flight}/sheet/{cc}/{cr}/image.webp")
def mosaic_sheet(site: str, flight: str, cc: int, cr: int):
    path = _ensure_sheet(site, flight, cc, cr)
    return FileResponse(
        path,
        media_type="image/webp",
        headers={"Cache-Control": "public, max-age=86400"},
    )
