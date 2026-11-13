"""Build the few-shot plan the Runs page draws, from the tile catalog.

The roles come from utils.novel_split, the same function train.py uses, so the
diagram matches the run.
"""

import json
import sys
from pathlib import Path

import duckdb
import yaml

from common import TILES_PARQUET, load_names, species_columns
from runtime import LOG_DIR, REPO_ROOT

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.novel_split import (  # noqa: E402
    LABELS,
    MIN_TRAIN_TILES,
    SPECIES,
    assign_roles,
    flight_summary,
    site_group,
)

_CACHE = {"sig": None, "rows": None}


class PlanError(ValueError):
    """The few-shot form cannot build this plan."""


def _fold(data_name):
    import re
    match = re.search(r"cv[1-5]", str(data_name or ""))
    return match.group(0) if match else "cv3"


def _crop(data_name):
    text = str(data_name or "")
    for size in ("2048", "1024", "512"):
        if size in text:
            return int(size)
    return None


def _loss(model_name):
    folder = REPO_ROOT / "configs" / "model"
    base = yaml.safe_load((folder / "model.yaml").read_text()) or {}
    path = folder / f"{model_name}.yaml"
    doc = yaml.safe_load(path.read_text()) if path.is_file() else {}
    merged = dict(base)
    merged.update(doc or {})
    return str(merged.get("loss") or "Focal")


def _finetune(request=None):
    """The fine-tune block of configs/train.yaml, with the form's values over it."""
    doc = yaml.safe_load((REPO_ROOT / "configs" / "train.yaml").read_text()) or {}
    block = dict(doc.get("finetune") or {})
    if isinstance(request, dict):
        for key in ("head_from", "head_lr_mult", "novel_fraction"):
            if key in request and request[key] is not None:
                block[key] = request[key]
    head_from = str(block.get("head_from") or "mean")
    return {
        "head_from": head_from,
        "head_lr_mult": block.get("head_lr_mult"),
        "novel_fraction": block.get("novel_fraction"),
    }


def _rows():
    if not TILES_PARQUET.is_file():
        raise PlanError("tile catalog is missing; start the explorer once so it can build")
    sig = TILES_PARQUET.stat().st_mtime_ns
    if _CACHE["sig"] == sig and _CACHE["rows"] is not None:
        return _CACHE["rows"]
    px = [f"{name}_px" for name in SPECIES]
    con = duckdb.connect()
    try:
        con.execute(
            "SELECT tile_id, site, site_name, CAST(flight AS VARCHAR), split, year, "
            "CAST(acq_date AS VARCHAR), cv1_role, cv2_role, cv3_role, cv4_role, cv5_role, "
            + ", ".join(px)
            + " FROM read_parquet(?)",
            [str(TILES_PARQUET)],
        )
        raw = con.fetchall()
    finally:
        con.close()
    rows = []
    for item in raw:
        px_map = {name: int(item[12 + i] or 0) for i, name in enumerate(SPECIES)}
        rows.append({
            "tile_id": item[0],
            "site": item[1],
            "site_name": item[2],
            "flight_no": item[3],
            "split": item[4],
            "year": item[5],
            "date": (item[6] or "")[:10],
            "fold_roles": {"cv1": item[7], "cv2": item[8], "cv3": item[9], "cv4": item[10], "cv5": item[11]},
            "px_map": px_map,
        })
    _CACHE.update(sig=sig, rows=rows)
    return rows


def _view(rows, species, fold):
    """Rows shaped for assign_roles, one species, one fold."""
    out = []
    for row in rows:
        out.append({
            "tile_id": row["tile_id"],
            "site": row["site"],
            "site_name": row["site_name"],
            "flight_no": row["flight_no"],
            "split": row["split"],
            "year": row["year"],
            "date": row["date"],
            "px": row["px_map"].get(species, 0),
            "fold_role": row["fold_roles"].get(fold) or "",
            "px_map": row["px_map"],
        })
    return out


def _names(raw):
    """Flight folders from the form. A string, a list, or empty."""
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        return [] if text in ("", "null", "None") else [text]
    return [str(item).strip() for item in raw if str(item).strip() not in ("", "null", "None")]


def _place(raw):
    """Placements from the form. A dict, or ``folder:role`` strings."""
    roles = {"novel_train", "replay", "val", "test", "same_site", "excluded"}
    if raw is None:
        return {}
    if isinstance(raw, str):
        raw = [] if raw.strip() in ("", "null", "None") else [raw]
    items = raw.items() if hasattr(raw, "items") and not isinstance(raw, (str, list, tuple)) else None
    out = {}
    if items is not None:
        pairs = [(str(key), str(value)) for key, value in items]
    else:
        pairs = []
        for item in raw:
            text = str(item).strip()
            if ":" not in text:
                continue
            folder, role = text.split(":", 1)
            pairs.append((folder.strip(), role.strip()))
    for folder, role in pairs:
        if not folder:
            continue
        if role not in roles:
            raise PlanError(f"unknown stage {role}")
        out[folder] = role
    return out


def _candidates(rows, test_mode, exclude=(), place=None):
    """Flights with enough labelled tiles to be the one training flight."""
    skipped = set(exclude)
    forced = place or {}
    counts = {}
    train_ok = {}
    for row in rows:
        if row["px"] <= 0:
            continue
        flight = row["site"]
        if flight in skipped or forced.get(flight) in ("excluded", "replay", "val", "test", "same_site"):
            continue
        counts[flight] = counts.get(flight, 0) + 1
        if row["fold_role"] == "train":
            train_ok[flight] = True
    best = {}
    for flight, count in counts.items():
        if count < MIN_TRAIN_TILES:
            continue
        if test_mode == "fold" and not train_ok.get(flight):
            continue
        group = site_group(flight)
        if group not in best or count > best[group][1]:
            best[group] = (flight, count)
    return sorted(flight for flight, _count in best.values())


def _palette():
    classes, display_order, _folds = load_names()
    return [
        {"stats_name": spec["stats_name"], "name": spec["name"], "color": spec["color"]}
        for spec in species_columns(classes, display_order)
    ]


def _mix(rows):
    """Labelled tiles and pixels of each species, per flight folder."""
    totals = {}
    for row in rows:
        slot = totals.setdefault(row["site"], {})
        for name, px in row["px_map"].items():
            if px <= 0:
                continue
            cell = slot.setdefault(name, {"tiles": 0, "px": 0})
            cell["tiles"] += 1
            cell["px"] += int(px)
    return totals


def _slim(flight, mix, palette):
    raw = mix.get(flight["folder"], {})
    total_px = sum(cell["px"] for cell in raw.values())
    species = []
    for spec in palette:
        cell = raw.get(spec["stats_name"], {"tiles": 0, "px": 0})
        species.append({
            **spec,
            "tiles": cell["tiles"],
            "px": cell["px"],
            "share": round(cell["px"] / total_px, 4) if total_px else 0,
        })
    return {
        "folder": flight["folder"],
        "site_name": flight["site_name"],
        "flight": flight["flight"],
        "date": flight["date"],
        "tiles": flight["tiles"],
        "novel_tiles": flight["novel_tiles"],
        "role": flight["role"],
        "species": species,
    }


def build_plan(settings):
    """The diagram: flights by role, the settings card, and warnings."""
    species = str(settings.get("species") or "")
    if species not in SPECIES:
        raise PlanError("pick one of the six species")
    test_mode = settings.get("test") or "flights"
    if test_mode not in ("flights", "fold"):
        raise PlanError("test must be flights or fold")
    schedule = settings.get("schedule") or "two_stage"
    if schedule not in ("joint", "two_stage"):
        raise PlanError("schedule must be joint or two_stage")
    flight = str(settings.get("flight") or "")
    place = _place(settings.get("place"))
    for name in _names(settings.get("exclude")):
        place.setdefault(name, "excluded")
    exclude = [name for name, role in place.items() if role == "excluded"]
    fold = _fold(settings.get("data"))
    rows = _view(_rows(), species, fold)
    known = {row["site"] for row in rows}
    unknown = [name for name in place if name not in known]
    if unknown:
        raise PlanError(f"unknown flight {unknown[0]}")
    if any("," in name or " " in name or "=" in name or ":" in name for name in place):
        raise PlanError("a flight name cannot be moved")
    labelled_flights = {row["site"] for row in rows if row["px"] > 0}
    if test_mode == "flights" and len(labelled_flights) < 2:
        raise PlanError(f"{LABELS[species]} is on one flight only, so there is nothing to test")
    if flight and flight not in {row["site"] for row in rows}:
        raise PlanError(f"unknown flight {flight}")
    if flight and flight not in labelled_flights:
        raise PlanError(f"{flight} has no labelled {LABELS[species]} tiles")
    if test_mode == "fold" and flight:
        on_train = any(row["px"] > 0 and row["fold_role"] == "train" and row["site"] == flight for row in rows)
        if not on_train:
            raise PlanError(f"{flight} is not in the {fold} training split. Use other flights, or pick a training flight.")

    roles = assign_roles(rows, species, [flight] if flight else [], test_mode, exclude=exclude, place=place)
    summary = flight_summary(rows, roles)
    palette = _palette()
    mix = _mix(rows)
    columns = {role: [] for role in ("novel_train", "replay", "val", "test", "same_site", "dropped", "excluded")}
    for item in summary:
        if item["role"] in columns:
            columns[item["role"]].append(_slim(item, mix, palette))

    warnings = []
    chosen = next((item for item in summary if item["folder"] == flight), None)
    if chosen and chosen["role"] == "novel_train" and chosen["novel_tiles"] < MIN_TRAIN_TILES:
        warnings.append(
            f"{flight} has {chosen['novel_tiles']} labelled tiles. "
            f"Under {MIN_TRAIN_TILES} is a weak training flight."
        )
    empty_test = [item["folder"] for item in columns["test"] if item["novel_tiles"] == 0]
    if empty_test:
        warnings.append(
            f"{len(empty_test)} test flights have no labelled {LABELS[species]} tiles. "
            "They are included because the site has the species."
        )
    for other in SPECIES:
        if other == species:
            continue
        kept = 0
        total = 0
        for row in rows:
            if row["px_map"].get(other, 0) <= 0:
                continue
            total += 1
            role = roles.get(row["tile_id"])
            if role in ("novel_train", "replay", "val"):
                kept += 1
        if total and kept == 0:
            warnings.append(f"{LABELS[other]} has no training tiles left. Its sites were all held out.")
    if flight:
        where = "other-flight test" if test_mode == "flights" else f"{fold} test"
        for name in SPECIES:
            tiles = sum(
                1 for row in rows
                if roles.get(row["tile_id"]) == "test" and row["px_map"].get(name, 0) > 0
            )
            if tiles == 0:
                warnings.append(
                    f"{LABELS[name]} has no labelled tiles on the {where}, so its test F1 will be empty."
                )
            elif tiles < MIN_TRAIN_TILES:
                warnings.append(
                    f"{LABELS[name]} has {tiles} labelled tiles on the {where}. "
                    f"Under {MIN_TRAIN_TILES} is too few for a test F1."
                )
        if not columns["novel_train"]:
            warnings.append("No flight is on the new species. Move one there before running.")
        if exclude or place:
            used = {"train": 0, "val": 0, "test": 0}
            for role in roles.values():
                if role in ("novel_train", "replay"):
                    used["train"] += 1
                elif role == "val":
                    used["val"] += 1
                elif role in ("test", "same_site"):
                    used["test"] += 1
            if used["train"] == 0:
                raise PlanError("leaving those flights out removes the training set")
            if used["val"] == 0:
                raise PlanError("leaving those flights out removes the validation set")
            if used["test"] == 0:
                raise PlanError("leaving those flights out removes the test set")
            if schedule == "two_stage" and not columns["replay"]:
                raise PlanError("leaving those flights out removes the five-species base")

    labelled = {}
    for row in rows:
        if row["px"] <= 0:
            continue
        slot = labelled.setdefault(row["site"], {
            "folder": row["site"],
            "site_name": row["site_name"],
            "flight": str(row["flight_no"] or ""),
            "date": row["date"],
            "novel_tiles": 0,
            "fold_train": False,
        })
        slot["novel_tiles"] += 1
        if row["fold_role"] == "train":
            slot["fold_train"] = True

    candidates = _candidates(rows, test_mode, exclude, place)
    if settings.get("sweep") and not candidates:
        raise PlanError(f"no {LABELS[species]} flight has {MIN_TRAIN_TILES} labelled tiles")

    return {
        "species": species,
        "species_label": LABELS[species],
        "palette": palette,
        "flight": flight,
        "exclude": exclude,
        "place": place,
        "schedule": schedule,
        "test": test_mode,
        "fold": fold,
        "sweep": bool(settings.get("sweep")),
        "candidates": candidates,
        "settings": {
            "data": settings.get("data"),
            "model": settings.get("model"),
            "loss": _loss(settings.get("model")),
            "crop": _crop(settings.get("data")),
            "seed": settings.get("seed"),
            "epochs": settings.get("max_epochs"),
            "ft_epochs": settings.get("ft_epochs"),
            "lr": settings.get("lr"),
            "ft_lr": settings.get("ft_lr"),
            "batch_size_train": settings.get("batch_size_train"),
            "precision": settings.get("precision"),
            "check_val_every_n_epoch": settings.get("check_val_every_n_epoch"),
            "finetune": _finetune(settings.get("finetune")) if schedule == "two_stage" else None,
        },
        "columns": columns,
        "labelled": sorted(labelled.values(), key=lambda item: (-item["novel_tiles"], item["folder"])),
        "warnings": warnings,
        "counts": {role: len(items) for role, items in columns.items()},
    }


def flight_tile_roles(settings, folder):
    """Role of every tile on one flight. The map colours these."""
    species = str(settings.get("species") or "")
    if species not in SPECIES:
        raise PlanError("pick one of the six species")
    fold = _fold(settings.get("data"))
    flight = str(settings.get("flight") or "")
    place = _place(settings.get("place"))
    for name in _names(settings.get("exclude")):
        place.setdefault(name, "excluded")
    test_mode = settings.get("test") or "flights"
    rows = _view(_rows(), species, fold)
    roles = assign_roles(rows, species, [flight] if flight else [], test_mode, place=place)
    tiles = [
        {"tile_id": row["tile_id"], "role": roles.get(row["tile_id"], "dropped")}
        for row in rows if row["site"] == folder
    ]
    if not tiles:
        raise PlanError(f"no tiles for {folder}")
    return {"folder": folder, "tiles": tiles}


def base_runs(species):
    """Finished five-species bases for this species, newest name first."""
    if not LOG_DIR.is_dir() or species not in SPECIES:
        return []
    found = []
    for exp in LOG_DIR.iterdir():
        if not exp.is_dir() or "_base" not in exp.name or species not in exp.name:
            continue
        ckpt = exp / "checkpoints"
        if not ckpt.is_dir() or not any(ckpt.glob("*.ckpt")):
            continue
        found.append(exp.name)
    return sorted(found, reverse=True)[:30]


def matching_base(species, data, model, seed):
    """Newest finished base for this species, data config, model, and seed."""
    suffix = f"_{data}_{model}_s{seed}_nv{species}_base"
    for name in base_runs(species):
        if name.endswith(suffix):
            return name
    return ""


def resolve_base(species, data, model, seed, requested):
    """Base run to load. ``__new__`` trains one. Empty uses the newest match."""
    text = str(requested or "").strip()
    if text == "__new__":
        return ""
    if text:
        return text
    return matching_base(species, data, model, seed)


def read_progress(exp_name):
    path = LOG_DIR / exp_name / "progress.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def read_novel_metrics(exp_name):
    best = None
    best_mtime = -1.0
    exp = LOG_DIR / exp_name
    if not exp.is_dir():
        return None
    for path in exp.glob("test_*/quantitative/metrics_novel.json"):
        try:
            mtime = path.stat().st_mtime
            doc = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if mtime >= best_mtime:
            best_mtime = mtime
            best = doc
    return best
