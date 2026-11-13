"""Which tiles a novel-species run trains on, validates on, and tests on.

One species is treated as new. Its labels stay on the chosen flights only.
Every other flight of a site that contains the species is left out of training.
Those flights are the test when ``test`` is ``flights``. They are dropped when
``test`` is ``fold``, and the split's own test set is used instead. Flights
named in ``exclude`` are left out of training, validation, and test.

No torch and no pandas, so the explorer can import this too.
"""

import csv
import os
from collections import defaultdict

SPECIES = (
    "A_altissima",
    "R_typhina",
    "B_davidii",
    "R_japonica",
    "B_orientalis",
    "S_inaequidens",
)
LABELS = {
    "A_altissima": "A. altissima",
    "R_typhina": "R. typhina",
    "B_davidii": "B. davidii",
    "R_japonica": "R. japonica",
    "B_orientalis": "B. orientalis",
    "S_inaequidens": "S. inaequidens",
}
MIN_TRAIN_TILES = 10
ROLES = ("novel_train", "replay", "val", "test", "same_site", "dropped", "excluded")


def site_group(name):
    """``Rueti_1_1`` -> ``Rueti_1``. A name that does not end in a number stays."""
    head, sep, tail = str(name).rpartition("_")
    if sep and tail.isdigit():
        return head
    return str(name)


def _px(value):
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def novel_site_groups(rows, species):
    """Site groups that have at least one labelled pixel of ``species``."""
    groups = set()
    for row in rows:
        if _px(row.get("px")) > 0:
            groups.add(site_group(row.get("site")))
    return groups


def novel_site_groups_from_csv(path, species):
    """Same as :func:`novel_site_groups`, reading ``stats_imagewise.csv``."""
    groups = set()
    column = f"{species}_px"
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if _px(row.get(column)) > 0:
                groups.add(site_group(row.get("site")))
    return groups


PLACE_ROLES = ("novel_train", "replay", "val", "test", "same_site", "excluded")


def _place_map(raw):
    """``{flight folder: role}`` from a dict or ``folder:role`` strings."""
    if raw is None:
        return {}
    if isinstance(raw, str):
        raw = [] if raw.strip() in ("", "null", "None", "[]") else [raw]
    if hasattr(raw, "items") and not isinstance(raw, (str, list, tuple)):
        return {str(key): str(value) for key, value in raw.items() if str(key)}
    out = {}
    try:
        items = list(raw)
    except TypeError:
        items = [raw]
    for item in items:
        text = str(item).strip()
        if not text or text in ("null", "None") or ":" not in text:
            continue
        folder, role = text.split(":", 1)
        folder, role = folder.strip(), role.strip()
        if folder:
            out[folder] = role
    return out


def keep_training_tile(flight, flights, novel_groups, keep=None):
    """True when a train or val tile is allowed to stay.

    The chosen flights stay, labels and all. Any other flight of a site that
    has the species is dropped. It is not turned into background. ``keep``
    names flights the plan moved into training or validation.
    """
    flight = str(flight)
    if keep and flight in keep:
        return True
    if flight in set(flights or []):
        return True
    return site_group(flight) not in novel_groups


def _is_val(row, test_mode):
    if test_mode == "fold":
        return str(row.get("fold_role") or "") == "val"
    return str(row.get("split") or "") == "val"


def _as_names(raw):
    """Flight-folder names from a config value. Empty and null become ``[]``."""
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        return [] if text in ("", "null", "None", "[]") else [text]
    try:
        names = [str(item).strip() for item in raw]
    except TypeError:
        text = str(raw).strip()
        return [] if text in ("", "null", "None", "[]") else [text]
    return [name for name in names if name not in ("", "null", "None")]


def assign_roles(rows, species, flights, test_mode="flights", fold_train_only=True, exclude=None, place=None):
    """Role per tile id.

    ``rows`` are dicts with ``tile_id``, ``site`` (flight folder), ``split``
    (``train``, ``val``, or ``test`` on disk), ``px`` for ``species``, and
    ``fold_role`` when ``test_mode`` is ``fold``.

    An empty ``flights`` list is the five-species base: every site of the
    species is left out, so one base serves every later training flight.

    ``exclude`` names flights that are not used at all. ``place`` moves a
    flight to a stage (``replay``, ``novel_train``, ``val``, ``test``,
    ``same_site``, or ``excluded``). Moved tiles are not relabelled as background.
    """
    if species not in SPECIES:
        raise ValueError(f"unknown species {species}")
    if test_mode not in ("flights", "fold"):
        raise ValueError(f"unknown novel test mode {test_mode}")
    chosen = {str(flight) for flight in (flights or [])}
    forced = _place_map(place)
    for name in exclude or []:
        forced.setdefault(str(name), "excluded")
    groups = novel_site_groups(rows, species)
    train_groups = {site_group(flight) for flight in chosen}
    roles = {}
    for row in rows:
        tile_id = row.get("tile_id")
        flight = str(row.get("site"))
        if flight in forced:
            roles[tile_id] = forced[flight]
            continue
        group = site_group(flight)
        fold_role = str(row.get("fold_role") or "")
        if flight in chosen:
            if _is_val(row, test_mode):
                role = "val"
            elif test_mode == "fold" and fold_role not in ("train", ""):
                role = "dropped" if fold_role != "test" else "test"
            else:
                role = "novel_train"
        elif group in train_groups:
            role = "same_site"
        elif group in groups:
            if test_mode == "flights":
                role = "test"
            elif fold_role == "test":
                role = "test"
            else:
                role = "dropped"
        elif test_mode == "flights":
            role = "val" if str(row.get("split") or "") == "val" else "replay"
        elif fold_role == "val":
            role = "val"
        elif fold_role == "train":
            role = "replay"
        elif fold_role == "test":
            role = "test"
        else:
            role = "dropped"
        if test_mode == "fold" and fold_train_only and role == "novel_train" and fold_role not in ("train", ""):
            role = "dropped"
        roles[tile_id] = role
    return roles


def image_dirs_for(rows, roles, keep_test=None):
    """``{train, val, test}`` directory lists, relative to the repo root.

    ``keep_test`` (the config's existing test dirs) is used for a base run,
    which does not rewrite the test split.
    """
    by_role = {"train": [], "val": [], "test": []}
    seen = set()
    for row in rows:
        role = roles.get(row.get("tile_id"))
        if role == "novel_train" or role == "replay":
            bucket = "train"
        elif role == "val":
            bucket = "val"
        elif role in ("test", "same_site"):
            bucket = "test"
        else:
            continue
        year = row.get("year")
        flight = row.get("site")
        split = row.get("split")
        if not year or not flight or not split:
            continue
        folder = f"../data/Neophytes/{year}/{flight}/{split}"
        if (bucket, folder) in seen:
            continue
        seen.add((bucket, folder))
        by_role[bucket].append(folder)
    if keep_test is not None:
        by_role["test"] = list(keep_test)
    return by_role


def _settings(novel):
    """``(species or None, flights, test mode, stage, excluded flights, placements)``."""
    if novel is None:
        return None, [], "flights", "joint", [], {}
    get = novel.get if hasattr(novel, "get") else None
    if get is None:
        return None, [], "flights", "joint", [], {}
    species = novel["species"] if "species" in novel else None
    if species is None or str(species) in ("", "null", "None"):
        return None, [], "flights", "joint", [], {}
    species = str(species)
    flights = _as_names(novel["flights"] if "flights" in novel else [])
    test_mode = str(novel["test"]) if "test" in novel and novel["test"] else "flights"
    stage = str(novel["stage"]) if "stage" in novel and novel["stage"] else "joint"
    exclude = _as_names(novel["exclude"] if "exclude" in novel else [])
    place = _place_map(novel["place"] if "place" in novel else None)
    return species, flights, test_mode, stage, exclude, place


def rows_from_csv(path, species, fold=None):
    """Tile rows for :func:`assign_roles` from ``stats_imagewise.csv``."""
    column = f"{species}_px"
    role_col = f"{fold}_role" if fold else None
    rows = []
    with open(path, newline="") as handle:
        for i, row in enumerate(csv.DictReader(handle)):
            path_text = row.get("img_path") or ""
            parts = os.path.normpath(path_text).split(os.sep)
            # .../YEAR/FLIGHT/SPLIT/images/file
            split = parts[-3] if len(parts) >= 3 else row.get("split")
            year = parts[-5] if len(parts) >= 5 else row.get("year")
            rows.append({
                "tile_id": row.get("img_path") or str(i),
                "site": row.get("site"),
                "split": split,
                "year": year,
                "px": _px(row.get(column)),
                "fold_role": row.get(role_col) if role_col else "",
            })
    return rows


def _flight_of_dir(folder):
    """``../data/Neophytes/2024/Aarau_1_1/train`` -> ``Aarau_1_1``."""
    parts = str(folder).replace("\\", "/").rstrip("/").split("/")
    return parts[-2] if len(parts) >= 2 else ""


def _drop_dirs(folders, exclude):
    """Directory list with the excluded flight folders removed."""
    if not exclude:
        return list(folders)
    return [folder for folder in folders if _flight_of_dir(folder) not in exclude]


def _require_splits(dirs):
    if not dirs["train"] or not dirs["val"] or not dirs["test"]:
        raise RuntimeError(
            "novel split produced an empty train, val, or test set "
            f"(train {len(dirs['train'])}, val {len(dirs['val'])}, test {len(dirs['test'])})"
        )


def apply_novel_image_dirs(cfg):
    """Rewrite ``cfg.data.image_dir`` for ``test: flights``.

    Fold mode keeps its own split, then drops any flight named in ``exclude``.
    A base stage (no chosen flight) rewrites train and val only, and keeps the
    config's test dirs so the data module still has a test tile. Excluded
    flights are removed from that test list too.
    """
    species, flights, test_mode, _stage, exclude, place = _settings(
        cfg.data.novel if "novel" in cfg.data else None)
    if not species:
        return False
    from omegaconf import open_dict

    forced = dict(place)
    for name in exclude:
        forced.setdefault(name, "excluded")
    # A base has no chosen flight. A flight parked on the new species stays out
    # of that stage, otherwise the base would learn the species.
    if not flights:
        forced = {folder: ("excluded" if role == "novel_train" else role) for folder, role in forced.items()}
    excluded = {folder for folder, role in forced.items() if role == "excluded"}
    moved = any(role != "excluded" for role in forced.values())
    if test_mode != "flights" and not moved:
        if not excluded:
            return False
        dirs = {
            "train": _drop_dirs(cfg.data.image_dir.train, excluded),
            "val": _drop_dirs(cfg.data.image_dir.val, excluded),
            "test": _drop_dirs(cfg.data.image_dir.test, excluded),
        }
        _require_splits(dirs)
        with open_dict(cfg):
            cfg.data.image_dir = dirs
        return True

    stats = str(cfg.data.dataset_stats)
    fold = None
    if test_mode == "fold":
        import re
        label = str(cfg.data.data_name) if "data_name" in cfg.data else ""
        match = re.search(r"cv[1-5]", label)
        fold = match.group(0) if match else "cv3"
    rows = rows_from_csv(stats, species, fold)
    roles = assign_roles(rows, species, flights, test_mode=test_mode, place=forced)
    keep_test = None if flights or test_mode == "fold" else _drop_dirs(cfg.data.image_dir.test, excluded)
    dirs = image_dirs_for(rows, roles, keep_test=keep_test)
    dirs = {bucket: _drop_dirs(items, excluded) for bucket, items in dirs.items()}
    _require_splits(dirs)
    with open_dict(cfg):
        cfg.data.image_dir = dirs
    return True


def flight_summary(rows, roles):
    """One record per flight folder, with tile counts per role."""
    buckets = defaultdict(lambda: {"tiles": 0, "novel_tiles": 0, "roles": defaultdict(int),
                                    "site_name": "", "flight": "", "date": None, "year": None})
    for row in rows:
        flight = str(row.get("site"))
        slot = buckets[flight]
        role = roles.get(row.get("tile_id"), "dropped")
        slot["tiles"] += 1
        slot["roles"][role] += 1
        if _px(row.get("px")) > 0:
            slot["novel_tiles"] += 1
        if row.get("site_name"):
            slot["site_name"] = str(row.get("site_name"))
        if row.get("flight_no") is not None and row.get("flight_no") != "":
            slot["flight"] = str(row.get("flight_no"))
        if row.get("date") and (slot["date"] is None or str(row["date"]) < slot["date"]):
            slot["date"] = str(row["date"])[:10]
        if row.get("year") is not None:
            slot["year"] = row.get("year")
    out = []
    for flight, slot in buckets.items():
        role = "dropped"
        for candidate in ("excluded", "novel_train", "same_site", "test", "replay", "val"):
            if slot["roles"].get(candidate):
                role = candidate
                break
        out.append({
            "folder": flight,
            "site_name": slot["site_name"] or site_group(flight),
            "flight": slot["flight"] or flight.rsplit("_", 1)[-1],
            "date": slot["date"],
            "year": slot["year"],
            "tiles": slot["tiles"],
            "novel_tiles": slot["novel_tiles"],
            "role": role,
            "role_tiles": dict(slot["roles"]),
        })
    out.sort(key=lambda item: item["folder"])
    return out
