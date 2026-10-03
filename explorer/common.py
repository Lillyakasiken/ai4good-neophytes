"""Shared paths and class metadata for the tile explorer.

Training code is not imported. The catalog and the API read the same YAML
configs the baseline uses, and resolve tile paths the way those configs do:
relative to the repo root.
"""

import hashlib
import os
import re
from pathlib import Path

import yaml

EXPLORER_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXPLORER_ROOT.parent
CONFIG_DIR = REPO_ROOT / "configs" / "data"
DATA_CSV = (REPO_ROOT / "../data/Neophytes/stats_imagewise.csv").resolve()

CACHE = EXPLORER_ROOT / "cache"
TILES_PARQUET = CACHE / "tiles.parquet"
META_PARQUET = CACHE / "raster_meta.parquet"
GRID_PARQUET = CACHE / "grid.parquet"
THUMBS_DIR = CACHE / "thumbs"
MASKS_DIR = CACHE / "masks"

FOLDS = ["cv1", "cv2", "cv3", "cv4", "cv5"]
ROLES = ("train", "val", "test", "unused")
PHASES = ("nonflower", "flower", "fruiting", "unknown")
PHASE_LABELS = {
    "nonflower": "non-flowering",
    "flower": "flowering",
    "fruiting": "fruiting",
    "unknown": "unknown",
}

# Mask values mapped to ignore_index by configs/data/data.yaml. They are drawn
# on the thumbnail so excluded pixels are visible, and they are not a species.
IGNORE_COLORS = {
    254: (115, 115, 115),
    255: (250, 204, 21),
}
EMPTY_RGB = (232, 232, 232)
THUMB_PX = 256

_DATE = re.compile(r"(?:^|[/_])(\d{4})_(\d{2})_(\d{2})_")


def norm_rel(path):
    """CSV / YAML path, with slashes normalised and '.' / '..' segments kept."""
    return os.path.normpath(str(path).strip().replace("\\", "/"))


def resolve_data_path(path):
    """Absolute path of a tile path stored relative to the repo root."""
    p = Path(path)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def tile_id(img_path):
    """Stable id from the relative image path. Used for thumbs and URLs."""
    return hashlib.sha1(norm_rel(img_path).encode()).hexdigest()[:16]


def split_key(path):
    """``year/site_folder/split`` for a tile or an ``image_dir`` entry.

    Both the stats CSV and the split YAMLs store paths like
    ``../data/Neophytes/2024/Aarau_1_1/train/...``. Matching on this suffix
    is the prefix match against those directories, without depending on the
    ``../data`` prefix.
    """
    parts = norm_rel(path).split("/")
    for i, part in enumerate(parts):
        if len(part) == 4 and part.isdigit() and i + 2 < len(parts):
            return f"{parts[i]}/{parts[i + 1]}/{parts[i + 2]}"
    return None


def parse_acq(path):
    """Acquisition date and month from a tile filename, or (None, None)."""
    m = _DATE.search(os.path.basename(str(path)))
    if not m:
        return None, None
    year, month, day = (int(m.group(i)) for i in range(1, 4))
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None, None
    return f"{year:04d}-{month:02d}-{day:02d}", month


def split_site_flight(folder, sites_longest_first):
    """Split ``Basel_2_6`` into site ``Basel_2`` and flight ``6``.

    Site names are the keys of ``cv_folds``. The longest matching site wins,
    so ``Basel_4_6`` is Basel_4 rather than a shorter prefix.
    """
    for site in sites_longest_first:
        if folder == site:
            return site, ""
        prefix = site + "_"
        if folder.startswith(prefix):
            return site, folder[len(prefix):]
    return folder, ""


def hex_to_rgb(value):
    h = value.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def load_names():
    """Classes, plot order, and held-out sites from the training config."""
    doc = yaml.safe_load((CONFIG_DIR / "neophytes_names_colors.yaml").read_text())
    classes = []
    for idx in sorted(doc["classes"], key=int):
        spec = doc["classes"][idx]
        classes.append({
            "id": int(idx),
            "name": spec["name"],
            "color": spec["color"],
            "rgb": hex_to_rgb(spec["color"]),
            "stats_name": spec["stats_name"],
        })
    display_order = [int(i) for i in doc["display_order"]]
    folds = {fold: list(sites) for fold, sites in doc["cv_folds"].items()}
    return classes, display_order, folds


def species_columns(classes, display_order):
    """Neophyte classes in the plot order (tallest first), background excluded."""
    by_id = {c["id"]: c for c in classes}
    return [by_id[i] for i in display_order]


def load_role_maps():
    """Per fold, map ``year/folder/split`` to train, val, or test.

    A directory listed in a fold's ``image_dir`` is exactly the set of tiles
    that ``NeophyteDataset`` walks for that role. Anything else is unused
    (held-out sites' val folders, and their on-disk test folders).
    """
    maps = {}
    for fold in FOLDS:
        doc = yaml.safe_load((CONFIG_DIR / f"split_{fold}.yaml").read_text())
        image_dir = doc.get("image_dir")
        if not image_dir:
            raise SystemExit(f"{fold}: split YAML has no image_dir")
        role_of = {}
        for role, dirs in image_dir.items():
            if role not in ("train", "val", "test"):
                raise SystemExit(f"{fold}: unexpected role {role}")
            for directory in dirs:
                key = split_key(directory)
                if key is None:
                    raise SystemExit(f"{fold}: cannot parse image dir {directory}")
                previous = role_of.get(key)
                if previous is not None and previous != role:
                    raise SystemExit(f"{fold}: {key} is both {previous} and {role}")
                role_of[key] = role
        maps[fold] = role_of
    return maps
