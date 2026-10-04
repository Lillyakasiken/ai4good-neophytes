"""Zero-shot Ultralytics masks for one tile.

The overlay is encoded into the API response. Nothing in this module is
written next to the tile. Checkpoints download into ``cache/weights`` on
first use. Ultralytics is imported only when a prediction actually runs.
"""

import base64
import threading
from pathlib import Path

import cv2
import numpy as np

from common import CACHE, EMPTY_RGB, load_names, species_columns
from vision import EDGES

WEIGHTS = CACHE / "weights"
MAX_TEXTS = 8
MAX_POINTS = 16
MAX_BOXES = 8
CONF_MIN = 0.05
CONF_MAX = 0.95
DEFAULT_CONF = 0.25
SAM_COLOR = "#1b3f36"
EXTRA_COLORS = ("#1b3f36", "#b42318", "#1d4e89", "#1b7a45", "#8a5a00", "#5c4d7a")

# Common names a text encoder is more likely to know than the binomials.
_PROMPTS = {
    "A_altissima": "tree of heaven",
    "R_typhina": "staghorn sumac",
    "B_davidii": "butterfly bush",
    "R_japonica": "Japanese knotweed",
    "B_orientalis": "warty cabbage",
    "S_inaequidens": "narrow-leaved ragwort",
}

MODELS = (
    {
        "id": "yoloe-26s",
        "label": "YOLOE-26s",
        "prompt": "text",
        "weight": "yoloe-26s-seg.pt",
        "text_encoder": "mobileclip2_b.ts",
    },
    {
        "id": "sam2.1-t",
        "label": "SAM 2.1 tiny",
        "prompt": "points",
        "weight": "sam2.1_t.pt",
        "text_encoder": None,
    },
)
BY_ID = {spec["id"]: spec for spec in MODELS}

_INSTALL = (
    "Ultralytics is not installed. From explorer/: "
    ".venv/bin/pip install -r requirements.txt"
)
_CLIP = (
    "Text prompts need the Ultralytics CLIP fork. From explorer/: "
    ".venv/bin/pip uninstall -y clip && "
    ".venv/bin/pip install git+https://github.com/ultralytics/CLIP.git"
)

_lock = threading.Lock()
_models = {}
_prompts = {}
_download = None
_presets = None


class SegmentUnavailable(RuntimeError):
    """Ultralytics or a checkpoint is not available yet."""


class SegmentFailed(RuntimeError):
    """The model ran and did not return a mask."""


def catalog():
    """Whitelist the Segment tab renders. Weights stay on the server."""
    return {
        "edges": list(EDGES),
        "default_model": "yoloe-26s",
        "default_conf": DEFAULT_CONF,
        "conf_min": CONF_MIN,
        "conf_max": CONF_MAX,
        "conf_step": 0.05,
        "max_texts": MAX_TEXTS,
        "max_points": MAX_POINTS,
        "max_boxes": MAX_BOXES,
        "models": [
            {"id": spec["id"], "label": spec["label"], "prompt": spec["prompt"]}
            for spec in MODELS
        ],
        "presets": _preset_rows(),
        "sam_color": SAM_COLOR,
    }


def run(rgb, alpha, model, max_edge, conf, texts, points, boxes=None):
    """Paint empty margin, predict, and return an RGBA overlay."""
    spec = _spec(model)
    if max_edge not in EDGES:
        raise ValueError("max_edge must be 256, 512, or 1024")
    conf = _conf(conf)
    texts = _texts(texts)
    points = _points(points)
    boxes = _boxes(boxes or [])
    if spec["prompt"] == "text" and not texts:
        raise ValueError("choose at least one phrase")
    if spec["prompt"] == "points":
        included = any(point["label"] == 1 for point in points) or any(box["label"] == 1 for box in boxes)
        if not included:
            raise ValueError("add an include point or an include region")
    with _lock:
        try:
            return _predict(rgb, alpha, spec, conf, texts, points, boxes)
        except (SegmentUnavailable, SegmentFailed, ValueError):
            raise
        except Exception as exc:
            raise SegmentFailed(_fail_text(exc)) from exc


def _preset_rows():
    global _presets
    if _presets is not None:
        return _presets
    classes, display_order, _folds = load_names()
    rows = []
    for spec in species_columns(classes, display_order):
        prompt = _PROMPTS.get(spec["stats_name"])
        if not prompt:
            continue
        rows.append({
            "stats_name": spec["stats_name"],
            "name": spec["name"],
            "prompt": prompt,
            "color": spec["color"],
        })
    _presets = rows
    return rows


def _spec(model):
    if not isinstance(model, str) or model not in BY_ID:
        names = ", ".join(spec["id"] for spec in MODELS)
        raise ValueError(f"model must be one of {names}")
    return BY_ID[model]


def _conf(value):
    if isinstance(value, bool) or isinstance(value, str):
        raise ValueError("conf must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("conf must be a number") from exc
    if not np.isfinite(number):
        raise ValueError("conf must be a number")
    return min(CONF_MAX, max(CONF_MIN, number))


def _texts(values):
    if not isinstance(values, list):
        raise ValueError("texts must be a list")
    cleaned = []
    seen = set()
    for raw in values:
        if not isinstance(raw, str):
            raise ValueError("each phrase must be text")
        text = " ".join(raw.split())
        if not text:
            continue
        if len(text) > 80:
            raise ValueError("a phrase must be at most 80 characters")
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
    if len(cleaned) > MAX_TEXTS:
        raise ValueError(f"at most {MAX_TEXTS} phrases")
    return cleaned


def _points(values):
    if not isinstance(values, list):
        raise ValueError("points must be a list")
    if len(values) > MAX_POINTS:
        raise ValueError(f"at most {MAX_POINTS} points")
    cleaned = []
    for raw in values:
        if not isinstance(raw, dict):
            raise ValueError("each point must be an object")
        try:
            x = float(raw.get("x"))
            y = float(raw.get("y"))
        except (TypeError, ValueError) as exc:
            raise ValueError("point x and y must be numbers") from exc
        if not np.isfinite(x) or not np.isfinite(y) or x < 0 or x > 1 or y < 0 or y > 1:
            raise ValueError("point x and y must be between 0 and 1")
        label = raw.get("label", 1)
        if isinstance(label, bool) or label not in (0, 1):
            raise ValueError("point label must be 1 or 0")
        cleaned.append({"x": x, "y": y, "label": int(label)})
    return cleaned


def _boxes(values):
    if not isinstance(values, list):
        raise ValueError("boxes must be a list")
    if len(values) > MAX_BOXES:
        raise ValueError(f"at most {MAX_BOXES} regions")
    cleaned = []
    for raw in values:
        if not isinstance(raw, dict):
            raise ValueError("each region must be an object")
        try:
            coords = [float(raw.get(key)) for key in ("x1", "y1", "x2", "y2")]
        except (TypeError, ValueError) as exc:
            raise ValueError("region edges must be numbers") from exc
        if any(not np.isfinite(value) or value < 0 or value > 1 for value in coords):
            raise ValueError("region edges must be between 0 and 1")
        x1, x2 = sorted((coords[0], coords[2]))
        y1, y2 = sorted((coords[1], coords[3]))
        if (x2 - x1) < 0.01 or (y2 - y1) < 0.01:
            raise ValueError("a region must cover part of the tile")
        label = raw.get("label", 1)
        if isinstance(label, bool) or label not in (0, 1):
            raise ValueError("region label must be 1 or 0")
        cleaned.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "label": int(label)})
    return cleaned


def _bootstrap():
    """Import Ultralytics and aim its weight lookup at the explorer cache.

    ``settings.update`` would rewrite the user's Ultralytics settings file.
    The in-memory assignment is enough for this process.
    """
    global _download
    if _download is not None:
        return _download
    try:
        import ultralytics.utils as utils
        from ultralytics.utils.downloads import attempt_download_asset
    except ImportError as exc:
        raise SegmentUnavailable(_INSTALL) from exc
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    dest = str(WEIGHTS)
    if utils.SETTINGS.get("weights_dir") != dest:
        dict.update(utils.SETTINGS, {"weights_dir": dest})
    utils.WEIGHTS_DIR = Path(dest)
    _download = attempt_download_asset
    return _download


def _ensure(name, download):
    dest = WEIGHTS / name
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    try:
        download(dest)
    except Exception as exc:
        raise SegmentUnavailable(
            f"Could not download {name}. The first run needs network access."
        ) from exc
    if not dest.is_file() or dest.stat().st_size == 0:
        raise SegmentUnavailable(
            f"Could not download {name}. The first run needs network access."
        )
    return dest


def _load(spec):
    cached = _models.get(spec["id"])
    if cached is not None:
        return cached
    download = _bootstrap()
    weight = _ensure(spec["weight"], download)
    if spec["text_encoder"]:
        _ensure(spec["text_encoder"], download)
    try:
        if spec["prompt"] == "text":
            from ultralytics import YOLOE
            model = YOLOE(str(weight))
        else:
            from ultralytics import SAM
            model = SAM(str(weight))
    except ImportError as exc:
        raise SegmentUnavailable(_INSTALL) from exc
    _models[spec["id"]] = model
    return model


def _bind_texts(model, model_id, texts):
    key = tuple(texts)
    if _prompts.get(model_id) == key:
        return
    model.set_classes(list(texts))
    _prompts[model_id] = key


def _predict(rgb, alpha, spec, conf, texts, points, boxes):
    image = np.array(rgb, copy=True)
    image[alpha == 0] = EMPTY_RGB
    height, width = image.shape[:2]
    # Ultralytics treats a numpy source as BGR.
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    model = _load(spec)
    if spec["prompt"] == "text":
        _bind_texts(model, spec["id"], texts)
        results = model.predict(bgr, conf=conf, verbose=False, save=False, retina_masks=True)
        color_of = _color_lookup(texts)
    else:
        prompts = _sam_kwargs(points, boxes, width, height)
        results = model.predict(bgr, conf=conf, verbose=False, save=False, **prompts)
        color_of = lambda _label, _index: SAM_COLOR
    result = results[0] if results else None
    found = [] if result is None else _read_masks(result, height, width, alpha, conf, color_of)
    if spec["prompt"] == "points":
        for index, item in enumerate(found, start=1):
            item["label"] = "region" if len(found) == 1 else f"region {index}"
            item["color"] = SAM_COLOR
    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    for item in reversed(found):
        red, green, blue = _hex_rgb(item["color"])
        overlay[item["layer"] > 0] = (red, green, blue, 255)
    instances = [
        {
            "label": item["label"],
            "confidence": item["confidence"],
            "share": item["share"],
            "color": item["color"],
        }
        for item in found
    ]
    return {
        "width": int(width),
        "height": int(height),
        "model": spec["id"],
        "overlay": _png_b64(overlay),
        "instances": instances,
    }


def _px(value, size):
    return min(size - 1, max(0.0, float(value) * size))


def _sam_kwargs(points, boxes, width, height):
    """Build the point and box prompts SAM's public predict call accepts.

    A box prompt is include-only. An exclude region is sent as background
    points inside the rectangle, which is the exclude input the model has.
    """
    xy = []
    labels = []
    for point in points:
        xy.append([_px(point["x"], width), _px(point["y"], height)])
        labels.append(point["label"])
    include = []
    for box in boxes:
        x1, x2 = _px(box["x1"], width), _px(box["x2"], width)
        y1, y2 = _px(box["y1"], height), _px(box["y2"], height)
        if box["label"] == 1:
            include.append([x1, y1, x2, y2])
            continue
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        for x, y in (
            (cx, cy),
            ((x1 + cx) / 2.0, cy),
            ((x2 + cx) / 2.0, cy),
            (cx, (y1 + cy) / 2.0),
            (cx, (y2 + cy) / 2.0),
        ):
            xy.append([x, y])
            labels.append(0)
    prompts = {}
    if include and xy:
        prompts["bboxes"] = include
        prompts["points"] = [xy for _ in include]
        prompts["labels"] = [labels for _ in include]
    elif include:
        prompts["bboxes"] = include
    elif xy:
        prompts["points"] = [xy]
        prompts["labels"] = [labels]
    return prompts


def _color_lookup(texts):
    preset = {row["prompt"].casefold(): row["color"] for row in _preset_rows()}
    assigned = {}
    extra = 0
    for text in texts:
        key = text.casefold()
        if key in preset:
            assigned[key] = preset[key]
        else:
            assigned[key] = EXTRA_COLORS[extra % len(EXTRA_COLORS)]
            extra += 1

    def color_of(label, index):
        return assigned.get(str(label).casefold(), EXTRA_COLORS[index % len(EXTRA_COLORS)])

    return color_of


def _as_float(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return float(value)


def _class_name(names, cls_id):
    if isinstance(names, dict):
        if cls_id in names:
            return str(names[cls_id])
        if str(cls_id) in names:
            return str(names[str(cls_id)])
        return str(cls_id)
    if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
        return str(names[cls_id])
    return str(cls_id)


def _read_masks(result, height, width, alpha, conf_min, color_of):
    masks = getattr(result, "masks", None)
    polys = [] if masks is None else list(getattr(masks, "xy", None) or [])
    boxes = getattr(result, "boxes", None)
    names = getattr(result, "names", {}) or {}
    valid = alpha != 0
    valid_count = int(np.count_nonzero(valid))
    found = []
    for index, poly in enumerate(polys):
        poly = np.asarray(poly, dtype=np.float32)
        if poly.ndim != 2 or poly.shape[0] < 3:
            continue
        score = None
        cls_id = index
        if boxes is not None and index < len(boxes):
            confs = getattr(boxes, "conf", None)
            clses = getattr(boxes, "cls", None)
            if confs is not None and index < len(confs):
                score = _as_float(confs[index])
            if clses is not None and index < len(clses):
                cls_id = int(_as_float(clses[index]))
        if score is not None and score < conf_min:
            continue
        layer = np.zeros((height, width), dtype=np.uint8)
        pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(layer, [pts], 255)
        layer[~valid] = 0
        count = int(np.count_nonzero(layer))
        if count == 0:
            continue
        label = _class_name(names, cls_id)
        found.append({
            "layer": layer,
            "label": label,
            "confidence": None if score is None else round(score, 3),
            "share": round(count / valid_count, 4) if valid_count else 0.0,
            "color": color_of(label, index),
            "score": -1.0 if score is None else score,
        })
    found.sort(key=lambda item: item["score"], reverse=True)
    return found


def _hex_rgb(value):
    hex_color = value.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def _png_b64(overlay):
    bgra = cv2.cvtColor(overlay, cv2.COLOR_RGBA2BGRA)
    ok, buf = cv2.imencode(".png", bgra, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    if not ok:
        raise SegmentFailed("could not encode the overlay")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _fail_text(exc):
    text = str(exc)
    lower = text.lower()
    if "simpletokenizer" in lower or "no module named 'clip'" in lower or 'no module named "clip"' in lower:
        return _CLIP
    line = text.splitlines()[-1].strip() if text else exc.__class__.__name__
    return f"segmentation failed: {line[:300]}"
