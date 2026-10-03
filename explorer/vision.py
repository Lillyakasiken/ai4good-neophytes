"""In-memory OpenCV previews for one tile.

The filtered image is encoded into the API response. Nothing in this module
is written to disk. A small LRU keeps the downsampled source array so that
dragging a slider does not reopen the GeoTIFF.
"""

import base64
import re
import threading
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError

from common import EMPTY_RGB

EDGES = (256, 512, 1024)
MAX_OPS = 12
_CACHE_MAX = 8
_DSM_NODATA = -32767.0
_TILE_SUFFIX = re.compile(r"(_\d{2}_\d{2}\.tiff?)$", re.IGNORECASE)

_cache_lock = threading.Lock()
_cache = OrderedDict()


def _num(key, label, default, lo, hi, step, integer=False, odd=False):
    spec = {
        "key": key,
        "label": label,
        "kind": "number",
        "min": lo,
        "max": hi,
        "step": step,
        "default": default,
    }
    if integer:
        spec["integer"] = True
    if odd:
        spec["odd"] = True
    return spec


def _choice(key, label, options, default):
    return {
        "key": key,
        "label": label,
        "kind": "select",
        "default": default,
        "options": [{"value": value, "label": text} for value, text in options],
    }


def _to_gray(image, mode):
    if mode == "gray":
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def _scale_fixed(value, lo, hi):
    out = (value - lo) / (hi - lo) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def _norm_u8(mag):
    peak = float(mag.max()) if mag.size else 0.0
    if peak <= 0:
        return np.zeros(mag.shape, dtype=np.uint8)
    return np.clip(mag / peak * 255.0, 0, 255).astype(np.uint8)


def _fit_ksize(ksize, image, minimum=1):
    limit = int(min(image.shape[:2]))
    if limit < minimum:
        raise ValueError("preview is too small for this filter")
    ksize = min(int(ksize), limit)
    if ksize % 2 == 0:
        ksize = ksize - 1 if ksize > minimum else ksize + 1
    return max(minimum, ksize)


def _kernel(ksize):
    ksize = int(ksize)
    return cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))


def _grayscale(image, mode):
    return _to_gray(image, mode), "gray"


def _brightness(image, mode, contrast, brightness):
    out = cv2.convertScaleAbs(image, alpha=float(contrast), beta=float(brightness))
    return out, mode


def _gamma(image, mode, gamma):
    table = np.array(
        [min(255, round((index / 255.0) ** (1.0 / float(gamma)) * 255.0)) for index in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image, table), mode


def _invert(image, mode):
    return cv2.bitwise_not(image), mode


def _equalize(image, mode):
    return cv2.equalizeHist(_to_gray(image, mode)), "gray"


def _clahe(image, mode, clip, grid):
    grid = max(1, int(grid))
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(grid, grid))
    if mode == "gray":
        return clahe.apply(image), "gray"
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lightness, a, b = cv2.split(lab)
    merged = cv2.merge((clahe.apply(lightness), a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB), "color"


def _channel(image, mode, channel):
    index = {"r": 0, "g": 1, "b": 2}[channel]
    return image[:, :, index].copy(), "gray"


def _hsv(image, mode, channel):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    index = {"h": 0, "s": 1, "v": 2}[channel]
    plane = hsv[:, :, index]
    if channel == "h":
        plane = np.clip(np.round(plane.astype(np.float32) * (255.0 / 179.0)), 0, 255).astype(np.uint8)
    return plane.copy(), "gray"


def _lab(image, mode, channel):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    index = {"l": 0, "a": 1, "b": 2}[channel]
    return lab[:, :, index].copy(), "gray"


def _chromaticity(image):
    rgb = image.astype(np.float32)
    total = rgb.sum(axis=2)
    total = np.where(total == 0, 1.0, total)
    return rgb[:, :, 0] / total, rgb[:, :, 1] / total, rgb[:, :, 2] / total


def _exg(image, mode):
    red, green, blue = _chromaticity(image)
    return _scale_fixed(2 * green - red - blue, -1.0, 2.0), "gray"


def _exr(image, mode):
    red, green, _blue = _chromaticity(image)
    return _scale_fixed(1.4 * red - green, -1.0, 1.4), "gray"


def _exgr(image, mode):
    red, green, blue = _chromaticity(image)
    exg = 2 * green - red - blue
    exr = 1.4 * red - green
    return _scale_fixed(exg - exr, -2.0, 2.0), "gray"


def _gli(image, mode):
    rgb = image.astype(np.float32)
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    den = 2 * green + red + blue
    den = np.where(den == 0, 1.0, den)
    return _scale_fixed((2 * green - red - blue) / den, -1.0, 1.0), "gray"


def _vari(image, mode):
    rgb = image.astype(np.float32)
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    den = green + red - blue
    den = np.where(np.abs(den) < 1e-6, 1.0, den)
    return _scale_fixed((green - red) / den, -1.0, 1.0), "gray"


def _ngrdi(image, mode):
    rgb = image.astype(np.float32)
    red, green = rgb[:, :, 0], rgb[:, :, 1]
    den = green + red
    den = np.where(den == 0, 1.0, den)
    return _scale_fixed((green - red) / den, -1.0, 1.0), "gray"


def _gray_world(image, mode):
    rgb = image.astype(np.float32)
    means = rgb.reshape(-1, 3).mean(axis=0)
    gray = float(means.mean())
    scale = np.where(means > 1e-6, gray / means, 1.0).astype(np.float32)
    return np.clip(rgb * scale, 0, 255).astype(np.uint8), "color"


def _hsv_range(image, mode, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h_lo, h_hi = int(h_lo), int(h_hi)
    low_sv = (int(s_lo), int(v_lo))
    high_sv = (int(s_hi), int(v_hi))
    if h_lo <= h_hi:
        mask = cv2.inRange(hsv, (h_lo, *low_sv), (h_hi, *high_sv))
    else:
        # Hue wraps at 180 (OpenCV); useful for red flowers.
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, (h_lo, *low_sv), (179, *high_sv)),
            cv2.inRange(hsv, (0, *low_sv), (h_hi, *high_sv)),
        )
    return mask, "gray"


def _local_variance(image, mode, ksize):
    gray = _to_gray(image, mode).astype(np.float32)
    ksize = _fit_ksize(ksize, gray, 3)
    mean = cv2.blur(gray, (ksize, ksize))
    mean_sq = cv2.blur(gray * gray, (ksize, ksize))
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return _norm_u8(np.sqrt(var)), "gray"


def _ndsm(image, mode, elev):
    if elev is None:
        raise ValueError("nDSM is not available for this tile")
    return elev.copy(), "gray"


def _gaussian(image, mode, ksize, sigma):
    ksize = _fit_ksize(ksize, image, 1)
    return cv2.GaussianBlur(image, (ksize, ksize), sigmaX=float(sigma)), mode


def _median(image, mode, ksize):
    ksize = _fit_ksize(ksize, image, 3)
    return cv2.medianBlur(image, ksize), mode


def _bilateral(image, mode, diameter, sigma_color, sigma_space):
    return cv2.bilateralFilter(image, int(diameter), float(sigma_color), float(sigma_space)), mode


def _box(image, mode, ksize):
    ksize = max(1, min(int(ksize), int(min(image.shape[:2]))))
    return cv2.blur(image, (ksize, ksize)), mode


def _unsharp(image, mode, ksize, amount):
    ksize = _fit_ksize(ksize, image, 1)
    blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
    return cv2.addWeighted(image, 1.0 + float(amount), blurred, -float(amount), 0), mode


def _sobel(image, mode, ksize):
    gray = _to_gray(image, mode)
    ksize = _fit_ksize(ksize, gray, 1)
    if ksize > 7:
        ksize = 7
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
    return _norm_u8(cv2.magnitude(dx, dy)), "gray"


def _scharr(image, mode):
    gray = _to_gray(image, mode)
    dx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    dy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    return _norm_u8(cv2.magnitude(dx, dy)), "gray"


def _laplacian(image, mode, ksize):
    gray = _to_gray(image, mode)
    ksize = _fit_ksize(ksize, gray, 1)
    if ksize > 7:
        ksize = 7
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=ksize)
    return _norm_u8(np.abs(lap)), "gray"


def _canny(image, mode, low, high, aperture):
    gray = _to_gray(image, mode)
    aperture = _fit_ksize(aperture, gray, 3)
    if aperture > 7:
        aperture = 7
    low_v, high_v = float(low), float(high)
    if low_v > high_v:
        low_v, high_v = high_v, low_v
    return cv2.Canny(gray, low_v, high_v, apertureSize=aperture), "gray"


def _morph(kind):
    def run(image, mode, ksize):
        ksize = _fit_ksize(ksize, image, 1)
        kernel = _kernel(ksize)
        if kind == "erode":
            out = cv2.erode(image, kernel)
        elif kind == "dilate":
            out = cv2.dilate(image, kernel)
        else:
            flag = {
                "open": cv2.MORPH_OPEN,
                "close": cv2.MORPH_CLOSE,
                "gradient": cv2.MORPH_GRADIENT,
                "tophat": cv2.MORPH_TOPHAT,
                "blackhat": cv2.MORPH_BLACKHAT,
            }[kind]
            out = cv2.morphologyEx(image, flag, kernel)
        return out, mode
    return run


def _threshold(image, mode, thresh):
    _out, binary = cv2.threshold(_to_gray(image, mode), float(thresh), 255, cv2.THRESH_BINARY)
    return binary, "gray"


def _otsu(image, mode):
    _out, binary = cv2.threshold(
        _to_gray(image, mode), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return binary, "gray"


def _adaptive(kind):
    method = cv2.ADAPTIVE_THRESH_MEAN_C if kind == "mean" else cv2.ADAPTIVE_THRESH_GAUSSIAN_C

    def run(image, mode, block, offset):
        gray = _to_gray(image, mode)
        limit = int(min(gray.shape[:2]))
        if limit < 3:
            raise ValueError("preview is too small for adaptive threshold")
        block = min(int(block), limit if limit % 2 else limit - 1)
        if block < 3:
            block = 3
        if block % 2 == 0:
            block += 1
        return cv2.adaptiveThreshold(gray, 255, method, cv2.THRESH_BINARY, block, float(offset)), "gray"
    return run


def _op(name, label, group, fn, params, needs="any", makes="same", hint=None, uses_elev=False):
    spec = {
        "name": name,
        "label": label,
        "group": group,
        "needs": needs,
        "makes": makes,
        "params": params,
        "fn": fn,
    }
    if hint:
        spec["hint"] = hint
    if uses_elev:
        spec["uses_elev"] = True
    return spec


_INDEX_HINT = "Drawn as gray. The index is stretched into 0–255 so it is visible."
_EDGE_HINT = "Drawn as gray and scaled to this preview."
_THRESH_HINT = "Turns the preview gray, then thresholds it."
_KSIZE = _num("ksize", "Kernel", 3, 1, 31, 2, integer=True, odd=True)

OPS = [
    _op("grayscale", "Grayscale", "intensity", _grayscale, [], needs="color", makes="gray"),
    _op("brightness", "Brightness / contrast", "intensity", _brightness, [
        _num("contrast", "Contrast", 1, 0.2, 3, 0.05),
        _num("brightness", "Brightness", 0, -100, 100, 1, integer=True),
    ]),
    _op("gamma", "Gamma", "intensity", _gamma, [
        _num("gamma", "Gamma", 1, 0.1, 5, 0.1),
    ]),
    _op("invert", "Invert", "intensity", _invert, []),
    _op(
        "equalize", "Histogram equalization", "intensity", _equalize, [],
        makes="gray",
        hint="Turns the preview gray, then spreads its values across the full range.",
    ),
    _op(
        "clahe", "CLAHE", "intensity", _clahe, [
            _num("clip", "Clip", 2, 1, 8, 0.1),
            _num("grid", "Grid", 8, 2, 32, 1, integer=True),
        ],
        hint="On a color image this adjusts lightness and keeps the color.",
    ),
    _op(
        "channel", "RGB channel", "channels", _channel, [
            _choice("channel", "Channel", [("r", "Red"), ("g", "Green"), ("b", "Blue")], "g"),
        ],
        needs="color", makes="gray",
    ),
    _op(
        "hsv", "HSV channel", "channels", _hsv, [
            _choice("channel", "Channel", [("h", "Hue"), ("s", "Saturation"), ("v", "Value")], "h"),
        ],
        needs="color", makes="gray",
        hint="Hue is stretched from OpenCV's 0–179 range so it uses the full gray scale.",
    ),
    _op(
        "lab", "Lab channel", "channels", _lab, [
            _choice("channel", "Channel", [("l", "L"), ("a", "a"), ("b", "b")], "a"),
        ],
        needs="color", makes="gray",
    ),
    _op("exg", "Excess green", "vegetation", _exg, [], needs="color", makes="gray", hint=_INDEX_HINT),
    _op("exr", "Excess red", "vegetation", _exr, [], needs="color", makes="gray", hint=_INDEX_HINT),
    _op("exgr", "Excess green − excess red", "vegetation", _exgr, [], needs="color", makes="gray", hint=_INDEX_HINT),
    _op("gli", "GLI", "vegetation", _gli, [], needs="color", makes="gray", hint=_INDEX_HINT),
    _op(
        "vari", "VARI", "vegetation", _vari, [],
        needs="color", makes="gray",
        hint=_INDEX_HINT,
    ),
    _op(
        "ngrdi", "NGRDI", "vegetation", _ngrdi, [],
        needs="color", makes="gray",
        hint=_INDEX_HINT,
    ),
    _op(
        "hsv_range", "HSV range mask", "vegetation", _hsv_range, [
            _num("h_lo", "H min", 25, 0, 179, 1, integer=True),
            _num("h_hi", "H max", 95, 0, 179, 1, integer=True),
            _num("s_lo", "S min", 25, 0, 255, 1, integer=True),
            _num("s_hi", "S max", 255, 0, 255, 1, integer=True),
            _num("v_lo", "V min", 25, 0, 255, 1, integer=True),
            _num("v_hi", "V max", 255, 0, 255, 1, integer=True),
        ],
        needs="color", makes="gray",
        hint="Binary mask. Defaults target green canopy. If H min > H max, hue wraps past red (OpenCV 0–179).",
    ),
    _op(
        "gray_world", "Gray-world white balance", "intensity", _gray_world, [],
        needs="color",
        hint="Scales each channel so their means match. Helps compare sites under different light.",
    ),
    _op(
        "local_variance", "Local variance", "texture", _local_variance, [
            _num("ksize", "Kernel", 7, 3, 31, 2, integer=True, odd=True),
        ],
        makes="gray",
        hint="Drawn as gray. Bright pixels are high local contrast (leaf texture, edges).",
    ),
    _op(
        "ndsm", "nDSM height", "elevation", _ndsm, [],
        makes="gray", uses_elev=True,
        hint="Drawn as gray. Height above ground (DSM − DTM), stretched per tile. Replaces the RGB preview.",
    ),
    _op("gaussian", "Gaussian blur", "smooth", _gaussian, [
        _num("ksize", "Kernel", 5, 1, 31, 2, integer=True, odd=True),
        _num("sigma", "Sigma", 0, 0, 20, 0.5),
    ]),
    _op("median", "Median blur", "smooth", _median, [
        _num("ksize", "Kernel", 5, 3, 31, 2, integer=True, odd=True),
    ]),
    _op("bilateral", "Bilateral filter", "smooth", _bilateral, [
        _num("diameter", "Diameter", 9, 1, 15, 2, integer=True, odd=True),
        _num("sigma_color", "Color σ", 75, 1, 200, 1, integer=True),
        _num("sigma_space", "Space σ", 75, 1, 200, 1, integer=True),
    ]),
    _op("box", "Box blur", "smooth", _box, [
        _num("ksize", "Kernel", 5, 1, 31, 2, integer=True, odd=True),
    ]),
    _op("unsharp", "Unsharp mask", "sharpen", _unsharp, [
        _num("ksize", "Kernel", 5, 1, 31, 2, integer=True, odd=True),
        _num("amount", "Amount", 1, 0, 3, 0.1),
    ]),
    _op("sobel", "Sobel", "edges", _sobel, [
        _num("ksize", "Kernel", 3, 1, 7, 2, integer=True, odd=True),
    ], makes="gray", hint=_EDGE_HINT),
    _op("scharr", "Scharr", "edges", _scharr, [], makes="gray", hint=_EDGE_HINT),
    _op("laplacian", "Laplacian", "edges", _laplacian, [
        _num("ksize", "Kernel", 3, 1, 7, 2, integer=True, odd=True),
    ], makes="gray", hint=_EDGE_HINT),
    _op("canny", "Canny", "edges", _canny, [
        _num("low", "Low", 50, 0, 255, 1, integer=True),
        _num("high", "High", 150, 0, 255, 1, integer=True),
        _num("aperture", "Aperture", 3, 3, 7, 2, integer=True, odd=True),
    ], makes="gray", hint="Drawn as gray. Bright pixels are edges."),
    _op("erode", "Erode", "morphology", _morph("erode"), [_KSIZE]),
    _op("dilate", "Dilate", "morphology", _morph("dilate"), [_KSIZE]),
    _op("morph_open", "Open", "morphology", _morph("open"), [_KSIZE]),
    _op("morph_close", "Close", "morphology", _morph("close"), [_KSIZE]),
    _op("gradient", "Morphological gradient", "morphology", _morph("gradient"), [_KSIZE]),
    _op("tophat", "Top-hat", "morphology", _morph("tophat"), [_KSIZE]),
    _op("blackhat", "Black-hat", "morphology", _morph("blackhat"), [_KSIZE]),
    _op("threshold", "Binary threshold", "threshold", _threshold, [
        _num("thresh", "Threshold", 127, 0, 255, 1, integer=True),
    ], makes="gray", hint=_THRESH_HINT),
    _op("otsu", "Otsu threshold", "threshold", _otsu, [], makes="gray", hint=_THRESH_HINT),
    _op("adaptive_mean", "Adaptive mean", "threshold", _adaptive("mean"), [
        _num("block", "Block", 11, 3, 99, 2, integer=True, odd=True),
        _num("offset", "Offset", 2, -20, 20, 1, integer=True),
    ], makes="gray", hint=_THRESH_HINT),
    _op("adaptive_gaussian", "Adaptive Gaussian", "threshold", _adaptive("gaussian"), [
        _num("block", "Block", 11, 3, 99, 2, integer=True, odd=True),
        _num("offset", "Offset", 2, -20, 20, 1, integer=True),
    ], makes="gray", hint=_THRESH_HINT),
]

GROUPS = (
    ("intensity", "Intensity"),
    ("channels", "Channels"),
    ("vegetation", "Vegetation"),
    ("elevation", "Elevation"),
    ("texture", "Texture"),
    ("smooth", "Smooth"),
    ("sharpen", "Sharpen"),
    ("edges", "Edges"),
    ("morphology", "Morphology"),
    ("threshold", "Threshold"),
)

BY_NAME = {spec["name"]: spec for spec in OPS}


def catalog():
    """Filter list the Vision tab renders. Functions stay on the server."""
    ops = []
    for spec in OPS:
        public = {key: value for key, value in spec.items() if key != "fn"}
        ops.append(public)
    return {
        "edges": list(EDGES),
        "max_ops": MAX_OPS,
        "groups": [{"id": group_id, "label": label} for group_id, label in GROUPS],
        "ops": ops,
    }


def _check_edge(max_edge):
    if max_edge not in EDGES:
        raise ValueError("max_edge must be 256, 512, or 1024")


def _clamp_param(label, param, value):
    if param["kind"] == "select":
        choices = [option["value"] for option in param["options"]]
        if value not in choices:
            raise ValueError(f"{label} {param['label'].lower()} must be one of {', '.join(choices)}")
        return value
    if isinstance(value, bool) or isinstance(value, str):
        raise ValueError(f"{label} {param['label'].lower()} must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} {param['label'].lower()} must be a number") from exc
    if not np.isfinite(number):
        raise ValueError(f"{label} {param['label'].lower()} must be a number")
    number = min(float(param["max"]), max(float(param["min"]), number))
    if param.get("integer"):
        number = int(round(number))
        number = min(int(param["max"]), max(int(param["min"]), number))
        if param.get("odd") and number % 2 == 0:
            number = number + 1 if number < int(param["max"]) else number - 1
        return int(number)
    return number


def normalize_ops(ops):
    if not isinstance(ops, list):
        raise ValueError("ops must be a list")
    if len(ops) > MAX_OPS:
        raise ValueError(f"at most {MAX_OPS} filters")
    cleaned = []
    for raw in ops:
        if not isinstance(raw, dict):
            raise ValueError("each filter must be an object")
        name = raw.get("name")
        spec = BY_NAME.get(name) if isinstance(name, str) else None
        if spec is None:
            shown = name if isinstance(name, str) and name else "missing"
            raise ValueError(f"unknown filter {shown}")
        known = {param["key"] for param in spec["params"]}
        extra = [key for key in raw if key != "name" and key not in known]
        if extra:
            raise ValueError(f"unknown param {extra[0]} for {spec['label']}")
        params = {"name": name}
        for param in spec["params"]:
            if param["key"] not in raw or raw[param["key"]] is None:
                value = param["default"]
            else:
                value = raw[param["key"]]
            params[param["key"]] = _clamp_param(spec["label"], param, value)
        cleaned.append(params)
    return cleaned


def _run(rgb, alpha, ops, elev=None):
    image = np.array(rgb, copy=True)
    image[alpha == 0] = EMPTY_RGB
    mode = "color"
    for op in ops:
        spec = BY_NAME[op["name"]]
        if spec["needs"] == "color" and mode != "color":
            raise ValueError(
                f"{spec['label']} needs a color image. An earlier step already turned the preview gray."
            )
        kwargs = {key: value for key, value in op.items() if key != "name"}
        if spec.get("uses_elev"):
            kwargs["elev"] = elev
        try:
            image, mode = spec["fn"](image, mode, **kwargs)
        except cv2.error as exc:
            raise ValueError(f"{spec['label']} could not run on this preview") from exc
    return image, mode


def apply(rgb, alpha, ops, elev=None):
    """Paint empty margin, run the chain, and return ``(image, mode)``."""
    return _run(rgb, alpha, normalize_ops(ops), elev=elev)


def _bins(values):
    if values.size == 0:
        return [0] * 256
    flat = np.asarray(values, dtype=np.uint8).reshape(-1)
    return np.bincount(flat, minlength=256)[:256].astype(int).tolist()


def _stats(values):
    if values.size == 0:
        return {"min": None, "max": None, "mean": None, "std": None}
    return {
        "min": int(values.min()),
        "max": int(values.max()),
        "mean": round(float(values.mean()), 2),
        "std": round(float(values.std()), 2),
    }


def _histogram(image, mode, alpha):
    """Gray histogram of valid pixels. Empty margin is left out."""
    valid = alpha != 0
    if mode == "gray":
        gray_values = image[valid]
        channels = (None, None, None)
    else:
        pixels = image[valid]
        gray_values = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)[valid]
        if pixels.size == 0:
            channels = (_bins(pixels), _bins(pixels), _bins(pixels))
        else:
            channels = (_bins(pixels[:, 0]), _bins(pixels[:, 1]), _bins(pixels[:, 2]))
    return {
        "gray": _bins(gray_values),
        "r": channels[0],
        "g": channels[1],
        "b": channels[2],
    }, _stats(gray_values)


def _jpeg_b64(image, mode):
    encoded = image if mode == "gray" else cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", encoded, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise ValueError("could not encode the preview")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _gray_png_b64(image, mode, alpha):
    """Lossless gray plane plus the valid-pixel mask, matching the histogram."""
    gray = image if mode == "gray" else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    bgra = cv2.merge((gray, gray, gray, alpha))
    ok, buf = cv2.imencode(".png", bgra, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    if not ok:
        raise ValueError("could not encode the gray plane")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def render(rgb, alpha, ops, elev=None):
    """JSON-ready preview. The JPEG lives only in this dict."""
    image, mode = apply(rgb, alpha, ops, elev=elev)
    histogram, stats = _histogram(image, mode, alpha)
    return {
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "mode": mode,
        "has_ndsm": elev is not None,
        "image": _jpeg_b64(image, mode),
        "gray": _gray_png_b64(image, mode, alpha),
        "histogram": histogram,
        "stats": stats,
    }


def _preview_hw(height, width, max_edge):
    scale = min(1.0, max_edge / float(max(height, width)))
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


def _to_uint8(data):
    """Same stretch as the thumbnail builder, applied to the preview read."""
    if data.dtype == np.uint8:
        return data
    peak = float(np.max(data)) if data.size else 0.0
    if peak <= 0:
        return np.zeros_like(data, dtype=np.uint8)
    if peak <= 255:
        return data.astype(np.uint8)
    return np.clip(data.astype(np.float32) / peak * 255.0, 0, 255).astype(np.uint8)


def _read_rgb(path, max_edge):
    with rasterio.open(path) as src:
        out_h, out_w = _preview_hw(src.height, src.width, max_edge)
        data = src.read(
            [1, 2, 3],
            out_shape=(3, out_h, out_w),
            resampling=Resampling.bilinear,
        )
        rgb = np.ascontiguousarray(np.moveaxis(_to_uint8(data), 0, -1))
        if src.count >= 4:
            alpha = src.read(4, out_shape=(out_h, out_w), resampling=Resampling.nearest)
        else:
            alpha = src.read_masks(1, out_shape=(out_h, out_w), resampling=Resampling.nearest)
    alpha = np.where(alpha != 0, np.uint8(255), np.uint8(0))
    return rgb, alpha


def _match_tile_file(folder, suffix):
    if folder is None or not folder.is_dir():
        return None
    needle = suffix.lower()
    for path in folder.iterdir():
        if path.is_file() and path.name.lower().endswith(needle):
            return path
    return None


def _elev_paths(img_path):
    """Sibling DSM/DTM files that share the tile's ``_RR_CC`` suffix."""
    path = Path(img_path)
    match = _TILE_SUFFIX.search(path.name)
    if match is None:
        return None, None
    suffix = match.group(1)
    split_dir = path.parent.parent
    return (
        _match_tile_file(split_dir / "dsm", suffix),
        _match_tile_file(split_dir / "dtm", suffix),
    )


def _read_float_band(path, out_h, out_w):
    with rasterio.open(path) as src:
        data = src.read(
            1,
            out_shape=(out_h, out_w),
            resampling=Resampling.bilinear,
        ).astype(np.float32)
        nodata = src.nodata if src.nodata is not None else _DSM_NODATA
    return data, float(nodata)


def _stretch_elev(ndsm, valid):
    """Percentile stretch of height (m) into a gray preview plane."""
    if not np.any(valid):
        return np.zeros(ndsm.shape, dtype=np.uint8)
    values = ndsm[valid]
    lo, hi = np.percentile(values, (2.0, 98.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(values.min())
        hi = float(values.max())
    if hi <= lo:
        out = np.zeros(ndsm.shape, dtype=np.uint8)
        out[valid] = 128
        return out
    scaled = (ndsm - lo) / (hi - lo) * 255.0
    out = np.clip(scaled, 0, 255).astype(np.uint8)
    out[~valid] = 0
    return out


def _read_ndsm(img_path, out_h, out_w):
    dsm_path, dtm_path = _elev_paths(img_path)
    if dsm_path is None or dtm_path is None:
        return None
    try:
        dsm, dsm_nodata = _read_float_band(dsm_path, out_h, out_w)
        dtm, dtm_nodata = _read_float_band(dtm_path, out_h, out_w)
    except (RasterioIOError, OSError):
        return None
    valid = (
        (dsm != dsm_nodata)
        & np.isfinite(dsm)
        & (dtm != dtm_nodata)
        & np.isfinite(dtm)
    )
    ndsm = dsm - dtm
    return _stretch_elev(ndsm, valid)


def load_source(tile_id, path, max_edge):
    """Return private copies of RGB, alpha, and optional nDSM gray plane."""
    _check_edge(max_edge)
    key = (tile_id, max_edge)
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
            rgb, alpha, elev = hit
            elev_out = None if elev is None else elev.copy()
            return rgb.copy(), alpha.copy(), elev_out
    try:
        rgb, alpha = _read_rgb(path, max_edge)
    except (RasterioIOError, OSError) as exc:
        raise FileNotFoundError("tile image could not be read") from exc
    elev = _read_ndsm(path, rgb.shape[0], rgb.shape[1])
    with _cache_lock:
        _cache[key] = (rgb, alpha, elev)
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
    elev_out = None if elev is None else elev.copy()
    return rgb.copy(), alpha.copy(), elev_out
