"""Long-lived tile predictor. Run with the repo-root ``.venv`` Python.

The explorer API writes one JSON object per line on stdin and reads one JSON
object per line on stdout. Torch stays loaded between requests. stderr is a log.
"""

import base64
import io
import json
import math
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
import rasterio
import yaml
from PIL import Image
from rasterio.enums import Resampling

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "explorer"))

from common import IGNORE_COLORS, THUMB_PX, load_names  # noqa: E402

warnings.filterwarnings("ignore")

RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
NDSM_MEAN = 1.43
NDSM_STD = 3.10
DSM_NODATA = -32767.0
LOG_ROOT = ROOT / "lightning_logs"
EXPLAIN_MODES = ("none", "gradcam", "saliency", "confidence", "entropy")
SALIENCY_EDGE = 1024

_model = None
_model_path = None
_device = None
# Activation bytes per tile, and the smallest stack that did not fit, by
# (checkpoint, height, width). The next forward is sized from free VRAM.
_batch_cost = {}
_batch_fail = {}
_MAX_STACK = 32


def _reply(payload):
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _to_uint8(data):
    if data.dtype == np.uint8:
        return data
    peak = float(np.max(data)) if data.size else 0.0
    if peak <= 0:
        return np.zeros_like(data, dtype=np.uint8)
    if peak <= 255:
        return data.astype(np.uint8)
    return np.clip(data.astype(np.float32) / peak * 255.0, 0, 255).astype(np.uint8)


def _read_rgb(path):
    """Native RGB, the same read test.py's dataset uses. No resample."""
    with rasterio.open(path) as src:
        data = src.read([1, 2, 3])
        rgb = np.ascontiguousarray(np.moveaxis(data, 0, -1))
        if rgb.dtype != np.uint8:
            rgb = _to_uint8(rgb)
        if src.count >= 4:
            alpha = src.read(4)
        else:
            alpha = src.read_masks(1)
    alpha = np.where(alpha != 0, np.uint8(255), np.uint8(0))
    return rgb, alpha


def _read_label(path):
    with rasterio.open(path) as src:
        return src.read(1)


def _match_tile_file(folder, suffix):
    if folder is None or not folder.is_dir():
        return None
    needle = suffix.lower()
    for path in folder.iterdir():
        if path.is_file() and path.name.lower().endswith(needle):
            return path
    return None


def _elev_paths(img_path):
    import re
    match = re.search(r"(_\d{2,3}_\d{2,3}\.tiff?)$", Path(img_path).name, re.IGNORECASE)
    if match is None:
        return None, None
    split_dir = Path(img_path).parent.parent
    suffix = match.group(1)
    return (
        _match_tile_file(split_dir / "dsm", suffix),
        _match_tile_file(split_dir / "dtm", suffix),
    )


def _read_float(path, out_h, out_w):
    with rasterio.open(path) as src:
        data = src.read(1, out_shape=(out_h, out_w), resampling=Resampling.bilinear).astype(np.float32)
        nodata = src.nodata if src.nodata is not None else DSM_NODATA
    return data, float(nodata)


def _read_ndsm(img_path, out_h, out_w):
    dsm_path, dtm_path = _elev_paths(img_path)
    if dsm_path is None or dtm_path is None:
        return None
    dsm, dsm_nodata = _read_float(dsm_path, out_h, out_w)
    dtm, dtm_nodata = _read_float(dtm_path, out_h, out_w)
    nodata = (
        (dsm == dsm_nodata) | ~np.isfinite(dsm)
        | (dtm == dtm_nodata) | ~np.isfinite(dtm)
    )
    ndsm = dsm - dtm
    ndsm[nodata] = 0.0
    return ndsm


def _hydra_config(exp_dir):
    matches = sorted(exp_dir.glob("hydra/**/.hydra/config.yaml"))
    if not matches:
        return None
    return yaml.safe_load(matches[-1].read_text()) or {}


def _norms(exp_dir):
    mean = RGB_MEAN.copy()
    std = RGB_STD.copy()
    ndsm_mean, ndsm_std = NDSM_MEAN, NDSM_STD
    ds2 = "_ds2" in exp_dir.name
    cfg = _hydra_config(exp_dir)
    data = (cfg or {}).get("data") or {}
    if isinstance(data.get("norm_mean"), (list, tuple)) and len(data["norm_mean"]) == 3:
        mean = np.array(data["norm_mean"], dtype=np.float32)
    if isinstance(data.get("norm_std"), (list, tuple)) and len(data["norm_std"]) == 3:
        std = np.array(data["norm_std"], dtype=np.float32)
    name = str(data.get("data_name") or "")
    if name.endswith("_ds2"):
        ds2 = True
    for step in data.get("ndsm_transforms") or []:
        if not isinstance(step, dict) or step.get("type") != "Normalize":
            continue
        params = step.get("params") or {}
        if params.get("mean") is not None:
            ndsm_mean = float(params["mean"])
        if params.get("std") is not None:
            ndsm_std = float(params["std"])
    return mean, std, ndsm_mean, ndsm_std, ds2


def _as_dict(value):
    if isinstance(value, dict):
        return dict(value)
    try:
        from omegaconf import OmegaConf
        container = OmegaConf.to_container(value, resolve=True)
    except Exception:
        container = dict(value)
    if not isinstance(container, dict):
        raise ValueError("checkpoint in_channels is not a mapping")
    return container


def _load(ckpt, device):
    global _model, _model_path, _device
    import torch
    from models.semseg_plm import SegmentationModel

    path = str(Path(ckpt).resolve())
    if _model is not None and _model_path == path and _device == device:
        _model.set_patch_2_img_size(False)
        return _model
    if _model is not None:
        del _model
        _model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    model = SegmentationModel.load_from_checkpoint(path, weights_only=False)
    model.eval()
    model.to(device)
    # Attribution backpropagates to inputs and detached features, not weights.
    for param in model.parameters():
        param.requires_grad_(False)
    _model = model
    _model_path = path
    _device = device
    # test.py turns patching off and scores the whole tile in one pass.
    model.set_patch_2_img_size(False)
    return model


def _release():
    global _model, _model_path
    if _model is None:
        return {"ok": True}
    import torch
    del _model
    _model = None
    _model_path = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"ok": True}


def _lut():
    classes, display_order, _ = load_names()
    lut = np.zeros((256, 4), dtype=np.uint8)
    for spec in classes:
        if spec["id"] == 0:
            continue
        lut[spec["id"], :3] = spec["rgb"]
        lut[spec["id"], 3] = 255
    for value, rgb in IGNORE_COLORS.items():
        lut[value, :3] = rgb
        lut[value, 3] = 220
    return lut, classes, display_order


def _png_b64(array):
    buf = io.BytesIO()
    Image.fromarray(array, mode="RGBA").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _jpeg_b64(rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _webp_b64(rgba, edge=None):
    """Square thumbnail of an RGBA overlay. Nearest keeps class edges crisp."""
    edge = THUMB_PX if edge is None else int(edge)
    image = Image.fromarray(np.ascontiguousarray(rgba), mode="RGBA")
    if image.size != (edge, edge):
        image = image.resize((edge, edge), Image.Resampling.NEAREST)
    buf = io.BytesIO()
    image.save(buf, format="WEBP", quality=80, method=0)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _score_f1(tp, fp, fn):
    denom = 2 * tp + fp + fn
    if denom <= 0:
        return None
    return 2.0 * tp / denom


def _score_iou(tp, fp, fn):
    denom = tp + fp + fn
    if denom <= 0:
        return None
    return tp / denom


def _score_precision(tp, fp):
    denom = tp + fp
    if denom <= 0:
        return None
    return tp / denom


def _score_recall(tp, fn):
    denom = tp + fn
    if denom <= 0:
        return None
    return tp / denom


def _mean_defined(values):
    nums = [value for value in values if value is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _round_metric(value):
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return round(value, 6)


def _explain_request(req, num_classes):
    mode = req.get("explain") or "none"
    if mode not in EXPLAIN_MODES:
        names = ", ".join(EXPLAIN_MODES)
        raise ValueError(f"explain must be one of {names}")
    raw = req.get("target_class")
    if raw is None or raw == "":
        return mode, None
    try:
        target = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("target_class must be an integer") from exc
    if target < 0 or target >= num_classes:
        raise ValueError(f"target_class must be between 0 and {num_classes - 1}")
    return mode, target


def _class_rows(classes, display_order):
    by_id = {spec["id"]: spec for spec in classes}
    rows = []
    seen = set()
    for class_id in list(display_order) + [spec["id"] for spec in classes]:
        if class_id in seen or class_id == 0:
            continue
        seen.add(class_id)
        spec = by_id.get(class_id)
        if spec is None:
            continue
        rows.append({
            "id": int(class_id),
            "name": spec["name"],
            "color": spec["color"],
        })
    return rows


def _class_score(logits, target_class):
    """Scalar used for Grad-CAM and input saliency.

    A chosen class sums its logits on pixels predicted as that class. With no
    choice, the score is the predicted species logit on every foreground pixel.
    """
    pred = logits.argmax(dim=1).detach()
    if target_class is None:
        foreground = pred[0] > 0
        picked = logits[0].gather(0, pred[0].unsqueeze(0)).squeeze(0)
        if bool(foreground.any()):
            return picked[foreground].sum()
        species = logits[:, 1:]
        if species.shape[1] == 0:
            return logits.sum()
        return species.sum()
    chosen = logits[0, target_class]
    mask = pred[0] == target_class
    if bool(mask.any()):
        return chosen[mask].sum()
    return chosen.sum()


def _encode_features(model, batch_x, aux, fusion):
    if fusion == "stem":
        features = model.model.encoder(batch_x)
        if model.elevation_fusion is not None and aux is not None:
            features = model.elevation_fusion(aux, features)
        return features
    return model.model.encoder(batch_x)


def _decoder_logits(model, features):
    decoded = model.model.decoder(features)
    return model.model.segmentation_head(decoded)


def _output_stride(model):
    encoder = getattr(getattr(model, "model", None), "encoder", None)
    stride = int(getattr(encoder, "output_stride", 32) or 32)
    return max(1, stride)


def _limit_pair(batch_x, aux, limit, stride):
    """Shrink a forward so attribution fits, keeping the encoder's stride."""
    import torch.nn.functional as F

    height, width = int(batch_x.shape[-2]), int(batch_x.shape[-1])
    if max(height, width) <= limit:
        return batch_x, aux, False
    scale = limit / float(max(height, width))

    def _fit(length):
        fitted = int(length * scale) // stride * stride
        return max(stride, fitted)

    size = (_fit(height), _fit(width))
    resized = F.interpolate(batch_x, size=size, mode="bilinear", align_corners=False)
    resized_aux = None
    if aux is not None:
        resized_aux = F.interpolate(aux, size=size, mode="bilinear", align_corners=False)
    return resized, resized_aux, True


def _unit_interval(heat, valid, stretch):
    heat = np.nan_to_num(heat.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    heat = np.maximum(heat, 0)
    if heat.shape != valid.shape:
        import cv2
        heat = cv2.resize(heat, (valid.shape[1], valid.shape[0]), interpolation=cv2.INTER_LINEAR)
    if stretch:
        sample = heat[valid > 0]
        if sample.size == 0:
            sample = heat.reshape(-1)
        peak = float(np.percentile(sample, 99)) if sample.size else 0.0
        if peak <= 0:
            peak = float(sample.max()) if sample.size else 0.0
        heat = heat / peak if peak > 0 else heat
    heat = np.clip(heat, 0.0, 1.0)
    heat[valid == 0] = 0
    return heat.astype(np.float32)


def _importance_png(heat, alpha):
    """Warm color whose opacity is the heat. Empty margin stays transparent."""
    u8 = np.clip(np.round(heat * 255.0), 0, 255).astype(np.uint8)
    rgba = np.zeros((u8.shape[0], u8.shape[1], 4), dtype=np.uint8)
    rgba[..., 0] = 214
    rgba[..., 1] = 92
    rgba[..., 2] = 31
    strength = u8.copy()
    strength[alpha == 0] = 0
    rgba[..., 3] = strength
    return _png_b64(rgba)


def _mask_edge(mask):
    import cv2

    if not np.any(mask):
        return mask
    kernel = np.ones((3, 3), np.uint8)
    plate = mask.astype(np.uint8)
    return (cv2.dilate(plate, kernel) > 0) & (cv2.erode(plate, kernel) == 0)


def _paint(rgba, mask, ids, lut, alpha=None):
    if not np.any(mask):
        return
    picked = lut[ids[mask]]
    if alpha is not None:
        picked = picked.copy()
        picked[:, 3] = alpha
    rgba[mask] = picked


def _comparison(pred, label, alpha, lut):
    """One layer: agreed fill, extra pixels, missed outlines, wrong-class borders."""
    rgba = np.zeros((*pred.shape, 4), dtype=np.uint8)
    visible = alpha > 0
    ignore = np.isin(label, (254, 255))
    valid = visible & ~ignore
    agree = valid & (pred == label) & (label > 0)
    extra = valid & (pred > 0) & (label == 0)
    missed = valid & (label > 0) & (pred == 0)
    wrong = valid & (pred > 0) & (label > 0) & (pred != label)
    _paint(rgba, agree, label, lut, alpha=168)
    _paint(rgba, extra, pred, lut, alpha=255)
    _paint(rgba, wrong, pred, lut, alpha=255)
    extra_edge = _mask_edge(extra)
    rgba[extra_edge] = (255, 255, 255, 255)
    _paint(rgba, _mask_edge(wrong), label, lut, alpha=255)
    _paint(rgba, _mask_edge(missed), label, lut, alpha=255)
    _paint(rgba, visible & ignore, label, lut)
    return rgba


def _confidence_map(probs, target_class):
    if target_class is None:
        if probs.shape[1] > 1:
            heat = probs[0, 1:].max(dim=0).values
        else:
            heat = probs[0, 0]
    else:
        heat = probs[0, target_class]
    return heat.detach().float().cpu().numpy()


def _entropy_map(probs):
    import torch
    safe = probs.clamp_min(1e-8)
    safe = safe / safe.sum(dim=1, keepdim=True)
    entropy = -(safe * safe.log()).sum(dim=1)[0]
    entropy = entropy / math.log(probs.shape[1])
    return entropy.detach().float().cpu().numpy()


def _gradcam(model, batch_x, aux, fusion, target_class):
    """Seg-Grad-CAM on the deepest encoder map the decoder actually uses.

    The encoder runs without a graph. Only the detached feature maps take
    gradients, so a full tile does not store the backbone's activations.
    """
    import torch

    with torch.no_grad():
        raw = list(_encode_features(model, batch_x, aux, fusion))
    deepest = None
    for index in range(len(raw) - 1, -1, -1):
        feat = raw[index]
        spatial = (
            torch.is_tensor(feat)
            and feat.ndim == 4
            and feat.shape[1] > 0
            and min(feat.shape[-2:]) > 0
        )
        if spatial:
            deepest = index
            break
    if deepest is None:
        raise ValueError("This checkpoint has no spatial features for Grad-CAM.")
    live = []
    for index, feat in enumerate(raw):
        if not torch.is_tensor(feat):
            live.append(feat)
            continue
        leaf = feat.detach()
        if index == deepest:
            leaf = leaf.requires_grad_(True)
        live.append(leaf)
    del raw
    with torch.enable_grad():
        logits = _decoder_logits(model, live)
        score = _class_score(logits, target_class)
        if not score.requires_grad:
            raise ValueError("Grad-CAM did not receive gradients for this checkpoint.")
        model.zero_grad(set_to_none=True)
        score.backward()
    chosen = None
    for index in range(len(live) - 1, -1, -1):
        feat = live[index]
        if torch.is_tensor(feat) and feat.grad is not None and feat.shape[1] > 0:
            chosen = feat
            break
    if chosen is None:
        raise ValueError("Grad-CAM did not receive gradients for this checkpoint.")
    weights = chosen.grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * chosen).sum(dim=1, keepdim=True))
    import torch.nn.functional as F
    cam = F.interpolate(cam, size=logits.shape[-2:], mode="bilinear", align_corners=False)
    heat = cam[0, 0].detach().float().cpu().numpy()
    probs = torch.softmax(logits.detach(), dim=1)
    del score, logits, live, cam
    return heat, probs


def _channel_attr(values, grads):
    return (values.detach().float() * grads.detach().float()).abs().sum(dim=1)[0]


def _saliency_groups(model, fusion, image, extra):
    """Channel groups that should not drown each other when summed."""
    if fusion == "stem":
        groups = [(image, image.grad)]
        if extra is not None and extra.grad is not None:
            groups.append((extra, extra.grad))
        return groups
    channels = _as_dict(model.in_channels or {"image": image.shape[1]})
    groups = []
    offset = 0
    for key, count in channels.items():
        if key not in ("image", "ndsm"):
            continue
        count = int(count)
        groups.append((image[:, offset:offset + count], image.grad[:, offset:offset + count]))
        offset += count
    if not groups:
        groups.append((image, image.grad))
    return groups


def _saliency(model, batch_x, aux, fusion, target_class):
    """Input times gradient, with RGB and nDSM kept as separate maps."""
    import torch

    image = batch_x.detach().requires_grad_(True)
    extra = None if aux is None else aux.detach().requires_grad_(True)
    use_amp = image.is_cuda
    with torch.enable_grad(), torch.autocast(
        device_type=image.device.type, dtype=torch.bfloat16, enabled=use_amp,
    ):
        logits = model._forward(image, extra)
    score = _class_score(logits.float(), target_class)
    if not score.requires_grad:
        raise ValueError("Saliency did not receive gradients for this checkpoint.")
    model.zero_grad(set_to_none=True)
    score.backward()
    if image.grad is None:
        raise ValueError("Saliency did not receive gradients for this checkpoint.")
    maps = [_channel_attr(values, grads) for values, grads in _saliency_groups(model, fusion, image, extra)]
    heat = None
    for part in maps:
        plane = part.detach().float().cpu().numpy()
        plane = np.nan_to_num(plane, nan=0.0, posinf=0.0, neginf=0.0)
        plane = np.maximum(plane, 0)
        peak = float(np.percentile(plane, 99)) if plane.size else 0.0
        if peak <= 0:
            peak = float(plane.max()) if plane.size else 0.0
        if peak > 0:
            plane = plane / peak
        plane = np.clip(plane, 0.0, 1.0)
        heat = plane if heat is None else np.maximum(heat, plane)
    del score, logits, image, extra
    import cv2
    heat = cv2.GaussianBlur(heat.astype(np.float32), (0, 0), sigmaX=2)
    return heat


def _forward_probs(model, batch_x, aux):
    import torch
    with torch.inference_mode():
        return model(batch_x, None if aux is None else aux)


def _importance(model, batch_x, aux, fusion, explain, target_class, alpha):
    """Return a [0, 1] map and, for Grad-CAM, the probabilities of that pass."""
    import torch

    reduced = False
    probs = None
    if explain == "gradcam":
        heat = None
        try:
            heat, probs = _gradcam(model, batch_x, aux, fusion, target_class)
        except torch.cuda.OutOfMemoryError:
            heat, probs = None, None
        if heat is None:
            torch.cuda.empty_cache()
            smaller, smaller_aux, did_scale = _limit_pair(
                batch_x, aux, SALIENCY_EDGE, _output_stride(model),
            )
            if not did_scale:
                raise ValueError("The GPU ran out of memory.")
            try:
                heat, _unused = _gradcam(model, smaller, smaller_aux, fusion, target_class)
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                raise ValueError("The GPU ran out of memory.") from exc
            del _unused
            reduced = True
        heat = _unit_interval(heat, alpha, stretch=True)
        return heat, probs, reduced
    if explain == "saliency":
        smaller, smaller_aux, _scaled = _limit_pair(
            batch_x, aux, SALIENCY_EDGE, _output_stride(model),
        )
        heat = None
        try:
            heat = _saliency(model, smaller, smaller_aux, fusion, target_class)
        except torch.cuda.OutOfMemoryError:
            heat = None
        if heat is None:
            torch.cuda.empty_cache()
            smaller, smaller_aux, _scaled = _limit_pair(
                smaller, smaller_aux, SALIENCY_EDGE // 2, _output_stride(model),
            )
            reduced = True
            try:
                heat = _saliency(model, smaller, smaller_aux, fusion, target_class)
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                raise ValueError("The GPU ran out of memory.") from exc
        heat = _unit_interval(heat, alpha, stretch=True)
        return heat, None, reduced
    probs = _forward_probs(model, batch_x, aux)
    if explain == "confidence":
        heat = _confidence_map(probs, target_class)
        return _unit_interval(heat, alpha, stretch=False), probs, False
    if explain == "entropy":
        heat = _entropy_map(probs)
        return _unit_interval(heat, alpha, stretch=False), probs, False
    return None, probs, False


TRACE_EDGE = 512
TRACE_GRID = 128
# Outside the photo, and not a class id. The pixel probe uses this to say "empty".
TRACE_EMPTY = 253


def _fit_hw(height, width, edge):
    if max(height, width) <= edge:
        return height, width
    scale = edge / float(max(height, width))
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


def _resize_hw(array, height, width, interpolation):
    import cv2

    if array.shape[0] == height and array.shape[1] == width:
        return array
    return cv2.resize(array, (width, height), interpolation=interpolation)


def _is_spatial(feat):
    shape = getattr(feat, "shape", None)
    if shape is None or getattr(feat, "ndim", 0) != 4:
        return False
    if int(shape[1]) <= 0:
        return False
    return min(int(shape[-2]), int(shape[-1])) > 0


def _preview_numpy(feat, edge):
    """(C, h, w) numpy copy, plus the feature map's real height, width, and channels."""
    import torch.nn.functional as F

    native_c = int(feat.shape[1])
    native_h = int(feat.shape[-2])
    native_w = int(feat.shape[-1])
    work = feat.detach().float()
    if work.ndim == 4:
        work = work[:1]
    if max(native_h, native_w) > edge:
        scale = edge / float(max(native_h, native_w))
        nh = max(1, int(round(native_h * scale)))
        nw = max(1, int(round(native_w * scale)))
        work = F.interpolate(work, size=(nh, nw), mode="bilinear", align_corners=False)
    return np.ascontiguousarray(work[0].cpu().numpy()), native_h, native_w, native_c


def _valid_for(alpha, height, width):
    import cv2

    if alpha.shape[0] == height and alpha.shape[1] == width:
        return alpha
    return cv2.resize(alpha, (width, height), interpolation=cv2.INTER_NEAREST)


def _pca_rgb(feat, valid):
    """Three-color picture of a feature map. Similar pixels get similar colors."""
    channels, height, width = feat.shape
    flat = np.nan_to_num(feat.reshape(channels, -1).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    keep = min(3, channels)
    try:
        count = flat.shape[1]
        sample = flat
        if count > 6000:
            pick = np.random.default_rng(0).choice(count, 6000, replace=False)
            sample = flat[:, pick]
        centered = sample - sample.mean(axis=1, keepdims=True)
        cov = centered @ centered.T
        _evals, evecs = np.linalg.eigh(cov)
        basis = evecs[:, np.argsort(_evals)[::-1][:keep]]
        projected = basis.T @ (flat - flat.mean(axis=1, keepdims=True))
    except np.linalg.LinAlgError:
        projected = flat[:keep]
    if keep < 3:
        projected = np.pad(projected, ((0, 3 - keep), (0, 0)))
    rgb = np.zeros((height, width, 3), np.uint8)
    for index in range(3):
        plane = projected[index].reshape(height, width)
        sample = plane[valid > 0] if valid is not None and np.any(valid > 0) else plane.reshape(-1)
        low = float(np.percentile(sample, 2)) if sample.size else 0.0
        high = float(np.percentile(sample, 98)) if sample.size else 1.0
        if high <= low:
            high = low + 1e-6
        rgb[..., index] = np.clip((plane - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)
    return rgb


def _pick_channels(feat, count=3):
    """A few channels that actually vary across the tile, spaced apart."""
    channels = feat.shape[0]
    if channels <= count:
        return list(range(channels))
    variance = np.nan_to_num(feat.reshape(channels, -1).var(axis=1), nan=0.0)
    order = np.argsort(variance)[::-1]
    gap = max(1, channels // 20)
    chosen = []
    for index in order:
        index = int(index)
        if all(abs(index - other) > gap for other in chosen):
            chosen.append(index)
        if len(chosen) == count:
            break
    if len(chosen) < count:
        for index in order:
            index = int(index)
            if index not in chosen:
                chosen.append(index)
            if len(chosen) == count:
                break
    return chosen


def _rgb_png(rgb, valid):
    rgba = np.zeros((rgb.shape[0], rgb.shape[1], 4), np.uint8)
    rgba[..., :3] = rgb[..., :3]
    if valid is None:
        rgba[..., 3] = 255
    else:
        rgba[..., 3] = np.where(valid > 0, 255, 0).astype(np.uint8)
    return _png_b64(np.ascontiguousarray(rgba))


def _gray_png(plane, valid):
    heat = _unit_interval(plane, valid, stretch=True)
    u8 = np.clip(np.round(heat * 255.0), 0, 255).astype(np.uint8)
    rgba = np.zeros((*u8.shape, 4), np.uint8)
    rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = u8
    rgba[..., 3] = np.where(valid > 0, 255, 0).astype(np.uint8)
    return _png_b64(rgba)


def _tint_png(prob, rgb, valid):
    prob = np.clip(np.nan_to_num(prob.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    if prob.shape != valid.shape:
        import cv2
        prob = cv2.resize(prob, (valid.shape[1], valid.shape[0]), interpolation=cv2.INTER_LINEAR)
    strength = np.clip(np.round(prob * 255.0), 0, 255).astype(np.uint8)
    strength[valid == 0] = 0
    rgba = np.zeros((*strength.shape, 4), np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    rgba[..., 3] = strength
    return _png_b64(rgba)


def _terrain_png(plane, valid):
    heat = _unit_interval(np.maximum(plane, 0), valid, stretch=True)
    stops = np.array([
        [27, 63, 54],
        [90, 140, 72],
        [196, 177, 90],
        [244, 248, 249],
    ], np.float32)
    position = heat * (len(stops) - 1)
    left = np.floor(position).astype(np.int32).clip(0, len(stops) - 2)
    mix = (position - left)[..., None]
    rgb = stops[left] * (1.0 - mix) + stops[left + 1] * mix
    rgba = np.zeros((*heat.shape, 4), np.uint8)
    rgba[..., :3] = np.clip(np.round(rgb), 0, 255).astype(np.uint8)
    rgba[..., 3] = np.where(valid > 0, 255, 0).astype(np.uint8)
    return _png_b64(rgba)


def _ink(rgb):
    red, green, blue = (int(channel) for channel in rgb)
    if red + green + blue < 48:
        return (18, 32, 36)
    return (red, green, blue)


def _color_overlay(ids, alpha, lut):
    rgba = lut[ids].copy()
    rgba[alpha == 0, 3] = 0
    return _png_b64(np.ascontiguousarray(rgba))


def _feature_pictures(feat, alpha):
    preview, native_h, native_w, native_c = _preview_numpy(feat, TRACE_EDGE)
    valid = _valid_for(alpha, preview.shape[1], preview.shape[2])
    energy = np.abs(preview).mean(axis=0)
    samples = []
    for index in _pick_channels(preview, 3):
        samples.append({
            "index": int(index),
            "image": _gray_png(preview[index], valid),
        })
    return {
        "preview": preview,
        "valid": valid,
        "height": native_h,
        "width": native_w,
        "channels": native_c,
        "energy": _importance_png(_unit_interval(energy, valid, stretch=True), valid),
        "pca": _rgb_png(_pca_rgb(preview, valid), valid),
        "samples": samples,
    }


def _class_catalog(num_classes):
    _lut_colors, classes, display_order = _lut()
    by_id = {spec["id"]: spec for spec in classes}
    rows = []
    for class_id in range(num_classes):
        spec = by_id.get(class_id)
        if spec is None:
            rows.append({"id": class_id, "name": f"Class {class_id}", "color": "#5d6e74", "rgb": (93, 110, 116)})
        else:
            rows.append({
                "id": class_id,
                "name": spec["name"],
                "color": spec["color"],
                "rgb": tuple(int(channel) for channel in spec["rgb"]),
            })
    return rows, display_order


def _head_facts(model, decoded, logits):
    import torch.nn as nn

    kernel = None
    for module in model.model.segmentation_head.modules():
        if isinstance(module, nn.Conv2d):
            kernel = int(module.kernel_size[0])
            break
    decoded_h = int(decoded.shape[-2])
    logits_h = int(logits.shape[-2])
    upsample = max(1, int(round(logits_h / float(decoded_h)))) if decoded_h else 1
    return {
        "kernel": kernel,
        "upsample": upsample,
        "in_channels": int(decoded.shape[1]),
        "out_classes": int(logits.shape[1]),
    }


def _inputs(req, on_device=True):
    """Tile tensors for one checkpoint, prepared the same way as test.py."""
    import cv2
    import torch

    ckpt = Path(req["ckpt"]).resolve()
    image_path = Path(req["image"])
    mask_path = Path(req["mask"]) if req.get("mask") else None
    if not ckpt.is_file() or LOG_ROOT.resolve() not in ckpt.parents:
        raise ValueError("checkpoint must be a file under lightning_logs")
    if not image_path.is_file():
        raise ValueError("tile image is not on disk")

    exp_dir = ckpt.parent.parent
    mean, std, ndsm_mean, ndsm_std, ds2 = _norms(exp_dir)
    rgb, alpha = _read_rgb(image_path)
    native_h, native_w = rgb.shape[:2]
    height, width = native_h, native_w
    label = None
    if mask_path is not None and mask_path.is_file():
        try:
            label = _read_label(mask_path)
        except (OSError, rasterio.errors.RasterioIOError):
            label = None
        else:
            if label.shape[:2] != (height, width):
                label = cv2.resize(label, (width, height), interpolation=cv2.INTER_NEAREST)

    # Full-resolution runs are scored on the native tile. A _ds2 run is scored
    # on the half-resolution grid (2048 -> 1024), cubic on the photo and nearest
    # on the label, matching transforms_orig_ds2.
    if ds2:
        height, width = max(1, height // 2), max(1, width // 2)
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_CUBIC)
        alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_NEAREST)
        if label is not None:
            label = cv2.resize(label, (width, height), interpolation=cv2.INTER_NEAREST)

    torch.set_float32_matmul_precision("medium")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load(ckpt, device)
    channels = _as_dict(model.in_channels or {"image": 3})
    fusion = str(model.fusion_mode or "concat")

    image = rgb.astype(np.float32) / 255.0
    image = (image - mean) / std
    image_t = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0)

    ndsm = None
    needs_ndsm = any(key == "ndsm" for key in channels)
    if needs_ndsm:
        ndsm = _read_ndsm(image_path, height, width)
        if ndsm is None:
            raise ValueError("This checkpoint expects an nDSM, and this tile has no DSM/DTM.")
        ndsm_t = torch.from_numpy(((ndsm - ndsm_mean) / ndsm_std).astype(np.float32)).unsqueeze(0).unsqueeze(0)
    else:
        ndsm_t = None

    aux_keys = [key for key in channels if key != "image"]
    unsupported = [key for key in aux_keys if key != "ndsm"]
    if unsupported:
        raise ValueError(
            "Tile prediction supports image and ndsm inputs. This checkpoint expects "
            + ", ".join(unsupported) + "."
        )

    if fusion == "stem":
        batch_x = image_t
        aux = ndsm_t
    else:
        parts = []
        for key in channels:
            if key == "image":
                parts.append(image_t)
            elif key == "ndsm":
                parts.append(ndsm_t)
        batch_x = torch.cat(parts, dim=1)
        aux = None

    if on_device:
        batch_x = batch_x.to(device)
        if aux is not None:
            aux = aux.to(device)
    return {
        "model": model,
        "batch_x": batch_x,
        "aux": aux,
        "fusion": fusion,
        "channels": channels,
        "rgb": rgb,
        "alpha": alpha,
        "label": label,
        "norm_image": image,
        "ndsm": ndsm,
        "height": height,
        "width": width,
        "native_height": native_h,
        "native_width": native_w,
        "ds2": ds2,
        "mean": mean,
        "std": std,
        "ndsm_mean": ndsm_mean,
        "ndsm_std": ndsm_std,
    }


def _tile_metrics(pred, label, alpha, classes):
    """Per-tile scores from the prediction and, when present, the label.

    Valid pixels sit inside the photo and are not marked unlabelled (254) or
    uncertain (255). Species averages skip a class that is absent from both
    the label and the prediction, matching ``metrics_from_confusion``.
    """
    visible = alpha > 0
    by_id = {spec["id"]: spec for spec in classes}
    pred_px = int(((pred > 0) & visible).sum())

    def _row(class_id, spec, px, scored=None):
        row = {
            "id": int(class_id),
            "stats_name": spec.get("stats_name") or "",
            "name": spec["name"],
            "color": spec["color"],
            "px": int(px),
        }
        if scored is None:
            return row
        tp, fp, fn = scored
        row.update({
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "f1": _round_metric(_score_f1(tp, fp, fn)),
            "iou": _round_metric(_score_iou(tp, fp, fn)),
            "precision": _round_metric(_score_precision(tp, fp)),
            "recall": _round_metric(_score_recall(tp, fn)),
        })
        return row

    if label is None:
        counts = []
        for class_id, spec in by_id.items():
            if class_id == 0:
                continue
            px = int(((pred == class_id) & visible).sum())
            if px <= 0:
                continue
            counts.append(_row(class_id, spec, px))
        return {
            "agreement": None,
            "agreement_species": None,
            "f1_species": None,
            "iou_species": None,
            "precision_species": None,
            "recall_species": None,
            "binary_f1": None,
            "binary_iou": None,
            "pred_px": pred_px,
            "label_px": None,
            "tp_px": None,
            "fp_px": None,
            "fn_px": None,
            "wrong_species_px": None,
            "valid_px": int(visible.sum()),
            "correct_px": None,
            "bin_tp": None,
            "bin_fp": None,
            "bin_fn": None,
            "counts": counts,
        }

    valid = visible & (label != 254) & (label != 255)
    valid_px = int(valid.sum())
    correct_px = int((valid & (pred == label)).sum()) if valid_px else 0
    species_ref = valid & (label > 0)
    label_px = int(species_ref.sum())
    tp_px = int((species_ref & (pred == label)).sum())
    fp_px = int((valid & (pred > 0) & (label == 0)).sum())
    fn_px = int((valid & (label > 0) & (pred == 0)).sum())
    wrong_species_px = int((valid & (pred > 0) & (label > 0) & (pred != label)).sum())
    bin_tp = int((valid & (pred > 0) & (label > 0)).sum())
    bin_fp = fp_px
    bin_fn = fn_px

    f1s, ious, precs, recs = [], [], [], []
    counts = []
    for class_id, spec in by_id.items():
        if class_id == 0:
            continue
        tp = int(((pred == class_id) & (label == class_id) & valid).sum())
        fp = int(((pred == class_id) & (label != class_id) & valid).sum())
        fn = int(((label == class_id) & (pred != class_id) & valid).sum())
        px = int(((pred == class_id) & visible).sum())
        if tp + fp + fn > 0:
            f1s.append(_score_f1(tp, fp, fn))
            ious.append(_score_iou(tp, fp, fn))
        if tp + fp > 0:
            precs.append(_score_precision(tp, fp))
        if tp + fn > 0:
            recs.append(_score_recall(tp, fn))
        if px <= 0 and tp + fp + fn <= 0:
            continue
        counts.append(_row(class_id, spec, px, (tp, fp, fn)))

    return {
        "agreement": _round_metric((correct_px / valid_px) if valid_px else None),
        "agreement_species": _round_metric((tp_px / label_px) if label_px else None),
        "f1_species": _round_metric(_mean_defined(f1s)),
        "iou_species": _round_metric(_mean_defined(ious)),
        "precision_species": _round_metric(_mean_defined(precs)),
        "recall_species": _round_metric(_mean_defined(recs)),
        "binary_f1": _round_metric(_score_f1(bin_tp, bin_fp, bin_fn)),
        "binary_iou": _round_metric(_score_iou(bin_tp, bin_fp, bin_fn)),
        "pred_px": pred_px,
        "label_px": label_px,
        "tp_px": tp_px,
        "fp_px": fp_px,
        "fn_px": fn_px,
        "wrong_species_px": wrong_species_px,
        "valid_px": valid_px,
        "correct_px": correct_px,
        "bin_tp": bin_tp,
        "bin_fp": bin_fp,
        "bin_fn": bin_fn,
        "counts": counts,
    }


def _predict(req):
    import cv2
    import torch

    prepared = _inputs(req)
    model = prepared["model"]
    batch_x = prepared["batch_x"]
    aux = prepared["aux"]
    fusion = prepared["fusion"]
    rgb = prepared["rgb"]
    alpha = prepared["alpha"]
    label = prepared["label"]
    height = prepared["height"]
    width = prepared["width"]
    ds2 = prepared["ds2"]

    explain, target_class = _explain_request(req, int(model.num_classes))
    importance = None
    reduced = False
    try:
        if explain == "none":
            probs = _forward_probs(model, batch_x, aux)
        else:
            importance, probs, reduced = _importance(
                model, batch_x, aux, fusion, explain, target_class, alpha,
            )
            if probs is None:
                probs = _forward_probs(model, batch_x, aux)
        pred = probs.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise ValueError("The GPU ran out of memory.") from exc

    if pred.shape != (height, width):
        pred = cv2.resize(pred, (width, height), interpolation=cv2.INTER_NEAREST)

    pred[alpha == 0] = 0
    lut, classes, display_order = _lut()
    overlay = lut[pred]
    overlay[alpha == 0, 3] = 0
    label_overlay = None
    if label is not None:
        shown = label.copy()
        shown[alpha == 0] = 0
        label_overlay = lut[shown]
        label_overlay[alpha == 0, 3] = 0
    metrics = _tile_metrics(pred, label, alpha, classes)

    comparison_rgba = None
    if label is not None:
        comparison_rgba = _comparison(pred, label, alpha, lut)

    return {
        "ok": True,
        "width": width,
        "height": height,
        "ds2": ds2,
        "agreement": metrics["agreement"],
        "agreement_species": metrics["agreement_species"],
        "f1_species": metrics["f1_species"],
        "iou_species": metrics["iou_species"],
        "precision_species": metrics["precision_species"],
        "recall_species": metrics["recall_species"],
        "binary_f1": metrics["binary_f1"],
        "binary_iou": metrics["binary_iou"],
        "pred_px": metrics["pred_px"],
        "label_px": metrics["label_px"],
        "tp_px": metrics["tp_px"],
        "fp_px": metrics["fp_px"],
        "fn_px": metrics["fn_px"],
        "wrong_species_px": metrics["wrong_species_px"],
        "valid_px": metrics["valid_px"],
        "correct_px": metrics["correct_px"],
        "bin_tp": metrics["bin_tp"],
        "bin_fp": metrics["bin_fp"],
        "bin_fn": metrics["bin_fn"],
        "counts": metrics["counts"],
        "classes": _class_rows(classes, display_order),
        "explain": explain,
        "target_class": target_class,
        "importance_reduced": reduced,
        "importance": None if importance is None else _importance_png(importance, alpha),
        "image": _jpeg_b64(rgb),
        "overlay": _png_b64(overlay),
        "label": None if label_overlay is None else _png_b64(label_overlay),
        "comparison": None if comparison_rgba is None else _png_b64(comparison_rgba),
    }


def _batch_tile_payload(prepared, pred):
    """Scores and 256 px overlays for one class map."""
    import cv2

    height = prepared["height"]
    width = prepared["width"]
    alpha = prepared["alpha"]
    label = prepared["label"]
    if pred.shape != (height, width):
        pred = cv2.resize(pred, (width, height), interpolation=cv2.INTER_NEAREST)
    pred = np.array(pred, copy=True)
    pred[alpha == 0] = 0
    lut, classes, _display_order = _lut()
    overlay = lut[pred]
    overlay[alpha == 0, 3] = 0
    comparison = None
    if label is not None:
        comparison = _comparison(pred, label, alpha, lut)
    return {
        "ok": True,
        "width": width,
        "height": height,
        "ds2": prepared["ds2"],
        "metrics": _tile_metrics(pred, label, alpha, classes),
        "pred_webp": _webp_b64(overlay),
        "comp_webp": None if comparison is None else _webp_b64(comparison),
    }


def _start_batch(height, width):
    """CPU fallback. A 2048 px tile is much heavier than a half-res one."""
    return 4 if max(int(height), int(width)) > 1024 else 8


def _vram_budget():
    """Bytes the next forward may spend, leaving room for fragmentation."""
    import torch

    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    free, _total = torch.cuda.mem_get_info()
    reusable = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    return max(0, int((free + max(0, reusable)) * 0.85))


def _fit_stack(cap_key, height, width, available):
    """Largest stack that should fit in the memory free right now.

    The first call for a checkpoint is one tile, so its measured cost can size
    the rest. A size that ran out of memory is not tried again.
    """
    available = max(1, min(int(available), _MAX_STACK))
    budget = _vram_budget()
    if budget is None:
        return min(available, _start_batch(height, width))
    cost = _batch_cost.get(cap_key)
    if not cost:
        return 1
    fitted = max(1, budget // int(cost))
    failed = _batch_fail.get(cap_key)
    if failed:
        fitted = min(fitted, max(1, int(failed) - 1))
    return max(1, min(available, fitted, _MAX_STACK))


def _class_maps(model, prepared_rows):
    """Class ids for one stack. Skips softmax: argmax is unchanged by it.

    Raises torch.cuda.OutOfMemoryError when the stack does not fit.
    """
    import torch

    batch_x = torch.cat([row["batch_x"] for row in prepared_rows], dim=0)
    aux = None
    if prepared_rows[0]["aux"] is not None:
        aux = torch.cat([row["aux"] for row in prepared_rows], dim=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_x = batch_x.to(device)
    if aux is not None:
        aux = aux.to(device)
    logits = None
    try:
        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model._forward(batch_x, aux)
            else:
                logits = model._forward(batch_x, aux)
            return logits.argmax(dim=1).to(dtype=torch.uint8).cpu().numpy()
    finally:
        del logits, batch_x, aux


def _remember_cost(cap_key, count, delta):
    if delta is None or count <= 0:
        return
    _batch_cost[cap_key] = max(1, int(delta) // int(count))


def _run_chunk(model, chunk):
    """Class maps, plus the activation bytes this stack peaked at."""
    import torch

    if not torch.cuda.is_available():
        return _class_maps(model, chunk), None
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    maps = _class_maps(model, chunk)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return maps, max(1, int(peak) - int(before))


def _score_prepared(model, prepared, cap_key):
    """Stack as many tiles as free VRAM allows. Halve that stack after an OOM."""
    import torch

    height = prepared[0]["height"]
    width = prepared[0]["width"]
    maps_out = [None] * len(prepared)
    index = 0
    noted = None
    while index < len(prepared):
        size = _fit_stack(cap_key, height, width, len(prepared) - index)
        while True:
            chunk = prepared[index:index + size]
            try:
                maps, delta = _run_chunk(model, chunk)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if size == 1:
                    raise ValueError("The GPU ran out of memory.")
                failed = _batch_fail.get(cap_key)
                _batch_fail[cap_key] = size if failed is None else min(failed, size)
                budget = _vram_budget()
                half = max(1, size // 2)
                if budget:
                    _batch_cost[cap_key] = max(int(_batch_cost.get(cap_key) or 1), budget // half)
                size = half
        _remember_cost(cap_key, len(chunk), delta)
        if noted != size:
            budget = _vram_budget()
            if budget is None:
                room = "cpu"
            else:
                room = f"{budget / (1024 ** 3):.1f} GB free"
            print(
                f"batch stack {size} for {height}x{width} ({room})",
                file=sys.stderr,
                flush=True,
            )
            noted = size
        for offset, plane in enumerate(maps):
            maps_out[index + offset] = plane
        index += len(maps)
    return maps_out


def _predict_batch(req):
    """Class masks and scores for one tile, or for a list of tiles.

    Tiles that share a size share one forward pass. A single-tile request
    keeps the old response shape. Skips importance maps.
    """
    listed = req.get("tiles")
    single = listed is None
    tiles = [{"image": req.get("image"), "mask": req.get("mask")}] if single else list(listed)
    ckpt = req["ckpt"]
    ready = []
    errors = {}
    for index, tile in enumerate(tiles):
        one = {"ckpt": ckpt, "image": tile.get("image"), "mask": tile.get("mask")}
        try:
            ready.append((index, _inputs(one, on_device=False)))
        except (OSError, ValueError, rasterio.errors.RasterioIOError) as exc:
            errors[index] = str(exc)
    groups = {}
    for index, row in ready:
        key = (
            int(row["height"]),
            int(row["width"]),
            int(row["batch_x"].shape[1]),
            row["aux"] is not None,
        )
        groups.setdefault(key, []).append((index, row))
    results = [None] * len(tiles)
    ckpt_path = str(Path(ckpt).resolve())
    for key, rows in groups.items():
        model = rows[0][1]["model"]
        cap_key = (ckpt_path, key[0], key[1])
        try:
            maps = _score_prepared(model, [row for _index, row in rows], cap_key)
        except ValueError as exc:
            for index, _row in rows:
                errors[index] = str(exc)
            continue
        for (index, row), plane in zip(rows, maps):
            results[index] = _batch_tile_payload(row, plane)
            row["batch_x"] = None
            row["aux"] = None
    for index, message in errors.items():
        results[index] = {"ok": False, "error": message}
    if single:
        return results[0]
    return {"ok": True, "results": results}


def _split_encoded(encoded):
    """Encoder outputs are a list of maps, except DPT which also returns prefix tokens."""
    if (
        isinstance(encoded, tuple)
        and len(encoded) == 2
        and isinstance(encoded[0], (list, tuple))
        and not _is_spatial(encoded[0])
    ):
        return list(encoded[0]), encoded[1]
    if isinstance(encoded, (list, tuple)) and not _is_spatial(encoded):
        return list(encoded), None
    return [encoded], None


def _scale_label(stride):
    stride = int(stride or 1)
    if stride <= 1:
        return "full size"
    return f"1/{stride}"


def _encoder_caption(index, total, stride):
    if total <= 1:
        return "Features"
    if index == 0:
        return "Fine detail"
    if index == total - 1:
        return "Wide view"
    if index == 1:
        return "Neighborhood"
    return _scale_label(stride)


def _decoder_caption(name):
    if name == "center":
        return "Bottleneck"
    if name.startswith("mlp_stage"):
        return "Align scale"
    if name in ("fuse_stage", "fusion_block"):
        return "Fuse scales"
    if name == "aspp":
        return "Wide context"
    if name == "block1":
        return "Fine shortcut"
    if name == "block2":
        return "Merged map"
    if name == "up":
        return "Upsample"
    if name in ("psp", "fpa"):
        return "Scene context"
    if name.startswith("gau"):
        return "Attention merge"
    if name.startswith("fpn_lateral"):
        return "Add detail"
    if name.startswith("reassemble"):
        return "Reshape"
    if name.startswith("fusion_blocks"):
        return "Fuse and grow"
    if ".x_" in name or name.startswith("x_"):
        return "Dense mix"
    if name.startswith("blocks"):
        return "Grow and mix"
    return "Decoder step"


def _decoder_targets(decoder):
    """High-level blocks worth drawing. Skips normalizations and other bookkeeping."""
    import torch.nn as nn

    names = {name for name, _child in decoder.named_children()}

    def expand(name):
        module = getattr(decoder, name)
        if isinstance(module, (nn.ModuleList, nn.ModuleDict)):
            return [(f"{name}.{key}", child) for key, child in module.named_children()]
        return [(name, module)]

    if {"mlp_stage", "fuse_stage"} <= names:
        chosen = ["mlp_stage", "fuse_stage"]
    elif "aspp" in names:
        chosen = [name for name in ("aspp", "up", "block1", "block2") if name in names]
    elif {"psp", "fusion_block"} <= names:
        chosen = [name for name in ("psp", "fpn_lateral_blocks", "fusion_block") if name in names]
    elif "fpa" in names:
        chosen = [name for name in ("fpa", "gau3", "gau2", "gau1") if name in names]
    elif {"reassemble_blocks", "fusion_blocks"} <= names:
        chosen = ["reassemble_blocks", "fusion_blocks"]
    elif "blocks" in names:
        chosen = [name for name in ("center", "blocks") if name in names]
    else:
        chosen = [name for name, _child in decoder.named_children()]

    targets = []
    for name in chosen:
        targets.extend(expand(name))
    return targets


def _best_channel_subset(target, nodes):
    """Parents of a concatenated tensor: earlier maps whose channels add up to ``target``."""
    import itertools

    pool = [node for node in nodes if node.get("channels")][-12:]
    best = None
    for size in range(2, min(6, len(pool)) + 1):
        for combo in itertools.combinations(range(len(pool)), size):
            if sum(pool[index]["channels"] for index in combo) != target:
                continue
            decoders = sum(1 for index in combo if pool[index]["kind"] == "decoder")
            score = (decoders, size, sum(combo))
            if best is None or score > best[0]:
                best = (score, [pool[index]["id"] for index in combo])
    if best is None:
        return []
    return best[1]


def _resolve_tensor(tensor, nodes, by_id, role, consumed):
    if not _is_spatial(tensor):
        return []
    known = by_id.get(id(tensor))
    if known:
        return [(known, role)]
    channels = int(tensor.shape[1])
    height = int(tensor.shape[-2])
    width = int(tensor.shape[-1])
    exact = [
        node for node in nodes
        if node["channels"] == channels and node["height"] == height and node["width"] == width
    ]
    if len(exact) == 1:
        return [(exact[0]["id"], role)]
    if len(exact) > 1:
        decoders = [node for node in exact if node["kind"] == "decoder"]
        if decoders:
            return [(decoders[-1]["id"], role)]
        unused = [node for node in exact if node["id"] not in consumed]
        chosen = (unused or exact)[0]
        consumed.add(chosen["id"])
        return [(chosen["id"], role)]
    combo = _best_channel_subset(channels, nodes)
    if combo:
        return [(node_id, "fuse") for node_id in combo]
    same_place = [
        node for node in nodes
        if node["kind"] == "encoder" and node["height"] == height and node["width"] == width
    ]
    if len(same_place) == 1:
        return [(same_place[0]["id"], role)]
    same_channels = [node for node in nodes if node["channels"] == channels]
    if len(same_channels) == 1:
        return [(same_channels[0]["id"], role)]
    return []


def _prefer_stream(pairs, by_node):
    """When a block sees both a decoder map and an encoder map, the encoder is the skip."""
    kinds = [by_node.get(node_id, {}).get("kind") for node_id, _role in pairs]
    if "decoder" not in kinds or "encoder" not in kinds:
        return pairs
    ordered = []
    for (node_id, role), kind in zip(pairs, kinds):
        if kind == "encoder":
            ordered.append((node_id, "extra"))
        elif role == "fuse":
            ordered.append((node_id, role))
        else:
            ordered.append((node_id, "main"))
    return ordered


def _dense_parents(name, inputs, made, encoder_ids):
    """Unet++ block x_i_j: previous nest, same-column nests, and one encoder scale."""
    import re

    match = re.search(r"x_(\d+)_(\d+)$", name)
    if not match or len(inputs) < 2:
        return None
    depth_i, layer_i = int(match.group(1)), int(match.group(2))
    if depth_i == layer_i:
        return None
    previous = made.get(f"blocks.x_{depth_i}_{layer_i - 1}")
    if previous is None:
        return None
    pairs = [(previous, "main")]
    for idx in range(depth_i + 1, layer_i + 1):
        nested = made.get(f"blocks.x_{idx}_{layer_i}")
        if nested:
            pairs.append((nested, "fuse"))
    coarse_to_fine = list(reversed(encoder_ids))
    if layer_i + 1 < len(coarse_to_fine):
        pairs.append((coarse_to_fine[layer_i + 1], "extra"))
    return pairs


def _edge_kind(parent_kind, role):
    if parent_kind == "ndsm":
        return "aux"
    if role == "fuse":
        return "fuse"
    if role == "extra":
        return "skip" if parent_kind == "encoder" else "fuse"
    return "forward"


def _place_columns(nodes, edges):
    encoders = [node for node in nodes if node["kind"] == "encoder"]
    for index, node in enumerate(encoders):
        node["column"] = 1 + index
    decoders = {node["id"]: node for node in nodes if node["kind"] == "decoder"}
    parents = {node_id: [] for node_id in decoders}
    for edge in edges:
        if edge["to"] in parents and edge["from"] in decoders:
            parents[edge["to"]].append(edge["from"])
    memo = {}

    def depth(node_id, stack):
        if node_id in memo:
            return memo[node_id]
        if node_id in stack:
            return 0
        stack.add(node_id)
        preds = parents[node_id]
        value = 0 if not preds else 1 + max(depth(pred, stack) for pred in preds)
        stack.remove(node_id)
        memo[node_id] = value
        return value

    base = 1 + len(encoders)
    for node_id, node in decoders.items():
        node["column"] = base + depth(node_id, set())
    for node in nodes:
        if node["kind"] in ("input", "ndsm"):
            node["column"] = 0


def _trace_graph(decoder, features, prefix, height, width, alpha, fusion, scales, input_channels, has_ndsm=False):
    """Run one decoder forward and record the blocks that actually ran.

    Works for any segmentation-models-pytorch decoder: sequential skip decoders,
    parallel multi-scale fusion, atrous shortcuts, or an unseen stack of blocks.
    """
    import torch

    nodes = []
    edges = []
    seen_edges = set()
    by_id = {}
    by_node = {}

    def add_edge(source, target, role):
        if not source or source == target or source not in by_node or target not in by_node:
            return
        kind = _edge_kind(by_node[source]["kind"], role)
        key = (source, target, kind)
        if key in seen_edges:
            return
        seen_edges.add(key)
        edges.append({"from": source, "to": target, "kind": kind})

    def add_node(node, tensor=None):
        nodes.append(node)
        by_node[node["id"]] = node
        if tensor is not None and torch.is_tensor(tensor):
            by_id[id(tensor)] = node["id"]
        return node

    add_node({
        "id": "input",
        "kind": "input",
        "kicker": "Input",
        "label": "Photo",
        "scale": "full size",
        "module": "",
        "stride": 1,
        "column": 0,
        "slot": 0,
        "height": int(height),
        "width": int(width),
        "channels": 3,
        "elev_scale": None,
        "shape": [1, 3, int(height), int(width)],
    })
    if has_ndsm:
        add_node({
            "id": "ndsm",
            "kind": "ndsm",
            "kicker": "Input",
            "label": "Canopy height",
            "scale": "full size",
            "module": "ndsm",
            "stride": 1,
            "column": 0,
            "slot": 1,
            "height": int(height),
            "width": int(width),
            "channels": 1,
            "elev_scale": None,
            "shape": [1, 1, int(height), int(width)],
        })

    spatial = []
    for index, feat in enumerate(features):
        if not _is_spatial(feat):
            continue
        native_h, native_w = int(feat.shape[-2]), int(feat.shape[-1])
        native_c = int(feat.shape[1])
        if native_h == height and native_w == width and native_c <= int(input_channels):
            by_id[id(feat)] = "input"
            continue
        spatial.append((index, feat))

    total = len(spatial)
    for order, (index, feat) in enumerate(spatial):
        native_h, native_w = int(feat.shape[-2]), int(feat.shape[-1])
        stride = max(1, int(round(height / float(native_h)))) if native_h else 1
        pictures = _feature_pictures(feat, alpha)
        node = add_node({
            "id": f"enc-{index}",
            "kind": "encoder",
            "kicker": "Encoder",
            "label": _encoder_caption(order, total, stride),
            "scale": _scale_label(stride),
            "module": f"encoder.stage{index}",
            "stride": stride,
            "column": 0,
            "slot": 0,
            "height": pictures["height"],
            "width": pictures["width"],
            "channels": pictures["channels"],
            "elev_scale": scales.get(index),
            "shape": [1, pictures["channels"], pictures["height"], pictures["width"]],
            "energy": pictures["energy"],
            "pca": pictures["pca"],
            "samples": pictures["samples"],
        }, feat)
        del pictures

    if spatial:
        add_edge("input", f"enc-{spatial[0][0]}", "main")
        for previous, current in zip(spatial, spatial[1:]):
            add_edge(f"enc-{previous[0]}", f"enc-{current[0]}", "main")
    if has_ndsm and spatial:
        if fusion == "stem":
            hooked = [f"enc-{index}" for index, _feat in spatial if scales.get(index) is not None]
            targets = hooked or [f"enc-{index}" for index, _feat in spatial]
            for target in targets:
                add_edge("ndsm", target, "main")
        else:
            add_edge("ndsm", f"enc-{spatial[0][0]}", "main")

    steps = []

    def hook(name):
        def _hook(_module, args, kwargs, output):
            incoming = []
            for value in list(args) + list(kwargs.values()):
                if _is_spatial(value):
                    incoming.append(value)
            out = output
            if isinstance(output, (list, tuple)):
                spatial_out = [item for item in output if _is_spatial(item)]
                out = spatial_out[-1] if spatial_out else None
            if not _is_spatial(out):
                return
            if len(incoming) == 1 and id(incoming[0]) == id(out):
                return
            steps.append({"name": name, "inputs": incoming, "output": out})
        return _hook

    handles = []
    try:
        for name, module in _decoder_targets(decoder):
            try:
                handles.append(module.register_forward_hook(hook(name), with_kwargs=True))
            except TypeError:
                handles.append(module.register_forward_hook(
                    lambda mod, args, out, name=name: hook(name)(mod, args, {}, out)
                ))
        if prefix is None:
            decoded = decoder(features)
        else:
            decoded = decoder(features, prefix)
    finally:
        for handle in handles:
            handle.remove()

    head_in = decoded
    if isinstance(decoded, (list, tuple)):
        spatial_out = [item for item in decoded if _is_spatial(item)]
        head_in = spatial_out[-1] if spatial_out else None

    consumed = set()
    made = {}
    encoder_ids = [f"enc-{index}" for index, _feat in spatial]
    rank = {"main": 0, "extra": 1, "fuse": 2}
    for step_index, step in enumerate(steps):
        out = step["output"]
        if id(out) in by_id:
            continue
        native_h, native_w = int(out.shape[-2]), int(out.shape[-1])
        stride = max(1, int(round(height / float(native_h)))) if native_h else 1
        pictures = _feature_pictures(out, alpha)
        node_id = f"dec-{step_index}"
        made[step["name"]] = node_id
        node = add_node({
            "id": node_id,
            "kind": "decoder",
            "kicker": "Decoder",
            "label": _decoder_caption(step["name"]),
            "scale": _scale_label(stride),
            "module": step["name"],
            "stride": stride,
            "column": 0,
            "slot": 0,
            "height": pictures["height"],
            "width": pictures["width"],
            "channels": pictures["channels"],
            "elev_scale": None,
            "shape": [1, pictures["channels"], pictures["height"], pictures["width"]],
            "energy": pictures["energy"],
            "pca": pictures["pca"],
            "samples": pictures["samples"],
        }, out)
        del pictures
        override = _dense_parents(step["name"], step["inputs"], made, encoder_ids)
        if override is not None:
            pairs = override
        else:
            pairs = []
            for input_index, tensor in enumerate(step["inputs"]):
                role = "main" if input_index == 0 else "extra"
                pairs.extend(_resolve_tensor(tensor, nodes[:-1], by_id, role, consumed))
            pairs = _prefer_stream(pairs, by_node)
        pairs = sorted(pairs, key=lambda item: rank.get(item[1], 9))
        for source, source_role in pairs:
            add_edge(source, node["id"], source_role)

    if head_in is not None and _is_spatial(head_in) and id(head_in) not in by_id:
        pictures = _feature_pictures(head_in, alpha)
        native_h = pictures["height"]
        stride = max(1, int(round(height / float(native_h)))) if native_h else 1
        add_node({
            "id": "dec-out",
            "kind": "decoder",
            "kicker": "Decoder",
            "label": "Decoder output",
            "scale": _scale_label(stride),
            "module": "decoder",
            "stride": stride,
            "column": 0,
            "slot": 0,
            "height": pictures["height"],
            "width": pictures["width"],
            "channels": pictures["channels"],
            "elev_scale": None,
            "shape": [1, pictures["channels"], pictures["height"], pictures["width"]],
            "energy": pictures["energy"],
            "pca": pictures["pca"],
            "samples": pictures["samples"],
        }, head_in)
        if spatial:
            add_edge(f"enc-{spatial[-1][0]}", "dec-out", "main")

    source = by_id.get(id(head_in)) if head_in is not None and _is_spatial(head_in) else None
    if source is None:
        decoders = [node for node in nodes if node["kind"] == "decoder"]
        source = decoders[-1]["id"] if decoders else (f"enc-{spatial[-1][0]}" if spatial else "input")

    _place_columns(nodes, edges)
    for node in nodes:
        parents = []
        for edge in edges:
            if edge["to"] == node["id"] and edge["from"] not in parents:
                parents.append(edge["from"])
        node["parents"] = parents

    steps.clear()
    return {"nodes": nodes, "edges": edges, "source": source}, head_in


def _append_readout(graph, num_classes, height, width):
    """Segmentation scores and the per-pixel winner, after the decoder."""
    columns = [int(node.get("column") or 0) for node in graph["nodes"]] or [0]
    score_col = max(columns) + 1
    source = graph.get("source") or "input"
    graph["nodes"].append({
        "id": "scores",
        "kind": "scores",
        "kicker": "Segmentation",
        "label": "Species scores",
        "scale": "full size",
        "module": "segmentation_head",
        "stride": 1,
        "column": score_col,
        "slot": 0,
        "height": int(height),
        "width": int(width),
        "channels": int(num_classes),
        "elev_scale": None,
        "shape": [1, int(num_classes), int(height), int(width)],
        "parents": [source],
    })
    graph["edges"].append({"from": source, "to": "scores", "kind": "forward"})
    graph["nodes"].append({
        "id": "decision",
        "kind": "decision",
        "kicker": "Classification",
        "label": "Winning species",
        "scale": "full size",
        "module": "argmax",
        "stride": 1,
        "column": score_col + 1,
        "slot": 0,
        "height": int(height),
        "width": int(width),
        "channels": 1,
        "elev_scale": None,
        "shape": [1, int(height), int(width)],
        "parents": ["scores"],
    })
    graph["edges"].append({"from": "scores", "to": "decision", "kind": "forward"})


def _trace(req):
    """Snapshots of one forward pass, as pictures rather than tensors.

    The encoder's first map is the photo itself, and MiT leaves an empty map
    in the list the decoder still expects. Those are kept for the forward and
    left out of the pictures.
    """
    import cv2
    import torch
    import torch.nn.functional as F

    prepared = _inputs(req)
    model = prepared["model"]
    batch_x = prepared["batch_x"]
    aux = prepared["aux"]
    fusion = prepared["fusion"]
    rgb = prepared["rgb"]
    alpha = prepared["alpha"]
    label = prepared["label"]
    height = prepared["height"]
    width = prepared["width"]
    ds2 = prepared["ds2"]
    mean = prepared["mean"]
    std = prepared["std"]

    try:
        hparams = model.hparams
        architecture = str(hparams["model"])
        encoder_name = str(hparams["encoder_name"])
        trained_crop = int(hparams["img_size"] or 0)
    except Exception:
        architecture, encoder_name, trained_crop = "", "", 0

    scales = {}
    if model.elevation_fusion is not None:
        for key, value in model.elevation_fusion.get_fusion_scales().items():
            scales[int(str(key).rsplit("stage", 1)[-1])] = round(float(value), 4)

    try:
        with torch.inference_mode():
            encoded = _encode_features(model, batch_x, aux, fusion)
            raw, prefix = _split_encoded(encoded)
            has_ndsm = prepared["ndsm"] is not None
            graph, head_in = _trace_graph(
                model.model.decoder,
                raw,
                prefix,
                height,
                width,
                alpha,
                fusion,
                scales,
                int(batch_x.shape[1]),
                has_ndsm=has_ndsm,
            )
            if not _is_spatial(head_in):
                raise ValueError("This checkpoint's decoder did not return a picture.")
            logits = model.model.segmentation_head(head_in)
            if not _is_spatial(logits):
                raise ValueError("This checkpoint did not return a class score for every pixel.")
            _append_readout(graph, int(logits.shape[1]), height, width)

            encoder_stages = []
            for node in graph["nodes"]:
                if node["kind"] != "encoder":
                    continue
                encoder_stages.append({
                    "index": int(node["id"].split("-", 1)[1]),
                    "height": node["height"],
                    "width": node["width"],
                    "channels": node["channels"],
                    "stride": node["stride"],
                    "elev_scale": node.get("elev_scale"),
                    "shape": node["shape"],
                    "energy": node.get("energy"),
                    "pca": node.get("pca"),
                    "samples": node.get("samples") or [],
                })
            source = next((node for node in graph["nodes"] if node["id"] == graph.get("source")), None)
            if source is None or "pca" not in source:
                decoder_view = _feature_pictures(head_in, alpha)
            else:
                decoder_view = source
            head = _head_facts(model, head_in, logits)
            pred = logits.argmax(dim=1)[0].detach().to(torch.uint8).cpu().numpy()
            logit_min = float(logits.detach().min())
            logit_max = float(logits.detach().max())
            probs = torch.softmax(logits.float(), dim=1)
            del raw, head_in, logits

            view_h, view_w = _fit_hw(height, width, TRACE_EDGE)
            preview = F.interpolate(probs, size=(view_h, view_w), mode="bilinear", align_corners=False)
            entropy = _entropy_map(preview)
            preview_np = preview[0].detach().float().cpu().numpy()
            grid_t = F.interpolate(probs, size=(TRACE_GRID, TRACE_GRID), mode="bilinear", align_corners=False)
            grid = (grid_t[0].clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()
            del probs, preview, grid_t
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise ValueError("The GPU ran out of memory.") from exc

    if pred.shape != (height, width):
        pred = cv2.resize(pred, (width, height), interpolation=cv2.INTER_NEAREST)
    pred[alpha == 0] = 0

    view_h, view_w = _fit_hw(height, width, TRACE_EDGE)
    valid_view = _valid_for(alpha, view_h, view_w)
    small_alpha = cv2.resize(alpha, (TRACE_GRID, TRACE_GRID), interpolation=cv2.INTER_NEAREST)
    grid = np.ascontiguousarray(grid)
    grid[:, small_alpha == 0] = 0

    photo = _resize_hw(rgb, view_h, view_w, cv2.INTER_AREA)
    norm_vis = np.clip((prepared["norm_image"] + 2.0) / 4.0 * 255.0, 0, 255).astype(np.uint8)
    norm_vis = _resize_hw(norm_vis, view_h, view_w, cv2.INTER_AREA)

    ndsm_image = None
    ndsm_stats = None
    if prepared["ndsm"] is not None:
        ndsm_view = _resize_hw(prepared["ndsm"], view_h, view_w, cv2.INTER_AREA)
        ndsm_image = _terrain_png(ndsm_view, valid_view)
        sample = prepared["ndsm"][alpha > 0]
        if sample.size:
            ndsm_stats = {
                "min": round(float(np.min(sample)), 2),
                "max": round(float(np.max(sample)), 2),
                "mean": round(float(np.mean(sample)), 2),
            }

    catalog, display_order = _class_catalog(int(model.num_classes))
    probabilities = []
    for row, plane in zip(catalog, preview_np):
        probabilities.append({
            "id": row["id"],
            "name": row["name"],
            "color": row["color"],
            "image": _tint_png(plane, _ink(row["rgb"]), valid_view),
        })

    lut, _classes, _order = _lut()
    pred_view = _resize_hw(pred, view_h, view_w, cv2.INTER_NEAREST)
    label_image = None
    label_grid = None
    agreement = None
    agreement_species = None
    if label is not None:
        shown = label.copy()
        shown[alpha == 0] = 0
        shown_view = _resize_hw(shown, view_h, view_w, cv2.INTER_NEAREST)
        label_image = _color_overlay(shown_view, valid_view, lut)
        label_small = cv2.resize(label, (TRACE_GRID, TRACE_GRID), interpolation=cv2.INTER_NEAREST)
        label_small = label_small.astype(np.uint8, copy=True)
        label_small[small_alpha == 0] = np.uint8(TRACE_EMPTY)
        label_grid = {
            "width": TRACE_GRID,
            "height": TRACE_GRID,
            "data": base64.b64encode(np.ascontiguousarray(label_small).tobytes()).decode("ascii"),
        }
        valid = (alpha > 0) & (label != 254) & (label != 255)
        if int(valid.sum()) > 0:
            agreement = float((pred[valid] == label[valid]).mean())
        species = valid & (label > 0)
        if int(species.sum()) > 0:
            agreement_species = float((pred[species] == label[species]).mean())

    by_id = {row["id"]: row for row in catalog}
    counts = []
    visible = alpha > 0
    scored = None if label is None else (alpha > 0) & (label != 254) & (label != 255)
    for class_id, row in by_id.items():
        if class_id == 0:
            continue
        px = int(((pred == class_id) & visible).sum())
        item = {
            "id": class_id,
            "name": row["name"],
            "color": row["color"],
            "px": px,
        }
        if scored is not None:
            item["tp"] = int(((pred == class_id) & (label == class_id) & scored).sum())
            item["fp"] = int(((pred == class_id) & (label != class_id) & scored).sum())
            item["fn"] = int(((label == class_id) & (pred != class_id) & scored).sum())
        if px <= 0 and not (item.get("tp") or item.get("fp") or item.get("fn")):
            continue
        counts.append(item)

    entropy_u = _unit_interval(entropy, valid_view, stretch=False)
    return {
        "ok": True,
        "width": width,
        "height": height,
        "native_width": prepared["native_width"],
        "native_height": prepared["native_height"],
        "ds2": ds2,
        "fusion": fusion,
        "architecture": architecture,
        "encoder": encoder_name,
        "trained_crop": trained_crop,
        "num_classes": int(model.num_classes),
        "in_channels": {str(key): int(value) for key, value in prepared["channels"].items()},
        "input_shape": [int(value) for value in batch_x.shape],
        "aux_shape": None if aux is None else [int(value) for value in aux.shape],
        "norm": {
            "rgb_mean": [round(float(value), 4) for value in mean],
            "rgb_std": [round(float(value), 4) for value in std],
            "ndsm_mean": round(float(prepared["ndsm_mean"]), 4),
            "ndsm_std": round(float(prepared["ndsm_std"]), 4),
        },
        "image": _jpeg_b64(photo),
        "normalized": _jpeg_b64(norm_vis),
        "ndsm": ndsm_image,
        "ndsm_stats": ndsm_stats,
        "graph": {"nodes": graph["nodes"], "edges": graph["edges"]},
        "encoder_stages": encoder_stages,
        "decoder": {
            "height": decoder_view["height"],
            "width": decoder_view["width"],
            "channels": decoder_view["channels"],
            "stride": max(1, int(round(height / float(decoder_view["height"])))) if decoder_view["height"] else 1,
            "shape": [1, decoder_view["channels"], decoder_view["height"], decoder_view["width"]],
            "energy": decoder_view["energy"],
            "pca": decoder_view["pca"],
            "samples": decoder_view["samples"],
        },
        "head": head,
        "logit_min": None if math.isnan(logit_min) else round(logit_min, 3),
        "logit_max": None if math.isnan(logit_max) else round(logit_max, 3),
        "classes": [{"id": row["id"], "name": row["name"], "color": row["color"]} for row in catalog],
        "display_order": [int(class_id) for class_id in display_order],
        "probabilities": probabilities,
        "entropy": _importance_png(entropy_u, valid_view),
        "prob_grid": {
            "width": TRACE_GRID,
            "height": TRACE_GRID,
            "classes": int(grid.shape[0]),
            "data": base64.b64encode(grid.tobytes()).decode("ascii"),
        },
        "label_grid": label_grid,
        "overlay": _color_overlay(pred_view, valid_view, lut),
        "label": label_image,
        "counts": counts,
        "agreement": agreement,
        "agreement_species": agreement_species,
    }


def main():
    for line in sys.stdin:
        text = line.strip()
        if not text:
            continue
        try:
            req = json.loads(text)
            cmd = req.get("cmd")
            if cmd == "ping":
                payload = {"ok": True}
            elif cmd == "release":
                payload = _release()
            elif cmd == "predict":
                payload = _predict(req)
            elif cmd == "predict_batch":
                payload = _predict_batch(req)
            elif cmd == "trace":
                payload = _trace(req)
            else:
                payload = {"ok": False, "error": "unknown command"}
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            payload = {"ok": False, "error": str(exc)}
        _reply(payload)


if __name__ == "__main__":
    main()
