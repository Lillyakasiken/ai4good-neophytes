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

from common import IGNORE_COLORS, load_names  # noqa: E402

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
    u8 = np.clip(np.round(heat * 255.0), 0, 255).astype(np.uint8)
    rgba = np.zeros((u8.shape[0], u8.shape[1], 4), dtype=np.uint8)
    rgba[..., 0] = u8
    rgba[..., 1] = u8
    rgba[..., 2] = u8
    rgba[..., 3] = np.where(alpha > 0, np.uint8(255), np.uint8(0))
    return _png_b64(rgba)


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


def _predict(req):
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
    height, width = rgb.shape[:2]
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
    model_h, model_w = height, width
    model_rgb = rgb

    torch.set_float32_matmul_precision("medium")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load(ckpt, device)
    channels = _as_dict(model.in_channels or {"image": 3})
    fusion = str(model.fusion_mode or "concat")

    image = model_rgb.astype(np.float32) / 255.0
    image = (image - mean) / std
    image_t = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0)

    ndsm = None
    needs_ndsm = any(key == "ndsm" for key in channels)
    if needs_ndsm:
        ndsm = _read_ndsm(image_path, model_h, model_w)
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

    explain, target_class = _explain_request(req, int(model.num_classes))
    batch_x = batch_x.to(device)
    if aux is not None:
        aux = aux.to(device)
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
    agreement = None
    agreement_species = None
    if label is not None:
        shown = label.copy()
        shown[alpha == 0] = 0
        label_overlay = lut[shown]
        label_overlay[alpha == 0, 3] = 0
        valid = (alpha > 0) & (label != 254) & (label != 255)
        if int(valid.sum()) > 0:
            agreement = float((pred[valid] == label[valid]).mean())
        species = valid & (label > 0)
        if int(species.sum()) > 0:
            agreement_species = float((pred[species] == label[species]).mean())

    by_id = {spec["id"]: spec for spec in classes}
    counts = []
    visible = alpha > 0
    for class_id, spec in by_id.items():
        if class_id == 0:
            continue
        px = int(((pred == class_id) & visible).sum())
        if px <= 0:
            continue
        counts.append({
            "id": class_id,
            "name": spec["name"],
            "color": spec["color"],
            "px": px,
        })

    return {
        "ok": True,
        "width": width,
        "height": height,
        "ds2": ds2,
        "agreement": agreement,
        "agreement_species": agreement_species,
        "counts": counts,
        "classes": _class_rows(classes, display_order),
        "explain": explain,
        "target_class": target_class,
        "importance_reduced": reduced,
        "importance": None if importance is None else _importance_png(importance, alpha),
        "image": _jpeg_b64(rgb),
        "overlay": _png_b64(overlay),
        "label": None if label_overlay is None else _png_b64(label_overlay),
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
            else:
                payload = {"ok": False, "error": "unknown command"}
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            payload = {"ok": False, "error": str(exc)}
        _reply(payload)


if __name__ == "__main__":
    main()
