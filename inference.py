#!/usr/bin/env python3
"""Apply a trained model to whole orthophoto GeoTiffs.

A full orthophoto is far too large for one forward pass, so it is cut into
overlapping patches, predicted in batches and written back into a georeferenced
GeoTiff. Two ways of dealing with the overlap:

  hard_crop (default)  every patch keeps only its inner region, so each output
                       pixel comes from exactly one patch. Fast, constant RAM.
  gaussian             predictions are blended across the overlap with a weight
                       that fades out towards the patch border. Smoother seams,
                       needs a rolling accumulator of several GB.

Empty (no-data) patches are skipped based on one downsampled read of the image,
which saves most of the work on orthophotos with large empty margins.

Models trained with the nDSM as an extra input (``ndsm`` in ``model.in_channels``)
also get it here: set ``data.dsm_path`` and ``data.dtm_path`` to one DSM and one
DTM GeoTiff per orthophoto, and the nDSM under every patch is computed on the fly.

Run:
    python inference.py exp_name=<run directory in lightning_logs/>
    python inference.py data.batch_size=16 data.patch_overlap=128
    python inference.py blending=gaussian
    python inference.py exp_name=<ndsm run> data.dsm_path=[<dsm.tif>] data.dtm_path=[<dtm.tif>]
"""

import gc
import logging
import os
import time
import cv2
import torch
import hydra
import rasterio
import numpy as np

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds
from rasterio.windows import bounds as window_bounds

# Suppress the per-read EPSG:2056 CRS mismatch warning from GDAL -- harmless for inference
logging.getLogger('rasterio._env').setLevel(logging.ERROR)

from models.semseg_plm import SegmentationModel
from utils import utils
from utils.transform_utils import get_transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# Shared utilities
# =============================================================================

def _patch_starts(size: int, patch_size: int, overlap: int) -> list:
    """
    Start positions along one axis with full coverage.

    Regular stride from 0, plus a final patch clamped to the far edge.
    This ensures no uncovered margins -- unlike the centered approach in
    calculate_patches(), which can leave a margin of up to step/2 on each side.
    """
    if patch_size >= size:
        return [0]
    step = patch_size - overlap
    starts = list(range(0, size - patch_size, step))
    last = size - patch_size
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def _blending_weight(size: int, overlap: int) -> np.ndarray:
    """2D weight map for smooth patch blending.

    Flat 1.0 in the center, linear ramp from 0→1 over overlap/2 pixels at
    each edge. This reaches exactly 0 at the patch border, so when a
    neighbouring patch enters the overlap zone it starts contributing with
    weight ≈0 and ramps up -- no discontinuity.
    """
    ramp = overlap // 2
    w = np.ones(size, dtype=np.float32)
    if ramp > 0:
        w[:ramp] = np.linspace(0, 1, ramp, endpoint=False)
        w[-ramp:] = np.linspace(1, 0, ramp, endpoint=False)
    return np.outer(w, w)


def _make_out_meta(image_path: str, nodata: int) -> dict:
    with rasterio.open(image_path) as src:
        out_meta = src.meta.copy()
    out_meta.update(
        count=1, dtype='uint8', nodata=nodata, compress='DEFLATE',
        tiled=True, blockxsize=512, blockysize=512,
    )
    return out_meta


def _make_unc_meta(image_path: str) -> dict:
    """Meta for the uncertainty map: 1 float32 band.

    band 1 = normalized entropy  (-sum_c p log p) / log K  in [0, 1]  (model-agnostic)
    """
    with rasterio.open(image_path) as src:
        out_meta = src.meta.copy()
    out_meta.update(
        count=1, dtype='float32', nodata=None, compress='DEFLATE',
        tiled=True, blockxsize=512, blockysize=512,
    )
    return out_meta


def _prefilter_overview(image_path: str, xs: list, ys: list):
    """Single downsampled read to determine which patches contain image data."""
    n_x, n_y = len(xs), len(ys)
    with rasterio.open(image_path) as src:
        overview = src.read(1, out_shape=(n_y, n_x),
                            resampling=rasterio.enums.Resampling.average,
                            out_dtype='float32')
    return overview


def _valid_area_m2(n_valid: int, patch_size: int, overlap: int, res) -> float:
    """Area covered by valid patches (one step×step cell per patch)."""
    step = patch_size - overlap
    return n_valid * step * step * abs(res[0] * res[1])


DSM_NODATA = -32767  # fallback when the raster has no nodata tag, as in NeophyteDataset


class ElevationReader:
    """nDSM (= DSM - DTM) under a window of the RGB orthophoto, prepared as in training.

    DSM and DTM are separate GeoTiffs, usually coarser than the RGB, so the window is
    matched by map coordinates, not by pixel offsets. The steps follow
    ``NeophyteDataset._load_sample`` plus ``ndsm_transforms``: DTM resampled onto the
    DSM grid, difference taken, nodata set to 0, bilinear resize to the RGB patch
    size, then normalised with the training mean/std.

    The rasters are opened lazily, so the reader can be pickled into DataLoader workers.
    """

    def __init__(self, dsm_path, dtm_path, rgb_path, mean, std):
        self.dsm_path, self.dtm_path = dsm_path, dtm_path
        self.mean, self.std = float(mean), float(std)
        with rasterio.open(rgb_path) as src:
            self.rgb_transform, rgb_crs = src.transform, src.crs
        for path in (dsm_path, dtm_path):
            with rasterio.open(path) as src:
                if src.crs != rgb_crs:
                    raise ValueError(f"CRS of {path} ({src.crs}) differs from the orthophoto ({rgb_crs})")
        self._dsm = self._dtm = None

    def __call__(self, x: int, y: int, size: int) -> np.ndarray:
        if self._dsm is None:
            self._dsm, self._dtm = rasterio.open(self.dsm_path), rasterio.open(self.dtm_path)
        left, bottom, right, top = window_bounds(Window(x, y, size, size), self.rgb_transform)

        # # DSM at its native resolution, DTM resampled onto the same grid
        dsm_win = from_bounds(left, bottom, right, top, transform=self._dsm.transform)
        shape = (max(1, round(dsm_win.height)), max(1, round(dsm_win.width)))
        dsm_nodata = self._dsm.nodata if self._dsm.nodata is not None else DSM_NODATA
        dtm_nodata = self._dtm.nodata if self._dtm.nodata is not None else DSM_NODATA
        dsm = self._dsm.read(1, window=dsm_win, out_shape=shape, resampling=Resampling.nearest,
                             boundless=True, fill_value=dsm_nodata).astype(np.float32)
        dtm = self._dtm.read(1, window=from_bounds(left, bottom, right, top, transform=self._dtm.transform),
                             out_shape=shape, resampling=Resampling.bilinear,
                             boundless=True, fill_value=dtm_nodata).astype(np.float32)

        nodata = (dsm == dsm_nodata) | np.isnan(dsm) | (dtm == dtm_nodata) | np.isnan(dtm)
        ndsm = dsm - dtm
        ndsm[nodata] = 0.0  # no height info where either is nodata
        ndsm = cv2.resize(ndsm, (size, size), interpolation=cv2.INTER_LINEAR)
        return ((ndsm - self.mean) / self.std)[np.newaxis].astype(np.float32)  # (1, H, W)


def _make_elevation_reader(cfg, model, idx: int, image_path: str):
    """ElevationReader for the idx-th orthophoto if the model needs the nDSM, else None.

    Fails loudly instead of predicting without the input the model was trained on."""
    aux_inputs = [k for k in model.hparams.in_channels if k != 'image']
    if not aux_inputs:
        return None
    if aux_inputs != ['ndsm']:
        raise NotImplementedError(f"inference.py supports only 'ndsm' as auxiliary input, "
                                  f"the model expects {aux_inputs}")
    dsm_paths, dtm_paths = cfg.data.get('dsm_path'), cfg.data.get('dtm_path')
    if not dsm_paths or not dtm_paths:
        raise ValueError("The model was trained with the nDSM: set data.dsm_path and "
                         "data.dtm_path (one GeoTiff per entry of data.image_dir).")
    n_images = len(cfg.data.image_dir)
    if len(dsm_paths) != n_images or len(dtm_paths) != n_images:
        raise ValueError(f"data.dsm_path and data.dtm_path need one entry per orthophoto "
                         f"({n_images}), got {len(dsm_paths)} and {len(dtm_paths)}.")
    return ElevationReader(dsm_paths[idx], dtm_paths[idx], image_path,
                           cfg.data.ndsm_norm.mean, cfg.data.ndsm_norm.std)


def _predict(model, imgs, ndsm=None):
    """Softmax probabilities; routes the nDSM to the model as in training
    (concatenated to RGB or fused into the encoder, depending on fusion_mode)."""
    inputs = {'image': imgs}
    if ndsm is not None:
        inputs['ndsm'] = ndsm
    x, aux = model.prepare_inputs(inputs)
    return model(x, aux)


def _print_stats(width, height, res, n_valid, n_total, area_m2, elapsed, out_path, blending):
    area_ha = area_m2 / 10_000
    full_area_ha = width * height * abs(res[0] * res[1]) / 10_000
    print(f"  Full image:    {width} × {height} px  |  {full_area_ha:.1f} ha")
    print(f"  Valid patches: {n_valid} of {n_total}  |  {area_ha:.1f} ha")
    print(f"  Blending:      {blending}")
    print(f"  Time:          {elapsed:.1f} s  |  {area_m2/elapsed:,.0f} m²/s  |  {elapsed/area_ha:.1f} s/ha")
    print(f"  Output:        {out_path}")


# =============================================================================
# Hard-crop inference (default) -- DataLoader + direct windowed write
# =============================================================================

class GeoTiffPatchDataset(Dataset):
    """
    Yields fixed-size patches from a single GeoTiff with full edge coverage.

    Edge-aware cropping: the overlap-crop (ho pixels on each side) is suppressed
    at image boundaries, so the output window reaches the actual image edge.
    All-zero (no-data) patches are prefiltered at init time.
    """

    def __init__(self, path: str, patch_size: int, overlap: int, transform=None, elevation=None):
        assert overlap % 2 == 0, "overlap must be even"
        self.path = path
        self.patch_size = patch_size
        self.ho = overlap // 2
        self.transform = transform
        self.elevation = elevation  # ElevationReader or None

        with rasterio.open(path) as src:
            self.width = src.width
            self.height = src.height
            self.res = src.res

        xs = _patch_starts(self.width, patch_size, overlap)
        ys = _patch_starts(self.height, patch_size, overlap)

        overview = _prefilter_overview(path, xs, ys)
        n_x, n_y = len(xs), len(ys)
        self.n_total_patches = n_x * n_y
        self.positions = [
            (xs[i], ys[j])
            for j in range(n_y)
            for i in range(n_x)
            if overview[j, i] > 0
        ]

    def _crops(self, x: int, y: int) -> tuple:
        ho = self.ho
        return (
            0 if x == 0 else ho,
            0 if x + self.patch_size >= self.width else ho,
            0 if y == 0 else ho,
            0 if y + self.patch_size >= self.height else ho,
        )

    @property
    def valid_area_m2(self) -> float:
        return _valid_area_m2(len(self.positions), self.patch_size, self.ho * 2, self.res)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        x, y = self.positions[idx]
        if not hasattr(self, '_src'):
            self._src = rasterio.open(self.path)
        patch = np.stack(
            self._src.read([1, 2, 3], window=Window(x, y, self.patch_size, self.patch_size)),
            axis=-1,
        )
        if self.transform:
            patch = self.transform(image=patch)['image']
        cl, cr, ct, cb = self._crops(x, y)
        out = dict(image=patch, x=x, y=y, cl=cl, cr=cr, ct=ct, cb=cb)
        if self.elevation is not None:
            out['ndsm'] = torch.from_numpy(self.elevation(x, y, self.patch_size))
        return out


def infer_image_hard_crop(image_path: str, out_path: str, model, cfg, elevation=None) -> tuple:
    """Hard-crop inference: argmax per patch, direct windowed write."""
    patch_size = cfg.data.patch_size
    overlap = cfg.data.patch_overlap
    transform = get_transforms(cfg.data.test_transforms)

    dataset = GeoTiffPatchDataset(image_path, patch_size, overlap, transform=transform,
                                  elevation=elevation)
    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=(device.type == 'cuda'),
    )

    area_m2 = dataset.valid_area_m2
    out_meta = _make_out_meta(image_path, cfg.data.nodata)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Optional uncertainty output: single normalized-entropy band (model-agnostic, one
    # forward pass -> cheap; no MC sampling / extra bands).
    want_unc = bool(cfg.get('uncertainty', False))
    unc_path = out_path.replace('_pred.tif', '_uncertainty.tif') if want_unc else None
    unc_dst = None

    t0 = time.perf_counter()
    autocast_ctx = torch.autocast(device.type, dtype=torch.float16) if device.type == 'cuda' else torch.autocast('cpu', enabled=False)

    with rasterio.open(out_path, 'w', **out_meta) as dst, torch.inference_mode(), autocast_ctx:
        if want_unc:
            unc_dst = rasterio.open(unc_path, 'w', **_make_unc_meta(image_path))
            unc_dst.set_band_description(1, 'normalized_entropy')  # (-sum_c p log p)/log K in [0,1]
        try:
            for batch in tqdm(loader, desc=Path(image_path).name, unit='batch'):
                imgs = batch['image'].to(device, non_blocking=True)
                ndsm = batch['ndsm'].to(device, non_blocking=True) if 'ndsm' in batch else None
                probs = _predict(model, imgs, ndsm)
                if want_unc:
                    # normalized predictive entropy from the single-forward softmax probs
                    norm_ent = (-(probs * torch.log(probs + 1e-10)).sum(dim=1)).float().cpu().numpy()
                    norm_ent /= float(np.log(probs.shape[1]))
                preds = torch.argmax(probs, dim=1).cpu().numpy().astype(np.uint8)

                for i in range(len(preds)):
                    x  = batch['x'][i].item()
                    y  = batch['y'][i].item()
                    cl = batch['cl'][i].item()
                    cr = batch['cr'][i].item()
                    ct = batch['ct'][i].item()
                    cb = batch['cb'][i].item()

                    row_slice = slice(ct, patch_size - cb if cb else None)
                    col_slice = slice(cl, patch_size - cr if cr else None)
                    cropped = preds[i][row_slice, col_slice]
                    out_h, out_w = cropped.shape
                    win = Window(x + cl, y + ct, out_w, out_h)
                    dst.write(cropped, 1, window=win)
                    if want_unc:
                        unc_dst.write(norm_ent[i][row_slice, col_slice], 1, window=win)
        finally:
            if unc_dst is not None:
                unc_dst.close()

    elapsed = time.perf_counter() - t0
    _print_stats(dataset.width, dataset.height, dataset.res,
                 len(dataset), dataset.n_total_patches, area_m2, elapsed, out_path,
                 'hard_crop')
    return elapsed, area_m2


# =============================================================================
# Gaussian-blended inference -- strip-based rolling accumulator
# =============================================================================

def infer_image_gaussian(image_path: str, out_path: str, model, cfg, elevation=None) -> tuple:
    """
    Gaussian-blended inference using a rolling accumulator.

    Processes patches row by row. A buffer of height=patch_size × width holds
    the weighted logit accumulation. After each row of patches, the fully-resolved
    top rows are written to disk and the buffer shifts down.

    RAM for the accumulators: (num_classes + 1) × patch_size × width × 4 bytes.
    For 7 classes, 1024 patch_size, 231k width: ~7.6 GB.
    """
    patch_size = cfg.data.patch_size
    overlap = cfg.data.patch_overlap
    batch_size = cfg.data.batch_size
    transform = get_transforms(cfg.data.test_transforms)
    num_classes = model.hparams.num_classes

    with rasterio.open(image_path) as src:
        width, height = src.width, src.height
        res = src.res

    xs = _patch_starts(width, patch_size, overlap)
    ys = _patch_starts(height, patch_size, overlap)

    overview = _prefilter_overview(image_path, xs, ys)
    n_x, n_y = len(xs), len(ys)
    n_total = n_x * n_y
    n_valid = 0

    # Gaussian weight map
    weight_map = _blending_weight(patch_size, overlap)

    # Rolling accumulator: height = patch_size, width = full image width
    acc_logits = np.zeros((num_classes, patch_size, width), dtype=np.float32)
    acc_weight = np.zeros((patch_size, width), dtype=np.float32)
    acc_y_start = 0

    acc_gb = (acc_logits.nbytes + acc_weight.nbytes) / 1e9
    print(f"  Accumulator RAM: {acc_gb:.1f} GB")

    out_meta = _make_out_meta(image_path, cfg.data.nodata)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    t0 = time.perf_counter()
    autocast_ctx = torch.autocast(device.type, dtype=torch.float16) if device.type == 'cuda' else torch.autocast('cpu', enabled=False)

    with rasterio.open(out_path, 'w', **out_meta) as dst, \
         rasterio.open(image_path) as src, \
         torch.inference_mode(), autocast_ctx:

        for j in tqdm(range(n_y), desc=Path(image_path).name, unit='row'):
            y = ys[j]

            # Valid patch x-positions for this row
            row_xs = [xs[i] for i in range(n_x) if overview[j, i] > 0]
            n_valid += len(row_xs)

            # Process patches in batches
            for b_start in range(0, max(len(row_xs), 1), batch_size):
                b_xs = row_xs[b_start:b_start + batch_size]
                if not b_xs:
                    break

                # Read and transform patches
                images = []
                for px in b_xs:
                    patch = np.stack(
                        src.read([1, 2, 3], window=Window(px, y, patch_size, patch_size)),
                        axis=-1,
                    )
                    images.append(transform(image=patch)['image'])

                imgs = torch.stack(images).to(device, non_blocking=True)
                ndsm = None
                if elevation is not None:
                    ndsm = torch.from_numpy(np.stack([elevation(px, y, patch_size) for px in b_xs]))
                    ndsm = ndsm.to(device, non_blocking=True)
                probs = _predict(model, imgs, ndsm)
                probs_np = probs.cpu().numpy()  # B x C x H x W

                # Accumulate weighted probabilities
                rel_y = y - acc_y_start
                for k, px in enumerate(b_xs):
                    acc_logits[:, rel_y:rel_y + patch_size, px:px + patch_size] += \
                        probs_np[k] * weight_map[np.newaxis, :, :]
                    acc_weight[rel_y:rel_y + patch_size, px:px + patch_size] += weight_map

            # How many top rows of the buffer are fully resolved?
            if j + 1 < n_y:
                done_rows = ys[j + 1] - acc_y_start
            else:
                done_rows = min(patch_size, height - acc_y_start)

            if done_rows > 0:
                # Blend: weighted average → argmax
                safe_w = np.maximum(acc_weight[:done_rows], 1e-8)
                blended = acc_logits[:, :done_rows, :] / safe_w[np.newaxis, :, :]
                pred = np.argmax(blended, axis=0).astype(np.uint8)
                pred[acc_weight[:done_rows] == 0] = 0  # no-data areas → background

                dst.write(pred, 1, window=Window(0, acc_y_start, width, done_rows))

                # Shift buffer: keep the unresolved overlap rows at the top
                remaining = patch_size - done_rows
                if remaining > 0:
                    acc_logits[:, :remaining, :] = acc_logits[:, done_rows:done_rows + remaining, :]
                    acc_weight[:remaining, :] = acc_weight[done_rows:done_rows + remaining, :]
                acc_logits[:, remaining:, :] = 0
                acc_weight[remaining:, :] = 0
                acc_y_start += done_rows

    elapsed = time.perf_counter() - t0
    area_m2 = _valid_area_m2(n_valid, patch_size, overlap, res)
    _print_stats(width, height, res, n_valid, n_total, area_m2, elapsed, out_path,
                 'gaussian')
    return elapsed, area_m2


# =============================================================================
# Main
# =============================================================================

def _require_exp_name(argv, script):
    """Fail early with a readable message instead of a Hydra interpolation error."""
    if not any(a.startswith("exp_name=") for a in argv):
        import sys
        sys.exit(f"Missing exp_name. Usage: python {script} exp_name=<run directory in "
                 f"lightning_logs/>\nAvailable runs: ls lightning_logs/")


@hydra.main(version_base=None, config_path="configs", config_name="inference")
def main(cfg: DictConfig) -> None:
    # Resolve checkpoint path
    ckpt_paths = utils.getListOfFiles(cfg.ckpt_dir)
    ckpt_paths.sort(key=utils.natural_keys)
    matching = [p for p in ckpt_paths if cfg.model.ckpt_type in p]
    assert matching, f"No checkpoint matching '{cfg.model.ckpt_type}' in {cfg.ckpt_dir}"
    ckpt_path = matching[-1]
    print(f"Device:     {device}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Blending:   {cfg.blending}\n")

    model = SegmentationModel.load_from_checkpoint(ckpt_path).to(device)
    model.eval()

    if cfg.get('uncertainty', False) and cfg.blending == 'gaussian':
        print("WARNING: uncertainty output is only supported for blending=hard_crop; "
              "no uncertainty map will be written.")

    infer_fn = infer_image_gaussian if cfg.blending == 'gaussian' else infer_image_hard_crop

    out_base = os.path.join(os.getcwd(), cfg.out_dir, cfg.exp_name)
    os.makedirs(out_base, exist_ok=True)

    total_time, total_area = 0.0, 0.0
    for idx, image_path in enumerate(cfg.data.image_dir):
        stem = Path(image_path).stem
        out_path = os.path.join(out_base, f"{stem}_pred.tif")
        print(f"Processing: {image_path}")
        elevation = _make_elevation_reader(cfg, model, idx, image_path)
        t, a = infer_fn(image_path, out_path, model, cfg, elevation=elevation)
        total_time += t
        total_area += a
        torch.cuda.empty_cache()
        gc.collect()

    if len(cfg.data.image_dir) > 1:
        print(
            f"\nTotal: {total_time:.1f}s  |  {total_area/10_000:.2f} ha  |  "
            f"{total_area/total_time:,.0f} m²/s"
        )


if __name__ == '__main__':
    import sys

    _require_exp_name(sys.argv[1:], "inference.py")
    main()
