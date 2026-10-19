#!/usr/bin/env python3
"""Score a trained U-Net like the PlantCLEF probe: window F1 on full val and test.

The probe/LoRA rows are window F1 and the U-Net's test.py numbers are pixel F1, so
their val -> test gaps are not comparable. Here the U-Net's predicted mask is turned
into one label per window with the same rule as plantclef_probe.window_label, on the
same exclusive grid and the same windows:

    reference  window_label(reference mask): dominant species >= --pos-thr of the valid
               pixels, background if no species pixel, skipped in between or if < 50% valid
    predicted  dominant predicted species if it covers >= --pos-thr of the valid pixels,
               else background (no window is skipped on the prediction side)

Valid = not ignore (254/255) and alpha > 0, as in the probe. Both splits as in the
probe: val = every tile of the val folders of the training sites (the probe's
val_full), test = every tile of the held-out sites. Pixel F1 (as test.py: ignore
excluded, alpha not) is reported too, as a check against test.py.

    python baseline_windows.py exp_name=<run> log_dir=<lightning_logs of that run>
    python baseline_windows.py ... +max_batches=5   # quick check

Output: results/baseline_windows/<exp_name>/summary.md. Derived from the confidential
dataset: keep it on the node.
"""

import os

import hydra
import numpy as np
from omegaconf import DictConfig

SIZES = (256, 512, 1024)
POS_THR = 0.01  # as plantclef_probe.py --pos-thr


def pred_window_label(pred, n_cls, pos_thr):
    """Window label of a predicted mask (invalid pixels = n_cls): dominant species or background."""
    px = np.bincount(pred.ravel(), minlength=n_cls + 1)[:n_cls]
    if px[1:].sum() == 0:
        return 0
    c = int(px[1:].argmax()) + 1
    return c if px[c] >= pos_thr * px.sum() else 0


@hydra.main(version_base=None, config_path="configs", config_name="test")
def main(cfg: DictConfig) -> None:
    import pandas as pd
    import rasterio
    import torch
    from tqdm import tqdm

    from datasets.neophyte_datamodule import NeophyteDataModule
    from models.semseg_plm import SegmentationModel
    from plantclef_probe import confusion, window_label
    from sam_oracle import N_CLS
    from utils.eval_utils import metrics_from_confusion
    from utils.transform_utils import get_transforms
    from utils.utils import getListOfFiles, natural_keys

    torch.set_float32_matmul_precision('medium')
    device = torch.device('cuda')
    out_dir = os.path.join("results", "baseline_windows", cfg.exp_name)
    os.makedirs(out_dir, exist_ok=True)

    # %% data (as test.py: no augmentation, no sampling) and checkpoint
    t = get_transforms(cfg.data.test_transforms, cfg.data.transforms_additional_targets)
    dm = NeophyteDataModule(cfg.data.image_dir, cfg.data.classes, t, t, t, dataset_stats=cfg.data.dataset_stats,
                            value_mapping=cfg.data.value_mapping, ignore_index=cfg.data.ignore_index,
                            batch_size_train=cfg.data.batch_size_test, batch_size_val=cfg.data.batch_size_test,
                            batch_size_test=cfg.data.batch_size_test, num_workers=cfg.data.num_workers)
    dm.prepare_data()
    dm.setup()
    ckpts = sorted(getListOfFiles(cfg.ckpt_path), key=natural_keys)
    ckpt = [s for s in ckpts if cfg.ckpt_type in s][-1]
    model = SegmentationModel.load_from_checkpoint(ckpt, weights_only=False).to(device).eval()
    model.set_patch_2_img_size(False)
    print("checkpoint:", ckpt, flush=True)
    names = [cfg.data.classes[i].name for i in range(N_CLS)]

    # %% one pass per split: pixel confusion + window confusion per size
    res, max_batches = {}, cfg.get("max_batches", 0)
    for split, loader, files in [("val", dm.val_dataloader(), dm.val_dataset.image_files),
                                 ("test", dm.test_dataloader(), dm.test_dataset.image_files)]:
        conf_px = np.zeros((N_CLS, N_CLS), np.int64)
        y = {s: [] for s in SIZES}
        p = {s: [] for s in SIZES}
        k = 0
        with torch.no_grad():
            for n, batch in enumerate(tqdm(loader, desc=split)):
                if max_batches and n >= max_batches:
                    break
                inputs, aux = model.prepare_inputs(batch)
                # # full precision, as test.py, so the pixel F1 reproduces its number
                probs = model(inputs.to(device), None if aux is None else aux.to(device))
                preds = probs.argmax(1).cpu().numpy()
                masks = batch['mask'].numpy()
                for b in range(len(preds)):
                    mask, pred = masks[b], preds[b]
                    valid = mask != cfg.data.ignore_index
                    conf_px += confusion(mask[valid], pred[valid])
                    with rasterio.open(files[k]) as src:
                        alpha = src.read(4) if src.count >= 4 else np.full(mask.shape, 255, np.uint8)
                    k += 1
                    valid &= alpha > 0
                    gt, pr = np.where(valid, mask, N_CLS), np.where(valid, pred, N_CLS)
                    for s in SIZES:
                        for y0 in range(0, gt.shape[0] - s + 1, s):
                            for x0 in range(0, gt.shape[1] - s + 1, s):
                                label = window_label(gt[y0:y0 + s, x0:x0 + s], POS_THR)
                                if label >= 0:
                                    y[s].append(label)
                                    p[s].append(pred_window_label(pr[y0:y0 + s, x0:x0 + s], N_CLS, POS_THR))
        assert max_batches or k == len(files), f"{split}: {k} tiles scored, {len(files)} files"
        res[split] = {"px": metrics_from_confusion(conf_px), "n_tiles": k,
                      **{s: metrics_from_confusion(confusion(np.array(y[s]), np.array(p[s]))) for s in SIZES},
                      **{f"n{s}": np.bincount(y[s], minlength=N_CLS) for s in SIZES}}
        print(f"{split}: {k} tiles, pixel F1-avg-wo0 {res[split]['px']['F1-avg-wo0']:.3f}, window F1-avg-wo0 "
              + ", ".join(f"{s}: {res[split][s]['F1-avg-wo0']:.3f}" for s in SIZES), flush=True)

    # %% summary
    row = lambda m: {**{names[c]: m["F1"][c] for c in range(1, N_CLS)}, "F1-avg-wo0": m["F1-avg-wo0"],
                     "P-avg-wo0": m["Precision-avg-wo0"], "R-avg-wo0": m["Recall-avg-wo0"]}
    gap = pd.DataFrame({u: {"full val F1-avg-wo0": res["val"][u]["F1-avg-wo0"],
                            "test F1-avg-wo0": res["test"][u]["F1-avg-wo0"]}
                        for u in ["px", *SIZES]}).T
    gap["gap"] = gap["full val F1-avg-wo0"] - gap["test F1-avg-wo0"]
    gap.index = ["pixel F1"] + [f"window F1, {s} px" for s in SIZES]
    md = [f"# U-Net scored like the PlantCLEF probe: {cfg.exp_name}\n",
          f"Checkpoint `{os.path.basename(ckpt)}`. Full val: {res['val']['n_tiles']} tiles (val folders of the "
          f"training sites); test: {res['test']['n_tiles']} tiles (held-out sites). Window rule as "
          f"plantclef_probe.window_label (pos-thr {POS_THR:.0%}); compare the window rows with the probe's and "
          f"LoRA's generalisation-gap tables at the same size.\n",
          "## Generalisation gap\n", gap.to_markdown(floatfmt=".3f")]
    for u, title in [("px", "pixel F1"), *[(s, f"window F1, {s} px") for s in SIZES]]:
        tab = pd.DataFrame({sp: row(res[sp][u]) for sp in ["val", "test"]}).T
        md += [f"\n## {title}\n", tab.to_markdown(floatfmt=".3f")]
        if u != "px":
            md.append("\nTest windows: " + ", ".join(f"{names[c]} {res['test'][f'n{u}'][c]}" for c in range(N_CLS))
                      + "; full val windows: " + ", ".join(f"{names[c]} {res['val'][f'n{u}'][c]}"
                                                           for c in range(N_CLS)) + ".")
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    import sys

    from test import _auto_data_config, _require_exp_name

    _require_exp_name(sys.argv[1:], "baseline_windows.py")
    if not any(a.startswith("data=") for a in sys.argv[1:]):
        exp_arg = next(a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("exp_name="))
        auto_cfg = _auto_data_config(exp_arg)
        if auto_cfg:
            sys.argv.append(f"data={auto_cfg}")
    main()
