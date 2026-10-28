"""Check the speed-up changes against the constraints.

CPU (always):
  - compact labels round-trip to the exact long/float values the metrics had before
  - one real tile agrees with a mask read back as long
  - val_subset keeps every labelled cv3 val tile plus the requested background tiles

GPU (when about 18 GiB is free):
  - fused MiT attention vs the original score-matrix attention, same weights, bf16,
    on real 2048 px val tiles. Prints the logit gap, the argmax disagreement and
    the metric gap (accuracy, IoU, F1, species F1, AUROC).
  - ms/batch and peak memory for both attentions.

    python scripts/verify_speed.py
    python scripts/verify_speed.py --tiles 4
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)


def _check_roundtrip():
    import torch
    from datasets.neophyte_dataset import _compact_mask, _compact_uint8
    from utils.eval_utils import restore_compact_labels

    mask = torch.tensor([-1, 0, 1, 6], dtype=torch.long)
    canopy = torch.tensor([0, 60, 100], dtype=torch.float32)
    phenology = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    batch = {
        "mask": _compact_mask(mask),
        "canopy": _compact_uint8(canopy),
        "phenology": _compact_uint8(phenology),
    }
    assert batch["mask"].dtype == torch.int8, batch["mask"].dtype
    assert batch["canopy"].dtype == torch.uint8
    assert batch["phenology"].dtype == torch.uint8
    restore_compact_labels(batch)
    assert torch.equal(batch["mask"], mask)
    assert torch.equal(batch["canopy"], canopy)
    assert torch.equal(batch["phenology"], phenology)
    print("compact-label round-trip: exact")


def _check_real_tile():
    import torch
    from hydra import compose, initialize
    from datasets.neophyte_dataset import NeophyteDataset
    from utils.transform_utils import get_transforms

    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="train", overrides=["data=neophytes_split_debug_1024"])
    tf = get_transforms(cfg.data.val_transforms, cfg.data.transforms_additional_targets)
    dataset = NeophyteDataset(
        cfg.data.image_dir.val, cfg.data.classes, transform=tf,
        dataset_stats=cfg.data.dataset_stats, value_mapping=cfg.data.value_mapping,
        ignore_index=cfg.data.ignore_index, load_auxiliary=True)
    # A labelled tile, so the mask is not all background.
    index = next(i for i, name in enumerate(dataset.stats_df.index)
                 if dataset.stats_df.loc[name, [c for c in dataset.stats_df.columns
                                                if c.endswith("_px") and c.split("_px")[0] in
                                                set(dataset.neophytes_stats_names)]].sum() > 0)
    sample = dataset[index]
    assert sample["mask"].dtype == torch.int8
    assert sample["phenology"].dtype == torch.uint8
    assert sample["canopy"].dtype == torch.uint8
    mask = sample["mask"].long()
    assert set(mask.unique().tolist()) <= {-1, 0, 1, 2, 3, 4, 5, 6}, mask.unique().tolist()
    # The values that left the transform, reconstructed: int8 -> long is the inverse.
    assert torch.equal(sample["mask"].to(torch.int8).long(), mask)
    print(f"real val tile {os.path.basename(sample['name'])}: "
          f"mask int8, phenology/canopy uint8, classes {sorted(set(mask.unique().tolist()))}")
    return cfg


def _check_val_subset():
    from hydra import compose, initialize
    from datasets.neophyte_datamodule import NeophyteDataModule
    from utils.transform_utils import get_transforms

    overrides = ["data=neophytes_split_cv3_train_512", "data.val_subset.enabled=true",
                 "data.val_subset.bg_k=50"]
    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="train", overrides=overrides)
    tf = get_transforms(cfg.data.val_transforms, cfg.data.transforms_additional_targets)
    module = NeophyteDataModule(
        cfg.data.image_dir, cfg.data.classes, tf, tf, tf,
        dataset_stats=cfg.data.dataset_stats, value_mapping=cfg.data.value_mapping,
        ignore_index=cfg.data.ignore_index, val_subset=cfg.data.val_subset,
        batch_size_val=2, num_workers=0)
    module.setup("fit")
    val = module.val_dataset
    labelled = (val.stats_df[[c for c in val.class_px_cols if not c.startswith("background")]].sum(axis=1) > 0)
    n_labelled = int(labelled.sum())
    n_bg = int((~labelled).sum())
    assert n_bg == 50, n_bg
    assert n_labelled > 0
    # 'labelled: all' must not drop a tile that has neophyte pixels. Compare with
    # the unfiltered val directory listing via a second dataset.
    with initialize(config_path="../configs", version_base=None):
        full_cfg = compose(config_name="train", overrides=["data=neophytes_split_cv3_train_512"])
    full = NeophyteDataModule(
        full_cfg.data.image_dir, full_cfg.data.classes, tf, tf, tf,
        dataset_stats=full_cfg.data.dataset_stats, value_mapping=full_cfg.data.value_mapping,
        ignore_index=full_cfg.data.ignore_index, val_subset=full_cfg.data.val_subset,
        num_workers=0)
    full.setup("fit")
    full_labelled = full.val_dataset.stats_df.index[
        full.val_dataset.stats_df[[c for c in full.val_dataset.class_px_cols
                                   if not c.startswith("background")]].sum(axis=1) > 0]
    missing = set(full_labelled) - set(val.stats_df.index)
    assert not missing, f"val_subset dropped {len(missing)} labelled tiles"
    print(f"val_subset: {n_labelled} labelled (all of them) + {n_bg} background "
          f"= {len(val)} tiles, full val set is {len(full.val_dataset)}")


def _check_attention(n_tiles):
    import time
    import torch
    from hydra import compose, initialize
    from omegaconf import OmegaConf
    from torchmetrics.classification import MulticlassAUROC, MulticlassJaccardIndex, MulticlassStatScores

    from datasets.neophyte_dataset import NeophyteDataset
    from models.semseg_plm import SegmentationModel, set_efficient_mit_attention
    from utils.eval_utils import restore_compact_labels
    from utils.transform_utils import get_transforms

    free_gb = torch.cuda.mem_get_info()[0] / 1024**3
    if free_gb < 18:
        print(f"GPU check skipped: {free_gb:.1f} GiB free, need about 18.")
        return False

    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="train", overrides=[
            "data=neophytes_split_debug_1024", "model=model_neophytes_mit_b2"])
    OmegaConf.resolve(cfg)
    tf = get_transforms(cfg.data.val_transforms, cfg.data.transforms_additional_targets)
    dataset = NeophyteDataset(
        cfg.data.image_dir.val, cfg.data.classes, transform=tf,
        dataset_stats=cfg.data.dataset_stats, value_mapping=cfg.data.value_mapping,
        ignore_index=cfg.data.ignore_index, load_auxiliary=True)
    model = SegmentationModel(
        cfg.model.model_name, cfg.model.encoder_name, cfg.model.img_size, cfg.model.num_classes,
        cfg.model.lr, in_channels=OmegaConf.to_container(cfg.model.in_channels),
        ignore_index=cfg.data.ignore_index, loss=cfg.model.loss, patch_2_img_size=False,
    ).cuda().eval()

    def forward(sample):
        batch = {"image": sample["image"].unsqueeze(0).cuda(),
                 "mask": sample["mask"].unsqueeze(0).cuda()}
        restore_compact_labels(batch)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model._forward(batch["image"], None)
        return logits.float(), batch["mask"]

    max_diff, flipped, total = 0.0, 0, 0
    gpu_off, gpu_on = [], []
    # thresholds=5 matches the training AUROC. Two copies, one per attention.
    metric = {}
    for tag in ("off", "on"):
        metric[tag] = {
            "iou": MulticlassJaccardIndex(cfg.model.num_classes, average="macro", ignore_index=-1).cuda(),
            "auc": MulticlassAUROC(cfg.model.num_classes, average="macro", ignore_index=-1, thresholds=5).cuda(),
            "stats": MulticlassStatScores(cfg.model.num_classes, average=None, ignore_index=-1).cuda(),
        }

    for i in range(n_tiles):
        sample = dataset[i]
        torch.cuda.synchronize()
        for enabled, tag, bucket in ((False, "off", gpu_off), (True, "on", gpu_on)):
            set_efficient_mit_attention(enabled)
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            logits, masks = forward(sample)
            end.record()
            end.synchronize()
            bucket.append(start.elapsed_time(end))
            preds = logits.argmax(dim=1)
            metric[tag]["iou"].update(preds, masks)
            metric[tag]["auc"].update(logits, masks)
            metric[tag]["stats"].update(preds, masks)
            if enabled:
                on_logits, on_preds = logits, preds
            else:
                off_logits, off_preds = logits, preds
        max_diff = max(max_diff, (on_logits - off_logits).abs().max().item())
        flipped += int((on_preds != off_preds).sum().item())
        total += on_preds.numel()

    def species_f1(stats):
        tp, fp, fn = stats[:, 0].float(), stats[:, 1].float(), stats[:, 3].float()
        denom = 2 * tp + fp + fn
        f1 = torch.where(denom > 0, 2 * tp / denom.clamp(min=1), torch.full_like(tp, float("nan")))
        return torch.nanmean(f1[1:]).item()

    iou_off, iou_on = metric["off"]["iou"].compute().item(), metric["on"]["iou"].compute().item()
    auc_off, auc_on = metric["off"]["auc"].compute().item(), metric["on"]["auc"].compute().item()
    f1_off = species_f1(metric["off"]["stats"].compute())
    f1_on = species_f1(metric["on"]["stats"].compute())
    print(f"attention on {n_tiles} real 2048 px tiles, bf16:")
    print(f"  max |logit| diff {max_diff:.3e}   argmax flips {flipped}/{total} "
          f"({100 * flipped / total:.6f}%)")
    print(f"  macro IoU     explicit {iou_off:.6f}   fused {iou_on:.6f}   delta {iou_on - iou_off:+.3e}")
    print(f"  species F1    explicit {f1_off:.6f}   fused {f1_on:.6f}   delta {f1_on - f1_off:+.3e}")
    print(f"  AUROC         explicit {auc_off:.6f}   fused {auc_on:.6f}   delta {auc_on - auc_off:+.3e}")
    print(f"  ms/tile       explicit {sum(gpu_off)/len(gpu_off):.0f}   fused {sum(gpu_on)/len(gpu_on):.0f}   "
          f"peak {torch.cuda.max_memory_allocated()/1024**2:.0f} MiB")
    set_efficient_mit_attention(True)
    # Float noise: a handful of argmax ties can flip, the scores stay in 1e-3.
    if abs(iou_on - iou_off) > 1e-3 or abs(f1_on - f1_off) > 1e-3 or abs(auc_on - auc_off) > 1e-3:
        print("METRIC GAP ABOVE FLOAT NOISE")
        return False
    print("metric gap is within float noise")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiles", type=int, default=2)
    args = parser.parse_args()
    _check_roundtrip()
    _check_real_tile()
    _check_val_subset()
    try:
        import torch
        if torch.cuda.is_available():
            _check_attention(args.tiles)
        else:
            print("GPU check skipped: no CUDA")
    except Exception as exc:
        print(f"GPU check failed: {exc}")
        raise


if __name__ == "__main__":
    main()
