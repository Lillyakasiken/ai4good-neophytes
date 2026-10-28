"""Time one validation step, split into forward, metric updates and phenology.

The cv3 validation pass is one 2048 px tile per sample, batch 2, MiT-b2, bf16.
This loads that val set and times 20 batches three ways, plus the same forward
with the original explicit attention, so the table shows where the step goes.

    python scripts/profile_val.py
    python scripts/profile_val.py --batches 5 --batch-size 1

Needs a free GPU. A 2048 px MiT-b2 forward does not fit beside a training run.
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    import torch
    from hydra import compose, initialize
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    from datasets.neophyte_dataset import NeophyteDataset
    from models.semseg_plm import SegmentationModel, set_efficient_mit_attention
    from utils.eval_utils import restore_compact_labels
    from utils.transform_utils import get_transforms

    free_gb = torch.cuda.mem_get_info()[0] / 1024**3
    if free_gb < 18:
        sys.exit(f"Need about 18 GiB free on the GPU for a 2048 px forward, found {free_gb:.1f}. "
                 f"Wait until the training run has finished.")

    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="train", overrides=[
            "data=neophytes_split_cv3_train_512",
            "model=model_neophytes_mit_b2",
        ])
    OmegaConf.resolve(cfg)

    val_tf = get_transforms(cfg.data.val_transforms, cfg.data.transforms_additional_targets)
    dataset = NeophyteDataset(
        cfg.data.image_dir.val, cfg.data.classes, transform=val_tf,
        dataset_stats=cfg.data.dataset_stats, value_mapping=cfg.data.value_mapping,
        ignore_index=cfg.data.ignore_index, load_auxiliary=True)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers else None,
        pin_memory=True)
    print(f"val tiles: {len(dataset)}  batch: {args.batch_size}  "
          f"timing {args.batches} batches after {args.warmup} warmup")

    model = SegmentationModel(
        cfg.model.model_name, cfg.model.encoder_name, cfg.model.img_size, cfg.model.num_classes,
        cfg.model.lr, in_channels=OmegaConf.to_container(cfg.model.in_channels),
        ignore_index=cfg.data.ignore_index, loss=cfg.model.loss,
        patch_2_img_size=False, fusion_mode=cfg.model.fusion_mode,
    ).cuda().eval()

    def move(batch):
        return {k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}

    def step(batch, metrics, phenology):
        # inference mode, like Lightning's validation loop. Without it the forward
        # keeps a graph and the timing is not the one a val pass actually pays.
        batch = move(batch)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            restore_compact_labels(batch)
            masks = batch["mask"]
            inputs, aux = model.prepare_inputs(batch)
            logits = model._forward(inputs, aux)
            loss = model.criterion(logits, masks.long())
            if metrics:
                preds = torch.argmax(logits, dim=1)
                for name in ("acc", "iou", "f1", "auc", "f1_per_class", "stats"):
                    getattr(model, f"val_{name}").update(preds.detach() if name != "auc" else logits.detach(),
                                                         masks.detach() if name != "auc" else masks.detach().long())
                if phenology:
                    model._accumulate_phenology("val", preds, masks, batch["phenology"])
        return loss

    def time_variant(name, fn):
        it = iter(loader)
        for _ in range(args.warmup):
            fn(next(it))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        gpu_ms, wall_ms = [], []
        for _ in range(args.batches):
            batch = next(it)
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            wall0 = time.perf_counter()
            start.record()
            fn(batch)
            end.record()
            end.synchronize()
            gpu_ms.append(start.elapsed_time(end))
            wall_ms.append((time.perf_counter() - wall0) * 1000)
        peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"{name:28}  gpu {sum(gpu_ms)/len(gpu_ms):8.1f} ms   "
              f"wall {sum(wall_ms)/len(wall_ms):8.1f} ms   peak {peak:8.0f} MiB")
        return sum(gpu_ms) / len(gpu_ms)

    print(f"{'variant':28}  {'gpu mean':>12}   {'wall mean':>12}   peak")
    set_efficient_mit_attention(False)
    explicit = time_variant("forward, explicit attention", lambda b: step(b, False, False))
    set_efficient_mit_attention(True)
    forward = time_variant("forward, fused attention", lambda b: step(b, False, False))
    with_metrics = time_variant("forward + metrics", lambda b: step(b, True, False))
    full = time_variant("forward + metrics + phenology", lambda b: step(b, True, True))

    print()
    print(f"fused attention vs explicit: {explicit - forward:+.0f} ms/batch "
          f"({explicit / max(forward, 1e-6):.2f}x the forward)")
    print(f"metric updates:              {with_metrics - forward:+.0f} ms/batch")
    print(f"phenology counts:            {full - with_metrics:+.0f} ms/batch")
    print(f"full step:                   {full:.0f} ms/batch")
    per_pass = full / 1000 * (len(dataset) / args.batch_size)
    print(f"full cv3 val pass at this rate, {len(dataset)} tiles: {per_pass / 60:.1f} min")

    # One batch under the profiler, so the kernel list is there if the totals surprise.
    from torch.profiler import ProfilerActivity, profile
    batch = next(iter(loader))
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        step(batch, True, True)
        torch.cuda.synchronize()
    print()
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))


if __name__ == "__main__":
    main()
