#!/usr/bin/env python3
"""LoRA on PlantCLEF 2024: adapt the frozen plant backbone to drone windows (arm B).

Same windows, labels and split as plantclef_probe.py, so the result is directly
comparable to its "linear PlantCLEF" row (arm A): the only difference is that the
backbone's features may now change through LoRA adapters.

Model: PlantCLEF 2024 ViT-B/14 reg4 "onlyclassifier_then_all", backbone frozen, LoRA
(peft) on the attention qkv of all 12 blocks, new 7-class head (background + our 6).
The head's species rows start from PlantCLEF's own rows for our species, the
background row from zero. LoRA and head are trained.

Windows of one size (--size, set it to the probe's best size), resized to 518 on the
GPU; label = dominant species >= --pos-thr of the valid pixels, background if the
window holds no species pixel, skipped in between (plantclef_probe.window_label).
  train  labelled training-site tiles + --bg-fraction background-only tiles, all their
         windows; class-balanced cross-entropy (as the probe's head); flips, 90 degree
         rotations and colour jitter
  val    the val folders, sampled the same way; picks the best epoch (F1-avg-wo0)
  test   every window of every held-out tile, best epoch, metrics_from_confusion

    python plantclef_lora.py --size 256
    python plantclef_lora.py --size 256 --max-tiles 20 --epochs 1 --out-dir results/plantclef_lora_debug   # quick check

Outputs in --out-dir: best.pt (LoRA + head), log.csv, summary.md. They are derived from
the confidential dataset: keep them on the node. Needs peft (personal sam2 venv).
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.windows import Window

from plantclef_probe import PC_DIR, PC_MODEL, SPECIES, confusion, split_tiles, window_label
from sam_oracle import N_CLS, load_config


class TileMasks(torch.utils.data.Dataset):
    """One tile -> reference mask (N_CLS = not counted: ignore or nodata)."""

    def __init__(self, tiles):
        self.tiles = tiles

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, i):
        row = self.tiles.iloc[i]
        with rasterio.open(row["mask_path"]) as src:
            mask = src.read(1).astype(np.int64)
        with rasterio.open(row["img_path"]) as src:
            alpha = src.read(4) if src.count >= 4 else np.full(mask.shape, 255, np.uint8)
        valid = (mask != 254) & (mask != 255) & (alpha > 0)
        return np.where(valid, mask, N_CLS), i


def window_index(tiles, size, pos_thr, workers):
    """(tile, y0, x0, label) of every scored window of an exclusive grid."""
    rows = []
    loader = torch.utils.data.DataLoader(TileMasks(tiles), batch_size=None, num_workers=workers)
    for gt, i in loader:
        gt = gt.numpy()
        for y0 in range(0, gt.shape[0] - size + 1, size):
            for x0 in range(0, gt.shape[1] - size + 1, size):
                label = window_label(gt[y0:y0 + size, x0:x0 + size], pos_thr)
                if label >= 0:
                    rows.append((int(i), y0, x0, label))
    return pd.DataFrame(rows, columns=["tile", "y0", "x0", "label"])


class Windows(torch.utils.data.Dataset):
    """One window -> RGB uint8 (3, size, size) read from the GeoTIFF, label."""

    def __init__(self, tiles, index, size):
        self.tiles, self.index, self.size = tiles, index, size

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        w = self.index.iloc[k]
        with rasterio.open(self.tiles.iloc[int(w["tile"])]["img_path"]) as src:
            img = src.read([1, 2, 3], window=Window(int(w["x0"]), int(w["y0"]), self.size, self.size))
        return torch.from_numpy(img), int(w["label"])


def to_input(img, res, mean, std, augment):
    """uint8 batch -> normalised 518 input on the GPU; flips, rot90 and colour jitter when training."""
    import torch.nn.functional as F

    x = img.cuda(non_blocking=True).float().div(255)
    if augment:
        if np.random.rand() < 0.5:
            x = x.flip(3)
        x = torch.rot90(x, int(np.random.randint(4)), dims=(2, 3))
        b = x.shape[0]
        # # per-image brightness, contrast and saturation in [0.8, 1.2]
        f = lambda: torch.empty(b, 1, 1, 1, device=x.device).uniform_(0.8, 1.2)
        x = x * f()
        m = x.mean((1, 2, 3), keepdim=True)
        x = (x - m) * f() + m
        g = x.mean(1, keepdim=True)
        x = (x - g) * f() + g
    x = F.interpolate(x, size=(res, res), mode="bicubic", antialias=True, align_corners=False)
    return (x.clamp(0, 1) - mean) / std


@torch.no_grad()
def predict(model, loader, res, mean, std):
    model.eval()
    preds, labels = [], []
    for img, y in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            preds.append(model(to_input(img, res, mean, std, False)).argmax(1).cpu())
        labels.append(y)
    return torch.cat(labels).numpy(), torch.cat(preds).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights-dir", default="../weights/plantclef2024/pretrained_models")
    parser.add_argument("--stats-csv", default="../data/Neophytes/stats_imagewise.csv")
    parser.add_argument("--split-yaml", default="configs/data/split_cv3.yaml")
    parser.add_argument("--classes-yaml", default="configs/data/neophytes_names_colors.yaml")
    parser.add_argument("--out-dir", default="results/plantclef_lora")
    parser.add_argument("--size", type=int, default=256, help="window size in tile px (use the probe's best)")
    parser.add_argument("--pos-thr", type=float, default=0.01)
    parser.add_argument("--bg-fraction", type=float, default=0.10)
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=32, help="LoRA scaling alpha")
    parser.add_argument("--lr", type=float, default=2e-4, help="LoRA learning rate")
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-tiles", type=int, default=0, help="tiles per split for a quick check (0 = all)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import timm
    from peft import LoraConfig, get_peft_model

    from utils.eval_utils import metrics_from_confusion

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    classes = load_config(args.classes_yaml)
    names = [classes[c]["name"] for c in range(N_CLS)]

    # %% windows per split (same split and labels as plantclef_probe.py)
    tiles = split_tiles(args.stats_csv, args.split_yaml, args.bg_fraction, args.max_tiles, rng)
    loaders, counts = {}, {}
    for split, t in tiles.items():
        index = window_index(t, args.size, args.pos_thr, args.workers)
        counts[split] = np.bincount(index["label"], minlength=N_CLS)
        print(f"{split}: {len(t)} tiles, {len(index)} windows, per class {counts[split].tolist()}", flush=True)
        loaders[split] = torch.utils.data.DataLoader(Windows(t, index, args.size), batch_size=args.batch,
                                                     shuffle=split == "train", num_workers=args.workers,
                                                     pin_memory=True, drop_last=split == "train")

    # %% model: PlantCLEF backbone + LoRA on qkv + new 7-class head (species rows from PlantCLEF's head)
    cid_to_spid = [line.strip() for line in open(os.path.join(args.weights_dir, "class_mapping.txt"))]
    spid_to_sp = pd.read_csv(os.path.join(args.weights_dir, "species_id_to_name.txt"), sep=";", quoting=1,
                             dtype={"species_id": str}).set_index("species_id")["species"].to_dict()
    binomial = {" ".join(spid_to_sp[s].split()[:2]): i for i, s in enumerate(cid_to_spid)}
    target_idx = torch.tensor([binomial[s] for s in SPECIES])
    model = timm.create_model(PC_MODEL, pretrained=False, num_classes=len(cid_to_spid),
                              checkpoint_path=os.path.join(args.weights_dir, PC_DIR, "model_best.pth.tar"))
    old_w, old_b = model.head.weight.data.clone(), model.head.bias.data.clone()
    model.reset_classifier(N_CLS)
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
        model.head.weight[1:] = old_w[target_idx]
        model.head.bias[1:] = old_b[target_idx]
    model.set_grad_checkpointing(True)
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.1,
                                             target_modules=["qkv"], modules_to_save=["head"]))
    model.print_trainable_parameters()
    model.cuda()
    cfg = timm.data.resolve_model_data_config(model.base_model.model)
    mean = torch.tensor(cfg["mean"], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device="cuda").view(1, 3, 1, 1)
    res = cfg["input_size"][-1]

    lora = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    head = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n]
    opt = torch.optim.AdamW([{"params": lora, "lr": args.lr}, {"params": head, "lr": args.lr_head}],
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, args.lr_head], pct_start=0.05,
                                                total_steps=args.epochs * len(loaders["train"]))
    # # class-balanced weights, as sklearn's class_weight="balanced" in the probe
    n = counts["train"].astype(float)
    weight = torch.tensor(np.where(n > 0, n.sum() / (N_CLS * np.maximum(n, 1)), 0.0), dtype=torch.float32).cuda()
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)

    # %% train, best epoch on val
    log, best = [], -1.0
    for epoch in range(args.epochs):
        model.train()
        t0, losses = time.time(), []
        for step, (img, y) in enumerate(loaders["train"]):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(model(to_input(img, res, mean, std, True)).float(), y.cuda())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
            if (step + 1) % 200 == 0:
                print(f"  epoch {epoch} step {step + 1}/{len(loaders['train'])} loss {np.mean(losses[-200:]):.4f}",
                      flush=True)
        y_val, p_val = predict(model, loaders["val"], res, mean, std)
        f1_val = metrics_from_confusion(confusion(y_val, p_val))["F1-avg-wo0"]
        log.append(dict(epoch=epoch, train_loss=np.mean(losses), val_f1_wo0=f1_val, minutes=(time.time() - t0) / 60))
        print(log[-1], flush=True)
        pd.DataFrame(log).to_csv(os.path.join(args.out_dir, "log.csv"), index=False)
        if f1_val > best:
            best = f1_val
            torch.save({k: v for k, v in model.state_dict().items() if "lora_" in k or "head" in k},
                       os.path.join(args.out_dir, "best.pt"))

    # %% test with the best epoch
    model.load_state_dict(torch.load(os.path.join(args.out_dir, "best.pt")), strict=False)
    y_te, p_te = predict(model, loaders["test"], res, mean, std)
    m = metrics_from_confusion(confusion(y_te, p_te))
    table = pd.DataFrame({"LoRA PlantCLEF": {**{names[c]: m["F1"][c] for c in range(1, N_CLS)},
                                             "F1-avg-wo0": m["F1-avg-wo0"], "P-avg-wo0": m["Precision-avg-wo0"],
                                             "R-avg-wo0": m["Recall-avg-wo0"]}}).T
    best_epoch = int(pd.DataFrame(log)["val_f1_wo0"].idxmax())
    md = [f"# LoRA on PlantCLEF, cv3 held-out sites, {args.size} px windows\n",
          f"LoRA r={args.rank}, alpha={args.alpha} on qkv; lr {args.lr} (LoRA) / {args.lr_head} (head); "
          f"{args.epochs} epochs, best epoch {best_epoch} (val F1-avg-wo0 {best:.3f}). Same windows, labels and "
          f"split as plantclef_probe.py; compare with its 'linear PlantCLEF' row at {args.size} px.\n",
          "Test windows: " + ", ".join(f"{names[c]} {counts['test'][c]}" for c in range(N_CLS)) + ".\n",
          table.to_markdown(floatfmt=".3f")]
    with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
