"""Evaluation of a trained segmentation model.

Everything quantitative is derived from confusion matrices: one for the whole
test set, and optionally one per site, per acquisition month and per
cross-validation fold. From a confusion matrix follow overall accuracy,
precision, recall, IoU and F1 -- per class and averaged.

The headline metric of this project is the **CV-fold averaged F1**, overall and
per class: mean ± std of the per-fold F1 over the five folds (see
``evaluate_cv.py`` for the aggregation across five separately trained models).
Averaging over folds is what makes a number comparable between methods, because
a single fold is dominated by which sites happen to be held out.

Alongside that, ``save_model_predictions`` writes qualitative figures
(image / reference / prediction) for a handful of test tiles.
"""

import json
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from matplotlib.colors import ListedColormap, PowerNorm
from sklearn.metrics import ConfusionMatrixDisplay
from torchmetrics.classification import BinaryAUROC, MulticlassAUROC, MulticlassConfusionMatrix
from tqdm import tqdm

from utils.utils import convert_ndarray_to_list, denormalize


# =============================================================================
# Helpers
# =============================================================================
def extract_site(path):
    """Site (e.g. ``Basel_2_1``) from a tile path ``.../<site>/<split>/images/<tile>``."""
    return os.path.normpath(path).split(os.sep)[-4]


def extract_month(path):
    """Acquisition month from a tile filename ``YYYY_MM_DD_...`` as a 2-char string."""
    return os.path.normpath(path).split(os.sep)[-1][5:7]


# Stage band values 1, 2, 3. Value 0 (unknown) and 4 (excluded) are not scored.
PHENO_STAGE_KEYS = ("non_flowering", "flowering", "fruiting")


def slug_stage_name(name):
    """``non-flowering`` -> ``non_flowering``, so the name is safe in a W&B key."""
    return str(name).strip().replace("-", "_").replace(" ", "_")


def phenology_stage_keys(names=None):
    """Three stage keys in band order (values 1, 2, 3)."""
    if names is None:
        return list(PHENO_STAGE_KEYS)
    keys = [slug_stage_name(n) for n in names]
    if len(keys) != len(PHENO_STAGE_KEYS):
        raise ValueError(f"Expected {len(PHENO_STAGE_KEYS)} phenology stages, got {keys}")
    return keys


def class_stats_names(classes):
    """Class key per class index, using ``stats_name`` (``B_davidii``) when present."""
    names = []
    for i in sorted(classes.keys(), key=lambda k: int(k)):
        entry = classes[i]
        names.append(str(entry.stats_name if "stats_name" in entry else entry.name))
    return names


def restore_compact_labels(batch):
    """Cast compact labels back to the dtypes the metrics have always received.

    ``NeophyteDataset`` ships the mask as int8 and canopy/phenology as uint8 so a
    2048 px tile is not copied around as int64. ``ignore_index`` is -1, which does
    not fit in uint8, so the mask stays int8 and ``.long()`` restores -1 exactly.
    Canopy and phenology values are non-negative and small, so uint8 round-trips
    through ``.float()`` and ``.long()``. Call this after the batch is on device:
    the cast then runs on the GPU and the host-to-device copy stays compact.
    """
    mask = batch.get("mask")
    if torch.is_tensor(mask) and mask.dtype != torch.long:
        batch["mask"] = mask.long()
    phenology = batch.get("phenology")
    if torch.is_tensor(phenology) and phenology.dtype != torch.long:
        batch["phenology"] = phenology.long()
    canopy = batch.get("canopy")
    if torch.is_tensor(canopy) and canopy.dtype != torch.float32:
        batch["canopy"] = canopy.float()
    return batch


def update_phenology_counts(counts, preds, masks, phenology):
    """Add label pixels to ``counts[species, stage, predicted class]``.

    ``counts`` is a CPU long tensor of shape ``[C, 3, C]``. Stage value 1 is index 0.
    Ignored labels, background, and stages other than 1..3 are skipped. The three
    stages are non-flowering, flowering and fruiting.
    """
    num_classes = counts.shape[0]
    preds = preds.reshape(-1)
    masks = masks.reshape(-1).long()
    phenology = phenology.reshape(-1).long()
    valid = (
        (masks >= 1) & (masks < num_classes)
        & (phenology >= 1) & (phenology <= len(PHENO_STAGE_KEYS))
        & (preds >= 0) & (preds < num_classes)
    )
    if not bool(valid.any()):
        return
    c = masks[valid]
    s = phenology[valid] - 1
    p = preds[valid].long()
    flat = c * (len(PHENO_STAGE_KEYS) * num_classes) + s * num_classes + p
    # bincount stays on the prediction device; only the  C*3*C  bin vector is copied
    inc = torch.bincount(flat, minlength=counts.numel()).to(dtype=torch.long)
    # .tolist() leaves inference mode (validation steps run under it), so the
    # accumulator, which lives outside that mode, can be updated.
    counts.view(-1).add_(torch.tensor(inc.detach().cpu().tolist(), dtype=torch.long))


def phenology_breakdown(counts, stage_names, class_names, min_px):
    """Shares per species and stage from a ``[C, 3, C]`` count tensor.

    For each cell the three shares sum to 1: recall (predicted as that species),
    miss_to_background, and confused (predicted as some other species). Cells with
    fewer than ``min_px`` label pixels have ``below_min_px`` set; their shares are
    still filled in when the cell is non-empty, so a caller can decide what to log.
    """
    if torch.is_tensor(counts):
        counts = counts.detach().cpu().numpy()
    counts = np.asarray(counts)
    stage_names = list(stage_names)
    species = [(i, class_names[i]) for i in range(1, counts.shape[0])]
    cells = {}
    for class_id, name in species:
        cells[name] = {}
        for s, stage in enumerate(stage_names):
            row = counts[class_id, s]
            support = int(row.sum())
            confused_with = {}
            if support > 0:
                correct = int(row[class_id])
                missed = int(row[0])
                recall = correct / support
                miss_to_background = missed / support
                confused = (support - correct - missed) / support
                for j in range(1, counts.shape[0]):
                    if j == class_id:
                        continue
                    confused_with[class_names[j]] = int(row[j]) / support
            else:
                recall = miss_to_background = confused = None
            cells[name][stage] = {
                "recall": recall,
                "miss_to_background": miss_to_background,
                "confused": confused,
                "support_px": support,
                "below_min_px": support < min_px,
                "confused_with": confused_with,
            }
    return {
        "stages": stage_names,
        "species": [name for _, name in species],
        "min_px": int(min_px),
        "cells": cells,
    }


def phenology_scalar_dict(prefix, breakdown):
    """One mean per stage for recall, background misses, and species confusion.

    Per-species numbers stay in ``phenology_table_rows``. A key per species made
    W&B open a chart for each one. Cells below ``min_px`` are left out of the
    mean so an empty stage does not pull it down, and a missing mean is not logged.
    """
    names = ("recall", "miss_to_background", "confused")
    buckets = {name: {stage: [] for stage in breakdown["stages"]} for name in names}
    for stages in breakdown["cells"].values():
        for stage, cell in stages.items():
            if cell["below_min_px"] or cell["recall"] is None:
                continue
            for name in names:
                buckets[name][stage].append(float(cell[name]))
    out = {}
    for name, by_stage in buckets.items():
        for stage, vals in by_stage.items():
            if vals:
                out[f"{prefix}/{name}/{stage}/mean"] = float(sum(vals) / len(vals))
    return out


def phenology_table_rows(breakdown):
    """One row per species and stage. Shares are blank when the cell is below ``min_px``."""
    columns = ["species", "stage", "recall", "miss_to_background", "confused", "support_px"]
    rows = []
    for species, stages in breakdown["cells"].items():
        for stage, cell in stages.items():
            show = cell["recall"] is not None and not cell["below_min_px"]
            rows.append([
                species,
                stage,
                float(cell["recall"]) if show else None,
                float(cell["miss_to_background"]) if show else None,
                float(cell["confused"]) if show else None,
                int(cell["support_px"]),
            ])
    return columns, rows


def format_worst_stages(breakdown):
    """One line per species: the stage with the lowest recall, among cells with enough pixels."""
    lines = []
    for species, stages in breakdown["cells"].items():
        eligible = [(stage, cell) for stage, cell in stages.items()
                    if cell["recall"] is not None and not cell["below_min_px"]]
        if not eligible:
            lines.append(f"{species}: no stage with at least {breakdown['min_px']} labelled pixels")
            continue
        stage, cell = min(eligible, key=lambda item: item[1]["recall"])
        lines.append(
            f"{species}: worst stage {stage}  recall={cell['recall']:.3f}  "
            f"miss_to_background={cell['miss_to_background']:.3f}  "
            f"confused={cell['confused']:.3f}  n={cell['support_px']}"
        )
    return "\n".join(lines)


def _pheno_species_order(breakdown, species_order):
    order = list(species_order) if species_order is not None else list(breakdown["species"])
    missing = [name for name in order if name not in breakdown["cells"]]
    if missing:
        raise KeyError(f"Species not in the phenology breakdown: {missing}")
    return order


def _fmt_support(n):
    n = int(n)
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.0f}k"
    return str(n)


def visualize_phenology_recall_heatmap(breakdown, save_name, species_order=None, display_names=None):
    """Species (rows) by stage (columns). Cells below ``min_px`` are grey.

    The number in each cell is the recall in percent, with the labelled pixel
    count underneath. Grey cells show only the pixel count.
    """
    order = _pheno_species_order(breakdown, species_order)
    stages = breakdown["stages"]
    labels = [display_names.get(name, name) if display_names else name for name in order]
    recall = np.full((len(order), len(stages)), np.nan)
    below = np.zeros_like(recall, dtype=bool)
    support = np.zeros_like(recall)
    for i, species in enumerate(order):
        for j, stage in enumerate(stages):
            cell = breakdown["cells"][species][stage]
            support[i, j] = cell["support_px"]
            below[i, j] = cell["below_min_px"] or cell["recall"] is None
            if cell["recall"] is not None:
                recall[i, j] = cell["recall"]

    shown = np.ma.masked_where(below | np.isnan(recall), recall)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#d9d9d9")
    fig_h = 0.7 * len(order) + 1.6
    fig, ax = plt.subplots(figsize=(7.2, fig_h))
    im = ax.imshow(shown, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels([s.replace("_", " ") for s in stages])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels)
    for i in range(len(order)):
        for j in range(len(stages)):
            if below[i, j]:
                text, color = f"n={_fmt_support(support[i, j])}", "#555555"
            else:
                text = f"{recall[i, j] * 100:.0f}%\n{_fmt_support(support[i, j])}"
                color = "white" if recall[i, j] >= 0.55 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color=color)
    ax.set_title("Stage recall (grey: too few labelled pixels)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="recall")
    plt.tight_layout()
    plt.savefig(save_name, dpi=200, bbox_inches="tight")
    plt.close()


def visualize_phenology_failure_bars(breakdown, save_name, species_order=None, display_names=None):
    """Stacked shares per species and stage: correct, missed to background, confused.

    A stage with too few pixels is left empty and marked ``low n``, so it is not
    read as a real failure rate.
    """
    order = _pheno_species_order(breakdown, species_order)
    stages = breakdown["stages"]
    labels = [display_names.get(name, name) if display_names else name for name in order]
    n_stages = len(stages)
    width = 0.8 / n_stages
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(order)), 4.8))
    colors = {"correct": "#4c78a8", "missed": "#c7c7c7", "confused": "#e45756"}
    x_base = np.arange(len(order))
    legend_done = set()
    for j, stage in enumerate(stages):
        offset = (j - (n_stages - 1) / 2) * width
        xs = x_base + offset
        correct, missed, confused, low = [], [], [], []
        for species in order:
            cell = breakdown["cells"][species][stage]
            if cell["below_min_px"] or cell["recall"] is None:
                correct.append(0.0)
                missed.append(0.0)
                confused.append(0.0)
                low.append(True)
            else:
                correct.append(cell["recall"])
                missed.append(cell["miss_to_background"])
                confused.append(cell["confused"])
                low.append(False)
        series = (
            ("correct", correct, colors["correct"]),
            ("missed to background", missed, colors["missed"]),
            ("confused with another species", confused, colors["confused"]),
        )
        bottom = np.zeros(len(order))
        for name, values, color in series:
            label = name if name not in legend_done else None
            legend_done.add(name)
            ax.bar(xs, values, width * 0.92, bottom=bottom, color=color, label=label)
            bottom += np.asarray(values)
        for x, is_low, species_i in zip(xs, low, range(len(order))):
            if is_low:
                n = breakdown["cells"][order[species_i]][stage]["support_px"]
                ax.text(x, 0.02, f"low n\n{_fmt_support(n)}", ha="center", va="bottom",
                        fontsize=7, color="#555555", rotation=90)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("share of labelled pixels")
    ax.set_xticks(x_base)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, frameon=False)
    # stage names under the group, one small label per bar slot via a second line
    ax.set_xlabel("  ".join(s.replace("_", "-") for s in stages) + "   (left to right within each species)")
    plt.tight_layout()
    plt.savefig(save_name, dpi=200, bbox_inches="tight")
    plt.close()


def collapse_confusion_to_binary(conf):
    """Collapse a multiclass confusion matrix into 2x2: background vs. any neophyte."""
    bg, fg = 0, slice(1, None)
    return np.array([
        [conf[bg, bg], conf[bg, fg].sum()],
        [conf[fg, bg].sum(), conf[fg, fg].sum()],
    ])


def metrics_from_confusion(conf):
    """Per-class and averaged scores from a confusion matrix (rows = reference).

    Classes absent from both reference and prediction give NaN instead of 0, so
    they drop out of the averages rather than dragging them down -- relevant for
    the per-site breakdown, where most sites carry only one or two species.

    ``*-avg-wo0`` excludes the background class: with >99% background pixels, any
    average including it says more about the background than about the model.
    """
    TP = np.diag(conf)
    FP = conf.sum(axis=0) - TP
    FN = conf.sum(axis=1) - TP

    has_gt = (TP + FN) != 0
    oa = np.sum(TP) / np.sum(conf)
    ua = np.divide(TP, TP + FP, out=np.full_like(TP, np.nan, dtype=float), where=(TP + FP) != 0)
    pa = np.divide(TP, TP + FN, out=np.full_like(TP, np.nan, dtype=float), where=has_gt)
    iou = np.divide(TP, TP + FP + FN, out=np.full_like(TP, np.nan, dtype=float), where=has_gt)
    f1 = np.divide(TP, TP + 0.5 * (FP + FN), out=np.full_like(TP, np.nan, dtype=float), where=has_gt)

    return {
        'OA': oa,
        'F1-avg': np.nanmean(f1), 'F1-avg-wo0': np.nanmean(f1[1:]), 'F1': f1,
        'IoU-avg': np.nanmean(iou), 'IoU-avg-wo0': np.nanmean(iou[1:]), 'IoU': iou,
        'Precision-avg': np.nanmean(ua), 'Precision-avg-wo0': np.nanmean(ua[1:]), 'Precision': ua,
        'Recall-avg': np.nanmean(pa), 'Recall-avg-wo0': np.nanmean(pa[1:]), 'Recall': pa,
    }


def binary_metrics_from_confusion(conf):
    """Background-vs-neophyte scores, i.e. "is there a neophyte here at all"."""
    conf_bin = collapse_confusion_to_binary(conf)
    TP = np.diag(conf_bin)
    FP = conf_bin.sum(axis=0) - TP
    FN = conf_bin.sum(axis=1) - TP
    return {
        'conf': conf_bin,
        'OA': np.sum(TP) / np.sum(conf_bin),
        'F1': TP / (TP + 0.5 * (FP + FN)),
        'IoU': TP / (TP + FP + FN),
        'Precision': TP / (TP + FP),
        'Recall': TP / (TP + FN),
        'class_names': ['background', 'neophyte'],
    }


# =============================================================================
# Quantitative evaluation
# =============================================================================
def evaluate_model(model, dataloader, num_classes, device, ignore_index=-1,
                   eval_auxiliary=False, eval_site=False, eval_month=False,
                   eval_cv=False, cv_folds=None, compute_auc=False, auc_thresholds=200,
                   save_dir=None, class_names=None, display_order=None,
                   phenology_names=None, phenology_class_names=None, phenology_min_px=10000,
                   autocast=False):
    """Run the model over a dataloader and score it.

    One pass accumulates every confusion matrix at once -- the global one plus the
    per-site / per-month / per-fold ones -- because a second pass over the test set
    costs as much as the first.

    Args:
        eval_auxiliary: also count correct pixels per phenological phase and per
            canopy density (bands 2 and 3 of the mask; needs ``load_auxiliary``).
        eval_site / eval_month: break the metrics down by site / acquisition month.
        eval_cv: break them down by CV fold, using ``cv_folds`` (fold name -> list
            of sites) from the data config. This is what makes a model trained on
            one fold comparable with the others.
        compute_auc: additionally compute threshold-independent AUROC (one-vs-rest
            per class, plus binary background-vs-neophyte). The score axis is
            discretised into ``auc_thresholds`` steps -- storing every pixel score
            would be terabytes.
        autocast: run the forward in bfloat16 (then cast probabilities back to
            fp32 before every metric). Off by default, so the forward stays fp32
            as in previous evaluations. The metric code itself is not changed.
        display_order: class order used for the JSON output and the plots (the
            class ids of the data config, background first).
        phenology_names / phenology_class_names / phenology_min_px: only used for the
            extra per-stage breakdown (``phenology_breakdown``). The existing
            phenology and canopy scores are computed exactly as before.

    Returns a dict with the confusion matrix and the metric dicts.
    """
    model.eval().to(device)

    conf_ = MulticlassConfusionMatrix(num_classes, ignore_index=ignore_index).to(device)
    site_confs, month_confs, cv_confs = {}, {}, {}
    cv_auc, cv_auc_bin = {}, {}
    site_to_cv = ({site: cv_name for cv_name, sites in cv_folds.items() for site in sites}
                  if (eval_cv and cv_folds) else {})

    metrics_canopy = {c: {can: {'correct': 0, 'total': 0, 'ratio': 0} for can in range(10, 101, 10)}
                      for c in range(num_classes)} if eval_auxiliary else {}
    metrics_phenology = {c: {phe: {'correct': 0, 'total': 0, 'ratio': 0} for phe in range(1, 4)}
                         for c in range(num_classes)} if eval_auxiliary else {}
    # separate from metrics_phenology: species x stage x predicted class
    pheno_counts = (torch.zeros(num_classes, len(PHENO_STAGE_KEYS), num_classes, dtype=torch.long)
                    if eval_auxiliary else None)

    auc_metric = MulticlassAUROC(num_classes=num_classes, average=None, ignore_index=ignore_index,
                                 thresholds=auc_thresholds).to(device) if compute_auc else None
    auc_metric_bin = BinaryAUROC(thresholds=auc_thresholds).to(device) if compute_auc else None

    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), desc="Evaluating"):
            # Compact labels are cast on the GPU, after the host-to-device copy.
            masks = batch['mask'].to(device)
            if masks.dtype != torch.long:
                masks = masks.long()
            names = batch['name']

            inputs, aux = model.prepare_inputs(batch)
            inputs = inputs.to(device)
            if aux is not None:
                aux = aux.to(device)
            if autocast and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    probs = model(inputs, aux)
                probs = probs.float()
            else:
                probs = model(inputs, aux)
            preds = torch.argmax(probs, dim=1)

            conf_.update(preds, masks)

            if auc_metric is not None:
                auc_metric.update(probs, masks)
                valid = masks != ignore_index
                if valid.any():
                    # binary foreground score = P(neophyte) = 1 - P(background)
                    fg_score = (1.0 - probs[:, 0]).clamp(0.0, 1.0)
                    auc_metric_bin.update(fg_score[valid], (masks[valid] > 0).long())

            if eval_site:
                for b, name in enumerate(names):
                    site = extract_site(name)
                    if site not in site_confs:
                        site_confs[site] = MulticlassConfusionMatrix(num_classes, ignore_index=ignore_index).to(device)
                    site_confs[site].update(preds[b].unsqueeze(0), masks[b].unsqueeze(0))

            if eval_month:
                for b, name in enumerate(names):
                    month = extract_month(name)
                    if month not in month_confs:
                        month_confs[month] = MulticlassConfusionMatrix(num_classes, ignore_index=ignore_index).to(device)
                    month_confs[month].update(preds[b].unsqueeze(0), masks[b].unsqueeze(0))

            if eval_cv and site_to_cv:
                # group the batch by fold so each fold gets one update instead of one per tile
                fold_idx = {}
                for b, name in enumerate(names):
                    # site folders are <Site>_<flight>_<tile-group>; the folds list <Site>_<flight>
                    cv_name = site_to_cv.get(extract_site(name).rsplit('_', 1)[0])
                    if cv_name is not None:
                        fold_idx.setdefault(cv_name, []).append(b)

                for cv_name, idx in fold_idx.items():
                    if cv_name not in cv_confs:
                        cv_confs[cv_name] = MulticlassConfusionMatrix(num_classes, ignore_index=ignore_index).to(device)
                    cv_confs[cv_name].update(preds[idx], masks[idx])

                    if compute_auc:
                        if cv_name not in cv_auc:
                            cv_auc[cv_name] = MulticlassAUROC(num_classes=num_classes, average=None,
                                                              ignore_index=ignore_index,
                                                              thresholds=auc_thresholds).to(device)
                            cv_auc_bin[cv_name] = BinaryAUROC(thresholds=auc_thresholds).to(device)
                        cv_auc[cv_name].update(probs[idx], masks[idx])
                        masks_f = masks[idx]
                        valid_f = masks_f != ignore_index
                        if valid_f.any():
                            fg_score_f = (1.0 - probs[idx][:, 0]).clamp(0.0, 1.0)
                            cv_auc_bin[cv_name].update(fg_score_f[valid_f], (masks_f[valid_f] > 0).long())

            if eval_auxiliary:
                canopy = batch['canopy'].to(device)
                phenology = batch['phenology'].to(device)
                if canopy.dtype != torch.float32:
                    canopy = canopy.float()
                if phenology.dtype != torch.long:
                    phenology = phenology.long()
                known = (canopy != ignore_index) & (phenology != ignore_index)
                valid_preds, valid_masks = preds[known], masks[known]
                valid_can, valid_phe = canopy[known], phenology[known]
                update_phenology_counts(pheno_counts, valid_preds, valid_masks, valid_phe)

                for class_id in range(num_classes):
                    is_class = valid_masks == class_id
                    if not is_class.any():
                        continue
                    correct = (valid_preds == class_id) & is_class
                    for phe_value in range(1, 4):
                        in_phe = valid_phe == phe_value
                        metrics_phenology[class_id][phe_value]['correct'] += (correct & in_phe).sum().item()
                        metrics_phenology[class_id][phe_value]['total'] += (is_class & in_phe).sum().item()
                    for can_value in range(10, 101, 10):
                        in_can = valid_can == can_value
                        metrics_canopy[class_id][can_value]['correct'] += (correct & in_can).sum().item()
                        metrics_canopy[class_id][can_value]['total'] += (is_class & in_can).sum().item()

    # ---- global metrics ----------------------------------------------------
    conf = conf_.compute().cpu().numpy()
    metrics_semseg = metrics_from_confusion(conf)
    metrics_semseg_binary = binary_metrics_from_confusion(conf)

    if auc_metric is not None:
        auc = auc_metric.compute().cpu().numpy()
        metrics_semseg['AUC-avg'] = np.nanmean(auc)
        metrics_semseg['AUC-avg-wo0'] = np.nanmean(auc[1:])
        metrics_semseg['AUC'] = auc
        metrics_semseg_binary['AUC'] = float(auc_metric_bin.compute().cpu())

    # ---- breakdowns --------------------------------------------------------
    metrics_semseg_site, metrics_semseg_site_binary = {}, {}
    for site, conf_site in site_confs.items():
        conf_s = conf_site.compute().cpu().numpy()
        metrics_semseg_site[site] = {**metrics_from_confusion(conf_s), 'conf': conf_s}
        metrics_semseg_site_binary[site] = binary_metrics_from_confusion(conf_s)

    metrics_semseg_month = {}
    for month, conf_month in month_confs.items():
        conf_m = conf_month.compute().cpu().numpy()
        metrics_semseg_month[month] = {**metrics_from_confusion(conf_m), 'conf': conf_m}

    metrics_semseg_cv = {}
    for cv_name, conf_cv in cv_confs.items():
        conf_c = conf_cv.compute().cpu().numpy()
        metrics_semseg_cv[cv_name] = {**metrics_from_confusion(conf_c), 'conf': conf_c}
        if cv_name in cv_auc:
            auc_c = cv_auc[cv_name].compute().cpu().numpy()
            metrics_semseg_cv[cv_name]['AUC-avg'] = np.nanmean(auc_c)
            metrics_semseg_cv[cv_name]['AUC-avg-wo0'] = np.nanmean(auc_c[1:])
            metrics_semseg_cv[cv_name]['AUC'] = auc_c
            metrics_semseg_cv[cv_name]['AUC-bin'] = float(cv_auc_bin[cv_name].compute().cpu())

    if metrics_semseg_cv:
        metrics_semseg_cv.update(aggregate_over_folds(metrics_semseg_cv))

    for class_dict in (metrics_canopy, metrics_phenology):
        for per_bin in class_dict.values():
            for counts in per_bin.values():
                if counts['total'] > 0:
                    counts['ratio'] = round(counts['correct'] / counts['total'] * 100, 2)

    phenology_breakdown_dict = None
    if pheno_counts is not None:
        stage_keys = phenology_stage_keys(phenology_names)
        if phenology_class_names is None:
            phenology_class_names = [f"class_{i}" for i in range(num_classes)]
        phenology_breakdown_dict = phenology_breakdown(
            pheno_counts, stage_keys, list(phenology_class_names), phenology_min_px)

    # ---- save --------------------------------------------------------------
    if save_dir:
        _PER_CLASS_KEYS = ('F1', 'IoU', 'Precision', 'Recall', 'AUC')

        def _reorder(m_dict):
            """Per-class arrays in display order, so the JSON matches the plots."""
            if display_order is None:
                return m_dict
            out = dict(m_dict)
            if class_names is not None:
                out['class_names'] = [class_names[i] for i in display_order]
            for key in _PER_CLASS_KEYS:
                if key in out and hasattr(out[key], '__len__'):
                    out[key] = np.array(out[key])[np.array(display_order)]
            return out

        def _dump(obj, name):
            with open(os.path.join(save_dir, name), 'w') as f:
                json.dump(convert_ndarray_to_list(obj), f, indent=4)

        np.save(os.path.join(save_dir, 'confusion_matrix.npy'), conf)
        _dump(_reorder(metrics_semseg), 'metrics_semseg.json')
        _dump(metrics_semseg_binary, 'metrics_semseg_binary.json')
        if metrics_semseg_site:
            _dump({s: _reorder(m) for s, m in metrics_semseg_site.items()}, 'metrics_semseg_site.json')
            _dump(metrics_semseg_site_binary, 'metrics_semseg_site_binary.json')
        if metrics_semseg_month:
            _dump({mo: _reorder(m) for mo, m in metrics_semseg_month.items()}, 'metrics_semseg_month.json')
        if metrics_semseg_cv:
            _dump({k: _reorder(m) for k, m in metrics_semseg_cv.items()}, 'metrics_semseg_cv.json')
        if eval_auxiliary:
            _dump(metrics_canopy, 'metrics_canopy.json')
            _dump(metrics_phenology, 'metrics_phenology.json')
            _dump(phenology_breakdown_dict, 'metrics_phenology_breakdown.json')

    return {
        'conf': conf,
        'semseg': metrics_semseg,
        'semseg_binary': metrics_semseg_binary,
        'semseg_site': metrics_semseg_site,
        'semseg_month': metrics_semseg_month,
        'semseg_cv': metrics_semseg_cv,
        'canopy': metrics_canopy,
        'phenology': metrics_phenology,
        'phenology_breakdown': phenology_breakdown_dict,
    }


def aggregate_over_folds(per_fold, keys=('OA', 'F1-avg-wo0', 'IoU-avg-wo0',
                                         'Precision-avg-wo0', 'Recall-avg-wo0',
                                         'AUC-avg', 'AUC-avg-wo0', 'AUC-bin'),
                         per_class_keys=('F1', 'IoU', 'Precision', 'Recall', 'AUC')):
    """Mean and std over CV folds -- the headline numbers of this project.

    ``per_fold`` maps fold name -> metric dict. Returns ``{'mean': ..., 'std': ...}``
    with the scalar metrics averaged and the per-class arrays averaged element-wise.
    Metrics missing from any fold are skipped.
    """
    fold_names = sorted(k for k in per_fold if k not in ('mean', 'std'))
    out = {}
    for agg_name, agg_fn in (('mean', np.nanmean), ('std', np.nanstd)):
        agg = {}
        for key in keys:
            if all(key in per_fold[k] for k in fold_names):
                agg[key] = agg_fn([per_fold[k][key] for k in fold_names])
        for key in per_class_keys:
            if all(key in per_fold[k] for k in fold_names):
                agg[key] = agg_fn([per_fold[k][key] for k in fold_names], axis=0)
        out[agg_name] = agg
    return out


# =============================================================================
# Plots
# =============================================================================
def visualize_scores_per_class(scores, save_name, display_labels, fontsize=5):
    """Grouped bar chart: one group of bars (F1, IoU, ...) per class."""
    x = np.arange(len(display_labels))
    score_names = list(scores.keys())
    width = 0.5 / len(score_names)
    group_offset = (len(score_names) - 1) * width / 2
    colors = plt.get_cmap("Set2")(np.linspace(0, 1, len(score_names)))

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (score_name, color) in enumerate(zip(score_names, colors)):
        ax.bar(x - group_offset + i * width, scores[score_name], width, label=score_name, color=color)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(score_names))
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=45, ha='right', fontsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_scores_per_class_cv(mean_scores, std_scores, save_name, display_labels, fontsize=5):
    """Same as :func:`visualize_scores_per_class`, with the std over CV folds as error bars."""
    x = np.arange(len(display_labels))
    score_names = list(mean_scores.keys())
    width = 0.5 / len(score_names)
    group_offset = (len(score_names) - 1) * width / 2
    colors = plt.get_cmap("Set2")(np.linspace(0, 1, len(score_names)))

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (score_name, color) in enumerate(zip(score_names, colors)):
        ax.bar(x - group_offset + i * width, mean_scores[score_name], width,
               yerr=std_scores[score_name], capsize=2, label=score_name, color=color,
               error_kw={'linewidth': 0.8})

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(score_names))
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=45, ha='right', fontsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_scores_per_auxiliary(scores, save_name, class_names, auxiliary_names,
                                   include_background=False, x_label='', class_order=None):
    """One panel per class: labelled vs. correctly predicted pixels per auxiliary bin
    (phenological phase or canopy density), with the hit rate printed on each bar."""
    if class_order is not None:
        class_ids = class_order
    else:
        class_ids = range(len(class_names)) if include_background else range(1, len(class_names))

    fig, axes = plt.subplots(1, len(class_ids), figsize=(5 * len(class_ids), 6))
    if len(class_ids) == 1:
        axes = [axes]

    for ax, class_id in zip(axes, class_ids):
        totals, corrects, ratio_labels = [], [], []
        for stage_id in scores[class_id]:
            data = scores[class_id][stage_id]
            totals.append(data.get("total", 0))
            corrects.append(data.get("correct", 0))
            ratio_labels.append(f"{data.get('ratio', 0):.1f}%" if data.get("total", 0) > 0 else "")

        x = np.arange(len(auxiliary_names))
        ax.bar(x, totals, label="total", color="lightgray")
        ax.bar(x, corrects, label="correct", color="steelblue")
        for x_pos, correct, label in zip(x, corrects, ratio_labels):
            if label:
                ax.text(x_pos, correct, label, ha='center', va='bottom', fontsize=10)

        ax.set_title(class_names[class_id], fontsize=14)
        ax.yaxis.grid(True, linestyle='--', alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(auxiliary_names, rotation=45, ha='right')
        ax.set_ylabel('pixel count')
        ax.set_xlabel(x_label)
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_name, dpi=250, bbox_inches='tight')
    plt.close()


def visualize_save_confusion(confusion, save_name, display_labels):
    ConfusionMatrixDisplay(confusion, display_labels=display_labels).plot()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_save_confusions(confusion, save_dir, display_labels=None):
    """Confusion matrix in four normalizations.

    Absolute counts are unreadable here (background dwarfs everything), so the
    normalized variants carry the information: by reference row = recall per
    class, by prediction column = precision per class.
    """
    variants = {
        'confusion.jpg': confusion,
        'confusion_norm_all.jpg': confusion / np.sum(confusion),
        'confusion_norm_pred.jpg': confusion / np.sum(confusion, axis=0)[np.newaxis, :],
        'confusion_norm_true.jpg': confusion / np.sum(confusion, axis=1)[:, np.newaxis],
    }
    for name, conf in variants.items():
        visualize_save_confusion(np.round(np.nan_to_num(conf, nan=0), 2),
                                 os.path.join(save_dir, name), display_labels=display_labels)


def visualize_f1_per_month(metrics_month, save_path, class_names, class_colors):
    """Per-class F1 across acquisition months -- species are easiest to spot while
    they flower, so the score is expected to move over the season."""
    month_labels = {'05': 'May', '06': 'Jun', '07': 'Jul', '08': 'Aug', '09': 'Sep', '10': 'Oct'}
    months_sorted = sorted(metrics_month.keys())

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for c in range(1, len(class_names)):  # skip background
        f1_vals = [metrics_month[m]['F1'][c] for m in months_sorted]
        valid_x = [i for i, v in enumerate(f1_vals) if not np.isnan(v)]
        valid_y = [v for v in f1_vals if not np.isnan(v)]
        ax.plot(valid_x, valid_y, 'o-', color=class_colors[c], label=class_names[c],
                linewidth=1.5, markersize=5)

    ax.set_xticks(range(len(months_sorted)))
    ax.set_xticklabels([month_labels.get(m, m) for m in months_sorted])
    ax.set_xlabel('Month')
    ax.set_ylabel('F1 Score')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='best', fontsize='small')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


# =============================================================================
# Qualitative results
# =============================================================================
def save_model_predictions(model, dataloader, device, save_dir, colors, class_names=None,
                           gray_boost=1.15, ignore_index=-1, ndsm_mean=0.0, ndsm_std=1.0,
                           entropy_gamma=0.5, autocast=False):
    """One figure per tile: [RGB | reference | prediction | entropy | (nDSM)].

    Masks are drawn over a brightened grayscale version of the tile so the plants
    stay visible under the colour overlay. Background and ignore pixels are
    transparent; ignore regions are hatched in the reference panel.

    The entropy panel shows the normalized predictive entropy of the softmax
    output, ``(-sum_c p log p) / log K``, in [0, 1]: 0 where the model puts all its
    probability on one class, 1 where it spreads it evenly over all of them. It is
    the cheapest look at *where* a prediction is shaky -- typically the boundaries
    of a plant and the places where two species are confused. ``entropy_gamma < 1``
    warps the colour scale so the small values, where almost every pixel sits, stay
    distinguishable.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval().to(device)

    cmap_no_bg = ListedColormap(colors[1:])
    legend_patches = [mpatches.Patch(color=colors[1:][i], label=class_names[1:][i])
                      for i in range(len(class_names) - 1)] if class_names else None

    sample_idx = 0
    with torch.no_grad():
        for batch in dataloader:
            imgs = batch["image"]
            masks = batch["mask"]
            if torch.is_tensor(masks) and masks.dtype != torch.long:
                masks = masks.long()
            names = batch.get("name", None)
            ndsm_batch = batch.get("ndsm", None)

            inputs, aux = model.prepare_inputs(batch)
            inputs = inputs.to(device)
            if aux is not None:
                aux = aux.to(device)
            if autocast and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    probs = model(inputs, aux)
                probs = probs.float()
            else:
                probs = model(inputs, aux)
            preds = probs.argmax(dim=1).cpu().numpy()
            num_classes = probs.shape[1]
            norm_entropy = (-(probs * torch.log(probs + 1e-10)).sum(dim=1)
                            / float(np.log(num_classes))).cpu().numpy()

            imgs = imgs.numpy()
            masks = masks.numpy()
            if ndsm_batch is not None:
                ndsm_batch = ndsm_batch.numpy()  # (B, 1, H, W)

            ncols = 4 + (1 if ndsm_batch is not None else 0)

            for b in range(imgs.shape[0]):
                img = denormalize(imgs[b])
                gray = np.clip(np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]) * gray_boost, 0, 1)

                ignore_mask = masks[b] == ignore_index
                ref_mask = np.ma.masked_where((masks[b] == 0) | ignore_mask, masks[b])
                pred_mask = np.ma.masked_where((preds[b] == 0) | (preds[b] == ignore_index), preds[b])

                plt.figure(figsize=(5 * ncols, 5), dpi=300)
                col = 1

                plt.subplot(1, ncols, col); col += 1
                plt.title("Image", fontsize=16)
                plt.imshow(img)
                plt.axis("off")

                plt.subplot(1, ncols, col); col += 1
                plt.title("Reference", fontsize=16)
                plt.imshow(gray, cmap="gray")
                plt.imshow(ref_mask, cmap=cmap_no_bg, vmin=1, vmax=len(colors) - 1, alpha=0.5)
                plt.contourf(ignore_mask.astype(int), levels=[0.5, 1.5], colors="none", hatches=["///"])
                plt.axis("off")

                plt.subplot(1, ncols, col); col += 1
                plt.title("Prediction", fontsize=16)
                plt.imshow(gray, cmap="gray")
                plt.imshow(pred_mask, cmap=cmap_no_bg, vmin=1, vmax=len(colors) - 1, alpha=0.5)
                plt.axis("off")

                plt.subplot(1, ncols, col); col += 1
                plt.title("Normalized entropy", fontsize=16)
                plt.imshow(gray, cmap="gray")
                im = plt.imshow(norm_entropy[b], cmap="viridis", alpha=0.6,
                                norm=PowerNorm(gamma=entropy_gamma, vmin=0, vmax=1.0))
                plt.colorbar(im, fraction=0.046, pad=0.04)
                plt.axis("off")

                if ndsm_batch is not None:
                    plt.subplot(1, ncols, col)
                    plt.title("nDSM [m]", fontsize=16)
                    im = plt.imshow(ndsm_batch[b, 0] * ndsm_std + ndsm_mean, cmap="viridis", vmin=0)
                    plt.colorbar(im, fraction=0.046, pad=0.04)
                    plt.axis("off")

                if legend_patches:
                    plt.figlegend(handles=legend_patches, loc="lower center",
                                  ncol=min(6, len(legend_patches)), fontsize=12,
                                  bbox_to_anchor=(0.5, -0.05))

                plt.tight_layout()
                fname = (os.path.splitext(os.path.basename(names[b]))[0] if names is not None
                         else f"sample_{sample_idx:05d}")
                plt.savefig(os.path.join(save_dir, f"{fname}.jpg"), dpi=300, bbox_inches="tight")
                plt.close()
                sample_idx += 1
