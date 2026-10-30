"""Evaluate a trained checkpoint on its test split.

    python test.py exp_name=<run directory in lightning_logs/>
    python test.py exp_name=<run> ckpt_type=last
    python test.py exp_name=<run> eval_site=True eval_month=True

The matching test data config is derived from the name embedded in
``exp_name``. A run named ``np_cv3_...`` or ``neophytes_split_cv3_train_...``
is tested on ``neophytes_split_cv3_test`` (``_ds2`` stays on the half-resolution
test config). Pass ``data=<config>`` explicitly to override that.

``train_site=Aarau_1`` (a site), ``train_site=Aarau_3_1`` (one flight), or
``train_site=[Aarau_3_1,Rueti_1_1]`` also scores those validation tiles. Those
are the same-site numbers. The test split is the held-out number. Both are
written to ``quantitative/metrics_site_split.json``.

Results land in ``lightning_logs/<exp_name>/test_<timestamp>_<data>_<ckpt>/``:
    quantitative/  metrics as JSON + score plots + confusion matrices
    qualitative/   prediction figures for a few picked test tiles
"""

import hydra
from omegaconf import DictConfig, OmegaConf


def _auto_data_config(exp_name: str) -> str | None:
    """Infer the test data config from the data name embedded in exp_name.

    Older runs are ``<stamp>_np_cv3_...``. Explorer runs are
    ``<stamp>_neophytes_split_cv3_train_1024_...``. A 512 crop and a 1024 crop
    of the same fold share one test config, because the ground sampling distance
    does not change. A ``_ds2`` run stays on the half-resolution test config.
    """
    import os
    import re

    # exp_name format: YYYYMMDD_HHMMSS[.ffffff]_<data_name>_<model_...>
    m = re.match(r'^\d{8}_[\d.]+_(.+)$', exp_name)
    if not m:
        return None
    rest = m.group(1)

    # more specific patterns first: a half-resolution run (_ds2) must not fall back
    # to the full-resolution test config, or it would be evaluated at the wrong scale
    patterns = [
        (r'^np_cv(\d+)_ds2', lambda m: f'neophytes_split_cv{m.group(1)}_test_ds2'),
        (r'^np_cv(\d+)', lambda m: f'neophytes_split_cv{m.group(1)}_test'),
        (r'^np_loc', lambda _: 'neophytes_split_local_test'),
        (r'^np_debug', lambda _: 'neophytes_split_debug_test'),
        (r'^neophytes_split_cv(\d+)_.*_ds2(?:_|$)',
         lambda m: f'neophytes_split_cv{m.group(1)}_test_ds2'),
        (r'^neophytes_split_cv(\d+)_', lambda m: f'neophytes_split_cv{m.group(1)}_test'),
        (r'^neophytes_split_local_', lambda _: 'neophytes_split_local_test'),
        (r'^neophytes_split_debug_', lambda _: 'neophytes_split_debug_test'),
    ]
    for pattern, make_cfg in patterns:
        pm = re.match(pattern, rest)
        if pm:
            test_cfg = make_cfg(pm)
            return test_cfg if os.path.isfile(os.path.join("configs", "data", f"{test_cfg}.yaml")) else None
    return None


def _site_keys(value):
    """A site, a flight, or a list of either. None when unset."""
    if value is None or value == "" or value == "null":
        return None
    if isinstance(value, str):
        return [value]
    items = [str(item) for item in value if str(item) not in ("", "null", "None")]
    return items or None


def _site_split_record(metrics, n_tiles, class_names):
    """Mean species F1, plus each class, for one group of tiles."""
    return {
        "tiles": int(n_tiles),
        "F1-avg-wo0": float(metrics["semseg"]["F1-avg-wo0"]),
        "F1": metrics["semseg"]["F1"],
        "class_names": list(class_names),
    }


def _write_site_split(cfg, model, device, train_site, held_out_metrics, class_names,
                      metrics_dir, n_held_out):
    """Score the training flights' validation tiles and store both groups.

    Same-site tiles are the val folders of ``train_site`` (one site, one flight,
    or a list). Held-out tiles are the test split already scored (the fold's
    other sites). Training did not use the val tiles, and it did not use the
    held-out sites.
    """
    import json
    import os
    from torch.utils.data import DataLoader

    from datasets.neophyte_dataset import NeophyteDataset
    from utils.eval_utils import evaluate_model
    from utils.transform_utils import get_transforms
    from utils.utils import convert_ndarray_to_list

    test_transforms = get_transforms(cfg.data.test_transforms, cfg.data.transforms_additional_targets)
    dsm_transforms = get_transforms(cfg.data.dsm_transforms, toTensor=False)
    ndsm_transforms = get_transforms(cfg.data.ndsm_transforms, toTensor=False)
    dataset = NeophyteDataset(
        cfg.data.image_dir.val, cfg.data.classes, transform=test_transforms,
        dsm_transform=dsm_transforms, ndsm_transform=ndsm_transforms,
        dataset_stats=cfg.data.dataset_stats, value_mapping=cfg.data.value_mapping,
        ignore_index=cfg.data.ignore_index, load_dsm=cfg.data.load_dsm,
        load_dtm=cfg.data.load_dtm, load_ndsm=cfg.data.load_ndsm,
        train_site=train_site,
    )
    shown = ", ".join(train_site)
    print(f"Same-site evaluation: {shown}, {len(dataset)} validation tiles.")
    loader = DataLoader(dataset, batch_size=cfg.data.batch_size_test, shuffle=False,
                        num_workers=cfg.data.num_workers)
    same = evaluate_model(model, loader, cfg.model.num_classes, device,
                          ignore_index=cfg.data.ignore_index)
    record = {
        "train_site": list(train_site),
        "same_site": _site_split_record(same, len(dataset), class_names),
        "held_out": _site_split_record(held_out_metrics, n_held_out, class_names),
    }
    path = os.path.join(metrics_dir, "metrics_site_split.json")
    with open(path, "w") as handle:
        json.dump(convert_ndarray_to_list(record), handle, indent=4)
    print(f"same-site ({shown}) mF1 (wo bg): {record['same_site']['F1-avg-wo0']:.4f}  |  "
          f"tiles: {len(dataset)}")
    print(f"held-out mF1 (wo bg): {record['held_out']['F1-avg-wo0']:.4f}  |  "
          f"tiles: {n_held_out}")
    print(f"Wrote {path}")


def _log_phenology_wandb(cfg, breakdown, heatmap_path, bars_path):
    """Attach the stage table and figures to the training run on W&B.

    The training run is the one whose name is ``exp_name``. If it cannot be
    found, a new run is created in the same project and grouped under that name.
    ``logger: csv`` skips this. A W&B failure is reported and does not fail the test.
    """
    if cfg.logger != "wandb":
        print(f"[test.py] logger={cfg.logger}: phenology table and figures stay on disk.")
        return

    import os

    import wandb

    def _num(value):
        return float("nan") if value is None else float(value)

    entity, project = cfg.wandb_entity, cfg.log_name
    run_id = None
    try:
        api = wandb.Api()
        matches = api.runs(f"{entity}/{project}", filters={"display_name": cfg.exp_name},
                           order="-created_at", per_page=5)
        for run in matches:
            if run.name == cfg.exp_name or getattr(run, "display_name", None) == cfg.exp_name:
                run_id = run.id
                break
    except Exception as exc:
        print(f"[test.py] W&B run lookup failed ({exc}). Starting a new run.")

    # keep the W&B files next to the run, not in the working directory
    log_dir = cfg.log_path if "log_path" in cfg else None
    init_kwargs = dict(entity=entity, project=project, job_type="phenology_test")
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        init_kwargs["dir"] = log_dir
    try:
        if run_id:
            print(f"[test.py] Logging phenology figures to W&B run {run_id} ({cfg.exp_name}).")
            wandb.init(id=run_id, resume="allow", **init_kwargs)
        else:
            print(f"[test.py] No W&B run named {cfg.exp_name}. Creating a new one.")
            wandb.init(name=f"{cfg.exp_name}__phenology", group=cfg.exp_name, **init_kwargs)

        species = list(breakdown["species"])
        columns = ["species", "stage", "recall", "miss_to_background", "confused",
                   "support_px", "below_min_px"] + [f"confused_with_{name}" for name in species]
        table = wandb.Table(columns=columns)
        for name, stages in breakdown["cells"].items():
            for stage, cell in stages.items():
                confused_with = cell["confused_with"]
                table.add_data(
                    name, stage, _num(cell["recall"]), _num(cell["miss_to_background"]),
                    _num(cell["confused"]), int(cell["support_px"]), bool(cell["below_min_px"]),
                    *[float(confused_with.get(other, 0.0)) for other in species],
                )
        wandb.log({
            "phenology/breakdown": table,
            "phenology/recall_heatmap": wandb.Image(heatmap_path),
            "phenology/failure_bars": wandb.Image(bars_path),
        })
        wandb.finish()
    except Exception as exc:
        print(f"[test.py] W&B logging failed ({exc}). The files on disk are unaffected.")
        try:
            wandb.finish()
        except Exception:
            pass


def _require_exp_name(argv, script):
    """Fail early with a readable message instead of a Hydra interpolation error."""
    if not any(a.startswith("exp_name=") for a in argv):
        import sys
        sys.exit(f"Missing exp_name. Usage: python {script} exp_name=<run directory in "
                 f"lightning_logs/>\nAvailable runs: ls lightning_logs/")


@hydra.main(version_base=None, config_path="configs", config_name="test")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    import os

    import matplotlib
    import numpy as np
    import torch
    import torch.multiprocessing
    from pytorch_lightning import seed_everything

    from datasets.neophyte_datamodule import NeophyteDataModule
    from models.semseg_plm import SegmentationModel
    from utils.eval_utils import (class_stats_names, evaluate_model, format_worst_stages,
                                  save_model_predictions, visualize_f1_per_month,
                                  visualize_phenology_failure_bars, visualize_phenology_recall_heatmap,
                                  visualize_save_confusions, visualize_scores_per_auxiliary,
                                  visualize_scores_per_class, visualize_scores_per_class_cv)
    from utils.transform_utils import get_transforms
    from utils.utils import getListOfFiles, make_dir, natural_keys, shrink_dict

    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.set_float32_matmul_precision('medium')
    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    matplotlib.rcParams.update(cfg.matplotlib_style)

    if cfg.pl_seed:
        seed_everything(cfg.pl_seed, workers=True)

    # %% data -- no augmentation, no weighted sampling, no few-shot filtering
    test_transforms = get_transforms(cfg.data.test_transforms, cfg.data.transforms_additional_targets)
    dsm_transforms = get_transforms(cfg.data.dsm_transforms, toTensor=False)
    ndsm_transforms = get_transforms(cfg.data.ndsm_transforms, toTensor=False)

    dataModule = NeophyteDataModule(cfg.data.image_dir, cfg.data.classes,
                                    test_transforms, test_transforms, test_transforms,
                                    dsm_transform=dsm_transforms, ndsm_transform=ndsm_transforms,
                                    dataset_stats=cfg.data.dataset_stats,
                                    value_mapping=cfg.data.value_mapping,
                                    ignore_index=cfg.data.ignore_index,
                                    load_auxiliary=cfg.data.load_auxiliary,
                                    load_dsm=cfg.data.load_dsm,
                                    load_dtm=cfg.data.load_dtm,
                                    load_ndsm=cfg.data.load_ndsm,
                                    batch_size_train=cfg.data.batch_size_test,
                                    batch_size_val=cfg.data.batch_size_test,
                                    batch_size_test=cfg.data.batch_size_test,
                                    num_workers=cfg.data.num_workers)
    dataModule.prepare_data()
    dataModule.setup()

    batch = next(iter(dataModule.test_dataloader()))
    print('Image batch shape: ', batch['image'].shape, '\nMask batch shape:  ', batch['mask'].shape)

    # %% checkpoint
    ckpts_paths = sorted(getListOfFiles(cfg.ckpt_path), key=natural_keys)
    matched = [s for s in ckpts_paths if cfg.ckpt_type in s]
    if not matched:
        matched = [s for s in ckpts_paths if 'last' in s]
        if not matched:
            raise FileNotFoundError(f"No checkpoint matching '{cfg.ckpt_type}' (or 'last') in {cfg.ckpt_path}")
        print(f"[test.py] WARNING: no '{cfg.ckpt_type}' checkpoint found, falling back to 'last'")
    ckpt_path = matched[-1]
    print('ckpt_path: ', ckpt_path)

    # # keep the epoch number in the output folder name (best_epoch=42 rather than best)
    cfg.ckpt_type = os.path.splitext(os.path.basename(ckpt_path))[0]
    OmegaConf.resolve(cfg)  # test_path interpolates ckpt_type, so resolve again

    model = SegmentationModel.load_from_checkpoint(ckpt_path, weights_only=False).to(device)
    # # evaluate on whole tiles: patching was a training-time constraint of the encoder
    model.set_patch_2_img_size(False)
    print('Model loaded successfully.')

    # # class names/colors in class-index order, plus the display order for the plots
    class_names = [cfg.data.classes[i].name for i in sorted(cfg.data.classes.keys(), key=int)]
    class_colors = [cfg.data.classes[i].color for i in sorted(cfg.data.classes.keys(), key=int)]
    display_order = [0] + list(cfg.data.display_order)
    class_names_disp = [class_names[i] for i in display_order]

    def reorder_scores(scores):
        return {k: np.array([v[i] for i in display_order]) for k, v in scores.items()}

    # %% quantitative evaluation over the full test set
    if not cfg.qualitative_only:
        print('Start quantitative evaluation of the full test dataset.')
        metrics_dir = make_dir(cfg.test_path, 'quantitative')
        cv_folds = OmegaConf.to_container(cfg.data.cv_folds, resolve=True) if 'cv_folds' in cfg.data else None
        # canopy/phenology only exist in the batch when the data config loads them
        eval_auxiliary = cfg.eval_auxiliary and cfg.data.load_auxiliary

        stats_names = class_stats_names(cfg.data.classes)
        metrics = evaluate_model(model, dataModule.test_dataloader(), cfg.model.num_classes, device,
                                 ignore_index=cfg.data.ignore_index,
                                 eval_auxiliary=eval_auxiliary, eval_site=cfg.eval_site,
                                 eval_month=cfg.eval_month, eval_cv=cfg.eval_cv, cv_folds=cv_folds,
                                 compute_auc=cfg.compute_auc, auc_thresholds=cfg.auc_thresholds,
                                 save_dir=metrics_dir, class_names=class_names,
                                 display_order=display_order,
                                 phenology_names=list(cfg.data.phenology_names),
                                 phenology_class_names=stats_names,
                                 phenology_min_px=cfg.phenology_min_px,
                                 autocast=cfg.autocast)

        score_keys = ['F1', 'IoU', 'Precision', 'Recall']
        visualize_scores_per_class(reorder_scores(shrink_dict(metrics['semseg'], score_keys)),
                                   os.path.join(metrics_dir, 'semseg_scores.png'), class_names_disp)
        visualize_save_confusions(metrics['conf'][np.ix_(display_order, display_order)],
                                  metrics_dir, display_labels=class_names_disp)

        print(f"Class-wise F1: {np.array2string(metrics['semseg']['F1'], formatter={'float_kind': lambda x: f'{x:.3f}'})}")
        print(f"mF1 (wo bg): {metrics['semseg']['F1-avg-wo0']:.4f}  |  "
              f"mIoU (wo bg): {metrics['semseg']['IoU-avg-wo0']:.4f}")
        if 'AUC' in metrics['semseg']:
            print(f"mAUC (wo bg): {metrics['semseg']['AUC-avg-wo0']:.4f}  |  "
                  f"binary AUC: {metrics['semseg_binary']['AUC']:.4f}")

        # # the headline metric: mean ± std over the CV folds present in this test split
        if cfg.eval_cv and 'mean' in metrics.get('semseg_cv', {}):
            cv_mean, cv_std = metrics['semseg_cv']['mean'], metrics['semseg_cv']['std']
            print(f"Per-CV mF1 (neophytes): {cv_mean['F1-avg-wo0']:.4f} ± {cv_std['F1-avg-wo0']:.4f}  |  "
                  f"mIoU: {cv_mean['IoU-avg-wo0']:.4f} ± {cv_std['IoU-avg-wo0']:.4f}  |  "
                  f"mRecall: {cv_mean['Recall-avg-wo0']:.4f} ± {cv_std['Recall-avg-wo0']:.4f}")
            visualize_scores_per_class_cv(reorder_scores(shrink_dict(cv_mean, score_keys)),
                                          reorder_scores(shrink_dict(cv_std, score_keys)),
                                          os.path.join(metrics_dir, 'semseg_scores_cv.png'),
                                          class_names_disp)

        if cfg.eval_month and metrics['semseg_month']:
            visualize_f1_per_month(metrics['semseg_month'], os.path.join(metrics_dir, 'f1_per_month.png'),
                                   class_names, class_colors)

        if eval_auxiliary:
            visualize_scores_per_auxiliary(metrics['phenology'],
                                           os.path.join(metrics_dir, 'phenology_scores.png'),
                                           class_names, cfg.data.phenology_names,
                                           x_label="phenological phase",
                                           class_order=list(cfg.data.display_order))
            visualize_scores_per_auxiliary(metrics['canopy'],
                                           os.path.join(metrics_dir, 'canopy_scores.png'),
                                           class_names, cfg.data.canopy_names,
                                           x_label=r"canopy density [\%]",
                                           class_order=list(cfg.data.display_order))

            breakdown = metrics["phenology_breakdown"]
            species_order = [stats_names[i] for i in cfg.data.display_order]
            display_by_stats = {stats: class_names[i] for i, stats in enumerate(stats_names)}
            heatmap_path = os.path.join(metrics_dir, "phenology_recall_heatmap.png")
            bars_path = os.path.join(metrics_dir, "phenology_failure_bars.png")
            visualize_phenology_recall_heatmap(breakdown, heatmap_path, species_order, display_by_stats)
            visualize_phenology_failure_bars(breakdown, bars_path, species_order, display_by_stats)
            print("Worst stage per species (stage recall):")
            print(format_worst_stages(breakdown))
            _log_phenology_wandb(cfg, breakdown, heatmap_path, bars_path)

        train_site = _site_keys(cfg.train_site)
        if train_site:
            _write_site_split(cfg, model, device, train_site, metrics, class_names,
                              metrics_dir, len(dataModule.test_dataset))

    # %% qualitative results for a few picked test tiles
    print('Start qualitative evaluation of some picked samples from the test set.')
    qual_dir = make_dir(cfg.test_path, 'qualitative')
    ndsm_norm = next((t.params for t in cfg.data.ndsm_transforms if t.type == 'Normalize'), None)
    save_model_predictions(model, dataloader=dataModule.pick_dataloader(), device=device,
                           save_dir=qual_dir, colors=class_colors, class_names=class_names,
                           ignore_index=cfg.data.ignore_index,
                           ndsm_mean=float(ndsm_norm.mean) if ndsm_norm else 0.0,
                           ndsm_std=float(ndsm_norm.std) if ndsm_norm else 1.0,
                           entropy_gamma=cfg.entropy_gamma,
                           autocast=cfg.autocast)

    hydra.core.global_hydra.GlobalHydra.instance().clear()


if __name__ == "__main__":
    import sys

    _require_exp_name(sys.argv[1:], "test.py")

    # # pick the test data config matching exp_name, unless the user named one
    if not any(a.startswith("data=") for a in sys.argv[1:]):
        exp_arg = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("exp_name=")), None)
        auto_cfg = _auto_data_config(exp_arg) if exp_arg else None
        if auto_cfg:
            print(f"[test.py] Auto-selected data config: {auto_cfg}")
            # insert before any flag so hydra reads it as an override, not a stray argument
            first_flag = next((i for i, a in enumerate(sys.argv[1:], 1) if a.startswith("-")), None)
            if first_flag is None:
                sys.argv.append(f"data={auto_cfg}")
            else:
                sys.argv.insert(first_flag, f"data={auto_cfg}")
    main()
