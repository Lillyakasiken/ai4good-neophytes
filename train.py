"""Train a segmentation model on the neophyte drone tiles.

    python train.py                                   # defaults from configs/train.yaml
    python train.py data=neophytes_split_cv1_train_1024 model=model_neophytes_mit_b2
    python train.py model.lr=0.0002 data.batch_size_train=4

Everything is configured through Hydra (configs/), so any value in the printed
config can be overridden on the command line with ``key=value``.

The run writes to ``lightning_logs/<exp_name>/``; evaluate it afterwards with
``python test.py exp_name=<exp_name>``.
"""

import hydra
from omegaconf import DictConfig, OmegaConf


def _make_model(cfg):
    from models.semseg_plm import SegmentationModel
    from utils.eval_utils import class_stats_names

    return SegmentationModel(
        cfg.model.model_name,
        cfg.model.encoder_name,
        cfg.model.img_size,
        cfg.model.num_classes,
        cfg.model.lr,
        in_channels=cfg.model.in_channels,
        ignore_index=cfg.data.ignore_index,
        optimizer=cfg.model.optimizer,
        lr_scheduler=cfg.model.lr_scheduler,
        loss=cfg.model.loss,
        weight=cfg.model.weight,
        patch_2_img_size=cfg.model.patch_2_img_size,
        fusion_mode=cfg.model.fusion_mode,
        phenology_names=list(cfg.data.phenology_names) if "phenology_names" in cfg.data else None,
        class_names=class_stats_names(cfg.data.classes) if "classes" in cfg.data else None,
        phenology_min_px=cfg.model.phenology_min_px,
    )


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    # # load only the elevation rasters the model actually consumes
    if "dsm" in cfg.model.in_channels: cfg.data.load_dsm = True
    if "dtm" in cfg.model.in_channels: cfg.data.load_dtm = True
    if "ndsm" in cfg.model.in_channels: cfg.data.load_ndsm = True
    print(OmegaConf.to_yaml(cfg))

    # # Imports inside main: hydra resolves the config before torch/cuda start up,
    # # so a config typo fails in seconds instead of after the import overhead.
    import os
    import time

    import matplotlib
    import pytorch_lightning as pl
    import torch
    import torch.multiprocessing
    from pytorch_lightning import seed_everything
    from pytorch_lightning.loggers import CSVLogger, WandbLogger

    from datasets.neophyte_datamodule import NeophyteDataModule
    from utils.callback_utils import get_callbacks
    from utils.eval_utils import class_stats_names
    from utils.novel_split import _settings, apply_novel_image_dirs
    from utils.transform_utils import get_transforms

    print('Imports completed.')

    # # avoids shared-memory allocation errors with many dataloader workers
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.set_float32_matmul_precision('medium')
    # Fixed crop and tile shapes, and deterministic_train is already off, so letting
    # cuDNN pick the fastest convolution is free. Skip it when determinism is requested.
    torch.backends.cudnn.benchmark = not cfg.deterministic_train

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    matplotlib.rcParams.update(cfg.matplotlib_style)

    if cfg.pl_seed:
        seed_everything(cfg.pl_seed, workers=True)

    # %% data
    # Rewrites train/val/test dirs when novel.test is "flights". Fold mode is unchanged.
    apply_novel_image_dirs(cfg)
    novel = cfg.data.novel if "novel" in cfg.data else None
    novel_species, _novel_flights, _novel_test, novel_stage, _novel_exclude, _novel_place = _settings(novel)
    finetune = cfg.get("finetune") if novel_species and novel_stage == "finetune" else None
    train_transforms = get_transforms(cfg.data.train_transforms, cfg.data.transforms_additional_targets)
    val_transforms = get_transforms(cfg.data.val_transforms, cfg.data.transforms_additional_targets)
    test_transforms = get_transforms(cfg.data.test_transforms, cfg.data.transforms_additional_targets)
    dsm_transforms = get_transforms(cfg.data.dsm_transforms, toTensor=False)
    ndsm_transforms = get_transforms(cfg.data.ndsm_transforms, toTensor=False)

    dataModule = NeophyteDataModule(cfg.data.image_dir,
                                    cfg.data.classes,
                                    train_transforms,
                                    val_transforms,
                                    test_transforms,
                                    dsm_transform=dsm_transforms,
                                    ndsm_transform=ndsm_transforms,
                                    dataset_stats=cfg.data.dataset_stats,
                                    weighted_sampling=cfg.data.weighted_sampling,
                                    cap_quantile=cfg.data.cap_quantile,
                                    lift_quantile=cfg.data.lift_quantile,
                                    bg_fraction=cfg.data.bg_fraction,
                                    min_strat_images=cfg.data.min_strat_images,
                                    value_mapping=cfg.data.value_mapping,
                                    ignore_index=cfg.data.ignore_index,
                                    load_auxiliary=cfg.data.load_auxiliary,
                                    load_dsm=cfg.data.load_dsm,
                                    load_dtm=cfg.data.load_dtm,
                                    load_ndsm=cfg.data.load_ndsm,
                                    few_shot_k=cfg.data.few_shot_k,
                                    few_shot_mode=cfg.data.few_shot_mode,
                                    few_shot_bg_k=cfg.data.few_shot_bg_k,
                                    few_shot_species=cfg.data.few_shot_species,
                                    train_site=cfg.data.train_site,
                                    train_stages=cfg.data.train_stages,
                                    stage_apply_val=cfg.data.stage_apply_val,
                                    max_nodata_frac=cfg.data.max_nodata_frac,
                                    val_few_shot_k=cfg.data.val_few_shot_k,
                                    val_few_shot_mode=cfg.data.val_few_shot_mode,
                                    val_few_shot_bg_k=cfg.data.val_few_shot_bg_k,
                                    val_subset=cfg.data.val_subset,
                                    load_into_ram=cfg.data.load_into_ram,
                                    novel=novel,
                                    novel_fraction=finetune.get("novel_fraction") if finetune else None,
                                    batch_size_train=cfg.data.batch_size_train,
                                    batch_size_val=cfg.data.batch_size_val,
                                    batch_size_test=cfg.data.batch_size_test,
                                    num_workers=cfg.data.num_workers)

    dataModule.prepare_data()
    dataModule.setup()
    print('# Train imgs: ', len(dataModule.train_dataset))
    print('# Val imgs:   ', len(dataModule.val_dataset))
    print('# Test imgs:  ', len(dataModule.test_dataset))

    # One sample, on the main process. A DataLoader here would start the persistent
    # worker pool twice (this check, then Trainer) and throw the first pool away.
    sample = dataModule.train_dataset[0]
    print('Image shape: ', tuple(sample['image'].shape), '\nMask shape:  ', tuple(sample['mask'].shape))

    # %% model
    model = _make_model(cfg)

    # %% training
    callbacks = get_callbacks(cfg.model.callbacks)
    if novel_species:
        from utils.novel_progress import NovelProgress
        stats_names = class_stats_names(cfg.data.classes)
        callbacks.append(NovelProgress(
            os.path.join(cfg.log_path, "progress.json"),
            novel_stage,
            novel_species,
            stats_names,
            cfg.model.max_epochs,
        ))
        # The five-species base still selects on the mean of those five.
        # A fine-tune has the new species in val, and that class is what the
        # saved checkpoint has to improve.
        if _novel_flights and novel_stage != "base":
            model.novel_class_index = stats_names.index(novel_species)
            for callback in callbacks:
                if getattr(callback, "monitor", None) == "val_f1_species":
                    callback.monitor = "val_f1_novel"
            print(f"Checkpoint metric: val_f1_novel ({novel_species})")
    if finetune and finetune.get("head_lr_mult"):
        model.head_lr_mult = float(finetune.head_lr_mult)
        print(f"Fine-tune lr: encoder {cfg.model.lr}, decoder and output layer "
              f"{cfg.model.lr * model.head_lr_mult:g}")
    if cfg.logger == 'wandb':
        logger = WandbLogger(name=cfg.exp_name, project=cfg.log_name, entity=cfg.wandb_entity)
    else:
        logger = CSVLogger(cfg.log_dir, name=cfg.exp_name)
    trainer = pl.Trainer(max_epochs=cfg.model.max_epochs,
                         precision=cfg.model.precision,
                         callbacks=callbacks,
                         logger=logger,
                         deterministic=cfg.deterministic_train,
                         check_val_every_n_epoch=cfg.model.check_val_every_n_epoch)

    # # resume an interrupted run: same exp_name -> same folder -> picks up last.ckpt.
    # # Passing it as ckpt_path restores the full training state (optimizer, epoch,
    # # LR schedule, callbacks), not just the weights -- loading only the weights would
    # # restart the cosine schedule at the initial LR and train max_epochs more epochs.
    last_ckpt_path = os.path.join(cfg.log_path, 'checkpoints/last.ckpt')
    resume_ckpt_path = last_ckpt_path if os.path.exists(last_ckpt_path) else None
    if resume_ckpt_path:
        print(f'Resuming training from: {resume_ckpt_path}')
    else:
        init_ckpt = cfg.init_ckpt if "init_ckpt" in cfg else None
        if init_ckpt and str(init_ckpt) not in ("", "null", "None"):
            print(f'Loading weights only from: {init_ckpt}')
            loaded = torch.load(str(init_ckpt), map_location='cpu', weights_only=False)
            model.load_state_dict(loaded['state_dict'])
            head_from = str(finetune.get("head_from") or "mean") if finetune else None
            if head_from:
                if head_from not in ("mean", "random"):
                    raise ValueError("finetune.head_from must be 'mean' or 'random'")
                stats_names = class_stats_names(cfg.data.classes)
                index = stats_names.index(novel_species)
                old_bias, new_bias = model.seed_class_head(index, head_from)
                shown = ("the average of the other species" if head_from == "mean"
                         else "random weights, bias 0")
                print(f"Output layer for {novel_species} restarted from {shown} "
                      f"(bias {old_bias:.3f} -> {new_bias:.3f})")

    start_time = time.time()
    trainer.fit(model, dataModule, ckpt_path=resume_ckpt_path)
    print('Training finished. Elapsed Time:', round((time.time() - start_time) / 60, 2), 'min')

    # %% quick test-set pass with the best checkpoint
    # # (the detailed evaluation lives in test.py). model.run_final_test=false skips
    # # it when test.py is run afterwards; the default is to keep this pass.
    if cfg.model.run_final_test:
        ckpt_path = trainer.checkpoint_callback.best_model_path
        print('ckpt_path: ', ckpt_path)
        from models.semseg_plm import SegmentationModel
        model = SegmentationModel.load_from_checkpoint(ckpt_path, weights_only=False).to(device)
        model.eval()
        trainer.test(model=model, dataloaders=dataModule)
    else:
        print('Skipping the end-of-training test pass (model.run_final_test=false). '
              'Run test.py for the full evaluation.')

    if cfg.logger == 'wandb':
        import wandb
        wandb.finish()

    hydra.core.global_hydra.GlobalHydra.instance().clear()


if __name__ == "__main__":
    main()
