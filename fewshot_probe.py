"""Few-shot evaluation on a held-out species (docs/plantclef2024_vit_analysis_zh.md 8.3).

Linear probes: a frozen backbone, a BatchNorm + 1x1 conv head on its features, trained
on the few-shot subset of a data config (one head per few_shot_seed, i.e. per draw of
the k tiles), all heads then scored in one pass over the test split:

    python fewshot_probe.py probe --backbone pc24   --out results/fewshot/cv3_bdavidii/probe_pc24
    python fewshot_probe.py probe --backbone ckpt:lightning_logs/<run>/checkpoints/best_epoch=23.ckpt \
        --out results/fewshot/cv3_bdavidii/probe_A

Fine-tuned models (train.py init_ckpt=... test_ckpt=null), scored the same way, several
checkpoints in one pass:

    python fewshot_probe.py eval --ckpts <run_s1>/checkpoints/final.ckpt <run_s2>/... \
        --names s1 s2 --out results/fewshot/cv3_bdavidii/ft_A

    python fewshot_probe.py check     # dataset stats of the held-out / few-shot configs (CPU)
    python fewshot_probe.py summary --root results/fewshot/cv3_bdavidii

Backbones:
    pc24, dinov2  ViT-B/14, outputs of blocks 3/6/9/12 concatenated (stride 14); 512
                  windows on the test tiles, the size the probe is trained on
    mit_b2        SMP MiT-b2 with ImageNet weights, the four stages upsampled to stride 4;
                  whole tiles, as the baseline is evaluated
    ckpt:<path>   the encoder of a trained SegmentationModel (its pyramid for a ViT)

Each head / checkpoint writes <out>/<name>.json: confusion matrix (total and per site),
per-class F1 / precision / recall, per-class average precision (threshold-free, from
1000-bin score histograms) and the training tiles. The probes run without validation,
like the fine-tuning: a new species has no validation labels.
"""

import argparse
import glob
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TRAIN_DATA = 'neophytes_split_cv3_train_512_fewshot_bdavidii'
HOLDOUT_DATA = 'neophytes_split_cv3_train_512_holdout_bdavidii'
TEST_DATA = 'neophytes_split_cv3_test'
VIT_MODEL_CFG = {'pc24': 'model_neophytes_vit_pc24', 'dinov2': 'model_neophytes_vit_dinov2'}
N_BINS = 1000


# =============================================================================
# config + data
# =============================================================================

def compose_cfg(overrides):
    """The composed train.py config (Hydra compose API) for the given overrides."""
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=os.path.abspath('configs'), version_base=None):
        return compose(config_name='train', overrides=overrides)


def make_datamodule(d, num_workers, train=True):
    """NeophyteDataModule from a data config, as in train.py (test.py for train=False)."""
    from datasets.neophyte_datamodule import NeophyteDataModule
    from utils.transform_utils import get_transforms

    test_tf = get_transforms(d.test_transforms, d.transforms_additional_targets)
    if train:
        train_tf = get_transforms(d.train_transforms, d.transforms_additional_targets)
        val_tf = get_transforms(d.val_transforms, d.transforms_additional_targets)
    else:
        train_tf = val_tf = test_tf
    return NeophyteDataModule(d.image_dir, d.classes, train_tf, val_tf, test_tf,
                              dsm_transform=get_transforms(d.dsm_transforms, toTensor=False),
                              ndsm_transform=get_transforms(d.ndsm_transforms, toTensor=False),
                              dataset_stats=d.dataset_stats,
                              weighted_sampling=d.weighted_sampling if train else False,
                              cap_quantile=d.cap_quantile, lift_quantile=d.lift_quantile,
                              bg_fraction=d.bg_fraction, min_strat_images=d.min_strat_images,
                              value_mapping=d.value_mapping, ignore_index=d.ignore_index,
                              load_auxiliary=d.load_auxiliary, load_dsm=d.load_dsm,
                              load_dtm=d.load_dtm, load_ndsm=d.load_ndsm,
                              few_shot_k=d.few_shot_k if train else None,
                              few_shot_mode=d.few_shot_mode, few_shot_bg_k=d.few_shot_bg_k,
                              few_shot_seed=d.few_shot_seed, max_nodata_frac=d.max_nodata_frac,
                              load_into_ram=False, batch_size_train=d.batch_size_train,
                              batch_size_val=d.batch_size_val, batch_size_test=d.batch_size_test,
                              num_workers=num_workers)


def test_loader(test_data, num_workers, need_ndsm):
    d = compose_cfg([f'data={test_data}']).data
    # # canopy/phenology and DSM/DTM only slow the loading down unless a model reads them
    d.load_auxiliary = False
    d.load_dsm = d.load_dtm = d.load_ndsm = need_ndsm
    dm = make_datamodule(d, num_workers, train=False)
    dm.setup('test')
    return dm.test_dataloader(), [d.classes[i].name for i in sorted(d.classes.keys(), key=int)], d.ignore_index


# =============================================================================
# frozen features
# =============================================================================

class ViTFeatures(nn.Module):
    """Concatenated outputs of four ViT blocks on the stride-14 token grid."""

    def __init__(self, timm_name, weights_path, block_indices=(2, 5, 8, 11)):
        super().__init__()
        import timm
        self.vit = timm.create_model(timm_name, pretrained=weights_path is None, num_classes=0,
                                     dynamic_img_size=True)
        if weights_path:
            self.vit.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
        self.block_indices = list(block_indices)
        self.pad_to = self.vit.patch_embed.patch_size[0]
        self.out_channels = self.vit.embed_dim * len(block_indices)
        self.sliding = True

    def forward(self, x):
        h, w = x.shape[-2:]
        x = F.pad(x, (0, -w % self.pad_to, 0, -h % self.pad_to))
        tokens = self.vit.forward_intermediates(x, indices=self.block_indices, norm=True,
                                                output_fmt='NCHW', intermediates_only=True)
        return torch.cat(tokens, dim=1)


class EncoderFeatures(nn.Module):
    """Stride 4/8/16/32 features of an SMP-style encoder, upsampled to stride 4."""

    def __init__(self, encoder, sliding=False):
        super().__init__()
        self.encoder = encoder
        self.pad_to = 32
        self.out_channels = sum(encoder.out_channels[2:])
        self.sliding = sliding

    def forward(self, x):
        h, w = x.shape[-2:]
        x = F.pad(x, (0, -w % self.pad_to, 0, -h % self.pad_to))
        feats = self.encoder(x)[2:]
        size = feats[0].shape[-2:]
        return torch.cat([feats[0]] + [F.interpolate(f, size=size, mode='bilinear', align_corners=False)
                                       for f in feats[1:]], dim=1)


def build_backbone(spec):
    if spec in VIT_MODEL_CFG:
        m = compose_cfg([f'model={VIT_MODEL_CFG[spec]}']).model
        return ViTFeatures(m.encoder_name, m.encoder_weights_path)
    if spec == 'mit_b2':
        import segmentation_models_pytorch as smp
        return EncoderFeatures(smp.encoders.get_encoder('mit_b2', in_channels=3, depth=5, weights='imagenet'))
    if spec.startswith('ckpt:'):
        from models.semseg_plm import SegmentationModel
        from models.vit_encoder import ViTPyramidEncoder
        model = SegmentationModel.load_from_checkpoint(spec[len('ckpt:'):], weights_only=False,
                                                       map_location='cpu')
        if model.elevation_fusion is not None:
            raise ValueError('ckpt backbones are RGB only; score nDSM models with "eval"')
        encoder = model.model.encoder
        return EncoderFeatures(encoder, sliding=isinstance(encoder, ViTPyramidEncoder))
    raise ValueError(f'Unknown backbone {spec!r}')


def head_logits(backbone, head, x):
    """Pixel logits of a probe head on input x (features computed once per call)."""
    feats = backbone(x).float()
    return pixel_logits(head(feats), x, backbone.pad_to)


def pixel_logits(logits, x, pad_to):
    h, w = x.shape[-2:]
    hp, wp = h + (-h % pad_to), w + (-w % pad_to)
    return F.interpolate(logits, size=(hp, wp), mode='bilinear', align_corners=False)[..., :h, :w]


# =============================================================================
# scoring
# =============================================================================

class Scorer:
    """Confusion matrix (total + per site) and per-class score histograms for AP."""

    def __init__(self, num_classes, ignore_index, device):
        self.k, self.ignore_index = num_classes, ignore_index
        self.conf = torch.zeros(num_classes * num_classes, dtype=torch.long, device=device)
        self.site_conf = {}
        # # [negatives | positives] x class x bin
        self.hist = torch.zeros(2 * num_classes * N_BINS, dtype=torch.long, device=device)
        self.offsets = torch.arange(num_classes, device=device) * N_BINS

    def update(self, probs, target, sites):
        k = self.k
        for b in range(probs.shape[0]):
            valid = target[b] != self.ignore_index
            t = target[b][valid]
            p = probs[b].permute(1, 2, 0)[valid].float()
            conf = torch.bincount(t * k + p.argmax(1), minlength=k * k)
            self.conf += conf
            self.site_conf[sites[b]] = self.site_conf.get(sites[b], 0) + conf
            bins = (p * N_BINS).long().clamp_(max=N_BINS - 1) + self.offsets
            pos = t.unsqueeze(1) == torch.arange(k, device=t.device)
            self.hist += torch.bincount((bins + pos.long() * k * N_BINS).flatten(),
                                        minlength=2 * k * N_BINS)

    def result(self, class_names):
        k = self.k
        conf = self.conf.view(k, k).cpu().numpy()
        out = dict(class_names=class_names, conf=conf.tolist(), **class_scores(conf))
        hist = self.hist.view(2, k, N_BINS).cpu().numpy().astype(np.float64)
        out['AP'] = [average_precision(hist[1, c], hist[0, c]) for c in range(k)]
        out['AP-avg-wo0'] = float(np.nanmean(out['AP'][1:]))
        out['site_conf'] = {s: c.view(k, k).cpu().numpy().tolist() for s, c in self.site_conf.items()}
        return out


def class_scores(conf):
    tp = np.diag(conf).astype(np.float64)
    fp, fn = conf.sum(0) - tp, conf.sum(1) - tp
    with np.errstate(invalid='ignore', divide='ignore'):
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan)
        precision = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
        recall = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
    return {'F1': f1.tolist(), 'Precision': precision.tolist(), 'Recall': recall.tolist(),
            'F1-avg-wo0': float(np.nanmean(f1[1:]))}


def average_precision(pos, neg):
    """Step-wise AP from score histograms (thresholds at the bin edges, high to low)."""
    if pos.sum() == 0:
        return float('nan')
    tp, fp = np.cumsum(pos[::-1]), np.cumsum(neg[::-1])
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / pos.sum()
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def write_json(path, result):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(result, f, indent=1)


def score(predictors, loader, class_names, ignore_index, out_dir, max_batches=None, extra=None):
    """One pass over the test split; predictors: {name: fn(batch) -> probs (B, K, H, W)}."""
    from tqdm import tqdm
    from utils.eval_utils import extract_site

    scorers = {name: Scorer(len(class_names), ignore_index, 'cuda') for name in predictors}
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc='Scoring')):
            if max_batches is not None and i >= max_batches:
                break
            target = batch['mask'].cuda().long()
            sites = [extract_site(n) for n in batch['name']]
            for name, predict in predictors.items():
                scorers[name].update(predict(batch), target, sites)
    for name, scorer in scorers.items():
        result = scorer.result(class_names)
        result.update((extra or {}).get(name, {}))
        write_json(os.path.join(out_dir, f'{name}.json'), result)
        print(f"{name}: F1 {np.round(result['F1'], 3).tolist()}  AP {np.round(result['AP'], 3).tolist()}")


# =============================================================================
# modes
# =============================================================================

def train_probe(backbone, seed, args):
    """Train one linear head on the few-shot subset of draw ``seed``."""
    import segmentation_models_pytorch as smp
    from pytorch_lightning import seed_everything

    seed_everything(seed, workers=True)
    d = compose_cfg([f'data={args.train_data}', f'data.few_shot_seed={seed}']).data
    dm = make_datamodule(d, args.num_workers)
    dm.setup('fit')
    loader = dm.train_dataloader()

    num_classes = len(d.classes)
    head = nn.Sequential(nn.BatchNorm2d(backbone.out_channels),
                         nn.Conv2d(backbone.out_channels, num_classes, 1)).cuda()
    steps = args.epochs * len(loader)
    if args.max_steps:
        steps = min(steps, args.max_steps)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=args.lr * 0.01)
    criterion = smp.losses.FocalLoss(mode='multiclass', ignore_index=d.ignore_index)

    head.train()
    step = 0
    for epoch in range(math.ceil(steps / len(loader))):
        for batch in loader:
            if step >= steps:
                break
            x, y = batch['image'].cuda(non_blocking=True), batch['mask'].cuda(non_blocking=True).long()
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                feats = backbone(x)
            loss = criterion(pixel_logits(head(feats.float()), x, backbone.pad_to), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
        print(f'seed {seed} epoch {epoch}: step {step}/{steps}, loss {loss.item():.4f}')
    head.eval()
    tiles = [os.path.basename(p) for p in dm.train_dataset.image_files]
    return head, {'seed': seed, 'steps': steps, 'lr': args.lr, 'train_tiles': tiles}


def run_probe(args):
    from models.semseg_plm import merge_output_patches_to_image, patch_image_to_batch

    backbone = build_backbone(args.backbone).cuda().eval().requires_grad_(False)
    heads, extra = {}, {}
    for seed in args.seeds:
        heads[f's{seed}'], extra[f's{seed}'] = train_probe(backbone, seed, args)
        torch.save(heads[f's{seed}'].state_dict(), os.path.join(make_out(args.out), f's{seed}_head.pt'))
    for e in extra.values():
        e['backbone'] = args.backbone

    loader, class_names, ignore_index = test_loader(args.test_data, args.num_workers, need_ndsm=False)
    cache = {}

    def features(batch):
        """Backbone features of the batch (shared by all heads), sliding or whole tile."""
        if cache.get('id') is not batch:
            x = batch['image'].cuda(non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                if backbone.sliding:
                    patches, num_patches, stride, pad = patch_image_to_batch(x, args.window)
                    cache.update(id=batch, x=x, inp=patches, feats=backbone(patches).float(),
                                 merge=(num_patches, stride, pad))
                else:
                    cache.update(id=batch, x=x, inp=x, feats=backbone(x).float(), merge=None)
        return cache

    def predictor(head):
        def predict(batch):
            c = features(batch)
            logits = pixel_logits(head(c['feats']), c['inp'], backbone.pad_to)
            if c['merge'] is not None:
                num_patches, stride, pad = c['merge']
                logits = merge_output_patches_to_image(logits, num_patches, args.window, stride,
                                                       c['x'].shape[0], c['x'].shape[-1], pad)
            return torch.softmax(logits, dim=1)
        return predict

    score({name: predictor(h) for name, h in heads.items()}, loader, class_names, ignore_index,
          args.out, args.max_test_batches, extra)


def run_eval(args):
    from models.semseg_plm import SegmentationModel

    names = args.names or [f'm{i}' for i in range(len(args.ckpts))]
    models = {}
    for name, path in zip(names, args.ckpts):
        model = SegmentationModel.load_from_checkpoint(path, weights_only=False, map_location='cpu')
        model.set_patch_2_img_size(False)  # whole tiles, as test.py
        models[name] = model.cuda().eval()
    need_ndsm = any('ndsm' in m.in_channels for m in models.values())
    loader, class_names, ignore_index = test_loader(args.test_data, args.num_workers, need_ndsm)

    def predictor(model):
        def predict(batch):
            inputs, aux = model.prepare_inputs(batch)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                return model(inputs.cuda(non_blocking=True), aux.cuda() if aux is not None else None)
        return predict

    extra = {name: {'ckpt': path} for name, path in zip(names, args.ckpts)}
    score({name: predictor(m) for name, m in models.items()}, loader, class_names, ignore_index,
          args.out, args.max_test_batches, extra)


def run_check(args):
    """Stats-level check of the two few-shot data configs (no images are read)."""
    from datasets.neophyte_dataset import NeophyteDataset

    def dataset(d):
        return NeophyteDataset(d.image_dir.train, d.classes, dataset_stats=d.dataset_stats,
                               weighted_sampling=d.weighted_sampling, cap_quantile=d.cap_quantile,
                               lift_quantile=d.lift_quantile, bg_fraction=d.bg_fraction,
                               min_strat_images=d.min_strat_images, value_mapping=d.value_mapping,
                               ignore_index=d.ignore_index, few_shot_k=d.few_shot_k,
                               few_shot_mode=d.few_shot_mode, few_shot_bg_k=d.few_shot_bg_k,
                               few_shot_seed=d.few_shot_seed, max_nodata_frac=d.max_nodata_frac)

    ds = dataset(compose_cfg([f'data={args.holdout_data}']).data)
    w = np.asarray(ds.sample_weights)
    bg_share = w[ds.stats_df['only_background'].to_numpy() == 1].sum() / w.sum()
    print(f'{args.holdout_data}: {len(ds)} tiles, B_davidii_px left: {int(ds.stats_df["B_davidii_px"].sum())}, '
          f'sampling share of background-only tiles: {bg_share:.3f}')
    assert ds.stats_df['B_davidii_px'].sum() == 0, 'held-out species still in the stats'

    draws = {}
    for seed in args.seeds:
        d = compose_cfg([f'data={args.train_data}', f'data.few_shot_seed={seed}']).data
        ds = dataset(d)
        new = ds.stats_df.index[ds.stats_df['B_davidii_px'] > 0]
        draws[seed] = set(new)
        w = np.asarray(ds.sample_weights)
        share = w[(ds.stats_df['B_davidii_px'] > 0).to_numpy()].sum() / w.sum()
        print(f'seed {seed}: {len(ds)} tiles, {len(new)} with B. davidii, sampling share {share:.3f}')
    seeds = list(draws)
    for i, a in enumerate(seeds):
        for b in seeds[i + 1:]:
            print(f'B. davidii tiles shared by seeds {a} and {b}: {len(draws[a] & draws[b])}')


def run_summary(args):
    """Table of the new-species scores per method and draw, plus the known species."""
    rows = []
    for path in sorted(glob.glob(os.path.join(args.root, '*', 's*.json'))):
        r = json.load(open(path))
        c = args.new_class
        known = [f for i, f in enumerate(r['F1']) if i not in (0, c)]
        rows.append((os.path.basename(os.path.dirname(path)), os.path.basename(path)[:-5],
                     r['F1'][c], r['Recall'][c], r['Precision'][c], r['AP'][c], float(np.nanmean(known))))
    if not rows:
        print('no results under', args.root)
        return
    lines = [f'new species: class {args.new_class}; known = mean F1 of the other species',
             f'{"method":<14}{"draw":<6}{"F1":>7}{"recall":>8}{"prec":>7}{"AP":>7}{"known":>8}']
    lines += [f'{m:<14}{s:<6}' + ''.join(f'{v:>{w}.3f}' for v, w in zip(vals, (7, 8, 7, 7, 8)))
              for m, s, *vals in rows]
    lines.append('')
    for method in dict.fromkeys(r[0] for r in rows):
        v = np.array([r[2:] for r in rows if r[0] == method])
        lines.append(f'{method:<14}mean  F1 {v[:, 0].mean():.3f} [{v[:, 0].min():.3f}, {v[:, 0].max():.3f}]  '
                     f'AP {v[:, 3].mean():.3f} [{v[:, 3].min():.3f}, {v[:, 3].max():.3f}]  '
                     f'known F1 {v[:, 4].mean():.3f}  (n={len(v)})')
    text = '\n'.join(lines)
    print(text)
    with open(os.path.join(args.root, 'summary.txt'), 'w') as f:
        f.write(text + '\n')


def make_out(path):
    os.makedirs(path, exist_ok=True)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode', choices=['probe', 'eval', 'check', 'summary'])
    p.add_argument('--backbone', help='probe: pc24, dinov2, mit_b2 or ckpt:<path>')
    p.add_argument('--ckpts', nargs='+', help='eval: checkpoints to score')
    p.add_argument('--names', nargs='+', help='eval: output names, one per checkpoint')
    p.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3], help='few-shot draws')
    p.add_argument('--train_data', default=TRAIN_DATA)
    p.add_argument('--holdout_data', default=HOLDOUT_DATA, help='check: base-training config')
    p.add_argument('--test_data', default=TEST_DATA)
    p.add_argument('--epochs', type=int, default=30, help='as many steps as a 30-epoch fine-tuning')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--window', type=int, default=512, help='sliding window of the ViT probes')
    p.add_argument('--num_workers', type=int, default=16)
    p.add_argument('--max_steps', type=int, default=None, help='smoke test: cap the training steps')
    p.add_argument('--max_test_batches', type=int, default=None, help='smoke test: cap the test pass')
    p.add_argument('--out', help='output directory')
    p.add_argument('--root', default='results/fewshot/cv3_bdavidii', help='summary: results root')
    p.add_argument('--new_class', type=int, default=1, help='summary: class index of the new species')
    args = p.parse_args()

    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.set_float32_matmul_precision('medium')
    {'probe': run_probe, 'eval': run_eval, 'check': run_check, 'summary': run_summary}[args.mode](args)


if __name__ == '__main__':
    main()
