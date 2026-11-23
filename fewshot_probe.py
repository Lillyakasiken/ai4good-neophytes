"""Few-shot evaluation on a held-out species (docs/plantclef2024_fewshot_en.md).

Linear probes: a frozen backbone, a BatchNorm + 1x1 conv head on its features, trained
on a few-shot subset of a training split. A grid of heads (new species x k x draw) is
trained one after the other, then all of them are scored in ONE pass over the test split:

    python fewshot_probe.py probe --backbone pc24 --fold cv3 --new_species all --k 10 \
        --seeds 1 2 3 --method probe_pc24 --out results/fewshot/loso/cv3_pc24_k10

Each head sees k labelled tiles of its new species, --replay_k tiles of every other
species and --bg_k background-only tiles. The draws are nested: with the same seed the
k = 3 tiles are among the k = 10 tiles, and the replay and background tiles do not
depend on the new species or on k. --draw flight takes the k tiles from ONE flight
(seeds 1, 2, 3 = three flights) instead of from all training sites; --draw flights<n>
spreads them over n flights (about k/n tiles each).

--calibrate (with --new_species) picks a bias for the logit of the new species without
any validation label: the few-shot subset is split in two halves, a head trained on one
half scores the other, and the bias with the best F1 of the new species is kept. The
test pass then also records the confusion matrix for every bias of CAL_DELTAS
('bias_sweep'), so the chosen bias can be compared with the best one in hindsight.
--heads_from reuses heads trained by an earlier job (e.g. to calibrate them):

    python fewshot_probe.py probe --backbone pc24 --fold cv3 --new_species all --k 10 --seeds 1 2 3 \
        --heads_from results/fewshot/loso/cv3_pc24_k10 --calibrate --method probe_pc24_cal \
        --out results/fewshot/loso/cv3_pc24_k10_cal

Combined model (section 6.2): the known species from a base U-Net trained with the new
species as background, the new species from a linear head; one test pass:

    python fewshot_probe.py combo --base_ckpt <base A>/checkpoints/best_*.ckpt --backbone pc24 \
        --heads_from results/fewshot/loso/cv3_pc24_k10 --new_species B_davidii --k 10 --seeds 1 2 3 \
        --method combo_A_pc24 --out results/fewshot/loso/cv3_combo_A_pc24_B_davidii

Without --new_species the subset comes from a data config (--train_data), one head per
few_shot_seed, as in the October runs:

    python fewshot_probe.py probe --backbone pc24 --out results/fewshot/cv3_bdavidii/probe_pc24

Fine-tuned models (train.py init_ckpt=... test_ckpt=null), scored the same way, several
checkpoints in one pass:

    python fewshot_probe.py eval --ckpts <run_s1>/checkpoints/final.ckpt <run_s2>/... \
        --names s1 s2 --method ft_P --new_species B_davidii --k 10 --out results/fewshot/loso/cv3_ft_P

    python fewshot_probe.py check --fold cv3 --new_species all --k 1 3 10 30   # tiles per draw (CPU)
    python fewshot_probe.py summary --root results/fewshot/loso

Backbones:
    pc24, dinov2    ViT-B/14, outputs of blocks 3/6/9/12 concatenated (stride 14); 512
                    windows on the test tiles, the size the probe is trained on
    vitckpt:<path>  the same features from the ViT of a trained ViT U-Net
    mit_b2          SMP MiT-b2 with ImageNet weights, the four stages upsampled to stride 4;
                    whole tiles, as the baseline is evaluated
    fusion          pc24 tokens upsampled onto the stride-4 grid of mit_b2 and concatenated
                    (species semantics + detail, section 6.3); 512 windows
    ckpt:<path>     the encoder of a trained SegmentationModel (its pyramid for a ViT)

Each head / checkpoint writes <out>/<name>.json: confusion matrix (total and per site),
per-class F1 / precision / recall, the average precision of the new species
(threshold-free, from 1000-bin score histograms; of every class without --new_species)
and the training tiles. The probes run without validation, like the fine-tuning: a new
species has no validation labels. A trained head is saved at once (<name>_head.pt) and
reused when the job is started again.
"""

import argparse
import glob
import json
import math
import os
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TRAIN_DATA = 'neophytes_split_cv3_train_512_fewshot_bdavidii'
HOLDOUT_DATA = 'neophytes_split_cv3_train_512_holdout_bdavidii'
VIT_MODEL_CFG = {'pc24': 'model_neophytes_vit_pc24', 'dinov2': 'model_neophytes_vit_dinov2'}
N_BINS = 1000
# biases added to the logit (log-probability) of the new species: --calibrate
CAL_DELTAS = [-4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]


# =============================================================================
# config + data
# =============================================================================

def compose_cfg(overrides):
    """The composed train.py config (Hydra compose API) for the given overrides."""
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=os.path.abspath('configs'), version_base=None):
        return compose(config_name='train', overrides=overrides)


def species_classes(classes):
    """{stats_name: class index} of the species (no background), in class-index order."""
    return {classes[i].stats_name: int(i) for i in sorted(classes.keys(), key=int) if int(i) != 0}


def draw_mode(draw):
    """few_shot_mode of the new species for a --draw value (random, flight, flights<n>)."""
    if draw == 'random':
        return 'random'
    if draw == 'flight':
        return 'random_flight'
    m = re.fullmatch(r'flights(\d+)', draw)
    if m and int(m.group(1)) > 0:
        return f'random_flights{m.group(1)}'
    raise ValueError(f'unknown --draw {draw!r}: random, flight or flights<n>')


def fewshot_overrides(fold, species, new_species, k, seed, replay_k, bg_k, draw):
    """Hydra overrides of the few-shot subset: k tiles of new_species, replay_k of the others.

    For fold cv3, B_davidii, k = 10 this composes the same data config as
    neophytes_split_cv3_train_512_fewshot_bdavidii (the October runs).
    """
    few = ','.join(f'{s}:{k if s == new_species else replay_k}' for s in species)
    mode = 'random' if draw == 'random' else f'{{{new_species}:{draw_mode(draw)}}}'
    return [f'data=neophytes_split_{fold}_train_512', f'data.few_shot_k={{{few}}}',
            f'data.few_shot_mode={mode}', f'data.few_shot_bg_k={bg_k}',
            f'data.few_shot_seed={seed}', 'data.cap_quantile=null']


def probe_jobs(args):
    """The heads to train: [(name, data overrides, metadata)]."""
    if not args.new_species:
        return [(f's{s}', [f'data={args.train_data}', f'data.few_shot_seed={s}'], {'seed': s})
                for s in args.seeds]
    species = species_classes(compose_cfg([f'data=neophytes_split_{args.fold}_train_512']).data.classes)
    new = list(species) if args.new_species == ['all'] else args.new_species
    unknown = [s for s in new if s not in species]
    if unknown:
        raise ValueError(f'unknown species {unknown}; choose from {list(species)} or "all"')
    jobs = []
    for sp in new:
        for k in args.k:
            for seed in args.seeds:
                meta = dict(new_species=sp, new_class=species[sp], k=k, seed=seed, fold=args.fold,
                            draw=args.draw, replay_k=args.replay_k, bg_k=args.bg_k)
                jobs.append((f'{sp}_k{k}_s{seed}', fewshot_overrides(
                    args.fold, species, sp, k, seed, args.replay_k, args.bg_k, args.draw), meta))
    return jobs


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


def tile_flight(path):
    """Flight (e.g. Basel_4_2) of a tile path .../<flight>/<split>/images/<tile>."""
    return os.path.normpath(path).split(os.sep)[-4]


# =============================================================================
# frozen features
# =============================================================================

def timm_vit(timm_name, weights_path):
    import timm
    vit = timm.create_model(timm_name, pretrained=weights_path is None, num_classes=0,
                            dynamic_img_size=True)
    if weights_path:
        vit.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
    return vit


class ViTFeatures(nn.Module):
    """Concatenated outputs of four ViT blocks on the stride-14 token grid."""

    def __init__(self, vit, block_indices=(2, 5, 8, 11)):
        super().__init__()
        self.vit = vit
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


class FusionFeatures(nn.Module):
    """ViT tokens (stride 14) upsampled onto the stride-4 grid of an SMP encoder and
    concatenated with its features: species semantics + spatial detail (section 6.3).

    The input is padded to a multiple of 28, so the ViT grid (H/14) and the stride-4 grid
    (H/4) cover exactly the same pixels; the encoder pads to 32 itself and is cropped back.
    """

    def __init__(self, vit_features, encoder_features):
        super().__init__()
        self.vit, self.cnn = vit_features, encoder_features
        self.pad_to = 28
        self.out_channels = vit_features.out_channels + encoder_features.out_channels
        self.sliding = True

    def parts(self, x):
        """ViT tokens at their own stride 14, encoder features cropped to the stride-4 grid."""
        h, w = x.shape[-2:]
        x = F.pad(x, (0, -w % self.pad_to, 0, -h % self.pad_to))
        gh, gw = x.shape[-2] // 4, x.shape[-1] // 4
        return self.vit(x), self.cnn(x)[..., :gh, :gw]

    def forward(self, x):
        vit, cnn = self.parts(x)
        vit = F.interpolate(vit, size=cnn.shape[-2:], mode='bilinear', align_corners=False)
        return torch.cat([vit, cnn], dim=1)


def fusion_head_logits(head, vit, cnn, chunk=8):
    """head(FusionFeatures) without building the 1792-channel stride-4 tensor.

    An eval-mode BN + 1x1 conv is one linear map per pixel and bilinear upsampling
    (weights summing to 1) commutes with it, so the ViT half of the head is applied at
    stride 14 and its logits upsampled. Same result as head(forward(x)) up to rounding;
    the test pass otherwise caches ~13 GiB for a batch of whole tiles.
    """
    bn, conv = head
    assert not bn.training
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    w = conv.weight[:, :, 0, 0]
    b = conv.bias + w @ (bn.bias - bn.running_mean * scale)
    w = (w * scale)[:, :, None, None]
    nv = vit.shape[1]
    out = []
    for i in range(0, cnn.shape[0], chunk):
        lv = F.conv2d(vit[i:i + chunk].float(), w[:, :nv])
        lv = F.interpolate(lv, size=cnn.shape[-2:], mode='bilinear', align_corners=False)
        out.append(lv + F.conv2d(cnn[i:i + chunk].float(), w[:, nv:], b))
    return torch.cat(out)


def build_backbone(spec):
    if spec in VIT_MODEL_CFG:
        m = compose_cfg([f'model={VIT_MODEL_CFG[spec]}']).model
        return ViTFeatures(timm_vit(m.encoder_name, m.encoder_weights_path))
    if spec == 'mit_b2':
        import segmentation_models_pytorch as smp
        return EncoderFeatures(smp.encoders.get_encoder('mit_b2', in_channels=3, depth=5, weights='imagenet'))
    if spec == 'fusion':
        return FusionFeatures(build_backbone('pc24'), build_backbone('mit_b2'))
    if spec.startswith(('ckpt:', 'vitckpt:')):
        from models.semseg_plm import SegmentationModel
        from models.vit_encoder import ViTPyramidEncoder
        kind, path = spec.split(':', 1)
        model = SegmentationModel.load_from_checkpoint(path, weights_only=False, map_location='cpu')
        if model.elevation_fusion is not None:
            raise ValueError('ckpt backbones are RGB only; score nDSM models with "eval"')
        encoder = model.model.encoder
        if kind == 'vitckpt':
            # # the ViT tokens themselves, the same features as pc24 / dinov2
            if not isinstance(encoder, ViTPyramidEncoder):
                raise ValueError(f'vitckpt: needs a ViT U-Net checkpoint, {path} has {type(encoder).__name__}')
            return ViTFeatures(encoder.vit, encoder.block_indices)
        return EncoderFeatures(encoder, sliding=isinstance(encoder, ViTPyramidEncoder))
    raise ValueError(f'Unknown backbone {spec!r}')


def make_head(in_channels, num_classes):
    return nn.Sequential(nn.BatchNorm2d(in_channels), nn.Conv2d(in_channels, num_classes, 1))


def pixel_logits(logits, x, pad_to):
    h, w = x.shape[-2:]
    hp, wp = h + (-h % pad_to), w + (-w % pad_to)
    return F.interpolate(logits, size=(hp, wp), mode='bilinear', align_corners=False)[..., :h, :w]


# =============================================================================
# scoring
# =============================================================================

class Scorer:
    """Confusion matrix (total + per site) and score histograms for the AP of ap_classes.

    With new_class and deltas: also one confusion matrix per bias added to the
    log-probability of new_class before the argmax (the bias sweep of --calibrate).
    """

    def __init__(self, num_classes, ignore_index, device, ap_classes=None, new_class=None, deltas=None):
        self.k, self.ignore_index = num_classes, ignore_index
        self.ap_classes = list(range(num_classes)) if ap_classes is None else list(ap_classes)
        self.conf = torch.zeros(num_classes * num_classes, dtype=torch.long, device=device)
        self.site_conf = {}
        # # class x [negatives | positives] x bin
        self.hist = torch.zeros(len(self.ap_classes), 2 * N_BINS, dtype=torch.long, device=device)
        self.new_class = new_class
        self.deltas = list(deltas) if deltas is not None and new_class is not None else None
        if self.deltas:
            self.delta_conf = torch.zeros(len(self.deltas), num_classes * num_classes, dtype=torch.long,
                                          device=device)

    def _update_deltas(self, probs, target, ignore):
        k, c = self.k, self.new_class
        logp = probs.clamp_min(1e-30).log()
        new = logp[:, c]
        other, arg = logp.index_fill(1, torch.tensor([c], device=logp.device), float('-inf')).max(1)
        base = target * k
        for i, d in enumerate(self.deltas):
            pred = arg.masked_fill(new + d >= other, c)
            idx = torch.where(ignore, k * k, base + pred)
            self.delta_conf[i] += torch.bincount(idx.flatten(), minlength=k * k + 1)[:k * k]

    def update(self, probs, target, sites):
        """probs (B, K, H, W), target (B, H, W). Dense operations only: ignored pixels
        go to one extra bin that is dropped, instead of being masked out."""
        k = self.k
        probs = probs.float()
        ignore = target == self.ignore_index
        if self.deltas:
            self._update_deltas(probs, target, ignore)
        conf_idx = torch.where(ignore, k * k, target * k + probs.argmax(1))
        for b in range(probs.shape[0]):
            conf = torch.bincount(conf_idx[b].flatten(), minlength=k * k + 1)[:k * k]
            self.conf += conf
            self.site_conf[sites[b]] = self.site_conf.get(sites[b], 0) + conf
            for j, c in enumerate(self.ap_classes):
                bins = (probs[b, c] * N_BINS).long().clamp_(max=N_BINS - 1) + (target[b] == c) * N_BINS
                bins = torch.where(ignore[b], 2 * N_BINS, bins)
                self.hist[j] += torch.bincount(bins.flatten(), minlength=2 * N_BINS + 1)[:2 * N_BINS]

    def result(self, class_names):
        k = self.k
        conf = self.conf.view(k, k).cpu().numpy()
        out = dict(class_names=class_names, conf=conf.tolist(), **class_scores(conf))
        hist = self.hist.view(-1, 2, N_BINS).cpu().numpy().astype(np.float64)
        out['AP'] = [float('nan')] * k
        for j, c in enumerate(self.ap_classes):
            out['AP'][c] = average_precision(hist[j, 1], hist[j, 0])
        if len(self.ap_classes) == k:
            out['AP-avg-wo0'] = float(np.nanmean(out['AP'][1:]))
        out['site_conf'] = {s: c.view(k, k).cpu().numpy().tolist() for s, c in self.site_conf.items()}
        if self.deltas:
            out['bias_sweep'] = {'deltas': self.deltas,
                                 'conf': [c.view(k, k).cpu().numpy().tolist() for c in self.delta_conf]}
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


def score(predictors, loader, class_names, ignore_index, out_dir, max_batches=None, extra=None,
          ap_classes=None, sweeps=None, device='cuda'):
    """One pass over the test split; predictors: {name: fn(batch) -> probs (B, K, H, W)}.

    ap_classes: {name: classes whose AP is computed} (a name not listed: every class).
    sweeps: {name: (new class, biases)} for the bias sweep of --calibrate.
    """
    from tqdm import tqdm
    from utils.eval_utils import extract_site

    ap_classes, sweeps = ap_classes or {}, sweeps or {}
    scorers = {name: Scorer(len(class_names), ignore_index, device, ap_classes.get(name),
                            *sweeps.get(name, (None, None)))
               for name in predictors}
    n, t_first = 0, None
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc='Scoring', mininterval=30)):
            if max_batches is not None and i >= max_batches:
                break
            target = batch['mask'].to(device).long()
            sites = [extract_site(p) for p in batch['name']]
            for name, predict in predictors.items():
                scorers[name].update(predict(batch), target, sites)
            n += 1
            if t_first is None:  # the first batch pays for the worker start and the warm-up
                t_first = time.time()
    if n > 1:
        per_batch = (time.time() - t_first) / (n - 1)
        print(f'scoring speed: {per_batch:.2f} s/batch with {len(predictors)} predictors; '
              f'a full pass ({len(loader)} batches) takes ~{per_batch * len(loader) / 60:.0f} min', flush=True)
    for name, scorer in scorers.items():
        result = scorer.result(class_names)
        result.update((extra or {}).get(name, {}))
        write_json(os.path.join(out_dir, f'{name}.json'), result)
        c = result.get('new_class')
        if c is None:
            print(f"{name}: F1 {np.round(result['F1'], 3).tolist()}  AP {np.round(result['AP'], 3).tolist()}")
        else:
            print(f"{name}: new species F1 {result['F1'][c]:.3f}  recall {result['Recall'][c]:.3f}  "
                  f"precision {result['Precision'][c]:.3f}  AP {result['AP'][c]:.3f}  "
                  f"F1-avg-wo0 {result['F1-avg-wo0']:.3f}")


# =============================================================================
# modes
# =============================================================================

def fit_head(backbone, loader, num_classes, ignore_index, steps, lr, name):
    """Train a BatchNorm + 1x1 conv head on the frozen features: AdamW, cosine decay, focal loss."""
    import segmentation_models_pytorch as smp

    head = make_head(backbone.out_channels, num_classes).cuda()
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr * 0.01)
    criterion = smp.losses.FocalLoss(mode='multiclass', ignore_index=ignore_index)

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
        if epoch % 10 == 9 or step >= steps:
            print(f'{name} epoch {epoch}: step {step}/{steps}, loss {loss.item():.4f}', flush=True)
    return head.eval()


def probe_steps(args, loader):
    steps = args.epochs * len(loader)
    return min(steps, args.max_steps) if args.max_steps else steps


def train_probe(backbone, name, overrides, meta, args):
    """Train one linear head on the few-shot subset given by the data overrides."""
    from pytorch_lightning import seed_everything

    seed_everything(meta['seed'], workers=True)
    d = compose_cfg(overrides).data
    dm = make_datamodule(d, args.num_workers)
    dm.setup('fit')
    loader = dm.train_dataloader()

    num_classes = len(d.classes)
    steps = probe_steps(args, loader)
    start = time.time()
    head = fit_head(backbone, loader, num_classes, d.ignore_index, steps, args.lr, name)

    dataset = dm.train_dataset
    info = dict(meta, steps=steps, lr=args.lr, num_classes=num_classes, backbone=args.backbone,
                method=args.method or os.path.basename(os.path.normpath(args.out)),
                train_minutes=round((time.time() - start) / 60, 2),
                train_tiles=[os.path.basename(p) for p in dataset.image_files])
    if 'new_species' in meta:
        # # what the draw really contains: a species can have fewer than k candidate tiles
        new = dataset.stats_df.index[dataset.stats_df[f"{meta['new_species']}_px"] > 0]
        info.update(n_new_tiles=len(new), new_flights=sorted({tile_flight(p) for p in new}))
        print(f"{name}: {len(dataset)} tiles, {len(new)} with {meta['new_species']} "
              f"from {info['new_flights']}, {info['train_minutes']} min", flush=True)
    return head, info


class ProbePredictor:
    """Softmax of linear heads on backbone features that all heads share (computed once
    per batch): 512 windows for a sliding backbone, the whole tile otherwise."""

    def __init__(self, backbone, window):
        self.backbone, self.window, self.cache = backbone, window, {}

    def features(self, batch):
        from models.semseg_plm import patch_image_to_batch

        c = self.cache
        if c.get('id') is not batch:
            x = batch['image'].cuda(non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                if hasattr(self.backbone, 'parts'):
                    # # fusion: keep the two halves apart (bf16, in chunks of windows)
                    patches, num_patches, stride, pad = patch_image_to_batch(x, self.window)
                    vit, cnn = zip(*(self.backbone.parts(patches[i:i + 8])
                                     for i in range(0, patches.shape[0], 8)))
                    c.update(id=batch, x=x, inp=patches, merge=(num_patches, stride, pad),
                             feats=(torch.cat(vit).bfloat16(), torch.cat(cnn).bfloat16()))
                elif self.backbone.sliding:
                    patches, num_patches, stride, pad = patch_image_to_batch(x, self.window)
                    c.update(id=batch, x=x, inp=patches, feats=self.backbone(patches).float(),
                             merge=(num_patches, stride, pad))
                else:
                    c.update(id=batch, x=x, inp=x, feats=self.backbone(x).float(), merge=None)
        return c

    def __call__(self, head):
        from models.semseg_plm import merge_output_patches_to_image

        def predict(batch):
            c = self.features(batch)
            out = fusion_head_logits(head, *c['feats']) if isinstance(c['feats'], tuple) else head(c['feats'])
            logits = pixel_logits(out, c['inp'], self.backbone.pad_to)
            if c['merge'] is not None:
                num_patches, stride, pad = c['merge']
                logits = merge_output_patches_to_image(logits, num_patches, self.window, stride,
                                                       c['x'].shape[0], c['x'].shape[-1], pad)
            return torch.softmax(logits, dim=1)
        return predict


def subset_dataset(ds, keep):
    """A shallow copy of a NeophyteDataset restricted to the rows where keep is True."""
    import copy

    keep = np.asarray(keep, dtype=bool)
    sub = copy.copy(ds)
    sub._filter_by(keep)
    if ds.sample_weights is not None:
        sub.sample_weights = np.asarray(ds.sample_weights, dtype=np.float64)[keep].tolist()
    return sub


def calibrate_head(backbone, name, overrides, meta, args):
    """Bias for the logit of the new species from the few-shot subset alone (section 6.4).

    The subset is split in two halves (its new-species tiles, and the replay and
    background tiles, each split in half). A head trained on one half (same epochs, so
    half the steps) scores the whole tiles of the other half, as on the test split, for
    every bias of CAL_DELTAS. The bias with the best F1 of the new species pooled over
    both halves is returned (ties: the smallest |bias|).
    """
    from pytorch_lightning import seed_everything
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from utils.transform_utils import get_transforms

    seed_everything(meta['seed'], workers=True)
    d = compose_cfg(overrides).data
    dm = make_datamodule(d, args.num_workers)
    dm.setup('fit')
    ds = dm.train_dataset
    c = meta['new_class']
    is_new = (ds.stats_df[f"{meta['new_species']}_px"] > 0).to_numpy()
    rng = np.random.RandomState(1000 + int(meta['seed']))
    half = np.zeros(len(is_new), dtype=int)
    for group in (is_new, ~is_new):
        idx = np.flatnonzero(group)
        rng.shuffle(idx)
        half[idx[1::2]] = 1

    test_tf = get_transforms(d.test_transforms, d.transforms_additional_targets)
    class_names = [d.classes[i].name for i in sorted(d.classes.keys(), key=int)]
    scorer = Scorer(len(class_names), d.ignore_index, 'cuda', ap_classes=[], new_class=c, deltas=CAL_DELTAS)
    workers = dict(num_workers=args.num_workers)
    for h in (0, 1):
        train = subset_dataset(ds, half != h)
        if train.sample_weights is not None:
            sampling = dict(sampler=WeightedRandomSampler(train.sample_weights, len(train), replacement=True))
        else:
            sampling = dict(shuffle=True)
        loader = DataLoader(train, batch_size=d.batch_size_train, drop_last=False, **sampling, **workers)
        head = fit_head(backbone, loader, len(class_names), d.ignore_index, probe_steps(args, loader), args.lr,
                        f'{name} calibration half {h}')
        held = subset_dataset(ds, half == h)
        held.transform = test_tf
        predict = ProbePredictor(backbone, args.window)(head)
        with torch.no_grad():
            for i, batch in enumerate(DataLoader(held, batch_size=1, shuffle=False, **workers)):
                if args.max_test_batches is not None and i >= args.max_test_batches:
                    break
                target = batch['mask'].cuda(non_blocking=True).long()
                scorer.update(predict(batch), target, ['calibration'] * target.shape[0])
    sweep = scorer.result(class_names)['bias_sweep']
    f1 = [class_scores(np.asarray(conf))['F1'][c] for conf in sweep['conf']]
    f1 = [-1.0 if f is None or not np.isfinite(f) else f for f in f1]
    best = max(range(len(CAL_DELTAS)), key=lambda i: (round(f1[i], 4), -abs(CAL_DELTAS[i])))
    print(f"{name}: calibration bias {CAL_DELTAS[best]:+.1f} (held-out F1 {f1[best]:.3f}, "
          f"at 0: {f1[CAL_DELTAS.index(0.0)]:.3f})", flush=True)
    return {'calib_delta': CAL_DELTAS[best], 'calib_heldout_F1': f1, 'calib_deltas': CAL_DELTAS,
            'calib_n_tiles': [int((half == 0).sum()), int((half == 1).sum())]}


def find_saved_head(dirs, name):
    for d in dirs:
        path = os.path.join(d, f'{name}_head.pt')
        if os.path.exists(path):
            return path
    return None


def load_head(path, backbone, spec):
    """(head, info) of a saved head, or None if it was trained on another backbone."""
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if not (isinstance(saved, dict) and 'info' in saved and saved['info'].get('backbone') == spec):
        return None
    head = make_head(backbone.out_channels, saved['info']['num_classes'])
    head.load_state_dict(saved['state_dict'])
    return head.cuda().eval(), dict(saved['info'])


def run_probe(args):
    jobs = probe_jobs(args)
    print(f'{len(jobs)} heads on {args.backbone}: {[j[0] for j in jobs]}', flush=True)
    if args.calibrate and not args.new_species:
        raise ValueError('--calibrate needs --new_species')
    os.makedirs(args.out, exist_ok=True)
    backbone = build_backbone(args.backbone).cuda().eval().requires_grad_(False)
    method = args.method or os.path.basename(os.path.normpath(args.out))
    heads, extra = {}, {}
    for name, overrides, meta in jobs:
        # # --max_steps is a smoke test: never reuse or keep its heads as real ones
        path = None if args.max_steps else find_saved_head([args.out] + (args.heads_from or []), name)
        loaded = load_head(path, backbone, args.backbone) if path else None
        if loaded is not None:
            heads[name], extra[name] = loaded
            extra[name].update(method=method, head=path)
            print(f'{name}: head loaded from {path}', flush=True)
        elif args.heads_from:
            raise FileNotFoundError(f'no {args.backbone} head {name} in {[args.out] + args.heads_from}')
        else:
            heads[name], extra[name] = train_probe(backbone, name, overrides, meta, args)
            torch.save({'state_dict': heads[name].state_dict(), 'info': extra[name]},
                       os.path.join(args.out, f'{name}_head.pt'))
        if args.calibrate:
            cal_path = os.path.join(args.out, f'{name}_calib.json')
            if os.path.exists(cal_path) and not args.max_steps:
                with open(cal_path) as f:
                    cal = json.load(f)
            else:
                cal = calibrate_head(backbone, name, overrides, meta, args)
                if not args.max_steps:
                    write_json(cal_path, cal)
            extra[name].update(cal)

    test_data = args.test_data or f'neophytes_split_{args.fold}_test'
    loader, class_names, ignore_index = test_loader(test_data, args.num_workers, need_ndsm=False)
    probe = ProbePredictor(backbone, args.window)
    for e in extra.values():
        e['test_data'] = test_data
    # # the AP of the new species only: with 18 heads in one pass, the score histograms
    # # of all 7 classes would be most of what a head costs
    ap_classes = {name: [e['new_class']] for name, e in extra.items() if 'new_class' in e}
    sweeps = {name: (e['new_class'], CAL_DELTAS) for name, e in extra.items()
              if args.calibrate and 'new_class' in e}
    score({name: probe(h) for name, h in heads.items()}, loader, class_names, ignore_index,
          args.out, args.max_test_batches, extra, ap_classes, sweeps)


def run_combo(args):
    """Known species from a base U-Net, the new species from a linear head (section 6.2).

    The base was trained with the new species as background, so its channel of the new
    species is dropped. With q = the head's probability of the new species, the combined
    probabilities are q for the new species and (1 - q) x the base's renormalised
    probabilities for the others. One test pass for all heads (one per draw).
    """
    from models.semseg_plm import SegmentationModel

    if not args.base_ckpt or not args.new_species or len(args.new_species) != 1 or not args.heads_from:
        raise ValueError('combo needs --base_ckpt, --heads_from and one --new_species')
    base = SegmentationModel.load_from_checkpoint(args.base_ckpt, weights_only=False, map_location='cpu')
    base.set_patch_2_img_size(False)  # whole tiles, as test.py
    base = base.cuda().eval()
    backbone = build_backbone(args.backbone).cuda().eval().requires_grad_(False)
    test_data = args.test_data or f'neophytes_split_{args.fold}_test'
    sp = args.new_species[0]
    c = species_classes(compose_cfg([f'data={test_data}']).data.classes)[sp]
    method = args.method or os.path.basename(os.path.normpath(args.out))

    heads, extra = {}, {}
    for seed in args.seeds:
        name = f'{sp}_k{args.k[0]}_s{seed}'
        path = find_saved_head(args.heads_from, name)
        loaded = load_head(path, backbone, args.backbone) if path else None
        if loaded is None:
            raise FileNotFoundError(f'no {args.backbone} head {name} in {args.heads_from}')
        heads[name], info = loaded
        if info.get('new_class') != c:
            raise ValueError(f'{path}: new class {info.get("new_class")}, expected {c}')
        extra[name] = dict(info, method=method, head=path, base_ckpt=args.base_ckpt, test_data=test_data)

    loader, class_names, ignore_index = test_loader(test_data, args.num_workers,
                                                    need_ndsm='ndsm' in base.in_channels)
    probe = ProbePredictor(backbone, args.window)
    cache = {}

    def base_probs(batch):
        if cache.get('id') is not batch:
            inputs, aux = base.prepare_inputs(batch)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                p = base(inputs.cuda(non_blocking=True), aux.cuda() if aux is not None else None).float()
            p[:, c] = 0
            cache.update(id=batch, p=p / p.sum(1, keepdim=True).clamp_min(1e-12))
        return cache['p']

    def combined(head):
        predict = probe(head)

        def f(batch):
            q = predict(batch).float()[:, c:c + 1]
            out = base_probs(batch) * (1 - q)
            out[:, c:c + 1] = q
            return out
        return f

    score({name: combined(h) for name, h in heads.items()}, loader, class_names, ignore_index,
          args.out, args.max_test_batches, extra, {name: [c] for name in heads})


def run_eval(args):
    from models.semseg_plm import SegmentationModel

    names = args.names or [f'm{i}' for i in range(len(args.ckpts))]
    models = {}
    for name, path in zip(names, args.ckpts):
        model = SegmentationModel.load_from_checkpoint(path, weights_only=False, map_location='cpu')
        model.set_patch_2_img_size(False)  # whole tiles, as test.py
        models[name] = model.cuda().eval()
    need_ndsm = any('ndsm' in m.in_channels for m in models.values())
    test_data = args.test_data or f'neophytes_split_{args.fold}_test'
    loader, class_names, ignore_index = test_loader(test_data, args.num_workers, need_ndsm)

    def predictor(model):
        def predict(batch):
            inputs, aux = model.prepare_inputs(batch)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                return model(inputs.cuda(non_blocking=True), aux.cuda() if aux is not None else None)
        return predict

    # # metadata for the summary: which species was new for these checkpoints
    meta = {'method': args.method or os.path.basename(os.path.normpath(args.out)), 'fold': args.fold,
            'test_data': test_data}
    if args.new_species:
        species = species_classes(compose_cfg([f'data={test_data}']).data.classes)
        meta.update(new_species=args.new_species[0], new_class=species[args.new_species[0]],
                    k=args.k[0], draw=args.draw)
    extra = {}
    for name, path in zip(names, args.ckpts):
        seed = re.fullmatch(r's(\d+)', name)
        extra[name] = dict(meta, ckpt=path, **({'seed': int(seed.group(1))} if seed else {}))
    score({name: predictor(m) for name, m in models.items()}, loader, class_names, ignore_index,
          args.out, args.max_test_batches, extra)


def stats_dataset(d):
    """The training split of a data config with its stats table only (no image is read)."""
    from datasets.neophyte_dataset import NeophyteDataset
    return NeophyteDataset(d.image_dir.train, d.classes, dataset_stats=d.dataset_stats,
                           weighted_sampling=d.weighted_sampling, cap_quantile=d.cap_quantile,
                           lift_quantile=d.lift_quantile, bg_fraction=d.bg_fraction,
                           min_strat_images=d.min_strat_images, value_mapping=d.value_mapping,
                           ignore_index=d.ignore_index, few_shot_k=d.few_shot_k,
                           few_shot_mode=d.few_shot_mode, few_shot_bg_k=d.few_shot_bg_k,
                           few_shot_seed=d.few_shot_seed, max_nodata_frac=d.max_nodata_frac)


def run_check(args):
    """Stats-level check of the few-shot subsets (no images are read)."""
    if args.new_species:
        return run_check_grid(args)

    d = compose_cfg([f'data={args.holdout_data}']).data
    ds = stats_dataset(d)
    # # the species that value_mapping sends to background
    class_ids = {int(i) for i in d.classes.keys()}
    held_out = [d.classes[int(old)].stats_name for mapping in (d.value_mapping or {}).values() if mapping
                for old, new in mapping.items() if int(new) == 0 and int(old) != 0 and int(old) in class_ids]
    if not held_out:
        raise ValueError(f'{args.holdout_data} holds no species out (value_mapping)')
    w = np.asarray(ds.sample_weights)
    bg_share = w[ds.stats_df['only_background'].to_numpy() == 1].sum() / w.sum()
    left = {s: int(ds.stats_df[f'{s}_px'].sum()) for s in held_out}
    print(f'{args.holdout_data}: {len(ds)} tiles, held-out pixels left: {left}, '
          f'sampling share of background-only tiles: {bg_share:.3f}')
    assert not any(left.values()), 'held-out species still in the stats'
    if args.holdout_only:
        return

    draws = {}
    for seed in args.seeds:
        ds = stats_dataset(compose_cfg([f'data={args.train_data}', f'data.few_shot_seed={seed}']).data)
        new = ds.stats_df.index[ds.stats_df['B_davidii_px'] > 0]
        draws[seed] = set(new)
        w = np.asarray(ds.sample_weights)
        share = w[(ds.stats_df['B_davidii_px'] > 0).to_numpy()].sum() / w.sum()
        print(f'seed {seed}: {len(ds)} tiles, {len(new)} with B. davidii, sampling share {share:.3f}')
    seeds = list(draws)
    for i, a in enumerate(seeds):
        for b in seeds[i + 1:]:
            print(f'B. davidii tiles shared by seeds {a} and {b}: {len(draws[a] & draws[b])}')


def run_check_grid(args):
    """Per new species: the candidate tiles of the training split and what each draw gets."""
    from datasets.neophyte_dataset import select_few_shot_for_class

    jobs = probe_jobs(args)
    df = stats_dataset(compose_cfg([f'data=neophytes_split_{args.fold}_train_512']).data).stats_df
    mode = draw_mode(args.draw)
    for sp in dict.fromkeys(meta['new_species'] for _, _, meta in jobs):
        cand = df.index[(df[f'only_{sp}'] == 1) & (df[f'{sp}_px'] > 0)]
        flights = {}
        for p in cand:
            flights[tile_flight(p)] = flights.get(tile_flight(p), 0) + 1
        print(f'{args.fold} {sp}: {len(cand)} candidate tiles in {len(flights)} flights {dict(sorted(flights.items()))}')
        for _, _, meta in jobs:
            if meta['new_species'] != sp:
                continue
            tiles = select_few_shot_for_class(df, sp, meta['k'], mode, random_state=meta['seed'])
            note = '' if len(tiles) == meta['k'] else f"  <-- only {len(tiles)} of {meta['k']}"
            print(f"  k={meta['k']} seed {meta['seed']} ({args.draw}): {len(tiles)} tiles from "
                  f"{sorted({tile_flight(p) for p in tiles})}{note}")
    # # the composed subset of the first job, with the sampling share of its new species
    name, overrides, meta = jobs[0]
    ds = stats_dataset(compose_cfg(overrides).data)
    w = np.asarray(ds.sample_weights)
    share = w[(ds.stats_df[f"{meta['new_species']}_px"] > 0).to_numpy()].sum() / w.sum()
    print(f"{name}: {len(ds)} tiles, sampling share of {meta['new_species']} {share:.3f}")


def summary_rows(root, default_new_class=1):
    """One row per scored head / checkpoint under <root>/<method dir>/<name>.json.

    The October results carry no metadata: they are B. davidii (class 1), k = 10, cv3.
    """
    rows = []
    for path in sorted(glob.glob(os.path.join(root, '*', '*.json'))):
        with open(path) as f:
            r = json.load(f)
        if 'F1' not in r:
            continue
        c = r.get('new_class', default_new_class)
        known = [f for i, f in enumerate(r['F1']) if i not in (0, c)]
        # # --calibrate: F1 of the new species at the chosen bias and at the best one in hindsight
        cal = {}
        if 'bias_sweep' in r:
            f1 = [class_scores(np.asarray(conf))['F1'][c] for conf in r['bias_sweep']['conf']]
            deltas = r['bias_sweep']['deltas']
            cal = dict(F1_cal=f1[deltas.index(r['calib_delta'])] if 'calib_delta' in r else None,
                       calib_delta=r.get('calib_delta'), F1_best=float(np.nanmax(f1)),
                       best_delta=deltas[int(np.nanargmax(f1))])
        name = os.path.basename(path)[:-5]
        seed = re.search(r's(\d+)$', name)
        rows.append(dict(
            method=r.get('method', os.path.basename(os.path.dirname(path))), fold=r.get('fold', 'cv3'),
            draw=r.get('draw', 'random'), species=r['class_names'][c], new_class=c, k=r.get('k', 10),
            name=name, seed=r.get('seed', int(seed.group(1)) if seed else None),
            F1=r['F1'][c], recall=r['Recall'][c],
            precision=r['Precision'][c], AP=r['AP'][c], known_F1=float(np.nanmean(known)),
            n_new_tiles=r.get('n_new_tiles'), new_flights=' '.join(r.get('new_flights', [])), **cal))
    return rows


def run_summary(args):
    """The new-species scores per method, species and k: summary.csv (one row per head),
    summary.txt (means over the draws and the method x species table of mean F1)."""
    import pandas as pd

    rows = summary_rows(args.root, args.new_class)
    if not rows:
        print('no results under', args.root)
        return
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.root, 'summary.csv'), index=False)

    pd.set_option('display.width', 250, 'display.max_rows', 2000, 'display.max_columns', 50)
    keys = ['fold', 'draw', 'k', 'new_class', 'species', 'method']
    g = df.groupby(keys, sort=True)
    table = g.agg(n=('F1', 'size'), F1=('F1', 'mean'), F1_min=('F1', 'min'), F1_max=('F1', 'max'),
                  recall=('recall', 'mean'), precision=('precision', 'mean'), AP=('AP', 'mean'),
                  known_F1=('known_F1', 'mean')).reset_index().drop(columns='new_class')
    lines = ['New species: F1 at the argmax, AP threshold-free; known_F1 = mean F1 of the other species.',
             'Means over the draws (n), F1 also with its min and max.', '',
             table.to_string(index=False, float_format=lambda v: f'{v:.3f}')]
    if 'F1_cal' in df:
        cal = df[df['F1_cal'].notna()].groupby(keys, sort=True).agg(
            n=('F1', 'size'), F1=('F1', 'mean'), F1_cal=('F1_cal', 'mean'), calib_delta=('calib_delta', 'mean'),
            F1_best=('F1_best', 'mean'), best_delta=('best_delta', 'mean')).reset_index().drop(columns='new_class')
        lines += ['', 'Calibration (--calibrate): F1 at the argmax, at the bias chosen on the few-shot subset '
                  '(F1_cal) and at the best bias in hindsight (F1_best, an upper bound); biases are means',
                  cal.to_string(index=False, float_format=lambda v: f'{v:.3f}')]
    for (fold, draw), part in df.groupby(['fold', 'draw']):
        pivot = part.pivot_table(index=['k', 'method'], columns='species', values='F1', aggfunc='mean')
        order = part.drop_duplicates('species').sort_values('new_class')['species']
        pivot = pivot.reindex(columns=list(order))  # a species without any score stays as a column
        pivot['mean'] = pivot.mean(axis=1, skipna=False)
        lines += ['', f'mean F1 of the new species, {fold}, {draw} draws (mean: all listed species)',
                  pivot.to_string(float_format=lambda v: f'{v:.3f}', na_rep='-')]
    text = '\n'.join(lines)
    print(text)
    with open(os.path.join(args.root, 'summary.txt'), 'w') as f:
        f.write(text + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode', choices=['probe', 'eval', 'combo', 'check', 'summary'])
    p.add_argument('--backbone', help='probe/combo: pc24, dinov2, mit_b2, fusion, ckpt:<path> or vitckpt:<path>')
    p.add_argument('--heads_from', nargs='+', default=None,
                   help='probe/combo: directories with trained heads (<name>_head.pt) to reuse instead of training')
    p.add_argument('--calibrate', action='store_true',
                   help='probe: choose a bias of the new species on the few-shot subset, sweep the biases on the test split')
    p.add_argument('--base_ckpt', default=None, help='combo: base U-Net trained with the new species as background')
    p.add_argument('--ckpts', nargs='+', help='eval: checkpoints to score')
    p.add_argument('--names', nargs='+', help='eval: output names, one per checkpoint (s<seed>)')
    p.add_argument('--fold', default='cv3', help='cross-validation fold of the training and test split')
    p.add_argument('--new_species', nargs='+', default=None,
                   help='probe/check: stats_names of the new species (e.g. B_davidii) or "all", one '
                        'group of heads each; eval: the species that was new for the checkpoints')
    p.add_argument('--k', nargs='+', type=int, default=[10], help='labelled tiles of the new species')
    p.add_argument('--replay_k', type=int, default=50, help='tiles of every other species')
    p.add_argument('--bg_k', type=int, default=200, help='background-only tiles')
    p.add_argument('--draw', default='random',
                   help='random: the k tiles come from all training sites; flight: from one flight per seed; '
                        'flights<n>: from n flights, about k/n each')
    p.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3], help='few-shot draws')
    p.add_argument('--method', default=None, help='label in the summary (default: name of --out)')
    p.add_argument('--train_data', default=TRAIN_DATA, help='probe without --new_species: data config')
    p.add_argument('--holdout_data', default=HOLDOUT_DATA, help='check: base-training config')
    p.add_argument('--holdout_only', action='store_true', help='check: only the base-training config')
    p.add_argument('--test_data', default=None, help='default: neophytes_split_<fold>_test')
    p.add_argument('--epochs', type=int, default=30, help='as many steps as a 30-epoch fine-tuning')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--window', type=int, default=512, help='sliding window of the ViT probes')
    p.add_argument('--num_workers', type=int, default=16)
    p.add_argument('--max_steps', type=int, default=None, help='smoke test: cap the training steps')
    p.add_argument('--max_test_batches', type=int, default=None, help='smoke test: cap the test pass')
    p.add_argument('--out', help='output directory')
    p.add_argument('--root', default='results/fewshot/loso', help='summary: results root')
    p.add_argument('--new_class', type=int, default=1,
                   help='summary: class index of the new species for results without metadata')
    args = p.parse_args()
    draw_mode(args.draw)  # fail early on an unknown --draw

    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.set_float32_matmul_precision('medium')
    {'probe': run_probe, 'eval': run_eval, 'combo': run_combo, 'check': run_check, 'summary': run_summary}[args.mode](args)


if __name__ == '__main__':
    main()
