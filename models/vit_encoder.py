"""Plain ViT encoders (DINOv2, PlantCLEF 2024) for the SMP U-Net decoder.

A ViT has a single stride-14 (or 16) token map, while the U-Net decoder expects the
stride-2/4/8/16/32 features of a hierarchical encoder. ``ViTPyramidEncoder`` bridges
the two with a ViTDet-style simple feature pyramid (Li et al. 2022): the outputs of
four evenly spaced blocks are upsampled with transposed convolutions (strides 4, 8),
kept (16) or max-pooled (32), and resampled onto the exact grid of a CNN encoder.

Its output channels equal those of MiT-b2 (64, 128, 320, 512), so the decoder built
on top has exactly the shape of the U-Net + MiT-b2 baseline and only the encoder
differs in a comparison (RQ1). The stride-2 stage is empty, as for MiT encoders.

Any input size works: the input is zero-padded (the mean colour after normalisation)
to a multiple of the patch size, and the position embedding is interpolated
(``dynamic_img_size``), so validation and test run on whole 2048 px tiles like the
baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.layers import LayerNorm2d
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.base import initialization as init
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder


def conv_norm_act(in_ch, out_ch, kernel_size):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False),
                         LayerNorm2d(out_ch), nn.GELU())


class ViTPyramidEncoder(nn.Module):
    """timm ViT + simple feature pyramid with the SMP encoder interface.

    Args:
        timm_name: timm model, e.g. ``vit_base_patch14_reg4_dinov2.lvd142m``.
        weights_path: backbone state dict (no classifier head) to load instead of the
            timm pretrained weights, e.g. the PlantCLEF 2024 ViTD2PC24All backbone.
        block_indices: blocks whose outputs feed strides 4, 8, 16 and 32.
        pyramid_channels: channels of the four pyramid levels.
        grad_checkpointing: recompute the ViT blocks in the backward pass (less memory).
    """

    def __init__(self, timm_name, weights_path=None, block_indices=(2, 5, 8, 11),
                 pyramid_channels=(64, 128, 320, 512), grad_checkpointing=False):
        super().__init__()
        self.vit = timm.create_model(timm_name, pretrained=weights_path is None, num_classes=0,
                                     dynamic_img_size=True)
        if weights_path:
            self.vit.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
        self.vit.set_grad_checkpointing(grad_checkpointing)

        self.block_indices = list(block_indices)
        self.patch_size = self.vit.patch_embed.patch_size[0]
        dim = self.vit.embed_dim
        # # token map (stride p) -> strides p/4, p/2, p, 2p; resampled to 4, 8, 16, 32 below
        self.necks = nn.ModuleList([
            nn.Sequential(nn.ConvTranspose2d(dim, dim // 2, 2, stride=2), LayerNorm2d(dim // 2), nn.GELU(),
                          nn.ConvTranspose2d(dim // 2, dim // 4, 2, stride=2)),
            nn.ConvTranspose2d(dim, dim // 2, 2, stride=2),
            nn.Identity(),
            nn.MaxPool2d(2, ceil_mode=True),
        ])
        neck_channels = (dim // 4, dim // 2, dim, dim)
        self.projections = nn.ModuleList([
            nn.Sequential(conv_norm_act(c_in, c_out, 1), conv_norm_act(c_out, c_out, 3))
            for c_in, c_out in zip(neck_channels, pyramid_channels)
        ])
        self.out_channels = (3, 0, *pyramid_channels)

    def forward(self, x):
        batch_size, _, height, width = x.shape
        p = self.patch_size
        padded = F.pad(x, (0, -width % p, 0, -height % p))
        tokens = self.vit.forward_intermediates(padded, indices=self.block_indices, norm=True,
                                                output_fmt='NCHW', intermediates_only=True)

        features = [x, x.new_empty(batch_size, 0, height // 2, width // 2)]
        for feat, neck, proj, stride in zip(tokens, self.necks, self.projections, (4, 8, 16, 32)):
            feat = neck(feat)
            # # neck output has stride p/16 * stride px; rescaling by p/16 lands on the
            # # stride grid of the (top-left aligned) padded input, the crop removes the pad
            if p != 16:
                feat = F.interpolate(feat, scale_factor=p / 16, mode='bilinear', align_corners=False,
                                     recompute_scale_factor=False)
            features.append(proj(feat[:, :, :height // stride, :width // stride]))
        return features


class ViTUnet(nn.Module):
    """U-Net decoder and head of ``smp.Unet`` on a :class:`ViTPyramidEncoder`.

    Same ``encoder`` / ``decoder`` / ``segmentation_head`` attributes as an SMP model,
    so ``SegmentationModel`` (including ``fusion_mode='stem'``) uses it unchanged.
    """

    def __init__(self, timm_name, classes, weights_path=None, grad_checkpointing=False,
                 decoder_channels=(256, 128, 64, 32, 16)):
        super().__init__()
        self.encoder = ViTPyramidEncoder(timm_name, weights_path=weights_path,
                                         grad_checkpointing=grad_checkpointing)
        self.decoder = UnetDecoder(encoder_channels=self.encoder.out_channels,
                                   decoder_channels=decoder_channels, n_blocks=len(decoder_channels))
        self.segmentation_head = SegmentationHead(in_channels=decoder_channels[-1], out_channels=classes,
                                                  kernel_size=3)
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)

    def forward(self, x):
        return self.segmentation_head(self.decoder(self.encoder(x)))


def layer_decay_param_groups(module, vit, lr, layer_decay):
    """Optimizer parameter groups of ``module`` with layer-wise LR decay for ``vit``.

    ViT block i (0-based) gets ``lr * layer_decay ** (num_blocks - i)``, the patch and
    position embeddings one step less than block 0. The final norm, which every pyramid
    level reads, and everything outside the ViT (pyramid, decoder, head, elevation
    fusion) train at ``lr``. The early blocks thus move least, which keeps the
    pretrained features (see Kumar et al. 2022 on why that matters out of distribution).
    """
    num_blocks = len(vit.blocks)
    vit_param_ids = {id(p) for p in vit.parameters()}

    def layer_id(name):
        if name.startswith('blocks.'):
            return int(name.split('.')[1]) + 1
        if name.startswith(('cls_token', 'reg_token', 'pos_embed', 'patch_embed')):
            return 0
        return num_blocks + 1  # norm, fc_norm

    groups = {}
    for name, param in vit.named_parameters():
        lid = layer_id(name)
        groups.setdefault(lid, []).append(param)
    param_groups = [{'params': params, 'lr': lr * layer_decay ** (num_blocks + 1 - lid)}
                    for lid, params in sorted(groups.items())]
    param_groups.append({'params': [p for p in module.parameters() if id(p) not in vit_param_ids], 'lr': lr})
    return param_groups
