"""Extract the ViT backbone of the PlantCLEF 2024 ViTD2PC24All checkpoint.

model_best.pth.tar (Zenodo 10.5281/zenodo.10848263) holds the training state of the
timm train script: weights, EMA weights, optimizer and args. This keeps the EMA
weights (the ones the organisers report) without the 7,806-class head, as a plain
state dict that loads with ``weights_only=True`` (configs/model/model_neophytes_vit_pc24.yaml).

    python scripts/extract_pc24_backbone.py <.../onlyclassifier_then_all/model_best.pth.tar> <out.pth>
"""
import argparse
import sys

import timm
import torch

# # the checkpoint stores the training args as an argparse.Namespace; allow only that
torch.serialization.add_safe_globals([argparse.Namespace])

ckpt = torch.load(sys.argv[1], map_location='cpu', weights_only=True)
state_dict = {k: v for k, v in ckpt['state_dict_ema'].items() if not k.startswith('head.')}
timm.create_model(ckpt['arch'], pretrained=False, num_classes=0).load_state_dict(state_dict)  # strict check
torch.save(state_dict, sys.argv[2])
print(f"{ckpt['arch']}, epoch {ckpt['epoch']}, val top-1 {ckpt['metric']:.2f}: "
      f"{len(state_dict)} tensors -> {sys.argv[2]}")
