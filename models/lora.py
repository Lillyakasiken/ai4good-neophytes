"""Low-rank adapters (LoRA, Hu et al. 2022) for the Linear layers of a network.

Each wrapped layer computes ``W x + b + (alpha / r) * B A x``. ``W`` and ``b`` are
the pretrained weights and stay frozen; only the small matrices ``A`` (r x in) and
``B`` (out x r) are trained. ``B`` starts at zero, so the adapted network starts out
identical to the pretrained one.

``LoRALinear`` subclasses ``nn.Linear`` and keeps the parameter names ``weight`` and
``bias``, so a state dict of the plain network loads into the wrapped one
(``strict=False``; only the ``lora_`` keys are missing).
"""

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Linear):
    """A ``nn.Linear`` with a trainable low-rank update next to its frozen weight."""

    def __init__(self, in_features, out_features, rank, alpha, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # as nn.Linear and the LoRA paper

    @classmethod
    def from_linear(cls, layer, rank, alpha):
        new = cls(layer.in_features, layer.out_features, rank, alpha, bias=layer.bias is not None)
        new.weight = layer.weight
        if layer.bias is not None:
            new.bias = layer.bias
        return new.to(layer.weight.device)

    def forward(self, x):
        return super().forward(x) + (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling


def add_lora(module, rank, alpha):
    """Replace every ``nn.Linear`` inside ``module`` by a ``LoRALinear``. Returns the count."""
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and not isinstance(child, LoRALinear):
            setattr(module, name, LoRALinear.from_linear(child, rank, alpha))
            count += 1
        else:
            count += add_lora(child, rank, alpha)
    return count


def is_lora_param(name):
    return "lora_A" in name or "lora_B" in name
