"""Fold every BatchNorm2d into the convolution that produces it.

Why this is the first thing that happens
----------------------------------------
In the MLP study a "neuron" was exactly a pair `(w, b)` only *after* BatchNorm1d
was folded in; before folding, deleting a row of `W` leaves a dangling row of BN
statistics and the arithmetic stops being a clean linear map. The same is true
here, with one extra wrinkle: a conv's BN is per-*output-channel*, so folding
turns "select `m` output channels" into "select `m` rows of a weight tensor",
which is the operation every strategy in this project is written against.

        w <- w * gamma / sqrt(var + eps)          (broadcast over C_in,k,k)
        b <- (b - mu) * gamma / sqrt(var + eps) + beta

The BN module is then replaced by `nn.Identity()` in place, so the reference
`forward()` of `models/vgg.py` and `models/resnet.py` is never touched -- the
folded network is the same object graph running the same code path.

For ResNet this is valid at every BN in the architecture, including `bn2` /
`bn3` whose output is added to a shortcut: BN is a per-channel affine map
applied *before* the addition, so folding it into the preceding conv leaves the
addition unchanged. The shortcut's own BN folds into the shortcut conv.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


@torch.no_grad()
def _fold_into(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> None:
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    b = conv.bias if conv.bias is not None else torch.zeros(
        conv.out_channels, device=conv.weight.device, dtype=conv.weight.dtype)
    new_b = (b - bn.running_mean) * s + bn.bias
    conv.weight.copy_(conv.weight * s.view(-1, 1, 1, 1))
    if conv.bias is None:
        conv.bias = nn.Parameter(new_b)
    else:
        conv.bias.copy_(new_b)


@torch.no_grad()
def fold_vgg(model: nn.Module) -> nn.Module:
    """`features` is a flat Sequential; BN always sits at `i+1` after a Conv."""
    m = copy.deepcopy(model)
    feats = m.features
    for i, mod in enumerate(feats):
        if isinstance(mod, nn.Conv2d) and i + 1 < len(feats) and \
                isinstance(feats[i + 1], nn.BatchNorm2d):
            _fold_into(mod, feats[i + 1])
            feats[i + 1] = nn.Identity()
    return m.eval()


@torch.no_grad()
def fold_resnet(model: nn.Module) -> nn.Module:
    m = copy.deepcopy(model)
    if isinstance(m.bn1, nn.BatchNorm2d):
        _fold_into(m.conv1, m.bn1)
        m.bn1 = nn.Identity()
    for stage in (m.layer1, m.layer2, m.layer3, m.layer4):
        for blk in stage:
            for ci, bi in (("conv1", "bn1"), ("conv2", "bn2"), ("conv3", "bn3")):
                if hasattr(blk, ci) and isinstance(getattr(blk, bi, None), nn.BatchNorm2d):
                    _fold_into(getattr(blk, ci), getattr(blk, bi))
                    setattr(blk, bi, nn.Identity())
            sc = blk.shortcut
            if len(sc) == 2 and isinstance(sc[0], nn.Conv2d) and isinstance(sc[1], nn.BatchNorm2d):
                _fold_into(sc[0], sc[1])
                sc[1] = nn.Identity()
    return m.eval()


def fold_bn(model: nn.Module, family: str) -> nn.Module:
    return fold_vgg(model) if family == "vgg" else fold_resnet(model)
