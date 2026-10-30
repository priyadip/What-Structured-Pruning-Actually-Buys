"""A convolution factored through a low-dimensional patch code.

        conv_V : R^(C_in*k*k) -> R^(C_out)          V u + c
        becomes
        conv_P : R^(C_in*k*k) -> R^r                P u          (k x k, no bias)
        conv_B : R^r          -> R^(C_out)          B z + c      (1 x 1)

This is the CNN form of the MLP study's input reduction. There, reducing the
input meant factoring the first layer as `W_1 ~ A P` with `P` an `r x D`
projection of the raw pixels. A conv layer never reads the whole image -- it
reads a `C_in*k*k` **patch** -- so the object with a reducible dimension is the
patch space, and every conv layer has one, not just the first.

Two consequences worth stating plainly:

* The first conv layer's patch space is the raw **RGB patch manifold**. Its GRIDE
  dimension is what bounds `r` there, and it is a different (much smaller) number
  than the image-manifold dimension the MLP study measured.
* Parameters and multiply-accumulates fall by the *same* factor here, because a
  conv reuses its weights at every spatial position. That is the opposite of
  channel compression, whose savings sit almost entirely in the deep,
  low-resolution layers -- so the two mechanisms suit different cost metrics.
  They do **not**, however, stack: at a fixed accuracy tolerance, chaining them
  beat the better single stage in only 3 of 9 cases (see `COMBINED_RESULTS.md`).
  Channel compression lowers `C_out`, which lowers the break-even rank
  `F*C_out/(F+C_out)`, and both stages draw on the same tolerance budget.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FactoredConv(nn.Module):
    """`conv_B(conv_P(x))`, exposing the geometry of the conv it replaces."""

    def __init__(self, src: nn.Conv2d, P: torch.Tensor, B: torch.Tensor,
                 bias: torch.Tensor):
        super().__init__()
        r = P.shape[0]
        self.proj = nn.Conv2d(src.in_channels, r, src.kernel_size, src.stride,
                              src.padding, src.dilation, src.groups, bias=False)
        self.mix = nn.Conv2d(r, src.out_channels, 1, bias=True)
        with torch.no_grad():
            self.proj.weight.copy_(P.reshape(r, src.in_channels, *src.kernel_size))
            self.mix.weight.copy_(B.reshape(src.out_channels, r, 1, 1))
            self.mix.bias.copy_(bias)
        self.in_channels, self.out_channels, self.rank = src.in_channels, src.out_channels, r
        self.kernel_size, self.stride = src.kernel_size, src.stride
        self.padding, self.dilation, self.groups = src.padding, src.dilation, src.groups

    def forward(self, x):
        return self.mix(self.proj(x))


class FactoredLinear(nn.Module):
    """The same factorisation for a classifier head: `W ~ B P`."""

    def __init__(self, src: nn.Linear, P: torch.Tensor, B: torch.Tensor,
                 bias: torch.Tensor):
        super().__init__()
        r = P.shape[0]
        self.proj = nn.Linear(src.in_features, r, bias=False)
        self.mix = nn.Linear(r, src.out_features, bias=True)
        with torch.no_grad():
            self.proj.weight.copy_(P)
            self.mix.weight.copy_(B)
            self.mix.bias.copy_(bias)
        self.in_features, self.out_features, self.rank = src.in_features, src.out_features, r

    def forward(self, x):
        return self.mix(self.proj(x))


def factor_module(src: nn.Module, P: torch.Tensor, B: torch.Tensor,
                  bias: torch.Tensor) -> nn.Module:
    if isinstance(src, nn.Conv2d):
        return FactoredConv(src, P, B, bias).to(P.device).eval()
    return FactoredLinear(src, P, B, bias).to(P.device).eval()


def factored_from_specs(proj: dict, mix: dict) -> nn.Module:
    """Build the factored module from its two recorded children alone.

    Deriving it from the *dense* skeleton is wrong whenever the network was
    channel-compressed first: the parent conv then has a different `in_channels`
    than the artefact does, and the `strict=True` load fails on a shape mismatch.
    The two child specs already carry every dimension needed, so nothing about
    the surrounding model has to be consulted.
    """
    if proj["type"] == "Conv2d":
        kh, kw = proj["kernel"]
        src = nn.Conv2d(proj["in"], mix["out"], (kh, kw), tuple(proj["stride"]),
                        tuple(proj["padding"]), tuple(proj["dilation"]),
                        proj["groups"], bias=True)
        return FactoredConv(src, torch.zeros(proj["out"], proj["in"] * kh * kw),
                            torch.zeros(mix["out"], proj["out"]),
                            torch.zeros(mix["out"]))
    src = nn.Linear(proj["in"], mix["out"], bias=True)
    return FactoredLinear(src, torch.zeros(proj["out"], proj["in"]),
                          torch.zeros(mix["out"], proj["out"]),
                          torch.zeros(mix["out"]))


def empty_factored(src: nn.Module, r: int) -> nn.Module:
    """A factored module of the right shapes, to be filled by `load_state_dict`.

    Rebuilding from disk must not depend on the values, only the geometry: the
    weights arrive from the safetensors file and a `strict=True` load is what
    proves the geometry was recorded correctly.
    """
    if isinstance(src, nn.Conv2d):
        F = src.in_channels * src.kernel_size[0] * src.kernel_size[1]
        return FactoredConv(src, torch.zeros(r, F), torch.zeros(src.out_channels, r),
                            torch.zeros(src.out_channels))
    return FactoredLinear(src, torch.zeros(r, src.in_features),
                          torch.zeros(src.out_features, r),
                          torch.zeros(src.out_features))


def break_even_rank(src: nn.Module, gs: int) -> float:
    """Rank above which the factorisation costs *more* than the dense layer.

    `C_in*k*k * r + r * C_out` vs `C_in*k*k * C_out`, i.e. `r* = F*C_out/(F+C_out)`
    with `F = C_in*k*k`. Reported alongside every result: a rank above this is a
    legitimate answer ("this layer's patch space is not low-rank") but it is not
    a compression, and quoting it as one would be dishonest.
    """
    if isinstance(src, nn.Conv2d):
        F, out = src.in_channels * gs, src.out_channels
    else:
        F, out = src.in_features, src.out_features
    return F * out / (F + out)
