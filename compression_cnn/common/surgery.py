"""The compressible units of a VGG / ResNet, and the lock-step walk that
compresses them.

What a "unit" is
----------------
A unit is one **producer** convolution whose output channels are being cut, plus
the single **consumer** that reads them. That is the exact CNN analogue of the
MLP study's (hidden layer l, layer l+1) pair, and the reason the same closed
forms apply.

For a VGG every conv is a unit: `features` is a chain, so conv `i`'s only
consumer is conv `i+1` (or the classifier).

For a ResNet only the **internal** convs of a block are units:

    BasicBlock   conv1 -> conv2          conv1's output is internal
                 conv2 -> residual add   conv2's output is *on the stream*
    Bottleneck   conv1 -> conv2, conv2 -> conv3    both internal

A channel on the residual stream cannot be cut in isolation: the stream is a sum
of every block's output plus an identity path, so its width is shared by the
whole stage, and no single consumer can be refitted to compensate because the
error is a sum over producers that appears in no individual target. That is what
skip connections change about this problem.

Restricting to internal channels costs almost nothing. A BasicBlock's two convs
cost `C_in*p*k^2 + p*C_out*k^2`, both linear in the internal width `p`, so
compressing `p` scales the entire block; only the 1x1 shortcut convs and the stem
are out of reach. On a ResNet18 that is ~172k of 11.2M parameters -- **98% is
reachable** without ever touching a stream. Stage-level stream surgery was
therefore not built: it would add complexity for at most a two-percent ceiling,
and `../data_reduction_and_fit` attacks the same weights from the other side.

The walk
--------
Dense and compressed networks are advanced **in lock-step** through the same
architecture, holding only the current layer's activations for each. The design
matrix comes from the compressed path (so each fit sees the error inherited from
upstream) while the target comes from the dense path (so the error is corrected
rather than compounded). This is `Bench.Y_orig` from the MLP study, computed on
the fly instead of stored -- which matters here, where storing every layer's
activations for a ResNet34 would not fit.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import linalg as LA
from . import patches as PT


# ---------------------------------------------------------------------------
# unit context -- everything a strategy may look at
# ---------------------------------------------------------------------------

@dataclass
class UnitCtx:
    name: str                       # dotted path of the producer in the model tree
    cons_path: str                  # dotted path of the consumer
    prod: nn.Conv2d                 # producer on the compressed path
    consumer: nn.Module             # consumer on the compressed path (conv or linear)
    h: torch.Tensor                 # producer input, compressed path  [N, Cin, H, W]
    A: torch.Tensor                 # consumer input, compressed path  [N, C, H', W']
    Y: torch.Tensor                 # consumer pre-activation, DENSE path
    feed: Callable                  # producer output -> consumer input (relu, pools)
    rows: torch.Tensor
    per_image: int
    gs: int
    ridge: float = 1e-4
    seed: int = 0
    metric: torch.Tensor | None = None   # logit-pullback metric on the consumer's output
    _stats: tuple | None = field(default=None, repr=False)
    _U: torch.Tensor | None = field(default=None, repr=False)

    @property
    def C(self) -> int:
        return int(self.A.shape[1])

    @property
    def U(self) -> torch.Tensor:
        if self._U is None:
            self._U = PT.design_matrix(self.A, self.consumer, self.rows, self.per_image)
        return self._U

    @property
    def n_unknowns(self) -> int:
        return self.C * self.gs + 1

    @property
    def overdetermination(self) -> float:
        """Rows per unknown in the *widest* system this unit can pose.

        Deep conv layers are the dangerous case and they are dangerous in the
        opposite direction to what intuition suggests: they have many channels
        but a tiny spatial extent, so each calibration image contributes only
        `H_out*W_out` rows -- four, or one at the classifier. A VGG11's last
        512-channel unit needs `512*9 = 4608` unknowns from `4*N` rows, so a
        1024-image calibration set leaves the fit *underdetermined* and the
        refit silently returns nonsense that still evaluates to a plausible
        accuracy. `require_overdetermined` is what stops that.
        """
        return self.rows.numel() / self.n_unknowns

    def require_overdetermined(self, factor: float = 2.0) -> None:
        if self.overdetermination < factor:
            need = int(np.ceil(factor * self.n_unknowns / self.per_image))
            raise RuntimeError(
                f"unit {self.name}: {self.rows.numel()} patch rows for "
                f"{self.n_unknowns} unknowns ({self.overdetermination:.2f}x). "
                f"Need >= {need} calibration images (--calib) or a larger "
                f"--patch-budget.")

    @property
    def stats(self):
        """`(G, P, us, ys, n)` over the FULL channel set -- computed once."""
        if self._stats is None:
            Yr = PT.target_matrix(self.Y, self.rows, self.per_image)
            self._stats = LA.gram_cross(self.U, Yr)
        return self._stats

    def cols(self, groups) -> torch.Tensor:
        return LA.group_cols(groups, self.gs, self.A.device)

    def ls(self, groups):
        G, P, us, ys, n = self.stats
        return LA.ls_consuming(G, P, us, ys, n, self.cols(groups), self.ridge)

    def filters_flat(self) -> torch.Tensor:
        """Producer filters as `[C_out, C_in*k*k]` -- the effective neuron cloud
        before projection."""
        return self.prod.weight.detach().reshape(self.prod.weight.shape[0], -1)

    @torch.no_grad()
    def producer_patches(self, budget: int = 20000):
        """`(U_in, G_in)` for the **producer's own** input.

        The effective neuron cloud is the filters projected onto the column space
        of what *this* layer reads -- `unfold(h)` under the producer's kernel --
        not what the consumer reads. The two differ whenever a pool sits between
        them, so they are sampled separately.
        """
        per = PT.n_positions(self.prod, self.h)
        gen = torch.Generator().manual_seed(self.seed + 7919)
        rows = PT.sample_rows(self.h.shape[0], per, budget, gen, self.h.device)
        U = PT.design_matrix(self.h, self.prod, rows, per)
        G = torch.zeros(U.shape[1], U.shape[1], dtype=torch.float64, device=U.device)
        for i in range(0, U.shape[0], 8192):
            u = U[i:i + 8192]
            G += (u.t() @ u).double()
        return U, G

    @torch.no_grad()
    def acts_for(self, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Consumer input produced by a *replacement* producer (merge strategies)."""
        z = F.conv2d(self.h, W, b, self.prod.stride, self.prod.padding,
                     self.prod.dilation, self.prod.groups)
        return self.feed(z)

    @torch.no_grad()
    def stats_for(self, A_new: torch.Tensor):
        U = PT.design_matrix(A_new, self.consumer, self.rows, self.per_image)
        Yr = PT.target_matrix(self.Y, self.rows, self.per_image)
        return LA.gram_cross(U, Yr)


# ---------------------------------------------------------------------------
# installing a compressed unit
# ---------------------------------------------------------------------------

@torch.no_grad()
def new_producer(old: nn.Conv2d, W: torch.Tensor, b: torch.Tensor) -> nn.Conv2d:
    c = nn.Conv2d(old.in_channels, W.shape[0], old.kernel_size, old.stride,
                  old.padding, old.dilation, old.groups, bias=True)
    c.weight.copy_(W)
    c.bias.copy_(b)
    return c.to(W.device).eval()


@torch.no_grad()
def new_consumer(old: nn.Module, V_flat: torch.Tensor, c: torch.Tensor, m: int) -> nn.Module:
    """`V_flat` is `[out, m*gs]` in the design matrix's column order."""
    if isinstance(old, nn.Conv2d):
        kh, kw = old.kernel_size
        mod = nn.Conv2d(m, old.out_channels, old.kernel_size, old.stride, old.padding,
                        old.dilation, old.groups, bias=True)
        mod.weight.copy_(V_flat.reshape(old.out_channels, m, kh, kw))
    else:
        mod = nn.Linear(V_flat.shape[1], old.out_features, bias=True)
        mod.weight.copy_(V_flat)
    mod.bias.copy_(c)
    return mod.to(V_flat.device).eval()


def set_module(root: nn.Module, path: str, mod: nn.Module) -> None:
    parts = path.split(".")
    obj = root
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    last = parts[-1]
    if last.isdigit():
        obj[int(last)] = mod
    else:
        setattr(obj, last, mod)


def get_module(root: nn.Module, path: str) -> nn.Module:
    obj = root
    for p in path.split("."):
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    return obj


# ---------------------------------------------------------------------------
# unit inventories
# ---------------------------------------------------------------------------

@dataclass
class UnitSpec:
    name: str
    producer_path: str
    consumer_path: str
    feed_paths: list[str]           # modules applied between producer and consumer
    out_channels: int


def vgg_units(model: nn.Module) -> list[UnitSpec]:
    feats = model.features
    conv_at = [i for i, m in enumerate(feats) if isinstance(m, nn.Conv2d)]
    specs = []
    for u, ci in enumerate(conv_at):
        end = conv_at[u + 1] if u + 1 < len(conv_at) else len(feats)
        specs.append(UnitSpec(
            name=f"features.{ci}",
            producer_path=f"features.{ci}",
            consumer_path=(f"features.{conv_at[u + 1]}" if u + 1 < len(conv_at)
                           else "classifier"),
            feed_paths=[f"features.{j}" for j in range(ci + 1, end)],
            out_channels=feats[ci].out_channels))
    return specs


def resnet_units(model: nn.Module) -> list[UnitSpec]:
    specs = []
    for si, sname in enumerate(("layer1", "layer2", "layer3", "layer4")):
        stage = getattr(model, sname)
        for bi, blk in enumerate(stage):
            base = f"{sname}.{bi}"
            pairs = [("conv1", "conv2")]
            if hasattr(blk, "conv3"):
                pairs.append(("conv2", "conv3"))
            for pi, (pc, cc) in enumerate(pairs):
                specs.append(UnitSpec(
                    name=f"{base}.{pc}",
                    producer_path=f"{base}.{pc}",
                    consumer_path=f"{base}.{cc}",
                    feed_paths=[],                       # ReLU is applied by the walker
                    out_channels=getattr(blk, pc).out_channels))
    return specs


def units_for(model: nn.Module, family: str) -> list[UnitSpec]:
    return vgg_units(model) if family == "vgg" else resnet_units(model)


# ---------------------------------------------------------------------------
# the lock-step walks
# ---------------------------------------------------------------------------

def _chunked(fn, X, chunk=256):
    """Apply `fn` in image chunks, writing into one preallocated output.

    `torch.cat([...])` would hold every chunk *and* the concatenation at once --
    a transient 2x on tensors that are already gigabytes at 32x32 with 64
    channels, which is what pushed a ResNet walk out of memory.
    """
    first = fn(X[:min(chunk, X.shape[0])])
    out = torch.empty((X.shape[0],) + first.shape[1:], dtype=first.dtype, device=first.device)
    out[:first.shape[0]] = first
    del first
    for i in range(chunk, X.shape[0], chunk):
        out[i:i + chunk] = fn(X[i:i + chunk])
    return out


def _pipe(ops):
    def f(z):
        for o in ops:
            z = o(z)
        return z
    return f


class Walker:
    """Advances dense and compressed nets together, yielding one `UnitCtx` each
    time a compressible unit is reached, and consuming the caller's decision."""

    def __init__(self, dense: nn.Module, comp: nn.Module, family: str, X: torch.Tensor,
                 patch_budget: int, ridge: float, seed: int, chunk: int = 256):
        self.d, self.c, self.family, self.X = dense, comp, family, X
        self.patch_budget, self.ridge, self.seed, self.chunk = patch_budget, ridge, seed, chunk
        self.gen = torch.Generator().manual_seed(seed)

    # -- helpers ---------------------------------------------------------
    def _ctx(self, name, cons_path, prod, consumer, h_c, A_c, Y_d, feed) -> UnitCtx:
        per_image = PT.n_positions(consumer, A_c)
        gs = PT.group_size(consumer, A_c)
        rows = PT.sample_rows(A_c.shape[0], per_image, self.patch_budget,
                              self.gen, A_c.device)
        return UnitCtx(name=name, cons_path=cons_path, prod=prod, consumer=consumer,
                       h=h_c, A=A_c, Y=Y_d, feed=feed, rows=rows, per_image=per_image,
                       gs=gs, ridge=self.ridge, seed=self.seed)

    # -- VGG -------------------------------------------------------------
    def walk_vgg(self, decide):
        """`decide(ctx)` must return the consumer input for the *compressed*
        path, i.e. the surviving channels of `ctx.A` (or the output of a
        replacement producer)."""
        h_d = h_c = self.X
        for sp in vgg_units(self.d):
            prod_d = get_module(self.d, sp.producer_path)
            prod_c = get_module(self.c, sp.producer_path)
            cons_d = get_module(self.d, sp.consumer_path)
            cons_c = get_module(self.c, sp.consumer_path)
            fd = _pipe([get_module(self.d, p) for p in sp.feed_paths])
            fc = _pipe([get_module(self.c, p) for p in sp.feed_paths])

            A_d = _chunked(lambda t: fd(prod_d(t)), h_d, self.chunk)
            Y_d = (_chunked(cons_d, A_d, self.chunk) if isinstance(cons_d, nn.Conv2d)
                   else _chunked(lambda t: cons_d(t.reshape(t.shape[0], -1)), A_d, self.chunk))
            A_c = _chunked(lambda t: fc(prod_c(t)), h_c, self.chunk)

            ctx = self._ctx(sp.name, sp.consumer_path, prod_c, cons_c, h_c, A_c, Y_d, fc)
            h_c = decide(ctx)
            h_d = A_d
            del A_c, Y_d, ctx

    # -- ResNet ----------------------------------------------------------
    def walk_resnet(self, decide):
        rl = torch.relu
        s_d = _chunked(lambda t: rl(self.d.bn1(self.d.conv1(t))), self.X, self.chunk)
        s_c = _chunked(lambda t: rl(self.c.bn1(self.c.conv1(t))), self.X, self.chunk)

        for sname in ("layer1", "layer2", "layer3", "layer4"):
            for bi, (bd, bc) in enumerate(zip(getattr(self.d, sname), getattr(self.c, sname))):
                base = f"{sname}.{bi}"
                h1_d = _chunked(lambda t: rl(bd.conv1(t)), s_d, self.chunk)
                A1_c = _chunked(lambda t: rl(bc.conv1(t)), s_c, self.chunk)

                if hasattr(bd, "conv3"):                       # Bottleneck
                    Y1_d = _chunked(bd.conv2, h1_d, self.chunk)
                    ctx = self._ctx(f"{base}.conv1", f"{base}.conv2", bc.conv1, bc.conv2, s_c, A1_c, Y1_d, rl)
                    h1_c = decide(ctx)
                    del A1_c, Y1_d, ctx

                    h2_d = _chunked(lambda t: rl(bd.conv2(t)), h1_d, self.chunk)
                    A2_c = _chunked(lambda t: rl(bc.conv2(t)), h1_c, self.chunk)
                    Y2_d = _chunked(bd.conv3, h2_d, self.chunk)
                    ctx = self._ctx(f"{base}.conv2", f"{base}.conv3", bc.conv2, bc.conv3, h1_c, A2_c, Y2_d, rl)
                    h2_c = decide(ctx)
                    del A2_c, ctx
                    z_d, z_c = Y2_d, _chunked(bc.conv3, h2_c, self.chunk)
                    del h1_d, h2_d, h1_c, h2_c
                else:                                          # BasicBlock
                    Y_d = _chunked(bd.conv2, h1_d, self.chunk)
                    ctx = self._ctx(f"{base}.conv1", f"{base}.conv2", bc.conv1, bc.conv2, s_c, A1_c, Y_d, rl)
                    h_c = decide(ctx)
                    del A1_c, ctx
                    z_d, z_c = Y_d, _chunked(bc.conv2, h_c, self.chunk)
                    del h1_d, h_c

                s_d = rl(z_d + _chunked(bd.shortcut, s_d, self.chunk))
                s_c = rl(z_c + _chunked(bc.shortcut, s_c, self.chunk))
                del z_d, z_c

    def walk(self, decide):
        (self.walk_vgg if self.family == "vgg" else self.walk_resnet)(decide)


# ---------------------------------------------------------------------------
# how many calibration images this architecture actually needs
# ---------------------------------------------------------------------------

@torch.no_grad()
def plan_calibration(model: nn.Module, family: str, X_small: torch.Tensor,
                     factor: float = 4.0, cap: int = 16384) -> tuple[int, list[dict]]:
    """Smallest calibration set that leaves *every* unit over-determined.

    Deep layers are the binding constraint and it is easy to get backwards: a
    512-channel unit needs `512*9+1` unknowns but its consumer sees a 4x4 or 2x2
    map, so each image contributes only 16 or 4 rows -- and the VGG classifier
    contributes exactly one. Reading the requirement off a shape-only dry run
    removes the guess, and it is the difference between a refit that is solving a
    least-squares problem and one that is inventing a null-space direction.
    """
    walker = Walker(model, model, family, X_small, patch_budget=10 ** 9,
                    ridge=1.0, seed=0, chunk=X_small.shape[0])
    info: list[dict] = []

    def probe(ctx):
        p = ctx.prod
        pk = p.kernel_size[0] * p.kernel_size[1]
        p_out = PT.n_positions(p, ctx.h)                     # producer output positions
        cons_out = (ctx.consumer.out_channels if isinstance(ctx.consumer, nn.Conv2d)
                    else ctx.consumer.out_features)
        info.append({
            "unit": ctx.name, "C": ctx.C, "gs": ctx.gs,
            "per_image": ctx.per_image, "unknowns": ctx.n_unknowns,
            "need": int(np.ceil(factor * ctx.n_unknowns / ctx.per_image)),
            # marginal cost of ONE channel of this unit: it appears once as an
            # output of the producer and once as an input of the consumer
            "params_per_ch": p.in_channels * pk + 1 + cons_out * ctx.gs,
            "macs_per_ch": p.in_channels * pk * p_out + cons_out * ctx.gs * ctx.per_image,
        })
        return ctx.A

    walker.walk(probe)
    need = max(i["need"] for i in info)
    return int(min(need, cap)), info


# ---------------------------------------------------------------------------
# parameter accounting
# ---------------------------------------------------------------------------

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def clone_eval(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model).eval()
