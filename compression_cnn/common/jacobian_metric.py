"""The metric the frozen tail of the network induces on a layer's output.

Why a metric at all
-------------------
`F1_rrr` is provably optimal among rank-`r` factorisations for the objective
`E || V u - B P u ||^2`. That objective is a surrogate: it treats every
coordinate of the layer's output as equally important, and the remaining layers
do not. The right norm is the one pulled back from the logits by the network's
own Jacobian,

        Omega_l = E_x [ J_l(x)^T J_l(x) ],   J_l = d logits / d z_l ,

and minimising the error in that norm is strictly better aligned with accuracy.
In the MLP input-reduction study this was `N1_jacobian`, and it beat the
closed-form ceiling on five of eight models.

Why the CNN version is a recursion and not autograd
---------------------------------------------------
The hard constraint is zero backward passes. For a ReLU network the Jacobian is
a product of weight tensors and diagonal activation masks, so `Omega` obeys a
backward *recursion over forward quantities* -- masks collected in one forward
pass, weights read off the frozen model. Nothing is differentiated.

The approximation, stated plainly
---------------------------------
A conv's Jacobian couples spatial positions, so the exact `Omega` is indexed by
`(channel, position)` pairs and is far too large. Errors from a rank-`r` patch
factorisation are approximately **spatially white**, so this module carries the
per-position channel metric only:

    conv, stride s   Omega_in = (1/s^2) sum_taps V_t^T Omega_out V_t
    ReLU             Omega <- Omega (*) E[m m^T]        m = 1[pre-activation > 0]
    2x2 max pool     Omega <- Omega / 4                 (one input in four is the max)
    k x k avg pool   Omega <- Omega / k^4               (each input carries 1/k^2)
    residual add     Omega_in = Omega_identity + Omega_branch   (cross terms dropped)

The last two lines are the genuinely lossy steps. `validate.py` checks the
result against a forward-only finite-difference probe rather than asserting it.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .surgery import _chunked


@torch.no_grad()
def _mask_moment(z: torch.Tensor, cap: int = 200000) -> torch.Tensor:
    """`E_{n,y,x}[ m m^T ]` for `m = 1[z > 0]`, over a sample of positions."""
    if z.dim() == 4:
        m = (z > 0).permute(0, 2, 3, 1).reshape(-1, z.shape[1])
    else:
        m = (z > 0)
    if m.shape[0] > cap:
        idx = torch.randperm(m.shape[0], device=m.device)[:cap]
        m = m.index_select(0, idx)
    md = m.double()
    return md.t() @ md / md.shape[0]


@torch.no_grad()
def _through_conv(Om: torch.Tensor, conv: nn.Conv2d) -> torch.Tensor:
    """`Omega_in = (1/s^2) sum_{taps} V_t^T Omega_out V_t`."""
    V = conv.weight.detach().double()                    # [out, in, kh, kw]
    o, i, kh, kw = V.shape
    Vt = V.permute(2, 3, 0, 1).reshape(kh * kw, o, i)
    Om_in = torch.einsum("toi,op,tpj->ij", Vt, Om, Vt)
    s = conv.stride[0] * conv.stride[1]
    return Om_in / s


@torch.no_grad()
def _sym(Om: torch.Tensor) -> torch.Tensor:
    return 0.5 * (Om + Om.t())


@torch.no_grad()
def metric_pair(Om: torch.Tensor, damp: float = 1e-3):
    """`(L, L^-1)` with `L L = Omega`, damped so flat directions stay invertible."""
    lam, Q = torch.linalg.eigh(_sym(Om))
    lam = lam.clamp(min=float(lam.max().clamp(min=1e-30)) * damp)
    r = torch.sqrt(lam)
    return (Q * r) @ Q.t(), (Q / r) @ Q.t()


# ---------------------------------------------------------------------------
# VGG
# ---------------------------------------------------------------------------

@torch.no_grad()
def vgg_metrics(model: nn.Module, X: torch.Tensor, chunk: int = 128) -> dict[str, torch.Tensor]:
    feats = model.features
    conv_at = [i for i, m in enumerate(feats) if isinstance(m, nn.Conv2d)]

    masks: dict[int, torch.Tensor] = {}
    h = X
    last_conv = None
    for i, mod in enumerate(feats):
        h = _chunked(mod, h, chunk)
        if isinstance(mod, nn.Conv2d):
            last_conv = i
            masks[i] = _mask_moment(h)
    feat_shape = h.shape[1:]                                 # [C, hh, ww]

    W = model.classifier.weight.detach().double()            # [K, C*hh*ww]
    C, hh, ww = feat_shape
    Wp = W.reshape(W.shape[0], C, hh * ww)
    Om = torch.einsum("kcp,kdp->cd", Wp, Wp) / (hh * ww)     # position-averaged

    out: dict[str, torch.Tensor] = {}
    for i in range(len(feats) - 1, -1, -1):
        mod = feats[i]
        if isinstance(mod, nn.Conv2d):
            out[f"features.{i}"] = _sym(Om).clone()          # metric at THIS conv's output
            Om = _through_conv(Om, mod)
        elif isinstance(mod, nn.ReLU):
            src = max(j for j in conv_at if j < i)
            Om = Om * masks[src]
        elif isinstance(mod, nn.MaxPool2d):
            Om = Om / (mod.kernel_size ** 2 if isinstance(mod.kernel_size, int)
                       else mod.kernel_size[0] * mod.kernel_size[1])
        elif isinstance(mod, nn.AvgPool2d):
            k = mod.kernel_size if isinstance(mod.kernel_size, int) else mod.kernel_size[0]
            Om = Om / (k ** 4)
    out["classifier"] = torch.eye(W.shape[0], dtype=torch.float64, device=X.device)
    return out


# ---------------------------------------------------------------------------
# ResNet
# ---------------------------------------------------------------------------

@torch.no_grad()
def resnet_metrics(model: nn.Module, X: torch.Tensor, chunk: int = 128) -> dict[str, torch.Tensor]:
    rl = torch.relu
    masks: dict[str, torch.Tensor] = {}
    s = _chunked(lambda t: model.bn1(model.conv1(t)), X, chunk)
    masks["stem"] = _mask_moment(s)
    s = rl(s)
    for sname in ("layer1", "layer2", "layer3", "layer4"):
        for bi, blk in enumerate(getattr(model, sname)):
            base = f"{sname}.{bi}"
            z1 = _chunked(blk.conv1, s, chunk)
            masks[f"{base}.conv1"] = _mask_moment(z1)
            h = rl(z1)
            del z1
            if hasattr(blk, "conv3"):
                z2 = _chunked(blk.conv2, h, chunk)
                masks[f"{base}.conv2"] = _mask_moment(z2)
                h = rl(z2)
                del z2
                z = _chunked(blk.conv3, h, chunk)
            else:
                z = _chunked(blk.conv2, h, chunk)
            del h
            z = z + _chunked(blk.shortcut, s, chunk)
            masks[f"{base}.out"] = _mask_moment(z)
            s = rl(z)
            del z
    spatial = s.shape[-1]                                    # 4 for 32x32 inputs
    del s

    W = model.linear.weight.detach().double()
    Om = (W.t() @ W) / (spatial ** 4)                        # avg_pool2d(out, spatial)

    out: dict[str, torch.Tensor] = {}
    for sname in ("layer4", "layer3", "layer2", "layer1"):
        for blk in reversed(list(getattr(model, sname))):
            base = _name_of(model, blk)
            Om_out = Om * masks[f"{base}.out"]               # through the post-add ReLU
            if hasattr(blk, "conv3"):
                out[f"{base}.conv3"] = _sym(Om_out).clone()
                Om_b = _through_conv(Om_out, blk.conv3) * masks[f"{base}.conv2"]
                out[f"{base}.conv2"] = _sym(Om_b).clone()
                Om_b = _through_conv(Om_b, blk.conv2) * masks[f"{base}.conv1"]
                out[f"{base}.conv1"] = _sym(Om_b).clone()
                Om_b = _through_conv(Om_b, blk.conv1)
            else:
                out[f"{base}.conv2"] = _sym(Om_out).clone()
                Om_b = _through_conv(Om_out, blk.conv2) * masks[f"{base}.conv1"]
                out[f"{base}.conv1"] = _sym(Om_b).clone()
                Om_b = _through_conv(Om_b, blk.conv1)
            sc = blk.shortcut
            Om_sc = (_through_conv(Om_out, sc[0]) if len(sc) and isinstance(sc[0], nn.Conv2d)
                     else Om_out)
            Om = Om_b + Om_sc                                # cross term dropped
    out["conv1"] = _sym(Om * masks["stem"]).clone()
    return out


def _name_of(root: nn.Module, mod: nn.Module) -> str:
    for n, m in root.named_modules():
        if m is mod:
            return n
    raise KeyError("block not found")


def pullback_metrics(model: nn.Module, family: str, X: torch.Tensor,
                     chunk: int = 128) -> dict[str, torch.Tensor]:
    """`Omega` at the *output* of every conv (and the classifier), keyed by path."""
    return (vgg_metrics if family == "vgg" else resnet_metrics)(model, X, chunk)
