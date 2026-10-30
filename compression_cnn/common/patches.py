"""Turning a convolution into the matrix problem the MLP study already solved.

A conv layer with weight `V [C_next, C, k, k]` applied to input `A [N, C, H, W]`
is the matrix product

        Y_flat = U V_flat^T + c ,
        U      = unfold(A)        [N*H_out*W_out , C*k*k]
        V_flat = V.reshape(C_next, C*k*k)

so every closed form in `../compression_mlp` transfers unchanged. Two practical points
this module handles:

**Row subsampling.** `N*H_out*W_out` is in the millions; the least-squares system
has `C*k*k` unknowns per output channel, so a few tens of thousands of rows is
already a heavily over-determined fit. Rows are drawn once per unit and the
*same* row set indexes both `U` and the target, which is what makes the two
consistent.

**Group structure.** The columns of `U` come in contiguous blocks of `k*k`, one
block per input channel. Removing a channel removes a whole block, so selection
must be group-wise -- this is the only structural difference from the MLP case,
where every column was its own group (`k*k = 1`).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def out_hw(conv: nn.Conv2d, hw: tuple[int, int]) -> tuple[int, int]:
    h, w = hw
    kh, kw = conv.kernel_size
    ph, pw = conv.padding if isinstance(conv.padding, tuple) else (conv.padding, conv.padding)
    sh, sw = conv.stride
    dh, dw = conv.dilation
    return ((h + 2 * ph - dh * (kh - 1) - 1) // sh + 1,
            (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1)


def n_positions(consumer: nn.Module, A: torch.Tensor) -> int:
    """Rows contributed by one image: `H_out*W_out` for a conv, 1 for a Linear."""
    if isinstance(consumer, nn.Conv2d):
        oh, ow = out_hw(consumer, A.shape[-2:])
        return oh * ow
    return 1


def group_size(consumer: nn.Module, A: torch.Tensor) -> int:
    """Columns of the design matrix that belong to one input channel."""
    if isinstance(consumer, nn.Conv2d):
        return consumer.kernel_size[0] * consumer.kernel_size[1]
    return int(A.shape[-2] * A.shape[-1])          # a Linear reads the flattened map


def sample_rows(n_images: int, per_image: int, budget: int, generator: torch.Generator,
                device: str) -> torch.Tensor:
    """Global row ids into the `[n_images*per_image, ...]` design matrix."""
    total = n_images * per_image
    if total <= budget:
        return torch.arange(total, device=device)
    idx = torch.randperm(total, generator=generator)[:budget].to(device)
    return idx.sort().values


@torch.no_grad()
def design_matrix(A: torch.Tensor, consumer: nn.Module, rows: torch.Tensor,
                  per_image: int, chunk: int = 64) -> torch.Tensor:
    """`U[rows]` for the given consumer, built one image-chunk at a time.

    Materialising the full unfold would cost `N * C*k*k * H*W` floats (gigabytes
    at 32x32 with 512 channels); chunking keeps the peak at `chunk` images while
    the returned matrix only ever holds the sampled rows.
    """
    N = A.shape[0]
    out, ptr = [], 0
    for i0 in range(0, N, chunk):
        i1 = min(i0 + chunk, N)
        lo, hi = i0 * per_image, i1 * per_image
        j1 = ptr
        while j1 < rows.numel() and rows[j1] < hi:
            j1 += 1
        sel = rows[ptr:j1] - lo
        ptr = j1
        if sel.numel() == 0:
            continue
        blk = A[i0:i1]
        if isinstance(consumer, nn.Conv2d):
            U = F.unfold(blk, consumer.kernel_size, consumer.dilation,
                         consumer.padding, consumer.stride)          # [n, C*k*k, L]
            U = U.transpose(1, 2).reshape(-1, U.shape[1])            # [n*L, C*k*k]
        else:
            U = blk.reshape(blk.shape[0], -1)                        # [n, C*H*W]
        out.append(U.index_select(0, sel))
    return torch.cat(out) if out else torch.zeros(0, 0, device=A.device)


@torch.no_grad()
def target_matrix(Y: torch.Tensor, rows: torch.Tensor, per_image: int) -> torch.Tensor:
    """The consumer's dense pre-activation, flattened to match `design_matrix`.

    `F.unfold` orders positions as `y*W_out + x`, and `Y` is `[N, C_next, H_out,
    W_out]`, so moving the channel axis last and flattening `(N, H_out, W_out)`
    puts the two in the same row order by construction.
    """
    if Y.dim() == 4:
        Yf = Y.permute(0, 2, 3, 1).reshape(-1, Y.shape[1])
    else:
        Yf = Y.reshape(Y.shape[0], -1)
    assert Yf.shape[0] == Y.shape[0] * per_image, (Yf.shape, per_image)
    return Yf.index_select(0, rows)


@torch.no_grad()
def chunked_forward(module: nn.Module, X: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    return torch.cat([module(X[i:i + chunk]) for i in range(0, X.shape[0], chunk)])
