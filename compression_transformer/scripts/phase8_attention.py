#!/usr/bin/env python3
"""Phase 8: attention heads as the second unit type.

`out_lin` reads the concatenation of all twelve heads and nothing else reads
them, so `(heads -> out_lin)` is formally the same producer/consumer pair as an
FFN unit with one change: a head is 64 adjacent columns of the design matrix and
is kept or dropped whole. That is exactly the grouped case the convolutional
kernel already solves, with `gs = 64` in place of `gs = k*k`.

The budget is the awkward part and it is worth being precise about why. GRIDE
estimates a dimension from nearest-neighbour distance ratios, so it needs a
cloud with enough points to have neighbours; twelve heads do not make such a
cloud. The dimension is therefore measured on the 768 output coordinates the
heads produce, which is a well-populated cloud and is the same `neuron` cloud
used for the FFN, and the width `round(alpha d)` is then quantised to whole
heads. The rule is unchanged; only its output has to land on a multiple of 64.

Removing head `h` deletes rows `[64h, 64h+64)` of `q_lin`, `k_lin` and `v_lin`
and the matching columns of `out_lin`, which is 196,800 parameters per head, and
leaves `attention_head_size` and therefore the softmax scaling untouched.
"""

from __future__ import annotations

import argparse
import copy
import json
import time

import numpy as np
import torch
import torch.nn as nn

import core as C
import harness as H

sys_path_note = "linalg comes from compression_cnn/common, unchanged"
from common import linalg as LA                                     # noqa: E402

OUT = C.RESULTS / "attention.csv"
HEAD = 64
N_HEADS = 12


class DirectUnit(C.Unit):
    """A unit whose design matrix is observed rather than rebuilt.

    An FFN unit can regenerate `U` from the producer's input for any candidate
    producer, which is what merging baselines need. Attention cannot: the map
    from hidden states to `out_lin`'s input runs through a softmax, so `U` is
    taken as given. Selection and the refit never needed anything else.
    """

    def __init__(self, U, Y, consumer, gs, ridge=1e-4, seed=0, chunk=16384):
        self.lin2 = consumer
        self.X, self.Y = U, Y
        self.ridge, self.seed, self.chunk = ridge, seed, chunk
        self.gs = gs
        self._U, self._stats = U, None
        self.lin1 = None
        self.act = None

    @property
    def C(self):
        return self.X.shape[1] // self.gs

    @property
    def n_unknowns(self):
        return self.X.shape[1] + 1

    @property
    def U(self):
        return self.X

    def cols(self, keep):
        return LA.group_cols(sorted(keep), self.gs, self.X.device)

    def free(self):
        self._stats = None


# ---------------------------------------------------------------------------
# capture and surgery
# ---------------------------------------------------------------------------

def attn_dense_cache(model, rows, batch=64):
    layers = model.distilbert.transformer.layer
    want = {}
    for i, l in enumerate(layers):
        want[f"u{i}"] = (l.attention.out_lin, "in")
        want[f"Z{i}"] = (l.attention.out_lin, "out")
    got = C.sweep(model, rows, want, batch)
    return {i: (got[f"u{i}"], got[f"Z{i}"]) for i in range(len(layers))}


def attn_comp_input(model, rows, layer, batch=64):
    a = model.distilbert.transformer.layer[layer].attention
    return C.sweep(model, rows, {"u": (a.out_lin, "in")}, batch)["u"]


def _slice_linear(lin: nn.Linear, rows_idx=None, cols_idx=None,
                  W=None, b=None) -> nn.Linear:
    Wn = lin.weight.detach() if W is None else W
    bn = lin.bias.detach() if b is None else b
    if rows_idx is not None:
        Wn, bn = Wn[rows_idx], bn[rows_idx]
    if cols_idx is not None:
        Wn = Wn[:, cols_idx]
    out = nn.Linear(Wn.shape[1], Wn.shape[0]).to(Wn.device, Wn.dtype)
    with torch.no_grad():
        out.weight.copy_(Wn)
        out.bias.copy_(bn)
    for p in out.parameters():
        p.requires_grad_(False)
    return out


def prune_heads(attn, keep_heads: list[int], V_new=None, c_new=None):
    dev = attn.q_lin.weight.device
    idx = torch.cat([torch.arange(h * HEAD, (h + 1) * HEAD, device=dev)
                     for h in sorted(keep_heads)])
    attn.q_lin = _slice_linear(attn.q_lin, rows_idx=idx)
    attn.k_lin = _slice_linear(attn.k_lin, rows_idx=idx)
    attn.v_lin = _slice_linear(attn.v_lin, rows_idx=idx)
    if V_new is None:
        attn.out_lin = _slice_linear(attn.out_lin, cols_idx=idx)
    else:
        attn.out_lin = _slice_linear(attn.out_lin, W=V_new, b=c_new)
    attn.n_heads = len(keep_heads)
    # attention_head_size and scaling are unchanged: a head keeps its 64 dims


# ---------------------------------------------------------------------------
# strategies over heads
# ---------------------------------------------------------------------------

def a_omp_ls(u, h):
    G, P, us, ys, n = u.stats
    keep = LA.group_select(G, P, u.gs, h)
    return keep, u.ls(keep)


def a_omp(u, h):
    G, P, us, ys, n = u.stats
    keep = LA.group_select(G, P, u.gs, h)
    return keep, None


def a_span_ls(u, h):
    G, P, us, ys, n = u.stats
    keep = LA.group_select(G, None, u.gs, h)
    return keep, u.ls(keep)


def a_magnitude_ls(u, h):
    V = u.lin2.weight.detach()
    imp = V.reshape(V.shape[0], u.C, u.gs).norm(dim=(0, 2))
    keep = torch.topk(imp, h).indices.sort().values.tolist()
    return keep, u.ls(keep)


def a_magnitude(u, h):
    V = u.lin2.weight.detach()
    imp = V.reshape(V.shape[0], u.C, u.gs).norm(dim=(0, 2))
    keep = torch.topk(imp, h).indices.sort().values.tolist()
    return keep, None


def a_random_ls(u, h):
    g = torch.Generator().manual_seed(u.seed)
    keep = sorted(torch.randperm(u.C, generator=g)[:h].tolist())
    return keep, u.ls(keep)


A_STRATEGIES = {"omp_ls": a_omp_ls, "omp": a_omp, "span_ls": a_span_ls,
                "magnitude_ls": a_magnitude_ls, "magnitude": a_magnitude,
                "random_ls": a_random_ls}


# ---------------------------------------------------------------------------
# the walk
# ---------------------------------------------------------------------------

def compress_attention(dense, rows, heads: list[int], strategy: str,
                       acache: dict, *, ridge=1e-4, seed=0, batch=64,
                       ffn_widths=None, fcache=None, ffn_strategy="omp_ls"):
    """Walk the blocks, compressing attention and optionally the FFN in the same
    pass. Attention comes first inside a block because that is the order the
    forward pass runs, so the FFN's design matrix already reflects the pruned
    attention above it."""
    comp = copy.deepcopy(dense)
    comp.eval()
    fn = A_STRATEGIES[strategy]
    for i in range(C.N_LAYERS):
        h = int(heads[i])
        attn = comp.distilbert.transformer.layer[i].attention
        if h < N_HEADS:
            Ud = attn_comp_input(comp, rows, i, batch)
            _, Y = acache[i]
            u = DirectUnit(Ud, Y, attn.out_lin, HEAD, ridge=ridge, seed=seed)
            keep, refit = fn(u, h)
            if refit is None:
                prune_heads(attn, keep)
            else:
                prune_heads(attn, keep, refit[0], refit[1])
            u.free()
            del Ud
            torch.cuda.empty_cache()
        if ffn_widths is not None:
            m = int(ffn_widths[i])
            layer = comp.distilbert.transformer.layer[i]
            a_dense, Yf = fcache[i]
            X = C.comp_input(comp, rows, i, batch)
            fu = C.Unit(layer.ffn.lin1, layer.ffn.lin2, layer.ffn.activation,
                        X, Yf, ridge=ridge, seed=seed)
            fu.require_overdetermined(2.0)
            if m < fu.C:
                W1, b1, W2, b2 = C.STRATEGIES[ffn_strategy](fu, m)
                C._replace_ffn(layer, W1, b1, W2, b2)
            fu.free()
            del X
            torch.cuda.empty_cache()
    return comp


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="1,2,3,4,6,8")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--joint", action="store_true",
                    help="also compress the FFN at its frontier budget")
    ap.add_argument("--ffn-alpha", type=float, default=12.125)
    ap.add_argument("--strategies",
                    default="omp_ls,span_ls,magnitude_ls,random_ls,omp,magnitude")
    a = ap.parse_args()

    s = H.Session()
    rows, fcache = s.cache(a.seed)
    acache = attn_dense_cache(s.model, rows)

    # --- the budget, measured on the 768 coordinates the heads produce -------
    gpath = C.RESULTS / "phase8_attention_gride.json"
    if gpath.exists():
        dims = json.loads(gpath.read_text())["dims"]
    else:
        dims = []
        for i in range(C.N_LAYERS):
            U, _ = acache[i]
            rng = np.random.default_rng(a.seed)
            sel = torch.as_tensor(np.sort(rng.choice(U.shape[0],
                                                     min(U.shape[0], 2048),
                                                     replace=False)),
                                  device=U.device)
            cloud = U.index_select(0, sel).t().contiguous().cpu().numpy()
            d = C.gride_id(cloud, seed=a.seed)
            dims.append(d)
            H.log(f"  attention layer {i}: d = {d:.2f} over 768 coordinates, "
                  f"{d / HEAD:.2f} heads' worth")
        # for reference only: GRIDE on the twelve heads themselves, which is the
        # estimate the report argues is not usable
        head_dims = []
        for i in range(C.N_LAYERS):
            U, _ = acache[i]
            hc = U.mean(0).reshape(N_HEADS, HEAD).cpu().numpy()
            try:
                head_dims.append(C.gride_id(hc, seed=a.seed))
            except Exception as exc:
                head_dims.append(None)
                H.log(f"  head-cloud GRIDE at layer {i} refused: {exc}")
        gpath.write_text(json.dumps({"dims": dims, "head_cloud_dims": head_dims,
                                     "n_points": 768, "head_size": HEAD}, indent=2))

    H.log(f"attention dims {[round(d, 2) for d in dims]}")

    base = C.param_counts(s.model)
    for alpha in [float(x) for x in a.alphas.split(",")]:
        heads = [int(np.clip(round(round(alpha * d) / HEAD), 1, N_HEADS))
                 for d in dims]
        H.log(f"alpha {alpha}: heads {heads}")
        for stg in [x for x in a.strategies.split(",") if x]:
            t0 = time.time()
            ffn_w = (s.widths(a.ffn_alpha) if a.joint else None)
            comp = compress_attention(s.model, rows, heads, stg, acache,
                                      seed=a.seed, ffn_widths=ffn_w,
                                      fcache=fcache if a.joint else None)
            acc = C.accuracy(comp, s.eval_ids, s.eval_mask, s.eval_lab)
            pc = C.param_counts(comp)
            row = {
                "timestamp": H.datetime.now().isoformat(timespec="seconds"),
                "tag": "joint" if a.joint else "attn_only",
                "strategy": stg, "alpha": alpha, "seed": a.seed,
                "cloud": "neuron_attn", "target": "dense", "sequential": 1,
                "ridge": 1e-4, "calib": s.calib, "rows": rows.n,
                "layers": json.dumps(heads),
                "acc": round(acc, 4), "dense_acc": round(s.dense_acc, 4),
                "delta": round(acc - s.dense_acc, 4),
                "hit": int(abs(acc - s.dense_acc) <= s.TOL),
                "params_total": pc["total"], "params_encoder": pc["encoder"],
                "params_ffn": pc["ffn"],
                "red_total_pct": round(100 * (1 - pc["total"] / base["total"]), 4),
                "red_encoder_pct": round(100 * (1 - pc["encoder"] / base["encoder"]), 4),
                "red_ffn_pct": round(100 * (1 - pc["ffn"] / base["ffn"]), 4),
                "widths": json.dumps(heads),
                "dims": json.dumps([round(d, 3) for d in dims]),
                "seconds": round(time.time() - t0, 1),
                "note": f"ffn_alpha={a.ffn_alpha}" if a.joint else "attention only",
            }
            H.csv_append(OUT, row)
            H.log(f"  {stg:14s} a={alpha:<5g} heads={heads} acc={acc:6.2f} "
                  f"({row['delta']:+6.2f}) red tot={row['red_total_pct']:5.1f}%")
            del comp
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
