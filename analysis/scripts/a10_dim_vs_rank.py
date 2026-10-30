#!/usr/bin/env python3
"""A10. The budget's dimension is not the theory's dimension.

Widths in the runs we re-analyse are set by `M_l = round(alpha d_l)`, where `d_l`
is a GRIDE nearest-neighbour intrinsic-dimension estimate of layer `l`'s unit
cloud. Corollary 2, by contrast, is governed by the spectral tail of the design
matrix, `tau_M^2 = sum_{i>M} sigma_i(U)^2`.

A reader is entitled to ask whether the paper is quietly conflating the two. It
is not, and the gap is large: this script puts the GRIDE estimate beside the
spectral effective rank (the number of singular directions holding 99% of the
energy) on the same 22 layers, and reports the ratio. Nothing in the theory
depends on `d_l` being a good estimate of anything; it appears only because it is
how the recorded sweeps indexed their budgets.
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

from load_all import load_all

RES = Path(__file__).resolve().parents[1] / "results"


def main():
    rows = json.loads((RES / "a7_mechanism.json").read_text())["rows"]
    deff = {(r["model"], r["layer"]): (r["d_eff"], r["N"]) for r in rows}

    gride = {}
    for r in load_all():
        if r.arch == "mlp" and r.dims:
            for l, d in enumerate(r.dims):
                gride[(r.model, l)] = d

    out = []
    for k in sorted(deff):
        if k not in gride:
            continue
        de, N = deff[k]
        out.append({"layer": f"{k[0]} L{k[1]}", "N": N,
                    "gride_d": round(gride[k], 2), "eff_rank_99": de,
                    "ratio": round(de / gride[k], 2)})

    ratios = [o["ratio"] for o in out]
    res = {"n_layers": len(out),
           "median_ratio": round(st.median(ratios), 2),
           "min_ratio": min(ratios), "max_ratio": max(ratios),
           "median_gride_d": round(st.median([o["gride_d"] for o in out]), 1),
           "median_eff_rank": st.median([o["eff_rank_99"] for o in out]),
           "per_layer": out}

    print(f"layers compared            {res['n_layers']}")
    print(f"median GRIDE d             {res['median_gride_d']}")
    print(f"median effective rank      {res['median_eff_rank']}")
    print(f"ratio  eff.rank / GRIDE d  median {res['median_ratio']}, "
          f"range {res['min_ratio']} to {res['max_ratio']}")
    (RES / "a10_dim_vs_rank.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {RES / 'a10_dim_vs_rank.json'}")


if __name__ == "__main__":
    main()
