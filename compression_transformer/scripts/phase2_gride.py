#!/usr/bin/env python3
"""Phase 2: the budget.

`d_l` is measured on the dense network, before anything is cut, on three
candidate unit clouds. The paper's two halves default to two different clouds
and an FFN layer is a legitimate instance of both, so all three are reported and
the default is stated rather than assumed. Seeds 0, 1 and 2 give the sampling
spread of the estimator itself, which is what the width rule inherits.
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch

import core as C

CLOUDS = ("neuron", "filter_proj", "weight")


def main():
    t0 = time.time()
    dev = C.device()
    model = C.load_model(dev)
    out = {"model": C.MODEL_ID, "clouds": {}, "n_units": C.HIDDEN_DIM}

    for seed in (0, 1, 2):
        rows = C.Rows(*C.calibration_set(2048, seed=seed), budget=65536, seed=seed)
        cache = C.dense_targets(model, rows)
        print(f"\nseed {seed}: {rows.n:,} rows")
        for i in range(C.N_LAYERS):
            layer = model.distilbert.transformer.layer[i]
            X, Y = cache[i]
            u = C.Unit(layer.ffn.lin1, layer.ffn.lin2, layer.ffn.activation,
                       X, Y, seed=seed)
            clouds = C.unit_clouds(layer.ffn.lin1.weight.detach(), X, u.U, seed=seed)
            want = CLOUDS if seed == 0 else ("neuron",)
            for name in want:
                d = C.gride_id(clouds[name], seed=seed)
                out["clouds"].setdefault(name, {}).setdefault(str(i), {})[str(seed)] = d
                print(f"  layer {i} {name:12s} d = {d:7.3f}   N/d = "
                      f"{C.HIDDEN_DIM / d:8.1f}")
            u.free()
            del clouds
            torch.cuda.empty_cache()
        del cache, rows
        torch.cuda.empty_cache()

    # default cloud, seed 0, is what every later phase uses
    dims = [out["clouds"]["neuron"][str(i)]["0"] for i in range(C.N_LAYERS)]
    out["default_cloud"] = "neuron"
    out["dims"] = dims
    out["n_over_d"] = [C.HIDDEN_DIM / d for d in dims]
    out["mean_n_over_d"] = float(np.mean(out["n_over_d"]))
    out["log_mean_n_over_d"] = float(np.log(out["mean_n_over_d"]))
    out["ladder"] = C.multiplier_ladder(dims, [C.HIDDEN_DIM] * C.N_LAYERS)

    spread = {}
    for i in range(C.N_LAYERS):
        v = [out["clouds"]["neuron"][str(i)][str(s)] for s in (0, 1, 2)]
        spread[str(i)] = {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                          "values": v}
    out["seed_spread_neuron"] = spread
    out["seconds"] = time.time() - t0

    (C.RESULTS / "phase2_gride.json").write_text(json.dumps(out, indent=2))
    print(f"\ndefault (neuron, seed 0) dims: {[round(d, 2) for d in dims]}")
    print(f"mean N/d = {out['mean_n_over_d']:.1f}, "
          f"log = {out['log_mean_n_over_d']:.3f}")
    print(f"ladder: {out['ladder']}")
    print(f"wrote phase2_gride.json ({out['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
