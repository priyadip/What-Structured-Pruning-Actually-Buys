#!/usr/bin/env python3
"""A9. The spectra behind the reversal.

Section "Where selection must matter" argues that target-aware selection recovers
on attention heads because the design matrix there has no spectral tail for the
bound of Corollary 2 to exploit. That is an assertion about the singular values
of two matrices, and it is cheap to check directly rather than to infer from the
accuracy outcome.

For one DistilBERT block we take the two design matrices a selection rule sees:
the post-GELU feed-forward activations, 3072 columns read by `lin2`, and the
concatenated attention head outputs, 768 columns in 12 groups of 64 read by
`out_lin`. Both are the input to a single consumer, both are measured on the
same calibration inputs. What differs is how fast the spectrum decays, and
therefore how much a change of selection can move the retained span.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compression_transformer" / "scripts"))

OUT = Path(__file__).resolve().parents[1] / "results"


def tail(sig, frac):
    """tau_M / ||U||_F at retained fraction `frac`, the quantity in Cor. 2."""
    tot = float((sig ** 2).sum())
    M = max(1, int(round(frac * len(sig))))
    return float(np.sqrt(max((sig[M:] ** 2).sum(), 0.0) / tot))


def main():
    import core as C

    dev = C.device()
    model = C.load_model(dev)
    rows = C.Rows(*C.calibration_set(512, seed=0), budget=16384, seed=0)

    layers = model.distilbert.transformer.layer
    want = {}
    for i, l in enumerate(layers):
        want[f"ffn{i}"] = (l.ffn.lin2, "in")        # post-GELU, 3072 columns
        want[f"att{i}"] = (l.attention.out_lin, "in")  # 12 heads x 64, 768 columns
    got = C.sweep(model, rows, want, batch=64)

    res = {"n_rows": rows.n, "per_block": {}}
    fracs = [0.05, 0.1, 0.25, 0.5, 0.75]
    print(f"rows: {rows.n}")
    print()
    print(f"{'block':>6s} {'unit':>10s} {'cols':>6s} "
          + " ".join(f"tau@{f:.2f}" for f in fracs))
    agg = {"ffn": [], "att": []}
    for i in range(C.N_LAYERS):
        for kind, lab in (("ffn", "FFN (3072)"), ("att", "attn (768)")):
            U = got[f"{kind}{i}"]
            sig = torch.linalg.svdvals(U.double()).cpu().numpy()
            t = [tail(sig, f) for f in fracs]
            agg[kind].append(t)
            res["per_block"].setdefault(str(i), {})[kind] = {
                "cols": int(U.shape[1]),
                "tail": {f"{f:.2f}": round(v, 4) for f, v in zip(fracs, t)},
                "sv_decay_ratio_10pct": float(sig[int(0.1 * len(sig))] / sig[0]),
            }
            print(f"{i:6d} {lab:>10s} {U.shape[1]:6d} "
                  + " ".join(f"{v:8.4f}" for v in t))
        del U

    print()
    print("mean over the six blocks:")
    print(f"{'unit':>12s} " + " ".join(f"tau@{f:.2f}" for f in fracs))
    for kind, lab in (("ffn", "FFN"), ("att", "attention")):
        m = np.mean(np.array(agg[kind]), axis=0)
        res.setdefault("mean_tail", {})[kind] = {f"{f:.2f}": round(float(v), 4)
                                                 for f, v in zip(fracs, m)}
        print(f"{lab:>12s} " + " ".join(f"{v:8.4f}" for v in m))

    # the headline contrast: how much energy survives past a quarter of the units
    ff = res["mean_tail"]["ffn"]["0.25"]
    at = res["mean_tail"]["att"]["0.25"]
    res["ratio_at_quarter"] = round(at / ff, 2)
    print()
    print(f"at a quarter of the units retained, the attention design matrix "
          f"leaves {at:.3f} of its energy in the tail against {ff:.3f} for the "
          f"feed-forward one, a factor of {at / ff:.1f}")

    (OUT / "a9_spectra.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT / 'a9_spectra.json'}")


if __name__ == "__main__":
    main()
