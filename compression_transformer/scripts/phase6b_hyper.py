#!/usr/bin/env python3
"""Phase 6b: the two knobs that are not part of the method's claim.

The ridge and the number of design-matrix rows are implementation choices, not
contributions, so the report has to show that the frontier is not resting on a
lucky setting of either. Both are swept at the frontier budget with everything
else fixed.
"""

from __future__ import annotations

import argparse

import core as C
import harness as H

OUT = C.RESULTS / "hyper.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=12.0)
    ap.add_argument("--ridges", default="1e-6,1e-5,1e-4,1e-3,1e-2")
    ap.add_argument("--rowsets", default="16384,32768,65536,131072")
    a = ap.parse_args()

    s = H.Session()
    for r in [float(x) for x in a.ridges.split(",")]:
        s.run("omp_ls", a.alpha, out=OUT, tag="ridge", ridge=r,
              target="residual", note=f"ridge={r:g}")
    del s

    for n in [int(x) for x in a.rowsets.split(",")]:
        s2 = H.Session(calib=max(2048, n // 32), rows_budget=n)
        s2.run("omp_ls", a.alpha, out=OUT, tag="rows", target="residual",
               note=f"rows_budget={n}")
        del s2
        C.torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
