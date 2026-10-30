#!/usr/bin/env python3
"""Rebuild the best configuration, save it, then reload it from disk and
re-evaluate with a fresh model object.

The paper's discipline: an artefact that has only ever existed in the process
that produced it has not been shown to be an artefact. This writes
`artifacts/distilbert_mnli_ffn.safetensors`, constructs a new model with the
compressed shapes, loads the tensors into it, and evaluates that. The two
accuracies must agree exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

import core as C
import harness as H

ART = C.TF_ROOT / "artifacts"


def main():
    s = json.loads((C.RESULTS / "summary.json").read_text())
    best = s["best_frontier"]
    cloud, target = best["config"].split("/")
    alpha, widths = best["alpha_star"], best["widths"]
    H.log(f"best configuration: {cloud} cloud, {target} target, alpha={alpha}")
    H.log(f"widths {widths}")

    sess = H.Session(cloud=cloud)
    path = ART / "distilbert_mnli_ffn.safetensors"
    row = sess.run("omp_ls", alpha, out=C.RESULTS / "artifact.csv", tag="artifact",
                   seed=0, cloud=cloud, target=target, save=path,
                   note="exported artefact")

    # ---- reload into a fresh model and re-evaluate -------------------------
    from safetensors.torch import load_file
    sd = load_file(str(path))
    fresh = C.load_model()
    for i, w in enumerate(widths):
        layer = fresh.distilbert.transformer.layer[i]
        dev, dt = layer.ffn.lin1.weight.device, layer.ffn.lin1.weight.dtype
        layer.ffn.lin1 = nn.Linear(C.DIM, w).to(dev, dt)
        layer.ffn.lin2 = nn.Linear(w, C.DIM).to(dev, dt)
    missing, unexpected = fresh.load_state_dict(sd, strict=True), None
    fresh.eval()
    acc = C.accuracy(fresh, sess.eval_ids, sess.eval_mask, sess.eval_lab)
    pc = C.param_counts(fresh)

    H.log(f"in-process accuracy {float(row['acc']):.4f}")
    H.log(f"reloaded accuracy   {acc:.4f}")
    # the CSV stores accuracy rounded to four places, so that rounding is the
    # tightest agreement this comparison can express
    assert round(acc, 4) == float(row["acc"]), "artefact does not reproduce"

    out = {
        "path": str(path.relative_to(C.TF_ROOT)),
        "config": best["config"], "alpha_star": alpha, "widths": widths,
        "dims": best["dims"],
        "accuracy_in_process": float(row["acc"]),
        "accuracy_reloaded": acc,
        "dense_accuracy": sess.dense_acc,
        "delta": acc - sess.dense_acc,
        "params": pc,
        "red_total_pct": float(row["red_total_pct"]),
        "red_encoder_pct": float(row["red_encoder_pct"]),
        "red_ffn_pct": float(row["red_ffn_pct"]),
        "bytes": path.stat().st_size,
    }
    (C.RESULTS / "artifact.json").write_text(json.dumps(out, indent=2))
    H.log(f"artefact verified: {acc:.2f}% at {out['red_total_pct']:.2f}% "
          f"whole-model reduction, {out['bytes'] / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
