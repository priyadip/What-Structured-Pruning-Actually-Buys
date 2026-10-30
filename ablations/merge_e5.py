#!/usr/bin/env python3
"""Merge the extended fashionmnist grid back into the main E5 CSV.

`e5_joint_budget.py` rewrites its output file each run, so the extension pass
would otherwise discard the first three models. This concatenates and
de-duplicates on (model, alpha_ch, alpha_r), keeping the later row.
"""
import csv
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"
main, ext = RES / "e5_joint_budget.csv", RES / "e5_joint_budget_ext.csv"
if not ext.exists():                     # the extension wrote over the main file
    main.replace(ext)
    print("no separate ext file; treating the latest write as the extension")

rows = {}
for f in (RES / "e5_joint_budget_full.csv", ext):
    if not f.exists():
        continue
    with f.open() as fh:
        for r in csv.DictReader(fh):
            rows[(r["model"], r["alpha_ch"], r["alpha_r"])] = r

out = sorted(rows.values(), key=lambda r: (r["model"], float(r["alpha_ch"]),
                                           float(r["alpha_r"] or 1e9)))
with (RES / "e5_joint_budget.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, list(out[0].keys()))
    w.writeheader()
    w.writerows(out)
(RES / "e5_joint_budget_full.csv").write_text((RES / "e5_joint_budget.csv").read_text())
print(f"merged -> {len(out)} rows, {len({r['model'] for r in out})} models")
