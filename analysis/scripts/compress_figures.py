#!/usr/bin/env python3
"""Compressed raster copies of the four figures the paper actually includes.

The paper itself embeds the PDFs, which are vector and stay as they are. What
this produces is a compressed PNG beside each one, for the places a PDF is
awkward: a slide, a README, an issue thread, an email to a co-author.

Line art with flat fills is exactly the case an indexed palette handles well, so
each figure is quantised to the smallest palette that still reproduces it and
then written with PNG's own optimiser. The palette size is chosen per figure by
measuring the error against the original rather than by picking a number that
sounds safe.

The acceptance test deliberately does not use the worst-case channel error. On an
antialiased plot a handful of edge pixels always land far from their nearest
palette entry, so a max-error rule vetoes palettes that are in fact
indistinguishable and silently falls back to lossless on every figure. What
matters instead is that the error is small on average and confined to a
negligible fraction of pixels, which is what `MAX_RMS` and `MAX_OUTLIER` test.

Only the figures referenced by paper.tex are processed; `fig1_main` and
`fig2_ranking` are left alone, being generated but no longer included.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = ROOT / "figures"
TEX = ROOT / "paper.tex"

MAX_RMS = 1.5        # root-mean-square channel error, out of 255
OUTLIER = 8          # a pixel channel is an outlier past this deviation
MAX_OUTLIER = 0.005  # at most half a percent of them, i.e. antialiasing edges
PALETTES = (16, 32, 48, 64, 96, 128, 192, 256)


def used_in_paper() -> list[str]:
    """Figure stems the paper includes, in order of first appearance."""
    src = TEX.read_text()
    seen, out = set(), []
    for m in re.finditer(r"figures/([A-Za-z0-9_]+)\.pdf", src):
        if m.group(1) not in seen:
            seen.add(m.group(1))
            out.append(m.group(1))
    return out


def flatten(img: Image.Image) -> Image.Image:
    """Matplotlib writes RGBA; composite onto white so the palette is not
    spent encoding transparency the page will never show."""
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    return img.convert("RGB")


def error(ref: np.ndarray, cand: Image.Image) -> tuple[float, float]:
    """Root-mean-square channel error, and the fraction of channel samples
    deviating by more than `OUTLIER`."""
    d = np.abs(ref - np.asarray(cand.convert("RGB"), dtype=np.int16))
    return float(np.sqrt((d.astype(np.float64) ** 2).mean())), float((d > OUTLIER).mean())


def compress(stem: str):
    src = FIG / f"{stem}.png"
    if not src.exists():
        print(f"  {stem}: no PNG to compress, skipped")
        return None
    original = flatten(Image.open(src))
    ref = np.asarray(original, dtype=np.int16)
    before = src.stat().st_size

    chosen = None
    rms = frac = 0.0
    for n in PALETTES:
        cand = original.quantize(colors=n, method=Image.MEDIANCUT,
                                 dither=Image.NONE)
        r, f = error(ref, cand)
        if r <= MAX_RMS and f <= MAX_OUTLIER:
            chosen, rms, frac = cand, r, f
            break
    if chosen is None:                       # nothing passed, stay lossless
        chosen = original

    dst = FIG / f"{stem}_compressed.png"
    chosen.save(dst, format="PNG", optimize=True)
    after = dst.stat().st_size
    ncol = len(chosen.getcolors(1 << 16) or [])
    print(f"  {stem:16s} {before/1024:7.1f} -> {after/1024:6.1f} KB "
          f"({100*(1-after/before):4.1f}% smaller)  {ncol:4d} colours, "
          f"rms {rms:4.2f}, {100*frac:4.2f}% of samples off by >{OUTLIER}")
    return before, after


def main():
    stems = used_in_paper()
    print(f"figures included by paper.tex: {', '.join(stems)}\n")
    tot_b = tot_a = 0
    for s in stems:
        r = compress(s)
        if r:
            tot_b += r[0]
            tot_a += r[1]
    print(f"\n  {'total':16s} {tot_b/1024:7.1f} -> {tot_a/1024:6.1f} KB "
          f"({100*(1-tot_a/tot_b):4.1f}% smaller)")
    print(f"\nwritten to {FIG}")


if __name__ == "__main__":
    main()
