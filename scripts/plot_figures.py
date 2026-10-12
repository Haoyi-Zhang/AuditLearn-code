#!/usr/bin/env python3
"""Render all 512 retained single-blocker widths with optional Matplotlib.

Run: python scripts/plot_figures.py [--data path/to/width_curve.dat]
Computer Modern Roman is bundled with Matplotlib and matches this JMLR paper.
Completion and intersection columns must coincide; the single dashed stroke
retains their joint legend meaning. No simulator, campaign, or raw-data write
is invoked. Native output: figures/widths.pdf, 375 x 180 PDF points; identical
bytes are copied to sibling paper/figures when that directory exists.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil

import matplotlib as mpl
mpl.use("Agg")
from matplotlib import font_manager as fm
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def configure_font() -> str:
    # Matplotlib bundles this genuine Computer Modern Roman face.
    path = Path(mpl.get_data_path()) / "fonts/ttf/cmr10.ttf"
    if not path.is_file():
        raise FileNotFoundError(path)
    fm.fontManager.addfont(str(path))
    family = fm.FontProperties(fname=str(path)).get_name()
    mpl.rcParams.update({
        "font.family": family, "font.size": 10, "mathtext.fontset": "cm",
        "pdf.fonttype": 42, "pdf.use14corefonts": False, "text.usetex": False,
        "axes.unicode_minus": False, "axes.formatter.use_mathtext": True,
        "axes.linewidth": .5, "path.simplify": False,
    })
    return family


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "results/width_curve.dat")
    args = parser.parse_args()
    family = configure_font()
    with args.data.open(encoding="utf-8-sig") as stream:
        keys = next(stream).split()
        rows = [dict(zip(keys, line.split())) for line in stream if line.strip()]
    from decimal import Decimal
    launches = [int(row["launches"]) for row in rows]
    if launches != list(range(1, 513)):
        raise ValueError("Expected the 512 retained launch counts")
    if any(Decimal(row["prefix"]) != 1 or
           Decimal(row["completion"]) != Decimal(row["intersection"]) for row in rows):
        raise ValueError("Single-blocker completion and intersection must coincide")
    fig = plt.figure(figsize=(375 / 72, 180 / 72))
    ax = fig.add_axes([33 / 375, 28 / 180, 334 / 375, 130 / 180])
    ax.plot(launches, [float(row["prefix"]) for row in rows],
            color="black", lw=.8, label="Prefix")
    ax.plot(launches, [float(row["completion"]) for row in rows],
            color="black", lw=.8, ls="--", label="Completion = Intersection")
    ax.set(xlim=(1, 512), ylim=(0, 1.05),
           xlabel="Prior launches of one type", ylabel="Per-launch confidence width")
    ax.set_xticks([1, 128, 256, 384, 512])
    ax.set_yticks([0, .2, .4, .6, .8, 1], ["0", "0.2", "0.4", "0.6", "0.8", "1"])
    ax.tick_params(length=2, width=.5, pad=2)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.04), ncol=2,
              frameon=False, fontsize=10, borderaxespad=0, handlelength=2,
              columnspacing=1.2)
    output = ROOT / "figures/widths.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, metadata={"Creator": None, "Producer": None, "CreationDate": None, "ModDate": None})
    plt.close(fig)
    paper = ROOT.parent / "paper/figures"
    if paper.is_dir():
        shutil.copyfile(output, paper / output.name)
    print(f"widths: {len(rows)} retained rows, 3 columns/2 visible series; embedded {family}")


if __name__ == "__main__":
    main()
