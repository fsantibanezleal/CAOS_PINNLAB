#!/usr/bin/env python3
"""Regenerate the figures for the PINN-Lab method-atlas report from the COMMITTED derived artifacts. Two figures:

  fig-atlas.pdf     - the honest relative-L2 error of every catalogued case, on a log axis, coloured by problem
                      category, with reference lines at 0.1%, 1% and 10%: PINNs reach sub-percent error on smooth
                      well-posed forward PDEs and degrade by orders of magnitude on inverse, high-frequency,
                      operator-learning and real-data cases.
  fig-solution.pdf  - a representative solved case (Allen-Cahn, a nonlinear reaction-diffusion PDE): the PINN
                      solution field and its pointwise error against the reference (relative-L2 ~1e-3).

Run:  python make_figs.py     (from repo root; reads data/derived/ and ../data/atlas.json)
Deps: matplotlib, numpy.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                          # repo root (figures/../../ = repo)
DER = ROOT / "data" / "derived"
DATA = HERE.parent / "data"

INK = "#1a1a2e"
GRID = "#d8d8e0"

CATCOL = {
    "canonical PDE": "#1b6ca8", "inverse": "#e07a3f", "dynamical": "#7d5ba6",
    "environmental (real/field)": "#b23a48", "mining process": "#3fa34d",
    "pollution transport": "#c99a1e", "control": "#5a6b7b", "epidemic ODE": "#8a5a44", "other": "#999999",
}

plt.rcParams.update({
    "font.family": "serif", "font.size": 9.4, "axes.edgecolor": INK,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 0.8, "figure.dpi": 200,
})


def fig_atlas():
    d = json.loads((DATA / "atlas.json").read_text(encoding="utf-8"))
    cases = sorted(d["cases"], key=lambda r: r["l2_relative"])
    labels = [c["case"] for c in cases]
    vals = [c["l2_relative"] for c in cases]
    cols = [CATCOL.get(c["category"], "#999") for c in cases]
    y = np.arange(len(cases))
    fig, ax = plt.subplots(figsize=(6.7, 5.0))
    ax.barh(y, vals, color=cols, edgecolor=INK, linewidth=0.5, height=0.68, zorder=3)
    for ref, lab in [(1e-3, "0.1%"), (1e-2, "1%"), (1e-1, "10%")]:
        ax.axvline(ref, color="#888", linewidth=0.9, linestyle="--", zorder=1)
        ax.text(ref, len(cases) - 0.2, lab, fontsize=7.2, color="#666", ha="center", va="bottom")
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.6)
    ax.set_xlabel("relative $L_2$ error vs reference (log scale)")
    ax.set_title("PINN accuracy atlas: sub-percent on smooth forward PDEs,\norders worse on inverse / high-frequency / real-data cases",
                 fontsize=8.8)
    ax.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    seen = []
    handles = []
    for c in cases:
        if c["category"] not in seen:
            seen.append(c["category"])
            handles.append(plt.Rectangle((0, 0), 1, 1, color=CATCOL.get(c["category"], "#999")))
    ax.legend(handles, seen, fontsize=6.8, frameon=True, facecolor="white", edgecolor=GRID,
              loc="lower right", ncol=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / "fig-atlas.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_solution():
    comp = json.loads((DER / "bench-allencahn" / "comparison.json").read_text(encoding="utf-8"))
    ax_ = comp["axes"]
    x = np.asarray(ax_["x"]); t = np.asarray(ax_["t"])
    u = np.asarray(comp["fields"]["adapted"])
    err = np.abs(np.asarray(comp["fields"]["err_adapted"]))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.0))
    ext = [t.min(), t.max(), x.min(), x.max()]
    im1 = a1.imshow(u, origin="lower", aspect="auto", extent=ext, cmap="RdBu_r")
    a1.set_xlabel("t"); a1.set_ylabel("x"); a1.set_title("(a) PINN solution $u(x,t)$", fontsize=9.0)
    fig.colorbar(im1, ax=a1, fraction=0.046, pad=0.02)
    im2 = a2.imshow(err, origin="lower", aspect="auto", extent=ext, cmap="magma")
    a2.set_xlabel("t"); a2.set_ylabel("x")
    a2.set_title("(b) pointwise $|$error$|$ vs reference", fontsize=9.0)
    fig.colorbar(im2, ax=a2, fraction=0.046, pad=0.02)
    fig.suptitle("Allen-Cahn (nonlinear reaction-diffusion), relative $L_2$ $\\approx$ 1.2e-3",
                 fontsize=9.2, y=1.02)
    fig.tight_layout()
    fig.savefig(HERE / "fig-solution.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    fig_atlas()
    fig_solution()
    print("wrote fig-atlas.pdf, fig-solution.pdf")


if __name__ == "__main__":
    main()
