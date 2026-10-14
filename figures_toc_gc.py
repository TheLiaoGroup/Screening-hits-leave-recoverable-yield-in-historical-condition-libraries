"""TOC graphic, 3.25 x 1.75 in at 300 dpi: the Figure 1b pair and every measured yield of it.

The pair, its ChemDraw master and the yields are those of Figure 1b (example_pair_gc.py); the header is
the cohort result of Figure 2.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figure1_gc

HERE = Path(__file__).resolve().parent
OUT, FIG = HERE / "out", HERE / "figures"
W, H, DPI = 3.25, 1.75, 300
BLUE, LIGHT, GREY, GAP = "#0072B2", "#56B4E9", "#B0B0B0", "#8A3B00"
WORDS = {4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
ROLES = ["initial six (Greedy)", "six added from history (Greedy)", "rest of library"]


def example():
    """The Figure 1b pair, its yields and the cohort rate, read from the deposited outputs."""
    E = json.loads((OUT / "example_pair.json").read_text(encoding="utf-8"))
    D = pd.read_csv(OUT / "example_pair.csv")
    assert set(D.role) == set(ROLES) and len(D) == E["measured_conditions"]
    first = D[D.role == ROLES[0]]
    assert np.isclose(first["yield"].max(), E["m6"]) and np.isclose(D["yield"].max(), E["b"])
    # Hits below 80%, the only hits that can fall more than twenty points short (Figure 2a).
    summary = json.loads((OUT / "posthit_summary.json").read_text(encoding="utf-8"))["below_ceiling"]
    return E, D, {"hits": summary["pairs"], "severe": summary["severe_k6"],
                  "one_in": int(round(summary["pairs"] / summary["severe_k6"]))}


def scheme(fig, E, x, top, width):
    """Place the ChemDraw master scaled to `width` inches; RDKit drawing if the master is absent."""
    from PIL import Image
    expected, master = figure1_gc.example_master(E)
    if master is not None:
        import pymupdf
        with pymupdf.open(master) as document:
            page = document[0]
            scale = width / (page.rect.width / 72)
            zoom = DPI * scale / 72
            pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=True)
            raster = Image.frombytes("RGBA", [pix.width, pix.height], pix.samples)
    else:
        from types import SimpleNamespace
        from chemical_schemes_gc import scheme_drawing
        row = SimpleNamespace(reactant1_smiles=expected[0], reactant2_smiles=expected[1],
                              product_smiles=expected[2])
        raster = scheme_drawing(row)[0]
        raster = raster.resize((int(width * DPI), int(width * DPI * raster.height / raster.width)))
    h = raster.height / DPI
    ax = fig.add_axes([x / W, (top - h) / H, width / W, h / H]); ax.axis("off")
    ax.imshow(raster, interpolation="nearest")
    fig._toc_master = master
    return top - h


def graphic():
    plt.rcParams.update({"font.family": "Arial", "font.size": 7.5})
    E, D, cohort = example()
    fig = plt.figure(figsize=(W, H))
    fig.text(W / 2 / W, 1.71 / H, "One hit in %s below 80%% leaves >20 yield points in its library"
             % WORDS[cohort["one_in"]], ha="center", va="top", fontsize=7.5, fontweight="bold")
    scheme(fig, E, 0.08, 1.49, W - 0.16)

    ax = fig.add_axes([0.16 / W, 0.32 / H, 2.86 / W, 0.66 / H])
    hit, best = 100 * E["m6"], 100 * E["b"]
    colours = {ROLES[0]: BLUE, ROLES[1]: LIGHT, ROLES[2]: GREY}
    for role in (ROLES[2], ROLES[0], ROLES[1]):
        d = D[D.role == role].sort_values("yield").reset_index(drop=True)
        jitter = 0.07 * ((np.arange(len(d)) % 3) - 1) if role == ROLES[2] else np.zeros(len(d))
        ax.scatter(100 * d["yield"], 0.2 + jitter, s=11, color=colours[role], lw=0, zorder=3)
    ax.scatter([hit], [0.2], s=34, facecolor="none", edgecolor="black", lw=0.8, zorder=4)
    ax.scatter([best], [0.2], s=70, marker=(5, 1, 0), facecolor=LIGHT, edgecolor="black", lw=0.6, zorder=4)
    # The gap of this pair, drawn between its two measured yields only.
    ax.annotate("", xy=(best - 1.5, 0.45), xytext=(hit, 0.45),
                arrowprops=dict(arrowstyle="->", lw=0.8, color=GAP, shrinkA=0, shrinkB=0))
    ax.text((hit + best) / 2, 0.53, "hit %.1f%%, library best %.1f%%" % (hit, best),
            ha="center", va="bottom", fontsize=7.5, color=GAP)
    ax.set_xlim(0, 100); ax.set_ylim(0, 1.0)
    ax.set_yticks([]); ax.set_xticks([0, 50, 100]); ax.set_xticklabels(["0%", "50%", "100%"])
    ax.tick_params(labelsize=7.5, length=2, pad=1)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6)

    legend = [("first six", BLUE), ("next six, from yield history", LIGHT), ("rest of library", GREY)]
    x = 0.10
    for label, colour in legend:
        fig.text(x / W, 0.075 / H, "●", color=colour, fontsize=7.5, va="center")
        t = fig.text((x + 0.09) / W, 0.075 / H, label, fontsize=7, va="center")
        fig.canvas.draw()
        x = t.get_window_extent().x1 / fig.dpi + 0.10
    fig._toc_example = E
    fig._toc_cohort = cohort
    return fig


if __name__ == "__main__":
    fig = graphic()
    fig.savefig(FIG / "TOC_graphic.tif", dpi=DPI, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(FIG / "TOC_graphic.png", dpi=DPI)
    plt.close(fig)
    print("TOC graphic written: 975 x 525 px; pair " + json.loads(
        (OUT / "example_pair.json").read_text(encoding="utf-8"))["subpair"])
