"""Figure 1: what earlier screening studies read from a pair x condition matrix, and one measured hit
that its library beats.

 a  schematic: optimisation, general conditions, coverage, and the residual-yield reading of this
    study (no data are plotted in this panel)
 b  the pair chosen by example_pair_gc.py: structures (ChemDraw master, RDKit fallback), and every
    measured yield of the pair, split into the initial greedy six, the six added from history and the
    rest of the library
The variable-level heatmap of the eighteen libraries is Supplementary Figure 6 (figures_kits.sfig6).

Writes figures/Figure1_hit_gap.png and a PDF whose chemical structures are vector paths.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from rdkit import Chem
from rdkit.Chem import AllChem

import figures_kits as fk
from chemical_schemes_gc import DPI, scheme_drawing

HERE = Path(__file__).resolve().parent
OUT, FIG = HERE / "out", HERE / "figures"
STEM = "Figure1_hit_gap"
W, H = 7.0, 5.45
BLUE, LIGHT, GREY, DARK, ORANGE = fk.BLUE, "#9ECAE1", fk.GREY, fk.DARK, fk.ORANGE
CELL = "#E6E6E6"
TEXT = 9.0


def chem(text):
    """Formula digits as Arial subscripts through mathtext (Arial has no Unicode subscript digits)."""
    import re
    from presentation_gc import _FORMULA_PATTERN
    def sub(match):
        f = match.group()
        return "$\\mathregular{" + re.sub(r"(?<=[A-Za-z)\]])(\d+)", r"_{\1}", f) + "}$"
    return _FORMULA_PATTERN.sub(sub, text)


def inch_axes(fig, x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


def letter(fig, y, label):
    t = fig.text(0.06 / W, y / H, label, fontsize=10.5, fontweight="bold", va="baseline")
    fig._panel_letters = getattr(fig, "_panel_letters", []) + [(t, label)]
    return t


def heading(fig, y, text):
    return fig.text(0.30 / W, y / H, text, fontsize=9.5, va="baseline")


def grid(ax, fills=(), rows=6, cols=8):
    ax.set_xlim(-0.1, 10.2); ax.set_ylim(rows + 0.1, -0.1); ax.axis("off")
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(Rectangle((c, r), 1, 1, facecolor=fills.get((r, c), CELL), edgecolor="white", lw=0.8))


def panel_a(fig):
    letter(fig, 5.29, "a")
    heading(fig, 5.29, "Representative evaluation endpoints for a substrate-pair × condition matrix (schematic)")
    # Same order as the Introduction, which follows the reference numbering.
    kinds = ["optimisation", "general", "coverage", "this work"]
    titles = {"optimisation": "Optimisation", "general": "General conditions", "coverage": "Coverage",
              "this work": "This work: residual yield"}
    notes = {"coverage": "Pairs that a few conditions take\npast a detection or yield\nthreshold (refs 6, 30)",
             "general": "One condition, best yield\naveraged over many pairs\n(refs 23, 24)",
             "optimisation": "Successive experiments\nraise one reaction's yield\n(refs 19–21)",
             "this work": "Residual measured yield of\na fixed hit cohort\n(1,428 hits, 18 libraries)"}
    for j, kind in enumerate(kinds):
        x0 = 0.15 + j * 1.72
        fig.text((x0 + 0.02) / W, 4.95 / H, titles[kind], fontsize=TEXT, fontweight="bold", va="baseline")
        ax = inch_axes(fig, x0, 3.90, 1.62, 0.9)
        if kind == "coverage":
            grid(ax, {(r, c): LIGHT for r in range(6) for c in (1, 4, 6)})
            for r in range(6):
                hit = r != 2
                ax.plot(8.9, r + 0.5, "o", ms=5, mfc=BLUE if hit else "white", mec=BLUE, mew=0.8)
        elif kind == "general":
            grid(ax, {(r, 3): BLUE for r in range(6)})
        elif kind == "optimisation":
            seq = [(2, 1), (2, 5), (2, 2), (2, 6)]
            grid(ax, {cell: BLUE for cell in seq})
            for k, (r, c) in enumerate(seq, 1):
                ax.text(c + 0.5, r + 0.55, str(k), ha="center", va="center", fontsize=8.5, color="white")
        else:
            best = [2, 6, 0, 7, 5, 3]
            panel = (0, 2, 5)
            grid(ax, {(r, c): LIGHT for r in range(6) for c in panel})
            gaps = [0.0, 1.3, 0.0, 1.7, 0.0, 0.8]
            for r, (c, g) in enumerate(zip(best, gaps)):
                ax.plot(c + 0.5, r + 0.5, marker=(5, 1, 0), ms=6.5, color="black", mew=0.4)
                if g:
                    ax.add_patch(Rectangle((8.3, r + 0.25), g, 0.5, facecolor=ORANGE, edgecolor="none"))
        fig.text((x0 + 0.02) / W, 3.77 / H, notes[kind], fontsize=8.5, va="top", linespacing=1.15)


def product_smiles(aryl_halide, boronic_acid):
    rxn = AllChem.ReactionFromSmarts("[c:1][Br,Cl,I].[c:2]B(O)O>>[c:1]-[c:2]")
    products = {Chem.MolToSmiles(p[0]) for p in rxn.RunReactants((Chem.MolFromSmiles(aryl_halide),
                                                                   Chem.MolFromSmiles(boronic_acid)))}
    assert len(products) == 1, products
    return products.pop()


def example_master(E):
    """Return the expected SMILES and the ChemDraw master of the example pair (None if absent).

    A master that shows another pair is refused; Figure 1b and the TOC graphic both read it here."""
    expected = [E["sub_1_smiles"], E["sub_2_smiles"], product_smiles(E["sub_1_smiles"], E["sub_2_smiles"])]
    master = FIG / "example_pair_scheme.pdf"
    if not master.exists():
        return expected, None
    cdxml = (FIG / "example_pair_scheme.cdxml").read_text(encoding="utf-8")
    drawn = {Chem.MolToSmiles(m) for m in Chem.MolsFromCDXML(cdxml)}
    assert drawn == {Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in expected}, \
        "the ChemDraw master shows another pair; rerun tools/build_chemdraw_example_gc.py"
    return expected, master


def panel_b(fig, vectors):
    E = json.loads((OUT / "example_pair.json").read_text(encoding="utf-8"))
    D = pd.read_csv(OUT / "example_pair.csv")
    letter(fig, 2.95, "b")
    heading(fig, 2.95, "One screening hit, and the recipes its %s library still holds" % fk.array_label(E["screen"]))
    expected, master = example_master(E)
    if master is not None:
        # The ChemDraw master (tools/build_chemdraw_example_gc.py), placed at its physical size.
        import pymupdf
        import xml.etree.ElementTree as ET
        from PIL import Image
        cdxml = (FIG / "example_pair_scheme.cdxml").read_text(encoding="utf-8")
        with pymupdf.open(master) as document:
            page = document[0]
            pix = page.get_pixmap(dpi=DPI, alpha=True)
            raster = Image.frombytes("RGBA", [pix.width, pix.height], pix.samples)
            w, h = page.rect.width / 72, page.rect.height / 72
        root = ET.fromstring(cdxml)
        label_size = float(root.attrib["LabelSize"])
        glyphs = [{"font_px": label_size * DPI / 72, "text": "native atom labels", "role": "atom"}]
        for span in root.iter("s"):
            script = int(span.get("face", "0")) & (32 | 64)
            glyphs.append({"font_px": float(span.get("size", label_size)) * (0.75 if script else 1) * DPI / 72,
                           "text": "".join(span.itertext()), "role": "subscript or charge" if script else "atom"})
        vector = master
    else:
        row = SimpleNamespace(reactant1_smiles=expected[0], reactant2_smiles=expected[1], product_smiles=expected[2])
        raster, lengths, glyphs, vector = scheme_drawing(row)
        w, h = raster.width / DPI, raster.height / DPI
        fig._chemical_geometry_audit = [{"transformation": "Figure 1b example", "bond_lengths_px": lengths,
                                         "glyphs": glyphs}]
    sx = inch_axes(fig, (W - w) / 2, 2.37 - h / 2, w, h); sx.axis("off")
    image = sx.imshow(raster, interpolation="nearest")
    sx._chemical_font_audit = [(image, glyphs)]
    vectors.append((image, vector))

    ax = inch_axes(fig, 1.55, 0.55, 5.25, 1.02)
    order = ["initial six (Greedy)", "six added from history (Greedy)", "rest of library"]
    labels = ["initial six", "six added\nfrom history", "rest of library (%d)" % (D.role == order[2]).sum()]
    colors = [BLUE, "#56B4E9", GREY]
    ax.axvspan(100 * (E["m6"] + 0.20), 100, color=ORANGE, alpha=0.12, lw=0)
    ax.axvline(20, color=GREY, lw=0.7, ls=(0, (3, 2)))
    for y, (role, color) in enumerate(zip(order, colors)):
        d = D[D.role == role].sort_values("yield").reset_index(drop=True)
        jitter = 0.16 * ((np.arange(len(d)) % 3) - 1) if role == order[2] else np.zeros(len(d))
        ax.scatter(100 * d["yield"], y + jitter, s=14, color=color, alpha=0.75 if role == order[2] else 1,
                   lw=0, zorder=3)
    ax.scatter([100 * E["m6"]], [0], s=60, facecolor="none", edgecolor="black", lw=0.9, zorder=4)
    ax.scatter([100 * E["b"]], [1], s=95, marker=(5, 1, 0), color="black", lw=0.5, zorder=4)
    ax.set_yticks(range(3)); ax.set_yticklabels(labels, fontsize=TEXT)
    ax.set_ylim(2.55, -0.55); ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 20)); ax.tick_params(labelsize=TEXT, length=2)
    ax.set_xlabel("measured yield of this pair (%)", fontsize=TEXT, labelpad=2)
    beat = E["conditions_beating_hit_by_20"]
    ax.text(100 * (E["m6"] + 0.20) + 1.2, 0, "%d recipes more than\n20 points above the hit" % beat, ha="left", va="center",
            fontsize=8.5, color="#8A3B00")
    ax.text(19, -0.32, "hit threshold", ha="right", va="center", fontsize=8.5, color="#555555")
    ax.annotate("hit, %.1f%%: %s" % (100 * E["m6"], chem(", ".join(E["hit_recipe"].split(", ")[:2]))),
                xy=(100 * E["m6"], -0.12), xytext=(100 * E["m6"], -0.95), textcoords="data",
                ha="right", va="bottom", fontsize=8.5, annotation_clip=False,
                arrowprops=dict(arrowstyle="-", lw=0.6, color="black", shrinkA=0, shrinkB=2, relpos=(1, 0)))
    ax.annotate("library best, %.1f%%: %s" % (100 * E["b"], chem(", ".join(E["best_recipe"].split(", ")[:3]))),
                xy=(100 * E["b"], 0.85), xytext=(97, -0.95), textcoords="data",
                ha="right", va="bottom", fontsize=8.5, annotation_clip=False,
                arrowprops=dict(arrowstyle="-", lw=0.6, color="black", shrinkA=0, shrinkB=3, relpos=(1, 0)))


# Formula subscripts are mathtext; every redraw (including the figure audit's) must map it to Arial.
plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Arial",
                     "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold"})


def build():
    fig = plt.figure(figsize=(W, H))
    vectors = []
    panel_a(fig); panel_b(fig, vectors)
    fig._vector_panels = vectors
    fig.canvas.draw()
    return fig


def write(fig):
    import pymupdf
    png = io.BytesIO(); fig.savefig(png, format="png", dpi=300)
    fig.canvas.draw(); renderer = fig.canvas.get_renderer()
    placements = []
    for image, svg in fig._vector_panels:
        box = image.get_window_extent(renderer)
        placements.append((pymupdf.Rect(box.x0 / fig.dpi * 72, (fig.bbox.height - box.y1) / fig.dpi * 72,
                                        box.x1 / fig.dpi * 72, (fig.bbox.height - box.y0) / fig.dpi * 72), svg))
        image.set_visible(False)
    pdf = io.BytesIO()
    try:
        fig.savefig(pdf, format="pdf", dpi=300)
    finally:
        for image, _ in fig._vector_panels:
            image.set_visible(True)
    with pymupdf.open(stream=pdf.getvalue(), filetype="pdf") as document:
        for rect, vector_source in placements:
            if isinstance(vector_source, Path):
                with pymupdf.open(vector_source) as source:
                    assert not source[0].get_images(), "ChemDraw master contains a raster image"
                    document[0].show_pdf_page(rect, source, 0)
                continue
            with pymupdf.open(stream=vector_source, filetype="svg") as vector:
                with pymupdf.open(stream=vector.convert_to_pdf(), filetype="pdf") as source:
                    assert not source[0].get_images(), "RDKit structure SVG contains a raster image"
                    document[0].show_pdf_page(rect, source, 0)
        payload = document.tobytes(garbage=4, deflate=True)
    for target, data in ((FIG / (STEM + ".pdf"), payload), (FIG / (STEM + ".png"), png.getvalue())):
        tmp = target.with_name(target.name + ".tmp"); tmp.write_bytes(data); tmp.replace(target)


if __name__ == "__main__":
    f = build(); write(f); plt.close(f); print("wrote", STEM)
