"""Figures.
 Fig 1  is drawn by figure1_gc.py, which reuses the library constants below
 Supplementary Fig 5  the fourteen transformations as generic schemes (the author ChemDraw panel)
 Supplementary Fig 3  retention against panel size, with historical-mean and random baselines
 Supplementary Fig 4  panels scored out of sample: the sealed hold-out and the reserve
 Supplementary Fig 2  composition restrictions and how well the kit size is determined
 Supplementary Fig 6  levels of each condition variable that each array moves
"""
import io, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent)); import lib
from presentation_gc import ARRAY_LABELS, array_label, array_name
from figure_style_gc import LARGE, SMALL, STRUCTURE_SCALE, align_panels, configure_fonts, panel_title
HERE = Path(__file__).parent; OUT = HERE / "out"; FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
configure_fonts()
BLUE, GREY, ORANGE, DARK = "#0072B2", "#999999", "#D55E00", "#003f66"
KC = pd.read_csv(OUT / "review_curve24.csv"); SU = pd.read_csv(OUT / "kit_summary.csv")
KS = pd.read_csv(OUT / "kit_sealed.csv"); KR = pd.read_csv(OUT / "kit_reserve.csv")
CI = pd.read_csv(OUT / "kit_k90ci.csv"); LB = pd.read_csv(OUT / "kit_library.csv")
AX = pd.read_csv(OUT / "kit_axes_summary.csv")
COR = json.load(open(OUT / "kit_correlates.json"))
SHORT = {}  # Display names are owned by array_labels.csv.
LAB = ARRAY_LABELS
ORDER = SU.sort_values("ret_k6", ascending=False).screen.tolist()


TRNAME = {"Acylation": "Amide coupling", "Suzuki": "Suzuki–Miyaura coupling", "SNAr": "SNAr",
          "Buchwald-Hartwig": "Buchwald–Hartwig amination", "Kumada": "Kumada coupling", "Heck": "Heck reaction",
          "Reductive-amination": "Reductive amination", "Ullmann": "Ullmann C–O coupling", "Chan-Lam": "Chan–Lam coupling",
          "Sonogashira": "Sonogashira coupling", "Hiyama": "Hiyama coupling", "Grignard": "Grignard addition",
          "Esterification": "Esterification", "C-O competitive reactions": "C–O coupling"}
GROUP = {"precatalyst": "catalyst", "metal": "catalyst", "ligand": "ligand", "base": "base or acid",
         "acid": "base or acid", "solvent": "solvent"}
COLS = ["catalyst", "ligand", "base or acid", "other reagent", "solvent"]


def sfig5(write_outputs=True):
    """The fourteen transformations as generic schemes: the author ChemDraw panel, else fixed-scale RDKit."""
    AS = pd.read_csv(lib.DATASET / "arrays_summary.csv")
    from chemical_schemes_gc import DPI, export_mols, external_panel, scheme_drawing
    examples = pd.read_csv(HERE / "reaction_schemes_gc.csv")
    assert set(examples.transformation) == set(AS.transformation)
    replacement = external_panel()
    native_panel = replacement is not None
    if native_panel:
        # The ChemDraw panel at STRUCTURE_SCALE of its ACS 1996 size: atom labels print at LARGE.
        w, h = (STRUCTURE_SCALE * n / DPI for n in replacement.size)
        fig = plt.figure(figsize=(7.0, h + 0.1))
    else:
        fig = plt.figure(figsize=(7.0, 6.1))
    vector_panels = []
    if replacement is not None:
        ax = fig.add_axes([(7.0 - w) / 2 / 7.0, 0.05 / (h + 0.1), w / 7.0, h / (h + 0.1)]); ax.axis("off")
        image = ax.imshow(replacement, interpolation="nearest")
        external_pdf = FIG / "transformation_schemes.pdf"
        if external_pdf.exists():
            vector_panels.append((image, external_pdf))
            import xml.etree.ElementTree as ET
            native_xml = ET.parse(FIG / "transformation_schemes.cdxml").getroot()
            label_size = float(native_xml.attrib["LabelSize"])
            glyphs = [{"font_px": label_size * DPI / 72,
                       "text": "native atom labels", "role": "atom"}]
            for span in native_xml.iter("s"):
                size = float(span.get("size", label_size))
                script = int(span.get("face", "0")) & (32 | 64)
                glyphs.append({"font_px": size * (0.75 if script else 1) * DPI / 72,
                               "text": "".join(span.itertext()),
                               "role": "subscript or charge" if script else "atom"})
            ax._chemical_font_audit = [(image, glyphs)]
        fig._chemical_external_panel = True
    else:
        export_mols()
        fig._chemical_geometry_audit = []
        rows = list(examples.itertuples())
        rendered = [scheme_drawing(row) for row in rows]
        row_heights = [max(rendered[row][0].height, rendered[row + 7][0].height) / DPI
                       for row in range(7)]
        row_middles, row_top = [], 5.85
        for height in row_heights:
            row_middles.append(row_top - 0.155 - height / 2)
            row_top -= height + 0.165
        for i, r in enumerate(rows):
            col, row = i // 7, i % 7
            raster, lengths, glyphs, svg = rendered[i]
            # Axes sizes are physical raster sizes: every molecule is translated, never resized.
            width, height = raster.width / DPI, raster.height / DPI
            left = 0.025 + col * 0.495
            bottom = (row_middles[row] - height / 2) / fig.get_figheight()
            ax = fig.add_axes([left, bottom, width / fig.get_figwidth(), height / fig.get_figheight()])
            ax.axis("off")
            image = ax.imshow(raster, interpolation="nearest")
            vector_panels.append((image, svg))
            ax._chemical_font_audit = [(image, glyphs)]
            ax.set_title(TRNAME[r.transformation], fontsize=9.8, loc="left", pad=1.0,
                         y=1 + (row_heights[row] - height) / (2 * height))
            fig._chemical_geometry_audit.append({"transformation": r.transformation,
                                                 "bond_lengths_px": lengths, "glyphs": glyphs})
    # Word uses the raster asset; the standalone PDF replaces chemical images
    # with the corresponding RDKit paths, or preserves an external PDF panel.
    if write_outputs:
        import pymupdf
        png_buffer = io.BytesIO()
        fig.savefig(png_buffer, format="png", dpi=300)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        placements = []
        for image, vector in vector_panels:
            box = image.get_window_extent(renderer)
            rect = pymupdf.Rect(box.x0 / fig.dpi * 72,
                               (fig.bbox.height - box.y1) / fig.dpi * 72,
                               box.x1 / fig.dpi * 72,
                               (fig.bbox.height - box.y0) / fig.dpi * 72)
            placements.append((rect, vector))
            image.set_visible(False)
        pdf_buffer = io.BytesIO()
        try:
            fig.savefig(pdf_buffer, format="pdf", dpi=300)
        finally:
            for image, _ in vector_panels:
                image.set_visible(True)
        with pymupdf.open(stream=pdf_buffer.getvalue(), filetype="pdf") as document:
            for rect, vector in placements:
                if isinstance(vector, Path):
                    with pymupdf.open(vector) as source:
                        document[0].show_pdf_page(rect, source, 0)
                else:
                    with pymupdf.open(stream=vector, filetype="svg") as svg_document:
                        vector_pdf = svg_document.convert_to_pdf()
                    with pymupdf.open(stream=vector_pdf, filetype="pdf") as source:
                        if source[0].get_images():
                            raise RuntimeError("RDKit chemical SVG unexpectedly contains a raster image")
                        document[0].show_pdf_page(rect, source, 0)
            pdf_payload = document.tobytes(garbage=4, deflate=True, no_new_id=True)
        for target, payload in ((FIG / "SupplementaryFigure5_schemes.pdf", pdf_payload),
                                (FIG / "SupplementaryFigure5_schemes.png", png_buffer.getvalue())):
            if not payload:
                raise RuntimeError("Supplementary Figure 5 render produced an empty %s" % target.name)
            temporary = target.with_name(target.name + ".tmp")
            temporary.write_bytes(payload)
            temporary.replace(target)
        plt.close(fig)
    return fig
def sfig3():
    """Retention against panel size for every array, laid out to fit a landscape SI page."""
    fig, axes = plt.subplots(4, 5, figsize=(7.0, 6.3), sharex=True, sharey=True)
    for ax in axes.ravel()[len(ORDER):]:
        ax.axis("off")
    for ax, s in zip(axes.ravel(), ORDER):
        g = KC[KC.screen == s].sort_values("k")
        ax.fill_between(g.k, g.retention_meanyield, g.retention_greedy, color=BLUE, alpha=0.12, lw=0)
        ax.plot(g.k, g.retention_greedy, color=BLUE, lw=1.4)
        ax.plot(g.k, g.retention_meanyield, color=DARK, lw=0.9, ls=(0, (3, 2)))
        ax.plot(g.k, g.retention_random, color=GREY, lw=0.9, ls=":")
        ax.axhline(0.9, color=ORANGE, lw=0.6, ls=":")
        # Long labels break before their parenthetical so each title fits one narrow panel.
        title = LAB[s] if len(LAB[s]) < 17 else LAB[s].replace(" (", chr(10) + "(", 1)
        ax.set_title(title, fontsize=LARGE, loc="left", y=1.0)
        ax.set_ylim(0, 1.02); ax.set_yticks([0, 0.5, 1])
        ax.set_xticks([1, 12, 24]); ax.tick_params(length=2)
    for ax in axes[-1, :3]:
        ax.set_xlabel("panel size, k")
    for ax in axes[-2, 3:]:
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.set_xlabel("panel size, k")
    fig.supylabel("retention of mean best yield", fontsize=LARGE, x=0.012)
    h = [plt.Line2D([], [], color=BLUE, lw=1.4, label="history-ranked (greedy)"),
         plt.Line2D([], [], color=DARK, lw=0.9, ls=(0, (3, 2)), label="historical mean"),
         plt.Line2D([], [], color=GREY, lw=0.9, ls=":", label="random"),
         plt.Line2D([], [], color=ORANGE, lw=0.6, ls=":", label="90% of the best yield")]
    fig.tight_layout(rect=(0.02, 0, 1, 1), h_pad=0.9, w_pad=0.4)
    # Titles of one row share their top edge, whether they run to one line or two.
    fig.canvas.draw()
    for row in axes:
        shown = [ax for ax in row if ax.get_title(loc="left")]
        top = max(ax._left_title.get_window_extent().y1 for ax in shown)
        for ax in shown:
            ax._left_title.set_verticalalignment("top")
            ax._left_title.set_y(ax.transAxes.inverted().transform((0, top))[1])
    slot = axes[-1, 3].get_position().union([axes[-1, 3].get_position(), axes[-1, 4].get_position()])
    fig.legend(handles=h, loc="center", frameon=False,
               bbox_to_anchor=((slot.x0 + slot.x1) / 2, (slot.y0 + slot.y1) / 2))
    fig.savefig(FIG / "SupplementaryFigure3_curves.pdf"); fig.savefig(FIG / "SupplementaryFigure3_curves.png", dpi=300); plt.close(fig)


def sfig2():
    """Per-array composition restrictions and the original censored k90 intervals."""
    A = pd.read_csv(OUT / "review_axes_constraints.csv")
    variables = ["ligand", "precatalyst", "base", "solvent", "metal", "activator", "reductant"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 6.0), gridspec_kw={"width_ratios": [1.0, 1.0]})
    labels=[]
    for i,v in enumerate(variables):
        counts=[]
        for constraint, offset, color, marker in [("fixed", -.18, BLUE, "o"), ("diverse", .18, ORANGE, "D")]:
            d=A[(A.variable==v)&(A.constraint==constraint)&A.feasible].sort_values("screen")
            counts.append(len(d))
            yy=i+offset+np.linspace(-.12,.12,len(d))
            for y,r in zip(yy,d.itertuples()):
                ax.hlines(y,r.difference_yield_pp_lo,r.difference_yield_pp_hi,color=color,alpha=.55,lw=.9)
                ax.plot(r.difference_yield_pp,y,marker,color=color,ms=3.5,
                        mfc="white" if r.cv_folds_free_violates==0 else color,mew=.6)
        labels.append(f"{v} ({counts[0]}/{counts[1]})")
    ax.set_yticks(range(len(variables))); ax.set_yticklabels(labels)
    ax.invert_yaxis(); ax.axvline(0,color="black",lw=.6)
    ax.set_xlabel("yield percentage points lost")
    D=CI.sort_values(["k90","ret6"],ascending=[True,False]); yy=np.arange(len(D))[::-1]
    cn=D.label.str.startswith("C–N").values
    bx.hlines(yy,D.k90_lo,D.k90_hi,color=GREY,lw=1.4)
    for i, row in enumerate(D.itertuples()):
        color, marker = (ORANGE, "s") if cn[i] else (BLUE, "o")
        bx.plot(row.k90, yy[i], marker, color=color, ms=4,
                mfc="white" if row.k90 > 12 else color, mew=0.8)
    bx.set_yticks(yy); bx.set_yticklabels([array_label(s) for s in D.screen])
    bx.set_xticks([2,4,6,8,10,13]); bx.set_xticklabels(["2","4","6","8","10",">12"])
    bx.set_xlim(1,13.8); bx.set_xlabel("conditions for 90% (search ≤12)")
    h=[plt.Line2D([],[],color=BLUE,marker="o",ls="none",ms=4,label="a: one level fixed"),
       plt.Line2D([],[],color=ORANGE,marker="D",ls="none",ms=4,label="a: six distinct levels"),
       plt.Line2D([],[],color=GREY,marker="o",mfc="white",ls="none",ms=4,label="a: unconstrained panel already complies"),
       plt.Line2D([],[],color=ORANGE,marker="s",ls="none",ms=4,label="b: five Buchwald–Hartwig campaigns"),
       plt.Line2D([],[],color=BLUE,marker="o",mfc="white",ls="none",ms=4,label="b: >12 is right-censored")]
    fig.legend(handles=h,loc="lower center",ncol=2,frameon=False)
    fig.tight_layout(rect=(0,.14,1,.94),w_pad=2.0)
    panel_title(fig, ax, "a", "Composition restrictions (fixed/diverse n)", row=0, col=0)
    panel_title(fig, bx, "b", "Panel size and uncertainty", row=0, col=1)
    align_panels(fig)
    fig.savefig(FIG/"SupplementaryFigure2_axes_and_size.pdf"); fig.savefig(FIG/"SupplementaryFigure2_axes_and_size.png",dpi=300); plt.close(fig)


def sfig4():
    """All held-out arrays, paired CV references and conditional pair-bootstrap intervals."""
    D=pd.read_csv(OUT/"review_generalization_heldout_cv_per_array.csv")
    D=D[D.k==6].sort_values("retention_heldout").reset_index(drop=True)
    fig,ax=plt.subplots(figsize=(7.0,5.8))
    for y,r in enumerate(D.itertuples()):
        color=BLUE if r.set=="sealed" else ORANGE
        marker="o" if r.set=="sealed" else "s"
        ax.hlines(y,r.pair_bootstrap_ci_lo,r.pair_bootstrap_ci_hi,color=color,lw=1.1)
        ax.plot(r.retention_heldout,y,marker,color=color,ms=4)
        ax.plot(r.retention_cv,y,"x",color=DARK,ms=4,mew=1)
        if r.n_test < 8:
            ax.text(1.035, y, "*", color=ORANGE, fontsize=LARGE, ha="left", va="center",
                    transform=ax.get_yaxis_transform(), clip_on=False)
    ax.set_yticks(range(len(D)))
    ax.set_yticklabels([f"{array_label(r.screen)}, n = {r.n_test}" for r in D.itertuples()])
    ax.set_xlim(0,1.03); ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xlabel("retention of mean best yield (k = 6)")
    h=[plt.Line2D([],[],color=BLUE,marker="o",ls="none",label="sealed pairs"),
       plt.Line2D([],[],color=ORANGE,marker="s",ls="none",label="reserve pairs"),
       plt.Line2D([],[],color=DARK,marker="x",ls="none",label="same-array pair CV"),
       plt.Line2D([],[],color=ORANGE,marker="*",ls="none",label="* n < 8; small sample, not significance")]
    fig.legend(handles=h,loc="lower center",ncol=2,frameon=False)
    fig.tight_layout(rect=(0,0.08,.97,1))
    fig.savefig(FIG/"SupplementaryFigure4_outofsample.pdf"); fig.savefig(FIG/"SupplementaryFigure4_outofsample.png",dpi=300); plt.close(fig)


def sfig6():
    """Levels of each condition variable that each array moves (formerly Figure 1c)."""
    L = LB.assign(role=LB.variable.map(lambda v: GROUP.get(v, "other reagent")))
    P = L.pivot_table(index="label", columns="role", values="levels", aggfunc="max", fill_value=0)
    P = P.reindex(columns=COLS, fill_value=0).reindex([SU[SU.screen == s].label.iloc[0] for s in ORDER])
    fig = plt.figure(figsize=(7.0, 3.6))
    bx = fig.add_axes([2.05 / 7.0, 0.35 / 3.6, 4.75 / 7.0, 3.1 / 3.6])
    Z = P.values.astype(float); Zm = np.where(Z == 0, np.nan, Z); top = np.log10(np.nanmax(Zm))
    bx.imshow(np.log10(Zm), cmap="Blues", vmin=0, vmax=top, aspect="auto")
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            v = int(Z[i, j])
            if v == 0:
                bx.text(j, i, "–", ha="center", va="center", fontsize=SMALL, color=GREY)
            else:
                bx.text(j, i, str(v), ha="center", va="center", fontsize=SMALL,
                        color="white" if np.log10(v) > 0.6 * top else "black")
    bx.set_xticks(range(len(COLS)))
    bx.set_xticklabels(["catalyst", "ligand", "base or acid", "other reagent", "solvent"])
    bx.set_yticks(range(len(P))); bx.set_yticklabels([array_label(s) for s in ORDER])
    bx.tick_params(length=0)
    for sp in bx.spines.values():
        sp.set_visible(False)
    fig.savefig(FIG / "SupplementaryFigure6_libraries.pdf"); fig.savefig(FIG / "SupplementaryFigure6_libraries.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    sfig2(); sfig3(); sfig4(); sfig5(); sfig6(); print("figures written to", FIG)
