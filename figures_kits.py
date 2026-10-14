"""Figures.
 Fig 1  is drawn by figure1_gc.py, which reuses the library constants below
 Supplementary Fig 5  the fourteen transformations as generic schemes (the author ChemDraw panel)
 Fig 2  what a screening hit leaves behind: severe shortfalls among hits, what six more conditions recover
 Supplementary Fig 3  retention against panel size, with historical-mean and random baselines
 Supplementary Fig 4  panels scored out of sample: the sealed hold-out and the reserve
 Supplementary Fig 2  composition restrictions and how well the kit size is determined
 Supplementary Fig 6  levels of each condition variable that each array moves
 Fig 3  yield history, not reagent spread: median retention curves and ligand-spread rules
"""
import io, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent)); import lib
from presentation_gc import ARRAY_LABELS, array_label, array_name
from figure_style_gc import configure_fonts, panel_title
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
    fig = plt.figure(figsize=(7.0, 5.35) if native_panel else (7.0, 6.1))
    vector_panels = []
    if replacement is not None:
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        image = ax.imshow(replacement)
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
            pdf_payload = document.tobytes(garbage=4, deflate=True)
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
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.set_ylim(0, 1.02); ax.set_yticks([0, 0.5, 1])
        ax.set_xticks([1, 12, 24]); ax.tick_params(length=2, labelsize=8.5)
    for ax in axes[-1, :3]:
        ax.set_xlabel("panel size, k", fontsize=8.5)
    for ax in axes[-2, 3:]:
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.set_xlabel("panel size, k", fontsize=8.5)
    fig.supylabel("retention of mean best yield", fontsize=8.5, x=0.012)
    h = [plt.Line2D([], [], color=BLUE, lw=1.4, label="Greedy"),
         plt.Line2D([], [], color=DARK, lw=0.9, ls=(0, (3, 2)), label="Historical mean"),
         plt.Line2D([], [], color=GREY, lw=0.9, ls=":", label="Random"),
         plt.Line2D([], [], color=ORANGE, lw=0.6, ls=":", label="90% of the best yield")]
    fig.tight_layout(rect=(0.02, 0, 1, 1), h_pad=0.9, w_pad=0.4)
    slot = axes[-1, 3].get_position().union([axes[-1, 3].get_position(), axes[-1, 4].get_position()])
    fig.legend(handles=h, loc="center", fontsize=8.5, frameon=False,
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
    ax.set_yticks(range(len(variables))); ax.set_yticklabels(labels,fontsize=8)
    ax.invert_yaxis(); ax.axvline(0,color="black",lw=.6)
    ax.set_xlabel("yield percentage points lost",fontsize=8)
    D=CI.sort_values(["k90","ret6"],ascending=[True,False]); yy=np.arange(len(D))[::-1]
    cn=D.label.str.startswith("C–N").values
    bx.hlines(yy,D.k90_lo,D.k90_hi,color=GREY,lw=1.4)
    for i, row in enumerate(D.itertuples()):
        color, marker = (ORANGE, "s") if cn[i] else (BLUE, "o")
        bx.plot(row.k90, yy[i], marker, color=color, ms=4,
                mfc="white" if row.k90 > 12 else color, mew=0.8)
    bx.set_yticks(yy); bx.set_yticklabels([array_label(s) for s in D.screen],fontsize=8)
    bx.set_xticks([2,4,6,8,10,13]); bx.set_xticklabels(["2","4","6","8","10",">12"])
    bx.set_xlim(1,13.8); bx.set_xlabel("conditions for 90% (search ≤12)",fontsize=8)
    h=[plt.Line2D([],[],color=BLUE,marker="o",ls="none",ms=4,label="a: one level fixed"),
       plt.Line2D([],[],color=ORANGE,marker="D",ls="none",ms=4,label="a: six distinct levels"),
       plt.Line2D([],[],color=GREY,marker="o",mfc="white",ls="none",ms=4,label="a: unconstrained panel already complies"),
       plt.Line2D([],[],color=ORANGE,marker="s",ls="none",ms=4,label="b: five Buchwald–Hartwig campaigns"),
       plt.Line2D([],[],color=BLUE,marker="o",mfc="white",ls="none",ms=4,label="b: >12 is right-censored")]
    fig.legend(handles=h,loc="lower center",ncol=2,fontsize=8,frameon=False)
    fig.tight_layout(rect=(0,.18,1,.94),w_pad=2.0)
    panel_title(fig, ax, "a", "Composition restrictions\n(fixed/diverse n)", fontsize=8, letter_size=9)
    panel_title(fig, bx, "b", "Panel size and uncertainty", fontsize=8, letter_size=9)
    fig.savefig(FIG/"SupplementaryFigure2_axes_and_size.pdf"); fig.savefig(FIG/"SupplementaryFigure2_axes_and_size.png",dpi=300); plt.close(fig)


def fig2():
    """Severe shortfalls among screening hits, the gap six more conditions recover, and the yield ceiling."""
    S = json.load(open(OUT / "posthit_summary.json")); B = pd.read_csv(OUT / "posthit_yield_bands.csv")
    an, nb, ho, ex = S["analysed"], S["molecule_node_intervals"], S["holdout"]["all"], S["extension_and_support"]
    # Word embeds this 7-inch source at 6 inches; 9.5 pt remains above 8 pt after scaling.
    fig = plt.figure(figsize=(7.0, 6.0))
    # Top row: a (rates) and b (yield bands); bottom row: c (recovery bars, long labels) and d (per-hit gain).
    top = fig.add_gridspec(1, 2, left=0.10, right=0.98, top=0.94, bottom=0.62, wspace=0.45)
    bottom = fig.add_gridspec(1, 1, left=0.30, right=0.62, top=0.455, bottom=0.095)
    right = fig.add_gridspec(1, 1, left=0.78, right=0.98, top=0.455, bottom=0.095)
    ax, cx, bx = fig.add_subplot(top[0, 0]), fig.add_subplot(top[0, 1]), fig.add_subplot(bottom[0, 0])
    dx = fig.add_subplot(right[0, 0])
    for panel in (ax, cx, bx, dx):
        panel.tick_params(labelsize=9.5)
    # a: hits below 80%, the only hits that can fall more than twenty points short.
    bc = S["below_ceiling"]
    ks = [6, 12, 24]
    rate = [100 * bc["severe_rate_k%d" % k] for k in ks]
    lo = [100 * bc["severe_rate_k%d_lo" % k] for k in ks]
    hi = [100 * bc["severe_rate_k%d_hi" % k] for k in ks]
    ax.vlines(ks, lo, hi, color=BLUE, lw=1.2)
    ax.plot(ks, rate, "o-", color=BLUE, ms=4.5, lw=1.2, label="analysed (%d hits)" % bc["pairs"])
    ax.plot([6, 12], [100 * ho["severe_rate_below_ceiling_k6"], 100 * ho["severe_rate_below_ceiling_k12"]], "s--",
            color=ORANGE, ms=4.5, lw=1, label="held-out (%d hits)" % ho["hit6_below_ceiling"])
    offsets = {6: ((-7, -3), "right"), 12: ((6, 4), "left"), 24: ((-6, 12), "right")}
    for k, r in zip(ks, rate):
        ax.annotate("%.1f%%" % r, (k, r), xytext=offsets[k][0], textcoords="offset points",
                    ha=offsets[k][1], fontsize=9.5, color=BLUE)
    ax.set_xticks(ks); ax.set_xlim(1.5, 26); ax.set_ylim(0, 30)
    ax.set_xlabel("conditions screened", fontsize=10)
    ax.set_ylabel("hits below 80% that are >20\npoints below the library best (%)", fontsize=10)
    ax.legend(fontsize=9.5, frameon=False, loc="upper right")
    panel_title(fig, ax, "a", "Hits far from the best recipe", fontsize=10)
    # The analysed cohort has molecular-reweighting intervals; held-out point
    # estimates are shown without intervals because the frozen scores do not
    # support the same uncertainty calculation.
    rows = [("analysed: Greedy", 100 * an["gap_recovery_k6_to_12"],
             100 * nb["gap_recovery_k6_to_12_lo"], 100 * nb["gap_recovery_k6_to_12_hi"], BLUE, ""),
            ("analysed: Historical mean", 100 * ex["gap_recovery_meanyield_plus6"],
             100 * nb["meanyield_recovery_lo"], 100 * nb["meanyield_recovery_hi"], "#56B4E9", ""),
            ("analysed: Random", 100 * ex["gap_recovery_random_plus6"], 100 * nb["random_recovery_lo"],
             100 * nb["random_recovery_hi"], GREY, ""),
            ("held-out: Greedy", 100 * ho["gap_recovery_k6_to_12"], None, None, BLUE, "///"),
            ("held-out: Historical mean", 100 * ho["gap_recovery_meanyield_plus6"], None, None, "#56B4E9", "///"),
            ("held-out: Random", 100 * ho["gap_recovery_random_plus6"], None, None, GREY, "///")]
    for y, (lab, v, l, h, c, hatch) in enumerate(rows):
        # Opaque colors preserve hatch contrast in PDF tiling patterns.
        rgb = np.asarray(matplotlib.colors.to_rgb(c))
        weight = 0.28 if hatch else 0.9
        face = tuple(1 - weight + weight * rgb)
        edge = tuple(0.4 + 0.6 * rgb) if hatch else face
        bx.barh(y, v, color=face, edgecolor=edge, hatch=hatch, height=0.6)
        if l is not None:
            bx.hlines(y, l, h, color="black", lw=0.9)
        bx.text((h if h is not None else v) + 1.5, y, "%.0f%%" % v, va="center", fontsize=9.5)
    bx.set_yticks(range(len(rows))); bx.set_yticklabels([r[0] for r in rows], fontsize=9.5); bx.invert_yaxis()
    bx.set_xlim(0, 62); bx.set_xlabel("share of the hits' yield gap recovered (%)", fontsize=10)
    panel_title(fig, bx, "c", "What six further conditions recover", fontsize=10)
    # d: what each hit gains from the six further Greedy conditions, hits below 80% against severe shortfalls.
    P = pd.read_csv(OUT / "posthit_pair_support.csv")
    gain = 100 * (P.m12 - P.m6)
    severe = (P.b - P.m6) > 0.20
    below = P.m6 < 0.80
    groups = [gain[below].values, gain[severe].values]
    assert len(P) == an["pairs"] and int(severe.sum()) == ex["severe_k6"] and int(below.sum()) == bc["pairs"]
    assert np.isclose(np.median(groups[0]), bc["gain_points_median"]) and \
        np.isclose(np.median(groups[1]), ex["severe_gain_points_median"])
    box = dx.boxplot(groups, vert=False, whis=(5, 95), showfliers=False, widths=0.5, patch_artist=True,
                     medianprops=dict(color="black", lw=1.2), whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
    for patch, color in zip(box["boxes"], ("#56B4E9", BLUE)):
        patch.set(facecolor=color, edgecolor="black", lw=0.8, alpha=0.85)
    for y, values in enumerate(groups, 1):
        dx.text(np.percentile(values, 95) + 2, y, "median %s" % ("%+.0f" % np.median(values) if np.median(values) else "0"), va="center", fontsize=9.5)
    dx.set_yticks([1, 2])
    dx.set_yticklabels(["hits below\n80%% (%d)" % bc["pairs"], "severe\n(%d)" % ex["severe_k6"]], fontsize=9.5)
    dx.invert_yaxis(); dx.set_xlim(0, 80); dx.set_xticks([0, 20, 40, 60, 80])
    dx.set_xlabel("yield points gained", fontsize=10)
    panel_title(fig, dx, "d", "Gain per hit, Greedy", fontsize=10)
    labels = ["%d–%d%%" % (round(100 * r.m6_lower), round(100 * r.m6_upper)) for r in B.itertuples()]
    for i, r in enumerate(B.itertuples()):
        if r.ceiling_permits_severe:
            v = 100 * r.severe_rate_k6
            cx.bar(i, v, color=BLUE, width=0.65)
            cx.text(i, v + 0.8, "%d/%d" % (r.severe_k6, r.pairs), ha="center", fontsize=9.5)
        else:
            cx.text(i, 3.5, "%d/%d\nceiling-limited" % (r.severe_k6, r.pairs),
                    ha="center", va="center", fontsize=9.5, color="#555555")
    cx.set_xticks(range(len(labels))); cx.set_xticklabels(labels, fontsize=9.5); cx.set_ylim(0, 30)
    cx.set_xlim(-0.5, len(labels) - 0.5)
    cx.set_xlabel("best yield among the six screened", fontsize=10)
    cx.set_ylabel("hits >20 points short (%)", fontsize=10)
    panel_title(fig, cx, "b", "Shortfall by the hit's yield", fontsize=10)
    fig.savefig(FIG / "Figure2_posthit.pdf"); fig.savefig(FIG / "Figure2_posthit.png", dpi=300); plt.close(fig)


def fig3():
    """Yield history, not reagent spread: (a) median retention against panel size over the eighteen arrays;
    (b) per array, six ligand levels chosen without yields against random, historical-mean and greedy panels."""
    R = pd.read_csv(OUT / "review_axes_recipes.csv")
    fig = plt.figure(figsize=(7.0, 4.1))
    ax = fig.add_axes([0.105, 0.265, 0.31, 0.635])
    bx = fig.add_axes([0.71, 0.265, 0.27, 0.635])
    for panel in (ax, bx):
        panel.tick_params(labelsize=9.5, length=2)
    methods = [("greedy", "Greedy", BLUE, "-", 1.6), ("meanyield", "Historical mean", DARK, (0, (3, 2)), 1.1),
               ("random", "Random", GREY, ":", 1.3)]
    med = KC.groupby("k")[["retention_" + m for m, *_ in methods]].median()
    q1 = KC.groupby("k").retention_greedy.quantile(0.25); q3 = KC.groupby("k").retention_greedy.quantile(0.75)
    ax.fill_between(med.index, q1, q3, color=BLUE, alpha=0.12, lw=0)
    for m, label, color, ls, lw in methods:
        ax.plot(med.index, med["retention_" + m], color=color, ls=ls, lw=lw, label=label)
    ax.axvline(6, color="#BBBBBB", lw=0.6, zorder=0)
    # Values at six conditions, set clear of the curves: above, below, and below the random curve.
    for (m, label, color, ls, lw), y in zip(methods, (0.97, 0.70, 0.47)):
        v = med.loc[6, "retention_" + m]
        ax.plot(6, v, "o", color=color, ms=3.5, zorder=4)
        ax.text(6.4, y, "%.0f%%" % (100 * v), fontsize=9.5, color=color, va="center")
    ax.set_xlim(1, 24); ax.set_ylim(0, 1.0); ax.set_xticks([1, 6, 12, 18, 24])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["0", "25", "50", "75", "100"])
    ax.set_xlabel("conditions in the panel", fontsize=10)
    ax.set_ylabel("share of library-best\nyield found (%)", fontsize=10)
    ax.legend(fontsize=9.5, frameon=False, loc="lower right")
    panel_title(fig, ax, "a", "Median of 18 libraries", fontsize=10)
    rules = [("ligand_uniform6", "six ligand levels", ORANGE, "D"), ("random6", "Random", GREY, "o"),
             ("topmean6", "Historical mean", DARK, "s"), ("greedy6", "Greedy", BLUE, "o")]
    P = R.pivot_table(index="screen", columns="method", values="retention")
    P = P.sort_values("greedy6")
    assert len(P) == 12 and P[[r[0] for r in rules]].notna().all().all()
    for y, (screen, row) in enumerate(P.iterrows()):
        bx.hlines(y, row[[r[0] for r in rules]].min(), row[[r[0] for r in rules]].max(), color="#DDDDDD", lw=1.0, zorder=1)
        for method, label, color, marker in rules:
            bx.plot(row[method], y, marker, color=color, ms=4.6, zorder=3,
                    mfc="white" if method == "topmean6" else color, mew=1.0)
    bx.set_yticks(range(len(P))); bx.set_yticklabels([array_label(s) for s in P.index], fontsize=9.5)
    bx.set_xlim(0, 1.0); bx.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); bx.set_xticklabels(["0", "25", "50", "75", "100"])
    bx.set_xlabel("share of library-best\nyield found by six (%)", fontsize=10)
    handles = [plt.Line2D([], [], color=c, marker=mk, ls="none", ms=4.6, mfc="white" if m == "topmean6" else c,
                          label="%s (median %.0f%%)" % (lab, 100 * P[m].median())) for m, lab, c, mk in rules[::-1]]
    fig.legend(handles=handles, fontsize=9.5, frameon=False, loc="lower center", bbox_to_anchor=(0.62, 0.0),
               ncol=2, columnspacing=1.2, handletextpad=0.3)
    panel_title(fig, bx, "b", "Ligand spread, no yields", fontsize=10)
    fig.savefig(FIG / "Figure3_rules.pdf"); fig.savefig(FIG / "Figure3_rules.png", dpi=300); plt.close(fig)


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
            ax.text(1.035, y, "*", color=ORANGE, fontsize=10.5, ha="left", va="center",
                    transform=ax.get_yaxis_transform(), clip_on=False)
    ax.set_yticks(range(len(D)))
    ax.set_yticklabels([f"{array_label(r.screen)}, n = {r.n_test}" for r in D.itertuples()],fontsize=10.5)
    ax.tick_params(labelsize=10.5)
    ax.set_xlim(0,1.03); ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xlabel("retention of mean best yield (k = 6)",fontsize=10.5)
    h=[plt.Line2D([],[],color=BLUE,marker="o",ls="none",label="sealed pairs"),
       plt.Line2D([],[],color=ORANGE,marker="s",ls="none",label="reserve pairs"),
       plt.Line2D([],[],color=DARK,marker="x",ls="none",label="same-array pair CV"),
       plt.Line2D([],[],color=ORANGE,marker="*",ls="none",label="* n < 8; small sample, not significance")]
    fig.legend(handles=h,loc="lower center",ncol=2,fontsize=9.5,frameon=False)
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
                bx.text(j, i, "–", ha="center", va="center", fontsize=8.5, color=GREY)
            else:
                bx.text(j, i, str(v), ha="center", va="center", fontsize=8.5,
                        color="white" if np.log10(v) > 0.6 * top else "black")
    bx.set_xticks(range(len(COLS)))
    bx.set_xticklabels(["catalyst", "ligand", "base or acid", "other reagent", "solvent"], fontsize=9)
    bx.set_yticks(range(len(P))); bx.set_yticklabels([array_label(s) for s in ORDER], fontsize=8.5)
    bx.tick_params(length=0)
    for sp in bx.spines.values():
        sp.set_visible(False)
    fig.savefig(FIG / "SupplementaryFigure6_libraries.pdf"); fig.savefig(FIG / "SupplementaryFigure6_libraries.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    fig2(); fig3(); sfig2(); sfig3(); sfig4(); sfig5(); sfig6(); print("figures written to", FIG)
