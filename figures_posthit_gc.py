"""Figures 2/3: measured hit-level shortfalls and recovery on the fixed hit cohort."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch, Circle
import numpy as np
import pandas as pd
import lib
from figure_style_gc import configure_fonts, align_panels, panel_title, add_panel, SMALL
from presentation_gc import array_label

HERE = Path(__file__).resolve().parent
OUT, FIG = HERE / "out", HERE / "figures"
BLUE, LIGHT, ORANGE, GREY = "#0072B2", "#56B4E9", "#D55E00", "#999999"
STEMS = {2: "Figure2_shortfalls", 3: "Figure3_recovery"}
configure_fonts()


def inputs():
    return (pd.read_csv(OUT / "posthit_pair_support.csv"),
            json.loads((OUT / "posthit_summary.json").read_text(encoding="utf-8")))


def library_intervals(pairs):
    """Conditional 95% intervals, sharing exponential weights across molecule identities."""
    rows = []
    for screen, group in pairs[pairs.m6 < .8].groupby("screen", sort=True):
        partners = group.subpair.str.split(" >> ", expand=True)
        molecules = sorted(set(partners[0]) | set(partners[1]))
        index = {name: i for i, name in enumerate(molecules)}
        a, b = [np.array([index[v] for v in partners[k]]) for k in (0, 1)]
        seed = int.from_bytes(hashlib.sha256(screen.encode()).digest()[:4], "little")
        weights = np.random.default_rng(seed).exponential(size=(10000, len(molecules)))
        w = weights[:, a] * np.where(a == b, 1, weights[:, b])
        severe = (group.b - group.m6 > .2).to_numpy()
        rates = (w @ severe) / w.sum(axis=1)
        lo, hi = np.percentile(rates, [2.5, 97.5])
        rows.append(dict(screen=screen, label=array_label(screen), hits_below80=len(group),
                         severe=int(severe.sum()), rate=float(severe.mean()), lo=lo, hi=hi))
    return pd.DataFrame(rows).sort_values(["rate", "screen"], ascending=[False, True])


def outcomes(pairs):
    severe = pairs[pairs.b - pairs.m6 > .2].copy()
    full = np.isclose(severe.m12, severe.b, atol=1e-12, rtol=0)
    improved = severe.m12 > severe.m6 + 1e-12
    within = severe.b - severe.m12 <= .2
    severe["group"] = np.select([full, within, improved], [0, 1, 2], default=3)
    severe["gap"] = severe.b - severe.m6
    result = severe.sort_values(["group", "gap", "screen", "subpair"],
                                ascending=[True, False, True, True], kind="stable")
    assert result.groupby("group").size().tolist() == [44, 61, 26, 53]
    return result


def holdout_intervals():
    """Conditional intervals for the fixed held-out hits, sharing molecule weights."""
    from posthit_gc import holdout_extensions
    pairs = holdout_extensions()
    pairs = pairs[pairs.m6 >= .2].sort_values(["screen", "set", "subpair"]).reset_index(drop=True)
    partners = pairs.subpair.str.split(" >> ", expand=True)
    molecules = sorted(set(partners[0]) | set(partners[1]))
    index = {name: i for i, name in enumerate(molecules)}
    a, b = [np.array([index[v] for v in partners[k]]) for k in (0, 1)]
    weights = np.random.default_rng(20261020).exponential(size=(10000, len(molecules)))
    w = weights[:, a] * np.where(a == b, 1, weights[:, b])
    denominator = w @ (pairs.full - pairs.m6).to_numpy()
    result = {}
    for method, field in (("history-ranked", "m12"), ("historical mean", "m6_meanyield_plus6"),
                          ("random", "m6_random_plus6")):
        gain = (pairs[field] - pairs.m6).to_numpy()
        result[method] = np.percentile((w @ gain) / denominator, [2.5, 97.5])
    return result, pairs


def heatmap(fig, ax, example, data):
    from figure1_gc import CMAP, NOT_IN_LIBRARY, CELL
    raw, _ = lib.load("analysis")
    library = set(raw[raw.screen == example["screen"]].condition_id)
    yields = dict(zip(data.condition_id, data["yield"]))
    roles = dict(zip(data.condition_id, data.role))
    ax.set(xlim=(0, 12), ylim=(8, 0), aspect="equal", xticks=[], yticks=[])
    for spine in ax.spines.values():
        spine.set_visible(False)
    marked, labelled, better_markers = [], [], []
    for k in range(1, 97):
        condition = "C%d" % k
        row, col = divmod(k - 1, 12)
        x, y = col + .5, row + .5
        if condition not in library:
            ax.plot(x, y, "x", color=NOT_IN_LIBRARY, ms=5, mew=.8)
            continue
        if condition not in yields:
            ax.add_patch(Rectangle((col + .06, row + .06), .88, .88,
                                  facecolor=CELL, edgecolor="#999999", lw=.5, hatch="///"))
            continue
        value = yields[condition]
        role = roles[condition]
        edge = BLUE if role.startswith("initial six") else LIGHT if role.startswith("six added") else "white"
        ax.add_patch(Rectangle((col + .04, row + .04), .92, .92, facecolor=CMAP(value),
                              edgecolor=edge, lw=1.3 if edge != "white" else .5))
        better = value - example["m6"] > .2
        if better:
            marker = Circle((col + .78, row + .22), .15, facecolor=ORANGE,
                            edgecolor="white", lw=1.0, zorder=5)
            ax.add_patch(marker)
            better_markers.append(marker)
            marked.append(condition)
        if edge != "white" and not better:
            color = "white" if value > .65 else "#222222"
            ax.text(x, y, str(k), ha="center", va="center", fontsize=SMALL, color=color)
            labelled.append(condition)
        if condition in (example["hit_condition"], example["best_condition"]):
            marker = "o" if condition == example["hit_condition"] else "*"
            mx, my = (x, y) if marker == "o" else (x - .22, y + .15)
            ax.plot(mx, my, marker, mfc="none" if marker == "o" else "black", mec="black",
                    ms=11 if marker == "o" else 4.5, mew=1)
    assert len(marked) == example["conditions_beating_hit_by_20"] if "conditions_beating_hit_by_20" in example else len(marked) == 7
    legend = [Patch(facecolor="none", edgecolor=BLUE, lw=1.2, label="first six"),
              Patch(facecolor="none", edgecolor=LIGHT, lw=1.2, label="six added"),
              Line2D([], [], marker="o", mfc=ORANGE, mec="white", mew=1,
                     linestyle="none", ms=6, label="%d >20 points better" % len(marked))]
    ax.legend(handles=legend, frameon=False, loc="upper left", borderaxespad=0,
              bbox_to_anchor=(3.14/6.5, .35/6.5), bbox_transform=fig.transFigure,
              ncol=2, handlelength=1.1, handleheight=.8, columnspacing=.8,
              labelspacing=.3)
    fig._heatmap_labelled = labelled
    fig._heatmap_better_markers = better_markers
    return marked


def compact_example(fig, example):
    """Reflow the native ChemDraw vectors without changing any molecular scale."""
    import copy
    import xml.etree.ElementTree as ET
    import pymupdf
    from PIL import Image
    from figure1_gc import scheme_raster
    from figure_style_gc import STRUCTURE_SCALE
    from chemical_schemes_gc import DPI

    _, _, _, glyphs, master = scheme_raster(example, fig)
    cdxml = ET.parse(FIG / "example_pair_scheme.cdxml").getroot()
    with pymupdf.open(master) as document:
        page = document[0]
        drawings = page.get_drawings()
        arrow_clip = (drawings[-2]["rect"] | drawings[-1]["rect"]) + (-1, -1, 1, 1)
        arrow_cdxml = next(cdxml.iter("graphic"))
        arrow_bounds = list(map(float, arrow_cdxml.get("BoundingBox").split()))
        origin = pymupdf.Point(max(arrow_bounds[0::2])-drawings[-1]["rect"].x1,
                               arrow_bounds[1]-(arrow_clip.y0+arrow_clip.y1)/2)
        clips = []
        for fragment in cdxml.iter("fragment"):
            box = pymupdf.Rect(list(map(float, fragment.get("BoundingBox").split())))
            box = pymupdf.Rect(box.x0-origin.x, box.y0-origin.y,
                              box.x1-origin.x, box.y1-origin.y)
            clips.append((box + (-1, -2, 1, 2)) & page.rect)
        assert len(clips) == 3
        plus = next(span for block in page.get_text("dict")["blocks"] if "lines" in block
                    for line in block["lines"] for span in line["spans"] if span["text"] == "+")
        plus_clip = pymupdf.Rect(plus["bbox"]) + (-1, -1, 1, 1)
        source_bytes = document.tobytes()
    left = max(box.width for box in clips[:2])
    height = clips[0].height + plus_clip.height + clips[1].height + 4
    width = left + 16 + arrow_clip.width + clips[2].width
    ns = "http://www.w3.org/2000/svg"
    ET.register_namespace("", ns)
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    svg = ET.Element("{%s}svg" % ns, {"width": "%.8fpt" % width,
        "height": "%.8fpt" % height, "viewBox": "0 0 %.8f %.8f" % (width, height)})
    definitions = ET.SubElement(svg, "{%s}defs" % ns)
    placements = [(clips[0], (left-clips[0].width)/2, 0),
                  (plus_clip, (left-plus_clip.width)/2, clips[0].height+2),
                  (clips[1], (left-clips[1].width)/2, height-clips[1].height),
                  (arrow_clip, left+8, (height-arrow_clip.height)/2),
                  (clips[2], left+16+arrow_clip.width, (height-clips[2].height)/2)]
    for index, (clip, x, y) in enumerate(placements):
        # Remove off-fragment PDF content before SVG conversion: MuPDF's SVG reader
        # ignores clipPath, so retaining the complete page would duplicate molecules.
        with pymupdf.open(stream=source_bytes, filetype="pdf") as part:
            page = part[0]
            for outside in (pymupdf.Rect(0, 0, clip.x0, page.rect.height),
                            pymupdf.Rect(clip.x1, 0, page.rect.width, page.rect.height),
                            pymupdf.Rect(0, 0, page.rect.width, clip.y0),
                            pymupdf.Rect(0, clip.y1, page.rect.width, page.rect.height)):
                if not outside.is_empty:
                    page.add_redact_annot(outside, fill=False)
            page.apply_redactions(images=0, graphics=2, text=0)
            source = ET.fromstring(page.get_svg_image(text_as_path=True))
        prefix = "native_%d_" % index
        for element in source.iter():
            if "id" in element.attrib:
                element.set("id", prefix+element.get("id"))
            href = "{http://www.w3.org/1999/xlink}href"
            if href in element.attrib:
                element.set(href, "#"+prefix+element.get(href).lstrip("#"))
        for child in source:
            if child.tag == "{%s}defs" % ns:
                for definition in child:
                    definitions.append(copy.deepcopy(definition))
        translated = ET.SubElement(svg, "{%s}g" % ns,
                                   {"transform": "translate(%.8f %.8f)" % (x-clip.x0, y-clip.y0)})
        for child in source:
            if child.tag != "{%s}defs" % ns:
                translated.append(copy.deepcopy(child))
    vector = ET.tostring(svg, encoding="utf-8", xml_declaration=True)
    with pymupdf.open(stream=vector, filetype="svg") as drawing:
        with pymupdf.open("pdf", drawing.convert_to_pdf()) as document:
            pix = document[0].get_pixmap(matrix=pymupdf.Matrix(DPI*STRUCTURE_SCALE/72,
                                                              DPI*STRUCTURE_SCALE/72), alpha=True)
            raster = Image.frombytes("RGBA", [pix.width, pix.height], pix.samples)
    geometry = [{"clip_pt": list(clip), "x_pt": x, "y_pt": y} for clip, x, y in placements]
    return raster, width*STRUCTURE_SCALE/72, height*STRUCTURE_SCALE/72, glyphs, vector, geometry


def build2():
    p, s = inputs()
    e = json.loads((OUT / "example_pair.json").read_text(encoding="utf-8"))
    d = pd.read_csv(OUT / "example_pair.csv")
    perlib = library_intervals(p)
    fig = plt.figure(figsize=(6.5, 6.5), dpi=100)
    from figure1_gc import inch_axes
    a = inch_axes(fig, .68, 3.88, 5.62, 2.02)
    x = np.linspace(0, 80, 161)
    a.fill_between(x, x + 20, 100, color=ORANGE, alpha=.09, lw=0)
    a.axvspan(80, 100, color="#EEEEEE", lw=0)
    a.plot([0, 100], [0, 100], color="#BBBBBB", lw=.6)
    a.plot([0, 80], [20, 100], color=ORANGE, lw=.7)
    a.scatter(100*p.m6, 100*p.b, s=8, color=BLUE, alpha=.28, edgecolors="none", rasterized=False)
    bands = pd.read_csv(OUT / "posthit_yield_bands.csv")
    for lo, hi in ((20,40),(40,60),(60,80)):
        a.axvline(lo, ls=(0,(2,3)), lw=.5, color="#888888")
        r = bands[np.isclose(bands.m6_lower, lo/100)].iloc[0]
        a.text((lo+hi)/2, 115, "%d/%d" % (r.severe_k6,r.pairs), ha="center", fontsize=SMALL)
    a.text(90, 115, "ceiling", ha="center", color="#666666", fontsize=SMALL)
    cohort_label = a.text(1, 101, "%s hits\n184 >20 points short" % format(len(p),","),
                          fontsize=SMALL, color="#8A3B00", va="top")
    a.plot([100*e["m6"], 100*e["b"]], [100*e["b"]]*2, color=ORANGE,lw=1.4)
    a.plot(100*e["m6"],100*e["b"],"o",mfc="white",mec="black",ms=5)
    a.plot(100*e["b"],100*e["b"],"*",color="black",ms=8)
    example_label = a.text(50*(e["m6"]+e["b"]), 103, "Suzuki: %.1f → %.1f%%" % (100*e["m6"], 100*e["b"]),
                           fontsize=SMALL, color="#8A3B00", va="bottom", ha="center")
    a.set(xlim=(0,100),ylim=(0,120),xticks=range(0,101,20),yticks=range(0,101,20),
          xlabel="best yield of the first six conditions (%)",ylabel="library-best yield (%)")
    panel_title(fig,a,"a","A hit can leave a large measured shortfall",0,0)
    b = inch_axes(fig, 1.66, .45, 1.25, 2.70)
    # Metals are determined from the measured condition metadata, not inferred from names.
    raw, cond = lib.load("analysis")
    from review_axes_gc import variable_frame
    metals = {}
    for screen in perlib.screen:
        group = raw[raw.screen == screen]
        v = variable_frame(screen,cond,group,sorted(group.condition_id.unique()))
        metals[screen] = "/".join(sorted(set(v.metal))) if "metal" in v else ""
    labels=[]
    for y,r in enumerate(perlib.itertuples()):
        b.hlines(y,100*r.lo,100*r.hi,color=BLUE,lw=.8)
        b.plot(100*r.rate,y,"o",color=BLUE,ms=3.6)
        name=r.label
        if metals[r.screen] and not any(m in name for m in metals[r.screen].split("/")):
            if " (" in name and name.endswith(")"):
                name = name.replace(" (", ", ").replace(")", "")
            name += " ("+metals[r.screen]+")"
        labels.append(name)
        b.text(67,y,"%d/%d"%(r.severe,r.hits_below80),va="center",fontsize=SMALL)
    pooled_rate = 100*s["below_ceiling"]["severe_rate_k6"]
    pooled_line = b.vlines(pooled_rate, -.35, len(labels)-.5, color=ORANGE,
                           ls=(0,(2,2)), lw=.8)
    pooled_label = b.text(pooled_rate, -.92, "19.2%", ha="center", va="bottom",
                          fontsize=SMALL, color=ORANGE)
    b.set(yticks=range(len(labels)),yticklabels=labels,xlim=(0,85),xticks=[0,20,40,60],
          ylim=(len(labels)-.4,-1.9),xlabel="hits >20 points short (%)")
    b.tick_params(axis="y",length=0,pad=3)
    panel_title(fig,b,"b","Shortfalls vary across libraries",1,0)
    from figure1_gc import scheme_raster, inch_axes, recipe_parts, chem, CMAP
    raster, scheme_width, scheme_height, glyphs, vector, native_geometry = compact_example(fig, e)
    scheme_left, scheme_top = 3.14, 3.15
    scheme_ax = inch_axes(fig, scheme_left, scheme_top-scheme_height,
                          scheme_width, scheme_height)
    scheme_ax.axis("off")
    image = scheme_ax.imshow(raster, interpolation="nearest", aspect="auto")
    scheme_ax._chemical_font_audit = [(image, glyphs)]
    fig._vector_panels = [(image, vector)]
    hit_parts, best_parts = recipe_parts(d,e["hit_condition"]), recipe_parts(d,e["best_condition"])
    shared=[]
    while hit_parts and best_parts and hit_parts[-1]==best_parts[-1]:
        shared.insert(0,hit_parts.pop()); best_parts.pop()
    common=fig.text(5.36/6.5,3.05/6.5,chem(", ".join(shared)).replace(", 100", "\n100"),
                    fontsize=SMALL,ha="center",va="center")
    recipes=[]
    for y, marker, label, parts in ((1.49,"o","hit (%s), %.1f%%"%(e["hit_condition"],100*e["m6"]),hit_parts),
                                    (.99,"*","library best (%s), %.1f%%"%(e["best_condition"],100*e["b"]),best_parts)):
        symbol=Line2D([4.97/6.5],[y/6.5],transform=fig.transFigure,ls="none",marker=marker,
                      ms=5 if marker=="o" else 7,mfc="white" if marker=="o" else "black",mec="black",mew=.7)
        fig.add_artist(symbol)
        recipes.append(fig.text(5.07/6.5,y/6.5,label,fontsize=SMALL,va="center"))
        recipes.append(fig.text(4.97/6.5,(y-.19)/6.5,chem(", ".join(parts)),fontsize=SMALL,va="center"))
    c=inch_axes(fig,3.14,.41,1.70,1.1333333333)
    marked=heatmap(fig,c,e,d)
    scale=inch_axes(fig,5.05,.49,1.30,.06)
    for i in range(100):scale.add_patch(Rectangle((i,0),1.02,1,facecolor=CMAP((i+.5)/100),edgecolor="none"))
    scale.set(xlim=(0,100),ylim=(0,1),yticks=[],xticks=[0,50,100],xlabel="yield (%)")
    scale.tick_params(length=2,pad=1)
    title = fig.text(0,0,"Seven recipes beat the hit by >20 points")
    add_panel(fig,"c",title,[c,scheme_ax,scale,common]+recipes,1,1)
    fig._example_recipe_labels=recipes
    fig._example_shared_conditions=common
    fig._example_scheme_audit = {"width_in": scheme_width, "height_in": scheme_height,
                                 "scale": .9,
                                 "scheme_axis": scheme_ax, "heatmap_axis": c,
                                 "native_geometry": native_geometry,
                                 "master": str(FIG / "example_pair_scheme.pdf")}
    fig._posthit_libraries=perlib
    fig._posthit_scatter=p
    fig._heatmap_better=marked
    align_panels(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    points = a.transData.transform(np.column_stack((100*p.m6, 100*p.b)))
    label_boxes = [text.get_window_extent(renderer) for text in (cohort_label, example_label)]
    for box in label_boxes:
        assert not any(box.padded(1).contains(*point) for point in points), "Figure 2 label covers a measured hit"
    line_top = b.transData.transform((pooled_rate, -.35))[1]
    pooled_gap = pooled_label.get_window_extent(renderer).y0 - line_top
    assert pooled_gap > 1, "Pooled percentage overlaps its dashed reference line"
    fig._posthit_annotation_audit = {
        "cohort_label": cohort_label, "example_label": example_label,
        "pooled_label": pooled_label, "pooled_line": pooled_line,
        "pooled_label_line_gap_px_300dpi": pooled_gap * 300 / fig.dpi,
        "library_labels": labels, "suzuki_leader": False,
    }
    segment_y = a.transData.transform((100*e["m6"],100*e["b"]))[1]
    fig._posthit_annotation_audit["suzuki_line_gap_px_300dpi"] = (
        example_label.get_window_extent(renderer).y0-segment_y)*300/fig.dpi
    return fig


def build3():
    p,s=inputs()
    rows=outcomes(p)
    fig=plt.figure(figsize=(6.5,7.65),dpi=100)
    a=fig.add_axes([.285,.455,.68,.44])
    names=["fully recovered", "remaining ≤20 points", "improved, still >20", "no improvement"]
    pos=0
    positions=[]
    group_spans=[]
    for group,part in rows.groupby("group",sort=True):
        start=pos
        for r in part.itertuples():
            gain, gap = 100*(r.m12-r.m6), 100*(r.b-r.m6)
            a.plot([0,gain],[pos,pos],color=LIGHT,lw=.6,solid_capstyle="butt")
            a.plot([gain,gap],[pos,pos],color=ORANGE,lw=.6,solid_capstyle="butt")
            positions.append(pos)
            pos+=1
        end=pos-1
        group_spans.append((int(group),start,end,len(part)))
        a.text(-.035,(start+end)/2,"%s\nn = %d"%(names[int(group)],len(part)),
               transform=a.get_yaxis_transform(),ha="right",va="center",fontsize=SMALL)
        pos+=7
    a.set(xlim=(0,81),ylim=(pos-6,-2),yticks=[],xticks=[0,20,40,60,80],
          xlabel="yield points above the hit")
    a.spines["left"].set_visible(False)
    a.legend(handles=[Line2D([],[],color=LIGHT,lw=2,label="recovered"),
                      Line2D([],[],color=ORANGE,lw=2,label="remaining shortfall")],
             loc="lower right",frameon=False,bbox_to_anchor=(1,1.025),ncol=2,
             columnspacing=1.2,handlelength=1.1)
    panel_title(fig,a,"a","Most severe shortfalls improve",0,0)
    b=fig.add_axes([.295,.135,.30,.18])
    an,nb,ex,ho=s["analysed"],s["molecule_node_intervals"],s["extension_and_support"],s["holdout"]["all"]
    held_intervals, held_pairs = holdout_intervals()
    data=[("history-ranked",an["gap_recovery_k6_to_12"],ho["gap_recovery_k6_to_12"],nb["gap_recovery_k6_to_12_lo"],nb["gap_recovery_k6_to_12_hi"],BLUE),
          ("historical mean",ex["gap_recovery_meanyield_plus6"],ho["gap_recovery_meanyield_plus6"],nb["meanyield_recovery_lo"],nb["meanyield_recovery_hi"],LIGHT),
          ("random",ex["gap_recovery_random_plus6"],ho["gap_recovery_random_plus6"],nb["random_recovery_lo"],nb["random_recovery_hi"],GREY)]
    recovery=[]
    for i,(name,v,h,lo,hi,color) in enumerate(data):
        for shift,value,cohort in [(-.17,v,"analysed"),(.17,h,"held-out")]:
            hatch="///" if cohort=="held-out" else ""
            lower, upper = held_intervals[name] if hatch else (lo, hi)
            b.barh(i+shift,100*value,height=.29,color=color if not hatch else "white",
                   edgecolor=color,lw=.6,hatch=hatch)
            b.hlines(i+shift,100*lower,100*upper,color="black",lw=.7)
            b.text(100*upper+1.4,i+shift,
                   "%.1f%%"%(100*value) if hatch else "%.0f%%"%(100*value),va="center",fontsize=SMALL)
            recovery.append(dict(method=name,cohort=cohort,recovery=value,lo=lower,hi=upper))
    b.set(yticks=range(3),yticklabels=[r[0] for r in data],ylim=(2.5,-.55),xlim=(0,70),xticks=[0,20,40,60],xlabel="combined shortfall recovered (%)")
    b.legend(handles=[Patch(facecolor=GREY,edgecolor=GREY,label="analysed"),Patch(facecolor="white",edgecolor=GREY,hatch="///",label="held-out")],
             frameon=False,loc="lower center",bbox_to_anchor=(.65,1),borderaxespad=0,
             ncol=2,handlelength=1.1,columnspacing=1)
    annotations=[fig.text(.035,.365,"all %s hits: mean +%.1f points"%(format(len(p),","),100*(p.m12-p.m6).mean()),fontsize=SMALL),
                 fig.text(.035,.342,"below 80%%: lifted ≥10 points %.1f%% vs %.1f%%"%(100*s["below_ceiling"]["gain10_greedy_rate"],100*s["below_ceiling"]["gain10_random_rate"]),fontsize=SMALL)]
    panel_title(fig,b,"b","History recovers more than random",1,0)
    c=fig.add_axes([.765,.135,.20,.18])
    bc=s["below_ceiling"]
    ks=[6,12,24]
    rates=[100*bc["severe_rate_k%d"%k] for k in ks]
    errorbars = c.vlines(ks,[100*bc["severe_rate_k%d_lo"%k] for k in ks],[100*bc["severe_rate_k%d_hi"%k] for k in ks],color=BLUE,lw=.7)
    c.plot(ks,rates,"o-",color=BLUE,lw=1,ms=4,label="analysed")
    c.plot([6,12],[100*ho["severe_rate_below_ceiling_k6"],100*ho["severe_rate_below_ceiling_k12"]],"s--",mfc="white",mec=BLUE,color=BLUE,lw=.8,ms=4,label="held-out")
    rate_labels = []
    for k,v in zip(ks,rates):
        offset = (-4,-2.5) if k==6 else (3,10) if k==24 else (-7,-7)
        text = c.annotate("%.1f%%"%v,(k,v),xytext=offset,
                          ha="left" if k==24 else "right", va="bottom" if k==24 else "top",
                          textcoords="offset points",fontsize=SMALL)
        rate_labels.append(text)
    c.set(xlim=(-3,27),ylim=(0,30),xticks=ks,xlabel="conditions screened",ylabel="hits >20 points short (%)")
    c.legend(frameon=False,loc="lower left",bbox_to_anchor=(-.15,1.02),handlelength=1.4)
    panel_title(fig,c,"c","Severe-shortfall rates fall",1,1)
    align_panels(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    rate_boxes = [text.get_window_extent(renderer) for text in rate_labels]
    rate_paths = [line.get_path().transformed(line.get_transform()) for line in c.lines]
    rate_paths += [spine.get_path().transformed(spine.get_transform()) for spine in c.spines.values()
                   if spine.get_visible()]
    from matplotlib.path import Path as MplPath
    rate_paths += [MplPath(c.transData.transform(segment)) for segment in errorbars.get_segments()]
    for index, box in enumerate(rate_boxes):
        assert not any(box.padded(1).overlaps(other) for other in rate_boxes[index+1:]), "Figure 3c percentage labels overlap"
        assert not any(path.intersects_bbox(box.padded(1), filled=False) for path in rate_paths), "Figure 3c percentage label crosses a data line"
        assert not any(box.padded(1).overlaps(t.get_window_extent(renderer)) for t in c.get_yticklabels()+[c.yaxis.label]), "Figure 3c percentage label covers the yield axis"
    analysed_points = c.transData.transform(np.column_stack((ks,rates)))
    heldout_points = c.transData.transform(np.column_stack(([6,12],
        [100*ho["severe_rate_below_ceiling_k6"],100*ho["severe_rate_below_ceiling_k12"]])))
    def point_box_distance(point, box):
        dx=max(box.x0-point[0],0,point[0]-box.x1)
        dy=max(box.y0-point[1],0,point[1]-box.y1)
        return float(np.hypot(dx,dy))
    associations=[]
    for k, box, point in zip(ks,rate_boxes,analysed_points):
        own_distance=point_box_distance(point,box)
        held_distance=min(point_box_distance(h,box) for h in heldout_points)
        assert own_distance < held_distance, "Figure 3c label is closer to a held-out square than its analysed circle"
        assert (held_distance-own_distance)*300/fig.dpi >= 8, "Figure 3c label needs a clear margin of association with its analysed circle"
        associations.append({"conditions":k,"analysed_point_px":point.tolist(),
            "label_bbox_px":list(box.extents),"analysed_distance_px_300dpi":own_distance*300/fig.dpi,
            "heldout_distance_px_300dpi":held_distance*300/fig.dpi})
    fig._rate_label_audit = {"labels": rate_labels, "paths": rate_paths,
                              "clearance_px_300dpi": 300 / fig.dpi,
                              "associations":associations}
    row_pitch=a.get_window_extent().height/(a.get_ylim()[0]-a.get_ylim()[1])*300/fig.dpi
    fig._gap_strip_audit={"rows":len(rows),"counts":[r[3] for r in group_spans],"row_pitch_300dpi":row_pitch,
                          "stroke_px_300dpi":.6*300/72,"minimum_white_gap_px":row_pitch-.6*300/72,
                          "fallback":False,"group_spans":group_spans}
    assert row_pitch-.6*300/72>=2,"Individual severe-case rows need at least two white pixels"
    fig._recovery_data=pd.DataFrame(recovery)
    fig._heldout_pairs=held_pairs
    fig._severe_rows=rows
    fig._scale_annotations=annotations
    return fig


def write(fig,number):
    stem=STEMS[number]
    from figure1_gc import write as write_vector_figure
    write_vector_figure(fig,stem)


def main():
    for number,build in ((2,build2),(3,build3)):
        fig=build()
        write(fig,number)
        if number==2:fig._posthit_libraries.to_csv(OUT/"figure2_library_intervals.csv",index=False)
        else:
            fig._severe_rows.to_csv(OUT/"figure3_severe_outcomes.csv",index=False)
            fig._recovery_data.to_csv(OUT/"figure3_recovery.csv",index=False)
            geometry=dict(fig._gap_strip_audit)
            geometry["rate_label_associations"]=fig._rate_label_audit["associations"]
            (OUT/"figure3_geometry.json").write_text(json.dumps(geometry,indent=2)+"\n",encoding="utf-8")
        plt.close(fig)
    print("Figures 2/3 written")


if __name__=="__main__":main()
