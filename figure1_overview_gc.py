"""Editable Figure 1 overview: history chooses conditions; measured libraries score them."""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pymupdf as fitz
import lib

HERE = Path(__file__).resolve().parent
OUT, FIG = HERE / "out", HERE / "figures"
QA = HERE / "internal" / "codex_2026-10-21q"
WIDTH, HEIGHT = 468.0, 360.0
BLUE, LIGHT, ORANGE = "#0072B2", "#56B4E9", "#D55E00"
INK, GREY, PALE = "#222222", "#777777", "#B5B5B5"
NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", NS)


def _font_path(bold=False):
    return Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / ("arialbd.ttf" if bold else "arial.ttf")


def facts():
    """Read displayed values from the same released outputs used by the evidence figures."""
    summary = json.loads((OUT / "posthit_summary.json").read_text(encoding="utf-8"))
    coverage = json.loads((OUT / "measurement_coverage.json").read_text(encoding="utf-8"))
    recipes = pd.read_csv(OUT / "review_axes_recipes.csv")
    scores = recipes.pivot(index="screen", columns="method", values="retention").dropna(subset=["ligand_uniform6"])
    if lib.DATASET is None:
        raise FileNotFoundError("Released arrays.csv is required for the Figure 1 record count")
    statuses = pd.read_csv(lib.DATASET / "arrays.csv", usecols=["measurement_status"])
    frozen_wells = next((path for path in (
        HERE / "06_CORPUS_HOLDOUT/data/wells.csv",
        lib.DATASET.parent / "06_CORPUS_HOLDOUT/data/wells.csv") if path.is_file()), None)
    if frozen_wells is None:
        raise FileNotFoundError("Released frozen-member 06_CORPUS_HOLDOUT/data/wells.csv is required")
    released_records = int(statuses.measurement_status.isin(["quantified", "zero_by_rule"]).sum())
    released_records += len(pd.read_csv(frozen_wells, usecols=["yield"]))
    reconciliation = OUT / "corpus_reconciliation.csv"
    if reconciliation.is_file():
        corpus = pd.read_csv(reconciliation)
        assert released_records == int(corpus.loc[corpus.row.str.startswith("used_"), "records"].sum()), \
            "Released record count differs from the full-project corpus reconciliation"
    libraries = pd.read_csv(OUT / "kit_summary.csv")
    # CMC campaigns belong to the same C-N transformation as the other BH libraries.
    transformations = libraries.screen.str.split("|", regex=False).str[0].replace(
        {"Buchwald-Hartwig(CMC)": "Buchwald-Hartwig"}).nunique()
    bc, support, holdout = summary["below_ceiling"], summary["extension_and_support"], summary["holdout"]["all"]
    return {
        "below80_hits": int(bc["pairs"]), "severe_hits": int(bc["severe_k6"]),
        "severe_pct": 100 * bc["severe_rate_k6"],
        "all_hits": int(support["pairs"]),
        "recovered_pct": 100 * support["gap_recovery_greedy_k6_to_12"],
        "random_pct": 100 * support["gap_recovery_random_plus6"],
        "holdout_pairs": int(holdout["pairs"]),
        "holdout_recovered_pct": 100 * holdout["gap_recovery_k6_to_12"],
        "holdout_random_pct": 100 * holdout["gap_recovery_random_plus6"],
        "ligand_below_random": int((scores.ligand_uniform6 < scores.random6).sum()),
        "ligand_libraries": len(scores), "libraries": len(libraries),
        "transformations": int(transformations), "measured_yields": int(coverage["measured_cells"]),
        "released_records": released_records,
        "sources": {"shortfall_and_recovery": "out/posthit_summary.json",
                    "coverage": "out/measurement_coverage.json", "ligand": "out/review_axes_recipes.csv",
                    "libraries_and_transformations": "out/kit_summary.csv",
                    "released_records": "data/arrays.csv: quantified|zero_by_rule rows + 06_CORPUS_HOLDOUT/data/wells.csv: frozen-member rows"},
    }


def illustration_data():
    """Keep illustrative measured cells separate from the 15 published overview facts."""
    from figure1_gc import CMAP

    example = json.loads((OUT / "example_pair.json").read_text(encoding="utf-8"))
    values = pd.read_csv(OUT / "example_pair.csv")
    arrays, _ = lib.load("analysis")
    matrix = lib.screen_matrix(arrays, example["screen"])
    folds = np.array_split(np.random.RandomState(3).permutation(len(matrix)), 10)
    test = folds[example["fold"]]
    train = np.setdiff1d(np.arange(len(matrix)), test)
    assert example["subpair"] in matrix.index[test]
    selected = lib.greedy_kit(matrix.iloc[train].to_numpy(), 12)
    ordered = matrix.columns[selected].tolist()
    recorded = values.dropna(subset=["greedy_order"]).sort_values("greedy_order")
    assert recorded.condition_id.tolist() == ordered
    history = matrix.iloc[train[:4]][ordered[6:12]]
    assert not history.isna().any().any()
    assert example["subpair"] not in history.index
    observed = dict(zip(values.condition_id, values["yield"]))
    library = set(matrix.columns)

    def color(value):
        rgba = CMAP(float(value))
        return "#" + "".join(f"{round(255*c):02X}" for c in rgba[:3])

    cells = []
    for number in range(1, 97):
        condition = f"C{number}"
        value = observed.get(condition)
        status = "not_in_library" if condition not in library else "unmeasured" if value is None else "measured"
        cells.append({"condition": condition, "row": (number-1)//12, "column": (number-1)%12,
                      "status": status, "yield": value,
                      "color": color(value) if status == "measured" else "#E6E6E6"})
    conditions = pd.read_csv(OUT / "kit_summary.csv").conditions
    return {"screen": example["screen"], "subpair": example["subpair"], "fold": example["fold"],
            "matrix_shape": list(matrix.shape), "training_pairs": len(train), "test_pairs": len(test),
            "condition_count_range": [int(conditions.min()), int(conditions.max())],
            "first_six": [{"condition": condition, "yield": observed[condition], "color": color(observed[condition])}
                          for condition in ordered[:6]],
            "next_six": ordered[6:12], "board_cells": cells,
            "history_row_rule": "First four training rows in the matrix's SMILES lexicographic order; no yield-based row selection",
            "history_rows": history.index.tolist(), "history_columns": history.columns.tolist(),
            "history_values": history.to_numpy().tolist(),
            "history_colors": [[color(value) for value in row] for row in history.to_numpy()],
            "history_selection": "Greedy joint best-yield improvement over all 218 training pairs; column order is steps 7-12, not descending column mean",
            "board_role": "Illustrative scoring map of this pair, not the input used to select its conditions",
            "sources": ["out/example_pair.json", "out/example_pair.csv", "out/kit_summary.csv", "lib.load('analysis')"]}


class Drawing:
    def __init__(self):
        self.root = ET.Element(f"{{{NS}}}svg", {"width": "468pt", "height": "360pt",
            "viewBox": "0 0 468 360", "font-family": "Arial", "font-size": "8", "letter-spacing": "0"})
        ET.SubElement(self.root, f"{{{NS}}}title").text = "Screening history can recover yield left after a hit"
        self.layers = {name: ET.SubElement(self.root, f"{{{NS}}}g", {"id": name}) for name in
                       ("main_flow", "fork", "prior_work", "design_branches", "result_numbers", "scale")}
        self.texts, self.segments = [], []

    def add(self, layer, tag, **attrs):
        return ET.SubElement(self.layers[layer], f"{{{NS}}}{tag}",
                             {k.replace("_", "-"): str(v) for k, v in attrs.items()})

    def text(self, layer, x, y, value, size=8, color=INK, anchor="start", bold=False, ident=None):
        attrs = dict(x=x, y=y, font_family="Arial", font_size=size, fill=color,
                     text_anchor=anchor, font_weight="bold" if bold else "normal")
        if ident:
            attrs["id"] = ident
        self.add(layer, "text", **attrs).text = value
        self.texts.append(dict(id=ident or f"text_{len(self.texts)}", layer=layer,
                               x=x, y=y, text=value, size=size, color=color, anchor=anchor, bold=bold))

    def line(self, layer, x1, y1, x2, y2, color=INK, width=.6, dashed=False, role="connector"):
        attrs = dict(x1=x1, y1=y1, x2=x2, y2=y2, stroke=color, stroke_width=width)
        if dashed:
            attrs["stroke_dasharray"] = "3 2"
        self.add(layer, "line", **attrs)
        self.segments.append(dict(layer=layer, points=[[x1, y1], [x2, y2]], width=width, role=role))

    def arrow(self, layer, x1, y1, x2, y2, color=INK, width=.6, dashed=False):
        self.line(layer, x1, y1, x2, y2, color, width, dashed)
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        ux, uy = dx / length, dy / length
        vertices = [(x2, y2), (x2 - 4 * ux + 1.65 * uy, y2 - 4 * uy - 1.65 * ux),
                    (x2 - 4 * ux - 1.65 * uy, y2 - 4 * uy + 1.65 * ux)]
        self.add(layer, "polygon", points=" ".join(f"{x:.3f},{y:.3f}" for x, y in vertices), fill=color)

    def curve(self, layer, points, color=INK, width=1.5):
        p0, p1, p2, p3 = points
        self.add(layer, "path", d=f"M {p0[0]} {p0[1]} C {p1[0]} {p1[1]} {p2[0]} {p2[1]} {p3[0]} {p3[1]}",
                 fill="none", stroke=color, stroke_width=width)
        samples = []
        for i in range(51):
            t = i / 50
            samples.append([(1-t)**3 * p0[k] + 3*(1-t)**2*t*p1[k] +
                            3*(1-t)*t*t*p2[k] + t**3*p3[k] for k in (0, 1)])
        self.segments.append(dict(layer=layer, points=samples, width=width, role="connector"))
        # A separate final segment fixes the arrowhead direction in SVG and PDF renderers.
        q = samples[-2]
        self.arrow(layer, q[0], q[1], p3[0], p3[1], color, width)

    def matrix(self, layer, x, y, columns, rows, cell=4, gap=2, fill="white", color=GREY):
        for row in range(rows):
            for col in range(columns):
                self.add(layer, "rect", x=x+col*(cell+gap), y=y+row*(cell+gap), width=cell,
                         height=cell, fill=fill, stroke=color, stroke_width=.6)

    def motif(self, x1, x2, y, recovered=None):
        if recovered is not None:
            split = x1 + (x2-x1)*recovered
            self.line("fork", x1, y, split, y, LIGHT, 3, role="motif")
            self.line("fork", split, y, x2, y, ORANGE, 3, role="motif")
            self.add("fork", "polygon", points=f"{split},{y-4} {split+4},{y} {split},{y+4} {split-4},{y}",
                     fill=LIGHT, stroke=INK, stroke_width=.6)
        else:
            self.line("fork", x1, y, x2, y, ORANGE, 3, role="motif")
        self.add("fork", "circle", cx=x1, cy=y, r=3.2, fill="white", stroke=INK, stroke_width=.6)
        points = []
        for i in range(10):
            angle = -math.pi/2 + math.pi*i/5
            radius = 4.8 if i % 2 == 0 else 2
            points.append(f"{x2+radius*math.cos(angle):.3f},{y+radius*math.sin(angle):.3f}")
        self.add("fork", "polygon", points=" ".join(points), fill="white", stroke=INK, stroke_width=.6)

    def plate(self, layer, x, y, data, cell=4, gap=1, outline_only=False):
        width, height = 12*(cell+gap)-gap, 8*(cell+gap)-gap
        self.add(layer, "rect", x=x-3, y=y-3, width=width+6, height=height+6,
                 rx=3, fill="white" if outline_only else "#FAFAFA", stroke=PALE, stroke_width=.6)
        for item in data["board_cells"]:
            cx, cy = x+item["column"]*(cell+gap), y+item["row"]*(cell+gap)
            self.add(layer, "rect", x=cx, y=cy, width=cell, height=cell,
                     fill="white" if outline_only else item["color"],
                     stroke=PALE if outline_only else "white", stroke_width=.6)
            if item["status"] == "not_in_library":
                for x1, y1, x2, y2 in ((cx+.7, cy+.7, cx+cell-.7, cy+cell-.7),
                                      (cx+.7, cy+cell-.7, cx+cell-.7, cy+.7)):
                    self.add(layer, "line", x1=x1, y1=y1, x2=x2, y2=y2, stroke=PALE, stroke_width=.6)
            elif item["status"] == "unmeasured":
                self.add(layer, "line", x1=cx+.4, y1=cy+cell-.4, x2=cx+cell-.4, y2=cy+.4,
                         stroke=GREY if not outline_only else PALE, stroke_width=.6)
        return [x-3, y-3, x+width+3, y+height+3]

    def prior_icon(self, x, y, kind):
        cell, gap = 3.8, .8
        chosen = {"general": {(row, 3) for row in range(6)},
                  "coverage": {(row, col) for row in range(6) for col in (1, 4, 6)},
                  "recommendation": {(row, col) for row, col in enumerate((1, 5, 0, 6, 3, 7))},
                  "iteration": set()}[kind]
        for row in range(6):
            for col in range(8):
                self.add("prior_work", "rect", x=x+col*(cell+gap), y=y+row*(cell+gap),
                         width=cell, height=cell, fill=LIGHT if (row, col) in chosen else "#ECEFF1",
                         stroke="white", stroke_width=.6)
        if kind == "iteration":
            nodes = [(x+4, y+5), (x+17, y+5), (x+17, y+22), (x+32, y+22)]
            self.add("prior_work", "polyline", points=" ".join(f"{a},{b}" for a, b in nodes),
                     fill="none", stroke=BLUE, stroke_width=.6)
            for number, (cx, cy) in enumerate(nodes, 1):
                self.add("prior_work", "circle", cx=cx, cy=cy, r=4.5, fill="white", stroke=BLUE, stroke_width=.6)
                self.text("prior_work", cx, cy+2.7, str(number), color=BLUE, anchor="middle")
        elif kind == "coverage":
            for row in (0, 2, 4):
                cy = y+row*(cell+gap)+2
                self.add("prior_work", "polyline", points=f"{x+38},{cy} {x+40},{cy+2} {x+43},{cy-2}",
                         fill="none", stroke=GREY, stroke_width=.6)


def build_svg(values=None):
    f = values or facts()
    d = Drawing()
    data = illustration_data()
    d.illustrations = data
    d.text("prior_work", 8, 9, "Selected research goals", color=GREY)
    cards = [(15, "iteration", "Single-reaction iteration", "Shields, Nature 2021"),
             (78, "general", "General conditions", "Wang, Nature 2024"),
             (141, "coverage", "Coverage", "Sivilotti, Digital Discovery 2025"),
             (204, "recommendation", "Per-pair recommendation", "Rinehart, Science 2023")]
    for y, kind, title, cite in cards:
        d.add("prior_work", "rect", x=8, y=y, width=140, height=54, rx=4,
              fill="#F4F6F7", stroke="#E3E7E9", stroke_width=.6)
        d.prior_icon(59, y+3, kind)
        d.text("prior_work", 78, y+39, title, size=9, color=GREY, anchor="middle")
        d.text("prior_work", 78, y+51, cite, color=GREY, anchor="middle")

    d.text("main_flow", 241, 31, "A + B", size=9, anchor="middle")
    d.text("main_flow", 269, 31, "\u00d7", size=9, anchor="middle")
    top_plate = d.plate("main_flow", 282, 12, data)
    low, high = data["condition_count_range"]
    d.text("main_flow", 311, 66, f"condition library ({low}\u2013{high} conditions)", size=9, anchor="middle")
    d.arrow("main_flow", 311, 71, 311, 81, color=BLUE, width=1.5)
    for index, item in enumerate(data["first_six"]):
        d.add("main_flow", "rect", x=285+index*8.8, y=87, width=7, height=7,
              fill=item["color"], stroke="white", stroke_width=1.8)
        d.add("main_flow", "rect", x=285+index*8.8, y=87, width=7, height=7,
              fill="none", stroke=BLUE, stroke_width=.6)
    d.text("main_flow", 311, 113, "first six conditions", size=9, color=BLUE, anchor="middle")
    d.arrow("main_flow", 311, 119, 311, 153, color=BLUE, width=1.5)
    d.text("result_numbers", 375, 88, "ligand spread < random", color=GREY)
    d.text("result_numbers", 375, 100,
           f"in {f['ligand_below_random']} of {f['ligand_libraries']} libraries", color=GREY)
    d.line("prior_work", 148, 145, 311, 145, color=PALE, width=.6)

    d.add("fork", "rect", x=158, y=151, width=302, height=207, rx=7,
          fill=LIGHT, fill_opacity=.06, stroke="none", id="this_work_container")
    d.text("fork", 170, 167, "this work", size=9, color=BLUE)
    d.add("fork", "ellipse", cx=311, cy=167, rx=25, ry=12, fill="white", stroke=BLUE, stroke_width=1.5)
    d.text("fork", 311, 170, "HIT", size=9, anchor="middle", color=BLUE)
    d.curve("fork", [(311, 180), (311, 190), (247, 189), (231, 198)], color=INK)
    d.curve("fork", [(311, 180), (311, 190), (375, 189), (391, 198)], color=BLUE)
    d.text("fork", 231, 214, "stop", size=9, anchor="middle")
    d.text("fork", 391, 213, "screen six more, ranked by", size=9, anchor="middle", color=BLUE)
    d.text("fork", 391, 225, "the screening history", size=9, anchor="middle", color=BLUE)
    d.motif(166, 296, 236)
    d.motif(326, 456, 236, recovered=f["recovered_pct"]/100)
    d.text("result_numbers", 166, 256, f"{f['severe_pct']:.1f}%", size=9, bold=True,
           ident="severe_percentage")
    d.text("result_numbers", 199, 256, "of hits below 80%")
    d.text("result_numbers", 166, 268, "leave >20 yield points")
    d.text("result_numbers", 166, 280, f"({f['severe_hits']}/{f['below80_hits']})")
    d.text("result_numbers", 326, 256, f"{f['recovered_pct']:.0f}%", size=9, bold=True,
           ident="recovered_percentage")
    d.text("result_numbers", 351, 256, f"recovered vs {f['random_pct']:.0f}% at random")
    d.text("result_numbers", 326, 268, f"combined shortfall of all {f['all_hits']:,} hits")

    hx, hy, hcell, hgap = 342, 278, 5, 1
    for row, colors in enumerate(data["history_colors"]):
        for col, color in enumerate(colors):
            d.add("design_branches", "rect", x=hx+col*(hcell+hgap), y=hy+row*(hcell+hgap),
                  width=hcell, height=hcell, fill=color, stroke="white", stroke_width=.6)
    diamond_x = 326+130*f["recovered_pct"]/100
    # One continuous selection route skirts the explanation, with one arrowhead at the diamond.
    d.line("design_branches", hx, 289, 311, 289, color=BLUE, width=.6)
    d.line("design_branches", 311, 289, 311, 242, color=BLUE, width=.6)
    d.line("design_branches", 311, 242, diamond_x, 242, color=BLUE, width=.6)
    d.arrow("design_branches", diamond_x, 242, diamond_x, 240, color=BLUE, width=.6)
    scoring_plate = d.plate("design_branches", 416, 278, data, cell=2.3, gap=.5, outline_only=True)
    # Keep the scoring route to the right of both result-text lines.
    d.line("design_branches", 451, 274, 456, 274, color=GREY, width=.6, dashed=True)
    d.arrow("design_branches", 456, 274, 456, 243, color=GREY, width=.6, dashed=True)
    for y, history, scoring in ((316, "screening yield", "this pair's"),
                               (327, "history ranks", "measured library"),
                               (338, "the next six", "scores only")):
        d.text("design_branches", 360, y, history, anchor="middle", color=BLUE)
        d.text("design_branches", 430, y, scoring, anchor="middle", color=GREY)
    d.text("design_branches", 170, 309, "the hit's outcome is not", color=GREY)
    d.text("design_branches", 170, 321, "used to choose the next six", color=GREY)
    d.text("result_numbers", 326, 353,
           f"{f['holdout_pairs']:,} held-out pairs: {f['holdout_recovered_pct']:.1f}% vs {f['holdout_random_pct']:.1f}%", color=GREY)

    d.add("scale", "rect", x=8, y=278, width=140, height=76, rx=4,
          fill="#F4F6F7", stroke="#E3E7E9", stroke_width=.6)
    d.text("scale", 17, 298, f"{f['libraries']} libraries \u00b7 {f['transformations']} transformations", color=GREY)
    d.text("scale", 17, 319, f"{f['measured_yields']:,} measured yields", color=GREY)
    d.text("scale", 17, 340, f"{f['released_records']:,} records released", color=GREY)
    d.art_geometry = {"top_plate_bbox_pt": top_plate, "scoring_plate_bbox_pt": scoring_plate,
                      "this_work_bbox_pt": [158, 151, 460, 358], "this_work_fill_opacity": .06,
                      "prior_card_bbox_pt": [[8, y, 148, y+54] for y, *_ in cards],
                      "scale_card_bbox_pt": [8, 278, 148, 354],
                      "coverage_line_pt": [[148, 145], [311, 145]],
                      "main_arrow_width_pt": 1.5, "other_connector_width_pt": .6,
                      "history_heatmap_bbox_pt": [hx, hy, hx+35, hy+23],
                      "history_selection_route_pt": [[hx, 289], [311, 289], [311, 242], [diamond_x, 242], [diamond_x, 240]],
                      "measured_library_scoring_route_pt": [[451, 274], [456, 274], [456, 243]],
                      "source_arrow_endpoints": {"history": {"at": [diamond_x, 240], "target": "added-result diamond"},
                                                 "measured_library": {"at": [456, 243], "target": "library-best star"}}}
    return d, f


def render_svg(svg_source, pdf_path=None, png_path=None):
    """Render SVG paths with MuPDF and embed editable Arial text in the vector PDF."""
    root = ET.fromstring(svg_source) if isinstance(svg_source, bytes) else ET.parse(svg_source).getroot()
    shapes = copy.deepcopy(root)
    for parent in shapes.iter():
        for child in list(parent):
            if child.tag == f"{{{NS}}}text":
                parent.remove(child)
            elif child.get("stroke-dasharray") and child.tag in (f"{{{NS}}}line", f"{{{NS}}}rect"):
                # MuPDF's SVG reader ignores dash arrays; expand them to explicit vector strokes.
                insertion = list(parent).index(child)
                strokes = []
                if child.tag.endswith("line"):
                    edges = [(float(child.get("x1")), float(child.get("y1")),
                              float(child.get("x2")), float(child.get("y2")))]
                else:
                    x, y, w, h = (float(child.get(k)) for k in ("x", "y", "width", "height"))
                    edges = [(x, y, x+w, y), (x+w, y, x+w, y+h),
                             (x+w, y+h, x, y+h), (x, y+h, x, y)]
                for x1, y1, x2, y2 in edges:
                    length = math.hypot(x2-x1, y2-y1)
                    for start in range(0, math.ceil(length), 5):
                        end = min(start+3, length)
                        strokes.append(ET.Element(f"{{{NS}}}line", {
                            "x1": str(x1+(x2-x1)*start/length), "y1": str(y1+(y2-y1)*start/length),
                            "x2": str(x1+(x2-x1)*end/length), "y2": str(y1+(y2-y1)*end/length),
                            "stroke": child.get("stroke"), "stroke-width": child.get("stroke-width", ".8")}))
                parent.remove(child)
                for offset, stroke in enumerate(strokes):
                    parent.insert(insertion+offset, stroke)
    with fitz.open(stream=ET.tostring(shapes), filetype="svg") as vector:
        pdf_bytes = vector.convert_to_pdf()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    fonts = {False: fitz.Font(fontfile=str(_font_path(False))),
             True: fitz.Font(fontfile=str(_font_path(True)))}
    for bold, name in ((False, "ArialRegular"), (True, "ArialBold")):
        page.insert_font(fontname=name, fontfile=str(_font_path(bold)))
    for element in root.iter(f"{{{NS}}}text"):
        text = "".join(element.itertext())
        size = float(element.get("font-size", "8"))
        bold = element.get("font-weight", "normal") == "bold"
        x, y = float(element.get("x")), float(element.get("y"))
        anchor = element.get("text-anchor", "start")
        width = fonts[bold].text_length(text, fontsize=size)
        x -= width / 2 if anchor == "middle" else width if anchor == "end" else 0
        color = element.get("fill", INK).lstrip("#")
        rgb = tuple(int(color[i:i+2], 16)/255 for i in (0, 2, 4))
        page.insert_text((x, y), text, fontname="ArialBold" if bold else "ArialRegular", fontsize=size, color=rgb)
    doc.set_metadata({"title": "Figure 1: screening histories recover yield after a hit",
                      "author": "", "creator": "figure1_overview_gc.py", "producer": "PyMuPDF"})
    pdf_bytes = doc.tobytes(garbage=4, deflate=True, no_new_id=True)
    png_bytes = page.get_pixmap(dpi=300, alpha=False).tobytes("png")
    if pdf_path is not None:
        Path(pdf_path).write_bytes(pdf_bytes)
    if png_path is not None:
        Path(png_path).write_bytes(png_bytes)
    doc.close()
    return {"pdf_bytes": pdf_bytes, "png_bytes": png_bytes}


def geometry_qa(drawing, values):
    fonts = {b: fitz.Font(fontfile=str(_font_path(b))) for b in (False, True)}
    boxes = []
    for t in drawing.texts:
        font = fonts[t["bold"]]
        width = font.text_length(t["text"], fontsize=t["size"])
        x = t["x"] - (width/2 if t["anchor"] == "middle" else width if t["anchor"] == "end" else 0)
        box = fitz.Rect(x, t["y"]-font.ascender*t["size"], x+width, t["y"]-font.descender*t["size"])
        boxes.append({**t, "bbox_pt": list(box)})
    intersections = []
    for i, a in enumerate(boxes):
        for b in boxes[i+1:]:
            overlap = fitz.Rect(a["bbox_pt"]) & fitz.Rect(b["bbox_pt"])
            if not overlap.is_empty:
                intersections.append(dict(first=a["id"], second=b["id"], overlap_pt=list(overlap)))
    outside = [t for t in boxes if not fitz.Rect(0, 0, WIDTH, HEIGHT).contains(fitz.Rect(t["bbox_pt"]))]
    crossings = []
    for t in boxes:
        for segment in drawing.segments:
            rect = fitz.Rect(t["bbox_pt"])
            padding = segment["width"]/2
            rect += (-padding, -padding, padding, padding)
            for first, second in zip(segment["points"], segment["points"][1:]):
                clipped = True
                # Liang-Barsky segment clipping avoids bounding-box false positives on curves.
                dx, dy = second[0]-first[0], second[1]-first[1]
                lower, upper = 0.0, 1.0
                for p, q in ((-dx, first[0]-rect.x0), (dx, rect.x1-first[0]),
                             (-dy, first[1]-rect.y0), (dy, rect.y1-first[1])):
                    if abs(p) < 1e-12:
                        if q < 0:
                            clipped = False
                            break
                    elif p < 0:
                        lower = max(lower, q/p)
                    else:
                        upper = min(upper, q/p)
                if clipped and lower <= upper:
                    crossings.append(dict(text=t["id"], line=segment))
                    break
    return {"width_pt": WIDTH, "height_pt": HEIGHT, "png_size_px": [1950, 1500],
            "facts": values, "layers": list(drawing.layers), "texts": boxes, "segments": drawing.segments,
            "text_text_intersections": intersections, "text_outside_canvas": outside,
            "text_segment_intersections": crossings,
            "bold_texts": [t["text"] for t in drawing.texts if t["bold"]],
            "render": "MuPDF vector paths with embedded Arial text; PDF rendered at 300 dpi",
            "illustrations": drawing.illustrations, "illustration_geometry": drawing.art_geometry,
            "source_arrow_endpoints": drawing.art_geometry["source_arrow_endpoints"],
            "visual_review": "Requires separate PNG inspection; the generator records geometry only."}


def _master_render(path):
    pdf_bytes = path.read_bytes()
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        if len(doc) != 1:
            raise ValueError("Figure1 overview master must contain exactly one page")
        page = doc[0]
        if abs(page.rect.width-WIDTH) > .1 or page.rect.height > HEIGHT+.1:
            raise ValueError("Figure1 overview master must be 468 pt wide and at most 360 pt high")
        spans = [span for block in page.get_text("dict")["blocks"] if "lines" in block
                 for line in block["lines"] for span in line["spans"] if span["text"].strip()]
        font_names = sorted({span["font"] for span in spans})
        sizes = sorted({round(span["size"], 2) for span in spans})
        if spans and (any("arial" not in name.lower() for name in font_names) or
                      any(min(abs(size-8), abs(size-9)) > .1 for size in sizes)):
            raise ValueError("Figure1 overview master must retain Arial 8/9 pt text")
        drawings = page.get_drawings()
        if not drawings:
            raise ValueError("Figure1 overview master must contain vector artwork")
        png_bytes = page.get_pixmap(dpi=300, alpha=False).tobytes("png")
        metadata = {"path": str(path), "width_pt": page.rect.width, "height_pt": page.rect.height,
                    "fonts": font_names, "font_sizes_pt": sizes, "vector_paths": len(drawings),
                    "image_count": len(page.get_images()), "text_spans": spans}
    return {"pdf_bytes": pdf_bytes, "png_bytes": png_bytes}, metadata


def generate(write_outputs=True):
    drawing, values = build_svg()
    svg = FIG / "Figure1_overview.svg"
    pdf, png = [FIG / ("Figure1_hit_gap." + ext) for ext in ("pdf", "png")]
    svg_bytes = ET.tostring(drawing.root, encoding="utf-8", xml_declaration=True)
    qa = geometry_qa(drawing, values)
    master = FIG / "Figure1_overview_master.pdf"
    if master.exists():
        rendered, metadata = _master_render(master)
        qa.update(source="master", master=metadata,
                  geometry_scope="Text/segment intersections describe the generated SVG; master requires independent geometry QA")
    else:
        rendered = render_svg(svg_bytes)
        qa.update(source="generated_svg", geometry_scope="Final generated SVG and PDF share Arial text positions")
    if write_outputs:
        FIG.mkdir(exist_ok=True)
        svg.write_bytes(svg_bytes)
        pdf.write_bytes(rendered["pdf_bytes"])
        png.write_bytes(rendered["png_bytes"])
        if (HERE / "internal").is_dir():
            QA.mkdir(parents=True, exist_ok=True)
            (QA / "figure1_overview_geometry.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return {"svg": svg, "pdf": pdf, "png": png, "svg_bytes": svg_bytes,
            **rendered, "facts": values, "qa": qa}


if __name__ == "__main__":
    result = generate()
    print(json.dumps({k: str(v) for k, v in result.items() if k in ("svg", "pdf", "png")}, indent=2))
    print("Text overlaps:", len(result["qa"]["text_text_intersections"]),
          "outside:", len(result["qa"]["text_outside_canvas"]))
