"""Select and draw the recovery examples using the predeclared rule below."""

from __future__ import annotations

SELECTION_RULE = (
    "Start with the improved severe cases in posthit_added_recipes.csv, using the same "
    "eligibility as the manuscript's 131 of 184 cases. Exclude the Figure 1 substrate pair. "
    "Group by transformation, choose the largest m12-m6 within each group, breaking ties "
    "by larger library-best yield b and then lexicographic subpair string. Sort those "
    "representatives by decreasing m12-m6 (with the same deterministic ties) and select "
    "the first six transformations, or all transformations if fewer than six qualify."
)

import json
import io
import math
from pathlib import Path
import re
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea, VPacker
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

import chemical_schemes_gc as schemes
import lib
from figure_style_gc import LARGE, SMALL, STRUCTURE_SCALE, add_panel, align_panels, configure_fonts
from presentation_gc import array_label

HERE = Path(__file__).resolve().parent
OUT, FIG = HERE / "out", HERE / "figures"
STEM = "Figure4_recovered_recipes"
W, H, DPI = 6.5, 8.9, 300
ORANGE = "#D55E00"
ADDED = "#56B4E9"
BEST = "#A7A7A7"
TITLE = "The recovering recipes keep the metal and change the ligand, base or additive"
_TEMPLATES = {
    "Buchwald-Hartwig": "[N;H1,H2:1].[c:2][Br,Cl,I]>>[N:1]-[c:2]",
    "Acylation": "[N;H1,H2:1].[C:2](=[O:3])[O;H1]>>[N:1][C:2]=[O:3]",
    "Suzuki": "[c:1][Br,Cl,I].[c:2]B(O)O>>[c:1]-[c:2]",
    "Ullmann": "[O;H1:1].[c:2][Br,Cl,I]>>[O:1]-[c:2]",
    "Grignard": "[#6:1][Mg][Br,Cl,I].[C:2]=[O:3]>>[#6:1]-[C:2][O:3]",
    "Kumada": "[c:1][Br,Cl,I].[#6:2][Mg][Br,Cl,I]>>[c:1]-[#6:2]",
}


def product_for(transformation, first, second, source_rows):
    """Prefer an observed product field; otherwise require one unique sanitized template product."""
    fields = [c for c in source_rows if "product" in c.lower() and "smiles" in c.lower()]
    found = set()
    for field in fields:
        for value in source_rows[field].dropna():
            mol = Chem.MolFromSmiles(str(value))
            assert mol is not None, (field, value)
            found.add(Chem.MolToSmiles(mol))
    if found:
        assert len(found) == 1, found
        return found.pop(), "dataset product SMILES", ""
    smarts = _TEMPLATES.get(transformation)
    if smarts is None:
        return "", "omitted: no unique product template", ""
    reaction = AllChem.ReactionFromSmarts(smarts)
    for products in reaction.RunReactants((Chem.MolFromSmiles(first), Chem.MolFromSmiles(second))):
        if len(products) != 1:
            continue
        try:
            Chem.SanitizeMol(products[0])
        except (ValueError, RuntimeError):
            continue
        found.add(Chem.MolToSmiles(products[0], isomericSmiles=True))
    if len(found) != 1:
        return "", "omitted: template has %d unique products" % len(found), smarts
    return found.pop(), "unique sanitized reaction-template product", smarts


def _components(condition, screen):
    parts = []
    for slot, role in lib.ROLES[screen].items():
        field = "Standardized Name" if slot == "solvent" else slot + "_Standardized Name"
        value = condition.get(field)
        value = str(value).strip() if pd.notna(value) else "none"
        value = value or "none"
        amount = condition.get(slot + "_HTE_Equiv") if slot != "solvent" else None
        parts.append({"slot": slot, "role": role, "name": value,
                      "equiv": float(amount) if pd.notna(amount) else None})
    return parts


def select_cases():
    """Apply SELECTION_RULE to the same 131 improved cases used by the manuscript."""
    severe = pd.read_csv(OUT / "posthit_added_recipes.csv")
    assert len(severe) == 184
    assert (severe.m6 >= .20).all() and (severe.b - severe.m6 > .20).all()
    improved = severe[severe.added_condition.notna()].copy()
    assert len(improved) == 131 and (improved.m12 > improved.m6).all()
    first = json.loads((OUT / "example_pair.json").read_text(encoding="utf-8"))
    a, conditions = lib.load("analysis")
    mapping = a.groupby("screen").transformation.first()
    improved["transformation"] = improved.screen.map(mapping)
    assert improved.transformation.notna().all()
    # Exclude the actual Figure 1 pair before the per-transformation maximum is selected.
    improved = improved[~((improved.screen == first["screen"]) & (improved.subpair == first["subpair"]))].copy()
    improved["gain"] = improved.m12 - improved.m6
    picks = (improved.sort_values(["gain", "b", "subpair", "screen"], ascending=[False, False, True, True])
             .drop_duplicates("transformation").head(6))
    cases = []
    for panel, (_, pick) in zip("abcdef", picks.iterrows()):
        rows = a[(a.screen == pick.screen) & (a.subpair == pick.subpair)]
        assert len(rows)
        rxn_class = rows.rxn_class.iloc[0]
        cond = conditions[conditions.rxn_class == rxn_class].set_index("condition_id")
        assert not cond.index.duplicated().any()
        hit, added = cond.loc[pick.hit_condition], cond.loc[pick.added_condition]
        hit_parts, added_parts = _components(hit, pick.screen), _components(added, pick.screen)
        by_slot = {p["slot"]: p for p in hit_parts}
        for part in added_parts:
            part["changed"] = (part["name"], part["equiv"]) != (by_slot[part["slot"]]["name"], by_slot[part["slot"]]["equiv"])
            by_slot[part["slot"]]["changed"] = part["changed"]
        first_smiles, second_smiles = pick.subpair.split(" >> ")
        product, product_source, template = product_for(pick.transformation, first_smiles, second_smiles, rows)
        changed = [c[len("changed_"):] for c in severe if c.startswith("changed_")
                   and pd.notna(pick[c]) and bool(pick[c])]
        changed_slots = [p["slot"] for p in added_parts if p["changed"]]
        # Cross-check each displayed value against the public measured rows.
        for cid, expected in ((pick.hit_condition, pick.m6), (pick.added_condition, pick.m12)):
            values = rows.loc[rows.condition_id == cid, "yield"]
            assert len(values) == 1 and np.isclose(values.iloc[0], expected, atol=1e-12)
        assert np.isclose(rows["yield"].max(), pick.b, atol=1e-12)
        case = {"panel": panel, "screen": pick.screen, "transformation": pick.transformation,
                "display_name": array_label(pick.screen),
                "panel_title": "Ullmann-library C–O" if pick.screen == "Ullmann|1" else array_label(pick.screen),
                "hit_metal": lib.metal_of(hit.get("Reagent1_Standardized Name", "none")),
                "added_metal": lib.metal_of(added.get("Reagent1_Standardized Name", "none")),
                "subpair": pick.subpair,
                "sub_1_smiles": first_smiles, "sub_2_smiles": second_smiles,
                "product_smiles": product, "product_source": product_source, "product_template": template,
                "m6": float(pick.m6), "m12": float(pick.m12), "b": float(pick.b), "gain": float(pick.gain),
                "hit_condition": pick.hit_condition, "added_condition": pick.added_condition,
                "changed_variables": changed, "changed_slots": changed_slots,
                "hit_components": hit_parts, "added_components": added_parts,
                "hit_temperature_C": float(hit.temp_id), "added_temperature_C": float(added.temp_id),
                "hit_time_h": float(hit.time_id), "added_time_h": float(added.time_id),
                "source": "out/posthit_added_recipes.csv; public data/arrays.csv and data/conditions.csv"}
        cases.append(case)
    return cases


def facts_for(cases):
    ligand = sum("ligand" in c["changed_variables"] for c in cases)
    base_only = sum(c["changed_variables"] == ["base"] for c in cases)
    return {"case_count": len(cases), "m6_min_pct": 100 * min(c["m6"] for c in cases),
            "m6_max_pct": 100 * max(c["m6"] for c in cases),
            "m12_min_pct": 100 * min(c["m12"] for c in cases),
            "m12_max_pct": 100 * max(c["m12"] for c in cases),
            "changed_only_base": base_only, "changed_ligand": ligand,
            "changed_metal": sum("metal" in c["changed_variables"] for c in cases),
            "library_best_above_m12": sum(c["b"] > c["m12"] + 1e-12 for c in cases),
            "omitted_products": sum(not c["product_smiles"] for c in cases)}


def write_data(cases):
    OUT.mkdir(exist_ok=True)
    rows = []
    for case in cases:
        rows.append({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, list) else v
                     for k, v in case.items()})
    pd.DataFrame(rows).to_csv(OUT / "recovered_recipes.csv", index=False, lineterminator="\n")
    facts = facts_for(cases)
    sentence = ("In the six transformations of Fig. 4, the added condition lifts the hit from "
                "%.1f\u2013%.1f%% to %.1f\u2013%.1f%%; %d of the six change the ligand." %
                (facts["m6_min_pct"], facts["m6_max_pct"], facts["m12_min_pct"], facts["m12_max_pct"], facts["changed_ligand"]))
    components = component_facts()
    pd.DataFrame(components).to_csv(OUT / "figure4_components.csv", index=False, lineterminator="\n")
    summary = {"title": TITLE, "selection_rule": SELECTION_RULE, "input_severe_cases": 184, "input_improved_cases": 131,
               "facts": facts, "sentence": sentence, "cases": cases,
               "component_comparison": components,
               "fact_sources": {k: "out/recovered_recipes.csv; deterministic selection from out/posthit_added_recipes.csv"
                                for k in facts}}
    (OUT / "recovered_recipes.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _native_schemes(cases):
    """Read the deposited native master, refusing another case, role, scale, or missing asset."""
    import copy
    import pymupdf
    from PIL import Image
    master = FIG / "recovered_recipes_schemes.pdf"
    source = FIG / "recovered_recipes_schemes.cdxml"
    assert master.exists() and source.exists(), "missing Figure 4 ChemDraw master"
    root = ET.fromstring(source.read_text(encoding="utf-8"))
    for key, value in (("BondLength", 14.4), ("LineWidth", .6), ("LabelSize", 10)):
        assert float(root.get(key)) == value, ("Figure 4 native ACS setting", key, root.get(key))
    for bond in root.iter("b"):
        for key, value in {"BondLength": 14.4, "LineWidth": .6, "BoldWidth": 2,
                           "BondSpacing": 18, "HashSpacing": 2.5, "MarginWidth": 1.6}.items():
            assert abs(float(bond.get(key, root.get(key))) - value) < 1e-9, ("native bond style", key)
    pages = root.find("page").findall("group")
    table = pd.read_csv(OUT / "recovered_recipes.csv").set_index("panel")
    assert len(pages) == len(cases) == len(table) == 6
    drawings = []
    zoom = DPI * STRUCTURE_SCALE / 72
    with pymupdf.open(master) as pdf:
        assert len(pdf) == len(cases)
        for index, (page, case) in enumerate(zip(pages, cases)):
            assert page.get("Name") == case["panel"], "native master panel order differs"
            fragments = page.findall("fragment")
            assert len(fragments) == 3, (case["panel"], "three roles required")
            for fragment, role in zip(fragments, ("sub_1_smiles", "sub_2_smiles", "product_smiles")):
                assert fragment.get("Name") == role
                only = copy.deepcopy(root)
                for sibling in list(only):
                    if sibling.tag == "page":
                        only.remove(sibling)
                ET.SubElement(only, "page", {"id": "1"}).append(copy.deepcopy(fragment))
                molecules = Chem.MolsFromCDXML(ET.tostring(only, encoding="unicode"))
                expected = Chem.MolToSmiles(Chem.MolFromSmiles(str(table.loc[case["panel"], role])))
                assert len(molecules) == 1 and Chem.MolToSmiles(molecules[0]) == expected, (case["panel"], role)
                assert Chem.MolToSmiles(Chem.MolFromSmiles(case[role])) == expected
            atoms = {node.get("id"): tuple(map(float, node.get("p").split()))
                     for node in page.iter("n") if node.get("p")}
            bonds = [math.dist(atoms[b.get("B")][:2], atoms[b.get("E")][:2]) * zoom
                     for b in page.iter("b") if b.get("B") in atoms and b.get("E") in atoms]
            # Native CDXML rounds atom coordinates to 0.01 pt (at most 0.04 printed pixels).
            assert bonds and max(abs(length - 14.4 * zoom) for length in bonds) < .05
            glyphs = [{"font_px": 10 * zoom, "text": "native atom labels", "role": "atom"}]
            for span in page.iter("s"):
                script = int(span.get("face", "0")) & (32 | 64)
                glyphs.append({"font_px": float(span.get("size", 10)) * (.75 if script else 1) * zoom,
                               "text": "".join(span.itertext()), "role": "subscript or charge" if script else "atom"})
            vector = pdf[index]
            assert not vector.get_images(), "Figure 4 master must contain only vector chemistry"
            pix = vector.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=True)
            raster = Image.frombytes("RGBA", (pix.width, pix.height), pix.samples)
            paths = vector.get_drawings()
            arrow = paths[-2]["rect"] | paths[-1]["rect"]
            drawings.append({"raster": raster, "width": vector.rect.width * STRUCTURE_SCALE / 72,
                             "height": vector.rect.height * STRUCTURE_SCALE / 72,
                             "glyphs": glyphs, "bond_lengths_px": bonds,
                             "arrow_fraction": (arrow.x0 + arrow.x1) / (2 * vector.rect.width),
                             "vector": (master, index), "page_index": index})
    return drawings


def _formula(text, bold=False):
    from presentation_gc import _FORMULA_PATTERN
    def replacement(match):
        command = "\\mathbf{" if bold else "\\mathregular{"
        return "$" + command + re.sub(r"(?<=[A-Za-z)\]])(\d+)", r"_{\1}", match.group()) + "}$"
    pattern = re.compile(_FORMULA_PATTERN.pattern + r"|(?<![A-Za-z0-9])Zn\(OAc\)2(?![A-Za-z0-9])")
    return pattern.sub(replacement, text)


def _text_chunks(parts, added=False, omit_slots=()):
    chunks = []
    for part in parts:
        name = part["name"]
        if name == "none" or part["slot"] in omit_slots:
            continue
        changed = bool(added and part["changed"])
        chunks.append((_formula(name, changed), changed))
    return chunks


def _packed_recipe(fig, x, y, width, title, parts, added=False, omit_slots=()):
    """Wrap whole chemical components at their measured widths in one compact table cell."""
    renderer = fig.canvas.get_renderer()
    rows, current, current_width = [], [], 0
    pieces = _text_chunks(parts, added, omit_slots)
    visible_parts = [p for p in parts if p["name"] != "none" and p["slot"] not in omit_slots]
    for part in parts:
        if part["name"] == "none" or part["slot"] in omit_slots:
            fig._recipe_component_audit = getattr(fig, "_recipe_component_audit", []) + [
                {"kind": title, "name": part["name"], "role": part["role"], "source_changed": bool(part["changed"]),
                 "emphasized": False, "absent": part["name"] == "none", "omitted": True, "text": None}]
    def text_area(label, changed=False):
        area = TextArea(label, textprops={"family": "Arial", "size": SMALL,
                                         "color": ORANGE if changed else "black",
                                         "weight": "bold" if changed else "normal"})
        area.set_figure(fig)
        fig._recipe_text_audit = getattr(fig, "_recipe_text_audit", []) + [(area._text, changed, title)]
        return area
    comma = text_area(", ")
    comma_width = comma.get_window_extent(renderer).width
    for (label, changed), part in zip(pieces, visible_parts):
        area = text_area(label, changed)
        fig._recipe_component_audit = getattr(fig, "_recipe_component_audit", []) + [
            {"kind": title, "name": part["name"], "role": part["role"], "source_changed": bool(part["changed"]),
             "emphasized": changed, "absent": False, "omitted": False, "text": area._text}]
        w = area.get_window_extent(renderer).width
        if current and current_width + comma_width + w > width * fig.dpi:
            rows.append(HPacker(children=current, align="baseline", pad=0, sep=0))
            current, current_width = [], 0
        assert w <= width * fig.dpi, (label, w, width * fig.dpi)
        if current:
            current.append(text_area(", "))
            current_width += comma_width
        current.append(area)
        current_width += w
    if current:
        rows.append(HPacker(children=current, align="baseline", pad=0, sep=0))
    box = VPacker(children=rows, align="left", pad=0, sep=1)
    artist = AnnotationBbox(box, (x / W, y / H), xycoords=fig.transFigure, frameon=False,
                            box_alignment=(0, 1), pad=0)
    fig.add_artist(artist)
    return artist


def _place_scheme(fig, drawing, x, top, vectors):
    raster = drawing["raster"]
    width, height = drawing["width"], drawing["height"]
    ax = fig.add_axes([x / W, (top - height) / H, width / W, height / H])
    ax.axis("off")
    image = ax.imshow(raster, interpolation="nearest", aspect="auto")
    ax._chemical_font_audit = [(image, drawing["glyphs"])]
    vectors.append((image, drawing["vector"]))
    alpha = raster.getchannel("A").getbbox()
    return ax, image, {"bond_lengths_px": drawing["bond_lengths_px"], "glyphs": drawing["glyphs"],
                       "bottom_padding_px": raster.height - alpha[3]}


def _panel(fig, case, drawing, left, top, width, vectors, row, col, scheme_height):
    assert drawing["width"] <= width
    slot_top = top - .30
    scheme_top = slot_top - scheme_height + drawing["height"]
    sx = left + (width - drawing["width"]) / 2
    axis, image, probe = _place_scheme(fig, drawing, sx, scheme_top, vectors)
    members = [axis]
    shared_slots = [p["slot"] for p in case["hit_components"] if p["role"] == "solvent" and not p["changed"]]
    shared_solvents = [p["name"] for p in case["hit_components"] if p["slot"] in shared_slots and p["name"] != "none"]
    assert case["hit_temperature_C"] == case["added_temperature_C"]
    assert case["hit_time_h"] == case["added_time_h"]
    shared = ", ".join(shared_solvents + ["%g °C, %g h" % (case["hit_temperature_C"], case["hit_time_h"])])
    conditions_x = sx + drawing["width"] * (.5 if case["panel"] == "b" else drawing["arrow_fraction"])
    shared_text = fig.text(conditions_x / W,
                           (slot_top + .065) / H, _formula(shared), fontsize=SMALL, ha="center", va="bottom")
    members.append(shared_text)
    title = fig.text(0, 0, "C–O (Ullmann library)" if case["transformation"] == "Ullmann" else case["panel_title"], fontsize=LARGE)
    # Native diagrams share a lower edge; all recipe rows share the same baseline.
    recipe_top = slot_top - scheme_height - .065
    recipes, labels, yields = [], [], []
    audit_start = len(getattr(fig, "_recipe_component_audit", []))
    next_top = recipe_top
    for kind, parts, condition, value in (("hit", case["hit_components"], case["hit_condition"], case["m6"]),
                                           ("added", case["added_components"], case["added_condition"], case["m12"])):
        label = fig.text(left / W, next_top / H, "%s (%s)" % (kind, condition), fontsize=SMALL, va="top")
        recipe = _packed_recipe(fig, left + .75, next_top, width - 1.17, kind, parts,
                                added=kind == "added", omit_slots=shared_slots)
        value_text = fig.text((left + width) / W, next_top / H, "%.1f%%" % (100 * value),
                              fontsize=SMALL, fontweight="bold", ha="right", va="top")
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bottom = min(a.get_window_extent(renderer).y0 for a in (label, recipe, value_text)) / DPI
        recipes.append(recipe); labels.append(label); yields.append(value_text)
        next_top = next_top - .185
    for component in fig._recipe_component_audit[audit_start:]:
        component["panel"] = case["panel"]
    note = None
    if case["b"] > case["m12"] + 1e-12:
        note = fig.text(left / W, next_top / H, "library best %.1f%%" % (100 * case["b"]),
                         fontsize=SMALL, va="top")
        members.append(note)
    members.extend(recipes + labels + yields)
    fig.canvas.draw()
    bottom = recipe_top - .48
    motif = fig.add_axes([(left + .75) / W, (bottom - .18) / H, (width - .78) / W, .12 / H])
    motif.set_xlim(0, 100)
    motif.set_ylim(-1, 1)
    motif.axis("off")
    motif.plot([100 * case["m6"], 100 * case["b"]], [0, 0], color=ORANGE, lw=2.0,
               solid_capstyle="butt", zorder=1)
    motif.plot([100 * case["m6"], 100 * case["m12"]], [0, 0], color=ADDED, lw=2.0,
               solid_capstyle="butt", zorder=2)
    motif.plot(100 * case["m6"], 0, "o", ms=3.2, mfc="white", mec="black", mew=.7, zorder=4)
    motif.plot(100 * case["m12"], 0, "D", ms=4.0, mfc=ADDED, mec=ADDED, mew=.6, zorder=3)
    motif.plot(100 * case["b"], 0, "*", ms=6.5, mfc="white", mec="black", mew=.6, zorder=5)
    members.append(motif)
    add_panel(fig, case["panel"], title, members, row=row, col=col)
    fig.canvas.draw()
    bottom = min(a.get_window_extent(fig.canvas.get_renderer()).y0 for a in members) / DPI
    return {"panel": case["panel"], "left": left, "top": top, "width": width,
            "members": members, "title": title, "recipes": recipes, "recipe_labels": labels,
            "recipe_yields": yields, "note": note, "scheme": axis, "image": image,
            "shared_conditions": shared_text, "chemical": [probe], "bottom": bottom,
            "gap_motif": motif, "scheme_slot_height": scheme_height,
            "scheme_slot_padding": 0.,
            "scheme_slot_bbox_px": [left * DPI, (slot_top - scheme_height) * DPI,
                                     (left + width) * DPI, slot_top * DPI]}


def component_facts():
    """Compare added and target-pair-best recipes on identical variable-specific cases."""
    added = pd.read_csv(OUT / "posthit_added_recipes.csv")
    gap = pd.read_csv(OUT / "posthit_gap_recipes.csv")
    summary = json.loads((OUT / "posthit_added_recipes.json").read_text(encoding="utf-8"))
    improved = added[added.added_condition.notna()]
    assert len(improved) == summary["improved_by_history_six"] == 131
    records = []
    for variable, retained in (("metal", True), ("ligand", False), ("base", False)):
        eligible = improved[improved["varies_" + variable].eq(True)]
        best = gap[gap.variable.eq(variable)].merge(
            eligible[["screen", "subpair"]], on=["screen", "subpair"], validate="one_to_one")
        assert set(map(tuple, eligible[["screen", "subpair"]].values)) == set(
            map(tuple, best[["screen", "subpair"]].values))
        denominator = len(eligible)
        changed_added = int(eligible["changed_" + variable].sum())
        assert denominator == summary["variables"][variable]["cases"]
        assert changed_added == summary["variables"][variable]["changed"]
        changed_best = int(best.changed.sum())
        records.append({"variable": variable, "action": "retained" if retained else "changed",
                        "cases": denominator,
                        "added": denominator - changed_added if retained else changed_added,
                        "library_best": denominator - changed_best if retained else changed_best,
                        "source": "out/posthit_added_recipes.csv; out/posthit_gap_recipes.csv",
                        "cohort": "improved severe cases; variable varies within library"})
    return records


def _component_strip(fig):
    facts = component_facts()
    members, axes, labels = [], [], []
    legend_y = .86
    title = fig.text(0, 0, "Recipe changes in the same improved cases", fontsize=LARGE)
    for x, color, label in ((2.08, ADDED, "added six"), (3.31, BEST, "library best")):
        patch = plt.Rectangle((x / W, legend_y / H), .12 / W, .075 / H,
                              transform=fig.transFigure, facecolor=color, edgecolor=color, lw=.6)
        fig.add_artist(patch)
        text = fig.text((x + .17) / W, (legend_y + .0375) / H, label,
                        fontsize=SMALL, va="center")
        members.extend([patch, text])
    for index, fact in enumerate(facts):
        left = .055 + index * 2.16
        group = fig.text((left + .99) / W, .64 / H,
                         fact["variable"] + " " + fact["action"], fontsize=SMALL, ha="center", va="bottom")
        axis = fig.add_axes([(left + .06) / W, .13 / H, 1.93 / W, .42 / H])
        axis.set_xlim(0, 1.31)
        axis.set_ylim(-.4, 1.4)
        axis.axis("off")
        for y, key, color in ((1, "added", ADDED), (0, "library_best", BEST)):
            axis.barh(y, fact[key] / fact["cases"], height=.44, color=color,
                      edgecolor=color, linewidth=.6)
            label = axis.text(1.04, y, "%d/%d" % (fact[key], fact["cases"]),
                              fontsize=SMALL, ha="left", va="center")
            labels.append(label)
        members.extend([group, axis])
        axes.append(axis)
    add_panel(fig, "g", title, members, row=3, col=0)
    fig._recovered_component_strip = {"facts": facts, "title": title, "members": members,
                                      "axes": axes, "labels": labels}


def audit_geometry(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    panels = fig._recovered_recipe_panels
    boxes = []
    geometry = {"structure_scale": STRUCTURE_SCALE, "width_in": W, "height_in": H, "panels": []}
    assert not hasattr(fig, "_recovered_header"), "Figure 4 has no overall in-figure heading"
    for panel in panels:
        titled = next(p for p in fig._panels if p["letter"].get_text() == panel["panel"])
        title_box = titled["title"].get_window_extent(renderer)
        letter_box = titled["letter"].get_window_extent(renderer)
        assert abs(title_box.y1 - letter_box.y1) < .5
        recipe_boxes = [matplotlib.transforms.Bbox.union([a.get_window_extent(renderer) for a in row])
                        for row in zip(panel["recipes"], panel["recipe_labels"], panel["recipe_yields"])]
        for recipe, label, value in zip(panel["recipes"], panel["recipe_labels"], panel["recipe_yields"]):
            cell, lead, tail = [a.get_window_extent(renderer) for a in (recipe, label, value)]
            assert lead.x1 + 4 < cell.x0 and cell.x1 + 4 < tail.x0, (panel["panel"], "recipe table columns overlap")
        assert recipe_boxes[0].y0 > recipe_boxes[1].y1 + 1
        chemical_boxes = [panel["image"].get_window_extent(renderer)]
        gap = chemical_boxes[0].y0 + panel["chemical"][0]["bottom_padding_px"] - recipe_boxes[0].y1
        assert 0 < gap <= .15 * DPI, (panel["panel"], "structure/table gap", gap)
        all_boxes = [title_box, letter_box, *recipe_boxes, *chemical_boxes,
                     panel["shared_conditions"].get_window_extent(renderer)]
        if panel["note"] is not None:
            note_box = panel["note"].get_window_extent(renderer)
            assert note_box.y1 < recipe_boxes[1].y0 - 1
            all_boxes.append(note_box)
        motif_box = panel["gap_motif"].get_window_extent(renderer)
        assert motif_box.y1 < min(box.y0 for box in all_boxes) - 1
        all_boxes.append(motif_box)
        for box in all_boxes:
            assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.width + 1 and box.y1 <= fig.bbox.height + 1
            assert box.x0 >= panel["left"] * DPI - 1
            assert box.x1 <= (panel["left"] + panel["width"]) * DPI + 1, (panel["panel"], box)
        union = matplotlib.transforms.Bbox.union(all_boxes)
        boxes.append(union)
        atom_sizes = [g["font_px"] / DPI * 72 for p in panel["chemical"] for g in p["glyphs"] if g["role"] == "atom"]
        assert min(atom_sizes) >= 8 and max(atom_sizes) <= 9.25, atom_sizes
        geometry["panels"].append({"panel": panel["panel"], "bbox_px": list(union.extents),
                                    "title_top_px": title_box.y1, "letter_top_px": letter_box.y1,
                                    "chemical_recipe_gap_px": gap, "minimum_atom_pt": min(atom_sizes),
                                    "scheme_slot_height_px": panel["scheme_slot_height"] * DPI,
                                    "scheme_slot_padding_px": panel["scheme_slot_padding"] * DPI,
                                    "scheme_slot_bbox_px": panel["scheme_slot_bbox_px"],
                                    "bond_min_px": min(panel["chemical"][0]["bond_lengths_px"]),
                                    "bond_max_px": max(panel["chemical"][0]["bond_lengths_px"]),
                                    "scheme_bbox_px": list(chemical_boxes[0].extents),
                                    "gap_motif_bbox_px": list(motif_box.extents)})
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            assert not a.overlaps(b), (a, b)
    for row in (0, 1, 2):
        tops = [p["title_top_px"] for p in geometry["panels"][row * 2:row * 2 + 2]]
        assert max(tops) - min(tops) < .5
    geometry["row_gaps_px"] = [min(b.y0 for b in boxes[row * 2:row * 2 + 2]) -
                                max(b.y1 for b in boxes[(row + 1) * 2:(row + 2) * 2])
                                for row in (0, 1)]
    assert min(geometry["row_gaps_px"]) >= 12
    strip = fig._recovered_component_strip
    strip_panel = next(p for p in fig._panels if p["letter"].get_text() == "g")
    strip_boxes = [strip["title"].get_window_extent(renderer),
                   strip_panel["letter"].get_window_extent(renderer)]
    strip_boxes.extend(a.get_window_extent(renderer) for a in strip["members"])
    strip_box = matplotlib.transforms.Bbox.union(strip_boxes)
    assert strip_box.y1 + 12 < min(b.y0 for b in boxes), ("gallery/component strip overlap", strip_box)
    for label in strip["labels"]:
        assert label.get_window_extent(renderer).x1 <= fig.bbox.width - 1
    assert abs(strip_boxes[0].y1 - strip_boxes[1].y1) < .5
    geometry["component_strip"] = {"bbox_px": list(strip_box.extents), "facts": strip["facts"],
                                   "gallery_gap_px": min(b.y0 for b in boxes) - strip_box.y1,
                                   "label_alignment_error_px": abs(strip_boxes[0].y1 - strip_boxes[1].y1)}
    geometry["layout"] = "3 rows x 2 columns; same-cohort component strip"
    fig._native_scheme_audit = [{"panel": p["panel"], "page": i + 1,
           "master": str(FIG / "recovered_recipes_schemes.pdf"), "scale": STRUCTURE_SCALE,
           "source_width_pt": (p["scheme_bbox_px"][2] - p["scheme_bbox_px"][0]) / DPI * 72 / STRUCTURE_SCALE,
           "source_height_pt": (p["scheme_bbox_px"][3] - p["scheme_bbox_px"][1]) / DPI * 72 / STRUCTURE_SCALE,
           "placed_bbox_px": p["scheme_bbox_px"], "bond_length_pt": 14.4, "label_pt": 10,
           "printed_label_pt": p["minimum_atom_pt"], "bond_min_px": p["bond_min_px"],
           "bond_max_px": p["bond_max_px"], "gap_in": p["chemical_recipe_gap_px"] / DPI,
           "validated_roles": ["sub_1_smiles", "sub_2_smiles", "product_smiles"]}
          for i, p in enumerate(geometry["panels"])]
    fig._recovered_recipes_geometry = geometry
    return geometry


def build(cases=None):
    """Build the figure without writing any file, including all measured geometry metadata."""
    cases = select_cases() if cases is None else cases
    configure_fonts()
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    vectors, panels = [], []
    drawings = _native_schemes(cases)
    margin, gutter = .055, .14
    width = (W - 2 * margin - gutter) / 2
    row_top = H - .20
    for row in range(3):
        scheme_height = max(1.285, max(drawing["height"] for drawing in drawings[2 * row:2 * row + 2]))
        for col in range(2):
            idx = 2 * row + col
            panels.append(_panel(fig, cases[idx], drawings[idx], margin + col * (width + gutter), row_top,
                                 width, vectors, row, col, scheme_height))
        row_top = min(p["bottom"] for p in panels[-2:]) - .15166666666666667
    fig._chemical_vectors = fig._vector_panels = vectors
    fig._recovered_recipe_panels = panels
    _component_strip(fig)
    align_panels(fig)
    # Formula subscripts add a few pixels to one conditions line; retain one header offset.
    renderer = fig.canvas.get_renderer()
    registered = {p["letter"].get_text(): p for p in fig._panels}
    header_offset = max(registered[p["panel"]]["title"].get_window_extent(renderer).y1 - p["top"] * DPI
                        for p in panels)
    for panel in panels:
        for text in (registered[panel["panel"]]["title"], registered[panel["panel"]]["letter"]):
            x, _ = text.get_position()
            text.set_position((x, (panel["top"] * DPI + header_offset) / fig.bbox.height))
    audit_geometry(fig)
    return fig


def write(fig, stem=STEM):
    """Preserve the native vector chemistry in PDF while emitting the matching 300 dpi PNG."""
    import pymupdf
    FIG.mkdir(exist_ok=True)
    png = io.BytesIO()
    fig.savefig(png, format="png", dpi=DPI)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placements = []
    for image, vector in fig._chemical_vectors:
        box = image.get_window_extent(renderer)
        placements.append((pymupdf.Rect(box.x0 / DPI * 72, (fig.bbox.height - box.y1) / DPI * 72,
                                        box.x1 / DPI * 72, (fig.bbox.height - box.y0) / DPI * 72), vector))
        image.set_visible(False)
    pdf = io.BytesIO()
    try:
        fig.savefig(pdf, format="pdf", dpi=DPI)
    finally:
        for image, _ in fig._chemical_vectors:
            image.set_visible(True)
    with pymupdf.open(stream=pdf.getvalue(), filetype="pdf") as doc:
        for rect, (master, page_index) in placements:
            with pymupdf.open(master) as source:
                assert not source[page_index].get_images()
                doc[0].show_pdf_page(rect, source, page_index)
        assert not doc[0].get_images(), "chemical drawings must remain vector paths"
        payload = doc.tobytes(garbage=4, deflate=True, no_new_id=True)
    (FIG / (stem + ".png")).write_bytes(png.getvalue())
    (FIG / (stem + ".pdf")).write_bytes(payload)


def main():
    cases = select_cases()
    summary = write_data(cases)
    figure = build(cases)
    write(figure)
    print(json.dumps({"facts": summary["facts"], "sentence": summary["sentence"],
                      "geometry": figure._recovered_recipes_geometry}, ensure_ascii=False, indent=2))
    plt.close(figure)


if __name__ == "__main__":
    main()
