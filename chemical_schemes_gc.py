"""Fixed-scale reaction drawings and editable MOL exports for Figure 1a."""
import io
import math
import re
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from matplotlib import font_manager
from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdAbbreviations, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

HERE = Path(__file__).resolve().parent
ARIAL = Path(font_manager.findfont(font_manager.FontProperties(family="Arial"), fallback_to_default=False))
DPI = 300
SOURCE_WIDTH_IN = 7.0
PRINT_WIDTH_IN = 6.0
PRINT_BOND_PT = 14.4 * 0.75
PRINT_FONT_PT = 8.1
PRINT_LINE_PT = 0.52
BOND_PX = PRINT_BOND_PT / 72 * DPI * SOURCE_WIDTH_IN / PRINT_WIDTH_IN
FONT_PX = PRINT_FONT_PT / 72 * DPI * SOURCE_WIDTH_IN / PRINT_WIDTH_IN
LINE_PX = PRINT_LINE_PT / 72 * DPI * SOURCE_WIDTH_IN / PRINT_WIDTH_IN
ROLES = ("reactant1", "reactant2", "product")


def export_mols():
    """Preserve full chemical identity in MOL files; abbreviations are drawing-only."""
    examples = pd.read_csv(HERE / "reaction_schemes_gc.csv")
    dest = HERE / "figures" / "schemes_mol"
    dest.mkdir(exist_ok=True)
    lines = ["# Editable structures for Figure 1a", "",
             "The 14 transformation examples represent 18 arrays. These MOL files preserve the full SMILES structures.",
             "Open them in ChemDraw and apply ACS Document 1996 at 100%: bond length 0.508 cm; line width 0.021 cm; bold bond 0.071 cm; atom margin 0.056 cm; hash spacing 0.088 cm; double-bond spacing 18%; Arial 10 pt.",
             "Export the assembled panel as `figures/transformation_schemes.cdxml` and `figures/transformation_schemes.pdf` (or PNG). The figure generator uses the assembled panel when available, otherwise its fixed-scale RDKit version.", "",
             "| File | Transformation | Role | SMILES | Source array | Example condition |",
             "| --- | --- | --- | --- | --- | --- |"]
    for number, row in enumerate(examples.itertuples(), 1):
        slug = re.sub(r"[^A-Za-z0-9]+", "_", row.transformation).strip("_")
        for role in ROLES:
            smiles = getattr(row, role + "_smiles")
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None, (row.transformation, role, smiles)
            rdDepictor.Compute2DCoords(mol)
            name = f"{number:02d}_{slug}_{role}.mol"
            (dest / name).write_text(Chem.MolToMolBlock(mol), encoding="ascii")
            lines.append(f"| {name} | {row.transformation} | {role} | `{smiles}` | `{row.source_array}` | {row.example_condition_id} |")
    (dest / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest


def drawing_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    if smiles == "C[Mg]Br":
        mol = Chem.MolFromSmiles("C")
        mol.GetAtomWithIdx(0).SetProp("_displayLabel", "MeMgBr")
    else:
        nitro = [a for a in rdAbbreviations.GetDefaultAbbreviations() if a.label == "NO2"]
        custom = rdAbbreviations.ParseAbbreviations(
            "Si(OMe)3 *[Si](OC)(OC)OC Si(OMe)<sub>3</sub> (MeO)<sub>3</sub>Si\n"
            "CMe2NH2 *C(C)(C)N CMe<sub>2</sub>NH<sub>2</sub> H<sub>2</sub>NCMe<sub>2</sub>")
        # CMe2NH2 preserves the Sonogashira substrate: substituting t-Bu would change its identity.
        mol = rdAbbreviations.CondenseMolAbbreviations(mol, list(custom) + nitro, maxCoverage=1.0)
    rdDepictor.Compute2DCoords(mol)
    return mol


def _options(drawer):
    opts = drawer.drawOptions()
    opts.useBWAtomPalette()
    # RDKit's fixedBondLength is the scale per coordinate unit; its 2D bonds are 1.5 units.
    opts.fixedBondLength = BOND_PX / 1.5
    opts.fixedFontSize = int(round(FONT_PX))
    opts.minFontSize = opts.maxFontSize = opts.fixedFontSize
    opts.fontFile = str(ARIAL)
    opts.bondLineWidth = LINE_PX
    opts.multipleBondOffset = 0.18
    opts.padding = 0
    opts.clearBackground = False
    opts.drawMolsSameScale = True
    return opts


def _molecule_drawing(smiles, atom_label=None):
    """Draw matching raster and vector versions at one fixed physical scale."""
    if not ARIAL.exists():
        raise FileNotFoundError("Arial is required for the reaction drawings: " + str(ARIAL))
    mol = drawing_molecule(smiles)
    if atom_label is not None:
        assert mol.GetNumAtoms() == 1
        mol.GetAtomWithIdx(0).SetProp("_displayLabel", atom_label)
    xy = np.array([[mol.GetConformer().GetAtomPosition(i).x, mol.GetConformer().GetAtomPosition(i).y]
                   for i in range(mol.GetNumAtoms())])
    scale = BOND_PX / 1.5
    margin = 4 * FONT_PX
    width, height = [int(math.ceil(span * scale + 2 * margin))
                     for span in np.ptp(xy, axis=0)]

    def draw(drawer):
        _options(drawer)
        drawer.DrawMolecule(mol)
        lengths = []
        for bond in mol.GetBonds():
            a, b = drawer.GetDrawCoords(bond.GetBeginAtomIdx()), drawer.GetDrawCoords(bond.GetEndAtomIdx())
            lengths.append(math.hypot(a.x - b.x, a.y - b.y))
        drawer.FinishDrawing()
        return lengths

    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    lengths = draw(drawer)
    image = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGBA")
    bbox = image.getbbox()
    assert bbox is not None, smiles
    bbox = (max(0, bbox[0] - 3), max(0, bbox[1] - 3), min(width, bbox[2] + 3), min(height, bbox[3] + 3))
    image = image.crop(bbox)
    svg = rdMolDraw2D.MolDraw2DSVG(width, height, -1, -1, True)
    draw(svg)
    glyphs = []
    for text in ET.fromstring(svg.GetDrawingText()).iter("{http://www.w3.org/2000/svg}text"):
        style = dict(item.split(":", 1) for item in text.attrib["style"].split(";") if ":" in item)
        font_px = float(style["font-size"].removesuffix("px"))
        glyphs.append({"font_px": font_px, "text": "".join(text.itertext()),
                       "role": "atom" if abs(font_px - round(FONT_PX)) < 0.1 else "subscript or charge"})
    vector = rdMolDraw2D.MolDraw2DSVG(width, height)
    draw(vector)
    cropped = ET.Element("{http://www.w3.org/2000/svg}g",
                         {"transform": "translate(%s,%s)" % (-bbox[0], -bbox[1])})
    cropped.extend(ET.fromstring(vector.GetDrawingText()))
    return image, lengths, glyphs, cropped


def render_molecule(smiles, atom_label=None):
    """Return one transparent raster, its drawn bond lengths, and glyph metadata."""
    return _molecule_drawing(smiles, atom_label)[:3]


def ink_size(image):
    """Measure actual nontransparent dark pixels, excluding antialiasing fringes."""
    rgba = np.asarray(image.convert("RGBA"))
    yy, xx = np.where((rgba[:, :, 3] > 128) & (rgba[:, :, :3].min(axis=2) < 128))
    assert len(xx), "scale probe has no ink"
    return int(xx.max() - xx.min() + 1), int(yy.max() - yy.min() + 1)


def scale_limits(bond_images, label_images):
    """Rendered probes detect molecule resizing independently of nominal draw settings."""
    bonds = [ink_size(image)[0] for image in bond_images]
    capitals = [ink_size(image)[1] for image in label_images]
    return max(bonds) / min(bonds) <= 1.05, max(capitals) - min(capitals) <= 1


def scheme_drawing(row):
    """Assemble matching raster and vector schemes by translation only."""
    from PIL import ImageDraw, ImageFont
    drawn = [_molecule_drawing(getattr(row, role + "_smiles")) for role in ROLES]
    mols = [part[0] for part in drawn]
    gap, arrow = int(0.10 * DPI), int(0.19 * DPI)
    height = max(image.height for image in mols)
    width = sum(image.width for image in mols) + 2 * gap + arrow
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    root = ET.Element("{http://www.w3.org/2000/svg}svg",
                      {"width": str(width), "height": str(height),
                       "viewBox": "0 0 %s %s" % (width, height)})
    pen = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(str(ARIAL), int(round(FONT_PX)))
    x = 0
    for i, image in enumerate(mols):
        y0 = int((height - image.height) / 2)
        canvas.alpha_composite(image, (x, y0))
        group = ET.SubElement(root, "{http://www.w3.org/2000/svg}g",
                              {"transform": "translate(%s,%s)" % (x, y0)})
        group.append(drawn[i][3])
        x += image.width
        if i == 0:
            pen.text((x + gap / 2, height / 2), "+", font=font, fill="black", anchor="mm")
            plus_image, _, _, plus_vector = _molecule_drawing("C", "+")
            plus = ET.SubElement(root, "{http://www.w3.org/2000/svg}g",
                                 {"transform": "translate(%s,%s)" %
                                  (x + gap / 2 - plus_image.width / 2,
                                   height / 2 - plus_image.height / 2)})
            plus.append(plus_vector)
            x += gap
        elif i == 1:
            start, end = x + gap / 2, x + gap / 2 + arrow
            y = height / 2
            pen.line((start, y, end, y), fill="black", width=max(1, int(round(LINE_PX))))
            pen.line((end - 9, y - 5, end, y, end - 9, y + 5), fill="black", width=max(1, int(round(LINE_PX))))
            ET.SubElement(root, "{http://www.w3.org/2000/svg}path",
                          {"d": "M %s,%s L %s,%s M %s,%s L %s,%s L %s,%s" %
                           (start, y, end, y, end - 9, y - 5, end, y, end - 9, y + 5),
                           "fill": "none", "stroke": "black", "stroke-width": str(LINE_PX)})
            x += gap + arrow
    return (canvas, [length for _, lengths, _, _ in drawn for length in lengths],
            [g for _, _, gs, _ in drawn for g in gs], ET.tostring(root, encoding="utf-8"))


def scheme_raster(row):
    """Expose the raster and audit metadata used by the existing figure checker."""
    return scheme_drawing(row)[:3]


def external_panel():
    """Prefer the author-edited panel without silently recomputing its chemistry."""
    png = HERE / "figures" / "transformation_schemes.png"
    pdf = HERE / "figures" / "transformation_schemes.pdf"
    if pdf.exists():
        import fitz
        with fitz.open(pdf) as document:
            assert len(document) == 1, "Figure 1a external PDF must be a single panel"
            pix = document[0].get_pixmap(dpi=DPI, alpha=True)
            return Image.frombytes("RGBA", [pix.width, pix.height], pix.samples)
    if png.exists():
        return Image.open(png).convert("RGBA")
    return None
