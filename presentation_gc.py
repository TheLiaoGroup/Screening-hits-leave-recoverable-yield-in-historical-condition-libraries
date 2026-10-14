"""Shared display names and formula typography for manuscript generators."""
import csv
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
with (HERE / "array_labels.csv").open(encoding="utf-8", newline="") as handle:
    _ROWS = list(csv.DictReader(handle))
ARRAY_LABELS = {row["screen"]: row["display_name"] for row in _ROWS}
assert len(ARRAY_LABELS) == len(_ROWS), "duplicate screen in array_labels.csv"
assert len(set(ARRAY_LABELS.values())) == len(_ROWS), "duplicate display name in array_labels.csv"
_ARRAY_NAMES = {name: name for name in ARRAY_LABELS.values()}
_ARRAY_NAMES.update({
    "C–O/C–N selectivity": "C–O coupling",
    "C-O/C-N selectivity": "C–O coupling",
    "C–O (Pd/Cu)": "C–O coupling",
})


def array_label(screen):
    return ARRAY_LABELS[screen]


def array_name(name):
    """Resolve exported display names without changing aggregate row labels."""
    return _ARRAY_NAMES.get(name, ARRAY_LABELS.get(name, name))


def render_array_labels(text):
    return re.sub(r"\{\{array:([^}]+)\}\}", lambda match: array_label(match[1]), text)


_SUBSCRIPT_DIGITS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
_PLAIN_DIGITS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
# Only recorded chemical formulas are eligible; condition IDs, catalyst generations,
# reagent-name locants, temperatures and equivalents remain as deposited.
_FORMULAS = (
    "SnCl2·2H2O", "PdCl2(PCy3)2", "PdCl2(PEt3)2", "PdCl2(PPh3)2",
    "Pd[P(o-tolyl)3]2", "[Pd(cinnamyl)Cl]2", "Cu(MeCN)4BF4",
    "PdCl2(dppf)", "PdCl2(dtbpf)", "Ni(dppf)Cl2", "NiCl2(dppe)",
    "NiCl2(dppp)", "Pd(PPh3)4", "Pd2(dba)3", "Pd(dba)2", "Pd(OAc)2",
    "Cu(OAc)2", "Cu(acac)2", "Ti(OiPr)4", "NaBH3CN", "Cs2CO3",
    "K2CO3", "K3PO4", "KHCO3", "Na2CO3", "Bu4NBr", "MnO2",
    "PdBr2", "PdCl2", "ZnBr2", "ZnCl2", "P(o-tolyl)3", "P(tBu)2Me",
    "P(tBu)3", "PCy3", "iPr2NH", "Et2O", "H2O", "I2",
)
_FORMULA_MAP = {formula: formula.translate(_SUBSCRIPT_DIGITS) for formula in _FORMULAS}
_FORMULA_MAP["SnCl2·2H2O"] = "SnCl₂·2H₂O"
_FORMULA_MAP["P(2,4-Me-Ph)3"] = "P(2,4-Me-Ph)₃"
_FORMULA_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:" + "|".join(re.escape(formula) for formula in
                                      sorted(_FORMULA_MAP, key=len, reverse=True)) + r")(?![A-Za-z0-9])"
)


def chemical_subscripts(text):
    return _FORMULA_PATTERN.sub(lambda match: _FORMULA_MAP[match[0]], text)


def plain_chemical_digits(text):
    return text.translate(_PLAIN_DIGITS)
