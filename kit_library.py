"""What the eighteen condition libraries are, and whether a kit built on one carries to another.

Five arrays are the same transformation, Buchwald-Hartwig amination, run as five independent campaigns.
If the kit size a transformation needs were a property of the transformation, the five would agree.
  out/kit_library.csv   one row per array x condition variable: how many levels it takes
  out/kit_overlap.csv   within a transformation, conditions two arrays share, at three match levels
  out/kit_transfer.csv  the donor array's kit scored on a recipient array of the same transformation
"""
import sys, warnings, itertools, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); import lib
warnings.filterwarnings("ignore"); pd.set_option("display.width", 250)
OUT = Path(__file__).parent / "out"; OUT.mkdir(exist_ok=True)
NAME = [f"Reagent{i}_Standardized Name" for i in range(1, 7)]
LEVELS = ("reagents", "reagents+solvent", "reagents+solvent+T")


def recipe_keys(cc, level):
    """condition_id -> recipe key: reagent names; + solvent; + temperature."""
    out = {}
    for cid, row in cc.iterrows():
        k = [str(row[c]) for c in NAME if pd.notna(row[c])]
        if level != "reagents": k.append("S:" + str(row["Standardized Name"]))
        if level == "reagents+solvent+T": k.append("T:" + str(row["temp_id"]))
        out[cid] = frozenset(k)
    return out


A, C = lib.load()
mats, keys, lib_rows = {}, {}, []
for screen in sorted(A.screen.unique()):
    d = A[A.screen == screen]; label = d.screen_label.iloc[0]; cls = d.rxn_class.iloc[0]
    M = lib.screen_matrix(A, screen); mats[screen] = M
    cc = C[C.rxn_class == cls].set_index("condition_id").reindex(M.columns)
    keys[screen] = {lv: recipe_keys(cc, lv) for lv in LEVELS}
    V = lib.condition_variables(screen, C, cls, list(M.columns))
    for var in V.columns:
        lib_rows.append({"screen": screen, "label": label, "variable": var, "levels": int(V[var].nunique()), "conditions": M.shape[1]})
L = pd.DataFrame(lib_rows); L.to_csv(OUT / "kit_library.csv", index=False)
LAB = dict(zip(L.screen, L.label))
TR = A.groupby("screen").transformation.first()
groups = [sorted(g.index) for _, g in TR.groupby(TR) if len(g) > 1]
ov = []
for grp in groups:
    for a, b in itertools.combinations(grp, 2):
        for lv in LEVELS:
            sa, sb = set(keys[a][lv].values()), set(keys[b][lv].values())
            ov.append({"transformation": TR[a], "array_a": LAB[a], "array_b": LAB[b], "match": lv,
                       "shared": len(sa & sb), "conditions_a": len(sa), "conditions_b": len(sb)})
OV = pd.DataFrame(ov); OV.to_csv(OUT / "kit_overlap.csv", index=False)
print(OV.pivot_table(index=["array_a", "array_b"], columns="match", values="shared").to_string())
tr = []
for grp in groups:
    for a in grp:
        Ma = mats[a]; kit = [Ma.columns[c] for c in lib.greedy_kit(Ma.values, 6)]
        ka = [keys[a]["reagents"][c] for c in kit]
        for b in grp:
            if a == b: continue
            inv = {}
            for cid, k in keys[b]["reagents"].items(): inv.setdefault(k, cid)
            mapped = [inv[k] for k in ka if k in inv]
            Yb = mats[b].values; full = lib.retention(Yb, list(range(Yb.shape[1])))[1].sum()
            rec = {"transformation": TR[a], "donor": LAB[a], "recipient": LAB[b], "members_found": len(mapped)}
            if mapped:
                idx = [list(mats[b].columns).index(c) for c in mapped]
                rec["retention_transferred"] = lib.retention(Yb, idx)[0].sum() / full
                rec["retention_own_same_size"] = lib.retention(Yb, lib.greedy_kit(Yb, len(mapped)))[0].sum() / full
            tr.append(rec)
TF = pd.DataFrame(tr); TF.to_csv(OUT / "kit_transfer.csv", index=False)
print(TF.round(2).to_string(index=False))
