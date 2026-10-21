"""The single substrate pair drawn in Figure 1, chosen by a stated rule and rebuilt from its fold.

Rule, applied to the analysed screening hits (greedy six-condition out-of-fold best m6 >= 0.20):
  1. a severe shortfall, b - m6 > 0.20, at a hit yield of 40-60% (the band with the highest rate);
  2. backed by at least five library recipes that exceed m6 by more than 0.20;
  3. fully recovered by the six further greedy conditions, m12 = b;
  4. the largest such gap b - m6.
The pair's out-of-fold greedy panel is rebuilt with the fold seed and fold count of review_tail_gc.py
and must reproduce the deposited m6, m12 and b. Writes out/example_pair.csv (one row per library
condition of the pair) and out/example_pair.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import lib
from review_tail_gc import FOLD_SEED, NFOLD

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
SEVERE, MIN_SUPPORT = 0.20, 5


def recipe(row, slots=6):
    parts = []
    for j in range(1, slots + 1):
        name = row.get(f"Reagent{j}_Standardized Name")
        if isinstance(name, str) and name.strip():
            equiv = row.get(f"Reagent{j}_HTE_Equiv")
            parts.append(f"{name} ({equiv:g})" if pd.notna(equiv) else name)
    return "; ".join(parts + [str(row["Standardized Name"]), f"{row['temp_id']:g} °C", f"{row['time_id']:g} h"])


def short_recipe(row, slots=6):
    names = [row.get(f"Reagent{j}_Standardized Name") for j in range(1, slots + 1)]
    return ", ".join([n for n in names if isinstance(n, str) and n.strip()] + [str(row["Standardized Name"])])


def select():
    support = pd.read_csv(OUT / "posthit_pair_support.csv")
    c = support[(support.b - support.m6 > SEVERE) & (support.m6 >= 0.40) & (support.m6 < 0.60)
                & (support.conditions_beating_m6_by_20 >= MIN_SUPPORT)
                & np.isclose(support.m12, support.b, atol=1e-12)]
    assert len(c), "no pair satisfies the Figure 1 rule"
    return c.assign(gap=c.b - c.m6).sort_values(["gap", "screen", "subpair"], ascending=[False, True, True]).iloc[0]


def main():
    pick = select()
    a, C = lib.load()
    m = lib.screen_matrix(a, pick.screen)
    y, cols = m.values, list(m.columns)
    i = list(m.index).index(pick.subpair)
    folds = np.array_split(np.random.RandomState(FOLD_SEED).permutation(y.shape[0]), NFOLD)
    fold = next(f for f, test in enumerate(folds) if i in test)
    train = np.setdiff1d(np.arange(y.shape[0]), folds[fold])
    kit = lib.greedy_kit(y[train], 12)
    rank = [c for c in np.argsort(-np.nanmean(y[train], axis=0)) if c not in kit[:6]][:6]
    row = pd.Series(y[i], index=cols)
    m6 = np.nanmax(row.iloc[kit[:6]]); m12 = np.nanmax(row.iloc[kit[:12]]); b = np.nanmax(y[i])
    mean6 = max(m6, np.nanmax(row.iloc[rank]))
    assert np.isclose(m6, pick.m6) and np.isclose(m12, pick.m12) and np.isclose(b, pick.b), "fold rebuild disagrees"
    assert np.isclose(mean6, pick.m6_meanyield_plus6), "historical-mean extension disagrees"
    rxn = a[a.screen == pick.screen].rxn_class.iloc[0]
    cond = C[C.rxn_class == rxn].drop_duplicates("condition_id").set_index("condition_id")
    role = {cols[c]: "initial six (Greedy)" for c in kit[:6]}
    role.update({cols[c]: "six added from history (Greedy)" for c in kit[6:12]})
    rows = []
    for cid in cols:
        if np.isnan(row[cid]):
            continue
        r = cond.loc[cid]
        rows.append({"screen": pick.screen, "subpair": pick.subpair, "condition_id": cid,
                     "role": role.get(cid, "rest of library"),
                     "greedy_order": (kit.index(cols.index(cid)) + 1) if cols.index(cid) in kit[:12] else np.nan,
                     "historical_mean_extension": cid in [cols[c] for c in rank],
                     "yield": float(row[cid]), "recipe": recipe(r), "short_recipe": short_recipe(r)})
    d = pd.DataFrame(rows).sort_values(["role", "yield"], ascending=[True, False])
    d.to_csv(OUT / "example_pair.csv", index=False)
    hit_cid = cols[kit[int(np.nanargmax(row.iloc[kit[:6]].values))]]
    best_cid = row.idxmax()
    rule = " ".join(line.strip() for line in __doc__.split("\n\n")[1].splitlines()[:5])
    summary = {"rule": rule, "screen": pick.screen, "subpair": pick.subpair,
               "sub_1_smiles": pick.subpair.split(" >> ")[0], "sub_2_smiles": pick.subpair.split(" >> ")[1],
               "fold": int(fold), "m6": float(m6), "m12": float(m12), "b": float(b),
               "meanyield_plus6": float(mean6), "random_plus6": float(pick.m6_random_plus6),
               "hit_condition": hit_cid, "best_condition": best_cid,
               "best_condition_greedy_order": int(kit.index(cols.index(best_cid)) + 1),
               "conditions_beating_hit_by_20": int((row > m6 + SEVERE).sum()),
               "measured_conditions": int(row.notna().sum()), "library_conditions": len(cols),
               "hit_recipe": short_recipe(cond.loc[hit_cid]), "best_recipe": short_recipe(cond.loc[best_cid])}
    assert summary["conditions_beating_hit_by_20"] == int(pick.conditions_beating_m6_by_20)
    (OUT / "example_pair.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
