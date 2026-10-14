"""Recipes that the six history-selected further conditions reach for each severe shortfall.

For every analysed hit more than twenty points below its library best, the greedy panel is refitted
on the same ten folds as the post-hit analysis (seed 3), and the best of the seventh to twelfth
conditions is compared with the hit recipe. For reference, each variable is also scored for the condition
that wins a random six-condition extension, conditioned on that extension raising the yield: the exact
probability that condition j is the best of six drawn uniformly from the measured conditions outside the
panel, restricted to conditions above m6. Writes out/posthit_added_recipes.json and .csv.
"""
import json
from math import comb
from pathlib import Path
import numpy as np
import pandas as pd
import lib

OUT = Path(__file__).resolve().parent / "out"
FOLD_SEED, NFOLD, HIT, SEVERE = 3, 10, 0.20, 0.20


def main() -> None:
    a, C = lib.load("analysis")
    rows = []
    for screen in sorted(a.screen.unique()):
        d = a[a.screen == screen]
        m = lib.screen_matrix(a, screen)
        y = m.values
        V = lib.condition_variables(screen, C, d.rxn_class.iloc[0], list(m.columns))
        folds = np.array_split(np.random.RandomState(FOLD_SEED).permutation(len(m)), NFOLD)
        for test in folds:
            kit = lib.greedy_kit(y[np.setdiff1d(np.arange(len(m)), test)], 12)
            for i in test:
                yi = np.where(np.isnan(y[i]), -np.inf, y[i])
                b, m6, m12 = (max(0.0, float(v)) for v in (yi.max(), yi[kit[:6]].max(), yi[kit[:12]].max()))
                if m6 < HIT or b - m6 <= SEVERE:
                    continue
                hit = m.columns[kit[:6][int(np.argmax(yi[kit[:6]]))]]
                added = m.columns[kit[6:12][int(np.argmax(yi[kit[6:12]]))]] if m12 > m6 else None
                r = {"screen": screen, "subpair": m.index[i], "b": b, "m6": m6, "m12": m12,
                     "hit_condition": hit, "added_condition": added}
                rest = [j for j in range(len(m.columns)) if j not in kit[:6] and np.isfinite(yi[j])]
                order = sorted(rest, key=lambda j: yi[j])
                k = min(6, len(order))
                weight = {j: comb(pos, k - 1) for pos, j in enumerate(order) if yi[j] > m6 and pos >= k - 1}
                total = sum(weight.values())
                for v in V.columns:
                    r["varies_" + v] = bool(V[v].nunique() > 1)
                    r["changed_" + v] = bool(V.at[hit, v] != V.at[added, v]) if added is not None else None
                    r["random_changed_" + v] = (sum(w for j, w in weight.items() if V.iloc[j][v] != V.at[hit, v]) / total
                                                if total else None)
                rows.append(r)
    T = pd.DataFrame(rows)
    assert len(T) == 184, len(T)
    imp = T[T.added_condition.notna()]
    out = {"severe_cases": len(T), "improved_by_history_six": len(imp),
           "within_20_after_twelve": int((T.b - T.m12 <= SEVERE).sum()),
           "fully_closed": int(np.isclose(T.b, T.m12, rtol=0, atol=1e-12).sum()),
           "distinct_added_recipes": int(imp[["screen", "added_condition"]].drop_duplicates().shape[0]),
           "variables": {}}
    for v in sorted(c[len("changed_"):] for c in T.columns if c.startswith("changed_")):
        g = imp[imp["varies_" + v].astype("boolean").fillna(False).astype(bool)]
        out["variables"][v] = {"cases": int(len(g)), "changed": int(g["changed_" + v].astype(bool).sum()),
                               "random_improving_changed": float(g["random_changed_" + v].astype(float).sum())}
    T.to_csv(OUT / "posthit_added_recipes.csv", index=False, lineterminator="\n")
    (OUT / "posthit_added_recipes.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
