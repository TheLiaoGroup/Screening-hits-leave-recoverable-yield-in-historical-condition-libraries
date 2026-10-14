"""Which axis must a six-condition screen spend its wells on?

For each array and each condition variable, the kit is rebuilt under two constraints and rescored on
held-out substrates: FIXED, all six members share one level of that variable (a solvent sweep, one
catalyst throughout); DIVERSE, every member takes a different level. A variable with fewer than six
levels cannot support a diverse kit of six, so the realised size is recorded and short kits are
excluded from the comparison rather than counted as a loss. Writes out/kit_axes.csv.
"""
import sys, warnings, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); import lib
warnings.filterwarnings("ignore"); pd.set_option("display.width", 250)
OUT = Path(__file__).parent / "out"; OUT.mkdir(exist_ok=True)
K = 6; NFOLD = 10; SEED = 3

A, C = lib.load(); rows = []
for screen in sorted(A.screen.unique()):
    d = A[A.screen == screen]; label = d.screen_label.iloc[0]
    M = lib.screen_matrix(A, screen); Y = M.values; S, n = Y.shape
    V = lib.condition_variables(screen, C, d.rxn_class.iloc[0], list(M.columns))
    mbf = float(np.nanmean(np.nanmax(Y, 1)))
    rng = np.random.RandomState(SEED); folds = np.array_split(rng.permutation(S), NFOLD)
    acc = {"free": []}; full_all = []
    for v in V.columns: acc["fixed:" + v] = []; acc["diverse:" + v] = []
    sizes = {"free": K}
    for f in folds:
        tr = np.setdiff1d(np.arange(S), f); Ytr, Yte = Y[tr], Y[f]
        full_all.append(lib.retention(Yte, list(range(n)))[1])
        acc["free"].append(lib.retention(Yte, lib.greedy_kit(Ytr, K))[0])
        for v in V.columns:
            lab = V[v].values
            div = lib.greedy_kit(Ytr, K, distinct=lab)
            sizes["diverse:" + v] = len(div)
            acc["diverse:" + v].append(lib.retention(Yte, div)[0])
            best, bestval = [], -1.0
            for lev in pd.unique(lab):
                mask = lab == lev
                if mask.sum() < K: continue
                kit = lib.greedy_kit(Ytr, K, allowed=mask)
                val = float(lib.retention(Ytr, kit)[0].mean())
                if val > bestval: best, bestval = kit, val
            sizes["fixed:" + v] = len(best)
            acc["fixed:" + v].append(lib.retention(Yte, best)[0] if best else np.zeros(len(f)))
    full = np.concatenate(full_all).mean()
    for key, vals in acc.items():
        var = key.split(":")[1] if ":" in key else ""
        rows.append({"screen": screen, "label": label, "kit": key, "variable": var,
                     "levels": int(V[var].nunique()) if var else n, "kit_size": sizes.get(key, K),
                     "retention": np.concatenate(vals).mean() / full, "mean_best_full": mbf})
    print(screen, "done", flush=True)
T = pd.DataFrame(rows); T.to_csv(OUT / "kit_axes.csv", index=False)
free = T[T.kit == "free"].set_index("screen")
out = []
for kind in ("fixed", "diverse"):
    sub = T[T.kit.str.startswith(kind + ":")]
    sub = sub[sub.kit_size == K]                    # a constraint that cannot fill six wells is not a comparison
    for v, g in sub.groupby("variable"):
        g = g.set_index("screen")
        delta = g.retention - free.retention.reindex(g.index)
        pts = 100 * delta * g.mean_best_full
        out.append({"constraint": kind, "variable": v, "arrays": len(g),
                    "median_retention_loss": -delta.median(), "median_points_lost": -pts.median(),
                    "worst_points_lost": -pts.min()})
Q = pd.DataFrame(out).sort_values(["constraint", "median_points_lost"], ascending=[True, False])
Q.to_csv(OUT / "kit_axes_summary.csv", index=False)
print(Q.round(2).to_string(index=False))
