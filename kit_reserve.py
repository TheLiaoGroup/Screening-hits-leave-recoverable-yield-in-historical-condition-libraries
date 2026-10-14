"""Kits tested on the reserve: substrates the frozen partition never drew into the analysis set.

Eight arrays carry a structure-dissimilar sealed hold-out; the other ten were analysed on a 70% random
sample of their substrates, and the rest are the reserve. Together the two cover every array. The kit
of size k is chosen on the analysis pairs and scored on the reserve pairs of the same array, against
random kits. Writes out/kit_reserve.csv.
"""
import sys, warnings, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); import lib
warnings.filterwarnings("ignore"); pd.set_option("display.width", 250)
OUT = Path(__file__).parent / "out"; OUT.mkdir(exist_ok=True)
MIN_PAIRS = 8

A, C = lib.load(); R, _ = lib.load("reserve")
rows = []; rng = np.random.RandomState(7)
for screen in sorted(R.screen.unique()):
    M = lib.screen_matrix(A, screen); conds = list(M.columns); Y = M.values
    Mr = R[R.screen == screen].pivot_table(index="subpair", columns="condition_id", values="yield", aggfunc="mean").reindex(columns=conds)
    Yr = Mr.values; kit12 = lib.greedy_kit(Y, 12)
    full = lib.retention(Yr, list(range(len(conds))))[1]
    cov = float(np.isfinite(np.where(np.isnan(Yr), np.nan, Yr)).mean())
    for k in (1, 3, 6, 12):
        b = lib.retention(Yr, kit12[:k])[0]
        rnd = np.mean([lib.retention(Yr, list(rng.choice(len(conds), k, replace=False)))[0].mean() for _ in range(200)])
        rows.append({"screen": screen, "label": A[A.screen == screen].screen_label.iloc[0], "reserve_pairs": Yr.shape[0],
                     "completeness": cov, "k": k, "retention_reserve": b.mean() / full.mean(), "retention_random": rnd / full.mean(),
                     "pairs_reaching_20_kit": int((b >= 0.2).sum()), "pairs_reaching_20_full": int((full >= 0.2).sum()), "mean_best_full": full.mean()})
T = pd.DataFrame(rows); T.to_csv(OUT / "kit_reserve.csv", index=False)
print(T.pivot(index="label", columns="k", values="retention_reserve").round(2).to_string())
big = T[(T.k == 6) & (T.reserve_pairs >= MIN_PAIRS)]
print("\nk = 6, arrays with at least %d reserve pairs (%d of %d):" % (MIN_PAIRS, len(big), (T.k == 6).sum()))
print(big[["label", "reserve_pairs", "completeness", "retention_reserve", "retention_random", "pairs_reaching_20_kit", "pairs_reaching_20_full"]].round(2).to_string(index=False))
print("\nmedian retention at six over those arrays: %.2f (random %.2f); pooled pairs %d" % (
    big.retention_reserve.median(), big.retention_random.median(), int(big.reserve_pairs.sum())))
