"""Kits tested on the sealed hold-out. For the eight arrays with sealed pairs (the fifth of substrates least similar to the rest,
never used in any analysis), the greedy kit of size k chosen on the analysis set is evaluated on the sealed pairs:
retention of mean best yield relative to the full array on those pairs, against random kits. Writes out/kit_sealed.csv."""
import sys, warnings, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); import lib
warnings.filterwarnings("ignore"); pd.set_option("display.width", 250)
OUT = Path(__file__).parent / "out"; OUT.mkdir(exist_ok=True)
A, C = lib.load(); S, _ = lib.load("sealed")
rows = []; rng = np.random.RandomState(7)
for screen in sorted(S.screen.unique()):
    M = lib.screen_matrix(A, screen); Y = M.values; conds = list(M.columns)
    Ms = S[S.screen == screen].pivot_table(index="subpair", columns="condition_id", values="yield", aggfunc="mean").reindex(columns=conds)
    Ys = Ms.values; kit12 = lib.greedy_kit(Y, 12)
    _, full = lib.retention(Ys, list(range(len(conds))))
    for k in (1, 3, 6, 12):
        b, _ = lib.retention(Ys, kit12[:k])
        rnd = np.mean([lib.retention(Ys, list(rng.choice(len(conds), k, replace=False)))[0].mean() for _ in range(200)])
        rows.append({"screen": screen, "label": A[A.screen == screen].screen_label.iloc[0], "sealed_pairs": Ys.shape[0], "k": k,
                     "retention_sealed": b.mean() / full.mean(), "retention_random": rnd / full.mean(),
                     "pairs_reaching_20_kit": int((b >= 0.2).sum()), "pairs_reaching_20_full": int((full >= 0.2).sum()), "mean_best_full": full.mean()})
T = pd.DataFrame(rows); T.to_csv(OUT / "kit_sealed.csv", index=False)
print(T.pivot(index="label", columns="k", values="retention_sealed").round(2).to_string())
print(T[T.k == 6][["label", "sealed_pairs", "retention_sealed", "retention_random", "pairs_reaching_20_kit", "pairs_reaching_20_full"]].round(2).to_string(index=False))
