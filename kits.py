"""How many conditions a transformation needs: cross-validated greedy kits on every array.
For k = 1..12: retention of mean best yield over held-out substrates (10-fold) by the greedy kit chosen on training substrates,
against two baselines chosen on the same training substrates - the k highest-mean-yield conditions, and random kits of the same
size; plus the identity of the k = 6 kit chosen on the full array.
Writes out/kit_curve.csv and out/kit_members.csv."""
import sys, warnings, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); import lib
warnings.filterwarnings("ignore"); pd.set_option("display.width", 250)
OUT = Path(__file__).parent / "out"; OUT.mkdir(exist_ok=True)
A, C = lib.load(); KS = list(range(1, 13)); rows = []; members = []
for screen in sorted(A.screen.unique()):
    d = A[A.screen == screen]; label = d.screen_label.iloc[0]; M = lib.screen_matrix(A, screen); Y = M.values; S = Y.shape[0]
    rng = np.random.RandomState(3); folds = np.array_split(rng.permutation(S), 10)
    full_all = []; greedy = {k: [] for k in KS}; random = {k: [] for k in KS}; meanyield = {k: [] for k in KS}
    for f in folds:
        tr = np.setdiff1d(np.arange(S), f); Ytr, Yte = Y[tr], Y[f]
        _, full = lib.retention(Yte, list(range(Y.shape[1]))); full_all.append(full)
        kit12 = lib.greedy_kit(Ytr, max(KS))
        rank = np.argsort(-np.nanmean(np.where(np.isnan(Ytr), 0.0, Ytr), 0))   # the obvious kit: highest mean yield on the training pairs
        for k in KS:
            greedy[k].append(lib.retention(Yte, kit12[:k])[0])
            meanyield[k].append(lib.retention(Yte, list(rank[:k]))[0])
            random[k].append(np.mean([lib.retention(Yte, list(rng.choice(Y.shape[1], k, replace=False)))[0] for _ in range(50)], 0))
    full = np.concatenate(full_all).mean()
    for k in KS:
        rows.append({"screen": screen, "label": label, "k": k, "retention_greedy": np.concatenate(greedy[k]).mean() / full,
                     "retention_random": np.concatenate(random[k]).mean() / full,
                     "retention_meanyield": np.concatenate(meanyield[k]).mean() / full, "mean_best_full": full})
    kit = lib.greedy_kit(Y, 6); cc = C[C.rxn_class == d.rxn_class.iloc[0]].set_index("condition_id")
    for rank, c in enumerate(kit):
        cid = M.columns[c]; r = cc.loc[cid]
        members.append({"screen": screen, "label": label, "rank": rank + 1, "condition_id": cid,
                        "composition": "; ".join(str(r[f"Reagent{i}_Standardized Name"]) for i in range(1, 7) if pd.notna(r[f"Reagent{i}_Standardized Name"])) + f"; {r['Standardized Name']}",
                        "mean_yield": float(np.nanmean(Y[:, c]))})
    print(screen, " ".join(f"k{k}:{np.concatenate(greedy[k]).mean()/full:.2f}" for k in (1, 3, 6, 10, 12)), flush=True)
pd.DataFrame(rows).to_csv(OUT / "kit_curve.csv", index=False); pd.DataFrame(members).to_csv(OUT / "kit_members.csv", index=False)
T = pd.DataFrame(rows); print(T.pivot(index="label", columns="k", values="retention_greedy").round(2).to_string())
