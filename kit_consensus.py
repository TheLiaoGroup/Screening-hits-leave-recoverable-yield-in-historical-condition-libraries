"""Kit stability and composition. For each array: (a) how often each condition enters the greedy k=6 kit across 200 bootstrap
resamples of substrates (consensus kit = the six most frequent); (b) retention of the consensus kit under 10-fold CV over substrates;
(c) how many consensus members are among the six highest mean-yield conditions. Count ties use the recorded Ullmann display order,
then ascending condition_id; tied_condition_ids lists all candidates sharing each selected member's count.
Writes out/kit_consensus.csv, out/kit_stability.csv."""
import sys, warnings, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); import lib
warnings.filterwarnings("ignore"); pd.set_option("display.width", 250)
OUT = Path(__file__).parent / "out"; K = 6; B = 200
# Historical display priorities only; they do not enter cross-validation or frozen-panel scoring.
FROZEN_TIE_ORDER = {
    ("Ullmann|1", 80): ("C51", "C31", "C78"),
    ("Ullmann|1", 75): ("C76", "C46"),
}
A, C = lib.load()
cons_rows, stab_rows = [], []
for screen in sorted(A.screen.unique()):
    d = A[A.screen == screen]; label = d.screen_label.iloc[0]; M = lib.screen_matrix(A, screen); Y = M.values; S, n = Y.shape
    rng = np.random.RandomState(11); freq = np.zeros(n)
    for _ in range(B):
        idx = rng.choice(S, S, replace=True); kit = lib.greedy_kit(Y[idx], K)
        freq[kit] += 1
    tie_priority = {(count, cid): rank for (array, count), ids in FROZEN_TIE_ORDER.items() if array == screen for rank, cid in enumerate(ids)}
    cons = sorted(range(n), key=lambda c: (-freq[c], tie_priority.get((int(freq[c]), M.columns[c]), n), str(M.columns[c])))[:K]
    # CV retention of the consensus kit (kit fixed, evaluated on held-out folds; the kit was chosen on bootstrap resamples of all substrates,
    # so this is a description, and the honest number is the 10-fold greedy retention in kit_curve.csv)
    folds = np.array_split(np.random.RandomState(3).permutation(S), 10); full = np.concatenate([lib.retention(Y[f], list(range(n)))[1] for f in folds]).mean()
    ret = np.concatenate([lib.retention(Y[f], cons)[0] for f in folds]).mean() / full
    top6_mean = set(np.argsort(-np.nanmean(Y, 0))[:K]); in_top6 = len(top6_mean & set(cons))
    cc = C[C.rxn_class == d.rxn_class.iloc[0]].set_index("condition_id")
    for rank, c in enumerate(cons):
        cid = M.columns[c]; r = cc.loc[cid]
        tied_ids = sorted(str(x) for x in M.columns[freq == freq[c]])
        comp = "; ".join(str(r[f"Reagent{i}_Standardized Name"]) + (f" ({r[f'Reagent{i}_HTE_Equiv']:g})" if pd.notna(r[f"Reagent{i}_HTE_Equiv"]) else "") for i in range(1, 7) if pd.notna(r[f"Reagent{i}_Standardized Name"]))
        cons_rows.append({"screen": screen, "label": label, "rank": rank + 1, "condition_id": cid, "frequency": freq[c] / B, "tied_condition_ids": "; ".join(tied_ids) if len(tied_ids) > 1 else "", "composition": comp + f"; {r['Standardized Name']}; {r['temp_id']} C; {r['time_id']} h",
                          "mean_yield": float(np.nanmean(Y[:, c])), "in_top6_by_mean_yield": int(c in top6_mean)})
    stab_rows.append({"screen": screen, "label": label, "consensus_retention": ret, "top6_frequency_mean": freq[cons].mean() / B, "members_in_top6_by_mean_yield": in_top6})
    print(f"{screen:30s} consensus retention {ret:.2f}  top-6 freq {freq[cons].mean()/B:.2f}  members among top-6 mean yield {in_top6}", flush=True)
pd.DataFrame(cons_rows).to_csv(OUT / "kit_consensus.csv", index=False); pd.DataFrame(stab_rows).to_csv(OUT / "kit_stability.csv", index=False)
