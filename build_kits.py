"""Per-array summary for the kit paper: kit size for 90% retention (k90), retention at k = 1/3/6, random baseline, rescue at 20% under
10-fold CV, interaction share and live fraction of the array, and their correlations with k90. Reads out/kit_curve.csv (from kits.py).
Writes out/kit_summary.csv and out/kit_correlates.json."""
import sys, json, warnings, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
sys.path.insert(0, str(Path(__file__).parent)); import lib
warnings.filterwarnings("ignore"); pd.set_option("display.width", 250)
OUT = Path(__file__).parent / "out"; OUT.mkdir(exist_ok=True)
A, C = lib.load(); KC = pd.read_csv(OUT / "kit_curve.csv"); rows = []
for screen in sorted(A.screen.unique()):
    d = A[A.screen == screen]; M = lib.screen_matrix(A, screen); Y = M.values; S, n = Y.shape
    R = lib.twoway(Y); ok = ~np.isnan(Y)
    inter_share = np.nansum(R ** 2) / np.sum((Y[ok] - Y[ok].mean()) ** 2)
    best = np.nanmax(Y, 1); live = (best >= 0.05).mean(); live20 = (best >= 0.2).mean()
    kc = KC[KC.screen == screen].set_index("k")
    k90 = int(kc.index[kc.retention_greedy >= 0.90].min()) if (kc.retention_greedy >= 0.90).any() else 13
    k90_my = int(kc.index[kc.retention_meanyield >= 0.90].min()) if (kc.retention_meanyield >= 0.90).any() else 13
    # rescue under 10-fold CV: pairs reaching 20% with the k-kit vs with the full array
    folds = np.array_split(np.random.RandomState(3).permutation(S), 10); reach = {3: 0, 6: 0}; reach_full = 0
    for f in folds:
        tr = np.setdiff1d(np.arange(S), f); kit = lib.greedy_kit(Y[tr], 6)
        full = lib.retention(Y[f], list(range(n)))[1]; reach_full += int((full >= 0.2).sum())
        for k in (3, 6): reach[k] += int((lib.retention(Y[f], kit[:k])[0] >= 0.2).sum())
    rows.append({"screen": screen, "label": d.screen_label.iloc[0], "pairs": S, "conditions": n, "mean_yield": float(np.nanmean(Y)),
                 "interaction_share": inter_share, "live_frac_5": live, "live_frac_20": live20, "mean_best_full": float(np.nanmean(best)),
                 "ret_k1": kc.loc[1, "retention_greedy"], "ret_k3": kc.loc[3, "retention_greedy"], "ret_k6": kc.loc[6, "retention_greedy"], "ret_k12": kc.loc[12, "retention_greedy"],
                 "rand_k3": kc.loc[3, "retention_random"], "rand_k6": kc.loc[6, "retention_random"],
                 "my_k1": kc.loc[1, "retention_meanyield"], "my_k3": kc.loc[3, "retention_meanyield"],
                 "my_k6": kc.loc[6, "retention_meanyield"], "my_k12": kc.loc[12, "retention_meanyield"], "k90": k90, "k90_meanyield": k90_my,
                 "reach20_k3": reach[3], "reach20_k6": reach[6], "reach20_full": reach_full})
T = pd.DataFrame(rows); T.to_csv(OUT / "kit_summary.csv", index=False)
print(T[["label", "pairs", "conditions", "interaction_share", "live_frac_5", "ret_k1", "ret_k3", "ret_k6", "rand_k6", "k90", "reach20_k6", "reach20_full"]].round(2).to_string(index=False))
Tk = T[T.k90 <= 12]
cor = {"n_arrays_k90_within_12": int(len(Tk)), "median_k90": float(T.k90.median()), "median_ret_k1": float(T.ret_k1.median()),
       "median_ret_k3": float(T.ret_k3.median()), "median_ret_k6": float(T.ret_k6.median()), "median_rand_k6": float(T.rand_k6.median()),
       "spearman_k90_interaction": stats.spearmanr(T.k90, T.interaction_share)[0], "p_k90_interaction": stats.spearmanr(T.k90, T.interaction_share)[1],
       "spearman_k90_live5": stats.spearmanr(T.k90, T.live_frac_5)[0], "p_k90_live5": stats.spearmanr(T.k90, T.live_frac_5)[1],
       "spearman_k90_conditions": stats.spearmanr(T.k90, T.conditions)[0], "p_k90_conditions": stats.spearmanr(T.k90, T.conditions)[1],
       "spearman_k90_pairs": stats.spearmanr(T.k90, T.pairs)[0], "p_k90_pairs": stats.spearmanr(T.k90, T.pairs)[1],
       "spearman_ret6_interaction": stats.spearmanr(T.ret_k6, T.interaction_share)[0], "p_ret6_interaction": stats.spearmanr(T.ret_k6, T.interaction_share)[1],
       "spearman_ret6_live5": stats.spearmanr(T.ret_k6, T.live_frac_5)[0], "p_ret6_live5": stats.spearmanr(T.ret_k6, T.live_frac_5)[1],
       "reach20_k6_total": int(T.reach20_k6.sum()), "reach20_full_total": int(T.reach20_full.sum()), "reach20_k3_total": int(T.reach20_k3.sum())}
# the obvious kit (highest mean yield) against the greedy kit, in retention and in yield points
d6 = T.ret_k6 - T.my_k6; pts6 = 100 * d6 * T.mean_best_full
cor.update({"median_my_k1": float(T.my_k1.median()), "median_my_k3": float(T.my_k3.median()),
            "median_my_k6": float(T.my_k6.median()), "median_my_k12": float(T.my_k12.median()),
            "median_k90_meanyield": float(T.k90_meanyield.median()),
            "greedy_beats_my_arrays": int((d6 > 0).sum()),
            "p_greedy_vs_my": float(stats.wilcoxon(T.ret_k6, T.my_k6).pvalue),
            "median_gain_points": float(pts6.median()), "max_gain_points": float(pts6.max()), "min_gain_points": float(pts6.min()),
            "median_gain_points_random": float((100 * (T.ret_k6 - T.rand_k6) * T.mean_best_full).median())})
# where choosing pays: the chosen-minus-random gap against how much of the array is alive
gap = T.ret_k6 - T.rand_k6
rg, pg = stats.spearmanr(gap, T.mean_yield)
cor.update({"spearman_gap6_meanyield": float(rg), "p_gap6_meanyield": float(pg)})
# arrays that are not on the floor: mean yield of at least 5%
NF = T[T.mean_yield >= 0.05]
cor.update({"n_nonfloor": int(len(NF)), "nonfloor_ret_k1": float(NF.ret_k1.median()),
            "nonfloor_ret_k3": float(NF.ret_k3.median()), "nonfloor_ret_k6": float(NF.ret_k6.median()),
            "nonfloor_rand_k6": float(NF.rand_k6.median()), "nonfloor_my_k6": float(NF.my_k6.median())})
# the repeatability of a well: Hiyama C43 and C46 are identical in composition, and C46 is in no array matrix
Q, _ = lib.load("replicate")
H = lib.screen_matrix(A, "Hiyama|1")["C43"]
P = Q.set_index("subpair")["yield"].reindex(H.index)
ok = H.notna() & P.notna()
cor.update({"replicate_pairs": int(ok.sum()), "replicate_r": float(np.corrcoef(H[ok], P[ok])[0, 1]),
            "replicate_mad_points": float(100 * np.abs(H[ok] - P[ok]).mean())})
json.dump(cor, open(OUT / "kit_correlates.json", "w"), indent=1); print(json.dumps(cor, indent=1))
