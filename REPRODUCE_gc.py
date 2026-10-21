"""Standalone reproduction of the headline numbers of the manuscript.
Reads only the released arrays.csv; imports no analysis code. Recomputes the cross-validated retention at
k = 1, 3, 6, the two baselines at six (the highest-mean-yield kit and a random kit), the pairs reaching 20%
with six conditions, the screening hits that remain more than twenty points below the library best at six
and twelve conditions and the share of their shortfall the nested twelve recovers, the retention at six on
the sealed hold-out of eight arrays and on the reserve of the seven arrays holding at least eight reserve
pairs, and the repeatability of the duplicated Hiyama condition.
It does not recompute the bootstrap interval on the kit size, the condition-library overlap or the kit
transfer; those come from kit_k90ci.py and kit_library.py in the code package."""
import os, sys, numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).parent
DATA = Path(os.environ.get("BH_DATA_DIR", str(HERE / "data")))
def _read():
    """Read arrays.csv locally or from BH_DATA_DIR; support explicitly supplied legacy DATASET_* files."""
    for d in (DATA, HERE / "data", HERE / "02_CODE_AND_DATA" / "data"):
        if (d / "arrays.csv").exists():
            r = pd.read_csv(d / "arrays.csv", low_memory=False).rename(columns={"array_id": "screen"}); r["subpair"] = r.sub_1_smiles + " >> " + r.sub_2_smiles
            r = r[r["measurement_status"].isin(["quantified", "zero_by_rule"])]  # measured cells only
            return r[r.set == "analysis"], r[r.set == "sealed"], r[r.set == "reserve"], r[r.set == "replicate"]
    out = []
    for f in ("DATASET_analysis.csv", "DATASET_validation_sealed.csv"):
        r = pd.read_csv(DATA / f, low_memory=False); r = r[~((r.screen == "Hiyama|1") & (r.condition_id == "C46"))]; out.append(r[r["yield"].notna()])
    return out + [out[0].iloc[:0], out[0].iloc[:0]]
A, S, R, Q = _read()
def greedy(Y, k):
    Yf = np.where(np.isnan(Y), -np.inf, Y); best = np.full(Y.shape[0], -np.inf); chosen = []; allowed = np.ones(Y.shape[1], bool)
    for _ in range(k):
        cand = np.where(allowed)[0]
        gains = [np.mean(np.where(np.isinf(np.maximum(best, Yf[:, c])), 0, np.maximum(best, Yf[:, c]))) for c in cand]
        c = cand[int(np.argmax(gains))]; chosen.append(c); best = np.maximum(best, Yf[:, c]); allowed[c] = False
    return chosen
def bestin(Y, kit):
    Yf = np.where(np.isnan(Y), -np.inf, Y); b = np.max(Yf[:, kit], 1); return np.where(np.isinf(b), 0, b)
POSTHIT = {"hits_k6": 1424, "hits_below80_k6": 956, "severe_k6": 184, "recovery_6_to_12": 0.433}
PAPER = {"median_k1": 0.54, "median_k3": 0.76, "median_k6": 0.85, "median_rand6": 0.59, "reach_k6": 1424, "reach_full": 1541, "reach_k3": 1349, "median_my6": 0.83}
SEALED = {"Buchwald-Hartwig|1": 0.82, "Suzuki|1": 0.83, "Sonogashira|1": 0.80, "SNAr|1": 0.77, "Kumada|1": 0.73, "Grignard|1": 0.84, "Chan-Lam|1": 0.69, "Buchwald-Hartwig|2": 0.66}
RESERVE = {"Acylation|1": 0.89, "Buchwald-Hartwig(CMC)|1": 0.67, "C-O competitive reactions|1": 0.60,
           "Heck|1": 0.83, "Hiyama|1": 0.98, "Reductive-amination|1": 0.96, "Ullmann|1": 0.78}
r1 = []; r3 = []; r6 = []; rr6 = []; my6 = []; reach6 = reach3 = reachf = 0; ok = bad = 0
hits = hits80 = sev6 = 0; gap6 = gain12 = 0.0
for screen in sorted(A.screen.unique()):
    M = A[A.screen == screen].pivot_table(index="subpair", columns="condition_id", values="yield", aggfunc="mean"); Y = M.values; n = Y.shape[1]
    rng = np.random.RandomState(3); folds = np.array_split(rng.permutation(Y.shape[0]), 10)
    acc = {1: [], 3: [], 6: []}; rnd = []; my = []; full_all = []
    for f in folds:
        tr = np.setdiff1d(np.arange(Y.shape[0]), f); kit = greedy(Y[tr], 12); full = bestin(Y[f], list(range(n))); full_all.append(full)
        for k in (1, 3, 6): acc[k].append(bestin(Y[f], kit[:k]))
        m6, m12 = acc[6][-1], bestin(Y[f], kit[:12]); h = m6 >= 0.2
        hits += int(h.sum()); hits80 += int((h & (m6 < 0.8)).sum()); sev6 += int((full - m6 > 0.2)[h].sum())
        gap6 += float((full - m6)[h].sum()); gain12 += float((m12 - m6)[h].sum())
        rnd.append(np.mean([bestin(Y[f], list(rng.choice(n, 6, replace=False))) for _ in range(50)], 0))
        rank = np.argsort(-np.nanmean(Y[tr], 0)); my.append(bestin(Y[f], list(rank[:6])))
        reach6 += int((acc[6][-1] >= 0.2).sum()); reach3 += int((acc[3][-1] >= 0.2).sum()); reachf += int((full >= 0.2).sum())
    fm = np.concatenate(full_all).mean()
    r1.append(np.concatenate(acc[1]).mean() / fm); r3.append(np.concatenate(acc[3]).mean() / fm); r6.append(np.concatenate(acc[6]).mean() / fm); rr6.append(np.concatenate(rnd).mean() / fm); my6.append(np.concatenate(my).mean() / fm)
    if screen in SEALED:
        Ms = S[S.screen == screen].pivot_table(index="subpair", columns="condition_id", values="yield", aggfunc="mean").reindex(columns=M.columns); Ys = Ms.values
        kit = greedy(Y, 12); ret = bestin(Ys, kit[:6]).mean() / bestin(Ys, list(range(n))).mean()
        good = abs(ret - SEALED[screen]) < 0.006; ok += good; bad += not good
        print(f"{'PASS' if good else 'FAIL'} sealed k=6 {screen:22s} {ret:.3f} (paper {SEALED[screen]:.2f})")
    if screen in RESERVE:
        Mr = R[R.screen == screen].pivot_table(index="subpair", columns="condition_id", values="yield", aggfunc="mean").reindex(columns=M.columns)
        Yr = Mr.values; kit = greedy(Y, 12); ret = bestin(Yr, kit[:6]).mean() / bestin(Yr, list(range(n))).mean()
        good = abs(ret - RESERVE[screen]) < 0.006; ok += good; bad += not good
        print(f"{'PASS' if good else 'FAIL'} reserve k=6 {screen:22s} {ret:.3f} (paper {RESERVE[screen]:.2f})")
checks = [("median retention k=1", np.median(r1), PAPER["median_k1"], 0.006), ("median retention k=3", np.median(r3), PAPER["median_k3"], 0.006),
          ("median retention k=6", np.median(r6), PAPER["median_k6"], 0.006), ("median random k=6", np.median(rr6), PAPER["median_rand6"], 0.006),
          ("median best-on-average k=6", np.median(my6), PAPER["median_my6"], 0.006),
          ("pairs >=20% with kit of 6", reach6, PAPER["reach_k6"], 0.5), ("pairs >=20% with kit of 3", reach3, PAPER["reach_k3"], 0.5), ("pairs >=20% any condition", reachf, PAPER["reach_full"], 0.5),
          ("hits with kit of 6 (>=20%)", hits, POSTHIT["hits_k6"], 0.5), ("hits >20 points below library best, k=6", sev6, POSTHIT["severe_k6"], 0.5),
          ("hits below 80% with kit of 6 (the only hits that can fall >20 points short)", hits80, POSTHIT["hits_below80_k6"], 0.5), ("hits' shortfall recovered 6->12", gain12 / gap6, POSTHIT["recovery_6_to_12"], 0.0006)]
for desc, val, paper, tol in checks:
    good = abs(val - paper) < tol; ok += good; bad += not good; print(f"{'PASS' if good else 'FAIL'} {desc}: {val:.3f} (paper {paper})")
# repeatability: Hiyama C43 and C46 are identical in composition; C46 is in no array matrix
if len(Q):
    H = A[(A.screen == "Hiyama|1") & (A.condition_id == "C43")].set_index("subpair")["yield"]
    P = Q.set_index("subpair")["yield"].reindex(H.index); m = H.notna() & P.notna()
    rv = float(np.corrcoef(H[m], P[m])[0, 1]); mad = float(100 * (H[m] - P[m]).abs().mean())
    for desc, val, paper, tol in (("replicate r", rv, 0.95, 0.006), ("replicate mean absolute difference, points", mad, 0.37, 0.006)):
        good = abs(val - paper) < tol; ok += good; bad += not good
        print(("PASS" if good else "FAIL") + " %s: %.3f (paper %s)" % (desc, val, paper))
print(f"\n{ok} checks pass, {bad} fail"); sys.exit(1 if bad else 0)
