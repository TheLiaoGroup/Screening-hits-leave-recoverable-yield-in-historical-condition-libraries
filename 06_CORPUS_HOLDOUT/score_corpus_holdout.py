# Public subset: input paths and source gates use the extracted-subset manifest.
# -*- coding: utf-8 -*-
"""Score the frozen corpus hold-out. Endpoints, panel selection and cross-validation are those
registered in FREEZE_corpus_holdout.json and are identical to the manuscript's kits.py.

Gate: every membership sha256 in the freeze is recomputed, the freeze file's own sha256 is
recomputed against the value recorded when it was frozen, and each block must yield exactly
pairs x conditions wells. Any mismatch aborts before a yield is read.

Writes holdout_scores.csv and holdout_pair_scores.csv and prints the unregistered controls.
Reads the freeze and the block membership tables; never writes them. The written record of this
freeze is kept by hand and gated by check_holdout_results.py.

Run: PYTHONIOENCODING=utf-8 py -3 score_corpus_holdout.py
"""
import hashlib, itertools, json, sys
from pathlib import Path
from subset_inputs import input_path, verify_inputs
import numpy as np, pandas as pd
from scipy.stats import beta as _beta

HERE = Path(__file__).resolve().parent
NS = HERE.parent
sys.path.insert(0, str(NS))
import lib

PRIVATE = input_path("wells.csv")
RAWCOND = input_path("conditions.csv")
FREEZE = HERE / "FREEZE_corpus_holdout.json"
# Content hash of the deposited FREEZE_corpus_holdout.json; not proof of chronology.
# A file cannot carry its own hash, so it is pinned here; edit the freeze and this gate fails.
FREEZE_SHA = "31f3d1318c8b1b729e32707029af9c55fd68d8f30103f01a6afae72a4d3409d4"

KS = list(range(1, 13))
NFOLD, SEED, N_RANDOM, N_BOOT = 10, 3, 50, 5000
HIT, SEVERE = 0.20, 0.20


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def clopper_pearson(k, n, alpha=0.05):
    """Exact 95% interval for a binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    lo = 0.0 if k == 0 else _beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else _beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lo), float(hi))


def boot_ratio_ci(num, den, n_boot=N_BOOT, seed=11):
    """Percentile interval for sum(num)/sum(den), resampling pairs with replacement."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    rng = np.random.RandomState(seed)
    n = len(num)
    draws = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        d = den[idx].sum()
        if d > 0:
            draws.append(num[idx].sum() / d)
    if len(draws) < n_boot * 0.9:
        raise SystemExit("bootstrap: too many resamples with a zero denominator")
    return tuple(float(v) for v in np.percentile(draws, [2.5, 97.5]))


def verify_freeze():
    """Hard gate. Returns the freeze document; exits non-zero on any mismatch."""
    bad = []
    got = sha(FREEZE)
    if got != FREEZE_SHA:
        bad.append("FREEZE_corpus_holdout.json sha256 %s != pinned %s" % (got, FREEZE_SHA))
    F = json.loads(FREEZE.read_text(encoding="utf-8"))
    if not F["blocks"]:
        bad.append("freeze lists no blocks")
    for b in F["blocks"]:
        f = HERE / "blocks" / b["membership_file"]
        if not f.exists():
            bad.append("%s: membership file missing" % b["block_id"]); continue
        g = sha(f)
        if g != b["membership_sha256"]:
            bad.append("%s: membership sha256 %s != frozen %s" % (b["block_id"], g, b["membership_sha256"]))
        mem = pd.read_csv(f)
        if list(mem.columns) != ["block", "pair", "condition_id"]:
            bad.append("%s: unexpected membership columns %s" % (b["block_id"], list(mem.columns)))
            continue
        if mem.pair.nunique() != b["pairs"] or mem.condition_id.nunique() != b["conditions"]:
            bad.append("%s: membership holds %d pairs x %d conditions, freeze says %d x %d"
                       % (b["block_id"], mem.pair.nunique(), mem.condition_id.nunique(), b["pairs"], b["conditions"]))
        if len(mem) != b["wells"]:
            bad.append("%s: membership holds %d rows, freeze says %d wells" % (b["block_id"], len(mem), b["wells"]))
    verify_inputs(FREEZE)
    if bad:
        print("FREEZE VERIFICATION FAILED - no yield was read:")
        for m in bad:
            print("  " + m)
        raise SystemExit(2)
    print("freeze verified: %s, %d blocks, %d membership hashes, public-subset manifest"
          % (FREEZE_SHA[:12], len(F["blocks"]), len(F["blocks"])))
    return F


def load_block_matrices(F):
    """pair x condition yield matrix per block, taken only from the frozen membership."""
    cols = ["rxn_class", "condition_id", "sub_1_smiles", "sub_2_smiles", "temp_id", "time_id",
            "yield", "check_passed", "allocation"]
    d = pd.read_csv(PRIVATE, low_memory=False, usecols=cols)
    d = d[d.allocation.str.contains("outside any qualifying", na=False)]
    d = d[d.check_passed == True]
    d = d[d["yield"].notna()]
    d["pair"] = [" >> ".join(sorted((str(a), str(b)))) for a, b in zip(d.sub_1_smiles, d.sub_2_smiles)]
    mats = {}
    for b in F["blocks"]:
        mem = pd.read_csv(HERE / "blocks" / b["membership_file"])
        pairs, conds = sorted(mem.pair.unique()), sorted(mem.condition_id.unique())
        want = set(zip(mem.pair, mem.condition_id))
        s = d[(d.rxn_class == b["rxn_class"]) & (d.temp_id == b["temperature_C"]) & (d.time_id == b["time_h"])]
        s = s[[(p, c) in want for p, c in zip(s.pair, s.condition_id)]]
        if len(s) == 0:
            raise SystemExit("%s: matched zero wells" % b["block_id"])
        dup = int(s.duplicated(subset=["pair", "condition_id"]).sum())
        if dup:
            raise SystemExit("%s: %d duplicated (pair, condition) rows" % (b["block_id"], dup))
        if len(s) != b["wells"]:
            raise SystemExit("%s: matched %d wells, freeze says %d" % (b["block_id"], len(s), b["wells"]))
        M = s.pivot_table(index="pair", columns="condition_id", values="yield", aggfunc="mean")
        M = M.reindex(index=pairs, columns=conds)
        if M.isna().any().any():
            raise SystemExit("%s: %d empty cells after pivot" % (b["block_id"], int(M.isna().sum().sum())))
        if M.shape != (b["pairs"], b["conditions"]):
            raise SystemExit("%s: matrix %s, freeze says %s" % (b["block_id"], M.shape, (b["pairs"], b["conditions"])))
        lo, hi = float(M.values.min()), float(M.values.max())
        if lo < 0 or hi > 1.05:
            raise SystemExit("%s: yields outside 0-1.05 (min %.3f max %.3f)" % (b["block_id"], lo, hi))
        mats[b["block_id"]] = M
        print("  %-26s %4d pairs x %3d conditions = %5d wells, yield %.3f-%.3f"
              % (b["block_id"], M.shape[0], M.shape[1], M.size, lo, hi))
    return mats


def cross_validate(M):
    """Manuscript-identical ten-fold CV over pairs. Returns per-pair panel bests and the curve."""
    Y = M.values
    S, ncond = Y.shape
    rng = np.random.RandomState(SEED)
    folds = np.array_split(rng.permutation(S), NFOLD)
    full = np.full(S, np.nan)
    fold_of = np.full(S, -1)
    greedy = {k: np.full(S, np.nan) for k in KS}
    topmean = {k: np.full(S, np.nan) for k in KS}
    random = {k: np.full(S, np.nan) for k in KS}
    panels = {}
    for fi, f in enumerate(folds):
        tr = np.setdiff1d(np.arange(S), f)
        Ytr, Yte = Y[tr], Y[f]
        _, fb = lib.retention(Yte, list(range(ncond)))
        full[f] = fb
        fold_of[f] = fi
        kit12 = lib.greedy_kit(Ytr, max(KS))
        rank = np.argsort(-np.nanmean(Ytr, 0))
        panels[fi] = {"greedy12": [M.columns[c] for c in kit12],
                      "topmean12": [M.columns[c] for c in rank[:max(KS)]]}
        for k in KS:
            greedy[k][f] = lib.retention(Yte, kit12[:k])[0]
            topmean[k][f] = lib.retention(Yte, list(rank[:k]))[0]
            random[k][f] = np.mean([lib.retention(Yte, list(rng.choice(ncond, k, replace=False)))[0]
                                    for _ in range(N_RANDOM)], 0)
    for name, arr in (("full", full), ("greedy6", greedy[6])):
        if np.isnan(arr).any():
            raise SystemExit("cross-validation left %d pairs unscored in %s" % (int(np.isnan(arr).sum()), name))
    return {"full": full, "fold": fold_of, "greedy": greedy, "topmean": topmean,
            "random": random, "panels": panels, "pairs": list(M.index)}


def endpoints(full, best, k_label):
    """E2 and E3 with exact intervals, plus the counts the freeze requires to be reported."""
    zero = int((full <= 0).sum())
    reach = full >= HIT
    prod = full > 0
    n2, h2 = int(reach.sum()), int(((best >= HIT) & reach).sum())
    n3, h3 = int(prod.sum()), int((((full - best) > SEVERE) & prod).sum())
    if n2 == 0 or n3 == 0:
        raise SystemExit("endpoint %s matched zero pairs (reachable %d, productive %d)" % (k_label, n2, n3))
    e2 = h2 / n2
    e3 = h3 / n3
    return {"n_zero_best": zero, "n_reachable": n2, "n_hit_kept": h2, "E2": e2,
            "E2_lo": clopper_pearson(h2, n2)[0], "E2_hi": clopper_pearson(h2, n2)[1],
            "n_productive": n3, "n_severe": h3, "E3": e3,
            "E3_lo": clopper_pearson(h3, n3)[0], "E3_hi": clopper_pearson(h3, n3)[1]}


def recipes():
    """condition_id -> reagent/solvent string, per reaction_type, from the frozen raw table."""
    x = pd.read_csv(RAWCOND, low_memory=False)
    x["reaction_type"] = x.reaction_type.astype(str).str.lower()
    out = {}
    for rt, s in x.groupby("reaction_type"):
        d = {}
        for _, r in s.iterrows():
            parts = [str(r["Reagent%d_Standardized Name" % i]) for i in range(1, 7)
                     if pd.notna(r["Reagent%d_Standardized Name" % i])]
            if pd.notna(r["Standardized Name"]):
                parts.append(str(r["Standardized Name"]))
            d[r["condition_id"]] = "; ".join(parts)
        out[rt] = d
    return out


def transfer_control(M):
    """Chan-Lam|25C only: the panel greedy picks on the ANALYSED Chan-Lam array, applied unchanged
    to these new pairs. Closest reading of taking an existing kit to new substrates."""
    A, _ = lib.load("analysis")
    Ma = lib.screen_matrix(A, "Chan-Lam|1")
    if set(Ma.columns) != set(M.columns):
        raise SystemExit("transfer control: analysed array and block do not share the same 32 conditions")
    kit12 = lib.greedy_kit(Ma.values, 12)
    ids = [Ma.columns[c] for c in kit12]
    col = {c: i for i, c in enumerate(M.columns)}
    idx = [col[c] for c in ids]
    Y = M.values
    _, full = lib.retention(Y, list(range(Y.shape[1])))
    res = {"analysed_pairs": int(Ma.shape[0]), "panel12": ids, "full": full}
    for k in (6, 12):
        best, _ = lib.retention(Y, idx[:k])
        res[k] = {"best": best, "R": float(best.sum() / full.sum()),
                  "ci": boot_ratio_ci(best, full, seed=21 + k)}
        res[k].update(endpoints(full, best, "transfer k=%d" % k))
    own = [M.columns[c] for c in lib.greedy_kit(Y, 6)]
    res["own6"] = own
    res["overlap6"] = len(set(ids[:6]) & set(own))
    return res


def thinning_diagnostic(F, summary, n_draw=200, seed=77):
    """Outside the freeze, and named as such. Thin each ANALYSED array to the number of productive
    pairs a block has, run the identical cross-validation, and read off where the block's R6 sits.
    Answers whether three or four productive pairs can give R6 near 0.5 on their own."""
    A, _ = lib.load("analysis")
    mats = {s: lib.screen_matrix(A, s) for s in sorted(A.screen.unique())}
    rng = np.random.RandomState(seed)

    def cv_r6(Y):
        n = Y.shape[0]
        r = np.random.RandomState(SEED)
        num = den = 0.0
        for f in np.array_split(r.permutation(n), NFOLD):
            tr = np.setdiff1d(np.arange(n), f)
            b, fu = lib.retention(Y[f], lib.greedy_kit(Y[tr], max(KS))[:6])
            num += b.sum(); den += fu.sum()
        return num / den if den > 0 else np.nan

    out = {}
    for bid, s in summary.items():
        npairs = s["block"]["pairs"]
        nlive = s["ep6"]["n_productive"]
        vals = []
        per = max(1, n_draw // len(mats))
        for M in mats.values():
            Yf = np.where(np.isnan(M.values), 0.0, M.values)
            full = Yf.max(1)
            live, dead = np.where(full > 0)[0], np.where(full <= 0)[0]
            if len(live) < nlive:
                continue
            nz = npairs - nlive
            for _ in range(per):
                idx = rng.choice(live, nlive, replace=False)
                if nz > 0 and len(dead):
                    idx = np.concatenate([idx, rng.choice(dead, nz, replace=len(dead) < nz)])
                v = cv_r6(Yf[idx.astype(int)])
                if np.isfinite(v):
                    vals.append(v)
        if not vals:
            raise SystemExit("thinning diagnostic matched zero draws for %s" % bid)
        v = np.array(vals)
        out[bid] = {"n": len(v), "median": float(np.median(v)),
                    "p5": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95)),
                    "pct_of_observed": float((v <= s["R6"]).mean()), "n_live": nlive}
    return out


def main():
    F = verify_freeze()
    print("loading yields")
    mats = load_block_matrices(F)
    rows, pair_rows, summary = [], [], {}
    for b in F["blocks"]:
        bid = b["block_id"]
        M = mats[bid]
        cv = cross_validate(M)
        full = cv["full"]
        for k in KS:
            g, t, r = cv["greedy"][k], cv["topmean"][k], cv["random"][k]
            ep = endpoints(full, g, "%s k=%d" % (bid, k))
            lo, hi = boot_ratio_ci(g, full, seed=100 + k)
            rows.append({"block": bid, "rxn_class": b["rxn_class"], "pairs": b["pairs"],
                         "conditions": b["conditions"], "k": k,
                         "R_greedy": g.sum() / full.sum(), "R_greedy_lo": lo, "R_greedy_hi": hi,
                         "R_topmean": t.sum() / full.sum(), "R_random": r.sum() / full.sum(),
                         "mean_full_best": full.mean(), "mean_greedy_best": g.mean(), **ep})
        for i, p in enumerate(cv["pairs"]):
            pair_rows.append({"block": bid, "pair": p, "fold": int(cv["fold"][i]),
                              "full_best": full[i], "best_k6": cv["greedy"][6][i],
                              "best_k12": cv["greedy"][12][i],
                              "loss_pp": (full[i] - cv["greedy"][6][i]) * 100,
                              "loss_pp_k12": (full[i] - cv["greedy"][12][i]) * 100,
                              "reaches20": bool(full[i] >= HIT),
                              "panel_reaches20": bool(cv["greedy"][6][i] >= HIT)})
        ep6 = endpoints(full, cv["greedy"][6], bid)
        mols, live_mols = set(), set()
        for i, q in enumerate(cv["pairs"]):
            mols |= set(q.split(" >> "))
            if full[i] > 0:
                live_mols |= set(q.split(" >> "))
        if not mols:
            raise SystemExit("%s: no molecules parsed from the pair keys" % bid)
        summary[bid] = {"cv": cv, "full": full, "ep6": ep6,
                        "R6": cv["greedy"][6].sum() / full.sum(),
                        "R6_ci": boot_ratio_ci(cv["greedy"][6], full, seed=106),
                        "R12": cv["greedy"][12].sum() / full.sum(),
                        "ep12": endpoints(full, cv["greedy"][12], bid),
                        "mean_full": full.mean(), "block": b,
                       "n_mol": len(mols), "n_live_mol": len(live_mols)}
        print("  %-26s R6 %.3f  E2 %.3f  E3(k6) %.3f  E3(k12) %.3f  zero-best %d"
              % (bid, summary[bid]["R6"], ep6["E2"], ep6["E3"], summary[bid]["ep12"]["E3"], ep6["n_zero_best"]))
    S = pd.DataFrame(rows)
    S.to_csv(HERE / "holdout_scores.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(HERE / "holdout_pair_scores.csv", index=False)
    rec = recipes()
    for b in F["blocks"]:
        if b["rxn_class"].lower() not in rec:
            raise SystemExit("no recipe table for reaction type %r" % b["rxn_class"])
    for r in cross_block_overlap(summary):
        print("  block overlap %s / %s: %d shared pairs, %d and %d productive in them, %d in both"
              % (r["a"], r["b"], r["shared"], r["live_a"], r["live_b"], r["live_both"]))
    tc = transfer_control(mats["Chan-Lam|25C"])
    print("  transfer control (unregistered): panel from %d analysed Chan-Lam pairs, R6 %.4f "
          "[%.4f, %.4f], R12 %.4f, E2 %d/%d, E3 %d/%d, %d of 6 shared with the block's own six"
          % (tc["analysed_pairs"], tc[6]["R"], tc[6]["ci"][0], tc[6]["ci"][1], tc[12]["R"],
             tc[6]["n_hit_kept"], tc[6]["n_reachable"], tc[6]["n_severe"], tc[6]["n_productive"],
             tc["overlap6"]))
    print("thinning diagnostic (outside the freeze)")
    for bid, t in thinning_diagnostic(F, summary).items():
        print("  %-26s %2d productive, thinned arrays give %.3f [%.3f, %.3f], this block at %.0f%%"
              % (bid, t["n_live"], t["median"], t["p5"], t["p95"], 100 * t["pct_of_observed"]))
    print("wrote holdout_scores.csv and holdout_pair_scores.csv. The written record of this freeze "
          "is kept by hand and gated by check_holdout_results.py")
    return summary


def cross_block_overlap(summary):
    """The freeze guarantees no overlap with released data; it never claimed the blocks are
    disjoint from each other. Report any substrate pair that two blocks share."""
    rows = []
    for a, b in itertools.combinations(sorted(summary), 2):
        sa = {q: summary[a]["full"][i] for i, q in enumerate(summary[a]["cv"]["pairs"])}
        sb = {q: summary[b]["full"][i] for i, q in enumerate(summary[b]["cv"]["pairs"])}
        sh = sorted(set(sa) & set(sb))
        if sh:
            rows.append({"a": a, "b": b, "shared": len(sh),
                         "live_a": sum(sa[q] > 0 for q in sh), "live_b": sum(sb[q] > 0 for q in sh),
                         "live_both": sum(sa[q] > 0 and sb[q] > 0 for q in sh),
                         "n_live_a": summary[a]["ep6"]["n_productive"],
                         "n_live_b": summary[b]["ep6"]["n_productive"]})
    return rows


if __name__ == "__main__":
    main()
