# Public subset: input paths and source gates use the extracted-subset manifest.
# -*- coding: utf-8 -*-
"""Score freeze 4. Reads FREEZE_transported_fullish.json and touches only the cells it names.

The registered criteria are applied as written: a block supports only if retention is at least 0.60
AND the transported panel sits at or above the 90th percentile of 20,000 random six-condition subsets
of that block's own measured conditions. Both gates, not either.
"""
import hashlib, json, sys
from pathlib import Path
from subset_inputs import input_path, verify_inputs
import numpy as np, pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
PRIVATE = input_path("wells.csv")
FREEZE = HERE / "FREEZE_transported_fullish.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cp(k, n):
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else float(stats.beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(0.975, k + 1, n - k))
    return (lo, hi)


def ratio(M, j):
    Yf = np.where(np.isnan(M), -np.inf, M)
    full = np.where(np.isinf(np.max(Yf, 1)), 0.0, np.max(Yf, 1))
    b = np.where(np.isinf(np.max(Yf[:, j], 1)), 0.0, np.max(Yf[:, j], 1))
    return full, b


def main():
    doc = json.loads(FREEZE.read_text(encoding="utf-8"))
    verify_inputs(FREEZE)
    T = doc["thresholds"]
    MIN_REACH, GATE, NDRAW = T["MIN_REACH"], T["PCT_GATE"], T["RANDOM_DRAWS"]

    cols = ["rxn_class", "condition_id", "sub_1_smiles", "sub_2_smiles", "yield", "check_passed"]
    D = pd.read_csv(PRIVATE, low_memory=False, usecols=cols)
    D = D[D.check_passed == True]
    D["pair"] = [" >> ".join(sorted((str(a), str(b)))) for a, b in zip(D.sub_1_smiles, D.sub_2_smiles)]

    res, pair_rows, seen = [], [], set()
    for k, b in enumerate(doc["blocks"]):
        fn = HERE / "blocks" / b["membership_file"]
        assert sha(fn) == b["membership_sha256"], "%s changed since the freeze" % fn.name
        mem = pd.read_csv(fn)
        pairs = sorted(set(mem.pair))
        assert len(pairs) == b["never_read_pairs"], "%s: pair count moved" % b["screen"]
        assert not (set(pairs) & seen), "%s: a pair is shared with an earlier block" % b["screen"]
        seen |= set(pairs)
        panel = b["panel6"]
        allc = sorted(set(mem.condition_id))
        keep = mem.groupby("pair").condition_id.apply(set).to_dict()

        s = D[D.pair.isin(set(pairs)) & D.condition_id.isin(set(allc))]
        Y = s.pivot_table(index="pair", columns="condition_id", values="yield", aggfunc="max")
        Y = Y.reindex(index=pairs, columns=allc)
        # blank every cell the freeze did not list, so nothing outside the frozen membership is read
        Msk = np.zeros(Y.shape, dtype=bool)
        for i, p in enumerate(pairs):
            Msk[i] = [c in keep[p] for c in allc]
        M = np.where(Msk, Y.values, np.nan)
        assert not np.isnan(M[:, [allc.index(c) for c in panel]]).any(), "%s: a panel cell is missing" % b["screen"]

        j = [allc.index(c) for c in panel]
        full, b6 = ratio(M, j)
        live, reach = full > 0, full >= 0.20
        n, nlive, nreach = len(full), int(live.sum()), int(reach.sum())
        R6 = float(b6.sum() / full.sum()) if full.sum() else float("nan")

        rs = np.random.RandomState(500 + k)
        draws = np.empty(NDRAW)
        for i in range(NDRAW):
            _, bb = ratio(M, list(rs.choice(len(allc), 6, replace=False)))
            draws[i] = bb.sum() / full.sum()
        pct = float((draws < R6).mean())

        bs = np.random.RandomState(600 + k)
        idx = bs.randint(0, n, size=(10000, n))
        bt = b6[idx].sum(1) / full[idx].sum(1)
        ci = (float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5)))

        hit_k = int((b6[reach] >= 0.20).sum())
        sev_k = int(((full - b6)[live] > 0.20).sum())
        singles = np.array([ratio(M, [q])[1].sum() / full.sum() for q in range(len(allc))])
        resolvable = nreach >= MIN_REACH
        gate_r, gate_p = R6 >= 0.60, pct >= GATE
        bias = b["coverage_bias_control"]

        res.append({
            "screen": b["screen"], "rxn_class": b["rxn_class"],
            "never_read_pairs": n, "library_conditions": b["library_conditions"],
            "coverage_median": b["coverage_median"],
            "pairs_with_any_product": nlive, "pairs_reaching_0.20": nreach,
            "resolvable": bool(resolvable),
            "F1_R6": R6, "F1_R6_ci": list(ci),
            "F2_random_six_percentile": pct,
            "F2_random_six_median": float(np.median(draws)),
            "F2_random_six_p95": float(np.percentile(draws, 95)),
            "F2_margin_over_random_median": float(R6 - np.median(draws)),
            "best_single_condition_R6": float(np.nanmax(singles)),
            "F3_hit_recovery": (hit_k / nreach) if nreach else float("nan"),
            "F3_counts": [hit_k, nreach], "F3_ci": list(cp(hit_k, nreach)),
            "F4_severe_loss": (sev_k / nlive) if nlive else float("nan"),
            "F4_counts": [sev_k, nlive],
            "inflation_from_partial_coverage": bias["inflation_from_partial_coverage"],
            "analysed_pairs_R6_full_library": bias["R6_full_library"],
            "gate_retention_0.60": bool(gate_r),
            "gate_random_six_percentile_%.2f" % GATE: bool(gate_p),
            "block_verdict": ("supports" if (gate_r and gate_p) else "does not support") if resolvable else "UNRESOLVED",
            "mean_per_pair_reference_best": float(full.mean()),
            "mean_yield_per_well": float(np.nanmean(M)),
        })
        for i, p in enumerate(pairs):
            pair_rows.append({"screen": b["screen"], "pair": p, "reference_best": full[i],
                              "best_of_6": b6[i], "loss_6": full[i] - b6[i],
                              "conditions_measured": int(Msk[i].sum())})

    R = pd.DataFrame(res)
    ok = R[R.resolvable]
    if len(ok) < 2:
        verdict, why = "UNRESOLVED", "fewer than two blocks are resolvable, so the registered floor applies"
    else:
        allsup = bool((ok.block_verdict == "supports").all())
        verdict = "supports" if allsup else "does not support"
        why = ("%d of %d resolvable blocks support under both gates; %s"
               % (int((ok.block_verdict == "supports").sum()), len(ok),
                  "all of them" if allsup else "failing: " + ", ".join(
                      ok[ok.block_verdict != "supports"].rxn_class)))

    pd.DataFrame(pair_rows).to_csv(HERE / "fullish_pair_scores.csv", index=False, lineterminator="\n")
    R.drop(columns=[c for c in R.columns if R[c].map(lambda v: isinstance(v, list)).any()]).to_csv(
        HERE / "fullish_scores.csv", index=False, lineterminator="\n")
    (HERE / "FULLISH_RESULT.json").write_text(json.dumps(
        {"scored_from": FREEZE.name, "freeze_sha256": sha(FREEZE),
         "registered_criteria": doc["criteria"]["outcomes_fixed_in_advance"],
         "blocks": res, "verdict": verdict, "verdict_basis": why}, indent=1,
        ensure_ascii=False, default=float) + "\n", encoding="utf-8")

    for r in res:
        print("%-12s %3d pairs (%d with product, %d reaching 0.20), library %d, coverage median %.2f"
              % (r["rxn_class"], r["never_read_pairs"], r["pairs_with_any_product"],
                 r["pairs_reaching_0.20"], r["library_conditions"], r["coverage_median"]))
        print("   F1 R6 = %.3f [%.3f, %.3f]   (partial-coverage inflation measured at %+.3f)"
              % (r["F1_R6"], *r["F1_R6_ci"], r["inflation_from_partial_coverage"]))
        print("   F2 random six: median %.3f, p95 %.3f -> panel percentile %.4f, margin %+.3f "
              "| best single condition %.3f"
              % (r["F2_random_six_median"], r["F2_random_six_p95"], r["F2_random_six_percentile"],
                 r["F2_margin_over_random_median"], r["best_single_condition_R6"]))
        print("   F3 hit recovery %d/%d = %.3f [%.3f, %.3f] | F4 severe loss %d/%d = %.3f"
              % (*r["F3_counts"], r["F3_hit_recovery"], *r["F3_ci"], *r["F4_counts"], r["F4_severe_loss"]))
        print("   -> %s\n" % r["block_verdict"])
    print("VERDICT: %s\n  %s" % (verdict, why))


if __name__ == "__main__":
    main()
