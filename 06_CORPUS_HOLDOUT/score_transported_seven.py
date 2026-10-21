# Public subset: input paths and source gates use the extracted-subset manifest.
# -*- coding: utf-8 -*-
"""Score freeze 3. Reads FREEZE_transported_seven.json and touches only the cells it names.

The registered rules are applied as written, including the resolvability gate, and the verdict is
whatever they produce. Nothing here may add, drop or redefine a block, pair, condition or endpoint.
"""
import hashlib, json, sys
from pathlib import Path
from subset_inputs import input_path, verify_inputs
import numpy as np, pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
PRIVATE = input_path("wells.csv")
FREEZE = HERE / "FREEZE_transported_seven.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cp(k, n):
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else float(stats.beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(0.975, k + 1, n - k))
    return (lo, hi)


def boot_ratio(num, den, seed, B=10000):
    rs = np.random.RandomState(seed)
    n = len(num)
    i = rs.randint(0, n, size=(B, n))
    r = num[i].sum(1) / den[i].sum(1)
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def main():
    doc = json.loads(FREEZE.read_text(encoding="utf-8"))
    verify_inputs(FREEZE)
    MIN_REACH = doc["thresholds"]["MIN_REACH"]

    cols = ["rxn_class", "condition_id", "sub_1_smiles", "sub_2_smiles", "yield", "check_passed"]
    D = pd.read_csv(PRIVATE, low_memory=False, usecols=cols)
    D = D[D.check_passed == True]
    D["pair"] = [" >> ".join(sorted((str(a), str(b)))) for a, b in zip(D.sub_1_smiles, D.sub_2_smiles)]

    res, pair_rows, seen = [], [], set()
    for k, b in enumerate(doc["blocks"]):
        fn = HERE / "blocks" / b["membership_file"]
        assert sha(fn) == b["membership_sha256"], "%s changed since the freeze" % fn.name
        mem = pd.read_csv(fn)
        pairs, refcols, panel = sorted(set(mem.pair)), b["reference_condition_ids"], b["panel6"]
        assert len(pairs) == b["never_read_pairs"], "%s: pair count moved" % b["screen"]
        assert not (set(pairs) & seen), "%s: a pair is shared with an earlier block" % b["screen"]
        seen |= set(pairs)

        s = D[D.pair.isin(set(pairs)) & D.condition_id.isin(set(refcols))]
        Y = s.pivot_table(index="pair", columns="condition_id", values="yield", aggfunc="max")
        Y = Y.reindex(index=pairs, columns=refcols)
        assert not Y.isna().values.any(), "%s: the frozen submatrix is not complete in the corpus" % b["screen"]

        M = Y.values
        full = M.max(1)
        j = [refcols.index(c) for c in panel]
        b6 = M[:, j].max(1)
        live, reach = full > 0, full >= 0.20
        n, nlive, nreach = len(full), int(live.sum()), int(reach.sum())
        R6 = float(b6.sum() / full.sum()) if full.sum() else float("nan")
        ci = boot_ratio(b6, full, seed=100 + k)
        hit_k = int((b6[reach] >= 0.20).sum())
        sev_k = int(((full - b6)[live] > 0.20).sum())
        ctl = b["control_on_analysed_pairs"]
        resolvable = nreach >= MIN_REACH

        res.append({
            "screen": b["screen"], "rxn_class": b["rxn_class"],
            "never_read_pairs": n, "reference_conditions": len(refcols),
            "array_library_conditions": b["array_library_conditions"],
            "pairs_with_any_product": nlive, "pairs_reaching_0.20": nreach,
            "resolvable": bool(resolvable),
            "resolvable_rule": "at least %d never-read pairs reaching 0.20" % MIN_REACH,
            "S1_R6": R6, "S1_R6_ci": list(ci),
            "control_R6_same_columns": ctl["R6_on_the_same_columns"],
            "control_R6_full_library": ctl["R6_on_the_full_library"],
            "S1_minus_control": (R6 - ctl["R6_on_the_same_columns"]) if ctl["R6_on_the_same_columns"] else None,
            "control_inside_S1_ci": bool(ctl["R6_on_the_same_columns"] is not None
                                         and ci[0] <= ctl["R6_on_the_same_columns"] <= ci[1]),
            "S2_hit_recovery": (hit_k / nreach) if nreach else float("nan"),
            "S2_counts": [hit_k, nreach], "S2_ci": list(cp(hit_k, nreach)),
            "control_hit_recovery": ctl["hit_recovery_on_the_same_columns"],
            "S3_severe_loss": (sev_k / nlive) if nlive else float("nan"),
            "S3_counts": [sev_k, nlive], "S3_ci": list(cp(sev_k, nlive)),
            "control_severe_loss": ctl["severe_loss_on_the_same_columns"],
            "mean_per_pair_reference_best": float(full.mean()),
            "mean_yield_per_well": float(np.nanmean(M)),
        })
        for i, p in enumerate(pairs):
            pair_rows.append({"screen": b["screen"], "pair": p, "reference_best": full[i],
                              "best_of_6": b6[i], "loss_6": full[i] - b6[i]})

    R = pd.DataFrame(res)
    ok = R[R.resolvable]
    floor = len(ok) < 4
    if floor:
        verdict, why = "UNRESOLVED", ("fewer than four blocks are resolvable, so the registered floor applies "
                                      "and no verdict is drawn")
    else:
        med, medc = float(ok.S1_R6.median()), float(ok.control_R6_same_columns.median())
        frac = float((ok.S1_R6 >= 0.60).mean())
        a, bb = (med >= medc - 0.10), (frac >= 2 / 3)
        verdict = "supports" if (a and bb) else "does not support"
        why = ("median R6 %.3f against median control %.3f (within 0.10: %s); %d of %d resolvable blocks "
               "at or above 0.60 (%.0f%%, needs two thirds: %s)"
               % (med, medc, a, int((ok.S1_R6 >= 0.60).sum()), len(ok), 100 * frac, bb))

    pd.DataFrame(pair_rows).to_csv(HERE / "seven_pair_scores.csv", index=False, lineterminator="\n")
    R.drop(columns=[c for c in R.columns if R[c].map(lambda v: isinstance(v, list)).any()]).to_csv(
        HERE / "seven_scores.csv", index=False, lineterminator="\n")
    (HERE / "SEVEN_RESULT.json").write_text(json.dumps(
        {"scored_from": FREEZE.name, "freeze_sha256": sha(FREEZE),
         "registered_criteria": doc["criteria"]["outcomes_fixed_in_advance"],
         "resolvability_rule": doc["criteria"]["resolvability_rule_fixed_before_scoring"],
         "blocks": res, "verdict": verdict, "verdict_basis": why,
         "unresolved_blocks": R[~R.resolvable].rxn_class.tolist()},
        indent=1, ensure_ascii=False, default=float) + "\n", encoding="utf-8")

    print("%-20s %5s %5s %6s %6s   %-16s %-8s %-8s %s"
          % ("transformation", "pairs", "cond", "prod", ">=.20", "R6 [95% CI]", "control", "delta", "resolvable"))
    for r in res:
        print("%-20s %5d %5d %6d %6d   %.3f [%.3f,%.3f] %8.3f %8.3f   %s"
              % (r["rxn_class"], r["never_read_pairs"], r["reference_conditions"],
                 r["pairs_with_any_product"], r["pairs_reaching_0.20"], r["S1_R6"], *r["S1_R6_ci"],
                 r["control_R6_same_columns"], r["S1_minus_control"],
                 "yes" if r["resolvable"] else "NO (< %d reaching 0.20)" % MIN_REACH))
    print()
    for r in res:
        print("   %-20s hit %d/%d = %.3f [%.3f,%.3f] (control %.3f) | severe %d/%d = %.3f (control %.3f)"
              % (r["rxn_class"], *r["S2_counts"], r["S2_hit_recovery"], *r["S2_ci"],
                 r["control_hit_recovery"], *r["S3_counts"], r["S3_severe_loss"], r["control_severe_loss"]))
    print("\nresolvable blocks: %d of %d" % (len(ok), len(R)))
    print("VERDICT: %s\n  %s" % (verdict, why))


if __name__ == "__main__":
    main()
