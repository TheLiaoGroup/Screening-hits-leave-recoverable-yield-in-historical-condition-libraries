# Public subset: input paths and source gates use the extracted-subset manifest.
# -*- coding: utf-8 -*-
"""Score freeze 2. Reads FREEZE_transported_panel.json and touches only the cells it names."""
import hashlib, json, sys
from pathlib import Path
from subset_inputs import input_path, verify_inputs
import numpy as np, pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
NS = HERE.parent
sys.path.insert(0, str(NS))
import lib

PRIVATE = input_path("wells.csv")
FREEZE = HERE / "FREEZE_transported_panel.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cp(k, n):
    """Clopper-Pearson 95% interval."""
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else stats.beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(0.975, k + 1, n - k)
    return (float(lo), float(hi))


def boot_ratio(num, den, B=10000, seed=11):
    """Paired bootstrap over pairs for a ratio of sums."""
    rs = np.random.RandomState(seed)
    n = len(num)
    idx = rs.randint(0, n, size=(B, n))
    r = num[idx].sum(1) / den[idx].sum(1)
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def main():
    doc = json.loads(FREEZE.read_text(encoding="utf-8"))
    verify_inputs(FREEZE)

    cols = ["rxn_class", "condition_id", "sub_1_smiles", "sub_2_smiles", "temp_id", "yield", "check_passed"]
    D = pd.read_csv(PRIVATE, low_memory=False, usecols=cols)
    D = D[D.check_passed == True]
    D["pair"] = [" >> ".join(sorted((str(a), str(b)))) for a, b in zip(D.sub_1_smiles, D.sub_2_smiles)]

    rows, out = [], []
    for b in doc["blocks"]:
        fn = HERE / "blocks" / b["membership_file"]
        assert sha(fn) == b["membership_sha256"], "%s changed since the freeze" % fn.name
        mem = pd.read_csv(fn)
        panel = b["panel12"]

        s = D[D.pair.isin(set(mem.pair)) & D.condition_id.isin(set(mem.condition_id))]
        Y = s.pivot_table(index="pair", columns="condition_id", values="yield", aggfunc="max")
        # keep exactly the frozen cells
        keep = mem.groupby("pair").condition_id.apply(set).to_dict()
        Y = Y.loc[sorted(keep)]
        for p in Y.index:
            Y.loc[p, [c for c in Y.columns if c not in keep[p]]] = np.nan
        assert Y.shape[0] == b["eligible_pairs"], "pair count moved: %d vs %d" % (Y.shape[0], b["eligible_pairs"])

        M = Y.values
        full = np.nanmax(np.where(np.isnan(M), -np.inf, M), axis=1)
        full = np.where(np.isinf(full), 0.0, full)
        i6 = [Y.columns.get_loc(c) for c in panel[:6]]
        b6 = np.nan_to_num(np.nanmax(M[:, i6], axis=1))
        assert not np.isnan(M[:, i6]).any(), "a panel cell is missing"
        # The freeze named out/kit_members.csv as the panel source and that file carries six
        # conditions, not twelve. The k = 12 endpoints are therefore undefined under this freeze and
        # are withdrawn rather than recomputed from another source after the k = 6 result was seen.
        has12 = len(panel) >= 12
        b12 = np.nan_to_num(np.nanmax(M[:, [Y.columns.get_loc(c) for c in panel[:12]]], axis=1)) if has12 else None

        live, reach = full > 0, full >= 0.20
        n, nlive, nreach = len(full), int(live.sum()), int(reach.sum())
        R6 = float(b6.sum() / full.sum())
        ci6 = boot_ratio(b6, full)
        R12 = float(b12.sum() / full.sum()) if has12 else None
        ci12 = boot_ratio(b12, full) if has12 else None
        hit_k = int((b6[reach] >= 0.20).sum())
        sev6_k = int(((full - b6)[live] > 0.20).sum())
        sev12_k = int(((full - b12)[live] > 0.20).sum()) if has12 else None

        r = {"screen": b["screen"], "label": b["label"], "pairs": n,
             "pairs_with_any_product": nlive, "pairs_reaching_0.20": nreach,
             "pairs_dead": n - nlive,
             "T1_R6": R6, "T1_R6_ci": ci6,
             "T1_R12": R12, "T1_R12_ci": ci12,
             "k12_status": ("scored" if has12 else
                            "WITHDRAWN: the frozen panel source out/kit_members.csv holds six "
                            "conditions, so no twelve-condition panel was frozen for this block"),
             "T2_hit_recovery": (hit_k / nreach if nreach else float("nan")),
             "T2_counts": [hit_k, nreach], "T2_ci": cp(hit_k, nreach),
             "T3_severe_k6": (sev6_k / nlive if nlive else float("nan")),
             "T3_severe_k6_counts": [sev6_k, nlive], "T3_severe_k6_ci": cp(sev6_k, nlive),
             "T3_severe_k12": ((sev12_k / nlive if nlive else float("nan")) if has12 else None),
             "T3_severe_k12_counts": ([sev12_k, nlive] if has12 else None),
             "mean_full_array_best": float(full.mean()),
             "mean_yield_all_measured_wells": float(np.nanmean(M)),
             "predictions_in_sample": b["predictions"]}
        out.append(r)
        for j, p in enumerate(Y.index):
            rows.append({"screen": b["screen"], "pair": p, "array_best": full[j],
                         "best_of_6": b6[j], "loss_6": full[j] - b6[j]})

    pd.DataFrame(rows).to_csv(HERE / "transported_pair_scores.csv", index=False, lineterminator="\n")
    pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in out]).to_csv(
        HERE / "transported_scores.csv", index=False, lineterminator="\n")

    crit = doc["criteria"]["outcomes_fixed_in_advance"]
    verdict = []
    for r in out:
        ok = (r["T1_R6"] >= 0.60) and (r["T2_hit_recovery"] >= 0.70)
        verdict.append((r["label"], "supports" if ok else "does not support"))
    (HERE / "TRANSPORTED_RESULT.json").write_text(
        json.dumps({"scored_from": FREEZE.name, "freeze_sha256": sha(FREEZE),
                    "thresholds": crit, "blocks": out, "verdict": verdict},
                   indent=1, ensure_ascii=False, default=float) + "\n", encoding="utf-8")

    for r in out:
        print("%s | %d never-read pairs, %d with product, %d reaching 0.20"
              % (r["label"], r["pairs"], r["pairs_with_any_product"], r["pairs_reaching_0.20"]))
        print("  T1 R6  = %.3f  [%.3f, %.3f]   (in-sample on the analysed pairs %.3f; manuscript CV 0.71)"
              % (r["T1_R6"], *r["T1_R6_ci"], r["predictions_in_sample"]["R6_in_sample"]))
        if r["T1_R12"] is None:
            print("  T1 R12 = %s" % r["k12_status"])
        else:
            print("  T1 R12 = %.3f  [%.3f, %.3f]" % (r["T1_R12"], *r["T1_R12_ci"]))
        print("  T2 hit recovery = %d/%d = %.3f  [%.3f, %.3f]   (in-sample %.3f)"
              % (*r["T2_counts"], r["T2_hit_recovery"], *r["T2_ci"],
                 r["predictions_in_sample"]["hit_recovery_in_sample"]))
        print("  T3 severe loss k6  = %d/%d = %.3f   (in-sample %.3f)"
              % (*r["T3_severe_k6_counts"], r["T3_severe_k6"],
                 r["predictions_in_sample"]["severe_loss_k6_in_sample"]))
        if r["T3_severe_k12"] is not None:
            print("  T3 severe loss k12 = %d/%d = %.3f" % (*r["T3_severe_k12_counts"], r["T3_severe_k12"]))
        print("  mean of the per-pair array best = %.3f ; mean yield over all measured wells = %.4f"
              % (r["mean_full_array_best"], r["mean_yield_all_measured_wells"]))
    print("VERDICT:", verdict)


if __name__ == "__main__":
    main()
