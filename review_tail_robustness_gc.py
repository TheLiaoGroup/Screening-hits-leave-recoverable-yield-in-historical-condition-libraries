"""R3 tail sensitivity: reuse deposited molecule splits; perturb frozen OOF panels.

Only writes out/review_tail_robustness_* and the R3 TAIL_ROBUSTNESS.md report.
The multiplicative-noise calculations are hypothetical sensitivity scenarios,
not measurement-error correction or an empirical upper bound on tail losses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

import lib

HERE = Path(__file__).resolve().parent


def _hist(*parts):
    """Locate an audit artefact at the tree root or, once archived, under 04_HISTORY."""
    here = HERE.joinpath(*parts)
    hist = HERE.joinpath("04_HISTORY", *parts)
    return hist if hist.exists() else here
OUT = HERE / "out"
PREFIX = "review_tail_robustness_"
SEED = 20260920
CVS = (0., .057, .06, .069, .10, .20)
DENOMINATORS = ("original_full_gt_0", "original_full_ge_0p2")
CONTROLS = ("random_pair_control", "same_test_pair_control")


def rng_for(*parts):
    # Exact stream construction used in review_generalization_gc.py. Names of
    # original controls are unchanged; new sensitivity streams use new names.
    digest = hashlib.sha256("|".join(map(str, (SEED,) + parts)).encode()).digest()
    return np.random.RandomState(int.from_bytes(digest[:4], "little"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(frame, suffix, compressed=False):
    path = OUT / (PREFIX + suffix + (".csv.gz" if compressed else ".csv"))
    frame.to_csv(path, index=False, compression="gzip" if compressed else None)
    return path


def mask_for(full, denominator):
    return full > 0 if denominator == "original_full_gt_0" else full >= .2


def tail(full, best, denominator):
    eligible = mask_for(full, denominator)
    loss = 100 * (full-best)
    count, n = int(((loss > 20) & eligible).sum()), int(eligible.sum())
    return count, n, count/n if n else np.nan


def perturb_and_score(y, panel, z, cvs):
    """One perturbed matrix supplies BOTH maxima; panel indexes are frozen."""
    assert z.shape == y.shape and panel.shape[0] == len(y)
    cvs = np.asarray(cvs, float)
    finite = np.isfinite(y)
    noisy = np.clip(y[None, :, :] * (1+cvs[:, None, None]*z[None, :, :]), 0, 1)
    assert np.array_equal(np.isnan(noisy), np.broadcast_to(~finite, noisy.shape))
    assert np.all(noisy[:, finite & (y == 0)] == 0)
    filled = np.where(np.isnan(noisy), 0, noisy)
    full = filled.max(axis=2)
    indexes = np.broadcast_to(panel, (len(cvs), len(y), panel.shape[1]))
    best = np.take_along_axis(filled, indexes, axis=2).max(axis=2)
    assert np.all(best <= full+1e-14)
    return full, best


def synthetic_checks():
    # A deliberately perturbed selected well becomes the full-array winner.
    # Perturbing only the full maximum or retaining its original winning column
    # would fail this check. A second row crosses the eligibility threshold,
    # demonstrating why membership must remain fixed to the ORIGINAL full max.
    y = np.array([[.5, .4, np.nan, 0.], [.2, .1, np.nan, 0.], [0., 0., np.nan, 0.]])
    panel = np.array([[1], [1], [1]])
    z = np.array([[-1., 1., 0., 0.], [-1., 0., 0., 0.], [100., 100., 0., 0.]])
    full, best = perturb_and_score(y, panel, z, [0., .5])
    np.testing.assert_allclose(full, [[.5, .2, 0.], [.6, .1, 0.]], atol=1e-14, rtol=0)
    np.testing.assert_allclose(best, [[.4, .1, 0.], [.6, .1, 0.]], atol=1e-14, rtol=0)
    assert (full[1]-best[1] == 0).all()
    assert mask_for(full[0], "original_full_ge_0p2").sum() == 2
    assert mask_for(full[1], "original_full_ge_0p2").sum() == 1
    return {"shared_noise_matrix_winner_switch": True, "frozen_denominator_threshold_crossing": True,
            "zero_and_missing_structure": True}


def array_bootstrap(values, key, draws=10000, statistic="mean"):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values): return np.nan, np.nan
    indices = rng_for("tail_array_bootstrap", key).randint(0, len(values), (draws, len(values)))
    reducer = np.mean if statistic == "mean" else np.median
    q = np.quantile(reducer(values[indices], axis=1), [.025, .975])
    return float(q[0]), float(q[1])


def load_inputs():
    data_dir = HERE / "02_CODE_AND_DATA" / "data"
    if not (data_dir / "arrays.csv").exists(): data_dir = HERE / "data"
    required = [data_dir / name for name in ("arrays.csv", "conditions.csv", "compounds.csv", "arrays_summary.csv")]
    required += [OUT / name for name in ("review_generalization_disjoint_assignments.csv.gz",
                 "review_generalization_disjoint_availability.csv", "review_generalization_disjoint_repeats.csv",
                 "review_loss_pair_scores.csv", "review_generalization_smiles_identity_audit.csv")]
    before = {str(p.relative_to(HERE)): sha(p) for p in required}
    raw = pd.read_csv(data_dir / "arrays.csv", low_memory=False)
    raw["yield"] = pd.to_numeric(raw["yield"], errors="coerce")
    analysis = raw[raw["set"] == "analysis"].rename(columns={"array_id": "screen", "array": "screen_label"}).copy()
    analysis["subpair"] = analysis.sub_1_smiles + " >> " + analysis.sub_2_smiles
    identity = pd.read_csv(OUT / "review_generalization_smiles_identity_audit.csv")
    assert identity.deposited_smiles.nunique() == identity.canonical_isomeric_smiles.nunique() == len(identity)
    assert set(analysis.sub_1_smiles) | set(analysis.sub_2_smiles) <= set(identity.deposited_smiles)
    assert analysis["yield"].dropna().between(0, 1).all()
    frames = {}
    for screen, d in analysis.groupby("screen", sort=True):
        matrix = lib.screen_matrix(analysis, screen)
        meta = d.drop_duplicates("subpair").set_index("subpair").loc[matrix.index]
        frames[screen] = (matrix, meta, str(d.screen_label.iloc[0]))
    assignments = pd.read_csv(required[4])
    assignments = assignments[assignments.scheme == "both_unseen"]
    avail = pd.read_csv(required[5])
    avail = avail[avail.scheme == "both_unseen"]
    previous = pd.read_csv(required[6])
    previous = previous[(previous.scheme == "both_unseen") & (previous.k == 6)]
    oldpairs = pd.read_csv(required[7])
    assert len(avail) == len(frames)*20
    assert set(avail["repeat"]) == set(range(20))
    return frames, assignments, avail, previous, oldpairs, required, before, {
        "analysis_rows": len(analysis), "analysis_non_numeric_or_missing_yields": int(analysis["yield"].isna().sum()),
        "array_pair_instances": sum(len(x[0]) for x in frames.values()), "arrays": len(frames),
        "identity_policy": "Existing canonical-isomeric identity audit retained; no deposited textual aliases; salts/stereochemistry unchanged",
    }


def disjoint_tail(frames, assignments, availability, previous):
    split_rows, pair_rows, panels, checks = [], [], [], []
    av_lookup = availability.set_index(["screen", "repeat"])
    old_lookup = previous.set_index(["screen", "repeat", "control"])
    maximum_error = 0.
    for screen, (matrix, meta, label) in frames.items():
        y = matrix.to_numpy(float)
        n, c = y.shape
        allidx = np.arange(n)
        p1, p2 = meta.sub_1_smiles.to_numpy(), meta.sub_2_smiles.to_numpy()
        for rep, a in assignments[assignments.screen == screen].groupby("repeat", sort=True):
            a = a.sort_values("pair_index")
            assert np.array_equal(a.pair_index, allidx)
            assert np.array_equal(a.subpair, matrix.index)
            tr = a.loc[a.assignment == "train", "pair_index"].to_numpy(int)
            te = a.loc[a.assignment == "test", "pair_index"].to_numpy(int)
            discarded = a.loc[a.assignment == "discarded_cross_edge", "pair_index"].to_numpy(int)
            av = av_lookup.loc[(screen, rep)]
            assert (len(tr), len(te), len(discarded)) == (av.n_train, av.n_test, av.n_discarded_cross_edges)
            assert len(tr)+len(te)+len(discarded) == n
            train_mols = set(p1[tr]) | set(p2[tr])
            test_mols = set(p1[te]) | set(p2[te])
            assert not (train_mols & test_mols)
            checks.append({"screen": screen, "label": label, "repeat": int(rep), "availability": av.availability,
                           "n_train": len(tr), "n_test": len(te), "n_discarded_cross_edges": len(discarded),
                           "both_roles_disjoint_from_all_training_molecules": True})
            if av.availability != "ok": continue
            # Only the two original controls are reconstructed; no split is redrawn.
            same_tr = np.sort(rng_for("same_test_control", screen, "both_unseen", rep).choice(np.setdiff1d(allidx, te), len(tr), replace=False))
            perm = rng_for("random_pair_control", screen, "both_unseen", rep).permutation(n)
            controls = {"disjoint": (tr, te), "same_test_pair_control": (same_tr, te),
                        "random_pair_control": (np.sort(perm[:len(tr)]), np.sort(perm[len(tr):len(tr)+len(te)]))}
            for control, (ctr, cte) in controls.items():
                assert len(ctr) == len(tr) and len(cte) == len(te) and not (set(ctr) & set(cte))
                kit = lib.greedy_kit(y[ctr], 6)
                best, full = lib.retention(y[cte], kit)
                key = (screen, rep, control)
                if full.sum() > 0:
                    assert key in old_lookup.index
                    old = old_lookup.loc[key]
                    error = max(abs(best.sum()/full.sum()-old.retention_greedy), abs(full.mean()-old.mean_best_full), abs(best.mean()-old.mean_best_greedy))
                    maximum_error = max(maximum_error, float(error))
                    assert error < 1e-12, (key, error)
                else:
                    assert key not in old_lookup.index
                common = {"screen": screen, "label": label, "scheme": "both_unseen", "repeat": int(rep),
                          "control": control, "k": 6, "n_train": len(ctr), "n_test": len(cte),
                          "n_discarded_cross_edges_original": len(discarded), "retention": best.sum()/full.sum() if full.sum() else np.nan}
                panels.append(common | {"train_indices": ";".join(map(str, ctr)), "test_indices": ";".join(map(str, cte)),
                                        "panel_condition_ids": ";".join(matrix.columns[kit])})
                for denominator in DENOMINATORS:
                    num, den, rate = tail(full, best, denominator)
                    split_rows.append(common | {"denominator": denominator, "tail_count": num, "denominator_count": den,
                                                "tail_fraction": rate, "score_available": bool(den)})
                for i, pairidx in enumerate(cte):
                    pair_rows.append({"screen": screen, "repeat": int(rep), "control": control, "pair_index": int(pairidx),
                                      "subpair": matrix.index[pairidx], "full_best_yield": full[i], "panel_best_yield": best[i],
                                      "loss_pp": 100*(full[i]-best[i]), "loss_gt_20pp": bool(100*(full[i]-best[i]) > 20),
                                      "original_full_gt_0": bool(full[i] > 0), "original_full_ge_0p2": bool(full[i] >= .2)})
        print("reused both-unseen splits:", screen, flush=True)
    splits, pairs = pd.DataFrame(split_rows), pd.DataFrame(pair_rows)
    # Recount the exported pair-event representation independently of the
    # split-rate calculation, including both explicit denominators.
    for denominator in DENOMINATORS:
        e = pairs.assign(eligible=pairs[denominator], severe=pairs[denominator] & pairs.loss_gt_20pp)
        recounted = e.groupby(["screen", "repeat", "control"])[["eligible", "severe"]].sum()
        scores = splits[splits.denominator == denominator].set_index(["screen", "repeat", "control"])
        np.testing.assert_array_equal(recounted.loc[scores.index, "eligible"], scores.denominator_count)
        np.testing.assert_array_equal(recounted.loc[scores.index, "severe"], scores.tail_count)
    save(splits, "disjoint_splits")
    save(pairs, "disjoint_pair_events", compressed=True)
    save(pd.DataFrame(panels), "disjoint_panels")
    save(pd.DataFrame(checks), "disjoint_availability")
    return splits, pairs, {"attempted_splits_verified": len(checks), "eligible_original_splits": int((availability.availability == "ok").sum()),
                           "pair_event_recount_matches_all_split_tail_counts": True,
                           "original_k6_aggregate_scores_reproduced": len(previous), "maximum_original_score_error": maximum_error}


def summarize_disjoint(splits, pairs, bootstrap_draws):
    matched, per_array, pooled, summary = [], [], [], []
    for minimum in (3, 8):
        for denominator in DENOMINATORS:
            s = splits[(splits.n_test >= minimum) & (splits.denominator == denominator)]
            left = s[s.control == "disjoint"].set_index(["screen", "repeat"])
            for control in CONTROLS:
                right = s[s.control == control].set_index(["screen", "repeat"])
                m = left.join(right, lsuffix="_disjoint", rsuffix="_control", how="inner")
                m = m[m.tail_fraction_disjoint.notna() & m.tail_fraction_control.notna()].copy()
                m["delta_fraction"] = m.tail_fraction_disjoint-m.tail_fraction_control
                common = {"minimum_test_pairs": minimum, "denominator": denominator, "control": control}
                for (screen, rep), r in m.iterrows():
                    matched.append(common | {"screen": screen, "label": r.label_disjoint, "repeat": int(rep),
                        "tail_fraction_disjoint": r.tail_fraction_disjoint, "tail_fraction_control": r.tail_fraction_control,
                        "delta_fraction": r.delta_fraction, "tail_count_disjoint": int(r.tail_count_disjoint),
                        "tail_count_control": int(r.tail_count_control), "denominator_count_disjoint": int(r.denominator_count_disjoint),
                        "denominator_count_control": int(r.denominator_count_control), "n_train": int(r.n_train_disjoint), "n_test": int(r.n_test_disjoint)})
                for screen, group in m.groupby(level="screen"):
                    row = common | {"screen": screen, "label": group.label_disjoint.iloc[0], "matched_repeats": len(group),
                        "tail_fraction_disjoint": group.tail_fraction_disjoint.mean(), "tail_fraction_control": group.tail_fraction_control.mean(),
                        "delta_fraction": group.delta_fraction.mean(), "delta_split_q025": group.delta_fraction.quantile(.025),
                        "delta_split_q975": group.delta_fraction.quantile(.975), "mean_n_train": group.n_train_disjoint.mean(),
                        "min_n_test": int(group.n_test_disjoint.min()), "max_n_test": int(group.n_test_disjoint.max()),
                        "mean_discarded_cross_edges": group.n_discarded_cross_edges_original_disjoint.mean()}
                    for side in ("disjoint", "control"):
                        row["tail_events_"+side] = int(group["tail_count_"+side].sum())
                        row["denominator_events_"+side] = int(group["denominator_count_"+side].sum())
                    per_array.append(row)
                a = pd.DataFrame([r for r in per_array if all(r[k] == v for k, v in common.items())])
                for metric in ("tail_fraction_disjoint", "tail_fraction_control", "delta_fraction"):
                    for statistic in ("mean", "median"):
                        values = a[metric].to_numpy()
                        lo, hi = array_bootstrap(values, (minimum, denominator, control, metric, statistic), bootstrap_draws, statistic)
                        summary.append(common | {"metric": metric, "statistic": statistic, "n_arrays": len(a),
                            "matched_split_pairs": len(m), "estimate": float(np.mean(values) if statistic == "mean" else np.median(values)),
                            "ci_lo": lo, "ci_hi": hi, "weighting": "equal repeats within each array, then equal arrays",
                            "interval": "descriptive array-bootstrap percentile; repeated pairs/molecules are not independent experiments"})
                p = common | {"matched_split_pairs": len(m), "n_arrays": len(a), "weighting": "pooled eligible test-pair events across matched repeats; repeats can reuse pairs"}
                for side, selected_control in (("disjoint", "disjoint"), ("control", control)):
                    num, den = int(m["tail_count_"+side].sum()), int(m["denominator_count_"+side].sum())
                    selected = pairs[(pairs.control == selected_control) & pairs[denominator]].copy()
                    selected = selected.set_index(["screen", "repeat"])
                    selected = selected[selected.index.isin(m.index)]
                    p["tail_events_"+side], p["denominator_events_"+side] = num, den
                    p["tail_fraction_"+side] = num/den if den else np.nan
                    p["distinct_array_pairs_"+side] = len(selected.reset_index()[["screen", "pair_index"]].drop_duplicates())
                p["delta_fraction"] = p["tail_fraction_disjoint"]-p["tail_fraction_control"]
                pooled.append(p)
    save(pd.DataFrame(matched), "disjoint_matched_splits")
    save(pd.DataFrame(per_array), "disjoint_per_array")
    save(pd.DataFrame(summary), "disjoint_summary")
    save(pd.DataFrame(pooled), "disjoint_pooled")
    return pd.DataFrame(summary), pd.DataFrame(pooled)


def frozen_noise(frames, oldpairs, replicates):
    rows, verification, frozen_panels = [], [], []
    for screen, (matrix, meta, label) in frames.items():
        y = matrix.to_numpy(float)
        finite = np.isfinite(y)
        observed_zero = finite & (y == 0)
        n, c = y.shape
        old = oldpairs[oldpairs.screen == screen].set_index("subpair").loc[matrix.index]
        columns = {condition: i for i, condition in enumerate(matrix.columns)}
        panel = np.array([[columns[name] for name in names.split(";")] for names in old.greedy_panel], int)
        assert panel.shape == (n, 6)
        for fold, group in old.groupby("fold"):
            assert group.greedy_panel.nunique() == 1
            frozen_panels.append({"screen": screen, "fold": int(fold), "test_pairs": len(group),
                                  "panel_condition_ids": group.greedy_panel.iloc[0], "source": "review_loss_pair_scores.csv; no noise refitting"})
        full = np.max(np.where(finite, y, 0), axis=1)
        best = np.max(np.take_along_axis(np.where(finite, y, 0), panel, axis=1), axis=1)
        np.testing.assert_allclose(full, old.full_best_yield, atol=1e-14, rtol=0)
        np.testing.assert_allclose(best, old.greedy_best_yield, atol=1e-14, rtol=0)
        np.testing.assert_allclose(100*(full-best), old.greedy_loss_pp, atol=1e-11, rtol=0)
        original_tail = 100*(full-best) > 20
        verification.append({"screen": screen, "pairs": n, "matrix_cells": y.size,
                             "observed_zero_cells": int(observed_zero.sum()), "missing_matrix_cells": int((~finite).sum()),
                             "maximum_full_score_error": float(np.max(np.abs(full-old.full_best_yield))),
                             "maximum_panel_score_error": float(np.max(np.abs(best-old.greedy_best_yield)))})
        cv_values = np.asarray(CVS)
        for rep in range(replicates):
            # One independent standard Normal per observed cell; common draws
            # across CV scenarios make their Monte Carlo comparison paired.
            z = rng_for("tail_noise", screen, rep).normal(size=y.shape)
            nf, nb = perturb_and_score(y, panel, z, cv_values)
            assert np.array_equal(nf[0], full) and np.array_equal(nb[0], best)
            loss = 100*(nf-nb)
            for ci, cv in enumerate(CVS):
                if cv == 0 and rep: continue
                for denominator in DENOMINATORS:
                    keep = mask_for(full, denominator)  # ALWAYS the original full maximum
                    den = int(keep.sum())
                    severe = loss[ci] > 20
                    num = int((severe & keep).sum())
                    original_num = int((original_tail & keep).sum())
                    rows.append({"screen": screen, "label": label, "cv": cv, "repeat": rep, "denominator": denominator,
                        "denominator_count_fixed": den, "tail_count": num, "tail_fraction": num/den if den else np.nan,
                        "original_tail_count": original_num, "original_tail_fraction": original_num/den if den else np.nan,
                        "new_tail_events": int((severe & ~original_tail & keep).sum()),
                        "lost_tail_events": int((~severe & original_tail & keep).sum()),
                        "loss_median_pp": float(np.median(loss[ci, keep])) if den else np.nan,
                        "loss_p90_pp": float(np.quantile(loss[ci, keep], .9)) if den else np.nan,
                        "sum_full_yield": float(nf[ci, keep].sum()), "sum_panel_yield": float(nb[ci, keep].sum())})
        print("frozen-panel noise scenarios:", screen, flush=True)
    repeats = pd.DataFrame(rows)
    save(repeats, "noise_per_array_repeats", compressed=True)
    save(pd.DataFrame(verification), "noise_reproduction")
    save(pd.DataFrame(frozen_panels), "noise_frozen_panels")
    corpus = []
    for (cv, rep, denominator), group in repeats.groupby(["cv", "repeat", "denominator"]):
        den, num = int(group.denominator_count_fixed.sum()), int(group.tail_count.sum())
        original_num = int(group.original_tail_count.sum())
        corpus.append({"cv": cv, "repeat": rep, "denominator": denominator, "denominator_count_fixed": den,
            "tail_count": num, "tail_fraction": num/den, "original_tail_count": original_num,
            "original_tail_fraction": original_num/den, "delta_fraction": (num-original_num)/den,
            "array_mean_tail_fraction": group.tail_fraction.mean(), "array_median_tail_fraction": group.tail_fraction.median(),
            "new_tail_events": int(group.new_tail_events.sum()), "lost_tail_events": int(group.lost_tail_events.sum()),
            "retention_in_fixed_stratum": group.sum_panel_yield.sum()/group.sum_full_yield.sum()})
    corpus = pd.DataFrame(corpus)
    zero = corpus[(corpus.cv == 0) & (corpus.denominator == "original_full_gt_0")].iloc[0]
    assert (zero.tail_count, zero.denominator_count_fixed) == (253, 1645)
    assert zero.tail_fraction == 253/1645
    assert np.isclose(zero.tail_fraction*100, 15.379939209726443, atol=1e-12)
    for den, group in corpus.groupby("denominator"):
        assert group.denominator_count_fixed.nunique() == 1
    np.testing.assert_array_equal(corpus.tail_count-corpus.original_tail_count, corpus.new_tail_events-corpus.lost_tail_events)
    save(corpus, "noise_corpus_repeats")
    summaries = []
    for (cv, denominator), group in corpus.groupby(["cv", "denominator"]):
        for metric in ("tail_fraction", "delta_fraction", "array_mean_tail_fraction", "array_median_tail_fraction",
                       "new_tail_events", "lost_tail_events", "retention_in_fixed_stratum"):
            values = group[metric].to_numpy()
            summaries.append({"cv": cv, "denominator": denominator, "metric": metric, "replicates": len(values),
                "denominator_count_fixed": int(group.denominator_count_fixed.iloc[0]), "mc_mean": values.mean(),
                "mc_q025": np.quantile(values, .025), "mc_q975": np.quantile(values, .975),
                "mc_mean_se": values.std(ddof=1)/np.sqrt(len(values)) if len(values) > 1 else 0.,
                "interpretation": "Monte Carlo scenario distribution conditional on reported yields and frozen OOF panels; not empirical measurement uncertainty"})
    summary = pd.DataFrame(summaries)
    save(summary, "noise_summary")
    per_array = repeats.groupby(["screen", "label", "cv", "denominator"]).agg(
        replicates=("tail_fraction", "size"), denominator_count_fixed=("denominator_count_fixed", "first"),
        original_tail_fraction=("original_tail_fraction", "first"), mc_mean=("tail_fraction", "mean"),
        mc_q025=("tail_fraction", lambda x: x.quantile(.025)), mc_q975=("tail_fraction", lambda x: x.quantile(.975))).reset_index()
    save(per_array, "noise_per_array_summary")
    return summary, corpus, {"zero_noise_tail_count": int(zero.tail_count), "zero_noise_productive_denominator": int(zero.denominator_count_fixed),
                            "zero_noise_tail_fraction": float(zero.tail_fraction), "frozen_greedy_panels": len(frozen_panels),
                            "zero_noise_exact_reproduction": True, "zeros_and_missing_structure_preserved": True}


def pct(x): return f"{100*x:.2f}"


def write_report(disjoint, pooled, noise, manifest):
    lines = ["# R3 tail robustness — retrospective sensitivity analyses", "",
        "The existing 20 both-unseen molecular partitions are retained. These calculations reuse reported yields; no new experiments were performed.", "",
        "## Molecule-disjoint tail rates", "",
        "All panels contain six greedily chosen conditions. Both reactant identities of every disjoint test pair are absent from all training molecules, in either role. Cross edges remain discarded. The same-test control retains exactly the disjoint test pairs and matches training size; the random-pair control matches both train and test sizes. Control RNG streams are reconstructed exactly from the existing code and all available k=6 aggregate scores are checked against the saved outputs.", "",
        "The loss is 100 × (full-array maximum − selected-panel maximum), with a strict >20 percentage-point tail. Denominators are reported separately for full>0 and full≥0.2. The minimum-test thresholds 3 and 8 apply to all test pairs before this denominator restriction. Splits with zero eligible denominator on either side are excluded only from that paired contrast; their counts remain in the split output.", "",
        "Within an array, matched repeat-level rates and their paired differences are averaged equally. The table below then averages arrays equally. Intervals resample those per-array means 10,000 times. They are descriptive array-bootstrap percentile intervals; arrays/campaigns may also share chemistry, and repeated pairs or molecules are not independent experiments. Median-over-array summaries are separately released. Positive differences mean more severe losses in the both-unseen test.", "",
        "| Minimum test pairs | Fixed denominator | Control | Arrays / paired splits | Disjoint % | Control % | Paired difference pp (descriptive 95% interval) |",
        "|---|---|---|---:|---:|---:|---:|"]
    for (minimum, denominator, control), g in disjoint[disjoint.statistic == "mean"].groupby(["minimum_test_pairs", "denominator", "control"]):
        x = g.set_index("metric")
        d = x.loc["delta_fraction"]
        lines.append(f"| {minimum} | {denominator} | {control} | {int(d.n_arrays)} / {int(d.matched_split_pairs)} | {pct(x.loc['tail_fraction_disjoint','estimate'])} | {pct(x.loc['tail_fraction_control','estimate'])} | {pct(d.estimate)} ({pct(d.ci_lo)}, {pct(d.ci_hi)}) |")
    lines += ["", "Pooled results below weight test-pair events. A pair evaluated in several repeats contributes several events; these denominators are not counts of independent experiments. The pooled rate difference is not the equal-array paired contrast above.", "",
        "| Minimum test pairs | Fixed denominator | Control | Disjoint tail / eligible events | Control tail / eligible events | Pooled difference pp | Distinct eligible array-pairs, disjoint / control |",
        "|---|---|---|---:|---:|---:|---:|"]
    for r in pooled.itertuples():
        lines.append(f"| {r.minimum_test_pairs} | {r.denominator} | {r.control} | {r.tail_events_disjoint}/{r.denominator_events_disjoint} | {r.tail_events_control}/{r.denominator_events_control} | {pct(r.delta_fraction)} | {r.distinct_array_pairs_disjoint}/{r.distinct_array_pairs_control} |")
    lines += ["", "## Frozen-panel multiplicative-noise scenarios", "",
        "The original ten-fold pair-CV panel for each test pair is loaded from review_loss_pair_scores.csv and never refitted. Every observed test-matrix cell is perturbed as y′ = clip[y × (1 + Normal(0,CV)), 0, 1]. Full-array and panel maxima are recomputed from exactly the same perturbed matrix; noise is not added only to the full maximum. Missing cells remain missing and retain the original scoring convention (zero if no panel yield is observed). Numerical zeros stay zero. The original full>0/full≥0.2 membership is frozen for every scenario, so denominators cannot drift.", "",
        "CV values 0.057, 0.06, 0.069, 0.10 and 0.20 are hypothetical scenarios regardless of whether a platform precision number is verified. Independent Normal errors across wells and molecules are assumed; common standard-Normal draws across CV scenarios pair the Monte Carlo comparisons. These assumptions omit correlated/run-level errors, absolute background errors, zero-censoring mechanisms, calibration bias and model-fitting uncertainty. Adding noise to already measured yields is a sensitivity perturbation, not recovery of latent true yields or a decomposition of observed loss. It cannot establish an empirical noise ceiling or the fraction of real losses explained by noise.", "",
        f"There are {manifest['noise_replicates']} replicates for each nonzero CV, with fixed master seed {SEED}. Zero noise is evaluated once and exactly reproduces 253/1,645 = 15.379939% for productive pairs. All noise intervals below are the 2.5th–97.5th percentiles across simulated scenarios, not confidence intervals for experimental error. Monte Carlo standard errors are also released.", "",
        "| CV scenario | Fixed denominator | Eligible original pairs | Mean tail % | Monte Carlo 2.5–97.5% | Mean change from zero-noise tail, pp |",
        "|---:|---|---:|---:|---:|---:|"]
    for (cv, denominator), g in noise.groupby(["cv", "denominator"]):
        x = g.set_index("metric")
        r = x.loc["tail_fraction"]
        lines.append(f"| {cv:.3f} | {denominator} | {int(r.denominator_count_fixed)} | {pct(r.mc_mean)} | {pct(r.mc_q025)}–{pct(r.mc_q975)} | {pct(x.loc['delta_fraction','mc_mean'])} |")
    lines += ["", "## Reproduction and files", "",
        f"All {manifest['disjoint']['attempted_splits_verified']} original both-unseen assignments passed the molecule-disjoint and partition-size checks. Of these, {manifest['disjoint']['eligible_original_splits']} meet the original computational eligibility criteria. Reconstructed scores match {manifest['disjoint']['original_k6_aggregate_scores_reproduced']} saved k=6 rows with maximum absolute difference {manifest['disjoint']['maximum_original_score_error']:.3g}. Frozen OOF scores, zero-noise outcomes, denominator constancy and preserved zero/missing structure are asserted. Raw CSV and reference-output hashes are unchanged after the run.", "",
        "- `out/review_tail_robustness_disjoint_splits.csv`: every reconstructed split/control/denominator, including undefined rates.",
        "- `disjoint_matched_splits`, `disjoint_per_array`, `disjoint_summary`, `disjoint_pooled`: paired repeat, array and pooled summaries with explicit weights.",
        "- `disjoint_pair_events.csv.gz`, `disjoint_panels`, `disjoint_availability`: exact scored events, panel members/train/test indices, and all 360 attempted partitions.",
        "- `noise_per_array_repeats.csv.gz`, `noise_corpus_repeats`, `noise_summary`, `noise_per_array_summary`: scenario draws and summaries with fixed denominators.",
        "- `noise_frozen_panels`, `noise_reproduction`: original panel identities and score/matrix checks.",
        "- `out/review_tail_robustness_manifest.json`: input hashes, seed, settings, runtime and verification results. All abbreviated filenames above carry the `review_tail_robustness_` prefix and `.csv` unless stated otherwise.", "",
        "Run `python -X utf8 review_tail_robustness_gc.py --noise-replicates 200`. No upstream output, source data or manuscript is regenerated.", ""]
    report = _hist("review_round3_2026-09-20") / "TAIL_ROBUSTNESS.md"
    report.parent.mkdir(exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise-replicates", type=int, default=200)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    args = parser.parse_args()
    assert args.noise_replicates >= 100, "At least 100 Monte Carlo replicates are required"
    started = time.perf_counter()
    synthetic = synthetic_checks()
    frames, assignments, avail, previous, oldpairs, inputs, hashes, data_meta = load_inputs()
    splits, pairs, disjoint_checks = disjoint_tail(frames, assignments, avail, previous)
    dsummary, pooled = summarize_disjoint(splits, pairs, args.bootstrap_draws)
    nsummary, corpus, noise_checks = frozen_noise(frames, oldpairs, args.noise_replicates)
    after = {str(p.relative_to(HERE)): sha(p) for p in inputs}
    assert hashes == after, "A raw/reference input changed during execution"
    manifest = {"script": Path(__file__).name, "script_sha256": sha(Path(__file__)), "master_seed": SEED,
        "noise_cvs": list(CVS), "noise_replicates": args.noise_replicates, "array_bootstrap_draws": args.bootstrap_draws,
        "input_sha256_before": hashes, "input_sha256_after": after, "all_inputs_unchanged": hashes == after,
        "python_version": platform.python_version(), "numpy_version": np.__version__, "pandas_version": pd.__version__, "data": data_meta,
        "disjoint": disjoint_checks, "noise": noise_checks, "synthetic_checks": synthetic,
        "runtime_seconds": round(time.perf_counter()-started, 2)}
    (OUT / (PREFIX+"manifest.json")).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(dsummary, pooled, nsummary, manifest)
    print(json.dumps({k: v for k, v in manifest.items() if not k.startswith("input_sha256")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
