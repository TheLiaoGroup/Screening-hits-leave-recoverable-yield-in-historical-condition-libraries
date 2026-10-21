"""Post-review generalization and extended-curve audit using only deposited data.

Run: python review_generalization_gc.py
No experimental data, frozen partitions, or original analyses are modified.
See out/review_generalization_methods.md for the full sampling definitions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase

import lib

HERE = Path(__file__).resolve().parent
SEED = 20260920
KS = (1, 3, 6)


def rng_for(*parts):
    """Stable separate streams: adding analyses does not alter existing draws."""
    digest = hashlib.sha256("|".join(map(str, (SEED,) + parts)).encode()).digest()
    return np.random.RandomState(int.from_bytes(digest[:4], "little"))


def save(df, name):
    df.to_csv(HERE / "out" / name, index=False)


def mols(pairs):
    return set(pairs.sub_1_smiles) | set(pairs.sub_2_smiles)


def describe_pairs(test, train_mols, role1=None, role2=None):
    s1 = test.sub_1_smiles.isin(train_mols).to_numpy()
    s2 = test.sub_2_smiles.isin(train_mols).to_numpy()
    out = test.copy()
    out["partner1_seen_any_role"] = s1
    out["partner2_seen_any_role"] = s2
    out["novelty"] = np.select([s1 & s2, ~s1 & ~s2], ["both_seen", "both_unseen"], default="one_unseen")
    if role1 is not None:
        out["partner1_seen_same_role"] = test.sub_1_smiles.isin(role1).to_numpy()
        out["partner2_seen_same_role"] = test.sub_2_smiles.isin(role2).to_numpy()
    return out


def bootstrap(x, statistic="median", draws=10000, key="default"):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return (np.nan, np.nan)
    r = rng_for("bootstrap", key)
    samples = x[r.randint(0, len(x), (draws, len(x)))]
    b = np.median(samples, axis=1) if statistic == "median" else np.mean(samples, axis=1)
    return tuple(np.quantile(b, [.025, .975]))


def summarize_values(values, key, statistic="median"):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    lo, hi = bootstrap(a, statistic, key=key)
    return {"n_arrays": len(a), "statistic": statistic,
            "estimate": float(np.median(a) if statistic == "median" else np.mean(a)),
            "ci_lo": lo, "ci_hi": hi}


def pair_ratio_ci(best, full, key, draws=10000):
    """Conditional fixed-kit descriptive CI; does not cluster shared molecules."""
    rr = rng_for("pair_ratio_ci", key)
    idx = rr.randint(0, len(full), (draws, len(full)))
    num, den = best[idx].sum(axis=1), full[idx].sum(axis=1)
    ok = den > 0
    q = np.quantile(num[ok]/den[ok], [.025, .975]) if ok.any() else [np.nan, np.nan]
    return {"pair_bootstrap_ci_lo": q[0], "pair_bootstrap_ci_hi": q[1],
            "pair_bootstrap_valid_draws": int(ok.sum())}


def pair_metadata(frame, matrix):
    p = frame[["screen", "screen_label", "subpair", "sub_1_smiles", "sub_2_smiles", "role_1", "role_2"]].drop_duplicates("subpair")
    return p.set_index("subpair").loc[matrix.index].reset_index()


def cv_curve(Y):
    """Original kits.py 10 folds, same seed and 50 random kits per k/fold.

    For k<=12 the original RNG stream is preserved byte-for-byte. A separate
    stream supplies new k=13..24 draws so prior baselines remain reproducible.
    """
    n, c = Y.shape
    rng = np.random.RandomState(3)
    extra = np.random.RandomState(SEED)
    folds = np.array_split(rng.permutation(n), 10)
    pred = {method: np.zeros((n, 24)) for method in ("greedy", "random", "meanyield")}
    fold_id = np.empty(n, dtype=int)
    full = lib.retention(Y, list(range(c)))[1]
    for fi, f in enumerate(folds):
        tr = np.setdiff1d(np.arange(n), f)
        fold_id[f] = fi
        kit = lib.greedy_kit(Y[tr], 24)
        rank = np.argsort(-np.nanmean(Y[tr], axis=0))
        for k in range(1, 25):
            pred["greedy"][f, k-1] = lib.retention(Y[f], kit[:k])[0]
            pred["meanyield"][f, k-1] = lib.retention(Y[f], list(rank[:k]))[0]
            rr = rng if k <= 12 else extra
            pred["random"][f, k-1] = np.mean([lib.retention(Y[f], list(rr.choice(c, k, replace=False)))[0] for _ in range(50)], axis=0)
    return pred, full, folds, fold_id


def score(Y, tr, te):
    kit = lib.greedy_kit(Y[tr], max(KS))
    rank = np.argsort(-np.nanmean(Y[tr], axis=0))
    full = lib.retention(Y[te], list(range(Y.shape[1])))[1]
    if full.sum() <= 0:
        return []
    rows = []
    for k in KS:
        best = lib.retention(Y[te], kit[:k])[0]
        meanbest = lib.retention(Y[te], list(rank[:k]))[0]
        rows.append({"k": k, "retention_greedy": best.sum()/full.sum(),
                     "retention_meanyield": meanbest.sum()/full.sum(),
                     "mean_best_full": full.mean(), "mean_best_greedy": best.mean(),
                     "test_nonzero_full": int((full > 0).sum())})
    return rows


def disjoint_experiments(screen, label, Y, pairs, repeats):
    rows, availability, partition_rows = [], [], []
    n = len(pairs)
    allidx = np.arange(n)
    p1, p2 = pairs.sub_1_smiles.to_numpy(), pairs.sub_2_smiles.to_numpy()
    universe = np.array(sorted(mols(pairs)))
    for scheme in ("partner1_unseen", "partner2_unseen", "both_unseen"):
        fraction = .3 if scheme == "both_unseen" else .2
        candidates = universe if scheme == "both_unseen" else np.unique(p1 if scheme == "partner1_unseen" else p2)
        nhold = max(1, int(np.ceil(len(candidates)*fraction)))
        for rep in range(repeats):
            r = rng_for("disjoint", screen, scheme, rep)
            held = set(r.choice(candidates, nhold, replace=False))
            a, b = np.isin(p1, list(held)), np.isin(p2, list(held))
            tr = allidx[~a & ~b]
            te = allidx[a & b] if scheme == "both_unseen" else allidx[a if scheme == "partner1_unseen" else b]
            excluded = np.setdiff1d(allidx, np.union1d(tr, te))
            tm = set(p1[tr]) | set(p2[tr])
            assert not (tm & held)
            if scheme == "both_unseen":
                assert not ((set(p1[te]) | set(p2[te])) & tm)
            reason = "ok" if len(tr) >= 10 and len(te) >= 3 else "train_below_10" if len(tr) < 10 else "test_below_3"
            if reason == "ok" and lib.retention(Y[te], list(range(Y.shape[1])))[1].sum() <= 0:
                reason = "zero_full_yield"
            base = {"screen": screen, "label": label, "scheme": scheme, "repeat": rep,
                    "held_molecule_fraction_requested": fraction, "held_molecules": nhold,
                    "candidate_molecules": len(candidates), "n_pairs": n,
                    "n_train": len(tr), "n_test": len(te), "n_discarded_cross_edges": len(excluded),
                    "n_test_molecules": len(set(p1[te]) | set(p2[te])), "availability": reason}
            availability.append(base)
            assignments = np.full(n, "discarded_cross_edge", dtype=object)
            assignments[tr] = "train"
            assignments[te] = "test"
            partition_rows.extend({"screen": screen, "scheme": scheme, "repeat": rep,
                                   "pair_index": i, "subpair": pairs.subpair.iloc[i], "assignment": assignments[i]}
                                  for i in allidx)
            if reason != "ok":
                continue
            controls = {"disjoint": (tr, te)}
            # Same held-out pairs; only whether training can share their molecules changes.
            rr = rng_for("same_test_control", screen, scheme, rep)
            controls["same_test_pair_control"] = (np.sort(rr.choice(np.setdiff1d(allidx, te), len(tr), replace=False)), te)
            # Conventional random-pair holdout, exact same train/test sample sizes.
            perm = rng_for("random_pair_control", screen, scheme, rep).permutation(n)
            controls["random_pair_control"] = (np.sort(perm[:len(tr)]), np.sort(perm[len(tr):len(tr)+len(te)]))
            availability[-1]["same_test_extra_candidate_pairs"] = len(excluded)
            for control, (ctr, cte) in controls.items():
                assert not (set(ctr) & set(cte))
                cm = set(p1[ctr]) | set(p2[ctr])
                bothseen = np.isin(p1[cte], list(cm)) & np.isin(p2[cte], list(cm))
                bothunseen = ~np.isin(p1[cte], list(cm)) & ~np.isin(p2[cte], list(cm))
                results = score(Y, ctr, cte)
                availability[-1][control+"_score_available"] = bool(results)
                for result in results:
                    rows.append(base | {"control": control,
                                        "test_both_seen_fraction": bothseen.mean(),
                                        "test_both_unseen_fraction": bothunseen.mean()} | result)
    return rows, availability, partition_rows


def sensitivity_experiments(screen, label, Y, repeats):
    rows = []
    n = len(Y)
    for rep in range(repeats):
        perm = rng_for("training_size", screen, rep).permutation(n)
        nt = max(3, int(np.ceil(.2*n)))
        te, pool = perm[:nt], perm[nt:]
        for frac in (.2, .4, .6, .8):
            ntr = min(len(pool), max(1, int(np.floor(frac*n))))
            tr = pool[:ntr]
            for result in score(Y, tr, te):
                rows.append({"screen": screen, "label": label, "repeat": rep,
                             "train_fraction_requested": frac, "n_train": ntr,
                             "n_test": len(te), "n_pairs": n} | result)
    return rows


METHODS = """# Post-review generalization and extended-curve analyses

These are retrospective post-review analyses, not a prespecified protocol or new
experiments. They use the deposited `02_CODE_AND_DATA/data/arrays.csv` without
changing its analysis/sealed/reserve/replicate labels. All source-validation flags are
retained. Yield is coerced to numeric; invalid values are treated as missing.
Duplicate measured pair-condition cells are averaged, as in `lib.screen_matrix`.
Missing wells are ignored when finding measured maxima, contribute no observed
success when all selected wells are missing, and contribute zero to training
mean-yield ranking, exactly as in `lib.py` and `kits.py`. Retention is the ratio
of mean selected-kit best yield to mean full-array best yield on the same test
pairs; it is not the mean of per-pair ratios. No yield normalization or retraining
uses test yields. The baseline full-array maximum itself requires retrospective
labels. Full measured labels are retained for every included pair.

## Identity and overlap

A molecule identity is an exact deposited substrate SMILES string, independently
audited against RDKit canonical isomeric SMILES. All 927 strings parse and map
one-to-one to 927 canonical isomeric SMILES, so canonicalization does not change
identity groups or novelty counts. All components, salts, and specified
stereochemistry are retained. No scaffold or tautomer standardization is implied.
Both partner roles are pooled when asking whether a molecule was seen,
and role-specific flags are also exported. A both-unseen pair has neither
molecule in the comparison training set, regardless of reactant role.
Within-array held-out novelty compares sealed/reserve pairs with that array's
analysis molecules. Global novelty compares with the entire analysis corpus.
For pair CV, within-array training is the other nine folds; global context adds
all other arrays' analysis molecules, which are not actually used to select that
array's kit. The original CV folds are ten folds with RandomState(3).

## Disjoint sample-out comparisons

Only the frozen analysis set is resplit. There are 20 repeats by default, with
fixed master seed 20260920 and SHA-256-derived, named streams per array/scheme/
repeat. Partner-1 and partner-2 tests sample ceil(20%) of distinct molecules in
the specified role. All pairs with the sampled molecule in that role are test
pairs. Training excludes pairs containing any sampled molecule in either role;
other edges involving a held-out molecule are discarded.

Both-unseen tests sample ceil(30%) of the union of molecule identities in either
role. Training contains edges with both endpoints outside that held-out set;
test contains edges with both endpoints in it. Mixed train/held-out edges are
discarded and counted. Thus neither molecule of any both-unseen test pair occurs
in training, including in the opposite role. The 30% node fraction yields about
9% test edges under independence, limiting empty tests in small arrays. Splits
with fewer than 10 training pairs, fewer than 3 test pairs, or zero test full-
array yield are unavailable rather than assigned retention zero. Every attempted
split and pair assignment is exported. At least 3 test pairs is a computational
availability rule, not a claim of precision; a >=8-test-pair sensitivity is also
reported. Training contains fewer pairs than standard 10-fold CV by design.

Each valid split has two controls at exactly matched training and test sizes:
(1) `same_test_pair_control` uses exactly the same test pairs, but draws training
pairs uniformly from every other analysis pair, allowing molecule overlap;
(2) `random_pair_control` randomly partitions all analysis pairs into training,
test, and unused pairs at the same sizes. The first directly compares selectors
on identical test outcomes; the second also includes test-composition changes.
The training/test pair sets never overlap. Greedy and training mean-yield kits
are scored at k=1,3,6. Different repeats can reuse pairs and molecules.
For one-partner splits, the same-test control is usually degenerate: test pairs
already contain all occurrences of the held-out role's molecules, leaving the
same training set. The extra eligible candidate count is therefore exported;
the conventional matched random-pair control is the relevant comparator for
those schemes. A zero full-yield denominator makes any control score undefined;
the matched contrast uses only repeats with both scores available and reports
that count. Thus control-specific disjoint summaries can differ slightly.

Per-array estimates average retention over available repeats. Disjoint-minus-
control contrasts are paired within each repeat before averaging. The across-
array mean and median of those paired differences, and separate method levels,
receive percentile 95% intervals from 10,000 array bootstrap resamples. These
are descriptive array-resampling intervals; arrays share a source platform and
some chemistry and are not independent population samples. Repeat quantiles
are labelled split-variability intervals, not confidence intervals. Per-array
point contrasts, training/test counts, and availability accompany pooled values.

## Held-out versus CV and curves through k=24

The original ten-fold pair CV, greedy selection, training-mean ranking, and 50
random kits per k/fold are retained. For k<=12 the original random-number stream
is preserved; new k=13..24 random draws use a separate RandomState(20260920).
This avoids changing old random baselines merely by extending the curve. Saved
k<=12 curves are checked numerically against the original output.

Held-out kits are trained on all analysis pairs of the same array, then tested
on its unchanged sealed or reserve pairs. Comparisons with pair CV are matched
array by array; separately subtracting unmatched medians is not used. Both the
median and mean of per-array heldout-minus-CV differences have the descriptive
array-bootstrap intervals above. Different train sizes and test compositions
mean this contrast is not a controlled causal estimate of novelty alone.
Frozen held-out retention also has per-array percentile 95% intervals from
10,000 bootstrap resamples of test pairs, with the selected kit fixed. These
describe conditional pair-sampling uncertainty only: shared molecules violate
independence, and the intervals do not capture kit-selection or laboratory
variation. Zero-denominator bootstrap draws are omitted and counted.

Training-size sensitivity fixes a random 20% test sample on each repeat, then
uses nested random training samples of approximately 20,40,60,80% of all analysis
pairs, at k=1,3,6. Test pairs are never used in training; rounding is exported.
It measures training-size sensitivity on pair splits, not molecule extrapolation.

All analyses estimate reuse within a single platform's existing condition menus.
Molecule-disjoint testing is stronger than pair-CV but does not establish novel-
scaffold, new-laboratory, new-instrument, or prospective experimental validity.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    start = time.perf_counter()
    (HERE / "out").mkdir(exist_ok=True)
    if lib.DATASET is None:
        raise FileNotFoundError("Released data/arrays.csv is required for the generalization analysis")
    path = lib.DATASET / "arrays.csv"
    raw = pd.read_csv(path, low_memory=False)
    raw = raw[raw["measurement_status"].isin(["quantified", "zero_by_rule"])].copy()  # measured cells only
    raw["yield"] = pd.to_numeric(raw["yield"], errors="coerce")
    bad = int(raw["yield"].isna().sum())
    raw = raw.rename(columns={"array_id": "screen", "array": "screen_label"})
    assert raw[["sub_1_smiles", "sub_2_smiles"]].notna().all().all()
    identities = sorted(set(raw.sub_1_smiles) | set(raw.sub_2_smiles))
    identity_rows = []
    for smiles in identities:
        molecule = Chem.MolFromSmiles(smiles)
        assert molecule is not None, ("invalid substrate SMILES", smiles)
        identity_rows.append({"deposited_smiles": smiles,
                              "canonical_isomeric_smiles": Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)})
    identity = pd.DataFrame(identity_rows)
    assert identity.canonical_isomeric_smiles.nunique() == len(identity), "Canonical aliases now exist; re-key molecule identities before analysis."
    save(identity, "review_generalization_smiles_identity_audit.csv")
    raw["subpair"] = raw.sub_1_smiles + " >> " + raw.sub_2_smiles
    data = raw[raw.set != "replicate"].copy()
    analysis = data[data.set == "analysis"]
    apairs = analysis[["screen", "subpair", "sub_1_smiles", "sub_2_smiles"]].drop_duplicates()
    all_analysis_mols = mols(apairs)
    curve_rows, cvpair_rows, holdout_rows, hp_rows, novelty_rows, mol_rows = [], [], [], [], [], []
    disjoint_rows, availability_rows, partition_rows, size_rows = [], [], [], []
    matrices = {}
    for screen in sorted(analysis.screen.unique()):
        d = analysis[analysis.screen == screen]
        label = d.screen_label.iloc[0]
        M = lib.screen_matrix(analysis, screen)
        assert M.shape[1] >= 24
        Y = M.to_numpy(dtype=float)
        pairs = pair_metadata(d, M)
        matrices[screen] = M
        pred, full, folds, fold_ids = cv_curve(Y)
        for k in range(1, 25):
            curve_rows.append({"screen": screen, "label": label, "n_pairs": len(Y), "n_conditions": Y.shape[1], "k": k,
                               **{"retention_"+m: float(p[:, k-1].sum()/full.sum()) for m, p in pred.items()},
                               "mean_best_full": full.mean()})
        for k in (1, 3, 6, 12, 24):
            cvpair_rows.extend({"screen": screen, "subpair": pairs.subpair.iloc[i], "fold": int(fold_ids[i]),
                                "k": k, "full": full[i], **{m: p[i, k-1] for m, p in pred.items()}} for i in range(len(Y)))
        otherpairs = apairs[apairs.screen != screen]
        other = mols(otherpairs)
        for fi, f in enumerate(folds):
            train = pairs.drop(f)
            train_mols = mols(train)
            for scope, comparison in (("within_array", train_mols), ("global_corpus", train_mols | other)):
                role1 = set(train.sub_1_smiles) | (set(otherpairs.sub_1_smiles) if scope == "global_corpus" else set())
                role2 = set(train.sub_2_smiles) | (set(otherpairs.sub_2_smiles) if scope == "global_corpus" else set())
                nv = describe_pairs(pairs.iloc[f], comparison, role1, role2)
                nv["set"] = "pair_cv"
                nv["scope"] = scope
                nv["fold"] = fi
                novelty_rows.extend(nv.to_dict("records"))
        kit = lib.greedy_kit(Y, 24)
        rank = np.argsort(-np.nanmean(Y, axis=0))
        for which in ("sealed", "reserve"):
            hd = data[(data.screen == screen) & (data.set == which)]
            if not len(hd):
                continue
            HM = lib.screen_matrix(hd, screen).reindex(columns=M.columns)
            hp = pair_metadata(hd, HM)
            assert not (set(M.index) & set(HM.index)), screen
            hy = HM.to_numpy(dtype=float)
            hf = lib.retention(hy, list(range(Y.shape[1])))[1]
            for scope, comparison in (("within_array", mols(pairs)), ("global_corpus", all_analysis_mols)):
                comparison_pairs = apairs if scope == "global_corpus" else pairs
                nv = describe_pairs(hp, comparison, set(comparison_pairs.sub_1_smiles), set(comparison_pairs.sub_2_smiles))
                nv["set"], nv["scope"], nv["fold"] = which, scope, -1
                novelty_rows.extend(nv.to_dict("records"))
                pm = mols(hp)
                mol_rows.append({"screen": screen, "label": label, "set": which, "scope": scope,
                                 "n_pairs": len(hp), "n_molecules": len(pm), "n_new_molecules": len(pm-comparison),
                                 "n_seen_molecules": len(pm & comparison), "comparison_molecules": len(comparison)})
            for k in (1, 3, 6, 12, 24):
                hb = lib.retention(hy, kit[:k])[0]
                hmean = lib.retention(hy, list(rank[:k]))[0]
                holdout_rows.append({"screen": screen, "label": label, "set": which, "k": k,
                                     "n_train": len(Y), "n_test": len(hy), "retention_heldout": hb.sum()/hf.sum(),
                                     "retention_heldout_meanyield": hmean.sum()/hf.sum(),
                                     "retention_cv": pred["greedy"][:, k-1].sum()/full.sum(),
                                     "heldout_minus_cv": hb.sum()/hf.sum()-pred["greedy"][:, k-1].sum()/full.sum(),
                                     "mean_best_full_heldout": hf.mean(),
                                     **pair_ratio_ci(hb, hf, (screen, which, k))})
                hp_rows.extend({"screen": screen, "set": which, "subpair": hp.subpair.iloc[i], "k": k,
                                "full": hf[i], "greedy": hb[i], "meanyield": hmean[i]} for i in range(len(hy)))
        ds, av, parts = disjoint_experiments(screen, label, Y, pairs, args.repeats)
        disjoint_rows.extend(ds)
        availability_rows.extend(av)
        partition_rows.extend(parts)
        size_rows.extend(sensitivity_experiments(screen, label, Y, args.repeats))
        print(f"{screen}: {len(Y)} analysis pairs, curve to 24, {len(av)} attempted disjoint splits", flush=True)

    curves = pd.DataFrame(curve_rows)
    cvp = pd.DataFrame(cvpair_rows)
    h = pd.DataFrame(holdout_rows)
    nv = pd.DataFrame(novelty_rows)
    ds = pd.DataFrame(disjoint_rows)
    av = pd.DataFrame(availability_rows)
    sz = pd.DataFrame(size_rows)
    save(curves, "review_curve24.csv")
    save(cvp, "review_curve24_pair_scores.csv")
    save(h, "review_generalization_heldout_cv_per_array.csv")
    save(pd.DataFrame(hp_rows), "review_generalization_heldout_pair_scores.csv")
    save(nv, "review_generalization_novelty_pairs.csv")
    save(pd.DataFrame(mol_rows), "review_generalization_molecule_novelty_per_array.csv")
    save(ds, "review_generalization_disjoint_repeats.csv")
    save(av, "review_generalization_disjoint_availability.csv")
    pd.DataFrame(partition_rows).to_csv(HERE / "out" / "review_generalization_disjoint_assignments.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    save(sz, "review_generalization_training_size_repeats.csv")

    ns = nv.groupby(["screen", "screen_label", "set", "scope", "novelty"]).size().unstack(fill_value=0).reset_index()
    ns["n_pairs"] = ns[["both_seen", "one_unseen", "both_unseen"]].sum(axis=1)
    ns["at_least_one_seen_fraction"] = (ns.both_seen+ns.one_unseen)/ns.n_pairs
    save(ns, "review_generalization_novelty_per_array.csv")
    # Molecule counts are deduplicated, not sums of the two endpoint counts.
    cvnov = nv[nv.set == "pair_cv"]
    cvmol = pd.concat([
        cvnov[["screen", "screen_label", "scope", "fold", "sub_1_smiles", "partner1_seen_any_role"]].rename(
            columns={"sub_1_smiles": "molecule", "partner1_seen_any_role": "seen"}),
        cvnov[["screen", "screen_label", "scope", "fold", "sub_2_smiles", "partner2_seen_any_role"]].rename(
            columns={"sub_2_smiles": "molecule", "partner2_seen_any_role": "seen"})], ignore_index=True).drop_duplicates()
    cvmol_fold = cvmol.groupby(["screen", "screen_label", "scope", "fold"]).agg(
        n_test_molecules=("molecule", "nunique"), n_seen_molecules=("seen", "sum"),
        n_unseen_molecules=("seen", lambda x: int((~x).sum()))).reset_index()
    save(cvmol_fold, "review_generalization_cv_molecule_novelty_per_fold.csv")
    unique_cv = cvmol.groupby(["screen", "screen_label", "scope", "molecule"], as_index=False).agg(seen_in_every_test_fold=("seen", "all"))
    cvmol_array = unique_cv.groupby(["screen", "screen_label", "scope"]).agg(
        n_unique_test_molecules=("molecule", "nunique"),
        n_seen_in_every_test_fold=("seen_in_every_test_fold", "sum"),
        n_unseen_in_at_least_one_test_fold=("seen_in_every_test_fold", lambda x: int((~x).sum()))).reset_index()
    save(cvmol_array, "review_generalization_cv_molecule_novelty_per_array.csv")
    global_cv = unique_cv[unique_cv.scope == "global_corpus"].groupby("molecule", as_index=False).agg(
        seen_in_every_test_fold=("seen_in_every_test_fold", "all"))
    save(pd.DataFrame([{"scope": "global_corpus", "n_unique_analysis_molecules": len(global_cv),
                        "n_seen_in_every_test_fold": int(global_cv.seen_in_every_test_fold.sum()),
                        "n_unseen_in_at_least_one_test_fold": int((~global_cv.seen_in_every_test_fold).sum())}]),
         "review_generalization_cv_molecule_novelty_corpus.csv")
    subgroup_rows = []
    hpair = pd.DataFrame(hp_rows)
    tagged = hpair.merge(nv[nv.set != "pair_cv"][["screen", "set", "scope", "subpair", "novelty"]], on=["screen", "set", "subpair"])
    for keys, group in tagged.groupby(["screen", "set", "scope", "novelty", "k"]):
        screen, which, scope, category, k = keys
        subgroup_rows.append({"screen": screen, "set": which, "scope": scope, "novelty": category, "k": k,
                              "n_test_pairs": len(group), "n_nonzero_full": int((group.full > 0).sum()),
                              "retention_greedy": group.greedy.sum()/group.full.sum() if group.full.sum() > 0 else np.nan,
                              "sum_best_greedy": group.greedy.sum(), "sum_best_full": group.full.sum()})
    save(pd.DataFrame(subgroup_rows), "review_generalization_heldout_novelty_retention.csv")
    corpus_rows = []
    for which in ("analysis", "sealed", "reserve", "heldout"):
        p = data[data.set.isin(("sealed", "reserve") if which == "heldout" else (which,))][["screen", "subpair", "sub_1_smiles", "sub_2_smiles"]].drop_duplicates()
        pm = mols(p)
        corpus_rows.append({"set": which, "n_array_pairs": len(p), "n_unique_ordered_pairs": p.subpair.nunique(),
                            "n_molecules": len(pm), "n_new_vs_analysis": len(pm-all_analysis_mols),
                            "n_seen_vs_analysis": len(pm & all_analysis_mols)})
    save(pd.DataFrame(corpus_rows), "review_generalization_corpus_molecules.csv")

    summary_rows = []
    for which in ("sealed", "reserve", "all_heldout"):
        hh = h if which == "all_heldout" else h[h.set == which]
        for k, hk in hh.groupby("k"):
            for metric in ("retention_heldout", "retention_cv", "heldout_minus_cv"):
                for statistic in ("median", "mean"):
                    summary_rows.append({"comparison": which, "k": k, "metric": metric,
                                         **summarize_values(hk[metric], (which, k, metric, statistic), statistic)})
    save(pd.DataFrame(summary_rows), "review_generalization_heldout_cv_summary.csv")

    contrast_rows, disjoint_summary = [], []
    for min_test in (3, 8):
        for (screen, label, scheme, k), group in ds[ds.n_test >= min_test].groupby(["screen", "label", "scheme", "k"]):
            piv = group.pivot(index="repeat", columns="control", values="retention_greedy")
            for control in ("same_test_pair_control", "random_pair_control"):
                pair = piv[["disjoint", control]].dropna()
                diff = pair.disjoint-pair[control]
                counts = group[group.control == "disjoint"].set_index("repeat").loc[pair.index]
                contrast_rows.append({"screen": screen, "label": label, "scheme": scheme, "k": k,
                                      "minimum_test_pairs": min_test, "control": control,
                                      "n_repeats": len(pair), "retention_disjoint": pair.disjoint.mean(),
                                      "retention_control": pair[control].mean(), "disjoint_minus_control": diff.mean(),
                                      "delta_split_q025": diff.quantile(.025), "delta_split_q975": diff.quantile(.975),
                                      "mean_n_train": counts.n_train.mean(), "mean_n_test": counts.n_test.mean(),
                                      "min_n_test": counts.n_test.min(), "max_n_test": counts.n_test.max(),
                                      "mean_n_discarded_cross_edges": counts.n_discarded_cross_edges.mean()})
    contrasts = pd.DataFrame(contrast_rows)
    save(contrasts, "review_generalization_disjoint_per_array.csv")
    for keys, group in contrasts.groupby(["scheme", "k", "minimum_test_pairs", "control"]):
        scheme, k, mintest, control = keys
        for metric in ("retention_disjoint", "retention_control", "disjoint_minus_control"):
            for statistic in ("median", "mean"):
                disjoint_summary.append({"scheme": scheme, "k": k, "minimum_test_pairs": mintest, "control": control,
                                         "metric": metric, **summarize_values(group[metric], (keys, metric, statistic), statistic)})
    save(pd.DataFrame(disjoint_summary), "review_generalization_disjoint_summary.csv")
    avsum = av.groupby(["screen", "label", "scheme"]).agg(attempted=("availability", "size"),
                valid=("availability", lambda x: (x == "ok").sum()),
                mean_n_train=("n_train", "mean"), mean_n_test=("n_test", "mean"), min_n_test=("n_test", "min"),
                max_n_test=("n_test", "max"), mean_n_discarded=("n_discarded_cross_edges", "mean")).reset_index()
    save(avsum, "review_generalization_disjoint_availability_summary.csv")

    sizearray = sz.groupby(["screen", "label", "train_fraction_requested", "k"], as_index=False).agg(
        retention_greedy=("retention_greedy", "mean"), retention_meanyield=("retention_meanyield", "mean"),
        n_train=("n_train", "first"), n_test=("n_test", "first"), repeats=("repeat", "size"))
    save(sizearray, "review_generalization_training_size_per_array.csv")
    ss = []
    for (frac, k), group in sizearray.groupby(["train_fraction_requested", "k"]):
        ss.append({"train_fraction_requested": frac, "k": k,
                   **summarize_values(group.retention_greedy, ("training_size", frac, k))})
    save(pd.DataFrame(ss), "review_generalization_training_size_summary.csv")

    curve_summary = []
    for k, group in curves.groupby("k"):
        for method in ("greedy", "random", "meanyield"):
            curve_summary.append({"k": k, "method": method, "arrays_ge_90": int((group["retention_"+method] >= .9).sum()),
                                  **summarize_values(group["retention_"+method], ("curve", k, method))})
    save(pd.DataFrame(curve_summary), "review_curve24_summary.csv")
    k90 = []
    for (screen, label), group in curves.groupby(["screen", "label"]):
        for method in ("greedy", "random", "meanyield"):
            reached = group[group["retention_"+method] >= .9]
            k90.append({"screen": screen, "label": label, "method": method,
                        "k90": int(reached.k.min()) if len(reached) else np.nan,
                        "right_censored_above_24": not bool(len(reached)),
                        "retention_k24": group.loc[group.k == 24, "retention_"+method].iloc[0]})
    save(pd.DataFrame(k90), "review_curve24_k90.csv")

    verification = {"source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "seed": SEED,
                    "repeats": args.repeats, "numeric_missing_yields": bad, "numpy": np.__version__, "pandas": pd.__version__,
                    "rdkit": rdBase.rdkitVersion, "deposited_molecule_strings": len(identity),
                    "canonical_isomeric_molecules": int(identity.canonical_isomeric_smiles.nunique()),
                    "n_analysis_pairs": len(apairs), "n_heldout_pairs": int(len(nv[(nv.scope == "within_array") & (nv.set != "pair_cv")]))}
    oldpath = HERE / "out" / "kit_curve.csv"
    if oldpath.exists():
        old = pd.read_csv(oldpath)
        check = curves.merge(old, on=["screen", "k"], suffixes=("_new", "_old"))
        for method in ("greedy", "random", "meanyield"):
            delta = float(np.max(np.abs(check["retention_"+method+"_new"]-check["retention_"+method+"_old"])))
            verification["original_curve_max_abs_difference_"+method] = delta
            assert delta < 1e-12, (method, delta)
    verification["runtime_seconds"] = time.perf_counter()-start
    (HERE / "out" / "review_generalization_verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    (HERE / "out" / "review_generalization_methods.md").write_text(METHODS, encoding="utf-8")
    novelty_summary = ns.groupby(["set", "scope"])[["n_pairs", "both_seen", "one_unseen", "both_unseen"]].sum()
    hdsummary = pd.DataFrame(summary_rows)
    dsummary = pd.DataFrame(disjoint_summary)
    csummary = pd.DataFrame(curve_summary)
    report = "# Checked generalization findings\n\n"
    report += "All estimates below are post-review analyses of frozen data. See `review_generalization_methods.md` for definitions and limitations.\n\n"
    report += "## Pair novelty counts\n\n```text\n"+novelty_summary.to_string()+"\n```\n\n"
    report += "Corpus-level molecule counts (not sums over arrays):\n\n```text\n"+pd.DataFrame(corpus_rows).to_string(index=False)+"\n```\n\n"
    report += "## Matched-array held-out versus pair-CV comparison at k=6\n\nThe contrast is the median of paired array differences; it is not a difference of separate medians. CIs resample arrays.\n\n```text\n"
    report += hdsummary[(hdsummary.k == 6) & (hdsummary.statistic == "median")].to_string(index=False)+"\n```\n\n"
    report += "## Molecule-disjoint tests at k=6\n\nPer-array estimates average available repeated splits; summary medians and CIs operate over arrays. Conventional random-pair controls are matched in train/test size. The same-test control is useful for both-unseen, but degenerates for one-role exclusion.\n\n```text\n"
    report += dsummary[(dsummary.k == 6) & (dsummary.minimum_test_pairs == 3) & (dsummary.statistic == "median")].to_string(index=False)+"\n```\n\n"
    report += "Attempted split availability:\n\n```text\n"+av.groupby(["scheme", "availability"]).size().to_string()+"\n```\n\n"
    report += "## Curves through k=24\n\n```text\n"+csummary[csummary.k.isin([1, 3, 6, 12, 18, 24])].to_string(index=False)+"\n```\n\n"
    report += "## Verification\n\n```json\n"+json.dumps(verification, indent=2)+"\n```\n"
    (HERE / "out" / "review_generalization_findings.md").write_text(report, encoding="utf-8")
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    main()
