"""Round-three conditional tail audit, full fixed-budget curves, and exploratory policies.

Run python review_tail_gc.py [--self-test]. Writes out/review_tail_* and
04_HISTORY/review_round3_2026-09-20/TAIL_ANALYSIS.md only. Established inputs are read-only.
"""
from pathlib import Path
from math import comb, floor, ceil
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
import lib

HERE = Path(__file__).resolve().parent


def _hist(*parts):
    """Locate an audit artefact at the tree root or, once archived, under 04_HISTORY."""
    here = HERE.joinpath(*parts)
    hist = HERE.joinpath("04_HISTORY", *parts)
    return hist if hist.exists() else here
OUT = HERE / "out"
SEED = 202609203
FOLD_SEED = 3
NFOLD = 10
B = 10000
MAXK = 24
COHORTS = ("all", "full_gt0", "full_ge20pct", "full_gt20pct")


def seed_for(label):
    return (SEED + int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], "little")) % 2**32


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, rows):
    d = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    d.to_csv(OUT / ("review_tail_" + name + ".csv"), index=False)
    return d


def divide(x, y):
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    return np.divide(x, y, out=np.full_like(x, np.nan), where=y != 0)


def cohort(full, name):
    if name == "all":
        return np.ones(len(full), bool)
    if name == "full_gt0":
        return full > 0
    if name == "full_ge20pct":
        return full >= .20
    if name == "full_gt20pct":
        return full > .20
    raise ValueError(name)


def exact_mcnemar(improved, worsened):
    """Two-sided exact binomial McNemar reference, conditional on discordants."""
    n = int(improved + worsened)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(improved, worsened) + 1)) / 2**n)


def node_pair_weights(weights, endpoint1, endpoint2):
    """Product over distinct, role-merged molecular endpoints."""
    pair = weights[:, endpoint1] * weights[:, endpoint2]
    same = endpoint1 == endpoint2
    if same.any():
        pair[:, same] = weights[:, endpoint1[same]]
    return pair


def self_test():
    assert exact_mcnemar(0, 5) == .0625
    assert exact_mcnemar(1, 1) == 1 and exact_mcnemar(0, 0) == 1
    w = np.array([[2., 3., 5.]])
    p = node_pair_weights(w, np.array([0, 2, 0, 0]), np.array([1, 0, 1, 0]))
    assert np.array_equal(p, [[6., 10., 6., 2.]])
    # A random mix of panels must average event indicators, not threshold mean yield.
    full, low, high, q = .8, .1, .8, .75
    risk = (1-q) * (100*(full-low) > 20) + q * (100*(full-high) > 20)
    assert risk == .25 and 100*(full-((1-q)*low+q*high)) < 20
    f = np.array([0, .1, .2, .21])
    assert cohort(f, "full_gt0").sum() == 3
    assert cohort(f, "full_ge20pct").sum() == 2 and cohort(f, "full_gt20pct").sum() == 1
    assert np.isnan(divide(0, 0)) and divide(2, 4) == .5
    print("Independent threshold, discordance, node-weight and mixture checks passed", flush=True)


def rebuild_curves():
    a, _ = lib.load()
    frames, gs, ts = [], [], []
    for screen in sorted(a.screen.unique()):
        d = a[a.screen == screen]
        m = lib.screen_matrix(a, screen)
        y = m.values
        n, nc = y.shape
        assert nc >= MAXK
        g = np.empty((n, MAXK)); t = np.empty_like(g)
        fold_id = np.empty(n, int)
        folds = np.array_split(np.random.RandomState(FOLD_SEED).permutation(n), NFOLD)
        for fi, test in enumerate(folds):
            train = np.setdiff1d(np.arange(n), test)
            assert not len(np.intersect1d(train, test))
            fold_id[test] = fi
            kit = lib.greedy_kit(y[train], MAXK)
            rank = np.argsort(-np.nanmean(y[train], axis=0))
            for k in range(1, MAXK+1):
                g[test, k-1] = lib.retention(y[test], kit[:k])[0]
                t[test, k-1] = lib.retention(y[test], list(rank[:k]))[0]
        p = d.drop_duplicates("subpair").set_index("subpair").loc[m.index].reset_index()
        meta = p[["screen", "subpair", "sub_1_smiles", "sub_2_smiles", "role_1", "role_2"]].copy()
        meta["label"] = d.screen_label.iloc[0]
        meta["fold"] = fold_id
        meta["full"] = lib.retention(y, list(range(nc)))[1]
        frames.append(meta.reset_index(drop=True)); gs.append(g); ts.append(t)
        print(screen, "all 24 budgets rebuilt", flush=True)
    pairs = pd.concat(frames, ignore_index=True)
    greedy, topmean = np.vstack(gs), np.vstack(ts)
    assert len(pairs) == 2357 and not pairs.duplicated(["screen", "subpair"]).any()
    assert np.all(np.diff(greedy, axis=1) >= -1e-12) and np.all(np.diff(topmean, axis=1) >= -1e-12)
    assert np.all(greedy <= pairs.full.values[:, None] + 1e-12)
    assert np.all(topmean <= pairs.full.values[:, None] + 1e-12)
    return pairs, greedy, topmean


def identity_audit(pairs):
    raw = sorted(set(pairs.sub_1_smiles) | set(pairs.sub_2_smiles))
    rows = []
    mapping = {}
    for smiles in raw:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("A substrate SMILES could not be parsed: " + smiles)
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        mapping[smiles] = canonical
        role1 = pairs.sub_1_smiles == smiles
        role2 = pairs.sub_2_smiles == smiles
        rows.append({"raw_smiles": smiles, "canonical_isomeric_smiles": canonical,
                     "pairs_role1": int(role1.sum()), "pairs_role2": int(role2.sum()),
                     "pairs_any_role": int((role1 | role2).sum()),
                     "arrays_any_role": int(pairs.loc[role1 | role2, "screen"].nunique()),
                     "appears_in_both_roles": bool(role1.any() and role2.any())})
    save("molecular_identity", rows)
    p1 = pairs.sub_1_smiles.map(mapping).to_numpy()
    p2 = pairs.sub_2_smiles.map(mapping).to_numpy()
    nodes = sorted(set(p1) | set(p2))
    index = {s: i for i, s in enumerate(nodes)}
    pairs["canonical_partner1"] = p1
    pairs["canonical_partner2"] = p2
    pairs["node1"] = [index[s] for s in p1]
    pairs["node2"] = [index[s] for s in p2]
    return {"raw_molecules": len(raw), "canonical_molecules": len(nodes),
            "cross_role_molecules": len(set(p1) & set(p2)),
            "same_molecule_pairs": int((p1 == p2).sum()),
            "role1_molecules": len(set(p1)), "role2_molecules": len(set(p2))}


def validate_prior(pairs, g, t):
    loss = pd.read_csv(OUT / "review_loss_pair_scores.csv")
    p = pairs.assign(greedy=g[:, 5], topmean=t[:, 5]).merge(loss, on=["screen", "subpair"], suffixes=("", "_old"), validate="one_to_one")
    assert len(p) == len(pairs)
    for new, old in [("full", "full_best_yield"), ("greedy", "greedy_best_yield"), ("topmean", "topmean_best_yield")]:
        assert np.allclose(p[new], p[old], rtol=0, atol=1e-12)
    previous = pd.read_csv(OUT / "review_curve24_pair_scores.csv")
    checked = 0
    for k, old in previous.groupby("k"):
        z = pairs.assign(greedy=g[:, k-1], meanyield=t[:, k-1]).merge(old, on=["screen", "subpair"], suffixes=("", "_old"), validate="one_to_one")
        assert len(z) == len(pairs)
        for name in ("full", "greedy", "meanyield"):
            assert np.allclose(z[name], z[name + "_old"], rtol=0, atol=1e-12)
        assert np.array_equal(z.fold, z.fold_old)
        checked += len(z)
    return {"prior_k6_pairs_checked": len(p), "prior_sparse_curve_pair_budgets_checked": checked,
            "prior_saved_budgets": sorted(previous.k.unique().tolist())}


def discordance(pairs, g, t):
    rows = []
    for name in COHORTS:
        for screen in ["ALL"] + sorted(pairs.screen.unique().tolist()):
            mask = cohort(pairs.full.values, name)
            if screen != "ALL":
                mask &= pairs.screen.values == screen
            ge = 100 * (pairs.full.values[mask] - g[mask, 5]) > 20
            te = 100 * (pairs.full.values[mask] - t[mask, 5]) > 20
            better = int((~ge & te).sum()); worse = int((ge & ~te).sum())
            rows.append({"screen": screen, "cohort": name, "pairs": int(mask.sum()),
                         "greedy_severe": int(ge.sum()), "topmean_severe": int(te.sum()),
                         "both_severe": int((ge & te).sum()), "neither_severe": int((~ge & ~te).sum()),
                         "greedy_improves": better, "greedy_worsens": worse,
                         "mcnemar_exact_iid_reference_p": exact_mcnemar(better, worse),
                         "p_is_valid_for_shared_molecules": False})
    result = save("discordance", rows)
    z = result[(result.screen == "ALL") & (result.cohort == "full_gt0")].iloc[0]
    assert z.pairs == 1642 and z.greedy_severe == 255 and z.topmean_severe == 294
    assert z.greedy_improves == 102 and z.greedy_worsens == 63
    return result


def molecular_bootstrap(pairs, g, t):
    """Reweight fixed k6 outcome indicators globally; no model is refitted."""
    full = pairs.full.values
    ge = (100*(full-g[:, 5]) > 20).astype(float)
    te = (100*(full-t[:, 5]) > 20).astype(float)
    groups = []
    masks = []
    for name in COHORTS:
        for screen in ["ALL"] + sorted(pairs.screen.unique().tolist()):
            keep = cohort(full, name)
            if screen != "ALL":
                keep &= pairs.screen.values == screen
            groups.append((name, screen)); masks.append(keep)
    cm = np.column_stack(masks).astype(float)
    design = np.column_stack([cm, cm * ge[:, None], cm * te[:, None]])
    ncols = len(groups)
    endpoint1 = pairs.node1.to_numpy(); endpoint2 = pairs.node2.to_numpy()
    _, role1 = np.unique(pairs.canonical_partner1, return_inverse=True)
    _, role2 = np.unique(pairs.canonical_partner2, return_inverse=True)
    nnode = max(endpoint1.max(), endpoint2.max()) + 1
    summaries, exports = [], {}
    for scheme in ("role_merged_node_exponential", "role_separated_pigeonhole_exponential"):
        rng = np.random.default_rng(seed_for(scheme))
        rates = np.empty((B, ncols, 2))
        for start in range(0, B, 250):
            count = min(250, B-start)
            if scheme == "role_merged_node_exponential":
                weights = node_pair_weights(rng.exponential(size=(count, nnode)), endpoint1, endpoint2)
            else:
                w1 = rng.exponential(size=(count, role1.max()+1))
                w2 = rng.exponential(size=(count, role2.max()+1))
                weights = w1[:, role1] * w2[:, role2]
            products = weights @ design
            denominator, ng, nt = np.split(products, 3, axis=1)
            rates[start:start+count, :, 0] = divide(ng, denominator)
            rates[start:start+count, :, 1] = divide(nt, denominator)
        exports[scheme] = rates
        point_g = divide(cm.T @ ge, cm.sum(axis=0))
        point_t = divide(cm.T @ te, cm.sum(axis=0))
        for name in COHORTS:
            selections = [(s, [i]) for i, (c, s) in enumerate(groups) if c == name]
            selections.append(("MACRO", [i for i, (c, s) in enumerate(groups) if c == name and s != "ALL"]))
            for screen, ix in selections:
                brg = rates[:, ix, 0].mean(axis=1)
                brt = rates[:, ix, 1].mean(axis=1)
                pg, pt = float(point_g[ix].mean()), float(point_t[ix].mean())
                bm = np.column_stack([brg, brt, brg-brt, brt-brg, divide(brt-brg, brt)])
                point = [pg, pt, pg-pt, pt-pg, float(divide(pt-pg, pt))]
                mask = cohort(full, name)
                if screen not in ("ALL", "MACRO"):
                    mask &= pairs.screen.values == screen
                for j, metric in enumerate(("greedy_tail_rate", "topmean_tail_rate", "greedy_minus_topmean_rate", "benefit_topmean_minus_greedy_rate", "relative_tail_reduction")):
                    valid = np.isfinite(bm[:, j])
                    lo, hi = np.quantile(bm[valid, j], [.025, .975]) if valid.any() else [np.nan, np.nan]
                    summaries.append({"scheme": scheme, "scope": "macro_equal_array_mean" if screen == "MACRO" else "pooled_pairs" if screen == "ALL" else "array",
                                      "screen": screen, "cohort": name, "k": 6, "metric": metric,
                                      "estimate": point[j], "interval_lo": lo, "interval_hi": hi,
                                      "draws": B, "valid_draws": int(valid.sum()), "pairs": int(mask.sum()),
                                      "arrays": int(pairs.loc[mask, "screen"].nunique()),
                                      "molecules_any_role": len(set(pairs.loc[mask, "node1"]) | set(pairs.loc[mask, "node2"])),
                                      "conditional_on_fixed_oof_predictions": True,
                                      "zero_event_boundary": bool(pg == 0 or pt == 0)})
        print(scheme, "10000 draws completed", flush=True)
    np.savez_compressed(OUT / "review_tail_bootstrap_draws.npz", **exports,
                        group_cohort=np.array([x[0] for x in groups]), group_screen=np.array([x[1] for x in groups]))
    return save("bootstrap_summary", summaries)


def observed_metrics(full, best):
    loss = 100*(full-best)
    return {"pairs": len(full), "severe_pairs": int((loss > 20).sum()),
            "tail_rate": float((loss > 20).mean()), "loss_gt10_rate": float((loss > 10).mean()),
            "retention": float(divide(best.sum(), full.sum())),
            "mean_best_yield": float(best.mean()), "mean_full_yield": float(full.mean()),
            "median_loss_pp": float(np.median(loss)), "p90_loss_pp": float(np.quantile(loss, .9)),
            "panel_ge20pct_pairs": int((best >= .20).sum()), "full_ge20pct_pairs": int((full >= .20).sum()),
            "hit_recovery_ge20pct": float(divide((best >= .20).sum(), (full >= .20).sum()))}


def fixed_curves(pairs, g, t):
    array_rows, pooled_rows = [], []
    full = pairs.full.values
    for k in range(1, MAXK+1):
        for method, predictions in (("greedy", g), ("topmean", t)):
            for name in COHORTS:
                base = cohort(full, name)
                pooled_rows.append({"k": k, "method": method, "cohort": name, "arrays": pairs.screen.nunique(),
                                    **observed_metrics(full[base], predictions[base, k-1])})
                for screen in sorted(pairs.screen.unique()):
                    mask = base & (pairs.screen.values == screen)
                    array_rows.append({"screen": screen, "label": pairs.loc[mask, "label"].iloc[0],
                                       "k": k, "method": method, "cohort": name,
                                       **observed_metrics(full[mask], predictions[mask, k-1])})
    arrays = save("curve_array", array_rows)
    pooled = save("curve_pooled", pooled_rows)
    macro = []
    for (k, method, name), d in arrays.groupby(["k", "method", "cohort"]):
        macro.append({"k": k, "method": method, "cohort": name, "arrays": len(d), "pairs_total": int(d.pairs.sum()),
                      "tail_rate_macro_mean": d.tail_rate.mean(), "tail_rate_macro_median": d.tail_rate.median(),
                      "retention_macro_mean": d.retention.mean(), "retention_macro_median": d.retention.median(),
                      "hit_recovery_macro_mean": d.hit_recovery_ge20pct.mean(),
                      "median_loss_pp_macro_mean": d.median_loss_pp.mean(),
                      "p90_loss_pp_macro_mean": d.p90_loss_pp.mean()})
    macro = save("curve_macro", macro)
    for _, d in pooled.groupby(["method", "cohort"]):
        assert np.all(np.diff(d.sort_values("k").tail_rate) <= 1e-12)
    return arrays, pooled, macro


def exploratory_policies(pairs, g):
    """Expected event probabilities, never indicators of an averaged yield."""
    rows = []
    full = pairs.full.values
    for screen in ["ALL"] + sorted(pairs.screen.unique().tolist()):
        unit = np.ones(len(pairs), bool) if screen == "ALL" else pairs.screen.values == screen
        fu, gu = full[unit], g[unit]
        for threshold in (.5, .7):
            expand = gu[:, 5] < threshold
            wells = 6 + 6*expand.astype(int)
            policy_best = np.where(expand, gu[:, 11], gu[:, 5])
            target = float(wells.mean())
            lower, upper = floor(target), ceil(target)
            q = target - lower
            mixed_best = (1-q)*gu[:, lower-1] + q*gu[:, upper-1]
            mixed_tail = (1-q)*(100*(fu-gu[:, lower-1]) > 20) + q*(100*(fu-gu[:, upper-1]) > 20)
            mixed_hits = (1-q)*(gu[:, lower-1] >= .20) + q*(gu[:, upper-1] >= .20)
            strategies = [("trigger_6_to_12", policy_best, (100*(fu-policy_best) > 20).astype(float), (policy_best >= .20).astype(float), wells),
                          ("randomized_floor_ceil_same_mean_budget", mixed_best, mixed_tail, mixed_hits, np.full(len(fu), target))]
            for k in (6, 9, 12):
                best = gu[:, k-1]
                strategies.append(("fixed_" + str(k), best, (100*(fu-best) > 20).astype(float), (best >= .20).astype(float), np.full(len(fu), k)))
            for strategy, best, tail, hits, budget in strategies:
                for name in COHORTS:
                    mask = cohort(fu, name)
                    rows.append({"scope": "pooled_pairs" if screen == "ALL" else "array", "screen": screen,
                                 "cohort": name, "threshold_best6_lt": threshold, "strategy": strategy,
                                 "budget_population_pairs": len(fu), "performance_cohort_pairs": int(mask.sum()),
                                 "mean_wells_all_pairs": float(budget.mean()), "mean_wells_in_cohort": float(budget[mask].mean()),
                                 "trigger_expand_fraction_all_pairs": float(expand.mean()),
                                 "matched_budget_floor": lower, "matched_budget_ceil": upper, "matched_probability_ceil": q,
                                 "retention": float(divide(best[mask].sum(), fu[mask].sum())),
                                 "expected_tail_cases": float(tail[mask].sum()), "tail_rate": float(tail[mask].mean()),
                                 "expected_hits_ge20pct": float(hits[mask].sum()),
                                 "hit_recovery_ge20pct": float(divide(hits[mask].sum(), (fu[mask] >= .20).sum())),
                                 "posthoc_exploration_not_validated_policy": True})
    result = save("exploratory_policies", rows)
    for _, d in result.groupby(["screen", "cohort", "threshold_best6_lt"]):
        target = d[d.strategy == "trigger_6_to_12"].mean_wells_all_pairs.iloc[0]
        actual = d[d.strategy == "randomized_floor_ceil_same_mean_budget"].mean_wells_all_pairs.iloc[0]
        assert np.isclose(target, actual, rtol=0, atol=1e-12)
    # Each macro comparator first matches mean budget within each array. This preserves
    # both each array's resource allocation and the pooled total, unlike a macro-only match.
    macro = result[result.scope == "array"].groupby(["cohort", "threshold_best6_lt", "strategy"]).agg(
        arrays=("screen", "size"), mean_wells_macro=("mean_wells_all_pairs", "mean"),
        tail_rate_macro_mean=("tail_rate", "mean"), retention_macro_mean=("retention", "mean"),
        hit_recovery_macro_mean=("hit_recovery_ge20pct", "mean")).reset_index()
    save("exploratory_policies_macro", macro)
    return result


METHODS = """# Round-three tail analysis: definitions and limits

Run `python review_tail_gc.py` with NumPy, pandas and RDKit. Seed 202609203 governs
bootstrap streams; all fits use the established ten substrate-pair folds with seed 3.
All original data and outputs are read-only and hashed before/after execution.

## Fixed predictions and estimands

The existing review_curve24_pair_scores.csv stores only k=1,3,6,12,24. This script
independently rebuilds greedy and topmean predictions at every k=1..24, on identical
training/test folds, and verifies every previously saved prediction to 1e-12.
Greedy prefixes and topmean rankings are nested within each fold. Topmean uses
means over measured training wells only, exactly as the established code.
Missing selected wells contribute no observed yield; the full reference is the
maximum observed array yield. Tail loss is an absolute yield deficit strictly
greater than 20 percentage points: 100*(full - panel) >20. This is not relative loss.
The full maximum is itself measured and subject to noise/selection effects.

Separate denominator cohorts are all pairs, full>0 (productive), full>=0.20
(20%-reachable, including exactly 20%), and full>0.20 (at risk for a >20-point loss).
The latter two contain 1541 and 1539 pairs; they must not be interchanged. Pooled
rates weight pairs equally, whereas macro means weight each of the 18 arrays
equally. Macro median rates are separately named. Retention is sum(panel maximum)
divided by sum(full maximum), never the mean of individual retention ratios.
No threshold is tuned. Curves describe fixed predictions, not a new validation set.

## Paired tail contrast and molecular reweighting

Both methods are evaluated on the same pairs. Greedy-improved means topmean loss
>20 but greedy loss<=20; worsened means the reverse. The exact two-sided McNemar
reference is twice the smaller Binomial(discordants,0.5) tail, capped at one.
Because pairs share molecules, this iid reference p is descriptive only and is
explicitly labelled not valid as a shared-molecule significance test.

Primary uncertainty: a role-merged molecular-node exponential multiplier scheme.
Canonical isomeric RDKit SMILES preserve fragments, salts and stereochemistry;
no tautomer/scaffold merging is performed. The same molecular node is shared
globally across arrays and across both roles. In each of 10,000 draws, every
unique node m receives an independent Exp(1) weight Z_m (mean and variance one).
A pair's weight is Z_u*Z_v for different endpoints, or Z_u for a repeated endpoint.
Thus pairs sharing either molecule in either role share the same random factor;
repeated unordered pairs across arrays receive identical weights. No pair is
independently resampled. Each cohort rate is the weighted count of severe cases
divided by its weighted pair count. The paired difference and relative reduction
are computed within the same draw; relative reduction=(topmean-greedy)/topmean.
Both signed rate contrasts are exported: greedy_minus_topmean_rate and its negative,
benefit_topmean_minus_greedy_rate. A positive benefit favors greedy.
Macro draws average the per-array weighted rates equally before contrasting.

Sensitivity: role-separated exponential pigeonhole reweighting uses independent
weights for role-1 identities and role-2 identities. It handles sharing within
both roles but treats the same chemical identity in opposite roles as independent;
therefore it is secondary to the role-merged analysis. The product-weight idea is
related to Owen and Eckles, Bootstrapping data arrays of arbitrary order, Annals of
Applied Statistics 6, 895-927 (2012), DOI 10.1214/12-AOAS547:
https://projecteuclid.org/journals/annals-of-applied-statistics/volume-6/issue-3/Bootstrapping-data-arrays-of-arbitrary-order/10.1214/12-AOAS547.pdf
Their crossed-factor theory is not claimed to prove nominal coverage for our
role-merged node scheme, nonlinear ratios or this fixed chemistry corpus.

Reported limits are the empirical 2.5th/97.5th percentiles: **95% conditional
molecule-multiplier intervals**, not a demonstrated nominal-coverage confidence
procedure. They describe sensitivity to molecular composition conditional on all
already-fitted OOF panels, arrays and measured outcomes. Models are not refitted;
training-set/fit variability, campaign effects independent of molecules, and new
chemistry/domain generalization are not represented. Sparse-array limits can be
unstable. A group with no observed events has a degenerate zero event-rate interval,
which is not evidence of zero underlying risk; a zero topmean event rate makes
relative reduction undefined (valid_draws=0). No molecule-bootstrap p-value is
manufactured by counting signs. Resampling arrays would target an additional and
different source of variation, and is not silently substituted for molecule reuse.

## Exploratory six-to-twelve trigger

For the two requested thresholds T=0.5 and 0.7, run six greedy-prefix conditions;
if the best observed yield is <T, add the next six conditions from the same trained
greedy sequence. This costs 6 or 12 wells. Fixed k=6,9,12 are reported alongside.
For a fair mean-budget reference, let b be the trigger's average wells over ALL
pairs in the reporting unit (array or pooled corpus). Randomly allocate floor(b)
or ceil(b) greedy-prefix wells independently of outcomes, with probability
b-floor(b) of the larger budget. Expected yield, hit and severe-loss event rates
are mixed linearly. We never threshold the mixture's mean yield as though it were
an observed panel outcome. The expected budget matches exactly. Performance can
then be reported on each denominator cohort, while resource matching remains over
all pairs that would be screened; both all-pair and cohort budgets are visible.
Macro comparisons match within each array before averaging, preserving local
resource allocation. Pooled randomization matches the overall pooled budget.
These are retrospective, requested-threshold explorations. The rule was not
prospectively selected or externally validated; no validated-policy claim follows.

## Outputs

pair_scores: all pairs at k=1..24; molecular_identity: cross-role/global identities;
discordance: event counts plus invalid-for-dependence iid p reference;
bootstrap_summary and bootstrap_draws.npz: k6 conditional intervals and all draws;
curve_array, curve_pooled, curve_macro: distinct fixed-budget estimands;
exploratory_policies and exploratory_policies_macro: cost-matched trigger audit;
manifest: hashes, definitions, versions and checks. All CSV names use review_tail_.
The companion TAIL_ANALYSIS.md records the main numeric findings for integration.
"""


def report_text(identity, discord, boot, pooled, macro, policies):
    z = discord[(discord.screen == "ALL") & (discord.cohort == "full_gt0")].iloc[0]
    lines = ["# Round-three tail analysis", "", "## Paired complementarity at six wells", "",
             f"Among {int(z.pairs)} productive pairs: greedy severe losses {int(z.greedy_severe)}, topmean {int(z.topmean_severe)}. "
             f"Greedy improves {int(z.greedy_improves)} discordant pairs and worsens {int(z.greedy_worsens)}; both severe {int(z.both_severe)}, neither {int(z.neither_severe)}.",
             f"Exact iid McNemar reference p={z.mcnemar_exact_iid_reference_p:.8g}; shared molecular identities invalidate treating this as an independent-pair significance claim.",
             f"Identity audit: {identity['canonical_molecules']} globally unique canonical molecules; {identity['cross_role_molecules']} appear in both roles; no canonicalization collision is assumed without checking.", "",
             "| Molecular scheme | Cohort | Scope | Metric | Estimate | Conditional 95% interval |", "|---|---|---|---|---:|---:|"]
    b = boot[boot.cohort.isin(["full_gt0", "full_ge20pct"]) & boot.screen.isin(["ALL", "MACRO"])]
    for _, r in b.iterrows():
        lines.append(f"| {r.scheme} | {r.cohort} | {r.scope} | {r.metric} | {r.estimate:.6f} | {r.interval_lo:.6f} to {r.interval_hi:.6f} |")
    lines += ["", "Rates and differences are fractions; multiply by 100 for percentage points. Relative reduction is a fraction of topmean risk. Intervals condition on existing predictions, omit refitting uncertainty and have no demonstrated nominal-coverage guarantee.",
              "Both molecular schemes' paired benefit intervals include zero, for both productive and >=20%-reachable cohorts. The observed difference must not be described as a robust or statistically significant tail-risk improvement.",
              "", "## Fixed-budget greedy tails", "", "| Cohort | k | Pairs | Severe pairs | Pooled tail rate | Pooled retention | Macro mean tail rate |", "|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in pooled[(pooled.method == "greedy") & pooled.k.isin([6,9,12])].iterrows():
        ma = macro[(macro.method == "greedy") & (macro.k == r.k) & (macro.cohort == r.cohort)].iloc[0]
        lines.append(f"| {r.cohort} | {int(r.k)} | {int(r.pairs)} | {int(r.severe_pairs)} | {r.tail_rate:.6f} | {r.retention:.6f} | {ma.tail_rate_macro_mean:.6f} |")
    lines += ["", "Full>=20% includes 1541 pairs; full>20% includes 1539. All fixed-budget points 1–24 and both methods are in the CSVs. The 20-point loss is absolute yield, not relative yield loss.",
              "", "## Exploratory trigger, productive-pair outcomes", "", "| T | Strategy | Mean wells, all pairs | Retention | Tail rate |", "|---:|---|---:|---:|---:|"]
    for _, r in policies[(policies.screen == "ALL") & (policies.cohort == "full_gt0")].iterrows():
        lines.append(f"| {r.threshold_best6_lt:.1f} | {r.strategy} | {r.mean_wells_all_pairs:.6f} | {r.retention:.6f} | {r.tail_rate:.6f} |")
    lines += ["", "The threshold rule is exploratory and not externally validated. Same-average-budget controls randomize floor/ceil fixed greedy budgets independently of outcomes. Tail probabilities are mixed directly; they are not obtained by thresholding expected yield.",
              "", "See out/review_tail_methods.md and manifest.json for complete implementation, sampling assumptions, limits, identities and source checks.", ""]
    return "\n".join(lines)


def main():
    OUT.mkdir(exist_ok=True)
    paths = {"arrays.csv": lib.DATASET / "arrays.csv", "conditions.csv": lib.DATASET / "conditions.csv", "lib.py": HERE / "lib.py",
             "review_loss_pair_scores.csv": OUT / "review_loss_pair_scores.csv", "review_curve24_pair_scores.csv": OUT / "review_curve24_pair_scores.csv"}
    before = {name: sha(p) for name, p in paths.items()}
    pairs, g, t = rebuild_curves()
    validation = validate_prior(pairs, g, t)
    identity = identity_audit(pairs)
    assert identity["raw_molecules"] == identity["canonical_molecules"] == 819
    assert (pairs.full >= .20).sum() == 1541 and (pairs.full > .20).sum() == 1539
    long = []
    for k in range(1, MAXK+1):
        frame = pairs[["screen", "subpair", "fold", "full"]].copy()
        frame["k"] = k; frame["greedy"] = g[:, k-1]; frame["topmean"] = t[:, k-1]
        frame["greedy_loss_pp"] = 100*(frame.full-frame.greedy)
        frame["topmean_loss_pp"] = 100*(frame.full-frame.topmean)
        long.append(frame)
    save("pair_scores", pd.concat(long, ignore_index=True))
    save("pair_identity", pairs)
    dis = discordance(pairs, g, t)
    arrays, pooled, macro = fixed_curves(pairs, g, t)
    policies = exploratory_policies(pairs, g)
    boot = molecular_bootstrap(pairs, g, t)
    assert before == {name: sha(p) for name, p in paths.items()}, "An established source changed during execution"
    manifest = {"seed": SEED, "fold_seed": FOLD_SEED, "folds": NFOLD, "bootstrap_draws": B,
                "bootstrap_distribution": "independent Exponential(1) molecular weights", "budgets": list(range(1, MAXK+1)),
                "cohorts": list(COHORTS), "numpy": np.__version__, "pandas": pd.__version__, "rdkit": rdBase.rdkitVersion,
                "identity": identity, "checks": {**validation, "self_tests": "pass", "fixed_curves_monotone": True,
                "same_average_policy_budget_verified": True, "raw_inputs_unchanged": True, "reachable_ge20": 1541, "reachable_gt20": 1539},
                "input_sha256": before, "script_sha256": sha(Path(__file__))}
    (OUT / "review_tail_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (OUT / "review_tail_methods.md").write_text(METHODS, encoding="utf-8")
    report = _hist("review_round3_2026-09-20") / "TAIL_ANALYSIS.md"
    report.parent.mkdir(exist_ok=True)
    report.write_text(report_text(identity, dis, boot, pooled, macro, policies), encoding="utf-8")
    headline = boot[(boot.scheme == "role_merged_node_exponential") & (boot.screen == "ALL") & (boot.cohort == "full_gt0")]
    print(headline[["metric", "estimate", "interval_lo", "interval_hi"]].to_string(index=False), flush=True)
    print("Tail audit verification passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test()
    if not args.self_test:
        main()
