"""Independent checks of constrained panels, metadata-only recipes, and claim limits.

Run: python review_axes_gc.py [--self-test]
Writes only out/review_axes*; raw data and established analyses remain read-only.
"""
from itertools import combinations, product
from math import comb
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import lib

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
K = 6
NFOLD = 10
FOLD_SEED = 3
SEED = 20260920
NBOOT = 4000


def stable_seed(name):
    return (SEED + int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")) % (2**32)


def expected_random(y, k):
    """Exact mean best yield for uniform k-subsets of condition columns.

    Missing wells score zero, matching lib.retention for nonnegative yields.
    No yield is used in the probability assigned to a panel.
    """
    n = y.shape[1]
    if not 1 <= k <= n:
        raise ValueError("Infeasible condition budget")
    ys = np.sort(np.nan_to_num(y, nan=0.0), axis=1)
    weights = np.array([comb(i, k - 1) if i >= k - 1 else 0 for i in range(n)]) / comb(n, k)
    return ys @ weights


def expected_level_diverse(y, labels, k):
    """Exact expectation: choose k distinct levels uniformly, then one condition
    uniformly within each level. Level labels alone define selection probabilities.

    P(max < t) is the kth elementary symmetric polynomial of the within-level
    probabilities P(y < t), divided by the number of k-level subsets.
    """
    levels = np.unique(labels)
    if len(levels) < k or k < 1:
        raise ValueError("Too few levels for the requested budget")
    yf = np.nan_to_num(y, nan=0.0)
    thresholds = np.sort(yf, axis=1)
    e = np.zeros((k + 1,) + thresholds.shape)
    e[0] = 1.0
    for seen, level in enumerate(levels, 1):
        p = (yf[:, labels == level, None] < thresholds[:, None, :]).mean(axis=1)
        for j in range(min(k, seen), 0, -1):
            e[j] += p * e[j - 1]
    survival = np.clip(1.0 - e[k] / comb(len(levels), k), 0.0, 1.0)
    widths = np.diff(thresholds, axis=1, prepend=0.0)
    return (widths * survival).sum(axis=1)


def interval(x):
    return [float(v) for v in np.percentile(x, [2.5, 97.5])]


def effect(a, b, full, weights):
    """Paired a-minus-b contrast; resample substrate pairs, do not refit models."""
    delta = a - b
    boot_delta = weights @ delta
    boot_full = weights @ full
    rlo, rhi = interval(boot_delta / boot_full)
    plo, phi = interval(100 * boot_delta)
    return {"difference_retention": float(delta.mean() / full.mean()),
            "difference_retention_lo": rlo, "difference_retention_hi": rhi,
            "difference_yield_pp": float(100 * delta.mean()),
            "difference_yield_pp_lo": plo, "difference_yield_pp_hi": phi}


def variable_frame(screen, c, d, columns):
    v = lib.condition_variables(screen, c, d.rxn_class.iloc[0], list(columns))
    # lib deliberately removes constants; restore metal to expose trivial cases.
    if screen in lib.METAL_SCREENS and "metal" not in v:
        cc = c[c.rxn_class == d.rxn_class.iloc[0]].set_index("condition_id").reindex(columns)
        v["metal"] = cc["Reagent1_Standardized Name"].map(lib.metal_of).values
    return v


def condition_audit(screen, label, v, n):
    rows = []
    for variable in v:
        counts = v[variable].value_counts()
        rows.append({"screen": screen, "label": label, "variable": variable,
                     "conditions": n, "levels": len(counts),
                     "largest_level": str(counts.index[0]),
                     "largest_level_conditions": int(counts.iloc[0]),
                     "largest_level_share": float(counts.iloc[0] / n),
                     "fixed_feasible_levels": int((counts >= K).sum()),
                     "diverse_feasible": len(counts) >= K,
                     "fixed_structurally_nonbinding": len(counts) == 1,
                     "diverse_structurally_nonbinding": len(counts) == n,
                     "level_counts_json": json.dumps(counts.to_dict(), sort_keys=True)})
    return rows


def association_audit(screen, label, v):
    if not {"ligand", "precatalyst"}.issubset(v):
        return None
    t = pd.crosstab(v.ligand, v.precatalyst)
    assoc = lib.design_association(v[["ligand", "precatalyst"]]).iloc[0].cramers_v
    lig_to_pre = (t > 0).sum(axis=1)
    pre_to_lig = (t > 0).sum(axis=0)
    ix, jx = np.triu_indices(len(v), 1)
    same_lig = v.ligand.values[ix] == v.ligand.values[jx]
    same_pre = v.precatalyst.values[ix] == v.precatalyst.values[jx]
    return {"screen": screen, "label": label, "conditions": len(v),
            "ligand_levels": t.shape[0], "precatalyst_levels": t.shape[1],
            "ligand_precatalyst_cramers_v": float(assoc),
            "ligands_seen_with_one_precatalyst": int((lig_to_pre == 1).sum()),
            "precatalysts_seen_with_one_ligand": int((pre_to_lig == 1).sum()),
            "condition_pairs_same_ligand": int(same_lig.sum()),
            "condition_pairs_same_precatalyst": int(same_pre.sum()),
            "same_ligand_pairs_also_same_precatalyst_fraction": float(same_pre[same_lig].mean()) if same_lig.any() else np.nan,
            "same_precatalyst_pairs_also_same_ligand_fraction": float(same_lig[same_pre].mean()) if same_pre.any() else np.nan,
            "identical_partitions": bool(np.array_equal(same_lig, same_pre))}


def write(name, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(OUT / ("review_axes_" + name + ".csv"), index=False)
    return frame


def summarize_axis_costs(frame):
    rows = []
    for (variable, constraint), g0 in frame[frame.feasible].groupby(["variable", "constraint"]):
        for scope in ("all_feasible", "cv_binding_at_least_once", "both_constraints_feasible"):
            g = g0
            if scope == "cv_binding_at_least_once":
                g = g[g.cv_folds_free_violates > 0]
            elif scope == "both_constraints_feasible":
                keys = frame[frame.feasible].groupby(["screen", "variable"]).size()
                both = {s for (s, v), count in keys.items() if v == variable and count == 2}
                g = g[g.screen.isin(both)]
            if len(g) == 0:
                continue
            rng = np.random.default_rng(stable_seed(variable + constraint + scope))
            x = g.difference_yield_pp.values
            lo, hi = interval(np.median(x[rng.integers(len(x), size=(NBOOT, len(x)))], axis=1))
            rows.append({"variable": variable, "constraint": constraint, "scope": scope,
                         "arrays": len(g), "pairs_total": int(g.pairs.sum()),
                         "median_yield_cost_pp": float(np.median(x)),
                         "array_bootstrap_median_cost_lo": lo, "array_bootstrap_median_cost_hi": hi,
                         "median_retention_cost": float(g.difference_retention.median()),
                         "min_yield_cost_pp": float(x.min()), "max_yield_cost_pp": float(x.max()),
                         "arrays_free_never_violates": int((g.cv_folds_free_violates == 0).sum()),
                         "arrays_structurally_nonbinding": int(g.structurally_nonbinding.sum()),
                         "arrays_pair_bootstrap_cost_above_zero": int((g.difference_yield_pp_lo > 0).sum())})
    return write("constraint_summary", rows)


def library_audit(a, matrices):
    kc = pd.read_csv(OUT / "kit_k90ci.csv").set_index("screen")
    screens = sorted(s for s in matrices if s.startswith("Buchwald-Hartwig"))
    rows = []
    for x, y in combinations(screens, 2):
        ax, ay = a[a.screen == x], a[a.screen == y]
        sx, sy = set(ax.subpair), set(ay.subpair)
        ux = {tuple(sorted(p.split(" >> "))) for p in sx}
        uy = {tuple(sorted(p.split(" >> "))) for p in sy}
        xci, yci = kc.loc[x], kc.loc[y]
        rows.append({"screen_a": x, "screen_b": y,
                     "label_a": ax.screen_label.iloc[0], "label_b": ay.screen_label.iloc[0],
                     "pairs_a": len(sx), "pairs_b": len(sy),
                     "shared_exact_substrate_pairs": len(sx & sy),
                     "shared_unordered_substrate_pairs": len(ux & uy),
                     "k90_a": int(xci.k90), "k90_b": int(yci.k90),
                     "k90_a_lo": xci.k90_lo, "k90_a_hi": xci.k90_hi,
                     "k90_b_lo": yci.k90_lo, "k90_b_hi": yci.k90_hi,
                     "k90_intervals_overlap": bool(max(xci.k90_lo, yci.k90_lo) <= min(xci.k90_hi, yci.k90_hi))})
    return write("bh_library_audit", rows)


def duplicate_audit():
    raw = pd.read_csv(lib.DATASET / "arrays.csv", low_memory=False)
    raw = raw[raw["measurement_status"].isin(["quantified", "zero_by_rule"])]  # measured cells only
    metadata = pd.read_csv(lib.DATASET / "conditions.csv", low_memory=False)
    dup_meta = metadata[(metadata.rxn_class == "Hiyama") & metadata.condition_id.isin(["C43", "C46"])].set_index("condition_id")
    composition = [col for col in metadata.columns if col.startswith("reagent")] + ["solvent", "solvent_id", "temperature_C", "time_h"]
    assert dup_meta.loc["C43", composition].fillna("missing").equals(dup_meta.loc["C46", composition].fillna("missing"))
    keys = ["array_id", "sub_1_smiles", "sub_2_smiles"]
    one = raw[(raw.array_id == "Hiyama|1") & (raw.condition_id == "C43") & (raw["set"] == "analysis")]
    two = raw[(raw.array_id == "Hiyama|1") & (raw.condition_id == "C46") & (raw["set"] == "replicate")]
    both = one[keys + ["yield"]].merge(two[keys + ["yield"]], on=keys, suffixes=("_C43", "_C46"), validate="one_to_one")
    both["absolute_difference_pp"] = 100 * (both.yield_C43 - both.yield_C46).abs()
    both["either_yield_above_5pct"] = both[["yield_C43", "yield_C46"]].max(axis=1) > .05
    both["both_zero"] = (both.yield_C43 == 0) & (both.yield_C46 == 0)
    write("duplicate_pairs", both)
    rows = []
    for subset, d in [("all", both), ("either_yield_above_5pct", both[both.either_yield_above_5pct]),
                      ("either_yield_positive", both[~both.both_zero])]:
        diff = d.absolute_difference_pp
        rows.append({"subset": subset, "pairs": len(d), "both_zero_pairs": int(d.both_zero.sum()),
                     "exactly_equal_pairs": int((diff == 0).sum()),
                     "pearson_r": float(d.yield_C43.corr(d.yield_C46)),
                     "mean_absolute_difference_pp": float(diff.mean()),
                     "median_absolute_difference_pp": float(diff.median()),
                     "p90_absolute_difference_pp": float(diff.quantile(.90)),
                     "p95_absolute_difference_pp": float(diff.quantile(.95)),
                     "max_absolute_difference_pp": float(diff.max()),
                     "mean_signed_C46_minus_C43_pp": float(100 * (d.yield_C46 - d.yield_C43).mean())})
    return write("duplicate_summary", rows)


def main():
    OUT.mkdir(exist_ok=True)
    paths = [lib.DATASET / "arrays.csv", lib.DATASET / "conditions.csv", HERE / "lib.py", HERE / "kit_axes.py",
             OUT / "kit_axes.csv", OUT / "kit_k90ci.csv", OUT / "kit_curve.csv"]
    before = {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    a, c = lib.load()
    audits, associations, costs, records, panel_rows = [], [], [], [], []
    recipe_rows, contrasts, matrices = [], [], {}
    for screen in sorted(a.screen.unique()):
        d = a[a.screen == screen]
        label = d.screen_label.iloc[0]
        m = lib.screen_matrix(a, screen)
        matrices[screen] = m
        y = m.values
        n_pairs, n = y.shape
        assert np.nanmin(y) >= 0 and np.nanmax(y) <= 1
        v = variable_frame(screen, c, d, m.columns)
        audits.extend(condition_audit(screen, label, v, n))
        assoc = association_audit(screen, label, v)
        if assoc is not None:
            associations.append(assoc)
        full = lib.retention(y, list(range(n)))[1]
        fold_id = np.empty(n_pairs, dtype=int)
        folds = np.array_split(np.random.RandomState(FOLD_SEED).permutation(n_pairs), NFOLD)
        scores = {"greedy6": np.empty(n_pairs), "topmean6": np.empty(n_pairs)}
        axis_specs = {}
        for variable in v:
            counts = v[variable].value_counts()
            for constraint in ("fixed", "diverse"):
                feasible = counts.max() >= K if constraint == "fixed" else len(counts) >= K
                spec = {"feasible": bool(feasible), "violated": 0,
                        "structurally_nonbinding": len(counts) == (1 if constraint == "fixed" else n)}
                if feasible:
                    scores[constraint + ":" + variable] = np.empty(n_pairs)
                axis_specs[(variable, constraint)] = spec
        for fold, test in enumerate(folds):
            fold_id[test] = fold
            train = np.setdiff1d(np.arange(n_pairs), test)
            yt = y[train]
            free = lib.greedy_kit(yt, K)
            # Match the kits.py baseline: condition means use measured training wells only.
            top = list(np.argsort(-np.nanmean(yt, axis=0))[:K])
            for method, kit in (("greedy6", free), ("topmean6", top)):
                scores[method][test] = lib.retention(y[test], kit)[0]
                panel_rows.append({"screen": screen, "fold": fold, "method": method,
                                   "selection_uses_yields": True, "conditions": ";".join(m.columns[kit])})
            for (variable, constraint), spec in axis_specs.items():
                labels = v[variable].values
                levels_free = len(np.unique(labels[free]))
                violates = levels_free != (1 if constraint == "fixed" else K)
                spec["violated"] += int(violates)
                if not spec["feasible"]:
                    continue
                if constraint == "diverse":
                    kit = lib.greedy_kit(yt, K, distinct=labels)
                else:
                    kit, val = [], -1.0
                    for level in pd.unique(labels):
                        allowed = labels == level
                        if allowed.sum() < K:
                            continue
                        candidate = lib.greedy_kit(yt, K, allowed=allowed)
                        candidate_val = lib.retention(yt, candidate)[0].mean()
                        if candidate_val > val:
                            kit, val = candidate, candidate_val
                assert len(kit) == K and len(set(kit)) == K
                assert len(np.unique(labels[kit])) == (1 if constraint == "fixed" else K)
                scores[constraint + ":" + variable][test] = lib.retention(y[test], kit)[0]
        rng = np.random.default_rng(stable_seed(screen + "pairs"))
        weights = rng.multinomial(n_pairs, np.full(n_pairs, 1 / n_pairs), size=NBOOT) / n_pairs
        assert np.all(weights @ full > 0)
        for (variable, constraint), spec in axis_specs.items():
            row = {"screen": screen, "label": label, "pairs": n_pairs, "conditions": n,
                   "variable": variable, "constraint": constraint, "levels": v[variable].nunique(),
                   "feasible": spec["feasible"], "kit_size": K if spec["feasible"] else np.nan,
                   "structurally_nonbinding": spec["structurally_nonbinding"],
                   "cv_folds": NFOLD, "cv_folds_free_violates": spec["violated"],
                   "mean_best_full": full.mean(), "free_retention": scores["greedy6"].mean() / full.mean()}
            if spec["feasible"]:
                z = scores[constraint + ":" + variable]
                row["constrained_retention"] = z.mean() / full.mean()
                row.update(effect(scores["greedy6"], z, full, weights))
            costs.append(row)
        # Six is possible only in arrays with at least six encoded ligand levels.
        if "ligand" in v and v.ligand.nunique() >= K:
            scores["random6"] = expected_random(y, K)
            scores["ligand_uniform6"] = expected_level_diverse(y, v.ligand.values, K)
            if "metal" in v:
                counts = v.metal.value_counts()
                majority = sorted(counts[counts == counts.max()].index)[0]
                keep = v.metal.values == majority
                if v.loc[keep, "ligand"].nunique() >= K:
                    scores["majoritymetal_ligand_uniform6"] = expected_level_diverse(y[:, keep], v.ligand.values[keep], K)
            methods = ["greedy6", "topmean6", "random6", "ligand_uniform6", "majoritymetal_ligand_uniform6"]
            for method in methods:
                if method not in scores:
                    continue
                z = scores[method]
                lo, hi = interval((weights @ z) / (weights @ full))
                recipe_rows.append({"screen": screen, "label": label, "method": method, "pairs": n_pairs,
                                    "conditions": n, "kit_size": K, "ligand_levels": v.ligand.nunique(),
                                    "retention": z.mean() / full.mean(), "retention_lo": lo, "retention_hi": hi,
                                    "mean_best_yield_pp": 100 * z.mean(),
                                    "selection_uses_yields": method in ("greedy6", "topmean6"),
                                    "evaluation": "out_of_fold" if method in ("greedy6", "topmean6") else "exact_expected_randomized_panel"})
            for left, right in [("ligand_uniform6", "random6"), ("greedy6", "ligand_uniform6"),
                                ("topmean6", "ligand_uniform6"), ("greedy6", "topmean6"),
                                ("majoritymetal_ligand_uniform6", "ligand_uniform6")]:
                if left in scores and right in scores:
                    row = {"screen": screen, "label": label, "pairs": n_pairs, "left": left, "right": right}
                    row.update(effect(scores[left], scores[right], full, weights))
                    contrasts.append(row)
        for method, z in scores.items():
            for i in range(n_pairs):
                records.append({"screen": screen, "pair_index": i, "subpair": m.index[i], "fold": int(fold_id[i]),
                                "method": method, "best_yield": z[i], "full_best_yield": full[i]})
        print(screen, "checked", flush=True)
    audit = write("design_audit", audits)
    write("ligand_precatalyst_association", associations)
    costs = write("constraints", costs)
    summarize_axis_costs(costs)
    write("pair_predictions", records)
    write("learned_panels", panel_rows)
    recipes = write("recipes", recipe_rows)
    contrasts = write("recipe_contrasts", contrasts)
    summary = []
    common = set(recipes[recipes.method == "ligand_uniform6"].screen)
    for method, g in recipes.groupby("method"):
        g = g[g.screen.isin(common)]
        summary.append({"method": method, "arrays": len(g), "pairs": int(g.pairs.sum()),
                        "median_retention": g.retention.median(), "mean_retention": g.retention.mean(),
                        "min_retention": g.retention.min(), "max_retention": g.retention.max()})
    write("recipe_summary", summary)
    contrast_summary = []
    for (left, right), g in contrasts.groupby(["left", "right"]):
        values = g.difference_retention.values
        rng = np.random.default_rng(stable_seed(left + right + "arraycontrast"))
        lo, hi = interval(np.median(values[rng.integers(len(values), size=(NBOOT, len(values)))], axis=1))
        contrast_summary.append({"left": left, "right": right, "arrays": len(g),
                                 "median_paired_retention_difference": float(np.median(values)),
                                 "array_bootstrap_median_difference_lo": lo,
                                 "array_bootstrap_median_difference_hi": hi})
    write("recipe_contrast_summary", contrast_summary)
    bh = library_audit(a, matrices)
    duplicate = duplicate_audit()
    old = pd.read_csv(OUT / "kit_axes.csv")
    checked = costs[costs.feasible].copy()
    checked["kit"] = checked.constraint + ":" + checked.variable
    matched = checked.merge(old, on=["screen", "kit"], suffixes=("_new", "_old"))
    assert np.allclose(matched.constrained_retention, matched.retention, rtol=0, atol=1e-12)
    old_curve = pd.read_csv(OUT / "kit_curve.csv").query("k == 6").set_index("screen")
    for method, source_col in [("greedy6", "retention_greedy"), ("topmean6", "retention_meanyield")]:
        z = recipes[recipes.method == method].set_index("screen")
        assert np.allclose(z.retention, old_curve.loc[z.index, source_col], rtol=0, atol=1e-12)
    assert set(recipes[recipes.method == "random6"].screen) == common
    assert set(recipes[recipes.method == "greedy6"].screen) == common
    assert set(recipes[recipes.method == "topmean6"].screen) == common
    assert len(bh) == 10 and len(duplicate) == 3
    assert duplicate.loc[duplicate.subset == "all", "pairs"].iloc[0] == 74
    after = {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    assert before == after, "Read-only source files changed"
    manifest = {"seed": SEED, "fold_seed": FOLD_SEED, "folds": NFOLD, "bootstrap_replicates": NBOOT,
                "budget": K, "input_sha256": before,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "numpy": np.__version__, "pandas": pd.__version__,
                "checks": {"exhaustive_recipe_enumeration": "pass", "existing_constraint_retentions_reproduced": len(matched),
                           "same_arrays_core_recipes": len(common), "existing_core_trained_baselines_reproduced": 2 * len(common),
                           "duplicate_composition_equal": True, "input_files_unchanged": True}}
    (OUT / "review_axes_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (OUT / "review_axes_methods.md").write_text(METHODS, encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False), flush=True)
    print(duplicate.to_string(index=False), flush=True)
    print("Verification passed; output prefix out/review_axes", flush=True)


METHODS = """# Review checks: constrained axes and data-independent panels

Reproduce with `python review_axes_gc.py`; `--self-test` runs independent exhaustive
small-library probability checks. All outputs use the `out/review_axes` prefix.
Seed 20260920; the existing ten folds (seed 3) and all analysis pairs are retained.
Existing `lib.py` definitions of ligand (added plus embedded), metal and yield apply.
Missing selected wells score zero; full denominator is the best observed well.

## Axis costs and what binds

Each six-condition constraint is refitted on nine folds and evaluated on the tenth,
exactly as in `kit_axes.py`. Fixed-axis selection also chooses the level on training
yields. These are yield-informed constrained greedy panels, not recipes inferred
from metadata. Positive costs are free-minus-constrained yield or retention.
Feasibility is decided solely from library counts; infeasible six-panels are not
scored as smaller panels. Constant metal axes, previously omitted by `lib.py`, are
restored and identified as structurally nonbinding. A CV binding count records how
many of the ten unconstrained panels violate each restriction; zero is empirical
nonbinding, not evidence that the axis can never matter. Negative observed costs
are possible with a greedy heuristic and held-out evaluation.

Per-array 95% percentile intervals use 4,000 paired bootstrap resamples of substrate
pairs and their out-of-fold predictions. The panels are not refitted in these
resamples: intervals describe pair-sampling variability conditional on fitted folds,
omit panel-fitting uncertainty, and are not adjusted for multiple comparisons.
Cross-array median intervals separately resample whole arrays and are descriptive
for this heterogeneous convenience corpus, not a universal population of chemistry.
Summary scopes explicitly distinguish all feasible, at-least-once CV-binding, and
the same arrays where both fixed and diverse six-condition constraints are feasible.
Cramer's V and pair partition overlap are design-association diagnostics; neither
estimates a causal ligand effect independent of the precatalyst and other factors.

## Train-free rule at the same six-well budget

`ligand_uniform6`: choose six distinct encoded ligand levels uniformly without
replacement, then choose one condition uniformly within each chosen level.
`random6`: choose six distinct condition columns uniformly without replacement.
These selection probabilities use metadata and budget only; no yields, substrate
features, training-fold performance or selected panel are used to tune the rule.
The exact expected maximum over every possible panel is calculated analytically,
using the elementary symmetric polynomial of within-level yield CDFs for the
ligand rule and order-statistic weights for random6. Thus no Monte Carlo panel
sampling error or selection of a favorable seed is present. This is performance
averaged over randomized six-panel realizations, not a fixed optimized panel.
The rules were specified for this retrospective audit, not prospectively registered.
Different expert choices of ligands/conditions are not evaluated by this rule.

`greedy6` and `topmean6` use training yields from the same ten folds and are scored
on the same held-out pairs. Topmean takes the six greatest column means over
measured wells only, with the existing `kits.py` NumPy ordering; greedy
maximizes training mean best yield. Core recipe
comparisons are restricted to the same 12 arrays with at least six ligand levels,
and every panel has exactly six distinct conditions. The ligand-free encoded level
`none` is a legitimate level. The optional `majoritymetal_ligand_uniform6` first
restricts to the metal with the most library columns (alphabetical tie-break), then
applies the identical ligand rule; it is reported separately and uses no yields.
Per-array recipe intervals and contrasts use the same paired bootstrap weights as
axis costs. They describe the expected recipe performance across sampled pairs;
they do not describe the variation between individual random panel realizations.

## Library-size and replicate limitations

BH overlap is counted for the exact released substrate SMILES, both ordered and
unordered pairs. Existing 400-refit-bootstrap k90 intervals are audited, not
reinterpreted as a test of library causality; 13 denotes right-censoring above 12.
Different substrate panels, condition sets, temperatures and sparse overlap prevent
a controlled attribution of differences to the condition library alone. Overlapping
intervals neither establish identical sizes nor establish an ordered ranking.

The Hiyama C43/C46 duplicate is joined one-to-one on both substrate SMILES and
summarized overall and for either well above 5% yield. A single condition's duplicate
distribution, dominated by zero yields, does not estimate corpus-wide measurement
error or the winner's curse of maxima over roughly 90 conditions. No noise simulation,
empirical noise bound, or separation of the retention shortfall into noise and
chemistry is claimed.

## Files

- `design_audit.csv`: level counts, library dominance, feasibility and triviality.
- `ligand_precatalyst_association.csv`: Cramer's V and shared-level association.
- `constraints.csv`: per-array fixed/diverse costs, conditional paired intervals,
  feasibility and CV binding counts. `difference_*` is free minus constrained.
- `constraint_summary.csv`: medians under each explicitly labelled array scope.
- `pair_predictions.csv`: all per-pair scores, full denominator and CV fold.
- `learned_panels.csv`: six condition IDs for every trained core panel and fold.
- `recipes.csv`, `recipe_contrasts.csv`, `recipe_summary.csv`,
  `recipe_contrast_summary.csv`: same-budget recipe
  results; contrast direction is explicitly `left` minus `right`.
- `bh_library_audit.csv`: pair overlap and overlap of existing censored k90 intervals.
- `duplicate_pairs.csv`, `duplicate_summary.csv`: replicate distribution.
- `manifest.json`: input hashes, versions, deterministic settings and checks.
"""


def self_test():
    """Independent exhaustive small cases verify design probabilities and scoring."""
    y = np.array([[0.0, .2, .9, .4], [.8, .1, np.nan, .6]])
    labels = np.array(["a", "a", "b", "c"])
    plain = np.mean([lib.retention(y, list(ix))[0]
                     for ix in combinations(range(4), 2)], axis=0)
    assert np.allclose(expected_random(y, 2), plain)
    panels = []
    for levels in combinations(np.unique(labels), 2):
        within = [np.flatnonzero(labels == level) for level in levels]
        panels.append(np.mean([lib.retention(y, list(ix))[0]
                               for ix in product(*within)], axis=0))
    assert np.allclose(expected_level_diverse(y, labels, 2), np.mean(panels, axis=0))
    assert np.allclose(expected_random(y, 4), [.9, .8])
    assert np.allclose(expected_level_diverse(y, np.arange(4), 2), plain)
    assert np.allclose(expected_level_diverse(y, labels, 1),
                       np.mean([np.nan_to_num(y[:, labels == l]).mean(1)
                                for l in np.unique(labels)], axis=0))
    try:
        expected_level_diverse(y, labels, 4)
    except ValueError:
        pass
    else:
        raise AssertionError("An infeasible six-panel must not become a short panel")
    assert stable_seed("a") == stable_seed("a") != stable_seed("b")
    print("Independent exhaustive recipe checks passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test()
    if not args.self_test:
        main()
