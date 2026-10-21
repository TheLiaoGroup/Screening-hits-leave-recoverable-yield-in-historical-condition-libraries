"""Pair-level shortfall and descriptive chemical coverage audit.

Run `python review_loss_gc.py`; writes only out/review_loss*.
Annotations are read from this release's bundled out/review_loss_annotations.csv
and its provenance sidecar.
"""
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
SEED = 3
ANNOTATION_COLUMNS = ["screen", "subpair", "sub_1_smiles", "sub_2_smiles", "sub_1_FG", "sub_2_FG"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, data):
    frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    frame.to_csv(OUT / ("review_loss_" + name + ".csv"), index=False)
    return frame


def fraction(n, d):
    return float(n / d) if d else np.nan


def metrics(d, method):
    loss = d[method + "_loss_pp"]
    productive = d.full_best_yield > 0
    risk = d.full_best_yield > .20
    high = d.full_best_yield > .50
    severe = loss > 20
    n = len(d)
    lp = loss[productive]
    return {"pairs_all": n, "pairs_productive": int(productive.sum()),
            "pairs_all_zero": int((~productive).sum()), "pairs_at_risk_20pp": int(risk.sum()),
            "pairs_full_yield_above_50pct": int(high.sum()),
            "loss_above_10pp_pairs": int((loss > 10).sum()), "loss_above_20pp_pairs": int(severe.sum()),
            "loss_above_10pp_fraction_productive": fraction((loss > 10).sum(), productive.sum()),
            "loss_above_20pp_fraction_all": fraction(severe.sum(), n),
            "loss_above_20pp_fraction_productive": fraction(severe.sum(), productive.sum()),
            "loss_above_20pp_fraction_at_risk": fraction(severe.sum(), risk.sum()),
            "productive_loss_median_pp": float(lp.median()) if len(lp) else np.nan,
            "productive_loss_p90_pp": float(lp.quantile(.90)) if len(lp) else np.nan,
            "severe_with_full_yield_above_50pct": int((severe & high).sum()),
            "severe_with_panel_yield_at_least_20pct": int((severe & (d[method + "_best_yield"] >= .20)).sum()),
            "full_above_50pct_panel_below_20pct": int((high & (d[method + "_best_yield"] < .20)).sum()),
            "severe_distinct_partner1": int(d.loc[severe, "sub_1_smiles"].nunique()),
            "severe_distinct_partner2": int(d.loc[severe, "sub_2_smiles"].nunique()),
            "severe_max_pairs_one_partner1": int(d.loc[severe, "sub_1_smiles"].value_counts().max()) if severe.any() else 0,
            "severe_max_pairs_one_partner2": int(d.loc[severe, "sub_2_smiles"].value_counts().max()) if severe.any() else 0,
            "severe_median_full_yield": float(d.loc[severe, "full_best_yield"].median()) if severe.any() else np.nan,
            "severe_median_panel_yield": float(d.loc[severe, method + "_best_yield"].median()) if severe.any() else np.nan}


def self_test():
    x = pd.DataFrame({"full_best_yield": [0., .1, .2, .6], "greedy_best_yield": [0., .1, 0., .1],
                      "sub_1_smiles": ["a", "a", "b", "b"], "sub_2_smiles": ["x", "y", "x", "y"]})
    x["greedy_loss_pp"] = 100 * (x.full_best_yield - x.greedy_best_yield)
    m = metrics(x, "greedy")
    assert m["pairs_all"] == 4 and m["pairs_productive"] == 3 and m["pairs_all_zero"] == 1
    assert m["loss_above_20pp_pairs"] == 1 and m["pairs_at_risk_20pp"] == 1
    assert m["loss_above_20pp_fraction_productive"] == 1 / 3
    assert m["loss_above_20pp_fraction_all"] == .25 and m["loss_above_20pp_fraction_at_risk"] == 1
    assert m["productive_loss_median_pp"] == 20
    assert m["full_above_50pct_panel_below_20pct"] == 1
    print("Loss threshold and denominator checks passed", flush=True)


def load_annotations(needed):
    bundle = OUT / "review_loss_annotations.csv"
    provenance_path = OUT / "review_loss_annotations_provenance.json"
    if not bundle.exists() or not provenance_path.exists():
        raise FileNotFoundError("This analysis requires out/review_loss_annotations.csv and "
                                "out/review_loss_annotations_provenance.json from the released package")
    annotations = pd.read_csv(bundle, keep_default_na=False)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert digest(bundle) == provenance["export_sha256"], "Bundled annotation checksum changed"
    loading = "bundled_pair_annotation_table"
    assert not annotations.duplicated(["screen", "subpair"]).any()
    assert set(ANNOTATION_COLUMNS).issubset(annotations)
    matched = needed.merge(annotations, on=["screen", "subpair"], how="left", validate="one_to_one", indicator=True)
    assert matched._merge.eq("both").all()
    assert (matched.sub_1_smiles + " >> " + matched.sub_2_smiles).equals(matched.subpair)
    return annotations, {**provenance, "loading_mode_this_run": loading,
                         "blank_partner1_FG_pairs": int((annotations.sub_1_FG == "").sum()),
                         "blank_partner2_FG_pairs": int((annotations.sub_2_FG == "").sum())}


def score_pairs(a):
    rows = []
    for screen in sorted(a.screen.unique()):
        d = a[a.screen == screen]
        m = lib.screen_matrix(a, screen)
        y = m.values
        n, ncond = y.shape
        yf = np.where(np.isnan(y), -np.inf, y)
        full = lib.retention(y, list(range(ncond)))[1]
        best_index = np.argmax(yf, axis=1)
        fold_ids = np.empty(n, int)
        scores = {method: np.empty(n) for method in ["greedy", "topmean"]}
        best_ids = {method: np.empty(n, dtype=object) for method in scores}
        panel_ids = {method: np.empty(n, dtype=object) for method in scores}
        folds = np.array_split(np.random.RandomState(SEED).permutation(n), NFOLD)
        for fold, test in enumerate(folds):
            train = np.setdiff1d(np.arange(n), test)
            assert not len(np.intersect1d(train, test))
            fold_ids[test] = fold
            panels = {"greedy": lib.greedy_kit(y[train], K),
                      "topmean": list(np.argsort(-np.nanmean(y[train], axis=0))[:K])}
            for method, panel in panels.items():
                assert len(panel) == K and len(set(panel)) == K
                scores[method][test] = lib.retention(y[test], panel)[0]
                within = yf[test][:, panel]
                chosen = np.argmax(within, axis=1)
                missing_all = ~np.isfinite(within).any(axis=1)
                best_ids[method][test] = [str(m.columns[panel[j]]) if not missing_all[i] else ""
                                         for i, j in enumerate(chosen)]
                panel_ids[method][test] = ";".join(m.columns[panel])
        meta = d.drop_duplicates("subpair").set_index("subpair")
        for i, pair in enumerate(m.index):
            row = {"screen": screen, "label": d.screen_label.iloc[0], "subpair": pair,
                   "sub_1_smiles": meta.loc[pair, "sub_1_smiles"], "sub_2_smiles": meta.loc[pair, "sub_2_smiles"],
                   "role_1": meta.loc[pair, "role_1"], "role_2": meta.loc[pair, "role_2"],
                   "fold": int(fold_ids[i]), "conditions": ncond, "full_best_yield": full[i],
                   "full_best_condition": m.columns[best_index[i]], "productive": bool(full[i] > 0),
                   "at_risk_loss_above_20pp": bool(full[i] > .20)}
            for method in scores:
                row[method + "_best_yield"] = scores[method][i]
                row[method + "_loss_pp"] = 100 * (full[i] - scores[method][i])
                row[method + "_best_condition"] = best_ids[method][i]
                row[method + "_panel"] = panel_ids[method][i]
            rows.append(row)
        print(screen, "scored", flush=True)
    pairs = pd.DataFrame(rows)
    assert not pairs.duplicated(["screen", "subpair"]).any()
    for method in ["greedy", "topmean"]:
        assert pairs[method + "_loss_pp"].between(-1e-10, 100 + 1e-10).all()
    return pairs


def molecule_summaries(pairs):
    rows = []
    for (screen, label), d in pairs.groupby(["screen", "label"]):
        for partner in (1, 2):
            col = f"sub_{partner}_smiles"
            for molecule, g in d.groupby(col):
                severe = g.greedy_loss_pp > 20
                risk = g.full_best_yield > .20
                productive = g.full_best_yield > 0
                rows.append({"screen": screen, "label": label, "partner": partner,
                             "role": g[f"role_{partner}"].iloc[0], "smiles": molecule,
                             "pairs": len(g), "productive_pairs": int(productive.sum()),
                             "at_risk_pairs": int(risk.sum()), "severe_pairs": int(severe.sum()),
                             "severe_fraction_productive": fraction(severe.sum(), productive.sum()),
                             "severe_fraction_at_risk": fraction(severe.sum(), risk.sum()),
                             "mean_full_yield": g.full_best_yield.mean(),
                             "mean_panel_yield": g.greedy_best_yield.mean(),
                             "mean_loss_pp": g.greedy_loss_pp.mean()})
    return write("molecule_summary", rows)


def fg_summaries(pairs):
    rows = []
    consistency = []
    for partner in (1, 2):
        fgcol, smcol = f"sub_{partner}_FG", f"sub_{partner}_smiles"
        tagset = pairs[fgcol].fillna("").map(lambda s: tuple(sorted(set(t.strip() for t in str(s).split(";") if t.strip()))))
        check = pairs.assign(tagset=tagset).groupby(smcol).tagset.nunique()
        consistency.append({"partner": partner, "unique_smiles": len(check),
                            "smiles_with_multiple_annotation_sets": int((check > 1).sum())})
        exp = pairs.assign(functional_group=tagset).explode("functional_group").dropna(subset=["functional_group"])
        for (screen, label, tag), g in exp.groupby(["screen", "label", "functional_group"]):
            d = pairs[pairs.screen == screen]
            bg = d[~d.subpair.isin(g.subpair)]
            severe, productive, risk = g.greedy_loss_pp > 20, g.full_best_yield > 0, g.full_best_yield > .20
            bg_severe, bg_productive, bg_risk = bg.greedy_loss_pp > 20, bg.full_best_yield > 0, bg.full_best_yield > .20
            array_severe = d.greedy_loss_pp > 20
            grouped = g.assign(severe=severe, productive=productive, risk=risk).groupby(smcol).agg(
                severe=("severe", "sum"), productive=("productive", "sum"), risk=("risk", "sum"))
            mol_productive = grouped[grouped.productive > 0]
            mol_risk = grouped[grouped.risk > 0]
            severe_counts = g.loc[severe, smcol].value_counts()
            row = {"screen": screen, "label": label, "partner": partner,
                   "role": g[f"role_{partner}"].iloc[0], "functional_group": tag,
                   "tagged_pairs_all": len(g), "tagged_pairs_productive": int(productive.sum()),
                   "tagged_pairs_at_risk": int(risk.sum()), "tagged_severe_pairs": int(severe.sum()),
                   "tagged_severe_fraction_productive": fraction(severe.sum(), productive.sum()),
                   "tagged_severe_fraction_at_risk": fraction(severe.sum(), risk.sum()),
                   "untagged_pairs_productive": int(bg_productive.sum()), "untagged_pairs_at_risk": int(bg_risk.sum()),
                   "untagged_severe_pairs": int(bg_severe.sum()),
                   "untagged_severe_fraction_productive": fraction(bg_severe.sum(), bg_productive.sum()),
                   "untagged_severe_fraction_at_risk": fraction(bg_severe.sum(), bg_risk.sum()),
                   "array_pairs_productive": int((d.full_best_yield > 0).sum()),
                   "array_pairs_at_risk": int((d.full_best_yield > .20).sum()),
                   "array_severe_pairs": int(array_severe.sum()),
                   "tag_prevalence_productive": fraction(productive.sum(), (d.full_best_yield > 0).sum()),
                   "tag_prevalence_at_risk": fraction(risk.sum(), (d.full_best_yield > .20).sum()),
                   "tag_prevalence_among_severe": fraction(severe.sum(), array_severe.sum()),
                   "tagged_distinct_molecules_all": g[smcol].nunique(),
                   "tagged_distinct_molecules_productive": g.loc[productive, smcol].nunique(),
                   "tagged_distinct_molecules_at_risk": g.loc[risk, smcol].nunique(),
                   "tagged_distinct_molecules_severe": g.loc[severe, smcol].nunique(),
                   "other_partner_distinct_molecules_at_risk": g.loc[risk, f"sub_{3-partner}_smiles"].nunique(),
                   "max_severe_pairs_one_tagged_molecule": int(severe_counts.max()) if len(severe_counts) else 0,
                   "max_fraction_severe_from_one_tagged_molecule": fraction(severe_counts.max(), severe.sum()) if len(severe_counts) else np.nan,
                   "equal_molecule_mean_severe_fraction_productive": float((mol_productive.severe / mol_productive.productive).mean()),
                   "equal_molecule_mean_severe_fraction_at_risk": float((mol_risk.severe / mol_risk.risk).mean()),
                   "tagged_productive_median_loss_pp": float(g.loc[productive, "greedy_loss_pp"].median()),
                   "tagged_at_risk_median_full_yield": float(g.loc[risk, "full_best_yield"].median()),
                   "tagged_at_risk_median_panel_yield": float(g.loc[risk, "greedy_best_yield"].median()),
                   "has_at_least_10_at_risk_pairs_and_5_tagged_molecules": bool(risk.sum() >= 10 and g.loc[risk, smcol].nunique() >= 5)}
            rows.append(row)
    write("annotation_consistency", consistency)
    return write("functional_groups", rows)


def distributions(pairs):
    rows = []
    for method in ("greedy", "topmean"):
        for scope, mask in [("all_pairs", np.ones(len(pairs), bool)), ("productive_full_above_zero", pairs.full_best_yield > 0),
                            ("at_risk_full_above_20pct", pairs.full_best_yield > .20),
                            ("high_yield_full_above_50pct", pairs.full_best_yield > .50)]:
            z = pairs.loc[mask, method + "_loss_pp"]
            row = {"method": method, "scope": scope, "pairs": len(z),
                   "mean_loss_pp": z.mean(), "loss_above_10pp_pairs": int((z > 10).sum()),
                   "loss_above_20pp_pairs": int((z > 20).sum()),
                   "loss_above_10pp_fraction": float((z > 10).mean()),
                   "loss_above_20pp_fraction": float((z > 20).mean())}
            for q in (0, .1, .25, .5, .75, .9, .95, .99, 1):
                row[f"loss_p{int(100*q):02d}_pp"] = float(z.quantile(q))
            rows.append(row)
    return write("distribution", rows)


def yield_bands(pairs):
    rows = []
    groups = [("ALL", "All arrays", pairs)] + [(s, d.label.iloc[0], d) for s, d in pairs.groupby("screen")]
    for screen, label, d in groups:
        for name, mask in [("full_zero", d.full_best_yield == 0),
                           ("full_above_0_to_20pct", (d.full_best_yield > 0) & (d.full_best_yield <= .20)),
                           ("full_above_20_to_50pct", (d.full_best_yield > .20) & (d.full_best_yield <= .50)),
                           ("full_above_50pct", d.full_best_yield > .50)]:
            for method in ("greedy", "topmean"):
                z = d.loc[mask, method + "_loss_pp"]
                rows.append({"screen": screen, "label": label, "full_yield_band": name, "method": method,
                             "pairs": len(z), "severe_pairs": int((z > 20).sum()),
                             "severe_fraction": float((z > 20).mean()) if len(z) else np.nan,
                             "median_loss_pp": float(z.median()) if len(z) else np.nan})
    return write("yield_bands", rows)


def zero_yields(a, pairs):
    """Observed well-level zeros and all-zero pair maxima use separate denominators."""
    assert a["yield"].notna().all(), "Recorded-well denominator assumes every released analysis yield is present"
    rows = []
    groups = [("ALL", "All arrays", a, pairs)]
    groups += [(s, d.screen_label.iloc[0], d, pairs[pairs.screen == s])
               for s, d in a.groupby("screen")]
    for screen, label, wells, substrate_pairs in groups:
        zero_wells = int((wells["yield"] == 0).sum())
        zero_pairs = int((substrate_pairs.full_best_yield == 0).sum())
        rows.append({"screen": screen, "label": label, "recorded_wells": len(wells),
                     "exact_zero_wells": zero_wells,
                     "exact_zero_well_fraction": fraction(zero_wells, len(wells)),
                     "pairs_all": len(substrate_pairs), "zero_full_max_pairs": zero_pairs,
                     "zero_full_max_pair_fraction": fraction(zero_pairs, len(substrate_pairs))})
    table = write("zero_yields", rows)
    total = table[table.screen == "ALL"].iloc[0]
    by_array = table[table.screen != "ALL"]
    for col in ["recorded_wells", "exact_zero_wells", "pairs_all", "zero_full_max_pairs"]:
        assert total[col] == by_array[col].sum()
    # the totals must be those of the released analysis set, counted here from the file itself
    released = pd.read_csv(lib.DATASET / "arrays.csv", low_memory=False)
    released = released[(released["set"] == "analysis") & released["measurement_status"].isin(["quantified", "zero_by_rule"])]
    assert total.recorded_wells == len(released) and total.exact_zero_wells == int((released["yield"] == 0).sum())
    assert total.pairs_all == released[["array_id", "sub_1_smiles", "sub_2_smiles"]].drop_duplicates().shape[0]
    return table


METHODS = """# Pair-level loss audit

Run `python review_loss_gc.py` using the released arrays and `lib.py`. The first
run exports source functional-group annotations; subsequent runs use only the
bundled `out/review_loss_annotations.csv` and its checksum/provenance JSON. Source:
the sibling `NC-The optimal conditions belong to the substrate more than to the
transformation/dataset/DATASET_analysis.csv`. The export contains one row per
analysis array and substrate pair, both partner SMILES and the two original FG
strings. Labels are copied verbatim, not inferred or optimized from outcomes.

For each array, the same ten substrate-pair folds (RandomState seed 3) are used as
the established analysis. Greedy panels contain six conditions selected only on
the other nine folds. The topmean reference uses six greatest training column
means over measured training wells only, matching `kits.py`. Both methods are
evaluated on identical held-out pairs. A selected missing well contributes zero;
the full reference is the maximum observed yield over that pair's array. Loss in
yield percentage points equals 100*(full maximum - held-out panel maximum).
All figures are absolute yield points, not percent loss relative to full yield.

Productive means full maximum strictly greater than zero. An all-zero pair cannot
show a positive deficit and is excluded from productive denominators. The at-risk
denominator for a loss strictly above 20 points requires full maximum >20%.
`zero_yields.csv` separately counts exact-zero recorded analysis wells (denominator:
all recorded analysis wells) and pairs whose observed full maximum equals zero
(denominator: all distinct analysis pairs). These two fractions are not interchangeable.
High-yield means full maximum >50%; panel hit means panel maximum >=20%.
The reviewer-specified >10- and >20-point thresholds and existing 20% hit threshold
are descriptive; no threshold was optimized for a functional-group association.
Quantiles use pandas' default linear interpolation. The report gives counts with
every denominator and distinguishes productive, all-pair and at-risk fractions.

Pair-held-out validation can reuse either molecule in training. Pair counts are
therefore not counts of independent molecules, and this audit is not partner-held-
out validation. For each array and partner, molecule summaries and functional-group
tables report distinct molecule counts, worst single-molecule concentration, and
equal-molecule averages of pair-level failure fractions. These averages are a
descriptive reuse sensitivity, not cluster-corrected inference or new held-out
predictions. They still share co-partners and condition libraries.
Distinct molecule counts refer to distinct released SMILES strings, without new
canonicalization or merging of tautomers, salts, or stereoisomers.

Functional-group strings are split on semicolons, stripped and deduplicated within
each pair/partner. Every available tag is tabulated separately within each array
and partner role, including tags with no severe losses. Tags overlap and their
counts must not be added. Blank strings remain unannotated, not a chemical class.
Tables provide productive and at-risk tagged/untagged denominators, whole-array
baseline denominators, tag prevalence among all productive cases versus severe
cases, and distinct tagged/co-partner molecule counts. No cross-array pooled FG
effect, p-value, causal enrichment, mechanism, or predictive chemical classifier is
claimed. A transparent support flag requires >=10 at-risk pairs and >=5 distinct
tagged molecules; all rows remain available, and the flag is not a significance
criterion. Co-occurring tags, array composition and partner reuse preclude a
general failure mechanism from these exploratory counts.

`high_yield_misses.csv` lists every case with full yield >50% and greedy yield <20%,
without selecting examples by chemical label. Best-condition IDs use the first
maximal observed column in array order; ties must not be interpreted chemically.
The full maximum is itself measured and subject to measurement error and selection
of a maximum; this audit does not decompose observed loss into chemistry and noise.

Files: annotations and annotations_provenance (standalone source export),
pair_scores (all exact pairs and both learned scores), distribution (global scopes),
array_summary (both methods with all denominators), yield_bands (fixed full-yield
strata), zero_yields (well-level and pair-level zero fractions),
functional_groups (within-array, partner-specific descriptors),
molecule_summary (reuse-sensitive counts), annotation_consistency (label conflicts),
high_yield_misses (complete predefined slice), findings, methods, and manifest.
All outputs use `out/review_loss*`; established data and analyses are read-only.
"""


def findings_text(summary, distribution, fg):
    g = summary[(summary.screen == "ALL") & (summary.method == "greedy")].iloc[0]
    t = summary[(summary.screen == "ALL") & (summary.method == "topmean")].iloc[0]
    lines = ["# Verified loss distribution", "",
             "Historical tail endpoint: the ratios below use the original productive-pair/all-pair denominators and are not the post-hit cohort endpoint (184/1,424).", "",
             f"All {int(g.pairs_all):,} analysis pairs: {int(g.pairs_productive):,} productive (full maximum >0) and {int(g.pairs_all_zero):,} all-zero.",
             f"Greedy six: productive median loss {g.productive_loss_median_pp:.4f} yield points; 90th percentile {g.productive_loss_p90_pp:.4f}.",
             f"Loss >10 points: {int(g.loss_above_10pp_pairs)}/{int(g.pairs_productive)} ({100*g.loss_above_10pp_fraction_productive:.2f}%).",
             f"Loss >20 points: {int(g.loss_above_20pp_pairs)}/{int(g.pairs_productive)} ({100*g.loss_above_20pp_fraction_productive:.2f}%) productive; "
             f"{int(g.loss_above_20pp_pairs)}/{int(g.pairs_all)} ({100*g.loss_above_20pp_fraction_all:.2f}%) all pairs; "
             f"{int(g.loss_above_20pp_pairs)}/{int(g.pairs_at_risk_20pp)} ({100*g.loss_above_20pp_fraction_at_risk:.2f}%) at-risk pairs (full >20%).", "",
             "The review's largest per-array percentages use productive denominators:"]
    for screen in ["Buchwald-Hartwig|2", "Ullmann|1", "Buchwald-Hartwig(CMC)|3"]:
        z = summary[(summary.screen == screen) & (summary.method == "greedy")].iloc[0]
        lines.append(f"- {z.label}: {int(z.loss_above_20pp_pairs)}/{int(z.pairs_productive)} productive ({100*z.loss_above_20pp_fraction_productive:.2f}%), "
                     f"versus {int(z.loss_above_20pp_pairs)}/{int(z.pairs_all)} all pairs ({100*z.loss_above_20pp_fraction_all:.2f}%). "
                     f"Severe cases span {int(z.severe_distinct_partner1)} partner-1 and {int(z.severe_distinct_partner2)} partner-2 molecules.")
    lines += ["", "# Chemical interpretation with denominators", "",
              f"Substantial yield is available in many severe-loss cases: {int(g.severe_with_full_yield_above_50pct)}/{int(g.loss_above_20pp_pairs)} "
              "losses above 20 points have a full-screen yield above 50%. "
              f"Nevertheless {int(g.severe_with_panel_yield_at_least_20pct)}/{int(g.loss_above_20pp_pairs)} still reach at least 20% within the kit, so most severe deficits "
              "are missed yield improvement, not loss of the 20% hit itself. "
              f"There are {int(g.full_above_50pct_panel_below_20pct)} pairs where full yield exceeds 50% but the panel remains below 20%.", "",
              "The following labels illustrate why common tags or repeated molecules must not be called a general failure mechanism:"]
    for screen, partner, tag in [("Buchwald-Hartwig|2", 1, "phenylamino"), ("Ullmann|1", 1, "phenol"), ("Ullmann|1", 1, "pyridine")]:
        z = fg[(fg.screen == screen) & (fg.partner == partner) & (fg.functional_group == tag)].iloc[0]
        lines.append(f"- {z.label}, partner {partner}, source tag '{tag}': {int(z.tagged_severe_pairs)}/{int(z.tagged_pairs_productive)} productive tagged pairs lose >20 points. "
                     f"The tag is present in {int(z.tagged_severe_pairs)}/{int(z.array_severe_pairs)} severe cases and "
                     f"{int(z.tagged_pairs_productive)}/{int(z.array_pairs_productive)} productive cases overall. "
                     f"The productive tagged pairs contain {int(z.tagged_distinct_molecules_productive)} distinct tagged molecules.")
    lines += ["", "These are overlapping, within-array descriptive tags. They do not establish chemical-class enrichment, independent molecule replication, "
              "or a universal motif that predicts failure. The high-yield deficit and array-specific risk are the defensible headline; individual FG associations are exploratory.",
              "", "# Topmean comparator", "",
              f"On the same productive pairs, topmean six has median loss {t.productive_loss_median_pp:.4f} points, P90 {t.productive_loss_p90_pp:.4f}, "
              f"and {int(t.loss_above_20pp_pairs)}/{int(t.pairs_productive)} ({100*t.loss_above_20pp_fraction_productive:.2f}%) losses above 20 points. "
              "This is a descriptive comparison using the same folds; tails are not independent experiments.", ""]
    return "\n".join(lines)


def main():
    OUT.mkdir(exist_ok=True)
    source_paths = [lib.DATASET / "arrays.csv", lib.DATASET / "conditions.csv", HERE / "lib.py"]
    source_hashes = {p.name: digest(p) for p in source_paths}
    a, _ = lib.load()
    pairs = score_pairs(a)
    ann, provenance = load_annotations(pairs[["screen", "subpair"]])
    pairs = pairs.merge(ann[["screen", "subpair", "sub_1_FG", "sub_2_FG"]], on=["screen", "subpair"], validate="one_to_one")
    write("pair_scores", pairs)
    summary = []
    groups = [("ALL", "All arrays", pairs)] + [(s, d.label.iloc[0], d) for s, d in pairs.groupby("screen")]
    for screen, label, d in groups:
        for method in ("greedy", "topmean"):
            summary.append({"screen": screen, "label": label, "method": method, **metrics(d, method)})
    summary = write("array_summary", summary)
    distribution = distributions(pairs)
    yield_bands(pairs)
    zero_yields(a, pairs)
    molecule_summaries(pairs)
    fg = fg_summaries(pairs)
    write("high_yield_misses", pairs[(pairs.full_best_yield > .50) & (pairs.greedy_best_yield < .20)])
    assert len(pairs) == 2357 and pairs.productive.sum() == 1642
    assert (pairs.greedy_loss_pp > 20).sum() == 255
    assert (pairs[pairs.full_best_yield == 0].greedy_loss_pp == 0).all()
    prior = OUT / "review_axes_pair_predictions.csv"
    prior_checked = 0
    assert prior.exists(), "run review_axes_gc.py first: its pair predictions are cross-checked here"
    if prior.exists():
        old = pd.read_csv(prior)
        for method in ("greedy", "topmean"):
            o = old[old.method == method + "6"]
            if not len(o):
                continue
            q = pairs.merge(o[["screen", "subpair", "best_yield", "full_best_yield"]], on=["screen", "subpair"], suffixes=("", "_prior"), validate="one_to_one")
            assert len(q) == len(pairs)
            assert np.allclose(q[method + "_best_yield"], q.best_yield, rtol=0, atol=1e-12)
            assert np.allclose(q.full_best_yield, q.full_best_yield_prior, rtol=0, atol=1e-12)
            prior_checked += len(q)
    assert prior_checked == 2 * len(pairs), "the cross-check against review_axes_pair_predictions.csv matched %d rows" % prior_checked
    assert source_hashes == {p.name: digest(p) for p in source_paths}
    manifest = {"fold_seed": SEED, "folds": NFOLD, "panel_size": K, "numpy": np.__version__, "pandas": pd.__version__,
                "script_sha256": digest(Path(__file__)), "read_only_inputs_sha256": source_hashes,
                "annotations": provenance, "checks": {"threshold_denominator_self_test": "pass", "unique_pair_keys": len(pairs),
                "all_pairs_annotated": True, "loss_nonnegative": True, "established_oof_predictions_matched": prior_checked,
                "read_only_inputs_unchanged": True}}
    (OUT / "review_loss_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (OUT / "review_loss_methods.md").write_text(METHODS, encoding="utf-8")
    (OUT / "review_loss_findings.md").write_text(findings_text(summary, distribution, fg), encoding="utf-8")
    print(distribution.to_string(index=False), flush=True)
    print("Loss audit checks passed; outputs out/review_loss*", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test()
    if not args.self_test:
        main()
