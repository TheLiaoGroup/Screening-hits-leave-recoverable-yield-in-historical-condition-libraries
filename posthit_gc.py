# -*- coding: utf-8 -*-
"""Residual yield after a screening hit, on the analysed arrays and the existing held-out pairs.

Cohort: pairs whose greedy six-condition out-of-fold maximum (m6) reaches 0.20. Reference: the
best measured yield of the same pair over its array's full library (b). Severe shortfall:
b - mk > 0.20 at budget k. Gap recovery from six to k: sum(mk - m6) / sum(b - m6) over the cohort.
The greedy panels are nested, so k = 12 adds six conditions to the k = 6 panel.

Comparators and audits:
  * random extension: exact expectation of max(m6, best of six conditions drawn uniformly from the
    pair's measured library conditions outside its greedy six), from order statistics;
  * support: how many library conditions exceed m6 by more than 0.20 for each severe shortfall;
  * yield bands of m6, with the ceiling above m6 >= 0.80 marked;
  * the post-hit slice of the registered never-read transported block;
  * the same endpoint on the 741 sealed/reserve pairs (panels fitted on analysed pairs only).
Intervals: role-merged molecule-node exponential reweighting, and whole-array resampling, both
conditional on the fixed out-of-fold outcomes.

Reads out/review_curve24_pair_scores.csv, out/review_loss_pair_scores.csv,
out/review_generalization_heldout_pair_scores.csv, the released arrays.csv and
06_CORPUS_HOLDOUT/transported_pair_scores.csv. Writes out/posthit_*.
Also writes out/measurement_coverage.json from the scorable analysis-pair design grid.
Use --coverage-only to rebuild that fact and Table 19 from the existing pair scores.
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

import lib

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
HIT = 0.20
SEVERE = 0.20
HIT_THRESHOLDS = (0.10, 0.20, 0.30)
# A shortfall above SEVERE cannot occur once m6 >= CEILING, because no yield exceeds 1.
CEILING = 1 - SEVERE
GAIN = 0.10
K_VALUES = (6, 12, 24)
BANDS = (0.20, 0.40, 0.60, 0.80, 1.0000001)
N_BOOT = 10_000
SEED = 20261006


def load() -> pd.DataFrame:
    d = pd.read_csv(OUT / "review_curve24_pair_scores.csv", low_memory=False)
    w = d.pivot_table(index=["screen", "subpair"], columns="k", values=["full", "greedy"]).reset_index()
    w.columns = ["%s_%s" % (a, b) if b != "" else a for a, b in w.columns]
    fulls = [c for c in w.columns if c.startswith("full_")]
    assert (w[fulls].max(1) - w[fulls].min(1)).max() < 1e-9, "full-library best varies with k"
    w["b"] = w["full_6"]
    for k in K_VALUES:
        w["m%d" % k] = w["greedy_%d" % k]
        w["L%d" % k] = w.b - w["m%d" % k]
    for k in (12, 24):
        assert (w["m%d" % k] >= w.m6 - 1e-10).all(), "greedy panels are not nested at k=%d" % k
    assert (w.b <= 1 + 1e-10).all(), "yield scale exceeds 0-1"
    meta = pd.read_csv(OUT / "review_loss_pair_scores.csv", low_memory=False,
                       usecols=["screen", "subpair", "sub_1_smiles", "sub_2_smiles", "greedy_panel",
                                "greedy_best_yield", "full_best_yield", "fold"])
    w = w.merge(meta, on=["screen", "subpair"], how="left", validate="1:1", indicator=True)
    assert w["_merge"].eq("both").all(), "pair metadata missing"
    assert len(w) == 2357, len(w)
    return w.drop(columns="_merge")


def summarize(d: pd.DataFrame, threshold: float = HIT) -> dict:
    c = d[d.m6 >= threshold]
    out = {"hit_threshold": threshold, "pairs": int(len(c))}
    for k in K_VALUES:
        out["severe_k%d" % k] = int((c["L%d" % k] > SEVERE).sum())
        out["severe_rate_k%d" % k] = float((c["L%d" % k] > SEVERE).mean())
    out["gap_points_k6"] = float(100 * c.L6.sum())
    for k in (12, 24):
        gain = c["m%d" % k] - c.m6
        out["gain_points_k6_to_%d" % k] = float(100 * gain.sum())
        out["gap_recovery_k6_to_%d" % k] = float(gain.sum() / c.L6.sum())
        out["severe_relief_k6_to_%d" % k] = int(((c.L6 > SEVERE) & (c["L%d" % k] <= SEVERE)).sum())
    return out


def per_array(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for screen, g in d.groupby("screen", sort=True):
        c = g[g.m6 >= HIT]
        o = c[c.m6 < CEILING]
        row = {"screen": screen, "pairs_all": int(len(g)), "pairs_hit6": int(len(c)),
               "pairs_hit6_below_ceiling": int(len(o)),
               "severe_rate_below_ceiling_k6": float((o.L6 > SEVERE).mean()) if len(o) else np.nan,
               "conditions": int(lib_conditions().get(screen, 0))}
        for k in K_VALUES:
            row["severe_k%d" % k] = int((c["L%d" % k] > SEVERE).sum())
            row["severe_rate_k%d" % k] = float((c["L%d" % k] > SEVERE).mean()) if len(c) else np.nan
        row["gap_points_k6"] = float(100 * c.L6.sum())
        for k in (12, 24):
            row["gain_points_k6_to_%d" % k] = float(100 * (c["m%d" % k] - c.m6).sum())
            row["gap_recovery_k6_to_%d" % k] = float((c["m%d" % k] - c.m6).sum() / c.L6.sum()) if c.L6.sum() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


_COND = {}


def lib_conditions() -> dict:
    if not _COND:
        su = pd.read_csv(OUT / "kit_summary.csv")
        _COND.update(dict(zip(su.screen, su.conditions)))
    return _COND


def node_index(d: pd.DataFrame):
    mols = sorted(set(d.sub_1_smiles) | set(d.sub_2_smiles))
    ix = {m: i for i, m in enumerate(mols)}
    return np.asarray([[ix[a], ix[b]] for a, b in zip(d.sub_1_smiles, d.sub_2_smiles)], int), len(mols)


def node_bootstrap(d: pd.DataFrame, extra: dict | None = None, threshold: float = HIT, seed_offset: int = 0) -> dict:
    """Role-merged molecule-node exponential reweighting; a pair's weight is the product of its two
    molecules' weights (a single weight when the identities coincide)."""
    c = d[d.m6 >= threshold].reset_index(drop=True)
    nodes, n_mol = node_index(c)
    same = nodes[:, 0] == nodes[:, 1]
    rng = np.random.RandomState(SEED + seed_offset + int(round(100 * threshold)))
    L = {k: c["L%d" % k].to_numpy(float) for k in K_VALUES}
    G = {k: (c["m%d" % k] - c.m6).to_numpy(float) for k in (12, 24)}
    X = {name: v[c.index] if len(v) == len(d) else v for name, v in (extra or {}).items()}
    draws = {}
    for _ in range(N_BOOT):
        w = rng.exponential(size=n_mol)
        pw = np.where(same, w[nodes[:, 0]], w[nodes[:, 0]] * w[nodes[:, 1]])
        for k in K_VALUES:
            draws.setdefault("severe_rate_k%d" % k, []).append((pw * (L[k] > SEVERE)).sum() / pw.sum())
        den = (pw * L[6]).sum()
        for k in (12, 24):
            draws.setdefault("gap_recovery_k6_to_%d" % k, []).append((pw * G[k]).sum() / den)
        for name, v in X.items():
            draws.setdefault(name, []).append((pw * v).sum() / den)
    out = {"hit_threshold": threshold, "pairs": int(len(c)), "molecules": n_mol, "draws": N_BOOT}
    for name, v in draws.items():
        out[name + "_lo"] = float(np.percentile(v, 2.5))
        out[name + "_hi"] = float(np.percentile(v, 97.5))
    return out


def array_bootstrap(d: pd.DataFrame) -> dict:
    a = per_array(d)
    rng = np.random.RandomState(SEED + 500)
    draws = {"severe_rate_k6": [], "severe_rate_k12": [], "gap_recovery_k6_to_12": []}
    for _ in range(N_BOOT):
        b = a.iloc[rng.randint(0, len(a), len(a))]
        draws["severe_rate_k6"].append(b.severe_k6.sum() / b.pairs_hit6.sum())
        draws["severe_rate_k12"].append(b.severe_k12.sum() / b.pairs_hit6.sum())
        draws["gap_recovery_k6_to_12"].append(b.gain_points_k6_to_12.sum() / b.gap_points_k6.sum())
    out = {"arrays": int(len(a)), "draws": N_BOOT, "resample_unit": "whole array"}
    for name, v in draws.items():
        out[name + "_lo"] = float(np.percentile(v, 2.5))
        out[name + "_hi"] = float(np.percentile(v, 97.5))
    return out


def condition_number(condition_id: str) -> tuple:
    """Natural order of condition identifiers such as C7 < C13 or A07 < B01."""
    prefix = condition_id.rstrip("0123456789")
    return prefix, int(condition_id[len(prefix):])


def p_gain(rest: np.ndarray, m6: float, k: int) -> float:
    """Exact probability that k conditions drawn uniformly from `rest` raise m6 by at least GAIN."""
    n = len(rest)
    if n == 0 or k == 0:
        return 0.0
    above = int((rest >= m6 + GAIN - 1e-12).sum())
    return 1 - comb(n - above, k) / comb(n, k)


def library_wells(d: pd.DataFrame) -> pd.DataFrame:
    a, _ = lib.load("analysis")
    summary = pd.read_csv(lib.DATASET / "arrays_summary.csv")
    if len(a) != int(summary.measurements.sum()):
        raise AssertionError("analysis rows %d != %d in arrays_summary.csv" % (len(a), int(summary.measurements.sum())))
    groups = a.groupby(["screen", "subpair"])
    c = d[d.m6 >= HIT]
    assert np.allclose(c.m6, c.greedy_best_yield) and np.allclose(c.b, c.full_best_yield), \
        "pair maxima disagree between review_curve24 and review_loss outputs"
    # Mean-yield extension: the six highest training-mean conditions outside the pair's greedy six,
    # training means over measured wells of the other folds' pairs of the same array,
    # as for the mean-yield baseline.
    assert d.fold.notna().all(), "fold assignment missing"
    means = {}
    for screen, g in d.groupby("screen"):
        M = a[a.screen == screen].pivot_table(index="subpair", columns="condition_id", values="yield", aggfunc="max")
        for f, gf in g.groupby("fold"):
            means[(screen, f)] = M.drop(index=gf.subpair).mean().sort_values(ascending=False, kind="stable")
    rows = []
    for r in c.itertuples():
        w = groups.get_group((r.screen, r.subpair))
        y = w.groupby("condition_id")["yield"].max()
        assert abs(y.max() - r.b) < 1e-9, "library maximum does not reproduce b"
        panel = r.greedy_panel.split(";")
        add6 = [k_ for k_ in means[(r.screen, r.fold)].index if k_ not in panel][:6]
        got = y.reindex(add6).dropna()
        mmy = max(r.m6, float(got.max())) if len(got) else r.m6
        rest = np.sort(y[~y.index.isin(panel)].to_numpy(float))
        n, k = len(rest), min(6, len(rest))
        e = r.m6
        if n:
            e = sum(max(rest[i], r.m6) * comb(i, k - 1) for i in range(k - 1, n)) / comb(n, k)
        top = w.loc[w["yield"] == y.max()]
        # The hit recipe is the earliest-chosen panel member at m6; the best recipe is the lowest-numbered
        # condition at b.
        hit_condition = y.reindex(panel).idxmax()
        best_condition = min(y.index[np.isclose(y, y.max(), rtol=0, atol=1e-12)], key=condition_number)
        rows.append({"screen": r.screen, "subpair": r.subpair, "b": r.b, "m6": r.m6, "m12": r.m12,
                     "m6_random_plus6": e, "m6_meanyield_plus6": mmy,
                     "p_random_gain10": p_gain(rest, r.m6, k),
                     "hit_condition": hit_condition, "best_condition": best_condition,
                     "conditions_beating_m6_by_20": int((y > r.m6 + SEVERE).sum()),
                     "conditions_within_5_of_best": int((y >= r.b - 0.05).sum()),
                     "best_well_check_passed": bool(top.check_passed.all())})
    return pd.DataFrame(rows)


def extension_and_support(lw: pd.DataFrame) -> dict:
    gap = (lw.b - lw.m6).sum()
    sev = lw[lw.b - lw.m6 > SEVERE]
    n = sev.conditions_beating_m6_by_20
    gain = 100 * (lw.m12 - lw.m6)
    sgain = 100 * (sev.m12 - sev.m6)
    return {"pairs": int(len(lw)),
            "gap_recovery_greedy_k6_to_12": float((lw.m12 - lw.m6).sum() / gap),
            "gap_recovery_random_plus6": float((lw.m6_random_plus6 - lw.m6).sum() / gap),
            "gap_recovery_meanyield_plus6": float((lw.m6_meanyield_plus6 - lw.m6).sum() / gap),
            "hit_gap_points_mean": float(100 * (lw.b - lw.m6).mean()),
            "hit_gain_points_mean": float(gain.mean()), "hit_gain_points_median": float(gain.median()),
            "hits_with_positive_gain": int((gain > 1e-9).sum()),
            "severe_gain_points_mean": float(sgain.mean()), "severe_gain_points_median": float(sgain.median()),
            "severe_k6": int(len(sev)),
            "severe_supported_by_one_condition": int((n == 1).sum()),
            "severe_supported_by_two_or_more": int((n >= 2).sum()),
            "severe_supported_by_three_or_more": int((n >= 3).sum()),
            "severe_rate_two_condition_floor": float((n >= 2).sum() / len(lw)),
            "severe_best_well_check_passed": int(sev.best_well_check_passed.sum()),
            "severe_gap_points_median": float(100 * (sev.b - sev.m6).median()),
            "severe_gap_points_mean": float(100 * (sev.b - sev.m6).mean())}


def below_ceiling(d: pd.DataFrame, lw: pd.DataFrame) -> dict:
    """Hits below m6 = CEILING, the only hits that can fall more than SEVERE short, and how often six
    further conditions raise them by at least GAIN (greedy, historical-mean, exact random expectation)."""
    o = d[(d.m6 >= HIT) & (d.m6 < CEILING)]
    w = lw[lw.m6 < CEILING]
    assert len(o) == len(w) and int((o.L6 > SEVERE).sum()) == int((w.b - w.m6 > SEVERE).sum())
    out = {"ceiling": CEILING, "gain_threshold": GAIN, "pairs": int(len(o))}
    for k in K_VALUES:
        out["severe_k%d" % k] = int((o["L%d" % k] > SEVERE).sum())
        out["severe_rate_k%d" % k] = float((o["L%d" % k] > SEVERE).mean())
    out["gain10_greedy"] = int((w.m12 - w.m6 >= GAIN - 1e-12).sum())
    out["gain10_greedy_rate"] = float((w.m12 - w.m6 >= GAIN - 1e-12).mean())
    out["gain10_meanyield_rate"] = float((w.m6_meanyield_plus6 - w.m6 >= GAIN - 1e-12).mean())
    out["gain10_random_rate"] = float(w.p_random_gain10.mean())
    out["gap_recovery_k6_to_12"] = float((w.m12 - w.m6).sum() / (w.b - w.m6).sum())
    gain = 100 * (w.m12 - w.m6)
    out["gain_points_mean"] = float(gain.mean())
    out["gain_points_median"] = float(gain.median())
    out["hits_with_positive_gain"] = int((gain > 1e-9).sum())
    nb = node_bootstrap(o.reset_index(drop=True), seed_offset=700)
    for k in K_VALUES:
        for end in ("lo", "hi"):
            out["severe_rate_k%d_%s" % (k, end)] = nb["severe_rate_k%d_%s" % (k, end)]
    return out


def gap_recipes(lw: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """For each severe shortfall, which condition variables differ between the hit recipe and the best
    recipe, against two references: the share of the pair's other measured conditions that differ from
    the hit recipe, and the share of the array's productive pairs whose own library-best recipe differs
    from it (where each library's best yields fall, whatever the pair)."""
    a, C = lib.load("analysis")
    sev = lw[lw.b - lw.m6 > SEVERE]
    rows = []
    for screen, g in sev.groupby("screen", sort=True):
        d = a[a.screen == screen]
        M = lib.screen_matrix(a, screen)
        V = lib.condition_variables(screen, C, d.rxn_class.iloc[0], list(M.columns))
        productive = M[M.max(axis=1) > 0]
        library_best = [min(row.index[np.isclose(row, row.max(), rtol=0, atol=1e-12)], key=condition_number)
                        for _, row in productive.iterrows()]
        for r in g.itertuples():
            y = M.loc[r.subpair].dropna()
            others = y.index[y.index != r.hit_condition]
            for v in V.columns:
                rows.append({"screen": screen, "subpair": r.subpair, "variable": v,
                             "hit_condition": r.hit_condition, "best_condition": r.best_condition,
                             "hit_level": V.at[r.hit_condition, v], "best_level": V.at[r.best_condition, v],
                             "changed": bool(V.at[r.hit_condition, v] != V.at[r.best_condition, v]),
                             "changed_if_random": float((V.loc[others, v] != V.at[r.hit_condition, v]).mean()),
                             "changed_if_library_best": float((V.loc[library_best, v] != V.at[r.hit_condition, v]).mean())})
    T = pd.DataFrame(rows)
    out = {"severe_cases": int(len(sev)),
           "distinct_best_recipes": int(sev[["screen", "best_condition"]].drop_duplicates().shape[0])}
    for v, g in T.groupby("variable"):
        out[v] = {"cases": int(len(g)), "changed": int(g.changed.sum()),
                  "changed_if_random": float(g.changed_if_random.sum()),
                  "changed_if_library_best": float(g.changed_if_library_best.sum())}
    for screen, g in sev.groupby("screen"):
        out.setdefault("by_array", {})[screen] = {"severe": int(len(g)),
                                                   "distinct_best_recipes": int(g.best_condition.nunique())}
    return out, T


def yield_bands(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lo, hi in zip(BANDS[:-1], BANDS[1:]):
        c = d[(d.m6 >= lo) & (d.m6 < hi)]
        rows.append({"m6_lower": lo, "m6_upper": min(hi, 1.0), "pairs": int(len(c)),
                     "severe_k6": int((c.L6 > SEVERE).sum()), "severe_rate_k6": float((c.L6 > SEVERE).mean()),
                     "severe_k12": int((c.L12 > SEVERE).sum()), "severe_rate_k12": float((c.L12 > SEVERE).mean()),
                     "ceiling_permits_severe": bool(lo < 1 - SEVERE)})
    return pd.DataFrame(rows)


def transported_slice() -> dict:
    t = pd.read_csv(HERE / "06_CORPUS_HOLDOUT" / "transported_pair_scores.csv")
    assert len(t) == 44, len(t)
    pos, hit = t[t.array_best > 0], t[t.best_of_6 >= HIT]
    return {"pairs": int(len(t)), "productive": int(len(pos)),
            "severe_among_productive": int((pos.loss_6 > SEVERE).sum()),
            "hit6": int(len(hit)), "severe_among_hit6": int((hit.loss_6 > SEVERE).sum())}


def holdout_extensions() -> pd.DataFrame:
    """Score the six-well extensions on the frozen held-out pairs.

    The held-out panels were fitted on all analysis pairs of each array.  The
    mean-yield extension takes the six highest training means outside that
    fitted greedy six; the random extension is the exact order-statistic
    expectation over the measured held-out wells outside the same six.
    """
    raw = pd.read_csv(lib.DATASET / "arrays.csv", low_memory=False)
    raw = raw[(raw["set"] != "replicate") & raw["measurement_status"].isin(["quantified", "zero_by_rule"])].copy()
    raw["screen"] = raw["array_id"]
    raw["subpair"] = raw["sub_1_smiles"] + " >> " + raw["sub_2_smiles"]
    analysis = raw[raw["set"] == "analysis"]
    held = raw[raw["set"].isin(["sealed", "reserve"])]
    pair_sets = held.drop_duplicates(["screen", "subpair"]).set_index(["screen", "subpair"])["set"].to_dict()
    rows = []
    for screen in sorted(analysis.screen.unique()):
        train = analysis[analysis.screen == screen]
        M = train.pivot_table(index="subpair", columns="condition_id", values="yield", aggfunc="mean")
        H = held[held.screen == screen].pivot_table(index="subpair", columns="condition_id",
                                                     values="yield", aggfunc="mean").reindex(columns=M.columns)
        Y = M.to_numpy(float)
        kit = lib.greedy_kit(Y, 24)
        panel = kit[:6]
        means = np.nanmean(Y, axis=0)
        rank = np.argsort(-means, kind="stable")
        add = [j for j in rank if j not in panel][:6]
        for subpair, r in H.iterrows():
            y = r.to_numpy(float)
            finite = np.isfinite(y)
            full = float(np.nanmax(y)) if finite.any() else 0.0
            m6 = float(np.nanmax(y[panel])) if np.isfinite(y[panel]).any() else 0.0
            m12 = float(np.nanmax(y[kit[:12]])) if np.isfinite(y[kit[:12]]).any() else 0.0
            my = float(np.nanmax(y[panel + add])) if np.isfinite(y[panel + add]).any() else m6
            rest = np.sort(y[[j for j in range(len(y)) if j not in panel and np.isfinite(y[j])]])
            n, k = len(rest), min(6, len(rest))
            if n == 0 or k == 0:
                rnd = m6
            else:
                rnd = sum(max(rest[i], m6) * comb(i, k - 1) for i in range(k - 1, n)) / comb(n, k)
            set_name = pair_sets[(screen, subpair)]
            rows.append({"screen": screen, "set": set_name, "subpair": subpair,
                         "full": full, "m6": m6, "m12": m12,
                         "m6_meanyield_plus6": my, "m6_random_plus6": rnd,
                         "p_random_gain10": p_gain(rest, m6, k)})
    out = pd.DataFrame(rows)
    if out.duplicated(["screen", "set", "subpair"]).any():
        raise AssertionError("held-out extension rows are not unique")
    return out


def holdout() -> tuple[dict, pd.DataFrame]:
    raw = pd.read_csv(OUT / "review_generalization_heldout_pair_scores.csv", low_memory=False)
    d = raw[raw.k.isin([6, 12])]
    assert not d.duplicated(["screen", "set", "subpair", "k"]).any()
    p = d.pivot_table(index=["screen", "set", "subpair"], columns="k", values=["full", "greedy"],
                      aggfunc="first").reset_index()
    p.columns = ["%s_%s" % (a, b) if b != "" else a for a, b in p.columns]
    assert np.allclose(p.full_6, p.full_12)
    p["full"], p["m6"], p["m12"] = p.full_6, p.greedy_6, p.greedy_12
    assert len(p) == 741, len(p)
    ext = holdout_extensions()
    p = p.reset_index().merge(ext[["screen", "set", "subpair", "m6_meanyield_plus6", "m6_random_plus6",
                                   "p_random_gain10"]],
                               on=["screen", "set", "subpair"], how="left", validate="1:1")
    if p[["m6_meanyield_plus6", "m6_random_plus6"]].isna().any().any():
        raise AssertionError("held-out extension scores missing")

    def one(g):
        h = g[g.m6 >= HIT]
        gap = (h.full - h.m6).sum()
        o = h[h.m6 < CEILING]
        return {"pairs": int(len(g)), "hit6": int(len(h)),
                "hit6_below_ceiling": int(len(o)),
                "severe_below_ceiling_k6": int((o.full - o.m6 > SEVERE).sum()),
                "severe_rate_below_ceiling_k6": float((o.full - o.m6 > SEVERE).mean()),
                "severe_rate_below_ceiling_k12": float((o.full - o.m12 > SEVERE).mean()),
                "gain10_greedy_below_ceiling": float((o.m12 - o.m6 >= GAIN - 1e-12).mean()),
                "gain10_meanyield_below_ceiling": float((o.m6_meanyield_plus6 - o.m6 >= GAIN - 1e-12).mean()),
                "gain10_random_below_ceiling": float(o.p_random_gain10.mean()),
                "severe_k6": int((h.full - h.m6 > SEVERE).sum()),
                "severe_rate_k6": float((h.full - h.m6 > SEVERE).mean()),
                "severe_k12": int((h.full - h.m12 > SEVERE).sum()),
                "severe_rate_k12": float((h.full - h.m12 > SEVERE).mean()),
                "gap_recovery_k6_to_12": float((h.m12 - h.m6).sum() / gap) if gap else np.nan,
                "gap_recovery_meanyield_plus6": float((h.m6_meanyield_plus6 - h.m6).sum() / gap) if gap else np.nan,
                "gap_recovery_random_plus6": float((h.m6_random_plus6 - h.m6).sum() / gap) if gap else np.nan}
    res = {"all": one(p), **{s: one(g) for s, g in p.groupby("set")}}
    table = pd.DataFrame([{"screen": s, "set": t, **one(g)} for (s, t), g in p.groupby(["screen", "set"])])
    return res, table


def measurement_coverage(raw: pd.DataFrame | None = None) -> tuple[dict, pd.DataFrame]:
    """Measured share of the complete library grid of scorable analysis pairs.

    The grid includes absent and excluded wells. Pairs with no quantified or rule-zero
    cell cannot be scored and do not enter either count.
    """
    if raw is None:
        raw = pd.read_csv(lib.DATASET / "arrays.csv", low_memory=False)
    an = raw[raw["set"].eq("analysis")].copy()
    key = ["array_id", "sub_1_smiles", "sub_2_smiles", "condition_id"]
    assert not an.duplicated(key).any(), "duplicate analysis pair-condition cells"
    n_cond = an.groupby("array_id").condition_id.nunique()
    measured = an[an.measurement_status.isin(["quantified", "zero_by_rule"])]
    pairs = measured.groupby(key[:-1]).size().rename("measured_conditions").reset_index()
    pairs["conditions"] = pairs.array_id.map(n_cond)
    pairs["measured_share"] = pairs.measured_conditions / pairs.conditions
    assert pairs.measured_share.between(0, 1, inclusive="right").all()
    by_array = []
    for aid, g in pairs.groupby("array_id", sort=True):
        by_array.append({"array_id": aid, "scorable_pairs": int(len(g)),
                         "conditions": int(n_cond[aid]),
                         "measured_cells": int(g.measured_conditions.sum()),
                         "designed_cells": int(g.conditions.sum())})
    numerator = int(pairs.measured_conditions.sum())
    denominator = int(pairs.conditions.sum())
    assert numerator == len(measured) and denominator > 0
    pct = 100 * numerator / denominator
    facts = {"definition": "Scorable analysis pairs times their array's full condition library; measured cells are quantified plus zero_by_rule.",
             "scorable_pairs": int(len(pairs)), "measured_cells": numerator,
             "quantified_cells": int(measured.measurement_status.eq("quantified").sum()),
             "zero_by_rule_cells": int(measured.measurement_status.eq("zero_by_rule").sum()),
             "designed_cells": denominator, "unmeasured_cells": denominator - numerator,
             "measured_share": numerator / denominator, "measured_pct": pct,
             "reported_pct": round(pct, 1), "reported_share": round(round(pct, 1) / 100, 3),
             "by_array": by_array}
    pairs["screen"] = pairs.array_id
    pairs["subpair"] = pairs.sub_1_smiles + " >> " + pairs.sub_2_smiles
    return facts, pairs[["screen", "subpair", "measured_share"]]


def completeness_sensitivity(lw: pd.DataFrame,
                             coverage: tuple[dict, pd.DataFrame] | None = None) -> pd.DataFrame:
    """The severe-shortfall rate among hits below 80%, restricted to substrate pairs measured on at least a
    given share of their array's conditions (the arrays admit pairs measured on at least 70%)."""
    facts, share = coverage if coverage is not None else measurement_coverage()
    h = lw.merge(share, on=["screen", "subpair"], how="left", validate="1:1")
    assert h.measured_share.notna().all() and len(h) == len(lw)
    rows = []
    for t in (0.70, 0.80, 0.90, facts["reported_share"], 1.0):
        q = h[(h.measured_share >= t - 1e-12) & (h.m6 < CEILING)]
        severe = int((q.b - q.m6 > SEVERE).sum())
        rows.append({"min_measured_share": t, "hits": int((h.measured_share >= t - 1e-12).sum()),
                     "hits_below_ceiling": len(q), "severe": severe,
                     "severe_rate_below_ceiling": severe / len(q) if len(q) else np.nan})
    return pd.DataFrame(rows)


def write_coverage_outputs(lw: pd.DataFrame, scored_pairs: pd.DataFrame) -> dict:
    coverage = measurement_coverage()
    facts, pairs = coverage
    keys = ["screen", "subpair"]
    assert set(map(tuple, pairs[keys].to_numpy())) == set(map(tuple, scored_pairs[keys].to_numpy())), \
        "coverage grid and scored analysis pairs differ"
    completeness_sensitivity(lw, coverage).to_csv(OUT / "posthit_completeness_sensitivity.csv",
                                               index=False, lineterminator="\n")
    (OUT / "measurement_coverage.json").write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8")
    return facts


def main() -> None:
    d = load()
    base = summarize(d)
    lw = library_wells(d)
    ext = extension_and_support(lw)
    full = d.merge(lw[["screen", "subpair", "m6_random_plus6", "m6_meanyield_plus6"]], on=["screen", "subpair"], how="left")
    hit = full[full.m6 >= HIT].reset_index(drop=True)
    rand_gain = (hit.m6_random_plus6 - hit.m6).to_numpy(float)
    my_gain = (hit.m6_meanyield_plus6 - hit.m6).to_numpy(float)
    greedy_gain = (hit.m12 - hit.m6).to_numpy(float)
    nb = node_bootstrap(hit, {"random_recovery": rand_gain, "greedy_minus_random_recovery": greedy_gain - rand_gain,
                              "meanyield_recovery": my_gain, "meanyield_minus_random_recovery": my_gain - rand_gain,
                              "greedy_minus_meanyield_recovery": greedy_gain - my_gain})
    ab = array_bootstrap(d)
    thresholds = pd.DataFrame([summarize(d, t) for t in HIT_THRESHOLDS])
    bands = yield_bands(d)
    arrays = per_array(d)
    ho, ho_table = holdout()
    ts = transported_slice()
    bc = below_ceiling(d, lw)
    recipes, recipe_rows = gap_recipes(lw)
    summary = {"definitions": {"hit": "greedy six-condition out-of-fold maximum >= 0.20",
                               "reference": "best measured yield of the pair over its array's full library",
                               "severe": "reference minus panel best > 0.20",
                               "below_ceiling": "hits with m6 < 0.80, the only hits that can fall more than 0.20 short",
                               "gain10": "six further conditions raise the hit by at least 0.10",
                               "intervals": "molecule-node reweighting and whole-array resampling, conditional on fixed out-of-fold outcomes"},
               "analysed": base, "molecule_node_intervals": nb, "array_intervals": ab,
               "extension_and_support": ext, "below_ceiling": bc, "gap_recipes": recipes,
               "holdout": ho, "transported_block": ts}
    recipe_rows.to_csv(OUT / "posthit_gap_recipes.csv", index=False, lineterminator="\n")
    lw.to_csv(OUT / "posthit_pair_support.csv", index=False, lineterminator="\n")
    bands.to_csv(OUT / "posthit_yield_bands.csv", index=False, lineterminator="\n")
    arrays.to_csv(OUT / "posthit_by_array.csv", index=False, lineterminator="\n")
    thresholds.to_csv(OUT / "posthit_thresholds.csv", index=False, lineterminator="\n")
    ho_table.to_csv(OUT / "posthit_holdout_by_array.csv", index=False, lineterminator="\n")
    write_coverage_outputs(lw, d)
    (OUT / "posthit_summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(bands.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-only", action="store_true",
                        help="recompute the coverage fact and completeness sensitivity from existing pair scores")
    args = parser.parse_args()
    if args.coverage_only:
        facts = write_coverage_outputs(pd.read_csv(OUT / "posthit_pair_support.csv"), load())
        print(json.dumps(facts, indent=2))
    else:
        main()
