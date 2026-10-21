# -*- coding: utf-8 -*-
"""Residual yield after a screening hit in three public measured condition matrices.

Source: Sivilotti, Friday & Jackson, Digital Discovery 2025 (DOI 10.1039/D4DD00365A), code and data
archive Zenodo 14861626, file zenodo_14861626_condition_set_opt-main.zip (SHA-256
8CE12C7CC95A001691EBE465EA125B908F75CBAB429BFF7390C5FA0B9697B2AF). Point SIVILOTTI_DIR at the
archive's datasets/real_datasets folder, or unpack the archive into external/ beside this script.
Each matrix is scored separately with ten-fold pair-level greedy selection, the endpoint of
posthit_gc.py. Yields are published percentages divided by 100, unclipped. The archive's
Buchwald-Hartwig file is described by its authors as a model-completed space; it is scored for
reference and is not a measured matrix. Writes out/public_matrices_posthit.json and _pair_scores.csv.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
_CANDIDATES = [Path(os.environ["SIVILOTTI_DIR"])] if os.environ.get("SIVILOTTI_DIR") else []
_CANDIDATES += [HERE / "external" / "condition_set_opt-main" / "datasets" / "real_datasets",
                HERE / "04_WORK" / "NC_UPGRADE" / "external" / "condition_set_opt-main" / "datasets" / "real_datasets"]
DATA = next((p for p in _CANDIDATES if (p / "deoxyf.csv").exists()), None)
OUT = HERE / "out"
MEASURED = ("deoxyf", "pd_aryl", "ni_borylation")


HIT = 0.20
SEVERE = 0.20
SEED = 3
NFOLD = 10

DATASETS = {
    "deoxyf": {"file": "deoxyf.csv", "pairs": ["substrate_name"],
               "conditions": ["base_name", "fluoride_name"]},
    "pd_aryl": {"file": "aryl-scope-ligand.csv", "pairs": ["electrophile_id", "nucleophile_id"],
                "conditions": ["ligand_name"]},
    "ni_borylation": {"file": "borylation.csv", "pairs": ["electrophile_id"],
                      "conditions": ["ligand_name", "solvent"]},
    "buchwald_hartwig": {"file": "buchwald-hartwig.csv", "pairs": ["Amine", "Bromide"],
                          "conditions": ["Catalyst", "Solvent", "Base"]},
}


def matrix(name: str, spec: dict) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(DATA / spec["file"], low_memory=False)
    need = set(spec["pairs"] + spec["conditions"] + ["yield"])
    if not need.issubset(raw.columns):
        raise AssertionError("%s is missing columns %s" % (name, sorted(need - set(raw.columns))))
    raw = raw.dropna(subset=["yield"]).copy()
    raw["pair"] = raw[spec["pairs"]].astype(str).agg(" >> ".join, axis=1)
    raw["condition_id"] = raw[spec["conditions"]].astype(str).agg(" | ".join, axis=1)
    raw["yield01"] = pd.to_numeric(raw["yield"], errors="raise") / 100.0
    dup = int(raw.duplicated(["pair", "condition_id"]).sum())
    M = raw.pivot_table(index="pair", columns="condition_id", values="yield01", aggfunc="mean")
    if M.empty or M.shape[0] < NFOLD or M.shape[1] < 6:
        raise AssertionError("%s matrix is too small: %s" % (name, M.shape))
    meta = {"file": spec["file"], "raw_rows": int(len(raw)), "pairs": int(M.shape[0]),
            "conditions": int(M.shape[1]), "duplicate_pair_condition_rows": dup,
            "missing_cells": int(M.isna().sum().sum()), "cells": int(M.size),
            "yield_min_percent": float(raw["yield01"].min() * 100),
            "yield_max_percent": float(raw["yield01"].max() * 100),
            "yield_mean_percent": float(raw["yield01"].mean() * 100)}
    return M, meta


def greedy_kit(Y: np.ndarray, k: int) -> list[int]:
    Yf = np.where(np.isnan(Y), -np.inf, Y)
    best = np.full(Y.shape[0], -np.inf)
    available = np.ones(Y.shape[1], dtype=bool)
    chosen = []
    for _ in range(k):
        candidates = np.where(available)[0]
        if not len(candidates):
            break
        scores = []
        for c in candidates:
            candidate_best = np.maximum(best, Yf[:, c])
            scores.append(np.where(np.isfinite(candidate_best), candidate_best, 0.0).mean())
        c = int(candidates[int(np.argmax(scores))])
        chosen.append(c)
        best = np.maximum(best, Yf[:, c])
        available[c] = False
    return chosen


def best_of(Y: np.ndarray, kit: list[int]) -> np.ndarray:
    z = Y[:, kit]
    out = np.nanmax(z, axis=1)
    return np.where(np.isfinite(out), out, 0.0)


def score(M: pd.DataFrame, name: str) -> tuple[pd.DataFrame, dict]:
    Y = M.to_numpy(float)
    n, c = Y.shape
    ks = tuple(k for k in (6, 12, 24) if k <= c)
    rng = np.random.RandomState(SEED)
    folds = np.array_split(rng.permutation(n), NFOLD)
    full = np.nanmax(Y, axis=1)
    full = np.where(np.isfinite(full), full, 0.0)
    fold_id = np.full(n, -1, dtype=int)
    outputs = {"m%d" % k: np.full(n, np.nan) for k in ks}
    for fi, test in enumerate(folds):
        train = np.setdiff1d(np.arange(n), test)
        kit = greedy_kit(Y[train], max(ks))
        fold_id[test] = fi
        for k in ks:
            outputs["m%d" % k][test] = best_of(Y[test], kit[:k])
    out = pd.DataFrame({"dataset": name, "pair": M.index.astype(str), "fold": fold_id, "full": full})
    for k in ks:
        out["m%d" % k] = outputs["m%d" % k]
        out["L%d" % k] = out.full - out["m%d" % k]
    eligible = out[out.full >= HIT]
    hit6 = out[out.m6 >= HIT]
    summary = {"dataset": name, "pairs": n, "conditions": c, "k_values": list(ks),
               "eligible_full_ge_20": int(len(eligible)), "hit6": int(len(hit6)),
               "hit12": int((out.m12 >= HIT).sum()) if "m12" in out else None,
               "missing_fraction": float(M.isna().sum().sum() / M.size)}
    for k in ks:
        severe = hit6["L%d" % k] > SEVERE
        summary["severe_k%d_among_hit6" % k] = int(severe.sum())
        summary["severe_rate_k%d_among_hit6" % k] = float(severe.mean()) if len(hit6) else None
    for k in (12, 24):
        if "m%d" % k not in out:
            continue
        gap_e = eligible.L6.sum()
        gap_h = hit6.L6.sum()
        summary["gap_recovery_k6_to_%d_eligible" % k] = float((eligible["m%d" % k] - eligible.m6).sum() / gap_e) if gap_e else None
        summary["gap_recovery_k6_to_%d_hit6" % k] = float((hit6["m%d" % k] - hit6.m6).sum() / gap_h) if gap_h else None
        summary["gain_points_k6_to_%d_hit6" % k] = float(100 * (hit6["m%d" % k] - hit6.m6).sum())
    return out, summary


def main() -> None:
    if DATA is None:
        raise SystemExit("Sivilotti archive not found; set SIVILOTTI_DIR (see module docstring)")
    scores, summaries = [], []
    for name, spec in DATASETS.items():
        M, meta = matrix(name, spec)
        scored, result = score(M, name)
        summaries.append({**meta, **result, "measured": name in MEASURED})
        scores.append(scored)
    meas = [s for s in summaries if s["measured"]]
    pooled = {"hit6": sum(s["hit6"] for s in meas), "severe_k6_among_hit6": sum(s["severe_k6_among_hit6"] for s in meas)}
    pd.concat(scores, ignore_index=True).to_csv(OUT / "public_matrices_posthit_pair_scores.csv", index=False,
                                                lineterminator=chr(10))
    result = {"source": "Zenodo 14861626; Sivilotti, Friday & Jackson, Digital Discovery 2025, DOI 10.1039/D4DD00365A",
              "datasets": summaries, "measured_matrices_combined_count": pooled}
    (OUT / "public_matrices_posthit.json").write_text(json.dumps(result, indent=1) + chr(10), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
