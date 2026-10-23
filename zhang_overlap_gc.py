"""Recompute overlap with the previously published Zhang amide collection.

The reference file is pinned by ``data/public_reference/manifest.json`` (URL, SHA-256, size). It is
published by Zhang et al. and is not redistributed here: download it into ``data/public_reference/``
as ``amide_coupling.csv``, or point ``ZHANG_REFERENCE`` at a copy. This script reads only this
release's arrays.csv and that pinned input; it does not import a companion-project analysis.  Numeric condition labels are compared after
removing a leading ``C``.  A row match requires canonicalized reactants, the condition
number and a yield within 1e-8.  The stricter match additionally requires the original
plate/condition label. The legacy same_measurement_id fields denote label equality;
cross-version identity of physical measurements is not established, and archive is
not included in this comparison key.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" if (ROOT / "data").is_dir() else ROOT / "02_CODE_AND_DATA" / "data"
OUT = ROOT / "out"
MANIFEST = DATA / "public_reference" / "manifest.json"
REFERENCE = Path(os.environ.get("ZHANG_REFERENCE", DATA / "public_reference" / "amide_coupling.csv"))
ATOL = 1e-8


def canonicalize(series: pd.Series) -> pd.Series:
    cache = {}
    for value in series.dropna().astype(str).unique():
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"RDKit could not parse released SMILES: {value}")
        cache[value] = Chem.MolToSmiles(molecule, isomericSmiles=True)
    return series.astype(str).map(cache)


def condition_number(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace("C", "", n=1, regex=False).astype(int)


def audit_role(current: pd.DataFrame, public: pd.DataFrame, role: str) -> tuple[dict, pd.DataFrame]:
    current = current.reset_index(drop=True).rename_axis("current_row").reset_index()
    current = current.rename(columns={"yield": "current_yield"})
    public = public.copy().rename(columns={"yield": "public_yield"})
    public["public_plate"] = public["name0"].astype(str).str.replace(r"-C\d+$", "", regex=True)
    merged = current.merge(
        public[
            [
                "a",
                "b",
                "condition_number",
                "public_yield",
                "public_plate",
                "name0",
                "archive",
            ]
        ],
        on=["a", "b", "condition_number"],
        how="inner",
    )
    merged["equal_yield"] = np.isclose(
        merged["current_yield"], merged["public_yield"], rtol=0, atol=ATOL
    )
    merged["same_measurement_id"] = merged["plate"].eq(merged["public_plate"])
    equal = merged[merged["equal_yield"]]
    strict = merged[merged["same_measurement_id"] & merged["equal_yield"]]
    pairs = current.groupby(["a", "b"]).ngroups
    summary = {
        "role": role,
        "current_measurements": int(len(current)),
        "current_pairs": int(pairs),
        "current_rows_with_public_key": int(merged["current_row"].nunique()),
        "current_rows_equal_yield": int(equal["current_row"].nunique()),
        "pairs_with_equal_yield_row": int(equal.groupby(["a", "b"]).ngroups),
        "fraction_equal_yield": float(equal["current_row"].nunique() / len(current)),
        "same_measurement_id": int(merged.loc[merged["same_measurement_id"], "current_row"].nunique()),
        "same_measurement_id_and_yield": int(strict["current_row"].nunique()),
        "fraction_same_measurement_id_and_yield": float(strict["current_row"].nunique() / len(current)),
        "same_measurement_id_different_yield": int(
            merged.loc[merged["same_measurement_id"] & ~merged["equal_yield"], "current_row"].nunique()
        ),
    }
    merged["role"] = role
    return summary, merged


def summarize_yield_discrepancies(matches: pd.DataFrame) -> dict:
    differences = matches.loc[
        matches["role"].eq("analysis")
        & matches["same_measurement_id"]
        & ~matches["equal_yield"]
    ].copy()
    ratios = differences.loc[differences["public_yield"].ne(0)].copy()
    ratios["current_to_public_ratio"] = ratios["current_yield"] / ratios["public_yield"]
    by_plate = []
    for plate, rows in differences.groupby("plate", sort=True):
        plate_ratios = ratios.loc[ratios["plate"].eq(plate), "current_to_public_ratio"]
        record_count = int(rows["current_row"].nunique())
        detail = {
            "plate": str(plate),
            "analysis_records": record_count,
            "analysis_pairs": int(rows.groupby(["a", "b"]).ngroups),
            "nonzero_public_records": int(
                rows.loc[rows["public_yield"].ne(0), "current_row"].nunique()
            ),
            "current_to_public_ratio_min": float(plate_ratios.min()) if len(plate_ratios) else None,
            "current_to_public_ratio_max": float(plate_ratios.max()) if len(plate_ratios) else None,
        }
        if record_count == 1:
            singleton = rows.sort_values(["current_row", "public_yield", "name0", "archive"]).iloc[0]
            detail["single_record"] = {
                "measurement_id": str(singleton["name0"]),
                "condition_id": str(singleton["condition_id"]),
                "current_yield": float(singleton["current_yield"]),
                "public_yield": float(singleton["public_yield"]),
            }
        by_plate.append(detail)
    return {
        "role": "analysis",
        "analysis_records": int(differences["current_row"].nunique()),
        "analysis_pairs": int(differences.groupby(["a", "b"]).ngroups),
        "current_to_public_ratio_min": float(ratios["current_to_public_ratio"].min()) if len(ratios) else None,
        "current_to_public_ratio_max": float(ratios["current_to_public_ratio"].max()) if len(ratios) else None,
        "by_plate": by_plate,
    }


def measurement_version_differences(current: pd.DataFrame, public: pd.DataFrame) -> pd.DataFrame:
    """Compare reconstructed labels before structures; labels do not prove physical identity."""
    left = current.loc[current["set"].eq("analysis")].copy()
    right = public.copy()
    left["measurement_id"] = left["plate"] + "-" + left["condition_id"]
    right["plate"] = right["name0"].str.replace(r"-C\d+$", "", regex=True)
    left["condition_number"] = condition_number(left["condition_id"])
    right["condition_number"] = condition_number(right["condition_id"])
    columns = ["plate", "condition_number", "sub_1_smiles", "sub_2_smiles", "yield"]
    joined = left[columns + ["measurement_id"]].merge(
        right[columns + ["name0", "archive"]], on=["plate", "condition_number"],
        suffixes=("_current", "_public"), validate="one_to_one")
    same_identity = np.ones(len(joined), dtype=bool)
    same_connectivity = np.ones(len(joined), dtype=bool)
    for role in (1, 2):
        a = canonicalize(joined[f"sub_{role}_smiles_current"])
        b = canonicalize(joined[f"sub_{role}_smiles_public"])
        same_identity &= a.eq(b).to_numpy()
        connectivity = lambda s: s.map(lambda v: Chem.MolToSmiles(Chem.MolFromSmiles(v), isomericSmiles=False))
        same_connectivity &= connectivity(a).eq(connectivity(b)).to_numpy()
    equal_yield = np.isclose(joined["yield_current"], joined["yield_public"], rtol=0, atol=ATOL)
    joined["comparison"] = np.where(~same_connectivity, "reactant_identity_conflict",
                                    np.where(same_identity & ~equal_yield,
                                             "same_identity_yield_difference", "other"))
    joined["equal_yield"] = equal_yield
    return joined.loc[joined["comparison"].ne("other")].sort_values(
        ["comparison", "plate", "condition_number"]).reset_index(drop=True)


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not REFERENCE.is_file():
        raise SystemExit("The published Zhang collection is not redistributed. Download %s\n"
                         "to %s (or set ZHANG_REFERENCE) and rerun." % (manifest["url"], REFERENCE))
    digest = hashlib.sha256(REFERENCE.read_bytes()).hexdigest()
    if digest != manifest["sha256"]:
        raise AssertionError(f"Zhang reference SHA-256 mismatch: {digest}")
    if REFERENCE.stat().st_size != int(manifest["bytes"]):
        raise AssertionError("Zhang reference byte count mismatch")

    public = pd.read_csv(REFERENCE, low_memory=False)
    arrays = pd.read_csv(DATA / "arrays.csv", low_memory=False)
    current = arrays[arrays["array_id"].eq("Acylation|1")].copy()
    if current.empty:
        raise AssertionError("Acylation|1 is absent from arrays.csv")

    smiles = set(public["sub_1_smiles"]) | set(public["sub_2_smiles"])
    smiles |= set(current["sub_1_smiles"]) | set(current["sub_2_smiles"])
    canonical = {}
    for value in smiles:
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"RDKit could not parse SMILES: {value}")
        canonical[value] = Chem.MolToSmiles(molecule, isomericSmiles=True)

    public = public.copy()
    public["a"] = public["sub_1_smiles"].map(canonical)
    public["b"] = public["sub_2_smiles"].map(canonical)
    public["condition_number"] = condition_number(public["condition_id"])
    summaries = []
    matches = []
    subsets = {}
    for role in ("analysis", "reserve"):
        subset = current[current["set"].eq(role)].copy()
        subset["a"] = subset["sub_1_smiles"].map(canonical)
        subset["b"] = subset["sub_2_smiles"].map(canonical)
        subset["condition_number"] = condition_number(subset["condition_id"])
        summary, detail = audit_role(subset, public, role)
        summaries.append(summary)
        matches.append(detail)
        subsets[role] = subset

    combined = pd.concat([subsets["analysis"], subsets["reserve"]], ignore_index=True)
    all_summary, all_detail = audit_role(combined, public, "all")
    summaries.append(all_summary)
    matches.append(all_detail)
    all_matches = pd.concat(matches, ignore_index=True)
    version_differences = measurement_version_differences(current, public)
    identity_conflicts = version_differences.loc[
        version_differences["comparison"].eq("reactant_identity_conflict")]

    OUT.mkdir(parents=True, exist_ok=True)
    result = {
        "public_source": manifest,
        "public_rows": int(len(public)),
        "matching_yield_tolerance": ATOL,
        "summary": summaries,
        "yield_discrepancies": summarize_yield_discrepancies(all_matches),
        "measurement_identity_conflicts": {
            "role": "analysis",
            "records": int(len(identity_conflicts)),
            "by_plate": identity_conflicts.groupby("plate").size().to_dict(),
            "comparison_file": "data/measurement_version_differences.csv",
            "authority": "Unresolved; no structure or yield version is selected by this comparison.",
        },
        "limitations": [
            "Numeric condition IDs are matched after removing the leading C; this is not an independent chemical-composition crosswalk.",
            "Same-yield matches count unique current rows, not join multiplicity.",
            "Unmatched rows are not proven unpublished.",
            "Prior public exposure does not imply inclusion in model training.",
            "Connectivity conflicts exclude differences confined to explicit stereochemistry. The released arrays do not include product SMILES.",
            "The measurement_id label reconstructs plate plus condition and omits archive. Matching labels do not establish the same physical experiment across versions.",
        ],
    }
    (OUT / "zhang_overlap_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    pd.DataFrame(summaries).to_csv(OUT / "zhang_overlap_summary.csv", index=False)
    all_matches.to_csv(OUT / "zhang_overlap_matches.csv", index=False)
    version_differences.to_csv(DATA / "measurement_version_differences.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
