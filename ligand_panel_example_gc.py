"""Deterministic, exhaustive recipe example for Supplementary Table 20.

Selection rules fixed before examining any individual draw's outcome:
1. Select the library with largest greedy6 minus ligand_uniform6 retention in
   review_axes_recipes.csv; break exact ties by the library's screen identifier.
2. Enumerate every unordered six-condition set whose six encoded ligand levels
   are distinct. Each legal condition set has weight one for this example's
   median, unlike the level-uniform random rule's method expectation.
3. Use the exact lower order median (numpy quantile method="lower") of the
   complete retention distribution. This is an attained sample median also
   when the number of sets is even. Record both central order statistics and
   their arithmetic mean. Break attaining-set ties by condition-ID order.
4. Score the selected fixed draw and the full-analysis history-ranked six
   recipes on the same measured library matrix, separately from CV estimates.

Missing measurements follow lib.retention; no candidate is imputed as measured.
No sampling or approximate quantile is permitted in this generator.
"""
import argparse
import hashlib
import itertools
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

import lib

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
K = 6
RESULT = OUT / "ligand_panel_example_gc.json"
RECIPES = OUT / "ligand_panel_example_recipes.csv"


def selected_library():
    scores = pd.read_csv(OUT / "review_axes_recipes.csv").pivot(
        index="screen", columns="method", values="retention")
    candidates = [str(s) for s in scores.index
                  if pd.notna(scores.loc[s, "ligand_uniform6"])]
    return min(candidates, key=lambda s: (
        -float(scores.loc[s, "greedy6"] - scores.loc[s, "ligand_uniform6"]), s))


def legal_count(level_sizes):
    counts = [1] + [0] * K
    for size in level_sizes:
        for k in range(K, 0, -1):
            counts[k] += int(size) * counts[k - 1]
    return counts[K]


def load_library():
    screen = selected_library()
    a, conditions = lib.load()
    rows = a[a.screen == screen]
    matrix = lib.screen_matrix(a, screen)
    variables = lib.condition_variables(
        screen, conditions, rows.rxn_class.iloc[0], list(matrix.columns))
    assert variables.ligand.nunique() >= K
    return screen, matrix, variables, conditions


def input_hashes():
    files = {"arrays.csv": lib.DATASET / "arrays.csv",
             "conditions.csv": lib.DATASET / "conditions.csv",
             "review_axes_recipes.csv": OUT / "review_axes_recipes.csv",
             "lib.py": HERE / "lib.py"}
    return {name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in files.items()}


def triple_basis(matrix, variables):
    labels = np.unique(variables.ligand, return_inverse=True)[1]
    if labels.max() >= 64:
        raise ValueError("Exact bitmask engine supports at most 64 encoded levels")
    triples = np.array(list(itertools.combinations(range(len(labels)), 3)))
    ligands = labels[triples]
    keep = ((ligands[:, 0] != ligands[:, 1]) &
            (ligands[:, 0] != ligands[:, 2]) &
            (ligands[:, 1] != ligands[:, 2]))
    triples, ligands = triples[keep], ligands[keep]
    masks = np.bitwise_or.reduce(
        np.left_shift(np.uint64(1), ligands.astype(np.uint64)), axis=1)
    y = np.nan_to_num(matrix.values, nan=0.0).T
    maxima = np.maximum.reduce(y[triples], axis=1)
    return triples, masks, maxima


def legal_tails(triples, masks, i):
    start = np.searchsorted(triples[:, 0], triples[i, 2], side="right")
    return np.flatnonzero((masks[start:] & masks[i]) == 0) + start


def exact_draw(matrix, variables):
    """Enumerate all legal sets; partition disk-backed complete scores exactly."""
    count = legal_count(variables.ligand.value_counts())
    triples, masks, maxima = triple_basis(matrix, variables)
    full_sum = float(np.nanmax(matrix.values, axis=1).sum())
    lower_rank, upper_rank = (count - 1) // 2, count // 2
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="P3_fig5_exact_") as temporary:
        scores = np.memmap(Path(temporary) / "scores.bin", dtype="float64",
                           mode="w+", shape=(count,))
        offset = 0
        checkpoint = 0
        for i in range(len(triples)):
            tails = legal_tails(triples, masks, i)
            n = len(tails)
            if n:
                scores[offset:offset + n] = np.maximum(
                    maxima[tails], maxima[i]).sum(axis=1)
                offset += n
            if offset - checkpoint >= 25_000_000:
                checkpoint = offset
                print("exact Figure 5:", offset, "/", count,
                      "elapsed %.1fs" % (time.perf_counter() - started), flush=True)
        assert offset == count, (offset, count)
        scores.flush()
        ordered = np.memmap(Path(temporary) / "partition.bin", dtype="float64",
                            mode="w+", shape=(count,))
        ordered[:] = scores
        ordered.partition((lower_rank, upper_rank))
        lower, upper = float(ordered[lower_rank]), float(ordered[upper_rank])
        del ordered
        selected_rank = None
        for start in range(0, count, 1_000_000):
            matches = np.flatnonzero(scores[start:start + 1_000_000] == lower)
            if len(matches):
                selected_rank = start + int(matches[0])
                break
        assert selected_rank is not None
        del scores
        offset = 0
        chosen = None
        for i in range(len(triples)):
            tails = legal_tails(triples, masks, i)
            if offset + len(tails) > selected_rank:
                chosen = np.concatenate([triples[i], triples[tails[selected_rank - offset]]])
                break
            offset += len(tails)
        assert chosen is not None
    ids = list(matrix.columns[chosen])
    actual = lib.retention(matrix.values, list(chosen))[0]
    assert float(actual.sum()) == lower
    print("exact Figure 5 complete: %.1fs" % (time.perf_counter() - started), flush=True)
    return {"legal_combinations": count, "median_method": "lower_order_median",
            "lower_order_rank_zero_based": lower_rank,
            "upper_order_rank_zero_based": upper_rank,
            "lower_central_yield_sum": lower, "upper_central_yield_sum": upper,
            "arithmetic_median_yield_sum": (lower + upper) / 2,
            "lower_median_retention": lower / full_sum,
            "upper_median_retention": upper / full_sum,
            "arithmetic_median_retention": ((lower + upper) / 2) / full_sum,
            "selected_combination_lexicographic_rank_zero_based": selected_rank,
            "ligand_conditions": ids}


def recipe_rows(result, variables, conditions):
    rxn_class = result["screen"].rsplit("|", 1)[0]
    metadata = conditions[conditions.rxn_class == rxn_class].set_index("condition_id")
    rows = []
    for method in ("ligand", "history"):
        for rank, condition in enumerate(result[method + "_conditions"], 1):
            source = metadata.loc[condition]
            row = {"screen": result["screen"], "method": method, "rank": rank,
                   "condition_id": condition,
                   "encoded_ligand": variables.loc[condition, "ligand"]}
            for role, column in (("precatalyst", "Reagent1_Standardized Name"),
                                 ("added_ligand", "Reagent2_Standardized Name"),
                                 ("base", "Reagent3_Standardized Name"),
                                 ("additive", "Reagent4_Standardized Name"),
                                 ("solvent", "Standardized Name"),
                                 ("temperature_C", "temp_id"), ("time_h", "time_id")):
                row[role] = "none" if pd.isna(source[column]) else source[column]
            for slot in range(1, 7):
                value = source["Reagent%d_HTE_Equiv" % slot]
                row["reagent%d_equiv" % slot] = "none" if pd.isna(value) else value
            rows.append(row)
    return pd.DataFrame(rows)


def build(write_outputs=False, exact=False):
    screen, matrix, variables, conditions = load_library()
    hashes = input_hashes()
    if not exact and RESULT.exists():
        result = json.loads(RESULT.read_text(encoding="utf-8"))
        assert result["input_sha256"] == hashes, "Figure 5 example inputs changed; run --exact"
        assert result["screen"] == screen
    else:
        result = exact_draw(matrix, variables)
        result["screen"] = screen
        result["pairs"] = len(matrix)
        result["condition_count"] = len(matrix.columns)
        result["encoded_ligand_levels"] = int(variables.ligand.nunique())
        result["level_sizes"] = sorted(map(int, variables.ligand.value_counts()), reverse=True)
        result["history_conditions"] = list(matrix.columns[lib.greedy_kit(matrix.values, K)])
        result["input_sha256"] = hashes
    for method in ("ligand", "history"):
        ids = result[method + "_conditions"]
        assert len(ids) == len(set(ids)) == K
        indices = [matrix.columns.get_loc(cid) for cid in ids]
        best, full = lib.retention(matrix.values, indices)
        value = float(best.mean() / full.mean())
        if method + "_retention" in result:
            assert result[method + "_retention"] == value
        result[method + "_retention"] = value
    assert variables.loc[result["ligand_conditions"], "ligand"].nunique() == K
    assert abs(result["ligand_retention"] - result["lower_median_retention"]) < 1e-14
    rows = recipe_rows(result, variables, conditions)
    if write_outputs:
        write(result, rows)
    return result, rows


def write(result, rows):
    OUT.mkdir(exist_ok=True)
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    rows.to_csv(RECIPES, index=False, lineterminator="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact", action="store_true", help="re-enumerate every legal combination")
    args = parser.parse_args()
    result, _ = build(write_outputs=True, exact=args.exact)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
