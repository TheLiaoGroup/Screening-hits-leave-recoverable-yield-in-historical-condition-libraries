# Dataset — Screening hits leave recoverable yield in historical condition libraries

Eighteen substrate-pair × condition arrays from one high-throughput platform (96-well plates, one substrate pair per plate,
reported yields from a calibrated analytical workflow), fourteen transformations, sealed held-out pairs from eight arrays,
reserve pairs from the other ten, and one replicate Hiyama condition.

| file | rows | content |
|---|---|---|
| `arrays.csv` | 251,925 | one measured well per row: `array_id`, `array` (source label; `array_labels.csv` gives the manuscript display name), `transformation`, `rxn_class`, `plate`, `sub_1_smiles`/`role_1`, `sub_2_smiles`/`role_2`, `condition_id` (unique within a transformation), `yield` (fraction, 0–1), `check_passed` (source-validation flag carried from the platform; all rows are used), `set` |
| `conditions.csv` | 1,473 | every condition of these arrays: `rxn_class`, `condition_id`, reagents 1–6 as name / CAS / equivalents / compound id, `solvent`, `temperature_C`, `time_h`, `arrays` that use it |
| `compounds.csv` | 502 | `compound_id`, `name`, `cas`, `cas_alt` for every reagent and solvent in `conditions.csv` |
| `arrays_summary.csv` | 18 | one row per array: label, roles, pairs, conditions, measurements, completeness, temperature, sealed pairs |
| `measurement_version_differences.csv` | 247 | amide-coupling records whose yield (79) or reactant assignment (168) differs from the published collection of Zhang et al.; both versions |
| `public_reference/manifest.json` | — | URL, SHA-256 and size of that published collection, which is not redistributed |

`set` marks the frozen partition, defined using seed 20260818; held-out rows do not enter panel fitting in this study.
`analysis` is the 191,174 measurements of the eighteen arrays (2,360 substrate
pairs) behind every retention curve, kit and table.
`sealed` is the 20,748 measurements of the 281 sealed pairs of eight
arrays: the fifth of each array's pairs whose nearest neighbour under mean ECFP4 Tanimoto over both
partners is most distant.
`reserve` is the 39,929 measurements of the 460 pairs of the other ten
arrays that the partition never drew: a 70 per cent random sample of substrate pairs was analysed there,
and these are the rest. Sealed and reserve together give every array a test on pairs excluded from panel fitting. The same corpus also supports companion manuscripts by the authors.
Reserve rows carry no plate identifier; every other column is as for the analysis rows.
`replicate` is the 74 wells of Hiyama condition C46, identical in composition to C43: C46 is in no
array matrix. It provides a repeatability check (Supplementary Note 1).

Among analysis records, 77,012 of 191,174 numerical yields are exactly zero (40.3%). Of 2,360 analysis array-pair instances, 715 have zero full-array maximum (30.3%). A well scored as no product by DESI and not selected for chromatography was recorded as zero. Recorded zeros are retained as supplied. These records are distinct from unmeasured matrix cells. Full counts and productive-pair loss distributions are in `out/review_loss*` beside the code.

Partner 1 and partner 2 are the substrates as listed; `role_1` and `role_2` say which is the aryl halide, amine, acid and so on.
Every array runs at one temperature and one time (`arrays_summary.csv`, `conditions.csv`). The duplicated Hiyama condition
(C46 = C43) is already removed, so the Hiyama array carries 91 conditions.

Loading: the released code (`lib.load()`) reads this directory when `BH_DATA_DIR` points at it or it sits as `data/` beside the code;
`python REPRODUCE_gc.py` recomputes the deposited core and post-hit headline numbers (29 checks) from the shipped files.
The separate `public_matrices_gc.py` directionality analysis requires the Sivilotti archive (Zenodo 14861626) (see the top-level README); its frozen result is supplied in `out/`.

Licence: CC BY 4.0 (see `LICENCE.txt`). Cite the paper.
