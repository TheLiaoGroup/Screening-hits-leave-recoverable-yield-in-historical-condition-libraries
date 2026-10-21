# Dataset — Screening Histories Recover the Yield That High-Throughput Hits Leave Behind

Eighteen substrate-pair × condition arrays from one high-throughput platform (96-well plates, one substrate pair per plate,
reported yields from a calibrated analytical workflow), fourteen transformations, sealed held-out pairs from eight arrays,
reserve pairs from the other ten, and one replicate Hiyama condition.

| file | rows | content |
|---|---|---|
| `arrays.csv` | 252,190 | one row per retained designed cell the platform screened: `array_id`, `array` (source label; `array_labels.csv` gives the manuscript display name), `transformation`, `rxn_class`, `plate`, `name0` (original measurement ID), `sub_1_smiles`/`role_1`, `sub_2_smiles`/`role_2`, `condition_id` (unique within a transformation), `yield` (fraction, 0–1; blank for an unmeasured cell), `measurement_status`, `check_passed` (source-validation flag carried from the platform), `set` |
| `conditions.csv` | 1,473 | every condition of these arrays: `rxn_class`, `condition_id`, reagents 1–6 as name / CAS / equivalents / compound id, `solvent`, `temperature_C`, `time_h`, `arrays` that use it |
| `compounds.csv` | 502 | `compound_id`, `name`, `cas`, `cas_alt` for every reagent and solvent in `conditions.csv` |
| `arrays_summary.csv` | 18 | one row per array: label, roles, pairs, conditions, measurements, completeness, temperature, sealed and reserve pairs |
| `measurement_version_differences.csv` | 0 | retained-record comparison with the pinned published collection; no excluded wells |
| `public_reference/manifest.json` | — | URL, SHA-256 and size of that published collection, which is not redistributed |

`measurement_status` follows the platform's rule:

| value | meaning | `yield` | rows |
|---|---|---|---:|
| `quantified` | chromatographic yield (UPLC–UV/MS calibrated against ¹H NMR) | the yield | 251,646 |
| `zero_by_rule` | the mass-spectrometry screen (DESI–TOF–MS) found no product; not chromatographed | 0 | 12 |
| `detected_not_quantified` | a product signal, but no usable chromatographic yield: the product peak overlapped another peak or the internal standard integrated abnormally | blank | 109 |
| `no_screen_record` | no screening record | blank | 423 |

The first two are measurements and every analysis uses them; the last two are unmeasured cells and are
never imputed. Excluded wells and designed cells without any well are not listed.

`set` marks the frozen partition, defined using seed 20260818; held-out rows do not enter panel fitting in this study.
`analysis` is the 190,905 measurements of the eighteen arrays (2,357 substrate
pairs) behind every retention curve, kit and table.
`sealed` is the 20,749 measurements of the 281 sealed pairs of eight
arrays: the fifth of each array's pairs whose nearest neighbour under mean ECFP4 Tanimoto over both
partners is most distant.
`reserve` is the 39,930 measurements of the 460 pairs of the other ten
arrays that the partition never drew: a 70 per cent random sample of substrate pairs was analysed there,
and these are the rest. Sealed and reserve together give every array a test on pairs excluded from panel fitting.
Reserve rows carry no plate identifier; every other column is as for the analysis rows.
`replicate` is the 74 wells of Hiyama condition C46, identical in composition to C43: C46 is in no
array matrix. It provides a repeatability check (Supplementary Note 1).

Among analysis measurements, 76,990 of 190,905 are exactly zero (40.3%). Of 2,357 analysis array-pair instances, 715 have zero full-array maximum (30.3%). Full counts and productive-pair loss distributions are in `out/review_loss*` beside the code.

Partner 1 and partner 2 are the substrates as listed; `role_1` and `role_2` say which is the aryl halide, amine, acid and so on.
Every array runs at one temperature and one time (`arrays_summary.csv`, `conditions.csv`). The duplicated Hiyama condition
(C46 = C43) is already removed, so the Hiyama array carries 91 conditions.

Loading: the released code (`lib.load()`) reads this directory when `BH_DATA_DIR` points at it or it sits as `data/` beside the code,
and keeps the measured cells; `python REPRODUCE_gc.py` recomputes the core and post-hit headline numbers from the shipped files.
The separate `public_matrices_gc.py` directionality analysis requires the Sivilotti archive (Zenodo 14861626) (see the top-level README); its result is supplied in `out/`.

Licence: CC BY 4.0 (see `LICENCE.txt`). Cite the paper.
