# Published amide-coupling collection (not redistributed)

The released amide-coupling records are compared with the published collection of Zhang et al.,
the 47,015-row file `data/all_HTE_with_condition.csv` of the `aichemeco/amide_coupling` repository at
the commit recorded in `manifest.json`. That file is not included here. To rerun
`zhang_overlap_gc.py`, download it from the URL in `manifest.json` into this folder as
`amide_coupling.csv` (or set `ZHANG_REFERENCE` to its path); the script checks its SHA-256 and size.

The overlap counts are written to `out/zhang_overlap_summary.json` and `out/zhang_overlap_summary.csv`.
Unmatched rows are not interpreted as proof that a measurement was unpublished.
