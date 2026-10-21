# Code and data

Code and data for *Screening Histories Recover the Yield That High-Throughput Hits Leave Behind*,
are deposited in Zenodo: https://doi.org/10.5281/zenodo.23319141
and on GitHub: https://github.com/TheLiaoGroup/Screening-hits-leave-recoverable-yield-in-historical-condition-libraries

## Reproduce the headline values

```text
python -m pip install -r requirements.txt
python REPRODUCE_gc.py
```

The expected result is `29 checks pass, 0 fail`. The script reads only `data/arrays.csv` (or a
directory named by `BH_DATA_DIR`) and imports no other module of this package.

## Rescore the corpus hold-out wells

```text
python 06_CORPUS_HOLDOUT/rescore.py
```

This rescores the released 23,617 fixed-member wells and 739 recipe rows, writes the results to
`out/corpus_holdout/` and compares them with the deposited scores. See `06_CORPUS_HOLDOUT/README.md`.

## Rebuild the analyses and figures

Run the scripts in this order from this folder. Each writes its derived files to `out/`, and the
figure builders write to `figures/`. The bootstrap and robustness analyses take longer than the
headline checks.

```text
python kits.py
python build_kits.py
python kit_sealed.py
python kit_reserve.py
python kit_consensus.py
python kit_k90ci.py
python kit_library.py
python kit_axes.py
python review_axes_gc.py
python review_loss_gc.py
python review_generalization_gc.py
python review_tail_gc.py
python review_tail_k12_gc.py
python review_tail_robustness_gc.py
python posthit_gc.py
python posthit_added_recipes_gc.py
python zhang_overlap_gc.py
python example_pair_gc.py
python ligand_panel_example_gc.py --exact
python recovered_recipes_gc.py
python figure1_gc.py
python figures_posthit_gc.py
python figures_ligand_gc.py
python figures_kits.py
python figures_tail_gc.py
python figures_toc_gc.py
```

The figure builders use Arial (regular, bold and italic). Figure 2c and Supplementary Figure 5 use
the supplied ChemDraw panels in `figures/` (`example_pair_scheme` and `transformation_schemes`, as
PDF and editable CDXML); RDKit drawings are used when those panels are absent.
Figure 4 reads `figures/recovered_recipes_schemes.pdf` and its editable CDXML master at
one fixed physical scale. It checks all six reactions against the selected substrate and
product SMILES and rejects missing or mismatched masters. Repainting the figures does not
require ChemDraw; the deposited vector masters are the drawing inputs.

Figure 1 is rendered from the editable, layered `figures/Figure1_overview.svg` by
`figure1_overview_gc.py`, called through `figure1_gc.py`. If supplied, the edited
`figures/Figure1_overview_master.pdf` takes priority. Its formal outputs retain the
`Figure1_hit_gap.pdf` and `.png` filenames.

`figures_posthit_gc.py` draws Figures 2 and 3, and `figures_ligand_gc.py` draws Figure 5.
The Supplementary Table 20 example is selected from every legal six-condition combination by an exact
order statistic. `ligand_panel_example_gc.py --exact` enumerates all combinations using a
temporary disk array; it requires about 5 GB of temporary space and several minutes.
Without `--exact`, the script validates the deposited input hashes, selected panel, and scores.

The public-matrix comparison (Supplementary Note 5) uses the archive of Sivilotti et al. (Zenodo
14861626), which is not redistributed. Its result is supplied in `out/`. To recompute it, extract
the archive, set `SIVILOTTI_DIR` to its `datasets/real_datasets/` subdirectory and run
`python public_matrices_gc.py`.

`zhang_overlap_gc.py` compares the released amide-coupling records with the published collection of
Zhang et al., which is not redistributed. Download it from the URL in
`data/public_reference/manifest.json` into that folder as `amide_coupling.csv` (the script checks its
SHA-256). The overlap counts are written to `out/zhang_overlap_summary.json` and
`out/zhang_overlap_summary.csv`.

## Contents

- `data/`: the eighteen arrays with held-out pairs and the replicate condition, condition and
  compound tables and a pinned pointer to the published amide-coupling collection (`data/README.md`);
- `06_CORPUS_HOLDOUT/`: fixed-member wells and recipes, frozen records, membership files,
  deposited scores and scoring code;
- `out/`: frozen inputs used by the analyses (functional-group annotations, software provenance
  and the public-matrix result);
- `figures/` and `source_data/`: figure files and the Source Data workbook;
- the root Python files: analysis and figure code;
- `requirements.txt`: pinned environment.

## Licences

Code: MIT (`LICENCE.txt`). Platform data, including `06_CORPUS_HOLDOUT/data/`: CC BY 4.0
(`data/LICENCE.txt`). Third-party data (the Zhang and Sivilotti collections) are not redistributed.
