# Corpus hold-out scoring inputs

These fixed-member inputs are released under CC BY 4.0. They contain the 23,617 wells named by the
four frozen registrations (16 blocks), the 739 corresponding recipe rows, the frozen registrations
and membership lists, the deposited scores, four scoring scripts and a portable runner. Original
CSV row numbers are retained. The runner uses `lib.py`, the analysis arrays and the condition table
from the package root.

Run from the package root with the environment in `requirements.txt`:

```text
python 06_CORPUS_HOLDOUT/rescore.py
```

The runner writes to `out/corpus_holdout/`, leaving the deposited scores unchanged, and compares
eight score tables and three result JSON files with the deposited versions. The four scorer copies
change only input paths, source-verification gates and reading the recipe subset from CSV instead
of XLSX; their algorithms, random seeds, panel sizes, thresholds and output calculations are
unchanged. The runner rescores the fixed members of each registration; the historical-control values
of the last three scorers are read from the frozen registrations.

The full-source hashes verified when the subset was extracted are recorded in the frozen records,
and `manifest.json` lists the hashes of the extracted files. The subset contains only effective
membership cells; twenty-six cells read by the original transported selectors and then masked out
are omitted without changing their scored matrices.
