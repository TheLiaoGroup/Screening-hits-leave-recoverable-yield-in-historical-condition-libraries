# Errata to the frozen corpus hold-out records

A scored freeze is never edited. Its sha256 establishes content identity; a hash alone does not
establish when registration occurred or whether it preceded scoring. Chronology requires separately
dated records. Corrections therefore live here, each naming the record it corrects by the hash the
record carried when it was scored. Every hash below is the live hash of
the file it names (`sha256sum` reproduces it); no freeze file contains an erratum of its own.

---

## FREEZE_transported_panel.json

sha256 `56a2af01eea141ed8749332ba8f600a32745bddd9f0fa34b90b936d64a8494e6`

**Where.** `criteria.outcomes_fixed_in_advance.rationale`

**What it says.** “the lowest of the eighteen”

**What is true.** joint-lowest at the two-decimal display of the manuscript's Table 1, and second on the raw values

| array | cross-validated retention at k = 6 | as Table 1 displays it |
|---|---|---|
| Buchwald-Hartwig\|2, the array this block belongs to | 0.710967 | 0.71 |
| C-O competitive reactions\|1 | 0.710221 | 0.71 |

**Effect on the registration.** None. The phrase sits in the rationale for the thresholds, not in a rule, an endpoint or a threshold, and no scored value depends on it.

The wrong phrase is left in the record. Removing it would rewrite a scored registration, which is
the thing this file exists to avoid.
