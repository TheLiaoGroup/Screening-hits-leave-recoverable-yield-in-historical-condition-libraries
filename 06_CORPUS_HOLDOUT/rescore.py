"""Rescore the public fixed-member wells and compare with the deposited results."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
from pandas.testing import assert_frame_equal


ROOT = Path(__file__).resolve().parent
SCORERS = ("score_corpus_holdout.py", "score_transported_panel.py",
           "score_transported_seven.py", "score_transported_fullish.py")
TABLES = ("holdout_scores.csv", "holdout_pair_scores.csv", "transported_scores.csv",
          "transported_pair_scores.csv", "seven_scores.csv", "seven_pair_scores.csv",
          "fullish_scores.csv", "fullish_pair_scores.csv")
RESULTS = ("TRANSPORTED_RESULT.json", "SEVEN_RESULT.json", "FULLISH_RESULT.json")


def compare_json(actual, expected, where="root"):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), where
        for key in expected:
            compare_json(actual[key], expected[key], where + "." + key)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), where
        for index, value in enumerate(expected):
            compare_json(actual[index], value, f"{where}[{index}]")
    elif isinstance(expected, float):
        assert ((math.isnan(expected) and math.isnan(actual))
                or math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12)), where
    else:
        assert actual == expected, (where, actual, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-package", type=Path, default=ROOT.parent,
                        help="Package root (defaults to the parent of this script's folder)")
    parser.add_argument("--output", type=Path,
                        help="Result folder (defaults to out/corpus_holdout in the package)")
    args = parser.parse_args()
    public = args.public_package.resolve()
    output = (args.output or public / "out" / "corpus_holdout").resolve()
    deposited = public / "06_CORPUS_HOLDOUT"
    for name in ("lib.py", "data/arrays.csv", "data/conditions.csv"):
        assert (public / name).is_file(), f"Missing public input: {name}"
    for protected in (ROOT, deposited, public / "data"):
        assert output != protected and protected not in output.parents and output not in protected.parents, \
            f"Output must not overlap deposited inputs: {protected}"
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    for name, expected in manifest["freeze_hashes"].items():
        source = deposited / name
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
        shutil.copy2(source, output / name)
    (output / "blocks").mkdir(exist_ok=True)
    for name, expected in manifest["membership_hashes"].items():
        source = deposited / "blocks" / name
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
        shutil.copy2(source, output / "blocks" / name)
    for name in (*SCORERS, "subset_inputs.py"):
        shutil.copy2(ROOT / name, output / name)
    env = os.environ.copy()
    env.update(P3_HOLDOUT_INPUT_DIR=str(ROOT), BH_DATA_DIR=str(public / "data"),
               PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=str(public), PYTHONIOENCODING="utf-8")
    for name in SCORERS:
        print("Scoring " + name, flush=True)
        process = subprocess.run([sys.executable, str(output / name)], cwd=output, env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
        (output / (Path(name).stem + ".log")).write_text(process.stdout, encoding="utf-8")
        if process.returncode:
            print(process.stdout)
            raise SystemExit(process.returncode)
    compared = []
    for name in TABLES:
        assert_frame_equal(pd.read_csv(output / name), pd.read_csv(deposited / name),
                           check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-12)
        compared.append(name)
    for name in RESULTS:
        compare_json(json.loads((output / name).read_text(encoding="utf-8")),
                     json.loads((deposited / name).read_text(encoding="utf-8")))
        compared.append(name)
    report = {"status": "matched", "recomputed_from_wells": 23617,
              "recipe_rows": 739, "artifacts_compared": compared,
              "scope": "Fixed-member scoring outputs matched. Transfer/thinning diagnostics executed but not compared here. Frozen historical-control values are read from registrations, not refitted."}
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Matched {len(compared)} deposited scoring artifacts from raw frozen-member well values.")


if __name__ == "__main__":
    main()
