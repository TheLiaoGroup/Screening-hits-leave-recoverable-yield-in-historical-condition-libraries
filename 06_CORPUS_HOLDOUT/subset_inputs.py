"""Locate and verify the public fixed-member subset and frozen records."""
import hashlib
import json
import os
from pathlib import Path


def input_root():
    return Path(os.environ.get("P3_HOLDOUT_INPUT_DIR", Path(__file__).resolve().parent))


def input_path(name):
    return input_root() / "data" / name


def verify_inputs(freeze_path):
    root = input_root()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    for name, expected in manifest["subset_files"].items():
        assert sha(input_path(name)) == expected["sha256"], f"Subset input changed: {name}"
    assert sha(freeze_path) == manifest["freeze_hashes"][freeze_path.name], "Frozen record changed"
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    for key in ("private_sha256", "raw_condition_sha256"):
        if key in frozen["source"]:
            assert frozen["source"][key] == manifest["original_sources"][key]
    for block in frozen["blocks"]:
        path = freeze_path.parent / "blocks" / block["membership_file"]
        assert sha(path) == block["membership_sha256"], f"Frozen membership changed: {path.name}"
