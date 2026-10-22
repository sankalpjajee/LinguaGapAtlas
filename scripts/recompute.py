#!/usr/bin/env python3
"""Recompute the headline statistics of "LinguaGapAtlas" from the corpus.

Reads the canonical corpus CSV (default: data/corpus.csv, resolved relative
to this package's root) and verifies the shares reported in the paper:

  * translated-data share        -> data_origin == "Translated"
  * culturally non-specific share -> cultural_specificity == "Universal"
  * low-resource share            -> resource_tier == "Low"

Each share is computed separately for benchmarks and for datasets.

Null rule: a value that is empty or one of the recognised null tokens is
UNDETERMINED. Undetermined rows are counted and reported, and are excluded
from that statistic's denominator -- never imputed and never silently
dropped. If a denominator is zero the share is reported as undefined rather
than raising or printing a misleading 0%.

Usage:
    python3 scripts/recompute.py [--csv PATH]

Exit code is 0 when the file was read and all statistics were computed
(undefined shares included); 2 when the input cannot be read or lacks the
required columns.
"""

import argparse
import csv
import sys
from pathlib import Path

NULL_TOKENS = {"", "null", "none", "na", "n/a", "undetermined", "unknown", "nan", "-"}

REQUIRED_COLUMNS = {"resource_id", "type", "data_origin", "resource_tier",
                    "cultural_specificity"}

# (label, column, target value)
CHECKS = [
    ("translated-data share", "data_origin", "Translated"),
    ("culturally non-specific share", "cultural_specificity", "Universal"),
    ("low-resource share", "resource_tier", "Low"),
]

GROUPS = [("Benchmark", "Benchmarks"), ("Dataset", "Datasets")]


def is_missing(value):
    """True if a raw CSV value counts as undetermined under the null rule."""
    if value is None:
        return True
    return value.strip().lower() in NULL_TOKENS


def load_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError("file has no header row")
        missing_cols = REQUIRED_COLUMNS - {c.strip() for c in reader.fieldnames}
        if missing_cols:
            raise ValueError(
                "missing required column(s): " + ", ".join(sorted(missing_cols)))
        return list(reader)


def share(rows, column, target):
    """Return (hits, coded, undetermined) for target within rows."""
    coded_values = [r.get(column, "") for r in rows if not is_missing(r.get(column, ""))]
    undetermined = len(rows) - len(coded_values)
    hits = sum(1 for v in coded_values if v.strip().lower() == target.lower())
    return hits, len(coded_values), undetermined


def fmt_pct(hits, denom):
    if denom == 0:
        return "undefined (no coded rows)"
    return f"{100.0 * hits / denom:.1f}%"


def main(argv=None):
    package_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", default=str(package_root / "data" / "corpus.csv"),
                        help="path to corpus.csv (default: data/corpus.csv in this package)")
    args = parser.parse_args(argv)

    try:
        rows = load_rows(args.csv)
    except (OSError, ValueError) as exc:
        print(f"error: cannot compute statistics: {exc}", file=sys.stderr)
        return 2

    print(f"Corpus: {args.csv}")
    print(f"Rows read: {len(rows)}")
    by_type = {}
    other = []
    for row in rows:
        t = (row.get("type") or "").strip().lower()
        if t == "benchmark":
            by_type.setdefault("Benchmark", []).append(row)
        elif t == "dataset":
            by_type.setdefault("Dataset", []).append(row)
        else:
            other.append(row)
    for key, label in GROUPS:
        print(f"  {label}: {len(by_type.get(key, []))}")
    if other:
        print(f"  WARNING: {len(other)} row(s) with missing/unrecognised type "
              f"(excluded from per-type shares, counted in totals below)")

    for key, label in GROUPS:
        group = by_type.get(key, [])
        print(f"\n{label} (n = {len(group)})")
        if not group:
            print("  no rows in this group; shares undefined")
            continue
        for check_label, column, target in CHECKS:
            hits, coded, undetermined = share(group, column, target)
            print(f"  {check_label} [{column} == {target}]: "
                  f"{hits}/{coded} coded rows = {fmt_pct(hits, coded)} "
                  f"(undetermined: {undetermined})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
