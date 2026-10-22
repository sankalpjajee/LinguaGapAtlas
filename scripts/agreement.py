#!/usr/bin/env python3
"""Inter-annotator agreement between two coders' label files.

Reads two CSVs with one row per resource (joined on `resource_id`) and, for
each taxonomy dimension, computes over the pairs where BOTH coders recorded a
value:

  * observed agreement (proportion of identical labels), and
  * Cohen's kappa.

Missing-value handling (the package null rule): an empty value or a null
token (null / NA / undetermined / ...) means UNDETERMINED. A pair where one
or both coders left the dimension undetermined is excluded from that
dimension's agreement calculation and the exclusion is counted and reported
-- never treated as agreement or disagreement, and never imputed.

The disagreement list (resource_id + both labels, per dimension) is printed
with --disagreements: it is the working list for adjudication and for writing
the annotation decision rules.

Usage:
    python3 scripts/agreement.py CODER_A.csv CODER_B.csv [--disagreements]

Exit code is 0 on success; 2 when an input cannot be read, lacks the join
key, or shares no resource_ids.
"""

import argparse
import csv
import sys
from collections import Counter

NULL_TOKENS = {"", "null", "none", "na", "n/a", "undetermined", "unknown", "nan", "-"}

DEFAULT_DIMENSIONS = ["data_origin", "resource_tier", "task_class",
                      "cultural_specificity", "inference_paradigm"]

JOIN_KEY = "resource_id"


def is_missing(value):
    if value is None:
        return True
    return value.strip().lower() in NULL_TOKENS


def load_labels(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or JOIN_KEY not in reader.fieldnames:
            raise ValueError(f"{path}: missing required '{JOIN_KEY}' column")
        labels = {}
        for row in reader:
            rid = (row.get(JOIN_KEY) or "").strip()
            if rid:
                labels[rid] = row
        return labels, set(reader.fieldnames)


def cohens_kappa(pairs):
    """Cohen's kappa for a list of (label_a, label_b) pairs."""
    n = len(pairs)
    if n == 0:
        return None
    observed = sum(1 for a, b in pairs if a == b) / n
    counts_a = Counter(a for a, _ in pairs)
    counts_b = Counter(b for _, b in pairs)
    categories = set(counts_a) | set(counts_b)
    expected = sum((counts_a.get(c, 0) / n) * (counts_b.get(c, 0) / n)
                   for c in categories)
    if expected >= 1.0:
        # Both marginals are concentrated on a single identical category:
        # agreement is perfect by construction.
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("coder_a", help="first coder's CSV (e.g. original labels)")
    parser.add_argument("coder_b", help="second coder's CSV (e.g. independent re-coding)")
    parser.add_argument("--dimensions", nargs="+", default=DEFAULT_DIMENSIONS,
                        help="dimension columns to compare (default: the five taxonomy dimensions)")
    parser.add_argument("--disagreements", action="store_true",
                        help="also list every disagreeing pair per dimension")
    args = parser.parse_args(argv)

    try:
        labels_a, cols_a = load_labels(args.coder_a)
        labels_b, cols_b = load_labels(args.coder_b)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    shared = sorted(set(labels_a) & set(labels_b))
    if not shared:
        print("error: the two files share no resource_id values", file=sys.stderr)
        return 2
    only_a = len(set(labels_a) - set(labels_b))
    only_b = len(set(labels_b) - set(labels_a))
    print(f"Resources in both files: {len(shared)} "
          f"(only in A: {only_a}, only in B: {only_b} -- unmatched rows are excluded)")

    dimensions = [d for d in args.dimensions if d in cols_a and d in cols_b]
    skipped = [d for d in args.dimensions if d not in dimensions]
    for d in skipped:
        print(f"WARNING: dimension '{d}' not present in both files; skipped")

    pooled_pairs = []
    print(f"\n{'dimension':<22} {'pairs':>5} {'agree':>5} {'observed':>9} "
          f"{'kappa':>7} {'excluded (undetermined)':>24}")
    for dim in dimensions:
        pairs, disagreements = [], []
        excluded_one, excluded_both = 0, 0
        for rid in shared:
            va, vb = labels_a[rid].get(dim, ""), labels_b[rid].get(dim, "")
            ma, mb = is_missing(va), is_missing(vb)
            if ma or mb:
                if ma and mb:
                    excluded_both += 1
                else:
                    excluded_one += 1
                continue
            pair = (va.strip(), vb.strip())
            pairs.append(pair)
            if pair[0].lower() != pair[1].lower():
                disagreements.append((rid, pair[0], pair[1]))
        pooled_pairs.extend(pairs)
        if pairs:
            observed = sum(1 for a, b in pairs if a.lower() == b.lower()) / len(pairs)
            # Kappa on case-normalised labels, consistent with the agreement test above.
            kappa = cohens_kappa([(a.lower(), b.lower()) for a, b in pairs])
            kappa_str = f"{kappa:.3f}" if kappa is not None else "n/a"
            obs_str = f"{observed:.3f}"
            agree_n = sum(1 for a, b in pairs if a.lower() == b.lower())
        else:
            obs_str, kappa_str, agree_n = "n/a", "n/a", 0
        print(f"{dim:<22} {len(pairs):>5} {agree_n:>5} {obs_str:>9} {kappa_str:>7} "
              f"{excluded_one + excluded_both:>10} (one: {excluded_one}, both: {excluded_both})")
        if args.disagreements and disagreements:
            for rid, va, vb in disagreements:
                print(f"    disagree {rid}: A={va!r} vs B={vb!r}")

    if pooled_pairs:
        overall_obs = sum(1 for a, b in pooled_pairs if a == b) / len(pooled_pairs)
        overall_kappa = cohens_kappa(pooled_pairs)
        print(f"\nOverall (pooled across dimensions): {len(pooled_pairs)} pairs, "
              f"observed agreement {overall_obs:.3f}, "
              f"Cohen's kappa {overall_kappa:.3f}")
    else:
        print("\nOverall: no dimension had any pair coded by both coders; "
              "agreement is undefined.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
