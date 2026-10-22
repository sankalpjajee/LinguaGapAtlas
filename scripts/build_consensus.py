#!/usr/bin/env python3
"""Build the canonical LinguaGapAtlas corpus from the five independent coding rounds.

Each cell of the five coded dimensions (type, task_class, data_origin,
resource_tier, cultural_specificity) is decided by majority vote across the
rounds that coded it: a value enters the canonical corpus only when at least
3 of the 5 rounds agree on it. Otherwise the cell is left EMPTY
(undetermined) -- never imputed. Labels are normalized before voting (case
unified; "Translation/alignment" -> Translation, "Instruction fine-tuning"
-> Instruction, "long-context" -> Long Context) and written back in the
canonical casing of the data dictionary.

Descriptive fields: year by majority vote (ties -> round A); language_count
by modal reported count (ties -> median); languages list taken from a round
that reported the chosen count and enumerated names.

Usage:
    python3 scripts/build_consensus.py --rounds A=coding/coderA-full.csv \
        B=coding/coderB-full.csv ... --base coding/coderA-full.csv \
        --out data/corpus.csv
Run from the corpus root (the directory containing coding/ and package/).
"""

import argparse
import collections
import csv
import sys
from pathlib import Path

SYN = {"translation/alignment": "translation",
       "instruction fine-tuning": "instruction",
       "long-context": "long context"}
NULL = {"", "null", "none", "na", "n/a", "undetermined", "unknown", "nan", "-"}

CANON = {
    "type": {"benchmark": "Benchmark", "dataset": "Dataset"},
    "data_origin": {"native": "Native", "translated": "Translated", "mixed": "Mixed"},
    "resource_tier": {"low": "Low", "medium": "Medium", "high": "High"},
    "cultural_specificity": {"universal": "Universal",
                             "region-specific": "Region-Specific",
                             "culturally anchored": "Culturally Anchored"},
    "task_class": {"pre-training": "Pre-training", "translation": "Translation",
                   "instruction": "Instruction", "qa": "QA", "nlu": "NLU",
                   "generation": "Generation", "other": "Other", "syntax": "Syntax",
                   "reasoning": "Reasoning", "multimodal": "Multimodal",
                   "summarization": "Summarization", "math": "Math", "rag": "RAG",
                   "code": "Code", "long context": "Long Context",
                   "embeddings": "Embeddings"},
}
DIMS = list(CANON)


def norm(v):
    v = (v or "").strip().lower()
    return SYN.get(v, v)


def coded(row, dim):
    v = norm(row.get(dim))
    return None if v in NULL else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", nargs="+", required=True,
                    help="LETTER=path entries, in round order")
    ap.add_argument("--base", required=True, help="round file supplying row scaffolding")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rounds = {}
    for entry in args.rounds:
        letter, path = entry.split("=", 1)
        with open(path) as f:
            rounds[letter] = {r["resource_id"]: r for r in csv.DictReader(f)}
    letters = list(rounds)
    with open(args.base) as f:
        base_rows = list(csv.DictReader(f))
        header = base_rows[0].keys()

    out_rows = []
    stats = collections.Counter()
    for base in base_rows:
        rid = base["resource_id"]
        row = dict(base)
        notes = []
        weak = 0
        for dim in DIMS:
            votes = [coded(rounds[L][rid], dim) for L in letters]
            cast = [v for v in votes if v is not None]
            if not cast:
                row[dim] = ""
                continue
            top, cnt = collections.Counter(cast).most_common(1)[0]
            if cnt >= 3:
                row[dim] = CANON[dim].get(top, top)
                if cnt < len(cast):
                    weak += 1
                    notes.append(f"{dim} {cnt}/{len(cast)}")
            else:
                row[dim] = ""
                weak += 1
                notes.append(f"{dim} no majority ({dict(collections.Counter(cast))})")
                stats[f"{dim}_no_majority"] += 1
        # year: majority, tie -> base round's year
        years = [norm(rounds[L][rid].get("year")) for L in letters]
        years = [y for y in years if y not in NULL]
        if years:
            top, cnt = collections.Counter(years).most_common(1)[0]
            row["year"] = top if cnt >= 2 else base["year"]
        # language_count: modal count, tie -> median of reported counts
        counts = []
        for L in letters:
            c = norm(rounds[L][rid].get("language_count"))
            if c not in NULL:
                try:
                    counts.append(int(float(c)))
                except ValueError:
                    pass
        if counts:
            cnt_counter = collections.Counter(counts)
            best = max(cnt_counter.values())
            modes = [c for c, n in cnt_counter.items() if n == best]
            chosen = modes[0] if len(modes) == 1 else sorted(counts)[len(counts) // 2]
            row["language_count"] = str(chosen)
            # languages: longest enumerated list among rounds reporting the chosen count
            best_list = ""
            for L in letters:
                c = norm(rounds[L][rid].get("language_count"))
                langs = (rounds[L][rid].get("languages") or "").strip()
                if c not in NULL and int(float(c)) == chosen and len(langs) > len(best_list):
                    best_list = langs
            if best_list:
                row["languages"] = best_list
        row["coder"] = "consensus-5rounds"
        row["confidence"] = "high" if weak == 0 else ("medium" if weak <= 2 else "low")
        vote_note = "Consensus of 5 independent coding rounds (majority >=3)."
        if notes:
            vote_note += " Split cells: " + "; ".join(notes) + "."
        row["evidence_note"] = vote_note + " Round A evidence: " + (base.get("evidence_note") or "")
        out_rows.append(row)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(header))
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {len(out_rows)} rows to {args.out}")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    sys.exit(main())
