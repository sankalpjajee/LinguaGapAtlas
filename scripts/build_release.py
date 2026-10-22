#!/usr/bin/env python3
"""Build the LinguaGapAtlas research release from the package's source files.

Amalgamates the canonical corpus (data/corpus.csv), the five independent
coding rounds (coding/coder{A..E}-full.csv), the screening log, the data
dictionary and the citation metadata into one self-describing release in
release/:

  linguagapatlas.json         the whole atlas as a single JSON document
  linguagapatlas.jsonl        one resource per line (the same records)
  linguagapatlas.sqlite       the same content as relational tables + views
  linguagapatlas.schema.json  JSON Schema for the document and each record
  croissant.json              Croissant (MLCommons) metadata for the files

Null rule: nothing is imputed. An empty source cell is written as null, and
every source cell can be recovered verbatim from the release (round labels
are given both as written and in canonical form, normalized exactly as
build_consensus.py normalizes them before voting).

The build refuses to write anything if data/corpus.csv is not exactly what
build_consensus.py produces from the five rounds, or if the vote recomputed
here disagrees with any consensus cell. Output is deterministic: rebuilding
from unchanged sources reproduces the JSON files byte for byte.

Usage:
    python3 scripts/build_release.py            # (re)write release/
    python3 scripts/build_release.py --check    # verify release/ is current

Exit code is 0 on success; 1 when --check finds a stale or missing release
file; 2 when a source file is missing or fails a consistency check.
"""

import argparse
import csv
import hashlib
import json
import re
import sqlite3
import subprocess
import sys
import tempfile
from collections import Counter
from itertools import combinations
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.dont_write_bytecode = True  # keep __pycache__ out of the checksummed package
sys.path.insert(0, str(SCRIPTS))
import agreement  # noqa: E402  (package script: Cohen's kappa)
import build_consensus  # noqa: E402  (package script: normalization + vocabularies)
import recompute  # noqa: E402  (package script: headline shares)

CORPUS = "data/corpus.csv"
SCREENING = "data/screening-log.csv"
DICTIONARY = "data-dictionary.md"
CITATION = "CITATION.cff"
SUMMARY = "rounds-summary.md"
ROUNDS = ["A", "B", "C", "D", "E"]
ROUND_FILE = "coding/coder{}-full.csv"
OUT_DIR = "release"

JSON_FILE = "linguagapatlas.json"
JSONL_FILE = "linguagapatlas.jsonl"
SQLITE_FILE = "linguagapatlas.sqlite"
SCHEMA_FILE = "linguagapatlas.schema.json"
CROISSANT_FILE = "croissant.json"

# Stated in README.md; the corpus artifact's own title comes from CITATION.cff.
PAPER_TITLE = ("LinguaGapAtlas: A Systematic Taxonomy of Bias and Imbalance in "
               "96 Multilingual AI Benchmarks & Datasets")
PAPER_STATUS = "Prepared for submission to ACM FAccT 2027."
REPOSITORY_URL = "https://github.com/sankalpjajee/LinguaGapAtlas"
LICENSE_URLS = {"CC-BY-4.0": "https://creativecommons.org/licenses/by/4.0/",
                "MIT": "https://opensource.org/licenses/MIT"}

MAJORITY = 3  # build_consensus.py: a label needs >= 3 of the 5 rounds
LABEL_FIELDS = ["type", "data_origin", "resource_tier", "task_class",
                "cultural_specificity", "inference_paradigm"]
DIMENSIONS = [("D1", "data_origin", "Data Origin"),
              ("D2", "resource_tier", "Linguistic Resource Tier"),
              ("D3", "task_class", "Task Class"),
              ("D4", "cultural_specificity", "Cultural Specificity"),
              ("D5", "inference_paradigm", "Inference Paradigm")]
INT_FIELDS = {"year", "language_count"}
CONFIDENCE = ["high", "medium", "low"]
# Columns that are identical in every round file and therefore stored once,
# at resource level, rather than in each round record.
ROUND_INVARIANT = {"resource_id", "resource_name", "coder", "screening_status"}

ADDED_FIELDS = [
    ("language_names", "array of string | null",
     "Derived from `languages`: the entries that name a language, in source order "
     "(entries such as '75 languages' that only state a count are left out). "
     "null when `languages` is null."),
    ("languages_fully_named", "boolean",
     "true when `language_names` names every covered language: no count-only entries "
     "and as many names as `language_count`."),
    ("consensus", "object",
     "Per label field, how the five rounds voted: the consensus value, its status "
     "(consensus | no_majority | not_coded), the supporting vote count, the number of "
     "rounds that coded the cell, whether the rounds split, and every vote with the "
     "rounds that cast it."),
    ("rounds", "array of object",
     "The five independent coding rounds' own records for this resource (A-E), with "
     "labels in canonical form and as written."),
]


class BuildError(Exception):
    pass


# ---------------------------------------------------------------- sources

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(rel):
    with open(ROOT / rel, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def read_cff(rel):
    """Read the subset of CFF used by CITATION.cff (scalars, lists, folded text)."""
    data, key, block = {}, None, None
    for line in (ROOT / rel).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if block is not None:
            if line.startswith("  ") and not line.lstrip().startswith("- "):
                block.append(line.strip())
                continue
            data[key] = " ".join(block)
            block = None
        if not line.startswith(" "):
            key, _, value = line.partition(":")
            value = value.split(" #")[0].strip()
            if value == ">-":
                block = []
            elif value:
                data[key] = value.strip('"')
            else:
                data[key] = []
        elif line.lstrip().startswith("- "):
            item = line.lstrip()[2:]
            if ":" in item and not item.startswith("http"):
                k, _, v = item.partition(":")
                data[key].append({k.strip(): v.strip().strip('"')})
            else:
                data[key].append(item.strip().strip('"'))
        else:
            k, _, v = line.strip().partition(":")
            data[key][-1][k.strip()] = v.strip().strip('"')
    if block is not None:
        data[key] = " ".join(block)
    return data


def read_dictionary(rel):
    """Field rows of data-dictionary.md's tables: {field: (type, description, tokens)}."""
    fields = {}
    for line in (ROOT / rel).read_text(encoding="utf-8").splitlines():
        if not line.startswith("| `"):
            continue
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", line)[1:-1]]
        name = cells[0].strip("`")
        tokens = re.findall(r"`([^`]+)`", " ".join(cells[1:]))

        def clean(text):
            return text.replace("\\|", "|").replace("`", "")
        fields[name] = (clean(cells[1]), clean(cells[2]), tokens, cells[2])
    return fields


# ------------------------------------------------------------ vocabularies

def build_vocabularies(dictionary):
    """{field: [{value, applies_to, listed_in_data_dictionary}]} in canonical order."""
    vocab = {}
    for field in LABEL_FIELDS:
        if field in build_consensus.CANON:
            values = list(build_consensus.CANON[field].values())
        else:  # inference_paradigm: not voted by build_consensus.py
            values = [t for t in dictionary[field][2] if t != "null"]
        listed = set(dictionary[field][2])
        applies = {}
        if field == "task_class":
            m = re.search(r"Benchmarks:(.*?)\.\s*Datasets:(.*?)\.", dictionary[field][3])
            for group, typ in ((m.group(1), "Benchmark"), (m.group(2), "Dataset")):
                for token in re.findall(r"`([^`]+)`", group):
                    applies[token] = [typ]
        vocab[field] = [{"value": v,
                         "applies_to": applies.get(v, ["Benchmark", "Dataset"]
                                                   if field != "task_class" else None),
                         "listed_in_data_dictionary": v in listed}
                        for v in values]
    return vocab


def canonical(field, raw, vocab):
    """Canonical label for a raw cell, or None when undetermined."""
    v = build_consensus.coded({field: raw}, field)
    if v is None:
        return None
    for entry in vocab[field]:
        if entry["value"].lower() == v:
            return entry["value"]
    raise BuildError(f"{field}: value {raw!r} is outside the controlled vocabulary")


# ------------------------------------------------------------- converters

def text(value):
    return value if value != "" else None


def integer(value, where):
    if value == "":
        return None
    if not re.fullmatch(r"-?\d+", value):
        raise BuildError(f"{where}: {value!r} is not an integer")
    return int(value)


def split_languages(value):
    """Split a `languages` cell on semicolons outside parentheses."""
    parts, depth, cur = [], 0, ""
    for ch in value:
        depth += (ch == "(") - (ch == ")")
        if ch == ";" and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    parts.append(cur.strip())
    return [p for p in parts if p]


def is_count_only(entry):
    return entry[0].isdigit() or re.search(r"\blanguages\b", entry, re.I) is not None


# --------------------------------------------------------------- consensus

def decide(field, rows_by_round, vocab):
    order = {e["value"]: i for i, e in enumerate(vocab[field])}
    votes = {}
    for letter in ROUNDS:
        value = canonical(field, rows_by_round[letter][field], vocab)
        if value is not None:
            votes.setdefault(value, []).append(letter)
    ranked = sorted(votes.items(), key=lambda kv: (-len(kv[1]), order[kv[0]]))
    cast = sum(len(rs) for rs in votes.values())
    support = len(ranked[0][1]) if ranked else 0
    if not cast:
        status, value = "not_coded", None
    elif support >= MAJORITY:
        status, value = "consensus", ranked[0][0]
    else:
        status, value = "no_majority", None
    return {"value": value, "status": status, "support": support,
            "rounds_coded": cast, "split": len(votes) > 1,
            "votes": [{"value": v, "count": len(rs), "rounds": rs} for v, rs in ranked]}


def check_corpus_is_consensus():
    """data/corpus.csv must be byte-identical to build_consensus.py's output."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "corpus.csv"
        cmd = [sys.executable, str(SCRIPTS / "build_consensus.py"), "--rounds",
               *[f"{L}={ROOT / ROUND_FILE.format(L)}" for L in ROUNDS],
               "--base", str(ROOT / ROUND_FILE.format("A")), "--out", str(out)]
        subprocess.run(cmd, check=True, capture_output=True)
        if out.read_bytes() != (ROOT / CORPUS).read_bytes():
            raise BuildError(f"{CORPUS} is not the output of build_consensus.py over "
                             "the five rounds; rebuild it before building the release")


# -------------------------------------------------------------- documents

def build_document():
    for rel in [CORPUS, SCREENING, DICTIONARY, CITATION, SUMMARY,
                *[ROUND_FILE.format(L) for L in ROUNDS]]:
        if not (ROOT / rel).is_file():
            raise BuildError(f"missing source file {rel}")
    check_corpus_is_consensus()

    dictionary = read_dictionary(DICTIONARY)
    cff = read_cff(CITATION)
    vocab = build_vocabularies(dictionary)
    columns, corpus = read_csv(CORPUS)
    rounds = {}
    for letter in ROUNDS:
        cols, rows = read_csv(ROUND_FILE.format(letter))
        if cols != columns:
            raise BuildError(f"round {letter}: columns differ from {CORPUS}")
        if [r["resource_id"] for r in rows] != [r["resource_id"] for r in corpus]:
            raise BuildError(f"round {letter}: resource_ids differ from {CORPUS}")
        rounds[letter] = {r["resource_id"]: r for r in rows}
    if len({r["resource_id"] for r in corpus}) != len(corpus):
        raise BuildError(f"{CORPUS}: duplicate resource_id")

    resources = []
    for row in corpus:
        rid = row["resource_id"]
        by_round = {L: rounds[L][rid] for L in ROUNDS}
        for letter, rr in by_round.items():
            for col in ROUND_INVARIANT - {"coder"}:
                if rr[col] != row[col]:
                    raise BuildError(f"{rid} round {letter}: {col} differs from {CORPUS}")
            if rr["coder"] != letter:
                raise BuildError(f"{rid} round {letter}: coder is {rr['coder']!r}")
        consensus = {f: decide(f, by_round, vocab) for f in LABEL_FIELDS}
        for f in LABEL_FIELDS:
            if (consensus[f]["value"] or "") != row[f]:
                raise BuildError(f"{rid} {f}: recomputed vote {consensus[f]['value']!r} "
                                 f"!= {CORPUS} value {row[f]!r}")
        if row["confidence"] not in CONFIDENCE:
            raise BuildError(f"{rid}: confidence {row['confidence']!r}")

        languages = text(row["languages"])
        entries = split_languages(languages) if languages else []
        names = [e for e in entries if not is_count_only(e)]
        count = integer(row["language_count"], f"{rid} language_count")
        record = {}
        for col in columns:
            if col in INT_FIELDS:
                record[col] = integer(row[col], f"{rid} {col}")
            else:
                record[col] = text(row[col])
            if col == "languages":
                record["language_names"] = names if languages else None
                record["languages_fully_named"] = bool(
                    names and len(names) == len(entries) and count == len(names))
        record["consensus"] = consensus
        record["rounds"] = [round_record(rid, L, by_round[L], columns, vocab)
                            for L in ROUNDS]
        resources.append(record)

    screen_cols, screen_rows = read_csv(SCREENING)
    candidates = [{c: (integer(r[c], f"screening {c}") if c == "year" else text(r[c]))
                   for c in screen_cols} for r in screen_rows]

    doc = {
        "$schema": f"./{SCHEMA_FILE}",
        "name": "LinguaGapAtlas",
        "title": cff["title"],
        "version": cff["version"],
        "description": cff["abstract"],
        "keywords": cff["keywords"],
        "authors": [{"given_names": a.get("given-names"), "family_names": a.get("family-names"),
                     "orcid": a.get("orcid")} for a in cff["authors"]],
        "citation": citation_text(cff),
        "paper": {"title": PAPER_TITLE, "status": PAPER_STATUS},
        "repository": REPOSITORY_URL,
        "license": {"data": cff["license"], "data_url": LICENSE_URLS[cff["license"]],
                    "code": "MIT", "code_url": LICENSE_URLS["MIT"],
                    "note": "The benchmarks and datasets classified here remain under their "
                            "creators' own licenses; this release contains only classification "
                            "metadata about them and links to their sources."},
        "null_rule": ("null means undetermined: the value could not be determined from the "
                      "resource's public documentation (or, for a consensus label, the rounds did "
                      "not reach a majority). Undetermined values are never imputed; exclude them "
                      "from a statistic's denominator and report their count."),
        "sources": [{"path": rel, "sha256": sha256(ROOT / rel)}
                    for rel in [CORPUS, *[ROUND_FILE.format(L) for L in ROUNDS],
                                SCREENING, DICTIONARY, CITATION]],
        "generator": "scripts/build_release.py",
        "fields": field_docs(columns, dictionary),
        "taxonomy": {
            "type": vocab["type"],
            "dimensions": [{"id": did, "field": f, "name": name,
                            "definition": dictionary[f][1],
                            "values": vocab[f]} for did, f, name in DIMENSIONS],
            "confidence": CONFIDENCE,
        },
        "coding": coding_section(),
        "statistics": statistics(corpus, rounds, resources, vocab),
        "caveats": [],
        "screening_log": {
            "source": SCREENING,
            "columns": screen_cols,
            "candidate_count": len(candidates),
            "candidates": candidates,
        },
        "resources": resources,
    }
    doc["caveats"] = caveats(doc, vocab)
    return doc


def round_record(rid, letter, row, columns, vocab):
    rec = {"round": letter, "resource_id": rid}
    for col in columns:
        if col in ROUND_INVARIANT:
            continue
        if col in LABEL_FIELDS:
            rec[col] = canonical(col, row[col], vocab)
        elif col in INT_FIELDS:
            rec[col] = integer(row[col], f"{rid} round {letter} {col}")
        else:
            rec[col] = text(row[col])
    rec["labels_as_written"] = {f: text(row[f]) for f in LABEL_FIELDS}
    return rec


def citation_text(cff):
    names = [f"{a['family-names']}, {' '.join(p[0] + '.' for p in a['given-names'].replace('.', '').split())}"
             for a in cff["authors"]]
    who = ", ".join(names[:-1]) + ", & " + names[-1]
    return (f"{who} {cff['title']} (corpus and reproducibility package, "
            f"v{cff['version']}). {REPOSITORY_URL}")


def field_docs(columns, dictionary):
    docs = []
    for col in columns:
        typ, desc = dictionary[col][0], dictionary[col][1]
        if col in INT_FIELDS:
            typ = "integer | null"
        docs.append({"field": col, "type": typ, "description": desc,
                     "source": DICTIONARY})
        if col == "languages":
            docs += [{"field": f, "type": t, "description": d, "source": "derived"}
                     for f, t, d in ADDED_FIELDS[:2]]
    docs += [{"field": f, "type": t, "description": d, "source": "derived"}
             for f, t, d in ADDED_FIELDS[2:]]
    return docs


def coding_section():
    return {
        "design": ("Five independent blind coding rounds (A-E), each coding all 96 resources "
                   "from primary sources under the same codebook (rounds-summary.md)."),
        "rounds": [{"round": L, "source": ROUND_FILE.format(L),
                    "sha256": sha256(ROOT / ROUND_FILE.format(L))} for L in ROUNDS],
        "label_normalization": {
            "case_insensitive": True,
            "synonyms": {k: build_consensus.CANON["task_class"][v]
                         for k, v in build_consensus.SYN.items()},
            "null_tokens": sorted(build_consensus.NULL),
        },
        "consensus_rules": {
            "labels": (f"type, data_origin, resource_tier, task_class and cultural_specificity "
                       f"take a value only when at least {MAJORITY} of the 5 rounds agree on it "
                       "after normalization; otherwise the cell is null. inference_paradigm is "
                       "not voted on (every round left it undetermined)."),
            "year": ("The most common year when at least 2 rounds report it, otherwise "
                     "round A's year."),
            "language_count": "The modal reported count; on a tie, the median reported count.",
            "languages": ("The longest list among the rounds that reported the chosen count; "
                          "round A's value when none did."),
            "confidence": ("high when no voted cell is split or unresolved, medium for 1-2 "
                           "such cells, low for 3 or more."),
            "scaffold": ("canonical_reference, provenance_url, the screening fields and the "
                         "evidence quoted in evidence_note are round A's (the base round)."),
        },
        "generator": "scripts/build_consensus.py",
    }


def mean(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else None


def statistics(corpus, rounds, resources, vocab):
    types = [e["value"] for e in vocab["type"]]
    composition = {t: sum(r["type"] == t for r in corpus) for t in types}
    composition["total"] = len(corpus)

    def shares(rows, typ):
        out = []
        for label, column, target in recompute.CHECKS:
            hits, coded, undetermined = recompute.share(rows, column, target)
            out.append({"type": typ, "statistic": label, "field": column, "target": target,
                        "hits": hits, "coded": coded, "undetermined": undetermined,
                        "share": round(hits / coded, 4) if coded else None})
        return out

    headline = [s for t in types for s in shares([r for r in corpus if r["type"] == t], t)]
    per_round = []
    for L in ROUNDS:
        rows = list(rounds[L].values())
        for t in types:
            group = [r for r in rows if canonical("type", r["type"], vocab) == t]
            per_round += [{"round": L, **s} for s in shares(group, t)]

    distributions = {}
    for t in types:
        group = [r for r in resources if r["type"] == t]
        distributions[t] = {}
        for _, f, _ in DIMENSIONS:
            counts = Counter(r[f] for r in group)
            distributions[t][f] = {
                "values": {e["value"]: counts[e["value"]] for e in vocab[f]},
                "undetermined": counts[None]}

    coverage = {}
    for f in LABEL_FIELDS:
        statuses = Counter(r["consensus"][f]["status"] for r in resources)
        coverage[f] = {s: statuses[s] for s in ("consensus", "no_majority", "not_coded")}
        coverage[f]["split"] = sum(r["consensus"][f]["split"] for r in resources)

    agree = {}
    for f in LABEL_FIELDS:
        pairs_out = []
        for a, b in combinations(ROUNDS, 2):
            pairs = [(canonical(f, rounds[a][r["resource_id"]][f], vocab),
                      canonical(f, rounds[b][r["resource_id"]][f], vocab)) for r in corpus]
            pairs = [p for p in pairs if p[0] is not None and p[1] is not None]
            kappa = agreement.cohens_kappa(pairs)
            pairs_out.append({
                "rounds": f"{a}-{b}", "pairs": len(pairs),
                "observed": round(sum(x == y for x, y in pairs) / len(pairs), 4) if pairs else None,
                "kappa": round(kappa, 4) if kappa is not None else None})
        agree[f] = {"mean_pairwise_kappa": mean(p["kappa"] for p in pairs_out),
                    "mean_pairwise_observed": mean(p["observed"] for p in pairs_out),
                    "pairwise": pairs_out}

    return {
        "note": ("Computed from this release's labels; shares exclude undetermined rows from "
                 "the denominator (scripts/recompute.py). Agreement is Cohen's kappa over the "
                 "10 round pairs on normalized labels, counting only resources both rounds coded."),
        "composition": composition,
        "headline_shares": headline,
        "headline_shares_by_round": per_round,
        "distributions": distributions,
        "consensus_coverage": coverage,
        "agreement": agree,
    }


def caveats(doc, vocab):
    stats = doc["statistics"]
    comp = stats["composition"]
    head = {(s["type"], s["field"]): s for s in stats["headline_shares"]}

    def pct(typ, field):
        s = head[(typ, field)]
        return f"{100 * s['hits'] / s['coded']:.1f}%" if s["coded"] else "undefined"

    tier = stats["distributions"]
    tier_coded = sum(sum(tier[t]["resource_tier"]["values"].values()) for t in tier)
    tier_low = sum(tier[t]["resource_tier"]["values"]["Low"] for t in tier)
    unlisted = [e["value"] for e in vocab["task_class"] if not e["listed_in_data_dictionary"]]
    used = {r["task_class"] for r in doc["resources"]}
    count_only = sum(1 for r in doc["resources"]
                     if r["languages"] and not r["languages_fully_named"])
    no_languages = sum(1 for r in doc["resources"] if r["languages"] is None)
    return [
        {"id": "ai-coders", "topic": "coding",
         "text": ("rounds-summary.md states: \"The second coders are AI agents; the paper must "
                  "disclose this as AI second-coding / human–AI agreement, not two-human IAA.\" "
                  "This package does not record which rounds were coded by whom.")},
        {"id": "differs-from-paper", "topic": "statistics",
         "text": (f"The consensus corpus has {comp['Benchmark']} benchmarks and "
                  f"{comp['Dataset']} datasets; the paper describes 65 and 31. Shares computed "
                  f"from this release for benchmarks (translated {pct('Benchmark', 'data_origin')}, "
                  f"Universal {pct('Benchmark', 'cultural_specificity')}, Low tier "
                  f"{pct('Benchmark', 'resource_tier')} of tier-coded rows) are not the published "
                  "26.2%, 55.4% and 6.2% (see rounds-summary.md). Figures in this release are "
                  "computed from the released labels.")},
        {"id": "tier-sparse", "topic": "resource_tier",
         "text": (f"resource_tier is coded for {tier_coded} of {comp['total']} resources "
                  f"({tier_low} Low, none Medium or High); the rest are undetermined because "
                  "sources rarely state a tier. The low-resource share is not estimable from this "
                  "release, and tier agreement (kappa 1.0 on a handful of jointly coded pairs) is "
                  "not a reliability estimate.")},
        {"id": "inference-paradigm-uncoded", "topic": "inference_paradigm",
         "text": ("inference_paradigm (D5) is null for every resource in the consensus and in "
                  "every round: the codebook does not enumerate its values "
                  "(methods/codebook-draft.md, D5), and the vocabulary listed in "
                  "data-dictionary.md is provisional.")},
        {"id": "screening-log-empty", "topic": "screening",
         "text": (f"{SCREENING} contains {doc['screening_log']['candidate_count']} candidate "
                  "rows: the 142-candidate screening record (46 exclusions) that "
                  "data-dictionary.md describes is not in this release. An empty candidate list "
                  "is not evidence that no candidates were excluded.")},
        {"id": "task-class-vocabulary", "topic": "task_class",
         "text": (f"task_class values {', '.join(v for v in unlisted if v in used)} occur in the "
                  "data but are not in data-dictionary.md's listed vocabulary (they belong to the "
                  "13-class set in methods/codebook-draft.md, ambiguity A6). Each vocabulary "
                  "entry carries listed_in_data_dictionary.")},
        {"id": "round-a-scaffold", "topic": "provenance",
         "text": ("At resource level, canonical_reference, provenance_url and the evidence quoted "
                  "in evidence_note are round A's (the consensus build's base round). Every "
                  "round's own reference, URL, evidence note and confidence is under rounds.")},
        {"id": "consensus-confidence", "topic": "confidence",
         "text": ("Resource-level confidence is derived from the vote (high: no split or "
                  "unresolved voted cell; medium: 1-2; low: 3 or more), not recorded by a coder. "
                  "Each round's own confidence is under rounds.")},
        {"id": "languages-count-only", "topic": "languages",
         "text": (f"{count_only} resources document their languages only partly or only as a "
                  "count (e.g. '75 languages'); for these, language_names holds just the "
                  "languages actually named and languages_fully_named is false. A further "
                  f"{no_languages} have no documented language list (languages and "
                  "language_names are null). language_count is as documented, so a value such "
                  "as 250 for '250+ languages' is a lower bound.")},
        {"id": "codebook-draft", "topic": "codebook",
         "text": ("The codebook (methods/codebook-draft.md) is a draft. Adjudication of the split "
                  "cells and the decision rules it is meant to produce are pending "
                  "(methods/iaa-protocol.md).")},
    ]


# ------------------------------------------------------------ JSON Schema

def build_schema(doc, vocab):
    def enum(field):
        return {"type": ["string", "null"], "enum": [e["value"] for e in vocab[field]] + [None]}

    s_text = {"type": ["string", "null"]}
    s_int = {"type": ["integer", "null"]}
    resource_props = {}
    for col in [f["field"] for f in doc["fields"]]:
        if col == "resource_id":
            s = {"$ref": "#/$defs/resource_id"}
        elif col == "resource_name":
            s = {"type": "string"}
        elif col == "type":
            s = {"enum": [e["value"] for e in vocab["type"]]}
        elif col in LABEL_FIELDS:
            s = enum(col)
        elif col in INT_FIELDS:
            s = s_int
        elif col == "confidence":
            s = {"enum": CONFIDENCE}
        elif col == "screening_status":
            s = {"const": "included"}
        elif col == "language_names":
            s = {"type": ["array", "null"], "items": {"type": "string"}}
        elif col == "languages_fully_named":
            s = {"type": "boolean"}
        elif col == "consensus":
            s = {"type": "object", "additionalProperties": False, "required": LABEL_FIELDS,
                 "properties": {f: {"$ref": f"#/$defs/decision_{f}"} for f in LABEL_FIELDS}}
        elif col == "rounds":
            s = {"type": "array", "minItems": len(ROUNDS), "maxItems": len(ROUNDS),
                 "items": {"$ref": "#/$defs/round"}}
        else:
            s = s_text
        desc = next(f["description"] for f in doc["fields"] if f["field"] == col)
        resource_props[col] = {**s, "description": desc}

    round_props = {}
    for col in doc["resources"][0]["rounds"][0]:
        if col == "round":
            s = {"enum": ROUNDS}
        elif col == "resource_id":
            s = {"$ref": "#/$defs/resource_id"}
        elif col in LABEL_FIELDS:
            s = enum(col)
        elif col in INT_FIELDS:
            s = s_int
        elif col == "confidence":
            s = {"type": ["string", "null"], "enum": CONFIDENCE + [None]}
        elif col == "labels_as_written":
            s = {"type": "object", "additionalProperties": False, "required": LABEL_FIELDS,
                 "properties": {f: s_text for f in LABEL_FIELDS},
                 "description": "The round's label strings exactly as written in its file."}
        else:
            s = s_text
        round_props[col] = s

    defs = {
        "resource_id": {"type": "string", "pattern": r"^QG-\d{3}$"},
        "resource": {"type": "object", "additionalProperties": False,
                     "required": list(resource_props), "properties": resource_props},
        "round": {"type": "object", "additionalProperties": False,
                  "required": list(round_props), "properties": round_props},
    }
    for f in LABEL_FIELDS:
        values = [e["value"] for e in vocab[f]]
        defs[f"decision_{f}"] = {
            "type": "object", "additionalProperties": False,
            "required": ["value", "status", "support", "rounds_coded", "split", "votes"],
            "properties": {
                "value": {"enum": values + [None]},
                "status": {"enum": ["consensus", "no_majority", "not_coded"]},
                "support": {"type": "integer", "minimum": 0, "maximum": len(ROUNDS)},
                "rounds_coded": {"type": "integer", "minimum": 0, "maximum": len(ROUNDS)},
                "split": {"type": "boolean"},
                "votes": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["value", "count", "rounds"],
                    "properties": {"value": {"enum": values},
                                   "count": {"type": "integer", "minimum": 1},
                                   "rounds": {"type": "array", "uniqueItems": True,
                                              "items": {"enum": ROUNDS}}}}}}}

    top_required = [k for k in doc if k != "$schema"]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"{REPOSITORY_URL}/release/{SCHEMA_FILE}",
        "title": f"{doc['name']} release document",
        "description": (f"Schema for {JSON_FILE} (the whole document) and, via "
                        f"#/$defs/resource, for each line of {JSONL_FILE}."),
        "type": "object",
        "required": top_required,
        "properties": {
            "$schema": {"type": "string"},
            "resources": {"type": "array", "items": {"$ref": "#/$defs/resource"}},
            **{k: {} for k in top_required if k != "resources"},
        },
        "$defs": defs,
    }


# ------------------------------------------------------------- Croissant

CROISSANT_CONTEXT = {
    "@language": "en", "@vocab": "https://schema.org/", "citeAs": "cr:citeAs",
    "column": "cr:column", "conformsTo": "dct:conformsTo",
    "cr": "http://mlcommons.org/croissant/", "rai": "http://mlcommons.org/croissant/RAI/",
    "data": {"@id": "cr:data", "@type": "@json"},
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"}, "dct": "http://purl.org/dc/terms/",
    "equivalentProperty": "cr:equivalentProperty",
    "examples": {"@id": "cr:examples", "@type": "@json"}, "extract": "cr:extract",
    "field": "cr:field", "fileProperty": "cr:fileProperty", "fileObject": "cr:fileObject",
    "fileSet": "cr:fileSet", "format": "cr:format", "includes": "cr:includes",
    "isLiveDataset": "cr:isLiveDataset", "jsonPath": "cr:jsonPath", "key": "cr:key",
    "md5": "cr:md5", "parentField": "cr:parentField", "path": "cr:path",
    "recordSet": "cr:recordSet", "references": "cr:references", "regex": "cr:regex",
    "repeated": "cr:repeated", "replace": "cr:replace", "samplingRate": "cr:samplingRate",
    "sc": "https://schema.org/",
    "separator": "cr:separator", "source": "cr:source", "subField": "cr:subField",
    "transform": "cr:transform",
}


def build_croissant(doc, hashes):
    descs = {f["field"]: f["description"] for f in doc["fields"]}
    scalar = [c for c in descs if c not in ("consensus", "rounds")]

    def dtype(col):
        if col in INT_FIELDS:
            return "sc:Integer"
        if col == "languages_fully_named":
            return "sc:Boolean"
        if col == "provenance_url":
            return "sc:URL"
        return "sc:Text"

    def field(rs, col, desc, source, **extra):
        return {"@type": "cr:Field", "@id": f"{rs}/{col}", "name": col, "description": desc,
                "dataType": dtype(col), **extra, "source": source}

    resource_fields = [
        field("resources", c, descs[c],
              {"fileObject": {"@id": JSONL_FILE}, "extract": {"column": c}},
              **({"repeated": True} if c == "language_names" else {}))
        for c in scalar]
    round_cols = [c for c in doc["resources"][0]["rounds"][0] if c != "labels_as_written"]
    round_fields = [
        field("round_labels", c,
              "Coding round (A-E)." if c == "round" else
              f"This round's {c}." + (" Canonical label." if c in LABEL_FIELDS else ""),
              {"fileObject": {"@id": JSON_FILE},
               "extract": {"jsonPath": f"$.resources[*].rounds[*].{c}"}},
              **({"references": {"field": {"@id": "resources/resource_id"}}}
                 if c == "resource_id" else {}))
        for c in round_cols]
    return {
        "@context": CROISSANT_CONTEXT,
        "@type": "sc:Dataset",
        "name": doc["name"],
        "description": doc["description"],
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "citeAs": doc["citation"],
        "creator": [{"@type": "sc:Person", "name": f"{a['given_names']} {a['family_names']}",
                     **({"sameAs": a["orcid"]} if a["orcid"] else {})}
                    for a in doc["authors"]],
        "keywords": doc["keywords"],
        "license": doc["license"]["data_url"],
        "url": doc["repository"],
        "version": doc["version"],
        "distribution": [
            {"@type": "cr:FileObject", "@id": JSONL_FILE, "name": JSONL_FILE,
             "description": "One resource per line: consensus labels, votes and the five rounds.",
             "contentUrl": JSONL_FILE, "encodingFormat": "application/jsonlines",
             "sha256": hashes[JSONL_FILE]},
            {"@type": "cr:FileObject", "@id": JSON_FILE, "name": JSON_FILE,
             "description": "The whole atlas as one JSON document (metadata, taxonomy, "
                            "statistics, caveats and all resources).",
             "contentUrl": JSON_FILE, "encodingFormat": "application/json",
             "sha256": hashes[JSON_FILE]},
            {"@type": "cr:FileObject", "@id": SQLITE_FILE, "name": SQLITE_FILE,
             "description": "The same content as SQLite tables and views.",
             "contentUrl": SQLITE_FILE, "encodingFormat": "application/vnd.sqlite3",
             "sha256": hashes[SQLITE_FILE]},
        ],
        "recordSet": [
            {"@type": "cr:RecordSet", "@id": "resources", "name": "resources",
             "description": "The 96 resources with their consensus labels (null = undetermined).",
             "key": {"@id": "resources/resource_id"}, "field": resource_fields},
            {"@type": "cr:RecordSet", "@id": "round_labels", "name": "round_labels",
             "description": "Each coding round's record for each resource (5 per resource).",
             "key": [{"@id": "round_labels/resource_id"}, {"@id": "round_labels/round"}],
             "field": round_fields},
        ],
    }


# ---------------------------------------------------------------- SQLite

SQL_SCHEMA = """
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE fields (field TEXT PRIMARY KEY, type TEXT, description TEXT, source TEXT);
CREATE TABLE vocabulary (
  field TEXT NOT NULL, value TEXT NOT NULL, position INTEGER NOT NULL,
  applies_to TEXT, listed_in_data_dictionary INTEGER NOT NULL,
  PRIMARY KEY (field, value));
CREATE TABLE resources (
  resource_id TEXT PRIMARY KEY, resource_name TEXT NOT NULL, canonical_reference TEXT,
  type TEXT NOT NULL CHECK (type IN ('Benchmark', 'Dataset')), year INTEGER,
  languages TEXT, languages_fully_named INTEGER NOT NULL, language_count INTEGER,
  data_origin TEXT, resource_tier TEXT, task_class TEXT, cultural_specificity TEXT,
  inference_paradigm TEXT, provenance_url TEXT, evidence_note TEXT, coder TEXT,
  confidence TEXT, screening_status TEXT NOT NULL, screening_reason TEXT);
CREATE TABLE resource_languages (
  resource_id TEXT NOT NULL REFERENCES resources (resource_id), position INTEGER NOT NULL,
  language TEXT NOT NULL, PRIMARY KEY (resource_id, position));
CREATE TABLE rounds (round TEXT PRIMARY KEY, source TEXT NOT NULL, sha256 TEXT NOT NULL);
CREATE TABLE round_labels (
  resource_id TEXT NOT NULL REFERENCES resources (resource_id),
  round TEXT NOT NULL REFERENCES rounds (round), type TEXT, year INTEGER, languages TEXT,
  language_count INTEGER, data_origin TEXT, resource_tier TEXT, task_class TEXT,
  cultural_specificity TEXT, inference_paradigm TEXT, canonical_reference TEXT,
  provenance_url TEXT, evidence_note TEXT, confidence TEXT, screening_reason TEXT,
  PRIMARY KEY (resource_id, round));
CREATE TABLE round_labels_as_written (
  resource_id TEXT NOT NULL, round TEXT NOT NULL, field TEXT NOT NULL, value TEXT NOT NULL,
  PRIMARY KEY (resource_id, round, field),
  FOREIGN KEY (resource_id, round) REFERENCES round_labels (resource_id, round));
CREATE TABLE consensus (
  resource_id TEXT NOT NULL REFERENCES resources (resource_id), field TEXT NOT NULL,
  value TEXT, status TEXT NOT NULL, support INTEGER NOT NULL, rounds_coded INTEGER NOT NULL,
  split INTEGER NOT NULL, PRIMARY KEY (resource_id, field));
CREATE TABLE consensus_votes (
  resource_id TEXT NOT NULL, field TEXT NOT NULL, value TEXT NOT NULL,
  count INTEGER NOT NULL, rounds TEXT NOT NULL, PRIMARY KEY (resource_id, field, value),
  FOREIGN KEY (resource_id, field) REFERENCES consensus (resource_id, field));
CREATE TABLE caveats (id TEXT PRIMARY KEY, topic TEXT NOT NULL, text TEXT NOT NULL);
CREATE TABLE screening_candidates (
  candidate_id TEXT PRIMARY KEY, resource_name TEXT, identified_via TEXT, year INTEGER,
  type TEXT, screening_stage TEXT, decision TEXT, exclusion_reason TEXT, screener TEXT,
  notes TEXT);
CREATE VIEW headline_shares AS
  SELECT type, 'translated-data share' AS statistic, 'data_origin' AS field,
         'Translated' AS target, COALESCE(SUM(data_origin = 'Translated'), 0) AS hits,
         COUNT(data_origin) AS coded, SUM(data_origin IS NULL) AS undetermined,
         ROUND(100.0 * SUM(data_origin = 'Translated') / NULLIF(COUNT(data_origin), 0), 1)
           AS percent
    FROM resources GROUP BY type
  UNION ALL
  SELECT type, 'culturally non-specific share', 'cultural_specificity', 'Universal',
         COALESCE(SUM(cultural_specificity = 'Universal'), 0), COUNT(cultural_specificity),
         SUM(cultural_specificity IS NULL),
         ROUND(100.0 * SUM(cultural_specificity = 'Universal')
               / NULLIF(COUNT(cultural_specificity), 0), 1)
    FROM resources GROUP BY type
  UNION ALL
  SELECT type, 'low-resource share', 'resource_tier', 'Low',
         COALESCE(SUM(resource_tier = 'Low'), 0), COUNT(resource_tier),
         SUM(resource_tier IS NULL),
         ROUND(100.0 * SUM(resource_tier = 'Low') / NULLIF(COUNT(resource_tier), 0), 1)
    FROM resources GROUP BY type;
CREATE VIEW split_cells AS
  SELECT c.resource_id, r.resource_name, c.field, c.value AS consensus_value, c.status,
         c.support, c.rounds_coded,
         (SELECT GROUP_CONCAT(v.value || ':' || v.count, ', ') FROM
            (SELECT value, count FROM consensus_votes cv
              WHERE cv.resource_id = c.resource_id AND cv.field = c.field
              ORDER BY count DESC, value) v) AS votes
    FROM consensus c JOIN resources r USING (resource_id)
   WHERE c.split = 1;
"""


def write_sqlite(doc, path):
    con = sqlite3.connect(path)
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SQL_SCHEMA)
    meta = {k: doc[k] for k in ("name", "title", "version", "description", "citation",
                                "repository", "null_rule", "generator")}
    meta.update({"paper_title": doc["paper"]["title"], "license_data": doc["license"]["data"],
                 "license_code": doc["license"]["code"], "license_note": doc["license"]["note"],
                 "authors": "; ".join(f"{a['given_names']} {a['family_names']}"
                                      for a in doc["authors"])})
    meta.update({f"source_sha256:{s['path']}": s["sha256"] for s in doc["sources"]})
    con.executemany("INSERT INTO metadata VALUES (?, ?)", sorted(meta.items()))
    con.executemany("INSERT INTO fields VALUES (?, ?, ?, ?)",
                    [(f["field"], f["type"], f["description"], f["source"])
                     for f in doc["fields"]])
    vocab = {"type": doc["taxonomy"]["type"],
             **{d["field"]: d["values"] for d in doc["taxonomy"]["dimensions"]}}
    con.executemany("INSERT INTO vocabulary VALUES (?, ?, ?, ?, ?)",
                    [(f, e["value"], i, ",".join(e["applies_to"]) if e["applies_to"] else None,
                      int(e["listed_in_data_dictionary"]))
                     for f, entries in vocab.items() for i, e in enumerate(entries)])
    con.executemany("INSERT INTO rounds VALUES (?, ?, ?)",
                    [(r["round"], r["source"], r["sha256"]) for r in doc["coding"]["rounds"]])
    res_cols = [c[1] for c in con.execute("PRAGMA table_info(resources)")]
    rl_cols = [c[1] for c in con.execute("PRAGMA table_info(round_labels)")]
    for r in doc["resources"]:
        con.execute(f"INSERT INTO resources VALUES ({', '.join('?' * len(res_cols))})",
                    [int(r[c]) if c == "languages_fully_named" else r[c] for c in res_cols])
        con.executemany("INSERT INTO resource_languages VALUES (?, ?, ?)",
                        [(r["resource_id"], i + 1, n)
                         for i, n in enumerate(r["language_names"] or [])])
        for f, d in r["consensus"].items():
            con.execute("INSERT INTO consensus VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (r["resource_id"], f, d["value"], d["status"], d["support"],
                         d["rounds_coded"], int(d["split"])))
            con.executemany("INSERT INTO consensus_votes VALUES (?, ?, ?, ?, ?)",
                            [(r["resource_id"], f, v["value"], v["count"],
                              ",".join(v["rounds"])) for v in d["votes"]])
        for rr in r["rounds"]:
            con.execute(f"INSERT INTO round_labels VALUES ({', '.join('?' * len(rl_cols))})",
                        [rr[c] for c in rl_cols])
            con.executemany("INSERT INTO round_labels_as_written VALUES (?, ?, ?, ?)",
                            [(r["resource_id"], rr["round"], f, v)
                             for f, v in rr["labels_as_written"].items() if v is not None])
    con.executemany("INSERT INTO caveats VALUES (?, ?, ?)",
                    [(c["id"], c["topic"], c["text"]) for c in doc["caveats"]])
    cols = doc["screening_log"]["columns"]
    if doc["screening_log"]["candidates"]:
        con.executemany(f"INSERT INTO screening_candidates ({', '.join(cols)}) "
                        f"VALUES ({', '.join('?' * len(cols))})",
                        [[c[k] for k in cols] for c in doc["screening_log"]["candidates"]])
    problems = con.execute("PRAGMA foreign_key_check").fetchall()
    if problems:
        raise BuildError(f"SQLite foreign key violations: {problems[:5]}")
    con.commit()
    con.execute("VACUUM")
    con.close()


# ------------------------------------------------------------------ main

def dumps(obj):
    return json.dumps(obj, ensure_ascii=False, indent=2) + "\n"


def render(out_dir):
    """Write every release file into out_dir; return {file: sha256}."""
    doc = build_document()
    vocab = {"type": doc["taxonomy"]["type"],
             **{d["field"]: d["values"] for d in doc["taxonomy"]["dimensions"]}}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / JSON_FILE).write_text(dumps(doc), encoding="utf-8")
    (out_dir / JSONL_FILE).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in doc["resources"]),
        encoding="utf-8")
    (out_dir / SCHEMA_FILE).write_text(dumps(build_schema(doc, vocab)), encoding="utf-8")
    sqlite_path = out_dir / SQLITE_FILE
    if sqlite_path.exists():
        sqlite_path.unlink()
    write_sqlite(doc, sqlite_path)
    hashes = {f: sha256(out_dir / f) for f in (JSON_FILE, JSONL_FILE, SQLITE_FILE)}
    (out_dir / CROISSANT_FILE).write_text(dumps(build_croissant(doc, hashes)),
                                          encoding="utf-8")
    return doc


def sqlite_dump(path):
    con = sqlite3.connect(path)
    try:
        return list(con.iterdump())
    finally:
        con.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="rebuild into a temporary directory and compare with release/")
    args = parser.parse_args(argv)
    target = ROOT / OUT_DIR
    try:
        if not args.check:
            doc = render(target)
            print(f"Wrote {OUT_DIR}/ ({len(doc['resources'])} resources):")
            for f in (JSON_FILE, JSONL_FILE, SQLITE_FILE, SCHEMA_FILE, CROISSANT_FILE):
                print(f"  {f}")
            return 0
        with tempfile.TemporaryDirectory() as tmp:
            fresh = Path(tmp)
            render(fresh)
            stale = []
            for f in (JSON_FILE, JSONL_FILE, SCHEMA_FILE, SQLITE_FILE):
                if not (target / f).is_file():
                    stale.append(f"{f} (missing)")
                elif f == SQLITE_FILE:
                    # Compared by content: file bytes depend on the SQLite version.
                    if sqlite_dump(fresh / f) != sqlite_dump(target / f):
                        stale.append(f)
                elif (fresh / f).read_bytes() != (target / f).read_bytes():
                    stale.append(f)
            # croissant.json embeds the SQLite file's checksum, which can differ by
            # SQLite version; compare it with that one checksum masked.
            if not (target / CROISSANT_FILE).is_file():
                stale.append(f"{CROISSANT_FILE} (missing)")
            else:
                def masked(p):
                    c = json.loads(p.read_text(encoding="utf-8"))
                    for d in c["distribution"]:
                        if d["@id"] == SQLITE_FILE:
                            d["sha256"] = None
                    return c
                if masked(fresh / CROISSANT_FILE) != masked(target / CROISSANT_FILE):
                    stale.append(CROISSANT_FILE)
        if stale:
            print("release/ is out of date: " + ", ".join(stale), file=sys.stderr)
            print("run: python3 scripts/build_release.py", file=sys.stderr)
            return 1
        print(f"{OUT_DIR}/ is up to date with the source files.")
        return 0
    except (BuildError, OSError, subprocess.CalledProcessError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
