---
license: cc-by-4.0
pretty_name: LinguaGapAtlas
tags:
- multilingual
- benchmarks
- datasets
- bias
- evaluation
- taxonomy
- fairness
- meta-analysis
size_categories:
- n<1K
configs:
- config_name: default
  data_files:
  - split: corpus
    path: linguagapatlas.jsonl
---

# LinguaGapAtlas — research release

The whole LinguaGapAtlas package, merged into one machine-readable release.
It covers 96 multilingual AI resources (benchmarks and datasets) classified on
five taxonomy dimensions. For each resource it holds:

- the **consensus labels**;
- **every vote** behind each consensus cell;
- the **five independent coding rounds' own records**, including each round's
  evidence note, source URL, reference and confidence;
- the **taxonomy and controlled vocabularies**;
- **statistics** computed from the labels;
- the package's **caveats**.

All of this ships in formats you can load directly; no CSV parsing needed.

Companion to the paper *LinguaGapAtlas: A Systematic Taxonomy of Bias and
Imbalance in 96 Multilingual AI Benchmarks & Datasets* (Jajee, Shaw & Soni;
prepared for submission to ACM FAccT 2027).

## Files

| File | Use it for |
| --- | --- |
| `linguagapatlas.json` | Everything in one self-describing JSON document: metadata, citation, licenses, field definitions, taxonomy and vocabularies, coding design and consensus rules, statistics, caveats, the screening log and all 96 resource records. |
| `linguagapatlas.jsonl` | One resource record per line (the same records as `resources` in the JSON). Loads directly into Hugging Face `datasets`, pandas, DuckDB, Spark, R. |
| `linguagapatlas.sqlite` | The same content as relational tables plus ready-made views, for SQL, R (`RSQLite`) or any SQLite client. |
| `linguagapatlas.schema.json` | JSON Schema (draft 2020-12) for the document; `#/$defs/resource` validates each JSONL line. Every controlled vocabulary is an enum. |
| `croissant.json` | [Croissant](https://mlcommons.org/croissant/) 1.0 metadata (the MLCommons format used by Hugging Face, Kaggle, OpenML and the NeurIPS Datasets & Benchmarks track), with record sets `resources` (96) and `round_labels` (480). |

## Quick start

**Python (standard library only)**

```python
import json

atlas = json.load(open("linguagapatlas.json", encoding="utf-8"))
benchmarks = [r for r in atlas["resources"] if r["type"] == "Benchmark"]

coded = [r for r in benchmarks if r["data_origin"] is not None]   # null = undetermined
translated = sum(r["data_origin"] == "Translated" for r in coded)
print(f"translated: {translated}/{len(coded)} coded benchmarks "
      f"({len(benchmarks) - len(coded)} undetermined)")
```

**pandas**

```python
import pandas as pd

df = pd.read_json("linguagapatlas.jsonl", lines=True,
                  dtype={"year": "Int64", "language_count": "Int64"})
df.groupby("type")["cultural_specificity"].value_counts(dropna=False)

# one row per resource x round
rounds = pd.json_normalize(df.to_dict("records"), record_path="rounds")
```

**Hugging Face `datasets`** (this file is also a dataset card)

```python
from datasets import load_dataset

ds = load_dataset("json", data_files="linguagapatlas.jsonl", split="train")
# or, from a directory or Hub repo containing this README: load_dataset(path)["corpus"]
```

**DuckDB**

```sql
SELECT task_class, count(*) AS n
FROM read_json_auto('linguagapatlas.jsonl')
WHERE type = 'Benchmark'
GROUP BY task_class ORDER BY n DESC;
```

**SQLite** (`sqlite3 linguagapatlas.sqlite`)

```sql
SELECT * FROM headline_shares;                 -- the paper's three shares, null rule applied
SELECT * FROM split_cells WHERE field = 'cultural_specificity';
SELECT resource_id, round, data_origin, evidence_note
FROM round_labels WHERE resource_id = 'QG-034';
```

**R**

```r
library(jsonlite)
atlas <- fromJSON("linguagapatlas.json")
table(atlas$resources$type, atlas$resources$data_origin, useNA = "ifany")

library(DBI)
con <- dbConnect(RSQLite::SQLite(), "linguagapatlas.sqlite")
dbGetQuery(con, "SELECT * FROM headline_shares")
```

**Croissant**

```python
import mlcroissant as mlc

ds = mlc.Dataset(jsonld="croissant.json")
for record in ds.records(record_set="resources"):
    ...
```

## The null rule

`null` means **undetermined**. Either the value could not be established from
the resource's public documentation, or (for a consensus label) the five rounds
did not reach a majority. Undetermined values are never imputed. When you
compute a share, exclude nulls from the denominator and report how many there
were, as `scripts/recompute.py` and the `headline_shares` view do. A null is not
evidence for any label.

## A resource record

Each line of the JSONL file (and each element of `resources` in the JSON) is
one resource. Abridged:

```json
{
  "resource_id": "QG-034",
  "resource_name": "MLQA",
  "canonical_reference": "Lewis, P., et al. (2020). MLQA: ...",
  "type": "Benchmark",
  "year": 2019,
  "languages": "English; Arabic; German; Spanish; Hindi; Vietnamese; Chinese (Simplified)",
  "language_names": ["English", "Arabic", "German", "Spanish", "Hindi", "Vietnamese", "Chinese (Simplified)"],
  "languages_fully_named": true,
  "language_count": 7,
  "data_origin": null,
  "resource_tier": null,
  "task_class": "QA",
  "cultural_specificity": "Universal",
  "inference_paradigm": null,
  "provenance_url": "https://arxiv.org/abs/1910.07475",
  "evidence_note": "Consensus of 5 independent coding rounds (majority >=3). ...",
  "coder": "consensus-5rounds",
  "confidence": "medium",
  "screening_status": "included",
  "screening_reason": null,
  "consensus": {
    "data_origin": {
      "value": null, "status": "no_majority", "support": 2, "rounds_coded": 5, "split": true,
      "votes": [{"value": "Native", "count": 2, "rounds": ["C", "E"]},
                {"value": "Translated", "count": 2, "rounds": ["A", "B"]},
                {"value": "Mixed", "count": 1, "rounds": ["D"]}]
    },
    "...": "one entry per label field: type, data_origin, resource_tier, task_class, cultural_specificity, inference_paradigm"
  },
  "rounds": [
    {"round": "A", "resource_id": "QG-034", "type": "Benchmark", "data_origin": "Translated",
     "evidence_note": "Paper: questions crowdsourced on English Wikipedia ...",
     "confidence": "medium", "labels_as_written": {"data_origin": "Translated", "...": "..."}, "...": "..."},
    "... rounds B-E ..."
  ]
}
```

- **Top-level fields** are the canonical corpus (`data/corpus.csv`), with the
  same names and meanings as in `data-dictionary.md`. Integers are typed and
  empty cells are `null`.
- **`language_names` / `languages_fully_named`** are derived from `languages`.
  Some sources state only a count (e.g. `"75 languages"`). For those,
  `language_names` holds only the languages actually named, and
  `languages_fully_named` is `false`.
- **`consensus`** gives, for each label field, the value (or `null`) and its
  `status`:
  - `consensus` means at least 3 of 5 rounds agreed;
  - `no_majority` means some rounds coded the cell but no value reached 3;
  - `not_coded` means every round left the cell undetermined.

  It also gives `support` (votes for the top value), `rounds_coded`, `split`
  (whether the rounds that coded the cell disagreed, i.e. whether the cell
  appears in `coding/disagreements.md`) and every vote with the rounds that
  cast it.
- **`rounds`** holds the five coding rounds' records, A–E. Each has labels in
  canonical form (normalized exactly as in `scripts/build_consensus.py`) and,
  in `labels_as_written`, verbatim from the round's file. Every cell of every
  source CSV can be recovered from the release.

## The JSON document

`linguagapatlas.json` has these top-level keys, in order:

| Key | Contents |
| --- | --- |
| `name`, `title`, `version`, `description`, `keywords`, `authors`, `citation`, `paper`, `repository`, `license` | Artifact metadata (from `CITATION.cff` and `README.md`). |
| `null_rule` | The rule above. |
| `sources`, `generator` | SHA-256 of every source file the release was built from, and the script that built it. |
| `fields` | Definition of every resource field (from `data-dictionary.md`, plus the derived fields). |
| `taxonomy` | The five dimensions (D1–D5): definitions and controlled vocabularies, with `listed_in_data_dictionary` and, for task classes, `applies_to` (Benchmark / Dataset). |
| `coding` | Round design and source files, the label normalization (case, synonyms, null tokens) and the consensus rules for labels, year, language count, languages and confidence. |
| `statistics` | Computed from the released labels: composition, the headline shares (overall and per round), value distributions by type, consensus coverage, and inter-round agreement (Cohen's kappa and observed agreement for all 10 round pairs, plus the mean). |
| `caveats` | Known limitations, each with an `id`. Read these before using the data. |
| `screening_log` | The screening record (`data/screening-log.csv`). |
| `resources` | The 96 resource records. |

## SQLite tables and views

| Table / view | Rows | Contents |
| --- | --- | --- |
| `resources` | 96 | The consensus corpus (one row per resource). |
| `resource_languages` | one per named language | `resource_id`, `position`, `language`. |
| `round_labels` | 480 | Each round's record per resource (canonical labels). |
| `round_labels_as_written` | one per non-empty label cell | Verbatim label strings from the round files. |
| `consensus` / `consensus_votes` | 576 / one per distinct vote | Decision per resource × label field, and the votes behind it. |
| `rounds` | 5 | Round letter, source file and checksum. |
| `vocabulary`, `fields`, `caveats`, `metadata` | | The taxonomy vocabularies, field definitions, caveats and artifact metadata. |
| `screening_candidates` | | The screening log. |
| view `headline_shares` | | Translated, Universal and Low-tier shares by type, with hits, coded, undetermined and percent. |
| view `split_cells` | | Every cell on which the rounds disagreed, with the vote split. |

## Caveats

Each caveat is spelled out under `caveats` in the JSON (and in the `caveats`
table). In brief:

- **The figures differ from the published paper.** The consensus corpus has
  58 benchmarks and 38 datasets, not 65 and 31, and its shares differ from
  the published 26.2% / 55.4% / 6.2% (see `rounds-summary.md`). The
  `statistics` in this release are computed from the released labels.
- **The coding rounds include AI coders.** Per `rounds-summary.md`, agreement
  across rounds is human–AI agreement, not two-human inter-annotator
  agreement.
- **`resource_tier` is sparse:** it is coded for 12 of 96 resources, all
  `Low`, so the low-resource share is not estimable. **`inference_paradigm`
  is uncoded** for every resource.
- **The screening log has no candidate rows**, so the 142-candidate
  screening record is not part of this release.
- **Three task classes are not in `data-dictionary.md`:** `Generation`,
  `Long Context` and `Embeddings` occur in the data but are missing from its
  listed vocabulary.
- **Resource-level references, URLs and evidence come from round A.**
  Resource-level confidence is derived from the vote. Each round's own values
  are under `rounds`.

## Rebuilding and verifying

The release is generated from the package's source files and must not be
edited by hand:

```sh
python3 scripts/build_release.py          # rebuild release/ from data/ and coding/
python3 scripts/build_release.py --check  # confirm release/ matches the sources
sha256sum -c MANIFEST.sha256              # verify every file byte for byte
```

The build is standard-library Python, and it is deterministic. It refuses to
write anything if `data/corpus.csv` is not exactly what
`scripts/build_consensus.py` produces from the five rounds, or if any
consensus cell disagrees with the recomputed vote. The `sources` key records
the checksum of every input.

## License and citation

Data: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Code
(`scripts/`): MIT. The classified benchmarks and datasets remain under their
creators' own licenses. This release contains only classification metadata
about them, with links to their sources.

> Jajee, S. K., Shaw, T., & Soni, V. *LinguaGapAtlas: A Corpus and Taxonomy of
> 96 Multilingual AI Benchmarks and Datasets* (corpus and reproducibility
> package, v1.0.0).

Machine-readable citation metadata: `CITATION.cff`, and `citation` in
`linguagapatlas.json`.
