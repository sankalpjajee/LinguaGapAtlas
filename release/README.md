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
prepared for submission to ACM FAccT 2027). The source files, codebook and
scripts are in the LinguaGapAtlas repository
(<https://github.com/sankalpjajee/LinguaGapAtlas>). Paths such as
`data-dictionary.md` or `rounds-summary.md` below refer to that repository.

## Files

| File | Use it for |
| --- | --- |
| `linguagapatlas.json` | Everything in one self-describing JSON document: metadata, citation, licenses, field definitions, taxonomy and vocabularies, coding design and consensus rules, statistics, caveats, the screening log and all 96 resource records. |
| `linguagapatlas.jsonl` | One resource record per line (the same records as `resources` in the JSON). Loads directly into Hugging Face `datasets`, pandas, DuckDB, Spark, R. |
| `linguagapatlas.sqlite` | Every source cell, the consensus votes, vocabularies, field definitions, caveats and metadata as relational tables, plus ready-made views. It does not hold the statistics or consensus rules; those are in the JSON document. |
| `linguagapatlas.schema.json` | JSON Schema (draft 2020-12) for the document; `#/$defs/resource` validates each JSONL line. Every controlled vocabulary is an enum. |
| `croissant.json` | [Croissant](https://mlcommons.org/croissant/) 1.0 metadata (the MLCommons format used by Hugging Face, Kaggle, OpenML and the NeurIPS Datasets & Benchmarks track), with record sets `resources` (96) and `round_labels` (480), and the caveats as `rai:dataLimitations`. |

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

ints = {"year": "Int64", "language_count": "Int64"}   # nullable integers
df = pd.read_json("linguagapatlas.jsonl", lines=True, dtype=ints)
df.groupby("type")["cultural_specificity"].value_counts(dropna=False)

# one row per resource x round
rounds = pd.json_normalize(df.to_dict("records"), record_path="rounds").astype(ints)
```

Columns that are null in every record (`inference_paradigm`,
`screening_reason`) load with a null or float type in pandas, Arrow and
DuckDB; treat them as text.

**Hugging Face `datasets`** (this file is also a dataset card)

```python
from datasets import load_dataset

ds = load_dataset("json", data_files="linguagapatlas.jsonl", split="train")
# or, from the directory or Hub repo containing this card:
# load_dataset("path/to/release")["corpus"]
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
SELECT * FROM headline_shares;    -- the three headline shares, recomputed from the consensus labels
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

**Validating with the JSON Schema**

```python
import json, jsonschema

schema = json.load(open("linguagapatlas.schema.json", encoding="utf-8"))
jsonschema.validate(json.load(open("linguagapatlas.json", encoding="utf-8")), schema)
line_schema = {"$ref": "#/$defs/resource", "$defs": schema["$defs"]}  # one JSONL line
for line in open("linguagapatlas.jsonl", encoding="utf-8"):
    jsonschema.validate(json.loads(line), line_schema)
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

<!-- BEGIN GENERATED: example-record (scripts/build_release.py) -->
```json
{
  "resource_id": "QG-034",
  "resource_name": "MLQA",
  "canonical_reference": "Lewis, P., et al. (2020). MLQA: Evaluating Cross-lingual Ext ...",
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
  "evidence_note": "Consensus of 5 independent coding rounds (majority >=3). Spl ...",
  "coder": "consensus-5rounds",
  "confidence": "medium",
  "screening_status": "included",
  "screening_reason": null,
  "consensus": {
    "data_origin": {
      "value": null,
      "status": "no_majority",
      "support": 2,
      "rounds_coded": 5,
      "split": true,
      "votes": [
        {"value": "Native", "count": 2, "rounds": ["C", "E"]},
        {"value": "Translated", "count": 2, "rounds": ["A", "B"]},
        {"value": "Mixed", "count": 1, "rounds": ["D"]}
      ]
    },
    "...": "one entry per label field: type, data_origin, resource_tier, task_class, cultural_specificity, inference_paradigm"
  },
  "rounds": [
    {
      "round": "A",
      "resource_id": "QG-034",
      "type": "Benchmark",
      "data_origin": "Translated",
      "evidence_note": "Paper: questions crowdsourced on English Wikipedia paragraph ...",
      "confidence": "high",
      "labels_as_written": {"data_origin": "Translated", "...": "..."},
      "...": "..."
    },
    "... rounds B-E ..."
  ]
}
```
<!-- END GENERATED: example-record -->

- **Top-level fields** are the canonical corpus (`data/corpus.csv`), with the
  same names and meanings as in `data-dictionary.md`. Integers are typed and
  empty cells are `null`. Two exceptions: `coder` is always
  `consensus-5rounds`, and `confidence` is derived from the vote (see the
  caveats).
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
  cast it. Count contested cells with `split`, not `status`: a `no_majority`
  cell can be one where only one or two rounds coded it and they agreed.
- **`rounds`** holds the five coding rounds' records, A–E. Each has labels in
  canonical form (normalized exactly as in `scripts/build_consensus.py`) and,
  in `labels_as_written`, verbatim from the round's file.

**Recovering the source CSVs.** Every cell of every source CSV can be
recovered from the release. Each file has the columns listed in
`coding.source_columns`, in that order:
- `corpus.csv` is the resource-level fields.
- A round file is that round's records, with:
  - labels from `labels_as_written`;
  - `resource_id`, `resource_name` and `screening_status` from the resource;
  - `coder` equal to the round letter.
- Write `null` as an empty cell.

## The JSON document

`linguagapatlas.json` has these top-level keys, in order:

| Key | Contents |
| --- | --- |
| `$schema` | Relative link to `linguagapatlas.schema.json`. |
| `name`, `title`, `version`, `date_released`, `doi`, `description`, `keywords`, `authors`, `citation`, `paper`, `repository`, `license` | Artifact metadata, from `CITATION.cff` and `README.md`. `date_released` and `doi` stay `null` until they are set in `CITATION.cff` at release time. `description` is computed from the records. |
| `null_rule` | The rule above. |
| `sources`, `generator` | SHA-256 of every file the release was built from (the corpus, the round files, the screening log, the data dictionary, the citation file, `rounds-summary.md`, `README.md`, and the three package scripts whose code it reuses), and the script that built it. |
| `fields` | Definition of every resource field (from `data-dictionary.md`, plus the derived fields). |
| `taxonomy` | The `type` vocabulary; the five dimensions (D1–D5) with definitions and controlled vocabularies, each value marked `listed_in_data_dictionary` and, for task classes, `applies_to` (Benchmark / Dataset); and the `confidence` levels. |
| `coding` | Round design and source files, the label normalization (case, synonyms, null tokens), the consensus rules for labels, year, language count, languages and confidence, and the source column order with the reconstruction recipe. |
| `statistics` | Computed from the released labels: composition, the headline shares (by consensus type, and per round by each round's own type), value distributions by type, consensus coverage, and inter-round agreement (Cohen's kappa and observed agreement for all 10 round pairs, plus the mean). |
| `caveats` | Known limitations, each with an `id`. Read these before using the data. |
| `screening_log` | The screening record (`data/screening-log.csv`). |
| `resources` | The 96 resource records. |

## SQLite tables and views

<!-- BEGIN GENERATED: sqlite-tables (scripts/build_release.py) -->
| Table / view | Rows | Contents |
| --- | --- | --- |
| `resources` | 96 | The consensus corpus, one row per resource. |
| `resource_languages` | 817 | One row per named language: resource_id, position, language. |
| `round_labels` | 480 | Each round's record per resource (canonical labels). |
| `round_labels_as_written` | 1906 | Verbatim label strings from the round files (non-empty cells). |
| `consensus` | 576 | The decision per resource and label field. |
| `consensus_votes` | 463 | The votes behind each decision, with the rounds that cast them. |
| `rounds` | 5 | Round letter, source file and checksum. |
| `vocabulary` | 32 | The controlled vocabularies. |
| `fields` | 22 | Field definitions. |
| `caveats` | 10 | Known limitations (read before use). |
| `metadata` | 27 | Artifact metadata and source checksums. |
| `screening_candidates` | 0 | The screening log. |
| `headline_shares` | 6 | View: translated, Universal and Low-tier shares by type, with hits, coded, undetermined and percent. |
| `split_cells` | 61 | View: every cell on which the rounds disagreed, with the vote split. |
<!-- END GENERATED: sqlite-tables -->

## Caveats

Read these before using the data. They are also under `caveats` in the JSON,
in the `caveats` table and in Croissant's `rai:dataLimitations`.

<!-- BEGIN GENERATED: caveats (scripts/build_release.py) -->
- **`ai-coders`**: rounds-summary.md states: "The second coders are AI agents; the paper must disclose this as AI second-coding / human–AI agreement, not two-human IAA." This package does not record which rounds were coded by whom.
- **`differs-from-paper`**: The consensus corpus has 58 benchmarks and 38 datasets; the paper describes 65 and 31. Shares computed from this release for benchmarks (translated 52.0%, Universal 79.6%, Low tier 100.0% of tier-coded rows) are not the published 26.2%, 55.4% and 6.2% (see rounds-summary.md), and first-release years run 2005-2025, not 2018-2025. Figures in this release are computed from the released labels.
- **`tier-sparse`**: resource_tier has a consensus value for 12 of 96 resources (12 Low; no other tier); 4 more were coded by only 1-2 rounds (no majority) and 80 were left undetermined by every round, because sources rarely state a tier. The low-resource share is not estimable from this release, and tier agreement (kappa 1.0 on a handful of jointly coded pairs) is not a reliability estimate.
- **`inference-paradigm-uncoded`**: inference_paradigm (D5) is null for every resource in the consensus and in every round: the codebook does not enumerate its values (methods/codebook-draft.md, D5), and the vocabulary listed in data-dictionary.md is provisional.
- **`screening-log-empty`**: data/screening-log.csv contains 0 candidate rows: the 142-candidate screening record (46 exclusions) that data-dictionary.md describes is not in this release. An empty candidate list is not evidence that no candidates were excluded.
- **`task-class-vocabulary`**: task_class values Generation, Long Context, Embeddings occur in the data but are not in data-dictionary.md's listed vocabulary (they belong to the 13-class set in methods/codebook-draft.md, ambiguity A6). data-dictionary.md lists task classes separately for benchmarks and datasets (applies_to), yet 6 resources carry a class listed only for the other type (QG-038, QG-039, QG-040, QG-048, QG-053, QG-062; codebook ambiguity A9).
- **`round-a-scaffold`**: At resource level, canonical_reference, provenance_url, screening_reason and the evidence quoted in evidence_note are round A's (the consensus build's base round). In 16 resources round A coded a different value from the consensus for at least one voted label, so the quoted evidence can support a label other than the consensus one. screening_reason is null at resource level because round A left it empty; other rounds record one for 64 resources. Every round's own reference, URL, evidence note, confidence and screening reason is under rounds.
- **`consensus-confidence`**: Resource-level confidence is derived from the vote (high: no split or no-majority voted cell; medium: 1-2; low: 3 or more), not recorded by a coder. Each round's own confidence is under rounds.
- **`languages-count-only`**: 21 resources document their languages only partly or only as a count (e.g. '75 languages'); for these, language_names holds just the languages actually named and languages_fully_named is false. A further 6 have no documented language list (languages and language_names are null). language_count is as documented, so a value such as 250 for '250+ languages' is a lower bound.
- **`codebook-draft`**: The codebook (methods/codebook-draft.md) is a draft. Adjudication of the split cells and the decision rules it is meant to produce are pending (methods/iaa-protocol.md).
<!-- END GENERATED: caveats -->

## Rebuilding and verifying

The release is generated from the package's source files by
`scripts/build_release.py`. This card is the one hand-written file. Its marked
sections (the example record, the table counts and the caveats) are generated
from the data and checked by `--check`.

```sh
python3 scripts/build_release.py          # rebuild release/ from data/ and coding/
python3 scripts/build_release.py --check  # confirm release/ matches the sources
sha256sum -c MANIFEST.sha256              # verify every file byte for byte
```

The build is standard-library Python, and it is deterministic. It refuses to
write anything in any of these cases:
- `data/corpus.csv` is not exactly what `scripts/build_consensus.py` produces
  from the five rounds;
- a consensus cell disagrees with the recomputed vote;
- a label falls outside the data dictionary's vocabulary.

The `sources` key records the checksum of every input.

## License and citation

Data: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Code
(`scripts/`): MIT. The classified benchmarks and datasets remain under their
creators' own licenses. This release contains only classification metadata
about them, with links to their sources.

> Jajee, S. K., Shaw, T., & Soni, V. *LinguaGapAtlas: A Corpus and Taxonomy of
> 96 Multilingual AI Benchmarks and Datasets* (corpus and reproducibility
> package, v1.0.0). https://github.com/sankalpjajee/LinguaGapAtlas

Machine-readable citation metadata: `CITATION.cff` in the repository, and
`citation` in `linguagapatlas.json`.
