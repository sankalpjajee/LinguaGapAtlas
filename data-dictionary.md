# Data Dictionary — `data/corpus.csv`

The canonical corpus file. **One row per resource**: 96 rows (65 benchmarks +
31 datasets). Field order in the CSV follows the order below.

## The null rule (applies to every field)

A value that could not be determined from the resource's public documentation
is stored as an **empty field**, which every consumer of this file must read as
`null` (undetermined). Undetermined values are **never imputed**: no label is
ever inferred from a related field, from a resource's reputation, or from the
corpus-wide distribution. Scripts report undetermined counts explicitly and
exclude them from that statistic's denominator, saying so.

## Fields

### Identifiers

| Field | Type | Description |
| --- | --- | --- |
| `resource_id` | string, required, unique | Stable ID assigned by this project, form `QG-001`…`QG-096`. Never reused, never renumbered between versions. |
| `resource_name` | string, required | The resource's canonical name as given by its creators (e.g. the benchmark/dataset name used in its paper). |
| `canonical_reference` | string | Full citation (authors, title, venue, year) of the paper or release that introduces the resource. `null` if the resource has no citable paper. |

### Resource description

| Field | Type | Description |
| --- | --- | --- |
| `type` | enum, required | `Benchmark` or `Dataset`. Benchmarks are resources designed for evaluation; datasets are resources designed primarily for training. A resource released as both is coded by its primary stated purpose and the dual nature is noted in `evidence_note`. |
| `year` | integer | Year of first public release (paper or public artifact, whichever came first). `null` if not determinable. |
| `languages` | string | Semicolon-separated list of language names (or ISO 639 codes) the resource covers, as documented by its creators. `null` if the documentation does not enumerate them. |
| `language_count` | integer | Number of languages covered, counted from `languages` where enumerated, otherwise the count stated by the creators. `null` if neither is available. Never estimated. |

### The five taxonomy dimensions

| Field | Controlled vocabulary | Definition |
| --- | --- | --- |
| `data_origin` | `Native` \| `Translated` \| `Mixed` | How the resource's content was produced. `Native`: authored originally in the target language(s). `Translated`: produced by translating content from another language (human or machine). `Mixed`: both routes used for substantial portions. `null` if the production route is not documented. |
| `resource_tier` | `High` \| `Medium` \| `Low` | Linguistic resource tier of the language(s) the resource targets, per the paper's tiering of language resourcedness. For multilingual resources, the tier of the lowest-resourced language substantially covered. `null` if the covered languages cannot be established. |
| `task_class` | see below | The task family the resource evaluates or supports. Benchmarks: `QA`, `Syntax`, `Reasoning`, `Multimodal`, `NLU`, `Summarization`, `Math`, `RAG`, `Code`, `Other`. Datasets: `Pre-training`, `Translation`, `Instruction`. A resource spanning several classes is coded by its primary class; secondary classes go in `evidence_note`. |
| `cultural_specificity` | `Universal` \| `Culturally Anchored` \| `Region-Specific` | `Universal`: designed to be culturally non-specific. `Culturally Anchored`: content is grounded in the cultural context of its language community without being tied to one sub-region. `Region-Specific`: content is grounded in a particular country/region's context. `null` if the documentation does not support a judgement. |
| `inference_paradigm` | `Discriminative` \| `Generative` \| `Retrieval-Augmented` \| `Mixed` \| `Other` | The inference/evaluation paradigm the resource is built around. Vocabulary to be confirmed against the final codebook before v1.0.0; rows coded before confirmation carry the coder's best judgement and are flagged in `confidence`. `null` if not determinable. |

### Provenance and coding record

| Field | Type | Description |
| --- | --- | --- |
| `provenance_url` | string (URL) | The evidence source the row's labels were taken from — the resource's paper, repository, or documentation page actually consulted. Every coded dimension in the row must be supported by this source or by `evidence_note`. `null` only for rows with no coded dimensions. |
| `evidence_note` | string | Free-text note recording what the source states that supports the labels (quoted or paraphrased), plus any dual-nature or secondary-class remarks. This is the audit trail for the row. |
| `coder` | string | Identifier of the coder who produced this row's labels (initials or coder ID used in the reliability study). |
| `confidence` | enum | Coder's confidence in the row's labels: `high`, `medium`, or `low`. `low` rows are candidates for adjudication and are reported as such. `null` if not recorded. |

### Screening record

| Field | Type | Description |
| --- | --- | --- |
| `screening_status` | enum, required | Disposition of this resource in the screening pipeline (see `data/screening-log.csv`): for rows in `corpus.csv` this is always `included`; the full candidate set, including exclusions, lives in the screening log. |
| `screening_reason` | string | For included rows, the eligibility basis (e.g. which inclusion criterion the resource met). Exclusion reasons are recorded in the screening log, not here. |

## Companion file: `data/screening-log.csv`

One row per **candidate** resource identified by the search (142 rows).
Columns: `candidate_id`, `resource_name`, `identified_via` (database / venue /
citation trail / other), `year`, `type` (as identified), `screening_stage`
(`identification` \| `eligibility` \| `included`), `decision` (`include` \|
`exclude`), `exclusion_reason` (controlled phrases from the codebook, e.g.
`duplicate`, `not-multilingual`, `no-public-artifact`, `out-of-date-range`,
`not-a-benchmark-or-dataset`; empty for included rows), `screener`, `notes`.
The 96 `include` rows correspond 1-to-1 to `corpus.csv` rows via
`resource_name`; the remaining 46 rows are the recorded exclusions.
