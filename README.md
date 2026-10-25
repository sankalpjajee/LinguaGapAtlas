# LinguaGapAtlas — 96-Resource Corpus and Taxonomy (Reproducibility Package)

Companion artifact for the paper of the same name:

> **LinguaGapAtlas: A Systematic Taxonomy of Bias and Imbalance in 96 Multilingual AI Benchmarks & Datasets**
> Sankalp K. Jajee, Tanisha Shaw, and Vatsalya Soni. Prepared for submission to ACM FAccT 2027.
> (Earlier versions of the paper circulated under the title "Quantifying the Gaps.")

## Purpose

This package is the single canonical, citable release of the paper's corpus:
one structured row for each of the 96 multilingual AI resources (65 benchmarks
and 31 datasets, 2018–2025) classified in the paper, together with the
screening record that produced the corpus, the codebook that defines the
classification, and scripts that recompute every headline number in the paper
from the released rows. It exists so that:

- every percentage reported in the paper can be recomputed from this artifact
  (`scripts/recompute.py`);
- the classification can be audited — each row carries the evidence source the
  label was taken from;
- the annotation reliability reported in the paper can be recomputed, pair by
  pair, from the coding rounds' label files (`scripts/agreement.py`);
- other researchers can build on the corpus under clear licenses.

## Using the corpus

To use the corpus in your own research, start with **`release/`**. It merges
the whole package into one release: the consensus labels, every vote behind
them, all five coding rounds' records and evidence, the taxonomy, statistics
and caveats. It ships as a single JSON document, JSON Lines (one resource per
line), an SQLite database, a JSON Schema and Croissant metadata. Each format
loads directly in Python, pandas, Hugging Face `datasets`, DuckDB, R or any
SQLite client. See [`release/README.md`](release/README.md) for loading
examples and the record structure. The release is generated from the files
below by `scripts/build_release.py` and is never edited by hand.

## The undetermined rule (read this first)

**A value that could not be determined from a resource's public documentation
is recorded as an explicit `null` — never imputed, never guessed, and never
silently dropped.**

Missing values are first-class data in this package: every statistic script
reports, for each dimension, how many rows were coded, how many were
undetermined, and which denominator a percentage uses. A blank in a source
document is *not* evidence for any label.

## Contents

| Path | What it is |
| --- | --- |
| `data/corpus.csv` | The canonical corpus: one row per included resource (96 rows), schema in `data-dictionary.md`. |
| `data/screening-log.csv` | The screening record for all 142 candidate resources identified, with the include/exclude decision and reason for each (96 included + 46 excluded). |
| `data-dictionary.md` | Field-by-field definition of `corpus.csv`: types, controlled vocabularies, and the null rule per field. |
| `scripts/recompute.py` | Recomputes the paper's headline shares (translated-data share, culturally non-specific share, low-resource share) from `data/corpus.csv`, separately for benchmarks and datasets, reporting counts and denominators. |
| `scripts/agreement.py` | Computes observed agreement and Cohen's kappa between two coder CSVs, per taxonomy dimension and overall. |
| `coding/` | The five independent coding rounds (`coderA-full.csv` … `coderE-full.csv`, same schema as `corpus.csv`) and the cross-round disagreement list. `data/corpus.csv` is their majority-vote consensus. |
| `scripts/build_consensus.py` | Builds `data/corpus.csv` from the five coding rounds by majority vote (≥3 of 5). |
| `release/` | The amalgamated research release, generated from the files above: `linguagapatlas.json`, `.jsonl`, `.sqlite`, `.schema.json`, `croissant.json`, and a dataset card (`release/README.md`). |
| `scripts/build_release.py` | Builds `release/` from the source files; `--check` verifies that `release/` is current. |
| `experiments/native-vs-translated/` | Runs language models on the corpus's natively written, culturally grounded benchmarks and on translated equivalents in the same languages, and compares the scores (see its `README.md`). It downloads the benchmark data at run time; none is stored here. |
| `rounds-summary.md` | Results of the five coding rounds: per-round headline statistics, agreement across rounds, consensus coverage and caveats. |
| `methods/` | The codebook draft (`codebook-draft.md`) and the inter-annotator agreement protocol (`iaa-protocol.md`). |
| `CITATION.cff` | Machine-readable citation metadata for this artifact. |
| `LICENSES.md` | The data and code licenses. |
| `MANIFEST.sha256` | SHA-256 checksums for every file in the release (see *File integrity* below). |
| `tests/` | Tiny, clearly-marked **synthetic** CSVs used only to smoke-test the scripts. They are not corpus data and must never be merged into `data/`. |

## Headline numbers this package must reproduce

From the paper (benchmarks, n = 65; datasets, n = 31):

- Low-resource languages appear in **6.2%** of benchmarks.
- **55.4%** of benchmarks are culturally non-specific (coded `Universal`).
- **26.2%** of benchmarks rely on translated data.

Run `python3 scripts/recompute.py` to verify these against `data/corpus.csv`.
Percentages are computed over rows where the relevant dimension is coded;
undetermined rows are reported as counts, never folded into a denominator
silently.

## Versioning

- This artifact is versioned independently of the paper using semantic
  versioning (`MAJOR.MINOR.PATCH`).
- **v1.0.0** is the release pinned to the FAccT 2027 submission: the exact
  corpus, codebook, and scripts the paper's numbers were computed from. Any
  post-submission correction increments the version and is described in a
  changelog entry; v1.0.0 itself is never silently edited, so the paper
  remains reproducible against it.
- Each release is tagged in the repository (`v1.0.0`, …) and the tag is what
  should be cited for reproducibility claims.

## Citation

Cite the paper and, where the corpus itself is used, this artifact. Machine-
readable metadata is in `CITATION.cff`. Plain form:

> Jajee, S. K., Shaw, T., & Soni, V. *LinguaGapAtlas: A Systematic
> Taxonomy of Bias and Imbalance in 96 Multilingual AI Benchmarks & Datasets*
> (corpus and reproducibility package, v1.0.0).

## Licenses

- **Data** (`data/`, `data-dictionary.md`): Creative Commons Attribution 4.0
  International (**CC BY 4.0**). You may share and adapt the corpus with
  attribution.
- **Code** (`scripts/`): **MIT License**. You may use, copy, modify, and
  redistribute the scripts with the license notice retained.

The underlying benchmarks and datasets classified here remain the property of
their respective creators under their own licenses; this package distributes
only our classification metadata about them, plus links to their sources.

## File integrity (MANIFEST.sha256)

Each release ships a `MANIFEST.sha256` listing the SHA-256 checksum of every
file in the package, so any copy can be verified byte-for-byte against the
release.

To generate the manifest for a release (from the package root):

```sh
find . -type f ! -name 'MANIFEST.sha256' -not -path './.git/*' \
  -not -path '*/__pycache__/*' -print0 \
  | sort -z | xargs -0 sha256sum > MANIFEST.sha256
```

To verify a downloaded copy against the manifest:

```sh
sha256sum -c MANIFEST.sha256
```

Regenerate the manifest whenever any file in the package changes, and commit
it together with that change. Checksums are always generated from the final
files, never written by hand.
