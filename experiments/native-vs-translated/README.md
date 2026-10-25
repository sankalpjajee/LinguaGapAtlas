# Native vs translated benchmarks: same models, same harness

This experiment runs each model in `models.tsv` on two kinds of benchmark in the same language and compares the scores:

- each of the eight natively written, culturally grounded benchmarks in the LinguaGapAtlas corpus;
- one translated equivalent for each.

Every multiple-choice set uses one prompt format, so a model's two scores differ only in where the questions came from.

## Pairs

| # | Native benchmark (corpus) | Language(s) | Translated equivalent |
| --- | --- | --- | --- |
| 1 | ArabicMMLU | ar | Global-MMLU ar (machine-translated, then post-edited by paid professionals) |
| 2 | CMMLU | zh | Global-MMLU zh (human-translated) |
| 3 | TurkishMMLU | tr | Global-MMLU tr (machine-translated, partly community-edited) |
| 4 | KMMLU | ko | Global-MMLU ko (human-translated) |
| 5 | M3Exam (text questions) | pt, sw | Global-MMLU pt, sw (human-translated) |
| 6 | SEACrowd, via a stand-in: SeaExam's native Indonesian exams | id | Global-MMLU id (human-translated) |
| 7 | FarsTail (NLI) | fa | ParsiNLU entailment: MNLI machine-translated into Persian, about two-thirds corrected by annotators |
| 8 | MultiLoKo | 30 languages | MultiLoKo's own English-sourced questions, human-translated into each language |

Notes:

- **Global-MMLU's human translations** (zh, ko, pt, sw, id) come from OpenAI's MMMLU, which professional translators produced.
- **M3Exam** runs in two of its nine languages. Portuguese and Swahili both have human-translated Global-MMLU sets, and Swahili is low-resource.
- **SEACrowd** is a data hub with no single test set, so it cannot be run as one benchmark. SeaExam's native Indonesian exams stand in for it; the SeaExam authors collected them, and they are not part of the original M3Exam.

**Most informative comparison:** MultiLoKo. It holds the benchmark fixed and changes only whether the questions were locally sourced or translated.

**Main confound in the other pairs:** in every other pair, the native and translated sets differ in content and difficulty. Read their raw gaps together with the model comparison: whether the translated set changes how far apart the models are, or which one is ahead.

## Models

| Model | Notes |
| --- | --- |
| `CohereLabs/aya-expanse-8b` | Instruction-tuned, 23 languages. Gated: accept its licence on the Hub and set `HF_TOKEN`. |
| `marin-community/marin-8b-multilingual-base` | Base model. The Hub page has no model card, so cite Marin's own documentation for its training data. |

Both models are scored the same way:

- **Multiple-choice sets:** 5-shot, with the examples taken from the same benchmark, and no chat template. This is the usual setting for comparing a base model with an instruction-tuned one.
- **MultiLoKo:** its own 5-shot prompt templates.

## Run

```bash
cd experiments/native-vs-translated
python -m venv .venv && source .venv/bin/activate
# install PyTorch for your CUDA version first, then:
pip install -r requirements.txt
export HF_TOKEN=...                      # needed for Aya Expanse

python prepare.py                        # downloads everything, 5-10 minutes
./run.sh models.tsv --limit 5            # smoke test: 5 questions per set
./run.sh models.tsv                      # full run, both models
python analyze.py                        # writes analysis/report.md, pairs.csv, comparisons.csv, multiloko.csv
```

`prepare.py` replaces the sets of any earlier run, so after `git pull` run it again, and delete `results/` if it holds runs of sets that have since changed. For each set, `analyze.py` uses the newest complete run. It uses a `--limit` smoke test only when no complete run exists, and says so.

**On SLURM:** edit the partition and environment lines in `slurm/eval.sbatch`, then:

```bash
mkdir -p logs && sbatch slurm/eval.sbatch    # one array task per model
python analyze.py                            # after both tasks finish
```

**Options:**
- `BACKEND=vllm ./run.sh` is several times faster if vLLM is installed.
- `TASK_GROUPS=lga_mc` or `TASK_GROUPS=lga_multiloko` runs one group only.
- `ONLY=1` runs only the first model.
- `BATCH_SIZE=8` fixes the multiple-choice batch size, which is `auto` by default. `GEN_BATCH_SIZE` sets MultiLoKo's, which is 16 by default. Generation never uses `auto`: lm-eval's probe tries batch 64 at the model's full context, and on a busy GPU it can crash instead of backing off.
- **Free GPU memory:** run each model on a GPU nothing else is using. lm-eval's `auto` batch size takes what is free, so a shared GPU makes the run much slower. Check with `nvidia-smi`.
- `python prepare.py --cap 3000` limits every multiple-choice set to 3,000 test questions, stratified by subject. That cuts the multiple-choice part to about a fifth. The gap intervals of the large pairs widen to roughly ±3 points.
- `python prepare.py --only m3exam,farstail --skip-multiloko` builds only the named native benchmarks, each with its translated pair.

**Scale.** Each model scores 165,535 multiple-choice questions and generates 15,000 MultiLoKo answers. Most of the questions come from two sources: the seven Global-MMLU sets, at 14,042 questions each, and KMMLU, at 35,030. The sample logs that `analyze.py` reads take a few GB per model, because lm-eval stores every prompt. Run time has not been measured on a GPU. Use the `--limit 5` smoke test to estimate it on your hardware.

## Data sources

`prepare.py` needs `git`, `openssl` and `tar` with xz support, and downloads everything itself:

- **Hugging Face Hub:** ArabicMMLU, CMMLU, KMMLU, TurkishMMLU, Global-MMLU, SeaExam and ParsiNLU.
- **GitHub (FarsTail):** read at a pinned commit.
- **MultiLoKo's GitHub repository:** cloned at a pinned commit, and its data archive decrypted with the password its README publishes.
- **Google Drive (M3Exam):** the authors' archive, unzipped with the password their README publishes.

**Do not commit or share** anything in `data/`, `external/` or `results/`; `.gitignore` excludes them:
- MultiLoKo and M3Exam lock their data so that crawlers cannot ingest it, and MultiLoKo asks users not to mirror it;
- KMMLU is CC BY-ND;
- ArabicMMLU and CMMLU are CC BY-NC.

## What the analysis reports

**Summary.** How many gaps are significant and in which direction. How many model comparisons reverse or significantly change between the native and the translated set.

**Multiple-choice pairs.** For each model, accuracy on the native set and on its translated set. Scores are reported raw and chance-corrected, because option counts range from 2 to 5. The gap (translated minus native) comes with a 95% bootstrap interval over questions.

**Model comparison, per pair** (`comparisons.csv`):
- how far ahead one model is on the native set, and on the translated set, each with a 95% interval paired over questions (both models answer the same ones);
- how much that difference changes, with an interval.

A ranking counts as reversed only when the difference is significant on both sets. A near-tie that flips is reported as a reversal "not significantly".

**Global-MMLU.** Accuracy on culturally agnostic vs culturally sensitive questions, for the 2,850 annotated questions per language. Most culturally sensitive questions come from the humanities and social sciences, which models find easier, so this gap mixes culture with subject. Global-MMLU's authors find the same.

**MultiLoKo, per language and model.** Scored with MultiLoKo's own `eval.py`:
- exact match on locally sourced questions;
- exact match on the human translations of English-sourced questions;
- the locality effect (local minus translated), with an interval.

Positive means the translated set underestimates the model; negative means it overestimates it.

**A fix to MultiLoKo's reference answers.** At the pinned commit, 158 of the 250 simplified Mandarin human translations store their accepted answers as one string joined by semicolons (`76；76年；…`), so no answer can match them; 3 other scored questions have the same problem. `analyze.py` scores against a copy in which those strings are split into separate answers, and the report says how many it split.

**MultiLoKo summary, across languages:**
- mean and mean absolute locality effect;
- how many languages go each way;
- the rank correlation of language difficulty between local and translated questions.

## Tested

The pipeline was tested with lm-eval 0.4.13 in three ways:
- lm-eval's dummy model on all 76 tasks;
- a 135M-parameter model on CPU on all 76 tasks, covering both multiple choice and generation;
- planted correct and incorrect answers on every question, whose gaps, ranking reversals and MultiLoKo scores the analysis recovered exactly;
- synthetic models with chosen accuracies, for which the model comparison reported a clear same ranking, a clear reversal and a small-sample near-tie each as expected.

The two 8B models were run on 2× H100. Their multiple-choice runs used an earlier version of the task list, with 29 sets, which contains these 16 sets with identical questions and prompts.
