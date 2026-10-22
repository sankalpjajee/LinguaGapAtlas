# LinguaGapAtlas — five-round coding results (2026-10-22)

Five independent blind rounds (A–E), each coding all 96 resources from primary sources under the same codebook. Labels normalized before comparison (case unified; "Translation/alignment"→Translation, "Instruction fine-tuning"→Instruction, "long-context"→Long Context — same labels, different string forms across rounds; the canonical package will enforce one casing).

## Headline statistics per round (recompute.py; undetermined excluded per dimension)

Benchmarks — translated-data share: A 47.4% · B 52.1% · C 46.4% · D 54.5% · E 52.9% → **average 50.7%**, range 46.4–54.5. Published: 26.2%.
Benchmarks — culturally non-specific (Universal): A 59.0% · B 72.0% · C 75.0% · D 78.7% · E 72.7% → **average 71.5%**, range 59.0–78.7 (A depressed by 25 undetermined culture cells). Published: 55.4%.
Datasets — translated: A 29.0% · B 31.7% · C 28.9% · D 31.1% · E 17.6% → **average 27.7%**, range 17.6–31.7.
Datasets — Universal: A 65.6% · B 65.1% · C 64.1% · D 73.9% · E 62.9% → **average 66.3%**, range 62.9–73.9.
Composition per round (Benchmark/Dataset): A 64/32 · B 53/43 · C 57/39 · D 50/46 · E 60/36 → average ≈ 57/39, vs the paper's claimed 65/31.
Low-resource share: **not estimable** — every round codes only 4–9 benchmark rows on tier (all Low), the rest undetermined because sources almost never state a tier. Published: 6.2%.

## Agreement across rounds (mean pairwise, normalized labels)

- type: Cohen's κ 0.82, raw agreement 91.2%
- task_class: κ 0.93, raw 93.3%
- data_origin: κ 0.89, raw 94.4%
- cultural_specificity: κ 0.80, raw 90.7%
- resource_tier: κ 1.00 on ~9 jointly coded pairs — not a reliability estimate; tier's problem is coverage, not agreement.

## Consensus and disagreements

Majority (≥3 of 5 rounds) exists for: type 96/96, task_class 96/96, data_origin 87/96, cultural_specificity 87/96, resource_tier 12/96.
Full disagreement list with vote splits: `coding/disagreements.md` (type 18 cells, task_class 14, data_origin 12, cultural_specificity 17, tier 0).

## Caveats carried forward

- The second coders are AI agents; the paper must disclose this as AI second-coding / human–AI agreement, not two-human IAA.
- Value-string formats varied across rounds (case and two synonym forms); normalization was mechanical only. The shipped package enforces canonical values (checker tool, approved item 3).
- Tier stays sparse unless Sankalp adopts one uniform language-to-tier mapping rule (his decision, still open).
- D5 inference_paradigm: uncodable as published; still open.
