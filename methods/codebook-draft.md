# Codebook — LinguaGapAtlas taxonomy (DRAFT, source-extracted)

**Status:** Draft assembled 2026-10-22 strictly from what the two versions of the paper
state. Nothing here is invented. Where the paper does not specify something a coder
needs, the gap is marked **[GAP]** — those gaps are exactly what the decision rules
and the IAA pass must resolve.

**Sources**
- [41p] Posted 41-page version, ResearchGate publication 399995100 (J. ACM format).
- [AAAI] AAAI-27 version (Overleaf "Quantifying_AAAI_27"), CameraReady2027 text.
- The AAAI version carries the more precise value sets and the only two stated
  decision rules; the 41-page version carries the fuller §2 narrative. Where they
  differ, both are recorded.

---

## 1. The five dimensions — definitions and allowed values

### D1 — Data Origin
- **Allowed values [AAAI]:** `native | translated` (binary).
- **Definition [AAAI]:** "Confronts the 'translationese' problem by quantifying
  reliance on machine- or human-translated content versus natively sourced data."
- **Definition [41p §2.2]:** "Data Origin (Native vs. Translated): … quantifying the
  reliance on machine-translated versus natively-sourced data, exposing the hidden
  costs of synthetic data generation."
- **Corpus distribution [41p Table 1]:** Benchmarks — Native 48 (73.8%), Translated
  17 (26.2%). Datasets — Native 23 (74.2%), Translated 8 (25.8%).
- **Stated decision rule (mixed cases) [AAAI, Annotation and Meta-Analysis]:**
  "a resource combining native and translated subsets is classified by its majority
  composition."
- **[GAP] Human- vs machine-translated:** [AAAI] counts both as `translated`;
  [41p] §2.2's gloss says "machine-translated." A human-translated resource
  (e.g., PAWS-X, built from human translations) is `translated` under [AAAI] but
  ambiguous under the [41p] wording. Rule needed.
- **[GAP] Synthetic / model-generated data:** neither version says whether
  LLM-generated or template-generated content (e.g., minimal pairs generated from
  grammar templates; synthetic instruction data) counts as `native`,
  `translated`, or neither. The binary has no third value.
- **[GAP] Majority composition — majority of what?** Items? Languages? Tokens?
  Unstated. A resource native in 20 languages and translated in 25 low-resource
  languages flips value depending on the unit.

### D2 — Linguistic Resource Tier
- **Allowed values [both]:** `high | medium | low`.
- **Definition [AAAI]:** "Following established resource-level classifications
  [citation unresolved in the draft — prints as (?)] , maps the chasm between
  resource-rich and resource-poor languages."
- **Definition [41p §2.2]:** "By categorizing the resource level of each language,
  we map the vast and growing chasm between resource-rich and resource-poor
  languages."
- **Corpus distribution [41p Table 1]:** Benchmarks — High 46 (70.8%), Medium 15
  (23.1%), Low 4 (6.2%). Datasets — High 24 (77.4%), Medium 5 (16.1%), Low 2 (6.5%).
- **Stated decision rule [AAAI]:** "resource tier reflects the tier of the languages
  a benchmark is designed to evaluate, not incidental coverage."
- **[GAP] No thresholds anywhere.** Neither version names the external
  classification it follows (the [AAAI] citation is literally unresolved), gives
  cut-offs (speakers? corpus size? Joshi-style classes?), or lists which languages
  fall in which tier. A coder cannot reproduce a tier label from the paper alone —
  the single largest reproducibility gap in the taxonomy.
- **[GAP] Multi-tier resources:** a benchmark spanning high- and low-resource
  languages gets one label; the rule says "the languages it is designed to
  evaluate" but not how to aggregate when the designed set spans tiers (lowest?
  modal? language-count-weighted?).

### D3 — Task Class
- **Allowed values [AAAI]:** 13 classes — `QA, syntax, reasoning, multimodal, NLU,
  summarization, math, RAG, code, generation, long-context, embeddings, other`.
- **[41p] discrepancy:** the 41-page Figure 1 / Table 1 use only 10 benchmark
  classes (QA 20, Syntax 12, Reasoning 6, Multimodal 6, NLU 5, Other 5,
  Summarization 4, Math 3, RAG 2, Code 2) — `generation`, `long-context`, and
  `embeddings` do not appear. The class set grew between versions without a stated
  mapping.
- **Datasets use a different axis [both]:** "primary purpose" — `pre-training`
  (20), `translation/alignment` (8), `instruction fine-tuning` (3). Whether D3 for
  datasets is the same dimension with different values, or a separate variable, is
  never stated.
- **Definition [41p §2.2]:** "This axis provides a granular view of the tasks being
  evaluated, highlighting the community's over-emphasis on certain tasks (like QA)
  at the expense of others."
- **[GAP] Class boundaries:** no definitions per class. Overlaps are routine:
  reading comprehension (QA or NLU?), math word problems (math or reasoning?),
  retrieval-augmented QA (RAG or QA?), code generation (code or generation?),
  embedding benchmarks (embeddings or NLU?). No primary-task rule is stated for
  multi-task suites (e.g., a suite spanning 8 tasks gets one class — which?).
- **[GAP] "Other" (5 benchmarks in [41p])** is undefined — a coder cannot tell what
  qualifies.

### D4 — Cultural Specificity
- **Allowed values [both]:** `universal | culturally anchored | region-specific`.
- **Definition [AAAI]:** "distinguishes benchmarks that are genuinely
  culture-independent from those that merely strip cultural context, quantifying
  Anglocentric and Western-centric defaults."
- **Definition [41p §2.2]:** "designed to measure the pervasive Anglocentric and
  Western-centric bias. It allows us to quantify how many benchmarks are truly
  universal versus those that are merely masquerading as such."
- **Corpus distribution [41p Table 1]:** Benchmarks — Universal 36 (55.4%),
  Culturally Anchored 17 (26.2%), Region-Specific 12 (18.5%). Datasets — Universal
  22 (71.0%), Anchored 6 (19.4%), Region-Specific 3 (9.7%).
- **[GAP] Anchored vs region-specific boundary** — the dimension's softest edge,
  and the paper's own §4.3 prose slides between the two ("'Culturally Anchored'
  (26.2%) and 'Region-Specific' (18.5%)"). No operational test is given: Is a
  benchmark built from one country's national exams anchored or region-specific?
  Is a pan-African benchmark region-specific (region = continent) or anchored?
  Is content in a local language but on universal topics (translated MMLU)
  universal or anchored?
- **[GAP] Universal vs stripped:** [AAAI] distinguishes "genuinely
  culture-independent" from "merely strip cultural context" — but both presumably
  code as `universal`, so the distinction the prose draws has no code. A coder
  cannot tell whether a translated, culture-stripped benchmark is `universal` by
  design or by stripping.

### D5 — Inference Paradigm
- **Allowed values: NOT ENUMERATED in either version.** This is the least
  specified dimension.
- **Definition [AAAI]:** "Captures the nature of the reasoning demanded
  (retrieval-style vs. multi-hop) and the evaluation regime (direct multilingual
  inference vs. Translate-Test), probing the brittleness of measured performance."
  — i.e., the dimension conflates TWO sub-axes (reasoning depth; evaluation
  regime) with no stated value set for either and no rule for combining them.
- **Definition [41p §2.2]:** "examines the nature of the reasoning required by the
  benchmark, distinguishing between simple retrieval and more complex, multi-hop
  reasoning" — mentions only the reasoning-depth sub-axis; the Translate-Test
  regime appears in [41p] only as a finding, not as a coded value.
- **[GAP] Applicability to datasets:** an inference paradigm is a property of an
  evaluation, not of a training corpus. Neither version states what D5 value the
  31 datasets carry, or whether D5 is benchmarks-only.
- **[GAP] No distribution for D5 is reported in [41p] Table 1** (the only dimension
  with no published marginals), so there is no published label set to validate a
  re-coding against. Until the authors' D5 labels are supplied, D5 cannot be part
  of an agreement computation — only of a fresh coding.

---

## 2. Corpus identification and screening (for the 142 → 96 screening log)

**Stated method [41p §2.1]:**
- Search venues: ACM Digital Library, arXiv, ACL Anthology, Hugging Face Datasets.
- Scope: multilingual / cross-lingual evaluation resources, 2018–2025 [41p];
  [AAAI] refines: benchmarks 2018–2025, training corpora 2005–2025.
- Seed terms [41p]: "multilingual evaluation," "cross-lingual NLP," "low-resource
  benchmarks," "multimodal reasoning" (described, not given as an exact string list).
- Three-part screen (both versions agree): (i) public availability of the dataset
  or a detailed benchmark description; (ii) explicit multilingual or cross-lingual
  scope; (iii) clear, reproducible evaluation results or metrics.
- Then deduplication → retained set.

**The count problem (must be fixed in the log, not papered over):**
- [41p §2.1] says "142 unique resources—comprising 65 evaluation benchmarks and 31
  training datasets—were retained," but 65 + 31 = 96, and [41p §2.2] says "each of
  the 96 resources." The 41-page text is internally inconsistent.
- [AAAI] states it cleanly: "From 142 candidate resources, screening and
  deduplication retained 96." **The screening log must follow the [AAAI] framing:**
  142 candidates → 96 retained → 46 excluded, each with the failed criterion
  (i/ii/iii) or `duplicate` as the reason.
- **[GAP] The 46 excluded candidates are not listed anywhere** in either version.
  The log cannot be reconstructed from the paper; it must be rebuilt or the
  exclusion side of the log declared unavailable.
- **Stated exception [AAAI only]:** "A small number of influential monolingual
  resources are retained only where they anchor an explicitly multilingual lineage"
  (example given: BLiMP). This overrides screen (ii) and must appear in the log as
  an explicit exception reason, with the lineage named. **[GAP]** "Influential" and
  "anchor" are undefined; how many such exceptions exist is unstated ("a small
  number").

**Annotation sources [AAAI]:** "Each resource was annotated on all five axes from
its primary paper, dataset card, and repository." — This sentence is the evidence
base for the blind re-coding protocol (same sources, no labels).

---

## 3. Ambiguity register — what will force a decision rule during re-coding

| # | Dimension | Ambiguity | Paper's position |
|---|-----------|-----------|------------------|
| A1 | D1 | Human-translated: `translated` or its own thing? | [AAAI] implies translated; [41p] gloss says machine-translated only |
| A2 | D1 | Synthetic/template-generated data: which value? | Silent; binary has no value for it |
| A3 | D1 | "Majority composition" — majority of what unit? | Rule stated, unit unstated |
| A4 | D2 | Tier thresholds / external classification | Absent; source citation unresolved |
| A5 | D2 | Aggregating tiers across a multi-tier language set | "Designed to evaluate" stated; aggregation unstated |
| A6 | D3 | Class set differs between versions (10 vs 13) | Unreconciled |
| A7 | D3 | Multi-task suites: single class by what rule? | Silent |
| A8 | D3 | Class boundary overlaps (QA/NLU, math/reasoning, RAG/QA, code/generation) | Silent |
| A9 | D3 | Datasets coded on "primary purpose" — same dimension or separate? | Unstated |
| A10 | D4 | Anchored vs region-specific operational boundary | Absent; prose conflates them |
| A11 | D4 | Culture-stripped-by-translation: `universal`? | Prose distinguishes, code doesn't |
| A12 | D5 | Value set never enumerated; two sub-axes conflated | Absent entirely |
| A13 | D5 | Applicability to the 31 datasets | Unstated |
| A14 | Screen | Monolingual-lineage exception criteria ("influential", "anchor") | Example given, criteria absent |
| A15 | All | "Designed to evaluate" intent vs actual contents when they diverge | Partially stated for D2 only |

Two of these were flagged in the AAAI draft's own author notes as the dimensions
needing explicit decision rules (cultural specificity of translated benchmarks;
multi-tier classification). The register above shows the full set a blind coder
will actually hit.

---

## 4. What the paper does NOT claim (guardrails for the FAccT cut)

- No inter-annotator agreement, double-coding, or adjudication is reported in
  either version. The [AAAI] limitations say only that annotation "involves
  judgment, which we mitigate with public classification criteria and released
  annotations open to community correction."
- No codebook beyond §2.2 / the D1–D5 glosses exists in either version. This
  document is a reconstruction of the stated criteria, not a recovery of a
  fuller original.
