# Blind IAA protocol — LinguaGapAtlas re-coding study (DRAFT)

**Purpose:** Produce a defensible inter-annotator agreement figure for the
five-dimension taxonomy, for the FAccT 2027 cut. The protocol is blind: the second
coder never sees the paper's labels until both codings are locked.
**Companion document:** `codebook-draft.md` (same directory) — the only coding
instructions the second coder receives, plus this protocol's §3.
**Status:** Draft for Sankalp's approval. Compute runs only on his word; nothing
here has been executed against the paper's labels yet.

---

## 1. Design

- **Unit of coding:** one resource × five dimensions (D1–D5), exactly as the
  paper's annotation (per the AAAI version: "Each resource was annotated on all
  five axes from its primary paper, dataset card, and repository").
- **Codings compared:**
  - *Coding A* — the paper's original labels (from the authors' released
    annotations / the corpus file built from them). Sealed until Coding B locks.
  - *Coding B* — an independent second coding produced under this protocol.
- **Coder identity — disclosed, not hidden:** Coding B is produced by an AI
  annotator (Muse) working from the same sources the paper used. In the paper
  this must be reported as what it is — a human–AI agreement study ("an
  independent second coding pass by an AI annotator, blind to the original
  labels"). It must never be described as a second human annotator.
- **D5 caveat:** no published D5 labels exist (the 41-page Table 1 reports no D5
  marginals). D5 enters the agreement computation only if the authors' original
  D5 labels are supplied with Coding A. Otherwise D5 is reported as a fresh
  coding with no agreement statistic — stated plainly, not silently dropped.

## 2. Sample design

- **Preferred: full census.** The corpus is 96 resources; double-coding all 96 is
  feasible and eliminates sampling caveats entirely. Agreement is then a
  population statistic, not an estimate.
- **Fallback (if time forces a sample):** stratified random sample of **n = 40**,
  strata = resource type (benchmark/dataset) × D2 tier (from Coding A marginals,
  used for stratification only, never shown to the coder), with deliberate
  over-sampling of the rare cells (low tier: only 4 benchmarks + 2 datasets;
  region-specific datasets: 3) so agreement on rare values is estimable at all.
  Report the sampling frame and per-stratum n.
- **No convenience sampling.** The first 40 alphabetically or the 40 easiest to
  document would bias agreement upward and is not acceptable.
- Whichever design is used is fixed before Coding B starts and reported in the
  paper.

## 3. Coder instructions (what Coding B sees)

The coder receives, per resource: name, and links to its **primary paper,
dataset card, and repository** — the paper's own stated annotation sources.
The coder receives `codebook-draft.md` §§1–2 (definitions, allowed values, the
two decision rules the paper states) and these standing instructions:

1. Code from the sources only. Do not consult the LinguaGapAtlas paper,
   its tables, figures, or any summary of its findings.
2. One value per dimension per resource, from the allowed set. If a dimension
   genuinely cannot be determined from the sources, code `undetermined` — an
   explicit null. Never guess to fill a cell, and never impute from similar
   resources. (`undetermined` cells are excluded from that dimension's kappa
   denominator and reported as their own count.)
3. For every cell, record a one-line evidence note (what in the source determined
   the value) and a confidence flag (clear / borderline). Borderline flags are
   data, not weakness — they locate the ambiguous definitions.
4. Where the codebook is ambiguous (the register in `codebook-draft.md` §3),
   apply your best reading of the stated definition, flag the cell with the
   ambiguity ID (A1–A15), and move on. Do **not** invent a rule mid-coding, and
   do not change earlier cells after encountering a hard case — consistency of
   one reading beats case-by-case optimisation.
5. Code resources in the randomised order given; do not code in the paper's
   presentation order (which groups by task class and would leak D3).

**Blinding mechanics:** Coding A lives in a separate file the coder never opens;
the coding form given to the coder contains no label columns; the comparison
script runs only after Coding B is complete and frozen (hash recorded).

## 4. Adjudication procedure

1. **Freeze both codings.** Record checksums; no edits after this point except
   through adjudication, which produces Coding C (adjudicated) as a *separate*
   file — A and B are never overwritten.
2. **Disagreement list:** every cell where A ≠ B, with the dimension, both
   values, the coder's evidence note, and the ambiguity ID if flagged.
3. **Adjudication meeting (Sankalp + co-authors):** for each disagreement,
   decide (a) A stands, (b) B stands, or (c) the definition itself is at fault.
   Every (c) must produce a written decision rule, added to the codebook as a
   numbered rule, in the authors' words. This list is the deliverable that
   converts disagreements into the paper's missing decision rules.
4. **Statistics are computed on A vs B (pre-adjudication).** Agreement computed
   after adjudication would be meaningless and must not be reported. Coding C is
   the corpus's corrected label set going forward, and the paper should say the
   released annotations are the adjudicated set.
5. If adjudication changes a label that feeds a headline number (6.2% / 55.4% /
   26.2% / tier and culture marginals), the number is recomputed from Coding C
   and the paper updated — or the change is reported. No silent swaps.

## 5. Statistics to report (exact list)

For each dimension D1–D5 separately, and overall:

1. **Percent agreement** (raw observed agreement) — always alongside kappa,
   never alone.
2. **Cohen's kappa** per dimension (nominal categories; two codings, A vs B).
   Overall kappa reported as the mean of per-dimension kappas AND a pooled
   kappa, labelled as such — the five dimensions are not interchangeable items,
   so a single pooled figure alone would mislead.
3. **95% confidence intervals** for each kappa (bootstrap over resources,
   10,000 resamples).
4. **Confusion matrix** per dimension (A rows × B columns) — shows *where*
   disagreement concentrates (expected: D4 anchored vs region-specific; D2
   medium vs low), which is what the decision rules then target.
5. **Marginal distributions** for both codings per dimension.
6. **Prevalence caveat, stated in the paper:** several dimensions are heavily
   skewed (D1 ≈ 74/26; low tier ≈ 6%; region-specific datasets = 3). Under
   skew, kappa is depressed even at high raw agreement (the kappa paradox).
   Therefore also report **PABAK** (prevalence-adjusted, bias-adjusted kappa)
   and/or **Gwet's AC1** per dimension as a robustness figure, and interpret
   kappa together with raw agreement and the marginals — never a bare kappa.
7. **Per-value agreement** for rare values (low tier; region-specific):
   agreement on that value specifically (e.g., of the resources either coder
   called `low`, how many both called `low`). With counts this small, report
   the counts themselves, not just a coefficient.
8. **`undetermined` counts** per dimension for Coding B, and how many A cells
   had no determinable source — a documentation-quality finding in its own
   right.
9. **Borderline-flag rate** and the ambiguity-ID tally: which codebook gaps
   (A1–A15) actually fired in practice. This is the empirical basis for the
   final decision-rule list.

**Interpretation bands** (report the numbers; if bands are quoted, use the
conventional Landis–Koch labels and name them as a convention, not a verdict):
0.61–0.80 substantial, 0.81+ almost perfect. The paper's claim should rest on
the per-dimension pattern, not on clearing a single threshold.

## 6. Reporting template (for the FAccT cut's methods section)

> "All 96 resources were independently re-coded on the five dimensions by a
> second annotator (an AI coding assistant) blind to the original labels,
> working from the same sources (primary paper, dataset card, repository) and
> the published classification criteria. Agreement was [X]% overall; Cohen's
> kappa per dimension ranged from [κmin] (D_) to [κmax] (D_) [95% CIs].
> Disagreements (n = _) were adjudicated by the authors; each definitional
> disagreement produced an explicit decision rule, listed in the codebook.
> Released annotations are the adjudicated set."

Every bracket is filled from §5 outputs. If the fallback sample is used,
"All 96" becomes the sample description from §2. No bracket may be filled with
an estimate, a target, or a number from another study.

## 7. Open decisions reserved to Sankalp / co-authors

1. Census (all 96) vs stratified sample (n = 40).
2. Whether the authors' original D5 labels exist and can be supplied for
   Coding A; if not, D5 is fresh-coded only.
3. Adjudication membership and scheduling (needs at least Sankalp; ideally
   one co-author).
4. Final wording of every decision rule produced in adjudication — the
   protocol generates the occasions for the rules; the authors write them.
5. Whether and where the human–AI nature of Coding B is disclosed beyond the
   methods sentence in §6 (recommended: also in the Generative AI Disclosure
   Statement the FAccT form requires).
