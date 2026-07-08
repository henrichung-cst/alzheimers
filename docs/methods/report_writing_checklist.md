# Report Writing Checklist

Guidelines for writing analysis reports that anticipate reviewer concerns, derived from reviewer feedback on the male kinase enrichment report (March 2026).

---

## 1. Annotate rather than delete — show what was filtered out

When applying any filter (expression atlas, p-value, fold-change threshold), add a column flag or annotation to the filtered item instead of silently removing it. Include a supplemental table or "near-miss" section for entries that narrowly failed a threshold but show consistent biological patterns.

**Why:** Reviewers with a biology-first mindset distrust invisible filtering. If they can't tell what was removed and why, they assume something interesting may have been lost. Making the filtering traceable at the individual-entry level builds trust and lets the reader form their own judgment.

---

## 2. Lead with pattern consistency before effect size or p-value

Present evidence of reproducibility across conditions, timepoints, and cell types as the primary argument. Frame statistical metrics (p-values, effect sizes) as supporting evidence for a pattern the reader has already seen, not as the sole basis for claiming significance.

**Why:** Biological reviewers weight convergence across independent observations more heavily than the magnitude of any single measurement. A kinase that appears weakly but consistently across every condition and timepoint is more credible to this audience than one with a single dramatic p-value in one comparison.

---

## 3. Define every metric in a terminology section before first use

Include a short definitions box near the top of the report covering all abbreviations and derived metrics (e.g., LFF vs. LFC, enrichment score, adjusted p-value). Define at the point of introduction, not implicitly through usage.

**Why:** Even momentary confusion about a metric erodes trust in the precision of everything that follows. A reader who has to infer what LFF means from table context will question whether other details were equally loose. The cost of a 3-line terminology box is zero; the cost of ambiguity is cumulative.

---

## 4. Organize around the reader's primary analytical axis

Before structuring a report, identify which dimension the reader thinks along — cell type, timepoint, condition, pathway — and use that as the primary organizational axis. Nest other dimensions within it.

**Why:** A report organized comparison-first (by timepoint and condition) frustrates a reader who thinks cell-type-first. When the structure doesn't match the reader's mental model, they have to mentally re-sort every section, which increases cognitive load and produces requests to reorganize the data. Ask early: "What's your entry point into this data?"

---

## 5. Design figures for biological reading order: direction, then magnitude, then confidence

Encode the most biologically immediate information (up/down direction) most prominently. Use continuous scales for magnitude (percentile ranks over binary cutoffs where possible). Treat statistical confidence as accessible metadata, not a primary visual channel.

**Why:** Biologist reviewers read a figure by asking: "What went up? What went down? By how much?" If p-value occupies equal visual weight as direction, it competes for attention with the information the reader actually prioritizes. Confidence is important but should not dominate the visual hierarchy.

---

## 6. Include a "near-miss" section for threshold-adjacent results

For any hard threshold applied in the analysis, explicitly present entries that fell just below the cutoff but showed strong signals on other criteria (e.g., missed p-value but had high consistency, or missed fold-change but appeared in every condition).

**Why:** Hard cutoffs are necessary for primary analysis but create anxiety about lost signal. A near-miss section demonstrates the analyst considered what was excluded and provides the reviewer with the context to evaluate whether the thresholds were appropriate — preempting the "what did we lose?" question entirely.
