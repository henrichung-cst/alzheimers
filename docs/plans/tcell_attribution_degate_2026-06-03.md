# T-cell attribution: de-gate the viewer; concordance is shown, never used to filter

> Status: implemented 2026-06-03. Module `alz/cross_reference/tcell_within_cohort.py`; payload `build_tcell_viewer.py`; UI in the T-cell viewer templates.

## Why this change

A review of the just-shipped within-cohort T-cell attribution flagged that **concordance values are surprisingly low** and asked whether the transcriptomics evidence simply fails to match the MEA-predicted kinase activity — i.e. a confound. This is the diagnostic, on record.

### Interrogation (donor1, read-only)

- Concordant rows are **31%** (4046/13062) — *below* the ~50% a coin flip gives.
- It is **not anticorrelation.** Activity-sign and transcript-sign are **statistically independent**:
  - `P(transcript↑ | activity↑) = 0.465` vs `P(transcript↑ | activity↓) = 0.455` → difference **+0.010**.
  - 2×2 sign table **odds ratio = 0.98** (1.0 = independent).
  - The 31% is the arithmetic product of two *independent* marginal biases: MEA NES skews **down** (66% negative — an exhaustion-timecourse signature) while transcript skews **up** (~60%). Two independent signals with opposite marginal skews agree well below 50% by construction.
- **Not a baseline artifact.** Both sides use `value(day) − value(d2)` with the same sign convention (`+ = up at the later day`): MEA in `alz/ingest/tcells_perdonor.py:98-106` (docstring :12-14); concordance in `tcell_within_cohort.py` (`tcell_lfc = m[state,day] − m[state,d2]`). d2-aligned.
- **The published mouse/AD Song method reproduces the same independence** on its 45,689 rows: `OR = 1.011`, `P(song_lfc↑|NES↑)=0.498` vs `0.496`, overall sign-agreement **50.1% (exactly chance)**. The mouse only *looks* fine because its NES/song_lfc marginals are balanced (chance lands at 50%); ours are skewed (chance lands at 31%). Same underlying OR≈1.
- The "low values" are mostly **magnitude**: a single kinase gene's pseudobulk log-Δ vs d2 is tiny (median `|lfc|`=0.048; mouse 0.274), so even sign-agreeing concordance reads ~0.05–0.07.

### Conclusion

Concordance is a **directionally-uninformative** axis for kinases — expected, because the MEA infers activity from **substrate phosphorylation** (post-translational), which is decoupled from the kinase's own mRNA. It is co-evidence at best, not validation, and this holds in the published method too. **Specificity** (which states preferentially express the kinase) remains the valid, informative localizer and is unaffected.

## Decision (user directive)

The viewer "should not gate on concordance or specificity, it should **show** that information, with labels, but it should never discard, hide or remove information." So every hard/default gate is stripped; all axes (NES, specificity tier, concordance, consistency) are shown for the human to read. Sorting is allowed (reversible); hard filters and pre-drops are not.

## What changed

- **`alz/cross_reference/tcell_within_cohort.py`** — removed the concordant-only ship gate; ships the entire kinase×state×day grid (13,062 rows) as the single `unified_attribution_tcells.csv` (the `_full` twin is gone). Added a shown boolean label `tcell_concordant = tcell_concordance > 0` (never a filter). Guard `13062 = 311×14×3` retained.
- **`alz/build_tcell_viewer.py`** — `attribution_index` built from the full grid (NaN→null guarded; `tcell_concordant` field added); `top_celltype_1` is the most cell-type-specific state (tier, concordance only as tiebreak); single audit-source registration; `meta.tcell_attribution_caveat` carries the honest concordance note; per-context note appends it.
- **UI** — the verdict tab (`kinase_audit.js`) now shows **all 14 states** (removed the tier≥2 default hide + toggle) with an honest concordance caveat in the explainer and column tooltip. Concordance is never used to filter anywhere. Specificity stays a sortable labeled column + Cell-types pill (≥1× uniform). The `#ke-filter-tcell` specificity min-tier control is retained as an **opt-in narrowing tool** (defaults to Any/off — hides nothing by default; the `ke-count` indicator shows `visible / total` when active); specificity is the informative localizer, so an opt-in narrow on it is legitimate, unlike the concordance gate. `_KEY` bumped v7→v8 (`temporal_v2.js`).

  **Distinction that drove the design:** *default* gates that hid rows before any user action (verdict tier≥2 hide, the concordant-only ship gate) were removed outright. The *opt-in* specificity `<select>` defaults to off and only narrows when the user chooses to, so it hides nothing by default and was kept.

## Verification

1. `pixi run tcell-within-cohort` → 13,062 rows shipped, guard ✓, both concordant (4046) and discordant (9016) present.
2. `pixi run tcell-viewer` → `PAYLOAD.attribution_index` ~13k rows (donor1), `tcell_concordant` present, `meta.tcell_attribution_caveat` set.
3. Verdict tab shows all 14 states (no "N hidden"); concordance caveat visible; specificity `<select>` defaults to Any and only narrows on opt-in (count indicator reflects it). Hard-refresh.
4. `_tcellUniform()` (parse-time PAYLOAD-null fix) intact.
5. donor2 still shows "No IMAC kinase MEA", no attribution.
