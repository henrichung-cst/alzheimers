# Plan — make NSCLC reference & within-cohort directly comparable on per-state detection

## Problem

Kinase MEA infers kinase *activity* from phosphosite abundance. It sometimes
attributes that activity to a T-cell state where the kinase is **not expressed**
— a biological false positive. We want a reference comparison that catches this:
"is this kinase actually present in *this* T-cell type?"

The current two specificity numbers can't answer it and aren't comparable:

- **within-cohort `tcell_specificity`** — share of mean per-cell log-expression
  over the **14 ProjecTILs T-states** (`alz/cross_reference/tcell_within_cohort.py`,
  `_compute_specificity`). Interpretation: *which T-state preferentially expresses
  the kinase* (intra-T ranking).
- **NSCLC reference `specificity_score`** — share over **7 coarse groups** (the 14
  T-states collapsed to one `T_NK`; `alz/reference/nsclc_expression.py`, guardrail
  2). Interpretation: *T compartment vs the 6 non-T TME lineages*.

They are orthogonal for two reasons:
1. **Resolution mismatch** — 14 T-states vs 7 coarse groups (T collapsed).
2. **Wrong instrument (the deeper cause)** — both lead with a *share*, a relative
   ranking. A share is not presence/absence. A kinase with near-zero expression
   gets a *high* share in whichever state has the fewest competing counts.

### The share is inversely predictive of truth (measured 2026-06-20)

Using the NSCLC reference's native per-14-state detection (already computed) as
the authoritative detector, against the within-cohort share localizer, on
bulk-significant (FDR<0.25) kinases:

| within-cohort localizer tier | localized (kinase,state) pairs | NOT detected in reference |
|---|---|---|
| ≥2× | 302 | 250 (83%) |
| ≥5× | 48 | 46 (96%) |
| =10× (peak) | 14 | 13 (93%) |

13/14 peak-localized kinases are absent from the reference at the state the share
assigned them: FGFR2→CD4.Tfh, LRRK2→CD4.Treg, DCLK1→CD8.TEMRA, MYO3A→CD8.TPEX,
CAMK1G/MARK1/RIPK4/HUNK→CD4.Th17, etc. — stromal/epithelial/tissue kinases that
are biologically impossible in T-cells. **The harder the share localizes, the more
likely it is a false positive.** Switching the comparison axis to detection does
not just add a column — it corrects a localizer that is actively misleading.

## Decision (approved 2026-06-20)

Compare on **per-(kinase, T-state) detection** at the shared 14 ProjecTILs-state
vocabulary, on both datasets. The primary statistic is **fraction of cells
expressing** (count > 0) — it needs **no normalization**, so it is identical in
meaning across the two pipelines and immune to the share artifact. Mean log-
expression is shown as secondary; a binary `detected` flag combines them.

The MEA false-positive flag: an MEA-attributed `(kinase, T-state)` is flagged when
the kinase is **not detected** in that exact state. The 897k-cell NSCLC reference
(~5,500 cells per T-state) is the authoritative detector; within-cohort donor1
(single shallow library) corroborates.

The coarse `T_NK`-vs-stroma lineage share is **retained but demoted** to a separate
"is this even a T-lineage kinase" audit (the drawer lineage strip). It is no longer
the per-state comparison axis.

## State crosswalk (1:1, both derive from ProjecTILs `functional.cluster`)

| within-cohort (sanitized) | NSCLC (ProjecTILs raw) |
|---|---|
| CD4CTLeomes | CD4.CTL_EOMES |
| CD4CTLexh | CD4.CTL_Exh |
| CD4CTLgnly | CD4.CTL_GNLY |
| CD4Naive | CD4.NaiveLike |
| CD4Tfh | CD4.Tfh |
| CD4Th17 | CD4.Th17 |
| CD8CM | CD8.CM |
| CD8EM | CD8.EM |
| CD8MAIT | CD8.MAIT |
| CD8Naive | CD8.NaiveLike |
| CD8TEMRA | CD8.TEMRA |
| CD8Tex | CD8.TEX |
| CD8Tpex | CD8.TPEX |
| Treg | CD4.Treg |

NSCLC `T_NK_other` (scGate-rejected marker-T) has no ProjecTILs analog → excluded
from the per-state comparison (still contributes to the coarse T_NK lineage share).
Crosswalk lives in one place (`build_tcell_viewer.py`), where both sources are
already consumed.

## Phasing (gated)

> **DONE 2026-06-20.** Superseded/absorbed by the repo-wide standard
> (`standard_attribution_metric.md`): the reference-side detection + MEA
> false-positive flag below shipped, *plus* the magnitude-aware breadth metric
> (effective number of cell types) and detection-gated `top_celltype`. The share
> was removed, not demoted. Phase 2 (within-cohort detection) remains as written.

### Phase 1 — reference-side detection + flag (no recompute; buildable now)
The NSCLC reference already carries per-14-state `mean_log2_expression`,
`fraction_cells_expressing`, `binary_expressed`. Nothing to recompute.

- `build_tcell_viewer.py`: in the attribution index, attach per-(kinase, state)
  reference detection (`nsclc_frac`, `nsclc_detected`) via the crosswalk, and a
  `mea_false_positive` flag = (row's kinase bulk-significant at the contrast) AND
  (within-cohort tier localizes it here, tier≥2×) AND (reference `nsclc_detected`
  is False). Ship a per-state reference-detection column generally.
- `kinase_audit.js` + `styles.css`: replace the constant `nsclc_tier` verdict
  column with a **per-state "NSCLC detected"** column (fraction % + ✓/✗), and a
  prominent false-positive badge on flagged rows that visually overrides a high
  within-cohort tier. Keep the drawer lineage strip (coarse T_NK audit).
- Deliverable: the verdict table answers "is this kinase present in this exact
  T-state per an independent 897k-cell reference," and explicitly marks the
  share-localized-but-absent rows as MEA false-positive candidates.

### Phase 2 — within-cohort fraction (R extractor change + re-run; gated compute)
Make detection symmetric so both columns mean the same thing.

- `alz/ingest/tcells_scrna_extract.R`: emit `pct_expressing.csv` (gene × state__day
  fraction of cells with count > 0). The counts matrix is already in memory
  (`GetAssayData(... layer="data")`); add one grouped `rowMeans(x > 0)` per
  `ts_group` alongside the existing AggregateExpression loop.
- Re-run `pixi run tcells-scrna-extract` under a memory cap
  (`systemd-run --user --scope -p MemoryMax=<N>G -p MemorySwapMax=0`), then
  `pixi run tcell-within-cohort`. **Gated** — needs the donor Seurat object;
  stop and report before running.
- `alz/cross_reference/tcell_within_cohort.py`: read `pct_expressing.csv`, pool
  over days → `tcell_fraction_expressing[gene,state]`; add `tcell_detected`. Add
  both to `tcell_specificity.csv` and `unified_attribution_tcells.csv`.
- Viewer: show within-cohort detection beside reference detection
  ("detected in both / reference only / cohort only / neither").

## Open decisions (resolve before / during implementation)

1. **Fate of `tcell_specificity` / `tcell_tier` in the viewer.** Detection becomes
   the primary axis + flag. Options: (a) keep the share tier as a labeled secondary
   descriptor with the FP flag overriding it; (b) drop the share tier column
   entirely (anti-shim — it's been shown inversely predictive for localization).
   Recommend (a) for one release with the flag, then (b).
2. **Detection-gated localization (possible Phase 3).** The attribution's
   `top_celltype` is currently share-driven (the inversely-predictive metric).
   Should localization itself be gated/weighted by detection (never localize a
   kinase to a state where it's undetected)? This changes `recover.py`/attribute
   logic, not just display. Raise after Phases 1–2 land.
3. **Detection thresholds.** Reference binary = `mean log2(CPM+1) > 1 AND frac >
   0.10`. Lead the cross-dataset call on **fraction > 0.10** (normalization-free);
   mean shown as secondary. Confirm the 0.10 cut or sweep it.

## Constraints

- Memory: Phase 2 R re-extraction runs capped; never load the Seurat object or any
  matrix whole outside the capped job. Fraction is computed inside R from the
  already-loaded assay, not by a downstream whole-frame read.
- No data files committed (repo `.gitignore` is `*`; `git add -f` code only).
- Anti-shim: when detection replaces the share as the comparison/flag axis, update
  docstrings, the explainer text, comments, and TODO.md in the same pass.
