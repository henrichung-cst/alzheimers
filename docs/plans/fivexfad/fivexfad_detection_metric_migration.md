# 5xFAD → standard detection metric (Phase 3)

> **Status: DONE 2026-06-22.** All steps below landed; completion recorded in
> [`standard_attribution_metric.md`](../attribution/standard_attribution_metric.md) Phase 3.
> Verified end-to-end: `pixi run 5xfad-snrna-specificity` regenerated the CSVs
> (35,144 rows, 14,898 detected), the package build wrote 382 attribution shards
> (35 promoted to very_high), and `test_fivexfad.py` passes (9 tests).

Closes the "consistency pass" left pending in
[`standard_attribution_metric.md`](../attribution/standard_attribution_metric.md). 5xFAD is the
last cohort still on legacy share/τ specificity. After this, **one metric, every
cohort** is literally true.

## Problem

5xFAD within-cohort snRNA attribution is built by
`alz/ingest/build_5xfad_snrna_attribution.R`, which computes its own
share-based localizer — not `alz/cross_reference/specificity.py`:

```r
spec_by_cluster <- tapply(expm1(pb), cell_type, mean)   # natural-log de-log
share  <- spec_by_cluster / sum(spec_by_cluster)        # share, no detection gate
tau    <- Σ(1 - x/max) / (n-1)                           # tissue-specificity index
fold   <- share[cell_type] / (1/n_clusters)
tier   <- fold>=2 "high" / fold>=1 "moderate" / "low"
```

This is exactly the share-is-not-presence failure the standard metric was built to
remove: a near-zero kinase scores a high share wherever cross-cluster competition is
lowest, with **no detection gate** to drop the noise cell types first. The output
columns `fivexfad_specificity`, `fivexfad_fold_over_uniform`, `fivexfad_tau`,
`fivexfad_top_cluster` are therefore not comparable to the `*_concentration` /
`*_detected` / `effective_n` columns every other cohort now reports. The 5xFAD WMB
cross-check is also still a share (`wmb_specificity`, `wmb_fold_over_uniform`),
joined from Song's `celltype_evidence_table.csv`.

## Seam (no new metric — reuse `specificity.compute`)

`specificity.py` is Python; the 5xFAD producer is R reading a large Seurat RDS. We
keep **one definition** by mirroring Song's seam (`snrna_integration.py`): the R
script becomes a pure pseudobulk + **detection** exporter, and a Python step calls
`specificity.compute`. We do **not** reimplement the metric in R (anti-shim).

Log-base bridge (exact, no approximation): Seurat's `data` layer is `ln(1+x)`, so
`mean_log2 = mean_ln / ln(2)` per cell type, and `delog2(mean_log2) = 2^(mean_ln/ln2) − 1
= expm1(mean_ln)` — the same linear weight the current script already uses. The only
genuinely **new** input the producer must emit is `fraction_cells_expressing` (the
detection gate), computed from the **counts** layer: fraction of cells with count > 0
per (gene, tissue, cell_type).

## Resolution

Native = the 46-cluster `new_clusters` spine, computed **per tissue**
(cortex / hippocampus) — 5xFAD's existing split, not a coarse rollup. Call
`specificity.compute` once per tissue subset (`group_col=None`, parity with Song's
native within-cohort call). Specificity is contrast-invariant; it is broadcast across
the age × genotype contrast rows exactly as today.

## Changes

### 1. Producer — `alz/ingest/build_5xfad_snrna_attribution.R`
- Add per-(gene, tissue, cell_type) **`fraction_cells_expressing`** from the counts
  layer (`fraction = nnzero(counts[gene, cells]) / length(cells)`), cell-pooled across
  samples/ages/genotypes within the tissue+cluster.
- Emit per-(gene, tissue, cell_type) **`mean_log2_expression` = mean_ln / log(2)** and
  **`n_cells`**, to a new `fivexfad_snrna_expression.csv`.
- **Delete** `specificity_tau`, `location_tier`, `share`, `fold`, `uniform`, and the
  `fivexfad_specificity / _fold_over_uniform / _tau / _top_cluster / confidence_tier /
  confidence_basis` columns from the attribution CSV. Keep the **direction** columns
  (`fivexfad_lfc`, `fivexfad_pval`, `fivexfad_fdr`, `n_snrna_samples_*`, `n_cells_*`) —
  that is the disease-direction signal, analogous to Song's concordance, and stays.

### 2. New Python step — `alz/cohorts/fivexfad/snrna_specificity.py` (+ pixi task)
- Read `fivexfad_snrna_expression.csv`; for each tissue call
  `specificity.compute(df, gene_col="gene_symbol", label_col="cell_type",
  mean_log2_col="mean_log2_expression", frac_col="fraction_cells_expressing",
  ncells_col="n_cells")`.
- Write `fivexfad_expression_specificity.csv` with `fivexfad_`-prefixed standard
  columns: `fivexfad_detected`, `fivexfad_concentration`,
  `fivexfad_concentration_of_total`, `fivexfad_concentration_tier`,
  `fivexfad_fraction_cells_expressing`, `fivexfad_effective_n`,
  `fivexfad_top_celltype`, `fivexfad_top_concentration`.
- Wire `5xfad-snrna-specificity` into `pixi.toml`, inserted between
  `5xfad-snrna-attribution` and `5xfad-viewer`'s `depends-on`.

### 3. Consumer — `alz/viewer/cohorts/fivexfad.py`
- `_build_fivexfad_attribution_rows`: join `fivexfad_expression_specificity.csv` on
  (kinase/gene, tissue, cell_type); replace the share columns with the standard ones.
  For the WMB cross-check, join `wmb_detected / wmb_concentration /
  wmb_concentration_tier` from `celltype_evidence_table.csv` (already present —
  build_unified_viewer.py:886) instead of `wmb_specificity / wmb_fold_over_uniform`.
- `_assign_fivexfad_song_aligned_confidence`: gate on `fivexfad_detected` (replaces
  the "no usable location" / sparse-basis checks) and `fivexfad_concentration_tier`
  (`≥ 2` → high, `≥ 1`/detected → moderate), replacing `fivexfad_fold_over_uniform`.
  The bulk-MEA-significance + direction-support gates are unchanged.
- `_promote_fivexfad_attribution_confidence`: unchanged (decomp-agreement promotion is
  metric-independent).
- `_f5_attr_record_cmp`, `_build_fivexfad_attribution_summary_index`, the `celltypes`
  payload: swap `fivexfad_specificity` → `fivexfad_concentration` (and the `top_*`
  fields likewise) in the sort keys and emitted dicts.

### 4. Viewer JS — `kinase_fivexfad.js`, `kinase_crosstable.js`
- Render the detection cell (✓/✗ + frac%) and `concentration_tier` pill, matching the
  Song/T-cell tabs (per Phase 2C/2D). Remove the share fold-pills and any
  `fivexfad_specificity` / `wmb_specificity` display. Reuse the shared detection-cell
  widget; do not reimplement.

### 5. Test — `alz/ingest/test_fivexfad.py`
- Update column assertions to the standard schema. Add a synthetic-input unit test for
  the new Python step (seeded fixture, no committed data) asserting `detected`,
  `concentration_tier`, and `effective_n` against hand-computed values — mirrors Song's
  specificity test.

### 6. Docs
- `docs/integrations/5xfad-kinase-mea-viewer.md` and
  `docs/plans/todo3_standardized_csv_export.md`: update column references.
- `docs/foundation/specificity_confidence.md`: flip the "5xFAD migration pending" line
  to done once landed.
- `standard_attribution_metric.md`: mark Phase 3 DONE.

## Sequencing (gated — the regen is yours to run)

1. **I implement** steps 1–2 (producer + Python step + pixi wiring).
2. **You run** `pixi run 5xfad-snrna-attribution && pixi run 5xfad-snrna-specificity`
   — heavy Seurat read of `fivex_renamed_from_merged.RDS`; off-limits for me to run
   in-session under the shared-box memory rule. Run it in tmux under a memory cap.
3. **I implement + verify** steps 3–6 against the regenerated CSV, then
   `pixi run 5xfad-viewer` and confirm the 5xFAD tab renders detection cells with no
   share pills. Report pass/fail.

We stop at the boundary after step 1 so the new columns exist before any consumer
references them — no half-migrated state where the viewer reads columns the producer
hasn't emitted yet.

## Anti-shim checklist
- `fivexfad_specificity`, `fivexfad_fold_over_uniform`, `fivexfad_tau`,
  `fivexfad_top_cluster` deleted everywhere — not kept beside the new columns.
- `specificity_tau` / `location_tier` R helpers deleted.
- 5xFAD WMB cross-check moved off `wmb_specificity` shares to WMB detection columns;
  `config.wmb_specificity_uniform()` removed if 5xFAD was its last caller (verify).
- Prior on-disk `fivexfad_snrna_attribution.csv` may stay as historical record; no code
  path reads its retired columns.
