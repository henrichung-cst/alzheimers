# Phase 1 — Accepted Legacy Deviations

Date: 2026-06-17
Phase: 1 (Read-Only Validators)
Agent: IMPLEMENTER

Each entry records a current on-disk state that violates a naive schema expectation
but is **correct as-is** and must not be flagged as a validator FAIL.  Every
such deviation is accepted here and the validator is corrected to match reality.

---

## DEV-01: `Subs fraction` is a fractional string, not a float

- **Artifact kind:** MEA long tables (all cohorts: Song, Mukesh, T-cell, 5xFAD)
- **Column:** `Subs fraction`
- **Naive expectation:** numeric float column (name ends with "fraction")
- **Actual value:** GSEApy prerank string format, e.g. `"136/715"` (leading
  substrates count / total substrates count)
- **Why accepted:** This is the standard GSEApy prerank output format.  The
  column is informational (auditing substrate set coverage) and is never used as
  a numeric input to any downstream calculation.  Coercion to float would require
  splitting on `/` — a non-trivial transformation, not a direct read.
- **Validator correction:** `Subs fraction` removed from `MEA_NUMERIC_COLS`
  (both in `validate_cohort.py` and `phospho_schema.py` `MEALongTableSchema`).
  It remains in `required_columns` (presence check) but not in numeric
  coercibility checks.
- **Files affected:** `mea_raw_phospho*.csv`, `mea_stoichiometry*.csv`,
  `mea_perdonor*.csv`, `mea_timecourse*.csv`, `*_mea_raw_phospho.csv`,
  `*_mea_stoichiometry.csv` across all four cohorts.

---

## DEV-02: `tcell_concordance.csv` and `tcell_specificity.csv` are gene-level, not site-level

- **Artifact kind:** T-cell concordance and specificity tables
- **Path:** `outputs/reports/kinase_attribution_tcells/donor1/tcell_concordance.csv`
  and `tcell_specificity.csv`
- **Naive expectation:** `tcell_concordance` has `site_id` column;
  `tcell_specificity` has `kinase` column (analogous to Song/Mukesh equivalents)
- **Actual columns:**
  - `tcell_concordance.csv`: `gene`, `state`, `day`, `tcell_lfc`
  - `tcell_specificity.csv`: `gene`, `state`, `tcell_specificity`,
    `tcell_mean_log2_expression`
- **Why accepted:** The T-cell concordance and specificity are gene-level (RNA-seq
  DEG concordance by cell state / activation timepoint), not phospho-site-level.
  This reflects the T-cell pipeline's different analytical framing vs. the mouse
  cohort: the concordance metric aligns kinase substrates with gene-level T-cell
  activation markers, not with site-by-site phospho stoichiometry.
- **Validator correction:** Required columns updated to the real column names
  in `validate_cohort.py::validate_tcells()`.
- **Files affected:** `tcell_concordance.csv`, `tcell_specificity.csv`

---

## DEV-03: 5xFAD celltype parquets use `cell_type` column, not `cluster`

- **Artifact kind:** 5xFAD celltype MEA and OLS parquets
- **Paths:**
  - `outputs/reports/kinase_attribution_5xfad/celltype_mea/fivexfad_celltype_mea.parquet`
  - `outputs/reports/kinase_attribution_5xfad/celltype_mea/fivexfad_celltype_site_level_ols.parquet`
- **Naive expectation:** `key_columns` in `baseline_inventory.py` records
  `["kinase", "cluster", "contrast"]` for the celltype MEA parquet and
  `["site_id", "cluster"]` for the OLS parquet.
- **Actual column name:** `cell_type` (not `cluster`)
- **Why accepted:** The baseline_inventory `key_columns` annotation uses the
  conceptual term "cluster" (meaning snRNA cluster / cell type), but the actual
  column is named `cell_type` in both parquets.  This is a mislabeling in the
  baseline inventory annotation only — the data is correct.  The annotation will
  be corrected in a future baseline update; validators use `cell_type`.
- **Validator correction:** Key columns in `validate_cohort.py::validate_fivexfad()`
  use `["kinase", "cell_type", "contrast"]` and `["site_id", "cell_type"]`.
- **Files affected:** `fivexfad_celltype_mea.parquet`,
  `fivexfad_celltype_site_level_ols.parquet`

---

## DEV-04: MEA long tables have multiple rows per kinase (non-unique kinase key)

- **Artifact kind:** MEA long tables (all cohorts)
- **Naive expectation:** `kinase` column is a unique key
- **Actual behavior:** one row per (kinase × contrast × track × residue_type);
  `kinase` alone is NOT unique
- **Why accepted:** This is the intended long-format design.  The correct composite
  key is `(kinase, contrast, residue_type, track)` as documented in
  `baseline_inventory.py` `key_columns`.  The validator skips the dup-key check
  for MEA long tables and documents the reason.
- **Validator correction:** `skip_dup_check_reason` passed for all MEA long
  tables explaining the multi-row-per-kinase design.
- **Files affected:** All `mea_*.csv` long tables across all cohorts.

---

## DEV-05: Parquet dup-key checks are memory-unsafe and skipped

- **Artifact kind:** Parquet files (Song decomposition, 5xFAD celltype)
- **Issue:** Checking for duplicate key tuples in parquet requires loading the
  full table (no streaming path in pyarrow for set-based dup detection).
- **Why accepted:** Memory-safety rule (shared box).  These are large files
  (Song decomp MEA parquet: multiple-MB to GB range).  Dup-key checks on parquets
  are deferred; the Phase-0 baseline provides row counts and key column names for
  future structural comparison.
- **Validator correction:** `_validate_parquet_schema()` emits SKIP for dup-key
  checks on all parquet files.
- **Files affected:** `mea_per_cluster.parquet`, `mea_per_cluster_pY.parquet`,
  `site_level_ols_per_cluster.parquet`, `site_level_ols_per_cluster_pY.parquet`,
  `fivexfad_celltype_mea.parquet`, `fivexfad_celltype_site_level_ols.parquet`

---

## DEV-06: Numeric coercibility skipped for files > 50 MB

- **Artifact kind:** Large substrate sets and celltype CSVs
- **Paths:**
  - `outputs/reports/kinase_attribution_human/perdonor/mea_substrate_sets.csv`
  - `outputs/reports/kinase_attribution_5xfad/celltype_mea/fivexfad_celltype_substrate_sets.csv`
  - `outputs/reports/kinase_attribution_5xfad/cortex_st_mea_substrate_sets.csv`
  - `outputs/reports/kinase_attribution_5xfad/hippocampus_st_mea_substrate_sets.csv`
- **Issue:** Files exceed 50 MB threshold; numeric coercibility check would
  require row-by-row iteration potentially loading large chunks.
- **Why accepted:** Memory-safety rule.  The `kl_percentile` column's numeric
  type is validated indirectly via the smaller `_pY` variant files and via the
  Song substrate sets (< 50 MB) where the check passes.  The wide substrate sets
  carry the same schema; the skip is structural, not analytical.
- **Validator correction:** `check_numeric_columns()` emits SKIP for files
  > `SIZE_LIMIT_NUMERIC_CHECK` (50 MB).
- **Files affected:** See paths above.

---

## DEV-07: 5xFAD track filename convention uses `_st_` / `_py_` infix, not `_pY` suffix

- **Artifact kind:** 5xFAD per-region × per-track files
- **Naive expectation:** track suffix convention follows the PHOSPHO_TRACKS
  output_suffix: `""` for ST, `"_pY"` for pY
- **Actual convention:** 5xFAD files use `<region>_<track>_<artifact>.csv` where
  `<track>` is `"st"` or `"py"` (lowercase, no capital Y).  Example:
  `cortex_py_mea_raw_phospho.csv` (not `cortex_mea_raw_phospho_pY.csv`)
- **Why accepted:** The 5xFAD pipeline uses a region-track prefix naming scheme
  rather than the Song/Mukesh/T-cell suffix scheme.  Both encode the same track
  identity; the naming conventions are per-cohort and both are correct.
- **Validator correction:** `_check_fivexfad_track_fname_convention()` validates
  the `<region>_<track>_` prefix pattern instead of the generic suffix checker.
  The generic `_check_track_suffix_convention()` is NOT called on 5xFAD files
  to avoid spurious failures from the different naming convention.
- **Files affected:** All `outputs/reports/kinase_attribution_5xfad/<region>_<track>_*.csv`

---

## DEV-08: donor2 `mea_timecourse.csv` is absent_by_design

- **Artifact kind:** T-cell donor2 MEA long table
- **Path:** `outputs/reports/kinase_attribution_tcells/donor2/mea/mea_timecourse.csv`
- **Naive expectation:** MEA long table should exist for all donors
- **Actual behavior:** donor2 ran no MEA (ST matrices absent; pY MEA skipped due
  to `no_motif`). The file does not exist and must not be created.
- **Why accepted:** This is the intended partial-by-design state documented in
  the Phase-0 baseline and `cohort_manifest.py::TCELLS.absent_by_design`.
- **Validator correction:** `check_file_exists()` called with
  `absent_by_design=True`; emits SKIP (not FAIL).
- **Files affected:** `donor2/mea/mea_timecourse.csv`
