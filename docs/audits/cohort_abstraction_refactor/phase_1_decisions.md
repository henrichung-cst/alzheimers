# Phase 1 — Decision Log

Date: 2026-06-17
Phase: 1 (Read-Only Validators)
Agent: IMPLEMENTER

Decision log using the `agent_protocol.md` template.

---

## Decision: Required vs. optional columns

- Date: 2026-06-17
- Phase: 1
- Agent: IMPLEMENTER
- Files affected: `alz/core/phospho_schema.py`, `alz/core/validate_cohort.py`
- Decision: Only columns that are structurally required by downstream consumers
  (viewer, attribution, recovery) are checked as **required**.  Informational
  audit columns (e.g. `mtime`, `n_wt_per_contrast` wide columns) are not
  required by validators.
- Reason: Broad required-column checks would produce spurious failures as the
  per-cohort column sets differ (e.g. Song OLS has wide `stoich_lfc_<contrast>`
  columns for each of 9 contrasts; 5xFAD OLS adds `n_obs_raw`; enforcing the
  full set here would require enumerating all contrast names, which is pipeline
  logic not schema logic).
- Alternatives considered: Enumerate all expected columns including contrast
  columns → too brittle; breaks when a new contrast is added.
- Analysis behavior changed: no
- Output schema changed: no
- Backward compatibility impact: none
- Validation/parity evidence: all four cohorts PASS on required columns
- Reviewer: pending human gate

---

## Decision: Accepted missing metadata fields

- Date: 2026-06-17
- Phase: 1
- Agent: IMPLEMENTER
- Files affected: `alz/core/cohort_manifest.py`
- Decision: No protected sample manifest exists for Mukesh or T-cell cohorts
  (their per-donor IDs are embedded in the matrix column names).  The 5xFAD
  `sample_manifest.csv` is the only protected sample manifest.  Song's input-side
  snRNA manifest is not a protected output.  This is recorded in
  `CohortManifest.sample_manifest_path = None` for Mukesh and T-cell.
- Reason: Mukesh and T-cell do not produce a canonical sample_manifest.csv
  output; sample metadata is inferred from column names by the viewer.  No
  validator can cross-reference sample IDs for these cohorts without loading
  all matrix files.
- Alternatives considered: Derive expected sample IDs by reading the NES/FDR
  matrix column names and cross-checking them against the MEA long table's
  `contrast` values → deferred to Phase 2 when shared schemas are enforced.
- Analysis behavior changed: no
- Output schema changed: no
- Backward compatibility impact: none
- Validation/parity evidence: column-presence checks PASS for matrix prefix columns
- Reviewer: pending human gate

---

## Decision: Absent tracks represented as absent_by_design SKIP

- Date: 2026-06-17
- Phase: 1
- Agent: IMPLEMENTER
- Files affected: `alz/core/validation.py::check_file_exists()`,
  `alz/core/validate_cohort.py::validate_tcells()`
- Decision: Files that are intentionally absent (donor2 `mea_timecourse.csv`)
  are recorded in `CohortManifest.absent_by_design` and emit **SKIP** (not FAIL)
  from the validator.  If such a file were to appear unexpectedly, the check
  would emit **DEVIATION**.
- Reason: `absent_by_design` state is documented in `baseline_inventory.py`
  and the Phase-0 notes.  Flagging it as FAIL would produce a permanent red
  that cannot be fixed without violating the partial-by-design design.
- Alternatives considered: Omit the check entirely → chosen not to, so that
  if the file accidentally appears in the future it is flagged.
- Analysis behavior changed: no
- Output schema changed: no
- Backward compatibility impact: none
- Validation/parity evidence: tcells SKIP count = 1 (donor2 mea_timecourse.csv)
- Reviewer: pending human gate

---

## Decision: Duplicate-key treatment for MEA long tables

- Date: 2026-06-17
- Phase: 1
- Agent: IMPLEMENTER
- Files affected: `alz/core/validate_cohort.py` (all cohort validators)
- Decision: MEA long tables are **not** checked for duplicate kinase keys.
  Instead, `skip_dup_check_reason` is passed documenting that one row per
  (kinase × contrast × residue_type × track) is the intended long-format design.
  Composite key `(kinase, contrast, residue_type, track)` is the true key; this
  is validated only via `baseline_inventory` key_columns annotation (not by the
  Phase-1 validator which uses per-row streaming).
- Reason: Streaming dup-check for the composite key on large MEA long tables
  (> 10 MB) would be slow and the composite correctness is better verified by
  the Phase-0 baseline row-count + key-column annotations.
- Alternatives considered: Stream-check the composite key → would be safe but
  slow; deferred to Phase 2/3 when the shared runner can gate on it.
- Analysis behavior changed: no
- Output schema changed: no
- Backward compatibility impact: none
- Validation/parity evidence: SKIP with documented reason logged in JSON reports
- Reviewer: pending human gate

---

## Decision: Wide matrices remain canonical for Phase 1

- Date: 2026-06-17
- Phase: 1
- Agent: IMPLEMENTER
- Files affected: `alz/core/phospho_schema.py`, `alz/core/cohort_manifest.py`
- Decision: NES/FDR matrices (Mukesh, T-cell) and OLS tables (Song, 5xFAD) remain
  wide-format in Phase 1.  The schema descriptors in `phospho_schema.py` describe
  the wide format as-is.  No melting or pivot is attempted.
- Reason: Phase 1 is read-only.  Reshaping to long format is a producer change
  (Phase 2+), not a validator task.  The viewer consumes the wide matrices
  directly; changing the format would require viewer adaptation outside Phase 1.
- Alternatives considered: Document a normative long-format target and emit
  DEVIATION for wide matrices → premature; the wide format is current canonical.
- Analysis behavior changed: no
- Output schema changed: no
- Backward compatibility impact: none
- Validation/parity evidence: NES/FDR matrix prefix-column checks PASS
- Reviewer: pending human gate

---

## Decision: Report format — JSON + Markdown, generated_at excluded from comparison content

- Date: 2026-06-17
- Phase: 1
- Agent: IMPLEMENTER
- Files affected: `alz/core/validation.py`
- Decision: Reports are emitted as `<cohort>_validation.json` (structured,
  machine-diffable) and `<cohort>_validation.md` (human-readable table).
  `generated_at` is a top-level field only; findings are sorted by
  `(artifact_path, check_name)` for determinism.  The orchestrator strips
  `generated_at` before diffing across runs.
- Reason: `generated_at` changes every run and must not be inside the
  deterministic comparison content.  Sorted findings ensure identical output
  regardless of Python dict insertion order.
- Alternatives considered: HTML report → over-engineered for this phase.
  Plain text → less diffable.
- Analysis behavior changed: no
- Output schema changed: no
- Backward compatibility impact: none
- Validation/parity evidence: confirmed two successive runs produce identical
  findings arrays (modulo generated_at)
- Reviewer: pending human gate

---

## Decision: `Subs fraction` excluded from numeric coercibility checks

- Date: 2026-06-17
- Phase: 1
- Agent: IMPLEMENTER
- Files affected: `alz/core/phospho_schema.py`, `alz/core/validate_cohort.py`
- Decision: The `Subs fraction` column in MEA long tables is GSEApy's fractional
  string (e.g. "136/715") and is excluded from numeric coercibility checks.  It
  remains in `required_columns` (presence check only).
- Reason: GSEApy prerank outputs this column as a string "N_leading/N_total".
  It is informational; no downstream code coerces it to float.  See
  `phase_1_legacy_deviations.md` DEV-01.
- Alternatives considered: Parse as fraction and validate both parts are integers
  → over-specified; the string format is GSEApy's internal concern.
- Analysis behavior changed: no
- Output schema changed: no
- Backward compatibility impact: none
- Validation/parity evidence: All MEA numeric checks PASS after exclusion
- Reviewer: pending human gate

---

## Decision: 5xFAD track filename convention validated separately

- Date: 2026-06-17
- Phase: 1
- Agent: IMPLEMENTER
- Files affected: `alz/core/validate_cohort.py::validate_fivexfad()`,
  `_check_fivexfad_track_fname_convention()`
- Decision: 5xFAD files use a `<region>_<track>_<artifact>` prefix scheme (e.g.
  `cortex_py_mea_raw_phospho.csv`) rather than the Song/Mukesh/T-cell suffix
  scheme (e.g. `mea_raw_phospho_pY.csv`).  A dedicated validator function
  `_check_fivexfad_track_fname_convention()` is used; the generic
  `_check_track_suffix_convention()` is NOT called on 5xFAD files.
- Reason: The naming conventions are per-cohort by design.  Both encode the
  same track identity; both are correct.  See `phase_1_legacy_deviations.md`
  DEV-07.
- Alternatives considered: Unify all cohorts on one naming convention in Phase 4
  (directory migration) → deferred; Phase 1 validates as-is.
- Analysis behavior changed: no
- Output schema changed: no
- Backward compatibility impact: none
- Validation/parity evidence: fivexfad track_filename_convention all PASS
- Reviewer: pending human gate

---

## Decision: Parquet dup-key checks deferred (memory-safety)

- Date: 2026-06-17
- Phase: 1
- Agent: IMPLEMENTER
- Files affected: `alz/core/validation.py::_validate_parquet_schema()`
- Decision: Duplicate-key checks for parquet files emit SKIP.  No full parquet
  table is loaded during validation.
- Reason: Memory-safety rule (shared box).  Parquet dup-check requires scanning
  all rows; no streaming set-based check is available without full load.
  Row counts and key column names are available from Phase-0 baseline metadata.
- Alternatives considered: DuckDB pushed-down COUNT(DISTINCT key) query →
  feasible but adds a DuckDB dependency not present in the pixi env at Phase 1.
  Deferred to Phase 2+ if needed.
- Analysis behavior changed: no
- Output schema changed: no
- Backward compatibility impact: none
- Validation/parity evidence: SKIP emitted with documented reason
- Reviewer: pending human gate
