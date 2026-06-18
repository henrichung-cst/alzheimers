# Phase 0 Decision Log

Refactor: cohort abstraction refactor
Phase: 0 — Inventory & Baseline Lock
Date: 2026-06-17
Agent: implementer (Phase 0)
Protocol: docs/plans/cohort_abstraction_refactor/agent_protocol.md

---

## Decision: canonical output roots chosen

- Date: 2026-06-17
- Phase: 0
- Agent: implementer
- Files affected: `outputs/reports/refactor_audit/phase_0/output_roots.json`
- Decision: Canonical output roots per cohort are:
  - song → `outputs/reports/kinase_attribution/`, `outputs/reports/decomposition/levy_t5/`, `outputs/reports/attribution_recovery/`
  - mukesh → `outputs/reports/kinase_attribution_human/` (incl. `perdonor/` subtree)
  - tcells → `outputs/reports/kinase_attribution_tcells/` (incl. `donor1/`, `donor2/`)
  - fivexfad → `outputs/reports/kinase_attribution_5xfad/` (incl. `celltype_mea/`)
  - incytr → `outputs/reports/incytr_pair_mode/` and `outputs/reports/incytr_pair_mode_tcells/`
  - viewer → `outputs/reports/unified_viewer/` and `outputs/reports/tcell_viewer/`
- Reason: These are the roots where pixi run tasks write their primary
  outputs, as confirmed by the 2026-06-17 structural maps in
  `docs/plans/cohort_abstraction_refactor/orchestration_plan.md` §2.
- Alternatives considered: `data/derived/` was considered but is an input
  staging area, not a canonical output root.
- Analysis behavior changed: no
- Output schema changed: no
- Backward compatibility impact: none (read-only inventory)
- Validation/parity evidence: `git status --short` shows only Phase-0 files
  added; no canonical outputs modified.
- Reviewer: pending human gate

---

## Decision: per-cohort protected-file categories (Song/5xFAD use OLS, not NES/FDR)

- Date: 2026-06-17
- Phase: 0
- Agent: implementer
- Files affected: `outputs/reports/refactor_audit/phase_0/protected_files.json`
- Decision: Protected-file list is templated per cohort, not uniform.
  - Song and 5xFAD: `site_level_ols.csv` / `*_site_level_ols.csv` are
    protected. No `kinase_donor_nes.csv` / `kinase_donor_fdr.csv` — these
    cohorts do not run per-donor GSEA. No `mea_timecourse.csv` — Song uses
    contrast-based MEA long; 5xFAD uses per-region/mod MEA long.
  - Mukesh (human NBB): `perdonor/kinase_donor_nes.csv` and
    `perdonor/kinase_donor_fdr.csv` (+ pY/raw variants) ARE protected.
    MEA long is `perdonor/mea_perdonor.csv` (+ pY/raw variants).
  - T-cell donor1: `donor1/mea/kinase_timepoint_nes.csv` and
    `donor1/mea/kinase_timepoint_fdr.csv` (+ pY/raw variants) ARE protected.
    MEA long is `donor1/mea/mea_timecourse.csv`.
- Reason: Song and 5xFAD use OLS (Ordinary Least Squares site-level
  regression) as their statistical model, not GSEA. NES (Normalized
  Enrichment Score) and FDR matrices are outputs of GSEA-based MEA, which
  is used by Mukesh (per-donor) and T-cell (per-timepoint) cohorts. A flat
  template would emit false "missing" alarms for Song/5xFAD. Confirmed by
  inspection of actual on-disk files and by orchestration plan §2.3.
- Alternatives considered: uniform template with all file types — rejected
  because Song and 5xFAD genuinely do not produce NES/FDR outputs.
- Analysis behavior changed: no
- Output schema changed: no
- Backward compatibility impact: none (read-only inventory)
- Validation/parity evidence: globbing confirms absence of `*nes*.csv` under
  `kinase_attribution/` and `kinase_attribution_5xfad/`.
- Reviewer: pending human gate

---

## Decision: T-cell donor2 accepted as partial/absent-by-design

- Date: 2026-06-17
- Phase: 0
- Agent: implementer
- Files affected: `outputs/reports/refactor_audit/phase_0/tcells_inventory.json`
- Decision: `donor2/mea/mea_timecourse.csv` is recorded as
  `exists=false, notes="absent_by_design: donor2 pY-only, no IMAC"`.
  The file is listed in the protected-file manifest but the inventory does
  not treat its absence as an error. donor2 pY normalized tables
  (`raw_phospho_normalized_pY.csv`, `stoichiometry_matrix_pY.csv`,
  `total_proteome_normalized.csv`) and `donor2/mea/mea_manifest.json` ARE
  present and protected.
- Reason: donor2 has no IMAC proteomics data. Cortex-IMAC and Hippo-Total
  exist only as proprietary `.sne` files (Spectronaut-native, unparseable
  on-box). This is a known and accepted input gap documented in
  `~/.claude/projects/-home-hchung-Projects-work-alzheimers/memory/MEMORY.md`
  (5xFAD cohort on hold note) and in the orchestration plan §2.4.
- Alternatives considered: omitting donor2 entirely from the protected list —
  rejected because the files that DO exist must be baselined, and recording
  the expected-absent file ensures future runs are alerted if it
  unexpectedly appears.
- Analysis behavior changed: no
- Output schema changed: no
- Backward compatibility impact: none (read-only inventory)
- Validation/parity evidence: `find` confirms the file is absent on disk.
- Reviewer: pending human gate

---

## Decision: parity tolerance policy RATIFIED

- Date: 2026-06-17
- Phase: 0
- Agent: implementer (carrying forward orchestration plan §6 ratification)
- Files affected: `outputs/reports/refactor_audit/phase_0/protected_files.json`
  (parity_policy field), `alz/core/baseline_inventory.py` (PARITY_RTOL/ATOL)
- Decision: Structural fields are exact; numeric fields use
  `numpy.isclose(rtol=1e-6, atol=1e-9)`. NaN positions are exact. Key/row
  changes are never tolerance-absorbable.
  - `PARITY_RTOL = 1e-6`
  - `PARITY_ATOL = 1e-9`
  These constants are recorded in `alz/core/baseline_inventory.py` and in
  `protected_files.json:parity_policy`. They are the only numeric tolerance
  knobs for this refactor. Wider tolerance on any specific field requires a
  logged drift exception naming the field and the non-deterministic source.
- Reason: MEA uses a fixed seed (`MEA_SEED=112123`,
  `MEA_PERMUTATION_NUM=1000`), so bit-reproducible outputs are the baseline
  expectation. The `rtol=1e-6, atol=1e-9` tolerance absorbs float-formatting
  and library round-trip noise while tripping on any real analytical drift.
  Ratified in orchestration plan §6 on 2026-06-17.
- Alternatives considered: exact-identity only — would fail on float I/O
  round-trips through CSV. Looser tolerance (rtol=1e-3) — would mask genuine
  analytical drift.
- Analysis behavior changed: no
- Output schema changed: no
- Backward compatibility impact: none (read-only policy document)
- Validation/parity evidence: n/a (policy declaration, not a code change)
- Reviewer: pending human gate — numbers are confirmable at Phase-0 gate

---

## Decision: key-column choices per table

- Date: 2026-06-17
- Phase: 0
- Agent: implementer
- Files affected: `outputs/reports/refactor_audit/phase_0/protected_files.json`,
  `alz/core/baseline_inventory.py`
- Decision: Key columns chosen per table category, based on header inspection:
  - MEA long tables (mea_raw_phospho, mea_stoichiometry, mea_timecourse,
    mea_perdonor): `[kinase, contrast, residue_type, track]`
  - NES/FDR matrices (kinase_donor_nes/fdr, kinase_timepoint_nes/fdr): `[kinase]`
    (kinase is the row index; columns are donor/timepoint IDs)
  - Recurrence tables: `[kinase]`
  - site_level_ols (Song, T-cell): `[site_id, gene_symbol]`
  - site_level_ols (5xFAD, per region/mod): `[site_id]`
  - unified_attribution / celltype_evidence_table: `[kinase, contrast, cell_type]`
  - kinase_activity_matrix: `[kinase, residue_type]`
  - kinase_hypothesis_table: `[kinase, residue_type]`
  - fivexfad_snrna_attribution: `[kinase, cell_type, tissue, age_months]`
  - mea_per_cluster (decomp): `[kinase, cluster, contrast]`
  - site_level_ols_per_cluster (decomp): `[site_id, cluster]`
  - incytr wide parquets: `[Sender, Receiver, Pathway]`
  - Normalized matrices (stoichiometry_matrix, raw_phospho_normalized, etc.):
    `[kinase]` (kinase is the row index; columns are sample IDs)
- Reason: Key columns are chosen from the first header row of each file
  type. They represent the logical primary key or the categorical
  stratification columns that must be exactly preserved across refactoring.
  Columns are inspected via `head -1` (streamed, no data load).
- Alternatives considered: using all categorical columns as keys — rejected
  as over-constraining; sample ID columns in matrix files are row index, not
  per-row keys.
- Analysis behavior changed: no
- Output schema changed: no
- Backward compatibility impact: none (read-only policy)
- Validation/parity evidence: header values verified by manual inspection
  above.
- Reviewer: pending human gate

---

## Decision: protected-surface tightening (215 → 165)

- Date: 2026-06-17
- Phase: 0
- Agent: orchestrator (post-audit, human-approved)
- Files affected: `alz/core/baseline_inventory.py` (`_build_protected_files`);
  regenerated `protected_files.json` + all `<cohort>_inventory.{json,csv}`
- Decision: Acting on `phase_0_protected_surface_audit.md` (5 read-only cohort
  auditors), the protected baseline was tightened from 215 to 165 files:
  - **DE-PROTECTED 68 entries** (kept on disk, removed from parity contract):
    Mukesh 12 (`_all` concat + `_raw` audit sidecars), T-cell 12 (`_raw` MEA/audit
    + `recurrence_pY/raw/raw_pY` + donor2 `total_proteome_normalized`), Song 9
    (per-cluster audit sidecars + `coverage_report` + `proportions_provenance` +
    `total_proteome_normalized_pY` duplicate), 5xFAD 2 (celltype `global_shift`/
    `winsorized_sites` source-files-only), Incytr/viewer 33 (30 `receiver_cache`
    shards = reshape of `wide/`; unified payload `.gz`; both tcell payload files).
  - **ADDED 18 entries** (actively read by the viewer, previously unprotected):
    5xFAD `{region}_{mod}_{contrast_qc,raw_phospho_normalized,stoichiometry_matrix,
    matched_total_protein}.csv` (16), `sample_manifest.csv`,
    `celltype_mea/fivexfad_snrna_pseudobulk_counts.csv`.
- Reason: De-protected files are derived transforms, write-only orphans, or
  secondary-track audit sidecars with no downstream reader (consumer evidence in
  the audit doc). Added files are required viewer inputs whose drift would
  otherwise pass undetected. Per-cohort counts now song 32 / mukesh 27 / tcells 30
  / fivexfad 49 / incytr 16 / viewer 11 = 165; all present (1 absent-by-design).
- Alternatives considered: (a) ratify all 215 — rejected, leaves the 5xFAD
  protection gap + dead surface; (b) delete the orphan outputs now — deferred:
  ceasing to *produce* them is a Phase-2/3 producer change (5xFAD exempt,
  unrecoverable behind the `.sne` hold).
- Analysis behavior changed: no (baseline-list only; no producer touched)
- Output schema changed: no
- Backward compatibility impact: none (read-only inventory tooling)
- Validation/parity evidence: `py_compile` clean; regenerated count = 165 with
  zero non-design-absent files; SSOT is `_build_protected_files`, JSON is generated.
- Reviewer: approved at Phase-0 gate (2026-06-17)
