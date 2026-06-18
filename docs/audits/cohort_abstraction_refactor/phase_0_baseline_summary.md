# Phase 0 Baseline Summary

Date: 2026-06-17
Agent: implementer (Phase 0, cohort abstraction refactor)
Git base: eb3387b7bdd0b87de0c7427b606feaafd6193876

## Purpose

This document records the state of canonical outputs before the cohort
abstraction refactor begins. It is generated from
`outputs/reports/refactor_audit/phase_0/` inventory files and must not be
modified after the Phase-0 gate is approved.

---

## Per-cohort file counts

| Cohort | Protected files | Present | Absent | Absent-by-design | Total MB |
|--------|----------------|---------|--------|-----------------|----------|
| song | 41 | 41 | 0 | 0 | 2407.2 |
| mukesh | 39 | 39 | 0 | 0 | 288.3 |
| tcells | 42 | 41 | 1 | 1 | 293.2 |
| fivexfad | 33 | 33 | 0 | 0 | 2404.0 |
| incytr | 46 | 46 | 0 | 0 | 407.9 |
| viewer | 14 | 14 | 0 | 0 | 152.9 |
| **TOTAL** | **215** | **214** | **1** | **1** | **5953.6** |

---

## Output roots

### song

- `outputs/reports/kinase_attribution/` — MEA long, OLS/effect, attribution,
  normalized matrices
- `outputs/reports/decomposition/levy_t5/` — per-cluster MEA parquets,
  OLS parquets, proportions, verification
- `outputs/reports/attribution_recovery/` — celltype evidence, kinase
  activity matrix, kinase hypothesis table

### mukesh

- `outputs/reports/kinase_attribution_human/` — normalized matrices,
  celltype specificity
- `outputs/reports/kinase_attribution_human/perdonor/` — NES/FDR matrices,
  MEA long per donor, audit tables, recurrence tables

### tcells

- `outputs/reports/kinase_attribution_tcells/donor1/` — normalized matrices,
  concordance, specificity, attribution
- `outputs/reports/kinase_attribution_tcells/donor1/mea/` — MEA long per
  timepoint, NES/FDR matrices, audit tables, recurrence, manifest
- `outputs/reports/kinase_attribution_tcells/donor2/` — pY-only normalized
  tables; mea/mea_manifest.json only (partial by design)

### fivexfad

- `outputs/reports/kinase_attribution_5xfad/` — per-region/mod MEA long,
  OLS/effect tables, audit tables, snRNA attribution
- `outputs/reports/kinase_attribution_5xfad/celltype_mea/` — celltype MEA
  parquet, OLS parquet, audit tables

### incytr

- `outputs/reports/incytr_pair_mode/wide/` — 9 AD contrast parquets
- `outputs/reports/incytr_pair_mode/receiver_cache/receiver=*/data_0.parquet`
  — 31 receiver cache shards
- `outputs/reports/incytr_pair_mode_tcells/donor1/wide/` — 3 T-cell parquets
- `outputs/reports/incytr_pair_mode_tcells/donor2/wide/` — 4 T-cell parquets

### viewer

- `outputs/reports/unified_viewer/` — index.html, payload JSON (104 MB),
  payload gz (10 MB), 7 edge-slice index.json files (771 total shards in
  fivexfad_attribution + fivexfad_celltype_mea families)
- `outputs/reports/tcell_viewer/` — index.html, payload JSON (18 MB),
  payload gz (2.9 MB), 1 edge-slice index.json

---

## Absent-by-design files

### `outputs/reports/kinase_attribution_tcells/donor2/mea/mea_timecourse.csv`

**Absent by design.** donor2 has no IMAC data (pY only): Cortex-IMAC &
Hippo-Total exist only as proprietary `.sne` (unparseable on-box). The file
is listed in the protected-file manifest so future runs are alerted if it
unexpectedly appears or disappears.

---

## Parity policy (ratified 2026-06-17)

| Field class | Rule |
|-------------|------|
| Row count | exact |
| Key set | exact set identity |
| Categorical fields | exact string match |
| Numeric fields | `numpy.isclose(rtol=1e-6, atol=1e-9)` |
| NaN positions | exact |
| Binary / parquet / large JSON | sha256 first; on mismatch fall to streamed structural diff |

These tolerance constants are the only knobs. Wider tolerance on a specific
field requires a logged drift exception naming the field and the
non-deterministic source.

---

## Notes

- Song and 5xFAD produce OLS/effect-size tables (`site_level_ols.csv`,
  `*_site_level_ols.csv`). They do NOT produce NES/FDR matrices. This is by
  design — their statistical model is OLS, not GSEA enrichment.
- Mukesh (human NBB) and T-cell produce NES/FDR matrices because those
  cohorts run GSEA-style MEA per donor or per timepoint.
- The large viewer payloads (104 MB JSON and 18 MB JSON) are baselined by
  sha256 + size + mtime. They are never JSON-parsed during inventory.
- Parquet files use `pyarrow.parquet.ParquetFile.metadata` for row counts;
  data is never loaded.
- CSV files >50 MB: row count via streamed line count only; column names
  skipped. All song decomposition and 5xFAD normalized matrices fall in this
  category.

---

## Inventory artifact locations

```
outputs/reports/refactor_audit/phase_0/output_roots.json
outputs/reports/refactor_audit/phase_0/protected_files.json
outputs/reports/refactor_audit/phase_0/song_inventory.{json,csv}
outputs/reports/refactor_audit/phase_0/mukesh_inventory.{json,csv}
outputs/reports/refactor_audit/phase_0/tcells_inventory.{json,csv}
outputs/reports/refactor_audit/phase_0/fivexfad_inventory.{json,csv}
outputs/reports/refactor_audit/phase_0/incytr_inventory.{json,csv}
outputs/reports/refactor_audit/phase_0/viewer_inventory.{json,csv}
```

Inventory generator: `alz/core/baseline_inventory.py`
CLI: `python -m alz.core.baseline_inventory --all`
