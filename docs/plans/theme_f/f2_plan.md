# Theme F2 — CSV export standard

**Contract:** `_contracts.md §F2`. **Audit:** `f_audit.md`. **Wave:** 4 (cross-cutting sweep, after table-adding themes). **Prereq:** C2 (the JS `COHORT_LABELS`/`COHORT_DISPLAY` map) + C1/C3 (their export column arrays exist by Wave 4). **Collision class:** both viewers' JS table tabs + the shared export util.

## Decisions (locked, P3 grill 2026-06-25)
- Headers stay **curated per-table key arrays** (the natural "data columns only, exclude UI" list), **routed through `COHORT_DISPLAY` at the export boundary** — NOT a derive-from-render-spec refactor.
- New exports for the three data-grid tables lacking one: **unified attribution-verdict, tcell KE, tcell KA verdict.** Pure-viz tabs (temporal, wiring, incytr heatmap) get none.
- **Filename token:** single-cohort tables use `COHORT_DISPLAY[cohort]`; the cross-cohort **crosstable** uses no cohort prefix (`crosstable.csv`); context-scoped tables (incytr) use the active context's display (unified → mapped, tcell → `donor1`/`donor2` passthrough).
- Working set + raw precision are already correct (existing `_Visible` pattern) — preserve.
- `_exportPeakAbsNes` → **signed** peak NES (F1 consistency; abs is sign-loss in the CSV).

## Stage 1 — Shared header/filename helper
Add to `06_export_csv.js`:
- `cohortHeader(header, displayMap)` — replace a leading cohort token (`Mouse`/`Song`/`song`→`MouseC1`, `5xFAD`/`fivexfad`→`MouseC2`, `Human`/`Mukesh`→`HumanC1`) per the C2 map; atlas tokens (`WMB`,`SEAAD`) pass through.
- `exportFilename(cohortDisplayOrNull, table)` → `<cohort_display>__<table>.csv` lowercased, or `<table>.csv` when null (cross-cohort).
Both read the C2 `COHORT_DISPLAY` map (single source from C2).

## Stage 2 — Standardize existing exports
Apply `cohortHeader` to header arrays + `exportFilename` to filenames:
- `exportCrosstableCsv` (`kinase_crosstable.js:1616`): headers → `MouseC1_med_NES`/`HumanC1_med_NES`/`MouseC2_med_NES`/`MouseC1_fold`; filename → `crosstable.csv` (cross-cohort, no prefix). **C1 dependency:** when C1 splits Song into 3 genotype columns, C1 extends this key array (`MouseC1_App_med_NES` …) — F2 just renames whatever columns exist.
- `exportKinaseCsv` (`kinase_explorer.js:879`): `song_topShare`/`song_topCell` → `MouseC1_topShare`/`MouseC1_topCell`; `_exportPeakAbsNes` → signed; filename → `mousec1__kinase.csv`.
- `exportFiveXFADCsv` (`kinase_fivexfad.js:757`): `5xFAD_snrna` → `MouseC2_snrna`; filename → `mousec2__kinase.csv`.
- `exportKinaseHumanCsv` (`kinase_human.js:1502`): filename → `humanc1__kinase.csv` (headers already generic).
- `_ipExportCurrentView` (`incytr_pathways.js:1590`): map `<ctx>` through `COHORT_DISPLAY` for unified contexts; tcell `donor1`/`donor2` passthrough; filename → `<ctx_display>__incytr_pathways.csv`.

## Stage 3 — Add the three missing exports
Mirror the existing pattern (curated keys, `_Visible` working set, `csvSerialize`+`csvDownload`):
- **Unified attribution-verdict** (`kinase_audit.js`): export the verdict table's current filtered+sorted rows; rename `song_lfc`/`song_specificity`/`song_tau` → `MouseC1_*`; filename `mousec1__attribution.csv` (verdict is contrast/Song-scoped).
- **tcell KE** (`tcell_viewer/.../kinase_explorer.js`): add `exportKinaseCsv` + stamp `_keVisible` in its render loop (currently absent); columns are `tcell_*`; filename `<donor>__kinase.csv`.
- **tcell KA verdict** (`tcell_viewer/.../kinase_audit.js`): export verdict rows; filename `<donor>__attribution.csv`.
Wire each to an export button in the tab toolbar (match the existing tabs' button).

## Verification (browser, human — authoritative)
- Each data table has an export button; clicking it downloads the **currently filtered+sorted** rows (apply a filter + sort, export, confirm the CSV matches what's on screen — same rows, same order).
- Headers carry **display names** (`MouseC1_*`, `MouseC2_*`, `HumanC1_*`), no `song_*`/`5xFAD_*`/`Mukesh`; atlas columns (WMB/SEAAD) unchanged.
- Values are **raw full-precision** (not the rounded/colored display) — spot-check a NES against the payload; the peak-NES export carries a **sign**.
- No UI-only columns (color swatches, badge HTML, sparklines) in any CSV.
- Filenames: `mousec1__kinase.csv`, `mousec2__kinase.csv`, `humanc1__kinase.csv`, `crosstable.csv`, `<ctx_display>__incytr_pathways.csv`, `donor1__kinase.csv`.

## Out of F2 scope
The signed-sort itself (F1), pure-viz tabs, any derive-columns-from-render-spec refactor, internal AuditTable drill-down exports (already exist), the C2 map itself (consumed, not defined here).
