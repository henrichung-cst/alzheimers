# F1/F2 Audit — signed-sort + CSV export sweeps

Read-only recon (2026-06-25). F1 (signed-sort) and F2 (CSV export) are cross-cutting sweeps over every data table in BOTH viewers (unified `alz/viewer/template/js/`, tcell `alz/tcell_viewer/template/js/`, shared `alz/viewer_shared/template/js/`). Contracts: `_contracts.md §F1, §F2`. Applied as **one Wave-4 sweep AFTER table-adding themes land.**

**Shared-helper home (both F1 + F2):** `alz/viewer_shared/template/js/06_export_csv.js` — already loaded by both viewers, already holds `csvEscape`/`csvSerialize`/`csvDownload`. No shared sort helper exists today (every tab rolls its own). Add `numCmp(av,bv,dir)` (signed, null-last regardless of dir) + `cohortHeader(...)` here.

## F1 — sort comparators
| Table | Comparator | Status |
|---|---|---|
| crosstable `_kxSortRows` | `kinase_crosstable.js:1297-1299` `Math.abs(av)/abs(bv)` | **OFFENDER** (single-value cols: m_med/h_med/f5_med/wmb/h_spec/m_spec) — nulls already last |
| KE `_makeKeCompare` | `kinase_explorer.js:469`, `_kineMaxAbsNesScoped` (Math.abs:783) | **OFFENDER (profile)** — nes_profile/peak_NES by max\|NES\|; null `-Infinity` → **asc nulls-first bug** |
| 5xFAD `_f5FilterSort` | `kinase_fivexfad.js:626`, key `peakAbsNes` | **OFFENDER (profile)** — pre-stored abs; nulls last OK |
| human `_khSort` | `kinase_human.js:275-279,303-320` Math.abs | **OFFENDER** — nes_profile (profile) + median_nes_sig_only (single); `-Infinity` asc bug |
| tcell KE `_makeKeCompare` | `tcell_viewer/.../kinase_explorer.js:315`, `_kineMaxAbsNesScoped:627` | **OFFENDER (profile)** — same as unified KE |
| audit `_attrVerdictCmp` (both viewers) | `kinase_audit.js:604` / tcell `:621` | already signed, null-last — adopt shared helper |
| 5xFAD `_f5AttrCmp` | `kinase_fivexfad.js:297` | already signed, null-last |
| incytr `_ipFilterRows` | `viewer_shared/.../incytr_pathways.js:1092` | already signed, null-last |
| incytr global index `_selectTopK` | `incytr_global_index.js:290` | already signed, NaN-last |
| AuditTable.filteredRows | `01_state.js:494` | signed; non-finite→string fallback (debug table) |

**Profile columns** (KE/5xFAD/human nes_profile, peak_NES) hold a per-contrast NES *vector*; current key = `max|NES|`. "Signed sort of a vector" is undefined → **decision: key = signed NES of the peak-magnitude contrast.**

**Out of F1 scope (magnitude is correct / not a table-sort):** best-pick selection (crosstable:1016 song-LFC, :1041 5xFAD-LFC), drill-down orderings (fivexfad:1842 per-cell β, human:565 per-site δ).

**Asc/desc toggle** confirmed per table (`kinase_wiring.js:14-23` header-click → `sortAsc`; incytr `sortDir ±1`). It is the mechanism to reach the negative tail.

## F2 — CSV exports
Existing (all working-set via a module-level `_Visible` stamped at render; raw precision; UI-only excluded via curated key arrays):
| Export | File:line | Filename today | Cohort-name headers |
|---|---|---|---|
| `exportCrosstableCsv` | `kinase_crosstable.js:1616` | `crosstable_<date>.csv` | `Mouse_med_NES`,`Human_med_NES`,`5xFAD_med_NES`,`Song_fold` |
| `exportKinaseCsv` | `kinase_explorer.js:879` | `kinase_<date>.csv` | `song_topShare`,`song_topCell`; `_exportPeakAbsNes` (=abs!) |
| `exportFiveXFADCsv` | `kinase_fivexfad.js:757` | `fivexfad_kinase_<date>.csv` | `5xFAD_snrna`; `peakAbsNes` |
| `exportKinaseHumanCsv` | `kinase_human.js:1502` | `kinase_human_<date>.csv` | generic |
| `_ipExportCurrentView` | `incytr_pathways.js:1590` | `incytr_pathways_<ctx>_<mode>_<date>.csv` | data-key headers; `<ctx>`=activeContext (internal id) |

**Tables LACKING export (data grids):** unified attribution-verdict (KA), tcell KE (no `exportKinaseCsv`, no `_keVisible` stamp in its render), tcell KA verdict. **Pure-viz (no export needed):** temporal, wiring, incytr heatmap.

**C2/C1 deps:** no JS `COHORT_DISPLAY` map exists yet (C2 produces it). Crosstable `Mouse_med_NES` → C1 splits Song into 3 genotype columns → C1 must extend this export's key array (F2 is Wave 4, after C1). `_ipExportCurrentView` filename `<ctx>` = `activeContext()` returns internal ids (`song_ad`/`fivexfad_cortex` unified; `donor1`/`donor2` tcell) → unified maps through `COHORT_DISPLAY`, tcell passes through.

**T-cell:** outside C2; context ids `donor1`/`donor2` are its only naming tokens; columns are `tcell_*` (no song/5xFAD). F2 filename token for tcell = the donor context id.
