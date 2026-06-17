# 5xFAD Viewer Performance Optimization Plan

## Summary

Implement a targeted 5xFAD viewer optimization that reduces initial payload size and removes expensive client-side scans. This work uses the 5xFAD packaging path only and does not require the full unified viewer rebuild path for validation.

## Key Changes

- Slim the initial 5xFAD payload by replacing the embedded full `celltype_mea_index` with:
  - `celltype_agreement_index`: compact per kinase/tissue/track/age categorical agreement rows for the main table.
  - `celltype_mea_shards`: lazy per-kinase JSON sidecars for decomposition detail views.
- Remove `Leading substrates` from the packaged cell-type MEA rows. The field is large and is not required by the 5xFAD main table or decomposition plot.
- Keep compact rows limited to categorical calls plus raw evidence columns with units or direct meaning: NES, FDR, substrate counts, cell-type counts, and top cell type evidence where needed.
- Replace `_f5CelltypeMeaRowsForAge()` full-map scans with direct lookup by `kinase|tissue|track|age`.
- Precompute main-table agreement state once during `_f5EnsureIndexes()` from `celltype_agreement_index`.
- Populate 5xFAD filter controls once after indexing, not on every table render.
- Lazy-load per-kinase decomposition rows only when the MEA Score or Attribution detail needs them.
- Preserve the safe default that `fivexfad_celltype_ols` is not rebuilt unless `FIVEXFAD_REBUILD_CELLTYPE_OLS=1`.
- Add a lightweight 5xFAD-only packaging command that refreshes the 5xFAD payload section and static HTML bundle without rebuilding Song/Mukesh/Incytr sections or shards.
- Keep tests on temporary detail, MEA, and cell-type OLS shard directories.

## Test Plan

- Unit/static tests:
  - `celltype_mea_index` no longer appears in the 5xFAD payload.
  - `Leading substrates` is not exposed in compact agreement rows or lazy MEA shards.
  - The 5xFAD payload exposes `celltype_agreement_index` and `celltype_mea_shards`.
  - The 5xFAD JS no longer scans all cell-type MEA rows during agreement rendering.
  - 5xFAD fixture tests use temporary detail and cell-type shard directories only.
- Runtime checks:
  - `node --check alz/viewer/template/js/tabs/kinase_fivexfad.js`
  - `python -m py_compile alz/build_unified_viewer.py alz/ingest/fivexfad_celltype_mea.py`
  - `python -m unittest alz.ingest.test_fivexfad`
- Performance acceptance:
  - Initial `supporting_5xfad.celltype_mea_index` contribution is removed.
  - Main-table agreement rendering is O(visible rows x ages), with cached direct lookups.
  - Full per-cell decomposition rows are fetched only for selected detail views.
  - No full unified viewer rebuild is required to validate the 5xFAD optimization.

## Assumptions

- The main table needs categorical agreement and raw evidence counts, not full per-cell substrate lists.
- Full per-cell decomposition rows can be lazy-loaded without changing the visible table contract.
- Per-cell substrate-site OLS sidecars are optional for initial page performance and remain explicitly regenerated only.
