# Minimum incorporation of decomposition results into the unified viewer

## Goal

Surface the deconvolution-branch per-(kinase × cell type × contrast) NES inside the existing Attribution sub-tab of the Kinase tab — as a peer column to **Song LFC** and **SEA-AD LFC** in the verdict table.

## Scope

In:
- Two new columns in the verdict table (`_renderAttributionVerdict`, `ATTR_VERDICT_COLS`): **Decomp NES** and **Decomp FDR**.
- One new payload array (`decomposition_index`) joined by `(kinase_id, contrast_id, cell_type)`.

Out:
- Footer banner / cohort annotation.
- A separate decomposition tab.
- Changes to `combined_score` or `combined_confidence`.
- Per-row `direction_match` column.
- Cohort metrics (`frac_match`, `cohort_fdr`, `cohort_concordant`) anywhere in the viewer — they live on `(cell type, contrast)`, not per kinase, and would be misread as kinase-level if surfaced in the per-kinase grid.

## Track handling

There is no ambiguity. The branch's MEA driver passes `kin_type="ser_thr"` for the `st` track and `kin_type="tyrosine"` for the `py` track, so the kinase library only scores Ser/Thr kinases on `st` and only Tyr kinases on `py`. Verified against `kinase_enrichment_wmb.csv`: 311 kinases on `st` only, 78 on `py` only, zero overlap. `(kinase, contrast, cell_type)` uniquely identifies a row in the source file. No mapping needed.

## Concrete changes (one file)

### 1. `code/build_unified_viewer.py` — payload assembly

- Register `outputs/reports/deconvolution/per_animal/kinase_enrichment_wmb.csv` alongside the existing CSV inputs.
- Subset to columns `(kinase, wmb_class, contrast, NES, FDR)`. Drop `track` after subsetting; it carries no additional information for the viewer.
- Filter to kinases present in `edge_metadata["kinases"]` (drops anything not in the live universe).
- Build:
  ```python
  decomposition_index = {
      "kinase_id":   uint16[],
      "contrast_id": uint8[],
      "cell_type":   str[],
      "decomp_nes":  float[],
      "decomp_fdr":  float[],
  }
  ```
- Add to `payload` next to `attribution_index`.

### 2. `code/build_unified_viewer.py` — JS verdict renderer

- Build a JS lookup `_decompByKey: Map<"kid|cid|ct", {nes, fdr}>` once at startup, alongside the existing kinase indexes.
- Extend `ATTR_VERDICT_COLS` (insert between `song_lfc` and `combined_score`):
  ```js
  {key:"decomp_nes", label:"Decomp NES", type:"num",
   title:"Decomposition NES: kinase enrichment from CTM-native proportional decomposition (bulk phospho weighted by snRNA share for the kinase's substrate set). Same join key as Song LFC. Hypothesis-strength signal — see Methods."},
  {key:"decomp_fdr", label:"Decomp FDR", type:"num",
   title:"Decomposition FDR for this (kinase, contrast, cell type) row. < 0.25 is the standard MEA gate."},
  ```
- In `_renderAttributionVerdict`, look up `(kinase_id, contrast_id, cell_type)` and render `decomp_nes` with the same `_attrLfcColor` background used for `song_lfc`. Render `decomp_fdr` as plain number, bold when `< 0.25`. "—" when missing.
- Sort comparator already handles `type:"num"` — no changes needed.

## Cell-type coverage

The branch ran 16 of 34 WMB classes. Missing classes show "—" in the new columns, matching the existing convention for missing Song / SEA-AD rows.

## Verification

1. Build viewer with `python code/build_unified_viewer.py`. Confirm `index.html` opens and the Kinase tab loads.
2. Pick CDK5, contrast `App_6mo`. Confirm two new columns appear after Song LFC, populated for cell types where the branch produced output.
3. Pick FYN (Tyr kinase). Confirm `decomp_nes` populates from the source `py` rows (verify by spot-checking `kinase_enrichment_wmb.csv` filtered to `kinase=FYN`).
4. Pick a kinase with no decomposition output for the selected contrast. Confirm "—" renders without errors.
5. Sort by `decomp_nes` and `decomp_fdr` columns. Confirm ascending/descending toggle works.
6. Switch contrast in the dropdown. Confirm decomp columns update.
7. Spot-check three rows: confirm `decomp_nes` magnitude/sign matches `kinase_enrichment_wmb.csv` for the same (kinase, cell type, contrast).

## Files modified

| Path | Change |
|---|---|
| `code/build_unified_viewer.py` | Add CSV registration, payload build for `decomposition_index`, two `ATTR_VERDICT_COLS` entries, JS lookup + render |

No new files, no Python module restructuring, no docs.

## Out of scope (deferred)

- Folding `decomp_nes` into `combined_score` or `combined_confidence`.
- Cohort-level metrics anywhere in the viewer.
- Per-row `direction_match` column.
- Standalone "Decomposition" tab.
- Decomposition coverage expansion beyond 16 WMB classes (separate task on the branch's per-animal gating).

## Estimated effort

A few hours: payload build is ~30 lines of Python, JS column wiring is two `ATTR_VERDICT_COLS` entries plus a lookup map and two cell renderers.
