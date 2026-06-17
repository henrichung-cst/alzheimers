# Viewer Payload Contract Performance Fix Plan

## Summary

The current unified viewer performance issue is not one isolated hot path. The live payload is still carrying stale 5xFAD fields, and the three cohort surfaces use different payload contracts:

- Song uses mostly columnar compact indexes plus lazy sidecars.
- Mukesh uses a mixed model: some heavy strings are sharded, but site matrices remain embedded.
- 5xFAD still uses row-oriented JSON for main rows and attribution, with only some detail data sharded.

The fix should converge the viewer onto one contract: compact initial indexes for table rendering, lazy per-entity sidecars for attribution/detail evidence, and no large substrate/site strings in the first-load payload.

## Current Measured State

From `outputs/reports/unified_viewer/unified_viewer.payload.json`:

- Total payload: 232 MB raw / 28 MB gzipped.
- `supporting_5xfad`: 160.7 MB.
- `supporting_5xfad.celltype_mea_index`: 95.3 MB.
- `supporting_5xfad.attribution_rows`: 56.1 MB in the stale production payload.
- `supporting_5xfad.rows`: 9.1 MB, including about 7 MB of embedded `leading_substrates`.
- `attribution_index`: 25.5 MB.
- `incytr_pathways`: 19.8 MB.
- `human`: 9.1 MB.

A temp run of the new 5xFAD MEA-shard code estimates `supporting_5xfad` at about 52 MB after removing `celltype_mea_index`, still dominated by row-oriented `attribution_rows`.

## Goals

- Bring first-load payload below the existing validation target: under 100 MB raw and under 20 MB gzipped.
- Make 5xFAD initial payload structurally comparable to Song and Mukesh.
- Keep table rendering O(visible rows x small fixed axes), not O(all attribution/decomposition rows).
- Keep analytical outputs defensible: expose categorical calls plus raw evidence fields, not synthetic scores.
- Avoid a full Song/Mukesh/Incytr rebuild while validating 5xFAD-only improvements where possible.

## Non-Goals

- Do not change upstream kinase, attribution, or MEA calculations.
- Do not introduce new numeric analysis scores.
- Do not rebuild `fivexfad_celltype_ols` unless `FIVEXFAD_REBUILD_CELLTYPE_OLS=1`.
- Do not redesign the viewer UI.

## Phase 1: Repair Live 5xFAD Output State

1. Run the lightweight 5xFAD package path after the current code changes:
   - `python alz/build_unified_viewer.py --supporting-5xfad-only`
   - or `pixi run 5xfad-viewer-package` where Pixi is available.

2. Verify production payload no longer contains:
   - `supporting_5xfad.celltype_mea_index`
   - `Leading substrates` in compact cell-type MEA rows.

3. Verify production output now contains:
   - `supporting_5xfad.celltype_agreement_index`
   - `supporting_5xfad.celltype_mea_shards`
   - `outputs/reports/unified_viewer/edge_slices/fivexfad_celltype_mea/`

4. Add a regression test that loads a built or fixture payload and fails if `supporting_5xfad.celltype_mea_index` appears.

Acceptance:

- `supporting_5xfad` drops by at least the old `celltype_mea_index` contribution.
- No full unified viewer rebuild is required for this phase.

## Phase 2: Remove Heavy 5xFAD Main-Row Strings

1. Remove `leading_substrates` from initial `supporting_5xfad.rows`.

2. Keep `leading_substrate_count` only if it is displayed or used in filtering; otherwise remove it too.

3. Ensure MEA detail tabs that need leading-edge motif strings read them from the existing per-kinase `fivexfad_detail` sidecars.

4. Add a test that asserts:
   - `supporting_5xfad.rows[*].leading_substrates` is absent.
   - detail sidecars still contain the fields needed for MEA preparation and running-enrichment views.

Acceptance:

- `supporting_5xfad.rows` drops from about 9 MB to roughly 2 MB.
- Main table behavior is unchanged.

## Phase 3: Replace 5xFAD Row-Oriented Attribution

1. Split current `supporting_5xfad.attribution_rows` into:
   - `celltype_attribution_summary_index`: compact per kinase/tissue/age summary rows for main-table filtering, badges, and counts.
   - `celltype_attribution_shards`: lazy per-kinase sidecars for the full attribution table and drawer.

2. Keep summary fields limited to:
   - `kinase`
   - `tissue`
   - `age_months`
   - high/moderate cell-type count
   - best categorical confidence tier
   - top cell type
   - raw native specificity, WMB specificity, SEA-AD LFC, and FDR values only where displayed.

3. Convert the JavaScript path:
   - Main table reads only `celltype_attribution_summary_index`.
   - Attribution tab calls `fetch()` for the selected kinase shard.
   - Cache loaded shards by kinase.

4. Keep raw evidence fields in the shard; do not emit derived continuous scores.

5. Add tests:
   - Initial 5xFAD payload has no `attribution_rows`.
   - Initial 5xFAD payload has `celltype_attribution_summary_index` and `celltype_attribution_shards`.
   - Attribution detail renders from a lazy shard.
   - Main table filtering does not require loading attribution shards.

Acceptance:

- 5xFAD initial attribution contribution drops from tens of MB to low single-digit MB.
- 5xFAD initial block target: under 10 MB.

## Phase 4: Normalize Cross-Dataset Contracts

Define a shared conceptual contract for all cohort kinase surfaces:

- `activity_index`: compact kinase x contrast/donor rows for table glyphs.
- `agreement_index`: compact categorical agreement rows.
- `attribution_summary_index`: compact per-kinase/per-scope attribution summaries.
- `attribution_shards`: lazy per-kinase full attribution rows.
- `decomposition_shards`: lazy per-kinase decomposition rows.
- `site_detail_shards`: lazy substrate/site/measurement trace rows.

Map existing datasets:

- Song:
  - Keep `kinases`, `agreement_index`, and `decomposition_index` for now.
  - Consider sharding the full `attribution_index` later; it is 25.5 MB.
- Mukesh:
  - Keep current per-donor substrate sharding.
  - Add caches for attribution summaries.
  - Later consider sharding `stoich_by_site` and `raw_phospho_by_site`.
- 5xFAD:
  - Move to the shared contract first because it is the largest active problem.

Acceptance:

- Documented payload keys are cohort-neutral.
- New 5xFAD keys follow the same naming and lazy-loading rules as Song/Mukesh equivalents.

## Phase 5: Client Hot-Path Cleanup

1. 5xFAD:
   - Precompute per-group display summaries during `_f5EnsureIndexes()`.
   - Avoid repeated `_f5ScopedAttrRows()` calls per table row.
   - Cache `_f5CellTypesCell`, native badge, WMB badge, and confidence badge inputs per filter state.

2. Mukesh:
   - Cache `_khAttributionSummary(r)` per `(kinase, filters affecting attribution)`.
   - Avoid repeated merge/sort work in `_khRowsForAttributionSummary()` during table filtering and rendering.

3. Song:
   - Keep existing indexes, but avoid materializing full attribution objects where index offsets are enough.

Acceptance:

- Table re-render work scales with visible rows and small cached summaries.
- Changing FDR/search/filter inputs does not recompute unrelated attribution structures.

## Phase 6: Defer Incytr Initial Payload

The `incytr_pathways` initial block is about 19.8 MB. It is independent of 5xFAD but contributes to every first load.

1. Move the Incytr global index metadata behind the Incytr tab.

2. Keep only a small manifest in the initial payload:
   - availability
   - URL to global index sidecar
   - sender/receiver vocabulary if needed for disabled state.

3. Load the global index only when an Incytr tab is opened.

Acceptance:

- Users opening kinase, human, or 5xFAD tabs do not pay the Incytr parse cost.
- Incytr behavior remains unchanged after first tab activation.

## Verification Commands

Static:

```bash
node --check alz/viewer/template/js/tabs/kinase_fivexfad.js
node --check alz/viewer/template/js/tabs/kinase_human.js
node --check alz/viewer/template/js/tabs/kinase_explorer.js
python -m py_compile alz/build_unified_viewer.py alz/ingest/fivexfad_celltype_mea.py
```

Focused tests:

```bash
python -m unittest alz.ingest.test_fivexfad
```

Payload audit:

```bash
python - <<'PY'
import json, os
p = "outputs/reports/unified_viewer/unified_viewer.payload.json"
with open(p) as f:
    data = json.load(f)
print("payload raw MB", os.path.getsize(p) / 1024 / 1024)
for k, v in sorted(data.items(), key=lambda kv: len(json.dumps(kv[1], separators=(",", ":"))), reverse=True):
    print(f"{len(json.dumps(v, separators=(',', ':'))) / 1024 / 1024:8.2f} MB  {k}")
f5 = data.get("supporting_5xfad") or {}
print("5xFAD keys", sorted(f5))
PY
```

Performance checks:

- Initial payload raw size under 100 MB.
- Initial payload gzip size under 20 MB.
- `supporting_5xfad` under 10 MB.
- No `celltype_mea_index` in payload.
- No `attribution_rows` in initial 5xFAD payload after Phase 3.
- 5xFAD detail attribution fetch occurs only after selecting the Attribution tab.

## Risks

- Lazy attribution shards change load timing; detail panels need clear loading and empty states.
- Existing bookmarks or localStorage may point at selected rows before shards are loaded.
- The current production `fivexfad_celltype_ols` directory has previously been observed as potentially incomplete; continue treating it as optional unless explicitly regenerated.
- Full viewer validation may still fail size caps until Incytr and Song global indexes are also deferred or compressed further.

## Suggested Implementation Order

1. Phase 1 and Phase 2 in one small PR.
2. Phase 3 as the main 5xFAD payload PR.
3. Phase 5 hot-path caches once payload size is under control.
4. Phase 6 Incytr deferral as a separate cross-viewer performance PR.
5. Phase 4 documentation should be updated alongside each payload-contract change.
