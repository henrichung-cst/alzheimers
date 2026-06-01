# T-cell viewer payload — reshape `build_tcell_viewer.py` to T-cell sources

**Date:** 2026-05-29
**Branch:** `perf/pairmode-memory-audit` (viewer work lives downstream of the audit branch)
**Status:** Superseded by schema-v2 viewer payload contract

**Supersession note (2026-06-01):** this plan records the earlier T-cell payload proposal and is
kept only as historical context. Do not follow its `selection.donor`, `by_donor`, or direct
`PAYLOAD.*` payload guidance. Current viewer routing is `selection.context` / `ctx=`, and shared
payload blocks are canonical under `*.by_context`; see
[`docs/foundation/viewer_payload_contract.md`](../foundation/viewer_payload_contract.md).

## Context

`alz/tcell_viewer/` and `alz/build_tcell_viewer.py` are a **literal byte-for-byte copy** of the unified viewer (`alz/viewer/` + `alz/build_unified_viewer.py`), with only the minimum renames to avoid output collision:

- output dir → `outputs/reports/tcell_viewer/`
- payload filename → `tcell_viewer.payload.json`
- inter-module imports → `tcell_viewer.paths`
- `verify_template.py` SRC → `build_tcell_viewer.py`
- pixi task → `tcell-viewer = "python alz/build_tcell_viewer.py"`

The builder still **reads mouse-cohort artifacts** (Song bulk MEA, levy_t5 decomposition, SEA-AD human cohort, WMB atlas, mouse pair-mode Incytr). Running `pixi run tcell-viewer` today produces a tcell-named HTML that renders **mouse data** — which is wrong, but the literal-copy was the requested checkpoint.

This plan is the next step: **reshape the payload builder to read T-cell data sources**, lifting the unified viewer's JS modules verbatim per `feedback_no_reimplementing_shared_viewer`. New cohort's payload conforms to the existing `PAYLOAD.*` contract; tabs that don't apply get dropped at `TAB_MANIFEST` level, not silently re-implemented thinner.

## T-cell data inventory

### Available
- **Bulk MEA (donor1 only — donor2 has no IMAC):**
  `outputs/reports/kinase_attribution_tcells/donor1/mea/`
  - `kinase_timepoint_nes.csv` / `kinase_timepoint_fdr.csv` (+ `_pY` variants, + `_raw` variants)
  - `mea_timecourse.csv` (+ `_pY`, `_raw`)
  - `mea_substrate_sets.csv` (98 MB; + `_pY` 1.3 MB, + `_raw` 107 MB)
  - `recurrence.csv` (+ `_pY`, `_raw`)
  - `winsorized_sites.csv`
  - `mea_global_shift.csv`
  - `mea_manifest.json`
- **Bulk substrates:**
  `outputs/reports/kinase_attribution_tcells/{donor1,donor2}/`
  - `raw_phospho_normalized.csv` (donor1 only, donor2 has only `_pY`)
  - `stoichiometry_matrix.csv` / `_pY.csv`
  - `total_proteome_normalized.csv` (both donors)
- **Pair-mode Incytr wide outputs (both donors):**
  `outputs/reports/incytr_pair_mode_tcells/{donor1,donor2}/wide/`
  - `d13_d2_incytr_output.parquet` (27 MB)
  - `d17_d2_incytr_output.parquet` (23 MB)
  - `d20_d2_incytr_output.parquet` (61 MB)
- **Incytr inputs (scRNA + bulk):**
  `data/derived/tcells_incytr_inputs/{donor1,donor2}/`
- **scRNA-derived cluster info:**
  `data/derived/tcells/{donor1,donor2}/` (ProjecTILs-annotated)

### Absent / N/A by design
- **No deconvolution.** T-cell MEA runs on bulk only; no `decomposition_index`, no per-cluster β shards, no decomp-OLS slices.
- **No Song concordance.** Song is a mouse bulk-proteomics dataset; T-cells have no cross-reference cohort.
- **No SEA-AD human concordance.** Mouse-disease–specific.
- **No WMB specificity / agreement_index.** WMB is a mouse brain atlas; T-cell clusters live in ProjecTILs space.
- **No App/Tau/ApTt genotype contrasts.** T-cell contrasts are **timepoint vs baseline d2** per donor: d13–d2, d17–d2, d20–d2.
- **No donor2 kinase MEA.** Donor2 has no IMAC channel — pY only, and no full MEA outputs were generated.

## Contract decisions

### Mode wiring
Add a new `tcell` mode alongside the existing `mouse` / `human` modes in `TAB_MANIFEST`. Header gets a third mode toggle. `Store.state.view.mode` defaults to `tcell` when launched from this builder (set in template defaults / boot.js gate).

### Per-donor selector
Donor1 vs donor2 is a **selection axis like genotype was for the mouse cohort**. Treat as a top-level `Store.state.selection.donor` ∈ `{donor1, donor2}`. Render a donor segmented-control next to the mode toggle in the header. Wired into `03_filters_hash.js` so it serializes into the URL hash.

### Contrast vocabulary
- Mouse cohort: `App_2mo`, `App_5mo`, `Tau_2mo`, …, `ApTt_5mo` (9 contrasts).
- T-cell cohort: `d13_d2`, `d17_d2`, `d20_d2` per donor (3 contrasts × 2 donors = 6 contrast-donor pairs).
- Payload `meta.contrasts` is the T-cell vocabulary. `meta.genotype_color_map` is dropped; replace with `meta.timepoint_color_map` keyed on `d13/d17/d20` (single-hue progression — not the mouse three-disease palette). The viewer's `__APP_COLOR__` / `__TAU_COLOR__` / `__APTT_COLOR__` sentinels get replaced with `__D13_COLOR__` / `__D17_COLOR__` / `__D20_COLOR__` in `write_html()`.

### TAB_MANIFEST changes (alz/tcell_viewer/template/js/02_ui_chrome.js)
Update each tab's `modes` array to declare T-cell support; drop tabs with no T-cell analog at the manifest level (per `feedback_no_reimplementing_shared_viewer` rule 3).

| Tab id           | T-cell decision                                       |
|------------------|-------------------------------------------------------|
| `temporalv2`     | **Keep**, donor1 only (donor2 → empty-state banner). MEA timecourse, kinase NES heat — substrate ready. |
| `crosstable`     | **Keep**, donor1 only. NES × contrast crosstable. |
| `kinasehuman`    | **Drop** (was `modes: ["human"]`; no T-cell analog).  |
| `kinase`         | **Keep**, donor1 only. Kinase detail panel (NES, FDR, leading edge, substrate motifs). |
| `incytrheatmap`  | **Keep**, both donors. Sender × receiver heat per contrast. |
| `incytrpathways` | **Keep**, both donors. Per-pair pathway table with evidence drawer. |
| `methods`        | **Keep**, replace `docs/pipeline_overview.html` with T-cell methods stub (or omit until a T-cell pipeline overview is written — drop the tab rather than ship a 404). |

The Audit drawer's transcript / omics / measurement-trace sub-rows require per-cluster pseudobulk — those exist for T-cells (`tcells_decompose.py` outputs) but plug into different paths; defer the Evidence drawer until cell-level substrates are wired.

### Payload key map

| `PAYLOAD.*` key                | T-cell source                                                                     | Builder function (new)              |
|--------------------------------|-----------------------------------------------------------------------------------|-------------------------------------|
| `kinases`                      | `kinase_attribution_tcells/donor1/mea/kinase_timepoint_{nes,fdr}.csv` (+ raw, pY) | `_build_tcell_kinases_slice()`      |
| `kinase_motifs`                | reused as-is (kinase-library lookup; cohort-agnostic)                             | unchanged                            |
| `celltypes`                    | ProjecTILs cluster labels from `data/derived/tcells/donor*/`                      | `_build_tcell_celltypes_slice()`    |
| `kinase_celltype_evidence`     | **Dropped** — no per-cluster MEA in T-cell pipeline                                | N/A                                  |
| `attribution_index`            | **Dropped** — no per-cluster attribution                                          | N/A                                  |
| `decomposition_index`          | **Dropped** — no deconvolution                                                    | N/A                                  |
| `agreement_index`              | **Dropped** — no WMB cross-reference                                              | N/A                                  |
| `subclass_breakdown`           | **Dropped** — keyed on Levy-T5 subclasses                                         | N/A                                  |
| `audit_tables`                 | T-cell ingest manifest + cluster annotation audit                                 | `build_tcell_audit_manifest()`      |
| `edge_slice_ref`               | Trim to `incytr_pathways_url`/`_index` only; drop decomp_ols, song_concordance, human_perdonor refs | inline      |
| `incytr_pathways`              | `incytr_pair_mode_tcells/donor*/wide/*.parquet` — per-donor sharded               | `_write_tcell_pair_pathways()`      |
| `meta`                         | T-cell schema: donors, timepoints, contrast list, color palette                   | inline                               |
| `human`                        | **Omitted** (key never set in T-cell payload)                                     | N/A                                  |

### Edge-slice shards
Output dir: `outputs/reports/tcell_viewer/edge_slices/incytr_pathways/`. Shard keying:
`{donor}_{contrast}_{sender}_{receiver}.parquet` — extends the mouse cohort's
`{contrast}_{sender}_{receiver}.parquet` pattern with a donor prefix. Index JSON records donors + contrasts + cluster axes per donor.

The pair-mode Incytr filter (`SigProb > 0.1 ∧ |PDS| ≥ 0.2`, per `alz/incytr_pair/filter_significant_paths.py`) applies unchanged — same Incytr scorer, same gate.

## Implementation steps

### Step 1 — Strip non-applicable code paths in `build_tcell_viewer.py`
Per anti-shim: pivots replace, they do not coexist. Remove (do not flag-gate):
- `build_human_slice()` and all `human_*` helpers, paths, and call sites.
- `_build_agreement_index()` and all `wmb_*` references.
- `_write_decomp_ols_slices()` and `EDGE_SLICES_DECOMP_OLS_DIR` plumbing.
- `_write_song_concordance_slices()` and `EDGE_SLICES_SONG_CONCORDANCE_DIR`.
- `ensure_omics_trace_*()`, `ensure_transcript_trace_*()`, `ensure_measurement_trace_*()` (require cell-level substrates wired for the mouse cohort, deferred for T-cell).
- `assert_pathway_fc_round_trips()` (mouse-specific sce4 parity gate).
- `_build_subclass_breakdown()` (Levy-T5 subclasses).
- `attribution_index` / `decomposition_index` build blocks.

Mirror strip in `alz/tcell_viewer/paths.py`: drop `EDGE_SLICES_*` constants for the dropped shard families; keep `EDGE_SLICES_INCYTR_PATHWAYS_DIR`.

### Step 2 — New T-cell payload functions
Add to `alz/build_tcell_viewer.py`:

- `_load_tcell_kinase_attribution(donor: str) -> dict` — reads `kinase_attribution_tcells/{donor}/mea/kinase_timepoint_{nes,fdr}.csv` and matched `_raw`, `_pY`, `_raw_pY` quartet. Returns `{nes, fdr, nes_raw, fdr_raw, nes_pY, fdr_pY, nes_raw_pY, fdr_raw_pY}` as long DataFrames keyed on `(kinase, contrast)`.
- `_build_tcell_kinases_slice(donor: str) -> dict` — flattens to the existing kinase slice schema (`{id, name, nes_by_contrast, fdr_by_contrast, ...}`); donor identity stamped into `meta` only, not into each kinase row.
- `_build_tcell_celltypes_slice(donor: str) -> dict` — ProjecTILs labels (CD8.TEx, CD4.TFH, CD8.TEM, etc.) from `data/derived/tcells/{donor}/`. Sanitize names per cluster-name alphanumeric rule (no `_`, no `.` → camelcase or strip).
- `_write_tcell_pair_pathways() -> dict` — per-donor loop over `{donor1, donor2} × {d13_d2, d17_d2, d20_d2}`; reuses `_write_incytr_pair_pathways()`'s sharding logic but emits the donor-prefixed keys.
- `build_tcell_audit_manifest() -> dict` — surface `ingest_manifest.json`, `cluster_annotations.csv`, `annotation_audit.json` (counts, drop fractions per donor).
- `build_tcell_payload() -> dict` — replaces `build_payload()`. Per-donor sub-slices are nested under `payload.kinases.by_donor = {donor1: ..., donor2: ...}` so the donor selector hot-swaps without re-fetching.

### Step 3 — Builder entrypoint
Replace `if __name__ == "__main__":` block: drop `load_all_data()` call; new entry calls `build_tcell_payload()` directly, then `write_payload()` + `write_html()`. Validation function (`validate()`) gets a T-cell-specific rewrite — checks raw/gzip sizes against the same 100 MB / 20 MB caps, asserts donor1 has MEA + Incytr, donor2 has Incytr only.

### Step 4 — JS-side mode + donor wiring
- `alz/tcell_viewer/template/js/01_state.js`: add `selection.donor` to default state; reducer handles `SET_DONOR` action.
- `02_ui_chrome.js`: update each tab's `modes` array per the table above; add `tcell` mode option in mode toggle render.
- `03_filters_hash.js`: serialize donor into hash (`#donor=donor1&tab=kinase&...`).
- `05_header.js`: add donor segmented-control to header; add timepoint legend (replaces genotype legend).
- `boot.js`: default `Store.state.view.mode = "tcell"` when `PAYLOAD.meta.cohort === "tcell"`.

Each affected tab module (`tabs/kinase_explorer.js`, `tabs/incytr_pathways.js`, etc.) needs **no rewrite** — they consume `PAYLOAD.kinases` / `PAYLOAD.incytr_pathways` blindly. The mode/donor changes happen at the slice-cache layer (`04_slice_cache.js`): when donor changes, the cache flushes and the per-tab `render()` re-pulls from `PAYLOAD.kinases.by_donor[currentDonor]`.

If the per-donor nesting breaks tab modules that index `PAYLOAD.kinases.<flat>` directly (likely), a small `04_slice_cache.js` shim returns `PAYLOAD.kinases.by_donor[donor]` when present, falls back to `PAYLOAD.kinases` otherwise. This is the **only** place a thin shim is acceptable — it preserves the contract; alternative is a per-donor rebuild of the kinase slice on every donor change.

### Step 5 — Template sentinels
`alz/tcell_viewer/template/index.html.j2`: replace `__APP_COLOR__` / `__TAU_COLOR__` / `__APTT_COLOR__` references with `__D13_COLOR__` / `__D17_COLOR__` / `__D20_COLOR__`. `write_html()` substitution table updated to match. Color palette: a single sequential hue (e.g. viridis `#440154 → #21918c → #fde725`) for d13 → d17 → d20.

### Step 6 — Methods tab
Three options, decide before implementing:
1. Write a T-cell-specific `docs/tcell_pipeline_overview.html` and source it.
2. Drop the Methods tab from T-cell `TAB_MANIFEST` until (1) exists.

Recommended: (2), because shipping the mouse pipeline overview as the T-cell Methods tab is misleading. Per honesty-over-polish: omit, do not stub.

## Verification

1. `pixi run tcell-viewer` runs end-to-end without `KeyError` / `FileNotFoundError` on mouse-cohort paths.
2. `outputs/reports/tcell_viewer/tcell_viewer.payload.json` raw ≤ 100 MB, gzip ≤ 20 MB (existing caps).
3. `outputs/reports/tcell_viewer/index.html` opens in a browser; donor1 / donor2 toggle swaps Incytr pathways; donor2 + MEA-keyed tabs show the empty-state banner ("Donor 2 has no IMAC; kinase MEA unavailable") rather than 0-row tables.
4. `outputs/reports/tcell_viewer/edge_slices/incytr_pathways/index.json` lists 6 donor-contrast combinations.
5. URL hash round-trip: setting `#donor=donor2&tab=incytrpathways&sender=CD8.TEM&receiver=CD8.TEx` reloads to the same state.
6. `outputs/reports/unified_viewer/` is **untouched** after `pixi run tcell-viewer` (no shared output state).
7. **No mouse-cohort vocabulary leaks** into the T-cell payload: grep `App`, `Tau`, `ApTt`, `wmb_class`, `levy_t5`, `Cholinergic`, `Microglia` in `tcell_viewer.payload.json` should return 0 matches.

## Out of scope (deferred, document in payload meta)

- **Evidence drawer.** Requires per-cluster transcript / protein / pY pseudobulk shards (`OMICS_TRACE_DIR`, `TRANSCRIPT_TRACE_DIR`) regenerated against the T-cell ProjecTILs spine. Land separately; until then, the Evidence sub-row stays collapsed with a "substrate trace not yet wired" notice.
- **scRNA UMAP overlay.** ProjecTILs UMAP is in the donor RDS; not in the current Incytr-input artifacts. Separate ingest step.
- **Cross-donor concordance.** No agreement_index analog yet — donor1 vs donor2 NES correlation, donor-pair Incytr survival rate, etc. Phase 2.

## Known residuals

- **Donor2 MEA absence is permanent** (no IMAC). The viewer must surface this as a first-class state, not a "data missing" banner that looks like a bug. Render an explanatory card on the kinase / temporalv2 / crosstable tabs for donor2.
- **Cluster name alphanumeric rule** (from project memory: no `_`, no `.`). ProjecTILs labels (`CD8.TEx`, `CD4.TFH`) contain dots and need sanitization. Decide at Step 2 implementation: strip dots (`CD8TEx`) or replace with camelcase (`Cd8Tex`). Stripping preserves readability; whichever choice, apply at the **decomposition / Incytr-input layer**, not in the viewer — viewer reads sanitized names from disk.

## Anti-shim checks (before merging)

- [ ] No mouse-cohort code paths flag-gated behind `if mode == "mouse"`. Mouse builder stays at `build_unified_viewer.py`; T-cell builder is independent.
- [ ] No `tcell_viewer/paths.py` constants kept "just in case" for the dropped Song / WMB / decomp_ols slice families.
- [ ] `feedback_no_reimplementing_shared_viewer` rule 1: every reused JS file is **byte-identical to the unified viewer's**, modulo TAB_MANIFEST mode-array edits and the donor selector additions. Run `diff -qr alz/viewer/template/js alz/tcell_viewer/template/js`; the expected delta is the documented edits, nothing else.

## References

- Unified viewer payload structure: `alz/build_unified_viewer.py:2533-2714` (`build_payload`)
- TAB_MANIFEST: `alz/tcell_viewer/template/js/02_ui_chrome.js:359-428`
- T-cell Incytr pair-mode plan: `docs/plans/tcells_incytr_pair_2026-05-28.md`
- T-cell ingest / per-cell aggregation: `docs/plans/tcells_percell_aggregation_2026-05-28.md`
- Pair-mode filter: `alz/incytr_pair/filter_significant_paths.py`
- Memory: `[[feedback_no_reimplementing_shared_viewer]]`, `[[project_tcell_exhaustion_cohort]]`
