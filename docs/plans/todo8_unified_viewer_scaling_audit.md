# Unified Viewer Scaling Audit

**Date:** 2026-06-19 (refreshed; original 2026-06-18)
**Status:** ALL P-ITEMS IMPLEMENTED — P1, P2, P4, P7 done/verified; P3, P5, P6, P8 implemented 2026-06-19. Full `pixi run viewer` + `pixi run tcell-viewer` rebuild done 2026-06-19 (11G cap, both exit 0): unified payload **53.38 MB raw / 5.63 MB gzip** (P5 sidecar confirmed; pathway round-trip verifier passed), tcell payload **11.17 MB raw / 1.50 MB gzip** emitted as sidecar (P8 confirmed). Remaining gate: browser click-through pass.

---

## Changelog vs. 2026-06-18 version

| Item | Change |
|---|---|
| **P3 + P5 + P6 + P8 implemented (2026-06-19)** | P3: removed eager `_kxBuildIndexes()` from `wireKinaseCrosstable` (now deferred to first render). P5: `gene_node_index` moved to per-context gzipped `edge_slices/incytr_pathways/{context}_gene_node_index.json.gz` sidecar, fetched on first gene search via `_ipEnsureGeneIndex`. P6: LRU cap (`_F5_CACHE_MAX=32`) on `_F5AttrCache`/`_F5CelltypeMeaCache`. P8: T-cell viewer now defaults to sidecar mode (`payload-data` = `null`); `--inline-payload` preserves air-gapped single-file; fixed the lifted loader to fetch `tcell_viewer.payload.json.gz` (was the unadapted `unified_viewer.*` name). Validation: 27 Python tests, template + payload-contract verifiers, P5 build-helper round-trip, P5 JS fetch+gunzip+resolve smoke, P8 HTML emission + loader/build filename match. |
| **P1 + P2 implemented + verified (2026-06-19)** | `celltype_mea_plot_index` and `celltype_attribution_summary_index` moved to gzipped `edge_slices/` sidecars, fetched once via `_f5EnsureShardData()`. Payload **105.2 MB → 69.0 MB raw, 10.1 MB → 8.49 MB gzip**. Tests + verifiers + live browser pass. See P1/P2 sections for the gzip-vs-raw note. |
| Payload sizes | Re-measured from current build (2026-06-19T00:27), **pre-P1/P2**. Total 105.2 MB / 10.0 MB gzip — within 1% of prior; per-key figures corrected. Post-P1/P2 totals are 69.0 MB / 8.49 MB. |
| Architecture map | `kinases` payload key is now context-aware: `kinases.by_context.song_ad.*`. Prior doc described it as a flat top-level columnar dict. |
| P3 (lazy tab wiring) | Partially resolved for 5xFAD. `wireFiveXFADKinase` defers all heavy work to `_f5EnsureIndexes`, which is lazy (called on render). `wireKinaseCrosstable` still runs `_kxBuildIndexes` eagerly at boot — half of P3 remains. |
| P4 (debounce) | **Resolved.** `_ipRenderTableDebounced` (180 ms) wired to both numeric sliders and the search box in `incytr_pathways.js`. |
| P7 (attribution_index Map) | **Resolved.** `_ensureAttributionRowsByKinase()` builds a lazy `Map<kinase_id, row_indices[]>` on first call; `kinase_audit.js` no longer scans `attribution_index` directly (uses `uaRows` from `AuditDataStore`). |
| T-cell viewer | Sidecar `.json.gz` exists on disk but index.html still inlines 19 MB payload. P8 remains. |
| 5xFAD incytr PTM run | In-flight (cortex only, 2/4 contrasts complete as of 2026-06-19). Quantified projected payload/edge-slice growth when it lands in the viewer. |
| NSCLC reference | Not on disk yet. Remains a future growth pressure item on the T-cell viewer only. |
| Edge slice file count | Prior doc listed "~2,470 files" — arithmetic error. Correct total is **2,966** (sum of per-dir counts). |
| New risk: context-keyed `kinases` | Payload key refactored from flat to `by_context`; no size change now, but each additional context (5xFAD incytr, NSCLC) will additively grow this key. Confirmed negligible at current scale. |
| New risk: 5xFAD incytr as second context | Quantified: cortex-only wide context would add ~20 MB raw / ~2 MB gzip to payload + ~310 MB edge slices + ~75 MB binary index on disk. Inlined `gene_node_index` growth is the dominant payload cost. |

---

## 1. Architecture Map

### 1.1 Build pipeline

| Script | Output |
|---|---|
| `alz/build_unified_viewer.py` | `outputs/reports/unified_viewer/{index.html, unified_viewer.payload.json, unified_viewer.payload.json.gz, edge_slices/**}` |
| `alz/build_tcell_viewer.py` | `outputs/reports/tcell_viewer/{index.html, tcell_viewer.payload.json, tcell_viewer.payload.json.gz, edge_slices/**}` |

The unified viewer currently covers **Song mouse-AD + Mukesh human-AD + 5xFAD**. The T-cell viewer is a sibling built by a shared-template system but is a separate output. Each viewer is a **single self-contained directory** — the HTML references the payload by relative URL and fetches edge slices on demand.

Since the original audit (2026-06-18) the cohort-abstraction refactor (ad23c86) has been fully merged. The viewer now uses per-cohort adapters `alz/viewer/cohorts/{song,mukesh,fivexfad}.py` and shared composer `alz/viewer/shared/{cohort_slice,compose}.py`. The JS still lives under `alz/viewer/template/js/` (tabs, widgets, 01_state.js, 06_export_csv.js — the last added in Wave 2 ec81e9d).

### 1.2 Payload delivery modes

1. **Unified viewer (sidecar/production):** `index.html` (738 KB) sets `<script id="payload-data">null</script>`. The browser fetches `unified_viewer.payload.json.gz` (10 MB) asynchronously, gunzips via `DecompressionStream`, and calls `JSON.parse`.
2. **T-cell viewer (sidecar after P8, 2026-06-19):** `index.html` now sets `<script id="payload-data">null</script>` and the browser fetches `tcell_viewer.payload.json.gz` (2.9 MB), gunzips, and parses — same path as the unified viewer. `--inline-payload` restores the air-gapped single-file mode. (Before P8 the full 19 MB JSON was baked into the HTML and the gz sidecar was unused.)
3. **Inline archival (`--inline-payload`):** Full JSON embedded in `index.html`. Used for air-gapped distribution of the unified viewer.

### 1.3 Runtime module graph

```
boot.js
 ├── _loadPayload()          # fetch + gunzip + JSON.parse (unified) or inline read (tcell)
 ├── 00_payload_adapter.js  → ViewerPayload.*  (context-aware getters)
 ├── 01_state.js            → Store (reducer), INITIAL_STATE, METRIC_DEFS, TAB_GUIDE
 ├── 02_ui_chrome.js        → TAB_MANIFEST (tab lifecycle registry)
 ├── 02_ui_chrome_common.js → shared chrome helpers
 ├── 03_filters_hash.js     → URL hash serialisation
 ├── 04_slice_cache.js      → SliceCache (LRU, 16-entry cap per shard type)
 ├── 05_header.js           → populateHeader(), syncHeaderFromStore()
 └── 06_export_csv.js       → csvEscape(), csvSerialize(), csvDownload() [Wave 2]
```

Tab wiring order (from `TAB_MANIFEST` in `02_ui_chrome.js`):

| Key | Group | Label | Modes | JS file |
|---|---|---|---|---|
| `temporalv2` | landscape | Temporal v2 | mouse | `kinase_explorer.js` + `temporal_v2.js` |
| `crosstable` | landscape | Crosstable | mouse, human | `kinase_crosstable.js` |
| `kinasehuman` | drilldown | Kinase (human) | human | `kinase_human.js` |
| `kinase` | drilldown | Kinase (mouse) | mouse | `kinase_explorer.js` + `kinase_wiring.js` + `kinase_audit.js` |
| `fivexfadkinase` | drilldown | Kinase (5xFAD) | fivexfad | `kinase_fivexfad.js` |
| `incytrheatmap` | landscape | Incytr Heatmap | mouse | `incytr_heatmap.js` |
| `incytrpathways` | drilldown | Incytr Pathways | mouse | `incytr_pathways.js` + `incytr_global_index.js` |
| `methods` | reference | Methods | mouse | (static HTML) |

Shared widgets: `evidence_row.js`, `multiselect.js`, `sequence_logo.js`, `transcript_trace.js`, `trend_filter.js`, `kinase_detail.js`.

### 1.4 Payload structure (unified viewer, build 2026-06-19T00:27)

**NOTE:** Sizes are measured by re-serializing each key with compact separators (`json.dumps(v, separators=(',',':'))`), which produces smaller numbers than default Python `json.dumps`. The on-disk `.json` file is 105.2 MB because the original serializer uses space-separated formatting. The gzip ratio is measured against the actual on-disk file.

| Key | Compact bytes | Notes |
|---|---|---|
| `supporting_5xfad` | **45,094,192 (43.0 MB)** | Dominant key. Sub-key breakdown below. |
| `attribution_index` | **26,756,426 (25.5 MB)** | Columnar, 108,531 rows × 30 columns. Per-contrast-per-kinase-per-celltype evidence table. |
| `incytr_pathways` | **20,748,986 (19.8 MB)** | `by_context.song_ad` only. Sub-key breakdown below. |
| `human` | **11,307,269 (10.8 MB)** | Mukesh cohort. Largest sub-key: `celltype_specificity` (5.4 MB). |
| `kinase_celltype_evidence` | 1,653,499 (1.6 MB) | Columnar, 8,146 rows. |
| `decomposition_index` | 2,464,063 (2.3 MB) | Columnar, 53,181 rows. |
| `audit_tables` | 709,970 (710 KB) | Per-kinase CSV previews. |
| `kinase_motifs` | 668,122 (668 KB) | Motif PSWMs. |
| `subclass_breakdown` | 691,751 (692 KB) | ~200 kinase IDs → per-celltype sub-dict. |
| `kinases` | 102,462 (102 KB) | Now **context-keyed**: `by_context.song_ad.*` — 389 kinases × NES/FDR per contrast. |
| `agreement_index` | 136,924 (137 KB) | Columnar. |
| `celltypes`, `edge_slice_ref`, `meta` | small | < 20 KB combined. |
| **TOTAL (on-disk uncompressed)** | **110,353,597 (105.2 MB)** | |
| **gzip compressed** | **10,570,356 (10.0 MB)** | 10× compression ratio. |

**`supporting_5xfad` sub-key breakdown:**

| Sub-key | Compact bytes | Notes |
|---|---|---|
| `celltype_attribution_summary_index` | 26,888,487 (25.6 MB) | 3,056 row objects with nested `celltypes` arrays. **Moved to sidecar by P2 (2026-06-19) — no longer inline.** |
| `celltype_mea_plot_index` | 14,435,012 (13.8 MB) | 79,433 row objects. **Moved to sidecar by P1 (2026-06-19) — no longer inline.** |
| `rows` | 2,036,319 (1.9 MB) | 6,224 bulk rows. |
| `celltype_agreement_index` | 1,666,328 (1.6 MB) | 3,112 rows. |
| shard reference dicts, qc, metadata | ~68 KB | |

**`incytr_pathways.by_context.song_ad` sub-key breakdown:**

| Sub-key | Compact bytes | Notes |
|---|---|---|
| `gene_node_index` | 15,646,385 (14.9 MB) | 5,984 genes × 15 parallel arrays. Only used for Incytr Pathways pair-mode gene search. **Moved to sidecar by P5 (2026-06-19) — no longer inline; referenced as `gene_node_index_shard`.** |
| `heatmap_counts_signed` | 3,589,124 (3.4 MB) | Precomputed 5D tensor. |
| `heatmap_counts` | 1,342,347 (1.3 MB) | Precomputed heatmap tensor. |
| `slice_index`, `global_index`, `celltype_qc`, etc. | ~190 KB | |

T-cell viewer (build 2026-06-18T22:51): 18.4 MB payload, two Incytr contexts (`donor1`, `donor2`). `incytr_pathways` = 12.3 MB (dominated by `gene_node_index` 3.3 + 3.9 MB per context). Inlined in `index.html` (19 MB total).

### 1.5 Edge slices (on-demand sidecar fetches)

All data under `edge_slices/` is **never pre-loaded** — fetched by `SliceCache` (per-kinase parquet/JSON) or, for the two whole-list index sidecars, by `_f5EnsureShardData()` on first 5xFAD/Crosstable render:

| Shard type | Files | On-disk size | LRU cap |
|---|---|---|---|
| `decomp_ols/` | 390 parquets | 311 MB | 16 |
| `fivexfad_attribution/` | 383 JSON.gz | 70 MB | unbounded Map |
| `fivexfad_celltype_mea/` | 390 JSON.gz | 22 MB | unbounded Map |
| `fivexfad_detail/` | 390 parquets | 194 MB | unbounded Map |
| `fivexfad_attribution_summary.json.gz` (P2) | 1 JSON.gz | 823 KB | loaded once, retained |
| `fivexfad_celltype_mea_index.json.gz` (P1) | 1 JSON.gz | 1.1 MB | loaded once, retained |
| `human_perdonor/` | 390 parquets | 12 MB | 16 |
| `incytr_pathways/` | 653 parquets + 1 bin.gz + 1 index.json | 155 MB + 42 MB index | 16 per context |
| `song_concordance/` | 368 parquets | 2.9 MB | 16 |
| **Total edge slices** | **2,968 files** | **769 MB** | |

*(Prior doc listed ~2,470 files — arithmetic error in summary; per-dir counts were correct. The two index sidecars were added by P1/P2 on 2026-06-19.)*

---

## 2. Scaling Risks (Measured)

### 2.1 Payload growth — O(cohorts) *(active)*

The payload grows **additively with each cohort addition**:

| Addition | Payload cost |
|---|---|
| Song mouse-AD (original) | ~25 MB raw (attribution_index + kinases + decomp) |
| Mukesh human-AD | +10.8 MB (`human` key) |
| 5xFAD support | +43.0 MB (`supporting_5xfad` key) |
| T-cell (separate viewer) | 18.4 MB payload, separate HTML |
| **In-flight: 5xFAD incytr as 2nd incytr context** | +~20 MB raw / ~2 MB gzip payload (gene_node_index dominant); +~310 MB edge parquets + ~75 MB binary index on disk |
| **Planned: NSCLC tcell reference (T-cell viewer)** | Unknown until data ingest; WMB analog at similar scale = +3–5 MB raw payload in T-cell viewer; NSCLC raw data not on disk yet |

At 2 further cohorts the raw payload crosses 150 MB, compressing to ~15 MB. `JSON.parse` of 15 MB on a mid-range laptop is 500–1,000 ms — noticeable but not a hard block. At 300+ MB raw / 30 MB compressed, cold-start parse becomes a genuine UX problem.

### 2.2 `supporting_5xfad.celltype_mea_plot_index` — full table in memory at tab-switch *(RESOLVED via P1, 2026-06-19)*

79,433 row objects (13.8 MB) were inlined in the payload and parsed at startup. Now written to `edge_slices/fivexfad_celltype_mea_index.json.gz` and fetched once on first 5xFAD/Crosstable render — out of the upfront parse. The in-memory index (`_F5CelltypeMeaRowsByAgeKey`, ~79k records) is still built at first render, but only when the tab is opened. Growth at 8 timepoints × 2 tissues still applies on disk/in-heap, no longer to startup parse.

### 2.3 `attribution_index` — O(1) per kinase after first call *(RESOLVED — was risk 2.3)*

`_ensureAttributionRowsByKinase()` builds a lazy `Map<kinase_id, row_indices[]>` on first call, making subsequent per-kinase lookups O(1). The one-time O(108k) build is deferred to first kinase selection. `kinase_audit.js` no longer scans `attribution_index` directly — it loads `uaRows` from `AuditDataStore` (CSV audit tables). **This bottleneck is resolved.** Retained for reference; superseded by §3.7.

### 2.4 `gene_node_index` — 14.9 MB inlined, used only for Incytr Pathways pair-mode *(RESOLVED via P5, 2026-06-19)*

Maps 5,984 genes to parallel arrays. Only needed when the Incytr Pathways tab runs a gene search, but was parsed upfront as part of the main payload. P5 moved it to a per-context gzipped sidecar (`edge_slices/incytr_pathways/{context}_gene_node_index.json.gz`), referenced from the block as `gene_node_index_shard` and fetched once on first gene-search interaction via `_ipEnsureGeneIndex` (mirroring `global_index`). Each additional Incytr context (5xFAD, NSCLC-tcell) now adds its index on disk, not to the upfront parse.

### 2.5 Incytr global index binary — 42 MB sidecar, full load required *(active)*

The `incytr_index.bin.gz` binary contains 4,480,480 rows as typed arrays (Float32/Uint16/Uint8 columns). Must be fully loaded before any global-mode filter. `filterRank` already uses typed arrays (correct approach). As additional Incytr contexts are added, each context gets its own binary index: 5xFAD cortex alone would be ~4M rows / ~38 MB, hippocampus ~2M rows / ~19 MB based on per-contrast row counts in the wide parquets.

### 2.6 T-cell viewer inlines payload — 19 MB *(RESOLVED via P8, 2026-06-19)*

The T-cell viewer inlined its 18.4 MB JSON payload directly in `index.html` (19 MB total); the on-disk sidecar `tcell_viewer.payload.json.gz` (2.9 MB) was unused. P8 made `build_tcell_viewer.write_html` default to sidecar mode (`payload-data` = `null`, mirroring the unified viewer), gated behind a new `--inline-payload` flag for air-gapped single-file distribution. **A latent bug surfaced:** the T-cell `01_state.js` was a lifted-but-unadapted copy that fetched `unified_viewer.payload.json.gz` — the wrong filename, harmless only because inline mode never hit the fetch path. Fixed to fetch the build's actual `tcell_viewer.payload.json.gz` / `.json`. `index.html` drops from 19 MB to ~750 KB.

### 2.7 5xFAD `celltype_attribution_summary_index` — O(n) full-table indexing on first tab open *(RESOLVED via P2, 2026-06-19)*

3,056 row objects (25.6 MB) were inlined and parsed at startup. Now written to `edge_slices/fivexfad_attribution_summary.json.gz`, fetched once on first 5xFAD/Crosstable render. The per-row `celltypes` sort and Map build still run on first render (~20–50 ms), but the 25.6 MB no longer loads into the upfront `JSON.parse`. The Crosstable's parallel build moved from `_kxBuildIndexes` (eager at boot) to `_kxBuildF5AttrIndex()` gated behind the same shard fetch.

### 2.8 Incytr Pathways `_ipRuntime.openKeys` — no cap *(low priority, no change)*

The set of open detail rows is unbounded. Unlikely to be a practical problem but has no guard.

### 2.9 Edge slice fetch — no total-bytes cap across SliceCache *(low priority, no change)*

SliceCache caps each shard type at 16 entries but does not track aggregate bytes. A session clicking through all shard types could retain several hundred MB in JS heap.

### 2.10 `fivexfad_attribution` and `fivexfad_celltype_mea` shard caches are unbounded Maps *(BOUNDED via P6, 2026-06-19; caveat below)*

`_F5AttrCache` and `_F5CelltypeMeaCache` were plain `Map`s with no eviction. P6 added an LRU cap (`_F5_CACHE_MAX=32`) matching `SliceCache`'s policy. Eviction is safe because `_f5AttributionReady`/`_f5CelltypeMeaReady` gate on the `*LoadedKinases` sets (not the cache) and the render path only ever targets the one selected kinase, so an evicted entry is never re-fetched and re-indexed.

**Honest caveat (not a half-fix to hide):** these caches hold the resolved-payload *wrappers*. The actual attribution/MEA rows are pushed into the persistent per-kinase indexes (`_F5AttrByKinase`, `_F5CelltypeMeaRowsByAgeKey`, `_F5CelltypeMeaByKey`) by reference, and those indexes are **never evicted** — they accumulate rows for every kinase ever clicked. So P6 bounds the cache half but the dominant per-kinase retention is the index, which P6 does not cap. Capping the index would require converting many synchronous index reads (`_f5AttrRowsForGroup`, etc.) to async-fetch-on-miss — a larger refactor than P6's scope (and the inverse of the sync-index design P1/P2 deliberately preserved). Tracked as a follow-up; not folded into P6.

### 2.11 NEW: In-flight 5xFAD incytr PTM run — projected viewer payload growth *(new finding)*

A 5xFAD incytr PTM run is currently generating parquets:
- `cortex/wide_ptm/`: 5-channel PTM run (pr, ps, py, AcK, KGG). 2/4 contrasts complete (3mo: 810 MB unfiltered / 13.9M rows; 6mo: 113 MB / 2.2M rows). 4 total contrasts per tissue.
- `cortex/wide/`: phospho-only derive (recoverable from wide_ptm). Currently empty (derive step pending).
- `hippocampus/`: not yet started.

When this data is wired into the viewer as a new Incytr context (planned per TODO #3), the viewer gains:
- **Payload (inline):** +14–20 MB raw / +1.5–2 MB gzip (gene_node_index for 5xFAD cell-type vocabulary, heatmap tensors per context)
- **Edge slices (on-disk):** +~310 MB parquets (655 shards × 2 tissues analog to song_ad) + ~75 MB binary index (cortex ~38 MB + hippo ~19 MB at estimated filtered row counts)
- **Binary index total:** currently 42 MB. With 5xFAD added: ~115–120 MB across two contexts.

At 5 channels (vs. 3 for song_ad), the `heatmap_counts_signed` tensor is proportionally larger (5/3 × 3.4 MB ≈ 5.7 MB per context). The `gene_node_index` size depends on the 5xFAD cell-type count (31 in cortex log vs. 34 in song_ad) — roughly similar.

### 2.12 NEW: `kinases` payload key is now context-keyed *(new finding — negligible at current scale)*

`kinases` was a flat columnar dict at the top level in the prior build. It is now `kinases.by_context.song_ad.*` (102 KB). With additional contexts the key grows additively, but given the small size (~102 KB per context) this is not a bottleneck. Noted because it changes the architecture map and any code that reads `PAYLOAD.kinases` directly (the adapter `ViewerPayload.kinases(ctx)` handles this transparently).

---

## 3. Performance Bottlenecks

### 3.1 Cold-start parse: `JSON.parse(text)` on 105 MB (10 MB over wire)

The browser fetches 10 MB gzip, decompresses to 105 MB text, calls `JSON.parse`. Estimated 300–800 ms synchronous block on a fast laptop. No tab can render until parse completes; the page shell is visible immediately. No change from prior audit.

### 3.2 No lazy initialization: crosstable wires eagerly at boot *(RESOLVED via P3, 2026-06-19)*

`boot()` calls `m.wire()` for every tab in `TAB_MANIFEST`. **For 5xFAD kinase tab:** `wireFiveXFADKinase` only wires DOM event listeners; the heavy index (`_f5EnsureIndexes`) is deferred to the first render call. **For Crosstable:** `wireKinaseCrosstable` previously called `_kxBuildIndexes` immediately (~170k row iterations at boot). P3 removed that call; `_kxRenderTable` / `renderKinaseCrosstable` already build the indexes on first render (guarded by `_KX_INITIALIZED`), so the ~170k iterations now move off the boot path to first Crosstable open — matching the 5xFAD tab's lazy behavior.

### 3.3 Kinase Explorer re-renders the full table on every filter change

`renderKinaseExplorer()` regenerates all `<tr>` HTML via `.map()` and sets `tbody.innerHTML` on every filter event. At 389 kinases and 9 NES columns, fast (< 5 ms). Becomes noticeable at ~1,000 kinases (multiple cohorts merged).

### 3.4 Crosstable `_kxBuildIndexes()` iterates all attribution/decomp/5xFAD data eagerly *(RESOLVED via P3 — see 3.2)*

On first Crosstable render (no longer at boot), `_kxBuildIndexes` iterates:
- 108,531 attribution rows (build `_KX_WMB_BY_KIN_CLUSTER`, `_KX_SONG_BY_KID`)
- 53,181 decomp rows (build `_KX_DECOMP_BY_KIN_CTX`)
- ~47 human kinase entries (build `_KX_HUMAN_BY_NAME`, `_KX_HUMAN_PERDONOR`)
- 6,224 5xFAD rows (build `_KX_F5_BY_KEY`)
- 3,056 5xFAD celltype attribution rows (build `_KX_F5_ATTR_BY_KEY`)
- 3,112 agreement rows (build `_KX_F5_AGREE_BY_KEY`)

Total: ~170k rows iterated unconditionally on boot.

### 3.5 The global Incytr index scan — O(4.48 M) with debounce *(RESOLVED)*

`filterRank` scans 4.48 M typed-array rows on each filter change. **P4 is implemented:** `_ipRenderTableDebounced` (180 ms debounce) wraps all slider and search-box events. Click events (sender/receiver chip selection) bypass the debounce and call `_ipRenderTable` directly (correct behavior). The underlying O(n) scan is the right design for typed arrays. No further action needed.

### 3.6 `heatmap_counts_signed` (3.4 MB) is fully parsed and retained

5D precomputed heatmap tensor parsed at startup as part of the main payload, never evicted. At multiple Incytr contexts, cumulative in-memory size grows. Currently 3.4 MB (song_ad only). With 5xFAD PTM context: +~5.7 MB (5 channels). Acceptable at 2 contexts; becomes a concern at 5+.

### 3.7 `attribution_index` lookup — O(1) after lazy Map build *(RESOLVED — was bottleneck)*

`_ensureAttributionRowsByKinase()` in `kinase_explorer.js` builds a `Map<kinase_id, row_indices[]>` lazily on first kinase selection (not at boot). All subsequent per-kinase lookups are O(1) index into the columnar arrays. `kinase_audit.js` uses `AuditDataStore` (loaded from CSV audit table files), not `attribution_index`, for attribution row display. **Both cited bottlenecks (kinase_audit line 136, kinaseQualifies O(n) scan) are resolved.**

---

## 4. Core Functionality Inventory

The following must be preserved after any refactor.

### Tabs

- **Kinase Explorer (mouse):** Sortable kinase table (389 kinases, 9 NES columns, cell-type columns, trajectory). Filter bar (FDR, disease, timepoint, celltype, confidence, n_sig, Song specificity, text search). Kinase detail panel (MEA bar chart, decomp bar chart, attribution drawer, measurement trace). **CSV export button (Wave 2).**
- **Kinase (human / Mukesh):** Per-donor NES table, donor selection, running enrichment, sequence logo, kinase site detail. **CSV export button (Wave 2).**
- **Kinase (5xFAD):** Tissue × age kinase summary tiles. Attribution drawer. MEA time-course chart. Running enrichment from edge shard. Agreement tiles. **CSV export button (Wave 2).**
- **Temporal v2:** Kinase temporal series visualization. **CSV export button (Wave 2).**
- **Crosstable:** Mouse/human/5xFAD agreement cross-table, sortable, filterable. **CSV export button (Wave 2).**
- **Incytr Heatmap:** 31×30 pair heatmap with threshold slider.
- **Incytr Pathways (pair mode):** Per-pair pathway table (paginated, 100 rows/page) loaded from edge slice parquets. Filter UI (sender, receiver, disease, timepoint, PDS slider, trajectory filter). Evidence detail panel per row. **CSV export (Wave 2).**
- **Incytr Pathways (global mode):** Global filter across 4.48 M pathway rows via binary index. Top-N cap (500/1000/5000). CSV export.
- **Methods:** Static reference tab.

### Cross-cutting features

- URL hash serialisation of filter/selection state (deep linking).
- Glossary tooltip system (`data-metric` → METRIC_DEFS).
- "How to read" drawer per tab.
- Export buttons (CSV download — standardized Wave 2 across all tabs).
- Tab mode gating (`mouse` / `human` / `fivexfad`).
- Prerequisite cards (graceful degradation when a payload key is absent).
- LRU SliceCache for edge slice parquets (16-entry cap per shard type).
- Backbone selection → kinase highlight cross-link.
- Context switch (currently song_ad; extensible to additional Incytr contexts).
- Signed NES display throughout (Wave 1: |NES| column now shows signed NES).

---

## 5. Prioritized Improvement Roadmap

Items are ordered by effort vs. impact. Resolved items moved to bottom.

---

### P1 — Move `supporting_5xfad.celltype_mea_plot_index` to an edge shard *(DONE 2026-06-19)*

**Status:** Implemented. `celltype_mea_plot_index` (79,433 rows) is written to `edge_slices/fivexfad_celltype_mea_index.json.gz` (1.1 MB gz) by `_write_fivexfad_index_shard` and fetched once on first 5xFAD/Crosstable render via `_f5EnsureShardData()` (kinase_fivexfad.js). The inline key is removed from the payload.

**Problem:** 13.8 MB of 79,433 row objects are inlined in the payload, parsed upfront, and indexed into memory on first 5xFAD tab render. Growth doubles at 8 timepoints or a second tissue.

**Proposed change:** At build time, write `celltype_mea_plot_index` to a per-kinase or single-index shard in `edge_slices/fivexfad_celltype_mea_index/`. Wire `_f5EnsureIndexes` to lazy-load on first 5xFAD tab render rather than reading from `PAYLOAD.supporting_5xfad.celltype_mea_plot_index`. The shard infrastructure for per-kinase celltype_mea already exists (`fivexfad_celltype_mea/` 390 files, 22 MB).

**Risk:** One additional roundtrip on first 5xFAD tab render. The prerequisite card infrastructure already handles "loading" states.

**Features touched:** 5xFAD Kinase tab (timing of data availability), Crosstable (also reads `celltype_mea_plot_index` via `_KX_F5_*` — will also need lazy loading or separate fetch).

**Regression proof:** 5xFAD tab renders correctly after shard load; Crosstable 5xFAD tiles populate within one roundtrip. Payload `unified_viewer.payload.json.gz` shrinks by ~1.4 MB (from ~10.0 MB to ~8.6 MB).

---

### P2 — Move `supporting_5xfad.celltype_attribution_summary_index` to a shard *(DONE 2026-06-19)*

**Status:** Implemented. `celltype_attribution_summary_index` (3,056 rows) is written to `edge_slices/fivexfad_attribution_summary.json.gz` (823 KB gz) and fetched once via `_f5EnsureShardData()`. The Crosstable's 5xFAD attribution index was decoupled from `_kxBuildIndexes` into `_kxBuildF5AttrIndex()` + `_kxEnsureF5AttrData()`, which reads the loaded rows via the cross-file accessor `_f5AttributionSummaryRows()` and re-renders when the shard resolves. The inline key is removed from the payload.

**Problem:** 25.6 MB, largest single key in the payload. 3,056 row objects with nested `celltypes` arrays. Parsed and Map-indexed at first 5xFAD tab render via `_f5EnsureIndexes`.

**Proposed change:** Write to `edge_slices/fivexfad_attribution_summary.json.gz` at build time; load lazily on first 5xFAD tab render. Same approach as the `fivexfad_attribution/` per-kinase shards (already exist at 70 MB / 383 files).

**Risk:** One additional roundtrip on first 5xFAD tab render.

**Features touched:** 5xFAD Kinase tab (table population timing), Crosstable (5xFAD confidence tier column).

**Regression proof:** Confidence summary populates after first tab render. **Combined P1+P2 measured result (2026-06-19 repackage): payload raw 105.2 MB → 69.0 MB; gzip 10.1 MB → 8.49 MB.** The raw/parse drop (−36 MB) matches the two keys' inlined size and is the dominant UX win (less synchronous `JSON.parse` before first render). The gzip drop (−1.6 MB) is smaller than the audit's earlier ~6.0 MB projection: these two keys are columnar/repetitive and compress far better than the payload's ~10× average, so they contributed less to the wire size than to the parse cost. Verified: `test_fivexfad.py` 9/9, payload-contract + template verifiers, Node smoke test of the index data-flow against the real shards, and a live browser pass confirming the 5xFAD tab and Crosstable render correctly (loading state → re-render on shard resolve) with no feature loss.

---

### P3 — Defer `_kxBuildIndexes` out of `wireKinaseCrosstable` boot path *(DONE 2026-06-19)*

**Status:** Implemented. Removed the `_kxBuildIndexes()` call from `wireKinaseCrosstable` (now only wires DOM listeners). The build is deferred to first render — both `_kxRenderTable` and `renderKinaseCrosstable` call `_kxBuildIndexes`, guarded by `_KX_INITIALIZED`. ~170k row iterations move off the boot path. Template verifier passes.

**Problem:** `wireKinaseCrosstable` (called at boot) calls `_kxBuildIndexes` immediately, iterating ~170k rows to build in-memory Maps. The 5xFAD tab equivalent (`_f5EnsureIndexes`) is already lazy (deferred to first render). Crosstable has the same guard machinery (`_KX_INITIALIZED`) but the first build still runs at boot.

**Proposed change:** Remove the `_kxBuildIndexes()` call from `wireKinaseCrosstable`. The call in `renderKinaseCrosstable` / `_kxRenderTable` (lines 1591–1594, 1302–1303) already ensures the indexes are built before first render. `wireKinaseCrosstable` should only wire DOM event listeners, not trigger the index build. This is a 1-line change: remove line 1533 (`_kxBuildIndexes();`) from `wireKinaseCrosstable`.

**Risk:** First render of Crosstable will have an added ~20–50 ms initialization pause (index build). This is already the 5xFAD tab's behavior and is acceptable.

**Features touched:** Crosstable — index build timing only.

**Regression proof:** Tab functionality is identical after first render. Boot time drops by ~20–50 ms.

---

### P4 — Debounce Incytr global index filter scan *(RESOLVED)*

**Status:** Implemented in Wave 2 / prior to this audit. `_ipRenderTableDebounced` (180 ms debounce) wired to sliders and search box in `wireIncytrPathways`. Click events bypass debounce as designed. No further action.

---

### P5 — Move `gene_node_index` out of the inlined payload into a sidecar *(DONE 2026-06-19)*

**Status:** Implemented. New shared helper `_write_gene_node_index_shard()` (payload_helpers.py) writes the index to a per-context gzipped sidecar (`edge_slices/incytr_pathways/gene_node_index.json.gz`; T-cell uses `{donor}__gene_node_index.json.gz`) and the block now carries `gene_node_index_shard` instead of the inline `gene_node_index`. JS: `_ipEnsureGeneIndex` fetches + gunzips once on first gene search (loading state → re-render), `_ipResolveGeneIndex` feeds `_ipGeneIndexMap`/`_ipGeneIndexMatches`; the inline read path was removed (anti-shim). Both builders (`song.py`, `build_tcell_viewer.py`) updated; song's build-cache file list includes the new sidecar. Validated: build-helper gz round-trip + a Node smoke exercising the real fetch→DecompressionStream→resolve→map path (1 fetch, cached).

**Problem:** 14.9 MB inlined in the payload, required only when the Incytr Pathways tab is opened in pair mode (gene search). Growth: each additional Incytr context (5xFAD, NSCLC) adds another 14–20 MB.

**Proposed change:** Write `gene_node_index` as a JSON.gz sidecar per context (`edge_slices/incytr_pathways/{context}_gene_node_index.json.gz`). Load lazily via `SliceCache` on first pair-mode gene search interaction. `IncytrGlobalIndex` already fetches the binary index lazily; `gene_node_index` can follow the same pattern.

**Risk:** Gene search in pair mode will show a "Loading…" state on first use. Moderate change touching `build_unified_viewer.py`, `payload_helpers.py`, and `incytr_pathways.js`.

**Features touched:** Incytr Pathways gene search (pair mode).

**Regression proof:** Gene search returns identical results; payload gzip shrinks by ~1.4 MB. **Combined P1+P2+P5 target: gzip ≤ 5.0 MB.**

---

### P6 — Cap the 5xFAD attribution and celltype MEA shard caches *(DONE 2026-06-19 — partial; see §2.10 caveat)*

**Status:** Implemented. Added `_f5CacheSet` (LRU, `_F5_CACHE_MAX=32`) and applied it to `_F5AttrCache`/`_F5CelltypeMeaCache` set + on-hit touch, mirroring `SliceCache._lruTouch`. Eviction is safe (the `*Ready` gates key off `*LoadedKinases`, not the cache). **Caveat:** this bounds the fetch-promise caches only; the persistent per-kinase indexes (`_F5AttrByKinase`, `_F5CelltypeMeaRowsByAgeKey`, `_F5CelltypeMeaByKey`) hold the rows by reference and remain uncapped — that is the dominant retention and a larger async-on-miss refactor, deliberately out of P6's scope (§2.10).

**Problem:** `_F5AttrCache` and `_F5CelltypeMeaCache` are plain `Map` objects with no eviction. A session clicking through all 390 kinases accumulates ~70 MB in-heap for attribution, ~22 MB for celltype_mea. Will double with hippocampus.

**Proposed change:** Apply the same LRU eviction pattern used in `SliceCache` (`_lruTouch`, `MAX = 16` or 32) to both caches. Add a local `_f5LruTouch` helper in `kinase_fivexfad.js` mirroring `SliceCache._lruTouch`.

**Risk:** User clicking back to an evicted kinase will trigger a re-fetch (brief loading state). Acceptable — same UX as edge slice parquets.

**Features touched:** 5xFAD Kinase tab (re-fetch on evicted kinases).

**Regression proof:** No feature change; memory footprint does not grow proportionally with session length.

---

### P7 — Pre-build attribution_index Map for O(1) kinase lookup *(RESOLVED)*

**Status:** Implemented prior to this audit. `_ensureAttributionRowsByKinase()` in `kinase_explorer.js` (line 36) builds a lazy `Map<kinase_id, row_indices[]>` on first call. All per-kinase attribution lookups are O(1) index operations. `kinase_audit.js` reads `uaRows` from `AuditDataStore`, not from `attribution_index` directly. **Resolved.**

---

### P8 — Switch T-cell viewer to sidecar payload mode *(DONE 2026-06-19)*

**Status:** Implemented. `build_tcell_viewer.write_html` gained `inline_payload` (default `False` → `payload-data` = `null`; `--inline-payload` bakes the JSON for air-gapped use), mirroring the unified builder. **Correction to the original P8 note ("no JS changes required"):** the lifted T-cell `01_state.js` fetched the wrong sidecar name (`unified_viewer.payload.json.gz`), masked by inline mode; fixed to `tcell_viewer.payload.json.gz` / `.json`. Validated: HTML emission test (sidecar→`null`, inline→embedded) + loader/build filename match. `index.html` 19 MB → ~750 KB.

**Problem:** T-cell viewer inlines 18.4 MB of JSON in `index.html` (19 MB total), causing the browser to parse the full document before any JS loads. A sidecar `tcell_viewer.payload.json.gz` (2.9 MB) already exists on disk but is NOT used — the `<script id="payload-data">` tag contains the full JSON inline, not `null`. Adding NSCLC reference data would push the inline payload further.

**Proposed change:** In `build_tcell_viewer.py`, emit the payload as a sidecar (the file already exists); change the inline block to write `null` instead of the serialized JSON. The shared `_loadPayload()` already handles the null-sidecar path. No JS changes required.

**Risk:** The tcell viewer must be served over HTTP (same requirement as the unified viewer). Air-gapped distribution uses `--inline-payload` (already supported).

**Features touched:** T-cell viewer delivery mode; air-gapped distribution unchanged.

**Regression proof:** T-cell viewer renders identically; `index.html` shrinks from 19 MB to ~750 KB.

---

## 6. Measurement and Benchmark Approach

Improvements must be measured, not asserted.

### 6.1 Payload size

```bash
# Pre/post each P-item
wc -c outputs/reports/unified_viewer/unified_viewer.payload.json
wc -c outputs/reports/unified_viewer/unified_viewer.payload.json.gz
# T-cell
wc -c outputs/reports/tcell_viewer/index.html
wc -c outputs/reports/tcell_viewer/tcell_viewer.payload.json.gz
```

Target after P1+P2+P5: gzip payload ≤ 5 MB (from 10.0 MB).
Target after P1+P2: gzip payload ≤ 6.0 MB.

### 6.2 Cold-start parse time

```javascript
// Add temporarily to _loadPayload() for benchmarking:
const t0 = performance.now();
PAYLOAD = JSON.parse(text);
console.log("JSON.parse ms:", performance.now() - t0, "text bytes:", text.length);
```

Target: parse time ≤ 400 ms at 5 MB gzip / ~50 MB uncompressed.

### 6.3 Boot-to-first-render time

DevTools → Performance → "Start profiling and reload page". Measure from `navigationStart` to first `renderKinaseExplorer()`. Baseline before P3 (crosstable lazy init). Target: boot drops by ~20–50 ms.

### 6.4 Filter latency (Incytr global mode) *(resolved via P4 — measure to confirm)*

Confirm response time ≤ 200 ms from slider release to table update with the 180 ms debounce in place.

### 6.5 Memory footprint

Chrome DevTools → Memory → Heap Snapshot at: (a) payload loaded, (b) 50 kinases clicked, (c) 5 Incytr pairs navigated. Compare heap before and after P6 (5xFAD cache cap).

### 6.6 Edge slice fetch count and size

DevTools → Network → filter by `parquet` or `.json.gz`. Typical session baseline: open kinase tab → 3 kinases → Incytr Pathways → 3 pairs.

---

## 7. Feature Preservation Checklist

Before shipping any P-item, verify all of the following:

- [ ] All 8 tabs (temporalv2, crosstable, kinasehuman, kinase, fivexfadkinase, incytrheatmap, incytrpathways, methods) render without console errors.
- [ ] URL hash deep-linking works: paste a URL with `#kinase=AKT1&contrast=App_4mo` and the correct kinase and contrast are selected.
- [ ] Incytr Pathways pair mode: selecting Microglia → Cholinergic Neurons loads the shard and renders a paginated table.
- [ ] Incytr Pathways global mode: setting PDS ≥ 0.5 filters to a non-empty result; CSV export contains only filtered rows.
- [ ] Kinase Audit: clicking a kinase shows the attribution verdict panel with confidence tiers.
- [ ] 5xFAD Kinase tab: kinase tile grid populates; clicking a tile shows the MEA time-course chart.
- [ ] Human Kinase tab: per-donor NES bar chart renders; sequence logo loads for a kinase with leading substrates.
- [ ] Crosstable: 5xFAD agreement tiles appear in the row for a kinase present in `supporting_5xfad`.
- [ ] Prerequisite cards appear (not crash) when a payload key is absent (e.g., `supporting_5xfad` missing).
- [ ] Glossary tooltips (`data-metric`) populate on all tabs.
- [ ] CSV export works on all 6 export-enabled tabs (Kinase Explorer, Kinase Human, Kinase 5xFAD, Temporal v2, Crosstable, Incytr Pathways).
- [ ] `pixi run python alz/viewer/verify_payload_contract.py` passes.
- [ ] `pixi run python alz/viewer/verify_template.py` passes.

---

## 8. Cohort-Growth Projection

| Scenario | Estimated raw payload | Estimated gzip |
|---|---|---|
| Current (Song + Mukesh + 5xFAD) | 105 MB | 10.0 MB |
| After P1+P2 (5xFAD large keys sharded out) — **MEASURED 2026-06-19** | **69.0 MB** | **8.49 MB** |
| After P1+P2+P5 (gene_node_index also sharded) | ~51 MB | ~5.0 MB |
| + 5xFAD incytr context (cortex wide) | ~70 MB | ~7.0 MB |
| + 5xFAD incytr context (cortex wide_ptm) | ~76 MB | ~7.6 MB |
| + NSCLC tcell reference (T-cell viewer only) | ~23 MB | ~3.5 MB (tcell) |
| + 2nd AD mouse cohort at current scale | ~90 MB | ~9.0 MB |
| + 5xFAD at 8 timepoints (no P1/P2) | ~210 MB | ~21 MB (parse pain) |
| + 5xFAD at 8 timepoints (with P1/P2) | ~80 MB | ~8.0 MB (acceptable) |

The conclusion is that **P1+P2 together are the highest-leverage changes**, directly addressing the 5xFAD data's dominant share of the payload (43 MB → ~1.4 MB net after both are sharded). P5 (now done) handles the Incytr gene_node_index growth pressure that compounds with each new Incytr context. P3, P6, P8 (now done) are incremental polish and memory hygiene. All eight P-items are implemented as of 2026-06-19. The full `pixi run viewer` rebuild on 2026-06-19 measured the post-P5 payload at **53.38 MB raw / 5.63 MB gzip** — down from 69.0 MB / 8.49 MB after P1+P2 (P5's gene_node_index sidecar removed ~15.6 MB raw / ~2.9 MB gzip, beating the projection because the index is highly repetitive and compresses well). The tcell viewer rebuilt to 11.17 MB raw / 1.50 MB gzip in P8 sidecar mode. The one item explicitly left open is capping the persistent 5xFAD per-kinase indexes (§2.10) — out of P6's scope.
