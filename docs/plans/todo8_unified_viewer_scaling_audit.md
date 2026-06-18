# Unified Viewer Scaling Audit

**Date:** 2026-06-18
**Status:** AUDIT — awaiting approval before any implementation

---

## 1. Architecture Map

### 1.1 Build pipeline

| Script | Output |
|---|---|
| `alz/build_unified_viewer.py` | `outputs/reports/unified_viewer/{index.html, unified_viewer.payload.json, unified_viewer.payload.json.gz, edge_slices/**}` |
| `alz/build_tcell_viewer.py` | `outputs/reports/tcell_viewer/{index.html, tcell_viewer.payload.json, tcell_viewer.payload.json.gz, edge_slices/**}` |

The unified viewer currently covers **Song mouse-AD + Mukesh human-AD + 5xFAD**; the T-cell viewer is a sibling built by a shared-template system but is a separate output. Each viewer is a **single self-contained directory** — the HTML references the payload by relative URL and fetches edge slices on demand.

### 1.2 Payload delivery modes

1. **Hosted/sidecar (production):** `index.html` references `unified_viewer.payload.json.gz` by relative URL. The browser fetches and gunzip-decompresses it via `DecompressionStream` before `JSON.parse`. The `<script id="payload-data">` tag contains only `null`.
2. **Inline archival (--inline-payload):** The full JSON is embedded inside the `<script id="payload-data">` tag. Used for air-gapped distribution.

The T-cell viewer uses **inline-only** (the 18 MB payload is baked into `index.html`). The unified viewer uses sidecar (only 749 KB HTML is served first; the 10 MB gzip follows asynchronously).

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
 └── 05_header.js           → populateHeader(), syncHeaderFromStore()
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

Shared widgets used across tabs: `evidence_row.js`, `multiselect.js`, `sequence_logo.js`, `transcript_trace.js`, `trend_filter.js`, `kinase_detail.js`.

### 1.4 Payload structure (unified viewer)

Measured sizes from `unified_viewer.payload.json` (2026-06-17 build):

| Key | Raw bytes | Notes |
|---|---|---|
| `supporting_5xfad` | **48,674,556** (48.7 MB) | Dominant key. Contains 5xFAD bulk rows (6,224), celltype_attribution_summary_index (3,056 rows, **28.7 MB**), celltype_mea_plot_index (79,433 rows, **15.9 MB**), agreement_index (3,112 rows, 1.8 MB), shard reference dicts. |
| `attribution_index` | **30,012,385** (30 MB) | Columnar format, 108,531 rows × 30 columns. Per-contrast-per-kinase-per-celltype evidence table. |
| `incytr_pathways` | **26,620,848** (26.6 MB) | One context (`song_ad`). Largest sub-keys: `gene_node_index` (19.4 MB, 5,984 genes × columns for 4.48 M pathway rows), `heatmap_counts_signed` (5.2 MB precomputed), `heatmap_counts` (1.9 MB). Binary global index is a **sidecar** (`incytr_index.bin.gz`, 42 MB on disk, loaded lazily). |
| `human` | **10,545,779** (10.5 MB) | Mukesh cohort. Largest sub-key: `celltype_specificity` (5.9 MB). |
| `decomposition_index` | 2,729,972 (2.7 MB) | Columnar, 53,181 rows. |
| `subclass_breakdown` | 701,245 (701 KB) | ~200 kinase IDs → per-celltype sub-dict. |
| `audit_tables` | 752,793 (753 KB) | Per-kinase CSV previews. |
| `kinase_motifs` | 767,009 (767 KB) | Motif PSWMs. |
| `kinase_celltype_evidence` | 1,832,732 (1.8 MB) | Columnar, 8,146 rows. |
| `agreement_index` | 157,036 (157 KB) | Columnar. |
| `kinases` | 115,333 (115 KB) | Columnar per context: 389 kinases × NES/FDR per contrast. |
| `celltypes`, `edge_slice_ref`, `meta`, `kinase_motifs` | small | < 1 MB combined. |
| **TOTAL uncompressed** | **108,614,903 (104 MB)** | |
| **gzip compressed** | **10,442,056 (10 MB)** | 10× compression ratio. |

T-cell viewer (2026-06-15 build): 18 MB inlined payload, dominated by `incytr_pathways` (14.9 MB, 2 contexts × per-context `gene_node_index` ~4.2–4.9 MB each, `top_instances` ~2.4 MB each).

### 1.5 Edge slices (on-demand sidecar fetches)

All data under `edge_slices/` is **never pre-loaded** — fetched by `SliceCache` on user interaction:

| Shard type | Files | On-disk size | LRU cap |
|---|---|---|---|
| `decomp_ols/` | 390 parquets | 311 MB | 16 |
| `fivexfad_attribution/` | 383 JSON.gz | 70 MB | (Map, unbounded) |
| `fivexfad_celltype_mea/` | 390 JSON.gz | 22 MB | (Map, unbounded) |
| `fivexfad_detail/` | 390 parquets | 194 MB | (Map, unbounded) |
| `human_perdonor/` | 390 parquets | 12 MB | 16 |
| `incytr_pathways/` | 655 parquets + 1 bin.gz | 155 MB + 42 MB index | 16 per context |
| `song_concordance/` | 368 parquets | 2.9 MB | 16 |
| **Total edge slices** | ~2,470 files | **764 MB** | |

---

## 2. Scaling Risks (Measured)

### 2.1 Payload growth — O(cohorts)

The payload grows **additively with each cohort addition**:

| Addition | Payload cost |
|---|---|
| Song mouse-AD (original) | ~25 MB raw (attribution_index + kinases + decomp) |
| Mukesh human-AD | +10.5 MB (`human` key) |
| 5xFAD support | +48.7 MB (`supporting_5xfad` key) |
| T-cell (separate viewer) | 18 MB payload, separate HTML |
| **Hypothetical: 2nd AD cohort (mouse)** | +25–50 MB raw; gzip ~3–5 MB |

The **5xFAD `celltype_attribution_summary_index` is the single largest contributor (28.7 MB raw)**. It is a flat list of 3,056 row objects, each carrying a `celltypes` array of deeply nested attribution evidence — the verbosity comes from nested sub-objects per cell type inside each row. A second cohort with the same 5xFAD data structure at larger scale (more ages, more tissues) could push this to 80–100 MB raw.

At 2 further cohorts at the current rate the raw payload crosses 200 MB, which compresses to ~20 MB. Browser `JSON.parse` time for 20 MB of text on a mid-range laptop is measurable (1–3 seconds). At 300+ MB raw / 30 MB compressed, cold-start parse time is a genuine UX problem.

### 2.2 `supporting_5xfad.celltype_mea_plot_index` — full table in memory at tab-switch

This is 79,433 row objects inlined in the payload. At tab open (`wireFiveXFADKinase`), the entire list is iterated in `_f5IndexCelltypeMeaRows` and `_f5IndexAttributionRows` — maps are built in-memory from the full 15.9 MB array. This is a **one-time O(n) scan** but retains all 79k records in `_F5CelltypeMeaRowsByAgeKey`. Growth path: each additional tissue or timepoint adds another proportional block of rows. At 4 ages × 2 tissues × 390 kinases × ~25 cell types = 78,000 rows currently, adding hippo + cortex at 8 ages doubles to ~160k rows.

### 2.3 `attribution_index` — O(n) scan on every kinase selection

`attribution_index` is a columnar dict with 108,531 rows. The kinase audit reads it by iterating the full column arrays on every kinase click (function `_attrRows` in `kinase_audit.js`, line 136: `for (let i = 0; i < AI.kinase_id.length; i++)`). At 108k rows this is fast (< 10 ms) but at 4 cohorts could reach ~400k rows and become noticeable.

The same pattern is used in `kinase_explorer.js` (`_ensureKinaseIndexes` / `kinaseQualifies`), which filters 108k rows on every table re-render triggered by filter changes.

### 2.4 `gene_node_index` — 19.4 MB inlined, used only for Incytr Pathways

`gene_node_index` maps 5,984 genes to parallel arrays of `gene_id`, `role_id`, `sender_id`, `receiver_id` with `n_rows`, `best_abs_pds` etc. This is only accessed when the Incytr Pathways tab is open, but it must be parsed upfront because it is inlined in the payload. At 2 additional Incytr contexts (e.g., a human-AD Incytr run) this key grows proportionally: each context adds 15–20 MB.

### 2.5 Incytr global index binary — 42 MB sidecar, full load required

The binary index (`incytr_index.bin.gz`) contains 4,480,480 rows as typed arrays (Float32/Uint16/Uint8 columns). It must be **fully loaded before any global-mode filter** can run. The `filterRank` function is a tight loop over all 4.48 M rows on every filter change — this is already the right design (typed arrays, no object allocation), but as additional Incytr contexts are added the binary grows: each context adds proportionally (donor1 + donor2 in the T-cell viewer are per-context in the inline payload; the unified viewer uses a single song_ad binary index). If multiple contexts share a single binary, the n-row count multiplies by context count.

### 2.6 T-cell viewer inlines payload; future cohorts would inflate `index.html`

The T-cell viewer inlines its 18 MB JSON payload directly in `index.html`. This is **safe at 18 MB** but means the browser must download the full HTML before any JS executes. Adding a second T-cell cohort or donor would push inline payload toward 35–40 MB, making the initial document load perceptibly slow over typical corporate VPN connections.

### 2.7 5xFAD `celltype_attribution_summary_index` — O(n) full-table indexing on first tab open

3,056 row objects, each with a `celltypes` array. Each row's `celltypes` array is sorted in-place inside `wireFiveXFADKinase`. This is O(3,056 × k log k) where k = avg celltypes per entry. Not a bottleneck today but the sort happens on first tab open, adding ~20–50 ms.

### 2.8 Incytr Pathways `_ipRuntime.openKeys` — no cap

The set of open detail rows (`_ipRuntime.openKeys`) is unbounded. Each open row inlines an evidence panel (Plotly chart + gene evidence). With 100 rows per page and aggressive usage, a user can accumulate a large number of open panels. This is unlikely to be a practical problem but has no guard.

### 2.9 Edge slice fetch — no total-bytes cap across SliceCache

SliceCache caps each shard type at 16 in-memory entries. A user clicking through 16 kinases in decomp_ols, 16 in human_perdonor, 16 incytr pairs, and 16 song concordances could retain ~(16 × avg_shard_size × 4) = several hundred MB in JS heap. The LRU cap prevents unbounded growth per type but does not track aggregate bytes.

### 2.10 The `fivexfad_attribution` and `fivexfad_celltype_mea` shard caches are unbounded Maps

`_F5AttrCache` and `_F5CelltypeMeaCache` are plain `Map` objects with no eviction. Once a 5xFAD kinase attribution shard is loaded it stays in memory for the session. With 390 kinases × 2 cache maps and shard sizes averaging 70 MB / 390 ≈ 180 KB per attribution shard, a user clicking through all kinases accumulates ~70 MB in-heap. For the celltype_mea shards it is ~55 KB each, so ~22 MB. These are within reason today but will double with a second 5xFAD tissue cohort.

---

## 3. Performance Bottlenecks

### 3.1 Cold-start parse: `JSON.parse(text)` on 104 MB (10 MB over wire)

The browser fetches 10 MB gzip, decompresses to 104 MB text, and calls `JSON.parse`. On a fast laptop this takes ~300–800 ms. The decompression itself uses native `DecompressionStream` (fast). `JSON.parse` of 104 MB is the single-largest synchronous event on the main thread at startup. It blocks tab interaction until complete.

**Caveat:** payload is fetched asynchronously after `DOMContentLoaded`; the page shell is visible immediately. But no tab can render until parse completes.

### 3.2 No lazy initialization: all tabs wire on boot

`boot()` calls `m.wire()` for every tab in `TAB_MANIFEST` **once**, synchronously. Some wire functions are trivial (closures), but for `kinase_crosstable` / `kinase_fivexfad`, wiring triggers index construction (e.g., `_initKinaseCrosstableData()` builds Maps from 5xFAD attribution rows, 53k decomp rows, 8k celltype evidence rows). This construction happens at boot even if the user never opens those tabs.

### 3.3 Kinase Explorer re-renders the full table on every filter change

`renderKinaseExplorer()` regenerates all `<tr>` HTML via `.map()` and sets `tbody.innerHTML` on every filter event. With 389 kinases and 9 NES columns each, this is fast (< 5 ms) today. It could become noticeable if kinase count grows to ~1,000 (multiple cohorts merged into one kinase universe).

### 3.4 Crosstable `wireKinaseCrosstable()` iterates all attribution/decomp/5xFAD data eagerly

At init time, `_initKinaseCrosstableData()` iterates:
- 108,531 attribution rows (build `_KX_HUMAN_BY_NAME`, `_KX_HUMAN_SPEC_BY_NAME`)
- 53,181 decomp rows (build `_KX_DECOMP_BY_KIN_CTX`)
- 6,224 5xFAD rows (build `_KX_F5_BY_KEY`)
- 3,056 5xFAD celltype attribution rows (build `_KX_F5_ATTR_BY_KEY`)
- 3,112 agreement rows (build `_KX_F5_AGREE_BY_KEY`)

All of this runs **on boot**, not lazily. Total iteration: ~170k rows.

### 3.5 The global Incytr index scan is O(4.48 M) on every filter change

`filterRank` scans all 4.48 M rows on each user keystroke / slider movement. The inner loop uses typed arrays with no allocations — this is the correct approach and runs in ~50–100 ms on a modern machine. However, there is **no debounce** between slider events and `filterRank` calls in `incytr_pathways.js`. Rapid slider movement triggers a scan per event.

### 3.6 `heatmap_counts_signed` (5.2 MB) is fully parsed and retained

This key contains a 5D precomputed heatmap tensor (31 senders × 30 receivers × 9 contrasts × 3 sign classes × 8 thresholds = 201,960 cells). It is parsed at startup as part of the main payload and never evicted. This is acceptable at current scale but if multiple Incytr contexts are added (each with their own signed heatmap), the cumulative in-memory size grows.

### 3.7 `subclass_breakdown` is a dict of ~200 kinase IDs each with per-celltype string tooltips

Each value is looked up at render time (`_sbk = (PAYLOAD.subclass_breakdown || {})[String(ctx.kinase_id)]`). This is O(1) per lookup — no bottleneck. Included here for completeness.

---

## 4. Core Functionality Inventory

The following must be preserved after any refactor. A regression is a feature lost or behavior change that a downstream viewer consumer (CSO, executive, research collaborator) would notice.

### Tabs

- **Kinase Explorer (mouse):** Sortable kinase table (389 kinases, 9 NES columns, cell-type columns, trajectory). Filter bar (FDR, disease, timepoint, celltype, confidence, n_sig, Song specificity, text search). Kinase detail panel (MEA bar chart, decomp bar chart, attribution drawer, measurement trace).
- **Kinase (human / Mukesh):** Per-donor NES table, donor selection, running enrichment, sequence logo, kinase site detail.
- **Kinase (5xFAD):** Tissue × age kinase summary tiles. Attribution drawer. MEA time-course chart. Running enrichment from edge shard. Agreement tiles.
- **Temporal v2:** Kinase temporal series visualization.
- **Crosstable:** Mouse/human/5xFAD agreement cross-table, sortable, filterable.
- **Incytr Heatmap:** 31×30 pair heatmap with threshold slider.
- **Incytr Pathways (pair mode):** Per-pair pathway table (paginated, 100 rows/page) loaded from edge slice parquets. Filter UI (sender, receiver, disease, timepoint, PDS slider, trajectory filter). Evidence detail panel per row (Plotly trajectory chart, node evidence).
- **Incytr Pathways (global mode):** Global filter across 4.48 M pathway rows via binary index. Top-N cap (500/1000/5000). CSV export.
- **Methods:** Static reference tab.

### Cross-cutting features

- URL hash serialisation of filter/selection state (deep linking).
- Glossary tooltip system (`data-metric` → METRIC_DEFS).
- "How to read" drawer per tab.
- Export buttons (CSV download from Incytr Pathways).
- Tab mode gating (`mouse` / `human` / `fivexfad`).
- Prerequisite cards (graceful degradation when a payload key is absent).
- LRU SliceCache for edge slice parquets.
- Backbone selection → kinase highlight cross-link.
- Context switch (currently song_ad; extensible to additional Incytr contexts).

---

## 5. Prioritized Improvement Roadmap

Items are ordered by effort vs. impact. Each item is strictly additive (no features removed) unless stated as a replacement.

---

### P1 — Move `supporting_5xfad.celltype_mea_plot_index` to an edge shard

**Problem:** 15.9 MB of 79,433 row objects are inlined in the payload, parsed upfront, and indexed into memory on first 5xFAD tab open. Growth doubles at 8 timepoints or a second tissue.

**Proposed change:** At build time, write `celltype_mea_plot_index` to `edge_slices/fivexfad_celltype_mea_index.json.gz` (already exists as a per-kinase shard directory — consolidate or add a keyed index shard). Wire `_f5IndexCelltypeMeaRows` to lazy-load on first tab open rather than pulling from `PAYLOAD.supporting_5xfad.celltype_mea_plot_index`. A single fetch of the full index on first tab open is acceptable; per-kinase lazy loading is already in place for attribution shards.

**Risk:** Adds one fetch roundtrip on first 5xFAD tab open (users will see a brief "Loading…" state). The shard infrastructure is already there (`fivexfad_celltype_mea/` parquets exist). Need to decide: one flat index or per-kinase continues as is. Per-kinase is already implemented; the only question is removing the inline copy. The 5xFAD tab already has a "5xFAD payload data are not available" prerequisite card, so a loading state is already handled architecturally.

**Features touched:** 5xFAD Kinase tab (timing of data availability), Crosstable (also reads `celltype_mea_plot_index` via `_KX_F5_*`).

**Regression proof:** After the change, 5xFAD kinase tab renders correctly after shard load; Crosstable 5xFAD tiles populate within one additional roundtrip. Payload `unified_viewer.payload.json.gz` shrinks by ~1.5 MB.

---

### P2 — Move `supporting_5xfad.celltype_attribution_summary_index` to a shard

**Problem:** 28.7 MB, largest single key in the payload. 3,056 row objects with nested `celltypes` arrays. Parsed and Map-indexed at first 5xFAD tab open.

**Proposed change:** Write to `edge_slices/fivexfad_attribution_summary.json.gz` at build time; load lazily on first 5xFAD tab open. Same approach as the `fivexfad_attribution/` per-kinase shards (already exist at 70 MB / 383 files). This could replace or supplement the per-kinase shard with a single summary index shard.

**Risk:** Same as P1 (one additional roundtrip). The per-kinase `fivexfad_attribution/` shards already deliver the full row data for selected kinases; the summary index is the "which kinases have high-confidence attribution" pre-filter. Loading the summary index on tab open (not on payload fetch) is the right layering.

**Features touched:** 5xFAD Kinase tab (table population timing), Crosstable (5xFAD confidence tier column).

**Regression proof:** 5xFAD tab confidence summary populates after first tab open. Payload gzip shrinks by ~2.8 MB (from 10 MB to ~7.2 MB).

---

### P3 — Lazy tab wiring: defer `wireKinaseCrosstable` and `wireFiveXFADKinase`

**Problem:** Both wire functions run at boot and iterate ~170k rows to build in-memory Maps. A user opening only the Kinase Explorer tab pays this cost unconditionally.

**Proposed change:** In `TAB_MANIFEST`, add an `initOnFirstRender` flag or make `wire()` a no-op that defers to a one-time `_ensureInit()` called inside `render()`. The pattern is already used in `kinase_fivexfad.js` for `_F5Wired` / `_F5ControlsPopulated` guards — unify this into the manifest contract.

Concretely: move `_initKinaseCrosstableData()` out of `wireKinaseCrosstable()` into a lazy init guard (`let _KX_INIT = false`) called from `renderKinaseCrosstable()`. Same for `wireFiveXFADKinase` → `_f5EnsureWired()`.

**Risk:** First render of these tabs will have an added ~20–50 ms initialization pause (visible as a flash). The fix is to show a "Preparing…" spinner on first render, then re-render. The alternative — keeping eager init — is O(boot) cost paid by every user regardless of which tabs they use.

**Features touched:** Crosstable, 5xFAD Kinase tab.

**Regression proof:** Tab functionality is identical after first render. Boot time drops by ~50–100 ms on cold start.

---

### P4 — Debounce the Incytr global index filter scan

**Problem:** `filterRank` scans 4.48 M typed-array rows on every slider movement or keystroke. With no debounce, rapid slider adjustment triggers 10–20 full scans per second. At 50–100 ms per scan, this can cause jank.

**Proposed change:** Wrap the `renderIncytrPathways()` call that follows filter state changes in a 100 ms `setTimeout` debounce (cancel-and-reset on each new event). The pattern is standard: `clearTimeout(_ipDebounce); _ipDebounce = setTimeout(render, 100)`. This does not change any feature — it only batches rapid user input.

**Risk:** Negligible. Users accustomed to immediate feedback will see a 100 ms lag. This is the standard trade-off. Keep debounce off for click events (sender/receiver chip selection); apply only to numeric sliders and text search.

**Features touched:** Incytr Pathways global mode filter responsiveness.

**Regression proof:** Filter results are identical; user test confirms no jank on slider drag.

---

### P5 — Move `gene_node_index` out of the inlined payload into a sidecar

**Problem:** 19.4 MB inlined in the payload, required only when the Incytr Pathways tab is opened in pair mode (gene search). This is the second-largest Incytr payload key after `heatmap_counts_signed` + `heatmap_counts` combined.

**Proposed change:** Write `gene_node_index` as a JSON.gz sidecar (`edge_slices/incytr_pathways/gene_node_index.json.gz`). Load it lazily via `SliceCache` on first pair-mode interaction that needs gene search. `IncytrGlobalIndex` already fetches the binary index lazily; `gene_node_index` can follow the same pattern.

**Risk:** Gene search in pair mode will show a "Loading…" state on first use. For the global index mode, gene search already requires the binary index to be loaded first. The sidecar is a single file, not per-pair. This is a moderate change touching `build_unified_viewer.py` (write sidecar), `payload_helpers.py` (remove from inlined block), and `incytr_pathways.js` (lazy load + re-render after load).

**Features touched:** Incytr Pathways gene search (pair mode).

**Regression proof:** Gene search returns identical results; payload gzip shrinks by ~1.9 MB (from ~10 MB to ~8.1 MB). Combined with P1–P2, total gzip target: ~4–5 MB.

---

### P6 — Cap the 5xFAD attribution and celltype MEA shard caches

**Problem:** `_F5AttrCache` and `_F5CelltypeMeaCache` are unbounded `Map` objects. A session clicking through all 390 kinases accumulates ~70 MB in-heap for attribution alone.

**Proposed change:** Apply the same LRU eviction pattern used in `SliceCache` to `_F5AttrCache` (cap at 32 entries) and `_F5CelltypeMeaCache` (cap at 32 entries). Implement as a local `_lruTouch` helper mirroring `SliceCache._lruTouch`.

**Risk:** A user clicking back to a previously-evicted kinase will trigger a re-fetch (flash of loading state). This is already the expected UX for edge slice parquets. Cap at 32 is conservative — most sessions navigate fewer than 32 kinases.

**Features touched:** 5xFAD Kinase tab (re-fetch on evicted kinases).

**Regression proof:** No feature change; memory footprint does not grow proportionally with session length.

---

### P7 — Convert `attribution_index` scan to a pre-built Map

**Problem:** `attribution_index` is a columnar dict (108,531 rows). Kinase audit reads it by iterating all rows on every kinase click. `kinase_explorer.js` `kinaseQualifies` similarly iterates for grid-filter qualification.

**Proposed change:** At tab wire time (or lazy init), build a `Map<kinase_id, row_indices[]>` over the attribution columnar arrays. This is a one-time O(n) scan that converts subsequent per-kinase lookups to O(1). The Map is ~108k entries pointing into the original typed arrays — no row object allocation needed.

**Risk:** The Map build adds ~20 ms to the init path (deferred to first kinase tab render if P3 is implemented). Subsequently, kinase selection and grid filter become significantly faster.

**Features touched:** Kinase Explorer attribution grid filter, Kinase Audit verdict panel.

**Regression proof:** Identical filter and audit results; kinase click latency measurably decreases.

---

### P8 — Switch T-cell viewer to sidecar payload mode

**Problem:** T-cell viewer inlines 18 MB of JSON in `index.html`, causing the browser to parse the full document before any JS loads. Adding a second donor would push inline payload to ~35 MB.

**Proposed change:** Adopt the same sidecar mode already used by the unified viewer: emit `tcell_viewer.payload.json.gz` alongside `index.html`; set `<script id="payload-data">null</script>`; browser fetches and gunzips on load. The shared `_loadPayload()` in `01_state.js` already handles both modes (the inline-null path falls through to the gzip fetch path).

**Risk:** The tcell viewer must be served over HTTP (same constraint already documented for the unified viewer). Air-gapped distribution would require `--inline-payload` flag (already supported). Build script change only.

**Features touched:** T-cell viewer delivery; air-gapped distribution mode unchanged.

**Regression proof:** T-cell viewer renders identically; `index.html` shrinks from 19 MB to ~750 KB.

---

## 6. Measurement and Benchmark Approach

Improvements must be measured, not asserted. Recommended benchmarks before and after each P-item:

### 6.1 Payload size

```bash
# Pre/post each P-item
wc -c outputs/reports/unified_viewer/unified_viewer.payload.json
wc -c outputs/reports/unified_viewer/unified_viewer.payload.json.gz
# T-cell
wc -c outputs/reports/tcell_viewer/index.html
wc -c outputs/reports/tcell_viewer/tcell_viewer.payload.json.gz 2>/dev/null || echo "sidecar not present"
```

Target after P1+P2+P5: gzip payload ≤ 6 MB (from 10 MB).

### 6.2 Cold-start parse time

Use the browser DevTools Performance timeline or `console.time / console.timeEnd` wrapped around `JSON.parse(text)` in `_loadPayload`. Baseline: measure on a mid-range machine (Chrome, no GPU acceleration).

```javascript
// Add temporarily to _loadPayload() for benchmarking:
const t0 = performance.now();
PAYLOAD = JSON.parse(text);
console.log("JSON.parse ms:", performance.now() - t0, "text bytes:", text.length);
```

Target: parse time ≤ 500 ms at 6 MB gzip / 60 MB uncompressed.

### 6.3 Boot-to-first-render time

DevTools → Performance → "Start profiling and reload page". Measure from `navigationStart` to the first `renderKinaseExplorer()` call. Baseline: record time before and after P3.

### 6.4 Filter latency (Incytr global mode)

In `filterRank()`, add `performance.now()` bookmarks around the main scan loop (temporarily). Record scan time at 4.48 M rows before debounce (P4) and confirm response time ≤ 200 ms from slider release to table update.

### 6.5 Memory footprint

Chrome DevTools → Memory → Heap Snapshot at three points: (a) after payload loaded, (b) after clicking through 50 kinases, (c) after opening Incytr Pathways and navigating 5 pairs. Compare heap before and after P6.

```javascript
// Spot-check in DevTools console:
performance.memory.usedJSHeapSize / 1e6 + " MB"
```

### 6.6 Edge slice fetch count and size

DevTools → Network → filter by `parquet` or `.json.gz`. Record number of fetches and total bytes transferred for a typical session (open kinase tab → select 3 kinases → open Incytr Pathways → navigate 3 pairs). This baseline documents what a "typical session" downloads beyond the initial payload.

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
- [ ] `pixi run python alz/viewer/verify_payload_contract.py` passes.
- [ ] `pixi run python alz/viewer/verify_template.py` passes.

---

## 8. Cohort-Growth Projection

| Scenario | Estimated raw payload | Estimated gzip |
|---|---|---|
| Current (Song + Mukesh + 5xFAD) | 104 MB | 10 MB |
| After P1+P2+P5 (shards moved out) | ~60 MB | ~6 MB |
| + 2nd AD mouse cohort at same scale | ~85 MB | ~8.5 MB |
| + T-cell folded into unified viewer | ~100 MB | ~10 MB |
| + 5xFAD at 8 timepoints (no P1/P2) | ~200 MB | ~20 MB (parse pain) |
| + 5xFAD at 8 timepoints (with P1/P2) | ~70 MB | ~7 MB (acceptable) |

The conclusion is that P1+P2 together are the highest-leverage changes. They address the 5xFAD data's dominant share of the payload without touching any other tab's UX. The remaining items (P3–P8) are incremental polish and future-proofing rather than immediate blockers.
