# Plan: Replace Incytr top-5000 pre-cap with a complete global filter-index

## Context

The Incytr Pathways tab's "Top overall" mode ships a **pre-capped** payload block
(`top_instances`, the 5000 highest-|PDS| paths, built by `_build_top_instances()` in
`alz/build_unified_viewer.py:2745`). All client-side filter levers (disease, timepoint,
pvalue, |PDS|, search, trajectory, sparse-cells) then sift **within those 5000 rows**.

Observed failure: select "Top 1000", enable the Sparse-cells filter (`excludeLowSignalCelltypes`,
median male pseudobulk n_cells ≤ 3) → only a handful of rows. Root cause: the 5000 are ranked
by |PDS| alone, and sparse cell types inflate |PDS| (tiny pseudobulk denominators), so they
dominate the top slots; the QC filter then guts the set. The cap is acting as a
**data-availability gate**, not a render budget. The bug class is general — *any* filter whose
survivors are under-represented in the |PDS|-top-5000 collapses (a rare sender, a single
contrast, a high-pvalue band). The within-render JS order is already correct (filter precedes
the slice); the defect is entirely the build-time pre-cap, and a true recompute is impossible
client-side because the rest of the universe is never shipped.

**Intended outcome:** the cap becomes the *terminal* step of `filter → rank → limit`, evaluated
over the **complete** universe (4,480,480 paths = 9 contrasts × ~961 pairs; the wide parquets are
already at the Allpathway floor, so |PDS|≥0.2 cuts nothing). "Top N of paths passing my filters"
becomes literally true for every filter combination.

## Approach (decided with user)

In-browser **binary columnar global index** — no new runtime dependency. Ship every path's
filter/sort columns once as a compact packed binary; the client maps it into TypedArrays and runs
`filter → rank → slice(N)`. `N` (500/1000/5000) becomes pure pagination. Rejected: DuckDB-WASM
(ships the same data *plus* a ~10–15 MB engine — strictly larger download, and adds a dependency);
fast-paint hybrid (renders a known-wrong set first — misleading, violates honesty rule).

Decisions: **all 5 score columns included** at float16 (full-universe-correct score sorting);
gene ids **uint16** (5,984 distinct genes); cell-type/contrast ids **uint8** (31 / 9). Per the
anti-shim rule this **replaces** `top_instances` outright — no flag, no fallback.

## Binary index format

One file `edge_slices/incytr_pathways/incytr_index.bin` (+ gzip `.bin.gz`), sibling of the
existing shards. Rows **pre-sorted by ABS(PDS) DESC at build time** ⇒ global rank = row position
(no rank column stored). Layout = concatenation of these little-endian columns, each length
`N = 4,480,480`:

| col | type | bytes | role |
|---|---|---|---|
| senderId | u8 | N | filter (sparse via median_n lookup) + search + sort |
| receiverId | u8 | N | " |
| contrastId | u8 | N | filter (disease/timepoint) + sort |
| ligandId, receptorId, emId, targetId | u16 ×4 | 8N | search + sort (genes) |
| labelBits | u8 | N | DEG/prG/none badge, 2 bits × 4 nodes (display) |
| trajBits | u8 | N | trend filter + trajectory badges (bitmask over traj-label vocab) |
| PDS | f32 | 4N | rank key + |PDS| filter + display |
| pvalue | f16-bits (u16) | 2N | pvalue filter (non-negative ⇒ raw-bit compare is monotonic) + display |
| TPDS, PPDS, PhPDS_ps, PhPDS_py, SiK_score | f16-bits (u16) ×5 | 10N | score sort + display |

Resident ≈ N × 31 B ≈ **140 MB** TypedArrays. Float16 stored as raw `u16` bit patterns, decoded
to f32 only when sorting by / displaying that column. Wire ≈ 30–50 MB gz (floats dominate; matches
the measured 83 MB full-precision → ~halved at f16).

**Manifest** in payload block as `block.global_index` (replaces `top_instances`):
`{ url, nrows, rank_by:"abs(PDS)", celltype_vocab:[≤31], contrast_vocab:[9], gene_vocab:[5984],
traj_label_vocab:[...], label_states:["DEG","prG"], score_columns:[...], columns:[{name,type}…] }`.
Also register `url` in `edge_slice_ref`. Cell-type median_n stays in the existing
`block.celltype_qc.by_celltype` (already shipped) — the sparse filter derives membership client-side
from sender/receiver id → name → median_n, so its **threshold is live**, not a frozen boolean.

## Build-side changes — `alz/build_unified_viewer.py`

Build the index **from the already-written shards**, as a post-pass after the shard loop in
`_write_incytr_pair_pathways()` (the shards already carry `traj_labels`, the `*_label` columns, and
float16 scores; sender/receiver come from `present_pairs`/filename, Path from L|R|E|T). This reuses
existing per-shard trajectory annotation and **bounds memory** — process one shard at a time, never
materialize 4.48M rows in pandas (CLAUDE.md memory rule).

- `total_n = sum(pair_row_counts)` (known from `slice_index`). Preallocate the output numpy column
  arrays at `total_n`.
- For each shard: pyarrow column read of `{Ligand,Receptor,EM,Target,contrast,PDS,pvalue,traj_labels,
  *_label, <scores>}`; encode strings → ids (gene dict built incrementally; contrast/celltype via the
  canonical `present_contrasts` / `senders_canonical` ∪ `receivers_canonical`); pack labelBits +
  trajBits; write float16 score/pvalue bit patterns; append at running offset; stash PDS for the sort.
- After all shards: `perm = argsort(-abs(PDS))` (4.48M f32 — fast); apply `perm` to every column;
  write columns concatenated → `incytr_index.bin`; gzip → `.bin.gz` (reuse the payload gz path).
- Emit `block.global_index` manifest + `edge_slice_ref` entry.
- **Delete** `_build_top_instances()`, the `_INCYTR_TOP_INSTANCE_LIMIT = 5000` constant
  (`:2142`), and the `"top_instances": top_instances` payload key (`:3071`). Update the
  `_build_top_instances` docstring/comments that describe the "small rank-and-drilldown surface".

## Client-side changes — `alz/viewer_shared/template/js/`

**New module `tabs/incytr_global_index.js`** (`IncytrGlobalIndex`):
- `ensureLoaded()` — fetch `block.global_index.url` (.bin.gz) → `DecompressionStream('gzip')` →
  ArrayBuffer; slice TypedArray views per `manifest.columns` offsets; build id→string lookups and
  a `f16bitsToF32` decoder; module-level cache. Returns a promise.
- `filterRank(f)` — integer-space scan returning `{indices, total}`:
  - Resolve filter inputs to integer sets **once**: disease/timepoint → allowed `contrastId` set
    (decode each contrast → `dis_tp`); sparse → `celltypeId` set with `median_n ≤ thr`; search tokens →
    per-token sets of matching geneIds + matching celltype/contrast ids (AND across tokens, OR across
    fields); trend → bit position in `trajBits`.
  - Linear scan `i ∈ 0..N`: `|PDS[i]|≥sPds`; `pvalue[i] < thrBits`; `contrastId ∈ allowed`;
    `senderId,receiverId ∉ lowSignal`; gene/celltype id ∈ token sets; `trajBits[i] & trendBit`.
    Collect passing `i`.
  - **Default sort = rank** ⇒ passing indices already in |PDS| order; take first `topLimit`, no sort.
  - **Non-default sort** ⇒ bounded heap of size `topLimit` keyed by the sort column (decode that one
    f16 score column to a cached Float32Array on demand; string keys compare via precomputed
    vocab→sortRank). O(n log topLimit), avoids a full 4.48M sort.
- `materialize(idx)` — build a display row object from the columns (decode ids→strings, f16→f32,
  labelBits→badges, trajBits→labels). Used for the ≤100 visible rows only.

**Rewire `tabs/incytr_pathways.js`** (top-mode path only; pair-mode untouched):
- `_ipTopRowsFiltered` (`:527`): for top mode, delegate to `IncytrGlobalIndex.filterRank`; return
  indices + total instead of scanning `top.rows`.
- `_ipRenderTopTable` (`:589`): `await IncytrGlobalIndex.ensureLoaded()` (loading state on first
  open); page-slice the indices; `materialize` the visible page. **All table columns + evidence
  badges render from the index** (labelBits) — no per-row shard fetch. The expanded-detail panel
  (FC values, trajectory chart) keeps fetching the pair shard on row-expand via the existing
  `SliceCache.loadIncytrShard(sender,receiver)` path.
- Count line (`:635`): `matching` = `total` (true universe count). Show "all Y matches shown" when
  `Y ≤ topLimit`, else "top N of Y matching — raise the cap or narrow filters to see more". Drop the
  "packaged" wording. This is the honesty safeguard: a collapse now reads as "only Y exist," never as
  silent truncation.
- `_ipEnsureShards` top-branch (`:806`): ensure global index loaded before render.
- Rewire all `block.top_instances` / `top.rows` / `top.rank_by` reads (`:529-530, 593-594, 637`) to
  the index module. Keep `_ipNormalizeTopRow`, `_ipScoreCols`, `_ipDecodeLabels`, `_ipHasTraj`,
  pagination (`_IP_PAGE_SIZE=100`), and the 500/1000/5000 selector — the selector is now a true
  display budget; refresh its tooltip ("rows to display"; no longer a packaged-set cap).

## Verification (end-to-end)

1. **Build:** `pixi run viewer`. Confirm `incytr_index.bin.gz` emitted (~30–50 MB), `block.global_index`
   manifest present, `top_instances` gone. `PAYLOAD.meta.generated_at` fresh; hard-refresh.
2. **Bug repro (primary gate):** top mode → "Top 1000" → enable Sparse cells. Before: a handful.
   After: up to 1000 real paths (or all matches if fewer), count line reports the true universe count.
   Confirm a known 1-cell pair (Cholinergic at 2mo) drops out while dense pairs remain.
3. **SQL ground-truth cross-check:** for ~5 filter combos (incl. sparse, single-contrast, gene
   search, score-column sort), compute the expected top-N in DuckDB over the wide parquets
   (`WHERE … ORDER BY abs(PDS) DESC LIMIT N`) and diff against the viewer's displayed
   sender/receiver/Path/contrast/PDS. Exact match required.
4. **Performance:** default-sort filter < ~100 ms; score-sort (heap) < ~300 ms on 4.48M; debounce the
   search input. Confirm resident TypedArrays ≈ 140 MB (DevTools memory), no per-object blowup.
5. No unit suite (per CLAUDE.md); the SQL cross-check is the correctness harness.

## Risk notes

- **Build memory:** the shard-by-shard accumulation + argsort is the only large step; keep output
  arrays preallocated and never concat shards in pandas. Peak ≈ 160 MB Python.
- **First-open latency:** top mode now requires the ~30–50 MB index fetch+decode before first paint
  (vs instant from the old inlined 5000). Acceptable with a loading state; if it proves bad, a
  background-load progressive enhancement is a *follow-up*, not part of this change (and must never
  paint a known-wrong set — the rejected fast-paint).
- **f16 PDS ties:** ranking order is locked by the build-time full-precision DuckDB/argsort and
  encoded as row position; the stored f16 PDS is display/threshold only, so precision loss can't
  reorder ranks.
