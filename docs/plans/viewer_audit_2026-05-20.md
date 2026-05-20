# Viewer Audit — 2026-05-20

Scope: `alz/build_unified_viewer.py`, `alz/integration/build_transcript_trace.py`,
`alz/viewer/template/js/**`, git history from first viewer commit through HEAD (`a152c52`).

---

## 1. Memory Hazards — Builder

### 1.1 Full-file loads without chunking

- **`build_unified_viewer.py:804`** — `pd.read_csv(stoich_path)` loads `stoichiometry_matrix.csv` (~13 MB on disk, ~40–60 MB in-memory wide frame with 72 numeric columns). No `usecols` filter. Used to build `stoich_by_site` list-of-lists that ends up in `PAYLOAD.human`. Fix: read only the donor + metadata columns needed.

- **`build_unified_viewer.py:805`** — `pd.read_csv(raw_path)` loads `raw_phospho_normalized.csv` (~14 MB on disk). Same issue. Both frames are kept alive in the `tracks` list until `build_human_slice()` returns.

- **`build_unified_viewer.py:797–809`** — `build_human_slice()` reads 10+ CSV files for each track (ST + pY) before entering any loop. Peak RSS contribution from this function alone is ~300–400 MB when both tracks have substrate data (the subs CSVs are 103.5 MB + 109.4 MB on disk).

- **`build_unified_viewer.py:1463`** — `pd.read_csv(src, usecols=lambda c: c in cols)` for `song_concordance.csv` (210 MB on disk). The `usecols` lambda is correct but the full file is still scanned row-by-row by the CSV reader before the lambda trims columns. The filtered frame (~30 MB) is then held in memory while shards are written. This is OK given the gene filter that follows, but the disk-to-RAM expansion is ~4×. Fix: convert song_concordance.csv to parquet upstream so the read is column-selective at block level.

- **`build_unified_viewer.py:929–951`** — `subs.iterrows()` over `mea_substrate_sets.csv` (~103.5 MB per track). `iterrows()` on a 100 MB CSV builds a Python dict per row; at ~400 K rows this is the dominant RSS spike in `build_human_slice`. Estimated peak: 800 MB–1.2 GB for both tracks. Fix: use `groupby("kinase") + groupby("contrast")` with vectorised aggregation instead of `iterrows`.

- **`build_unified_viewer.py:1387`** — `pq.read_table(DECOMP_OLS_PARQUET, columns=cols).to_pandas()` loads the 37.6 MB decomp-OLS parquet (3.77 M rows) fully into RAM as `pcdf`. The frame is then indexed (`pc_index = pcdf.set_index(…)`) creating a second copy. Estimated resident: ~350–500 MB. Fix: pre-filter by `contrast` and `track` before `.to_pandas()` using a PyArrow filter expression.

- **`build_unified_viewer.py:646`** — `pq.read_table(_decomp_path, columns=_decomp_read_cols).to_pandas()` loads `mea_per_cluster.parquet` fully. Size is small (< 5 MB), not a hazard.

- **`build_unified_viewer.py:633`** — `pd.read_csv(_ua_full_path)` for `unified_attribution_full.csv` (29 MB on disk). `usecols` list is specified so only 13 columns are read — this is OK.

### 1.2 Wide-to-long pivots on large frames

- **`build_unified_viewer.py:1642`** — `pivot_table(index=[_path_str, _disease], columns=_timepoint, …)` inside `_annotate_trajectory_columns()`. Called once per (sender, receiver) pair inside the per-pair loop. The pivot is over a small sub-frame (~30 K rows/pair avg), not the full table — acceptable as-is.

- **`build_transcript_trace.py:279–298`** — `np.tile(gene_cols_arr, n_rows)` + `np.repeat(sub_groups, n_genes)` + `sub_values.ravel()` expands each cluster's rows to `(n_rows × n_genes)` long. At 46 clusters × 24 rows × 25 K genes this is 46 × 24 × 25 K = 27.6 M total rows, but only **one cluster at a time** is materialised (~600 K rows per iteration, ~50 MB peak). The `del sub, long_*` calls at line 304 release each iteration's frame before the next. This is the already-mitigated path; no remaining hazard.

### 1.3 DuckDB: hardcoded `temp_directory`

- **`build_unified_viewer.py:1795`** and **`build_unified_viewer.py:2168`** — `SET temp_directory='/home/hchung/.cache/duckdb'` is hardcoded to a specific user path in **both** DuckDB connection sites. The CLAUDE.md note says `.envrc` sets `$DUCKDB_TEMP_DIR` but the builder ignores it. On any other machine (or CI), DuckDB spill lands in the default `/tmp`, which may be a tmpfs of <1 GB.
  - Recommended fix: `SET temp_directory='{os.environ.get("DUCKDB_TEMP_DIR", os.path.expanduser("~/.cache/duckdb"))}'`.

- **`alz/integration/build_transcript_trace.py`** — no DuckDB usage; not affected.

### 1.4 `CREATE TEMP TABLE src` materialises full receiver_cache into DuckDB RAM

- **`build_unified_viewer.py:1817–1828`** — `CREATE TEMP TABLE src AS SELECT … FROM read_parquet('{cache_glob}', …)` loads the entire 267 MB receiver_cache (reported 10.6 M rows) into DuckDB's in-process buffer pool. The `memory_limit` env-var cap (now 4 GB default) prevents OOM on an 8 GB box, but DuckDB still spills to `temp_directory` if queries push past the budget. This is already the post-mitigation path (per-pair streaming). The `CREATE TEMP TABLE` itself is the remaining pressure point: DuckDB ingest of 267 MB parquet peaks at ~800 MB–1.2 GB internal working set before the buffer pool stabilises.

### 1.5 `_write_incytr_pair_pathways` — unbounded `fetchdf()` of full src

- **`build_unified_viewer.py:2365`** — `con.execute(f"SELECT … FROM src").fetchdf()` pulls the **entire** pair-mode src table into a single Pandas DataFrame before per-pair groupby sharding. This is the old single-fetchdf anti-pattern, still present in the pair-mode code path. If pair-mode data ever grows to the size of the factorial cache (~10 M rows), this will OOM. Fix: apply the same per-pair streaming pattern from `_write_incytr_pathways()` (line 2000).

### 1.6 JSON payload size and `json.dumps` peak

- **`build_unified_viewer.py:2664`** — `json.dumps(payload, …)` of the full payload dict produces a 70 MB string (`unified_viewer.payload.json` is 70 MB on disk). The Python interpreter holds the source dict (~150 MB resident), the encoded bytes (~70 MB), and during `html.replace("__PAYLOAD_SENTINEL__", safe)` at line 2722, a second full HTML string (~140 MB) is created. Peak RSS from this step alone is ~400 MB. The 70 MB payload is then inlined verbatim into `index.html` (also 70 MB on disk). Browser parses the payload on load: `JSON.parse` of 70 MB at line 6 of `01_state.js` blocks the main thread for 200–800 ms and allocates ~250–350 MB JS heap.

- **Dominant contributor**: `PAYLOAD.human.perdonor_index.substrate_motifs` = 39 MB and `leading_substrates` = 11 MB (50 MB of the 70 MB total). These are per-(kinase, donor) motif strings that are only used when a specific kinase × donor panel is open. They should be sharded like decomp_ols slices rather than inlined in the payload. Fix: move `substrate_motifs` and `leading_substrates` to per-kinase parquet shards and fetch on demand.

### 1.7 Stale intermediate files from aborted runs

- `_write_decomp_ols_slices` clears stale parquets by individual `os.remove` (line 1356–1358) but does not use `shutil.rmtree` + re-create, so a failed run mid-write leaves partial shards alongside good ones.
- `_write_incytr_pathways` and `_write_incytr_pair_pathways` use the same individual-remove pattern (lines 1968–1970 and 2337). An aborted run leaves a mix of old and new parquets under `edge_slices/incytr_pathways/`, which confuses the pair-count in `index.json` written at the end.
- Song concordance and transcript_trace use `shutil.rmtree` — these are safe.

### 1.8 Incremental build gaps

- `--html` alone re-runs `load_all_data()` (reads all pipeline CSVs) but skips `build_payload()` (skips shard writes). This is correct.
- `--payload` always rebuilds **all** shard directories (decomp_ols, song_concordance, incytr_pathways, transcript_trace) unconditionally. There is no `--skip-decomp-ols` or `--skip-incytr` flag. On an 8 GB box a full `--payload` run peaks at ~2–3 GB RSS; the incytr step adds another ~1 GB DuckDB working set, bringing the total close to 4 GB.
- `ensure_measurement_trace_sources()` and `ensure_transcript_trace_sources()` have proper schema-version cache-hit guards (lines 316–320, 485–490) — these are reused correctly when the schema version matches.

---

## 2. Memory Hazards — Browser

### 2.1 Payload parse on page load

- **`js/01_state.js:6`** — `const PAYLOAD = JSON.parse(document.getElementById("payload-data").textContent)` — synchronous parse of a 70 MB string at page load. Allocates ~250–350 MB JS heap immediately, blocking the main thread for 200–800 ms on a modern desktop. No streaming or lazy parse.
  - Root cause: 39 MB `substrate_motifs` + 11 MB `leading_substrates` arrays inlined in the payload (see §1.6 above).

### 2.2 `_buildKinaseRowModel()` in `kinase_explorer.js`

- **`kinase_explorer.js:171–193`** — iterates `PAYLOAD.kinases` (all kinases, all 9 contrast NES/FDR columns) and builds `_keRows`, a JS object array holding references into the PAYLOAD arrays. The array is created once and held for the lifetime of the page. At ~240 kinases × 18 columns this is < 1 MB — not a hazard.

### 2.3 `_ensureKinaseIndexes()` maps built from full payload

- **`kinase_explorer.js:204–248`** — three Maps built from `PAYLOAD.kinase_celltype_evidence`, `PAYLOAD.decomposition_index`, `PAYLOAD.agreement_index`, `PAYLOAD.attribution_index`. These are built once and held in module-level variables. Total: ~12 MB attribution_index + 2.5 MB decomp + 0.6 MB evidence + 2.5 MB agreement ≈ 18 MB additional JS heap on first use. Bounded and acceptable.

### 2.4 `SliceCache` LRU eviction — correctly bounded

- **`js/04_slice_cache.js:10`** — `MAX = 16` per cache (backbone, decomp-ols, incytr, song concordance). LRU eviction via Map insertion-order delete. Each decomp-ols shard is ~0.5–2 MB; at MAX=16 the cache holds at most ~32 MB. Safe.

### 2.5 `TranscriptTraceStore` — **no eviction cap**

- **`js/widgets/transcript_trace.js:15`** — `const cache = new Map()` with no size limit and no eviction. Each cluster shard holds ~25 K gene × 24 group rows. At 46 clusters potentially loaded during a session (one per Incytr pathway row opened), the cache grows to 46 × rows, potentially 5–15 MB JS heap. At current scale this is unlikely to OOM a browser tab but it diverges from the `SliceCache` eviction discipline.
  - Fix: cap at MAX=8 entries using the same LRU pattern from `SliceCache._lruTouch`.

### 2.6 `MeasurementTraceStore` — **no eviction cap**

- **`js/01_state.js:324–351`** — `const cache = new Map()` with no eviction. One entry per `(residueType, sample)` key; each entry is a parsed CSV row array. At 33 males × 2 tracks this is up to 66 entries. Per-sample CSV files are ~100–500 KB. Bounded in practice (~30 MB worst case) but diverges from the eviction discipline.

### 2.7 `_khPerdonorFor()` linear scan

- **`kinase_human.js:80–95`** — `for (let i = 0; i < PI.kinase_id.length; i++)` scans the full 6,613-entry `perdonor_index` array on every kinase-selection or donor-selection event. At 6,613 rows this is fast (<1 ms) but scales poorly if the per-donor data grows (more donors, more kinases). Fix: build a `Map<kinaseId + "||" + donor, record>` index on first call.

### 2.8 `Promise.all` over transcript-trace elements

- **`incytr_pathways.js:801`** — `await Promise.all(elements.map(async (el, i) => { … }))` fires N concurrent `TranscriptTraceStore.values()` calls, where N = 4 (L, R, EM, T nodes per row). Each `values()` call may trigger a shard fetch. At N=4 parallel fetches per panel open, browser concurrency limit (6) won't be saturated. Not a hazard at current scale; acceptable.

### 2.9 `tbody.innerHTML = parts.join("")` full-table renders

- **`kinase_explorer.js:722`** and **`kinase_human.js:352`** — full table body serialised as one string and set via `innerHTML`. At 240 kinases with 12 columns each the string is ~50 KB — within browser parsing budget. Not a hazard.

### 2.10 `temporal_v2.js` — `JSON.parse/JSON.stringify` on state clone

- **`temporal_v2.js:746`** — `_state = JSON.parse(JSON.stringify(_defaults))` deep-clones the default state on reset. State object is small (< 1 KB). Not a hazard.

---

## 3. Features adopted from CR-03 branch via the reconcile epic

> **Reframed 2026-05-20 after the CR-03 adoption.** This section originally documented features as "regressions / lost in HEAD `a152c52`." That framing was correct relative to the pre-adoption main, but main has since been hard-reset onto `feat/cr03-human-celltype-specificity` and the four cherry-picks landed in Phases 2–5 of [`docs/plans/epic_reconcile_cr03_branch.md`](epic_reconcile_cr03_branch.md). Final Phase 5 SHA: `534f98e` (per-cluster slicing for transcript trace builder). The items below are now **features adopted from the CR-03 branch**, not regressions. The historical detail is retained because the §4 punch list still references them and the diff narrative is useful audit context.

### 3.1 Kinase Evidence Crosstable tab — restored on the branch; needs final TAB_MANIFEST wiring (see Phase 7)

- **Introduced**: commit `175c85b` (2026-05-19) — "feat(viewer): add Kinase Evidence Crosstable + Family column on mouse/human tabs"
- **Deleted**: commit `a152c52` (2026-05-20, HEAD) — "feat(viewer): Incytr pathways Measurement Trace panel (transcript v1)"
- **What it did**: `alz/viewer/template/js/tabs/kinase_crosstable.js` (564 lines). A wide one-row-per-kinase table joining mouse bulk MEA (9 contrasts × NES), per-cluster decomp NES (Levy-T5), WMB/SEA-AD/HBCA specificity, and per-donor human NES into a single sortable grid. Column-group visibility panel controlled breadth. Joins were client-side from existing PAYLOAD keys; no new shard or payload field was required.
- **What was removed alongside it**:
  - `alz/viewer/template/body.html`: `<div id="tab-crosstable" …>` panel and the full crosstable toolbar HTML (filters, table, visibility toggles)
  - `alz/viewer/template/index.html.j2`: `{{ raw('js/tabs/kinase_crosstable.js') }}` line removed
  - `alz/viewer/template/js/02_ui_chrome.js`: no TAB_MANIFEST entry was added for `crosstable` before deletion (git diff 175c85b → a152c52 shows no TAB_MANIFEST change), but the `body.html` panel div was present — this means the tab was **not yet wired** into the tab bar before it was deleted.
  - `alz/viewer/template/body.html`: "Family" column header `<th>` removed from both the Mouse Kinase Explorer table and the Human Kinase table headers.
- **Intentionality**: The deletion was part of the Measurement Trace commit's diff — the commit message says nothing about removing the crosstable. There is no corresponding "remove crosstable" commit or PR. The `kinase_crosstable.js` file was likely deleted as a workspace side-effect (possibly `git restore --staged` or bulk reset during a rebase) rather than intentional removal.
- **Restore path**: `git show 175c85b:alz/viewer/template/js/tabs/kinase_crosstable.js > alz/viewer/template/js/tabs/kinase_crosstable.js`, then restore the `body.html` tab panel div and the `index.html.j2` include line, and add a `crosstable` entry to `TAB_MANIFEST` (was missing before deletion — the tab was not yet wired).

### 3.2 Human Kinase tab — Cell-type Specificity sub-tab deleted

- **Introduced**: commit `0ae69f0` (2026-05-15) — "feat(cr03): add Cell-type specificity sub-tab to kinase_human.js"
- **Deleted**: commit `a152c52` (HEAD)
- **What it did**: Fourth sub-tab `{id: "celltype", label: "Cell-type specificity"}` in the Human Kinase detail panel. `_khRenderCelltypeSpecificity()` rendered two stacked mini-tables (SEA-AD MTG | Allen HBCA), each showing top-8 cell types by log2-specificity for the selected kinase. Populated from `PAYLOAD.human.celltype_specificity` (built by `alz/human_celltype_attribution.py`).
- **Companion deletions**: `alz/human_celltype_attribution.py` and `alz/human_reference_expression.py` were also deleted in the same inter-commit diff (`175c85b` → `a152c52`). `PAYLOAD.human.celltype_specificity` is no longer built.
- **Intentionality**: Ambiguous. `human_celltype_attribution.py` and `human_reference_expression.py` require `atlas_reference.py --hbca-download` (HBCA data); the HBCA runner script `run_hbca_download.sh` was also deleted in the same diff. This pattern suggests a deliberate removal of the HBCA-dependent features, but the commit message makes no mention of it. The payload key `celltype_specificity` would need to be rebuilt to restore this tab.

### 3.3 Human Kinase tab — Attribution sub-tab deleted

- **Present in**: commit `175c85b` — `{id: "attribution", label: "Attribution"}` as a fourth sub-tab
- **Deleted in**: `a152c52` (HEAD)
- **What it did**: `_khRenderAttribution()` collected ranked Levy-T5 evidence from `PAYLOAD.human.celltype_specificity` (SEA-AD MTG + HBCA references), computed a combined score (specificity × concordance), and rendered a per-cell-type table alongside the per-donor NES. Depended on `celltype_specificity` payload key (see §3.2).
- **Intentionality**: Consequential removal — `_khRenderAttribution` depended on `celltype_specificity`, so deleting that key made this sub-tab non-functional. Removal was correct given the dependency was dropped, but again unmentioned in the commit message.

### 3.4 Human Kinase tab — Cell type / Confidence / Specificity filters deleted from body.html

- **Commit `a152c52` removed from `body.html`**:
  - `<label id="kh-filter-celltype">` — "Require at least one high/moderate attribution row in the selected human reference cell type"
  - `<label id="kh-filter-confidence">` — minimum attribution confidence tier filter (high/moderate/low)
  - `<label id="kh-filter-specificity">` — minimum human reference specificity tier filter (≥1×/≥2×/≥5×/≥10×)
  - From the human table header: `n_attributed_celltypes`, `max_specificity_tier`, `conf` columns
- **Intentionality**: These filters drove from `celltype_specificity` payload data (see §3.2); their removal is logically consistent with the deletion of that payload block. Intentional.

### 3.5 Mouse Kinase Explorer and Human Kinase tables — "Family" column deleted

- **Introduced in**: `175c85b` — "Family column added to the existing mouse Kinase Viewer and Human Kinase tab so the META familyMap surfaces alongside name/gene/residue."
- **Deleted in**: `a152c52` (HEAD) from `body.html` table headers (both `<th data-col="family">` entries removed)
- **Note**: `kinase_explorer.js:182` still reads `family: famMap[K.name[i]] || ""` into `_keRows`, and `kinase_crosstable.js` (now deleted) also used it. The Family column was removed from the rendered headers but the underlying data is still in `_keRows`. Restoring the header `<th>` would re-surface the column without any builder change.

### 3.6 `MANIFEST.md` — out of sync with current tab set

- **`alz/viewer/template/MANIFEST.md`** still lists `kinase_human.js` with a dash for line count ("—") and says the pathway/overview/sender-matrix tabs were deleted but does not mention the crosstable tab that was added and then removed. The MANIFEST was last accurate at `175c85b`.

---

## 4. Punch List

### P0 — Data loss / broken features

1. **Restore `kinase_crosstable.js`** (§3.1). Added yesterday, deleted today in an unrelated commit. Restore from `git show 175c85b:alz/viewer/template/js/tabs/kinase_crosstable.js`, restore `body.html` panel, restore `index.html.j2` include, add `TAB_MANIFEST` entry. Also restore the "Family" `<th>` column to both Mouse Kinase and Human Kinase table headers (§3.5).

2. **Fix pair-mode `fetchdf()` of full src table** (`build_unified_viewer.py:2365`, §1.5). The pair-mode writer does a single `SELECT * FROM src` into a Pandas DataFrame (same pattern the factorial writer was fixed for). Apply per-pair streaming: after `CREATE TEMP TABLE src`, iterate `DISTINCT sender, receiver` pairs and `fetchdf()` one pair at a time, mirroring lines 1981–2033.

3. **Fix hardcoded DuckDB `temp_directory`** (`build_unified_viewer.py:1795` and `2168`, §1.3). Replace with `os.environ.get("DUCKDB_TEMP_DIR", os.path.expanduser("~/.cache/duckdb"))`. This makes the builder portable and respects the `.envrc` convention documented in CLAUDE.md.

### P1 — Memory hazards with clear fixes

4. **Move `substrate_motifs` + `leading_substrates` to per-kinase shards** (§1.6, §2.1). These two fields alone account for 50 MB of the 70 MB payload and 39 MB + 11 MB of the JS heap on parse. Shard to `edge_slices/human_perdonor/{kinase_id:03d}.parquet`, fetch on demand from the per-donor GSEA drawer. Payload drops to ~20 MB; JS parse time drops by ~70%.

5. **Replace `subs.iterrows()` with vectorised groupby** (`build_unified_viewer.py:944–951`, §1.3). `iterrows()` on a 100 MB CSV at ~400 K rows builds Python objects per row; this is the dominant builder RSS spike. `subs.groupby(["kinase", "contrast"])["motif"].agg(list).to_dict()` produces the same lookup in a single vectorised pass.

6. **Add LRU cap to `TranscriptTraceStore`** (`js/widgets/transcript_trace.js:15`, §2.5). Copy the `_lruTouch` pattern from `SliceCache`; cap at `MAX = 8` cluster entries. Without this the transcript-trace cache accumulates all opened clusters for the lifetime of the page, diverging from the bounded pattern everywhere else.

### P2 — Lower-priority quality / portability

7. **Shard-clear robustness** (`build_unified_viewer.py:1353–1358`, §1.7). Replace the per-file `os.remove` loop for `decomp_ols` and both incytr shard dirs with `shutil.rmtree(dir); os.makedirs(dir)` to prevent partial-write corruption on aborted runs.

8. **Decomp-OLS parquet: push column filter into PyArrow before `.to_pandas()`** (`build_unified_viewer.py:1387–1388`, §1.1). Add `filters=[("contrast", "in", list(contrast_to_id.keys()))]` to `pq.read_table()` to avoid loading off-scope contrast rows. Estimated savings: 30–50% of the 37.6 MB parquet depending on which contrasts are active.

9. **Index `_khPerdonorFor` lookups** (`kinase_human.js:80–95`, §2.7). Pre-build `Map<kinaseId+"||"+donor, index>` on first call to `_khPerdonorFor`. Currently O(6613) scan per call; with the map it is O(1).

10. **Add LRU cap to `MeasurementTraceStore`** (`js/01_state.js:324–351`, §2.6). Current unbounded cache grows to at most ~30 MB across 66 entries; not urgent but inconsistent with eviction discipline.

11. **Update `MANIFEST.md`** (`alz/viewer/template/MANIFEST.md`, §3.6). After restoring the crosstable (item 1), update MANIFEST to list `kinase_crosstable.js` and correct the `kinase_human.js` line count. Record the HBCA-dependent feature removals.

12. **Document intentionality of 3.2 / 3.3 / 3.4 in commit history**. The deletion of `human_celltype_attribution.py`, the attribution sub-tab, and the three filter controls was not mentioned in `a152c52`'s commit message. A follow-up `fix(viewer)` commit or PR note should confirm these were deliberate scope reductions (HBCA dependency dropped) vs. accidental.
