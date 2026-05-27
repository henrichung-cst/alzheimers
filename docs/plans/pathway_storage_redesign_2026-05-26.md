# Design Memo: Pathway Storage Redesign

**Date:** 2026-05-26  
**Status:** Research only — no code changed  
**Context:** Two OOMs in the viewer build pipeline (now fixed); evaluation of whether the wide-row-per-path storage is still the right long-term design.

---

## 1. Current State

Two storage layers exist:

| Layer | Location | Rows | Size | Purpose |
|---|---|---|---|---|
| Source wide parquets | `wide/` | 57M | 3.81 GB | Significance-filtered (SigProb>0.1 & \|PDS\|≥0.2) input to viewer build |
| Unfiltered archive | `wide_nboot0/` | 181M | 9.8 GB | Retained for audit; not consumed by viewer |
| Edge shards | `edge_slices/incytr_pathways/` | 42M | 543 MB | 961 parquets, one per (sender, receiver), fetched by JS on pair click |

The viewer build (`_write_incytr_pair_pathways`) reads the `wide/` parquets via a DuckDB VIEW, computes two fixed-size aggregation cubes (heatmap_counts, pathway_counts), then emits the 961 shards in one streaming pass per sender (31 passes × one ORDER BY receiver sort each). The two OOMs that triggered this review were caused by an earlier architecture (110 GB DuckDB spill from a reshape attempt, and 8 GB per-sender materialization from per-pair queries); the current streaming implementation avoids both. The build is no longer OOM-prone on the current inputs.

---

## 2. Redundancy Quantification

### 2a. Column-level redundancy in the current shards

Each edge shard row has 41 columns. Measured on the largest shard (431K rows):

- **FC columns** (16 × float16: 4 nodes × 4 metrics) + 4 label columns: **52% of compressed shard bytes**
- Scores, gene strings, contrast, traj columns: remaining 48%

Stripping FCs and labels from the shards would reduce 543 MB → ~261 MB for the fact portion.

### 2b. Node-FC uniqueness structure

Measured on AppP_2mo (20.9M rows), confirmed across other contrasts:

| Column group | Unique key | Distinct entries (one contrast) |
|---|---|---|
| `Ligand_sclog2FC` | (Ligand, Sender, **Receiver**, contrast) | ~27,500 |
| `Ligand_{pr,ps,py}_log2FC` | (Ligand, Sender, contrast) | ~2,024 |
| `Receptor_sclog2FC` | (Receptor, Receiver, contrast) | ~1,570 |
| `EM_sclog2FC` | (EM, Receiver, contrast) | ~1,919 |
| `Target_{all}_log2FC` | (Target, Receiver, contrast) | ~30,200 |

**Critical finding:** `Ligand_sclog2FC` is unique per (Ligand, Sender, **Receiver**) — not just (Ligand, Sender). The scRNA FC is computed per-pair by `Cal_scFC` with pair-specific gene subsets. It is NOT a pure (gene, celltype) quantity. The other three metrics (pr/ps/py) for Ligand, and all four metrics for Receptor/EM/Target, are uniquely determined by (gene, receiving celltype, contrast).

Within a single edge shard (sender and receiver fixed), `Ligand_sclog2FC` reduces to unique per (Ligand, contrast) since sender and receiver are constant. A per-shard node-FC sidecar for one pair would be ~23K rows vs 431K shard rows (18.7× reduction in FC entries per shard).

**Across all 9 contrasts, a global node-attribute table** (union of all unique (gene, celltype_role, contrast) entries) would have approximately 820K rows. At ~40 bytes per row (strings + four float16s + label), that is ~33 MB uncompressed, ~8–12 MB compressed.

### 2c. Path topology: NOT contrast-independent

Distinct LRET (Ligand|Receptor|EM|Target) tuples per contrast range from 1.9M to 5.8M, and differ across contrasts because gene.use (DEG ∪ prG) is computed per (cluster, contrast). The union across all 9 contrasts is 9.6M distinct LRET tuples. A shared path-topology table would contain these 9.6M entries, not the 3M assumed when the "contrast-independent topology" hypothesis was first raised. If partitioned by contrast it remains useful but not shared.

### 2d. PDS/TPDS/PPDS derivability

From `evaluation.R` and `math.R`:

```
TPDS = logi(aFC, k=2)          ← aFC = logi(SigProb_cond1/SigProb_cond2)
PPDS = mean(logi([L_pr_aFC, R_pr_aFC, EM_pr_aFC, T_pr_aFC], k=2))
PhPDS_ps = mean(logi([L_ps_aFC, R_ps_aFC, EM_ps_aFC, T_ps_aFC], k=2))
multimodel_score = TPDS + 0.5*(PPDS + PhPDS_ps + PhPDS_py + Ack + KGG + Rme1)
PDS = multimodel_score + SiK_directional_term
```

Verified numerically: `multimodel_score = TPDS + 0.5*(PPDS + PhPDS_ps + PhPDS_py + Ack + KGG + Rme1)` with zero residual. `PDS - multimodel_score` averages 0.0007, which is the SiK directional term.

**TPDS is NOT recomputable from node FCs.** It is `logi(SigProb_aFC)` where `SigProb_aFC` is a path-level Hill-function product of expression values across all four nodes simultaneously — a genuine path×pair×contrast quantity.

**PPDS/PhPDS ARE approximately recomputable from the stored pr/ps/py node FCs** (the node `log2FC` columns closely approximate `aFC` for the pr/ps/py omics layers since these are dense protein values; Hill-shrinkage is small). However: the shards currently store `pr_log2FC`, not `pr_aFC`, and the approximation is not exact. Additionally, the stored PPDS values in current shards are float16, which zeros out values below ~6e-5 (verified: some PPDS values fall in this range from source data with min nonzero |PPDS| ≈ 1.8e-5).

**Conclusion on derivability:** PDS is the ranking column (CLAUDE.md: filter/rank on |PDS|) and is already stored in every shard row as float16. Dropping PPDS/PhPDS from the shard to force recomputation would (a) require transmitting pr_aFC values (not currently stored), (b) introduce float16 precision discrepancies vs. what Incytr computed, and (c) complicate the JS evidence panel. Not recommended.

---

## 3. Candidate Representations

### Option A: Star schema (normalized relational parquet)

**Schema:**
- `path_topology_{contrast}.parquet`: (path_id int32, Ligand, Receptor, EM, Target) — one file per contrast, 1.9M–5.8M rows
- `node_attrs.parquet`: (gene, role, celltype, contrast, sclog2FC, pr_log2FC, ps_log2FC, py_log2FC, label) — 820K rows total
- `fact_{sender}_{receiver}.parquet`: (path_id, contrast, PDS, TPDS, PPDS, PhPDS_ps, PhPDS_py, SiK_score, pvalue, traj_labels, sign_vec) — 961 files, 42M rows total

**Storage estimate:**
- Fact shards: ~42M rows × 11 score floats (float16) + 2 string cols + path_id (int32) ≈ ~200–230 MB
- Path topology (9 per-contrast files, 9.6M union rows): strings at ~32 bytes compressed ≈ 60–80 MB
- Node attrs: 820K rows × ~40 bytes ≈ 8–12 MB compressed
- Total: **~270–320 MB** vs 543 MB current

**Viewer read path:** On pair click, JS fetches the fact shard (unchanged fetch pattern), then must JOIN or lookup LRET strings from the path topology files (which contrast? which file?) and node attrs. The path-topology JOIN is non-trivial: the shard stores a path_id that references a per-contrast file; if the viewer needs to display Ligand/Receptor/EM/Target strings and FC values for each row, it must either (a) pre-load all topology files (60–80 MB on page load) or (b) do a second fetch per pair click. Either way, the current zero-join on-click experience degrades. The heatmap cube computation at build time is unchanged.

**Build memory:** Lower peak than current because the fact columns are fewer. But the normalization join at build time (creating path_ids by hashing LRET strings per contrast) is a new DuckDB operation that must be consistent across all 9 parquets.

**Verdict:** 40–50% disk reduction. Not compelling given the viewer-fetch complexity added. The star schema benefits are primarily for aggregation queries (SQL GROUP BY on path_id), which the viewer does not do — it loads one shard and filters in JS.

### Option B: Per-pair shards stripped of FC columns + global node-FC sidecar (recommended)

**Schema:**
- `edge_slices/incytr_pathways/{sender}__{receiver}.parquet`: Ligand, Receptor, EM, Target, contrast, pvalue, PDS, TPDS, PPDS, PhPDS_ps, PhPDS_py, SiK_score, traj_labels, sign_vec — 41 → **20 columns**
- `edge_slices/incytr_pathways/node_fc.parquet` (or inlined to viewer JSON payload): (gene, role, celltype, contrast, sclog2FC, pr_log2FC, ps_log2FC, py_log2FC, label) — 820K rows, ~10 MB

**Storage estimate:**
- Fact shards (21 cols stripped of 16 FCs + 4 labels): ~261 MB (52% reduction verified on largest shard)
- Node FC sidecar: ~10 MB compressed
- Total: **~271 MB** — 2× reduction vs 543 MB current

**Viewer read path:**
1. On pair click: fetch shard (same as today, half the size)
2. JS builds a lookup Map from node_fc: `Map<gene+'|'+role+'|'+contrast, {sclog2FC, pr_log2FC, ...}>` — built once at page load from the 10 MB sidecar
3. For each shard row, evidence panel looks up FCs from the Map — O(1) per node per row

The heatmap cube, pathway_counts cube, traj_labels/sign_vec, PDS ranking, and filtering are completely unchanged. The evidence panel currently reads FC values directly from `r[fc_col]`; it would instead do `fcMap.get(r.Ligand + '|Ligand|' + r.contrast)`. Change is confined to `EvidencePanel.render()` and the shard-load precomputation in `incytr_pathways.js`.

**Key complication:** `Ligand_sclog2FC` is unique per (Ligand, Sender, Receiver, contrast), not just (Ligand, contrast). Since the shard has a fixed (Sender, Receiver), the lookup key for `Ligand_sclog2FC` within a shard is simply (Ligand, contrast) — no ambiguity. But the global `node_fc.parquet` must include the Sender dimension for Ligand sclog2FC, stored as (gene='Jag2', role='Ligand', sender='GluN', receiver='ChN', contrast='App_2mo', sclog2FC=2.31, ...). The full key is (gene, role, sender, receiver, contrast) for Ligand_sclog2FC, and (gene, role, receiver, contrast) for all others. This is awkward in a single flat table but manageable with a nullable sender column (NULL for Receptor/EM/Target rows).

**Build change:** The builder strips FC columns before writing shards and emits node_fc.parquet as a separate artifact. One DuckDB query per contrast to extract distinct (gene, role, celltype, sclog2FC, pr_log2FC, ...) tuples, UNION ALL across contrasts, dedup, write. Straightforward. Build memory is reduced since shard writes are smaller.

**Verdict:** 2× disk reduction, minimal viewer complexity increase (one sidecar fetch at page load, one Map lookup per evidence panel row). Preserves PDS ranking, traj_labels, all score columns. Recommended.

### Option C: Embedded graph (nodes = gene×celltype×contrast, edges = L→R→EM→T)

**Concept:** Store the pathway data as a graph database (e.g., Kuzu, DuckDB with adjacency tables, or a custom adjacency list). Nodes carry FC attributes; edges carry SigProb and path scores. The path topology becomes graph structure.

**Analysis:** This is appropriate when the primary queries are graph traversals (e.g., "what paths connect gene A to gene B across any celltype?"). The viewer's actual query is "give me all scored paths for (sender, receiver), ranked by |PDS|" — a rectangular table query, not a traversal. The graph structure offers no query-performance advantage for this access pattern.

PDS is a per-path×pair×contrast scalar that depends on SigProb (a path-level quantity involving all four nodes simultaneously via Hill function products), making it a genuine edge-of-edges attribute that doesn't simplify in a node/edge decomposition. The path scores cannot be dropped or derived from node attributes alone.

**Storage:** A graph representation of 9.6M distinct LRET tuples as edges (L→R→EM→T) involves 9.6M × 4-node traversals. Kuzu or a custom adjacency table would give no compression advantage over a parquet topology table; path-level scores still require a 42M-row fact table equivalent. Net storage would be 350–500 MB, similar to Option A, with substantially more infrastructure complexity.

**Verdict:** Wrong access pattern. Not recommended.

### Option D: Precomputed cube only, lazy full-path detail from source parquets

**Concept:** Shrink the edge shard to only the top-N paths by |PDS| per (sender, receiver, contrast), serving the rest lazily from the source wide parquets via a small query API.

**Analysis:** The viewer currently paginates the full shard in JS — users can scroll through ~47K rows per pair. If shards were pre-truncated to top-300 per contrast (2,700 rows per shard), total shard size drops from 543 MB to ~3 MB (1000×), but the user loses access to paths outside the top-300. The significance filter already establishes a quality floor; further truncation changes research scope, not just implementation. This is an analytical decision, not a storage one.

A lazy-fetch alternative (serve source parquets on demand via a small Python HTTP server or serverless function) would require a persistent backend, breaking the current static-file serving model (the viewer is a single `index.html` with no server). Not compatible with the current architecture.

**Verdict:** Truncation changes analytical scope (out of scope for this memo). Lazy fetch requires a backend (breaks current model).

---

## 4. Recommendation

**Implement Option B** (stripped shards + global node-FC sidecar).

**Migration sketch:**

1. In `_write_incytr_pair_pathways` in `build_unified_viewer.py`:
   - After creating the `src` VIEW, run one additional query: extract distinct (gene, role, sender_or_null, receiver, contrast, all 4 FCs, label) across all contrasts and write to `edge_slices/incytr_pathways/node_fc.parquet`
   - Modify `shard_select_cols` to drop `_INCYTR_FC_COLS` and `_INCYTR_LABEL_COLS`
   - Drop the `float16_cols` cast for the FC columns (they won't be in the shard)

2. In `incytr_pathways.js`:
   - At shard load (in `SliceCache.loadIncytrShard`), also fetch `node_fc.parquet` once and parse to a `Map`
   - In `_ipRenderEvidencePanel` / `EvidencePanel.render()`, look up node FCs from the Map instead of `r[fc_col]`

3. In `04_slice_cache.js`, add the node_fc fetch and parse step.

**Expected outcome:** Shard total drops from 543 MB to ~271 MB. Build peak memory reduced proportionally. Viewer fetch per pair click drops from ~565 KB avg to ~280 KB avg. Page-load addition: one 10 MB fetch for the node_fc sidecar (acceptable; modern browsers cache static assets). PDS ranking, traj_labels, score columns, heatmap cubes: all unchanged.

---

## 5. Tradeoff Summary

| Option | On-disk (shards) | Build complexity | Viewer complexity | PDS ranking preserved | Analytical scope |
|---|---|---|---|---|---|
| Current (no change) | 543 MB | Baseline | Baseline | Yes | Full |
| A: Star schema | ~320 MB (−41%) | Medium (path_id keying) | High (JOIN on fetch) | Yes | Full |
| **B: Stripped shards + FC sidecar** | **~271 MB (−50%)** | **Low (one extra query)** | **Low (Map lookup)** | **Yes** | **Full** |
| C: Graph | ~400 MB | Very high | Very high | Yes | Full |
| D: Truncated shards | ~3 MB (−99%) | Low | Low | Yes | Reduced (top-N only) |

**Note on the source `wide/` parquets (3.81 GB):** These are the build input, not the viewer artifact. They cannot be normalized to reduce build memory because the build already uses a DuckDB VIEW (not materialization) over them. They could be deleted after shards are built; the only cost is having to re-run Incytr to regenerate them. Whether to keep or discard them is a data-management decision, not addressed here.

**Note on `wide_nboot0/` (9.8 GB):** Audit/archive artifact. Not consumed by viewer or build. Can be archived to cold storage when disk pressure warrants.
