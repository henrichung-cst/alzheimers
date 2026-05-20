# Plan — Incytr pathway "Measurement Trace" panel (transcript v1)

## Context

The unified viewer's Incytr Pathways tab shows one row per (Ligand → Receptor → EM → Target) path in a contrast, with each element's LFC visible. There is no way to see the underlying values that produced those LFCs without leaving the viewer.

We are adding a per-row expandable **"Measurement Trace"** panel — modeled on the existing `measurement-trace` tab in the kinase audit drawer (`.../template/js/tabs/kinase_audit.js`) — that displays the transcript pseudobulk values backing the row, for the two arms of the row's contrast.

Purpose is audit, not re-scoring. The panel does not feed any filter or rank.

v1 is transcript-only. Protein and phospho are out of scope.

## Substrate

Incytr pathways are already cell-type-scoped (sender × receiver). Each row carries its sender cluster, receiver cluster, contrast, and the four element genes. The only thing we need to add is per-(cluster, gene, arm) transcript values.

**The viewer is cluster-scheme-agnostic.** The active cluster vocabulary is whatever the current Incytr run produced — discovered at build time from the pathway shards themselves (`edge_slices/incytr_pathways/*.parquet`). No spine name, supertype name, or cluster list is hardcoded anywhere in the builder, viewer, or JS. A future run on a different cluster scheme requires zero code change here; it requires a pseudobulk substrate keyed on the same scheme.

**Substrate contract.** The pseudobulk source path is a config-level constant (`config.PSEUDOBULK_FOR_VIEWER` or equivalent — match whatever the existing config module uses). It points to a table whose cluster labels are a **superset** of the clusters used in the active pathway run, with a sample-group axis joinable via a sibling sample-key file (path also in config).

For the current run, the constants resolve to `data/incytr_frozen/v2_46clusters/provenance/aggexp.csv` and `yuyu_samplekey.csv`. The 46-cluster aggexp is a superset of the levy_t5 clusters used by today's pathway output (levy_t5 spine labels are an exact-string subset of aggexp row names — `Astrocytes`, `Excitatory principal neurons in the hippocampal dentate gyrus`, etc., spaces preserved). No substrate regeneration required.

**aggexp.csv shape and key.** Produced by `aggregate_expression.R` via `do.call(rbind, datalist)` over 24 per-`Group` `AggregateExpression` frames, each transposed so rows are clusters. R `rbind` deduplicates colliding rownames by appending **integer suffixes with no separator** (`Astrocytes`, `Astrocytes1`, `Astrocytes2`, ... `Astrocytes23`). Some cluster names already end in digits (`cluster-27`, `Excitatory-Pyramidal-Satb2-Cux2`, `Foxp2-Excitatory-Neurons-layers-6-and-2-3`), so trailing-integer parsing is ambiguous and must not be used. The **only** reliable key is `(canonical_cluster_name_from_first_46_rows, Group column value)`. Concretely: the first 46 rows of aggexp.csv define the canonical cluster vocabulary in order; rows 47–92 are the same 46 clusters in the same order for the second Group; etc. The builder reads aggexp, slices the first 46 rows to extract the canonical names, and then derives `(cluster, group)` for every row by `(canonical_names[i % 46], Group_col[i])` — with an assertion that `Group_col` is constant within each 46-row block.

**Slug rule.** Filenames in `edge_slices/incytr_pathways/` use `sanitize_celltype(name) = name.replace(" ", "_").replace("/", "-")`, defined in `alz/integration/load.R` and mirrored in `alz/integration/export_factorial_inputs.py::_sanitize_cluster` and `alz/integration/pair_to_receiver_cache.py::_sanitize_celltype`. The builder must import / re-use this exact function (no re-definition). Parquet row values keep the unsanitized form (spaces preserved); only filenames are slugged.

**Contrast → arms mapping (new code, no existing helper).** Pair-mode rows carry a `contrast` like `App_4mo`. Pair-mode is **males-only** by default (per `analysis_mode` in `conf/base/parameters.yml`), so the two arms decode as `(ma_<age>_<geno_code>, ma_<age>_WTyp)` where the geno code is the inverse of `_GENO_MAP` in `pair_to_receiver_cache.py` (e.g., `App → AppP`). This mapping does **not** exist as a reusable utility; it must be written. The `analysis_mode` assumption is made explicit in the helper (hardcoded `ma_` prefix is acceptable for v1 with a comment; if the run is full-cohort, the contrast string itself is unchanged and we'd need to split into two panels — out of scope).

The builder fails loudly only in the genuinely broken case: a cluster appears in pathway output but is missing from the pseudobulk substrate, or a contrast's expected arm groups are not present in the sample-key file. Both point at substrate drift, not at the viewer.

The Song snRNA-seq data has **one pseudobulk library per (sex × timepoint × genotype) group** (24 groups; pair-mode currently uses 18 of them — males only, 9 contrasts × 2 arms with WT shared). A contrast therefore has N=1 per arm on the transcript axis. The panel shows the two values as bars with their numeric values printed alongside, and a header note ("Transcript pseudobulk · 1 library per arm · males only") so the bars cannot be misread as a distribution.

## Decisions (locked)

1. **Strict contrast scope.** Two bars per element, one per arm. No adjacent timepoints, no other groups, no greyed background.
2. **Transcript only in v1.** Protein and phospho deferred.
3. **N=1 caveat shown.** Header note in the panel, non-optional UX.
4. **Lazy shards, not inlined.** Embedding would push the 77MB payload further. Shards live under `outputs/reports/unified_viewer/audit_sources/transcript_trace/<cluster>.parquet`, fetched on row expansion. Mirrors the existing `edge_slices/` model.
5. **Read-only.** No effect on PDS, ranking, sorting, or filters.

## Actions

### 1. New: `alz/integration/build_transcript_trace.py`

Reshape the configured pseudobulk substrate to per-cluster shards. No spine, supertype, or cluster name appears as a literal in this file.

1. Read the pseudobulk path from config. Read the sample-key path from config.
2. **Parse aggexp into `(cluster, group, gene → value)`** using the block-of-46 contract documented in the Substrate section: first 46 rows define the canonical cluster vocabulary in order; thereafter every row's cluster is `canonical[i % 46]` and its group is the row's `Group` column value. Assert: (a) total row count is `46 × len(unique(Group))`; (b) `Group` is constant within each 46-row block; (c) the unique Group values match the `SCRNA_ID` column of `yuyu_samplekey.csv`. Hard-fail on any assertion (silent mis-binning is unacceptable per the no-known-wrong-output rule).
3. **Discover the active cluster set** from `EDGE_SLICES_INCYTR_PATHWAYS_DIR`. Two-step: (a) parse `sender__receiver` from filenames and take the union of slugged names; (b) read one parquet row per file to recover the unsanitized cluster strings the rows actually carry — this is what the JS sees on row data. Reconcile with the slug rule from `sanitize_celltype` (imported from `alz/integration/load.R` via the existing call sites, or re-imported from `pair_to_receiver_cache.py::_sanitize_celltype`). This is the authoritative cluster list for this build.
4. Coverage check: every unsanitized cluster string from step 3 must exist as a canonical name (first-46-rows set) in aggexp. If any are missing, **hard-fail** pointing at the substrate ("pathway output references cluster `X` that is absent from the pseudobulk at `<path>`"). Clusters present in pseudobulk but not in pathway output are silently skipped.
5. Join Group values to `(sex, timepoint, genotype)` via the sample-key file (`SCRNA_ID` → `MS_ID` decode: `ma_4mo_AppP` → `(M, 4mo, APP)` etc.).
6. Shard by cluster: one parquet per cluster at `transcript_trace/<sanitize_celltype(cluster)>.parquet` with columns `(gene, group, sex, timepoint, genotype, value)`. Filename slug uses the imported `sanitize_celltype`, not a local re-implementation.
7. Emit `transcript_trace/index.json`: schema version, generation timestamp, clusters present (unsanitized canonical names, post-validation), groups present, source file hashes, pseudobulk source path. The index is the discovery surface — JS reads it; nothing else.

### 2. Edit: `alz/viewer/paths.py`

Add `TRANSCRIPT_TRACE_DIR`, `TRANSCRIPT_TRACE_INDEX`, `TRANSCRIPT_TRACE_SCHEMA_VERSION = 1` alongside the existing `MEASUREMENT_TRACE_*` constants.

### 3. Edit: `alz/build_unified_viewer.py`

- Add `ensure_transcript_trace_sources()`, parallel to `ensure_measurement_trace_sources()`. If the index is missing or schema-mismatched, invoke the builder. Hard-fail if upstream substrate is missing — do not produce a viewer with an empty panel.
- Stamp `payload["meta"]["transcript_trace"] = {schema_version, clusters, sample_groups}` so the JS can detect availability.

### 4. New: `alz/viewer/template/js/widgets/transcript_trace.js`

Symmetric with the existing `MeasurementTraceStore`:

- `TranscriptTraceStore.loadCluster(cluster)` — fetch and cache the cluster's shard via the existing `SliceCache` path; cluster string is passed unsanitized and slugged at fetch time using a JS-side mirror of `sanitize_celltype` (`name.replaceAll(" ", "_").replaceAll("/", "-")`) — single helper, not scattered.
- `TranscriptTraceStore.contrastToArms(contrast)` — **new helper** (no upstream JS equivalent). Splits `<geno>_<age>` (e.g., `App_4mo`), inverts the `_GENO_MAP` from `pair_to_receiver_cache.py` (`App → AppP`, `Tau → Ttau`, `ApTt → ApTt`), and returns `[{arm: "<geno>", group: "ma_<age>_<geno_code>"}, {arm: "WT", group: "ma_<age>_WTyp"}]`. The `ma_` prefix is hardcoded with a comment that it encodes the males-only pair-mode assumption; revisit if the contrast schema gains a sex dimension.
- `TranscriptTraceStore.values(cluster, gene, contrast)` — calls `contrastToArms`, looks up the two `(group, gene)` cells from the cached shard, returns `[{arm, group, value}, {arm, group, value}]`. Missing gene → both values are null and the panel slot renders "no transcript pseudobulk for this gene."
- `renderTwoBarPanel(container, gene, cluster, contrast)` — small two-bar SVG with arm labels under and numeric values above. Fixed y-scale per panel (max × 1.15).

### 5. Edit: `alz/viewer/template/js/tabs/incytr_pathways.js`

On row expansion, render four element panels (L, R, EM, T). Sender cluster for L and EM, receiver cluster for R and T — these come straight off the row as opaque strings. **No cluster names, spine names, or vocabularies are referenced in JS.** The store fetches `transcript_trace/<cluster>.parquet` using whatever string is on the row; if the index does not list that cluster, the panel slot reads "no transcript trace for this cluster in this build." The header note is mandatory; missing-gene slots read "no transcript pseudobulk for this gene in this cluster." If `payload.meta.transcript_trace` is absent, render a single placeholder and skip the store entirely.

### 6. Edit: `alz/viewer/template/styles.css`

Styles for the subsection. Reuse `audit-measurement-trace` class names where the layout matches.

## Validation

1. Sample of 10 random `aggexp.csv` rows: block-of-46 parser recovers the right `(cluster, group)` pair, including for the digit-ending clusters (`cluster-27`, `Excitatory-Pyramidal-Satb2-Cux2`, `Foxp2-Excitatory-Neurons-layers-6-and-2-3`) where naive suffix-stripping would fail.
2. Every cluster appearing in `edge_slices/incytr_pathways/` filenames produces a shard. Clusters present in the pseudobulk but absent from pathway output do not produce shards.
3. All sample groups from the configured sample-key file appear in at least one shard.
4. Point the pseudobulk config constant at a substrate using *different* cluster labels (synthetic or real); the builder either produces correct shards without code change, or fails cleanly with the coverage error. No code path silently re-labels.
4. `contrastToArms` round-trip: for each of the 9 contrasts in the current run, the two returned `group` strings exist in the sample-key file's `SCRNA_ID` column.
5. Expand a row where the Ligand is a kinase known to be perturbed in the contrast — bar direction matches the LFC chip already on the row.
6. Expand a row whose EM gene is absent from `aggexp.csv` — panel slot reads "no transcript pseudobulk" rather than a zero bar.
7. `unified_viewer.payload.json.gz` size unchanged (no inlining).
8. Kinase audit measurement-trace still renders.

## Out of scope

- Protein and phospho modalities.
- Cross-timepoint or cross-genotype context bars.
- Any change to PDS / scoring / ranking / filters / row eligibility.

## Staging

Lands on `feat/cr04-incytr-viewer` where the viewer source lives.
