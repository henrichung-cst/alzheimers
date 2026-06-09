# Unified Viewer Performance Audit

Date: 2026-06-09

## Context

The unified viewer is deployed as a static folder in S3 and opened through
`index.html`. The current build emits both:

- `index.html` with the full payload inlined in `<script id="payload-data">`.
- `unified_viewer.payload.json` and `unified_viewer.payload.json.gz` sidecars.

Current generated sizes observed during the audit:

- `index.html`: about 64 MB.
- `unified_viewer.payload.json`: about 63 MB.
- `unified_viewer.payload.json.gz`: about 7.9 MB.

Payload block sizes, raw JSON:

- `attribution_index`: about 26.8 MB.
- `incytr_pathways`: about 23.2 MB.
- `human`: about 9.6 MB.
- `decomposition_index`: about 2.5 MB.
- `kinase_celltype_evidence`: about 1.7 MB.

The largest inline `incytr_pathways` sub-block is `gene_node_index`, about
15.7 MB raw. It is used for exact cross-pair gene search before loading a
specific pathway shard.

## Improvement Candidates

### 1. Make sidecar gzip payload the default for hosted S3 builds

Current behavior favors single-file portability: the full JSON payload is
embedded in `index.html`.

For S3-folder deployment, sidecar gzip should be the default:

- Write a small `index.html` containing `payload-data` as `null`.
- Keep `unified_viewer.payload.json.gz` beside it.
- Let the existing `_loadPayload()` path fetch and manually decompress
  `unified_viewer.payload.json.gz` with `DecompressionStream("gzip")`.
- Keep inline payload as an explicit archival/offline option, for example
  `--inline-payload`.

Expected benefit:

- Network transfer drops from about 64 MB HTML to small HTML plus about 8 MB
  gzip sidecar.
- Browser no longer parses a huge inline JSON text node as part of HTML
  document construction.
- Payload and shell can cache independently.

Tradeoffs:

- Not a single-file artifact anymore.
- Must be served over HTTP(S), not opened through `file://`.
- Requires the `.gz` sidecar to remain beside `index.html`.
- If S3/CloudFront is configured with `Content-Encoding: gzip`, the loader
  should not manually decompress the response body. In the current environment
  we suspect S3 does not provide gzip content encoding, so manual decompression
  of raw `.gz` bytes is the intended mode.

Priority: high for S3-folder deployment.

### 2. Index attribution rows by kinase in JavaScript

`getScopedAttribution()` currently scans every row in `PAYLOAD.attribution_index`
for each call. Several kinase-table sort/render paths call it per kinase, so a
single render can repeatedly rescan the full 108k-row attribution table.

Proposed change:

- Build a one-time index: `kinase_id -> attribution row indices`.
- Optionally add `kinase_id|contrast_id -> row indices` for contrast-scoped
  panels.
- Have `getScopedAttribution()` iterate only the indexed slice.

Expected benefit:

- Faster kinase filtering, confidence sorting, and cell-type pill rendering.
- Small memory cost for row-index arrays.

Tradeoffs:

- Slightly more startup or first-kinase-tab initialization work.
- Additional cache invalidation needed if payload contexts become dynamic.

Priority: high.

### 3. Run pathway round-trip verification once per payload build

The default payload path currently runs the pathway round-trip verifier inside
`build_payload()` and again at the end of the CLI payload build. One recent
default verifier pass took about 171 seconds.

Proposed change:

- Keep one default verifier call.
- Keep `--strict-roundtrip` for full-grid pre-publish/CI verification.
- Keep `--skip-roundtrip` for fast iteration.

Expected benefit:

- Saves one redundant verifier pass per default build.

Tradeoffs:

- None if one verifier pass remains.

Priority: high for build time.

### 4. Add source fingerprints for shard rebuild skipping

The build currently wipes and rewrites decomp OLS, Song concordance, and Incytr
pathway shard directories. This is robust but expensive.

Proposed change:

- Emit manifest files per shard family containing:
  - source paths and mtimes/hashes,
  - relevant schema versions,
  - relevant config values,
  - output row counts.
- Skip shard rebuilds only when the manifest exactly matches.

Expected benefit:

- Faster viewer iteration when only HTML/JS or small payload blocks change.

Tradeoffs:

- Incorrect invalidation risks stale shards, which is worse than slow builds.
- Requires careful manifest design and tests.

Priority: medium; implement only with strict invalidation.

### 5. Move `gene_node_index` to a lazy sidecar

`gene_node_index` is about 15.7 MB raw inside the main payload and is only needed
when the user performs cross-pair exact gene search in the Incytr tab.

Proposed change:

- Write `edge_slices/incytr_pathways/gene_node_index.json.gz`.
- Store only its URL and compact metadata in the main payload.
- Lazy-load it on first gene search.

Expected benefit:

- Reduces startup payload parse and transfer size.

Tradeoffs:

- First cross-pair gene search incurs one additional fetch and parse.
- Need loading/error UI for the gene-search panel.

Priority: medium-high after sidecar payload default.

### 6. Debounce and cache audit-table filtering

Audit tables filter and sort all loaded rows on each render. The search input
currently triggers render immediately on every keystroke.

Proposed change:

- Debounce search input by about 150-250 ms.
- Cache filtered/sorted results by `(query, sortCol, sortAsc, rowsVersion)`.

Expected benefit:

- Smoother audit-table interaction on large copied CSVs.

Tradeoffs:

- Search result updates are delayed slightly.
- Cache must be cleared when table rows change.

Priority: medium.

## Implemented During Audit

`SliceCache` now coalesces concurrent fetch/decode requests for:

- backbone buckets,
- decomp OLS shards,
- Song concordance shards,
- Incytr pathway shards.

This matches the pre-existing human per-donor shard behavior. It prevents rapid
UI changes from launching duplicate network reads and duplicate parquet decodes
for the same shard.

Tradeoffs:

- Concurrent callers share the same success or failure for a request.
- This is the expected behavior for identical shard requests.

