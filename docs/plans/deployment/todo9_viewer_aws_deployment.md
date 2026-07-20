# Viewer AWS Deployment Pipeline — Scaling Plan

**Date:** 2026-06-24
**Status:** Partial — A and B shipped; C/D exploratory and open

- **A — s3 sync deploy.** Shipped as `alz/runners/supporting/deploy_viewer.sh`
  (pixi tasks `deploy-viewer` / `deploy-tcell-viewer` / `deploy-all-viewers`). Three cache passes:
  shards at `max-age=86400 --delete`, payload sidecar at `max-age=300`, HTML `no-cache`.
  Two deliberate deviations from the sketch below: **no `--content-encoding gzip`** (the viewer
  gunzips client-side via `DecompressionStream`; setting the header makes the browser
  pre-decompress and the manual gunzip fails), and `*.bin.gz` rides the short payload cache
  because the binary indexes are byte-coupled to the payload manifest. No CloudFront exists, so
  no invalidation step is needed.
- **B — builder shard cache.** Shipped as `alz/viewer/shared/build_cache.py`: SHA-256 content
  hashing with an mtime+size fast path, output-existence validation, and a schema version to
  invalidate stale manifests. Wired across the `decomp_ols`, `song_concordance`,
  `incytr_pathways`, `human_perdonor`, and `fivexfad_*` shard families.

---

## Problem

The viewer output directory (~2 GB, 5,000+ files) is uploaded in full to AWS S3 on
every build. As the viewer grows (more cohorts, PTM tracks, Incytr contexts), this
is becoming the dominant deployment bottleneck. The 2,968 edge-slice files under
`edge_slices/` (769 MB) change partially on each build, not wholesale — yet the
entire tree is re-uploaded each time.

Current output structure (from `todo8` audit, 2026-06-19):

| Path | Files | Size |
|---|---:|---:|
| `edge_slices/decomp_ols/` | 390 parquets | 311 MB |
| `edge_slices/fivexfad_detail/` | 390 parquets | 194 MB |
| `edge_slices/incytr_pathways/` | 655 files | 197 MB |
| `edge_slices/fivexfad_attribution/` | 383 JSON.gz | 70 MB |
| `edge_slices/fivexfad_celltype_mea/` | 390 JSON.gz | 22 MB |
| `edge_slices/human_perdonor/` | 390 parquets | 12 MB |
| `edge_slices/song_concordance/` | 368 parquets | 3 MB |
| `audit_sources/` | ~200 files | ~50 MB |
| `unified_viewer.payload.json.gz` | 1 | 5.6 MB |
| `index.html` | 1 | 738 KB |
| **Total** | **~5,100** | **~2 GB** |

---

## Option A — Incremental S3 sync (immediate, ~2h)

**What:** Replace the full-folder upload with `aws s3 sync`, which compares ETags
(MD5 of object content) and skips unmodified files.

**Why it works:** Most edge shards don't change between builds. `decomp_ols/` shards
are static once MEA is run. `song_concordance/` shards are static once the concordance
run is done. Only Incytr shards change when a new contrast is added.

**Implementation:**
1. Add a `deploy-viewer` pixi task:
   ```toml
   deploy-viewer = "aws s3 sync outputs/reports/unified_viewer/ s3://<bucket>/unified_viewer/ --delete --cache-control 'max-age=3600' --content-encoding gzip --exclude '*.json' --include '*.json.gz'"
   ```
   (Cache headers: HTML = `no-cache`; `.json.gz` sidecars = short TTL; edge shards =
   long TTL since they're content-addressed.)
2. Add a `deploy-tcell-viewer` task similarly.
3. Set `--delete` so removed shards don't persist on S3.

**Cache-header strategy:**

| Path glob | `Cache-Control` | Rationale |
|---|---|---|
| `index.html` | `no-cache` | Always check for a new shell |
| `*.payload.json.gz` | `max-age=60` | Short; payload changes each build |
| `edge_slices/**` | `max-age=86400` | Shards are content-stable once written |
| `audit_sources/**` | `max-age=3600` | Medium; CSVs regenerated infrequently |

**Expected improvement:** First sync uploads everything. Subsequent syncs upload only
the diff — likely 20–200 files instead of 5,000.

**Risk:** None. `aws s3 sync` is idempotent. `--delete` removes stale shards. Revert
by reverting the pixi task.

---

## Option B — Builder-side shard generation cache (medium, ~1 day)

**What:** Track a content-hash manifest of every edge-shard input. On rebuild, skip
regenerating any shard whose inputs haven't changed.

**Why:** Option A skips re-uploading unchanged shards; this skips re-generating them.
Currently `_write_decomp_ols_slices`, `_write_incytr_pair_pathways`, and
`_write_song_concordance_slices` always regenerate all shards from scratch, even when
the upstream parquet hasn't changed. This burns CPU and also causes spurious mtime
changes that confuse `aws s3 sync`'s shallow comparison.

**Implementation sketch:**

```python
# viewer/shared/shard_cache.py
import hashlib, json, os

def source_hash(paths: list[str]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.encode())
        if os.path.exists(p):
            h.update(str(os.path.getmtime(p)).encode())
            h.update(str(os.path.getsize(p)).encode())
    return h.hexdigest()[:16]

def is_current(manifest_path: str, key: str, current_hash: str) -> bool:
    if not os.path.exists(manifest_path):
        return False
    with open(manifest_path) as f:
        m = json.load(f)
    return m.get(key) == current_hash

def record(manifest_path: str, key: str, hash_val: str) -> None:
    m = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            m = json.load(f)
    m[key] = hash_val
    with open(manifest_path, "w") as f:
        json.dump(m, f)
```

Each shard writer calls `source_hash([upstream_parquet_path])`, checks the manifest,
and returns early if current. The manifest lives at
`outputs/reports/unified_viewer/.shard_manifest.json` (gitignored).

**Which shard families benefit most:**

| Family | Input(s) | Change frequency |
|---|---|---|
| `decomp_ols/` | `site_level_ols.parquet` | Rare (only on MEA rerun) |
| `song_concordance/` | `song_concordance_*.parquet` | Rare |
| `human_perdonor/` | Mukesh source parquets | Rare |
| `incytr_pathways/` | filtered-wide parquets per contrast | Changes per Incytr run |
| `fivexfad_*/` | 5xFAD MEA + Incytr outputs | Rare |

**Expected improvement:** On a typical rebuild (only Incytr output changed), drops
from regenerating ~5,000 shards to ~655. Combined with Option A, upload time
approaches "only what changed."

**Risk:** Stale manifest can cause a shard to be served without update. Mitigate by
including the builder schema version in the manifest key; any schema bump invalidates
all entries.

---

## Option C — Parquet consolidation + HTTP Range requests (large, ~3–5 days)

**What:** Merge per-kinase/per-cluster shards into a small number of large Parquet
files, indexed by row group. The JS fetches the Parquet footer (one HEAD + one small
GET) to locate the relevant row group, then fetches only those bytes via HTTP `Range`.

**Why this is feasible:** S3 supports byte-range requests natively. Parquet stores
a footer index mapping row groups to byte offsets. `decomp_ols/` (390 files, 311 MB)
could become 1–4 Parquet files (~80 MB each, grouped by kinase family or contrast),
reducing file count from 390 to 4 and upload surface to only files whose row groups
changed.

**Rough consolidation targets:**

| Current shard family | Current files | Proposed | New files |
|---|---:|---|---:|
| `decomp_ols/` | 390 | 1 file per contrast (5 contrasts) | 5 |
| `fivexfad_detail/` | 390 | 1 file per contrast | 5 |
| `human_perdonor/` | 390 | 1 file | 1 |
| `song_concordance/` | 368 | 1 file | 1 |
| `incytr_pathways/` | 655 | 1 file per (context, contrast) | ~9 |
| **Total** | **~2,200** | | **~21** |

**JS-side change:** `SliceCache` currently fetches a whole parquet file by URL. It
would need a Parquet footer parser (or a precomputed row-group index sidecar) to
convert a kinase/cluster key to a byte range, then issue a `Range` request. A simpler
variant: emit a small `<family>_index.json` beside each consolidated file mapping
kinase_id → {offset, length}, avoiding the need to parse the Parquet footer in JS.

**Expected improvement:** File count drops from ~5,000 to ~200; upload surface for a
typical build drops to the handful of consolidated files that changed.

**Risk:** Medium. The JS fetch path changes substantially. Existing `SliceCache`
abstraction needs a new backend. Test coverage must verify byte-range correctness.

---

## Option D — DuckDB-Wasm (exploratory, weeks)

**What:** Run DuckDB in the browser via WASM, querying Parquet files on S3 directly
via HTTP range requests. Eliminates pre-sharding entirely — the viewer queries
on-demand rather than fetching pre-built slices.

**Architecture shift:** Instead of `SliceCache` fetching per-kinase parquets, the JS
issues DuckDB-Wasm SQL against a handful of large Parquet files on S3:
```sql
SELECT * FROM read_parquet('s3://bucket/decomp_ols.parquet')
WHERE kinase = 'CDK5' AND contrast = 'ma_2mo'
```
DuckDB-Wasm fetches only the relevant row groups via HTTP range request
automatically — the JS never needs to know about byte offsets.

**Cost:**
- 3 MB WASM bundle to load on first visit (~500ms cold start on fast connection).
- Per-query latency: ~50–200ms depending on network and row group size.
- Non-trivial rewrite of the `SliceCache` and all shard-fetch call sites.

**When to reach for it:** If Option C's consolidation still leaves the deployment
bottleneck unacceptable, or if the query patterns grow complex enough that pre-sharding
can't keep up. Not worth the complexity at current scale; revisit if the viewer adds
3+ new cohorts.

---

## Recommended sequencing

| Step | Option | Effort | Blocking? |
|---|---|---|---|
| 1 | A — `aws s3 sync` + cache headers | 2h | No |
| 2 | B — Builder shard cache | 1 day | No |
| 3 | C — Parquet consolidation | 3–5 days | After B |
| 4 | D — DuckDB-Wasm | Weeks | After C, if still needed |

Start with A immediately — it is risk-free and recovers most of the upload latency
for unchanged shards. B prevents unnecessary regeneration and makes A more precise
(fewer spurious mtime changes). C is the right structural fix if file count remains
the bottleneck after A+B. D is a complete rethink and should only be considered if
the viewer's query complexity outgrows what pre-built shards can serve.

---

## Open questions

1. **S3 bucket + credentials** — which bucket and IAM role does the deployment use?
   Needed to write the pixi task and test Option A.
2. **CloudFront?** — if a CDN sits in front of S3, Option A must also include a
   selective CloudFront invalidation step for `index.html` and `*.payload.json.gz`
   (not edge shards, which are long-TTL).
3. **Option C row-group index format** — prefer a precomputed `_index.json` sidecar
   (no Parquet footer parsing in JS) or a proper Parquet footer reader (more general,
   more JS complexity)?
