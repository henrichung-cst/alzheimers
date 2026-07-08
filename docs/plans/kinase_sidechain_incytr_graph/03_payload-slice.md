# kinase_sidechain_incytr_graph — subplan 03: viewer payload slice

**Goal:** Slice subplan 01's per-cohort backend artifact (kinase→kinase interactome + terminal-edge
map), plus subplan 02's t-cell motif edges, into the `edge_slice_ref` payload the sidechain viewer tab
consumes — for song, 5xFAD, and t-cells. Bounded by the kinome, not the pathway count (no per-pathway
precompute).

**Scope**
- In: Add a sidechain edge-slice writer (the interactome + terminal-edge map, provenance-tagged and
  weighted) and register it in each cohort's slice assembly. Emit the slice under the existing
  `edge_slices/incytr_pathways/` convention; wire the `edge_slice_ref` entry through `compose.py`.
- Out: No backend edge computation (01/02 own the math — this only *shapes* their output). No new tab
  or JS (04). No change to existing slices (kinases, celltypes, decomp, concordance, backbone) — this is
  purely additive.

**Files touched:** `alz/viewer/shared/compose.py`, `alz/viewer/shared/incytr_index.py`,
`alz/viewer/cohorts/song.py`, `alz/viewer/cohorts/fivexfad.py`, `alz/tcell_viewer/slices_incytr.py`.

**Risk tier:** **medium** — plumbing that shapes 01/02's output into the payload contract. Most failure
modes are loud: a duplicate slice key throws at `merge_edge_slice_ref`, a schema break fails contract
validation. The silent mode is a mis-shaped-but-valid payload (right keys, wrong values — e.g. weights
transposed, a cohort's slice keyed to another) that renders as a plausible-but-wrong graph. Medium.

**Reuse (do not rebuild):**
- `merge_edge_slice_ref` (`compose.py`) — the per-cohort `edge_slice_ref` merge; it **errors on
  duplicate keys**, so pick a unique slice key.
- `_write_gene_node_index_shard` / `_build_incytr_gene_node_index` (`incytr_index.py`) — the existing
  columnar-shard writer pattern; follow it for the new slice's on-disk form.
- The existing `_write_incytr_pair_pathways` (song) / `_write_donor_pair_pathways` (t-cell) as the
  per-cohort registration pattern.

## Operating rules (non-negotiable — this repo's standing constraints)

- ANTI-SHIM: a pivot REPLACES the old approach; it does not coexist with it. No feature flag
  defaulting to the old mode, no `if name == "old"` branch, no legacy wrapper, no fallback to
  superseded logic, no env-var escape hatch, no config key toggled `enabled: false`. Update
  docstrings, comments, README, and runner scripts in the SAME pass. NO TOMBSTONES: when you remove
  something, delete it outright — do not leave a pointer recording that it once existed ("formerly X",
  "removed — see Y", a struck line, a status annotation on an emptied entry). Git holds the history.

- MEMORY SAFETY (shared box, OOM crashes other users' jobs): assume every dataset is too big for RAM
  until proven otherwise. Never call `pd.read_parquet` / `read_parquet` / `read.csv` / `fread` /
  `arrow::read_*` on a multi-GB file — not to "have a look", not inside a script you author (the
  validator/derive step you write is what crashes the box). The memory bill is row-count × col-count
  decompressed, not file size. Stream instead: DuckDB (`COPY … TO`, SQL aggregation/join),
  `pyarrow.parquet.ParquetFile(...)` row-group iteration, `arrow::open_dataset()` with explicit column
  selection — return only scalars or a bounded result. Check size first (`ls -lh` / `du -h` / parquet
  metadata). Run any step that could touch a large artifact under a cap:
  `systemd-run --user --scope -p MemoryMax=<N>G -p MemorySwapMax=0 …`. If you must load something big,
  say so first and wait for confirmation.

- VERIFICATION USES `pixi run`: bare `python` is the system interpreter and lacks project deps
  (pyarrow/duckdb/…). Always `pixi run python …` or the relevant `pixi run <task>`. Run
  `pixi task list` before assuming a task name.

- GIT: do not push, open PRs, force-push, `git reset --hard`, or any destructive git op without
  explicit instruction. Commit messages carry NO LLM attribution / Co-Authored-By trailer. NEVER
  commit data files (`.rds`, `.csv`, `.parquet`, `.mtx`, `.xlsx`, `.h5ad`, or other data artifacts).

- HONESTY OVER POLISH: no `TODO` / "coming soon" / "to be implemented" / placeholder in user-facing
  surfaces — omit an incomplete feature entirely. Do not ship known-incorrect output (nulls, stale
  joins, wrong vocabularies) to keep a diff small; fix the upstream cause. A bad benchmark result is a
  finding to report, not a metric to relax.

- NO COLLABORATOR CONTACT — EVER: do not propose emailing/asking/requesting anything from external
  collaborators. All work proceeds from artifacts on disk + what is derivable from them. Missing
  intermediates (`*.rds`, derived CSVs, cached matrices) are derivable — reconstructing them from raw
  inputs + method IS the task, not a blocker.

- TOOL DISCIPLINE: standard commands are shadowed by modern replacements (`grep`→`rg`, `cat`→`bat`,
  `find` is a wrapper). GNU/POSIX flags break in plain shell calls — use `command grep` / `command find`
  / the absolute binary when you need real coreutils behavior. In fish scripts, use builtins
  (`string match -r`, `string replace`, `count`), and never use `_` as a loop variable (read-only).

**Curated specifics (invariants for this subplan's files)**
- **Conform to the payload contract:** `docs/foundation/viewer_payload_contract.md` defines the
  `edge_slice_ref` schema (schema_version, per-cohort merge). The new slice must validate under it.
- **Shared engine keys are frozen names.** `cell_type` / `specificity_celltype` / `confidence_basis`
  are cross-cohort engine keys — the unified viewer reads them by name across cohorts. Rename display
  *labels* if needed, never the keys, or you fork the shared viewer.
- **Payload is kinome-bounded, not path-bounded** (source plan): emit the interactome + terminal-edge
  map only. Do **not** precompute a per-pathway artifact; the client walks the interactome on selection.
- **On-disk slices go under `edge_slices/incytr_pathways/`;** t-cell shards use the `{donor}__` prefix
  (existing convention in `slices_incytr.py`).
- **Reads streamed** — slice builders that touch parquet use DuckDB, never `pd.read_parquet` on a wide
  shard.
- After a rebuild, PAYLOAD is inlined into `index.html` by `build_unified_viewer.py`; a stale browser
  cache serves old data — verify against `PAYLOAD.meta.generated_at`, not a soft refresh.

**Acceptance / verify:**
- The compose/build step (`pixi run viewer`, or the cohort slice task — check `pixi task list`) emits an
  `edge_slice_ref` entry for the sidechain slice for **all three cohorts**, and `merge_edge_slice_ref`
  raises **no duplicate-key error**.
- The emitted slice loads and, for one known song pathway, exposes the 4 node genes' terminal kinases +
  a reachable multi-hop chain (the same cascade subplan 01 verified).
- Payload keys validate against `viewer_payload_contract.md`.

**Declared dependencies:** **01 and 02** (data). Reads 01's per-cohort backend artifact (all cohorts)
and 02's t-cell motif edges (t-cell arm). File-disjoint from both — the dependency is artifact
consumption, not a shared file. Consumed by subplan 04.
