# kinase_sidechain_incytr_graph — subplan 03: viewer payload slice

**Goal:** Slice subplan 01's per-cohort backend artifact (kinase→kinase interactome + terminal-edge
map) into the `edge_slice_ref` payload the sidechain viewer tab consumes — for song, 5xFAD, and
t-cells. Bounded by the kinome, not the pathway count (no per-pathway precompute).

All four cohort dirs are already built and on disk under `outputs/reports/kinase_kinase_edges/`:
`song/`, `fivexfad_cortex/`, `fivexfad_hippocampus/`, `tcells_donor1/`. Each holds `interactome.csv`
+ `terminal_edges.csv` with an identical schema, so **one slice writer covers every cohort** — the
t-cell arm is not a special case.

**Scope**
- In: Add a sidechain edge-slice writer (the interactome + terminal-edge map, provenance-tagged and
  weighted) and register it in each cohort's slice assembly. Emit the slice under the existing
  `edge_slices/incytr_pathways/` convention; wire the `edge_slice_ref` entry through `compose.py`.
- Out: No backend edge computation (01 owns the math — this only *shapes* its output). No new tab
  or JS (04). No change to existing slices (kinases, celltypes, decomp, concordance, backbone) — this is
  purely additive.

**Files touched:** `alz/viewer/shared/compose.py`, `alz/viewer/shared/payload_helpers.py`,
`alz/viewer/cohorts/song.py`, `alz/viewer/cohorts/fivexfad.py`, `alz/tcell_viewer/slices_incytr.py`.
(`incytr_index.py` is **not** in scope — it owns backbone grains and sign-vector labels, nothing this
subplan needs.)

**Risk tier:** **medium** — plumbing that shapes 01's output into the payload contract. Most failure
modes are loud: a duplicate slice key throws at `merge_edge_slice_ref`, a schema break fails contract
validation. The silent mode is a mis-shaped-but-valid payload (right keys, wrong values — e.g. weights
transposed, a cohort's slice keyed to another) that renders as a plausible-but-wrong graph. Medium.

**Reuse (do not rebuild):**
- `merge_edge_slice_ref` (`compose.py`) — the per-cohort `edge_slice_ref` merge; it **errors on
  duplicate keys**, so pick a unique slice key.
- `_write_gene_node_index_shard` / `_build_incytr_gene_node_index` (**`alz/viewer/shared/payload_helpers.py`**)
  — the existing columnar-shard writer pattern; follow it for the new slice's on-disk form. All three
  cohort files already import both from there (`song.py:58`, `fivexfad.py:64`, `slices_incytr.py:23`);
  follow that same import, and keep any new shared helper in `payload_helpers.py` beside them.
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
- **The t-cell arm is donor1-only, and must say so.** `tcells_donor1` is the only t-cell cohort dir that
  exists: donor2 has no within-cohort attribution, so the bridge emits no donor2 motif source and 01
  produces no donor2 edges. Emit the donor1 slice and nothing else. Do **not** synthesize a donor2 slice,
  reuse donor1's under a donor2 key, or leave a donor2 placeholder — if the tab offers a donor selector,
  donor2 must be absent rather than empty-but-present.
- **Terminal edges are motif-anchored (settled — do not "fix" this).** Every `terminal_edges.csv` row is a
  motif edge; PhosphoSitePlus only corroborates (`provenance` = `both`), so **psp-only terminal edges do
  not exist by design** (verified 0 in all four cohorts). Purely-literature kinases still reach a pathway
  via the interactome walk. A slice that shows zero psp-only terminal edges is correct, not a bug.
- **01's artifacts are small — measured, not assumed.** Largest is `tcells_donor1/terminal_edges.csv` at
  31 MB (~352k rows × 10 cols); every other file is ≤ 9.5 MB. A direct `pd.read_csv` of these is fine and
  is the expected approach — do not build DuckDB streaming for a 31 MB CSV. The standing memory rule still
  binds for the **wide Incytr shards**, which this subplan has no reason to open: 01 already reduced them,
  so re-deriving anything from `incytr_pair_mode*/wide/` here is both a memory risk and duplicated math.
- After a rebuild, PAYLOAD is inlined into `index.html` by `build_unified_viewer.py`; a stale browser
  cache serves old data — verify against `PAYLOAD.meta.generated_at`, not a soft refresh.

**Acceptance / verify:**
- The compose/build step (`pixi run viewer`, or the cohort slice task — check `pixi task list`) emits an
  `edge_slice_ref` entry for the sidechain slice for **all three cohorts / four cohort dirs**, and
  `merge_edge_slice_ref` raises **no duplicate-key error**.
- The emitted slice loads and, for one known song pathway, exposes the 4 node genes' terminal kinases +
  a reachable multi-hop chain (the same cascade subplan 01 verified).
- Payload keys validate against `viewer_payload_contract.md`.
- **Edge counts survive the slice unchanged** — 01's verified output, which the slice only reshapes:

  | cohort dir | interactome edges | nodes | terminal edges |
  |---|---|---|---|
  | `song` | 3701 | 373 | 113437 |
  | `fivexfad_cortex` | 2977 | 375 | 64193 |
  | `fivexfad_hippocampus` | 1606 | 345 | 19968 |
  | `tcells_donor1` | 6676 | 392 | 351641 |

  A slice whose row count differs from these has dropped or duplicated edges — investigate, do not
  reconcile by adjusting the number.
- **Gene space stays cohort-native:** song/5xFAD nodes are mouse (`Gucy1b1`), t-cell nodes are human
  (`HEG1`). A cohort's slice containing the other's casing means a cross-species leak.

**Declared dependencies:** **01** (data) — reads 01's per-cohort backend artifact for *all four* cohort
dirs, t-cells included. 02 is a transitive dependency only: its bridge output is 01's motif source, so
the real t-cell chain is **02 → 01 → 03**, not `[01 ∥ 02] → 03`. 03 never reads 02's
`kinase_node_hits.parquet` directly — that is the raw motif hit table, not the interactome/terminal
structure this subplan slices. Both are satisfied and on disk; nothing here is blocked. File-disjoint
from both — the dependency is artifact consumption, not a shared file. Consumed by subplan 04.
