# kinase_sidechain_incytr_graph — subplan 02: t-cell single-hop motif bridge

**Goal:** Run the MEA/kldata motif single-hop bridge for the t-cell cohort, per donor, producing
kinase→node motif edges in the *same schema* song/5xFAD already emit — the t-cell motif-evidence
component the backend terminal-map (subplan 01's builder) and the payload (subplan 03) consume.

**Scope**
- In: Add a t-cell run path to `kinase_incytr_bridge.py`: per-donor scoping (donor1, donor2),
  cell-type attribution wiring against the t-cell sender/receiver clusters, and emission of motif
  terminal edges in the song-identical schema. Correct the generated MANIFEST note that falsely claims
  the t-cell cohort is excluded.
- Out: No new `gene_node_index` build — the t-cell index **already exists** (see curated specifics);
  consume it. No PSP / kinase→kinase work (subplan 01). No viewer payload (03). No change to the Incytr
  scorer, `gene.use` selection, or any scoring parameter — this bridge consumes Incytr output.

**Files touched:** `alz/cross_reference/kinase_incytr_bridge.py`.

**Risk tier:** **medium** — mostly mechanical: mirror the song/5xFAD motif path for a new cohort in a
byte-identical schema (a column-diff catches divergence loudly). The silent spot is the **10%
within-cohort detection floor** in cell-type attribution — get that wrong and a kinase is marked active
in a cluster it isn't, invisibly. Medium, driven by that one gate.

**Reuse (do not rebuild):**
- `build_substrate_bridge` in the same file — the t-cell MEA format is **byte-identical to song**, so
  this function needs **no change**. Per-donor scoping and cell-type attribution are the real work.
- `load_gene_node_index` — point it at the existing per-donor t-cell index shards.
- The song/5xFAD cell-type-match annotators (`annotate_celltype_match_song` /
  `annotate_celltype_match_fivexfad`) as the pattern for the t-cell annotator.

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
- **The t-cell `gene_node_index` is ALREADY built** — `alz/tcell_viewer/slices_incytr.py` calls
  `_build_incytr_gene_node_index` and writes `{donor}__gene_node_index.json.gz` under
  `edge_slices/incytr_pathways/`. The source plan's "the real work is a t-cell gene_node_index" is
  **stale** — do NOT rebuild it; load the existing shards. (Trust the code over the plan here.)
- **The "format barrier" note to correct is generated code, not a static file.** It is emitted by
  `write_outputs` and reads *"T-cell cohort excluded (mea_timecourse.csv format, different scoping)."*
  (seen in `outputs/reports/kinase_incytr_bridge/song/MANIFEST.md:25`). Fix the generator string in
  `kinase_incytr_bridge.py`; the MANIFEST.md files are regenerated output — do not hand-edit them.
- **T-cell within-cohort detection floor is 10%** — a correctness gate. Cell-type attribution must not
  mark a kinase active in a sub-10% t-cell sample. Do not lower the floor to gain hits.
- **Reads streamed.** t-cell Incytr output lives under
  `outputs/reports/incytr_pair_mode_tcells/donor{1,2}/wide/` — stream via DuckDB, never whole-read.
- **Per-donor, not pooled.** donor1 and donor2 are scored separately (donor1 KsG-on, donor2 KsG-off in
  the regeneration runbook); keep the motif bridge per-donor, same as their pathway output.

**Acceptance / verify:**
- `pixi run kinase-incytr-bridge -- --cohort tcells --donor donor1` (and `donor2`) emits motif terminal
  edges with a schema **column-identical** to the song output (diff the column set), and prints per-donor
  hit counts.
- The regenerated MANIFEST no longer states the t-cell cohort is excluded.
- A cell-type-attributed kinase resolves to a t-cell cluster present at ≥10% within-cohort detection
  (spot-check one).

**Declared dependencies:** none — parallel-safe (file-disjoint; extends existing song/5xFAD motif code
for a new cohort in the same schema). Contract with subplan 01: emit in the schema 01's terminal-map
builder accepts. Consumed by subplan 03 (t-cell arm).
