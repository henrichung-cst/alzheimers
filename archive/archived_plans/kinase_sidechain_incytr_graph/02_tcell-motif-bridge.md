# kinase_sidechain_incytr_graph — subplan 02: t-cell single-hop motif bridge

**Goal:** Run the MEA/kldata motif single-hop bridge for the t-cell cohort, per donor, producing
kinase→node motif edges in the *same schema* song/5xFAD already emit — the t-cell motif-evidence
component the backend terminal-map (subplan 01's builder) and the payload (subplan 03) consume.

**Scope**
- In: Add a t-cell run path to `kinase_incytr_bridge.py`: extend the CLI (`--cohort` gains `tcells`;
  add `--donor {donor1,donor2}` — the current surface is `--cohort {song,fivexfad,all}` + `--tissue`
  only), per-donor scoping, cell-type attribution sourced from the **existing** within-cohort module
  (see below — NOT the song/5xFAD cross-reference annotators), and emission of motif terminal edges in
  the song-identical schema. Correct the generated MANIFEST note that falsely claims the t-cell cohort
  is excluded.
- Out: No new `gene_node_index` build — the t-cell index **already exists** (see curated specifics);
  consume it. No re-derivation of t-cell within-cohort attribution — reuse `tcell_within_cohort.py`
  (see curated specifics). No PSP / kinase→kinase work (subplan 01). No viewer payload (03). No change
  to the Incytr scorer, `gene.use` selection, or any scoring parameter — this bridge consumes Incytr output.

**Files touched:** `alz/cross_reference/kinase_incytr_bridge.py`.

**Risk tier:** **medium** — mostly mechanical: mirror the song/5xFAD motif path for a new cohort in a
byte-identical schema (a column-diff catches divergence loudly). The 10% detection floor — the original
silent-risk flag — is **already solved**: t-cell attribution lives in `tcell_within_cohort.py`, which
sources the floor from the single cross-cohort gate `specificity.DETECTION_FRAC_MIN` (not a local copy).
The real silent spot is now **using the wrong attribution model** (mirroring the cross-species
song/5xFAD annotator instead of reusing the within-cohort module) and the **donor2 coverage gap**
(the module is donor1-only). Both are called out below.

**Reuse (do not rebuild):**
- `build_substrate_bridge` in the same file — needs **no change**. VERIFIED against current code: song
  `mea_perdonor.csv` and t-cell `mea_timecourse.csv` headers are column-identical
  (`kinase,ES,NES,p-value,FDR,Subs fraction,Leading substrates,contrast,residue_type,track`), and the
  function keys on `kinase/contrast/track/NES/FDR/Leading substrates` — no donor/timepoint column.
  Per-donor scoping is a caller concern (point at the per-donor MEA dir); the function is scope-agnostic.
- `load_gene_node_index` — point it at the existing per-donor t-cell shards:
  `outputs/reports/tcell_viewer/edge_slices/incytr_pathways/{donor}__gene_node_index.json.gz`.
- **`alz/cross_reference/tcell_within_cohort.py`** — the t-cell cell-type attribution. This is the
  correct reuse target, NOT the song/5xFAD annotators. The song/5xFAD `annotate_celltype_match_*`
  functions match against an **external cross-species reference** (WMB/SEA-AD atlas); t-cells have no
  external reference — attribution is **within-cohort**, off the cohort's own paired scRNA. Different
  method by design — that is why it is a separate module. It already emits
  `unified_attribution_tcells.csv` / `tcell_enrichment.csv` under
  `outputs/reports/kinase_attribution_tcells/{donor}/` and is already consumed by the t-cell viewer
  (`slices_kinase.py`, `slices_audit.py`). Attribute the motif bridge's t-cell edges via this module's
  outputs; do NOT reimplement a cross-reference annotator for t-cells.

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
  `_build_incytr_gene_node_index` and writes the shards on disk at
  `outputs/reports/tcell_viewer/edge_slices/incytr_pathways/{donor}__gene_node_index.json.gz`. The
  source plan's "the real work is a t-cell gene_node_index" is **stale** — do NOT rebuild it; load the
  existing shards. (Trust the code over the plan here.)
- **The "format barrier" note to correct is generated code, not a static file.** It is emitted by
  `write_outputs` (`kinase_incytr_bridge.py` L1158-1162) and reads *"T-cell cohort excluded
  (mea_timecourse.csv format, different scoping)."* (regenerated into
  `outputs/reports/kinase_incytr_bridge/song/MANIFEST.md:25`). The note is **factually wrong** — the
  MEA format is identical (verified above) and scoping is a caller concern. Fix the generator string;
  the MANIFEST.md files are regenerated output — do not hand-edit them.
- **The 10% detection floor is already implemented — reuse it, do not re-derive.**
  `tcell_within_cohort.py` sets `DETECTION_FRAC_MIN = specificity.DETECTION_FRAC_MIN` (the single
  cross-cohort gate, explicitly "not a local copy"). Any attribution the bridge consumes must inherit
  this floor via that module; never hardcode `0.1`. A kinase expressed in <10% of a state's cells
  cannot carry attribution — this is a correctness gate, do not lower it to gain hits.
- **donor2 coverage gap (BLOCKING — resolve at dispatch).** `tcell_within_cohort.py` is **donor1-only**
  (docstring; `CONTRAST_DAYS = ("d13","d17","d20")` are donor1's days). donor2 (d5/d7/d9/d11) has no
  within-cohort attribution module yet. The plan requires both donors. Before running the donor2 arm,
  confirm whether donor2 attribution exists elsewhere on disk (derivable from donor2's paired scRNA via
  the same method) or must be produced first. Do NOT emit donor2 motif edges with donor1's attribution,
  and do NOT silently ship a donor1-only feature as "both donors". If donor2 attribution cannot be
  produced from artifacts on disk, report that as a finding (no collaborator contact) rather than
  faking coverage.
- **Reads streamed.** t-cell Incytr output lives under
  `outputs/reports/incytr_pair_mode_tcells/donor{1,2}/wide/` (7 shards: donor1 ×3, donor2 ×4) — stream
  via DuckDB, never whole-read.
- **Per-donor, not pooled.** donor1 and donor2 are scored separately (donor1 KsG-on, donor2 KsG-off in
  the regeneration runbook); keep the motif bridge per-donor, same as their pathway output.

**Acceptance / verify:**
- `pixi run kinase-incytr-bridge -- --cohort tcells --donor donor1` (and `donor2`) emits motif terminal
  edges with a schema **column-identical** to the song output (diff the column set), and prints per-donor
  hit counts.
- The regenerated MANIFEST no longer states the t-cell cohort is excluded.
- A cell-type-attributed kinase resolves to a state detected at ≥ `specificity.DETECTION_FRAC_MIN`
  within-cohort (spot-check one); the floor is inherited from `tcell_within_cohort.py`, not hardcoded.
- The donor2 arm either emits edges backed by real donor2 within-cohort attribution, or reports the
  donor2 attribution gap explicitly — it must NOT reuse donor1's attribution or ship a donor1-only
  result labelled as both donors.

**Declared dependencies:** none — parallel-safe (file-disjoint; extends existing song/5xFAD motif code
for a new cohort in the same schema). Contract with subplan 01: emit in the schema 01's terminal-map
builder accepts. Consumed by subplan 03 (t-cell arm).
