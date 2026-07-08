# kinase_sidechain_incytr_graph — subplan 01: backend edge model (PSP + kinase→kinase)

**Goal:** Produce, per cohort, a weighted provenance-tagged kinase→kinase interactome plus a
gene→terminal-kinase-edges map, unifying literature (PhosphoSitePlus) with the existing motif
(MEA/kldata) evidence — the backend the viewer walks to draw one pathway's kinase sidechains.

**Scope**
- In: A new module that loads the PSP `Kinase_Substrate_Dataset`, homology-maps it to mouse for song +
  5xFAD, builds the kinase→kinase interactome (cycle-safe closure input) and the per-node terminal-edge
  map, tags every edge `motif` / `psp` / `both`, and computes the normalized-additive weight
  (literature `log1p(in_vivo_refs)`→[0,1] + contrast-specific motif corroboration→[0,1], corroboration
  bonus for `both`). Runs and emits artifacts for **song and 5xFAD**.
- Out: No t-cell run (subplan 02 produces t-cell motif edges; this module's terminal-map builder must
  *accept* them as input so 03 can assemble the t-cell map). No viewer payload shaping (03). No edit to
  `kinase_incytr_bridge.py` — import its helpers read-only. **No change to the Incytr scorer, the 4-node
  enumeration, or any sce4 parity constant** — this feature consumes Incytr output, it never rescores.

**Files touched:** `alz/cross_reference/kinase_kinase_edges.py` (new). Read-only imports:
`alz/cross_reference/kinase_incytr_bridge.py` (`load_gene_node_index`, `build_substrate_bridge`,
cell-type match), `alz/integration/build_yuyu_kldata.py` (homology pattern — copy, don't edit).

**Risk tier:** **high** — the normalized-additive weight, the `motif`/`psp`/`both` provenance tag, and
the homology mapping all produce edges that *look plausible when wrong* (a dropped normalization, an
`in_vitro`↔`in_vivo` swap, a mis-mapped ortholog). Nothing downstream flags a wrong weight — it renders
as a believable edge thickness. Silent-failure surface; needs the strongest model.

**Reuse (do not rebuild):**
- `kinase_incytr_bridge.py` — the motif terminal-edge builder + `celltype_match` already exist for
  song/5xFAD; import them, do not reimplement.
- `alz/integration/build_yuyu_kldata.py` — homologene human→mouse mapping; mirror it exactly.
- `data/derived/caches/kinase_to_gene_mapping.csv` (+ overrides) — kinase-abbrev↔gene reconciliation
  for the dual-role node collapse. Read it; do not build a new mapping.
- PSP dataset (read-only package file, do NOT vendor into the repo):
  `.pixi/envs/default/lib/python3.11/site-packages/kinase_library/databases/substrates/Kinase_Substrate_Dataset_count_07_2021.txt`

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
- **Reads streamed, always.** The motif component comes from Incytr `wide/*.parquet` (14M-row scale).
  `kinase_incytr_bridge.py` already streams these via DuckDB with a `memory_limit` param
  (`build_recep_em_fan_from_parquet`, `write_fivexfad_streamed`) — follow that pattern; never
  `pd.read_parquet` a wide shard.
- **Homology direction is cohort-specific.** PSP is human. song + 5xFAD are mouse → homology-map
  human→mouse (homologene, exactly as `build_yuyu_kldata.py`). t-cells are human → direct, no map
  (that arm is subplan 02's producer + 03's assembly; this module's builder must be cohort-parameterized
  so the human/mouse switch is a caller argument, not a hardcode).
- **Rank/weight on `|PDS|`, never `p_value_*`.** The Incytr pair pvalue is untrustworthy
  (`nboot=100`, informational only). The motif corroboration strength must derive from `|PDS|` /
  SigProb, not the p-value columns.
- **Dual-role node collapse** (a gene that is both a pathway node and a kinase → one node) uses
  `kinase_to_gene_mapping.csv` + its overrides for abbrev↔gene reconciliation. Drop PSP
  autophosphorylation self-loops.
- **Closure must be cycle-safe.** The kinase→kinase graph has cycles; the interactome is emitted as an
  edge list (the client walks it), so the artifact itself needn't close — but any closure you compute
  for verification must terminate on cyclic input.

**Acceptance / verify:**
- `pixi run python -m alz.cross_reference.kinase_kinase_edges --cohort song` (and `--cohort fivexfad`
  per tissue) emits the per-cohort backend artifact (kinase→kinase interactome + terminal-edge map) and
  **prints edge/node counts and the motif/psp/both breakdown**.
- A known cascade (e.g. MAP3K→MAP2K→MAPK terminating on a pathway EM/Target node) is present as a
  multi-hop chain in the song interactome; spot-check three edges' provenance tag and weight against
  the PSP file (`in_vivo`/`in_vitro` ref counts) and the motif source.
- Cyclic-graph closure terminates (a unit-level cycle-safety check on a synthetic cyclic edge set).

**Declared dependencies:** none — parallel-safe (file-disjoint, new module). Contract with subplan 02:
the terminal-map builder here must accept a motif-edge frame as input so 02's t-cell motif output feeds
the same builder in subplan 03. Consumed by subplan 03.
