# T-cell proteomics vs. transcriptomics correlation (donor2, day2)

**COMPLETED 2026-07-13.** Built `alz/analysis/tcell_proteome_transcriptome_correlation.py`; output under `outputs/reports/tcell_donor2_day2_protein_rna_correlation/` (`per_gene_correlation.csv` + scatterplot + `analysis.md`). Result: Spearman **rho = 0.546, n = 7693** matched genes (24 zero-RNA). Both silent-failure guards verified against raw data — protein column traces to `Day 2 Total Quantity`, semicolon multi-gene rows dropped, plus 83 duplicate single-gene symbols resolved by highest precursor support (a case the plan did not anticipate).

**Goal:** One genome-wide Spearman correlation at donor2/day2 — across all matched genes, x = bulk protein abundance, y = pseudobulked transcript abundance — plus a standalone per-gene CSV.

**Scope**
- In: a single standalone analysis script producing a per-gene CSV (`gene, protein_abundance, pseudobulk_transcript_abundance, rank`) and reporting the overall Spearman rho / p-value / n alongside. Exploratory — **not** wired into the viewer.
  - Donor2 only, day2 only (control / pre-exhaustion timepoint; no cross-donor, no exhaustion-timepoint effects).
  - Transcript pseudobulk: aggregate **all day2 cells directly from the raw donor2 Seurat object**, independent of any cell-state assignment. Do **NOT** use the state-grouped `aggexp_data.csv` — so the result does not depend on any ProjecTILs labeling.
- Out (hard boundary): any cell-state grouping, ProjecTILs labels, the T-cell label-validation outputs, the viewer, and donor1 / other days.

**Files touched (created):**
- `alz/analysis/tcell_proteome_transcriptome_correlation.py` (new — the whole thread; may shell to R or read the `.rds` via a Seurat/pyreadr path for pseudobulk, implementer's call)
- Output (new): a per-gene CSV under `outputs/reports/` (e.g. `outputs/reports/tcell_donor2_day2_protein_rna_correlation.csv`)

Reading the raw donor2 `.rds` is **read-only**.

**Risk tier:** high. The output is a single correlation coefficient. Its correctness hinges entirely on two silent join/selection decisions — matching `PG.Genes` to transcript symbols, and picking the **correct day2 protein column** out of five day columns. A wrong column or a subtly wrong gene join yields a *plausible* rho that nobody catches downstream. Fails silently, not loudly → high.

**Reuse (do not rebuild):** the raw donor2 Seurat object is the same `.rds` used by `alz/incytr_pair/build_tcells_seurat.R` (`data/datasets/tcells/donor2/scrna/Tcells_d2.singlet (1).rds`) — **read `alz/incytr_pair/build_tcells_seurat.R` first** and reuse that read pattern (`readRDS`, `DefaultAssay <- "RNA"`, `DietSeurat`) for the pseudobulk, but aggregate over **all** day2 cells with no state idents. Do not reinvent the load.

**Memory cap (mandatory — the `.rds` is 4.7 GB on disk and expands to ~15–30 GB in R):** `readRDS` on this object is the step that can OOM the shared box. Run the pseudobulk step under a hard cap — `systemd-run --user --scope -p MemoryMax=40G -p MemorySwapMax=0 pixi run …` — and `DietSeurat` to the RNA assay only *before* aggregating, so the peak stays bounded. Do not load the object without the cap.

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
- **Proteome file is bulk, one scalar per gene per day — no replicates.** A per-gene correlation "across cells" is impossible; this is *one* genome-wide correlation at day2. File: `data/datasets/tcells/donor2/proteomics/10Feb2026_Donor2_TotalProteome_ForPerseus.txt`, tab-separated, 13 columns, ~7871 gene rows.
- **Column selection (the silent-failure guard — get this exactly right):** the 13 columns are `PG.Genes`, `PG.ProteinDescriptions`, `PG.NrOfPrecursorsIdentified (Experiment-wide)`, five `…RunEvidenceCount` columns (D2/D5/D7/D9/D11 — these are precursor counts, **NOT** the abundance), then five `Day N Total Quantity` columns. Use **`Day 2 Total Quantity`** as the x (protein abundance). Do not use the `RunEvidenceCount` columns and do not use any other day.
- **Gene matching:** `PG.Genes` vs. transcript gene symbol. **Drop** multi-gene protein-group rows (semicolon-containing `PG.Genes`, e.g. `MAP1LC3B2;MAP1LC3B`) — ~31/7871 (~0.4%) — rather than splitting/duplicating. Symbols are human uppercase on both sides; direct match, no species mapping.
- **Statistic: Spearman (rank correlation) on RAW (non-log) values** — chosen precisely because bulk protein abundance and scRNA pseudobulk counts have very different distributions/dynamic ranges; do not log-transform, do not switch to Pearson.
- **Pseudobulk:** sum (or mean) transcript counts across **all day2 cells** of the raw donor2 object, RNA assay, one value per gene — no state/cluster grouping (this is the decoupling requirement, not an optimization).

**Acceptance / verify:** `pixi run python alz/analysis/tcell_proteome_transcriptome_correlation.py` writes the per-gene CSV with columns `gene, protein_abundance, pseudobulk_transcript_abundance, rank` and prints Spearman `rho`, `p_value`, `n`. Confirm: (a) `n` ≈ matched genes (thousands, well below 7871 after intersect + multi-gene drop), (b) no semicolon-containing gene symbols survive in the CSV, (c) the protein column traces to `Day 2 Total Quantity` (spot-check one gene's value against the raw file).

**Declared dependencies:** none.
