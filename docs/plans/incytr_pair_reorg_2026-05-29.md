# `alz/incytr_pair/` cleanup audit — 2026-05-29

Same folder-by-folder pass as `cross_reference` (84be18d) and `decomposition_mea`
(this branch). Two lenses: (1) dead-code / liveness, (2) **separation of concerns
between `alz/incytr_pair/` (the AD application) and `../incytr/` (the public method
package we own and release)**. The second is the priority for this folder.

The rule: anything intrinsic to the Incytr *method* must be CALLED from the
`Incytr` package, not copied / reimplemented / shadowed inside the application.
Only data-specific glue (AD I/O, paths, contrast loops, call-site parameter
overrides for sce4 parity) legitimately lives here.

---

## 1. Liveness map

| File | Status | Caller(s) | Role |
|---|---|---|---|
| `__init__.py` | LIVE | 4 `alz/integration/*` import `pair_to_receiver_cache` | package marker |
| `build_pair_inputs.sh` | LIVE | `run_all.sh:125`, `run_pair_mode_pipeline.sh:151` | input-prep orchestrator |
| `build_pair_seurat.R` | LIVE | `build_pair_inputs.sh:62` | builds `incytr_obj.rds` |
| `build_input_gene_list.R` | LIVE | `build_pair_inputs.sh:70` | builds `input_gene_list.csv` |
| `build_tcells_seurat.R` | LIVE | pixi `tcells-build-incytr-seurat` | T-cell Seurat objects |
| `build_tcells_input_gene_list.R` | LIVE | pixi `tcells-build-input-gene-list` | T-cell gene list |
| `run_pair_mode.sh` | LIVE | `run_all.sh:126`, `run_pair_mode_pipeline.sh:107` | 9-contrast loop |
| `run_pair_mode_tcells.sh` | LIVE | pixi `tcells-incytr` | T-cell contrast loop |
| `incytr_commandline.R` | LIVE | `run_pair_mode.sh:80`, `run_pair_mode_tcells.sh:83` | core driver -> `Cal_pairwise_grid` |
| `emit_expr_bygroup.R` | LIVE | `run_pair_mode_pipeline.sh:159` | transcript-trace substrate |
| `export_decomposition_for_pair.py` | LIVE | `build_pair_inputs.sh:66` + 2 bench scripts | provenance deconvolution |
| `filter_significant_paths.py` | LIVE | `run_pair_mode.sh:125`, `run_pair_mode_tcells.sh:120` | SigProb/PDS row filter |
| `pair_to_receiver_cache.py` | LIVE | `run_all.sh:127`, viewer build; `_sanitize_celltype` imported x4 | reshape for viewer |
| `verify_sce4_parity.py` | LIVE | `run_pair_mode_pipeline.sh:165`, pixi `verify-incytr-sce4` | parity regression gate |
| `reconstruct_labels.R` | **DEAD (deleted)** | only an existence-check at `run_pair_mode_pipeline.sh:139` — never invoked | one-time backfill of `*.label` cols |
| `reconstruct_node_fc.R` | **DEAD (deleted)** | only an existence-check at `run_pair_mode_pipeline.sh:139` — never invoked | one-time backfill of `<Node>_*_log2FC` cols |
| `README.md` | docs | CLAUDE.md, integration docs | module docs |

**Dead-code finding:** the two `reconstruct_*` scripts are one-time backfills that
patched parquets written *before* the driver began writing labels / node fold-change
columns inline (`incytr_commandline.R:387-394`, and via `Cal_scFC`/`Integr_multiomics`
in the grid call). Nothing invokes them; the only reference was the stale
existence-check loop at `run_pair_mode_pipeline.sh:139-145`.

---

## 2. Separation-of-concerns audit (priority)

The driver attaches the package properly (`library(Incytr)` + `Incytr::` /
`Incytr:::` qualification, no `source()` of package R files). Python files
reimplement **no** scoring math — all clean. The leaks are all in the R layer.

| ID | Local code | Nearest package fn | Note |
|---|---|---|---|
| A3 | `find_highexp_vec` — `build_input_gene_list.R:47-69` | `Find_highexp_gene` (`utils.R:41`) | comment admits it "Mirrors" it |
| A2 | `prg_by_cluster` — `incytr_commandline.R:242-254` | `proteomics_gene` (`utils.R:260`) | the A33/A36 receiver gene.use pipeline |
| A1 | `build_expr_substrate` — `incytr_commandline.R:340-357` | `Expr_bygroup` (`analysis.R:251`) | reaches non-exported `:::` kernel; perf precompute |
| A4/A5 | fold-change helpers in the two `reconstruct_*` scripts | `Cal_foldchange`/`Cal_scFC` | vanish with the §1 deletions |

Legitimate (no action): `floor_pr` (`pmax(.,1)`, override #2 — no package fn floors
pr); all call-site overrides (`mean_method=NULL`, `correction=0.01`, `cutoff_*=0`,
`pr.correction=0.001`, `fold_threshold=10`); `slice_omics`, input loading,
`dg_by_cluster`, `label_node`, `process_pair`, shard concat, RSS monitors.

---

## 3. The leaks are divergent reimplementations, not faithful mirrors

Reading package + local together: each local copy **diverges** from the nearest
package function in a parity-relevant way, so a blind reroute would change
`input_gene_list.csv` / `gene.use` and risk the documented sce4 parity. The correct
fix is to add a **faithful** package function (or param) reproducing the current app
behavior exactly, then call it. Confirmed:

- **A3** — trimean is **identical** to the package kernel (`grouped_quartile.cpp`
  and local `matrixStats::rowQuantiles` are both `quantile(type=7)` ->
  `0.25*Q1+0.5*Q2+0.25*Q3`). The divergence from `Find_highexp_gene` is the **cutoff
  scope**: local takes the 75th pct of nonzero entries over the **whole condition
  matrix** (one cutoff for all clusters); `Find_highexp_gene` takes it **per-cluster**
  (`utils.R:79,102`). The global-cutoff convention is what the sce4 provenance used.
- **A2** — `normalizeBetweenArrays` + `log2(c1/c2)` is **identical** to
  `proteomics_gene` once `floor_pr` removes zeros (`Cal_foldchange` only adds
  `correction` when a zero is present — `math.R:39-50`). Only divergence: local
  `> 1` (strict) vs `proteomics_gene`'s `>= cutoff`.
- **A1** — already calls the kernel (`Incytr:::grouped_weighted_quartile`); it only
  **duplicates `Expr_bygroup`'s chunking loop** and reaches a non-exported `:::`
  internal. No math diverges.

---

## 4. Pass B design — faithful package functions + call-site swaps

Resolve all three by adding parity-preserving surface to `../incytr` (the method),
then calling it from the app. Existing public behavior preserved (new params default
to current behavior; new functions are additive).

**Package (`../incytr`):**
1. `utils.R` — export `Find_highexp_gene_batch(data, group_labels,
   cutoff_percentile = 0.75, cutoff_scope = c("global","cluster"),
   mean_method = NULL)`. Cutoff once over `mat[mat>0]` for `cutoff_scope="global"`
   (sce4 convention); per-group trimean via `grouped_weighted_quartile`; returns
   `data.frame(gene_symbol, ave.exp, cluster)`. Reproduces `find_highexp_vec` (A3).
2. Export `precompute_expr_bygroup(data, idents, cond_cells, genes,
   mean_method = NULL, gene_chunk = 4000L)` — batch variant of `Expr_bygroup` over an
   explicit `gene_union`; returns the per-condition `expr.bygroup` list the driver
   injects via `expr_bygroup=`. Reuses the kernel -> byte-identical to
   `build_expr_substrate` (A1); retires the app's `:::` reach.
3. `proteomics_gene` — add `strict = FALSE` param; `strict=TRUE` uses `> cutoff`.
   Default preserves public behavior (A2).
4. Regenerate `NAMESPACE` + `man/` (roxygen if available, else hand-edit export
   lines + minimal `.Rd`). Rebuild via `pixi run install-incytr`.

**App (`alz/incytr_pair`):**
5. `build_input_gene_list.R` — delete `find_highexp_vec`; call
   `Incytr::Find_highexp_gene_batch(...)` with `cutoff_scope="global"`.
6. `incytr_commandline.R` — replace `prg_by_cluster` inline block with
   `Incytr::proteomics_gene(pr_1, pr_2, cell_group=clusters, style="log2FC",
   cutoff=1, pr.correction=0.001, strict=TRUE)` (pr already floored), split to
   per-cluster vectors intersected with `rownames(Data.input)`. Replace
   `build_expr_substrate` def + call with `Incytr::precompute_expr_bygroup(...)`.

**Verification (parity is the gate):**
7. Targeted parity probes on the real inputs: old-local-fn vs new-package-fn must be
   identical for HEG (A3), prG per cluster (A2), expr substrate (A1).
8. End-to-end: rebuild package, regenerate the two benchmark pairs (`PAIR_SUBSET`,
   `NBOOT=0`, scratch `OUTPUT_DIR_OVERRIDE`), then `verify_sce4_parity.py` must still
   give 573/600 (App residual) + max |Δ sclog2FC| = 0 on R/E/T. The ~180 h full
   9-contrast run is NOT required and is not run.

**Then:** refresh `README.md` (final clean-separation state), sweep docs referencing
the deleted scripts (archive docs stay — historical).

---

## 5. Status — both passes DONE

**Pass A** — `reconstruct_labels.R` + `reconstruct_node_fc.R` deleted;
`run_pair_mode_pipeline.sh:139` guard fixed.

**Pass B** — all three leaks resolved, byte-identical:
- Package (`../incytr`): added `Find_highexp_gene_batch` + `precompute_expr_bygroup`
  (exported in `NAMESPACE`), `strict` param on `proteomics_gene`; rebuilt via
  `pixi run install-incytr`.
- App: `build_input_gene_list.R` calls `Find_highexp_gene_batch` (A3, dropped dead
  `matrixStats` import); `incytr_commandline.R` calls `proteomics_gene(strict=TRUE)`
  (A2) and `precompute_expr_bygroup` (A1, retired the `:::` reach). A4/A5 gone with
  the deleted scripts.
- **Verification** (probe on real App_2mo inputs, old-fn vs new-fn): A2 prG
  `setequal` across all 31 clusters (54,121 genes); A3 HEG `max|Δ ave.exp|=0`, keys
  identical; A1 expr substrate `identical`. Byte-identity → sce4 parity preserved by
  construction (existing wide parquets unchanged). Both edited R scripts parse-check
  clean. Full 9-contrast regen not required and not run.

**Docs**: README rewritten (separation contract + corrected inventory); CLAUDE.md
relocation note + corrected pointers; `kinase_incytr_integration.md` and
`build_unified_viewer.py` comment swept of the deleted scripts.
