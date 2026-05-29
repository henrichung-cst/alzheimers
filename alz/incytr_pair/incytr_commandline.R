#!/usr/bin/env Rscript
# Pair-mode driver for the levy_t5 spine.
#
# Calls Incytr::Cal_pairwise_grid (upstream `Incytr` R package,
# ~/Projects/work/incytr/) ONCE PER PAIR with the pair's own tight gene
# filters (DG_sender ≈ 6K, DG_receiver ≈ 6K). The whole-grid-shared
# pathway_inference path was tried and abandoned: it requires
# gene.use_* = union over all 31 clusters ≈ 28K (essentially the whole
# transcriptome), which blew past the 30 GiB box on enumeration alone.
# Per-pair calls keep enumeration small (matching the historical
# per-pair runIncytr memory profile) while still routing through the
# upstream code paths that carry the sce4-parity defaults.
#
# The per-(gene, cluster, condition) trimean is pair-invariant, so it is
# computed ONCE over the union of all per-cluster gene.use sets and injected
# into every per-pair Cal_pairwise_grid call (expr_bygroup =), instead of
# recomputing Expr_bygroup 961x per contrast (Incytr perf Improvement 3).
#
# sce4 parity: five of the six historical sce4-calibration overrides are
# now upstream defaults (Cal_SigProb correction=0.01, cutoff_SigProb=0;
# Cal_scFC correction=0.01; Cal_PDS cutoff_PDS=0; Expr_bygroup mean_method
# = NULL trimean). Locked by tests/testthat/test-sce4_defaults.R in the
# Incytr package. The remaining two stay driver-side because they depend
# on AD-project inputs:
#   - DG no-cap (build_input_gene_list.R; DEG unioned with per-cluster prG
#     below — genes with |pr_log2FC|>1 in the cluster's deconvoluted proteome)
#   - pmax(pr_*, 1) floor on deconvoluted proteomics (just below)
# Receiver gene.use = DEG ∪ prG per cluster (replaces former full-proteome
# receiver). Investigation history: bench/bench.md A31–A33.
# Verification gate: `pixi run verify-incytr-sce4`.
#
# Usage (from any working directory):
#   Rscript alz/incytr_pair/incytr_commandline.R <condition1> <condition2> <input_gene_list.csv>

suppressPackageStartupMessages({
  library(Incytr)
  library(Seurat)
  library(readr)
  library(dplyr)
  library(arrow)
  library(data.table)
  library(parallel)
  library(DBI)
  library(duckdb)
})

REPO_ROOT  <- system("git rev-parse --show-toplevel", intern = TRUE)
# INPUTS_DIR can be overridden via env. The sce4 parity gate points this at
# the frozen 46-cluster provenance inputs (data/incytr_frozen/v2_46clusters);
# production runs use the derived levy_t5 31-cluster inputs (the default).
INPUTS_DIR <- Sys.getenv("INPUTS_DIR_OVERRIDE", unset = "")
if (!nzchar(INPUTS_DIR)) {
  INPUTS_DIR <- file.path(REPO_ROOT, "data", "derived", "incytr_inputs")
}
# OUTPUT_DIR can be overridden via env (used by smoke runs to write into a
# scratch directory without disturbing the canonical wide/ outputs).
OUTPUT_DIR <- Sys.getenv("OUTPUT_DIR_OVERRIDE", unset = "")
if (!nzchar(OUTPUT_DIR)) {
  OUTPUT_DIR <- file.path(REPO_ROOT, "outputs", "reports", "incytr_pair_mode", "wide")
}
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# =====================================================================
# Parallelism / memory config.
#
# The pair loop runs NPAIR_WORKERS pairs concurrently via mclapply over the
# 961-pair grid (each fork runs one single-pair Cal_pairwise_grid, writes its
# own shard, returns the path). Forks COW-inherit the post-rm(Data.input) base
# (~3 GB). Per audit 2026-05-27 Experiment A, in-fork gc() was REMOVED — it
# cost ~12% wall and ~400 MB cgroup memory.peak with no compensating reclaim;
# the parent still gc()s between scheduling waves. Measured on this 30 GB box:
# W=3 peaks ~13–15 GB (zero oom under a 24 G cgroup cap) at a near-linear
# 2.77× speedup. Whether the post-gc-removal footprint admits W=4 is pending
# Experiment B (parallel sweep re-run). ALWAYS run inside the cgroup scope in
# alz/incytr_pair/README.md. Permutation inside each pair stays single-core
# (perm.n.cores=NPERM_WORKERS, default 1) to avoid nested forking.
# See docs/plans/pairmode_perf_oom_2026-05-25.md and
# docs/plans/pairmode_memory_audit_2026-05-27.md.
# =====================================================================
N_PAIR_WORKERS <- as.integer(Sys.getenv("NPAIR_WORKERS", unset = "3"))
N_PERM_WORKERS <- as.integer(Sys.getenv("NPERM_WORKERS", unset = "1"))
PAIR_LIMIT     <- as.integer(Sys.getenv("PAIR_LIMIT",    unset = "0"))
NBOOT          <- as.integer(Sys.getenv("NBOOT",         unset = "100"))
stopifnot(!is.na(N_PAIR_WORKERS), N_PAIR_WORKERS >= 1L)
stopifnot(!is.na(N_PERM_WORKERS), N_PERM_WORKERS >= 1L)
stopifnot(!is.na(PAIR_LIMIT),     PAIR_LIMIT     >= 0L)
stopifnot(!is.na(NBOOT),          NBOOT          >= 0L)

Incytr::assert_core_budget(
  c(pairs = N_PAIR_WORKERS, perms = N_PERM_WORKERS),
  reserve = 2L
)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3L) {
  stop("Usage: incytr_commandline.R <condition1> <condition2> <input_gene_list.csv>",
       call. = FALSE)
}
condition1      <- args[1]
condition2      <- args[2]
input_gene_path <- args[3]

cat(sprintf("[pair-driver] contrast=%s vs %s  nboot=%d  pair_workers=%d  perm_workers=%d  pair_limit=%d\n",
            condition1, condition2, NBOOT, N_PAIR_WORKERS, N_PERM_WORKERS, PAIR_LIMIT))

# =====================================================================
# Inputs
# =====================================================================
# Species-specific Ligand/Receptor/EM/Target DBs. Mouse for the AD cohort,
# human for the T-cell cohort. Gene symbols in the substrate must match the
# DB's casing (mouse: title-case `App`; human: upper `APP`); a mismatch
# silently yields 0 enumerated paths.
SPECIES <- tolower(Sys.getenv("SPECIES", unset = "mouse"))
stopifnot(SPECIES %in% c("mouse", "human"))
DB.M <- if (SPECIES == "human") {
  list(Incytr::DB_Layer1_human_filtered,
       Incytr::DB_Layer2_human_filtered,
       Incytr::DB_Layer3_human_filtered)
} else {
  list(Incytr::DB_Layer1_mouse_filtered,
       Incytr::DB_Layer2_mouse_filtered,
       Incytr::DB_Layer3_mouse_filtered)
}
cat(sprintf("[pair-driver] species=%s\n", SPECIES))

Data.input <- readRDS(file.path(INPUTS_DIR, "incytr_obj.rds"))
Data.input@meta.data$Type <- Data.input@active.ident
# Mouse Seurat carries `Group` (= "ma_<age>_<geno>"); T-cell Seurat sets
# `condition` directly (= "d<day>") in build_tcells_seurat.R. Use Group when
# present, otherwise trust the builder's `condition`.
if (!is.null(Data.input@meta.data$Group)) {
  Data.input@meta.data$condition <- as.factor(Data.input@meta.data$Group)
} else if (!is.null(Data.input@meta.data$condition)) {
  Data.input@meta.data$condition <- as.factor(Data.input@meta.data$condition)
} else {
  stop("Seurat metadata has neither `Group` nor `condition`", call. = FALSE)
}

# Channel set + filenames + gene-key column are env-parameterized so the same
# driver runs both the mouse cohort (3-channel, "Gene Symbol" key) and the
# T-cell cohort (donor1 3-channel, donor2 2-channel — no IMAC — with
# `gene_symbol` key). Defaults reproduce the mouse path byte-exactly.
CHANNELS    <- strsplit(Sys.getenv("CHANNELS", unset = "pr,py,ps"), ",", fixed = TRUE)[[1]]
CHANNELS    <- trimws(CHANNELS[nzchar(CHANNELS)])
stopifnot(all(CHANNELS %in% c("pr", "py", "ps")))
stopifnot("pr" %in% CHANNELS)  # pr drives prG receiver gene-set — required
PR_FILE     <- Sys.getenv("PR_FILE", unset = "pr_yuyu_deconvoluted.csv")
PY_FILE     <- Sys.getenv("PY_FILE", unset = "py_yuyu_deconvoluted.csv")
PS_FILE     <- Sys.getenv("PS_FILE", unset = "ps_yuyu_deconvoluted.csv")
PR_GENE_COL <- Sys.getenv("PR_GENE_COL", unset = "Gene Symbol")
PY_GENE_COL <- Sys.getenv("PY_GENE_COL", unset = "gene_symbol")
PS_GENE_COL <- Sys.getenv("PS_GENE_COL", unset = "gene_symbol")
cat(sprintf("[pair-driver] channels=[%s]  pr_file=%s  pr_gene_col=%s\n",
            paste(CHANNELS, collapse = ","), PR_FILE, PR_GENE_COL))

pr <- read_csv(file.path(INPUTS_DIR, PR_FILE))
py <- if ("py" %in% CHANNELS) read_csv(file.path(INPUTS_DIR, PY_FILE)) else NULL
ps <- if ("ps" %in% CHANNELS) read_csv(file.path(INPUTS_DIR, PS_FILE)) else NULL

# Slice pr/ps/py into condition1 / condition2 wide-by-cluster tables.
# Each table is one row per gene_symbol, one column per cluster, suffixed
# `_pr` / `_ps` / `_py`. Mean-collapse duplicate gene rows.
#
# Column-name match is anchored: `^<condition>_` (NOT substring), so condition
# "d2" does not accidentally also pick up "d20_..." / "d13_..." columns on the
# T-cell cohort. Mouse columns "ma_2mo_AppP_<cluster>" still match cleanly.
slice_omics <- function(df, gene_col, condition, suffix) {
  pat <- paste0("^", condition, "_")
  out <- dplyr::select(df, matches(pat))
  colnames(out) <- paste0(sub(pat, "", colnames(out)), "_", suffix)
  out$gene_symbol <- df[[gene_col]]
  out %>% group_by(gene_symbol) %>% summarise_all(mean, na.rm = TRUE)
}
pr_1 <- slice_omics(pr, PR_GENE_COL, condition1, "pr")
pr_2 <- slice_omics(pr, PR_GENE_COL, condition2, "pr")
ps_1 <- if ("ps" %in% CHANNELS) slice_omics(ps, PS_GENE_COL, condition1, "ps") else NULL
ps_2 <- if ("ps" %in% CHANNELS) slice_omics(ps, PS_GENE_COL, condition2, "ps") else NULL
py_1 <- if ("py" %in% CHANNELS) slice_omics(py, PY_GENE_COL, condition1, "py") else NULL
py_2 <- if ("py" %in% CHANNELS) slice_omics(py, PY_GENE_COL, condition2, "py") else NULL

# sce4 reproduction (driver-side override #2): floor pr values < 1 to 1.
# pr_yuyu carries ~1e-5 deconvolution residuals where sce4-era values were
# hard zeros; without this Cal_foldchange's zero-check (which adds 1e-4
# only on exact zeros) never fires, producing 13-log2 outliers on
# one-sided genes (Olfm2, Fkbp15) vs sce4. Floor-to-1 matches sce4 within
# 0.015 log2 on every bug driver. Validated in bench/regen/.
floor_pr <- function(df) {
  num_cols <- setdiff(colnames(df), "gene_symbol")
  for (cc in num_cols) df[[cc]] <- pmax(df[[cc]], 1)
  df
}
pr_1 <- floor_pr(pr_1)
pr_2 <- floor_pr(pr_2)

# kldata (kinase-substrate library) is mouse-only for the AD cohort. For the
# T-cell cohort the kinase track lives on bulk (per meeting notes Stream D);
# pair-mode skips it via kldata=NULL. USE_KLDATA env controls.
USE_KLDATA <- as.logical(Sys.getenv("USE_KLDATA", unset = "TRUE"))
stopifnot(!is.na(USE_KLDATA))
if (USE_KLDATA) {
  kldata <- read_csv(file.path(INPUTS_DIR, "kldata.csv"))
  kldata <- kldata[, c("gene", "site_pos", "motif.geneName")]
} else {
  kldata <- NULL
  cat("[pair-driver] USE_KLDATA=FALSE  -> kinase scoring skipped (kldata=NULL)\n")
}
input_gene_list <- read_csv(input_gene_path)

# =====================================================================
# Gene-set construction (per-cluster, tight)
# =====================================================================
# sce4 reproduction (driver-side override #1, "DG no-cap"): per-cluster DG
# is the union of the input gene list's per-cluster DEGs with the
# proteomically-regulated gene set (prG): genes with |pr_log2FC| > 1 for
# that cluster.
#
# CRITICAL: prG must use the SAME pr_log2FC that the Incytr scorer computes,
# not a raw column ratio. Incytr::integrate_omics_layer (analysis.R) runs
# limma::normalizeBetweenArrays() on the cluster's [cond1, cond2] pr columns
# (quantile normalization across all genes) BEFORE Cal_foldchange. That
# re-maps each value by its within-column rank, so the raw deconvoluted
# ratio and the scored pr_log2FC differ substantially (e.g. Acvr1 in
# Cholinergic-Neurons: raw log2(50.74/28.22)=0.85 < 1, but quantile-
# normalized = ~2.0 > 1). Computing prG on the raw ratio dropped exactly the
# genes the scorer flags as pr-significant, which is why the Microglia->
# Cholinergic benchmark lost all 600 paths (Receptor overlap 0). prG now
# mirrors integrate_omics_layer term-for-term: normalize the floored pr
# columns, then |log2(c1_norm / c2_norm)| > 1. No correction term — floor_pr
# guarantees values >= 1 so Cal_foldchange's zero-correction never fires.
#
# This replaces the previous "full proteome" receiver (all pr genes with
# finite fold-magnitude after floor-to-1 ≈ all ~6k proteome genes) with
# a per-cluster tight set where proteomics is differentially regulated.
# Investigation history in bench/bench.md A31–A35.

# prG = proteomically-regulated genes per cluster: |pr_log2FC| > 1 where
# pr_log2FC is computed the way the Incytr scorer computes it — quantile
# normalization (limma::normalizeBetweenArrays) on the cluster's [cond1, cond2]
# pr columns then log2, NOT the raw column ratio. This gene-selection is method
# logic and lives in the package: Incytr::proteomics_gene(style="log2FC"). The
# AD app supplies only the floored pr tables, the strict |log2FC| > 1 cutoff
# (strict=TRUE), and the rownames intersect. floor_pr guarantees values >= 1, so
# Cal_foldchange's zero-correction never fires (pure log2 ratio). pr_1 / pr_2
# share columns named <cluster>_pr; clusters absent from pr get character(0).
clusters <- as.character(unique(Data.input@meta.data$Type))

pr_clusters <- clusters[vapply(clusters, function(cl) {
  col <- paste0(cl, "_pr")
  col %in% colnames(pr_1) && col %in% colnames(pr_2)
}, logical(1))]
pg <- if (length(pr_clusters)) {
  Incytr::proteomics_gene(
    as.data.frame(pr_1), as.data.frame(pr_2),
    cell_group = pr_clusters, style = "log2FC", cutoff = 1,
    pr.correction = 0.001, strict = TRUE
  )
} else NULL
prg_by_cluster <- lapply(setNames(clusters, clusters), function(cl) {
  if (is.null(pg)) return(character(0))
  intersect(pg$gene_symbol[pg$cluster == cl], rownames(Data.input))
})

deg_by_cluster <- lapply(setNames(clusters, clusters), function(cl) {
  intersect(unique(input_gene_list$gene[input_gene_list$cluster == cl]),
            rownames(Data.input))
})
# Per-cluster gene.use_* for the per-pair Cal_pairwise_grid calls.
# Each cluster gets DEG ∪ prG — the correct receiver for pair-mode Incytr.
dg_by_cluster <- lapply(setNames(clusters, clusters), function(cl) {
  union(deg_by_cluster[[cl]], prg_by_cluster[[cl]])
})
cat(sprintf("[pair-driver] per-cluster gene.use sizes: median=%d  min=%d  max=%d\n",
            as.integer(median(lengths(dg_by_cluster))),
            min(lengths(dg_by_cluster)),
            max(lengths(dg_by_cluster))))

# Every pair's pathway genes are a subset of (dg_sender u dg_receiver), so the
# union of all per-cluster DG sets bounds every gene any pair can ever score.
# The per-(gene, cluster, condition) trimean is pair-invariant, so we compute
# it ONCE over this union below and inject it into each per-pair grid call
# (Improvement 3), replacing 961 redundant Expr_bygroup passes per contrast.
gene_union <- intersect(unique(unlist(dg_by_cluster, use.names = FALSE)),
                        rownames(Data.input))
cat(sprintf("[pair-driver] precompute-trimean gene union: %d genes\n",
            length(gene_union)))

# =====================================================================
# Template object + pairs grid
# =====================================================================
# sender/receiver here are placeholders — Cal_pairwise_grid overwrites
# `obj@sender` / `obj@receiver` on each per-pair clone.
template <- create_Incytr(
  object     = GetAssayData(Data.input, assay = DefaultAssay(Data.input), layer = "data"),
  meta       = Data.input@meta.data,
  sender     = clusters[1],
  receiver   = clusters[1],
  group.by   = "Type",
  conditions = c(condition1, condition2),
  assay      = NULL,
  do.sparse  = TRUE
)

pairs <- expand.grid(sender = clusters, receiver = clusters,
                     stringsAsFactors = FALSE, KEEP.OUT.ATTRS = FALSE)

# PAIR_SUBSET: comma-separated `sender:receiver` list, taking precedence
# over PAIR_LIMIT. Used by smoke runs to hit exactly the parity-gate
# pairs without depending on cluster-ordering in Data.input.
pair_subset_env <- Sys.getenv("PAIR_SUBSET", unset = "")
if (nzchar(pair_subset_env)) {
  spec <- strsplit(pair_subset_env, ",", fixed = TRUE)[[1]]
  spec <- trimws(spec[nzchar(spec)])
  subset_df <- do.call(rbind, lapply(spec, function(p) {
    sr <- strsplit(p, ":", fixed = TRUE)[[1]]
    if (length(sr) != 2L) {
      stop(sprintf("PAIR_SUBSET entry '%s' is not 'sender:receiver'", p),
           call. = FALSE)
    }
    data.frame(sender = sr[1], receiver = sr[2], stringsAsFactors = FALSE)
  }))
  bad <- setdiff(unique(c(subset_df$sender, subset_df$receiver)), clusters)
  if (length(bad) > 0L) {
    stop("PAIR_SUBSET references unknown clusters: ",
         paste(bad, collapse = ", "), call. = FALSE)
  }
  pairs <- subset_df
  cat(sprintf("[pair-driver] PAIR_SUBSET active; running %d pairs\n",
              nrow(pairs)))
} else if (PAIR_LIMIT > 0L && PAIR_LIMIT < nrow(pairs)) {
  pairs <- pairs[seq_len(PAIR_LIMIT), , drop = FALSE]
  cat(sprintf("[pair-driver] PAIR_LIMIT active; running first %d pairs\n",
              nrow(pairs)))
}

multiomics_args <- list(
  pr.data_condition1 = pr_1, pr.data_condition2 = pr_2, pr.correction = 0.001, pr.q = NULL,
  ps.data_condition1 = ps_1, ps.data_condition2 = ps_2, ps.correction = 0.001, ps.q = NULL,
  py.data_condition1 = py_1, py.data_condition2 = py_2, py.correction = 0.001, py.q = NULL
)

# Condition->barcodes split must be captured before rm(Data.input) below.
cond_cells <- lapply(c(condition1, condition2), function(cc) {
  rownames(Data.input@meta.data)[
    as.character(Data.input@meta.data$condition) == cc]
})
names(cond_cells) <- c(condition1, condition2)

# =====================================================================
# Per-pair Cal_pairwise_grid loop
# =====================================================================
# Drop the redundant per-(node, omics) `_aFC` raw fold mirrors and the SiK
# sub-score breakdowns; the viewer keeps `<Node>_<omics>_log2FC` instead.
drop_pat <- "^(Ligand|Receptor|EM|Target)_(pr|ps|py)_aFC$|^SiK_(R|EM|T)_of_(EM|T|R)(_EI_.*)?$"
num_pat  <- "^(SigProb|p_value|SiK|log2FC|aFC|PDS|TPDS|PPDS|PhPDS|Ack_score|KGG_score|Rme1_score|multimodel_score|pr_|ps_|py_)"

# Shards persist across runs so an interrupted contrast resumes per-pair: a
# completed shard is reused, never recomputed. The dir is removed only after the
# final concat succeeds. A contrast that already finished is skipped upstream by
# the runner (its final parquet exists), so we only reach here for unfinished
# work. Shards are written atomically (tmp + rename) so a crash mid-write leaves
# either a complete shard or none — never a truncated one that resume would trust.
shard_dir <- file.path(OUTPUT_DIR, ".shards",
                       paste0(condition1, "_", condition2))
dir.create(shard_dir, recursive = TRUE, showWarnings = FALSE)

# Per-pair DEG/prG node labels. Pair-mode Export_results does not track
# per-node provenance; the upstream factorial path emitted these via
# assign_path_labels().
label_node <- function(node_genes, cluster) {
  deg <- deg_by_cluster[[cluster]]
  prg <- prg_by_cluster[[cluster]]
  set <- union(deg, prg)
  lbl <- setNames(rep("prG", length(set)), set)
  lbl[deg] <- "DEG"
  unname(lbl[node_genes])
}

rss_mb <- function() {
  s <- "/proc/self/status"; if (!file.exists(s)) return(NA_real_)
  v <- grep("^VmRSS:", readLines(s, warn = FALSE), value = TRUE)
  if (!length(v)) NA_real_ else
    as.numeric(sub("^VmRSS:\\s+([0-9]+)\\s+kB$", "\\1", v)) / 1024
}
hwm_mb <- function() {
  s <- "/proc/self/status"; if (!file.exists(s)) return(NA_real_)
  v <- grep("^VmHWM:", readLines(s, warn = FALSE), value = TRUE)
  if (!length(v)) NA_real_ else
    as.numeric(sub("^VmHWM:\\s+([0-9]+)\\s+kB$", "\\1", v)) / 1024
}

# Free the redundant full Seurat object before the pair loop. Everything the
# loop reads — `template`, `dg_by_cluster`, `deg/prg_by_cluster`, `pr/ps/py_*`,
# `kldata` — is already built; Data.input's counts + scale.data + meta are dead
# weight (~1.2 GB resident) that every Permutation_test fork would COW-inherit.
# The expression matrix stays alive via `template` (shared refcount), so this
# cannot change any computed value (verified byte-identical, max |Δ|=0 across
# all numeric columns on both sce4 benchmark pairs).
cat(sprintf("[pair-driver] rss before rm(Data.input): %.0f MB\n", rss_mb()))
rm(Data.input); gc(verbose = FALSE)
cat(sprintf("[pair-driver] rss after  rm(Data.input): %.0f MB\n", rss_mb()))

# Build the pair-invariant trimean substrate once (Improvement 3): the per-(gene,
# cluster, condition) trimean is pair-invariant, so it is computed ONCE over
# gene_union and injected into every per-pair Cal_pairwise_grid call
# (expr_bygroup =) instead of recomputing Expr_bygroup 961x. This is the method's
# Expr_bygroup, batched over an explicit gene set — it lives in the package
# (Incytr::precompute_expr_bygroup), gene-chunked to bound the dense submatrix.
# template@data shares the expression matrix (refcount) with the dropped
# Data.input, so this reads the same values. Forks below COW-inherit it read-only.
t_sub <- proc.time()[["elapsed"]]
expr_substrate <- Incytr::precompute_expr_bygroup(template@data, template@idents,
                                                  cond_cells, gene_union)
cat(sprintf("[pair-driver] built expr substrate in %.1fs  rss=%.0fMB\n",
            proc.time()[["elapsed"]] - t_sub, rss_mb()))

# One pair: run Cal_pairwise_grid, post-process, write its shard, return the
# shard path (or NA if the pair produced no rows / errored). Self-contained so
# it runs identically in-process (NPAIR_WORKERS=1) or in an mclapply fork.
process_pair <- function(i) {
  s <- pairs$sender[i]; r <- pairs$receiver[i]
  pair_df <- data.frame(sender = s, receiver = r, stringsAsFactors = FALSE)
  sub_path <- file.path(shard_dir, sprintf("pair_%04d.parquet", i))
  if (file.exists(sub_path) && file.info(sub_path)$size > 0) {
    cat(sprintf("[pair-driver] pair %d/%d  %s -> %s  RESUME (shard exists)\n",
                i, nrow(pairs), s, r))
    return(sub_path)
  }
  t_one <- proc.time()[["elapsed"]]
  res <- tryCatch(
    Incytr::Cal_pairwise_grid(
      template          = template,
      pairs             = pair_df,
      DB                = DB.M,
      gene.use_Sender   = dg_by_cluster[[s]],
      gene.use_Receiver = dg_by_cluster[[r]],
      multiomics        = multiomics_args,
      kldata            = kldata,
      mean_method       = NULL,
      fold_threshold    = 10,
      n.cores           = 1L,
      nboot             = NBOOT,
      seed.use          = 1L,
      perm.n.cores      = N_PERM_WORKERS,
      expr_bygroup      = expr_substrate
    ),
    error = function(e) {
      warning(sprintf("[pair-driver] Cal_pairwise_grid failed for %s -> %s: %s",
                      s, r, conditionMessage(e)), call. = FALSE)
      NULL
    }
  )
  dt_sec <- proc.time()[["elapsed"]] - t_one
  obj <- if (!is.null(res)) res[[paste(s, r, sep = "__")]] else NULL
  out <- if (!is.null(obj)) tryCatch(Export_results(obj, indicator = TRUE),
                                     error = function(e) NULL) else NULL
  # No in-fork gc() — see audit 2026-05-27 Experiment A. `rm()` releases the
  # bindings; fork teardown returns the pages; an explicit gc() here dirties
  # COW shared pages with no compensating reclaim (measured -3.6% cgroup
  # memory.peak and -12% wall on W=2 after removal).
  rm(res, obj)

  if (is.null(out) || nrow(out) == 0L) {
    cat(sprintf("[pair-driver] pair %d/%d  %s -> %s  rows=0  %.1fs  rss=%.0fMB\n",
                i, nrow(pairs), s, r, dt_sec, rss_mb()))
    return(NA_character_)
  }
  out$Sender   <- s
  out$Receiver <- r
  out$Ligand.label   <- label_node(out$Ligand,   s)
  out$Receptor.label <- label_node(out$Receptor, r)
  out$EM.label       <- label_node(out$EM,       r)
  out$Target.label   <- label_node(out$Target,   r)
  out <- out[, !grepl(drop_pat, colnames(out)), drop = FALSE]
  dt <- as.data.table(out)
  for (col in grep(num_pat, names(dt), value = TRUE)) {
    if (is.character(dt[[col]])) {
      set(dt, j = col, value = suppressWarnings(as.numeric(dt[[col]])))
    }
  }
  tmp_path <- paste0(sub_path, ".tmp")
  arrow::write_parquet(as.data.frame(dt), tmp_path, compression = "zstd")
  file.rename(tmp_path, sub_path)
  cat(sprintf("[pair-driver] pair %d/%d  %s -> %s  rows=%d  %.1fs  rss=%.0fMB  hwm=%.0fMB\n",
              i, nrow(pairs), s, r, nrow(out), dt_sec, rss_mb(), hwm_mb()))
  # No in-fork gc() here either — see Experiment A note above.
  rm(out, dt)
  sub_path
}

t_loop <- proc.time()[["elapsed"]]
results <- if (N_PAIR_WORKERS > 1L) {
  mclapply(seq_len(nrow(pairs)), process_pair,
           mc.cores = N_PAIR_WORKERS, mc.preschedule = FALSE)
} else {
  lapply(seq_len(nrow(pairs)), process_pair)
}
shard_paths <- unlist(results, use.names = FALSE)
shard_paths <- shard_paths[!is.na(shard_paths)]
cat(sprintf("[pair-driver] pair loop done in %.1f min (%d shards)\n",
            (proc.time()[["elapsed"]] - t_loop) / 60, length(shard_paths)))

if (length(shard_paths) == 0L) {
  stop("[pair-driver] no shards produced", call. = FALSE)
}

# =====================================================================
# Concat
# =====================================================================
out_path <- file.path(OUTPUT_DIR,
                      paste0(condition1, "_", condition2, "_incytr_output.parquet"))
# Stream the per-pair shards into one parquet via DuckDB. dplyr::collect() here
# materialized the full combined table (~all 961 pairs) in RAM before writing,
# which blew the 24G cgroup cap on the final concat (loop itself peaks ~5GB).
# DuckDB streams the read_parquet glob to disk and spills to DUCKDB_TEMP_DIR,
# staying well under the cap. union_by_name handles the per-shard schema
# differences that unify_schemas used to absorb.
tmp_out <- paste0(out_path, ".tmp")
con <- DBI::dbConnect(duckdb::duckdb())
DBI::dbExecute(con, "PRAGMA memory_limit='10GB'")
DBI::dbExecute(con, sprintf("PRAGMA temp_directory='%s'",
                            Sys.getenv("DUCKDB_TEMP_DIR",
                                       file.path(Sys.getenv("HOME"), ".cache/duckdb"))))
DBI::dbExecute(con, sprintf(
  "COPY (SELECT * FROM read_parquet([%s], union_by_name=true)) TO '%s' (FORMAT PARQUET, COMPRESSION ZSTD)",
  paste(sprintf("'%s'", shard_paths), collapse = ", "), tmp_out))
DBI::dbDisconnect(con, shutdown = TRUE)
file.rename(tmp_out, out_path)
sz <- file.info(out_path)$size
cat(sprintf("[pair-driver] wrote %s (%.1f MB)\n", out_path, sz / 1e6))
unlink(shard_dir, recursive = TRUE)
