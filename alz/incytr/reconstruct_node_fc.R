#!/usr/bin/env Rscript
# Reconstruct per-node log2 fold-change columns for existing pair-mode
# parquets:
#   <Node>_sclog2FC  -- single-cell RNA (Data.input@assays$originalexp@data)
#   <Node>_pr_log2FC -- deconvoluted proteome (pr_yuyu_deconvoluted.csv)
#   <Node>_ps_log2FC -- deconvoluted phospho-Ser/Thr (ps_yuyu_deconvoluted.csv)
#   <Node>_py_log2FC -- deconvoluted phospho-Tyr      (py_yuyu_deconvoluted.csv)
# Node in {Ligand, Receptor, EM, Target}, role mapped to cluster:
#   Ligand        → Sender   cluster
#   Receptor, EM, Target → Receiver cluster
#
# Conventions matched against alz/incytr/incytr_commandline.R:
#   * gene-level rollup for ps/py is mean across sites with the same
#     gene_symbol (Integr_multiomics does the same group_by + mean)
#   * additive pseudo-count = 0.001 (driver's pr.correction / ps.correction
#     / py.correction value)
#   * sc per-cluster, per-condition mean is computed on expm1 of the
#     log-normalized natural-log @data slot, then re-logged base 2
#
# Outputs the 16 columns alongside the existing schema; idempotent (skips
# parquets that already carry them).
#
# Usage (from any working directory):
#   pixi run Rscript alz/incytr/reconstruct_node_fc.R          # default: outputs/reports/incytr_pair_mode/wide/
#   pixi run Rscript alz/incytr/reconstruct_node_fc.R <dir>

suppressPackageStartupMessages({
  library(Seurat)
  library(readr)
  library(dplyr)
  library(arrow)
  library(Matrix)
})

# Resolve repo root so the script runs from any cwd.
REPO_ROOT <- system("git rev-parse --show-toplevel", intern = TRUE)
INPUTS_DIR <- file.path(REPO_ROOT, "data", "derived", "incytr_inputs")

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1L) args[1] else file.path(REPO_ROOT, "outputs", "reports", "incytr_pair_mode", "wide")
EPS <- 0.001

# --- Shared inputs ---------------------------------------------------------
cat("[node_fc] loading inputs\n")
Data.input <- readRDS(file.path(INPUTS_DIR, "incytr_obj.rds"))
Data.input@meta.data$Type <- as.character(Data.input@active.ident)
Data.input@meta.data$condition <- as.character(Data.input@meta.data$Group)

pr <- read_csv(file.path(INPUTS_DIR, "pr_yuyu_deconvoluted.csv"), show_col_types = FALSE)
ps <- read_csv(file.path(INPUTS_DIR, "ps_yuyu_deconvoluted.csv"), show_col_types = FALSE)
py <- read_csv(file.path(INPUTS_DIR, "py_yuyu_deconvoluted.csv"), show_col_types = FALSE)

# Gene-level rollup. ps/py are site-level; collapse by gene_symbol with
# mean(., na.rm=TRUE) to match Integr_multiomics's internal aggregation.
gene_aggregate <- function(df) {
  df <- df[, !colnames(df) %in% c("site_id", "Gene Symbol"), drop = FALSE]
  df %>%
    group_by(gene_symbol) %>%
    summarise(across(everything(), \(x) mean(x, na.rm = TRUE)), .groups = "drop")
}
pr_g <- gene_aggregate(pr)
ps_g <- gene_aggregate(ps)
py_g <- gene_aggregate(py)

# --- sc per-(cluster, condition) linear mean (over expm1 of @data) --------
cat("[node_fc] precomputing sc per-(cluster, condition) means\n")
expr <- Data.input@assays$originalexp@data    # natural-log normalized
meta <- Data.input@meta.data
clusters_all <- sort(unique(meta$Type))
conditions_all <- sort(unique(meta$condition))
genes_all <- rownames(expr)

# sc_mean[[paste0(condition, "::", cluster)]] = named numeric (gene → linear mean)
sc_mean <- list()
for (cond in conditions_all) {
  for (clu in clusters_all) {
    idx <- which(meta$condition == cond & meta$Type == clu)
    if (length(idx) == 0L) next
    sc_mean[[paste0(cond, "::", clu)]] <- as.numeric(
      Matrix::rowMeans(expm1(expr[, idx, drop = FALSE]))
    )
  }
}
sc_gene_index <- setNames(seq_along(genes_all), genes_all)

# --- Helpers --------------------------------------------------------------
# Build a per-cluster gene→log2FC lookup for one omics table and one (c1,c2).
omics_fc_by_cluster <- function(g_df, c1, c2) {
  out <- list()
  for (clu in clusters_all) {
    col1 <- paste0(c1, "_", clu)
    col2 <- paste0(c2, "_", clu)
    if (!(col1 %in% colnames(g_df)) || !(col2 %in% colnames(g_df))) next
    v <- log2((g_df[[col1]] + EPS) / (g_df[[col2]] + EPS))
    names(v) <- g_df$gene_symbol
    out[[clu]] <- v
  }
  out
}

# Same shape, but for sc — log2 of (linear mean + EPS).
sc_fc_by_cluster <- function(c1, c2) {
  out <- list()
  for (clu in clusters_all) {
    k1 <- paste0(c1, "::", clu); k2 <- paste0(c2, "::", clu)
    if (is.null(sc_mean[[k1]]) || is.null(sc_mean[[k2]])) next
    v <- log2((sc_mean[[k1]] + EPS) / (sc_mean[[k2]] + EPS))
    names(v) <- genes_all
    out[[clu]] <- v
  }
  out
}

# Vectorized lookup: clusters[i], genes[i] → log2FC.
lookup_by_pair <- function(fc_list, clusters_vec, genes_vec) {
  out <- rep(NA_real_, length(clusters_vec))
  if (length(out) == 0L) return(out)
  by_clu <- split(seq_along(clusters_vec), clusters_vec)
  for (clu in names(by_clu)) {
    rows <- by_clu[[clu]]
    v <- fc_list[[clu]]
    if (is.null(v)) next
    out[rows] <- unname(v[genes_vec[rows]])
  }
  out
}

# --- Drive over parquets --------------------------------------------------
parse_conditions <- function(parquet_path) {
  base <- sub("_incytr_output\\.parquet$", "", basename(parquet_path))
  m <- regmatches(base, regexec("^(.*?)_(ma_.*_WTyp)$", base))[[1]]
  if (length(m) < 3L) stop(sprintf("cannot parse conditions from '%s'", basename(parquet_path)))
  list(condition1 = m[2], condition2 = m[3])
}

FC_COLS <- c(
  outer(c("Ligand", "Receptor", "EM", "Target"),
        c("sclog2FC", "pr_log2FC", "ps_log2FC", "py_log2FC"),
        FUN = paste, sep = "_")
)

parquet_files <- list.files(out_dir, pattern = "_incytr_output\\.parquet$",
                            full.names = TRUE)
parquet_files <- parquet_files[!grepl("\\.old\\.parquet$|\\.fc\\.parquet$",
                                       parquet_files)]
if (length(parquet_files) == 0L) stop(sprintf("no parquets under '%s'", out_dir))

cat(sprintf("[node_fc] %d parquets to enrich\n", length(parquet_files)))

for (pq_path in parquet_files) {
  conds <- parse_conditions(pq_path)
  t0 <- proc.time()[["elapsed"]]
  df <- as.data.frame(arrow::read_parquet(pq_path))

  required <- c("Sender", "Receiver", "Ligand", "Receptor", "EM", "Target")
  if (!all(required %in% colnames(df))) {
    stop(sprintf("[node_fc] missing required columns in %s", pq_path))
  }
  if (all(FC_COLS %in% colnames(df))) {
    cat(sprintf("[node_fc] %s vs %s — already enriched, skipping\n",
                conds$condition1, conds$condition2))
    next
  }
  cat(sprintf("[node_fc] %s vs %s\n", conds$condition1, conds$condition2))

  pr_fc <- omics_fc_by_cluster(pr_g, conds$condition1, conds$condition2)
  ps_fc <- omics_fc_by_cluster(ps_g, conds$condition1, conds$condition2)
  py_fc <- omics_fc_by_cluster(py_g, conds$condition1, conds$condition2)
  sc_fc <- sc_fc_by_cluster(conds$condition1, conds$condition2)

  # Ligand uses Sender cluster; Receptor / EM / Target use Receiver cluster.
  df$Ligand_sclog2FC  <- lookup_by_pair(sc_fc, df$Sender,   df$Ligand)
  df$Ligand_pr_log2FC <- lookup_by_pair(pr_fc, df$Sender,   df$Ligand)
  df$Ligand_ps_log2FC <- lookup_by_pair(ps_fc, df$Sender,   df$Ligand)
  df$Ligand_py_log2FC <- lookup_by_pair(py_fc, df$Sender,   df$Ligand)

  for (node in c("Receptor", "EM", "Target")) {
    df[[paste0(node, "_sclog2FC")]]  <- lookup_by_pair(sc_fc, df$Receiver, df[[node]])
    df[[paste0(node, "_pr_log2FC")]] <- lookup_by_pair(pr_fc, df$Receiver, df[[node]])
    df[[paste0(node, "_ps_log2FC")]] <- lookup_by_pair(ps_fc, df$Receiver, df[[node]])
    df[[paste0(node, "_py_log2FC")]] <- lookup_by_pair(py_fc, df$Receiver, df[[node]])
  }

  arrow::write_parquet(df, pq_path, compression = "zstd")
  sz <- file.info(pq_path)$size
  cat(sprintf("[node_fc]   wrote %s (%.1f MB) in %.1f min\n",
              basename(pq_path), sz / 1e6,
              (proc.time()[["elapsed"]] - t0) / 60))
}

cat("[node_fc] done\n")
