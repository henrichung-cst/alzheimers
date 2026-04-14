#!/usr/bin/env Rscript
# Diagnostic: estimate L2⋈L3 join sizes WITHOUT materializing them.
# For each receiver, counts pruned L2/L3 edges and computes the
# join cardinality from EM degree overlap: sum over each EM gene of
# (L2 edges into EM) × (L3 edges out of EM).
#
# This never allocates the chain table, so it cannot OOM.
#
# Usage:
#   micromamba run -n incytr Rscript code/integration/tests/edge_pruning/diagnose_em_fanout.R

suppressPackageStartupMessages({
  library(Matrix)
  library(data.table)
  library(Incytr)
})

hill <- function(x, K = 0.5, N = 2) x^N / (x^N + K^N)

get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) return(dirname(normalizePath(sub("^--file=", "", file_arg[1]))))
  getwd()
}
repo_root <- normalizePath(file.path(get_script_dir(), "..", "..", "..", ".."))
int_dir   <- file.path(repo_root, "code", "integration", "intermediates")

# Load data
cat("Loading expression data...\n")
mat <- readMM(file.path(int_dir, "expression_matrix.mtx"))
mat <- as(mat, "CsparseMatrix")
genes    <- read.csv(file.path(int_dir, "expression_genes.csv"))$gene
barcodes <- read.csv(file.path(int_dir, "expression_barcodes.csv"))$barcode
meta     <- read.csv(file.path(int_dir, "expression_metadata.csv"), row.names = 1)
rownames(mat) <- genes; colnames(mat) <- barcodes

cat("Loading IncytrDB (L2 + L3 only)...\n")
data(DB_Layer2_mouse_filtered, package = "Incytr")
data(DB_Layer3_mouse_filtered, package = "Incytr")
all_genes <- rownames(mat)
l2_raw <- as.data.table(DB_Layer2_mouse_filtered[
  DB_Layer2_mouse_filtered$from %in% all_genes &
  DB_Layer2_mouse_filtered$to %in% all_genes, ])
l3_raw <- as.data.table(DB_Layer3_mouse_filtered[
  DB_Layer3_mouse_filtered$from %in% all_genes &
  DB_Layer3_mouse_filtered$to %in% all_genes, ])
rm(DB_Layer2_mouse_filtered, DB_Layer3_mouse_filtered)
gc(verbose = FALSE)
cat(sprintf("  L2=%d, L3=%d edges\n\n", nrow(l2_raw), nrow(l3_raw)))

conditions <- c("WT", "App")
cell_types <- sort(unique(meta$labels))
K <- 0.5; N <- 2; CUTOFF <- 0.01

# Compute mean expression
cat("Computing mean expression...\n")
mean_expr <- list()
for (ct in cell_types) {
  mean_expr[[ct]] <- list()
  for (cond in conditions) {
    cells <- which(meta$labels == ct & meta$condition == cond)
    if (length(cells) > 0) {
      mean_expr[[ct]][[cond]] <- setNames(Matrix::rowMeans(mat[, cells]), all_genes)
    } else {
      mean_expr[[ct]][[cond]] <- setNames(rep(0, length(all_genes)), all_genes)
    }
  }
}
rm(mat, genes, barcodes)
gc(verbose = FALSE)
cat("Done.\n\n")

# ---------------------------------------------------------------------------
# Per receiver: prune L2/L3, estimate join size from EM degree product
# ---------------------------------------------------------------------------
cat(sprintf("%-20s %8s %8s %8s %12s %12s %10s\n",
            "Receiver", "L2_kept", "L3_kept", "EM_genes",
            "Join_est", "Join_MB_est", "Top_EM_deg"))
cat(paste(rep("-", 86), collapse = ""), "\n")

results <- list()

for (recv in cell_types) {
  rc1 <- mean_expr[[recv]][["WT"]]
  rc2 <- mean_expr[[recv]][["App"]]

  # Prune L2 (Receptor -> EM)
  dt2 <- copy(l2_raw)
  h1 <- hill(rc1[dt2$from] * rc1[dt2$to], K, N)
  h2 <- hill(rc2[dt2$from] * rc2[dt2$to], K, N)
  dt2 <- dt2[(h1 >= CUTOFF) | (h2 >= CUTOFF)]
  n_l2 <- nrow(dt2)

  # Prune L3 (EM -> Target)
  dt3 <- copy(l3_raw)
  h1 <- hill(rc1[dt3$from] * rc1[dt3$to], K, N)
  h2 <- hill(rc2[dt3$from] * rc2[dt3$to], K, N)
  dt3 <- dt3[(h1 >= CUTOFF) | (h2 >= CUTOFF)]
  n_l3 <- nrow(dt3)

  # Count EM degree in L2 (how many Receptors point to each EM)
  l2_em_deg <- dt2[, .N, by = to]    # to = EM in L2
  setnames(l2_em_deg, c("EM", "l2_deg"))

  # Count EM degree in L3 (how many Targets each EM points to)
  l3_em_deg <- dt3[, .N, by = from]  # from = EM in L3
  setnames(l3_em_deg, c("EM", "l3_deg"))

  # Join cardinality = sum over shared EM genes of (l2_deg × l3_deg)
  em_merged <- merge(l2_em_deg, l3_em_deg, by = "EM")
  em_merged[, product := l2_deg * l3_deg]
  join_est <- sum(as.numeric(em_merged$product))
  n_em <- nrow(em_merged)

  # Top hub EM gene
  top_idx <- which.max(em_merged$product)
  top_em <- em_merged$EM[top_idx]
  top_deg <- em_merged$product[top_idx]
  top_l2 <- em_merged$l2_deg[top_idx]
  top_l3 <- em_merged$l3_deg[top_idx]

  mb_est <- join_est * 50 / 1024^2

  results[[recv]] <- data.frame(
    receiver = recv, l2_kept = n_l2, l3_kept = n_l3, n_em = n_em,
    join_est = join_est, mb_est = round(mb_est, 0),
    top_em = top_em, top_l2_deg = top_l2, top_l3_deg = top_l3,
    top_product = top_deg,
    stringsAsFactors = FALSE)

  cat(sprintf("%-20s %8d %8d %8d %12s %10.0f MB %s (%dx%d=%s)\n",
              recv, n_l2, n_l3, n_em,
              format(round(join_est), big.mark = ","),
              mb_est,
              top_em, top_l2, top_l3,
              format(top_deg, big.mark = ",")))

  rm(dt2, dt3, h1, h2, l2_em_deg, l3_em_deg, em_merged)
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
results_df <- do.call(rbind, results)
results_df <- results_df[order(-results_df$join_est), ]

cat("\n====== Summary (sorted by estimated join size) ======\n")
cat(sprintf("  Total estimated chains: %s\n",
            format(sum(results_df$join_est), big.mark = ",")))
cat(sprintf("  If all held in memory:  %.0f GB\n",
            sum(results_df$mb_est) / 1024))
cat(sprintf("  Largest receiver:       %s (%s chains, %.0f MB)\n",
            results_df$receiver[1],
            format(results_df$join_est[1], big.mark = ","),
            results_df$mb_est[1]))
cat(sprintf("  Smallest receiver:      %s (%s chains, %.0f MB)\n",
            results_df$receiver[nrow(results_df)],
            format(results_df$join_est[nrow(results_df)], big.mark = ","),
            results_df$mb_est[nrow(results_df)]))

cat("\n  Note: these are pre-dedup estimates (before removing self-edges).\n")
cat("  Actual chain counts will be smaller, but the join must be\n")
cat("  materialized first, so this reflects peak memory.\n")

# Top 10 hub EM genes across all receivers
cat("\n====== Top hub EM genes (largest fan-out) ======\n")
cat("  These genes appear in both L2 and L3 with high degree,\n")
cat("  causing the cartesian join explosion.\n\n")
hub_data <- results_df[, c("receiver", "top_em", "top_l2_deg", "top_l3_deg", "top_product")]
hub_data <- hub_data[order(-hub_data$top_product), ]
print(head(hub_data, 10), row.names = FALSE)

out_path <- file.path(get_script_dir(), "em_fanout_diagnostic.csv")
write.csv(results_df, out_path, row.names = FALSE)
cat(sprintf("\nSaved: %s\n", out_path))
