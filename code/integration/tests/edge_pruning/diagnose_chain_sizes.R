#!/usr/bin/env Rscript
# Diagnostic: measure receiver chain table sizes WITHOUT storing them all.
# Reports rows, estimated MB, and peak memory per receiver.
# No sender joins — just L2⋈L3 chain sizes.
#
# Usage (from repo root):
#   micromamba run -n incytr Rscript code/integration/tests/edge_pruning/diagnose_chain_sizes.R

suppressPackageStartupMessages({
  library(Matrix)
  library(data.table)
  library(Incytr)
})

hill <- function(x, K = 0.5, N = 2) x^N / (x^N + K^N)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) return(dirname(normalizePath(sub("^--file=", "", file_arg[1]))))
  getwd()
}
repo_root <- normalizePath(file.path(get_script_dir(), "..", "..", "..", ".."))
int_dir   <- file.path(repo_root, "code", "integration", "intermediates")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
cat("Loading expression data...\n")
mat <- readMM(file.path(int_dir, "expression_matrix.mtx"))
mat <- as(mat, "CsparseMatrix")
genes    <- read.csv(file.path(int_dir, "expression_genes.csv"))$gene
barcodes <- read.csv(file.path(int_dir, "expression_barcodes.csv"))$barcode
meta     <- read.csv(file.path(int_dir, "expression_metadata.csv"), row.names = 1)
rownames(mat) <- genes; colnames(mat) <- barcodes

cat("Loading IncytrDB...\n")
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

# ---------------------------------------------------------------------------
# Compute per-condition mean expression
# ---------------------------------------------------------------------------
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
cat("Done.\n\n")

# Free the expression matrix — we only need mean_expr from here
rm(mat, genes, barcodes)
gc(verbose = FALSE)

# ---------------------------------------------------------------------------
# Measure chain sizes per receiver (one at a time, freed after measuring)
# ---------------------------------------------------------------------------
cat("Measuring chain sizes per receiver...\n")
cat(sprintf("%-20s %8s %8s %10s %8s %10s\n",
            "Receiver", "L2_kept", "L3_kept", "Chains", "MB_est", "Mem_used"))
cat(paste(rep("-", 72), collapse = ""), "\n")

results <- list()
baseline_mem <- gc(verbose = FALSE)[2, 2]  # current Mb used

for (recv in cell_types) {
  rc1 <- mean_expr[[recv]][["WT"]]
  rc2 <- mean_expr[[recv]][["App"]]

  # Prune L2
  dt2 <- copy(l2_raw)
  h1 <- hill(rc1[dt2$from] * rc1[dt2$to], K, N)
  h2 <- hill(rc2[dt2$from] * rc2[dt2$to], K, N)
  dt2 <- dt2[(h1 >= CUTOFF) | (h2 >= CUTOFF)]
  n_l2 <- nrow(dt2)
  setnames(dt2, c("from", "to"), c("Receptor", "EM"), skip_absent = TRUE)

  # Prune L3
  dt3 <- copy(l3_raw)
  h1 <- hill(rc1[dt3$from] * rc1[dt3$to], K, N)
  h2 <- hill(rc2[dt3$from] * rc2[dt3$to], K, N)
  dt3 <- dt3[(h1 >= CUTOFF) | (h2 >= CUTOFF)]
  n_l3 <- nrow(dt3)
  setnames(dt3, c("from", "to"), c("EM", "Target"), skip_absent = TRUE)

  # Join L2 ⋈ L3 on EM
  setkey(dt2, EM); setkey(dt3, EM)
  chains <- dt3[dt2, allow.cartesian = TRUE, nomatch = 0]

  # Remove self-edges + deduplicate
  chains <- chains[Receptor != EM & Receptor != Target & EM != Target]
  chains <- unique(chains[, .(Receptor, EM, Target)])

  n_chains <- nrow(chains)
  # Estimate: 3 character columns, ~50 bytes per row overhead in data.table
  mb_est <- n_chains * 50 / 1024^2

  current_mem <- gc(verbose = FALSE)[2, 2]

  results[[recv]] <- data.frame(
    receiver = recv, l2_kept = n_l2, l3_kept = n_l3,
    chains = n_chains, mb_est = round(mb_est, 1),
    stringsAsFactors = FALSE)

  cat(sprintf("%-20s %8d %8d %10s %6.1f MB %8.0f MB\n",
              recv, n_l2, n_l3,
              format(n_chains, big.mark = ","),
              mb_est, current_mem))

  # Free immediately
  rm(dt2, dt3, chains, h1, h2)
  gc(verbose = FALSE)
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
results_df <- do.call(rbind, results)
results_df <- results_df[order(-results_df$chains), ]

cat("\n====== Summary (sorted by chain count) ======\n")
cat(sprintf("  Total chains across all receivers: %s\n",
            format(sum(results_df$chains), big.mark = ",")))
cat(sprintf("  Estimated total MB if all held in memory: %.0f MB\n",
            sum(results_df$mb_est)))
cat(sprintf("  Largest single receiver: %s (%s chains, %.0f MB)\n",
            results_df$receiver[1],
            format(results_df$chains[1], big.mark = ","),
            results_df$mb_est[1]))
cat(sprintf("  Smallest single receiver: %s (%s chains, %.0f MB)\n",
            results_df$receiver[nrow(results_df)],
            format(results_df$chains[nrow(results_df)], big.mark = ","),
            results_df$mb_est[nrow(results_df)]))

# Also show what the L1 join would produce for the biggest receiver
# (rough upper bound: max L1 edges after pruning × chains)
cat("\n  Note: pathway count per pair = L1_pruned × avg_chains_per_receptor_gene\n")
cat("  The L1 join is the second memory bottleneck — large chains × many L1 edges.\n")

out_path <- file.path(get_script_dir(), "chain_size_diagnostic.csv")
write.csv(results_df, out_path, row.names = FALSE)
cat(sprintf("\nSaved: %s\n", out_path))
