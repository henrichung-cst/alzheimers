#!/usr/bin/env Rscript
# Bootstrap sensitivity analyses for Phase 1 proof of concept.
#
# Part 1: L5 IT barcode bootstrap (500 iterations)
#   Resamples receiver barcodes with replacement, recomputes expression
#   scores while keeping pathway_inference() fixed. Assesses rank stability.
#
# Part 2: Detection threshold sensitivity
#   Reruns pathway_inference() at 20% threshold (vs default 50%).
#   Compares top-50 overlap. Wrapped in tryCatch for OOM.
#
# Outputs:
#   intermediates/bootstrap_stability.csv
#   intermediates/threshold_comparison.csv

suppressPackageStartupMessages({
  library(Matrix)
  library(data.table)
})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg[1]))))
  }
  return(file.path(getwd(), "code", "integration", "wrappers"))
}
script_dir <- get_script_dir()
repo_root <- normalizePath(file.path(script_dir, "..", "..", ".."))
int_dir <- file.path(repo_root, "code", "integration", "intermediates")

cat("=== Bootstrap Sensitivity Analysis ===\n\n")

library(Incytr)

# ---------------------------------------------------------------------------
# Load shared data
# ---------------------------------------------------------------------------
cat("Loading expression matrix...\n")
mat <- readMM(file.path(int_dir, "expression_matrix.mtx"))
mat <- as(mat, "dgCMatrix")
genes <- read.csv(file.path(int_dir, "expression_genes.csv"))$gene
barcodes <- read.csv(file.path(int_dir, "expression_barcodes.csv"))$barcode
rownames(mat) <- genes
colnames(mat) <- barcodes
meta <- read.csv(file.path(int_dir, "expression_metadata.csv"),
                 row.names = 1, check.names = FALSE)
cat(sprintf("  %d genes x %d cells\n", nrow(mat), ncol(mat)))

cat("Loading saved Incytr object...\n")
inc_orig <- readRDS(file.path(int_dir, "incytr_object.rds"))
cat(sprintf("  %d pathways in saved object\n", nrow(inc_orig@pathways)))

# Match config_integration.py: SENDER, RECEIVER, INCYTR_CONDITIONS
sender <- Sys.getenv("INCYTR_SENDER", "Microglia-PVM")
receiver <- Sys.getenv("INCYTR_RECEIVER", "L5 IT")
conditions <- c("WT", "App")

# Read config values from environment (set by run_phase1.sh or defaults)
n_bootstrap <- as.integer(Sys.getenv("N_BOOTSTRAP_ITERATIONS", "500"))
alt_threshold <- as.numeric(Sys.getenv("DETECTION_THRESHOLD_SENSITIVITY", "0.20"))

# ---------------------------------------------------------------------------
# Part 1: L5 IT bootstrap
# ---------------------------------------------------------------------------
cat(sprintf("\n--- Part 1: L5 IT Bootstrap (%d iterations) ---\n", n_bootstrap))

receiver_barcodes <- rownames(meta)[meta$labels == receiver]
n_receiver <- length(receiver_barcodes)
cat(sprintf("  Receiver barcodes: %d\n", n_receiver))

# Pre-compute: get baseline TPDS ranking for comparison
baseline_expronly <- read.csv(file.path(int_dir, "results_expronly.csv"),
                              check.names = FALSE)
tpds_col <- if ("TPDS" %in% names(baseline_expronly)) "TPDS" else "multimodel_score"
baseline_paths <- baseline_expronly$Path
baseline_ranks <- rank(-baseline_expronly[[tpds_col]], ties.method = "min")

# Bootstrap: resample receiver barcodes, recompute expression scores
rank_matrix <- matrix(NA_real_, nrow = length(baseline_paths),
                      ncol = n_bootstrap)

for (b in seq_len(n_bootstrap)) {
  if (b %% 50 == 0 || b == 1) cat(sprintf("  Iteration %d/%d\n", b, n_bootstrap))

  set.seed(b)
  # Resample receiver barcodes with replacement
  boot_barcodes <- sample(receiver_barcodes, n_receiver, replace = TRUE)

  # Create new barcode list: keep all non-receiver barcodes, add resampled
  other_barcodes <- rownames(meta)[meta$labels != receiver]
  all_boot_barcodes <- c(other_barcodes, boot_barcodes)

  # Subset matrix and metadata
  # Handle duplicated barcodes by making unique names
  boot_mat <- mat[, all_boot_barcodes]
  boot_meta <- meta[all_boot_barcodes, , drop = FALSE]

  # Make unique column names for duplicates
  if (any(duplicated(colnames(boot_mat)))) {
    colnames(boot_mat) <- make.unique(colnames(boot_mat), sep = "_")
    rownames(boot_meta) <- colnames(boot_mat)
  }

  inc_boot <- tryCatch({
    inc_b <- create_Incytr(
      object = boot_mat,
      meta = boot_meta,
      sender = sender,
      receiver = receiver,
      group.by = "labels",
      conditions = conditions
    )

    # Copy pathways from original (pathway_inference is fixed)
    inc_b@pathways <- inc_orig@pathways

    inc_b <- Expr_bygroup(inc_b)
    inc_b <- Cal_SigProb(inc_b, K = 0.5, N = 2, cutoff_SigProb = 0.01,
                          correction = 0.001)
    inc_b <- Cal_scFC(inc_b)
    inc_b <- Pathway_evaluation(inc_b, score.weight = rep(0, 6))

    results_b <- Export_results(inc_b)
    tpds_b <- results_b[[tpds_col]]
    rank(-tpds_b, ties.method = "min")
  }, error = function(e) {
    cat(sprintf("    WARNING: iteration %d failed: %s\n", b, e$message))
    rep(NA_real_, length(baseline_paths))
  })

  rank_matrix[, b] <- inc_boot
  rm(boot_mat, boot_meta)
  if (b %% 100 == 0) gc(verbose = FALSE)
}

# Compute stability metrics
mean_rank <- rowMeans(rank_matrix, na.rm = TRUE)
sd_rank <- apply(rank_matrix, 1, sd, na.rm = TRUE)
bootstrap_stability <- data.table(
  Path = baseline_paths,
  mean_rank = mean_rank,
  sd_rank = sd_rank,
  cv_rank = sd_rank / (mean_rank + 1e-10),
  frac_in_top20 = rowMeans(rank_matrix <= 20, na.rm = TRUE),
  frac_in_top50 = rowMeans(rank_matrix <= 50, na.rm = TRUE)
)

bs_path <- file.path(int_dir, "bootstrap_stability.csv")
fwrite(bootstrap_stability, bs_path)
cat(sprintf("\n  Wrote %s (%d pathways)\n", basename(bs_path),
            nrow(bootstrap_stability)))

n_unstable <- sum(bootstrap_stability$cv_rank > 1.0, na.rm = TRUE)
cat(sprintf("  Pathways with CV > 1.0: %d / %d\n",
            n_unstable, nrow(bootstrap_stability)))

# ---------------------------------------------------------------------------
# Part 2: Detection threshold sensitivity
# ---------------------------------------------------------------------------
cat(sprintf("\n--- Part 2: Detection Threshold Sensitivity (%.0f%% vs 50%%) ---\n",
            alt_threshold * 100))

# Load IncytrDB
data(DB_Layer1_mouse_filtered, package = "Incytr")
data(DB_Layer2_mouse_filtered, package = "Incytr")
data(DB_Layer3_mouse_filtered, package = "Incytr")
all_genes <- rownames(mat)
DB_Layer1_mouse_filtered <- DB_Layer1_mouse_filtered[
  DB_Layer1_mouse_filtered$from %in% all_genes &
  DB_Layer1_mouse_filtered$to %in% all_genes, ]
DB_Layer2_mouse_filtered <- DB_Layer2_mouse_filtered[
  DB_Layer2_mouse_filtered$from %in% all_genes &
  DB_Layer2_mouse_filtered$to %in% all_genes, ]
DB_Layer3_mouse_filtered <- DB_Layer3_mouse_filtered[
  DB_Layer3_mouse_filtered$from %in% all_genes &
  DB_Layer3_mouse_filtered$to %in% all_genes, ]
DB.M <- list(DB_Layer1_mouse_filtered, DB_Layer2_mouse_filtered,
             DB_Layer3_mouse_filtered)
rm(DB_Layer1_mouse_filtered, DB_Layer2_mouse_filtered,
   DB_Layer3_mouse_filtered)
gc(verbose = FALSE)

sender_cells <- which(meta$labels == sender)
receiver_cells <- which(meta$labels == receiver)
sender_expr <- Matrix::rowMeans(mat[, sender_cells] > 0)
receiver_expr <- Matrix::rowMeans(mat[, receiver_cells] > 0)

sender_genes_alt <- names(sender_expr[sender_expr >= alt_threshold])
receiver_genes_alt <- names(receiver_expr[receiver_expr >= alt_threshold])
cat(sprintf("  Sender genes (>%.0f%%): %d\n",
            alt_threshold * 100, length(sender_genes_alt)))
cat(sprintf("  Receiver genes (>%.0f%%): %d\n",
            alt_threshold * 100, length(receiver_genes_alt)))

inc_alt <- tryCatch({
  inc_a <- create_Incytr(
    object = mat,
    meta = meta,
    sender = sender,
    receiver = receiver,
    group.by = "labels",
    conditions = conditions
  )

  inc_a <- pathway_inference(inc_a, DB = DB.M,
                             gene.use_Sender = sender_genes_alt,
                             gene.use_Receiver = receiver_genes_alt)
  n_pw <- nrow(inc_a@pathways)
  cat(sprintf("  %d pathways at %.0f%% threshold\n", n_pw, alt_threshold * 100))

  if (n_pw > 500000) {
    cat("  WARNING: >500K pathways, skipping evaluation to avoid OOM.\n")
    NULL
  } else {
    inc_a <- Expr_bygroup(inc_a)
    inc_a <- Cal_SigProb(inc_a, K = 0.5, N = 2, cutoff_SigProb = 0.01,
                          correction = 0.001)
    inc_a <- Cal_scFC(inc_a)
    inc_a <- Pathway_evaluation(inc_a, score.weight = rep(0, 6))
    Export_results(inc_a)
  }
}, error = function(e) {
  cat(sprintf("  ERROR: threshold sensitivity failed: %s\n", e$message))
  NULL
})

if (!is.null(inc_alt)) {
  # Compare rankings with 50% baseline
  alt_dt <- as.data.table(inc_alt)
  alt_dt[, alt_rank := rank(-get(tpds_col), ties.method = "min")]

  base_dt <- data.table(Path = baseline_paths, rank_50pct = baseline_ranks)
  comparison <- merge(base_dt, alt_dt[, .(Path, rank_20pct = alt_rank)],
                      by = "Path", all = TRUE)
  comparison[, rank_change := abs(rank_50pct - rank_20pct)]

  tc_path <- file.path(int_dir, "threshold_comparison.csv")
  fwrite(comparison, tc_path)
  cat(sprintf("  Wrote %s\n", basename(tc_path)))

  # Top-50 overlap
  top50_base <- base_dt[rank_50pct <= 50]$Path
  top50_alt <- alt_dt[alt_rank <= 50]$Path
  overlap <- length(intersect(top50_base, top50_alt)) / 50
  cat(sprintf("  Top-50 overlap (50%% vs %.0f%%): %.0f%%\n",
              alt_threshold * 100, overlap * 100))
} else {
  cat("  Threshold comparison skipped (OOM or error).\n")
}

cat("\nBootstrap sensitivity complete.\n")
