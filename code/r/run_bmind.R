#!/usr/bin/env Rscript
# bMIND benchmark for SAP validation (section 5)
# Called by sap_validate.py via subprocess.
#
# Usage:
#   Rscript code/r/run_bmind.R --input-dir <dir> --output-dir <dir>
#
# Expects in input-dir:
#   bulk_matrix.tsv       — (J sites x 24 samples) phosphosite intensities
#   a_obs_matrix.tsv      — (24 samples x 6 cell types) composition fractions
#   reference_profiles.tsv — (genes x 6 cell types) mean expression reference
#
# Writes to output-dir:
#   X_bmind.tsv           — (J sites x 6*24 columns) deconvoluted profiles

suppressPackageStartupMessages({
  if (!requireNamespace("argparse", quietly = TRUE)) {
    install.packages("argparse", repos = "https://cloud.r-project.org")
  }
  library(argparse)
})

parser <- ArgumentParser(description = "bMIND deconvolution benchmark")
parser$add_argument("--input-dir", required = TRUE, help = "Input directory")
parser$add_argument("--output-dir", required = TRUE, help = "Output directory")
args <- parser$parse_args()

# Check/install bMIND
if (!requireNamespace("bMIND", quietly = TRUE)) {
  if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager", repos = "https://cloud.r-project.org")
  }
  BiocManager::install("bMIND", ask = FALSE)
}
library(bMIND)

cat("Reading inputs...\n")

bulk <- as.matrix(read.delim(
  file.path(args$input_dir, "bulk_matrix.tsv"),
  row.names = 1, check.names = FALSE
))
cat(sprintf("  Bulk: %d sites x %d samples\n", nrow(bulk), ncol(bulk)))

frac <- as.matrix(read.delim(
  file.path(args$input_dir, "a_obs_matrix.tsv"),
  row.names = 1, check.names = FALSE
))
cat(sprintf("  Fractions: %d samples x %d cell types\n", nrow(frac), ncol(frac)))

ref <- as.matrix(read.delim(
  file.path(args$input_dir, "reference_profiles.tsv"),
  row.names = 1, check.names = FALSE
))
cat(sprintf("  Reference: %d genes x %d cell types\n", nrow(ref), ncol(ref)))

# Check for per-cell-type covariance files (optional, improves prior)
ct_names <- colnames(ref)
cov_files <- file.path(args$input_dir, paste0("ref_cov_", ct_names, ".tsv"))
has_cov <- all(file.exists(cov_files))

# Run bMIND
if (has_cov) {
  cat("  Loading per-cell-type reference covariances...\n")
  # bMIND accepts covariance via the 'covariance' argument if available.
  # Build a block-diagonal covariance: list of (genes x genes) per cell type.
  cov_list <- lapply(cov_files, function(f) {
    as.matrix(read.delim(f, row.names = 1, check.names = FALSE))
  })
  names(cov_list) <- ct_names

  # Attempt to pass covariance; fall back to default if API doesn't support it
  cat("Running bMIND (with reference covariance)...\n")
  bmind_result <- tryCatch(
    bMIND(bulk, frac, ref, covariance = cov_list),
    error = function(e) {
      cat(sprintf("  Covariance argument not supported: %s\n", e$message))
      cat("  Falling back to default bMIND (mean reference only)...\n")
      bMIND(bulk, frac, ref)
    }
  )
} else {
  cat("Running bMIND (mean reference only)...\n")
  bmind_result <- bMIND(bulk, frac, ref)
}

# Extract cell-type-specific estimates
# bmind_result$A is the deconvoluted matrix: (sites x cell_types x samples)
A <- bmind_result$A
cat(sprintf("  bMIND output: %s\n", paste(dim(A), collapse = " x ")))

# Reshape to 2D: (sites x cell_types*samples)
n_sites <- dim(A)[1]
n_types <- dim(A)[2]
n_samples <- dim(A)[3]

out_matrix <- matrix(NA, nrow = n_sites, ncol = n_types * n_samples)
col_names <- c()
for (k in 1:n_types) {
  for (s in 1:n_samples) {
    col_idx <- (k - 1) * n_samples + s
    out_matrix[, col_idx] <- A[, k, s]
    col_names <- c(col_names, sprintf("ct%d_s%d", k - 1, s - 1))
  }
}
colnames(out_matrix) <- col_names
rownames(out_matrix) <- rownames(bulk)[1:n_sites]

out_path <- file.path(args$output_dir, "X_bmind.tsv")
write.table(out_matrix, out_path, sep = "\t", quote = FALSE)
cat(sprintf("  Written: %s (%d x %d)\n", out_path, nrow(out_matrix), ncol(out_matrix)))
cat("Done.\n")
