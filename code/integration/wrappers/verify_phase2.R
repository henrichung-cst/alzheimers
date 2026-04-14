#!/usr/bin/env Rscript
# verify_phase2.R — Regression test: compare Phase 2 Parquet output against
# Phase 1 per-pair CSV output.
#
# Reads receiver Parquet files, splits by sender, and compares against the
# corresponding results_full.csv in the per-pair directory structure.
#
# Usage:
#   Rscript verify_phase2.R [--receivers VLMC,Astrocyte,Chandelier] [--tol 1e-10]
#
# Environment:
#   PHASE1_DIR  - Directory containing Phase 1 per-pair CSVs (default: all_pairs/)
#   PHASE2_DIR  - Directory containing Phase 2 Parquet files (default: same as PHASE1_DIR)

suppressPackageStartupMessages({
  library(data.table)
  library(arrow)
})

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
tol <- 1e-10
receivers <- NULL

i <- 1
while (i <= length(args)) {
  if (args[i] == "--receivers" && i < length(args)) {
    receivers <- strsplit(args[i + 1], ",")[[1]]
    i <- i + 2
  } else if (args[i] == "--tol" && i < length(args)) {
    tol <- as.numeric(args[i + 1])
    i <- i + 2
  } else {
    i <- i + 1
  }
}

get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) return(dirname(normalizePath(sub("^--file=", "", file_arg[1]))))
  file.path(getwd(), "code", "integration", "wrappers")
}
script_dir <- get_script_dir()
repo_root <- normalizePath(file.path(script_dir, "..", "..", ".."))
int_dir <- file.path(repo_root, "code", "integration", "intermediates")

phase1_dir <- Sys.getenv("PHASE1_DIR", file.path(int_dir, "all_pairs"))
phase2_dir <- Sys.getenv("PHASE2_DIR", phase1_dir)

sanitize_name <- function(x) gsub("/", "-", gsub(" ", "_", x))

# ---------------------------------------------------------------------------
# Discover receivers
# ---------------------------------------------------------------------------
parquet_files <- list.files(phase2_dir, pattern = "^recv_.*\\.parquet$",
                            full.names = TRUE)
if (length(parquet_files) == 0) {
  cat("No Parquet files found in", phase2_dir, "\n")
  quit(status = 1)
}

if (!is.null(receivers)) {
  parquet_files <- parquet_files[
    basename(parquet_files) %in% paste0("recv_", sanitize_name(receivers), ".parquet")]
}

cat(sprintf("Verifying %d receiver(s) against Phase 1 CSVs\n", length(parquet_files)))
cat(sprintf("  Phase 1 dir: %s\n", phase1_dir))
cat(sprintf("  Phase 2 dir: %s\n", phase2_dir))
cat(sprintf("  Tolerance: %.0e\n\n", tol))

# ---------------------------------------------------------------------------
# Numeric columns to compare
# ---------------------------------------------------------------------------
num_cols <- c("SigProb_WT", "SigProb_App", "log2FC", "aFC",
              "Ligand_sclog2FC", "Receptor_sclog2FC", "EM_sclog2FC", "Target_sclog2FC",
              "TPDS", "multimodel_score", "PDS")

# Optional columns (may not exist in Phase 1 if kinase/phospho data absent)
opt_cols <- c("PhPDS_ps", "SiK_score_WT", "SiK_score_App", "kinase_boost")

n_pairs_checked <- 0
n_pairs_matched <- 0
n_pairs_mismatched <- 0
n_pairs_missing <- 0
max_diff_global <- 0
mismatch_details <- list()

for (pq_path in parquet_files) {
  # Read Parquet
  pq_dt <- as.data.table(read_parquet(pq_path))
  recv <- pq_dt$Receiver.group[1]
  if (is.na(recv) || is.null(recv)) {
    recv <- gsub("^recv_|\\.parquet$", "", basename(pq_path))
    recv <- gsub("_", " ", recv)
  }
  senders <- unique(pq_dt$sender)
  cat(sprintf("--- Receiver: %s (%d senders, %s pathways) ---\n",
              recv, length(senders), format(nrow(pq_dt), big.mark = ",")))

  for (send in senders) {
    n_pairs_checked <- n_pairs_checked + 1
    pair_dir <- file.path(phase1_dir,
                          paste0(sanitize_name(send), "__", sanitize_name(recv)))
    csv_path <- file.path(pair_dir, "results_full.csv")

    if (!file.exists(csv_path)) {
      n_pairs_missing <- n_pairs_missing + 1
      next
    }

    old_dt <- fread(csv_path)
    new_dt <- pq_dt[sender == send]

    # --- Pathway count ---
    if (nrow(old_dt) != nrow(new_dt)) {
      cat(sprintf("  FAIL %s -> %s: pathway count mismatch (%d vs %d)\n",
                  send, recv, nrow(old_dt), nrow(new_dt)))
      n_pairs_mismatched <- n_pairs_mismatched + 1
      mismatch_details[[length(mismatch_details) + 1]] <- list(
        sender = send, receiver = recv, type = "count",
        old = nrow(old_dt), new = nrow(new_dt))
      next
    }

    if (nrow(old_dt) == 0) {
      n_pairs_matched <- n_pairs_matched + 1
      next
    }

    # --- Match rows by Path ---
    setkey(old_dt, Path)
    setkey(new_dt, Path)

    # Check all Paths match
    if (!identical(sort(old_dt$Path), sort(new_dt$Path))) {
      cat(sprintf("  FAIL %s -> %s: Path mismatch\n", send, recv))
      n_pairs_mismatched <- n_pairs_mismatched + 1
      mismatch_details[[length(mismatch_details) + 1]] <- list(
        sender = send, receiver = recv, type = "paths")
      next
    }

    # --- Numeric column comparison ---
    pair_ok <- TRUE
    check_cols <- intersect(c(num_cols, opt_cols), intersect(names(old_dt), names(new_dt)))

    for (col in check_cols) {
      old_vals <- old_dt[.(new_dt$Path), get(col)]
      new_vals <- new_dt[[col]]

      # Handle NAs
      both_na <- is.na(old_vals) & is.na(new_vals)
      if (all(both_na)) next

      # Check for NA mismatches
      na_mismatch <- is.na(old_vals) != is.na(new_vals)
      if (any(na_mismatch)) {
        cat(sprintf("  FAIL %s -> %s: %s NA mismatch (%d rows)\n",
                    send, recv, col, sum(na_mismatch)))
        pair_ok <- FALSE
        next
      }

      # Compare non-NA values
      valid <- !is.na(old_vals) & !is.na(new_vals)
      if (!any(valid)) next

      max_diff <- max(abs(old_vals[valid] - new_vals[valid]))
      if (max_diff > tol) {
        cat(sprintf("  FAIL %s -> %s: %s max diff = %.2e\n",
                    send, recv, col, max_diff))
        pair_ok <- FALSE
      }
      max_diff_global <- max(max_diff_global, max_diff)
    }

    if (pair_ok) {
      n_pairs_matched <- n_pairs_matched + 1
    } else {
      n_pairs_mismatched <- n_pairs_mismatched + 1
    }
  }
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
cat("\n=== Verification Summary ===\n")
cat(sprintf("  Pairs checked:    %d\n", n_pairs_checked))
cat(sprintf("  Pairs matched:    %d\n", n_pairs_matched))
cat(sprintf("  Pairs mismatched: %d\n", n_pairs_mismatched))
cat(sprintf("  Pairs missing:    %d (no Phase 1 CSV)\n", n_pairs_missing))
cat(sprintf("  Max numeric diff: %.2e\n", max_diff_global))
cat(sprintf("  Tolerance:        %.2e\n", tol))

if (n_pairs_mismatched > 0) {
  cat("\nFAILED — mismatches detected.\n")
  quit(status = 1)
} else if (n_pairs_missing > 0 && n_pairs_matched == 0) {
  cat("\nWARNING — no Phase 1 CSVs found for comparison.\n")
  quit(status = 2)
} else {
  cat("\nPASSED — all checked pairs match within tolerance.\n")
  quit(status = 0)
}
