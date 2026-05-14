#!/usr/bin/env Rscript
# Memory-safe reshard of staged pair parquets into the receiver_cache layout.
#
# The original wrapper (alz/integration/factorial.R) collapses all pair
# parquets for a receiver into one in-memory long-format frame before writing.
# For large receivers (e.g. 30 Astro-Epen with ~273K wide rows × 9 contrasts)
# that frame OOMs the 16 GB ulimit. This script does the same job
# pair-at-a-time, writing one part-<sender>.parquet per pair under
# receiver_cache/receiver=<r>/ — Hive partition discovery in arrow/duckdb
# treats multiple parquets per partition as a single logical table.

suppressPackageStartupMessages({
  library(Incytr)
  library(arrow)
})

out_dir <- "outputs/reports/incytr_factorial"
staging <- file.path(out_dir, ".staging", "pair_parquets")

source("alz/integration/load.R")
source("alz/integration/persist.R")

# Minimal metadata loads (no expression matrix needed for reshard).
meta <- read.csv("data/incytr_factorial_inputs/expression_metadata.csv",
                 stringsAsFactors = FALSE)
if (!"condition" %in% colnames(meta)) {
  meta$condition <- paste(meta$genotype, meta$timepoint, sep = "_")
}
condition_names <- as.character(unique(meta$condition))
contrasts <- build_factorial_contrasts()
contrast_names <- names(contrasts)

message(sprintf("Conditions: %s", paste(condition_names, collapse = ", ")))
message(sprintf("Contrasts: %s", paste(contrast_names, collapse = ", ")))

# Build pair list from staged parquet filenames.
pair_files <- list.files(staging, pattern = "^pair_.*\\.parquet$",
                         full.names = TRUE)
pair_files <- pair_files[!grepl("multiomic_factorial_results", pair_files)]
message(sprintf("Found %d pair parquets in staging", length(pair_files)))

# Parse "pair_<sender>__<receiver>.parquet" → sender, receiver (sanitized).
parse_pair_name <- function(fp) {
  base <- sub("^pair_", "", sub("\\.parquet$", "", basename(fp)))
  parts <- strsplit(base, "__", fixed = TRUE)[[1]]
  if (length(parts) != 2L) stop("Unexpected pair filename: ", fp)
  list(sender_sanitized = parts[[1]], receiver_sanitized = parts[[2]])
}

written <- 0L
errors <- character(0)
metadata_rows <- vector("list", length(pair_files))

for (i in seq_along(pair_files)) {
  fp <- pair_files[[i]]
  ids <- parse_pair_name(fp)
  recv_dir <- file.path(out_dir, "receiver_cache",
                        paste0("receiver=", ids$receiver_sanitized))
  ensure_dir(recv_dir)
  out_file <- file.path(recv_dir, paste0("part-", ids$sender_sanitized, ".parquet"))

  res <- tryCatch({
    wide <- as.data.frame(arrow::read_parquet(fp))
    n_paths <- nrow(wide)
    long <- Incytr::factorial_results_long(wide,
                                            contrast_names = contrast_names,
                                            condition_names = condition_names)
    rm(wide); gc(verbose = FALSE)
    arrow::write_parquet(long, out_file)
    rm(long); gc(verbose = FALSE)
    list(ok = TRUE, n_paths = n_paths)
  }, error = function(e) {
    list(ok = FALSE, msg = conditionMessage(e), n_paths = NA_integer_)
  })

  if (!res$ok) {
    errors <- c(errors, sprintf("%s: %s", basename(fp), res$msg))
    message(sprintf("[%3d/%d] ERROR  %s -> %s",
                    i, length(pair_files), basename(fp), res$msg))
    next
  }

  written <- written + 1L
  metadata_rows[[i]] <- data.frame(
    sender    = gsub("_", " ", ids$sender_sanitized),
    receiver  = gsub("_", " ", ids$receiver_sanitized),
    n_post    = res$n_paths,
    n_pre     = res$n_paths,
    status    = "ok",
    stringsAsFactors = FALSE
  )

  if (i %% 5 == 0 || i == length(pair_files)) {
    rss_mb <- sum(gc()[, 2]) # approx
    message(sprintf("[%3d/%d] wrote %s  (gc total: %.0f MB)",
                    i, length(pair_files), basename(out_file), rss_mb))
  }
}

# Write pair_metadata.parquet
metadata_rows <- Filter(Negate(is.null), metadata_rows)
if (length(metadata_rows) > 0) {
  pair_metadata <- do.call(rbind, metadata_rows)
  write_pair_metadata_parquet(pair_metadata, out_dir = out_dir)
  message(sprintf("Wrote pair_metadata.parquet with %d rows",
                  nrow(pair_metadata)))
}

write_views_sql(out_dir)

message(sprintf("\nReshard complete: %d/%d pairs written, %d errors",
                written, length(pair_files), length(errors)))
if (length(errors) > 0) {
  message("Errors:")
  for (e in errors) message("  ", e)
}
