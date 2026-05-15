suppressPackageStartupMessages({
  library(arrow)
})

ensure_dir <- function(path) {
  if (!dir.exists(path)) {
    dir.create(path, recursive = TRUE, showWarnings = FALSE)
  }
}

write_receiver_parquet <- function(df, receiver, out_dir) {
  path <- file.path(out_dir, "receiver_cache", paste0("receiver=", sanitize_celltype(receiver)))
  ensure_dir(path)
  arrow::write_parquet(df, file.path(path, "data.parquet"))
}

write_pair_metadata_parquet <- function(metadata, out_dir) {
  # Partial sweeps (single receiver) used to overwrite the full pair_metadata
  # and erase prior rows, leaving the unified-viewer heatmap with only the
  # latest receiver's senders. Merge into any existing file instead: drop
  # rows for the (sender, receiver) pairs being rewritten, then rbind.
  ensure_dir(out_dir)
  path <- file.path(out_dir, "pair_metadata.parquet")
  if (file.exists(path)) {
    prior <- as.data.frame(arrow::read_parquet(path))
    new_keys <- paste(metadata$sender, metadata$receiver, sep = "")
    prior_keys <- paste(prior$sender, prior$receiver, sep = "")
    prior <- prior[!prior_keys %in% new_keys, , drop = FALSE]
    metadata <- rbind(prior, metadata)
  }
  arrow::write_parquet(metadata, path)
}

write_views_sql <- function(out_dir, source_path = "alz/integration/views.sql") {
  ensure_dir(out_dir)
  file.copy(source_path, file.path(out_dir, "views.sql"), overwrite = TRUE)
}
