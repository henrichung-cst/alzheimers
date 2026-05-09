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
  ensure_dir(out_dir)
  arrow::write_parquet(metadata, file.path(out_dir, "pair_metadata.parquet"))
}

write_views_sql <- function(out_dir, source_path = "alz/integration/views.sql") {
  ensure_dir(out_dir)
  file.copy(source_path, file.path(out_dir, "views.sql"), overwrite = TRUE)
}
