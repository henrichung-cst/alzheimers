#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("Missing required R package: jsonlite", call. = FALSE)
  }
})

args <- commandArgs(trailingOnly = TRUE)

repo_root <- tryCatch(
  system("git rev-parse --show-toplevel", intern = TRUE),
  error = function(e) getwd()
)
if (length(repo_root) == 0 || !nzchar(repo_root[[1]])) {
  repo_root <- getwd()
}

default_dir <- file.path(
  repo_root,
  "data", "datasets", "5xFAD", "primary", "scrna", "reclustering"
)
default_paths <- file.path(
  default_dir,
  c(
    "fivex_renamed_from_merged.RDS",
    "with_cluster_names_merged_object.RDS",
    "named_lore00.RDS"
  )
)

paths <- if (length(args) > 0) args else default_paths
paths <- normalizePath(paths, mustWork = FALSE)

safe_head <- function(x, n = 50L) {
  x <- as.character(x)
  utils::head(sort(unique(x[!is.na(x) & nzchar(x)])), n)
}

safe_names <- function(x) {
  out <- tryCatch(names(x), error = function(e) character())
  if (is.null(out)) character() else out
}

maybe_call <- function(expr, default = NULL) {
  tryCatch(expr, error = function(e) default)
}

derive_5xfad_design <- function(samples) {
  samples <- sort(unique(as.character(samples)))
  parsed <- strcapture(
    "^(5XFAD|WildT)_([0-9]{2}mo)_([CH])_([0-9]+)$",
    samples,
    proto = data.frame(
      genotype_raw = character(),
      age_raw = character(),
      tissue_raw = character(),
      sample_number = integer()
    )
  )
  ok <- !is.na(parsed$genotype_raw)
  if (!any(ok)) {
    return(list(parseable_samples = 0L, sample_count = length(samples)))
  }
  design <- data.frame(
    sample = samples[ok],
    genotype = ifelse(parsed$genotype_raw[ok] == "5XFAD", "TG", "WT"),
    age = sub("^0", "", parsed$age_raw[ok]),
    tissue = ifelse(parsed$tissue_raw[ok] == "C", "cortex", "hippocampus"),
    sample_number = parsed$sample_number[ok],
    stringsAsFactors = FALSE
  )
  condition <- paste(design$tissue, design$age, design$genotype, sep = "|")
  list(
    parseable_samples = nrow(design),
    sample_count = length(samples),
    samples = split(design$sample, condition),
    condition_counts = as.list(sort(table(condition)))
  )
}

summarize_seurat <- function(obj) {
  md <- maybe_call(obj@meta.data, data.frame())
  meta_cols <- colnames(md)

  active_ident_counts <- maybe_call(sort(table(as.character(Seurat::Idents(obj))), decreasing = TRUE), integer())
  assay_names <- maybe_call(Seurat::Assays(obj), character())
  reduction_names <- safe_names(maybe_call(obj@reductions, list()))

  assay_summary <- lapply(assay_names, function(assay) {
    a <- maybe_call(obj[[assay]], NULL)
    list(
      assay = assay,
      class = paste(class(a), collapse = "/"),
      dimensions = maybe_call(as.integer(dim(a)), integer()),
      layers = safe_names(maybe_call(a@layers, list())),
      slots_present = names(Filter(
        isTRUE,
        list(
          counts = maybe_call(!is.null(a@counts) && length(a@counts) > 0, FALSE),
          data = maybe_call(!is.null(a@data) && length(a@data) > 0, FALSE),
          scale.data = maybe_call(!is.null(a@scale.data) && length(a@scale.data) > 0, FALSE)
        )
      ))
    )
  })

  sample_cols <- grep("sample|orig.ident|library|sublib|batch", meta_cols, ignore.case = TRUE, value = TRUE)
  condition_cols <- grep("condition|group|genotype|age|month|tissue|region|sex", meta_cols, ignore.case = TRUE, value = TRUE)
  cluster_cols <- grep("cluster|leiden|cell.?type|annotation|coarse|fine|type", meta_cols, ignore.case = TRUE, value = TRUE)

  column_levels <- function(cols, max_cols = 30L) {
    cols <- utils::head(cols, max_cols)
    stats::setNames(lapply(cols, function(col) {
      vals <- md[[col]]
      if (is.factor(vals)) vals <- as.character(vals)
      list(
        n_unique = length(unique(vals)),
        values = safe_head(vals, 100L)
      )
    }), cols)
  }

  list(
    dimensions = maybe_call(as.integer(dim(obj)), integer()),
    assays = assay_summary,
    reductions = reduction_names,
    metadata_columns = meta_cols,
    sample_like_columns = sample_cols,
    condition_like_columns = condition_cols,
    cluster_like_columns = cluster_cols,
    sample_like_values = column_levels(sample_cols),
    condition_like_values = column_levels(condition_cols),
    cluster_like_values = column_levels(cluster_cols),
    derived_5xfad_design = if ("sample" %in% meta_cols) derive_5xfad_design(md$sample) else NULL,
    active_ident_counts = as.list(active_ident_counts)
  )
}

summarize_object <- function(path) {
  info <- file.info(path)
  out <- list(
    path = path,
    exists = file.exists(path),
    size_bytes = if (file.exists(path)) unname(info$size) else NA_real_
  )
  if (!file.exists(path)) {
    out$status <- "missing"
    return(out)
  }

  obj <- readRDS(path)
  out$status <- "loaded"
  out$class <- paste(class(obj), collapse = "/")

  if (inherits(obj, "Seurat")) {
    if (!requireNamespace("Seurat", quietly = TRUE)) {
      out$status <- "loaded_uninspected"
      out$error <- "Object inherits from Seurat but package Seurat is unavailable."
      return(out)
    }
    out$seurat <- summarize_seurat(obj)
  } else if (is.list(obj)) {
    out$list_names <- safe_names(obj)
    out$list_element_classes <- stats::setNames(
      lapply(obj, function(x) paste(class(x), collapse = "/")),
      safe_names(obj)
    )
  } else {
    out$dimensions <- maybe_call(as.integer(dim(obj)), integer())
    out$names <- safe_names(obj)
  }
  out
}

results <- lapply(paths, summarize_object)

out_path <- file.path(
  repo_root,
  "outputs", "reports", "5xfad_snrna_rds_inspection.json"
)
dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
jsonlite::write_json(results, out_path, auto_unbox = TRUE, pretty = TRUE, null = "null")

cat("[inspect-5xfad-snrna] wrote ", out_path, "\n", sep = "")
for (res in results) {
  cat("[inspect-5xfad-snrna] ", basename(res$path), ": ", res$status, "\n", sep = "")
}
