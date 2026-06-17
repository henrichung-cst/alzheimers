#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  if (!requireNamespace("Matrix", quietly = TRUE)) {
    stop("Missing required R package: Matrix", call. = FALSE)
  }
  if (!requireNamespace("Seurat", quietly = TRUE)) {
    stop("Missing required R package: Seurat", call. = FALSE)
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

rds_path <- if (length(args) >= 1) {
  args[[1]]
} else {
  file.path(
    repo_root,
    "data", "datasets", "5xFAD", "primary", "scrna", "reclustering",
    "fivex_renamed_from_merged.RDS"
  )
}
out_path <- if (length(args) >= 2) {
  args[[2]]
} else {
  file.path(repo_root, "outputs", "reports", "5xfad_snrna_transgene_audit.csv")
}

parse_sample_design <- function(samples) {
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
  data.frame(
    sample = samples,
    sample_label_genotype = ifelse(parsed$genotype_raw == "5XFAD", "TG", "WT"),
    age = sub("^0", "", parsed$age_raw),
    tissue = ifelse(parsed$tissue_raw == "C", "cortex", "hippocampus"),
    sample_number = parsed$sample_number,
    stringsAsFactors = FALSE
  )
}

get_counts <- function(obj) {
  assay <- Seurat::DefaultAssay(obj)
  tryCatch(
    Seurat::GetAssayData(obj, assay = assay, layer = "counts"),
    error = function(e) Seurat::GetAssayData(obj, assay = assay, slot = "counts")
  )
}

obj <- readRDS(rds_path)
if (!inherits(obj, "Seurat")) {
  stop("Input RDS is not a Seurat object: ", rds_path, call. = FALSE)
}
if (!"sample" %in% colnames(obj@meta.data)) {
  stop("Seurat metadata is missing required column: sample", call. = FALSE)
}

counts <- get_counts(obj)
genes <- c("TgAPP", "TgPSEN1")
missing_genes <- setdiff(genes, rownames(counts))
if (length(missing_genes) > 0) {
  stop("Missing transgene rows: ", paste(missing_genes, collapse = ", "), call. = FALSE)
}

samples <- sort(unique(as.character(obj@meta.data$sample)))
design <- parse_sample_design(samples)
sample_index <- split(seq_len(ncol(counts)), as.character(obj@meta.data$sample))

metric_rows <- do.call(rbind, lapply(samples, function(sample) {
  cols <- sample_index[[sample]]
  values <- counts[genes, cols, drop = FALSE]
  data.frame(
    sample = sample,
    cells = length(cols),
    TgAPP_count_sum = as.numeric(Matrix::rowSums(values)["TgAPP"]),
    TgAPP_detected_cells = as.integer(Matrix::rowSums(values["TgAPP", , drop = FALSE] > 0)["TgAPP"]),
    TgPSEN1_count_sum = as.numeric(Matrix::rowSums(values)["TgPSEN1"]),
    TgPSEN1_detected_cells = as.integer(Matrix::rowSums(values["TgPSEN1", , drop = FALSE] > 0)["TgPSEN1"]),
    stringsAsFactors = FALSE
  )
}))

out <- merge(design, metric_rows, by = "sample", sort = TRUE)
dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
utils::write.csv(out, out_path, row.names = FALSE, quote = TRUE)

cat("[audit-5xfad-snrna-transgenes] wrote ", out_path, "\n", sep = "")
cat("[audit-5xfad-snrna-transgenes] samples=", nrow(out), "\n", sep = "")
