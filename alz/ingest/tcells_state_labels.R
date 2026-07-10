# Shared reader and validator for cycle-independent per-cell T-cell labels.
#
# Both raw-RDS consumers source this file so barcode alignment, contaminant
# handling, and type validation have one implementation.

load_tcell_state_labels <- function(donor, barcodes, days, seurat_clusters,
                                    repo_root = ".") {
  path <- file.path(
    repo_root, "outputs", "reports", "tcell_labeling", "cells",
    paste0(donor, "_state_labels.csv")
  )
  if (!file.exists(path)) {
    stop("per-cell state labels missing — run `pixi run tcells-label`: ", path)
  }

  labels <- read.csv(
    path,
    stringsAsFactors = FALSE,
    check.names = FALSE,
    na.strings = c("", "NA")
  )
  required <- c(
    "barcode", "donor", "seurat_cluster", "day", "lineage", "label", "type"
  )
  missing_columns <- setdiff(required, colnames(labels))
  if (length(missing_columns)) {
    stop("state-label artifact missing column(s): ",
         paste(missing_columns, collapse = ", "))
  }
  if (any(is.na(labels$barcode) | labels$barcode == "")) {
    stop("state-label artifact has blank barcodes")
  }
  if (anyDuplicated(labels$barcode)) {
    duplicate_barcodes <- unique(labels$barcode[duplicated(labels$barcode)])
    stop("state-label artifact has duplicate barcode(s): ",
         paste(head(duplicate_barcodes, 5L), collapse = ", "))
  }
  if (length(barcodes) != length(days) || length(barcodes) != length(seurat_clusters)) {
    stop("object barcode/day/cluster vectors have different lengths")
  }
  if (anyDuplicated(barcodes)) stop("Seurat object has duplicate barcodes")

  missing_barcodes <- setdiff(barcodes, labels$barcode)
  extra_barcodes <- setdiff(labels$barcode, barcodes)
  if (length(missing_barcodes) || length(extra_barcodes)) {
    stop(
      "state-label/object barcode mismatch: missing=", length(missing_barcodes),
      " extra=", length(extra_barcodes)
    )
  }

  aligned <- labels[match(barcodes, labels$barcode), , drop = FALSE]
  if (!identical(as.character(aligned$barcode), as.character(barcodes))) {
    stop("state-label barcode alignment failed")
  }
  if (any(is.na(aligned$donor)) || any(aligned$donor != donor)) {
    stop("state-label donor column does not match requested donor ", donor)
  }

  days <- as.integer(days)
  artifact_days <- as.integer(aligned$day)
  if (any(is.na(artifact_days)) || any(artifact_days != days)) {
    stop("state-label days disagree with the Seurat metadata for ", donor)
  }

  clusters <- as.integer(as.character(seurat_clusters))
  artifact_clusters <- as.integer(aligned$seurat_cluster)
  if (any(is.na(clusters)) || any(is.na(artifact_clusters)) ||
      any(artifact_clusters != clusters)) {
    stop("state-label clusters disagree with the Seurat metadata for ", donor)
  }

  type <- trimws(as.character(aligned$type))
  type[is.na(type) | type == ""] <- NA_character_
  label <- trimws(as.character(aligned$label))
  if (any(is.na(label) | label == "")) stop("state-label artifact has blank labels")

  contaminant <- label == "contaminant"
  if (any(is.na(type) != contaminant)) {
    stop("blank type must identify exactly the contaminant cells")
  }
  bad_type <- unique(type[!is.na(type) & grepl("[^A-Za-z0-9]", type)])
  if (length(bad_type)) {
    stop("non-alphanumeric Incytr type(s): ", paste(bad_type, collapse = ", "))
  }
  expected_types <- c(
    "CD4", "CD4Activated", "CD4Cytotoxic", "CD4ExhaustionAssociated", "CD4NaiveMemory",
    "CD8", "CD8Activated", "CD8CytotoxicEffector", "CD8Exhausted",
    "CD8NaiveMemory"
  )
  unexpected_types <- setdiff(unique(type[!is.na(type)]), expected_types)
  if (length(unexpected_types)) {
    stop("unexpected per-cell marker type(s): ",
         paste(unexpected_types, collapse = ", "))
  }

  list(
    path = normalizePath(path, mustWork = TRUE),
    cells = aligned,
    type = type,
    label = label,
    keep = !contaminant
  )
}
