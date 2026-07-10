#!/usr/bin/env Rscript
# Per-donor T-cell Seurat for Incytr pair-mode.
#
# Reads the raw donor RDS, joins the cycle-independent per-cell label artifact,
# drops contaminant cells (blank Incytr type), and writes a slim Seurat with:
#   - Idents()       = state (Type)
#   - obj$Type       = state
#   - obj$condition  = sprintf("d%d", day)         (e.g. "d13")
#   - obj$Type_condition = paste0(Type, "_", condition)
#
# Pair-mode driver consumes this as data/derived/tcells_incytr_inputs/<donor>/incytr_obj.rds.
#
# Memory pattern carried from tcells_scrna_extract.R: DietSeurat → RNA only,
# drop scale.data immediately, subset down to labeled cells before any compute.
#
# Usage:  pixi run Rscript alz/incytr_pair/build_tcells_seurat.R <donor1|donor2>
suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
})

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1L)
donor <- args[[1]]

cfg <- list(
  donor1 = list(
    rds     = "data/datasets/tcells/donor1/scrna/Tcells.singlet.rds",
    day_col = "sample_ID"),
  donor2 = list(
    rds     = "data/datasets/tcells/donor2/scrna/Tcells_d2.singlet (1).rds",
    day_col = "Sample_Label")
)[[donor]]
stopifnot(!is.null(cfg))

REPO_ROOT <- system("git rev-parse --show-toplevel", intern = TRUE)
source(file.path(REPO_ROOT, "alz", "ingest", "tcells_state_labels.R"))

outdir <- file.path("data/derived/tcells_incytr_inputs", donor)

memline <- function(tag) {
  gc(full = TRUE)
  mi <- tryCatch(readLines("/proc/meminfo", n = 3), error = function(e) character())
  avail <- sub(".*?:\\s*", "", grep("MemAvailable", mi, value = TRUE))
  cat(sprintf("[mem %-18s] MemAvailable: %s\n", tag, ifelse(length(avail), avail, "?")))
}

cat("==== BUILD INCYTR SEURAT", donor, "====\n")
memline("start")

t0 <- Sys.time()
obj <- readRDS(cfg$rds)
cat("read in", round(as.numeric(Sys.time() - t0, units = "secs"), 1), "s ;",
    "dim", paste(dim(obj), collapse = " x "), "\n")
memline("after readRDS")

DefaultAssay(obj) <- "RNA"
obj <- DietSeurat(obj, assays = "RNA", dimreducs = NULL, graphs = NULL)
obj[["RNA"]]$scale.data <- NULL
memline("after DietSeurat")

# Day parsing ---------------------------------------------------------------
day_raw <- as.character(obj@meta.data[[cfg$day_col]])
day <- as.integer(sub(".*[Dd]ay[_ ]?(\\d+).*", "\\1", day_raw))
if (any(is.na(day))) {
  bad_day <- sort(unique(day_raw[is.na(day)]))
  stop("unparsed day labels: ", paste(bad_day, collapse = " | "))
}
obj$ts_day <- day

# Cycle-independent per-cell marker type -----------------------------------
joined <- load_tcell_state_labels(
  donor = donor,
  barcodes = colnames(obj),
  days = day,
  seurat_clusters = obj$seurat_clusters,
  repo_root = REPO_ROOT
)
obj$state <- joined$type
obj$state_label <- joined$label

# Drop contaminants ---------------------------------------------------------
n_total <- ncol(obj)
keep <- joined$keep
n_kept <- sum(keep)
drop_label_tab <- table(label = joined$label[!keep])
cat("cells: total=", n_total, " kept=", n_kept, " (",
    round(100 * n_kept / n_total, 1), "%) dropped=", n_total - n_kept, "\n", sep = "")
cat("drop breakdown by per-cell label:\n"); print(drop_label_tab)

obj <- subset(obj, cells = colnames(obj)[keep])
memline("after subset")

# Set driver-expected metadata ---------------------------------------------
obj$Type      <- as.character(obj$state)
obj$condition <- sprintf("d%d", obj$ts_day)
obj$Type_condition <- paste0(obj$Type, "_", obj$condition)

# Alphanumeric guard (Incytr driver splits Type_condition on `_`).
stopifnot(!any(grepl("[^A-Za-z0-9]", obj$Type)))
stopifnot(all(grepl("^d\\d+$", obj$condition)))

Idents(obj) <- factor(obj$Type, levels = sort(unique(obj$Type)))

dst <- file.path(outdir, "incytr_obj.rds")
cat("[seurat] writing", dst, "\n")
saveRDS(obj, dst)
cat("[seurat] final dim:", paste(dim(obj), collapse = " x "),
    " conditions=", length(unique(obj$condition)),
    " types=", length(unique(obj$Type)), "\n", sep = "")
cat("==== DONE", donor, "->", dst, "====\n")
