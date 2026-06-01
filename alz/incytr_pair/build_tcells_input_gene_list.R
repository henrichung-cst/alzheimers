#!/usr/bin/env Rscript
# Per-donor T-cell allmarkers.csv (one-vs-rest FindAllMarkers, keyed by
# Type_condition). Mirrors alz/incytr_pair/build_input_gene_list.R (mouse): the
# pair-mode driver derives the per-contrast DEG gene.use from allmarkers.csv
# directly and computes prG per-contrast itself, so this script no longer writes
# a pre-collapsed input_gene_list.csv (dropped 2026-05-31 alongside the mouse
# path; the all-condition DEG union was the dominant over-emission source vs
# sce4). condition vocabulary is `d{day}` (e.g. "d13"), not "ma_<age>_<geno>".
#
# Output (under data/derived/tcells_incytr_inputs/<donor>/):
#   allmarkers.csv       raw FindAllMarkers output (Type_condition idents)
#
# Usage:  pixi run Rscript alz/incytr_pair/build_tcells_input_gene_list.R <donor1|donor2>
suppressPackageStartupMessages({
  library(Seurat)
  library(presto)
  library(future)
})

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1L)
donor <- args[[1]]

REPO_ROOT <- system("git rev-parse --show-toplevel", intern = TRUE)
DONOR_DIR <- file.path(REPO_ROOT, "data", "derived", "tcells_incytr_inputs", donor)
OBJ_PATH  <- file.path(DONOR_DIR, "incytr_obj.rds")
stopifnot(file.exists(OBJ_PATH))

N_WORKERS <- min(4L, max(1L, parallel::detectCores() - 1L))
plan(multisession, workers = N_WORKERS)
options(future.globals.maxSize = 8 * 1024^3)
cat("[input_gene_list] future plan: multisession, workers=", N_WORKERS, "\n",
    sep = "")

cat("[input_gene_list] reading", OBJ_PATH, "\n")
obj <- readRDS(OBJ_PATH)
cat("[input_gene_list] dim:", paste(dim(obj), collapse = " x "), "\n")
stopifnot("Type" %in% colnames(obj@meta.data),
          "condition" %in% colnames(obj@meta.data),
          "Type_condition" %in% colnames(obj@meta.data))

Idents(obj) <- "Type_condition"
cat("[input_gene_list] Type_condition idents (n=",
    length(levels(Idents(obj))), ")\n", sep = "")

t0 <- Sys.time()
cat("[input_gene_list] FindAllMarkers (only.pos=TRUE, presto + future)\n")
# logfc.threshold = 0.1 (broad), matching the mouse path: the driver applies the
# avg_log2FC > 1 DEG cutoff at gene.use assembly; a higher threshold here would
# pre-prune real DEGs in the (1.0, 1.2) band. See build_input_gene_list.R.
markers <- FindAllMarkers(
  obj,
  only.pos        = TRUE,
  logfc.threshold = 0.1,
  verbose         = FALSE
)
cat("[input_gene_list] markers rows:", nrow(markers), "\n")
write.csv(markers, file.path(DONOR_DIR, "allmarkers.csv"), row.names = FALSE)
cat("[input_gene_list] DONE in",
    round(difftime(Sys.time(), t0, units = "mins"), 2), "min\n")
