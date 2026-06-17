#!/usr/bin/env Rscript
# Per-tissue 5xFAD allmarkers.csv (one-vs-rest FindAllMarkers, keyed by
# Type_condition). Cohort analog of build_tcells_input_gene_list.R: the pair-mode
# driver derives the per-contrast DEG gene.use from allmarkers.csv directly
# (avg_log2FC > 1 & p_val < 1e-4 on the contrast's two conditions) and computes
# prG per-contrast itself, so no pre-collapsed input_gene_list.csv is written.
# condition vocabulary is `<geno>_<age>` (e.g. "TG_3mo").
#
# 5xFAD has no sce4 reference, so the runner leaves SCE4_GENEUSE_DIR unset and the
# driver derives DEG u prG live — exactly the t-cell path.
#
# Output (under data/derived/5xfad_incytr_inputs/<tissue>/):
#   allmarkers.csv   raw FindAllMarkers output (Type_condition idents)
#
# Usage:  pixi run Rscript alz/incytr_pair/build_5xfad_input_gene_list.R <cortex|hippocampus>
suppressPackageStartupMessages({
  library(Seurat)
  library(presto)
  library(future)
})

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1L)
tissue <- args[[1]]
stopifnot(tissue %in% c("cortex", "hippocampus"))

REPO_ROOT  <- system("git rev-parse --show-toplevel", intern = TRUE)
TISSUE_DIR <- file.path(REPO_ROOT, "data", "derived", "5xfad_incytr_inputs", tissue)
OBJ_PATH   <- file.path(TISSUE_DIR, "incytr_obj.rds")
stopifnot(file.exists(OBJ_PATH))

# Sequential plan: presto's wilcoxauc is vectorized C++ and scores all 241
# Type_condition idents in one pass. The t-cell builder's multisession plan does
# NOT scale here — across 241 idents (vs the t-cell ~50) it re-serializes the
# 40k-cell object to each worker and thrashes/deadlocks on this shared box
# (observed: >40 min, no progress). presto alone is fast.
plan(sequential)
options(future.globals.maxSize = 8 * 1024^3)
cat("[input_gene_list] future plan: sequential (presto-vectorized)\n")

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
# logfc.threshold = 0.1 (broad): the driver applies the avg_log2FC > 1 DEG cutoff
# at gene.use assembly; a higher threshold here would pre-prune real DEGs in the
# (1.0, 1.2) band. Matches build_tcells_input_gene_list.R / build_input_gene_list.R.
markers <- FindAllMarkers(
  obj,
  only.pos        = TRUE,
  logfc.threshold = 0.1,
  verbose         = FALSE
)
cat("[input_gene_list] markers rows:", nrow(markers), "\n")
write.csv(markers, file.path(TISSUE_DIR, "allmarkers.csv"), row.names = FALSE)
cat("[input_gene_list] DONE in",
    round(difftime(Sys.time(), t0, units = "mins"), 2), "min\n")
