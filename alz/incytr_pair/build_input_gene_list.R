#!/usr/bin/env Rscript
# Build the per-cluster gene.use ingredients for the levy_t5 males-only Seurat.
# Emits the raw, condition-keyed marker substrate; the pair-mode driver assembles
# the per-CONTRAST gene.use from it (the two contrast conditions' DEG ∪ prG):
#   allmarkers.csv  one-vs-rest FindAllMarkers (Type_condition idents), run BROAD
#                   (logfc.threshold = 0.1) to match sce4's frozen marker table
#                   (floor avg_log2FC ≈ 0.10). The driver takes the two contrast
#                   conditions' markers (avg_log2FC > 1 & p < 1e-4) as the
#                   per-contrast DEG arm — symmetric with prG's |aFC| > 1.
#
# sce4 reproduction (2026-05-31, docs/plans/sce4_reproduction.md §6.5):
# sce4 used DEG ∪ prG, NO HEG (zero HEG labels across its ma_2mo + ma_4mo
# reference). The former DEG>1.5 cutoff + logfc.threshold=1.2 pre-prune dropped
# real DEGs (Prkca/Grm5/Robo2, FC 1.0-1.2) that sce4 kept; HEG was the leaky
# patch that re-admitted them along with hundreds of non-DE genes (~15x
# over-emission). Broad markers + DEG>1 make HEG unnecessary, so it is removed.
#
# Adapted from data/incytr/v2_46clusters/provenance/run_input_gene_list.R.
# Two changes vs that source:
#   1. Idents and Type already set in alz/incytr_pair/build_pair_seurat.R; we only
#      need to compose Type_condition.
#   2. condition = `ma_<age>_<geno>` (compound) NOT plain Genotype, because
#      pair-mode runs disease vs WT at the *same* timepoint. Using bare
#      Genotype like the v2 script would average across timepoints.
#
# Run from any working directory (paths resolved via git rev-parse).
suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
  # presto: C++ Wilcoxon. Seurat::FindAllMarkers auto-detects it via
  # requireNamespace and swaps in the vectorized backend. Loading here
  # makes the dependency explicit and fails fast if uninstalled.
  library(presto)
  library(future)
})

# Resolve repo root so the script runs from any cwd.
REPO_ROOT <- system("git rev-parse --show-toplevel", intern = TRUE)
OUT_DIR   <- file.path(REPO_ROOT, "data", "derived", "incytr_inputs")
OBJ_PATH  <- file.path(OUT_DIR, "incytr_obj.rds")

# Parallelize FindAllMarkers across idents. multisession (not multicore)
# because Seurat objects are R6-ish and forking is fragile on Linux.
N_WORKERS <- min(8L, max(1L, parallel::detectCores() - 1L))
plan(multisession, workers = N_WORKERS)
# 8 GiB cap on globals exported to workers — the subset Seurat object
# is well under this; bump if you ever swap in the full v2 obj.
options(future.globals.maxSize = 8 * 1024^3)
cat("[input_gene_list] future plan: multisession, workers=", N_WORKERS, "\n",
    sep = "")

cat("[input_gene_list] reading", OBJ_PATH, "\n")
obj <- readRDS(OBJ_PATH)
cat("[input_gene_list] dim:", paste(dim(obj), collapse = " x "), "\n")

stopifnot("Type" %in% colnames(obj@meta.data),
          "condition" %in% colnames(obj@meta.data))

obj$Type_condition <- paste0(obj$Type, "_", obj$condition)
Idents(obj) <- "Type_condition"
cat("[input_gene_list] Type_condition idents (n=",
    length(levels(Idents(obj))), ")\n", sep = "")

t0 <- Sys.time()
cat("[input_gene_list] FindAllMarkers (only.pos=TRUE, presto + future)\n")
# logfc.threshold = 0.1 reproduces sce4's broad frozen marker table (floor
# avg_log2FC ~= 0.10). The driver applies the avg_log2FC > 1 DEG cutoff at
# gene.use assembly; a higher FindAllMarkers threshold here would pre-prune
# real DEGs (Prkca/Grm5/Robo2, FC 1.0-1.2) that sce4 kept. min.pct stays at
# the Seurat 5 default (0.01).
markers <- FindAllMarkers(
  obj,
  only.pos       = TRUE,
  logfc.threshold = 0.1,
  verbose        = FALSE
)
cat("[input_gene_list] markers rows:", nrow(markers), "\n")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
write.csv(markers, file.path(OUT_DIR, "allmarkers.csv"), row.names = FALSE)
cat("[input_gene_list] DONE in",
    round(difftime(Sys.time(), t0, units = "mins"), 2), "min\n")
