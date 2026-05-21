#!/usr/bin/env Rscript
# Build input_gene_list.csv (DEG ∪ HEG) for the levy_t5 males-only Seurat.
#
# Adapted from data/incytr/v2_46clusters/provenance/run_input_gene_list.R.
# Two changes vs that source:
#   1. Idents and Type already set in alz/incytr/build_pair_seurat.R; we only
#      need to compose Type_condition.
#   2. condition = `ma_<age>_<geno>` (compound) NOT plain Genotype, because
#      pair-mode runs disease vs WT at the *same* timepoint. Using bare
#      Genotype like the v2 script would average across timepoints.
#
# Outputs (under data/derived/incytr_inputs/):
#   allmarkers.csv      raw FindAllMarkers output
#   HEG_df.csv          per-(condition, cluster) high-expression genes
#   input_gene_list.csv (gene, cluster) union, dedup
#
# Run from any working directory (paths resolved via git rev-parse).
suppressPackageStartupMessages({
  library(Seurat)
  library(stringr)
  library(Matrix)
  library(matrixStats)
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

# Vectorized HEG: trimean per gene, threshold = 75th-pct of nonzero entries
# in the condition's full matrix. Mirrors legacy Find_highexp_gene defaults.
find_highexp_vec <- function(data_cond, meta_cond, group_col, cutoff_percentile = 0.75) {
  nz <- data_cond@x[data_cond@x > 0]
  cutoff_exp <- as.numeric(quantile(nz, cutoff_percentile))
  groups <- sort(unique(as.character(meta_cond[[group_col]])))
  out <- vector("list", length(groups))
  for (i in seq_along(groups)) {
    g <- groups[i]
    cells <- rownames(meta_cond)[meta_cond[[group_col]] == g]
    if (length(cells) == 0) next
    m <- as.matrix(data_cond[, cells, drop = FALSE])
    qs <- matrixStats::rowQuantiles(m, probs = c(0.25, 0.5, 0.75))
    tri <- 0.25 * qs[, 1] + 0.5 * qs[, 2] + 0.25 * qs[, 3]
    keep <- which(tri >= cutoff_exp)
    if (length(keep) == 0) next
    out[[i]] <- data.frame(
      gene_symbol = rownames(m)[keep],
      ave.exp     = unname(tri[keep]),
      cluster     = g,
      stringsAsFactors = FALSE
    )
  }
  do.call(rbind, out)
}

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
# logfc.threshold = 1.2 < the post-filter cutoff (1.5), so we skip Wilcoxon
# tests on genes that could not pass downstream anyway. min.pct stays at the
# Seurat 5 default (0.01).
markers <- FindAllMarkers(
  obj,
  only.pos       = TRUE,
  logfc.threshold = 1.2,
  verbose        = FALSE
)
cat("[input_gene_list] markers rows:", nrow(markers), "\n")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
write.csv(markers, file.path(OUT_DIR, "allmarkers.csv"))

Idents(obj) <- "Type"
cell.groups <- sort(unique(obj$Type))
conditions  <- sort(unique(obj$condition))
cat("[input_gene_list] HEG loop:", length(conditions), "conditions x ",
    length(cell.groups), "clusters\n")

HEG.list <- vector("list", length(conditions))
for (k in seq_along(conditions)) {
  t_k <- Sys.time()
  cat("[input_gene_list]  condition ", k, "/", length(conditions), ": ",
      conditions[k], " ", sep = "")
  sub <- subset(obj, subset = condition == conditions[k])
  data_mat <- GetAssayData(sub, layer = "data", assay = "originalexp")
  HE <- find_highexp_vec(data_mat, sub@meta.data, group_col = "Type",
                         cutoff_percentile = 0.75)
  if (!is.null(HE) && nrow(HE) > 0) {
    HE$condition <- conditions[k]
    HEG.list[[k]] <- HE
  }
  cat("rows=", if (is.null(HE)) 0 else nrow(HE),
      " (", round(as.numeric(difftime(Sys.time(), t_k, units = "secs")), 1),
      "s)\n", sep = "")
}
HEG.df <- do.call(rbind, HEG.list)
if (is.null(HEG.df)) {
  HEG.df <- data.frame(
    gene_symbol = character(), ave.exp = numeric(),
    cluster = character(), condition = character(),
    stringsAsFactors = FALSE
  )
}
cat("[input_gene_list] HEG rows:", nrow(HEG.df), "\n")
write.csv(HEG.df, file.path(OUT_DIR, "HEG_df.csv"))

# Combine markers + HEG, dedupe to (gene, cluster).
HEG.unique <- unique(HEG.df[, c("gene_symbol", "cluster")])
names(HEG.unique) <- c("gene", "cluster")

m <- markers[markers$avg_log2FC > 1.5 & markers$p_val < 1e-4, ]
# `cluster` here is Type_condition like "Astrocytes_ma_2mo_WTyp" — strip suffix.
# Cluster names contain no underscores (asserted in build_pair_seurat.R),
# so the prefix up to the first "_" recovers the cluster.
m$cluster <- str_extract(as.character(m$cluster), "[^_]+")
m <- unique(m[, c("gene", "cluster")])

out <- unique(rbind(m, HEG.unique))
cat("[input_gene_list] combined unique (gene, cluster):", nrow(out), "\n")
write.csv(out, file.path(OUT_DIR, "input_gene_list.csv"))
cat("[input_gene_list] DONE in",
    round(difftime(Sys.time(), t0, units = "mins"), 2), "min\n")
