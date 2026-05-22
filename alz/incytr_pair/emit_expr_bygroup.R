#!/usr/bin/env Rscript
# Emit the per-(cluster, Group) expression matrix that Incytr's `Cal_scFC` uses
# as its substrate, so the unified viewer's transcript-trace panel can show
# the same numbers that feed `*_sclog2FC`.
#
# Substrate: `Data.input@assays$originalexp@data` (LogNormalize log1p-CP10K),
# averaged by (Type, Group) — bit-for-bit equivalent to what
# `Expr_bygroup(..., mean_method = "mean")` → `compute_group_expr` produces
# inside the pair-mode driver. The matching reader is
# `alz/integration/build_transcript_trace.py`.
#
# Output: outputs/reports/decomposition/levy_t5/transcript_per_cluster.parquet
#   columns: cluster (Type level), group (Group level), gene, value
#
# Contrast-invariant: runs once per spine build, not per (genotype, age).
#
# Can be invoked from any working directory; paths are resolved relative to
# the repo root detected via git.

suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
  library(arrow)
})

# Resolve repo root so the script runs from any cwd.
repo_root <- system("git rev-parse --show-toplevel", intern = TRUE)
rds_path <- file.path(repo_root, "data", "derived", "incytr_inputs", "incytr_obj.rds")
out_dir  <- file.path(repo_root, "outputs", "reports", "decomposition", "levy_t5")
out_path <- file.path(out_dir, "transcript_per_cluster.parquet")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

cat("[emit_expr_bygroup] loading", rds_path, "\n")
obj <- readRDS(rds_path)
obj@meta.data$Type <- obj@active.ident

mat <- obj@assays$originalexp@data   # sparse dgCMatrix, log1p-CP10K
meta <- obj@meta.data
stopifnot(ncol(mat) == nrow(meta))

types  <- as.character(meta$Type)
groups <- as.character(meta$Group)
key    <- paste(types, groups, sep = "\t")
levels_present <- sort(unique(key))

cat("[emit_expr_bygroup] cells:", ncol(mat),
    "  (Type x Group) levels present:", length(levels_present),
    "  genes:", nrow(mat), "\n")

# Build a sparse cell -> level indicator, then column-mean per level. This is
# equivalent to: for each (Type, Group), mean of `originalexp@data` across the
# member cells — i.e. exactly what compute_group_expr computes.
idx <- match(key, levels_present)
n_per_level <- as.integer(table(factor(idx, levels = seq_along(levels_present))))
indic <- sparseMatrix(
  i = seq_along(idx),
  j = idx,
  x = 1 / n_per_level[idx],
  dims = c(ncol(mat), length(levels_present))
)
group_mean <- mat %*% indic   # gene × level, dense-ish

genes <- rownames(mat)
n_g <- length(genes)
n_l <- length(levels_present)

# Long-form pivot. Avoid building a huge intermediate dense matrix: write the
# columns one level at a time and rbind.
chunks <- vector("list", n_l)
parts <- strsplit(levels_present, "\t", fixed = TRUE)
for (j in seq_len(n_l)) {
  vals <- as.numeric(group_mean[, j])
  chunks[[j]] <- data.frame(
    cluster = parts[[j]][1],
    group   = parts[[j]][2],
    gene    = genes,
    value   = vals,
    stringsAsFactors = FALSE
  )
}
long <- do.call(rbind, chunks)

cat("[emit_expr_bygroup] writing", out_path,
    " rows:", nrow(long),
    " unique clusters:", length(unique(long$cluster)),
    " unique groups:", length(unique(long$group)), "\n")

write_parquet(long, out_path, compression = "zstd")
cat("[emit_expr_bygroup] done\n")
