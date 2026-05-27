#!/usr/bin/env Rscript
# Emit the per-(cluster, Group) expression matrix that Incytr's `Cal_scFC` uses
# as its substrate, so the unified viewer's transcript-trace panel can show
# the same numbers that feed `*_sclog2FC`.
#
# Substrate: `Data.input@assays$originalexp@data` (LogNormalize log1p-CP10K),
# aggregated by (Type, Group) with the WEIGHTED-QUARTILE TRIMEAN
# (0.25*Q1 + 0.5*Q2 + 0.25*Q3, quantile type=7) — the exact per-group
# aggregation `Cal_scFC` consumes. Cal_scFC reads `object@expr.bygroup`
# (analysis.R:266), which `Cal_pairwise_grid` fills via a single
# `Expr_bygroup(mean_method = NULL)` call (grid.R:187); both Cal_SigProb and
# Cal_scFC (grid.R:209-210) read that one trimean-filled slot. The arithmetic
# mean does NOT reproduce `*_sclog2FC`: sparse genes whose trimean collapses to
# ~0 (Q1=Q2=0) but whose mean is positive diverge by several log2. We reuse
# Incytr's own kernel (`grouped_weighted_quartile`) for bitwise parity.
# Matching reader: `alz/integration/build_transcript_trace.py`.
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

# Per-(Type, Group) trimean via Incytr's own kernel, for bitwise parity with
# the `expr.bygroup` slot Cal_scFC consumes. Densify one level's cells at a
# time (genes × cells_in_level) to stay under the shared-box memory budget;
# partitioning per (Type,Group) level is identical to one Expr_bygroup call
# with Type labels per Group, since each cell's trimean depends only on the
# cells in its own (gene, group, type) set.
suppressPackageStartupMessages(library(Incytr))
idx <- match(key, levels_present)
genes <- rownames(mat)
n_g <- length(genes)
n_l <- length(levels_present)
parts <- strsplit(levels_present, "\t", fixed = TRUE)
# Gene-chunk size: bounds the dense submatrix to (GENE_CHUNK × cells_in_level).
# Trimean is per-gene independent, so chunking genes is exact.
GENE_CHUNK <- 4000L
gene_starts <- seq.int(1L, n_g, by = GENE_CHUNK)
chunks <- vector("list", n_l)
for (j in seq_len(n_l)) {
  cols <- which(idx == j)
  tri  <- numeric(n_g)
  for (gs in gene_starts) {
    ge <- min(gs + GENE_CHUNK - 1L, n_g)
    sub <- as.matrix(mat[gs:ge, cols, drop = FALSE])   # gene_chunk × cells, dense
    tri[gs:ge] <- Incytr:::grouped_weighted_quartile(sub, rep("g", length(cols)))[, 1]
  }
  chunks[[j]] <- data.frame(
    cluster = parts[[j]][1],
    group   = parts[[j]][2],
    gene    = genes,
    value   = as.numeric(tri),
    stringsAsFactors = FALSE
  )
  if (j %% 25 == 0) cat("[emit_expr_bygroup] trimean level", j, "/", n_l, "\n")
}
long <- do.call(rbind, chunks)

cat("[emit_expr_bygroup] writing", out_path,
    " rows:", nrow(long),
    " unique clusters:", length(unique(long$cluster)),
    " unique groups:", length(unique(long$group)), "\n")

write_parquet(long, out_path, compression = "zstd")
cat("[emit_expr_bygroup] done\n")
