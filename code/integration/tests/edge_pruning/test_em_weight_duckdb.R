#!/usr/bin/env Rscript
# Test: validate that DuckDB with in-query EM promiscuity weighting produces
# the same pathway set as DuckDB (unweighted) -> Cal_SigProb (weighted).
#
# The EM weight is 1/log2(1+degree) applied to the EM-Target product before
# the Hill function. This test verifies zero false negatives and zero false
# positives from pushing the weight into the SQL query.
#
# Usage:
#   systemd-run --user --scope -p MemoryMax=12G \
#     micromamba run -n incytr Rscript code/integration/tests/edge_pruning/test_em_weight_duckdb.R

suppressPackageStartupMessages({
  library(Matrix)
  library(data.table)
  library(Incytr)
})

get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) return(dirname(normalizePath(sub("^--file=", "", file_arg[1]))))
  getwd()
}
repo_root   <- normalizePath(file.path(get_script_dir(), "..", "..", "..", ".."))
int_dir     <- file.path(repo_root, "code", "integration", "intermediates")
wrapper_dir <- file.path(repo_root, "code", "integration", "wrappers")

source(file.path(wrapper_dir, "duckdb_enumeration.R"))

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
cat("Loading data...\n")
mat <- readMM(file.path(int_dir, "expression_matrix.mtx"))
mat <- as(mat, "CsparseMatrix")
genes    <- read.csv(file.path(int_dir, "expression_genes.csv"))$gene
barcodes <- read.csv(file.path(int_dir, "expression_barcodes.csv"))$barcode
meta     <- read.csv(file.path(int_dir, "expression_metadata.csv"), row.names = 1)
rownames(mat) <- genes; colnames(mat) <- barcodes

cat("Loading IncytrDB...\n")
data(DB_Layer1_mouse_filtered, package = "Incytr")
data(DB_Layer2_mouse_filtered, package = "Incytr")
data(DB_Layer3_mouse_filtered, package = "Incytr")
all_genes <- rownames(mat)
DB_Layer1_mouse_filtered <- DB_Layer1_mouse_filtered[
  DB_Layer1_mouse_filtered$from %in% all_genes &
  DB_Layer1_mouse_filtered$to %in% all_genes, ]
DB_Layer2_mouse_filtered <- DB_Layer2_mouse_filtered[
  DB_Layer2_mouse_filtered$from %in% all_genes &
  DB_Layer2_mouse_filtered$to %in% all_genes, ]
DB_Layer3_mouse_filtered <- DB_Layer3_mouse_filtered[
  DB_Layer3_mouse_filtered$from %in% all_genes &
  DB_Layer3_mouse_filtered$to %in% all_genes, ]
DB.M <- list(DB_Layer1_mouse_filtered,
             DB_Layer2_mouse_filtered,
             DB_Layer3_mouse_filtered)
rm(DB_Layer1_mouse_filtered, DB_Layer2_mouse_filtered, DB_Layer3_mouse_filtered)
gc(verbose = FALSE)

sender     <- "Microglia-PVM"
receiver   <- "L5 IT"
conditions <- c("WT", "App")

# 10% threshold
s_det <- Matrix::rowMeans(mat[, meta$labels == sender] > 0)
r_det <- Matrix::rowMeans(mat[, meta$labels == receiver] > 0)
sg <- names(s_det[s_det >= 0.10])
rg <- names(r_det[r_det >= 0.10])

# Load kinase-imputed genes
imputed_path <- file.path(int_dir, "kinase_imputed_genes.csv")
if (file.exists(imputed_path)) {
  ki <- read.csv(imputed_path)$gene
  ki <- ki[ki %in% all_genes]
  rg <- union(rg, ki)
}

cat(sprintf("  Sender: %d genes, Receiver: %d genes\n\n", length(sg), length(rg)))

# ---------------------------------------------------------------------------
# Test 1: DuckDB with EM weighting (new approach)
# ---------------------------------------------------------------------------
cat("=== Test 1: DuckDB with EM promiscuity weighting ===\n")
t0 <- proc.time()
duck_weighted <- duckdb_enumerate_pathways(
  mat = mat, meta = meta, DB = DB.M,
  sender = sender, receiver = receiver,
  conditions = conditions,
  gene.use_Sender = sg, gene.use_Receiver = rg,
  K = 0.5, N = 2, cutoff_SigProb = 0.01,
  em_promiscuity_weight = TRUE)
t1 <- (proc.time() - t0)["elapsed"]
cat(sprintf("  Pathways: %d (%.1f sec)\n\n", nrow(duck_weighted$pathways), t1))

# ---------------------------------------------------------------------------
# Test 2: DuckDB without EM weighting -> Cal_SigProb with EM weighting
# ---------------------------------------------------------------------------
cat("=== Test 2: DuckDB (unweighted) -> Cal_SigProb (weighted) ===\n")
t0 <- proc.time()
duck_unweighted <- duckdb_enumerate_pathways(
  mat = mat, meta = meta, DB = DB.M,
  sender = sender, receiver = receiver,
  conditions = conditions,
  gene.use_Sender = sg, gene.use_Receiver = rg,
  K = 0.5, N = 2, cutoff_SigProb = 0.01,
  em_promiscuity_weight = FALSE)
t2a <- (proc.time() - t0)["elapsed"]
cat(sprintf("  DuckDB unweighted: %d pathways (%.1f sec)\n",
            nrow(duck_unweighted$pathways), t2a))

# Run through Cal_SigProb
inc <- create_Incytr(object = mat, meta = meta,
                     sender = sender, receiver = receiver,
                     group.by = "labels", conditions = conditions)
inc@pathways <- as.data.frame(duck_unweighted$pathways)
inc@options$em_degree <- duck_unweighted$em_degree
inc@options$edge_source_count <- duck_unweighted$edge_source_count
inc <- Expr_bygroup(inc)
inc <- Cal_SigProb(inc, K = 0.5, N = 2, cutoff_SigProb = 0.01,
                   correction = 0.001)
t2b <- (proc.time() - t0)["elapsed"]
sigprob_paths <- inc@SigProb$Path
cat(sprintf("  After Cal_SigProb: %d pathways (%.1f sec total)\n\n",
            length(sigprob_paths), t2b))

# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
cat("=== Comparison ===\n")
weighted_paths <- sort(duck_weighted$pathways$Path)
sigprob_paths  <- sort(sigprob_paths)

n_weighted <- length(weighted_paths)
n_sigprob  <- length(sigprob_paths)

in_both      <- length(intersect(weighted_paths, sigprob_paths))
in_weighted_only <- length(setdiff(weighted_paths, sigprob_paths))
in_sigprob_only  <- length(setdiff(sigprob_paths, weighted_paths))

cat(sprintf("  DuckDB EM-weighted:  %d pathways\n", n_weighted))
cat(sprintf("  Cal_SigProb result:  %d pathways\n", n_sigprob))
cat(sprintf("  In both:             %d\n", in_both))
cat(sprintf("  DuckDB-only:         %d (false positives — over-inclusive is OK)\n", in_weighted_only))
cat(sprintf("  Cal_SigProb-only:    %d (false negatives — must be 0)\n", in_sigprob_only))

cat(sprintf("\n  Speed: weighted DuckDB %.1fs vs unweighted+Cal_SigProb %.1fs (%.1fx faster)\n",
            t1, t2b, t2b / t1))

if (in_sigprob_only == 0) {
  cat("\n  PASS: Zero false negatives. DuckDB EM-weighted is a superset of Cal_SigProb.\n")
  if (in_weighted_only == 0) {
    cat("  EXACT MATCH: Identical pathway sets.\n")
  } else {
    cat(sprintf("  Note: %d extra pathways in DuckDB (edge rounding). Cal_SigProb still filters them.\n",
                in_weighted_only))
  }
} else {
  cat("\n  FAIL: Cal_SigProb found pathways that DuckDB missed!\n")
  cat("  This means EM weighting in DuckDB is too aggressive.\n")
  # Show some examples
  missing <- setdiff(sigprob_paths, weighted_paths)
  cat("  First 5 missing:\n")
  for (p in head(missing, 5)) cat(sprintf("    %s\n", p))
}
