#!/usr/bin/env Rscript
# Test: pre-prune DB edges using per-edge SigProb before pathway enumeration.
#
# Idea: SigProb = hill(L*R) * hill(R*EM) * hill(EM*T). Each factor depends
# only on one edge (two adjacent genes). If any factor < cutoff, the full
# product is guaranteed < cutoff. So we can compute Hill values per edge,
# remove edges that can't contribute to a surviving pathway, and only then
# run pathway_inference on the smaller DB.
#
# This script:
#   1. Loads real data (expression matrix + DB layers)
#   2. At 50% threshold: runs standard and pruned pipelines, verifies identical results
#   3. At lower thresholds: runs pruned pipeline, measures reduction
#
# Usage (from repo root):
#   micromamba run -n incytr Rscript code/integration/tests/edge_pruning/test_edge_pruning.R

suppressPackageStartupMessages({
  library(Matrix)
  library(data.table)
  library(Incytr)
})

hill <- function(x, K = 0.5, N = 2) x^N / (x^N + K^N)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) return(dirname(normalizePath(sub("^--file=", "", file_arg[1]))))
  getwd()
}
repo_root <- normalizePath(file.path(get_script_dir(), "..", "..", "..", ".."))
int_dir   <- file.path(repo_root, "code", "integration", "intermediates")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
cat("Loading expression data...\n")
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
cat(sprintf("  Filtered DB: L1=%d, L2=%d, L3=%d edges\n\n",
            nrow(DB.M[[1]]), nrow(DB.M[[2]]), nrow(DB.M[[3]])))

sender   <- "Microglia-PVM"
receiver <- "L5 IT"
conditions <- c("WT", "App")
K <- 0.5; N <- 2; CUTOFF <- 0.01

# ---------------------------------------------------------------------------
# Pre-compute per-condition group means (for edge Hill values)
# ---------------------------------------------------------------------------
cat("Computing per-condition mean expression...\n")
s_cells <- which(meta$labels == sender)
r_cells <- which(meta$labels == receiver)
s_det   <- Matrix::rowMeans(mat[, s_cells] > 0)
r_det   <- Matrix::rowMeans(mat[, r_cells] > 0)

cond1_s <- which(meta$labels == sender   & meta$condition == "WT")
cond2_s <- which(meta$labels == sender   & meta$condition == "App")
cond1_r <- which(meta$labels == receiver & meta$condition == "WT")
cond2_r <- which(meta$labels == receiver & meta$condition == "App")

s_mean_c1 <- setNames(Matrix::rowMeans(mat[, cond1_s]), all_genes)
s_mean_c2 <- setNames(Matrix::rowMeans(mat[, cond2_s]), all_genes)
r_mean_c1 <- setNames(Matrix::rowMeans(mat[, cond1_r]), all_genes)
r_mean_c2 <- setNames(Matrix::rowMeans(mat[, cond2_r]), all_genes)

# ---------------------------------------------------------------------------
# Edge pruning function
# ---------------------------------------------------------------------------
prune_layer <- function(layer, from_c1, from_c2, to_c1, to_c2,
                        from_genes, to_genes, cutoff, K, N) {
  dt <- as.data.table(layer)
  dt <- dt[from %in% from_genes & to %in% to_genes]
  n_before <- nrow(dt)

  # Compute Hill value of gene product for both conditions
  h1 <- hill(from_c1[dt$from] * to_c1[dt$to], K, N)
  h2 <- hill(from_c2[dt$from] * to_c2[dt$to], K, N)

  # Keep if either condition exceeds cutoff
  dt <- dt[(h1 >= cutoff) | (h2 >= cutoff)]
  n_after <- nrow(dt)

  cat(sprintf("    %d -> %d edges (%.0f%% removed)\n",
              n_before, n_after, 100 * (1 - n_after / max(n_before, 1))))
  as.data.frame(dt)
}

prune_db <- function(DB, sender_genes, receiver_genes) {
  cat("  Pruning L1 (Ligand x Receptor):\n")
  p1 <- prune_layer(DB[[1]], s_mean_c1, s_mean_c2, r_mean_c1, r_mean_c2,
                     sender_genes, receiver_genes, CUTOFF, K, N)
  cat("  Pruning L2 (Receptor x EM):\n")
  p2 <- prune_layer(DB[[2]], r_mean_c1, r_mean_c2, r_mean_c1, r_mean_c2,
                     receiver_genes, receiver_genes, CUTOFF, K, N)
  cat("  Pruning L3 (EM x Target):\n")
  p3 <- prune_layer(DB[[3]], r_mean_c1, r_mean_c2, r_mean_c1, r_mean_c2,
                     receiver_genes, receiver_genes, CUTOFF, K, N)
  list(p1, p2, p3)
}

# ---------------------------------------------------------------------------
# Helper: run pipeline
# ---------------------------------------------------------------------------
run_pipeline <- function(db, sender_genes, receiver_genes, label) {
  cat(sprintf("\n  [%s]\n", label))
  gc(verbose = FALSE)
  mem_before <- gc(verbose = FALSE)[2, 6]

  inc <- create_Incytr(object = mat, meta = meta,
                       sender = sender, receiver = receiver,
                       group.by = "labels", conditions = conditions)
  t0 <- proc.time()
  inc <- tryCatch({
    inc <- pathway_inference(inc, DB = db,
                             gene.use_Sender = sender_genes,
                             gene.use_Receiver = receiver_genes)
    inc <- Expr_bygroup(inc)
    inc <- Cal_SigProb(inc, K = K, N = N,
                       cutoff_SigProb = CUTOFF, correction = 0.001)
    inc
  }, error = function(e) {
    cat(sprintf("    FAILED: %s\n", e$message))
    NULL
  })
  elapsed <- (proc.time() - t0)["elapsed"]
  mem_after <- gc(verbose = FALSE)[2, 6]

  if (!is.null(inc)) {
    cat(sprintf("    Enumerated: %d pathways\n", nrow(inc@pathways)))
    cat(sprintf("    Surviving:  %d pathways\n", nrow(inc@SigProb)))
    cat(sprintf("    Time: %.1f sec, Memory: ~%.0f MB\n",
                elapsed, mem_after - mem_before))
  }
  inc
}

# ===================================================================
# Test 1: 50% threshold — verify pruned == standard
# ===================================================================
cat("====== Test 1: 50% threshold (correctness check) ======\n")

sg_50 <- names(s_det[s_det >= 0.50])
rg_50 <- names(r_det[r_det >= 0.50])
cat(sprintf("  Sender: %d, Receiver: %d genes\n", length(sg_50), length(rg_50)))

inc_std <- run_pipeline(DB.M, sg_50, rg_50, "Standard")

cat("\n  Pruning edges...\n")
DB_pruned <- prune_db(DB.M, sg_50, rg_50)
inc_pru <- run_pipeline(DB_pruned, sg_50, rg_50, "Pruned")

if (!is.null(inc_std) && !is.null(inc_pru)) {
  paths_match <- identical(sort(inc_std@SigProb$Path), sort(inc_pru@SigProb$Path))
  cat(sprintf("\n  Surviving paths identical: %s\n", ifelse(paths_match, "YES", "NO")))
  if (paths_match && nrow(inc_std@SigProb) > 0) {
    sp_a <- inc_std@SigProb[order(inc_std@SigProb$Path), ]
    sp_b <- inc_pru@SigProb[order(inc_pru@SigProb$Path), ]
    num_cols <- setdiff(names(sp_a), "Path")
    max_diff <- max(sapply(num_cols, function(col)
      max(abs(sp_a[[col]] - sp_b[[col]]), na.rm = TRUE)))
    cat(sprintf("  Max SigProb difference: %.2e\n", max_diff))
  }
}
rm(inc_std, inc_pru, DB_pruned); gc(verbose = FALSE)

# ===================================================================
# Test 2: Lower thresholds — measure reduction
# ===================================================================
for (thr in c(0.20, 0.10, 0.05, 0.0)) {
  cat(sprintf("\n====== Threshold: %.0f%% ======\n", thr * 100))

  if (thr > 0) {
    sg <- names(s_det[s_det >= thr])
    rg <- names(r_det[r_det >= thr])
  } else {
    sg <- all_genes
    rg <- all_genes
  }
  cat(sprintf("  Sender: %d, Receiver: %d genes\n", length(sg), length(rg)))

  cat("\n  Pruning edges...\n")
  DB_pruned <- prune_db(DB.M, sg, rg)

  inc <- run_pipeline(DB_pruned, sg, rg, "Pruned")
  rm(inc, DB_pruned); gc(verbose = FALSE)
}
