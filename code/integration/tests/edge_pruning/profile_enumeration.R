#!/usr/bin/env Rscript
# Baseline measurements: pathway enumeration at different thresholds.
#
# Measures pathway count, time, and memory for the standard pipeline
# at 50%, 20%, and 10% thresholds. Stops before OOM.
#
# Usage (from repo root):
#   micromamba run -n incytr Rscript code/integration/tests/edge_pruning/profile_enumeration.R

suppressPackageStartupMessages({
  library(Matrix)
  library(data.table)
  library(Incytr)
})

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
# Load data (shared across all runs)
# ---------------------------------------------------------------------------
cat("Loading expression matrix...\n")
mat <- readMM(file.path(int_dir, "expression_matrix.mtx"))
mat <- as(mat, "CsparseMatrix")
genes    <- read.csv(file.path(int_dir, "expression_genes.csv"))$gene
barcodes <- read.csv(file.path(int_dir, "expression_barcodes.csv"))$barcode
meta     <- read.csv(file.path(int_dir, "expression_metadata.csv"), row.names = 1)
rownames(mat) <- genes; colnames(mat) <- barcodes
cat(sprintf("  %d genes x %d cells\n", nrow(mat), ncol(mat)))

cat("Loading IncytrDB...\n")
data(DB_Layer1_mouse_filtered, package = "Incytr")
data(DB_Layer2_mouse_filtered, package = "Incytr")
data(DB_Layer3_mouse_filtered, package = "Incytr")

# Pre-filter to genes in expression matrix
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
cat(sprintf("  Filtered DB: L1=%d, L2=%d, L3=%d edges\n",
            nrow(DB.M[[1]]), nrow(DB.M[[2]]), nrow(DB.M[[3]])))

sender   <- "Microglia-PVM"
receiver <- "L5 IT"
conditions <- c("WT", "App")

# Detection rates
s_cells <- which(meta$labels == sender)
r_cells <- which(meta$labels == receiver)
s_det   <- Matrix::rowMeans(mat[, s_cells] > 0)
r_det   <- Matrix::rowMeans(mat[, r_cells] > 0)

# ---------------------------------------------------------------------------
# Profile at each threshold
# ---------------------------------------------------------------------------
thresholds <- c(0.50, 0.20, 0.10)
results <- list()

for (thr in thresholds) {
  cat(sprintf("\n====== Threshold: %.0f%% ======\n", thr * 100))

  sg <- names(s_det[s_det >= thr])
  rg <- names(r_det[r_det >= thr])
  cat(sprintf("  Sender genes: %d, Receiver genes: %d\n", length(sg), length(rg)))

  # Count edges after gene filtering (without running full pipeline)
  dt1 <- as.data.table(DB.M[[1]])
  dt1 <- dt1[from %in% sg & to %in% rg]
  dt2 <- as.data.table(DB.M[[2]])
  dt2 <- dt2[from %in% rg & to %in% rg]
  dt3 <- as.data.table(DB.M[[3]])
  dt3 <- dt3[from %in% rg & to %in% rg]
  cat(sprintf("  Filtered edges: L1=%d, L2=%d, L3=%d\n",
              nrow(dt1), nrow(dt2), nrow(dt3)))

  # Run pathway_inference + SigProb
  gc(verbose = FALSE)
  mem_before <- gc(verbose = FALSE)[2, 6]  # max Vcells used (MB)

  inc <- create_Incytr(object = mat, meta = meta,
                       sender = sender, receiver = receiver,
                       group.by = "labels", conditions = conditions)

  t0 <- proc.time()
  inc <- tryCatch({
    inc <- pathway_inference(inc, DB = DB.M,
                             gene.use_Sender = sg,
                             gene.use_Receiver = rg)
    cat(sprintf("  Pathways enumerated: %d\n", nrow(inc@pathways)))

    inc <- Expr_bygroup(inc)
    inc <- Cal_SigProb(inc, K = 0.5, N = 2,
                       cutoff_SigProb = 0.01, correction = 0.001)
    cat(sprintf("  Surviving after SigProb: %d\n", nrow(inc@SigProb)))
    inc
  }, error = function(e) {
    cat(sprintf("  FAILED: %s\n", e$message))
    NULL
  })
  elapsed <- (proc.time() - t0)["elapsed"]
  mem_after <- gc(verbose = FALSE)[2, 6]

  if (!is.null(inc)) {
    n_enum <- nrow(inc@pathways)
    n_surv <- nrow(inc@SigProb)
    cat(sprintf("  Time: %.1f sec\n", elapsed))
    cat(sprintf("  Memory delta: ~%.0f MB\n", mem_after - mem_before))
    cat(sprintf("  Kill rate: %.1f%% of enumerated pathways removed by SigProb\n",
                100 * (1 - n_surv / n_enum)))

    results[[length(results) + 1]] <- data.frame(
      threshold = thr, sender_genes = length(sg), receiver_genes = length(rg),
      edges_l1 = nrow(dt1), edges_l2 = nrow(dt2), edges_l3 = nrow(dt3),
      pathways_enumerated = n_enum, pathways_surviving = n_surv,
      time_sec = elapsed, mem_delta_mb = mem_after - mem_before
    )
  }

  rm(inc, dt1, dt2, dt3); gc(verbose = FALSE)
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
cat("\n\n====== Summary ======\n")
results_df <- do.call(rbind, results)
print(results_df)

out_path <- file.path(get_script_dir(), "profile_results.csv")
write.csv(results_df, out_path, row.names = FALSE)
cat(sprintf("\nSaved: %s\n", out_path))
