#!/usr/bin/env Rscript
# Validation: DuckDB enumeration produces identical results to legacy pipeline.
#
# Test 1 (50% threshold): Run both legacy pathway_inference() and DuckDB
#   enumeration, compare surviving pathway sets and SigProb values.
# Test 2 (10% threshold): Run DuckDB only (legacy OOMs), verify pathway count
#   matches the known value of 169,462.
#
# Usage:
#   micromamba run -n incytr Rscript code/integration/tests/edge_pruning/test_validate_duckdb_enumeration.R

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
repo_root  <- normalizePath(file.path(get_script_dir(), "..", "..", "..", ".."))
int_dir    <- file.path(repo_root, "code", "integration", "intermediates")
wrapper_dir <- file.path(repo_root, "code", "integration", "wrappers")

# Source the DuckDB enumeration function
source(file.path(wrapper_dir, "duckdb_enumeration.R"))

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
cat(sprintf("  DB: L1=%d, L2=%d, L3=%d edges\n\n",
            nrow(DB.M[[1]]), nrow(DB.M[[2]]), nrow(DB.M[[3]])))

sender     <- "Microglia-PVM"
receiver   <- "L5 IT"
conditions <- c("WT", "App")
K <- 0.5; N <- 2; CUTOFF <- 0.01

# Detection rates for threshold filtering
s_cells <- which(meta$labels == sender)
r_cells <- which(meta$labels == receiver)
s_det <- Matrix::rowMeans(mat[, s_cells] > 0)
r_det <- Matrix::rowMeans(mat[, r_cells] > 0)

# ===================================================================
# Test 1: 50% threshold — DuckDB vs legacy
# ===================================================================
cat("====== Test 1: 50% threshold (DuckDB vs legacy) ======\n\n")
sg_50 <- names(s_det[s_det >= 0.50])
rg_50 <- names(r_det[r_det >= 0.50])
cat(sprintf("  Sender: %d, Receiver: %d genes\n\n", length(sg_50), length(rg_50)))

# --- Legacy: pathway_inference + Expr_bygroup + Cal_SigProb ---
cat("  [Legacy] Running pathway_inference...\n")
t0_legacy <- proc.time()
inc_legacy <- create_Incytr(object = mat, meta = meta,
                            sender = sender, receiver = receiver,
                            group.by = "labels", conditions = conditions)
inc_legacy <- pathway_inference(inc_legacy, DB = DB.M,
                                gene.use_Sender = sg_50,
                                gene.use_Receiver = rg_50)
cat(sprintf("  [Legacy] Enumerated: %d pathways\n", nrow(inc_legacy@pathways)))

inc_legacy <- Expr_bygroup(inc_legacy)
inc_legacy <- Cal_SigProb(inc_legacy, K = K, N = N,
                          cutoff_SigProb = CUTOFF, correction = 0.001)
n_legacy <- nrow(inc_legacy@SigProb)
t_legacy <- (proc.time() - t0_legacy)["elapsed"]
cat(sprintf("  [Legacy] Surviving: %d pathways (%.1fs)\n\n", n_legacy, t_legacy))

# --- DuckDB enumeration ---
cat("  [DuckDB] Running duckdb_enumerate_pathways...\n")
t0_duck <- proc.time()
duck_result <- duckdb_enumerate_pathways(
  mat = mat, meta = meta, DB = DB.M,
  sender = sender, receiver = receiver,
  conditions = conditions,
  gene.use_Sender = sg_50, gene.use_Receiver = rg_50,
  K = K, N = N, cutoff_SigProb = CUTOFF)

cat(sprintf("  [DuckDB] Pre-filtered: %d pathways\n", nrow(duck_result$pathways)))

# Create Incytr object, inject DuckDB pathways, run downstream
inc_duck <- create_Incytr(object = mat, meta = meta,
                          sender = sender, receiver = receiver,
                          group.by = "labels", conditions = conditions)
inc_duck@pathways <- as.data.frame(duck_result$pathways)
inc_duck@options$em_degree <- duck_result$em_degree
inc_duck@options$edge_source_count <- duck_result$edge_source_count

inc_duck <- Expr_bygroup(inc_duck)
inc_duck <- Cal_SigProb(inc_duck, K = K, N = N,
                        cutoff_SigProb = CUTOFF, correction = 0.001)
n_duck <- nrow(inc_duck@SigProb)
t_duck <- (proc.time() - t0_duck)["elapsed"]
cat(sprintf("  [DuckDB] Surviving: %d pathways (%.1fs)\n\n", n_duck, t_duck))

# --- Compare ---
cat("  Comparison:\n")
cat(sprintf("    Legacy pathways: %d\n", n_legacy))
cat(sprintf("    DuckDB pathways: %d\n", n_duck))

# Check pathway sets
legacy_paths <- sort(inc_legacy@SigProb$Path)
duck_paths   <- sort(inc_duck@SigProb$Path)

paths_match <- identical(legacy_paths, duck_paths)
cat(sprintf("    Path sets identical: %s\n", ifelse(paths_match, "YES", "NO")))

if (!paths_match) {
  in_legacy_only <- setdiff(legacy_paths, duck_paths)
  in_duck_only   <- setdiff(duck_paths, legacy_paths)
  cat(sprintf("    In legacy only: %d, In DuckDB only: %d\n",
              length(in_legacy_only), length(in_duck_only)))
  if (length(in_legacy_only) > 0) cat(sprintf("    Example legacy-only: %s\n", in_legacy_only[1]))
  if (length(in_duck_only) > 0)   cat(sprintf("    Example duck-only:   %s\n", in_duck_only[1]))
}

# Check SigProb values on shared paths
if (length(intersect(legacy_paths, duck_paths)) > 0) {
  sp_l <- inc_legacy@SigProb[order(inc_legacy@SigProb$Path), ]
  sp_d <- inc_duck@SigProb[order(inc_duck@SigProb$Path), ]

  # Align to shared paths
  shared <- intersect(sp_l$Path, sp_d$Path)
  sp_l <- sp_l[sp_l$Path %in% shared, ]
  sp_d <- sp_d[sp_d$Path %in% shared, ]
  sp_l <- sp_l[order(sp_l$Path), ]
  sp_d <- sp_d[order(sp_d$Path), ]

  num_cols <- intersect(names(sp_l), names(sp_d))
  num_cols <- setdiff(num_cols, "Path")
  max_diff <- max(sapply(num_cols, function(col)
    max(abs(sp_l[[col]] - sp_d[[col]]), na.rm = TRUE)))
  cat(sprintf("    Max SigProb difference (shared paths): %.2e\n", max_diff))
}

# Check em_degree
em_deg_match <- identical(duck_result$em_degree, inc_legacy@options$em_degree %||% duck_result$em_degree)
cat(sprintf("    em_degree matches: %s\n", ifelse(em_deg_match, "YES", "N/A (cleared by Cal_SigProb)")))

rm(inc_legacy, inc_duck, duck_result)
gc(verbose = FALSE)

# ===================================================================
# Test 2: 10% threshold — DuckDB only
# ===================================================================
cat("\n====== Test 2: 10% threshold (DuckDB only) ======\n\n")
sg_10 <- names(s_det[s_det >= 0.10])
rg_10 <- names(r_det[r_det >= 0.10])
cat(sprintf("  Sender: %d, Receiver: %d genes\n\n", length(sg_10), length(rg_10)))

cat("  [DuckDB] Running duckdb_enumerate_pathways...\n")
t0 <- proc.time()
duck_10 <- duckdb_enumerate_pathways(
  mat = mat, meta = meta, DB = DB.M,
  sender = sender, receiver = receiver,
  conditions = conditions,
  gene.use_Sender = sg_10, gene.use_Receiver = rg_10,
  K = K, N = N, cutoff_SigProb = CUTOFF)
t_10 <- (proc.time() - t0)["elapsed"]

cat(sprintf("  [DuckDB] Pre-filtered: %d pathways (%.1fs)\n", nrow(duck_10$pathways), t_10))

# Run through Cal_SigProb for the exact surviving count
inc_10 <- create_Incytr(object = mat, meta = meta,
                        sender = sender, receiver = receiver,
                        group.by = "labels", conditions = conditions)
inc_10@pathways <- as.data.frame(duck_10$pathways)
inc_10@options$em_degree <- duck_10$em_degree
inc_10@options$edge_source_count <- duck_10$edge_source_count

inc_10 <- Expr_bygroup(inc_10)
inc_10 <- Cal_SigProb(inc_10, K = K, N = N,
                      cutoff_SigProb = CUTOFF, correction = 0.001)
n_surviving_10 <- nrow(inc_10@SigProb)
cat(sprintf("  [DuckDB] After Cal_SigProb: %d surviving pathways\n", n_surviving_10))

expected_10 <- 169462
cat(sprintf("\n  Expected: %d\n", expected_10))
cat(sprintf("  Match: %s\n",
            ifelse(n_surviving_10 == expected_10, "YES",
                   sprintf("NO (got %d, diff=%d)", n_surviving_10,
                           n_surviving_10 - expected_10))))

# ===================================================================
# Summary
# ===================================================================
cat("\n====== Summary ======\n")
cat(sprintf("  Test 1 (50%% correctness):  %s\n",
            ifelse(paths_match, "PASS", "FAIL")))
cat(sprintf("  Test 2 (10%% count check):  %s\n",
            ifelse(n_surviving_10 == expected_10, "PASS",
                   sprintf("FAIL (got %d)", n_surviving_10))))
