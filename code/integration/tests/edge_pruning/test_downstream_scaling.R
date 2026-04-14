#!/usr/bin/env Rscript
# Test: measure time and memory for each downstream pipeline step at 10% threshold.
#
# Runs DuckDB enumeration (already validated), then times each step individually:
#   1. Expr_bygroup
#   2. Cal_SigProb
#   3. Cal_scFC
#   4. Pathway_evaluation (expression-only)
#   5. Integr_kinasedata (if kldata exists)
#   6. Pathway_evaluation (full)
#   7. Cal_PDS
#   8. Permutation_test (1 seed × 10 iterations — just to measure cost)
#
# Reports per-step time and memory. Does NOT crash — stops early if memory
# exceeds a threshold.
#
# Usage:
#   micromamba run -n incytr Rscript code/integration/tests/edge_pruning/test_downstream_scaling.R

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

MEM_LIMIT_MB <- 12000  # abort if R memory exceeds this

check_mem <- function(step_name) {
  mem_mb <- gc(verbose = FALSE)[2, 2]
  cat(sprintf("    Memory: %.0f MB\n", mem_mb))
  if (mem_mb > MEM_LIMIT_MB) {
    cat(sprintf("  ABORT: Memory (%.0f MB) exceeds limit (%d MB) after %s\n",
                mem_mb, MEM_LIMIT_MB, step_name))
    quit(status = 1)
  }
  mem_mb
}

time_step <- function(name, expr) {
  cat(sprintf("\n  [%s]\n", name))
  gc(verbose = FALSE)
  mem_before <- gc(verbose = FALSE)[2, 2]
  t0 <- proc.time()
  result <- tryCatch(expr, error = function(e) {
    cat(sprintf("    FAILED: %s\n", e$message))
    NULL
  })
  elapsed <- (proc.time() - t0)["elapsed"]
  mem_after <- check_mem(name)
  cat(sprintf("    Time: %.1f sec, Memory delta: %.0f MB\n",
              elapsed, mem_after - mem_before))
  result
}

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

sender     <- "Microglia-PVM"
receiver   <- "L5 IT"
conditions <- c("WT", "App")

# 10% threshold gene lists
s_det <- Matrix::rowMeans(mat[, meta$labels == sender] > 0)
r_det <- Matrix::rowMeans(mat[, meta$labels == receiver] > 0)
sg <- names(s_det[s_det >= 0.10])
rg <- names(r_det[r_det >= 0.10])
cat(sprintf("  10%% threshold: sender=%d, receiver=%d genes\n\n", length(sg), length(rg)))

# ---------------------------------------------------------------------------
# DuckDB enumeration
# ---------------------------------------------------------------------------
cat("====== DuckDB Enumeration ======\n")
duck <- time_step("DuckDB enumerate", {
  duckdb_enumerate_pathways(
    mat = mat, meta = meta, DB = DB.M,
    sender = sender, receiver = receiver,
    conditions = conditions,
    gene.use_Sender = sg, gene.use_Receiver = rg,
    K = 0.5, N = 2, cutoff_SigProb = 0.01)
})
if (is.null(duck)) quit(status = 1)
cat(sprintf("    Pre-filtered pathways: %d\n", nrow(duck$pathways)))

# ---------------------------------------------------------------------------
# Create Incytr object and inject pathways
# ---------------------------------------------------------------------------
cat("\n====== Downstream Pipeline Steps ======\n")
inc <- create_Incytr(object = mat, meta = meta,
                     sender = sender, receiver = receiver,
                     group.by = "labels", conditions = conditions)
inc@pathways <- as.data.frame(duck$pathways)
inc@options$em_degree <- duck$em_degree
inc@options$edge_source_count <- duck$edge_source_count
rm(duck)
gc(verbose = FALSE)

# Step 1: Expr_bygroup
inc <- time_step("Expr_bygroup", Expr_bygroup(inc))
if (is.null(inc)) quit(status = 1)

# Step 2: Cal_SigProb
inc <- time_step("Cal_SigProb", {
  Cal_SigProb(inc, K = 0.5, N = 2, cutoff_SigProb = 0.01, correction = 0.001)
})
if (is.null(inc)) quit(status = 1)
cat(sprintf("    Surviving pathways: %d\n", nrow(inc@SigProb)))

# Step 3: Cal_scFC
inc <- time_step("Cal_scFC", Cal_scFC(inc))
if (is.null(inc)) quit(status = 1)

# Step 4: Pathway_evaluation (expression-only baseline)
inc_base <- time_step("Pathway_evaluation (expr-only)", {
  Pathway_evaluation(inc, score.weight = rep(0, 6))
})
if (!is.null(inc_base)) {
  cat(sprintf("    Result rows: %d\n", nrow(Export_results(inc_base))))
  rm(inc_base)
  gc(verbose = FALSE)
}

# Step 5: Phospho integration (if files exist)
ps1_path <- file.path(int_dir, "ps_condition1.csv")
ps2_path <- file.path(int_dir, "ps_condition2.csv")
if (file.exists(ps1_path) && file.exists(ps2_path)) {
  inc <- time_step("Integr_multiomics (phospho)", {
    ps1 <- read.csv(ps1_path, check.names = FALSE)
    ps2 <- read.csv(ps2_path, check.names = FALSE)
    Integr_multiomics(inc, ps.data_condition1 = ps1, ps.data_condition2 = ps2)
  })
} else {
  cat("\n  [Integr_multiomics] SKIPPED — no phospho files\n")
}

# Step 6: Pathway_evaluation (full)
inc <- time_step("Pathway_evaluation (full)", Pathway_evaluation(inc))

# Step 7: Kinase integration
kldata_path <- file.path(int_dir, "kldata.csv")
kl_output_path <- file.path(int_dir, "kl_output.csv")
if (file.exists(kldata_path)) {
  inc <- time_step("Integr_kinasedata", {
    kldata <- read.csv(kldata_path, check.names = FALSE)
    pathway_genes <- unique(c(inc@pathways$Receptor, inc@pathways$EM, inc@pathways$Target))
    kldata <- kldata[kldata$gene %in% pathway_genes |
                     kldata[["motif.geneName"]] %in% pathway_genes, ]
    cell_groups <- unique(meta$labels)
    Integr_kinasedata(inc, kldata = kldata, cell_group = cell_groups)
  })
} else {
  cat("\n  [Integr_kinasedata] SKIPPED — no kldata.csv\n")
}

if (file.exists(kl_output_path) && file.exists(kldata_path)) {
  inc <- time_step("Integr_kinase_enrichment", {
    kl_out <- read.csv(kl_output_path, check.names = FALSE)
    pathway_genes <- unique(c(inc@pathways$Receptor, inc@pathways$EM, inc@pathways$Target))
    kl_out <- kl_out[kl_out$substrate %in% pathway_genes |
                     kl_out$kinase %in% pathway_genes, ]
    Integr_kinase_enrichment(inc, kl_output = kl_out, kldata = kldata,
                             cell_group = cell_groups)
  })
} else {
  cat("\n  [Integr_kinase_enrichment] SKIPPED — no kl_output.csv\n")
}

# Step 8: Cal_PDS
inc <- time_step("Cal_PDS", Cal_PDS(inc, KPDS.weight = 0.5, AKPDS.weight = 0.25))

# Step 9: Permutation test (minimal — 1 seed × 10 iterations to estimate cost)
cat("\n  [Permutation_test] Running 1 seed x 10 iterations (cost estimate)...\n")
perm_result <- time_step("Permutation_test (1x10)", {
  Permutation_test(inc, nboot = 10, seed.use = 1L, n.cores = 1L)
})

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
cat("\n====== Summary ======\n")
cat(sprintf("  Pre-filtered pathways (DuckDB): %d\n", nrow(inc@pathways)))
cat(sprintf("  Surviving pathways (Cal_SigProb): %d\n", nrow(inc@SigProb)))
mem_final <- gc(verbose = FALSE)[2, 2]
cat(sprintf("  Final R memory: %.0f MB\n", mem_final))
cat("\n  If permutation took T seconds for 10 iterations,\n")
cat("  the full 3 seeds x 50 iterations will take ~15T seconds.\n")
