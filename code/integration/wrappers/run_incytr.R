#!/usr/bin/env Rscript
# Main Incytr pipeline wrapper for Phase 1 proof of concept.
#
# Runs two passes:
#   1. Expression-only (baseline for sensitivity analysis)
#   2. Full integration (phospho + kinase evidence)
#
# Outputs:
#   intermediates/results_expronly.csv
#   intermediates/results_full.csv
#   intermediates/pvalues_seed{1..5}.csv (permutation stability)
#   intermediates/incytr_object.rds

suppressPackageStartupMessages({
  library(Matrix)
  library(data.table)
})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Determine repo root from this script's location
get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg[1]))))
  }
  return(file.path(getwd(), "code", "integration", "wrappers"))
}
script_dir <- get_script_dir()
repo_root <- normalizePath(file.path(script_dir, "..", "..", ".."))
int_dir <- file.path(repo_root, "code", "integration", "intermediates")
incytr_root <- normalizePath(file.path(repo_root, "..", "incytr"))

cat("Repo root:", repo_root, "\n")
cat("Incytr root:", incytr_root, "\n")
cat("Intermediates:", int_dir, "\n\n")

# ---------------------------------------------------------------------------
# Load Incytr
# ---------------------------------------------------------------------------
cat("Loading Incytr...\n")
library(Incytr)
cat("Incytr loaded.\n\n")

# ---------------------------------------------------------------------------
# 1. Load expression data
# ---------------------------------------------------------------------------
cat("Loading expression matrix...\n")
mat <- readMM(file.path(int_dir, "expression_matrix.mtx"))
# readMM returns dgTMatrix; Incytr requires dgCMatrix
mat <- as(mat, "dgCMatrix")
genes <- read.csv(file.path(int_dir, "expression_genes.csv"))$gene
barcodes <- read.csv(file.path(int_dir, "expression_barcodes.csv"))$barcode
rownames(mat) <- genes
colnames(mat) <- barcodes
cat(sprintf("  %d genes x %d cells\n", nrow(mat), ncol(mat)))

meta <- read.csv(file.path(int_dir, "expression_metadata.csv"),
                 row.names = 1, check.names = FALSE)
cat(sprintf("  Metadata: %d cells, conditions: %s\n",
            nrow(meta), paste(unique(meta$condition), collapse = ", ")))
cat(sprintf("  Cell types: %d unique\n", length(unique(meta$labels))))

# ---------------------------------------------------------------------------
# 2. Load IncytrDB (mouse)
# ---------------------------------------------------------------------------
cat("\nLoading IncytrDB mouse...\n")
data(DB_Layer1_mouse_filtered, package = "Incytr")
data(DB_Layer2_mouse_filtered, package = "Incytr")
data(DB_Layer3_mouse_filtered, package = "Incytr")
cat(sprintf("  Raw DB: L1=%d, L2=%d, L3=%d edges\n",
            nrow(DB_Layer1_mouse_filtered),
            nrow(DB_Layer2_mouse_filtered),
            nrow(DB_Layer3_mouse_filtered)))

# Pre-filter DB to genes in the expression matrix to reduce memory
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

# ---------------------------------------------------------------------------
# 3. Create Incytr object
# ---------------------------------------------------------------------------
sender <- "Microglia-PVM"
receiver <- "L5 IT"
conditions <- c("WT", "App")

cat(sprintf("\nCreating Incytr object: %s -> %s\n", sender, receiver))
inc <- create_Incytr(
  object = mat,
  meta = meta,
  sender = sender,
  receiver = receiver,
  group.by = "labels",
  conditions = conditions
)

# ---------------------------------------------------------------------------
# 4. Filter genes to reduce memory (use expressed genes only)
# ---------------------------------------------------------------------------
cat("\nFiltering to expressed genes...\n")
# Find genes expressed in at least 5% of cells in sender or receiver
sender_cells <- which(meta$labels == sender)
receiver_cells <- which(meta$labels == receiver)
sender_expr <- Matrix::rowMeans(mat[, sender_cells] > 0)
receiver_expr <- Matrix::rowMeans(mat[, receiver_cells] > 0)
# Use 50% expression threshold: gene must be detected in >=50% of cells
# in the cell type. This produces a manageable pathway count (~10-50K).
# Lower thresholds (5%: 9.5M pathways, 20%: 788K) cause OOM downstream.
sender_genes <- names(sender_expr[sender_expr >= 0.50])
receiver_genes <- names(receiver_expr[receiver_expr >= 0.50])
cat(sprintf("  Sender genes (>5%% cells): %d\n", length(sender_genes)))
cat(sprintf("  Receiver genes (>5%% cells): %d\n", length(receiver_genes)))

# ---------------------------------------------------------------------------
# 5. Pathway inference and expression scoring
# ---------------------------------------------------------------------------
cat("\nRunning pathway inference...\n")
inc <- pathway_inference(inc, DB = DB.M,
                         gene.use_Sender = sender_genes,
                         gene.use_Receiver = receiver_genes)
n_pathways <- nrow(inc@pathways)
cat(sprintf("  %d pathways inferred\n", n_pathways))

if (n_pathways == 0) {
  cat("ERROR: No pathways found. Check sender/receiver gene expression.\n")
  quit(status = 1)
}

cat("Computing expression by group...\n")
inc <- Expr_bygroup(inc)

# Use cutoff_SigProb to filter pathways with negligible signaling probability.
# Without this, 788K pathways survive and OOM downstream steps.
cat("Computing signaling probability (cutoff=0.01)...\n")
inc <- Cal_SigProb(inc, K = 0.5, N = 2, cutoff_SigProb = 0.01,
                   correction = 0.001)
cat(sprintf("  Pathways after SigProb cutoff: %d\n", nrow(inc@SigProb)))

cat("Computing single-cell fold changes...\n")
inc <- Cal_scFC(inc)

# ---------------------------------------------------------------------------
# 5. Expression-only evaluation (baseline)
# ---------------------------------------------------------------------------
cat("\n=== Expression-only evaluation (baseline) ===\n")
inc_base <- Pathway_evaluation(inc, score.weight = rep(0, 6))
results_expronly <- Export_results(inc_base)
expronly_path <- file.path(int_dir, "results_expronly.csv")
write.csv(results_expronly, expronly_path, row.names = FALSE)
cat(sprintf("  Wrote %s (%d pathways)\n", expronly_path, nrow(results_expronly)))
rm(inc_base, results_expronly)
gc(verbose = FALSE)

# ---------------------------------------------------------------------------
# 6. Phospho integration
# ---------------------------------------------------------------------------
cat("\n=== Integrating phosphoproteomics ===\n")
ps1_path <- file.path(int_dir, "ps_condition1.csv")
ps2_path <- file.path(int_dir, "ps_condition2.csv")

if (file.exists(ps1_path) && file.exists(ps2_path)) {
  ps1 <- read.csv(ps1_path, check.names = FALSE)
  ps2 <- read.csv(ps2_path, check.names = FALSE)
  cat(sprintf("  Phospho data: %d genes (condition 1), %d genes (condition 2)\n",
              nrow(ps1), nrow(ps2)))

  # Check that sender and receiver columns have non-NA data
  sender_col <- paste0(sender, "_ps")
  receiver_col <- paste0(receiver, "_ps")
  ps_cols <- setdiff(colnames(ps1), "gene_symbol")
  n_nonna <- sum(!is.na(as.matrix(ps1[, ps_cols])))
  cat(sprintf("  Non-NA phospho entries: %d\n", n_nonna))

  sender_has_data <- sender_col %in% ps_cols &&
    sum(!is.na(ps1[[sender_col]])) > 0
  receiver_has_data <- receiver_col %in% ps_cols &&
    sum(!is.na(ps1[[receiver_col]])) > 0
  cat(sprintf("  %s: %s, %s: %s\n",
              sender, ifelse(sender_has_data, "has data", "empty"),
              receiver, ifelse(receiver_has_data, "has data", "empty")))

  if (sender_has_data || receiver_has_data) {
    # Keep only columns with data, but always keep sender/receiver columns
    # (Integr_multiomics requires them even if NA)
    keep_cols <- c("gene_symbol")
    required_cols <- c(sender_col, receiver_col)
    for (col in ps_cols) {
      has_data <- sum(!is.na(ps1[[col]])) > 0 || sum(!is.na(ps2[[col]])) > 0
      is_required <- col %in% required_cols
      if (has_data || is_required) {
        keep_cols <- c(keep_cols, col)
      }
    }
    ps1 <- ps1[, keep_cols, drop = FALSE]
    ps2 <- ps2[, keep_cols, drop = FALSE]
    cat(sprintf("  Kept %d cell-type columns (incl. sender/receiver)\n",
                length(keep_cols) - 1))

    inc <- tryCatch(
      Integr_multiomics(inc,
                        ps.data_condition1 = ps1,
                        ps.data_condition2 = ps2),
      error = function(e) {
        cat(sprintf("  WARNING: Integr_multiomics failed: %s\n", e$message))
        cat("  Proceeding without phospho integration.\n")
        inc
      }
    )
    cat("  Phospho integration step complete.\n")
  } else {
    cat("  No phospho data for sender or receiver, skipping.\n")
  }
} else {
  cat("  WARNING: Phospho files not found, skipping.\n")
}

# ---------------------------------------------------------------------------
# 7. Full evaluation (with phospho)
# ---------------------------------------------------------------------------
cat("\nRunning pathway evaluation (with phospho)...\n")
inc <- Pathway_evaluation(inc)

# ---------------------------------------------------------------------------
# 8. Kinase integration
# ---------------------------------------------------------------------------
cat("\n=== Integrating kinase evidence ===\n")
kldata_path <- file.path(int_dir, "kldata.csv")
kl_output_path <- file.path(int_dir, "kl_output.csv")

if (file.exists(kldata_path)) {
  kldata <- read.csv(kldata_path, check.names = FALSE)
  cat(sprintf("  kldata (raw): %d rows, %d kinases\n",
              nrow(kldata), length(unique(kldata[["motif.geneName"]]))))

  # Pre-filter kldata to genes that appear in pathway nodes (R, EM, T)
  # to avoid OOM during Integr_kinasedata/enrichment cross-joins
  pathway_genes <- unique(c(inc@pathways$Receptor, inc@pathways$EM,
                            inc@pathways$Target))
  kldata <- kldata[kldata$gene %in% pathway_genes |
                   kldata[["motif.geneName"]] %in% pathway_genes, ]
  cat(sprintf("  kldata (filtered to pathway genes): %d rows, %d kinases\n",
              nrow(kldata), length(unique(kldata[["motif.geneName"]]))))

  cell_groups <- unique(meta$labels)
  inc <- Integr_kinasedata(inc, kldata = kldata,
                           cell_group = cell_groups)
  cat("  Structural kinase integration complete.\n")
} else {
  cat("  WARNING: kldata.csv not found, skipping structural kinase.\n")
}

if (file.exists(kl_output_path)) {
  kl_out <- read.csv(kl_output_path, check.names = FALSE)
  # Pre-filter kl_output to substrates in pathway genes
  kl_out <- kl_out[kl_out$substrate %in% pathway_genes |
                   kl_out$kinase %in% pathway_genes, ]
  cat(sprintf("  kl_output (filtered): %d rows, %d kinases\n",
              nrow(kl_out), length(unique(kl_out$kinase))))

  if (file.exists(kldata_path) && nrow(kl_out) > 0) {
    inc <- Integr_kinase_enrichment(inc, kl_output = kl_out,
                                    kldata = kldata,
                                    cell_group = cell_groups)
    cat("  Activity kinase integration complete.\n")
  }
} else {
  cat("  WARNING: kl_output.csv not found, skipping activity kinase.\n")
}

# ---------------------------------------------------------------------------
# 9. PDS
# ---------------------------------------------------------------------------
cat("\nComputing PDS...\n")
inc <- Cal_PDS(inc, KPDS.weight = 0.5, AKPDS.weight = 0.25)

# ---------------------------------------------------------------------------
# 10. Permutation test with stability check (5 seeds)
# ---------------------------------------------------------------------------
cat("\n=== Permutation testing (3 seeds x 50 permutations) ===\n")
cat("  Note: L5 IT is Tier 3 (<30 cells). P-values are for reference only.\n")

for (seed in 1:3) {
  cat(sprintf("  Seed %d...", seed))
  inc_perm <- tryCatch(
    # Use single core to avoid forking (each fork duplicates memory)
    Permutation_test(inc, nboot = 50, seed.use = as.integer(seed),
                     n.cores = 1L),
    error = function(e) {
      cat(sprintf(" ERROR: %s\n", e$message))
      NULL
    }
  )
  if (!is.null(inc_perm)) {
    pval_path <- file.path(int_dir, sprintf("pvalues_seed%d.csv", seed))
    write.csv(inc_perm@p_value, pval_path, row.names = FALSE)
    cat(sprintf(" done -> %s\n", basename(pval_path)))
    # Keep the last permutation result
    if (seed == 3) inc <- inc_perm
  }
}

# ---------------------------------------------------------------------------
# 11. Export full results
# ---------------------------------------------------------------------------
cat("\n=== Exporting results ===\n")
results_full <- Export_results(inc, indicator = TRUE)
full_path <- file.path(int_dir, "results_full.csv")
write.csv(results_full, full_path, row.names = FALSE)
cat(sprintf("  Wrote %s (%d pathways)\n", full_path, nrow(results_full)))

# Save R object for postprocessing
rds_path <- file.path(int_dir, "incytr_object.rds")
saveRDS(inc, rds_path)
cat(sprintf("  Wrote %s\n", rds_path))

cat("\nR wrapper complete.\n")
