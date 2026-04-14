#!/usr/bin/env Rscript
# Test: precompute pruned L2⋈L3 chains per receiver, then join L1 per sender.
#
# Structure:
#   1. Compute per-condition mean expression for all 22 cell types (once)
#   2+3. Per receiver: prune L2/L3 -> chains, then iterate all senders (L1 join)
#        Chains are discarded after each receiver to bound memory.
#
# Validates against the single-pair result from test_edge_pruning.R
# (Microglia-PVM -> L5 IT should produce 169,462 pathways).
#
# Usage (from repo root):
#   micromamba run -n incytr Rscript code/integration/tests/edge_pruning/test_receiver_precompute.R

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
l1_raw <- as.data.table(DB_Layer1_mouse_filtered[
  DB_Layer1_mouse_filtered$from %in% all_genes &
  DB_Layer1_mouse_filtered$to %in% all_genes, ])
l2_raw <- as.data.table(DB_Layer2_mouse_filtered[
  DB_Layer2_mouse_filtered$from %in% all_genes &
  DB_Layer2_mouse_filtered$to %in% all_genes, ])
l3_raw <- as.data.table(DB_Layer3_mouse_filtered[
  DB_Layer3_mouse_filtered$from %in% all_genes &
  DB_Layer3_mouse_filtered$to %in% all_genes, ])
rm(DB_Layer1_mouse_filtered, DB_Layer2_mouse_filtered, DB_Layer3_mouse_filtered)
gc(verbose = FALSE)
cat(sprintf("  DB: L1=%d, L2=%d, L3=%d edges\n\n",
            nrow(l1_raw), nrow(l2_raw), nrow(l3_raw)))

conditions <- c("WT", "App")
cell_types <- sort(unique(meta$labels))
K <- 0.5; N <- 2; CUTOFF <- 0.01

# ===================================================================
# Step 1: Compute per-condition mean expression for all cell types
# ===================================================================
cat("Step 1: Computing mean expression per cell type per condition...\n")
t0 <- proc.time()

# Named list: mean_expr[[cell_type]][[condition]] = named numeric vector
mean_expr <- list()
for (ct in cell_types) {
  mean_expr[[ct]] <- list()
  for (cond in conditions) {
    cells <- which(meta$labels == ct & meta$condition == cond)
    if (length(cells) > 0) {
      mean_expr[[ct]][[cond]] <- setNames(Matrix::rowMeans(mat[, cells]), all_genes)
    } else {
      mean_expr[[ct]][[cond]] <- setNames(rep(0, length(all_genes)), all_genes)
    }
  }
}
t_step1 <- (proc.time() - t0)["elapsed"]
cat(sprintf("  %d cell types x %d conditions = %d vectors (%.1f sec)\n\n",
            length(cell_types), length(conditions),
            length(cell_types) * length(conditions), t_step1))

# ===================================================================
# Steps 2+3: Per receiver — compute chains, then join all senders
# ===================================================================
# Memory optimization: process one receiver at a time instead of
# holding all 22 receiver chain tables simultaneously.
# L2⋈L3 is still computed once per receiver (22×), not per pair (462×).
cat("Steps 2+3: Per-receiver chains + sender enumeration...\n")
t0_step23 <- proc.time()
t_chain_total <- 0  # accumulate chain computation time

pair_results <- list()
n_pairs <- 0

for (recv in cell_types) {
  # --- Step 2 for this receiver: prune L2/L3, build chains ---
  t0_recv <- proc.time()
  rc1 <- mean_expr[[recv]][["WT"]]
  rc2 <- mean_expr[[recv]][["App"]]

  # Prune L2: both from and to are receiver genes
  dt2 <- copy(l2_raw)
  h1 <- hill(rc1[dt2$from] * rc1[dt2$to], K, N)
  h2 <- hill(rc2[dt2$from] * rc2[dt2$to], K, N)
  dt2 <- dt2[(h1 >= CUTOFF) | (h2 >= CUTOFF)]
  setnames(dt2, c("from", "to"), c("Receptor", "EM"), skip_absent = TRUE)

  # Prune L3: both from and to are receiver genes
  dt3 <- copy(l3_raw)
  h1 <- hill(rc1[dt3$from] * rc1[dt3$to], K, N)
  h2 <- hill(rc2[dt3$from] * rc2[dt3$to], K, N)
  dt3 <- dt3[(h1 >= CUTOFF) | (h2 >= CUTOFF)]
  setnames(dt3, c("from", "to"), c("EM", "Target"), skip_absent = TRUE)

  # Join L2 ⋈ L3 on EM
  setkey(dt2, EM); setkey(dt3, EM)
  chains <- dt3[dt2, allow.cartesian = TRUE, nomatch = 0]

  # Remove self-edges (R != EM, R != T, EM != T)
  chains <- chains[Receptor != EM & Receptor != Target & EM != Target]
  # Deduplicate
  chains <- unique(chains[, .(Receptor, EM, Target)])

  t_chain <- (proc.time() - t0_recv)["elapsed"]
  t_chain_total <- t_chain_total + t_chain

  cat(sprintf("  %-18s  L2: %6d -> %6d  L3: %6d -> %6d  chains: %7d  (%.1fs)",
              recv,
              nrow(l2_raw), nrow(dt2),
              nrow(l3_raw), nrow(dt3),
              nrow(chains), t_chain))
  rm(dt2, dt3, h1, h2)

  # --- Step 3 for this receiver: join each sender's L1 ---
  setkey(chains, Receptor)
  n_recv_pairs <- 0
  t0_senders <- proc.time()

  for (send in cell_types) {
    if (send == recv) next
    n_pairs <- n_pairs + 1
    n_recv_pairs <- n_recv_pairs + 1

    sc1 <- mean_expr[[send]][["WT"]]
    sc2 <- mean_expr[[send]][["App"]]

    # Prune L1: from=sender, to=receiver
    dt1 <- copy(l1_raw)
    h1 <- hill(sc1[dt1$from] * rc1[dt1$to], K, N)
    h2 <- hill(sc2[dt1$from] * rc2[dt1$to], K, N)
    dt1 <- dt1[(h1 >= CUTOFF) | (h2 >= CUTOFF)]

    if (nrow(dt1) == 0) {
      pair_results[[n_pairs]] <- data.frame(
        sender = send, receiver = recv,
        l1_edges = 0L, n_pathways = 0L, time_ms = 0,
        stringsAsFactors = FALSE)
      next
    }

    setnames(dt1, c("from", "to"), c("Ligand", "Receptor"), skip_absent = TRUE)
    dt1 <- dt1[, .(Ligand, Receptor)]

    # Join L1 ⋈ precomputed receiver chains
    setkey(dt1, Receptor)
    t0_join <- proc.time()
    pathways <- chains[dt1, allow.cartesian = TRUE, nomatch = 0]

    # Remove Ligand self-edges
    pathways <- pathways[Ligand != Receptor & Ligand != EM & Ligand != Target]
    pathways <- unique(pathways, by = c("Ligand", "Receptor", "EM", "Target"))
    t_join <- (proc.time() - t0_join)["elapsed"] * 1000

    pair_results[[n_pairs]] <- data.frame(
      sender = send, receiver = recv,
      l1_edges = nrow(dt1), n_pathways = nrow(pathways), time_ms = t_join,
      stringsAsFactors = FALSE)
  }
  t_senders <- (proc.time() - t0_senders)["elapsed"]
  cat(sprintf("  -> %d senders in %.1fs\n", n_recv_pairs, t_senders))

  # Free this receiver's chains before processing the next
  rm(chains)
  gc(verbose = FALSE)
}
t_step23 <- (proc.time() - t0_step23)["elapsed"]

results_df <- do.call(rbind, pair_results)
cat(sprintf("  %d pairs enumerated in %.1f sec (%.0f ms/pair avg)\n\n",
            n_pairs, t_step23, 1000 * t_step23 / n_pairs))

# ===================================================================
# Summary
# ===================================================================
cat("====== Summary ======\n")
cat(sprintf("  Step 1 (mean expression):    %.1f sec (once)\n", t_step1))
cat(sprintf("  Steps 2+3 (chains+pairs):    %.1f sec (%d receivers, %d pairs)\n",
            t_step23, length(cell_types), n_pairs))
cat(sprintf("    chain computation:         %.1f sec\n", t_chain_total))
cat(sprintf("    L1 joins + enumeration:    %.1f sec\n", t_step23 - t_chain_total))
cat(sprintf("  Total:                       %.1f sec\n\n",
            t_step1 + t_step23))

# Top pairs by pathway count
results_df <- results_df[order(-results_df$n_pathways), ]
cat("Top 20 pairs by pathway count:\n")
print(head(results_df, 20), row.names = FALSE)

cat(sprintf("\nTotal pathways across all pairs: %s\n",
            format(sum(results_df$n_pathways), big.mark = ",")))
cat(sprintf("Pairs with 0 pathways: %d\n", sum(results_df$n_pathways == 0)))
cat(sprintf("Pairs with >100K pathways: %d\n", sum(results_df$n_pathways > 100000)))

# Validation: check Microglia-PVM -> L5 IT
mpvm_l5it <- results_df[results_df$sender == "Microglia-PVM" &
                         results_df$receiver == "L5 IT", ]
cat(sprintf("\nValidation (Microglia-PVM -> L5 IT): %d pathways",
            mpvm_l5it$n_pathways))
if (mpvm_l5it$n_pathways == 169462) {
  cat(" — MATCHES prior result\n")
} else {
  cat(sprintf(" — MISMATCH (expected 169462)\n"))
}

# Save
out_path <- file.path(get_script_dir(), "all_pairs_results.csv")
write.csv(results_df, out_path, row.names = FALSE)
cat(sprintf("\nSaved: %s\n", out_path))
