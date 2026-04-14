#!/usr/bin/env Rscript
# Full 462-pair pathway enumeration using DuckDB.
#
# Structure:
#   1. Compute per-condition mean expression for all 22 cell types (once)
#   2. Per receiver: prune L2/L3, register in DuckDB
#      Per sender: prune L1, register, run 3-way join, collect count
#      Free receiver tables before next receiver.
#
# Validates Microglia-PVM -> L5 IT produces 4,715,939 enumerated pathways
# (169,462 survive SigProb — that filter is not applied here).
#
# Usage:
#   micromamba run -n incytr Rscript code/integration/tests/edge_pruning/test_duckdb_full.R

suppressPackageStartupMessages({
  library(Matrix)
  library(data.table)
  library(duckdb)
  library(DBI)
  library(Incytr)
})

hill <- function(x, K = 0.5, N = 2) x^N / (x^N + K^N)

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
cat(sprintf("  %d cell types x %d conditions (%.1f sec)\n\n",
            length(cell_types), length(conditions), t_step1))

# Free expression matrix
rm(mat, genes, barcodes)
gc(verbose = FALSE)

# ===================================================================
# Step 2+3: Per receiver — prune L2/L3, then DuckDB join per sender
# ===================================================================
cat("Step 2+3: DuckDB enumeration across all pairs...\n")
t0_enum <- proc.time()

# DuckDB connection — use temp file so it can spill to disk
duck_tmp <- tempfile(fileext = ".duckdb")
con <- dbConnect(duckdb(), dbdir = duck_tmp)
dbExecute(con, "SET memory_limit='6GB'")
dbExecute(con, "SET threads=4")

# The 3-way join query (parameterless — operates on registered tables)
JOIN_SQL <- "
  SELECT DISTINCT
    L1.\"from\" AS Ligand,
    L1.\"to\"   AS Receptor,
    L2.\"to\"   AS EM,
    L3.\"to\"   AS Target
  FROM L1
  JOIN L2 ON L1.\"to\" = L2.\"from\"
  JOIN L3 ON L2.\"to\" = L3.\"from\"
  WHERE L1.\"from\" != L1.\"to\"
    AND L1.\"from\" != L2.\"to\"
    AND L1.\"from\" != L3.\"to\"
    AND L1.\"to\"   != L2.\"to\"
    AND L1.\"to\"   != L3.\"to\"
    AND L2.\"to\"   != L3.\"to\"
"

# Count-only version (don't materialize 4.7M rows per pair in R)
COUNT_SQL <- paste0("SELECT COUNT(*) AS n FROM (", JOIN_SQL, ")")

pair_results <- list()
n_pairs <- 0

for (recv in cell_types) {
  t0_recv <- proc.time()
  rc1 <- mean_expr[[recv]][["WT"]]
  rc2 <- mean_expr[[recv]][["App"]]

  # Prune L2 (Receptor -> EM, both receiver genes)
  dt2 <- copy(l2_raw)
  h1 <- hill(rc1[dt2$from] * rc1[dt2$to], K, N)
  h2 <- hill(rc2[dt2$from] * rc2[dt2$to], K, N)
  dt2 <- dt2[(h1 >= CUTOFF) | (h2 >= CUTOFF)]

  # Prune L3 (EM -> Target, both receiver genes)
  dt3 <- copy(l3_raw)
  h1 <- hill(rc1[dt3$from] * rc1[dt3$to], K, N)
  h2 <- hill(rc2[dt3$from] * rc2[dt3$to], K, N)
  dt3 <- dt3[(h1 >= CUTOFF) | (h2 >= CUTOFF)]

  # Register pruned L2/L3 for this receiver
  duckdb_register(con, "L2", as.data.frame(dt2))
  duckdb_register(con, "L3", as.data.frame(dt3))

  n_l2 <- nrow(dt2); n_l3 <- nrow(dt3)
  rm(dt2, dt3, h1, h2)

  # Iterate senders
  n_recv_pairs <- 0
  recv_pathways <- 0

  for (send in cell_types) {
    if (send == recv) next
    n_pairs <- n_pairs + 1
    n_recv_pairs <- n_recv_pairs + 1

    sc1 <- mean_expr[[send]][["WT"]]
    sc2 <- mean_expr[[send]][["App"]]

    # Prune L1 (Ligand=sender, Receptor=receiver)
    dt1 <- copy(l1_raw)
    h1 <- hill(sc1[dt1$from] * rc1[dt1$to], K, N)
    h2 <- hill(sc2[dt1$from] * rc2[dt1$to], K, N)
    dt1 <- dt1[(h1 >= CUTOFF) | (h2 >= CUTOFF)]

    if (nrow(dt1) == 0) {
      pair_results[[n_pairs]] <- data.frame(
        sender = send, receiver = recv,
        l1_edges = 0L, n_pathways = 0L,
        stringsAsFactors = FALSE)
      next
    }

    # Register L1 and run count query
    duckdb_register(con, "L1", as.data.frame(dt1))
    n_pw <- dbGetQuery(con, COUNT_SQL)$n
    duckdb_unregister(con, "L1")

    pair_results[[n_pairs]] <- data.frame(
      sender = send, receiver = recv,
      l1_edges = nrow(dt1), n_pathways = n_pw,
      stringsAsFactors = FALSE)
    recv_pathways <- recv_pathways + n_pw
  }

  # Unregister receiver tables
  duckdb_unregister(con, "L2")
  duckdb_unregister(con, "L3")
  gc(verbose = FALSE)

  t_recv <- (proc.time() - t0_recv)["elapsed"]
  cat(sprintf("  %-18s  L2: %6d  L3: %6d  %2d senders  %12s pathways  (%.1fs)\n",
              recv, n_l2, n_l3, n_recv_pairs,
              format(recv_pathways, big.mark = ","), t_recv))
}

t_enum <- (proc.time() - t0_enum)["elapsed"]

# Shutdown DuckDB
dbDisconnect(con, shutdown = TRUE)
unlink(duck_tmp)

results_df <- do.call(rbind, pair_results)

# ===================================================================
# Summary
# ===================================================================
cat(sprintf("\n====== Summary ======\n"))
cat(sprintf("  Step 1 (mean expression):  %.1f sec\n", t_step1))
cat(sprintf("  Steps 2+3 (enumeration):   %.1f sec (%d pairs)\n", t_enum, n_pairs))
cat(sprintf("  Total:                     %.1f sec\n", t_step1 + t_enum))
cat(sprintf("  Avg per pair:              %.0f ms\n", 1000 * t_enum / n_pairs))

mem_final <- gc(verbose = FALSE)[2, 2]
cat(sprintf("  Final R memory:            %.0f MB\n\n", mem_final))

# Top pairs
results_df <- results_df[order(-results_df$n_pathways), ]
cat("Top 20 pairs by pathway count:\n")
print(head(results_df, 20), row.names = FALSE)

cat(sprintf("\nTotal pathways across all pairs: %s\n",
            format(sum(results_df$n_pathways), big.mark = ",")))
cat(sprintf("Pairs with 0 pathways: %d\n", sum(results_df$n_pathways == 0)))
cat(sprintf("Pairs with >1M pathways: %d\n", sum(results_df$n_pathways > 1000000)))

# Validation
mpvm_l5it <- results_df[results_df$sender == "Microglia-PVM" &
                         results_df$receiver == "L5 IT", ]
cat(sprintf("\nValidation (Microglia-PVM -> L5 IT): %s pathways",
            format(mpvm_l5it$n_pathways, big.mark = ",")))
if (mpvm_l5it$n_pathways == 4715939) {
  cat(" — MATCHES single-pair DuckDB result\n")
} else {
  cat(sprintf(" — expected 4,715,939\n"))
}

out_path <- file.path(get_script_dir(), "duckdb_all_pairs_results.csv")
write.csv(results_df, out_path, row.names = FALSE)
cat(sprintf("\nSaved: %s\n", out_path))
