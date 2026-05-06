#!/usr/bin/env Rscript
# All-pairs Incytr pipeline: enumerate and run downstream for all 462
# sender-receiver pairs (22 cell types x 21 non-self).
#
# Phase 2: Receiver-centric vectorized scoring. Enumerates backbones and
# attaches senders via DuckDB (Phase A+B), then scores all senders for each
# receiver in a single vectorized pass (Phase C) without creating Incytr S4
# objects. Produces receiver-indexed Parquet files.
#
# Environment variables:
#   PAIR_FILTER       - Filter pairs, e.g. "Microglia-PVM:L5 IT", "*:L5 IT"
#   FORCE_RERUN       - Set to 1 to ignore checkpoints
#   MEMORY_LIMIT_GB   - Abort if R memory exceeds this (default 10)
#   EXPR_DETECTION_THRESHOLD - Gene detection threshold (default 0.10)
#   SKIP_EXPRONLY     - Set to 1 to skip expression-only evaluation
#
# Output:
#   all_pairs/recv_{receiver}.parquet   (22 files, sender as column)
#   all_pairs/pair_summary.csv
#
# Usage:
#   systemd-run --user --scope -p MemoryMax=12G \
#     micromamba run -n incytr Rscript code/integration/wrappers/run_incytr_all_pairs.R

suppressPackageStartupMessages({
  library(Matrix)
  library(data.table)
  library(duckdb)
  library(DBI)
  library(matrixStats)
  library(arrow)
})

# =========================================================================
# Paths + shared helpers
# =========================================================================
get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg[1]))))
  }
  return(file.path(getwd(), "code", "integration", "wrappers"))
}
script_dir <- get_script_dir()

# Source shared helpers: hill(), build_hill_sql(), weighted_quantile_expr()
source(file.path(script_dir, "duckdb_enumeration.R"))
# Source Phase 2 vectorized scoring
source(file.path(script_dir, "receiver_scoring.R"))

sanitize_name <- function(x) gsub("/", "-", gsub(" ", "_", x))

# Record all senders as skipped and unregister receiver-scoped DuckDB tables.
# Called from early-exit paths (NO_EDGES, NO_BACKBONES, NO_L1).
# Uses <<- to update counters in the enclosing for-loop scope.
skip_recv_senders <- function(senders, recv, status, con) {
  for (send in senders) {
    n_done <<- n_done + 1
    summary_rows[[length(summary_rows) + 1]] <<- data.frame(
      sender = send, receiver = recv,
      n_pre = 0L, n_post = 0L, time_sec = 0,
      status = status, stringsAsFactors = FALSE)
  }
  duckdb_unregister(con, "L2")
  duckdb_unregister(con, "L3")
  duckdb_unregister(con, "receiver_expr")
}

repo_root  <- normalizePath(file.path(script_dir, "..", "..", ".."))
int_dir    <- file.path(repo_root, "code", "integration", "intermediates")

cat("Repo root:", repo_root, "\n")
cat("Intermediates:", int_dir, "\n\n")

# =========================================================================
# Config from environment
# =========================================================================
expr_threshold  <- as.numeric(Sys.getenv("EXPR_DETECTION_THRESHOLD", "0.10"))
force_rerun     <- Sys.getenv("FORCE_RERUN", "0") == "1"
memory_limit_gb <- as.numeric(Sys.getenv("MEMORY_LIMIT_GB", "10"))
pair_filter     <- Sys.getenv("PAIR_FILTER", "")
skip_expronly   <- Sys.getenv("SKIP_EXPRONLY", "0") == "1"
K <- 0.5; N <- 2; KN <- K^N; cutoff_SigProb <- 0.01
conditions <- c("WT", "App")

cat(sprintf("Config: threshold=%.0f%%, memory_limit=%dGB, force=%s, skip_expronly=%s\n",
            expr_threshold * 100, memory_limit_gb, force_rerun, skip_expronly))
if (pair_filter != "") cat(sprintf("Pair filter: %s\n", pair_filter))
cat("\n")

# =========================================================================
# Section 1: Load shared data
# =========================================================================
cat("=== Section 1: Loading shared data ===\n")

cat("Loading Incytr (for DB layers)...\n")
library(Incytr)

cat("Loading expression matrix...\n")
mat <- readMM(file.path(int_dir, "expression_matrix.mtx"))
mat <- as(mat, "dgCMatrix")
genes    <- read.csv(file.path(int_dir, "expression_genes.csv"))$gene
barcodes <- read.csv(file.path(int_dir, "expression_barcodes.csv"))$barcode
rownames(mat) <- genes; colnames(mat) <- barcodes
all_genes <- genes
cat(sprintf("  %d genes x %d cells\n", nrow(mat), ncol(mat)))

meta <- read.csv(file.path(int_dir, "expression_metadata.csv"),
                 row.names = 1, check.names = FALSE)
cell_types <- sort(unique(meta$labels))
cat(sprintf("  %d cell types: %s\n", length(cell_types),
            paste(cell_types, collapse = ", ")))

cat("Loading IncytrDB mouse...\n")
data(DB_Layer1_mouse_filtered, package = "Incytr")
data(DB_Layer2_mouse_filtered, package = "Incytr")
data(DB_Layer3_mouse_filtered, package = "Incytr")

DB_Layer1_mouse_filtered <- DB_Layer1_mouse_filtered[
  DB_Layer1_mouse_filtered$from %in% all_genes &
  DB_Layer1_mouse_filtered$to %in% all_genes, ]
DB_Layer2_mouse_filtered <- DB_Layer2_mouse_filtered[
  DB_Layer2_mouse_filtered$from %in% all_genes &
  DB_Layer2_mouse_filtered$to %in% all_genes, ]
DB_Layer3_mouse_filtered <- DB_Layer3_mouse_filtered[
  DB_Layer3_mouse_filtered$from %in% all_genes &
  DB_Layer3_mouse_filtered$to %in% all_genes, ]
cat(sprintf("  Filtered DB: L1=%d, L2=%d, L3=%d edges\n",
            nrow(DB_Layer1_mouse_filtered),
            nrow(DB_Layer2_mouse_filtered),
            nrow(DB_Layer3_mouse_filtered)))


# Convert to data.tables with just from/to for pruning
l1_raw <- as.data.table(DB_Layer1_mouse_filtered[, c("from", "to")])
l2_raw <- as.data.table(DB_Layer2_mouse_filtered[, c("from", "to")])
l3_raw <- as.data.table(DB_Layer3_mouse_filtered[, c("from", "to")])
rm(DB_Layer1_mouse_filtered, DB_Layer2_mouse_filtered, DB_Layer3_mouse_filtered)
gc(verbose = FALSE)

# =========================================================================
# Section 2: Load integration data
# =========================================================================
cat("\n=== Section 2: Loading integration data ===\n")

# Phospho
ps1 <- NULL; ps2 <- NULL
ps1_path <- file.path(int_dir, "ps_condition1.csv")
ps2_path <- file.path(int_dir, "ps_condition2.csv")
if (file.exists(ps1_path) && file.exists(ps2_path)) {
  ps1 <- read.csv(ps1_path, check.names = FALSE)
  ps2 <- read.csv(ps2_path, check.names = FALSE)
  cat(sprintf("  Phospho: %d genes, %d cell-type columns\n",
              nrow(ps1), ncol(ps1) - 1))
} else {
  cat("  Phospho files not found, skipping phospho integration.\n")
}

# Kinase data
kldata <- NULL
kldata_path <- file.path(int_dir, "kldata.csv")
if (file.exists(kldata_path)) {
  kldata <- read.csv(kldata_path, check.names = FALSE)
  cat(sprintf("  kldata: %d rows, %d kinases\n",
              nrow(kldata), length(unique(kldata[["motif.geneName"]]))))
} else {
  cat("  kldata.csv not found, skipping kinase integration.\n")
}

kl_out <- NULL
kl_output_path <- file.path(int_dir, "kl_output.csv")
if (file.exists(kl_output_path)) {
  kl_out <- read.csv(kl_output_path, check.names = FALSE)
  cat(sprintf("  kl_output: %d rows, %d kinases\n",
              nrow(kl_out), length(unique(kl_out$kinase))))
} else {
  cat("  kl_output.csv not found, skipping activity kinase.\n")
}

# Kinase-imputed genes (refined per-receiver with legacy fallback).
# Refined adapter emits intermediates/kinase_imputed_genes__{receiver}.csv
# with columns (gene, best_fdr, imputed_weight). Legacy adapter emits a
# flat intermediates/kinase_imputed_genes.csv keyed only on gene.
expr_imputation_floor <- as.numeric(Sys.getenv("EXPR_IMPUTATION_FLOOR", "0.05"))
cat(sprintf("  EXPR_IMPUTATION_FLOOR = %.3f\n", expr_imputation_floor))

legacy_imputed_df <- NULL
legacy_imputed_path <- file.path(int_dir, "kinase_imputed_genes.csv")
if (file.exists(legacy_imputed_path)) {
  tmp <- read.csv(legacy_imputed_path)
  if (nrow(tmp) > 0) {
    if (!"best_fdr" %in% names(tmp)) tmp$best_fdr <- 0
    if (!"imputed_weight" %in% names(tmp)) tmp$imputed_weight <- 1.0
    tmp <- tmp[tmp$gene %in% all_genes, , drop = FALSE]
    legacy_imputed_df <- tmp[, c("gene", "best_fdr", "imputed_weight")]
    cat(sprintf("  Legacy kinase_imputed_genes.csv: %d genes\n",
                nrow(legacy_imputed_df)))
  }
}

per_recv_imputed_files <- Sys.glob(file.path(int_dir, "kinase_imputed_genes__*.csv"))
cat(sprintf("  Per-receiver imputed files: %d\n", length(per_recv_imputed_files)))
if (length(per_recv_imputed_files) == 0 && is.null(legacy_imputed_df)) {
  cat("  No imputed files found; running expression-only.\n")
}

load_imputed_for_recv <- function(recv) {
  fname <- paste0("kinase_imputed_genes__", sanitize_name(recv), ".csv")
  per_recv <- file.path(int_dir, fname)
  if (file.exists(per_recv)) {
    df <- read.csv(per_recv)
    if (nrow(df) == 0) {
      return(data.frame(gene = character(), best_fdr = numeric(),
                        imputed_weight = numeric()))
    }
    if (!"best_fdr" %in% names(df)) df$best_fdr <- 0
    if (!"imputed_weight" %in% names(df)) df$imputed_weight <- pmax(0, 1 - df$best_fdr)
    df <- df[df$gene %in% all_genes, , drop = FALSE]
    return(df[, c("gene", "best_fdr", "imputed_weight")])
  }
  return(legacy_imputed_df)
}

# =========================================================================
# Section 3: Precompute per-cell-type data
# =========================================================================
cat("\n=== Section 3: Precomputing per-cell-type expression ===\n")
t0_precomp <- proc.time()

# Weighted quantile expression for all cell types x conditions
wq_expr <- list()
for (ct in cell_types) {
  wq_expr[[ct]] <- list()
  for (cond in conditions) {
    cells <- which(meta$labels == ct & meta$condition == cond)
    wq_expr[[ct]][[cond]] <- weighted_quantile_expr(mat[, cells, drop = FALSE])
  }
}

# Detection rates (pooled across conditions)
det_rates <- list()
for (ct in cell_types) {
  cells <- which(meta$labels == ct)
  det_rates[[ct]] <- Matrix::rowMeans(mat[, cells, drop = FALSE] > 0)
}

# Gene lists (expression threshold)
gene_lists <- list()
for (ct in cell_types) {
  gene_lists[[ct]] <- names(det_rates[[ct]][det_rates[[ct]] >= expr_threshold])
}

t_precomp <- (proc.time() - t0_precomp)["elapsed"]
cat(sprintf("  %d cell types x %d conditions (%.1f sec)\n",
            length(cell_types), length(conditions), t_precomp))
cat(sprintf("  Gene list sizes: min=%d, median=%d, max=%d\n",
            min(sapply(gene_lists, length)),
            median(sapply(gene_lists, length)),
            max(sapply(gene_lists, length))))

# =========================================================================
# Section 4: Setup output + pair list
# =========================================================================
cat("\n=== Section 4: Setup ===\n")

out_base <- file.path(int_dir, "all_pairs")
dir.create(out_base, showWarnings = FALSE, recursive = TRUE)

# Build pair list (sorted by receiver for nested loop efficiency)
all_pairs <- expand.grid(sender = cell_types, receiver = cell_types,
                         stringsAsFactors = FALSE)
all_pairs <- all_pairs[all_pairs$sender != all_pairs$receiver, ]

# Apply pair filter
if (pair_filter != "") {
  parts <- strsplit(pair_filter, ":")[[1]]
  if (length(parts) == 2) {
    if (parts[1] != "*") all_pairs <- all_pairs[all_pairs$sender == parts[1], ]
    if (parts[2] != "*") all_pairs <- all_pairs[all_pairs$receiver == parts[2], ]
  }
}

all_pairs <- all_pairs[order(all_pairs$receiver, all_pairs$sender), ]
rownames(all_pairs) <- NULL

n_total <- nrow(all_pairs)
cat(sprintf("  %d pairs to process\n", n_total))

if (n_total == 0) {
  cat("No pairs to process. Exiting.\n")
  quit(status = 0)
}

# =========================================================================
# Section 5: DuckDB setup + nested enumeration + downstream
# =========================================================================
cat("\n=== Section 5: Nested enumeration + downstream ===\n")
t0_all <- proc.time()

# --- DuckDB connection ---
duck_dir <- file.path(Sys.getenv("HOME"), ".cache", "duckdb_incytr")
dir.create(duck_dir, showWarnings = FALSE, recursive = TRUE)
duck_tmp <- tempfile(tmpdir = duck_dir, fileext = ".duckdb")
duck_tmp_dir <- paste0(duck_tmp, ".tmp")
con <- dbConnect(duckdb(), dbdir = duck_tmp)
on.exit({
  tryCatch(dbDisconnect(con, shutdown = TRUE), error = function(e) NULL)
  unlink(duck_tmp, force = TRUE)
  unlink(duck_tmp_dir, recursive = TRUE, force = TRUE)
}, add = TRUE)
dbExecute(con, "SET memory_limit='6GB'")
dbExecute(con, sprintf("SET threads=%d", 4L))
dbExecute(con, "SET max_temp_directory_size='20GiB'")
dbExecute(con, "SET preserve_insertion_order=false")

# --- Build SQL templates ---
h_l2_c1 <- build_hill_sql("r1.c1",  "r2.c1", N, KN)
h_l2_c2 <- build_hill_sql("r1.c2",  "r2.c2", N, KN)
h_l3_c1 <- build_hill_sql("r2.c1", "r3.c1", N, KN)
h_l3_c2 <- build_hill_sql("r2.c2", "r3.c2", N, KN)

# Phase A: Backbone enumeration (receiver-only, L2 x L3)
# Produces all R-EM-T triples with pre-computed receiver-side SigProb components.
# Filter: receiver component >= cutoff (lossless, since Hill_L1 <= 1).
sql_backbone <- sprintf('
  SELECT DISTINCT
    L2."from" AS Receptor,
    L2."to"   AS EM,
    L3."to"   AS Target,
    %s AS h_l2_c1,
    %s AS h_l2_c2,
    %s AS h_l3_c1,
    %s AS h_l3_c2
  FROM L2
  JOIN L3 ON L2."to" = L3."from"
  JOIN receiver_expr r1 ON L2."from" = r1.gene
  JOIN receiver_expr r2 ON L2."to"  = r2.gene
  JOIN receiver_expr r3 ON L3."to"  = r3.gene
  WHERE L2."from" != L2."to"
    AND L2."from" != L3."to"
    AND L2."to"   != L3."to"
    AND (
      (%s * %s) >= %f
      OR
      (%s * %s) >= %f
    )
', h_l2_c1, h_l2_c2, h_l3_c1, h_l3_c2,
   h_l2_c1, h_l3_c1, cutoff_SigProb,
   h_l2_c2, h_l3_c2, cutoff_SigProb)

# Phase B: All-sender ligand attachment
# Joins L1 edges from ALL senders against pre-enumerated backbones in one query.
# sender_all_expr is a long-format table: (gene, cell_type, c1, c2).
# Full SigProb filter applied here (Hill_L1 * receiver component >= cutoff).
# L1 Hill uses raw receiver R expression (from receiver_expr r1) × sender L expression.
# The backbone's h_l2/h_l3 carry the receiver-only SigProb components.
sql_attach <- sprintf('
  SELECT DISTINCT
    se.cell_type AS sender,
    L1."from" AS Ligand,
    bb.Receptor,
    bb.EM,
    bb.Target
  FROM sender_all_expr se
  JOIN L1 ON L1."from" = se.gene
  JOIN backbones bb ON L1."to" = bb.Receptor
  JOIN receiver_expr r1 ON bb.Receptor = r1.gene
  WHERE se.cell_type != ?
    AND L1."from" != bb.Receptor
    AND L1."from" != bb.EM
    AND L1."from" != bb.Target
    AND (
      (%s * bb.h_l2_c1 * bb.h_l3_c1) >= %f
      OR
      (%s * bb.h_l2_c2 * bb.h_l3_c2) >= %f
    )
', build_hill_sql("se.c1", "r1.c1", N, KN), cutoff_SigProb,
   build_hill_sql("se.c2", "r1.c2", N, KN), cutoff_SigProb)

# --- Summary tracking ---
summary_rows <- list()
n_done <- 0
n_skipped <- 0
n_errors <- 0
abort_memory <- FALSE

# --- Nested loop ---
receivers_in_order <- unique(all_pairs$receiver)

for (recv in receivers_in_order) {
  t0_recv <- proc.time()

  # Receiver expression
  r_c1 <- wq_expr[[recv]][[conditions[1]]]
  r_c2 <- wq_expr[[recv]][[conditions[2]]]

  # Receiver gene list (expression + kinase-imputed, per-receiver refined).
  recv_genes_expr <- gene_lists[[recv]]
  recv_imputed_df <- load_imputed_for_recv(recv)
  if (is.null(recv_imputed_df)) {
    recv_imputed_df <- data.frame(gene = character(), best_fdr = numeric(),
                                  imputed_weight = numeric())
  }

  # R3: expression floor — imputed substrate must have some RNA evidence
  # in this receiver even if bulk protein evidence exists in another type.
  if (expr_imputation_floor > 0 && nrow(recv_imputed_df) > 0) {
    dr <- det_rates[[recv]][recv_imputed_df$gene]
    dr[is.na(dr)] <- 0
    recv_imputed_df <- recv_imputed_df[dr >= expr_imputation_floor, , drop = FALSE]
  }

  ki_for_recv <- setdiff(recv_imputed_df$gene, recv_genes_expr)
  recv_genes <- union(recv_genes_expr, ki_for_recv)
  imputed_weight_vec <- setNames(recv_imputed_df$imputed_weight,
                                 recv_imputed_df$gene)

  # Patch kinase-imputed expression with SOFT rescue (R2):
  # rescued = imputed_weight * rowMeans, weighting by 1 - best_fdr so weakly
  # significant kinases rescue less aggressively than strong ones.
  r_c1_patched <- r_c1
  r_c2_patched <- r_c2
  ki_rm_c1 <- NULL; ki_rm_c2 <- NULL
  recv_cells_c1 <- which(meta$labels == recv & meta$condition == conditions[1])
  recv_cells_c2 <- which(meta$labels == recv & meta$condition == conditions[2])
  if (length(ki_for_recv) > 0) {
    ki_zero <- ki_for_recv[r_c1[ki_for_recv] == 0 | r_c2[ki_for_recv] == 0]
    ki_zero <- ki_zero[!is.na(ki_zero)]
    if (length(ki_zero) > 0) {
      ki_rm_c1 <- setNames(Matrix::rowMeans(mat[ki_zero, recv_cells_c1, drop = FALSE]), ki_zero)
      ki_rm_c2 <- setNames(Matrix::rowMeans(mat[ki_zero, recv_cells_c2, drop = FALSE]), ki_zero)
      w <- imputed_weight_vec[ki_zero]
      w[is.na(w)] <- 1.0
      r_c1_patched[ki_zero] <- pmax(r_c1_patched[ki_zero], w * ki_rm_c1)
      r_c2_patched[ki_zero] <- pmax(r_c2_patched[ki_zero], w * ki_rm_c2)
    }
  }

  # --- Prune L2/L3 for this receiver ---
  dt2 <- l2_raw[l2_raw$from %in% recv_genes & l2_raw$to %in% recv_genes]
  h1 <- hill(r_c1_patched[dt2$from] * r_c1_patched[dt2$to], K, N)
  h2 <- hill(r_c2_patched[dt2$from] * r_c2_patched[dt2$to], K, N)
  dt2 <- dt2[(h1 >= cutoff_SigProb) | (h2 >= cutoff_SigProb)]

  dt3 <- l3_raw[l3_raw$from %in% recv_genes & l3_raw$to %in% recv_genes]
  h1 <- hill(r_c1_patched[dt3$from] * r_c1_patched[dt3$to], K, N)
  h2 <- hill(r_c2_patched[dt3$from] * r_c2_patched[dt3$to], K, N)
  dt3 <- dt3[(h1 >= cutoff_SigProb) | (h2 >= cutoff_SigProb)]
  rm(h1, h2)

  n_l2 <- nrow(dt2); n_l3 <- nrow(dt3)

  # Register receiver tables in DuckDB
  duckdb_register(con, "L2", as.data.frame(dt2))
  duckdb_register(con, "L3", as.data.frame(dt3))
  recv_expr_df <- data.frame(gene = all_genes,
                             c1 = unname(r_c1_patched),
                             c2 = unname(r_c2_patched),
                             stringsAsFactors = FALSE)
  duckdb_register(con, "receiver_expr", recv_expr_df)
  rm(dt2, dt3, recv_expr_df)

  # Get senders for this receiver
  senders <- all_pairs$sender[all_pairs$receiver == recv]
  recv_pathways_total <- 0

  cat(sprintf("\n--- Receiver: %s  (L2=%d, L3=%d, %d senders) ---\n",
              recv, n_l2, n_l3, length(senders)))

  # Pre-check checkpoint: skip if receiver Parquet exists
  recv_parquet <- file.path(out_base, paste0("recv_", sanitize_name(recv), ".parquet"))
  if (!force_rerun && file.exists(recv_parquet)) {
    cat(sprintf("  Receiver %s checkpointed (Parquet), skipping.\n", recv))
    for (send in senders) {
      n_done <- n_done + 1
      n_skipped <- n_skipped + 1
      summary_rows[[length(summary_rows) + 1]] <- data.frame(
        sender = send, receiver = recv,
        n_pre = NA_integer_, n_post = NA_integer_, time_sec = 0,
        status = "CHECKPOINT", stringsAsFactors = FALSE)
    }
    duckdb_unregister(con, "L2")
    duckdb_unregister(con, "L3")
    duckdb_unregister(con, "receiver_expr")
    next
  }

  if (n_l2 == 0 || n_l3 == 0) {
    cat("  No surviving L2/L3 edges, skipping all senders.\n")
    skip_recv_senders(senders, recv, "NO_EDGES", con)
    next
  }

  # --- Phase A: Backbone enumeration (one query per receiver) ---
  t0_bb <- proc.time()
  backbone_df <- dbGetQuery(con, sql_backbone)
  t_bb <- (proc.time() - t0_bb)["elapsed"]
  cat(sprintf("  Backbone enumeration: %s R-EM-T triples (%.1fs)\n",
              format(nrow(backbone_df), big.mark = ","), t_bb))

  if (nrow(backbone_df) == 0) {
    cat("  No backbones survived receiver-side SigProb filter.\n")
    skip_recv_senders(senders, recv, "NO_BACKBONES", con)
    next
  }

  # --- Phase B: All-sender ligand attachment (one query per receiver) ---
  # Build long-format sender expression table (all senders for this receiver)
  t0_attach <- proc.time()
  sender_rows <- list()
  for (send in senders) {
    s_genes <- gene_lists[[send]]
    # Only include genes that appear as L1 "from" and are in this sender's gene list
    l1_sender_genes <- intersect(s_genes, l1_raw$from)
    if (length(l1_sender_genes) == 0) next
    sender_rows[[length(sender_rows) + 1]] <- data.frame(
      gene = l1_sender_genes,
      cell_type = send,
      c1 = unname(wq_expr[[send]][[conditions[1]]][l1_sender_genes]),
      c2 = unname(wq_expr[[send]][[conditions[2]]][l1_sender_genes]),
      stringsAsFactors = FALSE)
  }

  if (length(sender_rows) > 0) {
    sender_all_df <- rbindlist(sender_rows)
    rm(sender_rows)
  } else {
    cat("  No senders have L1 ligand genes.\n")
    skip_recv_senders(senders, recv, "NO_L1", con)
    next
  }

  # Prune L1 edges: keep only edges where from is any sender gene and to is a
  # backbone receptor. R-side Hill pre-pruning uses receiver expression.
  backbone_receptors <- unique(backbone_df$Receptor)
  dt1_all <- l1_raw[l1_raw$from %in% sender_all_df$gene &
                     l1_raw$to %in% backbone_receptors]

  # Register tables for attachment query
  duckdb_register(con, "backbones", backbone_df)
  duckdb_register(con, "sender_all_expr", sender_all_df)
  duckdb_register(con, "L1", dt1_all)
  rm(sender_all_df, dt1_all)

  all_pathways_df <- as.data.table(dbGetQuery(con, sql_attach, params = list(recv)))

  duckdb_unregister(con, "backbones")
  duckdb_unregister(con, "sender_all_expr")
  duckdb_unregister(con, "L1")
  rm(backbone_df)

  t_attach <- (proc.time() - t0_attach)["elapsed"]
  n_senders_with_pw <- length(unique(all_pathways_df$sender))
  cat(sprintf("  Sender attachment: %s pathways across %d senders (%.1fs)\n",
              format(nrow(all_pathways_df), big.mark = ","),
              n_senders_with_pw, t_attach))

  # --- Phase C: Vectorized scoring (replaces per-sender loop) ---
  recv_pathways_total <- nrow(all_pathways_df)

  tryCatch({
    recv_summary <- score_receiver_all_senders(
      all_pathways_df = all_pathways_df,
      recv = recv,
      wq_expr = wq_expr,
      gene_lists = gene_lists,
      recv_genes_expr = recv_genes_expr,
      recv_c1 = r_c1_patched, recv_c2 = r_c2_patched,
      ps1 = ps1, ps2 = ps2,
      kldata = kldata, kl_out = kl_out,
      cell_types = cell_types,
      conditions = conditions,
      K = K, N = N,
      cutoff_SigProb = cutoff_SigProb,
      output_dir = out_base,
      sanitize_name_fn = sanitize_name
    )

    for (i in seq_len(nrow(recv_summary))) {
      n_done <- n_done + 1
      summary_rows[[length(summary_rows) + 1]] <- recv_summary[i, , drop = FALSE]
    }
  }, error = function(e) {
    cat(sprintf("  ERROR scoring receiver %s: %s\n", recv, e$message))
    for (send in senders) {
      n_done <<- n_done + 1
      n_errors <<- n_errors + 1
      summary_rows[[length(summary_rows) + 1]] <<- data.frame(
        sender = send, receiver = recv,
        n_pre = 0L, n_post = 0L, time_sec = 0,
        status = paste0("ERROR: ", e$message), stringsAsFactors = FALSE)
    }
  })

  # Free pre-enumerated pathways for this receiver
  rm(all_pathways_df)

  # Unregister receiver tables
  duckdb_unregister(con, "L2")
  duckdb_unregister(con, "L3")
  duckdb_unregister(con, "receiver_expr")

  # GC + memory guard once per receiver (not per sender — avoids 21 stop-the-world pauses)
  mem_mb <- gc(verbose = FALSE)[2, 2]
  if (mem_mb > memory_limit_gb * 1024) {
    cat(sprintf("\nABORT: R memory %.0f MB exceeds %d GB limit.\n",
                mem_mb, memory_limit_gb))
    cat("Re-run to resume from checkpoint.\n")
    abort_memory <- TRUE
  }

  t_recv <- (proc.time() - t0_recv)["elapsed"]
  cat(sprintf("  Receiver %s complete: %s pathways across %d senders (%.1fs)\n",
              recv, format(recv_pathways_total, big.mark = ","),
              length(senders), t_recv))

  if (abort_memory) break
}  # end receiver loop

# =========================================================================
# Section 6: Cleanup + summary
# =========================================================================
t_all <- (proc.time() - t0_all)["elapsed"]

# DuckDB cleanup handled by on.exit()

# Write summary
if (length(summary_rows) > 0) {
  summary_df <- do.call(rbind, summary_rows)
  summary_path <- file.path(out_base, "pair_summary.csv")
  write.csv(summary_df, summary_path, row.names = FALSE)
  cat(sprintf("\n=== Summary ===\n"))
  cat(sprintf("  Pairs processed: %d/%d\n", n_done, n_total))
  cat(sprintf("  Checkpointed:    %d\n", n_skipped))
  cat(sprintf("  Errors:          %d\n", n_errors))
  cat(sprintf("  Total pathways:  %s (pre-SigProb)\n",
              format(sum(summary_df$n_pre, na.rm = TRUE), big.mark = ",")))
  cat(sprintf("  Total time:      %.1f min\n", t_all / 60))
  cat(sprintf("  Wrote %s\n", summary_path))
} else {
  cat("\nNo pairs processed.\n")
}

cat("\nAll-pairs pipeline complete.\n")
