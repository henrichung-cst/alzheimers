#!/usr/bin/env Rscript
# All-pairs Incytr pipeline: enumerate and run downstream for all 462
# sender-receiver pairs (22 cell types x 21 non-self).
#
# Uses a nested loop with shared L2/L3 pruning per receiver for efficient
# DuckDB enumeration. Full downstream pipeline per pair (no permutation tests).
#
# Environment variables:
#   PAIR_FILTER       - Filter pairs, e.g. "Microglia-PVM:L5 IT", "*:L5 IT"
#   FORCE_RERUN       - Set to 1 to ignore checkpoints
#   MEMORY_LIMIT_GB   - Abort if R memory exceeds this (default 10)
#   EXPR_DETECTION_THRESHOLD - Gene detection threshold (default 0.10)
#   SKIP_EXPRONLY     - Set to 1 to skip expression-only evaluation
#
# Outputs per pair:
#   all_pairs/{sender}__{receiver}/results_full.csv
#   all_pairs/{sender}__{receiver}/results_expronly.csv  (unless SKIP_EXPRONLY=1)
#   all_pairs/{sender}__{receiver}/edge_list_l{1,2,3}.csv
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

sanitize_name <- function(x) gsub("/", "-", gsub(" ", "_", x))

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

cat("Loading Incytr...\n")
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

# Compute em_degree and edge_source_count from full L3 (before reducing columns)
em_degree <- table(DB_Layer3_mouse_filtered$from)
l3_dt_full <- as.data.table(DB_Layer3_mouse_filtered[, c("from", "to", "source")])
edge_source_count <- l3_dt_full[, .(n_sources = uniqueN(source)),
                                by = .(EM = from, Target = to)]
rm(l3_dt_full)

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

# Kinase-imputed genes
kinase_imputed_genes <- character(0)
imputed_path <- file.path(int_dir, "kinase_imputed_genes.csv")
if (file.exists(imputed_path)) {
  kinase_imputed_genes <- read.csv(imputed_path)$gene
  kinase_imputed_genes <- kinase_imputed_genes[kinase_imputed_genes %in% all_genes]
  cat(sprintf("  Kinase-imputed genes: %d\n", length(kinase_imputed_genes)))
} else {
  cat("  No kinase_imputed_genes.csv, using expression-only.\n")
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

# Register global em_degree table
em_deg_df <- data.frame(gene = names(em_degree), degree = as.numeric(em_degree),
                        stringsAsFactors = FALSE)
duckdb_register(con, "em_degree_tbl", em_deg_df)
rm(em_deg_df)

# --- Build SQL template ---
h_l1_c1 <- build_hill_sql("se.c1",  "r1.c1", N, KN)
h_l2_c1 <- build_hill_sql("r1.c1",  "r2.c1", N, KN)
h_l1_c2 <- build_hill_sql("se.c2",  "r1.c2", N, KN)
h_l2_c2 <- build_hill_sql("r1.c2",  "r2.c2", N, KN)

em_w <- "(1.0 / LOG2(CAST(1 + COALESCE(ed.degree, 1) AS DOUBLE)))"
l3_prod_c1 <- sprintf("(r2.c1 * r3.c1 * %s)", em_w)
l3_prod_c2 <- sprintf("(r2.c2 * r3.c2 * %s)", em_w)
h_l3_c1 <- sprintf("POWER(%s, %d) / (POWER(%s, %d) + %f)", l3_prod_c1, N, l3_prod_c1, N, KN)
h_l3_c2 <- sprintf("POWER(%s, %d) / (POWER(%s, %d) + %f)", l3_prod_c2, N, l3_prod_c2, N, KN)

sql <- sprintf('
  SELECT DISTINCT
    L1."from" AS Ligand,
    L1."to"   AS Receptor,
    L2."to"   AS EM,
    L3."to"   AS Target
  FROM L1
  JOIN L2 ON L1."to" = L2."from"
  JOIN L3 ON L2."to" = L3."from"
  JOIN sender_expr   se ON L1."from" = se.gene
  JOIN receiver_expr r1 ON L1."to"  = r1.gene
  JOIN receiver_expr r2 ON L2."to"  = r2.gene
  JOIN receiver_expr r3 ON L3."to"  = r3.gene
  LEFT JOIN em_degree_tbl ed ON L2."to" = ed.gene
  WHERE L1."from" != L1."to"
    AND L1."from" != L2."to"
    AND L1."from" != L3."to"
    AND L1."to"   != L2."to"
    AND L1."to"   != L3."to"
    AND L2."to"   != L3."to"
    AND (
      (%s * %s * %s) >= %f
      OR
      (%s * %s * %s) >= %f
    )
', h_l1_c1, h_l2_c1, h_l3_c1, cutoff_SigProb,
   h_l1_c2, h_l2_c2, h_l3_c2, cutoff_SigProb)

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

  # Receiver gene list (expression + kinase-imputed)
  recv_genes_expr <- gene_lists[[recv]]
  ki_for_recv <- setdiff(kinase_imputed_genes, recv_genes_expr)
  recv_genes <- union(recv_genes_expr, ki_for_recv)

  # Patch kinase-imputed expression (rowMeans where weighted quantile is zero)
  # Cache rowMeans per receiver — reused in Expr_bygroup patching for each sender
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
      r_c1_patched[ki_zero] <- pmax(r_c1_patched[ki_zero], ki_rm_c1)
      r_c2_patched[ki_zero] <- pmax(r_c2_patched[ki_zero], ki_rm_c2)
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

  if (n_l2 == 0 || n_l3 == 0) {
    cat("  No surviving L2/L3 edges, skipping all senders.\n")
    for (send in senders) {
      n_done <- n_done + 1
      summary_rows[[length(summary_rows) + 1]] <- data.frame(
        sender = send, receiver = recv,
        n_pre = 0L, n_post = 0L, time_sec = 0,
        status = "NO_EDGES", stringsAsFactors = FALSE)
    }
    duckdb_unregister(con, "L2")
    duckdb_unregister(con, "L3")
    duckdb_unregister(con, "receiver_expr")
    next
  }

  for (send in senders) {
    n_done <- n_done + 1
    pair_label <- sprintf("%s -> %s", send, recv)

    # --- Checkpoint ---
    pair_dir <- file.path(out_base,
                          paste0(sanitize_name(send), "__", sanitize_name(recv)))
    dir.create(pair_dir, showWarnings = FALSE, recursive = TRUE)
    results_path <- file.path(pair_dir, "results_full.csv")

    if (!force_rerun && file.exists(results_path)) {
      n_skipped <- n_skipped + 1
      cat(sprintf("  [%d/%d] %s: SKIP (checkpoint)\n", n_done, n_total, pair_label))
      summary_rows[[length(summary_rows) + 1]] <- data.frame(
        sender = send, receiver = recv,
        n_pre = NA_integer_, n_post = NA_integer_, time_sec = 0,
        status = "CHECKPOINT", stringsAsFactors = FALSE)
      next
    }

    t0_pair <- proc.time()
    n_pre <- 0L; n_post <- 0L; status <- "OK"

    # --- Sender expression + L1 pruning ---
    send_genes <- gene_lists[[send]]
    s_c1 <- wq_expr[[send]][[conditions[1]]]
    s_c2 <- wq_expr[[send]][[conditions[2]]]

    dt1 <- l1_raw[l1_raw$from %in% send_genes & l1_raw$to %in% recv_genes]
    if (nrow(dt1) > 0) {
      h1 <- hill(s_c1[dt1$from] * r_c1_patched[dt1$to], K, N)
      h2 <- hill(s_c2[dt1$from] * r_c2_patched[dt1$to], K, N)
      dt1 <- dt1[(h1 >= cutoff_SigProb) | (h2 >= cutoff_SigProb)]
      rm(h1, h2)
    }

    if (nrow(dt1) == 0) {
      t_pair <- (proc.time() - t0_pair)["elapsed"]
      cat(sprintf("  [%d/%d] %s: 0 L1 edges (%.1fs)\n",
                  n_done, n_total, pair_label, t_pair))
      summary_rows[[length(summary_rows) + 1]] <- data.frame(
        sender = send, receiver = recv,
        n_pre = 0L, n_post = 0L, time_sec = round(t_pair, 1),
        status = "NO_L1", stringsAsFactors = FALSE)
      next
    }

    # --- DuckDB query ---
    duckdb_register(con, "L1", as.data.frame(dt1))
    sender_expr_df <- data.frame(gene = all_genes,
                                 c1 = unname(s_c1), c2 = unname(s_c2),
                                 stringsAsFactors = FALSE)
    duckdb_register(con, "sender_expr", sender_expr_df)
    rm(dt1, sender_expr_df)

    pathways_df <- dbGetQuery(con, sql)

    duckdb_unregister(con, "L1")
    duckdb_unregister(con, "sender_expr")

    n_pre <- nrow(pathways_df)

    if (n_pre == 0) {
      t_pair <- (proc.time() - t0_pair)["elapsed"]
      cat(sprintf("  [%d/%d] %s: 0 pathways (%.1fs)\n",
                  n_done, n_total, pair_label, t_pair))
      summary_rows[[length(summary_rows) + 1]] <- data.frame(
        sender = send, receiver = recv,
        n_pre = 0L, n_post = 0L, time_sec = round(t_pair, 1),
        status = "NO_PATHWAYS", stringsAsFactors = FALSE)
      next
    }

    # Build Path column
    pathways_df$Path <- paste(pathways_df$Ligand, pathways_df$Receptor,
                              pathways_df$EM, pathways_df$Target, sep = "*")
    recv_pathways_total <- recv_pathways_total + n_pre

    # --- Downstream (tryCatch) ---
    tryCatch({
      # Create Incytr object
      inc <- create_Incytr(object = mat, meta = meta,
                           sender = send, receiver = recv,
                           group.by = "labels", conditions = conditions)
      inc@pathways <- as.data.frame(pathways_df)
      inc@options$em_degree <- em_degree
      inc@options$edge_source_count <- edge_source_count

      # Label pathways: expression-confirmed vs kinase-imputed
      pw <- inc@pathways
      pw$pathway_evidence <- ifelse(
        pw$Ligand %in% send_genes & pw$Receptor %in% recv_genes_expr &
        pw$EM %in% recv_genes_expr & pw$Target %in% recv_genes_expr,
        "expression-confirmed", "kinase-imputed")
      # Vectorized imputed_nodes (avoids row-wise apply over potentially 100K+ rows)
      imp_l <- ifelse(!pw$Ligand %in% send_genes, "Ligand", "")
      imp_r <- ifelse(!pw$Receptor %in% recv_genes_expr, "Receptor", "")
      imp_e <- ifelse(!pw$EM %in% recv_genes_expr, "EM", "")
      imp_t <- ifelse(!pw$Target %in% recv_genes_expr, "Target", "")
      pw$imputed_nodes <- gsub("^;+|;+$", "", gsub(";{2,}", ";",
        paste(imp_l, imp_r, imp_e, imp_t, sep = ";")))
      inc@pathways <- pw
      pw_labels <- pw[, c("Path", "pathway_evidence", "imputed_nodes")]
      ensure_labels <- function(df) {
        if (!"pathway_evidence" %in% names(df))
          df <- merge(df, pw_labels, by = "Path", all.x = TRUE)
        df
      }

      # Expr_bygroup
      inc <- Expr_bygroup(inc)

      # Patch kinase-imputed gene expression (use cached rowMeans from outer loop)
      if (length(ki_for_recv) > 0 && !is.null(ki_rm_c1)) {
        for (ci in 1:2) {
          if (nrow(inc@expr.bygroup[[ci]]) == 0) next
          eg <- inc@expr.bygroup[[ci]]
          ki_in_eg <- intersect(names(ki_rm_c1), eg$Gene)
          if (length(ki_in_eg) == 0) next
          zero_ki <- ki_in_eg[eg[match(ki_in_eg, eg$Gene), recv] == 0]
          if (length(zero_ki) > 0) {
            rm_vals <- if (ci == 1) ki_rm_c1[zero_ki] else ki_rm_c2[zero_ki]
            idx <- match(zero_ki, eg$Gene)
            inc@expr.bygroup[[ci]][idx, recv] <- rm_vals
          }
        }
      }

      # Cal_SigProb
      inc <- Cal_SigProb(inc, K = K, N = N, cutoff_SigProb = cutoff_SigProb,
                         correction = 0.001)
      n_post <- nrow(inc@SigProb)

      # Cal_scFC
      inc <- Cal_scFC(inc)

      # Expression-only evaluation
      if (!skip_expronly) {
        inc_base <- Pathway_evaluation(inc, score.weight = rep(0, 6))
        results_expronly <- Export_results(inc_base)
        results_expronly <- ensure_labels(results_expronly)
        write.csv(results_expronly, file.path(pair_dir, "results_expronly.csv"),
                  row.names = FALSE)
        rm(inc_base, results_expronly)
      }

      # Phospho integration
      if (!is.null(ps1) && !is.null(ps2)) {
        sender_col <- paste0(send, "_ps")
        receiver_col <- paste0(recv, "_ps")
        ps_cols <- setdiff(colnames(ps1), "gene_symbol")
        sender_has <- sender_col %in% ps_cols && sum(!is.na(ps1[[sender_col]])) > 0
        receiver_has <- receiver_col %in% ps_cols && sum(!is.na(ps1[[receiver_col]])) > 0
        if (sender_has || receiver_has) {
          inc <- tryCatch(
            Integr_multiomics(inc,
                              ps.data_condition1 = ps1,
                              ps.data_condition2 = ps2),
            error = function(e) inc)
        }
      }

      # Full evaluation
      inc <- Pathway_evaluation(inc)

      # Kinase integration
      if (!is.null(kldata)) {
        pathway_genes <- unique(c(inc@pathways$Receptor, inc@pathways$EM,
                                  inc@pathways$Target))
        kl_filtered <- kldata[kldata$gene %in% pathway_genes |
                              kldata[["motif.geneName"]] %in% pathway_genes, ]
        inc <- Integr_kinasedata(inc, kldata = kl_filtered,
                                 cell_group = cell_types)

        if (!is.null(kl_out) && nrow(kl_out) > 0) {
          klo_filtered <- kl_out[kl_out$substrate %in% pathway_genes |
                                 kl_out$kinase %in% pathway_genes, ]
          if (nrow(klo_filtered) > 0) {
            inc <- Integr_kinase_enrichment(inc, kl_output = klo_filtered,
                                            kldata = kl_filtered,
                                            cell_group = cell_types)
          }
        }
      }

      # PDS
      inc <- Cal_PDS(inc, KPDS.weight = 0.5, AKPDS.weight = 0.25)

      # Export results
      results_full <- Export_results(inc, indicator = TRUE)
      results_full <- ensure_labels(results_full)

      # Kinase boost: difference between final PDS and expression-only TPDS
      if ("PDS" %in% names(results_full) && "TPDS" %in% names(results_full)) {
        results_full$kinase_boost <- results_full$PDS - results_full$TPDS
      }

      write.csv(results_full, results_path, row.names = FALSE)

      # Edge lists
      pw_dt <- as.data.table(pathways_df)
      write.csv(as.data.frame(pw_dt[, .(n_pathways = .N),
                by = .(from = Ligand, to = Receptor)]),
                file.path(pair_dir, "edge_list_l1.csv"), row.names = FALSE)
      write.csv(as.data.frame(pw_dt[, .(n_pathways = .N),
                by = .(from = Receptor, to = EM)]),
                file.path(pair_dir, "edge_list_l2.csv"), row.names = FALSE)
      write.csv(as.data.frame(pw_dt[, .(n_pathways = .N),
                by = .(from = EM, to = Target)]),
                file.path(pair_dir, "edge_list_l3.csv"), row.names = FALSE)

      rm(inc, results_full, pw_dt)
    }, error = function(e) {
      status <<- paste0("ERROR: ", e$message)
      n_errors <<- n_errors + 1
    })

    t_pair <- (proc.time() - t0_pair)["elapsed"]
    cat(sprintf("  [%d/%d] %s: %d -> %d pathways (%.1fs) %s\n",
                n_done, n_total, pair_label, n_pre, n_post, t_pair, status))

    summary_rows[[length(summary_rows) + 1]] <- data.frame(
      sender = send, receiver = recv,
      n_pre = n_pre, n_post = n_post,
      time_sec = round(t_pair, 1),
      status = status, stringsAsFactors = FALSE)
  }  # end sender loop

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
