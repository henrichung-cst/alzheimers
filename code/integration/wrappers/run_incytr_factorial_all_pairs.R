#!/usr/bin/env Rscript
# Factorial all-pairs Incytr pipeline: per-animal expression, OLS contrast
# estimation for 9 genotype x timepoint contrasts.
#
# Adapts the receiver-centric Phase 2 architecture from run_incytr_all_pairs.R
# for factorial mode: per-animal weighted quantile expression, per-animal
# SigProb, and OLS contrast estimation producing per-contrast TPDS/SE/pvalue.
#
# Phospho, kinase scoring, and downstream reranking are deferred to a second
# PR. This pipeline produces TPDS-only results.
#
# Environment variables:
#   PAIR_FILTER       - Filter pairs, e.g. "Microglia-PVM:L5 IT", "*:L5 IT"
#   FORCE_RERUN       - Set to 1 to ignore checkpoints
#   MEMORY_LIMIT_GB   - Abort if R memory exceeds this (default 10)
#   EXPR_DETECTION_THRESHOLD - Gene detection threshold (default 0.10)
#
# Output:
#   factorial/all_pairs/recv_{receiver}.parquet   (22 files, sender as column)
#   factorial/all_pairs/pair_summary.csv
#
# Usage:
#   systemd-run --user --scope -p MemoryMax=12G \
#     micromamba run -n incytr Rscript code/integration/wrappers/run_incytr_factorial_all_pairs.R

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

sanitize_name <- function(x) gsub("/", "-", gsub(" ", "_", x))

# EM weight from Incytr internals
em_weight_log <- function(degree) 1 / log2(1 + degree)

# Record all senders as skipped and unregister receiver-scoped DuckDB tables.
skip_recv_senders <- function(senders, recv, status, con) {
  for (send in senders) {
    n_done <<- n_done + 1
    summary_rows[[length(summary_rows) + 1]] <<- data.frame(
      sender = send, receiver = recv,
      n_pre = 0L, n_post = 0L, time_sec = 0,
      status = status, stringsAsFactors = FALSE)
  }
  for (tbl in c("L2", "L3", "receiver_expr")) {
    tryCatch(duckdb_unregister(con, tbl), error = function(e) NULL)
  }
}

repo_root  <- normalizePath(file.path(script_dir, "..", "..", ".."))
int_dir    <- file.path(repo_root, "code", "integration", "intermediates")
fac_dir    <- file.path(int_dir, "factorial")

cat("Repo root:", repo_root, "\n")
cat("Factorial intermediates:", fac_dir, "\n\n")

# =========================================================================
# Config from environment
# =========================================================================
expr_threshold  <- as.numeric(Sys.getenv("EXPR_DETECTION_THRESHOLD", "0.10"))
force_rerun     <- Sys.getenv("FORCE_RERUN", "0") == "1"
memory_limit_gb <- as.numeric(Sys.getenv("MEMORY_LIMIT_GB", "10"))
pair_filter     <- Sys.getenv("PAIR_FILTER", "")
K <- 0.5; N <- 2; KN <- K^N; cutoff_SigProb <- 0.01

cat(sprintf("Config: threshold=%.0f%%, memory_limit=%dGB, force=%s\n",
            expr_threshold * 100, memory_limit_gb, force_rerun))
if (pair_filter != "") cat(sprintf("Pair filter: %s\n", pair_filter))
cat("\n")

# =========================================================================
# Section 1: Load shared data
# =========================================================================
cat("=== Section 1: Loading shared data ===\n")

cat("Loading Incytr (for DB layers)...\n")
library(Incytr)

cat("Loading factorial expression matrix...\n")
mat <- readMM(file.path(fac_dir, "expression_matrix.mtx"))
mat <- as(mat, "dgCMatrix")
genes    <- read.csv(file.path(fac_dir, "expression_genes.csv"))$gene
barcodes <- read.csv(file.path(fac_dir, "expression_barcodes.csv"))$barcode
rownames(mat) <- genes; colnames(mat) <- barcodes
all_genes <- genes
cat(sprintf("  %d genes x %d cells\n", nrow(mat), ncol(mat)))

meta <- read.csv(file.path(fac_dir, "expression_metadata.csv"),
                 row.names = 1, check.names = FALSE)
cell_types <- sort(unique(meta$labels))
cat(sprintf("  %d cell types: %s\n", length(cell_types),
            paste(cell_types, collapse = ", ")))

# Animal metadata (design matrix)
animal_meta <- read.csv(file.path(fac_dir, "animal_metadata.csv"),
                        check.names = FALSE)
animal_ids <- animal_meta$animal_id
n_animals <- length(animal_ids)
cat(sprintf("  %d animals\n", n_animals))

# Design matrix
design_cols <- c("const", "App", "Tau", "Int",
                 "time_4mo", "time_6mo",
                 "App_x_time4", "App_x_time6",
                 "Tau_x_time4", "Tau_x_time6")
design_mat <- as.matrix(animal_meta[, design_cols])
rownames(design_mat) <- animal_ids
cat(sprintf("  Design matrix: %d x %d, rank %d\n",
            nrow(design_mat), ncol(design_mat),
            qr(design_mat)$rank))

# 9 contrast coefficient vectors
contrast_list <- list(
  App_2mo  = c(0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
  App_4mo  = c(0, 1, 0, 0, 0, 0, 1, 0, 0, 0),
  App_6mo  = c(0, 1, 0, 0, 0, 0, 0, 1, 0, 0),
  Tau_2mo  = c(0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
  Tau_4mo  = c(0, 0, 1, 0, 0, 0, 0, 0, 1, 0),
  Tau_6mo  = c(0, 0, 1, 0, 0, 0, 0, 0, 0, 1),
  ApTt_2mo = c(0, 1, 1, 1, 0, 0, 0, 0, 0, 0),
  ApTt_4mo = c(0, 1, 1, 1, 0, 0, 1, 0, 1, 0),
  ApTt_6mo = c(0, 1, 1, 1, 0, 0, 0, 1, 0, 1)
)
contrast_names <- names(contrast_list)
contrast_mat <- do.call(rbind, contrast_list)
cat(sprintf("  %d contrasts: %s\n", length(contrast_names),
            paste(contrast_names, collapse = ", ")))

# Pre-compute OLS hat matrix: (X'X)^{-1} X'
# beta = hat_mat %*% y  for any response vector y
XtX_inv <- solve(crossprod(design_mat))
hat_mat <- XtX_inv %*% t(design_mat)  # p x n

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

# Compute em_degree from full L3
em_degree <- table(DB_Layer3_mouse_filtered$from)

# Convert to data.tables with just from/to for pruning
l1_raw <- as.data.table(DB_Layer1_mouse_filtered[, c("from", "to")])
l2_raw <- as.data.table(DB_Layer2_mouse_filtered[, c("from", "to")])
l3_raw <- as.data.table(DB_Layer3_mouse_filtered[, c("from", "to")])
rm(DB_Layer1_mouse_filtered, DB_Layer2_mouse_filtered, DB_Layer3_mouse_filtered)
gc(verbose = FALSE)

# =========================================================================
# Section 2: Precompute per-animal, per-cell-type expression
# =========================================================================
cat("\n=== Section 2: Precomputing per-animal expression ===\n")
t0_precomp <- proc.time()

# Per-animal expression: animal_expr[[ct]][[animal_id]] = named numeric vector
# Only for animals with >= 1 cell in that cell type (min_cells = 1 for now)
animal_expr <- list()
animal_counts <- list()  # n_cells per (ct, animal)
for (ct in cell_types) {
  animal_expr[[ct]] <- list()
  animal_counts[[ct]] <- list()
  for (aid in animal_ids) {
    cells <- which(meta$labels == ct & meta$animal_id == aid)
    animal_counts[[ct]][[aid]] <- length(cells)
    if (length(cells) >= 1) {
      animal_expr[[ct]][[aid]] <- weighted_quantile_expr(mat[, cells, drop = FALSE])
    }
  }
}

# Detection rates (pooled across all animals)
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

# ------------------------------------------------------------------
# Kinase-imputed genes (factorial): per-receiver union across contrasts.
# Refined adapter (export_kinase_imputed_genes_factorial.py) writes
#   factorial/kinase_imputed_genes__{receiver}__{contrast}.csv
# with columns gene, best_fdr, imputed_weight, receiver. We build a
# per-receiver union keyed on gene with max(imputed_weight) across
# contrasts (the strongest rescue applies).
# ------------------------------------------------------------------
expr_imputation_floor <- as.numeric(Sys.getenv("EXPR_IMPUTATION_FLOOR", "0.05"))
cat(sprintf("  EXPR_IMPUTATION_FLOOR = %.3f\n", expr_imputation_floor))

load_imputed_for_recv_factorial <- function(recv) {
  pat <- paste0("kinase_imputed_genes__", sanitize_name(recv), "__*.csv")
  files <- Sys.glob(file.path(fac_dir, pat))
  if (length(files) == 0) return(NULL)
  frames <- lapply(files, function(p) {
    d <- tryCatch(read.csv(p), error = function(e) NULL)
    if (is.null(d) || nrow(d) == 0) return(NULL)
    if (!"best_fdr" %in% names(d)) d$best_fdr <- 0
    if (!"imputed_weight" %in% names(d)) d$imputed_weight <- pmax(0, 1 - d$best_fdr)
    d[, c("gene", "best_fdr", "imputed_weight")]
  })
  frames <- Filter(Negate(is.null), frames)
  if (length(frames) == 0) return(NULL)
  all_df <- do.call(rbind, frames)
  agg <- aggregate(
    cbind(best_fdr = all_df$best_fdr,
          imputed_weight = all_df$imputed_weight) ~ gene,
    data = all_df,
    FUN = function(x) if (identical(names(x), NULL)) x else x,
    simplify = TRUE
  )
  # aggregate above returns one row per gene but stacks numeric columns as
  # lists; use split/sapply for clarity instead.
  by_gene <- split(all_df, all_df$gene)
  out <- do.call(rbind, lapply(by_gene, function(g) {
    data.frame(gene = g$gene[1],
               best_fdr = min(g$best_fdr),
               imputed_weight = max(g$imputed_weight),
               stringsAsFactors = FALSE)
  }))
  rownames(out) <- NULL
  out <- out[out$gene %in% all_genes, , drop = FALSE]
  out
}

per_recv_imputed_files <- Sys.glob(
  file.path(fac_dir, "kinase_imputed_genes__*.csv"))
cat(sprintf("  Per-receiver factorial imputed files: %d\n",
            length(per_recv_imputed_files)))

# Pooled expression (all cells of each type, regardless of animal)
# Used for DuckDB edge pruning — gives a realistic single estimate per gene,
# unlike max-across-animals which overstates products (max_A * max_B where
# maxes come from different animals).
pooled_expr <- list()
for (ct in cell_types) {
  cells <- which(meta$labels == ct)
  pooled_expr[[ct]] <- weighted_quantile_expr(mat[, cells, drop = FALSE])
}

# Summary: animals with expression per cell type
for (ct in cell_types) {
  n_with <- sum(sapply(animal_expr[[ct]], function(x) !is.null(x)))
  cat(sprintf("  %s: %d/%d animals with cells, %d genes\n",
              ct, n_with, n_animals, length(gene_lists[[ct]])))
}

t_precomp <- (proc.time() - t0_precomp)["elapsed"]
cat(sprintf("  Precomputation: %.1f sec\n", t_precomp))

# =========================================================================
# Section 3: Setup output + pair list
# =========================================================================
cat("\n=== Section 3: Setup ===\n")

out_base <- file.path(fac_dir, "all_pairs")
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
# Section 4: Factorial scoring helpers
# =========================================================================

#' Compute per-animal SigProb for a pathway, then fit OLS to get per-contrast
#' TPDS, SE, and p-value.
#'
#' @param sigprob_mat matrix (n_pathways x n_animals) of SigProb values
#' @param hat_mat OLS hat matrix (p x n_animals)
#' @param XtX_inv (X'X)^{-1} matrix (p x p)
#' @param contrast_mat matrix (n_contrasts x p) of contrast vectors
#' @param n_animals integer
#' @param n_params integer (number of design matrix columns)
#'
#' @return data.table with per-contrast TPDS, SE, pvalue columns
fit_contrast_ols <- function(sigprob_mat, hat_mat, XtX_inv, contrast_mat,
                             n_animals, n_params) {
  n_pw <- nrow(sigprob_mat)
  df_resid <- n_animals - n_params

  # beta = hat_mat %*% t(sigprob_mat)  => p x n_pw
  beta_mat <- hat_mat %*% t(sigprob_mat)

  # Residuals: e = y - X %*% beta => n_animals x n_pw
  fitted <- design_mat %*% beta_mat  # n_animals x n_pw
  resid_mat <- t(sigprob_mat) - fitted  # n_animals x n_pw

  # MSE per pathway: sum(e^2) / df_resid
  mse <- colSums(resid_mat^2) / df_resid  # length n_pw

  # For each contrast c: TPDS = c' beta, Var = c' (X'X)^{-1} c * MSE
  result <- data.table(pathway_idx = seq_len(n_pw))

  for (i in seq_along(contrast_names)) {
    cname <- contrast_names[i]
    cvec <- contrast_mat[i, ]

    # TPDS = c' beta
    tpds <- as.numeric(cvec %*% beta_mat)  # length n_pw

    # SE = sqrt(c' (X'X)^{-1} c * MSE)
    c_var_factor <- as.numeric(t(cvec) %*% XtX_inv %*% cvec)  # scalar
    se <- sqrt(c_var_factor * mse)

    # t-statistic and p-value (two-sided)
    t_stat <- tpds / se
    pval <- 2 * pt(abs(t_stat), df = df_resid, lower.tail = FALSE)

    # Clamp TPDS to [-1, 1] via logistic transform matching Incytr convention
    # Incytr's TPDS = logi(aFC) where logi(x) = 2/(1+exp(-x)) - 1
    # But in factorial mode, the raw OLS beta is the effect size on SigProb
    # scale. We use the raw beta as TPDS (bounded by SigProb range [0,1]).
    result[, (paste0("TPDS_", cname)) := tpds]
    result[, (paste0("SE_", cname)) := se]
    result[, (paste0("pvalue_", cname)) := pval]
  }

  result[, n_animals := n_animals]
  result[, df_resid := df_resid]
  result[, pathway_idx := NULL]
  result
}

# =========================================================================
# Section 5: DuckDB setup + nested enumeration + factorial scoring
# =========================================================================
cat("\n=== Section 5: Nested enumeration + factorial scoring ===\n")
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

  # Get senders for this receiver
  senders <- all_pairs$sender[all_pairs$receiver == recv]

  # Pre-check checkpoint: skip if receiver Parquet exists
  recv_parquet <- file.path(out_base, paste0("recv_", sanitize_name(recv), ".parquet"))
  if (!force_rerun && file.exists(recv_parquet)) {
    cat(sprintf("\n--- Receiver: %s — checkpointed, skipping ---\n", recv))
    for (send in senders) {
      n_done <- n_done + 1
      n_skipped <- n_skipped + 1
      summary_rows[[length(summary_rows) + 1]] <- data.frame(
        sender = send, receiver = recv,
        n_pre = NA_integer_, n_post = NA_integer_, time_sec = 0,
        status = "CHECKPOINT", stringsAsFactors = FALSE)
    }
    next
  }

  # Receiver gene list
  recv_genes <- gene_lists[[recv]]

  # Identify animals with cells in this receiver
  recv_animal_ids <- names(animal_expr[[recv]])[
    sapply(animal_expr[[recv]], function(x) !is.null(x))]
  n_recv_animals <- length(recv_animal_ids)

  if (n_recv_animals < 3) {
    cat(sprintf("\n--- Receiver: %s — only %d animals with cells, skipping ---\n",
                recv, n_recv_animals))
    skip_recv_senders(senders, recv, "TOO_FEW_ANIMALS", con)
    next
  }

  # For DuckDB pruning, use pooled expression (all cells of this type,
  # regardless of animal). This gives a realistic single estimate per gene.
  # max-across-animals overstates products because max(A)*max(B) can come
  # from different animals, producing 100x more surviving edges.
  recv_expr_pool <- pooled_expr[[recv]]

  # Kinase imputation (factorial): union across contrasts, per-gene soft
  # rescue weight. Applied to pooled expression only (per-animal vectors
  # are not patched — imputation changes *which* pathways enumerate, not
  # the per-animal SigProbs used for OLS contrast estimation).
  ki_for_recv <- character(0)
  recv_imputed_df <- load_imputed_for_recv_factorial(recv)
  if (!is.null(recv_imputed_df) && nrow(recv_imputed_df) > 0) {
    if (expr_imputation_floor > 0) {
      dr <- det_rates[[recv]][recv_imputed_df$gene]
      dr[is.na(dr)] <- 0
      recv_imputed_df <- recv_imputed_df[dr >= expr_imputation_floor, , drop = FALSE]
    }
    ki_for_recv <- setdiff(recv_imputed_df$gene, recv_genes)
    recv_genes <- union(recv_genes, ki_for_recv)
    if (length(ki_for_recv) > 0) {
      recv_cells <- which(meta$labels == recv)
      ki_zero <- ki_for_recv[recv_expr_pool[ki_for_recv] == 0]
      ki_zero <- ki_zero[!is.na(ki_zero)]
      if (length(ki_zero) > 0) {
        rm_pool <- setNames(
          Matrix::rowMeans(mat[ki_zero, recv_cells, drop = FALSE]), ki_zero)
        w <- setNames(recv_imputed_df$imputed_weight, recv_imputed_df$gene)[ki_zero]
        w[is.na(w)] <- 1.0
        recv_expr_pool[ki_zero] <- pmax(recv_expr_pool[ki_zero], w * rm_pool)
        cat(sprintf("  Kinase-imputed rescue: %d genes patched (mean w = %.3f)\n",
                    length(ki_zero), mean(w)))
      }
    }
  }

  # --- Prune L2/L3 for this receiver ---
  dt2 <- l2_raw[l2_raw$from %in% recv_genes & l2_raw$to %in% recv_genes]
  h_pool <- hill(recv_expr_pool[dt2$from] * recv_expr_pool[dt2$to], K, N)
  dt2 <- dt2[h_pool >= cutoff_SigProb]

  dt3 <- l3_raw[l3_raw$from %in% recv_genes & l3_raw$to %in% recv_genes]
  h_pool <- hill(recv_expr_pool[dt3$from] * recv_expr_pool[dt3$to], K, N)
  dt3 <- dt3[h_pool >= cutoff_SigProb]
  rm(h_pool)

  n_l2 <- nrow(dt2); n_l3 <- nrow(dt3)

  cat(sprintf("\n--- Receiver: %s  (L2=%d, L3=%d, %d senders, %d animals) ---\n",
              recv, n_l2, n_l3, length(senders), n_recv_animals))

  if (n_l2 == 0 || n_l3 == 0) {
    cat("  No surviving L2/L3 edges, skipping all senders.\n")
    skip_recv_senders(senders, recv, "NO_EDGES", con)
    next
  }

  # Register receiver tables in DuckDB (using pooled expression for pruning)
  duckdb_register(con, "L2", as.data.frame(dt2))
  duckdb_register(con, "L3", as.data.frame(dt3))
  recv_expr_df <- data.frame(gene = all_genes,
                             c1 = unname(recv_expr_pool),
                             c2 = unname(recv_expr_pool),
                             stringsAsFactors = FALSE)
  duckdb_register(con, "receiver_expr", recv_expr_df)
  rm(dt2, dt3, recv_expr_df)

  # --- Phase A: Backbone enumeration (receiver-only L2 x L3) ---
  # Using pooled expression (all cells, regardless of animal) for SigProb
  # filtering. This is slightly lossy (an animal with higher expression than
  # the pool could have a pathway survive), but prevents 100x edge inflation
  # from max-across-animals pruning.
  em_w <- "(1.0 / LOG2(CAST(1 + COALESCE(ed.degree, 1) AS DOUBLE)))"
  h_l2 <- build_hill_sql("r1.c1", "r2.c1", N, KN)
  l3_prod <- sprintf("(r2.c1 * r3.c1 * %s)", em_w)
  h_l3 <- sprintf("POWER(%s, %d) / (POWER(%s, %d) + %f)", l3_prod, N, l3_prod, N, KN)

  sql_backbone <- sprintf('
    SELECT DISTINCT
      L2."from" AS Receptor,
      L2."to"   AS EM,
      L3."to"   AS Target
    FROM L2
    JOIN L3 ON L2."to" = L3."from"
    JOIN receiver_expr r1 ON L2."from" = r1.gene
    JOIN receiver_expr r2 ON L2."to"  = r2.gene
    JOIN receiver_expr r3 ON L3."to"  = r3.gene
    LEFT JOIN em_degree_tbl ed ON L2."to" = ed.gene
    WHERE L2."from" != L2."to"
      AND L2."from" != L3."to"
      AND L2."to"   != L3."to"
      AND (%s * %s) >= %f
  ', h_l2, h_l3, cutoff_SigProb)

  t0_bb <- proc.time()
  backbone_df <- dbGetQuery(con, sql_backbone)
  t_bb <- (proc.time() - t0_bb)["elapsed"]
  cat(sprintf("  Backbones: %s R-EM-T triples (%.1fs)\n",
              format(nrow(backbone_df), big.mark = ","), t_bb))

  if (nrow(backbone_df) == 0) {
    cat("  No backbones survived, skipping.\n")
    skip_recv_senders(senders, recv, "NO_BACKBONES", con)
    next
  }

  # --- Phase B: All-sender ligand attachment ---
  t0_attach <- proc.time()
  sender_rows <- list()
  for (send in senders) {
    s_genes <- gene_lists[[send]]
    l1_sender_genes <- intersect(s_genes, l1_raw$from)
    if (length(l1_sender_genes) == 0) next

    # Use pooled expression for this sender (matches receiver-side pruning)
    s_expr_pool <- pooled_expr[[send]][l1_sender_genes]

    sender_rows[[length(sender_rows) + 1]] <- data.frame(
      gene = l1_sender_genes,
      cell_type = send,
      c1 = unname(s_expr_pool),
      c2 = unname(s_expr_pool),
      stringsAsFactors = FALSE)
  }

  if (length(sender_rows) == 0) {
    cat("  No senders have L1 ligand genes.\n")
    skip_recv_senders(senders, recv, "NO_L1", con)
    next
  }

  sender_all_df <- rbindlist(sender_rows)
  rm(sender_rows)

  backbone_receptors <- unique(backbone_df$Receptor)
  dt1_all <- l1_raw[l1_raw$from %in% sender_all_df$gene &
                     l1_raw$to %in% backbone_receptors]

  duckdb_register(con, "backbones", backbone_df)
  duckdb_register(con, "sender_all_expr", sender_all_df)
  duckdb_register(con, "L1", dt1_all)
  rm(sender_all_df, dt1_all)

  h_l1 <- build_hill_sql("se.c1", "r1.c1", N, KN)
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
      AND (%s) >= %f
  ', h_l1, cutoff_SigProb)

  all_pathways_df <- as.data.table(dbGetQuery(con, sql_attach, params = list(recv)))

  duckdb_unregister(con, "backbones")
  duckdb_unregister(con, "sender_all_expr")
  duckdb_unregister(con, "L1")
  rm(backbone_df)

  t_attach <- (proc.time() - t0_attach)["elapsed"]
  n_senders_with_pw <- length(unique(all_pathways_df$sender))
  cat(sprintf("  Ligand attachment: %s pathways across %d senders (%.1fs)\n",
              format(nrow(all_pathways_df), big.mark = ","),
              n_senders_with_pw, t_attach))

  if (nrow(all_pathways_df) == 0) {
    cat("  No pathways after attachment.\n")
    skip_recv_senders(senders, recv, "NO_PATHWAYS", con)
    duckdb_unregister(con, "L2")
    duckdb_unregister(con, "L3")
    duckdb_unregister(con, "receiver_expr")
    next
  }

  # --- Phase C: Per-animal SigProb + OLS contrast estimation ---
  t0_score <- proc.time()
  dt <- copy(all_pathways_df)
  dt[, Path := paste(Ligand, Receptor, EM, Target, sep = "*")]

  senders_present <- unique(dt$sender)
  n_pre_by_sender <- dt[, .N, by = sender]
  setkey(n_pre_by_sender, sender)

  cat(sprintf("  Phase C: Factorial scoring (%s pathways, %d senders, %d animals)...\n",
              format(nrow(dt), big.mark = ","), length(senders_present), n_animals))

  # Compute per-animal SigProb for each pathway
  # Build a sigprob matrix: n_pathways x n_animals
  n_pw <- nrow(dt)
  sigprob_mat <- matrix(NA_real_, nrow = n_pw, ncol = n_animals,
                        dimnames = list(NULL, animal_ids))

  for (a_idx in seq_along(animal_ids)) {
    aid <- animal_ids[a_idx]

    # Sender expression for this animal (varies per pathway's sender)
    # Receiver expression for this animal
    r_expr_a <- animal_expr[[recv]][[aid]]
    if (is.null(r_expr_a)) {
      sigprob_mat[, a_idx] <- NA_real_
      next
    }

    # For each pathway, compute SigProb using this animal's expression
    # Receiver-side lookups
    R_a <- r_expr_a[dt$Receptor]
    EM_a <- r_expr_a[dt$EM]
    T_a <- r_expr_a[dt$Target]

    # Sender-side: need per-pathway sender's expression for this animal
    L_a <- rep(NA_real_, n_pw)
    for (send in senders_present) {
      s_expr_a <- animal_expr[[send]][[aid]]
      if (is.null(s_expr_a)) next
      mask <- dt$sender == send
      L_a[mask] <- s_expr_a[dt$Ligand[mask]]
    }

    # EM promiscuity weight
    em_w_vec <- em_weight_log(as.numeric(em_degree[dt$EM]))
    em_w_vec[is.na(em_w_vec)] <- 1

    # Hill components
    h1 <- hill(L_a * R_a, K, N)
    h2 <- hill(R_a * EM_a, K, N)
    h3 <- hill(EM_a * T_a * em_w_vec, K, N)

    sigprob_mat[, a_idx] <- h1 * h2 * h3
  }

  # Remove pathways with too many NA animals (< 3 non-NA)
  n_valid <- rowSums(!is.na(sigprob_mat))
  keep <- n_valid >= ncol(design_mat)  # need at least p observations
  if (sum(keep) < nrow(dt)) {
    cat(sprintf("    Dropped %d pathways with insufficient animal coverage\n",
                sum(!keep)))
    dt <- dt[keep]
    sigprob_mat <- sigprob_mat[keep, , drop = FALSE]
  }

  if (nrow(dt) == 0) {
    cat("  No pathways with sufficient animal coverage.\n")
    for (send in senders) {
      n_done <- n_done + 1
      summary_rows[[length(summary_rows) + 1]] <- data.frame(
        sender = send, receiver = recv,
        n_pre = n_pre_by_sender[.(send), N],
        n_post = 0L, time_sec = 0,
        status = "NO_COVERAGE", stringsAsFactors = FALSE)
    }
    duckdb_unregister(con, "L2")
    duckdb_unregister(con, "L3")
    duckdb_unregister(con, "receiver_expr")
    rm(all_pathways_df, dt, sigprob_mat)
    next
  }

  # Replace remaining NAs with 0 (animal has no cells of this type)
  sigprob_mat[is.na(sigprob_mat)] <- 0

  # --- OLS contrast estimation ---
  cat("    Fitting OLS contrasts...\n")
  t0_ols <- proc.time()

  n_params <- ncol(design_mat)
  ols_result <- fit_contrast_ols(sigprob_mat, hat_mat, XtX_inv, contrast_mat,
                                  n_animals, n_params)

  # Bind OLS results to pathway data
  dt <- cbind(dt, ols_result)

  t_ols <- (proc.time() - t0_ols)["elapsed"]
  cat(sprintf("    OLS: %d pathways x %d contrasts (%.1fs)\n",
              nrow(dt), length(contrast_names), t_ols))

  # --- Pathway evidence labels ---
  dt[, pathway_evidence := "expression-confirmed"]

  # Per-pathway labels of receiver-side nodes that came from kinase imputation
  # (Receptor / EM / Target). Ligand is sender-side and not imputed here.
  # String format ("" or e.g. "Receptor;EM") matches single-contrast schema in
  # receiver_scoring.R label_pathway_evidence().
  dt[, imp_r := ifelse(Receptor %in% ki_for_recv, "Receptor", "")]
  dt[, imp_e := ifelse(EM       %in% ki_for_recv, "EM",       "")]
  dt[, imp_t := ifelse(Target   %in% ki_for_recv, "Target",   "")]
  dt[, imputed_nodes := gsub("^;+|;+$", "", gsub(";{2,}", ";",
    paste(imp_r, imp_e, imp_t, sep = ";")))]
  dt[, c("imp_r", "imp_e", "imp_t") := NULL]

  # --- Write Parquet (atomic) ---
  # Select output columns
  out_cols <- c("sender", "Ligand", "Receptor", "EM", "Target", "Path",
                "pathway_evidence", "imputed_nodes", "n_animals", "df_resid")
  for (cname in contrast_names) {
    out_cols <- c(out_cols,
                  paste0("TPDS_", cname),
                  paste0("SE_", cname),
                  paste0("pvalue_", cname))
  }
  dt_out <- dt[, ..out_cols]

  tmp_path <- paste0(recv_parquet, ".tmp")
  tbl <- arrow_table(dt_out)
  tbl$metadata <- c(tbl$metadata, list(
    receiver = recv,
    pipeline_version = "factorial_v1",
    n_senders = as.character(length(senders_present)),
    n_animals = as.character(n_animals),
    n_contrasts = as.character(length(contrast_names)),
    timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%S")))
  write_parquet(tbl, tmp_path)
  file.rename(tmp_path, recv_parquet)
  cat(sprintf("    Wrote %s (%s rows)\n", basename(recv_parquet),
              format(nrow(dt_out), big.mark = ",")))

  n_post_by_sender <- dt[, .N, by = sender]
  setkey(n_post_by_sender, sender)

  t_score <- (proc.time() - t0_score)["elapsed"]
  cat(sprintf("    Total scoring: %.1fs\n", t_score))

  # --- Build summary ---
  for (send in senders) {
    n_done <- n_done + 1
    n_pre <- n_pre_by_sender[.(send), N]
    if (is.na(n_pre)) n_pre <- 0L
    n_post <- n_post_by_sender[.(send), N]
    if (is.na(n_post)) n_post <- 0L
    summary_rows[[length(summary_rows) + 1]] <- data.frame(
      sender = send, receiver = recv,
      n_pre = n_pre, n_post = n_post,
      time_sec = round(t_score / length(senders), 1),
      status = "OK", stringsAsFactors = FALSE)
  }

  # Cleanup
  rm(all_pathways_df, dt, dt_out, sigprob_mat, ols_result)

  # Unregister receiver tables
  duckdb_unregister(con, "L2")
  duckdb_unregister(con, "L3")
  duckdb_unregister(con, "receiver_expr")

  # GC + memory guard
  mem_mb <- gc(verbose = FALSE)[2, 2]
  if (mem_mb > memory_limit_gb * 1024) {
    cat(sprintf("\nABORT: R memory %.0f MB exceeds %d GB limit.\n",
                mem_mb, memory_limit_gb))
    cat("Re-run to resume from checkpoint.\n")
    abort_memory <- TRUE
  }

  t_recv <- (proc.time() - t0_recv)["elapsed"]
  cat(sprintf("  Receiver %s complete (%.1fs)\n", recv, t_recv))

  if (abort_memory) break
}  # end receiver loop

# =========================================================================
# Section 6: Cleanup + summary
# =========================================================================
t_all <- (proc.time() - t0_all)["elapsed"]

if (length(summary_rows) > 0) {
  summary_df <- do.call(rbind, summary_rows)
  summary_path <- file.path(out_base, "pair_summary.csv")
  write.csv(summary_df, summary_path, row.names = FALSE)
  cat(sprintf("\n=== Summary ===\n"))
  cat(sprintf("  Pairs processed: %d/%d\n", n_done, n_total))
  cat(sprintf("  Checkpointed:    %d\n", n_skipped))
  cat(sprintf("  Errors:          %d\n", n_errors))
  cat(sprintf("  Total pathways:  %s\n",
              format(sum(summary_df$n_post, na.rm = TRUE), big.mark = ",")))
  cat(sprintf("  Total time:      %.1f min\n", t_all / 60))
  cat(sprintf("  Wrote %s\n", summary_path))
} else {
  cat("\nNo pairs processed.\n")
}

cat("\nFactorial all-pairs pipeline complete.\n")
