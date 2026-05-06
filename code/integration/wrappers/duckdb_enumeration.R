# duckdb_enumeration.R — DuckDB-based pathway enumeration with in-query SigProb
#
# Replaces pathway_inference() for large gene sets where data.table cartesian
# joins OOM. Uses R-side edge pre-pruning + DuckDB 3-way join + inline SigProb
# filtering to return only surviving pathways (~169K vs 4.7M enumerated).
#
# Source this file from run_incytr.R, then call duckdb_enumerate_pathways().

suppressPackageStartupMessages({
  library(duckdb)
  library(DBI)
  library(data.table)
  library(Matrix)
  library(matrixStats)
})

hill <- function(x, K = 0.5, N = 2) x^N / (x^N + K^N)

# Build Hill function SQL fragment: POWER(a * b, N) / (POWER(a * b, N) + K^N)
build_hill_sql <- function(expr_a, expr_b, N, KN) {
  product <- sprintf("(%s * %s)", expr_a, expr_b)
  sprintf("POWER(%s, %d) / (POWER(%s, %d) + %f)", product, N, product, N, KN)
}

#' Weighted quantile expression matching Incytr's Expr_bygroup method:
#' 0.25*Q1 + 0.5*median + 0.25*Q3
weighted_quantile_expr <- function(mat_sub) {
  if (ncol(mat_sub) == 0) return(setNames(rep(0, nrow(mat_sub)), rownames(mat_sub)))
  m <- as.matrix(mat_sub)
  q <- matrixStats::rowQuantiles(m, probs = c(0.25, 0.5, 0.75), type = 7L,
                                 drop = FALSE)
  setNames(0.25 * q[, 1] + 0.5 * q[, 2] + 0.25 * q[, 3], rownames(mat_sub))
}

#' Enumerate pathways using DuckDB with in-query SigProb filtering.
#'
#' @param mat dgCMatrix expression matrix (genes x cells)
#' @param meta data.frame with 'labels' and 'condition' columns
#' @param DB list of 3 data.frames (L1, L2, L3) with from/to columns
#' @param sender character: sender cell type label
#' @param receiver character: receiver cell type label
#' @param conditions character(2): condition labels, e.g. c("WT", "App")
#' @param gene.use_Sender character: sender gene whitelist (NULL = all)
#' @param gene.use_Receiver character: receiver gene whitelist (NULL = all)
#' @param kinase_imputed_genes character: receiver genes added via kinase
#'   imputation (below expression threshold). These use rowMeans instead
#'   of weighted quantile to prevent zero-out. Default NULL (none).
#' @param imputed_weights optional named numeric in [0,1] keyed by gene:
#'   per-gene rescue weight (1 - best_fdr). Applied multiplicatively in the
#'   soft rescue: rescued = imputed_weight * rowMeans. Default NULL, in
#'   which case all imputed genes rescue with weight = 1 (legacy hard rescue).
#' @param K numeric: Hill function half-max (default 0.5)
#' @param N integer: Hill function exponent (default 2)
#' @param cutoff_SigProb numeric: SigProb threshold (default 0.01)
#' @param duckdb_memory character: DuckDB memory limit (default "6GB")
#' @param duckdb_threads integer: DuckDB thread count (default 4)
#' @param return_edges logical: also return compact edge lists (default FALSE)
#' @return list with $pathways, $edges, $stats
duckdb_enumerate_pathways <- function(
  mat, meta, DB, sender, receiver, conditions,
  gene.use_Sender = NULL, gene.use_Receiver = NULL,
  kinase_imputed_genes = NULL,
  imputed_weights = NULL,
  K = 0.5, N = 2, cutoff_SigProb = 0.01,
  duckdb_memory = "6GB", duckdb_threads = 4L,
  return_edges = FALSE
) {
  t0_total <- proc.time()
  all_genes <- rownames(mat)
  if (is.null(gene.use_Sender))  gene.use_Sender  <- all_genes
  if (is.null(gene.use_Receiver)) gene.use_Receiver <- all_genes

  # ------------------------------------------------------------------
  # 2. Filter DB layers by gene role and convert to data.table
  # ------------------------------------------------------------------
  t0_prep <- proc.time()

  dt1 <- as.data.table(DB[[1]][DB[[1]]$from %in% gene.use_Sender &
                                DB[[1]]$to %in% gene.use_Receiver,
                                c("from", "to")])
  dt2 <- as.data.table(DB[[2]][DB[[2]]$from %in% gene.use_Receiver &
                                DB[[2]]$to %in% gene.use_Receiver,
                                c("from", "to")])
  dt3 <- as.data.table(DB[[3]][DB[[3]]$from %in% gene.use_Receiver &
                                DB[[3]]$to %in% gene.use_Receiver,
                                c("from", "to")])

  n_l1_raw <- nrow(dt1); n_l2_raw <- nrow(dt2); n_l3_raw <- nrow(dt3)

  # ------------------------------------------------------------------
  # 3. Compute per-condition expression using weighted quantile
  #    For kinase-imputed receiver genes (below detection threshold but
  #    with protein-level kinase-substrate evidence), use rowMeans
  #    instead. The weighted quantile zeros out genes below ~50%
  #    detection, which defeats the purpose of kinase imputation.
  # ------------------------------------------------------------------
  s_cells_c1 <- which(meta$labels == sender   & meta$condition == conditions[1])
  s_cells_c2 <- which(meta$labels == sender   & meta$condition == conditions[2])
  r_cells_c1 <- which(meta$labels == receiver & meta$condition == conditions[1])
  r_cells_c2 <- which(meta$labels == receiver & meta$condition == conditions[2])

  s_c1 <- weighted_quantile_expr(mat[, s_cells_c1])
  s_c2 <- weighted_quantile_expr(mat[, s_cells_c2])
  r_c1 <- weighted_quantile_expr(mat[, r_cells_c1])
  r_c2 <- weighted_quantile_expr(mat[, r_cells_c2])

  # Patch kinase-imputed receiver genes: use rowMeans where weighted
  # quantile is zero. The weighted quantile zeros out genes below ~50%
  # detection, defeating kinase imputation for low-RNA genes.
  if (length(kinase_imputed_genes) > 0) {
    ki_zero <- kinase_imputed_genes[r_c1[kinase_imputed_genes] == 0 |
                                    r_c2[kinase_imputed_genes] == 0]
    ki_zero <- ki_zero[!is.na(ki_zero)]
    if (length(ki_zero) > 0) {
      rm_c1 <- setNames(Matrix::rowMeans(mat[ki_zero, r_cells_c1, drop = FALSE]),
                        ki_zero)
      rm_c2 <- setNames(Matrix::rowMeans(mat[ki_zero, r_cells_c2, drop = FALSE]),
                        ki_zero)
      # Soft rescue (R2): rescued = imputed_weight * rowMeans. If
      # imputed_weights is NULL, defaults to 1 (legacy hard rescue).
      w <- rep(1.0, length(ki_zero))
      if (!is.null(imputed_weights)) {
        wi <- imputed_weights[ki_zero]
        wi[is.na(wi)] <- 1.0
        w <- as.numeric(wi)
      }
      r_c1[ki_zero] <- pmax(r_c1[ki_zero], w * rm_c1)
      r_c2[ki_zero] <- pmax(r_c2[ki_zero], w * rm_c2)
      cat(sprintf("  Kinase-imputed rescue: %d genes patched (mean weight = %.3f)\n",
                  length(ki_zero), mean(w)))
    }
  }

  # ------------------------------------------------------------------
  # 4. R-side edge pre-pruning (per-edge Hill < cutoff in BOTH conditions)
  #    Lossless: if any single edge Hill < cutoff, the 3-edge product is
  #    guaranteed < cutoff since each factor is in [0,1].
  # ------------------------------------------------------------------
  prune_edges <- function(dt, from_c1, from_c2, to_c1, to_c2, weight = NULL) {
    prod_c1 <- from_c1[dt$from] * to_c1[dt$to]
    prod_c2 <- from_c2[dt$from] * to_c2[dt$to]
    if (!is.null(weight)) {
      prod_c1 <- prod_c1 * weight
      prod_c2 <- prod_c2 * weight
    }
    h1 <- hill(prod_c1, K, N)
    h2 <- hill(prod_c2, K, N)
    dt[(h1 >= cutoff_SigProb) | (h2 >= cutoff_SigProb)]
  }

  # R-side pruning uses rowMeans (not Expr_bygroup's weighted quantile), so
  # EM weight is NOT applied here to avoid false negatives. The DuckDB SQL
  # applies EM weighting as a best-effort pre-filter; Cal_SigProb does the
  # authoritative filtering with correct expression values.
  dt1 <- prune_edges(dt1, s_c1, s_c2, r_c1, r_c2)
  dt2 <- prune_edges(dt2, r_c1, r_c2, r_c1, r_c2)
  dt3 <- prune_edges(dt3, r_c1, r_c2, r_c1, r_c2)

  n_l1_pruned <- nrow(dt1); n_l2_pruned <- nrow(dt2); n_l3_pruned <- nrow(dt3)
  t_prep <- (proc.time() - t0_prep)["elapsed"]

  cat(sprintf("  Edge pruning: L1 %d->%d, L2 %d->%d, L3 %d->%d (%.1fs)\n",
              n_l1_raw, n_l1_pruned, n_l2_raw, n_l2_pruned,
              n_l3_raw, n_l3_pruned, t_prep))

  if (n_l1_pruned == 0 || n_l2_pruned == 0 || n_l3_pruned == 0) {
    cat("  No surviving edges — 0 pathways.\n")
    return(list(
      pathways = data.frame(Ligand = character(), Receptor = character(),
                            EM = character(), Target = character(),
                            Path = character(), stringsAsFactors = FALSE),
      edges = NULL,
      stats = list(n_pathways = 0L, time_sec = 0)))
  }

  # ------------------------------------------------------------------
  # 5. Build expression lookup tables for DuckDB
  # ------------------------------------------------------------------
  sender_expr_df <- data.frame(
    gene = all_genes, c1 = unname(s_c1), c2 = unname(s_c2),
    stringsAsFactors = FALSE)
  receiver_expr_df <- data.frame(
    gene = all_genes, c1 = unname(r_c1), c2 = unname(r_c2),
    stringsAsFactors = FALSE)

  # ------------------------------------------------------------------
  # 6. DuckDB: register tables, execute SigProb-filtering join
  # ------------------------------------------------------------------
  t0_duck <- proc.time()
  # Use home directory for DuckDB files — /tmp is often tmpfs (RAM-backed),
  # which defeats disk spilling and competes with the memory limit.
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

  dbExecute(con, sprintf("SET memory_limit='%s'", duckdb_memory))
  dbExecute(con, sprintf("SET threads=%d", duckdb_threads))
  dbExecute(con, "SET max_temp_directory_size='20GiB'")
  dbExecute(con, "SET preserve_insertion_order=false")

  duckdb_register(con, "L1", as.data.frame(dt1))
  duckdb_register(con, "L2", as.data.frame(dt2))
  duckdb_register(con, "L3", as.data.frame(dt3))
  duckdb_register(con, "sender_expr", sender_expr_df)
  duckdb_register(con, "receiver_expr", receiver_expr_df)

  # Free R copies — DuckDB has zero-copy references
  rm(dt1, dt2, dt3, sender_expr_df, receiver_expr_df)

  KN <- K^N  # 0.25 for default params

  # Hill SQL fragments for each edge × each condition
  h_l1_c1 <- build_hill_sql("se.c1",  "r1.c1", N, KN)
  h_l2_c1 <- build_hill_sql("r1.c1",  "r2.c1", N, KN)
  h_l1_c2 <- build_hill_sql("se.c2",  "r1.c2", N, KN)
  h_l2_c2 <- build_hill_sql("r1.c2",  "r2.c2", N, KN)
  h_l3_c1 <- build_hill_sql("r2.c1", "r3.c1", N, KN)
  h_l3_c2 <- build_hill_sql("r2.c2", "r3.c2", N, KN)

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

  pathways_df <- dbGetQuery(con, sql)
  t_duck <- (proc.time() - t0_duck)["elapsed"]

  cat(sprintf("  DuckDB join + SigProb filter: %d pathways (%.1fs)\n",
              nrow(pathways_df), t_duck))

  # ------------------------------------------------------------------
  # 7. Build Path column (matches Incytr convention: "*" separator)
  # ------------------------------------------------------------------
  if (nrow(pathways_df) > 0) {
    pathways_df$Path <- paste(pathways_df$Ligand, pathways_df$Receptor,
                              pathways_df$EM, pathways_df$Target, sep = "*")
  } else {
    pathways_df$Path <- character(0)
  }

  # ------------------------------------------------------------------
  # 8. Optional: extract compact edge lists from surviving pathways
  # ------------------------------------------------------------------
  edges <- NULL
  if (return_edges && nrow(pathways_df) > 0) {
    pw_dt <- as.data.table(pathways_df)

    edges_l1 <- pw_dt[, .(n_pathways = .N), by = .(from = Ligand, to = Receptor)]
    edges_l2 <- pw_dt[, .(n_pathways = .N), by = .(from = Receptor, to = EM)]
    edges_l3 <- pw_dt[, .(n_pathways = .N), by = .(from = EM, to = Target)]

    edges <- list(l1 = as.data.frame(edges_l1),
                  l2 = as.data.frame(edges_l2),
                  l3 = as.data.frame(edges_l3))
    rm(pw_dt)
  }

  # ------------------------------------------------------------------
  # 9. Cleanup and return (on.exit handles dbDisconnect + temp file)
  # ------------------------------------------------------------------

  t_total <- (proc.time() - t0_total)["elapsed"]

  stats <- list(
    n_pathways = nrow(pathways_df),
    n_l1_raw = n_l1_raw, n_l2_raw = n_l2_raw, n_l3_raw = n_l3_raw,
    n_l1_pruned = n_l1_pruned, n_l2_pruned = n_l2_pruned, n_l3_pruned = n_l3_pruned,
    time_prep_sec = t_prep, time_duckdb_sec = t_duck, time_total_sec = t_total)

  list(pathways = pathways_df,
       edges = edges,
       stats = stats)
}
