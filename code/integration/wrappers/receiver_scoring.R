# receiver_scoring.R — Vectorized all-sender scoring for one receiver
#
# Phase 2 of the receiver-centric refactor. Replaces the per-pair inner loop
# (462 Incytr S4 object creations + downstream scoring) with vectorized
# data.table operations over the unified all_pathways_df.
#
# For a given receiver, all receiver-side computation (R/EM/T expression,
# Hill components, fold changes, EI, SiK matching) is shared across senders.
# Only sender-side quantities (Ligand expression, Ligand FC, H_L1) vary.

suppressPackageStartupMessages({
  library(data.table)
  library(arrow)
  library(limma)
})

# =========================================================================
# Helpers
# =========================================================================
# hill() from duckdb_enumeration.R and logi() from Incytr are already loaded.

# Shared constants for the 6 SiK cases (matches Incytr's .SIK_NAMES/.SIK_CASE_KEY)
SIK_NAMES <- c("SiK_R_of_EM", "SiK_R_of_T", "SiK_EM_of_T",
               "SiK_EM_of_R", "SiK_T_of_R", "SiK_T_of_EM")
SIK_CASE_KEY <- data.frame(
  Kinase = c("Receptor", "Receptor", "EM", "EM", "Target", "Target"),
  Substrate = c("EM", "Target", "Target", "Receptor", "Receptor", "EM"),
  stringsAsFactors = FALSE
)

# Vectorized Cal_foldchange: takes vectors (not data.frame) for speed.
# N=2 hardcoded to match Cal_foldchange (incytr/R/math.R:123).
vec_foldchange <- function(c1, c2, correction = 0.0001, q = 0.75) {
  has_zero <- any(c1 == 0 | c2 == 0, na.rm = TRUE)
  if (has_zero) {
    c1 <- c1 + correction
    c2 <- c2 + correction
  }
  log2FC <- log2(c1 / c2)
  th <- quantile(c(c1, c2), q, na.rm = TRUE)
  Vmax <- pmax(c1, c2)
  adj <- pmin(2 * Vmax^2 / (Vmax^2 + th^2), 1)
  aFC <- log2FC * adj
  list(log2FC = log2FC, aFC = aFC)
}

#' Build flat sender expression table for all unique (sender, gene) pairs.
#' Avoids repeated mapply lookups into nested wq_expr list.
build_sender_expr_table <- function(senders, genes_per_sender, wq_expr, conditions) {
  parts <- vector("list", length(senders))
  for (i in seq_along(senders)) {
    s <- senders[i]
    g <- genes_per_sender[[s]]
    if (length(g) == 0) next
    parts[[i]] <- data.table(
      sender = s, gene = g,
      c1 = unname(wq_expr[[s]][[conditions[1]]][g]),
      c2 = unname(wq_expr[[s]][[conditions[2]]][g]))
  }
  rbindlist(parts)
}

# =========================================================================
# 1. SigProb (vectorized)
# =========================================================================
#' Compute SigProb for all sender-receiver pathways in one vectorized pass.
#'
#' @param dt data.table with columns: sender, Ligand, Receptor, EM, Target
#' @param wq_expr pre-computed expression list
#' @param recv character: receiver cell type
#' @param recv_c1,recv_c2 patched receiver expression vectors
#' @param conditions character(2)
#' @param em_degree named numeric vector
#' @param edge_source_count data.table (EM, Target, n_sources)
#' @param K,N Hill params
#' @param correction numeric for SigProb FC
#' @param cutoff_SigProb numeric
#' @param edge_confidence_bonus numeric
#'
#' @return dt with SigProb columns; rows below cutoff removed
compute_sigprob_vectorized <- function(dt, sender_expr_dt, recv_c1, recv_c2,
                                       em_degree, edge_source_count,
                                       K = 0.5, N = 2, correction = 0.001,
                                       cutoff_SigProb = 0.01,
                                       edge_confidence_bonus = 1.0) {
  # --- EM promiscuity weight (receiver-only, computed once) ---
  em_w <- em_weight_log(as.numeric(em_degree[dt$EM]))
  em_w[is.na(em_w)] <- 1

  # Edge confidence bonus
  if (edge_confidence_bonus > 1.0 && nrow(edge_source_count) > 0) {
    setkey(edge_source_count, EM, Target)
    n_src <- edge_source_count[.(dt$EM, dt$Target), n_sources]
    n_src[is.na(n_src)] <- 1L
    c_edge <- ifelse(n_src > 1L, edge_confidence_bonus, 1.0)
    em_target_weight <- em_w * c_edge
  } else {
    em_target_weight <- em_w
  }

  # --- Receiver expression lookups (shared across all senders) ---
  R_c1 <- recv_c1[dt$Receptor]; R_c2 <- recv_c2[dt$Receptor]
  EM_c1 <- recv_c1[dt$EM];      EM_c2 <- recv_c2[dt$EM]
  T_c1 <- recv_c1[dt$Target];   T_c2 <- recv_c2[dt$Target]

  # --- Sender expression lookups (keyed join from pre-built table) ---
  dt[sender_expr_dt, on = .(sender, Ligand = gene), `:=`(L_c1 = i.c1, L_c2 = i.c2)]

  # --- Hill components ---
  h_l1_c1 <- hill(dt$L_c1 * R_c1, K, N)
  h_l1_c2 <- hill(dt$L_c2 * R_c2, K, N)
  h_l2_c1 <- hill(R_c1 * EM_c1, K, N)
  h_l2_c2 <- hill(R_c2 * EM_c2, K, N)
  h_l3_c1 <- hill(EM_c1 * T_c1 * em_target_weight, K, N)
  h_l3_c2 <- hill(EM_c2 * T_c2 * em_target_weight, K, N)

  # --- SigProb ---
  sp_c1 <- h_l1_c1 * h_l2_c1 * h_l3_c1
  sp_c2 <- h_l1_c2 * h_l2_c2 * h_l3_c2

  # Filter: keep rows passing cutoff in at least one condition
  keep <- (sp_c1 >= cutoff_SigProb) | (sp_c2 >= cutoff_SigProb)
  dt <- dt[keep]
  sp_c1 <- sp_c1[keep]; sp_c2 <- sp_c2[keep]

  # Store em_target_weight for expression-only scoring path
  dt[, em_target_weight := em_target_weight[keep]]

  # --- SigProb fold change (per sender, matching Cal_foldchange semantics) ---
  # th must be computed per sender because in the original pipeline it's
  # computed over the genes in each pair's pathways.
  dt[, SigProb_c1 := sp_c1]
  dt[, SigProb_c2 := sp_c2]

  # Compute aFC per sender: group by sender for quantile threshold
  dt[, sp_log2FC := log2((SigProb_c1 + correction) / (SigProb_c2 + correction))]
  dt[, sp_Vmax := pmax(SigProb_c1, SigProb_c2)]
  dt[, sp_th := quantile(sp_Vmax, 0.75, na.rm = TRUE), by = sender]
  dt[, sp_adj := pmin(2 * sp_Vmax^2 / (sp_Vmax^2 + sp_th^2), 1)]
  dt[, SigProb_aFC := sp_log2FC * sp_adj]

  # Cleanup temp columns
  dt[, c("sp_log2FC", "sp_Vmax", "sp_th", "sp_adj", "L_c1", "L_c2") := NULL]

  dt
}

# =========================================================================
# 2. scFC (vectorized)
# =========================================================================
#' Compute per-gene fold change for all pathway genes across all senders.
compute_scfc_vectorized <- function(dt, sender_expr_dt, recv_c1, recv_c2,
                                    correction = 0.00001) {
  # --- Receiver FC: computed once for all unique R/EM/T genes ---
  recv_genes <- unique(c(dt$Receptor, dt$EM, dt$Target))
  r_c1_vals <- recv_c1[recv_genes]
  r_c2_vals <- recv_c2[recv_genes]
  recv_fc <- vec_foldchange(r_c1_vals, r_c2_vals, correction = correction)
  recv_fc_dt <- data.table(gene = recv_genes, aFC = recv_fc$aFC)
  setkey(recv_fc_dt, gene)

  # Map to pathway columns (receiver-side, shared)
  dt[, Receptor_sclog2FC := recv_fc_dt[.(dt$Receptor), aFC]]
  dt[, EM_sclog2FC := recv_fc_dt[.(dt$EM), aFC]]
  dt[, Target_sclog2FC := recv_fc_dt[.(dt$Target), aFC]]

  # --- Sender FC: per sender from pre-built expression table ---
  # th (quantile threshold) must be per-sender to match Cal_foldchange semantics
  sl <- unique(dt[, .(sender, Ligand)])
  sl[sender_expr_dt, on = .(sender, Ligand = gene), `:=`(c1 = i.c1, c2 = i.c2)]

  sl[, `:=`(aFC = {
    fc <- vec_foldchange(c1, c2, correction = correction)
    fc$aFC
  }), by = sender]

  dt[sl, on = .(sender, Ligand), Ligand_sclog2FC := i.aFC]

  dt
}

# =========================================================================
# 3. Phospho integration (vectorized)
# =========================================================================
#' Compute phospho fold changes for all pathway genes across all senders.
compute_phospho_vectorized <- function(dt, ps1, ps2, recv, conditions,
                                       correction = NULL, q = NULL) {
  if (is.null(ps1) || is.null(ps2)) return(dt)

  senders <- unique(dt$sender)
  recv_col <- paste0(recv, "_ps")
  ps_cols <- setdiff(colnames(ps1), "gene_symbol")

  # --- Receiver phospho (shared, computed once) ---
  recv_has <- recv_col %in% ps_cols && sum(!is.na(ps1[[recv_col]])) > 0
  if (recv_has) {
    r_genes <- ps1$gene_symbol
    r_c1_raw <- ps1[[recv_col]]
    r_c2_raw <- ps2[[recv_col]]
    common_r <- !is.na(r_c1_raw) & !is.na(r_c2_raw)
    r_genes <- r_genes[common_r]
    r_c1_raw <- r_c1_raw[common_r]
    r_c2_raw <- r_c2_raw[common_r]

    # limma normalization
    xr <- cbind(r_c1_raw, r_c2_raw)
    yr <- limma::normalizeBetweenArrays(xr)
    recv_ps_fc <- vec_foldchange(yr[, 1], yr[, 2], correction = correction, q = q)
    recv_ps_dt <- data.table(gene = r_genes,
                             ps_log2FC = recv_ps_fc$log2FC,
                             ps_aFC = recv_ps_fc$aFC)
    setkey(recv_ps_dt, gene)

    dt[, Receptor_ps_log2FC := recv_ps_dt[.(dt$Receptor), ps_log2FC]]
    dt[, Receptor_ps_aFC := recv_ps_dt[.(dt$Receptor), ps_aFC]]
    dt[, EM_ps_log2FC := recv_ps_dt[.(dt$EM), ps_log2FC]]
    dt[, EM_ps_aFC := recv_ps_dt[.(dt$EM), ps_aFC]]
    dt[, Target_ps_log2FC := recv_ps_dt[.(dt$Target), ps_log2FC]]
    dt[, Target_ps_aFC := recv_ps_dt[.(dt$Target), ps_aFC]]
  } else {
    dt[, c("Receptor_ps_log2FC", "Receptor_ps_aFC",
           "EM_ps_log2FC", "EM_ps_aFC",
           "Target_ps_log2FC", "Target_ps_aFC") := NA_real_]
  }

  # --- Sender phospho (per sender) ---
  sl <- unique(dt[, .(sender, Ligand)])
  sl[, c("Ligand_ps_log2FC", "Ligand_ps_aFC") := NA_real_]

  for (send in senders) {
    sender_col <- paste0(send, "_ps")
    if (!sender_col %in% ps_cols) next
    if (sum(!is.na(ps1[[sender_col]])) == 0) next

    s_genes <- ps1$gene_symbol
    s_c1_raw <- ps1[[sender_col]]
    s_c2_raw <- ps2[[sender_col]]
    common_s <- !is.na(s_c1_raw) & !is.na(s_c2_raw)
    s_genes <- s_genes[common_s]
    s_c1_raw <- s_c1_raw[common_s]
    s_c2_raw <- s_c2_raw[common_s]

    xs <- cbind(s_c1_raw, s_c2_raw)
    ys <- limma::normalizeBetweenArrays(xs)
    s_fc <- vec_foldchange(ys[, 1], ys[, 2], correction = correction, q = q)
    s_fc_dt <- data.table(gene = s_genes,
                          ps_log2FC = s_fc$log2FC,
                          ps_aFC = s_fc$aFC)
    setkey(s_fc_dt, gene)

    idx <- sl$sender == send
    sl[idx, Ligand_ps_log2FC := s_fc_dt[.(sl$Ligand[idx]), ps_log2FC]]
    sl[idx, Ligand_ps_aFC := s_fc_dt[.(sl$Ligand[idx]), ps_aFC]]
  }

  dt[sl, on = .(sender, Ligand),
     `:=`(Ligand_ps_log2FC = i.Ligand_ps_log2FC,
          Ligand_ps_aFC = i.Ligand_ps_aFC)]

  dt
}

# =========================================================================
# 4. TPDS / multimodel_score (vectorized)
# =========================================================================
#' Compute TPDS and multimodel_score for all pathways.
compute_evaluation_vectorized <- function(dt, score.weight = rep(0.5, 6),
                                          k_logi = 2) {
  # s0: SigProb-based score (base TPDS)
  dt[, TPDS := logi(SigProb_aFC, k = k_logi)]

  # s2: phospho-serine score
  has_ps <- "Ligand_ps_aFC" %in% names(dt) &&
    any(!is.na(dt$Ligand_ps_aFC))
  if (has_ps) {
    ps_mat <- as.matrix(dt[, .(Ligand_ps_aFC, Receptor_ps_aFC,
                               EM_ps_aFC, Target_ps_aFC)])
    ps_mat[is.na(ps_mat)] <- 0
    dt[, PhPDS_ps := rowMeans(logi(ps_mat, k = k_logi))]
  } else {
    dt[, PhPDS_ps := 0]
  }

  # Only scRNA (TPDS) and phospho-serine (PhPDS_ps) are available in the
  # all-pairs pipeline. Other omics layers (pr, py, Ack, KGG, Rme1) would
  # plug in here if data becomes available.
  dt[, PPDS := 0]
  dt[, PhPDS_py := 0]
  dt[, Ack_score := 0]
  dt[, KGG_score := 0]
  dt[, Rme1_score := 0]

  dt[, multimodel_score := TPDS + score.weight[2] * PhPDS_ps]

  dt
}

# =========================================================================
# 5. Structural kinase (SiK) — vectorized
# =========================================================================
#' Compute EI and SiK scores for all pathways of one receiver.
compute_kinase_structural_vectorized <- function(dt, kldata, wq_expr,
                                                  cell_types, recv, conditions,
                                                  fold_threshold = 10,
                                                  reverse_sik_weight = 0.3) {
  sik_names <- SIK_NAMES
  case_key <- SIK_CASE_KEY

  # Filter kldata to pathway genes (receiver-side: R/EM/T)
  gene_use <- unique(c(dt$Receptor, dt$EM, dt$Target))
  kl <- kldata[kldata$gene %in% gene_use & kldata[["motif.geneName"]] %in% gene_use, ]

  if (nrow(kl) == 0) {
    # No SiK matches — set all columns to NA/0
    for (nm in sik_names) dt[, (nm) := NA_character_]
    for (nm in sik_names) {
      dt[, (paste0(nm, "_EI_", conditions[1])) := NA_real_]
      dt[, (paste0(nm, "_EI_", conditions[2])) := NA_real_]
    }
    dt[, (paste0("SiK_score_", conditions[1])) := 0]
    dt[, (paste0("SiK_score_", conditions[2])) := 0]
    return(list(dt = dt, ei_lookup_1 = NULL, ei_lookup_2 = NULL))
  }

  kl_pairs <- paste0(kl[["motif.geneName"]], "|", kl$gene)

  # --- SiK case matching (vectorized over all pathways) ---
  n <- nrow(dt)
  sik_genes <- matrix(NA_character_, nrow = n, ncol = 6)
  sik_match <- matrix(FALSE, nrow = n, ncol = 6)

  for (i in 1:6) {
    pairs <- paste0(dt[[case_key$Kinase[i]]], "|", dt[[case_key$Substrate[i]]])
    matched <- pairs %in% kl_pairs
    sik_match[, i] <- matched
    sik_genes[matched, i] <- dt[[case_key$Kinase[i]]][matched]
  }

  # Store SiK gene names
  for (i in 1:6) dt[, (sik_names[i]) := sik_genes[, i]]

  # --- Compute EI once per receiver (shared across all senders) ---
  # Collect all kinase genes involved in any SiK case
  k_gene <- unique(na.omit(as.vector(sik_genes)))
  if (length(k_gene) == 0) {
    for (nm in sik_names) {
      dt[, (paste0(nm, "_EI_", conditions[1])) := NA_real_]
      dt[, (paste0(nm, "_EI_", conditions[2])) := NA_real_]
    }
    dt[, (paste0("SiK_score_", conditions[1])) := 0]
    dt[, (paste0("SiK_score_", conditions[2])) := 0]
    return(list(dt = dt, ei_lookup_1 = NULL, ei_lookup_2 = NULL))
  }

  # Build expression matrix for kinase genes across all cell types (both conditions)
  expr_mat_1 <- sapply(cell_types, function(ct) wq_expr[[ct]][[conditions[1]]][k_gene])
  expr_mat_2 <- sapply(cell_types, function(ct) wq_expr[[ct]][[conditions[2]]][k_gene])
  rownames(expr_mat_1) <- k_gene
  rownames(expr_mat_2) <- k_gene

  # Cal_EI (vectorized, matches Incytr exactly)
  cal_ei_vec <- function(mat, cell_group, fold_threshold) {
    cg <- intersect(cell_group, colnames(mat))
    m <- mat[, cg, drop = FALSE]
    max_row <- do.call(pmax, as.data.frame(m))
    min_row <- do.call(pmin, as.data.frame(m))
    second <- if (length(cg) == 1) {
      max_row
    } else {
      apply(m, 1, function(x) sort(x, partial = length(x) - 1)[length(x) - 1])
    }
    distance_max <- max_row - min_row
    distance_max[distance_max == 0] <- 0.00001
    all_equal <- (max_row == min_row)
    second[second == 0] <- 0.00001

    dist_mat <- m - min_row
    dist_mat[dist_mat == 0] <- 0.00001
    porp1 <- dist_mat / distance_max
    m_nz <- m; m_nz[m_nz == 0] <- 0.00001
    porp2 <- m_nz / second

    result <- porp1 * 0.99
    result[all_equal, ] <- 0
    result[porp1 == 1 & porp2 > fold_threshold & !all_equal] <- 1
    as.data.frame(result)
  }

  ei_1 <- cal_ei_vec(expr_mat_1, cell_types, fold_threshold)
  ei_2 <- cal_ei_vec(expr_mat_2, cell_types, fold_threshold)

  # EI lookup for receiver cell type
  ei_lookup_1 <- setNames(ei_1[[recv]], k_gene)
  ei_lookup_2 <- setNames(ei_2[[recv]], k_gene)

  # --- Compute SiK score per pathway (vectorized matrix multiply) ---
  ei_mat_1 <- matrix(0, nrow = n, ncol = 6)
  ei_mat_2 <- matrix(0, nrow = n, ncol = 6)

  for (i in 1:6) {
    genes <- sik_genes[, i]
    has_gene <- !is.na(genes)
    if (any(has_gene)) {
      ei_mat_1[has_gene, i] <- unname(ei_lookup_1[genes[has_gene]])
      ei_mat_2[has_gene, i] <- unname(ei_lookup_2[genes[has_gene]])
    }

    # Store per-case EI columns
    ei_col_1 <- rep(NA_real_, n); ei_col_2 <- rep(NA_real_, n)
    if (any(has_gene)) {
      ei_col_1[has_gene] <- unname(ei_lookup_1[genes[has_gene]])
      ei_col_2[has_gene] <- unname(ei_lookup_2[genes[has_gene]])
    }
    dt[, (paste0(sik_names[i], "_EI_", conditions[1])) := ei_col_1]
    dt[, (paste0(sik_names[i], "_EI_", conditions[2])) := ei_col_2]
  }
  ei_mat_1[is.na(ei_mat_1)] <- 0
  ei_mat_2[is.na(ei_mat_2)] <- 0

  sik_weights <- c(rep(1.0, 3), rep(reverse_sik_weight, 3))
  dt[, (paste0("SiK_score_", conditions[1])) :=
       as.numeric(ei_mat_1 %*% sik_weights) / sum(sik_weights)]
  dt[, (paste0("SiK_score_", conditions[2])) :=
       as.numeric(ei_mat_2 %*% sik_weights) / sum(sik_weights)]

  list(dt = dt, ei_lookup_1 = ei_lookup_1, ei_lookup_2 = ei_lookup_2)
}

# =========================================================================
# 6. Kinase activity (AKPDS) — vectorized
# =========================================================================
#' Compute activity kinase scores using kl_output evidence.
#'
#' Builds kinase evidence table (receiver-side, shared) and computes gated
#' activity scores. Returns dt with activity columns + kl.evidence table.
compute_kinase_activity_vectorized <- function(dt, kl_out, kldata,
                                               ei_lookup_1, ei_lookup_2,
                                               conditions,
                                               padj_threshold = 0.05,
                                               k_activity = 1) {
  # Build the kl_evidence table exactly as Incytr does
  pathdf <- dt[, .(Path, Ligand, Receptor, EM, Target)]

  # Filter kl_out (activity evidence) to pathway genes
  pathway_nodes <- unique(c(pathdf$Receptor, pathdf$EM, pathdf$Target))
  all_genes_in_dt <- unique(c(pathdf$Ligand, pathway_nodes))

  kl_evidence_raw <- data.frame(
    kinase = as.character(kl_out$kinase),
    substrate = as.character(kl_out$substrate),
    site_pos = if ("site_pos" %in% names(kl_out)) as.character(kl_out$site_pos) else NA_character_,
    score = if ("score" %in% names(kl_out)) as.numeric(kl_out$score) else NA_real_,
    p_value = if ("p_value" %in% names(kl_out)) as.numeric(kl_out$p_value) else NA_real_,
    padj = if ("padj" %in% names(kl_out)) as.numeric(kl_out$padj) else NA_real_,
    stringsAsFactors = FALSE
  )

  # Filter to pathway-relevant entries
  filtered_ev <- kl_evidence_raw[
    kl_evidence_raw$substrate %in% pathway_nodes &
    kl_evidence_raw$kinase %in% all_genes_in_dt, , drop = FALSE]

  if (nrow(filtered_ev) == 0) {
    a_col_1 <- paste0("activity_score_", conditions[1])
    a_col_2 <- paste0("activity_score_", conditions[2])
    dt[, (a_col_1) := 0]
    dt[, (a_col_2) := 0]
    dt[, n_activity_kinases := 0L]
    return(list(dt = dt, kl_evidence = data.frame()))
  }

  # Build kinase evidence table using the same logic as Incytr's build_kinase_evidence
  # (structural from kldata + activity from kl_evidence)
  sik_names <- SIK_NAMES
  case_key <- SIK_CASE_KEY

  # Structural pairs (filter kldata once, reuse for support_source assignment)
  structural_rows <- NULL
  kl_filt <- NULL
  kl_struct_pairs <- NULL
  if (!is.null(kldata) && nrow(kldata) > 0) {
    kl_filt <- kldata[kldata$gene %in% pathway_nodes &
                       kldata[["motif.geneName"]] %in% all_genes_in_dt, ]
    if (nrow(kl_filt) > 0) {
      kl_struct_pairs <- paste0(kl_filt[["motif.geneName"]], "|", kl_filt$gene)
      kl_pairs <- kl_struct_pairs
      parts <- vector("list", 6)
      for (i in 1:6) {
        pairs <- paste0(pathdf[[case_key$Kinase[i]]], "|",
                         pathdf[[case_key$Substrate[i]]])
        matched <- which(pairs %in% kl_pairs)
        if (length(matched) > 0) {
          parts[[i]] <- data.frame(
            Path = pathdf$Path[matched],
            Kinase = pathdf[[case_key$Kinase[i]]][matched],
            structural_case = sik_names[i],
            stringsAsFactors = FALSE)
        }
      }
      structural_rows <- as.data.frame(rbindlist(parts))
    }
  }

  # Activity pairs
  activity_rows <- NULL
  kl_sub <- filtered_ev[filtered_ev$substrate %in% pathway_nodes, , drop = FALSE]
  if (nrow(kl_sub) > 0) {
    total_subs_by_kinase <- tapply(filtered_ev$substrate, filtered_ev$kinase,
                                    function(x) length(unique(x)))
    by_kinase <- split(kl_sub, kl_sub$kinase)
    parts <- lapply(names(by_kinase), function(kg) {
      ki <- by_kinase[[kg]]
      subs <- unique(ki$substrate)
      best_score <- if (all(is.na(ki$score))) NA_real_ else max(ki$score, na.rm = TRUE)
      best_padj <- if (all(is.na(ki$padj))) NA_real_ else min(ki$padj, na.rm = TRUE)
      n_overlap <- length(subs)
      n_total <- max(total_subs_by_kinase[kg], 1)
      path_mask <- pathdf$Receptor %in% subs | pathdf$EM %in% subs | pathdf$Target %in% subs
      matched_paths <- pathdf$Path[path_mask]
      if (length(matched_paths) == 0) return(NULL)
      data.frame(Path = matched_paths, Kinase = kg,
                 n_klib_substrate_overlaps = n_overlap,
                 substrate_specificity = n_overlap / n_total,
                 kinase_library_score = best_score,
                 kinase_library_padj = best_padj,
                 stringsAsFactors = FALSE)
    })
    activity_rows <- as.data.frame(rbindlist(parts))
  }

  if (is.null(structural_rows) && is.null(activity_rows)) {
    a_col_1 <- paste0("activity_score_", conditions[1])
    a_col_2 <- paste0("activity_score_", conditions[2])
    dt[, (a_col_1) := 0]
    dt[, (a_col_2) := 0]
    dt[, n_activity_kinases := 0L]
    return(list(dt = dt, kl_evidence = data.frame()))
  }

  # Merge structural and activity evidence
  struct_agg <- NULL
  if (!is.null(structural_rows) && nrow(structural_rows) > 0) {
    dt_struct <- as.data.table(structural_rows)
    struct_agg <- as.data.frame(dt_struct[, .(
      structural_case = paste(sort(unique(structural_case)), collapse = ";"),
      n_structural_edges = uniqueN(structural_case)
    ), by = .(Path, Kinase)])
  }

  if (!is.null(struct_agg) && !is.null(activity_rows)) {
    merged <- merge(struct_agg, activity_rows, by = c("Path", "Kinase"), all = TRUE)
  } else if (!is.null(struct_agg)) {
    merged <- struct_agg
    merged$n_klib_substrate_overlaps <- NA_integer_
    merged$substrate_specificity <- NA_real_
    merged$kinase_library_score <- NA_real_
    merged$kinase_library_padj <- NA_real_
  } else {
    merged <- activity_rows
    merged$structural_case <- NA_character_
    merged$n_structural_edges <- NA_integer_
  }

  # Support source assignment
  has_struct_col <- !is.na(merged$structural_case)
  has_act_col <- !is.na(merged$n_klib_substrate_overlaps)
  merged$support_source <- ifelse(
    has_struct_col & has_act_col, "both",
    ifelse(has_struct_col, "reference_only", NA_character_))
  act_only <- which(is.na(merged$support_source))
  if (length(act_only) > 0) {
    if (!is.null(kl_struct_pairs)) {
      ref_kinases <- unique(filtered_ev$kinase[
        paste0(filtered_ev$kinase, "|", filtered_ev$substrate) %in% kl_struct_pairs])
      in_ref <- merged$Kinase[act_only] %in% ref_kinases
      merged$support_source[act_only] <- ifelse(in_ref, "off_pathway_reference", "novel_activity")
    } else {
      merged$support_source[act_only] <- "novel_activity"
    }
  }

  # EI lookup
  ei_col_1 <- paste0("receiver_EI_", conditions[1])
  ei_col_2 <- paste0("receiver_EI_", conditions[2])
  if (!is.null(ei_lookup_1)) {
    merged[[ei_col_1]] <- unname(ei_lookup_1[merged$Kinase])
    merged[[ei_col_2]] <- unname(ei_lookup_2[merged$Kinase])
    merged$receiver_expression_present <- !is.na(merged[[ei_col_1]]) | !is.na(merged[[ei_col_2]])
  } else {
    merged[[ei_col_1]] <- NA_real_
    merged[[ei_col_2]] <- NA_real_
    merged$receiver_expression_present <- NA
  }

  kl_evidence <- merged

  # --- Compute activity scores (source-aware model, matching Cal_PDS) ---
  a_col_1 <- paste0("activity_score_", conditions[1])
  a_col_2 <- paste0("activity_score_", conditions[2])

  # Gated activity computation
  compute_gated <- function(ev) {
    has_score <- !is.na(ev$kinase_library_score)
    has_padj <- !is.na(ev$kinase_library_padj) & ev$kinase_library_padj < padj_threshold
    passing <- ev[has_score & has_padj, , drop = FALSE]
    if (nrow(passing) == 0) return(NULL)
    activity_raw <- (logi(passing$kinase_library_score, k = k_activity) + 1) / 2
    gate <- as.numeric(passing$receiver_expression_present)
    gate[is.na(gate)] <- 0
    gated_1 <- activity_raw * gate
    gated_2 <- gated_1  # binary gating: same for both conditions
    data.table(Path = passing$Path, Kinase = passing$Kinase,
               support_source = passing$support_source,
               gated_1 = gated_1, gated_2 = gated_2)
  }

  aggregate_gated <- function(dt_gated, all_paths) {
    zeros <- data.table(Path = all_paths,
                        V1 = 0, V2 = 0,
                        n_activity_kinases = 0L)
    if (is.null(dt_gated) || nrow(dt_gated) == 0) return(zeros)
    agg <- dt_gated[, .(n_activity_kinases = uniqueN(Kinase),
                         V1 = mean(gated_1), V2 = mean(gated_2)),
                     by = .(Path)]
    idx <- match(zeros$Path, agg$Path)
    has_match <- !is.na(idx)
    zeros$V1[has_match] <- agg$V1[idx[has_match]]
    zeros$V2[has_match] <- agg$V2[idx[has_match]]
    zeros$n_activity_kinases[has_match] <- agg$n_activity_kinases[idx[has_match]]
    zeros
  }

  dt_gated <- compute_gated(kl_evidence)
  all_paths <- unique(kl_evidence$Path)

  act_all <- aggregate_gated(dt_gated, all_paths)
  act_both <- aggregate_gated(
    if (!is.null(dt_gated)) dt_gated[support_source == "both"] else NULL, all_paths)
  act_novel <- aggregate_gated(
    if (!is.null(dt_gated)) dt_gated[support_source == "novel_activity"] else NULL, all_paths)

  # Store in dt using keyed joins (avoids $<- copy-on-write)
  dt[, (a_col_1) := 0]
  dt[, (a_col_2) := 0]
  dt[, n_activity_kinases := 0L]
  dt[, act_both_1 := 0]
  dt[, act_both_2 := 0]
  dt[, act_novel_1 := 0]
  dt[, act_novel_2 := 0]

  # Activity scores (all sources) for the activity_score_* export columns
  if (nrow(act_all) > 0) {
    setkey(act_all, Path)
    dt[act_all, on = "Path", `:=`(
      V_a1 = i.V1, V_a2 = i.V2, n_act = i.n_activity_kinases)]
    dt[!is.na(V_a1), (a_col_1) := V_a1]
    dt[!is.na(V_a2), (a_col_2) := V_a2]
    dt[!is.na(n_act), n_activity_kinases := n_act]
    dt[, c("V_a1", "V_a2", "n_act") := NULL]
  }
  if (nrow(act_both) > 0) {
    setkey(act_both, Path)
    dt[act_both, on = "Path", `:=`(act_both_1 = i.V1, act_both_2 = i.V2)]
  }
  if (nrow(act_novel) > 0) {
    setkey(act_novel, Path)
    dt[act_novel, on = "Path", `:=`(act_novel_1 = i.V1, act_novel_2 = i.V2)]
  }

  # Aggregate evidence for export (one row per pathway)
  ev_dt <- as.data.table(kl_evidence)
  ev_summary <- as.data.frame(ev_dt[, .(
    kinase_evidence_sources = paste(sort(unique(support_source)), collapse = ";"),
    n_kinase_edges = sum(n_structural_edges, na.rm = TRUE) +
                     sum(n_klib_substrate_overlaps, na.rm = TRUE),
    best_klib_padj = if (all(is.na(kinase_library_padj))) NA_real_
                     else min(kinase_library_padj, na.rm = TRUE),
    has_novel_activity = any(support_source == "novel_activity")
  ), by = .(Path)])

  dt[, kinase_evidence_sources := NA_character_]
  dt[, n_kinase_edges := NA_integer_]
  dt[, best_klib_padj := NA_real_]
  dt[, has_novel_activity := NA]

  if (nrow(ev_summary) > 0) {
    ev_summary_dt <- as.data.table(ev_summary)
    setkey(ev_summary_dt, Path)
    dt[ev_summary_dt, on = "Path", `:=`(
      kinase_evidence_sources = i.kinase_evidence_sources,
      n_kinase_edges = i.n_kinase_edges,
      best_klib_padj = i.best_klib_padj,
      has_novel_activity = i.has_novel_activity)]
  }

  list(dt = dt, kl_evidence = kl_evidence)
}

# =========================================================================
# 7. PDS (vectorized)
# =========================================================================
#' Compute PDS (final score) for all pathways.
compute_pds_vectorized <- function(dt, conditions,
                                   KPDS.weight = 0.5,
                                   AKPDS.weight = 0.25,
                                   has_structural = FALSE,
                                   has_activity = FALSE) {
  base_score <- dt$multimodel_score

  # Condition-directional helper
  cdir <- function(base, w, c1, c2) {
    ifelse(base > 0, w * c1,
    ifelse(base < 0, -w * c2,
           w * (c1 - c2)))
  }

  pds <- base_score

  s4_1_col <- paste0("SiK_score_", conditions[1])
  s4_2_col <- paste0("SiK_score_", conditions[2])

  if (has_structural && has_activity) {
    # Source-aware model: structural with activity boost, novel additive
    s4_1 <- dt[[s4_1_col]]; s4_2 <- dt[[s4_2_col]]
    synergy_1 <- s4_1 * (1 + dt$act_both_1)
    synergy_2 <- s4_2 * (1 + dt$act_both_2)
    pds <- pds + cdir(base_score, KPDS.weight, synergy_1, synergy_2)
    pds <- pds + cdir(base_score, AKPDS.weight, dt$act_novel_1, dt$act_novel_2)
  } else if (has_structural) {
    s4_1 <- dt[[s4_1_col]]; s4_2 <- dt[[s4_2_col]]
    pds <- pds + cdir(base_score, KPDS.weight, s4_1, s4_2)
  } else if (has_activity) {
    a_col_1 <- paste0("activity_score_", conditions[1])
    a_col_2 <- paste0("activity_score_", conditions[2])
    pds <- pds + cdir(base_score, AKPDS.weight, dt[[a_col_1]], dt[[a_col_2]])
  }

  dt[, PDS := pds]

  # Cleanup temp activity columns
  temp_cols <- c("act_all_1", "act_all_2", "act_both_1", "act_both_2",
                 "act_novel_1", "act_novel_2")
  for (col in temp_cols) {
    if (col %in% names(dt)) dt[, (col) := NULL]
  }

  dt
}

# =========================================================================
# 8. Pathway evidence labels
# =========================================================================
#' Label pathways as expression-confirmed or kinase-imputed.
label_pathway_evidence <- function(dt, gene_lists, recv_genes_expr) {
  # Per-sender: check if Ligand is in sender's gene list (vectorized via join)
  gl_dt <- rbindlist(lapply(unique(dt$sender), function(s)
    data.table(sender = s, gene = gene_lists[[s]])))
  sl <- unique(dt[, .(sender, Ligand)])
  sl[, L_expressed := FALSE]
  sl[gl_dt, on = .(sender, Ligand = gene), L_expressed := TRUE]
  dt[sl, on = .(sender, Ligand), L_expressed := i.L_expressed]

  dt[, pathway_evidence := ifelse(
    L_expressed &
    Receptor %in% recv_genes_expr &
    EM %in% recv_genes_expr &
    Target %in% recv_genes_expr,
    "expression-confirmed", "kinase-imputed")]

  # Imputed nodes
  dt[, imp_l := ifelse(!L_expressed, "Ligand", "")]
  dt[, imp_r := ifelse(!Receptor %in% recv_genes_expr, "Receptor", "")]
  dt[, imp_e := ifelse(!EM %in% recv_genes_expr, "EM", "")]
  dt[, imp_t := ifelse(!Target %in% recv_genes_expr, "Target", "")]
  dt[, imputed_nodes := gsub("^;+|;+$", "", gsub(";{2,}", ";",
    paste(imp_l, imp_r, imp_e, imp_t, sep = ";")))]

  dt[, c("L_expressed", "imp_l", "imp_r", "imp_e", "imp_t") := NULL]
  dt
}

# =========================================================================
# 9. Indicator columns
# =========================================================================
#' Generate indicator counts (sc_up, sc_down, etc.)
compute_indicators_vectorized <- function(dt) {
  # scRNA indicators
  sc_mat <- as.matrix(dt[, .(Ligand_sclog2FC, Receptor_sclog2FC,
                              EM_sclog2FC, Target_sclog2FC)])
  M_sc <- max(abs(sc_mat), na.rm = TRUE) + 1
  sc_mat[is.na(sc_mat)] <- M_sc
  dt[, sc_up := rowSums(sc_mat > 0 & sc_mat < M_sc)]
  dt[, sc_down := rowSums(sc_mat < 0)]

  # Phospho indicators
  if ("Ligand_ps_aFC" %in% names(dt) && any(!is.na(dt$Ligand_ps_aFC))) {
    ps_mat <- as.matrix(dt[, .(Ligand_ps_log2FC, Receptor_ps_log2FC,
                                EM_ps_log2FC, Target_ps_log2FC)])
    M_ps <- max(abs(ps_mat), na.rm = TRUE) + 1
    ps_mat[is.na(ps_mat)] <- M_ps
    dt[, ps_up := rowSums(ps_mat > 0 & ps_mat < M_ps)]
    dt[, ps_down := rowSums(ps_mat < 0)]
  }

  dt
}

# =========================================================================
# 10. Format export columns
# =========================================================================
#' Arrange columns to match Export_results output schema.
format_export_columns <- function(dt, recv, conditions) {
  # Rename SigProb columns to match Incytr convention
  sp_c1_name <- paste0("SigProb_", conditions[1])
  sp_c2_name <- paste0("SigProb_", conditions[2])
  setnames(dt, "SigProb_c1", sp_c1_name)
  setnames(dt, "SigProb_c2", sp_c2_name)
  setnames(dt, "SigProb_aFC", "aFC")

  # Compute log2FC column (SigProb log2FC without aFC adjustment)
  dt[, log2FC := log2((get(sp_c1_name) + 0.001) / (get(sp_c2_name) + 0.001))]

  dt[, Sender.group := sender]
  dt[, Receiver.group := recv]
  dt[, ID_1 := paste0(Path, "_", sender, "_", recv)]
  dt[, ID_2 := paste0(sender, "_", recv)]
  dt[, kinase_boost := PDS - TPDS]

  # Remove em_target_weight temp column
  if ("em_target_weight" %in% names(dt)) dt[, em_target_weight := NULL]

  dt
}

# =========================================================================
# 11. Master function: score all senders for one receiver
# =========================================================================
#' @param all_pathways_df data.table from Phase B (sender, Ligand, Receptor, EM, Target)
#' @param recv character: receiver cell type
#' @param wq_expr pre-computed expression
#' @param gene_lists per-cell-type detected genes
#' @param recv_genes_expr expression-detected receiver genes (no kinase-imputed)
#' @param ki_for_recv kinase-imputed genes for this receiver
#' @param ki_rm_c1,ki_rm_c2 cached rowMeans for kinase-imputed genes
#' @param recv_c1,recv_c2 patched receiver expression vectors
#' @param em_degree,edge_source_count network metadata
#' @param ps1,ps2 phospho data (or NULL)
#' @param kldata,kl_out kinase data (or NULL)
#' @param cell_types character vector (all 22)
#' @param conditions character(2)
#' @param K,N Hill params
#' @param cutoff_SigProb numeric
#' @param skip_expronly logical
#' @param output_dir character: path for Parquet output
#' @param export_csv logical: write backward-compatible per-pair CSVs
#' @param sanitize_name function for path sanitization
#'
#' @return data.frame of pair_summary rows
score_receiver_all_senders <- function(all_pathways_df, recv,
                                       wq_expr, gene_lists,
                                       recv_genes_expr,
                                       recv_c1, recv_c2,
                                       em_degree, edge_source_count,
                                       ps1, ps2, kldata, kl_out,
                                       cell_types, conditions,
                                       K = 0.5, N = 2,
                                       cutoff_SigProb = 0.01,
                                       output_dir, export_csv = FALSE,
                                       sanitize_name_fn) {
  t0 <- proc.time()
  dt <- copy(all_pathways_df)
  dt[, Path := paste(Ligand, Receptor, EM, Target, sep = "*")]

  senders <- unique(dt$sender)
  n_pre_by_sender <- dt[, .N, by = sender]
  setkey(n_pre_by_sender, sender)

  cat(sprintf("  Phase C: Vectorized scoring (%s pathways, %d senders)...\n",
              format(nrow(dt), big.mark = ","), length(senders)))

  # Pre-build sender expression table (used by SigProb and scFC)
  ligands_per_sender <- lapply(senders, function(s) unique(dt[sender == s, Ligand]))
  names(ligands_per_sender) <- senders
  sender_expr_dt <- build_sender_expr_table(senders, ligands_per_sender,
                                             wq_expr, conditions)
  setkey(sender_expr_dt, sender, gene)

  # --- 1. SigProb ---
  t1 <- proc.time()
  dt <- compute_sigprob_vectorized(
    dt, sender_expr_dt, recv_c1, recv_c2,
    em_degree, edge_source_count,
    K = K, N = N, correction = 0.001,
    cutoff_SigProb = cutoff_SigProb)
  cat(sprintf("    SigProb: %s pathways survived (%.1fs)\n",
              format(nrow(dt), big.mark = ","),
              (proc.time() - t1)["elapsed"]))

  if (nrow(dt) == 0) {
    summary_df <- data.frame(
      sender = senders, receiver = recv,
      n_pre = n_pre_by_sender[.(senders), N],
      n_post = 0L, time_sec = 0,
      status = "NO_SIGPROB", stringsAsFactors = FALSE)
    return(summary_df)
  }

  n_post_by_sender <- dt[, .N, by = sender]
  setkey(n_post_by_sender, sender)

  # --- 2. scFC ---
  t2 <- proc.time()
  dt <- compute_scfc_vectorized(dt, sender_expr_dt, recv_c1, recv_c2,
                                 correction = 0.00001)
  cat(sprintf("    scFC: %.1fs\n", (proc.time() - t2)["elapsed"]))

  # --- 3. Phospho ---
  t3 <- proc.time()
  dt <- compute_phospho_vectorized(dt, ps1, ps2, recv, conditions)
  cat(sprintf("    Phospho: %.1fs\n", (proc.time() - t3)["elapsed"]))

  # --- 4. Pathway evidence labels ---
  dt <- label_pathway_evidence(dt, gene_lists, recv_genes_expr)

  # --- 5. TPDS + multimodel_score ---
  t5 <- proc.time()
  dt <- compute_evaluation_vectorized(dt)
  cat(sprintf("    Evaluation: %.1fs\n", (proc.time() - t5)["elapsed"]))

  # --- 6. Kinase structural (SiK) ---
  has_structural <- FALSE
  ei_lookup_1 <- NULL; ei_lookup_2 <- NULL
  if (!is.null(kldata) && nrow(kldata) > 0) {
    t6 <- proc.time()
    sik_result <- compute_kinase_structural_vectorized(
      dt, kldata, wq_expr, cell_types, recv, conditions)
    dt <- sik_result$dt
    ei_lookup_1 <- sik_result$ei_lookup_1
    ei_lookup_2 <- sik_result$ei_lookup_2
    has_structural <- !is.null(ei_lookup_1)
    cat(sprintf("    SiK: %.1fs\n", (proc.time() - t6)["elapsed"]))
  }

  # --- 7. Kinase activity (AKPDS) ---
  has_activity <- FALSE
  if (!is.null(kl_out) && nrow(kl_out) > 0) {
    t7 <- proc.time()
    act_result <- compute_kinase_activity_vectorized(
      dt, kl_out, kldata, ei_lookup_1, ei_lookup_2, conditions)
    dt <- act_result$dt
    has_activity <- any(dt[[paste0("activity_score_", conditions[1])]] != 0, na.rm = TRUE) ||
                    any(dt[[paste0("activity_score_", conditions[2])]] != 0, na.rm = TRUE)
    cat(sprintf("    AKPDS: %.1fs\n", (proc.time() - t7)["elapsed"]))
  }

  # --- 8. PDS ---
  dt <- compute_pds_vectorized(dt, conditions,
                                KPDS.weight = 0.5, AKPDS.weight = 0.25,
                                has_structural = has_structural,
                                has_activity = has_activity)

  # --- 9. Indicators ---
  dt <- compute_indicators_vectorized(dt)

  # --- 10. Format columns ---
  dt <- format_export_columns(dt, recv, conditions)

  # --- Write Parquet (atomic) ---
  recv_parquet <- file.path(output_dir, paste0("recv_", sanitize_name_fn(recv), ".parquet"))
  tmp_path <- paste0(recv_parquet, ".tmp")
  write_parquet(dt, tmp_path,
                key_value_metadata = list(
                  receiver = recv,
                  pipeline_version = "phase2",
                  n_senders = as.character(length(senders)),
                  timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%S")))
  file.rename(tmp_path, recv_parquet)
  cat(sprintf("    Wrote %s (%s rows)\n", basename(recv_parquet),
              format(nrow(dt), big.mark = ",")))

  # --- Optional CSV export (backward-compatible) ---
  if (export_csv) {
    for (send in senders) {
      pair_dir <- file.path(output_dir,
                            paste0(sanitize_name_fn(send), "__", sanitize_name_fn(recv)))
      dir.create(pair_dir, showWarnings = FALSE, recursive = TRUE)
      pair_dt <- dt[sender == send]

      # results_full.csv (drop 'sender' column for backward compat)
      export_dt <- pair_dt[, !c("sender"), with = FALSE]
      fwrite(export_dt, file.path(pair_dir, "results_full.csv"))

      # Edge lists
      pw_dt <- pair_dt[, .(Ligand, Receptor, EM, Target)]
      fwrite(pw_dt[, .(n_pathways = .N), by = .(from = Ligand, to = Receptor)],
             file.path(pair_dir, "edge_list_l1.csv"))
      fwrite(pw_dt[, .(n_pathways = .N), by = .(from = Receptor, to = EM)],
             file.path(pair_dir, "edge_list_l2.csv"))
      fwrite(pw_dt[, .(n_pathways = .N), by = .(from = EM, to = Target)],
             file.path(pair_dir, "edge_list_l3.csv"))
    }
    cat(sprintf("    Wrote CSV for %d pairs\n", length(senders)))
  }

  t_total <- (proc.time() - t0)["elapsed"]
  cat(sprintf("    Total scoring: %.1fs\n", t_total))

  # --- Build summary ---
  summary_rows <- data.frame(
    sender = senders,
    receiver = recv,
    stringsAsFactors = FALSE)
  summary_rows$n_pre <- n_pre_by_sender[.(senders), N]
  # Senders that had all pathways filtered by SigProb won't appear in n_post
  n_post_vals <- n_post_by_sender[.(senders), N]
  n_post_vals[is.na(n_post_vals)] <- 0L
  summary_rows$n_post <- n_post_vals
  summary_rows$time_sec <- round(t_total / length(senders), 1)
  summary_rows$status <- "OK"

  summary_rows
}
