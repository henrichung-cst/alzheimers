#!/usr/bin/env Rscript
# Postprocessing: discordance detection, Scenario B lookup, ranking correlation.
#
# Reads results from run_incytr.R and produces:
#   intermediates/results_concordant.csv
#   intermediates/results_discordant_A.csv
#   intermediates/results_discordant_B.csv
#   intermediates/sensitivity_report.csv
#   intermediates/ranking_correlation.json

suppressPackageStartupMessages({
  library(data.table)
})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
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

cat("=== Postprocessing ===\n\n")

# ---------------------------------------------------------------------------
# 1. Load results
# ---------------------------------------------------------------------------
expronly <- fread(file.path(int_dir, "results_expronly.csv"))
full <- fread(file.path(int_dir, "results_full.csv"))
cat(sprintf("Expression-only: %d pathways\n", nrow(expronly)))
cat(sprintf("Full results: %d pathways\n", nrow(full)))

# Load kl_output for discordance detection
kl_output_path <- file.path(int_dir, "kl_output.csv")
kl_output <- if (file.exists(kl_output_path)) fread(kl_output_path) else NULL

# ---------------------------------------------------------------------------
# 2. Sensitivity analysis: rank comparison
# ---------------------------------------------------------------------------
cat("\n--- Sensitivity Analysis ---\n")

# Identify score columns
# Expression-only uses TPDS (or multimodel_score with 0 weights = TPDS)
# Full uses PDS
tpds_col <- if ("TPDS" %in% names(expronly)) "TPDS" else "multimodel_score"
pds_col <- if ("PDS" %in% names(full)) "PDS" else "multimodel_score"

# Merge on Path
merged <- merge(
  expronly[, .(Path, expr_score = get(tpds_col))],
  full[, .(Path, full_score = get(pds_col))],
  by = "Path"
)

# Rank (higher score = lower rank number)
merged[, expr_rank := rank(-expr_score, ties.method = "min")]
merged[, full_rank := rank(-full_score, ties.method = "min")]
merged[, rank_change := abs(full_rank - expr_rank)]

# Decision gate metric: fraction of top-20 changing rank by >10
top20_expr <- merged[expr_rank <= 20]
frac_changed <- mean(top20_expr$rank_change > 10, na.rm = TRUE)
cat(sprintf("Top-20 pathways: %.0f%% changed rank by >10 positions\n",
            frac_changed * 100))

# Spearman correlation
rho <- cor(merged$expr_rank, merged$full_rank, method = "spearman",
           use = "complete.obs")
cat(sprintf("Spearman rho (expr vs full rank): %.3f\n", rho))

# Save sensitivity report
sensitivity <- merged[order(rank_change, decreasing = TRUE)]
fwrite(sensitivity, file.path(int_dir, "sensitivity_report.csv"))

# Save ranking correlation as JSON
json_path <- file.path(int_dir, "ranking_correlation.json")
writeLines(
  sprintf('{"spearman_rho": %.4f, "top20_frac_changed_gt10": %.4f, "n_pathways": %d}',
          rho, frac_changed, nrow(merged)),
  json_path
)
cat(sprintf("Wrote %s\n", basename(json_path)))

# ---------------------------------------------------------------------------
# 2b. PhPDS_ps vs kinase_support_score redundancy check
# ---------------------------------------------------------------------------
ks_path <- file.path(int_dir, "kinase_support_scores.csv")
if (file.exists(ks_path)) {
  cat("\n--- PhPDS_ps vs Kinase Support Redundancy ---\n")
  ks <- fread(ks_path)
  ks_merged <- merge(
    full[, .(Path, PhPDS_ps)],
    ks[, .(Path, kinase_support_score)],
    by = "Path"
  )
  # Filter to pathways with nonzero external score
  ks_nz <- ks_merged[kinase_support_score > 0 & !is.na(PhPDS_ps)]
  if (nrow(ks_nz) > 10) {
    rho_ks <- cor(ks_nz$PhPDS_ps, ks_nz$kinase_support_score,
                  method = "spearman", use = "complete.obs")
    cat(sprintf("Spearman rho (PhPDS_ps vs kinase_support_score): %.4f (n=%d)\n",
                rho_ks, nrow(ks_nz)))

    # Append to ranking_correlation.json
    json_data <- jsonlite::fromJSON(json_path)
    json_data$spearman_rho_phPDS_vs_kscore <- round(rho_ks, 4)
    writeLines(jsonlite::toJSON(json_data, auto_unbox = TRUE, pretty = TRUE),
               json_path)
    cat(sprintf("Updated %s\n", basename(json_path)))
  }
} else {
  cat("\n  kinase_support_scores.csv not found, skipping redundancy check.\n")
}

# ---------------------------------------------------------------------------
# 3. Permutation stability (Tier 2 check)
# ---------------------------------------------------------------------------
cat("\n--- Permutation Stability ---\n")
pval_files <- list.files(int_dir, pattern = "pvalues_seed.*\\.csv",
                         full.names = TRUE)
if (length(pval_files) >= 2) {
  pvals_list <- lapply(pval_files, fread)
  # Assuming each has Path and p-value columns
  # Combine to compute CV across seeds
  pval_cols <- setdiff(names(pvals_list[[1]]), "Path")
  if (length(pval_cols) > 0) {
    # Stack p-values across seeds for each pathway
    for (i in seq_along(pvals_list)) {
      pvals_list[[i]][, seed := i]
    }
    pvals_all <- rbindlist(pvals_list)

    # For each pathway, compute CV of p-values across seeds
    # Use the first p-value column (typically condition-specific)
    pcol <- pval_cols[1]
    stability <- pvals_all[, .(
      mean_p = mean(get(pcol), na.rm = TRUE),
      sd_p = sd(get(pcol), na.rm = TRUE),
      cv_p = sd(get(pcol), na.rm = TRUE) / (mean(get(pcol), na.rm = TRUE) + 1e-10)
    ), by = Path]

    n_unstable <- sum(stability$cv_p > 0.5, na.rm = TRUE)
    cat(sprintf("Pathways with CV > 0.5: %d / %d (%.0f%%)\n",
                n_unstable, nrow(stability),
                100 * n_unstable / max(1, nrow(stability))))
  } else {
    cat("No p-value columns found in permutation files.\n")
  }
} else {
  cat(sprintf("Only %d permutation files found, skipping stability check.\n",
              length(pval_files)))
}

# ---------------------------------------------------------------------------
# 4. Discordance detection
# ---------------------------------------------------------------------------
cat("\n--- Discordance Detection ---\n")

if (!is.null(kl_output) && nrow(kl_output) > 0) {
  # Get pathway gene information from full results
  pathway_genes <- full[, .(Path, Ligand, Receptor, EM, Target)]

  # Top quartile of expression scores
  expr_threshold <- quantile(merged$expr_score, 0.75, na.rm = TRUE)
  top_quarter <- merged[expr_score >= expr_threshold]
  cat(sprintf("Top-quartile expression threshold: %.4f (%d pathways)\n",
              expr_threshold, nrow(top_quarter)))

  # For each top-quartile pathway, check if any kinase in pathway nodes
  # (EM or Target) has FDR < 0.25 and opposite NES direction
  sig_kinases <- kl_output[padj < 0.25]
  sig_kinase_genes <- unique(sig_kinases$kinase)

  discordant_A <- data.table()
  concordant <- data.table()

  for (i in seq_len(nrow(top_quarter))) {
    path_id <- top_quarter$Path[i]
    pw <- pathway_genes[Path == path_id]
    if (nrow(pw) == 0) next

    # Check if EM or Target is a kinase with MEA evidence
    em_gene <- pw$EM[1]
    tg_gene <- pw$Target[1]
    pathway_direction <- sign(top_quarter$expr_score[i])

    is_discordant <- FALSE
    for (gene in c(em_gene, tg_gene)) {
      kin_rows <- sig_kinases[kinase == gene]
      if (nrow(kin_rows) > 0) {
        nes_direction <- sign(kin_rows$score[1])
        if (pathway_direction != 0 && nes_direction != 0 &&
            pathway_direction != nes_direction) {
          is_discordant <- TRUE
          break
        }
      }
    }

    if (is_discordant) {
      discordant_A <- rbind(discordant_A, full[Path == path_id])
    } else {
      concordant <- rbind(concordant, full[Path == path_id])
    }
  }

  # Remaining pathways (not in top quartile) go to concordant
  remaining_paths <- setdiff(full$Path, top_quarter$Path)
  concordant <- rbind(concordant, full[Path %in% remaining_paths])

  cat(sprintf("Concordant: %d pathways\n", nrow(concordant)))
  cat(sprintf("Discordant A (expr up, phospho down): %d pathways\n",
              nrow(discordant_A)))
} else {
  concordant <- full
  discordant_A <- data.table()
  cat("No kl_output data, all pathways classified as concordant.\n")
}

# ---------------------------------------------------------------------------
# 5. Scenario B lookup: expression-dark, phospho-bright
# ---------------------------------------------------------------------------
cat("\n--- Scenario B Lookup ---\n")
discordant_B <- data.table()

if (!is.null(kl_output) && nrow(kl_output) > 0) {
  # Load unified attribution for high-confidence kinases
  attr_path <- file.path(repo_root, "outputs", "reports",
                         "kinase_attribution", "unified_attribution.csv")
  if (file.exists(attr_path)) {
    attr_dt <- fread(attr_path)
    receiver <- "L5 IT"

    # High-confidence kinases attributed to receiver with FDR < 0.25
    high_conf <- attr_dt[
      contrast == "App_4mo" &
      cell_type == receiver &
      combined_confidence == "high"
    ]
    high_genes <- unique(high_conf$gene_symbol)

    sig_high <- sig_kinases[kinase %in% high_genes]
    if (nrow(sig_high) > 0) {
      # Load IncytrDB to find pathways containing these kinases
      db_dir <- file.path(incytr_root, "data")
      load(file.path(db_dir, "DB_Layer2_mouse_filtered.rda"))
      load(file.path(db_dir, "DB_Layer3_mouse_filtered.rda"))

      # Check if kinase appears as EM (via Layer 2 'to') or Target (via Layer 3 'to')
      kin_in_l2 <- DB_Layer2_mouse_filtered[to %in% sig_high$kinase]
      kin_in_l3 <- DB_Layer3_mouse_filtered[to %in% sig_high$kinase]
      potential_ems <- unique(c(kin_in_l2$to, kin_in_l3$to))

      # Check which are absent from expression-based results
      expressed_ems <- unique(full$EM)
      expressed_tgs <- unique(full$Target)
      dark_kinases <- setdiff(potential_ems, c(expressed_ems, expressed_tgs))

      if (length(dark_kinases) > 0) {
        # Build Scenario B table
        for (kin in dark_kinases) {
          kin_evidence <- sig_kinases[kinase == kin]
          discordant_B <- rbind(discordant_B, data.table(
            kinase = kin,
            score = kin_evidence$score[1],
            padj = kin_evidence$padj[1],
            scenario = "B_expression_dark_phospho_bright"
          ))
        }
      }
      cat(sprintf("Scenario B candidates: %d kinases\n", nrow(discordant_B)))
    } else {
      cat("No high-confidence significant kinases for Scenario B lookup.\n")
    }
  }
} else {
  cat("No kl_output data, skipping Scenario B.\n")
}

# ---------------------------------------------------------------------------
# 6. Write three-group results
# ---------------------------------------------------------------------------
cat("\n--- Writing Results ---\n")
fwrite(concordant, file.path(int_dir, "results_concordant.csv"))
cat(sprintf("  Concordant: %d pathways\n", nrow(concordant)))

fwrite(discordant_A, file.path(int_dir, "results_discordant_A.csv"))
cat(sprintf("  Discordant A: %d pathways\n", nrow(discordant_A)))

fwrite(discordant_B, file.path(int_dir, "results_discordant_B.csv"))
cat(sprintf("  Discordant B: %d candidates\n", nrow(discordant_B)))

cat("\nPostprocessing complete.\n")
