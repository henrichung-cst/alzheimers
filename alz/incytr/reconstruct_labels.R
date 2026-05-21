#!/usr/bin/env Rscript
# Reconstruct per-node DEG / prG labels for existing pair-mode parquets that
# were written before the driver added Ligand.label / Receptor.label /
# EM.label / Target.label inline.
#
# Replays the exact DEG/PRG sets that runIncytr() computed per pair:
#   DEG.Sender   = input_gene_list$gene where cluster == Sender
#   DEG.Receiver = input_gene_list$gene where cluster == Receiver
#   PRG.Sender   = top-500 by fold_magnitude(Sender.2, Sender.1) on pr table
#   PRG.Receiver = top-500 by fold_magnitude(Receiver.2, Receiver.1) on pr table
# Both intersected with rownames(Data.input). Label = "DEG" if in DEG set,
# "prG" if in PRG set only.
#
# Usage (from any working directory):
#   pixi run Rscript alz/incytr/reconstruct_labels.R           # default: outputs/reports/incytr_pair_mode/wide/
#   pixi run Rscript alz/incytr/reconstruct_labels.R <dir>     # custom output dir

suppressPackageStartupMessages({
  library(Seurat)
  library(readr)
  library(dplyr)
  library(arrow)
})

# Resolve repo root so the script runs from any cwd.
REPO_ROOT <- system("git rev-parse --show-toplevel", intern = TRUE)
INPUTS_DIR <- file.path(REPO_ROOT, "data", "derived", "incytr_inputs")

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1L) args[1] else file.path(REPO_ROOT, "outputs", "reports", "incytr_pair_mode", "wide")

# --- Shared inputs (loaded once) ---
cat("[reconstruct] loading inputs\n")
Data.input <- readRDS(file.path(INPUTS_DIR, "incytr_obj.rds"))
rn <- rownames(Data.input)
pr <- read_csv(file.path(INPUTS_DIR, "pr_yuyu_deconvoluted.csv"), show_col_types = FALSE)
input_gene_list <- read_csv(file.path(INPUTS_DIR, "input_gene_list.csv"), show_col_types = FALSE)

groups <- as.character(unique(Data.input@active.ident))

# --- Helpers (must match runIncytr in incytr_commandline.R exactly) ---
fold_magnitude <- function(alt, ref, cap = 10) {
  score <- rep(NA_real_, length(alt))
  valid <- is.finite(alt) & is.finite(ref)
  both_positive <- valid & alt > 0 & ref > 0
  fc <- alt[both_positive] / ref[both_positive]
  score[both_positive] <- pmin(pmax(fc, 1 / fc), cap)
  one_sided <- valid & ((alt > 0 & ref == 0) | (alt == 0 & ref > 0))
  score[one_sided] <- cap
  score
}

top_omics_genes <- function(df, score_col, n = 500L) {
  score <- df[[score_col]]
  keep <- !is.na(score) & is.finite(score)
  if (!any(keep)) return(character())
  genes <- df$gene_symbol[keep]
  score <- score[keep]
  genes[order(score, decreasing = TRUE)][seq_len(min(n, length(genes)))]
}

# Build pr_1 / pr_2 for a given comparison (matches runIncytr's input prep).
build_pr_split <- function(condition1, condition2) {
  pr_1 <- select(pr, contains(condition1))
  colnames(pr_1) <- paste0(sub(paste0(condition1, ".*_"), "", colnames(pr_1)), "_pr")
  pr_1$gene_symbol <- pr$`Gene Symbol`
  pr_1 <- pr_1 %>% group_by(gene_symbol) %>% summarise_all(mean, na.rm = TRUE)

  pr_2 <- select(pr, contains(condition2))
  colnames(pr_2) <- paste0(sub(paste0(condition2, ".*_"), "", colnames(pr_2)), "_pr")
  pr_2$gene_symbol <- pr$`Gene Symbol`
  pr_2 <- pr_2 %>% group_by(gene_symbol) %>% summarise_all(mean, na.rm = TRUE)

  list(pr_1 = pr_1, pr_2 = pr_2)
}

# Compute Sender / Receiver label vectors for one (Sender.group, Receiver.group) pair.
pair_labels <- function(Sender.group, Receiver.group, pr_1, pr_2) {
  sender_deg <- intersect(
    unique(input_gene_list$gene[input_gene_list$cluster == Sender.group]),
    rn
  )
  receiver_deg <- intersect(
    unique(input_gene_list$gene[input_gene_list$cluster == Receiver.group]),
    rn
  )

  fc_total <- data.frame(
    gene_symbol = pr_1[, "gene_symbol"],
    Sender.1    = pr_1[, paste0(Sender.group, "_pr")],
    Sender.2    = pr_2[, paste0(Sender.group, "_pr")],
    Receiver.1  = pr_1[, paste0(Receiver.group, "_pr")],
    Receiver.2  = pr_2[, paste0(Receiver.group, "_pr")]
  )
  names(fc_total) <- c("gene_symbol", "Sender.1", "Sender.2", "Receiver.1", "Receiver.2")
  fc_total$sender_fc2   <- fold_magnitude(fc_total$Sender.2, fc_total$Sender.1)
  fc_total$receiver_fc2 <- fold_magnitude(fc_total$Receiver.2, fc_total$Receiver.1)

  sender_prg   <- intersect(top_omics_genes(fc_total, "sender_fc2", 500L), rn)
  receiver_prg <- intersect(top_omics_genes(fc_total, "receiver_fc2", 500L), rn)

  sender_set   <- union(sender_deg, sender_prg)
  receiver_set <- union(receiver_deg, receiver_prg)
  sender_lbl   <- setNames(rep("prG", length(sender_set)),   sender_set)
  sender_lbl[sender_deg] <- "DEG"
  receiver_lbl <- setNames(rep("prG", length(receiver_set)), receiver_set)
  receiver_lbl[receiver_deg] <- "DEG"

  list(sender = sender_lbl, receiver = receiver_lbl)
}

# Parse condition pair from filename: "<c1>_<c2>_incytr_output.parquet".
parse_conditions <- function(parquet_path) {
  base <- sub("_incytr_output\\.parquet$", "", basename(parquet_path))
  # Conditions look like ma_<age>_<geno> — split on the WTyp boundary.
  m <- regmatches(base, regexec("^(.*?)_(ma_.*_WTyp)$", base))[[1]]
  if (length(m) < 3L) {
    stop(sprintf("cannot parse conditions from '%s'", basename(parquet_path)))
  }
  list(condition1 = m[2], condition2 = m[3])
}

# --- Drive over all parquets in out_dir ---
parquet_files <- list.files(out_dir, pattern = "_incytr_output\\.parquet$",
                            full.names = TRUE)
parquet_files <- parquet_files[!grepl("\\.old\\.parquet$|\\.labeled\\.parquet$",
                                       parquet_files)]
if (length(parquet_files) == 0L) {
  stop(sprintf("no parquets under '%s'", out_dir))
}
cat(sprintf("[reconstruct] %d parquets to relabel\n", length(parquet_files)))

for (pq_path in parquet_files) {
  conds <- parse_conditions(pq_path)
  cat(sprintf("[reconstruct] %s vs %s\n", conds$condition1, conds$condition2))
  t0 <- proc.time()[["elapsed"]]

  split <- build_pr_split(conds$condition1, conds$condition2)
  pr_1 <- split$pr_1; pr_2 <- split$pr_2

  # Build label lookups for every (Sender, Receiver) pair we see in the parquet.
  df <- as.data.frame(arrow::read_parquet(pq_path))
  if (!all(c("Sender", "Receiver", "Ligand", "Receptor", "EM", "Target") %in% colnames(df))) {
    stop(sprintf("[reconstruct] missing required columns in %s", pq_path))
  }

  pair_keys <- unique(df[, c("Sender", "Receiver"), drop = FALSE])
  cache <- new.env(parent = emptyenv())
  get_pair <- function(s, r) {
    key <- paste0(s, "|", r)
    if (!is.null(cache[[key]])) return(cache[[key]])
    lbls <- pair_labels(s, r, pr_1, pr_2)
    cache[[key]] <- lbls
    lbls
  }

  df$Ligand.label   <- NA_character_
  df$Receptor.label <- NA_character_
  df$EM.label       <- NA_character_
  df$Target.label   <- NA_character_

  for (i in seq_len(nrow(pair_keys))) {
    s <- pair_keys$Sender[i]; r <- pair_keys$Receiver[i]
    if (!s %in% groups || !r %in% groups) {
      warning(sprintf("[reconstruct] skipping unknown pair %s -> %s", s, r))
      next
    }
    lbls <- get_pair(s, r)
    sel <- df$Sender == s & df$Receiver == r
    if (!any(sel)) next
    df$Ligand.label[sel]   <- unname(lbls$sender[df$Ligand[sel]])
    df$Receptor.label[sel] <- unname(lbls$receiver[df$Receptor[sel]])
    df$EM.label[sel]       <- unname(lbls$receiver[df$EM[sel]])
    df$Target.label[sel]   <- unname(lbls$receiver[df$Target[sel]])
  }

  out_path <- sub("\\.parquet$", ".labeled.parquet", pq_path)
  arrow::write_parquet(df, out_path, compression = "zstd")
  sz <- file.info(out_path)$size
  cat(sprintf("[reconstruct]   wrote %s (%.1f MB) in %.1f min\n",
              basename(out_path), sz / 1e6,
              (proc.time()[["elapsed"]] - t0) / 60))
}

cat("[reconstruct] done\n")
